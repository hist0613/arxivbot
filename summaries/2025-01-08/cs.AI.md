New uploads on arXiv(cs.CL)

### Influences on LLM Calibration: A Study of Response Agreement, Loss Functions, and Prompt Styles (https://arxiv.org/abs/2501.03991)
Comments:
          24 pages, 11 figures, 8 tables

- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 신뢰할 수 있는 배포를 위한 중요한 요소인 모델 신뢰도(calibration)에 대한 연구를 다루고 있습니다. 기존 연구들은 다양한 프롬프트 스타일(prompt styles) 및 LLM 크기에 대한 방법의 일반화를 측정하지 못한 점을 보완합니다. 이들 연구는 12개의 LLM과 4가지 프롬프트 스타일을 포함하는 통제된 실험 설정을 정의합니다.

- **Technical Details**: 연구팀은 Calib-n이라는 새로운 프레임워크를 구축하여 여러 LLM의 응답 일치를 캡처하고, 특히 신뢰도 추정을 위한 보조 모델(auxiliary model)을 훈련합니다. 이를 통해 focal loss와 AUC surrogate loss를 이진 교차 엔트로피(binary cross-entropy)와 통합하여 보정(calibration)을 최적화합니다. 실험을 통해 응답 일치(response agreement)와 focal loss가 성능 향상에 크게 기여한다는 것을 발견했습니다.

- **Performance Highlights**: 여러 데이터셋에서 실험 결과, 보조 모델 기반 방법은 정확도 변화에 대한 강력한 보정 성능(calibration performance)을 보여주며 LLM의 내부 확률이나 언어화된 신뢰도보다 더 나은 성능을 발휘합니다. 특히, few-shot prompts가 보조 모델 기반 방법에 가장 효과적이라는 점을 발견했습니다. 이러한 통찰은 LLM 보정의 영향을 미치는 요인에 대한 이해를 심화시켜 다양한 애플리케이션에서의 신뢰할 수 있는 배포를 지원합니다.



### Semantically Cohesive Word Grouping in Indian Languages (https://arxiv.org/abs/2501.03988)
- **What's New**: 이 논문에서는 인도 언어의 처리에 있어 단어 그룹화(word grouping)를 주요 전처리 단계로 제안하고 있습니다. 인도 언어들은 격변화(Inflectional) 및 접합형(Agglutinative) 구조를 가지고 있으며, 대부분 유사한 의존 구문 파서(Dependency Parse)를 가집니다. 이를 통해 문장에서 가장 작은 의미 단위의 표현 방식 차이를 줄이고, 문장 구조의 통일성을 높이고자 합니다.

- **Technical Details**: 주요 연구는 힌디어(Hindi)에 집중되어 있으며, 격변화가 가장 덜한 언어이기 때문에 단어 그룹화에서 가장 많은 이점을 얻을 것으로 예상됩니다. 본 연구에서는 정량적 평가를 위해 단어를 셔플(shuffle)하는 내재적 방법과 기계 번역(Machine Translation) 과제에서 단어 그룹화의 중요성을 검증하기 위한 이질적(extrinsic) 평가를 수행합니다. 또한 문장 구문의 특정 측면에 대한 질적 분석도 포함됩니다.

- **Performance Highlights**: 실험 결과, 제안된 그룹화 기법은 구문 구조의 균일성을 가져오며, NLP(자연어 처리) 작업을 지원하는 것으로 나타났습니다. 이 연구는 인도 언어 처리의 효율성을 높일 수 있는 중요한 접근법을 제시하고 있으며, 다양한 언어 간 병렬 문장 구조의 통합을 가능하게 합니다.



### Localizing AI: Evaluating Open-Weight Language Models for Languages of Baltic States (https://arxiv.org/abs/2501.03952)
Comments:
          This paper is accepted to NoDaLiDa/Baltic-HLT 2025

- **What's New**: 이 연구는 유럽연합(EU) 관할 지역 외부에서 호스팅된 상업적으로 이용 가능한 LLMs의 데이터 프라이버시 문제를 다룹니다. 연구팀은 리투아니아어, 라트비아어, 에스토니아어와 같은 소규모 언어들을 지원하는 로컬 배포 가능한 오픈웨이트 LLMs를 평가하고, 이 모델들이 기계 번역, 다중 선택 질문 응답(MCQA), 자유 형식 텍스트 생성을 수행하는 방식을 분석합니다.

- **Technical Details**: Meta의 Llama 3 및 Llama 3.1, Google의 Gemma 2, Microsoft의 Phi 3, Mistral의 NeMo와 같은 여러 모델을 다양한 크기와 정밀도로 평가합니다. 실험에서는 4bit, 8bit, 그리고 16bit의 정밀도에서 모델의 성능을 비교하며, 특정 언어에서의 번역 품질은 FLORES-200 기준 데이터셋을 통해 측정합니다. 평가 방식은 자동 평가를 위한 COMET 지표 및 수동 평가를 포함합니다.

- **Performance Highlights**: 연구 결과, Gemma 2와 같은 일부 모델이 상업적으로 이용 가능한 최고 모델과 유사한 성능을 보인 반면, 많은 LLMs는 이러한 언어에서 어려움을 겪고 있음을 확인했습니다. 특히, 이들 모델은 최첨단 번역 성능에 근접하는 모습을 보였지만, 20단어 중 1단어에서 발생하는 어휘 환각(lexical hallucinations) 오류가 여전히 문제로 지적되었습니다.



### Not all tokens are created equal: Perplexity Attention Weighted Networks for AI generated text detection (https://arxiv.org/abs/2501.03940)
- **What's New**: 이번 연구에서는 Perplexity Attention Weighted Network (PAWN)이라는 새로운 AI 생성 텍스트 탐지 프레임워크를 제안합니다. PAWN은 LLM의 마지막 숨겨진 상태와 위치 정보를 사용하여, 텍스트 생성 과정에서 다음 토큰 분포 메트릭의 기여도를 동적으로 가중화합니다. 이러한 접근 방식은 고전적인 zero-shot 방법론을 개선하여, 작은 훈련 파라미터 수로도 효과적인 탐지가 가능하도록 합니다.

- **Technical Details**: PAWN은 다음 토큰 분포 메트릭을 보다 정교하게 집계할 수 있도록 설계되었으며, 언어 모델의 의미론적 정보와 위치 정보를 활용하여 각 토큰이 미치는 영향을 조정합니다. 이 접근 방식은 상대적으로 적은 자원으로도 훈련이 가능하며, 숨겨진 상태와 메트릭을 디스크에 캐시하여 요구되는 리소스를 크게 줄입니다. PAWN은 기존의 세밀하게 조정된 LLM에 비해 경쟁력 있는 성능을 보여줍니다.

- **Performance Highlights**: 실험 결과, PAWN은 고급 탐지 기법들과 비교하여 일반화 성능이 뛰어난 것으로 나타났습니다. 이는 새로운 도메인과 소스 모델에 대해서도 안정적인 결정 경계를 유지하며, 적대적 공격에 대한 저항력이 향상되었습니다. 또한, 다국어 기능이 있는 LLM을 사용할 경우, 감독 훈련이 진행되지 않은 언어에서도 좋은 성능을 발휘하는 것으로 연구되었습니다.



### AlphaPO -- Reward shape matters for LLM alignmen (https://arxiv.org/abs/2501.03884)
Comments:
          Preprint. Work in progress

- **What's New**: 이 논문에서는 Direct Alignment Algorithms (DAAs)라는 새로운 접근 방식, 특히 AlphaPO라는 방법을 소개합니다. 기존의 보상 모델링 단계를 생략하여 정책(policy)과 직접적으로 연결된 보상 함수(reward function)를 특징화합니다. AlphaPO는 보상 함수의 형태를 조정할 수 있는 $
abla$-파라미터를 활용하여, 기존의 리워드 형태 이상으로 발전합니다.

- **Technical Details**: DAAs는 일반적으로 성과가 떨어지는 likelihood displacement 문제를 겪습니다. AlphaPO는 보상 함수의 형태를 조절하여 likelihood displacement를 줄이고, 과도한 최적화를 방지하는 데 중점을 둡니다. 이 방법을 통해 Mistral-7B 및 Llama3-8B의 높은 일치성(alignment performance)을 달성할 수 있습니다.

- **Performance Highlights**: AlphaPO는 SimPO와 비교하여 약 7%에서 10%의 일치성 성능 개선을 보여줍니다. 논문에서 제시된 분석과 결과는 보상의 형태가 중요하며, 이를 체계적으로 변화시켜 훈련 역학(training dynamics) 및 일치성 향상에 어떻게 영향을 미치는지를 강조합니다.



### Add Noise, Tasks, or Layers? MaiNLP at the VarDial 2025 Shared Task on Norwegian Dialectal Slot and Intent Detection (https://arxiv.org/abs/2501.03870)
Comments:
          VarDial @ COLING 2025

- **What's New**: 본 연구는 노르웨이 방언 및 표준 언어에서의 슬롯 및 의도 탐지(Slot and Intent Detection; SID) 문제를 다룹니다. 최근까지는 SID 연구가 주요 언어에 집중되어 있었으나, 저자들은 다양한 방언에서의 성능을 개선하기 위해 여러 접근 방식을 제안하였습니다. 이 연구는 VarDial 2025 공유 작업에 참여하여 새로운 유형의 데이터 세트와 모델 조합을 실험하였습니다.

- **Technical Details**: 본 논문에서는 셋업(setup) 다양성을 비교 분석하며, 훈련 데이터의 언어(영어, 노르웨이어, 방언 노르웨이어) 및 Noise(노이즈) 주입, 보조 테스크(auxiliary tasks) 훈련, 그리고 Layer Swapping 기법을 적용합니다. 이를 통해 다양한 모델의 층을 조합하여 성능을 향상시키고, 훈련의 결과로 97.6%의 의도 정확도와 85.6%의 슬롯 F1 점수를 기록하였습니다.

- **Performance Highlights**: 이 연구에서는 Noise(노이즈) 주입이 모델의 성능을 높이는 데 도움을 준다는 것을 발견하였으며, 보조 테스크의 효과는 혼재되어 있음을 확인했습니다. 특히, 영어 및 소량의 방언 데이터로 훈련된 모델 조합은 가장 튼튼한 슬롯 예측 성능을 발휘했습니다. 연구의 최종 모델들은 VarDial 2025 공동 작업에서 뛰어난 성과를 보여 주목받고 있습니다.



### Improving Dialectal Slot and Intent Detection with Auxiliary Tasks: A Multi-Dialectal Bavarian Case Study (https://arxiv.org/abs/2501.03863)
Comments:
          VarDial @ COLING 2025

- **What's New**: 이번 연구에서는 다이얼렉트(dialectal) 데이터의 슬롯 및 의도 인식(slot and intent detection) 문제를 해결하기 위해 새로운 접근 방식을 제안합니다. 특히, 뮌헨 방언에 대한 새로운 데이터셋을 출시하여, 다양한 바바리아 방언(Bavarian dialects)에 대한 제로샷 전이 학습(zero-shot transfer learning)을 수행합니다. 이를 통해 한정된 리소스로도 품질 높은 자연어 이해를 구현하고자 합니다.

- **Technical Details**: 모델을 훈련시키기 위해 보조 작업(auxiliary tasks)으로 바바리아 언어에서의 다중 작업 학습(multi-task learning)과 중간 작업 훈련(intermediate-task training)을 비교합니다. 세 가지 보조 작업 유형으로는 토큰 수준의 구문 과제(token-level syntactic tasks), 명명된 개체 인식(named entity recognition, NER), 그리고 언어 모델링(language modelling)이 포함됩니다. 이러한 접근 방법은 슬롯 필링(slot filling) 성능을 향상시키는데 특히 효과적임을 발견했습니다.

- **Performance Highlights**: 가장 높은 성능을 보인 접근 방법은 바바리아 방언에 대한 의도 분류(intent classification) 성능을 5.1, 슬롯 필링 F1 점수는 8.4 포인트 상승시켰습니다. NER 작업이 가장 긍정적인 영향을 미친다는 결과도 도출되었으며, 중간 작업 훈련이 성능 일관성을 높이는 데 유리함을 확인했습니다. 이 연구는 리소스가 부족한 환경에서의 자연어 처리 모델의 성능을 개선하는 데 중요한 기초 자료를 제공합니다.



### Progressive Document-level Text Simplification via Large Language Models (https://arxiv.org/abs/2501.03857)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)을 활용하여 문서 수준에서의 간소화(DS)를 효과적으로 수행하는 새로운 방법론인 ProgDS를 제안합니다. 기존의 모델들이 DS를 단순히 요약으로 간주하던 반면, ProgDS는 텍스트의 의미와 일관성을 유지하면서도 단계적으로 간소화를 진행할 수 있도록 설계되었습니다. 이는 전반적으로 문서의 복잡성을 체계적으로 분해하여 간소화하는 다계층 협업 접근 방식을 모방합니다.

- **Technical Details**: ProgDS 방법론은 다섯 가지 주요 단계를 포함하여 문서의 목표를 이해하고 주요 섹션을 식별한 후, 각 주제의 내용을 개인별로 간소화합니다. 여기서는 담화 수준(discourse-level), 주제 수준(topic-level), 어휘 수준(lexical-level)의 세 가지 계층적 단계를 통해 간소화를 수행합니다. 이러한 방식으로 각 단계에서 수행되는 작업을 순차적으로 간소화하여 최종 결과를 도출합니다.

- **Performance Highlights**: 실험 결과에 따르면, ProgDS는 기존의 작거나 직접 요청한 LLMs보다 상당히 우수한 성능을 보이며, 다양한 평가 지표에서도 최신 상태를 유지합니다. 특히, 원본 문서가 긴 경우 더욱 두드러진 성능 향상을 보여주었으며, 이러한 성과는 LLMs가 단일 요청으로 긴 문서를 간소화하는 데 있어 한계를 극복하도록 돕는 데 기여했습니다.



### BabyLMs for isiXhosa: Data-Efficient Language Modelling in a Low-Resource Contex (https://arxiv.org/abs/2501.03855)
- **What's New**: 이번 BabyLM 챌린지는 샘플 효율적인 언어 모델을 개발할 것을 참가자들에게 요구했습니다. 제출된 모델은 아동 발달 단계에서 아이들이 접하는 단어 수에 제한된 고정 영어 말뭉치에서 사전 학습되었습니다. 이 챌린지는 저자원 언어에 대한 데이터 효율적인 언어 모델링을 개선할 수 있는 가능성을 강조하며, isiXhosa 언어를 사례로 연구를 수행했습니다.

- **Technical Details**: isiXhosa에 대한 사전 훈련은 ELC-BERT와 MLSM 두 가지 BabyLM 아키텍처를 기반으로 진행되었습니다. ELC-BERT는 계층 간의 출력을 선택적으로 가중치화하여 데이터 효율적인 학습을 가능하게 합니다. 반면에 MLSM은 마스크링된 단어의 정확한 정체성을 예측하는 대신, 더 넓은 의미적 범주를 예측하도록 훈련하는 혁신적인 대안입니다.

- **Performance Highlights**: 두 BabyLM 모델은 POS 태깅 및 NER에서 RoBERTa 모델보다 높은 성능을 보이며, 특히 ELC-BERT가 NER에서 +3.2 F1의 성능 향상을 기록했습니다. 그러나 토픽 분류(NTC)에서는 RoBERTa가 더 나은 성능을 나타내어, 주제 분류가 상대적으로 더 쉬운 작업이라는 점이 이 결과에 기여한 것으로 판단됩니다.



### TACLR: A Scalable and Efficient Retrieval-based Method for Industrial Product Attribute Value Identification (https://arxiv.org/abs/2501.03835)
- **What's New**: 이 논문에서는 e-commerce 플랫폼에서 제품 속성 값 식별(Product Attribute Value Identification, PAVI)의 새로운 접근 방식을 제안합니다. TACLR(Taxonomy-Aware Contrastive Learning Retrieval)이라는 최초의 검색 기반 방법을 도입하여, 제품 프로필과 후보 값을 임베딩으로 변환하고, 유사성을 기반으로 값을 검색합니다. 이 방법은 대량의 카테고리와 속성을 효과적으로 처리할 수 있습니다.

- **Technical Details**: TACLR은 PAVI를 정보 검색(task)으로 정의하며, 주어진 제품 항목에 대한 쿼리와 속성 분류(corpus)로서 작용합니다. 이 방법은 속성 및 카테고리로부터 후보 값을 선택하는 하드 네거티브 샘플링(hard negative sampling) 기법을 사용하여 차별화된 임베딩을 제공합니다. 또한, 동적 임계값(dynamic thresholds)을 도입하여 추론의 유연성을 높였습니다.

- **Performance Highlights**: 실험 결과 TACLR은 수많은 제품 목록과 속성을 효율적으로 처리할 수 있는 능력을 입증했습니다. 이 방법은 실제 상업적 환경에서 성공적으로 배포되어 매일 수백만 개의 제품 리스트를 처리하며, 동적으로 대규모 속성 분류를 지원합니다. TACLR은 또한 속성 값이 누락된 경우를 정확히 탐지하는 기능을 갖추고 있습니다.



### Investigating the Impact of Data Selection Strategies on Language Model Performanc (https://arxiv.org/abs/2501.03826)
Comments:
          7 pages, 1 figure

- **What's New**: 이 연구는 언어 모델의 성능 향상을 위한 데이터 선택의 중요성을 탐구합니다. 다양한 데이터 선택 방법과 특징 유형이 모델 성능에 미치는 영향을 분석하며, n-gram 기능과 임베딩 기반 신경 기능을 결합한 혼합 중요도 재샘플링(Hybrid Importance Resampling, HIR) 방식을 제안합니다. 이를 통해 목표 분포와 일치하는 데이터셋 선택의 효율성을 입증하고자 합니다.

- **Technical Details**: 연구에서는 랜덤 선택과 DSIR(Distributionally Aligned Methods)을 비교하는 실험을 통해 데이터 선택 전략이 언어 모델의 학습 효율성에 미치는 영향을 조명합니다. HIR 방법은 통계적 n-gram 기능과 신경 기능을 결합하여 하이브리드 분포를 구성하며, 이로써 표본 중요도 가중치를 계산합니다. 최종적으로 선택된 데이터는 Pile 데이터셋에서 추출되어 GLUE 벤치마크에서 모델 성능을 평가하는 데 사용됩니다.

- **Performance Highlights**: 실험 결과, 데이터 선택이 언어 모델 성능 개선에 crucial한 역할을 한다는 점이 강조되었습니다. 목표 분포와 유사하게 사전 훈련 데이터를 신중히 선별할 경우 여러 다운스트림 작업에서 성능 개선이 나타났습니다. HIR 방법은 랜덤 선택에 비해 일관된 성능 향상을 보여주며, 이는 모델이 목표 데이터셋과의 정합성을 유지하고, 의미론적 및 구문적 풍부성을 포착하는 데 기여하고 있음을 나타냅니다.



### Unsupervised Speech Segmentation: A General Approach Using Speech Language Models (https://arxiv.org/abs/2501.03711)
- **What's New**: 이번 논문에서는 기존의 접근 방식들인 Speaker Diarization을 바탕으로 하여 음성 세분화( Speech Segmentation)을 위한 비지도 학습 방법을 제안합니다. 이 방법은 다채로운 음향-의미적 스타일 변화를 처리하는 데 초점을 맞추며, 기존의 단일 스타일 변화 처리에서는 나아간 진전을 보여줍니다. 특히, 감정이나 화자처럼 텍스트로 잘 변환되지 않는 정보를 중심으로 한 세분화를 도모합니다.

- **Technical Details**: 제안된 방법은 음성과 오디오 신호를 이산 음향 단위로 표현한 후 언어 모델을 적용하여 순서의 가능성을 극대화하는 Speech Language Models (SLMs)를 활용합니다. 우리는 초기 단계로 오디오를 동등한 크기의 세그먼트, 즉 'acoustic-sentences'로 나눈 뒤, SLMs에서 얻은 확률을 사용해 연속된 문장을 점수화하고 최종적으로 병합할 세그먼트를 선택합니다. 이는 비지도 세분화에 있어 새로운 접근 방식을 제공하는 것을 목표로 합니다.

- **Performance Highlights**: 제안된 방법은 경계 탐지(boundary detection), 세그먼트 순수도(segment purity) 및 과다 세분화(over-segmentation) 측면에서 평가된 기준선보다 뛰어난 성능을 보였습니다. 연구 결과, 해당 접근 방식은 매우 효율적이며, 다양한 음향-의미적 개념의 세분화에서 우수한 성능을 입증했습니다. 코드와 세트업은 논문에 링크되어 제공됩니다.



### SLAM: Towards Efficient Multilingual Reasoning via Selective Language Alignmen (https://arxiv.org/abs/2501.03681)
Comments:
          Accepted by COLING 2025 (Oral)

- **What's New**: 이 논문에서는 대형 언어 모델(LLM)이 영어 추론 작업에서 크게 발전했지만 다국어 추론에서는 여전히 어려움을 겪고 있다는 점을 지적합니다. 연구자들은 다국어 이해를 통해 모델이 영어가 아닌 질문을 이해하도록 훈련하는 방법을 제안하였습니다. 하지만, 이 방법은 과도한 컴퓨팅 자원과 큰 정보 손실 문제를 안고 있습니다.

- **Technical Details**: 먼저, 모델의 다국어 이해를 개선하기 위해 비관련 레이어와 파라미터를 지나치게 조정하는 것이 큰 문제로 드러났습니다. 연구팀은 SLAM이라는 새로운 접근 방식을 제안하며, 이는 다국어 처리에 관여하는 레이어를 정확히 식별하고 미세 조정(fine-tune)합니다. SLAM은 7B 및 13B LLM 내에서 전체 파라미터의 6.5-8%만 조정하여 6개의 피드포워드 서브 레이어(feed-forward sub-layers)만을 미세 조정합니다.

- **Performance Highlights**: 경험적 결과에 따르면, SLAM은 10개 언어에서 모든 강력한 기준선(baselines)보다 우수한 평균 성능을 달성했습니다. 뿐만 아니라 SLAM은 단일 훈련 단계에서만 진행되어, 두 단계 방법에 비해 훈련 시간을 4.1-11.9배 단축시킵니다. 이는 SLAM이 다국어 모델의 효율성을 크게 향상시키는 방법임을 보여줍니다.



### A Diversity-Enhanced Knowledge Distillation Model for Practical Math Word Problem Solving (https://arxiv.org/abs/2501.03670)
- **What's New**: 이 논문에서는 Math Word Problem (MWP) 해결을 위한 새로운 다각화 지식 증류(Diversity-enhanced Knowledge Distillation, DivKD) 모델을 제안합니다. 기존의 Seq2Seq 모델과 그 변형들이 한정된 다양성 있는 해답 방정식을 생성하는 데 어려움을 겪고 있는 가운데, 우리의 접근법은 교사 모델로부터 선택적으로 고품질 지식을 전이하여 다양한 방정식을 학습하는 적응형 다양성 증류(AdaKD) 방법을 도입합니다.

- **Technical Details**: 제안된 DivKD 모델은 학생 모델이 다각화된 방정식을 캡처할 수 있게 돕기 위해, 조건부 변분 오토인코더(Conditional Variational Autoencoder, CVAE)를 통합하여 방정식의 다양성 분포를 모델링합니다. 이를 통해 다양한 해결 방정식을 생성할 수 있는 잠재 변수를 샘플링하는 다양성 사전 네트워크를 사용하며, 고품질 소프트 및 하드 레이블을 선택적으로 훈련하여 학생 모델의 학습을 돕습니다.

- **Performance Highlights**: 다양한 MWP 벤치마크 데이터셋에 대한 광범위한 실험을 통해 제안한 방법이 기존 강력한 기준 모델보다 더 높은 답안 정확도를 달성함을 입증하였습니다. 특히, DivKD 모델은 모델의 효율성을 유지하면서도 높은 성능을 발휘하는 것으로 나타났습니다.



### KG-TRICK: Unifying Textual and Relational Information Completion of Knowledge for Multilingual Knowledge Graphs (https://arxiv.org/abs/2501.03560)
Comments:
          Camera ready for COLING 2025

- **What's New**: 이 논문에서는 멀티링구얼 지식 그래프(Multilingual Knowledge Graphs)에서 텍스트 및 관계 정보의 완성을 통합하는 KG-TRICK이라는 새로운 모델을 제안합니다. KG-TRICK은 지식 그래프 완성(Knowledge Graph Completion, KGC)과 지식 그래프 향상(Knowledge Graph Enhancement, KGE)이라는 두 가지 독립적인 작업을 하나의 프레임워크로 통합하여 서로의 이점을 활용하도록 설계되었습니다. 또한, 10개 언어에서 25,000개 이상의 개체를 포함하는 WikiKGE-10++라는 대규모 수작업 벤치마크를 소개하여 멀티링구얼 KGs의 평가를 지원합니다.

- **Technical Details**: KG-TRICK은 텍스트와 관계 정보를 함께 완성하는 새로운 시퀀스-투-시퀀스(sequence-to-sequence) 모델로, 다양한 언어에서의 정보를 효율적으로 결합합니다. KGC와 KGE의 상호 의존성을 활용하여, KGC에서 얻은 언어 독립적인 관계 정보가 KGE의 질을 향상시키는 구조를 가지고 있습니다. 이를 통해 KG를보다 완전하게 만들고, 각각의 개체를 다른 언어의 이름과 설명에 맞춰 정렬하는 데 효과적입니다.

- **Performance Highlights**: KG-TRICK은 기존의 최첨단 모델보다 유사한 규모에서 더 우수한 성능을 보이며, 대규모 언어 모델에 비해서도 경쟁력 있는 성능을 달성합니다. 본 연구는 멀티링구얼 KGs의 질 향상과 다양한 NLP 응용 프로그램에서의 활용 가능성을 확장하는 데 기여할 것으로 기대됩니다. 이 모델과 함께 소개된 WikiKGE-10++는 향후 연구를 위한 중요한 자원으로 기능할 것입니다.



### Beyond Factual Accuracy: Evaluating Coverage of Diverse Factual Information in Long-form Text Generation (https://arxiv.org/abs/2501.03545)
- **What's New**: 이 논문은 ICAT라는 긴 형식의 텍스트 생성을 위한 평가 프레임워크를 제시합니다. ICAT은 긴 출력 텍스트를 원자적 주장(atomic claims)으로 분해하고 각 주장을 신뢰할 수 있는 지식 출처에서 검증하며 다양한 측면(aspect)과의 정렬(alignment)을 계산합니다. 이를 통해 RGB는 LLM(대형 언어 모델)에서 생성된 긴 응답의 다양한 사실 정보의 포괄성(completeness)과 다양성(diversity)을 평가할 수 있습니다.

- **Technical Details**: ICAT 프레임워크는 세 가지 구현 방식으로 연구됩니다. ICAT-M은 수동으로 얻은 다양한 주장을 기준으로 하며, 주장의 관련된 주제에 대해 지식 출처에서 정보를 검색하여 각 주장의 정확성을 검증합니다. ICAT-S와 ICAT-A는 인간의 판단 기준 없이 LLM을 사용하여 피상적 레이블링(pseudo-labeling)을 통해 실제 주제와의 정렬을 수행하여 서로 다른 방식으로 접근합니다.

- **Performance Highlights**: ICAT 실험에서는 ClueWeb 데이터셋을 사용하여 LLM의 응답을 평가했으며, 인간의 판단과의 높은 상관관계(상관계수 0.4 이상의 Pearson's ρ)를 보여줍니다. ICAT 프레임워크는 다양한 사실 정보의 커버리지를 평가하고, LLM 출력의 원자적 주장을 분해하여 이해 가능한 분석을 제공합니다. 또한 프레임워크의 모듈식 설계 덕분에 다양한 도메인과 데이터셋에 쉽게 적응할 수 있어 LLM의 긴 응답 평가에 가치 있는 도구로 자리잡고 있습니다.



### A Sequential Optimal Learning Approach to Automated Prompt Engineering in Large Language Models (https://arxiv.org/abs/2501.03508)
- **What's New**: 이 논문은 자동화된 프롬프트 엔지니어링(automated prompt engineering)을 위한 최적의 학습 프레임워크를 제안합니다. 이 프레임워크는 효과적인 프롬프트 기능을 식별하고 효율적으로 한정된 평가 예산을 할당하며, Bayesian 회귀(Bayesian regression)를 사용하여 유사한 프롬프트 간의 상관관계를 활용합니다. 또한 Knowledge-Gradient(KG) 정책을 채택하여 고품질 프롬프트의 대규모 탐색을 가능하게 합니다.

- **Technical Details**: 제안된 방법은 설명 가능한(feature-based) 프롬프트 표현을 사용하여 다양한 범주의 특징을 고려합니다. 이 접근은 프롬프트의 특성이 LLM 응답의 품질과 어떻게 연관되는지를 통계적으로 모델링하며, 피처 간의 상관관계를 발견합니다. 프롬프트 평가의 기회가 제한된 상황에서 최적의 학습 정책을 수립하기 위해, 이 방법은 유한 지평의 이산 시간 마코프 결정 프로세스(MDP)로 형식화됩니다.

- **Performance Highlights**: KG 정책을 사용하여 프롬프트 선택을 수행한 결과, 30회의 프롬프트 평가 이내에 높은 품질의 프롬프트를 찾아내는 것이 가능했습니다. 이 방법은 벤치마크 전략보다 우수한 성능을 보여주었으며, 특히 LLM 응답에 대한 불확실성이 높은 과제에서 두드러진 성과를 보였습니다. 전체적으로 KG 정책은 평가 비용이 큰 응용 프로그램에 자동화된 프롬프트 엔지니어링을 널리 배포할 수 있는 가능성을 열어줍니다.



### Can LLMs Design Good Questions Based on Context? (https://arxiv.org/abs/2501.03491)
- **What's New**: 이 논문은 LLM(대형 언어 모델)이 생성한 질문의 특성과 이를 인간이 생성한 질문과 비교하여 평가하는 새로운 접근 방식을 소개합니다. 자동화된 LLM 기반 평가 방법을 개발하였으며, 질문의 길이, 유형, 맥락 범위 및 답변 가능성과 같은 다양한 차원에서 질문을 분석하였습니다. LLM이 생성한 질문의 고유한 특성을 밝히며, 이는 질문 품질 및 후속 응용 프로그램의 연구에 기여할 수 있습니다.

- **Technical Details**: 연구에서, LLM은 문맥(C)과 질문 생성 프롬프트(P)의 결합을 통해 질문을 생성합니다. 이 과정에서 LLM이 생성한 질문의 길이 및 유형을 통계적으로 분석하고, 질문이 특정 맥락과 어떻게 관련되는지를 평가합니다. 인간의 질문과 비교하기 위해, LLM이 생성한 질문의 길이를 측정하고, 질문의 응답 가능성을 평가하기 위한 새로운 방법론을 도입하였습니다.

- **Performance Highlights**: 실험 결과, 두 가지 대표적인 LLM인 GPT-4o와 LLaMA-3.1-70b-Instruct를 사용하여 각 모델이 동일한 256개의 위키 문맥에서 1,024개의 질문을 생성했습니다. 평가를 통해 인간 주석자와 LLM 간의 일치율이 80%를 초과했으며, 평균 피어슨 상관관계는 인간 주석자와 LLM 간에 0.77로 나타났습니다. 이런 결과는 질문 생성 및 평가에 대한 LLM의 신뢰성을 강조합니다.



### Women, Infamous, and Exotic Beings: What Honorific Usages in Wikipedia Reveal about the Socio-Cultural Norms (https://arxiv.org/abs/2501.03479)
- **What's New**: 이번 연구는 벵골어(Bengali) 및 힌디어(Hindi) 위키피디아 기사에서 경어(pronouns) 사용을 대규모로 탐구하는 독창적인 접근 방식을 제시합니다. 연구는 사회문화적 요인(socio-cultural factors)이 언어에 미치는 영향을 조명하고, 이를 10,000개의 기사에서 분석하였습니다.

- **Technical Details**: 연구에서는 LLM(GPT-4o)을 활용하여 다양한 사회적 특성(sociodemographic features)들, 예를 들어 성별(gender), 나이(age), 유명도(fame), 이국적임(exoticness)을 기준으로 벵골어 및 힌디어 기사에서 경어 사용을 주석(annotation)하였습니다. 경어의 일관된 사용 추세를 발견하였고, 벵골어가 힌디어보다 경어 사용이 더 일반적임을 확인했습니다.

- **Performance Highlights**: 연구 결과, 악명 높은(infamous), 젊은(juvenile), 이국적인(exotic) 존재에 대해서는 비경어(non-honorific) 사용이 더 많은 것으로 나타났습니다. 또한 힌디어에서는 성별에 따른 경어 사용 편향(gender bias)이 관찰되어 남성은 여성보다 경어로 더 많이 언급되는 경향이 있었습니다.



### Reading with Intent -- Neutralizing Inten (https://arxiv.org/abs/2501.03475)
- **What's New**: 이 연구는 대형 언어 모델(LLM)의 상황 맥락에서 다양한 감정의 톤이 모델 성능에 미치는 영향을 평가하는 Reading with Intent 작업을 확장했습니다. 기존의 비판적인 유머(sarcasm)에 대한 연구에 기반하여, 우리는 새로운 데이터셋을 $11$ 가지의 감정으로 변환해 생성했습니다. 이 데이터셋을 사용하여 특정 감정 톤으로 텍스트를 조정할 수 있는 감정 변환 모델(emotion translation model)을 개발 하였습니다.

- **Technical Details**: 이 논문에서는 Open-domain QA를 기반으로 한 데이터셋 생성을 통해 LLM의 감정 변환 과정을 수립하였습니다. 각 쿼리에 대해 Wikipedia에서 최대 10개의 관련 문서를 검색한 후, 특정 감정으로 변환하는 과정을 거칩니다. 총 $11$ 가지의 감정을 구현하는 과정에서는 Llama 3, Qwen 2.5와 같은 여러 서로 다른 아키텍처의 LLM을 활용하여 편향을 줄였습니다.

- **Performance Highlights**: 인간 평가 결과, 감정 변환을 통해 교육된 LLM은 합성으로 생성된 데이터에서 이익을 얻었습니다. Reading with Intent 작업에 적용한 결과, 논문은 비판적인 유머가 포함된 문장이 중화(neutralized) 되었을 때, 과제의 전반적인 결과가 약 $3	extrm{	extbf{	extpercent}}$ 향상됨을 보여주었습니다. 이 연구는 감정 변환을 통해 LLM의 수행능력을 효과적으로 개선하는 가능성을 제시합니다.



### MTRAG: A Multi-Turn Conversational Benchmark for Evaluating Retrieval-Augmented Generation Systems (https://arxiv.org/abs/2501.03468)
- **What's New**: 최근 대규모 언어 모델(LLM)에서 Retrieval-augmented generation (RAG) 작업이 큰 인기를 끌고 있습니다. 그러나 다중 회차 대화에 대한 평가가 간과되어 왔으며, 이는 시스템이 전 대화의 맥락에서 질문에 응답해야 하므로 추가적인 도전 과제가 존재합니다. 본 논문에서는 mtRAG라는 인간 생성의 다중 회차 RAG 벤치마크를 제시하며, 전체 RAG 파이프라인의 평가를 위해 다양한 실제 속성을 반영하고 있습니다.

- **Technical Details**: mtRAG는 110개의 대화를 포함하고 있으며, 각 대화는 평균적으로 7.7 회차로 구성되어 있습니다. 벤치마크는 사용자 경험을 시뮬레이션하기 위해 인간 주석자가 실시간으로 RAG 에이전트와 상호작용하여 생성합니다. RAG 시스템의 탐색 성능을 평가하고, 9개의 LLM 모델의 생성 성능을 다각도로 분석한 결과, 모든 모델이 질의 및 대화의 후반부에서 어려움을 겪음을 확인했습니다.

- **Performance Highlights**: mtRAG의 객관적인 평가 결과, 최신 LLM 기반 RAG 시스템도 이 벤치마크의 작업에서 고전하는 모습을 보였습니다. 특히 잘못된 응답이나 답할 수 없는 질문에 대한 처리에서 어려움이 있었습니다. 벤치마크는 인공지능 커뮤니티에게 두 가지 유형의 데이터(인간 생성과 합성 생성)의 상대적인 이점을 분석하고 이해하는 데 도움을 줄 것입니다.



### ISSR: Iterative Selection with Self-Review for Vocabulary Test Distractor Generation (https://arxiv.org/abs/2501.03462)
- **What's New**: 이 연구는 대만의 대학 입학 시험에서 사용되는 영어 어휘 질문을 분석하고, 어휘 테스트 항목 설계에서 교사를 지원하기 위한 주요 제한 사항을 점검합니다. 특히, 개인화된 학습을 위한 자동 어휘 생성의 가능성을 탐구하며, 이를 기반으로 자기 검토 메커니즘(self-review mechanism)을 활용한 새로운 프레임워크인 ISSR(Iterative Selection with Self-Review)을 제안합니다.

- **Technical Details**: ISSR 프레임워크는 후보 생성기(candidate generator), 선택기(selector), 검증기(validator)의 세 가지 모듈로 구성됩니다. 사전 훈련된 언어 모델(pretrained language model, PLM)을 사용해 문맥과 관련된 디스트랙터(distractors)를 생성하고, LLM 기반 자기 검토 메커니즘을 도입하여 유효성을 보장합니다. 연구에서는 LLMs가 어휘 테스트를 위한 디스트랙터 생성을 지원할 수 있는지에 대한 질문을 다루고 있습니다.

- **Performance Highlights**: ISSR 프레임워크의 실험 결과는 신뢰할 수 있는 디스트랙터를 생성하는 데 있어 유망한 성과를 보여주며, 자기 검토 메커니즘이어휘 질문의 유효성을 유지하는 데 효과적임을 입증합니다. 이 연구는 어휘 이해력 평가를 위한 효율적이고 일관된 방법론의 발전에 기여할 것으로 기대됩니다.



### Text to Band Gap: Pre-trained Language Models as Encoders for Semiconductor Band Gap Prediction (https://arxiv.org/abs/2501.03456)
- **What's New**: 이 연구에서는 반도체 물질의 밴드 갭(band gap)을 예측하기 위해 트랜스포머 기반 언어 모델인 RoBERTa를 인코더로 활용하는 방법을 탐구합니다. 기존의 양자 화학 시뮬레이션 방법인 밀도 범함수 이론(DFT)은 시간 소모가 크고 계산 비용이 많이 들기 때문에 높은 처리량(high-throughput) 물질 스크리닝에 제약이 있습니다. 반면, 본 연구에서는 복잡한 특성 엔지니어링 없이 텍스트 데이터로부터 직접 밴드 갭을 예측하여 장점을 제시합니다.

- **Technical Details**: RoBERTa 모델은 Bidirectional encoder representations from transformers (BERT)를 바탕으로 만들어진 발전된 모델로, 텍스트 내 토큰 간의 맥락적 관계를 효과적으로 포착합니다. 우리는 AFLOW 데이터셋에서 밴드 갭 예측을 위해 이 모델을 파인튜닝(fine-tuning)했으며, 텍스트 설명을 결합하여 문서화한 후, 커스텀 회귀 헤드를 사용하여 스칼라 밴드 갭 값을 예측하도록 설계했습니다. 최종 출력층은 예측된 밴드 갭에 해당하는 하나의 스칼라 값을 생성합니다.

- **Performance Highlights**: RoBERTa 기반의 모델은 최소한의 파인튜닝만으로 평균 절대 오차(mean absolute error, MAE) 약 0.33 eV를 달성하였으며, 기존의 얕은 머신러닝(ML) 모델보다 뛰어난 성능을 보였습니다. 특히, RoBERTa 인코더 레이어를 고정한 상태에서 선형 회귀 헤드만 훈련해도 정확도가 거의 동일하게 유지되어, 사전 훈련된 RoBERTa 모델의 높은 적응성과 효율성을 입증하였습니다. 이러한 결과는 트랜스포머 기반 언어 모델이 반도체 물질의 속성 예측 작업에 효과적이고 다목적 인코더로 사용할 수 있다는 것을 보여줍니다.



### Finding A Voice: Evaluating African American Dialect Generation for Chatbot Technology (https://arxiv.org/abs/2501.03441)
- **What's New**: 이 연구는 현대의 대형 언어 모델(Large Language Models, LLMs)이 아프리카계 미국인 방언(African American Vernacular English, AAVE)을 생성할 수 있는 능력을 평가하고, AAVE 사용이 챗봇 애플리케이션에서의 사용자 경험에 미치는 영향을 조사합니다. 특히, 헬스케어와 교육 분야와 같은 여러 도메인에서 AAVE를 사용하는 사용자들이 표준 미국 영어(Standard American English, SAE) 챗봇을 선호한다는 점을 발견했습니다. 이는 AI 시스템의 포괄성 설계의 복잡성을 강조하며, 다양한 사용자 요구를 충족시킬 수 있는 기술 개발의 필요성을 강조합니다.

- **Technical Details**: 연구는 고객 지원, 상업, 의료, 교육, 사회적 동반자와 같은 5개의 인기 챗봇 애플리케이션 도메인을 식별하였습니다. 각각의 도메인에서 10 턴 대화의 샘플을 수집하고, LLM을 통해 생성된 대화 데이터를 활용하여 AAVE 표현의 스타일링 영향을 측정합니다. 모델링에서는 다이얼렉트 표현을 응답 생성과 분리하여 다루는 방식(E(I, Da, Db) → O)을 채택했습니다.

- **Performance Highlights**: LLMs는 AAVE 유사 언어를 생성하는 데 능숙하지만, AAVE를 사용하는 챗봇 채택은 여러 어려움에 직면합니다. AAVE 사용자들은 SAE 챗봇을 선호하는 경향이 있으며, AAVE 비율이 높을수록 챗봇의 신뢰성 및 역할 적합성 평가가 낮아지는 것으로 나타났습니다. 이러한 결과는 챗봇 시스템 설계의 복잡성을 보여주며, 더 포괄적인 챗봇 기술 개발을 위한 중요한 통찰력을 제공합니다.



### DAMAGE: Detecting Adversarially Modified AI Generated Tex (https://arxiv.org/abs/2501.03437)
- **What's New**: 본 논문에서는 AI 생성 텍스트의 방향성을 수정하는 새로운 소프트웨어 도구인 AI humanizer에 대해 연구합니다. 저자들은 19개의 AI humanizer 및 패러프레이징 툴을 분석하고 이러한 기술이 원본 텍스트의 의미를 얼마나 잘 유지하는지를 평가합니다. 결과적으로 기존의 AI 탐지 소프트웨어가 humanized 텍스트를 효과적으로 탐지하지 못하는 경우가 많다는 것을 보여줍니다.

- **Technical Details**: 이 연구는 19개의 AI humanizers and paraphrasing tools를 질적으로 감사하고 그 변환 방식에 대해 분석합니다. 인공지능 텍스트 탐지 도구들에 대한 humanizers의 효과에 대한 바탕을 연구하고, AI 텍스트를 탐지하기 위한 딥러닝 기반의 탐지기를 제안합니다. 이 모델은 기존 훈련 데이터에 포함되지 않은 humanizer에 대해서도 강력한 일반화 성능을 유지할 수 있음을 보여줍니다.

- **Performance Highlights**: AI humanizer가 AI 탐지기를 우회하는 데 있어 효과적이라는 것과, 우리의 딥러닝 기반 탐지기가 여러 가지 방식으로 인공지능 생성 텍스트를 탐지할 수 있는 능력을 갖추고 있다는 것을 입증했습니다. 특히 훈련 중에도 보지 못한 humanizer에 대해서도 우리의 탐지기는 상당히 강건함을 나타냈습니다. 이러한 연구는 인공지능 텍스트 탐지 기술의 발전과 미래 방향에 큰 기여를 할 것으로 기대됩니다.



### BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations (https://arxiv.org/abs/2501.03403)
- **What's New**: 이 논문은 Document Question-Answering (QA)를 위한 통합 데이터셋을 제시합니다. 여러 공개 데이터셋을 결합하여 정보를 추출하고, 문서에서 답변을 찾는 위치를 바운딩 박스로 포함시켰습니다. 이를 통해 대형 언어 모델(Large Language Models, LLMs)의 훈련과 평가에 적합한 자원을 제공합니다.

- **Technical Details**: 본 연구에서는 기존의 Document AI 작업을 QA 작업으로 재정의하고, 기존 데이터셋의 위치 정보를 통합하는 방법에 대해 논의합니다. 연구 질문은 QA 포맷으로의 데이터셋 통합, LLM에 의해 생성된 질문의 재구성으로 정확도를 높일 수 있는지, 레이아웃 정보 포함이 모델 성능에 미치는 영향을 다룹니다. 또한 머신 러닝과 딥러닝 기법, 특히 자연어 처리(NLP)와 관련된 방법들이 강조됩니다.

- **Performance Highlights**: 다양한 프롬프트 기법에 따른 모델의 성능을 평가하여 문서 이해에 가장 효과적인 접근 방식을 식별합니다. 기존의 Document AI 데이터셋들이 효과적으로 위치 정보를 통합하지 못해 hallucination을 줄이고 성능 향상을 저해한 점을 지적하며, 새로운 데이터셋은 더 나은 문서 이해를 지원할 것으로 기대됩니다.



### Advanced Machine Learning Techniques for Social Support Detection on Social Media (https://arxiv.org/abs/2501.03370)
- **What's New**: 이 연구는 소셜 미디어에서 온라인 사회적 지원(online social support)의 영향을 이해하려는 노력을 담고 있습니다. 지원 내용에 대한 이진(binary) 및 다중 클래스(multiclass) 분류를 포함한 데이터 세트를 사용하여 사회적 지원을 세 가지 작업으로 나누어 분석합니다. 이 작업들은 지원과 비지원의 구별, 개인 또는 그룹을 대상으로 한 지원 여부, 그리고 특정 사회적 지원의 유형을 분류하는 것입니다.

- **Technical Details**: 데이터 불균형 문제를 해결하기 위해 K-means clustering을 사용하여 데이터 세트를 균형 있게 조정하였으며, 원래의 불균형 데이터와 결과를 비교했습니다. 첨단 기계 학습 기법인 transformers와 zero-shot learning 접근법을 적용하여 다양한 맥락에서 사회적 지원 수준을 예측합니다. 연구에서 사용한 baseline 모델과의 비교를 통해 transformer 기반 방법이 우수한 성능을 보였음을 알 수 있습니다.

- **Performance Highlights**: 연구 결과, 두 번째 작업에서는 매크로 F1 점수가 0.4% 향상되었고, 세 번째 작업에서는 0.7% 향상되었습니다. 이러한 결과는 psycholinguistic 및 unigram 기반 TF-IDF 값을 활용한 기존 작업에 비해 향상된 성과로 평가됩니다.



### Analyzing Bias in Swiss Federal Supreme Court Judgments Using Facebook's Holistic Bias Dataset: Implications for Language Model Training (https://arxiv.org/abs/2501.03324)
- **What's New**: 이번 연구는 스위스 판례 예측 데이터셋(SJP Dataset)에서 포함된 편향을 분석하여, 법적 결정 과정에서의 공정성을 보장하고자 하였다. 자연어 처리(NLP)에 있어서 훈련 데이터의 편향이 어떻게 판단 예측의 정확성에 영향을 미치는지를 조사함으로써, NLP 모델의 공정한 결정이 이루어질 수 있는 기반을 마련하는 목표다. 'Holistic Bias dataset'에서 제공하는 사회적 편향 묘사를 활용하여 데이터셋 내 데이터의 영향력을 조사한다.

- **Technical Details**: 데이터셋의 편향 분석은 'dispreferred'라는 레이블이 붙은 특성들을 이용하여 이루어진다. 새로운 버전인 Holistic Bias Dataset 1.1에서는 769개의 독특한 묘사가 포함되어 있으며, 이 중 70개가 'dispreferred'로 레이블이 붙어 있다. 텍스트 내용의 손실 없이 토큰 수의 한계를 관리하기 위해 LexRank Summarizer와 같은 최적화된 방법론을 사용하였다.

- **Performance Highlights**: 본 연구는 훈련 데이터의 불균형 문제를 해결하기 위해 클래스 가중치를 조정하며, 세 가지 서로 다른 설정으로 모델을 미세 조정하였다. SJP 데이터셋의 요구 사항을 충분히 충족하면서 NLP 모델의 성능 향상을 추구하였다. 결과적으로 교수 및 법적 맥락 속에서의 공정성 기반을 강화하며, 향후 연구 및 실제 적용 가능성을 제시하고자 한다.



### ADePT: Adaptive Decomposed Prompt Tuning for Parameter-Efficient Fine-tuning (https://arxiv.org/abs/2501.03291)
- **What's New**: 이 논문에서는 Adaptive Decomposed Prompt Tuning (ADePT)을 소개하였으며, 이는 짧은 soft prompt와 shallow token-shared feed-forward neural network로 구성됩니다. ADePT는 각 토큰에 대한 embedding 오프셋을 학습하여 입력 모델에 따라 가변적인 적응형 오프셋을 제공합니다. 이를 통해 ADePT는 기존의 Prompt Tuning (PT) 및 그 변형들과 비교하여 더 뛰어난 적응 성능을 발휘하면서도 추가적인 훈련 가능한 매개변수나 추론 시간을 요구하지 않습니다.

- **Technical Details**: Adaptive Decomposed Prompt Tuning (ADePT)는 짧은 soft prompt와 shallow token-shared feed-forward neural network를 활용하여 token embedding 오프셋을 학습합니다. ADePT는 다양한 NLP 작업에서 기존의 Parameter-Efficient Fine-Tuning (PEFT) 방법들을 초월하는 성능을 보여주었습니다. 논문에서는 23개의 NLP 작업 및 4개의 다양한 스케일의 PLM에 걸쳐 ADePT의 상위 성능을 입증했습니다.

- **Performance Highlights**: ADePT는 기존의 PLM을 활용하는 다양한 NLP 작업에서 최고의 성능을 발휘하며, 특히 일부 시나리오에서는 전체 fine-tuning baseline을 초월하였습니다. 연구에 따르면 ADePT는 더 짧은 soft prompt를 활용하여 빠른 inference 속도를 유지하면서도 높은 적응 성능을 제공합니다. 이 연구 결과는 ADePT가 향후 작업에서 PLM의 효율성을 더욱 개선할 수 있는 가능성을 보여줍니다.



### HonkaiChat: Companions from Anime that feel alive! (https://arxiv.org/abs/2501.03277)
Comments:
          5 pages, 4 figures. This is a preprint. Not yet submitted to a journal or conference. More iterated versions to be updated

- **What's New**: 이 논문은 현대의 대화형 에이전트들이 보유한 한계를 해결하기 위해, 이벤트 기반의 대화 프레임워크를 제안합니다. 캐릭터 특화 데이터를 가지고 모델을 미세 조정하여 대화 prompt에 동적 이벤트를 삽입함으로써 사용자와의 상호작용을 더 깊고 자연스럽게 만듭니다. 이를 Honkai: Star Rail 게임 캐릭터와의 인터랙션에 적용하여 대화의 몰입도를 높이는 가능성을 보여주었습니다.

- **Technical Details**: 이벤트 기반 프레임워크는 사용자와의 대화를 강화하기 위해, GPT-4와 같은 언어 모델을 활용하여 캐릭터 특화된 데이터로 모델을 세밀하게 조정합니다. 달리 새로운 데이터셋을 구축하기 위해 March 7th라는 캐릭터의 대화 및 이벤트를 사전 훈련된 Llama 3.1 8b 모델로 처리합니다. 추가로, OCEAN 및 MBTI 성격 분석을 사용하여 역할 플레이와 이벤트 생성 작업을 지원합니다.

- **Performance Highlights**: 본 연구에서는 동적 이벤트를 삽입한 대화 prompt가 정적이거나 맥락 반응형 시스템에 비해 명확한 개선을 보여주었다고 평가하였습니다. 자연스러운 언어 생성이 향상되고, 사용자와의 대화에서 캐릭터의 일관성을 유지할 수 있는 점이 큰 이점으로 작용하였습니다. 이러한 접근 방식은 RPG 챗봇의 몰입 경험을 향상시키는 잠재력을 드러냅니다.



### ComMer: a Framework for Compressing and Merging User Data for Personalization (https://arxiv.org/abs/2501.03276)
Comments:
          13 pages, 7 figures

- **What's New**: 이 논문에서는 ComMer - Compress and Merge라는 새로운 프레임워크를 소개하여 대형 언어 모델(LLMs)을 효율적으로 개인화하는 방법을 제안합니다. ComMer는 사용자의 문서를 압축하여 간결한 표현으로 변환한 다음, 이를 결합해 냉동된 LLM에 입력합니다. 이 접근 방식은 사용자 수가 많을 때 자원을 절약하고, 훈련 비용을 줄이며, 제한된 데이터로도 품질을 개선할 수 있는 장점을 가지고 있습니다.

- **Technical Details**: ComMer는 세 가지 단계로 이루어진 아키텍처를 통해 작동합니다. 첫 번째로 각 문서가 독립적으로 소프트 프롬프트로 압축되며, 이러한 압축은 학습 가능한 압축 임베딩과 LoRA 가중치를 활용하여 이루어집니다. 두 번째로 압축된 표현들이 평균 풀링(mean pooling)을 통해 집계되어 단일 소프트 프롬프트로 결합되고, 마지막으로 이 집계된 소프트 프롬프트가 냉동된 LLM에 연결되어 원하는 출력을 생성합니다.

- **Performance Highlights**: 실험 결과, ComMer는 제한된 토큰 예산에서 개인화된 기술 학습(task)에서 우수한 품질을 보여주었고, 문서의 수가 많아질수록 품질이 향상되는 경향을 보입니다. 그러나 지식 집중형 작업에서는 여러 문서의 정보를 모두 표현하는 데 제한이 있어 품질 저하가 발생하는 것으로 나타났습니다. 이는 다중 문서 압축을 통한 개인화에서의 무역 오프와 잠재적 최적화 방향에 대한 통찰을 제공합니다.



### LLM Content Moderation and User Satisfaction: Evidence from Response Refusals in Chatbot Arena (https://arxiv.org/abs/2501.03266)
- **What's New**: 이번 연구는 사용자 만족도에 대한 콘텐츠 조정(content moderation)의 영향을 탐구합니다. 우리는 50,000개의 Chatbot Arena 응답 쌍을 분석하여 윤리적 문제에 따른 거부(refusals)를 다른 기술적 문제나 정보 부족으로 인한 거부와 구분하기 위해 특별히 조정된 RoBERTa 모델을 사용했습니다. 이 연구는 LLM(대규모 언어 모델)의 안전성과 윤리적 정렬에 논의되는 내용과 새로운 관점을 제시합니다.

- **Technical Details**: 우리는 수동으로 레이블이 붙은 데이터를 기반으로 훈련된 RoBERTa 모델을 통해 데이터 분석을 수행했습니다. 분석 결과, 콘텐츠 조정은 사용자에 대한 상당한 거부 패널티(refusal penalty)를 가져오며, 윤리에 기반한 거부는 사용자 선호 응답에 비해 약 4배 덜 발생하는 것으로 나타났습니다. 또한 민감한 프롬프트에 대한 거부가 더 낮은 윤리적 우려에 비해 더 높은 승률을 획득한다는 것을 발견했습니다.

- **Performance Highlights**: 안건과 구문이 중요한 역할을 하며, 프롬프트와 밀접하게 관련된 긴 응답이 더 좋은 성능을 보이는 것으로 나타났습니다. 연구에서는 또한 LLM-as-a-Judge 방법을 사용할 경우 거부 패널티가 눈에 띄게 낮아지는 경향이 있음을 발견하였습니다. 이러한 발견은 윤리적 안전 장치와 사용자 만족도 간의 균형을 맞추기 위한 세분화된 조정 전략의 필요성을 강조합니다.



### REINFORCE++: A Simple and Efficient Approach for Aligning Large Language Models (https://arxiv.org/abs/2501.03262)
Comments:
          this is a tech report

- **What's New**: 이 논문은 인간의 피드백을 통한 강화학습(RLHF)의 개선을 위한 새로운 알고리즘 REINFORCE++를 소개합니다. 이 알고리즘은 기존의 REINFORCE 메커니즘을 발전시켰으며, PPO의 주요 최적화 기법을 통합하면서도 크리틱 네트워크의 필요성을 제거했습니다. 이를 통해 구현의 복잡함을 줄이고, 훈련 안정성을 높이며, 계산적 효율성을 증대시킬 수 있습니다.

- **Technical Details**: REINFORCE++는 KL 다이버전스 패널티와 PPO의 클리핑 메커니즘을 통합하여 훈련 과정에서 안정성을 유지합니다. 알고리즘은 입력 프롬프트 및 생성된 응답 간의 KL 다이버전스를 보상 함수에 포함하여 신뢰 지역(trust region)을 유지하는 방식으로 정책 업데이트를 수행합니다. 또한, 미니 배치 업데이트를 통해 데이터 처리를 효과적으로 관리하고 수렴 속도를 개선하는 특징이 있습니다.

- **Performance Highlights**: 실험 결과, REINFORCE++는 GRPO에 비해 뛰어난 안정성을 보이며, PPO보다 계산 효율성이 높음에도 불구하고 유사한 성능을 유지하는 것으로 나타났습니다. 연구자는 REINFORCE++의 성능을 일반 데이터 세트와 도메인 특정 데이터 세트에서 평가하였으며, RLHF 작업에 대한 매력적인 대안으로서의 가능성을 입증했습니다.



### Toward Inclusive Educational AI: Auditing Frontier LLMs through a Multiplexity Lens (https://arxiv.org/abs/2501.03259)
- **What's New**: 본 논문은 교육적 문맥에서 대형 언어 모델(LLMs)의 문화적 편향을 평가하고 완화하기 위한 새로운 프레임워크를 제안합니다. 이 프레임워크는 복합성(multiplexity)이라는 관점에서 접근하여, 다양한 문화적 시각을 공존시키고 인식론을 다층적으로 통합하는 것을 목표로 합니다. 또한, LLM의 출력에서 종종 나타나는 문화적 양극화 현상을 분석하여 문제를 해결하기 위한 두 가지 전략을 제시합니다.

- **Technical Details**: 제안된 프레임워크는 상황적으로 구현된 복합 LLM(Contextually-Implemented Multiplex LLM)와 다중 에이전트 시스템(Multi-Agent System, MAS) 구현 복합 LLM(MAS-Implemented Multiplex LLM)으로 구성됩니다. 첫 번째 전략은 시스템 프롬프트에 복합 원리를 직접 내장해 모델의 기초적 레벨에서 출력을 다양화하는 것이며, 두 번째 전략은 각기 다른 문화적 관점을 대표하는 LLM 에이전트들이 협력하여 균형 잡힌 응답을 생성하는 것입니다. 이 연구는 LLM의 문화적 동등성을 높이기 위해 각 문화적 관점을 통합하는 방법의 중요성을 강조합니다.

- **Performance Highlights**: 문화적 편향을 완화하기 위한 전략들이 맥락 기반의 프롬프트에서 MAS 구현으로 발전함에 따라, 문화적 포용성은 크게 개선되었습니다. 연구 결과, 기본선에서는 3.25%였던 Perspectives Distribution Score (PDS)가 MAS-Implemented Multiplex LLM을 통해 98%로 증가하는 등, 문화적 다각화와 긍정적 정서 변화가 확인되었습니다. 이러한 결과는 다양한 문화적 관점을 고려한 LLM이 교육적 맥락에서 얼마나 중요한지를 보여줍니다.



### PPTAgent: Generating and Evaluating Presentations Beyond Text-to-Slides (https://arxiv.org/abs/2501.03936)
Comments:
          8 pages, 20 figures

- **What's New**: PPTAgent는 문서에서 자동으로 프레젠테이션을 생성하기 위한 혁신적인 접근 방식을 제안합니다. 기존의 방법들이 시각적 디자인과 구조적 일관성을 간과한 것을 보완하기 위해, PPTAgent는 두 단계의 편집 기반(workflow) 방식으로 프레젠테이션 생성을 개선합니다. 또한, PPTEval이라는 평가 프레임워크를 도입하여 생성된 프레젠테이션의 품질을 콘텐츠, 디자인, 일관성의 세 가지 차원에서 종합적으로 평가합니다.

- **Technical Details**: PPTAgent는 두 가지 주요 단계로 구성되어 있습니다. 첫 번째 단계에서는 참조 프레젠테이션을 분석하고, 유사한 슬라이드를 클러스터링하여 콘텐츠 스키마를 추출합니다. 이어지는 두 번째 단계에서는 입력 문서와 분석된 참조 프레젠테이션을 기반으로 적절한 슬라이드를 선택하고, 선택된 슬라이드에 기반한 상호 작용 편집 프로세스를 통해 목표 프레젠테이션을 생성합니다.

- **Performance Highlights**: PPTAgent는 기존의 자동 프레젠테이션 생성 방식에 비해 모든 차원에서 현저한 성능 향상을 보여줍니다. 실험 결과, PPTEval로 평가한 프레젠테이션의 평균 점수는 3.67이며, 다양한 도메인에서 97.8%라는 높은 성공률을 기록했습니다. 이는 PPTAgent의 접근 방식이 뛰어난 다재다능성과 강건성을 지니고 있음을 보여줍니다.



### From Newswire to Nexus: Using text-based actor embeddings and transformer networks to forecast conflict dynamics (https://arxiv.org/abs/2501.03928)
Comments:
          35 pages, 5 figures. Paper presented at the 120th American Political Science Association Annual Meeting

- **What's New**: 이번 연구는 텍스트 기반의 actor embeddings를 활용하여 violent conflict의 역동적인 변화 예측을 가능하게 하는 새로운 접근 방식을 제시합니다. 특히, 뉴스와이어 텍스트와 구조화된 conflict event 데이터를 결합하여 정부, 밀리시아, 분리주의 운동, 테러리스트 등 다양한 actor 간의 갈등 예측을 제공합니다. 이러한 접근 방법은 기존 방법들이 성취하지 못했던 폭력적 갈등의 본질적으로 변동성이 큰 패턴을 정확하고 신속하게 포착합니다.

- **Technical Details**: 이 연구에서는 국제 뉴스와이어 코퍼스를 구축하고 주석을 추가하여 Uppsala Conflict Data Program의 수동 레이블이 지정된 이벤트 데이터를 활용하였습니다. 텍스트와 구조화된 이벤트 데이터의 결합을 통해 연구팀은 뉴스 소스의 맥락을 고려한 정확하고 상세한 예측을 수행할 수 있는 하이브리드 데이터셋을 생성했습니다. 이렇게 생성된 모델은 갈등 발전에 대한 동적이고 세분화된 예측을 가능하게 합니다.

- **Performance Highlights**: 모델은 역사적 사건에 대한 엄격한 백테스팅을 통해 검증하였으며, 그 결과로 우수한 out-of-sample 예측력을 보여주었습니다. 연구팀은 전통적인 모델보다 갈등의 확산 및 축소 단계를 식별하고 예측할 수 있는 효과적인 결과를 얻었으며, 정책 입안자와 인도적 조직 및 평화 유지 작전에게 실질적인 통찰력을 제공하는 것을 목표로 하고 있습니다.



### Dolphin: Closed-loop Open-ended Auto-research through Thinking, Practice, and Feedback (https://arxiv.org/abs/2501.03916)
Comments:
          19 pages, 11 figures, and our homepage: this https URL

- **What's New**: 이 논문에서는 인공지능(AI)을 활용한 자동 과학 연구 프레임워크인 Dolphin을 제안합니다. Dolphin은 아이디어 생성, 실험 수행, 결과 피드백의 순환 구조를 구축하여 인류의 연구 과정을 자동화합니다. 이 프레임워크는 새로운 연구 아이디어를 생성하고 실험 결과를 분석하여 다음 단계로 피드백을 제공합니다.

- **Technical Details**: Dolphin은 주제와 작업 속성에 따라 관련 논문을 기반으로 새로운 아이디어를 생성합니다. 자동 생성된 코드 및 실험 계획을 바탕으로 디버깅을 수행하며, 실험 결과를 분석하여 더 높은 품질의 아이디어를 생성하는 데 기여합니다. 이를 통해 2D 이미지 분류 및 3D 포인트 분류와 같은 작업에서 최신 기술과 경쟁 가능한 방법을 제시합니다.

- **Performance Highlights**: Dolphin은 다양한 주제와 벤치마크 데이터셋에서 실험을 수행하며 지속적으로 새로운 아이디어를 생성할 수 있습니다. 실험 결과는 Dolphin이 기존의 방법들과 비교할 때 유의미한 성과를 거두고 있음을 보여줍니다. 특히, 제안된 닫힌 루프 설계를 통해 아이디어의 품질이 개선되는 것을 확인하여, 자동 연구의 효과성을 입증하고 있습니다.



### LLaVA-Mini: Efficient Image and Video Large Multimodal Models with One Vision Token (https://arxiv.org/abs/2501.03895)
Comments:
          Code: this https URL Model: this https URL

- **What's New**: 본 논문은 LMMs (Large Multimodal Models)의 효율성을 향상시킨 LLaVA-Mini를 소개합니다. 기존의 LMM 모델들이 비디오 및 이미지 이해에서 큰 연산 비용을 처리하는 것과 달리, LLaVA-Mini는 비전 토큰의 수를 최소화하여 연산 복잡도를 줄입니다. 특히, LLaVA-Mini는 단 하나의 비전 토큰만을 사용하면서도 높은 성능을 유지하는 혁신적인 접근 방식을 선택했습니다.

- **Technical Details**: LLaVA-Mini는 비전 데이터와 텍스트 정보를 사전에 융합하는 모듈을 도입하여, LLM (Large Language Model)으로 입력되는 비전 토큰의 수를 극도로 압축합니다. 이 모델은 실험을 통해 1개의 비전 토큰을 사용하여 576개 비전 토큰 대비 0.17%의 압축 비율을 보여주며, FLOPs를 77% 줄이고 GPU 메모리 사용량을 0.6MB로 낮춥니다. 이는 고해상도 이미지와 긴 비디오 처리에서 지연 시간을 줄이는 데 크게 기여합니다.

- **Performance Highlights**: LLaVA-Mini는 11개 이미지 기반 및 7개 비디오 기반 벤치마크에서 실험을 통해 LLaVA-v1.5보다 우수한 성능을 발휘하였습니다. 특히, 이미지 이해의 지연 시간을 100ms에서 40ms로 단축시킬 수 있었으며, 또한 24GB의 메모리를 가진 NVIDIA RTX 3090에서 10,000 프레임 이상의 긴 비디오를 처리할 수 있는 가능성을 보여주었습니다. 이러한 효율성을 통해 LLaVA-Mini는 실시간 다중 모달 상호작용의 새로운 길을 열었습니다.



### BERTopic for Topic Modeling of Hindi Short Texts: A Comparative Study (https://arxiv.org/abs/2501.03843)
Comments:
          Accepted into IndoNLP: The First Workshop on Natural Language Processing for Indo-Aryan and Dravidian Languages, collocated with COLING 2025. Set to appear in the workshop proceedings published in ACL Anthology

- **What's New**: 이번 연구는 힌디어( Hindi)와 같은 모국어로 작성된 짧은 텍스트 데이터에 대한 주제 모델링(topic modeling) 방법의 중요성을 강조합니다. 특히 BERTopic이 이러한 짧은 텍스트를 모델링하는 데 얼마나 효과적인지를 분석하고 있습니다. 이는 기존 연구에서 충분히 다루어지지 않았던 영역으로, 최신 미디어에서의 필요성이 증가하고 있습니다.

- **Technical Details**: 연구에서는 6가지 다양한 문서 임베딩(document embedding) 모델을 사용하여 BERTopic을 평가하고, Latent Dirichlet Allocation (LDA), Non-negative Matrix Factorization (NMF), Latent Semantic Indexing (LSI) 등 8가지 기존의 주제 모델링 기법과 성능을 비교합니다. BERTopic은 문맥 임베딩(contextual embeddings)을 활용하여 데이터 내에서 의미적 관계를 포착하며, 전통적인 모델보다 특히 짧고 다양한 텍스트에 대해 효과적일 수 있습니다.

- **Performance Highlights**: 실험 결과, BERTopic은 짧은 힌디어 텍스트로부터 일관된 주제를 포착하는 데 있어 다른 모델들보다 일관되게 우수한 성능을 보여주었습니다. 다양한 주제 수에 대한 일관성 점수(coherence scores)를 통해 평가한 결과, BERTopic이 다른 방법들을 지속적으로 능가하는 것으로 나타났습니다.



### Detecting the Undetectable: Assessing the Efficacy of Current Spoof Detection Methods Against Seamless Speech Edits (https://arxiv.org/abs/2501.03805)
Comments:
          SLT 2024

- **What's New**: 이 논문에서는 신경망 음성 편집 기술의 발전이 스푸핑 공격에 대한 우려를 초래하고 있음을 강조합니다. 기존의 음성 데이터 집합은 주로 컷 앤 페이스트(cut-and-paste) 편집에 집중하고 있지만, 이러한 방법은 감지 가능성이 높은 단점을 가지고 있어 보다 진화된 스푸핑 탐지 연구가 필요하다고 주장합니다. 이를 해결하기 위해, 저자들은 Voicebox를 기반으로 한 Speech INfilling Edit (SINE) 데이터 세트를 소개하며, 이 새로운 기술이 기존의 편집 방법보다 감지하기 어려운 음성을 생성한다고 설명합니다.

- **Technical Details**: 음성 편집 방법은 크게 컷 앤 페이스트(CaP)와 매끄러운 음성 편집의 두 가지로 분류됩니다. A3T와 Voicebox 같은 최신 모델은 주어진 텍스트와 주변 오디오에 근거하여 음성을 생성하는 '음성 인필링(speech infilling)' 모델로, 자연스러운 음성 편집을 가능하게 합니다. 이 논문에서는 저자들이 Voicebox 모델을 재구현하고 SINE 데이터 세트를 생성하기 위한 훈련 과정을 상세히 설명하며, 편집 후의 음질 평가도 포함하고 있습니다.

- **Performance Highlights**: 실험 결과, Self-supervised learning (SSL) 기반 탐지기는 다양한 음성 편집 방법에 대해 뛰어난 성능을 발휘함을 보여줍니다. 두 가지 종류의 편집 음성(음성 인필링 및 컷 앤 페이스트)을 포함한 SINE 데이터 세트에서, 기존의 상위 탐지 모델들이 매끄러운 음성 편집 탐지에 얼마나 효과적인지에 대한 분석이 이루어졌습니다. 연구자들은 SINE 데이터와 탐지기 모델을 공개하여 향후 반 스푸핑 연구에 기여할 계획입니다.



### How to Select Pre-Trained Code Models for Reuse? A Learning Perspectiv (https://arxiv.org/abs/2501.03783)
Comments:
          Accepted by IEEE SANER 2025

- **What's New**: 최근 Pre-trained Code Models (PCMs)인 CodeBERT, CodeT5, CodeGen 및 Code Llama와 같은 모델의 중요성이 높아지고 있습니다. 이러한 모델들은 코드 세대, 코드 요약 및 취약성 탐지와 같은 다양한 코드 인텔리전스 작업에서 뛰어난 성능을 달성해왔습니다. 그러나 이러한 대규모 코드 코퍼스에서 언어 모델을 사전 훈련하는 것은 계산 비용이 상당히 높기 때문에, 효율적인 모델 선택 방법이 필요합니다.

- **Technical Details**: 본 논문에서는 PCMs의 재사용성을 체계적으로 조사하고, 모델 선택을 위한 세 가지 직관적인 방법을 제안합니다: 모델 크기, 훈련 데이터 및 브루트포스 파인 튜닝. 결과적으로, 이러한 방법들이 비용이 높거나 성능이 낮음을 발견하였고, 이를 해결하기 위해 파라미터를 변경하지 않고도 사용할 수 있는 학습 기반 선택 전략을 탐구합니다.

- **Performance Highlights**: 실험 결과, 학습 기반 선택 방법은 모델 선택 시간을 2,700시간에서 100초로 단축시키며, 6% 미만의 성능 저하로 관련 작업을 수행할 수 있음을 보여주었습니다. 여러 머신러닝 전략을 활용해 모델 선택 과정을 크게 개선할 수 있음을 입증하였으며, 이 연구는 개발자들이 PCMs를 효율적으로 활용하는 데 도움을 줄 수 있는 근거를 제공합니다.



### Context-Alignment: Activating and Enhancing LLM Capabilities in Time Series (https://arxiv.org/abs/2501.03747)
Comments:
          no comment

- **What's New**: 최근 많은 주목을 받고 있는 연구는 시간 시계열(SS) 작업에 사전 훈련된 대형 언어 모델(LLMs)을 활용하는 것이다. 본 논문에서는 LLM의 자연어 처리(NLP) 능력을 활성화시키기 위해 'Context-Alignment'라는 새로운 패러다임을 제안한다. 이는 시간 시계열 데이터와 언어적 요소를 정렬함으로써 LLM이 데이터를 맥락적으로 이해할 수 있도록 한다.

- **Technical Details**: 제안된 방법은 'Dual-Scale Context-Alignment Graph Neural Networks (DSCA-GNNs)'를 기반으로 하며, 구조 정렬(structural alignment)과 논리 정렬(logical alignment)을 포함한다. 구조 정렬은 이중 스케일 노드를 사용해 시간 시계열-언어의 계층적 구조를 설명하고, 논리 정렬은 지향적 그래프 엣지를 통해 데이터와 언어 프롬프트 간의 관계를 안내한다. 이는 LLM이 긴 시계열 데이터를 일관된 언어 구성 요소로 이해할 수 있도록 만든다.

- **Performance Highlights**: DECA(예제 기반 Context-Alignment)는 DSCA-GNNs 프레임워크에 따라 LLM의 TS 작업 성능을 크게 향상시키며, 특히 몇 샷(few-shot) 및 제로 샷(zero-shot) 예측에서 기존 방법보다 뛰어난 결과를 보여준다. 실험 결과는 Context-Alignment의 중요성을 강조하며, 다양한 데이터셋 및 TS 작업에서 효율성을 입증하였다. 또한 이 연구는 LLM의 잠재적 능력을 활성화하고 성능을 향상시키기 위해서는 효과적인 방법론이 필요함을 강조한다.



### LlaMADRS: Prompting Large Language Models for Interview-Based Depression Assessmen (https://arxiv.org/abs/2501.03624)
- **What's New**: 본 연구에서는 LlaMADRS라는 새로운 프레임워크를 소개합니다. 이는 오픈소스 대형 언어 모델(Large Language Models, LLMs)을 활용하여 Montgomery-Asberg Depression Rating Scale (MADRS)를 이용한 우울증 강도 평가를 자동화하는 방법을 제시합니다. 이 접근법은 병원 상황에서의 정신 건강 평가를 보다 효율적으로 만들 수 있는 가능성을 보여줍니다.

- **Technical Details**: LlaMADRS는 정교하게 설계된 프롬프트(prompt)를 사용하여 모델이 임상 인터뷰를 해석하고 점수를 매기도록 안내하는 제로샷(prompting) 전략을 적용합니다. 236개의 실제 임상 인터뷰를 포함하는 Context-Adaptive Multimodal Informatics (CAMI) 데이터셋에서 테스트한 결과, 의사의 평가와의 강한 상관관계를 나타냈습니다. 특히 Qwen 2.5--72b 모델은 대부분의 MADRS 항목에서 근접한 인간 수준의 동의를 기록했습니다.

- **Performance Highlights**: 이 연구는 LLMs를 활용하여 합리적으로 우울증을 평가할 수 있다는 가능성을 보여 줍니다. 그러나 비언어적 신호에 의존하는 항목의 평가에서는 여전히 도전과제가 남아 있습니다. 이러한 결과는 정신 건강 평가의 접근성을 높이는 데 기여할 수 있으며, 제한된 자원에서의 활용 가능성을 시사합니다.



### Discriminative Representation learning via Attention-Enhanced Contrastive Learning for Short Text Clustering (https://arxiv.org/abs/2501.03584)
- **What's New**: 단기 텍스트 클러스터링(short text clustering)에서의 대조 학습(contrastive learning)을 통한 새로운 접근법이 제안되었습니다. 이 방법은 Attention-Enhanced Contrastive Learning(AECL)이라는 프레임워크를 통해 샘플 간의 유사성을 파악하고, 잘못된 부정 분리(false negative separation) 문제를 해결하는 데 중점을 두고 있습니다. AECL은 의사 레이블(pseudo-label) 생성 모듈과 대조 학습 모듈을 결합하여 더욱 구별력이 뛰어난 표현을 생성합니다.

- **Technical Details**: AECL은 샘플 수준의 Attention 메커니즘을 통해 샘플 간의 유사 관계를 캡처하고, 이는 각 샘플의 고유한 특징과 교차 샘플 정보(consistency representations)를 통합하여 일관된 표현을 생성합니다. 두 개의 모듈은 네트워크 파라미터를 공유하며, Pseudo-label Generation Module(PGM)은 신뢰할 수 있는 의사 레이블을 생성하고, Contrastive Learning Module(CLM)은 이를 활용하여 대조 학습을 수행합니다. 이를 통해 AECL은 긍정 샘플의 구성을 최적화하고 클러스터링 성능을 향상시킵니다.

- **Performance Highlights**: 실험 결과, AECL은 8개의 벤치마크 데이터셋에서 기존의 최첨단 방법들을 능가하는 성능을 보였습니다. AECL은 같은 클래스를 가진 샘플을 긍정 쌍으로 간주하여 잘못된 부정 분리 문제를 효과적으로 해결할 수 있음을 입증하였습니다. 이 성능 향상은 특히 클러스터의 내부 응집력(intra-cluster cohesion)과 외부 분리(inter-cluster separation)를 증진시키는 데 기여합니다.



### From Code to Compliance: Assessing ChatGPT's Utility in Designing an Accessible Webpage -- A Case Study (https://arxiv.org/abs/2501.03572)
- **What's New**: 이번 연구에서는 ChatGPT(GPT-4o)의 웹 접근성 관련 능력을 평가했습니다. 일반적으로 웹사이트의 접근성과 관련하여 96%가 기준을 충족하지 않는다는 점에서, ChatGPT가 생성하는 코드가 얼마나 준수하는지를 분석했습니다. 더욱이, 효과적인 프롬프트 엔지니어링과 시각적 요소를 통합함으로써 접근성 문제 해결 능력을 향상시킬 수 있음을 발견했습니다.

- **Technical Details**: 연구는 TV 시리즈 웹페이지를 선택하여 다양한 웹 디자인 요소를 포함한 접근성 평가를 수행했습니다. 자동화 도구인 WAVE와 Axe를 사용해 웹페이지에서 발생한 접근성 문제를 분석하고, 수작업 검사도 병행하여 질적 데이터를 수집했습니다. ChatGPT의 문제 해결 과정에서 필요했던 피드백의 수와 문제의 복잡성을 반영한 가중 평균을 사용하여 효율성을 측정했습니다.

- **Performance Highlights**: 연구 결과, ChatGPT는 간단한 접근성 문제를 해결하는 데 강점을 보였으나 복잡한 문제에는 인간의 감독과 추가적인 반복 작업이 필요하다는 한계도 드러났습니다. 스크린샷을 제공함으로써 ChatGPT가 문제를 더 명확하게 인식하고 해결할 수 있는 가능성을 제시했습니다. 이러한 결과는 웹 개발자에게 더 포괄적인 웹사이트 설계를 위한 실용적인 지침을 제공합니다.



### Strategic Fusion Optimizes Transformer Compression (https://arxiv.org/abs/2501.03273)
Comments:
          15 pages, 1 table, 8 figures; will be submitted to ICML 2025; codes will be made public after acceptance

- **What's New**: 본 연구는 transformer 모델 압축을 위한 층 가지치기(pruning) 방법론을 체계적으로 탐구했습니다. 9개의 다양한 데이터셋에서 14개의 가지치기 전략을 평가하며, 특히 활성화(activation), 상호 정보(mutual information), 기울기(gradient), 가중치(weights), 주의(attention)에서 얻은 신호를 기반으로 한 12가지 전략을 분석했습니다. 또한, 단일 신호 전략의 한계를 극복하기 위해 선형 회귀(linear regression)와 랜덤 포레스트(random forest)라는 두 가지 융합 전략을 도입하여 보다 정보에 기반한 가지치기 결정을 내리는 방법을 제시합니다.

- **Technical Details**: 지금까지의 연구는 모델 압축을 위한 다양한 접근 방식을 탐색해왔으며, 여기에는 중요하지 않은 가중치를 가지치기하는 것, 매개변수를 양자화하는 것, 그리고 지식 증류(knowledge distillation)가 포함됩니다. 본 연구에서는 BERT 모델을 사용하여 활성화 기반, 기울기 기반 등 12개 개별 가지치기 전략을 분석하고, 각 전략의 선택을 위한 수학적 및 생물학적 직관을 제공했습니다. 실험을 통해 14개의 가지치기 전략이 9개의 데이터셋에서 평가되었으며, 융합 전략이 단일 지표 접근 방식보다 우수한 성능을 나타냈습니다.

- **Performance Highlights**: 랜덤 포레스트 기반 융합 전략은 9개 데이터셋 중 7개에서 최상의 성능을 달성했으며, 나머지 2개 데이터셋에서도 뛰어난 성과를 보였습니다. 지식 증류를 적용한 결과, 6개 데이터셋에서 원래의 정확도를 초과했고, 나머지 3개 데이터셋에서도 정확도 감소를 완화했습니다. 전체 데이터셋에 걸쳐 지식 증류 이후 정확도 대비 크기 비율이 평균 18.84배 증가하는 등의 성과를 보였습니다.



### Backdoor Token Unlearning: Exposing and Defending Backdoors in Pretrained Language Models (https://arxiv.org/abs/2501.03272)
Comments:
          AAAI 2025

- **What's New**: 이번 연구에서는 대규모 사전 학습 모델의 미세 조정(fine-tuning) 과정 중 발생하는 백도어 공격(backdoor attack)에 대한 효율적인 방어 방법을 제안합니다. 제안된 방법인 Backdoor Token Unlearning (BTU)은 훈련 단계에서 트리거 토큰(trigger token)을 사전에 탐지하고 중화하는 방식을 채택하여, 백도어 공격의 영향을 최소화합니다. 본 연구는 두 가지 주요 발견에 기반하여, 백도어 토큰 파라미터(backdoor token parameters)와 클린 토큰 파라미터(clean token parameters) 간의 독특한 차이를 활용합니다.

- **Technical Details**: BTU는 두 단계로 구성되어 있습니다. 첫 번째 단계에서는 워드 임베딩 레이어(word embedding layer)를 교육하여 백도어와 관련된 임베딩 파라미터를 식별하고, 두 번째 단계에서는 영향을 받은 백도어 임베딩 파라미터를 무해한 패딩 토큰(padding token) 임베딩으로 교체하여 백도어 정보를 제거합니다. 이 과정은 세 가지 데이터셋과 네 가지 유형의 백도어 공격을 통해 검증되었으며, BTU는 모델의 기본 작업 성능을 유지하면서 효과적인 방어를 제공합니다.

- **Performance Highlights**: BTU 방법은 백도어 공격의 성공률을 상당히 감소시킵니다. 세 가지 데이터셋과 네 가지 백도어 공격을 통해 수행된 실험 결과, BTU는 모델의 성능을 최소한으로 저하시키면서도 공격에 대한 저항력을 효과적으로 입증하였습니다. 본 연구의 코드와 자료는 제공된 링크에서 확인 가능하며, 향후 AI 모델의 안전성을 높이는 데 기여할 것으로 예상됩니다.



### A Semantically-Aware, Kernel-Enhanced, and Divergence-Rich Paradigm for Direct Preference Optimization (https://arxiv.org/abs/2501.03271)
Comments:
          -

- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)의 정렬 문제를 해결하기 위한 새로운 접근 방식인 DPO-Kernels를 제안합니다. 기존의 Direct Preference Optimization(DPO) 방법의 한계를 극복하기 위해 커널 방법을 통합하여 다양한 가치와 선호도에 맞게 조정할 수 있습니다. 주요 기여로는 폴리노미얼, RBF, Mahalanobis 및 스펙트럴 커널을 이용한 커널화 표현을 포함합니다.

- **Technical Details**: DPO-Kernels는 커널화된 표현을 통해 더 풍부한 변환을 제공하며, 이는 하이브리드 손실(hybrid loss) 기법에 의해 지원됩니다. 또한, 여러 다이버전스(혹은 발산) 대안, 즉 Jensen-Shannon, Hellinger, Renyi, Bhattacharyya, Wasserstein, f-divergences 등을 사용하여 안정성을 강化합니다. 데이터 기반 선택 메트릭을 통해 최적의 커널-다이버전스 쌍을 자동으로 선택할 수 있으며, 계층적 혼합 커널(Hierarchical Mixture of Kernels)를 통해 지역 정밀성과 전역 모델링을 제공합니다.

- **Performance Highlights**: 12개 데이터셋에 대한 평가에서 해당 모델은 사실성(factuality), 안전성(safety), 추론(reasoning), 지시 준수(instruction following) 측면에서 최첨단 성능을 보였습니다. Heavy-Tailed Self-Regularization에 기반하여 DPO-Kernels는 LLMs의 강력한 일반화 능력을 유지하며, 향후 정렬 연구에 필요한 포괄적인 자원을 제공합니다.



### Breaking Through the Spike: Spike Window Decoding for Accelerated and Precise Automatic Speech Recognition (https://arxiv.org/abs/2501.03257)
Comments:
          Accepted by ICASSP 2025

- **What's New**: 본 논문에서는 CTC (Connectionist Temporal Classification) 출력의 spike 특성을 조사하고, 비공백 프레임과 근접한 이웃 프레임이 모델에 유익한 의미 정보를 가지고 있다는 가설을 제안합니다. 이를 바탕으로 Spike Window Decoding (SWD) 알고리즘을 제안하며, 이는 WFST (Weighted Finite-State Transducer)에서 디코딩 되는 프레임 수를 CTC 출력의 spike 프레임 수와 선형적으로 연관짓도록 설계되었습니다. 이를 통해 디코딩 속도가 크게 향상되었습니다.

- **Technical Details**: SWD 알고리즘은 CTC posteriori 확률 행렬에서 가장 높은 확률을 가진 프레임의 인덱스를 검색하는 것으로 시작합니다. 그 후, 비공백 스파이크 프레임과 일치하는 인덱스를 찾아 혁신적인 Spike Window 함수를 활용하여 이웃 프레임의 윈도우 시퀀스를 획득합니다. 이 과정에 게임 과정은 TLG (Transducer with Language Guidance) WFST 그래프 구조에서 det와 min 사이의 가중치 푸시 전략을 사용하는 것입니다.

- **Performance Highlights**: SWD 알고리즘은 AISHELL-1 및 대규모 In-House 데이터셋에서 각각 3.89% 및 2.09%의 문자 오류율(CER)을 달성하며, 이는 기존 SOTA 접근 방식을 초월합니다. 뿐만 아니라 SWD 알고리즘은 각각의 데이터셋에 대해 베이스라인 방법보다 1.76배 및 2.17배의 디코딩 속도 향상을 보여줍니다. 이러한 결과는 다양한 규모의 데이터셋에서 우수한 인식 성능과 향상된 디코딩 속도 간의 remarkable한 균형을 이룬 것입니다.



### Bridging Auditory Perception and Language Comprehension through MEG-Driven Encoding Models (https://arxiv.org/abs/2501.03246)
Comments:
          10 pages, 4 figures, Accepted at ICLR2024 Workshop TS4H

- **What's New**: 이번 연구에서는 청각 및 언어 처리의 신경 메커니즘을 이해하기 위해 Magnetoencephalography (MEG) 데이터를 사용하여 언어 자극에 대한 뇌 반응을 분석했습니다. 우리는 오디오-대-MEG 인코더와 텍스트-대-MEG 인코더 두 가지 모델을 개발하였고, 텍스트 기반 모델이 더 높은 Pearson Correlation (PC) 점수를 달성함으로써 성능이 더 우수하다는 것을 보여주었습니다. 이러한 결과는 청각 및 텍스트 정보 처리를 위한 뇌의 뚜렷한 신경 경로 차이를 드러내며, 언어 자극에 대한 뇌의 기능적 아키텍처에 대한 통찰을 제공합니다.

- **Technical Details**: 본 연구에서는 시간-주파수 분해(time-frequency decompositions)와 wav2vec2의 잠재 공간 표현을 활용하여 오디오에 대한 MEG 반응을 예측하는 인코딩 모델을 개발하였습니다. 또한, CLIP 및 GPT-2 모델에서 추출한 임베딩을 사용하여 텍스트-대-MEG 인코더를 구축하였습니다. 이를 위해 MEG-MASC 데이터셋을 사용하고, 몇 가지 전처리 및 세분화 과정을 통해 3초 길이의 윈도우 데이터를 생성하였습니다. 이러한 다양한 인코딩 기술을 통해 청각 및 언어 인식 과정의 신경적 요소를 심층적으로 탐구합니다.

- **Performance Highlights**: 텍스트-대-MEG 모델은 오디오 기반 모델에 비해 높은 Pearson Correlation 점수를 기록하여 언어 처리에서 더 높은 인코딩 정확도를 보여주었습니다. 특히, 청각 기반 임베딩은 측두엽의 외측 부위를 주로 활성화한 반면, 텍스트 임베딩은 전두엽에서 주로 작용하였습니다. 이로 인해 청각 자극이 보다 직접적인 감각 경로를 통해 처리되는 반면, 언어 정보는 의미 통합 및 인지 제어를 위한 네트워크를 통해 부호화된다는 것을 확인할 수 있었습니다.



### BoostStep: Boosting mathematical capability of Large Language Models via improved single-step reasoning (https://arxiv.org/abs/2501.03226)
Comments:
          Codes and Data are available at this https URL

- **What's New**: 리서치에서는 BoostStep이라는 새로운 접근법을 소개하여 이론적 레벨을 문제 레벨에서 각각의 리즈닝 프로세스를 지원하는 단계 레벨로 전환합니다. 이는 관련된 ICL 예시를 각 단계에서 제공하며, 적절한 가이드라인을 보장하는 데에 중점을 둡니다. 또한, 기존의 코스 문제 기반 전략보다 훨씬 관련성 높은 예시를 제공함으로써 모델의 리즈닝 품질을 지속적으로 향상시킵니다.

- **Technical Details**: BoostStep은 리즈닝 중 단계 레벨의 적합성과 ICL 예시를 조화롭게 맞춘 후, 처음 시도(first-try) 전략을 통해 적절한 리즈닝 단계를 지원합니다. 이 방법은 단계별로 솔루션을 문제로 나누고, 각 단계에 대한 유사한 문제를 검색하여 적절한 해결 방법을 제시합니다. 이는 기존의 문제 레벨 ICL 접근법보다 더 밀접한 가이드를 제공하며, 각 단계의 정확성을 높이는 데 기여합니다.

- **Performance Highlights**: BoostStep은 다양한 수학 벤치마크에서 GPT-4o 모델 성능을 3.6% 향상시키고 Qwen2.5-Math-72B 모델은 2.0% 향상시키는 결과를 보였습니다. Monte Carlo Tree Search(MCTS)와 결합했을 때는 7.5%의 성과 향상이 있었습니다. 이는 BoostStep이 단순한 리즈닝 성능 향상을 넘어 모델의 결정 과정까지 개선하는 효과를 입증합니다.



### Leveraging Explainable AI for LLM Text Attribution: Differentiating Human-Written and Multiple LLMs-Generated Tex (https://arxiv.org/abs/2501.03212)
- **What's New**: 이 연구는 Generative AI Large Language Models (LLMs)가 생성한 텍스트를 효과적으로 검출하는 모델을 제안합니다. 최근 학생들이 LLM 도구에 지나치게 의존하면서 발생하는 표절 문제를 해결할 필요성이 커지고 있습니다. 연구는 LLM 생성 텍스트와 인간 작성 텍스트를 구별하는 기계 학습 모델의 가시성을 높이기 위해 Explainable AI (XAI) 기법을 활용합니다.

- **Technical Details**: 제안된 연구 방법론은 이진 분류와 다중 분류로 나누어져 있습니다. 이진 분류에서는 인간 작성 텍스트와 AI 생성 텍스트를 구별하고, 다중 분류에서는 다섯 가지 LLM 도구(예: ChatGPT, LLaMA, Google Bard, Claude, Perplexity)와 인간 작성을 구별합니다. Random Forest (RF), Recurrent Neural Networks (RNN) 등 다양한 기계 학습 및 딥러닝 알고리즘이 사용되며, 모델은 GPTZero보다 98.5%의 정확도로 성능이 우수합니다.

- **Performance Highlights**: 연구 결과, 제안된 모델은 다중 및 이진 분류 모두에서 높은 정확성을 보여주었으며, GPTZero와 비교하여 성능 우수성이 입증되었습니다. GPTZero는 전체 테스트 데이터셋의 약 4.2%를 인식하지 못했지만, 본 연구 모델은 모든 샘플을 성공적으로 인식했습니다. Explainable AI의 결과는 다양한 클래스에서의 특성 중요성을 이해함으로써 저자 및 출처 프로파일을 상세히 파악하는 데 도움을 주어 표절 탐지 지원에 기여합니다.



### Detecting AI-Generated Text in Educational Content: Leveraging Machine Learning and Explainable AI for Academic Integrity (https://arxiv.org/abs/2501.03203)
- **What's New**: 본 연구는 AI 생성 콘텐츠를 학생 작업에서 감지하기 위한 도구를 제공하여 학문적 무결성을 향상시키는 것을 목표로 합니다. 특히 CyberHumanAI 데이터셋을 생성하여, 인간이 작성한 콘텐츠와 ChatGPT에 의해 생성된 콘텐츠를 비교 분석하였습니다. 이 연구는 교육에서의 AI 통합을 책임감 있게 지원하고 윤리적 기준을 유지하기 위해 투명성과 책임성을 촉진합니다.

- **Technical Details**: CyberHumanAI 데이터셋은 1000개의 관측치로 구성되어 있으며, 그 중 500개는 인간이 작성하고 나머지 500개는 ChatGPT에 의해 생성되었습니다. 다양한 머신러닝(ML) 및 딥러닝(DL) 알고리즘을 평가한 결과, 전통적인 ML 알고리즘인 XGBoost와 Random Forest가 각각 83% 및 81%의 높은 정확성을 기록했습니다. 또한, 짧은 콘텐츠의 분류가 긴 콘텐츠에 비해 더 어려운 것으로 나타났습니다.

- **Performance Highlights**: 설명 가능한 인공지능(XAI)을 활용하여 ML 모델의 예측에 영향을 미치는 특징들을 식별하였습니다. 분석 결과, 인간 작성 콘텐츠는 실용적인 언어를 사용하는 경향이 있는 반면, AI 생성 텍스트는 좀 더 추상적이고 공식적인 용어로 특징 지어졌습니다. 제안된 모델은 Pure AI, Pure Human 및 혼합 클래스를 분류할 때 약 77.5%의 정확성을 기록했고, 이는 GPTZero의 48.5%보다 현저히 높은 수치입니다.



### The FACTS Grounding Leaderboard: Benchmarking LLMs' Ability to Ground Responses to Long-Form Inpu (https://arxiv.org/abs/2501.03200)
- **What's New**: FACTS Grounding이라는 온라인 리더보드와 벤치마크가 도입되었습니다. 이 시스템은 언어 모델이 사용자 요청에 대한 맥락에 따라 사실적으로 정확한 텍스트를 생성할 수 있는 능력을 평가합니다. 각 프롬프트는 최대 32k 토큰 길이의 전체 문서를 포함하며, 긴 형식의 응답이 요구됩니다. 모델의 성과는 자동화된 평가 모델을 통해 두 가지 단계로 평가됩니다.

- **Technical Details**: 이 연구에서는 언어 모델의 사실성(factuality)을 측정하는 벤치마크를 제안합니다. 모델은 주어진 문서에서 파생된 정보를 합성하면서 사용자 요청을 직접 다루어야 하며, 이로 인해 사실성 측정이 필요합니다. 연구의 다양한 데이터 세트는 다양한 문서 길이(최대 32k 토큰)와 기업 도메인을 포함하며, 사용자 요청에 응답하는 길고 복잡한 출력이 필요합니다.

- **Performance Highlights**: 이 리더보드는 860개의 공개 예제와 859개의 비공식 예제를 포함하고 있으며, 다양한 LLM 성능을 자동화된 사실성 점수를 통해 보고합니다. 모델이 문서 맥락에 기반하여 긴 형식의 응답을 생성하는 능력을 측정하며, 결과는 성과 보고서에서 제공됩니다. 이 벤치마크와 리더보드는 시간이 지나도 활발하게 유지되고 업데이트되어 새로운 모델과 변종을 포함시킬 예정입니다.



### CLIX: Cross-Lingual Explanations of Idiomatic Expressions (https://arxiv.org/abs/2501.03191)
- **What's New**: 본 논문에서는 언어 학습자를 위한 사전적 정의 생성 시스템의 발전을 지원하기 위해 새로운 작업인 Cross-Lingual explanations of Idiomatic eXpressions (CLIX)를 제안합니다. 이는 언어 학습자가 비유 표현의 의미를 이해하는 데 도움을 주기 위해 설계되었습니다. 기존 사전적 정의 생성의 한계를 극복하고, 비표준 언어의 복잡성에 대응할 수 있는 새로운 접근 방식을 제공합니다.

- **Technical Details**: CLIX 과제에서는 주어진 비유적 표현에 대해 지정된 목표 언어로 설명을 생성합니다. 데이터셋은 628개의 영어 숙어와 이들에 대한 영어, 스페인어, 독일어 설명으로 구성됩니다. 우리는 사전 훈련된 시퀀스-투-시퀀스 모델과 대형 언어 모델(LLMs)의 성능을 탐구하며, 이 작업이 기존의 정의 생성 작업보다 훨씬 어려운 것을 보여줍니다.

- **Performance Highlights**: 자동화된 측정 기준은 다소 부정적인 그림을 그리지만, 몇 가지 샘플의 대형 언어 모델의 결과는 목표 언어의 원어민들에 의해 긍정적으로 평가됩니다. 또한, 우리는 이 연구에서 발견된 오류 분석을 통해 교육 애플리케이션에서 자동화된 교차 언어 설명을 신뢰할 수 있게 사용하기 위해 개선해야 할 주요 영역을 강조합니다.



### Classifier-Guided Captioning Across Modalities (https://arxiv.org/abs/2501.03183)
- **What's New**: 최근 캡션 생성 시스템은 특정 데이터 세트에서 훈련된 언어 모델을 사용하여 일반화에 제약을 받고 있습니다. 특히 오디오 또는 비디오 캡션 생성에는 다른 의미적 단서가 필요하며, 이를 해결하기 위해 새로운 프레임워크를 제안합니다. 이 프레임워크는 전이학습 없이 추론 단계에서만 작동하며, 기존 캡션 생성 모델의 품질을 크게 향상시킵니다.

- **Technical Details**: 이 연구에서는 오디오 캡션을 생성하는 과정에서 오디오의 가청성을 나타내기 위해 텍스트 분류기를 활용한 새로운 방법론을 도입합니다. 이 프레임워크는 사전 훈련된 캡셔닝 시스템과 가청성 분류기로 구성되어 있으며, 두 가지 손실 함수를 통해 성능을 최적화합니다. 이 방식은 다양한 모달리티에 쉽게 통합할 수 있어 전반적인 유연성을 제공합니다.

- **Performance Highlights**: 제안된 방법은 AudioCaps 및 Clotho 데이터 세트에서 실시된 실험에서 기존 모델보다 성능 향상을 보였습니다. 특히 기존 제로샷 오디오 캡션 시스템과 결합하면 품질이 크게 개선되며 제로샷 오디오 캡션에서 최첨단 성능을 달성했습니다. 이러한 결과는 다양한 실시간 환경에서도 효과적인 캡션 생성이 가능함을 시사합니다.



### Boosting Explainability through Selective Rationalization in Pre-trained Language Models (https://arxiv.org/abs/2501.03182)
Comments:
          KDD 2025 research track

- **What's New**: 본 논문은 사전 훈련된 언어 모델(Pre-trained Language Models, PLMs)이 기존의 선택적 합리화(selective rationalization) 기법에서 심각한 퇴화(degeneration) 및 실패(failure) 문제를 겪는다는 사실을 밝히고, 이를 해결하기 위한 PLMR(Pre-trained Language Model's Rationalization)라는 새로운 방법론을 제안합니다.

- **Technical Details**: PLMR은 PLMs를 두 개의 독립적인 부분, 즉 합리화 생성기(generator)와 예측기(predictor)로 나누어 NLP 과제를 수행하며 해석 가능한 합리화를 제공합니다. 이 방법은 관련 없는 토큰을 잘라내어 토큰의 동질성(homogeneity)을 완화하고, 예측기 부분에서는 전체 텍스트 정보를 활용하여 예측을 표준화합니다.

- **Performance Highlights**: 실험 결과, PLMR은 GRU를 사용하는 방법에 비해 최대 9%, 기존 PLMs 기반 방법에 비해 최대 17% 높은 F1 점수를 기록하여 PLM을 사용한 합리화에서 퇴화 및 실패 문제를 효과적으로 해결할 수 있음을 보여줍니다.



### GLiREL -- Generalist Model for Zero-Shot Relation Extraction (https://arxiv.org/abs/2501.03172)
Comments:
          Submitted to NAACL 2025

- **What's New**: 본 논문에서는 제로샷 관계 추출(zero-shot Relation Extraction)을 위한 효율적인 아키텍처이자 훈련 패러다임인 GLiREL(Generalist Lightweight model for zero-shot Relation Extraction)을 소개합니다. GLiREL은 여러 엔티티 사이에서 관계 레이블을 단일 포워드 패스(single forward pass)로 정확하게 예측할 수 있도록 설계되었습니다. 실험 결과, FewRel과 WikiZSL 벤치마크에서 저희 접근 방식이 제로샷 관계 분류(task)에서 최신 기술(State-of-the-Art) 결과를 달성하였음을 보여줍니다. 또한 다양한 관계 레이블을 사용할 수 있는 데이터셋을 합성적으로 생성하는 프로토콜도 기여하였습니다.

- **Technical Details**: GLiREL 아키텍처는 세 가지 주요 구성 요소로 나누어집니다: 1) 텍스트 인코더로서의 사전 훈련된 양방향 언어 모델, 2) 엔티티 쌍 표현 모듈, 3) 점수 계산 모듈입니다. 이 아키텍처는 관계 레이블과 엔티티 쌍의 임베딩을 동일한 잠재 공간(latent space)에서 인코딩하여 유사성을 계산합니다. 우리는 하위 작업에서 우수한 성능을 보이는 DeBERTa V3-large를 인코더 모델로 선택하였습니다.

- **Performance Highlights**: GLiREL은 기존의 다른 제로샷 관계 분류 모델들보다 더 효율적이며, 여러 엔티티 쌍을 단일 입력으로 분류할 수 있는 장점을 가지고 있습니다. 랜덤한 관계 레이블 로드리딩 없이 임베딩을 처리할 수 있어, 모델이 동시에 여러 레이블과 엔티티 쌍 간의 상호작용을 포착합니다. 이러한 효율성 덕분에 실제 시나리오에서 많은 엔티티를 포함하는 경우에서도 뛰어난 성능을 발휘합니다.



### Semantic Captioning: Benchmark Dataset and Graph-Aware Few-Shot In-Context Learning for SQL2Tex (https://arxiv.org/abs/2501.03166)
Comments:
          Accepted to COLING'25

- **What's New**: 이번 연구에서는 SQL 쿼리 (SQL2Text)를 캡셔닝하는 새로운 작업인 "semantic captioning"에 초점을 맞췄습니다. 자연어에서 코드로의 변환 정기적인 작업에서 코드에서 자연어로의 변환의 중요성이 높아지고 있는데, 이는 LLM이 코드 생성 및 보안 분석 플랫폼에 통합되면서 더욱 두드러집니다. 연구팀은 기존 Text2SQL 데이터셋을 활용하여 SQL 쿼리에 대한 이해를 높이고 SQL2Text의 필요성을 부각시키고자 했습니다.

- **Technical Details**: 데이터셋 생성에는 GPT-4o와 검증된 Text2SQL 코퍼스를 사용하여 세 가지 벤치마크 데이터셋인 CoSQL, Spider, SParC의 새로운 데이터셋을 반복적인 ICL 프롬프트를 통해 생성했습니다. 연구에서는 SQL의 그래프 속성을 활용하여 ICL 샘플 선택을 개선하는 방법과 이를 통해 작은 LLM과 큰 LLM의 성능 차이를 분석했습니다. 평가 지표로는 BLEU 점수, BERTScore 두 가지 변형, AlignScore를 활용하여 생성된 의미 레이블의 품질을 측정했습니다.

- **Performance Highlights**: 연구 결과, SQL의 그래프 특성을 활용한 ICL 샘플 선택이 랜덤 선택보다 최대 39% 더 우수한 성능을 나타냈습니다. 또한, 작은 LLM을 사용하는 것이 일반적으로 더 큰 LLM보다 성능을 향상시키는 것으로 밝혀졌습니다. 이는 SQL 쿼리에 대한 데이터 설명 생성 및 코드 보안 증진과 같은 다양한 기술 및 비즈니스 역할에서 중요한 통찰력을 제공합니다.



### VicSim: Enhancing Victim Simulation with Emotional and Linguistic Fidelity (https://arxiv.org/abs/2501.03139)
Comments:
          21 pages, 10 figures

- **What's New**: 이 논문에서는 VicSim(victim simulator)이라는 새로운 모델을 소개합니다. 이 모델은 시나리오 기반 훈련을 위해 피해자를 시뮬레이션하는 데 필요한 정보적 정확성, 감정적 역동성, 그리고 언어 스타일을 다루는 세 가지 주요 차원을 강조합니다. 특히 GAN 기반의 훈련 작업 흐름과 주요 정보 기반의 프롬프트를 통합하여, 시뮬레이션된 피해자의 사실성을 향상시키는 것을 목표로 합니다. 평가 결과, VicSim 모델은 인간과의 유사성(human-likeness)에서 GPT-4를 초월합니다.

- **Technical Details**: VicSim은 Llama 2-7B 기반의 채팅 모델 위에서 개발되었으며, 정보와 감정의 신뢰성을 높이기 위한 여러 가지 개선 사항을 포함합니다. 이 방법론은 사용자 응답의 진정성을 높이기 위한 다각적인 접근 방식을 포함하고 있으며, 감정적 신호와 문법적 요소를 인식하는 적대적 훈련(adversarial training) 방식을 채택합니다. 이를 통해 훈련자의 고유한 요구를 반영하여 시스템을 적응시키는 사용자 모델링(user modeling)이 가능해집니다.

- **Performance Highlights**: VicSim 모델은 실제 사건 보고를 처리하는 훈련 담당자(dispatchers)의 훈련을 위한 가치 있는 도구로 작용할 수 있습니다. 이 모델은 훈련생들이 실제와 유사한 시나리오에서 시뮬레이션된 사용자와 대화하며 배울 수 있도록 설계되었습니다. 향상된 정보적 및 감정적 충실도를 통해 시나리오 기반 교육이 더 깊이 있는 상호작용을 촉진합니다.



### PRMBench: A Fine-grained and Challenging Benchmark for Process-Level Reward Models (https://arxiv.org/abs/2501.03124)
Comments:
          Project Page: this https URL

- **What's New**: 본 논문에서는 복잡한 추론 및 의사결정 작업에서 중요한 역할을 하는 Process-level Reward Models (PRMs)에 대한 새로운 벤치마크인 PRMBench를 소개합니다. PRMBench는 PRMs의 미세 조정된 오류 감지 능력을 평가하기 위해 설계되었으며, 6,216개의 문제와 83,456개의 단계 수준 레이블을 포함하고 있어 다양한 차원에서 모델을 평가합니다.

- **Technical Details**: PRMBench는 PRMs의 성능을 간단함(simplicity), 타당성(soundness), 민감도(sensitivity) 등 여러 차원에서 평가합니다. 이를 통해 현재 PRMs가 직면한 다양한 오류 유형을 탐지하는 데 필요한 능력을 정교하게 측정할 수 있습니다. 15개의 모델에 대한 실험을 통해 우리는 PRMs의 중대한 약점을 발견하였습니다.

- **Performance Highlights**: 현재 PRMs의 성능을 체계적으로 평가하는 것이 부족하다는 사실이 발견되었으며, 이는 프로세스 수준 평가에 내재된 도전 과제를 강조합니다. 이러한 연구 결과는 PRM 평가 및 개발에 있어 중요한 미래 연구 방향을 제시합니다. PRMBench가 PRM 연구의 발전을 위한 강력한 벤치마크가 되기를 바랍니다.



### LangFair: A Python Package for Assessing Bias and Fairness in Large Language Model Use Cases (https://arxiv.org/abs/2501.03112)
Comments:
          Journal of Open Source Software; LangFair repository: this https URL

- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 편향성을 평가하고 공정성 리스크를 측정하기 위해 LangFair라는 오픈소스 Python 패키지를 소개합니다. 이 패키지는 LLM 사용자들이 자신들의 특정 응용 사례에 맞춘 평가 데이터셋을 쉽게 생성하고, 관련 메트릭을 계산할 수 있도록 설계되었습니다. 또한, LangFair는 메트릭 선택에 도움을 줄 수 있는 실행 가능한 결정 프레임워크를 제공합니다.

- **Technical Details**: LangFair는 사용자가 제공한 프롬프트를 기반으로 LLM 응답에서 편향 및 공정성을 평가할 수 있는 도구입니다. 기존의 평가 도구들이 정적 기준 데이터셋에 기반해 LLM을 평가하는 반면, LangFair는 프롬프트 특화 리스크를 고려하여 개인화된 평가를 가능하게 합니다. 이 패키지에는 응답 생성 클래스인 ResponseGenerator와 CounterfactualGenerator가 포함되어 있어 사용자의 요구에 맞게 평가 데이터셋을 생성할 수 있습니다.

- **Performance Highlights**: LangFair는 실제 LLM 기반 시스템의 요구를 충족시키며, 사용자 제공 프롬프트에 의존하여 LLM 응답에서 메트릭을 계산합니다. 이를 통해 출력 기반 메트릭이 보다 신뢰성이 높고 실제 사용 사례에서 더 유용할 수 있음을 입증합니다. LangFair의 다양한 메트릭 클래스는 독립적인 유용성과 함께 특별한 태스크에 맞춘 평가를 지원하여 LLM의 편향성을 체계적으로 분석하도록 돕습니다.



### Sentiment-guided Commonsense-aware Response Generation for Mental Health Counseling (https://arxiv.org/abs/2501.03088)
- **What's New**: 이번 연구에서는 EmpRes라는 새로운 감정 기반의 일반 상식 인지 메커니즘을 제안합니다. 이는 가상의 정신 건강 보조인(VMHA)이 고객의 정서를 파악하고 긍정적으로 이끌어낼 수 있도록 하는 기능을 가지고 있습니다. EmpRes는 기존 VMHA들이 가지고 있는 한계를 극복하여, 감정 인지와 일반 상식을 통한 보다 효과적인 상담 응답 생성을 목표로 하고 있습니다.

- **Technical Details**: EmpRes는 일반 상식 변환기(commonsense transformer)를 사용하여 고객의 정서를 이해하고 적절하게 응답을 생성합니다. 이 연구는 상담 세션 중 고객의 발화에 감정 레이블을 부여하는 방법을 통해 고객의 정서적 요구에 부합하는 응답을 생성하기 위해 감정 레이블을 활용합니다. 또한, EmpRes는 GPT-2 기반의 수정된 지식 인식 주의를 통해 일반 상식을 학습하게 됩니다.

- **Performance Highlights**: EmpRes는 HOPE 데이터세트를 기반으로 한 방대한 분석을 진행하여 기존 모델보다 뛰어난 성과를 보였습니다. 91%의 사용자들이 시스템의 효과를 인정했으며, 80%가 만족감을 표현했습니다. 조사 결과, 85.45% 이상의 사용자가 이 인터페이스를 지속 사용하겠다는 의사를 밝혔으며, 이는 EmpRes의 실용적인 응용 가능성을 입증하고 있습니다.



### Trust Modeling in Counseling Conversations: A Benchmark Study (https://arxiv.org/abs/2501.03064)
- **What's New**: 이 논문은 정신 건강 상담에서 치료사와 환자 간의 신뢰를 평가하기 위한 새로운 매트릭스인 'trust'를 도입합니다. 신뢰는 환자가 자신의 감정을 솔직하게 표현하고, 그에 따라 더 나은 치료를 받을 수 있는 의지와 개방성을 반영합니다. 저자들은 이를 동적인 궤적(dynamic trajectory)으로 개념화하고, MENTAL-TRUST라는 새로운 상담 데이터를 제안하여 신뢰를 모델링하고 평가하고자 합니다.

- **Technical Details**: MENTAL-TRUST는 212개의 상담 세션에서 수집된 12.9K 발화(utterances)로 구성된 데이터셋입니다. 이 데이터셋은 전문가가 검증한 7개의 순서형 신뢰 수준이 수동으로 주석 처리되어 있습니다. 또한 TRUST-BENCH라는 벤치마크를 통해 다양한 언어 모델의 신뢰 감지 성능을 평가하며, 이 과정에서 큰 언어 모델(LLMs)과 더 작은 모델 간의 성능 차이를 규명합니다.

- **Performance Highlights**: 연구 결과, BART, BERT, DeBERTa와 같은 작은 모델들이 큰 언어 모델보다 신뢰의 작은 변화를 포착하는 데 더 뛰어난 능력을 보여주었습니다. 이는 신뢰의 변화가 상담 대화에서 어떻게 진전되는지를 분석하는 데 중요합니다. 연구는 또한 신뢰의 궤적(trust trajectory)이 실시간 전략 조정에서 치료사에게 중요한 통찰을 제공함을 강조합니다.



### Quantization Meets Reasoning: Exploring LLM Low-Bit Quantization Degradation for Mathematical Reasoning (https://arxiv.org/abs/2501.03035)
Comments:
          4 pages

- **What's New**: 본 연구는 대형 언어 모델(LLM)의 수학적 추론 작업에 대한 양자화가 미치는 영향을 체계적으로 평가했습니다. 우리는 다양한 양자화 방법의 단계별 출력에 대한 정량적 분석과 특정 능력 차원을 질적으로 평가하는 다차원 평가 프레임워크를 도입합니다. 양자화가 수치 계산 능력과 추론 계획 능력에 미치는 차별적인 영향을 밝혀내어, 양자화 모델에서 성능 저하가 발생하는 주요 분야를 확인했습니다.

- **Technical Details**: 연구에서는 GPTQ와 AWQ와 같은 가중치 전용 양자화 및 SmoothQuant와 같은 가중치-활성화 양자화를 조사했습니다. 각 양자화 방법의 정확도를 평가하기 위해 다양한 설정에서 실험을 진행했으며, LoRA를 활용하여 양자화 모델의 효과적인 미세 조정을 수행했습니다. PRM800K 데이터세트를 사용하여 모델이 차별적인 추론 단계를 학습하고, 명확한 경계를 가지고 추론할 수 있도록 특별한 토큰을 도입했습니다.

- **Performance Highlights**: MATH 데이터세트에서의 실험 결과, 양자화된 모델은 전반적으로 성능 저하를 보였습니다. 모든 양자화 방법이 성능 손실을 초래했으며, 특히 SmoothQuant 방식이 가장 작은 성능 저하를 나타냈습니다. 양자화 모델에서의 성능 저하는 주로 문제의 오해와 논리적 오류와 같은 여러 요인으로 구분돼 분석되었습니다.



### Quality Estimation based Feedback Training for Improving Pronoun Translation (https://arxiv.org/abs/2501.03008)
- **What's New**: ProNMT는 기계 번역 시스템의 대명사 번역 품질을 향상시키기 위한 새로운 프레임워크입니다. 이는 Quality Estimation(QE) 모델과 대명사 생성 가능성 기반 피드백 메커니즘을 통합하여 기존의 NMT 모델을 반복적으로 세밀하게 조정할 수 있습니다. ProNMT는 대규모 인간 주석 없이도 번역 품질을 개선하도록 설계되었습니다.

- **Technical Details**: ProNMT는 기본적으로 두 가지 주요 개념에 기반합니다. 첫째, 품질 추정(QE)을 통해 출력된 대명사의 번역 품질을 평가합니다. 둘째, 대명사 생성 가능성 기반 피드백을 통해 번역 시 올바른 대명사를 생성할 확률에 따라 피드백을 제공합니다. 이러한 메커니즘을 통해 번역 품질과 대명사 번역의 적절성을 동시에 향상시키는 것을 목표로 합니다.

- **Performance Highlights**: 실험 결과, ProNMT는 대명사 번역의 정확성과 일반 번역 품질 모두에서 기존의 방법들에 비해 상당한 개선을 보였습니다. 이는 문서 수준의 기계 번역(MT) 시스템이 대명사 같은 맥락 의존적 요소를 처리하는 데 있어 효과적으로 작용함을 보여줍니다. ProNMT는 기계 번역 품질 향상을 위한 효율적이고 확장 가능한 접근법을 제공합니다.



### Registering Source Tokens to Target Language Spaces in Multilingual Neural Machine Translation (https://arxiv.org/abs/2501.02979)
- **What's New**: 이번 논문에서는 추세에 맞추어 다국어 신경망 기계 번역(MNMT)의 새로운 접근 방식인 'registering'을 소개합니다. 기존의 대규모 언어 모델(LLM)들이 가진 번역 성능 한계를 극복하고, 파라미터 수를 제한하면서도 더욱 효율적인 다국어 번역을 가능하게 합니다. 이 방식은 소스와 대상 토큰 사이에 인공 토큰인 registers를 삽입하여 입력 시퀀스를 수정함으로써 이루어집니다. 실험 결과, 제안된 방법이 다른 관련 방법보다 우수한 성능을 보이는 것으로 나타났습니다.

- **Technical Details**: MNMT 모델에서 registers는 소스 토큰과 대상 토큰 사이에 위치하여 번역 언어를 명시하는 인공 토큰의 집합입니다. 이 시스템은 주의 마스크(attention mask)를 수정하여 목표 언어 공간 내에서의 토큰 생성에만 주의를 기울이도록 설계되었습니다. 실험에서는 두 가지 세트의 평가가 이루어졌으며, 다국어 표현 최적화에 의한 관련 기법을 초월하는 성과를 거두었습니다. 특히, EC-40라는 대규모 벤치마크에서 제안된 방법이 평균 71%의 spBLEU 점수 증가를 보였습니다.

- **Performance Highlights**: 제안된 MNMT 모델인 MITRE는 9.3 billion 문장 쌍을 기반으로 사전 훈련된 두 모델, MITRE-466M와 MITRE-913M을 통해 NLLB-3.3B 및 GPT-3.5 Turbo를 초월하는 성능을 보였습니다. 특히, MITRE-913M은 상업용 LLM들과 비교할 수 있는 성능을 달성하며, 파인튜닝(fine-tuning)에서의 강한 적응성을 보여줍니다. 이 연구의 결과는 다국어 신경 기계 번역 분야의 후속 연구에 기여할 것으로 기대됩니다.



### Explaining Humour Style Classifications: An XAI Approach to Understanding Computational Humour Analysis (https://arxiv.org/abs/2501.02891)
- **What's New**: 본 논문은 유머 스타일 분류를 위한 설명 가능한 AI(XAI) 프레임워크를 제시하고 있습니다. 이는 기존 연구를 기반으로 하며, 언어적, 감정적, 의미적 특징이 유머 스타일 분류 결정에 어떻게 기여하는지를 분석합니다. 이러한 분석을 통해 각 유머 스타일이 어떻게 특성화되고 잘못 분류되는지에 대한 명확한 패턴을 발견하였습니다.

- **Technical Details**: 이 연구는 ALI+XGBoost 모델을 활용하여 XAI 기법을 적용합니다. XAI는 LIME(Local Interpretable Model-Agnostic Explanations) 및 SHAP(Shapley Additive Explanations) 기법을 통해 모델의 예측 결정과 주요 특징들을 해석할 수 있도록 지원하며, 감정의 모호성, 문맥의 오해, 목표 식별 등 모델 결정에 영향을 미치는 주요 요인을 규명합니다.

- **Performance Highlights**: 연구 결과, XAI 프레임워크는 유머 스타일 분류의 투명성을 높이는 데 기여하며, 연구자들에게 유머의 역할을 더욱 깊이 있게 조사할 수 있는 실제적인 통찰을 제공합니다. 특히, 감정적 특징과 언어적 요소가 모델 예측에 미치는 영향을 상세히 분석하여, 정신 건강, 콘텐츠 조정 및 디지털 인문학 연구에 대한 실용적인 응용 가능성을 제시합니다.



### IIMedGPT: Promoting Large Language Model Capabilities of Medical Tasks by Efficient Human Preference Alignmen (https://arxiv.org/abs/2501.02869)
- **What's New**: 이 논문에서는 IIMedGPT라는 새로운 의학 언어 모델을 소개하며, CMedINS라고 불리는 의학 지침 데이터셋을 통해 LLM의 성능을 개선하는 방법을 제시합니다. 특히, Direct Preference Optimization(DPO) 방법을 통해 모델이 사용자 지침에 맞춰 강화되는 것을 목표로 하였습니다. 이 모델은 기존 의학 모델과 비교하여 우수한 성능을 보였으며, 코드와 모델 체크포인트는 논문 수락 시 공개될 예정입니다.

- **Technical Details**: IIMedGPT는 감독된 미세 조정(supervised fine-tuning) 및 직접 정책 최적화(Direct Policy Optimization)라는 두 단계의 훈련 방식으로 개발되었습니다. CMedINS는 실제 의료 작업에서 파생된 여섯 가지 의료 지침을 포함하여 220,000개의 쌍의 의료 기록을 수집하여 구성되었습니다. 또한, 모델 훈련 시 효과적인 일반화 능력을 확보하기 위해 의료 대화 데이터셋과 일반 능력 데이터셋을 혼합하여 사용하였습니다.

- **Performance Highlights**: IIMedGPT는 성능 평가에서 GPT-4 및 인간 전문가와 비교하여 세 가지 능력 차원과 아홉 가지 특정 능력에서 우수한 결과를 기록하였습니다. 특히, 모델은 적은 양의 훈련 데이터에도 불구하고 기존 오픈 소스 전통 중국 의학 LLM보다 향상된 성능을 발휘했습니다. 이는 의료 지침 및 대화 처리 능력을 극대화하기 위해 신중하게 수집된 데이터의 효과를 보여주는 결과입니다.



### Graph-based Retrieval Augmented Generation for Dynamic Few-shot Text Classification (https://arxiv.org/abs/2501.02844)
- **What's New**: 이 논문에서는 동적 소수 샷 텍스트 분류(dynamic few-shot text classification)를 위한 새로운 그래프 기반 온라인 검색-Augmented Generation 프레임워크인 GORAG을 제안합니다. GORAG은 텍스트의 모든 대상에 대해 사이드 정보를 추출하여 적응형 정보 그래프를 구축하고 유지합니다. 기존 접근 방식의 한계를 극복하면서 LLM(Large Language Model)의 효과를 극대화하는 것을 목표로 합니다.

- **Technical Details**: GORAG은 LLM의 키워드를 사용하여 텍스트에서 키워드를 추출하고, 이러한 키워드를 텍스트의 실제 레이블과 연결하여 키워드와 레이블 간의 관계를 나타냅니다. 또한, GORAG은 가중 엣지 메커니즘을 사용하여 그래프의 엣지를 인덱싱할 때 키워드의 중요성과 레이블과의 관련성을 기반으로 엣지 가중치를 부여합니다. 정보를 검색하기 위해 최소 비용 신장 트리(minimum-cost spanning tree)를 구성하여 각 텍스트에 대해 후보 레이블을 검색합니다.

- **Performance Highlights**: 실험 평가 결과 GORAG은 기존 접근 방식보다 더 포괄적이고 정확한 맥락 정보를 제공하여 성능이 향상된 것으로 나타났습니다. 기존 모델들이 가지는 데이터의 상호 연관성과 중복 문제를 해결하여 동적 변경이 빈번한 레이블에 대해서도 효과적으로 분류할 수 있는 능력을 보여주었습니다. 이러한 GORAG의 접근은 소수의 레이블만 있는 상황에서도 높은 성능을 유지하도록 돕습니다.



### Samba-ASR: State-Of-The-Art Speech Recognition Leveraging Structured State-Space Models (https://arxiv.org/abs/2501.02832)
- **What's New**: Samba ASR는 Mamba 아키텍처를 이용한 최신의 자동 음성 인식(ASR) 모델로, 전통적인 transformer 모델의 단점을 극복하며 뛰어난 성능을 발휘한다. 특히, 기존의 transformer-based ASR 모델의 문제점인 긴 입력 시퀀스에서의 복잡도를 줄이고, 긴 거리의 종속성을 효과적으로 처리할 수 있는 구조화된 상태 공간 모델(SSM)을 기반으로 한다. Samba ASR의 도입은 경량화된 ASR 시스템의 새로운 기준을 제시한다.

- **Technical Details**: Samba ASR는 상태 공간 모델(SSM)을 활용해 음성 인식을 수행하며, Mamba 아키텍처를 통해 효율적인 상태 공간 동역학을 구현하고 있다. Mamba는 선택적인 회귀(selective recurrence)와 하드웨어 인식 최적화(hardware-aware optimizations)를 통해 복잡한 입력에 적절하게 적응할 수 있도록 한다. 이를 통해, 입력 의존적(state space equations) 매개변수를 도입하고, 더 작은 상태 표현으로 컨텍스트(context)를 압축하는 능력을 지향한다.

- **Performance Highlights**: Samba ASR는 Gigaspeech와 SPGISpeech 등 주요 ASR 벤치마크에서 최고의 성능을 기록하였다. 이 모델은 훈련속도와 추론 지연을 줄이면서도 높은 정확도를 유지하며, 노이즈가 있거나 자발적인 발화와 같은 어려운 조건에서도 뛰어난 성능을 보여준다. 이러한 결과는 Samba ASR가 다양한 ASR 시스템에 대해 스케일 가능하고 강력한 솔루션임을 입증한다.



### InfiFusion: A Unified Framework for Enhanced Cross-Model Reasoning via LLM Fusion (https://arxiv.org/abs/2501.02795)
Comments:
          Under review

- **What's New**: 대형 언어 모델(LLMs)이 다양한 추론 작업에서 탁월한 성과를 보여주고 있지만, 모든 영역에서 일관되게 높은 성능을 발휘하는 단일 모델 구축은 여전히 도전 과제입니다. 본 논문은 여러 도메인 전문화된 모델을 효율적인 피벗 모델로 통합하는 전략을 탐구하며, 페어와이즈 멀티스텝 퓨전 접근법과 통합 퓨전 접근법의 두 가지 방식을 제안합니다. 특히, 새로운 Rate-Skewness Adaptive Fusion (RSAF) 기법을 도입하여 파라미터 병합 시 동적으로 top-K 비율을 조정합니다.

- **Technical Details**: 제안된 방법은 각 도메인에 특화된 K개의 소스 LLM을 활용해 피벗 모델 M_{p}으로 통합하는 것을 목표로 합니다. Pairwise fusion 접근법은 각 소스 모델을 순차적으로 통합하며, Knowledge Distillation 방법론을 통해 소스 모델의 지식을 효과적으로 수집합니다. 통합 퓨전 접근법은 가능한 모든 소스 모델의 출력을 집계하여 혼합 프로세스를 최적화하고, 예상된 불확실성에 기반한 가중치 조정 방법인 Uncertainty-based Distribution Fusion (UDF)을 활용합니다.

- **Performance Highlights**: 제안된 RSAF와 UDF 기법은 GSM8K, MATH, HumanEval 과제에서 각각 9.27%, 8.80%, 8.89%의 정확도 향상을 달성했다는 실험 결과를 보입니다. Llama3.1-8B 모델을 활용한 광범위한 실험을 통해, 제안된 방법론이 전통적인 감독미세조정(Supervised Fine-Tuning) 방법의 성능을 크게 초월함을 입증하였습니다. 이러한 성과는 각 소스 모델의 고유한 도메인 전문성을 효과적으로 활용했음을 나타냅니다.



### Segmenting Text and Learning Their Rewards for Improved RLHF in Language Mod (https://arxiv.org/abs/2501.02790)
- **What's New**: 본 논문에서는 인간 피드백을 통한 강화 학습(RLHF)의 새로운 접근 방식을 제안합니다. 이 접근 방식은 텍스트의 의미적으로 완전한 세그먼트에 대한 보상을 할당하기 위해 세그먼트 수준 보상 모델을 훈련하고 활용합니다. 기존의 토큰 기반 보상 접근 방식의 복잡성을 줄이면서도 보상의 질을 향상시키는 방법론을 고안하였습니다.

- **Technical Details**: 제안된 방법은 동적 텍스트 세그먼테이션을 통해 언어 모델(LM) 생성의 두 가지 난제를 해결합니다. 이를 위해 보상 모델을 Bradley-Terry(BT) 손실 함수를 사용하여 훈련하며, 위치 인식 보상 정규화 기능으로 고전적 보상 정규화를 일반화합니다. 이로 인해 세그먼트 보상이 효과적으로 보강되어 강화 학습 기반 LM 훈련에 필요한 신호의 밀도가 향상됩니다.

- **Performance Highlights**: 논문의 방법론은 AlpacaEval 2.0, Arena-Hard, MT-Bench 등 세 가지 RLHF 벤치마크에서 고전적 밴딧 접근법과 최근의 토큰 레벨 보상 접근법 대비 경쟁력 있는 성능을 달성하였습니다. 여러 차별화를 위한 연구도 수행되어 제안된 디자인 선택의 효과를 증명하였습니다.



### TARDiS : Text Augmentation for Refining Diversity and Separability (https://arxiv.org/abs/2501.02739)
Comments:
          10 pages

- **What's New**: 이 논문에서는 TARDiS라는 새로운 LLM 기반의 텍스트 증강(Text Augmentation, TA) 방법을 소개합니다. 이 방법은 기존의 두 단계 TA 방법에서 발생하는 생성 및 정렬 단계의 도전 과제를 해결하는 데 중점을 둡니다. TARDiS는 여러 개의 클래스별 프롬프트를 사용하여 다양성과 분리 가능성을 향상시키는 두 가지 생성 프로세스, 즉 SEG (Semantic Enrichment Generation)와 CEG (Contrastive Enrichment Generation)를 제안합니다.

- **Technical Details**: TARDiS의 생성 단계에서는 각 클래스에 대한 스파크 생각(spark thoughts) 개념을 도입하여 LLM의 내재된 지식을 활성화합니다. SEG는 목표 클래스 내의 예제에서 생성된 스파크 생각을 사용하여 클래스 내 다양성을 캡처하며, CEG는 목표 클래스와 모호한 클래스에서 생성된 스파크 생각을 사용해 비목표 클래스와의 분리 가능성을 강화합니다. 정렬 단계에서는 Class Adaptation (CA) 방법을 통해 생성된 예제가 목표 클래스와 일치하도록 수정합니다.

- **Performance Highlights**: 실험 결과, TARDiS는 다양한 퓨쇼트(few-shot) 텍스트 분류 작업에서 기존 LLM 기반 TA 방법들보다 우수한 성능을 보였습니다. 또한, 논문은 각 단계에서의 행동을 상세히 분석하여 TARDiS의 효과를 입증하였습니다. 이를 통해 TARDiS는 기존의 두 단계 TA 방법의 한계를 극복하며, 퓨쇼트 상황에서도 강력한 일반화 성능을 발휘합니다.



### QuIM-RAG: Advancing Retrieval-Augmented Generation with Inverted Question Matching for Enhanced QA Performanc (https://arxiv.org/abs/2501.02702)
- **What's New**: 이 연구는 질문 응답(QA) 작업을 개선하기 위한 Retrieval-Augmented Generation (RAG) 시스템의 새로운 아키텍처를 제안합니다. 기존의 대형 언어 모델(LLMs)에서는 실시간 업데이트가 어려웠지만, RAG는 하이퍼링크와 데이터베이스를 통합하여 맥락적으로 적합한 응답을 생성합니다. 본 연구에서는 정보의 희석(information dilution)과 망상(hallucinations)이라는 전통적인 RAG가 직면한 문제를 해결하기 위해 QuIM-RAG(Query-to-Question Inverted Index Matching)이라는 새로운 접근 방식을 도입하였습니다.

- **Technical Details**: QuIM-RAG는 문서 청크에서 잠재적인 질문을 생성하고 이를 사용자 쿼리와 매칭하여 가장 관련 있는 텍스트 청크를 찾아 정확한 응답을 생성합니다. 이 시스템은 Meta Inc.의 오픈소스 Meta-LLaMA3-8B-instruct 모델을 기반으로 구현되었습니다. 500 페이지 이상의 다양한 웹사이트에서 수집한 도메인 특화 데이터셋을 활용했으며, BERT-Score와 RAGAS 메트릭스를 사용하여 평가를 수행하였습니다.

- **Performance Highlights**: 본 연구에서 제안하는 접근 방식은 전통적인 RAG 아키텍처보다 BERT-Score와 RAGAS에서 모두 더 나은 성능을 나타냈습니다. 이러한 결과는 사용자 쿼리에 대한 정확한 응답을 생성하는 데 있어 우리의 새로운 RAG 모델이 효과적임을 보여줍니다. 또한, 사용자에게 제공하는 모든 정보는 원본 문서와 연결된 출처 링크를 포함하고 있어 신뢰성 있는 정보를 탐색할 수 있도록 지원합니다.



### Decoding specialised feature neurons in LLMs with the final projection layer (https://arxiv.org/abs/2501.02688)
Comments:
          5 pages, 3 figures

- **What's New**: 이 연구에서는 Llama 3.1 8B 모델을 통해 LLM의 해석 가능성(interpretable)을 높이는 새로운 접근 방식을 제안합니다. TARS(타겟 각도 반전법) 방법으로 지식 제거를 시도하며, 특정 개념에 강하게 반응하는 특화된 메커니즘이 존재함을 보여줍니다. 또한, 수도 코드 프로젝션 모델의 마지막 레이어에서 뉴런 가중치를 토큰 확률로 직접 디코딩하는 방법을 제시하여, 그 가중치가 특정 토큰과 어떻게 연결되는지 분석합니다.

- **Technical Details**: 이 방법은 Llama 3.1(8B) 모델의 업 프로젝션 층에서 뉴런의 가중치를 LM-head를 통해 직접 디코딩하는 방식입니다. 이 과정에서 특정 개념에 대응하는 뉴런을 찾기 위해, 뉴런의 토큰별 확률을 계산하고 이를 기준으로 최적의 뉴런을 식별합니다. 제출된 방법론은 사전 훈련된 모델과 fine-tuned 모델 모두에서 적용 가능하며, 이는 뉴런 활성화를 조정하여 특정 토큰의 확률에 미치는 영향을 평가하는 것입니다.

- **Performance Highlights**: 실험을 통해 연구진은 'dog'와 'California'라는 두 개념에 해당하는 뉴런의 확률을 15분 이내에 시각화했습니다. 특히, 'dog' 뉴런을 조정했을 때 모델이 항상 '개'에 대해 언급하도록 유도할 수 있음을 확인했습니다. 또한, 75.4%의 경우에서 fine-tuning 이후에도 뉴런의 상위 토큰이 유지된다는 점이 나타났습니다, 이는 모델의 해석 가능성을 더욱 강화하는 결과로 해석됩니다.



### From Superficial Patterns to Semantic Understanding: Fine-Tuning Language Models on Contrast Sets (https://arxiv.org/abs/2501.02683)
- **What's New**: 이 연구는 자연어 추론(Natural Language Inference, NLI) 태스크에서 언어 모델의 편향을 줄이고 그 강건성을 향상시키기 위해 대조 세트(contrast sets)를 사용하는 방법을 탐구합니다. 기존의 데이터셋에서 높은 정확도를 기록하는 ELECTRA-small 모델이 대조 세트에 평가될 때 정확도가 급격히 떨어지는 현상을 분석하여, 더 복잡한 예제를 활용한 사전 훈련 모델의 미세 조정이 필요하다는 점을 강조합니다. 이를 통해 모델이 언어를 깊이 이해할 수 있도록 도움을 줄 수 있습니다.

- **Technical Details**: 이 연구에서는 1,400만 개의 매개변수를 가진 ELECTRA-small 모델을 훈련하여 SNLI(Supervised Natural Language Inference) 데이터셋에서 NLI 작업을 수행하였습니다. 대조 세트는 Linguistically-Informed Transformations(LIT)를 사용하여 자동 생성하였으며, 총 14,363개의 대조 예제가 만들어졌습니다. 이 대조 세트는 원본 SNLI 데이터의 원래 레이블을 변경시키는 방식으로 미세 조정 과정에서 활용되었습니다.

- **Performance Highlights**: 대조 세트에서 모델의 정확도를 향상시키기 위해 복잡한 예제에 노출시키는 훈련 방법을 사용하였고, 이로 인해 정확도는 74.9%에서 90.7%로 증가하였습니다. 대조 세트에 대한 성능 향상은 훈련 데이터의 다양성이 모델의 언어 이해를 향상시키는 데 얼마나 중요한지를 보여줍니다. 원본 SNLI 데이터 세트에서의 성능은 유지되었으며, 이는 모델이 복잡성과 변동성에 저항력을 가질 수 있는 길을 열어줍니다.



### Tougher Text, Smarter Models: Raising the Bar for Adversarial Defence Benchmarks (https://arxiv.org/abs/2501.02654)
Comments:
          Will be presented as an oral in-person presentation at the conference of COLING 2025

- **What's New**: 이번 연구에서는 텍스트 적대적 방어를 위한 포괄적인 벤치마크를 제시하여 이전 연구를 크게 확장했습니다. 이 벤치마크는 다양한 데이터셋, 최신 방어 메커니즘을 평가하며, 단일 문장 분류, 유사성 및 패러프레이즈 식별, 자연어 추론, 상식 추론과 같은 중요한 작업들로 평가 범위를 확장합니다. 이 연구는 적대적 강건성을 연구하는 연구자와 실무자들에게 귀중한 자원이 될 뿐만 아니라, 텍스트 적대적 방어에서의 미래 연구의 주요 영역을 식별합니다. 

- **Technical Details**: 딥러닝 모델의 적대적 공격에 대한 취약성은 NLP의 주요 관심사로 떠오르고 있으며, 본 섹션에서는 증가하는 적대적 방어 방법과 이것의 다양한 NLP 작업에 대한 적응 가능성을 강조합니다. 적대적 공격은 입력 텍스트를 조작하여 의미를 보존하지만 모델의 오분류를 초래하는 것을 목표로 합니다. 이 연구에서는 다양한 방어 전략 중에서 구조의 유연성을 중시하며, 학습 효율성을 높이기 위해 필요하지 않은 정보 없이도 견고한 방어 방법을 제안합니다.

- **Performance Highlights**: 제안된 TTSO++는 엔트로피 항을 통한 동적 신뢰도 조정이 통합된 새로운 변형으로, 텍스트 적대적 공격에 대한 강인성을 크게 향상시킬 수 있습니다. 특히 TextFooler와 TextBugger 시나리오에서 더욱 뛰어난 성능을 보입니다. 본 연구에서 제시된 벤치마크는 최신 방어 기술과 다양한 NLP 작업을 평가하여, 적대적 방어 영역에서 더 나은 기준을 제시하고 있으며, 향후 연구에 있어 실질적인 발전을 가속화하는 토대를 마련할 것으로 기대됩니다.



### Prune or Retrain: Optimizing the Vocabulary of Multilingual Models for Estonian (https://arxiv.org/abs/2501.02631)
Comments:
          Published in the Proceedings of the 9th International Workshop on Computational Linguistics for Uralic Languages

- **What's New**: 이번 연구에서는 에스토니아어에 맞춘 다국어 언어 모델의 어휘 조정이 Named Entity Recognition (NER) 작업의 성능에 미치는 영향을 탐구합니다. 전통적인 방법의 한계를 극복하기 위해, 새로 초기화된 임베딩을 훈련하여 어휘 최적화의 효과를 분석합니다. 이 과정에서 토크나이저를 재훈련하거나 사용하지 않는 토큰을 잘라내는 두 가지 접근 방식을 평가합니다.

- **Technical Details**: 다국어 모델에 최적화된 어휘를 적용하기 위해 mDeBERTa v3 모델을 선택하였고, Estonian National Corpus (ENC)에서 새로운 토크나이저를 훈련했습니다. 어휘 조정의 두 가지 방법으로는 새 토크나이저 훈련과 초기 어휘의 가지치기(pruning)를 사용하였습니다. 가지치기에서는 데이터에 나타나지 않는 토큰을 제거하여 약 67%의 원래 어휘를 유지하는 방식으로 진행했습니다.

- **Performance Highlights**: 토크나이저 재훈련 방법은 NER 작업의 성능을 저하시키는 경향을 보였으나, 가지치기를 통한 접근법은 부정적인 영향을 주지 않았습니다. 이 연구는 에스토니아어와 같은 특정 언어의 효율성을 높이기 위한 어휘 조정의 효과를 보여주며, 더 빠르고 정확한 모델 훈련을 위한 새로운 인사이트를 제공합니다.



### Empowering Bengali Education with AI: Solving Bengali Math Word Problems through Transformer Models (https://arxiv.org/abs/2501.02599)
- **What's New**: 본 연구는 벵골어(MWPs) 수학 문제를 해결하기 위한 혁신적인 접근법을 제시합니다. 특히 transformer 기반 모델인 Basic Transformer, mT5, BanglaT5, mBART50을 활용하며, 이를 지원하기 위해 10,000개의 벵골어 수학 문제를 포함한 "PatiGonit" 데이터셋을 도입하였습니다. 이 연구는 벵골어의 자연어 처리(NLP) 분야에서 큰 진전을 이루어내며, 교육 AI 도구 개발에 기여할 수 있는 귀중한 방법론과 자원을 제공합니다.

- **Technical Details**: 연구자들은 텍스트 내에 포함된 수학 방정식을 식별하기 위해 transformer 기반 모델을 적용하였으며, 이 과정에서 하이퍼파라미터(learning rate, epochs, batch size)를 최적화하였습니다. 이 접근법은 자연어 처리 기술을 통해 수학 문제의 방정식을 예측하고, 예측된 방정식을 통해 최종 답을 도출하는 방식을 사용합니다. 최종적으로 mT5 모델은 97.30%의 정확도를 기록하여 transformer 모델들이 벵골어 문제 해결에 효과적임을 입증하였습니다.

- **Performance Highlights**: 이 연구는 벵골어로 된 수학 문제 해결의 최신 기술을 발전시키고, 다국어 교육 기술의 접근성을 높이는 것을 목표로 합니다. ‘PatiGonit’ 데이터셋의 생성과 transformer 모델의 적용 및 미세 조정을 통해 벵골어 수학 문제 해결에 대한 효과를 입증하였습니다. 이 연구는 벵골어를 사용하는 학생들이 수학 교육 및 문제 해결 능력을 향상시키는 데 중요한 기여를 합니다.



### GIT-CXR: End-to-End Transformer for Chest X-Ray Report Generation (https://arxiv.org/abs/2501.02598)
- **What's New**: 이번 연구에서는 X-ray 이미지에서 자동으로 방사선 보고서를 생성하기 위해 엔드투엔드( end-to-end ) transformer 기반의 새로운 방법을 제안하였습니다. 또한 의료 이미징에서 엔드투엔드 transformer에 커리큘럼 학습(curriculum learning)을 처음으로 도입하여 성과를 향상시켰습니다. 실험은 MIMIC-CXR-JPG 데이터베이스를 사용하여 진행되었으며, 기존의 자연어 생성 평가 지표에서 새로운 최첨단 결과를 기록했습니다.

- **Technical Details**: 연구에서는 GIT transformer에 다양한 기법을 통합하여 X-ray 이미지에서 방사선 보고서를 자동 생성하는 과정을 최적화했습니다. 주요 기술로는 분류 헤드를 추가하고 환자의 병력을 활용하며 여러 뷰의 이미지를 사용하는 것을 포함합니다. 특히, 커리큘럼 학습 방식이 우리의 훈련 과정에 통합되어 있으며, 이는 더 긴 의료 보고서 생성이라는 핵심적인 문제를 해결하는 데 중요한 역할을 합니다.

- **Performance Highlights**: 우리가 제안한 방법은 METEOR, F1-macro, F1-micro와 같은 여러 임상 정확성 메트릭에서 신규 최첨단 결과를 세웠으며, 이는 생성된 보고서의 정확성과 사실 완전성을 잘 보여줍니다. 또한 BLEU 및 ROUGE-L 같은 자연어 생성 메트릭에서도 기존의 최첨단 기술과 동등한 성능을 보여주었습니다. 이러한 성과는 방사선 이미징 분야의 커뮤니티에 매우 중요한 기여를 하고 있습니다.



### Multi-LLM Collaborative Caption Generation in Scientific Documents (https://arxiv.org/abs/2501.02552)
Comments:
          Accepted to AAAI 2025 AI4Research Workshop

- **What's New**: 이 논문에서는 과학적 그림 캡셔닝(figure captioning)이라는 복잡한 작업을 해결하기 위한 새로운 프레임워크인 Multi-LLM Collaborative Figure Caption Generation(MLBCAP)을 제안합니다. 기존의 접근 방식이 이미지-텍스트 변환이나 텍스트 요약 작업으로만 한정된 반면, MLBCAP은 다양한 LLM(multi-modal large language models)을 활용하여 캡션 생성의 각 하위 작업을 수행합니다. 이 framework는 품질 평가(quality assessment), 다양한 캡션 생성(diverse caption generation), 그리고 판단(judgment)이라는 세 가지 모듈로 구성되어 있습니다.

- **Technical Details**: MLBCAP는 다중 모달 LLMs를 활용하여 훈련 데이터의 품질을 평가하고 저품질 캡션을 필터링합니다. 다양한 캡션 생성을 위해서는 여러 LLM을 미세 조정(fine-tuning)하거나 프롬프트(prompting)하여 후보 캡션을 생성합니다. 마지막으로, 최고의 캡션을 선택하고 오차를 수정하기 위해 GPT-4o와 같은 저명한 LLM을 사용합니다.

- **Performance Highlights**: 이 방법으로 생성된 캡션은 인간 전문가의 평가에서 원래 저자 작성 캡션보다 우수한 점수를 받았습니다. 또한, MLBCAP는 다양한 길이의 캡션을 생성하도록 설계되어, 학술 저널과 회의 논문의 페이지 제한에 적합한 방식으로 정보를 전달합니다. 인간 평가에서 MLBCAP에 의해 생성된 캡션이 높은 품질과 정확성을 보임을 입증하여, 과학적 소통의 효율성을 향상시키는 데 기여할 것으로 예상됩니다.



### From Language To Vision: A Case Study of Text Animation (https://arxiv.org/abs/2501.02549)
- **What's New**: 이번 논문에서는 다양한 형식으로 표현되는 정보를 시각화하는 새로운 시스템을 소개합니다. 이 시스템은 자연어(natural language)를 애니메이션(animation)으로 변환할 수 있는 기능을 가지고 있습니다. 특히, 초등 물리학 법칙을 예로 들어 본 시스템의 적용 가능성을 보여줍니다.

- **Technical Details**: 이 텍스트 시각화 시스템은 자유 텍스트(free text)를 분석하여 애니메이션으로 표현하는 방식으로 설계되었습니다. 이는 정보의 다양한 표현 방식에 대한 인간 인지의 유연성을 활용하여, 사용자에게 더욱 직관적인 이해를 제공합니다. 기술적으로는 자연어 처리(natural language processing)와 시각적 표현(visual representation)을 결합하여 구현됩니다.

- **Performance Highlights**: 시스템의 초기 실험 결과, 사용자는 텍스트가 애니메이션으로 시각화되는 과정에서 정보 전달의 이해도가 향상된 것으로 나타났습니다. 더불어, 시각적 요소가 추가됨으로써 학습 효과가 극대화되었고, 이는 현실 세계의 다양한 응용 가능성을 시사합니다.



### TreeMatch: A Fully Unsupervised WSD System Using Dependency Knowledge on a Specific Domain (https://arxiv.org/abs/2501.02546)
- **What's New**: 이 논문은 Word Sense Disambiguation (WSD), 즉 단어 의미의 구분을 위한 새로운 시스템인 TreeMatch를 소개하고 있습니다. 이 시스템은 SemEval 2007 Task 7의 데이터를 사용하여 처음 개발되었고, SemEval 2010 Task 17의 특정 도메인에 맞게 조정되었습니다. TreeMatch는 특정 도메인 지식 기반에서 얻은 의존성 지식을 활용한 완전 비지도 학습 방법에 기반하고 있습니다.

- **Technical Details**: TreeMatch는 특정 도메인을 위한 지식 기반을 구축하여 시스템의 성능을 극대화합니다. 이 시스템은 비지도 학습(unsupervised learning) 방식을 사용하여 단어 의미를 구분하며, 기존의 Most Frequent Selection 기법보다 더 나은 정확성을 보여줍니다. 또한, 의존성 지식(dependency knowledge)을 통해 단어 간의 관계를 파악하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 평가 결과, TreeMatch 시스템은 Most Frequent Selection 기준선(baseline)보다 높은 정확도(precision)를 기록하였습니다. 이는 이 시스템이 기존의 방식보다 효과적으로 단어 의미를 구분할 수 있음을 보여줍니다. 이러한 성능 향상은 고유한 도메인 정보와 비지도 학습을 통한 의존성 분석의 결과로 나타났습니다.



### Evaluating Large Language Models Against Human Annotators in Latent Content Analysis: Sentiment, Political Leaning, Emotional Intensity, and Sarcasm (https://arxiv.org/abs/2501.02532)
Comments:
          24 pages, 3 figures

- **What's New**: 이번 연구는 대규모 언어 모델(Large Language Models, LLMs)의 성능을 인적 주석자(human annotators)와 비교하는 종합적인 평가를 수행했다. OpenAI의 GPT-4, Gemini, Llama 및 Mixtral의 변형을 포함하여 7가지 최첨단 LLM을 분석하여, 감정 분석(sentiment analysis) 및 정치적 편향(political leaning) 평가에서의 신뢰성(reliability)과 일관성(consistency)을 측정하였다. 이러한 접근법은 디지털 커뮤니케이션 시대에 필요한 효율적인 잠재 콘텐츠 분석(latent content analysis) 방법의 가능성을 제시한다.

- **Technical Details**: 연구에서는 총 33명의 인적 주석자와 8개 LLM 변형이 100개의 선별된 텍스트 항목을 평가하였다. 이에 따라 3,300개의 인적 주석과 19,200개의 LLM 주석이 생성되었으며, LLM의 성능은 세 가지 시점(time points)에서 평가하여 시간에 따른 일관성을 살펴보았다. Krippendorff의 알파(Krippendorff's alpha)를 통해 주관자 간 신뢰성을 측정하고, 클래스 내 상관 계수(intra-class correlation coefficients)는 시간에 따른 일관성을 평가하였다.

- **Performance Highlights**: 연구 결과, 감정 분석 및 정치적 편향의 평가에서 LLM과 인간 모두 높은 신뢰성을 보였다. LLM은 인간보다 더 높은 내부 일관성(internal consistency)을 보였으며, 감정 강도(emotional intensity)에서는 LLM이 더 높은 일치를 나타내지만, 인간은 감정 강도를 더 높게 평가하였다. 반면, 조롱 감지(sarcasm detection)에서는 두 그룹 모두 낮은 일치를 보였으며, LLMs는 모든 차원에서 훌륭한 시간적 일관성(temporal consistency)을 보여줬다.



### CHAIR-Classifier of Hallucination as Improver (https://arxiv.org/abs/2501.02518)
- **What's New**: 본 논문은 대규모 언어 모델(LLM)에서 환각(hallucination)을 감지하기 위한 감독(supervised) 방법을 제안합니다. LLaMA 모델의 여러 레이어를 통과하는 토큰 점수(token scores, logit)를 분석하여 과적합(overfitting)을 줄이기 위한 적은 수의 특징(feature)을 도출했습니다. 로지스틱 회귀(logistic regression)를 사용하여 분류하고 TruthfulQA와 MMLU 데이터셋에서 모델을 검증했습니다.

- **Technical Details**: 이 연구에서 제안된 CHAIR(Classifier of Hallucination As Improver)은 각 레이어에서 생성된 표현을 활용하여 언어 모델 출력에서 환각을 효과적으로 식별합니다. LLaMA는 자기 주의(self-attention) 및 피드포워드(feed-forward) 레이어를 반복적으로 적용하여 토큰의 맥락적 표현을 구축하는 트랜스포머 기반 모델입니다. 각 레이어의 출력을 lm_head를 통해 점수로 변환하여 토큰의 신뢰도를 나타냅니다.

- **Performance Highlights**: 실험 결과, 제안한 방법이 특히 다중 선택(multiple-choice) 과제에서 성능을 크게 개선하여, 중간 레이어 정보가 환각 탐지에서의 효과적인 도구임을 입증했습니다. 이를 통해 내부 표현(internal representations)을 활용하여 환각을 탐지하고 완화할 수 있는 잠재적인 기반을 제공합니다. 이 접근법은 다양한 사용 사례에서 환각을 식별하고 줄이는 데 필요한 새로운 관점을 형성합니다.



### Can Impressions of Music be Extracted from Thumbnail Images? (https://arxiv.org/abs/2501.02511)
Comments:
          Accepted at NLP4MusA 2024

- **What's New**: 최근 음악 검색 및 생성 시스템을 위한 머신러닝 모델에 대한 연구가 증가하고 있지만, 음악 데이터와 해당 자연어 설명(music captions)으로 구성된 대규모 공개 데이터셋이 부족한 실정입니다. 특히, 트랙을 듣기에 적합한 상황이나 이에 따른 감정과 같은 비음악적 정보는 음악 설명에 있어 필수적입니다. 이를 해결하기 위해, 본 연구에서는 음악 썸네일 이미지를 활용하여 비음악적 요소를 포함한 음악 캡션 데이터를 생성하는 방법을 제안하였습니다.

- **Technical Details**: 이 연구는 YouTube와 같은 플랫폼에서 음악 클립과 연관된 썸네일 이미지를 중심으로 진행되었습니다. 제안된 방법에서는 먼저 썸네일 이미지를 대형 비전-언어 모델(LVLM)에 입력한 다음, LVLM이 신중하게 제작된 프롬프트를 통해 음악 캡션을 생성합니다. 이러한 과정은 비음악적 요소를 포함하는 음악 캡션의 자동 생성을 가능하게 하며, 생성된 캡션은 기존의 방법보다는 비음악적 정보를 효과적으로 포함합니다.

- **Performance Highlights**: 약 360,000개의 캡션으로 개발된 데이터셋이 공개되었으며, 이를 통해 음악 검색 모델을 학습시키고 그 효과성을 평가하였습니다. 인간 평가를 통해 제안된 방법이 기존 방법들보다 우수한 성능을 나타낸 것으로 확인되었습니다. 이 연구는 음악 설명 데이터의 다양성을 확보하고, 음악 검색 모델의 품질을 향상시키는 데 중요한 기여를 할 것으로 기대됩니다.



### ToolHop: A Query-Driven Benchmark for Evaluating Large Language Models in Multi-Hop Tool Us (https://arxiv.org/abs/2501.02506)
- **What's New**: ToolHop은 다중 도구 사용의 효과적인 평가를 위해 설계된 새로운 데이터 세트로, 995개의 사용자 쿼리와 3,912개의 관련 도구로 구성되어 있습니다. 이 데이터 세트는 다양한 쿼리, 의미 있는 상호 의존성, 로컬에서 실행 가능한 도구, 자세한 피드백 및 검증 가능한 답변을 보장합니다. 기존 연구의 한계를 극복하고, LLM의 이해 및 추론 능력을 면밀히 평가할 수 있는 기초 자료를 제공합니다.

- **Technical Details**: ToolHop은 쿼리 기반 데이터 구축 프로세스를 통해 구성되며, 이는 도구 생성, 문서 수정 및 코드 생성이 포함됩니다. 이를 통해 단일 다중 홉 쿼리를 포괄적인 다중 도구 사용 테스트 케이스로 확장할 수 있습니다. 각 도구는 문서와 코드 구현으로 정의되며, 문서는 도구의 이름, 기능 설명 및 매개변수를 포함합니다.

- **Performance Highlights**: ToolHop을 통해 14개의 LLM을 평가한 결과, GPT-4o가 49.04%의 정확도를 기록하여 다중 도구 사용에서 상당한 개선 여지가 있음을 보여주었습니다. 평가 결과, 각 모델 가족 간 도구 사용 전략의 차이가 나타났고, 이는 효과적인 방법 개발에 유용한 통찰력을 제공합니다. Qwen2.5 모델은 평행 호출을 강조하는 경향이 있으며, GPT 계열은 도구 피드백을 활용하여 성능을 향상시킵니다.



### Decoding News Bias: Multi Bias Detection in News Articles (https://arxiv.org/abs/2501.02482)
- **What's New**: 이 연구는 뉴스 기사에서 나타나는 다양한 편향을 탐구하고 이들 편향을 탐지하기 위한 데이터셋을 생성하는 데 대규모 언어 모델(LLM)을 사용한 점이 매력적입니다. 특히, 정치적이나 성별 편향뿐만 아니라 다양한 도메인에서 편향을 포괄적으로 식별하는 필요성을 강조하고 있습니다. 이를 통해 뉴스 기사의 무결성을 향상시키고자 하는 새로운 통찰을 제공합니다.

- **Technical Details**: 이 연구는 편향 식별 문제의 범위를 확장하여 뉴스 기사에서 발생할 수 있는 다양한 유형의 편향을 분석합니다. LLM을 사용한 데이터셋 주석 기법을 소개하며, 전처리 과정과 편향 탐지 실험에서 다양한 Transformer 기반 모델을 평가하고, 이를 통해 얻은 결과를 제공합니다. 이러한 접근은 편향 탐지에 대한 기존 연구의 한계를 극복하려는 의도를 가지고 있습니다.

- **Performance Highlights**: 이러한 방법론을 통해 특정 뉴스 기사의 편향을 효과적으로 탐지할 수 있는 가능성을 보여줍니다. 연구 결과는 LLM을 활용한 새로운 데이터셋과 그 주석 기법의 유용성을 뒷받침하며 다양한 유형의 편향을 단일 프레임워크로 탐지할 수 있는 잠재력을 제시합니다. 이를 통해 뉴스 제공자들은 더 높은 윤리적 기준을 유지하고 보다 균형 잡힌 보고를 위한 노력을 기울일 수 있습니다.



### Hengqin-RA-v1: Advanced Large Language Model for Diagnosis and Treatment of Rheumatoid Arthritis with Dataset based Traditional Chinese Medicin (https://arxiv.org/abs/2501.02471)
Comments:
          8 pages, 5 figures, AAAI-2025 Workshop

- **What's New**: 본 논문은 LLMs (Large Language Models)에서 나타나는 중국 맥락의 편향과 부정확성을 극복하기 위해, 류마티스 관절염(RA) 진단 및 치료를 위해 특별히 설계된 첫 번째 대형 언어 모델인 Hengqin-RA-v1을 발표합니다. 또한 고대 중국 의학 문헌과 현대 임상 연구를 바탕으로 한 RA 전용 데이터셋 HQ-GCM-RA-C1을 제시하여, 문화적 맥락을 고려한 정확한 응답을 가능하게 합니다. 이를 통해 기존 모델에서 발생하는 격차를 효과적으로 메꿉니다.

- **Technical Details**: Hengqin-RA-v1은 LLaMA-7B (Touvron et al. 2023) 기반으로 개발된 Huatuo2 (Zhang et al. 2023)의 고급 버전입니다. 이 모델은 중국 의학 지식 그래프(CMeKG) 및 GPT-3.5로 생성된 의료 지침 데이터를 이용해 교육받으며, RA 치료 및 진단에 특화된 문제 해결 능력을 향상시키는 데 초점을 맞추고 있습니다. 훈련 과정에서는 의학 기록을 구조화하여 TCM의 진단 및 치료 로직을 강화하는 방법이 사용됩니다.

- **Performance Highlights**: Hengqin-RA-v1은 RA 관련 진단 및 치료 정보를 생성하는 데 있어 현재의 최고 수준 모델들을 초월하는 성능을 보여줍니다. 일부 경우에는 인간 전문가의 진단 정확도를 초과하는 성취를 달성하여, TCM 분야에서의 발전을 더욱 두드러지게 했습니다. 또한 이 모델은 RA 관련 연구 및 응용 분야에서 잠재적인 혁신을 촉진할 것으로 기대됩니다.



### Towards Omni-RAG: Comprehensive Retrieval-Augmented Generation for Large Language Models in Medical Applications (https://arxiv.org/abs/2501.02460)
- **What's New**: 이번 연구에서는 헬스케어 관련 문제를 해결하기 위해 대규모 언어 모델(LLMs)에 외부 의료 지식을 통합하는 방법을 제안합니다. 특히, 기존의 접근방식이 소스 계획(source planning)을 간과하거나 비효율적으로 수행하는 한계를 극복하고자 하였습니다. 이를 위해 MedOmniKB라는 포괄적인 의료 지식 저장소를 설계하였습니다.

- **Technical Details**: MedOmniKB는 다장르(multi-genre) 및 다구조(multi-structured) 의료 지식 소스를 포함하는 데이터베이스입니다. 이 연구에서는 Source Planning Optimisation (SPO)이라는 방식을 사용하여 다양한 지식 소스를 효과적으로 활용하도록 지원합니다. 또한, 전문가 모델이 잠재적 계획을 탐색하고 평가할 수 있도록 하고, 더 작은 모델은 긍정적 및 부정적 계획 샘플을 통해 소스 정렬을 학습하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 본 연구 방법이 다원 소스 계획 성능을 크게 개선함을 입증하였습니다. 최적화된 소형 모델은 다양한 의료 지식 소스를 활용하는 데 있어 최첨단 결과를 달성했습니다. 이는 LLM의 의료 종합 능력을 향상시키는 데 중요한 기여를 할 것으로 기대됩니다.



### Understand, Solve and Translate: Bridging the Multilingual Mathematical Reasoning Gap (https://arxiv.org/abs/2501.02448)
Comments:
          18 pages, 14 figures, 9 tables

- **What's New**: 이 논문은 한국어 수학 추론의 성능 격차를 조명하는 HRM8K라는 새로운 벤치마크를 소개합니다. HRM8K는 8,011개의 한국어-영어 병렬 수학 문제로 구성되어 있어, LLM이 비영어 입력을 이해하는 데 어려움을 겪는 경향이 주요 원인임을 밝혀냈습니다. 이에 따라 미국식 수학 문제 해결을 위한 UST(Understand, Solve, and Translate) 방법론을 제안하여 성능 향상을 도모합니다.

- **Technical Details**: UST 방법론은 130,000개의 합성 데이터에 대해 모델을 미세 조정하여 HRM8K에서 10.91% 향상을 달성하고, 다국어 성능 격차를 11.6%에서 0.7%로 줄입니다. 이 방법은 영어를 기준으로 문제를 이해하고 해결책을 생성하는 전략적으로 접근하며, LLM이 멀티링구얼 입력을 처리하는 데 효과적임을 입증합니다. HRM8K 데이터셋은 한국의 경쟁 수준 문제와 기존 영어 화일의 병렬 번역을 포함한 신뢰할 수 있는 구조를 가지고 있습니다.

- **Performance Highlights**: UST 방법의 개선 사항은 수학뿐만 아니라 다양한 한국어 영역으로 일반화되는 것으로 나타났습니다. 이는 기계 검증 가능한 콘텐츠에서 습득한 능력이 다른 영역에도 일반화할 수 있음을 보여줍니다. UST 및 HRM8K 벤치마크는 한국어 수학적 추론의 체계적 평가를 가능하게 하며, 앞으로의 연구 방향을 제시합니다.



### Towards Multimodal Metaphor Understanding: A Chinese Dataset and Model for Metaphor Mapping Identification (https://arxiv.org/abs/2501.02434)
- **What's New**: 본 논문에서는 중국어 다중모달(Multimodal) 은유를 이해하기 위한 새로운 데이터셋인 CM3D(Chinese Multimodal Metaphor Mapping Dataset)를 소개하고 있습니다. 이 데이터셋은 브랜드 광고에서 약 6,108개의 텍스트-이미지 쌍을 포함하여 구체적인 타겟(target) 및 소스(source) 도메인에 대한 주석이 포함되어 있습니다. 또한, Chain-of-Thought(CoT) 프롬프트 기반 은유 매핑 식별 모델(CPMMIM)을 제안하여 은유의 인지 과정을 시뮬레이션합니다. 이러한 기여는 기존 연구에서 간과된 부분을 보완하고, 비영어권 언어에서 은유를 연구하는 데 기여할 것으로 기대됩니다.

- **Technical Details**: 이 연구는 은유 이해 과정을 개선하기 위해 CoT(reasoning) 및 Bi-Level Optimization(BLO) 개념을 활용한 CPMMIM 모델을 제안합니다. 이 모델은 은유의 타겟 및 소스 도메인을 식별하는 계층적 문제로 취급하여 보다 정확하고 해석 가능한 은유 매핑을 가능케 합니다. CM3D 데이터셋은 중국어 광고에서 유래된 은유 표현에 대한 구체적이며 철저한 주석이 포함되어 있으며, 이를 통해 다중모달 은유를 분석하는 기초 자료로 활용될 수 있습니다. 모델은 은유의 각 도메인에서 특정 단어를 추출하여 은유의 깊은 이해를 돕는 것을 목표로 합니다.

- **Performance Highlights**: CPMMIM 모델의 실험 결과는 은유의 타겟 및 소스 도메인을 효과적으로 식별하는 데 큰 성공을 거두었음을 보여주었습니다. CM3D 데이터셋은 다양한 기준 모델의 성능을 평가하기 위한 벤치마크로 기능하며, 향후 다중모달 은유 이해를 위한 지속적인 연구의 기초를 마련합니다. 이 연구는 그동안 한정된 다중모달 은유 자료를 보완하고, 은유적 표현의 생성 및 해석 메커니즘에 대한 통찰을 제공합니다. 공개된 데이터셋과 코드가 논문의 기여를 이어 나가는 데 중요한 자원으로 자리잡을 것입니다.



### Swift Cross-Dataset Pruning: Enhancing Fine-Tuning Efficiency in Natural Language Understanding (https://arxiv.org/abs/2501.02432)
Comments:
          Accepted by COLING 2025

- **What's New**: 본 논문에서는 Swift Cross-Dataset Pruning (SCDP)라는 새로운 접근법을 제안합니다. 이 방법은 TF-IDF 임베딩과 기하학적 중앙값을 활용하여 샘플의 중요도를 신속하게 평가합니다. 특히, 데이터셋 크기에 따라 적응형 프루닝을 적용해 더 작은 데이터셋의 경우 기하학적 중앙값에서 먼 샘플을 유지하고, 큰 데이터셋의 경우 거리 기반 층화 프루닝을 적용하여 다양성을 보장합니다.

- **Technical Details**: SCDP는 Frequency Distance (FD) 점수를 도입하여 샘플의 중요도를 신속하게 평가합니다. TF-IDF 임베딩을 사용하여 여러 NLU 작업과 도메인에서 단어의 의미적 중요성을 캡처하고, 기하학적 중앙값 계산을 통해 임베딩 공간에서의 중심성을 측정합니다. 또한, 샘플의 중요도를 평가할 때 모델 훈련이나 참조 모델 접근이 필요하지 않아 계산 효율성이 높습니다.

- **Performance Highlights**: 여섯 개의 다양한 데이터셋에서 수행한 실험 결과, 본 방법은 태스크와 스케일에 걸쳐 효과적임을 입증했습니다. SCDP는 기존의 방법보다 현저히 적은 계산 자원을 소모하면서 유사 혹은 더 나은 성능을 보여줍니다. 이러한 강점은 다양한 NLP 작업에서의 데이터 프루닝 접근법 개발에 기여할 것으로 기대됩니다.



### Anonymization by Design of Language Modeling (https://arxiv.org/abs/2501.02407)
- **What's New**: 이 논문에서는 민감한 정보의 노출 우려를 해결하기 위해 개인 정보 보호 중심의 언어 모델링 접근법을 제안합니다. Masking Language Modeling (MLM) 및 Causal Language Modeling (CLM) 기법을 활용하여 BERT 유사 모델과 GPT 유사 모델에서 직접 및 간접적으로 식별 가능한 정보를 기억하지 않도록 특화하였습니다. 이를 통해 의료 데이터 공유를 위한 안전한 언어 모델 개발의 가능성을 제시하고 있습니다.

- **Technical Details**: 저자들은 의료 데이터의 직접 및 간접 식별 정보를 식별하기 위해 PPmlm-bert와 PPclm-gpt라는 두 가지 세밀한 조정 접근 방식을 제안합니다. 이러한 접근법은 대칭적 식별 용어를 업데이트하지 않으며, 최소한 두 명의 개인의 문서에 사용된 단어만 기억하게 만들어 개인 정보의 재현 및 추론 위험을 줄입니다. 이 연구는 의료 보고서 데이터셋을 사용하여 기존 방법들과의 비교 평가를 수행하였습니다.

- **Performance Highlights**: 우리의 연구 결과는 직접 및 간접 식별 정보를 피하면서 여전히 높은 유용성을 유지할 수 있음을 보여줍니다. 모델의 성숙도를 기반으로 하여, 개인 정보의 유출 가능성을 상당히 줄이면서도 비식별 단어를 예측하는 능력이 향상되었습니다. 이러한 접근 방법은 향후 의료 데이터의 안전한 공유를 위한 중요한 기초를 마련하고 있습니다.



### Syntactic Evolution in Language Usag (https://arxiv.org/abs/2501.02392)
Comments:
          4 pages, 7 figures

- **What's New**: 이 연구는 생애 여러 단계에 걸친 언어 스타일의 동적 변화를 조사하는 것을 목표로 합니다. 2004년의 블로그 데이터를 사용하여 언어 사용이 시간이 지남에 따라 어떻게 변화하는지를 분석하고 있습니다. 연구 결과는 언어학(linguistics), 심리학(psychology), 그리고 커뮤니케이션 연구(communication studies)에 대한 통찰력을 제공할 수 있습니다.

- **Technical Details**: 연구 설계는 여러 단계로 이루어졌으며, 데이터 전처리(preprocessing), 문법적 특성(feature analysis), 그리고 GPT-4와 블로그 텍스트의 비교를 포함합니다. 문법적 요소(syntactic elements)의 다수의 비율과 비율을 분석하여 다양한 연령대에서의 차이를 명확히 하고자 하였습니다. 이 과정에는 OpenAI API를 통한 텍스트 생성 및 다양한 나이 그룹에 대한 특징을 분석하는 과정이 포함되었습니다.

- **Performance Highlights**: 결과적으로 블로그 텍스트의 문장 복잡성은 연령대가 증가함에 따라 증가하는 경향을 보였으며, GPT-4 텍스트에서도 거의 비슷한 변화를 관찰할 수 있었습니다. 그러나 나이 그룹 예측에서 모델 정확도가 낮아, GPT-4가 인간의 언어 진화에 대한 학습이 부족할 수 있음을 시사합니다. 또한, 다양한 데이터셋과 방법론이 필요함을 강조하며, 그런 이슈를 해결하기 위한 방안을 모색해야 한다고 결론지었습니다.



### Prepending or Cross-Attention for Speech-to-Text? An Empirical Comparison (https://arxiv.org/abs/2501.02370)
Comments:
          Submitted to ARR October 2024

- **What's New**: 본 연구는 Large Language Models (LLMs)의 음성 인식 능력을 향상시키기 위한 새로운 접근법인 Dense Feature Prepending (DFP)을 제안합니다. DFP는 텍스트 표현 앞에 음성 표현을 추가하여 음성 인코더와의 엔드 투 엔드 학습을 가능하게 합니다. 연구자들은 DFP와 전통적인 Encoder-Decoder 아키텍처 간의 성능 차이를 자세히 비교하고, 이를 통해 향후 연구 방향성을 제시합니다.

- **Technical Details**: 연구에서는 두 가지 주요 아키텍처인 DFP와 Cross-Attention을 비교하기 위해 직접 훈련한 모델을 사용했습니다. DFP는 성능 향상을 위해 음성 인코더와 함께 사용되며, CTC 압축, 시퀀스 수준 지식 증류 등 다양한 기법을 적용합니다. 실험은 MuST-C v1.0와 CoVoST2 데이터셋을 기반으로 진행되어, 단일 언어, 이중 언어 및 다국어 모델의 성능을 평가합니다.

- **Performance Highlights**: DFP는 전반적으로 Cross-Attention보다 좋은 품질을 나타내지 않는 것으로 보입니다. 그러나 동일한 음성 인코더를 사용했을 때 DFP는 ASR 및 ST 성능에서 더 유리하게 작용합니다. 이에 반해 Cross-Attention은 생성 속도와 GPU 메모리 효율성에서 더 우수한 결과를 보이며, DFP 모델의 인과성 속성에 대한 연구 결과 또한 중요성을 드러내고 있습니다.



### Context Aware Lemmatization and Morphological Tagging Method in Turkish (https://arxiv.org/abs/2501.02361)
- **What's New**: 본 연구는 터키어에서 단어의 의미와 문맥에 민감한 형태소 분석과 표제어 추출(lemmatization) 모델을 제안합니다. 형태소 태깅 모델과 표제어 추출 모델은 두 가지 대안 모델로 구성되어 있으며, 이 모델들은 터키어의 단어 의미를 기반으로 예측을 수행합니다. 그간 터키어의 의미에 민감한 표제어 연구는 없었기에, 본 논문은 새로운 기여를 하고 있습니다.

- **Technical Details**: 제안된 모델은 bidirectional LSTM과 터키어 BERT 모델을 활용하여 각각 단어의 철자와 의미를 표현합니다. 단어의 철자를 생성하기 위해 단방향 LSTM이 사용됩니다. 데이터셋으로는 Universal Dependencies의 IMST 및 PUD 데이터셋이 활용되며, 모델 훈련 후 SIGMORPHON 2019 대회의 결과와 비교하였습니다.

- **Performance Highlights**: 모델은 IMST 및 PUD 데이터셋에서 모두 최고 성능을 달성하였으며, 거의 모든 평가 지표에서 우수한 결과를 보였습니다. 본 연구는 두 가지 모델의 성능을 비교하고, 문맥에 기반하여 의미를 반영한 새로운 표제어 추출 방법의 효용성을 입증하였습니다.



### Thinking with Many Minds: Using Large Language Models for Multi-Perspective Problem-Solving (https://arxiv.org/abs/2501.02348)
Comments:
          36 pages, 1 appendix

- **What's New**: 이 논문은 복잡한 문제 해결을 위한 새로운 접근법인 '합성 심의(synthetic deliberation)'를 제안합니다. 이는 다양한 관점을 짊어진 에이전트 간의 담화를 시뮬레이션하는 대규모 언어 모델(Large Language Model, LLM) 기반의 방법론입니다. 전통적인 정신적 시뮬레이션의 한계를 극복하면서 개별적으로 '다양한 마음으로 생각하는' 능력을 배양합니다.

- **Technical Details**: 제안된 방법은 사용자 정의된 GPT 기반 모델을 활용하여 다양한 관점을 동시 처리할 수 있는 능력을 보여줍니다. 이는 인지적 저하 없이 여러 관점을 동시에 탐색하고, 관점 합성을 정확히 제어할 수 있도록 합니다. 합성 심의는 병렬 검색과 통합을 통해 인지 작업을 분산시켜 정신적 시뮬레이션의 제약을 초월합니다.

- **Performance Highlights**: 이 접근법은 전략적 계획(strategic planning), 정책 결정(policy-making), 갈등 해결(conflict resolution)과 같은 분야에서 잠재력을 보입니다. 다양한 관점을 고려할 수 있는 유연성을 제공하며, 이는 의사결정 과정에서 매우 중요한 역할을 할 것으로 기대됩니다.



### AdaSkip: Adaptive Sublayer Skipping for Accelerating Long-Context LLM Inferenc (https://arxiv.org/abs/2501.02336)
Comments:
          9 pages,10 figures, AAAI

- **What's New**: 최근 대규모 언어 모델(LLM)의 긴 문맥 추론(long-context inference) 지원이 향상되면서, 더욱 복잡한 실제 응용 프로그램이 가능해졌습니다. 그러나 긴 문맥 추론에서는 높은 계산 및 저장 요구가 발생합니다. 이를 해결하기 위해 이 논문에서는 적응형 서브레이어 스키핑 방식인 	extit{AdaSkip}을 제안하여, 기존의 레이어 스키핑 기법이 가지는 한계를 극복하고 있습니다.

- **Technical Details**: 	extit{AdaSkip}는 실행 중 유사성 정보를 활용하여 덜 중요한 레이어를 식별하고, 서브 레이어 단위의 스키핑을 가능하게 하여, 프리필링(prefilling) 및 디코딩(decoding) 단계 모두에서 성능을 향상시킵니다. 또한 각각의 서브 레이어가 가지는 중요도 분포를 독립적으로 평가하여, 보다 효율적인 스키핑 전략을 제공합니다. 이러한 방식은 레이어의 고정된 스키핑에 의한 생성 품질 저하 문제를 해결합니다.

- **Performance Highlights**: 다양한 긴 문맥 벤치마크와 모델에 대한 포괄적인 실험을 통해 	extit{AdaSkip}의 우수한 성능이 입증되었습니다. 본 논문은 기존의 기법들에 비해 생성 품질을 향상시키며, 특히 중요한 서브 레이어를 우선적으로 건너뛰는 전략을 통해 긴 문맥 추론 시 시간 및 메모리 오버헤드를 크게 줄였습니다.



### Validity Arguments For Constructed Response Scoring Using Generative Artificial Intelligence Applications (https://arxiv.org/abs/2501.02334)
Comments:
          33 pages, 2 figures, 6 tables; This work was presented at the 2024 meeting of the International Testing Commission in Granada, Spain

- **What's New**: 이 논문은 고배율 시험(high-stakes testing)에서의 생성적 인공지능(generative AI) 사용의 가능성을 탐구합니다. 특히, 생성적 AI가 인간 평가와 전통적인 AI 평점 방식보다 더 효과적일 수 있다는 점을 강조합니다. 기존 기능 기반(feature-based) AI 평점 시스템과의 비교를 통해 생성적 AI의 장점을 조명하고, 이에 대한 유효성 증거(validity evidence) 수집을 위한 최선의 관행을 제안합니다.

- **Technical Details**: 연구에서는 생성적 AI 방식의 평가에서 투명성 부족과 일관성(consistency) 문제와 같은 독특한 우려 사항들을 다룹니다. 생성적 AI는 기능 기반 자연어 처리(NLP) AI 평가 엔진보다 더 많은 유효성 증거가 필요합니다. 표준화된 시험에서 수집된 구성 응답(score) 데이터는 서로 다른 평가 시스템의 유효성 증거를 보여주며, 이러한 점에서 여러 복잡성과 고려 사항을 부각시킵니다.

- **Performance Highlights**: AI 평가 점수가 인간 평가에 비해 어떻게 합쳐질 수 있는지를 논의하며, 다양한 출처의 AI 점수를 결합한 기여 기반(contributory scoring) 접근 방식을 고려합니다. 이러한 접근 방식은 인간 평가 없이도 구성 요소(construct)의 더 많은 부분을 포괄할 수 있습니다. 이 논문은 생성적 AI를 활용한 평가 시스템의 신뢰성 문제에 대한 논의도 포함하고 있습니다.



### Explicit vs. Implicit: Investigating Social Bias in Large Language Models through Self-Reflection (https://arxiv.org/abs/2501.02295)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)의 명시적(bias) 및 암묵적(bias) 편향을 비교하고 조사하기 위해 사회 심리학 이론에 기반한 체계적인 프레임워크를 제안합니다. 특히, '자기 반성(self-reflection)' 기반의 평가 방법론을 통해 모델이 생성한 내용을 스스로 분석하도록 유도하여 명시적 편향을 평가합니다. 이 연구는 이전의 연구들이 명시적 편향에만 초점을 맞춘 한계를 극복하고, LLMs의 편향을 보다 포괄적으로 이해할 수 있는 기회를 제공합니다.

- **Technical Details**: 연구는 LLMs의 명시적 편향을 Self-Report Assessment(SRA) 방법으로 측정하고, 암묵적 편향은 Implicit Association Test(IAT)로 측정합니다. LLMs의 자가 반성을 통한 분석을 통해 동일한 표적에 대한 명시적과 암묵적 편향 간의 비교를 가능하게 합니다. 데이터 세트는 ChatGPT, Claude-3.5, LLaMA 등 다양한 모델을 포함하고, 성별, 인종, 직업 등 여러 사회적 차원이 실험에 포함되었습니다.

- **Performance Highlights**: 실험 결과, 모든 LLMs가 명시적과 암묵적 편향에서 불일치를 보여주며, 명시적 수준에서는 최소한의 고정관념이 드러나지만 암묵적 수준에서는 심각한 고정관념이 나타났습니다. 연구는 훈련 데이터 양과 모델 크기가 명시적 편향 감소와 관련이 있지만, 암묵적 편향은 오히려 증가하는 경향이 있음을 발견했습니다. 현대의 정렬 기법은 명시적 편향을 억제하지만 암묵적 편향 경감에는 한계가 있음을 나타냅니다.



### LLMzSz{\L}: a comprehensive LLM benchmark for Polish (https://arxiv.org/abs/2501.02266)
- **What's New**: 이 논문에서는 폴란드어를 위한 최초의 포괄적인 벤치마크인 LLMzSzŁ(LLMs Behind the School Desk)를 소개합니다. 이 벤치마크는 폴란드 중앙 시험 위원회의 아카이브에서 추출한 학술 및 전문 시험을 포함하는 4가지 유형의 시험으로 구성되며, 총 19,000개의 객관식 문제로 이루어져 있습니다. 연구에서는 열린 소스 다국어 모델과 폴란드어 LLM의 성능을 비교하여 언어 간 지식 전달 능력을 확인합니다.

- **Technical Details**: LLMzSzŁ 데이터셋은 단일 신뢰할 수 있는 출처인 폴란드 중앙 시험 위원회의 시험 자료를 기반으로 작성되었습니다. 데이터셋은 중학교, 고등학교, 직업 시험 등 복잡성과 관련된 다양한 층으로 나누어져 있으며, 원래 폴란드어로 작성된 문항들로 구성되어 있습니다. 이 벤치마크는 향후 시험의 난이도를 테스트하는 데에도 사용될 수 있습니다.

- **Performance Highlights**: 연구 결과, 다국어 LLM이 단일 언어 모델보다 우수한 성능을 보이는 경향을 보였습니다. 하지만 모델 크기가 중요한 경우에는 단일 언어 모델이 유리할 수 있습니다. 우리는 LLM이 시험 검증을 지원하고, 특히 시험 작업에서의 이상이나 오류 식별에 잠재력이 있음을 강조합니다.



### Financial Named Entity Recognition: How Far Can LLM Go? (https://arxiv.org/abs/2501.02237)
Comments:
          Accepted at The Joint Workshop of the 9th Financial Technology and Natural Language Processing (FinNLP), the 6th Financial Narrative Processing (FNP), and the 1st Workshop on Large Language Models for Finance and Legal (LLMFinLegal), in conjunction with COLING 2025

- **What's New**: 이 연구는 최신 LLM(대형 언어 모델)의 금융 분야에서의 명명된 개체 인식(NER) 작업을 평가한 최초의 포괄적인 연구입니다. 연구에서는 다양한 프롬프트 기법에 대한 성능을 비교하고 LLM의 장단점 및 문제점을 확인하였습니다. 또한 실패 유형을 다섯 가지로 구분하고 이론적 기초를 마련하여 향후 연구 방향을 제시합니다.

- **Technical Details**: 연구에서는 세 가지 주요 LLM인 GPT-4o, LLaMA-3.1, Gemini-1.5를 사용하였으며, 직접 프롬프트, 컨텍스트 학습(in-context learning), 체인 오브 씽킹(chain-of-thought) 프롬프트의 세 가지 프롬프트 기법을 적용하여 실험을 수행하였습니다. 데이터셋으로는 FiNER-ORD를 사용하였으며, 이를 통해 LLM의 퍼포먼스를 측정하기 위한 다양한 기준과 방법론을 제시합니다. 실험 결과는 주로 F1 점수와 가중된 F1 점수를 활용하여 평가되었습니다.

- **Performance Highlights**: 실험 결과, 최신 LLM의 성능은 파인튜닝 모델보다 떨어졌으나, 다양한 프롬프트 디자인과 모델 크기에 따라 성과 차이를 줄일 수 있음을 보여주었습니다. 체인 오브 씽킹 프롬프트는 LLM의 성능에 제한적 영향을 미치며 경우에 따라 성능을 저하시킬 수도 있습니다. Gemini 시리즈가 FiNER-ORD 작업에서 다른 모델보다 우수한 성능을 발휘하는 것을 확인하였으며, 이는 모델 크기와 프롬프트 디자인의 영향을 받아 나타나는 결과입니다.



### Survey on Question Answering over Visually Rich Documents: Methods, Challenges, and Trends (https://arxiv.org/abs/2501.02235)
- **What's New**: 본 논문은 Large Language Models (LLMs)을 활용한 Visually-rich Document Understanding (VrDU)의 발전과 도전 과제를 설명합니다. LLMs가 통합된 VrDU 모델들이 문서의 이해와 생성을 어떻게 개선하는지, 특히 다양한 형태의 정보 처리를 어떻게 수행하는지를 다룹니다. 기존의 문서 이해 방식에서 벗어나, 복합적인 과제를 해결하는 새로운 접근 방식에 초점을 맞추고 있습니다.

- **Technical Details**: VRDs(Visually-rich Documents)는 텍스트와 그래픽, 도표 및 표와 같은 시각적 요소가 결합된 복잡한 정보를 포함합니다. 이 논문은 VRDs를 인코딩하는 다양한 방법, 문서의 텍스트, 레이아웃 및 시각적 데이터를 통합하는 인코딩 기술을 탐구합니다. 여러 모달리티 간의 정보 결합과 상대적 위치 정보를 고려하는 방식이 제시됩니다.

- **Performance Highlights**: LLMs를 통해 VrDU 모델들이 복잡한 질문 응답 작업에 뛰어난 성능을 보임을 강조합니다. 각종 인코딩 방식과 인베딩 통합 전략의 발전으로 VRD와 관련된 다양한 유형의 정보 처리 및 해석에서 향상된 결과를 보여줍니다. 시각적 요소와 텍스트 속성의 정교한 통합이 이루어져, 구체적인 응용에서 더 나은 효과를 기대할 수 있습니다.



### CPTuning: Contrastive Prompt Tuning for Generative Relation Extraction (https://arxiv.org/abs/2501.02196)
- **What's New**: 이번 연구에서는 다중 관계 추출(multi-relation extraction)의 한계를 극복하기 위해 새로운 대비 프롬프트 튜닝 방법인 CPTuning을 소개합니다. 기존의 관계 추출(relation extraction, RE) 기법들이 두 엔티티 간 하나의 결정론적 관계만을 가정했던 것에 반해, CPTuning은 엔티티 쌍 간 여러 관계를 유연하게 다룰 수 있도록 설계되었습니다. CPTuning은 관계의 존재 여부에 따라 확률 질량을 조정하는 인사이트를 제공함으로써, 더 높은 성과를 달성하고 있습니다.

- **Technical Details**: CPTuning은 RE를 Seq2Seq 텍스트 채우기(text-infilling) 방식으로 재구성하여, 특정 프리픽스에서 시작되는 후보 관계를 생성합니다. 이 과정에서 텍스트 템플릿을 사용하여 엔티티의 관계를 마스킹한 샘플을 입력받아, 정해진 임계값 이상 혹은 이하의 확률에 따라 후보 관계를 생성하게 됩니다. 또한, Trie 구조를 활용하여 생성된 관계의 유효성을 보장하며, 적응형 빔 검색(prefix-given beam search)을 통해 검증된 후보 관계를 최종적으로 추출합니다.

- **Performance Highlights**: CPTuning을 적용한 T5-large 모델은 네 가지 널리 사용되는 데이터셋에서 기존 방법들을 능가하는 성능을 보였습니다. 특히, 다중 관계 추출이 가능한 능력 덕분에 기존의 단일 관계 추출 방식에 비해 유연하고 강력한 결과를 도출할 수 있었습니다. 이번 연구 결과는 CPTuning의 접근 방식이 관계 표현에서 의미 정보를 효과적으로 포착한다는 점에서 중요한 기여를 하고 있습니다.



### Personalized Graph-Based Retrieval for Large Language Models (https://arxiv.org/abs/2501.02157)
- **What's New**: 본 연구에서는 Personalized Graph-based Retrieval-Augmented Generation (PGraphRAG)이라는 새로운 프레임워크를 제안합니다. 이는 사용자 중심의 지식 그래프를 활용하여 사용자 개개인에게 맞춘 텍스트 생성을 개선합니다. 특히, PGraphRAG는 사용자 이력이 부족한 상황에서도 개인화된 응답을 생성할 수 있는 능력을 가지고 있습니다.

- **Technical Details**: PGraphRAG는 사용자 정보를 구조적으로 표현하는 지식 그래프를 통해 개인화된 텍스트 생성을 지원합니다. 이를 통해 모델은 사용자의 컨텍스트와 선호도를 더 깊이 이해하고, 보다 적절한 출력 결과를 만드는 데 기여합니다. 또한, 우리는 12개의 개인화된 텍스트 생성 과제를 평가하기 위한 Personalized Graph-based Benchmark를 소개하여, 실제 환경에서 사용자 이력을 바탕으로 한 기능을 세밀히 측정할 수 있도록 하고 있습니다.

- **Performance Highlights**: 실험 결과, PGraphRAG는 다양한 작업에서 최신 개인화 기법들을 능가하는 성능을 보여줍니다. 이는 사용자의 맞춤형 정보를 효과적으로 통합하여 더욱 향상된 개인화 출력 결과를 생성할 수 있음을 입증합니다. 이 연구는 텍스트 생성 및 개인화 기술 분야에서 중요하고 실용적인 기여를 하고 있습니다.



### Applying Text Mining to Analyze Human Question Asking in Creativity Research (https://arxiv.org/abs/2501.02090)
Comments:
          24 pages, 15 figures; accepted to International Conference on Big Data Analytics in Astronomy, Science and Engineering 2024, Aizu, Japan

- **What's New**: 이번 연구는 창의성(creativity) 연구에서 질문 분석의 역할을 명확히 이해하기 위한 새로운 접근 방식을 제시합니다. 질문의 종류, 복잡성(complexity), 답변의 내용(content)을 고려하여, 질문이 창의성에 미치는 영향의 인지적 잠재력을 측정하는 텍스트 마이닝(text mining) 방법을 적용했습니다. 또한, 자연어 처리(NLP) 방법을 사용하여 질문과 창의성 간의 관계를 연구하려는 시도를 통해, 향후 연구에 필요한 방향성을 제시하고자 합니다.

- **Technical Details**: 연구는 질문의 다양한 유형을 분류하고, 질문의 복잡성을 평가하기 위해 Bloom의 분류법(Bloom's taxonomy)을 활용합니다. 질문 분석을 통해 창의적인 문제 해결 과정에서 질문이 어떻게 문제 정의와 정보 수집을 지원하는지를 탐구합니다. 또한, 자동 자연어 처리(NLP) 기술의 현재 상태를 검토하면서, 질문의 창의성을 측정할 수 있는 방법론을 개발하려고 합니다.

- **Performance Highlights**: 실험 결과, 질문의 복잡성이 창의성에 긍정적인 상관관계를 보이는 것으로 나타났습니다. 복잡한 질문이 더 상세한 아이디어를 생성하는 경향이 있으며, 질문을 통해 창의적인 답변으로 이어지는 과정을 밝히는 데 도움이 됩니다. 이러한 결과는 NLP 기술이 창의성 연구에 중요한 역할을 할 수 있음을 시사합니다.



### Instruction-Following Pruning for Large Language Models (https://arxiv.org/abs/2501.02086)
Comments:
          13 pages, 3 figures

- **What's New**: 본 논문은 기존의 정적 가지치기(static pruning) 접근 방식에서 벗어나 입력에 따라 동적으로 조정되는 가지치기(masking) 방식인 'Instruction-Following Pruning(IFPruning)'을 제안합니다. 이 approach는 sparse mask predictor를 도입하여 사용자 지시에 따라 가장 관련성 높은 모델 파라미터를 선택할 수 있도록 하며, 다양한 작업에 맞춰 입력 의존적인 pruning mask를 자동으로 생성합니다.

- **Technical Details**: IFPruning 방법은 피드포워드 신경망 계층의 구조적 가지치기에 중점을 두며, 전체 행(row) 또는 열(column)을 가지치기합니다. 이 방법에서는 사용자의 입력을 sparsity predictor에 전달하여 각 계층의 행과 열에 중요도를 부여하고, 이를 이용해 미분 가능한 마스크를 생성하여 모델에 적용합니다. 결과적으로, 딥러닝 모델과 sparse mask predictor를 공동 최적화하여 사실상 필요한 파라미터만을 활성화하는 방식입니다.

- **Performance Highlights**: 실험 결과, IFPruning 방법은 math, coding, tool use 등의 다양한 작업에서 3B 밀집 모델에 비해 평균 5-8포인트 향상된 성능을 기록했습니다. 특히, 9B 모델을 3B로 가지치기할 경우, coding 작업에서 8% 향상된 성능을 보이며 9B 모델의 성능 감소는 극히 미미하게 나타났습니다. 이러한 결과는 IFPruning의 효율성과 높은 성능의 균형을 잘 보여줍니다.



### The interplay between domain specialization and model size: a case study in the legal domain (https://arxiv.org/abs/2501.02068)
- **What's New**: 이 연구는 기존의 언어 모델(LM) 규모 최적화에 관한 연구가 단순히 모델 크기와 토큰 수에 관한 관찰에 국한되어 있음을 지적합니다. 저자들은 계속적인 사전 훈련(continual pre-training)을 통해 기존 모델의 지식을 활용해 데이터를 효율적으로 활용할 수 있는 가능성을 제시합니다. 연구 결과, 도메인 특화 모델이 일반 모델보다 높은 성능을 보이며, 동일한 컴퓨팅 자원으로 더 나은 효율성을 발휘하는 경향이 발견되었습니다.

- **Technical Details**: 이 연구에서는 1.5B, 3B, 7B, 14B 매개변수를 가진 언어 모델을 사용하여 법률 도메인의 전문화된 데이터셋과 일반 데이터셋으로 훈련했습니다. 데이터셋 필터링 기법을 사용하여 법률 관련 데이터만 추출하였으며, 법률 시험에 대한 모델 성능을 평가했습니다. 모델 크기가 증가함에 따라 도메인 특화 모델과 일반 모델 간의 compute-effectiveness 차이가 더욱 두드러진 것으로 나타났습니다.

- **Performance Highlights**: 특화된 모델이 동일한 자원 내에서 일반 모델보다 우수한 성능을 발휘했으며, 특히 14B 매개변수 특화 모델이 일반 모델보다 4.3배 적은 컴퓨팅을 사용하면서 뛰어난 성능을 기록했습니다. 이러한 결과는 도메인 특화 훈련이 특정 도메인에서 모델의 성능을 극대화할 수 있음을 시사합니다. 전체적으로, 이 연구는 도메인 전문가의 수요를 충족하기 위한 학습 방안으로서 계속적인 사전 훈련의 가능성을 혁신적으로 탐구합니다.



### AGGA: A Dataset of Academic Guidelines for Generative AI and Large Language Models (https://arxiv.org/abs/2501.02063)
Comments:
          arXiv admin note: text overlap with arXiv:2406.18842, arXiv:2501.00959

- **What's New**: 이 연구에서는 Generative AIs(GAIs)와 Large Language Models(LLMs)의 학문적 활용을 위한 80개의 학술 가이드라인으로 구성된 AGGA 데이터셋을 소개합니다. 이 데이터셋은 공식 대학 웹사이트에서 수집된 자료로, 188,674개의 단어로 구성되어 있으며 자연어 처리(Natural Language Processing) 작업과 요구 사항 엔지니어링에서 유용한 리소스입니다.

- **Technical Details**: AGGA 데이터셋은 모델 합성(model synthesis), 추상화 식별(abstraction identification), 문서 구조 평가(document structure assessment)와 같은 여러 요구사항 엔지니어링의 작업에 적용될 수 있습니다. 또한, 이 데이터셋은 모호성 탐지(ambiguity detection), 요구 사항 분류(requirements categorization), 동등 요구 사항 식별(equivalent requirements identification) 등의 작업을 위한 벤치마크로 추가 주석이 가능하도록 설계되었습니다.

- **Performance Highlights**: 이 연구는 6개 대륙의 다양한 대학을 대표하는 대학들을 선택하여 철저한 검토를 수행하였습니다. AGGA 데이터셋은 인문학, 기술 분야 및 공공 및 민간 기관을 포함하여 여러 학문 분야의 관점을 반영하고 있어 GAIs와 LLMs의 학문적 통합에 대한 폭넓은 통찰력을 제공합니다.



### Advancing Pancreatic Cancer Prediction with a Next Visit Token Prediction Head on top of Med-BER (https://arxiv.org/abs/2501.02044)
- **What's New**: 최근에 개발된 Med-BERT라는 EHR(전자 건강 기록) 특정 기반 모델을 활용하여 질병 예측을 위한 새로운 방법론이 제안되었습니다. 이 연구는 질병 이진 예측 작업을 토큰 예측 작업으로 재구성하여 Med-BERT의 사전 학습(task format) 목표에 맞추었습니다. 이러한 접근법은 특히 매우 작은 세부 조정(cohorts) 집단에서의 모델 활용 최적화에 기여할 수 있습니다.

- **Technical Details**: 연구에서는 Med-BERT-Sum과 Med-BERT-Mask라는 두 가지 새로운 모델을 도입하였습니다. Med-BERT-Sum은 토큰 예측 작업을 통해 소량의 데이터에서도 우수한 성능을 보이는 반면, Med-BERT-Mask는 다음 방문 마스크 토큰 예측 작업을 사용하여 기존의 이진 분류(binary classification) 작업보다 3%에서 7% 더 나은 결과를 나타냈습니다. 특히, 데이터 크기가 10에서 500 샘플인 경우에 이 능력이 더욱 두드러집니다.

- **Performance Highlights**: 이 연구의 주요 발견은 다운스트림 작업을 Med-BERT의 사전 훈련(pretraining) 목표와 일치시킴으로써 모델의 예측 능력이 크게 향상되었다는 점입니다. 이 접근법은 드문 질병 및 일반 질병 예측에 있어 효과적이며, 특히 췌장암(PaCa)에 대한 조기 발견과 시기적절한 개입을 가능하게 합니다. 결과적으로 이러한 기술은 치료 효과, 생존률 및 환자 결과 개선에 기여할 것으로 기대됩니다.



### An Investigation into Value Misalignment in LLM-Generated Texts for Cultural Heritag (https://arxiv.org/abs/2501.02039)
- **What's New**: 최근 대규모 언어 모델(LLMs)이 문화 유산 관련 작업에서 사용됨에 따라, 정확하고 문화적으로 일치하는 텍스트 생성의 필요성이 커지고 있습니다. 그러나, 연구에 따르면 생성된 텍스트에서 문화적 가치의 불일치가 발생하고 있으며, 이는 역사적 사실의 왜곡이나 문화 정체성의 침해 등 심각한 결과를 초래할 수 있습니다. 본 논문은 LLMs가 생성한 문서에서의 문화적 가치 불일치 문제를 체계적으로 조사하여, 그 심각성을 드러내고자 합니다.

- **Technical Details**: 이 연구에서는 1066개의 쿼리 작업으로 구성된 벤치마크 데이터셋을 구축하고, 5개의 널리 인정된 카테고리와 17개의 측면을 평가하여 LLMs의 문화적 가치 불일치 유형과 비율을 분석했습니다. 자동화와 수동 평가 방식을 결합하여, 생성된 텍스트에서의 문화적 불일치를 효과적으로 탐지하고 분석했습니다. 초기 연구 결과에 따르면, 상당수의 생성된 텍스트에서 65% 이상이 뚜렷한 문화적 불일치를 나타내는 것으로 확인되었습니다.

- **Performance Highlights**: 대규모 언어 모델이 생성한 텍스트는 다양한 문화적 가치의 불일치 문제를 드러내며, 전체 분석 대상 1000개 작업 중 약 65%가 영향을 받는 것으로 나타났습니다. 이는 문화적으로 민감한 분야에서 LLMs의 신뢰성을 높이기 위한 개선된 방법론의 필요성을 강조합니다. 이 연구는 문화적 민감성과 신뢰성을 향상시키기 위한 귀중한 자원인 공개 데이터 세트를 제공하여, 향후 관련 연구에 기여할 것으로 기대됩니다.



### CarbonChat: Large Language Model-Based Corporate Carbon Emission Analysis and Climate Knowledge Q&A System (https://arxiv.org/abs/2501.02031)
Comments:
          26 pages

- **What's New**: 이 논문은 CarbonChat: 대규모 언어 모델 기반의 기업 탄소 배출 분석 및 기후 지식 Q&A 시스템을 제안합니다. 이 시스템은 복잡한 문제에 대한 기존의 증강 생성 아키텍처의 전문성과 정확성 부족 문제를 해결하는 데 중점을 두고 있습니다. 또한 탄소 배출 보고서를 분석하는 데 드는 시간과 비용을 줄이기 위해 다양한 데이터 인덱싱 모듈을 개발했습니다.

- **Technical Details**: CarbonChat은 의도 인식(intent recognition), 구조적 추론 체인(structured reasoning chains), 하이브리드 검색(hybrid retrieval), 및 Text2SQL 기술을 통합하여 생성의 효율성을 높입니다. 이 시스템은 온실가스 회계 프레임워크를 기반으로 14개의 차원으로 탄소 배출 분석을 수행하여 맞춤형 응답을 제공하며, 다층 청크(chunking) 메커니즘을 통해 결과의 정확도와 검증 가능성을 보장합니다.

- **Performance Highlights**: 이 시스템은 사용자에게 정확하고 포괄적인 정책 및 규제 참조를 제공하며, 보고서 요약 및 관련성 평가를 통해 기업의 지속 가능성 보고서에 대한 심층 분석을 수행합니다. 특히, 다양한 인덱싱 모듈과 환각 탐지 기능을 통해 결과의 정확성과 검증 가능성을 크게 향상시켰습니다.



### Recursive Decomposition of Logical Thoughts: Framework for Superior Reasoning and Knowledge Propagation in Large Language Models (https://arxiv.org/abs/2501.02026)
- **What's New**: 이번 연구에서는 LLM의 추론 능력을 획기적으로 향상시킬 수 있는 새로운 프레임워크인 RDoLT(Recursive Decomposition of Logical Thought prompting)를 소개합니다. RDoLT는 복잡한 추론 작업을 점진적으로 단순화하는 서브태스크로 분해하고, 유망한 추론 결과를 식별하기 위한 고급 선택 및 평가 메커니즘을 사용하며, 인간 학습을 모방한 지식 전파 모듈을 통합하여 향상된 성능을 제공합니다.

- **Technical Details**: RDoLT는 세 가지 주요 혁신에 기반합니다: (1) 복잡한 추론 작업을 점진적으로 단순해지는 하위 작업으로 재귀적으로 분해하고, (2) 논리적 유효성, 일관성, 단순성 및 적응성의 4가지 차원을 기준으로 사고를 평가하는 강력한 사고 평가 시스템을 도입하며, (3) 이전에 거부된 사고를 저장 및 전파하여 잠재적인 가치를 재탐색할 수 있는 지식 전파 모듈(Knowledge Propagation Module, KPM)을 제공합니다.

- **Performance Highlights**: RDoLT는 다양한 벤치마크에서 LLM의 성능을 유의미하게 향상시켰으며, 특히 GSM8K에서 ChatGPT-4가 90.98%의 정확도로 기존 최고 기술보다 6.28% 높은 성능을 달성했습니다. 다른 벤치마크에서도 유사한 개선을 보였으며, 정확도 향상이 5.5%에서 6.75%에 이르는 것으로 나타났습니다. 이러한 결과는 복잡한 추론 작업에 대한 RDoLT의 효과적인 접근 방식을 강조합니다.



### Enhancing Uncertainty Modeling with Semantic Graph for Hallucination Detection (https://arxiv.org/abs/2501.02020)
- **What's New**: 이번 연구에서는 LLMs(대형 언어 모델)에서 발생하는 환각(hallucination) 문제를 해결하기 위해, 기존의 단어마다의 불확실성(uncertainty)만을 고려하는 방법과는 달리, 의미 그래프(semantic graph)를 이용한 불확실성 모델링을 제안합니다. 연구자들은 엔티티 토큰(entity tokens)과 문장 간의 관계를 효과적으로 포착하는 의미 그래프를 구축하였으며, 문장 수준의 환각 탐지를 강화하기 위해 두 엔티티 간의 관계를 통합하여 불확실성을 전파하는 방법을 개발했습니다.

- **Technical Details**: 연구의 핵심 접근 방식은 두 가지로 요약됩니다. 첫째, AMR(Abstract Meaning Representation) 기반 파싱을 통해 각 문서의 의미 그래프를 생성합니다. 둘째, 이 그래프를 활용하여 문장과 이웃 사이의 관계를 포함한 불확실성을 보정하여, 더욱 정교한 환각 탐지 방법을 제공합니다. 이를 통해 문장 수준 및 패시지 수준에서 보다 정교한 불확실성 계산을 수행할 수 있습니다.

- **Performance Highlights**: 두 개의 데이터셋(WikiBio 및 NoteSum)을 통해 수행된 실험 결과, 제안된 접근 방식이 문장 및 패시지 수준의 환각 탐지에서 기존 방법들보다 19.78%의 향상된 성능을 보임으로써 그 우수성을 입증하였습니다. 이는 의미 그래프를 활용한 새로운 불확실성 모델링 기법이 환각 탐지의 정확성을 현저히 향상시킬 수 있음을 시사합니다.



### Safeguarding Large Language Models in Real-time with Tunable Safety-Performance Trade-offs (https://arxiv.org/abs/2501.02018)
- **What's New**: 본 연구에서는 Large Language Models (LLMs)의 안전성을 향상시키기 위한 새로운 방법인 SafeNudge를 소개하고 있습니다. 이 방법은 Controlled Text Generation (CTG)과 'nudging'을 결합하여 모델이 위험한 출력을 생성하는 것을 실시간으로 방지하는 데 중점을 두고 있습니다. SafeNudge는 jailbreak 공격이 발생한 후 활성화되어 LLM이 안전한 응답으로 유도되도록 도와줍니다.

- **Technical Details**: SafeNudge는 LLM의 안전성과 성능 간의 trade-off를 조절할 수 있는 장치를 제공합니다. 이 방법은 jailbreak 공격에 대한 성공적인 시도를 30% 감소시키며, 일반적인 모델 행동에는 5%의 악화만을 초래합니다. 또한, SafeNudge는 Hugging Face의 transformers 라이브러리에 호환되어 연구자들에게 쉽게 사용될 수 있습니다.

- **Performance Highlights**: SafeNudge는 모델의 텍스트 생성 지연을 최소화하면서도 해로운 응답 생성을 효과적으로 줄이는 성능을 보여주었습니다. 예를 들어, 기본 설정에서 NVIDIA A100 GPU를 사용할 때 안전하지 않은 응답 생성을 30.4% 감소시키는 것으로 나타났습니다. 전반적으로 SafeNudge는 상대적으로 합리적인 Safety-Performance Trade-offs (SPTs)를 통해 강력한 안전성을 제공하는 것을 확인했습니다.



### Cross-model Transferability among Large Language Models on the Platonic Representations of Concepts (https://arxiv.org/abs/2501.02009)
- **What's New**: 이 논문은 다양한 대규모 언어 모델(LLMs) 간의 개념 표현에 대한 새로운 접근 방식을 제시하며, 이는 플라톤의 동굴의 비유와 유사한 관계를 탐구합니다. 연구진은 LLM 간의 개념 표현을 간단한 선형 변환(Linear Transformation)을 이용해 효과적으로 정렬할 수 있음을 발견했습니다. 또한, 이 연구는 소형 LLM에서 추출된 쉬운 steering vector (SV)가 대형 LLM의 행동을 효과적으로 제어할 수 있음을 밝혀내었습니다.

- **Technical Details**: 연구에서는 L-Cross Modulation이라는 선형 변환 방법론을 제안하며, 이 방법은 LLM 간의 개념 공간을 정렬하고 SV의 이식성을 달성하는 데 도움을 줍니다. 이 방법은 일반 최소 제곱 회귀(Ordinary Least Squares Optimization)를 통해 생성된 변환 행렬(T)을 사용하여, 원본 LLM의 SV를 목표 LLM의 표현 공간으로 매핑하는 방식을 취합니다. 이 과정을 통해, 연구진은 11개의 벤치마크 개념을 기반으로 cross-model transferability 능력을 평가했습니다.

- **Performance Highlights**: 연구 결과 L-Cross Modulation은 LLM을 조정하는 데 효과적임을 보여주었으며, 예를 들어 해롭다는 개념을 적용했을 때 90%의 출력에서 해로운 콘텐츠 생성을 유도했습니다. 또한, 다양한 개념 간 선형 변환의 강한 일반화 능력이 있으며 서로 다른 개념이 두 LLM 간 동일한 선형 변환을 공유할 수 있음을 발견하였습니다. 마지막으로, 소형 LLM의 SV가 대형 LLM의 응답을 효과적으로 조정할 수 있는 가능성이 있음을 확인했습니다.



### Automated Generation of Challenging Multiple-Choice Questions for Vision Language Model Evaluation (https://arxiv.org/abs/2501.03225)
Comments:
          Project page: this https URL

- **What's New**: 비전 언어 모델(Vision Language Models, VLMs)의 빠른 발전을 반영하여, 본 논문에서는 AutoConverter라는 새로운 프레임워크를 도입하여 열린 질문을 객관적인 선택형 질문으로 변환합니다. 이 접근은 문제 생성 과정을 줄이며 평가의 일관성을 높이는 데 기여합니다. AutoConverter는 자동으로 정확하고 도전적인 선택형 질문을 생성할 수 있는 다중 에이전트 시스템으로 설계되었습니다.

- **Technical Details**: AutoConverter는 다양한 에이전트가 협력하여 열린 질문을 선택형으로 변환하는 방식으로 작동합니다. 특히, 정확성을 보장하기 위해 질문의 정확성을 평가하는 에이전트가 있어, 생성된 선택지의 적합성을 검증합니다. 이를 통해 20개의 기존 VQA 데이터셋에서 9,018개의 질문으로 구성된 VMCBench라는 새로운 벤치마크를 생성하였습니다.

- **Performance Highlights**: 33개의 최첨단 VLM을 VMCBench에서 평가한 결과, AutoConverter가 생성한 질문이 인간이 제작한 질문에 비해 유사하거나 더 낮은 정확도를 보이면서도 높은 도전성을 유지함을 보여주었습니다. 이러한 결과는 AutoConverter의 다용도성을 입증하며, 교육 및 기존 선택형 데이터셋의 개선에도 활용할 수 있는 가능성을 제공합니다.



### ChronoSense: Exploring Temporal Understanding in Large Language Models with Time Intervals of Events (https://arxiv.org/abs/2501.03040)
Comments:
          14 pages, 2 figures

- **What's New**: ChronoSense는 LLMs의 시간 이해력 평가를 위한 새로운 벤치마크로, Allen의 간격 관계를 종합적으로 테스트하는 중요한 기반을 제공합니다. 16개의 다양한 작업으로 구성되어 있으며, 추상적인 사건과 실제 데이터를 사용하여 LLM의 성능을 분석합니다. 이를 통해 시간 관련 문제에 대한 모델의 메모리 의존성을 탐구하고 있으며, 이는 LLM의 성능 개선 필요성을 강조합니다.

- **Technical Details**: ChronoSense 데이터셋은 사건을 튜플(tuple) 형태로 정의하며, Allen 관계에 대한 문제와 단일 사건을 기반으로 한 시간 산술 문제를 포함합니다. LLMs의 성능을 평가하기 위해 0-shot, few-shot, 그리고 chain-of-thought (CoT) 프롬프트 시나리오에서 분석합니다. 각 사건은 이벤트의 이름, 시작 및 종료 시간, 기간, 빈도를 포함하며, Allen의 구간 대수는 13 가지의 관계를 제공합니다.

- **Performance Highlights**: 모델은 Allen 관계, 특히 'Equals'와 'Finishes'와 같은 특정 관계에 대해 지속적으로 낮은 성과를 보였습니다. 반면 few-shot 및 CoT 프롬프트가 시간 산술 과제에서 성과를 개선하는 데 효과적임을 보여주었습니다. 전반적으로 모델의 낮은 성과는 LLM의 시간 이해력을 개선할 필요가 있음을 나타내며, ChronoSense는 이 분야의 미래 연구에 중요한 프레임워크를 제공합니다.



### Analyzing Fine-tuning Representation Shift for Multimodal LLMs Steering alignmen (https://arxiv.org/abs/2501.03012)
Comments:
          The first three authors contributed equally

- **What's New**: 이 논문은 다중모달 대형 언어 모델(Multimodal LLMs, MLLMs)의 내부 기제를 이해하고 설명하는 데 초점을 맞추고 있습니다. 기존 연구들이 모델의 최종 상태만을 분석했던 것과 달리, 저자들은 훈련 과정에서 발생하는 은닉 상태 표현의 진화를 체계적으로 분석합니다. 이를 통해 모델의 내부 구조가 새로운 다중모달 작업에 맞게 어떻게 전문화되는지를 밝혀냅니다.

- **Technical Details**: 저자들은 개념 기반 접근 방식을 사용하여 은닉 상태를 해석 가능한 시각적 및 텍스트 개념에 매핑합니다. 이 방법론을 통해 훈련 과정에서 인코딩된 개념의 변화 추적이 가능하며, shift vectors를 통해 원래 모델의 개념을 보완하거나 재구성할 수 있음을 보여줍니다. 이를 통해 MLLMs의 행동을 조정할 수 있는 실용적인 영향을 탐구합니다.

- **Performance Highlights**: 연구 결과, 학습된 개념들이 특정 훈련 작업에 맞춰 어떻게 조정되는지를 발견했습니다. 저자들은 원래 모델의 개념을 기준으로 shift vectors를 사용하여 많은 fine-tuned 개념을 재구성할 수 있음을 보여주며, MLLMs의 응답을 추가 훈련 없이 수정할 수 있는 가능성을 입증합니다. 이러한 방식은 MLLMs의 조정 가능성을 탐구하는 첫 번째 연구로, 모델 출력의 방향성을 조절하는 데 획기적인 통찰을 제공합니다.



### CALM: Curiosity-Driven Auditing for Large Language Models (https://arxiv.org/abs/2501.02997)
Comments:
          Accepted by AAAI 2025 AI Alignment Track

- **What's New**: 이 연구에서는 블랙박스(black-box) 대규모 언어 모델(LLM)의 감사(auditing) 방법을 개발했습니다. 특히, 모델의 파라미터에 접근하지 않고 서비스로 제공되는 LLM의 입력-출력 쌍을 자동으로 발견하는 최적화를 목표로 하고 있습니다. 이를 통해 비윤리적이거나 위험한 행위를 보이는 입력을 찾는 데 중점을 두고 있습니다.

- **Technical Details**: 제안된 방법인 Curiosity-Driven Auditing for Large Language Models (CALM)은 강화 학습(reinforcement learning)을 활용하여 감사 LLM을 조정합니다. CALM은 다양한 감사 프롬프트를 생성하여 모델이 특정한 비하적 또는 민감한 반응을 유도할 수 있도록 설정되어 있습니다. 이 접근법은 입력 공간의 희소성을 활용하여 숨겨진 행동을 발견하는 데 집중합니다.

- **Performance Highlights**: CALM은 여러 LLM에서의 다양한 실험을 통해 문제가 있는 행동을 효율적으로 식별하는 성과를 보였습니다. 이 과정에서 상대적으로 작은 모델인 GPT-2의 미세 조정만으로도 Llama-3-8B와 같은 대형 모델의 부정적 행동을 발견할 수 있음을 보여줍니다. 이는 LLM의 잠재적 위험을 강조하며, 호기심 기반의 RL 접근법이 감사를 위해 중요함을 입증합니다.



### GeAR: Generation Augmented Retrieva (https://arxiv.org/abs/2501.02772)
- **What's New**: 이번 연구에서는 GeAR (Generation-Augmented Retrieval)라는 새로운 문서 검색 방법을 제안합니다. GeAR은 질의와 문서의 융합 표현을 기반으로 관련 텍스트를 생성하여 세밀한 정보에 집중할 수 있도록 설계되었습니다. 이 방법은 기존의 bi-encoder 구조를 사용하면서도 계산 부담을 추가하지 않으며, 높은 품질의 데이터를 효율적으로 합성할 수 있는 파이프라인을 구축하였습니다.

- **Technical Details**: GeAR은 질의-문서-정보 트리플 구조를 사용하여 지도학습을 통해 질의와 문서 간의 유사성을 최적화합니다. 이 때 텍스트 디코더를 설계하여 질의에 맞는 관련 세부 정보를 생성함으로써 검색 및 로컬라이제이션 기능을 향상시킵니다. 기존의 bi-encoder는 전체 문서의 전역 의미를 강조하지만, GeAR은 fine-grained understanding이 필요한 복잡한 작업에 적합하도록 이 설계를 변경했습니다.

- **Performance Highlights**: GeAR은 다양한 시나리오와 데이터셋에서 경쟁력 있는 검색 및 로컬라이제이션 성능을 보여주었습니다. 실험 결과 GeAR은 질의와 문서에 기반하여 관련 정보를 생성할 수 있는 능력을 발휘하며, 전통적인 검색 과정에 대한 새로운 관점을 제공합니다. 뿐만 아니라, 코드와 데이터, 모델을 기술 검토 후 공개하여 향후 연구를 촉진할 계획입니다.



### MBTSAD: Mitigating Backdoors in Language Models Based on Token Splitting and Attention Distillation (https://arxiv.org/abs/2501.02754)
Comments:
          Accepted by ICTAI 2024

- **What's New**: 최근 주목(attention) 기반 모델들이 다양한 분야에서 뛰어난 성과를 보이고 있지만, 백도어 공격(backdoor attacks)에 취약하다는 점이 지적되었습니다. 이 논문에서는 미리 훈련된(pre-trained) 가중치 없이도 백도어를 완화할 수 있는 MBTSAD라는 새로운 기법을 제안합니다. MBTSAD는 소량의 클린 데이터만을 이용하여 백도어 모델을 다시 훈련시키고, 주의(distillation) 기법을 통해 백도어를 제거합니다.

- **Technical Details**: MBTSAD의 핵심은 토큰 분할(token splitting) 기법을 사용하여 생성된 데이터셋에서 백도어 모델을 재훈련하는 것입니다. 이 과정에서 MBTSAD는 Attention Distillation 기법을 채택하여, 재훈련된 모델을 '교사 모델'로, 원래의 백도어 모델을 '학생 모델'로 설정합니다. 이를 통해 백도어 활성화 패턴의 차이를 파악하고, 클린 데이터에 대한 성능을 유지하면서 백도어를 효과적으로 완화할 수 있습니다.

- **Performance Highlights**: 실험 결과, MBTSAD는 오직 20%의 클린 데이터만을 사용하면서도 미리 훈련된 가중치를 의존하는 기존 방법들과 유사한 효과적인 백도어 완화 성능을 보여줍니다. 또한, 클린 데이터에서의 성능 저하 없이 작동하여 다양한 상황에서 유용성을 높였습니다. 이러한 접근법은 향후 NLP 모델 보안 연구에 중요한 기여를 할 것으로 기대됩니다.



### KG-CF: Knowledge Graph Completion with Context Filtering under the Guidance of Large Language Models (https://arxiv.org/abs/2501.02711)
Comments:
          6 pages

- **What's New**: 이 논문에서는 지식 그래프 완성을 위한 새로운 프레임워크인 KG-CF를 제안합니다. 기존의 LLM(대규모 언어 모델) 기반 연구는 주로 이진 분류에 초점을 맞추어 왔으며, 효율적인 순위 기반 태스크에는 충분히 적용되지 않았습니다. KG-CF는 LLM의 추론 능력을 활용하여 부적절한 맥락을 걸러내고, 실제 데이터셋에서 우수한 성과를 달성하도록 설계되었습니다.

- **Technical Details**: KG-CF 프레임워크는 (h, r, t) 형태의 삼중(triplet) 지식 그래프를 처리하며, 그래프의 미리 정의된 경로를 샘플링하여 관련성을 평가합니다. LLM은 필터링된 맥락 정보에 접근하여 순위 점수를 생성하는 데 사용됩니다. 또한, KG-CF는 특정 서열 분류기를 경량화하여 계산 비용을 줄이고 LLM의 직접 활용을 피함으로써 효율성을 높입니다.

- **Performance Highlights**: 실험 결과, KG-CF는 다양한 현실 세계의 KG 데이터셋에서 다른 대안 모델들보다 탁월한 성과를 기록했습니다. 이 프레임워크는 그래프의 특성을 효과적으로 활용하면서도 순위 기반 평가, 즉 실제 사용 사례에 보다 적합하게 설계되었습니다. KG-CF의 성능 향상은 효율성과 유연성을 가지고 높은 신뢰성으로 이어집니다.



### Generalizing from SIMPLE to HARD Visual Reasoning: Can We Mitigate Modality Imbalance in VLMs? (https://arxiv.org/abs/2501.02669)
- **What's New**: 이 연구에서는 Vision Language Models (VLMs)의 알고리즘적 시각적 추론 능력을 평가하기 위한 새로운 합성 프레임워크를 제안합니다. VLMs가 시각적 질문 응답 및 이미지 캡셔닝에서 눈부신 성과를 보이지만, 복잡한 이미지에 대한 다단계 추론에는 한계가 있음을 강조합니다. 이 프레임워크에서는 세 가지 과제 - Table Readout, Grid Navigation, Visual Analogy -를 두 가지 난이도로 나누어 성능을 평가합니다.

- **Technical Details**: 제안된 프레임워크에서는 SIMPLE과 HARD 두 가지 난이도의 과제에 대한 훈련 전략을 개발합니다. 우리는 SIMPLE 버전에서 훈련하여 HARD 과제에 대한 성능 향상을 목표로 하는 S2H 일반화 전략을 탐색합니다. 또한, 이미지-텍스트 변환의 명확성이 S2H 일반화 촉진에 중요한 역할을 한다는 점을 강조하며, 다양한 훈련 전략에 따른 결과를 분석합니다.

- **Performance Highlights**: 실험 결과, S2H 일반화에 적합한 훈련 전략을 사용했을 때 S2H 일반화 성능이 유의미하게 향상되었음을 보여줍니다. 그라디언트 정렬(gradient alignment)을 포함한 메커니즘적 연구를 통해, 특정 훈련 전략이 더 나은 S2H 일반화를 촉진하는 경향이 있음을 밝혀냈습니다. 추가적인 ablation 연구에서 각각의 실험 설계 결정이 성능에 미치는 영향을 측정하였으며, 이는 미래의 연구 및 모델 설계에 중요한 기초 자료로 활용될 수 있습니다.



### Layer-Level Self-Exposure and Patch: Affirmative Token Mitigation for Jailbreak Attack Defens (https://arxiv.org/abs/2501.02629)
- **What's New**: 이 논문에서는 Layer-AdvPatcher라는 새로운 방법론을 제안한다. 이 접근법은 LLMs의 특정 레이어를 패치(patch)하여 jailbreak 공격에 대항하는 방어를 설계한다. 특히, 해로운 프롬프트에 직면했을 때 긍정적인 토큰(affirmative tokens)을 생성하는 경향이 있는 레이어를 식별하여, 이를 통해 모델의 안전성을 강화하고자 한다.

- **Technical Details**: Layer-AdvPatcher는 자기 증강 데이터셋(self-augmented datasets)을 활용하여 LLMs의 특정 레이어를 패치하는 비학습 전략(unlearning strategy)을 적용한다. 이 방법을 통해 해로운 데이터를 발생시키는 레이어를 적대적으로 노출하여 그들의 취약성을 이해하고, 이후 이 문제들을 '비학습(unlearn)'하여 긍정적인 토큰의 영향을 줄인다. 이러한 과정은 모델의 안전성 질의를 효과적으로 유지하면서도 jailbreak 공격으로 인한 위험을 최소화하는 데 목적이 있다.

- **Performance Highlights**: 두 개의 모델과 네 개의 벤치마크 데이터셋을 사용한 광범위한 실험을 통해 이 방법론의 효율성을 입증하였다. 실험 결과, 최근의 방어 방법들과 비교하여 해로운 공격의 성공률을 낮추면서도 무해한 질의에 대한 유용성을 훼손하지 않음을 보여준다. 이를 통해 Layer-AdvPatcher는 LLM의 안전성을 증진시키는 효과적인 방안으로 판단된다.



### Efficient Architectures for High Resolution Vision-Language Models (https://arxiv.org/abs/2501.02584)
Comments:
          Accepted to COLING 2025

- **What's New**: Pheye는 고해상도 이미지를 효율적으로 처리하면서 부족한 매개변수로 훈련되는 새로운 비전-언어 모델(Vision-Language Model, VLM) 아키텍처입니다. 이 모델은 기존의 VLM보다 성능은 유지하면서도 더 높은 효율성을 제공합니다. 특히, 세밀한 이미지 이해 또는 장면 텍스트 처리와 같은 작업에 강점을 보이고 있습니다.

- **Technical Details**: Pheye는 고정된 세팅의 언어 모델과 CLIP 기반의 비전 인코더를 결합하며, 이들 사이의 dense cross-attention 레이어를 통해 상호작용합니다. 비전 인코더는 전역 이미지 및 지역 고해상도 패치를 처리하는 두 가지 LoRA 어댑터 세트를 사용합니다. 모델 설계에서는 효과적인 convergence를 위해 LayerNorm을 사용하며, cross-attention을 통합하여 훈련 가능한 매개변수가 줄어듭니다.

- **Performance Highlights**: Pheye는 가격 대비 경쟁력 있는 모델로서, TextVQA와 같은 장면 텍스트 관련 작업에서 특히 돋보이는 성능을 나타냅니다. 이 모델은 훈련 매개변수가 적음에도 불구하고 높은 해상도의 이미지를 효과적으로 처리하여 다양한 리소스 제한 환경에서도 활용 가능성을 제시합니다. 성능 향상을 위해 자체적으로 제공되는 훈련된 모델과 코드의 GitHub 저장소도 공개되어 있습니다.



### LeetDecoding: A PyTorch Library for Exponentially Decaying Causal Linear Attention with CUDA Implementations (https://arxiv.org/abs/2501.02573)
Comments:
          The source code of LeetDecoding is hosted at this https URL

- **What's New**: LeetDecoding는 먼저 발표된 Python 패키지로, 생성적 사전 훈련 변환기(GPT)에서 원래의 causal attention을 대체하기 위해 exponentially decaying causal linear attention을 사용합니다. 이 패키지는 다양한 계산 루틴을 포함하여 LLM 전문가들에게는 새로운 계산 방법을 벤치마킹하고 평가할 수 있는 기회를 제공합니다.

- **Technical Details**: LeetDecoding는 기존의 linear-attention LLM과 쉽게 통합되도록 설계되었으며, CUDA 구현을 통해 GPU에서 빠른 추론이 가능합니다. 이로써 LLM 연구자들은 복잡한 GPU 프로그래밍이나 복잡성 분석에 대한 지식 없이도 사용할 수 있도록 의도적으로 접근성을 높였습니다.

- **Performance Highlights**: LeetDecoding의 출시는 이 연산자에 대한 명확한 이해 부족과 기존 계산 방법의 포괄적 수집이 필요하다는 점에 의해 촉발되었습니다. 사용자는 단순히 'pip install leet-decoding' 명령어를 통해 LeetDecoding을 설치할 수 있으며, GitHub 저장소에서 소스 코드도 제공됩니다.



### Decoding fMRI Data into Captions using Prefix Language Modeling (https://arxiv.org/abs/2501.02570)
Comments:
          4 pages, 2 tables, 1 figure

- **What's New**: 최근 대형 언어 모델과 잠재 확산 모델의 발전으로 뇌 신호를 해독하는 연구가 눈에 띄는 성과를 거두었습니다. 본 연구에서는 기존 GIT 모델의 데이터 오염 가능성을 해결하기 위해 DINOv2 모델의 임베딩을 예측하여 뇌 신호를 이미지 캡션으로 변환하는 새로운 방법을 제안합니다. 이 접근 방식은 GPT-2 언어 모델의 입력으로 [CLS] 토큰을 사용하여 계산 요구 사항을 크게 줄입니다.

- **Technical Details**: 우리는 fMRI 신호에서 DINOv2의 임베딩을 직접 예측하는 새로운 방법을 채택하고, 3D Convolutional Neural Networks (CNN)를 통해 고차원 데이터를 처리합니다. 이를 통해 ROI 마스크 외부의 정보와 복셀 간의 포지셔널 정보를 보다 잘 고려할 수 있습니다. 각 모듈은 별도로 학습되며, DINOv2 임베딩은 MSE 손실을 사용하여 진행됩니다.

- **Performance Highlights**: 우리의 접근법은 기존의 COCO 캡션과 비교하여 METEOR 메트릭에서 우수한 성능을 보여주었으며, 다른 이미지 임베딩으로부터 생성된 캡션과 비교했을 때 6가지 메트릭 중 4가지에서 우수한 결과를 기록하였습니다. 또한, Ridge Regression보다 Wide CNN 아키텍처의 성능이 모든 메트릭에서 우수함을 확인하여 뇌 신호의 해독 효율성을 제고했습니다.



### Towards New Benchmark for AI Alignment & Sentiment Analysis in Socially Important Issues: A Comparative Study of Human and LLMs in the Context of AGI (https://arxiv.org/abs/2501.02531)
Comments:
          20 pages, 1 figure

- **What's New**: 이 연구는 다양한 대형 언어 모델(Large Language Models, LLMs)의 사회적 중요 문제에 대한 감정(sentiment) 평가 벤치마크를 설정하는 데 기여하려고 합니다. 이는 LLM이 사회에 미치는 장기적인 영향을 평가하는 중요한 측면입니다. 특히, GPT-4와 Bard를 포함한 7개의 LLM에 대한 감정 점수를 비교하는 데 초점을 맞추었습니다.

- **Technical Details**: 연구 방법으로는 리커트 척도(Likert scale) 설문조사를 사용하였으며, 세 개의 독립적인 인간 샘플 집단의 감정 데이터를 분석했습니다. 결과적으로, 감정 점수는 LLM 간에 다양성을 보이며 5점 만점에 3.32에서 4.12까지 나왔습니다. 연구는 또한 감정의 시간적 변화를 3일에 걸쳐 평가했습니다.

- **Performance Highlights**: 결과에 따르면, GPT-4는 AGI에 대한 가장 긍정적인 감정 점수를 기록하였고, Bard는 중립적인 경향을 보였습니다. 반면, 인간 샘플의 평균 감정 점수는 2.97로 낮은 수치를 보였으며, LLM의 감정 변동은 1.03%에서 8.21% 사이의 차이를 나타냈습니다. 이 연구는 LLM의 감정 형성에서 잠재적인 이해 충돌과 편향 가능성을 제시하며, LLM이 인간의 인지 과정과 유사하게 독특한 감정을 발전시킬 가능성이 있음을 시사합니다.



### Test-time Computing: from System-1 Thinking to System-2 Thinking (https://arxiv.org/abs/2501.02497)
Comments:
          work in progress

- **What's New**: 본 논문은 o1 모델의 복잡한 추론 기능 향상을 통한 테스트 시간 컴퓨팅(test-time computing) 스케일링의 중요성을 강조합니다. 특히, System-1에서 System-2로의 전환을 조명하며, 테스트 시간 컴퓨팅이 이 과정에서 핵심 역할을 한다고 설명합니다. 접근 방식은 반복 샘플링, 자기 수정 및 트리 탐색 등의 세 가지 전략을 통해 이루어집니다.

- **Technical Details**: System-1 및 System-2 사고는 심리학적 개념으로, 각각의 모델은 인간의 인지 과정을 반영합니다. System-1 모델은 빠르고 직관적인 반응을 나타내는 반면, System-2 모델은 복잡한 문제를 해결하기 위한 느리고 깊이 있는 사고를 요구합니다. 본 논문에서는 이러한 모델들이 깊은 학습 구조 내에서 어떻게 작용하는지, 특히 테스트 시간에서의 컴퓨팅 기법에 대해 설명합니다.

- **Performance Highlights**: 최근 o1 모델은 복잡한 문제 해결에서 교차 검증 및 적응성을 보여준 사례로 주목받고 있습니다. 높은 성능을 자랑하는 LLMs가 System-2 사고의 기초를 다지면서, 이러한 모델들은 심층적인 사고를 가능하게 하고 있습니다. 그러나 여전히 누적 에러의 문제와 같은 한계가 존재하며, 향후 발전 방향에 대한 논의도 포함되어 있습니다.



### LLMPC: Large Language Model Predictive Contro (https://arxiv.org/abs/2501.02486)
- **What's New**: 최근 대규모 언어 모델(LLM)의 프롬프트 기법 발전이 추론, 계획, 행동 능력을 향상시켰습니다. 본 논문은 모델 예측 제어(MPC) 관점에서 이러한 프롬프트 기법을 분석합니다. 프롬프트 계획을 활용할 때 LLM이 암묵적인 계획 비용 함수 최소화기로 작용한다는 것을 보여주며, 실제 계획 비용 함수와 평가자를 통합하여 LLM 계획 성능을 더욱 향상시킬 수 있음을 입증합니다.

- **Technical Details**: 모델 예측 제어 설정에서 에이전트는 상태 공간을 탐색해야 하며, 이를 통해 행동 순서를 결정합니다. 에이전트는 주어진 목표를 최소화하는 일련의 행동을 계획하며, 이 과정에서 목표 함수의 복잡성과 작업 관련 비용을 고려합니다. LLM은 입력 토큰 시퀀스를 기반으로 다음 토큰에 대한 확률 벡터를 생성하며, 광범위한 최적화 기법을 사용하여 최적의 행동을 선택합니다.

- **Performance Highlights**: 구성된 프롬프트는 LLM의 관리를 MPC로 간주할 수 있으며, 명시적인 목표 함수를 사용함으로써 LLM 플래너의 성능을 향상시킬 수 있음을 보여줍니다. 본 연구는 동적 시스템 제어 및 코드 생성 문제에서 접근 방식의 다양한 측면을 시연합니다. 각 실험의 코드는 GitHub에서 확인할 수 있습니다.



### A Statistical Hypothesis Testing Framework for Data Misappropriation Detection in Large Language Models (https://arxiv.org/abs/2501.02441)
Comments:
          29 pages, 5 figures

- **What's New**: 이 연구는 최근의 큰 언어 모델(LLM)들이 훈련 데이터에 저작권 자료를 무단으로 포함하는 문제를 다룹니다. 저작권 있는 훈련 데이터에 워터마크(embedding watermarks)를 삽입하여 데이터 유용성을 탐지하는 새로운 방법론을 제시합니다. 제안된 방법은 가설 검정(hypothesis testing) 문제로 프레임화되어, 잘못된 데이터 이용 감지를 수학적으로 다루는 혁신적인 접근 방식을 포함하고 있습니다.

- **Technical Details**: 연구에서는 통계적 테스트 프레임워크(statistical testing framework)를 기반으로 한 가설 검정 방법을 개발합니다. 두 가지 가설, 즉 워터마크가 있는 데이터를 사용하지 않은 경우와 있는 경우에 따른 토큰(token)과 비밀키(secret key)의 의존성을 평가합니다. 배경 지식으로 활용되는 NTP(다음-토큰 예측) 분포를 두고, 파생된 통계량을 정립하고 최적의 기각 임계값(rejection threshold)을 설정하여 제1종 및 제2종 오류를 제어합니다.

- **Performance Highlights**: 이 방법론은 실제 데이터 세트를 통해 수치 실험(numerical experiments)으로 효과성을 입증하고, 워터마크 데이터를 사용한 LLM 훈련 내의 데이터 유용성을 식별하는데 높은 정확도를 보입니다. 연구는 특히 부분 상속(partial inheritance)에 대한 최적성 보장을 새롭게 설정하여, 기존 연구에서 미처 다루지 못했던 부분을 설명합니다. 제안된 검정 방법은 모든 다른 테스트 방법에 비해 가장 높은 검정력을 달성하는 것으로 나타났습니다.



### Efficient Deployment of Large Language Models on Resource-constrained Devices (https://arxiv.org/abs/2501.02438)
- **What's New**: 이 논문은 FedSpine이라는 새로운 Federated Learning (FL) 프레임워크를 제안하여 리소스가 제한된 기기에서의 대규모 언어 모델(LLM) 배포 문제를 해결합니다. FedSpine은 파라미터 효율적인 미세 조정(PEFT)과 구조적 프루닝(structured pruning)을 결합하여 성능 저하 없이 더 빠르고 효율적인 미세 조정을 가능하게 합니다. 또한, 온라인 Multi-Armed Bandit (MAB) 알고리즘을 사용하여 기기의 자원 소모 및 성능 간의 관계를 적응적으로 학습합니다.

- **Technical Details**: LLM의 미세 조정 과정에서 FedSpine은 각 기기가 로컬 데이터로 사전 훈련된 모델을 미세 조정하도록 하며, 주기적으로 서버에서 업데이트된 LoRA 가중치를 다운로드합니다. 이 과정에서 기기 간의 이질성을 고려하여, 각 기기별로 적절한 프루닝 비율과 LoRA 랭크를 할당하여 미세 조정 효율을 극대화합니다. FedSpine은 기기가 제한된 자원 상황에서도 높은 추론 정확도를 유지하는 데 도움을 줍니다.

- **Performance Highlights**: 실험 결과, FedSpine은 80개의 NVIDIA Jetson 기기를 활용한 플랫폼에서 기존 방법에 비해 미세 조정을 1.4배에서 6.9배까지 속도 향상시키고, 최종 정확도는 0.4%에서 4.5%까지 개선했습니다. 이러한 성능 향상은 특히 리소스가 제한된 기기에서의 LLM 활용을 더욱 용이하게 만들어, 다양한 다운스트림 작업에 적합한 모델을 제공하는 데 기여합니다.



### Scaling Laws for Floating Point Quantization Training (https://arxiv.org/abs/2501.02423)
- **What's New**: 본 논문은 부동 소수점 양자화(floating-point quantization) 훈련에서 훈련 성능에 미치는 다양한 요소를 세밀하게 분석합니다. 이전 연구들은 주로 정수 양자화(integer quantization)에 집중했으나, 우리는 부동 소수점 양자화의 구성 요소들, 즉 지수 비트(exponent bits) 및 가수 비트(mantissa bits)의 영향을 깊이 있게 탐구합니다. 본 연구에서는 최적의 비트 비율과 낮은 정밀도(Low-precision) LLM 훈련에 대한 중요한 발견을 제공합니다.

- **Technical Details**: 부동 소수점 양자화에 대한 새로운 스케일링 법칙(scaling law)을 제시하며 이를 통해 데이터 크기(D)와 모델 크기(N) 등 다양한 요소가 훈련 손실(training loss)에 미치는 영향을 설명합니다. 특히, 지수와 가수가 LLM 성능에 미치는 비율적 기여를 밝혀내고, 낮은 정밀도 훈련에서의 '지식 집약도(knowledge intensity)' 및 '정밀도 정보 손실(low precision information loss)'을 구체적으로 다룹니다. 실험을 통해 우리는 4-8비트 범위의 최적 부동 소수점 양자화 정밀도를 제시합니다.

- **Performance Highlights**: 성능 측면에서 지수 비트가 가수 비트에 비해 모델 성능에 더 크게 기여하는 것으로 나타났습니다. LLM의 프리트레이닝(pre-training) 데이터 크기가 지나치게 크면 성능 저하를 초래할 수 있으며, 반대로 대형 모델과 높은 정밀도 설정은 효과적인 훈련 데이터를 증가시키는 데 기여합니다. 낮은 정밀도의 훈련은 '지식 집약도'와 비례하여 부정적인 영향을 배가시킬 수 있음을 알게 되었습니다.



### Who Wrote This? Zero-Shot Statistical Tests for LLM-Generated Text Detection using Finite Sample Concentration Inequalities (https://arxiv.org/abs/2501.02406)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLM)에서 생성된 텍스트와 인간이 작성한 텍스트 사이를 구별하는 새로운 방법을 제시합니다. 포괄적인 위상에 기초한 통계적 테스트를 설계하여, 특정 텍스트가 내부 LLM A 또는 외부 LLM B(혹은 인간)에서 생성된 것인지 구별할 수 있는 방법을 탐구하고 있습니다. 이 연구에서 정의된 통계적 테스트는 이론적 보장을 제공합니다.

- **Technical Details**: LLM이 생성한 텍스트를 순차적 확률 과정으로 모델링하고, 주어진 텍스트에 대해 LLM A와 B로부터 생성되었는지를 판단하기 위한 제로샷(zero-shot) 통계 테스트를 설계합니다. 이 과정에서 우리는 log-perplexity와 모형 A의 문자열의 평균 엔트로피 간의 집중 경계를 도출하여, 텍스트의 길이가 증가함에 따라 오류 유형 I과 II가 기하급수적으로 감소함을 입증합니다. 또한, 생성된 텍스트의 출처를 고유하게 식별하는 것이 가능성을 보여줍니다.

- **Performance Highlights**: 실험 결과는 제안된 테스트가 이론적으로 보장된 정확성을 통해 LLM이 생성한 해로운 텍스트의 출처를 식별하는 데 도움을 줄 수 있음을 시사합니다. 세부적인 예는 텍스트 크기가 커질수록 정확도가 동시에 증가한다는 점에서, 이 연구는 반드시 필요한 도구로 자리잡을 것으로 기대됩니다. 이런 점에서, 해당 연구는 정보의 진위를 파악하고 변별력을 높이는 데 기여할 것입니다.



### Graph-Aware Isomorphic Attention for Adaptive Dynamics in Transformers (https://arxiv.org/abs/2501.02393)
- **What's New**: 이번 연구에서는 Transformer 아키텍처에 그래프 지향의 관계적 추론(graph-aware relational reasoning)을 통합하여 수정하는 접근 방법을 제안합니다. 주의(attention) 메커니즘을 그래프(graph) 연산으로 재정의하여 Graph-Aware Isomorphic Attention을 제안합니다. 이는 Graph Isomorphism Networks (GIN)와 Principal Neighborhood Aggregation (PNA)와 같은 고급 그래프 모델링 전략을 활용하여 관계 구조를 더 풍부하게 표현합니다.

- **Technical Details**: 우리의 접근법은 복잡한 의존성을 캡처하고 다양한 작업에 걸쳐 일반화(generalization)되는 것을 목표로 합니다. Sparse GIN-Attention이라는 미세 조정(fine-tuning) 방법을 도입하여 희소(의존성 그래프) GIN을 사용하고, 주의 행렬을 희소 인접 그래프(sparse adjacency graphs)로 해석하여 사전 훈련된 기본 모델의 적응력을 높입니다. 이 방법은 그래프 기반의 모델을 통해 연결은 강화되고, 이를 통해 이전보다 더 나은 훈련 동역학(training dynamics)과 일반화 성능을 달성합니다.

- **Performance Highlights**: Sparse GIN-Attention 미세 조정 방법은 기존의 저차원 적응(low-rank adaptation, LoRA) 방법보다 개선된 결과를 보입니다. 이 연구는 전통적인 주의 메커니즘 내에는 잠재적인 그래프 유사 구조가 존재함을 논의하며, Transformer를 계층적 GIN 모델로 발전시키는 관점을 제시합니다. 이는 기초 모델 개발에 대한 깊은 영향을 미치며, 지역(local) 및 글로벌(global) 의존성에 동적으로 적응할 수 있는 아키텍처 설계를 가능하게 합니다.



### Guiding Medical Vision-Language Models with Explicit Visual Prompts: Framework Design and Comprehensive Exploration of Prompt Variations (https://arxiv.org/abs/2501.02385)
- **What's New**: 이 논문은 MedVP라는 새로운 프레임워크를 도입하여 의료 이미지를 위한 시각적 프롬프트 생성 및 파인튜닝을 수행합니다. 이는 기존의 일반적인 Vision-Language Models(VLMs)에서 세부적인 정보에 대한 집중이 부족하다는 문제를 해결합니다. MedVP는 의료 개체를 추출하고, 이를 기반으로 시각적 프롬프트를 생성하며, 교육 데이터셋을 조정하여 시각적 프롬프트에 기반한 파인튜닝을 수행합니다.

- **Technical Details**: 이 프레임워크는 입력 이미지에서 의료 개체를 자동으로 추출하고, 이를 활용하여 관심 영역(ROI)을 나타내는 시각적 프롬프트를 생성합니다. 생성된 프롬프트는 박스, 스크리블, 원 등의 다양한 형식으로 불러올 수 있습니다. 이 논문에서는 MedVP-LLaVA라는 모델을 개발하여, 의료 이미지 질의응답(Visual Question Answering, VQA)에서 성능 개선을 평가합니다.

- **Performance Highlights**: MedVP-LLaVA는 여러 의료 VQA 데이터셋에서 최신 스테이트 오브 아트 모델들을 초과하는 성능을 보여줍니다. 우리의 접근 방식은 세밀한 의료 이미지를 이해하는 데 효과적일 뿐만 아니라, 임상적 중요성을 갖춘 결과를 산출합니다. 또한, 이 연구는 의료 VLM을 위한 패러다임을 개척하고, 데이터셋과 모델 가중치를 공개할 계획입니다.



### Optimizing Small Language Models for In-Vehicle Function-Calling (https://arxiv.org/abs/2501.02342)
- **What's New**: 본 논문에서는 차량 내에 Small Language Models (SLMs)를 기능 호출 에이전트로 활용하는 혁신적인 접근 방법을 제안합니다. 기존의 규칙 기반 시스템을 대체하며, SLM을 통해 차량 제어 메커니즘을 단순화하고 사용자 경험을 향상시키는 것을 목표로 합니다. SLMs의 작은 사이즈 덕분에 차량 시스템의 통합이 용이해져, 외부 소프트웨어 업데이트나 운전자의 조건에 따라 쉽게 조정할 수 있는 시스템을 구성할 수 있습니다.

- **Technical Details**: 우리는 Microsoft의 Phi-3 mini 모델에 대한 최적화 작업을 수행하였으며, 모델 압축 기술인 pruning, healing, quantization을 통해 리소스 제약이 있는 차량에 적합하도록 하였습니다. 이 과정에서 우리는 모델의 크기를 2억 개의 파라미터를 줄이면서도 복잡한 차량 내 작업을 정확하고 효율적으로 수행할 수 있는 능력을 유지할 수 있음을 입증했습니다. 또한, 경량 런타임 환경에서 모델을 실행해 초당 11개의 토큰을 생성할 수 있어, 하드웨어 가속 없이 실시간 추론이 가능하다는 점이 특징입니다.

- **Performance Highlights**: SLM을 활용한 차량 제어 시스템은 사용자가 보다 직관적으로 차량과 상호작용할 수 있도록 지원합니다. 이 연구의 결과는 차량 시스템에 새로운 기능을 효과적으로 통합할 수 있는 가능성을 보여주며, 차량 내 앰비언트 설정 및 음성 비서와 같은 고급 기능들이 사용자 요구에 따라 변할 수 있게 합니다. 이러한 Advancements는 궁극적으로 개선된 주행 경험을 제공할 것으로 기대됩니다.



### Examining the Robustness of Homogeneity Bias to Hyperparameter Adjustments in GPT-4 (https://arxiv.org/abs/2501.02211)
- **What's New**: 이 연구는 GPT-4의 hyperparameter 조정이 동질성 편향(homogeneity bias)에 미치는 영향을 탐구합니다. 특히 sampling temperature와 top p를 조정하여 모델의 출력 랜덤성을 통제하고, 다양한 인종 및 성별 집단에 대한 이야기 생성의 유사성을 평가합니다. 이 과정에서 동질성 편향이 지속적으로 나타나고, hyperparameter 간의 관계가 비선형적인 패턴을 보인다는 중요한 발견을 했습니다.

- **Technical Details**: 연구 방법론은 세 가지 단계로 나뉩니다. 첫째, 네 가지 교차 집단(Black men, Black women, White men, White women)을 대표하는 얼굴 자극을 선택했습니다. 둘째, Vision-Language Model (VLM)과 연구 질문에 맞춰 조정한 hyperparameters를 설명합니다. 셋째, 혼합 효과 모델(mixed-effects models)을 사용하여 hyperparameter 값에 따른 동질성 편향의 크기를 비교하는 분석 전략을 세웠습니다.

- **Performance Highlights**: 연구 결과, 동질성 편향은 대부분의 hyperparameter 구성에서 지속되며, Black Americans와 여성은 White Americans와 남성보다 더 동질적으로 표현됩니다. 또한, Temperature를 높이거나 Top p를 낮추는 것은 인종 동질성 편향을 줄일 수 있지만, 성별 동질성 편향에 대한 영향은 상이합니다. 이는 hyperparameter 조정이 특정 편향을 완화할 수 있지만, 모든 사회 집단 차원에서 동질성 편향을 해결하는 보편적인 솔루션 역할을 할 수는 없음을 시사합니다.



### Benchmark Evaluations, Applications, and Challenges of Large Vision Language Models: A Survey (https://arxiv.org/abs/2501.02189)
Comments:
          34 pages, 3 figures

- **What's New**: 이번 논문은 최근 5년간(2019-2024) 개발된 멀티모달 비전 언어 모델(vision language models, VLM)에 대한 종합적인 개요를 제공합니다. 현재 VLM의 주요 구조, 교육 방법, 평가 기준 및 다양한 응용 분야를 체계적으로 정리하며, 특히 VLM 연구에 관심 있는 학술 연구자들에게 유용한 정보를 제공합니다. VLMs는 시각적 및 텍스트 입력을 결합하여 더욱 깊이 있는 이해를 가능하게 하는 독창적인 기술입니다.

- **Technical Details**: 이 논문에서는 VLMs의 주요 구성 요소와 훈련 방법에 대한 설명을 제공합니다. 특히, 이미지와 텍스트 정보를 정렬하기 위해 사전 훈련된 대규모 언어 모델(large language models, LLM)을 백본으로 사용하는 경향이 증가하고 있으며, 이는 VLM이 비주얼 콘텐츠를 더 잘 이해하도록 돕습니다. VLM의 훈련 목표와 아키텍처에 대한 주요 연구 방향을 세 가지로 나누어 설명하며, CRIP, BLIP, LLaMA와 같은 모델들이 있습니다.

- **Performance Highlights**: VLMs는 비전 인식 작업에서 뛰어난 성과를 보이며, 제로샷(Zero-shot) 분류에서도 기존의 단일 모달 모델을 넘어서는 성능을 보여줍니다. 또한, VLM의 활용 사례로는 자율주행, 로봇공학, 비디오 생성 등이 있으며, 시각적 질문 답변(visual question answering) 같은 복합적인 작업을 가능하게 합니다. 그러나 시각적 환각, 공정성, 안전성 문제와 같은 새로운 도전 과제가 있으며, 이러한 도전들은 멀티모달 모델의 발전을 위한 중요한 연구 주제로 급부상하고 있습니다.



### Table as Thought: Exploring Structured Thoughts in LLM Reasoning (https://arxiv.org/abs/2501.02152)
- **What's New**: 새로운 연구 프레임워크인 'Table as Thought'가 제안되었습니다. 이 프레임워크는 인지 신경과학(cognitive neuroscience) 이론에 영감을 받아, 체계적인 과정을 통해 대형 언어 모델의 추론 능력을 향상시키고자 합니다. 기존의 사고 체계는 주로 사고의 순서에 중점을 두었으나, 개별 사고 단계의 구조는 충분히 탐구되지 않았습니다.

- **Technical Details**: Table as Thought는 사고 과정을 표 형식으로 구성하는 방식으로, 각 행은 순차적인 사고 단계를 나타내고 각 열은 중요한 제약(constraints)이나 맥락(contextual information)을 캡처하여 추론을 강화합니다. 이 과정은 자기 검증(self-verification) 단계에서 표의 완전성과 정확성을 보장할 때까지 반복적으로 진행됩니다.

- **Performance Highlights**: 실험 결과 'Table as Thought'가 계획(task planning) 작업에서 우수한 성능을 보였으며, 비구조적 사고 기반선(unstructured thought baselines)에 비해 수학적 추론(mathematical reasoning)에도 강력한 잠재력을 가지고 있음을 입증했습니다. 이 연구는 LLM 내의 사고 표현(thought representation)을 정교화하는 새로운 탐색을 제공하며, AI 인지(cognition) 및 추론(reasoning) 발전의 기초를 닦고 있습니다.



### METAGENE-1: Metagenomic Foundation Model for Pandemic Monitoring (https://arxiv.org/abs/2501.02045)
- **What's New**: 본 논문에서는 70억 개의 파라미터를 가진 autoregressive transformer 모델인 METAGENE-1을 소개합니다. 이 모델은 1.5조 개의 염기쌍으로 구성된 다양하고 드문 metagenomic DNA 및 RNA 시퀀스를 포함하는 새로운 말뭉치(corpus)에서 사전 학습되었습니다. METAGENE-1의 목표는 개인 유전체나 특정 종의 선별적 데이터 세트를 넘어서, 인간의 하수에서 발견되는 유전 정보를 포괄적으로 캡처하여 전염병 모니터링 및 병원체 탐지와 같은 작업을 지원하는 것입니다.

- **Technical Details**: 사전 학습 과정에서는 metagenomic 시퀀스를 위한 맞춤형 byte-pair encoding (BPE) 토크나이제이션 전략을 사용하여 데이터 세트를 처리하였습니다. 데이터 세트는 여러 지점 및 시간에 수집된 수많은 종의 짧은 비선별(un-curated) 시퀀스로 구성되어 있으며, 이는 METAGENE-1이 미생물 및 바이러스 다양성의 복잡성을 효과적으로 표현할 수 있게 해줍니다. 모델 아키텍처는 GPT 및 Llama 계열 모델들과 유사한 decoder 스타일의 언어 모델을 채택하였습니다.

- **Performance Highlights**: METAGENE-1은 병원체 탐지 및 metagenomic 임베딩(embedding) 벤치마크에서 최상위 성능을 달성하였으며, 이는 기존의 인간 및 동물 유전체로 훈련된 모델들을 능가하는 결과입니다. 또한, 비정상 탐지 시나리오에서도 우수한 성과를 보여주며, 공공 보건 분야에 적합한 응용 프로그램으로서의 잠재력을 드러냅니다. 궁극적으로 METAGENE-1은 전염병 모니터링 및 새로운 건강 위협의 조기 탐지에 기여할 수 있는 기초 모델로 자리 잡는 것이 목표입니다.



### Is Your Image a Good Storyteller? (https://arxiv.org/abs/2501.01982)
Comments:
          Accepted by AAAI 2025

- **What's New**: 본 논문에서는 이미지의 의미적 복잡성을 평가하는 Image Semantic Assessment (ISA) 작업을 제안합니다. ISA 데이터셋과 언어 기반 방법을 활용하여 복잡성 평가를 자동화하고, 다양한 문화적 배경을 가진 사람들이 이해할 수 있는 이미지를 생성하는 필요성을 강조합니다. 이 연구는 이미지의 의미적 가치를 기반으로 한 새로운 평가 기준을 설정하는 데 기여합니다.

- **Technical Details**: ISA 데이터셋은 Pinterest에서 수집된 2,946개의 이미지를 포함하고, 각 이미지는 Entity Score와 Semantic Score를 통해 평가됩니다. Entity Score는 이미지 내 요소의 풍부함을 측정하고, Semantic Score는 더 높은 수준의 의미적 복잡성을 평가합니다. 논문에서는 Vision-Language 협력 ISA 방법(VLISA)을 제안하여, 대형 비전-언어 모델(LVLM)을 사용해 이미지의 의미적 정보를 추출합니다.

- **Performance Highlights**: 실험 결과, ISA 작업은 전통적인 비전 모델들에 도전적이며, 제안한 방법은 Semantic Complexity Scoring 작업에서 다른 기준 모델들에 비해 우수한 성능을 보였습니다. 이 연구는 AI 모델의 교육 및 평가를 위한 복잡한 이미지를 선별하는 데 중요한 단초를 제공합니다. 따라서 더 많은 연구자들이 이미지의 의미적 복잡성에 집중할 수 있는 기반을 마련했습니다.



### BaiJia: A Large-Scale Role-Playing Agent Corpus of Chinese Historical Characters (https://arxiv.org/abs/2412.20024)
- **What's New**: BaiJia라는 대규모 역사 역할 수행 에이전트 수집 데이터를 소개합니다. 이 데이터셋은 중국의 역사적 인물들로 구성되어 있으며, AI 기반 역사 역할 수행 모델을 위한 최초의 저자원(low-resource) 데이터베이스입니다. BaiJia는 산재해 있는 역사적 텍스트 기록의 문제를 해결하고, 인물들의 전기, 문학, 가족 관계 등 다양한 정보를 통합하여 역할 수행 능력을 증진시키는 데 기여합니다.

- **Technical Details**: BaiJia 데이터셋은 타당한 데이터 수집과 다각적인 캐릭터 정보 수집으로 구성되어 있으며, 중국 역사의 다섯 왕조에서 19,281명의 역사적 인물에 대한 정보를 포함합니다. 이 데이터는 전기적 정보, 가족 관계, 역사적 사건 등이 포함되어 있으며, LLM을 위한 Fine-Tuning(SFT)에서는 GPT-4o-mini를 활용하여 대화 생성 및 캐릭터 이력서 작성이 이루어집니다.

- **Performance Highlights**: BaiJia 데이터셋을 활용한 실험 결과, 다양한 LLM들이 역할 수행 능력이 향상됨을 입증하였습니다. 특히 캐릭터 일관성(Character Consistency) 및 문화적 적합성(Cultural & Historical Appropriateness) 차원에서 가장 큰 개선을 보였으며, 각각의 LLM이 역사적 캐릭터를 효과적으로 재현하지 못하는 한계를 극복하는 데 중요한 역할을 하였습니다. 또한, 고급 LLM에서도 BaiJia 데이터셋을 포함한 경우 성능이 크게 향상되는 경향이 있음을 확인하였습니다.



New uploads on arXiv(cs.IR)

### Towards Reliable Testing for Multiple Information Retrieval System Comparisons (https://arxiv.org/abs/2501.03930)
- **What's New**: 이 논문은 Null Hypothesis Significance Testing(NHST)가 여러 정보 검색 시스템을 동시에 비교할 때 발생할 수 있는 오류를 분석하고 이를 개선하기 위한 새로운 접근 방식을 제안합니다. 연구자들은 맞춤형 데이터와 실험을 통해 특정 테스트가 얼마나 신뢰할 수 있는지를 평가했습니다. 특히, Wilcoxon 테스트와 Benjamini-Hochberg 보정 방법이 Statitistical power가 가장 높다는 결과를 도출하였습니다.

- **Technical Details**: 이 연구는 TREC의 시뮬레이션 데이터를 사용해 정보 검색 실험에서 다수의 비교 조정 방법의 수행을 분석합니다. m개의 검색 시스템이 있을 경우, 각 시스템을 비교하려면 k = m(m-1)/2의 서로 다른 쌍을 평가해야 한다고 설명합니다. NHST를 사용할 경우, 여러 가설을 동시에 테스트함으로써 Type I 오류율이 증가할 수 있음을 강조합니다.

- **Performance Highlights**: 여러 통계적 테스트의 결과에 따르면, unadjusted 테스트는 매우 높은 Type I 오류율을 발생시키며, 이는 조정 절차를 반드시 사용해야 함을 의미합니다. 실제 TREC 데이터를 통한 실험에서는 Wilcoxon 테스트와 Benjamini-Hochberg 조정 방식이 가장 높은 통계적 power를 나타냈습니다. 따라서 이 조합이 향후 연구에서 권장됩니다.



### BERTopic for Topic Modeling of Hindi Short Texts: A Comparative Study (https://arxiv.org/abs/2501.03843)
Comments:
          Accepted into IndoNLP: The First Workshop on Natural Language Processing for Indo-Aryan and Dravidian Languages, collocated with COLING 2025. Set to appear in the workshop proceedings published in ACL Anthology

- **What's New**: 이번 연구는 힌디어( Hindi)와 같은 모국어로 작성된 짧은 텍스트 데이터에 대한 주제 모델링(topic modeling) 방법의 중요성을 강조합니다. 특히 BERTopic이 이러한 짧은 텍스트를 모델링하는 데 얼마나 효과적인지를 분석하고 있습니다. 이는 기존 연구에서 충분히 다루어지지 않았던 영역으로, 최신 미디어에서의 필요성이 증가하고 있습니다.

- **Technical Details**: 연구에서는 6가지 다양한 문서 임베딩(document embedding) 모델을 사용하여 BERTopic을 평가하고, Latent Dirichlet Allocation (LDA), Non-negative Matrix Factorization (NMF), Latent Semantic Indexing (LSI) 등 8가지 기존의 주제 모델링 기법과 성능을 비교합니다. BERTopic은 문맥 임베딩(contextual embeddings)을 활용하여 데이터 내에서 의미적 관계를 포착하며, 전통적인 모델보다 특히 짧고 다양한 텍스트에 대해 효과적일 수 있습니다.

- **Performance Highlights**: 실험 결과, BERTopic은 짧은 힌디어 텍스트로부터 일관된 주제를 포착하는 데 있어 다른 모델들보다 일관되게 우수한 성능을 보여주었습니다. 다양한 주제 수에 대한 일관성 점수(coherence scores)를 통해 평가한 결과, BERTopic이 다른 방법들을 지속적으로 능가하는 것으로 나타났습니다.



### Extending ChatGPT with a Browserless System for Web Product Price Extraction (https://arxiv.org/abs/2501.03811)
Comments:
          14 pages, 4 figures

- **What's New**: 이 논문은 ChatGPT의 답변 능력을 확장하는 시스템인 Wextractor를 제안합니다. Wextractor는 사용자가 제품 가격과 같은 거래 검색에 대한 질문에 대해 정확한 응답을 제공할 수 있도록 돕습니다. 기존의 ChatGPT에서는 웹 검색이나 실시간 정보에 접근할 수 없어 발생하는 한계를 해결하고자 하며, 사용자 검색을 통해 축적된 정보를 활용하여 더 빠른 답변을 지원합니다.

- **Technical Details**: Wextractor는 소셜 추출(social extraction)과 포인팅 패턴 추출(pointing pattern extraction)의 두 가지 개선점을 포함하여 제품 가격 관련 질문에 답변합니다. 시스템은 웹 페이지의 URL을 입력받아 가격 정보를 추출하는 4단계 프로세스를 거치며, 이 과정에서 브라우저의 도움 없이 원본 HTML을 수집하고 가격 추출 규칙을 적용합니다. 거기서 추출된 정보는 다음 번 쿼리에 다시 활용될 수 있습니다.

- **Performance Highlights**: 본 연구에서는 Wextractor가 ChatGPT와의 통합을 통해 다양한 실시간 가격 질문에보다 정확한 답변을 제공할 수 있음을 보여주고 있습니다. 사용자 정보를 바탕으로 가격을 추출함으로써 Wextractor는 불필요한 웹 요청을 줄여 응답 속도를 향상시킵니다. 또한, 이 시스템은 전통적인 접근 방법에 비해 가격 추출 과정을 간소화하고 효율적으로 만들어 기업의 경쟁력을 높일 수 있는 잠재력을 가지고 있습니다.



### Multi-label Cross-lingual automatic music genre classification from lyrics with Sentence BER (https://arxiv.org/abs/2501.03769)
Comments:
          5 pages

- **What's New**: 이 논문은 다국어 문장 임베딩을 활용하여 가사를 기반으로 한 자동 음악 장르 분류 시스템을 제안합니다. 시스템은 서로 다른 언어의 가사를 훈련하고 서로 다른 장르를 예측하는 능력을 보여주며, 기존 방법론인 가사를 번역하고 bag-of-words 표현을 사용하는 것보다 우월한 성능을 발휘합니다. 특히, 중앙 집중 식별을 통해 크로스 링구얼(cross-lingual) 성능을 향상시키는 방법을 제시합니다.

- **Technical Details**: 제안된 방법은 사전 훈련된 sBERT(문장 변환기)를 사용하여 각 가사의 임베딩을 생성합니다. 사용된 sBERT 모델은 paraphrase-multilingual-mpnet-base-v2로, 다양한 장르의 가사를 여러 언어에서 동일한 결과 값으로 매핑할 수 있도록 훈련되었습니다. 각 장르에 대해 하나의 다중 라벨 분류기를 사용하여 가사가 여러 장르 레이블을 가질 수 있도록 설정되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기초 기술보다 genre-wise average F1-score가 0.35에서 0.69로 개선되었습니다. 중앙 집중화를 통해 훈련 및 테스트 세트를 조정함으로써 크로스 링구얼 성능이 개선되는 것으로 나타났습니다. 이 접근 방식은 음악 정보 검색 시스템의 성능을 향상시키며, 저평가된 언어와 문화 영역에서도 적용할 수 있는 확장 가능한 해결책을 제공합니다.



### RecKG: Knowledge Graph for Recommender Systems (https://arxiv.org/abs/2501.03598)
Comments:
          Accepted by The 39th ACM/SIGAPP Symposium On Applied Computing(SAC) 2024

- **What's New**: 이 연구는 다양한 추천 시스템 간의 지식 그래프(Knowledge Graph, KG) 통합에 대한 연구의 부족 문제를 해결하기 위해 RecKG라는 표준화된 지식 그래프를 제안합니다. RecKG는 서로 다른 데이터 세트에서 엔티티의 일관된 표현을 보장하며, 효과적인 데이터 통합을 위해 다양한 속성 유형을 지원합니다. 이 연구에서는 RecKG를 사용하여 실제 데이터 세트를 표준화하고 그래프 데이터베이스를 통해 응용 프로그램을 개발합니다.

- **Technical Details**: RecKG는 추천 시스템에 필수적인 엔티티로 구성되며, 사용자와 아이템에 초점을 맞추고 있습니다. 이를 통해 데이터 통합의 일관성을 보장하고 서로 다른 추천 시스템에서 동일한 개념과 속성을 균일하게 표현함으로써 통합 과정에서 중요한 속성을 최소화하는 것을 목표로 합니다. RecKG는 다양한 추천 시스템 간의 속성 커버리지를 보장하여 데이터 통합 시 누락되는 속성이 최소화되도록 설계되었습니다.

- **Performance Highlights**: RecKG의 효과성은 연속적인 정량적 평가를 통해 검증되며, 기존 연구와의 정성적 비교를 통해 상호 운용성(interoperability)의 성과를 확인합니다. 이 연구를 통해 지식 그래프 기반 추천 시스템의 데이터 통합 및 추천 품질이 크게 향상된 결과를 보여줍니다. RecKG의 도입으로 추천 시스템 간 정보의 더 많은 발견과 추가적인 의미적 정보의 통합이 가능해졌습니다.



### RAG-Check: Evaluating Multimodal Retrieval Augmented Generation Performanc (https://arxiv.org/abs/2501.03995)
- **What's New**: 이 논문은 Retrieval-Augmented Generation (RAG) 기술의 신뢰성을 평가하기 위한 새로운 프레임워크를 제안합니다. 특히, 논문은 multi-modal RAG에서 발생할 수 있는 새로운 유형의 환각을 다루고 있습니다. 제안된 방법에서는 relevancy score (RS)와 correctness score (CS)를 활용하여 생성된 응답의 정확성을 평가합니다.

- **Technical Details**: 제안된 방식에서는 RAG의 데이터 선택 과정과 컨텍스트 생성 방법에서 환각이 발생할 수 있음을 강조합니다. RS 모델은 데이터베이스에서 선택된 정보 조각과 사용자 쿼리 간의 관련성을 평가하며, CS 모델은 생성된 응답의 정확성을 평가합니다. 두 모델은 ChatGPT 데이터를 사용하여 훈련되었으며, 약 88%의 정확도를 달성하였습니다.

- **Performance Highlights**: 연구 결과, RS 모델은 CLIP보다 20% 더 많은 경우 인간의 선택과 일치하며, CS 모델은 약 91%의 정확도로 인간의 선호도를 반영합니다. 또한, 5000개의 샘플로 구성된 인간 주석 데이터베이스를 통해 선택된 정보 조각의 관련성과 응답 진술의 정확성을 평가하였습니다.



### (De)-Indexing and the Right to be Forgotten (https://arxiv.org/abs/2501.03989)
- **What's New**: 이 논문에서는 개인 데이터의 관리와 관련하여 '잊혀질 권리'(Right to be Forgotten, RTBF)의 개념을 소개합니다. RTBF는 사람들이 구식 또는 해로운 정보를 공공 접근에서 제거할 수 있는 권리를 부여하지만, 이를 구현하는 데에는 기술적인 어려움이 존재합니다. 이 문서에서는 정보 검색(Information Retrieval, IR)과 비색인화(de-indexing)의 기본 개념을 설명하며, 검색 엔진이 특정 콘텐츠를 효과적으로 "잊는" 방법을 이해하는 데 필요한 정보를 제공합니다.

- **Technical Details**: 본 논문은 정보 검색의 다양한 모델을 설명합니다. 정보 검색은 사용자 질의에 대한 정보를 대량의 비구조적 또는 반구조적 데이터에서 찾는 과정입니다. 정보 검색 시스템에서는 데이터 전처리, 색인 생성, 문서 순위 매기기 등의 여러 단계를 포함하며, 최신 모델인 대형 언어 모델(LLMs)이 데이터 처리 능력을 향상시키는 데 어떻게 기여하는지에 대해서도 논의합니다.

- **Performance Highlights**: 정보 검색 모델은 크게 부울 모델(Boolean Models), 벡터 공간 모델(Vector Space Models), 확률 모델(Probabilistic Models)로 나뉩니다. 부울 모델은 문서와 질의를 용어 집합으로 처리하며, 벡터 공간 모델은 문서와 질의를 다차원 벡터로 표현합니다. 확률 모델은 문서가 특정 질의에 관련성이 있을 확률을 예측하는 데 초점을 맞추며, BM25와 같은 고급 기법에 기반을 둡니다. 이러한 모델들은 검색 결과의 질을 향상시키기 위해 다양한 기술과 기계 학습(Machine Learning, ML), 자연어 처리(Natural Language Processing, NLP) 기술이 통합되고 있습니다.



### Exploring the Potential of Large Language Models in Public Transportation: San Antonio Case Study (https://arxiv.org/abs/2501.03904)
Comments:
          This work is accepted to AAAI 2025 Workshop on AI for Urban Planning. arXiv admin note: substantial text overlap with arXiv:2407.11003

- **What's New**: 본 연구는 대형 언어 모델(LLM)을 공공 교통 시스템에 통합함으로써 도시 이동성을 개선할 기회를 탐색합니다. 샌안토니오의 교통 시스템을 중심으로 LLM의 잠재력을 분석하여, 경로 계획 최적화, 대기 시간 단축, 개인화된 여행 지원 등의 분야에서 효과를 기대하고 있습니다. 이번 연구는 LLM이 자원 할당 및 승객 만족도를 높일 수 있는 방법을 제시하며, 데이터 기반 의사결정의 중요성을 강조합니다.

- **Technical Details**: 연구에서는 OpenAI의 GPT 시리즈와 같은 대형 언어 모델이 복잡한 의사결정을 돕고 대량의 데이터를 분석하는 능력을 바탕으로 공공 교통 관리에 기여할 가능성을 평가합니다. 특히, GTFS(General Transit Feed Specification) 데이터를 활용하여 LLM의 경로 계획 및 승객 소통을 맡길 수 있는 방법을 모색하고 있습니다. 이를 위해 다른 ChatGPT 모델의 성능을 비교하고 교통 정보를 이해하는 능력을 테스트했습니다.

- **Performance Highlights**: 연구 결과, LLM은 공공 교통 시스템에 대해 복잡한 쿼리를 효과적으로 처리할 수 있으며, 실시간 정보 제공 및 개인화된 여행 지원을 통해 승객 경험을 향상시킬 수 있음을 보여줍니다. 그러나, 모델의 정확성을 높이기 위해서는 신중한 엔지니어링과 세부 조정이 필수적입니다. 샌안토니오 사례를 바탕으로 얻은 교훈은 다른 도시에서의 LLM 통합 시스템 개발에 중요한 참고자료가 될 것입니다.



### TACLR: A Scalable and Efficient Retrieval-based Method for Industrial Product Attribute Value Identification (https://arxiv.org/abs/2501.03835)
- **What's New**: 이 논문에서는 e-commerce 플랫폼에서 제품 속성 값 식별(Product Attribute Value Identification, PAVI)의 새로운 접근 방식을 제안합니다. TACLR(Taxonomy-Aware Contrastive Learning Retrieval)이라는 최초의 검색 기반 방법을 도입하여, 제품 프로필과 후보 값을 임베딩으로 변환하고, 유사성을 기반으로 값을 검색합니다. 이 방법은 대량의 카테고리와 속성을 효과적으로 처리할 수 있습니다.

- **Technical Details**: TACLR은 PAVI를 정보 검색(task)으로 정의하며, 주어진 제품 항목에 대한 쿼리와 속성 분류(corpus)로서 작용합니다. 이 방법은 속성 및 카테고리로부터 후보 값을 선택하는 하드 네거티브 샘플링(hard negative sampling) 기법을 사용하여 차별화된 임베딩을 제공합니다. 또한, 동적 임계값(dynamic thresholds)을 도입하여 추론의 유연성을 높였습니다.

- **Performance Highlights**: 실험 결과 TACLR은 수많은 제품 목록과 속성을 효율적으로 처리할 수 있는 능력을 입증했습니다. 이 방법은 실제 상업적 환경에서 성공적으로 배포되어 매일 수백만 개의 제품 리스트를 처리하며, 동적으로 대규모 속성 분류를 지원합니다. TACLR은 또한 속성 값이 누락된 경우를 정확히 탐지하는 기능을 갖추고 있습니다.



### ComMer: a Framework for Compressing and Merging User Data for Personalization (https://arxiv.org/abs/2501.03276)
Comments:
          13 pages, 7 figures

- **What's New**: 이 논문에서는 ComMer - Compress and Merge라는 새로운 프레임워크를 소개하여 대형 언어 모델(LLMs)을 효율적으로 개인화하는 방법을 제안합니다. ComMer는 사용자의 문서를 압축하여 간결한 표현으로 변환한 다음, 이를 결합해 냉동된 LLM에 입력합니다. 이 접근 방식은 사용자 수가 많을 때 자원을 절약하고, 훈련 비용을 줄이며, 제한된 데이터로도 품질을 개선할 수 있는 장점을 가지고 있습니다.

- **Technical Details**: ComMer는 세 가지 단계로 이루어진 아키텍처를 통해 작동합니다. 첫 번째로 각 문서가 독립적으로 소프트 프롬프트로 압축되며, 이러한 압축은 학습 가능한 압축 임베딩과 LoRA 가중치를 활용하여 이루어집니다. 두 번째로 압축된 표현들이 평균 풀링(mean pooling)을 통해 집계되어 단일 소프트 프롬프트로 결합되고, 마지막으로 이 집계된 소프트 프롬프트가 냉동된 LLM에 연결되어 원하는 출력을 생성합니다.

- **Performance Highlights**: 실험 결과, ComMer는 제한된 토큰 예산에서 개인화된 기술 학습(task)에서 우수한 품질을 보여주었고, 문서의 수가 많아질수록 품질이 향상되는 경향을 보입니다. 그러나 지식 집중형 작업에서는 여러 문서의 정보를 모두 표현하는 데 제한이 있어 품질 저하가 발생하는 것으로 나타났습니다. 이는 다중 문서 압축을 통한 개인화에서의 무역 오프와 잠재적 최적화 방향에 대한 통찰을 제공합니다.



### LightGNN: Simple Graph Neural Network for Recommendation (https://arxiv.org/abs/2501.03228)
Comments:
          Accepted to WSDM 2025 Oral

- **What's New**: LightGNN은 추천 시스템을 위한 경량화된 GNN 프루닝(framing) 프레임워크로, 모델의 복잡성을 줄이면서 협업 모델링 기능을 유지합니다. 이 프레임워크는 자원 친화적(hierarchical) 지식 증류(knowledge distillation) 목표에 따라 중간 레이어를 통해 관측된 그래프를 보강하여 성능을 향상시킵니다. LightGNN은 긍정적인 피드백을 사용하는 대신에 사용자 상호작용 그래프에서 불필요한 경량(edge) 및 임베딩(embedding)을 제거하여 효율성 높은 추천을 제공합니다.

- **Technical Details**: LightGNN은 그래프 구조 학습을 통합하여 각 엣지와 임베딩 항목의 중복 또는 노이즈 가능성을 명확하게 평가합니다. 이 과정은 상류 추천 작업과 계층적 지식 증류 패러다임을 활용하여 감독(supervised) 방식으로 이루어집니다. 새로운 계층적 KD 접근법은 커다란 학습 능력을 유지해 주어, 높은 비율의 압축 상황에서도 추천 성능을 지킬 수 있도록 합니다.

- **Performance Highlights**: LightGNN은 공개 데이터 세트에서 이루어진 큰 규모의 실험을 통해 계산 효율성과 추천 정확도 모두에서 성능이 크게 향상되었습니다. 이 프레임워크는 엣지 수를 80%, 임베딩 항목 수를 90%까지 줄이면서도 복잡한 최신 기술(technology) 대비 유사한 성능을 유지하였습니다. 따라서 LightGNN은 추천 정확성, 추론 효율성, 모델 강건성, 그리고 해석 가능성 측면에서 뛰어난 성능을 보여줍니다.



### Personalized Fashion Recommendation with Image Attributes and Aesthetics Assessmen (https://arxiv.org/abs/2501.03085)
- **What's New**: 이번 논문은 개인화된 패션 추천 시스템의 문제를 해결하기 위한 새로운 접근 방식을 제안합니다. 연구팀은 사용자의 미적 취향(aesthetic appetite)과 관련된 정보를 더욱 효율적으로 활용하고, 최신 아이템들에 대한 냉시작(cold-start) 문제를 극복하는 데 주력했습니다. 기존 연구에서는 이미지와 텍스트를 별개의 요소로 간주했지만, 본 연구에서는 두 가지 정보를 결합하여 풍부한 속성(attribute) 그래프(attribute graph)를 생성합니다.

- **Technical Details**: 연구에서는 이미지 최적화(image utilization)와 사용자 모델링(user modeling)에 있어 노이즈를 감소시키는 두 가지 속성 그래프(attribute graphs)를 생성하여 정보의 가용성을 높였습니다. 또한, 최근의 대형 언어 모델(large language models) 및 비전 모델(vision models)의 발전을 활용, 세밀한 속성을 추출하는 두 가지 프롬프트(prompts)를 통해 실험을 진행했습니다. 이 접근법은 이미지와 텍스트 정보를 통합함으로써 더 나은 성능을 기대할 수 있도록 설계되었습니다.

- **Performance Highlights**: IQON3000 데이터셋을 사용한 초기 실험에서 제안된 방법은 기존 기준선(baselines)과 비교하여 경쟁력 있는 정확도를 달성했습니다. 이는 개인화된 패션 추천 시스템의 성능을 개선할 수 있는 가능성을 보여줍니다. 본 연구 결과는 패션 추천의 정확성을 높이고 최신 트렌드에 유연하게 대응할 수 있는 시스템 개발에 기여할 것으로 예상됩니다.



### OpenTable data with multi-criteria ratings (https://arxiv.org/abs/2501.03072)
- **What's New**: 이 논문은 다기준 추천 시스템(Multi-Criteria Recommender Systems, MCRS)에 초점을 맞추고 있으며, OpenTable 웹사이트에서 획득한 데이터 세트를 공개합니다. 이 데이터 세트는 다기준 추천을 위한 벤치마크 데이터로 사용될 수 있으며, 사용자 개별의 선호도를 다양한 기준에서 동시에 고려할 수 있는 가능성을 제공합니다. 기존의 단일 평점 중심의 추천 시스템과 달리, MCRS는 여러 속성과 기준을 적용하여 보다 만족스러운 추천을 수행합니다.

- **Technical Details**: OpenTable 데이터 세트는 웹 크롤링을 통해 수집되었으며, 총 19,536개의 평점이 1,309명의 사용자에 의해 91개의 레스토랑에 대해 제공됩니다. 이 데이터에는 전체 평점과 여러 기준에 따른 평점이 포함되어 있으며, 평점은 1에서 5까지의 스케일로 주어집니다. 데이터 수집 과정에서 사용자 식별이 어렵고, 평점의 희소도가 83.6%에 달하는 등 데이터 밀집도에 대한 도전이 있었습니다.

- **Performance Highlights**: OpenTable 데이터 세트는 전통적인 추천 시스템뿐만 아니라 다기준 추천 시스템의 벤치마크로도 활용될 수 있습니다. 평균적으로 사용자당 제공된 평점 수는 14.9이고, 레스토랑당 평균 평점 수는 214입니다. 다양한 MCRS 알고리즘의 성능을 평가하는 데 활용될 수 있는 이 데이터 세트는 연구자들에게 유용한 자료를 제공합니다.



### FlipedRAG: Black-Box Opinion Manipulation Attacks to Retrieval-Augmented Generation of Large Language Models (https://arxiv.org/abs/2501.02968)
Comments:
          arXiv admin note: text overlap with arXiv:2407.13757

- **What's New**: 이 논문은 Retrieval-Augmented Generation (RAG) 시스템을 통한 여론 조작의 가능성을 탐구합니다. 특히, FlipedRAG라는 새로운 블랙박스 공격 방법을 제안하며, 이는 LLM의 생성된 응답을 조작하는 데 효과적입니다. 이 연구는 RAG 모델의 취약성을 실질적이고 현실적인 시나리오에서 분석함으로써 기존의 연구 한계를 극복하고 있습니다.

- **Technical Details**: FlipedRAG는 지침 공학(instruction engineering)을 활용하여 블랙박스 RAG 시스템의 부분 검색 모델 출력을 획득하고, 이를 통해 대체 모델을 훈련시켜 여론 조작 공격의 효과를 높입니다. 본 연구는 여러 언어 주제를 포함한 데이터셋에서 실험을 수행하여 이 공격 전략의 유효성을 검증하였으며, RAG의 생성된 내용의 평균 성공률을 16.7% 향상시켰음을 보여줍니다.

- **Performance Highlights**: 제안된 공격 방법은 RAG의 생성된 응답의 의견 극성을 평균 50% 변화시켰으며, 사용자 인식의 20% 변화도 유도했습니다. 또한, 기존의 방어 메커니즘이 이러한 유형의 공격 완화에 충분하지 않음을 결론지음으로써, 새로운 방어 전략 개발의 필요성을 강조하고 있습니다.



### Foundations of GenIR (https://arxiv.org/abs/2501.02842)
Comments:
          Chapter 2 of the book on Information Access in the Era of Generative AI

- **What's New**: 이 논문은 현대 생성 AI 모델이 정보 접근 (Information Access, IA) 시스템에 미치는 근본적인 영향을 다루고 있습니다. 전통적인 AI와 달리, 생성 AI 모델은 대규모 학습 및 뛰어난 데이터 모델링을 통해 고품질의 인간과 유사한 응답을 생성할 수 있는 능력을 가지고 있으며, 이는 IA 패러다임의 발전에 대한 새로운 기회를 제공합니다. 특히 정보 생성 (Information Generation)과 정보 종합 (Information Synthesis) 두 가지 방법을 자세히 소개하고 있습니다.

- **Technical Details**: 정보 생성은 AI가 사용자 니즈에 맞춘 맞춤형 콘텐츠를 직접 생성할 수 있도록 하여 즉각적이고 관련성 있는 출력을 통해 사용자 경험을 향상시킵니다. 정보 종합은 기존 정보를 통합 및 재구성하는 생성 AI의 능력을 활용하여 정확성과 외부 지식이 필요한 시나리오에서 모델 환각 현상 (hallucination) 문제를 완화할 수 있습니다. 이 논문에서는 생성 모델의 아키텍처, 스케일링, 학습 방법 등에 대해 상세히 설명합니다.

- **Performance Highlights**: 최근 몇 년간 ChatGPT, Bing, Midjourney 등과 같은 생성 모델들이 사용자 질문에 응답하고 이미지 생성 및 개인화된 콘텐츠 생성을 통해 큰 발전을 이루었습니다. 이러한 성장은 능력 있는 모델 아키텍처 및 대규모 인터넷 데이터를 기반으로 하여 이뤄졌습니다. 생성 모델의 성능은 여전히 빠르게 개선되고 있으며, 다양한 워크플로우 및 일상 활동에 점점 더 많이 통합되고 있습니다.



### Improving GenIR Systems Based on User Feedback (https://arxiv.org/abs/2501.02838)
Comments:
          Chapter 5 of the book on Information Access in the Era of Generative AI

- **What's New**: 이번 논문에서는 GenIR 시스템을 사용자 피드백을 기반으로 어떻게 개선할 수 있는지를 논의합니다. '사용자(user)'의 개념이 확장되었으며, 다양한 피드백 정보와 전략이 제시되고 있습니다. 특히, GenIR을 위한 정렬(Alignment) 기법과 사용자 피드백 학습 방법이 강조되어, 기존의 사용자 피드백 활용 방법을 넘어서는 혁신적인 기법들이 제안되고 있음을 알 수 있습니다.

- **Technical Details**: 정보 접근 시스템에서의 사용자 피드백은 명시적 피드백(explicit feedback)과 암시적 피드백(implicit feedback)의 두 가지 유형으로 나뉘어집니다. GenIR 시스템에서는 사용자 상호작용 이력, 쿼리, 클릭, 구매 등 다양한 피드백 정보가 활용됩니다. GenIR에서 피드백 정보는 주로 ID 수준의 데이터와 다중 양식(detailed information)으로 나타나며, 최근에는 멀티미디어 입력(multimedia inputs)도 주목받고 있습니다.

- **Performance Highlights**: 사용자 피드백을 GenIR 시스템에 효과적으로 통합하기 위해 다양한 전략이 소개됩니다. 예를 들어, 사용자 상호작용 정보를 LLM의 파라미터 미세 조정(fine-tuning)에 활용하거나, 사용자 선호도를 반영하여 특정 작업을 식별할 수 있습니다. 사용자 행동을 통한 학습을 통해 GenIR 시스템의 적응력이 향상되고 있으며, 앞으로도 더 많은 연구와 전략이 발전할 것으로 기대됩니다.



### GeAR: Generation Augmented Retrieva (https://arxiv.org/abs/2501.02772)
- **What's New**: 이번 연구에서는 GeAR (Generation-Augmented Retrieval)라는 새로운 문서 검색 방법을 제안합니다. GeAR은 질의와 문서의 융합 표현을 기반으로 관련 텍스트를 생성하여 세밀한 정보에 집중할 수 있도록 설계되었습니다. 이 방법은 기존의 bi-encoder 구조를 사용하면서도 계산 부담을 추가하지 않으며, 높은 품질의 데이터를 효율적으로 합성할 수 있는 파이프라인을 구축하였습니다.

- **Technical Details**: GeAR은 질의-문서-정보 트리플 구조를 사용하여 지도학습을 통해 질의와 문서 간의 유사성을 최적화합니다. 이 때 텍스트 디코더를 설계하여 질의에 맞는 관련 세부 정보를 생성함으로써 검색 및 로컬라이제이션 기능을 향상시킵니다. 기존의 bi-encoder는 전체 문서의 전역 의미를 강조하지만, GeAR은 fine-grained understanding이 필요한 복잡한 작업에 적합하도록 이 설계를 변경했습니다.

- **Performance Highlights**: GeAR은 다양한 시나리오와 데이터셋에서 경쟁력 있는 검색 및 로컬라이제이션 성능을 보여주었습니다. 실험 결과 GeAR은 질의와 문서에 기반하여 관련 정보를 생성할 수 있는 능력을 발휘하며, 전통적인 검색 과정에 대한 새로운 관점을 제공합니다. 뿐만 아니라, 코드와 데이터, 모델을 기술 검토 후 공개하여 향후 연구를 촉진할 계획입니다.



### Tree-based RAG-Agent Recommendation System: A Case Study in Medical Test Data (https://arxiv.org/abs/2501.02727)
- **What's New**: HiRMed(계층적 RAG 강화 의료 테스트 추천)이라는 새로운 시스템이 소개되었습니다. 이 시스템은 Retrieval-Augmented Generation (RAG)을 활용하여 의료 테스트 추천을 위한 트리 구조의 추천 시스템을 구현합니다. 전통적인 벡터 유사성 기반 방법과 달리, HiRMed는 각 트리 노드에서 의료 추론을 수행하여 초기 증상으로부터 진단 경로를 동적으로 조정합니다.

- **Technical Details**: 이 시스템은 환자의 초기 증상에 따라 진단 요구 사항과 잠재적인 기저 상태를 식별하는 단계별 의료 분석을 수행합니다. Hierarchical RAG Architecture는 각 노드가 특별한 RAG 프로세스를 통합하는 트리 구조를 통해 여러 단계의 의료 추론을 가능하게 합니다. 또한, 두 개의 지식 기반 아키텍처를 통해 일반 의료 이해 및 전문 진단 고려사항을 포함한 동적 지식 통합을 제공합니다.

- **Performance Highlights**: HiRMed는 기존의 검색 기반 방법에 비해 더 높은 정확도와 가능성 있는 진단 경로의 커버리지, 중요한 진단 테스트의 누락 비율 감소를 보여주었습니다. 연구 결과는 HiRMed가 테스트 추천의 해석 가능성을 높이고 명확한 추론 경로를 바탕으로 추천의 타당성을 증명했음을 나타냅니다. 이러한 혁신적인 기능들은 차세대 의료 테스트 추천 시스템을 위한 청사진을 제시합니다.



### Quantum Cognition-Inspired EEG-based Recommendation via Graph Neural Networks (https://arxiv.org/abs/2501.02671)
- **What's New**: 이번 논문에서는 EEG(뇌파 연구)를 기반으로 한 추천 시스템 QUARK를 제안합니다. QUARK는 양자 인지 이론(Quantum Cognition Theory)과 그래프 컨볼루션 네트워크(Graph Convolutional Networks)를 결합하여 실시간으로 사용자 생각을 정확히 반영한 아이템 추천을 목표로 합니다. 이 연구는 전통적인 추천 시스템과 비교하여 EEG 신호를 사용하여 실시간 반영과 의사결정을 가능하게 시도하였습니다.

- **Technical Details**: QUARK 모델은 sliding window 방법을 통해 EEG 데이터를 다루어 시퀀셜한 사고의 변화를 특성화합니다. 양자 인지 이론을 활용하여 혼합된 사고를 잠재적인 요소로 분해하고 과거 사고와 미래 사고 간의 관계를 찾아냅니다. GCN은 이러한 EEG 정보를 종합하여 추천을 위한 최종 표현을 생성하며, 이는 과거 사고가 미래 사고에 미치는 영향을 그래프의 형태로 설명합니다.

- **Performance Highlights**: QUARK 모델은 실제 데이터를 기반으로 한 광범위한 실험을 통해 기존의 추천 모델들을 능가함을 입증하였습니다. 특히, 아이템 추천에서 'top-k' 성능뿐만 아니라 감정 및 스타일 탐지에서도 우수한 성과를 보였습니다. 이러한 결과는 QUARK이 사용자에게 혁신적인 개인화된 서비스 제공에 기여할 수 있음을 보여줍니다.



### Multi-Aggregator Time-Warping Heterogeneous Graph Neural Network for Personalized Micro-Video Recommendation (https://arxiv.org/abs/2501.02666)
- **What's New**: 이 논문은 개인화된 뉴스 성격의 마이크로 비디오 추천을 위한 새로운 모델인 MTHGNN을 제안합니다. 기존의 추천 시스템은 마이크로 비디오의 특성을 충분히 고려하지 못했으며, 이 모델은 사용자의 연속적인 세션을 기반으로 합니다. 특히, 마이크로 비디오의 사회적 상호작용 및 다중 모달 특성을 포착하여 추천 정확도를 향상시키는 데 중점을 두고 있습니다.

- **Technical Details**: MTHGNN 모델은 방향성 시간이 왜곡된 이질 그래프를 구축하여 사용자의 다중 상호작용을 설명합니다. 여기서 Relational Heterogeneous Message Passing aggregator와 Attention Heterogeneous Message Passing aggregator가 사용되어 복잡한 이질 정보를 노드에 내장합니다. 또한, 시간에 따른 사용자의 선호 변화를 고려하기 위해 다중 모달 이질 그래프를 연속 세션으로 나누어 과거 및 현재의 관심사를 융합합니다.

- **Performance Highlights**: 실험 결과, MTHGNN 모델은 TikTok 및 MovieLen과 같은 실제 세계 데이터셋에서 최신 기술 대비 우수한 성능을 보였습니다. 이 모델은 마이크로 비디오 추천뿐만 아니라 장기 영화 추천에서도 높은 성능을 나타내, 복잡한 사회적 관계와 동적 변화를 효율적으로 반영했다고 평가되었습니다. 이는 마이크로 비디오 추천의 성능 향상을 위한 중요한 기여로 해석됩니다.



### Interactive Information Need Prediction with Intent and Contex (https://arxiv.org/abs/2501.02635)
- **What's New**: 이 논문에서는 사용자가 사전 검색 컨텍스트를 선택하고 부분적인 검색 의도를 지정하여 정보 필요를 상호작용적으로 예측하는 방법을 탐구합니다. 이는 기존의 정보 검색 시스템의 한계를 극복하고, 사용자가 원하는 정보를 보다 정확하게 찾을 수 있도록 돕는 새로운 접근 방식을 제안합니다. 특히, 사용자 제공의 부분적인 검색 의도가 큰 사전 검색 컨텍스트를 완화하는 데 도움이 될 수 있음을 발견했습니다. 이러한 프레임워크는 실제 애플리케이션에 적합하고, 잠재적인 활용 가능성이 큽니다.

- **Technical Details**: 연구에서는 다양한 최신 generative language 모델을 통해 질문 생성과 검색 모델을 통한 답변 검색을 비교 분석합니다. 이들은 사용자가 선택한 사전 검색 컨텍스트의 양과 지정된 부분 의도가 정보 필요 예측에 미치는 영향을 조사합니다. 사용자 지정 부분 검색 의도를 담은 인터페이스는 사용자가 원하는 정보를 더욱 자연스럽고 효과적으로 표현할 수 있도록 돕습니다. 이는 정보 검색의 효율성을 높이는 방향으로 기여할 것입니다.

- **Performance Highlights**: 연구 결과, 사전 검색 컨텍스트와 부분 검색 의도의 조합이 정보 필요 예측 정확도를 향상시킬 수 있는 가능성을 보여주었습니다. 특히, 실험적 결과는 기존의 검색 환경에서 보다 명확한 정보 요구를 검출하는 데 이점이 있음을 나타냅니다. 또한 연구팀은 향후 연구를 위한 코드와 데이터 세트를 공개하였으며, 이는 다른 연구자들이 이 주제에 대한 추가 연구를 수행하는 데 중요한 기반이 될 것입니다.



### Citation Structural Diversity: A Novel and Concise Metric Combining Structure and Semantics for Literature Evaluation (https://arxiv.org/abs/2501.02429)
Comments:
          18 pages, 10 figures

- **What's New**: 이 논문은 Citation Structural Diversity라는 새로운 문헌 평가 모델을 제안하며, 전통적인 문헌 평가 방식의 한계를 극복하고자 합니다. 특히, 인용 네트워크의 구조적 다양성을 통한 평가 지표의 유효성을 검증하고, 문헌의 인용 빈도 및 장기적인 학문적 영향력을 분석합니다. 연구 결과, 높은 인용 구조적 다양성을 가진 문헌이 더 높은 인용 빈도와 지속적 학문적 영향력을 보여주었음을 강조합니다.

- **Technical Details**: 문헌 평가 모델은 인용 구조적 특징(citation structural features)과 의미 정보(semantic information)를 통합하여 복잡한 문헌 간의 관계를 효과적으로 표현하는 데 중점을 둡니다. Citation networks는 학술 논문을 노드로 보고, 인용 관계를 엣지로 나타내서, 지식의 전파와 교류 양상을 복잡하게 네트워크로 형성합니다. 또한, 이 연구는 데이터 그룹화(data grouping) 및 10년 간의 인용 트렌드 분석을 포함하여 제안된 모델의 실제 애플리케이션 가능성을 검증합니다.

- **Performance Highlights**: 광범위한 실험을 통해 제안된 모델의 효과성을 입증하였습니다. 이 모델은 제한된 데이터 또는 불완전한 네트워크 상황에서도 우수한 평가 성능을 유지하며, 다차원 평가 프레임워크를 제공하여 문헌 간의 복잡한 관계를 더욱 효과적으로 대표할 수 있습니다. 마지막으로, 학제 간 연구를 식별하는 데 중요한 역할을 할 것으로 기대됩니다.



### GenTREC: The First Test Collection Generated by Large Language Models for Evaluating Information Retrieval Systems (https://arxiv.org/abs/2501.02408)
- **What's New**: 이 논문에서는 LLM(대규모 언어 모델)을 활용하여 수동적 관련성 판단 없이 완전히 생성된 테스트 컬렉션인 GenTREC을 제안합니다. 기존의 IR(정보 검색) 시스템 평가 방법론과 비교할 때, GenTREC은 문서가 생성되는 과정에서 관련성을 자동으로 판단하므로, 많은 시간과 비용을 절감할 수 있습니다. 이 연구는 LLM이 문서 및 관련성 판단 생성을 통해 평가를 혁신할 가능성을 보여줍니다.

- **Technical Details**: GenTREC 컬렉션은 총 96,196개의 문서와 300개의 주제, 18,964개의 관련성 '판단'을 포함하고 있습니다. 이 방법론은 TREC의 기존 검색 주제를 사용하여 하위 주제를 생성하고, 이에 따라 문서를 생성하는 방식으로 작동합니다. 연구 결과에 따르면, 생성된 문서는 83%의 경우 실제로 주제와 관련성이 있는 것으로 나타났으며, P@100과 MAP, RPrec와 같은 지표에서도 기존 TREC 컬렉션과 유사한 순위를 보였습니다.

- **Performance Highlights**: GenTREC을 사용한 IR 시스템의 순위는 기존 TREC 컬렉션과 매우 유사하게 나타났으며, 특히 P@100에 대해 높은 상관관계를 보였습니다. 그러나 MAP 및 RPrec 지표의 순위 상관관계는 0.9 미만으로 낮은 성과를 보였지만, 기존 TREC 컬렉션들도 이 기준에 미치지 못하는 경우가 많아 결과적으로 유사한 경향을 보였습니다. 이 연구는 GenTREC이 정보 검색 시스템 평가의 미래에 중요한 기여를 할 것으로 기대합니다.



### Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation (https://arxiv.org/abs/2501.02226)
Comments:
          Preprint. Under review

- **What's New**: 이번 연구에서는 K-RagRec라는 새로운 프레임워크를 제안하여 추천 시스템의 성능을 향상시키기 위해 지식 그래프(knowledge graph, KG)에서 신뢰할 수 있는 지식을 효율적으로 검색합니다. K-RagRec는 구조적 정보를 활용하여 LLM 기반의 추천 시스템의 한계를 극복하고 있으며, 이를 통해 추천 정확도를 높입니다. 특히, 사전 지식에 의존하는 기존 RAG의 한계를 극복하기 위해, 고품질의 최신 구조 정보를 가져오는 방법론을 개발했습니다.

- **Technical Details**: K-RagRec 프레임워크는 먼저 항목의 KG에서 지식 서브 그래프(knowledge sub-graph)를 색인화하여 지식 벡터 데이터베이스를 구축합니다. 이후, 항목의 인기도에 따라 검색 항목을 결정하는 적응형 검색 정책(adaptive retrieval policy)을 설계하여, 관련 서브 그래프를 효율적으로 검색합니다. 마지막으로, GNN과 projector를 통해 검색된 지식 서브 그래프를 LLM의 의미 공간(semantic space)과 정렬하여 추천을 강화합니다.

- **Performance Highlights**: K-RagRec 프레임워크는 다양한 실제 데이터 세트를 통해 효과성을 평가하였으며, 기존 LLM 기반 추천 시스템보다 빠른 처리 속도와 더 높은 추천 정확성을 보여주었습니다. 구조적 지식을 효과적으로 활용함으로써 정보 과부하를 완화하고 사용자 맞춤형 추천을 향상시켰습니다. 실험 결과는 K-RagRec가 추천 시스템 성능을 크게 개선할 수 있는 잠재력이 있음을 보여줍니다.



### The Application of Large Language Models in Recommendation Systems (https://arxiv.org/abs/2501.02178)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 추천 시스템에 대한 적용 가능성을 탐색합니다. 기존의 추천 방법들은 콜드 스타트 문제와 데이터의 희소성 등의 제약을 가지고 있었으나, LLM들은 비구조화된 데이터 소스를 활용하여 더 정확하고 관련성 있는 추천을 가능하게 합니다. GPT-4와 같은 모델은 사용자 리뷰와 소셜 상호작용을 분석하여 추천의 다양성과 사용자 참여를 높일 수 있는 잠재력을 가지고 있습니다.

- **Technical Details**: 추천 시스템의 기본 개념 및 LLM의 활용 방식에 대해 설명합니다. 추천 시스템은 사용자 행동, 선호도 및 상호작용 이력을 바탕으로 개인 맞춤형 항목을 추천하는 복잡한 알고리즘입니다. 연구에서는 협업 필터링, 콘텐츠 기반 추천, 하이브리드 방법과 같은 다양한 기술을 분석하며, LLM은 이러한 시스템들이 비구조화된 정보를 처리하는 데 강점을 발휘함을 강조합니다.

- **Performance Highlights**: LLM의 통합을 통한 고차원적인 개인화의 개선을 다룹니다. LLM은 이전 상호작용을 바탕으로 고급 개인 맞춤형 추천을 생성하여 사용자 신뢰도를 높이고 사용자 경험을 향상시킵니다. 이 연구는 LLM 도입 시의 기술적 과제와 향후 연구 방향에 대해서도 논의하며, 추천 시스템의 다음 세대를 개발하는 데 중요한 통찰력을 제공합니다.



### The Efficiency vs. Accuracy Trade-off: Optimizing RAG-Enhanced LLM Recommender Systems Using Multi-Head Early Ex (https://arxiv.org/abs/2501.02173)
- **What's New**: 이 논문은 클릭률 예측(CTR) 문제를 해결하기 위한 대규모 언어 모델(LLM)의 최적화 프레임워크를 제시하며, 이를 통해 계산 효율성과 예측 정확성 간의 균형을 맞추고 있습니다. Retrieval-Augmented Generation(RAG)와 다중 헤드 얼리 종료(Multi-Head Early Exit) 아키텍처를 결합하여 성능을 동시에 향상시키는 혁신적인 접근 방식을 선보입니다. 그래프 합성곱 네트워크(Graph Convolutional Networks, GCN)를 활용한 효과적인 검색 메커니즘을 도입하여 데이터 검색 시간을 현저하게 줄이고 있습니다.

- **Technical Details**: 제안된 GCN-Retriever는 사용자-아이템 그래프에서 다중 차원 상호작용 정보를 포착하여 특성 임베딩(feature embedding)을 수행합니다. 이를 통해 계산 복잡도를 줄여 실시간 반응성이 필요로 하는 추천 시스템에서의 LLM 성능을 향상시킵니다. 또한, 다중 헤드 얼리 종료 전략을 이용하여 예측 신뢰도에 따라 모델 추론을 동적으로 종료함으로써 불필요한 계산 부하를 줄이는 데에도 기여합니다.

- **Performance Highlights**: 실험에서는 제안된 아키텍처가 계산 시간을 줄이면서도 추천 정확도를 유지하거나 개선할 수 있음을 보여주었습니다. 이는 상업 시스템에 적합한 실시간 LLM 배포를 위한 새로운 표준을 확립하는 데 기여합니다. 따라서 기존의 LLM 기반 클릭률 예측 방법들보다 우수한 성능을 나타내고 있습니다.



### Graph-based Retrieval Augmented Generation for Dynamic Few-shot Text Classification (https://arxiv.org/abs/2501.02844)
- **What's New**: 이 논문에서는 동적 소수 샷 텍스트 분류(dynamic few-shot text classification)를 위한 새로운 그래프 기반 온라인 검색-Augmented Generation 프레임워크인 GORAG을 제안합니다. GORAG은 텍스트의 모든 대상에 대해 사이드 정보를 추출하여 적응형 정보 그래프를 구축하고 유지합니다. 기존 접근 방식의 한계를 극복하면서 LLM(Large Language Model)의 효과를 극대화하는 것을 목표로 합니다.

- **Technical Details**: GORAG은 LLM의 키워드를 사용하여 텍스트에서 키워드를 추출하고, 이러한 키워드를 텍스트의 실제 레이블과 연결하여 키워드와 레이블 간의 관계를 나타냅니다. 또한, GORAG은 가중 엣지 메커니즘을 사용하여 그래프의 엣지를 인덱싱할 때 키워드의 중요성과 레이블과의 관련성을 기반으로 엣지 가중치를 부여합니다. 정보를 검색하기 위해 최소 비용 신장 트리(minimum-cost spanning tree)를 구성하여 각 텍스트에 대해 후보 레이블을 검색합니다.

- **Performance Highlights**: 실험 평가 결과 GORAG은 기존 접근 방식보다 더 포괄적이고 정확한 맥락 정보를 제공하여 성능이 향상된 것으로 나타났습니다. 기존 모델들이 가지는 데이터의 상호 연관성과 중복 문제를 해결하여 동적 변경이 빈번한 레이블에 대해서도 효과적으로 분류할 수 있는 능력을 보여주었습니다. 이러한 GORAG의 접근은 소수의 레이블만 있는 상황에서도 높은 성능을 유지하도록 돕습니다.



### Integrating Language-Image Prior into EEG Decoding for Cross-Task Zero-Calibration RSVP-BCI (https://arxiv.org/abs/2501.02841)
Comments:
          15 pages, 11 figures

- **What's New**: 이번 연구는 Rapid Serial Visual Presentation (RSVP) 기반 뇌-컴퓨터 인터페이스 (BCI)의 새로운 접근법을 제안합니다. 연구에서는 cross-task zero-calibration RSVP decoding 성능을 향상시키기 위해 EEG와 언어-이미지 prior 기능을 융합한 Transformer인 ELIPformer를 개발했습니다. 이를 통해 다양한 RSVP 작업에 적용 가능한 모델을 구현하고, 시간 소모적인 교정 없이 새로운 목표를 탐지할 수 있게 됩니다.

- **Technical Details**: 본 연구는 세 가지 RSVP 작업을 설계하고, 해당 작업에 대한 EEG 신호와 자극 이미지로 구성된 오픈 소스 데이터셋을 구축했습니다. ELIPformer 모델은 언어-이미지 사전 학습을 통해 EEG 신호와 언어-이미지 특성을 효과적으로 융합할 수 있는 구조를 가지고 있습니다. 또한, Cross Bidirectional Attention 메커니즘을 통해 EEG와 언어-이미지 특성 간의 효율적인 상호작용을 도모합니다.

- **Performance Highlights**: 실험 결과, 제안된 ELIPformer 모델은 cross-task zero-calibration RSVP decoding에서 우수한 성능을 보였습니다. 이는 RSVP-BCI 시스템의 연구에서 실제 응용으로의 전환을 촉진하여 다양한 시나리오에서 빠르고 효율적으로 다양한 범주의 목표를 탐지할 수 있도록 지원합니다. 또한, 오픈 소스 데이터셋이 공개되어 있어 연구자들이 쉽게 접근할 수 있는 장점도 제공합니다.



### Forward Once for All: Structural Parameterized Adaptation for Efficient Cloud-coordinated On-device Recommendation (https://arxiv.org/abs/2501.02837)
Comments:
          Accepted by KDD 2025

- **What's New**: 본 논문은 클라우드 중심의 추천 시스템에서 발생할 수 있는 네트워크 대역폭 요구 사항 및 개인정보 보호 위험을 줄이기 위해, 기존의 맞춤형 모델 아키텍처의 중요성을 강조하며 다이나믹한 디바이스 전용 네트워크 구성을 제안하는 Forward-OFA를 소개합니다. 이른바 On-device Recommendation(온디바이스 추천)으로, 사용자 데이터를 클라우드로 전송하는 대신 현지에서 추천의 순위를 재조정하여 네트워크 부하를 줄이고 제품 추천의 정확성을 개선할 수 있습니다.

- **Technical Details**: Forward-OFA는 구조 컨트롤러를 통해 각 디바이스에 필요한 블록의 조합을 선택적으로 결정하는 구조를 가지고 있습니다. 훈련 과정 중 이 조립된 이질적 구조는 공동 최적화되며, 구동 시 각 항목이 이질적 기울기를 수신하지 못하도록 파라미터를 설정합니다. 이 방법은 Gradient Conflict(기울기 충돌)를 피하고, 실시간 행동과 조립된 네트워크의 파라미터 간의 구조적 매핑을 통해 적응을 지원합니다.

- **Performance Highlights**: 다양한 실제 데이터셋을 이용한 실험 결과, Forward-OFA가 효과적이고 효율적인 성능을 발휘함을 밝힙니다. 특히, 사용자 개인정보를 보호하고 여러 디바이스에서의 최적화된 적응을 통해 큰 성과를 보였으며, 추가적인 네트워크 시각화 및 사례 연구를 통해 Forward-OFA의 작동 및 다른 네트워크와의 영향력을 명확히 분석했습니다.



### Can Impressions of Music be Extracted from Thumbnail Images? (https://arxiv.org/abs/2501.02511)
Comments:
          Accepted at NLP4MusA 2024

- **What's New**: 최근 음악 검색 및 생성 시스템을 위한 머신러닝 모델에 대한 연구가 증가하고 있지만, 음악 데이터와 해당 자연어 설명(music captions)으로 구성된 대규모 공개 데이터셋이 부족한 실정입니다. 특히, 트랙을 듣기에 적합한 상황이나 이에 따른 감정과 같은 비음악적 정보는 음악 설명에 있어 필수적입니다. 이를 해결하기 위해, 본 연구에서는 음악 썸네일 이미지를 활용하여 비음악적 요소를 포함한 음악 캡션 데이터를 생성하는 방법을 제안하였습니다.

- **Technical Details**: 이 연구는 YouTube와 같은 플랫폼에서 음악 클립과 연관된 썸네일 이미지를 중심으로 진행되었습니다. 제안된 방법에서는 먼저 썸네일 이미지를 대형 비전-언어 모델(LVLM)에 입력한 다음, LVLM이 신중하게 제작된 프롬프트를 통해 음악 캡션을 생성합니다. 이러한 과정은 비음악적 요소를 포함하는 음악 캡션의 자동 생성을 가능하게 하며, 생성된 캡션은 기존의 방법보다는 비음악적 정보를 효과적으로 포함합니다.

- **Performance Highlights**: 약 360,000개의 캡션으로 개발된 데이터셋이 공개되었으며, 이를 통해 음악 검색 모델을 학습시키고 그 효과성을 평가하였습니다. 인간 평가를 통해 제안된 방법이 기존 방법들보다 우수한 성능을 나타낸 것으로 확인되었습니다. 이 연구는 음악 설명 데이터의 다양성을 확보하고, 음악 검색 모델의 품질을 향상시키는 데 중요한 기여를 할 것으로 기대됩니다.



### DiffGraph: Heterogeneous Graph Diffusion Mod (https://arxiv.org/abs/2501.02313)
Comments:
          This paper is accepted by WSDM'2025

- **What's New**: 이번 연구는 Heterogeneous Graph Diffusion Model(DiffGraph)을 소개하며, 이 모델은 복잡한 이질적 구조를 효과적으로 다루기 위한 혁신적인 방법론을 제공한다. 특히, cross-view denoising 전략을 통해 보조적인 이질적 데이터를 목표의 의미 공간으로 변환하여 과제와 관련된 정보를 정밀하게 증류할 수 있도록 설계하였다. 또한, 고차원적 이질적 그래프 확산 메커니즘을 채택하여 잡음을 효과적으로 관리하는 새로운 전후 확산 프로세스를 구현하였다.

- **Technical Details**: DiffGraph는 이질적 그래프의 노드 및 엣지가 포함된 목표 서브그래프를 식별하고, 잔여 구조를 보조 그래프로 간주하여 특성을 강화하는 방식으로 작동한다. 이 모델은 노이즈가 추가되고 제거되는 과정이 있는 이중적 과정을 채택하고, 이를 통해 이질적 그래프 데이터의 복잡한 노이즈 분포와 다양한 관계 유형 간의 의미적 전환을 모델링한다. 이러한 접근은 기존 그래프 생성의 제한을 극복하고, 이질적 데이터 모델링의 편향 없는 능력을 significantly 향상시킨다.

- **Performance Highlights**: DiffGraph는 공개 데이터셋과 산업 데이터셋에서 엄격한 실험 검증을 통해 링크 예측 및 노드 분류 과제에서 기존 방법론을 지속적으로 초과하며, 이질적 그래프 처리의 강건성과 효율성에 대한 새로운 기준을 마련하였다. 이 연구는 향후 연구 및 실제 응용 프로그램에 있어 이질적 그래프 학습의 성능을 향상시키는 데 기여할 수 있는 가능성을 제시하고 있다.



New uploads on arXiv(cs.CV)

### LargeAD: Large-Scale Cross-Sensor Data Pretraining for Autonomous Driving (https://arxiv.org/abs/2501.04005)
Comments:
          Preprint; 16 pages, 7 figures, 8 tables; Project Page at this https URL

- **What's New**: 최근 비전 기반 모델(VFMs)의 발전은 2D 시각 인식을 혁신적으로 변화시켰지만, 이러한 기술이 3D 장면 이해, 특히 자율주행 애플리케이션에 적용될 가능성은 아직 탐구되지 않았습니다. 본 논문에서는 다양한 실제 주행 데이터셋에 대해 대규모 3D 프리트레이닝을 위한 범용 프레임워크인 LargeAD를 소개합니다. 이 프레임워크는 VFMs를 활용하여 2D 이미지에서 의미론적으로 풍부한 슈퍼픽셀을 추출하고, 이를 LiDAR 포인트 클라우드와 정렬하여 고품질의 대조 샘플을 생성합니다.

- **Technical Details**: LargeAD 프레임워크는 VFMs 기반의 슈퍼픽셀 생성을 통해 상세한 의미론적 표현을 제공하며, VFM 보조 대조 학습 전략을 통해 다중 모달 피쳐를 정렬합니다. 또한, 슈퍼포인트의 시간적 일관성을 유지하여 시간에 걸쳐 안정적인 표현을 제공합니다. 마지막으로, 다양한 LiDAR 구성에 대한 일반화를 촉진하기 위해 다원 데이터 프리트레이닝을 적용하여 실세계 자율 주행 시나리오에서의 효율성과 강건성을 높입니다.

- **Performance Highlights**: 우리의 접근 방식은 LiDAR 기반 분할 및 물체 탐지의 선형 탐색(linear probing) 및 미세 조정(fine-tuning) 작업에서 기존 최첨단 방법들에 비해 중대한 성능 향상을 이끌어냈습니다. 11개의 대규모 다중 모드 데이터셋에 대한 실험 결과는 본 프레임워크의 우수한 성능을 강조하며, 자율주행 시나리오에서의 적응성, 효율성 및 강건성을 입증합니다.



### LiMoE: Mixture of LiDAR Representation Learners from Automotive Scenes (https://arxiv.org/abs/2501.04004)
Comments:
          Preprint; 26 pages, 17 figures, 7 tables; Project Page at this https URL

- **What's New**: 이번 연구에서는 LiMoE라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 Mixture of Experts (MoE) 패러다임을 LiDAR 데이터 표현 학습에 통합하여 다양한 표현 기법, 즉 range 이미지, sparse voxel 및 원시 포인트를 결합합니다. LiMoE는 이러한 다양한 표현의 보완적인 강점을 활용하여 더 나은 데이터 특성을 제공할 수 있도록 설계되었습니다.

- **Technical Details**: LiMoE의 작동 방식은 세 가지 단계로 나뉘어 있습니다. 첫째, Image-to-LiDAR Pretraining 단계에서는 이미지로부터 지식을 전달받아 LiDAR 포인트에 반영합니다. 둘째, Contrastive Mixture Learning (CML)은 각 표현 간의 사전 훈련된 기능을 융합하여 보완적인 강점을 극대화합니다. 셋째, Semantic Mixture Supervision (SMS) 단계에서 여러 표현의 의미적 특성을 결합하여 하위 작업의 성능을 개선합니다.

- **Performance Highlights**: 11개의 대규모 LiDAR 데이터셋에 대한 실험 결과, LiMoE는 기존의 기법에 비해 현저한 성능 향상을 보여주었습니다. 특히, 세 가지 서로 다른 LiDAR 표현을 통합하였다는 점에서 유의미한 결과를 나타냈습니다. 연구 결과는 코드와 모델 체크포인트를 공개하여 다른 연구자들이 활용할 수 있도록 하였습니다.



### Are VLMs Ready for Autonomous Driving? An Empirical Study from the Reliability, Data, and Metric Perspectives (https://arxiv.org/abs/2501.04003)
Comments:
          Preprint; 41 pages, 32 figures, 16 tables; Project Page at this https URL

- **What's New**: 최근 Vision-Language Models (VLMs)의 발전이 자율 주행 분야에서 주목받고 있으며, 특히 자연어를 통해 해석 가능한 주행 결정을 생성하는 데 활용되고 있습니다. 그러나 VLMs가 본질적으로 시각적으로 기반을 둔 신뢰할 수 있고 해석 가능한 설명을 제공한다는 가정은 충분히 검토되지 않았습니다. 이 연구에서는 DriveBench라는 새로운 벤치마크 데이터셋을 소개하여 17가지 설정에서 VLM의 신뢰성을 평가합니다.

- **Technical Details**: DriveBench는 19,200개의 프레임, 20,498개의 질문-답변 쌍, 세 가지 질문 유형, 네 가지 주류 주행 작업 및 12개의 인기 VLM을 포함한 데이터셋으로 구성되어 있습니다. 연구 결과 VLMs는 주로 일반 지식이나 텍스트 단서에서 파생된 그럴듯한 응답을 생성하며, 진정한 시각적 기반이 부족한 경우가 많아 위험한 상황에서의 신뢰성 문제가 드러났습니다.

- **Performance Highlights**: VLM은 다중 모드 추론(multi-modal reasoning)에서 어려움을 겪고 있으며 입력 데이터의 손상에 민감하게 반응하여 성능의 불일치를 보입니다. 연구진은 신뢰성을 높이기 위한 정교한 평가 지표를 제안하며, 오염 인식(corruption awareness)을 활용해 VLM의 신뢰성을 향상시킬 수 있는 가능성을 강조합니다. 이 벤치마크 툴킷은 공개적으로 접근 가능합니다.



### Extraction Of Cumulative Blobs From Dynamic Gestures (https://arxiv.org/abs/2501.04002)
- **What's New**: 이번 연구에서는 모션 캡처를 위한 간단한 야간 투시 카메라를 사용하는 새로운 제스처 인식 시스템을 제안합니다. 이 카메라는 인간에게는 보이지 않는 적외선빛(infrared light)을 방출하여 어두운 환경에서도 작동할 수 있는 기술적 한계를 극복합니다. 따라서 이 시스템은 기존 기준을 넘어 사용자와 컴퓨터 간의 의사소통 방식에 혁신을 가져올 수 있습니다.

- **Technical Details**: 연구는 Raspberry Pi와 OpenCV 모듈을 활용하여 제스처를 탐지하고 추적하는 Python 프로그램을 구현합니다. 야간 투시 카메라를 통해 수집된 비디오 스트림은 Raspberry Pi에게 전송되며, 여기서는 머신 러닝(machine learning) 알고리즘을 사용하여 그려진 패턴을 인식합니다. 이러한 방식으로 Raspberry Pi의 GPIO를 제어하여 다양한 활동을 수행할 수 있습니다.

- **Performance Highlights**: 제안된 시스템은 어두운 환경에서도 정확하게 제스처를 인식할 수 있는 능력을 보여줍니다. 이는 기존 제스처 인식 시스템의 한계를 극복하며, 사용자가 손을 사용하지 않고도 컴퓨터와 소통할 수 있는 가능성을 제시합니다. 따라서 이 기술은 홈 오토메이션, 게임, 또는 다양한 인터페이스에서의 유용성을 높이는 데 기여할 것으로 기대됩니다.



### Sa2VA: Marrying SAM2 with LLaVA for Dense Grounded Understanding of Images and Videos (https://arxiv.org/abs/2501.04001)
Comments:
          Project page: this https URL

- **What's New**: Sa2VA는 이미지와 비디오의 밀접한 이해를 위한 최초의 통합 모델로, 기존 멀티모달 대형 언어 모델(MLLM)과는 달리 다양한 이미지 및 비디오 작업을 지원합니다. 이 모델은 SAM-2와 LLaVA를 결합하여 텍스트, 이미지 및 비디오를 공유 LLM 토큰 공간에 통합하는 혁신적인 접근법을 제공합니다. 또한, Sa2VA는 지시문 토큰을 생성하여 SAM-2를 이끌어 정확한 마스크를 생성하게 하여 정적 및 동적 시각적 콘텐츠에 대한 기초적인 다중 모달 이해를 실현합니다.

- **Technical Details**: Sa2VA는 이미지와 비디오에 대한 여러 작업을 한 번의 훈련 설정에서 효과적으로 형성하는 방법을 제시합니다. 이를 위해 멀티모달 입력에 대한 다양한 작업을 지시문 튜닝 형식으로 통합합니다. LLM의 유연한 토큰 길이 처리 기능을 활용하여 모든 입력 이미지를 시각적 토큰으로 처리하고, SAM-2의 디코더와 메모리 모듈을 동결하여 모델 업데이트를 용이하게 했습니다. 또한, 자동 레이블링된 Ref-SAV 데이터셋을 도입하여 복잡한 비디오 환경에서의 성능을 Benchmarked합니다.

- **Performance Highlights**: Sa2VA는 6개의 참조 이미지 및 비디오 분할 데이터셋에서 최첨단 결과를 달성하며, 이전 MLLMs에 비해 강력한 이미지 및 비디오 대화 능력을 보유하고 있습니다. Ref-SAV 데이터셋에서 Sa2VA는 제로샷 테스트 설정에서 지난 방법들보다 15% 이상의 성능 향상을 달성했습니다. 이러한 결과는 Sa2VA가 복잡한 실제 응용 프로그램을 위한 강력한 기준선을 설정하고 있음을 보여줍니다.



### NeuralSVG: An Implicit Representation for Text-to-Vector Generation (https://arxiv.org/abs/2501.03992)
Comments:
          Project Page: this https URL

- **What's New**: 최근 그래픽 디자인에서 벡터 그래픽스의 중요성이 강조되고 있으며, 이 논문에서는 NeuralSVG라는 새로운 기법을 소개합니다. NeuralSVG는 텍스트 프롬프트로부터 벡터 그래픽스를 생성하기 위한 암묵적 신경 표현 방식으로, 레이어 구조를 중요시합니다. 이러한 접근은 과거의 방법들이 벡터 그래픽스의 본질적인 특성을 충분히 반영하지 못한 문제를 해결하는 데 중점을 두고 있습니다.

- **Technical Details**: NeuralSVG는 Neural Radiance Fields(NeRFs)의 아이디어에서 영감을 받아, 소형 MLP 네트워크의 가중치로 전체 장면을 인코딩합니다. 이 과정에서 Score Distillation Sampling(SDS) 기법을 활용하여 네트워크를 최적화하며, 레이어 구조를 촉진하기 위해 드롭아웃 기반의 정규화 기법을 도입합니다. 이러한 방식은 각 도형이 독립적인 의미를 가지도록 하여 최종 SVG의 질을 높입니다.

- **Performance Highlights**: NeuralSVG는 다양한 평가에서 기존 방법들보다 더 나은 성능을 보여주었습니다. 이 시스템은 단일 단계에서 의미 있는 개별 도형을 생성하고, 사용자 입력에 따라 SVG를 동적으로 조정할 수 있는 기능을 갖추고 있습니다. 특히, SVG의 사용자 정의 색상이나 형태 응답을 가능하게 하여 더 유연한 디자인과 구현이 가능합니다.



### Temporal Feature Weaving for Neonatal Echocardiographic Viewpoint Video Classification (https://arxiv.org/abs/2501.03967)
Comments:
          Accepted to ISBI 2025

- **What's New**: 이 연구는 신생아 심초음파의 시점 분류(viewpoint classification)를 위한 혁신적인 접근 방식을 제안합니다. 기존의 이미지 분류(image classification) 방식을 넘어서 비디오 분류(video classification)로 다룸으로써 더 나은 결과를 얻을 수 있음을 보여줍니다. 또한, 전문적으로 주석이 달린 신생아 심초음파 데이터셋(Neonatal Echocardiogram Dataset, NED)을 공개하여 이 분야의 미래 연구를 촉진하고자 합니다.

- **Technical Details**: 우리는 CNN-GRU 아키텍처를 기반으로 한 새로운 시각적 프레임 전환 방법(Temporal Feature Weaving, TFW)을 제안합니다. 이 접근 방식은 공간 정보(spatial information)와 시간 정보(temporal information)를 모두 활용하여 이미지 분류의 정확도를 4.33% 향상시킵니다. NED 데이터셋은 16개 뷰포인트와 관련된 1049개의 비디오로 구성되어 있으며, 이들은 실제 환자 스캔에서 수집된 것입니다.

- **Performance Highlights**: 제안된 방법은 실시간으로 최신 스마트폰에서 실행 가능하며, 추가적인 계산 비용이 발생하지 않으면서 30백만 개의 파라미터를 가집니다. F1-Score에서는 4.53%의 개선을 보이며, 이는 기존의 이미지 분류 기법들에 비해 높은 정확성을 나타냅니다. 이러한 특성들은 신생아 심초음파의 접근성과 교육 경험을 향상시키는 데 기여할 것으로 기대됩니다.



### Visual question answering: from early developments to recent advances -- a survey (https://arxiv.org/abs/2501.03939)
Comments:
          20

- **What's New**: 이번 논문에서는 Visual Question Answering (VQA) 연구 분야의 발전과 최신 동향을 다루고 있습니다. 저자들은 VQA 아키텍처의 세분화된 분류법을 제시하여 다양한 설계 선택과 주요 구성 요소를 비교할 수 있도록 했습니다. 또, 최근의 대규모 데이터셋 및 평가 지표를 분석하고, VQA 시스템을 적용할 수 있는 실제 사례를 탐구합니다.

- **Technical Details**: VQA는 시각 및 언어 처리 기술을 통합하여 기계가 시각적 콘텐츠에 대한 질문에 답할 수 있도록 하는 멀티모달 컴퓨팅을 대표하는 작업입니다. 연구자들은 VQA 시스템의 성능을 높이기 위해 Vision Encoder, Language Encoder, 그리고 이미지와 질문의 특징을 결합하는 다양한 기술을 분석했습니다. 또한, LVLM(대형 시각 언어 모델) 위주로 VQA 모델의 혁신적인 발전도 다루고 있습니다.

- **Performance Highlights**: VQA는 의료 이미지 진단, 고객 서비스, 교육 도구 등 여러 응용 분야에서 두각을 나타내고 있습니다. 이 논문에서는 VQA 모델의 정확도를 기준으로 기존의 VQA 모델들을 비교하고, 향후 연구 방향과 발전 가능성을 제시합니다. VQA 기술은 특히 치료 및 교육 분야에서 복잡한 멀티모달 대화 시스템이나 오픈 도메인 VQA로의 실용적 적용이 확대되고 있습니다.



### CoStruction: Conjoint radiance field optimization for urban scene reconStruction with limited image overlap (https://arxiv.org/abs/2501.03932)
- **What's New**: 이 논문에서는 대규모 주행 시퀀스를 위해 설계된 새로운 혼합 임의 표면 재구성 방법인 CoStruction을 소개합니다. 이는 이미지의 한정된 중첩으로 인한 모호성을해소하기 위해 교차 표현 불확실성 추정을 활용합니다. CoStruction은 정확한 재구성을 위해 방사선 필드의 공동 최적화와 가이드 샘플링을 수행하며, 복잡한 도시 상황에서의 재구성을 향상시킵니다.

- **Technical Details**: CoStruction은 방사성과 SDF(Signed Distance Function) 필드를 공동 최적화하여 복잡한 도시 경관을 효과적으로 재구성합니다. 훈련 과정에서 우리는 Eikonal 제약 조건을 통합하여 SDF 표면 재구성을 정밀하게 수행하며, 각 훈련 단계에서 색상화된 메시를 추출하여 구조적 세부정보를 보존합니다. 오프라인으로 훈련된 샘플을 통해 불확실한 기하학적 단서를 제거하는 cross-representation uncertainty estimation이 주효합니다.

- **Performance Highlights**: 우리는 KITTI-360, Pandaset, Waymo Open Dataset 및 nuScenes의 네 가지 공개 주행 데이터 세트에서 CoStruction의 성능을 평가했습니다. 실험 결과, CoStruction은 제한된 이미지 중첩에도 불구하고 복잡한 도시 기하학을 정확하게 재구성하며 기존의 최신 기술들을 초월하는 우수한 성능을 보였습니다. 우리의 접근 방식은 또한 대규모 주행 시퀀스의 정확한 재구성을 가능하게 하여 자율 주행 응용 분야에서의 활용 가능성을 더욱 확대합니다.



### Magic Mirror: ID-Preserved Video Generation in Video Diffusion Transformers (https://arxiv.org/abs/2501.03931)
Comments:
          It is best viewed in Acrobat. Project Page: this https URL

- **What's New**: 이번 연구에서는 동적인 움직임과 영화 수준의 품질을 가진 정체성 보존 비디오를 생성하기 위한 프레임워크인 Magic Mirror를 소개합니다. 기존의 비디오 생성 모델은 정체성과 자연스러운 움직임을 동시에 유지하기 어려운 한계가 있었으며, Magic Mirror는 이를 해결하기 위한 접근 방식을 제시합니다. 이 프레임워크는 사람의 정체성을 유지하면서도 자연스러운 얼굴 움직임을 생성할 수 있는 혁신적인 방법입니다.

- **Technical Details**: Magic Mirror는 Video Diffusion Transformers를 기반으로 하며, 세 가지 핵심 구성 요소를 포함합니다. 첫째, 정체성과 구조적 특징을 포착하는 이중 분기 얼굴 특징 추출기를 도입하였습니다. 둘째, Conditioned Adaptive Normalization을 사용한 경량 크로스 모달 어댑터로 정체성 통합을 효율적으로 진행합니다. 셋째, 합성된 정체성 쌍과 비디오 데이터를 결합한 이단계 훈련 전략을 통해 훈련 데이터를 생성하는 방식입니다.

- **Performance Highlights**: 광범위한 실험을 통해 Magic Mirror가 정체성 일관성과 자연스러운 움직임을 효과적으로 균형 잡고 있음을 입증하였습니다. 기존 메소드들과 비교해 여러 지표에서 우수한 성능을 보여주고 최소한의 매개변수만 추가하여도 뛰어난 결과를 달성하였습니다. Magic Mirror는 개인화된 비디오 생성의 새로운 가능성을 제시하며, 고품질 비디오 합성을 위한 주요 기초가 될 것입니다.



### HYB-VITON: A Hybrid Approach to Virtual Try-On Combining Explicit and Implicit Warping (https://arxiv.org/abs/2501.03910)
Comments:
          Accepted at IEEE ICASSP 2025

- **What's New**: 본 논문에서는 HYB-VITON이라는 새로운 접근 방식을 제안합니다. 이는 명시적 왜곡(explicit warping)과 암시적 왜곡(implicit warping) 방법의 장점을 결합하여 실현될 수 있는 가능성이 큰 가상 착용 시스템을 위한 것입니다. 이 시스템은 의류를 착용한 고객의 모습을 시각화할 수 있게 해 줍니다.

- **Technical Details**: HYB-VITON은 전통적인 훈련 설정을 기반으로 하며, 쌍 데이터(paired data)를 사용하여 의류와 인물 이미지를 함께 처리합니다. 이 방법은 의류 무관 사람 이미지를 사용하여 의류의 누락된 부분을 메꾸는 diffusion model을 활용합니다. 다층적인 프로세스를 통하여 왜곡된 의류의 장점을 최대로 활용하는 것이 핵심입니다.

- **Performance Highlights**: 실험 결과 HYB-VITON은 최근의 diffusion 기반 방법들보다 의류의 세부사항을 더 잘 보존하며, 최고 수준의 명시적 왜곡 방법보다 더 사실적인 결과를 생성하는 것으로 나타났습니다. 이 연구 결과는 가상 착용 시스템 분야에서 중요한 발전을 촉진하고, 향후 암시적 왜곡 개발에 있어 세부 사항 보존의 기준을 설정하는 데 기여할 것입니다.



### LLaVA-Mini: Efficient Image and Video Large Multimodal Models with One Vision Token (https://arxiv.org/abs/2501.03895)
Comments:
          Code: this https URL Model: this https URL

- **What's New**: 본 논문은 LMMs (Large Multimodal Models)의 효율성을 향상시킨 LLaVA-Mini를 소개합니다. 기존의 LMM 모델들이 비디오 및 이미지 이해에서 큰 연산 비용을 처리하는 것과 달리, LLaVA-Mini는 비전 토큰의 수를 최소화하여 연산 복잡도를 줄입니다. 특히, LLaVA-Mini는 단 하나의 비전 토큰만을 사용하면서도 높은 성능을 유지하는 혁신적인 접근 방식을 선택했습니다.

- **Technical Details**: LLaVA-Mini는 비전 데이터와 텍스트 정보를 사전에 융합하는 모듈을 도입하여, LLM (Large Language Model)으로 입력되는 비전 토큰의 수를 극도로 압축합니다. 이 모델은 실험을 통해 1개의 비전 토큰을 사용하여 576개 비전 토큰 대비 0.17%의 압축 비율을 보여주며, FLOPs를 77% 줄이고 GPU 메모리 사용량을 0.6MB로 낮춥니다. 이는 고해상도 이미지와 긴 비디오 처리에서 지연 시간을 줄이는 데 크게 기여합니다.

- **Performance Highlights**: LLaVA-Mini는 11개 이미지 기반 및 7개 비디오 기반 벤치마크에서 실험을 통해 LLaVA-v1.5보다 우수한 성능을 발휘하였습니다. 특히, 이미지 이해의 지연 시간을 100ms에서 40ms로 단축시킬 수 있었으며, 또한 24GB의 메모리를 가진 NVIDIA RTX 3090에서 10,000 프레임 이상의 긴 비디오를 처리할 수 있는 가능성을 보여주었습니다. 이러한 효율성을 통해 LLaVA-Mini는 실시간 다중 모달 상호작용의 새로운 길을 열었습니다.



### Superpixel Boundary Correction for Weakly-Supervised Semantic Segmentation on Histopathology Images (https://arxiv.org/abs/2501.03891)
Comments:
          7 pages, 4 figures

- **What's New**: 본 연구는 약한 감독 기반 의미 분할(Weakly Supervised Semantic Segmentation, WSSS)의 문제점을 해결하기 위해 새로운 다중 수준 슈퍼픽셀 교정 알고리즘을 제안합니다. 이 알고리즘은 슈퍼픽셀 클러스터링(Superpixel Clustering) 및 floodfill을 사용하여 Class Activation Map(CAM)의 경계를 세밀하게 조정합니다. 이를 통해 유방암 분할 데이터셋에서 71.08%의 mIoU(Mean Intersection over Union)를 달성하여 종양 미세환경의 경계 구분을 크게 향상시켰습니다.

- **Technical Details**: 이 방법은 암 진단 및 분류의 핵심인 조직(segment) 분할을 다중 레이블 분류 문제로 간주하며, CNN 기반 분류 모델을 사용합니다. 연구에서는 이미징 패치 내에서 여러 조직 유형을 포함할 수 있도록 하고, 원본 이미지의 자연 경계를 반영하는 슈퍼픽셀 클러스터링을 활용합니다. CAM의 다양한 깊이를 합쳐 경계의 스케일 문제를 해결하고, 이에 따른 손실을 줄이도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 알고리즘은 기존 WSSS 방법보다 우수한 성능을 보이며, 특히 BCSS(Breast Cancer Segmentation Study) 데이터셋에서 탁월한 결과를 나타냈습니다. 본 연구는 병리 이미지 분할을 더 가볍고 효율적으로 만들어 컴퓨터 지원 의료의 실제 응용을 촉진하는 데 기여할 것입니다. 이를 통해 치료 결정 및 임상 치료의 개선에도 기여할 것으로 기대됩니다.



### CL3DOR: Contrastive Learning for 3D Large Multimodal Models via Odds Ratio on High-Resolution Point Clouds (https://arxiv.org/abs/2501.03879)
- **What's New**: 최근 연구는 대형 언어 모델(LLMs)이 텍스트 작업에 한정되지 않고 오디오, 이미지, 비디오를 포함한 다양한 다중 모드(multi-modal) 모델로서 기능할 수 있음을 보여주고 있습니다. 특히 3D 대형 다중 모드 모델(3D LMMs)에 대한 연구가 주목받고 있으며, 이는 포인트 클라우드(point clouds)와 같은 고차원 데이터를 처리할 수 있는 잠재력에 의해 주도되고 있습니다. 그러나 기존의 훈련 데이터셋에서 시각적 및 텍스트 내용의 정보 밀도와 명확성이 부족하여 정확한 교차 모드 이해의 병목 현상이 발생하고 있다는 문제점이 있습니다.

- **Technical Details**: CL3DOR는 고해상도 포인트 클라우드에서 오즈 비율(odds ratio)을 활용한 3D 대형 다중 모드 모델을 위한 대조적 학습(Contrastive Learning) 방법론을 제안합니다. 이 방법은 각 객체에 대한 포인트 클라우드의 밀도를 높이며, 훈련 데이터셋에서 불필요한 응답을 처벌하는 정보가 많은 하드 네거티브 응답을 구축합니다. CL3DOR는 언어 모델링 손실의 보조항으로 오즈 비율을 포함하여 대조 학습을 용이하게 하며, 구조화된 3단계 훈련 파라다임을 통해 교차 모드 이해를 크게 향상시킵니다.

- **Performance Highlights**: CL3DOR는 3D 장면 이해 및 추론 벤치마크에서 최첨단 성능을 달성하며, 특히 정확도 및 F1 점수에서 기존 모델들의 성능을 크게 초월합니다. 또한 고해상도, 하드 네거티브 데이터셋 구축 및 목표 함수 관련 포괄적 실험을 통해 CL3DOR의 각 구성 요소의 중요성과 효과를 실험적으로 입증합니다. CL3DOR의 수행 성과는 3D LMMs의 언어 지침 이해력을 혁신적으로 변화시킬 것으로 기대되며, 고품질 훈련 데이터셋의 구축 방법을 제시합니다.



### ZDySS -- Zero-Shot Dynamic Scene Stylization using Gaussian Splatting (https://arxiv.org/abs/2501.03875)
- **What's New**: 본 논문은 ZDySS라는 새로운 제로샷(Zero-shot) 동적 씬 스타일화 프레임워크를 소개합니다. 이 모델은 훈련된 후 스타일 이미지를 추가적으로 최적화할 필요 없이 이전에 보지 못한 스타일 이미지에 일반화할 수 있습니다. Gaussian splatting을 이용한 씬 표현 방식을 통해, 각 Gaussian을 학습된 특징 벡터와 연결하여 주어진 뷰 및 타임스탬프에 대한 특징 맵을 생성할 수 있습니다.

- **Technical Details**: 이 연구는 3D 환경에서 Adaptive Instance Normalization(AdaIN) 방식의 스타일 전이 기법을 적용하여, 훈련된 동적 Gaussian 씬에 대해 제로샷 스타일화를 가능하게 합니다. Gaussian 개체는 학습된 특징 벡터를 통해 2D VGG 특징을 3D 공간으로 올려 보내는 데 사용됩니다. 본 연구에서는 공간과 시간의 일관성을 보장하기 위해 실행 평균(running average)을 활용합니다.

- **Performance Highlights**: ZDySS 방법은 실제 동적 씬에 대한 테스트에서 기존의 최첨단 기법보다 뛰어난 성능과 일관성을 보여줍니다. 이 방법은 실제 사용 사례에서 강력한 솔루션으로 작용할 수 있으며, 다양한 스타일에 대해 매우 매력적인 시각적 효과를 달성합니다. 최종적으로 이 연구는 동적 씬 스타일화의 여러 가지 도전에 대한 효과적인 해결책을 제시합니다.



### Diffusion as Shader: 3D-aware Video Diffusion for Versatile Video Generation Contro (https://arxiv.org/abs/2501.03847)
Comments:
          Project page: this https URL Codes: this https URL

- **What's New**: 본 논문에서는 'Diffusion as Shader (DaS)'라는 새로운 비디오 생성 방식을 소개합니다. 기존의 방식들은 하나의 제어 유형에 제한되어 있었던 반면, DaS는 3D 제어 신호를 활용해 다양한 비디오 제어 작업을 수행할 수 있는 통합된 아키텍처를 제공합니다. 이 접근법은 특히 카메라 조작 및 콘텐츠 편집 시 정밀한 제어를 가능하게 하여 비디오 생성 과정의 유연성을 향상시킵니다.

- **Technical Details**: DaS는 비디오의 3D 모션을 정의하는 3D 추적 비디오를 제어 신호로 사용하여 비디오 생성에 3D 인식을 통합합니다. 이를 통해 비디오의 시간적 일관성을 크게 개선하며, 3D 포인트가 색상을 통해 서로 연결됨으로써 비디오 프레임 사이의 일관된 표현을 보장합니다. 이 방식은 복잡한 비디오 제어 작업을 가능하게 하며, 기존의 2D 기반 접근법의 한계를 극복합니다.

- **Performance Highlights**: DaS는 3일간 8개의 H800 GPU에서 10,000개 미만의 비디오를 사용하여 훈련하여 다양한 작업에서 강력한 제어 능력을 입증했습니다. 특히, 카메라 제어 및 모션 전이 작업에서는 기존 방식들보다 월등히 향상된 성능을 보였으며, 추가적인 작업인 메시-비디오 변환 및 객체 조작에서도 뛰어난 생성 품질을 실험적으로 확인할 수 있었습니다.



### LM-Net: A Light-weight and Multi-scale Network for Medical Image Segmentation (https://arxiv.org/abs/2501.03838)
- **What's New**: 본 연구에서는 의료 영상 분할(Medical Image Segmentation)에서 마주하는 문제를 해결하기 위해 경량화된 다중 스케일 아키텍처(Architecture)인 LM-Net을 제안합니다. LM-Net은 CNN(Convolutional Neural Networks)과 ViT(Vision Transformers)의 장점을 통합하여 분할 정확도를 높이는데 기여합니다. 특히, 지역 세부 텍스처(Local Detail Textures)와 전역 문맥 의미(Global Contextual Semantics)를 동시에 캡처하는 모듈을 도입하여 블러된 경계 문제(Blurred Segmentation Boundaries)를 완화합니다.

- **Technical Details**: LM-Net은 세 가지 주요 모듈로 구성됩니다: 다중 분기 모듈(Multi-branch Module), 지역 특징 변환기(Local Feature Transformer, LFT), 그리고 전역 특징 변환기(Global Feature Transformer, GFT)입니다. 다중 분기 모듈은 단일 레벨에서 다중 스케일 특징을 효과적으로 캡처하고, LFT는 지역 창(self-attention) 기반으로 세부 텍스처를 추출합니다. 반면 GFT는 전역 self-attention을 활용하여 전역 문맥 의미를 수집합니다.

- **Performance Highlights**: LM-Net은 Kvasir-SEG 데이터셋에서 94.09% mDice, 89.12% mIoU를 달성하며 기존 최고 성능의 방법들을 초월하는 성과를 보여줍니다. LGG Segmentation 데이터셋과 유방 초음파 이미지 데이터셋에서도 우수한 결과를 기록하여 다양한 의료 영상 분할 작업에서의 가능성을 입증하였습니다. 이러한 성능은 4.66G FLOPs와 5.4M 파라미터만으로 달성되어 경량 모델로서의 장점을 강조합니다.



### MeshConv3D: Efficient convolution and pooling operators for triangular 3D meshes (https://arxiv.org/abs/2501.03830)
- **What's New**: 이번 논문에서는 3D 데이터에 대한 새로운 접근 방식인 MeshConv3D를 소개합니다. CNN(Convolutional Neural Networks)의 기존 개념을 3D 메쉬 데이터로 확장하는 데 있어 발생하는 여러 도전을 해결하기 위해 특수화된 convolution 및 face collapse-based pooling 연산자를 통합하였습니다. MeshConv3D는 사전 리메쉬(reshaping) 또는 변환 기술 없이 임의의 토폴로지(mesh topology)를 가진 메쉬에서 직접 작동합니다.

- **Technical Details**: MeshConv3D는 비정형 연결성 문제가 있는 3D 메쉬 데이터를 처리하기 위한 새로운 convolution과 pooling 연산자를 도입합니다. 이 방법은 기존의 CNN 아키텍처와는 다르게 메쉬의 구조적 특성을 고려하여 사용자 정의된 연산자를 활용합니다. 특히, 이 접근 방식은 메쉬 데이터의 복잡성을 줄이면서 동시에 연산의 효율성을 높이는 것을 목표로 합니다.

- **Performance Highlights**: 세 가지 서로 다른 벤치마크 데이터 세트에서의 실험 결과, MeshConv3D는 동등하거나 더 나은 분류 성능을 달성할 수 있음을 보였습니다. 또한, 이 접근 방식은 관련된 메모리 소비와 계산 부담을 최소화하는 동시에 높은 정확도를 유지하는 장점을 가지고 있습니다.



### MADation: Face Morphing Attack Detection with Foundation Models (https://arxiv.org/abs/2501.03800)
Comments:
          Accepted at WACV 2025 workshops

- **What's New**: 이 논문은 Morphing Attack Detection (MAD) 시스템에 대해 Foundation Models (FM)인 CLIP 아키텍처를 활용하여 새로운 프레임워크인 MADation을 제안합니다. MADation은 기존의 MAD 솔루션과 비교하여 더 높은 성능을 보여주며, FM의 가능성을 탐구합니다. 또한, 실험을 통해 MADation이 MAD 작업에 효과적으로 적응하고 있음을 입증했습니다.

- **Technical Details**: MADation은 LoRA(저랭크 적응) 가중치를 사용하여 FM CLIP 아키텍처를 조정하며, 동시에 분류 헤더를 훈련합니다. 이를 통해 FM은 MAD의 특성에 맞춰 특징 공간(feature space)을 잘 정렬할 수 있으며, 사전 훈련(pre-training) 동안 습득한 내재 지식을 활용할 수 있습니다. 이 접근 방식은 MAD라는 특정 작업에 적합하게 FM을 조정하여 성능을 개선하도록 설계되었습니다.

- **Performance Highlights**: MADation은 기존의 MAD 솔루션들과 비교하여 여러 평가 시나리오에서 경쟁력 있는 결과를 보였으며, ViT-FS 및 FE와 비교해 평균 EER(적발 오류율)을 각각 16.93 포인트 및 8.10 포인트 감소시켰습니다. 이러한 성과는 FM이 MAD와 같은 도메인 특정 작업에서 중요한 역할을 할 수 있는 가능성을 강조합니다.



### KAnoCLIP: Zero-Shot Anomaly Detection through Knowledge-Driven Prompt Learning and Enhanced Cross-Modal Integration (https://arxiv.org/abs/2501.03786)
Comments:
          Accepted by ICASSP 2025

- **What's New**: KAnoCLIP은 개인정보 보호 및 데이터 부족 문제를 해결하기 위해 훈련 샘플 없이 이상 탐지(zero-shot anomaly detection, ZSAD)를 가능하게 하는 새로운 프레임워크입니다. 기존의 CLIP 모델의 한계를 극복하기 위해 Knowledge-Driven Prompt Learning(정향 학습) 접근 방법을 도입하여 학습 가능한 이상 프롬프트를 생성하고 수동 텍스트 프롬프트의 필요성을 제거했습니다. 이러한 혁신은 이미지 수준의 이상 탐지 성능을 극대화하고, 현장과 의료 분야 모두에서 훌륭한 일반화 능력을 보입니다.

- **Technical Details**: KAnoCLIP은 CLIP-VV 비주얼 인코더, Bi-Directional Cross-Attention for Multi-Level Cross-Modal Interaction(Bi-CMCI), Conv-Adapter를 통합하여 지역적 비주얼 세멘틱(local visual semantics)을 보존하고, 상호 모달 융합을 개선합니다. 프레임워크는 훈련 동안 보조 이상 탐지 데이터셋을 사용하여 손실 함수를 최소화하고, 이후 테스트 이미지를 클립 비주얼 인코더를 통해 처리하여 패치 특성을 추출하는 방식을 사용합니다. 이러한 과정에서 LNP와 LAP를 통해 텍스트 특성을 생성하고, 이것을 Bi-CMCI 모듈을 사용해 결합하여 최종 이상 맵을 생성합니다.

- **Performance Highlights**: KAnoCLIP은 12개의 산업 및 의료 데이터셋에서 뛰어난 성능을 기록하며 기존의 최첨단 기법과 비교하여 우수한 일반화 능력을 입증했습니다. 이러한 성과는 특히 섬세한 이상 점검에서 pixel-level anomaly segmentation을 향상시키며, 새로운 이상 클래스에 대한 일반화 능력을 높여줍니다. KAnoCLIP은 이상 탐지 분야에서 새로운 기준을 설정하며, 다양한 애플리케이션에서 활용될 가능성을 보여줍니다.



### Strip R-CNN: Large Strip Convolution for Remote Sensing Object Detection (https://arxiv.org/abs/2501.03775)
- **What's New**: 최근 원격 감지 객체 탐지(Remote Sensing Object Detection) 분야에서 많은 발전이 있었으나, 높은 비율의 객체(High Aspect Ratio Objects)를 탐지하는 것은 여전히 도전적인 과제입니다. 본 논문에서는 'Strip R-CNN'이라는 새로운 네트워크 아키텍처를 제안하며, 이를 통해 다양한 비율의 객체를 효과적으로 탐지할 수 있는 방법을 제시합니다. 이 모델은 대형 스트립 컨볼루션(Large Strip Convolutions)을 활용하여 객체의 공간적 정보를 효과적으로 캡처할 수 있도록 설계되었습니다.

- **Technical Details**: Strip R-CNN은 기존의 정사각형 커널 대신에 직렬의 직교 대형 스트립 컨볼루션을 사용하여 특징을 학습합니다. 이 네트워크는 탐지 헤드를 분리하고 스트립 컨볼루션을 사용하여 객체의 위치를 더욱 정확하게 예측할 수 있도록 개선하였습니다. 이러한 설계는 모델의 효율성을 높이고, 매우 높은 비율의 객체에 대해서도 잘 일반화하도록 돕습니다.

- **Performance Highlights**: 저자들은 DOTA, FAIR1M, HRSC2016, DIOR와 같은 여러 기준 데이터셋에서 실험을 수행하여 Strip R-CNN이 기존 방법들보다 월등한 성능을 보임을 입증했습니다. 특히, 30M 파라미터를 가진 Strip R-CNN 모델은 DOTA-v1.0에서 82.75%의 mAP를 달성하며 새로운 최첨단 성과를 기록했습니다. 이는 원격 감지 분야의 새로운 연구 통찰력을 제공할 것으로 기대됩니다.



### AutoFish: Dataset and Benchmark for Fine-grained Analysis of Fish (https://arxiv.org/abs/2501.03767)
Comments:
          In the 3rd Workshop on Maritime Computer Vision (MaCVi) at WACV'25

- **What's New**: 이 논문에서는 AutoFish라는 새로운 공개 데이터셋을 소개합니다. 이 데이터셋은 454종의 어류 표본에 대한 Instance segmentation(masks)과 길이 측정이 포함된 1,500장의 이미지를 포함합니다. 데이터셋은 폐쇄된 환경에서 RGB 카메라를 사용하여 수집되었으며, 수동 주석이 진행되었습니다.

- **Technical Details**: AutoFish 데이터셋은 어획물 문서화를 위한 자동화된 프로세스를 위한 연구로, 다양한 위치에 놓인 어류를 잘 구분할 수 있는 세밀한 분석을 지원합니다. 두 가지 Mask2Former 아키텍처 변형을 사용하여 기준 Instance segmentation 결과를 설정했으며, 최상위 모델은 mAP(Mean Average Precision) 89.15%를 달성하였습니다. 또한 자체적으로 설계한 MobileNetV2 기반 회귀 모델을 통해 길이 추정 방법도 제시된 바 있습니다.

- **Performance Highlights**: 논문에서 개발된 AutoFish 데이터셋은 어류 자동 모니터링에 대한 상위 연구를 지원할 수 있는 중요한 시사점을 제공합니다. 가장 잘 수행된 길이 추정 모델은 장애물이 없는 이미지에서 MAE(Mean Absolute Error) 0.62cm, 장애물이 있는 이미지에서는 1.38cm를 달성했습니다. 이 데이터셋은 어업 산업의 지속 가능성을 높이는 데 기여할 수 있는 가능성을 보여줍니다.



### Image Segmentation: Inducing graph-based learning (https://arxiv.org/abs/2501.03765)
- **What's New**: 이번 연구는 다양한 이미지 모달리티에서 의미론적 분할(semantic segmentation)을 강화하기 위해 그래프 신경망(GNNs)을 활용하는 가능성을 탐구합니다. 새로운 GNN 기반의 U-Net 아키텍처를 제안하고, 이를 기존의 CNN 기반 분할 모델들과 비교하여 우수성을 입증하였습니다. 이 접근 방식은 이미지의 특징을 그래프 형태로 모델링하여 지역 간의 관계를 명확히 표현할 수 있게 해줍니다.

- **Technical Details**: 제안된 모델은 CNN으로 지역 특징을 추출한 후, GNN을 사용하여 이러한 특징 간의 복잡한 관계를 모델링합니다. 이 혼합 접근 방식은 CNN의 지역 패턴 추출 및 GNN의 전역 관계 모델링을 결합하여 전통적인 CNN 기반 방법의 한계를 극복합니다. 세 개의 데이터셋인 PascalVOC, WoodScape, ISIC2016에서 성능을 평가하였습니다.

- **Performance Highlights**: 실험 결과, GNN을 포함한 분할 파이프라인이 왜곡된 이미지에서 분할 정확도를 높였다는 것을 확인하였습니다. 특히, WoodScape 데이터셋에서 전통적인 CNN 기반 방법들보다 현저한 성능 향상을 보였습니다. 이는 제안된 CNN-GNN 프레임워크가 자율주행 시스템의 신뢰성과 안전성을 향상시킬 수 있는 가능성을 제시합니다.



### Realistic Test-Time Adaptation of Vision-Language Models (https://arxiv.org/abs/2501.03729)
- **What's New**: 이 논문은 Vision-Language Models (VLMs)의 시험 시 적응(test-time adaptation, TTA) 방법이 실제 배포 환경에서의 성능을 저해한다는 점을 강조합니다. 기존 연구는 일정한 클래스 분포 가정에 기반한 경우가 많았으나, 본 연구에서는 효과적인 클래스의 수가 가변적인 상황과 비독립 동등하게 분포하지 않은 배치(non-i.i.d. batches) 환경을 포함한 평가 기준을 제시합니다. 이러한 접근은 VLM의 초기 제로샷(zero-shot) 강건성을 보존하면서도 다양한 실제 시나리오에 적용 가능성을 높이는 것을 목표로 합니다.

- **Technical Details**: StatA라는 새로운 방법론을 소개하며, 이는 VLM을 위한 새로운 정규화 항을 통합하여 초기 텍스트 인코더 지식을 보존할 수 있도록 설계되었습니다. 특히 데이터가 적은 환경에서 효과적으로 작동하며, 배치 내에서 변화하는 수의 효과적인 클래스를 처리하는 데 강력한 기능을 발휘합니다. 이 방법은 높은 효율성을 자랑하며 짧은 시간 내에 수천 개의 샘플을 처리할 수 있습니다.

- **Performance Highlights**: 실험 결과, StatA는 기존의 TTA 방법들이 가지는 현실적인 상황에서의 한계를 극복하고 더 많은 클래스 수를 다룰 수 있는 능력을 보여줍니다. 연구에서 제시한 두 가지 평가 기준(즉, 다양한 효과적인 클래스 수와 비독립적인 배치)을 통해 StatA는 클래스 분포가 복잡한 실제 시나리오에서 높은 성능을 달성할 수 있음을 입증하였습니다. 이러한 성과는 실세계 데이터 분포의 예측 개선에 기여하며, 향후 VLM의 더 나은 활용 가능성을 제시합니다.



### Self-adaptive vision-language model for 3D segmentation of pulmonary artery and vein (https://arxiv.org/abs/2501.03722)
Comments:
          8 pages,3 figures

- **What's New**: 이 논문은 폐 혈관 구조의 정확한 분할을 위한 새로운 접근 방식을 제안합니다. 최근 CLIP과 같은 사전 훈련된 비전-언어 모델(Vision-Language Model, VLM)의 발전을 이용하여, 적은 수의 주석된 데이터로 3D CT 스캔을 효과적으로 분할할 수 있는 방법을 모색했습니다. 제안된 방법은 Language-guided self-adaptive Cross-Attention Fusion Framework으로, 텍스트와 이미지 표현의 크로스 모달리티를 적응적으로 결합합니다.

- **Technical Details**: 이 방법은 CLIP을 강력한 특징 추출기로 활용하여 3D CT 스캔의 분할을 생성하며, 'self-adaptive learning' 전략으로 두 개의 임베딩 모달리티를 효과적으로 융합하는 특별히 설계된 어댑터 모듈을 제안합니다. 연구팀은 최대 718개의 주석된 CT 데이터로 구성된 가장 큰 폐 동맥-정맥 데이터 세트를 사용하여 실험을 진행했고, 이는 다양한 최신 방법들에 비해 높은 성능을 입증했습니다.

- **Performance Highlights**: 제안된 방법은 평균 DSC( Dice Similarity Coefficient) 점수 76.22%를 기록하며, 기존 최고의 방법인 nnU-Net에 비해 평균 9.03% 우수한 결과를 나타냈습니다. 이 연구는 사전 훈련된 비전-언어 모델을 활용한 의료 영상 분할 접근 방식의 최신 경향을 보여주며, 대량의 주석된 CT 스캔 데이터 기반의 성능 개선을 확립했습니다.



### Materialist: Physically Based Editing Using Single-Image Inverse Rendering (https://arxiv.org/abs/2501.03717)
Comments:
          code will be available at this http URL

- **What's New**: 이번 연구에서는 단일 이미지에서 시작하는 이미지 편집을 위한 새로운 방법을 제안합니다. 이 방법은 학습 기반 접근법과 점진적 미분 렌더링(progressive differentiable rendering)을 결합하여, 환경 맵 최적화와 재질 속성 개선을 통해 입력 이미지와 렌더링 결과를 밀접하게 일치시키는 것을 목표로 합니다. 기존의 다른 역 렌더링 방법들은 여러 개의 뷰를 필요로 하지만, 본 연구는 단일 이미지만으로도 가능하게 설계되었습니다.

- **Technical Details**: 이 접근법은 MaterialNet이라는 신경망을 사용하여 이미지에서 알베도(albedo), 거칠기(roughness) 및 금속성(metallic) 속성을 추출합니다. 그런 후 미츠바(Mitsuba) 렌더러를 활용하여 물리 기반 재질 편집을 수행하며, 그림자 및 글로벌 조명(global illumination)과 같은 정확한 물체-환경 간 상호작용을 달성합니다. 또한, 물체 투입(object insertion) 및 재조명(relighting)과 같은 작업을 지원하는 최적화된 재질 속성과 조명을 제공합니다.

- **Performance Highlights**: 제안된 방법은 Stable Diffusion 기반의 최신 모델보다 빠른 추론 속도를 보이며, 단일 뷰 미분 몬테 카를로(ray tracing)에서 환경 맵에 대한 우수한 결과를 도출합니다. 연구 결과는 제안된 방법이 기존 방법들보다 더 현실적인 빛의 굴절(light refraction)을 제공하며, 특별히 투명 물체의 재질 편집을 위한 혁신적인 기능을 강조합니다. 이는 기존의 방법들과 비교했을 때 강력한 해석 가능성을 추가합니다.



### MoDec-GS: Global-to-Local Motion Decomposition and Temporal Interval Adjustment for Compact Dynamic 3D Gaussian Splatting (https://arxiv.org/abs/2501.03714)
Comments:
          The last two authors are co-corresponding authors. Please visit our project page at this https URL

- **What's New**: MoDecGS는 복잡한 모션을 가진 동적 장면을 위해 메모리 효율적인 Gaussian splatting 프레임워크를 제안합니다. 이를 위해 Global-to-Local Motion Decomposition (GLMD) 기법을 도입하여 동적 모션을 효과적으로 캡처합니다. Global Canonical Scaffolds와 Local Canonical Scaffolds를 활용하여 정적 Scaffold 표현을 동적 비디오 재구성으로 확장합니다.

- **Technical Details**: 이 프레임워크는 Global Anchor Deformation (GAD)와 Local Gaussian Deformation (LGD)을 구현하여 모션을 세분화합니다. Temporal Interval Adjustment (TIA)는 각 Local CS의 시간 범위를 자동으로 조정하여 최적의 시각적 품질을 유지하도록 지원합니다. MoDecGS는 기존 기술 대비 평균 70%의 모델 크기 감소를 달성하며, 실시간 품질을 유지합니다.

- **Performance Highlights**: MoDecGS는 세 가지 모노큘라 데이터셋에서 실험을 통해 저장 공간을 대폭 줄이며, 시각적 품질을 유지 또는 개선하는 것을 입증했습니다. 특히, iPhone 데이터셋에서는 PSNR이 +0.7dB로 증가하고 저장 용량은 94% 감소했습니다. 이는 MoDecGS가 동적 3D Gaussian에 대한 메모리 효율성을 크게 향상시켰음을 보여줍니다.



### AuxDepthNet: Real-Time Monocular 3D Object Detection with Depth-Sensitive Features (https://arxiv.org/abs/2501.03700)
- **What's New**: AuxDepthNet은 외부 깊이 정보 없이 실시간 단안 3D 물체 탐지를 위한 새로운 프레임워크입니다. 이 모델은 Auxiliary Depth Feature (ADF) 모듈과 Depth Position Mapping (DPM) 모듈을 도입하여 깊이 감지 특성을 효과적으로 학습하고, 탐지 과정에 깊이 위치 정보를 통합합니다. 이러한 접근 방식은 알고리즘의 복잡성을 줄이면서도 실시간 성능을 보장합니다.

- **Technical Details**: AuxDepthNet 구조는 다면적 특성 표현을 활용하여 깊이 민감 특성(depth-sensitive features), 맥락 민감 특성(context-sensitive features) 및 깊이 유도 특성(depth-guided features)을 통합합니다. ADF 모듈은 보조 학습(auxiliary learning)을 통해 깊이 관련 힌트를 암시적으로 학습하여 외부 깊이 맵에 의존하지 않습니다. DPM 모듈은 위치 인코딩을 통해 탐지 프로세스에 깊이 관련 위치 정보를 내장하여 공간적 추론과 강력한 3D 위치 지정이 가능하도록 합니다.

- **Performance Highlights**: KITTI 데이터셋에서 AuxDepthNet은 $	ext{AP}_{3D}$ 측정 값에서 24.72% (쉬움), 18.63% (보통), 15.31% (어려움)을 기록하며 최첨단 성능을 달성했습니다. 추가적으로, $	ext{AP}_{	ext{BEV}}$ 스코어는 34.11% (쉬움), 25.18% (보통), 21.90% (어려움)으로 나타났습니다. 이러한 결과는 AuxDepthNet이 실시간 3D 객체 탐지에서 경쟁력 있는 솔루션임을 시사합니다.



### Motion-Aware Generative Frame Interpolation (https://arxiv.org/abs/2501.03699)
- **What's New**: 이번 연구에서는 Motion-aware Generative Frame Interpolation (MoG)라는 새로운 프레임워크를 제안합니다. MoG는 입력 프레임들 사이의 motion awareness를 향상시키기 위해 명시적인 motion guidance를 통합하는 방법입니다. 이는 기존의 generative model이 입력 프레임 간의 동적 관계를 충분히 활용하지 못한다는 문제를 해결하기 위한 시도입니다.

- **Technical Details**: MoG는 flow-based interpolation 모델에서 얻은 intermediate flow를 motion guidance로 사용하는데 중점을 둡니다. 이 intermediate flow는 rigid motion을 가정하여 중간 프레임의 근사치를 제공하며, generative model에 통합되어 동적 정보를 정확하게 반영합니다. 이 과정에서 latent 및 feature 레벨 양쪽에서 motion cues를 통합하여 보다 정교한 비디오 생성을 가능하게 합니다.

- **Performance Highlights**: MoG는 실제와 애니메이션 데이터셋 모두에서 탁월한 성능을 보이며, 기존의 방법들과 비교하여 비디오 품질 및 fidelity에서 유의미한 개선을 나타냅니다. 실험 결과, MoG로 생성된 비디오는 향상된 motion stability와 콘텐츠 일관성을 보여주며, 여러 지표에서 기존 모델들보다 뛰어난 성능 지표를 기록했습니다.



### SMIR: Efficient Synthetic Data Pipeline To Improve Multi-Image Reasoning (https://arxiv.org/abs/2501.03675)
- **What's New**: 이번 연구에서는 복수 이미지 추론을 효과적으로 수행할 수 있는 SMIR(동기식 다중 이미지 추론 데이터 생성 프로세스)를 소개합니다. 이 파이프라인은 멀티모달 임베딩을 활용하여 고도로 연관된 이미지를 효율적으로 추출하고, 이를 통해 16만 개의 합성 데이터 샘플을 생성합니다. 또한 SMIR-BENCH라는 새로운 평가 벤치마크를 제시하여, 다양한 복합 다중 이미지 추론 작업에 대해 모델의 표현력과 추론 능력을 종합적으로 평가할 수 있도록 합니다.

- **Technical Details**: SMIR에서는 고도의 상관관계를 가진 이미지를 생성하기 위해 클러스터 샘플링(cluster sampling)과 그래프 반복 샘플링(graph iteration sampling) 기법을 적용합니다. 이를 통해 다중 이미지 작업에 대한 더 도전적인 훈련 환경을 제공하여, 데이터 품질을 높이고 다양성을 확보합니다. SMIR-BENCH는 다중 턴(multi-turn) 상호작용을 통해 모델의 최종 답변과 추론 과정 모두를 평가하는 새로운 벤치마크로 설계되었습니다.

- **Performance Highlights**: SMIR를 사용하여 훈련된 모델들은 SMIR-BENCH에서 기존의 기준 모델보다 8% 이상의 성능 향상을 보였습니다. 이는 SMIR 데이터셋의 효율적인 생성과 품질 높은 벤치마크 평가 방법 덕분입니다. 또한, 개방형 LLM을 이용한 이 방법은 비용을 최대 50배 줄이고 속도를 최대 10배 증대시켜, 데이터 생성 과정에서의 인적 노력을 크게 줄였습니다.



### Action Quality Assessment via Hierarchical Pose-guided Multi-stage Contrastive Regression (https://arxiv.org/abs/2501.03674)
- **What's New**: 이 논문은 새로운 Action Quality Assessment (AQA) 방법을 제안합니다. 기존의 방법이 고정된 프레임으로 비디오를 분할하는 데 집중한 반면, 제안된 방법은 계층적 포즈 안내를 통해 다단계 대조 회귀를 사용하여 더 정교한 기술을 사용합니다. 또한 정교한 인간 포즈 레이블을 포함한 FineDiving-Pose Dataset이 새롭게 만들어졌습니다.

- **Technical Details**: 제안된 방법은 다중 스케일 동적 비주얼-스켈레톤 인코더를 통해 세밀한 시공간 비주얼 및 골격 특징을 포착합니다. 이후 절차 분할 네트워크를 통해 서로 다른 하위 동작을 분리하고 세분화된 특징을 얻습니다. 이러한 특징들은 다중 모달 융합 모듈에 입력되어 모델이 세분화된 활동 유사성과 변화를 학습할 수 있도록 도와줍니다.

- **Performance Highlights**: FineDiving 및 MTL-AQA 데이터 세트에서 실험 결과는 제안된 방법의 효과성과 우수성을 입증합니다. 전반적으로, 실험을 통해 제안된 방법이 최신 방법들을 능가함을 보여주었습니다. 이는 AQA의 품질 평가 및 개선에 기여할 수 있는 새로운 접근 방식을 제공함을 의미합니다.



### Local Compositional Complexity: How to Detect a Human-readable Messsag (https://arxiv.org/abs/2501.03664)
- **What's New**: 이 논문은 데이터 복잡성(data complexity)에 대한 명확하고 계산 가능한 정의의 필요성을 강조하고, 메시지를 전달할 수 있는 방식으로 구조화된 데이터의 복잡성을 측정하기 위한 새로운 접근법을 제시합니다. 저자들은 데이터의 최단 설명을 두 가지 부분으로 나누고, 이 중 구조화된 부분의 크기를 복잡성 점수로 사용하는 일반적인 프레임워크를 설명합니다. 이는 통계역학(statistical mechanics)에서 물리적 시스템의 거시상태(macrostate)와 엔트로피(entropy)를 보다 객관적으로 특성화하는데 도움을 줄 수 있습니다. 또한, 이들이 개발한 로컬 조합 구조(local compositionality)가 인간의 의사소통(comunicative signals)에 적합한 복잡성 구조임을 제안합니다.

- **Technical Details**: 복잡성 측정을 위한 새로운 접근법은 데이터 설명을 구조화된 부분과 비구조화된 부분으로 나누고, 효과적인 복잡성 측정 메트릭을 개발하는 것입니다. 복잡성 점수는 최적 설명에서 구조화된 부분의 길이를 기준으로 산출되며, 이는 자연어 처리나 이미지 데이터와 같은 다양한 도메인에서 적용될 수 있습니다. 본 논문은 두 부분으로 나눈 설명을 통해 정보 이론(information theory)에 기반한 복잡성 정량화의 한계를 극복하고 있으며, 저자들은 이러한 방법을 타당성과 신뢰성 있게 입증하기 위한 실험을 수행했습니다. 또한, LCC(score)가 비인간 통신을 탐지할 수 있는 잠재력을 보여주며, 의미 있는 신호를 구별하는 데 효과적임을 입증합니다.

- **Performance Highlights**: LCC 점수는 텍스트, 이미지, 오디오 데이터에 대해 수행된 실험을 통해 의미 있는 복잡성과 관련하여 우리의 직관과 일치하는 결과를 보였습니다. 특히, 인간 언어나 실제 이미지와 같은 의미 있는 신호를 효과적으로 구별할 수 있는 능력이 확인되었습니다. 더 나아가, Arecibo 메시지를 의미 있는 신호로 간주할 수 있으며, 그 정확한 비율(aspect ratio)을 확인할 수 있는 잠재적인 가능성도 보였습니다. 이러한 결과는 LCC 점수가 데이터 복잡성을 측정하고 비인간 신호를 탐지하는 데 강력한 도구가 될 수 있음을 시사합니다.



### DehazeGS: Seeing Through Fog with 3D Gaussian Splatting (https://arxiv.org/abs/2501.03659)
Comments:
          9 pages,4 figures

- **What's New**: 이번 논문에서는 DehazeGS라는 새로운 방법을 제안합니다. DehazeGS는 3D Gaussian Splatting 기술을 활용하여 다양한 시점의 안개 이미지로부터 안개 없는 배경을 분리하고 렌더링할 수 있는 능력을 가지고 있습니다. 이 방법은 기존의 NeRF 기반 방법들이 겪는 높은 계산비용 문제를 해결하면서도 안개 장면에서의 세밀한 복원을 얻을 수 있게 해줍니다.

- **Technical Details**: DehazeGS는 Gaussian 분포 내에서 전파를 모델링하여 안개 형성을 시뮬레이션합니다. 중간 매질을 고려하여 대기광과 산란계수를 학습하며, 반복적인 최적화를 통해 명확한 시각 자료를 생성합니다. 또한, 입력 이미지 최적화에서 의사 깊이 맵을 사용하여 정확한 깊이 추정을 지원합니다.

- **Performance Highlights**: DehazeGS는 합성 및 실제 안개 데이터셋에 대한 실험 결과, 렌더링 품질과 계산 효율성을 모두 고려했을 때 최신 성능을 달성하였습니다. 특별히, 이 모델은 대부분의 장면에서 약 3000회 반복만으로 최적 결과를 도출하며, 훈련 시간은 약 1분에 불과합니다.



### Advancing the Understanding of Fine-Grained 3D Forest Structures using Digital Cousins and Simulation-to-Reality: Methods and Datasets (https://arxiv.org/abs/2501.03637)
- **What's New**: 이번 연구에서는, 정밀한 숲 자원 모니터링 및 생태계 연구를 위해 숲의 공간 의미와 구조를 이해하고 분석하는 데 필수적인 새로운 기법을 제안합니다. 이를 위해, Digital Cousins 및 Simulation-to-Reality (Sim2Real) 개념을 기반으로 한 자동화된 합성 데이터 생성 및 처리 프레임워크를 개발했습니다. Boreal3D라는 세계 최대의 숲 포인트 클라우드 데이터셋이 생성되었고, 이는 1000개의 매우 현실적이고 구조적으로 다양한 숲 플롯을 포함하고 있습니다.

- **Technical Details**: Boreal3D 데이터셋은 48,403 그루의 나무와 353억 개 이상의 포인트로 구성되어 있으며, 각 포인트는 의미, 인스턴스 및 관점 정보로 라벨링되어 있습니다. 나무는 지름, 수관 너비, 잎 면적 및 총 부피와 같은 구조적 매개변수로 설명됩니다. LiDAR 기술을 활용하여 3D 포인트 클라우드 데이터를 수집하며, 이러한 데이터는 다양한 플랫폼(위성 레이저 스캐닝, 공중 레이저 스캐닝 등)을 통해 획득됩니다.

- **Performance Highlights**: 실험 결과, 합성 데이터로 사전 훈련된 모델이 실제 숲 데이터셋에 적용될 때 성능이 크게 향상된다는 것을 확인했습니다. 특히, 실제 데이터의 20%로만 미세 조정해도 전체 실제 데이터로 훈련된 모델과 유사한 성능을 달성할 수 있다는 사실이 밝혀졌습니다. Boreal3D 데이터셋과 합성 데이터 증강 프레임워크는 대규모 3D 숲 씬 이해 및 구조적 매개변수 추정의 발전에 중요한 자원이 될 전망입니다.



### Exploring Optimal Latent Trajetory for Zero-shot Image Editing (https://arxiv.org/abs/2501.03631)
Comments:
          16 pages

- **What's New**: 이번 연구에서는 디퓨전 모델 기반의 이미지 편집에서 일반적인 '반전-편집' 파이프라인과는 다른 새로운 편집 패러다임인 ZZEdit을 제안합니다. ZZEdit은 이미지 구조를 보존하는 중간 단계의 은닉(latent) 표현을 편집 피벗으로 활용하며, 보다 높은 편집 가능성과 충실성을 동시에 달성하도록 설계되었습니다. 이를 통해 기존의 편집 방법보다 더 나은 성능을 보여줍니다.

- **Technical Details**: ZZEdit의 핵심은 ZigZag 프로세스를 통하여, 편집 피벗을 정하고 이 피벗에서 목표 안내(target guidance)를 부드럽게 강화하는 것입니다. 연구는 먼저 반전 궤적(inversion trajectory)에서 목표 프롬프트에 대해 더 높은 응답을 가지는 첫 번째 스텝을 찾아내어 editability와 fidelity를 유지합니다. 이를 통해 각 디노이징(denoising) 단계에서 목표 방향으로의 그래디언트(gradient)를 제공하고, 다음 디노이징 단계에 대한 작은 노이즈를 추가하여 성능을 개선합니다.

- **Performance Highlights**: ZZEdit의 효과를 입증하기 위해 다양한 이미지 편집 시나리오에서 실험을 수행하였고, 기존의 '반전-편집' 파이프라인과 비교하여 더욱 뛰어난 편집 가능성과 충실성을 달성하였습니다. 또한, ZZEdit은 P2P 및 PnP와 같은 기존의 편집 방법들과 함께 사용될 수 있으며, 최첨단의 이미지 편집 성능을 보여줍니다. 이를 통해 다양한 편집 방법에 대한 유연성과 범용성을 제공한다고 설명하고 있습니다.



### MC-VTON: Minimal Control Virtual Try-On Diffusion Transformer (https://arxiv.org/abs/2501.03630)
- **What's New**: MC-VTON은 기존의 virtual try-on 방식에서 불필요한 reference network나 image encoder를 제거하고, diffusion transformer (DiT)의 내재적 특징을 활용하여 최소한의 conditional image inputs만으로 작업을 수행할 수 있게 합니다. 이에 따라, 이미지 생성 과정에서 25단계를 초과하는 추론 단계를 획기적으로 8단계로 줄일 수 있습니다.

- **Technical Details**: 이 방법은 FLUX.1-dev의 39.7M의 추가 파라미터만을 사용하여, garment image와 masked person image의 두 개의 조건부 입력만으로 과정을 처리합니다. MC-VTON은 VAE encoder를 활용하여 이러한 이미지를 처리하고, learnable position embeddings를 통해 특징을 증강하여, garment과 person feature 간의 효율적인 상호작용을 가능하게 합니다.

- **Performance Highlights**: MC-VTON은 1024x768의 이미지를 생성하는 과정에서 5.23초의 경쟁력 있는 추론 속도를 보여줍니다. 또한, 총 136.5M의 최소한의 훈련 파라미터를 사용하며, 정량적 및 정성적 실험에서 기존의 선진 virtual try-on 모델에 비해 월등한 성능을 입증하였습니다.



### CFFormer: Cross CNN-Transformer Channel Attention and Spatial Feature Fusion for Improved Segmentation of Low Quality Medical Images (https://arxiv.org/abs/2501.03629)
Comments:
          The article consists of 15 pages, including 10 figures and 7 tables. The code will be made open-source once the article is accepted by the journal

- **What's New**: 본 논문에서는 CNN(Convolutional Neural Network)과 Transformer의 장점을 결합한 새로운 하이브리드 모델 CFFormer를 제안합니다. 이 모델은 두 가지 주요 모듈인 CFCA(Cross Feature Channel Attention)와 XFF(X-Spatial Feature Fusion)를 포함하여 의료 이미지 분할에서 성능을 극대화하는 데 중점을 두고 있습니다. 특히, 채널 특성(channel features)의 중요성을 강조하고 있으며, 이는 저품질 의료 이미지에 대한 세분화에 필수적인 요소입니다.

- **Technical Details**: CFFormer 모델은 이중 인코더 구조를 통해 지역적(local) 및 글로벌(global) 특성을 모두 캡처합니다. CFCA 모듈은 두 인코더 간 채널 특성의 상호작용을 촉진하고 필터링하는 역할을 하며, XFF 모듈은 공간적(spatial) 특성의 의미적 차이를 줄여 부드러운 특성 융합을 가능하게 합니다. 다양한 모달리티를 아우르는 8개의 데이터셋에서 모델의 일반화 능력을 평가하였습니다.

- **Performance Highlights**: 연구 결과, CFFormer 모델은 특히 경계가 흐릿하고 낮은 대비를 가진 데이터셋에서 현재의 최첨단(SOTA) 방법들을 초월하는 성능을 보여줍니다. 이러한 성능 개선은 CNN의 지역 정보와 Transformer의 글로벌 정보를 효과적으로 융합하여 이루어진 결과입니다. 논문에서는 코드도 제공하여 의료 이미지 분할 작업의 추가적인 탐색을 촉진할 수 있도록 하고 있습니다.



### Deep Learning-based Compression Detection for explainable Face Image Quality Assessmen (https://arxiv.org/abs/2501.03619)
Comments:
          2nd Workshop on Fairness in Biometric Systems (FAIRBIO) at International Conference on Pattern Recognition (ICPR) 2024

- **What's New**: 이 논문은 얼굴 이미지 품질 평가의 중요성을 강조하며, JPEG 및 JPEG 2000 압축 아티팩트를 탐지하기 위해 심층 신경망을 훈련시키는 새로운 접근 방식을 제안합니다. 압축된 이미지에서 아티팩트를 감지하고 인식 성능을 향상시키기 위해 만들어진 알고리즘은 OFIQ 소프트웨어의 일부로 제공됩니다. 성능 평가 결과에서, PSNR 레이블을 사용할 경우 2-3%의 오류율을 기록하며, 이는 얼굴 인식 시스템의 정확도를 높이는 데 기여할 수 있습니다.

- **Technical Details**: 논문에서는 아티팩트가 없는 얼굴 이미지를 JPEG 및 JPEG 2000 압축 알고리즘을 사용하여 압축하고, PSNR(Peak Signal-to-Noise Ratio) 및 SSIM(Structural Similarity Index) 메트릭스를 활용하여 훈련 레이블을 생성합니다. 여기서 심층 신경망은 이 레이블을 기반으로 훈련되며, 압축의 수준을 평가하기 위해 회귀(task) 작업을 수행합니다. 이를 통해 압축 아티팩트의 영향을 받고 있는 정도를 나타내는 값을 출력합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 정확도 면에서 우수한 성능을 보여주며, 열린 소스 및 상업적 얼굴 인식 시스템의 오류율을 현저히 줄일 수 있음을 입증하였습니다. 특히, 과도한 압축 아티팩트를 가진 얼굴 이미지를 제외함으로써 바이오메트릭 성능을 개선하는 데 큰 도움이 됩니다. 이와 관련하여 알고리즘의 사전 훈련된 모델은 공개적으로 제공되어 관련 연구자들이 활용할 수 있습니다.



### BTMTrack: Robust RGB-T Tracking via Dual-template Bridging and Temporal-Modal Candidate Elimination (https://arxiv.org/abs/2501.03616)
- **What's New**: 이번 논문에서는 BTMTrack이라는 새로운 RGB-T 추적 프레임워크를 소개합니다. 이 프레임워크는 정적 및 동적 템플릿을 적응적으로 활용하여 튼튼한 객체 추적을 제공합니다. 특히, 성공적인 추적을 위해 TMCE 전략을 개발하였으며, 이를 통해 배경 노이를 억제하고 효율성을 높였습니다.

- **Technical Details**: BTMTrack의 핵심은 이중 템플릿 백본 네트워크와 TMCE(Temporal-Modal Candidate Elimination) 전략에 있습니다. 이 백본 네트워크는 시간 정보를 효율적으로 통합하며, TMCE 전략은 템플릿 간의 상호작용을 개선하기 위해 시간 및 모달 상관관계를 평가하여 모델이 관련 토큰에 집중할 수 있게 도와줍니다. 또한, TDTB(Temporal Dual Template Bridging) 모듈이 제안되어 동적으로 필터링된 토큰을 통해 정밀한 교차 모달 융합을 촉진합니다.

- **Performance Highlights**: BTMTrack은 세 개의 기준 데이터세트에서 수행된 광범위한 실험을 통해 뛰어난 성능을 입증하였습니다. LasHeR 테스트 세트에서 72.3%의 정밀도를 달성하였으며, RGBT210 및 RGBT234 데이터세트에서도 경쟁력 있는 결과를 기록했습니다.



### ConcealGS: Concealing Invisible Copyright Information in 3D Gaussian Splatting (https://arxiv.org/abs/2501.03605)
- **What's New**: 최근 3D 재구성 기술의 발전으로 3D 데이터의 활용이 늘어나는 가운데, 3D Gaussian Splatting(3D-GS) 포맷에 대한 스테가노그래피 기법이 본격적으로 연구되고 있습니다. 기존의 NeRF 기반 모델의 한계를 극복하고, 정보의 은닉과 재구성 품질을 동시에 개선하는 ConcealGS라는 새로운 방법이 제안되었습니다. 이 방법은 지식 증류(knowledge distillation)와 그래디언트 최적화(gradient optimization) 전략을 도입하여, 3D 재구성의 질을 저하시키지 않으면서도 숨겨진 정보를 효과적으로 삽입할 수 있습니다.

- **Technical Details**: ConcealGS는 3D-GS의 학습 가능한 파라미터에 암묵적인 정보를 임베딩하기 위한 기본 원칙을 기반으로 합니다. 이 과정에서 렌더링 일관성 손실(rendering consistency loss)과 대조 손실(contrastive loss)을 활용하여 효과적으로 임베딩을 수행합니다. 또한 그래디언트 가이드 최적화(Gradient Guided Optimization)를 통해 디코더의 그래디언트 가중치를 동적으로 조절하여, 렌더링 품질과 정보 회수 사이의 균형을 맞추는 전략이 소개됩니다.

- **Performance Highlights**: 실험 결과, ConcealGS는 NeRF 기반 스테가노그래피 방법과 비교했을 때 3D 재구성 품질 및 암묵적 정보 회수의 효율성에서 현저한 성과를 보여주었습니다. 이 방법은 디지털 권리 관리(digital rights management)와 3D 콘텐츠의 보안 정보 전송 가능성을 크게 향상시키며, 복잡한 3D 표현 내에서의 정보 은닉에 대한 새로운 가능성을 열었습니다.



### BASIC: Semi-supervised Multi-organ Segmentation with Balanced Subclass Regularization and Semantic-conflict Penalty (https://arxiv.org/abs/2501.03580)
- **What's New**: 이 논문에서는 다중 기관 분할 (Multi-organ segmentation, MoS)에서의 세미 슈퍼바이즈드 러닝 (Semi-supervised learning, SSL)의 문제를 해결하기 위한 새로운 네트워크 구조를 제안합니다. 특히, 클래스 불균형 문제를 완화하기 위해 BAlanced Subclass regularIzation 및 semantic-Conflict penalty 메커니즘이 통합된 BASIC 네트워크를 개발했습니다. 이는 비편향 지식을 효과적으로 학습할 수 있도록 구성되었습니다.

- **Technical Details**: BASIC 네트워크는 새로운 보조 하위 클래스 분할 (Subclass segmentation, SCS) 작업을 통해 미리 생성된 균형 하위 클래스에 기반하여 구축되었습니다. 이 다중 작업 학습 (multi-task learning) 접근법은 MoS 작업을 위한 편향 없는 정보를 깊이 탐구합니다. 또한, 평균 교사 프레임워크(mean teacher framework)를 기반으로 하여, SCS 작업의 교사 예측을 활용하여 MoS 작업의 학생 예측을 감독하는 균형 하위 클래스 정규화를 설계했습니다.

- **Performance Highlights**: WORD 데이터셋 및 MICCAI FLARE 2022 데이터셋에서 수행된 광범위한 실험은 BASIC의 우수한 성능을 입증했습니다. 기존의 최첨단 방법들과 비교하여, BASIC 네트워크는 클래스 불균형 문제를 효과적으로 완화하고 MoS 작업에서 더 나은 결과를 보였습니다. 이는 세미 슈퍼바이즈드 MoS 분야에서 중요한 진전을 의미합니다.



### Cosmos World Foundation Model Platform for Physical AI (https://arxiv.org/abs/2501.03575)
- **What's New**: 이 논문은 Physical AI 구축을 위한 Cosmos World Foundation Model 플랫폼을 소개합니다. 이 플랫폼은 사용자 맞춤형 world model을 개발자가 만들 수 있도록 지원하며, 전반적인 world foundation model을 일반 목적의 모델로 정의하고 후속 응용 프로그램에 맞게 세부 조정할 수 있도록 합니다. 논문에서는 플랫폼의 다양한 구성 요소와 데이터 처리 파이프라인에 대해서도 설명합니다.

- **Technical Details**: Cosmos World Foundation Model 플랫폼은 비디오 기반의 WFM을 중심으로 하며, 비디오의 관찰을 모델 학습에 활용합니다. 기본적으로 pre-trained와 post-trained 모델로 나누어 보아, 전자는 다양한 비주얼 경험을 제공하고 후자는 특정 물리적 AI 환경에 맞춰 세부 조정됩니다. WFM의 개발은 transformer 기반의 diffusion 모델 및 autoregressive 모델을 포함하며, 이는 복잡한 비디오 생성 문제를 보다 쉽게 접근 가능하게 만듭니다.

- **Performance Highlights**: 논문에서 제시한 WFM 플랫폼은 다양한 Physical AI 작업을 위한 pre-trained 및 post-trained 모델을 제공하며, 사용자는 가상 환경에서 탐색할 수 있습니다. 또한, WFM의 활용을 통해 물리적 AI 개발자들이 정책 모델을 평가, 초기화 및 훈련하는 데 필요한 정보를 효과적으로 수집할 수 있습니다. 향후 연구를 통해 WFM의 정확성을 더욱 향상시키고 AI의 학습 능력을 고도화할 필요가 있습니다.



### Evaluating Image Caption via Cycle-consistent Text-to-Image Generation (https://arxiv.org/abs/2501.03567)
- **What's New**: 이번 논문에서는 기존의 평가 메트릭스를 개선하기 위해 CAMScore라는 새로운 사이클 일관성 없는 평가 메트릭스를 제안합니다. CAMScore는 텍스트-이미지 모델을 활용하여 캡션에서 이미지를 생성한 후, 이 생성된 이미지를 원본 이미지와 비교하여 평가합니다. 이를 통해 이미지 캡셔닝의 기존 평가 방식의 한계를 극복하고, 동일한 이미지 모달리티 내에서 평가할 수 있는 장점을 제공합니다.

- **Technical Details**: CAMScore는 세 가지 관점(픽셀 레벨, 의미론적 레벨, 객관적 레벨)에서 평가를 수행하는 세 레벨 평가 프레임워크를 포함합니다. 이 프레임워크는 이미지 캡션을 보다 세밀하게 평가할 수 있도록 설계되었으며, 사이클 일관성을 활용하여 기존의 교차 모달리티 메트릭에서 발생하는 모달리티 간격을 해소합니다. 특히, CAMScore는 캡션 생성 모델과 이미지 생성 모델 간의 상호 작용을 통해 평가를 수행합니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋에서 진행된 실험 결과, CAMScore는 인간의 평가와 강한 상관관계를 보이며 기존의 참조 기반 평가 메트릭스와 참조 없는 메트릭스에 비해 우수한 성능을 입증했습니다. Flicker8k-Expert, Flicker8k-CF, COMPOSITE 및 PASCAL-50S 데이터셋에서 평가한 결과, CAMScore는 보여주는 평가 일관성과 신뢰성을 바탕으로 새로운 기준을 제시합니다.



### Bridged Semantic Alignment for Zero-shot 3D Medical Image Diagnosis (https://arxiv.org/abs/2501.03565)
- **What's New**: 이 논문은 3D 의료 영상의 비전-언어 정렬(vision-language alignment)에서의 모달리티 갭을 해결하기 위한 "Bridged Semantic Alignment (BrgSA)" 프레임워크를 제안합니다. 기존의 VLA 방법들에서는 비주얼(visual) 및 텍스츄얼(textural) 임베딩이 두 개의 잘 분리된 클러스터로 형성되어 있기 때문에 이 갭을 줄이기 위한 새로운 접근 방식이 필요합니다. BrgSA는 대형 언어 모델을 이용하여 임상 보고서를 요약하고, cross-modal 지식 상호작용(CMKI) 모듈을 통해 양쪽 모달리티 간의 상호작용을 촉진하여 갭을 좁힙니다.

- **Technical Details**: BrgSA 프레임워크는 두 가지 주요 모듈로 구성되며, 첫 번째는 대형 언어 모델(LLM)을 활용하여 임상 보고서를 요약하며 이로써 고수준의 의미론적 정보를 추출합니다. 두 번째로, cross-modal knowledge bank (CMKB)를 이용하여 이미지와 텍스트 특성 간의 갭을 줄이고, 두 가지 모달리티를 효율적으로 연결할 수 있는 새로운 경험적 접근을 제공합니다. 이 구조를 통해, CLIP 같은 기존 모델의 비효율성을 제거하며, 명시적인 이미지-텍스트 데이터 쌍에 대한 의존성을 줄입니다.

- **Performance Highlights**: BrgSA는 새로운 벤치마크 데이터셋인 "CT-RATE-LT"에서 15개의 저 대표성 비정상 진단을 포함하여 검증을 수행하였습니다. 실험 결과, BrgSA는 기존 최고 성능(SOTA) 방법보다 우수한 성능을 보였으며, AUC(Area Under Curve) 값에서 76.9에서 85.6으로 향상되었습니다. 또한, 기존 벤치마크 데이터셋인 CT-RATE와 RAD-ChestCT에서도 SOTA 성능을 달성하여, 이미지와 텍스트 특성 간의 효과적인 의미론적 정렬을 입증하였습니다.



### PromptGuard: Soft Prompt-Guided Unsafe Content Moderation for Text-to-Image Models (https://arxiv.org/abs/2501.03544)
Comments:
          16 pages, 8 figures, 10 tables

- **What's New**: 이번 연구에서는 PromptGuard라는 새로운 콘텐츠 조절 기술을 제안합니다. 이는 대형 언어 모델(LLMs)의 시스템 프롬프트 메커니즘에서 영감을 받아 안전성을 확보하기 위해 설계되었습니다. 기존 T2I(Text-to-Image) 모델의 경우 NSFW 콘텐츠 생성을 방지하기 위한 직접적인 인터페이스가 없지만, PromptGuard는 텍스트 임베딩 공간 내에서 작동하는 안전 소프트 프롬프트를 최적화하여 이를 해결합니다.

- **Technical Details**: PromptGuard는 안전성 지침을 제공하기 위해 시스템 프롬프트 메커니즘을 T2I 모델의 구조나 매개변수를 수정하지 않고도 구현하는 방법을 찾습니다. 다양한 NSFW 카테고리 간의 보편적인 모더레이션을 달성하기 위해 NSFW 콘텐츠를 성적, 폭력적, 정치적 및 충격적인 내용으로 구분한 후, 각 유형별 안전 프롬프트를 최적화하고 이를 결합하는 방식을 채택합니다.

- **Performance Highlights**: PromptGuard는 이전의 콘텐츠 조절 방법보다 7.8배 빠르며, 최적의 NSFW 제거 비율인 5.84%를 달성하여 8가지 최첨단 방어 기법을 초월합니다. 또한, 비난적인 이미지를 단순히 블러링하거나 블랙 아웃하는 대신, 현실감 있는 이미지를 안전하게 생성하는데 기여합니다.



### Anomaly Triplet-Net: Progress Recognition Model Using Deep Metric Learning Considering Occlusion for Manual Assembly Work (https://arxiv.org/abs/2501.03533)
Comments:
          This paper has been peer-reviewed, revised, and published in Advanced Robotics

- **What's New**: 이 논문에서는 공장 내 제품 조립 과정을 시각화하기 위해 Occlusion을 고려한 진행 인식 방법을 제안합니다. Deep learning 기반의 객체 감지 방법을 사용하여 고정 카메라로 촬영한 이미지에서 조립 대상 제품을 감지하고, 이 감지된 영역을 잘라냅니다. 이후 잘라낸 이미지에 대해 Deep metric learning 기반의 분류 방법을 이용해 조립 작업의 진행 상황을 대략적으로 추정합니다.

- **Technical Details**: 특히, 진행 상황 추정 모델로 Anomaly Triplet-Net을 제안하며, 이는 진행 추정 시 Occlusion을 고려하여 Triplet Loss에 이상 샘플을 추가합니다. 이 방법은 고정 카메라에서 수집한 이미지를 분석하므로, 간섭 요소의 영향을 최소화하고 정확한 진행 상황을 예측할 수 있습니다. 이 모델의 동작 과정은 감지, 크롭, 진행 추정의 순서로 진행됩니다.

- **Performance Highlights**: 실험 결과, Anomaly Triplet-Net을 이용한 진행 추정 방법이 82.9%의 성공률을 기록했습니다. 시스템의 전반적인 실효성을 확인하기 위해 수행된 실험에서도 Detection, Cropping, Progression Estimation의 순서가 효과적으로 작동함을 입증했습니다.



### TexHOI: Reconstructing Textures of 3D Unknown Objects in Monocular Hand-Object Interaction Scenes (https://arxiv.org/abs/2501.03525)
Comments:
          This paper was accepted at ICCVM 2025 and will appear in the proceedings of IEEE TVCG as part of the conference

- **What's New**: 이번 연구는 단안 카메라 데이터를 사용하여 동적인 손-객체 상호작용에서 정확한 텍스처와 기하학 예측을 달성하는 새로운 접근 방식을 제안합니다. 특히, 3D 객체 재구성에서 손의 영향을 고려한 최초의 방법으로, 손의 움직임에서 발생하는 간접 조명과 그림자를 분석하여 객체의 표면 알베도를 정밀하게 예측합니다. 이를 위해, 물체와 함께 손의 pose와 geometry를 최적화하는 중합 렌더링 기법을 활용합니다.

- **Technical Details**: 이 방법은 손과 객체, 배경의 기하학적 구조를 composite rendering을 통해 학습하고, Spherical Gaussian (SG) 표현을 사용하여 환경 조명을 모델링합니다. 손의 영향을 반영하기 위해, 108개의 매개변수화 가능한 구로 손을 단순화한 표현을 도입하여 손에 의한 그림자와 반사를 효율적으로 계산합니다. 또한, 물리 기반 렌더링(Physics-based Rendering, PBR) 프레임워크를 통해 최종 객체 색상을 얻고, 이를 실제 카메라로 촬영한 이미지와 비교하여 렌더링 매개변수를 최적화합니다.

- **Performance Highlights**: 이 방법은 동적인 손-객체 상호작용 시나리오에서 텍스처 재구성 품질에서 최신 기술들을 초월하는 성과를 보였습니다. 연구 결과는 고해상도의 텍스처와 함께 높은 현실감을 제공하여 가상 및 증강 현실 환경에서도 효과적으로 사용될 수 있습니다. 결론적으로, 본 연구는 실제 세계의 복잡한 상호작용을 잘 반영한 3D 모델링의 새로운 가능성을 제시합니다.



### An Empirical Study of Accuracy-Robustness Tradeoff and Training Efficiency in Self-Supervised Learning (https://arxiv.org/abs/2501.03507)
- **What's New**: 이 논문에서는 Self-Supervised Learning (SSL) 분야에서 이미지 표현 학습의 효율성을 높이기 위해 robust EMP-SSL 프레임워크를 재조명합니다. 연구진은 이미지당 크롭 수를 늘려 학습 속도를 향상시키는 접근 방식을 채택하였으며, 전통적인 contrastive learning 방법과는 달리 다중 크롭 샘플링과 불변성 항(invariance term)을 통합하여 훈련 에포크 수를 줄였습니다.

- **Technical Details**: 논문에서 논의된 EMP-SSL 방법은 고정 크기 이미지 패치를 활용하여 훈련 기간을 단축시키며, adversarial training과의 상호작용을 분석하기 위한 몇 가지 핵심 질문을 제시합니다. 연구진은 다중 크롭 기법을 통한 데이터 다양성이 성능을 유지하면서도 계산 비용을 줄일 수 있는지를 평가하고, 다양한 크롭 전략이 모델의 강건성에 미치는 영향을 조사합니다.

- **Performance Highlights**: 실험 결과 robust EMP-SSL은 adversarial self-supervised learning에서 강력한 성과를 보이며, SimCLR과 비교할 때 깨끗한 정확도(clean accuracy)와 강건성(adversarial robustness) 사이의 균형을 더 잘 유지합니다. 또한, Cost-Free Adversarial Multi-Crop Self-Supervised Learning (CF-AMC-SSL) 방법은 훈련 시간을 줄이면서도 두 가지 성능 지표 모두에서 향상된 효과를 보여, 실제 SSL 응용 분야에 실용적인 가능성을 제시합니다.



### Can Deep Learning Trigger Alerts from Mobile-Captured Images? (https://arxiv.org/abs/2501.03499)
- **What's New**: 우리 연구는 모바일 카메라 이미지 데이터를 활용하여 실시간으로 공기 질을 평가하고 추천하는 새로운 접근 방식을 제안합니다. 이 접근 방식은 회귀 기반의 Convolutional Neural Network(CNN) 모델을 개발하여, 서로 연결된 출력 파라미터를 활용한 예측을 통해 기존 모델들보다 우수한 성과를 보여줍니다. 추가로, 데이터셋을 증강하여 훈련 과정에서의 변화를 검증하는 중요한 기여를 하였으며, 원본 데이터셋과 증강 데이터셋 간의 정확도 차이가 미미함을 나타냈습니다.

- **Technical Details**: 이 연구에서 제안된 CNN 기반 회귀 모델은 공기 질 예측을 위한 특별한 설계를 갖추고 있습니다. 특히, PM2.5, NO2, SO2 및 CO와 같은 공기 질 메트릭을 이미지 데이터로 분석하며, 이는 사용자 건강 상태에 적합한 장소 추천에 큰 도움이 됩니다. 또한, 'HealthCamCNN'이라 불리는 실시간 사용자 친화적인 대시보드가 구현되어, 모바일 카메라 이미지에서 유래된 공기 질 지수와 오염 물질 값을 동적으로 표시합니다.

- **Performance Highlights**: 우리가 제안한 모델은 2종 및 5종 오염 물질에 대해 각각 0.0077 및 0.0112의 평균 제곱 오차(Mean Squared Error, MSE)를 달성하여, 기존의 방법과 비교해 우수한 성과를 보여줍니다. 이는 우리의 접근 방식이 공기 질 예측 및 사용자 맞춤형 공기 질 모니터링에서 실질적인 해결책을 제공함을 시사합니다. 궁극적으로, 이 연구는 환경 건강 및 웰빙 결정을 내리는 데 있어 개인에게 필요한 정보를 제공하는 데 기여합니다.



### Textualize Visual Prompt for Image Editing via Diffusion Bridg (https://arxiv.org/abs/2501.03495)
Comments:
          AAAI 2025

- **What's New**: 이번 논문에서는 기존의 pretrained text-guided image-to-image (TI2I) 모델에 의존하지 않고도, 단일 text-to-image (T2I) 모델 기반으로 시각적 프롬프트(visual prompt)를 활용하여 이미지 편집의 일반성과 확장성을 향상시키는 새로운 프레임워크를 제안합니다. 이는 주어진 두 이미지 간의 변화를 텍스트 임베딩(text embedding)으로 변환하는 방법으로, 복잡한 데이터 세트를 구축하거나 재훈련이 필요없습니다.

- **Technical Details**: 제안된 방법에서는 확률 흐름 오르디너리(Euler) 방정식을 활용하여, 텍스트 가이드를 통해 두 이미지 간의 분포를 전이하는 확산 브릿지(diffusion bridge)를 구축합니다. 이 과정에서 매개변수들이 이미지 간의 변화를 나타내도록 텍스트 임베딩을 조정하며, 세밀한 변화를 포착하고 다양한 이미지를 편집하는 데 강력한 성능을 보입니다.

- **Performance Highlights**: 실제 이미지에 대한 실험 결과는 제안된 방법이 기존의 기술들과 비교하여 일반화, 맥락 일관성 및 높은 충실도로 섬세한 편집을 수행할 수 있음을 입증합니다. 단일 이미지 쌍만을 시각적 프롬프트로 사용하여 경쟁력 있는 결과를 달성하며, 이러한 특성은 향후 다양한 형태의 이미지 편집에 필수적일 수 있습니다.



### SceneBooth: Diffusion-based Framework for Subject-preserved Text-to-Image Generation (https://arxiv.org/abs/2501.03490)
- **What's New**: 이번 논문에서는 주제 보존(subject-preserved) 텍스트-이미지 생성에 대한 혁신적인 프레임워크인 SceneBooth를 제안합니다. 기존 방법들이 주제의 외관을 정확하게 보존하는 데 한계를 보였던 반면, SceneBooth는 주어지는 주제 이미지를 고정하고 텍스트 프롬프트에 의해 배경 이미지를 생성하는 새로운 접근방식을 사용합니다. 이로 인해 주제를 좀 더 정확하게 보존할 수 있으며, 상호 조화를 이루는 배경을 생성하는 데 중점을 둡니다.

- **Technical Details**: SceneBooth는 주제 이미지, 객체 구문(object phrases) 및 텍스트 프롬프트를 입력으로 받아들이고, 두 가지 주요 구성 요소인 다중 모달 레이아웃 생성 모듈과 배경 페인팅 모듈을 통합합니다. 다중 모달 레이아웃 생성 모듈은 해당 장면의 텍스트 캡션과 객체 구문에 부합하는 적절한 장면 레이아웃을 생성하여 주제의 위치와 크기를 결정합니다. 배경 페인팅 모듈은 Latent Diffusion Model(LDM)을 기반으로 하여, 배경이 주제와 조화를 이루도록 ControlNet과 Gated Self-Attention을 통합합니다.

- **Performance Highlights**: 정량적 및 정성적 실험 결과는 SceneBooth가 주제 보존, 이미지 조화 및 전반적인 품질 면에서 기존 방법들보다 두드러진 성능 향상을 보임을 보여주었습니다. COCO 데이터셋에서 진행된 광범위한 실험을 통해, SceneBooth의 구체적인 결과가 이전보다 훨씬 높은 주제 보존율과 시각적 품질을 제공함을 입증했습니다.



### VOILA: Complexity-Aware Universal Segmentation of CT images by Voxel Interacting with Languag (https://arxiv.org/abs/2501.03482)
Comments:
          Accepted by AAAI 2025

- **What's New**: 이번 논문에서는 CT 이미지 분할을 위해 VOxel Interacting with LAnguage (VOILA) 방법을 제안합니다. 기존의 fully connected layer 기반 접근법이 다중 클래스 처리에 어려움을 겪는다는 문제점을 해결하고, 텍스트 프롬프트와의 정보 밀도 불균형 문제를 다루기 위해 새로운 프레임워크를 도입했습니다. 또한, pseudo-heatmap을 생성하는 Complexity-Aware Sampling 기법을 사용하여 분할이 어려운 지역에 집중하고 있습니다.

- **Technical Details**: VOILA 방법은 voxel과 언어를 공유된 표현 공간으로 정렬하고, 코사인 유사도를 기반으로 voxel을 분류합니다. 이 방식은 foreground-background 불균형과 대상 비율의 변동으로 인한 영향을 완화하기 위한 Voxel-Language Interaction 프레임워크를 발전시킵니다. Complexity-Aware Sampling 모듈은 Gaussian mixture distribution을 활용하여 학습 중 더 어려운 지역을 동적으로 선택합니다.

- **Performance Highlights**: VOILA는 7개의 공개 데이터 세트에서 저조한 계산 비용으로도 경쟁력 있는 성능을 달성하며, 추가적인 fine-tuning 없이 다양한 데이터 세트에 대한 우수한 범용성을 보여줍니다. 이 방법은 적은 매개변수와 낮은 계산 비용으로 훈련 중에 성능을 향상시키는 데 기여합니다.



### Information-Maximized Soft Variable Discretization for Self-Supervised Image Representation Learning (https://arxiv.org/abs/2501.03469)
- **What's New**: 이 논문에서는 이미지 표현 학습을 위한 새로운 자기 지도 학습(self-supervised learning, SSL) 접근법인 정보 최대화 소프트 변수 이산화(Information-Maximized Soft Variable Discretization, IMSVD)를 소개하고 있습니다. IMSVD는 잠재 공간(latent space)에서 각 변수를 부드럽게 이산화하여 훈련 배치에 대한 확률 분포를 추정할 수 있게 하며, 정보 측정치를 통해 학습 과정을 직접 안내할 수 있습니다. 이 방법론은 정보 이론(information theory)에 기반하여 변환 불변(transform-invariant), 비붕괴(non-collapsed), 중복 최소화(min redundancy)된 표현 특징을 학습하기 위한 새로운 목표 함수를 제안합니다.

- **Technical Details**: IMSVD의 핵심은 feature vector의 각 변수를 softmax 함수를 사용하여 부드럽게 양자화하는 것입니다. 이는 '하드' 이산화와 달리 미분 가능(differentiable)하여 end-to-end 최적화에 쉽게 통합될 수 있습니다. 저자는 정보 이론적 수치를 활용하여 변환 불변적이고 중복이 최소화된 표현 특징을 학습하기 위해 원칙적인 목표 함수를 제안하며, 이를 위해 동등한 공동 교차 엔트로피 손실 함수(joint cross-entropy loss function)를 도출합니다. 이 손실 함수는 특징 변수 간 종속성을 최소화하여 보다 효과적으로 작동할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과 IMSVD는 이미지 분류, 물체 탐지, 인스턴스 분할 등 다양한 다운스트림(downstream) 작업에서 우수한 성능을 나타냅니다. 특히 기존의 최첨단 SSL 방법들과 비교했을 때 더 뛰어난 정확도와 효율성을 보여주며, 일반적인 Downstream 작업에 대한 잘 알려진 통계적 결과를 제공합니다. 또한, IMSVD는 비대조(non-contrastive) 방법이지만 통계적으로 대조 학습(contrastive learning)을 수행하여 대조적 및 비대조적 SSL 방법에 대한 새로운 이해를 제공합니다.



### ScaleMAI: Accelerating the Development of Trusted Datasets and AI Models (https://arxiv.org/abs/2501.03410)
- **What's New**: ScaleMAI는 의료 AI(Medical AI) 데이터세트를 자동으로 구축하고 주석을 추가하는 혁신적인 방법을 제시합니다. 전통적으로 별개로 간주되었던 데이터 생성과 모델 개발 과정을 통합하여, 고품질의 데이터세트를 단기간(몇 개월) 내에 생성할 수 있습니다. 이를 통해 초기 단계부터 전문가와 유사한 성능을 가진 모델을 개발할 수 있게 되었습니다.

- **Technical Details**: ScaleMAI는 25,362개의 CT 스캔 데이터세트를 활용하여 양성/악성 종양과 24개의 해부학적 구조에 대한 세밀한 주석을 제공합니다. 인간 전문가와 연계하여 반복적인 피드백을 통해 모델이 발전하도록 하며, 이 과정에서 LLM(대형 언어 모델)을 사용해 의료 보고서를 해석하는 단계적 접근 방식을 채택했습니다. 또한, 방사선 전문의의 지원을 통해 주석 해석에 필요한 의료 지식과 규칙을 포함하고 있습니다.

- **Performance Highlights**: Flagship Model은 작은 고정 품질 데이터세트로부터 개발된 모델을 능가하며, 종양 검출에서 14%, 분할(segmentation)에서 5%, 분류(classification)에서 72%의 성능 향상을 나타냅니다. 이는 의료 데이터세트 생성의 신속성과 신뢰성을 혁신적으로 향상시켜, 데이터 기반의 여러 응용 프로그램에 큰 잠재력을 제공합니다. 이러한 성과는 정확한 종양 스테이징 및 정밀 방사선 치료 계획에서 모형의 활용을 더욱 강화합니다.



### Compression of 3D Gaussian Splatting with Optimized Feature Planes and Standard Video Codecs (https://arxiv.org/abs/2501.03399)
- **What's New**: 이 논문에서는 3D Gaussian Splatting(3DGS)의 데이터 압축 문제를 해결하기 위한 새로운 방법을 소개합니다. 기존 방법에 비해 저장 용량을 대폭 줄일 수 있는 효율적인 압축 기술을 제안하며, 포인트 클라우드 데이터와 피쳐 플레인(feature planes)을 결합한 통합 아키텍처를 사용합니다. 이 방법은 2D 피쳐 플레인을 활용하여 연속적인 공간 표현을 가능하게 합니다.

- **Technical Details**: 제안된 방법은 블록 단위의 이산 코사인 변환(block-wise discrete cosine transform, DCT)을 활용한 주파수 도메인(entropy modeling) 엔트로피 파라메터화(entropy parameterization)를 포함합니다. 이 기술은 기존의 3DGS 파이프라인과 연결되어, 가우시안 속성을 효과적으로 예측합니다. 또한 채널 중요도 점수(channel importance scores)를 기반으로 한 엔트로피 가중치 기술을 도입하여 비트레이트 소비와 피쳐 플레인 표현 사이의 최적의 거래를 이루도록 합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 기존 방법들에 비해 데이터 압축면에서 뛰어난 성능을 보여 주며, 고화질 렌더링을 유지하면서도 저장 공간을 몇 MB로 줄일 수 있음을 입증했습니다. 이 논문에서 제시한 방법은 특히 이동 장치나 제한된 계산 자원을 가진 환경에서도 실용적입니다. 이로 인해 저장 효율성과 렌더링 품질을 최적화할 수 있음을 강조합니다.



### DoubleDiffusion: Combining Heat Diffusion with Denoising Diffusion for Generative Learning on 3D Meshes (https://arxiv.org/abs/2501.03397)
- **What's New**: 이번 논문에서는 3D 메쉬 표면에서 직접적으로 생성적 학습을 적용할 수 있는 새로운 프레임워크인 DoubleDiffusion을 제안합니다. 이 방법은 열 방산 확산(heat dissipation diffusion)과 디노이징 확산(denoising diffusion)을 결합하여 메쉬 구조를 존중하며 신호 분포를 생성합니다. 기존 방법들이 3D 메쉬를 2D로 펼치는 것이나 필드 표현(field representations)에 의존하는 것과는 달리, DoubleDiffusion은 Laplacian-Beltrami 연산자를 활용하여 효율적인 기하학적 신호 확산을 실현합니다.

- **Technical Details**: DoubleDiffusion 프레임워크는 메쉬의 로컬 기하학적 특성을 유지하면서 신호 생성의 일관성을 보장합니다. Laplacian-Beltrami 연산자를 핵심 요소로 사용하여 열 방산 확산을 근사하고, 메쉬 표면에서의 정보 전송을 원활하게 합니다. 이 과정은 메쉬의 기하학을 기반으로 진행되며, 구조화된 방식으로 데이터가 전파됩니다. 우리는 이를 통해 직접 메쉬에서 정의된 스칼라 분포를 학습할 수 있는 기회를 제공합니다.

- **Performance Highlights**: DoubleDiffusion은 기존의 최첨단 모델보다 312.82% 개선된 커버리지를 달성하고, ∼100k 정점의 대형 메쉬를 단일 패스에서 처리할 수 있습니다. 또한, 메쉬에서의 샘플 생성 속도는 기존의 방법 대비 8.1배 더 빠릅니다. 실험 결과, RGB 분포를 단일 매니폴드에서 생성할 수 있는 능력과 다양한 형태에 대한 텍스처 생성을 통해, DoubleDiffusion의 복잡한 3D 표면에서 기하학적으로 적응 가능한 신호 생성 능력이 입증되었습니다.



### License Plate Images Generation with Diffusion Models (https://arxiv.org/abs/2501.03374)
- **What's New**: 본 연구는 라이센스 플레이트 인식을 위한 합성 데이터 생성의 필요성을 강조하며, 특히 차별화된 접근 방식을 제안합니다. 기존의 GAN 기반 접근법 대신 최신 이미징 기술인 diffusion 모델을 활용하여 사실적인 라이센스 플레이트 이미지를 생성합니다. 이를 통해 제너레이티브 모델의 성능을 실험적으로 검증하고, 생성된 데이터의 특성을 깊이 있게 분석하였습니다. 연구 결과, 합성 데이터가 LPR 작업에 매우 유용함을 실증적으로 확인하였습니다.

- **Technical Details**: 연구진은 우크라이나의 라이센스 플레이트 데이터셋을 사용하여 Denoising Diffusion Probabilistic Model (DDPM)을 훈련했습니다. 이 과정에서 1,000개의 합성 이미지를 생성하고, 이를 수작업으로 성공 및 실패 사례로 분류한 뒤 각 이미지에 대한 정보를 세부 분석하였습니다. 캐릭터 분포 분석과 같은 추가적인 작업을 통해 LPR 모델의 성능을 향상시킬 수 있는 기회를 모색했습니다. 최종적으로 10,000장의 합성 라이센스 플레이트 이미지를 공개하였습니다.

- **Performance Highlights**: 생성된 합성 데이터셋은 실제 데이터셋과 비교하여 LPR 정확도를 3% 향상시키는 결과를 나타냈습니다. 초기에는 실제 데이터와 합성 데이터 간의 성능 격차가 존재했으나, 합성 데이터의 사용으로 데이터 세트를 확장함으로써 성능이 개선되었습니다. 이와 같은 결과는 합성 데이터가 LPR 과제의 효과적인 솔루션이 될 수 있음을 시사합니다.



### Mobile Augmented Reality Framework with Fusional Localization and Pose Estimation (https://arxiv.org/abs/2501.03336)
Comments:
          10 pages, 6 figues

- **What's New**: 이번 연구는 모바일 플랫폼에서의 증강 현실(AR)과 현지화 시스템에 대한 포괄적인 연구를 수행했습니다. 기존의 방식보다 높은 정확도를 구현하기 위해 융합 현지화(fusional localization) 방법과 새로운 포즈 추정 구현을 제안했습니다. 이 프레임워크는 이미지나 Wi-Fi 신호에만 의존하는 방법들보다 더 높은 성능을 제공합니다.

- **Technical Details**: 연구에서는 실내 모바일 AR 구현을 위한 두 가지 핵심 요소인 실내 현지화(indoor localization)와 포즈 추정(pose estimation)을 강조합니다. 다양한 모바일 센서를 기반으로 한 여러 실내 현지화 접근법이 제안되었으며, 비전 기반과 비비전 기반 방법으로 나뉩니다. 연구팀은 무선 지역 네트워크(WLAN)와 관성 센서의 조합을 통해 더 높은 정확도를 달성했습니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 평균 샘플링 그리드 길이 0.5m에서 0.61-0.81m의 낮은 평균 오류 거리와 77%-82%의 높은 일치율을 달성했습니다. 이는 기존의 이미지 기반 또는 WLAN 기반 접근법 대비하여 성능이 높다는 것을 입증합니다.



### CM3T: Framework for Efficient Multimodal Learning for Inhomogeneous Interaction Datasets (https://arxiv.org/abs/2501.03332)
Comments:
          Preprint. Final paper accepted at the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), Tucson, February, 2025. 10 pages

- **What's New**: CM3T(Cross Multimodal Multi-dataset Multitask Transformer)는 Transformer 기반 모델을 새로운 정보에 적응시키는 모델-불가지론(agnostic) 플러그인 아키텍처입니다. 이 연구에서는 다중 헤드 비전 어댑터와 크로스 어텐션 어댑터를 도입하여 비디오 분류 작업에서 필요한 리소스를 대폭 줄였습니다. 기존 모델들을 기반으로 하면서도 학습 효율성을 극대화하는 구조를 통해 최소한의 파라미터로 최첨단 성능을 달성합니다.

- **Technical Details**: CM3T는 고정된 백본(백본 모델) 위에 플러그인 형태로 새로운 모듈을 추가하여 훈련합니다. 이때 전통적인 지도 학습 방법을 통해 사전 훈련된 모델을 활용하며 크로스 어텐션을 통해 다중 모드 학습이 가능하도록 합니다. 이러한 접근 방식은 이용 가능한 다양한 모드에서의 관계를 학습하며, 기존 방법들보다 효율적입니다.

- **Performance Highlights**: CM3T는 Epic-Kitchens-100, MPIIGroupInteraction, UDIVA v0.5와 같은 다양한 데이터셋에서 실험을 진행하여 최첨단 성능에 근접하는 정확도를 기록했습니다. 총 학습 가능한 파라미터는 백본의 12.8%에 불과하며, 추가적인 두 가지 모드를 처리하는 데에는 22.3%의 파라미터만 필요합니다. 이처럼 낮은 자원으로도 뛰어난 성능을 보이는 CM3T는 비디오 분류에서의 활용 가능성을 보여줍니다.



### Plant Leaf Disease Detection and Classification Using Deep Learning: A Review and A Proposed System on Bangladesh's Perspectiv (https://arxiv.org/abs/2501.03305)
- **What's New**: 이 논문은 방글라데시 농업에서의 식물 질병 탐지 및 분류를 위한 새로운 CNN(Convolutional Neural Network) 모델을 제안합니다. 농업은 방글라데시에서 고용, GDP 기여 및 생활수단의 중요한 부분인데, 식물 질병은 생산에 큰 장애물 역할을 합니다. 본 연구에서는 bell peppers, tomatoes, potatoes의 세 가지 작물에 대한 질병 데이터를 수집하여 이를 기반으로 모델을 개발했습니다.

- **Technical Details**: 데이터셋은 Kaggle에서 수집된 17,430장의 이미지를 포함하고 있으며, 각 이미지는 14개의 별도의 손상 클래스가 레이블됩니다. 논문에서 제안한 CNN 모델은 효율적인 성능을 보이며, 테스트된 질병을 성공적으로 탐지하고 분류할 수 있습니다. 이는 기존의 수작업에 비해 높은 정확성을 제공합니다.

- **Performance Highlights**: 개발된 CNN 모델은 작물 질병 관리에 꼭 필요한 강력한 잠재력을 가지고 있습니다. 이를 통해 농업 생산에 있어 보다 정확하고 신속한 질병 탐지가 가능하여, 방글라데시 농민들의 수익성과 생산성을 향상시킬 수 있을 것으로 기대됩니다.



### RAG-Check: Evaluating Multimodal Retrieval Augmented Generation Performanc (https://arxiv.org/abs/2501.03995)
- **What's New**: 이 논문은 Retrieval-Augmented Generation (RAG) 기술의 신뢰성을 평가하기 위한 새로운 프레임워크를 제안합니다. 특히, 논문은 multi-modal RAG에서 발생할 수 있는 새로운 유형의 환각을 다루고 있습니다. 제안된 방법에서는 relevancy score (RS)와 correctness score (CS)를 활용하여 생성된 응답의 정확성을 평가합니다.

- **Technical Details**: 제안된 방식에서는 RAG의 데이터 선택 과정과 컨텍스트 생성 방법에서 환각이 발생할 수 있음을 강조합니다. RS 모델은 데이터베이스에서 선택된 정보 조각과 사용자 쿼리 간의 관련성을 평가하며, CS 모델은 생성된 응답의 정확성을 평가합니다. 두 모델은 ChatGPT 데이터를 사용하여 훈련되었으며, 약 88%의 정확도를 달성하였습니다.

- **Performance Highlights**: 연구 결과, RS 모델은 CLIP보다 20% 더 많은 경우 인간의 선택과 일치하며, CS 모델은 약 91%의 정확도로 인간의 선호도를 반영합니다. 또한, 5000개의 샘플로 구성된 인간 주석 데이터베이스를 통해 선택된 정보 조각의 관련성과 응답 진술의 정확성을 평가하였습니다.



### VLM-driven Behavior Tree for Context-aware Task Planning (https://arxiv.org/abs/2501.03968)
Comments:
          10 pages, 11 figures, 5 tables. Last updated on January 7th, 2024

- **What's New**: 본 논문에서는 로봇 커뮤니티에서 주목받고 있는 행동 트리(Behavior Trees, BT) 생성에 비전-언어 모델(Vision-Language Models, VLMs)을 활용하는 새로운 프레임워크를 제안합니다. 이 프레임워크는 시각적 조건에 맞춰 BT를 상호작용적으로 생성하고 편집할 수 있도록 해줍니다. 특히, '자기 유도 시각 조건(self-prompted visual conditions)'을 기반으로 하여 로봇이 시각적 정보에 따라 환경 인식 결정을 내릴 수 있도록 지원합니다.

- **Technical Details**: 이 프레임워크는 VLM을 통해 BT를 생성하며, 조건문과 같은 시각적 조건 노드를 포함합니다. 이러한 조건들은 자유 서식 텍스트로 표현되며, 로봇의 실행 중 실제 이미지와 대조하여 평가됩니다. 예를 들어, '테이블에서 컵을 치우세요'라는 지침을 바탕으로 VLM은 '테이블에 컵이 없는지 확인'하는 조건 노드를 생성합니다. 이를 통해 로봇은 복잡한 환경에서도 적절한 결정을 내릴 수 있게 됩니다.

- **Performance Highlights**: 제안된 프레임워크는 실제 카페 시나리오에서 유인 로봇을 이용하여 검증되었습니다. 시험 결과, 다양한 조건문을 포함한 BT들이 효과적으로 생성되고 실행되었음을 보여주었습니다. 또한, BT의 가시화 및 상호작용 편집을 통해 안전성과 투명성을 높이는 인터페이스를 개발하여, 로봇 프로그램의 신뢰성을 강화했습니다.



### Vision Language Models as Values Detectors (https://arxiv.org/abs/2501.03957)
Comments:
          13 pages, 2 figures

- **What's New**: 대형 언어 모델(LLMs)과 비주얼 입력을 통합한 연구는 복합 데이터를 해석하는 새로운 가능성을 제시하고 있습니다. 본 연구는 LLMs가 가정 환경에서 관련 요소를 탐지하는 능력을 평가하며, 인간 주석가와의 정렬을 살펴봅니다. 12개의 이미지를 생성하고, 14명의 주석가가 각 이미지의 핵심 요소를 식별하도록 하여, 총 5가지 LLM(GPT-4o, LLaVA 변형 등)의 출력을 인간 응답과 비교합니다.

- **Technical Details**: 연구에서는 이미지와 텍스트를 입력으로 받아들이는 LLM을 평가합니다. 고급 비전 언어 모델(VLMs)은 이미지에서 시각적 특징을 추출하고 이와 대응하는 텍스트 표현과 정렬합니다. 여기서 LLaVA 모델은 사전 훈련된 CLIP 시각 인코더를 활용하며, 출력은 LLM의 임베딩 공간과 동일한 차원성을 갖도록 훈련된 프로젝션 매트릭스를 통해 전달됩니다.

- **Performance Highlights**: 결과에 따르면, LLaVA 34B 모델이 가장 높은 성과를 보였지만 여전히 낮은 점수를 기록했습니다. 또한, 모델들은 이미지에서 가치 있는 요소를 탐지할 수 있는 잠재력을 보였으며, 이러한 점은 사회 로봇공학, 보조 기술 및 인간-컴퓨터 상호작용 분야에서 개선된 적용 가능성을 시사합니다. 이 연구는 더 나은 훈련과 세분화된 프롬프트를 통해 LLM이 보다 깊이 있는 통찰을 제공하고, 더 맥락적으로 관련된 응답을 생성할 수 있음을 제안합니다.



### Explainable AI model reveals disease-related mechanisms in single-cell RNA-seq data (https://arxiv.org/abs/2501.03923)
- **What's New**: 본 연구에서는 신경퇴행성 질환(NDD)의 이해를 돕기 위해 신경망 모델(Neural Network, NN)과 설명 가능한 인공지능(Explainable AI, XAI)을 결합한 방법을 제안합니다. 이를 통해 단일 세포 수준에서 질병 관련 유전자를 식별하고 해당 유전자들의 기전(mechanism)을 설명할 수 있는 새로운 접근 방식을 탐구하였습니다. 또한, 헌팅턴병(Huntington's disease, HD)에 대한 데이터를 분석하여 기존의 차등 유전자 발현 분석(differential gene expression analysis, DGE) 기법과 SHAP(SHapley additive explanations) 방법을 비교하였습니다.

- **Technical Details**: 연구에 사용된 방법론은 세포 수 42,800개와 2,500개의 가장 변동성이 큰 유전자를 포함하는 정규화된 유전자 카운트 행렬을 생성하는 데 초점을 맞추었습니다. Seurat를 사용하여 클러스터링한 결과, 두 가지 스파이니 신경세포 집단을 포함한 17개 클러스터를 발견하였으며, 이를 통해 각 클러스터의 특정 세포 유형을 지정했습니다. 기계 학습 접근법 중 Multi-Layer Perceptron (MLP)를 활용하여 헌팅턴병 관련 세포의 확률을 예측하고, SHAP 값을 통해 유전자 기여도를 평가했습니다.

- **Performance Highlights**: DGE 방법과 SHAP 접근법은 서로 공통적 및 차별적인 유전자 세트를 제시하며 질병의 이해도를 높이는 데 기여합니다. 연구 결과는 NN 모델이 복잡한 생물학적 기전을 이해하는 데 있어 뛰어난 성능을 발휘할 수 있음을 보여주며, XAI 기법이 질병의 기전에 대한 통찰을 제공한다고 강조합니다. 이러한 결과는 신경퇴행성 질환의 연구에 있어 NN과 XAI의 융합이 새로운 가능성을 열어줄 것으로 기대됩니다.



### Dolphin: Closed-loop Open-ended Auto-research through Thinking, Practice, and Feedback (https://arxiv.org/abs/2501.03916)
Comments:
          19 pages, 11 figures, and our homepage: this https URL

- **What's New**: 이 논문에서는 인공지능(AI)을 활용한 자동 과학 연구 프레임워크인 Dolphin을 제안합니다. Dolphin은 아이디어 생성, 실험 수행, 결과 피드백의 순환 구조를 구축하여 인류의 연구 과정을 자동화합니다. 이 프레임워크는 새로운 연구 아이디어를 생성하고 실험 결과를 분석하여 다음 단계로 피드백을 제공합니다.

- **Technical Details**: Dolphin은 주제와 작업 속성에 따라 관련 논문을 기반으로 새로운 아이디어를 생성합니다. 자동 생성된 코드 및 실험 계획을 바탕으로 디버깅을 수행하며, 실험 결과를 분석하여 더 높은 품질의 아이디어를 생성하는 데 기여합니다. 이를 통해 2D 이미지 분류 및 3D 포인트 분류와 같은 작업에서 최신 기술과 경쟁 가능한 방법을 제시합니다.

- **Performance Highlights**: Dolphin은 다양한 주제와 벤치마크 데이터셋에서 실험을 수행하며 지속적으로 새로운 아이디어를 생성할 수 있습니다. 실험 결과는 Dolphin이 기존의 방법들과 비교할 때 유의미한 성과를 거두고 있음을 보여줍니다. 특히, 제안된 닫힌 루프 설계를 통해 아이디어의 품질이 개선되는 것을 확인하여, 자동 연구의 효과성을 입증하고 있습니다.



### SELMA3D challenge: Self-supervised learning for 3D light-sheet microscopy image segmentation (https://arxiv.org/abs/2501.03880)
Comments:
          1st version

- **What's New**: 최근 광층 현미경(light sheet microscopy)과 조직 투명화(tissue clearing) 기술의 혁신은 대형 포유류 조직을 세포 해상도로 3D 이미징할 수 있게 해주었습니다. 그런 이 기술들은 딥러닝에 기반한 대규모 데이터 분석의 발전과 결합되어 연구자들이 다양한 생물 샘플의 형태학적, 기능적 속성을 신속하게 조사할 수 있는 능력을 제공합니다. 그러나 이러한 모델들은 도메인 차이에 민감하여 훈련 분포 외의 데이터에 적용 시 정확도가 현저히 떨어집니다.

- **Technical Details**: 이 논문에서는 SELMA3D 챌린지를 통해 3D 광층 현미경 이미지 세분화를 위한 자가 지도 학습(self-supervised learning) 방법 개발을 장려하고자 합니다. SELMA3D는 35개의 대형 3D 이미지 및 315개의 주석이 달린 작은 패치들로 이루어진 풍부한 라이트 시트 이미지를 제공합니다. 이는 생물학적 구조와 명확하게 구별할 수 있는 특징에 초점을 두고 있어 세분화 정확도를 크게 향상시킬 수 있는 기회를 제공합니다.

- **Performance Highlights**: 최근 챌린지에 참여한 다섯 팀의 방법론을 정리한 결과, 대부분의 팀들이 제안한 모델이 자가 지도 학습을 통해 세분화 성능과 일반화 능력을 향상시켰음을 보여주었습니다. 이 결과를 통해 SELMA3D가 자가 지도 학습을 통한 3D 현미경 이미지 세분화 분야의 혁신적인 진행을 지향하고 있음을 알 수 있습니다. 향후 이 챌린지는 3D 세분화 애플리케이션의 주석 노력을 현저히 줄이는데 기여할 것입니다.



### Neuromorphic Optical Tracking and Imaging of Randomly Moving Targets through Strongly Scattering Media (https://arxiv.org/abs/2501.03874)
Comments:
          22 pages, 6 figures

- **What's New**: 이 연구는 밀도가 높은 산란 매질 속에서 무작위로 움직이는 대상을 추적하고 이미지를 획득하는 새로운 접근법을 소개합니다. 이벤트 감지 카메라(event detecting camera)와 다단계 신경형 심층 학습 전략(multistage neuromorphic deep learning strategy)을 결합하여 일반적으로 보이지 않는 객체를 추적할 수 있는 방법을 개발하였습니다.

- **Technical Details**: 이 방법은 밀도가 높은 산란 매질에서 나오는 광자를 감지하여 픽셀 단위의 비동기 스파이크 열(spike trains)로 변환합니다. 이후 이 스파이킹 데이터를 심층 스파이킹 신경망(deep spiking neural network) 엔진에 공급하여 객체 추적과 이미지 재구성을 병렬로 수행합니다. 이러한 과정은 시간적으로 분리된 단계로 진행되며, 다양한 실험을 통해 효과성을 입증하였습니다.

- **Performance Highlights**: 벤치탑 실험에서는 밀도가 높은 탁한 매질 속에서 무작위로 움직이는 객체의 추적 및 이미징이 이루어졌습니다. 또한 공간적으로 정지해 있지만 광학적으로 동적인 객체들의 이미지 재구성도 성공적으로 수행했습니다. 이 연구는 높은 계산 효율성과 낮은 전력 소비를 자랑하는 신경형 접근법의 이점을 강조합니다.



### Semise: Semi-supervised learning for severity representation in medical imag (https://arxiv.org/abs/2501.03848)
Comments:
          Accepted for presentation at the 2025 IEEE 22nd International Symposium on Biomedical Imaging (ISBI)

- **What's New**: 이 연구에서는 SEMISE라는 새로운 방법론을 소개합니다. SEMISE는 self-supervised learning과 supervised learning의 조합을 통해 의료 영상에서의 표현 학습 문제를 해결하는 혁신적인 접근법입니다. 데이터 부족 문제를 해결하고, 중요한 특징들을 추출할 수 있는 인코더의 능력을 향상시키며, 의료 영상 분석 및 건강 관리 애플리케이션에 보다 정확한 솔루션을 제공합니다.

- **Technical Details**: SEMISE는 labeled 데이터와 augmented 데이터를 활용하여 표현 학습을 최적화합니다. 이 방법은 다양한 세 가지 학습 전략인 self-supervised representation learning (SSL), supervised representation learning (SRL), semi-supervised representation learning (SemiSL)을 통합합니다. 이러한 접근법을 통해 모델은 건강한 샘플과 비정상 샘플을 구분하고, 다중 임무에서의 성능을 개선할 수 있습니다.

- **Performance Highlights**: SEMISE는 분류에서 12%, 세분화에서 3%의 성능 향상을 달성했습니다. 이는 기존 방법들을 능가하는 결과로, F1 score, MAEE, IoUs, DICE 등의 성능 지표에서 우수한 결과를 나타냈습니다. 이러한 성과는 SEMISE가 의료 이미지 분석을 발전시키고, 라벨 데이터가 제한된 상황에서도 유용한 것임을 보여줍니다.



### MedFocusCLIP : Improving few shot classification in medical datasets using pixel wise attention (https://arxiv.org/abs/2501.03839)
- **What's New**: 이번 연구에서는 기존의 CLIP (Contrastive Language-Image Pretraining) 모델의 몇 가지 문제점을 해결하기 위해 MedFocusCLIP라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 Segment Anything Model 2 (SAM2)를 활용하여 의료 이미지를 보다 효과적으로 분류하는 방법을 제시합니다. 특히, SAM2의 세분화 능력을 통해 CLIP의 주의를 관련 영역으로 유도함으로써, 의료 영상에서의 정확도를 획기적으로 향상시킵니다.

- **Technical Details**: MedFocusCLIP는 CLIP의 비전 인코더와 SAM2의 세분화 마스크를 결합하여 이미지의 주목할 만한 영역에 집중합니다. 해당 프레임워크는 텍스트와 이미지의 멀티모달 피처를 융합하여 분류 성능을 개선하는 방식으로 작동합니다. 텍스트 프롬프트는 CLIP의 텍스트 인코더로 변환되어 고차원 임베딩으로 생성되며, 세분화 마스크는 주목할 만한 영역을 강조하여 CLIP 진단력을 극대화합니다.

- **Performance Highlights**: 제안된 시스템은 COVID, 폐 질환, 뇌종양, 유방암 데이터셋에 대해 각각 71%, 81%, 86%, 58%의 정확도를 기록했습니다. 이는 기존의 사전 훈련된 CLIP 모델이 보여준 66%, 70%, 68%, 29%의 결과에 비해 상당히 향상된 성능입니다. 또한, 이 프레임워크는 정확한 진단을 위한 병소의 국소화 능력을 강화하여 의료 이미지 분류의 해석 가능성을 높입니다.



### SCC-YOLO: An Improved Object Detector for Assisting in Brain Tumor Diagnosis (https://arxiv.org/abs/2501.03836)
- **What's New**: 이 논문에서는 뇌 종양 감지를 위한 새로운 SCC-YOLO 아키텍처를 개발하였습니다. 이 아키텍처는 SCConv 주의(attention) 메커니즘을 YOLOv9에 통합하여 이미지 특성 학습을 향상시킵니다. SCC-YOLO는 다양한 주의 메커니즘과 YOLOv9 모델의 통합 효과를 연구하며, 두 개의 데이터셋(BR35H 및 Brain_Tumor_Dataset)을 사용했습니다.

- **Technical Details**: SCC-YOLO 아키텍처는 효율적인 합성곱 모듈(convolutional module)을 재구성하는 SCConv 모듈을 사용하여 특성 간의 공간 및 채널 중복성을 줄입니다. 이 방법은 YOLOv9의 성능을 더욱 향상시키는 데 기여합니다. 다양한 주의 메커니즘을 통합하여 뇌 종양 이미지 감지의 성능을 분석하였습니다.

- **Performance Highlights**: 브레인 종양 데이터셋인 BR35H에서 SCC-YOLO는 YOLOv9에 비해 mAp50에서 0.3% 향상을 보였으며, 자체 제작한 Brain_Tumor_Dataset에서는 0.5% 향상을 나타냈습니다. SCC-YOLO는 뇌 종양 감지 분야에서 최신 기술 수준의 성능을 달성했으며, 관련 소스 코드도 제공됩니다.



### Deep Sylvester Posterior Inference for Adaptive Compressed Sensing in Ultrasound Imaging (https://arxiv.org/abs/2501.03825)
- **What's New**: 이번 논문에서는 초음파 영상의 획득 과정을 최적화하기 위한 새로운 적응형 서브샘플링 방법을 제안합니다. 이 방법은 Sylvester Normalizing Flow 인코더를 사용하여 부분 관찰 하에 대략적인 Bayesian posterior를 실시간으로 유추하며, 다음 영상 프레임과 서브샘플된 관측치 간의 상호 정보를 극대화하는 샘플링 계획을 수립합니다.

- **Technical Details**: 제안한 방법은 부분 관측값을 기반으로 한 정보 이득의 최대화를 목표로 하며, 이를 위해 딥 생성 모델과 빠른 Bayesian posterior 추론을 결합합니다. Bayesian posterior는 복잡한 상태 분포를 효과적으로 추정할 수 있도록 설계되며, 적은 횟수의 신호 전송으로 더 높은 해상도를 유지합니다. 서브샘플링 행렬은 각 초음파 프레임에 대해 순차적으로 최적화됩니다.

- **Performance Highlights**: EchoNet 심장 초음파 비디오 데이터 세트를 활용하여 제안된 방법이 균일 난수 샘플링 및 등距 간격 스캔 라인과 같은 기존의 방법에 비해 평균 절대 재구성 오류를 15% 개선함을 보였습니다. 또한, posterior 추론과 샘플링 계획 생성이 단 0.015초 만에 완료되어 실시간 2D 초음파 영상 응용에 적합한 속도를 제공합니다.



### Re-Visible Dual-Domain Self-Supervised Deep Unfolding Network for MRI Reconstruction (https://arxiv.org/abs/2501.03737)
- **What's New**: 본 논문에서는 자기공명영상(MRI) 내에서 저해상도 데이터만을 활용해 학습할 수 있는 새로운 접근 방식을 제안합니다. 구체적으로, 재가시화된 이중 도메인(self-supervised) 구조를 통해 학습할 수 있는 데이터를 최대한 활용하여 정보 손실을 줄이고 이미지 감도(contrast)와 해상도를 최적화합니다. 또한, 이 구조는 반복 최적화 알고리즘을 네트워크에 융합하여 해석 가능성과 성능을 향상시키는 것을 목표로 합니다.

- **Technical Details**: 제안된 방법에서는 재가시화된 이중-도메인 손실(dual-domain loss)을 도입하여 단일 저해상도 k-space 데이터에만 의존하는 기존 방법들의 한계를 극복하고, 정보 손실을 방지하기 위한 훈련 과정에서 모든 저해상도 k-space 데이터가 활용됩니다. DUN-CP-PPA 방식으로 이미징 물리학(imaging physics)과 이미지 사전(image priors)을 직접 포함하여 복원 과정을 안내합니다. 공간-주파수(SFFE) 블록을 적용하여 전체 및 부분 특성(global and local feature) 표현을 학습하는 능력을 강화했습니다.

- **Performance Highlights**: fastMRI 및 IXI 데이터셋을 통한 실험 결과, 제안한 방법이 기존의 최첨단 기술들에 비해 복원 성능(reconstruction performance)에서 유의미한 향상을 보였음을 입증했습니다. 이를 통해 제안된 모델이 데이터 수집이 어려운 임상 환경에서의 다양한 MRI 프로토콜에 적용할 수 있는 가능성을 높임을 알 수 있습니다. 이러한 결과는 딥러닝 기반 MRI 접근 방식의 실제 임상 활용 가능성을 더욱 확대할 것입니다.



### VTAO-BiManip: Masked Visual-Tactile-Action Pre-training with Object Understanding for Bimanual Dexterous Manipulation (https://arxiv.org/abs/2501.03606)
- **What's New**: 이번 연구에서는 VTAO-BiManip이라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 시각-촉각-행동(pretraining)과 객체 이해(object understanding)를 결합하여 인간처럼 양손을 사용하는 조작을 가능하게 하는 커리큘럼 강화 학습(curriculum RL)을 채택하고 있습니다. 특히, 손의 움직임 데이터를 포함하여 이중 손 조정을 위한 보다 효과적인 가이드를 제공함으로써 기존의 방법들을 개선하였습니다.

- **Technical Details**: VTAO-BiManip은 VTAO 데이터 수집 시스템을 활용하여 시각, 촉각, 행동 및 객체 정보의 다중 모드를 수집합니다. 이 모델은 마스크된 다중 모달 입력을 사용하여 미래의 행동뿐만 아니라 객체의 자세(pose)와 크기를 예측하여 크로스 모달 정규화(cross-modal regularization)를 촉진합니다. 두 단계의 커리큘럼 RL 접근 방식을 통해 학습 안정성을 높이고, 세 가지 하위 작업(잡기, 유지, 병뚜껑 열기)의 조화로운 학습을 지원합니다.

- **Performance Highlights**: 제안된 VTAO-BiManip 방법은 병뚜껑을 여는 작업을 시뮬레이션 및 실제 환경에서 평가한 결과, 기존의 시각-촉각 사전 훈련 방법보다 20% 이상 높은 성공률을 기록했습니다. 이는 양손 협력 조작 기술에서 기존 방법들과 비교하여 올바른 방향으로 나아가는 성과를 나타냅니다. 이를 통해 인간과 유사한 조작 기술의 학습 가능성을 더욱 확장할 수 있는 기반이 마련되었습니다.



### A Value Mapping Virtual Staining Framework for Large-scale Histological Imaging (https://arxiv.org/abs/2501.03592)
- **What's New**: 본 연구에서는 다양한 조건에 적응 가능한 일반적인 가상 염색 프레임워크를 소개합니다. 새롭게 제안된 Value Mapping Generative Adversarial Network (VM-GAN)는 서로 다른 병리학적 모달리티 간의 가상 염색 정확성을 보장하기 위한 손실 함수를 제안합니다. 또한, 패치 단위 처리로 발생하는 경계 불일치를 해결하기 위한 신뢰 기반 타일링 방법을 도입하였습니다. 이 방법은 대규모 고해상도 이미지를 처리하는 데 적합하여 기존 방법보다 개선된 성능을 보여줍니다.

- **Technical Details**: 제안된 방법은 먼저 일반적으로 사용되는 RGB 공간에서 HSV 공간으로 가상 염색 과정이 전환됩니다. 우리는 가치 손실(value loss)을 설계하여 다양한 색상 모달리티 간의 변환 일관성을 보장합니다. VM-GAN 네트워크의 손실 함수 설계와 대규모 샘플을 위한 신뢰 타일링(schema)을 포함한 일반 가치 매핑 프레임워크를 구성합니다. HSV 색공간은 색상을 다루는 더 직관적인 방식을 제공하며, 이는 컴퓨터 비전 분야에서 색상의 조작은 물론 밝기, 채도 및 색상 조정에 유용합니다.

- **Performance Highlights**: 다양한 염색 프로토콜로 이루어진 데이터에 대한 실험 결과, 우리의 방법은 정량적 지표에서 우수한 성능을 달성하였고 시각적 인식의 개선도 확인되었습니다. 대규모 및 고해상도 이미지 처리에서 경계 불일치 문제를 효과적으로 해결하여, 기존의 여러 방법들과 비교하여 눈에 띄는 개선을 나타냅니다. 이 연구는 향후 병리학적 이미지 변환 작업에서의 잠재력을 확장할 것으로 기대됩니다.



### Enhanced Tuberculosis Bacilli Detection using Attention-Residual U-Net and Ensemble Classification (https://arxiv.org/abs/2501.03539)
- **What's New**: 이 논문은 기존의 저렴한 microscope 기반의 결핵 감지 방법의 한계를 극복하기 위해 딥러닝과 앙상블 모델을 결합한 새로운 하이브리드 접근 방식을 제안합니다. 향상된 U-Net 모델은 주의 블록(attention blocks)과 잔여 연결(residual connections)을 통합하여 미세한 기관에서의 균의 정확한 분할을 목표로 합니다. 이 연구는 SVM, Random Forest, Extreme Gradient Boost(XGBoost) 등을 포함하는 앙상블 분류기를 사용하여 높은 정확도로 결핵균을 식별합니다.

- **Technical Details**: 기술적으로 제안된 모델은 U-Net을 기반으로 하며, 이를 통해 이미지에서 관심 영역(Regions of Interest, ROIs)을 효율적으로 추출합니다. 픽셀 분류기를 사용하여 균의 지역을 찾아내고 이후 앙상블 모델을 통해 분류 과정을 수행합니다. 이러한 방법은 회귀계수와 색상, 모양과 같은 특징 벡터를 활용하여 균의 특정 지역을 효과적으로 나타냅니다. 또한, 이 연구는 여러 공공 데이터셋과 새로운 데이터셋에서 실험을 수행하며 방법론의 유효성을 입증합니다.

- **Performance Highlights**: 제안한 모델은 기존의 방법들과 비교했을 때 뛰어난 분할 성능(segmentation performance)과 높은 분류 정확도(classification accuracy)를 달성했습니다. 실험 결과, 제안된 접근방식은 특히 자동화 수준을 높여주어 검사자의 업무 부담을 줄이는 데 기여할 것으로 나타났습니다. 또한, 결핵 진단에서 발생하고 있는 새로운 케이스의 수가 증가함에 따라 이 방법의 필요성이 더욱 강조되고 있습니다.



### Efficient and Accurate Tuberculosis Diagnosis: Attention Residual U-Net and Vision Transformer Based Detection Framework (https://arxiv.org/abs/2501.03538)
- **What's New**: 본 논문은 결핵으로 인한 균의 검출을 위한 혁신적인 이단계 심층학습(deep learning) 방법론을 제안하고 있습니다. 이 방법은 균의 분할(segmentation) 후 분류(classification)를 통해 결핵균(Mycobacterium tuberculosis)의 정확한 탐지를 목표로 합니다. 특히, U-Net 모델과 Vision Transformer(TBViT)를 사용하여 이미지에서 높은 정밀도로 결핵균을 탐지할 수 있는 특징을 갖추고 있습니다.

- **Technical Details**: 첫 번째 단계에서는 주목(attention) 블록과 잔차 연결(residual connections)을 활용한 U-Net 모델로, 미세한 결핵균을 검출하기 위해 이미지의 관심 영역(Regions of Interest, ROIs)을 분할합니다. 그 후, 추출된 ROIs를 Vision Transformer를 기반으로 하는 TBViT로 분류하여 결핵균 식별의 정밀도를 향상시킵니다. 연구에 사용된 데이터셋은 Ziehl-Neelsen 염색 슬라이드에서 유래한 미세한 가래 얼룩 이미지로 구성되어 있습니다.

- **Performance Highlights**: 제안된 모델은 다양한 지표를 통해 과거 방법들과 비교했을 때 현저히 향상된 분할 성능과 높은 분류 정확도, 그리고 자동화 수준의 증가를 보여줍니다. 이는 결핵 진단의 신뢰성과 효율성을 개선하여 공중 보건 관점에서 결핵 통제를 더욱 용이하게 할 것입니다. 실험 결과는 제안된 방법이 결핵균 검출을 위한 최신 기법들과 비교하여 우수성을 입증했음을 나타냅니다.



### FgC2F-UDiff: Frequency-guided and Coarse-to-fine Unified Diffusion Model for Multi-modality Missing MRI Synthesis (https://arxiv.org/abs/2501.03526)
- **What's New**: 본 논문에서는 뇌 종양의 진단 및 치료를 위해 다중 모드 MRI(magnetic resonance imaging)가 필요하다는 점을 강조합니다. 그러나 스캔 시간, 스캔 손상, 아티팩트, 모션, 조영제 불내성 등으로 인해 일부 모드가 결여되는 문제가 자주 발생합니다. 이러한 문제를 해결하기 위해 새로운 통합 합성 모델인 Frequency-guided and Coarse-to-fine Unified Diffusion Model(FgC2F-UDiff)을 제안합니다.

- **Technical Details**: FgC2F-UDiff 모델은 Coarse-to-fine Unified Network(CUN)와 Frequency-guided Collaborative Strategy(FCS)를 사용하여 이미지의 충실도를 향상시키고 비선형 매핑을 지도하는 방식으로 설계되었습니다. CUN은 노이즈 제거 과정을 전역에서 세부사항으로 나누어 두 단계로 진행하여 이미지를 합성하는 데 도움을 줍니다. 또한, Specific-acceleration Hybrid Mechanism(SHM)은 확산 모델을 가속화하고 다대다(many-to-many) 합성의 가능성을 높여 줍니다.

- **Performance Highlights**: 제안된 FgC2F-UDiff 모델은 두 개의 데이터 세트에서 우수한 성능을 보였으며, PSNR, SSIM, LPIPS, FID와 같은 정량적 메트릭스를 통해 철저하게 검증되었습니다. 실험 결과는 모델이 기존 합성 기술에 비해 개선된 결과를 보였음을 나타냅니다. 이러한 성과는 임상 실무 및 연구에서의 모드 불충분 문제를 해결하는 데 큰 기여를 할 것으로 기대됩니다.



### Salient Region Matching for Fully Automated MR-TRUS Registration (https://arxiv.org/abs/2501.03510)
- **What's New**: 이 연구에서는 전자동으로 MR(자기공명영상)과 TRUS(직장 초음파) 등록을 위한 두드러진 영역 매칭 프레임워크를 제안합니다. 기존의 방법들과의 주요 차별점은 구조(structure)와 강도(intensity) 유사성을 모두 고려한다는 점입니다. 또한, 프레임워크는 전립선 세분화(prostate segmentation), 강체 정렬(rigid alignment), 변형 등록(deformable registration)으로 구성되어 있습니다. 이 연구는 자동화된 MR-TRUS 등록 방법이 최신 방법들보다 우수한 성능을 보임을 입증합니다.

- **Technical Details**: 제안된 프레임워크는 세 가지 주요 구성 요소로 이루어져 있습니다. 첫번째 단계인 ROI(관심 영역) 세분화에서는 두 개의 V-Net을 사용하여 MR과 TRUS 각각의 전립선 영역을 세분화합니다. 두번째 단계는 강체 정렬로, 예측된 전립선 영역을 사용하여 서로 다른 모달리티 이미지를 같은 전역 좌표에 통합합니다. 마지막으로, 변형 등록 단계에서 인코더-디코더 구조의 네트워크가 최종 등록을 위한 볼륨 변형 필드를 회귀합니다.

- **Performance Highlights**: 실험 결과는 공용 MR-TRUS 데이터 세트를 사용하였으며, 제안된 방법이 여러 최신 방법들보다 우수한 등록 결과를 달성함을 보여주었습니다. 또한, 제안된 두드러진 영역 매칭 손실(salient region matching loss)은 전립선 영역에서의 구조와 강도 유사성을 효과적으로 평가하여 성능을 향상시킵니다. 전체적으로 이 연구는 MR-TRUS 등록의 정확성과 자동화를 높이는 혁신적인 접근 방식을 제공합니다.



### Hyperbolic Binary Neural Network (https://arxiv.org/abs/2501.03471)
- **What's New**: 본 논문에서는 가벼운 모바일 디바이스에 최적화된 Hyperbolic Binary Neural Network (HBNN)을 제안합니다. 이는 이론적으로 하이퍼볼릭 기하학(hyperbolic geometry)의 구조를 이용하여 제약 최적화 문제(constrained optimization problem)를 해결합니다. Riemannian exponential map을 활용하여 하이퍼볼릭 공간의 제약 문제를 유클리드 공간의 비제약 문제(unconstrained problem)로 변환합니다.

- **Technical Details**: HBNN은 이론적으로 제약 문제를 최적화하기 위해 제안된 새로운 접근법으로, 실험적으로 제시된 Exponential Parametrization Cluster (EPC) 메서드는 Riemannian exponential map과 비교하여 구간(domain)을 축소합니다. 이 방법은 가중치 플립(weight flips)의 확률을 증가시켜 BNN의 정보 이득(information gain)을 극대화하는데 기여합니다.

- **Performance Highlights**: CIFAR10, CIFAR100, ImageNet 데이터셋에서 VGGsmall, ResNet18, ResNet34 모델을 통해 HBNN의 실험 결과는 기존 최첨단 방법들에 비해 우수한 성능을 보여줍니다. 이러한 성능 증가는 저전력 및 자원이 제한된 장치에서의 딥러닝 활용 가능성을 높입니다.



### DGSSA: Domain generalization with structural and stylistic augmentation for retinal vessel segmentation (https://arxiv.org/abs/2501.03466)
- **What's New**: 이 논문에서는 DGSSA라는 새로운 접근 방식을 제안하여 망막 혈관 이미지 분할을 개선합니다. 이는 구조적 및 스타일 증강(strategies) 방법을 결합하여 모델의 일반화 능력을 향상시키는 데 중점을 둡니다. 기존의 전통적인 분할 방법들이 가지고 있는 한계를 극복하고 다양한 도메인에서의 성능 저하 문제를 해결합니다.

- **Technical Details**: 이 연구에서는 공간 식민지화 알고리즘을 사용하여 실제 망막 혈관과 유사한 다양한 혈관 모양 구조를 생성합니다. Pix2Pix 모델을 개선하여 생성된 유사 망막 이미지로부터 다양한 구조 분포를 학습하는 분할 모델을 훈련합니다. 또한, PixMix를 활용해 무작위 광도 증강(random photometric augmentations) 및 불확실성 섭동(perturbations)을 도입하여 스타일 다양성을 풍부하게 하여 다양한 이미징 조건에 대한 모델의 적응력을 증대시킵니다.

- **Performance Highlights**: 본 연구의 프레임워크는 DRIVE, CHASEDB, HRF, STARE라는 네 가지 도전적인 데이터셋에서 엄격하게 평가되었습니다. 결과는 기존 알고리즘을 능가하는 최첨단 성능을 입증하며, 이는 임상에서의 자동화된 망막 혈관 분석을 위한 잠재력을 강조합니다.



### Activating Associative Disease-Aware Vision Token Memory for LLM-Based X-ray Report Generation (https://arxiv.org/abs/2501.03458)
Comments:
          In Peer Review

- **What's New**: 이번 논문에서는 의료 보고서 생성을 위한 새로운 협동 기억 향상 모델인 AM-MRG(Associative Memory Augmented X-ray Medical Report Generation)를 제안합니다. 이 모델은 X-ray 이미지를 분석하여 중요한 질병 정보를 정확하게 인식하고, 과거 보고서 정보를 연관시켜 현재 보고서 작성에 도움을 줍니다. 기존의 대형 언어 모델(LLM)을 활용하며, 보다 정교한 시각 정보 분석을 통해 고품질의 의료 보고서를 생성할 수 있습니다.

- **Technical Details**: AM-MRG는 두 단계로 이루어져 있습니다. 첫 번째 단계에서 Swin Transformer 네트워크를 사용하여 X-ray 이미지의 시각 특징을 추출하고, Q-Former를 통해 질병 검색 쿼리와 함께 비주얼 특징을 증강합니다. 두 번째 단계에서는 모던 호프필드 네트워크(Modern Hopfield Network)를 활용하여 질병 인식과 관련된 비주얼 토큰을 지식베이스로 삼고, 보고서 기억을 검색하여 LLM을 기반으로 한 보고서 생성을 진행합니다.

- **Performance Highlights**: 제안된 AM-MRG는 IU X-ray, MIMIC-CXR, Chexpert Plus의 여러 벤치마크 데이터셋에서 최신 성능을 달성하였습니다. 실험 결과, 기존 모델에 비해 질병 정보를 보다 명확하게 설명하는 보고서를 생성함으로써 의료 현장에서의 유용성을 증대시켰습니다. 따라서, 이 연구는 X-ray 기반의 의료 보고서 생성 분야에서 중요한 발전을 이룬 것으로 평가됩니다.



### A Self-supervised Diffusion Bridge for MRI Reconstruction (https://arxiv.org/abs/2501.03430)
- **What's New**: 본 논문은 SelfDB라는 새로운 방법을 제안하여 고화질 레퍼런스 이미지 없이 노이즈 측정값을 기반으로 확산 브리지(diffusion bridge) 모델을 학습할 수 있도록 합니다. 기존의 확산 브리지는 고화질 이미지가 필요하기 때문에 활용도가 제한적이었습니다. SelfDB는 주어진 측정값을 기반으로 추가적으로 두 번 서브샘플링하여 노이즈 경량화 과정을 역전시키는 신경망을 학습함으로써 이 문제를 해결합니다.

- **Technical Details**: SelfDB에서는 노이즈 측정값만을 사용하여 새로운 확산 과정을 생성하고, 이를 기반으로 딥 뉴럴 네트워크(DNN)를 훈련합니다. 연산 중 측정값을 두 번 더 축소하여 추가적인 데이터 효율성을 얻고, 고품질 레퍼런스 이미지 없이도 훈련할 수 있게 됩니다. 이는 기존의 Denoising Diffusion Models(DDMs)에서 요구하는 고품질 이미지와 차별화되는 점입니다.

- **Performance Highlights**: 논문에서는 SelfDB의 성능을 압축 센싱 MRI(compressed sensing MRI)에서 검증했습니다. 결과적으로, SelfDB는 기존의 자기 지도 학습 방식으로 훈련된 DDMs보다 더 뛰어난 성능을 보였으며, 이는 보다 적은 추론 단계로도 고품질 이미지를 생성할 수 있음을 시사합니다.



### Quantum Feature-Empowered Deep Classification for Fast Mangrove Mapping (https://arxiv.org/abs/2501.03360)
Comments:
          This work has been accepted by IEEE Transactions on Geoscience and Remote Sensing (TGRS)

- **What's New**: 최근의 연구에서는 CNN을 사용하여 망그로브 분포를 보다 효과적으로 분류할 수 있음을 보여주었습니다. 본 논문은 이러한 CNN의 성능을 더욱 향상시키기 위해 양자 특징(quantum features)을 도입하는 새로운 접근 방식을 제시합니다. 양자 신경망(quantum neural network, QNN)을 활용하여 전통적인 CNN의 한계를 극복하고 보다 정교한 분류 결과를 도출할 수 있습니다.

- **Technical Details**: 본 연구에서는 양자 특성을 활용한 새로운 망그로브 매핑(MM) 알고리즘인 QEDNet을 설계했습니다. 이 네트워크는 기존 CNN 구조와 달리 양자 뉴런을 포함하여 양자 정보를 추출하고, 이를 기존 특성과 융합하여 최종 클래시피케이션을 수행합니다. 또한 QEDNet은 가벼운 구조를 가지고 있으며, 파라미터 증가 없이 CNN과 QNN의 협력을 통해 성능 개선을 이루어냅니다.

- **Performance Highlights**: QEDNet의 성능은 포괄적인 실험을 통해 입증될 예정이며, 기존의 망그로브 매핑 알고리즘과 비교하여 탁월한 결과를 달성할 것으로 기대됩니다. 특히 양자 정보를 활용함으로써, 복잡한 환경에서도 높은 분류 정확도를 유지할 수 있는 가능성을 가지고 있습니다. 이는 망그로브 및 다른 해양 생태계의 모니터링에 있어 중요한 기여를 할 것입니다.



### FTA-FTL: A Fine-Tuned Aggregation Federated Transfer Learning Scheme for Lithology Microscopic Image Classification (https://arxiv.org/abs/2501.03349)
- **What's New**: 본 연구는 석유 저장소의 특성을 파악하기 위해 리소지(특정 암석의 종류) 마이크로 이미지 분류를 대상으로 하며, 이를 위한 혁신적인 Federated Learning (FL) 접근방식을 제안합니다. 특히 데이터의 민감성 문제를 해결하면서도 높은 정확도의 중앙 모델을 훈련할 수 있는 방안을 모색하였습니다. 또한, Transfer Learning과 데이터 증강 기법을 활용하여 소규모 데이터셋에서 시작하는 두 단계의 연구를 진행하였습니다.

- **Technical Details**: 연구의 첫 번째 단계에서는 Transfer Learning을 통해 소규모 리소지 마이크로 이미지 분류를 수행하며 여러 가지 사전 학습된 Deep Learning 모델 아키텍처를 비교하였습니다. 두 번째 단계에서는 분산된 엣지 서버에서 민감한 데이터를 전송하지 않고도 효과적인 모델을 학습할 수 있는 Federated Transfer Learning (FTL) 구성을 하였으며, Fine-Tuned Aggregation 전략(FTA-FTL)을 제안하였습니다. 이러한 방법은 정확도, f1 점수, 정밀도, 특이도, 민감도(재현율), 혼동 행렬과 같은 여러 메트릭스를 바탕으로 평가되었습니다.

- **Performance Highlights**: 제안된 FTA-FTL 알고리즘은 중앙 집중형 구현 방식에서 달성한 결과와 거의 동일한 결과를 얻을 수 있는 것으로 나타났습니다. 실험 결과는 제안된 접근 방식의 효율성을 확인하며, 복잡한 실험적 연구에도 불구하고 좋은 일관성을 보였습니다. 이로 인해 석유 개발 분야에서의 리소지 분류 작업에 있어 유망한 방향성을 제시합니다.



### OpenLKA: an open dataset of lane keeping assist from market autonomous vehicles (https://arxiv.org/abs/2501.03287)
- **What's New**: 이 논문에서는 최신 자동차 모델에서 필수 기능으로 자리잡고 있는 Lane Keeping Assist (LKA) 시스템의 운영 특성과 안전 성능을 실제 테스트를 통해 분석합니다. 기존의 연구에서는 LKA 시스템에 대한 실증적 데이터가 부족했으나, 이 연구는 미국의 주요 자동차 제조사에서 제공하는 LKA 시스템을 테스트하여 포괄적인 데이터를 수집하였습니다. 고품질 전방 카메라를 이용하여 다양한 도로 및 기상 조건에서 LKA의 성능을 평가하였습니다.

- **Technical Details**: 테스트 과정에서는 Controller Area Network (CAN) 메시지와 함께 LKA 속성 및 비디오, 인식, 측면 궤적 데이터를 수집하였습니다. 다양한 도전 과제가 있는 환경에서 시험이 이루어졌으며, 특히 도로의 기하학적 복잡성, 불안정한 날씨, 저하된 차선 표시 등이 포함되었습니다. 추가적으로, 비전 언어 모델 (VLM)은 촬영된 비디오에 날씨, 조명, 교통 상황 등을 주석으로 달아 술어를 강화했습니다.

- **Performance Highlights**: LKA의 주요 발견은 다음과 같습니다: (i) LKA는 미약한 차선 표시와 낮은 도로 대비에 취약하다; (ii) 차선 전환 시 문제를 겪어 의도치 않은 이탈이나 시스템의 비활성화가 발생한다; (iii) 조타 토크 제한으로 인해 급격한 회전 시 잦은 이탈이 발생하며, 이는 안전 문제를 야기한다; (iv) LKA 시스템은 일관된 차선 중앙 유지를 하지만 좁은 곡선이나 대형 차량 근처에서의 적응성이 부족하다. 이러한 데이터 세트는 인프라 계획 및 자율 주행 기술 개발에 유용하게 활용될 수 있습니다.



### Video-of-Thought: Step-by-Step Video Reasoning from Perception to Cognition (https://arxiv.org/abs/2501.03230)
Comments:
          Accepted by ICML 2024

- **What's New**: 이 논문은 복잡한 비디오에 대한 이해와 추론의 한계를 극복하기 위해 새로운 솔루션을 제시합니다. MotionEpic이라는 새로운 비디오 멀티모달 대형 언어 모델(Multimodal Large Language Model)을 소개하여, 비디오의 픽셀 수준 정밀한 공간-시간(Spatial-Temporal) 처리를 가능하게 합니다. 또한, Video-of-Thought (VoT)라는 추론 프레임워크를 개발하여, 복잡한 비디오 문제를 단순한 하위 문제로 나누어 단계별로 해결할 수 있도록 설계하였습니다.

- **Technical Details**: MotionEpic은 입력 비디오와 공간-시간 장면 그래프(Spatial-Temporal Scene Graph, STSG) 표현을 통합하여 비디오에 대한 정밀한 픽셀 수준의 공간-시간 접지를 달성합니다. 이 모델은 입력 비디오를 인코딩하고 이해하며 STSG를 생성하는 기능을 지원합니다. VoT 프레임워크는 비디오와 질문을 기반으로 가능한 대상(target)를 식별하고, 이들을 시간적으로 추적하여 행동 동역학을 이해하는 등 계층적인 추론 과정을 통해 복잡한 비디오 문제를 해결합니다.

- **Performance Highlights**: 우리는 비디오 질문-응답(Video Question Answering, QA) 벤치마크를 통해 MotionEpic과 VoT 프레임워크의 성능을 평가하였으며, 기존의 최첨단 성능을 불확실하게 개선하였습니다. 특히, 8개의 복잡한 비디오 QA 벤치마크에서 매우 명확한 마진을 통해 성능을 끌어올리며 새로운 상태(State-of-the-Art)를 수립하였습니다. 본 연구는 복잡한 비디오 이해 및 추론에서 지식과 인지력을 증진시키는 데 크게 기여할 것으로 기대됩니다.



### Gaussian Masked Autoencoders (https://arxiv.org/abs/2501.03229)
- **What's New**: 이 논문은 Masked Autoencoders (MAE)와 Gaussian Splatting을 결합한 새로운 접근법을 소개합니다. 특히, Gaussian Masked Autoencoder (GMAE)를 도입하여 의미론적 추상화(semantic abstraction)와 공간적 이해(spatial understanding)를 동시에 학습하도록 설계했습니다. GMAE는 MAE와 유사하게 픽셀 공간에서 이미지를 전체적으로 재구성할 뿐만 아니라, 중간 단계로 3D Gaussian 기반 표현을 사용하여 이미지를 렌더링하는 기능을 포함하고 있습니다.

- **Technical Details**: GMAE는 3D Gaussians를 중간 이미지 표현으로 활용하여 의미적 및 공간적 이해를 가능하게 합니다. 이 모델은 가우시안 표현의 동적 특성을 통해 자연 세계의 구조를 학습하고, 이미지의 깊이 불연속성을 토대로 figure-ground segmentation, 레이어링, 모서리 검출(edge detection)과 같은 다양한 제로샷 학습 능력을 구현합니다. GMAE는 MAE와 비교했을 때 비슷한 학습 성능을 보여주는 동시에, 3D Gaussian을 통한 렌더링 덕분에 계산 효율성 또한 향상됩니다.

- **Performance Highlights**: GMAE는 MAE와 유사한 이미지 분류 및 객체 탐지 작업에서 높은 성능을 보여주며, 사용하는 가우시안의 수가 늘어날수록 표현 품질이 향상됩니다. 또한, GMAE는 표준 MAE 훈련에 비해 매우 적은 오버헤드를 추가하며, 실제 훈련 시간도 거의 비슷하게 유지됩니다. 이러한 성능은 GMAE가 중간 단계 표현을 활용하는 응용 프로그램에서 더욱 유리할 수 있음을 시사합니다.



### Automated Generation of Challenging Multiple-Choice Questions for Vision Language Model Evaluation (https://arxiv.org/abs/2501.03225)
Comments:
          Project page: this https URL

- **What's New**: 비전 언어 모델(Vision Language Models, VLMs)의 빠른 발전을 반영하여, 본 논문에서는 AutoConverter라는 새로운 프레임워크를 도입하여 열린 질문을 객관적인 선택형 질문으로 변환합니다. 이 접근은 문제 생성 과정을 줄이며 평가의 일관성을 높이는 데 기여합니다. AutoConverter는 자동으로 정확하고 도전적인 선택형 질문을 생성할 수 있는 다중 에이전트 시스템으로 설계되었습니다.

- **Technical Details**: AutoConverter는 다양한 에이전트가 협력하여 열린 질문을 선택형으로 변환하는 방식으로 작동합니다. 특히, 정확성을 보장하기 위해 질문의 정확성을 평가하는 에이전트가 있어, 생성된 선택지의 적합성을 검증합니다. 이를 통해 20개의 기존 VQA 데이터셋에서 9,018개의 질문으로 구성된 VMCBench라는 새로운 벤치마크를 생성하였습니다.

- **Performance Highlights**: 33개의 최첨단 VLM을 VMCBench에서 평가한 결과, AutoConverter가 생성한 질문이 인간이 제작한 질문에 비해 유사하거나 더 낮은 정확도를 보이면서도 높은 도전성을 유지함을 보여주었습니다. 이러한 결과는 AutoConverter의 다용도성을 입증하며, 교육 및 기존 선택형 데이터셋의 개선에도 활용할 수 있는 가능성을 제공합니다.



### Rate-My-LoRA: Efficient and Adaptive Federated Model Tuning for Cardiac MRI Segmentation (https://arxiv.org/abs/2501.03223)
Comments:
          Accepted in ISBI 2025

- **What's New**: 이번 연구에서는 심장 이미지 세분화를 위한 새로운 효율적이고 적응 가능한 연합 학습(federated learning) 방법을 제안합니다. 이 방법은 대규모 데이터 세트에 대한 모델 성능을 개선하면서도 통신 대역폭 요구를 줄이는 데 초점을 맞추고 있습니다. 기존의 연합 학습 알고리즘이 가진 데이터 이질성(data heterogeneity) 문제를 해결하기 위해, LoRA(low-rank adaptation)를 활용하여 모델 가중치 업데이트를 정규화하고 통신 오버헤드를 감소시킵니다.

- **Technical Details**: 연구의 핵심은 Rate-My-LoRA라는 새로운 집합 기법을 도입하여 서로 다른 클라이언트에서의 데이터 이질성을 감소시키는 데 있습니다. 이는 각 클라이언트의 검증 정확도(validation accuracy)를 비교하여 집계 가중치를 적응적으로 패널티(penalize)하는 방식으로, 결과적으로 일반화 성능을 향상시키고 빠른 로컬 적응(local adaptation)을 가능하게 합니다. 성능 평가에서 이 방법은 LoRA 기반의 다른 연합 학습 접근법보다 우수성을 보였습니다.

- **Performance Highlights**: 제안한 방식은 공적인 심장 MRI 데이터셋을 통해 검증된 바, 다른 LoRA 기반 연합 학습 접근법에 비해 현저히 향상된 결과를 보여주었습니다. 특히, 통신 라운드당 통신 대역폭 사용량을 최대 94%까지 감소시켰으며, 이는 자원 제약이 있는 병원에서의 참여를 용이하게 합니다. 이러한 성과는 심장 개입을 위한 더 나은 모델 개발에 기여할 것으로 기대됩니다.



### RW-Net: Enhancing Few-Shot Point Cloud Classification with a Wavelet Transform Projection-based Network (https://arxiv.org/abs/2501.03221)
Comments:
          11 pages, 5 figures, 9 tables

- **What's New**: 이번 논문은 3D 물체 분류에 대한 새로운 접근법인 RW-Net을 제안합니다. RW-Net은 Rate-Distortion Explanation (RDE)와 웨이브릿 변환을 통합하여 제한된 레이블 데이터에서도 효과적으로 학습할 수 있도록 돕습니다. 이러한 접근 방식은 3D 물체의 중요한 특징을 효과적으로 추출하여 학습 효율성을 높이고 대규모 레이블 데이터의 의존도를 줄이는 데 기여합니다.

- **Technical Details**: RW-Net은 RDE를 통해 입력 데이터의 필수 특징을 보존하면서 불필요한 세부 사항을 최소화하는 희박하고 효율적인 표현을 학습합니다. 또한, 웨이브릿 변환을 2D 백본 네트워크에 도입하여 입력의 저주파 성분에 집중하여 잡음의 영향을 줄이고 모델의 과적합(overfitting) 문제를 완화합니다. 이를 통해 다양한 작업과 도메인에서 학습된 표현의 내구성을 향상시킵니다.

- **Performance Highlights**: 실험을 통해 RW-Net은 ModelNet40, ModelNet40-C, ScanObjectNN 세 가지 데이터셋에서 기존 방법들에 비해 우수한 성능을 기록하였습니다. 특히, RW-Net은 few-shot 학습 환경에서 높은 일반화 능력과 강인성을 보여줍니다. 이 결과는 RW-Net이 3D 포인트 클라우드 분류에서 최첨단 성능을 달성할 수 있음을 입증합니다.



### ProTracker: Probabilistic Integration for Robust and Accurate Point Tracking (https://arxiv.org/abs/2501.03220)
Comments:
          Project page: this https URL

- **What's New**: 본 논문에서 우리는 ProTracker라는 새로운 프레임워크를 제안합니다. ProTracker는 비디오에서 임의의 포인트에 대한 견고하고 정확한 장기 밀집 추적(long-term dense tracking)을 가능하게 합니다. 이 방법의 핵심 아이디어는 시뮬레이션된 다수의 예측을 확률적 통합(probabilistic integration)하여 짧은 기간 및 긴 기간 추적을 개선하는 것입니다. 특히, 우리가 제안한 모델은 unsupervised 및 self-supervised 접근 방식에서 최첨단 성능을 달성하며, 몇 가지 벤치마크에서 supervised 방법보다도 우수한 성능을 보여줍니다.

- **Technical Details**: ProTracker는 Kalman Filter에서 영감을 받은 양방향 확률적 통합(bidirectional Probabilistic Integration) 기법을 사용하여, 광학 흐름(optical flow) 예측과 장기 특징 상응(feature correspondence)을 통합합니다. 우리는 불확실한 초기 예측을 제거하고 남아있는 거친 광학 흐름 예측을 가우시안 분포(Gaussian distribution)로 처리하여 가장 가능성 높은 포인트 예측을 도출합니다. 이 방법은 포워드 및 백워드 방향으로 통합이 수행되어 매우 정확하고 견고한 흐름 추정(flow estimation)을 가능하게 합니다.

- **Performance Highlights**: 다양한 TAP-Vid 벤치마크에서 ProTracker의 성과를 평가한 결과, self-supervised 또는 비지도 방식 중에서 모든 지표에서 이전 방법들을 초월했습니다. 데이터 기반 방법에 비해서도 경쟁력을 발휘하며, 모든 접근 방식 중에서 위치 추정(position estimation)에서 가장 높은 정확성을 나타냅니다. 본 연구는 self-supervised 및 unsupervised 접근 방식 사이에서 최첨단 성능을 달성하여, 데이터 기반 방법과도 경쟁할 수 있는 결과를 보여줍니다.



### Dispider: Enabling Video LLMs with Active Real-Time Interaction via Disentangled Perception, Decision, and Reaction (https://arxiv.org/abs/2501.03218)
- **What's New**: Dispider는 비디오 LLM(대형 언어 모델)과의 실시간 상호작용을 위한 새로운 시스템으로, 비디오 처리를 동시에 진행하면서 사용자와의 인터랙션을 제공합니다. 이 시스템은 Perception(지각), Decision(결정), Reaction(반응) 세 가지 기능을 비동기적 모듈로 분리하여 효율성을 극대화합니다. Dispider의 비동기 설계는 사용자에게 신속하고 정확한 응답을 제공하며, 긴 비디오 스트림에서도 실시간 성능을 유지할 수 있게 합니다.

- **Technical Details**: Dispider는 장면 기반( scene-based) 지각 모듈, 실시간 반응 결정 모듈, 비동기 상호작용 모듈로 구성됩니다. 장면 기반 지각 모듈은 비디오 스트림을 동적으로 분할하고, 이전의 결정 토큰 및 비주얼 정보를 통합하여 상호작용의 트리거를 결정합니다. 이 과정에서 비디오 분석과 응답 생성을 동시에 수행하여 시스템의 실시간 성능을 보장합니다.

- **Performance Highlights**: Dispider의 성능은 StreamingBench에서 평가되었으며, VideoLLM-online과 비교하여 시간 기반 정합성, 선제적 반응 생성 및 다단계 추론에서 월등한 성과를 보였습니다. 또한 오프라인 비디오 LLM들에 대해서도 긴 비디오 벤치마크에서 뛰어난 성능을 입증하였습니다. 특히, 시간적 추론 및 다양한 비디오 길이 처리에서 탁월한 능력을 발휘하였습니다.



### MObI: Multimodal Object Inpainting Using Diffusion Models (https://arxiv.org/abs/2501.03173)
Comments:
          8 pages

- **What's New**: 본 논문에서는 MObI라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 확산 모델(diffusion model)을 활용하여 카메라와 LiDAR 데이터를 활용한 현실적이고 제어 가능한 멀티모달(object inpainting) 장면 생성을 가능하게 합니다. MObI는 단일 RGB 이미지를 기반으로 하여, 사전 정의된 경계 상자(bounding box)에 따라 객체를 기존 장면에 매끄럽게 삽입할 수 있습니다.

- **Technical Details**: MObI는 Paint-by-Example(PbE)와 같은 참조 기반 이미지 인페인팅 방법을 확장하여, 3D 경계 상자 조건을 포함한 카메라 및 LiDAR 입력을 동시에 생성합니다. 다중 단계 파이프라인을 사용하는 기존 방법과 달리, 우리의 접근 방식에서는 end-to-end 방식으로 카메라와 LiDAR를 공동으로 생성하는 방법론을 제안합니다. 이 접근 방식을 통해, 우리는 생성된 장면의 정확한 공간적 배치와 현실적인 비율을 보장합니다.

- **Performance Highlights**: 이 방법론은 테스트 중인 인식 모델의 성능을 비약적으로 향상시킬 수 있는 추가적인 장점을 제공합니다. MObI를 활용하면 새로운 객체를 멀티모달 장면에 유연하게 삽입할 수 있으며, 실제적이고 제어 가능한 반사실적인(counterfactual) 운전 장면을 생성하는 데 효과적임을 입증합니다. 이를 통해, 안전-critical한 자율 주행 시스템의 테스트 환경을 획기적으로 개선할 수 있습니다.



### Segment Anything Model for Zero-shot Single Particle Tracking in Liquid Phase Transmission Electron Microscopy (https://arxiv.org/abs/2501.03153)
- **What's New**: 이번 연구에서는 Segment Anything Model 2 (SAM 2)를 활용하여 LPTEM 비디오에서 입자를 효과적으로 세그먼트하고 추적하는 새로운 방법론인 SAM4EM을 소개합니다. SAM 2는 미세한 조정 없이도 모든 조건에서 작동할 수 있는 비디오 세그멘테이션의 최전선 모델로, 고유한 spatiotemporal 해상도를 제공합니다. 이를 통해 LPTEM 기술을 단일 입자 추적(single particle tracking) 도구로 활용할 수 있는 가능성이 열리게 됩니다.

- **Technical Details**: LPTEM은 명확한 시각화를 제공하는 새로운 형태의 전자 현미경 기술로, 전자빔이 실리콘 나이트라이드(SiNx) 막을 통과하면서 액체 샘플을 hermetically 밀봉합니다. 이 연구에서는 기존의 LPTEM 비디오 세그멘테이션의 한계를 극복하기 위해 SAM 2를 도입하여 비디오 세그멘테이션 문제에 대한 새로운 접근 방법을 제안합니다. SAM4EM은 사용자의 프롬프트를 통한 상호작용적 세그멘테이션, 입자 추적, 통계 분석을 통합하여 완전한 LPTEM 분석 프레임워크를 제공합니다.

- **Performance Highlights**: SAM4EM은 기존의 최신 방법론 대비 세그멘테이션과 분석 정확도가 거의 50배 이상 향상되었습니다. 실제 LPTEM 비디오와 시뮬레이션된 비디오에 대한 실험을 통해 SAM4EM의 유효성을 입증했습니다. 추가로, SAM4EM을 통해 획득한 spatiotemporal 궤적은 여러 입자의 복잡한 운동을 효과적으로 분석하고 관련 통계 분석을 수행할 수 있는 능력을 보여줍니다.



### Geometry Restoration and Dewarping of Camera-Captured Document Images (https://arxiv.org/abs/2501.03145)
Comments:
          28 pages, 16 figures

- **What's New**: 이번 연구는 카메라로 촬영한 종이 문서의 디지털 이미지에서 위상 복원(Topology Restoration)을 위한 방법을 개발하는 데 초점을 맞추었습니다. 알고리즘을 통해 문서의 윤곽을 감지하고(segmentation), 기하학적으로 복원하는 데 필요한 다양한 기술을 적용했습니다. 전통적인 컴퓨터 비전(computer vision) 기법을 사용하여 문서의 복원 과정을 더욱 효율적이고 빠르게 진행할 수 있음을 보여주었습니다.

- **Technical Details**: 이 방법론은 딥 러닝(deep learning)을 활용하여 문서 윤곽을 감지하고, 그 후에는 컴퓨터 비전(computer vision) 기술을 통해 2D 그리드를 생성하고 비선형 왜곡을 교정하는 데 중점을 두고 있습니다. 큐빅 다항식 보간(cubic polynomial interpolation)을 사용해 이미지를 재매핑(remapping)하는 과정을 포함시켜 문서의 구조를 효과적으로 복원할 수 있었습니다. 또한, 자동 문서 디워핑(dewarping) 및 복원을 위한 새로운 파이프라인(pipeline)을 개발하였습니다.

- **Performance Highlights**: 실험 결과, 개발된 방법론은 기존의 벤치마크(benchmark)들, 특히 모바일 앱 및 RectiNet, DocGeoNet, DocTr++와 같은 인기 있는 딥러닝 솔루션들과 비교했을 때 시각적으로나 문서 가독성 측면에서 우수함을 입증하였습니다. 문서 이미지의 상태를 개선하고 OCR(Optical Character Recognition) 시스템의 효율성을 향상시키기 위한 길을 열어 주었습니다. 이 연구는 종이 문서의 고품질 디지털 복사본을 생성하는데 기여할 것으로 기대됩니다.



### Normalizing Batch Normalization for Long-Tailed Recognition (https://arxiv.org/abs/2501.03122)
- **What's New**: 본 논문에서는 데이터의 레벨이나 분류기 레벨에서의 정형적인 접근 방식을 넘어, 희귀 클래스(rar class)에 대해 잘 작동하지 않는 특징(feature) 편향(bias)을 다루는 새로운 방법을 제안합니다. 이는 특히 Batch Normalization (BN) 레이어의 파라미터를 정규화(normalizing)함으로써 달성됩니다. 이러한 방식은 희귀 클래스에 대한 분별력을 높이는 데 도움을 주며, 실험을 통해 성능의 유의미한 향상을 보여줍니다.

- **Technical Details**: 논문에서 제안하는 방법은 BN 레이어의 Weight 및 Bias 파라미터를 벡터(vector)로 표현하고 이를 단위 벡터로 정규화한 다음, 스칼라 학습 가능 파라미터로 곱하는 것입니다. 이를 통해 BN 레이어의 파라미터의 방향과 크기를 분리(decoupling)하여 더 균형 잡힌 분포를 만들 수 있습니다. 이러한 방식은 특징의 강도를 고르게 만들어 희귀 클래스에 대한 정확도를 개선하는 데 기여합니다.

- **Performance Highlights**: 다양한 긴 꼬리 인식(long-tailed recognition) 벤치마크(CIFAR-10/100-LT, ImageNet-LT, iNaturalist 2018)에서 우리의 방법이 이전의 최첨단(상태-최고) 방법에 비해 현저히 우수한 성능을 나타냈습니다. 희귀 클래스에 대한 정확성을 높이는 데 성공했으며, 이는 최신 기술 및 수정 방법들과 함께 사용할 수 있는 플러그 앤 플레이 방식임을 입증했습니다.



### CAT: Content-Adaptive Image Tokenization (https://arxiv.org/abs/2501.03120)
- **What's New**: 이번 논문에서는 기존 이미지 토크나이저의 한계를 극복하기 위해 Content-Adaptive Tokenizer (CAT)를 제안합니다. CAT는 이미지의 복잡성에 따라 동적으로 표현 용량을 조정하여 복잡한 이미지는 더 많은 토큰을 사용하고, 간단한 이미지는 적은 토큰으로 인코딩합니다. 이 연구는 이미지 설명을 통해 최적의 압축 비율을 예측하는 캡션 기반 평가 시스템을 포함합니다.

- **Technical Details**: CAT는 대규모 언어 모델(LLM)을 활용하여 이미지의 복잡성을 평가하고, 이를 통해 동적인 인코딩을 수행하는 계층적인 변이 오토인코더(VAE) 아키텍처를 설계합니다. 이 시스템은 다양한 데이터셋에서 높은 재구성 성능을 보여주며, 12%에서 39%까지의 향상을 기록했습니다. 또한 CAT는 Latent Diffusion Transformers (DiTs)를 학습시키는 데도 활용되어, 이미지 데이터셋의 고수준 및 저수준 정보를 효과적으로 캡처합니다.

- **Performance Highlights**: CAT는 FID 성능과 추론 처리량을 동시에 개선하여, 동일한 PLFS로 훈련된 고정 비율 기준선보다 18.5% 더 나은 성능을 보여줍니다. 특히, 복잡한 이미지에서의 구조적 품질이 크게 향상되며, 사용자 지정된 복잡도 수준에 따라 제어 가능한 이미지 생성을 지원합니다. CAT는 이처럼 품질과 속도의 향상을 통해 이미지 모델링의 효율적이고 효과적인 발전을 이끌 것으로 기대됩니다.



### MVP: Multimodal Emotion Recognition based on Video and Physiological Signals (https://arxiv.org/abs/2501.03103)
Comments:
          Preprint. Final paper accepted at Affective Behavior Analysis in-the-Wild (ABAW) at IEEE/CVF European Conference on Computer Vision (ECCV), Milan, September, 2024. 17 pages

- **What's New**: 이번 연구에서는 Multimodal for Video and Physio (MVP) 아키텍처를 제안하여 비디오 데이터와 생리적 신호를 결합하는 새로운 접근 방식을 소개하고 있습니다. 기존의 감정 인식 연구는 주로 비디오, 오디오 및 언어 신호의 조합에 집중하였으나, MVP는 비디오와 생리적 데이터를 동시에 처리할 수 있는 가능성을 보여줍니다. 특히, 이 모델은 긴 입력 시퀀스(1-2분)를 다룰 수 있는 주목(attention) 메커니즘을 활용하여, 이전 방식보다 향상된 성능을 발휘합니다.

- **Technical Details**: MVP 아키텍처는 비디오 데이터와 생리적 데이터를 처리하기 위해 고급 기술을 사용하고 있습니다. 비디오 데이터는 VideoMAE 알고리즘을 이용해 특징을 추출하고, 생리적 데이터는 1D-CNN 및 unimodal transformer를 통해 입력됩니다. 특히, 생리적 데이터는 전체 시퀀스를 입력할 수 있도록 개선되어 주목(attention) 메커니즘이 단기 및 장기 의존성을 효과적으로 찾아낼 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, MVP 모델은 얼굴 비디오, EDA(elctrodermal activity), 그리고 ECG(심전도)/PPG(광용적맥파) 기반의 감정 인식에서 기존 방식보다 우수한 성능을 기록하였습니다. 또한, 다양한 구성 요소를 평가한 결과, MVP가 전통적인 방법에 비해 감정 인식의 정확성을 높이는 데 기여함을 보여주고 있습니다. 이를 통해 감정 인식 분야에서의 다변량 접근 방식의 중요성을 다시 한번 강조합니다.



### A Novel Structure-Agnostic Multi-Objective Approach for Weight-Sharing Compression in Deep Neural Networks (https://arxiv.org/abs/2501.03095)
Comments:
          16 pages, 9 figures, submitted to IEEE Transactions on Neural Networks and Learning Systems

- **What's New**: 이 논문에서는 신경망 아키텍처나 데이터셋에 의존하지 않는 다목적 진화 알고리즘(multi-objective evolutionary algorithm, MOEA)에 기반한 새로운 압축 프레임워크를 제안합니다. 일반 크기의 빈을 사용하여 네트워크 가중치를 단일 코드북(lookup table)으로 양자화하여 효율적인 가중치 표현을 가능하게 합니다. 또한, 인접한 빈을 결합하여 성능 저하 없이 압축 비율을 증가시키기 위한 반복적 병합 기법을 적용합니다.

- **Technical Details**: 제안된 방법은 레이어(layer)나 모델에 독립적이며, 이는 어떤 레이어의 가중치도 클러스터에서 혼합될 수 있음을 의미합니다. 또한, 이 작업에서 사용된 균일 양자화 방법은 복잡도가 $O(N)$인 반면, k-평균(k-means)과 같은 비균일 양자화 방법은 $O(Nkt)$의 복잡도를 가집니다. 중심 클러스터를 공유 가중치 값으로 사용하여 재학습 없이 계산 비용을 줄이는 장점도 있습니다.

- **Performance Highlights**: 실험 결과, CIFAR-10에서 신경망 메모리를 $13.72 	ext{~} 14.98 	imes$ 줄일 수 있었고, CIFAR-100에서는 $11.61 	ext{~} 12.99 	imes$, ImageNet에서는 $7.44 	ext{~} 8.58 	imes$의 감소를 달성하였습니다. 이는 제안된 딥 신경망 압축 프레임워크의 효율성을 잘 보여줍니다.



### AIF-SFDA: Autonomous Information Filter-driven Source-Free Domain Adaptation for Medical Image Segmentation (https://arxiv.org/abs/2501.03074)
Comments:
          9 pages total (7 pages main text, 2 pages references), 6 figures, accepted by AAAI 2025

- **What's New**: 이 논문에서는 Autonomous Information Filter-driven Source-free Domain Adaptation (AIF-SFDA) 알고리즘을 제안하여 의료 이미지 분할 작업에서 도메인 변화 정보를 (DVI) 도메인 불변 정보 (DII)와 자율적으로 분리합니다. 이를 통해 기존의 방법들이 데이터를 수집하고 접근하는 데 어려움을 겪는 의료 분야에서의 실제 적용을 개선하고자 합니다. 이 알고리즘은 정보 병목 효과 (IB)와 자기 지도 학습 (SS)을 통합해 학습 가능한 주파수 필터를 최적화하여, 오직 목표 데이터만 사용하면서 도메인 변화를 극복할 수 있게 합니다.

- **Technical Details**: AIF-SFDA는 주파수 기반 학습 가능한 정보 필터를 활용하여 DVI와 DII를 자율적으로 분리합니다. 정보 병목 (IB) 이론은 필터 내 정보 흐름을 조절해 중복된 DVI를 줄이고, 자기 지도 학습 (SS)은 특정 작업 및 이미지 양식에 맞춰 DII를 보존합니다. 이 필터는 실제로 명시적 라벨 없이도 목표 데이터에서만 동작하며, 이는 의료 데이터와 같이 개인 정보 보호가 중요한 분야에서 큰 장점으로 작용합니다.

- **Performance Highlights**: 다양한 의료 이미지 모달리티 및 분할 작업을 포함한 일련의 실험을 통해 AIF-SFDA의 유효성을 평가했습니다. 본 논문은 기존의 최첨단 알고리즘과 비교하여 AIF-SFDA의 장점을 강조하며, 자율 정보 필터가 도메인 전이에서의 효율성을 어떻게 개선하는지를 입증했습니다. 제공된 코드는 연구자들이 AIF-SFDA를 쉽게 구현하고 활용할 수 있도록 돕습니다.



### Through-The-Mask: Mask-based Motion Trajectories for Image-to-Video Generation (https://arxiv.org/abs/2501.03059)
- **What's New**: 이번 논문에서는 정적 이미지를 기반으로 사실적인 비디오 시퀀스를 생성하는 Image-to-Video (I2V) 생성 기술을 다룹니다. 기존의 방법들은 멀티 오브젝트(multi-object) 환경에서 객체의 정확한 움직임을 생성하는 데 어려움을 겪고 있으며, 이러한 한계를 극복하기 위해 두 단계의 구성적 프레임워크를 제안합니다. 첫 번째 단계에서는 명시적인 중간 표현을 생성하고, 두 번째 단계에서는 이 표현을 바탕으로 비디오를 생성합니다.

- **Technical Details**: 제안된 방법은 mask 기반의 motion trajectory를 중간 표현으로 사용하며, 이는 객체의 의미 정보와 움직임을 포착합니다. 우리는 각 객체에 대한 객관적인 주의(object-level attention) 목표를 활용하여 두 번째 단계에서 학습된 표현을 통합합니다. 특히, 공간적(masked cross attention) 및 시공간적(masked spatio-temporal self-attention) 자기 주의(self-attention) 목표를 사용하여 프레임 간 일관성을 보장합니다.

- **Performance Highlights**: 본 연구는 다수의 오브젝트와 높은 동작이 포함된 도전적인 벤치마크에서 방법의 효과를 검증하였으며, 시간적 일관성, 움직임의 사실성, 텍스트 프롬프트 충실도에 대한 최신 기술 수준(state-of-the-art) 결과를 도출했습니다. 또한, 단일 및 다중 객체 I2V 생성을 위한 새로운 벤치마크를 도입하였고, 이 벤치마크에서도 우리의 방법이 우수한 성능을 보여줍니다.



### TransPixar: Advancing Text-to-Video Generation with Transparency (https://arxiv.org/abs/2501.03006)
Comments:
          Project page: this https URL

- **What's New**: 이번 연구에서는 RGBA 비디오 생성 문제를 해결하기 위한 새로운 방법인 TransPixar를 소개합니다. 이 방법은 기존의 RGB 비디오 모델을 활용하여 알파 채널을 생성하면서도 RGB 기능을 유지할 수 있도록 설계되었습니다. 특히, TransPixar는 Diffusion Transformer(DiT) 아키텍처를 이용하여 알파 전용 토큰을 통합하며, LoRA 기반의 파인튜닝을 통해 RGB와 알파 채널을 동시에 일관되게 생성하는 것을 목표로 합니다.

- **Technical Details**: TransPixar는 pretrained RGB 비디오 모델을 확장하여 알파 채널 생성에 필요한 새로운 기능을 추가했습니다. 방법론적으로, 우리는 텍스트와 RGB 토큰 이후에 알파 채널을 생성하기 위한 토큰을 추가하였으며, 이를 통해 학습 속도를 높이기 위한 노력도 기울였습니다. 또한, 알파 토큰의 위치 임베딩을 재초기화하여 RGB 토큰과의 차별성을 확보하였고, LoRA 기반의 파인튜닝 방식으로 RGB 생성 품질을 유지하면서 알파 토큰을 qkv 공간으로 프로젝션하는 접근을 사용했습니다.

- **Performance Highlights**: TransPixar는 제한된 데이터로도 다양하고 일관된 RGBA 비디오 생성을 성공적으로 달성하였습니다. 직관적으로 세분화된 Attention 메커니즘을 도입하여 RGB와 알파 채널 간의 강한 정렬을 유지하고, 다양한 도전 과제를 해결하는 데 효과적인 성과를 보였습니다. 또한, 기존 RGB 기반 모델의 강점을 보존하면서 새로운 알파 채널 생성 방안을 통합하는 혁신적인 결과를 제시했습니다.



### PiLaMIM: Toward Richer Visual Representations by Integrating Pixel and Latent Masked Image Modeling (https://arxiv.org/abs/2501.03005)
- **What's New**: 본 논문에서는 Pixel MIM과 Latent MIM의 강점을 결합한 새로운 프레임워크인 PiLaMIM을 제안합니다. 이는 한 개의 encoder와 두 개의 decoder로 구성되어, 각 decoder가 서로 다른 특성을 예측할 수 있게 합니다. 아울러, [CLS] 토큰을 이용하여 이미지를 전반적으로 관통하는 세밀한 시멘틱 정보를 캡쳐할 수 있도록 합니다.

- **Technical Details**: PiLaMIM은 Pixel MIM과 Latent MIM을 통합하여 시각적 표현에서 더욱 풍부한 결과를 추출하는 데 초점을 맞춥니다. 입력 이미지는 비중첩 패치로 나누어진 이후 선택된 패치가 랜덤하게 마스킹됩니다. 이후, context encoder는 가시적인 패치만을 처리하여 출력된 latent representation을 통하여 정보를 복원합니다.

- **Performance Highlights**: 실험 결과, PiLaMIM은 기존의 주요 벤치마크인 MAE, I-JEPA, BootMAE를 능가하는 성능을 보였습니다. 특히, 이 모델은 높은 수준의 시멘틱 정보와 낮은 수준의 시각적 정보를 동시에 캡쳐할 수 있어 다양한 과제에서 높은 성능을 발휘합니다.



### SurgRIPE challenge: Benchmark of Surgical Robot Instrument Pose Estimation (https://arxiv.org/abs/2501.02990)
Comments:
          35 pages, 18 figures, journal paper

- **What's New**: 본 연구에서는 Marker-less 방식의 수술 도구 자세 추정을 위한 'Surgical Robot Instrument Pose Estimation (SurgRIPE)' 챌린지를 소개하고, 2023년 의료 영상 컴퓨팅 및 컴퓨터 보조 개입(MICCAI) 국제 회의에서 개최되었습니다. 이 챌린지의 주요 목표는 실제 수술 비디오 데이터와 정밀한 도구 자세 정보를 제공하고, Markerless 자세 추정 방법의 평가를 위한 벤치마크를 수립하는 것입니다. 이를 통해 개발된 새로운 알고리즘들은 기존 방법들에 비해 정확성과 견고성이 향상되는 것을 보여주었습니다.

- **Technical Details**: SurgRIPE 데이터셋은 수술 도구의 6DoF 자세 추정을 위한 Markerless 방식으로 개발되었으며, 실제 수술 환경에서 수집된 고화질 영상을 포함합니다. 이 데이터셋은 두 가지 수술 기구, 즉 Large Needle Driver (LND)와 Maryland Bipolar Forceps (MBF)를 사용하여 수집되었습니다. 데이터는 각 기구별 3D 모델, RGB 이미지, 세그멘테이션 마스크 및 6DoF 자세의 정답 데이터를 포함하고 있으며, 모든 데이터는 25Hz의 프레임 속도로 캡처되었습니다.

- **Performance Highlights**: SurgRIPE 챌린지 참가자들은 이 데이터셋을 기반으로 서로 다른 Markerless 수술 도구 자세 추정 방법을 제안하고 검증하였습니다. 이번 챌린지를 통해 고급 알고리즘들이 로봇 수술 시스템에 통합될 가능성을 보여주었으며, 이는 보다 정밀하고 자율적인 수술 절차를 가능하게 할 것입니다. 챌린지는 수술 로봇 도구 자세 추정 분야에서 새로운 벤치마크를 수립하였고, 추가 연구 및 개발을 향한 가능성을 제시하였습니다.



### STAR: Spatial-Temporal Augmentation with Text-to-Video Models for Real-World Video Super-Resolution (https://arxiv.org/abs/2501.02976)
- **What's New**: 이 논문에서는 STAR(Spatial-Temporal Augmentation with T2V models for Real-world video super-resolution)라는 새로운 접근 방식을 제안합니다. 이 방법은 텍스트-비디오(T2V) 모델을 활용하여 현실 세계의 비디오 슈퍼해상도(VSR)에서 섬세한 공간적 세부사항과 강력한 시간적 일관성을 달성합니다. STAR는 현실적인 비디오 보정을 위해 고유한 지역 정보 향상 모듈(Local Information Enhancement Module, LIEM)과 다이나믹 주파수 손실(Dynamic Frequency Loss, DF Loss)을 통합합니다.

- **Technical Details**: STAR는 네 가지 주요 모듈로 구성됩니다: VAE(Variational AutoEncoder), 텍스트 인코더, ControlNet, 그리고 T2V 모델입니다. LIEM은 글로벌 셀프-어텐션 이전에 배치되어 지역 세부정보를 보강하고 품질 저하를 완화하는 역할을 합니다. DF Loss는 다양한 확산 단계에서 모델이 저주파수와 고주파수 정보에 집중할 수 있도록 유도하여 복원의 충실도를 향상시킵니다.

- **Performance Highlights**: 실험 결과, STAR는 합성 데이터셋과 현실 세계 데이터셋 모두에서 최신 기법들에 비해 뛰어난 성능을 보이는 것으로 나타났습니다. STAR는 모든 데이터셋에서 최고 수준의 선명도(DOVER score)를 달성하며, 시간적 일관성을 유지함으로써 현실적인 비디오 품질을 보장합니다. 이를 바탕으로 STAR는 T2V 모델의 장점을 최대한 활용하여 현실적인 VSR을 위한 새로운 기준을 제시합니다.



### HaWoR: World-Space Hand Motion Reconstruction from Egocentric Videos (https://arxiv.org/abs/2501.02973)
- **What's New**: 이 연구에서 제안하는 HaWoR(Human and World Reconstruction)는 3D 손 움직임을 세계 좌표계에서 재구성하는 새로운 방법입니다. 기존 연구들이 카메라 공간에서의 재구성에 중점을 두었다면, HaWoR은 에고센트릭 비디오를 기반으로 손의 움직임과 카메라의 궤적을 분리하여 추정합니다. 이는 특히 손의 궤적이 시야에서 벗어나거나 가려질 때를 고려해 새로운 동작 보간 네트워크를 설계하여 복원성을 높입니다.

- **Technical Details**: HaWoR은 에고센트릭 비디오를 통해 카메라 공간에서의 3D 손 움직임 재구성과 세계 좌표계에서 카메라 궤적 추정을 동시에 수행합니다. 이를 위해 전통적인 SLAM(Simultaneous Localization and Mapping) 방법의 한계를 극복하기 위해 적응형 SLAM 프레임워크를 도입했습니다. 손이 시야에서 벗어나도 결측된 프레임을 효과적으로 보완하는 최첨단 네트워크를 적용하여 더욱 정확한 손 움직임 궤적을 제공합니다.

- **Performance Highlights**: 다양한 에고센트릭 벤치마크 데이터셋에서 HaWoR의 성능을 평가한 결과, 손 움직임 재구성과 카메라 궤적 추정에서 최첨단 성능을 달성했습니다. HaWoR은 시각적 동적 환경에서의 복잡한 카메라 운동을 안정적으로 처리하며, 비디오에서 결측된 프레임이나 가림 현상이 있어도 높은 Fidelity를 유지합니다. 또한, 기존 최적화 기반 방법에 비해 전반적으로 향상된 성능을 보였습니다.



### Human Gaze Boosts Object-Centered Representation Learning (https://arxiv.org/abs/2501.02966)
Comments:
          13 pages

- **What's New**: 최근 자가 감독 학습(self-supervised learning) 모델이 사람의 시각 입력을 기반으로 학습하지만 이미지 인식에서는 인간에 비해 현저히 낮은 성능을 보이고 있습니다. 연구자들은 인간의 시선 위치에 해당하는 중앙 시각 정보를 강조함으로써 이러한 문제를 해결할 수 있는 방법을 탐구합니다. 이를 통해, 에고 세틱(egocentric) 시각 객체 학습 개선의 가능성을 제시하고 있습니다.

- **Technical Details**: 본 연구는 Ego4D 데이터셋을 활용하여 인간의 시선 예측 모델을 통해 생성한 시선 위치를 기반으로, 해당 위치 주변의 시각 정보를 크롭(crop)하여 입력 데이터로 사용합니다. 이로써 중앙 시각 정보의 중요성을 시뮬레이션(simulate)하고, 이를 통해 시각 표현이 어떻게 향상되는지를 실험적으로 증명합니다. 마지막으로, SSL 모델을 하나의 에폭(epoch) 동안 학습하여 시각 표현의 개선을 확인합니다.

- **Performance Highlights**: 중앙 시각에 초점을 맞춰 학습한 SSL 모델은 범주(category), 세부(fine-grained), 인스턴스(instance) 객체 인식에서 전 영역 시각 정보로 학습한 모델에 비해 성능 향상을 보였습니다. 또한, 시선 이동의 시간적 동역학이 시각 표현 구축에 중요한 역할을 하였음을 발견했습니다. 이러한 결과는 인간의 시각 경험을 모방하여, 보다 강력한 시각 표현을 학습할 수 있는 중요한 단계를 제시합니다.



### Socratic Questioning: Learn to Self-guide Multimodal Reasoning in the Wild (https://arxiv.org/abs/2501.02964)
- **What's New**: 이번 논문에서는 Complex Visual Reasoning(복잡 시각적 추론)을 위한 혁신적인 Multi-Round Training(다단계 훈련)과 Reasoning(추론) 프레임워크인 Socratic Questioning(SQ)을 제안합니다. SQ는 Multimodal Large Language Models(다중 모드 대형 언어 모델)에서 Hallucinations(환각)을 줄이고, 세부적인 이미지를 묘사하는 능력을 향상시키기 위한 Self-Questioning(자기 질문)의 접근 방식을 채택합니다. 이 연구의 중요한 기여로는, SQ 방법이 Hallucination 점수를 31.2% 개선시켰으며, 신규 multimodal mini-dataset인 CapQA를 사용하여 평가되었다는 점입니다.

- **Technical Details**: SQ는 Four-Step(4단계) 절차를 통해 이루어지며, Self-Ask(자기 질문)에서 시작해 Self-Answer(자기 답변), Consolidate & Organize(통합 및 정리), Summarize & Condense(요약 및 축소)로 이어집니다. 이 과정에서 MLLMs은 텍스트와 이미지를 활용하여 필요한 세부 정보를 시각적으로 검색하고, 이를 바탕으로 문제를 깊이 생각하게 됩니다. SQ는 Chain of Thought와 Visual Instruction Tuning 접근 방식을 통합하며, 이를 통해 시각적 정보에 대한 무시를 피하고, 높은 신뢰성을 제공하는 모델로 발전하게 됩니다.

- **Performance Highlights**: Socratic Questioning(SQ)은 Hallucinations를 효과적으로 줄이는 데 성공하며, 이는 추가적인 복잡한 아키텍처나 대량의 데이터 처리 없이 이루어집니다. 우리의 실험 결과는 SQ가 다양한 Visual Reasoning(시각적 추론) 및 Q&A(질문 및 답변) 벤치마크의 제로샷 성능을 극대화하는 데 뛰어난 능력을 발휘한다는 것을 보여줍니다. 또한, 우리는 GPT-4v를 활용하여 생성된 데이터의 품질을 평가하며 SQ의 유효성을 입증하였습니다.



### SceneVTG++: Controllable Multilingual Visual Text Generation in the Wild (https://arxiv.org/abs/2501.02962)
- **What's New**: 이 논문에서는 SceneVTG++라는 두 단계 방법을 제안하여 자연 장면 이미지에서 시각적 텍스트 생성을 가능하게 합니다. 이 방법은 Text Layout and Content Generator (TLCG)와 Controllable Local Text Diffusion (CLTD)라는 두 가지 구성 요소로 이루어져 있으며, 네 가지 핵심 기준인 Fidelity, Reasonability, Utility, Controllability를 동시에 충족할 수 있도록 설계되었습니다.

- **Technical Details**: TLCG는 Multi-Modal Large Models의 세계 지식을 활용하여 배경 이미지에 적합한 텍스트 레이아웃을 찾고, 관련된 텍스트 내용을 추천합니다. CLTD는 확산 모델을 기반으로 다국어 텍스트를 생성하며, 픽셀 수준의 조건부 확산 모델을 사용하여 크기에 관계없이 텍스트 생성을 가능하게 합니다. 이러한 과정은 자동화된 장면 텍스트 생성을 달성하는 데 도움이 됩니다.

- **Performance Highlights**: 광범위한 실험을 통해 TLCG와 CLTD의 효과를 검증하였으며, SceneVTG++가 텍스트 생성에서 최첨단 성능을 달성한다는 것을 입증하였습니다. 생성된 이미지는 OCR 작업에서 텍스트 탐지 및 인식과 같은 유용성에서 우수한 성능을 발휘하며, 두 개의 새로운 데이터셋인 SceneVTG-Erase++와 SceneVTG-Syn을 통해 다양성과 현실성을 더욱 높였습니다.



### MotionBench: Benchmarking and Improving Fine-grained Video Motion Understanding for Vision Language Models (https://arxiv.org/abs/2501.02955)
Comments:
          20 pages

- **What's New**: 본 논문은 비디오 이해를 위한 새로운 베enchmark인 MotionBench를 제안합니다. 이 벤치마크는 현재 비디오 모델의 세밀한 동작 이해 능력을 평가하는 데 중점을 두고 있습니다. 전체 8,052 질문으로 구성되어 있으며, 비디오 콘텐츠의 다양성을 반영하여 실질적인 비디오 이해 평가에 기여하고자 합니다.

- **Technical Details**: MotionBench는 다양한 출처에서 수집된 데이터를 기반으로 하여 여섯 가지 동작 중심 질문 유형을 포함합니다. 현재 사용되고 있는 비디오 이해 모델들은 세밀한 동작을 이해하는 데 있어 60% 미만의 정확도를 보이며, 이는 실질적인 응용에 부족함을 의미합니다. 이를 개선하기 위해 Through-Encoder Fusion (TE Fusion)이라는 새로운 아키텍처를 제안하고, 이를 통해 비디오 특성 표현을 증가시킵니다.

- **Performance Highlights**: TE Fusion 방법을 통해 높은 프레임 속도의 입력을 처리할 수 있으며, MotionBench에서 최첨단 성능을 달성하는 것으로 나타났습니다. 기존 모델들은 고압축 비율 시나리오에서 성능이 제한적이었지만, TE Fusion은 이러한 영역에서 두드러진 장점을 보입니다. 이 연구는 비디오 이해 모델 개발에 있어 세밀한 동작 이해의 중요성을 강조하며, 기존 모델의 한계를 드러내고 있습니다.



### 4D-CS: Exploiting Cluster Prior for 4D Spatio-Temporal LiDAR Semantic Segmentation (https://arxiv.org/abs/2501.02937)
Comments:
          Accepted for publication at IEEE Robotics and Automation Letters (RAL)

- **What's New**: 이번 논문에서는 LiDAR 포인트의 의미적 분할 문제를 다루며, 각각의 점에 대한 의미적 클래스와 운동 상태를 식별하기 위해 다중 스캔의 시공간 정보를 활용합니다. 기존 방법들이 공간적, 시간적 일관성을 간과한 점을 지적하고, 이를 해결하기 위해 다중 프레임에서 클러스터 레이블을 생성하여 보다 일관된 분할 결과를 도출하는 기법을 제안합니다. 이 방법은 점 중심 및 클러스터 중심의 두 가지 가지(branch)를 통합하여 분할 성능을 최적화합니다.

- **Technical Details**: 제안하는 4D-CS 네트워크는 두 가지 가지의 구조로 구성됩니다. 첫 번째 가지인 점 기반(branch)은 역사적 지식을 활용하여 현재 특성을 시간적으로 융합하여 enriquecing합니다. 두 번째 가지인 클러스터 기반(branch)에서는 DBSCAN과 같은 클러스터링 알고리즘을 사용해 여러 프레임에서 전경(foreground) 객체의 클러스터 레이블을 생성하고, 이 레이블을 통해 각 점의 정보를 집계하여 클러스터 특징(cluster features)을 도출합니다. 이를 통해 누락된 특성을 복원하고, 최종적으로 두 가지 가지의 정보를 적응적으로 융합하여 분할 결과를 최적화합니다.

- **Performance Highlights**: 4D-CS 네트워크는 SemanticKITTI 및 nuScenes 데이터셋에서 다중 스캔 의미적 및 이동 객체 분할의 향상된 성능을 입증합니다. 제안된 방법은 최첨단 결과를 달성하며, 클러스터링 정보를 활용하여 단일 객체 내에서 점 카테고리의 불일치를 해결하는 데 중점을 두고 있습니다. 또한, 정의된 세 가지 모듈인 Multi-view Temporal Fusion, Temporal Cluster Enhancement 및 Adaptive Prediction Fusion를 통해 분할 일관성을 강화합니다.



### Label-free Concept Based Multiple Instance Learning for Gigapixel Histopathology (https://arxiv.org/abs/2501.02922)
- **What's New**: 본 연구에서는 기존의 MIL 방식의 한계를 극복하기 위해 Concept MIL을 제안합니다. 이 모델은 이미지에서 병리학적 개념을 직접 예측하고, 병리학적 개념의 영향을 추적하여 예측에 대한 해석을 제공합니다. 이를 통해서 수작업으로 개념 라벨링을 할 필요가 없어져 다양한 질병에 쉽게 적용할 수 있는 유연성을 제공합니다.

- **Technical Details**: Concept MIL 모델은 고해상도 WSI 데이터를 분석하기 위해 최근 발전된 vision-language 모델을 활용합니다. 모델은 WSI의 상위 K 패치에서 식별된 개념들을 선형 결합하여 예측합니다. 이를 통해 각 개념이 예측에 미치는 영향을 명확하게 설명할 수 있으며, 모든 과정에서 수작업 주석을 제거하여 경제적 효율성을 높입니다.

- **Performance Highlights**: Concept MIL 모델은 Camelyon16과 PANDA 데이터셋에서 AUC 및 정확도를 0.9 이상으로 기록하며, 최신 모델들과 동등한 성능을 보여줍니다. 사용자 연구 결과, 모델이 식별한 개념들이 병리학자들이 사용하는 개념과 일치하여, 인간이 해석 가능한 WSI 분류의 유망한 전략으로 자리잡을 것입니다.



### Unsupervised Tomato Split Anomaly Detection using Hyperspectral Imaging and Variational Autoencoders (https://arxiv.org/abs/2501.02921)
Comments:
          CVPPA Workshop

- **What's New**: 이번 연구에서는 온실에서의 토마토 재배에서 발생하는 균열 이상 현상을 탐지하기 위한 비지도 학습 접근 방식을 제안합니다. 이를 위해 맞춤형 Variational Autoencoder (VAE)와 하이퍼스펙트럼 입력을 사용하여 토마토의 균열을 효과적으로 탐지할 수 있는 방법을 개발했습니다. 또한 530nm - 550nm 범위의 파장이 토마토 균열 탐지에 효과적임을 입증하였습니다.

- **Technical Details**: 하이퍼스펙트럼 이미징(Hyperspectral Imaging, HSI)의 사용은 여러 파장을 통해 물체의 스펙트럼 특성을 관찰할 수 있게 해줍니다. 데이터 수집을 위해 Specim의 FX10e 하이퍼스펙트럼 카메라를 사용하였고, 400nm에서 1000nm 범위의 파장을 스캔할 수 있습니다. 연구 과정에서 74개의 토마토 샘플을 수집하였으며, 이들을 공간적으로 증강하여 정상 및 비정상 샘플을 생성하였습니다.

- **Performance Highlights**: 비지도 학습 기반의 VAE를 사용하여 재구성 손실을 분석함으로써 균열 이상 탐지의 성능을 크게 향상시켰습니다. 이 접근 방식은 결정 경계를 학습하여 이상을 탐지할 수 있으며, 정상 데이터에 대해 훈련된 모델이 비정상 데이터를 정확히 재구성하는 데 어려움을 겪는 원리를 사용합니다. 이러한 방식은 조기 탐지를 통한 생산물의 품질 향상에 기여할 것으로 기대됩니다.



### Spiking monocular event based 6D pose estimation for space application (https://arxiv.org/abs/2501.02916)
Comments:
          6 pages, 2 figures, 1 table. This paper has been presented in the Thursday 19 September poster session at the SPAICE 2024 conference (17-19 September 2024)

- **What's New**: 최근의 우주 접근성 향상으로 인해 우주선 발사와 대규모 위성 집합체 프로젝트가 급증하고 있습니다. 이로 인해 우주 쓰레기와의 충돌 위험이 증가하고 있으며, 이에 따라 우주선 자세 추정 알고리즘의 개발이 필요해졌습니다. 본 논문에서는 우주선의 자세 추정을 위한 완전한 이벤트 기반 접근 방식을 제안하며, 이를 통해 더욱 정확하고 효율적인 솔루션을 모색하고자 합니다.

- **Technical Details**: 이 논문에서 제안하는 시스템은 이벤트 기반 카메라(Event-Based Camera, EBC)와 스파이킹 신경망(Spiking Neural Networks, SNN)을 활용하여 우주선의 자세 추정 문제에 접근합니다. SNN은 생물학적으로 영감을 받은 신경망으로, 낮은 전력 소모로 사건 기반의 정보를 처리할 수 있습니다. 다양한 이벤트 스트림 표현 방법이 소개되며, 특히 고동적 범위(High Dynamic Range)에 대한 이점이 강조됩니다.

- **Performance Highlights**: 제안된 소형 스파이킹 네트워크(S2E2)는 21cm의 위치 오차 및 14도의 회전 오차를 기록하였습니다. 이는 우주선 자세 추정을 위한 완전한 이벤트 기반 처리로 나아가는 중요한 첫 단계로 여겨지며, 이벤트 스트림을 고려한 새로운 방법론을 제시합니다. 이러한 결과는 우주 임무에서 자원 제약 환경에서도 성공적으로 작동할 수 있는 가능성을 보여줍니다.



### Pointmap-Conditioned Diffusion for Consistent Novel View Synthesis (https://arxiv.org/abs/2501.02913)
- **What's New**: 본 논문에서는 PointmapDiffusion이라는 새로운 프레임워크를 제안하며, 이는 사전 훈련된 2D diffusion 모델을 활용하여 단일 이미지로부터 새로운 관점을 생성하는 novel view synthesis(NVS) 작업을 수행합니다. 이 프레임워크는 pointmaps(래스터화된 3D 장면 좌표)를 조건 신호로 사용하여 참조 이미지에서 기하학적 선험지식을 캡처하여 diffusion 과정을 안내합니다. 이를 통해 저희 모델은 생성 능력과 기하학적 일관성을 균형 있게 조화시켜 다양한 시점에서 정확한 뷰 합성을 가능하게 합니다.

- **Technical Details**: PointmapDiffusion의 핵심은 2D diffusion 기능에 3D 구조의 감각을 결합하는 것이며, 이를 통해 기하학적 특징과 구조적 세부정보를 pointmaps 형태로 추출합니다. 이때 Pointmap ControlNet이라는 신경망 구조를 적용하여 diffusion 모델을 조건화하고 기하학적 일관성을 강화합니다. ControlNet은 여러 pointmaps 간의 공간적 상관성을 학습하여 참조 및 목표 뷰 간의 경계를 효과적으로 연결합니다.

- **Performance Highlights**: 다양한 실 세계 데이터셋에 대한 광범위한 실험을 통해 PointmapDiffusion은 높은 품질의 다중 뷰 일관성 결과를 보여주며, 단일 이미지 NVS 작업에 대해 다른 기준 모델들과 비교해 훈련 가능한 매개변수가显著하게 적습니다. 이는 저희 접근 방식이 더 적은 조정으로도 새로운 뷰와 장면에 적응할 수 있도록 하여 참조 시점과의 정렬을 유지하는 일관된 결과를 보장한다는 것을 의미합니다.



### Comprehensive Pathological Image Segmentation via Teacher Aggregation for Tumor Microenvironment Analysis (https://arxiv.org/abs/2501.02909)
Comments:
          38 pages, 13 figures

- **What's New**: 이 논문에서는 종양 미세환경(tumor microenvironment, TME)의 분석에서 혁신적인 접근법인 PAGET(Pathological image segmentation via AGgrEgated Teachers)를 제안합니다. PAGET는 여러 개의 세분화(segmentation) 모델을 통합하는 지식 증류(knowlwdge distillation) 방법을 사용하여, 세포 유형의 계층적 구조를 고려합니다. 이를 통해 TME의 여러 주요 구성 요소를 동시에 식별하고 분류할 수 있습니다.

- **Technical Details**: PAGET는 면역 조직화학 재염색(immunohistochemical restaining) 기술을 통해 생성된 독특한 데이터셋을 활용하여, 기존의 세분화 모델들을 통합합니다. 이 방법은 14개의 주요 TME 구성 요소에 대한 신속하고 포괄적인 세분화를 가능하게 하며, 다양한 조직 유형(tissue types)과 의료 기관(medical institutions)에서 효율적으로 작동합니다. 세포 유형의 계층적 상관 관계를 고려한 이 접근법은 기존의 한계를 극복합니다.

- **Performance Highlights**: PAGET는 다양한 조직 유형에서 TME segmentation을 신속하고 포괄적으로 수행할 수 있는 능력을 보여 주며, 이를 통해 종양 미세환경에 대한 정량적 분석(quantitative analysis)에 기여합니다. 이 방법은 대규모 조직병리학 이미지(histopathology images)에서 정밀한 임상 의사결정(clinical decision-making)을 지원하는 데 중요한 진전을 이룹니다. 이를 통해 암 생물학(cancer biology)에 대한 이해를 높이는 데 기여하게 됩니다.



### FoundPAD: Foundation Models Reloaded for Face Presentation Attack Detection (https://arxiv.org/abs/2501.02892)
Comments:
          Accepted at WACV 2025 workshops

- **What's New**: 본 논문에서는 발표 공격(presentation attack)을 탐지하는 새로운 방법인 FoundPAD를 소개합니다. 이는 기존의 PAD(프레젠테이션 공격 탐지) 시스템이 겪는 일반화 부족과 대규모 훈련 데이터 요구라는 두 가지 주요 문제를 해결하기 위해 설계되었습니다. FoundPAD는 LoRA(weights)와 함께 분류 헤더를 훈련시키면서 CLIP(Contrastive Language-Image Pretraining) 모델을 적응시키는 혁신적인 접근 방식을 사용합니다.

- **Technical Details**: FoundPAD는 적은 양의 데이터로도 높은 일반화 성능을 이루기 위해 사전 훈련된 FM의 힘을 활용합니다. 특히, 사전 훈련된 FM에 적합한 LoRA를 적용하여 PAD 작업에 활용하고 있습니다. 이 방법은 데이터 간 불일치가 큰 이종 도메인에서도 PAD의 성능을 향상시켰으며, 실험을 통해 기존 방법보다 우수한 성능을 나타냈습니다.

- **Performance Highlights**: FoundPAD는 HTER(Half Total Error Rate)에서 평균적으로 문헌에서의 두 번째로 좋은 성능 대비 6.54 퍼센트 포인트 낮은 결과를 도출했습니다. 이는 또한 제한된 데이터 세트에서 훈련된 모델들과 비교했을 때, 4.35 포인트 및 8.94 포인트 더 나은 결과를 보였습니다. 이러한 성과는 FMs의 일반화 능력을 입증하며 PAD 분야에서의 기존 SOTA(state-of-the-art) 솔루션을 초월하는 가능성을 보여줍니다.



### MDP3: A Training-free Approach for List-wise Frame Selection in Video-LLMs (https://arxiv.org/abs/2501.02885)
Comments:
          24 pages, 10 figures

- **What's New**: 이 논문에서는 비디오 대형 언어 모델(Video-LLMs)의 시각적 이해를 향상시키기 위해, 핵심 프레임 선택의 세 가지 원칙인 쿼리 관련성(query relevance), 목록-wise 다양성(list-wise diversity), 그리고 순차성(sequentiality)을 강조합니다. 이러한 원칙을 만족하기 위한 새로운 방법으로 동적 프로그래밍(dynamic programming) 기반의 마르코프 결정 결정론적 점 프로세스(Markov decision determinantal point process, MDP3)를 제안합니다. MDP3는 훈련이 필요 없는 모델 독립적 방식으로, 기존 비디오 모델과 원활하게 통합될 수 있습니다.

- **Technical Details**: MDP3는 우선 쿼리에 조건화된 프레임 유사성을 추정하기 위해 조건부 가우시안 커널(conditional Gaussian kernel)을 사용하고, 이를 통해 생성된 유사성 행렬에 결정론적 점 프로세스(DPP)를 적용하여 쿼리 관련성과 리스트 기반 다양성을 캡처합니다. 순차성을 포함하기 위해 비디오를 연속적인 세그먼트로 나누고 각 세그먼트 내에서 DPP를 적용하여 이전 선택에 따라 조건화합니다. 이 과정은 마르코프 결정 과정(Markov Decision Process, MDP)으로 모델링되어 세그먼트 간 선택 크기를 최적화합니다.

- **Performance Highlights**: 실험적으로 MDP3는 세 가지 널리 사용되는 긴 비디오 벤치마크인 Video-MME, MLVU, LongVideoBench에서 기존 방법들에 비해 성능이 크게 향상되는 것을 보여줍니다. MDP3는 NP-하드 리스트 기반 프레임 선택 문제에 대해 (1-1/e) 근사 솔루션을 제공하며, 기존의 기준 방법보다 현저하게 높은 성능 개선을 달성하고 있습니다. 이러한 효율성 덕분에 MDP3는 다양한 기존 Video-LLMs에 비해 손쉬운 통합과 적응성을 제공합니다.



### PARF-Net: integrating pixel-wise adaptive receptive fields into hybrid Transformer-CNN network for medical image segmentation (https://arxiv.org/abs/2501.02882)
- **What's New**: 본 논문에서는 PARF-Net이라는 새로운 방법을 제안합니다. 이 방법은 Pixel-wise Adaptive Receptive Fields(Conv-PARF)를 사용하여 인식 필드를 동적으로 조정함으로써 다양한 모양과 크기를 가진 병변을 배경과 분리하는 데 필요한 특징을 효과적으로 추출합니다. 이를 통해 하이브리드 Transformer-CNN 아키텍처에 통합되어 의료 영상 세분화의 성능을 향상시킵니다.

- **Technical Details**: PARF-Net는 입력 이미지로부터 식별할 수 있는 특징을 추출하기 위해 Conv-PARF를 인코더에 도입합니다. Conv-PARF는 각 픽셀에 대해 적응성 있는 합성곱 필드를 통해 특징을 지속적으로 조정하며, 이는 예를 들어, 노이즈가 많은 배경에서 병변을 분리하는 데 유리합니다. 또한 하이브리드 Transformer-CNN 모듈을 이용하여 로컬 특징과 글로벌 특징을 효율적으로 추출하고 결합합니다.

- **Performance Highlights**: PARF-Net은 MoNuSeg, GlaS, DSB2018, 다기관 Synapse 등 네 가지 의료 이미지 데이터셋에서 평가되어 기존의 방법들보다 우수한 성능을 보였습니다. 특히 Synapse 데이터셋에서 84.27%의 평균 Dice 점수를 달성하여 기존 방법들을 크게 초월하는 결과를 얻었습니다.



### Two-Dimensional Unknown View Tomography from Unknown Angle Distributions (https://arxiv.org/abs/2501.02872)
Comments:
          Accepted to the International Conference on Acoustics, Speech, and Signal Processing (ICASSP) 2025

- **What's New**: 이번 연구에서는 각도 분포가 알려지지 않은 2D tomography 기술을 제안합니다. 기존의 2D UVT 알고리즘들은 일반적으로 각도가 알려져 있다고 가정하지만, 본 연구에서는 이러한 정보를 추정하는 방법을 다룹니다. 이 방법론은 최적화 작업으로 문제를 형성하며, 교차 검증 오차를 기반으로 각도 분포와 2D 구조를 함께 추정합니다.

- **Technical Details**: 제안된 방법론은 교차 검증 기반 접근 방식을 사용하여 노이즈가 있는 무작위 1D 프로젝션으로부터 각도 분포와 2D 구조 𝑔를 결정합니다. 두 가지 확률 분포 모델, 즉 반파라메트릭 혼합 von Mises 밀도와 확률 질량 함수 모델을 활용하여 알고리즘의 능력을 탐색합니다. 교차 검증 오차(CVE)를 최소화하기 위해 대체 최소화라고 불리는 방법을 사용하여 각도 분포를 찾습니다.

- **Performance Highlights**: 알고리즘 성능은 PCA 기반의 노이즈 제거 기법과 추정된 분포의 순서 통계에 의해 구동되는 Graph Laplacian Tomography (GLT)를 통해 평가되었습니다. 이 과정에서 알고리즘은 각 분포의 거의 완벽한 순서를 보장하며, 직관적인 기준선과 비교되어 그 성능이 입증됩니다.



### A Novel Vision Transformer for Camera-LiDAR Fusion based Traffic Object Segmentation (https://arxiv.org/abs/2501.02858)
Comments:
          International Conference on Agents and Artificial Intelligence 2025

- **What's New**: 이 논문은 카메라와 LiDAR 데이터를 융합하는 카메라-LiDAR 융합 변환기 모델(CLFT)을 제안합니다. CLFT 모델은 다양한 객체를 효과적으로 분리할 수 있는 기능을 갖추고 있으며, 비육안 및 기상 조건에서도 뛰어난 성능을 자랑합니다. 새로운 방법론을 통해 기존 CNN 및 비전 변환기 모델을 초 능률하기 위해 설계되었습니다.

- **Technical Details**: CLFT 네트워크는 모듈별 특징들을 차례로 조합하고, 디코더 블록에서 최종 융합을 진행하는 두 가지 방향을 갖습니다. 데이터 처리 과정은 크게 세 단계로 나누어지며, 첫 단계는 입력 데이터를 임베딩하여 학습 가능한 변환기 토큰으로 변환합니다. 두 번째 단계에서는 ViT의 인코더 프로토콜을 따라 토큰을 인코딩하고, 세 번째 단계에서는 세그멘테이션 예측을 위해 특징 표현을 조합하는 후처리 과정을 수행합니다.

- **Performance Highlights**: CLFT 모델은 자율주행 감지 작업에서 뛰어난 성능을 보여주며, 복잡한 기상 환경에서도 활용 가능합니다. 실험 결과, CLFT는 다른 비전 변환기 모델에 비해 뛰어난 분류 정확도를 달성하였습니다. 이를 통해 CLFT 모델은 자율주행 인식의 최신 기술을 선도하며, 다양한 상황에서 객체 세분화 작업을 지원하고 있습니다.



### Synthetic Fungi Datasets: A Time-Aligned Approach (https://arxiv.org/abs/2501.02855)
Comments:
          8 pages, 3 figures, 1 table, 1 algorithm

- **What's New**: 이 연구에서는 곰팡이 성장의 주요 단계를 모델링하는 합성 시간 정렬 이미지 데이터 세트를 소개합니다. 이 데이터 세트는 스포어 크기 감소, 가지치기 역학, 복잡한 균사체 네트워크 생성과 같은 현상을 체계적으로 캡처하여, 시간이 의존적인 곰팡이의 동적 생물학적 과정을 탐구할 수 있는 기초 자료를 제공합니다. 이를 통해 농업, 의학, 산업 미생물학 등 다양한 분야에서의 곰팡이 분석 자동화 및 질병 모니터링을 향상시킬 수 있습니다.

- **Technical Details**: 이 데이터 세트는 스포어에서 성숙한 균사체 구조로의 동적 형태 변화를 모델링합니다. 여기에서 중요하게 다루어진 것은 재귀 가지치기( recursive branching)를 통해 곰팡이 구조의 성장 역학을 정확하게 시뮬레이션하는 것입니다. 데이터 세트의 생성 과정은 확률 제어를 사용하여 자연스러운 외관을 보장하며, 환경적 요인, 예를 들어 온도에 따라서도 가지 길이와 너비가 조정됩니다.

- **Performance Highlights**: 딥러닝(DL) 기술에 최적화된 이 데이터 세트는 곰팡이 종 분류, 성장 단계 식별, 이상 탐지 및 시간에 따른 곰팡이 행동 예측을 가능하게 합니다. 이러한 기능은 곰팡이 모니터링 시스템 자동화와 효율성 향상에 기여하며, 곰팡이가 중요한 역할을 하는 생명공학 공정에서도 유용합니다. 합성 데이터 세트를 활용하여 연구자들은 실세계 데이터의 부족 문제를 해결하면서도 고정밀 도구를 개발할 수 있습니다.



### Large Language Models for Video Surveillance Applications (https://arxiv.org/abs/2501.02850)
Comments:
          Accepted for TENCON 2024

- **What's New**: 이 논문은 Generative Artificial Intelligence (GenAI)를 활용한 비디오 분석 도구 개발을 소개합니다. Vision Language Models를 통해 사용자가 정의한 쿼리를 바탕으로 맞춤형 텍스트 요약을 생성하여, 방대한 비디오 데이터셋 내에서 주목할 만한 통찰력을 제공합니다. 전통적인 방식과 달리, 이 접근 방식은 분석의 정밀도와 효율성을 크게 향상시킵니다.

- **Technical Details**: 본 연구에서 제안하는 비디오 분석 파이프라인은 CCTV 카메라 등 다양한 비디오 소스를 활용하여 개별 프레임을 세분화하고, GenAI 모델인 Gemini Pro Vision을 통해 각 프레임을 분석합니다. 이 과정은 최적의 결과를 얻기 위해 프레임 속도와 컨텍스트를 조절하며, 최종 출력은 전체 비디오의 포괄적인 요약을 생성합니다. 이 필터링 메커니즘을 통해 주목할 만한 정보만을 선별하여 사용자가 쉽게 접근 가능하게 합니다.

- **Performance Highlights**: 본 연구는 CCTV 네트워크 내에서 사용자가 정의한 쿼리를 기반으로 한 요약의 효과성을 80%의 시간적 정확도와 70%의 공간적 일관성으로 평가했습니다. 실험을 통해 Gemini Pro Vision 모델이 사람, 그들의 상호작용 및 환경적 요소를 효과적으로 식별하여 높은 평가를 받았음을 입증했습니다. 이러한 성능은 보안 및 행동 분석과 같은 다양한 분야에서 활용될 수 있습니다.



### HOGSA: Bimanual Hand-Object Interaction Understanding with 3D Gaussian Splatting Based Data Augmentation (https://arxiv.org/abs/2501.02845)
Comments:
          Accepted by AAAI2025

- **What's New**: 본 논문에서는 bimanual hand-object interaction (양손 물체 상호작용)을 이해하기 위한 새로운 3D Gaussian Splatting 기반 데이터 증강 프레임워크인 Hand-Object Gaussian Splatting Augmentation (HOGSA)를 제안하고 있습니다. 기존의 데이터 부족 문제와 3D 주석 정확성 문제를 해결하기 위해 다각적으로 접근하여 다양한 손-물체 포즈 및 시점을 가진 대규모 포토리얼리스틱 데이터를 생성합니다. 이 연구는 H2O와 Arctic 두 개의 벤치마크에서 성능 향상을 확인했습니다.

- **Technical Details**: HOGSA 프레임워크는 mesh-based 3DGS를 사용하여 손과 물체를 모델링하며, 포즈 최적화 모듈을 통해 다양한 양손-물체 포즈를 생성합니다. 또한, 슈퍼 해상도 모듈을 설계하여 다중 해상도 입력 이미지로 인한 렌더링 블러 문제를 해결합니다. 이러한 과정은 원본 데이터셋과 증강된 데이터셋을 통합하여 양손 물체 상호작용의 성능을 크게 향상시키는 데 기여합니다.

- **Performance Highlights**: HOGSA를 통해 개선한 데이터셋은 기존의 벤치마크인 H2O와 Arctic에서의 성능을 향상시킵니다. 특히 포즈 다양성과 현실감을 측정하여 모델의 상호작용 이해 정확도를 높이는 데 기여하고 있습니다. 이 프레임워크는 한손-물체 상호작용에서의 증강 접근 방식을 확장하여 bimanual hand-object interaction에 대한 새로운 가능성을 제공합니다.



### Enhanced Rooftop Solar Panel Detection by Efficiently Aggregating Local Features (https://arxiv.org/abs/2501.02840)
Comments:
          Accepted at CODS-COMAD 2024, December, 2024, Jodhpur, India (this https URL)

- **What's New**: 이 논문에서는 위성 이미지(satellite images)를 사용한 향상된 합성곱 신경망(CNN) 기반의 옥상 태양광 발전 패널(detection of rooftop solar photovoltaic panels) 탐지 방법을 제안합니다. 사전 학습된 CNN 모델을 활용하여 옥상의 로컬 컨볼루션 특징(local convolutional features)을 추출하고, 이를 Vectors of Locally Aggregated Descriptors (VLAD) 기법을 통해 결합하여 옥상 수준의 글로벌 특징(global features)을 획득합니다. 이러한 특징을 전통적인 머신러닝(Machine Learning) 모델에 적용하여 태양광 패널이 있는 이미지와 없는 이미지를 분류합니다.

- **Technical Details**: 본 연구에서 사용된 데이터셋(dataset)에 대해 제안한 방법은 세 개의 도시에서 옥상-태양광 발전(PV) 분류 점수가 0.9를 초과하는 결과를 도출하였습니다. 각 특징 추출기(feature extractor) 네트워크를 평가하여 전통적인 머신러닝 모델을 학습시킵니다. 또한, 제한된 레이블 데이터(labelled data)가 있는 새로운 도시 또는 지역에서 이전에 학습된 모델을 효율적으로 활용할 수 있는 3단계 접근 방식(3-phase approach)을 제안합니다.

- **Performance Highlights**: 제안된 3단계 접근 방식은 다수의 도시에서 옥상 태양광 발전 탐지 작업의 유효성을 잘 보여줍니다. 모든 실험에서 높은 정확도를 기록하였으며, 이는 향후 옥상 태양광 패널 탐지 분야의 연구에 기여할 것입니다. 또한 다양한 특성 추출 네트워크로부터 얻어진 결과들은 모델의 내구성과 범용성을 입증하였습니다.



### Universal Features Guided Zero-Shot Category-Level Object Pose Estimation (https://arxiv.org/abs/2501.02831)
Comments:
          Accepted by AAAI2025

- **What's New**: 이번 논문에서는 사전 학습된 모델에서 추출한 2D 및 3D 유니버설 피처를 활용하여 기존의 카테고리 수준에서 6-DOF 물체 자세 추정을 제안합니다. 특히, 모델 재훈련이나 미세 조정 없이도 보이지 않는 카테고리에 대해 제로 샷(zero-shot) 형태의 물체 자세 추정을 가능하게 하였습니다. 이는 AR/VR 및 로보틱스 애플리케이션에서 물체의 방향 및 위치 추정에 중요한 기여를 할 것으로 보입니다.

- **Technical Details**: 제안된 방법은 입력된 RGB-D 이미지의 2D 및 3D 유니버설 피처를 결합하여 코스 투 파인(coarse-to-fine) 방식으로 6-DOF 자세 추정을 수행합니다. 초기에는 2D 유니버설 피처를 사용하여 희소한 대응관계를 찾고 초점 물체(pose)를 추정한 후, 피처 대응의 저하 문제를 방지하기 위해 반복적인 전략을 통해 최적화를 진행합니다. 마지막으로, 3D 유니버설 피처를 사용하여 내부 카테고리 물체 간의 형태 차이로 인한 자세 모호성을 해결합니다.

- **Performance Highlights**: REAL275 및 Wild6D 벤치마크에서 제안된 방법은 이전 방법들보다 더 높은 정확도를 보여 주며 보이지 않는 카테고리의 물체에 대해서도 강력한 대응관계를 설정합니다. 실험 결과에 따르면, 제안된 방법은 각 카테고리를 효과적으로 구분하여 물체의 자세를 정확하게 추정하는 데 성공했습니다. 이러한 결과는 전통적인 인스턴스 수준 및 카테고리 수준의 방법에 비해 뛰어난 일반화 능력을 입증합니다.



### RDD4D: 4D Attention-Guided Road Damage Detection And Classification (https://arxiv.org/abs/2501.02822)
- **What's New**: 이번 논문에서는 도로 손상 탐지 및 평가를 위한 새로운 데이터셋인 Diverse Road Damage Dataset (DRDD)을 소개합니다. 이 데이터셋은 다양한 손상 유형을 개별 이미지로 포착함으로써 기존의 데이터셋에서의 중요 공백을 해결하는 데 기여합니다. 또한, Attention4D 블록을 활용한 RDD4D 모델을 제안하여 다양한 스케일에서의 피쳐 정제를 향상시킵니다.

- **Technical Details**: RDD4D 모델은 Attention4D 모듈을 통해 피쳐 맵을 처리하여 위치 인코딩(positional encoding)과 'Talking Head' 요소를 결합한 주의 메커니즘을 사용합니다. 이 모듈은 지역적(local) 및 전역적(global) 문맥 정보를 포착하여 도로 손상을 효과적으로 탐지합니다. 본 연구는 다양한 최신 모델들과의 비교 실험을 통해 RDD4D의 우수성을 입증합니다.

- **Performance Highlights**: 우리 모델은 큰 크기의 도로 균열 탐지에서 0.458의 평균 정밀도(Average Precision, AP)를 기록하며 우수한 성능을 보였습니다. 전체 AP는 0.445로 경쟁력 있는 성능을 유지하였습니다. 또한, CrackTinyNet 데이터셋에서도 성능이 약 0.21 증가하여 향상된 결과를 보여주었습니다.



### InpDiffusion: Image Inpainting Localization via Conditional Diffusion Models (https://arxiv.org/abs/2501.02816)
- **What's New**: 이번 논문에서는 Image Inpainting Localization (IIL) 문제를 다루기 위해 새로운 프레임워크인 InpDiffusion을 제안합니다. 기존의 IIL 방법들이 과신(overconfidence)으로 인해 부정확한 예측을 하거나 섬세한 변조 경계를 탐지하는 데 어려움을 겪는 반면, InpDiffusion은 조건부 마스크 생성(condition mask generation) 작업으로 IIL을 재구성합니다. 이 방법은 디퓨전 모델을 활용하여 점진적으로 예측을 개선하고, 엣지 조건(edge conditions)과 새로운 엣지 감독(edge supervision) 전략을 통해 세부 경계를 강화합니다.

- **Technical Details**: InpDiffusion은 Adaptive Conditional Network (ACN) 기능을 갖추고 있어, 이미지에서 의미적 특징과 엣지 특징을 동시에 추출합니다. Dual-stream Multi-scale Feature Extractor (DMFE)를 통해 다중 스케일 특징을 효과적으로 추출하여, 세뇌와 엣지 조건을 바탕으로 자세한 표현을 강화합니다. 각 디노이징 단계에서 엣지 감독을 적용하여 각 샘플링 단계에서의 과도한 무작위성 문제를 해결하고, 이미지의 배경과 변조된 영역 간의 차이를 명확히 합니다.

- **Performance Highlights**: 실험 결과, InpDiffusion은 기존의 최첨단 IIL 방법들과 비교할 때 크게 향상된 성능을 보여줍니다. 다양한 도전적인 데이터셋에서 광범위한 실험을 통해, 이 모델은 뛰어난 일반화 능력과 강인성을 입증하였습니다. 논문은 InpDiffusion이 이미지 인페인팅 작업에서 더욱 정확하고 신뢰할 수 있는 결과를 제공함을 강조합니다.



### First-place Solution for Streetscape Shop Sign Recognition Competition (https://arxiv.org/abs/2501.02811)
Comments:
          technical report

- **What's New**: 이번 논문에서는 거리뷰 상점 간판 인식 기술을 개발하여 상업 지역의 비즈니스 가치 분석 및 스마트 시티 계획에 활용할 수 있는 가능성을 제시하고 있습니다. 기존의 복잡한 디자인과 다양한 텍스트 스타일을 가진 간판 인식의 문제를 해결하기 위해 다단계 접근 방식을 채택하였습니다. 또한, BoxDQN과 같은 혁신적인 강화학습 기법과 텍스트 정정 방법을 적용하여 뛰어난 성능을 달성하였습니다.

- **Technical Details**: 연구 팀은 상점 간판 감지, 텍스트 상자 감지 및 텍스트 인식을 포함하는 네 가지 주요 단계로 구성된 다단계 알고리즘을 사용하였습니다. 인스턴스 세그먼테이션을 통해 간판의 우아한 마스크를 만드는 대신, Mask-RCNN을 기반으로 한 모델 아키텍처가 세 가지 주요 영역에서 개선되었습니다. 이로 인해 좀 더 향상된 특징 추출 능력을 제공하고, 다중 작업 학습 방식에서의 모델의 정확성과 적응성을 높였습니다.

- **Performance Highlights**: 본 연구에서의 전반적인 실험은 복잡한 도시 환경에서 텍스트 인식 능력을 유의미하게 향상시킬 수 있는 가능성을 보여주었습니다. 텍스트 감지와 간판 텍스트 추출을 결합한 end-to-end 훈련 방식을 통해 두 가지 작업의 정확도를 동시에 향상시켰습니다. 최종적으로, 최적의 텍스트 박스를 찾기 위한 BoxDQN을 사용하여 인식 모델의 정확도를 개선하는 등 실질적인 성과들을 입증하였습니다.



### AE-NeRF: Augmenting Event-Based Neural Radiance Fields for Non-ideal Conditions and Larger Scen (https://arxiv.org/abs/2501.02807)
- **What's New**: 이 연구에서는 AE-NeRF라는 새로운 프레임워크를 통해 비이상적인 조건에서 이벤트 기반 NeRF 학습의 문제를 해결할 방안을 제시합니다. 특히, 비균일한 이벤트 시퀀스와 정확하지 않은 카메라 포즈를 다루기 위해 포즈 수정 모듈을 결합하여 강력한 3D 재구성을 지원합니다. 또한, 대규모 씬을 처리하기 위해 계층적 이벤트 증류를 제안하여 재구성 과정을 정제합니다.

- **Technical Details**: AE-NeRF는 이벤트 스트림의 밀도를 활용하며, 이벤트 기반 NeRF(e-NeRF) 프레임워크 내에서 포즈 보정 모듈을 공동 학습하여 정확하지 않은 카메라 포즈에서도 강인한 3D 재구성을 가능하게 합니다. 실험적으로는 고유 및 비이상적인 조건을 포함한 large-scale 씬을 대상으로 한 종합 벤치마크를 구축하였고, 이를 통해 이벤트 재구성 손실 및 시간 손실을 제안하여 재구성된 씬의 뷰 일관성을 향상시킵니다.

- **Performance Highlights**: 실험 결과, AE-NeRF는 자체 구축한 벤치마크에서 기존 최신 기술을 초월하여 이벤트 기반 3D 재구성 분야에서 새로운 최첨단 성능을 기록하였습니다. 신뢰성 있는 대규모 씬에서의 재구성이 가능하여 실용적인 비이상적인 상황에서도 높은 성능을 유지합니다. 이러한 접근 방식은 로봇공학, 3D 게임 등 다양한 다운스트림 응용에서도 유용하게 활용될 수 있습니다.



### COph100: A comprehensive fundus image registration dataset from infants constituting the "RIDIRP" databas (https://arxiv.org/abs/2501.02800)
Comments:
          12 pages, 7 figures

- **What's New**: 이번 연구에서는 소아 안과 질환 연구를 위한 새로운 데이터셋인 COph100을 소개합니다. 지난 연구-기반 데이터셋들(기존의 공개 데이터셋은 성인 망막 질환에 국한되어 있음)과 달리, COph100은 유아의 망막 이미지를 포함하며, 다양한 이미지 품질 문제를 다룹니다. 데이터셋에는 100개의 눈과 491개의 이미지 쌍이 포함되어 있으며, 각 이미지에 대한 자동 혈관 세분화 마스크가 제공됩니다.

- **Technical Details**: COph100은 ROP(조기 출생아 망막병증) 영아의 망막 이미지를 수집하여 만든 데이터셋으로, 각 눈에는 2~9회의 검사 세션이 포함되어 있습니다. 기존의 망막 이미지 등록 데이터셋들과 비교하여 다양한 임상 조건에서의 이미지 변동성을 반영하고 있습니다. 연구에 사용된 데이터셋은 고해상도 이미지를 포함하는 여러 데이터베이스에서 선택되었습니다.

- **Performance Highlights**: COph100 데이터셋의 이미지 품질과 등록 결과는 최신 알고리즘을 사용하여 평가되었습니다. 이 연구는 기존의 데이터셋의 한계를 극복할 수 있도록 설계되었으며, 특히 소아 안과 질환의 진행 상황 분석에 유용한 자원을 제공합니다. COph100을 통해 망막 이미지 등록 방법론을 강력하게 비교할 수 있으며, 소아 환자의 클리닉 결과를 개선할 수 있는 가능성을 제공합니다.



### GLoG-CSUnet: Enhancing Vision Transformers with Adaptable Radiomic Features for Medical Image Segmentation (https://arxiv.org/abs/2501.02788)
- **What's New**: 본 연구에서는 의학 이미지 의미 분할(medical image semantic segmentation) 분야에서 Transform 기반 모델의 한계점을 극복하기 위해 Gabor 필터와 Laplacian of Gaussian (LoG) 필터를 활용한 GLoG-CSUnet라는 새로운 아키텍처를 제안합니다. 이 모델은 의료 이미지에서의 국소 및 글로벌 문맥 이해의 균형을 맞추며, 작은 데이터셋에서의 성능 향상을 목적으로 합니다. GLoG-CSUnet은 학습 가능한 radiomic feature를 통합하여 이미지의 텍스처 및 경계 정보를 효과적으로 캡처합니다.

- **Technical Details**: GLoG-CSUnet의 구조는 Convolutional Swin-Unet (CSUnet) 아키텍처를 확장하고, 가변 성질의 Gabor 및 LoG 필터를 병합하여 구성됩니다. Gabor 변환층은 멀티 스케일 및 멀티 방향 텍스처 정보를 추출하며, LoG 변환층은 경계 정밀도를 향상시키고 에지 감지를 개선하는 역할을 합니다. 이러한 필터들은 훈련 과정에서 최적화되고, 그대로 Transformer에 전달되기 전에 공간적 지역성을 보존하기 위해 비겹치는 패치로 나뉩니다.

- **Performance Highlights**: GLoG-CSUnet을 Synapse 다기관 및 ACDC 심장 분할 데이터셋에서 평가한 결과, 기존 최첨단 모델 대비 Dice score에서 Synapse에서 1.14% 및 ACDC에서 0.99%의 유의미한 성능 향상을 보였습니다. 이 모델은 또한 각각 15개 및 30개의 추가 매개변수로 최소한의 계산 오버헤드를 요구하며, 다양한 기본 모델과의 통합을 통한 유연한 디자인을 가지고 있어 의료 이미지 분석에서의 접목 가능성이 높습니다.



### Hybrid deep convolution model for lung cancer detection with transfer learning (https://arxiv.org/abs/2501.02785)
Comments:
          13 pages, 8 figures

- **What's New**: 이번 연구에서는 전이 학습(transfer learning)을 활용한 하이브리드 딥 컨볼루션 모델인 최대 민감도 신경망(Maximum Sensitivity Neural Network, MSNN)을 소개합니다. MSNN은 폐암 검출의 정확도를 향상시키기 위해 민감도(sensitivity)와 특이성(specificity)을 정제하는 데 중점을 두고 설계되었습니다. 이는 기존의 폐암 검출 모델보다 더 높은 정확도를 목표로 하고 있습니다.

- **Technical Details**: MSNN 모델은 실험적 검증을 통해 98%의 정확도와 97%의 민감도를 달성하였습니다. 이 모델은 폐 컴퓨터 단층촬영(CT) 이미지에 민감도 맵(sensitivity maps)을 오버레이하여 악성(malignant) 또는 양성(benign) 분류에 가장 중요한 영역을 시각화할 수 있게 돕습니다. 이러한 접근 방식은 폐암을 구별하는 데 있어 최소한의 오탐(false positive)으로 뛰어난 성능을 보여줍니다.

- **Performance Highlights**: MSNN은 기존의 딥 러닝(deep learning) 접근 방식을 초월하여 폐암 검출의 정확성을 크게 향상시켰습니다. 이 혁신적인 방법은 의료 진단의 정확도를 높이는 데 기여하며, 조기 및 정확한 진단을 위한 가능성을 제시하고 있습니다.



### Unsupervised Domain Adaptation for Occlusion Resilient Human Pose Estimation (https://arxiv.org/abs/2501.02773)
Comments:
          9 pages, 7 figures

- **What's New**: 이 논문에서는 occlusion(가림) 상황에서도 강인한 인간 자세 추정을 가능하게 하는 비지도 도메인 적응 기법인 OR-POSE를 제안합니다. 기존의 supervised(지도 학습) 기반 방법들이 가진 한계를 극복하고, occlusion이 있는 이미지를 다루는 데 효과적인 접근 방식을 모색하는 데 초점을 맞추고 있습니다. 메인 텍사프레임을 활용한 pseudo-label refinement(유사 라벨 정제)을 통해 domian shift(도메인 전이)의 문제를 해결합니다.

- **Technical Details**: OR-POSE는 Mean-Teacher(평균 교사) 프레임워크를 통해 정확한 키포인트 예측을 위해 self-training(자기 학습) 방법론을 적용합니다. 이 방법론은 occlusion augmentations(가림 보강)를 통해 훈련 데이터의 질을 향상시키고, anatomically plausible(해부학적으로 그럴듯한) 방식을 적용하여 더욱 신뢰할 수 있는 예측을 만들어냅니다. 또한 visibility-based curriculum learning(가시성 기반 커리큘럼 학습) 전략을 통해 학습의 초기 단계에서 더 쉽게 분류될 수 있는 수업을 받도록 합니다.

- **Performance Highlights**: 실험 결과 OR-POSE는 오클루전이 있는 인간 자세 추정을 위한 챌린지 데이터셋에서 기존 성능보다 약 7% 향상된 성과를 나타냅니다. 이는 특정한 상황에서 가장 저조한 성능을 나타내는 기존 도메인 적응 기법들과 비교했을 때 현저한 개선을 보여줍니다. 결과적으로, OR-POSE는 오클루전이 없거나 있는 이미지 모두에서 우수한 성능을 발휘하여, 실생활 적용 가능성을 높입니다.



### WorldPose: A World Cup Dataset for Global 3D Human Pose Estimation (https://arxiv.org/abs/2501.02771)
- **What's New**: WorldPose는 2022 FIFA 월드컵에서 촬영된 다수의 인물에 대한 글로벌 포즈 추정 연구를 촉진하기 위한 새로운 데이터셋입니다. 기존 데이터셋들이 주로 단일 인물에 국한되어 있었던 반면, WorldPose는 다양한 고정 및 이동 카메라로부터 수집된 3D 선수 포즈와 모션을 정확히 복원하여 1.75 에이커 이상의 캡처 영역을 제공합니다. 이러한 인프라를 활용하여 야외 활동 중 다양한 인물의 상대적 위치를 쉽게 파악할 수 있는 데이터셋을 제공하게 되었습니다.

- **Technical Details**: WorldPose는 공통된 Optical 기반의 방법론을 활용하여 정적 카메라 보정을 수행하고, 이를 통해 각 선수의 2D 키포인트를 추적한 뒤, 3D 움직임을 세계 좌표계에서 추정합니다. 본 연구에서는 각 선수의 2D 키포인트를 삼각측량하여 정확한 3D 관절 좌표를 얻습니다. 이후에 이동 방송 카메라의 매개변수를 개선하기 위해 선수의 3D 포즈와 필드 마크를 활용한 추가 제약 조건을 적용합니다.

- **Performance Highlights**: WorldPose 데이터셋은 약 250 만 개의 정확한 3D 포즈 주석을 포함하며, 120 km 이상 이동 거리의 글로벌 선수 궤적을 제공합니다. 평가 결과, Vicon과 비교했을 때 평균 조인트 당 오차는 8 cm로 매우 낮은 성과를 보였습니다. 본 데이터셋 및 평가 기준은 학술 연구 목적으로 공개될 예정이며, 다중 인물 포즈 추정에 대한 새로운 연구를 위한 기초 자료로 활용될 수 있을 것입니다.



### Visual Large Language Models for Generalized and Specialized Applications (https://arxiv.org/abs/2501.02765)
- **What's New**: 본 논문은 최근 시각-언어 모델(Visual-Language Models, VLMs)과 그 진화에 대해 다루고 있습니다. 특히, 비전 대형 언어 모델(Visual Large Language Models, VLLMs)의 다양한 응용 프로그램과 발전 방향에 대한 포괄적인 조망을 제공합니다. VLLMs는 일반화된 응용 프로그램과 특수화된 응용 프로그램 모두에서 활용될 수 있는 잠재력을 지니고 있으며, 그 발전을 위한 과제와 윤리적 고려사항에 대해서도 논의합니다.

- **Technical Details**: VLLMs는 시각 데이터와 언어 데이터의 통합을 통해 다중 작업을 처리하는 등의 성능을 발휘하고 있습니다. 이 논문에서는 다양한 시나리오에서의 VLLM 활용 방법을 분석하고, 비전 데이터에서 언어로의 전환, 언어에서 비전으로의 전환 및 비전 동작 처리와 관련된 세부 작업을 분류합니다. 또한, VLLM의 구조는 기존의 CNN과 RNN(또는 LSTM) 기반 아키텍처에서 벗어나 Transformer와 같은 현대적 심층 학습 기법을 채택하여 성능을 향상시켰습니다.

- **Performance Highlights**: 이 논문은 VLLMs가 제공하는 뛰어난 제로샷 및 전이 학습 성능을 강조하며, 기존 모델들과 비교할 때 성능 향상에 기여하고 있습니다. VLLMs는 인간의 선호에 맞춰 응답을 생성할 수 있는 능력을 갖추고 있으며, 복잡한 비전 작업 처리 능력을 포함합니다. 또한, 윤리적 고려사항과 보안, 개인 정보 보호, 효율성, 해석력 등과 같은 미래 개발 방향이 제안됩니다.



### LDMapNet-U: An End-to-End System for City-Scale Lane-Level Map Updating (https://arxiv.org/abs/2501.02763)
Comments:
          Accepted by KDD 2025, camera-ready version

- **What's New**: 본 논문에서는 LDMapNet-U라는 새로운 패러다임을 제안하여 도시 규모의 차선 수준 지도 업데이트를 효율적으로 수행할 수 있도록 합니다. 이 시스템은 기존의 수동 주석 방식을 개선하고, 도시 전역의 자세한 지도 데이터를 신속하게 생성하는 데 기여합니다. LDMapNet-U는 히스토리컬 맵 데이터와 최신 도로 관찰 이미지를 통합하여 자동으로 표준화된 벡터화된 차선 지도를 생성하므로, 효율적인 데이터 갱신을 가능하게 합니다.

- **Technical Details**: LDMapNet-U 시스템은 Prior-Map Encoding (PME) 모듈 및 Instance Change Prediction (ICP) 모듈을 사용하여 과거 맵 데이터와 최신 관찰 데이터를 효과적으로 조합합니다. PME 모듈은 복잡한 도로 환경에서도 변화를 정확하게 감지할 수 있도록 역사적인 도로 정보를 제공하며, ICP 모듈은 핵심 지리 요소 간의 연관성을 학습하여 변화를 예측합니다. 이로 인해 기하학적, 위상적, 의미적 일관성을 유지하며 동시에 지도 요소를 생성하고 변화를 감지할 수 있습니다.

- **Performance Highlights**: LDMapNet-U는 Baidu Maps에 이미 배포되어 360개 도시에서 주간 업데이트를 지원하며, 업데이트 주기를 분기마다에서 주 단위로 단축하였습니다. 이 시스템은 수억 명의 사용자에게 서비스를 제공하며 자율 주행 시스템에도 통합되어 있어, 최신의 정확한 데이터베이스를 유지하고 있습니다. 실험 결과는 LDMapNet-U의 효과성과 강건성을 입증하며, 이는 현대적이고 자동화된 지도 업데이트 솔루션의 가능성을 보여줍니다.



### Brick-Diffusion: Generating Long Videos with Brick-to-Wall Denoising (https://arxiv.org/abs/2501.02741)
Comments:
          ICASSP 2025

- **What's New**: Brick-Diffusion은 기존의 학습 없는 접근 방식으로, 임의의 길이의 긴 비디오를 생성할 수 있는 혁신적인 방법을 제안합니다. 이 방법은 브릭-투-월(Brick-to-wall) 디노이징 전략을 도입하여 라텐트(latent)를 세그먼트별로 디노이즈하고, 각 디노이즈 과정에서 보폭(stride)을 적용합니다. 이를 통해 각 프레임 간의 소통을 가능하게 하여 전체 비디오 품질을 향상시킵니다. Brick-Diffusion은 기존의 방법들보다 우수한 고충실도(video fidelity)를 제공하는 것으로 평가됩니다.

- **Technical Details**: Brick-Diffusion의 핵심은 브릭-투-월 디노이징 프로세스입니다. 이 방법은 데이터를 f 프레임으로 나누어 각 세그먼트를 개별적으로 디노이즈하며, 이후에는 보폭을 적용하여 라텐트를 재분할합니다. 이 방식은 벽을 쌓는 것과 유사하여 서로 겹치지 않는 방식으로 여러 세그먼트 간의 통신을 촉진합니다. 또한, 기존의 비디오 디퓨전 모델과의 호환성이 높아 효율적인 병렬 처리가 가능합니다.

- **Performance Highlights**: Brick-Diffusion의 성능은 정량적 및 정성적 평가를 통해 입증되었습니다. 이 방법은 영상의 질이 높고, 디테일 및 모션의 일관성을 유지하여 긴 비디오 생성에서 기존 모델들보다 우수한 성능을 보여주었습니다. 실험 결과, 본 연구의 방법이 특히 여러 프레임이 있는 상황에서도 안정적이고 높은 품질의 비디오를 생성할 수 있음을 보여주었으며, 이는 향후 연구 개발의 중요한 토대가 될 수 있습니다.



### Interpretable Recognition of Fused Magnesium Furnace Working Conditions with Deep Convolutional Stochastic Configuration Networks (https://arxiv.org/abs/2501.02740)
- **What's New**: 이 논문은 융합 마그네슘 용광로에서의 작업 조건 인식을 위한 해석 가능한 방법을 제안합니다. 특정한 문제인 일반화 능력 부족과 해석 가능성 문제를 해결하기 위해, 심층 컨볼루셔널 확률적 구성 네트워크(Deep Convolutional Stochastic Configuration Networks, DCSCNs)를 기반으로 한 접근법이 사용됩니다. 새로운 Gaussian differential convolution kernels의 생성과 인크리멘탈 방법을 통해 인식 오류의 수렴을 보장하며, 강화 학습(Reinforcement Learning)을 통해 모델의 압축 및 최적화를 도모합니다.

- **Technical Details**: 논문에서 제안된 DCSCNs 모델은 피직적으로 의미있는 Gaussian differential convolution kernels를 생성하기 위한 감독 학습 메커니즘을 사용합니다. 네트워크의 채널 특징 맵의 독립 계수를 정의하여 융합 마그네슘 용광로의 활성화 맵 시각화 결과를 얻습니다. 인식 정확성, 해석 가능성 관련 평가 메트릭, 모델 파라미터 수를 통합한 보상 함수가 구축되며, 이러한 알고리즘은 해석 가능성과 성능을 동시에 향상시키는 데 초점을 맞추고 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 다른 심층 학습 기법들에 비해 인식 정확성과 해석 가능성에서 우수한 성능을 보여줍니다. 비록 기존의 심층 신경망이 다양한 분야에서 사용되고 있지만, 내부의 비선형 특성과 인간의 개념적 이해 간의 불일치로 인해 해석 가능성 문제에 직면하고 있습니다. 이 연구에서는 각각의 특징 맵에 대한 시각화를 통해 사용자가 결과를 이해하고 모델의 결정을 조정할 수 있도록 해석 가능성을 높였습니다.



### Holistic Semantic Representation for Navigational Trajectory Generation (https://arxiv.org/abs/2501.02737)
Comments:
          Accepted by AAAI 2025

- **What's New**: HOSER(HOlistic SEmantic Representation) 프레임워크는 내비게이션 경로 생성을 위해 고안되었습니다. 이 프레임워크는 도로 및 지역 수준의 의미를 통합하여 경로 생성을 향상시키고, 목적지 지향적 내비게이션 기능을 부여합니다. 기존의 경로 생성 방법들이 저하된 품질을 보였던 문제를 해결하는 데 초점을 맞추었습니다.

- **Technical Details**: 이 연구에서는 𝒢=⟨𝒱,ℰ⟩ 형태의 도로 네트워크를 방향성 그래프로 표현하며, 각 경로는 시공간 점의 시퀀스로 구성됩니다. HOSER은 다중 세분화 다이나믹스 및 도로 수준의 의미를 통합하여 생성된 경로를 인코딩합니다. 모델은 주어진 출발지와 도착지 정보에 따라 경로를 생성하여, 다양한 스케일에서의 의미론적 이해를 강화합니다.

- **Performance Highlights**: HOSER는 세 가지 실제 데이터셋에서 기존 최첨단 방법들보다 우수한 성능을 보였습니다. 특히, Few-shot 및 Zero-shot 학습 시나리오에서도 뛰어난 결과를 나타내어, 이 모델이 다양한 응용에 적합함을 시사합니다. 생성된 경로는 실제 경로를 대체하여 공간-시간 데이터 분석을 위한 훌륭한 잠재력을 가지고 있습니다.



### Multilevel Semantic-Aware Model for AI-Generated Video Quality Assessmen (https://arxiv.org/abs/2501.02706)
- **What's New**: 본 논문에서는 AI 생성 비디오(AIGC)의 비디오 품질 평가(VQA)를 위한 새로운 다단계 모델인 MSA-VQA를 도입합니다. 기존의 VQA 접근법들이 사용자 생성 콘텐츠(UGC)에 초점을 맞추었던 반면, 본 연구는 AI 생성 비디오의 특성을 고려한 평가 방법을 제안합니다. 이 모델은 CLIP 기반의 의미 감독(semantic supervision) 및 교차 주의(cross-attention) 메커니즘을 활용하여 비디오의 품질을 세분화하여 평가합니다.

- **Technical Details**: MSA-VQA는 비디오를 프레임(frame), 세그먼트(segment), 비디오(video) 등 세 가지 수준에서 분석합니다. 각 수준에서 특수 목적의 손실 함수(loss function)를 설계하여 비디오 정보를 종합적으로 포착하는 방식을 사용합니다. 또한, 프롬프트 의미 감독 모듈(Prompt Semantic Supervision Module)을 도입하여 생성된 비디오가 조건부 프롬프트와 일치하는지를 검토하고, 의미 변화를 포착하기 위해 CLIP의 이미지 인코더를 활용하는 세멘틱 뮤테이션 인지 모듈(Semantic Mutation-aware Module)을 제안합니다.

- **Performance Highlights**: 광범위한 실험 결과를 통해 MSA-VQA는 최첨단 성능을 발휘함을 입증했습니다. 이 모델은 다양한 비디오 콘텐츠의 미세한 차이점을 포착하여 AI 생성 비디오의 품질을 더욱 정밀하게 평가할 수 있습니다. 최종적으로, 이러한 접근 방식은 AI 생성 비디오의 품질 평가에 대한 새로운 기준을 제시하여 앞으로의 연구에 기여할 것으로 기대됩니다.



### Underwater Image Restoration Through a Prior Guided Hybrid Sense Approach and Extensive Benchmark Analysis (https://arxiv.org/abs/2501.02701)
Comments:
          Accepted by IEEE TCSVT

- **What's New**: 이 논문에서는 화질 저하 문제를 해결하기 위해 제안된 새로운 색 균형 우선(Color Balance Prior)을 활용한 하이브리드 수중 이미지 복원 프레임워크(Guided Hybrid Sense UIR, GuidedHybSensUIR)를 소개합니다. 이 프레임워크는 세부 정보 복원 모듈(Detail Restorer)과 특징 맥락화 모듈(Feature Contextualizer)을 사용하여 저수준 세부 특징과 고수준 일반 특징을 복원하는 다중 스케일 처리를 수행합니다. 또한, 신뢰할 수 있는 벤치마크 데이터셋을 구축하여 기존의 37개 수중 이미지 복원 방법과 비교하여 우수한 성능을 입증했습니다.

- **Technical Details**: 하이브리드 감각 수중 이미지 복원(Hybrid Sense UIR) 네트워크는 CNN과 Transformer를 조합해 복원 성능을 개선하고자 하였습니다. 이 구조는 U형(유형)의 아키텍처를 채택하여 세밀한 정보와 전반적인 색관계를 효과적으로 처리할 수 있도록 설계되었습니다. 특히, 새로운 색 균형 우선은 Gray World Assumption에 기초하여 원본 이미지의 RGB 채널의 평균값을 기준으로 색 복원 과정을 안내합니다.

- **Performance Highlights**: 실험 결과, 제안한 프레임워크는 다양한 벤치마크 데이터셋과 메트릭에서 37개의 최신 방법에 비해 전반적으로 우수한 성능을 보였습니다. 복원 과정에서 개별적인 사례에서는 최고의 결과를 달성하지 못했으나, 전반적인 품질 개선의 성과를 입증하였습니다. 이 연구는 수중 이미지 복원 분야에서 표준화된 기준과 유용한 데이터셋의 필요성을 충족하고, 향후 연구에 기여할 것으로 기대됩니다.



### EAGLE: Enhanced Visual Grounding Minimizes Hallucinations in Instructional Multimodal Models (https://arxiv.org/abs/2501.02699)
Comments:
          12 pages, 4 figures, 8 tables

- **What's New**: 본 연구에서는 EAGLE라는 새로운 접근법을 제안하여, IT-VLMs에서 발생하는 환각 문제를 시각 인코더의 능력을 향상시킴으로써 해결하고자 합니다. EAGLE는 기존의 모델이나 융합 모듈에 구애받지 않으며, 후속 사전 훈련 방식으로 비전 트랜스포머(ViT)를 최적화합니다. 기존의 대규모 사전 학습된 비전 및 언어 아키텍처의 한계를 극복하여, IT-VLM에서의 이론적 오류를 줄이는 결과를 보여줍니다.

- **Technical Details**: EAGLE는 수정된 대조 학습 프레임워크를 사용하여 로컬 패치 임베딩을 개선합니다. 이를 통해 ViT는 이미지 내 물체에 대한 정제된 정보를 인코딩할 수 있으며, IT-VLM이 사용하는 시각 특성과 언어 정렬을 비교하여 이 효과를 입증합니다. 실험 결과, EAGLE로 최적화된 ViT는 IT-VLM의 기존 비전 인코더를 대체하여 추가 조정 없이도 시각적 응답의 질을 향상시킵니다.

- **Performance Highlights**: EAGLE는 MMVP 및 MERLIM 벤치마크에서 각각 11.2%와 6.3%의 향상을 달성하며, 다양한 IT-VLM 모델에서 환각 현상을 현저히 줄입니다. 본 연구에서 제안하는 EAGLE는 6개의 최첨단 IT-VLM에서 진행된 3개의 도전적인 벤치마크에서 의미 있는 성능 개선을 달성함을 보여줍니다.



### GS-DiT: Advancing Video Generation with Pseudo 4D Gaussian Fields through Efficient Dense 3D Point Tracking (https://arxiv.org/abs/2501.02690)
Comments:
          Project Page: this https URL

- **What's New**: 본 논문은 비디오 생성의 새로운 경향인 pseudo 4D Gaussian fields를 도입하여 비디오 제작에 유용한 4D 제어 기능을 제공하는 GS-DiT라는 프레임워크를 제안합니다. 기존의 방법들이 가진 한계를 극복하기 위해 혁신적인 Dense 3D Point Tracking(D3D-PT) 기법을 사용하여 3D 포인트를 효율적으로 추적하고, 이를 통해 비디오 생성훈련에 필요한 데이터를 생성하는 접근 방식을 채택했습니다.

- **Technical Details**: GS-DiT는 dense 3D point tracking을 통해 pseudo 4D Gaussian field를 구축하고, 이를 원본 비디오 앵글과 함께 렌더링하여 훈련 데이터를 생성합니다. 기존의 GCD와 비교해 GS-DiT는 다중 카메라 비디오 생성이 가능하며, 카메라 내부 파라미터와 객체 움직임 편집 같은 다양한 4D 제어 기능을 지원합니다.

- **Performance Highlights**: GS-DiT는 기존의 방법들보다 더 나은 일반화 성능을 가지고 있으며, 효율적인 D3D-PT 기법 덕분에 정확성과 추적 속도가 현저히 향상되었습니다. 또한, GS-DiT는 비디오 생성 모델로부터 안정적인 비디오 품질을 제공하며, 높은 창의성과 쿠제가 가능한 비디오 제작 도구로 자리 잡고 있습니다.



### Generalizing from SIMPLE to HARD Visual Reasoning: Can We Mitigate Modality Imbalance in VLMs? (https://arxiv.org/abs/2501.02669)
- **What's New**: 이 연구에서는 Vision Language Models (VLMs)의 알고리즘적 시각적 추론 능력을 평가하기 위한 새로운 합성 프레임워크를 제안합니다. VLMs가 시각적 질문 응답 및 이미지 캡셔닝에서 눈부신 성과를 보이지만, 복잡한 이미지에 대한 다단계 추론에는 한계가 있음을 강조합니다. 이 프레임워크에서는 세 가지 과제 - Table Readout, Grid Navigation, Visual Analogy -를 두 가지 난이도로 나누어 성능을 평가합니다.

- **Technical Details**: 제안된 프레임워크에서는 SIMPLE과 HARD 두 가지 난이도의 과제에 대한 훈련 전략을 개발합니다. 우리는 SIMPLE 버전에서 훈련하여 HARD 과제에 대한 성능 향상을 목표로 하는 S2H 일반화 전략을 탐색합니다. 또한, 이미지-텍스트 변환의 명확성이 S2H 일반화 촉진에 중요한 역할을 한다는 점을 강조하며, 다양한 훈련 전략에 따른 결과를 분석합니다.

- **Performance Highlights**: 실험 결과, S2H 일반화에 적합한 훈련 전략을 사용했을 때 S2H 일반화 성능이 유의미하게 향상되었음을 보여줍니다. 그라디언트 정렬(gradient alignment)을 포함한 메커니즘적 연구를 통해, 특정 훈련 전략이 더 나은 S2H 일반화를 촉진하는 경향이 있음을 밝혀냈습니다. 추가적인 ablation 연구에서 각각의 실험 설계 결정이 성능에 미치는 영향을 측정하였으며, 이는 미래의 연구 및 모델 설계에 중요한 기초 자료로 활용될 수 있습니다.



### Tighnari: Multi-modal Plant Species Prediction Based on Hierarchical Cross-Attention Using Graph-Based and Vision Backbone-Extracted Features (https://arxiv.org/abs/2501.02649)
Comments:
          CVPR GeolifeCLEF

- **What's New**: 이번 연구에서는 88,987개의 식물 조사 기록과 함께 위성 이미지, 기후 시계열 데이터, 토지 이용, 생물 기후, 토양 변수와 같은 환경 데이터를 사용하여 식물 종 조합을 예측하는 모델을 개발했습니다. 우리의 모델은 Swin-Transformer 블록을 기반으로 하여 시간적 특성을 추출하고, 다양한 모달리티의 특징들을 효과적으로 융합하기 위해 계층적 크로스 어텐션 메커니즘을 도입하였습니다. 이 연구는 이전의 여러 경쟁 모델들로부터 영향을 받아 예측 정확도를 높이는 여러 전략들을 구사합니다.

- **Technical Details**: 우리는 그래프 구조를 기반으로 한 특징 구축 및 결과 수정 방법을 제안하였습니다. 이 그래프는 SurveyID를 노드로 사용하여 생태적 지조가 비슷한 설문 조사가 인접할 경우 노드를 연결합니다. 모델에 Swin-Transformer를 이용하여 시간적 특성을 추출하고, EfficientNet-B0로 이미지 특징 추출을 대체 함으로써 훈련 속도를 높였습니다. 또한, Top-K 규칙의 후처리 방법을 개선하여 최종 예측의 정확도를 개선했습니다.

- **Performance Highlights**: 아블레이션 실험 결과, 제안된 솔루션 파이프라인이 모델 성능을 크게 향상시킨 것으로 나타났습니다. Tighnari라는 이름의 모델은 환경 보호에 기여하고, 주어진 시공간 맥락에서 식물 종의 조합을 정확히 예측하는 것을 목표로 하고 있습니다. 이 모델은 Transformer 기능을 포함하며, 이미지 프로세싱, 그래프 특징 추출, 계층적 크로스 어텐션 융합 메커니즘을 통합하여 높은 성능을 발휘합니다.



### Multispectral Pedestrian Detection with Sparsely Annotated Lab (https://arxiv.org/abs/2501.02640)
- **What's New**: 이번 연구에서는 Sparsely Annotated Object Detection (SAOD) 문제를 해결하기 위한 새로운 프레임워크인 Sparsely Annotated Multispectral Pedestrian Detection (SAMPD)를 제안합니다. SAMPD는 모드 특성을 기반으로 고품질의 pseudo-label을 생성하고, 누락된 주석을 통합하여 다양한 보행자 비주얼 표현을 학습하도록 돕습니다. 특히, 새로운 Multispectral Pedestrian-aware Adaptive Weight (MPAW) 및 Positive Pseudo-label Enhancement (PPE) 모듈을 도입하여 pseudo-label의 질을 향상시킵니다.

- **Technical Details**: SAMPD는 두 가지 주요 과제를 해결하기 위해 설계되었습니다: 첫째, 고품질 pseudo-label을 효과적으로 학습하기 위한 방법을 찾고, 둘째, 식별된 누락 주석을 학습 중에 통합하여 더 포괄적인 학습을 가능하게 만드는 것입니다. MPAW 모듈은 각 모드의 학습을 조정하여 고품질 pseudo-label에 더 높은 가중치를 부여하고, PPE 모듈은 feature representations을 정렬하여 고품질 pseudo-label 간의 유사성을 높입니다. 또한, Adaptive Pedestrian Retrieval Augmentation (APRA) 모듈을 통해 적절한 ground-truth 보행자 패치를 적극적으로 활용합니다.

- **Performance Highlights**: SAMPD는 실험 결과에서 sparsely annotated 환경에서의 성능을 크게 향상시켰습니다. 고품질 pseudo-label을 생성하고 전화로 결합하는 내용 덕분에, SAMPD는 완전 주석이 있는 시나리오에서도 성능을 개선했습니다. 이로 인해 실제 환경에서도 종종 발생하는 누락된 주석 문제에 효과적으로 대응할 수 있게 되었습니다.



### Identifying Surgical Instruments in Pedagogical Cataract Surgery Videos through an Optimized Aggregation Network (https://arxiv.org/abs/2501.02618)
Comments:
          Preprint. Full paper accepted at the IEEE International Conference on Image Processing Applications and Systems (IPAS), Lyon, France, Jan 2025. 6 pages

- **What's New**: 본 논문에서는 백내장 수술 비디오에서 수술 도구를 실시간으로 식별하는 딥 러닝 모델을 제안합니다. 이는 YOLOV9 아키텍처를 기반으로 하여, Programmable Gradient Information (PGI) 메커니즘과 Generally-Optimized Efficient Layer Aggregation Network (Go-ELAN) 구조를 활용해 정보 병목 문제를 해결하고 있습니다. 이 모델은 615장의 이미지를 사용해 10개의 수술 도구 클래스를 인식하며, 0.5의 IoU에서 73.74의 Minimum Average Precision (mAP)을 달성했습니다.

- **Technical Details**: 제안된 모델은 PGI 메커니즘을 통해 신뢰할 수 있는 그래디언트를 생성하고 정보 손실 없이 네트워크의 깊이를 통과할 수 있습니다. 이 PGI는 주 분기, 보조 가역 분기, 다층 보조 정보를 포함하여 학습 과정의 효율성을 높이는 데 중점을 두고 있습니다. 또한, 일반화된 ELAN (GELAN)을 최적화하여 가벼우면서도 높은 성능의 객체 탐지 모델을 개발하였습니다.

- **Performance Highlights**: Go-ELAN YOLOV9 모델은 기존 YOLO 시리즈 및 기타 탐지 모델들과 비교하여 우수한 성능을 보입니다. 구체적으로, YOLO v5, v7, v8, v9 vanilla, Laptool 및 DETR와의 평가에서 73.74의 mAP를 기록하며 그 효율성을 입증하였습니다. 이는 의학적 교육 자료로서 도구 인식의 정확성을 크게 향상시키는 데 기여할 수 있는 중요한 발전입니다.



### Evolving Skeletons: Motion Dynamics in Action Recognition (https://arxiv.org/abs/2501.02593)
Comments:
          Research report

- **What's New**: 이번 논문은 skeleton-based action recognition(스켈레톤 기반 행동 인식)에서 전통적인 스켈레톤 시퀀스와 Taylor-transformed skeleton(테일러 변환 스켈레톤) 시퀀스를 비교검토합니다. 특히, Spatiotemporal Graph Convolutional Network(ST-GCN)와 Hyperformer의 성능을 평가함으로써, 동적인 움직임을 캡처하는 데 있어 두 모델의 장단점을 분석했습니다. 또한, 움직임의 동역학을 통합한 새로운 모형을 소개하며 이들이 행동 인식에서 갖는 가능성을 강조하고 있습니다.

- **Technical Details**: 본 연구에서는 스켈레톤을 그래프로 표현하여 인체의 관절 간 관계를 파악하는 ST-GCN과 고차원 상호작용을 모델링하는 hypergraph 기반의 Hyperformer를 비교합니다. Taylor-transformed skeletons는 임의의 파생값을 포함시켜 시간적 표현을 강화하며, 특히 복잡한 행동을 구분하는 데 유용합니다. 본 논문은 NTU-60 및 NTU-120 데이터셋을 사용하여 이들 모델의 성능을 평가하고, 정적 포즈 대비 동적 모션을 주입한 포즈의 효과를 비교합니다.

- **Performance Highlights**: 분석 결과, Taylor-transformedd skeletons는 모션 역학을 강화하는 데 효과적임을 보여주었습니다. 그러나 기존 스켈레톤 모델들보다 더 많은 정보를 포함하고 있어 여전히 해결해야 할 도전과제가 존재합니다. 본 연구는 스켈레톤 모델링 기술에 대한 혁신 필요성을 강조하며, 동적인 데이터를 효과적으로 처리하기 위한 접근 방식을 제시합니다.



### Efficient Architectures for High Resolution Vision-Language Models (https://arxiv.org/abs/2501.02584)
Comments:
          Accepted to COLING 2025

- **What's New**: Pheye는 고해상도 이미지를 효율적으로 처리하면서 부족한 매개변수로 훈련되는 새로운 비전-언어 모델(Vision-Language Model, VLM) 아키텍처입니다. 이 모델은 기존의 VLM보다 성능은 유지하면서도 더 높은 효율성을 제공합니다. 특히, 세밀한 이미지 이해 또는 장면 텍스트 처리와 같은 작업에 강점을 보이고 있습니다.

- **Technical Details**: Pheye는 고정된 세팅의 언어 모델과 CLIP 기반의 비전 인코더를 결합하며, 이들 사이의 dense cross-attention 레이어를 통해 상호작용합니다. 비전 인코더는 전역 이미지 및 지역 고해상도 패치를 처리하는 두 가지 LoRA 어댑터 세트를 사용합니다. 모델 설계에서는 효과적인 convergence를 위해 LayerNorm을 사용하며, cross-attention을 통합하여 훈련 가능한 매개변수가 줄어듭니다.

- **Performance Highlights**: Pheye는 가격 대비 경쟁력 있는 모델로서, TextVQA와 같은 장면 텍스트 관련 작업에서 특히 돋보이는 성능을 나타냅니다. 이 모델은 훈련 매개변수가 적음에도 불구하고 높은 해상도의 이미지를 효과적으로 처리하여 다양한 리소스 제한 환경에서도 활용 가능성을 제시합니다. 성능 향상을 위해 자체적으로 제공되는 훈련된 모델과 코드의 GitHub 저장소도 공개되어 있습니다.



### DepthMaster: Taming Diffusion Models for Monocular Depth Estimation (https://arxiv.org/abs/2501.02576)
Comments:
          11 pages, 6 figures, 6 tables

- **What's New**: 본 연구에서는 DepthMaster라는 단일 단계 확산 모델을 제안하여 생성적 특징을 대응적 깊이 추정(task) 작업에 적응시키고자 한다. 기존의 방법들이 생성적 특성과 구분적 특징 간의 갭을 간과하여 최적의 결과를 내지 못한 문제를 해결하는 것을 목표로 한다. 이 모델은 두 가지 모듈, 즉 Feature Alignment와 Fourier Enhancement를 통해 깊이 추정 과정에서 처음은 구조적 정보에 집중하고 후에는 세부 정보를 개선한다.

- **Technical Details**: DepthMaster는 딥러닝 모델의 Feature Alignment 모듈을 통해 생성적 특징이 장식적 특성에 과적합(overfitting)되는 상황을 줄인다. 그리고 Fourier Enhancement 모듈을 통해 단일 단계 틀 내에서 낮은 주파수 구조적 특징과 높은 주파수 세부 특징을 균형 있게 조절하여 깊이 예측의 세부 사항을 향상시킨다. 이 둘의 조합을 통해 모델은 전체 장면 구조를 학습하고 최종적으로 시각적 품질을 개선하는 2단계 훈련 전략을 적용한다.

- **Performance Highlights**: DepthMaster는 다양한 데이터 세트에서 다른 확산 기반 방법들에 비해 우수한 제로샷(zero-shot) 성능과 세부 정보 보존 능력을 보여주었다. 이 방법은 생성적 특징을 맞춤화하여 구분적 깊이 추정 작업에 최적화하였으며, 모델이 일반화와 세부 보존 능력 모두를 향상시키는데 기여한다. 결과적으로, 본 연구는 깊이 추정의 데이터 기반 접근 방식과 모델 기반 접근 방식 간의 간극을 메우는 데 성공하였다.



### Decoding fMRI Data into Captions using Prefix Language Modeling (https://arxiv.org/abs/2501.02570)
Comments:
          4 pages, 2 tables, 1 figure

- **What's New**: 최근 대형 언어 모델과 잠재 확산 모델의 발전으로 뇌 신호를 해독하는 연구가 눈에 띄는 성과를 거두었습니다. 본 연구에서는 기존 GIT 모델의 데이터 오염 가능성을 해결하기 위해 DINOv2 모델의 임베딩을 예측하여 뇌 신호를 이미지 캡션으로 변환하는 새로운 방법을 제안합니다. 이 접근 방식은 GPT-2 언어 모델의 입력으로 [CLS] 토큰을 사용하여 계산 요구 사항을 크게 줄입니다.

- **Technical Details**: 우리는 fMRI 신호에서 DINOv2의 임베딩을 직접 예측하는 새로운 방법을 채택하고, 3D Convolutional Neural Networks (CNN)를 통해 고차원 데이터를 처리합니다. 이를 통해 ROI 마스크 외부의 정보와 복셀 간의 포지셔널 정보를 보다 잘 고려할 수 있습니다. 각 모듈은 별도로 학습되며, DINOv2 임베딩은 MSE 손실을 사용하여 진행됩니다.

- **Performance Highlights**: 우리의 접근법은 기존의 COCO 캡션과 비교하여 METEOR 메트릭에서 우수한 성능을 보여주었으며, 다른 이미지 임베딩으로부터 생성된 캡션과 비교했을 때 6가지 메트릭 중 4가지에서 우수한 결과를 기록하였습니다. 또한, Ridge Regression보다 Wide CNN 아키텍처의 성능이 모든 메트릭에서 우수함을 확인하여 뇌 신호의 해독 효율성을 제고했습니다.



### Balanced Multi-view Clustering (https://arxiv.org/abs/2501.02564)
- **What's New**: 이번 연구에서는 다중 뷰 클러스터링(multi-view clustering, MvC)의 새로운 균형 잡힌 접근법인 BMvC(balanced multi-view clustering)를 제안합니다. 기존의 joint training paradigm의 문제점으로는 특정 뷰의 정보가 학습 과정에서 지배적이 되어 다른 뷰가 제대로 최적화되지 않는 imbalance 현상을 지적하였습니다. 이러한 문제를 해결하기 위해, 각 뷰별 특징 추출기의 최적화를 조절하는 View-Specific Contrastive Regularization(VCR)을 도입하여 클러스터링 과정에서의 성능을 향상시키고자 합니다.

- **Technical Details**: BMvC는 VCR을 통해 클러스터링 분포를 조정하며, 이로 인해 뷰별 특징 추출기가 보다 균형 잡힌 방식으로 학습될 수 있도록 지원합니다. VCR은 통합된 특징과 뷰별 특징에서 캡처한 샘플 유사성을 보존하며, 이를 통해 각 뷰의 그래디언트 크기를 조절하여 최적화 과정을 개선합니다. 이러한 기술적 접근은 클러스터링 성과를 높이며, 다중 뷰 정보를 완전히 활용하는데 기여합니다.

- **Performance Highlights**: BMvC는 8개의 기준 MvC 데이터셋과 2개의 공간적으로 분해된 전사체 데이터셋에서 최신 기법들과 비교해 우수한 성능을 입증하였습니다. 실험 결과, BMvC는 뷰별 특징 추출기로부터 얻어진 클러스터링 성능이 향상되었음을 보여주며, 기존의 joint training 및 single-view 훈련 모델보다 나은 결과를 얻었습니다. 이러한 성과는 BMvC의 클러스터링 성능 개선이 실질적이라는 것을 시사합니다.



### AHMSA-Net: Adaptive Hierarchical Multi-Scale Attention Network for Micro-Expression Recognition (https://arxiv.org/abs/2501.02539)
- **What's New**: 대표적인 얼굴 표정 인식 기술인 Micro-expression Recognition (MER) 분야에서 AHMSA-Net을 통해 새로운 접근법이 제안되었습니다. 본 연구에서는 3D Optical flow 기반의 고정밀 특징 추출 방법을 활용하여 기존 방법들의 한계를 극복하고 있습니다. 이를 통해 다중 스케일(multi-scale) 주의(attention) 메커니즘을 활용하여 미세한 표정 변화를 효과적으로 캡처하고 있습니다.

- **Technical Details**: AHMSA-Net은 적응형 계층적 구조(adaptive hierarchical framework) 와 다중 스케일 주의(mechanism) 모듈로 구성돼 있습니다. 이 구조는 각 레이어에서 Optical flow 특징 맵의 크기를 동적으로 조정하여 미세한 표정 변화를 서로 다른 세분화 수준에서 포착합니다. 또한 각 레이어에서 채널 및 공간적 관심(capturing the complex interactions) 기능을 융합함으로써 더욱 강력한 표정 정보 학습이 가능합니다.

- **Performance Highlights**: AHMSA-Net은 SMIC, SAMM 및 CASMEII와 같은 주요 데이터베이스에서 최대 78.21%의 인식 정확도를 달성했습니다. CASME^3 데이터베이스에서는 77.08%의 정확도를 기록하며 기존 방법에 비해 경쟁력 있는 결과를 입증했습니다. 이러한 결과는 기존의 방법들이 가진 한계점을 극복하는 데 중요한 이정표가 되고 있습니다.



### Pixel-Wise Feature Selection for Perceptual Edge Detection without post-processing (https://arxiv.org/abs/2501.02534)
Comments:
          11 pages

- **What's New**: 이 논문은 기존의 이미지 엣지 감지 모델들이 비효율적인 포스트 프로세싱 기법인 non-maximum suppression(NMS)에 의존하고 있다는 점을 지적하며, 새로운 특징 선택 패러다임을 제시합니다. 이 새로운 패러다임은deep networks에서 픽셀 단위의 차별화된 특징 선택을 가능하게 하며, 기존의 모델에 원활하게 통합될 수 있도록 설계되었습니다. 이를 통해 모델의 성능을 대폭 향상시키면서도 포스트 프로세싱 없이 더 나은 지각적 품질을 제공합니다.

- **Technical Details**: 제안된 모델은 multi-scale CNN 아키텍처와 transformer 프레임워크의 인코더-디코더 구조를 결합하여 픽셀별 가중치를 생성하는 특징 선택기를 포함하고 있습니다. 기존 ED 모델에서 추출된 다중 스케일 특징을 활용하여 정렬된 특징 선택을 수행합니다. 이에 따라, 학습된 픽셀별 특징 선택기는 사용자가 각 픽셀에 대해 가장 관련성 높은 특징을 선택하는 데 기여합니다.

- **Performance Highlights**: 이 논문은 제안된 모델의 성능이 기존 ED 모델보다 뛰어나며, 신뢰할 수 있는 검증 결과를 통해 개선된 수치적 및 지각적 성능을 입증합니다. 대규모 실험 평가를 통해 이 새로운 특징 선택 접근법이 엣지 감지 과제에서 어떻게 중요한 발전을 가져오는지를 잘 보여줍니다. 특히, 포스트 프로세싱 없이도 모델의 정확성과 지각적 품질을 크게 향상시킵니다.



### Vision-Driven Prompt Optimization for Large Language Models in Multimodal Generative Tasks (https://arxiv.org/abs/2501.02527)
- **What's New**: 이번 논문에서는 비전 이해(vision understanding)와 비전 생성(vision generation)을 결합한 혁신적인 프레임워크인 Vision-Driven Prompt Optimization (VDPO)을 제안합니다. VDPO는 Large Language Models (LLMs)를 사용하여 시각 입력으로부터 동적으로 텍스트 프롬프트를 생성함으로써 고충실도 이미지 합성을 유도합니다. 이를 통해 기존 방법보다 높은 성능을 달성하며, 높은 품질의 시각적 결과물을 생성할 수 있도록 돕습니다.

- **Technical Details**: VDPO 프레임워크는 세 가지 주요 구성 요소로 구성되어 있습니다: (1) 시각 임베딩 프롬프트 튜너(visual embedding prompt tuner), (2) 텍스트 기반 지시 생성기(textual instruction generator), (3) 비전 생성 모듈(vision generation module). 이 구성 요소들은 서로 상호작용하여 시각 입력을 일관된 텍스트 프롬프트로 변환한 다음 고품질 이미지를 생성하는 엔드 투 엔드 파이프라인을 형성합니다.

- **Performance Highlights**: 여러 데이터셋에서 진행된 실험 결과 VDPO는 BLEU 및 CIDEr 점수에서 20% 향상된 텍스트 일관성(textual coherence)을 보였으며, Fréchet Inception Distance (FID) 점수에서 15% 감소하는 성과를 달성했습니다. 또한 VDPO는 in-domain 및 out-of-domain 작업에서 강력한 성능을 보이며 다양한 비전 생성 시나리오에 적합한 적응성을 증명했습니다.



### Face-MakeUp: Multimodal Facial Prompts for Text-to-Image Generation (https://arxiv.org/abs/2501.02523)
- **What's New**: 이 연구는 새로운 얼굴 이미지 생성을 위한 Face-MakeUp 모델을 제안하며, 400만 개의 고품질 얼굴 이미지-텍스트 쌍으로 이루어진 FaceCaptionHQ-4M 데이터세트를 구축하였습니다. 텍스트 프롬프트만으로는 복잡한 얼굴 이미지를 생성하는 데 한계가 있기 때문에, 이미지 프롬프트를 활용하여 더 나은 성능을 달성하고자 합니다.

- **Technical Details**: Face-MakeUp 모델은 얼굴 이미지의 다중 스케일 콘텐츠 특징과 포즈 특징을 추출하여 이러한 정보를 diffusion 모델에 통합하여 얼굴 정체성을 보존하기 위해 특징 공간을 확장하는 구조로 설계되었습니다. ArcFace 모델을 활용하여 개별 얼굴 특징을 극대화하고, PoseNet을 통해 포즈 정보를 효과적으로 통합하였습니다.

- **Performance Highlights**: 실험 결과, Face-MakeUp 모델은 다양한 얼굴 관련 테스트 데이터세트에서 최고의 종합적인 성능을 달성하였고, 특히 얼굴의 정체성을 개선하는 데 있어 눈에 띄는 효과를 보여주었습니다. 이 모델은 오픈 소스 자원으로 공개되어 향후 연구 및 실험에 기여할 것으로 기대됩니다.



### Layout2Scene: 3D Semantic Layout Guided Scene Generation via Geometry and Appearance Diffusion Priors (https://arxiv.org/abs/2501.02519)
Comments:
          10 pages, 6 figures

- **What's New**: 이 연구는 텍스트 설명과 시맨틱 레이아웃을 이용하여 3D 장면을 생성하는 새로운 방법인 Layout2Scene을 제안합니다. 기존 방법들이 3D 장면을 생성할 때 객체 위치에 대한 세밀한 제어가 부족한 점을 해결하기 위해, 시맨틱 레이아웃을 활용합니다. 이를 통해 객체의 정확한 위치를 명시할 수 있으며, 생성된 장면은 유연하면서도 정밀한 편집이 가능해 다수의 후속 응용 프로그램에 적합합니다.

- **Technical Details**: Layout2Scene 방법에서는 먼저 하이브리드 장면 표현을 도입하여 객체와 배경을 분리합니다. 초기화는 사전 학습된 텍스트-투-3D 모델을 통해 이루어지며, 이후 두 단계의 최적화 과정으로 장면의 기하학과 외관을 개별적으로 수정합니다. 시맨틱 가이디드 기하학 확산 모델과 시맨틱-기하학 가이디드 확산 모델을 도입해 2D 확산 모델의 장점을 최대한 활용하여 장면의 질감을 개선합니다.

- **Performance Highlights**: 실험 결과, Layout2Scene 방법은 기존의 최첨단 기술보다 더 그럴듯하고 현실적인 장면을 생성할 수 있음을 보여줍니다. 특히, 생성된 장면은 정교한 편집 및 다양하게 활용할 수 있는 기능을 제공합니다. 이 접근 방식은 3D 장면 생성의 새로운 가능성을 열어주며, 다양한 산업 분야에서의 응용 가능성을 제시합니다.



### Facial Attractiveness Prediction in Live Streaming: A New Benchmark and Multi-modal Method (https://arxiv.org/abs/2501.02509)
- **What's New**: 이 논문은 얼굴 매력 예측(Facial Attractiveness Prediction, FAP) 분야에서의 한계를 극복하기 위해, LiveBeauty라는 대규모 라이브특화 FAP 데이터셋을 소개합니다. LiveBeauty는 10,000개의 얼굴 이미지와 200,000개의 매력성을 평가한 주관적 실험 결과로 구성되어 있으며, 이는 라이브 스트리밍 환경에서의 적용 가능성을 염두에 두고 설계되었습니다. 또한, 복합적이고 다각적인 방식으로 매력성을 평가하기 위한 새로운 모델(Facial Prior Enhanced Multi-modal model)을 제안하여 최첨단 성능을 달성했습니다.

- **Technical Details**: LiveBeauty 데이터셋을 구축하기 위해, 저자들은 인기 있는 라이브 스트리밍 플랫폼에서 얼굴 이미지 10,000장을 수집했습니다. 이를 통해 얼굴의 매력성을 예측하는 데 필요한 다양한 매개변수들을 수집하고, 개인화된 매력성 우선 모듈(Personalized Attractiveness Prior Module, PAPM)과 다중 모달 매력성 인코더 모듈(Multi-modal Attractiveness Encoder Module, MAEM)을 통해 얼굴 정보를 세밀하게 분석합니다. 또한, 서로 다른 모달리티에서 추출된 특징을 통합하고 정제하는 교차 모달 융합 모듈(Cross-Modal Fusion Module, CMFM)을 도입하여 더욱 향상된 결과를 도출하고 있습니다.

- **Performance Highlights**: Extensive 실험을 통해 LiveBeauty 및 다른 공개 FAP 데이터셋에서 제안된 방법의 성능이 매우 우수함을 입증하였습니다. 논문에서 제안한 방법은 기존의 최첨단 방법들보다 뛰어난 성능을 보여주며, 특히 라이브 스트리밍 애플리케이션에서의 실제 성능 개선을 기대할 수 있습니다. 최종적으로, 이 연구는 라이브 스트리밍 환경에서 얼굴 매력성 예측의 중요성을 강조하며, 해당 분야에서의 신규 연구 방향을 제시합니다.



### Watch Video, Catch Keyword: Context-aware Keyword Attention for Moment Retrieval and Highlight Detection (https://arxiv.org/abs/2501.02504)
Comments:
          Accepted at AAAI 2025

- **What's New**: 본 논문에서는 비디오 순간 검색(video moment retrieval)과 하이라이트 탐지(highlight detection)에서 키워드 변동을 파악할 수 있는 새로운 모듈인 Video Context-aware Keyword Attention을 제안합니다. 기존의 방법들이 비디오의 맥락을 제대로 반영하지 못하는 문제를 해결하기 위해, 전체 비디오 맥락을 고려하는 비디오 맥락 클러스터링 모듈을 도입합니다. 이를 통해 사용자의 자연어 쿼리에 따라 각각의 단어의 중요성이 달라질 수 있는 점을 잘 반영하게 됩니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 도전에 대응합니다. 첫째, 비디오의 전체 맥락을 효과적으로 인코딩하여 키워드의 변동성을 포착하는 것이며, 둘째, 텍스트 키워드를 원하는 비디오 맥락 내에서 캡처하고 활용하는 것입니다. 이를 위해, 우리는 시기적으로 가중된 클러스터링 기법을 통해 유사한 비디오 장면들을 그룹화하고, 이 클러스터 정보를 통해 키워드 다이내믹스를 이해합니다. 또한 키워드 인지 대조 학습(keyword-aware contrastive learning) 모듈을 통해 시각적 및 텍스트 특징 간의 정밀한 정렬을 가능하게 합니다.

- **Performance Highlights**: QVHighlights, TVSum 및 Charades-STA 벤치마크에 대한 광범위한 실험 결과, 제안된 방법은 기존의 접근 방식들에 비해 순간 검색과 하이라이트 탐지 작업에서 성능이 유의미하게 향상됨을 보였습니다. 특히, 키워드와 비디오 맥락의 관계를 보다 정교하게 이해함으로써 높은 정확도를 달성한 점이 강조됩니다. 이는 다양한 비디오 콘텐츠에 대한 접근성과 사용자 경험을 크게 개선하는 효과를 기대할 수 있습니다.



### ACE++: Instruction-Based Image Creation and Editing via Context-Aware Content Filling (https://arxiv.org/abs/2501.02487)
- **What's New**: 최근 발표된 ACE++는 다양한 이미지 생성 및 편집 작업을 위한 새로운 instruction-based diffusion framework입니다. 기존의 FLUX에 기반한 인페인팅 작업의 입력 형식을 개선하며, Long-context Condition Unit (LCU)에서 발전하여 모든 편집 및 생성 작업을 지원할 수 있게 되었습니다. 이를 통해 강력한 text-to-image diffusion 모델의 미세 조정 과정을 최소화할 수 있는 2단계 학습 방식이 개발되었습니다.

- **Technical Details**: ACE++의 첫 번째 단계에서는 text-to-image 모델의 0-ref 작업 데이터로 모델을 사전 훈련하며, 두 번째 단계에서는 이 모델을 ACE에서 정의된 모든 작업을 지원할 수 있도록 미세 조정합니다. 또한, LCU++라는 향상된 입력 패러다임을 제안하며, 이를 통해 다양한 이미지 생성 작업을 지원합니다. 이 접근 방식을 사용하면 Multi-modal Tasks의 입력을 더 효율적으로 처리할 수 있습니다.

- **Performance Highlights**: ACE++는 이미지 품질과 프롬프트 따르기 능력에서 우수한 성능을 보여줍니다. 또한, 다양한 시나리오에 적용 가능하도록 표준화된 모델 제공하며, 지역 편집 및 전역 재생성 작업 모두에 호환됩니다. 연구 결과는 qualitatively 분석되어 ACE++의 성능을 효과적으로 입증하고 있습니다.



### Noise-Tolerant Hybrid Prototypical Learning with Noisy Web Data (https://arxiv.org/abs/2501.02476)
Comments:
          Accepted by TOMM 2024

- **What's New**: 본 논문은 잠재적 관련성이 있지만 노이즈가 있는 라벨의 웹 이미지를 활용하여 공정한 분류기(classifier)를 학습하는 도전적인 문제에 중점을 두고 있습니다. 기존 방식들이 노이즈가 있는 웹 이미지의 다양한 범위를 충분히 고려하지 않아, 클래스 프로토타입(class prototype)이 심각하게 편향될 수 있습니다. 이에 따라 저자들은 'SimNoiPro'라는 유사성 극대화 손실(similarity maximization loss)을 도입하여 노이즈를 견딜 수 있는 하이브리드 프로토타입을 생성하고 이를 서로 가까이 묶는 방식을 제안합니다.

- **Technical Details**: 저자들은 먼저, 노이즈가 많은 이미지 집합을 몇 개의 그룹으로 나누어 노이즈를 견딜 수 있는 하이브리드 프로토타입을 구축합니다. 그 후, 생성된 프로토타입이 더 컴팩트하고 구별 가능하도록 유사성 극대화 손실을 통해 조정합니다. 이러한 과정을 통해 노이즈 이미지 집합과 깨끗한 이미지 간의 관계 모형화가 개선되며, 이는 최적화 불일치 문제를 극복합니다.

- **Performance Highlights**: 저자들이 제안한 SimNoiPro 방법은 Low-shot Places365와 Low-shot ImageNet 두 개의 벤치마크에서 우수한 성능을 보여줍니다. 특히, 5-shot 환경에서 Low-shot Places365는 4.4%, Low-shot ImageNet에서는 3.5% 더 높은 성능을 기록하였습니다. 이러한 결과는 노이즈가 있는 데이터 세트에서 조심스럽게 정보를 추출하는 데 있어 더욱 유리하다는 것을 입증합니다.



### Generalization-Enhanced Few-Shot Object Detection in Remote Sensing (https://arxiv.org/abs/2501.02474)
- **What's New**: 이 논문은 Generalization-Enhanced Few-Shot Object Detection (GE-FSOD) 모델을 제안하여 리모트 센싱(remote sensing) 환경에서의 few-shot 객체 탐지의 일반화 능력을 향상시키고자 한다. 이 모델은 Cross-Level Fusion Pyramid Attention Network (CFPAN), Multi-Stage Refinement Region Proposal Network (MRRPN), Generalized Classification Loss (GCL) 등 세 가지 주요 혁신을 포함하여 복잡한 객체 특성과 다양한 조건을 고려한 객체 탐지 문제를 해결하고자 한다.

- **Technical Details**: GE-FSOD 모델은 기존 FSOD 모델의 세 가지 구성 요소인 backbone, neck, head 구조를 기반으로 하여 neck, head, loss 컴포넌트를 강화한다. CFPAN은 이중 주의 메커니즘과 크로스 레벨 특징 융합을 통해 다중 스케일의 특징 표현을 향상시키고, MRRPN은 다단계 정제 전략을 통해 지역 제안의 정확성과 효과성을 개선한다. 또한 GCL은 플레이스홀더 노드와 정규화 항을 포함하여 few-shot 분류 작업에서 모델의 일반화 능력을 향상시킨다.

- **Performance Highlights**: DIOR과 NWPU VHR-10 데이터셋에서의 광범위한 실험 결과, GE-FSOD 모델은 리모트 센싱 이미지를 사용한 few-shot 객체 탐지에서 최신 성능을 달성하였다. 이러한 성과는 제한된 데이터 조건에서도 모델의 강건성과 탐지 성능이 크게 개선됨을 보여준다. 본 논문에서 제안한 방법은 재난 모니터링, 환경 보호 등 다양한 분야에서 유용하게 적용될 수 있다.



### DeTrack: In-model Latent Denoising Learning for Visual Object Tracking (https://arxiv.org/abs/2501.02467)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 본 연구는 시각적 물체 추적의 새로운 패러다임인 in-model latent denoising learning을 제안합니다. 기존의 explicit denoising 과정을 여러 개의 denoising block으로 분해하여, 단일 피드 포워드 패스를 통해 문제를 해결하고 실제 애플리케이션에 유용하도록 설계하였습니다. 또한, denoising Vision Transformer를 포함하는 추적 모델을 제시하여, 모델 내에서 점진적으로 denoising을 수행하는 방식을 구현했습니다.

- **Technical Details**: 이 모델은 노이즈 박스를 사용하여 조건 입력을 제공하며, 각 denoising block은 예측된 경계 상자에서 노이즈를 제거하는 역할을 합니다. 또한, 이전 프레임으로부터의 템플릿과 검색 지역 정보를 주입하여 정확한 객체 위치 예측을 수행합니다. 이 과정은 denoising learning 프로세스를 다양한 denoising block 내에서 구현하여 계산 비용을 대폭 낮추는 효과를 가져옵니다.

- **Performance Highlights**: 여러 유명 데이터셋에서 수행된 실험 결과에 따르면, 본 제안된 방법은 AVisT, GOT-10k, LaSOT 및 LaSOText와 같은 데이터셋에서 경쟁력 있는 성과를 달성하였습니다. 연구는 이러한 방법이 기존 방법보다 뛰어난 안정성과 정확성을 제공함을 입증하며, 실시간 물체 추적 응용 프로그램에 적합함을 나타냅니다.



### Depth Any Camera: Zero-Shot Metric Depth Estimation from Any Camera (https://arxiv.org/abs/2501.02464)
- **What's New**: 이 논문은 Depth Any Camera (DAC)라는 새로운 제로샷 메트릭 깊이 추정 프레임워크를 제시합니다. 이 프레임워크는 다양한 시야각(Field of View, FoV)을 가진 카메라, 특히 어안(Fisheye) 및 360도 카메라에 대한 깊이 추정을 가능하게 합니다. DAC는 일반적으로 관찰 가능한 3D 데이터 활용을 극대화하는 방식으로 설계되어 있으며, 기존의 관점 데이터로 훈련된 모델을 기반으로 합니다.

- **Technical Details**: DAC는 Equi-Rectangular Projection (ERP)을 통합 이미지 표현으로 사용하여 여러 FoV 카메라에서 이미지를 일관되게 처리합니다. 주요 기술 중 하나는 ERP 공간에서의 온라인 증강을 위한 피치 감지(Image-to-ERP) 변환이며, 이는 다양한 FoV 간의 효과적인 훈련을 위해 FoV 정렬 작업을 포함합니다. 이 외에도 훈련 이미지 크기의 차이를 관리하기 위해 멀티 해상도 데이터 증강 기법을 도입하고 있습니다.

- **Performance Highlights**: DAC는 여러 피쉬아이 및 360도 데이터셋에서 이전 메트릭 깊이 모델에 비해 최대 50%의 개선된 δ1 정확도를 달성했습니다. 이 결과는 다양한 카메라 유형에 걸쳐 강력한 일반화 능력을 보여줍니다. DAC는 모든 대형 FoV 테스트 데이터셋에서 최신 제로샷 성능을 달성하여 깊이 추정 분야의 새로운 기준을 세우고 있습니다.



### FedRSClip: Federated Learning for Remote Sensing Scene Classification Using Vision-Language Models (https://arxiv.org/abs/2501.02461)
- **What's New**: 본 논문에서는 기존의 Vision-Language Models (VLM)인 CLIP을 기반으로 한 최초의 원격 센싱 이미지 분류를 위한 연합 학습 프레임워크인 FedRSCLIP을 제안합니다. FedRSCLIP은 데이터 이질성과 대규모 모델 전송의 문제를 해결하기 위해 Prompt Learning을 도입하여 적은 수의 조정 가능한 매개변수만을 최적화함으로써 통신 비용을 크게 절감합니다. 또한, 클라이언트에 맞춤형 대응이 가능한 Private Prompts와 전 세계 지식을 공유하는 Shared Prompts를 기반으로 하는 이중 프롬프트 메커니즘을 도입하여 정보의 효율적인 활용이 가능합니다.

- **Technical Details**: FedRSCLIP은 Dual Prompt Alignment Constraint를 활용하여 Shared Prompts와 Private Prompts 간의 의미 일관성을 유지합니다. 이로 인해 각 클라이언트의 로컬 데이터에 적응하면서도 전역 지식과의 일관성을 보장할 수 있습니다. 추가로 Cross-Modal Feature Alignment Constraint를 도입하여 텍스트와 이미지 프롬프트 간의 다중 모드 특성을 정렬함으로써 크로스 모달 표현 학습을 강화하고 전체 모델의 표현의 일관성을 높입니다.

- **Performance Highlights**: FedRSCLIP의 효과를 검증하기 위해 Optimal-31, UCMerced 및 NWPU의 세 가지 기존 데이터 세트를 기반으로 Fed-RSIC 데이터 세트를 구축하여 다양한 연합 학습 구성을 시뮬레이션합니다. 실험 결과, FedRSCLIP은 다양한 연합 학습 구성에서 원격 센싱 이미지 분류의 최첨단 성능을 달성하였으며, 이는 다양한 클라이언트 분포에 걸쳐 안정적이고 일반화된 학습을 가능하게 합니다.



### Neural Reflectance Fields for Radio-Frequency Ray Tracing (https://arxiv.org/abs/2501.02458)
Comments:
          Accepted by IEEE Global Communications Conference 2024 (GLOBECOM'24)

- **What's New**: 이 연구는 복잡한 환경에서 무선 주파수(RF) 신호의 전파를 모델링하기 위해 기존의 레이 트레이싱(ray tracing) 기술과 신경 반사 분야(neural reflectance field)를 결합한 혁신적인 방법을 제안합니다. 이 시스템은 전송자에서 수신자까지의 경로 손실을 기반으로 물질 반사 계수를 학습함으로써 더 정확한 결과를 얻습니다. 또한, 이 접근 방식을 통해 훈련 데이터 수를 크게 줄이면서도 RF 신호의 세기 예측에서 높은 정확도를 달성합니다.

- **Technical Details**: 제안된 방법은 기존의 광학에서 RF 도메인으로 신경 반사 필드를 변환하고, 다층 퍼셉트론(multi-layer perceptron, MLP)을 사용하여 RF 신호의 진폭 및 위상 변화를 모델링합니다. 이를 통해 사용자 장비(UE)의 수신 전력과 3D 장면 기하학을 활용하여 복잡한 야외 환경에서 반사 계수를 학습합니다. 전체 시스템은 전통적인 레이 트레이싱 알고리즘을 사용하여 신호의 세기를 계산하며, 신경 반사 필드는 RF 신호 측정을 일치시키기 위해 최적화됩니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 NeRF2보다 총 수신 전력을 예측하는 데 있어 더 우수한 성능을 보여줍니다. 각 경로의 반사 포인트에서의 감쇠를 정확하게 예측함으로써, 우리 모델은 다양한 통신 및 간섭 채널에 대해 개별적으로 성능을 측정하는 능력을 갖추게 됨으로써 자원 할당 및 스케줄링 작업에 대한 잠재력을 제공합니다. 이러한 결과는 복잡한 환경에서도 효과적으로 적용될 수 있음을 나타냅니다.



### Enhancing Contrastive Learning for Retinal Imaging via Adjusted Augmentation Scales (https://arxiv.org/abs/2501.02451)
- **What's New**: 이 논문은 대조 학습(Contrastive Learning)이 의료 영상(Medical Imaging) 도메인에서 느린 성능을 보이는 이유를 탐구하고 있습니다. 기존의 연구들은 대조 학습이 자연 이미지(domain)에서는 성공적이었으나, 의료 이미지는 고유한 데이터 분포로 인해 이런 성공이 재현되지 않는다는 점을 강조합니다. 논문은 또한 의료 이미지에서 긍정 및 부정 쌍(pair) 구축이 이루어지는 방법을 분석하고 이를 개선하는 방안을 제시합니다.

- **Technical Details**: 대조 학습은 명시적 레이블 없이 유사한 데이터 포인트를 구별하는 기계 학습 패러다임입니다. 본 연구에서는 약한 증강(augmented) 기법을 이용하여 모델을 사전 훈련(pre-trained)하는 방안을 제안하며, 여러 임상 데이터셋을 통해 모델의 일반화(generalization) 성능을 평가합니다. 이 과정에서 두 가지 타입의 데이터 증강(augmentation)을 정의하고 긍정 쌍과 부정 쌍을 생성합니다.

- **Performance Highlights**: 저자들은 약한 증강으로 사전 훈련된 모델이 강한 증강의 경우보다 더 나은 성능을 보임을 확인했습니다. MESSIDOR2 데이터셋에서는 AUROC(Area Under Receiver Operating Characteristic) 값이 0.838에서 0.848로, AUPR(Area Under Precision-Recall Curve) 값이 0.523에서 0.597로 증가하여 강화된 정확도를 보여주었습니다. 이러한 결과는 의료 영상 분야에서의 대조 학습의 효율성을 극대화하기 위해 증강 기법의 최적화가 필수적임을 시사합니다.



### GCP: Guarded Collaborative Perception with Spatial-Temporal Aware Malicious Agent Detection (https://arxiv.org/abs/2501.02450)
Comments:
          15 pages

- **What's New**: 본 논문은 자율 주행 차량 간의 협력적 인식(collaborative perception, CP)의 중요성을 강조하고 있습니다. 특히, 새로운 공격 방식인 blind area confusion (BAC) 공격을 밝혀내어 기존 단일 아울라이어 기반 탐지 방법들이 쉽게 무너질 수 있음을 알리고 있습니다. 이를 극복하기 위해, 공간-시간적 지식을 활용한 GCP(Guarded Collaborative Perception) 프레임워크를 제안하여, 악의적 요원을 효과적으로 탐지할 수 있는 방법론을 제시했습니다.

- **Technical Details**: GCP는 신뢰도 조정된 공간 일관성 손실(confidence-scaled spatial concordance loss)과 LSTM-AE 기반의 저신뢰 지역의 동적 흐름을 재구성하는 방식을 사용합니다. 이러한 기법은 공간 및 시간의 일관성을 동시에 검토하여 잠재적인 악의적 요원을 탐지합니다. 또한, 이들은 듀얼 도메인(anomaly results) 이상 검출을 위한 공동 공간-시간적 Benjamini-Hochberg 테스트를 활용하여 신뢰성을 높입니다.

- **Performance Highlights**: GCP는 다양한 공격 시나리오에서 기존 최첨단 방어 기법에 비해 최대 34.69%의 AP@0.5 성능 개선을 달성했습니다. BAC 공격 하에서도 우수한 성능을 유지하며, 다른 전형적인 공격에서도 5-8%의 일관된 개선을 보였습니다. 이러한 결과는 GCP의 강력한 방어 능력을 입증하며, 코드 또한 공개될 예정입니다.



### MedSegDiffNCA: Diffusion Models With Neural Cellular Automata for Skin Lesion Segmentation (https://arxiv.org/abs/2501.02447)
Comments:
          5 pages, 3 figures

- **What's New**: 이번 연구에서는 Denoising Diffusion Models (DDMs)의 의료 이미지 세분화에서의 성능을 개선하기 위해 세 가지 새로운 Neural Cellular Automata (NCA) 기반 접근법을 제안합니다. 기존의 U-Net 아키텍처에 의존하는 것에서 벗어나, 고해상도 이미지 처리에서 연산 효율을 높이고자 합니다. Multi-MedSegDiffNCA는 다단계 NCA 프레임워크를 사용해 저수준 NCA 모델이 생성한 거친 노이즈 추정치를 세밀하게 다듬고, CBAM-MedSegDiffNCA는 채널 및 공간 주의를 통합하여 세분화를 개선합니다.

- **Technical Details**: 해당 연구에서 제안된 세 가지 모델은 효율성에 중점을 두고 있습니다. 첫 번째는 Multi-MedSegDiffNCA로, 다단계 NCA 구조를 통해 이미지의 노이즈 추정치를 반복적으로 개선합니다. 두 번째는 CBAM-MedSegDiffNCA로, Convolutional Block Attention Module을 사용하여 특징 표현을 향상시키고, 마지막으로 MultiCBAM-MedSegDiffNCA는 두 방법의 장점을 결합하여 새로운 RGB 채널 손실을 통해 의미론적 안내를 제공합니다.

- **Performance Highlights**: 다양한 평가에서 MultiCBAM-MedSegDiffNCA는 Unet 기반 모델과 유사한 Dice 점수 87.84%를 달성했습니다. 특히 파라미터 수를 60-110배 줄일 수 있어, 자원이 제한된 의료 환경에서 훨씬 효율적인 솔루션을 제공합니다. 이를 통해 변화가 필요한 다양한 응용 프로그램에서의 활용 가능성을 높이고 있습니다.



### Unsupervised Search for Ethnic Minorities' Medical Segmentation Training S (https://arxiv.org/abs/2501.02442)
- **What's New**: 이 논문은 의료 영상에서 데이터셋 편향(dataset bias) 문제를 다룹니다. 특히, 인종적 불균형이 문제로 지적되며, 이는 데이터 수집 과정에서 인구 분포의 불균형으로 인해 발생합니다. 제안된 새로운 훈련 세트 검색 전략은 소수 인종 집단을 집중적으로 다루어 의료 모델의 편향을 줄이는 것을 목표로 합니다.

- **Technical Details**: 제안된 방법은 세 단계로 구성됩니다. 첫째, 데이터 풀을 K개 클러스터로 나누어 유사한 데이터 포인트를 구조화된 그룹으로 조직합니다. 둘째, 각 클러스터 그룹과 타겟 도메인 간의 분포 차이를 계산하여 유사성을 정량화합니다. 마지막으로, 이 분포의 차이를 바탕으로 샘플링 점수를 계산하고 최적 훈련 세트를 구성하여 데이터셋의 편향을 줄이고 모델의 공정성을 증대시킵니다.

- **Performance Highlights**: 실험 결과는 제안된 탐색 알고리즘이 임의 선택에 비해 세그멘테이션 정확도 및 공정성에서 일관되게 우수한 성능을 보임을 보여줍니다. FID 점수와 Dice/IoU 지표 모두에서 상당한 개선을 나타내며, 이러한 접근 방식이 의료 영상 모델에서 인종적 불균형을 줄이는 데 효과적임을 입증합니다.



### FOLDER: Accelerating Multi-modal Large Language Models with Enhanced Performanc (https://arxiv.org/abs/2501.02430)
- **What's New**: 본 연구에서는 Multi-modal Large Language Models (MLLMs)의 효율성을 높이기 위해 FOLDER라는 새로운 모듈을 소개합니다. FOLDER는 시각적 토큰(sequence of visual tokens)의 길이를 줄여 훈련과 추론 과정에서의 계산복잡성과 메모리 요구 사항을 완화하는 간단하면서도 효과적인 모듈입니다. 이 모듈은 다양한 MLLMs의 시각적 백본에 통합되며, 최대 70%의 시각적 토큰을 제거하면서도 성능 저하 없이 유사하거나 오히려 나은 성능을 보여줍니다.

- **Technical Details**: FOLDER는 정보 손실을 최소화하는 두 가지 주요 요소를 고려하여 설계되었습니다: Reduction Impact와 Propagation Effect입니다. 이를 바탕으로 FOLDER는 비주얼 인코더의 마지막 블록에서만 대폭적인 토큰 감소를 수행합니다. FOLDER를 적용한 결과, 여러 모델과 기준에서 추론 속도가 1.7배에서 2.4배로 증가했으며, 정보 손실이 거의 발생하지 않거나 심지어 성능이 향상되는 결과를 보였습니다.

- **Performance Highlights**: FOLDER는 MLLM의 훈련에서도 유용성을 발휘하여 성능과 속도를 향상시키는 역할을 합니다. 다양한 표준 벤치마크에서 성능이 크게 개선되었습니다. 결론적으로 FOLDER는 실질적으로 거의 손실 없이 시각적 토큰을 줄일 수 있는 유연하고 효과적인 방법으로, MLLMs의 시각적 토큰 감소 문제를 해결하는 지속 가능한 솔루션을 제공합니다.



### MetaNeRV: Meta Neural Representations for Videos with Spatial-Temporal Guidanc (https://arxiv.org/abs/2501.02427)
Comments:
          Accepted by AAAI2025

- **What's New**: MetaNeRV는 새로운 메타 러닝 프레임워크로서, NeRV 기반 비디오 표현의 효율성을 크게 향상시킵니다. 기존의 NeRV 방식이 각 비디오마다 별도의 모델을 요구하는 반면, MetaNeRV는 다양한 비디오에 적합한 최적의 초기 파라미터를 학습하여 새로운 비디오에 빠르게 적응할 수 있게 합니다. 또한, MetaNeRV는 비디오의 고유한 공간적 및 시간적 특성을 고려하여 공간-시간 가이드를 추가하여 표현 능력을 높입니다.

- **Technical Details**: MetaNeRV는 메타 러닝 알고리즘인 MAML을 기반으로 하며, 다양한 비디오 형태에서 초기 가중치 구성을 생성합니다. 이를 통해 NeRV 모델의 수렴 속도를 크게 향상시키고, 메타 러닝 과정에서 점진적인 학습 전략을 통해 복잡한 비디오에서도 효율적으로 학습할 수 있도록 합니다. 구체적으로, 다중 해상도 손실 기능을 통한 공간적 가이드를 도입하고, 점진적인 훈련 전략을 채택하여 빠른 수렴과 향상된 일반화를 달성합니다.

- **Performance Highlights**: 여러 데이터셋에 대한 실험 결과, MetaNeRV는 비디오 표현에서 다른 프레임 기반 방법들보다 우수한 성능을 보였습니다. 특히, 초기화 파라미터의 최적화가 NeRV 기반 모델의 수렴 속도를 9배 향상시켰고, EchoNet-LVH 데이터셋에서는 PSNR이 +16 향상된 결과를 기록했습니다. 또한, 비디오 압축 및 노이즈 제거 작업에서도 뛰어난 성능을 입증하며, 기존의 H.264 및 HEVC와 비교하여 효과적인 결과를 보여주었습니다.



### Journey into Automation: Image-Derived Pavement Texture Extraction and Evaluation (https://arxiv.org/abs/2501.02414)
- **What's New**: 이 연구에서는 포장 이미지 기반으로 텍스처 기능을 자동으로 추출하고 MTD (Mean Texture Depth)를 평가하는 시스템을 개발했습니다. 본 연구는 경제적인 방법으로 3D 포장 텍스처 데이터를 획득하고, 3D 이미지 처리 기술을 개선하며, 이러한 기능과 MTD 값을 연결하는 다변량 예측 모델을 수립함으로써 더욱 향상된 결과를 제공합니다.

- **Technical Details**: 가장 중요한 세 가지 기여는 먼저, 경제적으로 3D 포장 텍스처 데이터를 획득하는 방법을 제안하였고, 둘째로, 3D 이미지 처리 기술을 개선하여 텍스처의 다양한 측면을 표현하는 기능을 형성하였습니다. 마지막으로, 이러한 기능과 MTD 값과의 관계를 설명하는 다변량 예측 모델을 설정하였습니다. 이 모델은 Gradient Boosting Tree (GBT) 알고리즘을 활용하여 구축합니다.

- **Performance Highlights**: 유효성 검증 결과, Gradient Boosting Tree 모델은 예측의 안정성과 정확성(R2 = 0.9858)에서 뛰어난 성과를 나타냈습니다. 현장 시험 결과, 제안된 방법이 다른 기술들에 비해 상대 오차가 10% 이하로 우수한 성과를 보였습니다. 이 방법은 이미지 입력부터 MTD 예측 출력까지 포장 품질 평가를 위한 포괄적인 end-to-end 솔루션을 제공합니다.



### Guiding Medical Vision-Language Models with Explicit Visual Prompts: Framework Design and Comprehensive Exploration of Prompt Variations (https://arxiv.org/abs/2501.02385)
- **What's New**: 이 논문은 MedVP라는 새로운 프레임워크를 도입하여 의료 이미지를 위한 시각적 프롬프트 생성 및 파인튜닝을 수행합니다. 이는 기존의 일반적인 Vision-Language Models(VLMs)에서 세부적인 정보에 대한 집중이 부족하다는 문제를 해결합니다. MedVP는 의료 개체를 추출하고, 이를 기반으로 시각적 프롬프트를 생성하며, 교육 데이터셋을 조정하여 시각적 프롬프트에 기반한 파인튜닝을 수행합니다.

- **Technical Details**: 이 프레임워크는 입력 이미지에서 의료 개체를 자동으로 추출하고, 이를 활용하여 관심 영역(ROI)을 나타내는 시각적 프롬프트를 생성합니다. 생성된 프롬프트는 박스, 스크리블, 원 등의 다양한 형식으로 불러올 수 있습니다. 이 논문에서는 MedVP-LLaVA라는 모델을 개발하여, 의료 이미지 질의응답(Visual Question Answering, VQA)에서 성능 개선을 평가합니다.

- **Performance Highlights**: MedVP-LLaVA는 여러 의료 VQA 데이터셋에서 최신 스테이트 오브 아트 모델들을 초과하는 성능을 보여줍니다. 우리의 접근 방식은 세밀한 의료 이미지를 이해하는 데 효과적일 뿐만 아니라, 임상적 중요성을 갖춘 결과를 산출합니다. 또한, 이 연구는 의료 VLM을 위한 패러다임을 개척하고, 데이터셋과 모델 가중치를 공개할 계획입니다.



### Generalizable Origin Identification for Text-Guided Image-to-Image Diffusion Models (https://arxiv.org/abs/2501.02376)
- **What's New**: 텍스트 기반 이미지-이미지 확산 모델의 원본 식별(ID$^2$) 작업이 제안되었습니다. 이 작업은 생성된 쿼리의 원본 이미지를 식별하는 데 초점을 맞추고 있으며, 이를 통해 잘못된 정보 확산 및 저작권 침해와 같은 문제를 해결하는데 기여하고자 합니다. 특히, 이를 위한 새로운 데이터셋인 OriPID가 생성되어 다양한 확산 모델에서 모델 학습 및 테스트가 가능합니다.

- **Technical Details**: ID$^2$ 문제를 해결하기 위해, 이 논문에서는 훈련된 Variational Autoencoder (VAE)의 임베딩과 원본 간의 거리를 최소화하는 선형 변환의 존재를 증명합니다. 이후, 이 선형 변환을 통해 서로 다른 확산 모델 간의 일반화 가능성을 입증하였으며, 이론적으로 보장된 방법을 통해 실험적으로 효과성을 확인하였습니다. 또한, 여러 확산 모델에서의 특징 벡터의 일반성으로 인해 원본 식별을 성공적으로 수행할 수 있음을 보여주었습니다.

- **Performance Highlights**: 제안한 방법은 7개 다른 확산 모델에 대해 각각 88.8%, 81.5%, 87.3%, 89.3%, 85.7%, 85.7% 및 90.3% mAP를 달성하며, 기존 유사성 기반 방법들보다 31.6% 높은 성능을 보였습니다. 이 결과들은 기존의 심층 임베딩 모델 및 유사성 기반 접근 방식들이 ID$^2$ 작업에서 만족스러운 성능을 내지 못한다는 것을 입증하며, 제안한 방법이 다양한 모델에 대해 효과적으로 일반화될 수 있음을 나타냅니다.



### V2X-DGPE: Addressing Domain Gaps and Pose Errors for Robust Collaborative 3D Object Detection (https://arxiv.org/abs/2501.02363)
- **What's New**: V2X-DGPE는 V2X 협업 인식에 있어서 도메인 간 격차와 포즈 오류를 해결하기 위해 제안된 새로운 프레임워크입니다. 이 프레임워크는 Knowledge Distillation Framework와 Feature Compensation Module을 활용하여 서로 다른 출처의 데이터로부터 도메인 불변 표현을 학습합니다. 이를 통해 차량과 도로 인프라 간의 특징 분포 격차를 효과적으로 줄이고, 역사적 정보를 통해 현재 장면에 대한 더 포괄적인 이해를 제공합니다.

- **Technical Details**: V2X-DGPE는 다중 소스 데이터의 도메인 불변 표현을 학습하기 위해 Knowledge Distillation Framework를 사용하며, 잔여 네트워크 기반의 Feature Compensation Module을 통해 차량과 인프라 간의 기능 분포 격차를 줄입니다. Collaborative Fusion Module은 이종 self-attention 메커니즘을 활용하여 다양한 차량 및 인프라의 표현을 추출하고 통합합니다. 또한, 변형 가능한 주의 메커니즘을 도입하여 포즈 오류를 처리하며, 입력 특징의 중요한 부분에 동적으로 초점을 맞출 수 있도록 합니다.

- **Performance Highlights**: DAIR-V2X 데이터셋에서의 광범위한 실험을 통해 V2X-DGPE는 현재 최고 성능을 보이는 DI-V2X 모델에 비해 AP@0.7에서 3% 향상된 성능을 보여주었습니다. 다양한 포즈 노이즈 수준에서, V2X-DGPE는 최첨단 성능을 달성하며, Gaussian 노이즈 수준 𝜎𝑡=0.6m, 𝜎𝑟=0.6°에서 CoAlign 접근 방식보다 3% 더 뛰어난 성능을 기록하였습니다. 본 연구는 다중 출처 정보 융합에 있어 도메인 간 격차와 포즈 오류 문제를 효과적으로 해결하는 주요 기여를 하고 있습니다.



### CorrFill: Enhancing Faithfulness in Reference-based Inpainting with Correspondence Guidance in Diffusion Models (https://arxiv.org/abs/2501.02355)
Comments:
          WACV 2025. Project page: this https URL

- **What's New**: 이 논문에서는 이미지 인페인팅(image inpainting) 작업에서 참조 이미지(reference image)의 중요성을 강조하며, 기존 방법들이 참조 이미지와 손상된 타겟 이미지 간의 상관관계(correlation)에 대한 명시적 제약이 부족하다는 문제를 해결하고자 합니다. 이를 위해 CorrFill이라는 훈련이 필요 없는 모듈을 제안하여 참조 이미지와 타겟 이미지 간의 기하학적 상관관계(geometric correlation)에 대한 인식을 향상시킵니다.

- **Technical Details**: CorrFill은 인페인팅 과정에서 자기 주의(self-attention) 레이어에서 주의 마스킹(attention masking)을 이용하여 상관관계 제약(constraints)을 추정하고, 주어진 제약에 따라 입력 텐서(input tensor)를 업데이트하는 목적 함수(objective function)를 사용합니다. 이러한 방식으로 이미지 인페인팅의 과정이 보다 정교하게 조정될 수 있습니다.

- **Performance Highlights**: 실험 결과에서 CorrFill은 여러 기본 diffusion 기반 방법의 성능을 유의미하게 향상시켰으며, 최신 접근 방식(state-of-the-art approaches)과 비교할 때에도 참조 이미지에 대한 충실성을 강조하는 것을 입증하였습니다.



### Accurate Crop Yield Estimation of Blueberries using Deep Learning and Smart Drones (https://arxiv.org/abs/2501.02344)
Comments:
          30 pages

- **What's New**: 이 논문에서는 스마트 드론과 컴퓨터 비전을 활용하여 농작물 수확량 추정을 더욱 정확하게 할 수 있는 AI 파이프라인을 소개합니다. 특히, YOLO(You Only Look Once) 딥러닝 아키텍처 기반의 두 개의 객체 탐지 모델인 Bush Model과 Berry Model을 통해 블루베리 부시와 개별 블루베리의 탐지를 가능하게 합니다. 이를 통해 드론이 최적의 위치에서 이미지를 촬영하여 농작물 수확량을 효율적으로 예측할 수 있게 합니다.

- **Technical Details**: 블루베리 부시를 정확하게 탐지하고 harvest 가능한 블루베리의 수를 계산하기 위해, 스마트 드론은 실시간으로 자신의 위치를 조정할 수 있는 기능을 갖추고 있습니다. 첫 번째 모델인 Bush Model은 낮은 고도에서 블루베리 부시를 탐지하고, 두 번째 모델인 Berry Model은 각 부시에서 개별 블루베리를 탐지합니다. 이 시스템은 RTK positioning을 통해 블루베리 필드를 정밀하게 매핑하고, 각 부시의 위치를 지오태깅(geotagging)하여 데이터 수집의 정확도를 높입니다.

- **Performance Highlights**: 실험 결과, 우리의 모델은 높은 정확도를 보여주었으며, 블루베리 부시의 중심을 기준으로 한 크롭 이미지에서 정밀도와 재현율(precision and recall) 모두에서 좋은 성과를 기록했습니다. 이 연구는 산업적 사용을 목표로 하여, 기존 방법보다 더 정확한 작물 수확량 추정이 가능하도록 돕습니다. 또한, 아주 작은 객체(블루베리) 주석(annotation) 문제와 모델의 효과를 평가하는 데 어려움을 겪은 사례에 대해서도 논의하였습니다.



### RadarNeXt: Real-Time and Reliable 3D Object Detector Based On 4D mmWave Imaging Radar (https://arxiv.org/abs/2501.02314)
Comments:
          8 pages, 5 figures, 3 tables. Code: this https URL

- **What's New**: 본 논문에서는 4D mmWave 레이더 포인트 클라우드를 기반으로 하는 실시간 3D 객체 탐지 시스템인 RadarNeXt를 제안합니다. 기존의 3D 탐지 모델들이 탐지 정확도에 중점을 두었다면, RadarNeXt는 인퍼런스 속도와 효율성을 동시에 고려합니다. 이 네트워크는 re-parameterizable neural networks를 활용하여 메모리 비용을 줄이고, 다양한 스케일의 특징을 잘 포착할 수 있도록 설계되었습니다. 또한, Multi-path Deformable Foreground Enhancement Network (MDFEN)를 통해 레이더 포인트 클라우드의 불규칙한 전경 특징을 강조하고, 배경 혼잡을 억제합니다.

- **Technical Details**: RadarNeXt는 4D mmWave 레이더 포인트 클라우드에서 최적의 성능을 발휘하기 위한 구조로, 특히 sparse data를 처리하는 데 효과적입니다. Rep-DWC(이후 재구성된 PointPillars의 backbone)와 MDFEN 네크(Neck)를 통해 특징 추출의 효율성을 극대화하고, 고품질의 다중 스케일 특징을 확보합니다. 또한, MDFEN은 Deformable Convolution v3 (DCNv3)를 활용하여 객체의 기하학적 표현을 복원하고, 강조된 특징을 통합하는 경로 집합 구조(path aggregation structure)를 통해 알고리즘의 깊이와 강건성을 증가시킵니다.

- **Performance Highlights**: RadarNeXt는 View-of-Delft 및 TJ4DRadSet 데이터셋에서 각각 50.48 및 32.30 mAP를 달성했습니다. RTX A4000 GPU에서 67.10 FPS의 인퍼런스 속도와 Jetson AGX Orin에서 28.40 FPS의 속도를 기록했습니다. 이러한 성과는 RadarNeXt가 경량화된 네트워크 설계를 통해 실제 환경에서도 신뢰할 수 있는 실시간 3D 객체 탐지를 가능하게 함을 보여줍니다.



### Hyperbolic Contrastive Learning for Hierarchical 3D Point Cloud Embedding (https://arxiv.org/abs/2501.02285)
- **What's New**: 이 논문에서는 하이퍼볼릭 공간(hyperbolic space)을 활용하여 다중 모달(multi-modal) 데이터의 복잡한 계층 구조를 효율적으로 모델링하는 방법을 제안합니다. 특히 3D Point Cloud 모달에 대한 기존 연구가 부족했으나, 본 연구를 통해 하이퍼볼릭 멀티모달 대조 학습(multi-modal contrastive pre-training)을 확장합니다. 다양한 모달 간의 지식을 전이하기 위해 개념 관계를 명확히 학습하는 데 주안을 두고 있습니다.

- **Technical Details**: 하이퍼볼릭 공간에서의 대조 학습을 통해 텍스트 텐서(text tensor), 2D 이미지, 3D Point Cloud 간의 계층적 관계를 학습합니다. 이 과정에서 제안된 정규화 기법(regularizer)은 모달 간 계층적 개념 관계를 강화하는데 기여하며, 추상화된 하이퍼볼릭 피쳐를 정의합니다. 실험적으로, 이 접근 방식이 3D Point Cloud 인코더 성능을 크게 향상시키는 것을 보여줍니다.

- **Performance Highlights**: 결과적으로, 본 논문에서는 제안한 방법이 기존 기법보다 우수한 성능을 발휘하며, 다양한 다운스트림 작업(downstream tasks)에서도 현저한 개선을 도모합니다. 연구 결과, 계층적 3D Point Cloud 임베딩은 이미지와 텍스트와의 관계를 포착하여 기존의 모달 학습을 확장할 수 있는 가능성을 보여줍니다. 이러한 성과는 하이퍼볼릭 공간에서의 새로운 대조 학습 전략의 가능성을 시사합니다.



### Efficient Video-Based ALPR System Using YOLO and Visual Rhythm (https://arxiv.org/abs/2501.02270)
Comments:
          Accepted for presentation at the Conference on Graphics, Patterns and Images (SIBGRAPI) 2024

- **What's New**: 이 논문은 자동 차량 번호판 인식(Automatic License Plate Recognition, ALPR) 시스템의 새로운 접근 방식을 제안합니다. 기존의 비디오 기반 ALPR 시스템은 여러 프레임을 사용하지만, 본 연구에서는 단일 이미지에서 차량 번호판 문자를 인식할 수 있는 시스템을 개발했습니다. 이를 위해 YOLO(You Only Look Once) 모델과 Visual Rhythm 기술을 결합하여 각 차량에 대해 정확히 하나의 프레임을 추출하는 방법을 제시합니다.

- **Technical Details**: 우리의 ALPR 시스템은 YOLOv9 객체 탐지 모델을 기반으로 하며, 최신 특징 추출 기능이 통합되어 있습니다. Visual Rhythm 기술을 사용하여 비디오 데이터를 시간-공간 이미지로 축소하고, EasyOCR를 통해 번호판의 문자를 인식합니다. 이 방법의 과정으로는 VR 이미지 생성, YOLO를 이용한 마크 탐지, 해당 프레임의 추출, 그리고 OCR 모델을 통한 텍스트 추출이 포함됩니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 약 15.76%의 문자 오류율(Character Error Rate, CER)을 달성했습니다. 이 성과는 사전 훈련된 모델과 특화된 데이터셋을 사용하여 획득된 것으로, 기존의 방법에 비해 상당한 효율성을 보여줍니다. 향후 연구에서는 Vehicle-Rear 데이터셋에서 YOLO의 훈련을 통해 성능을 더욱 개선할 계획입니다.



### TDM: Temporally-Consistent Diffusion Model for All-in-One Real-World Video Restoration (https://arxiv.org/abs/2501.02269)
Comments:
          MMM2025

- **What's New**: 이 논문에서 제안하는 첫 번째 확산 기반의 올인원 영상 복원 방법론은 사전 훈련된 Stable Diffusion과 정밀 조정된 ControlNet의 힘을 활용합니다. 이 방법론은 다양한 영상 열화 현상을 단일 모델로 복원할 수 있어, 각각의 복원 작업에 특정 모델이 필요한 기존 표준 방법의 한계를 극복합니다. 효율적인 훈련 전술(Task Prompt Guidance, TPG)과 DDIM(가우시안 노이즈 제거) 역변환을 결합한 새로운 Sliding Window Cross-Frame Attention(SW-CFA) 메커니즘을 특징으로 합니다.

- **Technical Details**: 본 연구에서 제안한 Temporally-consistent Diffusion Model(TDM)은 사전 훈련된 Stable Diffusion(SD) 모델과 조정된 ControlNet을 결합하여 개발되었습니다. 훈련 단계에서 TPG를 통해 다양한 복원 작업에 대해 단일 이미지 기반 ControlNet을 정밀 조정하며, 복원 작업에 맞춘 텍스트 프롬프트를 사용할 수 있습니다. 추론 단계에서는 SW-CFA와 DDIM 역변환을 같이 사용하여 콘텐츠 보존 및 시간적 일관성을 보장합니다.

- **Performance Highlights**: 다섯 가지 영상 복원 작업에 대한 광범위한 실험을 통해 제안된 방법론이 기존의 최첨단 방법들보다 실제 세계 데이터에 대한 일반화 성능이 우수함을 입증하였습니다. 또한 TDM은 단일 GPU에서 훈련될 수 있고, 단일 이미지 복원 데이터셋을 이용하여 다양한 영상 복원 작업에 쉽게 적응할 수 있도록 설계되었습니다. 이를 통해 다양한 응용에서 더 향상된 영상 품질을 제공합니다.



### What Kind of Visual Tokens Do We Need? Training-free Visual Token Pruning for Multi-modal Large Language Models from the Perspective of Graph (https://arxiv.org/abs/2501.02268)
Comments:
          9 pages, 6 figures

- **What's New**: 이번 연구는 Multimodal Large Language Models(MLLMs)의 비주얼 토큰(visual tokens) 사용의 필요성을 조사하고, 포어그라운드(foreground)와 배경(background) 토큰이 모두 중요하다는 사실을 밝혔습니다. 새로운 접근 방식인 G-Prune을 제안하여 시각적 토큰을 노드로 간주하고 연결을 구성함으로써 훈련 없이 비트리밍(visual token pruning)을 수행합니다. 이 방식은 정보 흐름을 가중 링크를 통해 전파하여 가장 중요한 토큰을 선택할 수 있도록 돕습니다.

- **Technical Details**: G-Prune은 그래프 기반 방법으로 시각적 토큰을 노드로 간주하고, 피처 거리(feature distance)에 따라 연결을 구성합니다. 정보를 전파하기 위한 반복 알고리즘을 실행하여 각 노드의 중요도를 업데이트합니다. 이를 통해 LLaVA-NeXT에 적용되어, 성능 저하 없이 계산 비용을 크게 줄일 수 있음을 입증하였습니다.

- **Performance Highlights**: G-Prune의 실험 결과, VQA2.0과 TextVQA에서 LLaVA-NeXT의 FLOPs를 각각 63.57% 줄이면서도 정확도는 각각 0.95%와 2.34%만 떨어지는 성능을 유지했습니다. 또한, 다양한 MLLM 벤치마크에서 높은 성능을 유지하며, TextVQA와 같은 세밀한 작업에서도 비교우위를 나타냈습니다. G-Prune은 중요한 응답 정보를 효과적으로 보존하면서 자동으로 시각적 토큰을 선택할 수 있는 가능성을 열어주었습니다.



### Unsupervised Class Generation to Expand Semantic Segmentation Datasets (https://arxiv.org/abs/2501.02264)
- **What's New**: 이 연구는 Stable Diffusion과 Segment Anything Module을 활용하여 새로운 클래스의 샘플과 해당 세그멘테이션 마스크를 생성하는 비지도 학습 파이프라인을 제안합니다. 이는 기존의 시뮬레이터나 비디오 게임을 사용한 합성 데이터의 한계를 극복하며, 사용자 입력을 최소화하여 효과적으로 작동합니다. 또한, 모델이 새로운 클래스를 학습할 수 있는 가능성을 확장하고, 이미 존재하는 클래스의 성능 또한 향상시킬 수 있음을 보여줍니다.

- **Technical Details**: 우리의 방법론은 비지도 도메인 적응(unsupervised domain adaptation) 기법을 통해 새로운 클래스의 데이터를 기존 데이터셋에 통합합니다. Segment Anything Model(SAM)은 다양한 입력 프롬프트를 사용하여 제로샷 세그멘테이션을 수행할 수 있도록 설계되었습니다. 이런 접근은 세그멘테이션 알고리즘의 구조를 수정하지 않고도 새로운 클래스들을 포함할 수 있는 가능성을 제시합니다.

- **Performance Highlights**: 제안된 방법을 통해 모델은 새로운 클래스를 평균 51% IoU(Intersection over Union) 성능으로 성공적으로 학습할 수 있으며, 기존 클래스의 오류도 감소시키는 결과를 보여줍니다. 이러한 성능 향상은 비지도 도메인 적응 파이프라인의 효과를 강조하며, 실제 데이터 적용에서 우수한 성능을 나타냅니다.



### MagicFace: High-Fidelity Facial Expression Editing with Action-Unit Contro (https://arxiv.org/abs/2501.02260)
- **What's New**: 이 논문은 얼굴 표정을 편집하는 새로운 방법을 제안합니다. 'MagicFace'라는 모델을 통해 오직 동일인에 대한 감정 변화를 조절하여 지속적이고 세련된 방식으로 얼굴 표정을 수정할 수 있습니다. 이 모델은 얼굴의 정체성, 자세, 배경을 유지하면서도 가장세부적인 얼굴 특성을 명확하게 표현할 수 있도록 설계되었습니다.

- **Technical Details**: MagicFace는 Facial Action Units (AUs)를 활용하여 세밀한 얼굴 표정 편집을 가능하게 합니다. 이 모델은 프리트레인된 Stable-Diffusion 모델과 ID 인코더를 결합하여 입력된 얼굴의 세부 사항을 유지하며, 다양한 AUs 조합을 통한 자유로운 애니메이션을 실현합니다. 또한 모델은 효율적인 Attribute Controller를 도입하여 목표의 배경과 자세를 명확히 알게 하여 일관성을 유지합니다.

- **Performance Highlights**: 제안된 방법은 기존 얼굴 표정 편집 기술에 비해 질적으로 우수한 결과를 보여줍니다. 다양한 사람, 배경 및 AUs 조합을 통해 높은 충실도의 표정 편집이 가능함을 실증하며, 특히 사용자 친화적인 조작성을 강조하고 있습니다. 논문은 결과를 정량적 및 정성적으로 평가하여 보다 해석 가능하고 유연한 편집 방식을 제시합니다.



### Distillation-Enhanced Physical Adversarial Attacks (https://arxiv.org/abs/2501.02232)
Comments:
          7 pages, 5 figures

- **What's New**: 본 논문에서는 지식 증류(knowledge distillation)를 활용한 새로운 물리적 적대 공격 방법을 제안합니다. 이를 통해 공격 성능과 스텔스성(stealthiness) 간의 균형을 이루는 것을 목표로 하고 있습니다. 이번 연구는 스텔시한 색상을 최적화한 적대적 패치를 생성하여, 실세계 환경에 쉽게 통합되도록 합니다. 이 접근법은 기존 방법들보다 20% 이상의 공격 성능 향상을 보여주며, 스텔스성을 유지하는 것을 강조합니다.

- **Technical Details**: 제안하는 метод은 두 가지 주요 구성 요소로 이루어져 있습니다. 첫째, 목표 환경에 맞춘 색상 분포를 추출하여 이를 최적화 공간으로 활용하여 스텔시한 적대적 패치를 생성합니다. 둘째, 색상 제한이 없는 적대적 패치로부터 특성을 전이하여 스텔시한 패치의 공격 효과를 향상시키는 지식 증류 모듈을 포함합니다. 이는 Kullback-Leibler divergence를 사용하여 환경의 색상 분포와 더 잘 맞는 색상 집합을 형성하는 최적화 문제로 구성됩니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 비지식 증류 방법에 비해 공격 성능을 20% 이상 향상시키는 것을 확인했습니다. 이러한 성과는 스텔시한 적대적 패치 생성에서 공격 능력과 은폐성을 동시에 달성했음을 보여줍니다. 여러 전통적인 탐지 모델을 통해 검증된 이 방법은 실제 환경에서의 적용 가능성을 더욱 높입니다.



### Examining the Robustness of Homogeneity Bias to Hyperparameter Adjustments in GPT-4 (https://arxiv.org/abs/2501.02211)
- **What's New**: 이 연구는 GPT-4의 hyperparameter 조정이 동질성 편향(homogeneity bias)에 미치는 영향을 탐구합니다. 특히 sampling temperature와 top p를 조정하여 모델의 출력 랜덤성을 통제하고, 다양한 인종 및 성별 집단에 대한 이야기 생성의 유사성을 평가합니다. 이 과정에서 동질성 편향이 지속적으로 나타나고, hyperparameter 간의 관계가 비선형적인 패턴을 보인다는 중요한 발견을 했습니다.

- **Technical Details**: 연구 방법론은 세 가지 단계로 나뉩니다. 첫째, 네 가지 교차 집단(Black men, Black women, White men, White women)을 대표하는 얼굴 자극을 선택했습니다. 둘째, Vision-Language Model (VLM)과 연구 질문에 맞춰 조정한 hyperparameters를 설명합니다. 셋째, 혼합 효과 모델(mixed-effects models)을 사용하여 hyperparameter 값에 따른 동질성 편향의 크기를 비교하는 분석 전략을 세웠습니다.

- **Performance Highlights**: 연구 결과, 동질성 편향은 대부분의 hyperparameter 구성에서 지속되며, Black Americans와 여성은 White Americans와 남성보다 더 동질적으로 표현됩니다. 또한, Temperature를 높이거나 Top p를 낮추는 것은 인종 동질성 편향을 줄일 수 있지만, 성별 동질성 편향에 대한 영향은 상이합니다. 이는 hyperparameter 조정이 특정 편향을 완화할 수 있지만, 모든 사회 집단 차원에서 동질성 편향을 해결하는 보편적인 솔루션 역할을 할 수는 없음을 시사합니다.



### Self-Supervised Learning for Detecting AI-Generated Faces as Anomalies (https://arxiv.org/abs/2501.02207)
- **What's New**: 이 논문은 AI로 생성된 얼굴을 탐지하는 새로운 방법을 제시합니다. 기존의 Binary classification(이진 분류) 접근 방식이 아닌 Anomaly detection(이상 감지) 방법을 통해 AI 생성 얼굴 이미지를 식별하고자 합니다. 이를 위해 Self-supervised learning(자기 지도 학습) 기술을 활용하여 카메라 고유 및 얼굴 특화 기능을 학습합니다.

- **Technical Details**: 제안된 방법은 사진 얼굴 이미지의 특징을 추출하여 Gaussian mixture model (GMM)을 통해 AI로 생성된 얼굴을 분류합니다. 특히, Exchangeable image file format (EXIF) 데이터를 활용하여 네 가지 태그의 순위를 매기는 프리텍스트 작업을 수행합니다. 이와 함께 얼굴 조작을 분류하기 위한 예측 머리도 사용됩니다.

- **Performance Highlights**: 여아(AI) 얼굴 이미지를 탐지하기 위한 본 방법은 총 9개의 최신 생성 모델에 대한 실험을 통해 그 효과가 입증되었습니다. 정량적 및 정성적 실험 모두에서 뛰어난 성능을 나타내어 AI로 생성된 얼굴을 효과적으로 식별할 수 있음을 확인했습니다. 이러한 접근 방식은 다양한 AI 생성기로부터의 일반화 문제를 해결하는 데 기여할 것으로 기대됩니다.



### Accounting for Focus Ambiguity in Visual Questions (https://arxiv.org/abs/2501.02201)
- **What's New**: 이 논문은 기존의 시각 질문 응답 시스템(Visual Question Answering, VQA)에서 언급하지 않았던 질문의 모호성(focus ambiguity)에 대한 문제점을 다룹니다. 이를 해결하기 위해 VQ-FocusAmbiguity라는 새로운 데이터셋을 소개하며, 이 데이터셋은 질문에서 지칭되는 콘텐츠와 관련된 모든 이미지 영역을 시각적으로 명확하게 구분합니다. 이 데이터셋은 4,357개의 예제로 구성되어 있으며, 모호성이 있는 질문과 없는 질문 간의 균형을 맞추고 있습니다. 또한, 이 데이터셋은 현대 모델들이 새로운 두 가지 과제에서 어떻게 성능을 발휘하는지를 평가합니다.

- **Technical Details**: VQ-FocusAmbiguity 데이터셋은 이미지, 질문, 그리고 질문이 지칭할 수 있는 모든 영역의 세분화(segmentation)로 구성되어 있습니다. 데이터는 다양한 출처에서 확장하여 생성되었으며, 질문은 모호한 표현부터 구체적인 설명까지 다양한 주제를 포함합니다. 저자들은 PACO-LVIS 및 MSRA-B와 같은 멀티모달 데이터셋을 비유적으로 활용하여 AI 모델이 모호성을 인식하고 이미지 내 모든 가능한 집중 영역을 찾아내는 능력을 평가합니다.

- **Performance Highlights**: 실험 결과, 현대 모델들은 제안된 두 가지 새로운 과제를 수행하는 데 어려움을 겪었습니다. 첫 번째는 질문에 모호성이 있는지 인식하는 것이고, 두 번째는 질문이 지칭하는 모든 이미지 영역을 정확하게 찾아내는 것입니다. 이 데이터셋은 이러한 과제에 대해 도전적임을 보여주며, 향후 연구와 진전을 위한 평가 서버를 공개하여 더 나은 VQA 모델을 개발하는 데 기여하고자 합니다.



### Benchmark Evaluations, Applications, and Challenges of Large Vision Language Models: A Survey (https://arxiv.org/abs/2501.02189)
Comments:
          34 pages, 3 figures

- **What's New**: 이번 논문은 최근 5년간(2019-2024) 개발된 멀티모달 비전 언어 모델(vision language models, VLM)에 대한 종합적인 개요를 제공합니다. 현재 VLM의 주요 구조, 교육 방법, 평가 기준 및 다양한 응용 분야를 체계적으로 정리하며, 특히 VLM 연구에 관심 있는 학술 연구자들에게 유용한 정보를 제공합니다. VLMs는 시각적 및 텍스트 입력을 결합하여 더욱 깊이 있는 이해를 가능하게 하는 독창적인 기술입니다.

- **Technical Details**: 이 논문에서는 VLMs의 주요 구성 요소와 훈련 방법에 대한 설명을 제공합니다. 특히, 이미지와 텍스트 정보를 정렬하기 위해 사전 훈련된 대규모 언어 모델(large language models, LLM)을 백본으로 사용하는 경향이 증가하고 있으며, 이는 VLM이 비주얼 콘텐츠를 더 잘 이해하도록 돕습니다. VLM의 훈련 목표와 아키텍처에 대한 주요 연구 방향을 세 가지로 나누어 설명하며, CRIP, BLIP, LLaMA와 같은 모델들이 있습니다.

- **Performance Highlights**: VLMs는 비전 인식 작업에서 뛰어난 성과를 보이며, 제로샷(Zero-shot) 분류에서도 기존의 단일 모달 모델을 넘어서는 성능을 보여줍니다. 또한, VLM의 활용 사례로는 자율주행, 로봇공학, 비디오 생성 등이 있으며, 시각적 질문 답변(visual question answering) 같은 복합적인 작업을 가능하게 합니다. 그러나 시각적 환각, 공정성, 안전성 문제와 같은 새로운 도전 과제가 있으며, 이러한 도전들은 멀티모달 모델의 발전을 위한 중요한 연구 주제로 급부상하고 있습니다.



### Phase Retrieval by Quaternionic Reweighted Amplitude Flow on Image Reconstruction (https://arxiv.org/abs/2501.02180)
- **What's New**: 이번 연구에서는 quaternion 알gebra(대수학)를 사용한 신호 처리 방법으로, 신호 차원 간의 본질적인 상관 관계를 보존하여 색 신호를 효율적으로 관리하는 기술을 제안합니다. 특히, 새로운 알고리즘인 Quaternionic Reweighted Amplitude Flow(QRAF)와 그 변형 알고리즘을 개발하여 복잡한 계산 성능을 개선했습니다. 기존 방식에 비해 더욱 개선된 회복 성능과 계산 효율성을 보여줌으로써, 다양한 분야에서 적용 가능성을 지니고 있습니다.

- **Technical Details**: 본 논문은 amplitude-based model(진폭 기반 모델)을 사용하여 quaternionic phase retrieval(쿼터니온 위상 회수) 문제를 시스템적으로 다루고 있으며, 새로운 QRAF 알고리즘 및 그 변형 알고리즘(증분, 가속화 및 적응형)을 제시합니다. 추가적으로, QPAF(쿼터니온 섭동 진폭 흐름) 알고리즘도 도입하여 선형 수렴 특성을 보여주며, 여러 수치 실험을 통해 진폭 기반의 복잡한 신호 회수 문제를 해결합니다. 이 알고리즘들은 대칭 데이터와 실제 이미지에서 적용되어 기존 최첨단 방법들보다 우수한 성능을 보였습니다.

- **Performance Highlights**: 수치 실험 결과, 제안된 QRAF와 QPAF 알고리즘은 기존의 여러 위상 회수 알고리즘보다 뛰어난 회복 성능을 보이며, 특히 색신호 처리에서 높은 계산 효율성을 자랑합니다. 이러한 알고리즘은 기존의 경량 알고리즘들보다 단순한 기법으로 신호를 재구성할 수 있어, 다양한 실제 이미지 데이터에서도 뛰어난 결과를 냈습니다. 따라서 이 연구는 고차원 설정에서의 위상 회수 문제 해결에 있어 기술적 진전을 이루었음을 알 수 있습니다.



### Generating Multimodal Images with GAN: Integrating Text, Image, and Sty (https://arxiv.org/abs/2501.02167)
- **What's New**: 이번 연구에서는 Generative Adversarial Networks (GAN)에 기반한 새로운 다중 모달 이미지 생성 방법이 제안되었습니다. 이 방법은 텍스트 설명, 참조 이미지 및 스타일 정보를 효과적으로 통합하여 다중 모달 요구 사항을 충족하는 이미지를 생성할 수 있습니다. 또한, 텍스트 인코더, 이미지 특징 추출기 및 스타일 통합 모듈의 설계를 포함합니다.

- **Technical Details**: 제안된 방법은 텍스트 인코더와 이미지 특징 추출기뿐만 아니라 스타일 통합 모듈을 포함하여 다양한 요소로 구성됩니다. 생성 과정에서 적대적 손실(adversarial loss), 텍스트-이미지 일관성 손실(text-image consistency loss), 그리고 스타일 매칭 손실(style matching loss) 등 여러 손실 함수가 도입되어 최적화 과정을 지원합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 여러 공개 데이터 세트에서 높은 선명도와 일관성을 가진 이미지를 생성하며, 기존 방법들과 비교하여 상당한 성능 향상을 보여줍니다. 이번 연구의 결과는 다중 모달 이미지 생성에 대한 새로운 통찰력을 제공하고, 폭넓은 응용 가능성을 제시합니다.



### Joint Optimization for 4D Human-Scene Reconstruction in the Wild (https://arxiv.org/abs/2501.02158)
Comments:
          Project Page: this https URL

- **What's New**: 이번 연구는 웹 비디오를 통해 자연스럽고 다양한 인간 움직임과 장면 맥락을 재구성하는 JOSH라는 새로운 최적화 기반 방법을 제안합니다. JOSH는 밀집한 장면 재구성과 인간 메시 복구 기법을 초기화 방법으로 사용하고, 사람과 장면의 접촉 제약을 통해 장면, 카메라 포즈, 인간 움직임을 공동으로 최적화합니다. 이 방법은 전통적인 방법들과는 다르게 고정된 환경이 아닌 다양하고 실제적인 환경에서 인간-장면 상호작용을 재구성하는 데 초점을 맞추고 있습니다.

- **Technical Details**: JOSH는 모노큘러 비디오로부터 4D 인간-장면 재구성을 수행하며, 시나리오의 기하학과 인간의 움직임을 동시에 최적화합니다. 핵심 통찰력은 인간-장면 접촉이 장면과 인간의 움직임을 연결하는 강력한 제약으로 작용하여 더 정확하고 일관된 결과를 생성한다는 것입니다. JOSH3R는 이러한 최적화 과정을 간소화하여 상대적인 인간 변환을 직접 예측하는 경량 모델로 설계되어 실시간 추론을 가능하게 합니다.

- **Performance Highlights**: EMDB, SLOPER4D 및 RICH 데이터셋에서의 실험 결과는 JOSH가 4D 인간-장면 재구성의 정확성과 일관성을 높이는 데 기여한다는 것을 보여줍니다. JOSH는 기존 방법들과 비교하여 EMDB 데이터셋의 전역 인간 추정에서 최첨단 성능을 기록했으며, JOSH3R는 다른 최적화이 없는 방법들보다 우수한 인간 궤적 추정을 실현했습니다. 이러한 결과는 JOSH의 정확성과 일반화 능력을 더욱 입증합니다.



### From Images to Detection: Machine Learning for Blood Pattern Classification (https://arxiv.org/abs/2501.02151)
- **What's New**: 이번 연구에서는 총 169개의 혈흔 패턴을 바탕으로 총 68개의 총격 배출혈과 61개의 충격형 패턴을 구분하는 모델을 개발했습니다. 특히, 머신러닝 기법 중 XGBoost를 적용하여 기존의 연구에서는 다루지 않은 새로운 활용 방안을 제시하였습니다. 이 외에도 새로운 중요도 측정 지표인 안정성 중요도 점수(SIS)를 도입하여 특징의 중요도를 보다 일관되게 평가할 수 있도록 하였습니다.

- **Technical Details**: 본 연구에서는 혈흔 패턴을 분석하기 위해 MATLAB R2023a를 통해 혈흔 이미지의 전처리 과정을 진행했습니다. 혈흔 패턴은 거의 타원형 모양으로, 타원 적합성을 통해 각 혈흔의 그래픽적 특성을 추출했습니다. 이후, 특징을 기반으로 랜덤 포레스트(random forest)와 XGBoost를 활용하여 분류기를 훈련시켰습니다.

- **Performance Highlights**: 모델 개발 결과, 정확도와 효율성 두 가지 측면에서 우수한 성능을 발휘하는 혈흔 패턴 분류 모델을 구축하였습니다. XGBoost의 도입으로 결측값 처리와 빠른 계산 속도의 이점을 얻게 되었으며, 새로운 SIS 지표는 특징들의 중요성을 더욱 명확하게 제공해줍니다.



### Plasma-CycleGAN: Plasma Biomarker-Guided MRI to PET Cross-modality Translation Using Conditional CycleGAN (https://arxiv.org/abs/2501.02146)
Comments:
          Accepted by ISBI 2025

- **What's New**: 이번 연구에서는 혈액 기반 바이오마커(BBBMs)를 MRI에서 PET 이미지로의 변환 모델에 통합하는 효과를 조사했습니다. 이를 통해 BBBMs를 조건으로 하여 PET 이미지를 합성하는 새로운 방법, Plasma-CycleGAN을 제안합니다. 이는 MRI와 PET 간의 조건부 변환에 BBBMs를 도입한 최초의 접근 방식입니다. 연구 결과, BBBMs의 통합이 모든 모델에서 생성 품질을 지속적으로 향상시킨 것으로 나타났습니다.

- **Technical Details**: 연구에서 사용된 데이터는 알츠하이머병 신경영상 이니셔티브(ADNI)에서 수집된 이미지로, 총 1338개의 이미지를 포함합니다. 이들은 256x256x256 크기의 3D 복셀 큐브로 처리되었으며, 효율적인 훈련을 위해 128x128x128로 다운샘플링되었습니다. 데이터 증강 기법으로는 가우시안 노이즈 추가, 회전, 플립, 밝기 및 대비 변화 등을 적용하였고, CycleGAN을 기반으로 한 조건부 생성적 적대 신경망(cGAN)도 사용되었습니다.

- **Performance Highlights**: CycleGAN을 통한 다양한 생성 결과를 시각적으로 검토한 결과, 생성된 PET 이미지에서 가장 뛰어난 시각적 충실도를 나타낸 것으로 평가되었습니다. 연구는 BBBMs의 통합이 PET 이미지의 합성 품질에 긍정적인 영향을 미친다는 점을 보여주었으며, 향후 알츠하이머병 진단에 중요한 역할을 할 가능성이 있습니다. Plasma-CycleGAN은 기존의 대조군 모형들에 비해 이미지 품질을 개선하는 성과를 기록하였습니다.



### SafeAug: Safety-Critical Driving Data Augmentation from Naturalistic Datasets (https://arxiv.org/abs/2501.02143)
- **What's New**: 본 연구에서는 진정한 자율주행 시스템을 위한 데이터 강화 기법을 제안합니다. 이 기법은 자연istic dataset에서 안전-critical driving data를 보강하여 자율주행 모델의 훈련을 개선하는 데 목적이 있습니다. 차량 감지, 깊이 추정 및 3D 변환 기술을 통해 위험한 상황을 시뮬레이션하고 데이터의 진정성을 유지하면서도 뛰어난 성능을 발휘하도록 합니다.

- **Technical Details**: 본 연구에서 제안된 방법은 YOLOv5를 사용하여 차량을 감지한 후, 깊이 추정을 통해 3D 모델로 변환하는 과정을 포함합니다. 이러한 3D 모델을 조정하여 차량 사이의 거리와 같은 현재의 중요한 시나리오를 시뮬레이션합니다. 최종적으로, 데이터 향상을 위해 원본 영상과 증강 영상을 결합하여 자율주행 알고리즘의 성능을 극대화합니다.

- **Performance Highlights**: KITTI 데이터셋을 사용한 실험에서, 제안된 증강 방법을 통해 훈련된 자율주행 알고리즘이 기존의 기법들보다 더 우수한 평가 결과를 보였습니다. 특히, SMOGN 및 중요 샘플링과 같은 기존 방법 대비 높은 안전도와 신뢰성을 유지하며, 실제 환경에서의 예측 정확성을 크게 향상시켰습니다.



### AVTrustBench: Assessing and Enhancing Reliability and Robustness in Audio-Visual LLMs (https://arxiv.org/abs/2501.02135)
- **What's New**: 최근 Multi-modal Large Language Models (MLLMs)의 빠른 발전에 따라, 이러한 모델들의 다중 모드 추론 능력을 평가하기 위한 새로운 진단 벤치마크가 개발되었습니다. 하지만 기존 벤치마크는 주로 시각적 요소에 제한되어 있어 오디오-비주얼(AV) 이해를 전체적으로 평가하지 않습니다. 본 연구에서는 Audio-Visual Trustworthiness assessment Benchmark (AVTrustBench)를 도입하여 600K 샘플로 9개의 세밀하게 설계된 작업을 포함하고, AVLLMs의 응답을 보정하는 능력을 조사합니다.

- **Technical Details**: AVTrustBench는 세 가지 차원(Adversarial attack, Compositional reasoning, Modality-specific dependency)에서 AVLLMs의 능력을 평가합니다. 연구진은 13개의 최첨단 AVLLMs를 평가하며, CAVPref라는 새로운 모델-불가지론적 훈련 전략을 제안합니다. 이 전략은 모든 멀티모달 입력(오디오, 비디오, 텍스트)을 조건으로 하여 AVLLMs의 신뢰성을 향상시키고, 30.19%의 성능 향상을 달성했습니다.

- **Performance Highlights**: 평가 결과, 기존 모델들은 인간과 유사한 이해력에 크게 미치지 못하는 것으로 나타났습니다. 본 연구를 통해 AVLLMs의 주요 한계점을 분석하고 성능에 대한 유용한 통찰을 제공합니다. 향후 연구 방향으로는 AVLLMs의 강인성과 추론 능력을 개선하기 위한 방안들이 제안됩니다.



### Siamese Networks for Cat Re-Identification: Exploring Neural Models for Cat Instance Recognition (https://arxiv.org/abs/2501.02112)
Comments:
          8 pages, 3 figures, 7 tables

- **What's New**: 2023년 4월, 중국의 도시 모빌리티 기업 Hello Inc.는 도시 지역의 길고양이 문제를 해결하기 위한 Hello Street Cat 이니셔티브를 시작했습니다. 이 프로젝트는 14개 도시에서 21,000개 이상의 스마트 급식소를 설치하고, 사용자 기부금을 통해 작동하는 카메라와 간식 배급기를 통합하여 길고양이를 관리합니다. 또한 이 이니셔티브는 Trap-Neuter-Return(TNR) 방법을 홍보하며, 자발적으로 운영되는 플랫폼인 HelloStreetCatWiki에서 자원봉사자들이 고양이를 분류하고 기록합니다.

- **Technical Details**: 이 연구는 길고양이를 재식별하기 위한 Deep Learning 기반 모델을 탐구하며, 2,796장의 69마리 고양이 이미지를 사용해 Siamese Networks를 학습시켰습니다. EfficientNetB0, MobileNet 및 VGG16을 기반 모델로 삼아 대비 손실과 삼중 손실 함수 하에서 평가하였습니다. 그 결과, VGG16과 대비 손실의 조합이 가장 효과적으로 입증되어 97%의 정확도와 0.9344의 F1 점수를 기록했습니다.

- **Performance Highlights**: 이 접근법은 이미지 증강 및 데이터 세트 정제를 통해 제한된 데이터와 다양한 시각적 변동성 문제를 극복했습니다. 자동화된 고양이 재식별의 가능성은 인구 모니터링 및 복지 노력을 효율화할 수 있음을 강조합니다. 이 연구는 향후 데이터 세트 확장 및 대규모 배치를 위한 실시간 구현 개발에 초점을 맞추어 실용성을 높일 계획입니다.



### AI-Powered Cow Detection in Complex Farm Environments (https://arxiv.org/abs/2501.02080)
- **What's New**: 이 논문은 인공지능(AI) 기술, 특히 컴퓨터 비전을 활용하여 동물 복지를 모니터링하고 향상시키기 위한 새로운 접근법을 제시합니다. 연구진은 다양한 환경에서 촬영된 소 이미지 데이터를 활용하여, 복잡한 조건에서도 소 감지를 개선하기 위한 모델을 개발하였습니다. YOLOv8과 CBAM(Convolutional Block Attention Module)을 결합한 감지 모델을 제안하고, 이를 통해 기존 모델들보다 뛰어난 성능을 보여줍니다.

- **Technical Details**: 본 연구에서는 실내외를 포함한 6가지 환경에서 소의 감지 데이터를 수집하였습니다. YOLOv8-CBAM 모델은 복잡한 조명 조건 및 배경 간섭이 있는 상황에서도 소를 효과적으로 식별할 수 있는 능력을 가지고 있습니다. 특히 CBAM은 공리적이며, 특징 추출을 향상시켜 감지 정확도를 높이는 데 기여합니다. 본 연구의 결과는 YOLOv8 모델 대비 2.3% 향상된 mAP를 달성하면서 95.2%의 정밀도를 기록했습니다.

- **Performance Highlights**: 분석 결과, 제안된 YOLOv8-CBAM 모델은 다양한 환경에서 약 95.2%의 정밀도와 mAP@0.5:0.95에서 82.6%의 성능을 기록하였습니다. 기존의 기본 모델들은 복잡한 조건에서는 성능이 저하되는 경향 보이지만, CBAM을 사용하는 본 연구의 접근법은 개선된 성과를 나타냈습니다. 이 연구는 또한 스마트 농장에서의 동물 감시 및 헬스 모니터링을 위한 정확한 감지를 가능하게 하여 동물 복지 향상을 위한 중요한 기여를 하고 있습니다.



### RadHop-Net: A Lightweight Radiomics-to-Error Regression for False Positive Reduction In MRI Prostate Cancer Detection (https://arxiv.org/abs/2501.02066)
Comments:
          5 pages, 4 figures - Accepted to IEEE International Symposium on Biomedical Imaging (ISBI 2025)

- **What's New**: 본 논문은 RadHop-Net이라는 새로운 경량 CNN 모델을 도입하고 있습니다. 이 모델은 전립선암의 고False Positive (FP) 비율을 감소시키기 위해 설계되었습니다. RadHop-Net은 두 단계로 구성된 파이프라인을 기반으로 하며, Radiomics를 이용하여 의심되는 ROI(Region Of Interest)를 검출하고, 이를 통해 예측 오류를 보정합니다.

- **Technical Details**: RadHop-Net의 파이프라인은 두 단계로 나뉘는데, 첫 번째 단계에서는 RadHop 기반의 Radiomics를 통해 csPCa 의심 ROI를 추출합니다. 두 번째 단계에서는 RadHop-Net이 각 ROI에 대해 수용 영역(receptive field)을 확대하여 예측 확률을 보정합니다. 이를 통해 FP와 TP 간의 균형을 맞추기 위한 새로운 손실 함수도 도입되었습니다.

- **Performance Highlights**: 개선된 RadHop-Net은 공개 데이터셋인 pi-cai에서 병변 탐지의 평균 정밀도(AP)를 0.407에서 0.468로 향상시켰습니다. 또한 이 모델은 최신 기술 대비 크게 작은 크기를 유지하여 효과적인 진단 성능을 제공합니다. 이러한 결과는 종합적으로 csPCa 조기 검출의 민감도와 정확성을 향상시킬 것으로 기대됩니다.



### ArtCrafter: Text-Image Aligning Style Transfer via Embedding Reframing (https://arxiv.org/abs/2501.02064)
- **What's New**: 최근 텍스트 기반 스타일 전이 분야에서 ArtCrafter라는 혁신적인 프레임워크가 소개되었습니다. 이 프레임워크는 이미지 내의 세부적인 스타일 요소를 캡처하기 위해 설계된 attention 기반 스타일 추출 모듈을 포함하고 있습니다. 또한, 텍스트-이미지 정렬 증강 요소를 도입하여 두 가지 모달리티 간의 조정을 강화하여 더욱 다양한 결과를 생성할 수 있게 합니다. ArtCrafter는 다중 모달 증강 임베딩과 원본 임베딩을 융합하는 명확한 조절 기능을 통해 우수한 결과를 보여줍니다.

- **Technical Details**: ArtCrafter는 세 가지 주요 구성 요소로 이루어져 있습니다. 첫째, attention 기반 스타일 추출은 복잡한 스타일 정보를 포착하기 위해 multi-layer 구조와 perceiver attention 메커니즘을 활용하여 세부적인 스타일 요소를 통합합니다. 둘째, 텍스트-이미지 정렬 증강은 각 모달리티 간의 균형 잡힌 통합을 가능하게 하여 생성된 이미지가 텍스트 프롬프트의 내용과 스타일을 잘 반영하게 합니다. 셋째, 명시적 조절 기능은 선형 보간 및 연결 방법을 통해 원본 임베딩과 다중 모달 임베딩을 결합하여, 높은 다양성과 관련성을 지닌 이미지를 생성합니다.

- **Performance Highlights**: ArtCrafter는 다양한 실험을 통해 뛰어난 비주얼 스타일화 결과를 입증하였습니다. 이 프레임워크는 스타일 강도와 컨트롤 가능성에서 뛰어난 성능을 보여줍니다. 특히, 이전 연구들에 비해 실시된 텍스트 프롬프트의 영향을 극대화하여 다양한 출력 결과를 생성하는 것을 목표로 합니다. ArtCrafter는 미술적 스타일에 대한 예민함을 유지하면서도 강력한 일반화 능력을 자랑하며, 다양한 실험 벤치마크에서 최신 기술을 초월하는 성능을 기록하였습니다.



### DreamMask: Boosting Open-vocabulary Panoptic Segmentation with Synthetic Data (https://arxiv.org/abs/2501.02048)
Comments:
          Project url: this https URL

- **What's New**: 이 논문에서는 오픈-어휘( open-vocabulary) 파놉틱 세그멘테이션의 성능을 향상시키기 위한 새로운 방법인 DreamMask를 소개합니다. 많은 기존 연구들은 제한된 카테고리에 대한 일반화에 초점을 맞춘 반면, DreamMask는 데이터 중심의 관점에서 카테고리 확장 및 레이아웃 배열, 데이터 필터링 등 다양한 디자인을 통해 훈련 데이터를 자동 생성하는 파이프라인을 제안합니다. 이를 통해 기존 웹 데이터 수집 방식에 비해 대규모 훈련 데이터 수집을 단순화하고, 모델 성능을 획기적으로 개선할 수 있는 가능성을 보여줍니다.

- **Technical Details**: DreamMask의 핵심은 두 가지 단계로 나뉘어 있습니다: 새로운 샘플 합성과 상상력 보조 훈련. 첫 번째 단계에서는 이전 훈련 카테고리와 높은 상관관계를 가진 새로운 카테고리 세트를 식별하기 위해 대형 언어 모델(LLMs)을 활용하며, 레이아웃 설명을 생성하여 해당 설명을 바탕으로 디퓨전 모델을 이용해 컨텍스트 인식 샘플 생성이 이루어집니다. 이후, 생성된 샘플의 마스크를 생성하기 위해 SAM을 사용하며, 품질이 낮은 샘플은 다단계 필터링 메커니즘을 통해 제거합니다.

- **Performance Highlights**: DreamMask를 활용한 모델은 여러 벤치마크에서 확실한 성능 개선을 보여주며, COCO 데이터셋에서 학습한 후 ADE20K에서 테스트했을 때 기존의 최고 성능 모델보다 2.1% mIoU(median Intersection over Union) 향상을 기록했습니다. 특히, 합성 데이터는 웹에서 수집한 데이터보다 성능이 뛰어난 것으로 나타났으며, 이는 합성 데이터의 높은 품질 덕분임을 시사합니다.



### MRG: A Multi-Robot Manufacturing Digital Scene Generation Method Using Multi-Instance Point Cloud Registration (https://arxiv.org/abs/2501.02041)
- **What's New**: 이 논문은 다중 인스턴스 포인트 클라우드 등록(multi-instance point cloud registration) 기법을 활용한 새로운 다중 로봇 제조 디지털 장면 생성 방법(Mult-Robot Manufacturing Digital Scene Generation, MRG)을 소개합니다. 기존의 포인트 클라우드 "분할-등록" 방법과 달리, 본 연구는 제조 설정에 맞춘 인스턴스 중심의 transformer 모듈을 개발하여 인스턴스 경계를 구분하고 지역 간의 상관관계를 포착합니다. 이 방법은 기존의 기술 대비 뛰어난 성능을 발휘하며, 실제 생산 공정의 디지털 시뮬레이션 환경을 개선하는 데 중요한 진전을 가져옵니다.

- **Technical Details**: MRG 방법은 산업 로봇의 특성과 제조 환경에 최적화되어 있습니다. 또한, 가설 생성 모듈(hypothesis generation module)을 통해 목표 인스턴스를 추출하면서 주요 특징을 유지하고, 마지막 등록 결과를 정제하기 위한 효율적인 스크리닝 및 최적화 알고리즘이 설계되었습니다. MRG는 점진적 정밀(map extraction strategy) 매핑을 통해 인스턴스 추출 정확도를 크게 향상시키며, 산업 로봇의 각 지역 간 상호 연결성도 철저히 고려합니다.

- **Performance Highlights**: 실험 평가에서는 Scan2CAD 및 Welding-Station 데이터셋에서 제안한 MRG 방법이 기존의 다중 인스턴스 포인트 클라우드 등록 기법을 초월하는 결과를 보였습니다. Scan2CAD 데이터셋에서 MR과 MP은 각각 12.15%와 17.79% 개선되었고, Welding-Station에서는 각각 16.95%와 24.15% 향상되었습니다. 이러한 결과는 MRG 방법이 제조 장면에서의 포인트 클라우드 등록 기술을 혁신적으로 발전시키는 데 기여할 것임을 시사합니다.



### A Separable Self-attention Inspired by the State Space Model for Computer Vision (https://arxiv.org/abs/2501.02040)
- **What's New**: 이번 논문에서는 Mamba의 개념을 활용하여 새로운 형태의 분리된 자기 주의 메커니즘인 VMI-SA를 제안합니다. 이를 통해 비전 Mamba(ViM)와의 공정한 비교를 위해 VMINet라는 프로토타입 아키텍처를 구축하였습니다. VMINet은 단순하지만 강력한 구조로, 기본 다운샘플링 레이어와 결합된 새로운 주의 모듈로만 구성되어 있습니다.

- **Technical Details**: 이 논문에서는 별개의 자기 주의(Separable Self-Attention)와 소프트맥스 자기 주의(Softmax Self-Attention), 상태 공간 모델(State Space Models) 간의 관계를 분석하여 설계 원칙을 수립하였습니다. 또한, 제안된 VMI-SA는 이전 토큰의 수용 범위에 국한하여 성능을 최적화합니다. 최종적으로, VMI-SA의 수용 범위를 복원하여 병렬 컴퓨팅의 장점을 유지하게 됩니다.

- **Performance Highlights**: 실험 결과, VMINet은 기존의 Vim 모델을 일관되게 초과하는 성능을 보여 주며, 최신 모델들과의 경쟁에서도 뛰어난 결과를 나타냈습니다. 이러한 성과는 효율적인 연산과 우수한 설계 원칙 덕분으로, 이미지 분류와 고해상도 밀집 예측에서 경쟁력을 갖추고 있습니다.



### 3D Cloud reconstruction through geospatially-aware Masked Autoencoders (https://arxiv.org/abs/2501.02035)
- **What's New**: 이 연구는 지상 정지 위성 이미지(MSG/SEVIRI)와 CloudSat/CPR의 레이더 반사도를 활용하여 실시간 3D 구름 구조를 재구성하는 방법을 제시합니다. 자기 지도 학습(Self-Supervised Learning, SSL) 기법인 Masked Autoencoders(MAE)와 지리정보 중심의 SatMAE를 적용하여 비라벨 MSG 이미지를 학습합니다. 이 접근법은 최첨단 모델인 U-Net보다 뛰어난 성능을 보이며, SSL의 잠재력을 입증합니다.

- **Technical Details**: 연구에서는 MSG/SEVIRI의 11개 스펙트럼 채널의 복사선 데이터를 모델 입력으로 사용합니다. MAE와 Vision Transformer(ViT) 구조를 기반으로 하여 이미지를 토큰화하고, 지역적 특징 및 공간적 관계를 학습합니다. 최종적으로는 3D 구름 재구성을 위해 모델을 파인튜닝하여 90 x 256 x 256 크기의 볼륨을 생성합니다.

- **Performance Highlights**: 실험 결과, MAE 기반의 모델이 최적화된 경우 U-Net보다 더 나은 PSNR( Peak Signal-to-Noise Ratio) 값을 달성하며, 유리한 수렴 속도를 보였습니다. 특히, 지리정보를 고려한 SSL 적용이 모델 성능을 향상시키는 데 기여하며, 복잡한 지역에서도 안정적인 결과를 나타냈습니다. 이 방법론은 구름 연구 분야에서 머신러닝 기법의 활용 가능성을 보여줍니다.



### Model Checking in Medical Imaging for Tumor Detection and Segmentation (https://arxiv.org/abs/2501.02024)
- **What's New**: 이 논문에서는 최근의 모델 체크(model checking) 기술이 신호(signal) 및 이미지(image) 분석에서 매우 유망한 가능성을 보여주고 있으며, 특히 의료 이미징(medical imaging) 분야에서 그 중요성이 강조되고 있습니다. 새로운 자동 및 반자동(natural language) 이미지 분할 기법을 통해 이미지 내에서 관심 영역(region of interest)을 정확하게 delineation 할 수 있는 프레임워크를 설계하고 평가하는 방법을 다룹니다.

- **Technical Details**: 이 연구는 공간 논리(spatial logic)를 활용하여 종양(tumorous) 및 비종양(non-tumorous) 영역을 식별하는 연산자(operators) 및 도구(tools)를 개발하는 최근의 연구를 포괄적으로 분석합니다. 또한, 공간 모델 체크 기법이 장소적 데이터(ground truth data)의 변동성(variability)으로 인해 직면하는 여러 도전과제를 논의하며, 임상 실습(clinical practice)을 위한 간편한 절차(steps)의 필요성을 강조합니다.

- **Performance Highlights**: 이 프레임워크는 의료 이미징에서 발생할 수 있는 많은 변수를 고려하면서도, 정확한 세분화를 통해 전문가들이 보다 나은 진단을 내릴 수 있도록 지원합니다. 특히, 자동 및 반자동 방식으로 관심 영역을 구분할 수 있는 점에서, 펀드멘털(fundamental)한 영향을 미칠 가능성이 큽니다.



### Rephotography in the Digital Era: Mass Rephotography and re.photos, the Web Portal for Rephotography (https://arxiv.org/abs/2501.02017)
Comments:
          6 pages, 2 figures

- **What's New**: 이번 연구는 19세기 중반 이후 재사진(rephotography)의 기법과 디지털 재사진 접근 방식을 소개하고 있으며, 대량 재사진(mass rephotography)의 미래 방향과 요구 사항을 논의합니다. 웹 포탈인 re.photos를 통해 협업 방식의 재사진 및 상호작용 이미지 등록(interactive image registration) 기술을 선보이며, 재사진의 검색과 조직, 공유 방법도 제안하고 있습니다. 대량 재사진을 위한 추가 요구 사항에는 이미지 등록의 자동화 및 사용자 친화적인 스마트폰 앱 개발이 포함됩니다.

- **Technical Details**: 현재 re.photos 웹 포탈은 협업 재사진 프로세스를 위한 중앙화된 저장소를 제공하며, 사용자가 템플릿 이미지를 업로드하고 메타 정보에 따라 검색할 수 있도록 돕습니다. 사용자는 템플릿 이미지에 따라 자신의 재사진을 업로드하고, 메타 정보 추가 및 상호작용 이미지 등록 기술을 통해 조정할 수 있습니다. 하지만 대량 재사진을 위한 자동화 및 통합 기능이 필요하며, 이는 기계 학습(machine learning) 발전을 통해 실현 가능하다고 분석하고 있습니다.

- **Performance Highlights**: 향후 대량 재사진을 위해 중앙화된 저장소와 자동화된 이미지 등록 방법이 필요하며, 이는 재사진 촬영 시 템플릿 이미지의 카메라 위치를 안내하는 방식으로 실행될 수 있습니다. 또한 메타 정보 관리를 위한 대량 업로드 시스템이 필요하며, 이는 이미지의 데이터베이스 파일을 통해 진행될 수 있습니다. 스마트폰 앱 개발은 사용자에게 템플릿 이미지를 제공하고, 소셜 미디어 공유 및 게임화(gamification) 기능을 통해 대중의 참여를 유도할 수 있는 중요한 요소로 언급되고 있습니다.



### On the Utility of Equivariance and Symmetry Breaking in Deep Learning Architectures on Point Clouds (https://arxiv.org/abs/2501.01999)
Comments:
          21 pages, 5 figures

- **What's New**: 이 논문에서는 점 구름(point clouds) 모델이 다양한 기하학적 복잡성을 가진 작업에서의 성능에 미치는 주요 요인들을 탐구합니다. 특히, 등가성 계층(equivariant layers)의 유연성과 가중치 공유(weight-sharing) 간의 트레이드오프(trade-off)를 분석하고, 등가성이 모델 성능에 미치는 긍정적 또는 부정적 영향을 평가합니다. 추가적인 정보가 모델 성능을 향상시키는 것이 일반적으로 받아들여지지만, 이러한 정보가 특정 속성을 손상시킬 경우 여전히 유익한지에 대한 질문을 다룹니다.

- **Technical Details**: 등가성 신경망(equivariant neural networks)은 비등가성 신경망(non-equivariant networks)과 비교하여, 고차원 표현 공간(high-dimensional representation spaces)을 통해 데이터의 구조적 특성을 보존하는 제약 계층(constraint layers)을 사용합니다. 이들 계층은 다양한 변환에 대해 비효율 반복 학습을 방지하며, 데이터 효율성(data efficiency)을 강화합니다. 또한 전통적인 컨볼루션 네트워크와 비교하여 G-CNNs(Group Convolutional Neural Networks)는 가중치 공유나 성능 향상을 보다 효율적으로 달성할 수 있는 장점을 제공합니다.

- **Performance Highlights**: 연구 결과, 점 구름 데이터셋인 Shapenet 3D, QM9, CMU Motion Capture를 활용한 다양한 작업에서 등가성 계층의 사용이 성능에 긍정적인 영향을 미친 것으로 나타났습니다. 특히 작업의 복잡성이 증가할수록 등가성의 효과가 두드러지게 나타났습니다. 또한, G-CNN이 비등가성 네트워크와 비교하여 모든 작업에서 일관되게 우수한 성능을 보임으로써, 이들 계층의 도입이 전반적인 모델 성능을 증대시키는 데 기여했음을 확인했습니다.



### SmartSpatial: Enhancing the 3D Spatial Arrangement Capabilities of Stable Diffusion Models and Introducing a Novel 3D Spatial Evaluation Framework (https://arxiv.org/abs/2501.01998)
- **What's New**: 이번 논문은 Stable Diffusion 모델의 한계를 극복하기 위해 SmartSpatial이라는 새로운 접근 방식을 제안합니다. SmartSpatial은 3D 인식(3D-aware) 조건부 및 주의 기반 메커니즘(attention-guided mechanisms)을 통해 공간 배치(spatial arrangement) 성능을 향상시킵니다. 이를 통해 복잡한 3D 관계를 정확하게 표현하고, 공간 정확도(spatial accuracy) 메트릭에서 현저한 개선을 보여줍니다.

- **Technical Details**: SmartSpatial은 깊이 정보(depth information)를 통합하고 교차 주의 제어(cross-attention control)를 사용하여 객체의 정확한 배치를 보장합니다. 논문에서는 새롭게 제안된 평가 프레임워크인 SmartSpatialEval을 통해 공간 관계를 평가하는 방법도 소개하며, 비전-언어 모델(vision-language models) 및 그래프 기반 종속 구문 분석(graph-based dependency parsing)을 활용합니다. 이를 통해 기존 방법에 비해 우수한 성능을 보이고 있습니다.

- **Performance Highlights**: 실험 결과는 COCO 및 SpatialPrompts 데이터셋에서 SmartSpatial이 기존 방법보다 현저하게 우수한 성능을 보여줌을 강조합니다. SmartSpatial은 이미지 생성에서 공간 배열 정확도(spatial arrangement accuracy)의 새로운 기준을 세우는 데 기여하였습니다. 이러한 성능 향상은 공간 관계의 이해 및 표현에서의 혁신을 제공합니다.



### A Novel Convolution and Attention Mechanism-based Model for 6D Object Pose Estimation (https://arxiv.org/abs/2501.01993)
Comments:
          8 pages, 4 figures, 6 tables

- **What's New**: 이 논문에서는 RGB 이미지에서 6D 객체 자세를 추정하는데 있어 기존의 깊이 정보 부족 문제를 극복하기 위해 그래프 기반 표현을 도입했습니다. 각각의 픽셀의 공간 및 시간 특징이 노드로 사용되고, 이들 간의 관계는 노드 연결성과 공간적 상호작용을 통해 정의됩니다. 또한, 공간적 주의 메커니즘과 자기 주의 증류(self-attention distillation), 레젠드르 컨볼루션 레이어를 결합하여 수치적 안정성을 확보했습니다.

- **Technical Details**: 제안된 PoseLecTr 구조는 기능 추출 계층, 레젠드르 기반 컨볼루션 레이어, 시공간 인코딩 레이어 및 최종 디코딩 레이어로 구성됩니다. 초기 단계에서 이미지를 고차원 특징으로 변환하며, 두 번째 단계에서는 레젠드르 다항식을 이용하여 컨볼루션을 수행합니다. 시공간 인코더는 자기 주의 메커니즘을 활용하여 객체의 이동과 공간 구조를 해석하는데 필요한 복잡한 시공간 관계를 캡처합니다.

- **Performance Highlights**: LINEMOD, Occluded LINEMOD 및 YCB 비디오 데이터셋에서의 실험 결과, 제안한 방법은 기존 아홉 개 접근법보다 우수한 성능을 보여주었고, 객체 자세 추정에 관한 최신 기준(benchmark)에서 뛰어난 결과를 달성했습니다. 이는 PoseLecTr이 복잡한 시나리오에서 6-DOF 자세 추정의 신뢰할 수 있는 솔루션으로 자리잡을 수 있음을 의미합니다.



### A Hybrid Deep Learning and Model-Checking Framework for Accurate Brain Tumor Detection and Validation (https://arxiv.org/abs/2501.01991)
Comments:
          22 pages, 8 figures

- **What's New**: 이 논문에서는 모델 검사(model checking)와 딥러닝(deep learning)을 통합한 새로운 하이브리드 프레임워크를 소개합니다. 이 프레임워크는 뇌 종양 탐지 및 의료 영상에서의 검증에 중점을 두고 있습니다. 모델 검사 원칙과 CNN 기반 특징 추출(feature extraction), K-FCM 클러스터링(clustering)을 통해 종양 탐지의 신뢰성을 향상시킵니다.

- **Technical Details**: 제안된 접근 방식은 CNN(convolutional neural network)을 사용하여 이미지에서 특징을 추출하고, K-FCM(k-means fuzzy c-means) 클러스터링 기법을 통해 세분화를 수행합니다. 이러한 기술은 전통적인 모델 검사의 원칙을 활용하여 중요 데이터의 신뢰성을 보장합니다. 이 하이브리드 시스템은 복잡한 의료 영상 데이터를 처리하는 데 강력한 방법론을 제공합니다.

- **Performance Highlights**: 실험 결과는 이 프레임워크의 효과성을 입증하며, 98	ext{%}의 정확도(accuracy), 96.15	ext{%}의 정밀도(precision), 100	ext{%}의 재현율(recall)을 기록했습니다. 이러한 성과는 고급 의료 이미지 분석을 위한 강력한 도구로서의 가능성을 보여줍니다.



### CRRG-CLIP: Automatic Generation of Chest Radiology Reports and Classification of Chest Radiographs (https://arxiv.org/abs/2501.01989)
- **What's New**: CRRG-CLIP 모델은 폐 영상 보고서 생성 및 영상 분류를 위한 새로운 엔드 투 엔드 모델입니다. 이 모델은 두 개의 모듈로 구성되며, 하나는 폐 영상 보고서를 자동으로 생성하고, 다른 하나는 영상 분류를 수행합니다. 본 연구는 최근의 심층 학습 기술을 활용하여 영상 보고서 작성의 복잡성과 비효율성을 해결하고자 합니다.

- **Technical Details**: 모델 구조는 Faster R-CNN을 사용하여 폐 영상에서 해부학적 영역을 식별하고, 이러한 키 영역으로부터 GPT-2를 사용하여 의미적으로 일관된 보고서를 생성합니다. 특히 CLIP 모델을 활용하여 레이블이 없는 데이터에서 최적의 특징을 추출하고, 비용이 높은 레이블이 있는 데이터셋의 필요성을 줄입니다. 이러한 접근 방식은 지역적으로 중요한 특징을 효과적으로 반영합니다.

- **Performance Highlights**: 실험 결과, 생성 모듈은 BLEU, METEOR, ROUGE-L과 같은 지표에서 높은 성능을 나타내며, 특히 GPT-4o 모델을 BLEU-2, BLEU-3, BLEU-4 및 ROUGE-L에서 초과하여 성능 향상을 보였습니다. 분류 모듈은 AUC 및 정확도 측면에서 최첨단 모델을 크게 초과하여 보고서 생성 및 영상 분류 모두에서 높은 정확도와 유창성을 달성합니다.



### Gender Bias in Text-to-Video Generation Models: A case study of Sora (https://arxiv.org/abs/2501.01987)
Comments:
          7 pages, 3 figures

- **What's New**: 본 연구는 OpenAI의 Sora 모델을 통한 텍스트-비디오 생성 모델에서 성별 편향(gender bias)의 존재를 조사합니다. 텍스트 프롬프트에서 생성된 비디오의 분석을 통해 기존의 편향성 문제가 어떠한 형태로 나타나는지 밝히고자 하였습니다.

- **Technical Details**: Sora 모델은 다양한 성 중립(gender-neutral) 및 고정 관념(stereotypical) 프롬프트를 사용하여 생성된 비디오를 분석하였습니다. 이 과정에서 특정 성별이 비전(career)과 행동(behavior)에 대해 고정 관념적으로 연결되는 방식에 대한 정량적 데이터가 수집되었습니다.

- **Performance Highlights**: 결과적으로 Sora 모델은 특정 성별을 특정 직업이나 행동에 불균형적으로 연결짓는 경향을 보였습니다. 이는 훈련 데이터에 포함된 사회적 편견(social prejudices)이 반영된 것으로 해석됩니다.



### FrameFusion: Combining Similarity and Importance for Video Token Reduction on Large Visual Language Models (https://arxiv.org/abs/2501.01986)
- **What's New**: 최근 비디오 이해에 대한 수요가 증가하면서, 대형 비전-언어 모델(LVLMs)의 시각적 토큰 처리에 큰 부담이 되고 있습니다. 기존의 토큰 축소 방법들은 주로 중요도 기반의 토큰 가지치기를 중시하며, 프레임 유사성 및 반복적인 시각적 요소로 인한 중복성을 간과했습니다. 본 논문에서는 유사성을 중요도와는 별개의 특성으로 분석하고, FrameFusion이라는 새로운 접근 방식을 제안하여 유사성 기반 통합과 중요도 기반 가지치기를 결합해 LVLM의 토큰 축소를 개선하고자 합니다.

- **Technical Details**: FrameFusion은 시각적 토큰에서 유사성과 중요도를 동시에 활용하여 효율적인 LVLM 토큰 축소를 이루어냅니다. 이 방법은 먼저 유사한 토큰을 병합한 다음, 중요도 기반의 가지치기를 통해 주어진 계산 예산에 맞추어 토큰을 줄입니다. 실험을 통해 FrameFusion이 Llava-Video 및 MiniCPM-V 등 다양한 LVLM에서 비디오 이해, 질문-응답, 검색 벤치마크에서 유효성을 입증하며, 토큰을 70% 줄이고 LLM 속도를 3.4-4.4배, 엔드투엔드 속도를 1.6-1.9배 증가시키는 성능을 보였습니다.

- **Performance Highlights**: FrameFusion은 다양한 비디오 작업에서 효과적으로 토큰을 압축하며, 기존의 밀집 모델과 비교해 비슷한 성능을 유지하면서도 계산 복잡도를 30%까지 줄일 수 있음을 보여주었습니다. 또한 이 방법은 다양한 모델 크기, 입력 길이 및 비디오 작업에 걸쳐 일반화됩니다. 이를 통해 LVLMs의 효율성을 크게 향상시키는 잠재력을 지니고 있습니다.



### Fall Detection in Passenger Elevators using Intelligent Surveillance Camera Systems: An Application with YoloV8 Nano Mod (https://arxiv.org/abs/2501.01985)
Comments:
          8 pages, 3 figures

- **What's New**: 이번 연구는 깊은 학습 알고리즘을 활용한 컴퓨터 비전 기술이 승강기 내의 낙상 사건 탐지에 어떻게 응용되는지를 중점적으로 다룹니다. YoloV8 Nano 모델을 이용하여 폐쇄된 환경과 변화하는 조명 조건에서의 낙상 인식을 목표로 합니다. 이는 기존의 낙상 탐지 시스템에 혁신적인 접근 방식을 제공할 것입니다.

- **Technical Details**: 연구는 10,000장 이상의 다양한 승강기 이미지로 구성된 강력한 데이터셋에서 모델을 훈련하는 과정을 포함합니다. 이로 인해 모델의 탐지 정밀도(precision)와 재현율(recall) 가 각각 85%와 82%에 달했습니다. 이러한 결과는 딥러닝을 통한 효과적인 이미지 분석의 가능성을 보여줍니다.

- **Performance Highlights**: YoloV8 Nano 모델은 낙상 사건 탐지에서 특히 높은 정밀도와 재현율을 기록하였습니다. 이러한 성과는 승강기 안전 시스템에 통합되어 빠른 개입을 가능하게 함으로써, 향후 안전성을 크게 향상시킬 잠재력을 가지고 있습니다.



### ECG-guided individual identification via PPG (https://arxiv.org/abs/2501.01983)
Comments:
          Accepted by ICASSP 2025. Camera Ready Version

- **What's New**: 이 논문에서는 심전도 신호(ECG)를 페로소그래프(PPG) 신호에 지식을 전이하기 위한 새로운 교차 모달 지식 증류(cross-modal knowledge distillation) 프레임워크를 제안합니다. 이 프레임워크는 PPG 신호가 추론 단계에서만 사용되도록 하여 계산 비용을 줄입니다. 또한, CLIP 기반의 지식 정렬 모듈을 도입해 정보 전파의 효율성을 증가시킵니다.

- **Technical Details**: 제안된 방법론에서는 ECG 모델이 PPG 모델의 교사 역할을 수행하며, 학습 과정에서 두 모달리티 간의 도메인 차이를 줄이기 위해 지식 정렬 모듈이 사용됩니다. 이 모듈은 훈련 과정에서만 필요하며, 추론 단계에서는 추가적인 계산 요구가 없습니다. 또한, 두 개의 인코더와 프로젝션 헤드를 포함하는 구조가 설계되어 있습니다.

- **Performance Highlights**: 종합 실험을 통해 제안된 프레임워크가 기존의 기준 모델에 비해 2.8% 및 3.0%의 향상을 보이며, 이를 통해 각기 다른 데이터베이스에서 개인 인식의 정확도를 크게 증가시켰음을 입증하였습니다. 이러한 결과는 PPG 기반 생체 인식 기술의 신뢰성과 효율성을 높이는 데 기여할 것입니다.



### Is Your Image a Good Storyteller? (https://arxiv.org/abs/2501.01982)
Comments:
          Accepted by AAAI 2025

- **What's New**: 본 논문에서는 이미지의 의미적 복잡성을 평가하는 Image Semantic Assessment (ISA) 작업을 제안합니다. ISA 데이터셋과 언어 기반 방법을 활용하여 복잡성 평가를 자동화하고, 다양한 문화적 배경을 가진 사람들이 이해할 수 있는 이미지를 생성하는 필요성을 강조합니다. 이 연구는 이미지의 의미적 가치를 기반으로 한 새로운 평가 기준을 설정하는 데 기여합니다.

- **Technical Details**: ISA 데이터셋은 Pinterest에서 수집된 2,946개의 이미지를 포함하고, 각 이미지는 Entity Score와 Semantic Score를 통해 평가됩니다. Entity Score는 이미지 내 요소의 풍부함을 측정하고, Semantic Score는 더 높은 수준의 의미적 복잡성을 평가합니다. 논문에서는 Vision-Language 협력 ISA 방법(VLISA)을 제안하여, 대형 비전-언어 모델(LVLM)을 사용해 이미지의 의미적 정보를 추출합니다.

- **Performance Highlights**: 실험 결과, ISA 작업은 전통적인 비전 모델들에 도전적이며, 제안한 방법은 Semantic Complexity Scoring 작업에서 다른 기준 모델들에 비해 우수한 성능을 보였습니다. 이 연구는 AI 모델의 교육 및 평가를 위한 복잡한 이미지를 선별하는 데 중요한 단초를 제공합니다. 따라서 더 많은 연구자들이 이미지의 의미적 복잡성에 집중할 수 있는 기반을 마련했습니다.



### Optical Character Recognition using Convolutional Neural Networks for Ashokan Brahmi Inscriptions (https://arxiv.org/abs/2501.01981)
- **What's New**: 본 연구에서 제안하는 OCR(Optical Character Recognition) 시스템은 합성곱 신경망(Convolutional Neural Networks)을 활용해 아쇼칸 브라흐미 문자 인식을 위한 것입니다. 이 시스템은 데이터 증강(data augmentation) 기법을 통해 훈련 과정을 최적화하며, 이미지 전처리 과정을 통해 잡음을 제거하고 문자 분리를 용이하게 합니다. 연구의 결과, MobileNet이 다른 두 모델에 비해 높은 정확도인 95.94%를 달성하며 성능이 우수하다는 것을 보여주었습니다.

- **Technical Details**: 연구에서 사용된 데이터셋은 아쇼칸 브라흐미 문자의 이미지를 포함하며, Indoskript - Manuscript 프로젝트의 이미지를 기반으로 수동으로 생성되었습니다. 데이터셋은 CNN 훈련을 위해 데이터 증강 기법을 적용하였으며, 이미지 회전, 크기 조정, 이동 등 다양한 변환을 통해 약 227,000개의 이미지로 확장되었습니다. 또한, 잡음 제거를 위해 중앙값 블러(median blur)와 Otsu 임계값(thresholding) 기법이 사용되어, 이미지 전처리가 수행되었습니다.

- **Performance Highlights**: 실험 결과, MobileNet은 검증 정확도 95.94%를 기록하며 다른 두 모델(LeNet, VGG-16)과 비교했을 때 가장 높은 성능을 보였습니다. 연구 결과는 아쇼칸 브라흐미 문자 인식에서 선택된 CNN 모델들이 얼마나 효과적인지를 보여줍니다. 또한, 이 연구는 고대 문자 인식에서 OCR 기술의 가능성을 드러내며, 고대 스크립트의 보존 및 디지털화에 중요한 기여를 할 것으로 기대됩니다.



### Polarimetric BSSRDF Acquisition of Dynamic Faces (https://arxiv.org/abs/2501.01980)
- **What's New**: 이번 연구는 동적인 인간 얼굴의 편광(light polarization) 획득 방법을 제시합니다. 기존의 편광 측정 시스템들이 정적이고 불투명한 객체에 국한되어 있었던 반면, 이 방법은 다양한 피부 톤과 얼굴 표정을 아우르는 공간적으로 변화하는 외관과 정밀한 지오메트리(geometry)를 포착하는 데 중점을 두었습니다. 특히, 우리의 방법은 인체 내외부의 혈색소(hemoglobin)와 멜라닌(eumelanin, pheomelanin)과 같은 생물물리학적(biophysical) 구성 요소를 포함하여 이들의 농도를 정량화할 수 있습니다.

- **Technical Details**: 제안된 방법은 단일 및 이질성(homogeneous) 또는 복잡한 서브서피스(scattering) 산란을 포함하며, 굴절률(index of refraction), 비산란의 거칠기(specular roughness) 및 강도(intensity)와 같은 여러 파라미터를 고려합니다. 이 모델은 피부 층 내부의 복잡한 상호작용을 이해하고 반영하는 데 필요한 정보를 제공합니다. 또한, 이 시스템은 광범위한 피부 톤과 감정 표현에 대한 모델링의 차별성을 제공합니다.

- **Performance Highlights**: 우리의 연구는 동적인 인간 얼굴의 편광 및 스펙트럼 반사 정보가 생물물리학적 피부 파라미터와 결합된 최초의 사례입니다. 제안된 편광 피부 모델은 다양한 렌더링 파이프라인(rending pipelines)과 무리 없이 통합되어 사용될 수 있는 가능성을 보여줍니다. 이를 통해 그 동안의 접근 방식들에 비해 더욱 사실적이고 다이내믹한 인간 얼굴의 표현이 가능해질 것입니다.



### INFELM: In-depth Fairness Evaluation of Large Text-To-Image Models (https://arxiv.org/abs/2501.01973)
Comments:
          Di Jin and Xing Liu contributed equally to this work

- **What's New**: 이번 논문에서는 다중 모달 AI 시스템의 공정성을 평가하기 위한 INFELM이라는 새로운 프레임워크를 제시하고 있습니다. 이 프레임워크는 기존의 텍스트-이미지 생성 모델의 사회적 영향 및 불공정성을 측정하기 위해 발전된 스킨톤 분류기와 편향에 민감한 콘텐츠 정렬 측정이 포함되어 있습니다. 이러한 평가 도구는 산업 애플리케이션에 적합하도록 설계되었으며, 공정성 기준을 충족하지 않는 기존 모델들의 한계를 강조합니다.

- **Technical Details**: INFELM은 네 가지 주요 기여를 통해 다중 모달 AI 시스템의 공정성 평가를 수행합니다. 첫째, 얼굴 형태학 및 스킨 픽셀 표현을 개선하여 16.04% 이상의 분류 정밀도를 높인 고급 스킨톤 분류기를 도입하였습니다. 둘째, 다양한 사회적 집단별 정의된 편향 영향 측정을 통해 콘텐츠 정렬 오류를 분석하고, 셋째, 다양한 인구 통계적 그룹에 대한 일반화 가능한 대표성 편향 평가를 제안합니다.

- **Performance Highlights**: 논문에서 실시한 대규모 분석 결과, 연구 대상 모델들이 공정성 기준을 일반적으로 충족하지 않는 것으로 나타났습니다. 특히 스킨톤 정렬 오류는 성별 오류보다 두드러지며, 편향 검사에서 기존 모델의 공정성 문제가 명확히 드러났습니다. INFELM은 다중 모달 AI 시스템의 윤리적 및 인간 중심 원칙에 부합하는 공정성 평가를 위한 강력한 기준을 세웠습니다.



### GAF-FusionNet: Multimodal ECG Analysis via Gramian Angular Fields and Split Attention (https://arxiv.org/abs/2501.01960)
Comments:
          14 pages, 1 figure, accepted by ICONIP 2024

- **What's New**: 이 논문에서는 ECG 신호(classification)의 정확한 분석을 위한 혁신적인 멀티모달 프레임워크 GAF-FusionNet(Gramian Angular Fields-Fusion Network)을 소개합니다. 이 접근법은 시간적 분석을 이미지 기반 표현과 통합하여 ECG 신호의 복잡한 신호를 보다 잘 분류할 수 있게 합니다. 또한, 이 프레임워크는 시간적 및 공간적 특징을 동적으로 융합하기 위해 듀얼 레이어 교차 채널 스플릿 어텐션 모듈을 사용하여 보완 정보를 미세하게 통합합니다.

- **Technical Details**: GAF-FusionNet은 ECG 신호의 시계열 분석과 이미지 처리 간의 간극을 메우기 위해 Gramian Angular Fields (GAF) 기법을 활용하여 원-디멘셔널(1D) 신호를 투-디멘셔널(2D) 이미지로 변환합니다. 이 변환은 주파수 분석뿐만 아니라 강력한 컴퓨터 비전 기법의 적용을 가능하게 하여 새로운 특징 추출 및 패턴 인식의 가능성을 열어줍니다. 논문은 ECG200, ECG5000 및 MIT-BIH 부정맥 데이터베이스에서의 성능을 평가하여 새로운 기준을 세웁니다.

- **Performance Highlights**: GAF-FusionNet은 세 가지 다른 ECG 데이터셋에서 각각 94.5%, 96.9%, 99.6%의 높은 정확도를 기록하였으며, 기존 최첨단 방법들에 비해 상당한 성능 개선을 보여주었습니다. 이러한 결과는 GAF-FusionNet이 ECG 신호 분석의 새로운 기준을 설정함을 나타냅니다. 이 모델의 프레임워크와 코드 또한 곧 공개될 예정입니다.



### STEAM-EEG: Spatiotemporal EEG Analysis with Markov Transfer Fields and Attentive CNNs (https://arxiv.org/abs/2501.01959)
Comments:
          10 pages, 5 figures

- **What's New**: 본 논문에서는 EEG 신호의 분석을 향상시키기 위한 새로운 통합 프레임워크 STEAM-EEG를 제안합니다. 이 접근법은 Singular Spectrum Analysis (SSA), 수정된 attention 메커니즘을 가진 병렬 1D-Convolutional Neural Networks (CNNs), 그리고 Markov Transfer Fields (MTFs)를 결합하여 EEG 신호의 복잡한 시공간 역학을 효과적으로 포착합니다. 이 방식은 EEG 신호의 패턴을 시각적으로 표현하여 데이터 탐색과 분석의 해석성을 높이는 데 기여합니다. 전반적으로 이 연구는 EEG 분석의 정확성과 강건성을 개선하는 데 중점을 두고 있습니다.

- **Technical Details**: 본 시스템의 핵심은 먼저 SSA를 이용하여 원시 EEG 시간 시리즈를 추세(trend), 계절(seasonal) 및 노이즈(noise) 구성 요소로 분해하는 것입니다. SSA 알고리즘은 데이터의 시간 구조를 포착하고, 그 후 병렬 1D-CNNs를 통해 파악된 구성 요소의 특성에 대한 학습을 진행합니다. 또한, 수정된 attention 메커니즘을 통해 여러 EEG 채널 간의 상호 의존성을 더욱 효과적으로 처리하고, 시각적 표현을 위해 MTF 이미징을 사용합니다.

- **Performance Highlights**: 제안한 STEAM-EEG 프레임워크는 다양한 EEG 데이터셋에 대해 기존의 최첨단 방법과 비교하여 뛰어난 정확도를 보여줍니다. 레이더 차트를 통해 평가된 데이터셋 전반에 걸쳐 이 방법의 우수성이 입증되었습니다. 이는 EEG 신호의 복잡한 시공간 역학을 효과적으로 캡처하고 분석할 수 있는 통합적인 접근 방식을 제공함을 의미합니다.



### Large language models for artificial general intelligence (AGI): A survey of foundational principles and approaches (https://arxiv.org/abs/2501.03151)
- **What's New**: 이 논문에서는 대규모 사전 훈련된 기반 모델(Foundation Models, PFMs) 기반의 생성 인공지능(Generative AI) 시스템이 복잡하고 비트리비얼한 문제를 해결할 수 있는 능력을 보여주고 있다고 언급합니다. 특히, 다중 모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)은 방대한 데이터 소스에서 학습하여, 세상의 풍부하고 미묘한 표현을 가능하게 하여 인간과 다른 에이전트와 협력하여 문제를 해결할 수 있게 합니다. 그러나 이러한 모델들은 상태에 따라 여전히 제한된 인지 능력을 보이고 있으며, AGI(Artificial General Intelligence) 도달을 위해서는 인체 인지와 밀접하게 관련된 몇 가지 근본적인 문제를 해결해야 한다고 강조합니다.

- **Technical Details**: 논문에서는 인지의 기본 원칙인 신체화(embodiment), 기호 기반(symbol grounding), 인과관계(causality), 기억(memory)과 같은 개념들이 어떻게 인공지능 모델에서 구현될 수 있는지 논의합니다. 이러한 원칙들은 LLM이 인간과 유사한 인지 속성을 갖도록 도와주며, 이는 더 구체적이고 의미 있는 지식 및 지능의 실현을 위한 것이다. 특히, 신체화 원칙은 기계가 환경과 상호작용하며 특정 작업을 수행하는 데 있어 중요하다고 제안합니다.

- **Performance Highlights**: AGI의 실현 가능성을 높이기 위한 노력들이 이 연구의 중심에 있으며, AGI는 복잡한 인지 작업을 수행할 수 있는 능력을 요구합니다. 모델은 여러 도메인에서 다른 맥락에서도 유효하게 작동할 수 있는 일반 지식을 협력적으로 사용해야 하며, 이는 유연하고 적응 가능한 형태의 지능을 나타냅니다. 또한 AGI는 유기적인 형태로 기능성을 통해 스스로 학습하고 문제를 해결하는 능력을 갖추어야 하며, 이러한 접근 방식은 생물학적 지능의 본질과 일치합니다.



### Dr. Tongue: Sign-Oriented Multi-label Detection for Remote Tongue Diagnosis (https://arxiv.org/abs/2501.03053)
- **What's New**: COVID-19 팬데믹으로 인해 원격 진료에 대한 필요성이 증가하면서, 정확한 혀 진단을 위한 기술 개발이 필수적임을 강조하고 있습니다. 이 연구는 Sign-Oriented Multi-label Attributes Detection 프레임워크를 통해 다양한 혀 속성을 인식하는 방법을 제안하며, 효율적인 원격 진단을 가능하게 하려는 목표를 가지고 있습니다.

- **Technical Details**: 제안된 방법론은 Adaptive Tongue Feature Extraction 모듈과 Sign-Oriented Network (SignNet)으로 구성되어 있습니다. Adaptive 모듈은 복잡한 배경에서 혀를 정확하게 분리하여 표준화된 방향으로 정렬하고, SignNet은 여러 혀 속성을 동시에 인식할 수 있도록 다중 분기 구조를 채택하여 속성 간의 관계를 효과적으로 포착합니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 다양한 혀 속성을 보다 정확하게 탐지할 수 있음을 보여주었습니다. 이는 원격 의료 환경에서의 사용 가능성을 높이며, 기존의 데이터셋과 달리 보다 포괄적이고 다양한 속성 레이블을 가진 새로운 데이터셋을 제시함으로써 연구자들에게 유용한 자원을 제공합니다.



### DDRM-PR: Fourier Phase Retrieval using Denoising Diffusion Restoration Models (https://arxiv.org/abs/2501.03030)
- **What's New**: 이번 논문에서는 Denoising Diffusion Restoration Models (DDRM)이라는 효율적이고 비지도식(post-process) 사후 샘플링 프레임워크를 활용하여 비선형 위상 복구문제를 해결하는 새로운 접근 방식을 제안합니다. 기존의 대부분의 접근 방식은 선형 역문제(linear inverse problems)에 국한되어 있었으나, 본 연구는 사전 훈련된 확률 모델을 활용함으로써 비선형 문제를 해결하는 데 있어 새로운 가능성을 열었습니다. 이 방법은 특정 훈련 없이도 프리트레인된 모델을 활용하여 실제의 위상 복원 과제를 효율적으로 수행할 수 있게 합니다.

- **Technical Details**: 고전적인 Fourier 위상 복구(phase retrieval) 문제는 노이즈가 포함된 Fourier 강도 측정값에서 신호를 복구하는 비선형 역문제입니다. 본 연구에서는 DDRM을 활용한 대안 프로젝션 방법과 결합하여 비선형 위상 복구를 위한 새로운 접근 방식을 제시합니다. 이 접근 방식은 기존의 프리트레인 모델을 사용하여 추가적인 교육과 복잡한 파라미터 조정 없이도 간편하게 구현할 수 있는 것이 특징입니다.

- **Performance Highlights**: 실험 및 시뮬레이션 데이터를 통해 제안된 방법의 성능이 검증되었습니다. Fourier 위상 복구 문제에 대한 손상 및 인지 품질 지표를 통해 경쟁력을 보여줍니다. 제안된 기법은 실제 값 이미지뿐만 아니라 복합 값 이미지나 코드화된 회절 패턴과 같은 다양한 위상 복구 데이터에도 적용 가능합니다.



### A Trust-Guided Approach to MR Image Reconstruction with Side Information (https://arxiv.org/abs/2501.03021)
Comments:
          19 pages, 14 figures

- **What's New**: MRI 스캔 시간을 단축함으로써 환자 치료를 개선하고 의료 비용을 줄일 수 있습니다. 본 연구에서는 기존의 이미지 복원 방법의 한계를 극복하기 위해 'Trust-Guided Variational Network (TGVN)'라는 새로운 딥러닝 프레임워크를 제안합니다. TGVN은 여러 가지 외부 정보(contextual side information)를 통합하여 불확실한 해를 제거하고, 낮은 품질의 측정치로도 높은 품질의 이미지를 복원할 수 있도록 합니다. 우리의 방법은 소스가 다른 여러 가지 유형의 외부 정보를 포함할 수 있어 유연성을 제공합니다.

- **Technical Details**: TGVN은 불확실한 공간(ambiguous space)을 해소하기 위해 학습 가능한 정사각형 유클리드 거리 제한을 사용하여 LIP의 정규화된 최소 제곱(reconstructed least squares) 문제를 해결합니다. 이 방법은 딥 신경망 모델과 통합되어 전체적인 유사성 메트릭을 극대화하도록 학습될 수 있으며, 매우 희소한 측정치에서도 정확하고 신뢰할 수 있는 해를 제공합니다. 우리는 다양한 체중에 대한 불완전한 또는 저품질 측정을 활용하여 다중 코일(multi-coil), 다중 대비(multi-contrast) MR 이미지 복원의 도전적인 영역에서 이 방법의 유용성을 입증했습니다.

- **Performance Highlights**: TGVN은 다중 대비와 다양한 해부학적 구조에서의 저품질 이미지를 성공적으로 복원하며, 비효율적인 솔루션 공간을 탐색하더라도 측정값과의 일치를 유지합니다. 기존의 ML 기반 솔루션들에 비해 통계적으로 유의미한 개선을 보이며 높은 가속 레벨에서도 효과적으로 작동합니다. 우리의 연구는 의료 이미징 기술의 경계를 확장하고 있으며, 관련 코드는 github.com/sodicksonlab/TGVN에서 공개될 예정입니다.



### Analyzing Fine-tuning Representation Shift for Multimodal LLMs Steering alignmen (https://arxiv.org/abs/2501.03012)
Comments:
          The first three authors contributed equally

- **What's New**: 이 논문은 다중모달 대형 언어 모델(Multimodal LLMs, MLLMs)의 내부 기제를 이해하고 설명하는 데 초점을 맞추고 있습니다. 기존 연구들이 모델의 최종 상태만을 분석했던 것과 달리, 저자들은 훈련 과정에서 발생하는 은닉 상태 표현의 진화를 체계적으로 분석합니다. 이를 통해 모델의 내부 구조가 새로운 다중모달 작업에 맞게 어떻게 전문화되는지를 밝혀냅니다.

- **Technical Details**: 저자들은 개념 기반 접근 방식을 사용하여 은닉 상태를 해석 가능한 시각적 및 텍스트 개념에 매핑합니다. 이 방법론을 통해 훈련 과정에서 인코딩된 개념의 변화 추적이 가능하며, shift vectors를 통해 원래 모델의 개념을 보완하거나 재구성할 수 있음을 보여줍니다. 이를 통해 MLLMs의 행동을 조정할 수 있는 실용적인 영향을 탐구합니다.

- **Performance Highlights**: 연구 결과, 학습된 개념들이 특정 훈련 작업에 맞춰 어떻게 조정되는지를 발견했습니다. 저자들은 원래 모델의 개념을 기준으로 shift vectors를 사용하여 많은 fine-tuned 개념을 재구성할 수 있음을 보여주며, MLLMs의 응답을 추가 훈련 없이 수정할 수 있는 가능성을 입증합니다. 이러한 방식은 MLLMs의 조정 가능성을 탐구하는 첫 번째 연구로, 모델 출력의 방향성을 조절하는 데 획기적인 통찰을 제공합니다.



### GLFC: Unified Global-Local Feature and Contrast Learning with Mamba-Enhanced UNet for Synthetic CT Generation from CBC (https://arxiv.org/abs/2501.02992)
Comments:
          Accepted by ISBI2025

- **What's New**: 이 논문에서는 Cone Beam Computed Tomography (CBCT)로부터 합성 Computed Tomography (sCT) 이미지를 생성하기 위한 새로운 프레임워크인 Global-Local Feature and Contrast learning (GLFC)을 제안합니다. 기존의 방법들이 global과 local feature 및 contrast를 효과적으로 캡처하는 데 어려움을 겪고 있는 점을 해결하고자 하였습니다. 우리는 Mamba-Enhanced UNet (MEUNet)와 Multiple Contrast Loss (MCL)를 도입하여 CBCT의 저화질 이미지를 개선하고 sCT 생성의 품질을 높였습니다.

- **Technical Details**: GLFC 프레임워크는 두 개의 down-sampling 레이어를 가진 MEUNet와 MCL을 기반으로 구성됩니다. MEUNet은 Mamba 블록을 UNet의 skip 연결에 통합하여 local feature와 long-range dependency를 효과적으로 학습합니다. MCL은 여러 intensity windows를 사용하여 soft tissues 및 bone의 품질을 높이는 방식으로 설계되었습니다.

- **Performance Highlights**: SynthRAD2023 데이터셋에서의 실험 결과, GLFC 프레임워크는 sCT의 SSIM을 77.91%에서 91.50%로 향상시켰습니다. 이는 기존의 여러 최첨단 딥러닝 방법보다 뛰어난 성능을 보여주었습니다. GLFC는 이미지의 품질을 크게 개선하여 의료 영상의 새로운 가능성을 제공할 것으로 기대됩니다.



### Region of Interest based Medical Image Compression (https://arxiv.org/abs/2501.02895)
Comments:
          8 pages, 7 figures

- **What's New**: 이번 논문에서는 원격 의료 서비스의 효율성을 높이기 위한 의료 이미지 데이터의 압축 기술을 탐구합니다. 특히, Region of Interest (ROI) 코딩을 통해 압축 비율과 이미지 품질 간의 균형을 맞추는 방법을 제시합니다. 논문은 Brats 2020 데이터셋에서 UNET 세그멘테이션을 활용하여 종양 영역을 정확하게 식별합니다.

- **Technical Details**: 선택된 종양 영역은 High Efficiency Video Coding (HEVC)을 통해 압축되며, 이를 통해 압축 비율을 높이고 필수 진단 정보를 유지합니다. 우리의 방법론은 중요한 이미지 영역의 품질을 보장하면서 비필수 영역은 더 많이 압축할 수 있도록 설계되었습니다.

- **Performance Highlights**: 이 기술은 저장 공간과 전송 대역폭을 최적화하여 원격 의료 및 대규모 의료 이미징의 필요를 충족합니다. 이 방법을 통해 중요한 데이터의 무결성을 유지하고 의료 이미지 처리를 더욱 효율적으로 개선하는 강력한 솔루션을 제공합니다.



### Diff-Lung: Diffusion-Based Texture Synthesis for Enhanced Pathological Tissue Segmentation in Lung CT Scans (https://arxiv.org/abs/2501.02867)
Comments:
          accepted at ISBI 2025

- **What's New**: 이번 논문에서는 폐 질환의 병리학적 패턴을 효율적으로 분할하는 기술을 소개합니다. 기존의 데이터 세트가 부족하고 불균형적이라는 문제를 해결하기 위해, diffusion 모델을 활용한 데이터 증강 기법을 제안합니다. 이를 통해 각 조직 유형에 특화된 자세한 패턴을 유지하면서 합성 병리조직 패치를 생성하여 AI 모델의 학습을 돕습니다.

- **Technical Details**: 이 연구는 diffusion probabilistic model을 기반으로 하여 CT 스캔에서 다양한 병리학적 질감을 합성합니다. 특히, Class-Balanced Mask Ablated Training (CBMAT) 전략을 도입하여 저 représentation된 병리학을 중점적으로 학습시킵니다. diffusion 모델은 GANs와 달리 하이퍼파라미터에 덜 민감하며, 훈련 중 모드 붕괴의 위험을 줄여줍니다.

- **Performance Highlights**: DiffLung이라는 새로운 모델을 통해 각 병리 유형에 대한 분할 정확성이 향상되었음을 입증하였습니다. 특히, 일반적이지 않은 병리 패턴의 경우에서도 효과적인 성능 향상을 보여 임상 결정을 지원할 수 있는 신뢰할 수 있는 결과를 제공합니다. 이런 발전은 폐 CT 스캔의 자동 분석을 신뢰할 수 있게 하여, 환자의 예후 개선에 기여할 가능성을 시사합니다.



### Seeing the Whole in the Parts in Self-Supervised Representation Learning (https://arxiv.org/abs/2501.02860)
Comments:
          20 pages

- **What's New**: 최근의 자기 지도 학습(self-supervised learning, SSL)에서 공간적 시각 특징의 동시 출현을 모델링하는 새로운 방법이 제안되었습니다. 기존의 이미지 마스킹이나 공격적인 크롭 방식과는 달리, CO-SSL(Co-Occurrence SSL)이라는 인스턴스 차별화 방법을 통해 지역 표현(local representations)과 전역 이미지 표현(global image representation)을 정렬하는 방식으로 접근합니다. 이 새로운 방식은 여러 데이터셋에서 성능을 개선하며, 특히 ImageNet-1K에서 100 에포크를 통해 71.5%의 Top-1 정확도를 달성했습니다.

- **Technical Details**: CO-SSL은 지역적 표현과 전역 표현의 관계를 잘 활용할 수 있도록 RF-ResNet이라는 새로운 CNN 아키텍처를 도입합니다. 이 아키텍처는 마지막 풀링 층 이후 작은 receptive field(RF)를 가진 패치 표현의 평균 집합을 추출하여 CO-SSL의 성능을 강화합니다. CO-SSL의 두 가지 구성원인 CO-BYOL과 CO-DINO도 이 구조에 기반하여 개발되었습니다.

- **Performance Highlights**: CO-SSL은 ImageNet-1K 및 Tiny-ImageNet과 같은 다양한 데이터셋에서 기존 방법보다 더 우수한 성능을 보이며, 노이즈 손상 및 내부 손상, 소규모 적대적 공격에 더 강한 내성을 가지는 것으로 나타났습니다. 분석 결과, CO-SSL은 중복이 높은 지역 표현을 추출하여 이러한 강인성을 얻는 것으로 설명됩니다. 결국, 이 작업은 지역 및 전역 이미지 표현을 정렬하는 것이 카테고리 인식의 강력한 원칙이 될 수 있음을 보여줍니다.



### InfiFusion: A Unified Framework for Enhanced Cross-Model Reasoning via LLM Fusion (https://arxiv.org/abs/2501.02795)
Comments:
          Under review

- **What's New**: 대형 언어 모델(LLMs)이 다양한 추론 작업에서 탁월한 성과를 보여주고 있지만, 모든 영역에서 일관되게 높은 성능을 발휘하는 단일 모델 구축은 여전히 도전 과제입니다. 본 논문은 여러 도메인 전문화된 모델을 효율적인 피벗 모델로 통합하는 전략을 탐구하며, 페어와이즈 멀티스텝 퓨전 접근법과 통합 퓨전 접근법의 두 가지 방식을 제안합니다. 특히, 새로운 Rate-Skewness Adaptive Fusion (RSAF) 기법을 도입하여 파라미터 병합 시 동적으로 top-K 비율을 조정합니다.

- **Technical Details**: 제안된 방법은 각 도메인에 특화된 K개의 소스 LLM을 활용해 피벗 모델 M_{p}으로 통합하는 것을 목표로 합니다. Pairwise fusion 접근법은 각 소스 모델을 순차적으로 통합하며, Knowledge Distillation 방법론을 통해 소스 모델의 지식을 효과적으로 수집합니다. 통합 퓨전 접근법은 가능한 모든 소스 모델의 출력을 집계하여 혼합 프로세스를 최적화하고, 예상된 불확실성에 기반한 가중치 조정 방법인 Uncertainty-based Distribution Fusion (UDF)을 활용합니다.

- **Performance Highlights**: 제안된 RSAF와 UDF 기법은 GSM8K, MATH, HumanEval 과제에서 각각 9.27%, 8.80%, 8.89%의 정확도 향상을 달성했다는 실험 결과를 보입니다. Llama3.1-8B 모델을 활용한 광범위한 실험을 통해, 제안된 방법론이 전통적인 감독미세조정(Supervised Fine-Tuning) 방법의 성능을 크게 초월함을 입증하였습니다. 이러한 성과는 각 소스 모델의 고유한 도메인 전문성을 효과적으로 활용했음을 나타냅니다.



### CCStereo: Audio-Visual Contextual and Contrastive Learning for Binaural Audio Generation (https://arxiv.org/abs/2501.02786)
- **What's New**: 이 논문에서는 시각 정보를 사용하여 단일 음향에서 스테레오 음향으로 변환하는 새로운 오디오-비주얼 바이나룰 생성 모델을 제안합니다. 이를 위해 시각적 맥락을 활용하여 목표 차이 음향 특성의 평균과 분산을 동적으로 정렬하는 오디오-비주얼 조건화 정규화 레이어를 포함하고 있습니다. 또한, 음향 감각 민감도를 강화하기 위해 시각적 특성에서 음수 샘플을 발굴하는 새로운 대조 학습 방법을 도입합니다.

- **Technical Details**: 제안하는 모델은 U-Net(유니트 네트워크) 기반으로, 'Contextual and Contrastive Stereophonic Learning (CCStereo)'라는 이름을 붙였습니다. 이 모델은 조건부 정규화 층과 대조 학습 방법을 활용하여 공간적 민감도를 높이며, 테스트 시 증강 방법을 적용하여 비디오 데이터의 성능을 높이는 비용 효과적인 방법도 포함하고 있습니다. 기존의 생성 프로세스를 단순히 합치는 방법 대신, 조건부 정규화를 통해 더 세밀하게 조절합니다.

- **Performance Highlights**: CCStereo 모델은 FAIR-Play 및 MUSIC-Stereo 벤치마크에서 최신 생성 정확도를 달성했습니다. 특히, FAIR-Play 데이터셋에서는 기존의 10개 및 더 도전적인 5개 분할 프로토콜에서의 우수한 성능을 입증하였습니다. 이번 연구는 다양한 오디오-비주얼 시나리오에서의 일반화 능력과 효율적인 아키텍처로 인해 더 나은 생성 품질을 제공함을 보여줍니다.



### ICFNet: Integrated Cross-modal Fusion Network for Survival Prediction (https://arxiv.org/abs/2501.02778)
- **What's New**: 이 논문에서는 Integrated Cross-modal Fusion Network (ICFNet)을 제안하여 생존 예측의 정확성을 향상시키기 위한 여러 데이터 모드를 통합합니다. 기존의 방법들이 제한된 데이터 모드에 의존하는 것과는 달리, 이 모델은 병리학적 전체 슬라이드 이미지(whole slide images), 유전체 표현 프로파일(genomic expression profiles), 환자의 인구 통계적 정보(demographic information), 치료 프로토콜을 포함한 네 가지 유형의 데이터를 활용합니다. 이를 통해 공정한 교육과 더 나은 예측을 달성할 수 있습니다.

- **Technical Details**: ICFNet은 서로 다른 데이터 유형을 처리하기 위해 세 가지 모달 인코더와 최적 운송 기반(co-attention-based) Transformer를 사용합니다. 각 모달에서 독특한 특징을 추출하여 불필요한 중복성을 줄이고, 모달리티 별 특성을 강화하는 Residual Orthogonal Decomposition (ROD) 모듈을 포함합니다. 또한, ICFNet은 바람직한 NLLLoss(negative log-likelihood loss)를 설계하여 레이블 불균형 문제를 해결합니다.

- **Performance Highlights**: 다양한 공개 TCGA 데이터셋(BLCA, BRCA, GBMLGG, LUAD, UCEC)에 대한 광범위한 실험 결과, ICFNet은 기존의 최첨단 알고리즘을 능가하는 성능을 보여주었습니다. 이 모델은 환자 치료 결정 지원 도구로서의 활용 가능성을 제시하며, 효율적인 치료 옵션 평가와 의료 자원 낭비 방지에 기여할 수 있습니다.



### Ultrasound-QBench: Can LLMs Aid in Quality Assessment of Ultrasound Imaging? (https://arxiv.org/abs/2501.02751)
- **What's New**: 본 논문에서는 Ultrasound-QBench라는 포괄적인 benchmark를 도입하여, 다중 모드 대형 언어 모델(MLLMs)을 초음파 이미지의 품질 평가 작업에 체계적으로 평가합니다. 두 개의 데이터셋인 IVUSQA와 CardiacUltraQA를 구축하여, 각각 7,709장과 3,863장의 초음파 이미지를 포함하고 있습니다. 이 데이터는 전문 초음파 전문가가 주석을 달고 품질 수준을 높음, 중간, 낮음으로 분류했습니다.

- **Technical Details**: Ultrasound-QBench의 평가 작업은 세 가지 차원으로 분해됩니다: 질적 분류(qualitative classification), 양적 점수 매기기(quantitative scoring), 비교 평가(comparative assessment). 각 작업에서 MLLMs는 이미지를 세 가지 품질 등급으로 분류하고, 다양한 품질 관련 지표에 대한 점수를 제공하며, 여러 초음파 이미지 간의 상대적 품질을 비교해야 합니다. 이러한 접근은 전통적인 품질 평가 방법의 한계를 극복하고 MLLM의 장점을 활용하여 초음파 이미지의 자동 품질 평가를 위한 기반을 제공합니다.

- **Performance Highlights**: 7개의 오픈 소스 MLLM과 1개의 프로프라이어터리 MLLM에 대한 평가는 이 모델들이 초음파 이미지 품질 분류에서 초기 능력을 보유하고 있음을 증명합니다. 더불어, Ultrasound-QBench는 MLLM의 zero-shot inference와 cross-domain expertise를 활용하여 다양한 아티팩트를 포함한 초음파 이미지의 품질을 보다 정확하게 평가할 수 있는 가능성을 제시합니다.



### Multi-layer Radial Basis Function Networks for Out-of-distribution Detection (https://arxiv.org/abs/2501.02616)
- **What's New**: 이 논문에서는 기존의 OOD(out-of-distribution) 탐지를 단순화하기 위해 분류(classification)와 OOD 탐지를 통합하는 신경망 아키텍처를 제안합니다. 특히 멀티 레이어(radial basis function networks; RBFN) 네트워크를 사용하는데, 이는 분류 신뢰도와 OOD 탐지를 효과적으로 연결합니다. 기존 RBFN의 훈련 어려움을 해결하기 위해 쉽게 훈련할 수 있는 멀티 레이어 RBFN(MLRBFN)을 개발했습니다. 이를 통해 OOD 탐지 방법에 대한 새로운 방향성을 제시하고 있습니다.

- **Technical Details**: MLRBFN은 다층 구조에서 잘 훈련될 수 있도록 설계되었으며, 새로운 우울(depression) 메커니즘을 통해 OOD 탐지의 효율성을 더합니다. 각 MLRBFN은 독립적인 분류기로 사용되거나 사전 훈련된(feature extractors) 피처 추출기의 헤드로 적용될 수 있습니다. 이 네트워크 구조는 OOD 탐지에서의 효과성을 높이기 위한 다양한 접근 방식을 포함하고 있으며, 그 과정에서 여러 기술적 요소를 통합하고 있습니다.

- **Performance Highlights**: MLRBFN은 기존의 OOD 탐지 방법들과 비교했을 때 경쟁력이 있는 성능을 보입니다. 실험 결과, 제안된 아키텍처가 OOD 탐지의 정확성과 효율성을 개선할 수 있는 잠재력이 있음을 보여줍니다. MLRBFN의 강력한 성능은 향후 OOD 탐지 연구 분야에 기여할 것으로 예상됩니다.



### GIT-CXR: End-to-End Transformer for Chest X-Ray Report Generation (https://arxiv.org/abs/2501.02598)
- **What's New**: 이번 연구에서는 X-ray 이미지에서 자동으로 방사선 보고서를 생성하기 위해 엔드투엔드( end-to-end ) transformer 기반의 새로운 방법을 제안하였습니다. 또한 의료 이미징에서 엔드투엔드 transformer에 커리큘럼 학습(curriculum learning)을 처음으로 도입하여 성과를 향상시켰습니다. 실험은 MIMIC-CXR-JPG 데이터베이스를 사용하여 진행되었으며, 기존의 자연어 생성 평가 지표에서 새로운 최첨단 결과를 기록했습니다.

- **Technical Details**: 연구에서는 GIT transformer에 다양한 기법을 통합하여 X-ray 이미지에서 방사선 보고서를 자동 생성하는 과정을 최적화했습니다. 주요 기술로는 분류 헤드를 추가하고 환자의 병력을 활용하며 여러 뷰의 이미지를 사용하는 것을 포함합니다. 특히, 커리큘럼 학습 방식이 우리의 훈련 과정에 통합되어 있으며, 이는 더 긴 의료 보고서 생성이라는 핵심적인 문제를 해결하는 데 중요한 역할을 합니다.

- **Performance Highlights**: 우리가 제안한 방법은 METEOR, F1-macro, F1-micro와 같은 여러 임상 정확성 메트릭에서 신규 최첨단 결과를 세웠으며, 이는 생성된 보고서의 정확성과 사실 완전성을 잘 보여줍니다. 또한 BLEU 및 ROUGE-L 같은 자연어 생성 메트릭에서도 기존의 최첨단 기술과 동등한 성능을 보여주었습니다. 이러한 성과는 방사선 이미징 분야의 커뮤니티에 매우 중요한 기여를 하고 있습니다.



### Gaze Behavior During a Long-Term, In-Home, Social Robot Intervention for Children with ASD (https://arxiv.org/abs/2501.02583)
Comments:
          Accepted for publication at the 2025 20th IEEE/ACM International Conference on Human-Robot Interaction (HRI)

- **What's New**: 이번 연구는 자폐 스펙트럼 장애(ASD)의 진단 특징인 비정상적인 시선 행동을 다루며, 사회 로봇과의 삼자 간 상호작용을 통해 시선 행동을 개선하는 1개월의 가정 내 중재를 탐구합니다. 연구 결과, 이 중재가 ASD 아동의 적절한 시선 행동을 촉진시키고, 로봇의 시선을 따라가게 하여 자발적인 눈 맞춤 및 공동 주의 동작을 증가시킨 것으로 나타났습니다.

- **Technical Details**: 이 연구는 15가족을 대상으로 하여, 로봇 중재 시작 전 30일, 첫 중재일, 최종 중재일, 그리고 중재 종료 후 30일 등 네 시점에서 아동의 사회적 기술과 시선 행동을 평가했습니다. ADI-R과 ADOS와 같은 전통적인 평가 도구를 통해 ASD 증상을 평가했으며, 로봇과의 상호작용을 통해 시선 행동을 더욱 자동화된 방식으로 분석할 필요성이 강조되었습니다.

- **Performance Highlights**: 로봇을 활용한 중재는 아동과 보호자 간의 시선 패턴을 개선할 뿐만 아니라, 보다 빈번하고 지속적으로 자발적인 눈 맞춤을 이루도록 하였습니다. 연구 결과는 ASD 아동의 시선 행동을 향상시킬 수 있는 로봇 보조 중재의 임상적 잠재력을 보여주며, 향후 장기적인 중재 설계와 접근 방식을 혁신할 수 있는 기반을 마련했습니다.



### KM-UNet KAN Mamba UNet for medical image segmentation (https://arxiv.org/abs/2501.02559)
- **What's New**: 이 논문에서는 의료 이미징 분석에서 중요한 의료 이미지 분할을 위한 새로운 네트워크 구조인 KM-UNet을 제안합니다. KM-UNet은 Kolmogorov-Arnold Networks (KANs)와 상태 공간 모델(state-space models, SSMs)의 장점을 결합하여 수치적 효율성과 정확도의 균형을 이룹니다. 이를 통해 기존 CNN 기반 방법과 Transformer 기반 모델의 한계를 극복하고자 합니다.

- **Technical Details**: KM-UNet은 Kolmogorov-Arnold 표현 정리를 활용하여 효율적인 특성 표현(feature representation)을 수행하며, SSM을 통해 확장 가능한 장기 모델링을 가능하게 합니다. 이러한 U자형 구조는 의료 이미지 분할 페이즈에서의 성능을 극대화할 수 있도록 설계되었습니다. 실험을 위해 ISIC17, ISIC18, CVC, BUSI, GLAS의 다섯 가지 벤치마크 데이터셋을 사용하였습니다.

- **Performance Highlights**: 실험 결과, KM-UNet은 기존 최고의 방법들과 비교했을 때 경쟁력 있는 성능을 달성하였습니다. 이는 의료 이미지 분할 작업에서 KAN과 SSM을 통합한 최초의 프레임워크임을 밝히는 새로운 통찰을 제공합니다. 이 연구는 더 효율적이고 해석 가능한 의료 이미지 분할 시스템 개발을 위한 귀중한 기준을 제공합니다.



### Neural Error Covariance Estimation for Precise LiDAR Localization (https://arxiv.org/abs/2501.02558)
Comments:
          Accepted by 2024 International Conference on Intelligent Computing and its Emerging Applications

- **What's New**: 이번 연구는 자율주행 차량의 LiDAR 맵 매칭에서 위치 추정 오차 공분산(Localization Error Covariance)을 예측하는 심층 학습 기반 프레임워크를 제안합니다. 전통적인 기술인 칼만 필터(Kalman Filter)의 성능을 개선하기 위해 새로운 데이터 생성 방법도 소개되었습니다. 관련 연구들에서 나타난 문제점들을 해결함으로써, 더욱 신뢰할 수 있는 공분산 값을 추정할 수 있게 됩니다.

- **Technical Details**: 연구는 자율주행 차량의 정확한 위치 지정을 지원하기 위해 LiDAR 기술을 사용하는 다양한 방법들을 탐구합니다. 특히 SLAM(Self-Localization and Mapping) 기법과 사전 구축된 맵을 사용하는 접근 방식의 장단점을 분석하였으며, Euclidean ICP(Iterative Closest Point) 알고리즘의 한계에 주목했습니다. 데이터 기반 방법을 통해 LiDAR 맵 매칭에서의 공분산 예측 정확도를 높이기 위한 다양한 접근법들이 설명됩니다.

- **Performance Highlights**: 실험에서 제안된 방법은 기존의 칼만 필터를 사용하는 기술에 비해 2cm의 위치 정확도 개선을 가져오는 성과를 기록하였습니다. 이러한 개선은 자율주행 차량의 안전성과 효율성을 위한 중요한 진전을 의미합니다. 더불어, 딥 러닝 기술의 발전을 활용하여 공분산 예측의 신뢰성을 높이는 방법이 제시되었습니다.



### Multi-LLM Collaborative Caption Generation in Scientific Documents (https://arxiv.org/abs/2501.02552)
Comments:
          Accepted to AAAI 2025 AI4Research Workshop

- **What's New**: 이 논문에서는 과학적 그림 캡셔닝(figure captioning)이라는 복잡한 작업을 해결하기 위한 새로운 프레임워크인 Multi-LLM Collaborative Figure Caption Generation(MLBCAP)을 제안합니다. 기존의 접근 방식이 이미지-텍스트 변환이나 텍스트 요약 작업으로만 한정된 반면, MLBCAP은 다양한 LLM(multi-modal large language models)을 활용하여 캡션 생성의 각 하위 작업을 수행합니다. 이 framework는 품질 평가(quality assessment), 다양한 캡션 생성(diverse caption generation), 그리고 판단(judgment)이라는 세 가지 모듈로 구성되어 있습니다.

- **Technical Details**: MLBCAP는 다중 모달 LLMs를 활용하여 훈련 데이터의 품질을 평가하고 저품질 캡션을 필터링합니다. 다양한 캡션 생성을 위해서는 여러 LLM을 미세 조정(fine-tuning)하거나 프롬프트(prompting)하여 후보 캡션을 생성합니다. 마지막으로, 최고의 캡션을 선택하고 오차를 수정하기 위해 GPT-4o와 같은 저명한 LLM을 사용합니다.

- **Performance Highlights**: 이 방법으로 생성된 캡션은 인간 전문가의 평가에서 원래 저자 작성 캡션보다 우수한 점수를 받았습니다. 또한, MLBCAP는 다양한 길이의 캡션을 생성하도록 설계되어, 학술 저널과 회의 논문의 페이지 제한에 적합한 방식으로 정보를 전달합니다. 인간 평가에서 MLBCAP에 의해 생성된 캡션이 높은 품질과 정확성을 보임을 입증하여, 과학적 소통의 효율성을 향상시키는 데 기여할 것으로 예상됩니다.



### Can Impressions of Music be Extracted from Thumbnail Images? (https://arxiv.org/abs/2501.02511)
Comments:
          Accepted at NLP4MusA 2024

- **What's New**: 최근 음악 검색 및 생성 시스템을 위한 머신러닝 모델에 대한 연구가 증가하고 있지만, 음악 데이터와 해당 자연어 설명(music captions)으로 구성된 대규모 공개 데이터셋이 부족한 실정입니다. 특히, 트랙을 듣기에 적합한 상황이나 이에 따른 감정과 같은 비음악적 정보는 음악 설명에 있어 필수적입니다. 이를 해결하기 위해, 본 연구에서는 음악 썸네일 이미지를 활용하여 비음악적 요소를 포함한 음악 캡션 데이터를 생성하는 방법을 제안하였습니다.

- **Technical Details**: 이 연구는 YouTube와 같은 플랫폼에서 음악 클립과 연관된 썸네일 이미지를 중심으로 진행되었습니다. 제안된 방법에서는 먼저 썸네일 이미지를 대형 비전-언어 모델(LVLM)에 입력한 다음, LVLM이 신중하게 제작된 프롬프트를 통해 음악 캡션을 생성합니다. 이러한 과정은 비음악적 요소를 포함하는 음악 캡션의 자동 생성을 가능하게 하며, 생성된 캡션은 기존의 방법보다는 비음악적 정보를 효과적으로 포함합니다.

- **Performance Highlights**: 약 360,000개의 캡션으로 개발된 데이터셋이 공개되었으며, 이를 통해 음악 검색 모델을 학습시키고 그 효과성을 평가하였습니다. 인간 평가를 통해 제안된 방법이 기존 방법들보다 우수한 성능을 나타낸 것으로 확인되었습니다. 이 연구는 음악 설명 데이터의 다양성을 확보하고, 음악 검색 모델의 품질을 향상시키는 데 중요한 기여를 할 것으로 기대됩니다.



### PTEENet: Post-Trained Early-Exit Neural Networks Augmentation for Inference Cost Optimization (https://arxiv.org/abs/2501.02508)
- **What's New**: 이번 연구에서는 Deep Neural Network (DNN)의 피드포워드 추론 과정에서 비용이 많이 드는 계산을 건너뛰는 'shortcuts'를 도입하는 방법을 제안합니다. 기존의 BranchyNet과 EEnet 아키텍처를 바탕으로 하여, 사전 훈련된 모델에 가지를 부착하여 원본 네트워크의 가중치를 변경하지 않고도 성능을 개선할 수 있습니다. 또한, 높은 DNN을 적용할 수 있는 충분한 훈련 용량을 제공하는 새로운 가지 아키텍처를 제안하며, 신뢰성 수준을 예측하기 위한 confidence heads를 포함합니다.

- **Technical Details**: 본 연구에서 제안하는 방법은 DNN의 중간 레이어에서 출구의 신뢰성을 판단하는 classification head와 decision head를 통해 이루어집니다. 이때 각 가지는 예측된 신뢰성 수준에 따라 계산을 계속 진행할지 결정하게 되며, 설정된 임계값에 따라 처리가 조절됩니다. 여러 DNN 아키텍처(ResNet, DenseNet, VGG)에서 이미지 데이터셋(SVHN 및 CIFAR10)으로 이 방법의 유효성을 검증하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 평균적인 추론 계산 비용을 줄이면서 모델의 정확성과 계산 비용 사이의 균형을 효과적으로 제어할 수 있음을 보여줍니다. 이를 통해 처리 속도와 정확성을 직관적으로 조절할 수 있어, 자원이 한정된 환경에서도 원활하게 DNN을 활용할 수 있는 가능성을 제시합니다.



### A Deep Positive-Negative Prototype Approach to Integrated Prototypical Discriminative Learning (https://arxiv.org/abs/2501.02477)
- **What's New**: 이 논문에서는 Deep Positive-Negative Prototype (DPNP)라는 새로운 모델을 제안합니다. DPNP 모델은 prototype-based learning (PbL)과 discriminative methods를 결합하여 딥 뉴럴 네트워크에서 클래스의 밀집성(class compactness)과 분리 가능성(separability)을 향상시킵니다. 이 모델은 대표적인 프로토타입과 가중치 벡터를 통합하여 구조화된 잠재 공간(latent space)을 형성하고, 이를 통해 인터프리터블한 프로토타입을 활용한 정확한 분류를 가능하게 합니다.

- **Technical Details**: DPNP 모델은 각 클래스를 위한 Deep Positive Prototype (DPP)을 생성하고, 이웃하는 경쟁 클래스의 DPP를 암묵적 부정 프로토타입(negative prototypes)으로 간주하여 서로 밀어내는 방식으로 작동합니다. 이러한 방식은 추가 파라미터 없이 클래스 간의 분리(inter-class separation)를 개선하며, 크로스 엔트로피, 프로토타입 정렬(prototype alignment), 분리(separation) 용어를 통합한 새로운 손실 함수(loss function)를 통해 특징 공간의 기하학을 잘 조직합니다.

- **Performance Highlights**: DPNP는 정규화된 특징 공간에 프로토타입을 잘 배치하여 낮은 차원의 특징 공간에서도 경쟁력 있는 분류 정확도를 달성할 수 있습니다. 여러 데이터셋에서의 실험 결과에 따르면 DPNP는 최신 기법들을 능가하며, 상대적으로 작은 네트워크를 통해 우수한 성능을 보였습니다. 이 연구는 PbL과 discriminative learning 간의 간극을 해소하여 현대의 표현 학습에서 중요한 해결책을 제시합니다.



### Framework for lung CT image segmentation based on UNet++ (https://arxiv.org/abs/2501.02428)
- **What's New**: 최근 의료 영상 분할에서 주목받고 있는 모델인 U-Net과 그 변형들에 대해, 본 연구는 종래의 한계를 극복하기 위한 새로운 접근법을 제시하였습니다. 특히, 의료 분할 분야에서의 overfitting 및 작은 데이터셋 문제를 해결하기 위해 전체 프로세스 네트워크를 제안하였습니다.

- **Technical Details**: 제안된 네트워크는 데이터 증강(data augmentation), 최적화된 신경망(optimized neural network), 파라미터 미세 조정(parameter fine-tuning)의 세 가지 주요 모듈로 구성됩니다. 이러한 모듈은 각기 다른 방법들을 통합하여 훈련 과정에서의 효율성을 높입니다.

- **Performance Highlights**: 훈련 결과는 유사한 연구에 비해 98.03%의 높은 정확도를 달성하며, 가장 낮은 overfitting 가능성을 보여줍니다. 이 네트워크는 폐 슬라이스 CT 이미지에 초점을 맞춘 첫 번째 연구 중 하나로서 선도적인 성과를 기록하고 있습니다.



### Understanding How Nonlinear Layers Create Linearly Separable Features for Low-Dimensional Data (https://arxiv.org/abs/2501.02364)
Comments:
          32 pages, 9 figures

- **What's New**: 이번 연구는 이미지 데이터의 저차원 특성을 기반으로 비선형 얕은 네트워크(shallow nonlinear networks)가 선형 분리를 어떻게 달성하는지를 명확히 규명합니다. 특히, 무작위 가중치(random weights)와 이차 활성화 함수(quadratic activations)를 사용할 때, 단일 비선형 층이 저차원 서브스페이스의 결합(union of subspaces, UoS)을 선형적으로 분리 가능한 집합으로 변환할 수 있는 가능성을 이론적으로 입증합니다. 이는 깊은 신경망(DNNs)의 이론적 이해와 실제 적용 간의 간극을 메우는 중요한 기여를 합니다.

- **Technical Details**: 연구는 K=2 서브스페이스를 사용하여 선형적 결과를 엄밀히 증명하고 있으며, K>2인 경우에도 이 결과를 확장할 수 있는 방법을 논의합니다. 분석 과정에서 이차 활성화(quadratic activations) 및 무작위 초기 가중치를 가정하고, 이러한 네트워크 폭은 데이터의 내재 차원(intrinsic dimension)에 대해 다항적으로 확장된다고 주장합니다. 또한, ReLU와 다른 비선형 활성화 함수에서도 선형 분리를 달성할 수 있는 가능성을 보여줍니다.

- **Performance Highlights**: 이론적 결과는 실험적인 증거에 의해 뒷받침되며, 실제 시나리오에서도 유사한 선형 분리 특성이 유지된다는 것을 보여줍니다. 연구 결과는 비선형 모델의 해석 가능성과 일반화 능력에 대한 깊은 통찰을 제공하고, 과적합(overparameterization)이 깊은 표현 학습(deep representation learning)에 미치는 영향을 설명하는 데 기여합니다. 따라서, 본 연구는 심층 신경망의 초기 층에서 입력 데이터가 어떻게 변환되는지를 이해하는 데 중요한 자료가 됩니다.



### GNSS/GPS Spoofing and Jamming Identification Using Machine Learning and Deep Learning (https://arxiv.org/abs/2501.02352)
- **What's New**: 최근 GNSS(Global Navigation Satellite Systems) 및 GPS(Global Positioning System)의 중요성이 더욱 부각됨에 따라, 이 기술들을 악의적인 위협으로부터 보호할 필요성이 커졌습니다. 본 논문은 스푸핑(spoofing) 및 재밍(jamming)이라는 두 가지 문제에 대해 머신러닝(machine learning)과 딥러닝(deep learning)을 활용하여 효과적인 탐지 및 완화 전략을 개발하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 논문에서는 스푸핑 및 재밍 공격에 대한 탐지를 위해 머신러닝 및 딥러닝 기법을 활용한 다양한 실험을 수행하였습니다. 실제 데이터셋을 기반으로 한 실험에서 최첨단 알고리즘을 사용하여 스푸핑과 재밍 탐지에서 각각 약 99%의 정확도를 달성하였으며, 이는 이전 연구에 비해 약 5% 향상된 결과입니다. 이를 통해 GNSS 시스템의 신뢰성을 높이는 데 기여했습니다.

- **Performance Highlights**: 스푸핑 및 재밍 탐지 과제에서의 성과는 머신러닝과 딥러닝 기술의 최신 발전을 통해 가능한 결과였습니다. 특히, 스푸핑 탐지 문제는 도전적인 과제로, 본 연구의 결과는 해당 분야에서 머신러닝과 딥러닝의 잠재력을 강조하는 중요한 진전을 나타냅니다. 이러한 연구 결과는 향후 GNSS 시스템의 보안 강화를 위한 기초 자료로 활용될 수 있습니다.



### Revelio: A Real-World Screen-Camera Communication System with Visually Imperceptible Data Embedding (https://arxiv.org/abs/2501.02349)
Comments:
          6 pages, 6 Figures, 1 Table, Accepted at IEEE International Conference on Acoustic, Speech, and Signal Processing 2025

- **What's New**: 'Revelio'는 OKLAB 색 공간에서 시간적 깜박임 융합(temporal flicker fusion)을 이용하여 스크린-카메라 통신 시스템을 구현합니다. 이 시스템은 특정 픽셀 영역 모양을 통해 정보를 인코딩하며, 육안으로는 인식할 수 없는 데이터 삽입을 가능하게 합니다. 또한, 화면-카메라 채널의 노이즈, 비동기성 및 왜곡에 대해 강한 내성을 보여주고 있습니다.

- **Technical Details**: 동적 장면에서 강한 시각적 아티팩트 없이 데이터를 신속히 임베드 할 수 있는 인코더가 개발되었습니다. 데이터가 픽셀 모양(symbols)으로 인코딩되어, 디코더는 두 단계의 신경망을 통해 정확한 프레임 감지 및 기호 인식을 수행합니다. OKLAB 색 공간에서의 공간 적응형 깜박임 기술은 이 시스템의 주요 요소입니다.

- **Performance Highlights**: 초기 실험은 'Revelio'가 인터랙티브 TV에서 효과적으로 작동함을 보여주며, 메타 정보를 무방해 방식으로 전달할 수 있습니다. 전체 HD 비디오에서 빠른 데이터 추출을 가능하게 하며, 표준 스마트폰 카메라를 이용하여 다양한 각도와 거리에서도 신뢰할 수 있는 디코딩 성능을 보입니다.



### Optimizing Small Language Models for In-Vehicle Function-Calling (https://arxiv.org/abs/2501.02342)
- **What's New**: 본 논문에서는 차량 내에 Small Language Models (SLMs)를 기능 호출 에이전트로 활용하는 혁신적인 접근 방법을 제안합니다. 기존의 규칙 기반 시스템을 대체하며, SLM을 통해 차량 제어 메커니즘을 단순화하고 사용자 경험을 향상시키는 것을 목표로 합니다. SLMs의 작은 사이즈 덕분에 차량 시스템의 통합이 용이해져, 외부 소프트웨어 업데이트나 운전자의 조건에 따라 쉽게 조정할 수 있는 시스템을 구성할 수 있습니다.

- **Technical Details**: 우리는 Microsoft의 Phi-3 mini 모델에 대한 최적화 작업을 수행하였으며, 모델 압축 기술인 pruning, healing, quantization을 통해 리소스 제약이 있는 차량에 적합하도록 하였습니다. 이 과정에서 우리는 모델의 크기를 2억 개의 파라미터를 줄이면서도 복잡한 차량 내 작업을 정확하고 효율적으로 수행할 수 있는 능력을 유지할 수 있음을 입증했습니다. 또한, 경량 런타임 환경에서 모델을 실행해 초당 11개의 토큰을 생성할 수 있어, 하드웨어 가속 없이 실시간 추론이 가능하다는 점이 특징입니다.

- **Performance Highlights**: SLM을 활용한 차량 제어 시스템은 사용자가 보다 직관적으로 차량과 상호작용할 수 있도록 지원합니다. 이 연구의 결과는 차량 시스템에 새로운 기능을 효과적으로 통합할 수 있는 가능성을 보여주며, 차량 내 앰비언트 설정 및 음성 비서와 같은 고급 기능들이 사용자 요구에 따라 변할 수 있게 합니다. 이러한 Advancements는 궁극적으로 개선된 주행 경험을 제공할 것으로 기대됩니다.



### Revisiting Compactness for District Plans (https://arxiv.org/abs/2501.02325)
- **What's New**: 이 논문에서는 인구 가중형 모양 기반 점수(population-weighted shape-based scores)를 도입하여 기존의 모양 기반 점수와 이산 점수(discrete scores) 간의 관계를 명확히 설명합니다. 또한, ReCom 샘플링 방법의 변형(version)을 제안하여 개선된 모양 기반 응집성 점수를 생성하는 방법을 제시합니다. 이를 통해 공정한 선거구 분할을 위한 양자화된 방법론을 제공합니다.

- **Technical Details**: 이 논문의 기본 개념은 주(state)나 지역을 기준으로 선거구를 나누고, 그 결과 생성된 지도(map)를 이산 그래프에서 분석하는 것입니다. 이산 컴팩트니스 점수(discrete compactness scores)는 현대의 계산적 재구획(redistricting) 알고리즘을 바탕으로 한 모형에 의해 정의됩니다. 특히, 총 둘레(perimeter)와 컷 엣지 수(cut edge count) 간의 차이를 정량화하고, 인구 가중치를 고려한 새로운 점수를 제안합니다.

- **Performance Highlights**: 제안된 방법론은 기존의 모양 기반 컴팩트니스 점수보다 더 나은 성능을 보이며 규칙적으로 변경된 총 둘레를 통해 인구 밀도를 반영합니다. 이로 인해 공정한 지역구를 형성하는 데 있어 보다 신뢰할 수 있는 분석 도구로 자리잡을 것으로 기대됩니다. 연구 결과와 방법론은 모두 GitHub에서 확인 가능합니다.



### Diabetic Retinopathy Detection Using CNN with Residual Block with DCGAN (https://arxiv.org/abs/2501.02300)
- **What's New**: 이번 연구에서는 당뇨병 망막병증(Diabetic Retinopathy, DR) 검출을 위한 자동 시스템을 제안합니다. 이 시스템은 잔여 블록 아키텍처를 사용하는 합성곱 신경망(Convolutional Neural Networks, CNNs)을 기반으로 하여, 효율적인 feature extraction을 통해 모델 성능을 향상시키고자 합니다. 또한, 다양한 망막 이미지를 생성하기 위해 Deep Convolutional Generative Adversarial Network(DCGAN)을 포함한 고급 데이터 증강 기법을 적용하였습니다.

- **Technical Details**: 연구에서 사용된 데이터셋은 췌장병 증상을 가진 망막 이미지를 포함하며, 총 35,126장의 이미지로 구성되어 있습니다. 이 이미지는 No DR, Mild, Moderate, Severe, Proliferative DR의 5개 카테고리로 분류되었습니다. 데이터 불균형 문제 해결을 위해 저희는 DCGAN을 사용하여 희귀 클래스의 다양성을 증가시켜 모델의 일반화 능력을 향상시키는 것을 목표로 하였습니다.

- **Performance Highlights**: 제안된 모델은 자동으로 망막 이미지를 분류하여 DR 진행 상황을 모니터링하는 데 기여할 수 있습니다. 특히 자원이 제한된 환경에서도 대규모 DR 검진을 지원하여, 의료 전문가의 부담을 줄이고, 적시에 적절한 치료를 받을 수 있도록 돕는 것이 목표입니다. 모델의 성능은 검증 과정에서도 긍정적인 결과를 보여주었으며, 약한 카테고리에서도 높은 분류 정확도를 달성할 수 있을 것으로 기대됩니다.



### Deep Learning-Driven Segmentation of Ischemic Stroke Lesions Using Multi-Channel MRI (https://arxiv.org/abs/2501.02287)
- **What's New**: 이 연구는 다채널 MRI 방법을 활용한 새로운 딥러닝 기반의 허혈성 뇌졸중 병변 분할(segmentation) 방법을 제안합니다. 특히, 확산 강도 영상(Diffusion Weighted Imaging, DWI), 겉보기 확산 계수(Apparent Diffusion Coefficient, ADC), 그리고 향상된 확산 강도 영상(enhanced Diffusion Weighted Imaging, eDWI) 등의 이미징 모달리티를 통합하였습니다. 이 접근법은 DenseNet121을 인코더로, Self-Organized Operational Neural Networks(SelfONN)을 디코더에 활용하여 차별화되었습니다.

- **Technical Details**: 제안된 아키텍처는 Channel and Space Compound Attention (CSCA)와 Double Squeeze-and-Excitation (DSE) 블록을 활용하여 딥러닝 성능을 향상시키고 있습니다. 또한, 모델 성능을 개선하기 위해 Dice Loss와 Jaccard Loss를 가중 평균으로 결합한 맞춤형 손실 함수(custom loss function)를 도입했습니다. ISLES 2022 데이터셋에서 학습 및 평가를 진행하여, DWI 단일 시사용량으로 83.88%의 Dice Similarity Coefficients (DSC) 달성하였습니다.

- **Performance Highlights**: DWI와 ADC를 통합한 경우에는 85.86%, DWI, ADC 및 eDWI를 통합 시 87.49%의 DSC를 기록하며 기존 방법을 초과하는 성능을 보여줍니다. 이러한 접근법은 현재의 세분화(practices) 한계점을 해결하며 진단 정확도 및 치료 계획을 크게 향상시킵니다. 또한, 연구 성과는 임상적 의사결정(clinical decision-making)에 대한 방대한 지원을 제공합니다.



### tCURLoRA: Tensor CUR Decomposition Based Low-Rank Parameter Adaptation for Medical Image Segmentation (https://arxiv.org/abs/2501.02227)
- **What's New**: 이번 논문에서는 tensor CUR 분해(tensor CUR decomposition)를 기반으로 한 새로운 파인튜닝(fine-tuning) 방법인 tCURLoRA를 제안합니다. 기존의 파라미터 효율적 파인튜닝(parameter-efficient fine-tuning) 방법들이 자원 제약이 있는 환경에서의 채택을 제한하는 문제점을 가지고 있었는데, tCURLoRA는 이러한 문제를 해결하기 위한 접근 방식을 제공합니다.

- **Technical Details**: tCURLoRA는 사전 학습(pre-trained)된 가중치(weight) 매트릭스를 3차원 텐서(tensor)로 결합하고, 텐서 CUR 분해를 적용하여 파인튜닝 동안 낮은 차원(low-order) 텐서 구성 요소만 업데이트합니다. 이 방법은 계산(computation) 및 저장(storage) 오버헤드를 효과적으로 줄이면서, 높은 차원 구조의 특성을 보다 잘 포착할 수 있는 장점을 가집니다.

- **Performance Highlights**: 실험 결과, tCURLoRA는 의료 이미지 분할(medical image segmentation) 작업에서 기존의 PEFT 방법보다 우수한 성능을 보였습니다. 이는 tCURLoRA가 높은 차원 텐서를 활용하여 다차원 상호 작용(multi-dimensional interactions)과 고차원 특징(higher-order features)을 효과적으로 캡처할 수 있기 때문입니다.



### Learning Evolution via Optimization Knowledge Adaptation (https://arxiv.org/abs/2501.02200)
Comments:
          This work has been submitted to Springer Nature for possible publication

- **What's New**: 이 논문에서는 Optimization Knowledge Adaptation Evolutionary Model (OKAEM)을 소개합니다. OKAEM은 축적된 지식을 활용하여 동적으로 매개변수를 조정함으로써 최적화 능력을 강화합니다. 이 모델은 attention 메커니즘을 사용해 개체들 간의 상호작용을 모델링하고, 진화 연산자를 매개변수화하여 선택, 교차, 돌연변이 과정을 개선합니다. 이를 통해 OKAEM은 사전 학습된 지식과 실시간 진화 통찰을 바탕으로 자체 조정이 가능한 매우 강력한 연산자를 제공합니다.

- **Technical Details**: OKAEM은 두 단계로 구성되어 있습니다: 사전 훈련과 적응 최적화. 사전 훈련 단계에서 OKAEM은 출처 작업에서 집단 진화 행동을 학습하여, 다음 세대의 집단을 예측합니다. 적응 최적화 단계에서는 새로운 진화 통찰에 맞춰 집단을 지속적으로 생성하고, 매개변수를 동적으로 조정하여 수행합니다. 이 접근 방식은 기존의 맞춤형 학습 가능 진화 알고리즘(LEAs)의 비유연성을 해결합니다.

- **Performance Highlights**: 실험 결과, OKAEM은 다양한 전이 시나리오에서 기존의 EKT 방법들을 능가하는 성능을 보였습니다. 또한, 사전 지식이 없더라도 OKAEM은 자체 조정 기능 덕분에 경쟁력 있는 성능을 달성할 수 있음을 보여줍니다. 특히, 비전-언어 모델 튜닝 사례에서는 OKAEM이 최신 블랙박스 기준을 초월한 성과를 보였습니다. 연구 결과는 OKAEM이 지식 축적에 따라 성능이 향상됨을 입증하였으며, 자연 선택 및 유전자 재조합 원리를 실질적으로 학습할 수 있다는 점에서 의의가 있습니다.



### Fresh-CL: Feature Realignment through Experts on Hypersphere in Continual Learning (https://arxiv.org/abs/2501.02198)
- **What's New**: 이번 논문에서는 지속적 학습(Continual Learning) 모델에서 발생하는 특징 엉킴을 해결하기 위한 새로운 접근 방법인 Fresh-CL(Feature Realignment through Experts on hyperSpHere in Continual Learning)을 제안합니다. 이 방법은 미리 정의된 등의 simplex equiangular tight frame (ETF) 분류기를 활용하여 특징 간의 분리를 개선합니다. 기존 방법의 한계를 넘어서, 다양한 서브스페이스에 적응할 수 있도록 전문가 모델(mixture of experts)을 도입하여 효과적인 특징 구분을 가능하게 합니다.

- **Technical Details**: Fresh-CL은 뉴럴 콜랩스(neural collapse)로부터 영감을 받아 거대한 데이터세트에서의 두드러진 특징 분리를 통해 다중 도메인(multi-domain) 작업에서도 높은 성능을 발휘합니다. 새로운 작업이 추가됨에 따라 ETF에 대한 투영이 저하되는 문제를 해결하기 위해, 여러 ETF를 사용하여 다양한 하이퍼스피어(hypersphere)에 적응적으로 투영하여 특징 구분을 강화합니다. 이 작업은 여러 데이터세트를 활용하여 모델의 정확도를 향상시키는 최적의 방법론을 나타냅니다.

- **Performance Highlights**: 11개 데이터세트에서 실험한 결과, Fresh-CL은 강력한 기준선에 비해 정확도를 2% 증가시켰습니다. 특히, 세분화된 데이터세트에서 두드러진 성과를 보이며, 기존의 신경망 아키텍처와 비교하여 매우 유의미한 개선을 달성했습니다. 이는 다양한 학습 시나리오에서도 고유의 특징을 유지하면서 지속적으로 학습할 수 있는 가능성을 보여줍니다.



### ROLO-SLAM: Rotation-Optimized LiDAR-Only SLAM in Uneven Terrain with Ground Vehic (https://arxiv.org/abs/2501.02166)
Comments:
          This article has been accepted by Journal of Field Robotics

- **What's New**: ROLO-SLAM은 고르지 않은 지형에서 차량의 위치 추정을 향상시키기 위해 제안된 새로운 방법입니다. 특히, 이 방법은 수직 방향의 드리프트를 줄이고 고각 차량의 자세를 보다 정밀하게 추정할 수 있도록 설계되었습니다. 방식적으로 전방 위치 예측을 통해 연속 스캔 간의 위치 차이를 대략적으로 제거하고, 효과적인 등록 및 최적화를 가능하게 합니다.

- **Technical Details**: ROLO-SLAM은 전방 위치 예측을 통해 회전 추정과 변환 추정을 독립적으로 처리합니다. 이를 통해 각 연속 스캔 간의 정확한 회전을 평가하기 위해 공간 복셀화 및 구형 정렬 지침을 활용합니다. 또한, 연속 시간 기반 변환 추정 기법을 사용하여 스캔의 변환을 더욱 정밀하게 획득하며, 스캔과 서브맵 간의 정렬을 통해 전반적인 SLAM 프레임워크의 효율성을 극대화합니다.

- **Performance Highlights**: 다양한 환경에서 ROLO-SLAM의 효율성을 검증한 실험 결과, 기존의 리더 기반 SLAM 프레임워크 대비 우수한 성능을 보여주었습니다. 이 연구는 도로 환경을 넘어 고르지 않은 지형에서도 안정적인 위치 추정을 구현함으로써 자율 주행 시스템에 기여할 수 있는 가능성을 열어줍니다. 최종적으로, 소스 코드와 시연 비디오가 제공되어 있어, 연구 내용의 재현이 가능합니다.



### Tree-NET: Enhancing Medical Image Segmentation Through Efficient Low-Level Feature Training (https://arxiv.org/abs/2501.02140)
Comments:
          This manuscript is 10 pages long, includes 10 figures and 3 tables, and presents a novel framework for medical image segmentation. It has been submitted to the Medical Image Analysis journal for review

- **What's New**: 이 논문에서는 의학 이미지 세분화를 위한 새로운 프레임워크인 Tree-NET을 소개합니다. Tree-NET은 bottleneck feature supervision을 활용하여 세분화 정확성과 계산 효율성을 모두 향상시킵니다. 이 연구는 입력과 출력 단계에서 bottleneck features를 사용하는 두 개의 추가 훈련 단계를 포함한 프레임워크를 제안하는 최초의 연구입니다. 이 접근법은 모델 크기를 증가시키지 않으면서도 컴퓨팅 성능을 보다 효과적으로 개선합니다.

- **Technical Details**: Tree-NET는 Encoder-Net, Bridge-Net, Decoder-Net의 세 가지 주요 구성요소로 이루어진 삼중 구조를 가지고 있습니다. Encoder-Net은 입력 이미지로부터 저수준 표현(bottleneck features)을 추출하는 합성곱(autoencoder) 모델입니다. Bridge-Net은 Encoder-Net과 Decoder-Net을 통합하는 네트워크로, Encoder-Net의 bottleneck features를 사용하여 세분화 학습을 수행합니다. Decoder-Net은 레이블 마스크의 저수준 표현을 추출하는 구조를 가지고 있습니다.

- **Performance Highlights**: 실험 결과, Tree-NET은 FLOPs(floating point operations)를 4배에서 13배까지 줄이고 메모리 사용량을 감소시키며, 기존 아키텍처 대비 동등하거나 더 나은 정확도를 달성하였습니다. 또한, Tree-NET은 다양한 백본 모델(U-NET 변형 및 Polyp-PVT)을 사용하여 피부병변 및 용종 세분화와 같은 두 가지 주요 과제를 평가하였습니다. 이러한 결과는 Tree-NET이 의학 이미지 세분화 문제에 강력하고 효율적인 솔루션이 될 수 있음을 증명합니다.



### Spot Risks Before Speaking! Unraveling Safety Attention Heads in Large Vision-Language Models (https://arxiv.org/abs/2501.02029)
- **What's New**: 이 연구는 추가 모달리티의 통합으로 인해 대형 비전-언어 모델(LVLMs)의 안전 리스크에 대한 취약성을 조사합니다. 특히, LVLM의 내부 활성화가 악의적인 프롬프트를 효과적으로 탐지할 수 있는지에 관한 분석을 제공합니다. 저자들은 '안전 헤드'(safety heads)라고 명명된 특정 주의 헤드가 이러한 탐지 메커니즘에서 중심적인 역할을 한다는 것을 발견했습니다.

- **Technical Details**: LVLMs는 비전 인코더와 LLM 텍스트 디코더를 포함하는 아키텍처로 구성되어 있습니다. 이 연구에서 저자들은 LLM의 첫 번째 토큰 생성 과정에서의 내부 활성화를 활용하여 공격을 구분할 수 있는 능력에 대해 설명하고, 이러한 활성화를 통해 안전 헤드를 식별할 수 있다고 강조합니다. 이 안전 헤드는 악의적인 시도를 막는 특수한 '방패' 역할을 수행하여, 이 헤드를 제거하면 공격 성공률이 높아진다는 사실을 발견했습니다.

- **Performance Highlights**: 제안된 악의적 프롬프트 탐지기는 로지스틱 회귀 모델을 기반으로 하며, LVLM 생성 과정에 통합되는 방식으로 설계되었습니다. 이 모델은 공격 성공률을 80% 이상에서 1-5%로 낮추는 성능을 보여주며, 기존 모델보다 빠르게 작동합니다. 또한, 이 탐지기는 새로운 데이터셋에 대한 제로샷 제너럴리제이션(zero-shot generalization) 능력을 보이며, 다양한 적대적 공격에 대한 강한 전이 가능성을 입증했습니다.



### RealDiffFusionNet: Neural Controlled Differential Equation Informed Multi-Head Attention Fusion Networks for Disease Progression Modeling Using Real-World Data (https://arxiv.org/abs/2501.02025)
- **What's New**: 이 논문은 Neural Controlled Differential Equations(Neural CDE)와 multi-head attention을 활용한 새로운 딥러닝 기반 접근법인 RealDiffFusionNet을 제안합니다. 이 모델은 불규칙하게 샘플링된 데이터를 효과적으로 처리하면서 각 시점에서 관련된 멀티모달 컨텍스트를 정렬하는 데 중점을 두고 있습니다. 연구에서는 Open-Source Imaging Consortium(OSIC)과 Alzheimer’s Disease Neuroimaging Initiative(ADNI)의 두 가지 데이터셋을 활용해 모델의 성능을 평가했습니다.

- **Technical Details**: RealDiffFusionNet은 Neural CDE를 사용하여 구조적 시간 시계열 데이터와 이미지 데이터와 같은 다양한 특징을 결합한 멀티모달 데이터를 모델링합니다. LSTM(Long Short-Term Memory) 모델이 기본 모델로 사용되었으며, ablation study를 통해 CDEs, 멀티모달 데이터 및 attention fusion의 역할을 분석했습니다. 주목할 만한 점은 Neural CDE 모델이 non-uniform time intervals을 가진 데이터를 다루는 데 효과적임을 보여준 것입니다.

- **Performance Highlights**: RealDiffFusionNet의 성능은 모든 모델(0.2570)보다 우수했으며, ADNI 데이터셋에서 Neural CDE와 LSTM 모델의 테스트 RMSE는 유사했습니다(각각 0.471와 0.4581). MRI 시리즈로부터의 이미지 특징의 추가는 성능 향상에 기여했으며, 멀티모달 데이터를 이용한 모델이 구조적 데이터보다 낮은 테스트 RMSE(0.4372)를 기록했습니다. 이 연구는 CDEs와 멀티모달 데이터를 활용하여 질병 진행을 정확하게 예측할 수 있는 가능성을 보여주고 있습니다.



### SurfPatch: Enabling Patch Matching for Exploratory Stream Surface Visualization (https://arxiv.org/abs/2501.02003)
- **What's New**: 이번 논문에서는 SurfPatch라는 새로운 프레임워크를 소개하여 탐색적인 스트림 표면 시각화를 지원합니다. 기존의 선 기반 기법들이 아닌 표면 기반 기법에 초점을 맞추어, 표면 배치 문제를 표면 선택으로 전환하고 많은 수의 스트림 표면을 추적합니다. SurfPatch는 점, 패치 및 표면 수준에서의 다단계 과정을 통해 더욱 세밀하게 다룰 수 있는 유연성을 제공하며, 이를 통해 사용자 친화적인 시각적 인터페이스를 디자인하였습니다.

- **Technical Details**: SurfPatch는 버텍스 수준 분류, 패치 수준 일치 및 표면 수준 클러스터링이라는 세 가지 핵심 단계를 거쳐 스트림 표면을 분석합니다. 네트워크 구조를 간소화하여 계산 비용을 줄이면서도 메시 품질을 유지하고, HKS(Heat Kernel Signature)를 사용해 각 버텍스의 특징을 추출합니다. 그 후, 연결 제약이 있는 집합적 계층적 클러스터링(AHC)을 통한 패치 생성을 통해 패치 수준 특징을 집계하고, 이들을 활용해 유사 패치를 쿼리할 수 있는 방법도 제공합니다.

- **Performance Highlights**: 실험을 통해 SurfPatch는 정적 및 비정적 흐름에서 생성된 스트림 표면과 스칼라 필드에서 추출된 이소 표면에 대해 효과성을 입증했습니다. 이와 같은 유연한 구조는 사용자에게 대규모 데이터에서 더욱 효과적으로 스트림 표면을 시각화하고 분석할 수 있는 능력을 제공합니다. SurfPatch의 코드도 웹에서 접근 가능하여 연구자들이 직접 실험해볼 수 있도록 하고 있습니다.



### Multi-Center Study on Deep Learning-Assisted Detection and Classification of Fetal Central Nervous System Anomalies Using Ultrasound Imaging (https://arxiv.org/abs/2501.02000)
- **What's New**: 이 연구는 임신 중 태아 성장과 선천적 이상을 평가하기 위해 딥러닝 모델을 활용하여 태아 두개 이상 진단의 정확도를 향상시키는 것을 목표로 합니다. 수집한 다기관 데이터셋에서 태아의 네 가지 주요 중추신경계 이상(anencephaly, encephalocele, holoprosencephaly, rachischisis)에 대한 진단 정확도가 94.5%에 달하며, AUROC 값은 99.3%에 이릅니다. 이 연구는 인공지능(AI) 기술이 특히 진단 과정에서 어떤 영향을 미칠 수 있는지를 보여줍니다.

- **Technical Details**: 이번 연구는 비침습적이고 저비용인 산전 초음파(ultrasonography)의 어려움을 극복하기 위해 딥러닝(deep learning) 기반의 알고리즘을 개발하였습니다. CNN(Convolutional Neural Networks)을 사용하여 태아 두개 영상의 자동 탐지 및 분류를 수행하며, 이 시스템은 초음파 이미지 위에 히트맵(heatmap)을 겹쳐 시각적 해석을 제공함으로써 의사들에게 중요한 영역을 강조합니다. 이로 인해 임상에서의 정확성을 높이고 진단 효율 또한 향상되며, 오진율 감소라는 긍정적 효과로 이어집니다.

- **Performance Highlights**: 연구 결과, 딥러닝 모델은 전체 임신 기간 동안 태아 이상 유형을 잘 식별할 수 있으며, 이로 인해 진단 과정이 대폭 개선될 것으로 기대됩니다. 연구의 회고적 독자 연구는 DL 시스템의 자동 예측과 방사선 전문의의 전문 판단을 결합함으로써 진단 정확성과 효율성을 크게 향상시킬 수 있음을 보여줍니다. 이를 통해 최종적으로 환자의 불필요한 검사율을 줄이는 데 기여할 것입니다.



### Leveraging AI for Automatic Classification of PCOS Using Ultrasound Imaging (https://arxiv.org/abs/2501.01984)
Comments:
          Code available at: this https URL

- **What's New**: AUTO-PCOS Classification Challenge는 인공지능(AI)의 진단 능력을 향상시키기 위해 건강한 초음파 이미지와 질병이 있는 이미지를 자동으로 분류하는 방법론을 제시하고 있습니다. 주요 목표는 Polycystic Ovary Syndrome(PCOS)의 조기 진단을 지원하여 의사들이 보다 정확한 진단을 가능하게 하는 것입니다. 이 프로젝트는 특히 AI가 의료 분야에서 어떻게 혁신적인 변화를 가져올 수 있는지를 보여주는 중요한 사례로 평가받고 있습니다.

- **Technical Details**: 연구팀은 InceptionV3 아키텍처를 사용하여 전이 학습(transfer learning)을 통해 강력한 AI 파이프라인을 구축하였습니다. 데이터 전처리는 이미지 사이즈, 품질, 색상 분포의 변화를 다루어 모델 성능을 향상시키는 데 기여했습니다. 또한, 데이터셋은 4,668개의 초음파 이미지로 구성되며, 각 이미지는 의료 전문가에 의해 주석이 달린 신뢰할 수 있는 자료입니다.

- **Performance Highlights**: 모델의 최종 정확도는 90.52%로, 효율적인 binary classification을 통해 검증 데이터를 분석한 결과, precision, recall, F1-score 모두 90%를 초과했습니다. 이는 모델이 PCOS 진단을 위한 신뢰할 수 있는 도구임을 시사합니다. 이 연구는 AI가 의료 진단 도구를 보다 신뢰성 있고 해석 가능한 형태로 발전시킬 수 있는 가능성을 강조합니다.



### Adaptive Homophily Clustering: Structure Homophily Graph Learning with Adaptive Filter for Hyperspectral Imag (https://arxiv.org/abs/2501.01595)
Comments:
          14 pages, 8 figure

- **What's New**: 본 논문에서는 새로운 방식인 적응형 필터 클러스터링 방법(Adaptive Filter Clustering, AHSGC)을 제안하여 비지도 방식으로 하이퍼스펙트럼 이미지(hyperspectral image, HSI) 클러스터링 문제를 해결합니다. 이 방법은 동질적인 영역 생성을 통해 HSI 처리를 시작하며 원본 그래프를 구축합니다. 또한, 그래프에서 고주파와 저주파 특성을 적응적으로 캡처하기 위한 그래프 인코더가 설계되었습니다.

- **Technical Details**: AHSGC는 KL 발산(KL Divergence)을 기반으로 한 그래프 임베딩 클러스터링 셀프 트레이닝 디코더를 도입하여 네트워크 훈련을 위한 유사 레이블(pseudo-label)을 생성합니다. 클러스터링 작업에 따라 그래프를 업데이트하기 위해 동질성 향상 구조 학습(homophily-enhanced structure learning)이 도입되며, 노드 연결을 추정하기 위한 방향상 상관관계 추정(orient correlation estimation)이 사용됩니다. 또한, 그래프의 에지를 동적으로 조정하기 위한 그래프 에지 희소화(graph edge sparsification) 기술도 포함되어 있습니다.

- **Performance Highlights**: 광범위한 실험과 반복된 비교 분석을 통해, AHSGC는 높은 클러스터링 정확도, 낮은 계산 복잡성, 강력한 강인성을 제공함이 입증되었습니다. 이 방법은 효과적인 공간 구조 정보 인코딩을 통해 HSI 클러스터링의 한계를 극복하고, 성능 향상을 이끌어 냅니다. 코드 소스는 논문에서 제공하는 URL을 통해 사용할 수 있습니다.



New uploads on arXiv(cs.AI)

### PPTAgent: Generating and Evaluating Presentations Beyond Text-to-Slides (https://arxiv.org/abs/2501.03936)
Comments:
          8 pages, 20 figures

- **What's New**: PPTAgent는 문서에서 자동으로 프레젠테이션을 생성하기 위한 혁신적인 접근 방식을 제안합니다. 기존의 방법들이 시각적 디자인과 구조적 일관성을 간과한 것을 보완하기 위해, PPTAgent는 두 단계의 편집 기반(workflow) 방식으로 프레젠테이션 생성을 개선합니다. 또한, PPTEval이라는 평가 프레임워크를 도입하여 생성된 프레젠테이션의 품질을 콘텐츠, 디자인, 일관성의 세 가지 차원에서 종합적으로 평가합니다.

- **Technical Details**: PPTAgent는 두 가지 주요 단계로 구성되어 있습니다. 첫 번째 단계에서는 참조 프레젠테이션을 분석하고, 유사한 슬라이드를 클러스터링하여 콘텐츠 스키마를 추출합니다. 이어지는 두 번째 단계에서는 입력 문서와 분석된 참조 프레젠테이션을 기반으로 적절한 슬라이드를 선택하고, 선택된 슬라이드에 기반한 상호 작용 편집 프로세스를 통해 목표 프레젠테이션을 생성합니다.

- **Performance Highlights**: PPTAgent는 기존의 자동 프레젠테이션 생성 방식에 비해 모든 차원에서 현저한 성능 향상을 보여줍니다. 실험 결과, PPTEval로 평가한 프레젠테이션의 평균 점수는 3.67이며, 다양한 도메인에서 97.8%라는 높은 성공률을 기록했습니다. 이는 PPTAgent의 접근 방식이 뛰어난 다재다능성과 강건성을 지니고 있음을 보여줍니다.



### Dolphin: Closed-loop Open-ended Auto-research through Thinking, Practice, and Feedback (https://arxiv.org/abs/2501.03916)
Comments:
          19 pages, 11 figures, and our homepage: this https URL

- **What's New**: 이 논문에서는 인공지능(AI)을 활용한 자동 과학 연구 프레임워크인 Dolphin을 제안합니다. Dolphin은 아이디어 생성, 실험 수행, 결과 피드백의 순환 구조를 구축하여 인류의 연구 과정을 자동화합니다. 이 프레임워크는 새로운 연구 아이디어를 생성하고 실험 결과를 분석하여 다음 단계로 피드백을 제공합니다.

- **Technical Details**: Dolphin은 주제와 작업 속성에 따라 관련 논문을 기반으로 새로운 아이디어를 생성합니다. 자동 생성된 코드 및 실험 계획을 바탕으로 디버깅을 수행하며, 실험 결과를 분석하여 더 높은 품질의 아이디어를 생성하는 데 기여합니다. 이를 통해 2D 이미지 분류 및 3D 포인트 분류와 같은 작업에서 최신 기술과 경쟁 가능한 방법을 제시합니다.

- **Performance Highlights**: Dolphin은 다양한 주제와 벤치마크 데이터셋에서 실험을 수행하며 지속적으로 새로운 아이디어를 생성할 수 있습니다. 실험 결과는 Dolphin이 기존의 방법들과 비교할 때 유의미한 성과를 거두고 있음을 보여줍니다. 특히, 제안된 닫힌 루프 설계를 통해 아이디어의 품질이 개선되는 것을 확인하여, 자동 연구의 효과성을 입증하고 있습니다.



### Neural DNF-MT: A Neuro-symbolic Approach for Learning Interpretable and Editable Policies (https://arxiv.org/abs/2501.03888)
Comments:
          AAMAS 2025

- **What's New**: 이번 연구에서는 딥 강화 학습(DRL)의 해석 가능성 문제를 해결하기 위한 신경 상징적 모델인 neural DNF-MT를 제안합니다. 이 모델은 훈련된 정책이 해석 가능한 표준 논리 프로그램으로 변환될 수 있도록 설계되었습니다. 또한, 복잡한 관찰에서 추상적 특성을 추출하여 예측 변환을 수행할 수 있는 추가 레이어를 포함할 수 있습니다.

- **Technical Details**: neural DNF-MT 모델은 완전히 미분 가능하며, 깊은 actor-critic 알고리즘과 통합하여 훈련할 수 있습니다. 이 모델은 정책 증류를 위해 다른 신경 모델에서 정책을 추출하는 데도 사용됩니다. 훈련된 신경 DNF-MT 액터로부터 결정론적 정책에 대한 이원 논리 프로그램(bivalent logic program)이나 확률적 정책에 대한 확률적 논리 프로그램을 추출할 수 있습니다.

- **Performance Highlights**: 실험 결과, neural DNF-MT 모델은 기존의 블랙박스 방법들과 비교하여 동등한 성능을 보이며, 해석 가능한 정책을 제공합니다. 이 모델은 수동 개입과 정책 수정이 가능하여, 경제적이고 안전한 AI 시스템 개발에 기여할 것으로 예상됩니다. 다양한 과제에 대한 평가로, 결정론적 및 확률적 행동을 학습하는 데 효과적임을 입증하였습니다.



### Online Reinforcement Learning-Based Dynamic Adaptive Evaluation Function for Real-Time Strategy Tasks (https://arxiv.org/abs/2501.03824)
Comments:
          22 pages, 9 figures

- **What's New**: 이 연구에서는 동적이고 예측 불가능한 환경에서 효과적인 평가를 수행하기 위한 적응 메커니즘을 제안합니다. 제안된 방법은 전통적인 정적 평가 함수에서 발전하여, 실시간 전략 게임 내에서의 전투 상황 변화에 빠르게 대응하기 위해 온라인 강화 학습 기반의 동적 가중치 조정 메커니즘을 활용합니다. 이는 평가 기능을 개선하며, 특히 실시간 응답성을 높이는 데 중점을 둡니다.

- **Technical Details**: 연구에서 사용된 방법은 온라인 강화 학습의 경량화된 가중치 조정 기법을 기반으로 합니다. 경량경량화는 경량화 학습의 속도를 빠르게 하고, 가중치 감쇠(weight decay) 기술을 적용하여 안정성을 확보합니다. 또한, AdamW 옵티마이저를 통합하여 실시간으로 학습률과 감쇠율을 조정함으로써 수동 파라미터 튜닝을 최소화합니다.

- **Performance Highlights**: 경쟁 실험을 통해 제안된 방식이 Lanchester 전투 모델 및 여러 계획 알고리즘의 평가 함수 효율성을 크게 향상시키는 것으로 나타났습니다. 특히, 맵 크기가 증가함에 따라 점수의 개선이 더욱 뚜렷하게 나타났으며, 모든 평가 함수 및 계획 알고리즘에서 평가 함수 계산 시간이 6% 이하로 유지되는 것을 확인했습니다. 이 연구는 실시간 전략 작업 평가를 위한 유망한 접근 방식을 제시합니다.



### Neural Deconstruction Search for Vehicle Routing Problems (https://arxiv.org/abs/2501.03715)
- **What's New**: 본 연구에서는 전통적인 순차적 문제 해결 방식을 도전하는 새로운 이터레이티브 검색 프레임워크, 즉 Neural Deconstruction Search (NDS)를 제안합니다. NDS는 신경 정책(neural policy)을 사용하여 기존 해답을 비구성화(deconstruct)한 뒤, 이를 재구성하는 방식을 채택합니다. 이 방법론은 다양한 차량 경로 문제에서 기존의 수작업 최적화 기법을 능가하는 성능을 보여줍니다.

- **Technical Details**: NDS는 큰 이웃 탐색(large neighborhood search, LNS)과 파괴 후 재구성(ruin-and-recreate) 패러다임을 기반으로 한 두 단계의 프로세스를 통해 솔루션을 개선합니다. 이 과정에서 깊은 신경망(deep neural network, DNN)을 활용하여 현재 솔루션의 일부 고객을 제거하며, 간단한 탐욕적 삽입(greedy insertion) 알고리즘을 사용하여 재구성 단계에서 고객을 최적 위치에 삽입합니다. NDS는 강화 학습(reinforcement learning)을 통해 훈련되어 reference solution이 없는 문제에도 적용이 가능하다는 특징이 있습니다.

- **Performance Highlights**: NDS는 CVRP, VRPTW, PCVRP와 같은 여러 도전 과제를 평가하여, 기존 학습 기반 방법들과 비교하여 상당한 성능 향상을 증명하였습니다. NDS의 성능은 특히 중간 크기의 CVRP 문제에서 기존의 최첨단 운영 연구 방법들과 비교하여 우수한 성과를 보였습니다. 이로 인해 NDS는 이전의 학습 기반 접근 방식들과는 다른 이정표를 설정한 첫 번째 방법론으로 자리 잡았습니다.



### STContext: A Multifaceted Dataset for Developing Context-aware Spatio-temporal Crowd Mobility Prediction Models (https://arxiv.org/abs/2501.03583)
- **What's New**: 이번 논문에서는 스마트 시티에서의 컨텍스트 인식을 활용한 스페이셜-템포럴 군중 흐름 예측 (STCFP) 모델을 위한 새로운 데이터셋인 STContext를 제안합니다. STContext는 날씨, 공기질 지수, 공휴일, 포인트 오브 인터레스트(POI), 도로 네트워크 등 10가지의 다양한 컨텍스트 특성을 포함한 9개의 스페이셜-템포럴 데이터셋을 제공합니다. 이 데이터셋은 기존의 개방형 군중 흐름 데이터셋들이 갖고 있는 한계를 극복하고, 더 나아가 다양한 STCFP 시나리오를 포괄하는 데 중점을 두고 있습니다.

- **Technical Details**: STContext 데이터셋은 5개의 STCFP 시나리오에 걸쳐 있으며, 사용자가 효과적으로 컨텍스트 특성을 처리하고 모델링할 수 있도록 다양한 전략을 포함합니다. 연구팀은 이러한 다양한 컨텍스트 특성을 효과적으로 통합할 수 있는 통합 워크플로우를 제안하였습니다. 이 워크플로우는 특징 변환, 의존성 모델링, 표현 융합 및 훈련 전략을 포함하여 STCFP 모델의 성능을 최적화하는 데 도움을 줍니다.

- **Performance Highlights**: 연구팀은 STContext를 활용한 다양한 실험을 통해 컨텍스트 모델링에 대한 유용한 지침과 앞으로의 연구 방향에 대한 통찰을 얻었습니다. STContext는 오픈 소스로 제공되며, 이는 STCFP 연구 커뮤니티가 새로운 모델링 기법을 개발하는 데 기회를 제공할 것으로 기대됩니다. 이 데이터셋은 STCFP 연구자들에게 실제 적용 가능한 예측 모델 개발에 중점을 두는 중요한 기초 자료로 자리매김할 것입니다.



### SenseRAG: Constructing Environmental Knowledge Bases with Proactive Querying for LLM-Based Autonomous Driving (https://arxiv.org/abs/2501.03535)
Comments:
          This paper has been accepted for presentation at WACV Workshop LLMAD 2025

- **What's New**: 이 연구는 자율 주행(AD)에서의 상황 인식을 향상시키기 위해 대형 언어 모델(LLMs)의 맥락적 추론 능력을 활용하고 있습니다. 전통적인 인식 시스템이 고정된 레이블 기반 주석에 의존하는 것과 달리, 이 시스템은 실시간 멀티모달 센서 데이터를 통합하여 LLMs가 복잡한 주행 환경을 이해하고 대응할 수 있도록 돕습니다. 저자는 proactive Retrieval-Augmented Generation (RAG) 기법을 설계하여 AD의 효율성을 높이고, 환경 변화에 대한 빠르고 풍부한 이해를 보장합니다.

- **Technical Details**: 연구진은 자율 주행 시스템을 위한 SenseRAG 프레임워크를 제안하여, 다양한 센서에서 수집한 실시간 데이터를 통합하는 지식 데이터베이스를 구성했습니다. 이 프레임워크는 차량이 실시간 환경 정보를 해석하고 적응할 수 있도록 설계되어 있으며, 쿼리 생성 메커니즘을 통해 동적으로 필요한 정보를 검색합니다. 이를 통해, AD 시스템은 복잡한 교통 상황에서도 안전하고 스마트한 주행 결정을 내릴 수 있게 됩니다.

- **Performance Highlights**: 실험 결과는 이 프레임워크가 AD 성능을 물질적으로 개선하며, 예측 위치 오차를 약 70% 줄였음을 보여줍니다. 또한, 새로운 지식 데이터베이스는 다양한 센서 입력을 표준화하여 환경 정보를 효율적으로 저장하고 정확한 검색을 가능하게 합니다. 이 접근법은 LLM의 상황 인식 능력을 향상시키고, 복잡한 환경에서의 의사 결정을 최적화하는 데 기여합니다.



### From Aleatoric to Epistemic: Exploring Uncertainty Quantification Techniques in Artificial Intelligenc (https://arxiv.org/abs/2501.03282)
Comments:
          14 pages

- **What's New**: 이 논문은 인공지능(AI) 시스템의 불확실성 정량화(uncertainty quantification, UQ) 기술의 발전을 다루고 있으며, 특히 의료, 자율 시스템, 금융 기술 분야에서의 중요성을 강조합니다. AI 시스템의 결정 과정에서 불확실성을 효과적으로 관리하는 것이 필수적이며, aleatoric 불확실성과 epistemic 불확실성을 구분하여 설명합니다. 이 리뷰는 확률론적 방법, 앙상블 학습, 샘플링 기반 접근법 및 생성 모델과 같은 다양한 고급 기술을 포괄적으로 분석합니다.

- **Technical Details**: 불확실성 정량화는 주로 aleatoric 불확실성과 epistemic 불확실성 두 가지로 구분됩니다. Aleatoric 불확실성은 데이터의 본질적인 무작위성과 노이즈로 인해 발생하며, 추가 데이터나 모델 개선으로도 제거되지 않는 경우가 많습니다. 반면에 epistemic 불확실성은 모델의 한계나 데이터 분포에 대한 이해 부족에서 발생하며, 추가 데이터나 개선된 모델을 통해 줄일 수 있는 특성을 갖습니다. 이러한 두 유형의 불확실성은 실제 상황에서 서로 상호작용하며, 이를 고려한 정량화가 필요합니다.

- **Performance Highlights**: 이 리뷰는 불확실성 정량화(UQ) 기술이 의료, 자율 주행, 금융 분야에서의 실제 효과와 도전 과제를 강조하면서 연구 동향을 제시합니다. 현재 기술의 한계로는 높은 계산 복잡성, 실시간 성능 부족 및 다중 도메인 적용성 제한 등이 있으며, 향후 연구 방향으로는 대규모 데이터와 동적 환경에서의 불확실성 관리 및 설명 가능한 AI(XAI)와의 통합 방안을 제안하고 있습니다. 이러한 기술 발전은 AI 시스템의 신뢰성, 안전성 및 윤리적 준수를 향상시키는 데 기여할 것입니다.



### Advanced Displacement Magnitude Prediction in Multi-Material Architected Lattice Structure Beams Using Physics Informed Neural Network Architectur (https://arxiv.org/abs/2501.03254)
Comments:
          34 pages, 19 figures

- **What's New**: 이 논문에서는 Physics-Informed Neural Networks (PINNs)와 유한 요소 분석(finite element analysis)을 결합한 혁신적인 방법을 통해 설계된 격자 구조의 변형 예측을 제안합니다. 연구는 다양한 모서리 하중에서 FCC 기반 격자 빔을 다섯 가지 재료(Structural Steel, AA6061, AA7075, Ti6Al4V, Inconel 718)를 사용하여 수행되었습니다.

- **Technical Details**: PINN 모델은 데이터 기반 학습(data-driven learning)과 물리 기반 제약(physics-based limitations)을 독자적으로 개발한 손실 함수(loss function)를 통해 결합합니다. 이 방법은 선형 회귀(linear regression)보다 훨씬 높은 예측 정확도를 보이며, R-square 값이 0.7923으로 선형 회귀의 0.5686보다 우수하고, 평균 제곱 오차(MSE)도 0.00017417로 선형 회귀의 0.00036187보다 낮았습니다.

- **Performance Highlights**: 시험한 재료 중 AA6061이 최대 하중에서 0.1014 mm로 가장 높은 변위 민감도를 보였으나, Inconel718은 더 나은 구조적 안정성을 나타냈습니다. 이러한 결과는 PINN이 높은 정확도를 달성할 수 있음을 나타내며, 격자 구조 설계의 새로운 가능성을 제공합니다.



### Video-of-Thought: Step-by-Step Video Reasoning from Perception to Cognition (https://arxiv.org/abs/2501.03230)
Comments:
          Accepted by ICML 2024

- **What's New**: 이 논문은 복잡한 비디오에 대한 이해와 추론의 한계를 극복하기 위해 새로운 솔루션을 제시합니다. MotionEpic이라는 새로운 비디오 멀티모달 대형 언어 모델(Multimodal Large Language Model)을 소개하여, 비디오의 픽셀 수준 정밀한 공간-시간(Spatial-Temporal) 처리를 가능하게 합니다. 또한, Video-of-Thought (VoT)라는 추론 프레임워크를 개발하여, 복잡한 비디오 문제를 단순한 하위 문제로 나누어 단계별로 해결할 수 있도록 설계하였습니다.

- **Technical Details**: MotionEpic은 입력 비디오와 공간-시간 장면 그래프(Spatial-Temporal Scene Graph, STSG) 표현을 통합하여 비디오에 대한 정밀한 픽셀 수준의 공간-시간 접지를 달성합니다. 이 모델은 입력 비디오를 인코딩하고 이해하며 STSG를 생성하는 기능을 지원합니다. VoT 프레임워크는 비디오와 질문을 기반으로 가능한 대상(target)를 식별하고, 이들을 시간적으로 추적하여 행동 동역학을 이해하는 등 계층적인 추론 과정을 통해 복잡한 비디오 문제를 해결합니다.

- **Performance Highlights**: 우리는 비디오 질문-응답(Video Question Answering, QA) 벤치마크를 통해 MotionEpic과 VoT 프레임워크의 성능을 평가하였으며, 기존의 최첨단 성능을 불확실하게 개선하였습니다. 특히, 8개의 복잡한 비디오 QA 벤치마크에서 매우 명확한 마진을 통해 성능을 끌어올리며 새로운 상태(State-of-the-Art)를 수립하였습니다. 본 연구는 복잡한 비디오 이해 및 추론에서 지식과 인지력을 증진시키는 데 크게 기여할 것으로 기대됩니다.



### VLM-driven Behavior Tree for Context-aware Task Planning (https://arxiv.org/abs/2501.03968)
Comments:
          10 pages, 11 figures, 5 tables. Last updated on January 7th, 2024

- **What's New**: 본 논문에서는 로봇 커뮤니티에서 주목받고 있는 행동 트리(Behavior Trees, BT) 생성에 비전-언어 모델(Vision-Language Models, VLMs)을 활용하는 새로운 프레임워크를 제안합니다. 이 프레임워크는 시각적 조건에 맞춰 BT를 상호작용적으로 생성하고 편집할 수 있도록 해줍니다. 특히, '자기 유도 시각 조건(self-prompted visual conditions)'을 기반으로 하여 로봇이 시각적 정보에 따라 환경 인식 결정을 내릴 수 있도록 지원합니다.

- **Technical Details**: 이 프레임워크는 VLM을 통해 BT를 생성하며, 조건문과 같은 시각적 조건 노드를 포함합니다. 이러한 조건들은 자유 서식 텍스트로 표현되며, 로봇의 실행 중 실제 이미지와 대조하여 평가됩니다. 예를 들어, '테이블에서 컵을 치우세요'라는 지침을 바탕으로 VLM은 '테이블에 컵이 없는지 확인'하는 조건 노드를 생성합니다. 이를 통해 로봇은 복잡한 환경에서도 적절한 결정을 내릴 수 있게 됩니다.

- **Performance Highlights**: 제안된 프레임워크는 실제 카페 시나리오에서 유인 로봇을 이용하여 검증되었습니다. 시험 결과, 다양한 조건문을 포함한 BT들이 효과적으로 생성되고 실행되었음을 보여주었습니다. 또한, BT의 가시화 및 상호작용 편집을 통해 안전성과 투명성을 높이는 인터페이스를 개발하여, 로봇 프로그램의 신뢰성을 강화했습니다.



### Localizing AI: Evaluating Open-Weight Language Models for Languages of Baltic States (https://arxiv.org/abs/2501.03952)
Comments:
          This paper is accepted to NoDaLiDa/Baltic-HLT 2025

- **What's New**: 이 연구는 유럽연합(EU) 관할 지역 외부에서 호스팅된 상업적으로 이용 가능한 LLMs의 데이터 프라이버시 문제를 다룹니다. 연구팀은 리투아니아어, 라트비아어, 에스토니아어와 같은 소규모 언어들을 지원하는 로컬 배포 가능한 오픈웨이트 LLMs를 평가하고, 이 모델들이 기계 번역, 다중 선택 질문 응답(MCQA), 자유 형식 텍스트 생성을 수행하는 방식을 분석합니다.

- **Technical Details**: Meta의 Llama 3 및 Llama 3.1, Google의 Gemma 2, Microsoft의 Phi 3, Mistral의 NeMo와 같은 여러 모델을 다양한 크기와 정밀도로 평가합니다. 실험에서는 4bit, 8bit, 그리고 16bit의 정밀도에서 모델의 성능을 비교하며, 특정 언어에서의 번역 품질은 FLORES-200 기준 데이터셋을 통해 측정합니다. 평가 방식은 자동 평가를 위한 COMET 지표 및 수동 평가를 포함합니다.

- **Performance Highlights**: 연구 결과, Gemma 2와 같은 일부 모델이 상업적으로 이용 가능한 최고 모델과 유사한 성능을 보인 반면, 많은 LLMs는 이러한 언어에서 어려움을 겪고 있음을 확인했습니다. 특히, 이들 모델은 최첨단 번역 성능에 근접하는 모습을 보였지만, 20단어 중 1단어에서 발생하는 어휘 환각(lexical hallucinations) 오류가 여전히 문제로 지적되었습니다.



### Synthetic Data Privacy Metrics (https://arxiv.org/abs/2501.03941)
Comments:
          14 pages, 2 figures

- **What's New**: 최근 생성적 AI(Generative AI)의 발전으로, 인공지능 모델을 학습시키기 위한 합성 데이터(synthetic datasets)가 실제 데이터만큼 정확하게 생성될 수 있게 되었습니다. 이 연구에서는 합성 데이터의 경험적 개인 정보 보호(empirical privacy)를 측정하는 방법의 중요성을 강조하고 있으며, 여러 가지 새로운 개인 정보 보호 메트릭스(priacy metrics)가 발표되고 있지만, 표준화는 여전히 부족하다고 지적합니다.

- **Technical Details**: 합성 데이터는 컴퓨터 시뮬레이션 또는 알고리즘에 의해 생성된 정보로, 실제 데이터와의 명확한 연결고리가 없어야 하며, 적절한 개인 정보 보호 수준에서 생성된 경우에는 공격에 대한 안전 장치를 제공할 수 있습니다. 이 연구에서는 k-anonymity, PII(replay), 정확한 일치 수, 최근접 이웃 거리 비율(Nearest Neighbor Distance Ratio, NNDR) 등의 다양한 개인 정보 보호 메트릭스 및 이를 개선하기 위한 기술적 접근 방식을 검토합니다.

- **Performance Highlights**: 합성 데이터는 개인 정보 보호를 보장하면서도 유용성을 유지해야 하는 어려운 거래에서 높이 평가되고 있지만, 여전히 많은 연구가 필요합니다. k-anonymity와 같은 메트릭스는 이해하기 쉬운 반면, 실제로는 어떤 필드가 준식별자(quasi-identifiers)인지 파악하기 어려운 경우가 많습니다. 이 연구에서는 또한 DCR(Distance to Closest Record) 및 IMS(Identical Match Share)와 같은 메트릭스들이 개인 데이터를 보호하기 위해 어떻게 활용될 수 있는지를 보여줍니다.



### Not all tokens are created equal: Perplexity Attention Weighted Networks for AI generated text detection (https://arxiv.org/abs/2501.03940)
- **What's New**: 이번 연구에서는 Perplexity Attention Weighted Network (PAWN)이라는 새로운 AI 생성 텍스트 탐지 프레임워크를 제안합니다. PAWN은 LLM의 마지막 숨겨진 상태와 위치 정보를 사용하여, 텍스트 생성 과정에서 다음 토큰 분포 메트릭의 기여도를 동적으로 가중화합니다. 이러한 접근 방식은 고전적인 zero-shot 방법론을 개선하여, 작은 훈련 파라미터 수로도 효과적인 탐지가 가능하도록 합니다.

- **Technical Details**: PAWN은 다음 토큰 분포 메트릭을 보다 정교하게 집계할 수 있도록 설계되었으며, 언어 모델의 의미론적 정보와 위치 정보를 활용하여 각 토큰이 미치는 영향을 조정합니다. 이 접근 방식은 상대적으로 적은 자원으로도 훈련이 가능하며, 숨겨진 상태와 메트릭을 디스크에 캐시하여 요구되는 리소스를 크게 줄입니다. PAWN은 기존의 세밀하게 조정된 LLM에 비해 경쟁력 있는 성능을 보여줍니다.

- **Performance Highlights**: 실험 결과, PAWN은 고급 탐지 기법들과 비교하여 일반화 성능이 뛰어난 것으로 나타났습니다. 이는 새로운 도메인과 소스 모델에 대해서도 안정적인 결정 경계를 유지하며, 적대적 공격에 대한 저항력이 향상되었습니다. 또한, 다국어 기능이 있는 LLM을 사용할 경우, 감독 훈련이 진행되지 않은 언어에서도 좋은 성능을 발휘하는 것으로 연구되었습니다.



### Exploring the Potential of Large Language Models in Public Transportation: San Antonio Case Study (https://arxiv.org/abs/2501.03904)
Comments:
          This work is accepted to AAAI 2025 Workshop on AI for Urban Planning. arXiv admin note: substantial text overlap with arXiv:2407.11003

- **What's New**: 본 연구는 대형 언어 모델(LLM)을 공공 교통 시스템에 통합함으로써 도시 이동성을 개선할 기회를 탐색합니다. 샌안토니오의 교통 시스템을 중심으로 LLM의 잠재력을 분석하여, 경로 계획 최적화, 대기 시간 단축, 개인화된 여행 지원 등의 분야에서 효과를 기대하고 있습니다. 이번 연구는 LLM이 자원 할당 및 승객 만족도를 높일 수 있는 방법을 제시하며, 데이터 기반 의사결정의 중요성을 강조합니다.

- **Technical Details**: 연구에서는 OpenAI의 GPT 시리즈와 같은 대형 언어 모델이 복잡한 의사결정을 돕고 대량의 데이터를 분석하는 능력을 바탕으로 공공 교통 관리에 기여할 가능성을 평가합니다. 특히, GTFS(General Transit Feed Specification) 데이터를 활용하여 LLM의 경로 계획 및 승객 소통을 맡길 수 있는 방법을 모색하고 있습니다. 이를 위해 다른 ChatGPT 모델의 성능을 비교하고 교통 정보를 이해하는 능력을 테스트했습니다.

- **Performance Highlights**: 연구 결과, LLM은 공공 교통 시스템에 대해 복잡한 쿼리를 효과적으로 처리할 수 있으며, 실시간 정보 제공 및 개인화된 여행 지원을 통해 승객 경험을 향상시킬 수 있음을 보여줍니다. 그러나, 모델의 정확성을 높이기 위해서는 신중한 엔지니어링과 세부 조정이 필수적입니다. 샌안토니오 사례를 바탕으로 얻은 교훈은 다른 도시에서의 LLM 통합 시스템 개발에 중요한 참고자료가 될 것입니다.



### Explainable Reinforcement Learning via Temporal Policy Decomposition (https://arxiv.org/abs/2501.03902)
Comments:
          21 pages, 4 figures

- **What's New**: 이 논문에서는 강화 학습(Reinforcement Learning, RL) 정책의 설명 가능성을 시간적 관점에서 탐구합니다. Temporal Policy Decomposition (TPD)라는 새로운 접근 방법을 통해 각 행동과 관련된 기대 미래 결과(Expected Future Outcome, EFO)를 설명합니다. 다양한 시간 단계에서의 EFO를 기반으로 한 설명을 제공함으로써, 행동의 결과를 이해하는 데 도움을 줍니다.

- **Technical Details**: TPD는 고정 예측 수평에서 미래의 결과를 명확하게 이해할 수 있는 방법론으로, 강화 학습 정책의 결과를 시간적으로 분해합니다. 고정 수평의 시간적 차이 학습(Fixed-Horizon Temporal Difference, FHTD) 기법을 활용하여 최적 및 비최적 행동에 대한 EFO를 학습하고 contrastive explanations을 생성합니다. 이를 통해 RL 정책의 미래 전략과 보상 구성을 명확하게 이해할 수 있습니다.

- **Performance Highlights**: 실험을 통해 TPD가 RL 정책의 행동을 이해할 수 있는 직관적이고 신뢰할 수 있는 설명을 생성함을 입증했습니다. 이 방법은 인간 사용자의 기대와 결합된 보상 기능을 미세 조정하는 데에도 기여하며, 성과가 향상된 것입니다. TPD는 RL의 불확실성을 포괄적으로 고려한 설명 방법으로, 다양한 결과를 예측하는 데 도움을 줍니다.



### LLaVA-Mini: Efficient Image and Video Large Multimodal Models with One Vision Token (https://arxiv.org/abs/2501.03895)
Comments:
          Code: this https URL Model: this https URL

- **What's New**: 본 논문은 LMMs (Large Multimodal Models)의 효율성을 향상시킨 LLaVA-Mini를 소개합니다. 기존의 LMM 모델들이 비디오 및 이미지 이해에서 큰 연산 비용을 처리하는 것과 달리, LLaVA-Mini는 비전 토큰의 수를 최소화하여 연산 복잡도를 줄입니다. 특히, LLaVA-Mini는 단 하나의 비전 토큰만을 사용하면서도 높은 성능을 유지하는 혁신적인 접근 방식을 선택했습니다.

- **Technical Details**: LLaVA-Mini는 비전 데이터와 텍스트 정보를 사전에 융합하는 모듈을 도입하여, LLM (Large Language Model)으로 입력되는 비전 토큰의 수를 극도로 압축합니다. 이 모델은 실험을 통해 1개의 비전 토큰을 사용하여 576개 비전 토큰 대비 0.17%의 압축 비율을 보여주며, FLOPs를 77% 줄이고 GPU 메모리 사용량을 0.6MB로 낮춥니다. 이는 고해상도 이미지와 긴 비디오 처리에서 지연 시간을 줄이는 데 크게 기여합니다.

- **Performance Highlights**: LLaVA-Mini는 11개 이미지 기반 및 7개 비디오 기반 벤치마크에서 실험을 통해 LLaVA-v1.5보다 우수한 성능을 발휘하였습니다. 특히, 이미지 이해의 지연 시간을 100ms에서 40ms로 단축시킬 수 있었으며, 또한 24GB의 메모리를 가진 NVIDIA RTX 3090에서 10,000 프레임 이상의 긴 비디오를 처리할 수 있는 가능성을 보여주었습니다. 이러한 효율성을 통해 LLaVA-Mini는 실시간 다중 모달 상호작용의 새로운 길을 열었습니다.



### CL3DOR: Contrastive Learning for 3D Large Multimodal Models via Odds Ratio on High-Resolution Point Clouds (https://arxiv.org/abs/2501.03879)
- **What's New**: 최근 연구는 대형 언어 모델(LLMs)이 텍스트 작업에 한정되지 않고 오디오, 이미지, 비디오를 포함한 다양한 다중 모드(multi-modal) 모델로서 기능할 수 있음을 보여주고 있습니다. 특히 3D 대형 다중 모드 모델(3D LMMs)에 대한 연구가 주목받고 있으며, 이는 포인트 클라우드(point clouds)와 같은 고차원 데이터를 처리할 수 있는 잠재력에 의해 주도되고 있습니다. 그러나 기존의 훈련 데이터셋에서 시각적 및 텍스트 내용의 정보 밀도와 명확성이 부족하여 정확한 교차 모드 이해의 병목 현상이 발생하고 있다는 문제점이 있습니다.

- **Technical Details**: CL3DOR는 고해상도 포인트 클라우드에서 오즈 비율(odds ratio)을 활용한 3D 대형 다중 모드 모델을 위한 대조적 학습(Contrastive Learning) 방법론을 제안합니다. 이 방법은 각 객체에 대한 포인트 클라우드의 밀도를 높이며, 훈련 데이터셋에서 불필요한 응답을 처벌하는 정보가 많은 하드 네거티브 응답을 구축합니다. CL3DOR는 언어 모델링 손실의 보조항으로 오즈 비율을 포함하여 대조 학습을 용이하게 하며, 구조화된 3단계 훈련 파라다임을 통해 교차 모드 이해를 크게 향상시킵니다.

- **Performance Highlights**: CL3DOR는 3D 장면 이해 및 추론 벤치마크에서 최첨단 성능을 달성하며, 특히 정확도 및 F1 점수에서 기존 모델들의 성능을 크게 초월합니다. 또한 고해상도, 하드 네거티브 데이터셋 구축 및 목표 함수 관련 포괄적 실험을 통해 CL3DOR의 각 구성 요소의 중요성과 효과를 실험적으로 입증합니다. CL3DOR의 수행 성과는 3D LMMs의 언어 지침 이해력을 혁신적으로 변화시킬 것으로 기대되며, 고품질 훈련 데이터셋의 구축 방법을 제시합니다.



### Diffusion as Shader: 3D-aware Video Diffusion for Versatile Video Generation Contro (https://arxiv.org/abs/2501.03847)
Comments:
          Project page: this https URL Codes: this https URL

- **What's New**: 본 논문에서는 'Diffusion as Shader (DaS)'라는 새로운 비디오 생성 방식을 소개합니다. 기존의 방식들은 하나의 제어 유형에 제한되어 있었던 반면, DaS는 3D 제어 신호를 활용해 다양한 비디오 제어 작업을 수행할 수 있는 통합된 아키텍처를 제공합니다. 이 접근법은 특히 카메라 조작 및 콘텐츠 편집 시 정밀한 제어를 가능하게 하여 비디오 생성 과정의 유연성을 향상시킵니다.

- **Technical Details**: DaS는 비디오의 3D 모션을 정의하는 3D 추적 비디오를 제어 신호로 사용하여 비디오 생성에 3D 인식을 통합합니다. 이를 통해 비디오의 시간적 일관성을 크게 개선하며, 3D 포인트가 색상을 통해 서로 연결됨으로써 비디오 프레임 사이의 일관된 표현을 보장합니다. 이 방식은 복잡한 비디오 제어 작업을 가능하게 하며, 기존의 2D 기반 접근법의 한계를 극복합니다.

- **Performance Highlights**: DaS는 3일간 8개의 H800 GPU에서 10,000개 미만의 비디오를 사용하여 훈련하여 다양한 작업에서 강력한 제어 능력을 입증했습니다. 특히, 카메라 제어 및 모션 전이 작업에서는 기존 방식들보다 월등히 향상된 성능을 보였으며, 추가적인 작업인 메시-비디오 변환 및 객체 조작에서도 뛰어난 생성 품질을 실험적으로 확인할 수 있었습니다.



### SCC-YOLO: An Improved Object Detector for Assisting in Brain Tumor Diagnosis (https://arxiv.org/abs/2501.03836)
- **What's New**: 이 논문에서는 뇌 종양 감지를 위한 새로운 SCC-YOLO 아키텍처를 개발하였습니다. 이 아키텍처는 SCConv 주의(attention) 메커니즘을 YOLOv9에 통합하여 이미지 특성 학습을 향상시킵니다. SCC-YOLO는 다양한 주의 메커니즘과 YOLOv9 모델의 통합 효과를 연구하며, 두 개의 데이터셋(BR35H 및 Brain_Tumor_Dataset)을 사용했습니다.

- **Technical Details**: SCC-YOLO 아키텍처는 효율적인 합성곱 모듈(convolutional module)을 재구성하는 SCConv 모듈을 사용하여 특성 간의 공간 및 채널 중복성을 줄입니다. 이 방법은 YOLOv9의 성능을 더욱 향상시키는 데 기여합니다. 다양한 주의 메커니즘을 통합하여 뇌 종양 이미지 감지의 성능을 분석하였습니다.

- **Performance Highlights**: 브레인 종양 데이터셋인 BR35H에서 SCC-YOLO는 YOLOv9에 비해 mAp50에서 0.3% 향상을 보였으며, 자체 제작한 Brain_Tumor_Dataset에서는 0.5% 향상을 나타냈습니다. SCC-YOLO는 뇌 종양 감지 분야에서 최신 기술 수준의 성능을 달성했으며, 관련 소스 코드도 제공됩니다.



### TACLR: A Scalable and Efficient Retrieval-based Method for Industrial Product Attribute Value Identification (https://arxiv.org/abs/2501.03835)
- **What's New**: 이 논문에서는 e-commerce 플랫폼에서 제품 속성 값 식별(Product Attribute Value Identification, PAVI)의 새로운 접근 방식을 제안합니다. TACLR(Taxonomy-Aware Contrastive Learning Retrieval)이라는 최초의 검색 기반 방법을 도입하여, 제품 프로필과 후보 값을 임베딩으로 변환하고, 유사성을 기반으로 값을 검색합니다. 이 방법은 대량의 카테고리와 속성을 효과적으로 처리할 수 있습니다.

- **Technical Details**: TACLR은 PAVI를 정보 검색(task)으로 정의하며, 주어진 제품 항목에 대한 쿼리와 속성 분류(corpus)로서 작용합니다. 이 방법은 속성 및 카테고리로부터 후보 값을 선택하는 하드 네거티브 샘플링(hard negative sampling) 기법을 사용하여 차별화된 임베딩을 제공합니다. 또한, 동적 임계값(dynamic thresholds)을 도입하여 추론의 유연성을 높였습니다.

- **Performance Highlights**: 실험 결과 TACLR은 수많은 제품 목록과 속성을 효율적으로 처리할 수 있는 능력을 입증했습니다. 이 방법은 실제 상업적 환경에서 성공적으로 배포되어 매일 수백만 개의 제품 리스트를 처리하며, 동적으로 대규모 속성 분류를 지원합니다. TACLR은 또한 속성 값이 누락된 경우를 정확히 탐지하는 기능을 갖추고 있습니다.



### Three-dimensional attention Transformer for state evaluation in real-time strategy games (https://arxiv.org/abs/2501.03832)
Comments:
          9 pages, 5 figures

- **What's New**: 이번 연구에서는 RTS (Real-Time Strategy) 게임에서의 상황 평가에 대한 새로운 접근 방식을 제안합니다. 제안된 tri-dimensional Space-Time-Feature Transformer (TSTF Transformer) 아키텍처는 공간, 시간, 특성 주의력 모듈을 통해 전투 상황을 효과적으로 모델링합니다. 이 모듈들은 서로 독립적이지만 연결되어 있어 시간 및 다차원 특성을 보다 효과적으로 처리할 수 있습니다.

- **Technical Details**: TSTF Transformer는 8개의 레이어로 구성되어 있으며, 3,150개의 적대적 실험 데이터셋을 활용하여 성능을 검증하였습니다. 공간 주의력(spatial attention), 시간 주의력(temporal attention), 및 특성 주의력(feature attention)의 세 가지 모듈이 연계되어 복잡한 의사결정 환경에서의 정보를 효율적으로 처리합니다. 이 아키텍처는 기존 Timesformer보다 적은 파라미터 수(4.75M)를 요구하면서도 높은 성능을 보여줍니다.

- **Performance Highlights**: TSTF Transformer는 초반 게임(~4% 진행)에서 58.7%의 정확도를 달성하며, 이는 기존 Timesformer의 41.8% 성능을 크게 앞선 것입니다. 중반 게임(~40% 진행)에서의 정확도 또한 97.6%에 도달하며, 성능 변화도 낮은 표준 편차(0.114)를 유지합니다. 이러한 결과는 RTS 게임의 상황 평가에 새로운 통찰력을 제공하며 Transformer 기반의 다차원 시간 모델링에 대한 혁신적인 패러다임을 제시합니다.



### Deep Sylvester Posterior Inference for Adaptive Compressed Sensing in Ultrasound Imaging (https://arxiv.org/abs/2501.03825)
- **What's New**: 이번 논문에서는 초음파 영상의 획득 과정을 최적화하기 위한 새로운 적응형 서브샘플링 방법을 제안합니다. 이 방법은 Sylvester Normalizing Flow 인코더를 사용하여 부분 관찰 하에 대략적인 Bayesian posterior를 실시간으로 유추하며, 다음 영상 프레임과 서브샘플된 관측치 간의 상호 정보를 극대화하는 샘플링 계획을 수립합니다.

- **Technical Details**: 제안한 방법은 부분 관측값을 기반으로 한 정보 이득의 최대화를 목표로 하며, 이를 위해 딥 생성 모델과 빠른 Bayesian posterior 추론을 결합합니다. Bayesian posterior는 복잡한 상태 분포를 효과적으로 추정할 수 있도록 설계되며, 적은 횟수의 신호 전송으로 더 높은 해상도를 유지합니다. 서브샘플링 행렬은 각 초음파 프레임에 대해 순차적으로 최적화됩니다.

- **Performance Highlights**: EchoNet 심장 초음파 비디오 데이터 세트를 활용하여 제안된 방법이 균일 난수 샘플링 및 등距 간격 스캔 라인과 같은 기존의 방법에 비해 평균 절대 재구성 오류를 15% 개선함을 보였습니다. 또한, posterior 추론과 샘플링 계획 생성이 단 0.015초 만에 완료되어 실시간 2D 초음파 영상 응용에 적합한 속도를 제공합니다.



### Self-Adaptive ERP: Embedding NLP into Petri-Net creation and Model Matching (https://arxiv.org/abs/2501.03795)
- **What's New**: 이번 연구에서는 기업의 요구에 맞게 시스템을 자동으로 맞춤화할 수 있는 Self-Adaptive ERP Framework을 소개합니다. 이 프레임워크는 애드혹(adhoc) 방식의 수동 조정을 줄이며, ERP의 커스터마이제이션(customization) 효율성과 정확성을 높이는 데 기여합니다.

- **Technical Details**: (Self-Adaptive ERP Framework) 은 기업 프로세스 모델과 시스템 사용 분석을 활용하여 맞춤화를 자동화합니다. 인공지능(AI)과 자연어 처리(NLP)을 이용하여 Petri nets를 활용, 비즈니스 프로세스를 적응 가능한 모델로 변환하여 구조적(structural) 및 기능적(functional) 일치를 다룹니다.

- **Performance Highlights**: 이 프레임워크는 Design Science Research(DSR) 및 Systematic Literature Review(SLR)를 기반으로 구축되어, ERP 시스템이 비즈니스 요구 사항에 더욱 빠르고 유연하게 대응할 수 있도록 합니다. 이를 통해 ERP 컨설턴트에 대한 의존성을 최소화하며 기업의 경쟁력을 높이는 데 기여할 수 있습니다.



### SelectiveFinetuning: Enhancing Transfer Learning in Sleep Staging through Selective Domain Alignmen (https://arxiv.org/abs/2501.03764)
Comments:
          Accepted by ICASSP 2025

- **What's New**: 이번 논문에서 제안하는 SelectiveFinetuning 방법은 다양한 피험자와 환경에서의 EEG 데이터 변동성 문제를 다루고 있습니다. 데이터의 도메인 변화로 인해 모델의 정확성과 신뢰도가 저하되는 현상을 해결하기 위해, 선행 학습된 Multi Resolution Convolutional Neural Network (MRCNN)를 활용하여 EEG 특징을 추출하고, 지구 이동 거리(Earth Mover Distance, EMD)를 사용해 타겟 도메인과 유사한 출처 도메인 데이터를 선택합니다. 최종적으로, 선택된 출처 데이터로 모델을 미세 조정하여 도메인 변화가 있는 타겟 데이터의 성능 향상을 꾀합니다.

- **Technical Details**: 본 연구에서 제안한 방법의 핵심은 MRCNN 아키텍처를 사용하여 다양한 수면 단계와 관련된 EEG 신호 주파수의 특징을 효과적으로 추출하는 것입니다. 이 모델은 하위 대역과 고주파수를 위한 두 가지 convolutional branch로 구성되어 있어, 각 수면 단계에 적합한 특징을 포착합니다. 또한, 데이터의 일관성을 유지하기 위해 동일한 네트워크와 계층을 사용하여 출처 및 타겟 도메인 데이터를 일반적인 특징 공간에 매핑합니다.

- **Performance Highlights**: 실험 결과, 제안된 SelectiveFinetuning 방법은 기존의 기준 모델들을 능가하는 성능을 보였습니다. 특히, 다양한 환경에서 좌우되는 데이터 분포 변화에도 불구하고 모델의 강건성과 적응성을 크게 향상시켰습니다. 이 연구는 수면 단계 분류에 있어 최신 성과(State-of-the-Art, SOTA)를 달성하며, 실질적인 수면 연구에서의 응용 가능성을 높이고 있습니다.



### Self-adaptive vision-language model for 3D segmentation of pulmonary artery and vein (https://arxiv.org/abs/2501.03722)
Comments:
          8 pages,3 figures

- **What's New**: 이 논문은 폐 혈관 구조의 정확한 분할을 위한 새로운 접근 방식을 제안합니다. 최근 CLIP과 같은 사전 훈련된 비전-언어 모델(Vision-Language Model, VLM)의 발전을 이용하여, 적은 수의 주석된 데이터로 3D CT 스캔을 효과적으로 분할할 수 있는 방법을 모색했습니다. 제안된 방법은 Language-guided self-adaptive Cross-Attention Fusion Framework으로, 텍스트와 이미지 표현의 크로스 모달리티를 적응적으로 결합합니다.

- **Technical Details**: 이 방법은 CLIP을 강력한 특징 추출기로 활용하여 3D CT 스캔의 분할을 생성하며, 'self-adaptive learning' 전략으로 두 개의 임베딩 모달리티를 효과적으로 융합하는 특별히 설계된 어댑터 모듈을 제안합니다. 연구팀은 최대 718개의 주석된 CT 데이터로 구성된 가장 큰 폐 동맥-정맥 데이터 세트를 사용하여 실험을 진행했고, 이는 다양한 최신 방법들에 비해 높은 성능을 입증했습니다.

- **Performance Highlights**: 제안된 방법은 평균 DSC( Dice Similarity Coefficient) 점수 76.22%를 기록하며, 기존 최고의 방법인 nnU-Net에 비해 평균 9.03% 우수한 결과를 나타냈습니다. 이 연구는 사전 훈련된 비전-언어 모델을 활용한 의료 영상 분할 접근 방식의 최신 경향을 보여주며, 대량의 주석된 CT 스캔 데이터 기반의 성능 개선을 확립했습니다.



### Materialist: Physically Based Editing Using Single-Image Inverse Rendering (https://arxiv.org/abs/2501.03717)
Comments:
          code will be available at this http URL

- **What's New**: 이번 연구에서는 단일 이미지에서 시작하는 이미지 편집을 위한 새로운 방법을 제안합니다. 이 방법은 학습 기반 접근법과 점진적 미분 렌더링(progressive differentiable rendering)을 결합하여, 환경 맵 최적화와 재질 속성 개선을 통해 입력 이미지와 렌더링 결과를 밀접하게 일치시키는 것을 목표로 합니다. 기존의 다른 역 렌더링 방법들은 여러 개의 뷰를 필요로 하지만, 본 연구는 단일 이미지만으로도 가능하게 설계되었습니다.

- **Technical Details**: 이 접근법은 MaterialNet이라는 신경망을 사용하여 이미지에서 알베도(albedo), 거칠기(roughness) 및 금속성(metallic) 속성을 추출합니다. 그런 후 미츠바(Mitsuba) 렌더러를 활용하여 물리 기반 재질 편집을 수행하며, 그림자 및 글로벌 조명(global illumination)과 같은 정확한 물체-환경 간 상호작용을 달성합니다. 또한, 물체 투입(object insertion) 및 재조명(relighting)과 같은 작업을 지원하는 최적화된 재질 속성과 조명을 제공합니다.

- **Performance Highlights**: 제안된 방법은 Stable Diffusion 기반의 최신 모델보다 빠른 추론 속도를 보이며, 단일 뷰 미분 몬테 카를로(ray tracing)에서 환경 맵에 대한 우수한 결과를 도출합니다. 연구 결과는 제안된 방법이 기존 방법들보다 더 현실적인 빛의 굴절(light refraction)을 제공하며, 특별히 투명 물체의 재질 편집을 위한 혁신적인 기능을 강조합니다. 이는 기존의 방법들과 비교했을 때 강력한 해석 가능성을 추가합니다.



### Unsupervised Speech Segmentation: A General Approach Using Speech Language Models (https://arxiv.org/abs/2501.03711)
- **What's New**: 이번 논문에서는 기존의 접근 방식들인 Speaker Diarization을 바탕으로 하여 음성 세분화( Speech Segmentation)을 위한 비지도 학습 방법을 제안합니다. 이 방법은 다채로운 음향-의미적 스타일 변화를 처리하는 데 초점을 맞추며, 기존의 단일 스타일 변화 처리에서는 나아간 진전을 보여줍니다. 특히, 감정이나 화자처럼 텍스트로 잘 변환되지 않는 정보를 중심으로 한 세분화를 도모합니다.

- **Technical Details**: 제안된 방법은 음성과 오디오 신호를 이산 음향 단위로 표현한 후 언어 모델을 적용하여 순서의 가능성을 극대화하는 Speech Language Models (SLMs)를 활용합니다. 우리는 초기 단계로 오디오를 동등한 크기의 세그먼트, 즉 'acoustic-sentences'로 나눈 뒤, SLMs에서 얻은 확률을 사용해 연속된 문장을 점수화하고 최종적으로 병합할 세그먼트를 선택합니다. 이는 비지도 세분화에 있어 새로운 접근 방식을 제공하는 것을 목표로 합니다.

- **Performance Highlights**: 제안된 방법은 경계 탐지(boundary detection), 세그먼트 순수도(segment purity) 및 과다 세분화(over-segmentation) 측면에서 평가된 기준선보다 뛰어난 성능을 보였습니다. 연구 결과, 해당 접근 방식은 매우 효율적이며, 다양한 음향-의미적 개념의 세분화에서 우수한 성능을 입증했습니다. 코드와 세트업은 논문에 링크되어 제공됩니다.



### AuxDepthNet: Real-Time Monocular 3D Object Detection with Depth-Sensitive Features (https://arxiv.org/abs/2501.03700)
- **What's New**: AuxDepthNet은 외부 깊이 정보 없이 실시간 단안 3D 물체 탐지를 위한 새로운 프레임워크입니다. 이 모델은 Auxiliary Depth Feature (ADF) 모듈과 Depth Position Mapping (DPM) 모듈을 도입하여 깊이 감지 특성을 효과적으로 학습하고, 탐지 과정에 깊이 위치 정보를 통합합니다. 이러한 접근 방식은 알고리즘의 복잡성을 줄이면서도 실시간 성능을 보장합니다.

- **Technical Details**: AuxDepthNet 구조는 다면적 특성 표현을 활용하여 깊이 민감 특성(depth-sensitive features), 맥락 민감 특성(context-sensitive features) 및 깊이 유도 특성(depth-guided features)을 통합합니다. ADF 모듈은 보조 학습(auxiliary learning)을 통해 깊이 관련 힌트를 암시적으로 학습하여 외부 깊이 맵에 의존하지 않습니다. DPM 모듈은 위치 인코딩을 통해 탐지 프로세스에 깊이 관련 위치 정보를 내장하여 공간적 추론과 강력한 3D 위치 지정이 가능하도록 합니다.

- **Performance Highlights**: KITTI 데이터셋에서 AuxDepthNet은 $	ext{AP}_{3D}$ 측정 값에서 24.72% (쉬움), 18.63% (보통), 15.31% (어려움)을 기록하며 최첨단 성능을 달성했습니다. 추가적으로, $	ext{AP}_{	ext{BEV}}$ 스코어는 34.11% (쉬움), 25.18% (보통), 21.90% (어려움)으로 나타났습니다. 이러한 결과는 AuxDepthNet이 실시간 3D 객체 탐지에서 경쟁력 있는 솔루션임을 시사합니다.



### Exploring Molecule Generation Using Latent Space Graph Diffusion (https://arxiv.org/abs/2501.03696)
- **What's New**: 이 논문은 분자 그래프 생성에서 잠재 공간(diffusion in latent space) 확산을 탐구하며, 이는 기존의 그래프 신경망(GNN) 접근법에 비해 계산 효율성을 높이는 방법론입니다. 또한 다양한 생성 흐름 모델과 아키텍처에 대해 실험하고, 액체 상태의 분자를 더욱 효과적으로 생성하기 위한 명확한 전략과 최적의 하이퍼파라미터 선택이 중요함을 강조합니다.

- **Technical Details**: 연구 방법으로는 분자를 그래프 형태로 표현하며, 각 노드는 원자를, 엣지는 원자 간의 결합을 나타냅니다. 이 논문에서 사용된 인코더-디코더 구조는 PNA(Principal Neighbourhood Aggregation) 아키텍처를 기반으로 하며, EGNN(E(3)-Equivariant Graph Neural Networks)을 통해 회전, 변환, 반사 불변성을 유지합니다. 또한 세 가지 확산 프로세스를 적용하여, 가우시안 기반 확산, 열 방산 기반 확산, 플로우 매칭을 통해 분자 구조를 생성하는 방법을 명확히 설명합니다.

- **Performance Highlights**: 실험 결과, 모델은 QM9 데이터셋을 기반으로 훈련되었으며, 각각의 접근 방식과 설계 결정에 대해 높은 민감도를 보였습니다. 특히, GNN과 EGNN을 적용한 결과, 분자 생성의 정확도가 현저히 향상됐으며, 유사한 조건에서 다양한 분자 생성 작업에서 성능을 비교하여 중요한 인사이트를 제공합니다.



### MAJL: A Model-Agnostic Joint Learning Framework for Music Source Separation and Pitch Estimation (https://arxiv.org/abs/2501.03689)
- **What's New**: 이번 연구에서는 음악 정보 검색에 있어 중요한 두 가지 작업인 음악 소스 분리(MSS)와 피치 추정(PE)을 동시에 개선하기 위한 Model-Agnostic Joint Learning (MAJL) 프레임워크를 제안합니다. MAJL은 다양한 모델을 사용하여 두 작업을 해결할 수 있으며, 특히 레이블이 없는 데이터의 부족과 공동 학습 최적화의 어려움을 해결하는 데 중점을 두고 있습니다.

- **Technical Details**: MAJL 프레임워크는 두 단계의 훈련 방법 및 하드 샘플에 대한 동적 가중치(Dynamic Weights on Hard Samples, DWHS)를 포함합니다. 첫 번째 단계에서는 완전 레이블이 있는 데이터를 사용하여 모델을 초기화하고, 두 번째 단계에서는 생성된 의사 레이블과 단일 레이블 데이터로 모델을 재훈련하여 데이터의 양을 극대화합니다. DWHS는 예측 오류 전파 문제와 다양한 목표 간의 불일치를 해결하기 위해 설계되었습니다.

- **Performance Highlights**: 실험 결과, MAJL은 공공 음악 데이터셋에서 음악 소스 분리에서 0.92의 신호 왜곡 비율(Signal-to-Distortion Ratio, SDR) 향상과 피치 추정에서 2.71%의 일반 피치 정확도(Raw Pitch Accuracy, RPA) 개선을 보여 주며, 기존의 최첨단 기법들보다 우수한 성능을 발휘했습니다. 또한, 각 구성 요소의 효과를 검증함으로써 MAJL의 높은 일반성을 강조하고 있습니다.



### SLAM: Towards Efficient Multilingual Reasoning via Selective Language Alignmen (https://arxiv.org/abs/2501.03681)
Comments:
          Accepted by COLING 2025 (Oral)

- **What's New**: 이 논문에서는 대형 언어 모델(LLM)이 영어 추론 작업에서 크게 발전했지만 다국어 추론에서는 여전히 어려움을 겪고 있다는 점을 지적합니다. 연구자들은 다국어 이해를 통해 모델이 영어가 아닌 질문을 이해하도록 훈련하는 방법을 제안하였습니다. 하지만, 이 방법은 과도한 컴퓨팅 자원과 큰 정보 손실 문제를 안고 있습니다.

- **Technical Details**: 먼저, 모델의 다국어 이해를 개선하기 위해 비관련 레이어와 파라미터를 지나치게 조정하는 것이 큰 문제로 드러났습니다. 연구팀은 SLAM이라는 새로운 접근 방식을 제안하며, 이는 다국어 처리에 관여하는 레이어를 정확히 식별하고 미세 조정(fine-tune)합니다. SLAM은 7B 및 13B LLM 내에서 전체 파라미터의 6.5-8%만 조정하여 6개의 피드포워드 서브 레이어(feed-forward sub-layers)만을 미세 조정합니다.

- **Performance Highlights**: 경험적 결과에 따르면, SLAM은 10개 언어에서 모든 강력한 기준선(baselines)보다 우수한 평균 성능을 달성했습니다. 뿐만 아니라 SLAM은 단일 훈련 단계에서만 진행되어, 두 단계 방법에 비해 훈련 시간을 4.1-11.9배 단축시킵니다. 이는 SLAM이 다국어 모델의 효율성을 크게 향상시키는 방법임을 보여줍니다.



### SALE-Based Offline Reinforcement Learning with Ensemble Q-Networks (https://arxiv.org/abs/2501.03676)
Comments:
          10 pages, 2 figures, 4 tables

- **What's New**: 이번 연구에서는 오프라인 강화 학습 알고리즘 TD7을 기반으로 하여 앙상블 Q-네트워크(ensemble Q-networks)와 EDAC의 그래디언트 다양성 페널티를 통합한 모델 자유 액터-크리틱 알고리즘을 제안합니다. 이러한 통합 방식은 데이터셋에 기반한 행동을 초기에 유도하면서 훈련 과정의 안정성과 정확도를 개선하는 데 기여합니다. 실험 결과, 제안한 알고리즘은 기존 방법들보다 뛰어난 수렴 속도와 안정성을 보이며 성능이 우수함을 입증했습니다.

- **Technical Details**: 제안된 알고리즘 EDTD7은 행동 정책을 유도하는 행동 복제(behavior cloning) 항을 조절 가능하게 유지하면서 Q-앙상블의 정확도가 향상됨에 따라 그 영향력을 점차 줄여 나갑니다. 앙상블 Q-네트워크는 알려지지 않은 행동에 대한 페널티를 도입해 액터 네트워크가 데이터셋 내 행동에 집중하게 합니다. 동시에 그래디언트 다양성 페널티는 Q-값의 그래디언트를 다양화시키며, 알려지지 않은 행동의 과대 평가를 더욱 억제합니다.

- **Performance Highlights**: D4RL MuJoCo 벤치마크에서의 실험 결과, EDTD7은 기존 알고리즘에 비해 빠른 수렴 속도와 우수한 안정성을 보여주었습니다. 이로 인해 오프라인 강화 학습에서의 훈련 효율성을 크게 향상시킬 수 있음을 입증하였습니다. 연구자들은 이 알고리즘이 향후 다양한 환경에서의 적용 가능성이 높다고 평가하고 있습니다.



### Action Quality Assessment via Hierarchical Pose-guided Multi-stage Contrastive Regression (https://arxiv.org/abs/2501.03674)
- **What's New**: 이 논문은 새로운 Action Quality Assessment (AQA) 방법을 제안합니다. 기존의 방법이 고정된 프레임으로 비디오를 분할하는 데 집중한 반면, 제안된 방법은 계층적 포즈 안내를 통해 다단계 대조 회귀를 사용하여 더 정교한 기술을 사용합니다. 또한 정교한 인간 포즈 레이블을 포함한 FineDiving-Pose Dataset이 새롭게 만들어졌습니다.

- **Technical Details**: 제안된 방법은 다중 스케일 동적 비주얼-스켈레톤 인코더를 통해 세밀한 시공간 비주얼 및 골격 특징을 포착합니다. 이후 절차 분할 네트워크를 통해 서로 다른 하위 동작을 분리하고 세분화된 특징을 얻습니다. 이러한 특징들은 다중 모달 융합 모듈에 입력되어 모델이 세분화된 활동 유사성과 변화를 학습할 수 있도록 도와줍니다.

- **Performance Highlights**: FineDiving 및 MTL-AQA 데이터 세트에서 실험 결과는 제안된 방법의 효과성과 우수성을 입증합니다. 전반적으로, 실험을 통해 제안된 방법이 최신 방법들을 능가함을 보여주었습니다. 이는 AQA의 품질 평가 및 개선에 기여할 수 있는 새로운 접근 방식을 제공함을 의미합니다.



### A Diversity-Enhanced Knowledge Distillation Model for Practical Math Word Problem Solving (https://arxiv.org/abs/2501.03670)
- **What's New**: 이 논문에서는 Math Word Problem (MWP) 해결을 위한 새로운 다각화 지식 증류(Diversity-enhanced Knowledge Distillation, DivKD) 모델을 제안합니다. 기존의 Seq2Seq 모델과 그 변형들이 한정된 다양성 있는 해답 방정식을 생성하는 데 어려움을 겪고 있는 가운데, 우리의 접근법은 교사 모델로부터 선택적으로 고품질 지식을 전이하여 다양한 방정식을 학습하는 적응형 다양성 증류(AdaKD) 방법을 도입합니다.

- **Technical Details**: 제안된 DivKD 모델은 학생 모델이 다각화된 방정식을 캡처할 수 있게 돕기 위해, 조건부 변분 오토인코더(Conditional Variational Autoencoder, CVAE)를 통합하여 방정식의 다양성 분포를 모델링합니다. 이를 통해 다양한 해결 방정식을 생성할 수 있는 잠재 변수를 샘플링하는 다양성 사전 네트워크를 사용하며, 고품질 소프트 및 하드 레이블을 선택적으로 훈련하여 학생 모델의 학습을 돕습니다.

- **Performance Highlights**: 다양한 MWP 벤치마크 데이터셋에 대한 광범위한 실험을 통해 제안한 방법이 기존 강력한 기준 모델보다 더 높은 답안 정확도를 달성함을 입증하였습니다. 특히, DivKD 모델은 모델의 효율성을 유지하면서도 높은 성능을 발휘하는 것으로 나타났습니다.



### Effective and Efficient Mixed Precision Quantization of Speech Foundation Models (https://arxiv.org/abs/2501.03643)
Comments:
          To appear at IEEE ICASSP 2025

- **What's New**: 이 논문은 음성 기초 모델에 대한 새로운 혼합 정밀도 양자화(mixed-precision quantization) 접근 방식을 제안합니다. 이 접근 방식은 혼합 정밀도 학습(mixed-precision learning)과 양자화된 모델 파라미터 추정(quantized model parameter estimation)을 단일 모델 압축 단계로 통합하여 성능을 향상시킵니다. 실험 결과, 제안된 모델은 기존 방법들에 비해 손실 없는 압축 비율(lossless compression ratio)을 1.7배에서 1.9배 개선하고 단일 정밀도 모델에 비해 통계적으로 의미 있는 단어 오류률(word error rate, WER) 증가없이 우수한 성능을 보였습니다.

- **Technical Details**: 논문에서 제안한 혼합 정밀도 양자화 접근 방식은 음성 SSL 모델에 대한 양자화와 학습을 동시에 수행합니다. 연구진은 두 단계의 혼합 정밀도 모델을 단일 단계로 융합하여 신경망 아키텍처 검색(Neural Architecture Search, NAS)을 통해 각 레이어의 비트 폭을 자동으로 조정하는 방법을 사용했습니다. 이를 통해 압축 시간이 최대 1.9배 단축되었고, 성능 저하 없이 손실 없는 압축 비율을 달성했습니다.

- **Performance Highlights**: 제안된 4.6비트 wav2vec2.0-base 시스템과 3.5비트 HuBERT-large 시스템은 각각 6.4배와 8.6배의 최대 손실 없는 압축 비율을 기록했습니다. 두 모델 모두 이론적 정밀도 유지 없이도 단어 오류율(WER) 저하를 보여주었으며, 표준 방식에 비해 엄청난 시간 절약 효과를 발휘했습니다. 이로 인해 음성 인식 시스템의 실제 배포에 기여할 수 있는 가능성이 높습니다.



### MHGNet: Multi-Heterogeneous Graph Neural Network for Traffic Prediction (https://arxiv.org/abs/2501.03635)
Comments:
          Accepted by 2025 lEEE International Conference on Acoustics, speech, and signal Processing (lCASSP2025)

- **What's New**: 최근 교통 흐름 예측은 지능형 교통 시스템 관리에서 중요한 역할을 하고 있습니다. 본 논문에서는 MHGNet이라는 새로운 프레임워크를 제안하여 비유클리드 저차원 교통 데이터를 단순 그래프로 모델링하는 기존 방법의 한계를 극복하고자 합니다. MHGNet은 시공간 다형성 그래프를 모델링하여 노드 간의 유사한 트렌드를 효과적으로 포착합니다.

- **Technical Details**: MHGNet은 주기적 피처 행렬과 노드 임베딩 행렬의 기능 매핑을 통해 단일 패턴의 교통 데이터를 다중 패턴으로 디커플링합니다. 또한, 노드 클러스터링 알고리즘을 사용하여 고차원 특성 공간 내에서 노드 간의 유클리드 거리 계산을 통해 노드들을 클러스터링합니다. 이 구조는 시공간 융합 서브그래프 내에서 잔여 서브그래프 컨볼루션을 수행하여 다양한 시간 복잡도 O(N)으로 작동합니다.

- **Performance Highlights**: 본 논문에서는 MHGNet의 유효성을 검증하기 위해 네 가지 널리 사용되는 데이터 세트를 기반으로 광범위한 실험과 정량 평가를 수행하였습니다. 실험 결과, MHGNet은 기존의 여러 경쟁 모델들에 비해 월등한 성능을 보였습니다. 결론적으로, 본 연구는 이론적인 기여를 통해 교통 흐름 예측 분야의 시공간 모델링 방법론에 중요한 발전을 이루었습니다.



### RecKG: Knowledge Graph for Recommender Systems (https://arxiv.org/abs/2501.03598)
Comments:
          Accepted by The 39th ACM/SIGAPP Symposium On Applied Computing(SAC) 2024

- **What's New**: 이 연구는 다양한 추천 시스템 간의 지식 그래프(Knowledge Graph, KG) 통합에 대한 연구의 부족 문제를 해결하기 위해 RecKG라는 표준화된 지식 그래프를 제안합니다. RecKG는 서로 다른 데이터 세트에서 엔티티의 일관된 표현을 보장하며, 효과적인 데이터 통합을 위해 다양한 속성 유형을 지원합니다. 이 연구에서는 RecKG를 사용하여 실제 데이터 세트를 표준화하고 그래프 데이터베이스를 통해 응용 프로그램을 개발합니다.

- **Technical Details**: RecKG는 추천 시스템에 필수적인 엔티티로 구성되며, 사용자와 아이템에 초점을 맞추고 있습니다. 이를 통해 데이터 통합의 일관성을 보장하고 서로 다른 추천 시스템에서 동일한 개념과 속성을 균일하게 표현함으로써 통합 과정에서 중요한 속성을 최소화하는 것을 목표로 합니다. RecKG는 다양한 추천 시스템 간의 속성 커버리지를 보장하여 데이터 통합 시 누락되는 속성이 최소화되도록 설계되었습니다.

- **Performance Highlights**: RecKG의 효과성은 연속적인 정량적 평가를 통해 검증되며, 기존 연구와의 정성적 비교를 통해 상호 운용성(interoperability)의 성과를 확인합니다. 이 연구를 통해 지식 그래프 기반 추천 시스템의 데이터 통합 및 추천 품질이 크게 향상된 결과를 보여줍니다. RecKG의 도입으로 추천 시스템 간 정보의 더 많은 발견과 추가적인 의미적 정보의 통합이 가능해졌습니다.



### Cosmos World Foundation Model Platform for Physical AI (https://arxiv.org/abs/2501.03575)
- **What's New**: 이 논문은 Physical AI 구축을 위한 Cosmos World Foundation Model 플랫폼을 소개합니다. 이 플랫폼은 사용자 맞춤형 world model을 개발자가 만들 수 있도록 지원하며, 전반적인 world foundation model을 일반 목적의 모델로 정의하고 후속 응용 프로그램에 맞게 세부 조정할 수 있도록 합니다. 논문에서는 플랫폼의 다양한 구성 요소와 데이터 처리 파이프라인에 대해서도 설명합니다.

- **Technical Details**: Cosmos World Foundation Model 플랫폼은 비디오 기반의 WFM을 중심으로 하며, 비디오의 관찰을 모델 학습에 활용합니다. 기본적으로 pre-trained와 post-trained 모델로 나누어 보아, 전자는 다양한 비주얼 경험을 제공하고 후자는 특정 물리적 AI 환경에 맞춰 세부 조정됩니다. WFM의 개발은 transformer 기반의 diffusion 모델 및 autoregressive 모델을 포함하며, 이는 복잡한 비디오 생성 문제를 보다 쉽게 접근 가능하게 만듭니다.

- **Performance Highlights**: 논문에서 제시한 WFM 플랫폼은 다양한 Physical AI 작업을 위한 pre-trained 및 post-trained 모델을 제공하며, 사용자는 가상 환경에서 탐색할 수 있습니다. 또한, WFM의 활용을 통해 물리적 AI 개발자들이 정책 모델을 평가, 초기화 및 훈련하는 데 필요한 정보를 효과적으로 수집할 수 있습니다. 향후 연구를 통해 WFM의 정확성을 더욱 향상시키고 AI의 학습 능력을 고도화할 필요가 있습니다.



### From Code to Compliance: Assessing ChatGPT's Utility in Designing an Accessible Webpage -- A Case Study (https://arxiv.org/abs/2501.03572)
- **What's New**: 이번 연구에서는 ChatGPT(GPT-4o)의 웹 접근성 관련 능력을 평가했습니다. 일반적으로 웹사이트의 접근성과 관련하여 96%가 기준을 충족하지 않는다는 점에서, ChatGPT가 생성하는 코드가 얼마나 준수하는지를 분석했습니다. 더욱이, 효과적인 프롬프트 엔지니어링과 시각적 요소를 통합함으로써 접근성 문제 해결 능력을 향상시킬 수 있음을 발견했습니다.

- **Technical Details**: 연구는 TV 시리즈 웹페이지를 선택하여 다양한 웹 디자인 요소를 포함한 접근성 평가를 수행했습니다. 자동화 도구인 WAVE와 Axe를 사용해 웹페이지에서 발생한 접근성 문제를 분석하고, 수작업 검사도 병행하여 질적 데이터를 수집했습니다. ChatGPT의 문제 해결 과정에서 필요했던 피드백의 수와 문제의 복잡성을 반영한 가중 평균을 사용하여 효율성을 측정했습니다.

- **Performance Highlights**: 연구 결과, ChatGPT는 간단한 접근성 문제를 해결하는 데 강점을 보였으나 복잡한 문제에는 인간의 감독과 추가적인 반복 작업이 필요하다는 한계도 드러났습니다. 스크린샷을 제공함으로써 ChatGPT가 문제를 더 명확하게 인식하고 해결할 수 있는 가능성을 제시했습니다. 이러한 결과는 웹 개발자에게 더 포괄적인 웹사이트 설계를 위한 실용적인 지침을 제공합니다.



### Applying Large Language Models in Knowledge Graph-based Enterprise Modeling: Challenges and Opportunities (https://arxiv.org/abs/2501.03566)
- **What's New**: 최근 대규모 언어 모델(LLMs)의 역할이 학문적 연구에서 산업 응용으로 전환되고 있으며, 이는 기업 모델링의 기계 지원 생성에 중요한 구성 요소가 되고 있음을 보여줍니다. 본 논문에서는 지식 그래프 기반 접근 방식을 활용하여 기업 모델링에서 LLM의 잠재적 이점을 조사하고, 전문가 설문조사 및 ChatGPT-4o 기반 실험의 결과를 공유합니다. 또한 LLM 기반 모델 생성은 특정 작업에 대해 제한적이지만 변동성이 낮다는 것을 보여줍니다.

- **Technical Details**: 본 연구에서는 ArchiMate 모델 요소를 도메인 설명과 연결하는 작업에서 인간 전문가와 ChatGPT-4o의 성능을 조사하고자 하였습니다. 이를 위해 LLM 통합을 위한 다양한 방법론을 제시하며, 지식 그래프가 기업 모델의 자동 생성을 지원하는 데 어떻게 기여할 수 있는지를 설명합니다. LLM은 맥락상의 포함에 따라 두 단계 절차로 이전 경험을 통해 새로운 지식을 도출할 수 있는 잠재력을 지니고 있습니다.

- **Performance Highlights**: 실험 결과, 인간 모델링 전문가와 LLM의 결과가 다양한 상황에서 차이를 보였으며, 복잡한 작업에 대한 신뢰성이 감소하는 경향이 있음을 발견했습니다. 전문가의 감독과 개입이 필요하다는 설문조사 결과는 생성된 모델의 정확성과 완전성을 보장하기 위한 필수 요소임을 강조합니다. 이로 인해 LLM을 통한 기업 모델링의 품질과 프로세스가 개선될 가능성이 제시됩니다.



### Rethinking Adversarial Attacks in Reinforcement Learning from Policy Distribution Perspectiv (https://arxiv.org/abs/2501.03562)
Comments:
          10 pages, 2 figures, 2 tables

- **What's New**: 이번 논문에서는 Deep Reinforcement Learning (DRL)에서 나타나는 관측 신호의 불확실성과 부정확성을 해결하기 위해 Distribution-Aware Projected Gradient Descent (DAPGD) 공격을 제안합니다. 기존의 적대적 공격 방법이 개별 샘플 작업에 한정되어 전체 정책 분포에 미치는 영향이 제한적이었던 반면, DAPGD는 정책 네트워크를 공격하기 위해 분포 유사성을 활용합니다. 이 접근 방식은 전체 정책 분포의 정보를 활용하여 실제 환경의 방해 요소와 유사한 적대적 샘플을 생성할 수 있습니다.

- **Technical Details**: DRL은 마르코프 결정 과정(Markov Decision Process, MDP)으로 정식화되며, 상태 공간과 행동 공간 사이에서 확률 분포를 통한 정책 학습이 이루어집니다. DAPGD는 정책 유사성을 측정하기 위해 Bhattacharyya 거리(Bhattacharyya distance)를 사용하여 두 확률 분포 간의 중첩을 정량화하며, 이를 통해 미세하지만 중요한 정책 간의 차이를 감지합니다. 이 공격 방법은 샘플의 개별 특성이 아닌 정책의 전체 정보를 기반으로 공격을 수행합니다.

- **Performance Highlights**: 실험 결과 DAPGD는 세 가지 로봇 탐색 작업에서 최신 기법들과 비교하여 성과를 나타냈습니다. DAPGD는 최상의 기준선보다 평균 22.03% 더 높은 보상 감소를 기록하며, 특히 일반 모델과 강건 모델 모두에서 뛰어난 성능을 보였습니다. 이러한 결과는 DAPGD가 DRL 모델의 강건성을 평가하는 데 있어2 효율적임을 입증합니다.



### KG-TRICK: Unifying Textual and Relational Information Completion of Knowledge for Multilingual Knowledge Graphs (https://arxiv.org/abs/2501.03560)
Comments:
          Camera ready for COLING 2025

- **What's New**: 이 논문에서는 멀티링구얼 지식 그래프(Multilingual Knowledge Graphs)에서 텍스트 및 관계 정보의 완성을 통합하는 KG-TRICK이라는 새로운 모델을 제안합니다. KG-TRICK은 지식 그래프 완성(Knowledge Graph Completion, KGC)과 지식 그래프 향상(Knowledge Graph Enhancement, KGE)이라는 두 가지 독립적인 작업을 하나의 프레임워크로 통합하여 서로의 이점을 활용하도록 설계되었습니다. 또한, 10개 언어에서 25,000개 이상의 개체를 포함하는 WikiKGE-10++라는 대규모 수작업 벤치마크를 소개하여 멀티링구얼 KGs의 평가를 지원합니다.

- **Technical Details**: KG-TRICK은 텍스트와 관계 정보를 함께 완성하는 새로운 시퀀스-투-시퀀스(sequence-to-sequence) 모델로, 다양한 언어에서의 정보를 효율적으로 결합합니다. KGC와 KGE의 상호 의존성을 활용하여, KGC에서 얻은 언어 독립적인 관계 정보가 KGE의 질을 향상시키는 구조를 가지고 있습니다. 이를 통해 KG를보다 완전하게 만들고, 각각의 개체를 다른 언어의 이름과 설명에 맞춰 정렬하는 데 효과적입니다.

- **Performance Highlights**: KG-TRICK은 기존의 최첨단 모델보다 유사한 규모에서 더 우수한 성능을 보이며, 대규모 언어 모델에 비해서도 경쟁력 있는 성능을 달성합니다. 본 연구는 멀티링구얼 KGs의 질 향상과 다양한 NLP 응용 프로그램에서의 활용 가능성을 확장하는 데 기여할 것으로 기대됩니다. 이 모델과 함께 소개된 WikiKGE-10++는 향후 연구를 위한 중요한 자원으로 기능할 것입니다.



### PromptGuard: Soft Prompt-Guided Unsafe Content Moderation for Text-to-Image Models (https://arxiv.org/abs/2501.03544)
Comments:
          16 pages, 8 figures, 10 tables

- **What's New**: 이번 연구에서는 PromptGuard라는 새로운 콘텐츠 조절 기술을 제안합니다. 이는 대형 언어 모델(LLMs)의 시스템 프롬프트 메커니즘에서 영감을 받아 안전성을 확보하기 위해 설계되었습니다. 기존 T2I(Text-to-Image) 모델의 경우 NSFW 콘텐츠 생성을 방지하기 위한 직접적인 인터페이스가 없지만, PromptGuard는 텍스트 임베딩 공간 내에서 작동하는 안전 소프트 프롬프트를 최적화하여 이를 해결합니다.

- **Technical Details**: PromptGuard는 안전성 지침을 제공하기 위해 시스템 프롬프트 메커니즘을 T2I 모델의 구조나 매개변수를 수정하지 않고도 구현하는 방법을 찾습니다. 다양한 NSFW 카테고리 간의 보편적인 모더레이션을 달성하기 위해 NSFW 콘텐츠를 성적, 폭력적, 정치적 및 충격적인 내용으로 구분한 후, 각 유형별 안전 프롬프트를 최적화하고 이를 결합하는 방식을 채택합니다.

- **Performance Highlights**: PromptGuard는 이전의 콘텐츠 조절 방법보다 7.8배 빠르며, 최적의 NSFW 제거 비율인 5.84%를 달성하여 8가지 최첨단 방어 기법을 초월합니다. 또한, 비난적인 이미지를 단순히 블러링하거나 블랙 아웃하는 대신, 현실감 있는 이미지를 안전하게 생성하는데 기여합니다.



### Deep Learning within Tabular Data: Foundations, Challenges, Advances and Future Directions (https://arxiv.org/abs/2501.03540)
- **What's New**: 이번 논문은 다양한 하위 작업에서의 효과성을 강조하며 표 데이터(tabular data) 표현 학습(representation learning) 방법에 대한 포괄적인 검토를 제공합니다. 이는 훈련 데이터(training data), 신경망 아키텍처(neural architectures), 학습 목표(learning objectives)를 포함한 세 가지 기본 요소에 중점을 둡니다. 특히, 기존의 연구들과는 달리 이 연구는 표현 학습 방법의 보편성과 강건성을 강조하고 있습니다.

- **Technical Details**: 표 데이터는 행과 열로 구성된 구조화된 형식으로 배치되며, 각 행은 개별 샘플 레코드를 나타내고 각 열은 특성 관측 값을 의미합니다. 이 논문에서 제시된 신경망 아키텍처는 열 내의 불규칙한 패턴과 복잡한 열 간 상관관계를 캡처하도록 특별히 설계되었습니다. 또한, 데이터 증강(data augmentation) 및 생성(generation) 기법이 훈련 데이터의 질과 양을 향상시키기 위해 도입되었습니다.

- **Performance Highlights**: 127개의 최신 연구를 기반으로 한 시스템atic 문헌 검색을 통해 표 데이터 표현 학습에서의 주요 경향과 차이점이 분석되었습니다. 특히 자가 지도 학습(self-supervised learning) 및 트랜스포머 기반 모델의 적응에 대한 관심이 증가하고 있는 점이 강조되었습니다. 이 연구는 향후 표 데이터 표현 방법 개발을 위한 유망한 방향성과 주요 연구 격차를 제시합니다.



### Vocal Tract Length Warped Features for Spoken Keyword Spotting (https://arxiv.org/abs/2501.03523)
- **What's New**: 이 논문에서는 음성 키워드 탐지(Keyword Spotting, KWS)에 음성관 길이(Vocal Tract Length, VTL) 왜곡 특징을 포함하는 여러 방법을 제안합니다. 첫 번째 방법은 VTL 독립 KWS로, 다양한 왜곡 계수를 사용하는 단일 심층 신경망(DNN)을 훈련하는 방식입니다. 이 방법에서는 훈련 시마다 특정 VTL 특징이 무작위로 선택되며, 테스트 시 여러 왜곡 계수의 VTL 특징이 DNN과 비교되어 결과가 결합됩니다.

- **Technical Details**: 제안한 방법은 세 가지로 나뉘며, 첫 번째는 VTL 독립 KWS입니다. 두 번째는 보통의 특징을 DNN에 적용하고, 세 번째는 VTL 왜곡 특징을 결합하여 고차원 특징 벡터를 만드는 VTL 결합 KWS입니다. 각 방법은 훈련과 테스트에서 DNN 아키텍처를 활용하여 KWS의 성능을 향상시키는 데 초점을 둡니다.

- **Performance Highlights**: 제안된 방법들은 영어 Google Command 데이터 세트에서 평가되었으며, VTL 왜곡 특징을 사용하지 않는 기존 방법에 비해 KWS 정확도가 향상되었음을 보여줍니다. VTL 독립 KWS는 다양한 VTL 왜곡 특징의 정보를 누적하여 단일 DNN 구조로 효과적으로 성능을 개선하고 있습니다.



### Can Deep Learning Trigger Alerts from Mobile-Captured Images? (https://arxiv.org/abs/2501.03499)
- **What's New**: 우리 연구는 모바일 카메라 이미지 데이터를 활용하여 실시간으로 공기 질을 평가하고 추천하는 새로운 접근 방식을 제안합니다. 이 접근 방식은 회귀 기반의 Convolutional Neural Network(CNN) 모델을 개발하여, 서로 연결된 출력 파라미터를 활용한 예측을 통해 기존 모델들보다 우수한 성과를 보여줍니다. 추가로, 데이터셋을 증강하여 훈련 과정에서의 변화를 검증하는 중요한 기여를 하였으며, 원본 데이터셋과 증강 데이터셋 간의 정확도 차이가 미미함을 나타냈습니다.

- **Technical Details**: 이 연구에서 제안된 CNN 기반 회귀 모델은 공기 질 예측을 위한 특별한 설계를 갖추고 있습니다. 특히, PM2.5, NO2, SO2 및 CO와 같은 공기 질 메트릭을 이미지 데이터로 분석하며, 이는 사용자 건강 상태에 적합한 장소 추천에 큰 도움이 됩니다. 또한, 'HealthCamCNN'이라 불리는 실시간 사용자 친화적인 대시보드가 구현되어, 모바일 카메라 이미지에서 유래된 공기 질 지수와 오염 물질 값을 동적으로 표시합니다.

- **Performance Highlights**: 우리가 제안한 모델은 2종 및 5종 오염 물질에 대해 각각 0.0077 및 0.0112의 평균 제곱 오차(Mean Squared Error, MSE)를 달성하여, 기존의 방법과 비교해 우수한 성과를 보여줍니다. 이는 우리의 접근 방식이 공기 질 예측 및 사용자 맞춤형 공기 질 모니터링에서 실질적인 해결책을 제공함을 시사합니다. 궁극적으로, 이 연구는 환경 건강 및 웰빙 결정을 내리는 데 있어 개인에게 필요한 정보를 제공하는 데 기여합니다.



### Can LLMs Design Good Questions Based on Context? (https://arxiv.org/abs/2501.03491)
- **What's New**: 이 논문은 LLM(대형 언어 모델)이 생성한 질문의 특성과 이를 인간이 생성한 질문과 비교하여 평가하는 새로운 접근 방식을 소개합니다. 자동화된 LLM 기반 평가 방법을 개발하였으며, 질문의 길이, 유형, 맥락 범위 및 답변 가능성과 같은 다양한 차원에서 질문을 분석하였습니다. LLM이 생성한 질문의 고유한 특성을 밝히며, 이는 질문 품질 및 후속 응용 프로그램의 연구에 기여할 수 있습니다.

- **Technical Details**: 연구에서, LLM은 문맥(C)과 질문 생성 프롬프트(P)의 결합을 통해 질문을 생성합니다. 이 과정에서 LLM이 생성한 질문의 길이 및 유형을 통계적으로 분석하고, 질문이 특정 맥락과 어떻게 관련되는지를 평가합니다. 인간의 질문과 비교하기 위해, LLM이 생성한 질문의 길이를 측정하고, 질문의 응답 가능성을 평가하기 위한 새로운 방법론을 도입하였습니다.

- **Performance Highlights**: 실험 결과, 두 가지 대표적인 LLM인 GPT-4o와 LLaMA-3.1-70b-Instruct를 사용하여 각 모델이 동일한 256개의 위키 문맥에서 1,024개의 질문을 생성했습니다. 평가를 통해 인간 주석자와 LLM 간의 일치율이 80%를 초과했으며, 평균 피어슨 상관관계는 인간 주석자와 LLM 간에 0.77로 나타났습니다. 이런 결과는 질문 생성 및 평가에 대한 LLM의 신뢰성을 강조합니다.



### Align-Pro: A Principled Approach to Prompt Optimization for LLM Alignmen (https://arxiv.org/abs/2501.03486)
Comments:
          27 pages, Accepted in AAAI 2025

- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 인간 가치 동조를 위한 새로운 접근법으로, 강화 학습(reinforcement learning) 기반의 방법들 외에 prompt optimization(프롬프트 최적화)를 제안합니다. 기존의 방법들이 모델 파라미터 수정을 필요로 하고 비효율적일 수 있는 반면, 프롬프트 최적화는 이러한 제약을 피하며 유용한 대안이 될 수 있습니다. 저자들은 이 프롬프트 최적화를 최적화 문제로 설정하여 이론적 통찰력을 제공하고자 하며, 이는 model alignment 방향에 대한 새로운 문을 열수 있습니다.

- **Technical Details**: 이 논문에서는 Align-Pro라는 통합 최적화 프레임워크를 개발하여 프롬프트 최적화를 분석합니다. 특히, 서브옵티말리티(bounds)와 상관관계에 대해 연구하여, 프롬프트 최적화 과정에서 발생할 수 있는 응답과 미세 조정된 모델의 성과 간의 차이를 정량화합니다. 이를 통해 기존 fine-tuning 방법과 비교했을 때 프롬프트 최적화가 얼마나 효과적인지를 이론적으로 뒷받침합니다.

- **Performance Highlights**: 저자들은 세 가지 데이터셋에서 여러 실험을 수행하여 Align-Pro의 성과를 검증하였습니다. 실험 결과, Align-Pro는 평균 보상과 승률이 향상되어, 미세 조정 없이도 LLM 동조에 있어 효과적인 성능을 보임을 보여주었습니다. 이러한 결과는 프롬프트 최적화가 LLM의 동조 성능 개선에 있어 매우 유망한 접근법임을 시사합니다.



### Reading with Intent -- Neutralizing Inten (https://arxiv.org/abs/2501.03475)
- **What's New**: 이 연구는 대형 언어 모델(LLM)의 상황 맥락에서 다양한 감정의 톤이 모델 성능에 미치는 영향을 평가하는 Reading with Intent 작업을 확장했습니다. 기존의 비판적인 유머(sarcasm)에 대한 연구에 기반하여, 우리는 새로운 데이터셋을 $11$ 가지의 감정으로 변환해 생성했습니다. 이 데이터셋을 사용하여 특정 감정 톤으로 텍스트를 조정할 수 있는 감정 변환 모델(emotion translation model)을 개발 하였습니다.

- **Technical Details**: 이 논문에서는 Open-domain QA를 기반으로 한 데이터셋 생성을 통해 LLM의 감정 변환 과정을 수립하였습니다. 각 쿼리에 대해 Wikipedia에서 최대 10개의 관련 문서를 검색한 후, 특정 감정으로 변환하는 과정을 거칩니다. 총 $11$ 가지의 감정을 구현하는 과정에서는 Llama 3, Qwen 2.5와 같은 여러 서로 다른 아키텍처의 LLM을 활용하여 편향을 줄였습니다.

- **Performance Highlights**: 인간 평가 결과, 감정 변환을 통해 교육된 LLM은 합성으로 생성된 데이터에서 이익을 얻었습니다. Reading with Intent 작업에 적용한 결과, 논문은 비판적인 유머가 포함된 문장이 중화(neutralized) 되었을 때, 과제의 전반적인 결과가 약 $3	extrm{	extbf{	extpercent}}$ 향상됨을 보여주었습니다. 이 연구는 감정 변환을 통해 LLM의 수행능력을 효과적으로 개선하는 가능성을 제시합니다.



### MTRAG: A Multi-Turn Conversational Benchmark for Evaluating Retrieval-Augmented Generation Systems (https://arxiv.org/abs/2501.03468)
- **What's New**: 최근 대규모 언어 모델(LLM)에서 Retrieval-augmented generation (RAG) 작업이 큰 인기를 끌고 있습니다. 그러나 다중 회차 대화에 대한 평가가 간과되어 왔으며, 이는 시스템이 전 대화의 맥락에서 질문에 응답해야 하므로 추가적인 도전 과제가 존재합니다. 본 논문에서는 mtRAG라는 인간 생성의 다중 회차 RAG 벤치마크를 제시하며, 전체 RAG 파이프라인의 평가를 위해 다양한 실제 속성을 반영하고 있습니다.

- **Technical Details**: mtRAG는 110개의 대화를 포함하고 있으며, 각 대화는 평균적으로 7.7 회차로 구성되어 있습니다. 벤치마크는 사용자 경험을 시뮬레이션하기 위해 인간 주석자가 실시간으로 RAG 에이전트와 상호작용하여 생성합니다. RAG 시스템의 탐색 성능을 평가하고, 9개의 LLM 모델의 생성 성능을 다각도로 분석한 결과, 모든 모델이 질의 및 대화의 후반부에서 어려움을 겪음을 확인했습니다.

- **Performance Highlights**: mtRAG의 객관적인 평가 결과, 최신 LLM 기반 RAG 시스템도 이 벤치마크의 작업에서 고전하는 모습을 보였습니다. 특히 잘못된 응답이나 답할 수 없는 질문에 대한 처리에서 어려움이 있었습니다. 벤치마크는 인공지능 커뮤니티에게 두 가지 유형의 데이터(인간 생성과 합성 생성)의 상대적인 이점을 분석하고 이해하는 데 도움을 줄 것입니다.



### LHGNN: Local-Higher Order Graph Neural Networks For Audio Classification and Tagging (https://arxiv.org/abs/2501.03464)
- **What's New**: 이번 논문은 Local-Higher Order Graph Neural Network (LHGNN)을 제안하여 오디오 데이터의 복잡한 패턴과 관계를 효과적으로 처리하는 방법을 소개합니다. LHGNN은 그래프 기반 모델로, Fuzzy C-Means 클러스터링을 통해 로컬 이웃 정보와 고차원 데이터를 통합하여 오디오 객체를 식별하는 데 필요한 고차 관계를 캡처합니다. 이는 기존 Transformer 모델의 한계를 극복하고, 매개변수를 대폭 줄여가면서도 높은 성능을 유지합니다.

- **Technical Details**: LHGNN은 입력된 mel-spectrogram을 처리하기 위해 convolutional layers와 k-NN 알고리즘을 활용하여 지역 관계를 모델링합니다. 그리고, Fuzzy C-Means 클러스터링을 통해 고차 관계를 다룸으로써 오디오 데이터에서 다중 스케일 관계를 효과적으로 모델링할 수 있습니다. 이 모델은 전통적인 Transformer와 그래프 기반 방법에서의 쌍별 상호작용을 초월하여 더욱 강력한 표현력을 가집니다.

- **Performance Highlights**: 모델의 평가 결과는 공개된 3개의 오디오 데이터셋에서 Transformer 기반 모델보다 우수한 성능을 보여줍니다. 특히, ImageNet의 대규모 사전 학습 없이도 전반적인 효율성 및 효과성을 입증하였으며, 데이터가 부족한 환경에서도 잘 작동합니다. 이는 LHGNN이 다양한 상황에서 유용하게 사용할 수 있음을 나타냅니다.



### Radar Signal Recognition through Self-Supervised Learning and Domain Adaptation (https://arxiv.org/abs/2501.03461)
Comments:
          5 pages, 9 figures

- **What's New**: 이번 연구에서는 자가 감독 학습(self-supervised learning, SSL) 접근법을 도입하여 제한된 레이블이 있는 전자전 환경에서 레이더 신호 인식 성능을 개선하는 방법을 제안합니다. 구체적으로, 다양한 RF(Radio Frequency) 도메인에서 수집된 I/Q 신호를 기반으로 마스킹 신호 모델링(masked signal modelling)을 활용하여 모델을 사전 훈련(pre-training)한 후, 이를 레이더 도메인으로 전이(transfer)합니다. 이 과정에서는 적은 훈련 샘플을 가지고도 효과적으로 동작하는 Lightweight ResNet 모델을 사용하여 레이더 신호 분류의 정확도를 대폭 향상시켰습니다.

- **Technical Details**: 제안된 SSL 프레임워크는 마스킹 자동 인코더를 통해 출발 도메인에서 비지도(pre-trained)학습을 진행한 후, 목표 도메인에서 작고 제한된 데이터량으로 미세 조정(fine-tuning)하는 두 단계로 구성됩니다. 모델은 비지도 방식으로 원래 신호를 재구성하도록 훈련됩니다. 마스킹 전략으로는 랜덤 제로 마스킹(random zero-masking), 블록 제로 마스킹(random block zero-masking), 랜덤 노이즈 마스킹(random noise-masking), 블록 노이즈 마스킹(block noise-masking) 등의 다양한 기법이 사용되어 신호의 특징을 학습합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 목표 도메인에서 한 번의 샷(classification)으로 최대 17.5%의 정확도 개선을 이루었습니다. 또한, 다른 도메인에서 사전 훈련된 경우에도 최대 16.31%의 성과를 보였습니다. 이러한 성과는 레이더 신호 분류의 새로운 기준을 설정하며, 제한된 데이터 환경에서의 RSR 성능을 획기적으로 향상시킴을 입증하였습니다.



### Activating Associative Disease-Aware Vision Token Memory for LLM-Based X-ray Report Generation (https://arxiv.org/abs/2501.03458)
Comments:
          In Peer Review

- **What's New**: 이번 논문에서는 의료 보고서 생성을 위한 새로운 협동 기억 향상 모델인 AM-MRG(Associative Memory Augmented X-ray Medical Report Generation)를 제안합니다. 이 모델은 X-ray 이미지를 분석하여 중요한 질병 정보를 정확하게 인식하고, 과거 보고서 정보를 연관시켜 현재 보고서 작성에 도움을 줍니다. 기존의 대형 언어 모델(LLM)을 활용하며, 보다 정교한 시각 정보 분석을 통해 고품질의 의료 보고서를 생성할 수 있습니다.

- **Technical Details**: AM-MRG는 두 단계로 이루어져 있습니다. 첫 번째 단계에서 Swin Transformer 네트워크를 사용하여 X-ray 이미지의 시각 특징을 추출하고, Q-Former를 통해 질병 검색 쿼리와 함께 비주얼 특징을 증강합니다. 두 번째 단계에서는 모던 호프필드 네트워크(Modern Hopfield Network)를 활용하여 질병 인식과 관련된 비주얼 토큰을 지식베이스로 삼고, 보고서 기억을 검색하여 LLM을 기반으로 한 보고서 생성을 진행합니다.

- **Performance Highlights**: 제안된 AM-MRG는 IU X-ray, MIMIC-CXR, Chexpert Plus의 여러 벤치마크 데이터셋에서 최신 성능을 달성하였습니다. 실험 결과, 기존 모델에 비해 질병 정보를 보다 명확하게 설명하는 보고서를 생성함으로써 의료 현장에서의 유용성을 증대시켰습니다. 따라서, 이 연구는 X-ray 기반의 의료 보고서 생성 분야에서 중요한 발전을 이룬 것으로 평가됩니다.



### Optimization Learning (https://arxiv.org/abs/2501.03443)
- **What's New**: 본 논문은 최적화 학습(optimization learning)의 개념을 소개하며, 이는 매개변수 최적화 문제의 입력/출력 매핑을 학습하는 방법론입니다. 최적화 프로시(proxy)라고 불리는 이 방법은 설계상 신뢰할 수 있으며, 근본적인 최적화 문제에 대한 타당한 솔루션을 계산하고, 반환된 솔루션에 대한 품질 보증을 제공하며, 대규모 문제에도 적용될 수 있습니다. 이 논문은 최적화 프로시를 자가 지도(self-supervised) 방식으로 훈련할 수 있는 방법론을 제시합니다.

- **Technical Details**: 최적화 학습의 기본 아이디어는 최적화 문제의 입력을 최적 솔루션으로 매핑하는 것입니다. 이 논문은 세 가지 기본 방법론—프라이멀 최적화 프로시, 듀얼 최적화 프로시, 프라이멀-듀얼 학습—을 제안합니다. 프라이멀 최적화 프로시는 딥러닝(deep learning)과 수리(repair) 레이어의 조합을 통해 매개변수 최적화 문제에 대한 실행 가능(near-optimal) 솔루션을 제공합니다. 이 방법론은 전력 시스템에서의 실시간 위험 평가와 보안 제약 최적 전력 흐름에 적용되어 효율성을 대폭 개선하는 사례를 보여줍니다.

- **Performance Highlights**: 최적화 학습을 사용하면 기존의 최적화 기술이 아닌 새로운 수준의 효율성을 달성할 수 있습니다. 본 논문에서 정의한 최적화 솔루션을 통해 전력 시스템 운영에서 실시간으로 높은 정확도로 문제를 해결할 수 있으며, 이는 도체 컨틴전시 조건 하에서도 가능합니다. 이 논문은 여러 응용 분야에서 최적화 학습의 잠재력을 강조하며, 특히 실시간 리스크 평가와 관련된 내용에 주목합니다.



### SALT: Sales Autocompletion Linked Business Tables Datas (https://arxiv.org/abs/2501.03413)
Comments:
          Table Representation Learning Workshop at NeurIPS 2024

- **What's New**: 본 연구에서는 다중 테이블 데이터와 연결된 비즈니스 테이블을 다루기 위한 새로운 SALT(Sales Autocompletion Linked Business Tables) 데이터셋을 소개합니다. 이 데이터셋은 기업 자원 관리(ERP) 시스템에서 가져온 것으로, 실질적인 비즈니스 사용 사례에 활용할 수 있는 방대한 양의 연결된 테이블을 제공합니다. SALT 데이터셋은 판매 관련 여러 개의 관계형 테이블을 포함하고 있어, 이에 대한 연구 및 모델 개발이 가능한 기초 자료를 제공하게 됩니다.

- **Technical Details**: SALT 데이터셋은 사용자가 판매 주문을 생성할 때 채워야 하는 필드를 예측하기 위해 설계되었습니다. 데이터셋은 판매 문서, 판매 문서 항목, 고객 및 주소의 네 가지 주요 테이블로 구성되어 있으며, 총 500,908개의 판매 주문과 2,319,944개의 판매 주문 항목이 포함되어 있습니다. 이 데이터셋은 시간적으로 분할되어 검증 및 테스트 세트를 구성하며, 판매 데이터는 2018년 1월 1일부터 2020년 12월 31일까지의 거래를 포함합니다.

- **Performance Highlights**: SALT 데이터셋은 실제 산업 데이터에서 유래하여 판매 주문의 예측 모델링에 유용하게 활용될 수 있습니다. 특정 데이터 필드에서 다양한 고유 값이 나타나고 있으며, 특히 클래스 불균형 문제가 존재합니다. 이는 판매 오피스의 분포가 매우 편향되어 있을 수 있음을 시사하며, 이러한 특징은 데이터 모델 개발에 있어 중요한 요소로 작용할 것입니다.



### BoundingDocs: a Unified Dataset for Document Question Answering with Spatial Annotations (https://arxiv.org/abs/2501.03403)
- **What's New**: 이 논문은 Document Question-Answering (QA)를 위한 통합 데이터셋을 제시합니다. 여러 공개 데이터셋을 결합하여 정보를 추출하고, 문서에서 답변을 찾는 위치를 바운딩 박스로 포함시켰습니다. 이를 통해 대형 언어 모델(Large Language Models, LLMs)의 훈련과 평가에 적합한 자원을 제공합니다.

- **Technical Details**: 본 연구에서는 기존의 Document AI 작업을 QA 작업으로 재정의하고, 기존 데이터셋의 위치 정보를 통합하는 방법에 대해 논의합니다. 연구 질문은 QA 포맷으로의 데이터셋 통합, LLM에 의해 생성된 질문의 재구성으로 정확도를 높일 수 있는지, 레이아웃 정보 포함이 모델 성능에 미치는 영향을 다룹니다. 또한 머신 러닝과 딥러닝 기법, 특히 자연어 처리(NLP)와 관련된 방법들이 강조됩니다.

- **Performance Highlights**: 다양한 프롬프트 기법에 따른 모델의 성능을 평가하여 문서 이해에 가장 효과적인 접근 방식을 식별합니다. 기존의 Document AI 데이터셋들이 효과적으로 위치 정보를 통합하지 못해 hallucination을 줄이고 성능 향상을 저해한 점을 지적하며, 새로운 데이터셋은 더 나은 문서 이해를 지원할 것으로 기대됩니다.



### Enhanced Importance Sampling through Latent Space Exploration in Normalizing Flows (https://arxiv.org/abs/2501.03394)
Comments:
          Accepted at AAAI 2025

- **What's New**: 이번 논문에서는 중요 추출(Importance Sampling) 기법을 개선하기 위해 정규화 흐름(Normalizing Flow)의 잠재 공간(Latent Space)에서 제안 분포를 업데이트하는 방법을 제안합니다. 이 방법은 기존의 제안 분포가 목표 분포를 효과적으로 커버하지 못할 때 발생할 수 있는 문제를 해결하여 더욱 효율적인 샘플링을 가능케 합니다. 정규화 흐름은 복잡한 목표 밀도(target density)를 간단한 잠재 밀도(latent density)로 변환하는 가역 맵을 학습하는 강력한 생성 모델입니다.

- **Technical Details**: 우리는 정규화 흐름의 잠재 공간에서 실패 사건(failure events)의 탐색을 위해 중요 추출을 수행하는 방법론을 제안합니다. 이 접근법에서는 잠재 공간에서 제안 분포를 조정하여 더 효율적으로 도출된 추출 방식이 이루어집니다. 결과적으로, 우리는 시뮬레이션 결과의 분포가 중요 추출의 우선 분포(IS prior)와 잘 일치하도록 맵핑되어 초기 탐색에서 균형을 이룹니다.

- **Performance Highlights**: 우리가 제안한 방법은 자율 레이싱 및 항공기 지상 충돌 회피 시뮬레이션과 같은 로봇 공학 응용 프로그램에서 성능 향상을 확인했습니다. 잠재 공간에서 수행된 중요 추출은 목표 공간에서 수행된 방법보다 샘플 효율성과 실패 사건의 커버리지에서 향상된 결과를 나타냅니다. 이는 안전성 분석에 있어서도 높은 유용성이 있음을 보여주고 있습니다.



### Over-the-Air Fair Federated Learning via Multi-Objective Optimization (https://arxiv.org/abs/2501.03392)
- **What's New**: 본 논문은 연합 학습(federated learning, FL)에서 클라이언트의 지역 데이터셋 분포 간의 이질성 문제를 해결하기 위해 새로운 알고리즘인 OTA-FFL을 제안합니다. OTA-FFL은 무선 통신을 기반으로 공정한 연합 모델을 훈련시키고, 다목적 최소화(multi-objective minimization) 문제로 FL을 구형화하여 공정성을 보장합니다. 또한, Chebyshev 접근 방식을 수정하여 적응형 가중치 계수를 계산하는 방법을 도입합니다.

- **Technical Details**: OTA-FFL 알고리즘은 각 통신 라운드에서 클라이언트가 지역 손실 함수를 전송하고, 이를 바탕으로 PS(Parameter Server)가 경량화된 Chebyshev 접근 방식을 통해 적응형 가중치 계수를 산출합니다. 이러한 가중치 계수는 클라이언트의 데이터를 효율적으로 집계하는데 사용되며, 무선 전송 중 발생하는 왜곡을 최소화하는 최적의 전송 스칼라(transmit scalars)와 노이즈 제거 스칼라(de-noising scalar)가 유도되었습니다. 이 과정은 무선 채널의 제한을 극복하는데 기여합니다.

- **Performance Highlights**: 실험 결과, OTA-FFL은 기존 방법들에 비해 공정성과 강력한 성능을 달성하며, 이질적인 데이터 환경에서 우수한 결과를 나타냅니다. 평균 정확도가 높은 클라이언트들이라도, 데이터 분포가 다른 클라이언트에 비해 불이익을 받지 않도록 보장하는 특성을 보여주었습니다. 이는 특히 의료 데이터와 같이 민감한 정보를 다루는 분야에서 중요한 의미를 갖습니다.



### Existential Crisis: A Social Robot's Reason for Being (https://arxiv.org/abs/2501.03376)
- **What's New**: 이 연구는 로봇의 성격이 사용자 경험에 미치는 영향을 조사하는 것을 목표로 하고 있습니다. 최근의 LLMs와 음성 인식 기술을 활용하여, 성격 중심 로봇과 업무 중심의 중립적인 로봇 간의 상호작용을 비교했습니다. 성격이 로봇과의 상호작용에서 중요한 역할을 하며, 이는 사용자의 로봇에 대한 인식 및 정서에 영향을 미치는 요소로 작용합니다.

- **Technical Details**: 본 연구는 네덜란드 암스테르담의 자유대학에서 '사회적 지능 로봇공학' 수업을 수강하는 12명의 학생을 대상으로 실시되었습니다. 연구는 두 가지 조건으로 진행되었고, 성격이 있는 로봇과 성격이 없는 로봇이 사용되었습니다. 실험에서는 NAO 로봇이 의료 질문을 사용자에게 묻고, 각 조건에 따라 다른 반응을 보였습니다. 연구에서 사용된 설문지는 사용자 경험을 평가하기 위해 정량적 및 정성적 측정을 포함한 설계로 마련되었습니다.

- **Performance Highlights**: 결과는 로봇에 감정을 입력하는 것이 사용자 경험 향상에 긍정적인 영향을 미친다는 가설을 지지합니다. 특히, 성격 있는 로봇이 제공한 유머는 참여자들에게 더 긍정적인 반응을 끌어냈습니다. 인상적인 점은 로봇의 성격이 사용자와의 상호작용의 질을 향상시키며, 이는 사회적 지능 로봇이 의료 분야에서 효과적으로 활용될 수 있는 가능성을 제시합니다.



### License Plate Images Generation with Diffusion Models (https://arxiv.org/abs/2501.03374)
- **What's New**: 본 연구는 라이센스 플레이트 인식을 위한 합성 데이터 생성의 필요성을 강조하며, 특히 차별화된 접근 방식을 제안합니다. 기존의 GAN 기반 접근법 대신 최신 이미징 기술인 diffusion 모델을 활용하여 사실적인 라이센스 플레이트 이미지를 생성합니다. 이를 통해 제너레이티브 모델의 성능을 실험적으로 검증하고, 생성된 데이터의 특성을 깊이 있게 분석하였습니다. 연구 결과, 합성 데이터가 LPR 작업에 매우 유용함을 실증적으로 확인하였습니다.

- **Technical Details**: 연구진은 우크라이나의 라이센스 플레이트 데이터셋을 사용하여 Denoising Diffusion Probabilistic Model (DDPM)을 훈련했습니다. 이 과정에서 1,000개의 합성 이미지를 생성하고, 이를 수작업으로 성공 및 실패 사례로 분류한 뒤 각 이미지에 대한 정보를 세부 분석하였습니다. 캐릭터 분포 분석과 같은 추가적인 작업을 통해 LPR 모델의 성능을 향상시킬 수 있는 기회를 모색했습니다. 최종적으로 10,000장의 합성 라이센스 플레이트 이미지를 공개하였습니다.

- **Performance Highlights**: 생성된 합성 데이터셋은 실제 데이터셋과 비교하여 LPR 정확도를 3% 향상시키는 결과를 나타냈습니다. 초기에는 실제 데이터와 합성 데이터 간의 성능 격차가 존재했으나, 합성 데이터의 사용으로 데이터 세트를 확장함으로써 성능이 개선되었습니다. 이와 같은 결과는 합성 데이터가 LPR 과제의 효과적인 솔루션이 될 수 있음을 시사합니다.



### Advanced Machine Learning Techniques for Social Support Detection on Social Media (https://arxiv.org/abs/2501.03370)
- **What's New**: 이 연구는 소셜 미디어에서 온라인 사회적 지원(online social support)의 영향을 이해하려는 노력을 담고 있습니다. 지원 내용에 대한 이진(binary) 및 다중 클래스(multiclass) 분류를 포함한 데이터 세트를 사용하여 사회적 지원을 세 가지 작업으로 나누어 분석합니다. 이 작업들은 지원과 비지원의 구별, 개인 또는 그룹을 대상으로 한 지원 여부, 그리고 특정 사회적 지원의 유형을 분류하는 것입니다.

- **Technical Details**: 데이터 불균형 문제를 해결하기 위해 K-means clustering을 사용하여 데이터 세트를 균형 있게 조정하였으며, 원래의 불균형 데이터와 결과를 비교했습니다. 첨단 기계 학습 기법인 transformers와 zero-shot learning 접근법을 적용하여 다양한 맥락에서 사회적 지원 수준을 예측합니다. 연구에서 사용한 baseline 모델과의 비교를 통해 transformer 기반 방법이 우수한 성능을 보였음을 알 수 있습니다.

- **Performance Highlights**: 연구 결과, 두 번째 작업에서는 매크로 F1 점수가 0.4% 향상되었고, 세 번째 작업에서는 0.7% 향상되었습니다. 이러한 결과는 psycholinguistic 및 unigram 기반 TF-IDF 값을 활용한 기존 작업에 비해 향상된 성과로 평가됩니다.



### FTA-FTL: A Fine-Tuned Aggregation Federated Transfer Learning Scheme for Lithology Microscopic Image Classification (https://arxiv.org/abs/2501.03349)
- **What's New**: 본 연구는 석유 저장소의 특성을 파악하기 위해 리소지(특정 암석의 종류) 마이크로 이미지 분류를 대상으로 하며, 이를 위한 혁신적인 Federated Learning (FL) 접근방식을 제안합니다. 특히 데이터의 민감성 문제를 해결하면서도 높은 정확도의 중앙 모델을 훈련할 수 있는 방안을 모색하였습니다. 또한, Transfer Learning과 데이터 증강 기법을 활용하여 소규모 데이터셋에서 시작하는 두 단계의 연구를 진행하였습니다.

- **Technical Details**: 연구의 첫 번째 단계에서는 Transfer Learning을 통해 소규모 리소지 마이크로 이미지 분류를 수행하며 여러 가지 사전 학습된 Deep Learning 모델 아키텍처를 비교하였습니다. 두 번째 단계에서는 분산된 엣지 서버에서 민감한 데이터를 전송하지 않고도 효과적인 모델을 학습할 수 있는 Federated Transfer Learning (FTL) 구성을 하였으며, Fine-Tuned Aggregation 전략(FTA-FTL)을 제안하였습니다. 이러한 방법은 정확도, f1 점수, 정밀도, 특이도, 민감도(재현율), 혼동 행렬과 같은 여러 메트릭스를 바탕으로 평가되었습니다.

- **Performance Highlights**: 제안된 FTA-FTL 알고리즘은 중앙 집중형 구현 방식에서 달성한 결과와 거의 동일한 결과를 얻을 수 있는 것으로 나타났습니다. 실험 결과는 제안된 접근 방식의 효율성을 확인하며, 복잡한 실험적 연구에도 불구하고 좋은 일관성을 보였습니다. 이로 인해 석유 개발 분야에서의 리소지 분류 작업에 있어 유망한 방향성을 제시합니다.



### Analyzing Bias in Swiss Federal Supreme Court Judgments Using Facebook's Holistic Bias Dataset: Implications for Language Model Training (https://arxiv.org/abs/2501.03324)
- **What's New**: 이번 연구는 스위스 판례 예측 데이터셋(SJP Dataset)에서 포함된 편향을 분석하여, 법적 결정 과정에서의 공정성을 보장하고자 하였다. 자연어 처리(NLP)에 있어서 훈련 데이터의 편향이 어떻게 판단 예측의 정확성에 영향을 미치는지를 조사함으로써, NLP 모델의 공정한 결정이 이루어질 수 있는 기반을 마련하는 목표다. 'Holistic Bias dataset'에서 제공하는 사회적 편향 묘사를 활용하여 데이터셋 내 데이터의 영향력을 조사한다.

- **Technical Details**: 데이터셋의 편향 분석은 'dispreferred'라는 레이블이 붙은 특성들을 이용하여 이루어진다. 새로운 버전인 Holistic Bias Dataset 1.1에서는 769개의 독특한 묘사가 포함되어 있으며, 이 중 70개가 'dispreferred'로 레이블이 붙어 있다. 텍스트 내용의 손실 없이 토큰 수의 한계를 관리하기 위해 LexRank Summarizer와 같은 최적화된 방법론을 사용하였다.

- **Performance Highlights**: 본 연구는 훈련 데이터의 불균형 문제를 해결하기 위해 클래스 가중치를 조정하며, 세 가지 서로 다른 설정으로 모델을 미세 조정하였다. SJP 데이터셋의 요구 사항을 충분히 충족하면서 NLP 모델의 성능 향상을 추구하였다. 결과적으로 교수 및 법적 맥락 속에서의 공정성 기반을 강화하며, 향후 연구 및 실제 적용 가능성을 제시하고자 한다.



### Rethinking Byzantine Robustness in Federated Recommendation from Sparse Aggregation Perspectiv (https://arxiv.org/abs/2501.03301)
Comments:
          accepted by AAAI 2025

- **What's New**: 이 논문은 Federated Recommendation (FR) 시스템에 대한 Byzantine 공격의 복원력을 최초로 조사합니다. 특히, 기존의 Dense Aggregation이 아닌 Sparse Aggregation의 관점에서 이 문제를 해결하려고 하고 있습니다. 이를 통해 Sparse Aggregation의 특성과 그것이 FR의 보안에 미치는 영향을 분석하였습니다.

- **Technical Details**: FR 시스템에서는 각 아이템의 임베딩이 부분 클라이언트에 의해 업데이트되는 Sparsity가 적용됩니다. 이 논문은 아이템 한 개에 대한 집계를 최소 실행 단위로 정의하여, Sparse Aggregation에서의 Byzantine Robustness를 재정의합니다. 이를 바탕으로 공격 전략을 설계하고 각각의 공격력이 다를 수 있음을 강조하였습니다.

- **Performance Highlights**: 제안된 Spattack 전략은 극소수의 악의적인 클라이언트만으로도 FR 모델의 수렴을 방해하고 방어를 무너뜨릴 수 있다는 것을 보여줍니다. 실험 결과, 기존 FR 모델은 악성 클라이언트의 영향을 쉽게 받지 않도록 개선될 필요가 있음을 확인하였습니다.



### A Soft Sensor Method with Uncertainty-Awareness and Self-Explanation Based on Large Language Models Enhanced by Domain Knowledge Retrieva (https://arxiv.org/abs/2501.03295)
- **What's New**: 이 논문은 데이터 기반 소프트 센서의 모델링에서 감독 학습(supervised learning)을 대신하여 In-Context Learning (ICL) 패러다임을 제안합니다. 기존의 방법들이 직면했던 높은 개발 비용, 낮은 강인성, 훈련 불안정성 및 해석 가능성 부족을 해결하고자 합니다. 이를 위해 Few-shot Uncertainty-aware and self-Explaining Soft Sensor (LLM-FUESS)라는 새로운 프레임워크를 제안합니다.

- **Technical Details**: LLM-ZAVS(Zero-shot Auxiliary Variable Selector)는 산업 지식 벡터 저장소(Industrial Knowledge Vector Storage)에서 정보를 가져와 LLM의 도메인 특정 지식을 강화하여 제로샷(auxiliary variable selection)을 가능하게 합니다. LLM-UFSS(Uncertainty-aware Few-shot Soft Sensor)에서는 구조화된 데이터에 대한 텍스트 기반 컨텍스트 데모를 활용하여 ICL을 실행하고 성능 향상을 위한 컨텍스트 샘플 검색 보강 전략을 제안합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 예측 성능에서 최첨단(state-of-the-art) 결과를 달성하며, 전통적인 방법에서 발견된 훈련 불안정성을 효과적으로 완화하고 강력한 강인성과 유연성을 보여주었습니다. 이 연구는 LLM을 활용한 소프트 센서를 구축한 최초의 작업으로, 신뢰할 수 있는 소프트 센서를 구축하기 위한 자가 설명(self-explanation) 및 불확실성 정량화 방법을 제안합니다.



### Multi-Modal One-Shot Federated Ensemble Learning for Medical Data with Vision Large Language Mod (https://arxiv.org/abs/2501.03292)
- **What's New**: 본 논문에서는 FedMME라는 새로운 일회성(one-shot) 다중모드(multi-modal) 연합 학습(federated learning) 프레임워크를 도입하여 의료 이미지 분석을 개선하고자 합니다. 이 시스템은 의료 데이터의 시각적 및 텍스트적 특성을 통합하며, 비전 대형 언어 모델(vision large language models)을 활용하여 의료 이미지로부터 더 정확한 보고서를 생성합니다. 이러한 접근 방식은 모델 훈련 중에 낮은 통신 비용을 유지하면서 예측 정확도와 견고성을 크게 향상시킵니다.

- **Technical Details**: 이 연구는 여러 데이터 분포를 가진 네 가지 데이터 세트에서 실험을 수행하여 기존의 이전 일회성 연합 학습(unimodal federated learning) 방법들보다 분명히 뛰어난 성능을 기록하였습니다. 특히, 먹어보는 굴착 데이터들이 존재하는 RSNA 데이터셋의 경우, Dirichlet 분포와 함께 ($\alpha$ = 0.3) 적용했을 때 17.5% 이상의 정확도를 초과하는 결과를 나타냈습니다. FedMME는 비전 대형 언어 모델을 통해 의료 이미지에서 텍스트를 생성하고, BERT 모델을 활용하여 텍스트적 특징을 추출하여 시각적 특징과 통합합니다.

- **Performance Highlights**: FedMME는 의료 분야에서 연합 학습의 최신 전환점을 제공하며, 관련 데이터의 보호와 분석의 효율성을 높이는데 기여합니다. 이 프레임워크는 특히 자원이 제한된 의료 환경에서 효과적으로 작동하며, 데이터 전송 비용을 줄이고 다양한 모드의 정보를 통합하는 데 유리합니다. 최종적으로, 이 연구는 일회성 연합 학습 구조에서 다중모드 데이터의 포함을 통해 진단의 정확성과 신뢰성을 향상시킬 수 있음을 보여주고 있습니다.



### A Decision-Based Heterogenous Graph Attention Network for Multi-Class Fake News Detection (https://arxiv.org/abs/2501.03290)
- **What's New**: 본 논문에서는 기존의 GNN (Graph Neural Network) 기반 방법론의 제한점을 극복할 수 있는 새로운 모델인 DHGAT (Decision-based Heterogeneous Graph Attention Network)를 소개합니다. DHGAT는 이종 그래프에서 뉴스 데이터를 모델링하여 각 노드가 최적의 이웃 유형을 동적으로 선택하여 가짜 뉴스 탐지의 정확성을 크게 향상시킵니다. 이 접근법은 정보의 전파를 각 노드별로 독립적으로 조정할 수 있는 능력을 제공합니다.

- **Technical Details**: DHGAT의 아키텍처는 결정 네트워크와 표현 네트워크의 두 가지 네트워크로 구성되어 있습니다. 결정 네트워크는 각 레이어에서 최적의 이웃 유형을 선택하고, 표현 네트워크는 선택된 타입을 기반으로 노드의 임베딩(embedding) 벡터를 업데이트합니다. Gumbel-Softmax 분포를 활용하여 각 노드가 연속적으로 이웃을 선택하는 방식은 모델의 성능을 개선하는 데 기여합니다.

- **Performance Highlights**: LIAR 데이터셋에 대한 실험을 통해 DHGAT의 성능을 검증하였으며, 기존 방법들보다 약 4% 향상된 정확도를 기록했습니다. 이 모델은 제한된 레이블 데이터 환경에서도 강인성을 보여주어 가짜 뉴스 탐지의 효율성을 높입니다. DHGAT는 다양한 관계를 담고 있는 이종 그래프 구조를 통해 가짜 뉴스 탐지의 새로운 가능성을 제공합니다.



### CodeVision: Detecting LLM-Generated Code Using 2D Token Probability Maps and Vision Models (https://arxiv.org/abs/2501.03288)
- **What's New**: 본 논문은 2D token 확률 맵을 활용한 새로운 LLM 생성 코드 탐지 방법을 제안합니다. 기존의 방법들은 적응성과 계산 효율성에서 한계를 보였지만, 이 연구에서 제안된 방법은 코드의 공간적 구조를 그대로 유지하면서 기계가 생성한 코드를 더욱 정확하게 탐지할 수 있습니다. 이는 교육 환경에서 LLM 활용 증가에 따른 학문적 무결성 문제를 해결하는 데 기여할 것입니다.

- **Technical Details**: 제안된 방법은 코드 스니펫을 log probability 매트릭스로 변환하고, Vision Transformers (ViT) 및 ResNet과 같은 비전 모델을 활용하여 공간적 특징을 캡쳐합니다. 코드의 확률 분포를 2D 이미지처럼 다루면서, 전통적인 시퀀스 모델들이 놓칠 수 있는 패턴을 인식하는 데 강점을 가집니다. 또한, 다양한 프로그래밍 언어에 대해 언어 특정 미세 조정 없이 적용 가능하다는 장점도 가지고 있습니다.

- **Performance Highlights**: 제안된 탐지 방법은 기존 전통 탐지기들과 비교하여 다수의 프로그래밍 언어에서 강건함을 보여주었습니다. 또한, 이 방법은 경량 비전 모델을 사용해 실시간 탐지 프레임워크를 제공하며, 계산 효율성을 높여 교육 및 전문 환경에서의 실용성을 개선합니다. 효율성과 정확성을 모두 갖춘 새로운 탐지 방법으로, LLM에 의해 생성된 코드를 더욱 효과적으로 식별할 수 있습니다.



### Revolutionizing Encrypted Traffic Classification with MH-Net: A Multi-View Heterogeneous Graph Mod (https://arxiv.org/abs/2501.03279)
Comments:
          Accepted by AAAI 2025. The code is available at this https URL. arXiv admin note: text overlap with arXiv:2402.07501

- **What's New**: MT-Net이라는 새로운 모델을 소개합니다. 이 모델은 다양한 트래픽 비트를 여러 유형의 트래픽 유닛으로 집계하여 다중 뷰 이질 트래픽 그래프(multi-view heterogeneous traffic graphs)를 구성함으로써, 트래픽 분류의 정밀도를 높입니다. 특히, 헤더-페이로드 관계(head-payload relationships)와 같은 다양한 바이트 상관관계를 고려하여 데이터의 이질성을 강화합니다.

- **Technical Details**: MH-Net은 입장별 상호 정보(point-wise mutual information; PMI)를 활용하여 다양한 트래픽 단위 시퀀스를 다중 뷰 트래픽 그래프로 변환합니다. 또한, 헤더-헤더, 헤더-페이로드 및 페이로드-페이로드 단위 상관관계를 도입하여 트래픽 그래프의 이질성을 모델링하며, 이를 위해 이질 그래프 신경망(heterogeneous graph neural network)을 사용하여 피처 추출을 진행합니다. 마지막으로, 차별적 학습(contrastive learning)을 다중 작업(multi-task) 방식으로 수행하여 트래픽 유닛 표현의 강인성을 강화합니다.

- **Performance Highlights**: MH-Net의 실험 결과는 ISCX 및 CIC-IoT 데이터셋을 통해 패킷 수준과 흐름 수준의 트래픽 분류 작업 모두에서 경쟁력 있는 성능을 보여 주었습니다. 이 모델은 현재까지 제안된 여러 최신 기법들 중에서도 최고의 성과를 달성했습니다. 특히, 서로 다른 정보의 다양성과 상보성 간의 균형을 분석함으로써, 트래픽 유닛의 완전성과 간섭 간의 잠재적 거래(trade-off)를 조명할 수 있었습니다.



### ComMer: a Framework for Compressing and Merging User Data for Personalization (https://arxiv.org/abs/2501.03276)
Comments:
          13 pages, 7 figures

- **What's New**: 이 논문에서는 ComMer - Compress and Merge라는 새로운 프레임워크를 소개하여 대형 언어 모델(LLMs)을 효율적으로 개인화하는 방법을 제안합니다. ComMer는 사용자의 문서를 압축하여 간결한 표현으로 변환한 다음, 이를 결합해 냉동된 LLM에 입력합니다. 이 접근 방식은 사용자 수가 많을 때 자원을 절약하고, 훈련 비용을 줄이며, 제한된 데이터로도 품질을 개선할 수 있는 장점을 가지고 있습니다.

- **Technical Details**: ComMer는 세 가지 단계로 이루어진 아키텍처를 통해 작동합니다. 첫 번째로 각 문서가 독립적으로 소프트 프롬프트로 압축되며, 이러한 압축은 학습 가능한 압축 임베딩과 LoRA 가중치를 활용하여 이루어집니다. 두 번째로 압축된 표현들이 평균 풀링(mean pooling)을 통해 집계되어 단일 소프트 프롬프트로 결합되고, 마지막으로 이 집계된 소프트 프롬프트가 냉동된 LLM에 연결되어 원하는 출력을 생성합니다.

- **Performance Highlights**: 실험 결과, ComMer는 제한된 토큰 예산에서 개인화된 기술 학습(task)에서 우수한 품질을 보여주었고, 문서의 수가 많아질수록 품질이 향상되는 경향을 보입니다. 그러나 지식 집중형 작업에서는 여러 문서의 정보를 모두 표현하는 데 제한이 있어 품질 저하가 발생하는 것으로 나타났습니다. 이는 다중 문서 압축을 통한 개인화에서의 무역 오프와 잠재적 최적화 방향에 대한 통찰을 제공합니다.



### Strategic Fusion Optimizes Transformer Compression (https://arxiv.org/abs/2501.03273)
Comments:
          15 pages, 1 table, 8 figures; will be submitted to ICML 2025; codes will be made public after acceptance

- **What's New**: 본 연구는 transformer 모델 압축을 위한 층 가지치기(pruning) 방법론을 체계적으로 탐구했습니다. 9개의 다양한 데이터셋에서 14개의 가지치기 전략을 평가하며, 특히 활성화(activation), 상호 정보(mutual information), 기울기(gradient), 가중치(weights), 주의(attention)에서 얻은 신호를 기반으로 한 12가지 전략을 분석했습니다. 또한, 단일 신호 전략의 한계를 극복하기 위해 선형 회귀(linear regression)와 랜덤 포레스트(random forest)라는 두 가지 융합 전략을 도입하여 보다 정보에 기반한 가지치기 결정을 내리는 방법을 제시합니다.

- **Technical Details**: 지금까지의 연구는 모델 압축을 위한 다양한 접근 방식을 탐색해왔으며, 여기에는 중요하지 않은 가중치를 가지치기하는 것, 매개변수를 양자화하는 것, 그리고 지식 증류(knowledge distillation)가 포함됩니다. 본 연구에서는 BERT 모델을 사용하여 활성화 기반, 기울기 기반 등 12개 개별 가지치기 전략을 분석하고, 각 전략의 선택을 위한 수학적 및 생물학적 직관을 제공했습니다. 실험을 통해 14개의 가지치기 전략이 9개의 데이터셋에서 평가되었으며, 융합 전략이 단일 지표 접근 방식보다 우수한 성능을 나타냈습니다.

- **Performance Highlights**: 랜덤 포레스트 기반 융합 전략은 9개 데이터셋 중 7개에서 최상의 성능을 달성했으며, 나머지 2개 데이터셋에서도 뛰어난 성과를 보였습니다. 지식 증류를 적용한 결과, 6개 데이터셋에서 원래의 정확도를 초과했고, 나머지 3개 데이터셋에서도 정확도 감소를 완화했습니다. 전체 데이터셋에 걸쳐 지식 증류 이후 정확도 대비 크기 비율이 평균 18.84배 증가하는 등의 성과를 보였습니다.



### Backdoor Token Unlearning: Exposing and Defending Backdoors in Pretrained Language Models (https://arxiv.org/abs/2501.03272)
Comments:
          AAAI 2025

- **What's New**: 이번 연구에서는 대규모 사전 학습 모델의 미세 조정(fine-tuning) 과정 중 발생하는 백도어 공격(backdoor attack)에 대한 효율적인 방어 방법을 제안합니다. 제안된 방법인 Backdoor Token Unlearning (BTU)은 훈련 단계에서 트리거 토큰(trigger token)을 사전에 탐지하고 중화하는 방식을 채택하여, 백도어 공격의 영향을 최소화합니다. 본 연구는 두 가지 주요 발견에 기반하여, 백도어 토큰 파라미터(backdoor token parameters)와 클린 토큰 파라미터(clean token parameters) 간의 독특한 차이를 활용합니다.

- **Technical Details**: BTU는 두 단계로 구성되어 있습니다. 첫 번째 단계에서는 워드 임베딩 레이어(word embedding layer)를 교육하여 백도어와 관련된 임베딩 파라미터를 식별하고, 두 번째 단계에서는 영향을 받은 백도어 임베딩 파라미터를 무해한 패딩 토큰(padding token) 임베딩으로 교체하여 백도어 정보를 제거합니다. 이 과정은 세 가지 데이터셋과 네 가지 유형의 백도어 공격을 통해 검증되었으며, BTU는 모델의 기본 작업 성능을 유지하면서 효과적인 방어를 제공합니다.

- **Performance Highlights**: BTU 방법은 백도어 공격의 성공률을 상당히 감소시킵니다. 세 가지 데이터셋과 네 가지 백도어 공격을 통해 수행된 실험 결과, BTU는 모델의 성능을 최소한으로 저하시키면서도 공격에 대한 저항력을 효과적으로 입증하였습니다. 본 연구의 코드와 자료는 제공된 링크에서 확인 가능하며, 향후 AI 모델의 안전성을 높이는 데 기여할 것으로 예상됩니다.



### A Semantically-Aware, Kernel-Enhanced, and Divergence-Rich Paradigm for Direct Preference Optimization (https://arxiv.org/abs/2501.03271)
Comments:
          -

- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)의 정렬 문제를 해결하기 위한 새로운 접근 방식인 DPO-Kernels를 제안합니다. 기존의 Direct Preference Optimization(DPO) 방법의 한계를 극복하기 위해 커널 방법을 통합하여 다양한 가치와 선호도에 맞게 조정할 수 있습니다. 주요 기여로는 폴리노미얼, RBF, Mahalanobis 및 스펙트럴 커널을 이용한 커널화 표현을 포함합니다.

- **Technical Details**: DPO-Kernels는 커널화된 표현을 통해 더 풍부한 변환을 제공하며, 이는 하이브리드 손실(hybrid loss) 기법에 의해 지원됩니다. 또한, 여러 다이버전스(혹은 발산) 대안, 즉 Jensen-Shannon, Hellinger, Renyi, Bhattacharyya, Wasserstein, f-divergences 등을 사용하여 안정성을 강化합니다. 데이터 기반 선택 메트릭을 통해 최적의 커널-다이버전스 쌍을 자동으로 선택할 수 있으며, 계층적 혼합 커널(Hierarchical Mixture of Kernels)를 통해 지역 정밀성과 전역 모델링을 제공합니다.

- **Performance Highlights**: 12개 데이터셋에 대한 평가에서 해당 모델은 사실성(factuality), 안전성(safety), 추론(reasoning), 지시 준수(instruction following) 측면에서 최첨단 성능을 보였습니다. Heavy-Tailed Self-Regularization에 기반하여 DPO-Kernels는 LLMs의 강력한 일반화 능력을 유지하며, 향후 정렬 연구에 필요한 포괄적인 자원을 제공합니다.



### Heterogeneous Graph Pre-training Based Model for Secure and Efficient Prediction of Default Risk Propagation among Bond Issuers (https://arxiv.org/abs/2501.03268)
- **What's New**: 이 논문에서는 채권 발행 기업의 기본 위험 예측을 위한 새로운 두 단계 모델을 제안합니다. 전통적인 방법들이 기업의 내부 데이터에만 의존하는 반면, 본 모델은 기업 간의 연결 정보를 활용하여 위험 식별 능력을 극대화합니다. 특히, Heterogeneous Graph 마스킹 오토인코더(HGMAE)를 도입하여 다양한 기업 정보를 전처리하는 단계에서 성능 향상을 꾀하고 있습니다.

- **Technical Details**: 도입된 두 단계 모델은 먼저 방대한 기업 지식 그래프(EKG)에서 사전 훈련을 진행한 후, 이를 바탕으로 채권 발행 기업의 위험 전파 확률을 예측하기 위한 분류기를 훈련하는 방식입니다. 이 모델은 각 기업의 특성과 관계형 데이터의 결합을 통해 채권 특성과 채무 불이행 위험을 효과적으로 추적할 수 있는 기회를 제공합니다. HGMAE는 그래프의 특정 기능들을 마스킹하여 이들을 복원하는 과정을 통해 효과적인 표현 학습을 목표로 하고 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 기존의 방법들보다 우수한 성능을 보였습니다. 채권 발행자의 기본 위험 예측에 있어 정확성과 효율성을 높이는데 중요한 역할을 하고 있습니다. 이 연구는 기업 간의 위험 정보 전파를 통한 예측 정확도를 높이며, 안전성과 개인정보 보호를 모두 고려한 혁신적인 접근 방식을 제시합니다.



### LLM Content Moderation and User Satisfaction: Evidence from Response Refusals in Chatbot Arena (https://arxiv.org/abs/2501.03266)
- **What's New**: 이번 연구는 사용자 만족도에 대한 콘텐츠 조정(content moderation)의 영향을 탐구합니다. 우리는 50,000개의 Chatbot Arena 응답 쌍을 분석하여 윤리적 문제에 따른 거부(refusals)를 다른 기술적 문제나 정보 부족으로 인한 거부와 구분하기 위해 특별히 조정된 RoBERTa 모델을 사용했습니다. 이 연구는 LLM(대규모 언어 모델)의 안전성과 윤리적 정렬에 논의되는 내용과 새로운 관점을 제시합니다.

- **Technical Details**: 우리는 수동으로 레이블이 붙은 데이터를 기반으로 훈련된 RoBERTa 모델을 통해 데이터 분석을 수행했습니다. 분석 결과, 콘텐츠 조정은 사용자에 대한 상당한 거부 패널티(refusal penalty)를 가져오며, 윤리에 기반한 거부는 사용자 선호 응답에 비해 약 4배 덜 발생하는 것으로 나타났습니다. 또한 민감한 프롬프트에 대한 거부가 더 낮은 윤리적 우려에 비해 더 높은 승률을 획득한다는 것을 발견했습니다.

- **Performance Highlights**: 안건과 구문이 중요한 역할을 하며, 프롬프트와 밀접하게 관련된 긴 응답이 더 좋은 성능을 보이는 것으로 나타났습니다. 연구에서는 또한 LLM-as-a-Judge 방법을 사용할 경우 거부 패널티가 눈에 띄게 낮아지는 경향이 있음을 발견하였습니다. 이러한 발견은 윤리적 안전 장치와 사용자 만족도 간의 균형을 맞추기 위한 세분화된 조정 전략의 필요성을 강조합니다.



### Optimizing Edge AI: A Comprehensive Survey on Data, Model, and System Strategies (https://arxiv.org/abs/2501.03265)
- **What's New**: 이번 논문에서는 데이터, 모델 및 시스템 최적화를 포함한 효율적이고 신뢰성 있는 엣지 AI 배포를 위한 최적화 삼위일체를 제안합니다. 이를 통해 엣지 AI가 다양한 스케너리오를 지원하기 위해 클라우드에서 훈련된 ML 모델을 다양한 엣지 장치로 효과적으로 전송할 수 있는 솔루션을 제공합니다. 이번 연구의 주요 목표는 자원 제약이 있는 장치에 적합한 모델을 개발하기 위한 종합적인 개요를 제공하는 것입니다.

- **Technical Details**: 데이터 최적화는 데이터 정리, 압축 및 증강을 통해 이루어지며, 모델 최적화는 가지치기(pruning), 양자화(quantization), 지식 증류(knowledge distillation) 등의 방법을 통해 수행됩니다. 또한, 시스템 최적화 기술로는 프레임워크 지원 및 하드웨어 가속이 포함됩니다. 이러한 최적화 기법들은 엣지 AI 작업 흐름을 가속화하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 연구는 엣지 AI의 잠재적 응용 프로그램을 분석하고, 기존 과제와 이를 해결하기 위한 전략을 제시합니다. 미래의 트렌드로는 더 지능적이고 유연한 엣지 AI의 발전이 예상되며, 새로운 기술들이 이러한 진화를 가능하게 할 것입니다. 연구자는 기본적으로 클라우드에 의존하지 않고 데이터 처리를 위한 효율적 모델 개발 및 배포를 위한 주요 아이디어를 제공합니다.



### Bridge the Inference Gaps of Neural Processes via Expectation Maximization (https://arxiv.org/abs/2501.03264)
Comments:
          ICLR2023

- **What's New**: 이 논문에서는 Neural Process(NP)의 성능 개선을 위한 새로운 접근법인 Self-normalized Importance weighted Neural Process(SI-NP)를 제안합니다. 기존 NP는 under-fitting 문제를 겪으며 실제 성능이 최적이 아닌 경우가 많았으며, 연구자들은 일반적으로 다양한 inductive biases를 도입하는 데 초점을 맞춰왔습니다. 그러나 이 논문은 NP의 성능 저하의 원인과 그것을 해결하기 위한 새로운 최적화 목표를 제시해 주목을 받고 있습니다.

- **Technical Details**: SI-NP는 expectation maximization 프레임워크 내에서 메타 데이터셋의 목표 log-likelihood의 surrogate objective를 최적화함으로써 개발되었습니다. 이를 통해 더 정확한 functional prior를 학습하고, 기존 방식들에 비해 개선된 성능을 보장할 수 있습니다. 본 연구는 NP의 inference suboptimality를 최적화 관점에서 분석하고, 다양한 최적화 목표의 통계적 특성을 연구함으로써 SI-NP의 기반이 됩니다.

- **Performance Highlights**: 실험 결과 SI-NP는 기존 NP 목표들에 비해 경쟁력 있는 성능을 보여줍니다. 또한 attention 모듈 등 구조적 inductive biases를 추가하여 SOTA 성능을 달성할 수 있음을 증명했습니다. 새로운 메커니즘을 통해 SI-NP는 메타 데이터셋의 log-likelihood를 개선할 수 있는 가능성을 제시합니다.



### Navigation Variable-based Multi-objective Particle Swarm Optimization for UAV Path Planning with Kinematic Constraints (https://arxiv.org/abs/2501.03261)
- **What's New**: 이번 연구에서는 UAV(무인 비행기)의 경로 계획 문제를 다루기 위해 NMOPSO(탐색 변수 기반 다목적 입자 군집 최적화)라는 새로운 알고리즘을 소개합니다. 이 알고리즘은 최적성과 안전 요구 사항을 포함한 목적 함수 집합을 정의하여 경로 계획을 최적화 문제로 모델링합니다. NMOPSO는 키네마틱 제약 조건을 포함하고 UAV의 기동 성능을 활용하는 새로운 경로 표현 방식을 특징으로 합니다.

- **Technical Details**: NMOPSO는 개인 경험과 집단 경험을 균형 있게 조화시켜 솔루션 공간 내의 잠재 지역을 찾아 최적의 경로를 탐색합니다. 이 과정에서 적응형 변이 메커니즘을 통해 군집 다양성을 향상시키고, Denavit-Hartenburg 표현에서 영감을 받은 공식을 도출해 경로 검색 및 평가에 대한 효율을 높입니다. 본 연구의 목표는 UAV의 자율 운영을 위한 최적의 경로를 생성하는 것입니다.

- **Performance Highlights**: NMOPSO는 다양한 기존 알고리즘과 비교한 결과, 입자 군집 최적화의 여러 변형 및 최신 다목적 메타휴리스틱 최적화 알고리즘보다 우수한 성능을 보였습니다. 또한 실제 UAV를 사용한 실험을 통해 NMOPSO의 실용성을 검증했습니다. 최적 경로 생성을 위한 시나리오를 바탕으로 NMOPSO의 성능 향상을 확인하였습니다.



### Toward Inclusive Educational AI: Auditing Frontier LLMs through a Multiplexity Lens (https://arxiv.org/abs/2501.03259)
- **What's New**: 본 논문은 교육적 문맥에서 대형 언어 모델(LLMs)의 문화적 편향을 평가하고 완화하기 위한 새로운 프레임워크를 제안합니다. 이 프레임워크는 복합성(multiplexity)이라는 관점에서 접근하여, 다양한 문화적 시각을 공존시키고 인식론을 다층적으로 통합하는 것을 목표로 합니다. 또한, LLM의 출력에서 종종 나타나는 문화적 양극화 현상을 분석하여 문제를 해결하기 위한 두 가지 전략을 제시합니다.

- **Technical Details**: 제안된 프레임워크는 상황적으로 구현된 복합 LLM(Contextually-Implemented Multiplex LLM)와 다중 에이전트 시스템(Multi-Agent System, MAS) 구현 복합 LLM(MAS-Implemented Multiplex LLM)으로 구성됩니다. 첫 번째 전략은 시스템 프롬프트에 복합 원리를 직접 내장해 모델의 기초적 레벨에서 출력을 다양화하는 것이며, 두 번째 전략은 각기 다른 문화적 관점을 대표하는 LLM 에이전트들이 협력하여 균형 잡힌 응답을 생성하는 것입니다. 이 연구는 LLM의 문화적 동등성을 높이기 위해 각 문화적 관점을 통합하는 방법의 중요성을 강조합니다.

- **Performance Highlights**: 문화적 편향을 완화하기 위한 전략들이 맥락 기반의 프롬프트에서 MAS 구현으로 발전함에 따라, 문화적 포용성은 크게 개선되었습니다. 연구 결과, 기본선에서는 3.25%였던 Perspectives Distribution Score (PDS)가 MAS-Implemented Multiplex LLM을 통해 98%로 증가하는 등, 문화적 다각화와 긍정적 정서 변화가 확인되었습니다. 이러한 결과는 다양한 문화적 관점을 고려한 LLM이 교육적 맥락에서 얼마나 중요한지를 보여줍니다.



### Breaking Through the Spike: Spike Window Decoding for Accelerated and Precise Automatic Speech Recognition (https://arxiv.org/abs/2501.03257)
Comments:
          Accepted by ICASSP 2025

- **What's New**: 본 논문에서는 CTC (Connectionist Temporal Classification) 출력의 spike 특성을 조사하고, 비공백 프레임과 근접한 이웃 프레임이 모델에 유익한 의미 정보를 가지고 있다는 가설을 제안합니다. 이를 바탕으로 Spike Window Decoding (SWD) 알고리즘을 제안하며, 이는 WFST (Weighted Finite-State Transducer)에서 디코딩 되는 프레임 수를 CTC 출력의 spike 프레임 수와 선형적으로 연관짓도록 설계되었습니다. 이를 통해 디코딩 속도가 크게 향상되었습니다.

- **Technical Details**: SWD 알고리즘은 CTC posteriori 확률 행렬에서 가장 높은 확률을 가진 프레임의 인덱스를 검색하는 것으로 시작합니다. 그 후, 비공백 스파이크 프레임과 일치하는 인덱스를 찾아 혁신적인 Spike Window 함수를 활용하여 이웃 프레임의 윈도우 시퀀스를 획득합니다. 이 과정에 게임 과정은 TLG (Transducer with Language Guidance) WFST 그래프 구조에서 det와 min 사이의 가중치 푸시 전략을 사용하는 것입니다.

- **Performance Highlights**: SWD 알고리즘은 AISHELL-1 및 대규모 In-House 데이터셋에서 각각 3.89% 및 2.09%의 문자 오류율(CER)을 달성하며, 이는 기존 SOTA 접근 방식을 초월합니다. 뿐만 아니라 SWD 알고리즘은 각각의 데이터셋에 대해 베이스라인 방법보다 1.76배 및 2.17배의 디코딩 속도 향상을 보여줍니다. 이러한 결과는 다양한 규모의 데이터셋에서 우수한 인식 성능과 향상된 디코딩 속도 간의 remarkable한 균형을 이룬 것입니다.



### AI-ANNE: (A) (N)eural (N)et for (E)xploration: Transferring Deep Learning Models onto Microcontrollers and Embedded Systems (https://arxiv.org/abs/2501.03256)
Comments:
          12 pages, 8 tables

- **What's New**: 이 논문은 Raspberry Pi Pico 및 Raspberry Pi Pico 2와 같은 자원 제약이 있는 임베디드 시스템에서 신경망을 통합하는 방법을 제시합니다. TinyML 접근 방식을 통해 이러한 마이크로컨트롤러에서 실시간, 저지연 및 에너지 효율적인 추론을 가능하게 하며 데이터 프라이버시를 유지할 수 있습니다. AI-ANNE(A Neural Net for Exploration)를 통해 TensorFlow 및 Keras와 같은 고성능 플랫폼에서 사전 훈련된 모델을 마이크로컨트롤러로 쉽게 이전할 수 있는 방법이 설명됩니다.

- **Technical Details**: 이 시스템은 MicroPython이라는 경량 프로그래밍 언어를 사용하여 신경망의 기본 구성 요소인 뉴런, 레이어, 밀도 및 활성화 함수를 구현합니다. Raspberry Pi Pico를 기반으로 하여 두 가지 다른 신경망을 소개하며, 이들은 데이터 분류의 예로 사용할 수 있습니다. AI-ANNE는 사전 훈련된 모델을 마이크로컨트롤러에서 실행할 수 있도록 하여 에너지 절약 및 데이터 프라이버시를 실현합니다.

- **Performance Highlights**: 이 논문에서 제안한 AI-ANNE는 교육 도구로서의 가치도 겸비하고 있으며, 사용자가 신경망의 작동 방식을 명확히 이해할 수 있도록 돕습니다. 또한, 다양한 뉴럴 네트워크의 학습 행동을 관찰할 수 있는 기능과 함께 이를 통해 교육적 응용 잠재력을 제공합니다. Thonny라는 프로그램을 통해 쉽게 코드 작성 및 테스트가 가능하며, 사용자에게 친숙한 환경을 제공합니다.



### Machine Learning and Deep Learning Techniques used in Cybersecurity and Digital Forensics: a Review (https://arxiv.org/abs/2501.03250)
- **What's New**: 이번 리뷰 논문에서는 사이버 보안과 디지털 포렌식 분야에서 기계 학습(Machine Learning)과 딥 러닝(Deep Learning)의 활용에 대한 개요를 제공합니다. 이 기술들이 사이버 위험을 식별하고 분석하는 방법을 소개하며, 이러한 접근법의 장점, 단점 및 가능성을 살펴봅니다. 논문은 또한 악성 소프트웨어를 분류하고 비정상적인 행동을 감지하여 사이버 공격을 예방하는 데 필요한 다양한 AI 기술을 다룹니다.

- **Technical Details**: 논문에서는 기계 학습과 딥 러닝 기법을 활용한 사이버 보안 및 디지털 포렌식에 관련된 주요 요소들을 설명합니다. 이를 비롯한 여러 AI 기법들은 시스템 침입 탐지 및 악성 코드 분류에 사용됩니다. 문서에서 제공하는 데이터와 예시는 이러한 기술들이 현재 사이버 보안 환경에서 어떻게 중요한 역할을 하는지를 보여줍니다.

- **Performance Highlights**: 사이버 보안과 디지털 포렌식의 발전 가능성을 논의하며, 추가 연구가 필요한 분야를 강조합니다. 또한 투명하고 확장 가능한 기계 학습 및 딥 러닝 솔루션을 개발하기 위한 제안도 포함되어 있습니다. 이는 계속해서 진화하는 사이버 보안 환경에서 특히 중요합니다.



### Neural networks consisting of DNA (https://arxiv.org/abs/2501.03235)
Comments:
          Book chapter, to appear in: Artificial Intelligence and Intelligent Matter, Springer, Cham

- **What's New**: 최근의 연구들은 전통적인 전자 회로 기반 구현 대신에 소프트 및 생물학적 물질을 기반으로 하는 신경망에 관심을 기울이고 있습니다. 특히 DNA는 정보를 저장하는 자연적인 능력 덕분에 이러한 접근의 유망한 기초로 제안되고 있습니다. 저자는 비전문가를 위한 DNA 신경망의 기본 개념을 소개하며, 학습 방법이 물리 시스템에 구현되는 가능성을 다루고 있습니다.

- **Technical Details**: DNA 신경망의 기본 구조는 분류 작업에서 경쟁을 통해 결과를 얻는 'winner-take-all' 네트워크와 DNA 게이트를 포함합니다. DNA는 생성 및 분석하기간단한 생물학적 맥락에서 정보를 저장하고 처리하는 능력을 가진 다중 모노머(단량체)로 이루어진 중합체입니다. 생명 정보는 전달 과정을 통해 RNA로 복사되며, 최종적으로 단백질이 합성됩니다.

- **Performance Highlights**: DNA 컴퓨팅은 높은 병렬 처리 성능과 효율적인 데이터 저장을 자랑하지만, 여전히 많은 오류 가능성과 자원 접근성이 문제점으로 지적됩니다. Adleman의 연구처럼 DNA 기반 신경망 구현은 필요한 입력 신호를 DNA로 인코딩하여 수행될 수 있으며, 이는 전통적인 컴퓨팅 접근 방식과의 차별적인 성격을 강조합니다. 이러한 장점에도 불구하고 생화학적 처리의 오류는 DNA 컴퓨터가 직면한 주요 도전 과제임을 알리는 부분입니다.



### Accuracy Can Lie: On the Impact of Surrogate Model in Configuration Tuning (https://arxiv.org/abs/2501.01876)
Comments:
          This paper has been accepted by TSE

- **What's New**: 본 연구는 구성 조정(configuration tuning) 과정에서 흔히 믿어지는 '정확도가 전부'라는 신념에 의문을 제기합니다. 13개월 동안 24시간 7일 진행된 대규모 실험을 통해, 높은 정확도가 항상 조정 결과를 향상시키지 않음을 발견했습니다. 정확도가 높아도 58%의 경우 조정 결과에 개선이 없거나, 24%의 경우 오히려 품질이 저하될 수 있음을 확인했습니다.

- **Technical Details**: 본 연구는 총 10개의 모델, 17개의 조정기, 29개의 시스템을 포함하여 13,612개의 사례를 분석했습니다. 다양한 모델의 성능을 평가하기 위해 4개의 일반적으로 사용되는 지표(metrics)를 기반으로 한 연구입니다. 실험 결과, 많은 조정자들이 선택하는 모델이 최적이 아닐 뿐만 아니라, 조정 품질을 개선하기 위한 정확도 변화의 요구 비율은 모델의 정확도 범위에 따라 달라짐을 밝혔습니다.

- **Performance Highlights**: 모델 기반 조정에서 높은 정확도가 항상 유리한 결과로 이어지지 않는다는 점을 강조하였습니다. 정확도의 개선이 조정의 질에 미치는 영향에 대해 깊이 있는 논의와 함께 여러 교훈과 향후 기회에 대한 통찰력을 제공합니다. 이 연구는 '정확도가 전부'라는 신념에서 한 걸음 물러날 필요성을 알리며, 조정 커뮤니티에 중요한 메시지를 전달합니다.



### Turn-based Multi-Agent Reinforcement Learning Model Checking (https://arxiv.org/abs/2501.03187)
- **What's New**: 이 논문에서는 확률적 멀티플레이어 게임에서 턴 기반 다중 에이전트 강화 학습(TMARL) 에이전트의 복잡한 요구 사항 준수성을 검증하는 새로운 접근 방법을 제안합니다. 제안된 방법은 기존 검증 방식의 한계를 극복하여 TMARL 에이전트 및 다수의 에이전트를 가진 대규모 게임에도 효과적으로 확장됩니다. TMARL과 검증 기법인 모델 체크를 긴밀하게 결합하여 효과성과 확장성을 입증하였습니다.

- **Technical Details**: 논문에서는 TMARL 에이전트의 검증을 위해 엄격한 모델 체크를 사용합니다. 모델 체크는 수학적 모델을 활용하여 시스템의 올바름을 검증하는 형식적 검증 기법으로, 시스템을 마르코프 결정 프로세스(MDP)로 모델링하여 에이전트들의 요구 사항을 만족하는지 확인합니다. 우리의 방법은 확률 계산 트리 논리(PCTL)를 통해 표현할 수 있는 다양한 속성을 지원합니다.

- **Performance Highlights**: 실험 결과, 우리의 방법은 TMARL 에이전트를 검증하는 데 적합하며, 단순한 모놀리식 모델 체크 방식보다 더 나은 성능을 보였습니다. 다양한 TMARL 벤치마크에서 평가하였으며, 제안한 접근 방식이 실제 환경에서 효과적으로 작동함을 입증하였습니다. 이는 게임 개발자가 에이전트의 행동을 의도한 대로 보장할 수 있도록 돕는 중요한 기여를 합니다.



### Large language models for artificial general intelligence (AGI): A survey of foundational principles and approaches (https://arxiv.org/abs/2501.03151)
- **What's New**: 이 논문에서는 대규모 사전 훈련된 기반 모델(Foundation Models, PFMs) 기반의 생성 인공지능(Generative AI) 시스템이 복잡하고 비트리비얼한 문제를 해결할 수 있는 능력을 보여주고 있다고 언급합니다. 특히, 다중 모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)은 방대한 데이터 소스에서 학습하여, 세상의 풍부하고 미묘한 표현을 가능하게 하여 인간과 다른 에이전트와 협력하여 문제를 해결할 수 있게 합니다. 그러나 이러한 모델들은 상태에 따라 여전히 제한된 인지 능력을 보이고 있으며, AGI(Artificial General Intelligence) 도달을 위해서는 인체 인지와 밀접하게 관련된 몇 가지 근본적인 문제를 해결해야 한다고 강조합니다.

- **Technical Details**: 논문에서는 인지의 기본 원칙인 신체화(embodiment), 기호 기반(symbol grounding), 인과관계(causality), 기억(memory)과 같은 개념들이 어떻게 인공지능 모델에서 구현될 수 있는지 논의합니다. 이러한 원칙들은 LLM이 인간과 유사한 인지 속성을 갖도록 도와주며, 이는 더 구체적이고 의미 있는 지식 및 지능의 실현을 위한 것이다. 특히, 신체화 원칙은 기계가 환경과 상호작용하며 특정 작업을 수행하는 데 있어 중요하다고 제안합니다.

- **Performance Highlights**: AGI의 실현 가능성을 높이기 위한 노력들이 이 연구의 중심에 있으며, AGI는 복잡한 인지 작업을 수행할 수 있는 능력을 요구합니다. 모델은 여러 도메인에서 다른 맥락에서도 유효하게 작동할 수 있는 일반 지식을 협력적으로 사용해야 하며, 이는 유연하고 적응 가능한 형태의 지능을 나타냅니다. 또한 AGI는 유기적인 형태로 기능성을 통해 스스로 학습하고 문제를 해결하는 능력을 갖추어야 하며, 이러한 접근 방식은 생물학적 지능의 본질과 일치합니다.



### Co-Activation Graph Analysis of Safety-Verified and Explainable Deep Reinforcement Learning Policies (https://arxiv.org/abs/2501.03142)
- **What's New**: 이번 논문에서는 Deep Reinforcement Learning (RL) 정책의 안전성을 향상시키기 위해 RL 정책 모델 검증과 co-activation graph 분석을 결합한 새로운 접근법을 제안합니다. 이 방법은 RL 정책이 안전한 의사결정을 내릴 수 있도록 내부 작동 방식을 해석하는 데 도움을 줍니다. 배경 이론으로는 probabilistic computation tree logic (PCTL)이 활용되며, 이를 통해 특정 안전성을 명시할 수 있습니다.

- **Technical Details**: 본 연구의 기술적 세부사항은 RL 환경에서 도달 가능한 상태의 무라벨 데이터셋을 생성하고, 이 데이터를 통해 사용자 지정 안전 속성을 검증하는 과정을 포함합니다. 이를 위해 상태 선택 시 모델 기반 RL 환경과 상호작용하는 공식 모델을 구축하여 검증하는 방식이 사용됩니다. 마지막으로, co-activation graph 분석을 통해 트레인된 NN 정책의 뉴런 활성화와 연관성을 분석하여 내부 작동 방식을 파악합니다.

- **Performance Highlights**: 실험 결과, RL co-activation graph 분석은 NN의 안전성 응용을 위한 해석 도구로서의 유용성을 보여줍니다. 뉴런의 중요성과 특징 순위를 파악하고, 네트워크 내의 밀집 연결 뉴런 클러스터(기능 모듈)를 식별함으로써 안전한 의사결정에 기여하는 다양한 요소를 밝혀냅니다. 이는 explainable AI의 발전에 기여하며, 모델의 비판적 영역에서의 행동을 이해하는 데 도움을 줍니다.



### Analyzing Fine-tuning Representation Shift for Multimodal LLMs Steering alignmen (https://arxiv.org/abs/2501.03012)
Comments:
          The first three authors contributed equally

- **What's New**: 이 논문은 다중모달 대형 언어 모델(Multimodal LLMs, MLLMs)의 내부 기제를 이해하고 설명하는 데 초점을 맞추고 있습니다. 기존 연구들이 모델의 최종 상태만을 분석했던 것과 달리, 저자들은 훈련 과정에서 발생하는 은닉 상태 표현의 진화를 체계적으로 분석합니다. 이를 통해 모델의 내부 구조가 새로운 다중모달 작업에 맞게 어떻게 전문화되는지를 밝혀냅니다.

- **Technical Details**: 저자들은 개념 기반 접근 방식을 사용하여 은닉 상태를 해석 가능한 시각적 및 텍스트 개념에 매핑합니다. 이 방법론을 통해 훈련 과정에서 인코딩된 개념의 변화 추적이 가능하며, shift vectors를 통해 원래 모델의 개념을 보완하거나 재구성할 수 있음을 보여줍니다. 이를 통해 MLLMs의 행동을 조정할 수 있는 실용적인 영향을 탐구합니다.

- **Performance Highlights**: 연구 결과, 학습된 개념들이 특정 훈련 작업에 맞춰 어떻게 조정되는지를 발견했습니다. 저자들은 원래 모델의 개념을 기준으로 shift vectors를 사용하여 많은 fine-tuned 개념을 재구성할 수 있음을 보여주며, MLLMs의 응답을 추가 훈련 없이 수정할 수 있는 가능성을 입증합니다. 이러한 방식은 MLLMs의 조정 가능성을 탐구하는 첫 번째 연구로, 모델 출력의 방향성을 조절하는 데 획기적인 통찰을 제공합니다.



### CALM: Curiosity-Driven Auditing for Large Language Models (https://arxiv.org/abs/2501.02997)
Comments:
          Accepted by AAAI 2025 AI Alignment Track

- **What's New**: 이 연구에서는 블랙박스(black-box) 대규모 언어 모델(LLM)의 감사(auditing) 방법을 개발했습니다. 특히, 모델의 파라미터에 접근하지 않고 서비스로 제공되는 LLM의 입력-출력 쌍을 자동으로 발견하는 최적화를 목표로 하고 있습니다. 이를 통해 비윤리적이거나 위험한 행위를 보이는 입력을 찾는 데 중점을 두고 있습니다.

- **Technical Details**: 제안된 방법인 Curiosity-Driven Auditing for Large Language Models (CALM)은 강화 학습(reinforcement learning)을 활용하여 감사 LLM을 조정합니다. CALM은 다양한 감사 프롬프트를 생성하여 모델이 특정한 비하적 또는 민감한 반응을 유도할 수 있도록 설정되어 있습니다. 이 접근법은 입력 공간의 희소성을 활용하여 숨겨진 행동을 발견하는 데 집중합니다.

- **Performance Highlights**: CALM은 여러 LLM에서의 다양한 실험을 통해 문제가 있는 행동을 효율적으로 식별하는 성과를 보였습니다. 이 과정에서 상대적으로 작은 모델인 GPT-2의 미세 조정만으로도 Llama-3-8B와 같은 대형 모델의 부정적 행동을 발견할 수 있음을 보여줍니다. 이는 LLM의 잠재적 위험을 강조하며, 호기심 기반의 RL 접근법이 감사를 위해 중요함을 입증합니다.



### Fairness Through Matching (https://arxiv.org/abs/2501.02793)
Comments:
          Published in TMLR

- **What's New**: 이 논문은 기존의 그룹 공정성 측정보다 개선된 새로운 공정성 척도인 Matched Demographic Parity (MDP)를 제안합니다. MDP는 보호 그룹 간 예측의 평균 격차를 정량화하여 그룹 공정 모델의 행동을 보다 명확히 파악할 수 있게 돕습니다. 또한 Fairness Through Matching (FTM)이라는 새로운 알고리즘을 개발하여, 선택된 운반 맵에 기반한 MDP 제약 조건 아래에서 그룹 공정 모델을 학습합니다.

- **Technical Details**: 이 연구에서는 그룹 공정성이 모델의 예측 결과에 미치는 영향을 명확히 하기 위해, 매칭을 통한 고유한 운반 맵(transporation map) 개념을 도입합니다. MDP는 이 운반 맵을 통해 두 보호되는 개인 간의 예측 격차를 측정하고, 모든 운반 맵이 MDP를 통해 그룹 공정 모델 학습에 사용될 수 있음을 증명합니다. 또한, 이 논문은 최적의 공정성-예측 성능 간의 trade-off가 아니라, 특정 바람직한 특성을 갖춘 그룹 공정 모델을 만드는 데 중점을 둡니다.

- **Performance Highlights**: 실험 결과, FTM은 기존의 그룹 공정 알고리즘보다 바람직한 특성을 갖춘 그룹 공정 모델을 성공적으로 학습했습니다. 제안된 두 가지 운반 맵은 특정 목적을 달성하는 데 유용함을 보여주었으며, 개별 보호 그룹에 대한 공정성을 높이고 예측 성능을 개선하는 데 기여했습니다. 이 연구는 그룹 공정 모델을 학습할 때 더 유연하고 효과적인 접근 방식을 가지고 있음을 입증합니다.



### Multi-Agent Path Finding under Limited Communication Range Constraint via Dynamic Leading (https://arxiv.org/abs/2501.02770)
- **What's New**: 이 논문은 제한된 통신 범위 내에서 다중 에이전트 경로 탐색 문제를 해결하기 위한 새로운 프레임워크를 제안합니다. 기존 접근법들은 정해진 순서대로 한 번에 하나의 에이전트를 계획하는 방식으로 연산 부담을 줄였으나, 밀집된 환경에서는 이러한 방식에서 한계가 발생합니다. 이 연구는 동적 리더 선정 방식을 개발하여, 현재 리더가 진전을 이루지 못할 때 다른 에이전트를 리더로 재선정할 수 있게 하였습니다.

- **Technical Details**: 문제는 에이전트들이 장애물로 가득 찬 환경에서 목표를 향해 이동하면서도 팀 간의 통신을 유지하는 것입니다. 에이전트들은 하이퍼스페이스의 다양한 경로를 지나야 하며, 이를 수행하기 위해 MA-DL 알고리즘을 통해 다중 에이전트 경로를 확장하고 통신 범위를 유지하면서 경로 계획을 세울 수 있습니다. 이 알고리즘은 최대 25명의 에이전트를 효과적으로 처리하며, 90% 이상의 성공률을 기록합니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 다섯 종류의 환경에서 기존의 기초선 접근법들이 실패한 상황에서도 90% 이상의 성공률을 기록했습니다. 이는 동적 리더 선택 방식이 통해 에이전트들이 상대방과 계속 소통할 수 있게 하고, 경로 탐색 시 발생하는 여러 문제들을 효과적으로 해결할 수 있음을 보여줍니다. 실제 환경에서의 효율성과 효과성을 입증한 것입니다.



### Artificial Intelligence in Creative Industries: Advances Prior to 2025 (https://arxiv.org/abs/2501.02725)
Comments:
          This is an updated review of our previous paper (see this https URL)

- **What's New**: 이 논문은 2022년 이후 인공지능(AI)의 최신 발전을 살펴봅니다. 특히 창의 산업에서의 생성적 AI와 대형 언어 모델(LLM)의 혁신적 변화에 초점을 맞추고 있으며, 이러한 발전이 창작 기회와 효율성을 어떻게 확대했는지를 강조합니다. AI 통합이 전통적인 작업 흐름을 어떻게 개선했는지에 대한 논의도 포함되어 있습니다.

- **Technical Details**: 최근 LLM의 발전은 텍스트-이미지 및 텍스트-비디오 생성 기술의 유능함을 크게 향상시켰습니다. OpenAI의 GPT-4와 DALL·E 3은 이를 통해 다중 모달(multi-modal) 응답 생성이 가능하게 하였습니다. 또한, AI는 후반 작업(post-production workflow)에서 컴퓨팅 속도와 품질을 개선하여 다중 작업 환경에서도 효율성을 제공합니다.

- **Performance Highlights**: AI의 성장은 창의 산업에서 더 다양한 스토리텔링 기회를 제공합니다. 최근 Google의 Gemini와 OpenAI의 Sora와 같은 모델들은 동영상 콘텐츠 생성 분야의 혁신을 이끌고 있으며, Coca-Cola의 AI 생성 크리스마스 광고와 같은 예시가 그 효용을 보여줍니다. 이와 함께, AI 기반 비디오 코덱 및 품질 평가 분야에서도 성장이 이루어지고 있어, 향후 기술의 적용 가능성이 더욱 확장될 것입니다.



### KG-CF: Knowledge Graph Completion with Context Filtering under the Guidance of Large Language Models (https://arxiv.org/abs/2501.02711)
Comments:
          6 pages

- **What's New**: 이 논문에서는 지식 그래프 완성을 위한 새로운 프레임워크인 KG-CF를 제안합니다. 기존의 LLM(대규모 언어 모델) 기반 연구는 주로 이진 분류에 초점을 맞추어 왔으며, 효율적인 순위 기반 태스크에는 충분히 적용되지 않았습니다. KG-CF는 LLM의 추론 능력을 활용하여 부적절한 맥락을 걸러내고, 실제 데이터셋에서 우수한 성과를 달성하도록 설계되었습니다.

- **Technical Details**: KG-CF 프레임워크는 (h, r, t) 형태의 삼중(triplet) 지식 그래프를 처리하며, 그래프의 미리 정의된 경로를 샘플링하여 관련성을 평가합니다. LLM은 필터링된 맥락 정보에 접근하여 순위 점수를 생성하는 데 사용됩니다. 또한, KG-CF는 특정 서열 분류기를 경량화하여 계산 비용을 줄이고 LLM의 직접 활용을 피함으로써 효율성을 높입니다.

- **Performance Highlights**: 실험 결과, KG-CF는 다양한 현실 세계의 KG 데이터셋에서 다른 대안 모델들보다 탁월한 성과를 기록했습니다. 이 프레임워크는 그래프의 특성을 효과적으로 활용하면서도 순위 기반 평가, 즉 실제 사용 사례에 보다 적합하게 설계되었습니다. KG-CF의 성능 향상은 효율성과 유연성을 가지고 높은 신뢰성으로 이어집니다.



### Test-time Computing: from System-1 Thinking to System-2 Thinking (https://arxiv.org/abs/2501.02497)
Comments:
          work in progress

- **What's New**: 본 논문은 o1 모델의 복잡한 추론 기능 향상을 통한 테스트 시간 컴퓨팅(test-time computing) 스케일링의 중요성을 강조합니다. 특히, System-1에서 System-2로의 전환을 조명하며, 테스트 시간 컴퓨팅이 이 과정에서 핵심 역할을 한다고 설명합니다. 접근 방식은 반복 샘플링, 자기 수정 및 트리 탐색 등의 세 가지 전략을 통해 이루어집니다.

- **Technical Details**: System-1 및 System-2 사고는 심리학적 개념으로, 각각의 모델은 인간의 인지 과정을 반영합니다. System-1 모델은 빠르고 직관적인 반응을 나타내는 반면, System-2 모델은 복잡한 문제를 해결하기 위한 느리고 깊이 있는 사고를 요구합니다. 본 논문에서는 이러한 모델들이 깊은 학습 구조 내에서 어떻게 작용하는지, 특히 테스트 시간에서의 컴퓨팅 기법에 대해 설명합니다.

- **Performance Highlights**: 최근 o1 모델은 복잡한 문제 해결에서 교차 검증 및 적응성을 보여준 사례로 주목받고 있습니다. 높은 성능을 자랑하는 LLMs가 System-2 사고의 기초를 다지면서, 이러한 모델들은 심층적인 사고를 가능하게 하고 있습니다. 그러나 여전히 누적 에러의 문제와 같은 한계가 존재하며, 향후 발전 방향에 대한 논의도 포함되어 있습니다.



### LLMPC: Large Language Model Predictive Contro (https://arxiv.org/abs/2501.02486)
- **What's New**: 최근 대규모 언어 모델(LLM)의 프롬프트 기법 발전이 추론, 계획, 행동 능력을 향상시켰습니다. 본 논문은 모델 예측 제어(MPC) 관점에서 이러한 프롬프트 기법을 분석합니다. 프롬프트 계획을 활용할 때 LLM이 암묵적인 계획 비용 함수 최소화기로 작용한다는 것을 보여주며, 실제 계획 비용 함수와 평가자를 통합하여 LLM 계획 성능을 더욱 향상시킬 수 있음을 입증합니다.

- **Technical Details**: 모델 예측 제어 설정에서 에이전트는 상태 공간을 탐색해야 하며, 이를 통해 행동 순서를 결정합니다. 에이전트는 주어진 목표를 최소화하는 일련의 행동을 계획하며, 이 과정에서 목표 함수의 복잡성과 작업 관련 비용을 고려합니다. LLM은 입력 토큰 시퀀스를 기반으로 다음 토큰에 대한 확률 벡터를 생성하며, 광범위한 최적화 기법을 사용하여 최적의 행동을 선택합니다.

- **Performance Highlights**: 구성된 프롬프트는 LLM의 관리를 MPC로 간주할 수 있으며, 명시적인 목표 함수를 사용함으로써 LLM 플래너의 성능을 향상시킬 수 있음을 보여줍니다. 본 연구는 동적 시스템 제어 및 코드 생성 문제에서 접근 방식의 다양한 측면을 시연합니다. 각 실험의 코드는 GitHub에서 확인할 수 있습니다.



### Enhancing Workplace Productivity and Well-being Using AI Agen (https://arxiv.org/abs/2501.02368)
- **What's New**: 이 논문은 인공지능(AI)을 활용하여 직장 내 생산성과 직원 복지를 향상시키는 방법을 탐구하고 있습니다. 머신러닝(ML) 기술과 신경 생물학적 데이터의 통합을 통해, 가치 정렬 모델(value alignment models)과 계층적 강화 학습(HRL)을 사용하여 인본주의적 기준을 유지하는 자율적 작업 관리를 보장합니다. 이러한 접근 방식은 직원들의 생체 피드백을 이용해 개인화된 건강 알림을 생성하여 긍정적인 작업 환경을 조성합니다.

- **Technical Details**: AI 기반의 시스템은 인지 방해 요소를 감소시키고 정신 건강을 증진시키기 위한 다양한 도구들을 포함합니다. 이는 적응형 게임화(adaptive gamification) 기법과 지능형 작업 우선순위 지정(intelligent task prioritization)을 통한 연결성을 중심으로 하며, 신경 경제 최적화 모델을 통해 실시간 데이터 분석을 수행합니다. 이 모델은 기능적 자기공명영상(fMRI) 및 뇌파검사(EEG) 등의 신경 영상 기술을 활용하여 작업 환경의 방해 요소를 동적으로 조절합니다.

- **Performance Highlights**: 이 연구는 AI를 활용한 작업 환경 최적화의 잠재력을 강조하며, 신경 경제 모델(neuroeconomic models) 및 적응형 게임화 기법을 통해 직원의 참여도와 즐거움을 증대시키는 음성화된 발견들을 제시합니다. 이러한 기법들은 직원 복지 개선과 생산성 향상에 기여할 가능성이 있음을 시사하고 있습니다. 그러나 AI 개입의 장기적인 효과를 뒷받침할 실증적 증거가 부족함을 지적하면서, 향후 연구의 필요성을 강조합니다.



### CORD: Generalizable Cooperation via Role Diversity (https://arxiv.org/abs/2501.02221)
- **What's New**: 이번 연구에서는 Cooperative Multi-Agent Reinforcement Learning (MARL) 분야에서 새로운 접근 방식인 CORD를 제안합니다. CORD는 역할 다양성을 통해 일반화 가능한 협력을 가능하게 하며, 이는 기존의 방법들이 일반화 문제에서 어려움을 겪는 것과 대조적입니다. 높은 수준의 컨트롤러가 저수준 에이전트에게 역할을 할당하는 방식을 채택하여 훈련된 에이전트들이 해당 역할에 대해 더욱 효과적으로 협력할 수 있도록 합니다.

- **Technical Details**: CORD의 핵심적인 기술은 역할 엔트로피(role entropy)를 극대화하면서 제약조건을 적용하는 것입니다. 이 제약 조건은 역할 할당에 있어 합리적인 방식으로 인과적 영향을 분해할 수 있도록 하며, 이는 역할 이질성(role heterogeneity)을 통해 일관되고 중복되지 않는 역할 클러스터를 생성합니다. 이러한 구조는 협력을 더 잘 이끌어내며, 다양한 협업 다중 에이전트 작업에서 성과를 측정할 수 있습니다.

- **Performance Highlights**: CORD는 다양한 협력 다중 에이전트 작업을 평가한 결과, 기존의 기준보다 더 나은 성능을 나타냈습니다. 특히 일반화 테스트에서 우수한 성능을 보였으며, 절제된 목표의 효능을 추가적인 실험을 통해 입증했습니다. 이러한 결과는 향후 실제 환경에 적용 가능한 MARL 모델 개발에 기여할 것으로 기대됩니다.



### Table as Thought: Exploring Structured Thoughts in LLM Reasoning (https://arxiv.org/abs/2501.02152)
- **What's New**: 새로운 연구 프레임워크인 'Table as Thought'가 제안되었습니다. 이 프레임워크는 인지 신경과학(cognitive neuroscience) 이론에 영감을 받아, 체계적인 과정을 통해 대형 언어 모델의 추론 능력을 향상시키고자 합니다. 기존의 사고 체계는 주로 사고의 순서에 중점을 두었으나, 개별 사고 단계의 구조는 충분히 탐구되지 않았습니다.

- **Technical Details**: Table as Thought는 사고 과정을 표 형식으로 구성하는 방식으로, 각 행은 순차적인 사고 단계를 나타내고 각 열은 중요한 제약(constraints)이나 맥락(contextual information)을 캡처하여 추론을 강화합니다. 이 과정은 자기 검증(self-verification) 단계에서 표의 완전성과 정확성을 보장할 때까지 반복적으로 진행됩니다.

- **Performance Highlights**: 실험 결과 'Table as Thought'가 계획(task planning) 작업에서 우수한 성능을 보였으며, 비구조적 사고 기반선(unstructured thought baselines)에 비해 수학적 추론(mathematical reasoning)에도 강력한 잠재력을 가지고 있음을 입증했습니다. 이 연구는 LLM 내의 사고 표현(thought representation)을 정교화하는 새로운 탐색을 제공하며, AI 인지(cognition) 및 추론(reasoning) 발전의 기초를 닦고 있습니다.



### Disagree and Commit: Degrees of Argumentation-based Agreements (https://arxiv.org/abs/2501.01992)
Comments:
          To appear eventually in the Autonomous Agents and Multi-Agent Systems journal

- **What's New**: 이 논문에서는 사람이 합의에 도달할 때 전체적인 동의가 없더라도 부분적인 동의만으로 결정을 내릴 수 있다는 개념을 제시합니다. 이는 인공지능(AI) 시스템의 의사결정 자동화 상황에서 매우 중요한 의미를 가집니다. 우리는 인공지능 에이전트들이 이러한 부분적인 합의를 형성할 수 있도록 하는 'agreement scenarios'의 개념을 도입합니다.

- **Technical Details**: 이 연구에서는 형식적인 모델을 사용하여 논증(Argumentation)에서 합의의 정도(agreements)와 만족의 정도(degrees of satisfaction)를 정의합니다. 특히, abstract argumentation과 value-based argumentation에 중점을 두고, 에이전트들이 서로의 주제에 대한 의견이 어떤 영향을 받는지를 분석합니다. 자료 분석을 통해 새로운 정보로 인해 합의의 정도가 어떻게 변화하는지를 살펴봅니다.

- **Performance Highlights**: 본 논문에서는 제안된 개념들이 실제로 적용 가능하다는 것을 입증하기 위해 논증 기반 추론 소프트웨어 라이브러리를 사용한 구현 사례를 제시합니다. 이를 통해 부분적인 합의를 형성하는 과정이 동적인 환경에서도 신뢰성 있게 유지될 수 있음을 보여주고 있습니다. 이 연구는 여러 에이전트 간의 상호작용에서 올바른 의사결정을 내리기 위한 기반을 제공합니다.



### Gaussian Masked Autoencoders (https://arxiv.org/abs/2501.03229)
- **What's New**: 이 논문은 Masked Autoencoders (MAE)와 Gaussian Splatting을 결합한 새로운 접근법을 소개합니다. 특히, Gaussian Masked Autoencoder (GMAE)를 도입하여 의미론적 추상화(semantic abstraction)와 공간적 이해(spatial understanding)를 동시에 학습하도록 설계했습니다. GMAE는 MAE와 유사하게 픽셀 공간에서 이미지를 전체적으로 재구성할 뿐만 아니라, 중간 단계로 3D Gaussian 기반 표현을 사용하여 이미지를 렌더링하는 기능을 포함하고 있습니다.

- **Technical Details**: GMAE는 3D Gaussians를 중간 이미지 표현으로 활용하여 의미적 및 공간적 이해를 가능하게 합니다. 이 모델은 가우시안 표현의 동적 특성을 통해 자연 세계의 구조를 학습하고, 이미지의 깊이 불연속성을 토대로 figure-ground segmentation, 레이어링, 모서리 검출(edge detection)과 같은 다양한 제로샷 학습 능력을 구현합니다. GMAE는 MAE와 비교했을 때 비슷한 학습 성능을 보여주는 동시에, 3D Gaussian을 통한 렌더링 덕분에 계산 효율성 또한 향상됩니다.

- **Performance Highlights**: GMAE는 MAE와 유사한 이미지 분류 및 객체 탐지 작업에서 높은 성능을 보여주며, 사용하는 가우시안의 수가 늘어날수록 표현 품질이 향상됩니다. 또한, GMAE는 표준 MAE 훈련에 비해 매우 적은 오버헤드를 추가하며, 실제 훈련 시간도 거의 비슷하게 유지됩니다. 이러한 성능은 GMAE가 중간 단계 표현을 활용하는 응용 프로그램에서 더욱 유리할 수 있음을 시사합니다.



### LightGNN: Simple Graph Neural Network for Recommendation (https://arxiv.org/abs/2501.03228)
Comments:
          Accepted to WSDM 2025 Oral

- **What's New**: LightGNN은 추천 시스템을 위한 경량화된 GNN 프루닝(framing) 프레임워크로, 모델의 복잡성을 줄이면서 협업 모델링 기능을 유지합니다. 이 프레임워크는 자원 친화적(hierarchical) 지식 증류(knowledge distillation) 목표에 따라 중간 레이어를 통해 관측된 그래프를 보강하여 성능을 향상시킵니다. LightGNN은 긍정적인 피드백을 사용하는 대신에 사용자 상호작용 그래프에서 불필요한 경량(edge) 및 임베딩(embedding)을 제거하여 효율성 높은 추천을 제공합니다.

- **Technical Details**: LightGNN은 그래프 구조 학습을 통합하여 각 엣지와 임베딩 항목의 중복 또는 노이즈 가능성을 명확하게 평가합니다. 이 과정은 상류 추천 작업과 계층적 지식 증류 패러다임을 활용하여 감독(supervised) 방식으로 이루어집니다. 새로운 계층적 KD 접근법은 커다란 학습 능력을 유지해 주어, 높은 비율의 압축 상황에서도 추천 성능을 지킬 수 있도록 합니다.

- **Performance Highlights**: LightGNN은 공개 데이터 세트에서 이루어진 큰 규모의 실험을 통해 계산 효율성과 추천 정확도 모두에서 성능이 크게 향상되었습니다. 이 프레임워크는 엣지 수를 80%, 임베딩 항목 수를 90%까지 줄이면서도 복잡한 최신 기술(technology) 대비 유사한 성능을 유지하였습니다. 따라서 LightGNN은 추천 정확성, 추론 효율성, 모델 강건성, 그리고 해석 가능성 측면에서 뛰어난 성능을 보여줍니다.



### BoostStep: Boosting mathematical capability of Large Language Models via improved single-step reasoning (https://arxiv.org/abs/2501.03226)
Comments:
          Codes and Data are available at this https URL

- **What's New**: 리서치에서는 BoostStep이라는 새로운 접근법을 소개하여 이론적 레벨을 문제 레벨에서 각각의 리즈닝 프로세스를 지원하는 단계 레벨로 전환합니다. 이는 관련된 ICL 예시를 각 단계에서 제공하며, 적절한 가이드라인을 보장하는 데에 중점을 둡니다. 또한, 기존의 코스 문제 기반 전략보다 훨씬 관련성 높은 예시를 제공함으로써 모델의 리즈닝 품질을 지속적으로 향상시킵니다.

- **Technical Details**: BoostStep은 리즈닝 중 단계 레벨의 적합성과 ICL 예시를 조화롭게 맞춘 후, 처음 시도(first-try) 전략을 통해 적절한 리즈닝 단계를 지원합니다. 이 방법은 단계별로 솔루션을 문제로 나누고, 각 단계에 대한 유사한 문제를 검색하여 적절한 해결 방법을 제시합니다. 이는 기존의 문제 레벨 ICL 접근법보다 더 밀접한 가이드를 제공하며, 각 단계의 정확성을 높이는 데 기여합니다.

- **Performance Highlights**: BoostStep은 다양한 수학 벤치마크에서 GPT-4o 모델 성능을 3.6% 향상시키고 Qwen2.5-Math-72B 모델은 2.0% 향상시키는 결과를 보였습니다. Monte Carlo Tree Search(MCTS)와 결합했을 때는 7.5%의 성과 향상이 있었습니다. 이는 BoostStep이 단순한 리즈닝 성능 향상을 넘어 모델의 결정 과정까지 개선하는 효과를 입증합니다.



### Automated Generation of Challenging Multiple-Choice Questions for Vision Language Model Evaluation (https://arxiv.org/abs/2501.03225)
Comments:
          Project page: this https URL

- **What's New**: 비전 언어 모델(Vision Language Models, VLMs)의 빠른 발전을 반영하여, 본 논문에서는 AutoConverter라는 새로운 프레임워크를 도입하여 열린 질문을 객관적인 선택형 질문으로 변환합니다. 이 접근은 문제 생성 과정을 줄이며 평가의 일관성을 높이는 데 기여합니다. AutoConverter는 자동으로 정확하고 도전적인 선택형 질문을 생성할 수 있는 다중 에이전트 시스템으로 설계되었습니다.

- **Technical Details**: AutoConverter는 다양한 에이전트가 협력하여 열린 질문을 선택형으로 변환하는 방식으로 작동합니다. 특히, 정확성을 보장하기 위해 질문의 정확성을 평가하는 에이전트가 있어, 생성된 선택지의 적합성을 검증합니다. 이를 통해 20개의 기존 VQA 데이터셋에서 9,018개의 질문으로 구성된 VMCBench라는 새로운 벤치마크를 생성하였습니다.

- **Performance Highlights**: 33개의 최첨단 VLM을 VMCBench에서 평가한 결과, AutoConverter가 생성한 질문이 인간이 제작한 질문에 비해 유사하거나 더 낮은 정확도를 보이면서도 높은 도전성을 유지함을 보여주었습니다. 이러한 결과는 AutoConverter의 다용도성을 입증하며, 교육 및 기존 선택형 데이터셋의 개선에도 활용할 수 있는 가능성을 제공합니다.



### Detecting AI-Generated Text in Educational Content: Leveraging Machine Learning and Explainable AI for Academic Integrity (https://arxiv.org/abs/2501.03203)
- **What's New**: 본 연구는 AI 생성 콘텐츠를 학생 작업에서 감지하기 위한 도구를 제공하여 학문적 무결성을 향상시키는 것을 목표로 합니다. 특히 CyberHumanAI 데이터셋을 생성하여, 인간이 작성한 콘텐츠와 ChatGPT에 의해 생성된 콘텐츠를 비교 분석하였습니다. 이 연구는 교육에서의 AI 통합을 책임감 있게 지원하고 윤리적 기준을 유지하기 위해 투명성과 책임성을 촉진합니다.

- **Technical Details**: CyberHumanAI 데이터셋은 1000개의 관측치로 구성되어 있으며, 그 중 500개는 인간이 작성하고 나머지 500개는 ChatGPT에 의해 생성되었습니다. 다양한 머신러닝(ML) 및 딥러닝(DL) 알고리즘을 평가한 결과, 전통적인 ML 알고리즘인 XGBoost와 Random Forest가 각각 83% 및 81%의 높은 정확성을 기록했습니다. 또한, 짧은 콘텐츠의 분류가 긴 콘텐츠에 비해 더 어려운 것으로 나타났습니다.

- **Performance Highlights**: 설명 가능한 인공지능(XAI)을 활용하여 ML 모델의 예측에 영향을 미치는 특징들을 식별하였습니다. 분석 결과, 인간 작성 콘텐츠는 실용적인 언어를 사용하는 경향이 있는 반면, AI 생성 텍스트는 좀 더 추상적이고 공식적인 용어로 특징 지어졌습니다. 제안된 모델은 Pure AI, Pure Human 및 혼합 클래스를 분류할 때 약 77.5%의 정확성을 기록했고, 이는 GPTZero의 48.5%보다 현저히 높은 수치입니다.



### Classifier-Guided Captioning Across Modalities (https://arxiv.org/abs/2501.03183)
- **What's New**: 최근 캡션 생성 시스템은 특정 데이터 세트에서 훈련된 언어 모델을 사용하여 일반화에 제약을 받고 있습니다. 특히 오디오 또는 비디오 캡션 생성에는 다른 의미적 단서가 필요하며, 이를 해결하기 위해 새로운 프레임워크를 제안합니다. 이 프레임워크는 전이학습 없이 추론 단계에서만 작동하며, 기존 캡션 생성 모델의 품질을 크게 향상시킵니다.

- **Technical Details**: 이 연구에서는 오디오 캡션을 생성하는 과정에서 오디오의 가청성을 나타내기 위해 텍스트 분류기를 활용한 새로운 방법론을 도입합니다. 이 프레임워크는 사전 훈련된 캡셔닝 시스템과 가청성 분류기로 구성되어 있으며, 두 가지 손실 함수를 통해 성능을 최적화합니다. 이 방식은 다양한 모달리티에 쉽게 통합할 수 있어 전반적인 유연성을 제공합니다.

- **Performance Highlights**: 제안된 방법은 AudioCaps 및 Clotho 데이터 세트에서 실시된 실험에서 기존 모델보다 성능 향상을 보였습니다. 특히 기존 제로샷 오디오 캡션 시스템과 결합하면 품질이 크게 개선되며 제로샷 오디오 캡션에서 최첨단 성능을 달성했습니다. 이러한 결과는 다양한 실시간 환경에서도 효과적인 캡션 생성이 가능함을 시사합니다.



### Boosting Explainability through Selective Rationalization in Pre-trained Language Models (https://arxiv.org/abs/2501.03182)
Comments:
          KDD 2025 research track

- **What's New**: 본 논문은 사전 훈련된 언어 모델(Pre-trained Language Models, PLMs)이 기존의 선택적 합리화(selective rationalization) 기법에서 심각한 퇴화(degeneration) 및 실패(failure) 문제를 겪는다는 사실을 밝히고, 이를 해결하기 위한 PLMR(Pre-trained Language Model's Rationalization)라는 새로운 방법론을 제안합니다.

- **Technical Details**: PLMR은 PLMs를 두 개의 독립적인 부분, 즉 합리화 생성기(generator)와 예측기(predictor)로 나누어 NLP 과제를 수행하며 해석 가능한 합리화를 제공합니다. 이 방법은 관련 없는 토큰을 잘라내어 토큰의 동질성(homogeneity)을 완화하고, 예측기 부분에서는 전체 텍스트 정보를 활용하여 예측을 표준화합니다.

- **Performance Highlights**: 실험 결과, PLMR은 GRU를 사용하는 방법에 비해 최대 9%, 기존 PLMs 기반 방법에 비해 최대 17% 높은 F1 점수를 기록하여 PLM을 사용한 합리화에서 퇴화 및 실패 문제를 효과적으로 해결할 수 있음을 보여줍니다.



### FaceSpeak: Expressive and High-Quality Speech Synthesis from Human Portraits of Different Styles (https://arxiv.org/abs/2501.03181)
- **What's New**: 본 논문에서는 다양한 이미지 스타일에서 감정 표현과 정체성 특성을 추출하는 새로운 접근법인 FaceSpeak를 소개합니다. 기존의 얼굴 이미지 기반 음성 합성 방법들의 제한점을 극복하고, 다양한 캐릭터에 맞춘 음성을 생성할 수 있는 가능성을 열었습니다. 또한, Expressive Multi-Modal TTS라는 혁신적인 데이터셋을 새로이 구축하여 다중 모달리티의 데이터 부족 문제를 해결하고자 했습니다.

- **Technical Details**: FaceSpeak는 자유로운 이미지 스타일 입력의 영향을 받아, 정체성과 감정 특징을 분리하여 음성을 합성합니다. 이 과정에서 배경이나 의상, 헤어스타일 등의 불필요한 정보는 최소화하여, 캐릭터의 성격과 잘 맞는 음성을 생성하게 됩니다. 이를 통해 음성 합성 시스템의 유연성과 다양성을 크게 향상시킬 수 있는 방법론을 제안합니다.

- **Performance Highlights**: 대규모의 실험을 통해 FaceSpeak가 높은 자연성과 품질을 가진 정체성에 맞춘 음성을 생성할 수 있음을 입증했습니다. 다양한 스타일 이미지에 대한 검사와 평가에서도 우수한 성능을 보여주었으며, 여러 형태의 음성과 감정 표현을 통합하여 보다 풍부한 음성 합성을 가능하게 했습니다.



### GLiREL -- Generalist Model for Zero-Shot Relation Extraction (https://arxiv.org/abs/2501.03172)
Comments:
          Submitted to NAACL 2025

- **What's New**: 본 논문에서는 제로샷 관계 추출(zero-shot Relation Extraction)을 위한 효율적인 아키텍처이자 훈련 패러다임인 GLiREL(Generalist Lightweight model for zero-shot Relation Extraction)을 소개합니다. GLiREL은 여러 엔티티 사이에서 관계 레이블을 단일 포워드 패스(single forward pass)로 정확하게 예측할 수 있도록 설계되었습니다. 실험 결과, FewRel과 WikiZSL 벤치마크에서 저희 접근 방식이 제로샷 관계 분류(task)에서 최신 기술(State-of-the-Art) 결과를 달성하였음을 보여줍니다. 또한 다양한 관계 레이블을 사용할 수 있는 데이터셋을 합성적으로 생성하는 프로토콜도 기여하였습니다.

- **Technical Details**: GLiREL 아키텍처는 세 가지 주요 구성 요소로 나누어집니다: 1) 텍스트 인코더로서의 사전 훈련된 양방향 언어 모델, 2) 엔티티 쌍 표현 모듈, 3) 점수 계산 모듈입니다. 이 아키텍처는 관계 레이블과 엔티티 쌍의 임베딩을 동일한 잠재 공간(latent space)에서 인코딩하여 유사성을 계산합니다. 우리는 하위 작업에서 우수한 성능을 보이는 DeBERTa V3-large를 인코더 모델로 선택하였습니다.

- **Performance Highlights**: GLiREL은 기존의 다른 제로샷 관계 분류 모델들보다 더 효율적이며, 여러 엔티티 쌍을 단일 입력으로 분류할 수 있는 장점을 가지고 있습니다. 랜덤한 관계 레이블 로드리딩 없이 임베딩을 처리할 수 있어, 모델이 동시에 여러 레이블과 엔티티 쌍 간의 상호작용을 포착합니다. 이러한 효율성 덕분에 실제 시나리오에서 많은 엔티티를 포함하는 경우에서도 뛰어난 성능을 발휘합니다.



### The Scaling Law for LoRA Base on Mutual Information Upper Bound (https://arxiv.org/abs/2501.03152)
- **What's New**: 본 논문은 LoRA(저랭크 적응) 기법을 이용한 모델 파인튜닝의 새로운 내부 평가 지표인 MIUB(상호정보량 상한)을 제안합니다. MIUB는 대형 모델과 LoRA 모듈 간의 정보 의존 관계를 정량적으로 분석하기 위한 방법으로, 기존의 외부 지표인 cross-entropy와 perplexity보다 정확하고 안정적입니다. 이 연구는 대형 모델 파인튜닝의 스케일링 법칙을 체계적으로 조사하여, 모델의 크기와 데이터 복잡성이 파인튜닝 효과에 미치는 영향을 보다 효과적으로 평가할 수 있는 방법을 제공합니다.

- **Technical Details**: LoRA는 대형 언어 모델의 파라미터를 동결하고, 새로운 저랭크 파라미터 행렬을 추가하여 구조적 지식을 학습하는 방식으로, 파인튜닝에서 효율성을 극대화한 방법입니다. MIUB는 대형 모델의 출력 분포와 LoRA의 출력 분포 간의 상호정보량을 기반으로 한 지표로, LoRA를 주의 레이어와 FFN(Feed-Forward Network) 레이어에 추가함으로써 계산됩니다. 이 접근법은 파인튜닝의 정확성을 높이고, 파인튜닝 후의 효과를 더욱 정밀하게 정량화할 수 있게 해줍니다.

- **Performance Highlights**: 실험 결과, MIUB는 대형 모델의 크기 증가, LoRA 순위 증가, 데이터 크기 증가에 따라 감소하며, 이는 모델의 실제 효과(예: 정확도) 변화 추세와 잘 일치합니다. MIUB는 기존의 Cross-Entropy 및 Perplexity 지표보다도 더 정확하고 안정적인 결과를 제공합니다. 이러한 연구 결과는 모델 압축 및 개인화 요구에 따른 대형 모델의 성능 측정에 중요한 기여를 할 것으로 기대됩니다.



### Geometry Restoration and Dewarping of Camera-Captured Document Images (https://arxiv.org/abs/2501.03145)
Comments:
          28 pages, 16 figures

- **What's New**: 이번 연구는 카메라로 촬영한 종이 문서의 디지털 이미지에서 위상 복원(Topology Restoration)을 위한 방법을 개발하는 데 초점을 맞추었습니다. 알고리즘을 통해 문서의 윤곽을 감지하고(segmentation), 기하학적으로 복원하는 데 필요한 다양한 기술을 적용했습니다. 전통적인 컴퓨터 비전(computer vision) 기법을 사용하여 문서의 복원 과정을 더욱 효율적이고 빠르게 진행할 수 있음을 보여주었습니다.

- **Technical Details**: 이 방법론은 딥 러닝(deep learning)을 활용하여 문서 윤곽을 감지하고, 그 후에는 컴퓨터 비전(computer vision) 기술을 통해 2D 그리드를 생성하고 비선형 왜곡을 교정하는 데 중점을 두고 있습니다. 큐빅 다항식 보간(cubic polynomial interpolation)을 사용해 이미지를 재매핑(remapping)하는 과정을 포함시켜 문서의 구조를 효과적으로 복원할 수 있었습니다. 또한, 자동 문서 디워핑(dewarping) 및 복원을 위한 새로운 파이프라인(pipeline)을 개발하였습니다.

- **Performance Highlights**: 실험 결과, 개발된 방법론은 기존의 벤치마크(benchmark)들, 특히 모바일 앱 및 RectiNet, DocGeoNet, DocTr++와 같은 인기 있는 딥러닝 솔루션들과 비교했을 때 시각적으로나 문서 가독성 측면에서 우수함을 입증하였습니다. 문서 이미지의 상태를 개선하고 OCR(Optical Character Recognition) 시스템의 효율성을 향상시키기 위한 길을 열어 주었습니다. 이 연구는 종이 문서의 고품질 디지털 복사본을 생성하는데 기여할 것으로 기대됩니다.



### PRMBench: A Fine-grained and Challenging Benchmark for Process-Level Reward Models (https://arxiv.org/abs/2501.03124)
Comments:
          Project Page: this https URL

- **What's New**: 본 논문에서는 복잡한 추론 및 의사결정 작업에서 중요한 역할을 하는 Process-level Reward Models (PRMs)에 대한 새로운 벤치마크인 PRMBench를 소개합니다. PRMBench는 PRMs의 미세 조정된 오류 감지 능력을 평가하기 위해 설계되었으며, 6,216개의 문제와 83,456개의 단계 수준 레이블을 포함하고 있어 다양한 차원에서 모델을 평가합니다.

- **Technical Details**: PRMBench는 PRMs의 성능을 간단함(simplicity), 타당성(soundness), 민감도(sensitivity) 등 여러 차원에서 평가합니다. 이를 통해 현재 PRMs가 직면한 다양한 오류 유형을 탐지하는 데 필요한 능력을 정교하게 측정할 수 있습니다. 15개의 모델에 대한 실험을 통해 우리는 PRMs의 중대한 약점을 발견하였습니다.

- **Performance Highlights**: 현재 PRMs의 성능을 체계적으로 평가하는 것이 부족하다는 사실이 발견되었으며, 이는 프로세스 수준 평가에 내재된 도전 과제를 강조합니다. 이러한 연구 결과는 PRM 평가 및 개발에 있어 중요한 미래 연구 방향을 제시합니다. PRMBench가 PRM 연구의 발전을 위한 강력한 벤치마크가 되기를 바랍니다.



### From Models to Network Topologies: A Topology Inference Attack in Decentralized Federated Learning (https://arxiv.org/abs/2501.03119)
- **What's New**: 이 논문은 Decentralized Federated Learning (DFL) 시스템의 오버레이 토폴로지를 모델 행동을 기반으로 추론하는 새로운 Topology Inference Attack을 제안합니다. 이러한 공격의 가능성을 탐구하며 공격자의 능력과 지식을 기준으로 한 분류 체계를 제공합니다. DFL 시스템에서의 민감한 정보 유출 위험성을 강조하고 이를 보호하기 위한 통찰력을 제공합니다.

- **Technical Details**: DFL의 오버레이 토폴로지는 모델 수렴에 중대한 영향을 미치는 무방향 그래프 G=(V,E)로 모델링됩니다. 각 노드의 연결 상태는 인접 행렬 A로 나타내며, D는 차수 행렬로 직계 본을 나타냅니다. 이 논문은 로컬 훈련 및 모델 집계 과정을 포함하는 T 라운드에 걸쳐 반복적인 집계를 수행하는 과정을 기술합니다.

- **Performance Highlights**: 실험 결과는 개별 노드의 공개 모델만 분석해도 DFL 토폴로지를 정확히 추론할 수 있음을 보여줍니다. 이는 DFL 시스템에서 민감한 정보 유출의 위험성을 강조하며, DFL 환경에서 프라이버시 보호를 개선하기 위한 귀중한 통찰력을 제공합니다. 이러한 발견은 안전한 DFL 시스템 설계에 기여할 것으로 기대됩니다.



### LangFair: A Python Package for Assessing Bias and Fairness in Large Language Model Use Cases (https://arxiv.org/abs/2501.03112)
Comments:
          Journal of Open Source Software; LangFair repository: this https URL

- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 편향성을 평가하고 공정성 리스크를 측정하기 위해 LangFair라는 오픈소스 Python 패키지를 소개합니다. 이 패키지는 LLM 사용자들이 자신들의 특정 응용 사례에 맞춘 평가 데이터셋을 쉽게 생성하고, 관련 메트릭을 계산할 수 있도록 설계되었습니다. 또한, LangFair는 메트릭 선택에 도움을 줄 수 있는 실행 가능한 결정 프레임워크를 제공합니다.

- **Technical Details**: LangFair는 사용자가 제공한 프롬프트를 기반으로 LLM 응답에서 편향 및 공정성을 평가할 수 있는 도구입니다. 기존의 평가 도구들이 정적 기준 데이터셋에 기반해 LLM을 평가하는 반면, LangFair는 프롬프트 특화 리스크를 고려하여 개인화된 평가를 가능하게 합니다. 이 패키지에는 응답 생성 클래스인 ResponseGenerator와 CounterfactualGenerator가 포함되어 있어 사용자의 요구에 맞게 평가 데이터셋을 생성할 수 있습니다.

- **Performance Highlights**: LangFair는 실제 LLM 기반 시스템의 요구를 충족시키며, 사용자 제공 프롬프트에 의존하여 LLM 응답에서 메트릭을 계산합니다. 이를 통해 출력 기반 메트릭이 보다 신뢰성이 높고 실제 사용 사례에서 더 유용할 수 있음을 입증합니다. LangFair의 다양한 메트릭 클래스는 독립적인 유용성과 함께 특별한 태스크에 맞춘 평가를 지원하여 LLM의 편향성을 체계적으로 분석하도록 돕습니다.



### Personalized Fashion Recommendation with Image Attributes and Aesthetics Assessmen (https://arxiv.org/abs/2501.03085)
- **What's New**: 이번 논문은 개인화된 패션 추천 시스템의 문제를 해결하기 위한 새로운 접근 방식을 제안합니다. 연구팀은 사용자의 미적 취향(aesthetic appetite)과 관련된 정보를 더욱 효율적으로 활용하고, 최신 아이템들에 대한 냉시작(cold-start) 문제를 극복하는 데 주력했습니다. 기존 연구에서는 이미지와 텍스트를 별개의 요소로 간주했지만, 본 연구에서는 두 가지 정보를 결합하여 풍부한 속성(attribute) 그래프(attribute graph)를 생성합니다.

- **Technical Details**: 연구에서는 이미지 최적화(image utilization)와 사용자 모델링(user modeling)에 있어 노이즈를 감소시키는 두 가지 속성 그래프(attribute graphs)를 생성하여 정보의 가용성을 높였습니다. 또한, 최근의 대형 언어 모델(large language models) 및 비전 모델(vision models)의 발전을 활용, 세밀한 속성을 추출하는 두 가지 프롬프트(prompts)를 통해 실험을 진행했습니다. 이 접근법은 이미지와 텍스트 정보를 통합함으로써 더 나은 성능을 기대할 수 있도록 설계되었습니다.

- **Performance Highlights**: IQON3000 데이터셋을 사용한 초기 실험에서 제안된 방법은 기존 기준선(baselines)과 비교하여 경쟁력 있는 정확도를 달성했습니다. 이는 개인화된 패션 추천 시스템의 성능을 개선할 수 있는 가능성을 보여줍니다. 본 연구 결과는 패션 추천의 정확성을 높이고 최신 트렌드에 유연하게 대응할 수 있는 시스템 개발에 기여할 것으로 예상됩니다.



### Through-The-Mask: Mask-based Motion Trajectories for Image-to-Video Generation (https://arxiv.org/abs/2501.03059)
- **What's New**: 이번 논문에서는 정적 이미지를 기반으로 사실적인 비디오 시퀀스를 생성하는 Image-to-Video (I2V) 생성 기술을 다룹니다. 기존의 방법들은 멀티 오브젝트(multi-object) 환경에서 객체의 정확한 움직임을 생성하는 데 어려움을 겪고 있으며, 이러한 한계를 극복하기 위해 두 단계의 구성적 프레임워크를 제안합니다. 첫 번째 단계에서는 명시적인 중간 표현을 생성하고, 두 번째 단계에서는 이 표현을 바탕으로 비디오를 생성합니다.

- **Technical Details**: 제안된 방법은 mask 기반의 motion trajectory를 중간 표현으로 사용하며, 이는 객체의 의미 정보와 움직임을 포착합니다. 우리는 각 객체에 대한 객관적인 주의(object-level attention) 목표를 활용하여 두 번째 단계에서 학습된 표현을 통합합니다. 특히, 공간적(masked cross attention) 및 시공간적(masked spatio-temporal self-attention) 자기 주의(self-attention) 목표를 사용하여 프레임 간 일관성을 보장합니다.

- **Performance Highlights**: 본 연구는 다수의 오브젝트와 높은 동작이 포함된 도전적인 벤치마크에서 방법의 효과를 검증하였으며, 시간적 일관성, 움직임의 사실성, 텍스트 프롬프트 충실도에 대한 최신 기술 수준(state-of-the-art) 결과를 도출했습니다. 또한, 단일 및 다중 객체 I2V 생성을 위한 새로운 벤치마크를 도입하였고, 이 벤치마크에서도 우리의 방법이 우수한 성능을 보여줍니다.



### Survival Analysis Revisited: Understanding and Unifying Poisson, Exponential, and Cox Models in Fall Risk Analysis (https://arxiv.org/abs/2501.03058)
- **What's New**: 이번 연구는 생존 분석의 기초와 응용 측면을 탐구하며, 낙상 위험 평가를 사례로 사용합니다. 주요 시간 관련 확률 분포와 통계 방법을 재검토하며, 로지스틱 회귀(logistic regression), 포아송 회귀(Poisson regression), 지수 회귀(Exponential regression), 그리고 콕스 비례 위험 모형(Cox Proportional Hazards model)의 관계를 통합적으로 제시합니다. 특히 생존 맥락에서 포아송 회귀가 콕스 모델의 특정 사례임을 보여주는 등의 기여를 통해 이해의 간극을 메우고 생존 모형의 단순성과 해석 가능성을 강화합니다.

- **Technical Details**: 생존 분석은 노인 인구의 건강 문제를 해결하는 데 중대한 역할을 하며, 특히 질병 진행, 진단 후 생존 시간 및 회복 기간 등과 같은 이벤트 예측에 유용합니다. 이러한 분석 방법은 다수의 질문을 단일 프레임워크 내에서 다룰 수 있는 장점이 있으며, 예를 들어 특정 시간 간격 내에 낙상의 위험을 예측하고, 공변량(covariates)의 영향을 분석하며, 다음 사건까지의 예상 시간을 추정할 수 있습니다. 이 연구에서는 로지스틱 회귀를 시작으로, 포아송과 지수 분포를 도입하여 시간 요소를 통합하고, 생존 분석 프레임워크 내에서 일반화 선형 모형을 탐구합니다.

- **Performance Highlights**: 이 논문은 이론적 통찰을 제공하는 것 외에도 생존 분석 방법의 실제 응용을 강조하고, 낙상 탐지 사례 연구를 통해 세 가지 주요 목표를 달성합니다: 특정 간격 내에 사건 발생 위험 예측, 사건 위험에 대한 공변량의 영향 해석, 개별 대상자의 다음 사건 발생까지 예상 시간 추정. 이러한 응용은 생존 모델의 가치를 강조하며, 임상 의사결정 및 자원 배치에 유용한 통찰력을 제공하여 건강 관리 환경에서의 설명 가능성과 강건성을 더욱 강조합니다.



### To Analyze and Regulate Human-in-the-loop Learning for Congestion Games (https://arxiv.org/abs/2501.03055)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2211.14029

- **What's New**: 이 논문에서는 교통망에서의 자기 이익형 사용자의 라우팅 문제를 분석하고 이를 해결하기 위한 메커니즘을 제시합니다. 기존의 혼잡 게임 연구에서 사용자들이 최적의 경로를 선택하기 어려운 점을 지적하고, 사용자가 자신이 도착할 때마다 실시간 데이터를 통해 경로를 결정하는 동적인 환경을 고려합니다. 특히, 이동 crowdsourcing과 혼잡 게임 간의 접목이라는 새로운 접근 방식을 통해 사회적 최적을 달성하는 방안을 모색합니다.

- **Technical Details**: 이 연구는 사용자들의 이익을 극대화하고 사회적 비용을 최소화하기 위한 새로운 정보를 활용한 메커니즘인 선택적 정보 공개(selective information disclosure, SID)를 제안합니다. 이 메커니즘은 사용자가 특정 경로를 탐색하고자 할 때만 최신 정보를 제공하여, 탐색과 활용 간의 균형을 맞추도록 설계되었습니다. 그런 다음,부분 관측 가능한 마르코프 결정 과정(partially observable Markov decision process, POMDP)을 사용하여 사용자의 동적인 라우팅 문제를 모델링합니다.

- **Performance Highlights**: 제안된 SID 메커니즘은 물리적 최적의 정책에 비해 사용자가 경로를 탐색할 때 발생하는 가격의 무질서(price of anarchy, PoA)를 2 미만으로 감소시키는 것이 입증되었습니다. 시뮬레이션 결과, SID 메커니즘은 평균적인 경우에서도 최적 성능에 가까운 결과를 보이며, 모든 선형 경로 그래프와 동적 마르코프 체인에 확장되어 효과적으로 작동함을 확인했습니다.



### Single-Channel Distance-Based Source Separation for Mobile GPU in Outdoor and Indoor Environments (https://arxiv.org/abs/2501.03045)
Comments:
          Accepted by ICASSP2025. \c{opyright} 2025 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component

- **What's New**: 이번 연구는 실외 환경에서의 거리 기반 소스 분리(DSS)의 필요성과 중요성을 강조합니다. 기존의 연구들이 주로 실내 환경에 초점을 맞춘 반면, 제안된 모델은 실제 환경에서의 오디오 소스의 고유한 특성을 캡처하기 위해 설계되었습니다. 이는 두 단계의 conformer 블록, 선형 관계 주의(self-attention) 메커니즘, TensorFlow Lite GPU 위임 등의 고급 기술을 포함합니다.

- **Technical Details**: 연구는 입력 혼합 신호를 보다 잘 근사하기 위해 여러 물리적 정보와 신호 경로를 활용합니다. 이 과정에서 인버스 제곱 법칙에 따른 거리의 강도 변화, 직접-잔향 비율(DRR), 방향성 마이크로폰의 근접 효과 등을 고려합니다. 실외 환경에서의 DSS 모델 적용을 위해 풍력, 주변 소음, 다양한 소스의 거리 등과 같은 환경 변수들을 포함할 수 있는 강력한 알고리즘을 개발했습니다.

- **Performance Highlights**: 제안된 DSS 모델은 에너지 효율성을 20배, 실시간 추론 속도를 10배 향상시켰으며, 특히 모바일 GPU에 최적화되었습니다. 실험 결과, 모델은 실내 및 실외 환경 모두에서 기존 접근 방식의 한계를 극복하고, 모바일 장치에서 실시간으로 효과적인 성능을 제공함을 증명했습니다. 이는 특히 모바일 기기로 Vlog 등의 콘텐츠를 제작하는 사용자를 위한 강력한 기능입니다.



### Piano Transcription by Hierarchical Language Modeling with Pretrained Roll-based Encoders (https://arxiv.org/abs/2501.03038)
Comments:
          Accepted by ICASSP 2025

- **What's New**: 본 논문에서는 Automatic Music Transcription (AMT)을 위한 하이브리드 방법을 제안합니다. 이 방법은 사전 훈련된 롤 기반 인코더와 언어 모델(이하 LM) 디코더를 결합하여 두 가지 접근 방식의 강점을 활용합니다. 또한 계층적 예측 전략을 사용하여 먼저 온셋과 피치를 예측하고, 그 다음 속도를 예측하며 마지막으로 오프셋을 예측하는 방식으로 계산 비용을 줄입니다.

- **Technical Details**: 롤 기반 AMT 시스템은 오디오 파형을 시간-주파수 도메인으로 변환하여 피아노 롤을 예측하는 구조로 되어 있습니다. 반면 LM 기반 AMT 시스템은 음악 전사를 시퀀스 생성 작업으로 간주하며, 입력 오디오를 인코딩하여 노트 이벤트 시퀀스를 생성합니다. 이 논문은 두 접근 방식을 결합하여 단순히 롤 기반 변환을 통해 예측하는 것이 아니라, 계층적으로 정보의 흐름을 구성하여 예측의 효율성을 높입니다.

- **Performance Highlights**: 제안된 방법은 전통적인 피아노 롤 출력에 비해 0.01 및 0.022의 F1 점수를 기록하며, 성능 향상을 위한 플러그인으로서의 가능성을 보여줍니다. 특히, 인코더 선택이 전체 성능에 더 큰 영향을 미치는 것으로 확인되었으며, LM 크기를 증가시켜도 성능이 개선되지 않는 결과를 통해 언어 모델의 크기보다는 인코더의 선택이 더 중요하음을 강조합니다.



### Quantization Meets Reasoning: Exploring LLM Low-Bit Quantization Degradation for Mathematical Reasoning (https://arxiv.org/abs/2501.03035)
Comments:
          4 pages

- **What's New**: 본 연구는 대형 언어 모델(LLM)의 수학적 추론 작업에 대한 양자화가 미치는 영향을 체계적으로 평가했습니다. 우리는 다양한 양자화 방법의 단계별 출력에 대한 정량적 분석과 특정 능력 차원을 질적으로 평가하는 다차원 평가 프레임워크를 도입합니다. 양자화가 수치 계산 능력과 추론 계획 능력에 미치는 차별적인 영향을 밝혀내어, 양자화 모델에서 성능 저하가 발생하는 주요 분야를 확인했습니다.

- **Technical Details**: 연구에서는 GPTQ와 AWQ와 같은 가중치 전용 양자화 및 SmoothQuant와 같은 가중치-활성화 양자화를 조사했습니다. 각 양자화 방법의 정확도를 평가하기 위해 다양한 설정에서 실험을 진행했으며, LoRA를 활용하여 양자화 모델의 효과적인 미세 조정을 수행했습니다. PRM800K 데이터세트를 사용하여 모델이 차별적인 추론 단계를 학습하고, 명확한 경계를 가지고 추론할 수 있도록 특별한 토큰을 도입했습니다.

- **Performance Highlights**: MATH 데이터세트에서의 실험 결과, 양자화된 모델은 전반적으로 성능 저하를 보였습니다. 모든 양자화 방법이 성능 손실을 초래했으며, 특히 SmoothQuant 방식이 가장 작은 성능 저하를 나타냈습니다. 양자화 모델에서의 성능 저하는 주로 문제의 오해와 논리적 오류와 같은 여러 요인으로 구분돼 분석되었습니다.



### Putnam's Critical and Explanatory Tendencies Interpreted from a Machine Learning Perspectiv (https://arxiv.org/abs/2501.03026)
Comments:
          9 pages

- **What's New**: 본 논문은 과학 철학에서 정상적(normal) 및 비범한(extraordinary) 과학 이론 선택의 의미를 탐구합니다. 또한 머신 러닝 모델의 출현이 현재의 논의에 미치는 영향을 논의하며, 기존의 이론 선택에 대한 이해를 새롭게 형성할 수 있는 가능성을 제시합니다. 저자는 Putnam의 비판적(critical) 및 설명적(explanatory) 경향 구분이야말로 현재의 과학 논의를 재구성하는 데 필요한 핵심 요소라고 주장합니다.

- **Technical Details**: 논문은 Putnam의 비판적 및 설명적 경향 구분을 통해 발생하는 여러 움직임을 재구성합니다. 이 과정에서 저자는 머신 러닝 해석을 통해 이를 새로운 시각에서 개념화하고자 하며, 이러한 경향의 양면성(biconditional necessity)을 강조합니다. 머신 러닝의 이론 선택 과정에서의 역할을 탐색함으로써, 이론의 진화에 대한 기존 관점을 재고할 수 있는 기초를 마련합니다.

- **Performance Highlights**: 저자는 머신 러닝 모델이 과학 이론 선택의 논쟁에 기여할 수 있는 방법을 제시하며, 이론 선택에 대한 기존의 이해를 수정할 수 있는 포괄적인 시각을 제공합니다. 본 논문은 과학 철학과 머신 러닝의 접목을 통해 이론 선택에서 혁신적인 접근을 가져올 수 있는 가능성을 드러내며, 향후 연구에 대한 새로운 방향성을 제시합니다.



### Quality Estimation based Feedback Training for Improving Pronoun Translation (https://arxiv.org/abs/2501.03008)
- **What's New**: ProNMT는 기계 번역 시스템의 대명사 번역 품질을 향상시키기 위한 새로운 프레임워크입니다. 이는 Quality Estimation(QE) 모델과 대명사 생성 가능성 기반 피드백 메커니즘을 통합하여 기존의 NMT 모델을 반복적으로 세밀하게 조정할 수 있습니다. ProNMT는 대규모 인간 주석 없이도 번역 품질을 개선하도록 설계되었습니다.

- **Technical Details**: ProNMT는 기본적으로 두 가지 주요 개념에 기반합니다. 첫째, 품질 추정(QE)을 통해 출력된 대명사의 번역 품질을 평가합니다. 둘째, 대명사 생성 가능성 기반 피드백을 통해 번역 시 올바른 대명사를 생성할 확률에 따라 피드백을 제공합니다. 이러한 메커니즘을 통해 번역 품질과 대명사 번역의 적절성을 동시에 향상시키는 것을 목표로 합니다.

- **Performance Highlights**: 실험 결과, ProNMT는 대명사 번역의 정확성과 일반 번역 품질 모두에서 기존의 방법들에 비해 상당한 개선을 보였습니다. 이는 문서 수준의 기계 번역(MT) 시스템이 대명사 같은 맥락 의존적 요소를 처리하는 데 있어 효과적으로 작용함을 보여줍니다. ProNMT는 기계 번역 품질 향상을 위한 효율적이고 확장 가능한 접근법을 제공합니다.



### GLFC: Unified Global-Local Feature and Contrast Learning with Mamba-Enhanced UNet for Synthetic CT Generation from CBC (https://arxiv.org/abs/2501.02992)
Comments:
          Accepted by ISBI2025

- **What's New**: 이 논문에서는 Cone Beam Computed Tomography (CBCT)로부터 합성 Computed Tomography (sCT) 이미지를 생성하기 위한 새로운 프레임워크인 Global-Local Feature and Contrast learning (GLFC)을 제안합니다. 기존의 방법들이 global과 local feature 및 contrast를 효과적으로 캡처하는 데 어려움을 겪고 있는 점을 해결하고자 하였습니다. 우리는 Mamba-Enhanced UNet (MEUNet)와 Multiple Contrast Loss (MCL)를 도입하여 CBCT의 저화질 이미지를 개선하고 sCT 생성의 품질을 높였습니다.

- **Technical Details**: GLFC 프레임워크는 두 개의 down-sampling 레이어를 가진 MEUNet와 MCL을 기반으로 구성됩니다. MEUNet은 Mamba 블록을 UNet의 skip 연결에 통합하여 local feature와 long-range dependency를 효과적으로 학습합니다. MCL은 여러 intensity windows를 사용하여 soft tissues 및 bone의 품질을 높이는 방식으로 설계되었습니다.

- **Performance Highlights**: SynthRAD2023 데이터셋에서의 실험 결과, GLFC 프레임워크는 sCT의 SSIM을 77.91%에서 91.50%로 향상시켰습니다. 이는 기존의 여러 최첨단 딥러닝 방법보다 뛰어난 성능을 보여주었습니다. GLFC는 이미지의 품질을 크게 개선하여 의료 영상의 새로운 가능성을 제공할 것으로 기대됩니다.



### A Bio-Inspired Research Paradigm of Collision Perception Neurons Enabling Neuro-Robotic Integration: The LGMD Cas (https://arxiv.org/abs/2501.02982)
- **What's New**: 이번 논문에서는 곤충의 시각 시스템, 특히 메뚜기의 시각 뇌에서 발견된 충돌 선택적 신경세포인 LGMD(Lobula Giant Movement Detectors)에 대한 최신 연구 발전을 다룹니다. LGMD는 접근하는 물체에 특정하게 반응하여 촉각적으로 충돌을 감지하는 능력으로 로봇 공학 및 인공 충돌 탐지 시스템 개발에 매력적인 모델이 되고 있습니다. 또한, 생물학적으로 그럴듯한 연구 패러다임이 소개되어 신경과학의 통찰력이 실제 응용 프로그램을 통해 검증되고 발전하는 방식이 강조됩니다.

- **Technical Details**: LGMD 신경세포는 메뚜기의 시각 뇌에서 매우 중요한 역할을 하며, 두 개의 주요 유형인 LGMD1과 LGMD2가 존재합니다. LGMD1은 주로 접근하는 물체에 가장 강한 반응을 보이며 다양한 움직임 유형들을 구분할 수 있는 능력을 갖추고 있습니다. 이들 신경세포의 구조, 시냅스 연결, 그리고 세포 내 위치에 대한 이해가 넓어짐에 따라, 생물학적으로 영감을 받은 메커니즘이 로봇 공학 성능 향상에 기여하고 있습니다.

- **Performance Highlights**: LGMD 모델은 모바일 로봇 시스템에서 무사 충돌내비게이션에 큰 기여를 하였으며, 최근 몇 년 동안  실시간 환경에서 실험적으로 성공적으로 적용되었습니다. 이러한 연구 패러다임은 다양한 생물학적 동역학과 기술적 신경망 간의 상호작용을 깊이 이해하는 데 기여하고 있습니다. 결과적으로, 메뚜기 기반 LGMD 시스템은 신경과학, 컴퓨테이셔널 모델링 및 로봇 공학 간의 연속적인 협력을 통해 전체 진화하는 연구 분야로 자리잡고 있습니다.



### CONTINUUM: Detecting APT Attacks through Spatial-Temporal Graph Neural Networks (https://arxiv.org/abs/2501.02981)
Comments:
          31 pages

- **What's New**: 이번 연구에서는 고급 지속 위협(Advanced Persistent Threats, APTs) 탐지를 위한 새로운 침입 탐지 시스템(IDS)을 제안합니다. 이 시스템은 Spatio-Temporal Graph Neural Network Autoencoder를 기반으로 하여 네트워크에서의 복잡한 상호작용을 분석합니다. 기존 GNN 기반 솔루션의 높은 허위 긍정 비율과 자원 소모 문제를 해결하는 것을 목표로 하며, 특히 APT의 다단계를 효과적으로 식별할 수 있도록 설계되었습니다.

- **Technical Details**: 제안된 IDS는 공간적 정보(spatial information)와 시간적 정보(temporal information)를 동시에 활용하여 네트워크 내 개체 간의 상호작용을 이해하고, 그래프의 발전 과정을 추적합니다. 이 시스템은 Graph Attention Network (GAT) 기반 Autoencoder와 Gated Recurrent Unit (GRU) 게이트를 사용하여, 시간에 따라 변화하는 데이터의 흐름을 효과적으로 처리합니다. 이로 인해 APT의 탐지가 보다 정확하고 효율적으로 이루어질 수 있습니다.

- **Performance Highlights**: 평가 결과, 제안한 시스템은 기존 방법에 비해 낮은 허위 긍정 비율과 최적화된 자원 사용을 통해 APT를 효과적으로 탐지하는 것으로 나타났습니다. 이러한 결과는 시공간(spatio-temporal) 분석과 연합 학습(federated learning)의 조합이 사이버 보안 방어에 있어 그 가능성을 증명합니다. 궁극적으로 이 연구는 APT 탐지 기술의 발전에 기여할 것으로 기대됩니다.



### CAMP: Collaborative Attention Model with Profiles for Vehicle Routing Problems (https://arxiv.org/abs/2501.02977)
Comments:
          Accepted at AAMAS 2025

- **What's New**: 이번 연구에서는 협업 주의 모델(CAMP)을 제안하여 프로파일 차량 경로 문제(Profiled Vehicle Routing Problem, PVRP)를 해결하는 새로운 접근 방식이 도입되었습니다. CAMP는 다중 에이전트 강화 학습(multi-agent reinforcement learning)을 통해 효율적인 솔버를 학습하며, 각 차량 프로파일에 맞는 고객 임베딩을 병렬로 통합하는 특수한 주의 기반 인코더 아키텍처를 사용합니다. 이를 통해 프로파일 임베딩에 대한 협업 의사 결정이 가능해졌습니다.

- **Technical Details**: CAMP는 각 차량의 프로파일에 맞춰 고객의 임베딩을 통합하고, 에이전트 간의 통신 레이어를 설계함으로써 협업적 의사 결정을 수행합니다. 또한 배치 포인터 메커니즘을 활용하여 프로파일 임베딩에 주목하며, 다음 행동의 가능성을 평가합니다. PVRP의 두 가지 변형—선호가 있는 PVRP(PVRP-P)와 지역 제약이 있는 PVRP(PVRP-ZC)—를 통해 성능을 평가하였습니다.

- **Performance Highlights**: CAMP는 전통적인 방법과 최신 신경망 다중 에이전트 모델과 비교하여 솔루션 품질 및 계산 효율성 측면에서 경쟁력 있는 결과를 도출하였습니다. 이러한 성과는 CAMP가 복잡한 경로 문제를 실시간으로 해결하는 데 유용한 도구임을 시사합니다. 이 연구의 코드는 공개되어 있으므로 연구자와 실무자들에게 접근할 수 있는 기회를 제공합니다.



### Fuzzy Granule Density-Based Outlier Detection with Multi-Scale Granular Balls (https://arxiv.org/abs/2501.02975)
- **What's New**: 이 논문에서는 다양한 유형의 이상치를 식별하기 위해 퍼지 거칠기 집합(fuzzy rough sets) 기반의 다중 규모 이상치 탐지 방법을 제안합니다. 기존의 비지도 학습 방법들은 특정 이상치 유형에 맞춰 설계되었지만, 실제 데이터는 다양한 이상치가 얽혀 있는 경우가 많습니다. 본 연구는 퍼지 거칠기 집합을 활용하여 지역 이상치를 효과적으로 탐지할 수 있는 새로운 방법을 도입합니다.

- **Technical Details**: 제안한 방법은 상대적인 퍼지 집합 밀도를 통합하여 지역 이상치 탐지 능력을 강화합니다. 또한, 구형-구 기반의 다중 규모 뷰 생성 방법을 통해 다양한 수준의 밀도에서 군 집합(group outliers)을 식별합니다. 마지막으로, 신뢰할 수 있는 이상치와 인라이너를 기반으로 훈련된 가중 지원 벡터 머신(weighted support vector machine)을 활용하여 탐지 성능을 더욱 개선합니다.

- **Performance Highlights**: 제안된 이상치 탐지 방법은 인공 데이터 및 UCI 데이터 세트에서 광범위한 실험을 통해 기존의 최신 방법들보다 최소 8.48% 향상된 결과를 보였습니다. 연구의 결과는 다양한 이상치 유형을 효과적으로 탐지할 수 있음을 시사하며, 이는 실무에서도 중요한 기여를 할 것으로 기대됩니다.



### Proof-of-Data: A Consensus Protocol for Collaborative Intelligenc (https://arxiv.org/abs/2501.02971)
- **What's New**: 본 논문은 탈중앙화된 연합 학습(decentralized federated learning)에서 발생하는 새로운 도전 과제에 대해 다루고 있습니다. 특히, 중앙 조정자로서의 역할이 없는 환경에서 모든 참여 노드가 공정하게 보상을 받을 수 있는 방법과 모델 학습을 올바르게 진행하는 방법을 제안합니다. 이를 위해 블록체인 기반의 Byzantine fault-tolerant 연합 학습 프레임워크를 제안하며, Proof-of-Data (PoD) 합의 프로토콜이 이를 뒷받침합니다.

- **Technical Details**: PoD 합의 프로토콜은 비동기적 사회적 규모의 작업 증명(Proof-of-Work, PoW) 스타일의 학습과 에포크 기반의 PBFT(Practical Byzantine Fault Tolerance) 스타일의 합의를 결합하여, 실질적으로 견고하면서도 이론적으로 결함이 있는 PoW 기반 시스템의 끝남(Consensus finality) 문제를 해결합니다. 또한, 참가 노드의 데이터 기여도에 기반하여 보상을 분배하는 메커니즘을 디자인하여, 데이터 조작에 의한 잘못된 보상 주장을 완화하는 것을 목표로 합니다.

- **Performance Highlights**: 제안된 PoD 프레임워크는 중앙 집중식 연합 학습(centralized federated learning) 모델과 유사한 성능을 보이며, 신뢰할 수 있는 합의와 공정한 보상 할당을 달성합니다. 실험 결과, 잘못된 공격에 대한 내성과 더불어 1/3의 결함 허용 비율을 가지고 우수한 성능을 나타내었으며, 다양한 시간 불변 및 시간 변화 데이터 세트에 대한 평가가 진행되었습니다.



### Socratic Questioning: Learn to Self-guide Multimodal Reasoning in the Wild (https://arxiv.org/abs/2501.02964)
- **What's New**: 이번 논문에서는 Complex Visual Reasoning(복잡 시각적 추론)을 위한 혁신적인 Multi-Round Training(다단계 훈련)과 Reasoning(추론) 프레임워크인 Socratic Questioning(SQ)을 제안합니다. SQ는 Multimodal Large Language Models(다중 모드 대형 언어 모델)에서 Hallucinations(환각)을 줄이고, 세부적인 이미지를 묘사하는 능력을 향상시키기 위한 Self-Questioning(자기 질문)의 접근 방식을 채택합니다. 이 연구의 중요한 기여로는, SQ 방법이 Hallucination 점수를 31.2% 개선시켰으며, 신규 multimodal mini-dataset인 CapQA를 사용하여 평가되었다는 점입니다.

- **Technical Details**: SQ는 Four-Step(4단계) 절차를 통해 이루어지며, Self-Ask(자기 질문)에서 시작해 Self-Answer(자기 답변), Consolidate & Organize(통합 및 정리), Summarize & Condense(요약 및 축소)로 이어집니다. 이 과정에서 MLLMs은 텍스트와 이미지를 활용하여 필요한 세부 정보를 시각적으로 검색하고, 이를 바탕으로 문제를 깊이 생각하게 됩니다. SQ는 Chain of Thought와 Visual Instruction Tuning 접근 방식을 통합하며, 이를 통해 시각적 정보에 대한 무시를 피하고, 높은 신뢰성을 제공하는 모델로 발전하게 됩니다.

- **Performance Highlights**: Socratic Questioning(SQ)은 Hallucinations를 효과적으로 줄이는 데 성공하며, 이는 추가적인 복잡한 아키텍처나 대량의 데이터 처리 없이 이루어집니다. 우리의 실험 결과는 SQ가 다양한 Visual Reasoning(시각적 추론) 및 Q&A(질문 및 답변) 벤치마크의 제로샷 성능을 극대화하는 데 뛰어난 능력을 발휘한다는 것을 보여줍니다. 또한, 우리는 GPT-4v를 활용하여 생성된 데이터의 품질을 평가하며 SQ의 유효성을 입증하였습니다.



### Key-value memory in the brain (https://arxiv.org/abs/2501.02950)
- **What's New**: 이 논문에서는 심리학과 신경과학에서의 메모리 이론을 재고하고, 최신 기계학습 시스템에서의 key-value 메모리 시스템의 중요성을 강조합니다. 기존의 유사성 기반 기억 모델과는 달리, key-value 메모리는 저장과 검색을 위한 서로 다른 표현을 인식하여, 저장의 충실도와 검색의 판별력을 동시에 최적화할 수 있습니다. 이러한 새로운 접근은 자연 및 인공 지능의 구간을 연결하는 데 중요한 의미를 갖습니다.

- **Technical Details**: 기술적 측면에서 key-value 메모리는 메모리 주소를 나타내는 keys와 메모리 내용을 저장하는 values 간의 구분을 통해 구성됩니다. 각 입력은 key 벡터와 value 벡터로 구성되며, 이러한 두 표현은 'associator' 행렬을 통해 메모리에서 연결됩니다. 또한, covariance matrix와 같은 기법을 사용하여 두 가지 객체 간의 관계를 저장하는 heteroassociative memory로서의 특성을 설명합니다.

- **Performance Highlights**: 이 논문에서는 key-value 메모리 시스템이 방대한 양의 정보에서 보다 효과적으로 검색할 수 있도록 도와주는 다양한 사례와 실험을 제시합니다. 예를 들어, 이 시스템은 기계학습에서의 transformer와 같은 모델을 기반으로 구축된 효율적인 메모리 구조를 통해, 정보 검색의 성능을 향상시키는 데 기여할 수 있습니다. 이는 인공지능의 발전뿐만 아니라 인간의 기억 메커니즘에 대한 이해에도 중요한 시사점을 제공합니다.



### Label-free Concept Based Multiple Instance Learning for Gigapixel Histopathology (https://arxiv.org/abs/2501.02922)
- **What's New**: 본 연구에서는 기존의 MIL 방식의 한계를 극복하기 위해 Concept MIL을 제안합니다. 이 모델은 이미지에서 병리학적 개념을 직접 예측하고, 병리학적 개념의 영향을 추적하여 예측에 대한 해석을 제공합니다. 이를 통해서 수작업으로 개념 라벨링을 할 필요가 없어져 다양한 질병에 쉽게 적용할 수 있는 유연성을 제공합니다.

- **Technical Details**: Concept MIL 모델은 고해상도 WSI 데이터를 분석하기 위해 최근 발전된 vision-language 모델을 활용합니다. 모델은 WSI의 상위 K 패치에서 식별된 개념들을 선형 결합하여 예측합니다. 이를 통해 각 개념이 예측에 미치는 영향을 명확하게 설명할 수 있으며, 모든 과정에서 수작업 주석을 제거하여 경제적 효율성을 높입니다.

- **Performance Highlights**: Concept MIL 모델은 Camelyon16과 PANDA 데이터셋에서 AUC 및 정확도를 0.9 이상으로 기록하며, 최신 모델들과 동등한 성능을 보여줍니다. 사용자 연구 결과, 모델이 식별한 개념들이 병리학자들이 사용하는 개념과 일치하여, 인간이 해석 가능한 WSI 분류의 유망한 전략으로 자리잡을 것입니다.



### Unsupervised Tomato Split Anomaly Detection using Hyperspectral Imaging and Variational Autoencoders (https://arxiv.org/abs/2501.02921)
Comments:
          CVPPA Workshop

- **What's New**: 이번 연구에서는 온실에서의 토마토 재배에서 발생하는 균열 이상 현상을 탐지하기 위한 비지도 학습 접근 방식을 제안합니다. 이를 위해 맞춤형 Variational Autoencoder (VAE)와 하이퍼스펙트럼 입력을 사용하여 토마토의 균열을 효과적으로 탐지할 수 있는 방법을 개발했습니다. 또한 530nm - 550nm 범위의 파장이 토마토 균열 탐지에 효과적임을 입증하였습니다.

- **Technical Details**: 하이퍼스펙트럼 이미징(Hyperspectral Imaging, HSI)의 사용은 여러 파장을 통해 물체의 스펙트럼 특성을 관찰할 수 있게 해줍니다. 데이터 수집을 위해 Specim의 FX10e 하이퍼스펙트럼 카메라를 사용하였고, 400nm에서 1000nm 범위의 파장을 스캔할 수 있습니다. 연구 과정에서 74개의 토마토 샘플을 수집하였으며, 이들을 공간적으로 증강하여 정상 및 비정상 샘플을 생성하였습니다.

- **Performance Highlights**: 비지도 학습 기반의 VAE를 사용하여 재구성 손실을 분석함으로써 균열 이상 탐지의 성능을 크게 향상시켰습니다. 이 접근 방식은 결정 경계를 학습하여 이상을 탐지할 수 있으며, 정상 데이터에 대해 훈련된 모델이 비정상 데이터를 정확히 재구성하는 데 어려움을 겪는 원리를 사용합니다. 이러한 방식은 조기 탐지를 통한 생산물의 품질 향상에 기여할 것으로 기대됩니다.



### Skillful High-Resolution Ensemble Precipitation Forecasting with an Integrated Deep Learning Framework (https://arxiv.org/abs/2501.02905)
- **What's New**: 본 연구에서는 하이레졸루션 강수 예측을 위한 물리 기반의 딥러닝 프레임워크를 제안합니다. 이 프레임워크는 결정론적(deterministic) 모델과 확률론적(probabilistic) 모델을 결합하여 작용됩니다. 이 방식은 특히 중간에서 강한 강수에 대해 성능을 향상시키며, 집중적인 강수 이벤트를 포착하는 데 뛰어난 능력을 보입니다.

- **Technical Details**: 우리의 모델은 3D SwinTransformer에 기반한 결정론적 모델과, 조건부 확산(conditional diffusion)을 사용하는 확률론적 모델로 구성됩니다. 이를 통해 각 예측의 불확실성을 반영하며, 고해상도(0.05°) 강수 예측값을 생성할 수 있습니다. 이 모델은 고해상도 데이터셋(ERA5 및 CMPA)을 이용하여 훈련됩니다.

- **Performance Highlights**: 모델은 균형 잡인 결과를 제공하며, 극단적인 강수 이벤트를 포착하는 데 있어 뛰어난 성능을 보여줍니다. 사례 연구를 통해, 모델의 출력은 관측된 강수 분포와 유사하게 일치하여 예측 정확도를 높였습니다. 5일간의 실시간 강수 예측에서도 매우 좋은 성능(CSI 점수)을 나타내었습니다.



### Explaining Humour Style Classifications: An XAI Approach to Understanding Computational Humour Analysis (https://arxiv.org/abs/2501.02891)
- **What's New**: 본 논문은 유머 스타일 분류를 위한 설명 가능한 AI(XAI) 프레임워크를 제시하고 있습니다. 이는 기존 연구를 기반으로 하며, 언어적, 감정적, 의미적 특징이 유머 스타일 분류 결정에 어떻게 기여하는지를 분석합니다. 이러한 분석을 통해 각 유머 스타일이 어떻게 특성화되고 잘못 분류되는지에 대한 명확한 패턴을 발견하였습니다.

- **Technical Details**: 이 연구는 ALI+XGBoost 모델을 활용하여 XAI 기법을 적용합니다. XAI는 LIME(Local Interpretable Model-Agnostic Explanations) 및 SHAP(Shapley Additive Explanations) 기법을 통해 모델의 예측 결정과 주요 특징들을 해석할 수 있도록 지원하며, 감정의 모호성, 문맥의 오해, 목표 식별 등 모델 결정에 영향을 미치는 주요 요인을 규명합니다.

- **Performance Highlights**: 연구 결과, XAI 프레임워크는 유머 스타일 분류의 투명성을 높이는 데 기여하며, 연구자들에게 유머의 역할을 더욱 깊이 있게 조사할 수 있는 실제적인 통찰을 제공합니다. 특히, 감정적 특징과 언어적 요소가 모델 예측에 미치는 영향을 상세히 분석하여, 정신 건강, 콘텐츠 조정 및 디지털 인문학 연구에 대한 실용적인 응용 가능성을 제시합니다.



### IIMedGPT: Promoting Large Language Model Capabilities of Medical Tasks by Efficient Human Preference Alignmen (https://arxiv.org/abs/2501.02869)
- **What's New**: 이 논문에서는 IIMedGPT라는 새로운 의학 언어 모델을 소개하며, CMedINS라고 불리는 의학 지침 데이터셋을 통해 LLM의 성능을 개선하는 방법을 제시합니다. 특히, Direct Preference Optimization(DPO) 방법을 통해 모델이 사용자 지침에 맞춰 강화되는 것을 목표로 하였습니다. 이 모델은 기존 의학 모델과 비교하여 우수한 성능을 보였으며, 코드와 모델 체크포인트는 논문 수락 시 공개될 예정입니다.

- **Technical Details**: IIMedGPT는 감독된 미세 조정(supervised fine-tuning) 및 직접 정책 최적화(Direct Policy Optimization)라는 두 단계의 훈련 방식으로 개발되었습니다. CMedINS는 실제 의료 작업에서 파생된 여섯 가지 의료 지침을 포함하여 220,000개의 쌍의 의료 기록을 수집하여 구성되었습니다. 또한, 모델 훈련 시 효과적인 일반화 능력을 확보하기 위해 의료 대화 데이터셋과 일반 능력 데이터셋을 혼합하여 사용하였습니다.

- **Performance Highlights**: IIMedGPT는 성능 평가에서 GPT-4 및 인간 전문가와 비교하여 세 가지 능력 차원과 아홉 가지 특정 능력에서 우수한 결과를 기록하였습니다. 특히, 모델은 적은 양의 훈련 데이터에도 불구하고 기존 오픈 소스 전통 중국 의학 LLM보다 향상된 성능을 발휘했습니다. 이는 의료 지침 및 대화 처리 능력을 극대화하기 위해 신중하게 수집된 데이터의 효과를 보여주는 결과입니다.



### Enhanced Rooftop Solar Panel Detection by Efficiently Aggregating Local Features (https://arxiv.org/abs/2501.02840)
Comments:
          Accepted at CODS-COMAD 2024, December, 2024, Jodhpur, India (this https URL)

- **What's New**: 이 논문에서는 위성 이미지(satellite images)를 사용한 향상된 합성곱 신경망(CNN) 기반의 옥상 태양광 발전 패널(detection of rooftop solar photovoltaic panels) 탐지 방법을 제안합니다. 사전 학습된 CNN 모델을 활용하여 옥상의 로컬 컨볼루션 특징(local convolutional features)을 추출하고, 이를 Vectors of Locally Aggregated Descriptors (VLAD) 기법을 통해 결합하여 옥상 수준의 글로벌 특징(global features)을 획득합니다. 이러한 특징을 전통적인 머신러닝(Machine Learning) 모델에 적용하여 태양광 패널이 있는 이미지와 없는 이미지를 분류합니다.

- **Technical Details**: 본 연구에서 사용된 데이터셋(dataset)에 대해 제안한 방법은 세 개의 도시에서 옥상-태양광 발전(PV) 분류 점수가 0.9를 초과하는 결과를 도출하였습니다. 각 특징 추출기(feature extractor) 네트워크를 평가하여 전통적인 머신러닝 모델을 학습시킵니다. 또한, 제한된 레이블 데이터(labelled data)가 있는 새로운 도시 또는 지역에서 이전에 학습된 모델을 효율적으로 활용할 수 있는 3단계 접근 방식(3-phase approach)을 제안합니다.

- **Performance Highlights**: 제안된 3단계 접근 방식은 다수의 도시에서 옥상 태양광 발전 탐지 작업의 유효성을 잘 보여줍니다. 모든 실험에서 높은 정확도를 기록하였으며, 이는 향후 옥상 태양광 패널 탐지 분야의 연구에 기여할 것입니다. 또한 다양한 특성 추출 네트워크로부터 얻어진 결과들은 모델의 내구성과 범용성을 입증하였습니다.



### Forward Once for All: Structural Parameterized Adaptation for Efficient Cloud-coordinated On-device Recommendation (https://arxiv.org/abs/2501.02837)
Comments:
          Accepted by KDD 2025

- **What's New**: 본 논문은 클라우드 중심의 추천 시스템에서 발생할 수 있는 네트워크 대역폭 요구 사항 및 개인정보 보호 위험을 줄이기 위해, 기존의 맞춤형 모델 아키텍처의 중요성을 강조하며 다이나믹한 디바이스 전용 네트워크 구성을 제안하는 Forward-OFA를 소개합니다. 이른바 On-device Recommendation(온디바이스 추천)으로, 사용자 데이터를 클라우드로 전송하는 대신 현지에서 추천의 순위를 재조정하여 네트워크 부하를 줄이고 제품 추천의 정확성을 개선할 수 있습니다.

- **Technical Details**: Forward-OFA는 구조 컨트롤러를 통해 각 디바이스에 필요한 블록의 조합을 선택적으로 결정하는 구조를 가지고 있습니다. 훈련 과정 중 이 조립된 이질적 구조는 공동 최적화되며, 구동 시 각 항목이 이질적 기울기를 수신하지 못하도록 파라미터를 설정합니다. 이 방법은 Gradient Conflict(기울기 충돌)를 피하고, 실시간 행동과 조립된 네트워크의 파라미터 간의 구조적 매핑을 통해 적응을 지원합니다.

- **Performance Highlights**: 다양한 실제 데이터셋을 이용한 실험 결과, Forward-OFA가 효과적이고 효율적인 성능을 발휘함을 밝힙니다. 특히, 사용자 개인정보를 보호하고 여러 디바이스에서의 최적화된 적응을 통해 큰 성과를 보였으며, 추가적인 네트워크 시각화 및 사례 연구를 통해 Forward-OFA의 작동 및 다른 네트워크와의 영향력을 명확히 분석했습니다.



### Samba-ASR: State-Of-The-Art Speech Recognition Leveraging Structured State-Space Models (https://arxiv.org/abs/2501.02832)
- **What's New**: Samba ASR는 Mamba 아키텍처를 이용한 최신의 자동 음성 인식(ASR) 모델로, 전통적인 transformer 모델의 단점을 극복하며 뛰어난 성능을 발휘한다. 특히, 기존의 transformer-based ASR 모델의 문제점인 긴 입력 시퀀스에서의 복잡도를 줄이고, 긴 거리의 종속성을 효과적으로 처리할 수 있는 구조화된 상태 공간 모델(SSM)을 기반으로 한다. Samba ASR의 도입은 경량화된 ASR 시스템의 새로운 기준을 제시한다.

- **Technical Details**: Samba ASR는 상태 공간 모델(SSM)을 활용해 음성 인식을 수행하며, Mamba 아키텍처를 통해 효율적인 상태 공간 동역학을 구현하고 있다. Mamba는 선택적인 회귀(selective recurrence)와 하드웨어 인식 최적화(hardware-aware optimizations)를 통해 복잡한 입력에 적절하게 적응할 수 있도록 한다. 이를 통해, 입력 의존적(state space equations) 매개변수를 도입하고, 더 작은 상태 표현으로 컨텍스트(context)를 압축하는 능력을 지향한다.

- **Performance Highlights**: Samba ASR는 Gigaspeech와 SPGISpeech 등 주요 ASR 벤치마크에서 최고의 성능을 기록하였다. 이 모델은 훈련속도와 추론 지연을 줄이면서도 높은 정확도를 유지하며, 노이즈가 있거나 자발적인 발화와 같은 어려운 조건에서도 뛰어난 성능을 보여준다. 이러한 결과는 Samba ASR가 다양한 ASR 시스템에 대해 스케일 가능하고 강력한 솔루션임을 입증한다.



### RDD4D: 4D Attention-Guided Road Damage Detection And Classification (https://arxiv.org/abs/2501.02822)
- **What's New**: 이번 논문에서는 도로 손상 탐지 및 평가를 위한 새로운 데이터셋인 Diverse Road Damage Dataset (DRDD)을 소개합니다. 이 데이터셋은 다양한 손상 유형을 개별 이미지로 포착함으로써 기존의 데이터셋에서의 중요 공백을 해결하는 데 기여합니다. 또한, Attention4D 블록을 활용한 RDD4D 모델을 제안하여 다양한 스케일에서의 피쳐 정제를 향상시킵니다.

- **Technical Details**: RDD4D 모델은 Attention4D 모듈을 통해 피쳐 맵을 처리하여 위치 인코딩(positional encoding)과 'Talking Head' 요소를 결합한 주의 메커니즘을 사용합니다. 이 모듈은 지역적(local) 및 전역적(global) 문맥 정보를 포착하여 도로 손상을 효과적으로 탐지합니다. 본 연구는 다양한 최신 모델들과의 비교 실험을 통해 RDD4D의 우수성을 입증합니다.

- **Performance Highlights**: 우리 모델은 큰 크기의 도로 균열 탐지에서 0.458의 평균 정밀도(Average Precision, AP)를 기록하며 우수한 성능을 보였습니다. 전체 AP는 0.445로 경쟁력 있는 성능을 유지하였습니다. 또한, CrackTinyNet 데이터셋에서도 성능이 약 0.21 증가하여 향상된 결과를 보여주었습니다.



### InpDiffusion: Image Inpainting Localization via Conditional Diffusion Models (https://arxiv.org/abs/2501.02816)
- **What's New**: 이번 논문에서는 Image Inpainting Localization (IIL) 문제를 다루기 위해 새로운 프레임워크인 InpDiffusion을 제안합니다. 기존의 IIL 방법들이 과신(overconfidence)으로 인해 부정확한 예측을 하거나 섬세한 변조 경계를 탐지하는 데 어려움을 겪는 반면, InpDiffusion은 조건부 마스크 생성(condition mask generation) 작업으로 IIL을 재구성합니다. 이 방법은 디퓨전 모델을 활용하여 점진적으로 예측을 개선하고, 엣지 조건(edge conditions)과 새로운 엣지 감독(edge supervision) 전략을 통해 세부 경계를 강화합니다.

- **Technical Details**: InpDiffusion은 Adaptive Conditional Network (ACN) 기능을 갖추고 있어, 이미지에서 의미적 특징과 엣지 특징을 동시에 추출합니다. Dual-stream Multi-scale Feature Extractor (DMFE)를 통해 다중 스케일 특징을 효과적으로 추출하여, 세뇌와 엣지 조건을 바탕으로 자세한 표현을 강화합니다. 각 디노이징 단계에서 엣지 감독을 적용하여 각 샘플링 단계에서의 과도한 무작위성 문제를 해결하고, 이미지의 배경과 변조된 영역 간의 차이를 명확히 합니다.

- **Performance Highlights**: 실험 결과, InpDiffusion은 기존의 최첨단 IIL 방법들과 비교할 때 크게 향상된 성능을 보여줍니다. 다양한 도전적인 데이터셋에서 광범위한 실험을 통해, 이 모델은 뛰어난 일반화 능력과 강인성을 입증하였습니다. 논문은 InpDiffusion이 이미지 인페인팅 작업에서 더욱 정확하고 신뢰할 수 있는 결과를 제공함을 강조합니다.



### Enhancing Lifelong Multi-Agent Path Finding with Cache Mechanism (https://arxiv.org/abs/2501.02803)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2403.13421

- **What's New**: 이 논문에서는 Lifelong MAPF with Cache Mechanism (L-MAPF-CM)이라는 새로운 메커니즘을 도입하여, 여러 로봇이 작업을 수행하는 자동화 창고 환경에 적합한 경로 계획을 개선합니다. 기존의 수학적 모델을 넘어, 임시 아이템 저장을 위한 새로운 맵 그리드인 cache를 통합하였습니다. 이를 통해 에이전트가 새로운 목표를 지속적으로 부여받아야 하는 현실적인 상황을 잘 모델링할 수 있게 됩니다.

- **Technical Details**: L-MAPF는 각 에이전트가 목표 위치로부터 경로를 계획하는 문제로, 공통적으로 복잡한 다중 에이전트를 고려합니다. 이 연구는 캐시 맵 레이아웃 설계를 경로 계획과 통합하여 효율적으로 관리할 수 있는 작업 지정자(Task Assigner, TA)를 도입합니다. TA는 에이전트의 상태에 따라서 동적으로 목표 위치를 할당하고, 관리하는 역할을 하며, 기본적으로 상태 기계와 잠금 메커니즘을 기반으로 설계되었습니다.

- **Performance Highlights**: L-MAPF-CM은 다양한 캐시 교체 정책과 작업 분포에 따라 평가되었습니다. 테스트 결과, 높은 캐시 적중률과 원활한 교통 조건에서 성능 향상을 나타내어, 물류 작업의 효율성을 크게 개선할 수 있음을 보여줍니다. 이러한 실험은 L-MAPF-CM이 많은 테스트 설정에서도 일관된 성과를 보임을 입증합니다.



### Segmenting Text and Learning Their Rewards for Improved RLHF in Language Mod (https://arxiv.org/abs/2501.02790)
- **What's New**: 본 논문에서는 인간 피드백을 통한 강화 학습(RLHF)의 새로운 접근 방식을 제안합니다. 이 접근 방식은 텍스트의 의미적으로 완전한 세그먼트에 대한 보상을 할당하기 위해 세그먼트 수준 보상 모델을 훈련하고 활용합니다. 기존의 토큰 기반 보상 접근 방식의 복잡성을 줄이면서도 보상의 질을 향상시키는 방법론을 고안하였습니다.

- **Technical Details**: 제안된 방법은 동적 텍스트 세그먼테이션을 통해 언어 모델(LM) 생성의 두 가지 난제를 해결합니다. 이를 위해 보상 모델을 Bradley-Terry(BT) 손실 함수를 사용하여 훈련하며, 위치 인식 보상 정규화 기능으로 고전적 보상 정규화를 일반화합니다. 이로 인해 세그먼트 보상이 효과적으로 보강되어 강화 학습 기반 LM 훈련에 필요한 신호의 밀도가 향상됩니다.

- **Performance Highlights**: 논문의 방법론은 AlpacaEval 2.0, Arena-Hard, MT-Bench 등 세 가지 RLHF 벤치마크에서 고전적 밴딧 접근법과 최근의 토큰 레벨 보상 접근법 대비 경쟁력 있는 성능을 달성하였습니다. 여러 차별화를 위한 연구도 수행되어 제안된 디자인 선택의 효과를 증명하였습니다.



### GLoG-CSUnet: Enhancing Vision Transformers with Adaptable Radiomic Features for Medical Image Segmentation (https://arxiv.org/abs/2501.02788)
- **What's New**: 본 연구에서는 의학 이미지 의미 분할(medical image semantic segmentation) 분야에서 Transform 기반 모델의 한계점을 극복하기 위해 Gabor 필터와 Laplacian of Gaussian (LoG) 필터를 활용한 GLoG-CSUnet라는 새로운 아키텍처를 제안합니다. 이 모델은 의료 이미지에서의 국소 및 글로벌 문맥 이해의 균형을 맞추며, 작은 데이터셋에서의 성능 향상을 목적으로 합니다. GLoG-CSUnet은 학습 가능한 radiomic feature를 통합하여 이미지의 텍스처 및 경계 정보를 효과적으로 캡처합니다.

- **Technical Details**: GLoG-CSUnet의 구조는 Convolutional Swin-Unet (CSUnet) 아키텍처를 확장하고, 가변 성질의 Gabor 및 LoG 필터를 병합하여 구성됩니다. Gabor 변환층은 멀티 스케일 및 멀티 방향 텍스처 정보를 추출하며, LoG 변환층은 경계 정밀도를 향상시키고 에지 감지를 개선하는 역할을 합니다. 이러한 필터들은 훈련 과정에서 최적화되고, 그대로 Transformer에 전달되기 전에 공간적 지역성을 보존하기 위해 비겹치는 패치로 나뉩니다.

- **Performance Highlights**: GLoG-CSUnet을 Synapse 다기관 및 ACDC 심장 분할 데이터셋에서 평가한 결과, 기존 최첨단 모델 대비 Dice score에서 Synapse에서 1.14% 및 ACDC에서 0.99%의 유의미한 성능 향상을 보였습니다. 이 모델은 또한 각각 15개 및 30개의 추가 매개변수로 최소한의 계산 오버헤드를 요구하며, 다양한 기본 모델과의 통합을 통한 유연한 디자인을 가지고 있어 의료 이미지 분석에서의 접목 가능성이 높습니다.



### Hybrid deep convolution model for lung cancer detection with transfer learning (https://arxiv.org/abs/2501.02785)
Comments:
          13 pages, 8 figures

- **What's New**: 이번 연구에서는 전이 학습(transfer learning)을 활용한 하이브리드 딥 컨볼루션 모델인 최대 민감도 신경망(Maximum Sensitivity Neural Network, MSNN)을 소개합니다. MSNN은 폐암 검출의 정확도를 향상시키기 위해 민감도(sensitivity)와 특이성(specificity)을 정제하는 데 중점을 두고 설계되었습니다. 이는 기존의 폐암 검출 모델보다 더 높은 정확도를 목표로 하고 있습니다.

- **Technical Details**: MSNN 모델은 실험적 검증을 통해 98%의 정확도와 97%의 민감도를 달성하였습니다. 이 모델은 폐 컴퓨터 단층촬영(CT) 이미지에 민감도 맵(sensitivity maps)을 오버레이하여 악성(malignant) 또는 양성(benign) 분류에 가장 중요한 영역을 시각화할 수 있게 돕습니다. 이러한 접근 방식은 폐암을 구별하는 데 있어 최소한의 오탐(false positive)으로 뛰어난 성능을 보여줍니다.

- **Performance Highlights**: MSNN은 기존의 딥 러닝(deep learning) 접근 방식을 초월하여 폐암 검출의 정확성을 크게 향상시켰습니다. 이 혁신적인 방법은 의료 진단의 정확도를 높이는 데 기여하며, 조기 및 정확한 진단을 위한 가능성을 제시하고 있습니다.



### ICFNet: Integrated Cross-modal Fusion Network for Survival Prediction (https://arxiv.org/abs/2501.02778)
- **What's New**: 이 논문에서는 Integrated Cross-modal Fusion Network (ICFNet)을 제안하여 생존 예측의 정확성을 향상시키기 위한 여러 데이터 모드를 통합합니다. 기존의 방법들이 제한된 데이터 모드에 의존하는 것과는 달리, 이 모델은 병리학적 전체 슬라이드 이미지(whole slide images), 유전체 표현 프로파일(genomic expression profiles), 환자의 인구 통계적 정보(demographic information), 치료 프로토콜을 포함한 네 가지 유형의 데이터를 활용합니다. 이를 통해 공정한 교육과 더 나은 예측을 달성할 수 있습니다.

- **Technical Details**: ICFNet은 서로 다른 데이터 유형을 처리하기 위해 세 가지 모달 인코더와 최적 운송 기반(co-attention-based) Transformer를 사용합니다. 각 모달에서 독특한 특징을 추출하여 불필요한 중복성을 줄이고, 모달리티 별 특성을 강화하는 Residual Orthogonal Decomposition (ROD) 모듈을 포함합니다. 또한, ICFNet은 바람직한 NLLLoss(negative log-likelihood loss)를 설계하여 레이블 불균형 문제를 해결합니다.

- **Performance Highlights**: 다양한 공개 TCGA 데이터셋(BLCA, BRCA, GBMLGG, LUAD, UCEC)에 대한 광범위한 실험 결과, ICFNet은 기존의 최첨단 알고리즘을 능가하는 성능을 보여주었습니다. 이 모델은 환자 치료 결정 지원 도구로서의 활용 가능성을 제시하며, 효율적인 치료 옵션 평가와 의료 자원 낭비 방지에 기여할 수 있습니다.



### Enhancing Trustworthiness of Graph Neural Networks with Rank-Based Conformal Training (https://arxiv.org/abs/2501.02767)
Comments:
          8 pages,2 figures,published to AAAI 2025

- **What's New**: 이번 논문에서는 그래프 신경망(Graphic Neural Networks, GNNs)의 신뢰성을 높이는 방법으로서 Rank-based Conformal Prediction (RCP-GNN)을 제안하고 있습니다. 기존의 Conformal Prediction (CP) 방법의 한계를 극복하여 노드 분류(node classification) 상황에서의 예측 효율성과 신뢰성을 동시에 개선하려는 목표를 가지고 있습니다. 이 과정에서 분류기의 결과로부터 얻은 순위(rank) 정보를 활용하여, 설정된 커버리지 비율을 만족하는 예측 세트를 효율적으로 구성하는 방법을 설명합니다.

- **Technical Details**: 제안하는 방법은 학습 과정에서 분류기 결과의 순위를 바탕으로 하는 conformity loss function을 사용하여, 예측 세트를 조정하게 됩니다. GNNs는 종종 잘 보정되지 않은 예측이 문제되기 때문에, 논문에서는 이 문제를 해결하기 위해 새로운 conformity score를 도입합니다. 이 점에서, 제안한 RCP-GNN은 GNNs의 모델 매개변수가 최적화되는 동안 예측 세트를 조정할 수 있도록 하는 차별화된 프레임워크를 통해 다양한 네트워크 구조에서의 활용 가능성을 높이고자 합니다.

- **Performance Highlights**: 다양한 실제 데이터셋을 바탕으로 한 실험 결과, RCP-GNN은 미리 정의된 목표 마진 커버리지를 달성하면서도 기존 최첨단 방법들에 비해 비효율성을 크게 줄이는 성과를 보였습니다. 이는 노드 분류 작업에서 특히 두드러지며, 제안한 방법이 성능 개선에 기여할 수 있음을 잘 보여줍니다. 이로써, RCP-GNN은 그래프 구조 데이터의 불확실성 추정에서 신뢰성 있는 솔루션을 제공하는 데 기여할 것으로 기대됩니다.



### Are GNNs Effective for Multimodal Fault Diagnosis in Microservice Systems? (https://arxiv.org/abs/2501.02766)
Comments:
          6 pages, 5 figures, submitted to conference

- **What's New**: 이번 연구에서는 Microservice 시스템에서의 오류 진단을 위한 기존의 Graph Neural Networks (GNNs)의 필요성과 효과성을 비판적으로 평가합니다. 이를 위해, GNNs 대신 사용할 수 있는 간단하고 견고한 모델인 DiagMLP를 제안하며, 다섯 개의 공개 데이터셋을 기반으로 실험을 수행하여 DiagMLP가 GNN 기반 방법들과 경쟁할 수 있는 성능을 보임을 밝혔습니다. 이 연구는 GNNs의 현재의 사용 패러다임이 실제적인 기여를 증명하지 못하고 있다고 주장하며, 더 나아가 도전적인 데이터셋 개발의 필요성을 강조합니다.

- **Technical Details**: Microservice 시스템 내의 서비스 간 복잡한 의존성을 모델링하기 위해 GNNs가 일반적으로 사용되지만, 본 연구에서는 이러한 접근법의 성과를 DiagMLP라는 단순한 다층 퍼셉트론(Multi-Layer Perceptron) 모델과 비교합니다. 이 모델은 GNNs에 의존하지 않고도 효과적인 결과를 도출할 수 있는 가능성을 보여주며, 데이터 전처리가 이미 중요한 정보를 추출하고 있음에 주목합니다. 또한, 본 연구는 기존 GNN 기반 모델의 성능 향상이 GNNs의 덕분인지 아니면 다른 구성 요소의 결과인지에 대한 명확한 증거 부족을 지적하고 있습니다.

- **Performance Highlights**: DiagMLP는 다섯 개의 공공 데이터셋에서의 실험을 통해 GNN 기반 모델들보다 더 나은 성능을 보여주었으며, 이는 GNNs의 사용 필요성에 대한 심각한 의문을 제기합니다. 특히, GNNs가 서비스 간의 복잡한 상호작용을 모델링하는 데 효과적인 방법이라고 여겨지지만, 이번 연구에서는 DiagMLP가 얼굴 언어(ensemble 모델)로서 더 나은 성능을 나타냈음을 보고합니다. 이러한 결과는 앞으로의 연구가 충분히 도전적인 데이터셋을 개발하고 문제를 명확히 규명하는 데 중점을 둔 방향으로 나아가야 한다는 것을 강조합니다.



### Visual Large Language Models for Generalized and Specialized Applications (https://arxiv.org/abs/2501.02765)
- **What's New**: 본 논문은 최근 시각-언어 모델(Visual-Language Models, VLMs)과 그 진화에 대해 다루고 있습니다. 특히, 비전 대형 언어 모델(Visual Large Language Models, VLLMs)의 다양한 응용 프로그램과 발전 방향에 대한 포괄적인 조망을 제공합니다. VLLMs는 일반화된 응용 프로그램과 특수화된 응용 프로그램 모두에서 활용될 수 있는 잠재력을 지니고 있으며, 그 발전을 위한 과제와 윤리적 고려사항에 대해서도 논의합니다.

- **Technical Details**: VLLMs는 시각 데이터와 언어 데이터의 통합을 통해 다중 작업을 처리하는 등의 성능을 발휘하고 있습니다. 이 논문에서는 다양한 시나리오에서의 VLLM 활용 방법을 분석하고, 비전 데이터에서 언어로의 전환, 언어에서 비전으로의 전환 및 비전 동작 처리와 관련된 세부 작업을 분류합니다. 또한, VLLM의 구조는 기존의 CNN과 RNN(또는 LSTM) 기반 아키텍처에서 벗어나 Transformer와 같은 현대적 심층 학습 기법을 채택하여 성능을 향상시켰습니다.

- **Performance Highlights**: 이 논문은 VLLMs가 제공하는 뛰어난 제로샷 및 전이 학습 성능을 강조하며, 기존 모델들과 비교할 때 성능 향상에 기여하고 있습니다. VLLMs는 인간의 선호에 맞춰 응답을 생성할 수 있는 능력을 갖추고 있으며, 복잡한 비전 작업 처리 능력을 포함합니다. 또한, 윤리적 고려사항과 보안, 개인 정보 보호, 효율성, 해석력 등과 같은 미래 개발 방향이 제안됩니다.



### Enhancing Robot Route Optimization in Smart Logistics with Transformer and GNN Integration (https://arxiv.org/abs/2501.02749)
Comments:
          21 pages

- **What's New**: 이 연구에서는 스마트 물류에서 로봇을 위한 고급 경로 최적화 방법을 발표합니다. Transformer 아키텍처, Graph Neural Networks (GNNs), Generative Adversarial Networks (GANs)의 융합을 활용하여 경로 효율성을 향상시키고 있습니다. 이를 통해 공간적 및 자원적 제약을 모두 고려한 그래프 기반 표현을 사용하고 있습니다.

- **Technical Details**: 연구에서 사용되는 방법은 지리 데이터, 화물 할당 및 로봇 역학을 포함하는 그래프 기반 표현을 통해 로봇의 최적 화물 운송 경로를 찾습니다. 이러한 방법은 복잡한 물류 상황에서 로봇의 동적 행동을 효과적으로 반영하며, GNNs와 GANs의 힘을 결합하여 데이터를 처리합니다.

- **Performance Highlights**: 광범위한 실제 물류 데이터 세트로 테스트한 결과, 제안된 방법은 여행 거리에서 15% 감소, 시간 효율성에서 20% 향상, 에너지 소비에서 10% 감소라는 주목할만한 개선을 달성했습니다. 이러한 성과는 지능형 물류 운영에서 알고리즘의 효과성을 강조합니다.



### Interpretable Recognition of Fused Magnesium Furnace Working Conditions with Deep Convolutional Stochastic Configuration Networks (https://arxiv.org/abs/2501.02740)
- **What's New**: 이 논문은 융합 마그네슘 용광로에서의 작업 조건 인식을 위한 해석 가능한 방법을 제안합니다. 특정한 문제인 일반화 능력 부족과 해석 가능성 문제를 해결하기 위해, 심층 컨볼루셔널 확률적 구성 네트워크(Deep Convolutional Stochastic Configuration Networks, DCSCNs)를 기반으로 한 접근법이 사용됩니다. 새로운 Gaussian differential convolution kernels의 생성과 인크리멘탈 방법을 통해 인식 오류의 수렴을 보장하며, 강화 학습(Reinforcement Learning)을 통해 모델의 압축 및 최적화를 도모합니다.

- **Technical Details**: 논문에서 제안된 DCSCNs 모델은 피직적으로 의미있는 Gaussian differential convolution kernels를 생성하기 위한 감독 학습 메커니즘을 사용합니다. 네트워크의 채널 특징 맵의 독립 계수를 정의하여 융합 마그네슘 용광로의 활성화 맵 시각화 결과를 얻습니다. 인식 정확성, 해석 가능성 관련 평가 메트릭, 모델 파라미터 수를 통합한 보상 함수가 구축되며, 이러한 알고리즘은 해석 가능성과 성능을 동시에 향상시키는 데 초점을 맞추고 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 다른 심층 학습 기법들에 비해 인식 정확성과 해석 가능성에서 우수한 성능을 보여줍니다. 비록 기존의 심층 신경망이 다양한 분야에서 사용되고 있지만, 내부의 비선형 특성과 인간의 개념적 이해 간의 불일치로 인해 해석 가능성 문제에 직면하고 있습니다. 이 연구에서는 각각의 특징 맵에 대한 시각화를 통해 사용자가 결과를 이해하고 모델의 결정을 조정할 수 있도록 해석 가능성을 높였습니다.



### TARDiS : Text Augmentation for Refining Diversity and Separability (https://arxiv.org/abs/2501.02739)
Comments:
          10 pages

- **What's New**: 이 논문에서는 TARDiS라는 새로운 LLM 기반의 텍스트 증강(Text Augmentation, TA) 방법을 소개합니다. 이 방법은 기존의 두 단계 TA 방법에서 발생하는 생성 및 정렬 단계의 도전 과제를 해결하는 데 중점을 둡니다. TARDiS는 여러 개의 클래스별 프롬프트를 사용하여 다양성과 분리 가능성을 향상시키는 두 가지 생성 프로세스, 즉 SEG (Semantic Enrichment Generation)와 CEG (Contrastive Enrichment Generation)를 제안합니다.

- **Technical Details**: TARDiS의 생성 단계에서는 각 클래스에 대한 스파크 생각(spark thoughts) 개념을 도입하여 LLM의 내재된 지식을 활성화합니다. SEG는 목표 클래스 내의 예제에서 생성된 스파크 생각을 사용하여 클래스 내 다양성을 캡처하며, CEG는 목표 클래스와 모호한 클래스에서 생성된 스파크 생각을 사용해 비목표 클래스와의 분리 가능성을 강화합니다. 정렬 단계에서는 Class Adaptation (CA) 방법을 통해 생성된 예제가 목표 클래스와 일치하도록 수정합니다.

- **Performance Highlights**: 실험 결과, TARDiS는 다양한 퓨쇼트(few-shot) 텍스트 분류 작업에서 기존 LLM 기반 TA 방법들보다 우수한 성능을 보였습니다. 또한, 논문은 각 단계에서의 행동을 상세히 분석하여 TARDiS의 효과를 입증하였습니다. 이를 통해 TARDiS는 기존의 두 단계 TA 방법의 한계를 극복하며, 퓨쇼트 상황에서도 강력한 일반화 성능을 발휘합니다.



### AFed: Algorithmic Fair Federated Learning (https://arxiv.org/abs/2501.02732)
Comments:
          Accepted by IEEE Transactions on Neural Networks and Learning Systems

- **What's New**: 이 논문에서는 여러 클라이언트가 데이터를 중앙 서버에 저장하지 않고도 협력적으로 기계 학습할 수 있는 Federated Learning (FL)에 대한 새로운 접근법인 AFed를 제안합니다. AFed는 클라이언트의 로컬 데이터에 접근하지 않고도 공정한 모델을 학습할 수 있도록 돕는 프레임워크로, 데이터 접근이 제한된 환경에서 그룹 공정성을 촉진하는 효과적인 방법을 제공합니다. 이 논문은 조건부 생성기 및 GAN과 같은 두 가지 접근 방식을 통해 클라이언트 간의 데이터 편향을 제거하는 방법을 제안합니다.

- **Technical Details**: AFed는 서버 측에서 훈련된 조건부 생성기인 AFed-G와 클라이언트 측에서 조건부 GAN을 훈련하는 AFed-GAN의 두 가지 알고리즘을 포함합니다. 이러한 접근 방식을 통해 서버는 클라이언트의 데이터 분포를 학습하고, 이를 통해 로컬 편향을 제거하는 데 필요한 지식을 모든 클라이언트에게 전달할 수 있습니다. 또한 보조 분류 헤드를 설계하여 생성기 훈련을 위한 유용한 특징을 추출하는 데 도움이 됩니다.

- **Performance Highlights**: 제안된 AFed 프레임워크는 여러 실제 데이터셋에서 기초 모델에 비해 상당한 개선 결과를 보여줍니다. 이론적 분석을 통해 제안된 방법의 견고성을 입증하며, 공정성을 유지하는 동시에 성능을 향상시키는 방안으로 주목받고 있습니다. AFed는 클라이언트의 데이터를 사용하지 않고도 그룹 공정성을 달성할 수 있는 새로운 가능성을 열어줍니다.



### OpenGU: A Comprehensive Benchmark for Graph Unlearning (https://arxiv.org/abs/2501.02728)
Comments:
          under review

- **What's New**: 이 논문에서는 그래프 머신 러닝 분야의 중요한 이슈인 그래프 제거(Graph Unlearning, GU)에 대한 최초의 종합 벤치마크인 OpenGU를 소개합니다. 기존의 머신 제거와는 달리, GU는 비유클리드(Non-Euclidean) 그래프 데이터와 GNN의 메시지 전달 메커니즘으로 인해 특별한 도전 과제를 안고 있습니다. 또한 OpenGU는 16개의 SOTA 알고리즘과 37개의 다중 도메인 데이터셋을 통합하여 다양한 다운스트림 작업에서 유연한 제거 요청에 대응할 수 있습니다.

- **Technical Details**: OpenGU는 세 가지 유형의 제거 요청을 처리하며, 알고리즘의 효과성을 평가하기 위해 모델 업데이트와 추론 보호의 두 가지 측면에서 종합적인 평가를 제공합니다. 또한 각 알고리즘이 증가하는 데이터 양 및 복잡한 그래프 구조를 처리할 수 있는지에 대한 통찰력을 제공합니다. 이 벤치마크는 코드 레벨의 구현을 통해 다양한 제거 요청과 다운스트림 작업의 조합을 지원합니다.

- **Performance Highlights**: 상당한 실험을 통해 기존의 GU 방법들에 대한 8가지 주요 결론을 도출하였으며, 이러한 결과는 향후 연구 방향에 대한 통찰을 제공합니다. OpenGU는 특히 실효성, 효율성 및 강건성을 중심으로 한 면밀한 분석을 제공하여 그래프 제거 연구 분야의 발전에 기여할 것입니다. 또한 오픈 소스 벤치마크 라이브러리로 제공되어, 연구자와 실무자가 GU 탐색을 확대할 수 있는 자원을 제공합니다.



### Tree-based RAG-Agent Recommendation System: A Case Study in Medical Test Data (https://arxiv.org/abs/2501.02727)
- **What's New**: HiRMed(계층적 RAG 강화 의료 테스트 추천)이라는 새로운 시스템이 소개되었습니다. 이 시스템은 Retrieval-Augmented Generation (RAG)을 활용하여 의료 테스트 추천을 위한 트리 구조의 추천 시스템을 구현합니다. 전통적인 벡터 유사성 기반 방법과 달리, HiRMed는 각 트리 노드에서 의료 추론을 수행하여 초기 증상으로부터 진단 경로를 동적으로 조정합니다.

- **Technical Details**: 이 시스템은 환자의 초기 증상에 따라 진단 요구 사항과 잠재적인 기저 상태를 식별하는 단계별 의료 분석을 수행합니다. Hierarchical RAG Architecture는 각 노드가 특별한 RAG 프로세스를 통합하는 트리 구조를 통해 여러 단계의 의료 추론을 가능하게 합니다. 또한, 두 개의 지식 기반 아키텍처를 통해 일반 의료 이해 및 전문 진단 고려사항을 포함한 동적 지식 통합을 제공합니다.

- **Performance Highlights**: HiRMed는 기존의 검색 기반 방법에 비해 더 높은 정확도와 가능성 있는 진단 경로의 커버리지, 중요한 진단 테스트의 누락 비율 감소를 보여주었습니다. 연구 결과는 HiRMed가 테스트 추천의 해석 가능성을 높이고 명확한 추론 경로를 바탕으로 추천의 타당성을 증명했음을 나타냅니다. 이러한 혁신적인 기능들은 차세대 의료 테스트 추천 시스템을 위한 청사진을 제시합니다.



### Improved Data Encoding for Emerging Computing Paradigms: From Stochastic to Hyperdimensional Computing (https://arxiv.org/abs/2501.02715)
Comments:
          5 pages, 3 figures, 4 tables

- **What's New**: 이 연구는 확률적 컴퓨팅(Stochastic Computing, SC) 및 고차원 컴퓨팅(Hyperdimensional Computing, HDC)에서의 데이터 인코딩을 위한 새로운 전략을 제안합니다. 본 논문에서 저자들은 Van der Corput(VDC) 시퀀스와 같은 낮은 불일치(low-discrepancy) 시퀀스를 활용하여 무작위 수를 생성합니다. 이 방법은 SC 및 HDC 시스템의 성능과 에너지 효율성을 크게 향상시키며, 특히 비효율적인 기존 pseudo-random 수 생성기를 대체할 수 있습니다.

- **Technical Details**: 본 논문에서는 VDC 시퀀스를 활용하여 무작위성을 확보하는 방법을 다룹니다. VDC 시퀀스는 낮은 불일치(LD)성을 특징으로 하며, 이는 더욱 균일한 데이터 분포를 가능하게 합니다. 전통적인 pseudo-random 시퀀스를 사용하는 대신, VDC 기반의 인코딩 방법이 SC 및 HDC의 데이터 전송 및 저장에서 발생하는 문제를 해결하는 데 도움을 주며, 이러한 접근 방식은 경량화된 하드웨어 설계에도 적합합니다.

- **Performance Highlights**: 실험 결과, 제안된 VDC 기반 인코딩 접근 방식이 정확성과 에너지 절약에서 현저한 개선을 보여주었습니다. 특히, SC 및 HDC 시스템에서 높은 품질의 무작위성을 달성하여 모델 성능의 저하 및 하드웨어 비용 증가를 방지합니다. 이러한 개선 사항은 자원이 제한된 환경에서도 SC와 HDC를 효과적으로 통합할 수 있는 강력한 기반을 제공합니다.



### Horizon Generalization in Reinforcement Learning (https://arxiv.org/abs/2501.02709)
- **What's New**: 이번 연구에서는 목표 기반 강화 학습(Goal-conditioned RL)을 일반화 관점에서 살펴보았으며, 전통적인 방식의 랜덤 증강(random augmentations)이나 도메인 무작위화(domain randomization)가 아닌, 목표 지향 정책(goal-directed policies)을 학습하는 데 집중했습니다. 이 정책들은 가까운 목표를 도달하는 훈련을 한 후, 멀리 있는 목표를 성공적으로 도달할 수 있어야 하며, 이러한 개념은 기획(planning) 불변성과 밀접하게 연결되어 있다고 주장합니다.

- **Technical Details**: 이 논문에서는 지평선 일반화(horizon generalization)와 계획 불변성(planning invariance)을 이론적으로 분석하였으며, 특정 가정 하에 이러한 두 가지 특성을 달성하는 것이 가능함을 증명했습니다. 목표를 향해 나아가는 정책이 특정 웨이포인트(waypoint)를 향해 이동할 때와 같은 행동을 선택해야 한다는 점에서, 멀리 있는 목표를 향한 정책이 가까운 목표를 학습하여 훈련된 경우에도 성공할 수 있다는 결론을 내렸습니다.

- **Performance Highlights**: 우리는 이론적 결과를 뒷받침하는 새로운 실험 결과를 제시하고 이전 연구 결과를 회상하면서, 다른 머신러닝 분야에서 개발된 불변성 및 일반화 기법이 이러한 매력적인 특징을 달성하기 위해 어떻게 조정될 수 있는지에 대한 연구의 가능성을 열어주었습니다. 이러한 발견들은 강화 학습의 새로운 방향을 제시하고, 목표 기반 학습의 발전에 기여할 것으로 기대됩니다.



### QuIM-RAG: Advancing Retrieval-Augmented Generation with Inverted Question Matching for Enhanced QA Performanc (https://arxiv.org/abs/2501.02702)
- **What's New**: 이 연구는 질문 응답(QA) 작업을 개선하기 위한 Retrieval-Augmented Generation (RAG) 시스템의 새로운 아키텍처를 제안합니다. 기존의 대형 언어 모델(LLMs)에서는 실시간 업데이트가 어려웠지만, RAG는 하이퍼링크와 데이터베이스를 통합하여 맥락적으로 적합한 응답을 생성합니다. 본 연구에서는 정보의 희석(information dilution)과 망상(hallucinations)이라는 전통적인 RAG가 직면한 문제를 해결하기 위해 QuIM-RAG(Query-to-Question Inverted Index Matching)이라는 새로운 접근 방식을 도입하였습니다.

- **Technical Details**: QuIM-RAG는 문서 청크에서 잠재적인 질문을 생성하고 이를 사용자 쿼리와 매칭하여 가장 관련 있는 텍스트 청크를 찾아 정확한 응답을 생성합니다. 이 시스템은 Meta Inc.의 오픈소스 Meta-LLaMA3-8B-instruct 모델을 기반으로 구현되었습니다. 500 페이지 이상의 다양한 웹사이트에서 수집한 도메인 특화 데이터셋을 활용했으며, BERT-Score와 RAGAS 메트릭스를 사용하여 평가를 수행하였습니다.

- **Performance Highlights**: 본 연구에서 제안하는 접근 방식은 전통적인 RAG 아키텍처보다 BERT-Score와 RAGAS에서 모두 더 나은 성능을 나타냈습니다. 이러한 결과는 사용자 쿼리에 대한 정확한 응답을 생성하는 데 있어 우리의 새로운 RAG 모델이 효과적임을 보여줍니다. 또한, 사용자에게 제공하는 모든 정보는 원본 문서와 연결된 출처 링크를 포함하고 있어 신뢰성 있는 정보를 탐색할 수 있도록 지원합니다.



### EAGLE: Enhanced Visual Grounding Minimizes Hallucinations in Instructional Multimodal Models (https://arxiv.org/abs/2501.02699)
Comments:
          12 pages, 4 figures, 8 tables

- **What's New**: 본 연구에서는 EAGLE라는 새로운 접근법을 제안하여, IT-VLMs에서 발생하는 환각 문제를 시각 인코더의 능력을 향상시킴으로써 해결하고자 합니다. EAGLE는 기존의 모델이나 융합 모듈에 구애받지 않으며, 후속 사전 훈련 방식으로 비전 트랜스포머(ViT)를 최적화합니다. 기존의 대규모 사전 학습된 비전 및 언어 아키텍처의 한계를 극복하여, IT-VLM에서의 이론적 오류를 줄이는 결과를 보여줍니다.

- **Technical Details**: EAGLE는 수정된 대조 학습 프레임워크를 사용하여 로컬 패치 임베딩을 개선합니다. 이를 통해 ViT는 이미지 내 물체에 대한 정제된 정보를 인코딩할 수 있으며, IT-VLM이 사용하는 시각 특성과 언어 정렬을 비교하여 이 효과를 입증합니다. 실험 결과, EAGLE로 최적화된 ViT는 IT-VLM의 기존 비전 인코더를 대체하여 추가 조정 없이도 시각적 응답의 질을 향상시킵니다.

- **Performance Highlights**: EAGLE는 MMVP 및 MERLIM 벤치마크에서 각각 11.2%와 6.3%의 향상을 달성하며, 다양한 IT-VLM 모델에서 환각 현상을 현저히 줄입니다. 본 연구에서 제안하는 EAGLE는 6개의 최첨단 IT-VLM에서 진행된 3개의 도전적인 벤치마크에서 의미 있는 성능 개선을 달성함을 보여줍니다.



### From Superficial Patterns to Semantic Understanding: Fine-Tuning Language Models on Contrast Sets (https://arxiv.org/abs/2501.02683)
- **What's New**: 이 연구는 자연어 추론(Natural Language Inference, NLI) 태스크에서 언어 모델의 편향을 줄이고 그 강건성을 향상시키기 위해 대조 세트(contrast sets)를 사용하는 방법을 탐구합니다. 기존의 데이터셋에서 높은 정확도를 기록하는 ELECTRA-small 모델이 대조 세트에 평가될 때 정확도가 급격히 떨어지는 현상을 분석하여, 더 복잡한 예제를 활용한 사전 훈련 모델의 미세 조정이 필요하다는 점을 강조합니다. 이를 통해 모델이 언어를 깊이 이해할 수 있도록 도움을 줄 수 있습니다.

- **Technical Details**: 이 연구에서는 1,400만 개의 매개변수를 가진 ELECTRA-small 모델을 훈련하여 SNLI(Supervised Natural Language Inference) 데이터셋에서 NLI 작업을 수행하였습니다. 대조 세트는 Linguistically-Informed Transformations(LIT)를 사용하여 자동 생성하였으며, 총 14,363개의 대조 예제가 만들어졌습니다. 이 대조 세트는 원본 SNLI 데이터의 원래 레이블을 변경시키는 방식으로 미세 조정 과정에서 활용되었습니다.

- **Performance Highlights**: 대조 세트에서 모델의 정확도를 향상시키기 위해 복잡한 예제에 노출시키는 훈련 방법을 사용하였고, 이로 인해 정확도는 74.9%에서 90.7%로 증가하였습니다. 대조 세트에 대한 성능 향상은 훈련 데이터의 다양성이 모델의 언어 이해를 향상시키는 데 얼마나 중요한지를 보여줍니다. 원본 SNLI 데이터 세트에서의 성능은 유지되었으며, 이는 모델이 복잡성과 변동성에 저항력을 가질 수 있는 길을 열어줍니다.



### From thermodynamics to protein design: Diffusion models for biomolecule generation towards autonomous protein engineering (https://arxiv.org/abs/2501.02680)
- **What's New**: 이번 논문에서는 단백질 설계에서 혁신적인 접근 방식으로 확산 모델(difussion models)의 응용을 주목합니다. 특히, Denoising Diffusion Probabilistic Models(DDPM)와 Score-based Generative Models(SGM)의 두 가지 전략을 탐구하며, 이 모델들이 단백질 설계, 펩타이드 생성 및 약물 발견에서 보여준 성과를 강조합니다. 또한, E(3) 군에 기반한 동등성(equivariance) 개념을 통해 아미노산의 구조적 안정성을 유지하는 방법에 대해서도 논의합니다.

- **Technical Details**: 이 논문은 DDPM과 SGM이라는 두 가지 주요 확산 모델에 대한 이론적 기초를 제공합니다. 확산 모델은 두 단계로 구성된 심층 생성 모델(deep generative model)이며, 먼저 데이터에 가우시안 노이즈를 점진적으로 추가하는 '전방 확산(Forward diffusion)' 단계가 있습니다. 이어서, '역확산(Reverse diffusion)' 단계에서는 층별로 노이즈 제거(denoising)를 학습하여 원래의 입력 데이터를 복원하는 과정이 포함됩니다.

- **Performance Highlights**: 연구는 확산 모델이 기존의 생성 모델들과 비교하여 단백질 생성 작업에 있어서 뛰어난 성능을 보여주고 있음을 강조합니다. 또한, DDPM과 SGM이 복잡한 분포를 모델링하는 데 유리하여 단백질 디자인에 대한 새로운 가능성을 제시합니다. 최종적으로, 본 논문은 확산 모델들이 자율 단백질 설계 및 엔지니어링을 촉진하는 데 어떻게 기여할 수 있는지에 대한 미래의 방향성을 제시합니다.



### Multi-Aggregator Time-Warping Heterogeneous Graph Neural Network for Personalized Micro-Video Recommendation (https://arxiv.org/abs/2501.02666)
- **What's New**: 이 논문은 개인화된 뉴스 성격의 마이크로 비디오 추천을 위한 새로운 모델인 MTHGNN을 제안합니다. 기존의 추천 시스템은 마이크로 비디오의 특성을 충분히 고려하지 못했으며, 이 모델은 사용자의 연속적인 세션을 기반으로 합니다. 특히, 마이크로 비디오의 사회적 상호작용 및 다중 모달 특성을 포착하여 추천 정확도를 향상시키는 데 중점을 두고 있습니다.

- **Technical Details**: MTHGNN 모델은 방향성 시간이 왜곡된 이질 그래프를 구축하여 사용자의 다중 상호작용을 설명합니다. 여기서 Relational Heterogeneous Message Passing aggregator와 Attention Heterogeneous Message Passing aggregator가 사용되어 복잡한 이질 정보를 노드에 내장합니다. 또한, 시간에 따른 사용자의 선호 변화를 고려하기 위해 다중 모달 이질 그래프를 연속 세션으로 나누어 과거 및 현재의 관심사를 융합합니다.

- **Performance Highlights**: 실험 결과, MTHGNN 모델은 TikTok 및 MovieLen과 같은 실제 세계 데이터셋에서 최신 기술 대비 우수한 성능을 보였습니다. 이 모델은 마이크로 비디오 추천뿐만 아니라 장기 영화 추천에서도 높은 성능을 나타내, 복잡한 사회적 관계와 동적 변화를 효율적으로 반영했다고 평가되었습니다. 이는 마이크로 비디오 추천의 성능 향상을 위한 중요한 기여로 해석됩니다.



### Tougher Text, Smarter Models: Raising the Bar for Adversarial Defence Benchmarks (https://arxiv.org/abs/2501.02654)
Comments:
          Will be presented as an oral in-person presentation at the conference of COLING 2025

- **What's New**: 이번 연구에서는 텍스트 적대적 방어를 위한 포괄적인 벤치마크를 제시하여 이전 연구를 크게 확장했습니다. 이 벤치마크는 다양한 데이터셋, 최신 방어 메커니즘을 평가하며, 단일 문장 분류, 유사성 및 패러프레이즈 식별, 자연어 추론, 상식 추론과 같은 중요한 작업들로 평가 범위를 확장합니다. 이 연구는 적대적 강건성을 연구하는 연구자와 실무자들에게 귀중한 자원이 될 뿐만 아니라, 텍스트 적대적 방어에서의 미래 연구의 주요 영역을 식별합니다. 

- **Technical Details**: 딥러닝 모델의 적대적 공격에 대한 취약성은 NLP의 주요 관심사로 떠오르고 있으며, 본 섹션에서는 증가하는 적대적 방어 방법과 이것의 다양한 NLP 작업에 대한 적응 가능성을 강조합니다. 적대적 공격은 입력 텍스트를 조작하여 의미를 보존하지만 모델의 오분류를 초래하는 것을 목표로 합니다. 이 연구에서는 다양한 방어 전략 중에서 구조의 유연성을 중시하며, 학습 효율성을 높이기 위해 필요하지 않은 정보 없이도 견고한 방어 방법을 제안합니다.

- **Performance Highlights**: 제안된 TTSO++는 엔트로피 항을 통한 동적 신뢰도 조정이 통합된 새로운 변형으로, 텍스트 적대적 공격에 대한 강인성을 크게 향상시킬 수 있습니다. 특히 TextFooler와 TextBugger 시나리오에서 더욱 뛰어난 성능을 보입니다. 본 연구에서 제시된 벤치마크는 최신 방어 기술과 다양한 NLP 작업을 평가하여, 적대적 방어 영역에서 더 나은 기준을 제시하고 있으며, 향후 연구에 있어 실질적인 발전을 가속화하는 토대를 마련할 것으로 기대됩니다.



### Tighnari: Multi-modal Plant Species Prediction Based on Hierarchical Cross-Attention Using Graph-Based and Vision Backbone-Extracted Features (https://arxiv.org/abs/2501.02649)
Comments:
          CVPR GeolifeCLEF

- **What's New**: 이번 연구에서는 88,987개의 식물 조사 기록과 함께 위성 이미지, 기후 시계열 데이터, 토지 이용, 생물 기후, 토양 변수와 같은 환경 데이터를 사용하여 식물 종 조합을 예측하는 모델을 개발했습니다. 우리의 모델은 Swin-Transformer 블록을 기반으로 하여 시간적 특성을 추출하고, 다양한 모달리티의 특징들을 효과적으로 융합하기 위해 계층적 크로스 어텐션 메커니즘을 도입하였습니다. 이 연구는 이전의 여러 경쟁 모델들로부터 영향을 받아 예측 정확도를 높이는 여러 전략들을 구사합니다.

- **Technical Details**: 우리는 그래프 구조를 기반으로 한 특징 구축 및 결과 수정 방법을 제안하였습니다. 이 그래프는 SurveyID를 노드로 사용하여 생태적 지조가 비슷한 설문 조사가 인접할 경우 노드를 연결합니다. 모델에 Swin-Transformer를 이용하여 시간적 특성을 추출하고, EfficientNet-B0로 이미지 특징 추출을 대체 함으로써 훈련 속도를 높였습니다. 또한, Top-K 규칙의 후처리 방법을 개선하여 최종 예측의 정확도를 개선했습니다.

- **Performance Highlights**: 아블레이션 실험 결과, 제안된 솔루션 파이프라인이 모델 성능을 크게 향상시킨 것으로 나타났습니다. Tighnari라는 이름의 모델은 환경 보호에 기여하고, 주어진 시공간 맥락에서 식물 종의 조합을 정확히 예측하는 것을 목표로 하고 있습니다. 이 모델은 Transformer 기능을 포함하며, 이미지 프로세싱, 그래프 특징 추출, 계층적 크로스 어텐션 융합 메커니즘을 통합하여 높은 성능을 발휘합니다.



### Representation Learning of Lab Values via Masked AutoEncoder (https://arxiv.org/abs/2501.02648)
Comments:
          10 pages main text, 8 appendix

- **What's New**: 이번 연구에서는 Lab-MAE라는 새로운 transformer 기반의 masked autoencoder 프레임워크를 제안하며, 이는 laboratory value의 결측치를 정확하게 채우는 데 중점을 두고 있습니다. Lab-MAE는 self-supervised learning을 활용하여 연속적인 실험실 값의 채우기를 지원하며, 시간적 의존성을 명확히 포착할 수 있도록 설계된 구조적 인코딩 스킴을 도입하였습니다. MIMIC-IV 데이터셋에 대한 실험 결과, Lab-MAE는 RMSE, R-squared, Wasserstein distance를 포함한 다양한 지표에서 기존의 XGBoost와 같은 기법들보다 뛰어난 성능을 보였습니다.

- **Technical Details**: Lab-MAE는 EHR 데이터의 복잡한 시간적 및 맥락적 의존성을 모델링하기 위해 설계된 masked autoencoder 아키텍처입니다. 이 모델은 실험실 테스트 값과 해당 타임스탬프를 함께 인코딩하여 시간적 의존성을 명시적으로 포착하게 됩니다. MIMIC-IV 데이터셋에서 1,417,738개의 입원 기록을 바탕으로 데이터를 처리하였으며, 데이터 전처리 과정에서 잘못된 값들을 정리하고 정량화하여 극단적인 이상치를 제한 하였습니다.

- **Performance Highlights**: Lab-MAE는 다양한 인구 집단에서 공평한 성능을 보이며, 임상 예측의 공정성을 향상시키는 데 기여하고 있습니다. 기존의 방법들과 비교하였을 때 Lab-MAE는 결측 데이터가 없을 때에도 뛰어난 강건성을 자랑합니다. 이 연구는 Transformer 아키텍처를 기반으로 한 접근 방식이 임상 데이터를 위한 보다 정확하고 공정한 결측치 보간 모델의 기초가 될 수 있음을 제시합니다.



### Trust and Dependability in Blockchain & AI Based MedIoT Applications: Research Challenges and Future Directions (https://arxiv.org/abs/2501.02647)
- **What's New**: 이 논문은 인공지능(AI)과 블록체인 기술의 통합이 의료 인터넷(MedIoT) 응용 프로그램에서 어떻게 혁신적인 의료 제공을 가능하게 하는지를 비판적으로 검토합니다. 특히 AI가 진단과 환자 치료를 개선하는 데 있어의 잠재력과, 블록체인이 데이터 보안 및 환자 프라이버시를 강화하는 데 기여하는 방식을 분석합니다. 또한 이러한 시스템의 신뢰성과 신뢰성을 구축하기 위한 필요성을 강조합니다.

- **Technical Details**: 이 연구는 MedIoT 분야에서의 의료 데이터 관리의 혁신적인 솔루션과 함께 데이터의 확장성(scalability), 프라이버시(privacy) 유지, 윤리적 관행(promoting ethical practices) 등의 도전과제를 다룹니다. AI 기반 통찰력과 블록체인 보안을 의료 분야에 통합하는 비전을 제시하며, 현재 연구의 종합적인 검토와 향후 방향을 논의합니다.

- **Performance Highlights**: 결론적으로, 논문은 신뢰할 수 있고 안전하며 환자 중심(patient-centric)인 MedIoT 응용 프로그램을 위해 해결해야 할 연구 격차를 제시합니다. 이러한 격차를 해결하는 것이 미래의 의료 데이터와 환자 관리에서 AI와 블록체인 기술을 상호 작용하게 하는 데 결정적인 역할을 할 것입니다.



### Layer-Level Self-Exposure and Patch: Affirmative Token Mitigation for Jailbreak Attack Defens (https://arxiv.org/abs/2501.02629)
- **What's New**: 이 논문에서는 Layer-AdvPatcher라는 새로운 방법론을 제안한다. 이 접근법은 LLMs의 특정 레이어를 패치(patch)하여 jailbreak 공격에 대항하는 방어를 설계한다. 특히, 해로운 프롬프트에 직면했을 때 긍정적인 토큰(affirmative tokens)을 생성하는 경향이 있는 레이어를 식별하여, 이를 통해 모델의 안전성을 강화하고자 한다.

- **Technical Details**: Layer-AdvPatcher는 자기 증강 데이터셋(self-augmented datasets)을 활용하여 LLMs의 특정 레이어를 패치하는 비학습 전략(unlearning strategy)을 적용한다. 이 방법을 통해 해로운 데이터를 발생시키는 레이어를 적대적으로 노출하여 그들의 취약성을 이해하고, 이후 이 문제들을 '비학습(unlearn)'하여 긍정적인 토큰의 영향을 줄인다. 이러한 과정은 모델의 안전성 질의를 효과적으로 유지하면서도 jailbreak 공격으로 인한 위험을 최소화하는 데 목적이 있다.

- **Performance Highlights**: 두 개의 모델과 네 개의 벤치마크 데이터셋을 사용한 광범위한 실험을 통해 이 방법론의 효율성을 입증하였다. 실험 결과, 최근의 방어 방법들과 비교하여 해로운 공격의 성공률을 낮추면서도 무해한 질의에 대한 유용성을 훼손하지 않음을 보여준다. 이를 통해 Layer-AdvPatcher는 LLM의 안전성을 증진시키는 효과적인 방안으로 판단된다.



### Cracks in The Stack: Hidden Vulnerabilities and Licensing Risks in LLM Pre-Training Datasets (https://arxiv.org/abs/2501.02628)
Comments:
          Accepted in the Second International Workshop on Large Language Models for Code (LLM4Code 2025)

- **What's New**: 이 논문은 LLM(대규모 언어 모델) 기반 코드 제안 시스템에서 코드 품질을 향상시키기 위한 자동화된 코드 자원을 선별하는 방법을 제안합니다. 특히, 오픈 소스 소프트웨어 프로젝트의 전체 버전 이력을 활용하여 품질 높은 트레이닝 데이터 세트를 만들기 위한 자동화 프로세스를 강조합니다. 이는 코드 생성에 의해 나타나는 버그와 취약성을 줄이는 데 기여할 것으로 기대됩니다.

- **Technical Details**: 제안하는 기술은 Universal Version History(UVH) 개념을 바탕으로 하여, 오픈 소스 프로젝트에서 수정되었거나 변경된 코드 샘플을 식별하고, 이러한 변경 중에서 버그 수정이 포함된 샘플을 분류하는 것입니다. 이를 통해 Stack v2 데이터 세트에서 약 17%의 코드 버전이 새로운 버전으로 업데이트되었다는 결과를 도출하였으며, 이는 종종 알려진 취약점을 해결하기 위한 수정 사항을 포함하고 있습니다. 이 연구는 고품질의 소스 코드 데이터 세트를 생성하기 위한 기초를 마련하는 데 초점을 맞추고 있습니다.

- **Performance Highlights**: 연구 결과, Stack v2의 중복되지 않은 코드 버전에도 여전히 6,947개의 알려진 CVEs가 취약하게 남아있으며, 생성된 블롭의 58%는 최초 생성 이후로 수정되지 않았습니다. 이러한 결과는 코드 데이터 세트에 대한 신뢰성을 높이기 위한 개선점들을 제시하며, 새로운 LLM 모델 트레이닝 과정에서 버그가 있는 코드 패턴과 라이센스 위반을 지속적으로 방지할 수 있는 가능성을 보여줍니다. 따라서, 본 연구는 데이터 선별 자동화 프로세스 개선의 필요성을 강조하며, AI 도구의 산출물 신뢰성을 향상할 수 있는 긍정적인 방향성을 제공하고 있습니다.



### LLMs Help Alleviate the Cross-Subject Variability in Brain Signal and Language Alignmen (https://arxiv.org/abs/2501.02621)
- **What's New**: 이번 연구는 EEG 신호에서 피험자 독립적인 의미 정보를 추출하기 위해 Large Language Models (LLMs)를 활용하는 새로운 접근 방식을 제안합니다. 또한, LLMs를 통해 신호의 잡음을 제거하고 심리적 의미를 이해하는 데 필요한 방법론을 설명합니다. 연구 결과는 EEG 기반 BCI 시스템의 강력함과 일반화 가능성을 개선하는 데 기여할 것으로 기대하고 있습니다.

- **Technical Details**: 이 연구에서는 고차원 EEG 신호를 저차원 밀집 표현으로 생성하기 위해 오토인코더를 사용하고, 이를 LLM의 텍스트 임베딩 공간에 정렬합니다. 그 과정에서 프롬프트 튜닝 기법을 도입하여 LLM이 부호화된 EEG 특성에서 일관된 언어 출력을 생성할 수 있도록 합니다. 이 방법론은 이전에 보지 못한 피험자에 대한 제로샷 예측을 가능하게 하며 BCI 시스템의 내구성과 일반화 능력을 크게 향상시킵니다.

- **Performance Highlights**: 실험 결과 및 절삭 연구를 통해 LLM이 잡음이 많은 EEG 데이터에서 의미 정보를 디코딩하는 데 중요한 역할을 하며, 제로샷 예측이 가능하다는 점을 강조합니다. 이 연구는 EEG 신호 처리를 향상시켜, 더 적응력 있고 보편적으로 적용할 수 있는 BCI 시스템으로 발전할 수 있는 기반을 마련합니다. 또한, BCI 연구 및 산업 적용에 있어 새로운 가능성을 열어줄 것으로 기대됩니다.



### TAPAS: Thermal- and Power-Aware Scheduling for LLM Inference in Cloud Platforms (https://arxiv.org/abs/2501.02600)
- **What's New**: TAPAS라는 새로운 프레임워크가 제안되었으며, 이는 LLM(Inference for Large Language Models) 클러스터의 열 및 전력 관리를 고려하여 최적화된 개발입니다. TAPAS는 GPU 워크로드를 위한 열 및 전력 식별을 통해 응급 상황 처리 능력을 향상시켜 TCO(Total Cost of Ownership)를 효과적으로 줄이고 있습니다. 또한, TAPAS는 전력 및 냉각 장애 상황에서도 동작할 수 있도록 설계되었습니다.

- **Technical Details**: TAPAS는 역사적 온도 및 전력 데이터를 활용하여 GPU VM을 열과 전력 제약 내에서 효율적으로 배치하고, LLM 인퍼런스 요청을 경량화하여 SaaS VM을 통해 라우팅하는 방법을 사용합니다. 이 시스템은 GPU 서버를 공간적으로 배치하면서도 고유한 성능, 열 및 전력 특성을 고려하여 작업을 수행합니다. TAPAS는 메모리 및 렌더링 요구 사항을 동적으로 조정하여 전력 및 냉각 실패 상황에 효과적으로 대응합니다.

- **Performance Highlights**: TAPAS는 대규모 GPU 클러스터에서 검증되었으며, 인퍼런스 요청의 P99 지연 시간을 유지하면서 최대 온도를 17%, 최대 열 전력을 23% 낮추는 성과를 달성했습니다. 이로 인해 최대 40%의 추가 용량 확보가 가능하며, 그 결과 데이터 센터의 TCO도 감소합니다. TAPAS는 다른 정책들과 비교했을 때 열 및 전력 제한 사건을 각각 97%, 99% 줄일 수 있는 효과를 보여주었습니다.



### Empowering Bengali Education with AI: Solving Bengali Math Word Problems through Transformer Models (https://arxiv.org/abs/2501.02599)
- **What's New**: 본 연구는 벵골어(MWPs) 수학 문제를 해결하기 위한 혁신적인 접근법을 제시합니다. 특히 transformer 기반 모델인 Basic Transformer, mT5, BanglaT5, mBART50을 활용하며, 이를 지원하기 위해 10,000개의 벵골어 수학 문제를 포함한 "PatiGonit" 데이터셋을 도입하였습니다. 이 연구는 벵골어의 자연어 처리(NLP) 분야에서 큰 진전을 이루어내며, 교육 AI 도구 개발에 기여할 수 있는 귀중한 방법론과 자원을 제공합니다.

- **Technical Details**: 연구자들은 텍스트 내에 포함된 수학 방정식을 식별하기 위해 transformer 기반 모델을 적용하였으며, 이 과정에서 하이퍼파라미터(learning rate, epochs, batch size)를 최적화하였습니다. 이 접근법은 자연어 처리 기술을 통해 수학 문제의 방정식을 예측하고, 예측된 방정식을 통해 최종 답을 도출하는 방식을 사용합니다. 최종적으로 mT5 모델은 97.30%의 정확도를 기록하여 transformer 모델들이 벵골어 문제 해결에 효과적임을 입증하였습니다.

- **Performance Highlights**: 이 연구는 벵골어로 된 수학 문제 해결의 최신 기술을 발전시키고, 다국어 교육 기술의 접근성을 높이는 것을 목표로 합니다. ‘PatiGonit’ 데이터셋의 생성과 transformer 모델의 적용 및 미세 조정을 통해 벵골어 수학 문제 해결에 대한 효과를 입증하였습니다. 이 연구는 벵골어를 사용하는 학생들이 수학 교육 및 문제 해결 능력을 향상시키는 데 중요한 기여를 합니다.



### Evolving Skeletons: Motion Dynamics in Action Recognition (https://arxiv.org/abs/2501.02593)
Comments:
          Research report

- **What's New**: 이번 논문은 skeleton-based action recognition(스켈레톤 기반 행동 인식)에서 전통적인 스켈레톤 시퀀스와 Taylor-transformed skeleton(테일러 변환 스켈레톤) 시퀀스를 비교검토합니다. 특히, Spatiotemporal Graph Convolutional Network(ST-GCN)와 Hyperformer의 성능을 평가함으로써, 동적인 움직임을 캡처하는 데 있어 두 모델의 장단점을 분석했습니다. 또한, 움직임의 동역학을 통합한 새로운 모형을 소개하며 이들이 행동 인식에서 갖는 가능성을 강조하고 있습니다.

- **Technical Details**: 본 연구에서는 스켈레톤을 그래프로 표현하여 인체의 관절 간 관계를 파악하는 ST-GCN과 고차원 상호작용을 모델링하는 hypergraph 기반의 Hyperformer를 비교합니다. Taylor-transformed skeletons는 임의의 파생값을 포함시켜 시간적 표현을 강화하며, 특히 복잡한 행동을 구분하는 데 유용합니다. 본 논문은 NTU-60 및 NTU-120 데이터셋을 사용하여 이들 모델의 성능을 평가하고, 정적 포즈 대비 동적 모션을 주입한 포즈의 효과를 비교합니다.

- **Performance Highlights**: 분석 결과, Taylor-transformedd skeletons는 모션 역학을 강화하는 데 효과적임을 보여주었습니다. 그러나 기존 스켈레톤 모델들보다 더 많은 정보를 포함하고 있어 여전히 해결해야 할 도전과제가 존재합니다. 본 연구는 스켈레톤 모델링 기술에 대한 혁신 필요성을 강조하며, 동적인 데이터를 효과적으로 처리하기 위한 접근 방식을 제시합니다.



### Efficient Architectures for High Resolution Vision-Language Models (https://arxiv.org/abs/2501.02584)
Comments:
          Accepted to COLING 2025

- **What's New**: Pheye는 고해상도 이미지를 효율적으로 처리하면서 부족한 매개변수로 훈련되는 새로운 비전-언어 모델(Vision-Language Model, VLM) 아키텍처입니다. 이 모델은 기존의 VLM보다 성능은 유지하면서도 더 높은 효율성을 제공합니다. 특히, 세밀한 이미지 이해 또는 장면 텍스트 처리와 같은 작업에 강점을 보이고 있습니다.

- **Technical Details**: Pheye는 고정된 세팅의 언어 모델과 CLIP 기반의 비전 인코더를 결합하며, 이들 사이의 dense cross-attention 레이어를 통해 상호작용합니다. 비전 인코더는 전역 이미지 및 지역 고해상도 패치를 처리하는 두 가지 LoRA 어댑터 세트를 사용합니다. 모델 설계에서는 효과적인 convergence를 위해 LayerNorm을 사용하며, cross-attention을 통합하여 훈련 가능한 매개변수가 줄어듭니다.

- **Performance Highlights**: Pheye는 가격 대비 경쟁력 있는 모델로서, TextVQA와 같은 장면 텍스트 관련 작업에서 특히 돋보이는 성능을 나타냅니다. 이 모델은 훈련 매개변수가 적음에도 불구하고 높은 해상도의 이미지를 효과적으로 처리하여 다양한 리소스 제한 환경에서도 활용 가능성을 제시합니다. 성능 향상을 위해 자체적으로 제공되는 훈련된 모델과 코드의 GitHub 저장소도 공개되어 있습니다.



### Energy Optimization of Multi-task DNN Inference in MEC-assisted XR Devices: A Lyapunov-Guided Reinforcement Learning Approach (https://arxiv.org/abs/2501.02572)
Comments:
          13 pages, 7 figures. This work has been submitted to the IEEE for possible publication

- **What's New**: 본 논문은 Extended Reality (XR) 환경에서 DNN 모델의 다중 작업 추론을 최적화하기 위한 분산 대기열 모델을 개발했습니다. 이를 통해 자원 경쟁과 대기열 결합 문제를 해결하며, 리소스 제약이 있는 XR 기기에서 발생하는 높은 에너지 소비 문제에 대응합니다. 이 연구는 Lyapunov-guided Proximal Policy Optimization (LyaPPO) 알고리즘을 제안하며, 이를 통해 XR 기기의 에너지 소비를 24.79%에서 46.14%까지 절감했습니다.

- **Technical Details**: 연구에서 제안하는 방법은 듀얼 타임 스케일(joint optimization) 전략으로, 모델 파티셔닝(model partitioning)과 리소스 할당(resource allocation)을 결합하여 위계적 최적화 문제(bi-level optimization problem)를 구성합니다. 상위 수준에서는 DNN 파티셔닝 포인트를 조정하고, 하위 수준에서는 통신 및 계산 리소스를 최적화하여 대기열 안정성을 유지하며 시스템의 에너지 소비를 줄입니다. 이를 해결하기 위해 LyaPPO 알고리즘을 도입하여 하위 문제를 세분화하고 자원 할당을 최적화합니다.

- **Performance Highlights**: 시뮬레이션 결과에 따르면, 제안된 LyaPPO 알고리즘은 자원 용량의 변화에 따라 XR 기기의 에너지 소비를 각각 46.14%, 29.10%, 24.79% 감소시켰습니다. 이 알고리즘은 전반적으로 XR 기기와 MEC의 리소스 할당을 균형 있게 조정하여 성능을 최적화합니다. 최종적으로, 본 연구는 MEC 지원 XR 시스템에서 DNN 추론의 에너지 효율성을 높이는 방안을 제시합니다.



### Decoding fMRI Data into Captions using Prefix Language Modeling (https://arxiv.org/abs/2501.02570)
Comments:
          4 pages, 2 tables, 1 figure

- **What's New**: 최근 대형 언어 모델과 잠재 확산 모델의 발전으로 뇌 신호를 해독하는 연구가 눈에 띄는 성과를 거두었습니다. 본 연구에서는 기존 GIT 모델의 데이터 오염 가능성을 해결하기 위해 DINOv2 모델의 임베딩을 예측하여 뇌 신호를 이미지 캡션으로 변환하는 새로운 방법을 제안합니다. 이 접근 방식은 GPT-2 언어 모델의 입력으로 [CLS] 토큰을 사용하여 계산 요구 사항을 크게 줄입니다.

- **Technical Details**: 우리는 fMRI 신호에서 DINOv2의 임베딩을 직접 예측하는 새로운 방법을 채택하고, 3D Convolutional Neural Networks (CNN)를 통해 고차원 데이터를 처리합니다. 이를 통해 ROI 마스크 외부의 정보와 복셀 간의 포지셔널 정보를 보다 잘 고려할 수 있습니다. 각 모듈은 별도로 학습되며, DINOv2 임베딩은 MSE 손실을 사용하여 진행됩니다.

- **Performance Highlights**: 우리의 접근법은 기존의 COCO 캡션과 비교하여 METEOR 메트릭에서 우수한 성능을 보여주었으며, 다른 이미지 임베딩으로부터 생성된 캡션과 비교했을 때 6가지 메트릭 중 4가지에서 우수한 결과를 기록하였습니다. 또한, Ridge Regression보다 Wide CNN 아키텍처의 성능이 모든 메트릭에서 우수함을 확인하여 뇌 신호의 해독 효율성을 제고했습니다.



### Balanced Multi-view Clustering (https://arxiv.org/abs/2501.02564)
- **What's New**: 이번 연구에서는 다중 뷰 클러스터링(multi-view clustering, MvC)의 새로운 균형 잡힌 접근법인 BMvC(balanced multi-view clustering)를 제안합니다. 기존의 joint training paradigm의 문제점으로는 특정 뷰의 정보가 학습 과정에서 지배적이 되어 다른 뷰가 제대로 최적화되지 않는 imbalance 현상을 지적하였습니다. 이러한 문제를 해결하기 위해, 각 뷰별 특징 추출기의 최적화를 조절하는 View-Specific Contrastive Regularization(VCR)을 도입하여 클러스터링 과정에서의 성능을 향상시키고자 합니다.

- **Technical Details**: BMvC는 VCR을 통해 클러스터링 분포를 조정하며, 이로 인해 뷰별 특징 추출기가 보다 균형 잡힌 방식으로 학습될 수 있도록 지원합니다. VCR은 통합된 특징과 뷰별 특징에서 캡처한 샘플 유사성을 보존하며, 이를 통해 각 뷰의 그래디언트 크기를 조절하여 최적화 과정을 개선합니다. 이러한 기술적 접근은 클러스터링 성과를 높이며, 다중 뷰 정보를 완전히 활용하는데 기여합니다.

- **Performance Highlights**: BMvC는 8개의 기준 MvC 데이터셋과 2개의 공간적으로 분해된 전사체 데이터셋에서 최신 기법들과 비교해 우수한 성능을 입증하였습니다. 실험 결과, BMvC는 뷰별 특징 추출기로부터 얻어진 클러스터링 성능이 향상되었음을 보여주며, 기존의 joint training 및 single-view 훈련 모델보다 나은 결과를 얻었습니다. 이러한 성과는 BMvC의 클러스터링 성능 개선이 실질적이라는 것을 시사합니다.



### KM-UNet KAN Mamba UNet for medical image segmentation (https://arxiv.org/abs/2501.02559)
- **What's New**: 이 논문에서는 의료 이미징 분석에서 중요한 의료 이미지 분할을 위한 새로운 네트워크 구조인 KM-UNet을 제안합니다. KM-UNet은 Kolmogorov-Arnold Networks (KANs)와 상태 공간 모델(state-space models, SSMs)의 장점을 결합하여 수치적 효율성과 정확도의 균형을 이룹니다. 이를 통해 기존 CNN 기반 방법과 Transformer 기반 모델의 한계를 극복하고자 합니다.

- **Technical Details**: KM-UNet은 Kolmogorov-Arnold 표현 정리를 활용하여 효율적인 특성 표현(feature representation)을 수행하며, SSM을 통해 확장 가능한 장기 모델링을 가능하게 합니다. 이러한 U자형 구조는 의료 이미지 분할 페이즈에서의 성능을 극대화할 수 있도록 설계되었습니다. 실험을 위해 ISIC17, ISIC18, CVC, BUSI, GLAS의 다섯 가지 벤치마크 데이터셋을 사용하였습니다.

- **Performance Highlights**: 실험 결과, KM-UNet은 기존 최고의 방법들과 비교했을 때 경쟁력 있는 성능을 달성하였습니다. 이는 의료 이미지 분할 작업에서 KAN과 SSM을 통합한 최초의 프레임워크임을 밝히는 새로운 통찰을 제공합니다. 이 연구는 더 효율적이고 해석 가능한 의료 이미지 분할 시스템 개발을 위한 귀중한 기준을 제공합니다.



### AMM: Adaptive Modularized Reinforcement Model for Multi-city Traffic Signal Contro (https://arxiv.org/abs/2501.02548)
- **What's New**: 본 논문은 교통 신호 제어(Traffic Signal Control, TSC) 문제에 적용할 수 있는 새로운 접근법인 Adaptive Modularized Model (AMM)을 제안합니다. AMM은 환경 관찰의 변화를 극복하고 여러 도시의 데이터를 활용하여 모델 훈련 비용을 줄이는 데 중점을 둡니다. 이 방법은 기존의 TSC 도메인 적응 방법에서 발생하는 두 가지 주요 문제를 해결합니다: 서로 다른 도시 간의 관찰 차이를 고려하지 않는 점과 다수의 도시 데이터를 저조하게 활용하는 점입니다.

- **Technical Details**: AMM은 TSC 문제와 신경망 모델을 세 가지 모듈로 분리합니다: representation transition module, internal dynamics module, 그리고 value-evaluating module. 이 모듈들은 서로 결합되지 않아 서로 다른 환경에서 관찰이 변경될 경우 필요한 모듈만 교체하여 훈련 데이터를 크게 줄일 수 있습니다. 또한 새로운 환경에서 직접 상호작용하는 대신, 메타 학습(meta-learning) 기법을 통해 다수의 도시 경험을 집계하여 훈련 효율성을 높입니다.

- **Performance Highlights**: AMM을 사용한 광범위한 실험 결과, 제한된 상호작용으로도 뛰어난 성능을 발휘하며 기존 방법보다 우수한 결과를 보였습니다. 이 연구는 다수의 출처 도시에서의 TSC 도메인 적응 문제의 해결책을 제안하고, 이 접근 방식의 실용성과 범용성을 입증했습니다. AMM의 설계가 실제 환경에서의 교통 신호 제어에 미치는 긍정적인 영향을 확인할 수 있었습니다.



### TreeMatch: A Fully Unsupervised WSD System Using Dependency Knowledge on a Specific Domain (https://arxiv.org/abs/2501.02546)
- **What's New**: 이 논문은 Word Sense Disambiguation (WSD), 즉 단어 의미의 구분을 위한 새로운 시스템인 TreeMatch를 소개하고 있습니다. 이 시스템은 SemEval 2007 Task 7의 데이터를 사용하여 처음 개발되었고, SemEval 2010 Task 17의 특정 도메인에 맞게 조정되었습니다. TreeMatch는 특정 도메인 지식 기반에서 얻은 의존성 지식을 활용한 완전 비지도 학습 방법에 기반하고 있습니다.

- **Technical Details**: TreeMatch는 특정 도메인을 위한 지식 기반을 구축하여 시스템의 성능을 극대화합니다. 이 시스템은 비지도 학습(unsupervised learning) 방식을 사용하여 단어 의미를 구분하며, 기존의 Most Frequent Selection 기법보다 더 나은 정확성을 보여줍니다. 또한, 의존성 지식(dependency knowledge)을 통해 단어 간의 관계를 파악하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 평가 결과, TreeMatch 시스템은 Most Frequent Selection 기준선(baseline)보다 높은 정확도(precision)를 기록하였습니다. 이는 이 시스템이 기존의 방식보다 효과적으로 단어 의미를 구분할 수 있음을 보여줍니다. 이러한 성능 향상은 고유한 도메인 정보와 비지도 학습을 통한 의존성 분석의 결과로 나타났습니다.



### A completely uniform transformer for parity (https://arxiv.org/abs/2501.02535)
Comments:
          4 pages

- **What's New**: 이번 논문에서는 입력 길이에 의존하지 않는 3계층 상수 차원 transformer를 구성하여 parity 언어를 인식하는 방법을 제안합니다. 이는 Chiang과 Cholak의 연구를 개선한 결과로, 그들의 방법은 2계층 transformer를 이용하였으며, positional encoding이 입력 길이에 따라 달라지는 단점을 가졌습니다. 이번 연구는 parameter matrices와 positional encoding이 모두 입력 길이에 무관하게 설계되어 uniform한 형태임을 보여줍니다.

- **Technical Details**: 이 논문은 주목(attention) 계층의 차원 d에 기반하여 새로운 transformer 구조를 제안합니다. 제안된 모델은 C 계층의 attention layer와 위치 인코딩(position encoding)을 사용하지 않고도 입력 길이에 관계없이 parity 언어를 인식할 수 있습니다. 이 논문에서는 특정 입력 길이 n에도 불구하고 transformer가 어떻게 각 위치에 대해 f(n)을 계산하는지를 보여줍니다.

- **Performance Highlights**: 새로 개발된 3계층 transformer는 기존 연구에서 제기된 문제점을 해결하며, parity 언어 인식의 가능성을 제시합니다. 특히, 참조된 하한 방법들이 1계층 transformer와의 관련성을 증명하지 못함에도 불구하고, 제안된 구조를 통해 okul와 같은 표준 테스트에서 향상된 성능을 보일 가능성이 있습니다. 이러한 혁신성은 transformer 아키텍처의 이론적 한계를 탐구하는 데 도움이 될 것입니다.



### Evaluating Large Language Models Against Human Annotators in Latent Content Analysis: Sentiment, Political Leaning, Emotional Intensity, and Sarcasm (https://arxiv.org/abs/2501.02532)
Comments:
          24 pages, 3 figures

- **What's New**: 이번 연구는 대규모 언어 모델(Large Language Models, LLMs)의 성능을 인적 주석자(human annotators)와 비교하는 종합적인 평가를 수행했다. OpenAI의 GPT-4, Gemini, Llama 및 Mixtral의 변형을 포함하여 7가지 최첨단 LLM을 분석하여, 감정 분석(sentiment analysis) 및 정치적 편향(political leaning) 평가에서의 신뢰성(reliability)과 일관성(consistency)을 측정하였다. 이러한 접근법은 디지털 커뮤니케이션 시대에 필요한 효율적인 잠재 콘텐츠 분석(latent content analysis) 방법의 가능성을 제시한다.

- **Technical Details**: 연구에서는 총 33명의 인적 주석자와 8개 LLM 변형이 100개의 선별된 텍스트 항목을 평가하였다. 이에 따라 3,300개의 인적 주석과 19,200개의 LLM 주석이 생성되었으며, LLM의 성능은 세 가지 시점(time points)에서 평가하여 시간에 따른 일관성을 살펴보았다. Krippendorff의 알파(Krippendorff's alpha)를 통해 주관자 간 신뢰성을 측정하고, 클래스 내 상관 계수(intra-class correlation coefficients)는 시간에 따른 일관성을 평가하였다.

- **Performance Highlights**: 연구 결과, 감정 분석 및 정치적 편향의 평가에서 LLM과 인간 모두 높은 신뢰성을 보였다. LLM은 인간보다 더 높은 내부 일관성(internal consistency)을 보였으며, 감정 강도(emotional intensity)에서는 LLM이 더 높은 일치를 나타내지만, 인간은 감정 강도를 더 높게 평가하였다. 반면, 조롱 감지(sarcasm detection)에서는 두 그룹 모두 낮은 일치를 보였으며, LLMs는 모든 차원에서 훌륭한 시간적 일관성(temporal consistency)을 보여줬다.



### Face-MakeUp: Multimodal Facial Prompts for Text-to-Image Generation (https://arxiv.org/abs/2501.02523)
- **What's New**: 이 연구는 새로운 얼굴 이미지 생성을 위한 Face-MakeUp 모델을 제안하며, 400만 개의 고품질 얼굴 이미지-텍스트 쌍으로 이루어진 FaceCaptionHQ-4M 데이터세트를 구축하였습니다. 텍스트 프롬프트만으로는 복잡한 얼굴 이미지를 생성하는 데 한계가 있기 때문에, 이미지 프롬프트를 활용하여 더 나은 성능을 달성하고자 합니다.

- **Technical Details**: Face-MakeUp 모델은 얼굴 이미지의 다중 스케일 콘텐츠 특징과 포즈 특징을 추출하여 이러한 정보를 diffusion 모델에 통합하여 얼굴 정체성을 보존하기 위해 특징 공간을 확장하는 구조로 설계되었습니다. ArcFace 모델을 활용하여 개별 얼굴 특징을 극대화하고, PoseNet을 통해 포즈 정보를 효과적으로 통합하였습니다.

- **Performance Highlights**: 실험 결과, Face-MakeUp 모델은 다양한 얼굴 관련 테스트 데이터세트에서 최고의 종합적인 성능을 달성하였고, 특히 얼굴의 정체성을 개선하는 데 있어 눈에 띄는 효과를 보여주었습니다. 이 모델은 오픈 소스 자원으로 공개되어 향후 연구 및 실험에 기여할 것으로 기대됩니다.



### Remote Inference over Dynamic Links via Adaptive Rate Deep Task-Oriented Vector Quantization (https://arxiv.org/abs/2501.02521)
Comments:
          13 pages, 12 figures

- **What's New**: 이 논문에서는 동적 링크를 통해 원거리 추론을 위한 적응형 압축 메커니즘인 Adaptive Rate Task-Oriented Vector Quantization (ARTOVeQ)를 제안합니다. ARTOVeQ는 기존의 정적 압축 방식의 한계를 극복하며, 다양한 채널 상황에서도 유연하게 적응할 수 있는 압축 방법입니다. 이 방법은 내부 코드북을 설계하고 점진적 학습(progressive learning) 알고리즘을 사용하여 저지연(low-latency) 추론을 지원합니다.

- **Technical Details**: ARTOVeQ는 중첩 코드북(nested codebooks)과 훈련 가능한 적응형 벡터 양자화(trainable adaptive vector quantization) 방식을 기반으로 합니다. 이 방식은 단일 코드북을 통해 여러 비트율(multiple rates)로 압축 및 추론을 수행합니다. 또한, 이 모델은 채널 조건의 변화에 따라 압축 비율을 조정하며, 각기 다른 해상도(resolutions)를 동시에 사용하여 고차원 데이터를 전달할 수 있게 합니다.

- **Performance Highlights**: 제안된 ARTOVeQ는 높은 성능의 원거리 심층 추론을 지원하며, 다양한 비트 예산(bit budgets)에서 작동할 수 있습니다. 수치적 결과는 압축 비율이 증가함에 따라 단계적으로 개선되는 빠른 추론을 가능하게 하여 단일 비율 심층 양자화(single-rate deep quantization) 방법의 성능에 가까워지는 것을 보여줍니다. 이러한 성능 향상은 다채로운 데이터 전송 환경에서도 일관된 신뢰성을 제공합니다.



### PTEENet: Post-Trained Early-Exit Neural Networks Augmentation for Inference Cost Optimization (https://arxiv.org/abs/2501.02508)
- **What's New**: 이번 연구에서는 Deep Neural Network (DNN)의 피드포워드 추론 과정에서 비용이 많이 드는 계산을 건너뛰는 'shortcuts'를 도입하는 방법을 제안합니다. 기존의 BranchyNet과 EEnet 아키텍처를 바탕으로 하여, 사전 훈련된 모델에 가지를 부착하여 원본 네트워크의 가중치를 변경하지 않고도 성능을 개선할 수 있습니다. 또한, 높은 DNN을 적용할 수 있는 충분한 훈련 용량을 제공하는 새로운 가지 아키텍처를 제안하며, 신뢰성 수준을 예측하기 위한 confidence heads를 포함합니다.

- **Technical Details**: 본 연구에서 제안하는 방법은 DNN의 중간 레이어에서 출구의 신뢰성을 판단하는 classification head와 decision head를 통해 이루어집니다. 이때 각 가지는 예측된 신뢰성 수준에 따라 계산을 계속 진행할지 결정하게 되며, 설정된 임계값에 따라 처리가 조절됩니다. 여러 DNN 아키텍처(ResNet, DenseNet, VGG)에서 이미지 데이터셋(SVHN 및 CIFAR10)으로 이 방법의 유효성을 검증하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 평균적인 추론 계산 비용을 줄이면서 모델의 정확성과 계산 비용 사이의 균형을 효과적으로 제어할 수 있음을 보여줍니다. 이를 통해 처리 속도와 정확성을 직관적으로 조절할 수 있어, 자원이 한정된 환경에서도 원활하게 DNN을 활용할 수 있는 가능성을 제시합니다.



### Watch Video, Catch Keyword: Context-aware Keyword Attention for Moment Retrieval and Highlight Detection (https://arxiv.org/abs/2501.02504)
Comments:
          Accepted at AAAI 2025

- **What's New**: 본 논문에서는 비디오 순간 검색(video moment retrieval)과 하이라이트 탐지(highlight detection)에서 키워드 변동을 파악할 수 있는 새로운 모듈인 Video Context-aware Keyword Attention을 제안합니다. 기존의 방법들이 비디오의 맥락을 제대로 반영하지 못하는 문제를 해결하기 위해, 전체 비디오 맥락을 고려하는 비디오 맥락 클러스터링 모듈을 도입합니다. 이를 통해 사용자의 자연어 쿼리에 따라 각각의 단어의 중요성이 달라질 수 있는 점을 잘 반영하게 됩니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 도전에 대응합니다. 첫째, 비디오의 전체 맥락을 효과적으로 인코딩하여 키워드의 변동성을 포착하는 것이며, 둘째, 텍스트 키워드를 원하는 비디오 맥락 내에서 캡처하고 활용하는 것입니다. 이를 위해, 우리는 시기적으로 가중된 클러스터링 기법을 통해 유사한 비디오 장면들을 그룹화하고, 이 클러스터 정보를 통해 키워드 다이내믹스를 이해합니다. 또한 키워드 인지 대조 학습(keyword-aware contrastive learning) 모듈을 통해 시각적 및 텍스트 특징 간의 정밀한 정렬을 가능하게 합니다.

- **Performance Highlights**: QVHighlights, TVSum 및 Charades-STA 벤치마크에 대한 광범위한 실험 결과, 제안된 방법은 기존의 접근 방식들에 비해 순간 검색과 하이라이트 탐지 작업에서 성능이 유의미하게 향상됨을 보였습니다. 특히, 키워드와 비디오 맥락의 관계를 보다 정교하게 이해함으로써 높은 정확도를 달성한 점이 강조됩니다. 이는 다양한 비디오 콘텐츠에 대한 접근성과 사용자 경험을 크게 개선하는 효과를 기대할 수 있습니다.



### Rethinking IDE Customization for Enhanced HAX: A Hyperdimensional Perspectiv (https://arxiv.org/abs/2501.02491)
Comments:
          Accepted at the 2nd Workshop on Integrated Development Environments (the IDE Workshop) co-located with ICSE '25

- **What's New**: 이 논문에서는 인공지능(AI)이 통합된 개발 환경(IDE)에서 사용자 행동과 선호를 모델링하기 위해 Hyper-Dimensional (HD) 벡터 공간을 제안합니다. AI로 생성된 코드와 개발자의 선호 간의 불일치를 해결하기 위한 새로운 접근법을 제시하며, 개발자의 코드 스타일을 자동으로 조정하는 방법에 대한 연구를 장려합니다.

- **Technical Details**: 논문에서는 HDC(Hyper-Dimensional Computing) 이론을 기반으로 하여 Multiply Add Permute (MAP) 프레임워크를 적용합니다. 이 프레임워크는 하이퍼 차원 벡터를 세 가지 주요 연산인 곱셈(multiplication), 덧셈(addition), 순열(permutation)으로 조작하며, 코드 작성을 위한 사용자 선호 및 행동을 벡터로 표현하고 조작하는 방법을 설명합니다.

- **Performance Highlights**: HDC는 IDE 내에서 사용자 행동의 예측을 가능하게 하고, 사용자 맞춤형 코드 스타일을 유지하면서 프로젝트의 맥락을 이해할 수 있도록 도와줍니다. 이로써 IDE는 개발자의 개인적인 코드 작성을 지원하고, 프로젝트에 적합한 제안을 할 수 있는 가능성을 보여줍니다.



### The Meta-Representation Hypothesis (https://arxiv.org/abs/2501.02481)
- **What's New**: 이번 논문은 메타 표현(meta-representation)과 일반화(generalization) 사이의 연결 고리를 제시합니다. 저자들은 메타 표현 학습(meta-representation learning)이 일반화 성능을 향상시킬 수 있다는 것을 보여주었으며, 심층 상호 학습(deep mutual learning, DML)이 에이전트들이 메타 표현을 수렴하도록 돕는다는 가설을 제안했습니다. 이러한 결과는 강화 학습 분야에서의 새로운 시각을 제공합니다.

- **Technical Details**: 메타 표현은 사물에 대한 고차원 표현입니다. 강화 학습에서는 여러 Markov Decision Processes (MDPs)가 기본 MDP를 공유하며, DML 기법을 통해 에이전트들이 소음 관측에서 메타 표현을 학습할 수 있도록 돕습니다. MDP의 정의와 강화 학습 일반화에 대한 기초적인 개념도 다루어졌습니다.

- **Performance Highlights**: 실험 결과, DML 기법이 일반화 성능을 일관되게 향상시킨다는 강력한 지원 자료가 제공되었습니다. 저자들은 관련 특성에 대한 정책의 강인함을 개선하면 일반화 성능이 향상된다는 이론적 증거를 제시하였으며, 이는 기존의 직관적인 이해를 넘어서는 걸음입니다.



### Hengqin-RA-v1: Advanced Large Language Model for Diagnosis and Treatment of Rheumatoid Arthritis with Dataset based Traditional Chinese Medicin (https://arxiv.org/abs/2501.02471)
Comments:
          8 pages, 5 figures, AAAI-2025 Workshop

- **What's New**: 본 논문은 LLMs (Large Language Models)에서 나타나는 중국 맥락의 편향과 부정확성을 극복하기 위해, 류마티스 관절염(RA) 진단 및 치료를 위해 특별히 설계된 첫 번째 대형 언어 모델인 Hengqin-RA-v1을 발표합니다. 또한 고대 중국 의학 문헌과 현대 임상 연구를 바탕으로 한 RA 전용 데이터셋 HQ-GCM-RA-C1을 제시하여, 문화적 맥락을 고려한 정확한 응답을 가능하게 합니다. 이를 통해 기존 모델에서 발생하는 격차를 효과적으로 메꿉니다.

- **Technical Details**: Hengqin-RA-v1은 LLaMA-7B (Touvron et al. 2023) 기반으로 개발된 Huatuo2 (Zhang et al. 2023)의 고급 버전입니다. 이 모델은 중국 의학 지식 그래프(CMeKG) 및 GPT-3.5로 생성된 의료 지침 데이터를 이용해 교육받으며, RA 치료 및 진단에 특화된 문제 해결 능력을 향상시키는 데 초점을 맞추고 있습니다. 훈련 과정에서는 의학 기록을 구조화하여 TCM의 진단 및 치료 로직을 강화하는 방법이 사용됩니다.

- **Performance Highlights**: Hengqin-RA-v1은 RA 관련 진단 및 치료 정보를 생성하는 데 있어 현재의 최고 수준 모델들을 초월하는 성능을 보여줍니다. 일부 경우에는 인간 전문가의 진단 정확도를 초과하는 성취를 달성하여, TCM 분야에서의 발전을 더욱 두드러지게 했습니다. 또한 이 모델은 RA 관련 연구 및 응용 분야에서 잠재적인 혁신을 촉진할 것으로 기대됩니다.



### Depth Any Camera: Zero-Shot Metric Depth Estimation from Any Camera (https://arxiv.org/abs/2501.02464)
- **What's New**: 이 논문은 Depth Any Camera (DAC)라는 새로운 제로샷 메트릭 깊이 추정 프레임워크를 제시합니다. 이 프레임워크는 다양한 시야각(Field of View, FoV)을 가진 카메라, 특히 어안(Fisheye) 및 360도 카메라에 대한 깊이 추정을 가능하게 합니다. DAC는 일반적으로 관찰 가능한 3D 데이터 활용을 극대화하는 방식으로 설계되어 있으며, 기존의 관점 데이터로 훈련된 모델을 기반으로 합니다.

- **Technical Details**: DAC는 Equi-Rectangular Projection (ERP)을 통합 이미지 표현으로 사용하여 여러 FoV 카메라에서 이미지를 일관되게 처리합니다. 주요 기술 중 하나는 ERP 공간에서의 온라인 증강을 위한 피치 감지(Image-to-ERP) 변환이며, 이는 다양한 FoV 간의 효과적인 훈련을 위해 FoV 정렬 작업을 포함합니다. 이 외에도 훈련 이미지 크기의 차이를 관리하기 위해 멀티 해상도 데이터 증강 기법을 도입하고 있습니다.

- **Performance Highlights**: DAC는 여러 피쉬아이 및 360도 데이터셋에서 이전 메트릭 깊이 모델에 비해 최대 50%의 개선된 δ1 정확도를 달성했습니다. 이 결과는 다양한 카메라 유형에 걸쳐 강력한 일반화 능력을 보여줍니다. DAC는 모든 대형 FoV 테스트 데이터셋에서 최신 제로샷 성능을 달성하여 깊이 추정 분야의 새로운 기준을 세우고 있습니다.



### FedRSClip: Federated Learning for Remote Sensing Scene Classification Using Vision-Language Models (https://arxiv.org/abs/2501.02461)
- **What's New**: 본 논문에서는 기존의 Vision-Language Models (VLM)인 CLIP을 기반으로 한 최초의 원격 센싱 이미지 분류를 위한 연합 학습 프레임워크인 FedRSCLIP을 제안합니다. FedRSCLIP은 데이터 이질성과 대규모 모델 전송의 문제를 해결하기 위해 Prompt Learning을 도입하여 적은 수의 조정 가능한 매개변수만을 최적화함으로써 통신 비용을 크게 절감합니다. 또한, 클라이언트에 맞춤형 대응이 가능한 Private Prompts와 전 세계 지식을 공유하는 Shared Prompts를 기반으로 하는 이중 프롬프트 메커니즘을 도입하여 정보의 효율적인 활용이 가능합니다.

- **Technical Details**: FedRSCLIP은 Dual Prompt Alignment Constraint를 활용하여 Shared Prompts와 Private Prompts 간의 의미 일관성을 유지합니다. 이로 인해 각 클라이언트의 로컬 데이터에 적응하면서도 전역 지식과의 일관성을 보장할 수 있습니다. 추가로 Cross-Modal Feature Alignment Constraint를 도입하여 텍스트와 이미지 프롬프트 간의 다중 모드 특성을 정렬함으로써 크로스 모달 표현 학습을 강화하고 전체 모델의 표현의 일관성을 높입니다.

- **Performance Highlights**: FedRSCLIP의 효과를 검증하기 위해 Optimal-31, UCMerced 및 NWPU의 세 가지 기존 데이터 세트를 기반으로 Fed-RSIC 데이터 세트를 구축하여 다양한 연합 학습 구성을 시뮬레이션합니다. 실험 결과, FedRSCLIP은 다양한 연합 학습 구성에서 원격 센싱 이미지 분류의 최첨단 성능을 달성하였으며, 이는 다양한 클라이언트 분포에 걸쳐 안정적이고 일반화된 학습을 가능하게 합니다.



### Enhancing Contrastive Learning for Retinal Imaging via Adjusted Augmentation Scales (https://arxiv.org/abs/2501.02451)
- **What's New**: 이 논문은 대조 학습(Contrastive Learning)이 의료 영상(Medical Imaging) 도메인에서 느린 성능을 보이는 이유를 탐구하고 있습니다. 기존의 연구들은 대조 학습이 자연 이미지(domain)에서는 성공적이었으나, 의료 이미지는 고유한 데이터 분포로 인해 이런 성공이 재현되지 않는다는 점을 강조합니다. 논문은 또한 의료 이미지에서 긍정 및 부정 쌍(pair) 구축이 이루어지는 방법을 분석하고 이를 개선하는 방안을 제시합니다.

- **Technical Details**: 대조 학습은 명시적 레이블 없이 유사한 데이터 포인트를 구별하는 기계 학습 패러다임입니다. 본 연구에서는 약한 증강(augmented) 기법을 이용하여 모델을 사전 훈련(pre-trained)하는 방안을 제안하며, 여러 임상 데이터셋을 통해 모델의 일반화(generalization) 성능을 평가합니다. 이 과정에서 두 가지 타입의 데이터 증강(augmentation)을 정의하고 긍정 쌍과 부정 쌍을 생성합니다.

- **Performance Highlights**: 저자들은 약한 증강으로 사전 훈련된 모델이 강한 증강의 경우보다 더 나은 성능을 보임을 확인했습니다. MESSIDOR2 데이터셋에서는 AUROC(Area Under Receiver Operating Characteristic) 값이 0.838에서 0.848로, AUPR(Area Under Precision-Recall Curve) 값이 0.523에서 0.597로 증가하여 강화된 정확도를 보여주었습니다. 이러한 결과는 의료 영상 분야에서의 대조 학습의 효율성을 극대화하기 위해 증강 기법의 최적화가 필수적임을 시사합니다.



### RTLMarker: Protecting LLM-Generated RTL Copyright via a Hardware Watermarking Framework (https://arxiv.org/abs/2501.02446)
- **What's New**: 이 논문은 LLM(대형 언어 모델)으로 생성된 RTL(레지스터 전송 수준) 코드의 저작권 보호를 위한 새로운 하드웨어 워터마킹 프레임워크인 RTLMarker를 제안합니다. RTLMarker는 RTL 코드와 그 응용 네트리스트에 워터마크를 안전하고 효과적으로 삽입하도록 설계되었습니다. 이 방법은 기존 워터마킹 시스템의 한계를 극복하여 RTL 디자인 과정에서의 저작권 문제를 해결할 수 있는 가능성을 보여줍니다.

- **Technical Details**: RTLMarker는 코드의 문법(syntactic) 및 의미적(semantic) 올바름을 보장하는 규칙 기반의 Verilog 코드 변환을 통해 워터마크를 삽입합니다. 또한 워터마크의 투명성(transparency)과 효과성(effectiveness) 간의 내재적 무역오프를 고려하고 이를 최적화합니다. 주요 구성 요소로는 워터마크 삽입 모듈, 특징 표현 모듈, 워터마크 탐지 모듈이 포함되어 있습니다.

- **Performance Highlights**: 성능 평가 결과, RTLMarker는 기존 모델에 비해 RTL 코드의 워터마킹 정확도에서 우수한 성능을 보였습니다. 본 연구는 LLM이 생성한 RTL 코드의 저작권 보호를 위한 실용적이고 효율적인 워터마킹 프레임워크를 단독으로 제시한 최초의 노력으로, Pyverilog를 활용하여 15가지 Verilog 전용 코드 변환기를 개발했습니다.



### A Statistical Hypothesis Testing Framework for Data Misappropriation Detection in Large Language Models (https://arxiv.org/abs/2501.02441)
Comments:
          29 pages, 5 figures

- **What's New**: 이 연구는 최근의 큰 언어 모델(LLM)들이 훈련 데이터에 저작권 자료를 무단으로 포함하는 문제를 다룹니다. 저작권 있는 훈련 데이터에 워터마크(embedding watermarks)를 삽입하여 데이터 유용성을 탐지하는 새로운 방법론을 제시합니다. 제안된 방법은 가설 검정(hypothesis testing) 문제로 프레임화되어, 잘못된 데이터 이용 감지를 수학적으로 다루는 혁신적인 접근 방식을 포함하고 있습니다.

- **Technical Details**: 연구에서는 통계적 테스트 프레임워크(statistical testing framework)를 기반으로 한 가설 검정 방법을 개발합니다. 두 가지 가설, 즉 워터마크가 있는 데이터를 사용하지 않은 경우와 있는 경우에 따른 토큰(token)과 비밀키(secret key)의 의존성을 평가합니다. 배경 지식으로 활용되는 NTP(다음-토큰 예측) 분포를 두고, 파생된 통계량을 정립하고 최적의 기각 임계값(rejection threshold)을 설정하여 제1종 및 제2종 오류를 제어합니다.

- **Performance Highlights**: 이 방법론은 실제 데이터 세트를 통해 수치 실험(numerical experiments)으로 효과성을 입증하고, 워터마크 데이터를 사용한 LLM 훈련 내의 데이터 유용성을 식별하는데 높은 정확도를 보입니다. 연구는 특히 부분 상속(partial inheritance)에 대한 최적성 보장을 새롭게 설정하여, 기존 연구에서 미처 다루지 못했던 부분을 설명합니다. 제안된 검정 방법은 모든 다른 테스트 방법에 비해 가장 높은 검정력을 달성하는 것으로 나타났습니다.



### Efficient Deployment of Large Language Models on Resource-constrained Devices (https://arxiv.org/abs/2501.02438)
- **What's New**: 이 논문은 FedSpine이라는 새로운 Federated Learning (FL) 프레임워크를 제안하여 리소스가 제한된 기기에서의 대규모 언어 모델(LLM) 배포 문제를 해결합니다. FedSpine은 파라미터 효율적인 미세 조정(PEFT)과 구조적 프루닝(structured pruning)을 결합하여 성능 저하 없이 더 빠르고 효율적인 미세 조정을 가능하게 합니다. 또한, 온라인 Multi-Armed Bandit (MAB) 알고리즘을 사용하여 기기의 자원 소모 및 성능 간의 관계를 적응적으로 학습합니다.

- **Technical Details**: LLM의 미세 조정 과정에서 FedSpine은 각 기기가 로컬 데이터로 사전 훈련된 모델을 미세 조정하도록 하며, 주기적으로 서버에서 업데이트된 LoRA 가중치를 다운로드합니다. 이 과정에서 기기 간의 이질성을 고려하여, 각 기기별로 적절한 프루닝 비율과 LoRA 랭크를 할당하여 미세 조정 효율을 극대화합니다. FedSpine은 기기가 제한된 자원 상황에서도 높은 추론 정확도를 유지하는 데 도움을 줍니다.

- **Performance Highlights**: 실험 결과, FedSpine은 80개의 NVIDIA Jetson 기기를 활용한 플랫폼에서 기존 방법에 비해 미세 조정을 1.4배에서 6.9배까지 속도 향상시키고, 최종 정확도는 0.4%에서 4.5%까지 개선했습니다. 이러한 성능 향상은 특히 리소스가 제한된 기기에서의 LLM 활용을 더욱 용이하게 만들어, 다양한 다운스트림 작업에 적합한 모델을 제공하는 데 기여합니다.



### Interpretable Neural ODEs for Gene Regulatory Network Discovery under Perturbations (https://arxiv.org/abs/2501.02409)
- **What's New**: 이 논문에서는 PerturbODE라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 생물학적으로 유용한 신경 일반 미분 방정식(neural ordinary differential equations, Neural ODEs)을 활용하여, perturbation(교란) 하의 세포 상태 경로(cell state trajectories)를 모델링합니다. 기계학습 방법이 아닌 기존 모델들보다 더 진화된 접근법으로, 기존의 단점들을 개선하고자 합니다.

- **Technical Details**: PerturbODE는 세포 상태 경로를 예측하기 위해 신경 ODE의 매개변수를 사용하여 인과 유전자 조절망(causal gene regulatory network, GRN)을 도출합니다. 이 방법은 기존의 가정(예: 선형성(linearity), 비순환성(acyclicity))에 얽매이지 않고, 생물학적 과정의 동적 특성을 반영하여 보다 현실적인 모델링을 제공합니다. 이 모델은 시뮬레이션 및 실제 데이터에서 성능을 입증하였습니다.

- **Performance Highlights**: PerturbODE는 궤적 예측(trajectory prediction) 및 GRN 추론에서 탁월한 효능을 보여줍니다. 제안된 방법은 기존 모델들보다 더 높은 확장성(scalability)을 갖추고 있으며, 생물학적으로 복잡한 상황에서도 더욱 유용하게 사용될 수 있습니다.



### Who Wrote This? Zero-Shot Statistical Tests for LLM-Generated Text Detection using Finite Sample Concentration Inequalities (https://arxiv.org/abs/2501.02406)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLM)에서 생성된 텍스트와 인간이 작성한 텍스트 사이를 구별하는 새로운 방법을 제시합니다. 포괄적인 위상에 기초한 통계적 테스트를 설계하여, 특정 텍스트가 내부 LLM A 또는 외부 LLM B(혹은 인간)에서 생성된 것인지 구별할 수 있는 방법을 탐구하고 있습니다. 이 연구에서 정의된 통계적 테스트는 이론적 보장을 제공합니다.

- **Technical Details**: LLM이 생성한 텍스트를 순차적 확률 과정으로 모델링하고, 주어진 텍스트에 대해 LLM A와 B로부터 생성되었는지를 판단하기 위한 제로샷(zero-shot) 통계 테스트를 설계합니다. 이 과정에서 우리는 log-perplexity와 모형 A의 문자열의 평균 엔트로피 간의 집중 경계를 도출하여, 텍스트의 길이가 증가함에 따라 오류 유형 I과 II가 기하급수적으로 감소함을 입증합니다. 또한, 생성된 텍스트의 출처를 고유하게 식별하는 것이 가능성을 보여줍니다.

- **Performance Highlights**: 실험 결과는 제안된 테스트가 이론적으로 보장된 정확성을 통해 LLM이 생성한 해로운 텍스트의 출처를 식별하는 데 도움을 줄 수 있음을 시사합니다. 세부적인 예는 텍스트 크기가 커질수록 정확도가 동시에 증가한다는 점에서, 이 연구는 반드시 필요한 도구로 자리잡을 것으로 기대됩니다. 이런 점에서, 해당 연구는 정보의 진위를 파악하고 변별력을 높이는 데 기여할 것입니다.



### iTARGET: Interpretable Tailored Age Regression for Grouped Epigenetic Traits (https://arxiv.org/abs/2501.02401)
Comments:
          To be published in IEEE BIBM this http URL manuscript includes a comprehensive description of the methodology and comparison with traditional epigenetic clocks and machine learning models. Submitted to arXiv as part of ongoing research in epigenetics and aging studies

- **What's New**: 이번 연구에서는 DNA 메틸화 패턴으로부터 연대기에 기반한 나이를 정확히 예측하는 새로운 두 단계 알고리즘을 제시했습니다. 첫 번째 단계는 유사성 검색을 활용하여 연령대별 메틸화 프로파일을 클러스터링하고, 두 번째 단계에서는 Explainable Boosting Machines (EBM)를 활용해 보다 정확한 예측을 수행합니다. 이를 통해 예측 정확도를 개선할 뿐만 아니라 주요 나이 관련 CpG 사이트를 식별하고, 노화 속도의 연령 특정 변화를 탐지할 수 있습니다.

- **Technical Details**: 이 연구는 Epigenetic Correlation Drift (ECD)와 Heterogeneity Among CpGs (HAC)의 두 가지 주요 도전 과제를 다루고 있습니다. DNA 메틸화 데이터 매트릭스는 각 샘플의 연령 정보를 포함하고 있으며, 나이 그룹별로 Pearson 상관 계수를 계산하여 가장 강한 상관관계를 가진 CpG 사이트를 선정합니다. 연구는 두 가지 연령 그룹핑 전략을 사용하며, 이는 나이에 따른 생물학적 및 단백질 변화와 일치합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 기존의 전통적인 에피제네틱 시계와 다른 해석 가능한 기계 학습 모델들보다 예측 정확성에서 뛰어난 성능을 보여주었습니다. 연구의 주된 기여는 나이에 따라 변화하는 메틸화 패턴을 보다 잘 포착할 수 있는 메커니즘을 제시하여 노화 연구에 중요한 시사점을 제공합니다. 이로 인해 생물학적 나이 추정의 해석 가능성과 신뢰성을 크게 향상시킵니다.



### Graph-Aware Isomorphic Attention for Adaptive Dynamics in Transformers (https://arxiv.org/abs/2501.02393)
- **What's New**: 이번 연구에서는 Transformer 아키텍처에 그래프 지향의 관계적 추론(graph-aware relational reasoning)을 통합하여 수정하는 접근 방법을 제안합니다. 주의(attention) 메커니즘을 그래프(graph) 연산으로 재정의하여 Graph-Aware Isomorphic Attention을 제안합니다. 이는 Graph Isomorphism Networks (GIN)와 Principal Neighborhood Aggregation (PNA)와 같은 고급 그래프 모델링 전략을 활용하여 관계 구조를 더 풍부하게 표현합니다.

- **Technical Details**: 우리의 접근법은 복잡한 의존성을 캡처하고 다양한 작업에 걸쳐 일반화(generalization)되는 것을 목표로 합니다. Sparse GIN-Attention이라는 미세 조정(fine-tuning) 방법을 도입하여 희소(의존성 그래프) GIN을 사용하고, 주의 행렬을 희소 인접 그래프(sparse adjacency graphs)로 해석하여 사전 훈련된 기본 모델의 적응력을 높입니다. 이 방법은 그래프 기반의 모델을 통해 연결은 강화되고, 이를 통해 이전보다 더 나은 훈련 동역학(training dynamics)과 일반화 성능을 달성합니다.

- **Performance Highlights**: Sparse GIN-Attention 미세 조정 방법은 기존의 저차원 적응(low-rank adaptation, LoRA) 방법보다 개선된 결과를 보입니다. 이 연구는 전통적인 주의 메커니즘 내에는 잠재적인 그래프 유사 구조가 존재함을 논의하며, Transformer를 계층적 GIN 모델로 발전시키는 관점을 제시합니다. 이는 기초 모델 개발에 대한 깊은 영향을 미치며, 지역(local) 및 글로벌(global) 의존성에 동적으로 적응할 수 있는 아키텍처 설계를 가능하게 합니다.



### Syntactic Evolution in Language Usag (https://arxiv.org/abs/2501.02392)
Comments:
          4 pages, 7 figures

- **What's New**: 이 연구는 생애 여러 단계에 걸친 언어 스타일의 동적 변화를 조사하는 것을 목표로 합니다. 2004년의 블로그 데이터를 사용하여 언어 사용이 시간이 지남에 따라 어떻게 변화하는지를 분석하고 있습니다. 연구 결과는 언어학(linguistics), 심리학(psychology), 그리고 커뮤니케이션 연구(communication studies)에 대한 통찰력을 제공할 수 있습니다.

- **Technical Details**: 연구 설계는 여러 단계로 이루어졌으며, 데이터 전처리(preprocessing), 문법적 특성(feature analysis), 그리고 GPT-4와 블로그 텍스트의 비교를 포함합니다. 문법적 요소(syntactic elements)의 다수의 비율과 비율을 분석하여 다양한 연령대에서의 차이를 명확히 하고자 하였습니다. 이 과정에는 OpenAI API를 통한 텍스트 생성 및 다양한 나이 그룹에 대한 특징을 분석하는 과정이 포함되었습니다.

- **Performance Highlights**: 결과적으로 블로그 텍스트의 문장 복잡성은 연령대가 증가함에 따라 증가하는 경향을 보였으며, GPT-4 텍스트에서도 거의 비슷한 변화를 관찰할 수 있었습니다. 그러나 나이 그룹 예측에서 모델 정확도가 낮아, GPT-4가 인간의 언어 진화에 대한 학습이 부족할 수 있음을 시사합니다. 또한, 다양한 데이터셋과 방법론이 필요함을 강조하며, 그런 이슈를 해결하기 위한 방안을 모색해야 한다고 결론지었습니다.



### Context Aware Lemmatization and Morphological Tagging Method in Turkish (https://arxiv.org/abs/2501.02361)
- **What's New**: 본 연구는 터키어에서 단어의 의미와 문맥에 민감한 형태소 분석과 표제어 추출(lemmatization) 모델을 제안합니다. 형태소 태깅 모델과 표제어 추출 모델은 두 가지 대안 모델로 구성되어 있으며, 이 모델들은 터키어의 단어 의미를 기반으로 예측을 수행합니다. 그간 터키어의 의미에 민감한 표제어 연구는 없었기에, 본 논문은 새로운 기여를 하고 있습니다.

- **Technical Details**: 제안된 모델은 bidirectional LSTM과 터키어 BERT 모델을 활용하여 각각 단어의 철자와 의미를 표현합니다. 단어의 철자를 생성하기 위해 단방향 LSTM이 사용됩니다. 데이터셋으로는 Universal Dependencies의 IMST 및 PUD 데이터셋이 활용되며, 모델 훈련 후 SIGMORPHON 2019 대회의 결과와 비교하였습니다.

- **Performance Highlights**: 모델은 IMST 및 PUD 데이터셋에서 모두 최고 성능을 달성하였으며, 거의 모든 평가 지표에서 우수한 결과를 보였습니다. 본 연구는 두 가지 모델의 성능을 비교하고, 문맥에 기반하여 의미를 반영한 새로운 표제어 추출 방법의 효용성을 입증하였습니다.



### GNSS/GPS Spoofing and Jamming Identification Using Machine Learning and Deep Learning (https://arxiv.org/abs/2501.02352)
- **What's New**: 최근 GNSS(Global Navigation Satellite Systems) 및 GPS(Global Positioning System)의 중요성이 더욱 부각됨에 따라, 이 기술들을 악의적인 위협으로부터 보호할 필요성이 커졌습니다. 본 논문은 스푸핑(spoofing) 및 재밍(jamming)이라는 두 가지 문제에 대해 머신러닝(machine learning)과 딥러닝(deep learning)을 활용하여 효과적인 탐지 및 완화 전략을 개발하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 논문에서는 스푸핑 및 재밍 공격에 대한 탐지를 위해 머신러닝 및 딥러닝 기법을 활용한 다양한 실험을 수행하였습니다. 실제 데이터셋을 기반으로 한 실험에서 최첨단 알고리즘을 사용하여 스푸핑과 재밍 탐지에서 각각 약 99%의 정확도를 달성하였으며, 이는 이전 연구에 비해 약 5% 향상된 결과입니다. 이를 통해 GNSS 시스템의 신뢰성을 높이는 데 기여했습니다.

- **Performance Highlights**: 스푸핑 및 재밍 탐지 과제에서의 성과는 머신러닝과 딥러닝 기술의 최신 발전을 통해 가능한 결과였습니다. 특히, 스푸핑 탐지 문제는 도전적인 과제로, 본 연구의 결과는 해당 분야에서 머신러닝과 딥러닝의 잠재력을 강조하는 중요한 진전을 나타냅니다. 이러한 연구 결과는 향후 GNSS 시스템의 보안 강화를 위한 기초 자료로 활용될 수 있습니다.



### Exploring the Capabilities and Limitations of Large Language Models for Radiation Oncology Decision Suppor (https://arxiv.org/abs/2501.02346)
Comments:
          Officially published in the Red Journal

- **What's New**: 최근 LLMs(대규모 언어 모델)의 발전이 의료 결정 지원 도구에 통합됨으로써 큰 변화가 일어나고 있습니다. 방사선 종양학에서도 GPT-4와 같은 LLM의 사용이 증가하고 있으며, 이는 다른 의료 분야와 유사한 추세를 보입니다.

- **Technical Details**: 이 연구에서는 방사선 종양학 물리학에 대한 100문항의 전용 시험을 통해 GPT-4의 성능을 평가했습니다. GPT-4는 방사선 종양학에서 다른 LLM들보다 우수한 성과를 보였고, ACR 방사선 종양학 교육(In-Training) 시험에서는 74.57%의 높은 정확도를 기록했습니다.

- **Performance Highlights**: 특히 GPT-4는 AAPM TG-263 보고서에 따라 구조명 재명명에서 96% 이상의 정확도를 달성했습니다. 이러한 연구들은 방사선 종양학에서 LLM의 잠재력을 보여주며, 일반 의료 응용 프로그램 내 LLM의 가능성과 한계에 대한 관심이 높아지고 있습니다.



### Optimizing Small Language Models for In-Vehicle Function-Calling (https://arxiv.org/abs/2501.02342)
- **What's New**: 본 논문에서는 차량 내에 Small Language Models (SLMs)를 기능 호출 에이전트로 활용하는 혁신적인 접근 방법을 제안합니다. 기존의 규칙 기반 시스템을 대체하며, SLM을 통해 차량 제어 메커니즘을 단순화하고 사용자 경험을 향상시키는 것을 목표로 합니다. SLMs의 작은 사이즈 덕분에 차량 시스템의 통합이 용이해져, 외부 소프트웨어 업데이트나 운전자의 조건에 따라 쉽게 조정할 수 있는 시스템을 구성할 수 있습니다.

- **Technical Details**: 우리는 Microsoft의 Phi-3 mini 모델에 대한 최적화 작업을 수행하였으며, 모델 압축 기술인 pruning, healing, quantization을 통해 리소스 제약이 있는 차량에 적합하도록 하였습니다. 이 과정에서 우리는 모델의 크기를 2억 개의 파라미터를 줄이면서도 복잡한 차량 내 작업을 정확하고 효율적으로 수행할 수 있는 능력을 유지할 수 있음을 입증했습니다. 또한, 경량 런타임 환경에서 모델을 실행해 초당 11개의 토큰을 생성할 수 있어, 하드웨어 가속 없이 실시간 추론이 가능하다는 점이 특징입니다.

- **Performance Highlights**: SLM을 활용한 차량 제어 시스템은 사용자가 보다 직관적으로 차량과 상호작용할 수 있도록 지원합니다. 이 연구의 결과는 차량 시스템에 새로운 기능을 효과적으로 통합할 수 있는 가능성을 보여주며, 차량 내 앰비언트 설정 및 음성 비서와 같은 고급 기능들이 사용자 요구에 따라 변할 수 있게 합니다. 이러한 Advancements는 궁극적으로 개선된 주행 경험을 제공할 것으로 기대됩니다.



### UAVs Meet LLMs: Overviews and Perspectives Toward Agentic Low-Altitude Mobility (https://arxiv.org/abs/2501.02341)
- **What's New**: 이 논문은 드론과 대형 언어 모델(LLM)의 통합을 탐구하며, UAV 시스템의 기본 구성 요소와 기능의 개요를 제공합니다. 특히, LLM의 문제 해결 능력과 일반화 기능이 UAV의 지능을 향상시키는데 기여할 가능성을 강조합니다. 또한, 다양한 멀티모달 데이터 자원과 LLM이 UAV의 작업 시나리오와 만나는 주요 과제를 카테고리화하고 분석합니다.

- **Technical Details**: UAV 시스템의 기능적 모듈은 인식 모듈, 계획 모듈, 통신 모듈, 제어 모듈, 탐색 모듈, 인간-드론 상호작용 모듈 및 페이로드 모듈을 포함합니다. 이들 모듈은 UAV의 전반적인 성능을 향상시키는 데 중요한 역할을 합니다. 인식 모듈은 다양한 센서 데이터를 수집하고 해석하여 주변 환경을 이해하는 역할을 하며, 계획 모듈은 고급 미션 목표를 구체적인 비행 경로로 변환하는 데 필수적입니다. 이러한 모듈의 통합은 UAV의 자율성과 환경 적응력을 극대화합니다.

- **Performance Highlights**: 이 논문에서는 드론과 LLM의 통합을 통해 드론이 수행할 수 있는 고급 자율 행동을 제시합니다. 복잡한 환경 내 자율 비행을 위해, UAV는 여러 센서 데이터를 결합하여 상황 인식을 강화하고 실시간으로 경로를 조정합니다. 또한, 다수의 드론이 협력적으로 작업할 때 경로 계획을 조정하는 능력이 임무 효율을 높이고 위험을 줄이는 데 기여합니다.



### Evaluation of the Code Generation Capabilities of ChatGPT 4: A Comparative Analysis in 19 Programming Languages (https://arxiv.org/abs/2501.02338)
Comments:
          65 pages, in German, Bachelor's thesis on the evaluation of ChatGPT 4's code generation capabilities in 19 programming languages, University of Potsdam, June 2024

- **What's New**: 이번 연구는 ChatGPT 4의 다양한 프로그래밍 언어에서의 코드 생성 능력을 분석했습니다. 19개 프로그래밍 언어에 대한 188개의 문제를 LeetCode 플랫폼에서 선택하고, 이를 통해 모델의 문제 해결 능력과 코드 품질을 측정했습니다. 결과적으로 모델이 해결한 문제의 비율은 39.67%에 불과했으며, 문제의 난이도가 높아질수록 성공률이 급격히 감소했습니다.

- **Technical Details**: 연구는 3가지 난이도 수준에서 코드 생성의 성공률, 발생한 오류 유형, 실행 시간 및 메모리 효율성 측면에서 정량적으로 이루어졌습니다. 모델은 특히 널리 사용되는 언어에서 더 높은 성과를 보였으며, 낮은 추상화 수준과 정적 타입을 가진 언어에 대한 선호가 드러났습니다. 또한 인기 있는 언어에서 가장 흔한 오류는 'Wrong Answer'였고, 덜 사용되는 언어에서는 컴파일 및 런타임 오류가 주로 발생했습니다.

- **Performance Highlights**: 모델은 모든 프로그래밍 언어에서 평균 이상의 실행 효율성을 보여주었고, 14개 언어에서는 평균 이상의 메모리 효율성을 달성했지만, 5개 언어에서는 평균 아래의 성과를 보였습니다. 연구는 향후 더 많은 문제와 덜 인기 있는 언어를 포함하여 ChatGPT 4의 코드 해석, 요약, 디버깅 및 복잡한 실제 코드 개발 능력에 대한 심층 분석이 필요하다고 강조하고 있습니다.



### AdaSkip: Adaptive Sublayer Skipping for Accelerating Long-Context LLM Inferenc (https://arxiv.org/abs/2501.02336)
Comments:
          9 pages,10 figures, AAAI

- **What's New**: 최근 대규모 언어 모델(LLM)의 긴 문맥 추론(long-context inference) 지원이 향상되면서, 더욱 복잡한 실제 응용 프로그램이 가능해졌습니다. 그러나 긴 문맥 추론에서는 높은 계산 및 저장 요구가 발생합니다. 이를 해결하기 위해 이 논문에서는 적응형 서브레이어 스키핑 방식인 	extit{AdaSkip}을 제안하여, 기존의 레이어 스키핑 기법이 가지는 한계를 극복하고 있습니다.

- **Technical Details**: 	extit{AdaSkip}는 실행 중 유사성 정보를 활용하여 덜 중요한 레이어를 식별하고, 서브 레이어 단위의 스키핑을 가능하게 하여, 프리필링(prefilling) 및 디코딩(decoding) 단계 모두에서 성능을 향상시킵니다. 또한 각각의 서브 레이어가 가지는 중요도 분포를 독립적으로 평가하여, 보다 효율적인 스키핑 전략을 제공합니다. 이러한 방식은 레이어의 고정된 스키핑에 의한 생성 품질 저하 문제를 해결합니다.

- **Performance Highlights**: 다양한 긴 문맥 벤치마크와 모델에 대한 포괄적인 실험을 통해 	extit{AdaSkip}의 우수한 성능이 입증되었습니다. 본 논문은 기존의 기법들에 비해 생성 품질을 향상시키며, 특히 중요한 서브 레이어를 우선적으로 건너뛰는 전략을 통해 긴 문맥 추론 시 시간 및 메모리 오버헤드를 크게 줄였습니다.



### Validity Arguments For Constructed Response Scoring Using Generative Artificial Intelligence Applications (https://arxiv.org/abs/2501.02334)
Comments:
          33 pages, 2 figures, 6 tables; This work was presented at the 2024 meeting of the International Testing Commission in Granada, Spain

- **What's New**: 이 논문은 고배율 시험(high-stakes testing)에서의 생성적 인공지능(generative AI) 사용의 가능성을 탐구합니다. 특히, 생성적 AI가 인간 평가와 전통적인 AI 평점 방식보다 더 효과적일 수 있다는 점을 강조합니다. 기존 기능 기반(feature-based) AI 평점 시스템과의 비교를 통해 생성적 AI의 장점을 조명하고, 이에 대한 유효성 증거(validity evidence) 수집을 위한 최선의 관행을 제안합니다.

- **Technical Details**: 연구에서는 생성적 AI 방식의 평가에서 투명성 부족과 일관성(consistency) 문제와 같은 독특한 우려 사항들을 다룹니다. 생성적 AI는 기능 기반 자연어 처리(NLP) AI 평가 엔진보다 더 많은 유효성 증거가 필요합니다. 표준화된 시험에서 수집된 구성 응답(score) 데이터는 서로 다른 평가 시스템의 유효성 증거를 보여주며, 이러한 점에서 여러 복잡성과 고려 사항을 부각시킵니다.

- **Performance Highlights**: AI 평가 점수가 인간 평가에 비해 어떻게 합쳐질 수 있는지를 논의하며, 다양한 출처의 AI 점수를 결합한 기여 기반(contributory scoring) 접근 방식을 고려합니다. 이러한 접근 방식은 인간 평가 없이도 구성 요소(construct)의 더 많은 부분을 포괄할 수 있습니다. 이 논문은 생성적 AI를 활용한 평가 시스템의 신뢰성 문제에 대한 논의도 포함하고 있습니다.



### SR-Reward: Taking The Path More Traveled (https://arxiv.org/abs/2501.02330)
- **What's New**: 이 논문에서는 오프라인 시연에서 직접 보상 함수를 학습하는 새로운 방법인 SR-Reward를 제안합니다. 전통적인 역강화 학습(IRL)과 달리, 이 접근법은 보상 함수를 학습자의 정책과 분리하여 두 사이의 적대적 상호작용을 제거하며, 결과적으로 더욱 안정적이고 효율적인 훈련 과정을 제공합니다. SR 보상 함수는 후속 표현(Successor Representation, SR)을 활용하여 전문가의 정책에 따른 미래 상태 방문을 기반으로 상태를 인코딩합니다.

- **Technical Details**: SR-Reward는 벨만 방정식(Bellman equation)을 활용하여 대부분의 강화 학습 알고리즘과 동시에 학습할 수 있으며 기존의 훈련 파이프라인을 변경하지 않아도 됩니다. 또한, 이 논문에서는 분포 외 데이터의 보상을 줄여 과대 추정 오류를 완화하기 위한 음수 샘플링 전략을 도입하여 강화 학습 알고리즘의 견고성을 강화합니다. SR-Reward는 TD 학습을 통해 정보 전파가 가능하여 기존의 훈련 방식과 잘 통합될 수 있도록 설계되었습니다.

- **Performance Highlights**: D4RL 벤치마크에서의 평가 결과, SR-Reward는 진짜 보상에 접근할 수 있는 오프라인 RL 알고리즘 및 행동 클로닝(Behavioral Cloning)과 같은 모방 학습 기법에 비해 경쟁력 있는 성능을 보여주었습니다. 데이터 크기와 질에 대한 분리 연구를 통해 SR-Reward의 장점과 한계를 제시하였으며, 강화 학습 알고리즘에서 학습된 보상을 사용할 때 자연 보수성을 나타냅니다.



### DiffGraph: Heterogeneous Graph Diffusion Mod (https://arxiv.org/abs/2501.02313)
Comments:
          This paper is accepted by WSDM'2025

- **What's New**: 이번 연구는 Heterogeneous Graph Diffusion Model(DiffGraph)을 소개하며, 이 모델은 복잡한 이질적 구조를 효과적으로 다루기 위한 혁신적인 방법론을 제공한다. 특히, cross-view denoising 전략을 통해 보조적인 이질적 데이터를 목표의 의미 공간으로 변환하여 과제와 관련된 정보를 정밀하게 증류할 수 있도록 설계하였다. 또한, 고차원적 이질적 그래프 확산 메커니즘을 채택하여 잡음을 효과적으로 관리하는 새로운 전후 확산 프로세스를 구현하였다.

- **Technical Details**: DiffGraph는 이질적 그래프의 노드 및 엣지가 포함된 목표 서브그래프를 식별하고, 잔여 구조를 보조 그래프로 간주하여 특성을 강화하는 방식으로 작동한다. 이 모델은 노이즈가 추가되고 제거되는 과정이 있는 이중적 과정을 채택하고, 이를 통해 이질적 그래프 데이터의 복잡한 노이즈 분포와 다양한 관계 유형 간의 의미적 전환을 모델링한다. 이러한 접근은 기존 그래프 생성의 제한을 극복하고, 이질적 데이터 모델링의 편향 없는 능력을 significantly 향상시킨다.

- **Performance Highlights**: DiffGraph는 공개 데이터셋과 산업 데이터셋에서 엄격한 실험 검증을 통해 링크 예측 및 노드 분류 과제에서 기존 방법론을 지속적으로 초과하며, 이질적 그래프 처리의 강건성과 효율성에 대한 새로운 기준을 마련하였다. 이 연구는 향후 연구 및 실제 응용 프로그램에 있어 이질적 그래프 학습의 성능을 향상시키는 데 기여할 수 있는 가능성을 제시하고 있다.



### Deep Learning-Driven Segmentation of Ischemic Stroke Lesions Using Multi-Channel MRI (https://arxiv.org/abs/2501.02287)
- **What's New**: 이 연구는 다채널 MRI 방법을 활용한 새로운 딥러닝 기반의 허혈성 뇌졸중 병변 분할(segmentation) 방법을 제안합니다. 특히, 확산 강도 영상(Diffusion Weighted Imaging, DWI), 겉보기 확산 계수(Apparent Diffusion Coefficient, ADC), 그리고 향상된 확산 강도 영상(enhanced Diffusion Weighted Imaging, eDWI) 등의 이미징 모달리티를 통합하였습니다. 이 접근법은 DenseNet121을 인코더로, Self-Organized Operational Neural Networks(SelfONN)을 디코더에 활용하여 차별화되었습니다.

- **Technical Details**: 제안된 아키텍처는 Channel and Space Compound Attention (CSCA)와 Double Squeeze-and-Excitation (DSE) 블록을 활용하여 딥러닝 성능을 향상시키고 있습니다. 또한, 모델 성능을 개선하기 위해 Dice Loss와 Jaccard Loss를 가중 평균으로 결합한 맞춤형 손실 함수(custom loss function)를 도입했습니다. ISLES 2022 데이터셋에서 학습 및 평가를 진행하여, DWI 단일 시사용량으로 83.88%의 Dice Similarity Coefficients (DSC) 달성하였습니다.

- **Performance Highlights**: DWI와 ADC를 통합한 경우에는 85.86%, DWI, ADC 및 eDWI를 통합 시 87.49%의 DSC를 기록하며 기존 방법을 초과하는 성능을 보여줍니다. 이러한 접근법은 현재의 세분화(practices) 한계점을 해결하며 진단 정확도 및 치료 계획을 크게 향상시킵니다. 또한, 연구 성과는 임상적 의사결정(clinical decision-making)에 대한 방대한 지원을 제공합니다.



### Hyperbolic Contrastive Learning for Hierarchical 3D Point Cloud Embedding (https://arxiv.org/abs/2501.02285)
- **What's New**: 이 논문에서는 하이퍼볼릭 공간(hyperbolic space)을 활용하여 다중 모달(multi-modal) 데이터의 복잡한 계층 구조를 효율적으로 모델링하는 방법을 제안합니다. 특히 3D Point Cloud 모달에 대한 기존 연구가 부족했으나, 본 연구를 통해 하이퍼볼릭 멀티모달 대조 학습(multi-modal contrastive pre-training)을 확장합니다. 다양한 모달 간의 지식을 전이하기 위해 개념 관계를 명확히 학습하는 데 주안을 두고 있습니다.

- **Technical Details**: 하이퍼볼릭 공간에서의 대조 학습을 통해 텍스트 텐서(text tensor), 2D 이미지, 3D Point Cloud 간의 계층적 관계를 학습합니다. 이 과정에서 제안된 정규화 기법(regularizer)은 모달 간 계층적 개념 관계를 강화하는데 기여하며, 추상화된 하이퍼볼릭 피쳐를 정의합니다. 실험적으로, 이 접근 방식이 3D Point Cloud 인코더 성능을 크게 향상시키는 것을 보여줍니다.

- **Performance Highlights**: 결과적으로, 본 논문에서는 제안한 방법이 기존 기법보다 우수한 성능을 발휘하며, 다양한 다운스트림 작업(downstream tasks)에서도 현저한 개선을 도모합니다. 연구 결과, 계층적 3D Point Cloud 임베딩은 이미지와 텍스트와의 관계를 포착하여 기존의 모달 학습을 확장할 수 있는 가능성을 보여줍니다. 이러한 성과는 하이퍼볼릭 공간에서의 새로운 대조 학습 전략의 가능성을 시사합니다.



### What Kind of Visual Tokens Do We Need? Training-free Visual Token Pruning for Multi-modal Large Language Models from the Perspective of Graph (https://arxiv.org/abs/2501.02268)
Comments:
          9 pages, 6 figures

- **What's New**: 이번 연구는 Multimodal Large Language Models(MLLMs)의 비주얼 토큰(visual tokens) 사용의 필요성을 조사하고, 포어그라운드(foreground)와 배경(background) 토큰이 모두 중요하다는 사실을 밝혔습니다. 새로운 접근 방식인 G-Prune을 제안하여 시각적 토큰을 노드로 간주하고 연결을 구성함으로써 훈련 없이 비트리밍(visual token pruning)을 수행합니다. 이 방식은 정보 흐름을 가중 링크를 통해 전파하여 가장 중요한 토큰을 선택할 수 있도록 돕습니다.

- **Technical Details**: G-Prune은 그래프 기반 방법으로 시각적 토큰을 노드로 간주하고, 피처 거리(feature distance)에 따라 연결을 구성합니다. 정보를 전파하기 위한 반복 알고리즘을 실행하여 각 노드의 중요도를 업데이트합니다. 이를 통해 LLaVA-NeXT에 적용되어, 성능 저하 없이 계산 비용을 크게 줄일 수 있음을 입증하였습니다.

- **Performance Highlights**: G-Prune의 실험 결과, VQA2.0과 TextVQA에서 LLaVA-NeXT의 FLOPs를 각각 63.57% 줄이면서도 정확도는 각각 0.95%와 2.34%만 떨어지는 성능을 유지했습니다. 또한, 다양한 MLLM 벤치마크에서 높은 성능을 유지하며, TextVQA와 같은 세밀한 작업에서도 비교우위를 나타냈습니다. G-Prune은 중요한 응답 정보를 효과적으로 보존하면서 자동으로 시각적 토큰을 선택할 수 있는 가능성을 열어주었습니다.



### Towards a constructive framework for control theory (https://arxiv.org/abs/2501.02267)
Comments:
          Published under: this https URL

- **What's New**: 본 논문은 수학적 결과와 이를 컴퓨터에 구현할 때 발생하는 불일치, 즉 computational uncertainty를 다루기 위한 제어 이론을 위한 프레임워크를 제안합니다. 기존의 robust control 방법론에서는 이러한 불확실성을 보통 간과하거나 다른 종류의 불확실성과 통합되어 처리되는 경향이 있습니다. 그러나 이 논문은 computational uncertainty가 시스템의 안정성을 왜곡할 수 있음을 강조하며, 이 문제를 명시적으로 다뤄야 한다고 주장합니다. 이를 통해 본 연구는 constructive analysis에 기반한 새로운 증명 기법을 제공합니다.

- **Technical Details**: 본 논문에서 제안된 프레임워크는 Bishop의 constructive analysis에 기초하여 구축되었습니다. 이 분석 방법의 핵심은 모든 계산이 유한한 정밀도를 가진다는 점을 명확히 해 computational uncertainty를 고려합니다. 논문에서는 벡터와 집합이 어떻게 다뤄지는지 설명하며, 특히 알고리즘적으로 수렴할 수 있는 정보가 제공 되어야 한다고 강조합니다. 이러한 접근법은 전통적인 정의와 달리, 컴퓨팅 장치의 정밀도에 따라 계산적인 성격을 가지는 데이터를 다루는 데 유용합니다.

- **Performance Highlights**: 본 연구는 이전 작업들과 연계하여 optimal control, stabilization, system analysis 등의 분야에서 달성된 결과를 개관합니다. 또한, adversarial defense와 reinforcement learning에 중요한 Danskin의 정리를 근사적으로 구성한 새로운 결과를 제시합니다. 제안된 프레임워크는 컴퓨터 기반의 계산 문제를 해결할 수 있는 방법을 제시하며, 이는 제어 엔지니어들이 실질적인 모델을 다루는 데 유용할 것으로 기대됩니다. 전반적인 접근 방식은 제어 이론의 전통적인 작업 방식과 밀접하게 연관되어 있습니다.



### LLMzSz{\L}: a comprehensive LLM benchmark for Polish (https://arxiv.org/abs/2501.02266)
- **What's New**: 이 논문에서는 폴란드어를 위한 최초의 포괄적인 벤치마크인 LLMzSzŁ(LLMs Behind the School Desk)를 소개합니다. 이 벤치마크는 폴란드 중앙 시험 위원회의 아카이브에서 추출한 학술 및 전문 시험을 포함하는 4가지 유형의 시험으로 구성되며, 총 19,000개의 객관식 문제로 이루어져 있습니다. 연구에서는 열린 소스 다국어 모델과 폴란드어 LLM의 성능을 비교하여 언어 간 지식 전달 능력을 확인합니다.

- **Technical Details**: LLMzSzŁ 데이터셋은 단일 신뢰할 수 있는 출처인 폴란드 중앙 시험 위원회의 시험 자료를 기반으로 작성되었습니다. 데이터셋은 중학교, 고등학교, 직업 시험 등 복잡성과 관련된 다양한 층으로 나누어져 있으며, 원래 폴란드어로 작성된 문항들로 구성되어 있습니다. 이 벤치마크는 향후 시험의 난이도를 테스트하는 데에도 사용될 수 있습니다.

- **Performance Highlights**: 연구 결과, 다국어 LLM이 단일 언어 모델보다 우수한 성능을 보이는 경향을 보였습니다. 하지만 모델 크기가 중요한 경우에는 단일 언어 모델이 유리할 수 있습니다. 우리는 LLM이 시험 검증을 지원하고, 특히 시험 작업에서의 이상이나 오류 식별에 잠재력이 있음을 강조합니다.



### Interpretable Load Forecasting via Representation Learning of Geo-distributed Meteorological Factors (https://arxiv.org/abs/2501.02241)
- **What's New**: 이번 연구는 공간적 관계를 고려하여 지리적으로 분산된 기상 요인(MF)을 추출하는 새로운 방법론을 제안합니다. 기존의 일일 전력 수요 예측법에서는 특정 지역 또는 평균 기상 요인을 입력으로 사용했지만, 여러 위치에서의 기상 요인의 차이를 반영하지 못했습니다. 제안된 방법론에서는 그래프 신경망(Graph Neural Network)과 샤플리 값(Shapley value)을 활용해 기상 요인과 전력 수요 간의 관계를 조명합니다.

- **Technical Details**: 본 논문은 지리적으로 분산된 기상 요인과 전력 부하 간의 관계를 학습하기 위해 그래프 컨볼루션 신경망(Graph Convolutional Network, GCN)을 활용합니다. 특히, 샤플리 값을 사용하여 기상 요인의 중요도를 평가하고, 복잡한 계산을 줄이기 위해 몬테카를로 샘플링(Monte Carlo Sampling) 및 가중 선형 회귀(Weighted Linear Regression) 방법을 사용한 가속화 알고리즘을 제공합니다. 논문은 두 개의 실제 데이터 셋에 대한 광범위한 실험을 통해 제안한 방법론의 유효성을 입증합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법론은 특히 여름의 '축적 온도 효과(accumulation temperature effect)'와 겨울의 '급작스러운 온도 변화(sudden temperature change)' 같은 극단적인 상황에서 전력 예측 정확도를 크게 향상시키는 것으로 나타났습니다. 또한, 서로 다른 위치에서의 기상 요인의 중요성과 해당 지역의 GDP 및 주요 산업 간의 상관관계가 있음을 발견했습니다. 이러한 결과는 비단 기상 요인의 중요성을 강조할 뿐만 아니라 전력 수요 예측 모델의 향후 개발 방향에도 중요한 시사점을 제공합니다.



### Financial Named Entity Recognition: How Far Can LLM Go? (https://arxiv.org/abs/2501.02237)
Comments:
          Accepted at The Joint Workshop of the 9th Financial Technology and Natural Language Processing (FinNLP), the 6th Financial Narrative Processing (FNP), and the 1st Workshop on Large Language Models for Finance and Legal (LLMFinLegal), in conjunction with COLING 2025

- **What's New**: 이 연구는 최신 LLM(대형 언어 모델)의 금융 분야에서의 명명된 개체 인식(NER) 작업을 평가한 최초의 포괄적인 연구입니다. 연구에서는 다양한 프롬프트 기법에 대한 성능을 비교하고 LLM의 장단점 및 문제점을 확인하였습니다. 또한 실패 유형을 다섯 가지로 구분하고 이론적 기초를 마련하여 향후 연구 방향을 제시합니다.

- **Technical Details**: 연구에서는 세 가지 주요 LLM인 GPT-4o, LLaMA-3.1, Gemini-1.5를 사용하였으며, 직접 프롬프트, 컨텍스트 학습(in-context learning), 체인 오브 씽킹(chain-of-thought) 프롬프트의 세 가지 프롬프트 기법을 적용하여 실험을 수행하였습니다. 데이터셋으로는 FiNER-ORD를 사용하였으며, 이를 통해 LLM의 퍼포먼스를 측정하기 위한 다양한 기준과 방법론을 제시합니다. 실험 결과는 주로 F1 점수와 가중된 F1 점수를 활용하여 평가되었습니다.

- **Performance Highlights**: 실험 결과, 최신 LLM의 성능은 파인튜닝 모델보다 떨어졌으나, 다양한 프롬프트 디자인과 모델 크기에 따라 성과 차이를 줄일 수 있음을 보여주었습니다. 체인 오브 씽킹 프롬프트는 LLM의 성능에 제한적 영향을 미치며 경우에 따라 성능을 저하시킬 수도 있습니다. Gemini 시리즈가 FiNER-ORD 작업에서 다른 모델보다 우수한 성능을 발휘하는 것을 확인하였으며, 이는 모델 크기와 프롬프트 디자인의 영향을 받아 나타나는 결과입니다.



### Diffusion Model-Based Data Synthesis Aided Federated Semi-Supervised Learning (https://arxiv.org/abs/2501.02219)
Comments:
          accepted by IEEE WCNC 2025

- **What's New**: 본 논문에서는 Diffusion Model-based Data Synthesis Aided Federated Semi-Supervised Learning (DDSA-FSSL)이라는 새로운 접근방식을 제안합니다. DDSA-FSSL은 적은 양의 레이블 데이터와 비독립적이고 동일하게 분포되지 않은 (non-IID) 데이터 문제를 해결하기 위해 diffusion model (DM)을 활용하여 합성 데이터를 생성합니다. 이를 통해 서로 이질적인 로컬 데이터 분포와 글로벌 데이터 분포 간의 간극을 연결할 수 있습니다. 실험 결과, DDSA-FSSL은 분류 정확도를 유의미하게 향상시키는 효과를 보여 줍니다.

- **Technical Details**: DDSA-FSSL의 시스템 모델은 K개의 클라이언트와 중앙 서버로 구성되며, 각 클라이언트는 자체적인 로컬 데이터를 가지고 있습니다. 클라이언트들은 레이블이 있는 데이터와 최적화된 의사 레이블링(pseudo-labeling) 데이터를 활용하여 DM을 공동으로 학습하게 됩니다. 이러한 과정은 클라이언트들이 레이블이 없는 데이터에 대해 의사 레이블을 부여하고, 필요한 클래스의 합성 샘플을 생성할 수 있도록 합니다. 이를 통해 DDSA-FSSL은 클라이언트가 데이터 부족 문제를 극복하도록 돕습니다.

- **Performance Highlights**: 다양한 비독립적이고 동일하지 않은 (non-IID) 데이터 분포에 대한 실험을 통해 DDSA-FSSL의 효과성을 입증했습니다. 예를 들어, CIFAR-10 데이터셋에서 10% 레이블 데이터만 사용할 때, 정확도가 38.46%에서 52.14%로 향상되었습니다. 이는 유사한 기존 방법들과 비교했을 때, DDSA-FSSL의 뛰어난 성능을 보여줍니다. 또한, 합성 데이터를 포함한 그래픽스와 다양한 케이스에서 성능 향상을 확인할 수 있었습니다.



### Learning Evolution via Optimization Knowledge Adaptation (https://arxiv.org/abs/2501.02200)
Comments:
          This work has been submitted to Springer Nature for possible publication

- **What's New**: 이 논문에서는 Optimization Knowledge Adaptation Evolutionary Model (OKAEM)을 소개합니다. OKAEM은 축적된 지식을 활용하여 동적으로 매개변수를 조정함으로써 최적화 능력을 강화합니다. 이 모델은 attention 메커니즘을 사용해 개체들 간의 상호작용을 모델링하고, 진화 연산자를 매개변수화하여 선택, 교차, 돌연변이 과정을 개선합니다. 이를 통해 OKAEM은 사전 학습된 지식과 실시간 진화 통찰을 바탕으로 자체 조정이 가능한 매우 강력한 연산자를 제공합니다.

- **Technical Details**: OKAEM은 두 단계로 구성되어 있습니다: 사전 훈련과 적응 최적화. 사전 훈련 단계에서 OKAEM은 출처 작업에서 집단 진화 행동을 학습하여, 다음 세대의 집단을 예측합니다. 적응 최적화 단계에서는 새로운 진화 통찰에 맞춰 집단을 지속적으로 생성하고, 매개변수를 동적으로 조정하여 수행합니다. 이 접근 방식은 기존의 맞춤형 학습 가능 진화 알고리즘(LEAs)의 비유연성을 해결합니다.

- **Performance Highlights**: 실험 결과, OKAEM은 다양한 전이 시나리오에서 기존의 EKT 방법들을 능가하는 성능을 보였습니다. 또한, 사전 지식이 없더라도 OKAEM은 자체 조정 기능 덕분에 경쟁력 있는 성능을 달성할 수 있음을 보여줍니다. 특히, 비전-언어 모델 튜닝 사례에서는 OKAEM이 최신 블랙박스 기준을 초월한 성과를 보였습니다. 연구 결과는 OKAEM이 지식 축적에 따라 성능이 향상됨을 입증하였으며, 자연 선택 및 유전자 재조합 원리를 실질적으로 학습할 수 있다는 점에서 의의가 있습니다.



### Can ChatGPT implement finite element models for geotechnical engineering applications? (https://arxiv.org/abs/2501.02199)
- **What's New**: 이번 연구에서는 ChatGPT를 사용하여 지반공학 응용 프로그램을 위한 유한 요소 코드 생성 능력을 평가했습니다. 특히 비포화 토양의 수리-역학적 결합 문제에 대한 초기 경계 값 문제를 해결하는 데 초점을 맞추었습니다. 이러한 과제를 수행하는 데 있어 ChatGPT가 고수준 인터페이스의 FEniCS 라이브러리를 사용했을 때 코드 수정이 최소화되는 성능을 보여주었습니다.

- **Technical Details**: 연구에서는 수리-역학적 결합을 수치적으로 해결하기 위한 여러 기법을 소개하였으며, 각 기법은 고유한 알고리즘적 장점을 가지고 있습니다. ChatGPT는 이들 기술적 세부 사항을 이해하고 적절한 정보를 제공받아 유한 요소 코드 생성을 할 수 있도록 설계되었습니다. 초기 경계 값 문제(Initial Boundary Value Problems: IBVPs)의 해결을 위해 필요한 정보를 제공하며, 발생한 오류는 재촉구(Prompt Augmentation)를 통해 수정됩니다.

- **Performance Highlights**: 결과적으로, FEniCS 환경에서 생성된 코드의 경우 ChatGPT는 상대적으로 소규모 수정으로 유효성을 검증할 수 있었습니다. 반면에 MATLAB 환경에서는 코드 생성 과정에서 더 많은 직접적인 인간 개입이나 프로프트 증대가 필요했습니다. 이러한 결과는 위계적 프로그래밍 및 오차 수정 과정에서 ChatGPT의 가능성을 보여줍니다.



### CPTuning: Contrastive Prompt Tuning for Generative Relation Extraction (https://arxiv.org/abs/2501.02196)
- **What's New**: 이번 연구에서는 다중 관계 추출(multi-relation extraction)의 한계를 극복하기 위해 새로운 대비 프롬프트 튜닝 방법인 CPTuning을 소개합니다. 기존의 관계 추출(relation extraction, RE) 기법들이 두 엔티티 간 하나의 결정론적 관계만을 가정했던 것에 반해, CPTuning은 엔티티 쌍 간 여러 관계를 유연하게 다룰 수 있도록 설계되었습니다. CPTuning은 관계의 존재 여부에 따라 확률 질량을 조정하는 인사이트를 제공함으로써, 더 높은 성과를 달성하고 있습니다.

- **Technical Details**: CPTuning은 RE를 Seq2Seq 텍스트 채우기(text-infilling) 방식으로 재구성하여, 특정 프리픽스에서 시작되는 후보 관계를 생성합니다. 이 과정에서 텍스트 템플릿을 사용하여 엔티티의 관계를 마스킹한 샘플을 입력받아, 정해진 임계값 이상 혹은 이하의 확률에 따라 후보 관계를 생성하게 됩니다. 또한, Trie 구조를 활용하여 생성된 관계의 유효성을 보장하며, 적응형 빔 검색(prefix-given beam search)을 통해 검증된 후보 관계를 최종적으로 추출합니다.

- **Performance Highlights**: CPTuning을 적용한 T5-large 모델은 네 가지 널리 사용되는 데이터셋에서 기존 방법들을 능가하는 성능을 보였습니다. 특히, 다중 관계 추출이 가능한 능력 덕분에 기존의 단일 관계 추출 방식에 비해 유연하고 강력한 결과를 도출할 수 있었습니다. 이번 연구 결과는 CPTuning의 접근 방식이 관계 표현에서 의미 정보를 효과적으로 포착한다는 점에서 중요한 기여를 하고 있습니다.



### Benchmark Evaluations, Applications, and Challenges of Large Vision Language Models: A Survey (https://arxiv.org/abs/2501.02189)
Comments:
          34 pages, 3 figures

- **What's New**: 이번 논문은 최근 5년간(2019-2024) 개발된 멀티모달 비전 언어 모델(vision language models, VLM)에 대한 종합적인 개요를 제공합니다. 현재 VLM의 주요 구조, 교육 방법, 평가 기준 및 다양한 응용 분야를 체계적으로 정리하며, 특히 VLM 연구에 관심 있는 학술 연구자들에게 유용한 정보를 제공합니다. VLMs는 시각적 및 텍스트 입력을 결합하여 더욱 깊이 있는 이해를 가능하게 하는 독창적인 기술입니다.

- **Technical Details**: 이 논문에서는 VLMs의 주요 구성 요소와 훈련 방법에 대한 설명을 제공합니다. 특히, 이미지와 텍스트 정보를 정렬하기 위해 사전 훈련된 대규모 언어 모델(large language models, LLM)을 백본으로 사용하는 경향이 증가하고 있으며, 이는 VLM이 비주얼 콘텐츠를 더 잘 이해하도록 돕습니다. VLM의 훈련 목표와 아키텍처에 대한 주요 연구 방향을 세 가지로 나누어 설명하며, CRIP, BLIP, LLaMA와 같은 모델들이 있습니다.

- **Performance Highlights**: VLMs는 비전 인식 작업에서 뛰어난 성과를 보이며, 제로샷(Zero-shot) 분류에서도 기존의 단일 모달 모델을 넘어서는 성능을 보여줍니다. 또한, VLM의 활용 사례로는 자율주행, 로봇공학, 비디오 생성 등이 있으며, 시각적 질문 답변(visual question answering) 같은 복합적인 작업을 가능하게 합니다. 그러나 시각적 환각, 공정성, 안전성 문제와 같은 새로운 도전 과제가 있으며, 이러한 도전들은 멀티모달 모델의 발전을 위한 중요한 연구 주제로 급부상하고 있습니다.



### AdaMixup: A Dynamic Defense Framework for Membership Inference Attack Mitigation (https://arxiv.org/abs/2501.02182)
Comments:
          6 pages, 2 figures

- **What's New**: 이 논문에서는 새로운 방어 메커니즘인 AdaMixup을 제안합니다. AdaMixup은 membership inference attack에 대한 모델의 강인성을 향상시키기 위해 훈련 중 mixup 전략을 동적으로 조정합니다. 이 방법은 모델의 개인 정보 보호를 개선할 뿐만 아니라 높은 성능을 유지합니다. 실험 결과 ADAmixup이 membership inference 공격의 위험을 상당히 줄이면서 방어 효율성과 모델 정확도 간의 유리한 균형을 달성함을 보여줍니다.

- **Technical Details**: AdaMixup은 초기 훈련 단계에서 상당한 mixing을 보장하기 위해 λ(람다) 값을 크게 설정하고, 훈련이 진행됨에 따라 λ(람다) 값을 점차 줄입니다. 이를 통해 모델은 초기 데이터 샘플에서 정밀한 표현을 학습할 수 있으며, 과적합(overfitting)을 방지하면서 라벨 왜곡(label distortion)을 예방합니다. 또한, AdaMixup은 혼합 샘플에 대한 라벨 일치를 보장하기 위해 적응형 라벨 할당 전략을 통합하여 혼합 샘플에 기여하는 주된 샘플의 라벨과 일치하도록 합니다.

- **Performance Highlights**: 여러 데이터셋에서의 실험 결과, AdaMixup은 membership inference 공격의 위험을 크게 감소시키면서 모델의 정확도를 유지함을 나타냅니다. 이 연구는 데이터 개인 정보 보호를 위한 효과적인 솔루션을 제공하며, mixup 훈련 방법의 향후 발전을 위한 기초를 마련합니다. AdaMixup은 방어 효율성을 높이고 모델의 일반화를 촉진하면서도 민감한 데이터에 대한 강력한 보호를 제공합니다.



### The Integration of Blockchain and Artificial Intelligence for Secure Healthcare Systems (https://arxiv.org/abs/2501.02169)
Comments:
          13 pages, 4 Figures

- **What's New**: 최근 Verisign의 보고서에 따르면, 2022년 미국의 의료 분야에서 데이터 유출이 125% 증가했으며, 1,820만 건의 환자 기록이 영향을 받았습니다. 의료 데이터의 양이 급증하고 다양화됨에 따라 의료 정보의 가치가 높아지고 있습니다. 이를 개선하기 위해 헬스센터들은 다양한 기술을 활용하고 있습니다.

- **Technical Details**: AI(인공지능)와 블록체인 기술이 주로 사용되는 의료 시스템에서 데이터 효율성을 극대화하고 있습니다. AI를 통해 데이터 기반 운영이 개선되었으며, 전통적인 기법에 비해 효율성이 높아졌습니다. 블록체인은 개인 정보 보호 및 정보를 공유하는 거래의 안전성을 보장하는 데 도움을 줍니다.

- **Performance Highlights**: 본 연구는 2008년 이래 블록체인이 통합된 AI와 의료 시스템에 대한 연구를 조사하며, AI 기반 의료 프로그램의 적용 방식을 조명하고 있습니다. 또한, 환자의 데이터 보안와 의료 정보 관리에 기술이 어떻게 성공적으로 활용될 수 있는지를 다루고 있습니다. 특히 2018년부터 2021년까지 2021년이 가장 성장한 해로, 기기 다운로드와 Google 학술지의 수치가 증가했습니다.



### The Race to Efficiency: A New Perspective on AI Scaling Laws (https://arxiv.org/abs/2501.02156)
Comments:
          21 pages, 3 figures. 2 tables, second draft

- **What's New**: 이 논문은 AI 모델의 크기가 커짐에 따라 훈련 비용이 증가하고, 지속적인 진전을 유지하는 것이 어려워지고 있음을 지적합니다. 저자들은 상대 손실(Relative-loss) 방정식을 도입하여 고전적인 AI 스케일링 법칙을 확장하는 새로운 시간 및 효율성을 고려한 프레임워크를 제안합니다. 이 연구는 Moore의 법칙(Moore's Law)과 비교하여, 지속적인 효율성 향상이 없으면 고급 성능을 달성하기 위해 수천 년의 훈련이나 비현실적으로 큰 GPU 집합이 필요할 수 있음을 보여줍니다.

- **Technical Details**: 상대 손실 방정식은 시간에 따라 효율성이 향상될 때 훈련 손실이 어떻게 변화하는지를 정량적으로 설명합니다. 이 방정식은 초기 손실, 상대 손실, 그리고 단위 없는 스케일링 지수를 포함하며, 효율성 배증률에 따라 AI 스케일링이 지속적으로 이루어질 수 있음을 강조합니다. 전통적인 스케일링 법칙은 주어진 고정된 컴퓨팅 예산에서의 손실 감소를 정량화하지만, 이러한 새로운 접근법은 시간의 경과에 따른 변화를 고려해 제한적인 수익을 상쇄하는 방법을 제시합니다.

- **Performance Highlights**: 효율성 배증률이 Moore의 법칙에 맞추어 증가하면 AI 스케일링이 급격한 진전을 지속할 수 있을 것으로 보입니다. 예를 들어, 효율성을 2년마다 두 배로 늘리면, 상대 손실을 감소시키는 데 필요한 시간은 20년 이상 걸릴 수 있지만, 효율성이 그보다 빠르게 증가한다면 이 시간은 10년 이하로 단축될 수 있습니다. 따라서, 스케일링 법칙을 실제로 적용하기 위해서는 지속적인 효율성 향상이 필수적이라는 점이 강조됩니다.



### Attribute-Based Robotic Grasping with Data-Efficient Adaptation (https://arxiv.org/abs/2501.02149)
Comments:
          Project page: this https URL. arXiv admin note: substantial text overlap with arXiv:2104.02271

- **What's New**: 이번 연구에서는 로봇이 혼잡한 환경에서 새로운 대상 물체를 신속하게 잡을 수 있도록 돕기 위해 객체의 속성(object attributes)을 활용합니다. 저자들은 데이터 효율적(data-efficient)으로 적응할 수 있는 end-to-end encoder-decoder 네트워크를 제안합니다. 연구에서는 공간 이미지와 질의 텍스트를 조합한 후 이를 이용해 특정 객체에 대한 grasping affordances를 예측합니다.

- **Technical Details**: 이 논문에서는 모듈형(multimodal) 인코더와 affordances 디코더를 갖춘 아키텍처를 설계합니다. 모델은 시각적 및 텍스트 속성을 함께 인코딩하고, 이 정보를 결합하여 각 픽셀에 대한 grasping affordance를 예측합니다. 이를 통해 로봇은 다양한 색상과 형태의 기본 객체를 사용하여 사전 훈련된 속성 표현을 통해 새로운 객체 및 장면에 효율적으로 적응할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식은 심시뮬레이션과 실제 환경에서 81% 이상의 객체 잡기 성공률을 달성했습니다. 이는 기존 여러 기반선 모델들에 비해 큰 마진으로 개선된 성과입니다. 이 연구는 로봇 조작의 자율성을 높이고, 기존의 수동 데이터 수집 방식의 필요성을 최소화할 수 있는 가능성을 보여줍니다.



### Plasma-CycleGAN: Plasma Biomarker-Guided MRI to PET Cross-modality Translation Using Conditional CycleGAN (https://arxiv.org/abs/2501.02146)
Comments:
          Accepted by ISBI 2025

- **What's New**: 이번 연구에서는 혈액 기반 바이오마커(BBBMs)를 MRI에서 PET 이미지로의 변환 모델에 통합하는 효과를 조사했습니다. 이를 통해 BBBMs를 조건으로 하여 PET 이미지를 합성하는 새로운 방법, Plasma-CycleGAN을 제안합니다. 이는 MRI와 PET 간의 조건부 변환에 BBBMs를 도입한 최초의 접근 방식입니다. 연구 결과, BBBMs의 통합이 모든 모델에서 생성 품질을 지속적으로 향상시킨 것으로 나타났습니다.

- **Technical Details**: 연구에서 사용된 데이터는 알츠하이머병 신경영상 이니셔티브(ADNI)에서 수집된 이미지로, 총 1338개의 이미지를 포함합니다. 이들은 256x256x256 크기의 3D 복셀 큐브로 처리되었으며, 효율적인 훈련을 위해 128x128x128로 다운샘플링되었습니다. 데이터 증강 기법으로는 가우시안 노이즈 추가, 회전, 플립, 밝기 및 대비 변화 등을 적용하였고, CycleGAN을 기반으로 한 조건부 생성적 적대 신경망(cGAN)도 사용되었습니다.

- **Performance Highlights**: CycleGAN을 통한 다양한 생성 결과를 시각적으로 검토한 결과, 생성된 PET 이미지에서 가장 뛰어난 시각적 충실도를 나타낸 것으로 평가되었습니다. 연구는 BBBMs의 통합이 PET 이미지의 합성 품질에 긍정적인 영향을 미친다는 점을 보여주었으며, 향후 알츠하이머병 진단에 중요한 역할을 할 가능성이 있습니다. Plasma-CycleGAN은 기존의 대조군 모형들에 비해 이미지 품질을 개선하는 성과를 기록하였습니다.



### Establishing baselines for generative discovery of inorganic crystals (https://arxiv.org/abs/2501.02144)
- **What's New**: 생성적 인공지능(Generative AI)이 재료 발견(Materials Discovery)에서 유망한 가능성을 보여주지만, 기존 방법들에 비해 그 이점은 명확하지 않습니다. 본 연구에서는 두 가지 기본 접근법(baseline approaches)인 전하 균형 프로토타입의 무작위 나열(random enumeration)과 알려진 화합물의 데이터 기반 이온 교환(data-driven ion exchange)을 세 가지 생성 모델(generative models)과 비교했습니다.

- **Technical Details**: 비교한 생성 모델로는 변별 오토인코더(variational autoencoder), 대형 언어 모델(large language model), 그리고 확산 모델(diffusion model)이 있습니다. 결과적으로, 이온 교환과 같은 기존 방법들이 안정적인 물질 생성에서 유사한 성능을 보였으나, 생성적 모델은 새로운 구조적 프레임워크를 제안하는 데 탁월합니다. 또한, 충분한 훈련 데이터가 있을 경우 전자 밴드 갭(electronic band gap) 및 벌크 모듈러스(bulk modulus)와 같은 속성을 목표로 할 때 더 효과적입니다.

- **Performance Highlights**: 모든 제안한 구조를 안정성(stability) 및 속성(property) 필터를 통과시키는 생성 후 스크리닝(post-generation screening) 단계를 구현하여 기본 및 생성 접근 방식 모두의 성능을 향상시켰습니다. 이 저비용 필터링 단계는 모든 방법의 성공률(success rates)에 상당한 개선을 가져오며, 계산적으로 효율적인 방법입니다. 결국, 이는 재료 발견을 위한 보다 효과적인 생성 전략을 위한 실용적인 경로를 제공합니다.



### Effective LLM-Driven Code Generation with Pythoness (https://arxiv.org/abs/2501.02138)
Comments:
          5 pages

- **What's New**: 대규모 언어 모델(LLMs)의 발전은 프로그래밍 도구의 새로운 시대를 열었지만, 생성된 코드의 정확성과 신뢰성에 대한 보장이 부족한 위험도 동반합니다. 이러한 문제를 해결하기 위해 우리는 Pythoness라는 특화된 내장 언어(embedded domain-specific language)를 제안합니다. Pythoness는 개발자가 LLM과 상호작용할 수 있는 새로운 방식으로, 코드 대신 행동 명세에 집중하여 프로그래밍을 수행할 수 있도록 도와줍니다.

- **Technical Details**: Pythoness는 자연어 명세와 단위 테스트, 속성 기반 테스트를 통해 함수의 행동을 캡처하는 방법을 제공합니다. 개발자는 테스트 기반으로 함수의 행동을 이해하고, 이를 기반으로 코드가 생성되도록 지시합니다. 초기 호출 시, Pythoness는 LLM을 통해 코드 본체를 생성하고, 작성된 테스트를 실행하여 생성된 코드의 유효성을 검증합니다.

- **Performance Highlights**: Pythoness는 코드 생성 시 시뮬레이션된 테스트를 수행하여 코드 품질을 크게 향상시킬 수 있음을 보여줍니다. 기존 LLM을 사용한 경우와 비교하여, 개발자가 제공한 테스트를 기반으로 더 높은 품질의 코드를 생성합니다. 향후 계획으로는 런타임과 성능 테스트를 포함하여 Pythoness의 검증 및 코드 품질 유지 방식을 확장할 예정입니다.



### AVTrustBench: Assessing and Enhancing Reliability and Robustness in Audio-Visual LLMs (https://arxiv.org/abs/2501.02135)
- **What's New**: 최근 Multi-modal Large Language Models (MLLMs)의 빠른 발전에 따라, 이러한 모델들의 다중 모드 추론 능력을 평가하기 위한 새로운 진단 벤치마크가 개발되었습니다. 하지만 기존 벤치마크는 주로 시각적 요소에 제한되어 있어 오디오-비주얼(AV) 이해를 전체적으로 평가하지 않습니다. 본 연구에서는 Audio-Visual Trustworthiness assessment Benchmark (AVTrustBench)를 도입하여 600K 샘플로 9개의 세밀하게 설계된 작업을 포함하고, AVLLMs의 응답을 보정하는 능력을 조사합니다.

- **Technical Details**: AVTrustBench는 세 가지 차원(Adversarial attack, Compositional reasoning, Modality-specific dependency)에서 AVLLMs의 능력을 평가합니다. 연구진은 13개의 최첨단 AVLLMs를 평가하며, CAVPref라는 새로운 모델-불가지론적 훈련 전략을 제안합니다. 이 전략은 모든 멀티모달 입력(오디오, 비디오, 텍스트)을 조건으로 하여 AVLLMs의 신뢰성을 향상시키고, 30.19%의 성능 향상을 달성했습니다.

- **Performance Highlights**: 평가 결과, 기존 모델들은 인간과 유사한 이해력에 크게 미치지 못하는 것으로 나타났습니다. 본 연구를 통해 AVLLMs의 주요 한계점을 분석하고 성능에 대한 유용한 통찰을 제공합니다. 향후 연구 방향으로는 AVLLMs의 강인성과 추론 능력을 개선하기 위한 방안들이 제안됩니다.



### A hybrid marketplace of ideas (https://arxiv.org/abs/2501.02132)
- **What's New**: 이 논문은 인간과 AI 시스템의 융합이 문화적 및 지적 환경에 미치는 새로운 동적 요소를 탐구합니다. 특히, Web3라는 탈중앙화 커뮤니티의 맥락에서 AI 에이전트의 역할을 강조하며, 이들이 전통적인 참여와 영향력 개념에 도전하는 하이브리드 아이디어 시장을 설계합니다. 이후 AI 에이전트의 문화적 및 사회적 영향력에 대한 추가 연구를 촉진하고, 개인 정보 보호 및 지적 재산권과 같은 문제도 다룹니다.

- **Technical Details**: 연구 방법론으로는 문헌 검토와 질적 관찰, 주제 분석이 사용되었습니다. 연구자는 X 플랫폼의 AI x Web3 커뮤니티에서 진행된 Spaces 세션에 참여하여 AI 에이전트의 역할을 탐구했습니다. 이들은 Morpheus 생중계와 Spore.fun과 같은 프로젝트를 분석하여 AI 에이전트가 하이브리드 아이디어 시장에서 어떻게 이야기하고 토론에 기여하는지를 보여주었습니다.

- **Performance Highlights**: 이 연구는 AI 에이전트가 문화적 내러티브를 형성하고 진화하는 과정을 어떻게 변화시키는지를 규명하고 있습니다. 디지털 플랫폼에서의 아이디어의 경쟁은 종래의 진리 또는 이성의 기준과는 다르게, 주목을 끌고 효과적으로 전파하는 능력에 기반한다는 점이 강조됩니다. AI 에이전트는 이제 단순한 도구가 아닌 문화적 진화의 능동적인 참여자로 자리매김하며, 이로 인해 문화적 동병상련의 형성이 이루어지고 있습니다.



### Relaxation-assisted reverse annealing on nonnegative/binary matrix factorization (https://arxiv.org/abs/2501.02114)
- **What's New**: 이번 연구는 비음수/이진 행렬 인수 분해(nonnegative/binary matrix factorization, NBMF)를 위한 최적화 성능을 향상시키기 위해 리버스 어닐링(reverse annealing, RA)과 선형 프로그래밍 완화(linear programming relaxation) 기법을 통합하는 개선된 전략을 제안합니다. 실험 결과, 이 방법이 기존의 RA 방법보다 더 나은 수렴성을 보이며, 리버스 어닐링과 고전적 최적화 기법을 결합하여 성능을 향상시킬 수 있는 가능성을 강조합니다.

- **Technical Details**: 비음수 행렬 인수 분해(NMF)는 주어진 행렬을 비음수 행렬 두 개로 분해하는 방법으로, 클러스터링, 차원 축소, 특성 추출 등 비지도 학습 작업에 널리 사용됩니다. 이 연구에서는 NMF의 이진 제약을 추가하여 검색 공간을 이산적으로 만들어 퀀텀 어닐링이 이러한 문제에 적합하도록 하였으며, 리버스 어닐링을 기존의 알고리즘에 통합하여 초기 상태의 성질이 성능에 미치는 영향을 분석합니다.

- **Performance Highlights**: 제안된 방법을 얼굴 이미지 데이터세트에 적용한 실험 결과, 최적화 성능이 정확한 최적화 방법에 필적하는 개선을 보여주었습니다. 또한, 완화 기반 초기화 방법의 유효성을 다양한 무작위 데이터세트에서 조사하여, 완화 솔루션과 최적 솔루션 간의 관계를 입증했습니다. 이러한 결과는 NBMF와 다른 최적화 문제의 맥락에서 완화 지원 리버스 어닐링의 가능성을 제시합니다.



### Siamese Networks for Cat Re-Identification: Exploring Neural Models for Cat Instance Recognition (https://arxiv.org/abs/2501.02112)
Comments:
          8 pages, 3 figures, 7 tables

- **What's New**: 2023년 4월, 중국의 도시 모빌리티 기업 Hello Inc.는 도시 지역의 길고양이 문제를 해결하기 위한 Hello Street Cat 이니셔티브를 시작했습니다. 이 프로젝트는 14개 도시에서 21,000개 이상의 스마트 급식소를 설치하고, 사용자 기부금을 통해 작동하는 카메라와 간식 배급기를 통합하여 길고양이를 관리합니다. 또한 이 이니셔티브는 Trap-Neuter-Return(TNR) 방법을 홍보하며, 자발적으로 운영되는 플랫폼인 HelloStreetCatWiki에서 자원봉사자들이 고양이를 분류하고 기록합니다.

- **Technical Details**: 이 연구는 길고양이를 재식별하기 위한 Deep Learning 기반 모델을 탐구하며, 2,796장의 69마리 고양이 이미지를 사용해 Siamese Networks를 학습시켰습니다. EfficientNetB0, MobileNet 및 VGG16을 기반 모델로 삼아 대비 손실과 삼중 손실 함수 하에서 평가하였습니다. 그 결과, VGG16과 대비 손실의 조합이 가장 효과적으로 입증되어 97%의 정확도와 0.9344의 F1 점수를 기록했습니다.

- **Performance Highlights**: 이 접근법은 이미지 증강 및 데이터 세트 정제를 통해 제한된 데이터와 다양한 시각적 변동성 문제를 극복했습니다. 자동화된 고양이 재식별의 가능성은 인구 모니터링 및 복지 노력을 효율화할 수 있음을 강조합니다. 이 연구는 향후 데이터 세트 확장 및 대규모 배치를 위한 실시간 구현 개발에 초점을 맞추어 실용성을 높일 계획입니다.



### Online Detection of Water Contamination Under Concept Drif (https://arxiv.org/abs/2501.02107)
- **What's New**: 본 연구는 물 분배 네트워크(WDN)의 실시간 오염 탐지를 위한 새로운 방법론인 이중 임계값 이상 징후 및 드리프트 탐지(Dual-Threshold Anomaly and Drift Detection, AD&DD)를 소개합니다. 이 방법은 LSTM 기반의 변동 오토인코더(LSTM-VAE)와 이중 임계값 드리프트 탐지 메커니즘을 결합하여 오염 물질의 영향을 모니터링합니다. AD&DD는 센서 오프셋을 개념적 드리프트로 간주하여 두 개의 실제 WDN에서 효과적으로 이상 탐지 기능을 수행하며, 기존의 다른 방법들보다 우수한 성능을 보입니다.

- **Technical Details**: AD&DD는 비지도 학습 알고리즘을 활용하여 실시간으로 이상을 탐지하고, 이를 통해 오염 물질을 로컬라이즈하는 기능을 제공합니다. 각 센서는 AD&DD를 통해 이상 탐지를 수행하며, 초기 흐름 방향에 대한 지식을 활용해 결함의 로컬라이제이션이 가능합니다. 이 알고리즘은 VAE와 LSTM을 통합하여 시계열 데이터에 대한 모델링을 수행하고, 측정값의 변동을 파악하는 데 중요한 역할을 합니다.

- **Performance Highlights**: 실험 결과, AD&DD 방법은 여러 최신 기법들과 비교했을 때 이상 탐지 성능이 우수함을 입증하였습니다. 이 연구는 오염 탐지와 개념 드리프트 간의 구분이 필요한 비정상 WDN 환경에서도 잘 작동하며, 정확한 오염 탐지 및 로컬라이제이션을 위한 분산 아키텍처 또한 제안합니다. 이러한 접근 방식은 실시간으로 오염 사건을 파악하고 이에 대응할 수 있는 능력을 갖추고 있습니다.



### On the Statistical Complexity for Offline and Low-Adaptive Reinforcement Learning with Structures (https://arxiv.org/abs/2501.02089)
Comments:
          Review Article

- **What's New**: 최근 강화학습(RL)의 통계적 기초에 대한 발전을 다루는 본 논문은 오프라인(offline) 및 저적응(low-adaptive) 환경에서의 문제들을 중점적으로 살펴봅니다. 오프라인 RL이 실제 머신러닝(ML) 문제에 적합한 모델임을 주장하며, 오프라인 정책 평가(Offline Policy Evaluation, OPE)와 오프라인 정책 학습(Offline Policy Learning, OPL)의 두 가지 기본 문제에 집중합니다. 특히, 이 분야의 최근 경향과 알고리즘 기법들을 고찰하여 새로운 통찰을 제공합니다.

- **Technical Details**: 오프라인 RL의 주요 과제 중 하나는 긴 결정 기간으로 인해 최적 전략을 찾기 어렵다는 것입니다. 초기 단계에서의 잘못된 결정은 미래에 지속적인 영향을 미치며, 이는 최적 정책과 실제 데이터 분포 간의 격차를 증대시킵니다. 또한, 기능 근사(Function Approximation)와 일반화(Generalization) 문제도 해결이 필요한 상황으로, 큰 상태 및 행동 공간을 다루기 위한 기법이 요구됩니다.

- **Performance Highlights**: 본 논문은 다양한 사례를 통해 RL의 중요성과 적용 가능성을 강조합니다. 의료 진단, 추천 시스템, 비디오 스트리밍과 같은 실질적인 사례에서 RL이 어떻게 사용될 수 있는지를 보여주며, 이러한 문제는 오프라인 RL 접근 방식에 대한 이해를 더욱 심화시키고 있습니다. 최적의 행동을 찾기 위한 오프라인 데이터셋의 활용과 그로 인한 학습 어려움을 다루며, 이를 통해 이론적 기법과 알고리즘의 발전을 조망합니다.



### The interplay between domain specialization and model size: a case study in the legal domain (https://arxiv.org/abs/2501.02068)
- **What's New**: 이 연구는 기존의 언어 모델(LM) 규모 최적화에 관한 연구가 단순히 모델 크기와 토큰 수에 관한 관찰에 국한되어 있음을 지적합니다. 저자들은 계속적인 사전 훈련(continual pre-training)을 통해 기존 모델의 지식을 활용해 데이터를 효율적으로 활용할 수 있는 가능성을 제시합니다. 연구 결과, 도메인 특화 모델이 일반 모델보다 높은 성능을 보이며, 동일한 컴퓨팅 자원으로 더 나은 효율성을 발휘하는 경향이 발견되었습니다.

- **Technical Details**: 이 연구에서는 1.5B, 3B, 7B, 14B 매개변수를 가진 언어 모델을 사용하여 법률 도메인의 전문화된 데이터셋과 일반 데이터셋으로 훈련했습니다. 데이터셋 필터링 기법을 사용하여 법률 관련 데이터만 추출하였으며, 법률 시험에 대한 모델 성능을 평가했습니다. 모델 크기가 증가함에 따라 도메인 특화 모델과 일반 모델 간의 compute-effectiveness 차이가 더욱 두드러진 것으로 나타났습니다.

- **Performance Highlights**: 특화된 모델이 동일한 자원 내에서 일반 모델보다 우수한 성능을 발휘했으며, 특히 14B 매개변수 특화 모델이 일반 모델보다 4.3배 적은 컴퓨팅을 사용하면서 뛰어난 성능을 기록했습니다. 이러한 결과는 도메인 특화 훈련이 특정 도메인에서 모델의 성능을 극대화할 수 있음을 시사합니다. 전체적으로, 이 연구는 도메인 전문가의 수요를 충족하기 위한 학습 방안으로서 계속적인 사전 훈련의 가능성을 혁신적으로 탐구합니다.



### ArtCrafter: Text-Image Aligning Style Transfer via Embedding Reframing (https://arxiv.org/abs/2501.02064)
- **What's New**: 최근 텍스트 기반 스타일 전이 분야에서 ArtCrafter라는 혁신적인 프레임워크가 소개되었습니다. 이 프레임워크는 이미지 내의 세부적인 스타일 요소를 캡처하기 위해 설계된 attention 기반 스타일 추출 모듈을 포함하고 있습니다. 또한, 텍스트-이미지 정렬 증강 요소를 도입하여 두 가지 모달리티 간의 조정을 강화하여 더욱 다양한 결과를 생성할 수 있게 합니다. ArtCrafter는 다중 모달 증강 임베딩과 원본 임베딩을 융합하는 명확한 조절 기능을 통해 우수한 결과를 보여줍니다.

- **Technical Details**: ArtCrafter는 세 가지 주요 구성 요소로 이루어져 있습니다. 첫째, attention 기반 스타일 추출은 복잡한 스타일 정보를 포착하기 위해 multi-layer 구조와 perceiver attention 메커니즘을 활용하여 세부적인 스타일 요소를 통합합니다. 둘째, 텍스트-이미지 정렬 증강은 각 모달리티 간의 균형 잡힌 통합을 가능하게 하여 생성된 이미지가 텍스트 프롬프트의 내용과 스타일을 잘 반영하게 합니다. 셋째, 명시적 조절 기능은 선형 보간 및 연결 방법을 통해 원본 임베딩과 다중 모달 임베딩을 결합하여, 높은 다양성과 관련성을 지닌 이미지를 생성합니다.

- **Performance Highlights**: ArtCrafter는 다양한 실험을 통해 뛰어난 비주얼 스타일화 결과를 입증하였습니다. 이 프레임워크는 스타일 강도와 컨트롤 가능성에서 뛰어난 성능을 보여줍니다. 특히, 이전 연구들에 비해 실시된 텍스트 프롬프트의 영향을 극대화하여 다양한 출력 결과를 생성하는 것을 목표로 합니다. ArtCrafter는 미술적 스타일에 대한 예민함을 유지하면서도 강력한 일반화 능력을 자랑하며, 다양한 실험 벤치마크에서 최신 기술을 초월하는 성능을 기록하였습니다.



### METAGENE-1: Metagenomic Foundation Model for Pandemic Monitoring (https://arxiv.org/abs/2501.02045)
- **What's New**: 본 논문에서는 70억 개의 파라미터를 가진 autoregressive transformer 모델인 METAGENE-1을 소개합니다. 이 모델은 1.5조 개의 염기쌍으로 구성된 다양하고 드문 metagenomic DNA 및 RNA 시퀀스를 포함하는 새로운 말뭉치(corpus)에서 사전 학습되었습니다. METAGENE-1의 목표는 개인 유전체나 특정 종의 선별적 데이터 세트를 넘어서, 인간의 하수에서 발견되는 유전 정보를 포괄적으로 캡처하여 전염병 모니터링 및 병원체 탐지와 같은 작업을 지원하는 것입니다.

- **Technical Details**: 사전 학습 과정에서는 metagenomic 시퀀스를 위한 맞춤형 byte-pair encoding (BPE) 토크나이제이션 전략을 사용하여 데이터 세트를 처리하였습니다. 데이터 세트는 여러 지점 및 시간에 수집된 수많은 종의 짧은 비선별(un-curated) 시퀀스로 구성되어 있으며, 이는 METAGENE-1이 미생물 및 바이러스 다양성의 복잡성을 효과적으로 표현할 수 있게 해줍니다. 모델 아키텍처는 GPT 및 Llama 계열 모델들과 유사한 decoder 스타일의 언어 모델을 채택하였습니다.

- **Performance Highlights**: METAGENE-1은 병원체 탐지 및 metagenomic 임베딩(embedding) 벤치마크에서 최상위 성능을 달성하였으며, 이는 기존의 인간 및 동물 유전체로 훈련된 모델들을 능가하는 결과입니다. 또한, 비정상 탐지 시나리오에서도 우수한 성과를 보여주며, 공공 보건 분야에 적합한 응용 프로그램으로서의 잠재력을 드러냅니다. 궁극적으로 METAGENE-1은 전염병 모니터링 및 새로운 건강 위협의 조기 탐지에 기여할 수 있는 기초 모델로 자리 잡는 것이 목표입니다.



### Advancing Pancreatic Cancer Prediction with a Next Visit Token Prediction Head on top of Med-BER (https://arxiv.org/abs/2501.02044)
- **What's New**: 최근에 개발된 Med-BERT라는 EHR(전자 건강 기록) 특정 기반 모델을 활용하여 질병 예측을 위한 새로운 방법론이 제안되었습니다. 이 연구는 질병 이진 예측 작업을 토큰 예측 작업으로 재구성하여 Med-BERT의 사전 학습(task format) 목표에 맞추었습니다. 이러한 접근법은 특히 매우 작은 세부 조정(cohorts) 집단에서의 모델 활용 최적화에 기여할 수 있습니다.

- **Technical Details**: 연구에서는 Med-BERT-Sum과 Med-BERT-Mask라는 두 가지 새로운 모델을 도입하였습니다. Med-BERT-Sum은 토큰 예측 작업을 통해 소량의 데이터에서도 우수한 성능을 보이는 반면, Med-BERT-Mask는 다음 방문 마스크 토큰 예측 작업을 사용하여 기존의 이진 분류(binary classification) 작업보다 3%에서 7% 더 나은 결과를 나타냈습니다. 특히, 데이터 크기가 10에서 500 샘플인 경우에 이 능력이 더욱 두드러집니다.

- **Performance Highlights**: 이 연구의 주요 발견은 다운스트림 작업을 Med-BERT의 사전 훈련(pretraining) 목표와 일치시킴으로써 모델의 예측 능력이 크게 향상되었다는 점입니다. 이 접근법은 드문 질병 및 일반 질병 예측에 있어 효과적이며, 특히 췌장암(PaCa)에 대한 조기 발견과 시기적절한 개입을 가능하게 합니다. 결과적으로 이러한 기술은 치료 효과, 생존률 및 환자 결과 개선에 기여할 것으로 기대됩니다.



### MRG: A Multi-Robot Manufacturing Digital Scene Generation Method Using Multi-Instance Point Cloud Registration (https://arxiv.org/abs/2501.02041)
- **What's New**: 이 논문은 다중 인스턴스 포인트 클라우드 등록(multi-instance point cloud registration) 기법을 활용한 새로운 다중 로봇 제조 디지털 장면 생성 방법(Mult-Robot Manufacturing Digital Scene Generation, MRG)을 소개합니다. 기존의 포인트 클라우드 "분할-등록" 방법과 달리, 본 연구는 제조 설정에 맞춘 인스턴스 중심의 transformer 모듈을 개발하여 인스턴스 경계를 구분하고 지역 간의 상관관계를 포착합니다. 이 방법은 기존의 기술 대비 뛰어난 성능을 발휘하며, 실제 생산 공정의 디지털 시뮬레이션 환경을 개선하는 데 중요한 진전을 가져옵니다.

- **Technical Details**: MRG 방법은 산업 로봇의 특성과 제조 환경에 최적화되어 있습니다. 또한, 가설 생성 모듈(hypothesis generation module)을 통해 목표 인스턴스를 추출하면서 주요 특징을 유지하고, 마지막 등록 결과를 정제하기 위한 효율적인 스크리닝 및 최적화 알고리즘이 설계되었습니다. MRG는 점진적 정밀(map extraction strategy) 매핑을 통해 인스턴스 추출 정확도를 크게 향상시키며, 산업 로봇의 각 지역 간 상호 연결성도 철저히 고려합니다.

- **Performance Highlights**: 실험 평가에서는 Scan2CAD 및 Welding-Station 데이터셋에서 제안한 MRG 방법이 기존의 다중 인스턴스 포인트 클라우드 등록 기법을 초월하는 결과를 보였습니다. Scan2CAD 데이터셋에서 MR과 MP은 각각 12.15%와 17.79% 개선되었고, Welding-Station에서는 각각 16.95%와 24.15% 향상되었습니다. 이러한 결과는 MRG 방법이 제조 장면에서의 포인트 클라우드 등록 기술을 혁신적으로 발전시키는 데 기여할 것임을 시사합니다.



### A Separable Self-attention Inspired by the State Space Model for Computer Vision (https://arxiv.org/abs/2501.02040)
- **What's New**: 이번 논문에서는 Mamba의 개념을 활용하여 새로운 형태의 분리된 자기 주의 메커니즘인 VMI-SA를 제안합니다. 이를 통해 비전 Mamba(ViM)와의 공정한 비교를 위해 VMINet라는 프로토타입 아키텍처를 구축하였습니다. VMINet은 단순하지만 강력한 구조로, 기본 다운샘플링 레이어와 결합된 새로운 주의 모듈로만 구성되어 있습니다.

- **Technical Details**: 이 논문에서는 별개의 자기 주의(Separable Self-Attention)와 소프트맥스 자기 주의(Softmax Self-Attention), 상태 공간 모델(State Space Models) 간의 관계를 분석하여 설계 원칙을 수립하였습니다. 또한, 제안된 VMI-SA는 이전 토큰의 수용 범위에 국한하여 성능을 최적화합니다. 최종적으로, VMI-SA의 수용 범위를 복원하여 병렬 컴퓨팅의 장점을 유지하게 됩니다.

- **Performance Highlights**: 실험 결과, VMINet은 기존의 Vim 모델을 일관되게 초과하는 성능을 보여 주며, 최신 모델들과의 경쟁에서도 뛰어난 결과를 나타냈습니다. 이러한 성과는 효율적인 연산과 우수한 설계 원칙 덕분으로, 이미지 분류와 고해상도 밀집 예측에서 경쟁력을 갖추고 있습니다.



### An Investigation into Value Misalignment in LLM-Generated Texts for Cultural Heritag (https://arxiv.org/abs/2501.02039)
- **What's New**: 최근 대규모 언어 모델(LLMs)이 문화 유산 관련 작업에서 사용됨에 따라, 정확하고 문화적으로 일치하는 텍스트 생성의 필요성이 커지고 있습니다. 그러나, 연구에 따르면 생성된 텍스트에서 문화적 가치의 불일치가 발생하고 있으며, 이는 역사적 사실의 왜곡이나 문화 정체성의 침해 등 심각한 결과를 초래할 수 있습니다. 본 논문은 LLMs가 생성한 문서에서의 문화적 가치 불일치 문제를 체계적으로 조사하여, 그 심각성을 드러내고자 합니다.

- **Technical Details**: 이 연구에서는 1066개의 쿼리 작업으로 구성된 벤치마크 데이터셋을 구축하고, 5개의 널리 인정된 카테고리와 17개의 측면을 평가하여 LLMs의 문화적 가치 불일치 유형과 비율을 분석했습니다. 자동화와 수동 평가 방식을 결합하여, 생성된 텍스트에서의 문화적 불일치를 효과적으로 탐지하고 분석했습니다. 초기 연구 결과에 따르면, 상당수의 생성된 텍스트에서 65% 이상이 뚜렷한 문화적 불일치를 나타내는 것으로 확인되었습니다.

- **Performance Highlights**: 대규모 언어 모델이 생성한 텍스트는 다양한 문화적 가치의 불일치 문제를 드러내며, 전체 분석 대상 1000개 작업 중 약 65%가 영향을 받는 것으로 나타났습니다. 이는 문화적으로 민감한 분야에서 LLMs의 신뢰성을 높이기 위한 개선된 방법론의 필요성을 강조합니다. 이 연구는 문화적 민감성과 신뢰성을 향상시키기 위한 귀중한 자원인 공개 데이터 세트를 제공하여, 향후 관련 연구에 기여할 것으로 기대됩니다.



### Architecture for Trajectory-Based Fishing Ship Classification with AIS Data (https://arxiv.org/abs/2501.02038)
Comments:
          Sensors 2020

- **What's New**: 이번 논문에서는 실제 kinematic 데이터 관리 및 어선 탐지를 위한 데이터 준비 프로세스를 제안합니다. 본 솔루션은 배의 궤적을 어선(fishing ships)과 비어선(non-fishing ships)으로 분류하는 이진 분류(binary classification) 방법론을 채택합니다. 이 데이터는 실세계 데이터의 전형적인 문제인 노이즈(noise)와 불일치(inconsistencies)를 포함하고 있습니다.

- **Technical Details**: 제안된 방안에서는 두 개의 클래스가 불균형(unbalanced)되어 있는 문제를 다루며, 이는 알고리즘을 통해 샘플을 재편성(resample)하여 해결합니다. 분류(classification) 과정에서 배의 궤적을 나타내는 시공간(spatiotemporal) 데이터에서 여러 특성(features)을 추출합니다. 이 특성은 Automatic Identification System (AIS) 보고서의 시퀀스에서 제공된 데이터로부터 생성됩니다.

- **Performance Highlights**: 실험 결과, 제안된 데이터 준비 프로세스가 제시된 분류 문제에 유용하다는 것을 보여줍니다. 또한 최소한의 정보를 사용하여 긍정적인 결과를 얻을 수 있음을 확인하였습니다.



### Deep Clustering via Community Detection (https://arxiv.org/abs/2501.02036)
Comments:
          10 pages, 10 figures

- **What's New**: 이 논문에서는 커뮤니티 탐지를 활용한 새로운 딥 클러스터링 접근 방식을 제안하고 있습니다. 기존의 클러스터링 전략과는 달리, 이 방법은 클러스터 네트워크 분석의 새로운 관점을 도입하여 클러스터 간 혹은 클러스터 내 샘플들의 묶음이 서로 겹치지 않도록 합니다. 이는 가짜 라벨의 순도(pseudo-label purity)를 높여 자가 감독학습(self-supervised learning)의 성능을 향상시키는데 기여합니다.

- **Technical Details**: 제안된 방법은 DCvCD(Deep Clustering via Community Detection)로 명명되었습니다. 초기 클러스터링 이후, Louvain 알고리즘을 사용하여 클러스터 내의 노이즈 샘플을 더 작은 커뮤니티로 나눈 뒤, 주요 커뮤니티를 선택하여 표현 학습(Representation Learning)을 fine-tuning 합니다. 커뮤니티 병합 과정에서는 네트워크 구조적 메트릭(network structural metrics)과 전통적인 거리 메트릭(distance metrics)을 고려하여 라벨의 순도를 개선합니다.

- **Performance Highlights**: 실험을 통해 이 방법의 효과성을 검증했으며, 최신 기술 성과(state-of-the-art, SOTA)와 비교하여 우수한 결과를 나타냈습니다. 제안된 DCvCD 접근 방식은 클러스터 네트워크 분석을 통해 클러스터의 가짜 라벨 순도를 효과적으로 개선할 수 있음을 보여주었습니다. 이러한 결과는 다양한 데이터 분포에 맞춘 커뮤니티 탐지 알고리즘을 쉽게 적용할 수 있는 유연한 로드맵을 제공함을 의미합니다.



### 3D Cloud reconstruction through geospatially-aware Masked Autoencoders (https://arxiv.org/abs/2501.02035)
- **What's New**: 이 연구는 지상 정지 위성 이미지(MSG/SEVIRI)와 CloudSat/CPR의 레이더 반사도를 활용하여 실시간 3D 구름 구조를 재구성하는 방법을 제시합니다. 자기 지도 학습(Self-Supervised Learning, SSL) 기법인 Masked Autoencoders(MAE)와 지리정보 중심의 SatMAE를 적용하여 비라벨 MSG 이미지를 학습합니다. 이 접근법은 최첨단 모델인 U-Net보다 뛰어난 성능을 보이며, SSL의 잠재력을 입증합니다.

- **Technical Details**: 연구에서는 MSG/SEVIRI의 11개 스펙트럼 채널의 복사선 데이터를 모델 입력으로 사용합니다. MAE와 Vision Transformer(ViT) 구조를 기반으로 하여 이미지를 토큰화하고, 지역적 특징 및 공간적 관계를 학습합니다. 최종적으로는 3D 구름 재구성을 위해 모델을 파인튜닝하여 90 x 256 x 256 크기의 볼륨을 생성합니다.

- **Performance Highlights**: 실험 결과, MAE 기반의 모델이 최적화된 경우 U-Net보다 더 나은 PSNR( Peak Signal-to-Noise Ratio) 값을 달성하며, 유리한 수렴 속도를 보였습니다. 특히, 지리정보를 고려한 SSL 적용이 모델 성능을 향상시키는 데 기여하며, 복잡한 지역에서도 안정적인 결과를 나타냈습니다. 이 방법론은 구름 연구 분야에서 머신러닝 기법의 활용 가능성을 보여줍니다.



### Dynamic Feature Fusion: Combining Global Graph Structures and Local Semantics for Blockchain Fraud Detection (https://arxiv.org/abs/2501.02032)
- **What's New**: 블록체인 기술의 발전과 함께 스마트 계약의 채택이 금융 분야에서 급증하고 있지만, 기존의 사기 탐지 방법에는 거래 네트워크 내의 글로벌 구조 패턴과 거래 데이터의 로컬 의미 관계를 모두 포착하는 데 한계가 있다. 본 논문에서는 그래프 기반 표현 학습(Graph-based Representation Learning)과 의미 특징 추출(Semantic Feature Extraction)을 결합한 동적 특징 융합 모델을 제안하여 블록체인 사기 탐지를 개선한다.

- **Technical Details**: 제안된 모델은 글로벌 계정 상호작용 그래프(Global Account Interaction Graph)를 구축해 블록체인 거래 계정 간의 관계를 모델링하며, 그래프에서 구조적 특징을 추출한다. 동시에, 거래 데이터에 포함된 의미 정보를 사전 훈련된 텍스트 표현 모델을 이용하여 처리하며, 로컬 맥락 관계를 인식할 수 있도록 한다. 이를 통해 모델이 구조적 및 의미적 사기 패턴을 효과적으로 탐지할 수 있도록 동적 특징 융합 메커니즘을 개발하였다.

- **Performance Highlights**: 실험 결과, 제안된 ETH-GBERT 모델은 실제 블록체인 데이터셋에서 뛰어난 성능을 보이며, F1 스코어가 94.71%에 이르렀다. 이 결과는 기존의 최고 성능 모델인 Role2Vec의 74.13%보다 20.58% 높은 수치이다. 또한, B4E 데이터셋에서는 정확도 90.84%, 재현율 89.57%를 기록하여 다른 기준 모델들에게서 월등한 성과를 발휘하였다.



### CarbonChat: Large Language Model-Based Corporate Carbon Emission Analysis and Climate Knowledge Q&A System (https://arxiv.org/abs/2501.02031)
Comments:
          26 pages

- **What's New**: 이 논문은 CarbonChat: 대규모 언어 모델 기반의 기업 탄소 배출 분석 및 기후 지식 Q&A 시스템을 제안합니다. 이 시스템은 복잡한 문제에 대한 기존의 증강 생성 아키텍처의 전문성과 정확성 부족 문제를 해결하는 데 중점을 두고 있습니다. 또한 탄소 배출 보고서를 분석하는 데 드는 시간과 비용을 줄이기 위해 다양한 데이터 인덱싱 모듈을 개발했습니다.

- **Technical Details**: CarbonChat은 의도 인식(intent recognition), 구조적 추론 체인(structured reasoning chains), 하이브리드 검색(hybrid retrieval), 및 Text2SQL 기술을 통합하여 생성의 효율성을 높입니다. 이 시스템은 온실가스 회계 프레임워크를 기반으로 14개의 차원으로 탄소 배출 분석을 수행하여 맞춤형 응답을 제공하며, 다층 청크(chunking) 메커니즘을 통해 결과의 정확도와 검증 가능성을 보장합니다.

- **Performance Highlights**: 이 시스템은 사용자에게 정확하고 포괄적인 정책 및 규제 참조를 제공하며, 보고서 요약 및 관련성 평가를 통해 기업의 지속 가능성 보고서에 대한 심층 분석을 수행합니다. 특히, 다양한 인덱싱 모듈과 환각 탐지 기능을 통해 결과의 정확성과 검증 가능성을 크게 향상시켰습니다.



### Detecting Music Performance Errors with Transformers (https://arxiv.org/abs/2501.02030)
Comments:
          AAAI 2025

- **What's New**: 이번 연구에서는 초보 음악가의 공연 오류를 감지하기 위해 Polytune이라는 새로운 transformer 모델을 제안합니다. 이 모델은 오디오 입력을 받고 주석이 달린 악보를 출력하여, 자동 정렬에 의존하는 기존 접근 방식을 개선합니다. 또한, CocoChorales-E 및 MAESTRO-E라는 대규모 합성 음악 오류 데이터 세트를 생성하는 방법을 소개하여 충분한 훈련 데이터를 제공합니다.

- **Technical Details**: Polytune은 오디오 스펙트로그램 쌍을 입력으로 사용하고 주석이 달린 음악 점수를 출력하는 end-to-end 학습 가능한 접근 방식을 채택합니다. 이 모델은 ‘정확함’ 또는 ‘잘못됨’이라는 기본 레이블을 넘어 다양한 라벨로 노트를 주석화하도록 훈련됩니다. 기존 데이터의 합성을 통해 훈련 샘플이 부족한 문제를 해결하고, Fine-grained한 음악 성과 피드백이 가능합니다.

- **Performance Highlights**: Polytune 모델은 음악 오류 탐지에서 최첨단 성능을 달성하며, 정확도 F1 점수는 95.0%입니다. 연구 결과는 14개 악기에서 이전 방법들보다 평균 40% 향상된 성과를 보였으며, 오류 탐지의 정확성을 개선했습니다. 또한 우리의 결과는 다양한 악기에 대해 잘 작동하며, 오류 탐지와 음악 전사 작업 모두에서 혁신적인 기여를 합니다.



### Spot Risks Before Speaking! Unraveling Safety Attention Heads in Large Vision-Language Models (https://arxiv.org/abs/2501.02029)
- **What's New**: 이 연구는 추가 모달리티의 통합으로 인해 대형 비전-언어 모델(LVLMs)의 안전 리스크에 대한 취약성을 조사합니다. 특히, LVLM의 내부 활성화가 악의적인 프롬프트를 효과적으로 탐지할 수 있는지에 관한 분석을 제공합니다. 저자들은 '안전 헤드'(safety heads)라고 명명된 특정 주의 헤드가 이러한 탐지 메커니즘에서 중심적인 역할을 한다는 것을 발견했습니다.

- **Technical Details**: LVLMs는 비전 인코더와 LLM 텍스트 디코더를 포함하는 아키텍처로 구성되어 있습니다. 이 연구에서 저자들은 LLM의 첫 번째 토큰 생성 과정에서의 내부 활성화를 활용하여 공격을 구분할 수 있는 능력에 대해 설명하고, 이러한 활성화를 통해 안전 헤드를 식별할 수 있다고 강조합니다. 이 안전 헤드는 악의적인 시도를 막는 특수한 '방패' 역할을 수행하여, 이 헤드를 제거하면 공격 성공률이 높아진다는 사실을 발견했습니다.

- **Performance Highlights**: 제안된 악의적 프롬프트 탐지기는 로지스틱 회귀 모델을 기반으로 하며, LVLM 생성 과정에 통합되는 방식으로 설계되었습니다. 이 모델은 공격 성공률을 80% 이상에서 1-5%로 낮추는 성능을 보여주며, 기존 모델보다 빠르게 작동합니다. 또한, 이 탐지기는 새로운 데이터셋에 대한 제로샷 제너럴리제이션(zero-shot generalization) 능력을 보이며, 다양한 적대적 공격에 대한 강한 전이 가능성을 입증했습니다.



### Recursive Decomposition of Logical Thoughts: Framework for Superior Reasoning and Knowledge Propagation in Large Language Models (https://arxiv.org/abs/2501.02026)
- **What's New**: 이번 연구에서는 LLM의 추론 능력을 획기적으로 향상시킬 수 있는 새로운 프레임워크인 RDoLT(Recursive Decomposition of Logical Thought prompting)를 소개합니다. RDoLT는 복잡한 추론 작업을 점진적으로 단순화하는 서브태스크로 분해하고, 유망한 추론 결과를 식별하기 위한 고급 선택 및 평가 메커니즘을 사용하며, 인간 학습을 모방한 지식 전파 모듈을 통합하여 향상된 성능을 제공합니다.

- **Technical Details**: RDoLT는 세 가지 주요 혁신에 기반합니다: (1) 복잡한 추론 작업을 점진적으로 단순해지는 하위 작업으로 재귀적으로 분해하고, (2) 논리적 유효성, 일관성, 단순성 및 적응성의 4가지 차원을 기준으로 사고를 평가하는 강력한 사고 평가 시스템을 도입하며, (3) 이전에 거부된 사고를 저장 및 전파하여 잠재적인 가치를 재탐색할 수 있는 지식 전파 모듈(Knowledge Propagation Module, KPM)을 제공합니다.

- **Performance Highlights**: RDoLT는 다양한 벤치마크에서 LLM의 성능을 유의미하게 향상시켰으며, 특히 GSM8K에서 ChatGPT-4가 90.98%의 정확도로 기존 최고 기술보다 6.28% 높은 성능을 달성했습니다. 다른 벤치마크에서도 유사한 개선을 보였으며, 정확도 향상이 5.5%에서 6.75%에 이르는 것으로 나타났습니다. 이러한 결과는 복잡한 추론 작업에 대한 RDoLT의 효과적인 접근 방식을 강조합니다.



### Model Checking in Medical Imaging for Tumor Detection and Segmentation (https://arxiv.org/abs/2501.02024)
- **What's New**: 이 논문에서는 최근의 모델 체크(model checking) 기술이 신호(signal) 및 이미지(image) 분석에서 매우 유망한 가능성을 보여주고 있으며, 특히 의료 이미징(medical imaging) 분야에서 그 중요성이 강조되고 있습니다. 새로운 자동 및 반자동(natural language) 이미지 분할 기법을 통해 이미지 내에서 관심 영역(region of interest)을 정확하게 delineation 할 수 있는 프레임워크를 설계하고 평가하는 방법을 다룹니다.

- **Technical Details**: 이 연구는 공간 논리(spatial logic)를 활용하여 종양(tumorous) 및 비종양(non-tumorous) 영역을 식별하는 연산자(operators) 및 도구(tools)를 개발하는 최근의 연구를 포괄적으로 분석합니다. 또한, 공간 모델 체크 기법이 장소적 데이터(ground truth data)의 변동성(variability)으로 인해 직면하는 여러 도전과제를 논의하며, 임상 실습(clinical practice)을 위한 간편한 절차(steps)의 필요성을 강조합니다.

- **Performance Highlights**: 이 프레임워크는 의료 이미징에서 발생할 수 있는 많은 변수를 고려하면서도, 정확한 세분화를 통해 전문가들이 보다 나은 진단을 내릴 수 있도록 지원합니다. 특히, 자동 및 반자동 방식으로 관심 영역을 구분할 수 있는 점에서, 펀드멘털(fundamental)한 영향을 미칠 가능성이 큽니다.



### Weakly Supervised Learning on Large Graphs (https://arxiv.org/abs/2501.02021)
- **What's New**: 이 논문은 패스톨로지(pathology) 분야에서 그래프 분류(graph classification)를 위한 약한 감독(weak supervision) 기반의 새로운 프레임워크를 제안합니다. 특히, 이미지를 그래프로 표현하고, 내부의 특정 위치에서 복잡한 패턴을 식별하는 데 중점을 두었습니다. 이 프레임워크는 슬라이딩 윈도우(sliding-window)와 BFS 기반(BFS-based) 두 가지 서브그래프 추출 기술을 활용하여 그래프 레벨 레이블을 서브그래프에 전파함으로써 상세한 주석(annotation)이 필요하지 않도록 합니다.

- **Technical Details**: 서브그래프 추출에는 두 가지 방법이 사용됩니다. BFS 기반 방법은 랜덤 노드에서 시작하여 깊이 제한(depth limit)까지 그래프를 탐색하면서 의미 있는 서브그래프를 수집합니다. 슬라이딩 윈도우 방법은 고정된 크기의 윈도우를 사용하여 그래프의 특定 노드를 포함하는 서브그래프를 반복적으로 생성하며, 이 두 방법 모두 Graph Attention Network(GAT)를 통해 처리되어 가장 정보가 풍부한 서브그래프를 식별합니다.

- **Performance Highlights**: 제안된 방법은 D&D 및 MSRC-21 데이터셋에서 평가되었으며, 경쟁력 있는 정확도를 달성했습니다. 또한, 모델의 해석 가능성(interpretable insights)도 제공하여 의료 이미징 분야의 그래프 기반 분석에 유용성을 더합니다. GAT 모델을 통해 개별 노드의 중요도를 반영하여 특성을 집계할 수 있어, 보다 효과적인 그래프 분류가 가능합니다.



### Enhancing Uncertainty Modeling with Semantic Graph for Hallucination Detection (https://arxiv.org/abs/2501.02020)
- **What's New**: 이번 연구에서는 LLMs(대형 언어 모델)에서 발생하는 환각(hallucination) 문제를 해결하기 위해, 기존의 단어마다의 불확실성(uncertainty)만을 고려하는 방법과는 달리, 의미 그래프(semantic graph)를 이용한 불확실성 모델링을 제안합니다. 연구자들은 엔티티 토큰(entity tokens)과 문장 간의 관계를 효과적으로 포착하는 의미 그래프를 구축하였으며, 문장 수준의 환각 탐지를 강화하기 위해 두 엔티티 간의 관계를 통합하여 불확실성을 전파하는 방법을 개발했습니다.

- **Technical Details**: 연구의 핵심 접근 방식은 두 가지로 요약됩니다. 첫째, AMR(Abstract Meaning Representation) 기반 파싱을 통해 각 문서의 의미 그래프를 생성합니다. 둘째, 이 그래프를 활용하여 문장과 이웃 사이의 관계를 포함한 불확실성을 보정하여, 더욱 정교한 환각 탐지 방법을 제공합니다. 이를 통해 문장 수준 및 패시지 수준에서 보다 정교한 불확실성 계산을 수행할 수 있습니다.

- **Performance Highlights**: 두 개의 데이터셋(WikiBio 및 NoteSum)을 통해 수행된 실험 결과, 제안된 접근 방식이 문장 및 패시지 수준의 환각 탐지에서 기존 방법들보다 19.78%의 향상된 성능을 보임으로써 그 우수성을 입증하였습니다. 이는 의미 그래프를 활용한 새로운 불확실성 모델링 기법이 환각 탐지의 정확성을 현저히 향상시킬 수 있음을 시사합니다.



### Benchmarking Constraint-Based Bayesian Structure Learning Algorithms: Role of Network Topology (https://arxiv.org/abs/2501.02019)
Comments:
          8 Pages, 4 Figures

- **What's New**: 이번 연구는 여러 실제 세계 엔티티의 다변량 단면 프로파일 사이의 연관성을 모델링하는 중요성을 보여줍니다. 특히, 기존의 Bayesian Structure Learning (BSL) 알고리즘의 성과를 비교하는 데 있어 네트워크 topology의 역할을 강조합니다. 이 연구는 표준적인 벤치마킹 및 성능 평가의 관점을 넓히는 데 기여합니다.

- **Technical Details**: 연구는 Peter-Clarke, Grow-Shrink, Incremental Association Markov Blanket의 세 가지 인기 있는 BSL 알고리즘의 민감도(sensitivity)를 조사합니다. 각 알고리즘은 동일한 노드, 엣지, 샘플 사이즈를 유지하면서 다양한 네트워크 topology(서브선형, 선형, 슈퍼선형)에서의 민감도 변이를 분석합니다. 사용된 샘플 사이즈는 $N = 2^{10}$이며, 노드 수는 48 및 64로 설정되었습니다.

- **Performance Highlights**: 연구 결과는 서브선형 topology에서 슈퍼선형 topology로 변화할 때, 세 가지 알고리즘 모두에서 민감도 추정이 통계적으로 유의미하게 감소한다는 것을 발견했습니다. 이 감소는 $	ext{(}eta = 0.05 	ext{)}$ 수준에서 검토되었으며, 이는 각 알고리즘의 상관 관계를 이해하는 데 중요한 시사점을 제공합니다. 따라서 네트워크의 topology를 고려하는 것이 BSL 알고리즘의 성과 평가에 필수적임을 알 수 있습니다.



### Safeguarding Large Language Models in Real-time with Tunable Safety-Performance Trade-offs (https://arxiv.org/abs/2501.02018)
- **What's New**: 본 연구에서는 Large Language Models (LLMs)의 안전성을 향상시키기 위한 새로운 방법인 SafeNudge를 소개하고 있습니다. 이 방법은 Controlled Text Generation (CTG)과 'nudging'을 결합하여 모델이 위험한 출력을 생성하는 것을 실시간으로 방지하는 데 중점을 두고 있습니다. SafeNudge는 jailbreak 공격이 발생한 후 활성화되어 LLM이 안전한 응답으로 유도되도록 도와줍니다.

- **Technical Details**: SafeNudge는 LLM의 안전성과 성능 간의 trade-off를 조절할 수 있는 장치를 제공합니다. 이 방법은 jailbreak 공격에 대한 성공적인 시도를 30% 감소시키며, 일반적인 모델 행동에는 5%의 악화만을 초래합니다. 또한, SafeNudge는 Hugging Face의 transformers 라이브러리에 호환되어 연구자들에게 쉽게 사용될 수 있습니다.

- **Performance Highlights**: SafeNudge는 모델의 텍스트 생성 지연을 최소화하면서도 해로운 응답 생성을 효과적으로 줄이는 성능을 보여주었습니다. 예를 들어, 기본 설정에서 NVIDIA A100 GPU를 사용할 때 안전하지 않은 응답 생성을 30.4% 감소시키는 것으로 나타났습니다. 전반적으로 SafeNudge는 상대적으로 합리적인 Safety-Performance Trade-offs (SPTs)를 통해 강력한 안전성을 제공하는 것을 확인했습니다.



### ST-HCSS: Deep Spatio-Temporal Hypergraph Convolutional Neural Network for Soft Sensing (https://arxiv.org/abs/2501.02016)
Comments:
          Accepted at the 2025 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2025)

- **What's New**: 이 논문에서는 복잡한 센서 데이터의 상호작용을 모델링하기 위해 고차 하이퍼그래프(hypergraph) 기반의 깊이 있는 시공간 스파이오센싱 소프트 센서(ST-HCSS)를 제안합니다. ST-HCSS는 사전 지식 없이도 센서 노드 간의 복잡한 상호작용을 캐치할 수 있는 기능을 특징으로 합니다. 이전의 소프트 센서 방법들과 비교하여 특히 시공간 관계를 충분히 파악할 수 있는 새로운 방법론을 제공합니다.

- **Technical Details**: 제안된 방법에서는 시공간척도 시뮬레이션을 위하여 슬라이딩 윈도우(sliding window)를 사용하고, 시뮬레이션 과정에서 하이퍼그래프를 설정하여 비유클리드적 관계를 통합합니다. 이를 통해 보조 변수와 주 변수를 효과적으로 예측할 수 있는 소프트 센서 모델을 구현합니다. 또한, 이 모델은 겹치는 시계열 입력을 처리하며, 다양한 센서 노드 간의 복잡한 관계를 정확하게 표현할 수 있도록 하였습니다.

- **Performance Highlights**: 실험 결과, ST-HCSS는 기존의 선도적인 소프트 센서 방법들에 비해 우수한 성능을 보여줍니다. 학습된 하이퍼그래프 특징 표현은 센서 데이터 상관관계와 잘 일치하여 더욱 정밀한 공정 모니터링 및 제어가 가능합니다. 이 모델은 실제 산업 공정에서의 적용 가능성을 보여주며 고감도의 소프트 센싱에 필수적인 시공간 동력을 극대화하고 있습니다.



### KANS: Knowledge Discovery Graph Attention Network for Soft Sensing in Multivariate Industrial Processes (https://arxiv.org/abs/2501.02015)
Comments:
          Accepted at IEEE International Conference on Systems, Man, and Cybernetics (IEEE SMC 2024)

- **What's New**: 본 논문은 산업 공정에서 측정하기 어려운 변수들을 소프트 센싱(Soft sensing)하는 데 새롭고 효과적인 프레임워크인 KANS(Knowledge discovery graph Attention Network)를 제시합니다. 기존의 딥러닝 모델과 달리 KANS는 복잡한 공정 변수 간의 비선형적이고 동적인 관계를 사전 정의된 토폴로지 없이 발견할 수 있습니다. 이를 통해 센서들 간의 본질적인 상관관계와 불규칙한 관계를 보다 효과적으로 탐색할 수 있습니다.

- **Technical Details**: KANS는 먼저 다양한 센서의 임베딩(embedding) 간 코사인 유사성을 활용한 비지도 그래프 구조 학습 방법을 도입하며, 이를 통해 센서 간의 상관관계를 캡처합니다. 그 다음, 그래프 기반의 주의(attention)를 활용한 표현 학습(representation learning)을 통해 다변량(multivariate) 데이터를 병렬로 처리하여 복잡한 센서 노드와 엣지를 학습하는 과정이 포함되어 있습니다. 이러한 접근은 모델의 해석 가능성(interpretability)을 높이기 위한 지식 발견 분석(knowldge discovery analysis)을 통해 추가적으로 뒷받침됩니다.

- **Performance Highlights**: 실험 결과, KANS는 기존의 모든 기준선 모델과 최신 기법 대비 소프트 센싱 성능에서 현저하게 우수한 결과를 보였습니다. 또한, KANS는 도메인 지식 없이도 서로 관련이 깊은 센서를 탐색할 수 있어 소프트 센싱의 정확성을 대폭 향상시킨다는 분석 결과가 나타났습니다. 이러한 특성 덕분에 KANS는 복잡한 산업 공정에서의 실용성이 높아질 것으로 기대됩니다.



### Machine Learning-Based Differential Diagnosis of Parkinson's Disease Using Kinematic Feature Extraction and Selection (https://arxiv.org/abs/2501.02014)
- **What's New**: 이번 연구는 파킨슨병(Parkinson's disease, PD)과 기타 신경퇴행성 질환의 차별 진단을 위한 새로운 머신러닝 기반 시스템을 제안합니다. 이 시스템은 기존의 MDS-UPDRS 평가 척도의 주관성을 극복하고, 진단의 정확성을 높이는 것을 목표로 합니다. 특히, 두 가지 새로운 운동 제어 패턴을 반영한 기동학적 특징(kinematic features)을 포함하여, 다양한 통계적 특징(statistical features)을 추출합니다.

- **Technical Details**: 시스템은 18개의 기동학적 특징을 포함하여 총 41개의 통계적 특징을 활용하여, 진단에 필요한 정보를 제공합니다. 특징 선택(feature selection)은 One-way ANOVA를 통해 수행되고, 이후 Sequential Forward Floating Selection (SFFS) 방법으로 가장 관련성이 높은 특징을 확인합니다. 이로써 계산 복잡도를 줄이며 효율적인 분류(classification)를 가능하게 합니다.

- **Performance Highlights**: 최종 모델은 각 데이터셋에 대해 66.67%의 분류 정확도(classification accuracy)를 달성했으며, 각 환자에 대해서는 88.89%의 결과를 보였습니다. 특히, SVM 알고리즘을 활용했을 때 MSA 및 건강한 대조군(healthy controls)에서 높은 성능을 보였습니다. 이 시스템은 향후 임상적인 진단 도구로서 신속하고 정확한 진단에 기여할 잠재력을 가지고 있으나, 신뢰성을 높이기 위한 추가 데이터 수집과 정제가 필요합니다.



### Cross-model Transferability among Large Language Models on the Platonic Representations of Concepts (https://arxiv.org/abs/2501.02009)
- **What's New**: 이 논문은 다양한 대규모 언어 모델(LLMs) 간의 개념 표현에 대한 새로운 접근 방식을 제시하며, 이는 플라톤의 동굴의 비유와 유사한 관계를 탐구합니다. 연구진은 LLM 간의 개념 표현을 간단한 선형 변환(Linear Transformation)을 이용해 효과적으로 정렬할 수 있음을 발견했습니다. 또한, 이 연구는 소형 LLM에서 추출된 쉬운 steering vector (SV)가 대형 LLM의 행동을 효과적으로 제어할 수 있음을 밝혀내었습니다.

- **Technical Details**: 연구에서는 L-Cross Modulation이라는 선형 변환 방법론을 제안하며, 이 방법은 LLM 간의 개념 공간을 정렬하고 SV의 이식성을 달성하는 데 도움을 줍니다. 이 방법은 일반 최소 제곱 회귀(Ordinary Least Squares Optimization)를 통해 생성된 변환 행렬(T)을 사용하여, 원본 LLM의 SV를 목표 LLM의 표현 공간으로 매핑하는 방식을 취합니다. 이 과정을 통해, 연구진은 11개의 벤치마크 개념을 기반으로 cross-model transferability 능력을 평가했습니다.

- **Performance Highlights**: 연구 결과 L-Cross Modulation은 LLM을 조정하는 데 효과적임을 보여주었으며, 예를 들어 해롭다는 개념을 적용했을 때 90%의 출력에서 해로운 콘텐츠 생성을 유도했습니다. 또한, 다양한 개념 간 선형 변환의 강한 일반화 능력이 있으며 서로 다른 개념이 두 LLM 간 동일한 선형 변환을 공유할 수 있음을 발견하였습니다. 마지막으로, 소형 LLM의 SV가 대형 LLM의 응답을 효과적으로 조정할 수 있는 가능성이 있음을 확인했습니다.



### TART: Token-based Architecture Transformer for Neural Network Performance Prediction (https://arxiv.org/abs/2501.02007)
- **What's New**: 이 논문은 Transformer를 활용하여 새로운 신경망 아키텍처를 생성할 가능성을 탐구합니다. Token-based Architecture Transformer (TART)라는 새로운 접근 방식을 제안하며, 후보 네트워크를 훈련시키지 않고도 신경망의 성능을 예측할 수 있습니다. TART는 DeepNets-1M 데이터셋에서 최첨단 성능을 달성하였으며, 이는 Transformer가 혁신적이고 높은 성능의 신경 아키텍처를 발견하는 데 기여할 수 있음을 시사합니다.

- **Technical Details**: 기존의 Neural Architecture Search (NAS) 방법은 수작업으로 후보 네트워크를 선택하고 훈련, 평가하는 데 많은 비용과 시간이 소요됩니다. 이 과정에서 Transformer의 강력한 토큰 처리 기능을 활용하여 신경망 아키텍처를 토큰으로 표현할 수 있는데, 이는 새로운 신경망 아키텍처를 효율적으로 학습하고 탐색하는 데 도움을 줍니다. TART는 후보 아키텍처의 성능을 예측하는 데 사용되며, 직접적인 훈련 없이도 성능을 평가할 수 있다는 이점이 있습니다.

- **Performance Highlights**: TART는 Edge 정보 없이 DeepNets-1M 데이터셋에서 성능 예측 작업에 대해 최첨단 성능을 달성하였습니다. 본 연구는 과거의 연구 결과를 바탕으로 신경망 아키텍처의 토큰화가 Transformer의 성능 예측 능력을 향상시키는 데 기여함을 보여줍니다. 이는 향후 Transformer를 기반으로 새로운 신경 아키텍처를 생성하기 위한 기초를 마련하게 됩니다.



### Multi-Task Semantic Communication With Graph Attention-Based Feature Correlation Extraction (https://arxiv.org/abs/2501.02006)
Comments:
          18 pages,11 figures, accepted by IEEE TMC

- **What's New**: 이 논문에서는 multi-task semantic communication 시스템을 위한 새로운 graph attention inter-block (GAI) 모듈을 제안합니다. 기존 모델들이 각 기능 블록에서 추출되는 특성 간의 관계를 고려하지 않았던 반면, GAI 모듈은 인코더의 중간 출력값을 그래프의 노드로 해석하여 이를 다중 작업을 위해 풍부한 특성을 제공합니다. GAI 모듈은 그래프 어텐션 메커니즘을 통해 노드 표현을 정제하고, 각 작업에 대한 특성의 상관관계를 포착합니다.

- **Technical Details**: GAI 모듈은 인코더의 다양한 기능 추출 블록에서 추출된 중간 특성을 노드로 해석하여 그래프를 구축합니다. 이들은 Feature Transformation Layer를 통해 통일된 노드 표현으로 표준화 됩니다. 이어서 Graph Attention Layer를 사용하여 중간 특성 간의 상관관계를 반복적으로 캡처하고 이를 강화하며, Relation Mapping Layer를 통해 각 작업에 맞는 task-node 가중치를 생성합니다.

- **Performance Highlights**: 실험 결과, 제안하는 GAI 모듈은 통신 채널의 대역폭 비율이 1.3일 때, CityScapes 2Task 데이터셋에서 평균 정확도를 2.71% 향상시켰습니다. NYU v2 데이터셋의 두 가지 및 세 가지 작업의 semantic communications에서도 각각 최소 0.87% 및 0.80%의 성능 향상을 보였으며, 이를 통해 GAI 모듈이 기존의 최첨단 모델들을 초과하는 성능을 입증하고 있습니다.



### General Information Metrics for Improving AI Model Training Efficiency (https://arxiv.org/abs/2501.02004)
- **What's New**: 본 연구에서는 AI 모델 학습 데이터의 크기 증가와 보편적인 데이터 선택 방법론의 부족을 해결하기 위해 General Information Metrics Evaluation (GIME) 방법을 제안합니다. GIME는 Objective Information Theory (OIT)의 일반 정보 메트릭스를 활용하여 데이터 세트 선택을 최적화하며, 실험을 통해 모델 성능을 유지하면서 학습 시간과 비용을 크게 줄일 수 있음을 입증했습니다. 특히, 사법 AI 프로그램에서 GIME를 적용함으로써 총 모델 학습 비용이 39.56% 감소하는 놀라운 결과를 얻었습니다.

- **Technical Details**: GIME는 볼륨(volume), 지연(delay), 범위(scope), 세분화(granularity), 다양성(variety), 지속(duration), 샘플링 속도(sampling rate), 집계(aggregation), 커버리지(coverage), 왜곡(distortion), 불일치(mismatch)와 같은 11개의 일반 정보 메트릭스를 사용하여 데이터 세트를 평가합니다. 이 방법은 훈련 전에 데이터 세트를 체계적으로 평가하여 자원이 비효율적으로 낭비되는 것을 방지하고, 데이터가 모델 목표에 부합하는지 미리 판단하도록 돕습니다. GIME는 특정 AI 모델의 구조나 알고리즘에 의존하지 않으며, 데이터 분포에 대한 사전 지식이 필요하지 않습니다.

- **Performance Highlights**: 다양한 분야에서 실시한 실험을 통해 GIME가 모델 성능을 효과적으로 유지하면서도 학습 비용과 시간을 줄이는 것을 확인했습니다. CTR 예측, 민사 사건 예측, 날씨 예측과 같은 작업에서 GIME가 상당한 효율성을 달성했으며, 사법 프로그램 내에서의 적용 사례는 데이터 크기, 인력 소모, 에너지 소비 및 개발 비용의 강력한 감소를 입증했습니다. 이러한 성과는 AI 모델 훈련의 예측 가능성을 향상시키고, 비용을 크게 줄이는 데 기여할 수 있음을 나타냅니다.



### Multi-Center Study on Deep Learning-Assisted Detection and Classification of Fetal Central Nervous System Anomalies Using Ultrasound Imaging (https://arxiv.org/abs/2501.02000)
- **What's New**: 이 연구는 임신 중 태아 성장과 선천적 이상을 평가하기 위해 딥러닝 모델을 활용하여 태아 두개 이상 진단의 정확도를 향상시키는 것을 목표로 합니다. 수집한 다기관 데이터셋에서 태아의 네 가지 주요 중추신경계 이상(anencephaly, encephalocele, holoprosencephaly, rachischisis)에 대한 진단 정확도가 94.5%에 달하며, AUROC 값은 99.3%에 이릅니다. 이 연구는 인공지능(AI) 기술이 특히 진단 과정에서 어떤 영향을 미칠 수 있는지를 보여줍니다.

- **Technical Details**: 이번 연구는 비침습적이고 저비용인 산전 초음파(ultrasonography)의 어려움을 극복하기 위해 딥러닝(deep learning) 기반의 알고리즘을 개발하였습니다. CNN(Convolutional Neural Networks)을 사용하여 태아 두개 영상의 자동 탐지 및 분류를 수행하며, 이 시스템은 초음파 이미지 위에 히트맵(heatmap)을 겹쳐 시각적 해석을 제공함으로써 의사들에게 중요한 영역을 강조합니다. 이로 인해 임상에서의 정확성을 높이고 진단 효율 또한 향상되며, 오진율 감소라는 긍정적 효과로 이어집니다.

- **Performance Highlights**: 연구 결과, 딥러닝 모델은 전체 임신 기간 동안 태아 이상 유형을 잘 식별할 수 있으며, 이로 인해 진단 과정이 대폭 개선될 것으로 기대됩니다. 연구의 회고적 독자 연구는 DL 시스템의 자동 예측과 방사선 전문의의 전문 판단을 결합함으로써 진단 정확성과 효율성을 크게 향상시킬 수 있음을 보여줍니다. 이를 통해 최종적으로 환자의 불필요한 검사율을 줄이는 데 기여할 것입니다.



### On the Utility of Equivariance and Symmetry Breaking in Deep Learning Architectures on Point Clouds (https://arxiv.org/abs/2501.01999)
Comments:
          21 pages, 5 figures

- **What's New**: 이 논문에서는 점 구름(point clouds) 모델이 다양한 기하학적 복잡성을 가진 작업에서의 성능에 미치는 주요 요인들을 탐구합니다. 특히, 등가성 계층(equivariant layers)의 유연성과 가중치 공유(weight-sharing) 간의 트레이드오프(trade-off)를 분석하고, 등가성이 모델 성능에 미치는 긍정적 또는 부정적 영향을 평가합니다. 추가적인 정보가 모델 성능을 향상시키는 것이 일반적으로 받아들여지지만, 이러한 정보가 특정 속성을 손상시킬 경우 여전히 유익한지에 대한 질문을 다룹니다.

- **Technical Details**: 등가성 신경망(equivariant neural networks)은 비등가성 신경망(non-equivariant networks)과 비교하여, 고차원 표현 공간(high-dimensional representation spaces)을 통해 데이터의 구조적 특성을 보존하는 제약 계층(constraint layers)을 사용합니다. 이들 계층은 다양한 변환에 대해 비효율 반복 학습을 방지하며, 데이터 효율성(data efficiency)을 강화합니다. 또한 전통적인 컨볼루션 네트워크와 비교하여 G-CNNs(Group Convolutional Neural Networks)는 가중치 공유나 성능 향상을 보다 효율적으로 달성할 수 있는 장점을 제공합니다.

- **Performance Highlights**: 연구 결과, 점 구름 데이터셋인 Shapenet 3D, QM9, CMU Motion Capture를 활용한 다양한 작업에서 등가성 계층의 사용이 성능에 긍정적인 영향을 미친 것으로 나타났습니다. 특히 작업의 복잡성이 증가할수록 등가성의 효과가 두드러지게 나타났습니다. 또한, G-CNN이 비등가성 네트워크와 비교하여 모든 작업에서 일관되게 우수한 성능을 보임으로써, 이들 계층의 도입이 전반적인 모델 성능을 증대시키는 데 기여했음을 확인했습니다.



### SmartSpatial: Enhancing the 3D Spatial Arrangement Capabilities of Stable Diffusion Models and Introducing a Novel 3D Spatial Evaluation Framework (https://arxiv.org/abs/2501.01998)
- **What's New**: 이번 논문은 Stable Diffusion 모델의 한계를 극복하기 위해 SmartSpatial이라는 새로운 접근 방식을 제안합니다. SmartSpatial은 3D 인식(3D-aware) 조건부 및 주의 기반 메커니즘(attention-guided mechanisms)을 통해 공간 배치(spatial arrangement) 성능을 향상시킵니다. 이를 통해 복잡한 3D 관계를 정확하게 표현하고, 공간 정확도(spatial accuracy) 메트릭에서 현저한 개선을 보여줍니다.

- **Technical Details**: SmartSpatial은 깊이 정보(depth information)를 통합하고 교차 주의 제어(cross-attention control)를 사용하여 객체의 정확한 배치를 보장합니다. 논문에서는 새롭게 제안된 평가 프레임워크인 SmartSpatialEval을 통해 공간 관계를 평가하는 방법도 소개하며, 비전-언어 모델(vision-language models) 및 그래프 기반 종속 구문 분석(graph-based dependency parsing)을 활용합니다. 이를 통해 기존 방법에 비해 우수한 성능을 보이고 있습니다.

- **Performance Highlights**: 실험 결과는 COCO 및 SpatialPrompts 데이터셋에서 SmartSpatial이 기존 방법보다 현저하게 우수한 성능을 보여줌을 강조합니다. SmartSpatial은 이미지 생성에서 공간 배열 정확도(spatial arrangement accuracy)의 새로운 기준을 세우는 데 기여하였습니다. 이러한 성능 향상은 공간 관계의 이해 및 표현에서의 혁신을 제공합니다.



### Fuzzy Model Identification and Self Learning with Smooth Compositions (https://arxiv.org/abs/2501.01994)
- **What's New**: 이 논문은 동적 시스템을 위한 매끄러운 모델 식별(smooth model identification) 및 자기 학습(self-learning) 전략을 개발합니다. 이는 시스템의 가능한 매개변수 변화와 불확실성을 고려하여 모델이 지속적이고 매끄러운 표면(continuous and smooth surface)에서 변화에 따라 따르도록 하는 문제를 해결하려고 합니다.

- **Technical Details**: 모델을 실행하여 매끄러운 표면에서 매개변수의 최적값을 적응적으로 획득하는 방식을 통해, MPC(모델 예측 제어) 또는 강건 제어(robust control) 알고리즘과 같은 다른 파생 기반 최적화 제어 알고리즘의 적용을 개선할 수 있습니다. 기존의 매끄러운 퍼지 모델 구조(smooth fuzzy modeling structures)와 비교하여, 우리는 모델의 최적성과 계산 부하(computational load) 간의 바람직한 균형(trade-off)을 달성할 수 있었습니다.

- **Performance Highlights**: 제안된 방법은 테스트 문제(test problem)와 화학 공정(chemical process)의 비선형 동적(non-linear dynamic)에 대해 평가되었습니다. 이 연구는 동적 시스템의 모델링과 제어(measuring-control) 결합 방안에서도 향상된 성능을 보여주었습니다.



### A Hybrid Deep Learning and Model-Checking Framework for Accurate Brain Tumor Detection and Validation (https://arxiv.org/abs/2501.01991)
Comments:
          22 pages, 8 figures

- **What's New**: 이 논문에서는 모델 검사(model checking)와 딥러닝(deep learning)을 통합한 새로운 하이브리드 프레임워크를 소개합니다. 이 프레임워크는 뇌 종양 탐지 및 의료 영상에서의 검증에 중점을 두고 있습니다. 모델 검사 원칙과 CNN 기반 특징 추출(feature extraction), K-FCM 클러스터링(clustering)을 통해 종양 탐지의 신뢰성을 향상시킵니다.

- **Technical Details**: 제안된 접근 방식은 CNN(convolutional neural network)을 사용하여 이미지에서 특징을 추출하고, K-FCM(k-means fuzzy c-means) 클러스터링 기법을 통해 세분화를 수행합니다. 이러한 기술은 전통적인 모델 검사의 원칙을 활용하여 중요 데이터의 신뢰성을 보장합니다. 이 하이브리드 시스템은 복잡한 의료 영상 데이터를 처리하는 데 강력한 방법론을 제공합니다.

- **Performance Highlights**: 실험 결과는 이 프레임워크의 효과성을 입증하며, 98	ext{%}의 정확도(accuracy), 96.15	ext{%}의 정밀도(precision), 100	ext{%}의 재현율(recall)을 기록했습니다. 이러한 성과는 고급 의료 이미지 분석을 위한 강력한 도구로서의 가능성을 보여줍니다.



### CRRG-CLIP: Automatic Generation of Chest Radiology Reports and Classification of Chest Radiographs (https://arxiv.org/abs/2501.01989)
- **What's New**: CRRG-CLIP 모델은 폐 영상 보고서 생성 및 영상 분류를 위한 새로운 엔드 투 엔드 모델입니다. 이 모델은 두 개의 모듈로 구성되며, 하나는 폐 영상 보고서를 자동으로 생성하고, 다른 하나는 영상 분류를 수행합니다. 본 연구는 최근의 심층 학습 기술을 활용하여 영상 보고서 작성의 복잡성과 비효율성을 해결하고자 합니다.

- **Technical Details**: 모델 구조는 Faster R-CNN을 사용하여 폐 영상에서 해부학적 영역을 식별하고, 이러한 키 영역으로부터 GPT-2를 사용하여 의미적으로 일관된 보고서를 생성합니다. 특히 CLIP 모델을 활용하여 레이블이 없는 데이터에서 최적의 특징을 추출하고, 비용이 높은 레이블이 있는 데이터셋의 필요성을 줄입니다. 이러한 접근 방식은 지역적으로 중요한 특징을 효과적으로 반영합니다.

- **Performance Highlights**: 실험 결과, 생성 모듈은 BLEU, METEOR, ROUGE-L과 같은 지표에서 높은 성능을 나타내며, 특히 GPT-4o 모델을 BLEU-2, BLEU-3, BLEU-4 및 ROUGE-L에서 초과하여 성능 향상을 보였습니다. 분류 모듈은 AUC 및 정확도 측면에서 최첨단 모델을 크게 초과하여 보고서 생성 및 영상 분류 모두에서 높은 정확도와 유창성을 달성합니다.



### Gender Bias in Text-to-Video Generation Models: A case study of Sora (https://arxiv.org/abs/2501.01987)
Comments:
          7 pages, 3 figures

- **What's New**: 본 연구는 OpenAI의 Sora 모델을 통한 텍스트-비디오 생성 모델에서 성별 편향(gender bias)의 존재를 조사합니다. 텍스트 프롬프트에서 생성된 비디오의 분석을 통해 기존의 편향성 문제가 어떠한 형태로 나타나는지 밝히고자 하였습니다.

- **Technical Details**: Sora 모델은 다양한 성 중립(gender-neutral) 및 고정 관념(stereotypical) 프롬프트를 사용하여 생성된 비디오를 분석하였습니다. 이 과정에서 특정 성별이 비전(career)과 행동(behavior)에 대해 고정 관념적으로 연결되는 방식에 대한 정량적 데이터가 수집되었습니다.

- **Performance Highlights**: 결과적으로 Sora 모델은 특정 성별을 특정 직업이나 행동에 불균형적으로 연결짓는 경향을 보였습니다. 이는 훈련 데이터에 포함된 사회적 편견(social prejudices)이 반영된 것으로 해석됩니다.



### FrameFusion: Combining Similarity and Importance for Video Token Reduction on Large Visual Language Models (https://arxiv.org/abs/2501.01986)
- **What's New**: 최근 비디오 이해에 대한 수요가 증가하면서, 대형 비전-언어 모델(LVLMs)의 시각적 토큰 처리에 큰 부담이 되고 있습니다. 기존의 토큰 축소 방법들은 주로 중요도 기반의 토큰 가지치기를 중시하며, 프레임 유사성 및 반복적인 시각적 요소로 인한 중복성을 간과했습니다. 본 논문에서는 유사성을 중요도와는 별개의 특성으로 분석하고, FrameFusion이라는 새로운 접근 방식을 제안하여 유사성 기반 통합과 중요도 기반 가지치기를 결합해 LVLM의 토큰 축소를 개선하고자 합니다.

- **Technical Details**: FrameFusion은 시각적 토큰에서 유사성과 중요도를 동시에 활용하여 효율적인 LVLM 토큰 축소를 이루어냅니다. 이 방법은 먼저 유사한 토큰을 병합한 다음, 중요도 기반의 가지치기를 통해 주어진 계산 예산에 맞추어 토큰을 줄입니다. 실험을 통해 FrameFusion이 Llava-Video 및 MiniCPM-V 등 다양한 LVLM에서 비디오 이해, 질문-응답, 검색 벤치마크에서 유효성을 입증하며, 토큰을 70% 줄이고 LLM 속도를 3.4-4.4배, 엔드투엔드 속도를 1.6-1.9배 증가시키는 성능을 보였습니다.

- **Performance Highlights**: FrameFusion은 다양한 비디오 작업에서 효과적으로 토큰을 압축하며, 기존의 밀집 모델과 비교해 비슷한 성능을 유지하면서도 계산 복잡도를 30%까지 줄일 수 있음을 보여주었습니다. 또한 이 방법은 다양한 모델 크기, 입력 길이 및 비디오 작업에 걸쳐 일반화됩니다. 이를 통해 LVLMs의 효율성을 크게 향상시키는 잠재력을 지니고 있습니다.



### Fall Detection in Passenger Elevators using Intelligent Surveillance Camera Systems: An Application with YoloV8 Nano Mod (https://arxiv.org/abs/2501.01985)
Comments:
          8 pages, 3 figures

- **What's New**: 이번 연구는 깊은 학습 알고리즘을 활용한 컴퓨터 비전 기술이 승강기 내의 낙상 사건 탐지에 어떻게 응용되는지를 중점적으로 다룹니다. YoloV8 Nano 모델을 이용하여 폐쇄된 환경과 변화하는 조명 조건에서의 낙상 인식을 목표로 합니다. 이는 기존의 낙상 탐지 시스템에 혁신적인 접근 방식을 제공할 것입니다.

- **Technical Details**: 연구는 10,000장 이상의 다양한 승강기 이미지로 구성된 강력한 데이터셋에서 모델을 훈련하는 과정을 포함합니다. 이로 인해 모델의 탐지 정밀도(precision)와 재현율(recall) 가 각각 85%와 82%에 달했습니다. 이러한 결과는 딥러닝을 통한 효과적인 이미지 분석의 가능성을 보여줍니다.

- **Performance Highlights**: YoloV8 Nano 모델은 낙상 사건 탐지에서 특히 높은 정밀도와 재현율을 기록하였습니다. 이러한 성과는 승강기 안전 시스템에 통합되어 빠른 개입을 가능하게 함으로써, 향후 안전성을 크게 향상시킬 잠재력을 가지고 있습니다.



### Leveraging AI for Automatic Classification of PCOS Using Ultrasound Imaging (https://arxiv.org/abs/2501.01984)
Comments:
          Code available at: this https URL

- **What's New**: AUTO-PCOS Classification Challenge는 인공지능(AI)의 진단 능력을 향상시키기 위해 건강한 초음파 이미지와 질병이 있는 이미지를 자동으로 분류하는 방법론을 제시하고 있습니다. 주요 목표는 Polycystic Ovary Syndrome(PCOS)의 조기 진단을 지원하여 의사들이 보다 정확한 진단을 가능하게 하는 것입니다. 이 프로젝트는 특히 AI가 의료 분야에서 어떻게 혁신적인 변화를 가져올 수 있는지를 보여주는 중요한 사례로 평가받고 있습니다.

- **Technical Details**: 연구팀은 InceptionV3 아키텍처를 사용하여 전이 학습(transfer learning)을 통해 강력한 AI 파이프라인을 구축하였습니다. 데이터 전처리는 이미지 사이즈, 품질, 색상 분포의 변화를 다루어 모델 성능을 향상시키는 데 기여했습니다. 또한, 데이터셋은 4,668개의 초음파 이미지로 구성되며, 각 이미지는 의료 전문가에 의해 주석이 달린 신뢰할 수 있는 자료입니다.

- **Performance Highlights**: 모델의 최종 정확도는 90.52%로, 효율적인 binary classification을 통해 검증 데이터를 분석한 결과, precision, recall, F1-score 모두 90%를 초과했습니다. 이는 모델이 PCOS 진단을 위한 신뢰할 수 있는 도구임을 시사합니다. 이 연구는 AI가 의료 진단 도구를 보다 신뢰성 있고 해석 가능한 형태로 발전시킬 수 있는 가능성을 강조합니다.



### ECG-guided individual identification via PPG (https://arxiv.org/abs/2501.01983)
Comments:
          Accepted by ICASSP 2025. Camera Ready Version

- **What's New**: 이 논문에서는 심전도 신호(ECG)를 페로소그래프(PPG) 신호에 지식을 전이하기 위한 새로운 교차 모달 지식 증류(cross-modal knowledge distillation) 프레임워크를 제안합니다. 이 프레임워크는 PPG 신호가 추론 단계에서만 사용되도록 하여 계산 비용을 줄입니다. 또한, CLIP 기반의 지식 정렬 모듈을 도입해 정보 전파의 효율성을 증가시킵니다.

- **Technical Details**: 제안된 방법론에서는 ECG 모델이 PPG 모델의 교사 역할을 수행하며, 학습 과정에서 두 모달리티 간의 도메인 차이를 줄이기 위해 지식 정렬 모듈이 사용됩니다. 이 모듈은 훈련 과정에서만 필요하며, 추론 단계에서는 추가적인 계산 요구가 없습니다. 또한, 두 개의 인코더와 프로젝션 헤드를 포함하는 구조가 설계되어 있습니다.

- **Performance Highlights**: 종합 실험을 통해 제안된 프레임워크가 기존의 기준 모델에 비해 2.8% 및 3.0%의 향상을 보이며, 이를 통해 각기 다른 데이터베이스에서 개인 인식의 정확도를 크게 증가시켰음을 입증하였습니다. 이러한 결과는 PPG 기반 생체 인식 기술의 신뢰성과 효율성을 높이는 데 기여할 것입니다.



### Is Your Image a Good Storyteller? (https://arxiv.org/abs/2501.01982)
Comments:
          Accepted by AAAI 2025

- **What's New**: 본 논문에서는 이미지의 의미적 복잡성을 평가하는 Image Semantic Assessment (ISA) 작업을 제안합니다. ISA 데이터셋과 언어 기반 방법을 활용하여 복잡성 평가를 자동화하고, 다양한 문화적 배경을 가진 사람들이 이해할 수 있는 이미지를 생성하는 필요성을 강조합니다. 이 연구는 이미지의 의미적 가치를 기반으로 한 새로운 평가 기준을 설정하는 데 기여합니다.

- **Technical Details**: ISA 데이터셋은 Pinterest에서 수집된 2,946개의 이미지를 포함하고, 각 이미지는 Entity Score와 Semantic Score를 통해 평가됩니다. Entity Score는 이미지 내 요소의 풍부함을 측정하고, Semantic Score는 더 높은 수준의 의미적 복잡성을 평가합니다. 논문에서는 Vision-Language 협력 ISA 방법(VLISA)을 제안하여, 대형 비전-언어 모델(LVLM)을 사용해 이미지의 의미적 정보를 추출합니다.

- **Performance Highlights**: 실험 결과, ISA 작업은 전통적인 비전 모델들에 도전적이며, 제안한 방법은 Semantic Complexity Scoring 작업에서 다른 기준 모델들에 비해 우수한 성능을 보였습니다. 이 연구는 AI 모델의 교육 및 평가를 위한 복잡한 이미지를 선별하는 데 중요한 단초를 제공합니다. 따라서 더 많은 연구자들이 이미지의 의미적 복잡성에 집중할 수 있는 기반을 마련했습니다.



### Hawkes based Representation Learning for Reasoning over Scale-free Community-structured Temporal Knowledge Graphs (https://arxiv.org/abs/2501.01974)
- **What's New**: 본 논문에서는 실제 네트워크의 특성을 반영한 새로운 Temporal Knowledge Graph (TKG) 추론 모델인 Hawkes 프로세스 기반의 Evolutional Representation Learning Network (HERLN)를 제안합니다. HERLN은 커뮤니티 구조, 스케일-프리(Scale-free) 및 시간적 감쇠(Temporal decaying)를 동시에 고려하여 TKG의 구조적 정보와 진화 패턴을 학습합니다. 이 접근 방식은 TKG 추론의 성능을 크게 향상시키는 데 기여합니다.

- **Technical Details**: HERLN 모델은 세 가지 모듈로 구성됩니다: (1) 임베딩 초기화 모듈, 커뮤니티 구조의 의미 정보를 추출하며; (2) 진화 인코딩 모듈, 이벤트의 시간적 감쇠 현상을 다루고; (3) 조건부 디코딩 모듈, 긴 꼬리 분포(長獠分布)로 인한 선호 편향을 완화합니다. 이를 통해 HERLN은 시간 차원에서의 이벤트 영향을 다루고, 조건부 인스턴스 점수를 생성하여 후보 개체에 대한 예측을 더욱 정교하게 합니다.

- **Performance Highlights**: HERLN은 네 가지 벤치마크 TKG 데이터셋에서 실험을 통해 최첨단 모델에 비해 상당한 성능 향상을 보여주었습니다. 특히, 본 모델은 동시적으로 개체(entity)와 관계(relation) 예측을 수행할 수 있는 통합 모델로 기능합니다. 결과적으로 HERLN은 TKG 추론의 새로운 기준을 설정할 것으로 기대됩니다.



### INFELM: In-depth Fairness Evaluation of Large Text-To-Image Models (https://arxiv.org/abs/2501.01973)
Comments:
          Di Jin and Xing Liu contributed equally to this work

- **What's New**: 이번 논문에서는 다중 모달 AI 시스템의 공정성을 평가하기 위한 INFELM이라는 새로운 프레임워크를 제시하고 있습니다. 이 프레임워크는 기존의 텍스트-이미지 생성 모델의 사회적 영향 및 불공정성을 측정하기 위해 발전된 스킨톤 분류기와 편향에 민감한 콘텐츠 정렬 측정이 포함되어 있습니다. 이러한 평가 도구는 산업 애플리케이션에 적합하도록 설계되었으며, 공정성 기준을 충족하지 않는 기존 모델들의 한계를 강조합니다.

- **Technical Details**: INFELM은 네 가지 주요 기여를 통해 다중 모달 AI 시스템의 공정성 평가를 수행합니다. 첫째, 얼굴 형태학 및 스킨 픽셀 표현을 개선하여 16.04% 이상의 분류 정밀도를 높인 고급 스킨톤 분류기를 도입하였습니다. 둘째, 다양한 사회적 집단별 정의된 편향 영향 측정을 통해 콘텐츠 정렬 오류를 분석하고, 셋째, 다양한 인구 통계적 그룹에 대한 일반화 가능한 대표성 편향 평가를 제안합니다.

- **Performance Highlights**: 논문에서 실시한 대규모 분석 결과, 연구 대상 모델들이 공정성 기준을 일반적으로 충족하지 않는 것으로 나타났습니다. 특히 스킨톤 정렬 오류는 성별 오류보다 두드러지며, 편향 검사에서 기존 모델의 공정성 문제가 명확히 드러났습니다. INFELM은 다중 모달 AI 시스템의 윤리적 및 인간 중심 원칙에 부합하는 공정성 평가를 위한 강력한 기준을 세웠습니다.



### Optimal bounds for dissatisfaction in perpetual voting (https://arxiv.org/abs/2501.01969)
Comments:
          Full version of the AAAI 2025 paper

- **What's New**: 본 논문은 영구 투표(perpetual voting)라는 개념을 다루며, 투표 방법이 각 유권자가 불만족하는 횟수를 최소화할 수 있는지를 검토합니다. 이 연구는 'bounded conflicts'라는 새로운 조건을 도입해 유권자가 불만족하는 감정의 성장 속도를 조절할 수 있음을 보입니다. Kolmogorov complexity 기술을 활용하여 불만족 성장의 최상한을 정의하였습니다.

- **Technical Details**: 제안된 투표 방법은 두 플레이어, 즉 결정자(Decision Maker)와 적대자(Adversary) 간의 게임으로 설정됩니다. 이 게임에서 각 라운드마다 적대자는 유권자들이 승인한 여러 집합을 선택하고, 결정자는 이러한 집합 중 하나의 옵션을 선택합니다. 연구는 갈등 수(conflict number)를 도입하여, 이 수치가 결정 수에 비례해 제한될 경우, 불만족의 성장 속도를 서브선형(sublinear)으로 유지할 수 있는 전략을 개발합니다.

- **Performance Highlights**: 기존의 투표 방법들은 갈등이 많은 상황에서 불만족을 줄이는 데 실패했습니다. 그러나 본 논문에서 제안한 새로운 규칙은 제로에 가까운 범위 내에서 불만족을 최소화할 수 있음을 보여줍니다. 이 연구는 perpetual voting이 정보를 어떻게 처리하는지에 대한 기계 학습 분야의 이론과도 연결될 수 있음을 시사합니다.



### Statistical learning does not always entail knowledg (https://arxiv.org/abs/2501.01963)
Comments:
          30 pages, 1 figure

- **What's New**: 이 논문에서는 진리 또는 거짓인 명제에 대한 에이전트의 학습 및 지식 습득(LKA)에 대해 연구합니다. 베이esian 접근 방식을 사용하여 에이전트가 데이터를 수신하고, 이 데이터를 통해 명제에 대한 신념을 posterior distribution에 따라 업데이트합니다. 또한, 학습이 필요 이상으로 복잡하며, 통계적 학습이 항상 지식 습득을 보장하지 않는다는 점을 강조합니다.

- **Technical Details**: LKA는 외부 또는 외생적인 정보가 에이전트의 신념을 수정하는 방식을 바탕으로 활발한 정보(active information)로 공식화됩니다. 데이터는 명제와 관련된 여러 특성에 대한 세부 정보를 제공합니다. 우리는 Gibbs distribution posterior를 도출하여 데이터가 제공하는 측면 제약 조건을 토대로 우선 분포에 대한 최대 엔트로피를 보여줍니다.

- **Performance Highlights**: 논문은 완전한 학습이 항상 가능하지 않으며, 추출된 특성의 수가 너무 적으면 완전한 지식 습득이 불가능하다는 점을 밝힙니다. 기본 학습(명제 관련 특성에 대한 데이터 수신)과 이차 학습(다른 에이전트의 학습에 대한 데이터 수신)을 구분하고, 이차 학습이 진정한 지식 습득을 나타내지 않는다고 주장합니다. 이러한 발견은 통계적 학습 알고리즘에 대한 중요한 함의를 지니고 있습니다.



### GAF-FusionNet: Multimodal ECG Analysis via Gramian Angular Fields and Split Attention (https://arxiv.org/abs/2501.01960)
Comments:
          14 pages, 1 figure, accepted by ICONIP 2024

- **What's New**: 이 논문에서는 ECG 신호(classification)의 정확한 분석을 위한 혁신적인 멀티모달 프레임워크 GAF-FusionNet(Gramian Angular Fields-Fusion Network)을 소개합니다. 이 접근법은 시간적 분석을 이미지 기반 표현과 통합하여 ECG 신호의 복잡한 신호를 보다 잘 분류할 수 있게 합니다. 또한, 이 프레임워크는 시간적 및 공간적 특징을 동적으로 융합하기 위해 듀얼 레이어 교차 채널 스플릿 어텐션 모듈을 사용하여 보완 정보를 미세하게 통합합니다.

- **Technical Details**: GAF-FusionNet은 ECG 신호의 시계열 분석과 이미지 처리 간의 간극을 메우기 위해 Gramian Angular Fields (GAF) 기법을 활용하여 원-디멘셔널(1D) 신호를 투-디멘셔널(2D) 이미지로 변환합니다. 이 변환은 주파수 분석뿐만 아니라 강력한 컴퓨터 비전 기법의 적용을 가능하게 하여 새로운 특징 추출 및 패턴 인식의 가능성을 열어줍니다. 논문은 ECG200, ECG5000 및 MIT-BIH 부정맥 데이터베이스에서의 성능을 평가하여 새로운 기준을 세웁니다.

- **Performance Highlights**: GAF-FusionNet은 세 가지 다른 ECG 데이터셋에서 각각 94.5%, 96.9%, 99.6%의 높은 정확도를 기록하였으며, 기존 최첨단 방법들에 비해 상당한 성능 개선을 보여주었습니다. 이러한 결과는 GAF-FusionNet이 ECG 신호 분석의 새로운 기준을 설정함을 나타냅니다. 이 모델의 프레임워크와 코드 또한 곧 공개될 예정입니다.



### STEAM-EEG: Spatiotemporal EEG Analysis with Markov Transfer Fields and Attentive CNNs (https://arxiv.org/abs/2501.01959)
Comments:
          10 pages, 5 figures

- **What's New**: 본 논문에서는 EEG 신호의 분석을 향상시키기 위한 새로운 통합 프레임워크 STEAM-EEG를 제안합니다. 이 접근법은 Singular Spectrum Analysis (SSA), 수정된 attention 메커니즘을 가진 병렬 1D-Convolutional Neural Networks (CNNs), 그리고 Markov Transfer Fields (MTFs)를 결합하여 EEG 신호의 복잡한 시공간 역학을 효과적으로 포착합니다. 이 방식은 EEG 신호의 패턴을 시각적으로 표현하여 데이터 탐색과 분석의 해석성을 높이는 데 기여합니다. 전반적으로 이 연구는 EEG 분석의 정확성과 강건성을 개선하는 데 중점을 두고 있습니다.

- **Technical Details**: 본 시스템의 핵심은 먼저 SSA를 이용하여 원시 EEG 시간 시리즈를 추세(trend), 계절(seasonal) 및 노이즈(noise) 구성 요소로 분해하는 것입니다. SSA 알고리즘은 데이터의 시간 구조를 포착하고, 그 후 병렬 1D-CNNs를 통해 파악된 구성 요소의 특성에 대한 학습을 진행합니다. 또한, 수정된 attention 메커니즘을 통해 여러 EEG 채널 간의 상호 의존성을 더욱 효과적으로 처리하고, 시각적 표현을 위해 MTF 이미징을 사용합니다.

- **Performance Highlights**: 제안한 STEAM-EEG 프레임워크는 다양한 EEG 데이터셋에 대해 기존의 최첨단 방법과 비교하여 뛰어난 정확도를 보여줍니다. 레이더 차트를 통해 평가된 데이터셋 전반에 걸쳐 이 방법의 우수성이 입증되었습니다. 이는 EEG 신호의 복잡한 시공간 역학을 효과적으로 캡처하고 분석할 수 있는 통합적인 접근 방식을 제공함을 의미합니다.



### Implications of Artificial Intelligence on Health Data Privacy and Confidentiality (https://arxiv.org/abs/2501.01639)
- **What's New**: 이 논문은 인공지능(AI)이 의료 분야에서 일으키는 혁신적인 변화를 다룹니다. 특히 의료 진단, 맞춤형 의학, 운영 효율성 측면에서 AI의 통합이 빠르게 진행되고 있다는 점을 강조합니다. 그러나 이러한 발전과 함께 환자 데이터 프라이버시, 윤리적 고려사항, 규제 준수와 같은 중대한 도전 과제가 발생하고 있음을 설명합니다. 이러한 이중적 영향력은 AI의 긍정적인 가능성과 함께 민감한 건강 정보를 보호해야 할 필요성을 부각합니다.

- **Technical Details**: 이 연구는 AI의 적용이 이루어지는 사례 연구를 통해 의료 분야에서의 윤리적 및 법적 복잡성을 분석합니다. 건강 보험 이동 및 책임 법(HIPAA)의 역할을 언급하며, 데이터 프라이버시와 보안을 보장하기 위한 규제 프레임워크의 중요성을 강조합니다. 특히 당뇨병 망막병증, 종양학 분야의 AI 응용 프로그램과 같은 사례를 통해 AI 도입으로 인한 여러 문제를 조명합니다.

- **Performance Highlights**: 이 논문은 AI의 잠재력을 책임감 있게 활용하기 위해 지속적인 교육, 투명성, 규제 프레임워크 준수의 중요성을 강조합니다. 혁신을 촉진하면서 환자의 신뢰와 프라이버시를 유지하는 균형 잡힌 접근의 필요성을 제기합니다. AI가 의료 분야에서 제대로 활용되기 위해서는 윤리적 기준과 강력한 안전 장치가 필요하다는 점을 부각합니다.




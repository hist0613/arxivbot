New uploads on arXiv(cs.CL)

### E2LLM: Encoder Elongated Large Language Models for Long-Context Understanding and Reasoning (https://arxiv.org/abs/2409.06679)
Comments:
          12 pages, 4 figures

- **What's New**: 이 논문에서는 E2LLM(Encoder Elongated Large Language Models)이라는 새로운 접근 방식을 소개합니다. 이 방법은 긴 컨텍스트의 성능을 향상시키고 계산 복잡성을 줄이며, 사전 학습된 모델을 효과적으로 활용합니다.

- **Technical Details**: E2LLM은 긴 컨텍스트를 여러 개의 청크로 나누고, 각 청크를 사전 학습된 텍스트 인코더(BERT 등)를 통해 임베딩 벡터로 압축합니다. 그런 다음 어댑터를 사용하여 인코더의 출력을 디코더 전용 LLM의 입력 임베딩 공간과 정렬합니다. 두 가지 훈련 목표가 설정되며, 이는 인코더가 인코딩한 입력 텍스트의 재구성과 긴 컨텍스트 지침 미세 조정을 포함합니다.

- **Performance Highlights**: 실험 결과 E2LLM은 긴 컨텍스트 시나리오에서 우수한 성능을 보이며, 효율성, 성능 및 사전 학습된 모델과의 호환성을 균형 있게 유지함을 입증합니다.



### LLaMA-Omni: Seamless Speech Interaction with Large Language Models (https://arxiv.org/abs/2409.06666)
Comments:
          Preprint. Project: this https URL

- **What's New**: 이 논문에서는 LLaMA-Omni라는 새로운 모델 아키텍처를 제안하여, 오픈소스 대규모 언어 모델(LLM)과의 저지연(large latency) 및 고품질의 음성 상호작용(speech interaction)을 가능하게 합니다. 이는 전통적인 텍스트 기반 상호작용보다 월등한 사용자 경험을 제공합니다.

- **Technical Details**: LLaMA-Omni는 사전 훈련된 음성 인코더(speech encoder), 음성 어댑터(speech adaptor), LLM, 스트리밍 음성 디코더(streaming speech decoder)로 구성됩니다. 이 모델은 음성 트랜스크립션(transcription) 없이 사용자의 음성 지시를 직접 처리하고, 텍스트와 음성 응답을 동시에 생성할 수 있는 기능을 제공합니다.

- **Performance Highlights**: 실험 결과, LLaMA-Omni는 콘텐츠와 스타일 모두에서 이전 음성 언어 모델에 비해 더 나은 응답을 제공하며, 응답 지연(latency)은 226ms로 매우 낮습니다. 또한 LLaMA-Omni의 훈련은 4개의 GPU에서 3일 이내에 완료되어, 향후 효율적인 음성 언어 모델 개발이 가능할 것으로 기대됩니다.



### TeXBLEU: Automatic Metric for Evaluate LaTeX Forma (https://arxiv.org/abs/2409.06639)
Comments:
          5 pages, 3 figures

- **What's New**: 이 연구는 LaTeX 형식의 수학 표현을 평가하기 위한 새로운 메트릭인 TeXBLEU를 제안합니다. 기존 평가 메트릭이 LaTeX 문서에서의 수학 표현을 제대로 평가하지 못하는 문제를 해결하고, n-gram 기반의 BLEU 메트릭을 기반으로 합니다.

- **Technical Details**: TeXBLEU는 arXiv 논문의 데이터 세트에 맞춰 훈련된 새로운 tokenizer와 사전 훈련된 GPT-2 기반 임베딩 모델을 포함합니다. 이 메트릭은 토큰의 위치 인코딩을 고려하며, n-gram을 기반으로 한 유사성을 측정하여 LaTeX 수학 표현의 구조적 유사성을 파악합니다.

- **Performance Highlights**: TeXBLEU는 MathBridge 데이터 세트의 1,000개 데이터 포인트에 대해 기존 메트릭인 BLEU, Rouge, CER, WER와 비교했을 때 인간 평가 데이터와의 평균 상관 계수 0.71을 기록하면서, BLEU보다 87% 향상된 성능을 나타냈습니다.



### A Practice of Post-Training on Llama-3 70B with Optimal Selection of Additional Language Mixture Ratio (https://arxiv.org/abs/2409.06624)
Comments:
          11 pages, 4 figures

- **What's New**: 이 논문에서는 Llama-3 8B 및 70B 모델에 대한 지속적 사전 훈련(Continual Pre-Training, CPT)을 수행하여 모델의 중국어 능력을 향상시키는 방법을 제안합니다. 특히, Additional Language Mixture Ratio (ALMR)와 Learning Rate (LR) 간의 최적 상관관계를 규명하며, 이로 인해 모델의 성능을 극대화하는 실험 설정을 안내합니다.

- **Technical Details**: 논문에서 제안하는 ALMR과 LR 간의 최적 상관관계를 기반으로 하여, Llama-3 8B 모델은 1.7T 훈련 토큰을 사용하여 사전 훈련을 진행합니다. 실험 결과, ALMR 설정 및 LR 조정이 중요함을 보여주며, 전체 실험 과정에서 효과적인 하이퍼파라미터 조정이 이루어집니다. 또한, 70B 모델에 대해서도 동일한 ALMR과 낮은 LR을 적용하고, 최종적으로 Supervised Fine-tuning (SFT)과 Direct Preference Optimization (DPO)를 통해 모델을 정밀 조정합니다.

- **Performance Highlights**: 중국어 관련 벤치마크뿐만 아니라 수학, 코딩, 감정 지능과 같은 특정 도메인에서도 모델 성능이 향상되었습니다. 최종적으로 배포된 70B 모델은 산업용 채팅 시스템에서 만족스러운 성능을 입증하며, 감정 지능을 개선한 챗봇을 통해 인간과의 상호작용에서도 뛰어난 성과를 보입니다.



### Exploring Italian sentence embeddings properties through multi-tasking (https://arxiv.org/abs/2409.06622)
Comments:
          9 pages, 9 figures, 3 tables

- **What's New**: 이 연구는 기존의 큰 언어 모델(LLM)이 이탈리아어에서 추상적인 언어 정보를 어느 정도까지 인코딩하는지를 다룬다. 여러 개의 Blackbird Language Matrices(BLMs) 문제에 기반한 큐레이션된 합성 데이터를 대규모로 활용하여, 사전 훈련된 언어 모델로 구축된 문장 표현이 특정한 구문 및 의미 정보를 어떻게 인코딩하는지를 연구한다.

- **Technical Details**: 이 논문은 문장 임베딩을 작업에 관련된 정보로 압축하는 단계를 두 개의 레벨로 모델링하는 구조를 채택하였다. 각 작업에 대해 문장 임베딩에 인코딩된 단서가 다르게 나타남을 보여, 사전 훈련된 문장 임베딩에는 구성 요소(constituents) 또는 주제 역할(theme roles)과 같은 추상적 언어 개념이 포함되어 있지 않다는 것을 암시한다.

- **Performance Highlights**: 다양한 작업에서 성능과 오류 분석을 통해, 문장 구조와 청크 속성이 여러 작업 간에 공유될 수 있을 것으로 예상되었으나, 결과적으로 각 작업의 단서가 문장 임베딩에 다른 방식으로 인코딩되었음을 발견했다. 이는 LLM이 깊은 언어적 이해를 가지고 있지 않음을 시사하며, 오히려 유용한 표면적 단서(surface indicators)에 의존한다.



### Alleviating Hallucinations in Large Language Models with Scepticism Modeling (https://arxiv.org/abs/2409.06601)
Comments:
          11 pages, 6 figures

- **What's New**: 본 논문에서는 큰 언어 모델(LLMs)의 할루시네이션 문제를 해결하기 위해 회의적 사고를 모델링하는 새로운 접근법인 Skepticism Modeling(SM)을 제안하였습니다. SM은 토큰(token)과 로짓(logit) 정보를 결합하여 자기 추정(self estimation) 능력을 향상시키는 방식으로 구성됩니다.

- **Technical Details**: Skepticism Modeling(SM) 접근법은 고유 텍스트 토큰 뒤에 회의적 감정을 나타내는 토큰을 추가하여 모델이 더 나은 자기 평가를 할 수 있도록 트레이닝을 진행합니다. 이 과정에는 지속적 사전 훈련(Continual Pre-Training, CPT)과 감독된 미세 조정(Supervised Fine-Tuning, SFT) 단계를 포함합니다. 실험에서는 모델이 제시된 질문-답변 샘플에 대한 회의 성과를 평가하고, 새로운 질문-답변 쌍을 추가하여 모델의 훈련 경험을 바탕으로 자기 측정을 수행하는 방식입니다.

- **Performance Highlights**: 실험 결과, SM 접근법은 여러 QA 벤치마크에서 최첨단 성능(STATE-OF-THE-ART, SOTA)을 달성하며, 도메인 외 일반화(out-of-domain generalization) 능력을 입증하였습니다. 또한, 불합리하거나 비현실적인 질문에 대해서도 상당한 강인성을 발휘하는 모습을 보였습니다.



### GroUSE: A Benchmark to Evaluate Evaluators in Grounded Question Answering (https://arxiv.org/abs/2409.06595)
- **What's New**: 이 연구는 Retrieval-Augmented Generation (RAG) 시스템의 성능 및 평가 방법에 대한 새로운 통찰을 제공합니다. 특히, RAG 평가 시스템에서 LLM을 평가자로 사용했을 때의 문제점을 다루며, GroUSE라는 메타 평가 벤치마크를 도입했습니다.

- **Technical Details**: 연구에서는 7가지 생성기 실패 모드를 식별하고, 평가자 모델의 보정(calibration)과 구별(discrimination) 능력을 평가하기 위해 144개의 단위 테스트로 구성된 GroUSE를 제안합니다. 연구 결과, 기존의 RAG 평가 프레임워크가 중요한 실패 모드를 간과하는 경우가 많음을 확인했습니다.

- **Performance Highlights**: GroUSE 테스트에서 폐쇄형 모델(closed models)은 높은 성능을 보였으나, 최첨단 오픈소스 평가자 모델은 제안된 기준에 대해 일반화하지 못했습니다. Llama-3 모델을 GPT-4의 평가 흔적(trace)으로 미세 조정(finetuning)하여 평가 능력을 크게 향상시키는 성과를 보였습니다.



### Exploring syntactic information in sentence embeddings through multilingual subject-verb agreemen (https://arxiv.org/abs/2409.06567)
Comments:
          11 pages, 5 tables, 5 figures

- **What's New**: 이번 연구에서는 다국어 프리트레인(Pretrained) 언어 모델이 언어 간에 유효한 추상적 언어 표현을 얼마나 잘 포착하는지를 조사합니다. 이를 위해 특정 속성을 가진 대규모의 큐레이션된 합성 데이터(Synthetic Data)를 개발하고, 프리트레인된 언어 모델로 구축한 문장 표현을 연구합니다.

- **Technical Details**: 우리는 Blackbird Language Matrices (BLMs)라는 새로운 다중 선택 과제를 사용하여 다양한 문장 구조에서의 주어-동사 일치를 초점으로 여러 언어에서 분석을 진행합니다. 두 단계 아키텍처를 사용하여 개별 문장의 구문적 객체와 그 특성을 감지하고, 문장 입력 시퀀스 전반에 걸쳐 패턴을 찾습니다.

- **Performance Highlights**: 우리의 실험 결과, 세 언어(영어, 프랑스어, 이탈리아어, 그리고 루마니아어) 간의 전이는 부족하게 나타났습니다. 이는 프리트레인 모델이 구문 정보를 얕고 언어 특정적인 단서에 기반하여 인코딩함을 보여주며, 언어 모델이 문법적 구조를 추상화하는 단계로 나아가지 못했음을 시사합니다.



### From LIMA to DeepLIMA: following a new path of interoperability (https://arxiv.org/abs/2409.06550)
Comments:
          16 pages, 5 figures, submitted to Language Resources and Evaluation

- **What's New**: 최근 LIMA (Libre Multilingual Analyzer) 프레임워크의 아키텍처가 새로운 텍스트 분석 모듈을 추가하여 진화했습니다. 이 모듈들은 딥 뉴럴 네트워크 (Deep Neural Networks) 기반으로 개발되었습니다.

- **Technical Details**: LIMA의 기능은 지원하는 언어의 수를 확장하면서 기존의 구성 가능한 아키텍처와 기존에 개발된 규칙 기반 및 통계 분석 구성 요소의 가용성을 유지했습니다. Universal Dependencies 2.5 코퍼스, WikiNer 코퍼스 및 CoNLL-03 데이터셋을 바탕으로 60개 이상의 언어에 대해 모델이 훈련되었습니다.

- **Performance Highlights**: Universal Dependencies를 통해 지원 언어의 수를 늘리고 다른 플랫폼과 통합될 수 있는 모델 생성을 가능하게 했습니다. 이러한 딥 러닝 자연어 처리 모델의 통합은 모델과 데이터를 정규화하여 기술 상호 운용성을 향상시키는 새로운 경로로 볼 수 있습니다.



### Mapping News Narratives Using LLMs and Narrative-Structured Text Embeddings (https://arxiv.org/abs/2409.06540)
Comments:
          19 pages, 13 figures, 4 tables

- **What's New**: 본 논문에서는 서사(narrative)의 양적 분석을 위한 새로운 방법론을 제시하며, Greimas의 행동자 모델(Actantial Model)을 기반으로 한 수치적 서사 표현(numerical narrative representation)을 도입합니다. 이 모델은 서사를 다루는 데 있어 장르에 구애받지 않아 높은 일반화 가능성을 지니고 있습니다.

- **Technical Details**: Greimas의 행동자 모델은 6개의 기능적 역할(주체(Subject), 객체(Object), 발신자(Sender), 수신자(Receiver), 도우미(Helper), 반대자(Opponent))로 구성됩니다. 텍스트에서 이 행동자를 추출하기 위해 오픈 소스 LLM을 사용하고, 이를 Narrative-Structured Text Embedding으로 통합해 텍스트의 의미(semantics)와 서사 구조(narrative structure)를 포착합니다.

- **Performance Highlights**: 이스라엘-팔레스타인 갈등에 관한 Al Jazeera와 The Washington Post의 5342개의 뉴스 기사를 분석하여 18개의 독특한 서사 트렌드를 식별했습니다. 이 연구는 서로 다른 서사 구조를 가진 기사들 간의 명확한 차이를 성공적으로 보여주었습니다.



### Questioning Internal Knowledge Structure of Large Language Models Through the Lens of the Olympic Games (https://arxiv.org/abs/2409.06518)
- **What's New**: 이 연구는 대형 언어 모델(LLMs)의 내부 지식 구조를 분석하며, 올림픽 메달 수치를 사용하여 이와 관련된 성능을 조사합니다.

- **Technical Details**: LLMs는 1964년부터 2022년까지의 올림픽 메달 데이터를 바탕으로 두 가지 과제에 대해 평가되었습니다: 1) 특정 팀의 메달 수 계산 및 2) 특정 순위를 달성한 팀의 식별. 실험에는 SOTA 모델과 오픈소스 모델이 포함되었으며, 모델의 성능을 비교하고 분석했습니다.

- **Performance Highlights**: SOTA LLM 모델들은 팀별 메달 수를 보고하는 데 뛰어난 성능을 보였으나, 특정 순위와 관련된 질문에서는 40% 이하의 정확도만을 기록했습니다. 이 연구는 또한 LLM이 사용자 의혹에 반응할 때 성능 저하를 보이며, 신뢰성을 유지하기 위한 추가 연구의 필요성을 강조합니다.



### An Effective Context-Balanced Adaptation Approach for Long-Tailed Speech Recognition (https://arxiv.org/abs/2409.06468)
Comments:
          Accepted by SLT 2024

- **What's New**: 이 논문은 E2E (End-to-end) 자동 음성 인식 (ASR) 모델에서의 rare words에 대한 인식 성능을 개선하기 위한 새로운 접근법을 제시합니다. 특히, 문맥 어댑터 (Contextual Adapter, CA)의 사용과 함께 문맥 리스트의 단어 빈도 분포를 조정하는 방법을 탐구합니다.

- **Technical Details**: 연구팀은 CA를 통해 외부 지식(문맥 단어 리스트)을 E2E ASR 모델에 주입하는 방법을 사용하였습니다. 또한, 문맥 리스트의 빈도 분포를 조정하여 모델 성능에 미치는 영향을 분석하고, 간단하지만 효과적인 문맥 균형 학습 목표를 통해 CA를 확장했습니다.

- **Performance Highlights**: AISHELL-1 벤치마크 데이터셋을 기반으로 한 일련의 실험에서는 훈련 코퍼스의 모든 어휘 단어를 문맥 리스트로 사용하고 균형 목표와 결합할 때 최상의 성능 성과를 나타냈습니다. 이 접근법은 문자 오류율 (Character Error Rate, CER)을 최대 1.21% 감소시키고, 제로샷 (zero-shot) 단어의 오류율을 9.44% 더 감소시키는 결과를 보여주었습니다.



### Coarse-Grained Sense Inventories Based on Semantic Matching between English Dictionaries (https://arxiv.org/abs/2409.06386)
Comments:
          The 11th International Conference on Advanced Informatics: Concepts, Theory and Applications (ICAICTA 2024)

- **What's New**: 이 논문에서는 WordNet의 세부적인 의미(meaning)로 인해 발생하는 사용성의 한계를 극복하기 위한 coarse-grained sense inventories(조잡한 의미 목록)를 개발하고, Cambridge 사전과의 의미 정의를 맞추는 과정을 제안합니다.

- **Technical Details**: 본 연구에서는 WordNet과 Cambridge 사전의 의미 정의를 비교하여, 새로운 coarse-grained sense inventories를 개발했습니다. 이 inventories는 선택된 사전을 바탕으로 의미 정의만을 사용하여 생성되며, CEFR 수준으로 할당됩니다.

- **Performance Highlights**: 제안된 inventories는 대규모 자원에 대한 의존성이 적고, 반자동으로 확장 및 개선할 수 있다는 장점이 있습니다. 또한, 관련된 의미를 잘 집계하여 교육적 활용 가능성을 높입니다.



### SpeechTaxi: On Multilingual Semantic Speech Classification (https://arxiv.org/abs/2409.06372)
- **What's New**: 이번 연구에서는 다국어 음성 분류(Semantic Speech Classification)를 위한 SpeechTaxi라는 80시간의 다국어 데이터셋을 새롭게 소개합니다. 이 데이터셋은 28개의 다양한 언어로 구성되어 있으며, 성경 구절을 기반으로 하고 있습니다.

- **Technical Details**: 연구에서는 (1) 최신 다국어 음성 인코더(Multilingual Speech Encoder, MSE)를 통한 엔드투엔드(End-to-End, E2E) 분류기와 (2) 텍스트 기반 분류기로 분류를 위임하는 연속 변환(Cascading, CA) 방식을 비교합니다. E2E 방식은 단일 언어 데이터에서 훈련할 경우 효과적이나, MSE는 다국어 훈련 및 제로샷 전이에서 CA에 비해 성능이 떨어지는 경향이 있습니다.

- **Performance Highlights**: MSE를 기반으로 한 E2E 방식이 단일 언어 설정에서는 CA 방식보다 뛰어난 성능을 보였으나, MSE는 제로샷 전이 및 다국어 훈련 성능에서 CA에 뒤처졌습니다. 최종적으로, 로마자 텍스트로의 전사를 기반으로 하는 새로운 CA 방식을 제안하였으며, 저자원 언어에 대한 강력한 해결책으로 제시됩니다.



### Retrieval Or Holistic Understanding? Dolce: Differentiate Our Long Context Evaluation Tasks (https://arxiv.org/abs/2409.06338)
- **What's New**: 이 논문은 LLM의 긴 컨텍스트(large context) 이해에 관한 새로운 프레임워크인 Dolce를 제안합니다. 이를 통해 retrieval(검색)과 holistic understanding(전체적 이해) 문제를 자동으로 식별하고 각 문제의 난이도를 정량적으로 측정할 수 있습니다.

- **Technical Details**: Dolce 프레임워크는 문제를 두 개의 파라미터 λ(복잡성)과 k(중복성)로 매개변수화하며, 이를 통해 생성된 λ-k 평면에서 각 문제를 다섯 가지 미리 정의된 초점 카테고리 중 하나에 할당합니다. 이 방법론은 비모수적 배경 노이즈 컴포넌트와 매개변수적/비매개변수적 하이브리드 오라클 컴포넌트를 결합한 혼합 모델을 사용하여 문제의 λ와 k를 찾는 데 도움을 줍니다.

- **Performance Highlights**: 실험 결과, 44개 기존 긴 컨텍스트 평가 작업에서 retrieval focused 문제는 0%에서 67% 사이, holistic understanding focused 문제는 0%에서 90% 사이로 식별되었습니다. 이러한 결과는 LLM의 긴 컨텍스트 능력을 이해하고 개발하는 데 기여하고 있습니다.



### Extracting Paragraphs from LLM Token Activations (https://arxiv.org/abs/2409.06328)
- **What's New**: 이번 연구는 Generative large language models (LLMs)의 내부 작동 방식을 탐색하며, 특히 텍스트 생성 시작 시점에서 그들이 얼마나 문맥적 이해를 가지고 있는지를 결정하는 요소들을 Investigation하기 위해 단일 토큰 activations를 분석합니다.

- **Technical Details**: 연구는 모델의 attention scores에 텍스트 구조가 어떻게 인코딩되는지를 보여주며, '

' 이중 개행 토큰을 통해 활성화가 전이되는 방식을 탐구합니다. 활성화 패치(patching) 기법을 사용하여 모델이 과거와 다음 문단간의 관련성을 어떻게 계획하는지를 분석합니다.

- **Performance Highlights**: 이 연구의 실험 결과는, 이중 개행 토큰의 활성화가 다음 문단에 대한 상당한 정보를 포함하고 있어, 모델이 텍스트 생성을 시작할 때 이후의 내용을 예측하는 능력을 가지고 있음을 보여줍니다. 전이된 활성화의 결과는 원본 문단과 높은 의미 유사성을 보이며, 모델이 다음 문단을 얼마나 잘 이해하고 있는지를 시사합니다.



### Keyword-Aware ASR Error Augmentation for Robust Dialogue State Tracking (https://arxiv.org/abs/2409.06263)
- **What's New**: 이 논문에서는 Dialogue State Tracking (DST) 모델의 견고성을 향상시키기 위한 새로운 데이터 증강 방법인 Error Positioning Augmentation (EPA)를 제안합니다. 이 방법은 ASR (Automatic Speech Recognition) 시스템의 오류를 고려해 키워드에 오류를 배치함으로써 DST 성능을 개선합니다.

- **Technical Details**: EPA는 대규모 언어 모델(LLMs)을 활용하여 ASR 오류가 발생하는 키워드와 관련된 오류 패턴을 생성합니다. 이 과정은 문장 수준의 ASR 오류와 키워드 수준의 음성 변형을 포함하는 두 단계로 이루어져 있습니다. 예를 들어, 사용자가 제공한 작은 데이터셋(10개 이하의 샘플)만으로도 다양하고, 정확한 음성유사성을 가진 키워드 오류를 생성할 수 있습니다.

- **Performance Highlights**: EPA를 통해 DST 모델의 정확도가 45.76%에서 51.12%로 향상되었습니다. 키워드 증강이 주효한 결과로, ASR 오류가 있는 값에 대한 오류율이 크게 감소했습니다.



### Inference is All You Need: Self Example Retriever for Cross-domain Dialogue State Tracking with ChatGP (https://arxiv.org/abs/2409.06243)
- **What's New**: 이 논문에서는 전통적인 대화 상태 추적(Dialogue State Tracking, DST) 방식의 한계를 극복하기 위해 새로운 방법인 ChatGPT를 활용한 도메인 전이(domain transfer) 메커니즘을 제안합니다. 이 방법은 파라미터 업데이트 없이 추론(inference)과 인-컨텍스트 학습(in-context learning)을 통해 DST를 수행하며, 새로운 도메인에 대한 적응력을 높였습니다.

- **Technical Details**: 제안된 방법은 탐색 예제 검색 시스템을 구성하고, 대화 중의 정보 추적을 위해 LLM(대형 언어 모델)을 활용합니다. 이 시스템은 예제 검색과 인-컨텍스트 학습을 통해 새로운 도메인에서도 효과적으로 상태를 추론하도록 지원합니다. 특히, 모델은 기존 데이터 세트에서 특정 예제를 선택하고, 그 선택 과정에서 명확한 설명을 제공합니다.

- **Performance Highlights**: MultiWOZ 데이터 세트를 기반으로 한 실험 결과는 제안된 파라미터 없는 접근 방식이 다양한 도메인에서 경쟁력 있는 성능을 보여주며 일반화 가능성이 높음을 입증하였습니다. 이 연구는 도메인 전이 학습(domain transfer learning) 연구의 새로운 방향을 제시합니다.



### NLP-Powered Repository and Search Engine for Academic Papers: A Case Study on Cyber Risk Literature with CyL (https://arxiv.org/abs/2409.06226)
- **What's New**: 이 논문에서는 연구자들이 관련 자료를 효과적으로 검색하는 데 어려움을 겪고 있다는 문제를 해결하기 위한 새로운 프레임워크를 제안합니다. 이 프레임워크는 Natural Language Processing (NLP) 기술을 활용하여 특정 연구 분야 내 학술 문헌을 자동으로 검색하고 요약 및 클러스터링합니다.

- **Technical Details**: 제안된 시스템은 CyLit이라는 이름의 NLP 기반 저장소를 통해 사이버 리스크 문헌에 특화된 자료를 제공합니다. 이 시스템은 대량의 데이터를 자동으로 처리하고, 연구자가 동적이고 빠르게 발전하는 사이버 리스크 분야에서 트렌드를 추적할 수 있도록 지원합니다.

- **Performance Highlights**: CyLit의 문헌 분류 결과는 기존 서베이 논문이나 ChatGPT로 생성된 결과들과 비교하여 사이버 리스크 연구 문헌에 대한 독특한 통찰력을 제공합니다. NLP 기법을 사용하여 연구자들이 학술 자료를 발견하고 분석하며 활용하는 방식을 혁신할 것을 목표로 하고 있습니다.



### Advancing Topic Segmentation of Broadcasted Speech with Multilingual Semantic Embeddings (https://arxiv.org/abs/2409.06222)
- **What's New**: 최근 방송 뉴스 도메인에서의 음성 기반 주제 분할에 대한 새로운 연구가 발표되었습니다. 이 연구는 기존의 텍스트 기반 분할 알고리즘에서 벗어나 사전 훈련된 음성 인코더를 사용하여 직접적으로 의미론적 표현을 캡처하여 주제를 분할하는 종단 간(end-to-end) 방식의 방법을 제안합니다.

- **Technical Details**:  제안된 접근 방식인 SpeechTopSeg는 Whisper 모델을 사용하여 자동 음성 인식(ASR) 처리를 건너뛰고, 새로운 SONAR 인코더를 사용하여 직접적으로 음성에서 의미적 잠재 표현을 추출합니다. 이 인코더는 다국어에 적합하여 다양한 언어 설정에 적응할 수 있는 가능성을 보여줍니다.

- **Performance Highlights**: 전통적인 파이프라인 접근법의 경우 영어에서 $P_k$ 점수 0.2431을 달성한 반면, 종단 간 모델에서는 0.2564의 경쟁력 있는 $P_k$ 점수를 기록했습니다. 다국어 교육을 진행했을 때 이 점수는 각각 0.1988과 0.2370으로 향상되었습니다.



### SubRegWeigh: Effective and Efficient Annotation Weighing with Subword Regularization (https://arxiv.org/abs/2409.06216)
Comments:
          14 pages, 1 figures, 10 tables

- **What's New**: 새로운 방법 SubRegWeigh는 어노테이션 오류를 탐지하는 데 필요한 시간을 크게 단축할 수 있는 토큰화 기법인 subword regularization을 활용합니다. 이 방법은 CrossWeigh에 비해 4배에서 5배 빠른 성능을 보이며, 텍스트 분류와 NER 작업 모두에서 성능 개선을 이루어냈습니다.

- **Technical Details**: SubRegWeigh는 여러 개의 분할 결과를 바탕으로 어노테이션 오류를 탐지하고, 손실 값에 가중치를 부여하여 오류가 있는 학습 샘플의 영향을 감소시키기 위한 효율적이고 효과적인 방법입니다. 이 방법은 K-fold cross-validation을 사용하지 않으므로 훈련이 한 번만 필요하며, 다양한 토큰화 후보를 입력하여 여러 출력 결과를 얻습니다.

- **Performance Highlights**: SubRegWeigh는 CoNLL-2003 데이터셋에서 CrossWeigh에 비해 어노테이션 가중치를 4배에서 5배 더 빠르게 수행했으며, SOTA(상태 최첨단 기술) 성능을 달성했습니다. 실험 결과, 이 기법은 가짜 부정확 레이블에서도 잘 어노테이션 오류를 인식하였습니다.



### NOVI : Chatbot System for University Novice with BERT and LLMs (https://arxiv.org/abs/2409.06192)
- **What's New**: 대학교 신입생의 적응을 돕기 위해, SKKU 'Everytime'의 게시물과 댓글 데이터를 활용하여 GPT-4o 기반의 챗봇 시스템 NOVI를 개발했습니다. NOVI는 신입생뿐만 아니라 다양한 사람들이 새로운 환경에 적응하는 데 도움을 줄 것으로 기대됩니다.

- **Technical Details**: NOVI는 LangChain을 기반으로 개발되었으며, BLEU, Perplexity, ROUGE-1, ROUGE-2, ROUGE-L, METEOR 등의 다양한 평가 지표를 통해 성능을 평가했습니다. 시스템 흐름은 사용자 쿼리를 처리하기 위해 Flask를 사용하여 RetrievalQA로 데이터를 전송하고, 유용한 정보를 찾아 모델을 미세 조정하는 방식으로 이루어집니다.

- **Performance Highlights**: 이 연구는 신입생들이 대학 생활에 적응하는 데 필요한 구체적인 정보를 제공함으로써, 성공적인 첫 해를 시작할 수 있도록 실질적인 도움을 줄 것으로 기대하고 있습니다. 이는 LLM 연구의 미래 발전을 위한 기초를 마련하는데 기여할 것입니다.



### Can Large Language Models Unlock Novel Scientific Research Ideas? (https://arxiv.org/abs/2409.06185)
Comments:
          24 pages, 12 figures, 6 tables

- **What's New**: 이 논문은 LLMs(대형 언어 모델)가 연구 논문을 기반으로 새로운 연구 아이디어를 생성할 수 있는 능력을 탐구합니다. 연구팀은 4개의 LLM을 다양한 분야에서 조사하여, LLM의 생성 아이디어가 저자의 관점과 얼마나 일치하는지를 평가했습니다.

- **Technical Details**: 연구에서는 Claude-2, GPT-4, GPT-3.5 및 Gemini 모델을 분석하였고, ‘Idea Alignment Score(아이디어 정렬 점수)’와 ‘Idea Distinctness Index(아이디어 독창성 지수)’를 제안하여 생성된 아이디어의 질을 평가했습니다. 인공지능 모델의 성능을 연구하기 위해 460개의 생성 아이디어를 인적 평가하여 혁신성과 관련성을 평가했습니다.

- **Performance Highlights**: Claude-2와 GPT-4는 GPT-3.5와 Gemini보다 저자의 관점과 더 잘 일치하는 미래 연구 아이디어를 생성했습니다. Claude-2는 다른 모델들보다 더 다양한 연구 아이디어를 생성하였고, 연구 결과는 LLM의 아이디어 생성 능력과 한계를 보여줍니다.



### Larger Language Models Don't Care How You Think: Why Chain-of-Thought Prompting Fails in Subjective Tasks (https://arxiv.org/abs/2409.06173)
Comments:
          5 pages, 2 figures, 1 table

- **What's New**: 이 논문은 In-Context Learning (ICL)와 Chain-of-Thought (CoT) 기법이 대형 언어 모델(LLM)에서 어떻게 상호 작용하는지를 조사합니다. 새로운 발견으로, CoT를 사용하더라도 LLM이 과거 지식(prior knowledge)에 의존하는 경향이 지속됨을 보이며, 이는 주관적인 감정 및 도덕적 판단 같은 복잡한 주제에 대한 성능 저하로 이어진다는 점을 강조합니다.

- **Technical Details**: 논문에서는 ICL과 CoT의 조합이 LLM의 추론을 어떻게 변화시키는지를 살펴봅니다. 특히, CoT가 주어진 증거를 무시하고 문제 해결을 위해 과거의 추론 체인을 불러오는 경향을 보여주며, 이로 인해 예측의 포스터리어가 간소화되고 있음을 발견했습니다. 연구는 6개의 최신 LLM을 대상으로 하여 CoT의 성능을 평가하고, CoT로 생성된 추론의 합리성을 분석했습니다.

- **Performance Highlights**: 연구 결과에 따르면, CoT를 적용한 LLM이 주관적 작업, 특히 감정 및 도덕 관련 태스크에서 여전히 성능 저하를 보였습니다. 또한, 더 큰 LLM일수록 CoT를 사용하더라도 이전의 추론 체인이 여전히 고정적(prior)이라는 것을 확인했습니다.



### Deep Learning and Large Language Models for Audio and Text Analysis in Predicting Suicidal Acts in Chinese Psychological Support Hotlines (https://arxiv.org/abs/2409.06164)
- **What's New**: 이 연구는 심리 지원 핫라인에서 수집한 오디오 및 텍스트 데이터를 사용하여 자살 위험을 예측하는 LLM 기반의 혁신적인 접근 방식을 제시합니다. 기존의 연구들과는 달리, Clinical 환경에서 수집한 데이터를 활용하여 1시간 이상의 긴 대화를 분석하는 점이 특징입니다.

- **Technical Details**: 본 연구는 LLM을 활용하여 약 1시간 분량의 대화에서 핵심 기능을 추출한 후, 장기적인 대화의 문맥 정보를 통합하여 자살 행동을 예측하는 파이프라인을 구축합니다. 제안된 LLM 파이프라인은 F1 점수 76.47%를 달성하며, 이는 기존의 수작업 스케일 접근법보다 27.82% 포인트 향상된 결과입니다.

- **Performance Highlights**: 연구 결과, LLM 기반의 간단한 파이프라인은 테스트 세트에서 46개 사례에 대해 76%의 F1 점수를 기록하였으며, 이는 수작업 스케일 평가와 결합했을 때 7% 높은 점수로, 자살 위험 예측에 있어 LLM의 가능성을 보여줍니다.



### Accelerating Large Language Model Pretraining via LFR Pedagogy: Learn, Focus, and Review (https://arxiv.org/abs/2409.06131)
- **What's New**: 이 논문에서는 기존의 LLM (Large Language Model) 사전 학습 방법론과는 다르게, 인간의 학습 방식에서 영감을 얻어 데이터 샘플링 방법을 개선한 새로운 LFR (Learn, Focus, and Review) pedagogy를 제안합니다. 이 방법은 복잡한 데이터 블록에 집중하고 정기적으로 리뷰하여 장기 기억에 정보를 효과적으로 저장할 수 있도록 설계되었습니다.

- **Technical Details**: LFR는 모델의 perplexities를 기록하고, 기억 가능성이 높은 높은 perplexity를 가진 데이터 블록을 자주 다시 방문합니다. GPT-2 모델을 OpenWebText 데이터셋에서 사전 학습하였으며, 학습 속도는 20배 빨라졌고, 모델의 정확도는 기존 OpenAI 모델보다 낮은 perplexity와 높은 정확도를 달성하였습니다.

- **Performance Highlights**: LFR을 통해 GPT-2 모델을 345M에서 1.5B 파라미터로 사전 학습한 결과, 언어 모델링, 질문 응답, 번역 및 문제 해결 등 6개의 하위 작업에서 일관되게 낮은 perplexity와 높은 정확도를 기록했습니다. 또한, 사전 학습 과정에서 약 20배 더 적은 반복 학습으로 이러한 개선을 이루었습니다.



### Estimating the Completeness of Discrete Speech Units (https://arxiv.org/abs/2409.06109)
Comments:
          SLT2024

- **What's New**: 이번 연구에서는 residual vector quantization(RVQ)가 적용된 HuBERT 표현의 정보 완전성(information completeness)과 접근성(information accessibility)을 분석했습니다. 기존의 주장들과는 달리, 벡터 양자화가 음소 정보와 화자 정보를 분리하지 않으며, 잔여 정보(residual information)가 중요하다는 점을 강조합니다.

- **Technical Details**: 정보 이론의 관점에서 벡터 양자화 전후의 정보 완전성과 접근성을 평가하였으며, RVQ 후의 HuBERT 표현에서 나타나는 정보의 양을 낮은 경계(lower bound) 형태로 제시했습니다. 또한, k-means 군집화 방법을 통해 얻은 이산 음성 단위의 정보 완전성을 정량적으로 측정하였습니다. RVQ는 여러 개의 코드북(codebook)을 사용하여 잔여 정보를 세밀하게 캡처하는 방식으로 작동합니다.

- **Performance Highlights**: 실험 결과, HuBERT 이산 단위는 화자 정보가 충분하고, 잔여에 음소 정보가 잘 나타나며, 정보의 잔여분이 자주 버려지는 대신 활용되어야 함을 보여주었습니다. 음성 인식(ASR) 및 화자 검증 등의 다양한 작업에서 이 정보를 성공적으로 평가할 수 있었습니다.



### Doppelg\"anger's Watch: A Split Objective Approach to Large Language Models (https://arxiv.org/abs/2409.06107)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 "generation supervision" 문제를 연구하고, 유용성(helpfulness)과 감독 신호(supervision signal)를 분리하는 새로운 bicameral 아키텍처(bicameral architecture)인 Doppelgänger를 제안합니다.

- **Technical Details**: Doppelgänger는 토큰 생성을 감독하고, 각 토큰까지의 시퀀스에 대한 감독 점수(supervision score)를 예측하는 모듈입니다. 이 새로운 구조는 Transformer 아키텍처를 확장한 것으로, 언어 모델의 기본 능력은 변경되지 않으면서도 다양한 감독 신호에 대한 성능을 독립적으로 최적화할 수 있습니다.

- **Performance Highlights**: 이 접근 방식은 기존의 방법들과 비교해 감독 신호의 최적화를 돕고, LLM의 유용성을 유지하며, 훈련 과정에서 대규모 데이터가 필요하지 않습니다. 이는 다양한 모달리티에서도 적용할 수 있는 장점이 있습니다.



### ClarQ-LLM: A Benchmark for Models Clarifying and Requesting Information in Task-Oriented Dialog (https://arxiv.org/abs/2409.06097)
- **What's New**: 본 논문은 ClarQ-LLM이라는 새로운 평가 프레임워크를 소개하며, 이는 영어-중국어 대화 작업과 대화 에이전트, 평가 메트릭을 포함하여 작업 지향적인 대화에서 에이전트가 명확화 질문을 요청하는 능력을 평가하기 위한 강력한 벤치마크를 제공합니다.

- **Technical Details**: ClarQ-LLM은 31개의 다양한 작업 유형과 각 유형당 10개의 고유한 대화 시나리오로 구성됩니다. 이 시나리오는 정보 요청자(정보 seeker)가 제공자(정보 provider)와의 상호작용을 통해 정보 불확실성을 해결해야 합니다. 기존의 벤치마크와 달리, ClarQ-LLM은 GPT-4o 또는 LLAMA3.1-405B로 구동되는 제공자 대화 에이전트를 포함하여 에이전트 간의 상호작용을 통해 정보를 수집하는 능력을 평가합니다.

- **Performance Highlights**: LLAMA3.1 405B 시커(agent)는 최대 60.05%의 성공률을 달성했으며, 이는 인간 시커의 85% 성공률과 비교하여 상대적으로 낮습니다. 이는 ClarQ-LLM이 향후 연구에 있어 상당한 도전을 제기함을 보여줍니다.



### DetoxBench: Benchmarking Large Language Models for Multitask Fraud & Abuse Detection (https://arxiv.org/abs/2409.06072)
Comments:
          12 pages

- **What's New**: 이 논문은 대규모 언어 모델(LLMs)의 실제 적용, 특히 사기 및 악용 탐지와 같은 고위험 도메인에서의 성능을 평가하기 위한 종합적인 벤치마크 스위트를 제안합니다.

- **Technical Details**: 벤치마크는 스팸 이메일, 증오 발언, 성차별적 언어 등 다양한 실제 상황에서 LLM의 성능을 평가합니다. Anthropic, Mistral AI, AI21 모델을 포함한 최신 LLM들을 비교하여 이 분야에서의 능력을 종합적으로 평가했습니다.

- **Performance Highlights**: LLMs는 개별적인 사기 및 악용 탐지 작업에서 우수한 성능을 보였으나, 성차별적 언어의 다양한 형태를 식별하는 등의 섬세한 실용적 추론이 요구되는 작업에서는 성능 차이가 크게 나타났습니다.



### Identifying the sources of ideological bias in GPT models through linguistic variation in outpu (https://arxiv.org/abs/2409.06043)
- **What's New**: 이 논문은 GPT-3.5와 GPT-4와 같은 Generative AI 모델들이 사회적 고정관념 및 편견을 계속 전파한다는 기존 연구에 이어, 이들 모델이 정치적으로 민감한 주제에 대한 이념적 입장을 취하는지를 탐구합니다.

- **Technical Details**: 이 연구는 정치적 태도가 상반된 국가에서의 언어 변화를 활용하여 기계 학습 모델의 이념적 편향(ideological bias)을 식별하는 독창적인 접근 방식을 제안합니다. 연구 결과, 보수적인 사회와 잘 매핑되는 언어(예: 폴란드어)에서는 GPT 출력이 더 보수적이며, 자유로운 사회에서만 사용되는 언어(예: 스웨덴어)에서는 더 자유로운 경향이 있음을 발견했습니다.

- **Performance Highlights**: GPT-3.5에서 관찰된 언어 간의 차이가 GPT-4에서도 지속되지만, GPT-4는 OpenAI의 필터링 정책으로 인해 상당히 더 자유로워졌습니다. 이 연구의 주요 결론은 Generative 모델의 학습이 편견을 줄이기 위해 고품질 데이터세트에 집중해야 한다는 것이며, 필터링에 의한 편견 추가는 근본적인 학습 데이터의 편견을 제거하지 못함을 보여줍니다.



### Improved Visually Prompted Keyword Localisation in Real Low-Resource Settings (https://arxiv.org/abs/2409.06013)
- **What's New**: 이번 연구에서는 영어 외에도 실제 저자원 언어인 요루바어를 위한 Visually Prompted Keyword Localisation (VPKL) 기술을 처음으로 도입하였으며, 전사 없이 자동으로 쌍을 생성하는 few-shot learning 방법을 활용했습니다.

- **Technical Details**: VPKL은 주어진 이미지 쿼리를 사용하여 음성 발화에서 특정 키워드의 존재를 탐지하고 위치를 파악하는 두 단계로 구성됩니다. 연구에서는 LocAttNet 모델을 사용하며, 이 모델은 이미지와 음성의 쌍을 통해 시각적으로 기반한 음성 모델을 학습합니다. 또한, 포지티브 및 네거티브 예제를 자동으로 생성하기 위한 few-shot mining 기법을 사용하여 데이터 부족 문제를 해결하였습니다.

- **Performance Highlights**: 요루바어의 경우, few-shot mining 기법을 사용할 때 성능이 기존의 전사가 있는 데이터에 비해 더 큰 하락폭을 보였으며, 정밀도 등의 지표에서도 적당한 성과를 보였지만, 완벽한 쌍을 사용할 경우보다 평균 11% 성능 저하가 발생하는 것으로 나타났습니다.



### TransformerRanker: A Tool for Efficiently Finding the Best-Suited Language Models for Downstream Classification Tasks (https://arxiv.org/abs/2409.05997)
- **What's New**: 이 논문에서는 TransformerRanker라는 경량 라이브러리를 소개합니다. 이 라이브러리는 고비용의 fine-tuning 없이도 NLP의 분류 작업을 위해 사전 훈련된 언어 모델(Pre-trained Language Model, PLM)을 효율적으로 순위화할 수 있도록 돕습니다.

- **Technical Details**: TransformerRanker는 Python으로 작성되었으며, PyTorch와 HuggingFace 생태계에 의존합니다. 사용자는 HuggingFace 모델 허브에 있는 다양한 PLM의 목록과 특정 분류 작업을 선택하여 가장 적합한 모델을 쉽게 평가할 수 있습니다. 이 라이브러리는 transferability estimation, H-Score, kNN 등을 구현하여 PLM의 순위를 매기는 여러 방법을 제공합니다.

- **Performance Highlights**: 실험 결과, TransformerRanker는 다양한 PLM의 성능을 효과적으로 비교하고, 사용자가 선택한 분류 작업에 가장 적합한 모델을 빠르게 찾아낼 수 있습니다. 이 라이브러리는 pip-installable open-source로 공개되어 있습니다.



### MessIRve: A Large-Scale Spanish Information Retrieval Datas (https://arxiv.org/abs/2409.05994)
- **What's New**: 이번 논문에서는 스페인어 정보 검색(IR)을 위한 대규모 데이터셋인 MessIRve를 소개합니다. 이 데이터셋은 약 73만 개의 쿼리와 위키피디아에서 수집된 관련 문서로 구성되어 있습니다. MessIRve는 스페인어를 사용하는 다양한 지역을 반영하여 다국적 스페인어 데이터를 제공하는데 중점을 둡니다.

- **Technical Details**: MessIRve는 Google의 자동완성 API에서 쿼리를 수집하고, 그에 해당하는 Google 검색의 ‘featured snippets’를 위키피디아에서 추출하여 관련 문서로 활용하였습니다. 데이터셋은 20개 스페인어 사용 국가에서 수집된 쿼리와 문서로 구성되며, 이로 인해 다양한 스페인어 사용자의 정보 요구를 포괄합니다.

- **Performance Highlights**: MessIRve는 스페인어 정보 검색 시스템 개발을 위한 새로운 기준을 제공합니다. 기존 데이터셋과 비교하여 MessIRve는 더 다양한 주제를 다루며, 스페인어 IR 연구의 발전을 도모할 수 있는 강력한 도구로 작용할 것입니다.



### AI for Mathematics Mathematical Formalized Problem Solving and Theorem Proving in Different Fields in Lean4 (https://arxiv.org/abs/2409.05977)
- **What's New**: 이 논문은 Lean 4와 같은 컴퓨터화된 검증 가능한 포멀 언어(formal languages)를 사용하여 수학 정리를 증명하는 새로운 접근 방식을 제안합니다. 이 방법론은 대규모 언어 모델(LLMs)을 활용하여 자연어(Natural Language) 증명에서 formal steps 및 complete proofs를 생성하는 것입니다.

- **Technical Details**: 저자는 기존의 수학적 형식화 언어는 빠르게 발전하는 언어와의 보폭을 맞추는 데 한계가 있음을 강조하며, AI가 수학 형식화 과정을 지원할 수 있는 방법과 기본 구조 및 전술(tactics)을 소개합니다. 이는 Lean 4에서 증명을 수행하는 방법과 자연어(NL) 증명과의 비교를 포함합니다.

- **Performance Highlights**: 이 방법론은 IMO(International Mathematical Olympiad) 문제를 포함한 예제를 통해 수학적 문제를 효과적으로 해결할 수 있는 가능성을 보여줍니다. 또한, 추상 대수학(abstract algebra)의 정리 증명 시연을 통해 실제 적용 사례를 제시합니다.



### A Small Claims Court for the NLP: Judging Legal Text Classification Strategies With Small Datasets (https://arxiv.org/abs/2409.05972)
- **What's New**: 최근 언어 모델링의 발전으로 텍스트 분류 작업에서 라벨이 부착된 데이터의 필요성이 크게 감소하였습니다. 특히 법률 분야와 같이 전문가 수준의 주석자가 필요한 도메인에서는 여전히 높은 양의 라벨 데이터가 요구됩니다.

- **Technical Details**: 이 연구에서는 50개의 미리 정의된 주제를 가진 법률 분야의 분류 작업을 수행하기 위해 적은 양의 라벨 데이터와 대량의 비라벨 데이터를 최적화하는 전략을 조사하였습니다. 연구팀은 브라질 공공검찰청의 요구 기록을 사용하여 수동으로 입력해야 하는 분야에 대해 자동화된 분류 시스템을 개발하였습니다.

- **Performance Highlights**: 결과적으로, 클래스 50개 예측에서 80.7%의 정확도를 달성하였으며, 이는 수동 입력의 히트율을 크게 초과하는 결과입니다. BERT 언어 모델은 최상의 성능을 보였고, Unsupervised Data Augmentation (UDA)를 사용하여 데이터 증강 및 반지도 학습 전략을 조합하여 이 성과를 달성했습니다.



### Geometric-Averaged Preference Optimization for Soft Preference Labels (https://arxiv.org/abs/2409.06691)
- **What's New**: 이 연구에서는 인간의 선호가 이진적이고 결정론적이라는 기존의 가정을 뒤집고, 개인별로 다를 수 있는 분포적(preferential distribution) 선호를 다룬다. 이를 통해 Direct Preference Optimization (DPO) 알고리즘을 개선하는 방법을 제안한다.

- **Technical Details**: 우리는 distributional soft preference labels를 도입하고, LLM의 출력 가능도를 손실 함수에 가중 지오메트릭 평균(weighted geometric average)을 사용하여 DPO를 개선한다. 이를 통해 소프트 레이블(soft labels)에 기반하여 학습 손실의 스케일이 조정되며, 동등하게 선호되는 응답의 손실은 제로에 가까워진다.

- **Performance Highlights**: 실험 결과, geometric averaging이 정렬 연구를 위한 표준 벤치마크에서 성능을 일관되게 향상시킨다는 것을 시연하였다. 특히, 이 연구에서는 이진 레이블에 비해 더 선호되는 응답을 관찰할 수 있었고, 소극적으로 신뢰할 수 있는 레이블이 대부분을 차지하는 데이터에서 유의미한 개선을 보였다.



### Sortformer: Seamless Integration of Speaker Diarization and ASR by Bridging Timestamps and Tokens (https://arxiv.org/abs/2409.06656)
- **What's New**: 본 논문에서는 기존의 end-to-end diarization 모델과는 다른 비전통적인 목표로 학습한 새로운 신경망 모델인 Sortformer를 제안합니다. 특히, Sort Loss라는 새로운 손실 함수를 도입하여 다중 화자의 오디오를 정렬하는 문제에 접근합니다.

- **Technical Details**: Sort Loss는 diarization 모델이 자동으로 순열을 해결할 수 있도록 하며, 이는 기존의 permutation invariant loss (PIL)와 함께 사용될 수 있습니다. 실험에서는 Sortformer 모델을 사용하여 자동 음성 인식 (ASR) 시스템에서 speaker supervision을 통합하고, sinusoidal kernel function을 이용하여 발화자 레이블 추정 기능을 ASR 인코더 상태에 삽입합니다.

- **Performance Highlights**: 제안된 multispeaker ASR 아키텍처는 adapter 기술을 사용하여 성능 향상을 이루며, Sort Loss와 PIL을 결합한 결과 state-of-the-art 성능을 달성합니다. 또한, 이 시스템은 기존의 ASR 모델과의 호환성을 유지하여 훈련 및 미세 조정 과정을 단순화합니다.



### MoWE-Audio: Multitask AudioLLMs with Mixture of Weak Encoders (https://arxiv.org/abs/2409.06635)
- **What's New**: 이번 논문에서는 AudioLLM(오디오 대형 언어 모델)의 성능을 향상시키기 위해 ‘약한’ 인코더(MoWE: mixtures of weak encoders)를 결합하는 새로운 방법을 제안합니다. 기존의 AudioLLM은 주로 '강한' 인코더와 사전 학습된 대형 언어 모델을 조합하여 사용하고 있지만, 새로운 작업 및 데이터셋에 대한 제한된 능력으로 인해 많은 도전을 받고 있습니다.

- **Technical Details**: MoWE는 기본 인코더를 보완하기 위해 여러 개의 작은 인코더 풀을 도입하고, 오디오 입력에 따라 선택적으로 활성화하여 특징(feature) 추출을 강화하는 전략을 사용합니다. 이러한 접근 방식은 모델 크기를 크게 증가시키지 않으면서도 다중 작업 성능을 개선할 수 있습니다. 제안된 AudioLLM 프레임워크는 강력한 기본 오디오 인코더와 다수의 약한 오디오 인코더로 구성되어 있습니다. 활성화된 인코더의 임베딩은 결합되어 AudioLLM 파이프라인에서 추가 처리됩니다.

- **Performance Highlights**: 실험 결과, MoWE는 오디오LLMs의 다중 작업 성능을 효과적으로 개선하여 다양한 오디오 작업에 대한 적용 가능성을 넓혔습니다. 기존의 강력한 오디오 인코더와 결합함으로써, 새로운 데이터셋 및 작업에 대한 적응력이 크게 향상되었습니다.



### HexaCoder: Secure Code Generation via Oracle-Guided Synthetic Training Data (https://arxiv.org/abs/2409.06446)
Comments:
          24 pages, 16 tables, 8 figures

- **What's New**: HexaCoder는 LLM이 보안이 강화된 코드를 자동으로 생성할 수 있도록 지원하는 혁신적인 접근 방식입니다. 이 방법은 보안 취약점이 있는 코드와 수정된 코드 쌍을 생성하여 데이터 수집의 부담을 줄입니다.

- **Technical Details**: HexaCoder는 두 가지 주요 구성 요소로 구성됩니다: 오라클 기반 데이터 합성 파이프라인과 두 단계의 보안 코드 생성 과정입니다. 데이터 합성 파이프라인은 특정 CWE 유형에 대해 취약한 코드와 수정된 코드의 쌍을 생성하며, 여기서 LLM을 사용하여 취약한 코드를 복구합니다. 보안 오라클은 취약점을 감지하고 LLM이 이를 수정하도록 합니다. 이 과정에서 각 데이터 샘플은 보안 관련 라이브러리와 코드를 포함하여 보안의 두 가지 측면을 통합하는 데 기여합니다.

- **Performance Highlights**: 우리는 세 가지 서로 다른 벤치마크에서 네 가지 LLM을 평가하여 HexaCoder가 생성한 코드의 보안을 개선하면서도 기능적 정확성을 유지한다는 것을 입증하였습니다. 특히, 생성된 취약한 코드의 수를 기존 방법에 비해 최대 85% 감소시켰습니다.



### Length Desensitization in Directed Preference Optimization (https://arxiv.org/abs/2409.06411)
Comments:
          21 pages, 9 figures

- **What's New**: 이 논문에서는 Direct Preference Optimization (DPO)의 최적화 목적이 데이터 길이에 따라 자연스럽게 영향을 받아 과도한 단어 수를 초래하는 현상을 분석하고, 이를 해결하기 위한 길이 비감도 개선 방법인 LD-DPO를 제안합니다. LD-DPO는 길이 선호도를 다른 내재적 선호도와 분리하여 DPO의 최적화를 보다 효과적으로 수행할 수 있도록 합니다.

- **Technical Details**: DPO는 Reinforcement Learning from Human Feedback (RLHF) 프로세스의 일환으로 LLMs의 인간 선호도 정렬을 도와주는 기술입니다. 그러나 DPO는 과도한 자세함으로 인해verbosity라는 문제가 발생합니다. LD-DPO는 이러한 문제를 해결하기 위해 길이에 대한 민감성을 완화하는 알고리즘입니다. 두 가지 모델 설정 (Base 및 Instruct)인 Llama2-13B, Llama3-8B 및 Qwen2-7B을 사용하여 다양한 벤치마크에서 실험을 수행했습니다.

- **Performance Highlights**: LD-DPO는 DPO 및 다른 기준 방법들과 비교하여 10-40%의 길이 감소를 달성하며 더 간결한 응답을 제공합니다. MT-Bench 및 ProofWriter 테스트에서 LD-DPO는 모델의 추론 성능을 유의미하게 향상시킬 뿐만 아니라, 길이 민감성이 모델의 기본 능력과 부정적인 상관관계를 보이는 흥미로운 현상을 관찰했습니다.



### Enhancing Sequential Recommendations through Multi-Perspective Reflections and Iteration (https://arxiv.org/abs/2409.06377)
Comments:
          First 3 authors contributes equally to this work

- **What's New**: 이 논문에서는 SeqRec(Sequential Recommendation)에서 사용자의 동적 선호를 모델링하고 학습하기 위해 Mixture of REflectors (MoRE)라는 새로운 프레임워크를 제안합니다. MoRE는 사용자의 명시적 및 암시적 선호, 협업 신호를 다룰 수 있는 세 가지 반사기(reflector)를 포함합니다. 또한, 이 프레임워크는 사용자 맞춤형 추천을 위한 효과적이고 효율적인 반사 업데이트 전략인 refining-and-iteration을 도입합니다.

- **Technical Details**: MoRE 프레임워크는 LLM(대형 언어 모델)에 기반하여 명시적(user preferences), 암묵적(implicit preferences) 사용자 선호 및 협업 신호(collaborative signals)에 대한 반사를 생성하는 세 가지 반사기를 도입합니다. 각 반사기는 self-improving 전략을 통합하여 생성된 반사의 품질을 평가 및 업데이트합니다. 또한, 메타 반사기(meta-reflector)는 각 사용자 추천에 가장 적합한 반사를 선택하기 위해 contextual bandit 알고리즘을 사용합니다.

- **Performance Highlights**: 광범위한 실험을 통해 MoRE는 state-of-the-art 방법들과 비교했을 때 일관되게 더 나은 추천 성능을 보여주며, 훈련 시간 및 GPU 메모리 사용량이 적습니다. 이 연구는 세 가지 실제 데이터 세트에서 MoRE가 기존의 모든 접근 방식에 비해 우수한 성능을 발휘함을 입증했습니다.



### Enhancing Temporal Understanding in Audio Question Answering for Large Audio Language Models (https://arxiv.org/abs/2409.06223)
Comments:
          5 pages, 3 figures

- **What's New**: 이 논문은 오디오 질문 응답(AQA) 분야에서 오디오의 시간적 맥락을 이해하는 능력을 향상시키기 위한 새로운 다수의 기법을 제안합니다. 특히, 대규모 오디오 언어 모델(LALM)이 시간적 추론에 대한 한계를 극복하기 위해 데이터 증강 기법과 커리큘럼 학습을 통해 성능을 향상시키는 방법을 다룹니다.

- **Technical Details**: 논문에서는 LLM(사전훈련된 대형 언어 모델)을 사용하여 신뢰할 수 있는 오디오 시간 질문과 답변 데이터를 생성하는 데이터 증강 전략을 소개합니다. 또한, 기존의 오디오 질문 응답 모델을 위해 시간 추론 기술을 통합하기 위한 지속적 미세 조정 커리큘럼 학습 전략을 제안합니다. 마지막으로, LALM의 응답과 실제 데이터를 비교할 수 있는 새로운 자동 평가 지표인 Open-Eval을 개발했습니다.

- **Performance Highlights**: 제안된 방법은 최신의 LALM을 사용하여 공개 오디오 벤치마크 데이터셋에서 효과를 입증했습니다. Open-Eval 성능 비교는 기존 메트릭들이 정확한 결과를 포착하지 못하는 경우가 많음을 보여줍니다. 본 연구는 시간 추론 성능을 크게 향상시키면서도 기존 미세 조정 작업의 성능 손실을 최소화하도록 설계되었습니다.



### STUN: Structured-Then-Unstructured Pruning for Scalable MoE Pruning (https://arxiv.org/abs/2409.06211)
- **What's New**: 이번 연구에서는 Mixture-of-experts (MoE) 모델을 위한 효과적인 pruning 방법론을 제안하며, 특히 Structured-Then-UNstructured pruning (STUN) 기법이 기존의 unstructured pruning보다 성능을 더욱 향상시킬 수 있다는 점을 강조합니다.

- **Technical Details**: 이 연구에서 제안하는 STUN 기법은 1개의 H100 GPU와 2시간의 훈련 시간으로 Snowflake Arctic라는 480B 크기의 MoE 모델에서 40%의 희소성을 달성하면서 성능 저하 없이 작업을 수행할 수 있습니다. 제안된 방법은 expert pruning과 unstructured pruning을 조합하여 성능을 최적화하고, 기존의 O(k^n/√n) 복잡도를 O(1)으로 감소시킵니다.

- **Performance Highlights**: 연구 결과, STUN 기법을 적용하면 MoE 모델에서 unstructured pruning이 실패하는 generative 작업에서도 성능을 유지하면서 압축 비율을 대폭 향상시킬 수 있는 것으로 나타났습니다. 특히 GSM8K와 같은 높은 성능을 요구하는 작업에서 효과적입니다.



### SHAPE-IT: Exploring Text-to-Shape-Display for Generative Shape-Changing Behaviors with LLMs (https://arxiv.org/abs/2409.06205)
Comments:
          Accepted for ACM UIST 2024

- **What's New**: 본 논문은 text-to-shape-display라는 혁신적인 접근 방식을 소개하여, 자연어 명령을 통해 pin-based shape displays의 동적인 형태 변화를 생성합니다. 이 시스템은 대규모 언어 모델(LLMs)과 AI-chaining을 활용하여 사용자가 프로그래밍 없이도 텍스트 프롬프트를 통해 형태 변화 행동을 작성할 수 있도록 합니다.

- **Technical Details**: 우리는 SHAPE-IT이라는 LLM 기반의 저작 도구를 개발했습니다. 이 도구는 24x24 형태 디스플레이를 제어하며, 사용자의 텍스트 명령을 실행 가능한 코드로 변환합니다. 시스템 설계에서는 Primitive(기본 형상), Animation(움직임), Interaction(상호작용)이라는 세 가지 핵심 생성 요소를 기반으로 합니다. 또한 AI-chaining을 활용하여 여러 AI 모델 및 작업을 연결, 복잡한 작업을 처리할 수 있습니다.

- **Performance Highlights**: SHAPE-IT의 성능을 평가한 결과, 50개의 샘플에서 82%의 컴파일 성공률을 기록했습니다. 10명의 사용자 연구를 통해 사용자의 피드백을 수집했으며, SHAPE-IT가 구체적인 상호작용 디자인을 통해 빠른 아이디어 발상에 기여할 수 있음을 확인했습니다. 다만, 정확성과 관련된 도전 과제가 발견되어, 앞으로 AI 활용 방안을 더 향상시키기 위한 연구가 필요합니다.



### SQLucid: Grounding Natural Language Database Queries with Interactive Explanations (https://arxiv.org/abs/2409.06178)
Comments:
          Accepted to UIST'24

- **What's New**: 본 논문은 비전문 사용자가 복잡한 데이터베이스 쿼리 프로세스를 이해하고 참여할 수 있도록 돕는 새로운 사용자 인터페이스인 SQLucid를 소개합니다.

- **Technical Details**: SQLucid는 시각적 대응(visual correspondence), 중간 쿼리 결과(intermediate query results), 단계별 SQL 설명(step-by-step SQL explanations) 등을 자연어로 통합하여 사용자의 이해를 증진시키는 기능을 제공합니다. 사용자는 생성된 SQL 쿼리의 개별 단계를 설명하는 NL 설명을 통해 쿼리의 동작을 확인하고, 이러한 설명을 수정하여 모델에 잘못된 부분을 피드백할 수 있습니다.

- **Performance Highlights**: 두 가지 사용자 연구와 하나의 정량적 실험을 통해 SQLucid의 효과를 검증한 결과, 작업 완료 정확도가 49%에서 89%로 향상되었으며, 사용자 신뢰도가 크게 개선되었습니다.



### Investigating Causal Cues: Strengthening Spoofed Audio Detection with Human-Discernible Linguistic Features (https://arxiv.org/abs/2409.06033)
- **What's New**: 이 논문에서는 사람의 귀로 식별할 수 있는 Expert Defined Linguistic Features (EDLFs)를 사용하여 여러 유형의 스푸핑 오디오를 탐지하는 새로운 접근 방법을 제시합니다. 이러한 EDLFs는 음성과 관련된 다양한 특징을 포함하여, 전통적인 오디오 데이터 기능을 보완하는 데 도움을 줍니다.

- **Technical Details**: 하이브리드 데이터셋을 사용하여 사회언어학적 주석이 추가된 여러 유형의 스푸핑 오디오 간의 인과 관계 발견(causal discovery) 및 추론(inferences)을 조사합니다. 연구에서는 기저 무결성 검증(expert ground truth validation) 과정과 비교하여 인과 모델(causal models)의 결과를 분석합니다.

- **Performance Highlights**: 인과 모델은 특히 스푸핑 오디오를 판별하기 위해 언어적 특징을 통합하는 유용성을 보여줍니다. 또한, 인공지능(AI) 모델을 강화하기 위한 인간 지식의 필요성과 기회를 강조하며, 스푸핑 오디오 탐지를 위한 EDLFs 레이블링 자동화를 통한 성능 개선 가능성을 제시합니다.



### Assessing SPARQL capabilities of Large Language Models (https://arxiv.org/abs/2409.05925)
Comments:
          peer reviewed publication at NLP4KGc @ Semantics 2024, see this https URL

- **What's New**: 이 논문에서는 대규모 언어 모델(LLM)과 지식 그래프(KG)의 통합 가능성에 대해 다루고 있으며, 특히 자동화된 벤치마크 작업을 통해 SPARQL SELECT 쿼리를 처리하는 LLM의 기본 능력을 평가합니다.

- **Technical Details**: LLM-KG-Bench 프레임워크를 활용하여 SPARQL SELECT 쿼리와 관련된 다양한 벤치마크 작업을 구현하였고, 주요 작업 유형은 SPARQL 구문 수정(SSF), 텍스트를 SPARQL로 변환(T2S), SPARQL에서 답변으로(S2A), 텍스트에서 답변으로(T2A)입니다.

- **Performance Highlights**: 현재 평가된 LLM들 중 가장 우수한 성능을 보이는 모델 조차도 여러 경우에서 의미론적으로 올바른 SPARQL SELECT 쿼리를 생성하는 데 어려움을 겪고 있으며, LLM의 성능은 특정 모델 및 작업의 복잡성에 크게 의존합니다.



### Programming Refusal with Conditional Activation Steering (https://arxiv.org/abs/2409.05907)
- **What's New**: 이 논문에서는 조건부 활성화 조정(Conditional Activation Steering, CAST) 방법을 제안하여 LLM의 응답 행동을 보다 세밀하게 제어할 수 있는 새로운 접근 방식을 소개합니다.

- **Technical Details**: CAST는 추론(inference) 중 LLM의 활성화 패턴을 분석하여 입력 맥락에 따라 활성화 조정을 선택적으로 적용하거나 보류하는 방법입니다. 다양한 프롬프트(prompt) 카테고리가 모델의 숨겨진 상태(hidden state)에서 서로 다른 패턴을 활성화한다는 관찰에 기초하고 있습니다.

- **Performance Highlights**: CAST 방법을 사용하면 '만약 입력이 증오 발언(hate speech) 또는 성인 콘텐츠(adult content)에 관한 것이라면 거부하라' 또는 '입력이 법률 조언에 관한 것이 아니면 거부하라'와 같은 규칙을 통해 LLM의 응답을 체계적으로 제어할 수 있습니다.



New uploads on arXiv(cs.IR)

### Critical Features Tracking on Triangulated Irregular Networks by a Scale-Space Method (https://arxiv.org/abs/2409.06638)
Comments:
          13pages, ACM SIGSPATIAL 2024

- **What's New**: 본 논문에서는 Triangulated Irregular Networks (TINs)를 사용하는 새로운 scale-space 분석 파이프라인을 소개합니다. 이는 전통적인 그리드 기반의 방법의 한계를 극복하고, 불규칙하게 분포된 포인트 클라우드(point clouds)에서 직접 생성된 TINs의 지형 특징을 효과적으로 분석할 수 있도록 합니다.

- **Technical Details**: 지형 고도 함수(terrain elevation function)를 입력 신호로 사용하여 중요한 지형적(topographic) 특징을 다양한 스케일에서 식별하고 추적합니다. TINs는 불규칙한 경계(irregular boundaries)를 가진 지형을 효과적으로 분석할 수 있으며, 기존의 그리드 기반 방법에 비해 유연성과 정확성을 제공합니다.

- **Performance Highlights**: 우리의 TIN 기반 파이프라인은 그리드 기반 방법에 비해 효율적이고, 정확하며, 해상도(Resolution) 견고성이 우수한 성능을 보여줍니다. 실험 결과, 다양한 지형과 불규칙한 경계에 대해 훨씬 효과적인 결과를 도출했습니다.



### Operational Advice for Dense and Sparse Retrievers: HNSW, Flat, or Inverted Indexes? (https://arxiv.org/abs/2409.06464)
- **What's New**: 이 논문에서는 HNSW (Hierarchical Navigable Small World) 인덱스와 flat 인덱스 간의 trade-offs를 설명하며, 검색의 효율성과 품질을 고려한 운영 지침을 제공합니다. 또한, dense retrieval 모델과 sparse retrieval 모델에 대한 비교 결과를 제공하여 검색 practitioners가 디자인 공간을 탐색하는 데 도움을 줍니다.

- **Technical Details**: 실험 결과는 BEIR 데이터셋을 사용하여 open-source Lucene 검색 라이브러리를 활용하였습니다. 이 연구는 HNSW와 flat 인덱스(양자화 변형 포함)의 인덱싱 시간, 쿼리 평가 성능, 검색 품질 측면에서의 trade-off를 조사했습니다. 또한, BGE와 SPLADE++ 모델을 적용한 dense 및 sparse retrieval 모델의 비교를 통해 다양한 코퍼스에 대한 효과-효율 trade-off를 분석했습니다.

- **Performance Highlights**: 연구 결과 HNSW 인덱스는 대규모 데이터셋에서 성능이 우수하지만, 작은 데이터셋에서는 flat 인덱스가 더 나은 효율을 보일 수 있다는 것을 발견했습니다. 이러한 발견은 검색 practitioners가 다양한 사용 사례에 맞추어 인덱스 선택을 할 수 있도록 돕는 중요한 지침을 제공합니다.



### Enhancing Sequential Recommendations through Multi-Perspective Reflections and Iteration (https://arxiv.org/abs/2409.06377)
Comments:
          First 3 authors contributes equally to this work

- **What's New**: 이 논문에서는 SeqRec(Sequential Recommendation)에서 사용자의 동적 선호를 모델링하고 학습하기 위해 Mixture of REflectors (MoRE)라는 새로운 프레임워크를 제안합니다. MoRE는 사용자의 명시적 및 암시적 선호, 협업 신호를 다룰 수 있는 세 가지 반사기(reflector)를 포함합니다. 또한, 이 프레임워크는 사용자 맞춤형 추천을 위한 효과적이고 효율적인 반사 업데이트 전략인 refining-and-iteration을 도입합니다.

- **Technical Details**: MoRE 프레임워크는 LLM(대형 언어 모델)에 기반하여 명시적(user preferences), 암묵적(implicit preferences) 사용자 선호 및 협업 신호(collaborative signals)에 대한 반사를 생성하는 세 가지 반사기를 도입합니다. 각 반사기는 self-improving 전략을 통합하여 생성된 반사의 품질을 평가 및 업데이트합니다. 또한, 메타 반사기(meta-reflector)는 각 사용자 추천에 가장 적합한 반사를 선택하기 위해 contextual bandit 알고리즘을 사용합니다.

- **Performance Highlights**: 광범위한 실험을 통해 MoRE는 state-of-the-art 방법들과 비교했을 때 일관되게 더 나은 추천 성능을 보여주며, 훈련 시간 및 GPU 메모리 사용량이 적습니다. 이 연구는 세 가지 실제 데이터 세트에서 MoRE가 기존의 모든 접근 방식에 비해 우수한 성능을 발휘함을 입증했습니다.



### HierLLM: Hierarchical Large Language Model for Question Recommendation (https://arxiv.org/abs/2409.06177)
- **What's New**: 이번 연구에서는 학생들의 학습 효율성을 향상시키기 위한 질문 추천 시스템을 제안합니다. 기존의 방법들이 가지는 콜드 스타트(cold start) 문제와 대규모 질문 집합의 문제를 해결하기 위해 대형 언어 모델(LLM)을 기반으로 한 계층적 구조의 HierLLM을 도입하였습니다.

- **Technical Details**: HierLLM은 질문 추천 시스템에서의 학습 이력을 기반으로 한 학습 상태 추정의 한계를 극복하기 위해 LLM의 추론 기능을 활용합니다. 질문 집합의 개수가 많을 때, 관련 개념을 먼저 파악한 후 이를 기반으로 질문을 추천함으로써 선택의 범위를 좁힙니다. 이러한 계층적 구조를 통해 추천 프로세스가 더욱 간소화됩니다.

- **Performance Highlights**: HierLLM의 효과를 검증하기 위해 다양한 실험을 수행한 결과, 기존 방법들에 비해 뛰어난 성능을 보여주었습니다. 실험 결과는 HierLLM이 최신 기술 수준(state-of-the-art)에 도달했음을 입증합니다.



### What makes a good concept anyway ? (https://arxiv.org/abs/2409.06150)
- **What's New**: 이 연구에서는 의료 분야의 온톨로지(ontology)에 새로운 개념을 추가할 때 '좋은' 개념을 정의하기 위한 메트릭(metric)을 제안했습니다.

- **Technical Details**: 메트릭은 개념의 이름 길이(단어 수), 의료 문헌에서의 개념 발생 빈도, 구성 단어의 구문적 범주 등의 요소를 결합하여 구성되었습니다. 추가 요소로 특정 외국어에 매핑 후 용어의 단순성을 고려했습니다. 이러한 요소의 가중치는 Bayesian optimization을 통해 최적화하였습니다.

- **Performance Highlights**: 이 메트릭은 세 명의 의료 전문가와의 50.67%의 전반적인 일치를 보였으며, 이는 Krippendorff의 알파(Krippendorff's alpha)로 측정되었습니다.



### LexBoost: Improving Lexical Document Retrieval with Nearest Neighbors (https://arxiv.org/abs/2409.05882)
Comments:
          ACM DocEng 2024

- **What's New**: LexBoost라는 새로운 접근법을 제안합니다. 이는 고전적인 lexical retrieval 방법의 면수성과 density retrieval의 효율성을 결합한 모델로, 문서의 이웃과의 관계를 활용하여 문서를 더 효과적으로 순위 매깁니다.

- **Technical Details**: LexBoost는 문서의 lexical relevance 점수와 이웃 문서의 점수를 결합하여 문서의 순위를 매기는 방법입니다. 이는 Cluster Hypothesis를 적용한 것으로, dense 모델을 통해 인접 문서를 오프라인에서 미리 식별하고, sparse 모델을 통해 온라인에서 문서의 relevance를 추정합니다. 추가적인 지연(latency) 없이도 기존의 lexical methods에 비해 성능을 향상시킵니다.

- **Performance Highlights**: LexBoost는 BM25와 같은 최신 lexical retrieval 방법에 비해 통계적으로 유의미한 개선을 달성하며, 전통적인 dense re-ranking보다도 더 나은 성능을 보여줍니다. 이를 통해 높은 latency가 요구되는 exhaustive dense retrieval에 맞먹는 결과를 도출할 수 있습니다.



### CF-KAN: Kolmogorov-Arnold Network-based Collaborative Filtering to Mitigate Catastrophic Forgetting in Recommender Systems (https://arxiv.org/abs/2409.05878)
Comments:
          9 pages, 7 figures, 4 tables

- **What's New**: 이 논문에서는 추천 시스템의 협업 필터링(Collaborative Filtering, CF)에서의 새로운 접근 방식인 CF-KAN을 제안합니다. CF-KAN은 Kolmogorov-Arnold networks (KANs)를 활용하여 비선형 함수 학습을 통해 기존 MLP의 단점을 극복합니다.

- **Technical Details**: CF-KAN은 KAN 기반의 오토인코더를 바탕으로 구성되어 있으며, 이는 사용자-항목 상호작용의 복잡성을 효과적으로 포착할 수 있게 설계되었습니다. KAN의 엣지 레벨 학습을 통해 희소한 사용자-항목 상호작용을 모델링 할 수 있어, 지속적 학습이 필요한 동적 환경에서의 강력한 내성을 보여줍니다.

- **Performance Highlights**: CF-KAN은 기존 최첨단 방법들보다 최대 8.2% 향상된 Recall@20을 보이며, 동적 추천 환경에서 MLP 변형 모델보다 우수한 성능을 보입니다. 또한, 훈련 속도가 매우 빠르며, 카타스트로픽 포겟팅(catastrophic forgetting)에 대한 내성 또한 입증되었습니다.



### CSRec: Rethinking Sequential Recommendation from A Causal Perspectiv (https://arxiv.org/abs/2409.05872)
- **What's New**: 본 연구에서는 Causal Sequential Recommendation (CSRec)이라는 새로운 접근 방식을 제안하여, 추천 시스템의 사용자 의사결정 과정에서의 복잡한 인과 관계를 분석하는 방법을 소개합니다.

- **Technical Details**: CSRec는 사용자의 구매 결정에 영향을 미치는 다양한 요소들(추천 시스템, 사용자의 과거 구매, 사용자 선호 등)을 격리하고 정량적으로 분석할 수 있는 프레임워크를 제공합니다. 이는 인과 그래프(causal graph)를 바탕으로 하여, 사용자가 추천된 아이템을 수락할 확률을 예측하는 데 중점을 둡니다.

- **Performance Highlights**: CSRec는 기존의 추천 시스템에서의 성능을 향상시키며, 여러 베이스라인과 비교해 우수한 결과를 보였습니다. 실험 결과, CSRec는 현재의 최첨단(SOTA, state-of-the-art) 모델들에 대해 경쟁력 있는 성능을 입증하였습니다.



### Latent Diffusion Bridges for Unsupervised Musical Audio Timbre Transfer (https://arxiv.org/abs/2409.06096)
- **What's New**: 이번 논문은 오디오 신호의 음색(timbre) 특성을 변형하면서 멜로디 구조(melodic structure)를 유지하는 새로운 방법론을 제시합니다.

- **Technical Details**: 제안된 방법은 두 개의 diffusion bridge를 기반으로 하며, CocoChorales Dataset을 사용하여 학습되었습니다. 각 diffusion 모델은 특정 악기와 함께 Gaussian prior에 대해 훈련되며, 추론(inference) 과정에서 소스 모델(source model)과 타겟 모델(target model)로 지정되어 음색 전이를 용이하게 합니다.

- **Performance Highlights**: 실험 결과, 우리의 방법은 VAEGAN 및 Gaussian Flow Bridges(GFB)와 비교하여 Fréchet Audio Distance(FAD)가 더 낮고 멜로디 보존(melody preservation) 측면에서도 더 나은 성능을 나타냈습니다. 또한 Gaussian prior로부터의 노이즈 수준($\sigma$)을 조정함으로써 멜로디 보존 정도와 음색 전이의 양을 조절할 수 있다는 점을 발견했습니다.



### Assessing SPARQL capabilities of Large Language Models (https://arxiv.org/abs/2409.05925)
Comments:
          peer reviewed publication at NLP4KGc @ Semantics 2024, see this https URL

- **What's New**: 이 논문에서는 대규모 언어 모델(LLM)과 지식 그래프(KG)의 통합 가능성에 대해 다루고 있으며, 특히 자동화된 벤치마크 작업을 통해 SPARQL SELECT 쿼리를 처리하는 LLM의 기본 능력을 평가합니다.

- **Technical Details**: LLM-KG-Bench 프레임워크를 활용하여 SPARQL SELECT 쿼리와 관련된 다양한 벤치마크 작업을 구현하였고, 주요 작업 유형은 SPARQL 구문 수정(SSF), 텍스트를 SPARQL로 변환(T2S), SPARQL에서 답변으로(S2A), 텍스트에서 답변으로(T2A)입니다.

- **Performance Highlights**: 현재 평가된 LLM들 중 가장 우수한 성능을 보이는 모델 조차도 여러 경우에서 의미론적으로 올바른 SPARQL SELECT 쿼리를 생성하는 데 어려움을 겪고 있으며, LLM의 성능은 특정 모델 및 작업의 복잡성에 크게 의존합니다.



New uploads on arXiv(cs.CV)

### GeoCalib: Learning Single-image Calibration with Geometric Optimization (https://arxiv.org/abs/2409.06704)
Comments:
          Presented at ECCV 2024

- **What's New**: 이 논문은 단일 이미지에서 3D 기하학의 보편적인 규칙을 활용하여 카메라 매개변수를 더욱 견고하고 정확하게 추정하는 새로운 딥 신경망 (DNN) 모델인 GeoCalib을 소개합니다.

- **Technical Details**: GeoCalib은 최적화 프로세스를 통해 프로젝티브 기하학에 대한 지식을 활용하며, 이는 차별화 가능한 (differentiable) 최적화가 가능하여 단일 이미지에서 수직 방향과 카메라 내부 매개변수를 학습하고 추정할 수 있습니다. 이 방법은 명시적인 감독 없이 유용한 시각적 신호를 추출할 수 있도록 돕습니다.

- **Performance Highlights**: 여러 벤치마크 실험에서 GeoCalib은 기존의 전통적인 기법 및 기존의 딥러닝 기법보다 더 견고하고 정확한 결과를 보이며, 실패 사례를 플래그하는 데 도움이 되는 불확실성을 내부적으로 추정합니다.



### LEIA: Latent View-invariant Embeddings for Implicit 3D Articulation (https://arxiv.org/abs/2409.06703)
Comments:
          Accepted to ECCV 2024. Project Website at this https URL

- **What's New**: LEIA는 기존의 방법들이 아닌, 다중 시점에서 관찰한 객체의 다양한 상태를 활용하여 새로운 모습의 동적 3D 객체 상태를 생성할 수 있는 혁신적인 방법을 제안합니다.

- **Technical Details**: LEIA는 하이퍼네트워크(hypernetwork)를 이용하여 다양한 상태의 NeRF를 매개변수화하는 방법으로, 뷰 불변의 잠재 표현(view-invariant latent representation)을 학습합니다. 이는 멀티뷰 이미지(multi-view images)를 사용하여 객체의 동적 표현을 구현합니다.

- **Performance Highlights**: LEIA는 기존의 동작 정보를 필요로 하는 방법들보다 더 뛰어난 성능을 보여주며 다양한 관절 구성에 대해 관측된 적 없는 새로운 상태의 객체를 생성할 수 있습니다.



### Hint-AD: Holistically Aligned Interpretability in End-to-End Autonomous Driving (https://arxiv.org/abs/2409.06702)
Comments:
          CoRL 2024, Project Page: this https URL

- **What's New**: 이 논문에서는 자율주행의 해석 가능성 문제를 해결하기 위한 새로운 시스템인 Hint-AD를 소개합니다. 이는 AD 모델의 전체적인 인식-예측-계획 출력에 맞춰 언어를 생성하며, 이를 통해 사용자와의 신뢰를 향상시킵니다.

- **Technical Details**: Hint-AD는 세 가지 모듈로 이루어져 있습니다: 1) Holistic Token Mixer, 2) Language Decoder, 3) 전통적인 AD 프레임워크입니다. Intermediate outputs를 언어 디코더에 맞게 변형하는 Holistic Token Mixer 모듈을 개발하였고, 데이터 정렬 작업을 통해 언어 출력과 AD 모델의 중간 출력 간의 연관성을 강화했습니다.

- **Performance Highlights**: Hint-AD는 여러 언어 작업에서 최첨단 성능을 기록하였으며, 실험 결과에 따르면 드라이빙 설명(기준 대비 20.4% 향상), 3D 밀집 캡셔닝(185% 향상), VQA(1.2% 향상), 드라이빙 명령 예측(1.2% 향상)을 달성했습니다.



### GigaGS: Scaling up Planar-Based 3D Gaussians for Large Scene Surface Reconstruction (https://arxiv.org/abs/2409.06685)
- **What's New**: GigaGS는 대규모 장면에서 고품질 표면 재구성을 위해 3D Gaussian Splatting (3DGS)을 최초로 활용한 연구입니다. 이는 공간 지역의 상호 가시성에 기반하여 카메라를 그룹화하고 병렬 처리하는 새로운 분할 전략을 적용합니다.

- **Technical Details**: GigaGS는 Level-of-Detail (LoD) 프레임워크 내에서 다중 뷰 사진 및 기하학적 일관성 제약을 통합하여 재구성의 품질을 향상시킵니다. 각 공간 블록은 독립적으로 최적화되며 전체 장면을 형성하기 위해 매끄럽게 병합됩니다.

- **Performance Highlights**: 다양한 데이터셋을 대상으로 한 포괄적인 실험에서 GigaGS의 일관된 개선 결과가 확인되었으며, 이는 대규모 표면 재구성에서의 우수성을 입증합니다.



### Alignist: CAD-Informed Orientation Distribution Estimation by Fusing Shape and Correspondences (https://arxiv.org/abs/2409.06683)
Comments:
          Accepted to ECCV 2024

- **What's New**: 이 논문에서는 CAD 모델을 활용하여 3D 물체의 pose distribution을 더 효과적으로 추정하는 Alignist라는 새로운 방법을 제안합니다. 이 방법은 기존의 단일 pose 추정치에 의존하는 것에서 벗어나, 다수의 학습 이미지가 필요하지 않도록 합니다.

- **Technical Details**: Alignist는 CAD 모델에서 얻은 대칭을 고려한 대응(distribution)과 모양 정보를 활용하여 pose 분포를 학습합니다. CAD 모델의 사전 지식을 통해, 네트워크는 더욱 선명한 모드에 집중하여 빠른 수렴 속도를 보입니다. 이 방법은 dual-branch MLP를 통해 S⁢O⁢(3)𝑆𝑂3SO(3)에서 두 가지 분포를 추정하고, 일반화에 도움을 주는 훈련 방식으로 작동합니다.

- **Performance Highlights**: SYMSOL-I 및 T-Less 데이터셋에서 우리의 접근법은 벤치마크 결과를 달성하였으며, 특히 텍스처가 없는 물체에서의 불확실성과 모호성을 효과적으로 캡처하는 데에 성공하였습니다.



### A Semantic Segmentation Approach on Sweet Orange Leaf Diseases Detection Utilizing YOLO (https://arxiv.org/abs/2409.06671)
- **What's New**: 이 연구는 YOLOv8와 같은 첨단 인공지능 모델을 활용하여 오렌지 나뭇잎 질병 진단을 위한 새로운 방법을 소개합니다. 기존의 수작업 검사 방법의 비효율성을 극복하고자 합니다.

- **Technical Details**: YOLOv8은 객체 탐지(object detection)와 이미지 분석(image analysis)에 최적화된 성능을 지니며, 훈련(training) 및 검증(validation) 단계에서 80.4%의 정확도를 기록했습니다. VIT는 특징 추출(feature extraction)에서 99.12%의 높은 정확도를 보여, 농업에서의 질병 탐지 가능성을 시연합니다.

- **Performance Highlights**: 이 연구는 AI 기술의 시범적 도입을 통해 질병 탐지의 혁신적인 변화를 가져올 것으로 기대하며, 환경적으로도 지속 가능한 농업 및 농약 사용 감소의 가능성을 강조합니다.



### Data Collection-free Masked Video Modeling (https://arxiv.org/abs/2409.06665)
Comments:
          ECCV 2024

- **What's New**: 이 논문에서는 비디오 트랜스포머(Transformer)를 위한 효과적인 자기 지도 학습(self-supervised learning) 프레임워크를 소개합니다. 이 프레임워크는 정적 이미지(static images)를 활용하여 비디오 데이터 수집 비용을 줄일 수 있습니다. 특히 모듈인 Pseudo Motion Generator (PMG)를 정의하여 이미지 변환을 반복적으로 적용하여 의사 모션 비디오(pseudo-motion videos)를 생성합니다.

- **Technical Details**: 제안된 프레임워크는 정적 이미지로부터 의사 모션 비디오를 생성하기 위해 PMG 모듈을 사용합니다. 이러한 비디오는 마스크 비디오 모델링(masked video modeling)에 활용되며, 기존의 방법들보다 스페이션-템포럴(spatio-temporal) 특성을 효과적으로 학습할 수 있습니다. 이 연구에서는 실험을 통해 제안된 프레임워크가 정적 이미지를 사용하는 기존 방법들보다 유의미하게 개선된 성능을 보이며, 실제 및 합성 비디오를 사용하는 기존 사전 학습 방법들을 부분적으로 초월함을 입증합니다.

- **Performance Highlights**: 행동 인식(action recognition) 작업에서 실험 결과, 제안된 프레임워크는 의사 모션 비디오를 통해 비디오 트랜스포머가 전이 가능하고 강력한 비디오 특성을 학습할 수 있게 하여 기존 방법들보다 현저한 성능 개선을 보입니다.



### World-Grounded Human Motion Recovery via Gravity-View Coordinates (https://arxiv.org/abs/2409.06662)
Comments:
          Accepted at SIGGRAPH Asia 2024 (Conference Track). Project page: this https URL

- **What's New**: 새로운 Gravity-View (GV) 좌표계 시스템을 도입하여 단일 비디오에서 세계에 기반한 인간 모션을 효과적으로 복원하는 방법을 제안합니다.

- **Technical Details**: GV 좌표계는 중력을 기반으로 하고 카메라 시점 방향에 의해 정의되어 이미지-포즈 매핑의 모호성을 크게 줄입니다. 제안된 방법은 각 프레임에서 인간 포즈를 추정하고, 이를 전역 좌표계로 변환하여 누적 오류를 피하는 방식으로 작동합니다. 트랜스포머 모델 및 Rotary Positional Embedding (RoPE)을 활용하여 장기간의 모션 시퀀스를 효과적으로 처리합니다.

- **Performance Highlights**: 자연적인 모션을 카메라 공간과 세계에 기반한 환경 모두에서 복원하며, 현재 기술 대비 정확성과 속도에서 우수한 성능을 보여줍니다.



### Image Vectorization with Depth: convexified shape layers with depth ordering (https://arxiv.org/abs/2409.06648)
- **What's New**: 이 논문에서는 이미지 벡터화(Image Vectorization)의 새로운 방법을 제안합니다. 이 방법은 깊이 순서를 고려하여 각 형태 레이어를 수축(convexify)하는 데 Curvature-based Inpainting 기법을 사용합니다. 기존 방법들과 달리 이 과정은 점진적인 곡선 추가나 훈련이 필요하지 않으며, 유기적인 형태 재구성을 위해 깊이 정보를 활용합니다.

- **Technical Details**: 이미지의 색상 양자화(raster image)로부터 연결된 같은 색깔의 구성 요소를 형태 레이어로 정의한 후, 새로운 깊이 순서 에너지를 이용하여 이들 사이의 깊이 순서를 구성합니다. 평면에서 각 형태의 경계를 매끄럽게 연장하기 위해 Euler의 엘라스틱 곡률 기반 인페인팅을 활용합니다. 결과적으로, SVG(Scalable Vector Graphics) 형식으로 이러한 벡터화된 이미지를 저장합니다.

- **Performance Highlights**: 이 방법은 기존의 계층 기반 벡터화 기법과 비교하여 높은 품질을 유지하면서 계산 복잡도를 감소시킵니다. 또한, 다양한 수치 결과와 최근의 벡터화 기법에 대한 비교를 통해 제안된 모델의 유효성을 입증하고 있습니다.



### EyeCLIP: A visual-language foundation model for multi-modal ophthalmic image analysis (https://arxiv.org/abs/2409.06644)
- **What's New**: EyeCLIP은 277만 개 이상의 멀티 모달 안과 이미지와 일부 텍스트 데이터를 활용하여 개발된 시각-언어(visual-language) 기반 모델입니다. 본 연구는 기존의 단일 모달리티 모델의 한계를 넘어, 다양한 모달리티에서의 멀티 뷰 정보를 활용하여 눈 질병을 조기에 감지할 수 있는 방법을 제안합니다.

- **Technical Details**: EyeCLIP은 자가 감독(self-supervised) 복원, 멀티 모달 이미지 대조 학습(multi-modal image contrastive learning), 이미지-텍스트 대조 학습(image-text contrastive learning)을 결합한 프리트레이닝(pretraining) 전략을 도입하여, 다양한 모달리티의 공유 표현(shared representation)을 학습합니다. 이러한 방법은 라벨이 붙지 않은 다수의 멀티 모달 데이터를 효과적으로 해석할 수 있게 합니다.

- **Performance Highlights**: 14개의 벤치마크 데이터셋을 사용한 평가 결과, EyeCLIP은 눈 및 전신 질병 관련 다양한 다운스트림 작업에서 최첨단 성능을 달성했습니다. 특히, EyeCLIP은 실세계의 긴 꼬리(long-tail) 시나리오에서 몇 샷(few-shot) 및 제로 샷(zero-shot) 능력을 보여주며, 질병 분류, 시각적 질문 응답(visual question answering), 크로스 모달 검색(cross-modal retrieval)에서 뛰어난 성과를 보이고 있습니다.



### SaRA: High-Efficient Diffusion Model Fine-tuning with Progressive Sparse Low-Rank Adaptation (https://arxiv.org/abs/2409.06633)
Comments:
          Parameter efficient finetuning method

- **What's New**: 최근 확산 모델(Diffusion Models)의 발전으로 이미지 및 비디오 생성 작업에서 상당한 진전을 이루었습니다. 이 논문에서는 비효율적인 매개변수를 완전히 활용하기 위한 새로운 모델 미세 조정 방법(SARA)을 제안합니다.

- **Technical Details**: 이 연구에서는 먼저 사전 훈련된 확산 모델에서 매개변수의 중요성을 조사하고, 절대값 기준으로 가장 작은 10%에서 20%의 매개변수가 생성 과정에서 기여하지 않는다는 사실을 발견했습니다. 이를 기반으로, SARA 방법은 이 임시 비효율적인 매개변수를 재활용하여 태스크 전반에 걸친 지식을 학습하기 위한 희소 가중치 행렬 최적화에 해당합니다. 또한, 과적합(overfitting)을 완화하기 위해 핵 노름(nuclear norm) 기반의 저순위 희소 훈련 체계를 제안합니다.

- **Performance Highlights**: 제안된 방법은 SD 모델에 대한 미세 조정을 통해 검증되었으며, 모델의 일반화 능력을 유지하면서 LoRA와 같은 전통적인 미세 조정 방법보다 우수한 성능을 보여주었습니다. SARA는 효율적인 구현을 위한 단 한 줄의 코드 수정을 요구하며 기존 방법과도 원활하게 호환됩니다.



### Towards Localizing Structural Elements: Merging Geometrical Detection with Semantic Verification in RGB-D Data (https://arxiv.org/abs/2409.06625)
Comments:
          6 pages, 5 figures. 3 tables

- **What's New**: 이 논문은 RGB-D 카메라를 활용하여 건물 구성 요소의 실시간 로컬라이징(LOCALIZING) 파이프라인을 제안하며, 벽 및 바닥 표면 탐지를 위한 순수 3D 평면 검출을 위한 기하학적 계산과 포인트 클라우드 데이터의 의미 범주 검증을 통합합니다.

- **Technical Details**: 제안된 파이프라인은 RGB-D 카메라에서 수집된 포인트 클라우드를 활용하여 3D 평면을 탐지하고, 의미 범주를 검증하는 병렬 다중 스레드 아키텍처를 채택하여 환경 내 모든 평면의 자세와 방정식을 정확하게 추정합니다. 이 과정에서, 파노틱 세그멘테이션 검증을 통해 맵 구조를 형성하는 평면을 필터링하고, 검증된 건물 구성 요소만 유지합니다.

- **Performance Highlights**: VSLAM 프레임워크에 제안된 방법을 통합하여, 탐지된 환경 기반의 의미 요소로 맵을 조정함으로써 장면 이해와 맵 재구성 정확도를 개선할 수 있음을 입증했습니다. 이를 통해 로봇이 구조적 개체(예: 방)를 감지하고, 건물 구성 요소 간의 레이아웃 및 관계를 파악할 수 있게 합니다.



### MVGaussian: High-Fidelity text-to-3D Content Generation with Multi-View Guidance and Surface Densification (https://arxiv.org/abs/2409.06620)
Comments:
          13 pages, 10 figures

- **What's New**: 본 논문은 텍스트-3D 콘텐츠 생성의 새로운 통합 프레임워크를 제안하여 기존의 문제점인 'Janus 문제' 및 긴 학습 시간과 디테일 부족 문제를 해결합니다.

- **Technical Details**: 제안하는 프레임워크는 Score Distillation Sampling (SDS)과 3D Gaussian Splatting(3DGS)의 결합을 통해 다각도 가이드를 사용하여 3D 모델 구조를 반복적으로 형성하고, gaussian을 표면에 가깝게 정렬하는 새로운 밀도화 알고리즘을 소개합니다.

- **Performance Highlights**: 이 방법은 25분 만에 고품질 결과를 생성하며, 기존 방법에 비해 훈련 속도와 품질 모두에서 현저한 향상을 보입니다.



### Hierarchical Multi-Label Classification with Missing Information for Benthic Habitat Imagery (https://arxiv.org/abs/2409.06618)
- **What's New**: 이번 연구에서는 최첨단 self-supervised learning (SSL) 기법을 대규모 해저 이미지 데이터셋인 BenthicNet에 적용하여 복잡한 계층적 다중 레이블(HML) 분류 작업에서의 성능을 분석했습니다.

- **Technical Details**: BenthicNet 데이터셋의 88%는 주석이 없는 이미지로 구성되어 있으며, 이를 SSL 기법으로 사전 학습(Pre-training)하여 모델 학습에 활용합니다. 또한, 계층적 학습 방식인 C-HMCNN을 기반으로 하여 다양한 결측 정보가 존재하는 환경에서 적용합니다.

- **Performance Highlights**: 모델은 in-domain 해양 데이터에서 self-supervision으로 사전 학습을 받을 경우, ImageNet에서 사전 학습된 모델보다 더 깊고 정확한 분류 성능을 달성할 수 있다는 것을 발견했습니다.



### When to Extract ReID Features: A Selective Approach for Improved Multiple Object Tracking (https://arxiv.org/abs/2409.06617)
Comments:
          8 pages, 5 figures. Presents a selective approach for ReID feature extraction in Multiple Object Tracking, reducing computational overhead while maintaining accuracy. Tested on StrongSORT and Deep OC-SORT using MOT17, MOT20, and DanceTrack datasets. Code: this https URL, this https URL

- **What's New**: 이 논문에서는 feature extraction의 오버헤드를 최소화하면서도 정확성과 모듈성, 구현 용이성을 유지하는 선택적 접근 방식을 조사합니다. StrongSORT 및 Deep OC-SORT에 적용하여 효과를 입증하며, MOT17, MOT20 및 DanceTrack 데이터셋에서의 실험 결과를 보여줍니다.

- **Technical Details**: 제안된 방법은 Kalman Filter를 사용하는 기존의 TbD (Tracking-by-Detection) 트래커에 적용될 수 있으며, feature extraction을 위한 탐지를 제한하는 메커니즘과 시간적 관련성을 향상시켜 매칭된 feature의 정확도를 높이는 기능을 포함합니다. 이러한 접근 방식은 SOTA (State-Of-The-Art) 방법에 통합될 수 있습니다.

- **Performance Highlights**: 실험 결과는 제안된 방법이 occlusion 동안 feature extraction의 장점을 유지하면서도 실행 시간을 크게 줄이며, 특히 DanceTrack에서 변형 및 외관 유사성을 예방하여 정확도를 개선함을 보여줍니다.



### Improving the Precision of CNNs for Magnetic Resonance Spectral Modeling (https://arxiv.org/abs/2409.06609)
Comments:
          11 pages, 1 figure, 2 tables

- **What's New**: 이번 연구는 자기 공명 스펙트로스코픽 이미징(Magnetic Resonance Spectroscopic Imaging)의 첨단 가능성을 탐구하며, 머신러닝을 활용하여 데이터 처리의 도전 과제를 해결하는 방안을 제시합니다.

- **Technical Details**: 연구에서는 CNN(Convolutional Neural Networks)을 활용하여 스펙트럼 모델링의 정밀도를 높이는 방법을 다루며, 정확한 오류 특성을 분석하기 위해 평균 오류 메트릭스(mean error metrics) 외에도 표준 편차(standard deviations), 신뢰 구간(confidence intervals) 등의 포괄적인 정밀 메트릭스가 필요하다는 점을 강조합니다.

- **Performance Highlights**: 연구 결과는 CNN을 사용한 회귀 작업(regression tasks) 시 그 기술의 장점과 단점을 비교하며, 각 기술의 작동 메커니즘에 대한 깊이 있는 통찰을 제공합니다.



### A Practical Gated Recurrent Transformer Network Incorporating Multiple Fusions for Video Denoising (https://arxiv.org/abs/2409.06603)
Comments:
          5 pages, 5 figures

- **What's New**: 본 논문은 단일 프레임 지연만으로도 최고 수준(State-of-the-art, SOTA)의 비디오 디노이징 성능을 달성하는 multi-fusion gated recurrent Transformer network(GRTN)를 제안합니다. 기존 다중 프레임 비디오 디노이징 방식의 지연 문제를 해결합니다.

- **Technical Details**: GRTN은 공간 디노이징 모듈과 시간 디노이징 모듈로 구성되어 있습니다. 'reset gate'와 'update gate'를 사용하여 이전 프레임의 관련 정보를 선택하고 현재 프레임의 특징과 융합합니다. 또한, RSSTE(Residual Simplified Swin Transformer with Euclidean distance)를 도입하여 노이즈가 있는 특징을 계산하는 데 있어 강인성을 높입니다.

- **Performance Highlights**: 비교 실험 결과, 제안된 GRTN은 단일 프레임 지연만으로 SOTA 다중 프레임 디노이징 네트워크와 동등한 성능을 보여줍니다. 이는 실제 카메라 응용에서의 효율성을 크게 향상시킵니다.



### Lightweight Multiscale Feature Fusion Super-Resolution Network Based on Two-branch Convolution and Transformer (https://arxiv.org/abs/2409.06590)
Comments:
          11 pages,12 figures

- **What's New**: 이 논문에서는 두 가지 경량화 구조인 Convolution과 Transformer를 결합한 새로운 다중 스케일 특징 융합 네트워크 모델을 제안합니다. 이 모델은 국소 및 전역 정보를 상호 융합하여 이미지 복원 성능을 향상시킵니다.

- **Technical Details**: 제안된 모델은 두 개의 분기 구조를 가지며, 하나의 분기는 Transformer를 사용하여 지역 특징을 추출하고, 다른 분기는 깊이 분리가 가능한 합성곱과 주의(attention) 모듈을 통해 전역 거칠고 큰 특징을 추출합니다. 이 네트워크는 중간 단계에서 두 분기로부터 추출된 특징 정보를 상호작용시켜 이미지 복원에 유리한 충분한 정보를 보존합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 동일한 파라미터 수를 가지는 다른 경량화 모델에 비해 최적의 이미지 복원 성능을 보였습니다.



### Seg-HGNN: Unsupervised and Light-Weight Image Segmentation with Hyperbolic Graph Neural Networks (https://arxiv.org/abs/2409.06589)
Comments:
          BMVC 2024

- **What's New**: 이번 연구에서는 이미지 분석을 위한 새로운 방법으로 하이퍼볼릭 매니폴드(hyperbolic manifold)를 제안합니다. 특히 Seg-HGNN이라는 경량 하이퍼볼릭 그래프 신경망(graph neural network)을 도입하여 이미지 세분화(image segmentation) 문제를 해결합니다.

- **Technical Details**: Seg-HGNN은 하이퍼볼릭 기하학(hyperbolic geometry)을 활용하여 복잡한 위계적 관계를 포착하고, 주어진 소규모 임베딩(embedding) 크기에서도 효과적인 결과를 냅니다. 훈련 가능한 매개변수(parameter)가 7.5k 미만이며, GTX1650 같은 일반적인 GPU에서 초당 약 2개의 이미지를 처리할 수 있습니다.

- **Performance Highlights**: Seg-HGNN은 VOC-07 및 VOC-12 데이터셋에서 현재 최고의 비지도(unsupervised) 방법보다 각각 2.5% 및 4% 더 향상된 성능을 보여주며, CUB-200 및 ECSSD 데이터셋에서는 세분화에서 각각 0.8% 및 1.3%의 성능 개선을 기록했습니다.



### Transtreaming: Adaptive Delay-aware Transformer for Real-time Streaming Perception (https://arxiv.org/abs/2409.06584)
Comments:
          Submitted to AAAI 2025

- **What's New**: 이 연구에서는 "Transtreaming"이라는 혁신적인 실시간 객체 감지 방법을 소개합니다. 이 방법은 동적 계산 지연을 처리하며, 여러 미래 프레임을 예측하고 현 시점과 가장 잘 일치하는 출력을 선택할 수 있는 적응형 지연 인식 transformer를 기반으로 합니다.

- **Technical Details**: Transtreaming은 실시간 스트리밍 인식을 위한 새로운 접근 방식을 제안하며, 마감 시간을 준수하기 위해 런타임 정보를 통합하여 여러 프레임을 동시 예측합니다. 이 방법은 기존 최첨단 기술을 초월하는 성능을 보여주며, 강력한 V100부터 보통의 2080Ti까지 다양한 장치에서 안정적인 성능을 발휘합니다.

- **Performance Highlights**: Transtreaming은 모든 플랫폼에서 인식 정확도의 최고 수준을 달성하였으며, 대부분의 SOTA 방법들이 낮은 성능의 장치에서 단일 프레임 내에서 계산을 완료하는 데 어려움을 겪는 반면, 모든 종류의 장치에서 엄격한 실시간 처리 요구 사항을 충족합니다.



### Semi-Supervised 3D Object Detection with Chanel Augmentation using Transformation Equivarianc (https://arxiv.org/abs/2409.06583)
Comments:
          Accepted to 2024 IEEE International Conference on Image Processing (ICIP)

- **What's New**: 이번 논문에서는 3D 반지도 학습(3D semi-supervised learning)에서의 성능 향상을 위한 새로운 teacher-student 프레임워크를 제안합니다. 이 프레임워크는 채널 증강(channel augmentation)을 활용하여, 적은 수의 레이블이 있는 데이터로도 안정적이고 효과적인 3D 객체 탐지를 가능하게 합니다.

- **Technical Details**: 제안된 방법은 Transformation Equivariant Detector (TED)를 기반으로 하며, 이는 다양한 변환 조합을 탐색하고 다채널 변환 불변(transform equivariant) 특성을 효율적으로 집계합니다. Teacher 네트워크에 대해 고정된 채널 증강을 채택하여 안정적인 pseudo-label을 생성하고, Student 네트워크는 다양한 강한 채널 증강을 통해 데이터의 다양성을 높입니다.

- **Performance Highlights**: KITTI 데이터셋에서 기존 SOTA(State of the Art) 3D 반지도 객체 탐지 모델들을 능가하는 성능 향상을 달성하였습니다. 표준으로 사용된 HSSDA 모델과 비교하여 channel IoU consistency를 적용함으로써 더욱 신뢰성 높은 pseudo-box를 사용하여 성능을 극대화했습니다.



### Quantifying and Enabling the Interpretability of CLIP-like Models (https://arxiv.org/abs/2409.06579)
- **What's New**: 이 논문에서는 CLIP 같은 모델의 해석 가능성을 정량화하기 위한 연구를 제안하고, OpenAI와 OpenCLIP의 여섯 가지 서로 다른 CLIP 모델에서 그 결과를 분석했습니다. 특히 TEXTSPAN 알고리즘을 사용하여 각 Attention Head를 구체적인 속성으로 분해하고, 새로운 메트릭스를 통해 해석 가능성을 평가했습니다.

- **Technical Details**: 이 연구에서는 TextSpan 알고리즘을 사용하여 각 Attention Head와 연결된 텍스트 설명을 분해하고, In-Context Learning을 통해 속성을 라벨링합니다. 두 가지 메트릭스인 entanglement score (엉킴 점수)와 association score (연관 점수)를 도입하여 해석 가능성을 정량화했습니다.

- **Performance Highlights**: 결과적으로, 큰 CLIP 모델이 더 작은 모델보다 일반적으로 더 해석 가능하다는 것을 발견했습니다. CLIP-InterpreT라는 해석 가능성 분석 도구를 소개하고, 이 도구는 다섯 가지 유형의 분석을 제공하여 사용자가 CLIP 모델의 내부 작동 원리를 이해할 수 있도록 돕습니다.



### PoseEmbroider: Towards a 3D, Visual, Semantic-aware Human Pose Representation (https://arxiv.org/abs/2409.06535)
Comments:
          Published in ECCV 2024

- **What's New**: 본 연구는 3D 포즈, 인물 사진 및 텍스트 설명을 통해 인간 포즈의 시맨틱(semantic), 비주얼(visual) 및 3D 정보를 통합하여 향상된 포즈 표현을 생성하는 새로운 방법을 제시합니다.

- **Technical Details**: 우리는 다양한 모달리티를 결합하여 더욱 풍부한 포즈 임베딩 공간을 구축하는 멀티모달(multi-modal) 프레임워크를 설계했습니다. 이 과정에서 트랜스포머(transformer)를 이용하여 정보를 집계하고, 각각의 모달리티 공간으로 재투영하여 학습합니다.

- **Performance Highlights**: 제안된 포즈 표현은 (1) SMPL 회귀(SMPL regression) 작업 및 (2) 세부 지침 생성(pose instruction generation) 작업에서 개선된 성능을 보여주며, 특히 자동 피트니스 코칭의 응용 가능성을 가진다는 점이 주목할 만합니다.



### In Flight Boresight Rectification for Lightweight Airborne Pushbroom Imaging Spectrometry (https://arxiv.org/abs/2409.06520)
Comments:
          10 pages, 6 figures

- **What's New**: 최근 하이퍼스펙트럴 카메라가 UAV와 같은 가벼운 공중 플랫폼에서 작동하도록 소형화되었습니다. 이 논문은 원시 스펙트럼 이미지만을 사용하여 '푸시-브룸' 하이퍼스펙트럴 센서의 Tie Point 추출 및 카메라 캘리브레이션을 위한 새로운 방법을 제안합니다.

- **Technical Details**: 이 방법은 롤 모션으로 인한 고주파 왜곡을 교정하기 위해 확률적 모델을 사용하여 이미지 라인 간의 이동을 보정합니다. 기존 GPS/INS 데이터나 지형 모델 없이도 서브 픽셀 정확도로 이동을 추정할 수 있습니다. 또한, 피치 모션과 속도 차이로 인한 왜곡을 해결하기 위해 y-scale 불변 매칭 접근 방식을 제안합니다.

- **Performance Highlights**: 제안된 방법은 하이퍼스펙트럴 카메라의 보정 정확도를 약 0.2°로 유도하며, 기존의 최첨단 Tie Point 생성 방법보다 성능이 우수합니다. 실험 결과, 이 방법은 자동 정합의 수를 증가시키고, 매칭의 정확도를 향상시킵니다.



### Aligning Machine and Human Visual Representations across Abstraction Levels (https://arxiv.org/abs/2409.06509)
Comments:
          51 pages

- **What's New**: 이 논문에서는 인간의 행동을 더 잘 모사하는 인공지능 시스템을 만들기 위해, 인간의 개념 지식이 계층적으로 구성되는 방식과 AI 모델의 표현 간의 불일치를 강조합니다.

- **Technical Details**: AI 모델이 인간의 유사성 판단을 더 잘 반영하도록 하는 새로운 Kullback-Leibler 다이버전스(KL divergence) 기반의 목표 함수를 사용하며, AligNet이라는 대규모 유사성 데이터셋을 생성합니다. 이 데이터셋은 두 가지 레벨의 개념적 구체성을 모두 포함하여 기계 학습 모델에 인간 지식의 구조를 주입합니다.

- **Performance Highlights**: AligNet을 통해 모델과 인간의 판단 간의 정렬을 개선하며, 머신 러닝 과제에서 일반화 및 분포 외(out-of-distribution) 강건성(robustness)을 향상시킵니다. 이 개선된 모델들은 다수의 시각적 유사성 작업에서 인간의 행동을 더 잘 근사할 수 있습니다.



### Neural Laplacian Operator for 3D Point Clouds (https://arxiv.org/abs/2409.06506)
Comments:
          SIGGRAPH Asia 2024 (Journal Track)

- **What's New**: 이 논문에서는 K-nearest neighbors (KNN) 그래프를 이용하여 포인트 클라우드에서 Laplacian operator를 정의하는 새로운 접근 방식을 제안합니다. 기존 방법들은 주로 지역 삼각 측량(Local Triangulation)에 의존했으나, 본 연구는 이를 단순화하여 수학적 정확성을 향상시킵니다.

- **Technical Details**: KNN 그래프에서 Laplacian operator를 학습하기 위해 그래프 신경망(Graph Neural Networks, GNNs)을 사용합니다. 학습 과정에서는 ground-truth Laplacian operator의 동작을 모방하는probe functions를 활용하여 GNN을 훈련시키고, 각 엣지의 가중치를 자동으로 학습하도록 합니다. 이로써 필요한 구조를 직접적으로 개선할 필요 없이 신뢰할 수 있는 Laplacian matrix를 작성할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 기존의 최첨단 방법들에 비해 10배 이상의 오차 감소를 달성하고, 희소 포인트 클라우드(Sparse Point Clouds)의 경우에도 우수한 성능을 보입니다. 본 연구의 NeLo는 노이즈에 강하며, 미지의 형태에 대한 일반화 능력이 뛰어나고, 다양한 Laplacian 기반 기하 처리 알고리즘에 직접 적용하여 높은 정확도를 구현합니다.



### Elucidating Optimal Reward-Diversity Tradeoffs in Text-to-Image Diffusion Models (https://arxiv.org/abs/2409.06493)
- **What's New**: 이 논문에서는 Text-to-Image (T2I) diffusion 모델의 보상 해킹(reward hacking) 문제를 해결하기 위한 새로운 방법인 Annealed Importance Guidance (AIG) 를 제안합니다. AIG는 다양한 이미지를 생성하면서도 보상을 최적화하는 방법으로, 기존의 방식보다 나은 성능을 보입니다.

- **Technical Details**: AIG는 Annealed Importance Sampling에 영감을 받아 생성된 회귀적 규제 방식으로, diffusion 모델의 다양성을 유지하면서도 보상-다양성 간의 Pareto-Optimal 무역균형을 달성합니다. 또한, 노이즈 분포에서 데이터로 변환되는 Markov 과정을 사용하며 KL divergence와 LoRA 스케일링 기법을 분석합니다.

- **Performance Highlights**: 실험 결과, AIG는 Stable Diffusion 모델에 적용되었을 때 이미지의 다양성과 품질 모두를 향상시켰습니다. 사용자가 진행한 연구에 따르면, AIG는 다양한 모델 아키텍처 및 보상 함수에서 생성된 이미지의 질과 다양성을 개선하는 것으로 나타났습니다.



### UAVDB: Trajectory-Guided Adaptable Bounding Boxes for UAV Detection (https://arxiv.org/abs/2409.06490)
Comments:
          7 pages, 5 figures, 3 tables

- **What's New**: 본 논문에서는 드론 기술의 발전에 따라 UAV(무인 항공기) 탐지의 정확성 향상을 위한 새로운 방법론인 Patch Intensity Convergence (PIC) 기법을 제안합니다. 이 기법은 UAV 탐지 작업을 위한 고해상도 바운딩 박스 생성을 가능하게 하며, 라벨링 작업의 필요성을 줄여줍니다.

- **Technical Details**: UAVDB는 PIC 기법을 바탕으로 생성된 UAV 탐지를 위한 새로운 데이터베이스입니다. UAVDB는 저해상도 이미지 대신 고해상도 비디오를 활용하여 다양한 크기와 거리의 UAV를 정확히 탐지할 수 있도록 설계되었습니다. PIC 기법은 UAV의 경로 데이터를 통해 바운딩 박스를 적응적으로 형성하며, 이를 통해 다양한 스케일에서 UAV를 탐지할 수 있습니다.

- **Performance Highlights**: YOLOv8 시리즈 탐지기를 사용하여 UAVDB의 성능을 철저히 벤치마킹한 결과, 모든 모델에서 높은 AP 점수를 기록하며, 다양한 스케일과 시나리오에서 동등한 성능을 입증하였습니다. 이 결과는 UAVDB가 UAV 탐지 기술 발전에 중요한 데이터베이스가 될 수 있음을 시사합니다.



### Mitigating Hallucination in Visual-Language Models via Re-Balancing Contrastive Decoding (https://arxiv.org/abs/2409.06485)
Comments:
          PRCV

- **What's New**: 이 논문에서는 Re-Balancing Contrastive Decoding (RBD) 방법을 제안하여 Visual-Language Models (VLMs)의 주의 분포를 재조정하고, 이미지 정보에 대한 신뢰성을 높이며, 텍스트의 편향성을 줄이는 데 중점을 두고 있습니다.

- **Technical Details**: RBD 방법은 텍스트와 비주얼 두 가지 가지를 사용하여 VLM의 주의 분포를 조정합니다. 텍스트 가지는 이미지 노이즈를 주입하여 모델의 텍스트 의존성을 자극하고, 비주얼 가지는 주요 토큰을 선택하여 주의 메커니즘을 개선합니다. 실험 결과는 RBD 방법이 기존 방법들을 초월하여 Hallucinations을 줄이면서도 모델의 일반적인 능력은 유지함을 보였습니다.

- **Performance Highlights**: RBD 방법은 CHAIR 및 POPE 메트릭에서 기존의 최첨단 방법들을 초과하는 성능을 보여주며, VLM의 전체 효율성을 유지하면서 Hallucinations 문제를 완화합니다.



### NeIn: Telling What You Don't Wan (https://arxiv.org/abs/2409.06481)
- **What's New**: 이 논문은 시각-언어 모델(vision-language models, VLMs)에서 부정 표현을 이해하는 능력을 평가하기 위한 첫 번째 대규모 데이터셋인 Negative Instruction (NeIn)을 소개합니다. 기존 데이터셋의 부재로 인하여 VLMs가 부정적 질의를 이해하는 데 어려움을 겪고 있다는 점을 지적합니다.

- **Technical Details**: NeIn 데이터셋은 총 530,694개의 쿼드러플(quadruples)로 구성되어 있으며, 각각 소스 이미지, 원본 캡션, 부정 문장, 목표 이미지가 포함됩니다. 데이터셋은 MS-COCO를 기반으로 자동 생성되었으며, BLIP 및 MagicBrush 두 가지 VLM을 활용하여 목표 이미지를 생성하고, BLIP를 통해 잘못된 샘플을 필터링합니다.

- **Performance Highlights**: 실험 결과, 최신 VLM들조차도 부정 질의를 이해하는 데 어려움을 겪는 것으로 나타났습니다. 이 연구는 VLM의 부정 이해를 개선하기 위한 새로운 연구 방향을 열어주며, 향후 연구자들이 사용할 수 있는 평가 방법론을 제안합니다.



### Weakly-supervised Camera Localization by Ground-to-satellite Image Registration (https://arxiv.org/abs/2409.06471)
Comments:
          Accepted by ECCV 2024

- **What's New**: 이 논문은 소비자 수준의 GPS 및 나침반 센서 또는 도시 규모의 이미지 검색을 통해 얻은 대략적인 위치 및 방향 후에 지상에서 위성 이미지 매칭을 통해 카메라 포즈 정확도를 향상시키기 위한 약한 감독 학습 전략을 제안합니다.

- **Technical Details**: 이 연구는 RTK(Real Time Kinematics) 기반의 데이터 수집 과정에서 발생하는 문제를 해결하는 방식으로, 노이즈가 있는 포즈 라벨로 학습하는 네트워크를 훈련하기 위해 대조 학습(contrastive learning)을 활용합니다. 특히, 각 지상 이미지에 대해 긍정적 및 부정적 위성 이미지를 유도하여, 매칭 프로세스에서 최적의 상대 포즈를 기반으로 특성 표현(feature representation)을 학습합니다.

- **Performance Highlights**: 실험 결과, 본 연구의 약한 감독 학습 전략은 정확한 포즈 라벨에 의존하는 최근의 최첨단 방법보다 교차 지역에서 가장 우수한 성능을 보여줍니다.



### Learning Generative Interactive Environments By Trained Agent Exploration (https://arxiv.org/abs/2409.06445)
- **What's New**: 본 논문에서는 Genie라는 세계 모델을 발전시킨 GenieRedux와 GenieRedux-G 모델을 소개합니다. 이 모델들은 강화 학습 기반의 에이전트를 사용하여 다양한 환경에서 훈련 데이터를 생성하여 더 나은 성능을 발휘합니다.

- **Technical Details**: GenieRedux는 원래 Genie 모델의 구성 요소들을 구현하였으며, STTN(Spatiotemporal Transformer) 아키텍처를 활용합니다. GenieRedux-G는 에이전트의 행동을 조건으로 하여 예측을 수행하며, 이는 예측의 불확실성을 제거하는 데 도움을 줍니다.

- **Performance Highlights**: 실험 결과, GenieRedux-G는 Visual Fidelity와 controllability에서 뛰어난 성능을 보였습니다. 특히, Coinrun 환경에서의 평가에서는 훈련된 에이전트를 기준으로 한 모델이 무작위 에이전트에 비해 다양한 상황에서 더 우수한 성능을 나타냈습니다.



### Knowledge Distillation via Query Selection for Detection Transformer (https://arxiv.org/abs/2409.06443)
- **What's New**: 이 논문은 DETR 모델의 크기를 줄이며 성능을 유지하기 위해 knowledge distillation 기술을 활용한 새로운 접근 방식을 제안합니다.

- **Technical Details**: 강조된 요소로는 Group Query Selection 전략과 QSKD (Knowledge Distillation via Query Selection for DETR) 프레임워크가 있습니다. Group Query Selection은 Generalized Intersection over Union (GIoU)을 기반으로 쿼리를 세분화하여 유용한 hard-negative 쿼리를 찾습니다. QSKD 프레임워크는 Attention-Guided Feature Distillation (AGFD) 및 Local Alignment Prediction Distillation (LAPD)을 포함하여 교사 모델의 중간 특징 및 출력을 최적화합니다.

- **Performance Highlights**: MS-COCO 데이터셋에서 다양한 DETR 아키텍처의 평균 정확도 (Average Precision, AP)를 크게 향상시켰습니다. 예를 들어, Conditional DETR ResNet-18의 AP는 35.8에서 39.9로 증가했습니다.



### Prompt2Fashion: An automatically generated fashion datas (https://arxiv.org/abs/2409.06442)
- **What's New**: 이 연구에서는 사용자의 요구에 따라 다양한 경우, 스타일 및 체형에 맞춘 패션 이미지 데이터셋을 자동으로 생성할 수 있는 방법을 제안합니다. 기존의 주석이 달린 이미지를 사용하지 않고 AI를 통해 완전히 생성된 이 데이터셋은 다양한 표준과 개인화 요구 사항을 충족합니다.

- **Technical Details**: 우리는 여러 Large Language Models (LLMs)와 Diffusion Model을 활용하여 패션 이미지를 생성하는 방법을 개발하였습니다. 사용자 지정 프롬프트에 따라 ‘스타일, 경우, 성별’ 및 ‘스타일, 경우, 유형’의 변량 트리플렛을 사용하여 최종 이미지를 생성합니다. LLM의 언어 생성 능력과 Diffusion Model의 이미지 생성 능력을 검증했습니다.

- **Performance Highlights**: 이 데이터셋은 2000개의 샘플로 구성되어 있으며, 각 샘플은 LLM 출력, 원본 트리플렛 및 생성된 이미지로 이루어집니다. 인간 평가자의 등급도 활용되어 패션 콘텐츠의 품질과 적합성을 보장하며, 다양한 체형과 상황을 반영하여 마케팅 및 디자인에 활용될 수 있는 가능성을 보여줍니다.



### A Likelihood Ratio-Based Approach to Segmenting Unknown Objects (https://arxiv.org/abs/2409.06424)
Comments:
          13 pages, 2 figures, and 4 tables

- **What's New**: 본 논문에서는 Out-of-Distribution (OoD) 세그멘테이션 과제를 해결하기 위한 새로운 방법을 제안합니다. 대규모 기초 모델을 활용하여 OoD 세그멘테이션 성능을 향상시키는 "Adaptive Unknown Estimation Module (UEM)"을 도입하여, 기존 네트워크의 학습된 특징 표현을 유지하면서 아웃라이어 (outlier) 분류를 수행할 수 있는 방법을 제공합니다.

- **Technical Details**: UEM은 아웃라이어 데이터의 분포와 알려진 클래스의 일반 분포를 학습합니다. 이를 바탕으로 likelihood-ratio 기반의 아웃라이어 점수 함수가 제안되며, 이 함수는 UEM의 신뢰성과 픽셀 단위 세그멘테이션 네트워크의 신뢰성을 융합하여 알려지지 않은 객체를 감지합니다. 제안된 방법은 다양한 데이터셋에서 최첨단 성능을 달성하며, 평균 정밀도 평균 (average precision)에서 이전 최고 방법 보다 5.74% 향상된 결과를 보입니다.

- **Performance Highlights**: 제안된 방법은 여러 데이터셋에서 최첨단 성능을 기록하였으며, 높은 인라이어 (inlier) 성능을 유지한 채로 false-positive 비율을 낮추었습니다. 모델의 전반적인 성능 향상과 함께 기존의 아웃라이어 감독 기법보다 더 나은 결과를 보였습니다.



### Sources of Uncertainty in 3D Scene Reconstruction (https://arxiv.org/abs/2409.06407)
Comments:
          To appear in ECCV 2024 Workshop Proceedings. Project page at this https URL

- **What's New**: 이 논문은 Neural Radiance Fields (NeRFs)와 3D Gaussian Splatting (GS) 방법의 불확실성(uncertainty) 출처를 체계적으로 분류하고, 이를 다루기 위한 새로운 기법을 제안합니다.

- **Technical Details**: 논문에서는 노이즈(noise), 가림(occlusion), 혼란스러운 이상치(confounding outliers), 부정확한 카메라 위치(camera pose)로 인한 불확실성을 다룰 수 있는 방법들을 소개합니다. NeRF 및 GS 기반 방법에 불확실성 추정 기법을 추가하고, 학습된 불확실성(output) 및 앙상블(ensembles) 기법을 통해 재구성의 민감도를 평가하는 실험을 진행했습니다.

- **Performance Highlights**: 연구 결과, NeRF/GS 기반의 방법들이 불확실성을 고려한 3D 재구성을 위해 다양한 불확실성 측면을 다루어야 한다는 점이 강조되었습니다.



### AMNS: Attention-Weighted Selective Mask and Noise Label Suppression for Text-to-Image Person Retrieva (https://arxiv.org/abs/2409.06385)
- **What's New**: 본 논문은 텍스트-이미지 인물 검색에서 발생하는 두 가지 새로운 문제인 Noisy Correspondence (NC)와 노이즈 페어링 문제를 다루며, Random Masking의 영향을 분석합니다. 새로운 Bidirectional Similarity Distribution Matching (BSDM) 손실 함수와 Attention-Weighted Selective Mask (AWM) 전략을 제안하여 분별력을 높임으로써 모델의 인식 능력을 개선합니다.

- **Technical Details**: 새로운 교차 모드 아울링 경과 Bidirectional Similarity Distribution Matching (BSDM) 손실 함수는 예측 분포와 실제 분포 간의 양방향 관계를 고려하여 노이즈 라벨로 인한 부정적 영향을 최소화합니다. 이와 함께 Weight Adjustment Focus (WAF) 손실을 도입하여 어려운 샘플에 대한 모델의 처리 능력을 강화합니다. AWM 전략은 이미지의 적절한 영역을 강조하여 특징을 더욱 개선할 수 있도록 돕습니다.

- **Performance Highlights**: 제안된 방법은 다른 연구자들의 방법에 통합되었을 때 성능이 향상되는 것으로 나타났으며, mAP 및 mINP 지표에서 거의 모든 비교 방법 중 최고 성능을 달성하였습니다. 이는 제안된 방법의 탁월성을 입증하고 실제 검색 시스템에서 더욱 효과적으로 사용될 수 있음을 나타냅니다.



### A Cross-Font Image Retrieval Network for Recognizing Undeciphered Oracle Bone Inscriptions (https://arxiv.org/abs/2409.06381)
- **What's New**: 이번 연구에서는 오라클 뼈 비문(OBI) 문자를 해독하기 위해 통합된 크로스 폰트 이미지 검색 네트워크(CFIRN)를 소개합니다. 이는 OBI 문자와 다른 문자 형태 간의 연관성을 설정하며, 고대 필기학자들이 해석하는 행동을 시뮬레이션합니다.

- **Technical Details**: CFIRN은 시암 네트워크(siamese network)를 이용하여 다양한 서체의 문자 이미지에서 깊은 특징을 추출합니다. 이 과정에서는 다중 스케일 특징 통합(multiscale feature integration, MFI) 모듈과 다중 스케일 세분화 분류기(multiscale refinement classifier, MRC)가 사용됩니다. 또한, ConvNeXt 기반의 인코더를 채택하여 OBI 문자와 갤러리 폰트 간의 특징을 공유하며, 다양한 스케일 정보를 통합하여 정밀한 특징 표현을 생성합니다.

- **Performance Highlights**: 세 가지 고대 문자 이미지 검색 데이터셋에서 실험을 실시한 결과, OBI 문자가 주어졌을 때 CFIRN이 다른 서체의 문자와 효과적으로 정확한 매칭을 달성하는 것을 증명하였습니다.



### Distilling Generative-Discriminative Representations for Very Low-Resolution Face Recognition (https://arxiv.org/abs/2409.06371)
- **What's New**: 이 논문에서는 매우 저해상도 얼굴 인식을 용이하게 하는 생성-판별적 표현 증류 접근법을 제안합니다. 이는 생성 표현과 교차 해상도 정렬 지식 증류를 결합하여 수행됩니다.

- **Technical Details**: 제안된 접근법은 두 개의 증류 모듈을 통해 생성적 및 판별적 모델을 함께 증류하여 매우 저해상도 얼굴 인식을 돕습니다. 첫째, 생성 표현 증류는 얼굴 초해상화 고정 모델의 인코더를 사용하여 특징 회귀를 통해 학생 모델을 학습시키고, 그 후 학생 모델의 기초 구조를 동결합니다. 둘째, 판별적 표현 증류는 미리 훈련된 얼굴 인식기를 사용하여 교차 해상도 관계적 대비 증류를 통해 학생 모델을 학습하도록 지원합니다.

- **Performance Highlights**: 광범위한 실험을 통해 제안된 방법은 저해상도 얼굴의 인식 정확도를 향상시키며, 매우 저해상도의 얼굴에서 누락된 세부 정보를 복구하는 데 효과적임을 보여주었습니다.



### Texture-AD: An Anomaly Detection Dataset and Benchmark for Real Algorithm Developmen (https://arxiv.org/abs/2409.06367)
- **What's New**: Texture-AD 데이터셋은 산업 결함 탐지 알고리즘의 성능을 평가할 수 있도록 구성된 최초의 데이터셋으로, 여러 제품의 다양한 명세를 갖춘 훈련 세트와 실제 생산 과정에서 촬영된 결함 있는 제품으로 구성된 테스트 세트를 제공합니다.

- **Technical Details**: Texture-AD는 15종의 천, 14종의 반도체 웨이퍼, 10종의 금속판 등의 다양한 이미지로 구성되어 있으며, 이를 통해 표면 결함을 포함한 10종 이상의 결함을 탐지할 수 있습니다. 이 데이터셋은 고해상도 이미지 43,120,431,204,312,043,120을 포함하며, 각 제품의 결함 부위에 대한 픽셀 단위 주석을 제공합니다.

- **Performance Highlights**: 실험 결과, Texture-AD는 최신 알고리즘에게도 어려운 도전 과제가 되었으며, 알고리즘의 견고성과 일반화 능력을 평가하는 기준으로 활용될 수 있습니다.



### DiffQRCoder: Diffusion-based Aesthetic QR Code Generation with Scanning Robustness Guided Iterative Refinemen (https://arxiv.org/abs/2409.06355)
- **What's New**: DiffQRCoder는 효율적으로 스캔 가능한 미적 QR 코드를 생성하기 위한 새로운 방법론으로, Scanning-Robust Perceptual Guidance (SRPG)를 도입하여 미적 요소를 보존하면서도 QR 코드의 정확성을 보장합니다.

- **Technical Details**: DiffQRCoder는 두 단계로 구성된 아키텍처를 통해 동작합니다. 첫 번째 단계에서는 ControlNet을 사용하여 시각적으로 매력적이지만 스캔할 수 없는 QR 코드가 생성되고, 두 번째 단계에서는 Gaussian noise를 추가하여 QR 코드를 잠재 표현으로 변환한 후, 제작한 QR 코드를 SRPG를 통해 학생하는 과정을 거칩니다. 또한 Scanning Robust Manifold Projected Gradient Descent (SR-MPGD)라는 후처리 기술도 개발하였습니다.

- **Performance Highlights**: 실험 결과에 따르면, DiffQRCoder는 기존 방법들과 비교해 스캔 성공률(SSR)에서 우수한 성능을 보여주었고, ControlNet 전용 접근 방식을 통해 SSR을 60%에서 99%로 크게 향상시켰습니다. 주관적 평가는 미적 매력이 뛰어난 QR 코드를 생산함을 보여줍니다.



### Multi-Weather Image Restoration via Histogram-Based Transformer Feature Enhancemen (https://arxiv.org/abs/2409.06334)
Comments:
          arXiv admin note: text overlap with arXiv:2409.03249

- **What's New**: 이 논문에서는 다양한 기상 조건에서 손상된 이미지를 복원하기 위한 새로운 접근 방식을 제안합니다. Task Sequence Generator와 히스토그램 기반 Transformer 모듈을 활용하여 복잡한 기상 조건에서 효과적으로 기능을 추출하고 자동화된 방식으로 이미지 품질을 향상시키는 모델을 개발했습니다.

- **Technical Details**: 이 모델은 Task Intra-patch Block (TIPB)과 Task Sequence Generator를 사용하여 이미지에서 작업별 특징을 효과적으로 추출합니다. TIPB는 이미지를 작은 패치로 나누어 손상된 이미지에서 특징을 추출하고, 추출된 특징은 Task-Seuqence Generator에 입력되어 각 단계에서 작업 시퀀스를 생성합니다. 또한, 히스토그램 기반의 transformer 모듈은 네트워크의 주춧돌로서 사용되어 전역 및 지역 동적 범위 기능을 포획합니다.

- **Performance Highlights**: 제안된 모델은 공개 데이터셋에서 최첨단 성능을 달성하였으며, 복잡한 기상 조건을 자동으로 처리할 수 있는 가능성이 있습니다.



### SDF-Net: A Hybrid Detection Network for Mediastinal Lymph Node Detection on Contrast CT Images (https://arxiv.org/abs/2409.06324)
Comments:
          10 pages, 4 figures

- **What's New**: 이 논문에서 제안하는 Swin-Det Fusion Network (SDF-Net)은 림프절(lymph node) 탐지의 어려움을 해결하기 위해 세분화(segmentation)와 탐지(detection) 기능을 통합합니다. 특히, 자동 융합 모듈(auto-fusion module)을 설계하여 서로 다른 수준에서 특성 맵을 융합하여 림프절의 다양한 형태와 크기에 대한 탐지 능력을 향상시킵니다.

- **Technical Details**: SDF-Net은 두 가지 경로(경로 1: 세분화 경로(seg-path), 경로 2: 탐지 경로(det-path)를 사용하여 LN 탐지를 수행합니다. 세분화 경로에서는 가우시안 커널(Gaussian kernel)을 기반으로 가상 마스크(pseudo-mask)를 생성하고, 탐지 경로에서는 비 앵커(anchor-free) 접근 방식을 채택하여 위치와 경계 상자(bounding box) 정보를 학습합니다. 이 가우시안 커널은 주석화된 3D 경계 상자에서 자동으로 생성되어 림프절의 해부학적 정보를 보강하며, 이를 통해 효과적인 학습을 지원합니다.

- **Performance Highlights**: 비교 분석 결과, SDF-Net은 복잡한 림프절 탐지 문제를 해결하는 데 있어 유망한 성능을 나타냅니다. 다양한 형태와 크기를 가진 림프절 탐지의 민감도 및 정확성을 향상시킴으로써, 주위의 연조직과의 구분이 가능하도록 기능합니다.



### G3PT: Unleash the power of Autoregressive Modeling in 3D Generation via Cross-scale Querying Transformer (https://arxiv.org/abs/2409.06322)
Comments:
          9 pages, 5 figures

- **What's New**: G3PT는 기존의 정형화된 3D 데이터 방식에서 벗어나 순서가 없는 3D 데이터를 생성하는 크로스 스케일 오토리그레시브(Autoregressive) 모델을 제안합니다. 이 모델은 포인트 기반 3D 데이터를 다양한 세부 수준으로 매핑하여 자연스럽게 순차적 관계를 수립합니다.

- **Technical Details**: G3PT는 크로스 스케일 쿼리 트랜스포머(Cross-scale Querying Transformer, CQT)를 기반으로 하여 전역적으로 다른 세부 스케일의 토큰을 연결합니다. 이 방식은 정해진 시퀀스를 요구하지 않으며, 고해상도 포인트 클라우드를 잠재적 특성 맵으로 인코딩한 후, 점들을 쿼리하여 3D 점유 그리드로 디코딩합니다.

- **Performance Highlights**: G3PT는 기존의 3D 생성 방법에 비해 뛰어난 생성 품질을 보여주며, 3D 생성에서 처음으로 구분되는 스케일링 법칙(scaling-law) 행동을 나타냅니다. 다양한 조건 구조를 지원하며, G3PT는 시험을 통해 이전의 LRM 기반 및 확산(diffusion)기반 방법들보다 우수한 성능을 입증했습니다.



### Seam Carving as Feature Pooling in CNN (https://arxiv.org/abs/2409.06311)
- **What's New**: 본 논문은 이미지 분류 작업에서 Convolutional Neural Networks (CNNs) 내 feature pooling 기법으로 seam carving의 가능성을 조사합니다. 전통적인 max pooling 층을 seam carving 연산으로 대체하는 방안을 제안합니다.

- **Technical Details**: seam carving은 이미지의 가장 중요한 특성을 유지하면서 이미지의 크기를 조정하는 첨단 콘텐츠 인식 이미지 리사이징 기술입니다. 이 알고리즘은 저 에너지 경로를 식별하여 제거하거나 삽입합니다. 본 연구에서는 max pooling을 seam carving으로 대체한 CNN 아키텍처를 제안합니다.

- **Performance Highlights**: Caltech-UCSD Birds 200-2011 데이터셋에 대한 실험 결과, seam carving 기반 CNN이 max pooling을 사용하는 모델보다 정확도, 정밀도, 재현율 및 F1-score 등 다양한 성능 메트릭에서 뛰어난 성능을 보였습니다. 또한, seam carving 기법을 사용한 모델은 훈련 및 검증 데이터 세트에서 보다 일관되고 안정적인 손실 감소를 시현하였습니다.



### PPMamba: A Pyramid Pooling Local Auxiliary SSM-Based Model for Remote Sensing Image Semantic Segmentation (https://arxiv.org/abs/2409.06309)
- **What's New**: 본 논문에서는 기존의 State Space Model(SSM)-기반 방식보다 CJNN 및 Mamba를 통합한 새로운 네트워크인 Pyramid Pooling Mamba (PPMamba)를 제안합니다. 이 네트워크는 원거리 의존성을 효과적으로 캡처함과 동시에 지역적인 의미 정보를 보존하는 데 어려움을 겪는 문제를 해결합니다.

- **Technical Details**: PPMamba의 핵심 구조인 Pyramid Pooling-State Space Model (PP-SSM) 블록은 다중 방향에서 피처 맵을 선택적으로 스캔하여 다양한 피처 정보를 캡처하는 다방향 상태 공간 모델(OSS)을 통합합니다. 이를 통해 로컬 보조 메커니즘과 결합된 다중 스케일에서 피처를 추출할 수 있는 피라미드 형태의 합성곱 가지를 포함하여 구조화됩니다.

- **Performance Highlights**: ISPRS Vaihingen과 LoveDA Urban 데이터셋을 이용한 방대한 실험 결과, PPMamba는 최신 모델들과 비교했을 때 경쟁력 있는 성능을 달성했습니다. PPMamba는 지역 세부 정보를 효과적으로 처리하면서도 글로벌 컨텍스트 모델링을 통합하여 RS 이미지 분할의 독창적인 과제를 해결할 수 있는 잠재력을 보여줍니다.



### High-Performance Few-Shot Segmentation with Foundation Models: An Empirical Study (https://arxiv.org/abs/2409.06305)
Comments:
          under review

- **What's New**: 본 논문에서는 기존의 few-shot segmentation (FSS) 방법들의 한계를 극복하고, 재훈련된 모델들이 아닌, foundation 모델들로부터의 암묵적인 지식을 활용하는 새로운 접근 방식을 제안합니다. 이를 통해 coarse correspondence를 생성하고, lightweight decoder를 통해 정밀한 세그멘테이션을 위한 refinement를 수행합니다.

- **Technical Details**: 제안된 방법에서는 다양한 foundation 모델(DINO, DINOv2, MAE, CLIP 등)에서 암묵적인 지식을 추출하여 coarse correspondence를 구축합니다. 그 후 이 correspondence를 lightweight decoder를 통해 정제하여 세밀한 세그멘테이션을 실행합니다. 실험적으로 DINOv2와 DFN의 조합이 COCO-20i에서 기존 최첨단 방법보다 17.5% 성능 향상을 달성했습니다.

- **Performance Highlights**: 제안된 방법은 PASCAL-5i 및 COCO-20i 데이터셋에서 mask FSS 설정과 class-aware mask FSS 설정 모두에서 새로운 최첨단 성능을 기록했습니다. DINOv2와 DFN 조합은 COCO-20i에서 mIoU 기준으로 17.5% 성능이 개선되었습니다.



### An Attribute-Enriched Dataset and Auto-Annotated Pipeline for Open Detection (https://arxiv.org/abs/2409.06300)
- **What's New**: Objects365-Attr 데이터셋의 소개로 기존 OVD(object vocabulary detection) 및 REC(referring expression comprehension) 데이터셋의 한계를 극복하며, 다양하고 상세한 속성(attribute) 기술이 포함된 대규모 데이터셋을 개발하였습니다.

- **Technical Details**: Objects365-Attr 데이터셋은 5.6백만 개의 객체 속성 설명을 포함하며, 기존 Objects365 데이터셋을 기반으로 색상, 재료, 상태, 질감 및 톤과 같은 속성 주석을 강화하였습니다. 본 연구에서는 YOLO-World를 활용하여 다양한 규모에서의 성능을 평가하였습니다.

- **Performance Highlights**: YOLO-World의 객체 감지 성능이 속성 데이터를 효과적으로 활용할 수 있는 능력에 크게 영향을 받는다는 것을 보여주었습니다. 데이터셋을 기존의 OVD 및 REC 모델의 pre-training 과정에 통합하면 성능이 눈에 띄게 향상된다는 결과를 도출하였습니다.



### Enhancing Long Video Understanding via Hierarchical Event-Based Memory (https://arxiv.org/abs/2409.06299)
- **What's New**: 이번 논문은 장기 비디오 이해를 위한 새로운 계층적 이벤트 기반 메모리 강화 LLM(HEM-LLM)을 제시합니다. 기존 모델들이 비디오의 다양한 의미 정보를 압축하여 사용했던 것과 달리, HEM-LLM은 각 사건을 개별적으로 처리하여 정보 중복을 피합니다.

- **Technical Details**: HEM-LLM은 적응형 시퀀스 세분화 스킴을 통해 긴 비디오 내 여러 사건을 분리합니다. 또한 개별 사건에 대한 메모리 모델링을 수행하며, 사건 간의 장기 의존성을 강화하기 위해 이전 사건 정보를 압축하여 주입합니다. 이로써 비디오의 장기적 의미 이해를 높입니다.

- **Performance Highlights**: 다양한 비디오 이해 작업(비디오 질문 답변, 비디오 캡셔닝 등)에서 HEM-LLM은 아홉 개의 벤치마크 데이터 세트에서 최첨단 성능을 기록했습니다. 이는 기존 모델보다 긴 비디오의 내용 이해에서 월등한 능력을 보여줍니다.



### EntAugment: Entropy-Driven Adaptive Data Augmentation Framework for Image Classification (https://arxiv.org/abs/2409.06290)
Comments:
          Accepted by ECCV 2024

- **What's New**: 이 논문에서는 기존의 데이터 증강 방법이 직면한 문제를 해결하기 위해, EntAugment라는 새로운 데이터 증강 프레임워크를 제안합니다. EntAugment는 훈련 중 각 샘플의 증강 강도를 동적으로 조정하여, 모델의 학습 진행 상태에 따라 최적의 증강 정도를 결정합니다.

- **Technical Details**: EntAugment는 훈련 샘플의 복잡성과 모델의 진화하는 상태를 기반으로 하여 증강 강도를 설정합니다. 주요 기술적 요소로는 softmax 함수를 사용하여 모델 출력의 확률 분포에서 유도된 정보 엔트로피를 활용합니다. 이를 통해 각 샘플의 복잡성을 측정하고, EntLoss라는 엔트로피 정규화 항을 도입하여 Cross-Entropy (CE) 손실을 보완합니다.

- **Performance Highlights**: 다양한 이미지 분류 작업 및 네트워크 아키텍처에 대해 광범위한 실험을 수행한 결과, 제안된 방법이 기존의 최첨단 데이터 증강 방법(SOTA)보다 테스트 세트 성능에서 우수한 성능을 보이는 것으로 나타났습니다. 특히, 추가적인 모델이나 눈에 띄는 계산 비용 없이 효과성과 효율성을 강조합니다.



### Context Enhancement with Reconstruction as Sequence for Unified Unsupervised Anomaly Detection (https://arxiv.org/abs/2409.06285)
- **What's New**: 본 논문은 단일 모델로 다양한 이상 감지(anomaly detection)를 수행하는 n-class-one-model 설정에 대한 새로운 접근 방식인 Reconstruction as Sequence (RAS) 방법을 제안합니다. 이 방법은 특징 재구성(feature reconstruction) 과정에서 문맥 인식(contextual awareness)을 개선하여 이상 감지 성능을 향상시키고자 합니다.

- **Technical Details**: RAS 방법은 트랜스포머(transformer) 기술을 기반으로 하며, RASFormer 블록을 통합하여 서로 다른 이미지 영역 간의 공간 관계(spatial relationships) 및 재구성 과정 전반에 걸쳐 순차적 종속성(sequential dependencies)을 포착할 수 있도록 설계되었습니다. 이 방법은 특징 재구성 중 문맥적 일치를 효과적으로 증가시켜 우수한 성능을 달성합니다.

- **Performance Highlights**: 실험 결과, RAS 방법은 기존 방법들보다 현저히 우수한 성능을 나타내며 여러 벤치마크 데이터셋에서 최첨단 성능(state-of-the-art performance)을 기록했습니다. 이는 RAS 방법이 제안하는 새로운 접근이 효과적임을 잘 보여줍니다.



### Mahalanobis k-NN: A Statistical Lens for Robust Point-Cloud Registrations (https://arxiv.org/abs/2409.06267)
- **What's New**: 본 논문에서는 Mahalanobis k-NN을 소개하며, 학습 기반의 포인트 클라우드 등록에서 다양한 밀도의 포인트 클라우드에 대한 특징 일치 문제를 해결하는 통계적 기법을 제시합니다.

- **Technical Details**: Mahalanobis k-NN은 지역 이웃의 분포와 표면 기하를 캡처하는 성질을 채택하여 포인트 클라우드 등록의 새로운 방법론을 제공합니다. Deep Closest Point (DCP)와 Deep Universal Manifold Embedding (DeepUME) 두 가지 방법론에 통합되어 점진적 개선을 이루어냅니다.

- **Performance Highlights**: ModelNet40 및 ScanObjectNN 데이터셋에서 포인트 클라우드 몇 샷 분류 작업을 진행한 결과, Mahalanobis 방식이 도입됨으로써 평균 정확도가 약 20% 향상되었습니다. 코드와 데이터는 공개 доступ이 되어, 재현 가능성과 미래 연구에 기여할 것입니다.



### ALSS-YOLO: An Adaptive Lightweight Channel Split and Shuffling Network for TIR Wildlife Detection in UAV Imagery (https://arxiv.org/abs/2409.06259)
- **What's New**: 이번 연구에서는 열 적외선(TIR) 드론 이미지를 위한 ALSS-YOLO라는 경량화된 효과적인 감지기를 개발하였습니다. 새로운 Adaptive Lightweight Channel Split and Shuffling (ALSS) 모듈과 Lightweight Coordinate Attention (LCA) 모듈을 도입하여 흐릿한 작은 표적의 감지 능력을 향상시키고, 검출 정확도를 높이는 것을 목표로 했습니다.

- **Technical Details**: ALSS-YOLO는 ALSS 모듈을 통해 적응형 채널 분할 전략을 사용하여 특징 추출과 정보 교환을 최적화하며, LCA 모듈은 적응형 풀링과 그룹화 합성을 통해 차원 간 특징 정보를 통합하여 검출 정밀도와 강인성을 유지합니다. 또한 FineSIOU 손실 함수는 작은 물체와의 관련 손실 값을 강조하여 위치 정확성을 개선합니다.

- **Performance Highlights**: BIRDSAI 데이터셋에서 ALSS-YOLO(145.2만 파라미터)는 YOLOv8-n'(179.5만 파라미터)보다 1.7% 높은 mAP0.50을 달성했으며, ALSS-YOLO-s(222.6만 파라미터) 및 ALSS-YOLO-m(292.4만 파라미터) 모델이 다른 경량 객체 감지기보다 mAP 점수와 파라미터 효율성에서 유의미한 개선을 보였습니다.



### Test-Time Certifiable Self-Supervision to Bridge the Sim2Real Gap in Event-Based Satellite Pose Estimation (https://arxiv.org/abs/2409.06240)
Comments:
          This work has been accepted for publication at IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2024). Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 이 연구에서는 우주 기반 비전 위성 자세 추정에서 Sim2Real (simulation-to-real) 도메인 격차를 줄이기 위해 테스트 시간 자기 지도 학습(self-supervision) 새로운 접근 방식을 제안합니다. 이 방법은 'certifier 모듈'을 사용하여 추정된 위성 자세를 평가하고 교정합니다.

- **Technical Details**: 제안된 방법은 먼저 예측된 위성 자세의 고밀도 포인트 클라우드를 수집하고, 이 데이터를 이벤트(event) 데이터와 정렬하여 자세 추정을 교정합니다. 'certifier'는 수정된 자세의 정확성을 검증하고, 인증된 테스트 입력만이 백프롭(backpropagate)되어 예측된 랜드마크를 정제합니다. 더욱이, 이 방법은 기존의 자세 추정 방법들보다 효율적이며, 자신이 중심이 되는 학습 방식으로 공통의 RGB 데이터 및 실제 이벤트 데이터를 사용한 새로운 데이터셋을 개발했습니다.

- **Performance Highlights**: 이 연구의 성과는 기존의 테스트 시간 적응(test-time adaptation) 기법들보다 좋은 성능을 보이며, 위성 자세 추정의 Sim2Real 격차를 줄이는데 기여합니다.



### Recurrent Neural Networks for Still Images (https://arxiv.org/abs/2409.06235)
- **What's New**: 이번 연구에서는 정적인 이미지 처리에 Recurrent Neural Network (RNN)를 적용하는 새로운 방법을 소개합니다.

- **Technical Details**: 기존의 Convolutional Neural Networks (CNN) 및 Transformer와 달리 RNN은 픽셀을 시퀀스로 해석하여 정적 이미지를 처리할 수 있다는 점을 강조합니다. 또한, 두 차원 입력(2D inputs)에 맞춰 설계된 새로운 RNN 구조와 전통적인 BiDirectional RNN (BiRNN)보다 메모리 효율성이 높은 커스텀 BiRNN을 제안합니다. 실험은 Convolutional Recurrent Neural Networks (CRNN)에서 수행되었으며, 주로 Conv2D 레이어와 RNN 레이어로 구성됩니다.

- **Performance Highlights**: COCO 및 CIFAR100 데이터셋에 대한 실험 결과, 특히 작은 네트워크에서 더 나은 성능을 보여주었습니다.



### A Latent Implicit 3D Shape Model for Multiple Levels of Deta (https://arxiv.org/abs/2409.06231)
Comments:
          Published in GCPR 2024 proceedings

- **What's New**: 이 논문에서는 다중 세부 수준(multiple levels of detail)을 지원하고 매끄러운 표면을 보장하는 새로운 형태 모델링 접근법을 제안합니다. 고유한 지연 조건(latent conditioning)을 통해 여러 형태를 동시에 모델링하는 능력을 갖추고 있습니다.

- **Technical Details**: 본 연구는 다중 스케일(multiscale) 및 대역 제한(bandwidth-limited) 신경망 아키텍처를 적용하여 3D 형태에서 SDF(Signed Distance Function) 값을 추정합니다. 초기 레이어는 빠르게 추정된 SDF 값을 제공하며, 후속 레이어는 추가 세부 정보를 더합니다.

- **Performance Highlights**: 제안된 방법은 모든 수준(level of detail)에서 표면의 매끄러움을 크게 향상시키며, 기존의 최첨단 모델들과 동일하거나 더 높은 품질의 기하학적 결과를 보여줍니다. 또한, 효율적인 형태 편집 및 설계를 위한 인터랙티브한 지연 공간 탐색 기능도 제공합니다.



### MIP-GAF: A MLLM-annotated Benchmark for Most Important Person Localization and Group Context Understanding (https://arxiv.org/abs/2409.06224)
Comments:
          Accepted for publication at WACV 2025

- **What's New**: 이 논문은 대규모 'in-the-wild' 데이터셋을 주석 처리하여 이미지에서 '가장 중요한 사람 (Most Important Person, MIP)'의 인간 인식을 확인하는 문제를 다룹니다. 새롭게 제안된 방법론은 MLLM(다중 모달 대형 언어 모델)을 기반으로 합니다.

- **Technical Details**: MIP-GAF(그룹 AFfect)는 이미지 내에서 사람들 간의 상호작용의 추론 측면을 다루는 대규모 MLLM 기반 MIP 로컬라이제이션 벤치마크입니다. MIP 감지를 위해 4개의 학습 패러다임(제로 샷, 완전 감독, 반 감독, 자가 감독)에서 평가를 수행했습니다.

- **Performance Highlights**: 기존 데이터셋과 비교할 때 MIP-GAF에서 얻은 성능 저하가 ∼19.21 mAP 및 ∼24.21 mAP에 달하며, 이는 MIP 감지 알고리즘이 'in-the-wild' 상황에 더 강건해져야 함을 시사합니다. 이 데이터셋은 향후 연구에 귀중한 자산이 될 것입니다.



### CerviXpert: A Multi-Structural Convolutional Neural Network for Predicting Cervix Type and Cervical Cell Abnormalities (https://arxiv.org/abs/2409.06220)
- **What's New**: 이 논문에서는 CerviXpert라는 다구조(Structural) 합성곱 신경망(Convolutional Neural Network, CNN) 모델을 제안하여 자궁경부암 식별의 정확성과 효율성을 동시에 달성하고자 합니다. 특히, 기존의 고비용 딥러닝 모델들 대신 계산 복잡성을 줄이고 실용성을 고려한 접근법을 채택했습니다.

- **Technical Details**: CerviXpert는 세 가지 세포 이상 유형(정상, 비정상, 양성)을 분류하는 데 초점을 맞추었으며, 일반적인 딥러닝 기술인 ResNet50, VGG16, MobileNetV2 및 InceptionV3보다 효율적인 성능을 제공합니다. 연구진은 공개 데이터셋인 SiPaKMeD에서 방대한 실험을 수행하여 제안하는 방법의 유효성을 입증했습니다.

- **Performance Highlights**: CerviXpert는 자궁경부암 진단에 있어 높은 정확도를 달성하면서도 계산 비용이 저렴한 솔루션을 제공합니다. 이 모델은 실질적으로 자궁경부암 스크리닝 프로세스를 개선할 것으로 기대됩니다.



### DACAT: Dual-stream Adaptive Clip-aware Time Modeling for Robust Online Surgical Phase Recognition (https://arxiv.org/abs/2409.06217)
Comments:
          5 pages, 4 figures

- **What's New**: 본 논문에서는 DACAT라는 새로운 이중 흐름 모델을 제안하여 시간적 관계를 향상시키는 클립 인식 맥락 정보를 유연하게 학습함으로써 비디오 기반 외과 단계 인식을 개선합니다.

- **Technical Details**: DACAT는 두 가지 주요 스트림으로 구성됩니다: 1) Frame-wise Branch (FWB)로 프레임 별 특징을 처리하고, 2) Adaptive Clip-aware Branch (ACB)가 프레임 기반 피처 캐시에서 현재 프레임과 가장 관련이 깊은 클립을 읽어옵니다. 이 두 스트림은 cross-attention 기법을 통해 결합되어, 현재 프레임의 정보와 가장 적합한 과거 클립의 정보를 통합하여 시간 모델링을 개선합니다.

- **Performance Highlights**: DACAT는 세 개의 공개 데이터셋(Cholec80, M2CAI16, AutoLaparo)에서 기존 최첨단 방법들에 비해 각각 4.5%, 4.6%, 2.7%의 Jaccard 점수를 향상시켜 뛰어난 성능을 입증하였습니다.



### Towards Generalizable Scene Change Detection (https://arxiv.org/abs/2409.06214)
Comments:
          7 pages, 5 figures

- **What's New**: 본 논문에서는 기존의 SCD(장면 변화 감지) 방법의 한계점을 극복하기 위해 Generalizable Scene Change Detection Framework (GeSCF)를 제안합니다. GeSCF는 재훈련 없이도 unseen domains에 대한 일반화를 도모하며, 새로운 평가 지표와 데이터셋을 통해 SCD의 성능을 측정합니다.

- **Technical Details**: GeSCF는 foundation model의 지역적 의미를 활용하여 유사성 분포를 적응형으로 임계처리하여 초기 pseudo-change mask를 생성합니다. 이후 Segment Anything Model (SAM)의 클래스 비의존 마스크를 통해 pseudo-masks를 정제하고, 모든 설정에서 교환 가능성을 유지하여 완전한 시간 일관성을 보장합니다.

- **Performance Highlights**: 광범위한 실험을 통해 GeSCF는 다양한 도전적인 환경에서 뛰어난 성능을 보이며, 기존 최첨단 SCD 방법론을 능가하였습니다. 특히 unseen domains에서의 변화 감지를 0-shot으로 수행하며, 특정 데이터셋에서 미세 조정된 SCD 모델과 동등 이상의 성능을 달성하였습니다.



### INTRA: Interaction Relationship-aware Weakly Supervised Affordance Grounding (https://arxiv.org/abs/2409.06210)
- **What's New**: 이번 연구에서는 새로운 약한 감독(weakly supervised) 어포던스 그라운딩(affordance grounding) 방법인 INTRA(INTeraction Relationship-aware weakly supervised Affordance grounding)를 제안합니다. 본 방법은 객체의 상호작용을 나타내는 독특한 특징을 식별하는 데 중점을 두며, 이전의 방식과 달리 쌍(other) 이미지 없이도 동작할 수 있도록 설계되었습니다.

- **Technical Details**: INTRA는 대조 학습(contrastive learning)을 이용하여 외재적(exocentric) 이미지의 상호작용 관계를 반영한 어포던스 지도 생성(Text-conditioned affordance map generation)을 통해 기존의 문제를 표현 학습(representation learning)으로 재구성합니다. 또한, 비전-언어 모델(Vision-Language Model, VLM)의 임베딩을 활용하여 텍스트 동의어 증강(text synonym augmentation)을 통해 더 다양한 상황에 적응할 수 있는 능력을 강화합니다.

- **Performance Highlights**: INTRA는 AGD20K, IIT-AFF, CAD, UMD와 같은 다양한 데이터셋에서 이전 기법보다 뛰어난 성능을 보여주었으며, 새로운 상호작용 및 객체에 대한 어포던스 그라운딩에서도 우수한 도메인 확장성을 입증하였습니다.



### AgileIR: Memory-Efficient Group Shifted Windows Attention for Agile Image Restoration (https://arxiv.org/abs/2409.06206)
- **What's New**: AgileIR을 소개하며, Swin Transformer의 메모리 사용량을 줄이고 훈련 속도를 향상시키는 새로운 시도를 합니다.

- **Technical Details**: AgileIR은 Group Shifted Window Attention (GSWA) 메커니즘을 도입하여 Shift Window Multi-head Self Attention (SW-MSA)와 Window Multi-head Self Attention (W-MSA)의 구조를 단순화하고, back propagation에서 메모리 사용량을 줄입니다. 또한, shifted window masking과 shifted learnable biases를 통해 모델의 창 내 상호작용을 개선합니다.

- **Performance Highlights**: AgileIR은 SwinIR과 비교하여 Set5 평가 데이터셋에서 32.20 dB의 성능을 유지하면서도, 50% 이상의 메모리 절약 및 대규모 배치 크기 활용을 가능하게 합니다.



### RealisDance: Equip controllable character animation with realistic hands (https://arxiv.org/abs/2409.06202)
Comments:
          Technical Report

- **What's New**: RealisDance는 캐릭터 애니메이션 생성을 위한 새로운 접근 방식을 제안합니다. 이전 방법들이 의도한 포즈 조건에서의 부정확성으로 인해 발생했던 문제들을 해결합니다.

- **Technical Details**: RealisDance는 세 가지 유형의 포즈(DWPose, SMPL-CS, HaMeR)와 포즈 게이팅 모듈을 활용하여 동영상 생성의 오류를 줄입니다. 또한, 포즈 가이던스 네트워크에는 시간적 주의(temporal attention) 기법이 적용되어 비디오의 부드러움을 보장합니다.

- **Performance Highlights**: 정성적 실험에서 RealisDance는 다른 기존 방법들에 비해 손의 품질과 비디오의 부드러움에서 상당한 우위를 보여주었습니다.



### Deep kernel representations of latent space features for low-dose PET-MR imaging robust to variable dose reduction (https://arxiv.org/abs/2409.06198)
Comments:
          19 pages, 15 figures, 4 tables, Submitted to IEEE Transactions on Medical Imaging

- **What's New**: 이번 연구는 저선량 방사선 치료(로우-도즈 PET) 이미지를 복원하는 새로운 딥러닝 기반 방법을 제안하며, 이는 이전의 훈련 데이터와 일치하지 않는 새로운 저선량 감소 인자를 크게 처리할 수 있는 능력을 갖추고 있습니다.

- **Technical Details**: 제안된 방법은 깊은 잠재 공간 딥러닝(Deep Latent Space) 기능을 로버스트 커널 표현(Robust Kernel Representation)을 사용하여 모델링하는 데 중점을 두며, 이는 저선량 치료의 분포에 대한 성능을 개선하는 데 기여합니다. 정보 제약(Information Constraints)을 적용하여 훈련 대 데이터와 일반화 가능성을 조정하며, MR(Magnetic Resonance)과 PET 이미지를 모두 활용한 경우에 흥미롭고 혁신적인 접근법입니다.

- **Performance Highlights**: F18-FDG 및 F18-FDOPA 뇌 이미지를 사용한 테스트에서 기존의 딥러닝 방법과 비교하여 현저히 개선된 성능을 보여 주며, 저선량에서의 기존 이미지 재구성 방법에 대한 강력한 대안을 제공합니다.



### UdeerLID+: Integrating LiDAR, Image, and Relative Depth with Semi-Supervised (https://arxiv.org/abs/2409.06197)
- **What's New**: 이 논문에서는 LiDAR 데이터, 시각적 이미지 및 이미지에서 파생된 상대 깊이 맵을 통합하여 도로 분할을 위한 혁신적인 방법인 UdeerLID+ 프레임워크를 제안합니다. 이 방법은 반지도 학습(semisupervised learning) 패러다임에서 개발되어 대규모 데이터셋의 부족 문제를 해결합니다.

- **Technical Details**: UdeerLID+는 Multi-source Encoder-Decoder 기반의 세그멘테이터와 Meta Pseudo Labels를 사용하는 두 개의 주요 구성 요소로 구성되어 있습니다. 이 프레임워크는 LiDAR, 이미지 및 깊이 데이터를 융합하여 도로 탐지를 향상시킵니다. 또한 두 단계의 적응 프로세스를 통해 LiDAR 정보를 시각적 도메인으로 통합합니다.

- **Performance Highlights**: KITTI 데이터셋에서의 실험 결과, UdeerLID+는 다양한 도시 조건에서 도로 탐지의 정확성을 크게 향상시켜 기존의 최첨단 방법들을 초월하는 성능을 입증했습니다.



### MyGo: Consistent and Controllable Multi-View Driving Video Generation with Camera Contro (https://arxiv.org/abs/2409.06189)
- **What's New**: MyGo는 자율주행 모델을 위한 고품질 드라이빙 비디오 생성을 위한 새로운 프레임워크로, onboard 카메라의 움직임을 조건으로 사용하여 카메라 제어 가능성을 향상시킵니다.

- **Technical Details**: MyGo는 pre-trained 비디오 diffusion 모델에 카메라 매개변수를 주입하기 위해 추가적인 플러그인 모듈을 사용하며, 공간-시간 일관성을 높이기 위해 epipolar 제약과 이웃 뷰 정보를 활용합니다. 이 방법은 SVD U-Net 구조에 카메라 제어 모듈을 도입합니다.

- **Performance Highlights**: MyGo는 nuScenes 데이터셋에서 최신 비디오 합성 성능을 달성했으며, RealEstate10K 데이터셋에서는 최고의 카메라 제어 정확성을 기록했습니다.



### Bottleneck-based Encoder-decoder ARchitecture (BEAR) for Learning Unbiased Consumer-to-Consumer Image Representations (https://arxiv.org/abs/2409.06187)
Comments:
          2022 LXAI Workshop at the 39th International Conference on Machine Learning (ICML), Baltimore, Maryland

- **What's New**: 본 논문은 신규 이미지 피처 추출 메커니즘을 제안하며, 잔여 연결(residual connections)을 활용하여 오토인코더(autoencoder) 구성에서 지각적 이미지 정보를 인코딩합니다. 이는 범죄 활동과 관련된 C2C(Consumer-to-Consumer) 온라인 플랫폼에서의 문제 해결을 목표로 합니다.

- **Technical Details**: 제안된 아키텍처는 잔여 연결을 포함한 경량 이미지 피처 추출 메커니즘으로 구성되어 있으며, 이미지 재구성 오류 기준을 최소화하여 저차원 이미지 표현을 학습하는 컨볼루션 오토인코더(convolutional autoencoder) 형태입니다. 이 모델은 현대 머신 러닝 기법을 사용하여 적은 매개변수로 대규모 저차원 표현을 학습할 수 있습니다.

- **Performance Highlights**: 초기 결과는 제안된 아키텍처가 다른 이미지 데이터셋과 함께 풍부한 표현 공간을 학습할 수 있음을 시사하며, 이는 C2C 시장 관련 트래픽킹 탐지 작업에서 필요한 개인 식별 정보를 숨기면서도 주요 정보를 유지하는 데 효과적입니다.



### EDADepth: Enhanced Data Augmentation for Monocular Depth Estimation (https://arxiv.org/abs/2409.06183)
- **What's New**: 본 연구에서는 추가 학습 데이터 없이 단안 깊이를 추정하기 위한 새로운 방법인 EDADepth를 제안합니다. 이 방법은 Swin2SR 모델을 이용하여 입력 이미지의 질을 향상시키고, BEiT 사전 학습된 시맨틱 세분화 모델을 통해 텍스트 임베딩을 추출합니다. 또한, BLIP-2 토크나이저를 도입하여 텍스트 임베딩에서 토큰을 생성합니다.

- **Technical Details**: EDADepth는 데이터 증강(data augmentation) 기반의 모델로, Swin2SR로 향상된 입력 이미지를 사용하여 시맨틱 컨텍스트를 추출하고, BEiT 모델을 통해 텍스트 임베딩을 생성합니다. 기존의 저화질 NYU-Depth V2 데이터셋을 사용하여, 우리의 접근 방식이 MDE (Monocular Depth Estimation) 과정에서 효과적으로 기능함을 입증합니다.

- **Performance Highlights**: 우리는 NYUv2 및 KITTI 데이터셋에서 {	extdelta}3 메트릭에 대한 최신 성능(SOTA)을 달성하였으며, RMSE 및 REL 메트릭에서도 SOTA 모델에 버금가는 성능을 기록하였습니다. 또한, 추정된 깊이의 시각화 품질에서 SOTA 확산 기반 모델들에 비해 개선된 결과를 보여줍니다.



### Loss Distillation via Gradient Matching for Point Cloud Completion with Weighted Chamfer Distanc (https://arxiv.org/abs/2409.06171)
Comments:
          10 pages, 7 figures, 7 tables, this paper was accepted to IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2024

- **What's New**: 이 논문은 3D 포인트 클라우드(completion) 작업에서 HyperCD를 기반으로 한 새로운 가중치 적용 Chamfer Distance(loss function)를 제안하여, 데이터 관련 파라미터 조정 없이도 높은 성능을 발휘할 수 있는 방법을 모색하고 있습니다.

- **Technical Details**: 제안된 방법은 'Loss Distillation via Gradient Matching'이라는 검색 기법을 통해 HyperCD의 학습 행동을 모방하여 가중치가 적용된 Chamfer Distance(weighted CD) loss 함수의 후보를 찾습니다. 새로운 이기종 최적화(bilevel optimization) 공식을 통해 backbone 신경망을 가중치 CD loss로 훈련합니다.

- **Performance Highlights**: Landau weighted CD, 또는 Landau CD는 HyperCD보다 더 나은 성능을 보여주며, 여러 벤치마크 데이터셋에서 최신 기술(state-of-the-art)을 기록하고 있습니다.



### Revisiting Prompt Pretraining of Vision-Language Models (https://arxiv.org/abs/2409.06166)
- **What's New**: 본 논문에서는 Prompt Pretraining의 한계를 재조명하고, 모델의 적합성과 일반화 능력을 향상시키기 위한 새로운 프레임워크인 Revisiting Prompt Pretraining (RPP)을 제안합니다. RPP는 prompt 구조와 supervision 측면에서 두 가지 문제해결 접근법을 사용합니다.

- **Technical Details**: RPP는 공유 learnable prompt token에서 파생된 query, key 및 value 벡터의 제약을 깨고, 개별적으로 학습 가능한 쿼리, 키, 값 프롬프트를 도입하여 파라미터 다양성을 높입니다. 또한, Contrastive Language Image Pretraining (CLIP) 교사 모델에서 추출된 soft labels를 활용하여 기본적인 일반화 가능성을 강화합니다.

- **Performance Highlights**: RPP는 다양한 시각 인식 태스크에서 SOTA 성능을 달성하며, 예를 들어 ImageNet-21K 검증 정확도를 POMP보다 0.9% 향상시키고, 14개 데이터셋에서 평균 0.43%의 Zero-shot 일반화 향상을 보였습니다.



### UniLearn: Enhancing Dynamic Facial Expression Recognition through Unified Pre-Training and Fine-Tuning on Images and Videos (https://arxiv.org/abs/2409.06154)
- **What's New**: 이번 연구에서는 Dynamic Facial Expression Recognition (DFER) 성능을 향상시키기 위해 static facial expression recognition (SFER) 데이터를 통합한 새로운 학습 패러다임인 UniLearn을 소개합니다. 이 접근 방식은 비디오 데이터와 정적인 이미지 데이터를 동시에 활용하여 DFER 성능을 더욱 높입니다. 또한, Mixture of Adapter Experts (MoAE) 모듈을 도입하여 SFER과 DFER 간의 부정적 전이를 예방합니다.

- **Technical Details**: UniLearn은 ViT (Vision Transformer)를 활용하여 static 이미지와 비디오 데이터를 통해 dual-modal self-supervised pre-training을 실시합니다. 이후, 다중 작업 학습(multi-task learning) 설정을 통해 FER 이미지 및 비디오 데이터셋에서 공동으로 세밀하게 조정(fine-tuning)됩니다. MoAE 모듈은 ViT 층에 통합되어, task-specific 지식을 포착하고 task-agnostic 지식에 집중할 수 있도록 돕습니다.

- **Performance Highlights**: UniLearn은 FERV39K, MAFW, DFEW 벤치마크에서 각각 53.65%, 58.44%, 76.68%의 가중 평균 재현율(weighted average recall, WAR)을 기록하며 최첨단 성능을 달성했습니다. 기존의 VideoMAE와 비교하여, FERV39K, MAFW, DFEW 데이터셋에서 각각 1.26%, 4.93%, 2.08%의 높은 성과를 보였습니다.



### DECOLLAGE: 3D Detailization by Controllable, Localized, and Learned Geometry Enhancemen (https://arxiv.org/abs/2409.06129)
Comments:
          ECCV 2024 (poster). Code: this https URL

- **What's New**: 본 논문에서는 3D 모델링 기법을 제안하여 사용자들이 기계 학습을 활용하여 3D 형상을 세밀화할 수 있도록 합니다. 사용자는 대략적인 중합 형상에서 원하는 스타일을 '페인팅'하여 다양한 지역에서 고해상도 기하학적 세부사항을 생성할 수 있습니다.

- **Technical Details**: 이 방법은 Pyramid GAN 위에 마스킹 인식(masking-aware)을 기반으로 하여, 구조적 손실(structural losses) 및 사전(priors)을 통해 원하는 대략적인 구조와 세밀한 특성을 모두 보존하는 기능을 가집니다. 이를 통해 NEURAL NETWORK가 고해상도로 업샘플링(up-sampling)을 진행하면서도 여러 스타일을 혼합(mixing) 사용할 수 있습니다.

- **Performance Highlights**: 광범위한 실험 결과, 제안 방법이 기존의 전역 세밀화(global detailization) 기술에 비하여 구조를 보존하면서도 고해상도 스타일화된 기하학 형태를 생성하는 능력이 우수하며, 독창적인 인터랙티브 창작(workflow) 및 응용 프로그램을 가능하게 합니다.



### SGC-VQGAN: Towards Complex Scene Representation via Semantic Guided Clustering Codebook (https://arxiv.org/abs/2409.06105)
- **What's New**: SGC-VQGAN을 도입하여 일관된 의미 학습(Consistent Semantic Learning)을 통해 코드북의 의미론을 강화함으로써 향상된 성능과 범용성을 제시합니다.

- **Technical Details**: Semantic Online Clustering 방식을 활용하여, 세그멘테이션 모델의 추정 결과를 사용하여 시공간적으로 일관된 의미 코드북을 구성합니다. 멀티 레벨(레벨) 기능을 통합하는 피라미드 기능 학습(Pyramid Feature Learning) 파이프라인을 통해 이미지 세부 정보와 의미를 동시에 캡처합니다.

- **Performance Highlights**: SGC-VQGAN은 재구성 품질 및 다양한 다운스트림(Downstream) 태스크에서 SOTA(State Of The Art) 성능을 달성하였으며, 추가 파라미터 학습 없이 직접적으로 다운스트림 작업에 적용할 수 있는 간편한 방식이 특징입니다.



### LSE-NeRF: Learning Sensor Modeling Errors for Deblured Neural Radiance Fields with RGB-Event Stereo (https://arxiv.org/abs/2409.06104)
- **What's New**: 본 논문에서는 빠른 카메라 움직임에도 불구하고 선명한 Neural Radiance Field (NeRF)를 재구성하는 방법을 제안합니다. 블러 아티팩트를 해결하기 위해 (블러) RGB 이미지와 이진 카메라 데이터를 활용하며, 새로운 이진 세팅에 대한 이벤트 카메라 데이터셋을 소개합니다.

- **Technical Details**: 우리의 방법은 RGB와 이진 데이터의 차이를 모델링하기 위해 감마 함수를 사용하여 센서 응답의 차이를 모델링하며, 다양한 카메라 하드웨어 기능으로 인해 발생할 수 있는 측정값 간 변화를 고려하기 위해 시간별 임베딩을 활용합니다. 이러한 방법은 NeRF 학습 과정과 함께 수행됩니다.

- **Performance Highlights**: 제안된 방법은 고해상도 이진 이벤트-RGB 데이터셋을 통해 평가되었으며, 기존 방법들보다 우수한 재구성을 제공합니다. 특히, 이전의 단일 카메라 상황에 비해 이진 세팅에서 더 높은 성능을 기록합니다.



### SVS-GAN: Leveraging GANs for Semantic Video Synthesis (https://arxiv.org/abs/2409.06074)
- **What's New**: 이 논문에서는 Semantic Video Synthesis (SVS)의 개념을 새롭게 제안하며, 이를 위한 SVS-GAN이라는 전용 프레임워크를 소개합니다. SVS-GAN은 고유한 아키텍처와 손실 함수를 특화하여 SVS에 맞게 설계되었습니다. 특히, SVS-GAN은 퍼셉션 응용 프로그램에서 비디오 생성의 새로운 가능성을 탐색합니다.

- **Technical Details**: SVS-GAN은 Triple-Pyramid Generator 아키텍처를 사용하고, U-Net 기반의 이미지 판별기를 포함하여 OASIS 손실을 적용합니다. 이 프레임워크는 주어진 시퀀스의 세멘틱 맵에 맞춰 감정적 일관성을 유지하면서 현실적인 비디오를 생성합니다. GPU의 제약을 고려하여 1024×512 해상도에서 단일 GPU로 효율적으로 작동하며, 각 프레임에 대해 40ms의 추론 시간을 기록합니다.

- **Performance Highlights**: SVS-GAN은 Cityscapes 및 KITTI-360 데이터셋에서 기존 최첨단 모델을 능가하는 성능을 보여주며, 생성된 비디오의 화질, 시간적 일관성 및 세멘틱 정렬 개선을 입증하였습니다. 이 연구는 자율 주행 시나리오에서의 비디오 생성을 위해 구체적으로 검증되었으며, 각 구성 요소의 영향을 분석하는 상세한 아블레이션 연구를 포함하고 있습니다.



### DiffusionPen: Towards Controlling the Style of Handwritten Text Generation (https://arxiv.org/abs/2409.06065)
- **What's New**: 이번 연구에서는 Latent Diffusion Model 기반의 Handwritten Text Generation (HTG) 방법인 DiffusionPen(DiffPen)을 소개합니다. DiffPen은 단 5개의 스타일 샘플을 기반으로 하여 사용자가 본 적이 없는 스타일과 단어들을 재현할 수 있는 능력을 갖추고 있습니다.

- **Technical Details**: 이 모델은 텍스트와 스타일을 조건으로 하여 고품질의 실제 손글씨를 생성합니다. 이를 위해 하이브리드 스타일 추출기를 활용하여 메트릭 학습(metric learning)과 분류(classification)를 결합하여 텍스트와 스타일 특성을 캡처합니다. 또한, 여러 스타일 혼합(multi-style mixtures) 및 노이즈 임베딩(noisy embeddings) 전략을 통해 데이터의 견고성과 다양성을 향상시킵니다.

- **Performance Highlights**: IAM 오프라인 손글씨 데이터베이스에 대한 광범위한 실험 결과를 통해, DiffPen은 기존 방법들보다 질적, 양적으로 우수한 성능을 보였으며, 생성된 추가 데이터로 인해 Handwriting Text Recognition (HTR) 시스템의 성능이 개선됨을 확인했습니다.



### Online 3D reconstruction and dense tracking in endoscopic videos (https://arxiv.org/abs/2409.06037)
- **What's New**: 이 연구에서는 내시경 비디오 데이터를 이용한 실시간 밀집(한) 3D 장면 재구성 및 추적을 위한 온라인 프레임워크를 소개합니다. 이 방법은 Gaussian splatting 기법을 활용하여 해부학적 구조의 동적인 변화를 효과적으로 모델링합니다.

- **Technical Details**: 제안된 방법은 정적 Gaussian 모델을 결합하여 비디오 시퀀스에서 표면 포인트를 밀집 추적합니다. 장면의 변형을 모델화하기 위해 소규모의 제어점을 사용하며, 이 제어점은 장면의 복잡성에 비례하여 배치됩니다. 최적화는 광학적, 기하학적, 물리적 제약을 바탕으로 온라인으로 이루어집니다.

- **Performance Highlights**: StereoMIS 데이터셋을 이용한 실험에서 본 연구의 접근법은 최신 추적 알고리즘을 초월하여 오프라인 3D 재구성 방법과 유사한 성능을 달성했습니다. 결과적으로 이는 수술 보조 시스템의 기능을 향상시키는데 기여합니다.



### Enhanced Generative Data Augmentation for Semantic Segmentation via Stronger Guidanc (https://arxiv.org/abs/2409.06002)
- **What's New**: 데이터 증강(data augmentation)은 라벨이 필요한 작업에 대해 훈련 데이터를 만드는 데 널리 사용되는 기법입니다. 본 연구에서는 Controllable Diffusion Model을 이용해 의미 분할(semantic segmentation) 작업을 위한 효과적인 데이터 증강 방법을 제안합니다. 이 방법은 효율적인 프롬프트 생성(class prompt appending)과 시각적 프라이어 조합(visual prior combination)을 포함하여 실제 이미지의 라벨링된 클래스에 대한 주의를 강화합니다.

- **Technical Details**: 제안하는 방법에는 세 가지 주요 구성 요소가 포함됩니다: (1) 텍스트 프롬프트 생성, (2) 시각적 프라이어 조합, (3) 제어 가능한 확산 생성(controllable diffusion generation). Class-Prompt Appending은 이미지를 통해 감지된 클래스와 생성된 캡션(caption)을 결합하여 프롬프트를 생성합니다. 이를 통해 생성된 결과는 제어 가능한 확산 모델에 공급되어 합성 이미지를 생성합니다. 또한, 클래스 균형(class balancing) 알고리즘을 사용하여 클래스 간의 고른 분포를 유지합니다.

- **Performance Highlights**: PASCAL VOC 데이터셋에서 평가한 결과, 제안하는 방법이 의미 분할 작업에서 합성 이미지를 생성하는 데 매우 효과적임을 확인했습니다. 이 접근법은 고급 이미지 생성 모델들이 단순 이미지 변형에 비해 더 다양한 내용과 구조를 가진 이미지를 생성할 수 있음을 보여줍니다.



### Advance and Refinement: The Evolution of UAV Detection and Classification Technologies (https://arxiv.org/abs/2409.05985)
Comments:
          19 pages, 5 figures

- **What's New**: 최근 UAV(무인 항공기) 탐지 및 분류 시스템의 발전을 자세히 분석한 리뷰 논문이다.

- **Technical Details**: 이 리뷰는 2020년부터 오늘날까지의 UAV 탐지 방법론을 다루며, 레이더(radar), 무선 주파수(radio frequency), 광학(optical), 음향(acoustic) 센서와 같은 다양한 탐지 기술을 포함한다. 또한, 고급 센서 융합(sensor fusion) 기술을 통한 통합에 중점을 두고 있다. UAV 탐지 및 분류를 위한 기본 기술의 정확성과 범위를 상세히 분석하였다.

- **Performance Highlights**: 인공지능(artificial intelligence) 및 기계 학습(machine learning)에서의 최신 혁신이 시스템의 정확성과 효율성을 향상시키는 데 미치는 영향을 설명하며, 앞으로의 UAV 탐지 기술 발전이 성능과 신뢰성을 더욱 향상시킬 것이라는 예측으로 마무리한다.



### Transformer-Enhanced Iterative Feedback Mechanism for Polyp Segmentation (https://arxiv.org/abs/2409.05875)
- **What's New**: FANetv2는 내시경 이미지에서 폴립(폴립) 세그멘테이션을 위해 설계된 새로운 인코더-디코더 네트워크입니다. 이 네트워크는 Otsu 임계값(Binarization)으로 생성된 초기 입력 마스크를 기반으로 하여, 피드백 어텐션 메커니즘을 통해 이진 세그멘테이션 마스크를 반복적으로 정제합니다.

- **Technical Details**: FANetv2는 변동성이 있는 내시경 이미지를 처리하고, 폴립의 수(하나 또는 여러 개) 및 크기(작은, 중간, 큰)에 대한 정보를 통합하여 세그멘테이션 정확도를 향상시키는 텍스트 기반의 어텐션 메커니즘을 구현합니다. 네트워크의 테스트 과정에서 동적 마스크 정제를 통해 입력 이미지에 대해 반복적으로 마스크를 업데이트하고, 이는 실시간 세그멘테이션 성능을 향상시킵니다.

- **Performance Highlights**: BKAI-IGH 및 CVC-ClinicDB 데이터셋에서의 평가 결과, FANetv2는 0.9186과 0.9481의 높은 Dice Similarity Coefficient (DSC) 및 각각 2.83과 3.19의 낮은 Hausdorff 거리(Hausdorff distance)를 기록하며, 12개의 최첨단(SOTA) 알고리즘보다 우수한 성능을 보였습니다.



### SpecGaussian with Latent Features: A High-quality Modeling of the View-dependent Appearance for 3D Gaussian Splatting (https://arxiv.org/abs/2409.05868)
Comments:
          9 pages,6 figures, 5 tables, ACM Multimedia 2024

- **What's New**: 본 논문에서는 3D Gaussian Splatting(3D-GS) 방법의 한계를 극복하기 위해 Lantent-SpecGS를 제안합니다. 이 방법은 각 3D Gaussian 내에 보편적인 잠재 신경 기술(latent neural descriptor)을 활용하여 더욱 효과적인 3D 특징 필드를 표현합니다. 이로써 변동 색상, 기하학적 정보, 그리고 각도에 의존하는 색상 처리의 향상을 달성합니다.

- **Technical Details**: Lantent-SpecGS 접근법은 두 개의 병렬 CNN을 설계하여 diffuse(color)와 specular(color)를 각각 디코딩합니다. 뷰 포인트(viewpoint)에 따라 색상을 합치는 마스크를 배우며, 이를 통해 최종 렌더링 이미지를 생성합니다. Cook-Torrance 모델을 채택하여 색상을 diffuse 색상과 view-dependent 색상으로 분해하고, view-mask 맵을 생성하여 추가적인 렌더링 정밀도를 확보합니다.

- **Performance Highlights**: 실험 결과는 Lantent-SpecGS가 기존 방법보다 더욱 정교한 specular reflection 모델링과 view-dependent 색상 합성을 수행함을 보여줍니다. Shiny Dataset을 포함한 여러 공개 데이터셋에서 뛰어난 성능을 보이며, 특히 복잡한 조명 조건에서의 렌더링 품질이 향상됩니다.



### A study on Deep Convolutional Neural Networks, Transfer Learning and Ensemble Model for Breast Cancer Detection (https://arxiv.org/abs/2409.06699)
- **What's New**: 이번 연구는 자궁암 탐지에서 원래의 CNN 아키텍처와 전이 학습 및 앙상블 모델을 포함한 D-CNN의 성능을 비교하고 있습니다. 기존의 CNN 아키텍처의 효율성과 정확성을 조사한 비교 연구는 드물었습니다.

- **Technical Details**: 연구에서는 SE-ResNet152, MobileNetV2, VGG19, ResNet18, InceptionV3, DenseNet-121 등 6개의 CNN 기반 딥러닝 아키텍처를 활용하여 유방암 탐지 능력을 평가했습니다. 앙상블 모델은 다양한 D-CNN의 조합을 통한 모델입니다.

- **Performance Highlights**: 앙상블 모델은 유방암 탐지 및 분류에서 99.94%의 높은 정확도를 기록했습니다. 그러나 전이 학습이 SE-ResNet152, MobileNetV2, VGG19, ResNet18, InceptionV3 및 DenseNet-121 모델의 정확도를 높이지 못한 부정적인 결과도 함께 보고되었습니다.



### A comprehensive study on Blood Cancer detection and classification using Convolutional Neural Network (https://arxiv.org/abs/2409.06689)
- **What's New**: 본 연구는 혈액 악성종양을 탐지하고 분류하기 위해 기존 CNN 아키텍처를 활용한 새로운 실험을 수행하였습니다. 특히, DIX라는 새로운 앙상블 모델(DenseNet201, InceptionV3, Xception)을 개발하여 높은 정확도를 달성했습니다.

- **Technical Details**: 연구는 세 가지 주요 실험을 통해 진행되었습니다. 첫 번째 실험에서는 원래의 CNN 네트워크 6개를 사용하였고, 두 번째 실험에서는 전이학습(transfer learning)을 적용했습니다. 세 번째 실험에서는 새롭게 개발한 DIX 모델을 통해 혈액암 탐지 및 분류를 시도하였습니다.

- **Performance Highlights**: DIX 모델은 99.12%의 정확도로 원래 모델과 전이학습 성능을 능가하는 결과를 나타냈습니다. 그러나 전이학습은 원래 CNN의 정확도를 증가시키지 못했다는 부정적인 결과도 함께 제시되었습니다.



### A study on deep feature extraction to detect and classify Acute Lymphoblastic Leukemia (ALL) (https://arxiv.org/abs/2409.06687)
- **What's New**: 이번 연구는 급성 림프모구 백혈병(Acute Lymphoblastic Leukaemia, ALL)의 진단을 위해 딥러닝(Deep Learning) 기술, 특히 컨볼루션 신경망(Convolutional Neural Networks, CNNs)의 활용을 조사하였습니다.

- **Technical Details**: 연구에서 사용된 CNN 모델로는 InceptionV3, ResNet101, VGG19, DenseNet121, MobileNetV2 등이 있으며, 혈액 스미어(blood smear) 이미지를 기반으로 특징을 추출합니다. 특징 선택(Feature Selection) 기법으로는 ANOVA, Recursive Feature Elimination (RFE), Random Forest, Lasso, 주성분 분석(Principal Component Analysis, PCA) 등이 사용되었습니다. 분류(Classification) 단계에서는 나이브 베이즈(Naïve Bayes), 랜덤 포레스트(Random Forest), 서포트 벡터 머신(Support Vector Machine, SVM), K-최근접 이웃(K-Nearest Neighbours, KNN) 기법이 사용되었습니다.

- **Performance Highlights**: ResNet101 모델이 87%의 정확도로 최고의 결과를 제공하며, DenseNet121과 VGG19가 뒤를 이었습니다. 연구는 CNN 기반 모델이 ALL 진단의 속도와 정확성을 높여 의료 전문의의 필요성을 줄일 수 있는 잠재력이 있다고 강조합니다.



### Constructing an Interpretable Deep Denoiser by Unrolling Graph Laplacian Regularizer (https://arxiv.org/abs/2409.06676)
- **What's New**: 이번 논문에서는 Plug-and-Play (PnP) 아키텍처를 통해 일반화된 그래프 기반 깊이 이미지 노이저(GDD)를 구축하는 프레임워크를 제안합니다. 그래프 라플라시안 정칙화기(GLR)를 사용하여 최대 사후 확률(MAP) 문제의 솔루션을 언롤링(unrolling)함으로써 GDD를 개발하였습니다.

- **Technical Details**: GDD는 그래프 신호 처리(GSP)와 관련된 최신 이론을 활용하며, 그래프 라플라시안 행렬 $	extbf{L}$을 초기화하기 위해 truncated Taylor Series Expansion (TSE)을 적용합니다. 그 후, 수치적 자원 소모를 줄이기 위해 공명 기울기(conjugate gradient, CG) 알고리즘을 언롤링 하여 피드포워드 네트워크(feed-forward network)를 생성합니다.

- **Performance Highlights**: 실험 결과, GDD는 경쟁사에 비해 이미지 노이징 성능에서 경쟁력 있는 결과를 보여주며, 훨씬 적은 매개변수(parameter)로 구현되었습니다. 또한, GDD는 공변량 이동(covariate shift)에 대해 더 나은 강인성을 보여줍니다.



### Interactive 3D Segmentation for Primary Gross Tumor Volume in Oropharyngeal Cancer (https://arxiv.org/abs/2409.06605)
- **What's New**: 이 연구에서는 두 단계의 인터랙티브 클릭 정제(2S-ICR) 프레임워크를 제안하여 구강인두암(OPC)에서 주요 종양 용적(GTVp) 세분화를 향상시킵니다. 2S-ICR 프레임워크는 PET-CT 스캔에서의 효과적인 GTVp 세분화를 보여줍니다.

- **Technical Details**: 이 연구에서는 인터랙티브 딥러닝 모델을 활용해 GTVp를 세분화하는 데 있어 기존의 Click-based 인터랙션 방식을 개선하는 새로운 2단계 프레임워크를 제안합니다. 이 모델은 2021 HECKTOR 데이터셋과 텍사스 대학교 MD 앤더슨 암 센터의 외부 데이터셋을 활용하여 평가되었습니다.

- **Performance Highlights**: 2S-ICR 프레임워크는 비상호작용 모드에서 평균 Dice 유사도 계수(DSC) 0.713을 기록하였으며, 상호작용 후에는 0.824로 상승했습니다. 이는 기존 방법들을 초월한 성능을 나타냅니다.



### Multi-scale Cycle Tracking in Dynamic Planar Graphs (https://arxiv.org/abs/2409.06476)
Comments:
          TopoInVis 2024, 11 pages

- **What's New**: 본 논문은 입자 간 상호 작용으로 구성된 미세한 입자들이 나타내는 2D 힘 네트워크에서 사이클을 분석하기 위한 중첩 추적 프레임워크를 제시합니다. 이러한 사이클을 다양한 스케일에서 이해하는 것은 시스템의 기계적 및 운동학적 성질에 중요한 기여를 합니다.

- **Technical Details**: 제안된 접근 방식은 힘 네트워크 내에서 사이클에 의해 경계 값으로 나누어진 2D 도메인에서 사이클 계층을 계산합니다. 이를 통해 우리는 merge tree를 기반으로 한 중첩 추적 그래프의 개념을 조정하고, 강도 값의 범위에 적합한 시간 종속 삼각 분할로 정의된 영역의 분해를 추적합니다. 특히, 가중치가 부여된 평면 그래프를 분석하기 위한 위상적 프레임워크를 이용하여 발생하는 루프를 추출합니다.

- **Performance Highlights**: 본 논문은 여러 실험에서의 사이클 추적을 시연하며, 각 실험에서 수집된 포토엘라스틱 디스크의 힘 네트워크를 활용하여 분해된 도메인에서의 구조적 변화를 평가합니다. 이 방법은 실험 데이터의 다양한 조건에서 일어나는 사이클의 변화를 시각화하고, 직관적인 색상 전략을 통해 추가 정보를 전달하는 데 도움을 줍니다.



### Continual Domain Incremental Learning for Privacy-aware Digital Pathology (https://arxiv.org/abs/2409.06455)
Comments:
          Accepted in MICCAI 2024

- **What's New**: 이 연구는 Generative Latent Replay-based Continual Learning (GLRCL) 접근 방식을 통해, 과거 데이터를 저장하지 않고도 연속적으로 학습할 수 있는 방법을 제안합니다. Gaussian Mixture Models (GMM)를 이용하여 이전 데이터의 분포를 캡처하고, 이를 새로운 데이터와 융합하여 Latent Replay를 수행합니다.

- **Technical Details**: GLRCL은 과거의 샘플을 저장하는 대신, GMM을 사용하여 이전 학습 도메인의 특성을 캡슐화합니다. 이러한 접근 방식은 새로운 도메인에서 과거 도메인에서 학습된 정보를 활용하여 무엇이 잊혀지는지를 방지합니다. 이 작업은 디지털 병리학 데이터에서의 다양한 도메인 변화(예: 염색이나 장기 변화)를 대응하기 위해 체계적으로 평가됩니다.

- **Performance Highlights**: 제안된 GLRCL 접근 방식은 인기 있는 buffer-free CL 방법들과 비교하여 성능이 현저히 향상되었으며, 대규모 버퍼를 필요로 하는 리허설 기반 CL 접근 방식과 유사한 성능을 보였습니다. 이는 데이터 프라이버시 문제를 해결하면서도 높은 정확도를 유지하는 데 기여합니다.



### Unrevealed Threats: A Comprehensive Study of the Adversarial Robustness of Underwater Image Enhancement Models (https://arxiv.org/abs/2409.06420)
- **What's New**: 본 논문은 UWIE(Underwater Image Enhancement) 모델의 적대적 공격(adversarial attack)에 대한 취약성을 최초로 종합적으로 다루었습니다. 여러 UWIE 모델들에 대한 공격 프로토콜을 제안하고, 특별히 수중 이미지의 특성을 반영한 두 가지 공격 방법인 Pixel Attack과 Color Shift Attack을 설계하였습니다.

- **Technical Details**:  UWIE 모델에 대한 적대적 공격을 통해 3개의 공통적인 수중 이미지 벤치마크 데이터셋을 사용하여 5개의 UWIE 모델의 적대적 강건성(adversarial robustness)을 평가하였습니다. 이 연구는 RGB 및 YUV 색 공간을 기반으로 한 두 가지 공격 방법을 제안하며, 이미지를 손상시키는 작은 변동이 UWIE 모델의 향상된 결과 생성에 미치는 영향을 분석하였습니다.

- **Performance Highlights**: 결과적으로, 5개 모델은 서로 다른 수준의 적대적 공격에 대한 취약성을 보였으며, 잘 설계된 작은 변동이 UWIE 모델의 결과 생성에 부정적인 영향을 미쳤습니다. 또한, 적대적 훈련(adversarial training)을 통해 UWIE 모델에 대한 공격 저항력을 성공적으로 강화하였습니다.



### Towards Robust Uncertainty-Aware Incomplete Multi-View Classification (https://arxiv.org/abs/2409.06270)
Comments:
          Ongoing work: 9 pages, 6 figures, 2 tables

- **What's New**: 본 논문에서는 불완전한 다중 뷰 분류(multi-view classification, MVC) 문제를 효과적으로 해결하기 위한 새로운 방법인 교차 점진적 학습 네트워크(Alternating Progressive Learning Network, APLN)를 제안합니다.

- **Technical Details**: APLN은 먼저 조잡한 보간(coarse imputation)을 수행하고, 이어서 데이터를 잠재 공간(latent space)으로 매핑하여 증거 분포를 점진적으로 학습합니다. 이 과정에서 불확실성을 고려하여 Dirichlet 분포를 모델링하고, 갈등을 보다 잘 처리하기 위해 갈등 인식 Dempster-Shafer 조합 규칙(conflict-aware Dempster-Shafer combination rule, DSCR)을 도입합니다.

- **Performance Highlights**: APLN은 기존 방법들보다 높은 성능을 보였으며, 특히 불확실성과 갈등이 높은 환경에서보다 신뢰할 수 있는 결정을 내리는 데 기여합니다.



### Denoising: A Powerful Building-Block for Imaging, Inverse Problems, and Machine Learning (https://arxiv.org/abs/2409.06219)
- **What's New**: 최근의 이미지 denoising 기술들은 이론적 한계에 가까운 성공을 거두었으나, 노이즈 제거 이상의 다양한 응용 분야에서의 이 중요성이 충분히 인정받지 못하고 있다는 점에서 이 논문은 그 격차를 해소하고자 한다.

- **Technical Details**: 이 논문에서는 denoiser의 구조와 바람직한 특성에 대한 포괄적인 관점을 제시한다. '아이디얼(ideal)' denoiser가 가져야 할 두 가지 주요 특성은 (1) 노이즈가 없을 때 입력을 그대로 재현하는 것, (2) 이상적인 denoiser는 특정 상태에서 보존적 벡터 필드를 형성한다는 것이다.

- **Performance Highlights**: 이 논문은 denoising 기술이 이미지 처리, 역문제 및 머신러닝과 같은 복잡한 작업의 필수적인 구성 요소로 발전하고 있음을 강조하며, 이에 따라 denoising의 중요성이 지속적으로 증가하고 있음을 보여준다.



### Draw an Audio: Leveraging Multi-Instruction for Video-to-Audio Synthesis (https://arxiv.org/abs/2409.06135)
Comments:
          14 pages, 11 figures

- **What's New**: 본 논문은 'Draw an Audio'라는 새로운 비디오-오디오(V2A) 합성 모델을 소개합니다. 이 모델은 드로잉 마스크(drawn masks)와 음량 신호(loudness signals)를 통해 여러 입력 명령어를 지원하면서, 콘텐츠 일관성(content consistency), 시간 일관성(temporal consistency), 그리고 음량 일관성(loudness consistency) 문제를 동시에 해결할 수 있습니다.

- **Technical Details**: 'Draw an Audio'는 Mask-Attention Module (MAM)과 Time-Loudness Module (TLM)을 통해 기능합니다. MAM은 마스크된 비디오 지시를 사용하여 모델이 관심 영역에 집중하도록 하고, TLM은 보조 음량 신호를 활용하여 생성된 오디오가 비디오와 시간 및 음량 차원에서 일치하도록 보장합니다. 또한, 'VGGSound-Caption'이라는 대규모 V2A 데이터셋을 확장하여 캡션 프롬프트(caption prompts)로 주석을 추가했습니다.

- **Performance Highlights**: 광범위한 실험 결과, 'Draw an Audio'는 두 개의 대규모 V2A 데이터셋에 대한 도전적인 벤치마크에서 최첨단 성능을 달성합니다. 이 연구는 Foley 디자이너의 요구를 보다 효과적으로 충족시키는 오디오 합성 프레임워크를 제공합니다.



### PaRCE: Probabilistic and Reconstruction-Based Competency Estimation for Safe Navigation Under Perception Uncertainty (https://arxiv.org/abs/2409.06111)
- **What's New**: 이 논문은 인식 기반 내비게이션 시스템의 안전한 운행을 위한 새로운 방법인 PaRCE(Probabilistic and Reconstruction-based Competency Estimation)를 개발하였습니다. 이 방법은 입력 이미지의 전반적인 친숙도 점수를 추정하고 특정 지역의 정보를 제공합니다.

- **Technical Details**: PaRCE 방법은 모델이 입력 이미지에 대한 예측의 불확실성을 인식하고, 이를 바탕으로 안전하고 효과적인 내비게이션을 수행할 수 있도록 하는데 초점을 맞춥니다. 이 시스템은 전반적인 친숙도 점수와 지역적 친숙도 맵을 생성하여 알맞은 경로를 계획하고 제어하는 데 활용됩니다.

- **Performance Highlights**: 본 연구에서 제안한 시스템은 기존 비교군에 비해 낯선 장애물과의 충돌을 현저히 줄였으며, 이러한 지역적 친숙도 정보는 효율적인 내비게이션을 가능하게 하였습니다.



### Analyzing Tumors by Synthesis (https://arxiv.org/abs/2409.06035)
Comments:
          Accepted as a chapter in the Springer Book: "Generative Machine Learning Models in Medical Image Computing."

- **What's New**: 이 논문에서는 인공지능(AI) 기반의 종양 감지 기술이 8천만 건 이상의 CT 스캔을 해석하는 데 도움을 줄 수 있는 가능성을 제시합니다. 특히 초기 단계의 종양에 대한 CT 스캔의 희소성으로 인해, 실 데이터를 통한 AI 개발이 어려운 문제를 해결하기 위해 종양 합성 기술을 다룹니다.

- **Technical Details**: 종양 합성 기술은 실제 종양 데이터를 생성하는 데 필요한 데이터의 부족, 주석의 어려움, 낮은 유병률 문제를 해결합니다. 이 연구는 실제 데이터와 합성 데이터에서의 AI 개발을 검토하고, 종양 합성을 위한 두 가지 주요 트렌드 즉, 모델링 기반 방법과 학습 기반 방법에 대해 설명합니다. Pixel2Cancer와 같은 모델링 기반 방법은 종양의 발전을 일반 규칙에 따라 모사하는 반면, DiffTumor와 같은 학습 기반 방법은 특정 기관의 일부 주석된 예시에서 학습하여 다른 기관에서 합성 종양을 생성합니다.

- **Performance Highlights**: 이 연구에서는 간, 췌장, 신장에 대한 사례 연구가 포함되어 있으며, 합성 종양으로 훈련된 AI가 실제 데이터로만 훈련된 AI와 동등하거나 더 나은 성능을 달성할 수 있음을 보여줍니다. 종양 합성 기술은 데이터 세트를 확장하고 AI의 신뢰성을 높이며, 종양 탐지 성능을 개선하고 환자 개인 정보를 보호하는 데 중요한 가능성을 지니고 있습니다.



### NESI: Shape Representation via Neural Explicit Surface Intersection (https://arxiv.org/abs/2409.06030)
- **What's New**: 새로운 접근 방식으로 NESI(Neural Surfaces with Intersections)를 제안하여, 3D 형태의 압축된 표현 방식을 개선하였습니다. 이는 이전의 implicit 또는 parametric representations과는 다른 학습된 대안을 제공합니다.

- **Technical Details**: NESI는 서로 다른 방향으로 정렬된 height-field로 경계가 형성된 half-space들을 볼륨 부울 교차( volumetric Boolean intersections )를 사용하여 결합합니다. 입력 형상은 Double Height-Field (DHF) Hull로 형성된 후, 내부의 표면 영역을 포착하기 위해 추가적인 localized height-fields (HFs)와 교차하여 정제됩니다. 최종적으로 두 개의 높이 필드로 타이트하게 경계를 설정하고, R^2의 부분 도메인에서 정의된 신경 함수로서 DHF hull과 local HFs를 효과적으로 압축합니다.

- **Performance Highlights**: 유사한 매개변수 수 또는 저장 용량에서 NESI는 최신 기술(state of the art)에 비해 약간 낮은 매개변수 수에서도 근사 오류를 현저히 줄입니다. 고품질의 압축 근사값을 제공하며, 다양한 처리 작업을 지원할 수 있는 장점이 있습니다.



### Pioneering Precision in Lumbar Spine MRI Segmentation with Advanced Deep Learning and Data Enhancemen (https://arxiv.org/abs/2409.06018)
- **What's New**: 본 연구는 허리 척추(segmentation of lumbar spine) 구조의 정확한 분할을 위한 진보된 심층 학습(deep learning) 접근법을 제시하며, 클래스 불균형(class imbalance) 및 데이터 전처리(data preprocessing)와 같은 주요 도전 과제를 해결하는 데 초점을 맞추었습니다.

- **Technical Details**: 이 연구는 Magnetic Resonance Imaging (MRI) 스캔을 이용하여 척추의 세 가지 중요한 클래스인 척추(vertebrae), 척추관(spinal canal), 및 추간판(intervertebral discs; IVDs)을 정확하게 표현하기 위해 세심하게 데이터 전처리 과정을 수행했습니다. 수정된 U-Net 모델은 리키 ReLU(leaky Rectified Linear Units) 및 Glorot uniform initializer를 포함하여 내재된 많은 문제들을 해결하고 훈련 과정의 안정성을 개선했습니다. 생성된 맞춤형 결합 손실 함수(custom combined loss function)는 클래스 불균형 문제를 효과적으로 해결하고, 세분화(segmentation) 정확도를 크게 향상시켰습니다.

- **Performance Highlights**: 모델의 성능 평가는 Mean Intersection over Union (Mean IoU), Dice coefficient, Average Surface Distance (ASD), Normalized Surface Distance (NSD), precision, recall, 및 F1 score와 같은 다양한 메트릭을 사용하여 수행되었습니다. 결과들은 기존 방법들보다 우수한 성능을 보였으며, 데이터 전처리 기술, 수정된 U-Net 아키텍처의 강건성 및 맞춤형 손실 함수의 효과를 강조합니다.



### Improved Visually Prompted Keyword Localisation in Real Low-Resource Settings (https://arxiv.org/abs/2409.06013)
- **What's New**: 이번 연구에서는 영어 외에도 실제 저자원 언어인 요루바어를 위한 Visually Prompted Keyword Localisation (VPKL) 기술을 처음으로 도입하였으며, 전사 없이 자동으로 쌍을 생성하는 few-shot learning 방법을 활용했습니다.

- **Technical Details**: VPKL은 주어진 이미지 쿼리를 사용하여 음성 발화에서 특정 키워드의 존재를 탐지하고 위치를 파악하는 두 단계로 구성됩니다. 연구에서는 LocAttNet 모델을 사용하며, 이 모델은 이미지와 음성의 쌍을 통해 시각적으로 기반한 음성 모델을 학습합니다. 또한, 포지티브 및 네거티브 예제를 자동으로 생성하기 위한 few-shot mining 기법을 사용하여 데이터 부족 문제를 해결하였습니다.

- **Performance Highlights**: 요루바어의 경우, few-shot mining 기법을 사용할 때 성능이 기존의 전사가 있는 데이터에 비해 더 큰 하락폭을 보였으며, 정밀도 등의 지표에서도 적당한 성과를 보였지만, 완벽한 쌍을 사용할 경우보다 평균 11% 성능 저하가 발생하는 것으로 나타났습니다.



### Enhancing Cross-Modality Synthesis: Subvolume Merging for MRI-to-CT Conversion (https://arxiv.org/abs/2409.05982)
- **What's New**: 본 논문에서는 자기 공명 영상 (MRI)으로부터 합성된 컴퓨터 단층 촬영 (sCT)에서 화질을 향상시키기 위한 새로운 Subvolume 합병 기법을 도입했습니다. 이 방법은 SwinUNETR 프레임워크를 활용하여, 겹치는 서브 볼륨을 최적화함으로써 stitching artifacts를 줄이고 평균 절대 오차 (MAE)를 개선했습니다.

- **Technical Details**: SwinUNETR 구조를 기반으로 한 본 연구에서는 32×96×96 크기의 서브 볼륨을 랜덤하게 선택하고, 각 서브 볼륨을 패치로 나눈 후, 2×2×2 사이즈의 패치 시퀀스를 처리합니다. 이러한 구조는 CNN 기반의 디코더와 결합되어 최종적으로 1×1×1 컨볼루션 계층을 통해 단일 채널의 합성 CT 이미지를 생성합니다.

- **Performance Highlights**: 논문에서 제안하는 새로운 서브 볼륨 합병 기법을 사용 시, MAE는 52.65 HU에서 47.75 HU로 감소하며, PSNR도 27.84에서 28로 증가함을 보여줍니다. 50%에서 70%의 겹침 비율을 설정하여 이미지 품질과 계산 효율성 간의 균형을 달성하였습니다.



### Towards Narrowing the Generalization Gap in Deep Boolean Networks (https://arxiv.org/abs/2409.05905)
- **What's New**: 최근의 심층 신경망에서의 크기와 복잡성 증가로 인해 계산 요구가 급격히 증가하고 있습니다. 본 연구에서는 논리 게이트를 사용하여 구축된 불리언 네트워크가 보다 효율적인 구현을 가능하게 할 수 있는 가능성을 탐색합니다. 특히, 심층 불리언 네트워크의 성능을 향상시키기 위한 새로운 전략을 제안합니다.

- **Technical Details**: 논문에서는 논리 건너뛰기 연결(logical skip connections)과 공간 보존 샘플링(spatiality preserving sampling)과 같은 혁신적인 방법을 도입했습니다. 이러한 방법들은 비전 과제에서 검증되어 기존 접근 방식보다 개선된 성능을 보여주었습니다. 또한, 1비트 논리 연산을 통해 높은 성능을 유지하면서 계산 비용을 최소화하는 가능성을 분석하였습니다.

- **Performance Highlights**: 제안된 깊은 불리언 네트워크는 기존의 다층 퍼셉트론보다 더 나은 성능을 보이며, 계산 복잡성과 필요한 파라미터 수를 줄이는 데 성공하였습니다. 이로 인해 불리언 네트워크가 효율적이고 고성능의 심층 학습 모델로서의 큰 잠재력을 지니고 있음을 시사합니다.



### Memory-Optimized Once-For-All Network (https://arxiv.org/abs/2409.05900)
- **What's New**: 이번 논문에서는 Deep Neural Network (DNN)를 리소스 제약이 있는 장치에서 최적화된 메모리 사용을 최대화하도록 설계된 메모리 최적화 OFA(MOOFA) 슈퍼넷을 소개합니다. 이 슈퍼넷은 다양한 구성에서 특성의 다양성을 향상시켜 DNN 배포를 개선하는 것을 목표로 합니다.

- **Technical Details**: MOOFA는 각 레이어의 메모리 소비를 분해하여 메모리 할당을 최적화하며, 레이어별로 더 다양한 특징을 추출할 수 있도록 설계되었습니다. 이를 통해 사용 가능한 메모리 한계 내에서 DNN 성능을 향상시킬 수 있습니다. OFA(supernet)에서 발생하는 비효율성을 주소하며, 일반화 능력을 향상시키기 위한 구조를 제공합니다.

- **Performance Highlights**: ImageNet 데이터셋을 기반으로 한 실험에서 MOOFA 슈퍼넷은 기존 OFA 슈퍼넷 대비 메모리 활용도와 모델 정확도에서 개선을 보였습니다. 특히 메모리 제약이 있는 환경에서 더 많은 특성을 효과적으로 추출하여 성능이 개선된 것으로 나타났습니다.



New uploads on arXiv(cs.AI)

### Superior Computer Chess with Model Predictive Control, Reinforcement Learning, and Rollou (https://arxiv.org/abs/2409.06477)
- **What's New**: 이 논문에서는 모델 기반 예측 제어(MPC), 롤아웃(rollout), 강화 학습(RL) 방법론을 컴퓨터 체스에 적용합니다. 새로운 이동 선택 아키텍처를 소개하며, 여기서 체스 엔진들이 구성 요소로 사용됩니다.

- **Technical Details**: 우리의 기본 아키텍처는 한 번의 미리보기 검색(one-move lookahead search)을 통해 이동을 선택하며, 중간 이동은 명목적 상대 엔진(nominal opponent engine)에 의해 생성되고, 결국 다른 체스 엔진에 의해 위치 평가(position evaluation)가 이루어집니다. 이론적으로, 우리의 방법론은 뉴턴 방법(Newton's method)의 초선형 수렴(superlinear convergence) 프레임워크에 의존합니다.

- **Performance Highlights**: MPC-MC 아키텍처는 Stockfish 등의 엔진을 기반으로 할 경우, 그 엔진보다 더욱 우수한 성능을 나타내며, 빠른 시간 제약에서 압도적인 승리를 거둡니다. 또한, 우리의 아키텍처는 체스뿐만 아니라, 장기, 장기, 체크, 바둑, 리버시 등과 같은 모든 2인 제로섬 게임에서도 적용될 수 있습니다.



### MAGDA: Multi-agent guideline-driven diagnostic assistanc (https://arxiv.org/abs/2409.06351)
- **What's New**: 이번 연구에서는 진단 지원을 위한 제로샷(Zero-shot) 방법론을 제안합니다. MAGDA(Multi-Agent Guideline-driven Diagnostic Assistance)라는 다중 에이전트 시스템을 모델로 하여 클리닉 가이드라인을 통해 진단을 지원하는 접근 방식을 제시합니다.

- **Technical Details**: MAGDA는 세 가지 에이전트를 포함하여 이미지 분석 및 진단을 수행합니다: (1) 스크리닝 에이전트(𝒮)는 CLIP 모델을 사용하여 이미지를 분석합니다; (2) 진단 에이전트(𝒟)는 이미지에서 얻은 정보를 바탕으로 진단을 내립니다; (3) 정제 에이전트(ℛ)는 진단의 질을 평가하여 최종 예측을 제공합니다. 이 시스템은 의료 이미지와 클리닉 가이드라인을 입력으로 받고 제로샷 진단을 수행합니다.

- **Performance Highlights**: CheXpert 및 ChestX-ray 14 Longtail 데이터셋에서 기존의 제로샷 방법보다 성능 향상을 보이며, 드문 질병에 대한 일반화 가능성도 입증하였습니다.



### Case Study: Leveraging GenAI to Build AI-based Surrogates and Regressors for Modeling Radio Frequency Heating in Fusion Energy Scienc (https://arxiv.org/abs/2409.06122)
- **What's New**: 본 연구는 Generative AI (GenAI)를 활용하여 핵융합 에너지 연구에서 시뮬레이션 모델을 위한 AI 서그겟(이하 아이 서그겟) 개발을 위한 구체적인 사례 연구를 제시합니다.

- **Technical Details**: 이 연구는 GenAI를 사용하여 모델 개발 과정의 모든 단계에서 지원을 제공하고, 탐색적 데이터 분석과 k-fold cross-validation을 통한 모델 최적화 등 다양한 모델 개발 단계의 접근 방식을 제안하고 초안을 생성하는 데 중점을 두었습니다.

- **Performance Highlights**: GenAI를 활용함으로써 이전 연구에 비해 개발 시간 단축과 모델 성능이 향상되었으며, 입력 모델 특징의 수를 줄이는 다양한 전략을 연구하고 적용하였습니다.



### MLLM-FL: Multimodal Large Language Model Assisted Federated Learning on Heterogeneous and Long-tailed Data (https://arxiv.org/abs/2409.06067)
- **What's New**: 새로운 논문에서는 MLLM-FL (Multimodal Large Language Model Assisted Federated Learning)이라는 연합 학습(frational learning) 프레임워크를 제안하여, 데이터 불균형과 이질성 문제를 해결하고자 합니다. 이 프레임워크는 MLLMs (Multimodal Large Language Models)을 활용하여, 대규모 오픈 소스 데이터를 효과적으로 활용하고, 서버 측에서의 계산 자원을 극대화합니다.

- **Technical Details**: MLLM-FL 프레임워크는 세 가지 주요 단계로 구성되어 있습니다: 1단계는 글로벌 멀티모달 프리트레이닝으로, 온라인에서 수집된 비라벨 데이터에 대한 설명을 생성하는 것입니다. 2단계는 연합 미세 조정(federated finetuning)으로, 클라이언트에서 로컬 데이터셋을 통해 훈련을 진행합니다. 3단계는 서버 측에서 MLLM의 감독 하에 글로벌 정렬(global alignment)을 수행하며, 이는 모델의 출력이 작업 특성에 맞게 조정되도록 합니다.

- **Performance Highlights**: 실험 결과 MLLM-FL은 다양한 데이터셋에서 기존의 최첨단 연합 학습 방법론보다 성능이 우수하며, 데이터 불균형과 클래스 분포 문제를 효과적으로 해결할 수 있음을 보여줍니다. 또한, 이 프레임워크는 개인 정보 보호를 강화하고 클라이언트 장비에서의 계산 부담을 크게 줄이는 장점을 갖추고 있습니다.



### Deep Generative Model for Mechanical System Configuration Design (https://arxiv.org/abs/2409.06016)
- **What's New**: 이번 연구는 기계 시스템 구성 설계에 대한 딥 생성 모델인 GearFormer를 제안하여 설계 요구 사항을 충족하는 최적의 구성 조합을 예측합니다. 이는 설계 복잡성을 비약적으로 줄이고, 시간 소모가 많은 기존 방법에 비해 효율성을 크게 향상시킵니다.

- **Technical Details**: 기존의 메타 휴리스틱 최적화 방법 대신 Transformer 기반의 모델을 사용하여 Gear 대상의 구성 요소 및 인터페이스 최적 조합을 생성하는 새로운 접근 방식을 보여줍니다. GearFormer는 문법, 부품 카탈로그, 물리적 시뮬레이터를 통해 생성된 합성 데이터셋을 학습합니다. 또한, 전통적인 탐색 방법인 진화 알고리즘 및 몬테 카를로 트리 탐색과 결합하여 성능을 향상시킵니다.

- **Performance Highlights**: GearFormer는 전통적인 탐색 방법보다 성능이 월등히 뛰어난 솔루션을 생성하며, 명시된 설계 요구 사항을 만족시킵니다. 시연 결과, 생성 시간 측면에서도 기존 방법보다 수 배 빠릅니다. 또한, GearFormer와 탐색 방법을 결합함으로써 더욱 개선된 품질의 솔루션을 도출하는 장점을 제공합니다.



### Hint-AD: Holistically Aligned Interpretability in End-to-End Autonomous Driving (https://arxiv.org/abs/2409.06702)
Comments:
          CoRL 2024, Project Page: this https URL

- **What's New**: 이 논문에서는 자율주행의 해석 가능성 문제를 해결하기 위한 새로운 시스템인 Hint-AD를 소개합니다. 이는 AD 모델의 전체적인 인식-예측-계획 출력에 맞춰 언어를 생성하며, 이를 통해 사용자와의 신뢰를 향상시킵니다.

- **Technical Details**: Hint-AD는 세 가지 모듈로 이루어져 있습니다: 1) Holistic Token Mixer, 2) Language Decoder, 3) 전통적인 AD 프레임워크입니다. Intermediate outputs를 언어 디코더에 맞게 변형하는 Holistic Token Mixer 모듈을 개발하였고, 데이터 정렬 작업을 통해 언어 출력과 AD 모델의 중간 출력 간의 연관성을 강화했습니다.

- **Performance Highlights**: Hint-AD는 여러 언어 작업에서 최첨단 성능을 기록하였으며, 실험 결과에 따르면 드라이빙 설명(기준 대비 20.4% 향상), 3D 밀집 캡셔닝(185% 향상), VQA(1.2% 향상), 드라이빙 명령 예측(1.2% 향상)을 달성했습니다.



### HybridFC: A Hybrid Fact-Checking Approach for Knowledge Graphs (https://arxiv.org/abs/2409.06692)
- **What's New**: 이번 논문은 Knowledge Graph (KG)에 대한 사실 확인(fact-checking) 접근 방식을 제안합니다. 저자들은 HybridFC라는 하이브리드 접근 방식을 통해 다양한 기존 사실 확인 방법의 장점을 결합하여 예측 성능을 크게 향상시켰습니다.

- **Technical Details**: HybridFC는 텍스트 기반, 경로 기반(path-based), 규칙 기반(rule-based), 임베딩 기반(embedding-based) 접근 방식을 앙상블 학습(ensemble learning) 환경에서 통합하여 각 접근 방식의 한계를 극복합니다. 이 방법은 미리 훈련된 KG 임베딩과 문장 변환기(sentence transformer) 모델을 사용하여 전이 학습(transfer learning)을 활용합니다.

- **Performance Highlights**: HybridFC는 FactBench 데이터셋의 Receiver Operating Characteristic (ROC) 곡선에서 0.14에서 0.27까지 기존 최첨단 기술들보다 우수한 성능을 보이며 평균적으로 다른 텍스트-, 경로-, 규칙-, 임베딩 기반 접근 방식보다 최소 0.14 AUROC 점수를 초과했습니다.



### Geometric-Averaged Preference Optimization for Soft Preference Labels (https://arxiv.org/abs/2409.06691)
- **What's New**: 이 연구에서는 인간의 선호가 이진적이고 결정론적이라는 기존의 가정을 뒤집고, 개인별로 다를 수 있는 분포적(preferential distribution) 선호를 다룬다. 이를 통해 Direct Preference Optimization (DPO) 알고리즘을 개선하는 방법을 제안한다.

- **Technical Details**: 우리는 distributional soft preference labels를 도입하고, LLM의 출력 가능도를 손실 함수에 가중 지오메트릭 평균(weighted geometric average)을 사용하여 DPO를 개선한다. 이를 통해 소프트 레이블(soft labels)에 기반하여 학습 손실의 스케일이 조정되며, 동등하게 선호되는 응답의 손실은 제로에 가까워진다.

- **Performance Highlights**: 실험 결과, geometric averaging이 정렬 연구를 위한 표준 벤치마크에서 성능을 일관되게 향상시킨다는 것을 시연하였다. 특히, 이 연구에서는 이진 레이블에 비해 더 선호되는 응답을 관찰할 수 있었고, 소극적으로 신뢰할 수 있는 레이블이 대부분을 차지하는 데이터에서 유의미한 개선을 보였다.



### Benchmarking Sub-Genre Classification For Mainstage Dance Music (https://arxiv.org/abs/2409.06690)
Comments:
          Submitted to ICASSP 2025

- **What's New**: 본 연구는 메인스테이지 댄스 음악 분류에 대한 종합적인 데이터셋과 베이스라인을 소개하여 시장의 필요한 요구를 충족하고 있습니다. 기존의 전통적인 음악 장르 데이터셋과 비교하여, 소프트 레이블링을 통해 EDM 서브 장르의 복잡성을 명확히 담아내고 있습니다.

- **Technical Details**: 이 연구에서는 주로 집합 장르를 포함한 1,000개 이상의 트랙으로 구성된 새로운 데이터셋을 제작하였으며, 컨볼루션 신경망(CNNs) 및 비전 변환기(ViTs)를 사용하여 기능을 추출합니다. 데이터셋의 레이블은 8개의 서브 장르를 포함하며, 각 트랙에는 소프트 레이블을 사용하여 여러 장르 특성을 반영합니다.

- **Performance Highlights**: 개발된 딥 러닝 모델은 현재의 최첨단 다중 모델 언어 모델(MLLMs) 보다 우수한 성능을 보였으며, 음악 추천, DJ 셋 큐레이션 및 상호작용 멀티미디어와 같은 다양한 응용 프로그램에 적합하다는 점에서 주목받고 있습니다.



### Liability and Insurance for Catastrophic Losses: the Nuclear Power Precedent and Lessons for AI (https://arxiv.org/abs/2409.06673)
Comments:
          Accepted to Generative AI and Law Workshop at the International Conference on Machine Learning (ICML 2024)

- **What's New**: 이 논문은 자율성이 증가하는 AI 시스템이 초래할 수 있는 재앙적 손실에 대한 우려를 배경으로 삼아, Critical AI Occurrences (CAIOs)로 인한 피해에 대해 AI 모델 개발자들에게 제한적이고 엄격하며 독점적인 제3자 책임을 부여해야 한다고 주장합니다.

- **Technical Details**: CAIO는 경제적 임계값, 특정 기술적 실패 방식 및 영향을 받는 인물/대상 등을 기준으로 정의될 수 있습니다. 개발자는 손실을 방지하고 새로운 규정을 위한 안전한 설계를 촉진하기 위해 의무적으로 CAIO 보험을 보장해야 한다고 합니다.

- **Performance Highlights**: 이 연구는 참여자들이 스스로의 범위를 넘어서는 위험을 최소화하도록 함으로써, AI에 대한 안전한 기술의 개발을 독려할 수 있는 법적 틀을 제시합니다. 보험 회사는 규제 기관의 부족을 보완하고, 위험 관리 및 안전 연구를 통해 손실 예방을 위한 역할을 할 것으로 기대됩니다.



### Insuring Uninsurable Risks from AI: The State as Insurer of Last Resor (https://arxiv.org/abs/2409.06672)
Comments:
          Accepted to Generative AI and Law Workshop at the International Conference on Machine Learning (ICML 2024)

- **What's New**: 이 논문은 AI 시스템이 예기치 않은 리스크를 초래할 수 있다는 전문가의 경고에 대응하기 위해 정부 제공 의무 보상 프로그램인 AIDIP(AI Disaster Insurance Program)를 제안합니다. 이 프로그램은 AI 개발자들에게 위험 가격 책정을 통해 사회적으로 최적의 수준의 관리(care)를 유도합니다.

- **Technical Details**: AIDIP는 정부 기관이 주관하여 AI 모델 훈련 시 발생할 수 있는 재해 시나리오의 위험을 평가하고, Bayesian Truth Serum(BTS) 메커니즘을 활용해 전문가들의 솔직하고 노력적인 응답을 유도합니다. 이 시스템은 정보 비대칭 문제를 해결하고, 위험 추정 시 정부의 신뢰성을 높이며, 매년 정기적으로 설문 조사를 진행합니다.

- **Performance Highlights**: 이 프로그램은 리스크를 관리하는 데 있어 개발자들에게 명확한 신호를 제공하고, 보다 정확한 위험 기반 가격 책정을 통해 소송을 줄이며, AE 개발자들이 위험을 완화해야 하는 바를 구체화하는 데 큰 장점을 지닙니다.



### LLaMA-Omni: Seamless Speech Interaction with Large Language Models (https://arxiv.org/abs/2409.06666)
Comments:
          Preprint. Project: this https URL

- **What's New**: 이 논문에서는 LLaMA-Omni라는 새로운 모델 아키텍처를 제안하여, 오픈소스 대규모 언어 모델(LLM)과의 저지연(large latency) 및 고품질의 음성 상호작용(speech interaction)을 가능하게 합니다. 이는 전통적인 텍스트 기반 상호작용보다 월등한 사용자 경험을 제공합니다.

- **Technical Details**: LLaMA-Omni는 사전 훈련된 음성 인코더(speech encoder), 음성 어댑터(speech adaptor), LLM, 스트리밍 음성 디코더(streaming speech decoder)로 구성됩니다. 이 모델은 음성 트랜스크립션(transcription) 없이 사용자의 음성 지시를 직접 처리하고, 텍스트와 음성 응답을 동시에 생성할 수 있는 기능을 제공합니다.

- **Performance Highlights**: 실험 결과, LLaMA-Omni는 콘텐츠와 스타일 모두에서 이전 음성 언어 모델에 비해 더 나은 응답을 제공하며, 응답 지연(latency)은 226ms로 매우 낮습니다. 또한 LLaMA-Omni의 훈련은 4개의 GPU에서 3일 이내에 완료되어, 향후 효율적인 음성 언어 모델 개발이 가능할 것으로 기대됩니다.



### World-Grounded Human Motion Recovery via Gravity-View Coordinates (https://arxiv.org/abs/2409.06662)
Comments:
          Accepted at SIGGRAPH Asia 2024 (Conference Track). Project page: this https URL

- **What's New**: 새로운 Gravity-View (GV) 좌표계 시스템을 도입하여 단일 비디오에서 세계에 기반한 인간 모션을 효과적으로 복원하는 방법을 제안합니다.

- **Technical Details**: GV 좌표계는 중력을 기반으로 하고 카메라 시점 방향에 의해 정의되어 이미지-포즈 매핑의 모호성을 크게 줄입니다. 제안된 방법은 각 프레임에서 인간 포즈를 추정하고, 이를 전역 좌표계로 변환하여 누적 오류를 피하는 방식으로 작동합니다. 트랜스포머 모델 및 Rotary Positional Embedding (RoPE)을 활용하여 장기간의 모션 시퀀스를 효과적으로 처리합니다.

- **Performance Highlights**: 자연적인 모션을 카메라 공간과 세계에 기반한 환경 모두에서 복원하며, 현재 기술 대비 정확성과 속도에서 우수한 성능을 보여줍니다.



### EyeCLIP: A visual-language foundation model for multi-modal ophthalmic image analysis (https://arxiv.org/abs/2409.06644)
- **What's New**: EyeCLIP은 277만 개 이상의 멀티 모달 안과 이미지와 일부 텍스트 데이터를 활용하여 개발된 시각-언어(visual-language) 기반 모델입니다. 본 연구는 기존의 단일 모달리티 모델의 한계를 넘어, 다양한 모달리티에서의 멀티 뷰 정보를 활용하여 눈 질병을 조기에 감지할 수 있는 방법을 제안합니다.

- **Technical Details**: EyeCLIP은 자가 감독(self-supervised) 복원, 멀티 모달 이미지 대조 학습(multi-modal image contrastive learning), 이미지-텍스트 대조 학습(image-text contrastive learning)을 결합한 프리트레이닝(pretraining) 전략을 도입하여, 다양한 모달리티의 공유 표현(shared representation)을 학습합니다. 이러한 방법은 라벨이 붙지 않은 다수의 멀티 모달 데이터를 효과적으로 해석할 수 있게 합니다.

- **Performance Highlights**: 14개의 벤치마크 데이터셋을 사용한 평가 결과, EyeCLIP은 눈 및 전신 질병 관련 다양한 다운스트림 작업에서 최첨단 성능을 달성했습니다. 특히, EyeCLIP은 실세계의 긴 꼬리(long-tail) 시나리오에서 몇 샷(few-shot) 및 제로 샷(zero-shot) 능력을 보여주며, 질병 분류, 시각적 질문 응답(visual question answering), 크로스 모달 검색(cross-modal retrieval)에서 뛰어난 성과를 보이고 있습니다.



### MoWE-Audio: Multitask AudioLLMs with Mixture of Weak Encoders (https://arxiv.org/abs/2409.06635)
- **What's New**: 이번 논문에서는 AudioLLM(오디오 대형 언어 모델)의 성능을 향상시키기 위해 ‘약한’ 인코더(MoWE: mixtures of weak encoders)를 결합하는 새로운 방법을 제안합니다. 기존의 AudioLLM은 주로 '강한' 인코더와 사전 학습된 대형 언어 모델을 조합하여 사용하고 있지만, 새로운 작업 및 데이터셋에 대한 제한된 능력으로 인해 많은 도전을 받고 있습니다.

- **Technical Details**: MoWE는 기본 인코더를 보완하기 위해 여러 개의 작은 인코더 풀을 도입하고, 오디오 입력에 따라 선택적으로 활성화하여 특징(feature) 추출을 강화하는 전략을 사용합니다. 이러한 접근 방식은 모델 크기를 크게 증가시키지 않으면서도 다중 작업 성능을 개선할 수 있습니다. 제안된 AudioLLM 프레임워크는 강력한 기본 오디오 인코더와 다수의 약한 오디오 인코더로 구성되어 있습니다. 활성화된 인코더의 임베딩은 결합되어 AudioLLM 파이프라인에서 추가 처리됩니다.

- **Performance Highlights**: 실험 결과, MoWE는 오디오LLMs의 다중 작업 성능을 효과적으로 개선하여 다양한 오디오 작업에 대한 적용 가능성을 넓혔습니다. 기존의 강력한 오디오 인코더와 결합함으로써, 새로운 데이터셋 및 작업에 대한 적응력이 크게 향상되었습니다.



### A Practice of Post-Training on Llama-3 70B with Optimal Selection of Additional Language Mixture Ratio (https://arxiv.org/abs/2409.06624)
Comments:
          11 pages, 4 figures

- **What's New**: 이 논문에서는 Llama-3 8B 및 70B 모델에 대한 지속적 사전 훈련(Continual Pre-Training, CPT)을 수행하여 모델의 중국어 능력을 향상시키는 방법을 제안합니다. 특히, Additional Language Mixture Ratio (ALMR)와 Learning Rate (LR) 간의 최적 상관관계를 규명하며, 이로 인해 모델의 성능을 극대화하는 실험 설정을 안내합니다.

- **Technical Details**: 논문에서 제안하는 ALMR과 LR 간의 최적 상관관계를 기반으로 하여, Llama-3 8B 모델은 1.7T 훈련 토큰을 사용하여 사전 훈련을 진행합니다. 실험 결과, ALMR 설정 및 LR 조정이 중요함을 보여주며, 전체 실험 과정에서 효과적인 하이퍼파라미터 조정이 이루어집니다. 또한, 70B 모델에 대해서도 동일한 ALMR과 낮은 LR을 적용하고, 최종적으로 Supervised Fine-tuning (SFT)과 Direct Preference Optimization (DPO)를 통해 모델을 정밀 조정합니다.

- **Performance Highlights**: 중국어 관련 벤치마크뿐만 아니라 수학, 코딩, 감정 지능과 같은 특정 도메인에서도 모델 성능이 향상되었습니다. 최종적으로 배포된 70B 모델은 산업용 채팅 시스템에서 만족스러운 성능을 입증하며, 감정 지능을 개선한 챗봇을 통해 인간과의 상호작용에서도 뛰어난 성과를 보입니다.



### One-Shot Imitation under Mismatched Execution (https://arxiv.org/abs/2409.06615)
- **What's New**: 본 논문에서는 로봇과 인간 시연자 간의 실행 불일치를 다루기 위해 RHyME이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 최적 운송 비용(optimal transport costs)을 활용하여 로봇 시나리오에서 사람의 시연을 자동으로 시뮬레이션 할 수 있도록 합니다.

- **Technical Details**: RHyME는 로봇과 시연자 간의 시퀀스 레벨 유사성을 최적 운송 비용으로 정의하여, 사람의 시연 클립을 검색하고 조합하여 긴 시나리오를 상상합니다. 이로 인해 로봇 정책이 쌍 데이터 없이 효과적으로 훈련됩니다. RHyME는 여러 크로스-엠바디먼트 데이터 세트에서 다양한 실행 불일치 정도에 대해 성능 분석을 제공합니다.

- **Performance Highlights**: 실험 결과, RHyME은 모든 수준의 시연 불일치에 대해 기존 방법들보다 높은 성능을 보여주었습니다. 특히, 상이한 시각적 표현을 교차 활용하여 학습하고 활용하는 통찰력을 발견했습니다.



### Label-free Monitoring of Self-Supervised Learning Progress (https://arxiv.org/abs/2409.06612)
- **What's New**: 이번 연구는 Self-Supervised Learning (SSL) 방법론을 새롭게 평가하기 위해 라벨이 없는 데이터에서의 인코더 품질을 모니터링하는 몇 가지 평가 지표를 제안합니다. 기존의 방법들은 주석이 있는 데이터에 의존해 왔지만, 본 연구에서는 라벨이 없는 데이터의 임베딩에서 직접 평가할 수 있는 방법을 모색합니다.

- **Technical Details**: 우리는 $k$-means 클러스터링과 실루엣 점수(silhouette score), 클러스터 일치성을 이용해 클러스터 품질을 측정했습니다. 또한, 임베딩 분포의 엔트로피(entropy)를 측정하여, 학습이 진행됨에 따라 동일 샘플의 서로 다른 뷰 간 거리가 줄어들고 있다는 가설을 기반으로 평가의 일관성을 조사했습니다. 중립적인 라벨이 없는 클러스터링 메트릭은 SimCLR과 MoCo-v2로 학습했을 때만 LP 정확도(linear probe accuracy)와 상관관계를 갖는 것으로 나타났습니다.

- **Performance Highlights**: 클러스터링 기반 접근법은 동일한 아키텍처 비교에서만 유효하다는 결과가 도출되었습니다. 반면, 엔트로피는 아키텍처에 독립적일 수 있으며, 학습이 진전됨에 따라 엔트로피는 일반적으로 감소하였으나 SimSiam의 경우 트렌드가 반전되었습니다. 이는 예기치 않은 행동으로 추가 연구가 필요합니다.



### Simulation-based Scenario Generation for Robust Hybrid AI for Autonomy (https://arxiv.org/abs/2409.06608)
Comments:
          6 pages, 5 figures, 1 table

- **What's New**: 최근의 드론 (UAV) 수색 및 구조, 비상 관리, 법 집행 분야의 발전은 저렴한 플랫폼과 센서 페이로드의 출현으로 더욱 가속화되고 있습니다. 하지만 현재 UAV 시뮬레이션 환경은 이 복합적 접근법에 적합한 의미론적 맥락을 결여하고 있습니다. 이를 해결하기 위해 HAMERITT라는 하이브리드 AI 임무 환경이 개발되었습니다.

- **Technical Details**: HAMERITT (Hybrid Ai Mission Environment for RapId Training and Testing)는 시뮬레이션 기반 자율성 소프트웨어 프레임워크로, 신경-상징적 알고리즘의 훈련, 테스트 및 보증을 지원합니다. 이는 상징적 정보와 원시 센서 데이터를 결합하여 임무 관련 시나리오를 생성하여 자율 기동 및 인식 reasoning을 가능하게 합니다.

- **Performance Highlights**: HAMERITT는 다양한 도심 지역에서 단일 UAV 수색 임무를 위해 여러 시나리오를 생성하며, Microsoft 시뮬레이터를 활용하여 복잡한 환경 조건을 제공합니다. 향후 ANSR 프로그램의 Phase 2에서는 performer 알고리즘을 통합하여 더욱 정확한 공통 작전 그림 (Common Operational Picture)을 제공할 계획입니다.



### An Ontology-based Approach Towards Traceable Behavior Specifications in Automated Driving (https://arxiv.org/abs/2409.06607)
Comments:
          22 pages, 12 figures, submitted for publication

- **What's New**: 자동차 자동화 시스템의 동작을 명확히 규명하기 위해, 본 논문에서는 자동화 차량의 행동 사양을 명세하는 데 사용되는 'Semantic Norm Behavior Analysis'라는 새로운 접근 방식을 제안합니다. 이 방법은 온톨로지(ontology)를 기반으로 하여 특정 작업 환경에서의 행동을 포멀하게 표현하고, 해당 행동과 이해관계자의 요구 간의 관계를 명확히 합니다.

- **Technical Details**: 본 논문에서는 자동화 차량의 안전한 행동을 보장하기 위해 요구사항을 설정하고, 이와 관련된 가정들을 문서화하는 방법을 제시합니다. 'Scenario-based approach'를 통해 자동화 차량의 동작 조건을 구조화하고, 다양한 이해관계자 요구를 통합하여 일관된 행동 사양을 도출합니다. 온톨로지를 사용하여 명세화를 지원하고, 자동화된 일관성 검증을 가능하게 합니다. 또한, 행동 사양의 예시적 적용을 통해 결과를 평가합니다.

- **Performance Highlights**: 제안된 방법론은 차량의 안전한 행동을 보장하기 위한 명확한 요구사항을 정의하고, 다양한 이해관계자의 요구를 충족시키는데 초점을 맞추고 있습니다. 실험 결과는 이 방법이 개발 프로세스에서의 협업과 소통을 증진시키며, 보다 안전한 자동화 차량 개발에 기여할 수 있음을 보여줍니다.



### Developing the Temporal Graph Convolutional Neural Network Model to Predict Hip Replacement using Electronic Health Records (https://arxiv.org/abs/2409.06585)
Comments:
          Accepted to the 2024 International Conference on Machine Learning and Applications (ICMLA). 8 pages, 3 figures, 7 tables

- **What's New**: 이 연구는 Temporal Graph Convolutional Neural Network (TG-CNN) 모델을 사용하여 고관절 교체 수술을 1년 전에 예측하여, 환자의 삶의 질을 향상시키고 건강 서비스의 효율성을 높이는 방법을 제시합니다.

- **Technical Details**: 연구는 NHS의 ResearchOne 전자 건강 기록(EHR) 데이터를 기반으로 하여 40-75세 환자들에 대한 고관절 교체 리스크를 예측하는 데 초점을 맞췄습니다. 모델은 9,187명의 고관절 교체 환자와 9,187명의 대조군에 대해 훈련되었습니다. TG-CNN 모델을 통해 데이터 분석을 수행하였으며, 3D CNN과 LSTM이 사용되었습니다.

- **Performance Highlights**: 모델의 예측 성능은 AUROC 0.724, AUPRC 0.185로, 재교정 후에도 유의미한 결과를 나타내었습니다. 이를 통해 TG-CNN 모델이 고관절 교체 리스크를 효과적으로 예측하며, 환자의 경과를 이해하고 관리하는 데 기여할 수 있음을 보여줍니다.



### Quantifying and Enabling the Interpretability of CLIP-like Models (https://arxiv.org/abs/2409.06579)
- **What's New**: 이 논문에서는 CLIP 같은 모델의 해석 가능성을 정량화하기 위한 연구를 제안하고, OpenAI와 OpenCLIP의 여섯 가지 서로 다른 CLIP 모델에서 그 결과를 분석했습니다. 특히 TEXTSPAN 알고리즘을 사용하여 각 Attention Head를 구체적인 속성으로 분해하고, 새로운 메트릭스를 통해 해석 가능성을 평가했습니다.

- **Technical Details**: 이 연구에서는 TextSpan 알고리즘을 사용하여 각 Attention Head와 연결된 텍스트 설명을 분해하고, In-Context Learning을 통해 속성을 라벨링합니다. 두 가지 메트릭스인 entanglement score (엉킴 점수)와 association score (연관 점수)를 도입하여 해석 가능성을 정량화했습니다.

- **Performance Highlights**: 결과적으로, 큰 CLIP 모델이 더 작은 모델보다 일반적으로 더 해석 가능하다는 것을 발견했습니다. CLIP-InterpreT라는 해석 가능성 분석 도구를 소개하고, 이 도구는 다섯 가지 유형의 분석을 제공하여 사용자가 CLIP 모델의 내부 작동 원리를 이해할 수 있도록 돕습니다.



### Indirect Dynamic Negotiation in the Nash Demand Gam (https://arxiv.org/abs/2409.06566)
Comments:
          Appears in IEEE Access

- **What's New**: 이 논문은 불완전 정보 상황에서의 순차적인 양자 협상 문제를 다루고 있습니다. 저자들은 대응 모델을 제안하여 에이전트들이 간접적으로 협상하고 상대방의 모델을 학습하도록 돕습니다.

- **Technical Details**: 제안된 모델은 Bayesian learning 및 Markov decision processes를 활용하여, 개인의 이익을 극대화하는 확률적 결정 에이전트를 구성합니다. 이 에이전트는 학습 능력을 갖추고 있으며, 간접 협상 및 사적 정보의 비공유를 통해 협상할 수 있도록 설계되었습니다.

- **Performance Highlights**: 이 모델은 협상 과정에서 에이전트 행동의 조정을 유도하고, 성공률을 극대화하여 개인적인 이익을 증가시키는 결과를 보여주었습니다.



### ChatGPT's Potential in Cryptography Misuse Detection: A Comparative Analysis with Static Analysis Tools (https://arxiv.org/abs/2409.06561)
Comments:
          ESEM 2024

- **What's New**: 최근 연구에서 ChatGPT가 Java Cryptography Architecture(JCA)의 암호화 API 오용을 탐지하는 성능을 평가하였습니다. 이 연구는 ChatGPT의 능력을 CryptoAPI-Bench 벤치마크와 비교하여 밝혔습니다.

- **Technical Details**: 연구는 ChatGPT의 암호화 API 오용 탐지 능력을 CryptoAPI-Bench를 기반으로 평가했습니다. ChatGPT는 12개 오용 카테고리에서 평균 86%의 F-measure을 달성하였으며, 이는 최첨단 정적 분석 도구인 CryptoGuard보다 우수한 성능을 보였습니다.

- **Performance Highlights**: ChatGPT는 5개 카테고리에서 CryptoGuard를 초과하는 성능을 보였으며, 프롬프트 엔지니어링 기술을 적용한 후 평균 F-measure이 94.6%로 상승하여 10개 카테고리에서 최고의 성능을 기록했습니다.



### Questioning Internal Knowledge Structure of Large Language Models Through the Lens of the Olympic Games (https://arxiv.org/abs/2409.06518)
- **What's New**: 이 연구는 대형 언어 모델(LLMs)의 내부 지식 구조를 분석하며, 올림픽 메달 수치를 사용하여 이와 관련된 성능을 조사합니다.

- **Technical Details**: LLMs는 1964년부터 2022년까지의 올림픽 메달 데이터를 바탕으로 두 가지 과제에 대해 평가되었습니다: 1) 특정 팀의 메달 수 계산 및 2) 특정 순위를 달성한 팀의 식별. 실험에는 SOTA 모델과 오픈소스 모델이 포함되었으며, 모델의 성능을 비교하고 분석했습니다.

- **Performance Highlights**: SOTA LLM 모델들은 팀별 메달 수를 보고하는 데 뛰어난 성능을 보였으나, 특정 순위와 관련된 질문에서는 40% 이하의 정확도만을 기록했습니다. 이 연구는 또한 LLM이 사용자 의혹에 반응할 때 성능 저하를 보이며, 신뢰성을 유지하기 위한 추가 연구의 필요성을 강조합니다.



### Sine, Transient, Noise Neural Modeling of Piano Notes (https://arxiv.org/abs/2409.06513)
- **What's New**: 본 논문은 새로운 피아노 소리 모사 방법을 제안합니다. 우리는 사인파(sine), 과도(transient), 잡음(noise) 분해를 활용하여 피아노 음을 복제하는 미분 가능 스펙트럼 모델링 합성기(differentiable spectral modeling synthesizer)를 설계했습니다. 각 세 가지 하위 모듈이 피아노 녹음에서 이 구성 요소들을 학습하고 해당하는 조화(harmonic), 과도, 잡음 신호를 생성합니다.

- **Technical Details**: 이 모델은 세 가지 독립적으로 학습 가능한 모델로 나뉘어 복잡성을 줄입니다. 근사 조화 내용(quasi-harmonic content)은 물리학에서 유도된 공식을 사용하여 미분 가능 사인 모델을 통해 생성하며, 매개변수는 오디오 녹음에서 자동으로 추정됩니다. 잡음 하위 모듈은 학습 가능한 시간 가변 필터를 사용하며, 과도는 심층 컨볼루션 네트워크(deep convolutional network)를 통해 생성됩니다.

- **Performance Highlights**: 모델은 목표의 일부 분포를 일치시키지만 스펙트럼의 높은 부분의 에너지를 예측하는 데 있어 더 많은 도전 과제가 존재합니다. 전반적으로 과도 및 잡음 구성 요소의 에너지 분포는 정확하게 모델링되었으며, 모델은 계산적으로 더 효율적이고 메모리 효율성이 높습니다. 그러나 지각 테스트는 음의 공격 단계 모델링의 정확성이 제한적임을 드러냈습니다. 그럼에도 불구하고 일반적으로 단일 음과 트라이코드를 모사하는 데 지각적 정확성을 달성합니다.



### Aligning Machine and Human Visual Representations across Abstraction Levels (https://arxiv.org/abs/2409.06509)
Comments:
          51 pages

- **What's New**: 이 논문에서는 인간의 행동을 더 잘 모사하는 인공지능 시스템을 만들기 위해, 인간의 개념 지식이 계층적으로 구성되는 방식과 AI 모델의 표현 간의 불일치를 강조합니다.

- **Technical Details**: AI 모델이 인간의 유사성 판단을 더 잘 반영하도록 하는 새로운 Kullback-Leibler 다이버전스(KL divergence) 기반의 목표 함수를 사용하며, AligNet이라는 대규모 유사성 데이터셋을 생성합니다. 이 데이터셋은 두 가지 레벨의 개념적 구체성을 모두 포함하여 기계 학습 모델에 인간 지식의 구조를 주입합니다.

- **Performance Highlights**: AligNet을 통해 모델과 인간의 판단 간의 정렬을 개선하며, 머신 러닝 과제에서 일반화 및 분포 외(out-of-distribution) 강건성(robustness)을 향상시킵니다. 이 개선된 모델들은 다수의 시각적 유사성 작업에서 인간의 행동을 더 잘 근사할 수 있습니다.



### Elucidating Optimal Reward-Diversity Tradeoffs in Text-to-Image Diffusion Models (https://arxiv.org/abs/2409.06493)
- **What's New**: 이 논문에서는 Text-to-Image (T2I) diffusion 모델의 보상 해킹(reward hacking) 문제를 해결하기 위한 새로운 방법인 Annealed Importance Guidance (AIG) 를 제안합니다. AIG는 다양한 이미지를 생성하면서도 보상을 최적화하는 방법으로, 기존의 방식보다 나은 성능을 보입니다.

- **Technical Details**: AIG는 Annealed Importance Sampling에 영감을 받아 생성된 회귀적 규제 방식으로, diffusion 모델의 다양성을 유지하면서도 보상-다양성 간의 Pareto-Optimal 무역균형을 달성합니다. 또한, 노이즈 분포에서 데이터로 변환되는 Markov 과정을 사용하며 KL divergence와 LoRA 스케일링 기법을 분석합니다.

- **Performance Highlights**: 실험 결과, AIG는 Stable Diffusion 모델에 적용되었을 때 이미지의 다양성과 품질 모두를 향상시켰습니다. 사용자가 진행한 연구에 따르면, AIG는 다양한 모델 아키텍처 및 보상 함수에서 생성된 이미지의 질과 다양성을 개선하는 것으로 나타났습니다.



### An Effective Context-Balanced Adaptation Approach for Long-Tailed Speech Recognition (https://arxiv.org/abs/2409.06468)
Comments:
          Accepted by SLT 2024

- **What's New**: 이 논문은 E2E (End-to-end) 자동 음성 인식 (ASR) 모델에서의 rare words에 대한 인식 성능을 개선하기 위한 새로운 접근법을 제시합니다. 특히, 문맥 어댑터 (Contextual Adapter, CA)의 사용과 함께 문맥 리스트의 단어 빈도 분포를 조정하는 방법을 탐구합니다.

- **Technical Details**: 연구팀은 CA를 통해 외부 지식(문맥 단어 리스트)을 E2E ASR 모델에 주입하는 방법을 사용하였습니다. 또한, 문맥 리스트의 빈도 분포를 조정하여 모델 성능에 미치는 영향을 분석하고, 간단하지만 효과적인 문맥 균형 학습 목표를 통해 CA를 확장했습니다.

- **Performance Highlights**: AISHELL-1 벤치마크 데이터셋을 기반으로 한 일련의 실험에서는 훈련 코퍼스의 모든 어휘 단어를 문맥 리스트로 사용하고 균형 목표와 결합할 때 최상의 성능 성과를 나타냈습니다. 이 접근법은 문자 오류율 (Character Error Rate, CER)을 최대 1.21% 감소시키고, 제로샷 (zero-shot) 단어의 오류율을 9.44% 더 감소시키는 결과를 보여주었습니다.



### Multimodal Large Language Model Driven Scenario Testing for Autonomous Vehicles (https://arxiv.org/abs/2409.06450)
- **What's New**: 이 논문은 OmniTester라는 새로운 방법론을 소개하며, 이 방법론은 대규모 언어 모델(LLM)을 활용하여 자율주행차(AV) 테스트를 위한 현실적이고 다양한 시나리오를 생성하는 데 중점을 두고 있습니다. 이는 기존의 테스트 방법들이 가지는 제한점을 극복하고, 시나리오의 제어 가능성을 높이기 위해 설계되었습니다.

- **Technical Details**: OmniTester는 prompt engineering과 Simulation of Urban Mobility(SUMO) 도구의 통합을 통해 실제 도로 환경을 모방하는 지형을 생성합니다. 또한 Retrieval-Augmented Generation(RAG) 메커니즘과 자가 개선(self-improvement) 기능을 통합하여 LLM의 성능을 지속적으로 향상시키는 시스템입니다.

- **Performance Highlights**: 실험을 통해 세 가지 유형의 복잡한 시나리오 생성에서 OmniTester의 제어 가능성과 현실성을 입증했으며, 교통 사고 보고서에서 설명된 새로운 시나리오를 재구성하는 데 효과적임을 나타냈습니다.



### HexaCoder: Secure Code Generation via Oracle-Guided Synthetic Training Data (https://arxiv.org/abs/2409.06446)
Comments:
          24 pages, 16 tables, 8 figures

- **What's New**: HexaCoder는 LLM이 보안이 강화된 코드를 자동으로 생성할 수 있도록 지원하는 혁신적인 접근 방식입니다. 이 방법은 보안 취약점이 있는 코드와 수정된 코드 쌍을 생성하여 데이터 수집의 부담을 줄입니다.

- **Technical Details**: HexaCoder는 두 가지 주요 구성 요소로 구성됩니다: 오라클 기반 데이터 합성 파이프라인과 두 단계의 보안 코드 생성 과정입니다. 데이터 합성 파이프라인은 특정 CWE 유형에 대해 취약한 코드와 수정된 코드의 쌍을 생성하며, 여기서 LLM을 사용하여 취약한 코드를 복구합니다. 보안 오라클은 취약점을 감지하고 LLM이 이를 수정하도록 합니다. 이 과정에서 각 데이터 샘플은 보안 관련 라이브러리와 코드를 포함하여 보안의 두 가지 측면을 통합하는 데 기여합니다.

- **Performance Highlights**: 우리는 세 가지 서로 다른 벤치마크에서 네 가지 LLM을 평가하여 HexaCoder가 생성한 코드의 보안을 개선하면서도 기능적 정확성을 유지한다는 것을 입증하였습니다. 특히, 생성된 취약한 코드의 수를 기존 방법에 비해 최대 85% 감소시켰습니다.



### Learning Generative Interactive Environments By Trained Agent Exploration (https://arxiv.org/abs/2409.06445)
- **What's New**: 본 논문에서는 Genie라는 세계 모델을 발전시킨 GenieRedux와 GenieRedux-G 모델을 소개합니다. 이 모델들은 강화 학습 기반의 에이전트를 사용하여 다양한 환경에서 훈련 데이터를 생성하여 더 나은 성능을 발휘합니다.

- **Technical Details**: GenieRedux는 원래 Genie 모델의 구성 요소들을 구현하였으며, STTN(Spatiotemporal Transformer) 아키텍처를 활용합니다. GenieRedux-G는 에이전트의 행동을 조건으로 하여 예측을 수행하며, 이는 예측의 불확실성을 제거하는 데 도움을 줍니다.

- **Performance Highlights**: 실험 결과, GenieRedux-G는 Visual Fidelity와 controllability에서 뛰어난 성능을 보였습니다. 특히, Coinrun 환경에서의 평가에서는 훈련된 에이전트를 기준으로 한 모델이 무작위 에이전트에 비해 다양한 상황에서 더 우수한 성능을 나타냈습니다.



### GeMuCo: Generalized Multisensory Correlational Model for Body Schema Learning (https://arxiv.org/abs/2409.06427)
Comments:
          Accepted at IEEE Robotics and Automation Magazine

- **What's New**: 이번 연구에서는 로봇이 스스로 센서와 액추에이터 간의 상관관계를 설명하는 몸(schema) 모델을 자율적으로 습득하고, 이를 기반으로 상태를 추정 및 제어하며 비정상 감지 및 시뮬레이션을 수행하는 일반화된 다중 감각 상관 모델(GeMuCo)을 제안합니다. 이 모델은 로봇이 변화하는 환경에 맞춰 실시간으로 적응할 수 있도록 돕습니다.

- **Technical Details**: GeMuCo는 다중 센서 및 제어 입력 데이터를 수집한 후, 네트워크의 입력/출력과 가능한 마스크 세트를 자동으로 결정하는 구조 결정기(Structure Determinator)를 통해 학습됩니다. 이를 통해 모델은 정적 및 동적 모션을 처리할 수 있는 능력을 갖추게 됩니다.

- **Performance Highlights**: 연구 결과는 각각의 축 구동 로봇에 대한 도구 사용, 근골격 로봇을 위한 관절-근육 매핑 학습, 저강성 플라스틱 휴머노이드 로봇을 위한 전신 도구 조작 학습에 적용되어 GeMuCo의 효과성을 입증했습니다. 이 연구는 로봇의 자율 학습 능력 개발에 기여할 것으로 기대됩니다.



### Exploring the Integration of Large Language Models in Industrial Test Maintenance Processes (https://arxiv.org/abs/2409.06416)
Comments:
          Under submission to ACM TOSEM

- **What's New**: 본 연구에서는 소프트웨어 테스트의 유지보수 과정에서 대규모 언어 모델(LLMs)을 활용할 수 있는 가능성을 탐구합니다. 조사 결과, 테스트 유지보수가 필요한 트리거를 식별하고 LLM 에이전트가 취할 수 있는 행동을 제시함으로써 LLM이 산업 환경에서 효율성을 높여줄 수 있다는 점을 강조합니다.

- **Technical Details**: 이 연구는 두 가지 다중 LLM 아키텍처를 제안하여, 소스 코드 변경 후 어느 테스트 케이스가 유지보수가 필요한지를 예측할 수 있습니다. 연구는 Ericsson AB에서 진행되었으며, 구체적으로 37개의 저수준 변경 사항과 7개의 고수준 개발 결정이 테스트 유지보수를 유발하는 트리거로 파악되었습니다. 또한, LLM의 산업 내 배치에 대한 기술적 및 조직적 고려사항을 논의합니다.

- **Performance Highlights**: 제안된 LLM 에이전트 프로토타입의 성능과 유용성을 평가한 결과, 실제 코드베이스에서의 테스트 유지보수 과정에서 LLM의 적용 가능성과 효과적인 지원 역할을 할 수 있음을 확인했습니다. 이 연구는 소프트웨어 개발자에게 실질적인 지침을 제공하여, 인공지능 기술을 테스트 유지보수에 어떻게 적용할 수 있는지에 대한 이해를 높이고 있습니다.



### Symmetry Breaking in Neural Network Optimization: Insights from Input Dimension Expansion (https://arxiv.org/abs/2409.06402)
Comments:
          29 pages, 8 figures

- **What's New**: 이번 연구에서는 대칭 깨기(symmetry breaking)가 신경망 최적화에서 가지는 중요성을 강조하는 가설을 제시하고, 입력 차원 확장을 통한 네트워크 성능 향상을 demonstrat하고 있습니다. 이는 기존의 최적화 기법의 기초 원리를 이해하는 데 도움이 됩니다.

- **Technical Details**: 연구에서는 입력 차원 확대 기법을 통해 레이어의 대칭을 깨는 과정을 도모하였고, 이를 통해 신경망에서 더욱 매끄러운 파라미터 전환이 가능해짐을 보였습니다. 또한 drop-out, 배치 정규화(batch normalization), 균등성(equivariance) 등의 최적화 기법들이 대칭 깨기 원칙에 부합함을 보여주었습니다.

- **Performance Highlights**: 실험 결과, ResNet-18 모델에서 CIFAR-10 데이터셋에 대한 정확도가 94%로 증가하며, 이는 대칭 깨기 원칙을 적용한 입력 차원 확장 덕분으로 평가됩니다. 다양한 데이터셋에서도 일관되게 모델 성능이 향상되었으며, 특히 FGVC-Aircraft와 DTD 데이터셋에서 눈에 띄는 성과를 보여주었습니다.



### Distilling Generative-Discriminative Representations for Very Low-Resolution Face Recognition (https://arxiv.org/abs/2409.06371)
- **What's New**: 이 논문에서는 매우 저해상도 얼굴 인식을 용이하게 하는 생성-판별적 표현 증류 접근법을 제안합니다. 이는 생성 표현과 교차 해상도 정렬 지식 증류를 결합하여 수행됩니다.

- **Technical Details**: 제안된 접근법은 두 개의 증류 모듈을 통해 생성적 및 판별적 모델을 함께 증류하여 매우 저해상도 얼굴 인식을 돕습니다. 첫째, 생성 표현 증류는 얼굴 초해상화 고정 모델의 인코더를 사용하여 특징 회귀를 통해 학생 모델을 학습시키고, 그 후 학생 모델의 기초 구조를 동결합니다. 둘째, 판별적 표현 증류는 미리 훈련된 얼굴 인식기를 사용하여 교차 해상도 관계적 대비 증류를 통해 학생 모델을 학습하도록 지원합니다.

- **Performance Highlights**: 광범위한 실험을 통해 제안된 방법은 저해상도 얼굴의 인식 정확도를 향상시키며, 매우 저해상도의 얼굴에서 누락된 세부 정보를 복구하는 데 효과적임을 보여주었습니다.



### Texture-AD: An Anomaly Detection Dataset and Benchmark for Real Algorithm Developmen (https://arxiv.org/abs/2409.06367)
- **What's New**: Texture-AD 데이터셋은 산업 결함 탐지 알고리즘의 성능을 평가할 수 있도록 구성된 최초의 데이터셋으로, 여러 제품의 다양한 명세를 갖춘 훈련 세트와 실제 생산 과정에서 촬영된 결함 있는 제품으로 구성된 테스트 세트를 제공합니다.

- **Technical Details**: Texture-AD는 15종의 천, 14종의 반도체 웨이퍼, 10종의 금속판 등의 다양한 이미지로 구성되어 있으며, 이를 통해 표면 결함을 포함한 10종 이상의 결함을 탐지할 수 있습니다. 이 데이터셋은 고해상도 이미지 43,120,431,204,312,043,120을 포함하며, 각 제품의 결함 부위에 대한 픽셀 단위 주석을 제공합니다.

- **Performance Highlights**: 실험 결과, Texture-AD는 최신 알고리즘에게도 어려운 도전 과제가 되었으며, 알고리즘의 견고성과 일반화 능력을 평가하는 기준으로 활용될 수 있습니다.



### Connecting Concept Convexity and Human-Machine Alignment in Deep Neural Networks (https://arxiv.org/abs/2409.06362)
Comments:
          First two authors contributed equally

- **What's New**: 이번 연구는 신경망의 'convexity'와 인간-기계 정렬(human-machine alignment) 사이의 관계를 조사합니다. 이는 신경망 표현들이 인간의 인지 과정과 어떻게 일치하는지를 이해하는 데 중요한 발걸음을 제공합니다.

- **Technical Details**: 본 연구는 pretrained 및 fine-tuned 비전 트랜스포머 모델에서 convexity와 인간-기계 정렬의 상관관계를 식별합니다. 연구에서는 graph convexity score와 odd-one-out 정확도를 사용하여 이 두 측정을 실험적으로 조사합니다.

- **Performance Highlights**: 신경망의 latent 공간에서 형성된 convex 영역은 인간의 개념과 어느 정도 일치함을 발견하였으며, alignment 최적화는 일반적으로 convexity를 향상시키지만, fine-tuning을 통한 convexity 증가는 일관되지 않은 영향을 미친다는 복잡한 관계를 보여줍니다.



### VoiceWukong: Benchmarking Deepfake Voice Detection (https://arxiv.org/abs/2409.06348)
- **What's New**: VoiceWukong은 최신 기술과 다양한 목소리 조작을 포함하는 포괄적인 딥페이크 음성 탐지 벤치마크를 제안합니다. 이 데이터셋은 19개의 상용 도구 및 15개의 오픈 소스 도구를 통해 생성된 음성 샘플을 포함하고 있습니다.

- **Technical Details**: VoiceWukong 데이터셋은 265,200개의 영어 및 148,200개의 중국어 딥페이크 음성 샘플로 구성되어 있으며, 총 38개의 데이터 변형이 포함되어 있습니다. 12개 최첨단 탐지기 모델을 평가한 결과, AASIST2가 13.50%의 equal error rate (EER)를 달성했습니다.

- **Performance Highlights**: 대부분의 탐지기들은 20% 이상의 EER을 기록하였고, 사용자 연구에서 인간이 딥페이크 음성을 탐지하는 데 있어 자동 시스템보다 더 나은 성능을 보였습니다. 또한, 인간의 false acceptance rates (FARs)는 클래스 0의 경우 18.97%로 나타났고, 82%를 초과하는 클래스 2의 음성에서는 대부분의 탐지기보다 낮은 성과를 보였습니다.



### Compute-Update Federated Learning: A Lattice Coding Approach (https://arxiv.org/abs/2409.06343)
Comments:
          Extended version of the preprint available at arXiv:2403.01023

- **What's New**: 이 논문은 디지털 통신을 통해 공중에서 계산하는 것을 가능하게 하는 연합 학습 프레임워크를 소개합니다. 새로운 공동 소스-채널 코딩 기법을 이용하여, 장치에서 채널 상태 정보에 의존하지 않고 라티스 코드를 사용하여 모델 파라미터를 양자화하고 장치의 간섭을 활용합니다.

- **Technical Details**: 제안된 수신기 구조는 서버에서의 새로운 수신기 구조를 포함하며, 이는 양자화된 모델 파라미터의 정수 조합을 신뢰성 있게 디코딩하여 집합을 위해 사용합니다. 수학적 접근법을 통해 제안된 기법의 수렴 한계를 도출하였으며, 각 통신 라운드에서 효과적인 정수 계수를 결정하기 위한 집합 메트릭과 알고리즘을 제안하였습니다.

- **Performance Highlights**: 우리의 결과는 채널 동적 상태와 데이터 이질성에 관계없이, 제안된 기법이 여러 매개변수에서 일관되게 우수한 학습 정확도를 제공하며, 다른 공중 계산 방법론을 현저히 초월한다는 것을 보여줍니다.



### Towards Agentic AI on Particle Accelerators (https://arxiv.org/abs/2409.06336)
Comments:
          4 pages, 3 figures, Machine Learning and the Physical Sciences at Workshop at the 38th conference on Neural Information Processing Systems (NeurIPS)

- **What's New**: 이 논문은 가속기 제어를 위한 새로운 패러다임인 분산형 다중 에이전트 프레임워크를 제안합니다. 이 시스템은 대규모 언어 모델(LLMs)을 기반으로 하며, 각 에이전트가 고유한 가속기 구성 요소를 전문적으로 제어하는 것을 목표로 합니다. 이 접근법은 인공지능(AI)이 가속기 운영에 어떻게 활용될 수 있는지, 자율 복잡계 시스템의 구현 가능성, 그리고 운영 데이터 레이블링과 전문가의 지침을 위한 인간-인-루프(human-in-the-loop) 통합의 중요성에 대해 탐구합니다.

- **Technical Details**:  제안된 시스템은 LLM에 의해 제어되는 자율 에이전트가 하위 구성 요소를 제어하고, 고차원 작업 및 커뮤니케이션을 처리하는 방식으로 구성됩니다. 이 프레임워크는 가속기가 전이 상태에서 안정성을 유지하면서도 더 복잡한 기능을 수행할 수 있는 가능성을 보여줍니다. 이를 통해 예를 들어, LLM 전원 피드백 시스템 및 유럽 XFEL 이탈피드백 관리자를 통해 운영 효율성을 높일 수 있습니다.

- **Performance Highlights**: 논문에서 제시된 두 가지 예시, 즉 Advanced Light Source (ALS) 궤도 피드백 시스템과 유럽 XFEL 길이 피드백 관리자는 제안된 프레임워크가 가속기의 운영 안정성 및 효율성을 어떻게 향상시킬 수 있는지를 입증합니다. 지금까지의 기존 시스템과의 비교를 통해, 새로운 에이전트 기반 접근 방식이 최적의 운영 성능을 달성하는 데 중요한 역할을 할 수 있음을 보여줍니다.



### LAMP: Learnable Meta-Path Guided Adversarial Contrastive Learning for Heterogeneous Graphs (https://arxiv.org/abs/2409.06323)
Comments:
          19 pages, 7 figures

- **What's New**: 논문에서는 Heterogeneous Graph Contrastive Learning(HGCL)과 관련하여, 메타-패스 조합이 비지도 설정에서의 모델 성능에 미치는 영향을 분석했습니다. 이를 통해 새로운 접근 방법인 LAMP (Learnable Meta-Path)를 소개하였고, 이는 다양한 메타-패스 서브 그래프를 통합하여 안정성을 제공하도록 설계되었습니다.

- **Technical Details**: LAMP는 다양한 메타-패스 서브 그래프를 통합한 새로운 적대적 대비 학습(adversarial contrastive learning) 방식으로, 그래프 내 엣지 밀도를 유지하면서 모델의 성능과 견고성을 향상시킵니다. AGCL에서 많이 사용되는 적대적 교육(adversarial training) 전략을 활용하여 엣지 프루닝(edge pruning)을 수행하며, 메타-패스와 네트워크 스키마 뷰 간의 차이를 최대화하여 보다 의미 있는 정보를 포착합니다.

- **Performance Highlights**: LAMP는 Heterogeneous Graph Benchmark(HGB)에서 4개의 다양한 데이터 세트를 사용한 실험에서 기존의 최첨단 비지도 모델들과 비교하여 정확도 및 견고성 면에서 유의미하게 향상된 성능을 보여주었습니다.



### PharmacoMatch: Efficient 3D Pharmacophore Screening through Neural Subgraph Matching (https://arxiv.org/abs/2409.06316)
- **What's New**: 이 연구에서 우리는 약물 발견을 위한 가상 스크리닝 방법의 효율성을 향상시키기 위해 새로운 대조 학습 기반 접근 방식을 제시합니다. 이 방법은 약물 후보 물질의 검색을 보다 효율적으로 수행하도록 개선되었습니다.

- **Technical Details**: 우리의 방법은 3D pharmacophore 스크리닝을 정보의 질량 매칭 문제로 재정의하여, 쿼리-타겟 관계를 임베딩 공간에서 효율적으로 인코딩합니다. 우리는 그래프 신경망(GNN) 인코더를 사용하여 3D pharmacophores의 의미 있는 벡터 표현을 생성합니다. 이 인코더는 자가 지도 학습(self-supervised learning)을 통해 학습되며, 대조 손실 목표를 사용하여 쿼리와 타겟 간의 관계를 캡처합니다.

- **Performance Highlights**: 이 방법을 사용하여 가상 스크리닝 데이터셋에서 실험을 수행한 결과, pharmacophore 매칭의 실행 시간이 크게 단축되었습니다. 이는 매우 큰 데이터셋을 간단하고 빠르게 스크리닝할 수 있는 가능성을 보여줍니다.



### An End-to-End Approach for Chord-Conditioned Song Generation (https://arxiv.org/abs/2409.06307)
- **What's New**: 본 논문에서는 소리 생성 네트워크에 코드(chords)를 도입하여 기존 방법보다 음악적 성능과 생성 제어 정밀도를 향상시키는 Chord-Conditioned Song Generator (CSG) 모델을 제안합니다.

- **Technical Details**: CSG는 코드와 가사 토큰을 입력으로 받아 오토회귀 방식으로 노래 프레임을 생성합니다. 입력 토큰은 고차원 공간으로 매핑되고, 이 후에 Attention with Dynamic Weights Sequence를 활용하여 코드와 가사를 결합하여 의미적 융합을 수행합니다. 이를 통해 모델의 정확성과 생성된 음악의 질을 개선합니다.

- **Performance Highlights**: 실험 결과, 제안된 CSG 모델은 기타 접근 방법들보다 음악적 성능과 생성된 노래의 제어 정밀도에서 뛰어난 성과를 보여주었습니다.



### Enhancing Long Video Understanding via Hierarchical Event-Based Memory (https://arxiv.org/abs/2409.06299)
- **What's New**: 이번 논문은 장기 비디오 이해를 위한 새로운 계층적 이벤트 기반 메모리 강화 LLM(HEM-LLM)을 제시합니다. 기존 모델들이 비디오의 다양한 의미 정보를 압축하여 사용했던 것과 달리, HEM-LLM은 각 사건을 개별적으로 처리하여 정보 중복을 피합니다.

- **Technical Details**: HEM-LLM은 적응형 시퀀스 세분화 스킴을 통해 긴 비디오 내 여러 사건을 분리합니다. 또한 개별 사건에 대한 메모리 모델링을 수행하며, 사건 간의 장기 의존성을 강화하기 위해 이전 사건 정보를 압축하여 주입합니다. 이로써 비디오의 장기적 의미 이해를 높입니다.

- **Performance Highlights**: 다양한 비디오 이해 작업(비디오 질문 답변, 비디오 캡셔닝 등)에서 HEM-LLM은 아홉 개의 벤치마크 데이터 세트에서 최첨단 성능을 기록했습니다. 이는 기존 모델보다 긴 비디오의 내용 이해에서 월등한 능력을 보여줍니다.



### User Preferences for Large Language Model versus Template-Based Explanations of Movie Recommendations: A Pilot Study (https://arxiv.org/abs/2409.06297)
Comments:
          Presented to the Dutch-Belgian Workshop on Recommender Systems 2023 (14-15 December, 2023 - Antwerp, Belgium)

- **What's New**: 이 논문에서는 추천 시스템의 설명 방식을 혁신하기 위해 대규모 언어 모델(LLMs)을 활용한 새로운 접근 방식을 제안합니다. 기존의 템플릿 기반 설명 대신 LLM 기반 설명이 사용자 경험을 어떻게 향상시킬 수 있는지에 대한 파일럿 연구 결과를 공유합니다.

- **Technical Details**: 논문에서는 그래프 기반 추천 방법(Personalized PageRank 및 RippleNet)을 사용하여 생성된 추천 사항의 설명을 LLM을 통해 자연어로 변환하는 방법을 설명합니다. 25명의 사용자와 함께 설명의 효과를 평가했으며, LLM이 제공하는 설명이 더 풍부하고 매력적일 수 있음을 강조하였습니다.

- **Performance Highlights**: LLM 기반 설명이 사용자 기대에 더 부합하는 경향을 보였고, 특히 LLM에서 지식 그래프를 이용한 설명이 가장 좋은 평가를 받았습니다. 그러나 샘플 수가 적어 결과에는 큰 변동이 있었으며, 설명의 신뢰성에 대한 추가적인 연구가 필요함을 시사합니다.



### Catch Me if You Can: Detecting Unauthorized Data Use in Deep Learning Models (https://arxiv.org/abs/2409.06280)
- **What's New**: 최근 딥러닝(DL)의 발전으로 인해 데이터 사용에 대한 사용자들의 권리가 제한되고 있어, 이를 해결하기 위한 MembershipTracker라는 데이터 출처 도구가 제안되었습니다.

- **Technical Details**: MembershipTracker는 데이터에 소규모의 특화된 변화를 주는 경량화된 데이터 마킹 기법과 사용자 샘플의 강한 암기를 감사(audit)하는 MI 기반 검증 프로세스로 구성되어 있습니다. 사용자는 오직 훈련 세트의 0.005%에서 0.1%의 소량 데이터만 마킹하면 되며, 모델의 무단 데이터 사용 감지가 가능합니다.

- **Performance Highlights**: MembershipTracker는 다양한 설정에서 효과성이 입증되었으며, 특히 ImageNet-1k 데이터셋을 포함한 산업 규모의 훈련에서 높은 성능을 보였습니다. FPR(거짓 긍정 비율)은 평균 0%로, TPR(진정 긍정 비율)은 100%에 달하였으며, 데이터 출처 추적에서 실용적인 도구로 자리잡게 되었습니다.



### Ferret: Federated Full-Parameter Tuning at Scale for Large Language Models (https://arxiv.org/abs/2409.06277)
- **What's New**: 전 세계 여러 실제 응용 분야에서 대형 언어 모델(LLMs)의 사용이 필수불가결해졌습니다. 그러나 데이터 프라이버시와 통신 효율성이 중요한 연합 설정에서 이러한 모델을 대규모로 미세 튜닝하는 것은 큰 도전 과제가 있습니다. 본 연구에서는 LLM의 연합 전체 매개변수 튜닝을 위한 Ferret를 제안합니다. 이는 공유 난수를 사용하여 LLM의 전체 매개변수를 대규모로 튜닝할 수 있는 최초의 1차 방법입니다.

- **Technical Details**: Ferret은 다음 세 가지 측면을 통해 이를 달성합니다: (1) 효율적인 로컬 업데이트를 위해 널리 적용되는 1차 방법을 사용하며; (2) 이러한 업데이트를 저차원 공간으로 사영하여 통신 오버헤드를 크게 줄이고; (3) 이 저차원 공간에서 공유 난수를 통해 로컬 업데이트를 재구성하여 효과적인 전체 매개변수 글로벌 집합을 촉진하여 빠른 수렴을 보장하고 경쟁력 있는 최종 성능을 달성합니다.

- **Performance Highlights**: Ferret는 기존의 연합 전체 매개변수 튜닝 방법의 확장성을 크게 향상시켜 높은 계산 효율성, 감소된 통신 오버헤드 및 빠른 수렴을 달성하고 있습니다. 실험 결과, Ferret은 기존 방법보다 우수한 성능을 보여주어 대규모 연합 환경에서 LLM의 배치에 있어 바람직한 솔루션이 됩니다.



### Towards Robust Uncertainty-Aware Incomplete Multi-View Classification (https://arxiv.org/abs/2409.06270)
Comments:
          Ongoing work: 9 pages, 6 figures, 2 tables

- **What's New**: 본 논문에서는 불완전한 다중 뷰 분류(multi-view classification, MVC) 문제를 효과적으로 해결하기 위한 새로운 방법인 교차 점진적 학습 네트워크(Alternating Progressive Learning Network, APLN)를 제안합니다.

- **Technical Details**: APLN은 먼저 조잡한 보간(coarse imputation)을 수행하고, 이어서 데이터를 잠재 공간(latent space)으로 매핑하여 증거 분포를 점진적으로 학습합니다. 이 과정에서 불확실성을 고려하여 Dirichlet 분포를 모델링하고, 갈등을 보다 잘 처리하기 위해 갈등 인식 Dempster-Shafer 조합 규칙(conflict-aware Dempster-Shafer combination rule, DSCR)을 도입합니다.

- **Performance Highlights**: APLN은 기존 방법들보다 높은 성능을 보였으며, 특히 불확실성과 갈등이 높은 환경에서보다 신뢰할 수 있는 결정을 내리는 데 기여합니다.



### Keyword-Aware ASR Error Augmentation for Robust Dialogue State Tracking (https://arxiv.org/abs/2409.06263)
- **What's New**: 이 논문에서는 Dialogue State Tracking (DST) 모델의 견고성을 향상시키기 위한 새로운 데이터 증강 방법인 Error Positioning Augmentation (EPA)를 제안합니다. 이 방법은 ASR (Automatic Speech Recognition) 시스템의 오류를 고려해 키워드에 오류를 배치함으로써 DST 성능을 개선합니다.

- **Technical Details**: EPA는 대규모 언어 모델(LLMs)을 활용하여 ASR 오류가 발생하는 키워드와 관련된 오류 패턴을 생성합니다. 이 과정은 문장 수준의 ASR 오류와 키워드 수준의 음성 변형을 포함하는 두 단계로 이루어져 있습니다. 예를 들어, 사용자가 제공한 작은 데이터셋(10개 이하의 샘플)만으로도 다양하고, 정확한 음성유사성을 가진 키워드 오류를 생성할 수 있습니다.

- **Performance Highlights**: EPA를 통해 DST 모델의 정확도가 45.76%에서 51.12%로 향상되었습니다. 키워드 증강이 주효한 결과로, ASR 오류가 있는 값에 대한 오류율이 크게 감소했습니다.



### DiPT: Enhancing LLM reasoning through diversified perspective-taking (https://arxiv.org/abs/2409.06241)
Comments:
          LLM Reasoning with Perspectives, Preprint

- **What's New**: 이 논문은 DiPT라는 새로운 접근 방식을 제안합니다. 이는 기존의 언어 모델 추론 방법에 다양화된 관점을 추가하여 문제의 맥락에 대한 깊은 이해를 촉진하고 가장 효과적인 해결 경로를 식별할 수 있도록 합니다.

- **Technical Details**: DiPT는 Diversified Perspective-Taking 방법론을 통해 기존의 추론 방법을 확장하는 프레임워크입니다. 이 방법은 추론 단계에서 다양한 시각을 분석하도록 모델에 지시하며, 학습 단계에서는 데이터 품질을 개선하는 전반적인 레시피로 작용합니다. 특히, 기존의 instruction-tuning 데이터셋을 다양한 관점에서의 이유를 포함시켜 정보를 풍부하게 만듭니다.

- **Performance Highlights**: DiPT는 기존 방법에 유연하게 통합되어 정확도를 최대 6%까지 향상시키고 질문의 재구성으로 인한 불일치를 감소시킵니다. 또한, DiPT는 모델의 맥락 이해를 개선시키고, 다양한 관점으로 강화된 데이터로 미세 조정할 경우 모델의 추론 능력이 향상됨을 보여줍니다.



### CerviXpert: A Multi-Structural Convolutional Neural Network for Predicting Cervix Type and Cervical Cell Abnormalities (https://arxiv.org/abs/2409.06220)
- **What's New**: 이 논문에서는 CerviXpert라는 다구조(Structural) 합성곱 신경망(Convolutional Neural Network, CNN) 모델을 제안하여 자궁경부암 식별의 정확성과 효율성을 동시에 달성하고자 합니다. 특히, 기존의 고비용 딥러닝 모델들 대신 계산 복잡성을 줄이고 실용성을 고려한 접근법을 채택했습니다.

- **Technical Details**: CerviXpert는 세 가지 세포 이상 유형(정상, 비정상, 양성)을 분류하는 데 초점을 맞추었으며, 일반적인 딥러닝 기술인 ResNet50, VGG16, MobileNetV2 및 InceptionV3보다 효율적인 성능을 제공합니다. 연구진은 공개 데이터셋인 SiPaKMeD에서 방대한 실험을 수행하여 제안하는 방법의 유효성을 입증했습니다.

- **Performance Highlights**: CerviXpert는 자궁경부암 진단에 있어 높은 정확도를 달성하면서도 계산 비용이 저렴한 솔루션을 제공합니다. 이 모델은 실질적으로 자궁경부암 스크리닝 프로세스를 개선할 것으로 기대됩니다.



### Towards Generalizable Scene Change Detection (https://arxiv.org/abs/2409.06214)
Comments:
          7 pages, 5 figures

- **What's New**: 본 논문에서는 기존의 SCD(장면 변화 감지) 방법의 한계점을 극복하기 위해 Generalizable Scene Change Detection Framework (GeSCF)를 제안합니다. GeSCF는 재훈련 없이도 unseen domains에 대한 일반화를 도모하며, 새로운 평가 지표와 데이터셋을 통해 SCD의 성능을 측정합니다.

- **Technical Details**: GeSCF는 foundation model의 지역적 의미를 활용하여 유사성 분포를 적응형으로 임계처리하여 초기 pseudo-change mask를 생성합니다. 이후 Segment Anything Model (SAM)의 클래스 비의존 마스크를 통해 pseudo-masks를 정제하고, 모든 설정에서 교환 가능성을 유지하여 완전한 시간 일관성을 보장합니다.

- **Performance Highlights**: 광범위한 실험을 통해 GeSCF는 다양한 도전적인 환경에서 뛰어난 성능을 보이며, 기존 최첨단 SCD 방법론을 능가하였습니다. 특히 unseen domains에서의 변화 감지를 0-shot으로 수행하며, 특정 데이터셋에서 미세 조정된 SCD 모델과 동등 이상의 성능을 달성하였습니다.



### Adaptive Transformer Modelling of Density Function for Nonparametric Survival Analysis (https://arxiv.org/abs/2409.06209)
- **What's New**: 이번 논문에서는 Transformer 아키텍처에 기반한 새로운 비모수적 생존 회귀 모델인 UniSurv를 제안합니다. UniSurv는 이전 분포 가정 없이 고품질의 단일 모드 확률 밀도 함수(PDF)를 생성할 수 있으며, Margin-Mean-Variance 손실을 최적화하여 생존 예측에서의 민감도를 크게 향상시킵니다.

- **Technical Details**: UniSurv는 정적 특성과 동적 특성을 각각 다른 임베딩 브랜치를 통해 처리하며, 긴급 데이터의 높은 결측 비율 케이스에 효과적으로 대응합니다. 이 모델은 시간 불변 및 시간 가변 공변량을 포함하고, 오른쪽 검열이 있는 생존 데이터 세트를 전제로 하여 작동합니다.

- **Performance Highlights**: 실험 결과, UniSurv는 다양한 데이터 세트에서 다른 기존 모델들에 비해 검열 예측에서 현저히 높은 민감도를 보이며, 보다 정확한 PDF 생성에 성공했습니다.



### NOVI : Chatbot System for University Novice with BERT and LLMs (https://arxiv.org/abs/2409.06192)
- **What's New**: 대학교 신입생의 적응을 돕기 위해, SKKU 'Everytime'의 게시물과 댓글 데이터를 활용하여 GPT-4o 기반의 챗봇 시스템 NOVI를 개발했습니다. NOVI는 신입생뿐만 아니라 다양한 사람들이 새로운 환경에 적응하는 데 도움을 줄 것으로 기대됩니다.

- **Technical Details**: NOVI는 LangChain을 기반으로 개발되었으며, BLEU, Perplexity, ROUGE-1, ROUGE-2, ROUGE-L, METEOR 등의 다양한 평가 지표를 통해 성능을 평가했습니다. 시스템 흐름은 사용자 쿼리를 처리하기 위해 Flask를 사용하여 RetrievalQA로 데이터를 전송하고, 유용한 정보를 찾아 모델을 미세 조정하는 방식으로 이루어집니다.

- **Performance Highlights**: 이 연구는 신입생들이 대학 생활에 적응하는 데 필요한 구체적인 정보를 제공함으로써, 성공적인 첫 해를 시작할 수 있도록 실질적인 도움을 줄 것으로 기대하고 있습니다. 이는 LLM 연구의 미래 발전을 위한 기초를 마련하는데 기여할 것입니다.



### Can Large Language Models Unlock Novel Scientific Research Ideas? (https://arxiv.org/abs/2409.06185)
Comments:
          24 pages, 12 figures, 6 tables

- **What's New**: 이 논문은 LLMs(대형 언어 모델)가 연구 논문을 기반으로 새로운 연구 아이디어를 생성할 수 있는 능력을 탐구합니다. 연구팀은 4개의 LLM을 다양한 분야에서 조사하여, LLM의 생성 아이디어가 저자의 관점과 얼마나 일치하는지를 평가했습니다.

- **Technical Details**: 연구에서는 Claude-2, GPT-4, GPT-3.5 및 Gemini 모델을 분석하였고, ‘Idea Alignment Score(아이디어 정렬 점수)’와 ‘Idea Distinctness Index(아이디어 독창성 지수)’를 제안하여 생성된 아이디어의 질을 평가했습니다. 인공지능 모델의 성능을 연구하기 위해 460개의 생성 아이디어를 인적 평가하여 혁신성과 관련성을 평가했습니다.

- **Performance Highlights**: Claude-2와 GPT-4는 GPT-3.5와 Gemini보다 저자의 관점과 더 잘 일치하는 미래 연구 아이디어를 생성했습니다. Claude-2는 다른 모델들보다 더 다양한 연구 아이디어를 생성하였고, 연구 결과는 LLM의 아이디어 생성 능력과 한계를 보여줍니다.



### Larger Language Models Don't Care How You Think: Why Chain-of-Thought Prompting Fails in Subjective Tasks (https://arxiv.org/abs/2409.06173)
Comments:
          5 pages, 2 figures, 1 table

- **What's New**: 이 논문은 In-Context Learning (ICL)와 Chain-of-Thought (CoT) 기법이 대형 언어 모델(LLM)에서 어떻게 상호 작용하는지를 조사합니다. 새로운 발견으로, CoT를 사용하더라도 LLM이 과거 지식(prior knowledge)에 의존하는 경향이 지속됨을 보이며, 이는 주관적인 감정 및 도덕적 판단 같은 복잡한 주제에 대한 성능 저하로 이어진다는 점을 강조합니다.

- **Technical Details**: 논문에서는 ICL과 CoT의 조합이 LLM의 추론을 어떻게 변화시키는지를 살펴봅니다. 특히, CoT가 주어진 증거를 무시하고 문제 해결을 위해 과거의 추론 체인을 불러오는 경향을 보여주며, 이로 인해 예측의 포스터리어가 간소화되고 있음을 발견했습니다. 연구는 6개의 최신 LLM을 대상으로 하여 CoT의 성능을 평가하고, CoT로 생성된 추론의 합리성을 분석했습니다.

- **Performance Highlights**: 연구 결과에 따르면, CoT를 적용한 LLM이 주관적 작업, 특히 감정 및 도덕 관련 태스크에서 여전히 성능 저하를 보였습니다. 또한, 더 큰 LLM일수록 CoT를 사용하더라도 이전의 추론 체인이 여전히 고정적(prior)이라는 것을 확인했습니다.



### MCDGLN: Masked Connection-based Dynamic Graph Learning Network for Autism Spectrum Disorder (https://arxiv.org/abs/2409.06163)
Comments:
          8 pages, 7 figures

- **What's New**: 이 연구에서는 유동적인 뇌 연결성을 탐지하기 위해 Masked Connection-based Dynamic Graph Learning Network (MCDGLN)라는 새로운 접근법을 제안합니다. 기존의 연구들이 정적 뇌 네트워크에만 초점을 맞춘 것에 비해, 우리 모델은 동적인 연결성을 포착하고 네트워크 잡음을 줄이는 방법을 사용합니다.

- **Technical Details**: MCDGLN은 BOLD 신호를 슬라이딩 시간 창을 이용하여 세분화하고, 특수한 가중치 엣지 집계 모듈(WEA)을 통해 동적 기능 연결성을 통합합니다. 또한, 계층 그래프 컨볼루션 네트워크(HGCN)를 활용하여 그래프 레벨의 특성을 추출하고, 자기 주의 메커니즘을 통해 중요 특성을 강조합니다. 커스터마이즈된 태스크 전용 마스크를 사용하여 정적 기능 연결성을 정제하는 단계도 포함됩니다.

- **Performance Highlights**: ABIDE I 데이터셋에 적용한 결과, ASD와 일반 대조군(TC) 그룹 간의 분류 정확도는 73.3%에 달했습니다. WEA와 ACE의 역할이 뚜렷하게 나타나며, 이는 ASD 특유의 특성을 포착하는데 중요한 기여를 합니다.



### Multiclass Arrhythmia Classification using Smartwatch Photoplethysmography Signals Collected in Real-life Settings (https://arxiv.org/abs/2409.06147)
- **What's New**: 이번 연구는 premature atrial contraction (PAC) 및 premature ventricular contraction (PVC) 감지의 민감도를 83%로 향상시키고, atrial fibrillation (AF) 감지의 정확도를 97.31%로 유지하는 멀티모달 데이터를 사용하여 성과를 달성했습니다.

- **Technical Details**: 연구는 1D-Bi-GRU (Bi-directional Gated Recurrent Unit) 모델을 사용하여 1D PPG (photoplethysmography), accelerometer (ACC), 그리고 heart rate (HR) 데이터를 결합하여 세 가지 부정맥 유형을 분류합니다. 방대한 스마트워치 데이터를 통해 72명의 독립 피험자를 대상으로 테스트했습니다.

- **Performance Highlights**: PAC/PVC 감지의 최종 민감도는 이전 최고 성과보다 20.81% 증가한 83%로, AF 감지의 정확도는 97.31%로 보고되었습니다. 이 모델은 기존 모델에 비해 14배 가볍고 2.7배 빠른 처리 속도를 보여줍니다.



### Draw an Audio: Leveraging Multi-Instruction for Video-to-Audio Synthesis (https://arxiv.org/abs/2409.06135)
Comments:
          14 pages, 11 figures

- **What's New**: 본 논문은 'Draw an Audio'라는 새로운 비디오-오디오(V2A) 합성 모델을 소개합니다. 이 모델은 드로잉 마스크(drawn masks)와 음량 신호(loudness signals)를 통해 여러 입력 명령어를 지원하면서, 콘텐츠 일관성(content consistency), 시간 일관성(temporal consistency), 그리고 음량 일관성(loudness consistency) 문제를 동시에 해결할 수 있습니다.

- **Technical Details**: 'Draw an Audio'는 Mask-Attention Module (MAM)과 Time-Loudness Module (TLM)을 통해 기능합니다. MAM은 마스크된 비디오 지시를 사용하여 모델이 관심 영역에 집중하도록 하고, TLM은 보조 음량 신호를 활용하여 생성된 오디오가 비디오와 시간 및 음량 차원에서 일치하도록 보장합니다. 또한, 'VGGSound-Caption'이라는 대규모 V2A 데이터셋을 확장하여 캡션 프롬프트(caption prompts)로 주석을 추가했습니다.

- **Performance Highlights**: 광범위한 실험 결과, 'Draw an Audio'는 두 개의 대규모 V2A 데이터셋에 대한 도전적인 벤치마크에서 최첨단 성능을 달성합니다. 이 연구는 Foley 디자이너의 요구를 보다 효과적으로 충족시키는 오디오 합성 프레임워크를 제공합니다.



### Accelerating Large Language Model Pretraining via LFR Pedagogy: Learn, Focus, and Review (https://arxiv.org/abs/2409.06131)
- **What's New**: 이 논문에서는 기존의 LLM (Large Language Model) 사전 학습 방법론과는 다르게, 인간의 학습 방식에서 영감을 얻어 데이터 샘플링 방법을 개선한 새로운 LFR (Learn, Focus, and Review) pedagogy를 제안합니다. 이 방법은 복잡한 데이터 블록에 집중하고 정기적으로 리뷰하여 장기 기억에 정보를 효과적으로 저장할 수 있도록 설계되었습니다.

- **Technical Details**: LFR는 모델의 perplexities를 기록하고, 기억 가능성이 높은 높은 perplexity를 가진 데이터 블록을 자주 다시 방문합니다. GPT-2 모델을 OpenWebText 데이터셋에서 사전 학습하였으며, 학습 속도는 20배 빨라졌고, 모델의 정확도는 기존 OpenAI 모델보다 낮은 perplexity와 높은 정확도를 달성하였습니다.

- **Performance Highlights**: LFR을 통해 GPT-2 모델을 345M에서 1.5B 파라미터로 사전 학습한 결과, 언어 모델링, 질문 응답, 번역 및 문제 해결 등 6개의 하위 작업에서 일관되게 낮은 perplexity와 높은 정확도를 기록했습니다. 또한, 사전 학습 과정에서 약 20배 더 적은 반복 학습으로 이러한 개선을 이루었습니다.



### On the Weaknesses of Backdoor-based Model Watermarking: An Information-theoretic Perspectiv (https://arxiv.org/abs/2409.06130)
- **What's New**: 이 논문은 머신러닝 모델의 지적 재산을 보호하기 위한 새로운 방법, In-distribution Watermark Embedding (IWE)을 제안합니다. 기존의 워터마크 삽입 기술이 백도어 공격에 취약하다는 점을 지적하고 이에 대한 해결책을 제안합니다.

- **Technical Details**: IWE는 정상 샘플과 트리거 세트 샘플 간에 적절한 겹침을 확보하여 워터마크의 복원력을 높입니다. 또한, 모델의 로짓(logit)을 워터마크 정보의 전달체로 활용하여 모델 성능에 미치는 영향을 최소화합니다.

- **Performance Highlights**: CIFAR-100 및 Caltech-101와 같은 실제 데이터셋에서 다양한 공격에 대해 뛰어난 방어 성능을 보였으며, 정확도 손실은 0.1% 미만으로 유지되었습니다.



### PaRCE: Probabilistic and Reconstruction-Based Competency Estimation for Safe Navigation Under Perception Uncertainty (https://arxiv.org/abs/2409.06111)
- **What's New**: 이 논문은 인식 기반 내비게이션 시스템의 안전한 운행을 위한 새로운 방법인 PaRCE(Probabilistic and Reconstruction-based Competency Estimation)를 개발하였습니다. 이 방법은 입력 이미지의 전반적인 친숙도 점수를 추정하고 특정 지역의 정보를 제공합니다.

- **Technical Details**: PaRCE 방법은 모델이 입력 이미지에 대한 예측의 불확실성을 인식하고, 이를 바탕으로 안전하고 효과적인 내비게이션을 수행할 수 있도록 하는데 초점을 맞춥니다. 이 시스템은 전반적인 친숙도 점수와 지역적 친숙도 맵을 생성하여 알맞은 경로를 계획하고 제어하는 데 활용됩니다.

- **Performance Highlights**: 본 연구에서 제안한 시스템은 기존 비교군에 비해 낯선 장애물과의 충돌을 현저히 줄였으며, 이러한 지역적 친숙도 정보는 효율적인 내비게이션을 가능하게 하였습니다.



### Doppelg\"anger's Watch: A Split Objective Approach to Large Language Models (https://arxiv.org/abs/2409.06107)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 "generation supervision" 문제를 연구하고, 유용성(helpfulness)과 감독 신호(supervision signal)를 분리하는 새로운 bicameral 아키텍처(bicameral architecture)인 Doppelgänger를 제안합니다.

- **Technical Details**: Doppelgänger는 토큰 생성을 감독하고, 각 토큰까지의 시퀀스에 대한 감독 점수(supervision score)를 예측하는 모듈입니다. 이 새로운 구조는 Transformer 아키텍처를 확장한 것으로, 언어 모델의 기본 능력은 변경되지 않으면서도 다양한 감독 신호에 대한 성능을 독립적으로 최적화할 수 있습니다.

- **Performance Highlights**: 이 접근 방식은 기존의 방법들과 비교해 감독 신호의 최적화를 돕고, LLM의 유용성을 유지하며, 훈련 과정에서 대규모 데이터가 필요하지 않습니다. 이는 다양한 모달리티에서도 적용할 수 있는 장점이 있습니다.



### Latent Diffusion Bridges for Unsupervised Musical Audio Timbre Transfer (https://arxiv.org/abs/2409.06096)
- **What's New**: 이번 논문은 오디오 신호의 음색(timbre) 특성을 변형하면서 멜로디 구조(melodic structure)를 유지하는 새로운 방법론을 제시합니다.

- **Technical Details**: 제안된 방법은 두 개의 diffusion bridge를 기반으로 하며, CocoChorales Dataset을 사용하여 학습되었습니다. 각 diffusion 모델은 특정 악기와 함께 Gaussian prior에 대해 훈련되며, 추론(inference) 과정에서 소스 모델(source model)과 타겟 모델(target model)로 지정되어 음색 전이를 용이하게 합니다.

- **Performance Highlights**: 실험 결과, 우리의 방법은 VAEGAN 및 Gaussian Flow Bridges(GFB)와 비교하여 Fréchet Audio Distance(FAD)가 더 낮고 멜로디 보존(melody preservation) 측면에서도 더 나은 성능을 나타냈습니다. 또한 Gaussian prior로부터의 노이즈 수준($\sigma$)을 조정함으로써 멜로디 보존 정도와 음색 전이의 양을 조절할 수 있다는 점을 발견했습니다.



### Scalable Multitask Learning Using Gradient-based Estimation of Task Affinity (https://arxiv.org/abs/2409.06091)
Comments:
          16 pages

- **What's New**: 이번 논문은 Grad-TAG라는 새로운 알고리즘을 제안하여 멀티태스크 학습에서의 task affinity를 반복적인 훈련 없이 추정할 수 있는 방법을 다룹니다. 이 접근법은 gradient 기반의 손실 근사를 통해 이루어지며, 기존 방법에 비해 계산 비용을 획기적으로 줄입니다.

- **Technical Details**: Grad-TAG 알고리즘은 'base' 모델을 하나 훈련한 뒤, 특정 task 조합에 대한 손실을 선형화 기법을 사용해 추정합니다. 이 과정에서 저차원 프로젝션을 사용한 손실 근사를 통해, logistic regression으로 task 조합의 레이블을 예측합니다.

- **Performance Highlights**: 우리의 실험 결과, Grad-TAG는 7개의 데이터셋에서 우수한 성능을 보여주며, task affinity의 추정 정확도가 진짜 친화도와의 거리가 2.7% 이내입니다. 가장 큰 그래프 데이터셋에서는 5% 이내의 거리로 추정하였으며, 필요한 계산 리소스는 전체 훈련의 3%에 불과하였습니다.



### MTLSO: A Multi-Task Learning Approach for Logic Synthesis Optimization (https://arxiv.org/abs/2409.06077)
- **What's New**: 이번 논문에서는 Logic Synthesis Optimization(LSO)를 개선하기 위한 Multi-Task Learning을 활용한 새로운 방법론인 MTLSO를 제안합니다. MTLSO는 제한된 데이터의 활용을 극대화하고, 이진 다중 레이블 그래프 분류의 보조 작업을 도입하여 태스크 간의 시너지를 통해 'Quality of Results'(QoR)를 예측합니다.

- **Technical Details**: MTLSO는 기존의 GNN(Graph Neural Network) 기법을 기반으로 하며, 계층적 그래프 표현 학습 전략을 채택하여 대형 AIG(And-Inverter Graph)들의 표현력을 향상시킵니다. 본 연구에서는 QoR 예측 작업과 함께 추가적인 그래프 분류 작업을 통해 데이터의 효율성을 높이는데 주목하고 있습니다.

- **Performance Highlights**: 다양한 데이터셋에 대한 실험 결과, 제안된 방법은 지연(delay)에서 평균 8.22% 성능 향상, 면적(area)에서는 5.95%의 성능 향상을 달성하였습니다. 이는 전통적인 GNN 기법보다 우수한 성능을 보여줍니다.



### Privacy-Preserving Data Linkage Across Private and Public Datasets for Collaborative Agriculture Research (https://arxiv.org/abs/2409.06069)
- **What's New**: 이 논문은 디지털 농업에서의 데이터 공유와 관련된 프라이버시 문제를 해결하기 위한 새로운 프레임워크를 제안합니다. 이 프레임워크는 농부들이 데이터를 안전하게 공유할 수 있도록 도와주며, 데이터의 분석과 관련된 유용성을 유지합니다.

- **Technical Details**: 제안된 프레임워크는 (1) 개인 데이터셋을 기반으로 유사한 농부를 식별하고, (2) 시간과 위치와 같은 집계 정보를 제공하며, (3) 가격 및 제품 가용성의 추세를 결정하고, (4) 식품 불안정성 통계와 같은 공공 정책 데이터와 추세를 상관시키는 알고리즘을 포함합니다.

- **Performance Highlights**: 실제 Farmer's Market 데이터셋을 이용하여 프레임워크의 유효성을 입증했습니다. 이를 통해 기계 학습 모델이 개인 정보를 보호한 데이터에서 훈련되었으며, 정책 결정자와 연구자들이 식품 불안정성과 가격 문제를 해결하는 데 도움을 줄 수 있는 결과를 보여주었습니다.



### SongCreator: Lyrics-based Universal Song Generation (https://arxiv.org/abs/2409.06029)
Comments:
          work in progress

- **What's New**: 본 논문에서는 SongCreator라는 새로운 곡 생성 시스템을 제안합니다. 이는 가사에 기반하여 고품질 곡을 생성할 수 있도록 설계되었습니다. 기존 시스템들이 극복하지 못한 음성과 반주 간의 복잡한 상호작용을 효과적으로 Modeling하여 고음질의 Song Generation을 구현합니다.

- **Technical Details**: SongCreator는 두 가지 주요 설계를 채택합니다. 첫째, Dual-Sequence Language Model (DSLM)을 통해 음성과 반주 정보를 별도로 처리합니다. 둘째, 특수한 Attention Mask 전략을 사용하여 다양한 곡 생성 작업, 예를 들어 가사에서 곡으로, 곡 편집 등의 과제를 모두 처리할 수 있도록 합니다. 이러한 설계로 인해 음성과 반주 간의 상호작용을 잘 반영합니다.

- **Performance Highlights**: SongCreator는 8개의 과제에서 최첨단(Classification) 또는 경쟁력 있는 성능을 보여주었습니다. 특히, 가사-곡 및 가사-음성 변환에 있어서 이전 연구들보다 월등히 높은 성과를 달성하였습니다. 또한, 제공된 다양한 프롬프트를 통해 음성과 반주 간의 독립적인 조절이 가능하다는 것이 큰 장점입니다.



### MessIRve: A Large-Scale Spanish Information Retrieval Datas (https://arxiv.org/abs/2409.05994)
- **What's New**: 이번 논문에서는 스페인어 정보 검색(IR)을 위한 대규모 데이터셋인 MessIRve를 소개합니다. 이 데이터셋은 약 73만 개의 쿼리와 위키피디아에서 수집된 관련 문서로 구성되어 있습니다. MessIRve는 스페인어를 사용하는 다양한 지역을 반영하여 다국적 스페인어 데이터를 제공하는데 중점을 둡니다.

- **Technical Details**: MessIRve는 Google의 자동완성 API에서 쿼리를 수집하고, 그에 해당하는 Google 검색의 ‘featured snippets’를 위키피디아에서 추출하여 관련 문서로 활용하였습니다. 데이터셋은 20개 스페인어 사용 국가에서 수집된 쿼리와 문서로 구성되며, 이로 인해 다양한 스페인어 사용자의 정보 요구를 포괄합니다.

- **Performance Highlights**: MessIRve는 스페인어 정보 검색 시스템 개발을 위한 새로운 기준을 제공합니다. 기존 데이터셋과 비교하여 MessIRve는 더 다양한 주제를 다루며, 스페인어 IR 연구의 발전을 도모할 수 있는 강력한 도구로 작용할 것입니다.



### A Comprehensive Comparison Between ANNs and KANs For Classifying EEG Alzheimer's Data (https://arxiv.org/abs/2409.05989)
- **What's New**: 이번 연구에서는 Electroencephalogram (EEG) 기술을 활용하여 알츠하이머병 조기 진단의 정확성을 높이기 위해, 인공신경망(ANN)과 Kolmogorov-Arnold Networks (KAN)의 성능을 비교했습니다. KAN은 가중치 및 활성화 함수를 보다 유연하게 관리하며, 이전 모델들보다 높은 정확성을 보여줍니다.

- **Technical Details**: EEG는 비침습적인 방법으로 뇌의 전기 신호를 기록하며, 이를 통해 인지 장애를 진단하는 데 유용한 데이터를 제공합니다. KAN은 ANN에 비해 학습 가능한 활성화 함수를 사용해 더 높은 유연성과 정확성을 가지고 있으며, KAN 기반의 알고리즘은 다양한 학습률, 에포크(epoch), 노드 수에 대한 파라메트릭 실험을 통해 성능을 평가했습니다.

- **Performance Highlights**: 연구 결과, ANN이 EEG 신호에서 알츠하이머병을 예측하는 데 더욱 정확한 성능을 보였으나, KAN의 접근 방식이 향후 개선의 여지가 있음을 나타내었습니다. 알츠하이머병으로 진단된 36명과 건강한 대조군 29명의 EEG 신호를 분석하여, 뇌의 특정 영역에서 알츠하이머병에 관련된 변화를 추적했습니다.



### DeepFM-Crispr: Prediction of CRISPR On-Target Effects via Deep Learning (https://arxiv.org/abs/2409.05938)
Comments:
          11 page, 2 figures, accepted to ICMLA 2024

- **What's New**: 본 연구에서는 Cas13d를 위한 새로운 딥러닝 모델인 DeepFM-Crispr를 도입했습니다. 이 모델은 sgRNA의 On-target 효율성 및 Off-target 효과를 예측하기 위해 고안되었으며, 진화적 및 구조적 데이터의 포괄적인 표현을 활용하여 RNA 이차 구조 예측을 개선합니다.

- **Technical Details**: DeepFM-Crispr는 대용량 언어 모델과 transformer 기반 아키텍처를 활용하여 sgRNA의 예측을 수행합니다. 이 모델은 RNA-FM, ResNet 및 Seq-DenseNet 아키텍처를 통합하여 복잡한 생물학적 데이터를 처리하고 예측 점수를 생성합니다. 구체적으로, sgRNA의 이차 구조를 예측하는 데 있어서, ResNet 모델을 사용하여 각 뉴클레오타이드 위치의 결합 여부(1) 또는 비결합 여부(0)를 결정합니다.

- **Performance Highlights**: DeepFM-Crispr는 기존의 전통적인 모델과 최신 딥러닝 방법론보다 예측 정확성과 신뢰성 측면에서 우수한 성능을 보였습니다. 22,599개의 Cas13d sgRNA 데이터셋에 대한 검증 결과가 이를 뒷받침합니다.



### Alt-MoE: Multimodal Alignment via Alternating Optimization of Multi-directional MoE with Unimodal Models (https://arxiv.org/abs/2409.05929)
Comments:
          work in progress

- **What's New**: 본 논문에서는 다중 모달 정렬(multi-modal alignment)을 위한 새로운 훈련 프레임워크인 Alt-MoE를 제안합니다. Alt-MoE는 Mixture of Experts (MoE) 구조를 통해 다양한 모달리티를 연결하여 지식을 통합하며, 유일한 방향의 정렬 전략을 사용하여 양방향 정렬로 수렴합니다.

- **Technical Details**: Alt-MoE의 주요 요소는 다중 방향 MoE를 통합하여 크로스 모달 연결을 구현하고, 서로 다른 모달리티의 표현을 일치시키기 위해 Alternating Gradient Descent (AGD) 방법을 사용합니다. 이러한 프레임워크는 잠재 공간(latent space)에서 작동하며, 복잡한 다중 모달 작업을 간단한 단일 방향 서브 작업으로 분해합니다.

- **Performance Highlights**: Alt-MoE는 LLAMA3, Qwen2, DINOv2와 같은 여러 우수한 유니모달 모델에서 검증되었으며, 다양한 다운스트림 작업과 데이터셋에서 경쟁력 있는 결과를 달성했습니다. 또한, 대규모 온라인 다중 모달 검색 작업을 수행할 수 있는 능력이 뛰어납니다.



### Assessing SPARQL capabilities of Large Language Models (https://arxiv.org/abs/2409.05925)
Comments:
          peer reviewed publication at NLP4KGc @ Semantics 2024, see this https URL

- **What's New**: 이 논문에서는 대규모 언어 모델(LLM)과 지식 그래프(KG)의 통합 가능성에 대해 다루고 있으며, 특히 자동화된 벤치마크 작업을 통해 SPARQL SELECT 쿼리를 처리하는 LLM의 기본 능력을 평가합니다.

- **Technical Details**: LLM-KG-Bench 프레임워크를 활용하여 SPARQL SELECT 쿼리와 관련된 다양한 벤치마크 작업을 구현하였고, 주요 작업 유형은 SPARQL 구문 수정(SSF), 텍스트를 SPARQL로 변환(T2S), SPARQL에서 답변으로(S2A), 텍스트에서 답변으로(T2A)입니다.

- **Performance Highlights**: 현재 평가된 LLM들 중 가장 우수한 성능을 보이는 모델 조차도 여러 경우에서 의미론적으로 올바른 SPARQL SELECT 쿼리를 생성하는 데 어려움을 겪고 있으며, LLM의 성능은 특정 모델 및 작업의 복잡성에 크게 의존합니다.



### $\mathbb{USCD}$: Improving Code Generation of LLMs by Uncertainty-Aware Selective Contrastive Decoding (https://arxiv.org/abs/2409.05923)
Comments:
          13pages,8 figures

- **What's New**: 본 논문에서는 LLMs (Large Language Models)의 코드 생성 능력을 향상시키기 위해 불확실성 인식 선택적 대조 디코딩 기법인 USCD(uncertainty-aware selective contrastive decoding)를 제안합니다. 이 기법은 코드 생성 시 발생하는 출력 노이즈의 영향을 줄이는데 초점을 맞추었습니다.

- **Technical Details**: USCD 기법은 'lame prompt'라는 부정적인 프롬프트를 이용해 입력-출력 예제를 제거하여 출력 노이즈를 유도하고, 이후 예측 분포의 불확실성을 바탕으로 해당 노이즈를 선택적으로 제거합니다. 이 과정은 inference-only 방식으로 이루어져, 외부 피드백 없이도 유연하게 적용 가능합니다.

- **Performance Highlights**: 다양한 LLMs (예: Inocder-6b, CodeLlama-7b 등)에 대한 실험 결과, HumanEval, MBPP, MultiPL-E 벤치마크에서 USCD는 코드 생성의 pass@1 점수를 평균 16.59% 향상시키는 것으로 나타났습니다.



### STLLM-DF: A Spatial-Temporal Large Language Model with Diffusion for Enhanced Multi-Mode Traffic System Forecasting (https://arxiv.org/abs/2409.05921)
Comments:
          26 pages, 11 figures

- **What's New**: 이번 연구에서는 Intelligent Transportation Systems (ITS)에서의 데이터 손실 문제와 다중 순차 작업 처리를 위한 Spatial-Temporal Large Language Model Diffusion (STLLM-DF)라는 혁신적인 모델을 제안합니다. 이 모델은 Denoising Diffusion Probabilistic Models (DDPMs)와 Large Language Models (LLMs)를 활용하여 다중 작업의 운송 예측 정확성을 향상시킵니다.

- **Technical Details**: STLLM-DF는 강력한 노이즈 제거 기능을 지닌 DDPM을 사용하여 시공간적 (spatial-temporal) 관계를 포함한 데이터를 처리합니다. 또한, LLM을 통해 높은 수준의 특징 추출이 가능하여, 입력 데이터의 노이즈 영향을 최소화하고 ITS의 예측 정확성을 크게 개선합니다.

- **Performance Highlights**: STLLM-DF 모델은 기존 모델들에 비해 평균 2.40%의 MAE 감소, 4.50%의 RMSE 감소 및 1.51%의 MAPE 감소를 달성하며, 다중 작업 성능이 연속적으로 향상됨을 보여주었습니다.



### KModels: Unlocking AI for Business Applications (https://arxiv.org/abs/2409.05919)
- **What's New**: 본 논문은 KModels라는 새로운 프레임워크를 제안하여, 비즈니스 애플리케이션에 AI 통합을 위한 기술적 요구사항을 줄이면서도 품질이나 포괄성을 유지하는 방법을 모색합니다. KModels는 모델 개발자가 클라이언트의 로컬 환경에서 모델을 훈련하고 배포할 수 있도록 지원합니다.

- **Technical Details**: KModels는 Kubeflow Pipelines와 KServe와 같은 검증된 라이브러리와 플랫폼을 활용하여 AI 모델의 배포 및 생애 주기 관리를 간소화합니다. 이를 통해 비기술 사용자들이 복잡한 생산 환경 문제를 고려하지 않고도 AI 모델을 배포하고 운영할 수 있습니다. KModels는 클라이언트의 데이터 센터에서 로컬 데이터로 훈련된 AI 모델을 성공적으로 배포하는 사례를 보여줍니다.

- **Performance Highlights**: KModels를 통해 세 개의 AI 모델이 기존 작업 관리 시스템 내에 성공적으로 배포되었으며, 그 중 한 모델은 작업 지시서의 Failure Code 명세 정확도를 46%에서 83%로 향상시켰습니다. 이는 접근 가능하고 로컬화된 AI 솔루션의 상당한 이점을 보여줍니다.



### Unlocking Potential Binders: Multimodal Pretraining DEL-Fusion for Denoising DNA-Encoded Libraries (https://arxiv.org/abs/2409.05916)
- **What's New**: 이번 연구에서는 Multimodal Pretraining DEL-Fusion 모델(MPDF)을 제안하여 DNA-인코딩 라이브러리(DNA-encoded library, DEL)의 노이즈 문제를 해결하고, 다양한 스케일에서의 화합물 정보를 통합하여 데이터의 품질을 향상시킵니다.

- **Technical Details**: MPDF 모델은 화합물 그래프, ECFP 및 텍스트 설명 간의 대조적 목표를 기반으로 한 사전 훈련(Pretraining) 작업을 설계합니다. 또한, 화합물 정보를 원자, 하위 분자, 분자 수준에서 통합하기 위해 DEL-퓨전 신경망(DEL-fusion neural network)을 개발하여 복합적인 특징을 명확히 합니다.

- **Performance Highlights**: 세 가지 DEL 데이터셋(P, A, OA)에서 평가된 결과, MPDF 모델은 기존의 방법들과 비교하여 데이터 처리 및 분석에서 우수한 성능을 보였습니다. 특히, 고친화성 분자를 식별하는 데 있어 새로운 통찰을 제공합니다.



### Property Neurons in Self-Supervised Speech Transformers (https://arxiv.org/abs/2409.05910)
Comments:
          Accepted by SLT 2024

- **What's New**: 이 연구는 self-supervised speech Transformers의 feedforward layers에서 특정 speech 속성과 관련된 뉴런을 식별하여, 이러한 속성의 저장 방식 및 모델 편집(model editing)과 모델 가지치기(model pruning)에 기여하는 방법론을 제안합니다.

- **Technical Details**: 주요 뉴런(property neurons)을 정의하고, 이들 뉴런이 음성 인식 시 나타나는 성별(gender), 음성의 음조(pitch), 음소(phones)와 어떻게 연관되는지 분석합니다. 이를 위해 다양한 그룹의 뉴런을 구성하고, 해당 뉴런이 특정 속성과 얼마나 높은 확률로 공존하는지를 계산하여 필요한 뉴런을 필터링합니다.

- **Performance Highlights**: 모델에서 특정 속성과 관련된 그룹 뉴런을 제거할 경우, 모델이 특정 인식을 실패하는 현상이 나타났으며, 이러한 뉴런을 보호하면서 가지치기를 진행할 경우, 모델 압축이 현저하게 개선됨을 시연하였습니다.



### Programming Refusal with Conditional Activation Steering (https://arxiv.org/abs/2409.05907)
- **What's New**: 이 논문에서는 조건부 활성화 조정(Conditional Activation Steering, CAST) 방법을 제안하여 LLM의 응답 행동을 보다 세밀하게 제어할 수 있는 새로운 접근 방식을 소개합니다.

- **Technical Details**: CAST는 추론(inference) 중 LLM의 활성화 패턴을 분석하여 입력 맥락에 따라 활성화 조정을 선택적으로 적용하거나 보류하는 방법입니다. 다양한 프롬프트(prompt) 카테고리가 모델의 숨겨진 상태(hidden state)에서 서로 다른 패턴을 활성화한다는 관찰에 기초하고 있습니다.

- **Performance Highlights**: CAST 방법을 사용하면 '만약 입력이 증오 발언(hate speech) 또는 성인 콘텐츠(adult content)에 관한 것이라면 거부하라' 또는 '입력이 법률 조언에 관한 것이 아니면 거부하라'와 같은 규칙을 통해 LLM의 응답을 체계적으로 제어할 수 있습니다.



### Simplex-enabled Safe Continual Learning Machin (https://arxiv.org/abs/2409.05898)
- **What's New**: 본 논문은 안전이 중요한 자율 시스템을 위한 SeC-Learning Machine을 제안합니다. 이 시스템은 Simplex 기반의 지속 가능한 학습을 통해 안전성을 보장하며, Deep Reinforcement Learning (DRL)과 물리 기반 제어를 결합한 Phy-DRL 기술을 활용합니다.

- **Technical Details**: SeC-Learning Machine은 HP (High Performance) 학생, HA (High Assurance) 교사, 코디네이터로 구성되어 있습니다. HP 학생은 훈련 완료된 Phy-DRL 모델로, 실제 환경에서 안전한 행동 정책을 학습합니다. HA 교사는 검증된 설계를 바탕으로 한 안전 보장을 지원하며, 코디네이터는 HP 학생과 HA 교사 간의 상호작용을 촉진합니다.

- **Performance Highlights**: 실제 폴 시스템과 사족보행 로봇에 대한 실험을 통해 SeC-Learning Machine의 안전성과 성능 특성이 입증되었습니다. 이 시스템은 Sim2Real 격차를 해결하며, 알려지지 않은 상황에 대한 내성을 학습할 수 있습니다.



### MA-CDMR: An Intelligent Cross-domain Multicast Routing Method based on Multiagent Deep Reinforcement Learning in Multi-domain SDWN (https://arxiv.org/abs/2409.05888)
- **What's New**: 이 논문은 여러 컨트롤러가 있는 소프트웨어 정의 무선 네트워크(SDWN)에서의 교차 도메인 멀티캐스트 라우팅 문제를 해결하기 위한 다중 에이전트 딥 강화 학습 기반 방법론을 제안합니다. 이는 NP-hard 최적화 문제로, 네트워크 크기가 증가함에 따라 기존 해결책의 한계를 극복하고자 합니다.

- **Technical Details**: 제안된 방법은 멀티컨트롤러 통신 메커니즘 및 멀티캐스트 그룹 관리 모듈을 포함하여, 서로 다른 제어 도메인 간의 네트워크 정보를 전달하고 동기화합니다. 또한, 최적의 교차 도메인 멀티캐스트 트리 구성에 대한 이론적 분석이 포함되어 있으며, 각 컨트롤러를 위한 에이전트가 설정되고 여러 에이전트 간의 협력 메커니즘이 설계되었습니다. 이를 통해 교차 도메인 멀티캐스트 라우팅의 최적화를 도모합니다.

- **Performance Highlights**: 제안된 멀티 에이전트 강화 학습 방법은 온라인 및 오프라인 교육을 결합하여 실시간 환경 의존성을 줄이고, 여러 에이전트의 수렴 속도를 증가시키는 성과를 보여줍니다.



### FairEvalLLM. A Comprehensive Framework for Benchmarking Fairness in Large Language Model Recommender Systems (https://arxiv.org/abs/2405.02219)
- **What's New**: 이 논문은 대규모 언어 모델(Large Language Models, LLMs)에 기반한 추천 시스템(RecLLMs)의 공정성을 평가하기 위한 프레임워크를 제시합니다. 다양한 공정성 차원을 포괄하는 통합 접근 방식의 필요성을 다루며, 사용자 속성에 대한 민감도, 내재적 공정성(intrinsic fairness), 기반 이점에 대한 공정성 논의 등을 포함합니다.

- **Technical Details**: 제안된 프레임워크는 반사실적 평가(counterfactual evaluations)를 도입하고 다양한 사용자 그룹 고려사항을 통합하여 RecLLMs의 공정성 평가를 향상시킵니다. 또한, 인구통계학적 데이터와 사용자 선호도, 최근 상호작용을 기반으로 한 정보성 사용자 프로필(informative user profiles) 생성 방법을 구조화하여 개인화(personalization)를 강화하는 데 필수적임을 주장합니다.

- **Performance Highlights**: 두 개의 데이터셋(LastFM-1K 및 ML-1M)에서 80명의 사용자 샘플을 통해 50가지의 시나리오를 시험하여 4000개의 추천을 생성하였습니다. 민감한 속성이 포함된 시나리오에서는 공정성 문제는 크지 않았지만, 내재적 공정성 측면에서 인구 집단 간의 불공정성이 여전히 중요하게 남아있음을 발견했습니다.



### Generative User-Experience Research for Developing Domain-specific Natural Language Processing Applications (https://arxiv.org/abs/2306.16143)
- **What's New**: 이 논문은 기계 학습(ML) 및 자연어 처리(NLP) 애플리케이션의 개발 과정에 생성적 사용자 경험(UX) 연구를 통합하는 새로운 방법론을 제안합니다. 이 방법론은 초기 프로토타입 개발 단계에서 도메인 사용자와의 협업을 통해 사용자의 필요를 이해하고 시스템 유용성을 평가합니다.

- **Technical Details**: 제안된 방법론은 데이터 기반 접근 방식과 사용자 중심 접근 방식을 결합하여 도메인 특화 NLP 애플리케이션의 전체 주기를 고려합니다. 이는 탐색적 데이터 분석과 사용자 인터뷰, 현장 관찰을 통해 아이디어를 생성하고, 중요 사용자 피드백을 통해 프로토타입을 구현하며, 프로토타입과의 상호작용 전후에 사용자 유용성을 평가하는 체계를 포함합니다.

- **Performance Highlights**: 사례 연구의 주요 결과는 도메인 전문가를 초기 개발 단계에 참여시키는 것이 최종 NLP 애플리케이션에 대한 그들의 관심과 신뢰를 증가시켰다는 것입니다. 이 방법론은 데이터와 사용자 주도의 기회를 효율적으로 고려하여 NLP 애플리케이션 개발에 중요한 요소로 작용할 수 있습니다.



### NeurLZ: On Enhancing Lossy Compression Performance based on Error-Controlled Neural Learning for Scientific Data (https://arxiv.org/abs/2409.05785)
- **What's New**: NeurLZ는 과학 데이터를 위한 새로운 방법론으로, 크로스-필드 학습 기법과 경량 스킵 DNN 모델을 통합하여 에러를 관리하면서 손실 압축 품질을 크게 향상시킵니다.

- **Technical Details**: NeurLZ는 세 가지 주요 기여를 통해 다양한 데이터 환경에서의 압축 성능을 극대화합니다: (1) 고충실도 세부정보 유지를 위한 경량 스킵 모델 설계, (2) 크로스-필드 학습 접근 방식을 통한 데이터 예측 정확도 향상, (3) 사용자 요구 사항에 따른 엄격한 오류 경계 제공.

- **Performance Highlights**: NeurLZ는 Nyx, Miranda, Hurricane 데이터셋에 대한 테스트에서 최대 90%의 비트 전송률 감소를 달성하였으며, 기존 최고의 방법과 비교하여 동등한 데이터 왜곡 아래에서 압축 성능이 크게 향상되었습니다.



### What Did My Car Say? Impact of Autonomous Vehicle Explanation Errors and Driving Context On Comfort, Reliance, Satisfaction, and Driving Confidenc (https://arxiv.org/abs/2409.05731)
Comments:
          23 pages, 4 figures

- **What's New**: 이 연구는 자율주행차(Autonomous Vehicles, AV)의 설명 오류가 승객의 신뢰와 의존성에 미치는 영향을 조사했습니다. 자율주행 기능이 완전히 동일한 주행 상황에서도 설명 오류가 차량의 주행 능력에 대한 평가를 감소시키는 놀라운 결과를 보였습니다.

- **Technical Details**: 232명의 참가자를 대상으로 한 시뮬레이션 기반 운전 연구에서, 설명 오류의 영향을 평가했습니다. 주요 결과 기준은 AV에 대한 의존 편안함(comfort in relying on the AV), 통제에 대한 선호(preference for control), AV의 운전 능력에 대한 신뢰(confidence in the AV's ability), 설명 만족도(explanation satisfaction)입니다. 상황적 피해(perceived harm)와 운전 난이도(driving difficulty)가 결과에 직접적인 영향을 미쳤고, 오류와 결과 간의 관계도 영향을 받았습니다.

- **Performance Highlights**: 연구 결과, 설명 오류는 모든 결과에 부정적인 영향을 미쳤으며, 특히 오류의 심각성과 잠재적 피해가 부정적인 영향을 확대했습니다. 높은 전문성을 가진 참가자일수록 AV에 대한 신뢰가 더 높았고, 그에 따라 더 긍정적인 평가와 연결되었습니다.




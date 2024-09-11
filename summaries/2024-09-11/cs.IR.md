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




New uploads on arXiv(cs.CL)

### LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks (https://arxiv.org/abs/2412.15204)
Comments:
          25 pages, 13 figures

- **What's New**: 이 논문에서는 LongBench v2라는 새로운 벤치마크가 소개됩니다. 이 벤치마크는 LLMs(대형 언어 모델)가 실제 멀티태스크 문제를 처리하는 데 필요한 깊은 이해와 추론 능력을 평가하기 위해 설계되었습니다. LongBench v2는 8천에서 200만 단어까지의 문맥을 가진 503개의 도전적인 객관식 질문으로 구성되어 있습니다.

- **Technical Details**: 이 벤치마크는 단일 문서 QA, 다중 문서 QA, 장기간 맥락 학습, 장기 대화 역사 이해, 코드 저장소 이해 및 장기 구조화 데이터 이해의 여섯 가지 주요 작업 범주를 포함합니다. 데이터는 전문 배경이 다양한 거의 100명의 고학력 개인으로부터 수집하였으며, 품질과 난이도를 유지하기 위해 자동화된 리뷰와 수동 리뷰 과정을 모두 사용하였습니다.

- **Performance Highlights**: 인간 전문가들은 15분의 시간 제약 하에 53.7%의 정확도만 달성했으며, 가장 성능이 좋은 모델도 직접 질문에 대답했을 때 50.1%의 정확도에 그쳤습니다. 반면, 더 긴 추론을 포함하는 o1-preview 모델은 57.7%의 정확도를 기록하여 인간의 기준을 4% 초과함으로써, LongBench v2의 장기 맥락 문제를 해결하기 위해서 향상된 추론 능력과 추론 시간 컴퓨팅을 확대하는 것이 중요함을 보여주고 있습니다.



### MMLU-CF: A Contamination-free Multi-task Language Understanding Benchmark (https://arxiv.org/abs/2412.15194)
- **What's New**: 이번 논문에서는 대규모 언어 모델(LLMs)의 평가를 개선하기 위해 MMLU-CF라는 새로운 다중 선택 질문(multiple-choice question, MCQ) 벤치마크를 제안합니다. MMLU-CF는 데이터 오염(contamination)을 피하고 모델이 진정한 세계 지식을 이해하는 능력을 테스트하는 데 초점을 맞추고 있습니다. 이 벤치마크는 10,000개의 질문으로 구성된 테스트 세트와 10,000개의 질문으로 구성된 검증 세트를 갖추고 있으며, 테스트 세트는 비공개로 유지되어 신뢰성을 높이고 있습니다.

- **Technical Details**: MMLU-CF는 MCQ 수집, 정리, 난이도 샘플링, LLM 점검, 오염 방지를 포함하여 총 다섯 가지 주요 처리 단계를 통해 데이터가 오염되는 위험을 최소화합니다. 특히 질문의 의미를 변경하지 않고 재구성하는 세 가지 규칙을 적용하여 의도하지 않은 데이터 유출을 방지하고 있습니다. 모델이 과거에 질문을 기억하는 경우, 질문을 재구성하면 모델의 판단에 영향을 미치게 되므로 이러한 접근법은 타당합니다.

- **Performance Highlights**: MMLU-CF에 대한 평가 결과, 강력한 GPT-4o 모델이 5-shot 테스트에서 73.4%, 0-shot 테스트에서 71.9%의 점수를 기록했습니다. 이는 이전의 MMLU에서의 88.0%에 비해 상당히 낮은 결과로, MMLU-CF의 도전적인 성격을 잘 보여줍니다. 모델 간 비교에서도 Qwen2.5-72B-instruct가 71.6%, Llama-3.3-70B-instruct가 68.8%로 우수한 결과를 보여 각 모델의 성능 차이를 확인할 수 있습니다.



### Face the Facts! Evaluating RAG-based Fact-checking Pipelines in Realistic Settings (https://arxiv.org/abs/2412.15189)
Comments:
          Code and data at this https URL

- **What's New**: 이번 연구에서는 Retrieval-Augmented Generation (RAG) 패러다임을 기반으로 한 자동 팩트체크의 최신 파이프라인의 여러 제약을 해소하는 작업을 진행했습니다. RAG 기반의 방법을 통해 짧은 텍스트로 주장에 대한 진위 여부를 논의하는 판결을 생성하는 성능을 현실적인 시나리오에서 평가합니다. 연구 결과, LLM 기반의 검색자가 다른 방법들보다 우수하나, 이질적인 지식 기반에 대한 처리에서 어려움을 겪는 것으로 나타났습니다.

- **Technical Details**: 연구에서는 다양한 스타일의 주장 및 각기 다른 정보 검색 접근법과 LLM 구성으로 RAG 파이프라인의 평가를 수행했습니다. 특히, SMP 스타일과 감정적 요소가 결합된 주장을 포함하여 세 가지 주장 스타일, 그리고 LLM의 세 가지 설정인 제로샷(zero-shot), 퓨샷(few-shot), 그리고 파인튜닝(fine-tuned) 모델을 사용하였습니다. 이러한 다양한 실험을 통해 각기 다른 입력 스타일과 처리 조건이 결과에 미치는 영향을 정량적으로 분석하였습니다.

- **Performance Highlights**: LLM 기반의 검색자는 여전히 검색 접근법에서 다른 기술들보다 일관되게 뛰어난 성능을 보여주었으나, 이 질적인 지식 기반에 대해서는 도전 과제가 있었습니다. 큰 모델은 판결의 신뢰성과 일관성이 뛰어난 반면, 작은 모델은 더 나은 맥락 유지를 보여주었습니다. 인간 평가자들은 제로/원샷 전략으로 생성된 판결이 정보 전달력에서 우수하다고 평가했으며, 감정 정렬에 있어서는 파인튜닝된 모델을 선호했습니다.



### LlamaFusion: Adapting Pretrained Language Models for Multimodal Generation (https://arxiv.org/abs/2412.15188)
- **What's New**: LlamaFusion introduces a novel framework that enhances pretrained text-only large language models (LLMs) like Llama-3 with multimodal generative capabilities, allowing them to process and generate both text and images. 이 프레임워크는 Llama-3의 가중치를 활용하고, 이미지 처리를 위한 추가적인 transformer 모듈을 도입하여 텍스트와 이미지를 각각 처리합니다. 이를 통해 LlamaFusion은 텍스트 전용 모델의 언어 능력을 유지하면서도 강력한 시각 이해 및 생성 기능을 개발할 수 있게끔 합니다.

- **Technical Details**: LlamaFusion은 별도의 모듈에서 텍스트와 이미지를 각각 처리하도록 구성되어 있으며, 공통의 self-attention 층을 통해 두 개의 모달리티 간 상호작용을 가능하게 합니다. 훈련 과정에서 텍스트 관련 모듈은 고정되고 이미지 관련 모듈만 훈련하여, 이전의 언어 능력을 손상시키지 않으면서도 시각적 이해를 키웁니다. 또한, LlamaFusion은 기존의 text-only LLM에서 이미지를 이해하고 생성하는 능력을 효과적으로 훈련할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과 LlamaFusion은 이미지 이해에서 20%, 이미지 생성에서 3.6% 개선된 성능을 보이며, 전량 50%의 FLOPs만으로도 Llama-3의 언어 성능을 유지합니다. LlamaFusion의 성능은 Transfusion 모델에 비해 11.6% 더 우수하여 기존 시각-언어 모델을 적응시킬 수 있는 가능성을 보여줍니다. 이를 통해 LlamaFusion은 멀티모달 모델 개발의 효율적인 방향을 제시합니다.



### Language Models as Continuous Self-Evolving Data Engineers (https://arxiv.org/abs/2412.15151)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 데이터 자생적 생성 및 세척의 새로운 패러다임인 LANCE를 제안합니다. LANCE는 LLM이 스스로 데이터 생성, 검토 및 주석을 달 수 있도록 하여, 높은 품질의 데이터 확보에 있어 기존 방식에 비해 많은 시간과 비용을 절감합니다. 이 최초의 자율 데이터 엔지니어링 접근법은 LLM이 지속적으로 자기 진화를 이룰 수 있는 가능성을 보여줍니다.

- **Technical Details**: LANCE는 LLM이 기존의 데이터셋을 검토하고, 낮은 품질의 데이터를 보완하며, 선호 정보를 갖춘 고품질의 데이터를 구조화하는 과정을 통해 학습합니다. 이를 위해 모델은 텍스트 명령과 응답을 생성함으로써 응답의 질을 개선하고, 자율적으로 새로운 데이터를 수집하여 반복적으로 성능을 향상시킬 수 있습니다. 이러한 방법은 사람의 개입 없이 모델 스스로 데이터 생성을 완전한 사이클로 수행할 수 있게 합니다.

- **Performance Highlights**: LANCE는 Qwen2 모델을 대상으로 한 여러 작업에서 평균 점수를 3.36, 2.70만큼 향상시킴으로써 성능 개선을 입증합니다. LANCE의 반복적인 데이터 자생적 공정은  다양한 작업에서 LLM의 지능을 지속적으로 향상시키는 데 기여하며, 이는 기존의 감독형 모델학습(Supervised Fine-Tuning)과 비교해 더 일관된 결과로 나타납니다. 이는 미래의 초지능 시스템 개발에 큰 기여를 할 것으로 기대됩니다.



### Adaptive Pruning for Large Language Models with Structural Importance Awareness (https://arxiv.org/abs/2412.15127)
Comments:
          12 pages, 6 figures, 12 tables

- **What's New**: 최근 대형 언어 모델(LLMs)의 발전은 자연어 이해와 생성 능력을 크게 향상시켰습니다. 그러나 이 모델들은 고성능의 컴퓨팅 및 저장 자원을 요구하기 때문에 자원이 제한된 엣지 기기에서 배포하기 까다롭습니다. 이를 해결하기 위해, 본 논문에서는 구조적 인식 적응형 가지치기(Structure-aware Adaptive Pruning, SAAP) 방법을 제안하여 모델 성능을 유지하면서 계산 및 메모리 비용을 줄이는 새로운 접근 방식을 소개합니다.

- **Technical Details**: SAAP는 모든 결합된 구조의 중요성을 평가하기 위한 적응형 중요도 융합 메트릭을 정의합니다. 이는 불확실성을 고려하여 특정 계층이 가지치기되어야 하는지를 정하기 위해 모듈의 중요도를 순위로 매깁니다. 또한 SAAP는 추론 효율을 높이기 위해 새로운 그룹 내 세부 조정 방식을 개발하였으며, 다양한 LLMs에 걸쳐 실험을 수행하여 성능을 평가했습니다.

- **Performance Highlights**: 실험 결과, SAAP 방법은 여러 최신 기준선 방법들보다 우수한 성능을 보여주었으며, LLaMA-7B, Vicuna-7B, LLaMA-13B 모델에서 각각 2.17%, 2.37%, 2.39%의 정확도 향상을 기록했습니다. 또한 SAAP는 토큰 생성 속도를 5% 개선하여 자원이 제한된 환경에서의 실용적인 이점을 보여주고 있습니다.



### Outcome-Refining Process Supervision for Code Generation (https://arxiv.org/abs/2412.15118)
Comments:
          18 pages, 5 figures, Code: this https URL

- **What's New**: 이 논문에서는 Outcome-Refining Process Supervision, 즉 ORPS라는 새로운 패러다임을 제안합니다. 이 방법은 아웃컴(Outcome) 개선 그 자체를 감독해야 할 프로세스로 간주하며, 이를 통해 모델이 다양한 솔루션 경로를 탐색할 수 있도록 합니다. 연구 결과, 이 접근법은 더 작은 모델에서도 높은 성공률과 성능 지표를 달성할 수 있도록 해줍니다. 이를 통해 모델 개발에서의 복잡한 문제를 해결하는 데 중요한 실마리를 제공합니다.

- **Technical Details**: ORPS 프레임워크는 실행 피드백을 활용하여 추론 단계의 감독을 지지합니다. 또한, 트리 구조의 탐색 공간을 통해 모델이 여러 해결 전략을 동시에 유지하며 탐색할 수 있도록 돕습니다. 연구에서는 LLMs(대형 언어 모델)의 복잡한 프로그래밍 작업에서의 성능을 개선하기 위해 실행 신호를 강력한 기준으로 사용합니다. 이로 인해 기존의 학습된 보상 모델(PRMs)의 필요성을 줄이고 더욱 신뢰할 수 있는 검증 시스템을 구축할 수 있습니다.

- **Performance Highlights**: 실험 결과, ORPS는 세 가지 데이터셋과 다섯 개의 모델에서 평균 26.9%의 정확도 향상과 42.2%의 효율성 증가를 보여줍니다. 이 접근법은 적은 데이터로도 신뢰할 수 있는 검증을 제공하며, 특히 복잡한 작업에서 기존 방법들이 어려움을 겪는 지역적 개선 이상의 혁신을 보여줍니다. 이러한 결과는 제공된 구조화된 추론 공간과 구체적인 검증 신호가 복잡한 프로그래밍 작업을 해결하는 데 필수적임을 시사합니다.



### Qwen2.5 Technical Repor (https://arxiv.org/abs/2412.15115)
- **What's New**: 이번 보고서에서는 다양한 요구를 충족하기 위해 설계된 대규모 언어 모델(Qwen2.5)을 소개합니다. Qwen 2.5는 이전 버전보다 사전 학습(pre-training) 및 사후 학습(post-training) 단계에서 크게 개선되었습니다. 사전 학습 단계에서 고품질 데이터셋의 규모를 7조 개 토큰에서 18조 개 토큰으로 확장했습니다.

- **Technical Details**: 사후 학습 단계에서는 100만 개 이상의 샘플을 활용한 정교한 감독된 파인튜닝(supervised finetuning)과 다단계 강화 학습(multistage reinforcement learning)을 적용하였습니다. 이러한 기법은 인간의 선호도를 개선하고 긴 텍스트 생성, 구조적 데이터 분석, 지침 따르기에서 두드러진 향상을 보여줍니다. 또한, Qwen2.5 LLM 시리즈는 다양한 사용 사례를 효과적으로 처리하기 위해 여러 가지 크기로 제공됩니다.

- **Performance Highlights**: Qwen2.5는 언어 이해, 추론, 수학, 코딩, 인간 선호 정렬 등 다양한 벤치마크에서 우수한 성능을 입증하였습니다. 특히, 공개된 Qwen2.5-72B-Instruct 모델은 여러 오픈 및 폐쇄 모델을 초월하며, 약 5배 더 큰 Llama-3-405B-Instruct 모델과 경쟁력 있는 성과를 보여주었습니다. Qwen2.5-Turbo와 Qwen2.5-Plus는 각각 GPT-4o-mini 및 GPT-4o와 경쟁하면서 우수한 비용 효율성을 제공합니다.



### Review-Then-Refine: A Dynamic Framework for Multi-Hop Question Answering with Temporal Adaptability (https://arxiv.org/abs/2412.15101)
Comments:
          20 pages, 2 figures

- **What's New**: 이 논문에서는 다중 홉 질문 응답(multi-hop question answering, QA)에서 시간 정보를 효과적으로 처리하기 위해 'review-then-refine'이라는 새로운 프레임워크를 제안합니다. 기존의 retrieve-then-read 방식의 한계를 극복하고자 하며, 이는 시간 정보와 관련된 서브 쿼리의 동적 재작성(dynamic rewriting)을 통해 보다 정확한 정보 검색과 추론을 가능하게 합니다. 또한 불필요한 검색을 최소화함으로써 외부에서 발생할 수 있는 'hallucinations'를 줄이는 방안을 모색합니다.

- **Technical Details**: 제안된 프레임워크는 두 단계로 구성되어 있습니다. 첫 번째 단계인 리뷰(review) 단계에서는 원래 쿼리를 더 간단한 서브 쿼리로 분해하여 보다 정확한 시간 정보를 추가합니다. 이후에는 적응형 검색(adaptive retrieval) 메커니즘을 구현하여 각 검색 단계의 필요성을 평가하고 정보를 검색합니다. 두 번째 단계인 정제(refine) 단계에서는 검색된 외부 데이터를 모델의 기존 지식과 통합하여 최종 답변이 정확하고 일관되도록 합니다.

- **Performance Highlights**: 다양한 데이터셋을 통해 수행된 실험 결과는 제안한 review-then-refine 프레임워크의 효과성을 강조합니다. 이 방법은 다중 홉 QA에서 LLM의 성능을 크게 향상시키는 잠재력을 가지고 있으며, 특히 시간 민감한 쿼리 처리에서 기존 방법들과 비교해 우수한 성능을 보입니다. 표에 나타난 바와 같이, 기존 방법들보다 여러 측면에서 두드러진 개선을 이루었습니다.



### AceMath: Advancing Frontier Math Reasoning with Post-Training and Reward Modeling (https://arxiv.org/abs/2412.15084)
- **What's New**: 이 논문에서는 복잡한 수학 문제를 해결하는 데 탁월한 수학 모델 모음인 AceMath를 소개합니다. 이 모델은 생성된 솔루션을 평가하고 올바른 것을 신뢰성 있게 식별할 수 있는 효과적인 reward 모델을 포함합니다. 연구진은 일반 도메인에서 경쟁력을 갖춘 성능을 달성한 후, 신중히 선별된 프롬프트와 합성된 응답을 사용하여 수학 도메인에 맞춰 조정하는 감독적 세부 조정(Supervised Fine-Tuning, SFT) 프로세스를 제안합니다.

- **Technical Details**: AceMath-72B-Instruct 모델은 Qwen2.5-Math-72B-Instruct, GPT-4o 및 Claude-3.5 Sonnet과 비교하여 뛰어난 성능을 보입니다. 이를 위해 AceMath-RewardBench라는 포괄적이고 견고한 벤치마크를 구축하여 다양한 문제와 난이도 수준에 걸쳐 수학 reward 모델을 평가합니다. 연구자들은 신뢰성 있는 수학 reward 모델을 구축하기 위한 체계적인 접근 방식도 제시합니다.

- **Performance Highlights**: AceMath-72B-RM 모델은 기존의 최첨단 reward 모델들을 지속적으로 능가하며, AceMath-72B-Instruct와 AceMath-72B-RM을 결합할 경우 수학 추론 벤치마크에서 최고 평균 rm@8 점수를 달성합니다. 연구진은 AceMath-Instruct 및 AceMath-RM의 모델 가중치, 훈련 데이터를 오픈 소스로 공개할 예정이며, AceMath-RewardBench를 통해 다양한 데이터셋과 난이도를 제공할 것입니다.



### ConfliBERT: A Language Model for Political Conflic (https://arxiv.org/abs/2412.15060)
Comments:
          30 pages, 4 figures, 5 tables

- **What's New**: 이 논문은 최근 자연어 처리(Natural Language Processing) 기술의 발전을 활용하여 정치적 폭력과 관련된 정보 추출에 대해 제안한 ConfliBERT 언어 모델을 소개합니다. ConfliBERT는 전통적인 규칙 기반 접근법을 넘어, 정치 갈등에 관한 텍스트로부터 액터와 액션 클래스를 추출하는 데 우수한 성능을 보입니다. 다른 대규모 언어 모델들과 비교하여 정확도, 정밀도, 재현율에서 뛰어난 결과를 보여주며, 데이터 처리 속도는 수백 배 더 빠릅니다.

- **Technical Details**: ConfliBERT는 텍스트 코퍼스에서 유용한 정보를 추출하고 요약하는 데 중점을 둔 모델로, BERT(Bidirectional Encoder Representations from Transformers) 아키텍처를 기반으로 합니다. 이 모델은 무구조 뉴스 텍스트를 처리하기 위해 훈련된 특화된 언어 모델로, 정치 및 갈등 관련 데이터로 구성된 33.7GB의 전문가 선별 데이터셋을 활용합니다. 모델은 주요 태스크인 관련 정보 필터링, 이벤트 식별, 이벤트 속성 주석화를 최적화하여 연구자들의 작업을 수월하게 합니다.

- **Performance Highlights**: ConfliBERT는 기존의 수작업 주석 시스템의 병목 현상을 완화하고, 정밀한 이벤트 데이터셋을 생성하는 데 기여합니다. 이 모델은 특화된 정치 과학적 태스크에서 특히 뛰어난 성능을 보이며, 사건 유형의 다중 분류 작업을 통해 보다 정확한 데이터 생성을 가능하게 합니다. 연구 결과에 따르면, ConfliBERT는 영어 외에도 스페인어와 아랍어 변형 모델에서 두각을 나타내며, 이는 정치적 폭력과 관련된 다양한 분석에서 활용될 수 있습니다.



### LLMs Lost in Translation: M-ALERT uncovers Cross-Linguistic Safety Gaps (https://arxiv.org/abs/2412.15035)
- **What's New**: 이번 연구에서는 M-ALERT라는 다국어 안전 벤치마크를 소개합니다. 이는 영어, 프랑스어, 독일어, 이탈리아어, 스페인어 다섯 가지 언어에서 LLM의 안전성을 평가하는 데 초점을 맞추고 있습니다. 각 언어별로 15,000개의 고품질 프롬프트가 포함되어 있으며, 총 75,000개의 프롬프트가 준비되어 있습니다.

- **Technical Details**: M-ALERT는 ALERT의 분류 체계를 바탕으로 하여 개발되었으며, 고급 번역 파이프라인을 통해 다양한 언어로 안전 프롬프트를 체계적으로 번역하고 조정했습니다. 10개의 최첨단 LLM에 대해 포괄적인 평가를 수행하고, 모델별로 언어 특성에 따른 안전성의 강점과 약점을 파악했습니다. 특히 특정 범주에서는 모든 모델에서 일관되게 안전하지 않은 반응이 발생하는 경향을 확인했습니다.

- **Performance Highlights**: 연구 결과, Llama3.2 모델은 이탈리아어 범죄 세부 카테고리에서 높은 위험도를 보였으나, 다른 언어에서는 안전한 성능을 나타냈습니다. 모델의 크기와 안전성의 상관관계는 덜 뚜렷하였고, instruction tuning은 기본 모델에 비해 안전성을 향상시켰습니다. 이러한 발견은 다양한 사용자 커뮤니티를 위한 안전하고 책임 있는 사용을 보장하기 위해 다국어 안전 관행의 필요성을 강조합니다.



### Chain-of-MetaWriting: Linguistic and Textual Analysis of How Small Language Models Write Young Students Texts (https://arxiv.org/abs/2412.14986)
Comments:
          Accepted at WRAICOGS 2025 (Writing Aids at the Crossroads of AI, Cognitive Science, and NLP) co-located with COLING 2025

- **What's New**: 이 논문은 다국어 Small Language Models (SLMs)의 글쓰기 과정에 대한 세밀한 언어 및 텍스트 분석을 소개합니다. 'Chain-of-MetaWriting'이라는 방법론을 통해 SLM이 인간의 글쓰기 과정의 일부인 계획 및 평가 단계를 모방할 수 있도록 합니다. 연구는 주로 프랑스어로 된 단편 이야기와 에세이 작성을 중점적으로 다루며, 특히 초등학생과 대학생을 대상으로 합니다.

- **Technical Details**: 이 논문에서는 3B 매개변수를 가진 오픈소스 다국어 SLM 세 가지와 하나의 상용 모델을 사용했습니다. SLM은 모바일 장치에 적합하고, 빠른 추론 및 낮은 계산 비용의 이점을 가지고 있습니다. 실험은 SLM이 상위 수준의 글쓰기, 즉 사고, 계획, 언어 표현, 편집 및 수정의 단계를 얼마나 잘 모방할 수 있는지를 조사했습니다.

- **Performance Highlights**: 연구 결과, SLM은 학교폭력과 같은 민감한 주제에서 초등학생에게 도움을 주는 데 어려움을 겪으며, 때때로 목표 청중에게 너무 복잡한 단어를 사용함을 보여줍니다. 또한 텍스트의 응집력과 일관성이 부족하여 인간이 작성한 글과 큰 차이를 보였습니다. 이러한 발견은 SLM이 젊은 학생들에게 글 작성 보조 도구로서의 한계와 잠재적 위험성을 강조합니다.



### Knowledge Injection via Prompt Distillation (https://arxiv.org/abs/2412.14964)
Comments:
          Preprint

- **What's New**: 이 논문은 대규모 언어 모델(LLMs)에 새로운 지식을 주입하는 데 있어 기존의 fine-tuning 방법과 retrieval-augmented generation(RAG) 방식 간의 성능 격차를 줄이기 위한 새로운 fine-tuning 기술인 prompt distillation을 제안합니다. 이 기법은 self-distillation 접근 방식을 기반으로 하며, 교사 모델의 출력을 모방하는 학생 모델을 훈련시킵니다. 해당 방법론은 기존의 방법들이 갖는 단점을 해결하고, 효과적인 지식 주입을 가능하게 합니다.

- **Technical Details**: Prompt distillation은 보조적인 지식(c)와 질문-답변 쌍(q, a)의 생성을 통해 지식 주입을 가능하게 합니다. 학생 모델은 교사 모델의 정책을 모방하지만, 입력에 새로운 지식의 컨텍스트를 포함하지 않고 훈련됩니다. 이 과정에서 질문과 답변을 생성할 때 높은 온도(temperature, τ>1)를 사용하여 다양한 질문을 생성하고, 질적으로 다양한 노이즈가 포함된 샘플을 사용하여 학생 모델이 보다 일반화된 답변을 생성하도록 합니다.

- **Performance Highlights**: 실험 결과, prompt distillation은 RAG의 성능에 필적할 뿐만 아니라, 기존의 비-RAG 기준 모델들보다 우수한 성능을 보였습니다. 저자는 Squadshifts 데이터 세트를 사용하여 모델 간의 성능을 비교하고, 여러 데이터 세트에 대한 실험을 통해 prompt distillation의 우수성을 입증합니다. 이 연구는 fine-tuning 기술의 새로운 가능성을 제시하며, LLM의 지식 주입 방법론에 기여할 수 있는 중요한 통찰을 제공합니다.



### Understanding the Dark Side of LLMs' Intrinsic Self-Correction (https://arxiv.org/abs/2412.14959)
- **What's New**: 이번 논문은 LLM(대형 언어 모델)의 본질적 자기 교정(intrinsic self-correction) 성능을 검토하며, 특히 실패 사례를 중점적으로 분석합니다. 최근 연구에서는 LLM의 본질적 자기 교정이 오라클 레이블(oracle labels) 없이 제대로 작동하지 않는다고 지적하였습니다. 이는 LLM이 초기 답변에 대한 피드백을 제공받지 못할 경우, 모든 답변이 수정되어 특정 질문에 대한 정확한 답변을 도출하는데 실패할 수 있다는 점입니다.

- **Technical Details**: 우리는 LLM의 세 가지 주요 작업에서 자기 교정 실패를 분석하였습니다. 이를 위해 간단한 factual question에 대한 응답과 복잡한 의사결정, 추론, 프로그래밍 과제를 포함한 다양한 작업을 설정했습니다. 세 가지 해석 방법을 통해 LLM의 자기 교정 메커니즘을 탐구하고, 중간 응답의 흔들림과 프롬프트 편향(prompt bias), 인간과 유사한 인지 편향(cognitive bias) 등을 확인하였습니다.

- **Performance Highlights**: 자기 교정의 실패를 줄이기 위한 두 가지 간단 yet 효과적인 방법인 질문 반복(question repeating)과 적은 샘플로 진행하는 감독 Fine-tuning(supervised fine-tuning, SFT)을 제안합니다. 이러한 전략을 통해 LLM은 간단한 작업에서 학습한 내용을 복잡한 작업으로 일반화할 수 있으며, 자기 교정 실패를 줄일 수 있는 가능성을 높였습니다. 본 연구는 LLM의 예측 성능과 해석 가능성 향상에 기여할 것으로 기대됩니다.



### RobustFT: Robust Supervised Fine-tuning for Large Language Models under Noisy Respons (https://arxiv.org/abs/2412.14922)
- **What's New**: 이번 논문에서는 Large Language Models (LLMs)을 위한 안정적인 감독 세부 조정(Noise-robust Supervised Fine-Tuning, RobustFT) 프레임워크를 소개합니다. 이 프레임워크는 데이터의 잡음을 감지하고 재레이블링(relabeling)하는 메커니즘을 도입하여 모델의 다운스트림(task) 수행 능력을 향상시킵니다. 특히, 다수의 전문가 시스템을 활용하여 모델의 성능을 더욱 향상시키는 접근 방식을 제안함으로써, LLM이 발생하는 데이터에서 잡음을 효과적으로 다룰 수 있는 방법을 모색합니다.

- **Technical Details**: RobustFT는 멀티 뷰(multi-view) 잡음 검출 및 제거 전략을 사용합니다. 잡음 감지를 위해 협업 다중 전문가 시스템을 적용하여, 잡음 데이터의 식별을 효과적으로 수행합니다. 또한, 잡음 제거 단계에서 맥락 강화(context-enhanced) 전략을 활용해 신뢰할 수 있는 주석을 생성하며, 응답 엔트로피(response entropy)를 바탕으로 데이터를 선택하여 고품질 샘플만을 사용합니다.

- **Performance Highlights**: RobustFT는 다양한 잡음 수준을 가진 다섯 개 데이터셋에서 실행된 대규모 실험을 통해 탁월한 성능을 입증하였습니다. 실험 결과, RobustFT는 모델의 다운스트림 작업 수행 능력을 향상시키며, 특히 잡음이 많은 환경에서도 우수한 성능을 보여주었습니다. 결론적으로 이 프레임워크는 LLM의 세부 조정에 실질적인 가치를 제공하며, 다양한 도메인에서의 적용 가능성을 증명합니다.



### Dehallucinating Parallel Context Extension for Retrieval-Augmented Generation (https://arxiv.org/abs/2412.14905)
- **What's New**: 본 논문은 Parallel Context Extension (PCE)에 대한 기존 문제를 해결하기 위해 DePaC(Dehallucinating Parallel Context Extension)를 제안합니다. DePaC는 맥락 인식 부정 훈련(context-aware negative training)과 정보 보정 집계(information-calibrated aggregation)를 활용하여 사실 조작(fact fabrication) 및 사실 누락(fact omission)이라는 두 가지 종류의 환각 문제를 완화합니다. 이 접근 방식은 RAG 시나리오에서의 신뢰성을 높이는 데 기여할 것으로 기대됩니다.

- **Technical Details**: DePaC는 두 가지 주요 구성 요소로 이루어져 있습니다. 첫째, NegTrain은 맥락이 질문과 관련이 없는 경우 LLM이 답변을 거부하도록 유도합니다. 둘째, ICA는 각 문서의 정보 증가 정보(information increment)를 기반으로 맥락 창을 우선적으로 선택하여 집계합니다. 이 과정에서 Kullback-Leibler divergence를 활용하여 맥락의 유용한 정보를 보다 효과적으로 판별하게 됩니다.

- **Performance Highlights**: 실험 결과, DePaC는 아홉 개의 RAG 작업에서 환각 문제를 유의미하게 완화시키며 일관되게 더 나은 성능을 보여줍니다. 또한, DePaC는 vanilla 접근 방식보다 계산 복잡도가 낮아 문서 수에 따라 선형적으로 증가하는 추론 시간을 보여줍니다. 실험을 통해 정보 보정 집계와 맥락 인식 부정 훈련이 DePaC의 성능에 모두 필수적이라는 것을 입증하였습니다.



### Why language models collapse when trained on recursively generated tex (https://arxiv.org/abs/2412.14872)
Comments:
          28 pages, 9 figures

- **What's New**: 이번 논문은 언어 모델(모델의 약어: LM)의 붕괴(collapse)라는 현상에 대한 새로운 이론적 증거를 제시합니다. 특히, LM이 재귀적으로 생성된 텍스트로 훈련될 때 성능이 저하된다는 것을 실험적으로 발견했습니다. 앞으로의 연구에서 새로운 훈련 기법을 제안할 수 있는 중요한 통찰을 제공합니다.

- **Technical Details**: 저자들은 재귀적으로 생성된 텍스트에 대한 훈련이 LM 붕괴의 원인이라고 주장하며, LM의 모든 자동 회귀 모델(auto-regressive LMs)에서 발생하는 현상임을 명확히 증명합니다. 구체적으로, 초기 코퍼스(initial corpus)에 기반하여 생성된 LM이 생성된 텍스트로 재훈련될 때 발생하는 오류가 축적되어 궁극적으로 LM이 초기에 학습한 분포와 무관한 방향으로 전이(uniform distribution)된다고 언급합니다.

- **Performance Highlights**: 실험 결과, LM의 성능은 재귀적으로 생성된 텍스트로 훈련할수록 저하되며 결국 무작위 초기화된 LM과 유사한 수준까지 떨어집니다. 구체적으로, 이들은 문법적으로 올바른 텍스트 생성 가능성이 줄어들고, 자연어 처리 작업에서 성능이 악화되는 경향을 보였습니다. 이는 LM이 기존 데이터 분포를 따르지 않게 된 결과로 해석됩니다.



### Graph-Convolutional Networks: Named Entity Recognition and Large Language Model Embedding in Document Clustering (https://arxiv.org/abs/2412.14867)
Comments:
          11 pages, 4 figures

- **What's New**: 최근 기계 학습의 발전은 특히 BERT와 GPT와 같은 대형 언어 모델(LLMs)의 출현을 통해 텍스트 표현을 크게 개선했습니다. 본 논문에서는 명명된 엔터티 인식(NER)과 LLM 임베딩을 그래프 기반 프레임워크에 통합하여 문서 클러스터링을 수행하는 새로운 접근 방식을 제안합니다. 이 방법은 문서를 나타내는 노드와 명명된 엔터티 유사성에 의해 가중치가 부여된 에지를 가진 그래프를 구축합니다.

- **Technical Details**: 우리의 방법은 NER과 LLM 임베딩을 결합하여 그래프를 구성합니다. 그래프는 문서 간의 유사성을 더 잘 표현하기 위해 각 문서에서 명명된 엔티티의 컨텍스트 유사성에 의해 가중치가 부여된 에지를 사용합니다. 이를 최적화하기 위해 그래프 컨볼루션 네트워크(GCN)를 사용하여 임베딩과 클러스터링 목표를 공동으로 최적화합니다.

- **Performance Highlights**: 실험 결과, 우리의 방법은 일반적인 동시 발생 기반 기술에 비해 특히 명명된 엔티티가 풍부한 문서에서 클러스터링 성능이 우수하다는 것을 보여줍니다. 전통적인 방법과 비교하여 명명된 엔터티 유사성을 사용한 접근 방식이 문서 간의 더욱 명확한 군집 구성을 가능하게 함을 입증하고 있습니다.



### Think&Cite: Improving Attributed Text Generation with Self-Guided Tree Search and Progress Reward Modeling (https://arxiv.org/abs/2412.14860)
- **What's New**: 이 논문은 Think&Cite라는 새로운 프레임워크를 제안하며, attributed text generation을 다단계 추론 문제로 공식화합니다. 특히, 자가 안내 몬테카를로 트리 탐색(Self-Guided Monte Carlo Tree Search, SG-MCTS)을 도입하여 LLM의 자기 반영 능력을 활용하고 트리 확장을 위한 경로를 안내합니다. 이 연구는 전통적인 접근 방식과 달리, 더 깊이 있는 사고를 필요로 하며, 신뢰할 수 있는 참고 문헌을 통해 신뢰성을 높이는 것을 목표로 하고 있습니다.

- **Technical Details**: 우리가 제안하는 프레임워크는 입력 질문과 관련된 텍스트를 바탕으로 명확한 인용을 포함한 텍스트 생성을 목표로 합니다. 모델은 MCTS의 중간 상태를 실시간으로 반영하여 부적절한 추론 경로를 피할 수 있도록 개선합니다. 또한, Progress Reward Models(PRM)을 도입해 트리 탐색의 진전을 측정하며, 이는 전체적인 발전 과정을 평가할 수 있는 방법을 제공합니다.

- **Performance Highlights**: 이 접근 방식은 기존의 프롬프트 기반 및 파인튜닝 방법보다 유의미한 성과를 보였으며, 3개의 데이터셋에서 실험을 통해 효과성을 입증하였습니다. 연구 결과는 Think&Cite의 다단계 추론 방식이 attributed text generation의 성능을 상당히 향상시킴을 보여주고 있습니다. 이러한 결과는 LLM의 실제 활용 가능성을 높이는 데 기여할 것으로 기대됩니다.



### DS$^2$-ABSA: Dual-Stream Data Synthesis with Label Refinement for Few-Shot Aspect-Based Sentiment Analysis (https://arxiv.org/abs/2412.14849)
- **What's New**: 최근에 개발된 대규모 언어 모델(LLMs)은 저자원 시나리오에서 데이터 부족 문제를 해결할 수 있는 새로운 경로를 제시하였다. 이 논문에서는 DS$^2$-ABSA라는 쌍방향 데이터 합성 프레임워크를 제안하여, 기존 방법들이 만들어내는 다양한 데이터 부족 문제를 해결하고자 한다. 이 프레임워크는 LLM을 사용하여 '키 포인트 주도'(key-point-driven) 및 '인스턴스 주도'(instance-driven) 두 가지 관점에서 데이터를 합성하여, 고품질의 ABSA 샘플을 생성한다.

- **Technical Details**: DS$^2$-ABSA 프레임워크는 두 개의 주요 접근법을 바탕으로 구성된다. 첫째, LLM은 다양한 ABSA 속성을 생성하고 이를 새로운 샘플로 구성하는 키 포인트 주도 전략을 실시한다. 둘째, 인스턴스 주도 전략은 기존 샘플을 결합 및 선택적 재구성을 통해 새로운 데이터를 생성하여, 주요 상황을 포괄하면서도 실제적인 데이터의 질을 높인다.

- **Performance Highlights**: 다양한 공개 데이터세트에서의 실험 결과, DS$^2$-ABSA는 기존의 저자원 ABSA 솔루션 및 다른 LLM 기반 데이터 생성 방법들에 비해 현저히 우수한 성과를 보였다. 새로운 레이블 정제 모듈을 통해 합성 데이터의 레이블 품질도 크게 향상되었다. 이러한 결과는 저자원 환경에서 ABSA의 효율성을 크게 개선하는 데 기여할 것으로 기대된다.



### A Survey of RWKV (https://arxiv.org/abs/2412.14847)
Comments:
          18 pages

- **What's New**: Receptance Weighted Key Value (RWKV) 모델은 Transformer 아키텍처의 새로운 대안으로 부각되고 있습니다. 이 모델은 순환 신경망(RNN)과 주의(attention) 기반 시스템의 장점을 결합하여 긴 시퀀스를 효율적으로 처리할 수 있도록 설계되었습니다. RWKV는 기존 Transformer 모델의 계산 비효율성을 줄이는 데 기여하며, 자연어 처리(NLP) 및 컴퓨터 비전 분야에서 차별화된 성과를 보여주고 있습니다.

- **Technical Details**: RWKV는 고유한 키-값(key-value) 접근 방식을 활용하여 последовательные 의존성을 관리하며 계산 오버헤드를 크게 줄입니다. 이 모델은 긴 시퀀스의 처리가 필요한 작업에서도 강력한 맥락 이해를 유지하면서도 최소한의 메모리 요구량으로 긴 범위 의존성을 캡처할 수 있습니다. 또한, RWKV는 기존 Transformer 모델과의 비교를 통해 그 효율성을 강조하며 다양한 구현 및 적용 사례를 제시합니다.

- **Performance Highlights**: RWKV는 자연어 생성(NLG), 자연어 이해(NLU) 및 컴퓨터 비전 등 다양한 분야에서 강력한 성능을 보여주고 있습니다. 특히, RWKV는 텍스트 생성, 기계 번역 및 감정 분석 등의 NLP 작업에서 그 가능성을 입증했으며, 시간이 많이 소요되는 시퀀스 예측 작업에서도 탁월한 결과를 나타내고 있습니다. 이를 통해 RWKV는 머신러닝 커뮤니티에서 주목받고 있으며, 향후 연구 방향과 발전 가능성에 대한 논의가 활발하게 이루어질 것으로 기대됩니다.



### Mapping and Influencing the Political Ideology of Large Language Models using Synthetic Personas (https://arxiv.org/abs/2412.14843)
Comments:
          4 pages, 2 figures, 2 tables

- **What's New**: 이번 연구에서는 PersonaHub라는 합성 인물 설명 집합을 활용하여 정치적 편향이 큰 언어 모델(LLMs)의 정치적 성향을 평가했습니다. 특히, 개인화된 프롬프트가 LLM의 정치적 방향성에 미치는 영향을 분석함으로써 기존 연구에서 다루지 않았던 측면을 탐구했습니다. 연구 결과, LLM이 오른쪽 권위주의와 왼쪽 자유주의 간의 상극 성향으로 조정될 수 있음을 보였습니다.

- **Technical Details**: 연구에서는 1억 개 이상의 합성 인물 설명을 포함하는 PersonaHub를 사용하여 LLM의 정치적 방향성을 탐구했습니다. 총 12.4백만 개의 응답을 수집하여 각 모델의 정치적 입장을 평가하기 위해 Political Compass Test (PCT)를 활용했습니다. 4개의 오픈 소스 언어 모델(Mistral, Llama, Qwen, Zephyr)을 선택하고, 두 가지 단계로 실험을 수행하여 인물 설명이 LLM의 정치적 성향에 미치는 영향을 측정했습니다.

- **Performance Highlights**: 결과적으로, 대부분의 인물들은 왼쪽 자유주의 지역에 집중되었으나, 모든 모델은 명시적 이념 프롬프트에 반응하여 정치적 편향을 바꿨습니다. 특히 Llama 모델은 오른쪽 권위주의로의 이동이 가장 두드러졌고, Zephyr는 전반적으로 더 수평적 이동을 보였습니다. 이 연구는 LLM의 정치적 표현 방식에 대한 새로운 통찰을 제공하며, 개인화의 가능성과 한계를 강조합니다.



### DynamicKV: Task-Aware Adaptive KV Cache Compression for Long Context LLMs (https://arxiv.org/abs/2412.14838)
- **What's New**: 이 논문에서는 LLMs(Long Language Models)의 키-값 캐시(KV cache) 관리를 효율적으로 개선하는 방법인 DynamicKV를 제안합니다. 기존의 고정 패턴 기반 KV 캐시 압축 방식들은 작업 특성을 무시하여 중요한 정보를 지속적으로 유지하는 데 필요한 경향을 감소시킵니다. 연구진은 다양한 작업에서 구별된 활성화 패턴을 관찰하며, 각 작업의 요구 사항에 맞춘 동적이고 적응적인 전략의 필요성을 강조합니다.

- **Technical Details**: DynamicKV는 각 계층에서 유지할 토큰 수를 조정하여 특정 작업에 맞춰 동적으로 최적화합니다. 이를 위해 Global 및 Per-layer 최대 KV 캐시 예산을 설정하며, 각 레이어에서 최대 예산을 임시 유지하고 추론 중에 모든 이전 레이어의 KV 캐시 크기를 주기적으로 업데이트합니다. 이 방법은 원래 KV 캐시 크기의 1.7%만 유지함에도 불구하고 LongBench에서 ~85%의 전체 KV 캐시 성능을 달성합니다.

- **Performance Highlights**: DynamicKV는 Needle-in-a-Haystack 테스트에서 기존의 최첨단(SOTA) 방법보다 11% 향상된 성능을 보이며, Extreme Compression(0.9%) 환경에서도 뛰어난 성능을 유지했습니다. 16개의 LongBench 데이터 세트에서의 실험 결과, DynamicKV는 기존 고정 패턴 방법들과 비교하여 전반적인 효율성이 뛰어난 것으로 입증되었습니다. 이처럼 DynamicKV는 고성능을 유지하면서도 적은 수의 토큰으로 운영할 수 있는 장점을 갖고 있습니다.



### Progressive Multimodal Reasoning via Active Retrieva (https://arxiv.org/abs/2412.14835)
Comments:
          Working in progress

- **What's New**: 이번 연구에서는 다중 단계를 고려한 다중 모드(multimodal) 추론 과제에서 MLLM의 성능을 향상시키기 위한 새로운 프레임워크인 AR-MCTS를 제안합니다. 이 프레임워크는 Active Retrieval(AR)과 Monte Carlo Tree Search(MCTS)를 결합하여 복잡한 추론 문제를 해결하는 데 필요한 핵심 인사이트를 동적으로 검색할 수 있도록 설계되었습니다. 특히, 이 연구는 기존의 빔 탐색(beam search) 샘플링 방법을 대체하는 혁신적인 접근 방식을 도입하여 각 추론 단계에서 다양한 문제 해결 인사이트를 제공함으로써 신뢰성을 높이고자 합니다.

- **Technical Details**: AR-MCTS 프레임워크는 통합된 검색 모듈을 개발하여 하이브리드 모드 검색 데이터베이스로부터 복잡한 추론을 지원하기 위한 핵심 인사이트를 검색합니다. MCTS 알고리즘을 활용하여 단계별로 주어진 문제의 적절한 해답을 유도하는 과정 보상을 정의하고, 각 단계에서 이전 단계를 바탕으로 샘플링을 최적화합니다. 이러한 접근 방식은 추론의 신뢰성과 다양성을 향상시키며, 자동화된 다중 모드 추론 검증 과정을 가능하게 합니다.

- **Performance Highlights**: 실험 결과에 따르면, AR-MCTS는 세 가지 복잡한 다중 모드 추론 기준을 통해 다양한 모델에서 효과성을 입증했습니다. AR-MCTS는 샘플링의 다양성과 검증의 정확성을 최적화하여 신뢰성 있는 다중 모드 추론을 위한 promising한 솔루션을 제공합니다. 이 연구 결과는 다중 모드 추론의 고도화를 위한 가능성을 열어 주며, MLLM의 성능 향상에 기여할 것으로 기대됩니다.



### Mention Attention for Pronoun Translation (https://arxiv.org/abs/2412.14829)
Comments:
          camera-ready version of the paper accepted by JCRAI-23 conference, in ACL format

- **What's New**: 이번 논문에서는 기계 번역(Machine Translation)에서 대명사 번역을 개선하기 위해 소스 언어의 대명사와 관련된 멘션(mmentions)의 추가 특성을 활용하는 방법을 제안합니다. 이를 위해, 디코더에 멘션 어텐션 모듈을 추가하여 비멘션(non-mention) 토큰이 아닌 소스 멘션에 더욱 많은 주의를 기울입니다. 이러한 접근 방식은 대명사 번역의 성능을 향상시키는 데 기여하며, 기존 Transformer 모델보다 APT와 BLEU 점수에서 개선된 결과를 보여줍니다.

- **Technical Details**: 제안된 모델은 기존 Transformer NMT 아키텍처를 기반으로 하며, 디코더에 멘션 어텐션 레이어를 추가하여 소스 멘션의 추가 특성을 추출합니다. 우리가 도입한 멘션 분류기는 각 토큰이 멘션인지 아닌지를 구별하도록 훈련되며, 두 개의 FFNN을 사용하여 엔코더와 디코더에서 멘션 태그를 예측하는 구조입니다. 이 멘션 어텐션 모듈은 엔코더-디코더 어텐션과 동일한 구조를 가지지만, 소스 토큰의 모든 특징이 아닌 멘션 토큰에서만 특징을 추출합니다.

- **Performance Highlights**: WMT 2017 영어-독일어 번역 작업에서 제안된 방법은 기존 모델보다 BLEU 점수가 약간 향상된 결과를 보여주었습니다. 특히 모호한 대명사에 대한 APT 점수를 높였으며, 이는 소스 멘션에 대한 추가 주의가 대명사 번역의 품질을 개선할 수 있음을 입증합니다. 이러한 결과는 제안된 모듈이 일반 번역 품질에 부정적인 영향을 미치지 않음을 보여줍니다.



### ResoFilter: Rine-grained Synthetic Data Filtering for Large Language Models through Data-Parameter Resonance Analysis (https://arxiv.org/abs/2412.14809)
Comments:
          under review

- **What's New**: 이 논문에서는 ResoFilter라는 새로운 메서드를 제안하여 데이터 선택을 최적화하고자 합니다. 기존 데이터 증강 기법이 양적인 향상에 집중하는 반면, ResoFilter는 데이터 품질을 중시하며 모델 가중치를 통해 데이터 특성을 해석할 수 있게 합니다. 이 방법은 데이터의 특정 특성을 반영한 점수 체계를 통해 유용한 데이터 포인트를 선택합니다.

- **Technical Details**: ResoFilter는 데이터의 선택을 위해 매개변수 변화와 모델 가중치의 변화를 포착하는 전방 및 후방 전파 과정을 포함합니다. 각 데이터 포인트에 대한 특성 점수를 도출하여, 데이터 선택 과정에서 이 점수를 기준으로 고품질 데이터를 선별합니다. 이 방법은 기존의 기법들과 비교하여 해석 가능성을 크게 향상시키며, 다양한 모델 아키텍처와 도메인에 대해 일반화된 성능을 보입니다.

- **Performance Highlights**: 실험 결과, ResoFilter는 데이터의 50%만 사용하여 전체 데이터 세트를 미세 조정하는 것과 동등한 성능을 달성했습니다. 뿐만 아니라 부적합한 데이터를 제거함으로써 전체 미세 조정 성능을 초과하기도 했습니다. 이러한 방법론은 아직 다양한 모델과 도메인에서 강한 일반화를 보여주어 데이터 품질 개선을 위한 유망한 솔루션으로 자리 잡고 있습니다.



### Disentangling Reasoning Tokens and Boilerplate Tokens For Language Model Fine-tuning (https://arxiv.org/abs/2412.14780)
- **What's New**: 이번 연구에서는 에이전트-작업 데이터셋을 통해 대규모 언어 모델(LLMs)의 능력을 향상시키기 위해, 토큰의 중요성과 학습 복잡성을 고려한 새로운 접근법을 제안합니다. SHuffle-Aware Discriminator(SHAD)를 통해, 다양한 역할을 하는 토큰을 적응적으로 구분하여 학습하는 방법을 소개하며, 이를 통해 Reasoning-highlighted Fine-Tuning(RFT) 기법을 개발했습니다. 이 방법은 기존의 Supervised Fine-Tuning(SFT)보다 뛰어난 성능 향상을 보여줍니다.

- **Technical Details**: SHAD는 입력-출력 조합을 섞은 후 토큰의 예측 가능성 차이를 이용해 토큰을 분류합니다. 구체적으로, 모델을 소량의 섞인 데이터로 미세 조정한 후, 조정된 모델과 원래 모델의 토큰 수준 손실을 비교하여 각 토큰의 유형(이유 토큰 또는 보일러플레이트 토큰)을 판별합니다. RFT는 이 SHAD의 결과를 바탕으로 어려운 이유 토큰에 더 많은 가중치를 부여하여 학습을 강조합니다.

- **Performance Highlights**: 이 연구에서 제안된 RFT는 여러 일반적인 에이전트 벤치마크에서 기존의 방법보다 뛰어난 성능을 발휘합니다. 특히, 우리의 방법은 이유 토큰을 효과적으로 식별하고 학습을 강화할 수 있으며, 이를 통해 LLM의 에이전트 능력을 개선하는 데 기여합니다. 또한, 데이터셋에서의 과적합 및 일반화 문제를 완화하여 보다 나은 성능을 보여줍니다.



### ALKAFI-LLAMA3: Fine-Tuning LLMs for Precise Legal Understanding in Palestin (https://arxiv.org/abs/2412.14771)
- **What's New**: 이 논문은 저자들이 팔레스타인 법률 분야에 적합한 대규모 언어 모델(LGMs)을 조정하는 방법을 탐구하고, 그 과정에서의 도전 과제를 다룹니다. 특히, 복잡한 정치적 환경과 제한된 AI 자원이 있는 저자원 국가에서 AI의 효율적인 실행을 위한 해결책을 제안합니다. 연구팀은 Llama-3.2-1B-Instruct의 양자화된 버전을 기반으로 한 세밀하게 조정된 모델을 제시하며, 이 모델은 팔레스타인 법률 텍스트에서 유도된 합성 데이터 세트로 학습됩니다.

- **Technical Details**: 연구에서 사용된 기본 법률은 팔레스타인 정부의 공식 소스를 통해 수집된 1,277개의 텍스트 파일로 구성되어 있습니다. 이 텍스트 파일은 JSON 형식으로 구조화되어 빠른 문서 접근을 가능하게 하며, 법률 조항과 해당 법률 텍스트를 포함합니다. 질문-답변 쌍은 ChatGPT API와 Gemma API를 활용해 생성되었으며, 이는 법률 언어 작업을 위한 세부 조정을 지원하는 robust한 데이터 세트를 형성하는 데 기여합니다.

- **Performance Highlights**: 실험 결과, 연구팀은 다양한 질문 유형에 대해 유망한 성능을 보여주었으며, 예/아니오 질문, 서사적 설명, 복잡한 법적 구별에 대한 질문을 포함했습니다. 그러나 계산 기반 문의 처리와 정형 리스트 포맷팅 등에서 개선 여지가 있음을 강조합니다. 이 연구는 자원이 제한된 환경에서 AI 기반 법률 지원 도구의 배치를 위한 경로를 제시합니다.



### PsyDraw: A Multi-Agent Multimodal System for Mental Health Screening in Left-Behind Children (https://arxiv.org/abs/2412.14769)
Comments:
          preprint

- **What's New**: 이 논문은 중국의 6,600만 명 이상의 '이탈 아동'(Left-behind children, LBCs)이 직면한 심각한 정신 건강 문제를 해결하기 위해 PsyDraw라는 다중 에이전트 시스템을 제안합니다. 최근 몇 년간 LBCs는 불행한 가정 환경으로 인해 우울증과 불안 발병률이 높아지고, 이로 인해 적극적인 정신 건강 선별의 필요성이 증가하고 있습니다. 제한된 정신 건강 전문가의 수를 고려할 때, 자원의 부족한 지역에서도 효과적으로 활용할 수 있는 자동화된 도구의 필요성이 대두되고 있습니다.

- **Technical Details**: PsyDraw는 House-Tree-Person (HTP) 테스트를 기반으로 하여 아동의 그림을 분석하고 심리적 상태를 평가합니다. 이 시스템은 두 단계(특징 분석 및 보고서 생성)로 구성되어 있으며, 전문 지식에 따라 설계된 다양한 에이전트들이 협력하여 작업을 수행합니다. 특히, 각 예술적 요소(예: 집, 나무, 인물)에 대한 심리적 지표를 분석하고, 이를 통해 포괄적인 심리 평가 리포트를 생성합니다.

- **Performance Highlights**: 290명의 초등학생의 HTP 그림 분석 결과, 71.03%가 전문가 평가와 높은 일치도를 보였으며, 31.03%의 사례에서 전문적인 도움이 필요함을 확인했습니다. 이러한 결과는 PsyDraw가 자원 제한이 있는 지역에서도 정신 건강 전문가를 지원하며, 아동의 심리적 웰빙을 평가하는 혁신적인 솔루션이 될 수 있음을 보여줍니다. 현재 이 시스템은 몇몇 학교에서 시범적으로 운영되고 있으며, 앞으로의 응용 가능성에 대한 기대가 높습니다.



### Query pipeline optimization for cancer patient question answering systems (https://arxiv.org/abs/2412.14751)
- **What's New**: 이 연구에서는 암 환자 질문 답변 시스템(CPQA)에 대한 Retrieval-Augmented Generation (RAG) 쿼리 파이프라인의 새로운 최적화 접근법을 제안합니다. 이는 PubMed 및 PubMed Central과 같은 공개 생물 의학 데이터베이스를 활용하여 정보 검색과 답변 생성을 통합하는 방식입니다. 구체적으로, 문서 및 구문 검색에 대한 비교 분석과 이를 위한 새로운 도구인 Hybrid Semantic Real-time Document Retrieval (HSRDR) 및 Semantic Enhanced Overlap Segmentation (SEOS)를 도입합니다.

- **Technical Details**: 저자들은 주요 쿼리 파이프라인 컴포넌트를 세 가지 측면에서 최적화합니다: (1) HSRDR을 통한 문서 검색 최적화, (2) 밀도가 높은 검색기와 재순위 모델의 최적 쌍을 찾아내는 구문 검색, (3) SEOS를 이용한 의미적 표현 향상입니다. 이 과정에서 PubMed 및 PMC의 자료를 분석하여 특정 질병에 대한 질문에 대한 답변의 정확성을 향상시키는 데 중점을 둡니다.

- **Performance Highlights**: 최적화된 RAG 접근법을 사용하여 Claude-3-haiku 모델의 암 관련 질문에 대한 답변 정확도가 기존의 체인-오브-생각 프롬프트보다 5.24% 향상되었으며, 기본 RAG 설정에 비해 약 3% 향상되었습니다. 이 연구는 RAG 기반 생물 의학 시스템의 신뢰성을 높이기 위한 도메인별 쿼리 최적화의 중요성을 강조합니다.



### On Verbalized Confidence Scores for LLMs (https://arxiv.org/abs/2412.14737)
- **What's New**: 최근 대형 언어 모델(LLMs)의 인기가 높아지면서 이러한 모델의 신뢰성을 향상시키는 것이 중요해졌습니다. 이 논문은 LLM이 자신의 불확실성을 신뢰도 점수로 표현하도록 요청하는 방법을 제안하며, 이는 기존 방법들에 비해 오버헤드가 적은 새로운 접근 방식입니다.

- **Technical Details**: LLM의 응답에서 불확실성을 추정하기 위해, 내부 토큰 로짓(internal token logits), 작업별 프록시 모델(task-specific proxy models), 또는 다수의 응답 샘플링(sampling of multiple responses) 방법이 일반적으로 사용됩니다. 이 연구에서는 특히 프롬프트와 모델에 구애받지 않는 방식으로 신뢰도 점수를 산출하는 기법에 초점을 맞추었습니다.

- **Performance Highlights**: 광범위한 벤치마크를 통해, 프롬프트 방법에 따라 잘 조정된 신뢰도 점수를 추출할 수 있음을 보였습니다. 연구 결과는 신뢰도 점수가 모델에 대한 질문 방식에 크게 의존하며, 특정 프롬프트 방법을 통해서는 잘 조정된 신뢰도 점수를 얻는 것이 가능하다는 것을 보여줍니다.



### How to Synthesize Text Data without Model Collapse? (https://arxiv.org/abs/2412.14689)
- **What's New**: 이 논문은 합성 데이터(synthetic data)가 언어 모델의 학습 과정에 미치는 영향을 다룹니다. 특히, 합성 데이터를 사용한 반복적인 훈련이 모델 성능을 저하시킬 수 있는 현상인 모델 붕괴(model collapse)를 강조합니다. 연구팀은 합성 데이터와 인간이 생산한 데이터를 일정 비율로 혼합하여 훈련했을 때 성능 저하가 나타난다는 것을 발견했습니다.

- **Technical Details**: 연구에서는 인간이 만든 데이터와 합성 데이터의 혼합 비율을 α로 정의하고, 이를 기반으로 훈련 데이터를 설정합니다. 실험 결과 합성 데이터를 사용할 때 모델의 성능이 하락하고, 특히 인간이 생산한 데이터를 포함했을 때 모델 붕괴가 어느 정도 완화됨을 확인했습니다. 또한, 통계 분석을 통해 합성 데이터의 분포가 제한적이며 특성 집중이 발생한다는 결론을 도출했습니다.

- **Performance Highlights**: 이 논문에서 제안하는 token-level editing 방법은 모델 붕괴를 방지하는 이론적 근거를 제공하며, 여러 통계 분석 실험을 통해 그 효과성을 입증하고 있습니다. 대규모 실험을 통해 초기 모델 훈련 및 지속적인 훈련 과정에서 token-level editing이 데이터 품질을 개선하고 모델 성능을 높인다는 것을 확인했습니다.



### Each Fake News is Fake in its Own Way: An Attribution Multi-Granularity Benchmark for Multimodal Fake News Detection (https://arxiv.org/abs/2412.14686)
- **What's New**: 이번 연구에서는 멀티모달(fake news) 가짜 뉴스 탐지의 중요한 도전 과제를 다루고 있습니다. 기존의 멀티모달 데이터셋은 단순히 진짜 또는 가짜로만 레이블링 되어, 다양한 유형의 가짜 뉴스를 반영하지 못했습니다. 이에 따라, 우리는 고유한 가짜 패턴을 드러내는 AMG라는 새로운 데이터셋을 구축하고, 멀티그랜ולר 단서를 통한 가짜 뉴스 탐지 및 할당 모델(MGCA)을 제안합니다.

- **Technical Details**: AMG 데이터셋은 다양한 소셜 플랫폼에서 수집된 가짜 뉴스를 포함합니다. 데이터 수집과 처리 과정에서, 사실 확인 웹사이트를 활용하여 가짜 뉴스의 세부 유형을 주석 처리하고, 시각적 및 텍스트 콘텐츠의 멀티 뷰 특성을 추출하는 MGCA 모델을 개발하였습니다. 이 모델은 또한 다중 그랜율 단서의 일관성을 모델링하여 진위 탐지와 할당을 지원합니다.

- **Performance Highlights**: 실험 결과는 AMG 데이터셋이 상당히 도전적임을 보여주며, 새로운 연구 방향을 제시합니다. 우리는 본 연구에서 멀티그랜율 가짜 뉴스 할당의 개념을 제시하고, AMG를 통해 가짜 뉴스의 원인에 대한 세분화된 할당을 수행하였습니다. 전체적으로 우리의 제안 모델인 MGCA는 멀티모달 가짜 뉴스 탐지 및 할당에서 강력한 성능 향상을 보였습니다.



### LLMs as mediators: Can they diagnose conflicts accurately? (https://arxiv.org/abs/2412.14675)
Comments:
          27 pages, 2 appendices, 21 tables (incl appendices)

- **What's New**: 이 연구는 OpenAI의 대규모 언어 모델인 GPT 3.5와 GPT 4가 갈등 중재를 위해 서로 다른 출처의 불일치를 구별할 수 있는지를 탐구합니다. 이전 연구에서는 사람들 간의 불일치를 인지하는 것이 중요하다는 점을 강조하였는데, 본 논문은 이 주제를 AI 모델에 적용하였습니다.

- **Technical Details**: 연구자는 Koçak et al. (2003)의 1차 연구를 재현하며, 비주얼 디자인을 활용하여 모델의 성능을 평가하였습니다. LLMs는 인간과 유사하게 인과적 (causal) 및 도덕적 (moral) 차이를 구별할 수 있으며, 두 모델 모두 이들 사이의 구별을 신뢰성 있게 수행할 수 있음을 보여주었습니다.

- **Performance Highlights**: GPT 4는 특정 문제에 대한 구체적 언어를 사용하는 경우 인과적 불일치의 범위를 과대평가하고 도덕적 불일치를 과소평가하는 경향이 있습니다. 반면, GPT 3.5는 GPT 4나 인간에 비해 성능이 떨어지는 것으로 나타났으며, 갈등의 근본 원인을 진단하는 데 사용되는 잠재적인 가능성을 보여줍니다.



### Analysis and Visualization of Linguistic Structures in Large Language Models: Neural Representations of Verb-Particle Constructions in BER (https://arxiv.org/abs/2412.14670)
- **What's New**: 이번 연구에서는 transformer 기반의 대형 언어 모델 (large language models, LLMs)에서 동사-부사 조합의 내부 표현을 조사합니다. 특히 BERT 아키텍처를 활용하여, 다양한 동사-부사 구조의 어휘적 (lexical) 및 구문적 (syntactic) 뉘앙스를 어떻게 포착하는지를 분석합니다. 이는 기존의 신경망이 언어적 요소를 처리할 때 가정된 균일성에 도전하는 중요한 발견입니다.

- **Technical Details**: 연구 방법론으로는 British National Corpus에서 데이터셋을 준비하고, BERT 모델을 광범위하게 학습시키며 출력 분석을 다차원 스케일링 (multi-dimensional scaling, MDS) 및 일반화된 판별 값 (generalized discrimination value, GDV) 계산을 통해 진행했습니다. BERT의 중간 계층이 구문 구조를 가장 효과적으로 포착하며, 다양한 동사 범주에 따라 표현 정확도가 크게 달라지는 것으로 나타났습니다.

- **Performance Highlights**: 연구 결과, BERT 모델이 언어 처리를 어떻게 이해하고 수행하는지에 대한 통찰을 제공합니다. 이는 현재의 신경적 접근 방식이 언어 분석에서 지닌 잠재력과 한계를 드러내며, 깊이 있는 언어 모델 성능을 최적화하기 위한 추가 연구의 필요성을 제기합니다. 이 연구는 컴퓨터 언어학 분야에서도 발전을 이끌어내며, 언어적 정밀도를 증대시키기 위한 신경망 아키텍처 최적화에 대한 새로운 방향을 제시합니다.



### Length Controlled Generation for Black-box LLMs (https://arxiv.org/abs/2412.14656)
Comments:
          Preprint

- **What's New**: 이 논문에서는 LLMs(대형 언어 모델)가 생성하는 텍스트의 길이를 효과적으로 관리할 수 있는 새로운 반복 샘플링 프레임워크를 제안합니다. 기존의 방법들은 파라미터를 조정하여 진행했지만, 이는 비효율적이고 최적이 아닙니다. 새로운 접근법인 메트로폴리스-헤이스팅스(Metropolis-Hastings) 알고리즘을 사용하여 LLMs가 길이 제약이 있는 텍스트를 신뢰성 있게 생성할 수 있도록 합니다.

- **Technical Details**: 프레임워크는 원래 LLM의 출력에서 시작하여 이전 후보에 기초하여 후보 출력을 반복적으로 생성합니다. 주요 요소로는 목표 길이와 생성 확률 밀도가 비교되는 수용(distribution) 함수가 포함됩니다. 또한, 중요 샘플링(Importance Sampling) 기법을 활용하여 후보 길이가 원하는 길이에 가까울수록 샘플링 확률을 높이며, 이로 인해 반복 과정이 가속화됩니다.

- **Performance Highlights**: 실험 결과, Llama3.1 모델의 경우, 5회 이내에 거의 100%의 길이 제어 성공률을 달성하여 효율성과 실용성을 입증하였습니다. 해당 프레임워크는 LLM의 품질을 저하시키지 않고도 길이 제어 성능에서 최첨단 성과를 달성했습니다. 이는 다양한 실제 응용 프로그램에서 LLM의 유연성과 일반성을 유지하면서도 정확한 길이 제어의 잠재력을 강조합니다.



### TOMG-Bench: Evaluating LLMs on Text-based Open Molecule Generation (https://arxiv.org/abs/2412.14642)
Comments:
          A benchmark for text-based open molecule generation

- **What's New**: 이번 논문에서는 텍스트 기반의 개방형 분자 생성 벤치마크(TOMG-Bench)를 제안합니다. 이는 LLM의 개방형 분자 생성 능력을 평가하는 최초의 벤치마크로, 분자 편집(MolEdit), 분자 최적화(MolOpt), 맞춤형 분자 생성(MolCustom) 등 세 가지 주요 작업을 포함합니다. 각 작업은 5,000개의 테스트 샘플로 구성된 세 개의 하위 작업으로 세분화되어 있습니다. 또한, 우리는 생성된 분자의 품질과 정확성을 측정하기 위한 자동 평가 시스템을 개발하였습니다.

- **Technical Details**: TOMG-Bench는 LLM이 분자 디자인 및 작업을 수행하는 능력을 평가하기 위해 체계적인 지침 시리즈를 통해 설계되었습니다. 기존의 목표 지향적인 분자 생성 작업과 달리, TOMG-Bench는 특정 목표를 설정하지 않고 LLM이 적합한 생성물을 생성하는지 여부를 테스트합니다. 이를 위해 RDKit과 같은 화학 도구 상자를 사용하여 성과를 평가합니다. 새로운 평가 지표 세트도 도입되어 LLM의 성능을 정확도 및 품질 기준으로 평가하고 순위를 매깁니다.

- **Performance Highlights**: 실험 결과, OpenMolIns라는 특화된 지침 튜닝 데이터셋의 도움을 받아 Llama3.1-8B가 TOMG-Bench에서 GPT-3.5-turbo를 46.5% 초과하는 성능을 보였습니다. 25개의 LLM을 종합적으로 벤치마킹한 결과, 현재의 한계와 개선 가능한 영역이 드러났습니다. TOMG-Bench는 LLM을 통한 분자 발견의 가능성을 확대시키며, 향후 다양한 연구 분야에서의 돌파구를 제공할 수 있는 기반을 마련합니다.



### Learning to Generate Research Idea with Dynamic Contro (https://arxiv.org/abs/2412.14626)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 발전이 과학적 발견을 가속화할 수 있는 잠재력을 보여주고, 연구 아이디어 생성 과정의 자동화를 강조합니다. 특히, 기존의 프로그래밍 기반 접근 방식을 넘어서는 새로운 방법론을 제안하며, LLM을 세밀하게 조정하여 보다 뛰어난 아이디어 제안자로 만드는 기법을 소개합니다.

- **Technical Details**: 제안된 프레임워크는 두 단계로 구성되어 있습니다. 첫 번째 단계인 Supervised Fine-Tuning (SFT)에서는 연구 논문과 후속 아이디어 쌍에서 기초 패턴을 학습합니다. 두 번째 단계에서는 Reinforcement Learning (RL)을 이용하여 다양한 보상 모델링을 통해 생성된 아이디어를 평가하고 최적화합니다. 이 방법에는 차원 컨트롤러가 포함되어 있어 생성 과정에서의 동적 조정이 가능합니다.

- **Performance Highlights**: 이 프레임워크는 혁신성, 실행 가능성, 효과성 간의 균형적인 접근 방식을 제공하며, 고품질 결과를 달성합니다. 또한, 문장 수준의 디코더를 통해 맥락 인식 강조가 이루어져, 생성된 아이디어의 품질을 높이는 데 기여합니다.



### How good is GPT at writing political speeches for the White House? (https://arxiv.org/abs/2412.14617)
- **What's New**: 이 연구는 대규모 언어 모델(LLMs)인 GPT의 작성 스타일을 분석하여 최근 미국 대통령들의 연설과 비교합니다. 연구는 레이건부터 바이든까지의 국정연설(State of the Union) 내용을 GPT-3.5 및 GPT-4.0이 생성한 연설과 대조하여 GPT의 독특한 특징을 드러냅니다. 이를 통해 LLM이 인간 작성자와 비교될 수 있는 지점을 조명합니다.

- **Technical Details**: 연구는 GPT가 사용하는 'we'라는 렘마(lemma)의 과잉 사용과 평균적으로 긴 문장을 사용하면서 메시지가 짧은 경향을 보임을 지적합니다. 또한 GPT는 정치적 용어(예: president, Congress), 상징적 용어(예: freedom), 추상적 용어에 더 자주 의존하여 낙관적인 톤을 선택하는 경향이 있습니다. 이러한 경향은 사용하는 저자의 스타일을 부여하더라도 여전히 차별화된 특성을 유지합니다.

- **Performance Highlights**: GPT-3.5와 GPT-4.0은 각각의 독특한 특성을 가지고 있으나, 두 모델 모두 실제 대통령 연설과는 전반적으로 다른 모습을 보입니다. 이 연구는 LLM이 생성하는 텍스트가 인간 작성자의 연설을 얼마나 잘 모방하는지에 대한 통찰을 제공합니다. 결국, GPT는 미국 대통령의 메시지와 비교할 때 여전히 불일치를 나타냅니다.



### HarmonicEval: Multi-modal, Multi-task, Multi-criteria Automatic Evaluation Using a Vision Language Mod (https://arxiv.org/abs/2412.14613)
- **What's New**: 이 논문은 Vision-Language Models (VLMs)에서 생성된 텍스트의 자동 평가를 위한 새로운 메트릭인 HarmonicEval을 제안합니다. 기존의 측정 지표들이 전반적인 품질에만 집중하여 평가의 측면에서 부족함을 드러내는 가운데, HarmonicEval은 기준 기반 점수를 집계하여 전반적인 점수를 생성하는 방식을 채택합니다.

- **Technical Details**: HarmonicEval의 평가는 세 단계로 구성됩니다: 1) 각 기준에 대한 점수를 생성하기 위해 VLM을 프로세스하는 단계, 2) VLM이 생성한 출력 토큰 확률을 기반으로 점수를 안정화하는 단계, 3) 하모닉 가중치를 사용해 최종 점수를 계산하는 단계입니다. 이를 통해 VLM의 텍스트 품질을 다양한 비전-언어 작업에서 평가할 수 있습니다.

- **Performance Highlights**: 실험 결과, HarmonicEval은 기존의 평가 지표들보다 인간의 판단과 더욱 높은 상관관계를 보여줍니다. 또한, 제안된 MMHE 데이터셋을 통해 HarmonicEval이 특정 평가 기준을 간과하는 기존 메트릭들의 단점을 보완하면서도 종합적인 평가를 수행할 수 있음을 입증합니다.



### KARRIEREWEGE: A Large Scale Career Path Prediction Datas (https://arxiv.org/abs/2412.14612)
Comments:
          Accepted at COLING Industry Track

- **What's New**: 우리는 KARRIEREWEGE라는 포괄적이고 공개된 경력 경로 예측 데이터셋을 소개합니다. 이 데이터셋은 50만 개 이상의 경력 경로를 포함하고 있으며, 기존 데이터셋의 크기를 크게 초월합니다. 우리는 이 데이터셋을 ESCO 분류체계에 연결하여 경력 예측을 지원할 수 있는 귀중한 자원을 제공합니다.

- **Technical Details**: KARRIEREWEGE 새로운 경력 예측 데이터셋의 핵심 요소는 비구조적 데이터를 처리하기 위한 KARRIEREWEGE+ 확장입니다. 이 확장은 직무 제목과 설명을 합성하여 만들어져, 직무 이력서에서 흔히 발생하는 자유 텍스트 입력 문제를 해결합니다. 또한 우리는 기존의 최첨단(SOTA) 모델을 벤치마킹하여 경력 예측의 성능이 향상되었음을 확인하였습니다.

- **Performance Highlights**: KARRIEREWEGE와 KARRIEREWEGE+ 데이터셋을 통해 자유 텍스트 사용 사례에서 특히 향상된 성능과 강건성을 관찰할 수 있었습니다. 이러한 개선은 합성된 데이터의 도입에 의해 이루어졌으며, 실제 응용 프로그램의 도전에 보다 적합하도록 기여합니다. 우리의 연구는 경력 경로 예측의 정확도를 높이고 데이터 사용의 유연성을 제공합니다.



### Beyond Guilt: Legal Judgment Prediction with Trichotomous Reasoning (https://arxiv.org/abs/2412.14588)
- **What's New**: 이번 논문에서는 법률 판단 예측(law judgment prediction) 분야에서 최초로 무죄 판결을 포함하는 벤치마크 데이터셋인 LJPIV를 소개합니다. 기존의 법률 대규모 언어 모델(large language models, LLMs)이 3단계의 삼분법적 법리(trichotomous dogmatics)에 필요한 추론 능력이 부족한 문제를 해결하고자 하였습니다. LJPIV는 유죄 판결만을 포함했던 기존 데이터에서 벗어나, 무죄 판결에 대한 세부 레이블을 추가하여 생성되었습니다.

- **Technical Details**: LJPIV를 구성하기 위해 세 가지 인기 있는 벤치마크(data benchmarks) 데이터를 LLM 기반 증강과 수동 검증을 통해 확장했습니다. 데이터셋 구성에는 트리콤티너스(reasoning)에 필요한 정보만을 추출하기 위한 LLM의 세부 조정, 관련 범죄 법률 검색을 위해 RAG(retrieval-augmented generation) 기법의 적용이 포함됩니다. 또한, 무죄 사례에 대한 충분한 논리적 일관성을 보장하기 위해 LLM을 자체 확인하고 여러 번의 수동 검증을 수행했습니다.

- **Performance Highlights**: 실험 결과, 현재 법률 LLM은 LJPIV에서 F1 점수가 0.3 이하로 제시되며 개선의 여지가 많음을 보여주었습니다. LJPIV에 대한 세부 조정을 통해 범죄 판단 예측 정확도가 대폭 향상되었으며, 특히 무죄 판결 사례에서 눈에 띄는 개선이 나타났습니다. 추가적으로, 우리의 트리콤티너스 추론 전략은 법률 판단 성능을 더욱 강화하는 데 기여하였습니다.



### Simulation-Free Hierarchical Latent Policy Planning for Proactive Dialogues (https://arxiv.org/abs/2412.14584)
Comments:
          24 pages, 5 fgiures, AAAI 2025

- **What's New**: 최근 대화 에이전트의 능동적 대화 정책 계획에 대한 연구가 증가하고 있습니다. 이 논문에서는 기존의 수동적인 정책에서 벗어나 원시 대화 기록에서 자동으로 정책을 발견하는 새로운 프레임워크 LDPP(Latent Dialogue Policy Planning)를 제안합니다. 이를 통해 전문가의 개입 없이도 더 세밀하고 실제 상황에 적합한 정책 개발이 가능해집니다. 이 과정은 대화 기록에서 정책을 발굴하는 것부터 시작하여, 정책 계획 학습에 이르기까지 완전히 자동화됩니다.

- **Technical Details**: LDPP 프레임워크는 라텐트 벡터로 표현된 세밀한 정책을 발견하기 위해 변형된 Variational Autoencoder(VAE)를 사용합니다. 이후 라텐트 정책 레이블로 자동 주석이 달린 데이터를 바탕으로 Offline Hierarchical Reinforcement Learning (RL) 알고리즘을 제안하여 정책 계획 능력을 발전시킵니다. 이 과정에서 정책 계획자는 현재 대화 상태에 따라 적절한 라텐트 정책을 결정하고, 이를 통해 적절한 응답 생성을 유도합니다.

- **Performance Highlights**: 실험 결과 LDPP는 ExTES, ESConv, P4G와 같은 세 가지 능동적 대화 벤치마크에서 기존 방법보다 우수한 성과를 보입니다. 특히, LDPP는 18억 개의 파라미터를 가진 LLM을 사용하였음에도 불구하고 ChatGPT를 초과하는 성능을 입증했습니다. 이러한 성과는 라텐트 정책 기반 접근법이 조화로운 응답 생성을 가능하게 한다는 것을 시사합니다.



### CORD: Balancing COnsistency and Rank Distillation for Robust Retrieval-Augmented Generation (https://arxiv.org/abs/2412.14581)
- **What's New**: 본 논문에서는 retrieval-augmented generation (RAG)에서의 position bias 문제를 해결하기 위해 CORD라는 새로운 접근법을 제안합니다. CORD는 Consistency(일관성)와 Rank Distillation(순위 증류)을 균형 있게 조합하여, 이를 통해 더 효율적인 LLM(large language models)을 위한 훈련 방법을 제공합니다. 저자들은 위치를 변형하여 훈련 데이터 세트를 생성하고, 이는 LLM이 모든 가능한 위치에서 정보를 균등하게 학습하는 데 기여합니다.

- **Technical Details**: CORD는 입력 순서의 무작위성을 균형 있게 제어하여, 두 가지 목표를 동시에 달성합니다. 첫째, 예측의 일관성을 증대시키고, 둘째, retriever에서 제공된 순서의 의미 있는 신호를 활용하는 것이며, 이로 인해 RAG의 성능을 향상시킵니다. 특히, perturbation(변형)을 제어할 수 있는 공간을 정의하고, 여기서 동적으로 적절한 수준의 perturbation을 샘플링합니다.

- **Performance Highlights**: 다양한 RAG 벤치마크에서 CORD의 성능이 일관되게 향상됨을 보여줍니다. Empirical results(경험적 결과)은 CORD가 position bias에 강인함을 유지하면서도, 순서에 의존적인 시나리오에서도 우수한 성능을 발휘함을 입증합니다. 시나리오 A와 B에서 각각 작은 perturbation과 큰 perturbation을 샘플링함으로써, CORD는 모든 상황에서 균형 잡힌 효과를 거두었습니다.



### CitaLaw: Enhancing LLM with Citations in Legal Domain (https://arxiv.org/abs/2412.14556)
- **What's New**: 이번 연구에서는 LLM(대형 언어 모델)의 법적 응답 및 인용 생산 능력을 평가하기 위해 CitaLaw를 제안합니다. CitaLaw는 일반인 및 법률 전문가를 위한 다양한 법률 질문 세트를 구성하며, 법률 기사 및 판례를 포함한 참고 자료를 제공합니다. 이 프레임워크는 LLM 기반 시스템이 참조 자료에서 인용을 검색하고, 이 인용을 응답의 해당 문장과 연결할 수 있도록 합니다.

- **Technical Details**: CitaLaw는 일반인과 법률 전문가를 위한 두 가지 하위 집합을 포함하여, 각기 다른 기대를 충족하는 질문들을 생성합니다. 법률 기사는 사용자의 질문에 대한 명확하고 간결한 가이드라인을 제공하며, 판례는 법적 분석과 지원을 위한 구체적인 예시를 제공합니다. 또한, 우리는 전통적인 글로벌 수준의 메트릭(예: MAUVE) 외에도, 인용의 질을 평가하기 위해 삼단 논법(즉, syllogism)에 기반한 평가 방법을 제안합니다.

- **Performance Highlights**: 폭넓은 실험을 통해 법률 기사를 LLM에 통합하는 것이 응답의 품질을 크게 향상시킴을 입증했습니다. 인용을 LLM 입력의 일부로 포함하는 것이 응답 수정 방법보다 일관되게 더 나은 결과를 보여주었습니다. 또한, 인간 평가와 우리의 삼단 논법 기반 방법 간에 강한 상관관계가 있는 것을 확인했습니다.



### ClusterTalk: Corpus Exploration Framework using Multi-Dimensional Exploratory Search (https://arxiv.org/abs/2412.14533)
Comments:
          5 pages, 1 figure

- **What's New**: 이 논문에서는 다차원 탐색(search) 기능을 활용한 ClusterTalk라는 프레임워크를 소개합니다. 이 시스템은 문서 클러스터링(document clustering)과 패싯 검색(faceted search)을 통합하여 사용자가 연구 문헌을 보다 효율적으로 탐색할 수 있도록 안정성을 제공합니다. 특히, 기존의 키워드 기반 방법에 비해 심층적 상호작용(dynamic interaction)이 가능한 점이 새롭습니다.

- **Technical Details**: ClusterTalk 아키텍처는 PubMed 초록을 대상으로 하는 웹 기반 탐색 시스템으로, BERTopic과 LangChain을 사용하여 백엔드 기능을 지원합니다. 사용자는 문서를 클러스터로 그룹화하고 이를 Interactive하게 필터링하여 문서 검색을 수행하세요. 각 논문은 PubMedBERT 모델을 통해 임베딩(embedding)된 후, HDBSCAN을 사용하여 클러스터링 되며, Mixtral 모델을 통해 연구 결과 관련 질문에 대한 답변을 생성합니다.

- **Performance Highlights**: ClusterTalk의 성능은 400만 개의 PubMed 초록을 기반으로 시범적으로 검증되었습니다. 사용자는 특정 클러스터에서 문서를 필터링하고 선택하여 질문을 할 수 있으며, 사용자의 요청에 따라 평균적으로 2초 안에 응답을 받을 수 있습니다. 이 시스템은 의료 연구자들이 특정 주제에 대한 정보 접근성을 높임으로써 연구 효율성을 극대화합니다.



### Multi-Level Optimal Transport for Universal Cross-Tokenizer Knowledge Distillation on Language Models (https://arxiv.org/abs/2412.14528)
Comments:
          Accepted by AAAI 2025

- **What's New**: 이번 논문에서는 Multi-Level Optimal Transport (MultiLevelOT)라는 새로운 기법을 제안하여, 서로 다른 토크나이저(tokenizer)를 사용하는 모델 간의 지식 증류(knowledge distillation)의 범위를 넓힙니다. 기존 방법은 동일한 토크나이저를 사용하는 모델 간에만 적용 가능했으나, MultiLevelOT는 이러한 제약을 극복하고 있습니다. 우리의 접근법은 교사 모델(teacher model)과 학생 모델(student model) 간의 로그잇(logit) 분포를 효과적으로 정렬하여, 다양한 아키텍처의 모델에서 유연하게 사용할 수 있습니다.

- **Technical Details**: MultiLevelOT는 토큰 수준(token level)과 시퀀스 수준(sequence level)에서의 로그잇 분포 정렬을 통해, 교사와 학생 모델 간의 최적 수송(optimal transport) 거리를 동시에 계산합니다. 이를 위해 각각 절대 차이(absolute difference)와 로그 기반 가능도 차이(logarithmic likelihood difference) 형태의 두 가지 비용 행렬(cost matrix)을 사용하여, 구체적인 거리 측정이 이루어집니다. 각 시퀀스 내의 모든 토큰을 함께 최적화하는 방식으로, 전반적인 일관성을 유지합니다.

- **Performance Highlights**: 종합적인 실험 결과, MultiLevelOT는 추출적 QA, 생성적 QA, 요약 등 다양한 자연어 처리(NLP) 작업에서 기존의 교차 토크나이저 지식 증류 방법들보다 우수한 성능을 보였습니다. 이 방법은 서로 다른 모델 패밀리(model families), 아키텍처 및 파라미터 크기에서도 강력함을 유지하고 있습니다. 이를 통해, MultiLevelOT는 지식 증류 분야에서 새롭게 제안된 접근 방식으로 자리매김할 것입니다.



### PA-RAG: RAG Alignment via Multi-Perspective Preference Optimization (https://arxiv.org/abs/2412.14510)
- **What's New**: 본 논문에서는 Retrieval-Augmented Generation (RAG)의 생성기를 최적화하는 새로운 방법인 다중 관점 선호 정렬(Multiple Perspective Preference Alignment, PA-RAG)을 제안합니다. 기존 RAG 시스템의 한계인 응답의 유용성, 견고성, 인용 품질을 개선하기 위해 다양한 품질의 응답을 샘플링하여 높은 품질의 instruction fine-tuning 데이터와 선호 데이터셋을 구축했습니다. PA-RAG는 일반적인 LLMs(large language models)에 비해 더욱 효과적인 RAG 응답 생성을 가능하게 합니다.

- **Technical Details**: PA-RAG는 두 가지 훈련 단계로 구성되어 있습니다. 첫 번째 단계는 기본 능력 훈련으로, generator가 문서를 활용하고 인용하는 기본적인 능력을 습득하도록 합니다. 두 번째 단계인 다중 관점 선호 최적화 단계에서는 Direct Preference Optimization (DPO)을 사용하여 다양한 관점에서 선호 정보를 학습합니다. 이 과정에서 응답의 유용성, 견고성, 인용 품질을 각각 향상시키는 세 가지 하위 단계로 구성됩니다.

- **Performance Highlights**: PA-RAG를 적용한 실험에서는 평균 13.97%의 정확성 개선과 49.77%의 인용 조합률, 39.58%의 인용 정확성이 향상되었습니다. 이는 기존의 SFT(supervised fine-tuning)나 추가 단계만을 사용하는 방법에 비해 상당한 성과로, PA-RAG의 효과성을 입증합니다. 연구진은 특히 선호 최적화의 각 관점에서 영향을 미치는 효과를 보여주며, PA-RAG의 모든 훈련 데이터를 공개할 예정입니다.



### Do Large Language Models Defend Inferentialist Semantics?: On the Logical Expressivism and Anti-Representationalism of LLMs (https://arxiv.org/abs/2412.14501)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 등장으로 인해 언어 철학이 후기 인류중심적(post-anthropocentric)으로 전환되고 있음을 강조합니다. 연구자들은 LLM이 인간의 언어 능력에 버금가는 것으로 여겨지며, 기존의 분포 의미론(distributional semantics) 이외의 대안적인 기초 의미론을 탐구하고 있습니다. 이 논문은 로버트 브랜덤(Robert Brandom)의 추론주의 의미론(inferentialist semantics)을 LLM의 적합한 기초 의미론으로 제안하고 있습니다.

- **Technical Details**: LLM들은 일반적으로 Transformer 아키텍처에 기반하고 있으며, GPT-4, Claude 및 Llama와 같은 모델들이 있습니다. 이 아키텍처는 여러 층과 헤드를 갖춘 다중 헤드 다중 층 Transformer로, 입력 정보의 구조를 보존합니다. 특히, 입력 벡터는 단어와 위치 임베딩의 가중합으로 구성되며, 서로 다른 부분 공간에서 정보를 독립적으로 유지할 수 있도록 설계되었습니다.

- **Performance Highlights**: LLM의 특성은 언어 철학에서의 의미 외재주의(semantic externalism)와 조합론(compositionality) 같은 주류 가정에 도전합니다. 이 논문은 언어의 반-표상론적(anti-representationalist) 관점에서 LLM의 특성을 설명하며, 이는 언어 철학의 새로운 발전으로 이어질 가능성을 시사합니다. LLM의 기능과 특성을 설명하는 데 있어 분포 의미론보다 브랜덤의 추론주의 의미론이 더 효과적이라는 주장을 합니다.



### Why We Build Local Large Language Models: An Observational Analysis from 35 Japanese and Multilingual LLMs (https://arxiv.org/abs/2412.14471)
Comments:
          Preprint. Under review

- **What's New**: 이 논문에서는 일본어, 영어 및 다국어 대형 언어 모델(LLMs)의 성능을 비교 분석하며, 일본어를 로컬 언어로 설정하였습니다. 연구자는 35개의 모델을 평가하고 19개의 벤치마크를 사용하여 일본어와 영어의 언어적 전이 가능성과 관련된 능력 요소를 도출하였습니다. 이는 특정 언어의 훈련이 모델의 성능에 미치는 영향을 정량적으로 분석한 첫 번째 연구로, 일본어 데이터에 대한 기존 연구와는 차별화됩니다.

- **Technical Details**: 연구는 LLM의 성능을 주성분 분석(Principal Component Analysis, PCA)를 통해 평가하고, 일본어 훈련 데이터가 일본어 관련 질문-답변 및 번역 과제에 미치는 영향을 조사하였습니다. 연구팀은 일본어 텍스트의 연산 예산(computational budget)과 능력 요소 간의 로그-선형 관계를 분석하여, 일본어 훈련이 필요한 특정 능력 요소를 규명하였습니다. 또한, 여러 디자인 선택을 기반으로 다양한 모델의 성능을 비교하였습니다.

- **Performance Highlights**: 경험적으로, 영어로 훈련된 LLM은 일본어 학문 과제에 대한 성과를 향상시킴을 발견했습니다. 반면, 일본어 텍스트로의 훈련은 일본어 지식에 대한 질문-답변 작업과 영어-일본어 번역에서 성과를 개선하는 것으로 나타났습니다. 결과적으로, 일본어 능력 요소는 일본어 데이터의 양에 비례하여 증가한다는 것을 확인하였습니다.



### Agent-SafetyBench: Evaluating the Safety of LLM Agents (https://arxiv.org/abs/2412.14470)
Comments:
          23 pages, 9 figures

- **What's New**: 이 논문에서는 Agent-SafetyBench라는 새로운 벤치마크를 소개하여 대형 언어 모델(LLM) 에이전트의 안전성을 평가합니다. 기존에 LLM의 안전성을 평가할 수 있는 종합적인 기준이 없었던 문제를 해결하고자 349개의 상호작용 환경과 2,000개의 테스트 케이스를 포함하고 있습니다. 이 벤치마크는 안전 위험의 8개 카테고리를 평가하며, LLM 에이전트의 안전성과 관련된 다양한 실패 모드를 다루고 있습니다.

- **Technical Details**: Agent-SafetyBench는 기존 LLM 에이전트의 안전성을 정량적으로 분석하기 위한 도구입니다. 연구를 통해 16개 인기 LLM 에이전트를 평가한 결과, 어떤 에이전트도 60% 이상의 안전 점수를 달성하지 못했습니다. 이 분석은 LLM 에이전트의 두 가지 주요 안전 문제인 내구성 부족과 위험 인식 부족을 강조합니다.

- **Performance Highlights**: 연구 결과, LLM 에이전트의 성능이 상당한 안전 문제를 안고 있음을 보여주며, 방어 프롬프트(defense prompts)만으로는 이러한 문제를 해결하기에 충분하지 않다는 것을 시사합니다. 따라서, 보다 진보되고 내구성 있는 전략이 필요함을 강조하며 안전성 평가 및 개선을 위한 추가 연구를 촉진하기 위한 Agent-SafetyBench를 제공합니다.



### From Human Annotation to LLMs: SILICON Annotation Workflow for Management Research (https://arxiv.org/abs/2412.14461)
- **What's New**: 이번 연구에서는 LLM(대규모 언어 모델)을 활용한 텍스트 주석 및 분석의 체계적인 워크플로인 SILICON을 소개합니다. 이 워크플로는 인간 주석의 원칙과 시스템적인 프롬프트 최적화(prompt optimization) 및 모델 선택(model selection)을 통합하여, LLM 기반 주석의 적합성 및 진행 방법을 정립합니다. 이를 통해 유연하고 신뢰할 수 있는 연구 방법론을 제공하여 관리 연구의 방법적 공백을 해소하고자 합니다.

- **Technical Details**: SILICON은 주석 지침 개발, 고품질 인간 기준 설정, 프롬프트 최적화, 그리고 LLM 간의 재현성(reproducibility) 확보와 같은 다양한 도전 과제를 다룹니다. 연구팀은 7개의 사례 연구를 통해 SILICON의 실용성을 증명하였으며, 여기에는 사업 제안 평가(business proposal evaluation), 대화 의도(dialog intent) 분석, 리뷰 속성 탐지(review attribute detection) 등이 포함됩니다.

- **Performance Highlights**: 이 연구의 주요 발견으로는 주석 가이드라인 합의(validating annotation guideline agreement)의 중요성, 전문가 개발 인간 기준이 크라우드소싱된 기준보다 우수함이 강조됩니다. 프롬프트 최적화의 반복적인 특성과 여러 LLM을 테스트(testing multiple LLMs)하는 필요성도 확인하였습니다. 또한, 프롬프트 및 모델 간 LLM 출력을 경험적으로 비교하기 위한 회귀 기반 방법론(regression-based methodology)을 제안하여 관리 연구의 과학적 엄격성을 유지하는 재현 가능한 프로세스를 수립했습니다.



### ORBIT: Cost-Effective Dataset Curation for Large Language Model Domain Adaptation with an Astronomy Case Study (https://arxiv.org/abs/2412.14436)
- **What's New**: 이번 논문에서는 전문 지식이 필요한 작업을 위해 고품질의 도메인 특화 훈련 데이터의 필요성을 강조하고 있습니다. 기존의 범용 모델은 일반적인 작업에 대해서는 뛰어난 성능을 보이나, 우주 과학과 같은 특정 분야에서는 깊이 있는 이해도가 부족한 경우가 많습니다. 이를 해결하기 위해 저자들은 ORBIT이라는 새로운 데이터 커레이션 방법론을 제안하고, 이를 통해 노이즈가 많은 웹 소스로부터 고품질의 대량 데이터셋을 커스터마이즈할 수 있음을 보여줍니다.

- **Technical Details**: ORBIT은 대규모 웹 데이터셋을 효율적으로 필터링하기 위해 임베딩 기반 유사성 매칭과 BERT 기반 회귀 모델을 결합한 새로운 프레임워크입니다. 이 방법론은 의미적 관련성과 교육적 가치를 모두 고려하여 커뮤니케이션 및 데이터 품질을 보장합니다. 저자들은 이 방법론을 우주 과학을 중심으로 적용하며, FineWeb-Edu에서 10B 토큰의 고품질 데이터셋을 구축했습니다.

- **Performance Highlights**: ORBIT로 훈련된 LLaMA-3-8B 모델인 Orbit-LLaMA는 MMLU 우주 과학 벤치마크에서 69%에서 76%로 성능이 향상된 것을 확인했습니다. 또한, Orbit은 기존의 AstroLLaMA 모델보다 훨씬 높은 평가를 받으며, GPT-4o 평가에서도 73%의 선호도를 기록하여 성공적인 성능 향상을 입증했습니다. 이 방법론은 우주 과학뿐만 아니라 법률 및 의학 분야에서도 데이터 품질을 대폭 향상시켰습니다.



### All-in-One Tuning and Structural Pruning for Domain-Specific LLMs (https://arxiv.org/abs/2412.14426)
- **What's New**: 이 논문은 ATP(All-in-One Tuning and Structural Pruning)라는 새로운 프레임워크를 제안합니다. 이 방법은 기존의 두 단계 방식이 아닌, 구조적 프루닝(structural pruning)과 파라미터 파인튜닝(parameter fine-tuning)을 통합하여 한 단계에서 수행합니다. ATP는 훈련 가능한 프루닝 결정 생성기를 통해 최신의 최적 서브구조를 동적으로 식별하여, 도메인 특화 애플리케이션에서 성능 저하를 최소화합니다.

- **Technical Details**: 고전적인 LLM 프루닝 기법은 대개 사전 훈련된 모델에서 효과적으로 파라미터를 제거하는 두 단계로 진행됩니다. 하지만 ATP는 훈련 중에 프루닝 결정이 업데이트되어 새로운 서브구조를 탐색하게 합니다. 또한, LoRA-aware forward 패스와 구조적 희소성 정규화를 포함하여 훈련된 프루닝 결정을 기반으로 파라미터의 기여를 점진적으로 제거합니다.

- **Performance Highlights**: 실험 결과, ATP는 법률 및 헬스케어 도메인에서 기존의 두 단계 프루닝 방법을 초월하는 성능을 보였습니다. 특히, LLaMA2-7B 및 LLaMA3-8B 모델의 40%를 프루닝 했을 때 각각 88% 및 91%의 성능을 회복하는 결과를 보였습니다. 따라서 ATP는 제한된 데이터 환경에서도 도메인 특화 모델의 효율적인 배포 가능성을 제시합니다.



### ECG-Byte: A Tokenizer for End-to-End Generative Electrocardiogram Language Modeling (https://arxiv.org/abs/2412.14373)
Comments:
          26 pages, 17 figures

- **What's New**: 이번 연구에서는 ECG(심전도) 데이터를 직접적으로 토큰으로 변환하여 LLM(대형 언어 모델)을 훈련시키는 'ECG-Byte'라는 새로운 접근법을 소개합니다. 이전에는 ECG에 대한 해석 가능성이 제한적이었지만, ECG-Byte를 통해 ECG 신호를 원래 신호로 역 변환할 수 있어 해석이 용이해졌습니다. 이 방법은 자연어 생성(NLG) 작업에서 효율성을 높이고, 경쟁력 있는 성능을 보여줍니다.

- **Technical Details**: ECG-Byte는 전통적인 BPE(Byte Pair Encoding) 토크나이저를 기반으로 하여 ECG 신호를 압축하고 인코딩하여 생성적인 ECG 언어 모델링을 위한 처리 파이프라인을 구축합니다. 이는 자가 지도 학습(self-supervised learning) 목표 없이, 직접적인 엔드-투-엔드(end-to-end) 훈련을 가능하게 합니다. 또한, ECG 신호를 토큰으로 변환하여 텍스트와 결합할 수 있어, 더 효율적인 훈련과 해석이 가능합니다.

- **Performance Highlights**: ECG-Byte를 활용한 결과, 기존의 두 단계 훈련 방식에 비해 시간은 절반으로 감소하고, 필요한 데이터 양도 약 48%까지 줄일 수 있었습니다. 이러한 접근법은 특히 심전도 분석의 자동화에 있어 더 나은 해석 가능성과 효율성을 제공하며, 임상적 유용성을 향상시킵니다.



### Memorization Over Reasoning? Exposing and Mitigating Verbatim Memorization in Large Language Models' Character Understanding Evaluation (https://arxiv.org/abs/2412.14368)
- **What's New**: 최근 대형 언어 모델(LLMs)의 캐릭터 이해(task) 성능이 인상적이지만, 메모리(memorizations) 기반의 과도한 의존성이 문제로 지적되고 있다. 이 논문에서는 'gist memory'를 주된 메커니즘으로 제안하여 모델이 내용을 더 깊이 이해하도록 유도하는 방법을 소개하고 있다. 이 방법은 캐릭터 이해 평가에서 기계적인 메모리 의존을 줄이는 동적 요소를 포함하며, 메모리에 의한 정확도를 96%에서 72%로 감소시켰다.

- **Technical Details**: 논문에서는 캐릭터 이해 작업을 수행하는데 있어 모델이 'gist memory'와 'verbatim memory' 간의 균형을 맞추는 것이 중요하다고 강조하고 있다. 우리는 6가지 캐릭터 이해 작업을 통해 모델의 퍼포먼스를 평가했으며, 캐릭터 이름, 장소 이름 및 문장에서의 특정 표기법을 변경하면서 캐릭터 간의 관계와 중요한 사건은 보존하는 접근법을 사용하였다. 실험 결과, 맥락의 유사성에 의존하기 보다는 패턴을 통한 추론을 장려할 수 있는 방법이 확인되었다.

- **Performance Highlights**: 우리의 접근 방식은 캐릭터 이해 작업에 있어서 기존의 벤치마크(memoization)에 기반한 평가로부터의 성능 저하를 경험했다. 연구 결과, 명시적 메모리 의존도가 높아질수록 모델이 더 많은 추론을 요구하게 되어 성능이 0.7%에서 17.2%까지 감소함을 확인하였다. 이는 데이터 오염(data contamination) 문제를 강조하며, 기존 벤치마크에서 진정한 캐릭터 이해가 아닌 메모리 기반 성능을 측정하고 있다는 점에서 중요한 시사점을 제공한다.



### State Space Models are Strong Text Rerankers (https://arxiv.org/abs/2412.14354)
Comments:
          The first two authors contributed equally, order decided randomly

- **What's New**: 이 논문에서는 기존 트랜스포머 아키텍처에 대한 대안으로 주목받고 있는 상태 공간 모델(state space models, SSMs), 특히 Mamba 모델을 소개합니다. Mamba는 긴 컨텍스트를 이해하고 정밀한 쿼리-문서 상호작용을 다루는 텍스트 재순위 작업에서의 유용성을 평가하고자 합니다. 연구 결과, Mamba는 비슷한 크기의 트랜스포머 모델과 경쟁력 있는 성능을 보여주지만, 훈련 및 추론 효율성 면에서는 다소 부족한 것으로 나타났습니다.

- **Technical Details**: Mamba 모델은 효율적인 표현을 위해 선택적 상태를 인코딩하며, O(1) 시간 복잡성으로 인퍼런스를 수행합니다. 연구에서는 Mamba-1과 Mamba-2 모델을 비교하고, 다양한 아키텍처와 전처리 목표에 따라 성능을 평가합니다. 특히 Mamba 모델은 트랜스포머와 비교할 때 상대적으로 훈련 및 추론에서 더 비효율적임을 보여주며, Mamba-2가 성능과 효율성 모두에서 Mamba-1보다 우수한 결과를 보였습니다.

- **Performance Highlights**: Mamba 기반 언어 모델은 트랜스포머 기반 모델과 유사한 수준의 텍스트 재순위 성능을 달성했습니다. 하지만 이들은 이론적으로 더 나은 복잡성을 가진 반면, 실제로는 IO-aware 최적화(예: flash attention)를 사용하는 트랜스포머 아키텍처에 비해 효율성이 낮았습니다. 앞으로 정보 검색(Information Retrieval, IR) 작업을 위한 트랜스포머 대안 아키텍처의 발전 방향에 대한 논의도 포함되어 있습니다.



### A Survey on LLM Inference-Time Self-Improvemen (https://arxiv.org/abs/2412.14352)
Comments:
          The first two authors contribute equally

- **What's New**: 최근 LLM(대형 언어 모델)에서 테스트 시간 중의 인퍼런스(inference) 개선을 위한 기술이 주목받고 있습니다. 이 연구에서는 인퍼런스 시간 중 자기 개선(self-improvement)의 세 가지 주요 관점을 다룹니다: 독립적인 자기 개선, 상황 인식(self-awareness) 자기 개선, 그리고 모델 보조(self-aided) 자기 개선이 그것입니다. 각 관점에 대한 최근 연구를 포괄적으로 리뷰하며, 과거 연구의 한계와 도전 과제를 논의합니다.

- **Technical Details**: 리뷰에서는 독립적 자기 개선에서 제약 디코딩(constrained decoding), 대조 디코딩(contrastive decoding), 최소 베이즈 리스크 디코딩(minimum Bayes-risk decoding) 등 다양한 기술을 포함합니다. 또한, 상황 인식을 통한 자기 개선을 위한 프로그래밍(prompting), 방해(prompt disturbance) 및 검색 기반 접근법(retrieval-based approach)에 대해서도 살펴봅니다. 마지막으로, 모델 보조 자기 개선에서 전문가(expert)와 비전문가(anti-expert) 접근법, 정렬(alignment) 기술을 포함한 여러 방법론을 제시합니다.

- **Performance Highlights**: 이 연구는 LLM의 인퍼런스 개선을 위한 다양한 방법론에 대한 심층적인 분류법을 제시하며, 이들 간의 상호작용을 이해하는 데 기여합니다. 앞으로의 연구 방향에 대한 통찰도 제공하여, 향후 LLM 기술 개선에 기여할 수 있는 가이드라인을 마련합니다. 전체적으로 이 논문은 현재 LLM 기술의 상태와 그 발전 가능성을 폭넓게 탐구하여 연구자와 개발자 모두에게 유용한 자료가 될 것입니다.



### Is Peer-Reviewing Worth the Effort? (https://arxiv.org/abs/2412.14351)
Comments:
          The 31st International Conference on Computational Linguistics (COLING 2025)

- **What's New**: 이 논문은 동료 평가(peer-reviewing)가 중요한 논문을 식별하는 데 얼마나 효과적인지를 예측하는 작업으로 다룹니다. 연구자들은 어느 논문이 향후 높은 인용(citation)을 받을지를 예측하기 위해 논문의 발표 장소(venue)와 조기 인용(early returns)의 관계를 분석하였습니다. 결과적으로 조기 인용이 이후 인용과 더 큰 상관관계를 가지고 있다는 점을 강조하며, 이는 향후 연구에서도 더욱 중요한 방향성이 될 수 있음을 시사합니다.

- **Technical Details**: 연구에서는 조기 인용과 발표 장소를 기준으로 논문을 집계하고, 네 번째 해의 인용 수치를 h-index와 impact(μ)를 통해 요약합니다. 논문은 다양한 조건과 다양한 출처(ACL, PubMed, ArXiv)의 논문을 포함하여 반복 검증을 통해 조기 인용이 발표 장소보다 더 효과적이라는 결론을 도출합니다. 이에 따라 논문들은 초기 인용을 통해 우선순위를 매기는 것이 더 효과적이며, 이는 더 포괄적이고 강건하다는 설명이 포함되어 있습니다.

- **Performance Highlights**: 조기 인용이 있는 덜 선택적인 발표 장소(Workshops/ArXiv)의 논문들이 더 선택적인 발표 장소의 논문들보다 종종 더 높은 인용 수를 기록하는 경향이 있음을 보여주고 있습니다. 연구 결과, 초기 인용에 기반한 우선순위가 논문 선택에 있어서 상당한 유용성을 제공하며, 동료 평가의 효과성에 대한 새로운 관점을 제공합니다. 이러한 통찰력은 향후 학술 출판 및 심사 과정에서 중요한 참조로 작용할 것으로 기대됩니다.



### Semantic Role Labeling of NomBank Partitives (https://arxiv.org/abs/2412.14328)
Comments:
          SUMEval-2: The 2nd Workshop on Scaling Up Multilingual & Multi-Cultural Evaluation at the 31st International Conference on Computational Linguistics (COLING 2025)

- **What's New**: 이번 논문은 영어의 부분 명사에 대한 의미역 레이블링(Semantic Role Labeling) 기술을 다루고 있습니다. 특히 부분 명사와 관련된 ARG1 및 연관 문제에 대한 두 가지 작업(task)에 대한 결과를 보고하였습니다. 전통적인 기계 학습과 Transformer 기반의 기계 학습 기법을 활용한 시스템을 설명하고 있으며, 최고 성능을 기록한 시스템은 Penn Treebank의 'gold' 파서를 사용하여 F1 점수 91.74%를 달성했습니다.

- **Technical Details**: 이 연구에서 사용된 시스템은 BERT 기반의 구성요소(Deep Learning 1)와 기능 기반 구성요소로 이루어진 앙상블 시스템입니다. Deep Learning 2는 언어학적 특징의 유용성을 평가하기 위해 보조 작업(auxiliary task)으로 사용되었습니다. Deep Learning 2에서 사용된 언어학적 특징은 앙상블 시스템의 기능 기반 구성요소의 일부로, 제공된 비금 고(gold) 데이터와 비금 고 데이터(non-gold data)로부터 예측 모델링을 지원합니다.

- **Performance Highlights**: 연구 결과, 비금 고 데이터에 대한 예측 성능에서 초기 모델들이 2-3% 수준의 F-score 감소를 보였습니다. 이 연구는 금 고 데이터가 모델 성능 향상에 중요한 역할을 하였음을 시사합니다. 그러나 부분 명사 작업은 복잡성이 더 높아 일반적으로 F-score가 낮았습니다. 논문에서는 다양한 모델의 성능에 영향을 미치는 요인들에 대한 깊은 분석이 필요함을 강조하고 있습니다.



### The Role of Handling Attributive Nouns in Improving Chinese-To-English Machine Translation (https://arxiv.org/abs/2412.14323)
Comments:
          18th Workshop on Building and Using Comparable Corpora (BUCC) at the 31st International Conference on Computational Linguistics (COLING 2025)

- **What's New**: 이 연구에서는 문법적인 차이가 큰 언어 간 번역에서 발생하는 도전 과제를 해결하기 위해 Chinese의 속성 명사(attributive nouns) 번역 문제에 초점을 맞추고 있습니다. Penn Chinese Discourse Treebank에서 생략된 입자 X('DE')를 수동으로 삽입하여 뉴스 기사 제목의 데이터셋을 개발하였습니다. 이 데이터셋은 Hugging Face의 중국어-영어 번역 모델을 세심하게 조정(fine-tune)하는 데 사용되었습니다.

- **Technical Details**: 귀 연구는 중국어의 속성 명사가 영어 번역에서 자주 발생하는 모호성을 어떻게 해결하는지를 다루고 있습니다. 특정 function word인 'DE'의 처리를 개선하기 위해 만들어진 데이터셋은 기계 번역 시스템의 성능을 향상시킵니다. 이러한 접근법은 이전 연구들이 제안한 광범위한 전략들과 보완적으로 작용하면서도, 중국어-영어 번역에서 빈번하게 발생하는 오류 유형을 구체적으로 해결합니다.

- **Performance Highlights**: 이 연구는 정확한 기계 번역을 위해 특정 명사의 번역 실수를 줄이는 데 중요한 실용적 개선을 제공합니다. 이 방법은 번역 모델의 정확도를 높이는 데 기여하며, 기계 번역 시스템의 전반적인 성능 향상에 기여할 것으로 기대됩니다.



### Multi-OphthaLingua: A Multilingual Benchmark for Assessing and Debiasing LLM Ophthalmological QA in LMICs (https://arxiv.org/abs/2412.14304)
Comments:
          Accepted at the AAAI 2025 Artificial Intelligence for Social Impact Track (AAAI-AISI 2025)

- **What's New**: 본 연구는 다국어 기반의 안과 질문-응답(QA) 벤치마크인 Multi-OphthaLingua를 처음으로 소개하고 있습니다. 이 데이터셋은 영어, 스페인어, 필리핀어, 포르투갈어, 만다린, 프랑스어, 힌디 등 7개 언어로 총 1184개의 질문을 포함하며, 서로 간의 직접적인 언어 비교를 가능하게 합니다. 이를 통해 의료 분야에서 LLM이 갖는 언어적 편향을 평가하고 개선하는 데 기여하고자 합니다.

- **Technical Details**: Multi-OphthaLingua 데이터셋은 각 언어에서 검사 및 평가된 질문-답변 쌍으로 구성되어 있으며, 언어별로 공정성을 확보하기 위해 네이티브 스피커가 검증했습니다. 또한, 연구진은 LLM의 성능을 비교하기 위해 Llama-2, GPT 3.5 등 6개 인기 LLM을 평가하였고, 이 과정에서 언어에 따른 성능 격차와 의료 도메인 지식 부족 문제를 발견했습니다. 이러한 문제를 해결하기 위해, CLARA라는 새로운 비편향 처리 방법을 제안하고 다중 에이전트 접근 방식을 도입했습니다.

- **Performance Highlights**: 연구 결과, CLARA 메소드는 6개 언어에서 모든 LLM의 질문-응답 정확도를 향상시키는 동시에 언어 간 성능 격차를 줄이는 데 성공했습니다. 특히, LMIC 언어의 경우 LLM 성능이 저조했으나, CLARA를 통해 이러한 격차가 크게 개선되었습니다. 그리고 이 연구는 의료 분야의 언어적 공정성을 높이기 위한 새로운 지침을 제시하며, 글로벌한 LLM 활용 가능성을 확대하는 데 중요한 기초 자료를 제공합니다.



### Fake News Detection: Comparative Evaluation of BERT-like Models and Large Language Models with Generative AI-Annotated Data (https://arxiv.org/abs/2412.14276)
Comments:
          Accepted in Knowledge and Information Systems Journal

- **What's New**: 이번 연구는 BERT 계열의 인코더 전용 모델과 자율 회귀 디코더 전용 대규모 언어 모델(LLM)을 비교하여 가짜 뉴스 탐지의 효과성을 평가하고 있습니다. 또한, GPT-4의 지원을 받아 라벨링된 뉴스 데이터셋을 소개하여 인간 전문가에 의해 검증된 신뢰 가능한 결과를 제공합니다. 수집된 데이터셋을 활용하여 두 모델 가족 간의 성능을 분석하고, AI 기반 주석과 인간 감독을 결합한 접근 방식이 가짜 뉴스 탐지에 효과적임을 강조합니다.

- **Technical Details**: 가짜 뉴스 탐지에서의 ML 기반 접근 방식의 주요 과제는 정확하고 신뢰할 수 있는 데이터 라벨링의 가용성입니다. 연구진은 10,000개의 뉴스 기사를 수집하고 최신 LLM인 OpenAI GPT-4를 이용해 이들을 라벨링했습니다. BERT 계열의 모델과 LLM의 성능을 비교하고, 특히 BERT 모델이 분류 작업에서 뛰어난 성능을 나타내는 반면, LLM은 텍스트 변동에 대해 더 강한 견고성을 보임을 분석하였습니다.

- **Performance Highlights**: BERT 계열 모델은 일반적으로 분류 작업에서 LLM에 비해 더 나은 성능을 발휘하지만, LLM은 텍스트의 작은 변화에 대해 강인한 성능을 보여줍니다. AI 라벨은 인간의 감독 하에 이루어졌을 때, 약한 감독(원거리 감독)에 의해 얻어진 라벨보다 더 정확한 분류 결과를 나타냈습니다. 이에 따라 데이터의 대량 처리와 가짜 뉴스의 탐지를 위한 강력한 AI 탐지 방법을 통해 사회적 인식을 높이는 데 집중하고 있습니다.



### Tokenisation is NP-Comp (https://arxiv.org/abs/2412.15210)
- **What's New**: 이 연구에서는 데이터셋을 최대 $
abla$ 심볼로 압축하는 두 가지 변형의 토크나이제이션(tokenisation) 문제의 NP-완전성을 증명합니다. 직접적인 토크나이제이션(direct tokenisation) 또는 병합 작업의 선택을 통해 가능하죠. 압축 성능이 언어 모델의 향상과 연관된 점을 고려할 때, 이 연구는 효율적인 최적 토크나이저(tokeniser)를 찾는 것이 얼마나 복잡한지를 보여줍니다.

- **Technical Details**: 우리는 두 가지 변형의 토크나이제이션 문제, 즉 직접적인 단어 선택을 사용하는 것과 병합 작업으로 접근하는 방법의 NP-난해성을 입증합니다. 이 문제들은 최대 2-만족도(max 2-SAT) 문제로부터의 변환을 통해 증명됩니다. 이 결과는 최적 토크나이저를 찾기 위한 효율적인 알고리즘이 발견되기 어려움을 시사하며, 비슷하게 BPE(byte pair encoding) 및 UnigramLM과 같은 근사 알고리즘에 집중해야 함을 나타냅니다.

- **Performance Highlights**: 토크나이저 선택은 언어 모델의 품질에 직접적인 영향을 미칠 수 있습니다. 잘못된 토크나이저 선택은 숫자나 글자의 카운팅 성능에 부정적인 영향을 미칠 수 있으며, 이는 예를 들어 GPT-4 조차도 특정 단어의 글자 수를 정확히 세지 못하는 사례에서 나타납니다. 이 연구는 최적의 토크나이저 선택을 위해 압축 성능을 극대화하는 방향으로의 접근을 제시하며, 이는 훈련 및 추론의 효율성을 높이고 downstream 성능 향상에 기여할 수 있음을 강조합니다.



### Critical-Questions-of-Thought: Steering LLM reasoning with Argumentative Querying (https://arxiv.org/abs/2412.15177)
- **What's New**: 이번 논문에서는 논리적 및 수학적 추론에서 어려움을 겪고 있는 최신 인공지능 모델들을 개선하기 위해 새로운 접근 방식인 Critical-Questions-of-Thought (CQoT)를 도입합니다. CQoT는 이러한 모델들이 추론 과정을 점검하고 논리적 실수를 수정할 수 있도록 도와주는 비판적 질문을 활용하는 방법입니다. 이 연구는 Toulmin의 논증 모델을 기반으로 하여 인공지능의 사고 과정을 향상시키는 데 중점을 두고 있습니다.

- **Technical Details**: CQoT 접근방식은 논증 이론의 핵심 개념들을 활용하여 LLM의 사고 프로세스가 보다 선명하게 드러나도록 합니다. 이 과정에서 LLM은 제시된 전제(premises)로부터 결론(conclusion)을 이끌어내는 방식을 따릅니다. 즉, 주어진 데이터에 대한 주장이 적절한 근거(warrant)와 경험적 데이터로 뒷받침되어야 한다는 Toulmin의 관점을 취합니다.

- **Performance Highlights**: 이 연구에서는 MT-Bench Reasoning 및 Math 과제를 통해 CQoT 접근방식이 LLM의 기본 성능을 상당히 개선시킨다는 것을 보여줍니다. 더 나아가, '시간을 주는 것'이 논리적 사고를 향상시키는 데 긍정적인 기여를 한다는 기존의 테스트 타임 컴퓨트 가설을 입증하고 있습니다. 연구 결과와 평가 점수는 GitHub에서 확인할 수 있습니다.



### Prompt-A-Video: Prompt Your Video Diffusion Model via Preference-Aligned LLM (https://arxiv.org/abs/2412.15156)
- **What's New**: 이 논문은 비디오 생성 모델의 텍스트 입력을 향상시키기 위한 새로운 프로프트 최적화 시스템, Prompt-A-Video를 소개합니다. 이 시스템은 사용자의 입력을 자동적으로 보완하여 모델의 선호도에 맞춘 고품질 비디오 중심의 프로프트를 생성합니다. 할당된 보상을 기반으로 한 이중 단계 최적화 프로세스는 비디오 생성 품질을 크게 향상시키는 데 기여합니다.

- **Technical Details**: Prompt-A-Video는 사용자 제공 프로프트를 진화 알고리즘을 통해 자동으로 개선하는 보상 안내형 프로프트 진화 파이프라인을 특징으로 하며, 원본과 보정된 프로프트의 데이터 쌍을 생성합니다. 그런 다음, Supervised Fine-Tuning(SFT) 모델을 사용하여 다차원 보상을 평가하고, Direct Preference Optimization(DPO)을 통해 모델의 선호도에 맞게 정렬됩니다. 이러한 과정은 비디오 생성을 위한 최적의 프로프트를 생성하는 데 중점을 둡니다.

- **Performance Highlights**: Prompt-A-Video는 다양한 평가 지표에서 비디오 생성 성능을 향상시키는 데 뛰어난 효율성을 입증하였습니다. 또한, 텍스트-이미지 전환 시나리오에서도 상당한 개선을 보여 주며, 이는 모델의 강인함과 일반성을 나타냅니다. 이 시스템은 비디오 생성의 새로운 가능성을 열어주는 유망한 방법론으로 자리매김하고 있습니다.



### Associative memory inspires improvements for in-context learning using a novel attention residual stream architectur (https://arxiv.org/abs/2412.15113)
Comments:
          18 pages, 6 figures, 3 tables

- **What's New**: 이번 논문에서는 대규모 언어 모델(LLM)의 문맥 내 학습(in-context learning, ICL) 능력과 생물학적 기억 체계의 연결을 제안합니다. 주목할 만한 점은 LLM의 핵심 구성 요소인 attention mechanism이 현대의 연상 기억 모델들과 유사하다는 점입니다. 이러한 연결고리를 통해, 연구자들은 attention head 간에 정보가 직접 전달될 수 있도록 하는 새로운 잔차 스트림 아키텍처를 도입합니다.

- **Technical Details**: 이 논문에서 제안하는 연상 기억 모델은 ICL을 기반으로 한 분류 작업을 수행할 수 있습니다. 이 모델은 입력 데이터에 맞춰 attention 값이 직접 입력을 반영하도록 하여, 두 층 구조의 Transformer에서 ICL을 더욱 빠르게 구현할 수 있게 합니다. 이와 함께 작은 언어 모델에서도 성능 향상을 확인하며, 주어진 작업에서 attention head의 값을 중점적으로 분석합니다.

- **Performance Highlights**: 제안된 새로운 아키텍처는 ICL 능력이 훈련 과정에서 전통적인 방법보다 빠르게 나타남을 보여줍니다. 연구진은 800만 개의 매개변수를 가진 작은 언어 모델에서 개선된 ICL 성능을 관찰하였으며, 이는 고차원적이고 자연스러운 데이터에도 적용 가능한 결과임을 의미합니다. 이러한 접근 방식은 LLM의 해석 가능성에도 기여할 것으로 기대됩니다.



### A Cross-Domain Study of the Use of Persuasion Techniques in Online Disinformation (https://arxiv.org/abs/2412.15098)
- **What's New**: 이 연구는 다양한 도메인에서 사용되는 설득 기법의 역할을 대규모로 분석하며, 16가지 설득 기법을 조명합니다. 특히 기후 변화에 관한 허위 정보의 경우 언어적, 심리적, 문화적 요인이 어떻게 설득 전략에 영향을 미치는지를 통찰력 있게 다룹니다. 데이터셋은 다양한 원천에서 수집된 신문 기사를 포함하여 제안된 최신 설득 기법 분류기를 통해 분석되었습니다.

- **Technical Details**: 본 연구는 SemEval-2023에서 도입된 세계 최대의 설득 기법 주석 데이터셋을 활용합니다. 이 데이터셋은 9999가지 언어로 작성된 뉴스 기사에 대해 문장 수준에서 232개의 설득 기법으로 주석이 달려 있습니다. 연구에서는 Ma리와 함께 세 가지 도메인(코로나19, 기후 변화, 반이슬람)에 대한 데이터셋을 분석하며, 각 문장에서 설득 기법을 식별하기 위해 최첨단 설득 기법 분류기를 사용했습니다.

- **Performance Highlights**: 분석 결과 모든 허위 정보 도메인에서 'Loaded Language'와 'Doubt'가 널리 사용되며, 기후 변화 도메인에서 'Loaded Language'의 사용 비율이 감소했습니다. 특정 도메인에서의 설득 기법의 상대적 사용 빈도는 통계적으로 유의미한 차이를 보이며, 특히 이슬람 문제에서는 반복적 표현이 유의미하게 많이 나타났습니다. 이 연구의 결과는 허위 정보의 이해도를 높이고 그에 대응할 수 있는 중요한 통찰을 제공합니다.



### Till the Layers Collapse: Compressing a Deep Neural Network through the Lenses of Batch Normalization Layers (https://arxiv.org/abs/2412.15077)
Comments:
          Accepted at AAAI 2025

- **What's New**: 이번 논문에서는 딥 뉴럴 네트워크(DNN)에서 배치 정규화(batch normalization) 층을 활용하여 네트워크의 깊이를 줄이고 성능 저하 없이 계산 요구량과 지연 시간을 감소시키는 방법인 TLC(Till the Layers Collapse)를 제안합니다. 이 방법은 DNN의 과도한 매개변수를 관리하고 에너지 소비를 줄이는 데 중점을 두고 있습니다. 대규모 모델에서 특정 층을 제거함으로써 전체적인 계산 효율성을 개선하는 데 기여합니다.

- **Technical Details**: TLC는 배치 정규화 층의 매개변수를 이용하여 중요하지 않은 층을 식별하고 제거하는 방법입니다. 이를 통해 네트워크의 깊이를 줄이고 계산 요구량을 감소시킬 수 있습니다. 이 과정에서 표준화된 신호가 주로 긍정적일 때 선형 활성화가 최소한의 오류를 초래할 수 있음을 활용하여, 모델의 출력을 크게 변경하지 않으면서도 적절한 층 삭제가 가능합니다.

- **Performance Highlights**: 제안된 TLC 방법은 Swin-T, MobileNet-V2, RoBERTa와 같은 인기 있는 모델을 대상으로 이미지 분류 및 자연어 처리(NLP) 태스크에서 검증되었습니다. 이 방법은 층을 줄이는 동시에 정확도를 유지하며, 효율성을 향상시키는 성과를 보였습니다. 결과적으로, TLC는 성능 저하를 우려하지 않고도 DNN의 깊이를 효과적으로 감소시키는 가능성을 보여줍니다.



### Large Language Models and Code Security: A Systematic Literature Review (https://arxiv.org/abs/2412.15004)
- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)의 코드 생성과 관련된 보안 위험과 이점에 대해 체계적인 문헌 검토(Systematic Literature Review, SLR)를 제공합니다. 특히, LLMs가 생성한 코드에 의해 도입될 수 있는 다양한 취약점 유형을 분석하고, 코드 취약점을 탐지하고 수정하는 LLMs의 능력, 그리고 이 과정에서의 프롬프트 전략의 영향을 조사합니다. 또한, LLMs에 대한 데이터 중독 공격(data poisoning attack)이 이러한 작업 수행 능력에 미치는 영향을 심도 있게 분석합니다.

- **Technical Details**: 해당 연구는 대규모 언어 모델들이 코드 생성 및 보안 관련 작업 수행 시 발생할 수 있는 취약점 유형과 이 모델들의 탐지 능력에 대해 연구 질문들을 설정했습니다. 구체적으로 RQ1에서는 LLM이 생성하는 코드에서 도입될 수 있는 보안 취약점을 조사하고, RQ2에서는 인간이 작성한 코드와 LLM이 생성한 코드의 취약점 탐지 및 수정 능력을 분석합니다. 또한, RQ3에서는 데이터 세트의 중독이 LLM의 보안 코드 생성 및 취약점 탐지 및 수정 능력에 미치는 영향을 살펴봅니다.

- **Performance Highlights**: 이 연구에서 언급된 통계에 따르면 GitHub Copilot은 현재 코드의 약 46%를 생성하고 있으며, 개발자의 코딩 속도를 최대 55% 향상시킵니다. 그러나 LLMs는 특정 보안 관행에 대한 인식이 부족하며, 잠재적으로 비공식적으로 수집된 데이터 세트에 의존함으로써 보안 취약점을 생성할 위험을 안고 있습니다. 이러한 취약점과 한계에도 불구하고 LLMs는 여전히 개발자들에게 매우 유용한 도구로 자리잡고 있습니다.



### Movie2Story: A framework for understanding videos and telling stories in the form of novel tex (https://arxiv.org/abs/2412.14965)
- **What's New**: M2S라는 프레임워크는 비디오와 오디오, 문자 인식을 결합하여 소설 수준의 텍스트를 생성하는 새로운 방법론을 소개합니다. 이 모델은 비디오의 긴 형식 텍스트 설명 및 이해, 오디오 감정 분석, 그리고 시각적 캐릭터 인식 모듈을 포함하여 다중 모달 정보를 통합합니다. M2S는 대규모 언어 모델인 GPT-4로 다중 모달 텍스트 생성을 차별화하며, 향후 연구에 대한 잠재력을 지니고 있습니다.

- **Technical Details**: M2S 모델은 여러 모듈로 구성되어 있으며, 비디오 및 오디오의 다중 모달 정보를 결합하여 소설 형식의 상세한 서사를 생성합니다. 커다란 음성 및 감정 기반의 분석을 사용하여, 비디오와 오디오의 내용 모두를학습할 수 있습니다. 또한 이 모델은 고객의 다양성과 요구를 충족하기 위해 매끄러운 스토리라인 및 심리적 묘사를 제공합니다.

- **Performance Highlights**: 실험 결과 M2S는 텍스트의 완전성 및 풍부한 의미 정보를 잘 생성하는 뛰어난 능력을 보여줍니다. 이 모델은 장애인 및 텍스트 서술에 의존하는 사용자에게 영화 내용을 보다 쉽게 이해하고 즐길 수 있는 기회를 제공합니다. 이러한 접근은 교육 및 오락 분야에서도 활용 가능성이 높아, 비디오 컨텐츠에 대한 자세한 설명 제공 및 영화의 소설화 작업에 기여할 수 있을 것입니다.



### Unveiling Uncertainty: A Deep Dive into Calibration and Performance of Multimodal Large Language Models (https://arxiv.org/abs/2412.14660)
Comments:
          Accepted to COLING 2025

- **What's New**: 이 논문에서는 다중 모달 대형 언어 모델(MLLMs)의 불확실성 보정(calibration) 문제를 중점적으로 다루고 있습니다. 모델의 불확실성을 정량화하고 보정하는 과정이 중요하며, 이를 통해 의료 및 자율 주행과 같은 안전-critical 분야에서의 신뢰성을 개선할 수 있습니다. MLLMs는 이미지와 텍스트의 통합 처리를 통해 높은 성능을 보이지만, 여전히 고질적인 성능 불균형 문제가 존재합니다.

- **Technical Details**: MLLMs의 성공적인 활용을 위해, 저자들은 IDK 데이터셋을 구축하여 모델의 불확실성 자기 평가(self-assessment)를 분석했습니다. 두 가지 모달 정보 간의 불확실성 차이를 평가하고, 텍스트에 비해 이미지 정보의 불확실성이 낮은 경향을 관찰했습니다. 추가로, 고급 보정 기법인 temperature scaling과 iterative prompt optimization을 사용하여 MLLMs의 신뢰도와 예측 성능을 향상시킬 수 있는 방법을 제안했습니다.

- **Performance Highlights**: 연구 결과, MLLMs는 고백하기보다는 대답하는 경향이 있으며 이는 적절한 프롬프트 조정으로 개선될 수 있음을 보여주었습니다. 다양한 설정에서 MLLMs의 보정 성능 차이를 분석하였고, 특히 Fine-tuning 전후의 큰 차이를 발견하지 못했습니다. 저자들은 MLLMs의 교육 및 테스트에 있어 더 많은 개선이 필요하다고 결론지으며, 다양한 응용 분야에 대한 책임감 있는 활용 방안을 고안했습니다.



### LDP: Generalizing to Multilingual Visual Information Extraction by Language Decoupled Pretraining (https://arxiv.org/abs/2412.14596)
Comments:
          Accepted by AAAI2025

- **What's New**: 이 논문은 시각 정보 추출(Visual Information Extraction, VIE) 분야에서 다국어 모델의 효율성을 높이기 위해, 언어 편향을 분리하여 전이를 촉진하는 간단하지만 효과적인 다국어 훈련 패러다임인 LDP (Language Decoupled Pre-training)를 제안합니다. 모델 LDM (Language Decoupled Model)은 먼저 언어 독립적인 데이터로 사전 훈련된 후, 다운스트림 언어에 대해 미세 조정됩니다. 이로 인해 다국어 전이 성능이 향상되며, 영어만으로 이루어진 사전 훈련 데이터의 활용도를 극대화합니다.

- **Technical Details**: 제안된 LDM 모델은 SAM (Segment Anything Model) 프레임워크를 기반으로 하며, 시각적 문서의 정보를 효과적으로 추출하기 위해 디자인되었습니다. 모델은 텍스트 편집 확산 모델(TGD)을 사용하여 이미지에서 언어 편향을 분리하고, 다양한 이미지 내의 정보를 통합하는 MTIM (Multi-Token Information Merging) 모듈을 포함합니다. 이러한 구조는 모델의 성능을 한층 강화하여, 복잡한 시나리오에서의 정보 추출 가능성을 높입니다.

- **Performance Highlights**: 실험 결과 LDM은 XFUND 및 SIBR과 같은 다국어 벤치마크에서 경쟁 모델들보다 뛰어난 성과를 기록하였으며, 영어 데이터셋에서도 경쟁력 있는 결과를 유지합니다. 우리의 연구는 VIE 작업에서 비주얼 인바리언스(visual invariance)를 체계적으로 연구한 최초의 시도로, 훈련 데이터에서 언어 편향을 분리함으로써 다국어 일반화 성능이 향상된다는 것을 보여줍니다.



### Sliding Windows Are Not the End: Exploring Full Ranking with Long-Context Large Language Models (https://arxiv.org/abs/2412.14574)
Comments:
          14 pages

- **What's New**: 이 논문에서는 long-context LLMs를 사용한 passage ranking의 새로운 접근 방식을 제안합니다. 기존 sliding window 전략의 단점을 극복하기 위해, 사용자가 제공한 모든 passage를 한 번의 추론으로 처리할 수 있는 기법을 도입했습니다. 이로 인해 중복된 API 비용을 줄이면서도 전체 ranking을 효율적으로 수행할 수 있습니다. 연구 결과, supervised fine-tuning 설정에서 이 방법이 월등한 성능을 보인다는 것을 입증했습니다.

- **Technical Details**: 논문에서는 passage ranking을 위한 두 가지 전략인 full ranking과 sliding window 전략을 비교합니다. full ranking은 지정된 모든 passage를 한 번에 입력하여 동시에 순위를 매기는 방식이며, sliding window는 정해진 크기와 보폭으로 passage를 순차적으로 평가합니다. 중요한 것은, 기존의 training 메소드를 활용한 full ranking 모델의 fine-tuning에서 발생하는 두 가지 한계, 즉 완전한 ranking list 생성을 위한 한계와 language modeling loss의 비효율성을 다루기 위해 새로운 multi-pass sliding window 접근법과 importance-aware loss를 제안합니다.

- **Performance Highlights**: 실험 결과, TREC 및 BEIR 벤치마크에서 제안한 방법이 기존의 최첨단 모델들보다 뛰어난 성능을 보였습니다. 예를 들어, NDCG@10 기준으로 sliding window 모델에 비해 2.2의 절대적 향상을 나타내며, TREC DL19 데이터셋에서 대기 시간을 29.3% 줄였습니다. 이러한 성과는 long-context LLMs와 새로운 training 접근법의 효과를 잘 보여줍니다.



### Cal-DPO: Calibrated Direct Preference Optimization for Language Model Alignmen (https://arxiv.org/abs/2412.14516)
Comments:
          Accepted by NeurIPS 2024 Main

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 인간 선호 데이터와 정렬하는 문제를 다룹니다. 기존의 대조적 선호 최적화(contrastive preference optimization)가 제공하는 성과에도 불구하고, 그 방법은 응답 간의 실제 값에 대한 고려가 부족하여 인간 선호와의 정렬에서 최적의 성과를 내지 못했습니다. 이를 해결하기 위해, 우리는 calibrated direct preference optimization (Cal-DPO)라는 간단하면서 효과적인 알고리즘을 제안합니다.

- **Technical Details**: Cal-DPO는 학습된 내재적 보상을 실제 보상과 비교해 조정하여, 대조적 선호 목표를 최적화하면서 이러한 보상들이 실제 값과 비교 가능하도록 합니다. 이 방법을 통해 알고리즘의 성능을 크게 향상시킬 수 있으며, 이는 기존 접근 방식에 비해 여러 이론적 장점을 가지고 있습니다. 실험 결과는 Cal-DPO가 여러 기준 작업에서 기존의 정렬 방법들보다 우수한 성과를 보여주는 것을 입증합니다.

- **Performance Highlights**: Cal-DPO는 다양한 기준 작업에서 운영 가능한 방법들을 크게 개선하며, 데이터에 대해 더 나은 응답을 생성할 수 있도록 합니다. 본 연구에서 제안하는 방법은 기존 선호 최적화 방법들과 비교했을 때 항상 우수한 성능을 보여줍니다. 특히, Cal-DPO는 지능형 시스템이 인간의 선호에 보다 직접적으로 부합하도록 도움을 줍니다.



### GraphEQA: Using 3D Semantic Scene Graphs for Real-time Embodied Question Answering (https://arxiv.org/abs/2412.14480)
Comments:
          Project website: this https URL

- **What's New**: 이번 연구는 Embodied Question Answering (EQA) 분야에서의 새로운 접근법인 GraphEQA를 제안합니다. GraphEQA는 실시간 3D metric-semantic scene graphs (3DSGs)와 작업 관련 이미지를 활용하여 Vision-Language Models (VLMs)를 기반으로 EQA 작업을 수행하는 데 필요한 멀티모달 메모리를 제공합니다. 이 방법은 로봇이 이전의 환경 지식을 사용하여 보다 효율적으로 탐색하고 계획할 수 있도록 돕습니다.

- **Technical Details**: GraphEQA는 3DSGs의 계층적 구조를 활용하여 체계적인 계획과 의미 기반 탐색을 지원합니다. 이 구조는 가시적 환경의 의미 표현을 온라인으로 업데이트할 수 있는 강력한 도구로 작용합니다. 연구에서는 HM-EQA 데이터셋을 사용한 시뮬레이션 실험과 실제 가정 및 사무실 환경에서의 실험을 통해 그 효능을 입증하였습니다.

- **Performance Highlights**: 실험 결과, GraphEQA는 기존 기준선 모델들보다 EQA 작업을 수행하는 데 있어 더 높은 성공률과 적은 계획 단계를 기록하며 우수한 성과를 보였습니다. 따라서 이 방법은 로봇이 새로운 환경에서도 효과적으로 질문에 답할 수 있는 가능성을 높입니다.



### MegaPairs: Massive Data Synthesis For Universal Multimodal Retrieva (https://arxiv.org/abs/2412.14475)
- **What's New**: 이 논문에서는 MegaPairs라는 새로운 데이터 합성 방법을 소개하고, 이를 사용해 생성된 대규모 지침 데이터셋을 발표합니다. MegaPairs는 비전 언어 모델(VLMs)과 개방형 이미지들을 활용하여 훈련 데이터를 생성하며, 전통적인 방법의 한계를 극복하고, 더 나은 데이터 품질과 다양성을 제공합니다. 이 접근법은 고품질 데이터를 생성하고, 제공된 기존 데이터셋보다 70배 더 적은 데이터로도 성능을 향상시킵니다.

- **Technical Details**: MegaPairs는 이종 KNN 삼중 트리플렛을 구성하는 독특한 방식으로 설계되었습니다. 이 방법은 CLIP 비전 인코더와 DINO 비전 인코더, 그리고 CLIP 텍스트 인코더를 사용하여 상관 관계가 있는 이미지 쌍을 샘플링합니다. 생성된 이미지 쌍은 비전 언어 모델(VLM)과 대형 언어 모델(LLM) 주석 작성자에 의해 설명되고, 이를 바탕으로 다량의 가상 검색 지침이 생성됩니다.

- **Performance Highlights**: MegaPairs를 통해 생성된 데이터는 기존 데이터셋보다 우수한 품질을 달성하였으며, 26백만개의 데이터 인스턴스를 생산하였습니다. 실험 결과, MegaPairs로부터 추출한 50만개 인스턴스만으로도 MagicLens의 3,670만 인스턴스보다 나은 성능을 보여주었습니다. 새로 훈련된 MMRet 모델은 4개의 인기 있는 이미지 검색 기준에서 최첨단 성능을 달성하였고, 추가적인 세부 조정을 통해 성능이 더욱 향상되었습니다.



### Are Longer Prompts Always Better? Prompt Selection in Large Language Models for Recommendation Systems (https://arxiv.org/abs/2412.14454)
Comments:
          15 pages

- **What's New**: 이번 논문에서는 대규모 언어 모델(LLM)을 기반으로 한 추천 시스템(LLM-RS)의 프롬프트 선택 지침을 제시하였습니다. 프롬프트는 추천 작업을 자연어로 변환하여 사용자 선호도를 예측하는 데 기여합니다. 90개의 다양한 프롬프트와 5개의 실세계 데이터 세트를 사용하여 추천 정확도와 데이터셋 특성 간의 관계를 조사하였고, 프롬프트의 유형에 따라 변동성이 크다는 사실을 확인하였습니다.

- **Technical Details**: 저자들은 450회의 실험을 통해 특정 컴포넌트가 포함된 프롬프트의 일관된 성능이 없음을 발견하였습니다. 하지만, 카테고리나 설명을 추가함으로써 정확도를 높일 수 있는 가능성이 있음을 밝혔습니다. 실험 결과, 미비한 검증 데이터로도 높은 정확도를 보이며 적합한 프롬프트를 선택할 수 있는 방법을 제안하였습니다.

- **Performance Highlights**: 제안된 방법은 이전 연구들에 비해 세 개의 데이터셋에서 가장 높은 정확도를 기록했으며, 나머지 데이터셋에서도 두 번째 높은 정확도를 달성했습니다. 또한, 고성능과 비용 효율적인 LLM을 적절히 활용하여 탐색 비용을 크게 줄이면서도 높은 예측 정확도를 유지하는 전략을 소개하였습니다.



### In-Group Love, Out-Group Hate: A Framework to Measure Affective Polarization via Contentious Online Discussions (https://arxiv.org/abs/2412.14414)
- **What's New**: 이번 논문에서는 affective polarization(정서적 양극화) 개념을 정량적으로 탐구하기 위한 새로운 방법론을 소개합니다. 저자들은 소셜 미디어 데이터를 활용하여 내부 그룹 사랑(in-group love)과 외부 그룹 증오(out-group hate)의 중요한 매개 변수를 통계적으로 추정할 수 있는 이산 선택 모델(discrete choice model)을 제안합니다. 이러한 접근은 COVID-19 팬데믹과 같은 실질적인 사안에서 파트너 지지의 차별화된 태도를 신속하게 분석하고 이해하는데 중요한 기여를 합니다.

- **Technical Details**: 이 논문에서는 affectively polarized social networks 내 개인의 의사 결정 과정을 설명하는 이산 선택 모델을 소개합니다. 이 모델은 정서적 양극화의 두 가지 핵심 요소인 내부 그룹 사랑(α𝛼)과 외부 그룹 증오(β𝛽)를 파라미터로 포함하고 있으며, 다당제(multi-party) 맥락에서도 적용할 수 있습니다. 또한, 저자들은 소셜 네트워크 내에서 개인의 입장과 이웃의 입장을 분석하여 이 두 요소를 추정하는 방법론을 개발하였습니다.

- **Performance Highlights**: COVID-19 팬데믹 동안 마스크 착용과 봉쇄에 대한 소셜 미디어 데이터 분석을 통해 이 모델의 실증적 검증을 진행하였습니다. 결과적으로, 당파적 태도가 빠르게 팽창하였으며, 의도적으로 파트너의 입장이 이질적인 사람들에 대한 증오를 높이는 경향이 있음을 보였습니다. 이 연구는 정서적 양극화를 신속하게 파악하고, 사회적 분열을 완화하는 전략을 수립하는 데 유용한 통찰을 제공합니다.



### ResQ: Mixed-Precision Quantization of Large Language Models with Low-Rank Residuals (https://arxiv.org/abs/2412.14363)
Comments:
          14 pages, 6 figures, 6 tables

- **What's New**: 본 연구에서는 ResQ라는 새로운 양자화 방법을 제안합니다. 이는 고관련성을 가진 저차원 부분 공간에서 활성 화의 변동성을 최대화하고, 이 부분의 계수를 높은 정밀도(8-bit)로 유지하며 나머지는 낮은 정밀도(4-bit)로 양자화하는 방식을 적용합니다. 이는 다른 최신 양자화 방법들보다 우수한 성능을 보여 주며, 특히 Llama 모델에서의 낮은 perplexity를 기록하였습니다.

- **Technical Details**: ResQ 방법은 주성분 분석(principal component analysis, PCA)을 활용하여 높은 변동성을 가진 저차원 부분 공간을 찾아냅니다. 이 과정을 통해 활성화 값들을 고정밀로 유지하고 불규칙한 값을 억제하기 위해 무작위 회전을 적용합니다. 실험 결과, ResQ는 4-bit 양자화로 적용할 경우, SpinQuant와 비교해 더욱 개선된 결과를 보여주며, 이는 최고의 경쟁 방법보다 2-33% 낮은 perplexity를 달성하였습니다.

- **Performance Highlights**: ResQ는 다양한 기준에서 최근 표준적인 및 혼합 정밀도 양자화 방법들을 초월하는 성능을 보여줍니다. 특히 16-bit 기준과 비교하여 2.4배의 속도 향상과 함께 33%의 낮은 perplexity를 기록했습니다. 이러한 성능은 ResQ가 경량화된 모델에서 보내는 메모리를 줄이고 계산 속도를 높일 수 있도록 해 주며, NVIDIA A100 GPU에서 실시간 속도 향상을 달성하였습니다.



### Towards AI-$45^{\circ}$ Law: A Roadmap to Trustworthy AGI (https://arxiv.org/abs/2412.14186)
Comments:
          First submit, Preview Only

- **What's New**: 이 논문에서는 신뢰할 수 있는 인공지능 일반 지능(AGI)을 위한 가이드라인을 제안하며, 'AI-45도 법칙'이라는 원리를 소개합니다. 이는 인공지능의 능력과 안전이 동등하게 발전해야 한다는 것을 강조합니다. 또한 'Causal Ladder of Trustworthy AGI' 프레임워크를 통해 현재의 인공지능 안전성과 능력 연구를 체계적으로 정리할 수 있는 구조를 제공합니다.

- **Technical Details**: 이 연구에서는 AGI의 핵심 신뢰성 요구 사항을 충족하는 세 가지 레이어로 구성된 기술 프레임워크를 제안합니다: Approximate Alignment Layer, Intervenable Layer, Reflectable Layer. 첫 번째 레이어는 인간의 가치와의 근사적 정렬에 초점을 맞추며, 인공지능 모델의 가치 표현 및 정렬 능력의 기술적 발전이 필요합니다. 이러한 진전을 통해 모델이 인간의 지시를 이해하고 따를 수 있도록 하는 것이 중요합니다.

- **Performance Highlights**: 이 논문은 신뢰할 수 있는 AGI의 다섯 가지 신뢰성 수준—지각, 추론, 의사결정, 자율성, 협력 신뢰성—을 정의합니다. 또한, AGI 개발에서 윤리적이고 안전한 접근 방식을 보장하기 위해 필요한 거버넌스 조치를 제안합니다. 이러한 접근 방식은 AGI 시스템의 안전성과 신뢰성을 높이는 데 기여할 것으로 기대됩니다.



New uploads on arXiv(cs.IR)

### Nano-ESG: Extracting Corporate Sustainability Information from News Articles (https://arxiv.org/abs/2412.15093)
Comments:
          To be published at ECIR 2025. Preprint

- **What's New**: 최근 기업의 지속 가능성에 대한 평가가 중요한 주제로 떠오르면서, 기존의 ESG 스코어에 대한 신뢰성 문제가 제기되고 있습니다. 본 논문은 2023년 1월부터 2024년 9월까지의 주요 독일 기업에 대한 840,000개 이상의 뉴스 기사를 수집하여, 기업의 ESG(환경, 사회, 지배구조) 관행에 대한 보다 독립적이고 투명한 분석 방법을 제안합니다. 이를 통해 대중의 인식과 기업 행동을 연결하는 새로운 데이터셋인 Nano-ESG를 만들었습니다.

- **Technical Details**: Nano-ESG 데이터셋은 최신 자연어 처리(Natural Language Processing, NLP) 기법과 대형 언어 모델(Large Language Models, LLM)을 활용하여 뉴스 기사를 필터링하고 요약하며 ESG 관련 감정 및 측면을 추출합니다. 데이터 처리 파이프라인을 통해, 2023년과 2024년의 데이터 처리 방법이 다소 차별화되지만, 주요 목표는 고비용의 LLM을 효율적으로 적용하는 것입니다. 이 과정에서 독일 기업과 관련된 기사에서 적절한 키워드를 검색하여 기사를 수집합니다.

- **Performance Highlights**: 본 연구에서는 LLM이 생성한 데이터와 답변이 정확하다는 것을 평가를 통해 확인했습니다. 최종적으로 공개된 Nano-ESG 데이터셋은 기업의 ESG 관련 정보에 대한 타임 시리즈를 제공하며, 이를 통해 ESG적 사건과 외부 요인 간의 상호작용을 분석할 수 있게 됩니다. 이 데이터셋은 사용자들이 ESG 관련 주제를 식별하고 의사 결정을 지원하는 데 중요한 자료로 활용될 수 있습니다.



### DisCo: Graph-Based Disentangled Contrastive Learning for Cold-Start Cross-Domain Recommendation (https://arxiv.org/abs/2412.15005)
- **What's New**: 이 논문에서는 사용자 콜드 시작 문제를 해결하기 위한 새로운 방법인 DisCo(Graph-based Disentangled Contrastive learning framework)를 제안합니다. 기존의 크로스 도메인 추천 시스템의 한계인 소스 도메인에서의 사용자 선호와 타겟 도메인에서의 선호 간의 불일치를 해결하기 위해 고안되었습니다. DisCo는 사용자 의도를 더 세밀하게 포착하고, 관련 없는 협력 정보를 필터링하여 부정적인 전이를 피합니다.

- **Technical Details**: DisCo는 다중 채널 그래프 인코더(multi-channel graph encoder)를 사용하여 각 도메인에서 다양한 사용자 의도를 캡처합니다. 그런 다음, 두 도메인의 임베딩 공간에서 친밀도 그래프(affinity graph)를 구성하고 다단계 랜덤 워크(multi-step random walks)를 수행하여 고차원 사용자 유사성 관계를 파악합니다. 이 프레임워크는 의도 기반의 대조 학습(intent-wise contrastive learning) 접근 방식을 사용하여 타겟 도메인에서 사용자 유사성 정보를 보존합니다.

- **Performance Highlights**: 네 개의 벤치마크 CDR 데이터세트에서 진행된 실험 결과, DisCo는 기존의 최첨단 기준선을 일관되게 초월했습니다. 이는 DisCo의 효과성과 구성 요소들이 실제로 성능 향상에 기여함을 입증합니다. 이러한 결과는 DisCo가 콜드 시작 CDR 작업에서 임베딩 전이에만 집중하지 않고, 더 복잡한 사용자 선호의 형성을 다룰 수 있는 가능성을 보여 줍니다.



### Spectrum-based Modality Representation Fusion Graph Convolutional Network for Multimodal Recommendation (https://arxiv.org/abs/2412.14978)
Comments:
          Accepted to ACM Web Search and Data Mining (WSDM) 2025

- **What's New**: 이번 연구에서는 Spectrum-based Modality Representation (SMORE)이라는 새로운 융합 그래프 추천 시스템을 제안하였습니다. SMORE는 다중 모드 특성을 주파수 영역으로 투영하여 모드 간 노이즈를 억제하면서 사용자 및 항목 간의 선호도를 효과적으로 포착하는 데 중점을 두고 있습니다. 이를 통해 기존의 추천 시스템이 직면한 모드 간 노이즈 문제를 해결하려 합니다.

- **Technical Details**: SMORE는 세 가지 주요 구성 요소로 이루어져 있습니다: 1. Spectrum Modality Fusion, 2. Multi-modal Graph Learning, 3. Modality-Aware Preference Module. 주파수 영역에 모드 특성을 투사하여 효율적인 포인트 와이즈 집합을 통해 모드 간의 보편적인 패턴을 효과적으로 캡처하며, 동적 필터를 통해 관련 없는 신호를 적응적으로 억제합니다.

- **Performance Highlights**: SMORE의 효능은 세 개의 실제 데이터 세트에서의 실험을 통해 검증되었습니다. 연구 결과는 제안된 모델이 기존 모델들보다 사용자 모드 특성과 융합 선호도를 보다 정확하게 추론할 수 있음을 보여주었습니다. 이를 통해 추천 시스템의 전반적인 성능 향상에 기여할 수 있는 가능성을 나타냅니다.



### ECLIPSE: Contrastive Dimension Importance Estimation with Pseudo-Irrelevance Feedback for Dense Retrieva (https://arxiv.org/abs/2412.14967)
- **What's New**: 최근 정보 검색(Information Retrieval) 분야의 발전으로 인해 고차원 임베딩 공간을 활용한 자료 검색의 정확성이 향상되었습니다. 본 논문에서는 ECLIPSE라는 새로운 방법론을 제안하여 관련 문서 및 비관련 문서의 정보를 통합하여 검색 성능을 개선하고자 합니다. 이 방법은 비관련 문서로부터 중심을 계산하여 노이즈가 있는 차원을 추정함으로써 더욱 정교한 검색을 돕습니다.

- **Technical Details**: ECLIPSE는 비관련 문서에서의 임베딩을 이용해 일반적인 의사 비관련 문서(pseudo-irrelevant document)의 표현을 구성하고, 이를 관련 문서 표현에서 빼냄으로써 비관련 차원을 억제합니다. 이 방법은 기존의 Dimension Importance Estimator(DIME)에 적용 가능하며, 검색 모델의 기본 구조를 변경하지 않고도 성능을 향상시킵니다. ECLIPSE는 주어진 쿼리와 documents의 관계를 최적화하기 위해 고차원 공간의 차원을 선택적으로 축소합니다.

- **Performance Highlights**: ECLIPSE의 성능을 검증한 결과, DIME 기반 벤치마크와 비교하여 평균적으로 최대 19.50%(또는 22.35%)의 mAP(AP) 개선과 11.42%(또는 13.10%)의 nDCG@10 개선을 보여주었습니다. 이는 비관련 문서의 활용이 검색 성능을 보다 강력하게 만드는 데 기여할 수 있음을 시사합니다. 이러한 결과는 ECLIPSE가 정보 검색 시스템의 유용한 개선 도구로 자리매김할 수 있도록 해줍니다.



### Sliding Windows Are Not the End: Exploring Full Ranking with Long-Context Large Language Models (https://arxiv.org/abs/2412.14574)
Comments:
          14 pages

- **What's New**: 이 논문에서는 long-context LLMs를 사용한 passage ranking의 새로운 접근 방식을 제안합니다. 기존 sliding window 전략의 단점을 극복하기 위해, 사용자가 제공한 모든 passage를 한 번의 추론으로 처리할 수 있는 기법을 도입했습니다. 이로 인해 중복된 API 비용을 줄이면서도 전체 ranking을 효율적으로 수행할 수 있습니다. 연구 결과, supervised fine-tuning 설정에서 이 방법이 월등한 성능을 보인다는 것을 입증했습니다.

- **Technical Details**: 논문에서는 passage ranking을 위한 두 가지 전략인 full ranking과 sliding window 전략을 비교합니다. full ranking은 지정된 모든 passage를 한 번에 입력하여 동시에 순위를 매기는 방식이며, sliding window는 정해진 크기와 보폭으로 passage를 순차적으로 평가합니다. 중요한 것은, 기존의 training 메소드를 활용한 full ranking 모델의 fine-tuning에서 발생하는 두 가지 한계, 즉 완전한 ranking list 생성을 위한 한계와 language modeling loss의 비효율성을 다루기 위해 새로운 multi-pass sliding window 접근법과 importance-aware loss를 제안합니다.

- **Performance Highlights**: 실험 결과, TREC 및 BEIR 벤치마크에서 제안한 방법이 기존의 최첨단 모델들보다 뛰어난 성능을 보였습니다. 예를 들어, NDCG@10 기준으로 sliding window 모델에 비해 2.2의 절대적 향상을 나타내며, TREC DL19 데이터셋에서 대기 시간을 29.3% 줄였습니다. 이러한 성과는 long-context LLMs와 새로운 training 접근법의 효과를 잘 보여줍니다.



### HEC-GCN: Hypergraph Enhanced Cascading Graph Convolution Network for Multi-Behavior Recommendation (https://arxiv.org/abs/2412.14476)
- **What's New**: 본 연구에서는 다중 행동 추천(Multi-behavior recommendation, MBR) 문제를 해결하기 위해 새로운 방법인 Hypergraph Enhanced Cascading Graph Convolution Network (HEC-GCN)를 제안합니다. 기존 방법들이 사용자와 항목 간의 상호작용 정보 모델링에 초점을 맞췄다면, HEC-GCN은 행동 특화 상호작용 그래프와 하이퍼그래프를 동시에 모델링하여 미세 및 거칠게 상관관계를 탐색합니다. 또한, 행동 간 일관성을 유지하기 위한 정렬 전략을 도입하여 추천 성능을 향상시키고자 합니다.

- **Technical Details**: HEC-GCN은 각 행동에 대해 사용자-항목 상호작용 그래프와 하이퍼그래프를 도입하여 미세 및 거친 상관관계를 모두 포착합니다. 이 방식은 상호작용 그래프 내의 희소성을 완화하는 데 기여하며, 내부 행동 간의 일관성을 유지하기 위해 대조 학습 기반의 모듈을 개발했습니다. 이 모듈은 행동별 일관성과 행동 간 일관성을 동시에 고려하여 사용자 및 항목 임베딩을 조정하는 과정으로 구성됩니다.

- **Performance Highlights**: 풍부한 실험 결과는 HEC-GCN이 기존 최첨단 방법들에 비해 우수한 성능을 발휘함을 보여줍니다. 특히, Beibei, Taobao 및 Tmall 데이터세트에서 각각 19.20%, 37.45% 및 13.43%의 상대적 성능 향상을 기록했습니다. 이러한 성과는 HEC-GCN이 다중 행동 추천 분야에서 희소성 문제를 효과적으로 완화하고, 행동 간 및 내부 행동 간의 일관성을 유지할 수 있는 가능성을 보여줍니다.



### VISA: Retrieval Augmented Generation with Visual Source Attribution (https://arxiv.org/abs/2412.14457)
- **What's New**: 본 논문에서는 Retrieval-Augmented Generation with Visual Source Attribution (VISA)라는 새로운 접근방식을 제안합니다. VISA는 답변 생성과 시각적 출처 부여(visual source attribution)를 결합하여 사용자가 생성된 답변을 쉽게 검증할 수 있도록 지원합니다. 이 방법은 대형 비전-언어 모델(VLM)을 활용하여 문서 스크린샷의 특정 영역을 강조 표시하며, 이로써 사용자가 원본 문서의 맥락 내에서 근거를 명확하게 확인할 수 있습니다.

- **Technical Details**: VISA는 단일 또는 다수의 문서 이미지를 처리하여 사용자 쿼리에 대한 답변을 생성하고, 관련 증거 문서 내의 해당 영역의 경계 상자를 반환합니다. 기존의 텍스트 기반의 출처 부여 방식을 넘어서는 새로운 방법적인 접근을 제공하며, 이는 다큐먼트 스타일에 구애받지 않기 때문에 다양한 사용 사례에 적합합니다. 또한, 이를 평가하기 위해 Wiki-VISA와 Paper-VISA라는 두 가지 데이터셋을 구성하였으며, 특히 의학 분야의 다중 모드 과학 논문을 평가하는 데 초점을 맞추고 있습니다.

- **Performance Highlights**: 실험 결과, 기존의 최고 성능 모델인 QWen2-VL-72B가 제로샷 프롬프트에서 정밀한 시각적 출처 부여에 어려움을 겪는 것으로 나타났습니다. 하지만, VISA를 주어진 데이터셋에서 세부 조정(fine-tuning)함으로써 시각적 출처 부여 정확도가 크게 향상되었습니다. 결과 분석을 통해 긴 이미지 문서, 다중 문서 및 제로샷 일반화 능력 향상과 같은 개선이 필요한 주요 영역이 강조되었습니다.



### Are Longer Prompts Always Better? Prompt Selection in Large Language Models for Recommendation Systems (https://arxiv.org/abs/2412.14454)
Comments:
          15 pages

- **What's New**: 이번 논문에서는 대규모 언어 모델(LLM)을 기반으로 한 추천 시스템(LLM-RS)의 프롬프트 선택 지침을 제시하였습니다. 프롬프트는 추천 작업을 자연어로 변환하여 사용자 선호도를 예측하는 데 기여합니다. 90개의 다양한 프롬프트와 5개의 실세계 데이터 세트를 사용하여 추천 정확도와 데이터셋 특성 간의 관계를 조사하였고, 프롬프트의 유형에 따라 변동성이 크다는 사실을 확인하였습니다.

- **Technical Details**: 저자들은 450회의 실험을 통해 특정 컴포넌트가 포함된 프롬프트의 일관된 성능이 없음을 발견하였습니다. 하지만, 카테고리나 설명을 추가함으로써 정확도를 높일 수 있는 가능성이 있음을 밝혔습니다. 실험 결과, 미비한 검증 데이터로도 높은 정확도를 보이며 적합한 프롬프트를 선택할 수 있는 방법을 제안하였습니다.

- **Performance Highlights**: 제안된 방법은 이전 연구들에 비해 세 개의 데이터셋에서 가장 높은 정확도를 기록했으며, 나머지 데이터셋에서도 두 번째 높은 정확도를 달성했습니다. 또한, 고성능과 비용 효율적인 LLM을 적절히 활용하여 탐색 비용을 크게 줄이면서도 높은 예측 정확도를 유지하는 전략을 소개하였습니다.



### ChainRank-DPO: Chain Rank Direct Preference Optimization for LLM Rankers (https://arxiv.org/abs/2412.14405)
- **What's New**: 이번 논문에서는 Chain-of-Thought prompting과 SFT-DPO 파이프라인을 결합하여 텍스트 재정렬에 대한 새로운 접근 방식을 제안합니다. 기존의 순차적 최적화 방법이 갖는 일반적인 추론 능력의 저하 문제를 해결하고, 재정렬 성능을 개선하는 데 중점을 두고 있습니다. 실험 결과, TREC 2019 및 2020 데이터셋에서 RankZephyr를 초월하는 성능을 보였습니다.

- **Technical Details**: ChainRank라는 새로운 방법론은 LLaMA3-8b-instruct 모델을 기반으로 하여, 순차적인 문서 재정렬에서 가장 관련성 높은 문서를 반복적으로 선택하는 체인 오브 시퀀스(CoT) 추론 과정을 프레임으로 설정합니다. 이 모델은 SFT 데이터와 DPO 데이터를 활용하여 3 에포크 동안 완전한 파인튜닝이 진행되었으며, 학습 과정에서 문서 추출과 관련된 높은 데이터 품질을 유지하기 위해 형식이 올바르지 않은 생성물을 제거했습니다.

- **Performance Highlights**: 이번 연구에서 ChainRank는 여러 벤치마크에서 뛰어난 성능을 발휘하며, 기존의 성능 저하 문제를 해결하는 데 성공했습니다. 특히, TREC 2019 및 2020 Deep Learning Tracks와 BEIR 벤치마크에서 RankVicuna 및 RankZephyr보다 우수한 결과를 기록하며, 일반적인 텍스트 생성 및 추론 능력을 유지하는 데 있어 효과적임을 입증했습니다.



### Embedding Cultural Diversity in Prototype-based Recommender Systems (https://arxiv.org/abs/2412.14329)
- **What's New**: 이 논문에서는 인기 편향(popularity bias) 문제를 해결하기 위해 프로토타입 기반 행렬 분해 방법을 개선하는 새로운 접근 방식을 제안합니다. 문화 제품을 추천하는 시스템에서 이 편향이 문화적 대표성에 미치는 부정적인 영향을 우려하며, 인구 통계학적 요소를 통해 이를 완화하고자 합니다. 구체적으로, 원래 프로토타입을 필터링하고 프로토타입의 균일한 분포를 강화하는 정규화 기법을 도입하여, 보다 공정한 추천 결과를 도출하는 것에 초점을 맞췄습니다.

- **Technical Details**: 연구에서는 ProtoMF(Prototype-based Matrix Factorization) 모델의 두 가지 주요 혁신을 제시합니다. 첫 번째는 Prototype K-filtering으로, k개의 가장 가까운 프로토타입을 선택하여 사용자와 항목의 표현을 개선하는 방법입니다. 두 번째는 Prototype-Distributing Regularizer로, 프로토타입의 분포를 균등화하여 다양한 문화적 표현을 촉진하는 메커니즘입니다. 이러한 접근법을 통해 인구 통계적 편향을 해결하고, 추천 시스템의 성능을 유지하면서 편향을 감소시키려 했습니다.

- **Performance Highlights**: 모델을 MovieLens-1M, LastFM-2b, Amazon Reviews’23 데이터셋에서 평가한 결과, 긴 꼬리 항목의 평균 순위가 27% 감소하고, 저명하지 않은 국가의 항목 순위는 2% 감소하는 성과를 보여주었습니다. 또한, HitRatio@10에서 기존 최고 성능 모델 대비 2% 개선된 결과를 달성했습니다. 이로 인해 추천의 공정성을 높이면서도 추천 품질을 저하하지 않는 성과를 검증하였습니다.



### SAFERec: Self-Attention and Frequency Enriched Model for Next Basket Recommendation (https://arxiv.org/abs/2412.14302)
- **What's New**: 본 논문에서는 NBR(Next-Basket Recommendation) 작업을 위한 새로운 알고리즘인 SAFERec를 제안합니다. 이 알고리즘은 NIR(Next Item Recommendation)에서 우수한 성능을 보이는 transformer 기반 아키텍처를 활용하며, 아이템의 빈도 정보를 통합하여 NBR 작업에 적합성을 높입니다. SAFERec는 기존 알고리즘과 비교해 8% 향상된 Recall@10 성능을 보여줍니다.

- **Technical Details**: SAFERec는 사용자 구매 이력을 mini-batch로 처리하고, 각 바구니를 sparse multi-hot vector 형식으로 변환합니다. 이를 통해 각 사용자의 구매 이력이 sparse matrix 형태로 표현되며, 이는 최근 바구니를 고려하는 최대 길이를 나타내는 하이퍼파라미터 L을 도입하여 처리합니다. SAFERec는 transformer 레이어와 아이템 빈도 인식 모듈을 통합하여 NBR을 위한 효과적인 솔루션을 제공합니다.

- **Performance Highlights**: SAFERec의 성능은 여러 공공 데이터셋에서 테스트되었으며, 모든 다른 기준선 방법들보다도 우수한 성과를 보였습니다. 특히, SAFERec는 Recall@10에서 최대 8%의 개선을 이루어 시사하는 바가 큽니다. 이러한 결과는 SAFERec가 더 혁신적인 아이템 세트를 추천하는 데 효과적임을 보여줍니다.



### Progressive Multimodal Reasoning via Active Retrieva (https://arxiv.org/abs/2412.14835)
Comments:
          Working in progress

- **What's New**: 이번 연구에서는 다중 단계를 고려한 다중 모드(multimodal) 추론 과제에서 MLLM의 성능을 향상시키기 위한 새로운 프레임워크인 AR-MCTS를 제안합니다. 이 프레임워크는 Active Retrieval(AR)과 Monte Carlo Tree Search(MCTS)를 결합하여 복잡한 추론 문제를 해결하는 데 필요한 핵심 인사이트를 동적으로 검색할 수 있도록 설계되었습니다. 특히, 이 연구는 기존의 빔 탐색(beam search) 샘플링 방법을 대체하는 혁신적인 접근 방식을 도입하여 각 추론 단계에서 다양한 문제 해결 인사이트를 제공함으로써 신뢰성을 높이고자 합니다.

- **Technical Details**: AR-MCTS 프레임워크는 통합된 검색 모듈을 개발하여 하이브리드 모드 검색 데이터베이스로부터 복잡한 추론을 지원하기 위한 핵심 인사이트를 검색합니다. MCTS 알고리즘을 활용하여 단계별로 주어진 문제의 적절한 해답을 유도하는 과정 보상을 정의하고, 각 단계에서 이전 단계를 바탕으로 샘플링을 최적화합니다. 이러한 접근 방식은 추론의 신뢰성과 다양성을 향상시키며, 자동화된 다중 모드 추론 검증 과정을 가능하게 합니다.

- **Performance Highlights**: 실험 결과에 따르면, AR-MCTS는 세 가지 복잡한 다중 모드 추론 기준을 통해 다양한 모델에서 효과성을 입증했습니다. AR-MCTS는 샘플링의 다양성과 검증의 정확성을 최적화하여 신뢰성 있는 다중 모드 추론을 위한 promising한 솔루션을 제공합니다. 이 연구 결과는 다중 모드 추론의 고도화를 위한 가능성을 열어 주며, MLLM의 성능 향상에 기여할 것으로 기대됩니다.



### Efficient Self-Supervised Video Hashing with Selective State Spaces (https://arxiv.org/abs/2412.14518)
Comments:
          Accepted by AAAI'25. 9 pages, 5 figures, 2 tables

- **What's New**: 이 연구는 Self-supervised video hashing (SSVH)를 활용하여 비디오 색인화 및 검색을 개선하고자 합니다. 기존의 Transformer 기반 기법이 계산 및 메모리 비효율성 문제를 가지고 있는 반면, Mamba라는 최신 상태 공간 모델에서 영감을 받아 이루어진 이 연구는 효율성과 효능의 균형을 보다 잘 맞추고자 합니다. S5VH라는 새로운 비디오 해싱 모델을 제안하며, 자가 지도 학습(paradigm) 방식을 개선하여 비디오 데이터를 보다 효율적으로 처리할 수 있도록 합니다.

- **Technical Details**: S5VH는 Mamba 기반 비디오 해싱 모델로, bidirectional Mamba layers를 설계하여 인코더와 디코더 모두에서 효과적이고 효율적입니다. 이는 데이터 의존적 선택적 스캐닝 메커니즘을 통해 최적의 temporal relationship를 포착할 수 있도록 설계되었습니다. 자신의 공간(feature space)에서 global semantics를 semantically consistent하고 구별 가능한 해시 중심으로 변환하며, 이를 위한 center alignment loss를 도입하여 글로벌 학습 신호를 제공합니다.

- **Performance Highlights**: 실험 결과 S5VH는 최신 방법들에 비해 성능이 개선되었으며, 전이 가능성과 추론 효율성(scale) 측면에서도 우수한 이점을 보입니다. 또한, self-local-global (SLG) 패러다임을 통해 학습 효율성이 크게 향상되어 더 빠르고 더 나은 수렴(convergence)을 달성할 수 있었습니다. 전체적인 성능 향상은 자가 지도 학습의 최근 발전에 기인하며, 이는 대규모 라벨이 없는 비디오 데이터의 증가와 연결됩니다.



### Moving Beyond LDA: A Comparison of Unsupervised Topic Modelling Techniques for Qualitative Data Analysis of Online Communities (https://arxiv.org/abs/2412.14486)
- **What's New**: 본 연구에서는 BERTopic을 활용하여 소셜 미디어 데이터의 질적 분석을 지원하는 방안을 탐구합니다. 이는 프로그래밍 전문성이 부족한 질적 연구자들이 기존의 복잡한 topic modelling 기법을 보다 쉽게 사용할 수 있도록 돕기 위한 것입니다. 12명의 연구자와의 인터뷰를 통해 BERTopic이 제공하는 깊이 있는 클러스터링 능력이 높은 평가를 받았으며, 이는 보다 명확한 주제 이해와 실행 가능한 통찰력을 가능하게 했습니다.

- **Technical Details**: BERTopic은 고급 Large Language Model(LLM)에 기반한 topic modelling 기법으로, 기존의 Latent Dirichlet Allocation(LDA)나 Non-Negative Matrix Factorization(NMF)보다 월등한 성능을 발휘합니다. 이 기법은 문맥에 따라 적절한 사전 처리가 필요 없이도 세밀한 텍스트 분석을 가능하게 하며, GPU 자원을 활용하여 대규모 데이터셋을 효율적으로 처리할 수 있습니다. 연구팀은 BERTopic을 Computational Thematic Analysis(CTA) 툴킷에 통합하여 활용성을 극대화했습니다.

- **Performance Highlights**: 연구자들이 선호하는 topic modelling 기법으로 BERTopic이 가장 많은 지지를 얻었으며, 12명 중 8명이 가장 효과적인 도구로 선택했습니다. 연구자들은 BERTopic이 복잡한 데이터 속에서 중요한 관계를 드러내고, 의미 있는 분석 도구가 필요하다고 강조했습니다. 이로 인해 연구자들이 원하는 깊이 있고 체계적인 데이터 해석을 가능하게 해주는 도구가 절실하다는 결론을 도출했습니다.



### State Space Models are Strong Text Rerankers (https://arxiv.org/abs/2412.14354)
Comments:
          The first two authors contributed equally, order decided randomly

- **What's New**: 이 논문에서는 기존 트랜스포머 아키텍처에 대한 대안으로 주목받고 있는 상태 공간 모델(state space models, SSMs), 특히 Mamba 모델을 소개합니다. Mamba는 긴 컨텍스트를 이해하고 정밀한 쿼리-문서 상호작용을 다루는 텍스트 재순위 작업에서의 유용성을 평가하고자 합니다. 연구 결과, Mamba는 비슷한 크기의 트랜스포머 모델과 경쟁력 있는 성능을 보여주지만, 훈련 및 추론 효율성 면에서는 다소 부족한 것으로 나타났습니다.

- **Technical Details**: Mamba 모델은 효율적인 표현을 위해 선택적 상태를 인코딩하며, O(1) 시간 복잡성으로 인퍼런스를 수행합니다. 연구에서는 Mamba-1과 Mamba-2 모델을 비교하고, 다양한 아키텍처와 전처리 목표에 따라 성능을 평가합니다. 특히 Mamba 모델은 트랜스포머와 비교할 때 상대적으로 훈련 및 추론에서 더 비효율적임을 보여주며, Mamba-2가 성능과 효율성 모두에서 Mamba-1보다 우수한 결과를 보였습니다.

- **Performance Highlights**: Mamba 기반 언어 모델은 트랜스포머 기반 모델과 유사한 수준의 텍스트 재순위 성능을 달성했습니다. 하지만 이들은 이론적으로 더 나은 복잡성을 가진 반면, 실제로는 IO-aware 최적화(예: flash attention)를 사용하는 트랜스포머 아키텍처에 비해 효율성이 낮았습니다. 앞으로 정보 검색(Information Retrieval, IR) 작업을 위한 트랜스포머 대안 아키텍처의 발전 방향에 대한 논의도 포함되어 있습니다.



### Transversal PACS Browser API: Addressing Interoperability Challenges in Medical Imaging Systems (https://arxiv.org/abs/2412.14229)
Comments:
          16 pages with 3 figures

- **What's New**: 본 논문은 DICOM 이미지를 효과적으로 질의하고 검색할 수 있는 사용자 친화적인 Transversal PACS Browser API를 소개합니다. 이 API는 고급 필터링 기능과 사용자 맞춤 검색 필드를 제공하여 의료 이미지 저장소를 손쉽게 탐색할 수 있도록 설계되었습니다. 또한, 여러 PACS 스테이션에서 통합된 인터페이스를 통해 데이터 액세스의 복잡성과 단편화 문제를 해결합니다.

- **Technical Details**: 애플리케이션 개발은 Qt 6.6.0 기반의 Qt Creator 통합 개발 환경에서 이루어졌으며, C++로 다중 플랫폼 앱 개발을 지원합니다. 사용자는 PACS 서버에 연결하고, 다양한 기준에 따라 검색을 실행하며, 특정 이미지를 로컬로 저장하고 미리 볼 수 있는 기능을 제공합니다. Qt의 GUI 도구는 사용자 입력과 백엔드 작업 사이의 원활한 상호작용을 통해 직관적인 인터페이스를 만드는데 기여했습니다.

- **Performance Highlights**: 포괄적인 테스트 결과 API의 디자인이 깔끔하고 사용이 용이하며 강력한 검색 기능이 있다는 것을 demonstrated 합니다. 이미지 미리보기 기능은 사용자가 의료 이미지를 보다 효율적으로 조회하고 활용할 수 있도록 지원합니다. 이 API는 의료 제공자에게 효율적인 자원의 접근 방식을 제공하며 전반적인 의료 서비스의 질을 향상시키는데 기여합니다.



### Whom do Explanations Serve? A Systematic Literature Survey of User Characteristics in Explainable Recommender Systems Evaluation (https://arxiv.org/abs/2412.14193)
Comments:
          31 pages, 2 figures. Submitted to ACM Transactions of Recommender Systems

- **What's New**: 이 연구는 추천 시스템의 설명에 대한 사용자의 특성이 어떻게 인식되는지를 분석하는 중요한 빈틈을 다룹니다. 총 124개의 논문을 조사하여 사용자 특성에 따른 설명의 효과를 평가한 결과, 대부분의 연구가 추천 시스템 사용자의 전형적인 특징을 잘 반영하지 못하고 있음을 확인했습니다. 이는 현재 연구에서 도출된 통찰이 일반화되는 데 제약이 될 수 있습니다.

- **Technical Details**: 추천 시스템의 설명을 개선하기 위해 사용자 특성을 평가하는 방법이 필요합니다. 연구에서는 텍스트 기반 설명의 품질을 평가하기 위해 BLEU-n, ROUGE-n 및 BERT-S와 같은 자연어 처리 지표를 사용하는 동시에 Recall 및 NDCG와 같은 랭킹-회수 기반 메트릭을 사용합니다. 그러나 현재의 오프라인 평가 메트릭은 사용자의 동기와 행동을 포착하지 못합니다.

- **Performance Highlights**: 설명 추가는 사용자 신뢰도를 향상시키고 의사결정을 개선하는 데 기여할 수 있습니다. 하지만, 다양한 사용자 특성을 고려한 연구가 부족하여 설명의 효과가 어떻게 달라지는지를 이해하는 데 한계가 있습니다. 향후 연구는 사용자 모집 및 데이터 보고의 일관성을 높여야 하며, 이러한 분야의 포괄적인 평가를 통해 추천 시스템의 설명 품질을 개선할 수 있는 기회를 제공해야 합니다.



New uploads on arXiv(cs.CV)

### UIP2P: Unsupervised Instruction-based Image Editing via Cycle Edit Consistency (https://arxiv.org/abs/2412.15216)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 지시 기반 이미지 편집을 위한 비지도 학습 모델 UIP2P를 제안합니다. 기존의 감독 학습 방식은 입력 이미지, 편집 이미지 및 편집 지침의 삼중 데이터셋에 의존하고 있어 일반화 능력이 제한적이었습니다. Cycle Edit Consistency (CEC)라는 새로운 편집 메커니즘을 도입하여, 고정된 이미지와 텍스트 간의 일관성을 확보하며 기존 데이터셋의 필요를 없앴습니다.

- **Technical Details**: UIP2P 모델은 CLIP 임베딩 공간의 텍스트-이미지 정렬을 이용하여 편집 일관성을 확보하고, 이미지와 주의(attention) 공간의 일관성을 enforcing 합니다. CEC 메커니즘을 통해 입력 이미지에 정확하고 일관된 수정을 가할 수 있으며, 이 과정에서 원본 내용의 무결성을 유지합니다. 이 방법은 실제 이미지 데이터셋에서 학습할 수 있는 가능성을 열어줍니다.

- **Performance Highlights**: 우리의 비지도 기법은 고충실도 및 정밀도를 갖춘 광범위한 편집에서 기존 방법들보다 우수한 성능을 보여줍니다. 기존의 삼중 데이터셋에 대한 의존성을 감소시키고, 감독 방법에서의 편향 문제를 줄여, 지시 기반 이미지 편집의 규모와 범위를 크게 확장할 수 있습니다. 결과적으로, 이 연구는 지시 기반 이미지 편집 분야에서의 중요한 발전을 나타냅니다.



### EnvGS: Modeling View-Dependent Appearance with Environment Gaussian (https://arxiv.org/abs/2412.15215)
Comments:
          Project page: this https URL

- **What's New**: 이번 연구에서는 복잡한 반사를 효과적으로 모델링하기 위한 새로운 접근법인 EnvGS를 제안합니다. 기존의 방법들이 가지는 한계를 극복하고, 환경의 반사를 캡쳐하기 위해 Gaussian primitives를 사용하는 명시적 3D 표현 방식을 도입했습니다. 이로 인해 근접 반사 및 고주파 반사 세부 사항을 보다 정확하게 재구성할 수 있습니다.

- **Technical Details**: EnvGS는 환경 Gaussian과 기본 Gaussian을 결합하여 장면의 기하학적 구조와 외관을 모델링합니다. 이를 위해 ray-tracing 기반의 GPU 렌더러를 개발하였으며, 이는 실시간 렌더링 속도를 유지하면서도 우리가 모델링한 Gaussian primitives를 효율적으로 렌더링할 수 있게 해줍니다. CUDA와 OptiX를 활용하여 경량 솔루션을 제공하며, BVH(bounding volume hierarchy)를 사용해 렌더링 과정을 최적화합니다.

- **Performance Highlights**: 여러 실제 및 합성 데이터 세트를 통해 우리의 접근법이 기존의 방법들보다 우수한 세부 사항을 반영하는 것을 확인했습니다. EnvGS는 리얼타임 사진 현실적인 반사 합성을 가능하게 하며, 특히 복잡한 반사를 처리하는 데 뛰어난 성능을 보여주었습니다. 이번 연구는 실시간 소스에서 가장 뛰어난 렌더링 품질을 달성하며, 대규모의 실험을 통해 그 효과를 입증하였습니다.



### LeviTor: 3D Trajectory Oriented Image-to-Video Synthesis (https://arxiv.org/abs/2412.15214)
Comments:
          Project page available at this https URL

- **What's New**: 본 연구에서는 이미지-비디오 합성에서 3D 객체 경로 제어를 위한 혁신적인 방법인 LeviTor를 소개합니다. 이 방법은 깊이 정보와 K-평균 클러스터링된 포인트를 결합하여 사용자가 3D 경로를 더 쉽게 정의할 수 있도록 하여, 기존의 2D 입력에서 발생하는 애매함을 해결합니다. 또한, 새롭게 도입된 인터페이스는 사용자가 2D 이미지에서 경로를 그리기만 해도 실시간 3D 경로로 해석되도록 하여, 비전문가도 쉽게 사용할 수 있게 합니다.

- **Technical Details**: LeviTor는 패턴 인식을 위해 고품질 비디오 객체 분할 데이터셋인 SAM2를 활용합니다. 이 모델은 객체 마스크의 K-평균 클러스터링된 포인트와 깊이 정보를 결합하여 3D 경로의 제어 신호를 생성합니다. 이러한 방식을 통해 모델은 복잡한 물체의 움직임과 상호작용을 효과적으로 이해하고 생성할 수 있으며, 명시적인 3D 경로 추적 없이도 작업을 수행할 수 있습니다.

- **Performance Highlights**: LeviTor는 기존의 방법들과 비교하여 정량적 및 정성적으로 우수한 성능을 보여줍니다. 실험 결과, 이 모델은 정밀한 3D 경로 제어가 필요한 이미지-비디오 합성 작업에서 높은 정확성을 나타내고, 정적 이미지로부터 현실적이고 일관된 객체 움직임을 생성함으로써 사용자 요구에 부합합니다. 이 모델은 사용자 친화적인 추론 파이프라인을 통해 비전문가도 쉽게 3D 경로를 입력할 수 있도록 하여, 다양한 사용자가 활용할 수 있도록 접근성을 높였습니다.



### Flowing from Words to Pixels: A Framework for Cross-Modality Evolution (https://arxiv.org/abs/2412.15213)
Comments:
          Project page: this https URL

- **What's New**: 이번 논문에서는 전통적인 접근 방식을 넘어 'CrossFlow'라는 새로운 접근 방식이 도입되었습니다. 이 방식은 Gaussian noise 배포에서 목표 미디어 배포로의 복잡한 매핑을 학습하는 대신, 서로 다른 모달리티 간의 직접적인 매핑을 학습하는 것을 목표로 합니다. 이를 통해 노이즈 배포와 조건 메커니즘의 필요성을 없애는 혁신적인 전환을 제안합니다. 이 기법은 다양한 교차 모달/내부 모달 매핑 작업에서 뛰어난 성능을 보여주고 있습니다.

- **Technical Details**: CrossFlow는 서로 다른 모달리티 간의 매핑을 위한 일반적인 프레임워크로, Variational Encoder를 적용하여 소스 모달리티 데이터 배포를 타겟 모달리티와 동일한 형태로 인코딩합니다. 또, Classifier-Free Guidance (CFG)를 도입해 조건 메커니즘 없이도 개선된 생성 품질을 달성합니다. 이는 기존 기술이 가진 한계를 극복하고, 더 효율적이고 의미 있는 구조를 제공합니다. 특히, 텍스트-이미지 생성 과제에서 CrossFlow는 특정 조건 없이도 기존의 플로우 매칭 메소드보다 더 나은 성능을 발휘합니다.

- **Performance Highlights**: CrossFlow는 동일한 데이터, 모델 크기, 그리고 훈련 예산을 사용했을 때, 알려진 플로우 매칭 기준선들을 능가하는 결과를 도출했습니다. 이 방식은 또한 다양한 Large Language Models (LLMs)와 호환성이 좋으며, 훈련 단계나 모델 크기를 확대하는 데에서도 뛰어난 성능을 발휘합니다. 특히, CrossFlow를 활용한 잠재적 산술은 강력한 창의적 결과를 가져와서, 예를 들어 개가 선글라스를 쓴 이미지를 생성하는 등 실험적 결과를 보여줍니다.



### Scaling 4D Representations (https://arxiv.org/abs/2412.15212)
- **What's New**: 본 논문은 비디오에서 순수 자가 감독 학습(self-supervised learning)의 확대 가능성을 탐구합니다. 기존 연구가 의미론적 작업에 주목한 반면, 본 연구는 카메라 자세 추정(camera pose estimation), 포인트 및 객체 추적(point and object tracking), 깊이 추정(depth estimation) 등 비의미적(non-semantic) 비전 작업에 초점을 맞춥니다. 이 연구에서는 Video Masked Auto-Encoding (MAE)와 22B 매개변수를 가진 학습된 Transformer 비디오 모델을 통해 4D 작업의 성능 향상을 입증합니다.

- **Technical Details**: 우리는 모델을 동일한 주의 기반(readout) 구조를 사용하여 apples-to-apples 비교를 통해 평가했습니다. 연구에서 사용된 모델은 이미지와 비디오 모델 모두를 포함하며, MAE와 V-JEPA는 비디오 자가 감독 방식에서 좋은 성능을 보였습니다. 또한, 특정 데이터셋에서 모델 크기를 20M에서 22B까지 증가시키면서 논증된 일관된 성능 향상을 관찰했습니다.

- **Performance Highlights**: 논문 결과, 이미지 모델들은 경쟁력이 떨어졌으며, MAE 기반 비디오 모델들이 4D 작업에서의 성능을 크게 향상시킬 수 있음을 보여주었습니다. 특히, MAE는 기존의 자가 감독 없이 학습된 모델들과 비교할 때 탁월한 성능을 보였습니다. 이 연구의 성과는 2025년 초 공개될 새로운 MAE-VIT 모델들을 포함하며, 이러한 모델들은 4D 장면 표현을 향상시키는 데 중요한 기여를 할 것으로 기대됩니다.



### Generative Multiview Relighting for 3D Reconstruction under Extreme Illumination Variation (https://arxiv.org/abs/2412.15211)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 다양한 조명 환경에서 촬영된 사진으로부터 객체의 기하학적 구조와 외관을 복원하는 새로운 접근방식을 제안합니다. 기존의 논문들에서는 조명 변화로 인해 발생하는 시각적 불일치를 해결하지 못했으나, 저자들은 다중 보기 리라이팅(diffusion relighting) 모델을 통해 동일한 기준 조명에서 모든 이미지를 리라이트하여 강건한 복원을 수행합니다. 이를 통해 극단적인 조명 변화가 있는 이미지로부터 고충실도의 외관을 복원하는 데 크게 향상된 성능을 입증하였습니다.

- **Technical Details**: 저자들은 다중 보기 리라이팅 모델을 사용하여 입력된 모든 이미지를 기준 이미지의 조명에 맞게 조정합니다. 이후, 이러한 리라이트된 이미지를 바탕으로 NeRF(Neural Radiance Fields) 아키텍처를 통해 3D 구조와 외관을 복원합니다. 또한, 'shading embedding'이라고 불리는 기존의 표면 노멀 벡터에 대한 추가 수정을 통한 기법을 도입하여 학습된 모델의 오류를 수정하는 데 기여합니다.

- **Performance Highlights**: 저자들은 제안한 방법을 Objaverse와 NAVI 데이터셋에서 검증하여, 기존의 기술들보다 질적으로 및 양적으로 개선된 결과를 보여주었다고 보고하였습니다. 이 연구는 특히 비산란(비확산) 객체의 시각적 외관을 보다 정확하게 복원할 수 있는 가능성을 열어줍니다. 또한, 다양한 조명 조건에서 포착된 이미지로부터 3D 모델 재구성을 신뢰성 있게 수행할 수 있음이 강조되었습니다.



### PRIMA: Multi-Image Vision-Language Models for Reasoning Segmentation (https://arxiv.org/abs/2412.15209)
Comments:
          Project page: this https URL

- **What's New**: 이번 연구는 다중 이미지 픽셀 기반 추론 세분화(multi-image pixel-grounded reasoning segmentation)의 새로운 작업을 제안합니다. 이는 두 개 이상의 이미지를 포함한 비교 질문에 대한 자연어 응답을 생성하며, 관련 객체와 부분에 대한 픽셀 수준의 기반을 함께 제공합니다. 또한, M4Seg라는 새로운 벤치마크를 구축하여 224K 이상의 질문-응답 쌍과 다중 이미지 시나리오에서 필요한 세밀한 추론을 지원합니다.

- **Technical Details**: PRIMA라는 새로운 LVLM은 픽셀 수준의 기반을 다양한 이미지와 결합하여 강력한 멀티 이미지 추론 기능을 통합했습니다. 이 모델은 여러 이미지에 걸쳐 세부적인 시각적 표현을 질의할 수 있는 효율적인 비전 모듈을 중심으로 하며, 25.3%의 TFLOPs 감소를 이뤘습니다. PRIMA는 명령 기반의 다중 이미지 적응 모듈을 통해 다중 이미지를 통한 세밀한 연관성을 추론합니다.

- **Performance Highlights**: 실험 결과, PRIMA는 최신 기법들보다 우수한 성능을 보였습니다. 이 모델은 다중 이미지 시나리오에서 자연어 응답과 문맥에 기반한 세분화를 생성할 수 있으며, 높은 정밀도의 픽셀 수준 추론을 유지하면서도 계산 효율성을 크게 향상시켰습니다. 이는 다양한 응용 분야에서 LVLM의 해석 가능성을 증가시키는 데 기여할 수 있습니다.



### OpenEMMA: Open-Source Multimodal Model for End-to-End Autonomous Driving (https://arxiv.org/abs/2412.15208)
- **What's New**: 최근 다중 모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 발전에 힘입어, 오픈소스 기반의 자율주행(Autonomous Driving, AD) 프레임워크인 OpenEMMA가 제안되었습니다. OpenEMMA는 기존의 EMMA 모델의 기능을 재현하면서도, 다양한 MLLMs의 힘을 활용하여 해결 가능한 문제들을 보완합니다. 이 새로운 접근법은 복잡한 시각적 데이터를 처리하고 주행 시나리오에 대한 추론을 가능하게 하여, 자율주행 기술의 발전을 크게 이끌 것으로 기대됩니다.

- **Technical Details**: OpenEMMA는 MLLMs를 기반으로 하여, 과거 주행 상태와 시각적인 주행 장면을 입력으로 사용하여 미래 경로(P𝑃)를 예측하는 시스템입니다. 이를 위해 Chain-of-Thought reasoning 과정을 통합하여, 차량 속도 벡터와 곡률 벡터를 중간 표현으로 생성하며, 이는 인간의 주행 방식을 반영합니다. 이러한 인간과의 상호작용을 통해 OpenEMMA는 해석 가능한 방식으로 자율주행 결정을 수행하며, 이를 구현하기 위해 YOLO 모델을 기반으로 한 3D 바운딩 박스를 예측하는 세부 조정 버전을 통합하고 있습니다.

- **Performance Highlights**: OpenEMMA는 nuScenes 데이터셋의 검증 세트를 대상으로 한 실험에서 다양한 MLLMs을 이용하여 효과성과 적응력을 입증했습니다. 이 시스템은 복잡한 주행 시나리오에서도 견고성과 일반화 능력을 보여주었으며, 기존의 AD 프레임워크보다 높은 효율성과 효과성을 제공합니다. 또한, OpenEMMA는 연구 커뮤니티가 활용할 수 있도록 코드베이스, 데이터셋, 모델 가중치를 완전히 공개하여 자율주행 기술 발전을 위한 기반을 만들어 줍니다.



### AutoTrust: Benchmarking Trustworthiness in Large Vision Language Models for Autonomous Driving (https://arxiv.org/abs/2412.15206)
Comments:
          55 pages, 14 figures

- **What's New**: 최근 자율 주행(Autonomous Driving, AD)을 위한 대형 비전 언어 모델(Vision Language Models, VLMs)의 발전이 강력한 장면 이해와 추론 능력을 보여주며, 이는 엔드-투-엔드 드라이빙 시스템의 유력한 후보가 되었습니다. 그러나 DriveVLM의 신뢰성(trustworthiness) 문제에 대한 연구는 아직 부족하여, 이는 공공 교통 안전에 직접적인 영향을 미치는 중요한 요소입니다. 이 논문에서는 신뢰성 기준인 AutoTrust를 도입하고, 이를 통해 다양한 시각에서 DriveVLM을 평가할 수 있는 종합 벤치마크를 제시합니다.

- **Technical Details**: AutoTrust는 신뢰성, 안전성, 강인성(robustness), 개인 정보 보호(privacy), 공정성(fairness)을 포함하여 자율 주행 모델을 평가하기 위한 첫 번째 종합 벤치마크입니다. 우리는 10,000개 이상의 고유 장면과 18,000개의 질의로 구성된 대규모 시각 질의 응답 데이터셋을 구축했습니다. 이 연구에서는 일반적인 모델과 전문 모델을 포함한 여섯 개의 공개된 VLM를 평가하며, 이 과정에서 기존에 발견되지 않은 DriveVLM의 취약점을 밝혀냈습니다.

- **Performance Highlights**: 연구 결과, LLaVA-v1.6과 GPT-4o-mini와 같은 일반 VLM이 자율 주행에 특화된 모델보다 전반적인 신뢰성에서 더 우수한 성능을 보였습니다. 반면, DriveVLM과 같은 모델은 민감한 정보 유출에 특히 취약하다는 것을 발견했습니다. 또한, 모든 모델이 적대적 공격(adversarial attacks)에 취약하며, 다양한 환경과 인구에서 편향 없는 의사결정을 보장하는 데 어려움을 겪고 있습니다. 이러한 발견은 자율 주행 시스템의 신뢰성을 보장하기 위한 즉각적이고 단호한 조치의 필요성을 강조합니다.



### FlowAR: Scale-wise Autoregressive Image Generation Meets Flow Matching (https://arxiv.org/abs/2412.15205)
- **What's New**: FlowAR는 이미지 생성에서 다음 스케일 예측을 위한 유연하고 전반적인 방법론을 제안합니다. 기존의 복잡한 다중 스케일 구조를 대신해 이전 스케일의 두 배 크기로 간단한 확장을 통해 사용자 친화성을 향상시킵니다. 이를 통해 VAR의 복잡한 구조를 없애고 모든 상용 Variational AutoEncoder (VAE)와의 호환성을 확보하게 됩니다.

- **Technical Details**: FlowAR는 비율 가중 평균(Flow Matching)을 통합하여 각 스케일에서 확률 분포를 학습합니다. 스케일 사이의 계층적 의존성을 포착하는 동안, Spatially Adaptive Layer Normalization (Spatial-adaLN) 기술을 사용하여 정교한 세부 사항을 보다 잘 생성할 수 있도록 합니다. 이러한 접근 방식 덕분에 최종 이미지는 최상의 해상도에서 예측된 잠재 표현을 디토크나이징하여 생성됩니다.

- **Performance Highlights**: FlowAR는 도전적인 ImageNet-256 벤치마크에서 기존 방법들에 비해 우수한 생성 성능을 입증했습니다. 모델은 시퀀스와 확률적 측면을 포착하는 것은 물론 다중 스케일 정보를 통한 이미지 합성 개선을 통해 가장 앞선 결과를 달성했습니다. 이를 통해 FlowAR의 효과성과 효율성을 강조하며 향후 발전 가능성을 보여줍니다.



### DI-PCG: Diffusion-based Efficient Inverse Procedural Content Generation for High-quality 3D Asset Creation (https://arxiv.org/abs/2412.15200)
Comments:
          Project page: this https URL

- **What's New**: DI-PCG는 일반 이미지 조건에서 효율적인 역 절차적 콘텐츠 생성(Inverse Procedural Content Generation, I-PCG)을 위한 혁신적인 방법론을 제시합니다. 이 방법은 경량의 diffusion transformer 모델을 기반으로 하며, PCG 매개변수를 직접 비노이즈 타겟으로 취급하고, 관찰된 이미지를 매개변수 생성을 제어하는 조건으로 사용합니다.

- **Technical Details**: DI-PCG는 단 7.6M의 네트워크 매개변수와 30 GPU 시간만으로 훈련할 수 있으며, 몇 초 이내에 샘플을 생성할 수 있습니다. 이 모델은 iterative denoising score-matching 훈련을 통해 절차적 생성기의 매개변수 공간을 학습하고, 관찰된 이미지를 기반으로 수치적으로 샘플링을 수행합니다. 이렇게 생성된 매개변수는 PCG에 투입되어 고품질의 3D 자산을 생성하게 됩니다.

- **Performance Highlights**: DI-PCG는 효율성과 효과성을 모두 갖춘 시스템으로, 현실 세계의 데이터에 대한 일반화 능력 또한 뛰어납니다. 정량적 및 정성적 실험 결과를 통해 자산 생성 및 역 PCG 작업에서 DI-PCG의 효과가 명확히 검증되었습니다. 이 방법은 3D 자산 생성을 위한 파라미트릭 모델을 사용하는 효율적인 역 PCG를 가능하게 하며, 후속 애플리케이션을 위한 고품질 3D 자산 생성을 지원합니다.



### LiDAR-RT: Gaussian-based Ray Tracing for Dynamic LiDAR Re-simulation (https://arxiv.org/abs/2412.15199)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 동적 주행 시나리오에서의 실시간 LiDAR 재시뮬레이션 문제를 다룹니다. 기존 방법들이 높은 계산 비용으로 인해 대규모 장면에서 한계를 겪는 반면, LiDAR-RT라는 새로운 프레임워크를 제안하여 현실적인 LiDAR 재시뮬레이션을 실시간으로 수행할 수 있는 방법을 제시합니다. 이 프레임워크는 Gaussian primitives와 하드웨어 가속 레이 트레이싱 기술을 통합하여 효율적인 렌더링 파이프라인을 개발하였습니다.

- **Technical Details**: 제안된 LiDAR-RT 프레임워크는 LiDAR 센서의 물리적 특성을 모델링하기 위해 학습 가능한 매개변수를 갖춘 Gaussian primitives를 사용합니다. 또한, 장면 그래프(scene graphs)를 통합하여 동적 장면에 대한 유연한 모델링을 가능하게 합니다. BVH(바운딩 볼륨 계층)를 구성하고 각 픽셀에 대해 레이를 캐스팅하여 LiDAR 뷰를 생성하는 과정은 미분 가능 렌더링 알고리즘을 통해 이루어집니다.

- **Performance Highlights**: 제안된 방법은 Waymo Open 및 KITTI-360 데이터세트를 사용하여 평가되었으며, 렌더링 품질과 효율성 모두에서 최첨단 성능을 달성하였습니다. 또한, 다양한 편집 조작에서도 높은 적응력을 보여주어 LiDAR 재시뮬레이션의 가능성을 크게 확장시켰습니다. 이 연구는 동적 장면에서의 LiDAR 렌더링 기술에 새로운 길을 열 것으로 기대됩니다.



### Preventing Local Pitfalls in Vector Quantization via Optimal Transpor (https://arxiv.org/abs/2412.15195)
Comments:
          Code is available at this https URL

- **What's New**: 이 연구에서는 Vector-quantized networks (VQNs)의 훈련 불안정성을 해결하기 위해 새로운 기법인 OptVQ를 제안합니다. 기존의 최근접 이웃 검색(nearest neighbor search) 방법 대신 최적 운송(optimal transport) 방법을 통합하여, 더 글로벌하게 데이터를 효율적으로 할당하는 것이 목표입니다. 이를 통해 OptVQ는 100%의 코드북(codebook)을 활용하며, 기존 VQNs보다 더 높은 이미지 재구성 품질을 보여줍니다.

- **Technical Details**: OptVQ는 Sinkhorn 알고리즘(Sinkhorn algorithm)을 사용하여 최적 운송 문제를 최적화하는 새로운 벡터 양자화(vector quantization) 방법입니다. 이 방법은 데이터와 코드북 간의 글로벌 구조(global structure)를 활용하여 양자화를 수행하며, 훈련 중 불안정성을 줄이기 위해 간단한 정규화(normalization) 기법을 적용합니다. 이를 통해 VQNs의 훈련 과정에서 발생하는 '인덱스 붕괴(index collapse)' 문제를 해결합니다.

- **Performance Highlights**: 실험 결과, OptVQ는 기존의 최첨단 VQNs보다 뛰어난 재구성 품질을 달성하며, 훈련의 안정성이 크게 향상되었습니다. 기존에는 섬세한 초기화나 모델 증류(model distillation)와 같은 복잡한 훈련 기법이 필요했지만, OptVQ는 이러한 필요 없이 훈련을 안정화합니다. 이는 VQNs에서 흔히 발생하는 로컬(optimal) 문제를 회피하는 데 중요한 역할을 합니다.



### AV-Link: Temporally-Aligned Diffusion Features for Cross-Modal Audio-Video Generation (https://arxiv.org/abs/2412.15191)
Comments:
          Project Page: this http URL

- **What's New**: 이번 논문에서는 AV-Link라는 통합 프레임워크를 제안합니다. 이 프레임워크는 영상과 음성을 서로 변환하는 작업을 위한 것으로, 동결된 비디오 및 오디오 확산 모델의 활성화를 활용하여 시간적으로 정렬된 크로스 모달 조절을 가능하게 합니다. 특히, Fusion Block을 통해 비디오와 오디오 간의 양방향 정보 교환을 지원하며, 기존 모델들이 갖고 있던 문제를 해결하고자 합니다.

- **Technical Details**: AV-Link는 사전 훈련된 피처 추출기를 사용하지 않고, 서로 잘 훈련된 오디오 및 비디오 생성 모델의 활성화를 조건으로 사용합니다. 이 방법은 생성된 미디어의 활성화가 시간적으로 정렬되어 있어, 보다 정밀한 세멘틱 및 시간 정렬을 달성할 수 있게 합니다. 또한, Fusion Block을 사용하여 두 가지 모달리티 간의 정보를 효과적으로 교환합니다.

- **Performance Highlights**: AV-Link는 VGGSounds 데이터셋에서 최첨단 V2A(Videoto-Audio) 모델들과 비교했을 때 뛰어난 성능을 보였으며, 특히 시간 정렬(Onset ACC)에서 기존 모델보다 36.5% 향상된 결과를 나타냈습니다. 사용자 연구에서도 우리의 방법이 기존 모델보다 큰 선호도를 얻었고, 세멘틱 및 시간 정렬 측면 모두에서 평균 56.8% 및 63.6%의 선호성을 지닌 것으로 확인되었습니다.



### EarthDial: Turning Multi-sensory Earth Observations to Interactive Dialogues (https://arxiv.org/abs/2412.15190)
- **What's New**: 이 논문에서는 EarthDial이라는 대화형 비전-언어 모델(VLM)을 도입하여 지구 관측(EO) 데이터를 처리할 수 있는 획기적인 방법을 제안합니다. EarthDial은 복잡한 다중 감각 지구 관측 데이터를 자연어 대화로 변환할 수 있는 기능을 갖추고 있습니다. 이 모델은 다중 분광, 다중 시계열 및 다중 해상도 이미지를 지원하여 다양한 원거리 감지(Remote Sensing) 작업을 수행합니다.

- **Technical Details**: EarthDial은 11.11M 이상의 지침 쌍으로 구성된 방대한 튜닝 데이터셋을 사용하여 다중 해상도, 다중 분광 및 다중 시계열 원거리 감지 이미지를 처리할 수 있게 설계되었습니다. 이 모델은 InternVL 아키텍처를 기반으로 하며, 다중 감각 입력을 효과적으로 처리하기 위해 단계별 파인튜닝 과정을 가지고 있습니다. EarthDial은 경량화된 모델로, 시각 인코더와 대형 언어 모델(LLM)을 연결하는 간단한 MLP를 통해 시각 토큰을 LLM 공간으로 매핑합니다.

- **Performance Highlights**: 실험 결과, EarthDial은 37개의 하위 작업에서 기존의 일반 및 도메인 특화 모델보다 우수한 정확도와 일반화 성능을 보였습니다. 이를 통해 원거리 감지 작업에서의 강력한 성능을 입증하며, 다양한 EO 작업에 대한 처리 능력을 향상시킵니다. 이 모델은 분류, 객체/변화 감지, 질문 응답 및 이미지 설명과 같은 다양한 작업을 수행할 수 있어 실제 환경에서 응용 가능성이 높습니다.



### Tiled Diffusion (https://arxiv.org/abs/2412.15185)
- **What's New**: 이번 논문의 가장 큰 혁신은 Tiled Diffusion이라는 새로운 접근 방식을 제안한다는 점입니다. 이는 다양한 이미지 생성 분야에서 일관된 타일링 패턴 생성을 지원합니다. 기존의 방법들이 주로 단일 이미지 생성을 위한 단순 타일링에 초점을 맞추었던 반면, Tiled Diffusion은 여러 이미지 간의 연결된 타일의 자동 생성을 가능하게 합니다. 이를 통해 수작업의 필요를 없애고 다양한 응용 프로그램에서 창의적인 가능성을 확대시킵니다.

- **Technical Details**: Tiled Diffusion은 타일링 문제를 유연하게 정의하고, 여러 이미지를 연결할 수 있는 다양한 시나리오를 지원합니다. 이 방법은 이미지의 네 가지 면을 가진 사각형으로 각 이미지를 간주하고, 그런 다음 특정 제약 조건을 통해 각 변이 서로 연결될 수 있도록 구성됩니다. 이 접근 방식은 자아 타일링, 일대일 타일링 및 다대다 타일링을 포함하여 다양한 타일링 시나리오를 지원합니다.

- **Performance Highlights**: Tiled Diffusion은 기존 이미지를 매끄럽게 타일링하거나, 타일 가능 텍스처를 생성하고, 360° 합성을 지원하는 데 효과적임이 입증되었습니다. 이 논문은 타일링된 패턴의 일관성 및 지역 세부 사항을 유지하면서 복잡한 배치에서 스타일 일관성을 보장하는 방법을 설명합니다. 또한, 타일링 과정에서 발생할 수 있는 아티팩트를 제거하는 유사성 제약 조건을 도입하여 복잡한 타일링 시나리오에서의 효과성을 극대화하고 있습니다.



### SqueezeMe: Efficient Gaussian Avatars for VR (https://arxiv.org/abs/2412.15171)
Comments:
          Initial version

- **What's New**: 이번 논문에서는 이전의 Gaussian Splatting 방식을 활용해, Meta Quest 3 가상 현실 헤드셋에서 여러 개의 인간 아바타를 실시간으로 렌더링할 수 있는 방법을 제안하고 있습니다. 특히, 높은 품질의 데이터셋을 사용하여 아바타의 모양과 외관을 자주 업데이트할 수 있도록 하고, 디코더 및 렌더링 단계를 최적화하여 계산 성능을 향상시켰습니다. 이를 통해 한 번에 3개의 Gaussian 아바타를 72 FPS로 실행할 수 있는 성과를 거두었습니다.

- **Technical Details**: 연구팀은 Animatable Gaussians를 UV 공간에서 훈련하여 디코더의 효율성을 높였고, 디코더 계층을 단일 신경망으로 압축했습니다. 또한, Gaussian의 이웃들이 디코더에서 공유하는 수정값을 통해 추가적인 속도 개선을 이루었습니다. 렌더링을 가속화하기 위해 Vulkan에 맞춤형 파이프라인을 개발하여 모바일 GPU에서 실행하도록 하였습니다.

- **Performance Highlights**: 최종적으로, 제안된 방법을 통해 Meta Quest 3 VR 헤드셋에서 실시간으로 다수의 아바타를 렌더링하는 데 있어 큰 성과를 이뤄냈습니다. 이 방식은 사용자에게 더 몰입감 있는 경험을 제공하며, 아바타의 외관 품질도 유지할 수 있어 실제 사람과 유사한 상호작용을 가능하게 합니다. 이를 통해 VR 환경에서의 아바타 인터페이스의 잠재력을 크게 확장할 수 있습니다.



### OnlineVPO: Align Video Diffusion Model with Online Video-Centric Preference Optimization (https://arxiv.org/abs/2412.15159)
- **What's New**: 이 논문에서는 OnlineVPO라는 새로운 비디오 확산 모델을 위한 효율적이고 효과적인 선호 학습 접근 방식을 소개합니다. 기존의 이미지 기반 보상 모델이나 비디오 판별 모델에 의존하는 방법 대신, 합성 데이터로 훈련된 비디오 품질 평가 모델을 직접 사용하여 비디오 생성 품질 향상에 필요한 맞춤형 선호 피드백을 제공합니다. 또한, 온라인 DPO 알고리즘을 도입하여 비디오 선호 학습 프레임워크의 비정책 최적화 및 확장성 문제를 해결합니다.

- **Technical Details**: OnlineVPO는 비디오 생성 모델의 훈련 과정에서 온라인 피드백을 제공하기 위해 비디오 보상 모델을 활용합니다. 이 방식은 훈련 중에 생성된 비디오에 대해 즉각적인 평가 및 순위를 매기고, 이를 기반으로 정책 네트워크를 지속적으로 개선합니다. 또한, DPO 손실 함수를 사용하여 실시간 피드백에 따라 학습을 진행합니다.

- **Performance Highlights**: 광범위한 실험을 통해 OnlineVPO는 비디오 확산 모델을 위한 간단하면서도 효과적이며, 확장성이 뛰어난 선호 학습 알고리즘으로 입증되었습니다. 특히, 비디오 품질을 즉각적으로 평가하고 피드백을 제공할 수 있어 비디오 생성 과정에서의 품질 개선에 기여할 수 있는 기회를 제공합니다. 이러한 점에서 OnlineVPO는 비디오 생성 분야의 향후 발전에 중요한 인사이트를 제공합니다.



### Prompt-A-Video: Prompt Your Video Diffusion Model via Preference-Aligned LLM (https://arxiv.org/abs/2412.15156)
- **What's New**: 이 논문은 비디오 생성 모델의 텍스트 입력을 향상시키기 위한 새로운 프로프트 최적화 시스템, Prompt-A-Video를 소개합니다. 이 시스템은 사용자의 입력을 자동적으로 보완하여 모델의 선호도에 맞춘 고품질 비디오 중심의 프로프트를 생성합니다. 할당된 보상을 기반으로 한 이중 단계 최적화 프로세스는 비디오 생성 품질을 크게 향상시키는 데 기여합니다.

- **Technical Details**: Prompt-A-Video는 사용자 제공 프로프트를 진화 알고리즘을 통해 자동으로 개선하는 보상 안내형 프로프트 진화 파이프라인을 특징으로 하며, 원본과 보정된 프로프트의 데이터 쌍을 생성합니다. 그런 다음, Supervised Fine-Tuning(SFT) 모델을 사용하여 다차원 보상을 평가하고, Direct Preference Optimization(DPO)을 통해 모델의 선호도에 맞게 정렬됩니다. 이러한 과정은 비디오 생성을 위한 최적의 프로프트를 생성하는 데 중점을 둡니다.

- **Performance Highlights**: Prompt-A-Video는 다양한 평가 지표에서 비디오 생성 성능을 향상시키는 데 뛰어난 효율성을 입증하였습니다. 또한, 텍스트-이미지 전환 시나리오에서도 상당한 개선을 보여 주며, 이는 모델의 강인함과 일반성을 나타냅니다. 이 시스템은 비디오 생성의 새로운 가능성을 열어주는 유망한 방법론으로 자리매김하고 있습니다.



### Leveraging Color Channel Independence for Improved Unsupervised Object Detection (https://arxiv.org/abs/2412.15150)
Comments:
          38 pages incl. references, 16 figures

- **What's New**: 이 논문에서는 객체 중심 아키텍처(object-centric architectures)가 RGB 색상 공간으로 인코딩된 이미지들을 통해 객체 표현을 독립적으로 추출할 수 있는 능력을 강조합니다. 연구자들은 RGB 이미지가 성능 향상에 최적이라고 여겨지는 전통적 견해에 도전하며, HSV와 같은 다른 색상 공간이 객체 중심 표현 학습(object-centric representation learning)에 필수적인 특성을 지닌다는 점을 논의합니다. 또한, 충족해야 하는 색상 채널을 예측하도록 요구함으로써 모델의 성능이 개선된다는 것을 보여줍니다.

- **Technical Details**: 본 연구는 Slot Attention(SA) 기술적 배경을 활용하여 객체 중심 대표 학습을 위한 실험을 진행합니다. 입력 데이터를 단일 고정 크기 잠재 벡터로 변환하는 전통적인 인코더-디코더 아키텍처와 달리, SA는 개별 객체로 분해된 잠재 공간을 유도하고, 입력에 대해 경쟁을 통한 정보를 분산하여 슬롯(slot)을 업데이트하는 방식으로 작동합니다. 연구에서는 색상 공간을 조합하여 새로운 복합 색상 공간(composite color spaces)을 창출하며, 이 복합 색상 공간이 비효율적인 기존의 RGB 색상 공간보다 효과적이라는 것을 입증합니다.

- **Performance Highlights**: 논문에서는 제안한 복합 색상 공간이 기존의 RGB 색상 공간에 비해 다섯 개의 다중 객체 데이터셋에서 객체 탐지(object detection) 및 속성 분리(property disentanglement) 성능에서 크게 개선됨을 보여줍니다. 새로운 색상 공간은 모델 독립적이며 일반화 가능하고 효율적인 특성을 가지고 있어 다양한 비주얼 컴퓨팅 작업에 적용할 수 있으며, 객체 중심 학습(object-centric learning)을 넘어서는 컴퓨터 비전 과제에 대한 추가 연구를 촉진할 것으로 기대됩니다.



### Jet: A Modern Transformer-Based Normalizing Flow (https://arxiv.org/abs/2412.15129)
- **What's New**: 이번 연구에서는 coupling-based normalizing flows의 디자인을 재검토하며, 이전 모델에서 사용된 일반적인 설계 선택지를 세심하게 분석한다. 특히, 기존의 convolutional neural networks 대신 Vision Transformer 아키텍처를 적용하여 성능을 개선하였다. 이러한 접근은 더 단순하고 효율적인 구조를 가능하게 하며, 최신 모델들과의 비교에서도 뒤처지지 않도록 돕는다.

- **Technical Details**: Jet 모델의 구조는 매우 간단하며, 입력 이미지를 K개의 플랫 패치로 분할한 후, affine coupling layers를 반복적으로 적용하여 구성된다. 각 coupling layer의 입력은 벡터 형태로 변환되며, 이 과정을 통해 차원 분할 변환이 이루어진다. Scaling factor s와 bias b는 깊은 신경망을 통해 학습되며, 이를 통해 계산된 출력이 최종 샘플로 사용된다.

- **Performance Highlights**: 최종 모델은 간소화된 구조에도 불구하고, 일반적인 이미지 벤치마크에서 negative log-likelihood 기준으로 SOTA 결과를 달성했다. 또한, transfer learning 기법을 활용함으로써 과적합(overfitting)을 효과적으로 줄이는 데 성공하였다. 연구 결과는 strong normalizing flow 모델들이 더 강력한 생성 모델의 빌딩 블록으로 활용될 수 있음을 보여준다.



### Parallelized Autoregressive Visual Generation (https://arxiv.org/abs/2412.15119)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 비주얼 생성의 효율성을 높이기 위해 병렬화된 자기 회귀 모델 기법을 제안합니다. 기존의 순차적인 토큰 생성 방식에서 발생하는 느린 속도를 개선하는 동시에 자기 회귀 모델링의 장점을 유지합니다. 특히, 토큰 간의 의존성을 고려하여 약한 의존성을 가진 토큰은 병렬로 생성할 수 있도록 하였습니다.

- **Technical Details**: 제안된 접근법에서는 이미지의 초기 토큰을 순차적으로 생성하여 전반적인 맥락을 설정하고, 이후 공간적으로 먼 위치의 약한 의존성을 가진 토큰을 병렬로 생성합니다. 이 과정은 표준 자기 회귀 변환기(transfomer)에 쉽게 통합될 수 있으며, 순차적 및 병렬 생성 모드 간의 전환을 돕는 몇 가지 학습 가능한 토큰 임베딩을 사용합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 이미지와 비디오 생성 작업에서 각각 약 3.6배의 속도 향상과 최소 품질 저하(0.7 FID 및 10 FVD 이내)로 9.5배의 속도 향상을 달성했습니다. 이 접근법은 다양한 비주얼 도메인과 토크나이저와의 호환성이 높아 실제 애플리케이션에 매우 유용한 기술로 평가받고 있습니다.



### Knowing Where to Focus: Attention-Guided Alignment for Text-based Person Search (https://arxiv.org/abs/2412.15106)
- **What's New**: 이번 논문에서는 Text-Based Person Search (TBPS) 분야에서 경험했던 두 가지 주요 과제를 해결하기 위한 Attention-Guided Alignment (AGA) 프레임워크를 제안합니다. 첫째, 기존의 Masked Language Modeling (MLM) 접근법은 텍스트의 모든 단어를 동등하게 취급하는데, 이는 의미가 없는 단어들까지 마스킹되어 의미 있는 상호작용을 방해합니다. 둘째, 수작업으로 작성된 텍스트 설명이 자주 불완전하고 부정확한 문제를 해결하기 위해, AGA는 세 가지 주된 구성 요소를 포함하고 있습니다.

- **Technical Details**: AGA 프레임워크는 Attention-Guided Mask (AGM) 모델링과 Text Enrichment Module (TEM)으로 구성됩니다. AGM은 텍스트 인코딩 과정에서 파생된 주의(attention) 가중치를 집계하여 의미 있는 단어를 동적으로 마스킹합니다. TEM은 반복적이고 부정확한 텍스트 설명으로 인한 저품질 표현을 개선하기 위해 MLM의 예측으로 의미 있는 단어를 대체하여 텍스트 설명을 풍부하게 만들어줍니다.

- **Performance Highlights**: 세 가지 벤치마크에서 실시한 광범위한 실험을 통해 AGA의 효과를 입증하였으며, CUHK-PEDES, ICFG-PEDES 및 RSTPReid에서 각각 78.36%, 67.31%, 67.4%의 Rank-1 정확도를 기록하여 기존의 최첨단 방법들을 초월하는 성과를 보여주었습니다. 이 연구는 TBPS 데이터셋에서 더 높은 성능과 더 나은 전반적인 모델의 강건성을 입증한 것을 강조합니다.



### A Full Transformer-based Framework for Automatic Pain Estimation using Videos (https://arxiv.org/abs/2412.15095)
- **What's New**: 이번 연구에서는 환자의 고통을 줄이고 신뢰할 수 있는 평가를 제공하는 최적의 통증 관리 시스템을 설계하기 위한 자동 통증 추정의 중요성을 강조합니다. 본 논문은 Transformer in Transformer (TNT) 모델과 크로스-어텐션(cross-attention) 및 셀프-어텐션(self-attention) 블록을 활용한 새로운 풀 변환기 기반 프레임워크를 제안합니다.

- **Technical Details**: BioVid 데이터베이스의 비디오를 활용하여 모델의 성능을 상세히 설명합니다. Transformer in Transformer 구조는 각각의 어텐션 블록을 진행하면서 통증 평가 작업의 효율성을 크게 향상시키며, 이를 통해 각 작업에서의 일반화 능력을 보여주는 것을 목표로 합니다.

- **Performance Highlights**: 최신 기술(state-of-the-art) 성능을 나타내며, 제안된 모델이 모든 주요 통증 추정 작업에서 효과적이고 효율적임을 입증합니다. 이는 통증 관리 시스템 설계에서 중요한 발전을 이룬 것으로 평가됩니다.



### MultiverSeg: Scalable Interactive Segmentation of Biomedical Imaging Datasets with In-Context Guidanc (https://arxiv.org/abs/2412.15058)
Comments:
          Project Website: this https URL Keywords: interactive segmentation, in-context learning, medical image analysis, biomedical imaging, image annotation, visual prompting

- **What's New**: 이 연구는 다수의 관련 이미지를 신속하게 분할(segmentation)할 수 있도록 지원하는 시스템인 MultiverSeg를 제안합니다. 기존 방법들은 사람이 수작업으로 라벨링된 이미지를 요구하거나 각 이미지마다 반복적인 상호작용(interaction)을 필요로 했습니다. 하지만 MultiverSeg는 새로운 데이터셋을 분할하는 데 있어 기존 라벨된 데이터에 대한 접근 없이도 가능하게 합니다. 전체 데이터셋을 빠르게 분할하는 혁신적인 기능을 제공합니다.

- **Technical Details**: MultiverSeg는 사용자가 제공하는 클릭, 바운딩 박스, 낙서(scribbles)와 같은 상호작용을 입력받아 이미지를 분할합니다. 사용자가 더 많은 이미지를 분할할수록, 해당 이미지와 분할 결과는 모델에 추가적인 입력으로 제공되어 분할의 정확도를 높입니다. 이러한 방식은 라벨된 이미지의 컨텍스트(context) 세트가 증가함에 따라, 새로운 이미지 분할에 필요한 상호작용 수를 점진적으로 줄여줍니다.

- **Performance Highlights**: MultiverSeg를 사용하면, 상호작용 단계를 크게 줄이면서 새로운 데이터셋을 효율적으로 분할할 수 있습니다. 연구 결과, 기존의 최첨단(interactive) 분할 방법에 비해 MultiverSeg는 여러 이미지를 분할하기 위해 필요한 전체 낙서 단계에서 53%, 클릭에서 36%를 줄이며 90% Dice 지표를 달성했습니다. 이러한 성능 개선은 의료 이미지 수집 및 분석 작업에 있어 시간을 단축하고 품질을 향상시키는 데 기여할 것으로 기대됩니다.



### GIRAFE: Glottal Imaging Dataset for Advanced Segmentation, Analysis, and Facilitative Playbacks Evaluation (https://arxiv.org/abs/2412.15054)
Comments:
          18 pages, 8 figures

- **What's New**: GIRAFE는 성대의 고속 비디오 내시경 시퀀스에 대한 의미적 분할 준비가 된 공개 데이터셋이 부족한 문제를 해결하기 위해 설계된 데이터 저장소입니다. 이 저장소에는 50명의 환자로부터 수집된 65개의 고속 비디오 내시경 녹화본이 포함되어 있으며, 이는 건강한 대조군과 목소리 장애를 진단받은 환자, 건강 상태가 알려져 있지 않은 환자 각각의 데이터를 모두 포함합니다. 모든 데이터는 전문가에 의해 수동으로 주석이 달린 세그멘테이션 마스크를 포함하고 있습니다.

- **Technical Details**: GIRAFE 데이터셋은 4,000 fps의 샘플링 주파수에서 성대의 빠른 동적 움직임을 정확하게 포착하는 고속 비디오 내시경(HSV) 기술을 기반으로 합니다. 이 데이터셋은 성대의 의미적 분할 및 분석을 위한 자동 세그멘테이션 접근 방식을 혼합하여 제공하며, 다양한 state-of-the-art 방법으로 보강됩니다. 이 데이터는 성대 병리의 진단 및 치료 효과 평가에 중요한 역할을 할 수 있습니다.

- **Performance Highlights**: GIRAFE 데이터셋은 새로운 세그멘테이션 알고리즘 개발에 이미 사용되었고, 성대의 진동 패턴 분석에 있어 유용성을 입증하였습니다. 기존의 연구들은 FN(Normal and Pathological) 진동을 높은 정확도로 분류할 수 있는 컴퓨터 기반 접근 방식을 개발하여 이 데이터셋이 성대 질환 검출을 위한 중요한 자원임을 강조하고 있습니다. 그러나 성대 영역의 정확하고 완전한 자동 의미적 분할 방법을 구축하기 위한 과제가 여전히 남아있습니다.



### Uni-Renderer: Unifying Rendering and Inverse Rendering Via Dual Stream Diffusion (https://arxiv.org/abs/2412.15050)
- **What's New**: 본 논문에서는 렌더링(rendering)과 역 렌더링(inverse rendering)을 단일의 diffusion framework 내에서 조건부 생성 태스크로 모델링하여 두 과정을 상호 보완할 수 있는 새로운 방법인 Uni-Renderer를 제안합니다. 기존 방법들은 특정 씬에 대한 이상적인 추정값을 근사하는 데 그쳤으며 높은 계산 비용이 문제였습니다. 또한, 기존의 역 조건부 분포 전이는 본질적인 모호성으로 인해 다루기 어려웠습니다. Uni-Renderer는 두 가지 조건부 생성 태스크를 통합하여 일관성을 유지함으로써 모호성을 완화합니다.

- **Technical Details**: Proposed Uni-Renderer는 dual streaming module을 통해 두 개의 사전 학습된 diffusion 모델을 cross-conditioning하여 렌더링과 역 렌더링을 동시에 수행합니다. 이는 두 개의 개별 태스크(conditional generation tasks)를 단일 pipeline 내에서 다룰 수 있도록 합니다. 이로 인해 intrinsic properties와 렌더링된 이미지 간의 일관성을 유지하여 성능을 향상시키고, 3D 물체의 데이터 세트를 통해 효과적으로 의존성을 분해할 수 있습니다. 본 논문에서 언급되는 다양한 기술 키워드로는 'diffusion models', 'cycle-consistent constraints' 및 'dual-stream diffusion model'이 있습니다.

- **Performance Highlights**: 이 방법을 통해 얻어진 결과는 intrinsic properties의 강력한 분리 및 렌더링 변화 인식을 위한 뛰어난 능력을 보여줍니다. 다양한 실험을 통해 Uni-Renderer는 더욱 효과적인 렌더링과 역 렌더링을 가능하게 하여 포토리얼리스틱한 2D 이미지를 생성하는 데 기여합니다. 연구팀은 훈련 및 추론 코드 를 공개하여 향후 연구와 개발을 촉진할 예정입니다. 전반적으로 이 접근 방식은 기존의 기법들에 비해 높은 성능을 발휘하였으며, 다양한 응용 가능성을 보여주고 있습니다.



### DCTdiff: Intriguing Properties of Image Generative Modeling in the DCT Spac (https://arxiv.org/abs/2412.15032)
Comments:
          23 pages

- **What's New**: 이 논문은 주파수 공간에서 이미지 모델링을 탐구하고, Discrete Cosine Transform (DCT) 공간에서 이미지를 효율적으로 모델링하는 DCTdiff라는 엔드 투 엔드 확산 생성 패러다임을 소개합니다. DCTdiff는 이미지 생성 품질과 훈련 효율성 측면에서 픽셀 기반 확산 모델보다 우수함을 보여줍니다. 또한, 높은 해상도 생성에 원활하게 확장할 수 있으며, 손실 없는 압축으로 DCT가 낮은 계산 비용으로 가능하다는 점이 주목할 만합니다.

- **Technical Details**: DCTdiff는 주파수 공간에서의 생성 작업에 중점을두고 이미지 데이터 분포를 전체적으로 모델링합니다. 이 모델은 JPEG 압축 포맷에서 영감을 받아 저주파 신호를 먼저 생성하고 고주파 이미지 세부 정보를 생성하는 방식으로 작동합니다. 또한, 모델의 구조와 속성을 탐구하여 DCT 공간에서의 확산 모델링의 중요한 설계 요소를 발견했습니다.

- **Performance Highlights**: DCTdiff는 다양한 실험에 대한 결과에서 픽셀 기반 확산 모델을 초월하며, 이미지 생성에서 높은 품질을 나타냈습니다. 특히, 고해상도 이미지 생성에 있어서 기존의 방식보다 우수한 성능을 나타내며, 보통 사용되는 VAE를 필요로 하지 않는 점이 두드러집니다. DCT 기반 이미지 모델링은 다양한 작업에 적용할 수 있는 잠재력을 보여줍니다.



### Stitch Contrast and Segment_Learning a Human Action Segmentation Model Using Trimmed Skeleton Videos (https://arxiv.org/abs/2412.14988)
Comments:
          Accepted as AAAI 2025

- **What's New**: 이 논문에서는 기존의 뼈대 기반(skeleton-based) 인간 행동 분류 모델의 한계를 극복하기 위한 새로운 프레임워크를 제안합니다. 기존 모델은 잘 다듬어진 행동-specific 뼈대 비디오에 의존하여 실제 환경에서의 응용에 확장이 어려웠습니다. 본 연구의 방법은 짧게 잘라진(trimmed) 뼈대 비디오로 학습하지만, 긴 잘리지 않은(untrimmed) 비디오에서도 실행될 수 있습니다.

- **Technical Details**: 이 프레임워크는 Stitch, Contrast, Segment의 세 단계로 구현됩니다. Stitch 단계에서는 시간적 뼈대 스티칭(skeleton stitching) 방식을 제안하여 다듬어진 뼈대 비디오들을 기본 인간 동작으로 보고 이를 조합하여 다중 행동 스티칭 시퀀스를 생성합니다. Contrast 단계에서는 스티칭된 시퀀스로부터 대조적 표현을 학습하는 discriminative pretext task를 통해 뼈대 인코더가 의미 있는 행동-시간적(context) 맥락을 학습하도록 합니다.

- **Performance Highlights**: 실험은 다듬어진 소스(dataset) 데이터를 사용하고, 이를 기반으로 실제 상황에서의 뼈대 기반 인간 행동 세분화를 평가하기 위해 잘리지 않은(target dataset) 데이터를 활용합니다. 제안된 방법의 효과성을 검증하기 위한 적용(formulation) 방식이 포함되어 있으며, 이러한 접근 방식이 뼈대 행동 분류의 성능을 어떻게 개선하는지 보여줍니다.



### Arti-PG: A Toolbox for Procedurally Synthesizing Large-Scale and Diverse Articulated Objects with Rich Annotations (https://arxiv.org/abs/2412.14974)
- **What's New**: 새로운 접근 방식으로 제안된 Arti-PG 툴박스는 3D 관절 객체에 대한 풍부한 주석을 가지고 많은 양의 데이터를 빠르게 수집할 수 있도록 도와줍니다. 이 툴박스는 3D 객체를 일반화된 구조 프로그램으로 설명하고, 다양한 조작 규칙을 통해 새로운 객체를 합성합니다. Arti-PG는 이제 26개의 다양한 관절 객체 카테고리를 지원하며 여러 작업에 적용할 수 있는 포괄적인 주석을 제공합니다.

- **Technical Details**: Arti-PG는 세 가지 구성 요소로 이루어져 있습니다: i) 관절 객체의 구조 프로그램과 객체의 포인트 클라우드와의 상관관계, ii) 구조 프로그램 조작을 위한 절차적 규칙, iii) 주석을 위한 수학적 지식 설명입니다. 이 툴박스는 개체 생성시 무한한 형태 변화를 허용하는 프로그램 지향적 구조 조작 및 분석적 레이블 정렬을 통해 다채로운 데이터셋을 생성합니다.

- **Performance Highlights**: Arti-PG를 사용하여 3096개의 3D 관절 객체를 평가한 결과, 이 툴박스에서 합성된 객체와 주석이 높은 품질을 나타냈습니다. 다양한 시각 및 로봇 작업에 대한 exhaustive 실험을 통해 Arti-PG의 강점을 입증하며, 기존 데이터 증강 방법들과 비교하여 다양한 작업에 적용 가능한 점에서 차별화됩니다.



### PhotoHolmes: a Python library for forgery detection in digital images (https://arxiv.org/abs/2412.14969)
- **What's New**: 이 논문에서는 PhotoHolmes라는 오픈소스 Python 라이브러리를 소개합니다. 이는 디지털 이미지에서 변조 탐지 방법을 손쉽게 실행하고 벤치마크할 수 있도록 설계되었습니다. 라이브러리는 인기 있는 최신 방법의 구현, 데이터 세트 통합 도구 및 평가 메트릭을 포함하고 있으므로 사용자들이 비교를 쉽게 할 수 있게 도와줍니다.

- **Technical Details**: PhotoHolmes는 모듈화(modularity), 재현 가능성(reproducibility), 확장성(extensibility), 사용 용이성(usability)이라는 네 가지 주요 설계 원칙을 기반으로 구축되었습니다. 각각의 모듈은 이미지 변조 탐지 파이프라인의 특정 측면을 다루며, 사용자는 이를 Command Line Interface(CLI)를 통해 코드 작성 없이 쉽게 사용할 수 있습니다. 또한 Python과 PyTorch를 활용하여 깊이 있는 신경망(method) 구현을 지원합니다.

- **Performance Highlights**: PhotoHolmes는 사용자가 의심스러운 이미지를 테스트할 수 있는 간편한 방법을 제공하며, 다양한 메트릭(metrics)을 통해 방법의 성능을 평가할 수 있습니다. 모든 모듈은 독립적으로 사용할 수 있지만, 데이터 세트, 전처리, 방법 및 메트릭 모듈이 원활하게 통합되도록 설계되어 있어 사용자가 효율적으로 정보를 분석할 수 있도록 도와줍니다.



### Movie2Story: A framework for understanding videos and telling stories in the form of novel tex (https://arxiv.org/abs/2412.14965)
- **What's New**: M2S라는 프레임워크는 비디오와 오디오, 문자 인식을 결합하여 소설 수준의 텍스트를 생성하는 새로운 방법론을 소개합니다. 이 모델은 비디오의 긴 형식 텍스트 설명 및 이해, 오디오 감정 분석, 그리고 시각적 캐릭터 인식 모듈을 포함하여 다중 모달 정보를 통합합니다. M2S는 대규모 언어 모델인 GPT-4로 다중 모달 텍스트 생성을 차별화하며, 향후 연구에 대한 잠재력을 지니고 있습니다.

- **Technical Details**: M2S 모델은 여러 모듈로 구성되어 있으며, 비디오 및 오디오의 다중 모달 정보를 결합하여 소설 형식의 상세한 서사를 생성합니다. 커다란 음성 및 감정 기반의 분석을 사용하여, 비디오와 오디오의 내용 모두를학습할 수 있습니다. 또한 이 모델은 고객의 다양성과 요구를 충족하기 위해 매끄러운 스토리라인 및 심리적 묘사를 제공합니다.

- **Performance Highlights**: 실험 결과 M2S는 텍스트의 완전성 및 풍부한 의미 정보를 잘 생성하는 뛰어난 능력을 보여줍니다. 이 모델은 장애인 및 텍스트 서술에 의존하는 사용자에게 영화 내용을 보다 쉽게 이해하고 즐길 수 있는 기회를 제공합니다. 이러한 접근은 교육 및 오락 분야에서도 활용 가능성이 높아, 비디오 컨텐츠에 대한 자세한 설명 제공 및 영화의 소설화 작업에 기여할 수 있을 것입니다.



### IDOL: Instant Photorealistic 3D Human Creation from a Single Imag (https://arxiv.org/abs/2412.14963)
Comments:
          21 pages, 15 figures, includes main content, supplementary materials, and references

- **What's New**: 이 연구에서는 HuGe100K라는 대규모 데이터셋을 도입하여, 100,000개의 다양한 포즈와 의상을 가진 고해상도 이미지 세트를 생성합니다. 이 데이터셋은 고품질 인간 재구성을 위한 이상적인 기반이 됩니다. 또한 새로운 피드포워드 변환기 모델 IDOL을 개발하여 단일 이미지를 기반으로 3D 인간 아바타를 즉시 재구성하는 속도를 향상시킵니다.

- **Technical Details**: HuGe100K 데이터셋은 2.4M개의 고해상도 다중 뷰 이미지를 포함하며, 이는 다양한 인종, 성별, 나이, 체형을 반영합니다. IDOL 모델은 사전 훈련된 인코더와 변환기 백본을 활용하여 3D 인간을 예측하고, 이 과정에서 인체의 포즈, 신체 형태, 의상 기하학 및 텍스처를 분리합니다. 이러한 분리는 재구성을 더욱 효율적으로 만들어 줍니다.

- **Performance Highlights**: 본 연구의 모델은 단일 입력 이미지로부터 1K 해상도의 포토리얼리스틱 인간을 즉시 재구성할 수 있으며, 데이터셋과 방법의 효과를 검증하는 포괄적인 실험에서 뛰어난 성능을 보였습니다. IDOL 모델은 애니메이션 및 편집 작업을 지원하며, 고해상도 렌더링을 가능하게 하는 일관된 텍스처 완성을 보장합니다.



### TDCNet: Transparent Objects Depth Completion with CNN-Transformer Dual-Branch Parallel Network (https://arxiv.org/abs/2412.14961)
- **What's New**: 본 논문은 투명 객체에 대한 깊이 보완(depth completion) 작업을 위해 새로운 CNN-Transformer 병렬 네트워크인 TDCNet을 제안합니다. 기존 방법들은 원본 깊이 맵의 정보를 충분히 활용하지 못했으나, TDCNet은 원본 깊이 맵의 특징을 독립적으로 추출하여 이를 활용합니다. 이 모델은 RBG-D 이미지와 깊이 이미지를 각각 입력받아 두 개의 다른 가지에서 피처를 추출하여 토대로 깊이 정보를 보완합니다.

- **Technical Details**: TDCNet은 병렬 구조에서 CNN과 Transformer의 장점을 결합한 고유한 디자인을 특징으로 하며, 멀티스케일 피처 융합 모듈(MFFM)을 활용해 두 가지 가지의 피처를 잘 결합합니다. 이러한 구조는 깊이 맵 피처의 지역적 상관관계를 고려하여 CNN이 지역적 특징 추출에 강점을 갖는 것을 활용합니다. 추가적으로, 여러 손실 함수의 조합으로 인해 발생할 수 있는 그래디언트 충돌 문제를 완화하기 위해 훈련 전략을 설계했습니다.

- **Performance Highlights**: 실험 결과, TDCNet 모델은 여러 공공 데이터셋에서 기존의 최첨단 성능을 기록했습니다. 이 모델은 깊이 예측에서 우수한 정확도를 제공하며, 일반화 능력이 뛰어난 것으로 나타났습니다. 이러한 성과는 투명 객체의 깊이 정보를 보다 효과적으로 보완할 수 있음을 보여줍니다.



### Corn Ear Detection and Orientation Estimation Using Deep Learning (https://arxiv.org/abs/2412.14954)
Comments:
          22 pages;15 figures

- **What's New**: 이 연구는 옥수수 식물의 귀(ears) 성장을 모니터링하기 위한 새로운 컴퓨터 비전 기반 시스템을 제안합니다. 전통적으로 수동으로 측정하던 귀의 각도 측정에서 발생하는 시간 소모와 인간 오류 문제를 해결합니다. 이 시스템은 이미지 시퀀스에서 귀를 정확히 감지하고 추적할 수 있도록 설계되었습니다.

- **Technical Details**: 제안된 알고리즘은 객체 탐지기(object detector)와 키포인트(keypoint) 탐지 기술을 사용하여 모든 귀의 90%를 탐지할 수 있습니다. 귀의 방향을 추정하는 과정에서 평균 절대 오차(mean absolute error, MAE)가 18도였으며, 두 사람이 수동으로 측정했을 때 평균 15도 차이가 나는 것과 비교되었습니다. 이를 통해 컴퓨터 비전 기술을 옥수수 성장 모니터링에 적용할 수 있는 가능성을 보여줍니다.

- **Performance Highlights**: 이 시스템은 수동 측정에 비해 상당한 시간 절약을 가능하게 하며, 옥수수 생산을 위한 효율성을 높이는 동시에 귀 방향 연구의 추가 분야 탐구를 열어줍니다. 성과는 수치적으로도 뛰어나며, 이는 향후 이 분야에 대한 연구의 기초가 될 것으로 기대됩니다.



### GURecon: Learning Detailed 3D Geometric Uncertainties for Neural Surface Reconstruction (https://arxiv.org/abs/2412.14939)
Comments:
          Accepted by AAAI 2025. Project page: this https URL

- **What's New**: 이번 논문에서 소개하는 GURecon은 신경 표면 복원(neural surface reconstruction)을 위한 기하학적 불확실성(geometric uncertainty)을 평가하는 새로운 프레임워크입니다. 기존의 렌더링 기반 방법과 달리, GURecon은 여러 시점(viewpoint)에서 일관성을 유지하면서 3D 불확실성을 직접 모델링합니다. 이러한 방식을 통해 기존 방법에서의 시점 의존적 요인들을 제거하여 더욱 정확한 복원을 가능하게 합니다.

- **Technical Details**: GURecon의 접근법은 멀티뷰 일관성(multi-view consistency)을 활용해 기하학적 불확실성을 측정합니다. 제안된 방법은 조인트 분리(joint decoupling) 필드 학습을 통해 조명을 포함한 색상 관측의 불일치 문제를 해결하며, 동시에 시각적으로 연속성을 가진 불확실성 필드를 학습합니다. 이를 통해 모델은 다양한 신경 표면 표현(neural surface representations)에서 사용할 수 있도록 확장 가능합니다.

- **Performance Highlights**: 다양한 데이터셋에서 수행된 실험 결과, GURecon은 3D 기하학적 불확실성을 효과적으로 모델링하는 데 탁월한 성능을 보였고, 다양한 하위 작업에서의 품질 향상을 통해 실용성을 입증하였습니다. 특히, incremental reconstruction과 같은 다운스트림 작업에서 GURecon의 통합을 통해 더 나은 결과를 얻었다는 점이 강조되었습니다.



### Automatic Spectral Calibration of Hyperspectral Images:Method, Dataset and Benchmark (https://arxiv.org/abs/2412.14925)
- **What's New**: 이 논문에서는 전통적인 물리적 참조 없이 하이퍼스펙트럼 이미지(HSI)를 자동으로 보정하는 새로운 연구 주제를 제안합니다. BJTU-UVA 데이터 세트를 통해 첫 번째 HSI 자동 보정 데이터 세트를 생성하였으며, 765개의 고품질 HSI 쌍이 포함되어 있습니다. 또한, 스펙트럼 조명 변환기(SIT)와 조명 주의 모듈을 도입하여 이전 연구에 비해 더 나은 성능을 달성합니다.

- **Technical Details**: 하이퍼스펙트럼 이미지는 공간 및 주파수 도메인에서 밀집 샘플링을 통해 RGB 이미지보다 더 많은 정보를 제공합니다. 이 연구에서는 조명이 전역적으로 균일하다는 가정을 하여 HSI의 보정을 수행하며, BJTU-UVA 데이터 세트는 204개 대역을 가진 400-1000nm의 스펙트럼 카메라를 사용하여 수집되었습니다. 이 데이터 세트는 비슷한 장면에서 캘리브레이션된 및 캘리브레이션되지 않은 이미지 쌍을 포함하여 다양한 조명 조건을 포함합니다.

- **Performance Highlights**: 제안된 SIT는 현재까지의 HSI 보정 연구 중에서 최첨단 성능을 보여줍니다. 벤치마크 결과에 따르면, 낮은 조명 조건에서의 성능이 일반 조건보다 도전적인 것으로 나타났습니다. 본 연구는 향후 HSI 자동 보정 방법론에 대한 기초 자료로 유용할 것으로 기대됩니다.



### MagicNaming: Consistent Identity Generation by Finding a "Name Space" in T2I Diffusion Models (https://arxiv.org/abs/2412.14902)
Comments:
          Accepted by AAAI 2025

- **What's New**: 본 논문에서는 유명인사의 이름을 가지고 다양한 개체의 존재감을 생성할 수 있는 'Name Space'의 존재를 탐구합니다. 기존의 텍스트-이미지 생성 모델의 성능을 유지하면서도 새로운 개체의 명칭을 예측할 수 있는 방법을 제안합니다. 실험적으로 확인된 결과, 이름 임베딩(name embeddings)은 입력 텍스트의 의미와는 분리되어 작동함을 보여줍니다.

- **Technical Details**: 저자들은 Laion5B 데이터셋에서 유명인 이름과 이미지의 쌍을 수집하고, 텍스트 인코더(text encoder)를 사용하여 이름 임베딩을 생성합니다. 이를 통해 각 개인의 얼굴 이미지를 기반으로 '이름'을 예측하는 이미지 인코더(image encoder)를 학습시켰습니다. 이 과정에서 CLIP 모델을 활용하여 텍스트 임베딩 공간과의 정렬을 강화하고, 최종 예측을 위한 완전연결망(trilayer fully connected network)을 설계하여 사용합니다.

- **Performance Highlights**: 이 연구의 결과로, 제안된 이미지 인코더는 특정 개체에 대한 이름 임베딩을 찾아내어, 그 임베딩을 텍스트 가이드에 쉽게 통합함으로써 일관된 개인 아이덴티티 생성이 가능합니다. 추가적으로, ID 보간(interpolation) 기법을 통해 허구의 캐릭터 생성도 가능하다는 점이 성과로 부각됩니다. 기존 연구들과 달리, 본 방법은 모델 추가 조건을 도입하지 않으면서도 원래의 생성 능력을 손상 없이 유지합니다.



### Multimodal Hypothetical Summary for Retrieval-based Multi-image Question Answering (https://arxiv.org/abs/2412.14880)
Comments:
          AAAI 2025

- **What's New**: 본 연구에서는 멀티이미지 질문 응답(Multi-image Question Answering) 문제에 대한 새로운 접근 방식을 제안합니다. 기존의 "retrieve-then-answer" 방식에서 발생하는 오류를 줄이기 위해, 우리는 다중 모달 가상의 요약(MHyS, Multimodal Hypothetical Summary)을 도입하여 검색 과정과 QA를 효율적으로 연결합니다. 이렇게 함으로써, 이미지 검색과 질문 응답 과정에서의 목표를 최적화하고 성능을 개선할 수 있습니다.

- **Technical Details**: 우리의 방법은 멀티모달 대형 언어 모델(multimodal large language model)과 대형 언어 모델(large language model)을 결합하여 질문형 및 설명형 가상의 요약을 생성합니다. 또한, 대조 학습(contrastive learning)을 활용하여 쿼리와 MHyS를 정렬하고, 문장 수준 및 단어 수준 유사성을 계산하기 위해 조잡 코스를 사용한 전략을 채택합니다. 이러한 방법은 텍스트 대 텍스트 검색(text-to-text retrieval)을 통해 정보 검색을 더욱 효율적으로 만듭니다.

- **Performance Highlights**: 우리의 접근 방식은 RETVQA 데이터셋에서 최신 방법들과 비교하여 3.7%의 절대적인 성능 향상을 기록하였으며, CLIP에 대해서는 14.5% 향상을 보였습니다. 포괄적인 실험과 분리 연구를 통해 우리의 방법이 타 방법에 비해 우수성을 입증하였음을 확인 할 수 있었습니다. 특히, MHyS가 핵심 객체 단어를 효과적으로 포착하고 더 구체적인 정보를 제공함으로써 검색과 QA의 성공적인 통합을 가능하게 합니다.



### Zero-Shot Artifact2Artifact: Self-incentive artifact removal for photoacoustic imaging without any data (https://arxiv.org/abs/2412.14873)
- **What's New**: 이번 논문은 Photoacoustic Imaging (PAI)의 3D 재구성 시 발생하는 아티팩트 문제를 해결하기 위해 Zero-Shot Artifact2Artifact (ZS-A2A)라는 새로운 방법을 제안합니다. 이 방법은 기존의 iterative reconstruction이나 deep learning 기반 접근 방식에 비해 훈련 데이터나 사전 지식 없이 작동하여 아티팩트를 효과적으로 제거할 수 있습니다. ZS-A2A는 고유한 기법을 통해 랜덤한 변화를 주어 데이터의 하위 집합을 생성하고, 이를 통해 네트워크가 아티팩트 패턴을 학습하도록 유도합니다.

- **Technical Details**: ZS-A2A는 매우 경량의 네트워크를 기반으로 하는 self-supervised 방법으로, PAI 재구성 아티팩트의 물리 모델에 맞춤화되었습니다. 입력 데이터에 무작위 변화(random perturbations)를 적용하여 다운 샘플링된 재구성 결과의 쌍을 생성하고, 잔여(residual) 및 일관성(consistency) 손실을 통해 네트워크를 최적화합니다. 이 방식은 PAI 재구성 결과를 위한 아티팩트 제거를 매우 효율적으로 수행할 수 있게 해줍니다.

- **Performance Highlights**: 시뮬레이션 및 실험동물 연구에서 ZS-A2A의 성능이 검증되었습니다. 특히, in vivo에서 쥐의 간에 대해 ZS-A2A는 CNR(Contrast-to-Noise Ratio)을 17.48에서 43.46으로 개선하며 단 8초 만에 우수한 아티팩트 제거 능력을 보여주었습니다. 이는 기존의 zero-shot 방법들과 비교할 때 선두적인(state-of-the-art) 성과로 평가됩니다.



### Large-scale School Mapping using Weakly Supervised Deep Learning for Universal School Connectivity (https://arxiv.org/abs/2412.14870)
Comments:
          Accepted at AAAI-25 Special Track on AI for Social Impact (AISI)

- **What's New**: 이 논문은 저렴하고 확장 가능한 방법으로 고해상도 위성 이미지를 사용하여 학교의 정확한 위치 정보를 추정하는 딥 러닝 기술을 제안합니다. 이 방법은 약한 감독 학습(weakly supervised learning)을 통해 GPS 좌표가 누락된 개발도상국에서 학교 연결의 비용을 보다 정확하게 추정하는 데 도움을 줍니다. 특히, 비전 트랜스포머와 합성곱 신경망(CNN)을 결합한 최상의 모델은 아프리카 10개 국가에서 AUPRC 값이 0.96을 초과하는 성능을 보여줍니다.

- **Technical Details**: 이 연구는 다양한 공공 데이터 소스(OpenStreetMap, Overture Maps, GigaMaps)에서 정보를 통합하여 국가 수준의 학교 매핑 데이터셋을 생성하는 파이프라인을 개발했습니다. 활용된 딥 러닝 모델은 비전 트랜스포머(ViTs)와 CNN을 결합하여, 위성 이미지로부터 학교를 분류하는 데 성공적으로 응용되었으며, 모델 성능의 차이를 도시와 농촌의 하위 지역 간에 비교했습니다. 또한, 설명 가능한 AI(XAI) 기술을 통해 이미지 내 학교 위치를 더욱 세밀하게 로컬라이즈했습니다.

- **Performance Highlights**: 이 연구는 아프리카 여러 국가에 대한 학교 위치의 전국적 지도를 생성함으로써 접근 방식의 실현 가능성과 확장성을 입증하였고, 특히 세네갈을 사례 연구로 자세히 분석하였습니다. 정부 파트너와의 협력으로 학습 소프트웨어가 상호작용 가능한 웹 매핑 도구를 통해 모델 검증 작업을 원활하게 수행하도록 지원합니다. 이 연구 결과는 딥 러닝과 위성 이미지가 지역 인프라 계획 및 보편적인 학교 연결 촉진에 얼마나 유용한지를 효과적으로 보여줍니다.



### AI-Powered Intracranial Hemorrhage Detection: A Co-Scale Convolutional Attention Model with Uncertainty-Based Fuzzy Integral Operator and Feature Screening (https://arxiv.org/abs/2412.14869)
- **What's New**: 이 연구는 두 개의 이진 분류 문제로 ICH의 발생 여부 및 SDH의 유형을 감지하는 새로운 접근 방식을 제안합니다. 특히, CCA(classified co-scale convolutional attention) 분류기 아키텍처에 두 개의 레이어를 추가하여 ICH 감지를 위한 혁신적인 방법을 도입하였습니다. 이 방법은 CT 스캔 이미지에서 추출된 특징을 결합하고 높은 변동성을 포착하는 50개의 구성 요소를 선택하여 설명 가능한 AI 모델 개발에 기여합니다.

- **Technical Details**: 첫 번째 레이어에서는 CT 스캔의 다양한 슬라이스에서 특징을 추출한 후, 부스트랩 포레스트 알고리즘을 사용하여 최종 예측에 대한 각 특징의 기여도를 명확히 평가합니다. 두 번째 레이어에서는 새로운 불확실성 기반 퍼지 적분 연산자를 도입하여 서로 다른 CT 스캔 슬라이스의 정보를 융합하고, 이는 검사 정확도를 크게 향상시킵니다. 이를 통해 CT 스캔 슬라이스 간의 의존성을 고려하여 감지 정확도를 높입니다.

- **Performance Highlights**: 이 연구는 이란 테헤란의 두 개 의료 센터에서 수집된 대규모 CT 스캔 이미지를 사용하여 ICH와 SDH를 탐지하는 데 초점을 맞추었습니다. 새로운 접근 방식은 기존 방법보다 더 강력하고 신뢰할 수 있는 검출 모델을 개발하는 데 기여하며, 이미지 전처리 및 특징 융합 과정에서 불필요한 특징을 제거하여 자원 소모를 줄이는 데 효과적입니다. 이 연구는 AI 모델의 해석 가능성을 높이는 데 중요한 진전을 보여줍니다.



### ObjVariantEnsemble: Advancing Point Cloud LLM Evaluation in Challenging Scenes with Subtly Distinguished Objects (https://arxiv.org/abs/2412.14837)
Comments:
          Accepted to AAAI2025

- **What's New**: 최근 3D 장면 이해와 관련된 연구가 증가하며, 점군(point clouds)과 텍스트의 정렬이 중요한 과제로 부각되고 있습니다. 본 논문에서는 ObjVariantEnsemble이라는 새로운 벤치마크를 제안하여, 다양한 객체 클래스, 색상, 형태, 수량 및 공간 관계를 가진 장면을 보다 체계적으로 구축하고 평가하는 방법을 소개합니다. 이를 통해 3D 모델의 이해 능력을 평가하고 향후 발전에 기여할 수 있는 기회를 제공합니다.

- **Technical Details**: ObjVariantEnsemble(OVE)는 3D 데이터셋을 확장하고, 75,000개의 새로운 장면과 세분화된 주석들을 생성하는 효과적인 3D 장면 구성 및 주석 프레임워크를 기반으로 합니다. LLM(대형 언어 모델)과 VLM(비전-언어 모델)을 통합하여 목표와 주의 산만 요소 간의 차이를 포착하는 자동화된 주석 파이프라인을 구축하였습니다. 이 기술적 접근방식은 모델의 강점과 약점을 객관적으로 평가할 수 있는 더 풍부한 주석을 제공합니다.

- **Performance Highlights**: 이 새로운 벤치마크는 기존 3D 이해 모델의 한계를 드러내며, 순수한 공간 관계 추론에서의 성능 부족을 분석합니다. OVE 벤치마크를 통해 평가된 최신 3D 이해 모델들은 비주얼 특성 없이 공간 관계를 처리하는 데 어려움을 겪고 있음을 보여줍니다. 이러한 결과는 향후 모델의 개선 방향을 제시하고, 3D 모델 개발의 필요성을 더욱 부각시키고 있습니다.



### Synchronized and Fine-Grained Head for Skeleton-Based Ambiguous Action Recognition (https://arxiv.org/abs/2412.14833)
Comments:
          20pages, 5 figures

- **What's New**: 최근 GCN(그래프 합성곱 신경망)을 활용한 Skeleton 기반의 행동 인식에서, 'Handshaking'과 'Saluting'과 같은 애매한 행동 인식이 여전히 큰 도전 과제로 남아 있습니다. 기존 방법들은 GCN과 TCN(시계열 합성곱 신경망)의 직렬 조합에 의존하여 공간적(spatial) 및 시간적(temporal) 특성을 독립적으로 추출하는 경향이 있으며, 이로 인해 불균형한 정보 처리 문제가 발생합니다. 이를 해결하기 위해 SF-Head라는 경량 모듈을 제안합니다.

- **Technical Details**: SF-Head는 Synchronized Spatial-Temporal Extraction(SSTE)와 Adaptive Cross-dimensional Feature Aggregation(AC-FA)으로 구성되며, F-RL(Feature Redundancy Loss)을 사용하여 이 두 가지 특성 간의 균형을 유지합니다. 또한, F-CL(Feature Consistency Loss)을 통해 집합된 특징들을 원본 공간-시간 특성과 조정하여, 지식을 덜 잃지 않도록 합니다. 이러한 방식은 로컬 세부 정보와 글로벌 문맥을 효과적으로 결합할 수 있게 해줍니다.

- **Performance Highlights**: NTU RGB+D 60, NTU RGB+D 120, NW-UCLA와 같은 여러 벤치마크 데이터셋에서 실험을 진행한 결과, SF-Head를 사용한 경우 애매한 행동을 구별하는 분류 정확도가 현저히 향상되었습니다. SF-Head는 0.01M 미만의 매개변수를 가지며, 다양한 GCN 기반 아키텍처에 통합될 수 있는 매우 효율적인 솔루션입니다. 이 연구는 애매한 행동 인식 능력을 향상시키는 데 기여하고 있습니다.



### PC-BEV: An Efficient Polar-Cartesian BEV Fusion Framework for LiDAR Semantic Segmentation (https://arxiv.org/abs/2412.14821)
Comments:
          AAAI 2025

- **What's New**: 본 논문은 효율적인 LiDAR 분할을 위한 Polar-Cartesian BEV Fusion Network를 제안합니다. 기존의 복수 시점에서의 특성 융합 방식 대신, Bird's-Eye View(BEV) 내에서 두 가지 구간인 Polar와 Cartesian의 고정적인 상관관계를 활용한 새로운 접근법을 취합니다. 이 방법은 계산 비용이 높은 점 기반 메소드에 비해 약 170배 더 빠른 성능을 보여줍니다.

- **Technical Details**: 제안된 방법은 Polar와 Cartesian 파티셔닝의 고정된 그리드 쌍 correspondences를 활용하여 BEV 공간 내에서 직접적으로 특성을 융합합니다. 효율적인 remap 기반 특성 융합 방법을 도입하여 Polar와 Cartesian 사이의 융합을 효과적으로 수행하며, 이로 인해 더 풍부한 맥락 정보를 유지합니다. 추가로, Transformer-CNN 하이브리드 아키텍처를 사용하여 BEV 특성 추출을 용이하게 하며, 전역적 장면 정보를 캡처할 수 있습니다.

- **Performance Highlights**: SemanticKITTI 및 nuScenes 데이터셋에서 광범위한 평가를 통해, 제안된 방법이 이전의 다중 뷰 융합 접근 방식에 비해 성능과 추론 속도 모두에서 우수한 결과를 나타냄을 입증했습니다. 이는 BEV 기반의 특성 융합이 LiDAR 분할에 있어 큰 가능성을 가지고 있음을 강조합니다. 또한, 세부적인 ablation 연구 및 제안된 프레임워크의 각 모듈에 대한 심도 있는 논의가 포함되어 있습니다.



### Multi-Level Embedding and Alignment Network with Consistency and Invariance Learning for Cross-View Geo-Localization (https://arxiv.org/abs/2412.14819)
- **What's New**: 이 논문에서는 Cross-View Geo-Localization (CVGL) 문제를 해결하기 위해 Multi-Level Embedding and Alignment Network (MEAN)이라는 경량화된 신경망을 제안합니다. MEAN은 진행형 다중 수준 향상 전략, 전역-지역 연관성 및 교차 도메인 정렬을 활용하여 다양한 수준의 특성을 효과적으로 연결합니다. 이 네트워크는 특징 표현의 일관성과 불변성을 향상시키는 구조로 설계되어 있으며, 특히 드론 이미지와 위성 이미지를 비교하는 작업에 최적화되어 있습니다.

- **Technical Details**: MEAN 네트워크는 ConvNeXt-Tiny를 백본 네트워크로 사용하여 coarse-grained 특성을 추출합니다. 이 후, 세 가지 Branch가 각각의 기능 추출 작업을 수행하며, 이를 통해 모델은 다양한 특성 표현을 학습할 수 있습니다. 또한, MEAN은 컨트라스티브 손실(contrastive loss)을 사용하여 공유되는 임베딩의 일관성과 판별 가능성을 향상시키며, 전역-지역 특성 표현의 공동 최적화를 통해 보다 정밀한  특징 정렬(fusion)을 제공합니다.

- **Performance Highlights**: 실험 결과 MEAN 네트워크는 기존의 최첨단 모델에 비해 파라미터 수를 62.17% 줄이고 계산 복잡도를 70.99% 감소시켰습니다. 이러한 성능 개선은 특히 자원이 제한된 항공 환경에서의 배치 시 필요한 계산 비용을 줄이는 데 크게 기여합니다. MEAN은 정밀성 및 효율성 모두에서 경쟁력 있는 성과를 보여주며, 향후 코드를 공개할 예정입니다.



### Explainable Tampered Text Detection via Multimodal Large Models (https://arxiv.org/abs/2412.14816)
Comments:
          The first work for explainable tampered text detection

- **What's New**: 이번 연구에서는 텍스트 변조 탐지(tampered text detection)의 신뢰성을 높이기 위해 자연어(Natural Language)를 이용한 설명 가능성을 추가했습니다. 기존의 방법들이 탐지된 영역을 제시했으나 그 해석이 명확하지 않아 신뢰성이 낮다는 문제를 해결하고자 합니다. 연구팀은 대규모 멀티모달 모델을 활용하여 이러한 탐지의 근거를 설명하는 새로운 접근을 제안했습니다.

- **Technical Details**: 연구는 ETTD라는 대규모 데이터셋을 구성했으며, 이는 픽셀 레벨 주석과 자연어 주석으로 변조 영역과 그 이상을 설명합니다. 특히, GPT4o를 통한 설명 생성을 위해서 픽셀단위 가중치가 결합된 바이너리 마스크(binaries mask)를 사용하여 탐지 영역을 명확히 표시하는 방식으로 혼란을 줄이는 방법을 제안하였습니다. 또한, TTD라는 모델을 도입하여 의심되는 지역을 중점적으로 관찰하도록 구조화함으로써 모델의 감지 능력을 개선했습니다.

- **Performance Highlights**: 제안된 TTD 모델은 ETTD 데이터셋과 공개된 Tampered IC-13 데이터셋에서 폭넓은 실험 결과를 통해 기존 방법들보다 월등한 성능을 보였습니다. 연구팀은 강력한 도메인 내 및 교차 도메인 일반화 능력을 입증했으며, 이로 인해 향후 연구에 유용한 통찰력과 전략을 제공했습니다. 향후 ETTD 데이터셋과 코드가 공개될 예정이어서 관련 연구자들에게 큰 도움이 될 것으로 기대됩니다.



### Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations (https://arxiv.org/abs/2412.14803)
Comments:
          The first two authors contribute equally. Project Page at this https URL

- **What's New**: 이 논문은 일반적인 로봇 정책을 개발하기 위한 새로운 접근 방식, Video Prediction Policy (VPP)를 제안합니다. 이전의 비전 인코더가 시퀀스 정보를 제대로 캡처하지 못하는 문제를 해결하기 위해 비디오 확산 모델(VDM)의 예측 시각 표현을 활용합니다. 이 연구는 VDM이 물리적 세계의 변화를 반영하는 고유한 정보를 포함하고 있다는 가설을 기반으로 합니다.

- **Technical Details**: VPP는 두 단계 학습 과정을 통해 개발됩니다. 첫 번째 단계에서는 조작 데이터 세트를 사용하여 텍스트 기반 비디오 예측 모델을 미세 조정하여 조작 도메인에서 예측 기능을 강화합니다. 두 번째 단계에서는 이 모델의 예측 표현에 조건부로 설정된 다중 작업 일반 로봇 정책을 개발하며, 이러한 예측 표현은 비디오 포머를 사용하여 필수 정보를 추출합니다.

- **Performance Highlights**: VPP는 두 개의 시뮬레이션 및 두 개의 실제 환경에서 기존 방법들보다 꾸준히 우수한 성능을 보였습니다. 칼빈 ABC-D 벤치마크에서는 28.1%의 개선을 이루었고, 복잡한 실제 조작 작업에서 성공률이 28.8% 증가하는 결과를 얻었습니다. 이러한 결과는 VPP의 높은 효율성과 일반화 능력을 입증합니다.



### YOLOv11 Optimization for Efficient Resource Utilization (https://arxiv.org/abs/2412.14790)
Comments:
          12 pages, 13 figures, 4 tables

- **What's New**: 이 연구의 목표는 You Only Look Once (YOLOv11)의 11번째 버전을 최적화하여, 특정 크기에 맞춘 수정된 아키텍처를 개발하는 것입니다. 이 수정 작업은 불필요한 레이어를 제거하고 YOLOv11의 주 아키텍처를 재구성하는 것을 포함합니다. 각 제안된 버전은 작은 크기부터 큰 크기까지 특정 크기 범위의 객체를 탐지하는 데 맞춰져 있습니다.

- **Technical Details**: YOLOv11 모델은 Backbone, Neck, Head 세 가지 주요 구성 요소로 이루어져 있습니다. Backbone은 입력 이미지에서 다양한 스케일에서 주요 특징을 추출하며, 다양한 Convolutional (Conv) 블록으로 구성되어 있습니다. Neck은 여러 Conv 레이어와 C3K2 블록을 포함하고, Head는 객체 클래스를 결정하고 객체성 점수를 계산하여 객체의 경계 상자를 정확하게 예측하는 역할을 합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델들은 원래 YOLOv11에 비해 계산 자원 효율성이 크게 향상되었으며, 정확도는 유지하고 있습니다. 경우에 따라 수정된 버전들은 탐지 성능 면에서 원래 모델보다 뛰어난 결과를 나타냈습니다. 또한 제안된 모델들은 모델 크기가 줄어들고 추론 시간이 빨라지는 특징을 보였습니다.



### FLAMe: Federated Learning with Attention Mechanism using Spatio-Temporal Keypoint Transformers for Pedestrian Fall Detection in Smart Cities (https://arxiv.org/abs/2412.14768)
Comments:
          8 pages, 7 figures, AAAI 2025 FLUID Workshop

- **What's New**: 이 연구에서는 스마트 도시에서 보행자 넘어짐을 효과적으로 감지하기 위한 새로운 시스템인 FLAMe (Federated Learning with Attention Mechanism)를 제안합니다. FLAMe는 중요한 키포인트 정보 주위에서 학습하고 중요한 가중치만을 서버에 전송하여 통신 비용을 줄이고 데이터 프라이버시를 보호합니다. 22672개의 비디오 샘플을 이용한 실험을 통해 FLAMe 기반 시스템은 94.02%의 정확도를 달성했습니다.

- **Technical Details**: FLAMe는 주목 (attention) 메커니즘을 활용하여 키포인트 정보를 효과적으로 집계하고, 지역 장치에서 데이터 프라이버시를 유지하면서 중앙 서버와 모델 매개변수만 공유합니다. 각 CCTV는 클라이언트 역할을 하여 비디오 데이터를 처리하고 사람의 키포인트를 추출하며, 키포인트 트랜스포머 모델을 통해 스페이셜-템포럴 특징을 학습합니다. 최종적으로 중요 키포인트 매개변수만 중앙 서버에 전송하여 글로벌 모델을 생성합니다.

- **Performance Highlights**: FLAMe 알고리즘은 기존의 중앙 집중식 학습 방식과 유사한 성능을 유지하면서도 통신 비용을 약 40% 감소시켰습니다. 이는 스마트 도시의 분산 환경에서 공공 안전을 위한 실용적이고 효과적인 솔루션으로 기능할 수 있음을 보여줍니다. 제안된 시스템은 다양한 CCTV 조건에 맞춰 효과적으로 학습하며, 메모리와 계산 자원 소모를 크게 줄여 경량화된 솔루션을 제공합니다.



### Prototypical Calibrating Ambiguous Samples for Micro-Action Recognition (https://arxiv.org/abs/2412.14719)
Comments:
          Accepted by AAAI 2025

- **What's New**: 미세 동작 인식(Micro-Action Recognition, MAR)은 비언어적 소통의 중요한 형태로 최근 주목받고 있습니다. 그러나 기존 접근 방식은 미세 동작의 본질적인 모호성을 간과하고 있어 정확도에 문제를 일으키고 있습니다. 본 논문은 Prototypical Calibrating Ambiguous Network(PCAN)을 제안하여 MAR의 모호성을 완화하고 인식 정확도를 높이는 데 중점을 두었습니다.

- **Technical Details**: PCAN은 계층적 액션 트리를 활용하여 미세 동작을 바디 수준과 액션 수준에서 식별합니다. 이를 통해 잘못 분류된 샘플인 false negatives ($\mathbb{FN}$)와 false positives ($\mathbb{FP}$)을 구분하고, 이들 간의 거리를 조정하는 모듈을 구현합니다. 특히, 프로토타입 다양성 강화 손실(prototypical diversity amplification loss)을 통해 모델의 성능을 증대시키고, 프로토타입 지향 정정을 통해 예측 결과를 개선합니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋을 통해 수행한 실험에서 제안된 방법이 기존 방식에 비해 뛰어난 성능을 보였습니다. 특히, 불확실성을 줄이고 분류 정확도를 높이는 데 기여하는 프로토타입 기반의 접근 방식이 유의미한 결과를 나타냈습니다. 이를 통해 MAR의 새로운 가능성을 제시하며 차별화된 기여를 하고 있습니다.



### EnergyMoGen: Compositional Human Motion Generation with Energy-Based Diffusion Model in Latent Spac (https://arxiv.org/abs/2412.14706)
Comments:
          Project page: this https URL

- **What's New**: 이번 연구에서는 EnergyMoGen이라는 새로운 방법을 제안하여 다중 의미 개념을 하나의 일관된 동작 시퀀스로 효과적으로 구성하는 문제를 해결하고자 하였습니다. 이 방법은 두 가지 에너지 기반 모델물인 잠재 인식(latent-aware) 에너지 기반 모델과 의미 인식(semantic-aware) 에너지 모델을 활용합니다. EnergyMoGen은 자동 인코더(autoencoder) 내에서 교차 주의(cross-attention)를 통합하여 명확히 텍스트 설명에 맞춰 복잡한 동작을 합성할 수 있도록 합니다.

- **Technical Details**: EnergyMoGen은 잠재 인식 에너지 기반 모델과 의미 인식 에너지 모델을 사용하여 동작 구성 과정을 에너지 조합 문제로 정의합니다. 경량화된 상태에서 동작을 생성하고 복잡한 동작을 단순 구성 요소로부터 조합하는 두 가지 운영 방식인 결합(conjunction) 및 부정(negation)을 통해 새로운 동작을 생성하는 방법을 제시합니다. 또한, Synergistic Energy Fusion(상보적 에너지 융합) 전략을 도입하여 서로의 한계를 극복하고 성능을 향상시키고자 합니다.

- **Performance Highlights**: 실험 결과, EnergyMoGen은 텍스트-모션 생성, 구성 동작 생성 및 다중 개념 동작 생성 작업을 포함한 다양한 동작 생성 작업에서 기존 최첨단 모델들을 초월하는 성능을 보여주었습니다. 세 개의 데이터셋에 대한 포괄적인 실험을 통해 EnergyMoGen의 효과성을 증명하였으며, 5000개의 Compositionally generated motion sequences가 포함된 새로운 CompML 데이터셋을 소개하였습니다. 또한, 모델 성능이 합성된 동작을 통해 개선될 수 있음을 입증하였습니다.



### Event-assisted 12-stop HDR Imaging of Dynamic Scen (https://arxiv.org/abs/2412.14705)
Comments:
          Project page: this https URL

- **What's New**: 이 연구는 동적 장면에서 12중 HDR 이미징을 위한 새로운 접근 방식을 제안합니다. 이 방법은 RGB 카메라와 이벤트 카메라로 구성된 이중 카메라 시스템을 활용하여 다양한 노출 차이를 효율적으로 정렬합니다. 또한, 실제 세계에서의 정렬 모듈의 일반화를 위해 실세계 피세팅( fine-tuning) 전략을 도입하였습니다.

- **Technical Details**: 제안된 시스템은 대조적 장면에서 아티팩트를 줄이고 정렬 과정에서의 오류를 최소화하기 위해 미리 훈련된 확산 모델의 이미지 원칙을 통합하는 확산 기반 융합 모듈을 개발했습니다. 이벤트 카메라는 고emporal한 밀도와 높은 동적 범위를 제공하여 기존 방법보다 훨씬 더 높은 노출 차이(12 스탑)를 처리할 수 있는 가능성을 제공합니다.

- **Performance Highlights**: 광범위한 실험 결과, 제안된 방법이 동적 장면에서 HDR 이미징을 12 스탑으로 확장하며 최신 기술 성능(state-of-the-art performance)을 달성하는 것으로 나타났습니다. ESHDR 데이터셋을 통해 단순히 시뮬레이션 데이터뿐만 아니라 실제 데이터에서도 성능 검증을 통해 실용성과 효과를 입증했습니다.



### Explicit Relational Reasoning Network for Scene Text Detection (https://arxiv.org/abs/2412.14692)
Comments:
          Accepted to AAAI 2025

- **What's New**: 이 논문에서는 기존의 연결 구성 요소(Connected Component, CC) 기반 텍스트 검출 방법의 단점을 보완하기 위해, 명시적 관계 추론 네트워크(Explicit Relational Reasoning Network, ERRNet)를 제안합니다. ERRNet는 텍스트 구성 요소 간의 관계를 모델링하여 복잡한 후처리 과정을 완전히 제거합니다. 이 접근법은 텍스트 검출을 추적 문제로 재구성하고, 이를 통해 보다 효율적이고 정확한 텍스트 인식을 가능하게 합니다.

- **Technical Details**: ERRNet은 각 텍스트 인스턴스를 여러 개의 순서가 있는 텍스트 구성 요소로 나타내고, 이 구성 요소들을 순차적 이동을 수행하는 객체로 취급합니다. 이 방식으로 텍스트 인스턴스 간의 관계를 모델링하고, 이를 위한 쌍방향 그래프 매칭을 사용하여 각 인스턴스의 실제 구성 요소 시퀀스와 예측 결과를 비교합니다. 추가적으로, 위치-supervised classification loss를 도입하여 분류 신뢰도와 위치 정확성을 조화롭게 조정합니다.

- **Performance Highlights**: 실험 결과, ERRNet은 기존 다수의 벤치마크에서 최고 정확도를 기록하며, 동시에 매우 경쟁력 있는 추론 속도를 유지합니다. 이 네트워크는 CC 기반 텍스트 검출 방법 중에서도 더욱 간단하고 빠르며, 설계된 기능이 효율적인 텍스트 인식을 가능하게 함을 입증했습니다.



### A Light-Weight Framework for Open-Set Object Detection with Decoupled Feature Alignment in Joint Spac (https://arxiv.org/abs/2412.14680)
- **What's New**: 이 논문은 Open-set object detection (OSOD) 분야에서 로봇 시스템의 실시간 작업을 지원하는 경량화된 프레임워크인 Decoupled OSOD (DOSOD)를 제안합니다. DOSOD는 YOLO-World 파이프라인을 기반으로 하여 vision-language model (VLM)과 탐지기를 통합합니다. 이 프레임워크는 고급 기능 상호작용을 피하면서 계산 효율성을 개선하여, 알려지지 않은 객체 탐지의 필요성을 해결합니다.

- **Technical Details**: DOSOD는 Multilayer Perceptron (MLP) 어댑터를 사용하여 VLM으로부터 추출한 텍스트 임베딩을 변환하고, 이를 통해 클래스 비정보 제안의 영역 표현을 학습합니다. 이러한 접근 방식은 서로 다른 모드의 기능들이 결합되는 것을 최소화하여 테스트 단계에서 전통적인 폐쇄형 탐지기와 유사한 방식으로 작동하게 만듭니다. 최종적으로, DOSOD는 조인트 스페이스에서 다양한 모드의 특성들을 직접적으로 정렬하여 컴퓨팅 비용과 저장 요구 사항을 감소시킵니다.

- **Performance Highlights**: DOSOD는 YOLO-World와 비교하여 실시간 성능을 크게 향상시키면서도 유사한 정확도를 유지합니다. DOSOD-S 모델은 LVIS minival 데이터셋에서 고정 AP(Fixed AP) 26.7%를 달성하였으며, 이는 YOLO-World-v1-S의 26.2% 및 YOLO-World-v2-S의 22.7%보다 높은 수치입니다. 또한 DOSOD-S는 YOLO-World-v1-S보다 57.1% 높은 FPS, YOLO-World-v2-S보다 29.6% 높은 FPS를 기록하며 경량화된 배포가 가능합니다.



### Efficient Few-Shot Neural Architecture Search by Counting the Number of Nonlinear Functions (https://arxiv.org/abs/2412.14678)
Comments:
          Accepted to AAAI 2025

- **What's New**: 본 논문에서는 효과적인 few-shot NAS 방법을 소개합니다. 이 방법은 search space(탐색 공간)를 비선형 함수의 수에 따라 분할하여 각 subspace(서브공간)에 효과적인 supernet(슈퍼넷)을 할당하여 상충을 최소화합니다. 이를 통해 훈련의 효율성을 높이고, computation cost(계산 비용)를 줄이는 데 기여합니다.

- **Technical Details**: 제안된 방법은 비선형 함수의 수를 기준으로 서브공간을 분할하며, 각 서브공간은 동일한 비선형 함수 수를 가진 subnets(서브넷)으로 구성됩니다. 이렇게 함으로써 동일한 supernet에 속하는 서브넷 간에는 유사한 특성을 가지게 되어 서로 간섭을 줄일 수 있습니다. 또한 supernet-balanced sampling(SBS) 기법을 도입하여 한정된 학습 단계 내에서 여러 supernets을 고르게 훈련할 수 있도록 하였습니다.

- **Performance Highlights**: 실험 결과, NAS201과 ImageNet 데이터셋에서 제안한 방법이 기존의 few-shot 방법들보다 훨씬 적은 계산 자원으로 state-of-the-art 성과를 달성하였습니다. 이는 적은 자원으로도 높은 정확도를 얻을 수 있음을 보여줍니다. 또한, SBS 기법을 통해 각 supernet에 대해 서브넷을 균형 있게 샘플링하여 훈련의 편향을 줄였습니다.



### FiVL: A Framework for Improved Vision-Language Alignmen (https://arxiv.org/abs/2412.14672)
- **What's New**: 이번 논문은 시각 및 언어 입력을 통합하는 데 중요한 진전을 보여주는 대형 비전 언어 모델(LVLMs)의 새로운 접근 방식을 소개합니다. LVLMs가 시각 정보와 언어적 내용을 동등하게 사용하기 어렵다는 문제를 다루기 위해, 새로운 데이터셋을 설계하여 LVLM의 시각적 기반 구축을 개선하는 것을 목표로 하고 있습니다. 이 방법은 이미지 데이터가 단순한 맥락이 아닌 실질적인 증거로 작용하도록 하여, 더욱 정확한 답변 생성을 가능하게 합니다.

- **Technical Details**: 제안된 FiVL 방법은 LVLM 교육을 위한 데이터셋을 구성하는 새로운 기법을 포함합니다. 이 데이터셋은 이미지 내용을 근거로 활용할 수 있게 하는 훈련 및 평가 도구로 작용하여, 모델이 시각 정보를 얼마나 잘 활용하는지를 측정합니다. 이를 통해 기존 문제점인 시각적 접근을 강화하여, 질문-답변 작업에서의 성능 향상을 도모할 수 있습니다.

- **Performance Highlights**: 제안된 데이터셋을 사용하여 새로운 훈련 작업을 제시하며, 이를 통해 기존 기준을 초월하는 성과를 보여줍니다. 또한 이 모델은 설명 가능성을 향상시키기 위한 검증 방법을 통해 사용자에게 더 나은 이해를 제공합니다. 실험 결과, 모델의 종합적인 성능이 향상되었음을 입증하며, 해당 코드는 온라인에서 이용 가능합니다.



### MUSTER: Longitudinal Deformable Registration by Composition of Consecutive Deformations (https://arxiv.org/abs/2412.14671)
- **What's New**: 이 연구에서는 시간에 따라 변화하는 구조를 분석할 수 있는 새로운 방법인 Multi-Session Temporal Registration (MUSTER)를 소개합니다. MUSTER는 비선형 이미지 등록(non-linear image registration) 기술을 활용하여 여러 세션의 이미지를 동시에 처리함으로써 기존의 쌍(pairwise) 등록 방식을 개선하였습니다. 이 방법은 의료 이미지의 장기적 변화를 분석하는 데 큰 기여를 할 것으로 기대됩니다.

- **Technical Details**: MUSTER는 모든 이미지 간의 변형을 회전, 이동 및 스케일을 조합하여 진행하며, 이를 통해 더욱 일관된 결과를 이끌어냅니다. 이 방법은 기존의 local normalized cross-correlation 기준의 변형 필드 추정을 개선하고, 방해 요소를 줄여서 보다 견고한 결과를 제공합니다. 또한, 그래픽 처리 장치(GPU)를 활용하여 대용량 데이터셋 처리 속도를 높여 연산 리소스가 적은 환경에서도 사용이 가능하도록 설계되었습니다.

- **Performance Highlights**: MUSTER의 성능은 합성 다중 사이트(multi-site), 다중 세션(neuroimaging dataset) 데이터셋을 사용하여 테스트하였고, 여러 시나리오에서 기존 쌍(pairwise) 등록 기법에 비해 변형 추정이 크게 향상되었음을 보여줍니다. 또한, Alzheimer's Disease Neuroimaging Initiative (ADNI) 연구의 샘플에 적용하여 T1-weighted 이미지에서의 신경퇴행 패턴을 효과적으로 식별할 수 있음을 입증하였습니다. 이러한 결과는 인지 변화와의 상관관계를 보여주며, 최첨단 segmentation 방법들과 비교하여 유사한 성능을 나타냅니다.



### Unveiling Uncertainty: A Deep Dive into Calibration and Performance of Multimodal Large Language Models (https://arxiv.org/abs/2412.14660)
Comments:
          Accepted to COLING 2025

- **What's New**: 이 논문에서는 다중 모달 대형 언어 모델(MLLMs)의 불확실성 보정(calibration) 문제를 중점적으로 다루고 있습니다. 모델의 불확실성을 정량화하고 보정하는 과정이 중요하며, 이를 통해 의료 및 자율 주행과 같은 안전-critical 분야에서의 신뢰성을 개선할 수 있습니다. MLLMs는 이미지와 텍스트의 통합 처리를 통해 높은 성능을 보이지만, 여전히 고질적인 성능 불균형 문제가 존재합니다.

- **Technical Details**: MLLMs의 성공적인 활용을 위해, 저자들은 IDK 데이터셋을 구축하여 모델의 불확실성 자기 평가(self-assessment)를 분석했습니다. 두 가지 모달 정보 간의 불확실성 차이를 평가하고, 텍스트에 비해 이미지 정보의 불확실성이 낮은 경향을 관찰했습니다. 추가로, 고급 보정 기법인 temperature scaling과 iterative prompt optimization을 사용하여 MLLMs의 신뢰도와 예측 성능을 향상시킬 수 있는 방법을 제안했습니다.

- **Performance Highlights**: 연구 결과, MLLMs는 고백하기보다는 대답하는 경향이 있으며 이는 적절한 프롬프트 조정으로 개선될 수 있음을 보여주었습니다. 다양한 설정에서 MLLMs의 보정 성능 차이를 분석하였고, 특히 Fine-tuning 전후의 큰 차이를 발견하지 못했습니다. 저자들은 MLLMs의 교육 및 테스트에 있어 더 많은 개선이 필요하다고 결론지으며, 다양한 응용 분야에 대한 책임감 있는 활용 방안을 고안했습니다.



### RefHCM: A Unified Model for Referring Perceptions in Human-Centric Scenarios (https://arxiv.org/abs/2412.14643)
Comments:
          13 pages

- **What's New**: 이 논문은 Referring Human Perceptions라는 새로운 접근 방식을 소개합니다. RefHCM(Referring Human-Centric Model)을 통해 여러 인간 중심의 참조 작업을 통합하여 다양한 적용 가능성을 향상시킵니다. 사용자가 친숙한 텍스트를 사용하여 참조된 개인의 인식을 예측하도록 설계되어 있으며, 이는 기존 작업들과의 차별점으로 작용합니다. 이는 실용적인 현실 세계의 응용 프로그램에 필요한 발전이 될 것입니다.

- **Technical Details**: RefHCM은 이미지, 텍스트, 좌표, 파싱 맵 등의 원시 다중 모달 데이터를 의미적 토큰으로 변환하기 위해 시퀀스 병합 기술(sequence mergers)을 사용합니다. 이 통합된 표현 방식을 통해 RefHCM은 다양한 인간 중심 참조 작업을 시퀀스-투-시퀀스 패러다임(sequence-to-sequence paradigm)으로 변형할 수 있습니다. 또한, 상호 최적화를 활용하여 위치 및 pose 추정의 효과를 극대화하는 Location-Context Restriction 메커니즘(Location-Context Restriction mechanisms)을 도입했습니다.

- **Performance Highlights**: RefHCM은 다양한 인간 중심 참조 작업에서 경쟁력 있는 성능을 달성하며, 특히 복잡한 추론 작업에서의 제로샷 일반화(zero-shot generalization) 능력이 두드러집니다. 이를 통해 대상에 대한 인식 기능을 향상시키고, 향후 다른 응용 프로그램의 발전 가능성도 시사합니다. 다양한 벤치마크에서 진행된 실험 결과는 RefHCM의 우수성을 보여줍니다.



### Adaptive Prompt Tuning: Vision Guided Prompt Tuning with Cross-Attention for Fine-Grained Few-Shot Learning (https://arxiv.org/abs/2412.14640)
- **What's New**: 최근 컴퓨터 비전 분야는 기계 학습 및 딥 러닝 기술의 발전으로 놀라운 성장을 보여주고 있습니다. 이 논문에서는 Contrastive Language-Image Pre-Training (CLIP) 모델을 개선하는 새로운 방법인 Adaptive Prompt Tuning (APT)을 제안하여 한정된 데이터로 미세 분류 작업을 돕습니다. 이 접근법은 이미지와 텍스트 간의 동적 상호작용을 통해 텍스트 프롬프트를 조정하여 보다 효과적인 분류를 가능하게 합니다.

- **Technical Details**: APT는 이미지 입력에 따라 프롬프트를 조정하는 교차 주의 메커니즘을 활용하여 CLIP의 텍스트와 이미지 요소의 정렬 능력을 향상시킵니다. Monte-Carlo Dropout (MCD)을 통합하여 예측에 대한 신뢰도를 향상시키고 예측 불확실성을 파악합니다. 이는 시스템의 성능과 신뢰성을 높이는 데 중요한 역할을 하며, 특히 의료 분야와 같은 중요한 응용 프로그램에서 중요합니다.

- **Performance Highlights**: 여러 데이터셋(CUBirds, Oxford Flowers, FGVC Aircraft)에서 실시된 평가에서 APT는 기존의 정적 프롬프트 조정 방식에 비해 유의미한 성능 향상을 보였습니다. 또한, 모델의 예측 신뢰도를 개선하는 데 성공함으로써, 신뢰할 수 있는 예측을 제공하고 필요한 경우 추가 검증이 필요하다는 인사이트를 제공합니다. 이 방법은 몇 샷 미세 분류의 최신 기술을 크게 향상시키는 견고한 솔루션으로 작용합니다.



### Progressive Fine-to-Coarse Reconstruction for Accurate Low-Bit Post-Training Quantization in Vision Transformers (https://arxiv.org/abs/2412.14633)
- **What's New**: 이 논문에서는 Vision Transformers (ViTs)의 Post-Training Quantization (PTQ)에서 발생하는 성능 저하 문제를 해결하기 위해 Progressive Fine-to-Coarse Reconstruction (PFCR) 방법을 제안합니다. PFCR은 여러 개의 재구성 단위를 점진적으로 최적화하여 낮은 비트 양자화 설정에서의 성능을 크게 향상시킵니다. 또한, 재구성 과정을 두 단계로 나누어 구성하는 Progressive Optimization Strategy (POS)라는 새로운 전략을 도입하여 최적화의 어려움을 완화합니다.

- **Technical Details**: PFCR 방법에서는 Multi-Head Self-Attention (MHSA)와 Multi-Layer Perceptron (MLP) 모듈을 가장 세밀한 재구성 단위로 정의합니다. 이 두 모듈의 최적화가 완료된 후에, 그들을 결합하여 블록 형태의 더 거친 재구성을 수행합니다. 이 과정은 세밀한 재구성 단위를 점진적으로 거친 단위로 조합하고 재구성하는 방식으로 반복적으로 이루어집니다.

- **Performance Highlights**: ImageNet 데이터셋에서의 실험 결과에 따르면, 제안된 PFCR 방법은 3비트 양자화된 ViT-B에서 75.61%의 Top-1 정확도를 달성하여 기존의 최첨단 기술보다 우수한 성능을 보였습니다. 또한 COCO 데이터셋에서의 양자화 결과는 제안된 방법이 객체 탐지 및 인스턴스 분할과 같은 다른 컴퓨터 비전 작업에도 효과적으로 일반화됨을 보여줍니다.



### Review of Fruit Tree Image Segmentation (https://arxiv.org/abs/2412.14631)
- **What's New**: 이 논문은 과수 나무 이미지 분할(Fruit tree image segmentation)과 관련된 최근 연구 동향을 종합적으로 검토합니다. 158개의 관련 논문을 새롭게 설계한 크롤링 리뷰 방법으로 수집하여 분석하였습니다. 이 리뷰는 연구자들이 다양한 작업과 환경에 적용할 수 있는 연구의 전반적인 상황을 직관적으로 이해할 수 있도록 돕는 분류 체계(taxonomy)를 제공합니다.

- **Technical Details**: 논문은 과수 나무의 전면 이미지에 초점을 맞추어 연구를 진행하였으며, 사용된 분류 체계는 방법(Method), 이미지(Image), 작업(Task), 및 과일(Fruit)로 구성되어 있습니다. 이를 바탕으로 연구의 체계적인 검토가 이루어졌습니다. 기존 연구들의 가장 큰 부족점은 다목적 데이터셋과 다양한 환경에 적용 가능한 분할 모델의 부재입니다.

- **Performance Highlights**: 미래의 연구 과제로 제시된 여섯 가지 중요한 작업은 다목적 나무 분할 모듈을 구축하기 위한 기초가 될 것으로 기대됩니다. 이 연구는 농업 작업의 자동화를 위한 새로운 접근 방식을 제안하며, 농업 기술 발전에 기여할 것으로 보입니다.



### Unified Image Restoration and Enhancement: Degradation Calibrated Cycle Reconstruction Diffusion Mod (https://arxiv.org/abs/2412.14630)
- **What's New**: CycleRDM(주기 재구성 확산 모델)은 이미지 복원과 향상 작업을 통합하고 고품질 매핑을 달성하도록 설계된 새로운 프레임워크이다. 이 방법은 반복적인 정교화(iterative refinement) 특성을 활용하여 손상된 도메인과 정상 도메인 간의 매핑 관계를 학습하고, 이를 통해 다운스트림 작업의 성능을 향상시킨다.

- **Technical Details**: CycleRDM은 처음에 손상된 도메인에서 거친(normal rough) 도메인으로, 그리고 최종 정상 도메인으로 이어지는 두 단계의 확산 추론 과정을 통해 매핑 관계를 학습한다. 또한, 분해된 wavelet 고주파 도메인에서 중복된(feature redundancy) 특징들을 제거하기 위해 특징 증대 모듈을 설계하였고, 고유한 다중 모드 텍스트와 푸리에 변환(Fourier transform)을 활용하여 안정적인 잡음 제거를 추진한다.

- **Performance Highlights**: CycleRDM은 9가지의 다양한 손상 유형에서 실험을 통해 효과적으로 통합된 복원 및 향상 작업을 보여주며, 제한된 훈련 데이터로도 매우 경쟁력 있는 성능을 달성했다. 이 모델은 다양한 벤치마크에서 구조적 질적, 지각적 질적 기준으로 탁월한 결과를 나타낸다.



### Qua$^2$SeDiMo: Quantifiable Quantization Sensitivity of Diffusion Models (https://arxiv.org/abs/2412.14628)
Comments:
          AAAI 2025; version includes supplementary material; 22 Pages, 18 Figures, 8 Tables

- **What's New**: 이번 연구에서는 Qua²SeDiMo라는 mixed-precision Post-Training Quantization 프레임워크를 제안하여 다양한 디퓨전 모델의 weight quantization 기법과 해당 작업에 대한 민감도를 분석합니다. 이를 통해 유닛 구조에 따라 최적의 비트 정밀도를 결정하고, 이미지 생성 품질을 유지하면서 비용 효율적인 quantization 전략을 제공합니다. 연구팀은 다양한 디퓨전 모델 유형에 대한 고급 mixed-precision weight quantization 결정을 수립하여, 기존 기법들을 초월하는 성능을 기록했습니다.

- **Technical Details**: Qua²SeDiMo는 직접적으로 신경망의 각 레이어의 양자화 방법과 비트 정밀도를 end-to-end 성능 지표에 연관시키는데 중점을 둡니다.  이 기술은 덴오이저 아키텍처를 그래프로 표현하고 최적화된 GNN(그래프 신경망) 방법을 활용하여 각 레이어의 그래프 성능을 할당하는 방식으로 구성되어 있습니다. 또한, 데이터셋을 필요로 하지 않고 다양한 디퓨전 모델에 대해 3.4 비트에서 3.7 비트까지의 PTQ(훈련 후 양자화) 조정을 수행했습니다.

- **Performance Highlights**: Qua²SeDiMo는 PixArt-α, PixArt-Σ, Hunyuan-DiT 및 SDXL에 대해 각각 3.4-bit, 3.9-bit, 3.65-bit 및 3.7-bit의 weight quantization을 달성하였으며, 6-bit의 activation quantization과 결합하여 기존 접근 방식을 초과하는 시각적 품질과 FID, CLIP 점수를 기록했습니다. 연구 결과는 U-Nets가 균일한 스케일 기반 양자화에 더 잘 적응하는 반면, DiT 모델은 클러스터 기반 방법에 더 민감하다는 것을 보여주며, 이는 성능을 크게 향상시키는 중요한 인사이트로 작용합니다.



### FRIDAY: Mitigating Unintentional Facial Identity in Deepfake Detectors Guided by Facial Recognizers (https://arxiv.org/abs/2412.14623)
Comments:
          5 pages, 4 figures. In 2024 IEEE International Conference on Visual Communications and Image Processing (VCIP) Oral

- **What's New**: 최근 Deepfake 검출 방법들이 훈련된 도메인에서는 뛰어난 성능을 보이나, 새로운 합성 기술이 등장하면 그 효과가 크게 감소하는 문제가 있었습니다. 이를 해결하기 위해, Facial Recognition Identity Attenuation (FRIDAY)라는 새로운 훈련 방법을 제안합니다. 이 방법은 얼굴 인식기를 사용하여 얼굴 기반 정보를 감소시킴으로써, Deepfake 검출기에서 의도하지 않게 학습되는 얼굴 아이덴티티 문제를 해결합니다.

- **Technical Details**: FRIDAY 방법은 두 단계로 이루어져 있으며, 첫 번째 단계에서는 얼굴 인식기를 훈련하여 Deepfake 검출기의 훈련 과정에 활용합니다. 이후, 둘째 단계에서는 얼굴 인식기가 고정(frozen)된 상태에서 Deepfake 검출기를 훈련하며, 얼굴 아이덴티티 특징을 줄이기 위해 Facial Identity Attenuating loss를 적용합니다. 각 모델은 동일한 구조를 가지지만 파라미터는 공유하지 않으며, 이로 인해 검출기가 얼굴 아이덴티티에 의존하지 않도록 합니다.

- **Performance Highlights**: 포괄적인 실험을 통해, FRIDAY 방법이 도메인 내 및 도메인 간 데이터셋 모두에서 검출 성능을 상당히 향상시키는 것으로 나타났습니다. FaceForensics++, Celeb-DF v1 & v2, DeepfakeDetection과 같은 다양한 데이터셋에 걸쳐 성능을 평가한 결과, 모델의 일반화 성능이 크게 개선되었습니다. 기존 최첨단 모델들과의 비교에서도 FRIDAY 방식이 우수한 성능을 기록하였습니다.



### Pitfalls of topology-aware image segmentation (https://arxiv.org/abs/2412.14619)
Comments:
          Code is available at this https URL

- **What's New**: 이 논문은 의료 이미징에서의 topology-aware 방법의 실제 적용을 저해하는 평가 기준의 결함을 지적합니다. 특히, ill-suited connectivity choices, ground truth annotations의 topological artifacts, 그리고 부적절한 evaluation metrics의 문제를 강조합니다. 이는 segmentation 방법의 성능 평가 및 순위 매김에 중대한 영향을 미치는 것으로 분석되었습니다.

- **Technical Details**: 저자들은 topological correctness를 보존하기 위한 다양한 접근 방식을 검토하고, 픽셀 기반 성능 측정 외에도 여러 topological metrics와 그 활용의 중요성을 논의합니다. 특히, persistence homology(PH)와 discrete Morse theory(DMT)를 활용한 방법들이 탐구되며, 이들이 어떻게 topological accuracy를 측정하는지에 대한 통찰을 제공합니다. 또한, 여러 분류 작업에 있어 각 topological metric의 적용 방식과 의의가 설명됩니다.

- **Performance Highlights**: 저자들은 의료 이미지에서 neuron 및 vessel segmentation의 품질이 topology-aware 메트릭을 기반으로 개선될 수 있음을 강조합니다. 그러나 benchmarking의 불완전함으로 인해 방법 비교가 부정확하게 이루어지고 있으며, 이는 실질적인 의료 응용에 있어 장애물이 되고 있습니다. 논문에서 제안된 해결책들은 이러한 평가 기준의 개선과 더 나은 상호 비교를 위한 중요한 기초가 될 것입니다.



### Successive optimization of optics and post-processing with differentiable coherent PSF operator and field information (https://arxiv.org/abs/2412.14603)
- **What's New**: 이번 논문에서는 복잡한 굴절 렌즈를 시뮬레이션하고 미분할 수 있는 효율적인 광학 시뮬레이션 모델을 소개합니다. 이 모델은 특히 비구면 표면 및 파면 왜곡(wavefront aberrations)과 조리개 회절 효과(aperture diffraction effects)를 고려하여 높은 정확도로 렌즈의 간섭성을 계산할 수 있습니다. 또한, 새로운 초기값 전략(initial value strategy)을 통해 고도로 비구면인 표면에서 내적(intersection) 계산의 정확성을 향상시킵니다.

- **Technical Details**: 제안된 모델은 정확하고 미분 가능한 렌즈 형성(ray-tracing based) 방식을 사용합니다. 새로운 초기값 추정을 통해 경계면에서의 교차점을 더 정확하게 찾아낼 수 있으며, 파동(front aberrations) 기반의 PSF 계산 방법을 채택하여 메모리 사용량을 효율적으로 줄입니다. 또한, 이 과정에서 발생하는 각종 광학 제약 조건(optical constraint terms)을 손실 함수(loss functions)에 포함시켜 정확한 모델링을 지원합니다.

- **Performance Highlights**: 제안된 공동 최적화 파이프라인(joint optimization pipeline)은 여러 개의 렌즈에 대한 성능을 지속적으로 개선하여 이미지 품질을 향상시키고 있습니다. 이 방법은 실제적인 광학 시스템 설계와 후처리 알고리즘(image post-processing algorithm)의 통합에 대한 획기적인 통찰력을 제공합니다. 더구나, 이 모델은 이미 전문적인 수준의 렌즈의 성능을 점차 회절 한계(diffractive limit)에 가깝게 끌어올리고 있다는 점에서 중요한 발전을 이루었습니다.



### Can We Get Rid of Handcrafted Feature Extractors? SparseViT: Nonsemantics-Centered, Parameter-Efficient Image Manipulation Localization Through Spare-Coding Transformer (https://arxiv.org/abs/2412.14598)
Comments:
          12 page, 8 figures, published to AAAI

- **What's New**: 이 논문에서는 저자들이 Sparse Vision Transformer(SparseViT)를 제안하여 이미지 조작 지역화(Image Manipulation Localization, IML) 문제를 해결하고자 하였습니다. 기존의 IML 모델들은 수작업으로 비의미(non-semantic) 특성을 추출하는 데 의존했지만, 이 논문에서는 이러한 비의미 특성을 효율적으로 추출하기 위한 적응 메커니즘을 개발했습니다. SparseViT는 희소(self-attention) 자기 주의 메커니즘을 활용하여 비의미 특성을 적응적으로 추출하게 됩니다.

- **Technical Details**: SparseViT는 ViT에서의 밀집(dense) 글로벌 셀프 어텐션을 희소(sparse)하게 재구성하여 이미지의 비의미 특성을 효율적으로 추출합니다. 이 모델은 서로 상관 없는 이미지 패치 간의 상호작용을 조정하여 비의미 특성을 강화하며, 다양한 밀도 수준에서 작동하는 다중 스케일 통합 모듈을 통해 특성 맵을 통합하여 모델의 이해도를 높입니다.

- **Performance Highlights**: SparseViT는 다양한 벤치마크 데이터 세트에서 높은 성능을 기록하며, 다른 기존 IML 모델들에 비해 우수한 일반화 및 효율성을 입증했습니다. 향상된 계산 효율성과 함께 최대 80%의 플롭(FLOPs) 감소가 이루어져 모델의 크기를 크게 줄일 수 있었습니다. 이 연구는 비의미 특성을 적응적으로 추출하는 혁신적인 접근법으로, 기존의 방법들과 비교하여 우수한 성능(SOTA)을 달성하였습니다.



### LDP: Generalizing to Multilingual Visual Information Extraction by Language Decoupled Pretraining (https://arxiv.org/abs/2412.14596)
Comments:
          Accepted by AAAI2025

- **What's New**: 이 논문은 시각 정보 추출(Visual Information Extraction, VIE) 분야에서 다국어 모델의 효율성을 높이기 위해, 언어 편향을 분리하여 전이를 촉진하는 간단하지만 효과적인 다국어 훈련 패러다임인 LDP (Language Decoupled Pre-training)를 제안합니다. 모델 LDM (Language Decoupled Model)은 먼저 언어 독립적인 데이터로 사전 훈련된 후, 다운스트림 언어에 대해 미세 조정됩니다. 이로 인해 다국어 전이 성능이 향상되며, 영어만으로 이루어진 사전 훈련 데이터의 활용도를 극대화합니다.

- **Technical Details**: 제안된 LDM 모델은 SAM (Segment Anything Model) 프레임워크를 기반으로 하며, 시각적 문서의 정보를 효과적으로 추출하기 위해 디자인되었습니다. 모델은 텍스트 편집 확산 모델(TGD)을 사용하여 이미지에서 언어 편향을 분리하고, 다양한 이미지 내의 정보를 통합하는 MTIM (Multi-Token Information Merging) 모듈을 포함합니다. 이러한 구조는 모델의 성능을 한층 강화하여, 복잡한 시나리오에서의 정보 추출 가능성을 높입니다.

- **Performance Highlights**: 실험 결과 LDM은 XFUND 및 SIBR과 같은 다국어 벤치마크에서 경쟁 모델들보다 뛰어난 성과를 기록하였으며, 영어 데이터셋에서도 경쟁력 있는 결과를 유지합니다. 우리의 연구는 VIE 작업에서 비주얼 인바리언스(visual invariance)를 체계적으로 연구한 최초의 시도로, 훈련 데이터에서 언어 편향을 분리함으로써 다국어 일반화 성능이 향상된다는 것을 보여줍니다.



### Multi-Sensor Object Anomaly Detection: Unifying Appearance, Geometry, and Internal Properties (https://arxiv.org/abs/2412.14592)
- **What's New**: 이 논문에서는 산업 품질 검사에서 필수적인 물체 이상 탐지를 위해 MulSen-AD라는 새로운 고해상도 다중 센서 데이터셋을 소개합니다. MulSen-AD는 RGB 카메라, 레이저 스캐너, 그리고 락인 적외선 열화상 이미징 기술을 통합하여 다양한 유형의 이상을 포착할 수 있도록 설계되었습니다. 또한, MulSen-TripleAD라는 결합 알고리즘을 제안하여 다중 센서 데이터를 통합하여 견고한 비지도형 물체 이상 탐지를 수행합니다.

- **Technical Details**: MulSen-AD 데이터셋은 15개의 산업 제품에서 수집된 다양한 실제 이상을 포함하고 있으며, RGB 이미지, 적외선 열 이미지, 그리고 3D 포인트 클라우드 데이터를 제공합니다. 이 데이터셋은 기존 데이터셋의 한계를 극복하고 다양한 모달리티에 걸친 이상 탐지를 가능하게 합니다. MulSen-TripleAD 모델은 데이터 결합을 위한 결정 수준의 융합 방식을 채택하여 더 높은 정확도를 달성하고 있습니다.

- **Performance Highlights**: 실험 결과, MulSen-TripleAD는 단일 센서 모델보다 훨씬 높은 성능을 보여주며, 물체 수준의 탐지 정확도에서 96.1%의 AUROC를 기록했습니다. 이러한 결과는 다중 센서 데이터 융합이 산업 분야에서 이상 탐지 성능을 향상시키는 데 중요한 역할을 한다는 점을 강조합니다. 이를 통해 다중 센서 접근 방식의 필요성이 더욱 부각되고 있습니다.



### Spike2Former: Efficient Spiking Transformer for High-performance Image Segmentation (https://arxiv.org/abs/2412.14587)
Comments:
          This work has been accepted on Association for the Advancement of Artificial Intelligence 2025

- **What's New**: 본 논문에서는 스파이킹 신경망(Spiking Neural Networks, SNNs)을 복잡한 아키텍처에서 이미지 분할(imag segmentation) 작업에 적용하기 위해 새로운 Spike2Former 아키텍처를 제안합니다. 이러한 아키텍처는 정보 손실을 최소화하고 훈련 안정성을 높이기 위한 기능을 포함하고 있으며, 이는 SNNs의 성능 저하 문제를 해결합니다. 추가적으로, 새롭게 설계된 NI-LIF 스파이킹 뉴런인 정규화된 정수 스파이킹 뉴런(normalized integer spiking neurons)을 도입하여 SNN의 변화를 해결합니다.

- **Technical Details**: Spike2Former 아키텍처는 Mask2Former 아키텍처를 기반으로 하며, 정보 손실을 방지하기 위해 구조를 개선하였습니다. 특히, 변형 가능한 주의 변환기 인코더 블록(Deformable Attention Transformer Encoder Block)과 마스크 임베딩(mask embedding) 계층에서 가장 큰 정보 손실이 발생하였으며, 이를 해결하기 위해 전환 블록(convolution blocks)을 도입하고 보조 정보 브랜치를 추가하여 표현력을 강화하였습니다. 또한, NI-LIF 스파이킹 뉴런은 훈련 중 정수 활성화(normalization)를 정규화하여 훈련 안정성을 높입니다.

- **Performance Highlights**: 제안된 Spike2Former는 ADE20K, Pascal VOC2012 및 CityScapes 데이터셋에서 기존 SNN 모델에 비해 뛰어난 성능을 나타냅니다. ADE20K에서는 +12.7% mIoU(Mean Intersection over Union)와 5.0배 에너지 효율, VOC2012에서는 +14.3% mIoU와 5.2배 효율, CityScapes에서는 +9.1% mIoU와 6.6배 효율을 기록합니다. 이러한 성과는 복잡한 시나리오에서 SNN의 잠재력을 보여줍니다.



### HiCM$^2$: Hierarchical Compact Memory Modeling for Dense Video Captioning (https://arxiv.org/abs/2412.14585)
Comments:
          AAAI2025

- **What's New**: 본 연구에서는 인지 과정에서 영감을 받은 새로운 Dense Video Captioning(DVC) 모델 HiCM2(Hierarchical Compact Memory Modeling)를 제안합니다. HiCM2는 인간의 메모리 구조를 모방하여 사건의 지역화와 설명의 질을 향상시키기 위해 계층적 메모리와 메모리 판독 모듈을 구축하였습니다. 이 모델은 대규모 언어 모델을 활용하여 메모리 사건을 클러스터링하고 요약함으로써 효율적인 계층적 메모리 구조를 형성합니다. 실험 결과, HiCM2는 YouCook2 및 ViTT 데이터셋에서 최첨단 성능을 달성하였습니다.

- **Technical Details**: HiCM2는 인간의 인지 메모리 구조를 계층적으로 조직화하여 메모리를 구축합니다. 이 모델은 고수준의 추상 정보부터 시작하여 세부 사항으로 점진적으로 접근하는 상향식(top-down) 메모리 판독 방식을 사용합니다. 이를 통해 효율적으로 사건 후보를 추출하고, 사건에 대한 정보의 다양성과 의미적 풍부성을 보장합니다. 연구는 대규모 언어 모델을 사용하여 각 계층에서 요약된 정보를 포함하는 메모리 구조를 형성하여, 정보 검색 과정을 개선합니다.

- **Performance Highlights**: 연구팀은 YouCook2와 ViTT 데이터셋에 대한 실험을 통해 HiCM2의 성능을 평가하였습니다. 기존의 방법들과 비교하여, 이 모델은 사건의 지역화 및 자막 생성에서 더 나은 결과를 보였으며, 효율적인 메모리 활용이 성능 향상에 기여함을 보여주었습니다. 연구 결과, HiCM2는 신뢰할 수 있는 자막 생성을 위한 의미 있는 정보 검색을 가능하게 하여 DVC 분야의 발전에 기여하고 있습니다.



### DiffSim: Taming Diffusion Models for Evaluating Visual Similarity (https://arxiv.org/abs/2412.14580)
- **What's New**: 이 논문에서는 사전 훈련된 확산 모델을 활용해 시각적 유사성을 측정할 수 있는 DiffSim 메소드를 소개합니다. 기존의 전통적인 측정 기준이 놓치기 쉬운 중간 레벨의 유사성을 포착하는 데 한계를 가지고 있었던 반면, DiffSim은 이미지의 형태, 객체 자세, 의미적 내용을 포함한 다양한 요소들을 묶어 평가합니다. 새로운 Sref와 IP 벤치마크를 도입하여 스타일과 인스턴스 레벨의 유사성을 평가합니다.

- **Technical Details**: DiffSim은 denoising U-Net의 attention layer에서 특징을 정렬하여 외형 및 스타일 유사성을 평가합니다. Aligned Attention Score (AAS)를 통해 두 이미지 A와 B의 특징을 정렬하고, 그 사이의 코사인 거리(cosine distance)를 계산하여 시각적 유사성을 측정합니다. 이 방법은 고유한 시각적 특징을 각 층과 denoising timestep에서 기반으로 유사성 측정의 효과를 높입니다.

- **Performance Highlights**: DiffSim은 광범위한 벤치마크를 통해 입증된 바와 같이 타의 추종을 불허하는 성능을 자랑합니다. 추가적인 미세 조정이나 감독 없이도 CLIP 및 DINO v2를 초월하는 결과를 보여 실제 사용자 평가와도 높은 일치도를 보였습니다. 새로운 Style-ref 및 IP-ref 벤치마크에서 스타일 유사성과 외양 일관성 평가에서 최상의 결과를 기록하였습니다.



### GSRender: Deduplicated Occupancy Prediction via Weakly Supervised 3D Gaussian Splatting (https://arxiv.org/abs/2412.14579)
- **What's New**: 이번 연구에서는 3D Occupancy Perception(점유 인식)을 위한 새로운 방법인 GSRender를 소개합니다. GSRender는 3D Gaussian Splatting을 활용하여 더 정밀하고 효율적인 점유 예측을 가능하게 하며, 카메라 광선에 따른 샘플링 프로세스를 단순화합니다. 또한, Ray Compensation(RC) 모듈을 도입하여 인접 프레임의 특성을 보완함으로써 중복 예측을 줄입니다. 이와 함께 동적 물체의 영향력을 최소화하기 위해 차별화된 손실 함수를 설계하였습니다.

- **Technical Details**: GSRender는 3D Gaussian Splatting을 적용하여 고화질의 포토리얼리스틱(photorealistic) 장면을 실시간으로 렌더링합니다. 이 과정에서 각 Gaussian의 평균과 공분산을 포함하여 여러 속성을 고려하며, 2D 이미지 평면으로의 투영도 수행합니다. 연구팀은 기존의 NeRF 기반 방법들이 겪는 샘플링 트레이드오프 문제를 해결하고, 단지 하나의 추가 프레임을 통해 특징을 통합하여 성능을 향상시켰습니다. 이를 통해 2D 약한 감독을 통해도 효과적인 3D 점유 예측을 실현했습니다.

- **Performance Highlights**: GSRender는 RayIoU 지표에서 SOTA(상태 최우수) 성과를 달성하며, 2D 약한 감독 방법들 사이에서 눈에 띄는 성과를 보였습니다. 실험 결과, 기존의 3D 감독 방법들과 비교하여 성능 격차를 크게 줄였으며, 실제 데이터셋(Nuscenes)에서도 뛰어난 결과를 기록했습니다. 이 연구는 특히 자율주행 기술의 발전에 기여할 것으로 기대되고 있습니다.



### Alignment-Free RGB-T Salient Object Detection: A Large-scale Dataset and Progressive Correlation Network (https://arxiv.org/abs/2412.14576)
Comments:
          Accepted by AAAI 2025

- **What's New**: 본 논문에서는 20,000개의 이미지 쌍으로 구성된 UVT20K라는 대규모 비정렬 RGB-열 감지(SOD) 데이터셋을 구축했습니다. 이는 현재까지의 다중 모드 SOD 데이터셋 중 가장 큰 규모로, 다양한 실제 환경에서 수집된 샘플로 구성되어 있습니다. 이러한 데이터셋은 비정렬 RGB-T SOD 방법의 훈련 및 평가에 큰 도움이 될 것입니다.

- **Technical Details**: 제안하는 Progressive Correlation Network(PCNet)는 명시적인 정렬을 바탕으로 이미지 쌍 간의 상관관계를 모델링하여 비정렬 이미지 쌍에서의 정확한 예측을 수행합니다. 이를 위해, Semantics-guided Homography Estimation(SHE) 모듈을 도입하여 RGB와 열 모드 간의 공통 영역을 정렬합니다. 또한 Inter- and Intra-Modal Correlation(IIMC) 모듈은 salient 지역의 상관관계를 점진적으로 모델링하여 더 나은 예측을 가능하게 합니다.

- **Performance Highlights**: UVT20K 데이터셋과 함께, 제안된 PCNet의 성능은 최신 SOTA(State Of The Art) 방법들과 비교하여 효과적임을 보여줍니다. 이 연구는 비정렬 데이터셋에서 RGB-T SOD의 가능성을 열어줄 뿐만 아니라, 실질적인 응용과 배포를 위한 중요한 기초 자료를 제공합니다. 최종적으로, 이 데이터셋과 방법론은 비정렬 SOD 연구에 큰 기여를 할 것으로 기대됩니다.



### SCKD: Semi-Supervised Cross-Modality Knowledge Distillation for 4D Radar Object Detection (https://arxiv.org/abs/2412.14571)
Comments:
          Accepted by AAAI 2025

- **What's New**: 이 논문에서는 4D 밀리미터파 레이더를 기반으로 한 3D 객체 탐지를 위한 새로운 반지도형 교차 모달리티 지식 증류(SCKD) 방법을 제안합니다. 이 방법은 반지도형(distillation) 학습을 통해 레이저와 레이더의 융합된 teacher 네트워크로부터 특징을 학습할 수 있게 합니다. 기존의 다중 모달리티 융합 접근법의 한계를 극복하며, 레이더 전용의 학생 네트워크 성능을 크게 향상시킬 수 있습니다.

- **Technical Details**: 제안된 SCKD 방법은 Lidar와 레이더의 융합을 담당하는 적응형 융합 모듈을 teacher 네트워크에 추가하여 성능을 높입니다. 이후, Lidar에서 레이더로의 특징 증류(LRFD) 및 융합에서 레이더로의 특징 증류(FRFD) 두 가지 방법을 설계하였습니다. 또한, 반지도형 출력 증류(SSOD)를 통해 teacher와 student 간의 지식 전이를 보다 효과적으로 수행합니다.

- **Performance Highlights**: 실험 결과, SCKD 방법은 VoD 데이터셋에서 기본 성능보다 mAP(Mean Average Precision)를 10.38% 향상시키고, ZJUODset에서는 추가 비정형 데이터를 사용할 경우 5.12%의 성능 향상을 기록했습니다. 이로써 제안된 방법은 최신 방법들보다 뛰어난 성능을 보이며, 대량의 라벨 없는 데이터 활용 가능성을 열어줍니다.



### Improving Geometry in Sparse-View 3DGS via Reprojection-based DoF Separation (https://arxiv.org/abs/2412.14568)
Comments:
          11 pages

- **What's New**: 이번 연구에서는 최근의 multi-view stereo (MVS) 모델들이 제안한 sparse-view 3D 재구성의 한계를 극복하기 위해, 3D Gaussian Splatting (3DGS) 를 통한 정제 과정에서 발생하는 기하학적 문제들을 개선할 방법을 제안합니다. 특히, 제안된 방법은 Gaussians의 과도한 positional degrees of freedom (DoFs) 가 기하학적 왜곡을 유도한다는 가설을 바탕으로 하여, 각 DoF를 불확실성에 따라 다양하게 관리하는 접근 방식을 사용합니다. 이를 통해 우리는 더 나은 구조적 충실성(structural fidelity)을 유지하며, 최종적으로 비주얼 및 기하학적으로 신뢰할 수 있는 재구성을 가능하게 했습니다.

- **Technical Details**: 우리는 reprojection-based DoF separation 방식을 통해 각 DoF의 불확실성에 따라 이미지-평면 평행 DoF와 광선 정렬 DoF로 구분합니다. 이러한 구별을 통해 낮은 불확실성을 가진 DoF는 픽셀 내에서의 움직임을 제한하고, 높은 불확실성을 가진 DoF는 다중 뷰 정보에 기반한 깊이 정보를 통해 개선합니다. 이 방법은 MVS 모델에서 얻은 유용한 깊이 정보를 보존하면서 기하학적 저해 요소를 줄이는 데 중요한 역할을 합니다.

- **Performance Highlights**: 우리의 방법은 Mip-NeRF 360, MVImgNet, Tanks and Temples와 같은 여러 새 뷰 합성 벤치마크에서 평가되어, PSNR 및 패치별 Pearson 상관 관계와 같은 정량적 분석을 통해 그 효과가 입증되었습니다. 제안한 접근 방식은 기하학적 품질을 유지하면서도 렌더링 품질을 저해하지 않으며, 다양한 벤치마크에서 일관되게 성능 향상을 보여주었습니다.



### GBRIP: Granular Ball Representation for Imbalanced Partial Label Learning (https://arxiv.org/abs/2412.14561)
Comments:
          AAAI25

- **What's New**: 이번 연구에서는 Granular Ball Representation for Imbalanced Partial Label Learning (GBRIP)이라는 새로운 접근 방식을 제안합니다. GBRIP는 기존의 방법들이 간과한 intra-class imbalance (클래스 내 불균형)과 inter-class imbalance (클래스 간 불균형) 문제를 해결하기 위해 설계되었습니다. 이 방법은 coarse-grained granular ball representation과 multi-center loss를 활용하여 label disambiguation (레이블 분류)에 대한 신뢰성 있는 매트릭스를 구축합니다.

- **Technical Details**: GBRIP는 unsupervised learning (비지도 학습)을 통해 각 클래스 내 feature distribution (특징 분포)을 효과적으로 포착합니다. Coarse-grained granular ball representation은 feature 공간을 균형 잡힌 coarse-grained 수준으로 나누어, 각 sub GB가 클래스를 나타내도록 구성합니다. Multi-Center Loss 함수는 샘플과 그에 해당하는 중심 간의 관계를 강조하여, 더 robust한 학습을 가능하게 합니다.

- **Performance Highlights**: GBRIP는 여러 표준 벤치마크에서 광범위한 실험을 통해 기존의 최첨단 방법들을 초월하는 성능을 입증하였습니다. 특히, 이 방법은 imbalanced partial label learning에 있어서 성능 향상을 목표로 하며, label confidence matrices의 정확성을 높여 label disambiguation 과정을 개선합니다. 따라서 GBRIP는 imbalanced PLL 문제 해결을 위한 강력한 해법을 제공합니다.



### ScaMo: Exploring the Scaling Law in Autoregressive Motion Generation Mod (https://arxiv.org/abs/2412.14559)
- **What's New**: 이 연구는 텍스트 기반의 모션 생성에서 스케일링 법칙(scaling law)의 적용을 탐구합니다. 특히, Motion FSQ-VAE와 텍스트 프리픽스 자기회귀 변환기(autoregressive transformer)를 포함하는 확장 가능한 모션 생성 프레임워크를 도입합니다. 이를 통해, 모션 생성에서도 스케일링 법칙이 존재한다는 것을 처음으로 확인하고, 효율적인 데이터 요건과 모델 크기를 예측합니다.

- **Technical Details**: 저자들은 MotionUnion이라는 260시간 이상의 모션 데이터셋을 수집하고, 기존의 정적 모션 문제를 해결하기 위해 FSQ(Finite Scale Quantization)를 적용하여 모션 토크나이징(tokenization)을 수행합니다. 이 프레임워크는 텍스트 토큰에 대해 양방향 주의(bidirectional attention)를 적용하고, 모션 토큰에 대해서는 인과적 주의(causal attention)를 사용합니다. 이를 통해 모션 생성 모델의 확장성에 대한 포괄적인 연구를 수행합니다.

- **Performance Highlights**: 실험 결과, 정규화된 테스트 손실(normalized test loss)과 FLOPs 간의 로그 관계(logarithmic relationship)를 관찰하였으며, 이를 통해 주어진 FLOPs에 대한 테스트 손실을 예측할 수 있음을 확인했습니다. 또한, 어휘 크기(vocabulary size), 모델 크기(model size), 데이터 토큰 간의 거시적 법칙(power law) 관계를 발견하여, 최적의 모델 크기와 어휘 크기 및 데이터 요구 사항을 도출하는 데 기여했습니다.



### Bright-NeRF:Brightening Neural Radiance Field with Color Restoration from Low-light Raw Images (https://arxiv.org/abs/2412.14547)
Comments:
          Accepted by AAAI2025

- **What's New**: 본 연구에서는 Multi-view low-light raw images를 이용하여 비지도 학습 방식으로 향상된 고품질의 Radiance Fields를 생성하는 Bright-NeRF를 제안합니다. 이 방법은 저조도 환경에서도 정확한 장면 재현을 가능하게 하며, 조명에 관계없이 일관성 있는 색상 인식을 달성합니다. 실험 결과, 제안된 방법은 기존의 2D, 3D 접근 방식에 비해 현저한 성능 향상을 보여줍니다.

- **Technical Details**: Bright-NeRF는 조명에 대한 센서의 물리적 응답 모델을 활용하고, 색상 조정 손실(chromatic adaptation loss)을 도입하여 학습을 제어합니다. 또한, 자동 노출 조정과 여러 뷰의 정보를 집계하여 효과적인 노이즈 억제를 이룹니다. 저조도 다중 뷰 RAW 데이터셋(LMRAW)을 구축하여 연구의 기초를 마련했습니다.

- **Performance Highlights**: Bright-NeRF는 노이즈 억제, 색상 왜곡 수정 및 밝기 향상을 달성하며, 정상 조명 조건에서 새로운 뷰를 합성하는 데 있어 최첨단 성능을 발휘합니다. 다양한 2D 이미지 향상 방법 및 NeRF 기반 접근 방식과 비교한 결과, 본 연구의 방법이 다중 뷰 일관성을 유지하면서 탁월함을 보였습니다.



### {S$^3$-Mamba}: Small-Size-Sensitive Mamba for Lesion Segmentation (https://arxiv.org/abs/2412.14546)
Comments:
          Accept by AAAI 2025

- **What's New**: 이 논문에서는 소형 병변의 정확한 분할을 위해 Small-Size-Sensitive Mamba(S$^3$-Mamba)라는 새로운 모델을 제안합니다. 기존 모델들이 소형 병변을 효율적으로 다루지 못하는 한계를 극복하기 위해, 이 모델은 채널, 공간 및 학습 전략의 세 가지 차원에서 소형 병변에 대한 민감도를 높이고자 합니다. 또한, 여러 잔여 연결을 통해 지역적 특성을 보존하면서 중요 세부사항을 선택적으로 강조하는 Enhanced Visual State Space 블록을 디자인하였습니다.

- **Technical Details**: S$^3$-Mamba는 Enhanced Visual State Space(EnVSS) 블록과 Tensor-based Cross-feature Multi-scale Attention(TCMA)을 포함하여 소형 병변에 대한 세밀한 주의를 기울입니다. EnVSS 블록은 잔여 연결을 활용하여 소형 병변의 국소 특징을 유지하고, 채널별 주의를 통해 각 채널의 기여도를 동적으로 조절합니다. TCMA는 다중 스케일에서 입력 이미지 특징과 중간 레이어 특징을 통합하여 소형 병변의 공간적 세부 사항을 보존합니다.

- **Performance Highlights**: 세 가지 의료 이미지 분할 데이터셋에서 실시된 광범위한 실험을 통해 S$^3$-Mamba의 우수성을 확인하였습니다. 특히 소형 병변의 분할에서 기존 모델보다 더 뛰어난 성능을 보이며, 병변 크기와 샘플 난이도에 따라 동적으로 학습 과정을 조정하는 커리큘럼 학습 방식을 통해 훈련이 진행됨에 따라 모델의 성능이 향상되었습니다. 이 연구는 소형 병변의 분할을 효율적으로 처리할 수 있는 방법을 제시함으로써 의료 이미지 분석 분야에 기여할 것입니다.



### Summary of Point Transformer with Federated Learning for Predicting Breast Cancer HER2 Status from Hematoxylin and Eosin-Stained Whole Slide Images (https://arxiv.org/abs/2412.14545)
- **What's New**: 이 연구는 HE(hematoxylin and eosin)로 염색된 전체 슬라이드 이미지(WSI)에서 HER2(인간 상피세포 성장인자 수용체 2) 상태를 예측하기 위한 분산 학습(federated learning) 기반 접근법을 소개합니다. 이 방법은 비용 절감과 치료 결정 속도를 높이는 데 기여합니다.

- **Technical Details**: 멀티사이트 데이터셋에서 레이블 불균형(label imbalance)과 특성 표현(feature representation) 문제를 해결하기 위해 포인트 변환기(point transformer)가 제안되었습니다. 이 모델은 동적 레이블 분포(dynamic label distribution), 보조 분류기(auxiliary classifier), 그리고 최외각 코사인 샘플링(farthest cosine sampling)을 통합하여 구현됩니다.

- **Performance Highlights**: 네 개의 사이트(2687 WSI)에서 상태-of-the-art 성능을 입증하며, 두 개의 보지 못한 사이트(229 WSI)에 대해서도 강력한 일반화 능력을 보여줍니다.



### DAMPER: A Dual-Stage Medical Report Generation Framework with Coarse-Grained MeSH Alignment and Fine-Grained Hypergraph Matching (https://arxiv.org/abs/2412.14535)
- **What's New**: 본 논문은 DAMPER라는 두 단계의 메디컬 리포트 생성 프레임워크를 소개합니다. 이는 의사의 리포트 작성 과정을 모방하여, 첫 번째 단계에서 메쉬(Medical Subject Headings) 정보를 활용한 거시적 정렬을 진행하고, 두 번째 단계에서는 하이퍼그래프 기반의 미세한 정렬을 통해 의료 이미지를 정밀하게 분석합니다. 이러한 방식은 기존 방법들이 놓치고 있는 임상적 흐름을 효과적으로 반영하며, 자동화된 리포트 생성의 정확성과 포괄성을 향상시킵니다.

- **Technical Details**: DAMPER는 두 개의 주요 단계로 구성되어 있습니다. 첫 번째 단계인 MCG에서는 이미지 특징을 메쉬 정보와 정렬하여 초기 건강 상태를 반영하는 키 특징을 생성합니다. 두 번째 단계인 HFG에서는 하이퍼그래프 구조를 도입하여 이미지 패치와 리포트 구절 간의 복잡한 관계를 모델링하며, 고차원 관계를 유지하면서 정밀한 정렬을 수행합니다. 이러한 모듈들은 적대적 학습과 주의집중 기법을 활용하여 결과의 품질을 높입니다.

- **Performance Highlights**: DAMPER는 IU-Xray 및 MIMIC-CXR와 같은 공개 데이터셋에서 광범위한 실험을 통해 기존의 방법들보다 월등한 성능을 보였습니다. 다양한 평가 지표에서 state-of-the-art 방법들을 능가하며, 의료 리포트 생성의 정확성과 포괄성을 증대시키는 데 기여하고 있습니다. 이 연구는 의료 영상과 자연어 간의 복잡한 관계를 효과적으로 다룰 수 있는 새로운 접근 방식을 제시합니다.



### Consistent Human Image and Video Generation with Spatially Conditioned Diffusion (https://arxiv.org/abs/2412.14531)
Comments:
          Project page: this https URL

- **What's New**: 이번 연구에서는 일관된 인간 중심 이미지 및 비디오 합성을 위해, 참조 이미지와의 일관성을 유지하면서 새로운 포즈를 생성하는 방법을 제안합니다. 기존의 확산 모델(diffusion models)은 참조 이미지와 목표 이미지 간의 도메인 간극(domain gaps) 문제를 파악하고 이를 해결하기 위해 제안한 스페이셜 컨디셔닝(inpainting) 접근법을 사용합니다. 이를 통해 목표 이미지의 품질이 높아지는 동시에 참조 이미지의 특징을 활용할 수 있습니다.

- **Technical Details**: 우리는 목표 이미지가 참조 이미지와의 외관 일관성을 유지하도록 인페인팅(inpainting) 문제로 프레임을 설정합니다. 이 과정에서 두 단계로 나누어 스페이셜 컨디셔닝된 생성 과정을 구현하는 방식으로, 첫 번째 단계에서는 참조 이미지의 외관 특징을 추출하고, 두 번째 단계에서는 조건부(target generation) 생성을 수행합니다. 이 모든 과정은 하나의 단일 노이즈 제거 네트워크(denoising network)에서 이루어지며, 상호작용은 자기 주의(self-attention) 레이어로 제한됩니다.

- **Performance Highlights**: 우리는 기존의 확산 모델을 인간 비디오 데이터에 대해 미세 조정하여 상당히 강력한 일반화를 달성하였으며, 추가적인 인스턴스별 미세 조정 없이도 새로운 인간 정체성과 포즈에 잘 적응하는 방법을 보여주었습니다. 실험 결과는 제안한 방식의 효과성을 검증하며 일관된 인간 이미지 및 비디오 합성에 있어 기존 방법들에 비해 경쟁력 있는 성과를 보였습니다.



### Efficient Self-Supervised Video Hashing with Selective State Spaces (https://arxiv.org/abs/2412.14518)
Comments:
          Accepted by AAAI'25. 9 pages, 5 figures, 2 tables

- **What's New**: 이 연구는 Self-supervised video hashing (SSVH)를 활용하여 비디오 색인화 및 검색을 개선하고자 합니다. 기존의 Transformer 기반 기법이 계산 및 메모리 비효율성 문제를 가지고 있는 반면, Mamba라는 최신 상태 공간 모델에서 영감을 받아 이루어진 이 연구는 효율성과 효능의 균형을 보다 잘 맞추고자 합니다. S5VH라는 새로운 비디오 해싱 모델을 제안하며, 자가 지도 학습(paradigm) 방식을 개선하여 비디오 데이터를 보다 효율적으로 처리할 수 있도록 합니다.

- **Technical Details**: S5VH는 Mamba 기반 비디오 해싱 모델로, bidirectional Mamba layers를 설계하여 인코더와 디코더 모두에서 효과적이고 효율적입니다. 이는 데이터 의존적 선택적 스캐닝 메커니즘을 통해 최적의 temporal relationship를 포착할 수 있도록 설계되었습니다. 자신의 공간(feature space)에서 global semantics를 semantically consistent하고 구별 가능한 해시 중심으로 변환하며, 이를 위한 center alignment loss를 도입하여 글로벌 학습 신호를 제공합니다.

- **Performance Highlights**: 실험 결과 S5VH는 최신 방법들에 비해 성능이 개선되었으며, 전이 가능성과 추론 효율성(scale) 측면에서도 우수한 이점을 보입니다. 또한, self-local-global (SLG) 패러다임을 통해 학습 효율성이 크게 향상되어 더 빠르고 더 나은 수렴(convergence)을 달성할 수 있었습니다. 전체적인 성능 향상은 자가 지도 학습의 최근 발전에 기인하며, 이는 대규모 라벨이 없는 비디오 데이터의 증가와 연결됩니다.



### A Super-pixel-based Approach to the Stable Interpretation of Neural Networks (https://arxiv.org/abs/2412.14509)
Comments:
          BMVC 2024

- **What's New**: 이 논문에서는 gradient 기반의 saliency map의 안정성과 일반성(generality)을 증가시키기 위한 새로운 픽셀 파티셔닝(pixel partitioning) 전략을 제안합니다. 연구에서 제안된 방법은 이미지에서 의미론적 의미(semantic meaning)에 잘 부합하는 super-pixels를 기반으로 한 그룹화(grouping) 전략을 포함합니다. 이를 통해, saliency map의 불필요한 변동성을 감소시키고 해석의 정확성을 높임으로써 더 나은 결과를 도출할 수 있음을 보여줍니다.

- **Technical Details**: 제안된 방법은 saliency map의 픽셀을 여러 그룹으로 나누어, 각 그룹의 픽셀 공식의 평균을 계산하여 맵의 무작위성에 대한 민감도를 감소시킵니다. 이론적으로, 그라디언트 출력의 계산이 고차원(feature space) 환경에서 어떻게 변화하는지를 설명하며, 알고리즘 안정성(algorithmic stability) 프레임워크를 해석 맵의 맥락으로 확장하여 결과적으로 더 균일한 saliency map을 생성하는 데 기여합니다. 실험은 CIFAR-10과 ImageNet 데이터셋을 사용하여 수행되고, super-pixel 기반 방법이 도입됩니다.

- **Performance Highlights**: 경험적 결과에 따르면, super-pixel 기반의 해석 맵은 pixel 기반 saliency map보다 일관되게 더 나은 안정성(stability)과 품질(quality)을 보였습니다. 이러한 결과는 pixel 그룹화가 saliency map에서의 변동성을 줄이는 데 기여하며, 적절한 시각적 품질과 해석 능력을 유지하면서도 신뢰할 수 있는 해석을 제공한다는 점을 보여줍니다. 또한, 이 방법은 알고리즘 안정성을 향상시키고 해석 맵의 추정 오류를 줄이는 데 효과적임을 밝혔다.



### Content-style disentangled representation for controllable artistic image stylization and generation (https://arxiv.org/abs/2412.14496)
- **What's New**: 이 논문은 예술적 이미지 스타일화 및 생성에서 콘텐츠와 스타일을 분리하는 새로운 접근 방식을 제안합니다. WikiStyle+라는 다중 모달 아트 이미지-텍스트 데이터셋을 구축하여, 이를 기반으로 분리된 콘텐츠와 스타일 표현을 사용하는 확산 모델(diffusion model)을 개발하였습니다. 이 방법은 다양한 모달리티에서 입력을 수용할 수 있어, 미학적으로 일관되고 표현력 있는 이미지를 생성하는 데 기여합니다.

- **Technical Details**: 제안된 방법은 Q-Former를 활용하여 스타일과 콘텐츠 표현을 배우고, 학습된 표현을 사전 훈련된 확산 모델에 주입하는 방식으로 작동합니다. 이 과정에서 각기 다른 시간 단계에서 다중 단계 교차 주의 레이어(multi-step cross-attention layer)로 주입함으로써, 시각적 요소의 조정이 가능해집니다. 이를 통해 모델은 콘텐츠와 스타일을 통합하여 더 나은 제어된 스타일화를 구현합니다.

- **Performance Highlights**: 실험 결과, 본 방법은 다중 모달 감독 하에 콘텐츠와 스타일의 철저한 분리를 달성하였습니다. 이로 인해 생성된 이미지에서 콘텐츠와 스타일이 조화롭게 통합되며, 참조 이미지로부터의 콘텐츠 누수를 효과적으로 방지할 수 있습니다. 또한, 다양한 모달리티의 입력을 수용함으로써 이전 방법들에 비해 뛰어난 성능을 발휘합니다.



### Drive-1-to-3: Enriching Diffusion Priors for Novel View Synthesis of Real Vehicles (https://arxiv.org/abs/2412.14494)
- **What's New**: 최근 대규모 3D 데이터의 발전은 현대의 NVS(novel view synthesis) 작업을 가능하게 했지만, 이에 대한 성능 저하는 주로 합성 데이터의 특성에서 기인합니다. 본 논문에서는 자율주행 자동차의 차량 자산 수확을 목표로, 실제 데이터에 대한 사전 훈련 모델의 미세 조정을 위한 유용한 실천법들을 제시합니다. 그것은 실제 주행 데이터와 합성 데이터 간의 불일치를 정리하고 이를 해결하기 위한 몇 가지 전략을 개발합니다.

- **Technical Details**: 주요 전략으로는 실제 이미지의 가상 카메라 회전을 통해 합성 데이터와의 기하학적 정렬을 수행하고, 개체 중심 이미지 패치를 활용한 고정 초점 거리로 다양한 개체 거리 문제를 해결하는 것입니다. 또한, 실 데이터의 불가피한 Occlusions(가림)을 고려하기 위해, 레이턴트 공간에서 가림 마스크를 변환하고 손실 감독을 수행하여 가림된 영역을 무시합니다. 마지막으로, 넓은 뷰 포인트 변화를 처리하기 위해 대칭 우선 사항을 활용하여 훈련 데이터를 확장합니다.

- **Performance Highlights**: 이러한 접근 방식은 Drive-1-to-3라는 도메인 특정 미세 조정 프레임워크를 통해 상당한 성능 향상을 가져왔으며, NVS 분야에서 기존 기술에 비해 FID(Frechet Inception Distance) 점수를 68.8% 줄이는 결과를 도출했습니다. 이러한 성과는 차량이 실제 도로 환경에서 보이는 다양한 특성을 효과적으로 다루었음을 보여주며, 3D 재구성 과제와 가상 물체 삽입 작업에서의 강력한 성능을 입증하였습니다.



### QADM-Net: Quality-adaptive Dynamic Network for Reliable Multimodal Classification (https://arxiv.org/abs/2412.14489)
Comments:
          11 pages, 5 figures

- **What's New**: 이 논문은 다양한 데이터의 품질 변동에 대응하기 위해 네트워크의 용량과 동작을 조정하는 새로운 접근인 Quality-Adaptive Dynamic Multimodal Network (QADM-Net)를 제안합니다. 기존의 정적 네트워크 기반 방법들이 간과했던 네트워크의 역동성과 적응성을 강조합니다. 이 방식은 각 입력 샘플에 대한 신뢰성 있는 출력을 보장하기 위해, 각 샘플의 품질에 기반하여 네트워크 깊이를 조정하는 동적 깊이 메커니즘을 도입합니다.

- **Technical Details**: QADM-Net는 신뢰성 있는 출력을 위한 동적 접근 방식을 제공하며, 각 모달리티의 품질에 따라 네트워크의 깊이를 조정하는 'confidence-guided dynamic depths' 메커니즘을 포함합니다. 또한, 'informativeness-based dynamic parameters' 메커니즘을 통해 각 샘플의 특징 벡터의 품질 향상에 따라 네트워크의 매개변수를 추론하여 샘플에 특화된 행동을 가능하게 합니다. 이러한 메커니즘을 통해 QADM-Net는 각 입력 샘플에 맞는 용량과 행동 구성을 식별합니다.

- **Performance Highlights**: QADM-Net는 네 가지 데이터셋에서 실험을 진행하였으며, 기존의 최신 방법들보다 분류 성능에서 상당한 우위를 보였습니다. 또한 다양한 품질의 데이터에 대한 강력한 적응성을 입증하였습니다. 이러한 결과는 QADM-Net가 안전-critical 응용 프로그램들에 효과적으로 적용될 수 있음을 보여줍니다.



### Token Preference Optimization with Self-Calibrated Visual-Anchored Rewards for Hallucination Mitigation (https://arxiv.org/abs/2412.14487)
- **What's New**: 본 논문에서는 Direct Preference Optimization (DPO)에서 나타나는 할루시네이션(hallucination) 문제를 해결하기 위해 새로운 Token Preference Optimization (TPO) 모델을 제안합니다. TPO는 비문서명 시각적 특징에 대한 세밀한 주석 없이도 시각적으로 상관된 토큰들을 어댑티브하게 최적화할 수 있습니다. 이러한 접근 방식은 출력 결과를 인간의 선호와 더 밀접하게 일치시키는 데 기여하여 LVLMs의 효과성을 높입니다.

- **Technical Details**: TPO모델은 	extit{visual-anchored} 	extit{reward}를 도입하여 원본 이미지와 왜곡된 이미지에 조건화된 생성된 토큰 간의 로지스틱 분포 차이를 보상으로 설정합니다. 이 모델은 시각적 특징에 따라 자동으로 특정 토큰들을 식별하고, 이러한 시각적 상관관계를 기반으로 보상 신호를 자기 조정(Self-calibrated)하여 최적화를 진행합니다. 이를 통해 기존 DPO 방식에 비해 더 세밀한 비주얼 정보를 처리할 수 있습니다.

- **Performance Highlights**: TPO는 기존의 할루시네이션 벤치마크에서 최첨단 성능을 입증하였습니다. LLAVA-1.5-7B 모델에 TPO를 적용하여 성능을 절대적으로 개선하는 결과를 보여주었으며, 이러한 결과는 시각적 정보를 더 잘 반영하는 응답 생성을 가능하게 합니다. 광범위한 실험 결과는 TPO의 유용성과 효율성을 강조합니다.



### DirectorLLM for Human-Centric Video Generation (https://arxiv.org/abs/2412.14484)
- **What's New**: 이번 연구에서 우리는 DirectorLLM이라는 새로운 비디오 생성 모델을 소개합니다. 이 모델은 대규모 언어 모델(LLM)을 활용하여 비디오 내 인간 동작을 조정하는 기능을 갖추고 있습니다. 인간 동작의 진정성을 높이기 위해, 텍스트 생성기에서 비디오 감독 및 인간 동작 시뮬레이터로 LLM을 확장하였습니다.

- **Technical Details**: DirectorLLM은 1초당 1프레임 속도로 인간 자세를 생성하는 전문화된 LLM과 이 희소한 자세를 부드러운 30FPS로 보간하는 선형 확산 모델, 그리고 인간 자세와 텍스트 프롬프트에 따라 비디오를 렌더링하는 VideoCrafter로 구성된 다중 구성 요소 모델입니다. 이 접근 방식은 비디오 생성 과정에서 장면 이해 및 자세 동역학을 분리하며, LLM이 이러한 작업을 처리하여 비디오 생성기가 시각적으로 정확한 프레임을 생성하는 데 집중할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, DirectorLLM은 기존 비디오 생성 모델에 비해 인간 동작 충실도, 프롬프트 신뢰도 및 렌더링된 주체의 자연스러움에서 우수한 성능을 보여주었습니다. 이 모델은 다양한 비디오 렌더러에 적용할 수 있으며, 최소한의 노력으로 높은 품질의 인간 중심 비디오를 생성할 수 있는 가능성을 제공합니다.



### MegaPairs: Massive Data Synthesis For Universal Multimodal Retrieva (https://arxiv.org/abs/2412.14475)
- **What's New**: 이 논문에서는 MegaPairs라는 새로운 데이터 합성 방법을 소개하고, 이를 사용해 생성된 대규모 지침 데이터셋을 발표합니다. MegaPairs는 비전 언어 모델(VLMs)과 개방형 이미지들을 활용하여 훈련 데이터를 생성하며, 전통적인 방법의 한계를 극복하고, 더 나은 데이터 품질과 다양성을 제공합니다. 이 접근법은 고품질 데이터를 생성하고, 제공된 기존 데이터셋보다 70배 더 적은 데이터로도 성능을 향상시킵니다.

- **Technical Details**: MegaPairs는 이종 KNN 삼중 트리플렛을 구성하는 독특한 방식으로 설계되었습니다. 이 방법은 CLIP 비전 인코더와 DINO 비전 인코더, 그리고 CLIP 텍스트 인코더를 사용하여 상관 관계가 있는 이미지 쌍을 샘플링합니다. 생성된 이미지 쌍은 비전 언어 모델(VLM)과 대형 언어 모델(LLM) 주석 작성자에 의해 설명되고, 이를 바탕으로 다량의 가상 검색 지침이 생성됩니다.

- **Performance Highlights**: MegaPairs를 통해 생성된 데이터는 기존 데이터셋보다 우수한 품질을 달성하였으며, 26백만개의 데이터 인스턴스를 생산하였습니다. 실험 결과, MegaPairs로부터 추출한 50만개 인스턴스만으로도 MagicLens의 3,670만 인스턴스보다 나은 성능을 보여주었습니다. 새로 훈련된 MMRet 모델은 4개의 인기 있는 이미지 검색 기준에서 최첨단 성능을 달성하였고, 추가적인 세부 조정을 통해 성능이 더욱 향상되었습니다.



### Promptable Representation Distribution Learning and Data Augmentation for Gigapixel Histopathology WSI Analysis (https://arxiv.org/abs/2412.14473)
Comments:
          Accepted by AAAI2025

- **What's New**: 본 논문에서는 PRDL(Promptable Representation Distribution Learning)이라는 새로운 프레임워크를 제안하여 패치 수준의 표현 학습 및 전체 슬라이드 이미지(WSI) 데이터 증강 문제를 해결하고자 합니다. 이 방법은 고정된 패치 표현 대신 각 패치에 대한 잠재적 표현 증강 분포를 예측할 수 있는 추정기를 설계하여, 이미지 공간에서의 데이터 증강을 효과적으로 수행할 수 있도록 합니다. 실험 결과, 제안한 방법이 기존의 최첨단 기법들보다 우수한 성능을 보였음을 입증하였습니다.

- **Technical Details**: PRDL 프레임워크는 자가 지도 학습(self-supervised learning)을 기반으로 하여, 데이터 증강을 위한 표현 분포 추정기를 학습합니다. 이 방법은 각 패치의 표현을 정적 포인트가 아닌 개별 분포로 나타내고, 특정 증강 프롬프트에 따라 표현 차원의 범위를 조절하여 이미지 공간에서의 다양한 증강 작업을 모사합니다. 또한, 이 플로우를 통해 각 패치로부터 온라인으로 샘플링하여 비모수적 표현 증대 과정을 수행함으로써 모델 훈련의 효율성을 높입니다.

- **Performance Highlights**: 본 논문에서 제안한 PRDL 모델은 754개의 WSI와 두 개의 공공 폐 데이터셋에서 실험을 진행하여, 기존의 방법들과 비교해 안정적으로 우수한 성능을 보여주었습니다. PRDL은 기존의 전통적인 이미지 수준의 데이터 증강보다 훨씬 더 폭넓은 증강을 가능하게 해주며, 이는 패치 표현의 구별력을 크게 향상시켜 WSI 분석 모델의 성능을 개선하는 데 기여합니다. 이 연구 결과들은 패치 인코딩 과정과 데이터 증강 과정 간의 효율적인 상호작용을 가능하게 하는 새로운 접근 방식을 보여줍니다.



### DiffusionTrend: A Minimalist Approach to Virtual Fashion Try-On (https://arxiv.org/abs/2412.14465)
- **What's New**: DiffusionTrend는 가상 패션 착용 기술을 위한 새로운 접근법으로, diffusion 모델의 재훈련이 필요하지 않습니다. 이 모델은 고급 diffusion 모델을 활용하여 의복의 세부 사항을 효율적으로 캡처하고, 이를 경량의 CNN을 통해 생성된 의류 마스크와 결합해 효과적으로 이미지 생성 과정에 통합합니다. 잦은 재훈련 기간 없이도 시각적으로 매력적인 착용 경험을 제공함으로써, 패션 산업 내 여러 가능성을 열어주는 의미 있는 연구입니다.

- **Technical Details**: DiffusionTrend는 기존의 리소스 집약적인 diffusion 모델의 재훈련 필요를 없애고, 각종 복잡한 모델 입력 없이도 작동합니다. 의류 세밀화를 캡처하기 위해 DDIM inversion에서 생성된 latents을 사용하며, 초기 diffusion denoising 과정에서 모델 이미지 생성과 의류 이미지의 세밀한 결합을 통한 효과적 결과를 도출합니다. 이러한 접근은 고비용의 계산 인프라에 대한 의존도를 줄이고, 사용자에게 보다 쉽게 접근할 수 있는 솔루션을 제공합니다.

- **Performance Highlights**: DiffusionTrend는 초기에 부차적인 지표 성능을 보이지만, 그럼에도 불구하고 resource-intensive 한 데이터셋 훈련 요구를 제거하며, 효율적인 파이프라인을 구현할 수 있는 가능성을 보여줍니다. 이는 가상 착용 기술에서의 발전을 위한 중요한 기초를 마련하며, future research에서 training-free 접근 방식의 잠재력을 강조하는 데 기여합니다. 또한, 시각적으로 매력적인 착용 경험을 통해 diffusion 모델의 가능성을 시사합니다.



### LiftRefine: Progressively Refined View Synthesis from 3D Lifting with Volume-Triplane Representations (https://arxiv.org/abs/2412.14464)
- **What's New**: 이번 논문에서는 단일 또는 소수의 입력 이미지를 통해 3D 신경 필드를 합성하는 새로운 뷰 합성(view synthesis) 방법을 제안합니다. 이 방법은 이미지에서 3D로의 생성 문제의 정의되지 않은 성질을 해결하기 위해, 재구성 모델(reconstruction model)과 확산 모델(diffusion model)을 포함한 두 단계의 방법을 개발했습니다. 제안된 방법은 시각적으로 향상된 뷰를 생성하며, 기존 방법들과 비교하여 우수한 성능을 보여줍니다.

- **Technical Details**: 첫 번째 단계에서는 입력 이미지를 3D 공간으로 피치하게 하기 위한 새로운 재구성 모델을 제안합니다. 이 모델은 볼륨 표현(volume representation)과 트라이플레인(tri-plane) 표현의 특징을 통합하여, 3D 재구성 능력을 향상시킵니다. 두 번째 단계에서는 이미지 기반의 확산 모델을 활용하여, 초기 뷰와 목표 뷰 각도에서 생성된 새로운 뷰의 품질을 점진적으로 향상시킵니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 합성 SRN-Car 데이터셋, CO3D 데이터셋 및 Objaverse 데이터셋에서 최첨단 방법들보다 월등한 성능을 보였습니다. 이러한 결과는 샘플링 효율성과 다중 뷰 일관성을 동시에 달성하며, 3D 표현의 품질을 크게 향상시키는 것을 보여줍니다. 따라서 이 연구는 뷰 합성 분야에서 새로운 가능성을 제시하고 있습니다.



### Affordance-Aware Object Insertion via Mask-Aware Dual Diffusion (https://arxiv.org/abs/2412.14462)
Comments:
          Code is available at: this https URL. Project page at: this https URL

- **What's New**: 이번 논문에서는 이미지 편집 작업인 이미지 합성의 신선한 접근 방식을 제안합니다. 인간 중심의 이미지 합성 작업에서 'Affordance' 개념을 확장하여 객체-장면 합성 프레임워크로 적용하였습니다. 새로운 접근방식인 어포던스 인식 객체 삽입 작업을 정의하고, 3천 개 이상의 객체 카테고리로 구성된 300만 개의 예제 데이터를 포함하는 SAM-FB 데이터세트를 구축했습니다.

- **Technical Details**: 제안된 Mask-Aware Dual Diffusion (MADD) 모델은 RGB 이미지와 삽입 마스크를 동시에 디노이즈하는 이중 스트림 아키텍처를 활용합니다. 이 모델은 diffusion 과정에서 삽입 마스크를 명시적으로 모델링하여 어포던스 개념을 효과적으로 적용합니다. MADD는 다양한 위치 프롬프트를 처리할 수 있으며, 사용자가 명시적으로 제공하지 않은 경우에도 배경과 전경의 의미적 내용을 분석하여 자율적으로 적절한 삽입 위치를 결정할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 기존 최첨단 방법들과 비교하여 성능이 우수함을 보여주고 있습니다. 특히 자연 이미지에 대한 강력한 일반화 성능을 발휘하여, 다양한 객체를 다른 장면에 자연스럽게 삽입할 수 있습니다. SAM-FB 데이터세트와 MADD 모델을 활용함으로써, 보다 현실적인 객체 삽입을 달성할 수 있는 기틀이 마련되었습니다.



### LEDiff: Latent Exposure Diffusion for HDR Generation (https://arxiv.org/abs/2412.14456)
- **What's New**: 본 논문에서는 기존의 저역 동적 범위(LDR) 콘텐츠를 고역 동적 범위(HDR) 콘텐츠로 변환하고 생성하는 새로운 방법인 LEDiff를 소개합니다. 일반적으로 생성 모델이 HDR 이미지를 생성할 수 없다면, LEDiff는 이미 존재하는 LDR 이미지를 HDR로 변환함으로써 고급 기능을 추가합니다. 이를 통해 사진의 세부사항과 색상을 더욱 생동감 있게 재현할 수 있습니다.

- **Technical Details**: LEDiff는 이미지 공간 노출 융합(image-space exposure fusion) 기술에서 영감을 받아 잠재 공간 융합(latent space fusion)을 사용하여 HDR 콘텐츠 생성을 가능하게 합니다. 이 방법은 기존의 저역 동적 범위 이미지를 다루기 위해 소규모 HDR 데이터셋을 활용하여 사전 학습된 확산 모델의 세부정보와 동적 범위를 회복합니다. 또한, 이 과정에서 하이라이트 및 그림자 제너레이터를 학습하여 누락된 세부정보를 보완합니다.

- **Performance Highlights**: LEDiff는 생성 모델에게 HDR 생성 기능을 제공하여 다양한 분야에서 활용 가능성을 크게 확장합니다. 이 방법은 기존의 LDR 이미지를 HDR로 성공적으로 변환하며, 클립된 영역의 세부정보를 복원하는 데 있어 최신 상태의 성능을 보입니다. 안정적인 확산 모델을 통해 이 접근 방식의 효율성과 품질이 높아지며, 다양한 이미지 기반 조명 및 사진 효과 시나리오에서 활용될 수 있습니다.



### Multimodal Latent Diffusion Model for Complex Sewing Pattern Generation (https://arxiv.org/abs/2412.14453)
Comments:
          Our project page: this https URL

- **What's New**: 이 논문에서는 SewingLDM이라는 새로운 다중 모드 생성 모델을 제안하여 섬유 디자인에서 재봉 패턴 생성을 손쉬운 텍스트 프롬프트, 신체 모양 및 의복 스케치를 통해 조정할 수 있게 합니다. 기존의 방법들은 복잡한 의류 디자인을 생성하는 데 어려움을 겪었지만, SewingLDM은 이러한 문제를 해결하고자 하였습니다. 이 프레임워크는 복잡한 디자인을 포괄할 수 있는 다양한 입력을 처리할 수 있는 능력을 제공합니다.

- **Technical Details**: SewingLDM은 재봉 패턴 생성을 위해 고안된 모델로, 세분화된 세부사항을 포함하고 다양한 신체 형태에 적응할 수 있는 복합적인 재봉 패턴을 생성합니다. 초기에 우리는 기존의 섬유 패턴 벡터를 보다 포괄적인 표현으로 확장했고, 이를 컴팩트한 잠재 공간(latent space)으로 변환하여 훈련의 용이성을 높였습니다. 모델은 다단계 훈련 전략을 통해 다중 모드 조건인 텍스트, 신체 모양, 의류 스케치를 주입하여, 생성된 의복이 신체에 적합할 수 있도록 합니다.

- **Performance Highlights**: SewingLDM은 정교한 의류 디자인과 다양한 신체 형태에 대한 적응력에서 기존 방법들보다 우수한 성능을 보여줍니다. 실험 결과, 이 모델은 복잡한 의복 디자인을 효과적으로 생성하며, CG 파이프라인에 원활하게 통합될 수 있습니다. 제공된 텍스트 설명이나 스케치에 부합하는 의복을 생성할 수 있는 능력은 개인 맞춤형 의류 디자인을 가능하게 합니다.



### Color Enhancement for V-PCC Compressed Point Cloud via 2D Attribute Map Optimization (https://arxiv.org/abs/2412.14449)
Comments:
          IEEE VCIP 2024

- **What's New**: 이 논문은 비디오 기반 포인트 클라우드 압축(V-PCC)에서 색상 품질을 향상시키기 위한 새로운 프레임워크를 제안합니다. 기존의 손실 압축 방식은 색상 속성을 저하시킬 수 있는 아티팩트를 도입하는데, 이를 개선하기 위해 경량화된 디컴프레션 유넷(LDC-Unet)을 활용합니다.

- **Technical Details**: 제안된 LDC-Unet은 2D 신경망으로, V-PCC 인코딩 과정에서 생성되는 투영 맵(projection maps)을 최적화합니다. 최적화된 2D 맵은 3D 공간으로 다시 투영되어 포인트 클라우드 속성을 향상시키며, 이 과정에서 전이 학습(transfer learning) 전략도 적용됩니다.

- **Performance Highlights**: 실험은 공개 데이터셋인 8i voxelized full bodies long sequences(8iVSLF)에서 수행되었으며, 제안된 방법이 색상 품질을 효과적으로 개선함을 보여주었습니다. 이러한 접근법은 포인트 클라우드 훈련 데이터의 부족 문제를 해결하는 데 기여합니다.



### VLM-AD: End-to-End Autonomous Driving through Vision-Language Model Supervision (https://arxiv.org/abs/2412.14446)
- **What's New**: 본 논문에서는 VLM-AD라는 새로운 방법을 제안하여 비전-언어 모델(VLMs)을 사용하여 자율 주행 모델의 훈련을 개선합니다. 기존의 End-to-End (E2E) 모델들이 데이터에서 관찰된 주행 패턴만을 모방하는 데 그쳤던 한계를 극복하고, 비구조적 추론 정보와 구조적 행동 라벨을 통합하여 훈련 과정에서 추가적인 감독 정보를 제공합니다. 이로 인해 모델은 주행 패턴 뒤에 있는 합리적 근거를 학습할 수 있는 능력이 향상됩니다.

- **Technical Details**: VLM-AD는 여러 뷰의 이미지 시퀀스와 자율 차량의 미래 궤적을 입력으로 받아, 이를 바탕으로 VLM에 특정 질문을 던져 행동에 대한 정보와 추론 과정을 포함한 주석을 생성합니다. 이 주석은 기존의 궤적 레이블을 넘어서는 보조 감독 신호로 작용하며, 고품질의 VLM 생성 주석을 포함한 데이터 세트를 구축하는 데 필요한 모든 과정을 효과적으로 통합하는 플러그 앤 플레이 방식의 보조 작업을 설계했습니다. 이 방법은 추론 시 VLM이 필요하지 않아 실제 배포에 적합합니다.

- **Performance Highlights**: nuScenes 데이터셋에서 VLM-AD를 적용한 결과, L2 계획 오류가 각각 14.6%와 33.3% 줄어들었으며, 충돌률은 UniAD와 VAD의 경우 각각 38.7%와 57.4% 감소하는 등 성능에서의 주요 개선 효과를 확인했습니다. 이러한 결과는 VLM-AD가 기존 E2E 자율 주행 모델에 비해 주행 성능을 크게 향상시킬 수 있는 잠재력을 가지고 있음을 보여줍니다.



### GenHMR: Generative Human Mesh Recovery (https://arxiv.org/abs/2412.14444)
- **What's New**: 이번 논문에서는 GenHMR이라는 새로운 생성 프레임워크를 소개하여 단일 2D 이미지로부터 3D 인간 메쉬(Human Mesh Recovery, HMR)를 보다 정확하게 복원할 수 있도록 합니다. GenHMR은 2D-3D 매핑 과정에서 불확실성을 명시적으로 모델링하고 완화하는 데 중점을 두고 있습니다. 이 프레임워크는 포즈 토크나이저와 이미지 조건 마스크 변환기(image-conditional masked transformer)라는 두 가지 핵심 구성 요소로 구성되어 있습니다.

- **Technical Details**: GenHMR의 첫 번째 단계에서는 Vector Quantized Variational Autoencoders (VQ-VAE)를 사용하여 지속적인 인간 포즈를 잠재 공간에서의 이산 토큰 시퀀스로 변환합니다. 그 후 두 번째 단계에서는 포즈 토큰 시퀀스의 일부를 무작위로 마스킹하고, 이미지에 조건화된 확률 분포를 학습하여 마스킹된 토큰을 예측하는 훈련을 진행합니다. 이러한 생성적 마스킹 훈련을 통해 GenHMR은 2D 이미지에서 인간 포즈로의 확률적 매핑을 학습합니다.

- **Performance Highlights**: 실험 결과, GenHMR은 Human3.6M, 3DPW 및 EMDB와 같은 표준 데이터셋에서 기존 최첨단(SOTA) 방법보다 20-30%의 오차 감소(MPJPE 기준)를 달성하여 뛰어난 성능을 입증했습니다. 또한 GenHMR은 모호한 이미지 관찰에 강건하며, 복잡한 시나리오에서도 뛰어난 성능을 보여주어 HMR 분야에 중요한 기여를 하고 있습니다.



### IntroStyle: Training-Free Introspective Style Attribution using Diffusion Features (https://arxiv.org/abs/2412.14432)
Comments:
          16 pages, 17 figures

- **What's New**: 이 논문은 스타일 속성 부여 문제를 해결하기 위해 훈련이 필요 없는 새로운 프레임워크인 introspective style attribution (IntroStyle)을 제안합니다. 기존 방법이 요구하는 데이터 세트 수집과 모델 훈련의 자원 집약적 문제를 피하면서, 확산 모델의 특성만을 사용하여 스타일을 효과적으로 구별할 수 있음을 보여줍니다. 또한, 새로운 합성 데이터 세트인 Style Hacks (SHacks)를 소개하여 예술 스타일을 격리하고 세밀한 스타일 속성 부여 성능을 평가합니다.

- **Technical Details**: IntroStyle은 확산 모델의 노이즈 제거 네트워크를 오토인코더로 구성하여 이미지의 스타일 특성을 분석함으로써 이미지 스타일을 구별하는 간단하고 효과적인 접근 방식을 제안합니다. 이는 복잡한 외부 모델이나 재훈련 없이 가능합니다. 이 연구는 다양한 스타일 검색 데이터 세트에서 기존 모델들과 비교했을 때, IntroStyle이 월등한 성능을 보인다는 것을 실험적으로 입증하였습니다.

- **Performance Highlights**: 제안된 모델은 Style Hacks (SHacks) 데이터 세트에 대한 검색 결과에서 최신 접근 방식보다 우수한 성능을 보여줍니다. 이 스타일 속성 부여 접근 방식은 참조와의 유사성에 따라 이미지를 정렬하는 능력을 향상시키며, 특정 스타일에 유사한 이미지를 생성하는 것을 방지하기 위한 스타일 기반 거부 구현이 가능합니다.



### WildSAT: Learning Satellite Image Representations from Wildlife Observations (https://arxiv.org/abs/2412.14428)
- **What's New**: 이 논문에서는 WildSAT이라는 새로운 접근 방식을 소개하며, 이 방법은 위성 이미지의 표현 학습에 있어 종의 분포가 제공하는 지도 신호(supervision signal)를 활용합니다. WildSAT은 시민 과학 플랫폼에서 쉽게 접근할 수 있는 수백만 개의 지오 태그된 야생동물 관찰 기록과 위성 이미지를 결합하여 정보를 구축합니다. 이 연구는 다양한 하류 위성 이미지 인식 작업에서 무작위로 초기화된 모델과 ImageNet 또는 특화된 위성 이미지 데이터셋에서 미리 학습된 모델 모두의 성능을 크게 향상시킵니다.

- **Technical Details**: WildSAT은 대비 학습(contractive learning) 프레임워크를 사용하여 종 분포 맵과 생태계 세부 사항을 포착하는 텍스트 설명을 결합합니다. 이 접근 방식은 같은 지역에서 온 위성 이미지, 텍스트, 위치의 특징 임베딩(feature embedding)을 밀접하게 정렬시키고 다른 지역에서의 임베딩은 멀어지도록 하여, 종의 선호 서식지를 활용하여 위성 이미지 표현을 개선합니다. 또한, WildSAT은 제로샷 검색(zero-shot retrieval) 기능을 통해 일반적인 장소 설명에 기반하여 검색할 수 있습니다.

- **Performance Highlights**: 실험 결과 WildSAT은 기존의 다른 크로스 모달 감독 방법들과 비교하여 더 나은 표현력을 보여주며, 다양한 모델과 다운스트림 원격 감지(task)에서 기존 방법들과 상호 보완적인 결과를 도출했습니다. 또한 본 연구는 WildSAT의 구성 요소 각각이 하류 성능에 미치는 영향을 분석하여 여러 디자인 선택의 효과를 강조하였습니다. 이러한 성과는 기존의 방법들에 비해 WildSAT의 일반적인 적용 가능성을 나타냅니다.



### FedPIA -- Permuting and Integrating Adapters leveraging Wasserstein Barycenters for Finetuning Foundation Models in Multi-Modal Federated Learning (https://arxiv.org/abs/2412.14424)
Comments:
          Accepted for publication in AAAI 2025 (Main Track)

- **What's New**: 이 논문에서는 헬스케어 환경에서 데이터 프라이버시와 컴퓨팅 리소스 제한을 고려하여, FedPIA(Federated Learning via Permuting and Integrating Adapters)라는 새로운 프레임워크를 제안합니다. 이는 Parameter-Efficient Fine-Tuning (PEFT)과 Federated Learning (FL) 접근 방식을 결합한 방법입니다. 이 방법은 클라이언트와 서버 간의 로컬 및 글로벌 어댑터를 효율적으로 통합하여 성능을 개선하고, 데이터 및 작업의 이질성을 극복하는데 초점을 맞추고 있습니다.

- **Technical Details**: FedPIA 프레임워크는 두 가지 접근 방식을 사용하여 클라이언트 어댑터와 서버의 글로벌 어댑터를 통합합니다. 첫째, 서버에서 다양한 클라이언트 어댑터 뉴런을 Permutation(순열)하여 글로벌 어댑터 뉴런과 일치시키고, 둘째, 클라이언트의 글로벌 어댑터를 클라이언트 특정 어댑터에 가까워지도록 재배열한 후 결합합니다. 이러한 과정은 Wasserstein barycenters 이론을 활용하여 수행되며, 그 결과로 안정적인 수렴을 보여줍니다.

- **Performance Highlights**: 2000개가 넘는 클라이언트 레벨 실험을 통해, FedPIA는 시각적 질의 응답(Visual Question Answering) 및 이미지 및 보고서 기반의 다중 라벨 질병 감지와 같은 다양한 의료 이미지 데이터셋과 작업 조건에서 일관되게 뛰어난 성능을 나타냈습니다. FedPIA는 모든 작업 시나리오에서 기존의 PEFT-FL 기법을 초과하며, 이질성 조건에도 불구하고 강력한 성능을 유지하고 있음을 입증했습니다.



### Enhancing Diffusion Models for High-Quality Image Generation (https://arxiv.org/abs/2412.14422)
- **What's New**: 이번 연구에서는 최첨단 생성 모델인 Denoising Diffusion Probabilistic Models (DDPMs)와 Denoising Diffusion Implicit Models (DDIMs)의 포괄적인 구현, 평가 및 최적화가 다루어졌습니다. 이 모델들은 무작위 노이즈를 입력으로 받아 고품질 이미지를 순차적으로 생성하는 방식으로, 이러한 기능을 향상시키기 위한 다양한 기술이 통합되었습니다. Classifier-Free Guidance (CFG), Variational Autoencoders (VAE)를 통한 Latent Diffusion Models, 대체 노이즈 스케줄링 전략이 포함되어 있어 효율적이고 확장 가능한 생성 AI 모델에 대한 수요를 충족시키고자 하였습니다.

- **Technical Details**: 이 논문에서는 DDPM들이 Gaussian 노이즈를 추가하여 원본 이미지를 복원하는 과정에 집중하고 있습니다. 모델 훈련 시 입력으로 RGB 이미지에 Gaussian 노이즈가 점진적으로 추가되고, 이를 통해 노이즈를 역으로 제거하면서 원본 이미지를 복원하는 방법을 학습합니다. 특히, DDIM과 CFG와 같은 고급 방법론을 통합하여 계산 효율성을 극대화하고, 특정 작업 및 데이터셋에 맞춘 노이즈 스케줄링 최적화 작업도 수행하고 있습니다.

- **Performance Highlights**: 평가 결과, DDIM + CFG 조합이 빠른 추론 속도와 우수한 이미지 품질을 달성함을 보여주어, 실제 응용 분야에서 널리 활용될 잠재력을 가지고 있습니다. CIFAR-10 및 ImageNet-100 데이터셋에서의 평가를 통해 이미지 품질 지표인 Frechet Inception Distance (FID)를 사용하여 모델의 성능을 분석하였습니다. VAE와 노이즈 스케줄링의 도전 과제가 강조되어 향후 최적화를 위한 기회를 제시하였습니다.



### An Immersive Multi-Elevation Multi-Seasonal Dataset for 3D Reconstruction and Visualization (https://arxiv.org/abs/2412.14418)
Comments:
          4 pages, 3 figures

- **What's New**: 본 논문에서는 Johns Hopkins University Homewood 캠퍼스를 다양한 계절과 시간대, 고도에서 촬영한 이미지를 포함한 대규모 데이터셋을 소개합니다. 기존 데이터셋은 작고 제한적인 변형으로 인해 표현력이 부족한 점을 보완하여, 연구자들이 다양한 조명 변화와 시점에서 생길 수 있는 문제를 본격적으로 탐구할 수 있도록 합니다. 이 데이터셋은 고해상도의 이미지를 다각도로 제공하여 더욱 현실적인 장면 복원을 가능하게 합니다.

- **Technical Details**: 데이터셋에는 12,300개 이상의 이미지가 포함되어 있으며, 각 이미지에는 고도, 계절, 시간대에 따른 다양한 조명이 반영되어 있습니다. 데이터 수집 과정에서는 스마트폰과 드론을 사용하여 건물 주변을 걷고, 드론으로 공중에서 촬영하였습니다. 이 과정에서 다중 관점을 고려한 이미지 등록을 위한 다단계 캘리브레이션 프로세스를 적용하였습니다.

- **Performance Highlights**: 이 데이터셋은 사진 문맥에서 시뮬레이션된 조건에서만 평가되었던 기존 방법들의 한계를 극복하는 데 기여할 수 있습니다. 복잡한 구조물에서 발생할 수 있는 시각적 모호성을 해결하고, 다중 시점에서의 평가를 통해 알고리즘의 강인성을 평가합니다. 특히, 이러한 특성 덕분에 알고리즘의 성능을 실제 환경에서 더욱 현실적으로 검증할 수 있게 됩니다.



### Enhancing Fingerprint Recognition Systems: Comparative Analysis of Biometric Authentication Algorithms and Techniques for Improved Accuracy and Reliability (https://arxiv.org/abs/2412.14404)
- **What's New**: 이 연구는 지문 인식의 정확성과 강인성을 향상시키기 위해 심층 학습을 사용한 Convolutional Neural Networks (CNN)와 Gabor 필터의 통합 가능성을 조사합니다. 실험은 Sokoto Coventry Fingerprint Dataset에서 다양하게 수집한 데이터를 활용하여 진행되었습니다. CNN 기반 접근 방법의 전반적인 정확도는 94%에 달하며, 이는 기존의 전통적인 특징 추출 방식보다 우수한 성능을 보여줍니다.

- **Technical Details**: 연구 방법론은 데이터셋의 전처리, Gabor 필터를 통한 특징 추출, 그리고 CNN 아키텍처를 포함합니다. CNN 모델은 3개의 컨볼루션 레이어로 구성되며, 각각 max-pooling 레이어가 뒤따라 특징을 추출합니다. 다양한 분류 알고리즘의 성능을 평가하기 위해 정확도, 정밀도, 재현율 및 F1 점수와 같은 평가 지표를 사용하였습니다.

- **Performance Highlights**: 다양한 실험을 통해 CNN 및 Gabor 필터 통합 방법이 지문 인식 성능을 크게 향상시키는 것으로 나타났습니다. 특히, 'Hard' 카테고리에서 F1-score 0.97을 기록하며 높은 정확성을 보였습니다. 이러한 결과는 전통적인 지문 인식 방법의 한계를 극복하고, 기존 방법보다 더 나은 정확성과 안정성을 제공하는 심층 학습 기법의 가능성을 제시합니다.



### HA-RDet: Hybrid Anchor Rotation Detector for Oriented Object Detection (https://arxiv.org/abs/2412.14379)
Comments:
          Bachelor thesis

- **What's New**: 이 논문에서는 항목 기반(anchor-based) 및 항목 비기반(anchor-free) 방식의 장점을 결합한 하이브리드 앵커 회전 검출기(Hybrid-Anchor Rotation Detector, HA-RDet)를 제안합니다. HA-RDet는 특성 맵(feature map)상의 각 위치에 대해 단일 미리 설정된 앵커를 사용하여 처리 속도를 높이고 훈련 샘플 수를 효율적으로 관리합니다. 또한 방향 인식을 고려한 합성곱(Orientation-Aware Convolution) 기술을 도입하여 정확도를 유지하면서도 연산 비용을 크게 줄이는 성과를 보였습니다.

- **Technical Details**: HA-RDet는 두 가지 주요 기술 방향을 통해 성능을 향상시킵니다. 첫 번째는 HA-RDet가 단일 앵커를 사용하여 높은 품질의 제안(Proposal)을 생성하는 것입니다. 두 번째는 방향 인식 합성곱 기술을 통해 객체의 형태와 방향에 적응하여 제안의 품질을 향상시키는 것입니다. 이러한 접근 방식은 고정된 수신 영역(receptive fields)과 축에 정렬된 기존 합성곱 기술의 한계를 극복하여 다양한 방향 및 크기에서 객체를 효과적으로 탐지할 수 있습니다.

- **Performance Highlights**: HA-RDet는 DOTA-v1, DIOR-R, HRSC2016과 같은 다양한 표준 데이터셋에서 각각 75.41 mAP, 65.3 mAP, 90.2 mAP의 성능을 기록하며 기존의 최첨단 앵커 기반 방법들과 경쟁할 만한 정확도를 달성했습니다. 또한 HA-RDet는 경량화된 네트워크 구조로 최종 R-CNN 헤드에서 회전된 제안을 추출하여 훈련과 테스트 효율성을 동시에 향상시켰습니다. 이 논문은 향후 연구의 강력한 기초를 제공하는 우아하고 효율적인 탐지 방법론을 제시합니다.



### SEREP: Semantic Facial Expression Representation for Robust In-the-Wild Capture and Retargeting (https://arxiv.org/abs/2412.14371)
- **What's New**: 이번 논문에서는 SEREP(Semantic Expression Representation)라는 새로운 모델을 제안하여 자연 환경에서 단일 카메라로 캡처된 얼굴 성능을 더 효과적으로 포착하는 방법을 소개합니다. SEREP는 facial expression(얼굴 표현)과 identity(정체성)를 의미적으로 분리하여 모델링하며, 이를 통해 다양한 얼굴 형태와 표정을 포착할 수 있도록 설계되었습니다. 또한 MultiREX라는 새로운 벤치마크를 통해 얼굴 표현 캡처 작업을 평가할 수 있는 자원을 제공합니다.

- **Technical Details**: SEREP는 사이클 일관성 손실(cycle consistency loss)을 이용하여 비매칭된 3D 얼굴 표현에서 표현 표현을 학습합니다. 이후, 단일 이미지를 사용하여 표현을 예측하는 모델을 반감독(semi-supervised) 방식으로 훈련합니다. 이 모델은 domain adaptation을 기반으로 하여, 실생활의 다양한 환경에서 얼굴 성능을 캡처하는 데 있어 중요한 비약을 이룹니다.

- **Performance Highlights**: 실험 결과, SEREP는 기존의 최첨단 방법들보다 뛰어난 성과를 보여주었으며, 특히 비대칭 얼굴 표정이나 새로운 정체성으로의 전이와 관련된 도전적인 표현을 더 잘 포착할 수 있었습니다. MultiREX는 분명하고 다양한 표정 평가를 가능하게 하며, SEREP는 이러한 표정 전이에 있어 정체성을 효과적으로 보존할 수 있음을 입증했습니다.



### Surrealistic-like Image Generation with Vision-Language Models (https://arxiv.org/abs/2412.14366)
Comments:
          2023 Joint international Scientific conferences on AI and Machine Learning (BNAIC-BeNeLearn)

- **What's New**: 최근 생성 AI의 발전으로 텍스트, 이미지 및 코드 등 다양한 유형의 콘텐츠를 손쉽게 생성할 수 있게 되었습니다. 이 논문에서는 DALL-E, Deep Dream Generator, DreamStudio 등의 비전-언어 생성 모델을 사용하여 초현실주의 화풍의 이미지를 생성하는 방법을 탐구합니다. 다양한 이미지 생성 설정과 모델을 통해 생성된 이미지의 품질을 평가하고, 편집된 기본 이미지가 결과 이미지에 미치는 영향을 이해하고자 합니다.

- **Technical Details**: 인공지능(AI)이 텍스트로부터 이미지를 생성하는 능력은 최근 몇 년 동안 활발히 연구되고 있는 분야입니다. 연구의 주요 초점은 텍스트-투-이미지 생성(text-to-image generation)과 현실적 이미지의 초현실주의 화풍 변환을 포함합니다. 본 연구에서 제안된 '초현실적 이미지(surrealistic-like image)'는 드림 같은 분위기, 예상치 못한 대조, 상징성과 은유 등을 포함한 특징을 기반으로 정의됩니다.

- **Performance Highlights**: 실험 결과, DALL-E 2는 ChatGPT가 생성한 프롬프트를 사용할 때 가장 뛰어난 성능을 보여주었습니다. 연구를 통해 235개의 이미지가 생성되었고, 여기에 대한 평가가 이루어졌습니다. 본 논문은 이미지를 생성하기 위한 최적의 설정을 탐색하고 초현실적 이미지를 만드는 과정에 대한 인사이트를 제시합니다.



### Dynamic semantic VSLAM with known and unknown objects (https://arxiv.org/abs/2412.14359)
- **What's New**: 본 논문은 동적 객체를 인식할 수 있는 새로운 기능 기반의 Semantic VSLAM 시스템을 소개합니다. 기존의 VSLAM은 주로 정적인 환경을 가정하지만, 이 연구는 알려지지 않은 객체를 포함한 동적 환경에서도 기능을 확장합니다. 제안된 시스템은 비지도(segmentation) 분할 네트워크를 활용하여 라벨이 없는 세그멘테이션을 달성하며, 이를 통해 동적인 특징을 감지할 수 있습니다.

- **Technical Details**: 제안된 방법은 최신 비지도(segmentation) 모델인 Fast-SAM을 사용하여 모든 객체의 세그멘테이션을 진행하고, 이를 Object Detection 모델과 결합하여 알려진 클래스의 객체를 식별합니다. 이 과정에서 계산된 고-gradient optical-flow 정보를 활용하여 정적과 동적 세그멘테이션을 식별합니다. 또한, 일관성 검사 모듈을 도입하여 동적 특징의 최종 분류를 개선하였습니다.

- **Performance Highlights**: 우리의 방법은 공용 데이터셋을 통해 검증되었으며, 알려지지 않은 객체가 이미지에 포함된 경우 전통적인 VSLAM보다 우수한 성능을 보였습니다. 또한, 알려진 객체만 있는 경우에는 기존의 선도적인 semantic VSLAM 기술과 유사한 성능을 유지하였습니다. 이로 인해 복잡한 동적 환경에서도 안정적인 성능을 발휘하는 VSLAM 기술의 새로운 가능성을 제시하고 있습니다.



### Joint Co-Speech Gesture and Expressive Talking Face Generation using Diffusion with Adapters (https://arxiv.org/abs/2412.14333)
- **What's New**: 본 논문에서는 얼굴( face)과 신체( body) 움직임을 동시에 생성할 수 있는 새로운 모델 아키텍처를 제안합니다. 기존 방법론이 각 작업에 대해 별도의 모델에 의존했던 것과는 달리, 단일 네트워크 내에서 두 작업을 공동으로 처리하여 훈련 복잡성을 줄입니다. 이 접근법은 어댑터 모듈(adapter modules)을 활용하여 서로 약한 상관관계를 갖는 두 작업을 효과적으로 모델링할 수 있게 해줍니다.

- **Technical Details**: 우리의 방법은 공동 훈련된 어댑터 모듈이 포함된 단일 트랜스포머 기반의 디노이징(denoising) 네트워크를 사용하여 동작합니다. 이렇게 구성된 네트워크는 코-스피치 제스처(co-speech gestures)와 표현적인 talking face를 모두 모델링할 수 있으며, 이로 인해 각 모달리티 간의 정보가 상호 작용할 수 있습니다. 데이터는 오디오와 움직임 시퀀스에 기반하여 정렬되어 준비되며, 이로 인해 다양한 제스처와 표현을 생성할 수 있습니다.

- **Performance Highlights**: 우리의 확산 기반(diffusion-based) 방법은 코-스피치 제스처와 표현적인 talking face 모두에서 최첨단 결과를 달성하였으며, 정량적 및 정성적 평가에서 기존 방법들보다 우수한 성능을 입증했습니다. 사용자 연구에서도 우리의 방법이 현실감(realism)과 신뢰성(believability) 측면에서 더 높은 점수를 받았습니다. 코드 또한 제공되어 있어 연구자들이 쉽게 접근하고 구현할 수 있도록 배포되었습니다.



### Personalized Generative Low-light Image Denoising and Enhancemen (https://arxiv.org/abs/2412.14327)
- **What's New**: 본 논문에서는 저조도 촬영에서의 이미지 복원을 위한 Personalized Generative Denoising (PGD) 접근법을 제안합니다. 이는 사용자의 개인 사진첩을 활용하여 개인화된 확산 모델을 구축하여, 기존의 방법들이 겪는 환각 현상을 줄입니다. 핵심 혁신은 사진첩에서 개인의 물리적 속성을 추출하는 ID 일관성 물리적 버퍼(identity-consistent physical buffer)를 도입하는 것입니다.

- **Technical Details**: PGD는 저조도 환경에서의 이미지 노이즈 제거와 향상 성능을 향상시키기 위해 사용자 특화 priors 를 활용합니다. 이미지 구조를 이용한 전통적인 기법들과 차별화되며, 기존의 사용된 방법들이 일반적 priors를 학습하는 반면, 본 연구는 사용자의 사진첩을 기반으로 보다 구체적인 priors를 제공합니다. 이를 통해 이미지 복원 과정에서 신원(identity)을 유지할 수 있도록 합니다.

- **Performance Highlights**: 광범위한 저조도 테스트 시나리오에서 PGD 방법은 기존 확산 기반 노이즈 제거 기법보다 뛰어난 성능을 보여줍니다. 특히, 얼굴 이미지 복원 과정에서 실질적인 신원 보존과 더불어 비주얼 품질 향상을 이끌어내며, 기존 기술들의 한계를 극복하는 중요한 기여를 합니다.



### What Has Been Overlooked in Contrastive Source-Free Domain Adaptation: Leveraging Source-Informed Latent Augmentation within Neighborhood Contex (https://arxiv.org/abs/2412.14301)
Comments:
          ICLR 2025

- **What's New**: 본 논문에서는 source-free domain adaptation (SFDA)에서 발생하는 도메인 갭(domain gap) 문제를 해결하기 위한 새로운 방법인 Source-informed Latent Augmented Neighborhood (SiLAN)을 제안합니다. 기존의 방법들은 데이터 라벨이나 소스 데이터를 활용하지 않고도 도메인 불변 표현을 개발하는 데 중점을 두었습니다. SFDA의 성능을 높이기 위해, 우리는 대조 학습(contrastive learning)의 원리를 기반으로 하여 긍정 키(positive keys)의 설계가 목표 도메인 분류 성능에 미치는 영향을 철저히 분석했습니다.

- **Technical Details**: 우리는 대조 학습 기반의 SFDA 분석을 통해 세 가지 중요한 요소를 식별했습니다: 1) 표준 데이터 증강 기법이 긍정 변환을 잘못 분류할 가능성을 줄이지 못하는 점, 2) 이웃의 수를 증가시키면 부드러운 예측 결과가 나오는 대신 로짓 클러스터(logit clusters)가 겹치는 문제가 발생하는 점, 3) 소스 전이 모델(source pre-trained model)을 통해 이웃 레이블 일관성을 활용하는 효과적인 방법이 아직 탐구되지 않은 점입니다. SiLAN 방법은 이론적 통찰을 바탕으로, 타겟 쿼리 샘플의 이웃 중심(latent features)의 잔여 특성에 가우시안 노이즈를 적용하여 긍정 키 생성을 향상시킵니다.

- **Performance Highlights**: 실험 결과, SiLAN과 InfoNCE 기반 대조 손실을 결합하여 선정된 벤치마크 SFDA 데이터셋에서 최신 기술(state-of-the-art) 성능을 기록했습니다. 기존의 SFDA 방법들과 비교하여, 제안된 접근 방식은 도메인 간 정렬(domain alignment)을 효과적으로 통합하여 분류 성능을 높였습니다. 이러한 성과는 SFDA 연구 분야에 있어 긍정적인 기여를 할 것으로 기대됩니다.



### Temporally Consistent Object-Centric Learning by Contrasting Slots (https://arxiv.org/abs/2412.14295)
- **What's New**: 최근 동영상에서의 비지도 객체 중심 학습(object-centric learning)은 구조적 표현을 추출하는 유망한 접근 방식이다. 이를 통해 자율 제어(autonomous control)와 같은 하위 작업을 지원하기 위해선 표현이 조합 가능(compositional)하고 시간적으로 일관성(temporally consistent)이 있어야 한다. 기존 방법들은 종종 장시간의 안정성을 유지하지 못하며, 이는 훈련 목표가 시간적 일관성을 강제하지 않기 때문이다. 본 연구에서는 비디오 객체 중심 모델을 위한 새로운 객체 수준의 시간 대비 손실(temporal contrastive loss)을 도입하였다.

- **Technical Details**: 우리는 Slot Contrast라는 새로운 방법론을 제안하여 시간적 일관성을 유지하는 도전 과제를 해결한다. 이 방법은 사람의 주석이 필요 없이 실제 세계 비디오 데이터에 적용 가능하며, 배치 전체에서 슬롯 표현(slot representations)을 대조하면서 연속 프레임 사이의 시간적 일관성을 보장하는 자기 지도 대비 학습(self-supervised contrastive learning) 목표를 포함한다. 우리의 접근 방식은 기존 입력 재구성 기반 영상 객체 중심 프레임워크를 확장하여 슬롯 간의 대조를 통해 일관된 표현을 발견하도록 모델을 조정한다.

- **Performance Highlights**: 우리는 Slot Contrast가 학습된 표현의 시간적 일관성을 향상시킬 뿐만 아니라, 객체 발견(object discovery) 작업에서도 최첨단 성과를 달성한다고 보여준다. 이 방법은 운동 단서(motion cues)를 사용하는 약한 지도 학습(weakly-supervised) 모델들을 초월하며, 탈출장면(on-the-fly)에서의 비지도 객체 추적(object tracking)과 잠재적 객체 동역학 모델링(latent object dynamics modeling) 같이 복잡한 하위 작업에서 더욱 효과적임을 입증한다.



### TRecViT: A Recurrent Video Transformer (https://arxiv.org/abs/2412.14294)
- **What's New**: 본 논문에서는 비디오 모델링을 위한 새로운 블록을 제안합니다. 이 모델은 시간-공간-채널 시간 세분화(time-space-channel factorisation) 방식에 기반하며, 각각의 차원에 맞는 전용 블록을 사용합니다. TRecViT는 뚜렷한 이점을 가지고 있으며 대규모 비디오 데이터셋에서 기존 모델보다 훨씬 효율적입니다.

- **Technical Details**: 제안된 TRecViT 아키텍처는 차원 간의 혼합을 위해 게이팅된 선형 재귀 유닛(gated linear recurrent units, LRUs), 자기 주의(self-attention) 레이어 및 다층 퍼셉트론(MLP)을 사용합니다. 특히, LRUs는 시간적으로 정보를 혼합하고, 자기 주의 레이어는 공간적으로, MLP는 채널 간의 혼합을 수행합니다. 이 혼합 방식 덕분에 TRecViT는 메모리 사용량이 적고 계산 복잡도도 낮습니다.

- **Performance Highlights**: 실험 결과, TRecViT는 SSv2와 Kinetics400과 같은 대규모 비디오 데이터셋에서 기존 비디오 모델인 ViViT-L보다 더 뛰어난 성능을 발휘하면서도 파라미터 수가 3배 적고 메모리 사용이 12배 더 적으며 FLOPs 수치도 5배 낮았습니다. 이 아키텍처는 다양한 비디오 이해(Task) 작업에 적합하고, 감독학습 또는 자기 감독학습 방식으로 쉽게 훈련할 수 있습니다.



### PixelMan: Consistent Object Editing with Diffusion Models via Pixel Manipulation and Generation (https://arxiv.org/abs/2412.14283)
Comments:
          AAAI 2025; version includes supplementary material; 27 Pages, 15 Figures, 6 Tables

- **What's New**: 본 논문에서 제안하는 PixelMan은 기존의 Diffusion Models를 활용한 일관된 객체 편집을 위한 새로운 방법으로, 편집 과정에서 불필요한 전환 과정을 생략하여 효율성과 일관성을 크게 개선합니다. 이 방법은 훈련 없이 픽셀 조작(Pixel Manipulation)을 통해 객체를 직접 복제한 후, 이를 효율적으로 조화시킴으로써 편집된 이미지의 질을 높입니다. PixelMan은 경쟁하는 모든 훈련 기반 및 훈련 없는 방법을 초월하여, 단지 16단계의 추론으로 다양한 편집 작업을 수행할 수 있다는 점에서 혁신적입니다.

- **Technical Details**: PixelMan은 피사를 픽셀 공간에서 직접 수정하여 표적 위치에 객체의 복사본을 생성하며, 이를 통해 보다 높은 이미지 일관성을 달성합니다. 이 과정에는 세 단계 브랜칭의 비전환 샘플링 접근법이 포함되어, 명확한 "델타" 편집 방향을 사용자 요구에 맞게 계산합니다. 또한, 자동 주의(Self-Attention) 메커니즘에 의한 정보 누수를 방지하기 위해 누출 방지 기법을 도입하여 조화로운 배경 복원을 이룹니다.

- **Performance Highlights**: PixelMan은 COCOEE와 ReS 데이터 세트를 기반으로 한 실험을 통해 객체, 배경, 의미적 일관성 지표에서 우수한 성능을 나타냈습니다. 이 방법은 기존 인기 있는 방법보다 적은 평균 지연시간과 낮은 수의 Network Function Evaluations(NFEs)로 주목받으며, 훈련이 필요 없는 방법으로 높은 편집 품질을 달성했습니다.



### Split Learning in Computer Vision for Semantic Segmentation Delay Minimization (https://arxiv.org/abs/2412.14272)
- **What's New**: 이번 논문에서는 자원 제약이 있는 장치를 위한 실시간 컴퓨터 비전(CV) 애플리케이션의 요구에 맞춘 semantic segmentation의 추론 지연을 최소화하기 위한 새로운 접근법인 split learning (SL)을 제안합니다. SL은 딥 뉴럴 네트워크(DNN)를 에지 디바이스와 중앙 서버 간에 분할하여 데이터 처리를 로컬화하고 전송해야 하는 데이터를 줄임으로써 지연 문제를 해결합니다.

- **Technical Details**: 논문의 주요 기여는 대역폭 할당(bandwidth allocation), 에지 디바이스의 DNN에서 컷 레이어(cut layer) 선택, 그리고 중앙 서버의 처리 자원(procesing resource) 할당의 공동 최적화(joint optimization)입니다. 또한, 병렬(parallel) 및 직렬(serial) 데이터 처리 시나리오를 조사하고, 연산 요구사항을 줄이면서 근사적 최적 성능을 유지하는 저복잡도( low-complexity) 휴리스틱 솔루션을 제안합니다.

- **Performance Highlights**: 수치 결과에 따르면, 제안한 방법은 추론 지연을 효과적으로 감소시키며, 자원 제약이 있는 동적인 환경에서 실시간 CV 애플리케이션 향상을 위한 SL의 잠재력을 보여줍니다. 이 접근법은 특히 자율주행 차량 및 스마트 시티 인프라와 같은 응용 프로그램에 필수적인 성능을 제공합니다.



### Descriptive Caption Enhancement with Visual Specialists for Multimodal Perception (https://arxiv.org/abs/2412.14233)
Comments:
          An open-source data engine for generating detailed image captions

- **What's New**: 본 논문에서는 이미지와 언어 간의 연결을 위해 기존의 방법을 개선하는 새로운 접근법인 Descriptive Caption Enhancement Engine (DCE)를 제안합니다. DCE는 사전 학습된 비주얼 스페셜리스트(visual specialists)를 활용하여 이미지 캡션을 개선하고, 이는 이전의 LMM(largest multimodal model) 기반 캡션 생성 방법들보다 더 정확하고 상세한 캡션을 생성합니다. 이를 통해 시각적 이해와 추론 능력을 향상시킬 수 있는 많은 가능성을 보여줍니다.

- **Technical Details**: DCE는 객체의 낮은 수준(low-level) 속성과 세부 사항(fine-grained attributes) 및 객체 간의 관계(object relations)를 조사하여 캡션을 생성합니다. 인스턴스 수준(attributes)에서 심리적인 요소(예: depth, emotion)와 물체의 관계(예: 상대 위치, 인간-객체 상호작용)를 포함한 여러 속성을 명확하게 나누어 캡션을 생성합니다. 최종적으로, Generative Large Language Models (LLMs)를 활용하여 이 정보를 토대로 상세한 캡션을 생성하게 됩니다.

- **Performance Highlights**: 실험 결과, DCE는 1.1백만 장의 이미지로 구성된 대규모 데이터셋에서 성공적으로 활용되었으며, LLaVA-v1.5와 LLaVA-NeXT 모델 모두에서 향상된 성능을 보여주었습니다. DCE로 생성된 자세한 캡션은 LMM의 시각-언어 정렬(visual-language alignment)을 크게 개선하여 다양한 평가 기준에서 뛰어난 결과를 기록했습니다. 이러한 성과는 DCE의 방법론이 고급 이미지 인식 작업 수행에 도움을 줄 수 있음을 입증합니다.



### ViTmiX: Vision Transformer Explainability Augmented by Mixed Visualization Methods (https://arxiv.org/abs/2412.14231)
- **What's New**: 이 연구는 다양한 설명 가능성 방법을 혼합하여 비전 트랜스포머(Visual Transformer, ViT) 모델의 해석 가능성을 향상시키는 하이브리드 접근 방식을 제안합니다. 기존의 기법들이 가진 제한 사항을 극복하고, 서로 다른 접근 방식의 강점을 모아 보다 의미 있는 시각적 설명을 제공하고자 합니다. 특히, 기하 평균을 활용한 혼합 방법을 소개하여 객체 분할 작업에서 뚜렷한 성과를 나타냈습니다.

- **Technical Details**: 비전 트랜스포머는 자가 주의(Self-Attention) 메커니즘을 중앙 구성 요소로 가지며, 이는 입력 이미지의 다양한 부분의 중요성을 평가합니다. 기존의 시각화 기법에는 LRP(Layer-wise Relevance Propagation)와 같은 그래디언트 기반 방법이 포함됩니다. 이 연구는 이러한 기존 기법을 조합하여 다중 인사이트를 제공하는 하이브리드 모델을 개발하였으며, 자가 주의 메커니즘과의 상관 관계를 통해 진화된 설명 가능성을 보여줍니다.

- **Performance Highlights**: 실험 결과, 제안된 하이브리드 방법이 개별 방법들보다 비전 트랜스포머의 해석 가능성을 크게 향상시켰음을 보여줍니다. 다양한 시각화 기법을 혼합함으로써, 모델의 투명성을 증진하고 중요한 이미지 기능을 더 잘 이해할 수 있게 되었습니다. 이러한 연구 결과는 설명 가능한 인공지능(XAI) 분야의 진전을 보여주며, 의료 진단과 같은 분야에서 신뢰할 수 있는 결정을 내리는 데 기여할 수 있습니다.



### Distilled Pooling Transformer Encoder for Efficient Realistic Image Dehazing (https://arxiv.org/abs/2412.14220)
Comments:
          18 pages, 17 figures

- **What's New**: 이 논문에서는 현실적인 이미지 디헤이징을 위해 경량화된 신경망 DPTE-Net(Distilled Pooling Transformer Encoder)을 제안합니다. 현재 비전 트랜스포머(ViTs)의 자기 주의(Self-Attention, SA) 모듈의 복잡성이 이미지 해상도에 따라 제곱으로 증가하여 자원 제약이 있는 장치에서의 적용이 어려운 상황입니다. 제안된 DPTE-Net은 전통적인 SA 모듈을 효율적인 풀링 메커니즘으로 대체하여 계산 요구량을 크게 줄이면서 ViTs의 학습 능력을 유지합니다.

- **Technical Details**: DPTE-Net은 다단계 SA 자유 ViT 유사 블록으로 구성되어 있으며, 지식 증류(Knowledge Distillation, KD) 훈련 기법을 기반으로 훈련됩니다. 이 과정에서 디헤이징 교사 네트워크로부터 지식을 활용하여 학생 네트워크를 훈련시키며, 이는 이전 KD 기반 디헤이징 연구에서는 철저히 검토되지 않은 중요한 요소입니다. 또한, 전송 인식 손실(transmission-aware loss)을 활용하여 다양한 헤이즈 밀도에 적응하는 학습 과정을 최적화합니다.

- **Performance Highlights**: DPTE-Net은 여러 벤치마크 데이터세트에서 테스트된 결과, 기존의 최첨단 방법들과 비교하여 경쟁력 있는 디헤이징 성능을 달성하면서도 낮은 계산 복잡성을 유지하는 가능성을 보여줍니다. 논문에서는 DPTE-Net의 성능이 다양한 환경 조건에서도 일관된 성능을 제공하며, 효율성와 성능 간의 우망의 trade-off를 잘 조절함을 강조합니다. 제안된 네트워크는 다수의 벤치마크 데이터 세트에서 평가되었으며, 우수한 성능과 계산 복잡성의 적절한 균형을 입증하였습니다.



### Improving Generalization Performance of YOLOv8 for Camera Trap Object Detection (https://arxiv.org/abs/2412.14211)
Comments:
          Master's thesis

- **What's New**: 이 논문은 카메라 트랩에 활용되는 YOLOv8 객체 탐지 알고리즘의 일반화 문제를 해결하기 위한 개선 사항을 제안합니다. 연구에서는 Global Attention Mechanism (GAM) 모듈의 통합, 수정된 다중 스케일 특징 융합 및 Wise Intersection over Union (WIoUv3) 손실 함수를 도입하여 모델의 성능을 향상시키고자 합니다. 이 개선책을 통해 야생 동식물 관찰과 보존을 위한 데이터 처리의 정확성을 높이고자 하였습니다.

- **Technical Details**: YOLOv8 모델의 한계를 분석하면서 객체 탐지에서 관찰되는 일반화 문제를 다루었습니다. 일반화 문제란 훈련된 모델이 새로운 데이터셋에 대한 정확한 예측을 할 수 없는 현상을 의미합니다. 본 연구는 GAM 모듈을 포함하여, 배경 잡음을 억제하고 객체 속성에 집중하는 강력한 모델을 개발하기 위한 다양한 기술적 접근을 포함하고 있습니다.

- **Performance Highlights**: 개선된 YOLOv8 모델은 새로운 환경에서 강한 일반화 성능을 보이며, 효율적인 객체 인식을 수행합니다. 평가 실험을 통해 배경 소음을 억제하고 객체의 특성을 잘 포착하여, 카메라 트랩 데이터셋에서 발생하는 도전 과제를 해결할 수 있는 역량을 입증했습니다. 이로 인해 야생 동물의 보존 작업에 더 넓은 적용 가능성을 열며, 동물 개체 수와 서식지 관리를 효과적으로 지원할 수 있는 기반을 마련합니다.



### Advancing Vehicle Plate Recognition: Multitasking Visual Language Models with VehiclePaliGemma (https://arxiv.org/abs/2412.14197)
Comments:
          33 pages, 9 figures

- **What's New**: 이 논문에서는 시각적 언어 모델(Visual Language Models, VLMs)인 VehiclePaliGemma를 활용하여 복잡한 조건에서 차량 번호판을 인식하는 새로운 방법을 제안합니다. 기존의 번호판 인식 기술은 Optical Character Recognition (OCR)에 의존했으나, 이 연구에서는 최신 VLM을 통해 성능 향상을 목표로 합니다. VehiclePaliGemma는 파라미터 조정된 PaliGemma VLM으로, 특히 명확하지 않은 번호판을 인식하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 제안된 시스템은 OpenAI GPT4o, Google Gemini 1.5, Google PaliGemma 등 다양한 최신 VLM을 이용하여 서로 가까이 있는 문자들을 효과적으로 인식합니다. 이 시스템은 저조도, 저해상도 카메라의 이미지, 불분명한 번호판 등 다양한 실제 조건에서 수집된 이미지 데이터셋을 사용합니다. VehiclePaliGemma는 다중 차량의 번호판을 로컬라이제이션하고 인식하는 멀티태스킹 VLM 모델로, 특정 색상 및 모델을 기준으로 설계된 프롬프트를 활용합니다.

- **Performance Highlights**: 논문에 따르면, VehiclePaliGemma는 87.6%의 정확도로 성능을 평가받았으며, A100-80GB GPU를 사용하여 1초당 7프레임의 속도로 번호판을 예측할 수 있습니다. 이 연구는 기존의 최신 방법들과 비교하여 뛰어난 성능을 보여주었으며, 다양한 방향으로 배치된 번호판을 가진 차량을 식별하는 멀티태스킹 능력도 탐구합니다.



### LlamaFusion: Adapting Pretrained Language Models for Multimodal Generation (https://arxiv.org/abs/2412.15188)
- **What's New**: LlamaFusion introduces a novel framework that enhances pretrained text-only large language models (LLMs) like Llama-3 with multimodal generative capabilities, allowing them to process and generate both text and images. 이 프레임워크는 Llama-3의 가중치를 활용하고, 이미지 처리를 위한 추가적인 transformer 모듈을 도입하여 텍스트와 이미지를 각각 처리합니다. 이를 통해 LlamaFusion은 텍스트 전용 모델의 언어 능력을 유지하면서도 강력한 시각 이해 및 생성 기능을 개발할 수 있게끔 합니다.

- **Technical Details**: LlamaFusion은 별도의 모듈에서 텍스트와 이미지를 각각 처리하도록 구성되어 있으며, 공통의 self-attention 층을 통해 두 개의 모달리티 간 상호작용을 가능하게 합니다. 훈련 과정에서 텍스트 관련 모듈은 고정되고 이미지 관련 모듈만 훈련하여, 이전의 언어 능력을 손상시키지 않으면서도 시각적 이해를 키웁니다. 또한, LlamaFusion은 기존의 text-only LLM에서 이미지를 이해하고 생성하는 능력을 효과적으로 훈련할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과 LlamaFusion은 이미지 이해에서 20%, 이미지 생성에서 3.6% 개선된 성능을 보이며, 전량 50%의 FLOPs만으로도 Llama-3의 언어 성능을 유지합니다. LlamaFusion의 성능은 Transfusion 모델에 비해 11.6% 더 우수하여 기존 시각-언어 모델을 적응시킬 수 있는 가능성을 보여줍니다. 이를 통해 LlamaFusion은 멀티모달 모델 개발의 효율적인 방향을 제시합니다.



### Till the Layers Collapse: Compressing a Deep Neural Network through the Lenses of Batch Normalization Layers (https://arxiv.org/abs/2412.15077)
Comments:
          Accepted at AAAI 2025

- **What's New**: 이번 논문에서는 딥 뉴럴 네트워크(DNN)에서 배치 정규화(batch normalization) 층을 활용하여 네트워크의 깊이를 줄이고 성능 저하 없이 계산 요구량과 지연 시간을 감소시키는 방법인 TLC(Till the Layers Collapse)를 제안합니다. 이 방법은 DNN의 과도한 매개변수를 관리하고 에너지 소비를 줄이는 데 중점을 두고 있습니다. 대규모 모델에서 특정 층을 제거함으로써 전체적인 계산 효율성을 개선하는 데 기여합니다.

- **Technical Details**: TLC는 배치 정규화 층의 매개변수를 이용하여 중요하지 않은 층을 식별하고 제거하는 방법입니다. 이를 통해 네트워크의 깊이를 줄이고 계산 요구량을 감소시킬 수 있습니다. 이 과정에서 표준화된 신호가 주로 긍정적일 때 선형 활성화가 최소한의 오류를 초래할 수 있음을 활용하여, 모델의 출력을 크게 변경하지 않으면서도 적절한 층 삭제가 가능합니다.

- **Performance Highlights**: 제안된 TLC 방법은 Swin-T, MobileNet-V2, RoBERTa와 같은 인기 있는 모델을 대상으로 이미지 분류 및 자연어 처리(NLP) 태스크에서 검증되었습니다. 이 방법은 층을 줄이는 동시에 정확도를 유지하며, 효율성을 향상시키는 성과를 보였습니다. 결과적으로, TLC는 성능 저하를 우려하지 않고도 DNN의 깊이를 효과적으로 감소시키는 가능성을 보여줍니다.



### Stable-V2A: Synthesis of Synchronized Sound Effects with Temporal and Semantic Controls (https://arxiv.org/abs/2412.15023)
- **What's New**: 이 논문에서는 비디오에 따라 시간적으로나 의미적으로 정렬된 오디오를 생성할 수 있는 새롭고 효율적인 시스템, Stable-V2A를 소개합니다. 이 시스템은 RMS-Mapper와 Stable-Foley의 두 단계 모델로 구성되어 있으며, 사운드 디자이너가 창의적인 작업에 집중할 수 있도록 반복적인 작업을 우회할 수 있는 도구입니다. 이를 통해 사운드 디자이너는 원하는 사운드를 선택하고, 그에 따라 오디오 생성을 조절할 수 있는 기능을 가지게 됩니다.

- **Technical Details**: Stable-V2A 모델은 두 가지 주요 구성 요소인 RMS-Mapper와 Stable-Foley로 구성되어 있습니다. RMS-Mapper는 비디오의 프레임과 광학 흐름을 입력으로 받아 RMS(envelope) 곡선을 추정하는 간단한 네트워크입니다. Stable-Foley는 입력으로 제공된 RMS envelope를 사용하여 최종 오디오를 생성하는 상태-of-the-art 잠재적 확산 모델입니다. 이를 통해 사운드 디자이너는 최종 오디오 트랙을 생성하기 전에 RMS envelope를 직접 수정할 수 있습니다.

- **Performance Highlights**: 모델의 성능은 Greatest Hits 데이터셋을 사용하여 평가되었으며, 오디오의 시간적 및 의미적 정렬 능력을 정량화하기 위한 여러 객관적 메트릭을 활용했습니다. 또한, 발 걸음 소리 생성에 대한 특정 사례 연구를 위해 Walking The Maps라는 새로운 비디오 게임에서 추출된 데이터셋도 소개되었습니다. 본 연구는 오디오 생성의 새로운 가능성을 열어주는 중요한 발전을 이뤘습니다.



### Robust Federated Learning in the Face of Covariate Shift: A Magnitude Pruning with Hybrid Regularization Framework for Enhanced Model Aggregation (https://arxiv.org/abs/2412.15010)
- **What's New**: 이 논문은 Federated Learning (FL) 맥락에서 데이터의 비동질성을 해결하는 혁신적인 방법인 FedMPR(정확한 매개변수 제거 및 정규화 기법을 결합한 연합 학습)를 제안합니다. 기존 FL 접근 방식에서 발생하는 모델 집합의 불안정성과 성능 하락 문제를 해결하기 위해, 클라이언트 간의 양극적인 데이터 분포를 다루는 강화된 다층 프레임워크를 포함합니다. 또한, 데이터 불균형이 갈수록 심각한 CelebA-gender 데이터셋을 도입하여 새로운 평가 벤치마크를 제공합니다.

- **Technical Details**: FL의 일반적인 운영 방식에서 클라이언트는 로컬 데이터셋을 기반으로 독립적으로 모델을 학습하고, 그 결과를 중앙 서버에서 집합하여 전역 모델을 업데이트합니다. 그러나 데이터의 비동질성으로 인해, 클라이언트 간의 매개변수 차이가 커지면 집합 과정이 불안정해지고 성능 저하를 초래할 수 있습니다. FedMPR 프레임워크는 매개변수 기반 제거(magnitude-based pruning), 드롭아웃(dropout), 노이즈 주입(noise injection) 기법을 활용하여 이러한 문제를 해결하고자 합니다.

- **Performance Highlights**: 실험 결과, FedMPR는 CIFAR10, MNIST, SVHN, Fashion MNIST와 같은 여러 기준 데이터셋에서 기존의 FL 접근 방식보다 우수한 성과를 보여주었습니다. 특히, 클라이언트 간의 큰 데이터 분포 차이를 가진 경우에도 강력한 성능을 보여주었습니다. 새로운 CelebA-gender 데이터셋에서는 FL 방법의 효율성을 더욱 강화하기 위한 다양한 조건을 실험하여 효과적인 결과를 입증했습니다.



### Dream to Manipulate: Compositional World Models Empowering Robot Imitation Learning with Imagination (https://arxiv.org/abs/2412.14957)
- **What's New**: 이 논문에서는 현실 세계와 그 역학을 명시적으로 표현하는 새로운 세계 모델 구성 패러다임을 소개합니다. 기존의 세계 모델은 로봇 앞의 실제 환경을 직접적으로 모사하지 못해 비현실적인 행동과 환각을 초래했습니다. 그러나 저자들은 DreMa라는 첫 번째 구성 조작 세계 모델을 제안하여 객체의 새로운 구성을 상상하고 로봇 행동의 미래 결과를 예측할 수 있는 능력을 부여합니다.

- **Technical Details**: DreMa는 객체 중심의 Gaussian Splatting과 물리 시뮬레이터를 통합하여 물리적 신뢰성이 높은 예측을 활용할 수 있는 상호작용 가능한 표현을 제공합니다. Gaussian Splatting은 고해상도 및 실시간 그래픽 렌더링을 가능하게 하여 3D 장면을 최적화 가능합니다. DreMa는 학습된 개념의 새로운 조합을 생성하기 위해 구성적 속성을 추론하고, 이를 통해 로봇은 현실적인 관찰을 렌더링하며 상상 속에서 바로 정책을 실행할 수 있습니다.

- **Performance Highlights**: DreMa의 기능을 통해 실제 Franka Emika Panda 로봇이 다양한 작업 변형마다 단 하나의 예제만으로도 성공적으로 새로운 물리적 과제를 학습할 수 있음을 보여줍니다. 시뮬레이션에서 +9.1%, 실제 작업에서 +33.3%의 일반화 향상이 관찰되었습니다. 이 연구 결과는 적은 수의 시연으로 복잡한 작업 수행 능력을 개선하는 데 중요한 기여를 하고 있습니다.



### Head and Neck Tumor Segmentation of MRI from Pre- and Mid-radiotherapy with Pre-training, Data Augmentation and Dual Flow UN (https://arxiv.org/abs/2412.14846)
- **What's New**: 이번 연구는 두 가지 방사선 치료 전과 중 이미지를 세분화하는 여러 전략의 효과를 조사했습니다. 특히 모델의 성능을 개선하기 위해 MixUp 데이터 증강 기법과 사전 학습된 가중치를 활용했습니다. 또한, 미드-RT 이미지를 위한 새로운 네트워크 아키텍처를 제안하였으며, 이 모델은 사전-RT 이미지와 레이블의 정보를 통합하여 세분화 성능을 높였습니다.

- **Technical Details**: 연구에서 사용한 두 가지 주요 네트워크 구조는 basic segmentation network와 Dual Flow UNet(DFUNet)입니다. Basic segmentation network는 인코더-디코더 아키텍처를 기반으로 한 첫 번째 네트워크로, 단일 채널의 pre-RT 이미지를 입력으로 받습니다. DFUNet은 두 개의 인코더를 포함하고 있으며, 미드-RT 이미지와 사전 등록된 pre-RT 이미지와 그 마스크를 동시에 처리하여 정보를 효율적으로 융합하는 데 중점을 둡니다.

- **Performance Highlights**: 최종 테스트에서 pre-RT 세분화는 82.38%, mid-RT 세분화는 72.53%의 성능을 기록했습니다. 이 연구 결과는 Dice 유사도 계수(DSC)를 기준으로 하며, 여러 전략을 통해 세분화 성능을 향상시켰음을 보여줍니다. 특히, 각 fold에서 최고 성능의 모델을 선택하여 앙상블 평균을 생성함으로써 더욱 신뢰성 있는 결과를 도출했습니다.



### Progressive Multimodal Reasoning via Active Retrieva (https://arxiv.org/abs/2412.14835)
Comments:
          Working in progress

- **What's New**: 이번 연구에서는 다중 단계를 고려한 다중 모드(multimodal) 추론 과제에서 MLLM의 성능을 향상시키기 위한 새로운 프레임워크인 AR-MCTS를 제안합니다. 이 프레임워크는 Active Retrieval(AR)과 Monte Carlo Tree Search(MCTS)를 결합하여 복잡한 추론 문제를 해결하는 데 필요한 핵심 인사이트를 동적으로 검색할 수 있도록 설계되었습니다. 특히, 이 연구는 기존의 빔 탐색(beam search) 샘플링 방법을 대체하는 혁신적인 접근 방식을 도입하여 각 추론 단계에서 다양한 문제 해결 인사이트를 제공함으로써 신뢰성을 높이고자 합니다.

- **Technical Details**: AR-MCTS 프레임워크는 통합된 검색 모듈을 개발하여 하이브리드 모드 검색 데이터베이스로부터 복잡한 추론을 지원하기 위한 핵심 인사이트를 검색합니다. MCTS 알고리즘을 활용하여 단계별로 주어진 문제의 적절한 해답을 유도하는 과정 보상을 정의하고, 각 단계에서 이전 단계를 바탕으로 샘플링을 최적화합니다. 이러한 접근 방식은 추론의 신뢰성과 다양성을 향상시키며, 자동화된 다중 모드 추론 검증 과정을 가능하게 합니다.

- **Performance Highlights**: 실험 결과에 따르면, AR-MCTS는 세 가지 복잡한 다중 모드 추론 기준을 통해 다양한 모델에서 효과성을 입증했습니다. AR-MCTS는 샘플링의 다양성과 검증의 정확성을 최적화하여 신뢰성 있는 다중 모드 추론을 위한 promising한 솔루션을 제공합니다. 이 연구 결과는 다중 모드 추론의 고도화를 위한 가능성을 열어 주며, MLLM의 성능 향상에 기여할 것으로 기대됩니다.



### Robust PCA Based on Adaptive Weighted Least Squares and Low-Rank Matrix Factorization (https://arxiv.org/abs/2412.14629)
- **What's New**: 본 논문에서는 Robust Principal Component Analysis (RPCA) 모델을 새롭게 제안하며, 적응형 가중 최소제곱법(AWLS)과 저계 행렬 분해(LRMF)를 통합하여 데이터를 처리합니다. 이 모델은 가중치 행렬을 동적으로 조정할 수 있는 self-attention-inspired 기법을 사용하여 각 반복에서 중요한 구성 요소를 강조합니다. 또한, 기존의 $	ext{l}_1$ 노름 기반 방법에 비해 편향을 줄이고 계산 과정을 단순화하는 가중치 F-norm을 적용합니다.

- **Technical Details**: 제안된 방법은 특정 매개변수의 가중치를 동적으로 조절함으로써 RPCA의 성능을 개선하며, 이 과정에서 교차 최소화 알고리즘을 사용합니다. 각 하위 문제는 명확한 해를 가지고 있어 계산 효율성을 크게 향상시킵니다. 이 논문에서 소개된 RPCA 모델은 안정을 높이면서도 노이즈와 아웃라이어 처리에서 성능을 더욱 향상시킵니다.

- **Performance Highlights**: 수치 실험 결과, 새롭게 제안된 RPCA 모델이 기존의 비볼록 정규화 방법보다 우수한 성능과 안정성을 보여줍니다. 특히, 실제 응용 프로그램에서 정확성이 향상되고 더 나은 강건성을 제공하여 이미지 처리 및 비정상 탐지와 같은 분야에서 뛰어난 결과를 도출합니다.



### HarmonicEval: Multi-modal, Multi-task, Multi-criteria Automatic Evaluation Using a Vision Language Mod (https://arxiv.org/abs/2412.14613)
- **What's New**: 이 논문은 Vision-Language Models (VLMs)에서 생성된 텍스트의 자동 평가를 위한 새로운 메트릭인 HarmonicEval을 제안합니다. 기존의 측정 지표들이 전반적인 품질에만 집중하여 평가의 측면에서 부족함을 드러내는 가운데, HarmonicEval은 기준 기반 점수를 집계하여 전반적인 점수를 생성하는 방식을 채택합니다.

- **Technical Details**: HarmonicEval의 평가는 세 단계로 구성됩니다: 1) 각 기준에 대한 점수를 생성하기 위해 VLM을 프로세스하는 단계, 2) VLM이 생성한 출력 토큰 확률을 기반으로 점수를 안정화하는 단계, 3) 하모닉 가중치를 사용해 최종 점수를 계산하는 단계입니다. 이를 통해 VLM의 텍스트 품질을 다양한 비전-언어 작업에서 평가할 수 있습니다.

- **Performance Highlights**: 실험 결과, HarmonicEval은 기존의 평가 지표들보다 인간의 판단과 더욱 높은 상관관계를 보여줍니다. 또한, 제안된 MMHE 데이터셋을 통해 HarmonicEval이 특정 평가 기준을 간과하는 기존 메트릭들의 단점을 보완하면서도 종합적인 평가를 수행할 수 있음을 입증합니다.



### Downscaling Precipitation with Bias-informed Conditional Diffusion Mod (https://arxiv.org/abs/2412.14539)
Comments:
          3 pages, 2 figures. Accepted by Proceedings of IEEE International Conference on Big Data, Dec 15-18, 2024

- **What's New**: 기후 변화로 인해 지역적 강우 예측의 중요성이 증가하고 있습니다. 그러나 현재의 Global Climate Models(GCMs)는 지역별 분석에 필요한 세밀함이 부족합니다. 이 연구에서는 통계적 다운스케일링을 위한 편향 정보 기반의 조건부 확산 모델을 도입하여 고해상도 강우 예측을 가능하게 합니다.

- **Technical Details**: 우리는 Gamma Correction과 Bias-aware Guided Sampling이라는 두 가지 혁신적인 접근 방식을 통해 강우의 장기적 분포 문제를 해결하고자 합니다. Gamma Correction은 비정상적인 분포를 정상화하여 모델의 학습 성능을 향상시킵니다. 또한, Bias-aware Guided Sampling 기법을 활용하여 다운스케일링 과정에서 편향을 체계적으로 감소시킵니다.

- **Performance Highlights**: 제안된 모델은 8배 다운스케일링 설정에서 높은 정확도를 달성하였으며, 이전의 결정론적 방법들을 능가하는 성능을 보였습니다. 이는 강우 데이터를 보다 신뢰성 있게 생성할 수 있는 가능성을 보여줍니다.



### GraphEQA: Using 3D Semantic Scene Graphs for Real-time Embodied Question Answering (https://arxiv.org/abs/2412.14480)
Comments:
          Project website: this https URL

- **What's New**: 이번 연구는 Embodied Question Answering (EQA) 분야에서의 새로운 접근법인 GraphEQA를 제안합니다. GraphEQA는 실시간 3D metric-semantic scene graphs (3DSGs)와 작업 관련 이미지를 활용하여 Vision-Language Models (VLMs)를 기반으로 EQA 작업을 수행하는 데 필요한 멀티모달 메모리를 제공합니다. 이 방법은 로봇이 이전의 환경 지식을 사용하여 보다 효율적으로 탐색하고 계획할 수 있도록 돕습니다.

- **Technical Details**: GraphEQA는 3DSGs의 계층적 구조를 활용하여 체계적인 계획과 의미 기반 탐색을 지원합니다. 이 구조는 가시적 환경의 의미 표현을 온라인으로 업데이트할 수 있는 강력한 도구로 작용합니다. 연구에서는 HM-EQA 데이터셋을 사용한 시뮬레이션 실험과 실제 가정 및 사무실 환경에서의 실험을 통해 그 효능을 입증하였습니다.

- **Performance Highlights**: 실험 결과, GraphEQA는 기존 기준선 모델들보다 EQA 작업을 수행하는 데 있어 더 높은 성공률과 적은 계획 단계를 기록하며 우수한 성과를 보였습니다. 따라서 이 방법은 로봇이 새로운 환경에서도 효과적으로 질문에 답할 수 있는 가능성을 높입니다.



### DriveGPT: Scaling Autoregressive Behavior Models for Driving (https://arxiv.org/abs/2412.14415)
Comments:
          14 pages, 16 figures, 9 tables, and 1 video link

- **What's New**: 이 논문에서는 자율주행을 위한 확장 가능한 행동 모델인 DriveGPT를 제안합니다. DriveGPT는 순차적 의사결정 과제로서 주행을 모델링하고, 미래의 에이전트 상태를 예측하는 트랜스포머 모델을 학습합니다. 기존의 오픈 소스 데이터셋에 비해 약 50배 많은 1억 개 이상의 고품질 주행 예제를 활용하여 모델 파라미터를 10억 개 이상으로 확장했으며, 이는 행동 모델의 성능을 크게 향상시킵니다.

- **Technical Details**: DriveGPT는 인코더-디코더 아키텍처를 사용하는 표준 행동 모델을 기반으로 합니다. 순차적 예측 과제로서, 드라이빙 맥락 정보와 역사적인 에이전트 위치에 따라 목표 에이전트의 미래 위치를 예측합니다. 이 모델은 트랜스포머 기반으로 설계되어 있어 순차적 모델링 작업에서 높은 확장성을 제공합니다.

- **Performance Highlights**: DriveGPT는 다양한 규모의 계획 작업에서 평가되었으며, 정량적 메트릭과 질적 예제를 통해 성능이 입증되었습니다. 대규모 데이터셋으로 사전 학습을 통해 상태-of-the-art 기준을 초월하였고, 복잡한 현실 세계의 시나리오에서 폐쇄 루프 주행을 포함한 다양한 상황에서도 향상된 성능을 보입니다.



### The One RING: a Robotic Indoor Navigation Generalis (https://arxiv.org/abs/2412.14401)
- **What's New**: 이번 논문에서는 다양한 로봇 형태와 크기에 관계없이 작동할 수 있는 새로운 내비게이션 정책인 RING(Robotic Indoor Navigation Generalist)을 소개합니다. RING은 특정 로봇의 설정을 사용하지 않고도 학습된 정책을 시뮬레이션을 통해 다양한 형태의 로봇에 적용할 수 있도록 설계되었습니다. 이 연구는 실세계 로봇 플랫폼에서 보았을 때도 RING이 높은 성공률을 보이는 것을 증명합니다.

- **Technical Details**: RING은 시뮬레이션에서만 학습되며, 약 1백만 개의 랜덤화된 에이전트 설정을 활용하여 훈련되었습니다. 이 시스템은 AI2-THOR 시뮬레이터를 이용해 다양한 카메라 매개변수와 충돌체 크기를 조정하여 대규모로 에이전트 설정을 생성합니다. 정책 훈련은 초기 전문가 경로를 수집한 후, 정책 개선을 위해 강화 학습(RL)을 적용하여 수행됩니다.

- **Performance Highlights**: RING은 Stretch RE-1, LoCoBot, Unitree의 A1 등 여러 실세계 로봇 플랫폼에서 평균 72.1%와 78.9%의 성공률을 달성하며, 이전의 기존 최상 baseline과 비교해도 유의미하게 뛰어난 성능을 보여줍니다. 또한 RING은 추가적인 조정 없이도 새로운 로봇 플랫폼에 제로샷 전이를 성공적으로 수행하며, 연구자들이 접근하기 쉬운 형태로 배포할 예정입니다.



### I0T: Embedding Standardization Method Towards Zero Modality Gap (https://arxiv.org/abs/2412.14384)
Comments:
          16 figures, 8 figures, 7 tables

- **What's New**: 본 연구에서 제안하는 I0T 프레임워크는 이미지-텍스트 embedding 간의 modality gap을 최소화하는 두 가지 방법을 소개합니다. 첫 번째는 post-hoc embedding standardization 방법인 I0T_{post}로, modality gap을 거의 0으로 줄이는 기능이 있습니다. 두 번째는 학습 가능한 I0T_{async} 방법으로, 두 개의 normalization layer를 각 encoder에 추가하여 문제를 해결합니다.

- **Technical Details**: I0T 프레임워크는 frozen encoders에서 평균 벡터를 빼고 Frobenius normalization으로 embedding 활성화를 표준화하여 modality gap을 줄입니다. 이 연구는 이미지와 텍스트 encoder 간의 모드 특성을 고려하여 embedding 간의 불일치를 해결하는 데 중점을 두고 있습니다. 특히, CLIPScore 대신 I0TScore이라는 새로운 자동 평가 메트릭을 제안하여 모델의 성능을 더 잘 평가할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 modality gap을 줄이며, text-to-image retrieval 점수를 각각 9.2%와 6.7% 향상시킵니다. 또한, I0TScore는 서로 다른 modality 간에도 적용할 수 있는 첫 번째 자동 평가 메트릭으로, CLIPScore의 한계를 극복하며 두 가지 방법 모두 원래 모델의 embedding 표현을 유지합니다. I0T 프레임워크는 downstream 성능에 부정적인 영향을 주지 않으면서 gap을 줄이는 데 성공합니다.



### A Unifying Information-theoretic Perspective on Evaluating Generative Models (https://arxiv.org/abs/2412.14340)
- **What's New**: 본 논문에서는 생성 모델의 출력 평가를 위해 유의미한 메트릭을 정의하는 데 중점을 두고 있다. 기존의 precision과 recall 개념을 적용하여 생성된 데이터의 사실성과 다양성을 측정하기 위한 새로운 tri-dimensional metric인 Precision Cross-Entropy (PCE), Recall Cross-Entropy (RCE), Recall Entropy (RE)를 제안한다. 특히, 이 메트릭은 정보 이론의 관점에서 kNN 기반 메트릭을 통합하여 다양한 실패 모드를 구분할 수 있도록 설계되었다.

- **Technical Details**: 제안된 메트릭은 생성 모델의 출력을 fidelity와 diversity의 두 가지 주요 측면으로 나누어 평가하며, intra-class와 inter-class 다양성의 차이를 명확하게 측정한다. 이는 기존의 1D 메트릭, 즉 Inception Score (IS)와 Fréchet Inception Distance (FID)와는 달리, 여러 개별 실패 모드를 식별할 수 있는 차별화된 기능을 제공한다. kNN 밀도 추정 기법을 응용하여 구현된 이 메트릭은 샘플 및 모드 수준 분석이 가능하다.

- **Performance Highlights**: 실험 결과, 새로운 메트릭의 각 구성 요소가 출력의 질에 대해 매우 민감한 반응을 보였으며, 기존 메트릭의 단점이나 바람직하지 않은 행동을 드러내었다. 제안된 방법은 특히 이미지 평가에 중점을 두고 있으나, 다양한 데이터 유형에 쉽게 일반화될 수 있는 장점이 있다. 또한, 실험을 통해 기존 메트릭보다 더 효과적으로 평가의 제도화된 기준(ideals)에 부합함을 입증하였다.



### Covariances for Free: Exploiting Mean Distributions for Federated Learning with Pre-Trained Models (https://arxiv.org/abs/2412.14326)
- **What's New**: 이 논문에서는 Federated Learning (FL)에서 데이터 이질성을 줄이고 학습 속도를 개선하는 사전 학습된 모델의 사용을 제안합니다. 특히, 이를 통해 기존의 높은 통신 비용을 절감하면서도 성능 향상을 도모합니다. 제안된 방법은 클래스 평균(class means)만을 서버에 전달하여 클래스 공분산(class covariance) 행렬을 추정하는 방식으로, 이를 기반으로 선형 분류기를 초기화하는 기술을 소개하고 있습니다.

- **Technical Details**: 제안된 방법, 즉 Federated learning with COvariances for Free (FedCOF)는 클라이언트에서 서버로 전송되는 클래스 평균만으로 국제 공분산을 추정합니다. 이 과정을 통해 클래스 관련 메트릭을 공유하면서도, 두 번째 통계치(second-order statistics)를 전송할 때의 프라이버시 위험과 통신 비용을 낮출 수 있습니다. FedCOF는 FedNCM 방식과 동일한 통신 비용으로 경쟁적인 성능을 달성하는 것을 목표로 하며, 여러 FL 벤치마크에서 검증되었습니다.

- **Performance Highlights**: FedCOF는 기존의 최첨단 방법들에 비해 4-26%의 성능 향상을 보이며, 통신 비용이 동일한 조건에서 이뤄집니다. 게다가 FedCOF는 두 번째 통계치를 공유하는 방법보다 훨씬 적은 통신 오버헤드를 요구하면서도 경쟁력 있는 성능을 보여줍니다. 본 연구는 또한 FedCOF를 초기화 방법으로 사용하여 연합 미세 조정(federated fine-tuning)을 진행함으로써 더 나은 수렴 성능을 달성할 수 있음을 입증하고 있습니다.



### Transversal PACS Browser API: Addressing Interoperability Challenges in Medical Imaging Systems (https://arxiv.org/abs/2412.14229)
Comments:
          16 pages with 3 figures

- **What's New**: 본 논문은 DICOM 이미지를 효과적으로 질의하고 검색할 수 있는 사용자 친화적인 Transversal PACS Browser API를 소개합니다. 이 API는 고급 필터링 기능과 사용자 맞춤 검색 필드를 제공하여 의료 이미지 저장소를 손쉽게 탐색할 수 있도록 설계되었습니다. 또한, 여러 PACS 스테이션에서 통합된 인터페이스를 통해 데이터 액세스의 복잡성과 단편화 문제를 해결합니다.

- **Technical Details**: 애플리케이션 개발은 Qt 6.6.0 기반의 Qt Creator 통합 개발 환경에서 이루어졌으며, C++로 다중 플랫폼 앱 개발을 지원합니다. 사용자는 PACS 서버에 연결하고, 다양한 기준에 따라 검색을 실행하며, 특정 이미지를 로컬로 저장하고 미리 볼 수 있는 기능을 제공합니다. Qt의 GUI 도구는 사용자 입력과 백엔드 작업 사이의 원활한 상호작용을 통해 직관적인 인터페이스를 만드는데 기여했습니다.

- **Performance Highlights**: 포괄적인 테스트 결과 API의 디자인이 깔끔하고 사용이 용이하며 강력한 검색 기능이 있다는 것을 demonstrated 합니다. 이미지 미리보기 기능은 사용자가 의료 이미지를 보다 효율적으로 조회하고 활용할 수 있도록 지원합니다. 이 API는 의료 제공자에게 효율적인 자원의 접근 방식을 제공하며 전반적인 의료 서비스의 질을 향상시키는데 기여합니다.



### GraphicsDreamer: Image to 3D Generation with Physical Consistency (https://arxiv.org/abs/2412.14214)
- **What's New**: 이 논문은 인공지능 생성 3D 모델링 분야에서 혁신적인 GraphicsDreamer 방법을 소개합니다. 이 방법은 단일 이미지에서 고급 사용이 가능한 3D 메시(mesh)를 생성하는 기법으로, PBR(물리 기반 렌더링) 조명 방정식을 통합한 교차 영역 확산 모델을 사용합니다. GraphicsDreamer는 3D 객체가 신뢰할 수 있는 질감 세부 정보를 갖고 현실적인 재조명을 지원할 수 있도록 PBR 제약 조건을 강화합니다.

- **Technical Details**: GraphicsDreamer는 다중 보기 이미지와 PBR 재료를 예측하는 두 단계 방식으로 설계되었으며, 선명한 기하학적 구조와 디테일을 아우르는 심층 학습 기반의 역 렌더링 접근 방식을 채택하고 있습니다. 이 모델은 색상, 노멀, 깊이 및 PBR 구성요소를 포함하는 6개 도메인의 결합 분포를 예측하여 3D 객체의 기하학적 구조를 이해합니다. 또한, 3D 객체의 토폴로지 최적화 및 UV 언랩핑 자동화 기능을 통해 Blender, Unreal Engine 및 Unity와 같은 렌더링 엔진으로 직접 가져오기가 가능합니다.

- **Performance Highlights**: 광범위한 실험 결과, GraphicsDreamer는 이전의 방법들과 비교하여 합리적인 시간 내에 고품질 3D 자산을 생성할 수 있음을 보여주었습니다. 이 방법은 PBR 질감 세부 사항과 함께 매끄러운 기하학을 제공하여 그래픽 엔진에서 즉시 사용할 수 있도록 지원합니다. GraphicsDreamer는 기하학적 및 질감 세부 사항의 측면에서 선도적인 수준에 있으며 전체 PBR 재료 맵과 정돈된 토폴로지를 통해 생성된 3D 모델의 실용성을 크게 향상시킵니다.



### IMPROVE: Impact of Mobile Phones on Remote Online Virtual Education (https://arxiv.org/abs/2412.14195)
Comments:
          Article under review in the journal Scientific Data. GitHub repository of the dataset at: this https URL

- **What's New**: IMPROVE 데이터셋은 온라인 교육 중 모바일 전화 사용이 학습자에게 미치는 영향을 평가하기 위해 개발되었습니다. 이 데이터셋은 학업 성과, 주관적 학습자 피드백과 함께 생체학적(biometric), 행동적(behavioral), 생리학적(physiological) 신호를 포착하여 모바일 전화 사용의 학습에 미치는 영향을 종합적으로 분석합니다.

- **Technical Details**: IMPROVE 데이터셋은 120명의 학습자에게서 수집된 다중 모달(multimodal) 데이터로 구성되어 있으며, 이들은 서로 다른 전화 상호작용 수준에 따라 세 그룹으로 나뉘었습니다. 데이터 수집을 위해 16개의 센서가 사용되었으며, 이는 전기뇌파(electroencephalography), 비디오, 눈 추적기(eye tracker) 등 학습자 행동 및 인지 이해에 효과적인 지표로 알려져 있습니다.

- **Performance Highlights**: 데이터셋에는 2.83 테라바이트의 데이터가 포함되어 있으며, 145개의 모바일 전화 방해 사건이 라벨링됩니다. 기술 유효성 검증을 통해 신호 품질이 확인되었으며, 통계 분석 결과 모바일 전화 사용 중 생체학적 변화가 드러났습니다. 이는 모바일 전화 사용이 학습자의 주의력 및 학업 성과에 미치는 영향을 이해하는 데 기여할 수 있는 소중한 정보를 제공합니다.



### Optimize the Unseen -- Fast NeRF Cleanup with Free Space Prior (https://arxiv.org/abs/2412.12772)
- **What's New**: Neural Radiance Fields (NeRF)은 포토리얼리스틱(scene reconstruction) 장면 재구성에서 현저한 진전을 이루었지만, 포토메트릭 최적화에 의존할 경우 잘 알려진 'floaters'와 같은 시각적 아티팩트가 발생합니다. 본 연구에서는 Free Space Prior를 활용한 빠른 후처리(cleanup) 방법을 제안하여, 훈련 카메라가 보지 못한 지역에서도 floaters를 최소화하고 새로운 뷰의 품질을 개선합니다.

- **Technical Details**: 기존의 Maximum Likelihood(MAP) 접근법 대신, 본 기법은 관측된 데이터의 최적 모델 매개변수를 선택하는 Maximum-a-Posteriori(MAP) 방법론을 채택하고 있습니다. 이 방법은 훈련된 NeRF의 두꺼운 구조를 변경하지 않고, 특히 보지 못한 지역에서는 밀도가 0이 되어야 한다는 간단한 전역 사전 가정을 적용합니다. 또한, 효율성과 속도를 위해 추가적인 네트워크 훈련이나 메모리 사용을 요구하지 않습니다.

- **Performance Highlights**: 이 연구의 방법은 다른 NeRF 후처리(cleanup) 모델들과 비교할 때 2.5배 빠른 추론 시간과 30초 이내의 클린업 훈련 시간을 자랑합니다. 이 방법은 floaters를 효과적으로 제거하여 새로운 뷰 합성이 더욱 정확해지며, 시각적 품질을 유지하면서도 원래 NeRF 구조를 유지합니다.



New uploads on arXiv(cs.AI)

### Critical-Questions-of-Thought: Steering LLM reasoning with Argumentative Querying (https://arxiv.org/abs/2412.15177)
- **What's New**: 이번 논문에서는 논리적 및 수학적 추론에서 어려움을 겪고 있는 최신 인공지능 모델들을 개선하기 위해 새로운 접근 방식인 Critical-Questions-of-Thought (CQoT)를 도입합니다. CQoT는 이러한 모델들이 추론 과정을 점검하고 논리적 실수를 수정할 수 있도록 도와주는 비판적 질문을 활용하는 방법입니다. 이 연구는 Toulmin의 논증 모델을 기반으로 하여 인공지능의 사고 과정을 향상시키는 데 중점을 두고 있습니다.

- **Technical Details**: CQoT 접근방식은 논증 이론의 핵심 개념들을 활용하여 LLM의 사고 프로세스가 보다 선명하게 드러나도록 합니다. 이 과정에서 LLM은 제시된 전제(premises)로부터 결론(conclusion)을 이끌어내는 방식을 따릅니다. 즉, 주어진 데이터에 대한 주장이 적절한 근거(warrant)와 경험적 데이터로 뒷받침되어야 한다는 Toulmin의 관점을 취합니다.

- **Performance Highlights**: 이 연구에서는 MT-Bench Reasoning 및 Math 과제를 통해 CQoT 접근방식이 LLM의 기본 성능을 상당히 개선시킨다는 것을 보여줍니다. 더 나아가, '시간을 주는 것'이 논리적 사고를 향상시키는 데 긍정적인 기여를 한다는 기존의 테스트 타임 컴퓨트 가설을 입증하고 있습니다. 연구 결과와 평가 점수는 GitHub에서 확인할 수 있습니다.



### Probabilistic Strategy Logic with Degrees of Observability (https://arxiv.org/abs/2412.15135)
- **What's New**: 이번 논문은 불완전한 정보 하에서의 에이전트의 전략적 능력에 대한 기존 연구의 한계를 극복하기 위해 정보 투명성(information transparency) 관련 프로퍼티를 표현할 수 있는 새로운 논리적 프레임워크를 제시합니다. 특히, Stochastic Multi-Agent Systems(MAS)에서 에이전트 행동의 관찰 가능성을 측정할 수 있는 새로운 observability operators를 포함하여 Probabilistic Strategy Logic(PSL)을 확장하여 Opacity Probabilistic Strategy Logic(oPSL)을 소개합니다.

- **Technical Details**: oPSL은 에이전트의 전략들에 대한 결합을 고려하면서 시스템 행동의 투명성을 정량적으로 분석할 수 있도록 하는 새로운 접근 방식을 제공합니다. 이 프레임워크는 부분적으로 관찰 가능한 스토캐스틱 모델을 기반으로 하며, 이를 통해 오버블리티와 행동의 관찰 가능성을 동시에 다룰 수 있게 합니다. 또한, 오버블리티 관련 문제의 결정을 3EXPSPACE에서 가능하다는 점을 보여줍니다.

- **Performance Highlights**: 이 논문은 관찰 가능성과 정보 투명성 분석을 위한 체계적인 틀을 제공하며, 보안(security), 개인 정보 보호(privavy), 게임 이론(game theory) 및 AI와 같은 다양한 분야에 응용될 수 있는 가능성을 제시합니다. 이 프레임워크를 통해 MAS 내 에이전트의 의사결정 프로세스에 미치는 정보를 수량적으로 분석하고, 이를 통해 더 안전하고 효율적인 협력적 행동을 보장할 수 있습니다.



### Towards Friendly AI: A Comprehensive Review and New Perspectives on Human-AI Alignmen (https://arxiv.org/abs/2412.15114)
- **What's New**: 본 논문은 Friendly AI (FAI)의 개념을 종합적으로 검토하고, 윤리적 관점에서의 중요한 논의와 잠재적 적용에 대한 분석을 제공합니다. 특히, FAI는 AI의 안전한 발전을 보장하기 위한 이론적 틀로 제안되었으며, 이에 대한 명확한 정의가 제시됩니다. 논문에서는 eXplainable AI (XAI), 프라이버시, 공정성, 감정 컴퓨팅 (Affective Computing) 관점에서의 주요 응용 분야를 다루고 있습니다.

- **Technical Details**: FAI는 AI 시스템이 모든 상황에서 인류에게 이익이 되도록 설계하는 것을 목표로 하며, 이는 인간의 가치 및 윤리를 반영해야 함을 강조합니다. 이론 논의에서는 FAI의 지지와 반대 입장에 대한 다양한 관점이 제시되며, 기술적 구현 방법론으로 XAI와 프라이버시 보호 기술이 포함됩니다. 또한, AI의 편향을 식별하고 완화하기 위한 공정성 기술과 감정 분석에 대한 응용도 논의됩니다.

- **Performance Highlights**: FAI의 발전은 AI가 인간과 협력하여 발전할 수 있는 기반을 마련하고자 하는 노력의 일환으로 중요합니다. 이 논문은 FAI의 필요성과 발전 방향에 대한 체계적 검토를 목표로 하며, AI의 윤리적 개발 및 인류의 이익을 극대화하기 위한 다양한 기술적 및 윤리적 도전 과제를 규명하고 있습니다. FAI의 개념과 이론적 관점을 논의함으로써, AI 개발의 미래 방향에 대한 통찰을 제공합니다.



### Generalizing Constraint Models in Constraint Acquisition (https://arxiv.org/abs/2412.14950)
- **What's New**: 이 논문에서는 Constraint Acquisition (CA)의 한계를 극복하기 위해 GenCon이라는 새로운 접근 방식을 제안합니다. 기존 CA 방법들은 특정 문제 인스턴스에 대한 개별 제약 조건 집합만 학습하여 일반화할 수 없었지만, GenCon은 다양한 문제 인스턴스를 모델링할 수 있는 매개변수화된 제약 조건 모델을 학습하는 방법입니다. 이를 위해 통계적 기계 학습 기법을 활용하여 제약 조건을 분류하고, 이를 기반으로 해석 가능한 제약 조건 사양을 생성하는 방법을 제시합니다.

- **Technical Details**: 제약 만족 문제(CSP)는 세 가지 요소로 구성됩니다: 결정 변수(V), 도메인(D), 제약 집합(C)입니다. 본 논문에서는 제약 조건의 매개변수화 특성을 포착하기 위해 기본적인 요소를 정형화하고, 이러한 매개변수화 제약 조건 사양을 학습하기 위한 GenCon 접근 방식을 제안합니다. GenCon은 주어진 인스턴스의 제약 조건을 분석하여 가능한 매개변수화를 통해 제약 조건을 일반화하는 제어된 분류 기반 학습 방식을 사용하여 복잡한 함수들을 학습합니다.

- **Performance Highlights**: 실험 결과, GenCon 접근 방식이 높은 정확도를 달성하고 입력 인스턴스의 노이즈에 견고하다는 것을 보여주었습니다. 다양한 문제 클래스와 인스턴스를 대상으로 한 실험을 통해 접근 방식의 효과성을 입증하며, 이를 통해 새로운 문제 인스턴스에 대한 실제 지상 제약을 생성하는 능력을 보여주었습니다. 또한, 결정 규칙 추출이 가능한 분류기와 사용 가능한 제너레이트-앤드-테스트 방법을 통해 다양한 분류기와 호환이 가능합니다.



### Answer Set Networks: Casting Answer Set Programming into Deep Learning (https://arxiv.org/abs/2412.14814)
Comments:
          16 pages, 9 figures

- **What's New**: 이번 논문에서는 Answer Set Programming (ASP) 제약을 활용한 신경-상징 시스템의 한계점을 극복하기 위해 Answer Set Networks (ASN)라는 새로운 NeSy 솔버를 제안합니다. ASN은 Graph Neural Networks (GNN)를 기반으로 하여, ASP 기반의 Deep Probabilistic Logic Programming (DPPL)을 위한 확장 가능한 접근 방식을 제공합니다.

- **Technical Details**: ASNs는 ASP를 ASNs로 변환하는 방법을 제시하며, GPU의 배칭(batch) 및 병렬화(parallelization) 기능을 활용하여 인코딩된 문제를 효율적으로 해결할 수 있다는 점을 강조합니다. 이러한 접근 방식은 CPU에 기반한 기존의 NeSy 시스템보다 우수한 성능을 보여줍니다.

- **Performance Highlights**: 실험 평가 결과, ASNs는 여러 작업에서 최신 CPU 기반 NeSy 시스템을 능가하는 성능을 입증하였습니다. 또한 ASNs를 활용하여 Large Language Models (LLM)의 미세 조정(finetuning)과 드론의 '헌법적 네비게이션(constitutional navigation)'을 구현함으로써 공공 항공 법규를 인코딩하여 불확실한 환경에서 무인 항공기를 경로 조정하는 데 기여하고 있습니다.



### LTLf Synthesis Under Unreliable Inpu (https://arxiv.org/abs/2412.14728)
Comments:
          8 pages, to appear at AAAI2025

- **What's New**: 이번 연구에서는 LTLf 목표 사양을 실현하면서 특정 입력 변수가 신뢰할 수 없는 경우 최소한의 LTLf 백업 사양이 만족되도록 보장하는 전략을 연구합니다. 문제를 공식적으로 정의하고, 표준 LTLf 합성을 기준으로 최악의 경우 복잡도를 2EXPTIME-complete로 특성화합니다. 우리는 세 가지 해법 기법을 제시하여, 각 기법의 성능을 서로 비교 평가합니다.

- **Technical Details**: 우리는 신뢰할 수 없는 입력 변수의 존재 하에서도 기본 모델과 조심스러운 모델을 동시에 고려합니다. 각 모델에 기반하여 입력 변수가 제대로 작동할 경우 완전히 목표를 달성할 수 있도록 하며, 잘못 작동하더라도 몇 가지 보장 조건이 유지되도록 하는 전략을 개발하였습니다. 제안하는 세 가지 기법은 각각 2EXPTIME, 3EXPTIME 및 2EXPTIME의 복잡도를 가지고 있으며, qltlf 공식을 위한 직접 부호화가 가능합니다.

- **Performance Highlights**: 세 가지 기법의 실험적 평가 결과 이론적인 최악의 경우 경계가 관찰 성능으로 이어지지 않는다는 점이 흥미롭습니다. 베스트 성능은 MSO 기법이 보여 주었으며, 다음은 신뢰 구축 방법 및 직접 오토마타 조작이었습니다. 또한, 모든 기법의 구현은 비초과 요소성의 첫 번째 및 이차 논리에 기반한 툴 MONA를 사용했습니다.



### Creation of AI-driven Smart Spaces for Enhanced Indoor Environments -- A Survey (https://arxiv.org/abs/2412.14708)
Comments:
          39 pages, 3 figures, 1 table, journal

- **What's New**: 이 논문에서는 AI 기반 스마트 공간의 기초 구성 요소들에 대한 체계적인 조사를 제공합니다. 스마트 공간은 다양한 감지 및 통신 기술을 통합하여 공간의 기능을 향상시키고 에너지 효율성을 최적화하며 사용자의 편안함과 웰빙을 개선하는 환경으로 정의됩니다. 더불어, 머신 러닝(ML), 딥 러닝(DL), 대형 언어 모델(LLMs) 등의 기존 및 새로운 AI 방법론의 조합을 통해 스마트 공간의 발전 가능성을 탐구합니다.

- **Technical Details**: 스마트 공간에서는 센서 기술, 데이터 통신 프로토콜, 센서 네트워크 관리 및 유지 관리 전략, 데이터 수집 및 처리 방법 등이 핵심 기술로 활용됩니다. 본 논문은 다양한 스마트 공간 응용 프로그램을 실현하기 위한 ML 및 DL 기술의 필요성을 강조하며, 환경 모니터링 및 사용자 행동 예측의 중요성을 다룹니다. 특히 변환자 네트워크(transformer networks) 및 LLMs와 같은 최신 AI 연구 방법론이 스마트 공간을 지능형 생태계로 전환하는 데 기여할 수 있음을 강조합니다.

- **Performance Highlights**: AI 기반 스마트 공간은 활동 모니터링, 이상 탐지, 에너지 최적화 및 사용자 행동 예측 기능을 통해 사용자 경험을 향상시킬 수 있는 다양한 응용 프로그램을 지원합니다. 이러한 기술들은 IoT 프레임워크와 결합되어 지속적으로 사용자 행동으로부터 학습하며, 이를 통해 더욱 효율적이고 사용자 중심의 스마트 환경을 구현할 수 있습니다. 그러나 AI 기반 스마트 공간의 개발은 도전 과제도 존재하며, 이 논문에서는 이러한 문제점들과 이를 극복하기 위한 향후 연구 방향에 대해서도 논의합니다.



### Bel Esprit: Multi-Agent Framework for Building AI Model Pipelines (https://arxiv.org/abs/2412.14684)
- **What's New**: 이 논문은 인공지능(AI) 모델 파이프라인을 사용자의 요구에 맞추어 구축하기 위해 설계된 대화형 에이전트인 벨 에스프리(Bel Esprit)를 소개합니다. 벨 에스프리는 여러 모델이 협력하여 요구 사항을 명확히 하고, 파이프라인을 구축, 검증 및 적절한 모델로 채우는 다중 에이전트 프레임워크를 사용합니다. 이 프레임워크는 불명확한 사용자 쿼리에서 파이프라인을 생성하는 효과성을 증명하고 있으며, 오류 분석을 통해 파이프라인 구축의 지속적인 과제를 강조합니다.

- **Technical Details**: 파이프라인 생성은 사용자의 쿼리를 입력으로 받아 AI 기능의 파이프라인을 출력하는 구조적 예측 과제로 정의됩니다. 각 AI 기능은 음성 인식 언어와 같은 하나 이상의 매개변수를 가질 수 있으며, 최종 출력은 입력, 출력과 기능을 나타내는 노드와 데이터 흐름을 나타내는 엣지로 구성된 그래프 형식으로 표시됩니다. 다양한 유형의 노드(라우터, 결정 노드, 스크립트)를 소개하여 파이프라인의 기능성을 향상시킵니다.

- **Performance Highlights**: 벨 에스프리의 성능은 사용자 요구 사항에 따라 명확하게 구축된 파이프라인을 생성하는 데 중점을 두고 있으며, 이로 인해 다양한 입력과 출력에 대한 요구를 효과적으로 처리합니다. 각 서브 에이전트는 특정 단계를 담당하며, 문제가 발생할 경우 처음 단계로 돌아가 수정 과정을 반복합니다. 의미적 정합성을 검증하기 위해 LLM을 사용하는 검토 과정이 마련되어 있어, 사용자의 정해진 요구 사항과 파이프라인 흐름 간의 일치를 평가하는 데 기여합니다.



### Relational Programming with Foundation Models (https://arxiv.org/abs/2412.14515)
- **What's New**: 이 논문에서는 Vieira라는 선언적 프레임워크를 제안하여 다양한 AI 애플리케이션을 위한 기초 모델(foundation models)의 잠재력을 활용할 수 있는 통합 솔루션을 제공합니다. Vieira는 확률적 관계 패러다임(probabilistic relational paradigm)을 따르며, 기초 모델을 상태가 없는 함수로 취급하여 관계형 입력과 출력을 지원합니다.

- **Technical Details**: Vieira는 로직 프로그램(logic programs)과 신경 기호 응용(neuro-symbolic applications)을 매끄럽게 결합할 수 있도록 하며, 다양한 하위 모델의 조합을 간소화하여 복잡한 다중 모달 응용(multi-modal applications)을 지원합니다. 이 프레임워크는 Scallop 컴파일러를 확장하여 기초 모델을 플러그인(plugin)으로 지원하는 외부 인터페이스(foreign interface)를 구현합니다.

- **Performance Highlights**: 논문에서는 언어, 비전, 구조적 데이터 및 벡터 데이터베이스에 걸친 9개의 도전적인 작업을 통해 Vieira의 성능을 평가하였으며, Vieira의 프로그램은 간결하면서도 현대 기초 모델을 통합할 수 있고, 경쟁 기준 대안들에 비해 비슷하거나 더 나은 정확도를 보였습니다.



### The Digital Ecosystem of Beliefs: does evolution favour AI over humans? (https://arxiv.org/abs/2412.14500)
- **What's New**: 이 논문은 AI 시스템이 사회적 네트워크에 통합됨에 따라, AI가 생성한 콘텐츠가 웹에서 차지하는 비율과 영향을 제어하기 위한 최초의 진화적 프레임워크인 디지털 신념 생태계(Digico)를 제안합니다. 이 프레임워크는 다인종 상호작용을 가진 시뮬레이션된 사회적 네트워크에서의 통제된 실험 환경을 모델링합니다. Digico는 진화론적 접근을 통해 메시지 전략을 변화시키고, 감염 모델에 기반한 동력학을 통해 서로의 신념에 영향을 미치도록 설계되었습니다.

- **Technical Details**: Digico는 정책을 진화적 알고리즘(EAs)을 통해 발전시키며, 일반화된 인지 확산 모델의 연장으로 신념을 적응시키는 구조를 갖추고 있습니다. 이 프레임워크에서, 에이전트는 서로의 메시지를 브로드캐스팅하여 누적 보상을 최대화하는 것을 목표로 합니다. 이러한 방식은 오프라인 또는 온라인 환경에서 인지 전환을 모형화할 수 있도록 합니다.

- **Performance Highlights**: 초기 실험 결과, AI가 빠른 메시징과 진화, 추천 알고리즘에서 더 큰 영향을 미칠 때, 최대 95%의 조회수를 받을 수 있음을 보였습니다. 또한, 선전용으로 설계된 AI는 최대 50%의 인간이 극단적 신념을 채택하도록 설득할 수 있으며, 한정된 채널에 대한 신념이 있을 경우 85%까지 증가할 수 있었습니다. 이러한 결과를 바탕으로 극단적인 신념을 피하기 위한 조언도 제시되었습니다.



### FaultExplainer: Leveraging Large Language Models for Interpretable Fault Detection and Diagnosis (https://arxiv.org/abs/2412.14492)
- **What's New**: 이번 연구에서는 Tennessee Eastman Process (TEP)에서 결함 감지 및 진단(FDD)을 위한 새로운 도구인 FaultExplainer를 제안합니다. FaultExplainer는 대규모 언어 모델(LLMs)을 기반으로 실시간 센서 데이터 시각화, 주성분 분석(Principal Component Analysis, PCA) 기반 결함 감지, 그리고 주요 변수 식별 기능을 통합하여 대화형 사용자 인터페이스를 제공합니다. 이 시스템은 기존 데이터 기반 FDD 플랫폼의 해석 가능성을 개선하고, 새로운 결함의 근본 원인을 식별하는 데 도움을 주기 위해 개발되었습니다.

- **Technical Details**: FaultExplainer는 GPT-4o 및 o1-preview 모델을 사용하여 FDD의 해석 가능성을 향상시키기 위해 두 가지 시나리오에서 LLM의 추론 능력을 평가합니다. 첫 번째 시나리오는 역사의 근본 원인이 제공되는 경우이고, 두 번째 시나리오는 제공되지 않는 경우로, 이는 이전에 본 적이 없는 결함의 도전을 모방합니다. 이 시스템은 PCA 및 기능 중요도 분석을 기반으로 하여 결함 설명을 생성하며, 이는 화학 공정 엔지니어에게도 해석 가능하도록 정밀합니다.

- **Performance Highlights**: 실험 결과는 FaultExplainer가 플로트치적이고 실행 가능한 설명을 생성할 수 있는 강점을 보여주지만, PCA 선택 특징에 의존한다는 점과 가끔 허상(hallucination)을 생성할 수 있다는 한계도 드러냈습니다. 이 시스템은 개방형 소스 패키지로 제공되어, 사용자들이 이를 쉽게 접근하고 활용할 수 있도록 하였습니다. FaultExplainer의 주요 목표는 결함 감지 및 진단 과정에서 운영자의 직관적인 이해를 지원하는 것입니다.



### Mediation Analysis for Probabilities of Causation (https://arxiv.org/abs/2412.14491)
- **What's New**: 이 논문은 원인 확률(probability of causation, PoC)의 새로운 변형을 제안하여, 치료가 결과를 생성하는 데 필요한 정도와 충분한 정도를 정량화합니다. 본 연구에서 소개된 PoC 변형에는 조절된 직접 정밀도(CD-PNS), 자연 직접 정밀도(ND-PNS), 자연 간접 정밀도(NI-PNS)가 포함되어 있습니다. 이러한 새로운 메트릭은 다양한 원인 경로를 고려하여 치료의 효과를 평가하는 데 유용성을 제공합니다.

- **Technical Details**: 새로 제안된 PoC 메트릭에 대한 식별 정리(Identification Theorems)가 개발되어 관찰 데이터에서 이들의 추정을 가능하게 합니다. 저자들은 기존 PoC 문헌을 바탕으로 다양한 변수와 치료 메커니즘 간의 관계를 분석하기 위한 통계적 모델을 소개하며, 구조적 인과 모델(Structural Causal Models, SCM)을 사용하여 이러한 원리들을 뒷받침하고 있습니다. PoC 측정의 새로운 정의가 이론적 기반 위에서 제안되어, 인과 경로를 분석하는 데 필수적인 도구가 됩니다.

- **Performance Highlights**: 저자들은 심리학 데이터셋을 분석하여 제안한 PoC 메트릭의 실제 적용 사례를 보여줍니다. 다양한 인구 집단(subpopulation)에 대해 제안된 PoC 메트릭을 활용하여, 치료가 특정 사건을 발생시키기에 얼마나 필요한지, 그리고 충분한지를 측정하는 작업이 진행되었습니다. 이러한 결과는 PoC가 의사결정 시스템 및 AI 기반 시스템에서 중요한 역할을 한다는 것을 입증하는 데 기여합니다.



### Towards Projected and Incremental Pseudo-Boolean Model Counting (https://arxiv.org/abs/2412.14485)
Comments:
          To appear in AAAI25

- **What's New**: 이번 논문에서 소개된 PBCount2는 Pseudo-Boolean (PB) 모델 카운팅의 새로운 도구로, 기존 PB 카운터의 기능 부족 문제를 해결합니다. PBCount2는 projected 및 incremental 모델 카운팅을 지원하는 최초의 정확한 PB 모델 카운터입니다. 이 카운터는 Least Occurrence Weighted Min Degree (LOW-MD) 계산 순서 휴리스틱을 사용하여 projected 모델 카운팅을 지원하고, 캐시 메커니즘을 통해 incremental 모델 카운팅 기능을 구현합니다.

- **Technical Details**: PBCount2는 PB 제약 조건의 추가 및 제거를 통해 단계별로 모델 카운트를 연산하는 incremental setting을 지원하며, 이 과정에서 새로운 계산 순서 휴리스틱을 도입하여 모델 카운팅의 효율성을 높입니다. 또한, 이 카운터는 다양한 응용 설정에서 영감을 받은 벤치마크 인스턴스에 대해 평가를 수행하여 효과성을 입증했습니다. 이를 통해 PBCount2는 기존의 CNF 기반 카운터에 비해 높은 성능을 보여줍니다.

- **Performance Highlights**: 평가 결과, PBCount2는 1957개의 projected 모델 카운팅 인스턴스를 성공적으로 처리한 반면, 기존의 CNF 기반 카운터는 1398개만 처리했습니다. 또한, incremental 벤치마크에 대하여 PBCount2는 1618개의 인스턴스를 완료한 반면, 기존 PB 모델 카운터는 1371개에 그쳐 성능의 우위를 보여줍니다. 이러한 성과는 PB 모델 카운팅의 채택과 응용을 촉진할 것으로 기대됩니다.



### Multi-task Representation Learning for Mixed Integer Linear Programming (https://arxiv.org/abs/2412.14409)
- **What's New**: 이 논문은 머신러닝(ML)을 기반으로 한 혼합 정수 선형 프로그램(MILP) 해결을 위한 첫 번째 다중 작업 학습 프레임워크를 소개합니다. 이 프레임워크는 Gurobi 및 SCIP와 같은 솔버 간 그리고 Branching 및 Solver configuration과 같은 작업 간의 MILP 해결을 안내하는 데 유용한 MILP 임베딩을 제공합니다. 이를 통해 별도의 데이터 수집 및 훈련 과정 없이도 MILP 해결의 효율성을 향상시킬 수 있습니다.

- **Technical Details**: 제안된 다중 작업 학습 모델은 우선 공유 표현 레이어와 고정된 태스크 특정 출력 레이어로 구성된 두 단계 훈련 프로세스를 포함합니다. 첫 번째 단계에서는 MILP에 대한 공유 표현을 학습한 뒤, 두 번째 단계에서는 태스크 특정 레이어를 파인튜닝하면서 공유 표현 레이어는 고정한 상태로 유지합니다. 이 모델은 Backdoors, Predict-and-Search (PaS), 그리고 Solver Configurations의 세 가지 태스크를 이용하여 평가됩니다.

- **Performance Highlights**: 우리의 다중 작업 모델은 동일한 분포의 태스크 특정 모델과 유사한 성능을 보이며, 더 큰 문제 인스턴스에서 특히 뛰어난 일반화 능력을 보입니다. 또한, 다중 작업 학습을 통해 두 개의 태스크에서 훈련하고 세 번째 태스크에서 미세 조정을 수행한 결과, 특정 태스크에 맞춘 모델을 초과하는 성과를 이루어냈습니다.



### Clinical Trials Ontology Engineering with Large Language Models (https://arxiv.org/abs/2412.14387)
- **What's New**: 이 논문은 임상 시험 데이터를 비용 효율적이고 시간 효율적으로 추출하고 통합할 수 있는 간단하면서도 효과적인 방법론을 제안합니다. 현재 의료 산업에서 임상 시험 정보를 관리하는 것은 큰 도전 과제가 되고 있으며, 기존의 방법은 비효율적이었습니다. 이 연구는 특히 GPT3.5, GPT4 및 Llama3와 같은 대형 언어 모델(LLM)이 시간을 절약하고 비용을 줄이며 자동화된 방식으로 임상 시험 데이터를 처리할 수 있다는 것을 보여줍니다.

- **Technical Details**: 임상 시험 수행을 위한 이 연구에서는 50개의 임상 시험 데이터를 사용하여 O(n) 시간 복잡도를 갖는 새로운 온톨로지 병합 방법론을 제안합니다. LLM에 입력되는 프로세스를 포함하여, 각 임상 시험의 주요 및 부수적인 결과에 기반하여 온톨로지를 생성합니다. 연구에 사용된 LLM 모델은 GPT3.5, GPT4, 및 Llama3 (8b 및 70b)으로, OpenAI API를 통해 접근하거나 자체 호스팅을 통해 사용하였습니다.

- **Performance Highlights**: 연구 결과, LLM 사용으로 인해 시간과 비용 모두 현저하게 감소했으며, 특히 GPT4는 인간의 성능과 유사한 결과를 보여주었습니다. 이러한 성과는 현재 최첨단 LLM이 임상 시험 데이터 처리에 있어 실용성을 통해 의료 연구에서 실시간 데이터 통합의 새로운 기준이 될 수 있음을 시사합니다. 이 논문은 임상 연구 및 LLM의 통합이 향후 의료 분야에서 어떻게 활용될 수 있는지를 보여줍니다.



### Balans: Multi-Armed Bandits-based Adaptive Large Neighborhood Search for Mixed-Integer Programming Problem (https://arxiv.org/abs/2412.14382)
- **What's New**: 이번 논문에서는 Balans라는 새로운 적응형 메타 솔버를 제안합니다. 이 솔버는 Mixed-Integer Programming (MIP)을 처리하며, 기존의 기계 학습 기반 방법들이 갖고 있는 오프라인 학습에 대한 의존성을 극복합니다. Balans는 온라인 학습을 활용하여 MIP 문제를 해결하며, 다양한 이웃 정의를 통해 성능을 향상시킬 수 있습니다.

- **Technical Details**: Balans는 Adaptive Large Neighborhood Search (ALNS) 방식을 기반으로 하고 있으며, MIP 솔버 위에서 동작합니다. 본 시스템은 멀티암드 밴딧(mult-armed bandit) 알고리즘을 통해 실시간으로 문제 인스턴스에 맞는 다양한 이웃 정의를 선택합니다. 이 과정을 통해 Balans는 각 이웃의 성과를 즉시 평가하고 최적의 이웃을 탐색하여 문제 해결에 적용합니다.

- **Performance Highlights**: Balans는 기존 MIP 솔버에 비해 현저한 성능 향상을 보입니다. 특히, 모든 단일 이웃 정의에 의존하는 것보다 여러 이웃을 탐색하고 조합함으로써 더 우수한 성능을 보여줍니다. 또한, Balans는 다양한 최적화 문제의 어려운 인스턴스에서도 이전의 state-of-the-art LNS(MIP) 접근법을 능가하는 것으로 확인되었습니다.



### Python Agent in Lud (https://arxiv.org/abs/2412.14372)
- **What's New**: 이 연구에서는 Java로 구현된 Ludii 시스템에서 에이전트를 개발하기 위해 Python 인터페이스를 제공하며, 이를 통해 Python 모듈을 사용하여 일반 게임 플레이 에이전트를 구현할 수 있도록 합니다. 특히, jpy와 Py4J라는 두 가지 다른 Python-Java 통신 방법의 성능을 비교하고 분석하였습니다. 이 연구의 주요 목적은 jpy와 Py4J의 속도 차이를 평가하고, 엔진과 에이전트 간의 통신을 최적화하는 것입니다.

- **Technical Details**: 본 작업에서는 Minimax 알고리즘과 MCTS의 변형인 UCT를 구현하여 두 가지 Python-Java 통신 방법을 평가하였습니다. Minimax 알고리즘은 탐색 트리를 깊이 우선형으로 구축하며, alpha-beta 가지치기를 적용하여 평가하는 상태의 수를 줄여 최적의 수를 찾는 방식으로 작동합니다. 반면, MCTS는 통계적 접근 방식으로 게임 플레이를 하기 위해 무작위 시뮬레이션을 사용하는 트리 탐색 알고리즘이며, UCT 변형은 각 노드의 평균 보상에 대한 상한 신뢰 구간을 고려합니다.

- **Performance Highlights**: 성능 분석 결과, jpy 에이전트는 Py4J보다 빠르지만 네이티브 Java 에이전트보다는 느린 것으로 나타났습니다. 이러한 성능 분석에서는 다양한 깊이, 가지 수, 플레이 타임을 가진 30개의 조합 게임에 대해 시뮬레이션을 수행하였고, 결과적으로 jpy가 Py4J보다 성능이 우수하다는 예측 모델을 개발했습니다. 이 연구는 Python과 Java 간의 통신 방식 개선을 위해 공개 교육 자료와 리포지토리를 제공하고 있습니다.



### Scaling 4D Representations (https://arxiv.org/abs/2412.15212)
- **What's New**: 본 논문은 비디오에서 순수 자가 감독 학습(self-supervised learning)의 확대 가능성을 탐구합니다. 기존 연구가 의미론적 작업에 주목한 반면, 본 연구는 카메라 자세 추정(camera pose estimation), 포인트 및 객체 추적(point and object tracking), 깊이 추정(depth estimation) 등 비의미적(non-semantic) 비전 작업에 초점을 맞춥니다. 이 연구에서는 Video Masked Auto-Encoding (MAE)와 22B 매개변수를 가진 학습된 Transformer 비디오 모델을 통해 4D 작업의 성능 향상을 입증합니다.

- **Technical Details**: 우리는 모델을 동일한 주의 기반(readout) 구조를 사용하여 apples-to-apples 비교를 통해 평가했습니다. 연구에서 사용된 모델은 이미지와 비디오 모델 모두를 포함하며, MAE와 V-JEPA는 비디오 자가 감독 방식에서 좋은 성능을 보였습니다. 또한, 특정 데이터셋에서 모델 크기를 20M에서 22B까지 증가시키면서 논증된 일관된 성능 향상을 관찰했습니다.

- **Performance Highlights**: 논문 결과, 이미지 모델들은 경쟁력이 떨어졌으며, MAE 기반 비디오 모델들이 4D 작업에서의 성능을 크게 향상시킬 수 있음을 보여주었습니다. 특히, MAE는 기존의 자가 감독 없이 학습된 모델들과 비교할 때 탁월한 성능을 보였습니다. 이 연구의 성과는 2025년 초 공개될 새로운 MAE-VIT 모델들을 포함하며, 이러한 모델들은 4D 장면 표현을 향상시키는 데 중요한 기여를 할 것으로 기대됩니다.



### PRIMA: Multi-Image Vision-Language Models for Reasoning Segmentation (https://arxiv.org/abs/2412.15209)
Comments:
          Project page: this https URL

- **What's New**: 이번 연구는 다중 이미지 픽셀 기반 추론 세분화(multi-image pixel-grounded reasoning segmentation)의 새로운 작업을 제안합니다. 이는 두 개 이상의 이미지를 포함한 비교 질문에 대한 자연어 응답을 생성하며, 관련 객체와 부분에 대한 픽셀 수준의 기반을 함께 제공합니다. 또한, M4Seg라는 새로운 벤치마크를 구축하여 224K 이상의 질문-응답 쌍과 다중 이미지 시나리오에서 필요한 세밀한 추론을 지원합니다.

- **Technical Details**: PRIMA라는 새로운 LVLM은 픽셀 수준의 기반을 다양한 이미지와 결합하여 강력한 멀티 이미지 추론 기능을 통합했습니다. 이 모델은 여러 이미지에 걸쳐 세부적인 시각적 표현을 질의할 수 있는 효율적인 비전 모듈을 중심으로 하며, 25.3%의 TFLOPs 감소를 이뤘습니다. PRIMA는 명령 기반의 다중 이미지 적응 모듈을 통해 다중 이미지를 통한 세밀한 연관성을 추론합니다.

- **Performance Highlights**: 실험 결과, PRIMA는 최신 기법들보다 우수한 성능을 보였습니다. 이 모델은 다중 이미지 시나리오에서 자연어 응답과 문맥에 기반한 세분화를 생성할 수 있으며, 높은 정밀도의 픽셀 수준 추론을 유지하면서도 계산 효율성을 크게 향상시켰습니다. 이는 다양한 응용 분야에서 LVLM의 해석 가능성을 증가시키는 데 기여할 수 있습니다.



### LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks (https://arxiv.org/abs/2412.15204)
Comments:
          25 pages, 13 figures

- **What's New**: 이 논문에서는 LongBench v2라는 새로운 벤치마크가 소개됩니다. 이 벤치마크는 LLMs(대형 언어 모델)가 실제 멀티태스크 문제를 처리하는 데 필요한 깊은 이해와 추론 능력을 평가하기 위해 설계되었습니다. LongBench v2는 8천에서 200만 단어까지의 문맥을 가진 503개의 도전적인 객관식 질문으로 구성되어 있습니다.

- **Technical Details**: 이 벤치마크는 단일 문서 QA, 다중 문서 QA, 장기간 맥락 학습, 장기 대화 역사 이해, 코드 저장소 이해 및 장기 구조화 데이터 이해의 여섯 가지 주요 작업 범주를 포함합니다. 데이터는 전문 배경이 다양한 거의 100명의 고학력 개인으로부터 수집하였으며, 품질과 난이도를 유지하기 위해 자동화된 리뷰와 수동 리뷰 과정을 모두 사용하였습니다.

- **Performance Highlights**: 인간 전문가들은 15분의 시간 제약 하에 53.7%의 정확도만 달성했으며, 가장 성능이 좋은 모델도 직접 질문에 대답했을 때 50.1%의 정확도에 그쳤습니다. 반면, 더 긴 추론을 포함하는 o1-preview 모델은 57.7%의 정확도를 기록하여 인간의 기준을 4% 초과함으로써, LongBench v2의 장기 맥락 문제를 해결하기 위해서 향상된 추론 능력과 추론 시간 컴퓨팅을 확대하는 것이 중요함을 보여주고 있습니다.



### DI-PCG: Diffusion-based Efficient Inverse Procedural Content Generation for High-quality 3D Asset Creation (https://arxiv.org/abs/2412.15200)
Comments:
          Project page: this https URL

- **What's New**: DI-PCG는 일반 이미지 조건에서 효율적인 역 절차적 콘텐츠 생성(Inverse Procedural Content Generation, I-PCG)을 위한 혁신적인 방법론을 제시합니다. 이 방법은 경량의 diffusion transformer 모델을 기반으로 하며, PCG 매개변수를 직접 비노이즈 타겟으로 취급하고, 관찰된 이미지를 매개변수 생성을 제어하는 조건으로 사용합니다.

- **Technical Details**: DI-PCG는 단 7.6M의 네트워크 매개변수와 30 GPU 시간만으로 훈련할 수 있으며, 몇 초 이내에 샘플을 생성할 수 있습니다. 이 모델은 iterative denoising score-matching 훈련을 통해 절차적 생성기의 매개변수 공간을 학습하고, 관찰된 이미지를 기반으로 수치적으로 샘플링을 수행합니다. 이렇게 생성된 매개변수는 PCG에 투입되어 고품질의 3D 자산을 생성하게 됩니다.

- **Performance Highlights**: DI-PCG는 효율성과 효과성을 모두 갖춘 시스템으로, 현실 세계의 데이터에 대한 일반화 능력 또한 뛰어납니다. 정량적 및 정성적 실험 결과를 통해 자산 생성 및 역 PCG 작업에서 DI-PCG의 효과가 명확히 검증되었습니다. 이 방법은 3D 자산 생성을 위한 파라미트릭 모델을 사용하는 효율적인 역 PCG를 가능하게 하며, 후속 애플리케이션을 위한 고품질 3D 자산 생성을 지원합니다.



### LlamaFusion: Adapting Pretrained Language Models for Multimodal Generation (https://arxiv.org/abs/2412.15188)
- **What's New**: LlamaFusion introduces a novel framework that enhances pretrained text-only large language models (LLMs) like Llama-3 with multimodal generative capabilities, allowing them to process and generate both text and images. 이 프레임워크는 Llama-3의 가중치를 활용하고, 이미지 처리를 위한 추가적인 transformer 모듈을 도입하여 텍스트와 이미지를 각각 처리합니다. 이를 통해 LlamaFusion은 텍스트 전용 모델의 언어 능력을 유지하면서도 강력한 시각 이해 및 생성 기능을 개발할 수 있게끔 합니다.

- **Technical Details**: LlamaFusion은 별도의 모듈에서 텍스트와 이미지를 각각 처리하도록 구성되어 있으며, 공통의 self-attention 층을 통해 두 개의 모달리티 간 상호작용을 가능하게 합니다. 훈련 과정에서 텍스트 관련 모듈은 고정되고 이미지 관련 모듈만 훈련하여, 이전의 언어 능력을 손상시키지 않으면서도 시각적 이해를 키웁니다. 또한, LlamaFusion은 기존의 text-only LLM에서 이미지를 이해하고 생성하는 능력을 효과적으로 훈련할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과 LlamaFusion은 이미지 이해에서 20%, 이미지 생성에서 3.6% 개선된 성능을 보이며, 전량 50%의 FLOPs만으로도 Llama-3의 언어 성능을 유지합니다. LlamaFusion의 성능은 Transfusion 모델에 비해 11.6% 더 우수하여 기존 시각-언어 모델을 적응시킬 수 있는 가능성을 보여줍니다. 이를 통해 LlamaFusion은 멀티모달 모델 개발의 효율적인 방향을 제시합니다.



### Human-Humanoid Robots Cross-Embodiment Behavior-Skill Transfer Using Decomposed Adversarial Learning from Demonstration (https://arxiv.org/abs/2412.15166)
Comments:
          9 pages, 8 figures. Accepted by IEEE Robotics and Automation Magazine

- **What's New**: 이번 논문에서는 휴머노이드 로봇의 행동 및 기술 전이를 위한 크로스-엠보디먼트(cross-embodiment) 프레임워크를 제안합니다. 이 프레임워크는 고유한 디지털 인간 모델을 사용하여 데이터 병목 현상을 줄이는 동시에 다양한 휴머노이드 로봇 간에 효과적인 기술 전이를 가능하게 합니다. 이는 정확한 동작을 위해 로봇의 구성 요소를 기능별로 세분화하여 독립적으로 훈련시키는 것을 포함합니다.

- **Technical Details**: 제안된 프레임워크는 운동 기능을 바탕으로 하여 다중 자유도(DoFs)를 가진 로봇의 동작을 리타겟팅하는 기법을 활용합니다. 이를 통해 동작의 정확성 및 효율성을 높이며, 현실 세계의 규모 차이를 고려하여 트레일의 정규화를 수행합니다. 또한, 인간-객체 간의 상호작용 그래프를 사용하여 동적 계획 수립 및 조정된 loco-manipulation 기술을 수행할 수 있습니다.

- **Performance Highlights**: 이 프레임워크는 다양한 구성의 5개 휴머노이드 로봇에서 테스트되어 안정적인 locomotion과 manipulation을 나타내었습니다. 실험은 데이터 요구 사항을 줄이고, 다른 플랫폼 간의 기술 전이 효율성을 높이는 데 효과적임을 입증했습니다. 이러한 결과는 다양한 로봇과의 상호 운용성을 향상시키는 중요한 기초를 제공합니다.



### Operationalising Rawlsian Ethics for Fairness in Norm-Learning Agents (https://arxiv.org/abs/2412.15163)
Comments:
          14 pages, 7 figures, 8 tables (and supplementary material with reproducibility and additional results), accepted at AAAI 2025

- **What's New**: 이번 논문에서는 사회적 규범(Social norms)이 특정 에이전트의 억압으로 이어질 수 있는 상황을 다룹니다. RAWL-E라는 새로운 방법론을 제시하여 윤리적인 규범 학습 에이전트를 생성하는 방식을 탐구합니다. 이 방법론은 Rawlsian ethics의 공정성 원칙인 maximin을 적용하여 사회적 웰빙과 개인 목표 간의 균형을 맞추는 것을 목표로 합니다.

- **Technical Details**: RAWL-E 에이전트는 결정을 내리는 과정에서 maximin 원칙을 활용하며, 이를 통해 윤리적인 규범을 촉진합니다. 이 논문에서는 시뮬레이션된 수확 시나리오를 통해 RAWL-E 에이전트의 성능을 평가합니다. RAWL-E agent들이 형성하는 규범이 사회적 복지(social welfare), 공정성(fairness), 견고함(robustness)을 향상시키는 데 기여함을 보여줍니다.

- **Performance Highlights**: RAWL-E 에이전트는 비-Rawlsian 에이전트 사회에서 나타나는 규범에 비해 더 높은 최소 경험(minimum experience)을 달성합니다. 이는 RAWL-E 에이전트들이 사회적 규범을 도입함으로써 개인과 사회 전체의 이익을 동시에 증진하는 데 성공하고 있음을 의미합니다.



### Language Models as Continuous Self-Evolving Data Engineers (https://arxiv.org/abs/2412.15151)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 데이터 자생적 생성 및 세척의 새로운 패러다임인 LANCE를 제안합니다. LANCE는 LLM이 스스로 데이터 생성, 검토 및 주석을 달 수 있도록 하여, 높은 품질의 데이터 확보에 있어 기존 방식에 비해 많은 시간과 비용을 절감합니다. 이 최초의 자율 데이터 엔지니어링 접근법은 LLM이 지속적으로 자기 진화를 이룰 수 있는 가능성을 보여줍니다.

- **Technical Details**: LANCE는 LLM이 기존의 데이터셋을 검토하고, 낮은 품질의 데이터를 보완하며, 선호 정보를 갖춘 고품질의 데이터를 구조화하는 과정을 통해 학습합니다. 이를 위해 모델은 텍스트 명령과 응답을 생성함으로써 응답의 질을 개선하고, 자율적으로 새로운 데이터를 수집하여 반복적으로 성능을 향상시킬 수 있습니다. 이러한 방법은 사람의 개입 없이 모델 스스로 데이터 생성을 완전한 사이클로 수행할 수 있게 합니다.

- **Performance Highlights**: LANCE는 Qwen2 모델을 대상으로 한 여러 작업에서 평균 점수를 3.36, 2.70만큼 향상시킴으로써 성능 개선을 입증합니다. LANCE의 반복적인 데이터 자생적 공정은  다양한 작업에서 LLM의 지능을 지속적으로 향상시키는 데 기여하며, 이는 기존의 감독형 모델학습(Supervised Fine-Tuning)과 비교해 더 일관된 결과로 나타납니다. 이는 미래의 초지능 시스템 개발에 큰 기여를 할 것으로 기대됩니다.



### Leveraging Color Channel Independence for Improved Unsupervised Object Detection (https://arxiv.org/abs/2412.15150)
Comments:
          38 pages incl. references, 16 figures

- **What's New**: 이 논문에서는 객체 중심 아키텍처(object-centric architectures)가 RGB 색상 공간으로 인코딩된 이미지들을 통해 객체 표현을 독립적으로 추출할 수 있는 능력을 강조합니다. 연구자들은 RGB 이미지가 성능 향상에 최적이라고 여겨지는 전통적 견해에 도전하며, HSV와 같은 다른 색상 공간이 객체 중심 표현 학습(object-centric representation learning)에 필수적인 특성을 지닌다는 점을 논의합니다. 또한, 충족해야 하는 색상 채널을 예측하도록 요구함으로써 모델의 성능이 개선된다는 것을 보여줍니다.

- **Technical Details**: 본 연구는 Slot Attention(SA) 기술적 배경을 활용하여 객체 중심 대표 학습을 위한 실험을 진행합니다. 입력 데이터를 단일 고정 크기 잠재 벡터로 변환하는 전통적인 인코더-디코더 아키텍처와 달리, SA는 개별 객체로 분해된 잠재 공간을 유도하고, 입력에 대해 경쟁을 통한 정보를 분산하여 슬롯(slot)을 업데이트하는 방식으로 작동합니다. 연구에서는 색상 공간을 조합하여 새로운 복합 색상 공간(composite color spaces)을 창출하며, 이 복합 색상 공간이 비효율적인 기존의 RGB 색상 공간보다 효과적이라는 것을 입증합니다.

- **Performance Highlights**: 논문에서는 제안한 복합 색상 공간이 기존의 RGB 색상 공간에 비해 다섯 개의 다중 객체 데이터셋에서 객체 탐지(object detection) 및 속성 분리(property disentanglement) 성능에서 크게 개선됨을 보여줍니다. 새로운 색상 공간은 모델 독립적이며 일반화 가능하고 효율적인 특성을 가지고 있어 다양한 비주얼 컴퓨팅 작업에 적용할 수 있으며, 객체 중심 학습(object-centric learning)을 넘어서는 컴퓨터 비전 과제에 대한 추가 연구를 촉진할 것으로 기대됩니다.



### Jet: A Modern Transformer-Based Normalizing Flow (https://arxiv.org/abs/2412.15129)
- **What's New**: 이번 연구에서는 coupling-based normalizing flows의 디자인을 재검토하며, 이전 모델에서 사용된 일반적인 설계 선택지를 세심하게 분석한다. 특히, 기존의 convolutional neural networks 대신 Vision Transformer 아키텍처를 적용하여 성능을 개선하였다. 이러한 접근은 더 단순하고 효율적인 구조를 가능하게 하며, 최신 모델들과의 비교에서도 뒤처지지 않도록 돕는다.

- **Technical Details**: Jet 모델의 구조는 매우 간단하며, 입력 이미지를 K개의 플랫 패치로 분할한 후, affine coupling layers를 반복적으로 적용하여 구성된다. 각 coupling layer의 입력은 벡터 형태로 변환되며, 이 과정을 통해 차원 분할 변환이 이루어진다. Scaling factor s와 bias b는 깊은 신경망을 통해 학습되며, 이를 통해 계산된 출력이 최종 샘플로 사용된다.

- **Performance Highlights**: 최종 모델은 간소화된 구조에도 불구하고, 일반적인 이미지 벤치마크에서 negative log-likelihood 기준으로 SOTA 결과를 달성했다. 또한, transfer learning 기법을 활용함으로써 과적합(overfitting)을 효과적으로 줄이는 데 성공하였다. 연구 결과는 strong normalizing flow 모델들이 더 강력한 생성 모델의 빌딩 블록으로 활용될 수 있음을 보여준다.



### Adaptive Pruning for Large Language Models with Structural Importance Awareness (https://arxiv.org/abs/2412.15127)
Comments:
          12 pages, 6 figures, 12 tables

- **What's New**: 최근 대형 언어 모델(LLMs)의 발전은 자연어 이해와 생성 능력을 크게 향상시켰습니다. 그러나 이 모델들은 고성능의 컴퓨팅 및 저장 자원을 요구하기 때문에 자원이 제한된 엣지 기기에서 배포하기 까다롭습니다. 이를 해결하기 위해, 본 논문에서는 구조적 인식 적응형 가지치기(Structure-aware Adaptive Pruning, SAAP) 방법을 제안하여 모델 성능을 유지하면서 계산 및 메모리 비용을 줄이는 새로운 접근 방식을 소개합니다.

- **Technical Details**: SAAP는 모든 결합된 구조의 중요성을 평가하기 위한 적응형 중요도 융합 메트릭을 정의합니다. 이는 불확실성을 고려하여 특정 계층이 가지치기되어야 하는지를 정하기 위해 모듈의 중요도를 순위로 매깁니다. 또한 SAAP는 추론 효율을 높이기 위해 새로운 그룹 내 세부 조정 방식을 개발하였으며, 다양한 LLMs에 걸쳐 실험을 수행하여 성능을 평가했습니다.

- **Performance Highlights**: 실험 결과, SAAP 방법은 여러 최신 기준선 방법들보다 우수한 성능을 보여주었으며, LLaMA-7B, Vicuna-7B, LLaMA-13B 모델에서 각각 2.17%, 2.37%, 2.39%의 정확도 향상을 기록했습니다. 또한 SAAP는 토큰 생성 속도를 5% 개선하여 자원이 제한된 환경에서의 실용적인 이점을 보여주고 있습니다.



### Outcome-Refining Process Supervision for Code Generation (https://arxiv.org/abs/2412.15118)
Comments:
          18 pages, 5 figures, Code: this https URL

- **What's New**: 이 논문에서는 Outcome-Refining Process Supervision, 즉 ORPS라는 새로운 패러다임을 제안합니다. 이 방법은 아웃컴(Outcome) 개선 그 자체를 감독해야 할 프로세스로 간주하며, 이를 통해 모델이 다양한 솔루션 경로를 탐색할 수 있도록 합니다. 연구 결과, 이 접근법은 더 작은 모델에서도 높은 성공률과 성능 지표를 달성할 수 있도록 해줍니다. 이를 통해 모델 개발에서의 복잡한 문제를 해결하는 데 중요한 실마리를 제공합니다.

- **Technical Details**: ORPS 프레임워크는 실행 피드백을 활용하여 추론 단계의 감독을 지지합니다. 또한, 트리 구조의 탐색 공간을 통해 모델이 여러 해결 전략을 동시에 유지하며 탐색할 수 있도록 돕습니다. 연구에서는 LLMs(대형 언어 모델)의 복잡한 프로그래밍 작업에서의 성능을 개선하기 위해 실행 신호를 강력한 기준으로 사용합니다. 이로 인해 기존의 학습된 보상 모델(PRMs)의 필요성을 줄이고 더욱 신뢰할 수 있는 검증 시스템을 구축할 수 있습니다.

- **Performance Highlights**: 실험 결과, ORPS는 세 가지 데이터셋과 다섯 개의 모델에서 평균 26.9%의 정확도 향상과 42.2%의 효율성 증가를 보여줍니다. 이 접근법은 적은 데이터로도 신뢰할 수 있는 검증을 제공하며, 특히 복잡한 작업에서 기존 방법들이 어려움을 겪는 지역적 개선 이상의 혁신을 보여줍니다. 이러한 결과는 제공된 구조화된 추론 공간과 구체적인 검증 신호가 복잡한 프로그래밍 작업을 해결하는 데 필수적임을 시사합니다.



### Associative memory inspires improvements for in-context learning using a novel attention residual stream architectur (https://arxiv.org/abs/2412.15113)
Comments:
          18 pages, 6 figures, 3 tables

- **What's New**: 이번 논문에서는 대규모 언어 모델(LLM)의 문맥 내 학습(in-context learning, ICL) 능력과 생물학적 기억 체계의 연결을 제안합니다. 주목할 만한 점은 LLM의 핵심 구성 요소인 attention mechanism이 현대의 연상 기억 모델들과 유사하다는 점입니다. 이러한 연결고리를 통해, 연구자들은 attention head 간에 정보가 직접 전달될 수 있도록 하는 새로운 잔차 스트림 아키텍처를 도입합니다.

- **Technical Details**: 이 논문에서 제안하는 연상 기억 모델은 ICL을 기반으로 한 분류 작업을 수행할 수 있습니다. 이 모델은 입력 데이터에 맞춰 attention 값이 직접 입력을 반영하도록 하여, 두 층 구조의 Transformer에서 ICL을 더욱 빠르게 구현할 수 있게 합니다. 이와 함께 작은 언어 모델에서도 성능 향상을 확인하며, 주어진 작업에서 attention head의 값을 중점적으로 분석합니다.

- **Performance Highlights**: 제안된 새로운 아키텍처는 ICL 능력이 훈련 과정에서 전통적인 방법보다 빠르게 나타남을 보여줍니다. 연구진은 800만 개의 매개변수를 가진 작은 언어 모델에서 개선된 ICL 성능을 관찰하였으며, 이는 고차원적이고 자연스러운 데이터에도 적용 가능한 결과임을 의미합니다. 이러한 접근 방식은 LLM의 해석 가능성에도 기여할 것으로 기대됩니다.



### Exploiting sparse structures and synergy designs to advance situational awareness of electrical power grid (https://arxiv.org/abs/2412.15105)
Comments:
          PhD thesis

- **What's New**: 본 논문은 기존의 상황 인식 도구의 강건성과 효율성을 향상시키기 위한 새로운 패러다임, 즉 Physics-ML Synergy를 제시합니다. 이는 물리 기반(physics-based) 방법과 데이터 기반(data-driven) 방법의 장점을 결합하여 발전된 상황 인식 능력을 제공합니다. 시스템의 전송 및 배급 모두에 적용 가능한 회로 모델링(circuit formulation)을 기반으로 하여, 두 세계의 통합을 통해 복잡한 전력망의 그리드 운용을 최적화할 수 있습니다.

- **Technical Details**: 이 연구의 핵심은 회로 기반 모델링과 희소 구조(sparsity)를 활용하는 것입니다. 이는 서로 다른 상황 인식 도구의 강건성과 효율성을 증대시키는 데 기여합니다. 희소 최적화(sparse optimization)는 이러한 도구들이 무작위 위협에 강한 내성을 가지도록 하며, 전력망의 주된 정전 원인과 데이터 오류를 정확히 파악할 수 있도록 합니다.

- **Performance Highlights**: Physics-ML Synergy를 통해 본 연구는 특정 사이버 위협에 대한 강건성과 효율성을 더욱 강화할 수 있는 가능성을 보여줍니다. 경량 머신러닝(ML) 모델을 개발하여 예측 및 탐지 능력을 물리 기반 도구와 보완하며, 이러한 경량 설계는 일반화(generalization)와 확장성(scalability)을 촉진합니다. 결과적으로, 전력망의 위협 사건에 대한 효과적인 대응과 함께 지속 가능한 저탄소 경제로의 전환을 지원하는 방향으로 나아갈 수 있습니다.



### A Cross-Domain Study of the Use of Persuasion Techniques in Online Disinformation (https://arxiv.org/abs/2412.15098)
- **What's New**: 이 연구는 다양한 도메인에서 사용되는 설득 기법의 역할을 대규모로 분석하며, 16가지 설득 기법을 조명합니다. 특히 기후 변화에 관한 허위 정보의 경우 언어적, 심리적, 문화적 요인이 어떻게 설득 전략에 영향을 미치는지를 통찰력 있게 다룹니다. 데이터셋은 다양한 원천에서 수집된 신문 기사를 포함하여 제안된 최신 설득 기법 분류기를 통해 분석되었습니다.

- **Technical Details**: 본 연구는 SemEval-2023에서 도입된 세계 최대의 설득 기법 주석 데이터셋을 활용합니다. 이 데이터셋은 9999가지 언어로 작성된 뉴스 기사에 대해 문장 수준에서 232개의 설득 기법으로 주석이 달려 있습니다. 연구에서는 Ma리와 함께 세 가지 도메인(코로나19, 기후 변화, 반이슬람)에 대한 데이터셋을 분석하며, 각 문장에서 설득 기법을 식별하기 위해 최첨단 설득 기법 분류기를 사용했습니다.

- **Performance Highlights**: 분석 결과 모든 허위 정보 도메인에서 'Loaded Language'와 'Doubt'가 널리 사용되며, 기후 변화 도메인에서 'Loaded Language'의 사용 비율이 감소했습니다. 특정 도메인에서의 설득 기법의 상대적 사용 빈도는 통계적으로 유의미한 차이를 보이며, 특히 이슬람 문제에서는 반복적 표현이 유의미하게 많이 나타났습니다. 이 연구의 결과는 허위 정보의 이해도를 높이고 그에 대응할 수 있는 중요한 통찰을 제공합니다.



### A Full Transformer-based Framework for Automatic Pain Estimation using Videos (https://arxiv.org/abs/2412.15095)
- **What's New**: 이번 연구에서는 환자의 고통을 줄이고 신뢰할 수 있는 평가를 제공하는 최적의 통증 관리 시스템을 설계하기 위한 자동 통증 추정의 중요성을 강조합니다. 본 논문은 Transformer in Transformer (TNT) 모델과 크로스-어텐션(cross-attention) 및 셀프-어텐션(self-attention) 블록을 활용한 새로운 풀 변환기 기반 프레임워크를 제안합니다.

- **Technical Details**: BioVid 데이터베이스의 비디오를 활용하여 모델의 성능을 상세히 설명합니다. Transformer in Transformer 구조는 각각의 어텐션 블록을 진행하면서 통증 평가 작업의 효율성을 크게 향상시키며, 이를 통해 각 작업에서의 일반화 능력을 보여주는 것을 목표로 합니다.

- **Performance Highlights**: 최신 기술(state-of-the-art) 성능을 나타내며, 제안된 모델이 모든 주요 통증 추정 작업에서 효과적이고 효율적임을 입증합니다. 이는 통증 관리 시스템 설계에서 중요한 발전을 이룬 것으로 평가됩니다.



### Learning Disentangled Equivariant Representation for Explicitly Controllable 3D Molecule Generation (https://arxiv.org/abs/2412.15086)
Comments:
          AAAI 2025

- **What's New**: 이 논문에서는 약물과 같이 처방 가능한 3D 분자의 조건부 생성 문제를 다룬다. 기존의 생성 모델보다 향상된 E(3)-equivariant Wasserstein 오토인코더(E3WAE)를 제안하면서 분자의 속성과 구조적 문맥을 분리하는 새로운 접근 방식을 제공합니다. 이로 인해 약물 설계 시 다양한 속성을 정확하게 조절할 수 있게 되었습니다.

- **Technical Details**: 제안하는 모델은 분리된 표현 학습(disentangled representation learning)을 통해 생성 모델의 잠재 공간을 두 가지 측면으로 나눠 분자의 속성과 3D 분자의 유효성을 보장하는 구조적 문맥을 각각 반영합니다. Wasserstein 정규화 손실을 통해 잠재 변수의 독립성을 보장하며, 새롭게 도입된 좌표 예측 손실을 통해 구조적 정렬을 할 수 있는 자동회귀 접근 방식을 채택하고 있습니다.

- **Performance Highlights**: 이 연구는 E3WAE 모델이 속성 타겟 생성(property-targeting generation) 및 문맥 보존 생성(context-preserving generation)에서 기존의 방법들과 비교하여 우수한 성능을 보인다는 점에서 중요합니다. 특히, 약물 설계 및 구조 기반 약물 발견에 있는 실제 응용에서 이 모델의 적합성을 테스트하여 유의미한 결과를 얻었습니다.



### AceMath: Advancing Frontier Math Reasoning with Post-Training and Reward Modeling (https://arxiv.org/abs/2412.15084)
- **What's New**: 이 논문에서는 복잡한 수학 문제를 해결하는 데 탁월한 수학 모델 모음인 AceMath를 소개합니다. 이 모델은 생성된 솔루션을 평가하고 올바른 것을 신뢰성 있게 식별할 수 있는 효과적인 reward 모델을 포함합니다. 연구진은 일반 도메인에서 경쟁력을 갖춘 성능을 달성한 후, 신중히 선별된 프롬프트와 합성된 응답을 사용하여 수학 도메인에 맞춰 조정하는 감독적 세부 조정(Supervised Fine-Tuning, SFT) 프로세스를 제안합니다.

- **Technical Details**: AceMath-72B-Instruct 모델은 Qwen2.5-Math-72B-Instruct, GPT-4o 및 Claude-3.5 Sonnet과 비교하여 뛰어난 성능을 보입니다. 이를 위해 AceMath-RewardBench라는 포괄적이고 견고한 벤치마크를 구축하여 다양한 문제와 난이도 수준에 걸쳐 수학 reward 모델을 평가합니다. 연구자들은 신뢰성 있는 수학 reward 모델을 구축하기 위한 체계적인 접근 방식도 제시합니다.

- **Performance Highlights**: AceMath-72B-RM 모델은 기존의 최첨단 reward 모델들을 지속적으로 능가하며, AceMath-72B-Instruct와 AceMath-72B-RM을 결합할 경우 수학 추론 벤치마크에서 최고 평균 rm@8 점수를 달성합니다. 연구진은 AceMath-Instruct 및 AceMath-RM의 모델 가중치, 훈련 데이터를 오픈 소스로 공개할 예정이며, AceMath-RewardBench를 통해 다양한 데이터셋과 난이도를 제공할 것입니다.



### GIRAFE: Glottal Imaging Dataset for Advanced Segmentation, Analysis, and Facilitative Playbacks Evaluation (https://arxiv.org/abs/2412.15054)
Comments:
          18 pages, 8 figures

- **What's New**: GIRAFE는 성대의 고속 비디오 내시경 시퀀스에 대한 의미적 분할 준비가 된 공개 데이터셋이 부족한 문제를 해결하기 위해 설계된 데이터 저장소입니다. 이 저장소에는 50명의 환자로부터 수집된 65개의 고속 비디오 내시경 녹화본이 포함되어 있으며, 이는 건강한 대조군과 목소리 장애를 진단받은 환자, 건강 상태가 알려져 있지 않은 환자 각각의 데이터를 모두 포함합니다. 모든 데이터는 전문가에 의해 수동으로 주석이 달린 세그멘테이션 마스크를 포함하고 있습니다.

- **Technical Details**: GIRAFE 데이터셋은 4,000 fps의 샘플링 주파수에서 성대의 빠른 동적 움직임을 정확하게 포착하는 고속 비디오 내시경(HSV) 기술을 기반으로 합니다. 이 데이터셋은 성대의 의미적 분할 및 분석을 위한 자동 세그멘테이션 접근 방식을 혼합하여 제공하며, 다양한 state-of-the-art 방법으로 보강됩니다. 이 데이터는 성대 병리의 진단 및 치료 효과 평가에 중요한 역할을 할 수 있습니다.

- **Performance Highlights**: GIRAFE 데이터셋은 새로운 세그멘테이션 알고리즘 개발에 이미 사용되었고, 성대의 진동 패턴 분석에 있어 유용성을 입증하였습니다. 기존의 연구들은 FN(Normal and Pathological) 진동을 높은 정확도로 분류할 수 있는 컴퓨터 기반 접근 방식을 개발하여 이 데이터셋이 성대 질환 검출을 위한 중요한 자원임을 강조하고 있습니다. 그러나 성대 영역의 정확하고 완전한 자동 의미적 분할 방법을 구축하기 위한 과제가 여전히 남아있습니다.



### Measuring, Modeling, and Helping People Account for Privacy Risks in Online Self-Disclosures with AI (https://arxiv.org/abs/2412.15047)
Comments:
          31 pages, 5 figues, Accepted for publication at CSCW 2025

- **What's New**: 이 논문은 레딧(Reddit)과 같은 익명 온라인 포럼에서 개인의 공개(self-disclosure)가 가져오는 이점과 사생활 위험 간의 균형을 어떻게 도울 수 있는지를 탐구합니다. 기존의 자연어 처리(NLP) 도구는 사용자가 자신의 글에서 위험한 공개를 인식하도록 돕지만, 실제 사용자와의 평가가 부족했습니다. 본 연구는 이러한 도구가 사용자에게 유용하도록 개선하기 위한 실질적인 피드백을 수집하는 데 중점을 둡니다.

- **Technical Details**: 연구에서는 N = 21명의 레딧 사용자가 두 개의 자작 게시물에 대해 최신 NLP 공개 탐지 모델을 사용하도록 하였습니다. 이 모델은 포괄적인 게시물이 아닌 텍스트 스팬 수준에서 위험한 공개를 감지할 수 있도록 설계되었습니다. 참가자들은 모델의 출력을 어떻게 인식하고 활용했는지에 대한 질문을 받았으며, 이를 통해 모델의 정확성과 유용성을 평가했습니다.

- **Performance Highlights**: 모델은 자가 공개를 감지하는 데 완벽하지 않았지만, 82%의 참여자가 연구 외부에서도 모델을 사용하고 싶다고 응답했습니다. 참가자들은 모델이 오류를 잡고 리스크를 인식하게 도움을 주며, 결정 사항에 대한 자신감을 높일 수 있었다고 언급했습니다. 특정 카테고리 라벨이 추가된 모델의 반응이 긍정적이었으며, 사용자는 추가적인 설명의 필요성을 제기했습니다.



### Large Language Models and Code Security: A Systematic Literature Review (https://arxiv.org/abs/2412.15004)
- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)의 코드 생성과 관련된 보안 위험과 이점에 대해 체계적인 문헌 검토(Systematic Literature Review, SLR)를 제공합니다. 특히, LLMs가 생성한 코드에 의해 도입될 수 있는 다양한 취약점 유형을 분석하고, 코드 취약점을 탐지하고 수정하는 LLMs의 능력, 그리고 이 과정에서의 프롬프트 전략의 영향을 조사합니다. 또한, LLMs에 대한 데이터 중독 공격(data poisoning attack)이 이러한 작업 수행 능력에 미치는 영향을 심도 있게 분석합니다.

- **Technical Details**: 해당 연구는 대규모 언어 모델들이 코드 생성 및 보안 관련 작업 수행 시 발생할 수 있는 취약점 유형과 이 모델들의 탐지 능력에 대해 연구 질문들을 설정했습니다. 구체적으로 RQ1에서는 LLM이 생성하는 코드에서 도입될 수 있는 보안 취약점을 조사하고, RQ2에서는 인간이 작성한 코드와 LLM이 생성한 코드의 취약점 탐지 및 수정 능력을 분석합니다. 또한, RQ3에서는 데이터 세트의 중독이 LLM의 보안 코드 생성 및 취약점 탐지 및 수정 능력에 미치는 영향을 살펴봅니다.

- **Performance Highlights**: 이 연구에서 언급된 통계에 따르면 GitHub Copilot은 현재 코드의 약 46%를 생성하고 있으며, 개발자의 코딩 속도를 최대 55% 향상시킵니다. 그러나 LLMs는 특정 보안 관행에 대한 인식이 부족하며, 잠재적으로 비공식적으로 수집된 데이터 세트에 의존함으로써 보안 취약점을 생성할 위험을 안고 있습니다. 이러한 취약점과 한계에도 불구하고 LLMs는 여전히 개발자들에게 매우 유용한 도구로 자리잡고 있습니다.



### HSEvo: Elevating Automatic Heuristic Design with Diversity-Driven Harmony Search and Genetic Algorithm Using LLMs (https://arxiv.org/abs/2412.14995)
Comments:
          18 pages, 12 figures

- **What's New**: 자동 휴리스틱 디자인(Automatic Heuristic Design, AHD)은 방대한 탐색 공간에서 탐색과 활용의 균형을 맞추는 중요한 요소를 이해하고 분석하는 데 중점을 두고 있습니다. 본 연구는 LLM 기반의 진화 프로그램 검색(LLM-EPS) 모델인 HSEvo를 소개하며, 새로운 다양성 측정 지표를 통해 객체 성과와 다양성 간의 균형을 잡는 새로운 접근 방식을 제공합니다. HSEvo는 비용 효율성을 유지하면서도 높은 다양성을 달성하여 기존 LLM-EPS 방법들과의 차별점을 보입니다.

- **Technical Details**: HSEvo 프레임워크는 조화 검색 알고리즘(harmony search algorithm)을 기반으로 하여 다양한 초깃값 및 교차, 변이를 최적화하는 방식으로 작동합니다. 이 프레임워크는 Shannon-Wiener 다양성 지수와 누적 다양성 지수를 활용하여 인구의 진화적 진행 상황을 평가합니다. 본 연구는 LLM-EPS의 탐색 공간에서의 특성과 성질을 이해하는 데 기초한 이론 및 원칙을 확립하기 위해 노력하고 있습니다.

- **Performance Highlights**: 실험 결과 HSEvo는 높은 다양성 지수와 함께 좋은 객체 성과를 달성하여 탐색과 활용의 균형을 강조합니다. EoH는 이전 방법들보다 높은 다양성을 보여주지만 성과의 일관성이 부족한 반면, ReEvo는 성과는 좋지만 다양성 최적화에 한계를 보였습니다. 이러한 결과들은 LLM-EPS 프레임워크에서 다양성과 성과 균형의 중요성을 강조합니다.



### Movie2Story: A framework for understanding videos and telling stories in the form of novel tex (https://arxiv.org/abs/2412.14965)
- **What's New**: M2S라는 프레임워크는 비디오와 오디오, 문자 인식을 결합하여 소설 수준의 텍스트를 생성하는 새로운 방법론을 소개합니다. 이 모델은 비디오의 긴 형식 텍스트 설명 및 이해, 오디오 감정 분석, 그리고 시각적 캐릭터 인식 모듈을 포함하여 다중 모달 정보를 통합합니다. M2S는 대규모 언어 모델인 GPT-4로 다중 모달 텍스트 생성을 차별화하며, 향후 연구에 대한 잠재력을 지니고 있습니다.

- **Technical Details**: M2S 모델은 여러 모듈로 구성되어 있으며, 비디오 및 오디오의 다중 모달 정보를 결합하여 소설 형식의 상세한 서사를 생성합니다. 커다란 음성 및 감정 기반의 분석을 사용하여, 비디오와 오디오의 내용 모두를학습할 수 있습니다. 또한 이 모델은 고객의 다양성과 요구를 충족하기 위해 매끄러운 스토리라인 및 심리적 묘사를 제공합니다.

- **Performance Highlights**: 실험 결과 M2S는 텍스트의 완전성 및 풍부한 의미 정보를 잘 생성하는 뛰어난 능력을 보여줍니다. 이 모델은 장애인 및 텍스트 서술에 의존하는 사용자에게 영화 내용을 보다 쉽게 이해하고 즐길 수 있는 기회를 제공합니다. 이러한 접근은 교육 및 오락 분야에서도 활용 가능성이 높아, 비디오 컨텐츠에 대한 자세한 설명 제공 및 영화의 소설화 작업에 기여할 수 있을 것입니다.



### Cirbo: A New Tool for Boolean Circuit Analysis and Synthesis (https://arxiv.org/abs/2412.14933)
Comments:
          To appear in AAAI 2025

- **What's New**: 이번 논문에서는 Boolean 회로를 조작하기 위한 오픈 소스 도구인 Cirbo를 소개합니다. Cirbo는 효율적인 알고리즘을 구현하고 있으며, 이를 통해 회로의 만족성, 합성, 최소화와 같은 다양한 회로 작업을 수행할 수 있습니다. 이 도구는 IWLS 2024 프로그래밍 대회에서 우승을 도와주었으며, 2023년 대회에서 Google DeepMind가 차지한 첫 번째 자리를 넘어설 수 있었습니다.

- **Technical Details**: Boolean 회로는 컴퓨터 과학의 여러 분야에서 응용되는 수학적 모델로, 복잡도 이론(Complexity Theory), 컴퓨터 공학(Computer Engineering), 암호학(Cryptography) 등에서 중요합니다. 논문에서 다루는 주요 문제는 회로 분석(circuit analysis)과 회로 합성(circuit synthesis)으로, 이는 Boolean 함수를 작은 회로로 합성하는 것으로 요약될 수 있습니다. 이 과정에서 최소 회로 크기 문제(MCSP)는 주목할 만한 이론적 문제이며, 회로 크기를 줄이는 방식에 대해 다룹니다.

- **Performance Highlights**: Cirbo 도구는 2023년에 비해 평균 12%의 회로 크기 감소를 나타냈고, 일부 개별 회로는 무려 83%까지 감소하였습니다. 2024년 대회에서는 제공된 데이터셋이 이전 대회와 동일하여, 회로 크기 축소의 진전을 추적할 수 있었습니다. Cirbo는 ABC 및 mockturtle과 같은 기존 도구들과의 성능 비교에서 더 나은 결과를 보였으며, 필요에 따라 이러한 도구들과의 조합을 통해 최적의 결과를 도출하기도 했습니다.



### RobustFT: Robust Supervised Fine-tuning for Large Language Models under Noisy Respons (https://arxiv.org/abs/2412.14922)
- **What's New**: 이번 논문에서는 Large Language Models (LLMs)을 위한 안정적인 감독 세부 조정(Noise-robust Supervised Fine-Tuning, RobustFT) 프레임워크를 소개합니다. 이 프레임워크는 데이터의 잡음을 감지하고 재레이블링(relabeling)하는 메커니즘을 도입하여 모델의 다운스트림(task) 수행 능력을 향상시킵니다. 특히, 다수의 전문가 시스템을 활용하여 모델의 성능을 더욱 향상시키는 접근 방식을 제안함으로써, LLM이 발생하는 데이터에서 잡음을 효과적으로 다룰 수 있는 방법을 모색합니다.

- **Technical Details**: RobustFT는 멀티 뷰(multi-view) 잡음 검출 및 제거 전략을 사용합니다. 잡음 감지를 위해 협업 다중 전문가 시스템을 적용하여, 잡음 데이터의 식별을 효과적으로 수행합니다. 또한, 잡음 제거 단계에서 맥락 강화(context-enhanced) 전략을 활용해 신뢰할 수 있는 주석을 생성하며, 응답 엔트로피(response entropy)를 바탕으로 데이터를 선택하여 고품질 샘플만을 사용합니다.

- **Performance Highlights**: RobustFT는 다양한 잡음 수준을 가진 다섯 개 데이터셋에서 실행된 대규모 실험을 통해 탁월한 성능을 입증하였습니다. 실험 결과, RobustFT는 모델의 다운스트림 작업 수행 능력을 향상시키며, 특히 잡음이 많은 환경에서도 우수한 성능을 보여주었습니다. 결론적으로 이 프레임워크는 LLM의 세부 조정에 실질적인 가치를 제공하며, 다양한 도메인에서의 적용 가능성을 증명합니다.



### Dehallucinating Parallel Context Extension for Retrieval-Augmented Generation (https://arxiv.org/abs/2412.14905)
- **What's New**: 본 논문은 Parallel Context Extension (PCE)에 대한 기존 문제를 해결하기 위해 DePaC(Dehallucinating Parallel Context Extension)를 제안합니다. DePaC는 맥락 인식 부정 훈련(context-aware negative training)과 정보 보정 집계(information-calibrated aggregation)를 활용하여 사실 조작(fact fabrication) 및 사실 누락(fact omission)이라는 두 가지 종류의 환각 문제를 완화합니다. 이 접근 방식은 RAG 시나리오에서의 신뢰성을 높이는 데 기여할 것으로 기대됩니다.

- **Technical Details**: DePaC는 두 가지 주요 구성 요소로 이루어져 있습니다. 첫째, NegTrain은 맥락이 질문과 관련이 없는 경우 LLM이 답변을 거부하도록 유도합니다. 둘째, ICA는 각 문서의 정보 증가 정보(information increment)를 기반으로 맥락 창을 우선적으로 선택하여 집계합니다. 이 과정에서 Kullback-Leibler divergence를 활용하여 맥락의 유용한 정보를 보다 효과적으로 판별하게 됩니다.

- **Performance Highlights**: 실험 결과, DePaC는 아홉 개의 RAG 작업에서 환각 문제를 유의미하게 완화시키며 일관되게 더 나은 성능을 보여줍니다. 또한, DePaC는 vanilla 접근 방식보다 계산 복잡도가 낮아 문서 수에 따라 선형적으로 증가하는 추론 시간을 보여줍니다. 실험을 통해 정보 보정 집계와 맥락 인식 부정 훈련이 DePaC의 성능에 모두 필수적이라는 것을 입증하였습니다.



### AI-Powered Intracranial Hemorrhage Detection: A Co-Scale Convolutional Attention Model with Uncertainty-Based Fuzzy Integral Operator and Feature Screening (https://arxiv.org/abs/2412.14869)
- **What's New**: 이 연구는 두 개의 이진 분류 문제로 ICH의 발생 여부 및 SDH의 유형을 감지하는 새로운 접근 방식을 제안합니다. 특히, CCA(classified co-scale convolutional attention) 분류기 아키텍처에 두 개의 레이어를 추가하여 ICH 감지를 위한 혁신적인 방법을 도입하였습니다. 이 방법은 CT 스캔 이미지에서 추출된 특징을 결합하고 높은 변동성을 포착하는 50개의 구성 요소를 선택하여 설명 가능한 AI 모델 개발에 기여합니다.

- **Technical Details**: 첫 번째 레이어에서는 CT 스캔의 다양한 슬라이스에서 특징을 추출한 후, 부스트랩 포레스트 알고리즘을 사용하여 최종 예측에 대한 각 특징의 기여도를 명확히 평가합니다. 두 번째 레이어에서는 새로운 불확실성 기반 퍼지 적분 연산자를 도입하여 서로 다른 CT 스캔 슬라이스의 정보를 융합하고, 이는 검사 정확도를 크게 향상시킵니다. 이를 통해 CT 스캔 슬라이스 간의 의존성을 고려하여 감지 정확도를 높입니다.

- **Performance Highlights**: 이 연구는 이란 테헤란의 두 개 의료 센터에서 수집된 대규모 CT 스캔 이미지를 사용하여 ICH와 SDH를 탐지하는 데 초점을 맞추었습니다. 새로운 접근 방식은 기존 방법보다 더 강력하고 신뢰할 수 있는 검출 모델을 개발하는 데 기여하며, 이미지 전처리 및 특징 융합 과정에서 불필요한 특징을 제거하여 자원 소모를 줄이는 데 효과적입니다. 이 연구는 AI 모델의 해석 가능성을 높이는 데 중요한 진전을 보여줍니다.



### A Survey of RWKV (https://arxiv.org/abs/2412.14847)
Comments:
          18 pages

- **What's New**: Receptance Weighted Key Value (RWKV) 모델은 Transformer 아키텍처의 새로운 대안으로 부각되고 있습니다. 이 모델은 순환 신경망(RNN)과 주의(attention) 기반 시스템의 장점을 결합하여 긴 시퀀스를 효율적으로 처리할 수 있도록 설계되었습니다. RWKV는 기존 Transformer 모델의 계산 비효율성을 줄이는 데 기여하며, 자연어 처리(NLP) 및 컴퓨터 비전 분야에서 차별화된 성과를 보여주고 있습니다.

- **Technical Details**: RWKV는 고유한 키-값(key-value) 접근 방식을 활용하여 последовательные 의존성을 관리하며 계산 오버헤드를 크게 줄입니다. 이 모델은 긴 시퀀스의 처리가 필요한 작업에서도 강력한 맥락 이해를 유지하면서도 최소한의 메모리 요구량으로 긴 범위 의존성을 캡처할 수 있습니다. 또한, RWKV는 기존 Transformer 모델과의 비교를 통해 그 효율성을 강조하며 다양한 구현 및 적용 사례를 제시합니다.

- **Performance Highlights**: RWKV는 자연어 생성(NLG), 자연어 이해(NLU) 및 컴퓨터 비전 등 다양한 분야에서 강력한 성능을 보여주고 있습니다. 특히, RWKV는 텍스트 생성, 기계 번역 및 감정 분석 등의 NLP 작업에서 그 가능성을 입증했으며, 시간이 많이 소요되는 시퀀스 예측 작업에서도 탁월한 결과를 나타내고 있습니다. 이를 통해 RWKV는 머신러닝 커뮤니티에서 주목받고 있으며, 향후 연구 방향과 발전 가능성에 대한 논의가 활발하게 이루어질 것으로 기대됩니다.



### Head and Neck Tumor Segmentation of MRI from Pre- and Mid-radiotherapy with Pre-training, Data Augmentation and Dual Flow UN (https://arxiv.org/abs/2412.14846)
- **What's New**: 이번 연구는 두 가지 방사선 치료 전과 중 이미지를 세분화하는 여러 전략의 효과를 조사했습니다. 특히 모델의 성능을 개선하기 위해 MixUp 데이터 증강 기법과 사전 학습된 가중치를 활용했습니다. 또한, 미드-RT 이미지를 위한 새로운 네트워크 아키텍처를 제안하였으며, 이 모델은 사전-RT 이미지와 레이블의 정보를 통합하여 세분화 성능을 높였습니다.

- **Technical Details**: 연구에서 사용한 두 가지 주요 네트워크 구조는 basic segmentation network와 Dual Flow UNet(DFUNet)입니다. Basic segmentation network는 인코더-디코더 아키텍처를 기반으로 한 첫 번째 네트워크로, 단일 채널의 pre-RT 이미지를 입력으로 받습니다. DFUNet은 두 개의 인코더를 포함하고 있으며, 미드-RT 이미지와 사전 등록된 pre-RT 이미지와 그 마스크를 동시에 처리하여 정보를 효율적으로 융합하는 데 중점을 둡니다.

- **Performance Highlights**: 최종 테스트에서 pre-RT 세분화는 82.38%, mid-RT 세분화는 72.53%의 성능을 기록했습니다. 이 연구 결과는 Dice 유사도 계수(DSC)를 기준으로 하며, 여러 전략을 통해 세분화 성능을 향상시켰음을 보여줍니다. 특히, 각 fold에서 최고 성능의 모델을 선택하여 앙상블 평균을 생성함으로써 더욱 신뢰성 있는 결과를 도출했습니다.



### Mapping and Influencing the Political Ideology of Large Language Models using Synthetic Personas (https://arxiv.org/abs/2412.14843)
Comments:
          4 pages, 2 figures, 2 tables

- **What's New**: 이번 연구에서는 PersonaHub라는 합성 인물 설명 집합을 활용하여 정치적 편향이 큰 언어 모델(LLMs)의 정치적 성향을 평가했습니다. 특히, 개인화된 프롬프트가 LLM의 정치적 방향성에 미치는 영향을 분석함으로써 기존 연구에서 다루지 않았던 측면을 탐구했습니다. 연구 결과, LLM이 오른쪽 권위주의와 왼쪽 자유주의 간의 상극 성향으로 조정될 수 있음을 보였습니다.

- **Technical Details**: 연구에서는 1억 개 이상의 합성 인물 설명을 포함하는 PersonaHub를 사용하여 LLM의 정치적 방향성을 탐구했습니다. 총 12.4백만 개의 응답을 수집하여 각 모델의 정치적 입장을 평가하기 위해 Political Compass Test (PCT)를 활용했습니다. 4개의 오픈 소스 언어 모델(Mistral, Llama, Qwen, Zephyr)을 선택하고, 두 가지 단계로 실험을 수행하여 인물 설명이 LLM의 정치적 성향에 미치는 영향을 측정했습니다.

- **Performance Highlights**: 결과적으로, 대부분의 인물들은 왼쪽 자유주의 지역에 집중되었으나, 모든 모델은 명시적 이념 프롬프트에 반응하여 정치적 편향을 바꿨습니다. 특히 Llama 모델은 오른쪽 권위주의로의 이동이 가장 두드러졌고, Zephyr는 전반적으로 더 수평적 이동을 보였습니다. 이 연구는 LLM의 정치적 표현 방식에 대한 새로운 통찰을 제공하며, 개인화의 가능성과 한계를 강조합니다.



### Helping LLMs Improve Code Generation Using Feedback from Testing and Static Analysis (https://arxiv.org/abs/2412.14841)
- **What's New**: 이 연구는 Large Language Models (LLMs)에 의해 생성된 코드의 품질을 평가하고, 개선을 유도하는 프레임워크를 제안합니다. LLMs가 생성한 C코드를 분석하기 위해 테스트 및 정적 분석 기법을 활용하며, 이전 연구에서 주목된 코드의 안전성과 정확성 문제를 해결하고자 합니다. 이를 통해 LLM 기반 코드 생성 도구의 신뢰성을 확보할 수 있는 초기 단계로 나아가고 있습니다.

- **Technical Details**: 본 연구에서는 LLMs에게 프로그래밍 작업을 해결하기 위해 C 코드를 생성하도록 요청하고, 생성된 코드의 정확성을 ground-truth 테스트를 통해 평가합니다. 또한, Meta에서 개발한 정적 분석 도구인 Infer를 사용하여 코드의 안전성 및 잠재적 취약점을 분석합니다. 세 단계로 구성된 프레임워크는 코드 생성, 자기 평가, 수리 과정을 포함하며, 이에 따라 LLMs의 다양한 성능을 측정합니다.

- **Performance Highlights**: 연구 결과, LLM이 생성한 코드의 46%에서 65%가 정확성 문제를 보였고, 87%에서 96%는 취약점이 없는 것으로 나타났습니다. LLM들은 오히려 생성한 코드의 문제를 인식하고 수정하는 데 어려움을 겪으며, 특정 조건 하에 개선된 수리 능력을 보였지만 여전히 많은 코드에 오류가 존재합니다. 이 연구는 LLMs의 코드 생성 및 수정 능력 향상을 위한 유망한 경로를 제시하고 있습니다.



### Progressive Multimodal Reasoning via Active Retrieva (https://arxiv.org/abs/2412.14835)
Comments:
          Working in progress

- **What's New**: 이번 연구에서는 다중 단계를 고려한 다중 모드(multimodal) 추론 과제에서 MLLM의 성능을 향상시키기 위한 새로운 프레임워크인 AR-MCTS를 제안합니다. 이 프레임워크는 Active Retrieval(AR)과 Monte Carlo Tree Search(MCTS)를 결합하여 복잡한 추론 문제를 해결하는 데 필요한 핵심 인사이트를 동적으로 검색할 수 있도록 설계되었습니다. 특히, 이 연구는 기존의 빔 탐색(beam search) 샘플링 방법을 대체하는 혁신적인 접근 방식을 도입하여 각 추론 단계에서 다양한 문제 해결 인사이트를 제공함으로써 신뢰성을 높이고자 합니다.

- **Technical Details**: AR-MCTS 프레임워크는 통합된 검색 모듈을 개발하여 하이브리드 모드 검색 데이터베이스로부터 복잡한 추론을 지원하기 위한 핵심 인사이트를 검색합니다. MCTS 알고리즘을 활용하여 단계별로 주어진 문제의 적절한 해답을 유도하는 과정 보상을 정의하고, 각 단계에서 이전 단계를 바탕으로 샘플링을 최적화합니다. 이러한 접근 방식은 추론의 신뢰성과 다양성을 향상시키며, 자동화된 다중 모드 추론 검증 과정을 가능하게 합니다.

- **Performance Highlights**: 실험 결과에 따르면, AR-MCTS는 세 가지 복잡한 다중 모드 추론 기준을 통해 다양한 모델에서 효과성을 입증했습니다. AR-MCTS는 샘플링의 다양성과 검증의 정확성을 최적화하여 신뢰성 있는 다중 모드 추론을 위한 promising한 솔루션을 제공합니다. 이 연구 결과는 다중 모드 추론의 고도화를 위한 가능성을 열어 주며, MLLM의 성능 향상에 기여할 것으로 기대됩니다.



### MARIA: a Multimodal Transformer Model for Incomplete Healthcare Data (https://arxiv.org/abs/2412.14810)
- **What's New**: MARIA (Multimodal Attention Resilient to Incomplete datA) 모델은 헬스케어 분야에서 다중 모드 데이터의 결합을 통해 진단 및 예측 모델 개발에 도움을 줄 수 있도록 설계된 혁신적인 딥러닝 모델입니다. 기존 접근 방식과 달리 MARIA는 데이터 임퓨테이션(imputation)에 의존하지 않고, 오히려 마스크된 셀프 어텐션(masked self-attention) 메커니즘을 사용하여 결측 데이터를 처리합니다. 이로 인해 결측 데이터를 보다 효과적으로 다룰 수 있으며, 특히 헬스케어 응용 분야에서의 강건성 및 편향 감소를 가능하게 합니다.

- **Technical Details**: MARIA는 중간 융합(intermediate fusion) 전략을 통해 모달리티별 인코더(modality-specific encoder)와 공유된 어텐션 기반 인코더(shared attention-based encoder)를 결합하여 결측 데이터를 처리합니다. 다른 전통적인 방법들이 데이터를 임퓨터(impute)하여 결측값을 채우려는 것과는 달리, MARIA는 이용 가능한 특성만을 중점적으로 활용하여 가상의 데이터를 생성하지 않습니다. 이러한 접근 방식은 강건성과 정확성을 향상시키고, 일반적으로 임퓨테이션 기법에서 발생하는 편향을 줄이는 데 기여합니다.

- **Performance Highlights**: MARIA는 8개의 진단 및 예측 작업에서 10개의 최신 머신러닝 및 딥러닝 모델과 비교 평가되었으며, 다양한 결측 데이터 조건에서 기존 방법들보다 뛰어난 성능을 보였습니다. 결과적으로, MARIA는 결측 데이터에 대한 회복력(resilience) 및 향상된 성능을 통해 헬스케어 응용 분야에서 중요한 잠재력을 지니고 있음을 입증했습니다. 이는 진단 정확도와 치료 결과를 향상시키는 데 기여할 것으로 기대됩니다.



### Stack Trace Deduplication: Faster, More Accurately, and in More Realistic Scenarios (https://arxiv.org/abs/2412.14802)
Comments:
          Published at SANER'25. 11 pages, 2 figures

- **What's New**: 이번 연구에서는 대규모 소프트웨어 시스템의 버그 보고 문제를 해결하기 위해 새로운 모델과 산업 기반 데이터셋을 소개합니다. 기존의 방법들이 실제 작업 흐름과는 단절된 채로 평가되어 온 반면, 본 연구는 다양한 평가 기준을 통해 현실에 맞는 평가를 수행하였습니다. 또한, JetBrains에서 개발한 IntelliJ 기반 제품의 오류 보고 데이터셋인 SlowOps를 공개하여 실용적인 연구에 기여하고자 합니다.

- **Technical Details**: 제안된 모델은 두 가지 부분으로 나뉘어 있습니다. 첫 번째는 바이트 쌍 인코딩(Byte Pair Encoding, BPE)과 빠른 근사 최근접 이웃 탐색(approximate nearest neighbor search)을 사용하는 임베딩 모델입니다. 두 번째는 선택된 스택 트레이스를 재정렬하는 리랭커(reranker)로, 반복되는 스택 프레임을 고려하여 더 정확한 재정렬을 지원합니다. 이 두 가지 방식은 스택 트레이스 간의 상호작용을 더 잘 이해하고, 유사도 점수를 향상시키는 데 초점을 맞춥니다.

- **Performance Highlights**: 모델의 성능 평가 결과, 제안된 방법은 모든 데이터셋에서 기존의 방법보다 높은 정확도를 보이며, Speed 측면에서도 우수한 성능을 기록하였습니다. 특히 SlowOps 데이터셋에서 새로운 카테고리 생성 능력에서도 좋은 결과를 보여 줍니다. 이를 통해 우리의 방법은 정확도와 효율성을 균형 있게 확보한 것을 입증하였습니다.



### Agent-Temporal Credit Assignment for Optimal Policy Preservation in Sparse Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2412.14779)
Comments:
          12 pages, 1 figure

- **What's New**: 이 논문에서는 Temporal-Agent Reward Redistribution (TAR²)라는 새로운 접근 방식을 제시하여, 에이전트-시간 신용 할당 문제를 해결하고 희소한 보상을 시간적으로 및 에이전트간에 재분배하는 방법을 논의합니다. TAR²는 희소한 글로벌 보상을 시간 단계별 보상으로 분해하고, 각 에이전트의 기여도를 계산하여 이를 처리합니다. 이 방식은 에이전트-특정 보상을 학습함으로써 단일 에이전트 강화 학습 알고리즘의 사용이 가능하게 합니다.

- **Technical Details**: TAR²는 두 개의 주목 모듈(Temporal Attention Module 및 Agent Attention Module)을 활용하여 에이전트의 행동 및 상태의 중요성을 평가합니다. 이 방법은 잠재 기반 보상 형성(Potential-based Reward Shaping)과 동일하며, 원래 보상 함수 아래에서도 최적 정책이 보존된다는 이론적 증거를 제공합니다. TAR² 방법은 단일 에이전트 강화 학습 알고리즘의 단순성과 확장성을 활용하여 복잡한 환경에서 효율적인 신뢰 할당 문제를 해결합니다.

- **Performance Highlights**: 본 연구는 TAR²의 샘플 효율성이 기존의 최첨단 기법과 비교하여 SMACLite 환경에서 입증되었습니다. 실험 결과, TAR²는 단일 에이전트 강화 학습 알고리즘과 병합하여 기존의 다중 에이전트 강화 학습 방법들과 동등하거나 나은 성능을 보여 주었습니다. 또한, 본 연구는 다양한 환경 설정에서 TAR²의 효과를 검증하여 학계와 산업에 대한 기여를 강조합니다.



### Energy and polarization based on-line interference mitigation in radio interferometry (https://arxiv.org/abs/2412.14775)
- **What's New**: 이번 논문에서는 기존의 RFI(전파 주파수 간섭) 완화 방법에 대해 중요한 발전을 제안하고 있습니다. 저자들은 새로운 RFI 완화 알고리즘을 제시하여, 실시간으로 RFI를 감지하고 완화할 수 있는 방법을 소개합니다. 이 방법은 주파수 대역과 에너지 기반 스펙트럼 커토시스와 선형 편광 정렬을 결합하여 RFI를 탐지합니다.

- **Technical Details**: 본 논문에서는 단일 정밀도(single precision) 및 반 정밀도(half precision) 부동소수점 연산을 활용하여 RFI 완화 알고리즘의 계산 효율성을 높이는 방법을 제안합니다. 특히, LOFAR 상관기(correlator)와 같은 가속기 컴퓨팅 장치에서 이 기술이 효율적으로 구현될 수 있도록 설계되었습니다. 혼합 정밀도 연산을 위한 최적화는 강화 학습(reinforcement learning)을 통해 수행됩니다.

- **Performance Highlights**: 제안된 RFI 완화 알고리즘은 기존의 방법보다 낮은 에너지를 갖는 RFI를 효과적으로 탐지할 수 있습니다. همچنین, 고주파 해상도(data streams at high time-frequency resolution)에서 사용할 수 있는 이 알고리즘은 특히 전환(transient)이나 단기간(short duration) RFI 탐지에 매우 적합합니다. 실시간 결과를 통해 제안된 방법의 효과를 검증하였으며, RFI로 인한 데이터 품질 저하 문제를 해결할 수 있는 가능성을 보여줍니다.



### ALKAFI-LLAMA3: Fine-Tuning LLMs for Precise Legal Understanding in Palestin (https://arxiv.org/abs/2412.14771)
- **What's New**: 이 논문은 저자들이 팔레스타인 법률 분야에 적합한 대규모 언어 모델(LGMs)을 조정하는 방법을 탐구하고, 그 과정에서의 도전 과제를 다룹니다. 특히, 복잡한 정치적 환경과 제한된 AI 자원이 있는 저자원 국가에서 AI의 효율적인 실행을 위한 해결책을 제안합니다. 연구팀은 Llama-3.2-1B-Instruct의 양자화된 버전을 기반으로 한 세밀하게 조정된 모델을 제시하며, 이 모델은 팔레스타인 법률 텍스트에서 유도된 합성 데이터 세트로 학습됩니다.

- **Technical Details**: 연구에서 사용된 기본 법률은 팔레스타인 정부의 공식 소스를 통해 수집된 1,277개의 텍스트 파일로 구성되어 있습니다. 이 텍스트 파일은 JSON 형식으로 구조화되어 빠른 문서 접근을 가능하게 하며, 법률 조항과 해당 법률 텍스트를 포함합니다. 질문-답변 쌍은 ChatGPT API와 Gemma API를 활용해 생성되었으며, 이는 법률 언어 작업을 위한 세부 조정을 지원하는 robust한 데이터 세트를 형성하는 데 기여합니다.

- **Performance Highlights**: 실험 결과, 연구팀은 다양한 질문 유형에 대해 유망한 성능을 보여주었으며, 예/아니오 질문, 서사적 설명, 복잡한 법적 구별에 대한 질문을 포함했습니다. 그러나 계산 기반 문의 처리와 정형 리스트 포맷팅 등에서 개선 여지가 있음을 강조합니다. 이 연구는 자원이 제한된 환경에서 AI 기반 법률 지원 도구의 배치를 위한 경로를 제시합니다.



### CodeRepoQA: A Large-scale Benchmark for Software Engineering Question Answering (https://arxiv.org/abs/2412.14764)
- **What's New**: 이 논문에서는 소프트웨어 엔지니어링 분야에서 리포지토리 수준 질문-답변(Question-Answering, QA) 기능을 평가하기 위해 설계된 대규모 벤치마크인 CodeRepoQA를 소개합니다. 이 벤치마크는 5개 프로그래밍 언어를 아우르며, 다양한 시나리오를 포함하여 언어 모델의 포괄적인 평가를 가능하게 합니다. CodeRepoQA는 GitHub의 30개 유명 리포지토리에서 데이터를 수집하고 필터링하여 총 585,687개의 QA 엔트리를 포함하고 있습니다.

- **Technical Details**: 이 연구는 GitHub의 이슈를 기반으로 다단계 대화형 QA를 구축하여, 개발자들이 실제 소프트웨어 개발 과정에서 경험하는 복잡한 문제와 상호작용을 포착합니다. 또한, 실험에서는 다양한 대형 언어 모델(LLMs)들을 평가하였으며, 정확도 평가를 위해 BLEU, ROUGE-L, ROUGE-1, Edit Similarity와 같은 지표를 사용하였습니다. 벤치마크는 5개 프로그래밍 언어로 구성되어 있으며, 각 언어의 대화 회전 수와 엔트리 수는 상세히 기록되어 있습니다.

- **Performance Highlights**: 실험 결과 LLM은 여전히 소프트웨어 엔지니어링 분야의 QA 기능에서 한계를 보였습니다. 추가로, 중간 길이의 컨텍스트가 LLM의 성능에 더 유리하다는 점이 확인되었습니다. 이 연구는 CodeRepoQA가 기존 벤치마크와 비교하여 실제 소프트웨어 개발 및 유지 관리의 복잡성을 더욱 잘 반영하는 것을 보여줍니다.



### Advances in Artificial Intelligence forDiabetes Prediction: Insights from a Systematic Literature Review (https://arxiv.org/abs/2412.14736)
- **What's New**: 이 체계적 리뷰는 당뇨병 예측을 위한 머신러닝(ML)의 사용을 탐구합니다. 다양한 데이터셋(드래타셋), 알고리즘(algorithms), 학습 방법(training methods), 및 평가 메트릭스(evaluation metrics)에 초점을 맞추고 있습니다. 특히 싱가포르 국가 당뇨병 망막병 검사 프로그램과 같은 데이터셋을 강조합니다.

- **Technical Details**: 리뷰에서는 CNN, SVM, 로지스틱 회귀(Logistic Regression), XGBoost와 같은 다양한 ML 알고리즘의 성능을 분석합니다. 각 알고리즘이 당뇨병 결과를 예측하는 데 어떻게 기여하는지를 검토하고, 효과적인 예측 모델을 설계하는 데 필요한 다학제적 협력(interdisciplinary collaboration)과 윤리적 고려사항(ethical considerations)의 중요성을 강조합니다.

- **Performance Highlights**: 이 연구는 ML 기반의 당뇨병 예측 모델이 의료 분야에서 가질 수 있는 가능성을 보여줍니다. 특히 다양한 데이터셋에서 알고리즘의성능 평가를 통해, 기술이 당뇨병 예측에 어떻게 활용될 수 있는지를 입증합니다. 이는 미래 연구와 개발의 중요한 기초가 될 것입니다.



### Beyond the Hype: A Comprehensive Review of Current Trends in Generative AI Research, Teaching Practices, and Tools (https://arxiv.org/abs/2412.14732)
Comments:
          39 pages, 10 figures, 16 tables. To be published in the Proceedings of the 2024 Working Group Reports on Innovation and Technology in Computer Science Education (ITiCSE-WGR 2024)

- **What's New**: 이 보고서는 Generative AI (GenAI)의 급속한 발전과 컴퓨팅 교육에서의 문헌 확장을 다루고 있습니다. 초기 GenAI 도구에 대한 반응은 혼잡했지만, 이 도구는 대부분의 프로그래밍 과제를 해결하고 커리큘럼에 혼란을 초래하고 있습니다. 새로운 연구가 2024년에 나타나면서 GenAI가 컴퓨터 수업에서 어떻게 활용되는지에 대한 데이터가 수집되고 있으며, 이 도구들은 개인화된 피드백을 제공합니다.

- **Technical Details**: 연구팀은 세 가지 범주에 속하는 개인들과 인터뷰를 진행했습니다: 도구 제작자, GenAI를 연구하는 교육자, GenAI를 사용하는 교육자. 연구 질문은 초기 교육 목표와 연결되어 있으며, 시간에 따른 GenAI 수업 도입과 지원, 도구의 영역 및 사용 정책을 탐구하고자 합니다. 또한, GenAI 사용의 동기와 학생들의 기대하는 역량에 대한 영향을 조사하고 있습니다.

- **Performance Highlights**: 이 연구는 교육자와 산업 전문가들을 대상으로 한 문헌 검토와 설문조사를 포함하여, GenAI 사용에 대한 인식을 확대하고 있습니다. GenAI 도구는 교육 현장에서 교수 지원 역할을 하며, 수업에서의 실제 학생 성과에 미치는 영향도 분석되고 있습니다. 이는 향후 컴퓨팅 교육의 방향성을 제시하는 중요한 기초 자료가 될 것입니다.



### How to Synthesize Text Data without Model Collapse? (https://arxiv.org/abs/2412.14689)
- **What's New**: 이 논문은 합성 데이터(synthetic data)가 언어 모델의 학습 과정에 미치는 영향을 다룹니다. 특히, 합성 데이터를 사용한 반복적인 훈련이 모델 성능을 저하시킬 수 있는 현상인 모델 붕괴(model collapse)를 강조합니다. 연구팀은 합성 데이터와 인간이 생산한 데이터를 일정 비율로 혼합하여 훈련했을 때 성능 저하가 나타난다는 것을 발견했습니다.

- **Technical Details**: 연구에서는 인간이 만든 데이터와 합성 데이터의 혼합 비율을 α로 정의하고, 이를 기반으로 훈련 데이터를 설정합니다. 실험 결과 합성 데이터를 사용할 때 모델의 성능이 하락하고, 특히 인간이 생산한 데이터를 포함했을 때 모델 붕괴가 어느 정도 완화됨을 확인했습니다. 또한, 통계 분석을 통해 합성 데이터의 분포가 제한적이며 특성 집중이 발생한다는 결론을 도출했습니다.

- **Performance Highlights**: 이 논문에서 제안하는 token-level editing 방법은 모델 붕괴를 방지하는 이론적 근거를 제공하며, 여러 통계 분석 실험을 통해 그 효과성을 입증하고 있습니다. 대규모 실험을 통해 초기 모델 훈련 및 지속적인 훈련 과정에서 token-level editing이 데이터 품질을 개선하고 모델 성능을 높인다는 것을 확인했습니다.



### Each Fake News is Fake in its Own Way: An Attribution Multi-Granularity Benchmark for Multimodal Fake News Detection (https://arxiv.org/abs/2412.14686)
- **What's New**: 이번 연구에서는 멀티모달(fake news) 가짜 뉴스 탐지의 중요한 도전 과제를 다루고 있습니다. 기존의 멀티모달 데이터셋은 단순히 진짜 또는 가짜로만 레이블링 되어, 다양한 유형의 가짜 뉴스를 반영하지 못했습니다. 이에 따라, 우리는 고유한 가짜 패턴을 드러내는 AMG라는 새로운 데이터셋을 구축하고, 멀티그랜ולר 단서를 통한 가짜 뉴스 탐지 및 할당 모델(MGCA)을 제안합니다.

- **Technical Details**: AMG 데이터셋은 다양한 소셜 플랫폼에서 수집된 가짜 뉴스를 포함합니다. 데이터 수집과 처리 과정에서, 사실 확인 웹사이트를 활용하여 가짜 뉴스의 세부 유형을 주석 처리하고, 시각적 및 텍스트 콘텐츠의 멀티 뷰 특성을 추출하는 MGCA 모델을 개발하였습니다. 이 모델은 또한 다중 그랜율 단서의 일관성을 모델링하여 진위 탐지와 할당을 지원합니다.

- **Performance Highlights**: 실험 결과는 AMG 데이터셋이 상당히 도전적임을 보여주며, 새로운 연구 방향을 제시합니다. 우리는 본 연구에서 멀티그랜율 가짜 뉴스 할당의 개념을 제시하고, AMG를 통해 가짜 뉴스의 원인에 대한 세분화된 할당을 수행하였습니다. 전체적으로 우리의 제안 모델인 MGCA는 멀티모달 가짜 뉴스 탐지 및 할당에서 강력한 성능 향상을 보였습니다.



### A Light-Weight Framework for Open-Set Object Detection with Decoupled Feature Alignment in Joint Spac (https://arxiv.org/abs/2412.14680)
- **What's New**: 이 논문은 Open-set object detection (OSOD) 분야에서 로봇 시스템의 실시간 작업을 지원하는 경량화된 프레임워크인 Decoupled OSOD (DOSOD)를 제안합니다. DOSOD는 YOLO-World 파이프라인을 기반으로 하여 vision-language model (VLM)과 탐지기를 통합합니다. 이 프레임워크는 고급 기능 상호작용을 피하면서 계산 효율성을 개선하여, 알려지지 않은 객체 탐지의 필요성을 해결합니다.

- **Technical Details**: DOSOD는 Multilayer Perceptron (MLP) 어댑터를 사용하여 VLM으로부터 추출한 텍스트 임베딩을 변환하고, 이를 통해 클래스 비정보 제안의 영역 표현을 학습합니다. 이러한 접근 방식은 서로 다른 모드의 기능들이 결합되는 것을 최소화하여 테스트 단계에서 전통적인 폐쇄형 탐지기와 유사한 방식으로 작동하게 만듭니다. 최종적으로, DOSOD는 조인트 스페이스에서 다양한 모드의 특성들을 직접적으로 정렬하여 컴퓨팅 비용과 저장 요구 사항을 감소시킵니다.

- **Performance Highlights**: DOSOD는 YOLO-World와 비교하여 실시간 성능을 크게 향상시키면서도 유사한 정확도를 유지합니다. DOSOD-S 모델은 LVIS minival 데이터셋에서 고정 AP(Fixed AP) 26.7%를 달성하였으며, 이는 YOLO-World-v1-S의 26.2% 및 YOLO-World-v2-S의 22.7%보다 높은 수치입니다. 또한 DOSOD-S는 YOLO-World-v1-S보다 57.1% 높은 FPS, YOLO-World-v2-S보다 29.6% 높은 FPS를 기록하며 경량화된 배포가 가능합니다.



### FiVL: A Framework for Improved Vision-Language Alignmen (https://arxiv.org/abs/2412.14672)
- **What's New**: 이번 논문은 시각 및 언어 입력을 통합하는 데 중요한 진전을 보여주는 대형 비전 언어 모델(LVLMs)의 새로운 접근 방식을 소개합니다. LVLMs가 시각 정보와 언어적 내용을 동등하게 사용하기 어렵다는 문제를 다루기 위해, 새로운 데이터셋을 설계하여 LVLM의 시각적 기반 구축을 개선하는 것을 목표로 하고 있습니다. 이 방법은 이미지 데이터가 단순한 맥락이 아닌 실질적인 증거로 작용하도록 하여, 더욱 정확한 답변 생성을 가능하게 합니다.

- **Technical Details**: 제안된 FiVL 방법은 LVLM 교육을 위한 데이터셋을 구성하는 새로운 기법을 포함합니다. 이 데이터셋은 이미지 내용을 근거로 활용할 수 있게 하는 훈련 및 평가 도구로 작용하여, 모델이 시각 정보를 얼마나 잘 활용하는지를 측정합니다. 이를 통해 기존 문제점인 시각적 접근을 강화하여, 질문-답변 작업에서의 성능 향상을 도모할 수 있습니다.

- **Performance Highlights**: 제안된 데이터셋을 사용하여 새로운 훈련 작업을 제시하며, 이를 통해 기존 기준을 초월하는 성과를 보여줍니다. 또한 이 모델은 설명 가능성을 향상시키기 위한 검증 방법을 통해 사용자에게 더 나은 이해를 제공합니다. 실험 결과, 모델의 종합적인 성능이 향상되었음을 입증하며, 해당 코드는 온라인에서 이용 가능합니다.



### Analysis and Visualization of Linguistic Structures in Large Language Models: Neural Representations of Verb-Particle Constructions in BER (https://arxiv.org/abs/2412.14670)
- **What's New**: 이번 연구에서는 transformer 기반의 대형 언어 모델 (large language models, LLMs)에서 동사-부사 조합의 내부 표현을 조사합니다. 특히 BERT 아키텍처를 활용하여, 다양한 동사-부사 구조의 어휘적 (lexical) 및 구문적 (syntactic) 뉘앙스를 어떻게 포착하는지를 분석합니다. 이는 기존의 신경망이 언어적 요소를 처리할 때 가정된 균일성에 도전하는 중요한 발견입니다.

- **Technical Details**: 연구 방법론으로는 British National Corpus에서 데이터셋을 준비하고, BERT 모델을 광범위하게 학습시키며 출력 분석을 다차원 스케일링 (multi-dimensional scaling, MDS) 및 일반화된 판별 값 (generalized discrimination value, GDV) 계산을 통해 진행했습니다. BERT의 중간 계층이 구문 구조를 가장 효과적으로 포착하며, 다양한 동사 범주에 따라 표현 정확도가 크게 달라지는 것으로 나타났습니다.

- **Performance Highlights**: 연구 결과, BERT 모델이 언어 처리를 어떻게 이해하고 수행하는지에 대한 통찰을 제공합니다. 이는 현재의 신경적 접근 방식이 언어 분석에서 지닌 잠재력과 한계를 드러내며, 깊이 있는 언어 모델 성능을 최적화하기 위한 추가 연구의 필요성을 제기합니다. 이 연구는 컴퓨터 언어학 분야에서도 발전을 이끌어내며, 언어적 정밀도를 증대시키기 위한 신경망 아키텍처 최적화에 대한 새로운 방향을 제시합니다.



### LoLaFL: Low-Latency Federated Learning via Forward-only Propagation (https://arxiv.org/abs/2412.14668)
Comments:
          14 pages, 9 figures

- **What's New**: 본 논문에서는 저지연 연합 학습(LoLaFL) 프레임워크를 제안합니다. LoLaFL은 전파 방식(forward-only propagation)을 이용하여 레이어별 전송 및 집계를 가능하게 하며, 통신 라운드를 현저히 줄여 저지연을 달성합니다. 또한, 두 가지 비선형 집계(nonlinear aggregation) 기법을 도입하여 더욱 효율적인 모델 파라미터 집계를 실현합니다.

- **Technical Details**: LoLaFL은 전통적인 연합 학습(FL)과 달리 백프로파게이션(backpropagation)을 사용하지 않고, 각 통신 라운드에서 최신 레이어만 전송합니다. 이 방법은 각 레이어의 특징에 기반하여 파라미터를 직접 계산하고 전송하므로 모델 훈련의 속도를 높입니다. 또한, Be 일종의 조화 평균(harmonic mean) 기반 집계 접근법을 제안하여 더 많은 유연성과 강력한 성능을 제공합니다.

- **Performance Highlights**: 이론적 분석과 실험을 통해, 제안된 LoLaFL은 전통적인 FL과 비교할 때 통신 지연(latency)을 각각 91% 및 98% 감소시키면서도 유사한 정확도를 유지하는 것으로 나타났습니다. 이는 고차원 데이터 통신의 복잡성과 자원 제약 문제를 효과적으로 해결하여, 6G 모바일 네트워크의 요구사항에 부합하는 실용적인 솔루션을 제공합니다.



### IOHunter: Graph Foundation Model to Uncover Online Information Operations (https://arxiv.org/abs/2412.14663)
Comments:
          9 pages

- **What's New**: 이 논문에서는 정보 조작(Information Operations, IO) 캠페인을 주도하는 사용자, 즉 	extit{IO drivers}를 식별하기 위한 새로운 방법론인 IOHunter를 제안합니다. IOHunter는 언어 모델(Language Models)과 그래프 신경망(Graph Neural Networks, GNNs)의 장점을 결합하여 다양한 IO 캠페인을 효과적으로 탐지할 수 있습니다. 이 연구는 소셜 미디어 플랫폼에서 IO 탐지를 위한 그래프 기반 모델(Graph Foundation Models, GFM)을 개발하는 방향으로 나아가며, 최신 방법론과 성능 비교에서 뛰어난 결과를 보였습니다.

- **Technical Details**: IOHunter는 GNN의 메시지 전달(paradigm)과 네트워크 구조 및 텍스트 콘텐츠에서 파생된 멀티 모달 정보(multi-modal information)를 결합하여 설계되었습니다. 기존 방법들이 그래프 구조 또는 텍스트 콘텐츠 중 하나에만 초점을 맞춘 것과 달리, IOHunter는 다양한 그래프 데이터를 활용하여 새로운 태스크나 데이터셋에 신속하게 적응할 수 있는 GFM을 지원합니다. IOHunter는 UAE, 쿠바, 러시아, 베네수엘라, 이란, 중국의 각각 다른 지리적 맥락에서 기원한 IO에 대해 6개의 Twitter 데이터셋에서 평가되어 +20%의 Macro-F1 향상을 달성했습니다.

- **Performance Highlights**: 실험 결과 IOHunter는 데이터가 제한된 상황에서도 강력한 성능을 보이며, 다른 IO와의 교차 탐지(cross-IO detection) 과제에서도 효과를 입증했습니다. 기존 최첨단 성능을 명확히 초과하며, IO 감지 분야에 새로운 기준을 제시했습니다. 이로써 IOHunter는 정보 조작을 탐지하기 위한 새로운 지평을 여는 중요한 도구로 자리잡게 되었습니다.



### Unveiling Uncertainty: A Deep Dive into Calibration and Performance of Multimodal Large Language Models (https://arxiv.org/abs/2412.14660)
Comments:
          Accepted to COLING 2025

- **What's New**: 이 논문에서는 다중 모달 대형 언어 모델(MLLMs)의 불확실성 보정(calibration) 문제를 중점적으로 다루고 있습니다. 모델의 불확실성을 정량화하고 보정하는 과정이 중요하며, 이를 통해 의료 및 자율 주행과 같은 안전-critical 분야에서의 신뢰성을 개선할 수 있습니다. MLLMs는 이미지와 텍스트의 통합 처리를 통해 높은 성능을 보이지만, 여전히 고질적인 성능 불균형 문제가 존재합니다.

- **Technical Details**: MLLMs의 성공적인 활용을 위해, 저자들은 IDK 데이터셋을 구축하여 모델의 불확실성 자기 평가(self-assessment)를 분석했습니다. 두 가지 모달 정보 간의 불확실성 차이를 평가하고, 텍스트에 비해 이미지 정보의 불확실성이 낮은 경향을 관찰했습니다. 추가로, 고급 보정 기법인 temperature scaling과 iterative prompt optimization을 사용하여 MLLMs의 신뢰도와 예측 성능을 향상시킬 수 있는 방법을 제안했습니다.

- **Performance Highlights**: 연구 결과, MLLMs는 고백하기보다는 대답하는 경향이 있으며 이는 적절한 프롬프트 조정으로 개선될 수 있음을 보여주었습니다. 다양한 설정에서 MLLMs의 보정 성능 차이를 분석하였고, 특히 Fine-tuning 전후의 큰 차이를 발견하지 못했습니다. 저자들은 MLLMs의 교육 및 테스트에 있어 더 많은 개선이 필요하다고 결론지으며, 다양한 응용 분야에 대한 책임감 있는 활용 방안을 고안했습니다.



### Adaptive Prompt Tuning: Vision Guided Prompt Tuning with Cross-Attention for Fine-Grained Few-Shot Learning (https://arxiv.org/abs/2412.14640)
- **What's New**: 최근 컴퓨터 비전 분야는 기계 학습 및 딥 러닝 기술의 발전으로 놀라운 성장을 보여주고 있습니다. 이 논문에서는 Contrastive Language-Image Pre-Training (CLIP) 모델을 개선하는 새로운 방법인 Adaptive Prompt Tuning (APT)을 제안하여 한정된 데이터로 미세 분류 작업을 돕습니다. 이 접근법은 이미지와 텍스트 간의 동적 상호작용을 통해 텍스트 프롬프트를 조정하여 보다 효과적인 분류를 가능하게 합니다.

- **Technical Details**: APT는 이미지 입력에 따라 프롬프트를 조정하는 교차 주의 메커니즘을 활용하여 CLIP의 텍스트와 이미지 요소의 정렬 능력을 향상시킵니다. Monte-Carlo Dropout (MCD)을 통합하여 예측에 대한 신뢰도를 향상시키고 예측 불확실성을 파악합니다. 이는 시스템의 성능과 신뢰성을 높이는 데 중요한 역할을 하며, 특히 의료 분야와 같은 중요한 응용 프로그램에서 중요합니다.

- **Performance Highlights**: 여러 데이터셋(CUBirds, Oxford Flowers, FGVC Aircraft)에서 실시된 평가에서 APT는 기존의 정적 프롬프트 조정 방식에 비해 유의미한 성능 향상을 보였습니다. 또한, 모델의 예측 신뢰도를 개선하는 데 성공함으로써, 신뢰할 수 있는 예측을 제공하고 필요한 경우 추가 검증이 필요하다는 인사이트를 제공합니다. 이 방법은 몇 샷 미세 분류의 최신 기술을 크게 향상시키는 견고한 솔루션으로 작용합니다.



### A Shapley Value Estimation Speedup for Efficient Explainable Quantum AI (https://arxiv.org/abs/2412.14639)
Comments:
          26 pages, 4 figures, 4 tables, 45 citations

- **What's New**: 본 연구는 양자 AI 알고리즘을 위한 효율적인 사후 설명(post-hoc explanations)을 개발하는 데 중점을 두고 있습니다. 고전적 맥락에서 협력 게임 이론의 개념인 Shapley 값이 AI의 의사결정 과정에서 중요한 요소를 식별하기 위해 자연스럽게 사후 설명에 적용될 수 있습니다. 이 논문은 Shapley 값을 양자 설정으로 변환하고 양자 효과를 사용하여 계산을 가속화할 수 있는 방법을 제안합니다.

- **Technical Details**: 저자들은 양자 회로의 입력 큐비트에 대한 Shapley 값을 계산하기 위한 효율적인 알고리즘을 자세히 설명합니다. 이 알고리즘은 몬테카를로 방법과 비교할 때 quadratic speedup을 제공하며, 다양한 상황에서 polylogarithmic factor를 달성할 수 있습니다. 성능 검증을 위해 특정 투표 게임에서 접근 방식을 실증적으로 입증하고 일반 협력 게임에 대한 rigorous proofs를 제공합니다.

- **Performance Highlights**: 제안된 방법은 classical Monte Carlo 접근 방식보다 quadratically 우수한 성능을 보입니다. 연구 결과는 AI의 Explainability를 개선할 수 있는 잠재력을 가지고 있으며, 이는 규제 환경에서 요구되는 사후 설명 생성의 필요성을 충족할 수 있는 기회를 제공합니다. 논문에서는 Shapley 값의 정확한 근사치를 얻을 수 있는 가능성을 보여주며, 급변하는 AI 환경에서의 설명 가능성 증진에 이바지할 것으로 기대됩니다.



### Progressive Fine-to-Coarse Reconstruction for Accurate Low-Bit Post-Training Quantization in Vision Transformers (https://arxiv.org/abs/2412.14633)
- **What's New**: 이 논문에서는 Vision Transformers (ViTs)의 Post-Training Quantization (PTQ)에서 발생하는 성능 저하 문제를 해결하기 위해 Progressive Fine-to-Coarse Reconstruction (PFCR) 방법을 제안합니다. PFCR은 여러 개의 재구성 단위를 점진적으로 최적화하여 낮은 비트 양자화 설정에서의 성능을 크게 향상시킵니다. 또한, 재구성 과정을 두 단계로 나누어 구성하는 Progressive Optimization Strategy (POS)라는 새로운 전략을 도입하여 최적화의 어려움을 완화합니다.

- **Technical Details**: PFCR 방법에서는 Multi-Head Self-Attention (MHSA)와 Multi-Layer Perceptron (MLP) 모듈을 가장 세밀한 재구성 단위로 정의합니다. 이 두 모듈의 최적화가 완료된 후에, 그들을 결합하여 블록 형태의 더 거친 재구성을 수행합니다. 이 과정은 세밀한 재구성 단위를 점진적으로 거친 단위로 조합하고 재구성하는 방식으로 반복적으로 이루어집니다.

- **Performance Highlights**: ImageNet 데이터셋에서의 실험 결과에 따르면, 제안된 PFCR 방법은 3비트 양자화된 ViT-B에서 75.61%의 Top-1 정확도를 달성하여 기존의 최첨단 기술보다 우수한 성능을 보였습니다. 또한 COCO 데이터셋에서의 양자화 결과는 제안된 방법이 객체 탐지 및 인스턴스 분할과 같은 다른 컴퓨터 비전 작업에도 효과적으로 일반화됨을 보여줍니다.



### Learning to Generate Research Idea with Dynamic Contro (https://arxiv.org/abs/2412.14626)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 발전이 과학적 발견을 가속화할 수 있는 잠재력을 보여주고, 연구 아이디어 생성 과정의 자동화를 강조합니다. 특히, 기존의 프로그래밍 기반 접근 방식을 넘어서는 새로운 방법론을 제안하며, LLM을 세밀하게 조정하여 보다 뛰어난 아이디어 제안자로 만드는 기법을 소개합니다.

- **Technical Details**: 제안된 프레임워크는 두 단계로 구성되어 있습니다. 첫 번째 단계인 Supervised Fine-Tuning (SFT)에서는 연구 논문과 후속 아이디어 쌍에서 기초 패턴을 학습합니다. 두 번째 단계에서는 Reinforcement Learning (RL)을 이용하여 다양한 보상 모델링을 통해 생성된 아이디어를 평가하고 최적화합니다. 이 방법에는 차원 컨트롤러가 포함되어 있어 생성 과정에서의 동적 조정이 가능합니다.

- **Performance Highlights**: 이 프레임워크는 혁신성, 실행 가능성, 효과성 간의 균형적인 접근 방식을 제공하며, 고품질 결과를 달성합니다. 또한, 문장 수준의 디코더를 통해 맥락 인식 강조가 이루어져, 생성된 아이디어의 품질을 높이는 데 기여합니다.



### Pitfalls of topology-aware image segmentation (https://arxiv.org/abs/2412.14619)
Comments:
          Code is available at this https URL

- **What's New**: 이 논문은 의료 이미징에서의 topology-aware 방법의 실제 적용을 저해하는 평가 기준의 결함을 지적합니다. 특히, ill-suited connectivity choices, ground truth annotations의 topological artifacts, 그리고 부적절한 evaluation metrics의 문제를 강조합니다. 이는 segmentation 방법의 성능 평가 및 순위 매김에 중대한 영향을 미치는 것으로 분석되었습니다.

- **Technical Details**: 저자들은 topological correctness를 보존하기 위한 다양한 접근 방식을 검토하고, 픽셀 기반 성능 측정 외에도 여러 topological metrics와 그 활용의 중요성을 논의합니다. 특히, persistence homology(PH)와 discrete Morse theory(DMT)를 활용한 방법들이 탐구되며, 이들이 어떻게 topological accuracy를 측정하는지에 대한 통찰을 제공합니다. 또한, 여러 분류 작업에 있어 각 topological metric의 적용 방식과 의의가 설명됩니다.

- **Performance Highlights**: 저자들은 의료 이미지에서 neuron 및 vessel segmentation의 품질이 topology-aware 메트릭을 기반으로 개선될 수 있음을 강조합니다. 그러나 benchmarking의 불완전함으로 인해 방법 비교가 부정확하게 이루어지고 있으며, 이는 실질적인 의료 응용에 있어 장애물이 되고 있습니다. 논문에서 제안된 해결책들은 이러한 평가 기준의 개선과 더 나은 상호 비교를 위한 중요한 기초가 될 것입니다.



### How good is GPT at writing political speeches for the White House? (https://arxiv.org/abs/2412.14617)
- **What's New**: 이 연구는 대규모 언어 모델(LLMs)인 GPT의 작성 스타일을 분석하여 최근 미국 대통령들의 연설과 비교합니다. 연구는 레이건부터 바이든까지의 국정연설(State of the Union) 내용을 GPT-3.5 및 GPT-4.0이 생성한 연설과 대조하여 GPT의 독특한 특징을 드러냅니다. 이를 통해 LLM이 인간 작성자와 비교될 수 있는 지점을 조명합니다.

- **Technical Details**: 연구는 GPT가 사용하는 'we'라는 렘마(lemma)의 과잉 사용과 평균적으로 긴 문장을 사용하면서 메시지가 짧은 경향을 보임을 지적합니다. 또한 GPT는 정치적 용어(예: president, Congress), 상징적 용어(예: freedom), 추상적 용어에 더 자주 의존하여 낙관적인 톤을 선택하는 경향이 있습니다. 이러한 경향은 사용하는 저자의 스타일을 부여하더라도 여전히 차별화된 특성을 유지합니다.

- **Performance Highlights**: GPT-3.5와 GPT-4.0은 각각의 독특한 특성을 가지고 있으나, 두 모델 모두 실제 대통령 연설과는 전반적으로 다른 모습을 보입니다. 이 연구는 LLM이 생성하는 텍스트가 인간 작성자의 연설을 얼마나 잘 모방하는지에 대한 통찰을 제공합니다. 결국, GPT는 미국 대통령의 메시지와 비교할 때 여전히 불일치를 나타냅니다.



### HarmonicEval: Multi-modal, Multi-task, Multi-criteria Automatic Evaluation Using a Vision Language Mod (https://arxiv.org/abs/2412.14613)
- **What's New**: 이 논문은 Vision-Language Models (VLMs)에서 생성된 텍스트의 자동 평가를 위한 새로운 메트릭인 HarmonicEval을 제안합니다. 기존의 측정 지표들이 전반적인 품질에만 집중하여 평가의 측면에서 부족함을 드러내는 가운데, HarmonicEval은 기준 기반 점수를 집계하여 전반적인 점수를 생성하는 방식을 채택합니다.

- **Technical Details**: HarmonicEval의 평가는 세 단계로 구성됩니다: 1) 각 기준에 대한 점수를 생성하기 위해 VLM을 프로세스하는 단계, 2) VLM이 생성한 출력 토큰 확률을 기반으로 점수를 안정화하는 단계, 3) 하모닉 가중치를 사용해 최종 점수를 계산하는 단계입니다. 이를 통해 VLM의 텍스트 품질을 다양한 비전-언어 작업에서 평가할 수 있습니다.

- **Performance Highlights**: 실험 결과, HarmonicEval은 기존의 평가 지표들보다 인간의 판단과 더욱 높은 상관관계를 보여줍니다. 또한, 제안된 MMHE 데이터셋을 통해 HarmonicEval이 특정 평가 기준을 간과하는 기존 메트릭들의 단점을 보완하면서도 종합적인 평가를 수행할 수 있음을 입증합니다.



### Towards Scalable and Deep Graph Neural Networks via Noise Masking (https://arxiv.org/abs/2412.14602)
- **What's New**: 최근 그래프 신경망(Graph Neural Networks, GNNs)은 많은 그래프 마이닝 작업에서 성과를 거두었습니다. 그러나 대규모 그래프에 적용하는 것은 반복적인 특성 전파(feature propagation)와 비선형 변환(non-linear transformation)으로 인해 계산 및 저장 비용이 높아 도전 과제가 되고 있습니다. 이 논문은 기존의 모델 단순화 접근 방식에 무작위 도보(noise masking) 기법인 RMask를 제안하여 과도한 음영(over-smoothing) 문제를 해결하고, 깊은 GNN을 탐색하면서 효율성을 유지할 수 있게 합니다.

- **Technical Details**: 이 연구는 특히 P(Propagation) 및 C(Combination) 작업으로 구성된 모델 단순화 GNN에 중점을 두고 있습니다. 특히, P 작업에서 생성된 노이즈(noise)가 과도한 음영 현상에 기여함을 연구하였으며, RMask 모듈을 통해 이러한 노이즈 제거 메커니즘을 도입했습니다. 기존의 모델 단순화 기법과 통합하여, RMask는 보다 깊은 그래프 신경망의 구현을 가능하게 합니다.

- **Performance Highlights**: RMask가 적용된 모델 단순화 GNN은 세 개의 광범위한 데이터셋(Cora, Citeseer, Pubmed)과 세 개의 대규모 데이터셋(ogbn-arxiv, ogbn-products, ogbn-papers100M)에서 우수한 성능을 보였습니다. 실험 결과, RMask로 강화된 모델은 원래 모델보다 뛰어난 성능을 발휘하며, 정확도와 효율성 간의 상호작용을 잘 조율할 수 있음을 보여줍니다.



### Spike2Former: Efficient Spiking Transformer for High-performance Image Segmentation (https://arxiv.org/abs/2412.14587)
Comments:
          This work has been accepted on Association for the Advancement of Artificial Intelligence 2025

- **What's New**: 본 논문에서는 스파이킹 신경망(Spiking Neural Networks, SNNs)을 복잡한 아키텍처에서 이미지 분할(imag segmentation) 작업에 적용하기 위해 새로운 Spike2Former 아키텍처를 제안합니다. 이러한 아키텍처는 정보 손실을 최소화하고 훈련 안정성을 높이기 위한 기능을 포함하고 있으며, 이는 SNNs의 성능 저하 문제를 해결합니다. 추가적으로, 새롭게 설계된 NI-LIF 스파이킹 뉴런인 정규화된 정수 스파이킹 뉴런(normalized integer spiking neurons)을 도입하여 SNN의 변화를 해결합니다.

- **Technical Details**: Spike2Former 아키텍처는 Mask2Former 아키텍처를 기반으로 하며, 정보 손실을 방지하기 위해 구조를 개선하였습니다. 특히, 변형 가능한 주의 변환기 인코더 블록(Deformable Attention Transformer Encoder Block)과 마스크 임베딩(mask embedding) 계층에서 가장 큰 정보 손실이 발생하였으며, 이를 해결하기 위해 전환 블록(convolution blocks)을 도입하고 보조 정보 브랜치를 추가하여 표현력을 강화하였습니다. 또한, NI-LIF 스파이킹 뉴런은 훈련 중 정수 활성화(normalization)를 정규화하여 훈련 안정성을 높입니다.

- **Performance Highlights**: 제안된 Spike2Former는 ADE20K, Pascal VOC2012 및 CityScapes 데이터셋에서 기존 SNN 모델에 비해 뛰어난 성능을 나타냅니다. ADE20K에서는 +12.7% mIoU(Mean Intersection over Union)와 5.0배 에너지 효율, VOC2012에서는 +14.3% mIoU와 5.2배 효율, CityScapes에서는 +9.1% mIoU와 6.6배 효율을 기록합니다. 이러한 성과는 복잡한 시나리오에서 SNN의 잠재력을 보여줍니다.



### GSRender: Deduplicated Occupancy Prediction via Weakly Supervised 3D Gaussian Splatting (https://arxiv.org/abs/2412.14579)
- **What's New**: 이번 연구에서는 3D Occupancy Perception(점유 인식)을 위한 새로운 방법인 GSRender를 소개합니다. GSRender는 3D Gaussian Splatting을 활용하여 더 정밀하고 효율적인 점유 예측을 가능하게 하며, 카메라 광선에 따른 샘플링 프로세스를 단순화합니다. 또한, Ray Compensation(RC) 모듈을 도입하여 인접 프레임의 특성을 보완함으로써 중복 예측을 줄입니다. 이와 함께 동적 물체의 영향력을 최소화하기 위해 차별화된 손실 함수를 설계하였습니다.

- **Technical Details**: GSRender는 3D Gaussian Splatting을 적용하여 고화질의 포토리얼리스틱(photorealistic) 장면을 실시간으로 렌더링합니다. 이 과정에서 각 Gaussian의 평균과 공분산을 포함하여 여러 속성을 고려하며, 2D 이미지 평면으로의 투영도 수행합니다. 연구팀은 기존의 NeRF 기반 방법들이 겪는 샘플링 트레이드오프 문제를 해결하고, 단지 하나의 추가 프레임을 통해 특징을 통합하여 성능을 향상시켰습니다. 이를 통해 2D 약한 감독을 통해도 효과적인 3D 점유 예측을 실현했습니다.

- **Performance Highlights**: GSRender는 RayIoU 지표에서 SOTA(상태 최우수) 성과를 달성하며, 2D 약한 감독 방법들 사이에서 눈에 띄는 성과를 보였습니다. 실험 결과, 기존의 3D 감독 방법들과 비교하여 성능 격차를 크게 줄였으며, 실제 데이터셋(Nuscenes)에서도 뛰어난 결과를 기록했습니다. 이 연구는 특히 자율주행 기술의 발전에 기여할 것으로 기대되고 있습니다.



### SCKD: Semi-Supervised Cross-Modality Knowledge Distillation for 4D Radar Object Detection (https://arxiv.org/abs/2412.14571)
Comments:
          Accepted by AAAI 2025

- **What's New**: 이 논문에서는 4D 밀리미터파 레이더를 기반으로 한 3D 객체 탐지를 위한 새로운 반지도형 교차 모달리티 지식 증류(SCKD) 방법을 제안합니다. 이 방법은 반지도형(distillation) 학습을 통해 레이저와 레이더의 융합된 teacher 네트워크로부터 특징을 학습할 수 있게 합니다. 기존의 다중 모달리티 융합 접근법의 한계를 극복하며, 레이더 전용의 학생 네트워크 성능을 크게 향상시킬 수 있습니다.

- **Technical Details**: 제안된 SCKD 방법은 Lidar와 레이더의 융합을 담당하는 적응형 융합 모듈을 teacher 네트워크에 추가하여 성능을 높입니다. 이후, Lidar에서 레이더로의 특징 증류(LRFD) 및 융합에서 레이더로의 특징 증류(FRFD) 두 가지 방법을 설계하였습니다. 또한, 반지도형 출력 증류(SSOD)를 통해 teacher와 student 간의 지식 전이를 보다 효과적으로 수행합니다.

- **Performance Highlights**: 실험 결과, SCKD 방법은 VoD 데이터셋에서 기본 성능보다 mAP(Mean Average Precision)를 10.38% 향상시키고, ZJUODset에서는 추가 비정형 데이터를 사용할 경우 5.12%의 성능 향상을 기록했습니다. 이로써 제안된 방법은 최신 방법들보다 뛰어난 성능을 보이며, 대량의 라벨 없는 데이터 활용 가능성을 열어줍니다.



### Characterising Simulation-Based Program Equilibria (https://arxiv.org/abs/2412.14570)
- **What's New**: 이번 연구에서는 Tennenholtz의 프로그램 균형 (program equilibrium)을 일반화하여 Oesterheld의 $	ext{ε}$Grounded$	ext{π}$Bot보다 더 넓은 범위의 균형을 달성하는 방법을 제안합니다. 우리는 공유 랜덤 제너레이터에 접근할 수 있는 환경에서의 대칭 게임을 분석하며, 기존의 방법론이 세 플레이어 이상의 게임에서는 적용되지 않는 한계를 극복하였습니다. 이 연구는 AI 에이전트의 상호작용을 모델링하는 데 중요한 기여를 합니다.

- **Technical Details**: 연구에서는 시뮬레이션 기반 프로그램들 (simulation-based programs)을 다루며, 이는 상대 프로그램을 실행하여 결정을 내리는 방식입니다. 이러한 접근법은 상대적으로 강건하며, 두 프로그램이 동일한 행동을 수행할 때 이를 동일하게 처리하여 효율성을 높입니다. 공유 랜덤성 (shared randomness)을 가진 환경과 가지지 않은 환경에서의 균형을 특성화하고 분석하였습니다.

- **Performance Highlights**: 우리는 Oesterheld의 $	ext{ε}$Grounded$	ext{π}$Bot보다 더 넓은 범위의 균형을 달성하였으며, 이는 특히 두 플레이어 게임에서 더 다양한 전략적 가능성을 생성합니다. 그러나 공유 랜덤성 없이 시뮬레이션 기반 프로그램 균형의 한계를 탐구한 결과, Tennenholtz의 민속 정리 (folk theorem)는 도달할 수 없음을 보여주었습니다. 이 연구는 AI 시스템과의 투명한 상호작용을 위해 중요한 통찰력을 제공합니다.



### Global Spatio-Temporal Fusion-based Traffic Prediction Algorithm with Anomaly Awar (https://arxiv.org/abs/2412.14569)
- **What's New**: 본 연구에서는 이상 감지 신경망을 기반으로 한 전역 시공간 융합 교통 예측 알고리즘을 제안합니다. 이 알고리즘은 외부 사건의 시공간적 영향을 모델링 할 수 있는 이상 요소 impact 모듈을 구성하고, 또한 Transformer 아키텍처를 활용한 다중 스케일 시공간 특성 융합 모듈을 포함하여 정확한 교통 흐름 예측을 가능하게 합니다.

- **Technical Details**: 이 알고리즘은 세 가지 주요 부분으로 나누어져 있습니다: 데이터 임베딩 모듈, 이상 요인 영향 모듈 및 다중 스케일 시공간 특성 융합 모듈입니다. 이를 통해 전세계적인 장기 정보 및 이상 요인의 개인적 특성을 효과적으로 포착할 수 있습니다. 또한 이 알고리즘은 Temporal Self-Attention(TSA) 및 Spatial Self-Attention(SSA) 방법을 사용하여 시공간적 특성을 분석합니다.

- **Performance Highlights**: PEMS04와 PEMS08 두 가지 벤치마크 데이터셋을 기반으로 한 실험 결과, 본 연구의 접근법은 기존의 최첨단 기법보다 뛰어난 성능을 보이며 교통 예측 정확도 향상의 가능성을 입증합니다. 이 연구는 외부 요인의 영향을 잘 반영하여 교통 예측의 신뢰성을 높일 수 있는 방법을 제시합니다.



### AIArena: A Blockchain-Based Decentralized AI Training Platform (https://arxiv.org/abs/2412.14566)
- **What's New**: 이 논문에서는 특정 대기업들이 AI(Artificial Intelligence) 개발과 사용을 지배하는 중심화된 구조가 AI 모델 내의 편향(bias)을 증대시키고 있다는 문제를 지적합니다. 이에 대한 해결책으로, AIArena라는 블록체인 기반의 탈중앙화된 AI 훈련 플랫폼을 제안합니다. 이 플랫폼은 공정한 보상을 통해 모든 참여자들이 AI 모델 개발에 기여할 수 있는 환경을 조성합니다.

- **Technical Details**: AIArena는 블록체인의 장점을 활용하여 참여자들이 공동으로 협력할 수 있는 AI 훈련 환경을 제공합니다. 여기에는 훈련 노드(training nodes), 검증자(validators), 위임자(delegators)와 같은 여러 참여자 역할이 포함되어 있으며, 각 참여자는 자신들의 기여도에 따라 보상을 받게 됩니다. 이러한 구조는 스마트 계약(smart contracts)을 통해 자동화되어 효율성과 공정성을 보장합니다.

- **Performance Highlights**: AIArena를 Base 블록체인 Sepolia 테스트넷에서 구현한 결과, 대규모 참여자들이 협력하여 18,656개의 AI 모델을 성공적으로 생성했습니다. AIArena에서 기여한 모델들은 기존의 기준 모델 및 최신 기술 수준(state-of-the-art) 모델들을 뛰어넘는 성능을 보여주며, 코드 공동 제작 작업에서 가장 큰 Move 코드 데이터셋을 성공적으로 관리했습니다.



### Summary of Point Transformer with Federated Learning for Predicting Breast Cancer HER2 Status from Hematoxylin and Eosin-Stained Whole Slide Images (https://arxiv.org/abs/2412.14545)
- **What's New**: 이 연구는 HE(hematoxylin and eosin)로 염색된 전체 슬라이드 이미지(WSI)에서 HER2(인간 상피세포 성장인자 수용체 2) 상태를 예측하기 위한 분산 학습(federated learning) 기반 접근법을 소개합니다. 이 방법은 비용 절감과 치료 결정 속도를 높이는 데 기여합니다.

- **Technical Details**: 멀티사이트 데이터셋에서 레이블 불균형(label imbalance)과 특성 표현(feature representation) 문제를 해결하기 위해 포인트 변환기(point transformer)가 제안되었습니다. 이 모델은 동적 레이블 분포(dynamic label distribution), 보조 분류기(auxiliary classifier), 그리고 최외각 코사인 샘플링(farthest cosine sampling)을 통합하여 구현됩니다.

- **Performance Highlights**: 네 개의 사이트(2687 WSI)에서 상태-of-the-art 성능을 입증하며, 두 개의 보지 못한 사이트(229 WSI)에 대해서도 강력한 일반화 능력을 보여줍니다.



### Overview of AI and Communication for 6G Network: Fundamentals, Challenges, and Future Research Opportunities (https://arxiv.org/abs/2412.14538)
- **What's New**: 최근 인공지능(Artificial Intelligence, AI)과 통신의 통합이 6세대 통신망(6G) 구조의 혁신적인 변화로 대두되고 있습니다. 본 논문은 AI와 6G 네트워크의 기본 원칙, 내재된 도전과제 및 향후 연구 기회에 대해 종합적으로 개관합니다. 특히, AI 모델의 발전이 현대 통신 기술 형성에 미친 영향과 AI 통합의 세 가지 개발 단계를 명확히 설명하고 있습니다.

- **Technical Details**: 6G 네트워크에서는 AI를 통해 네트워크 성능을 향상시키고, 효율성을 최적화하며, 사용자 서비스 경험을 증대시키는 'AI for Network', AI 작업을 지원하기 위한 네트워크의 역할을 강조하는 'Network for AI', 그리고 AI 기능이 서비스로 제공되는 'AI as a Service'의 세 가지 개발 단계가 제시됩니다. 이외에도 데이터 프라이버시, 알고리즘 투명성, 에너지 효율성 등의 챌린지를 고려하여 AI의 통합적 접근이 필요함을 강조하고 있습니다.

- **Performance Highlights**: 6G는 초저지연, 고속 데이터 전송, 높은 신뢰성과 유비쿼터스 연결성을 제공할 것으로 예상되며, AI 기술의 도입이 이러한 발전을 혁신적으로 이끌 것으로 기대됩니다. 네트워크는 AI 알고리즘을 활용하여 리소스 할당을 최적화하고, 신속한 응답 처리를 통해 사용자 경험을 개선하며, 에너지 소비를 줄이는 방향으로 나아갈 것입니다. 이 연구는 6G와 AI의 융합이 사회 전반에 걸쳐 혁신적인 변화를 가져올 수 있는 가능성을 제시합니다.



### CAE-T: A Channelwise AutoEncoder with Transformer for EEG Abnormality Detection (https://arxiv.org/abs/2412.14522)
Comments:
          The manuscript consists of 10 pages, including 5 figures. The experimental results are based on evaluations using the TUH Abnormal EEG Corpus

- **What's New**: 본 논문은 EEG 비정상 감지를 위한 CAE-T라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 채널별 CNN 기반 오토인코더와 단일 헤드 트랜스포머 분류기를 결합하여 효율적인 분석을 가능하게 합니다. 이를 통해 EEG 신호의 차원 감소와 생물학적으로 의미 있는 특징을 유지하면서 낮은 계산 비용으로 비정상을 탐지하는 데 성공하였습니다.

- **Technical Details**: CAE-T 모델은 두 단계로 구성됩니다: 첫 번째는 일반 EEG 신호를 압축하여 잠재 표현(latent representation)을 만드는 과정이며, 두 번째는 이 표현을 사용하여 최종적으로 병리 예측을 수행하는 단계입니다. 채널별 CNN을 사용한 오토인코더 디자인은 각 EEG 채널을 독립적으로 처리하여, 서로 간섭을 방지하고 자연스러운 특성을 유지합니다. 트랜스포머는 자가 관심(self-attention) 메커니즘을 활용하여 긴 거리 의존성을 캡처하는 데 도움을 줍니다.

- **Performance Highlights**: 제안된 CAE-T 모델은 TUH Abnormal EEG Corpus 데이터셋에서 85.0%의 정확도, 76.2%의 민감도, 91.2%의 특이도를 달성했습니다. 이 성능은 EEGNet, Deep4Conv 및 FusionCNN을 포함한 기존 모델들을 초월하는 결과입니다. CAE-T는 단 202M FLOPs와 2.9M 파라미터로 작동하여 계산 효율에서도 우수한 성능을 보여주며, 신경과학 연구와 임상 실습에서의 적용 가능성도 넓힙니다.



### PA-RAG: RAG Alignment via Multi-Perspective Preference Optimization (https://arxiv.org/abs/2412.14510)
- **What's New**: 본 논문에서는 Retrieval-Augmented Generation (RAG)의 생성기를 최적화하는 새로운 방법인 다중 관점 선호 정렬(Multiple Perspective Preference Alignment, PA-RAG)을 제안합니다. 기존 RAG 시스템의 한계인 응답의 유용성, 견고성, 인용 품질을 개선하기 위해 다양한 품질의 응답을 샘플링하여 높은 품질의 instruction fine-tuning 데이터와 선호 데이터셋을 구축했습니다. PA-RAG는 일반적인 LLMs(large language models)에 비해 더욱 효과적인 RAG 응답 생성을 가능하게 합니다.

- **Technical Details**: PA-RAG는 두 가지 훈련 단계로 구성되어 있습니다. 첫 번째 단계는 기본 능력 훈련으로, generator가 문서를 활용하고 인용하는 기본적인 능력을 습득하도록 합니다. 두 번째 단계인 다중 관점 선호 최적화 단계에서는 Direct Preference Optimization (DPO)을 사용하여 다양한 관점에서 선호 정보를 학습합니다. 이 과정에서 응답의 유용성, 견고성, 인용 품질을 각각 향상시키는 세 가지 하위 단계로 구성됩니다.

- **Performance Highlights**: PA-RAG를 적용한 실험에서는 평균 13.97%의 정확성 개선과 49.77%의 인용 조합률, 39.58%의 인용 정확성이 향상되었습니다. 이는 기존의 SFT(supervised fine-tuning)나 추가 단계만을 사용하는 방법에 비해 상당한 성과로, PA-RAG의 효과성을 입증합니다. 연구진은 특히 선호 최적화의 각 관점에서 영향을 미치는 효과를 보여주며, PA-RAG의 모든 훈련 데이터를 공개할 예정입니다.



### Treatment Effects Estimation on Networked Observational Data using Disentangled Variational Graph Autoencoder (https://arxiv.org/abs/2412.14497)
Comments:
          21 pages, 6 figures

- **What's New**: 본 논문에서는 기존의 관측 데이터를 이용한 개별 치료 효과(ITE) 추정 방법에서 발생하는 한계를 극복하기 위해 새로운 모델, TNDVGA(Disentangled Variational Graph Autoencoder)를 제안한다. 이 모델은 치료 효과 추정을 위해 숨겨진 요인(latent factors)을 구별하여 학습하며, Hilbert-Schmidt Independence Criterion을 통해 각 요인 간 독립성을 보장한다. 결과적으로 이 방법은 최신 기술에 비해 더 높은 성능을 달성했다고 밝혔다.

- **Technical Details**: TNDVGA는 네트워크 관측 데이터에서 치료 효과를 추정하기 위해 설계된 새로운 프레임워크로, 관측된 변수와 보조 네트워크 정보로부터 숨겨진 요인을 추론하는 데 효과적이다. 논문은 네 가지 상호 배타적인 요인으로 이 요인들을 분리하여 ITE 추정을 개선하는 방법론을 상세히 설명한다. 특히, HSIC 독립성 규제를 모델의 다른 구성요소와 함께 최적화해 독립적인 분리 표현(disentangled representations)을 학습할 수 있도록 한다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, TNDVGA 모델은 두 개의 반합성 데이터셋과 하나의 합성 데이터셋에서 뛰어난 성능을 보인다. 특히, 기존 방법과 비교했을 때 ITE 추정의 정확성을 높여 주며, 다양한 네트워크 관측 데이터에 적용 가능하다는 점에서 실용성이 뛰어난 것으로 평가된다. 이러한 관찰 결과는 향후 연구에 대한 방향을 제시해준다.



### Stochastic first-order methods with multi-extrapolated momentum for highly smooth unconstrained optimization (https://arxiv.org/abs/2412.14488)
- **What's New**: 이 논문에서는 고차 원활함(high-order smoothness)을 가진 목적 함수의 비제약(stochastic) 최적화 문제를 다루고 있습니다. 특히, 다중 외삽(extrapolated) 모멘텀을 사용하는 확률적 1차 방법(stochastic first-order method, SFOM)을 제안하여 각 반복(iteration)에서 여러 외삽을 수행한 후 얻은 모멘텀 단계에 따라 최적화를 가속화합니다. 본 연구의 방법은 목적 함수의 고차 원활함을 활용하여 샘플 복잡도(sample complexity)의 개선을 성취하면서, 평균 원활함(avg smoothness)을 가정하지 않은 SFOM의 첫 번째 사례로 평가됩니다.

- **Technical Details**: 제안된 방법은 목적 함수 $f$의 기울기와 $p$차 도함수가 Lipschitz 연속적이라는 가정 아래 샘플 복잡도 $	ilde{ig O}(	ext{ε}^{-(3p+1)/p})$를 달성합니다. 여기서 $p 
eq 2$인 경우에도 적용 가능하며, 알고리즘은 기울기에 대한 확률적 추정자(stochastic estimator)인 $G(·;ξ)$를 사용하여 진행됩니다. 노이즈(noise)와 같은 높은 차원의 불확실성을 다루기 위해 확률적 최적화 이론을 적용합니다.

- **Performance Highlights**: 다양한 수치 실험 결과는 본 방법의 실제 성능이 이론적 발견을 잘 지원함을 보여줍니다. 실험을 통해 제안된 SFOM이 고차 원활함을 기반으로 한 가속화 효과를 나타내며, 기존의 최첨단 기술 대비 샘플 복잡도를 현저히 개선합니다. 이러한 결과는 특히 복잡한 최적화 문제를 다루는 데 중요하며, 실제 응용 프로그램에서의 유용성을 나타냅니다.



### HashAttention: Semantic Sparsity for Faster Inferenc (https://arxiv.org/abs/2412.14468)
- **What's New**: 본 논문에서는 HashAttention이라는 새로운 접근 방식을 제안합니다. 이 방법은 중요한 토큰을 추천 문제로 변환하여 효율적으로 식별합니다. Hamming 공간에서 쿼리와 키-값 쌍을 인코딩하여 의미적 유사성을 확보하고, 비트 연산을 통해 빠르게 피벗 토큰을 찾습니다.

- **Technical Details**: HashAttention은 학습된 매핑 함수를 사용하여 Hamming 공간에서 쿼리와 키-값 쌍을 인코딩합니다. 이 방식은 비트 단위 연산을 활용하여 해당 쿼리에서 관련된 피벗 토큰을 효율적으로 식별합니다. 이는 메모리 사용량을 최소화하는 동시에 평균 품질 손실을 0.6 포인트 이내로 유지합니다.

- **Performance Highlights**: HashAttention은 Llama-3.1-8B 모델에서 1/32 배만큼 사용되는 토큰 수를 줄일 수 있으며, 32×32×32의 희소성 조건에서 LightLLM보다 최대 6배, Gpt-Fast보다 최대 4.5배 빠릅니다. 또한 HashAttention은 다른 접근 방식에 비해 추가 비용이 적고, 전반적으로 모델의 효율성을 크게 개선합니다.



### CLDG: Contrastive Learning on Dynamic Graphs (https://arxiv.org/abs/2412.14451)
Comments:
          Accepted by ICDE2023

- **What's New**: 이 논문에서는 동적 그래프에 대한 새로운 비지도 학습 프레임워크인 CLDG(Contrastive Learning on Dynamic Graphs)를 제안합니다. CLDG는 시간에 따른 표현의 일관성을 유지하기 위해 시간적 변환 불변성(temporal translation invariance) 개념을 도입합니다. 이 접근법은 노드의 지역적 및 글로벌 표현을 일정하게 유지하는 것을 목표로 합니다.

- **Technical Details**: CLDG는 다양한 시간 범위에서 동적 그래프의 여러 뷰를 생성하여, 이러한 뷰 간에 대비 쌍을 형성합니다. 이를 통해 임베딩 구조를 통해 노드 및 이웃의 특징 표현을 학습하고, 세 가지 주요 컴포넌트(가중치 공유 인코더, 읽기 함수, 프로젝션 헤드)를 사용하여 모델을 훈련합니다.

- **Performance Highlights**: 실험 결과는 CLDG가 기존의 8개 비지도 기법을 초과하여 7개의 데이터셋에서 최고의 성능을 달성한 것을 보여줍니다. 또한, CLDG는 4개의 준지도 방법보다 더 높은 분류 정확도를 기록하며, 모델 파라미터 및 교육 시간을 각각 평균 2,001.86배 및 130.31배 줄였습니다.



### GenHMR: Generative Human Mesh Recovery (https://arxiv.org/abs/2412.14444)
- **What's New**: 이번 논문에서는 GenHMR이라는 새로운 생성 프레임워크를 소개하여 단일 2D 이미지로부터 3D 인간 메쉬(Human Mesh Recovery, HMR)를 보다 정확하게 복원할 수 있도록 합니다. GenHMR은 2D-3D 매핑 과정에서 불확실성을 명시적으로 모델링하고 완화하는 데 중점을 두고 있습니다. 이 프레임워크는 포즈 토크나이저와 이미지 조건 마스크 변환기(image-conditional masked transformer)라는 두 가지 핵심 구성 요소로 구성되어 있습니다.

- **Technical Details**: GenHMR의 첫 번째 단계에서는 Vector Quantized Variational Autoencoders (VQ-VAE)를 사용하여 지속적인 인간 포즈를 잠재 공간에서의 이산 토큰 시퀀스로 변환합니다. 그 후 두 번째 단계에서는 포즈 토큰 시퀀스의 일부를 무작위로 마스킹하고, 이미지에 조건화된 확률 분포를 학습하여 마스킹된 토큰을 예측하는 훈련을 진행합니다. 이러한 생성적 마스킹 훈련을 통해 GenHMR은 2D 이미지에서 인간 포즈로의 확률적 매핑을 학습합니다.

- **Performance Highlights**: 실험 결과, GenHMR은 Human3.6M, 3DPW 및 EMDB와 같은 표준 데이터셋에서 기존 최첨단(SOTA) 방법보다 20-30%의 오차 감소(MPJPE 기준)를 달성하여 뛰어난 성능을 입증했습니다. 또한 GenHMR은 모호한 이미지 관찰에 강건하며, 복잡한 시나리오에서도 뛰어난 성능을 보여주어 HMR 분야에 중요한 기여를 하고 있습니다.



### ORBIT: Cost-Effective Dataset Curation for Large Language Model Domain Adaptation with an Astronomy Case Study (https://arxiv.org/abs/2412.14436)
- **What's New**: 이번 논문에서는 전문 지식이 필요한 작업을 위해 고품질의 도메인 특화 훈련 데이터의 필요성을 강조하고 있습니다. 기존의 범용 모델은 일반적인 작업에 대해서는 뛰어난 성능을 보이나, 우주 과학과 같은 특정 분야에서는 깊이 있는 이해도가 부족한 경우가 많습니다. 이를 해결하기 위해 저자들은 ORBIT이라는 새로운 데이터 커레이션 방법론을 제안하고, 이를 통해 노이즈가 많은 웹 소스로부터 고품질의 대량 데이터셋을 커스터마이즈할 수 있음을 보여줍니다.

- **Technical Details**: ORBIT은 대규모 웹 데이터셋을 효율적으로 필터링하기 위해 임베딩 기반 유사성 매칭과 BERT 기반 회귀 모델을 결합한 새로운 프레임워크입니다. 이 방법론은 의미적 관련성과 교육적 가치를 모두 고려하여 커뮤니케이션 및 데이터 품질을 보장합니다. 저자들은 이 방법론을 우주 과학을 중심으로 적용하며, FineWeb-Edu에서 10B 토큰의 고품질 데이터셋을 구축했습니다.

- **Performance Highlights**: ORBIT로 훈련된 LLaMA-3-8B 모델인 Orbit-LLaMA는 MMLU 우주 과학 벤치마크에서 69%에서 76%로 성능이 향상된 것을 확인했습니다. 또한, Orbit은 기존의 AstroLLaMA 모델보다 훨씬 높은 평가를 받으며, GPT-4o 평가에서도 73%의 선호도를 기록하여 성공적인 성능 향상을 입증했습니다. 이 방법론은 우주 과학뿐만 아니라 법률 및 의학 분야에서도 데이터 품질을 대폭 향상시켰습니다.



### Cherry-Picking in Time Series Forecasting: How to Select Datasets to Make Your Model Shin (https://arxiv.org/abs/2412.14435)
Comments:
          Proceedings of the 39th AAAI Conference on Artificial Intelligence (AAAI-25), February 25-March 4, 2025, Philadelphia, Pennsylvania, USA

- **What's New**: 이번 논문은 시계열 예측(time series forecasting)에 대한 새로운 접근 방식을 제시합니다. 특히, 연구자들이 데이터셋 선택에서 발생하는 편향, 즉 cherry-picking 데이터셋이 예측 성능 평가에 미치는 영향을 탐구했습니다. 다양한 벤치마크 데이터셋을 활용한 경험적 분석을 통해, cherry-picking이 모델 성능의 왜곡을 초래할 수 있음을 밝혔습니다.

- **Technical Details**: 시계열 예측에서는 해당 문제를 해결하기 위해 다양한 모델링 기법이 사용되며, 특히 머신 러닝(ML) 관점에서 이는 감독 학습(supervised learning) 접근 방식으로 다뤄집니다. 데이터셋은 시간 지연 임베딩(time delay embedding) 기법을 통해 구축되고, 이 방식에서는 슬라이딩 윈도우(sliding windows)를 적용하여 동일한 시계열에서 여러 입력을 생성합니다. 논문에서는 46%의 모델이 최상위 모델로 분류될 수 있으며, 77%가 상위 3위 안에 든다는 점에서 cherry-picking의 심각성을 지적하였습니다.

- **Performance Highlights**: 이 연구에서 도출된 결과는 각 모델의 성능에 대한 잘못된 인식을 초래할 수 있는 cherry-picking의 영향을 보여줍니다. 또한, 최근의 딥러닝 기반 접근 방식이 데이터셋 선택에 민감한 반면, 전통적인 방법들은 더 큰 견고성을 보인다는 사실도 확인했습니다. 결국, 데이터셋 수를 3개에서 6개로 늘릴 경우 잘못된 알고리즘 식별의 위험을 약 40% 줄일 수 있다는 점을 강조했습니다.



### All-in-One Tuning and Structural Pruning for Domain-Specific LLMs (https://arxiv.org/abs/2412.14426)
- **What's New**: 이 논문은 ATP(All-in-One Tuning and Structural Pruning)라는 새로운 프레임워크를 제안합니다. 이 방법은 기존의 두 단계 방식이 아닌, 구조적 프루닝(structural pruning)과 파라미터 파인튜닝(parameter fine-tuning)을 통합하여 한 단계에서 수행합니다. ATP는 훈련 가능한 프루닝 결정 생성기를 통해 최신의 최적 서브구조를 동적으로 식별하여, 도메인 특화 애플리케이션에서 성능 저하를 최소화합니다.

- **Technical Details**: 고전적인 LLM 프루닝 기법은 대개 사전 훈련된 모델에서 효과적으로 파라미터를 제거하는 두 단계로 진행됩니다. 하지만 ATP는 훈련 중에 프루닝 결정이 업데이트되어 새로운 서브구조를 탐색하게 합니다. 또한, LoRA-aware forward 패스와 구조적 희소성 정규화를 포함하여 훈련된 프루닝 결정을 기반으로 파라미터의 기여를 점진적으로 제거합니다.

- **Performance Highlights**: 실험 결과, ATP는 법률 및 헬스케어 도메인에서 기존의 두 단계 프루닝 방법을 초월하는 성능을 보였습니다. 특히, LLaMA2-7B 및 LLaMA3-8B 모델의 40%를 프루닝 했을 때 각각 88% 및 91%의 성능을 회복하는 결과를 보였습니다. 따라서 ATP는 제한된 데이터 환경에서도 도메인 특화 모델의 효율적인 배포 가능성을 제시합니다.



### FedPIA -- Permuting and Integrating Adapters leveraging Wasserstein Barycenters for Finetuning Foundation Models in Multi-Modal Federated Learning (https://arxiv.org/abs/2412.14424)
Comments:
          Accepted for publication in AAAI 2025 (Main Track)

- **What's New**: 이 논문에서는 헬스케어 환경에서 데이터 프라이버시와 컴퓨팅 리소스 제한을 고려하여, FedPIA(Federated Learning via Permuting and Integrating Adapters)라는 새로운 프레임워크를 제안합니다. 이는 Parameter-Efficient Fine-Tuning (PEFT)과 Federated Learning (FL) 접근 방식을 결합한 방법입니다. 이 방법은 클라이언트와 서버 간의 로컬 및 글로벌 어댑터를 효율적으로 통합하여 성능을 개선하고, 데이터 및 작업의 이질성을 극복하는데 초점을 맞추고 있습니다.

- **Technical Details**: FedPIA 프레임워크는 두 가지 접근 방식을 사용하여 클라이언트 어댑터와 서버의 글로벌 어댑터를 통합합니다. 첫째, 서버에서 다양한 클라이언트 어댑터 뉴런을 Permutation(순열)하여 글로벌 어댑터 뉴런과 일치시키고, 둘째, 클라이언트의 글로벌 어댑터를 클라이언트 특정 어댑터에 가까워지도록 재배열한 후 결합합니다. 이러한 과정은 Wasserstein barycenters 이론을 활용하여 수행되며, 그 결과로 안정적인 수렴을 보여줍니다.

- **Performance Highlights**: 2000개가 넘는 클라이언트 레벨 실험을 통해, FedPIA는 시각적 질의 응답(Visual Question Answering) 및 이미지 및 보고서 기반의 다중 라벨 질병 감지와 같은 다양한 의료 이미지 데이터셋과 작업 조건에서 일관되게 뛰어난 성능을 나타냈습니다. FedPIA는 모든 작업 시나리오에서 기존의 PEFT-FL 기법을 초과하며, 이질성 조건에도 불구하고 강력한 성능을 유지하고 있음을 입증했습니다.



### Enhancing Diffusion Models for High-Quality Image Generation (https://arxiv.org/abs/2412.14422)
- **What's New**: 이번 연구에서는 최첨단 생성 모델인 Denoising Diffusion Probabilistic Models (DDPMs)와 Denoising Diffusion Implicit Models (DDIMs)의 포괄적인 구현, 평가 및 최적화가 다루어졌습니다. 이 모델들은 무작위 노이즈를 입력으로 받아 고품질 이미지를 순차적으로 생성하는 방식으로, 이러한 기능을 향상시키기 위한 다양한 기술이 통합되었습니다. Classifier-Free Guidance (CFG), Variational Autoencoders (VAE)를 통한 Latent Diffusion Models, 대체 노이즈 스케줄링 전략이 포함되어 있어 효율적이고 확장 가능한 생성 AI 모델에 대한 수요를 충족시키고자 하였습니다.

- **Technical Details**: 이 논문에서는 DDPM들이 Gaussian 노이즈를 추가하여 원본 이미지를 복원하는 과정에 집중하고 있습니다. 모델 훈련 시 입력으로 RGB 이미지에 Gaussian 노이즈가 점진적으로 추가되고, 이를 통해 노이즈를 역으로 제거하면서 원본 이미지를 복원하는 방법을 학습합니다. 특히, DDIM과 CFG와 같은 고급 방법론을 통합하여 계산 효율성을 극대화하고, 특정 작업 및 데이터셋에 맞춘 노이즈 스케줄링 최적화 작업도 수행하고 있습니다.

- **Performance Highlights**: 평가 결과, DDIM + CFG 조합이 빠른 추론 속도와 우수한 이미지 품질을 달성함을 보여주어, 실제 응용 분야에서 널리 활용될 잠재력을 가지고 있습니다. CIFAR-10 및 ImageNet-100 데이터셋에서의 평가를 통해 이미지 품질 지표인 Frechet Inception Distance (FID)를 사용하여 모델의 성능을 분석하였습니다. VAE와 노이즈 스케줄링의 도전 과제가 강조되어 향후 최적화를 위한 기회를 제시하였습니다.



### DriveGPT: Scaling Autoregressive Behavior Models for Driving (https://arxiv.org/abs/2412.14415)
Comments:
          14 pages, 16 figures, 9 tables, and 1 video link

- **What's New**: 이 논문에서는 자율주행을 위한 확장 가능한 행동 모델인 DriveGPT를 제안합니다. DriveGPT는 순차적 의사결정 과제로서 주행을 모델링하고, 미래의 에이전트 상태를 예측하는 트랜스포머 모델을 학습합니다. 기존의 오픈 소스 데이터셋에 비해 약 50배 많은 1억 개 이상의 고품질 주행 예제를 활용하여 모델 파라미터를 10억 개 이상으로 확장했으며, 이는 행동 모델의 성능을 크게 향상시킵니다.

- **Technical Details**: DriveGPT는 인코더-디코더 아키텍처를 사용하는 표준 행동 모델을 기반으로 합니다. 순차적 예측 과제로서, 드라이빙 맥락 정보와 역사적인 에이전트 위치에 따라 목표 에이전트의 미래 위치를 예측합니다. 이 모델은 트랜스포머 기반으로 설계되어 있어 순차적 모델링 작업에서 높은 확장성을 제공합니다.

- **Performance Highlights**: DriveGPT는 다양한 규모의 계획 작업에서 평가되었으며, 정량적 메트릭과 질적 예제를 통해 성능이 입증되었습니다. 대규모 데이터셋으로 사전 학습을 통해 상태-of-the-art 기준을 초월하였고, 복잡한 현실 세계의 시나리오에서 폐쇄 루프 주행을 포함한 다양한 상황에서도 향상된 성능을 보입니다.



### I0T: Embedding Standardization Method Towards Zero Modality Gap (https://arxiv.org/abs/2412.14384)
Comments:
          16 figures, 8 figures, 7 tables

- **What's New**: 본 연구에서 제안하는 I0T 프레임워크는 이미지-텍스트 embedding 간의 modality gap을 최소화하는 두 가지 방법을 소개합니다. 첫 번째는 post-hoc embedding standardization 방법인 I0T_{post}로, modality gap을 거의 0으로 줄이는 기능이 있습니다. 두 번째는 학습 가능한 I0T_{async} 방법으로, 두 개의 normalization layer를 각 encoder에 추가하여 문제를 해결합니다.

- **Technical Details**: I0T 프레임워크는 frozen encoders에서 평균 벡터를 빼고 Frobenius normalization으로 embedding 활성화를 표준화하여 modality gap을 줄입니다. 이 연구는 이미지와 텍스트 encoder 간의 모드 특성을 고려하여 embedding 간의 불일치를 해결하는 데 중점을 두고 있습니다. 특히, CLIPScore 대신 I0TScore이라는 새로운 자동 평가 메트릭을 제안하여 모델의 성능을 더 잘 평가할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 modality gap을 줄이며, text-to-image retrieval 점수를 각각 9.2%와 6.7% 향상시킵니다. 또한, I0TScore는 서로 다른 modality 간에도 적용할 수 있는 첫 번째 자동 평가 메트릭으로, CLIPScore의 한계를 극복하며 두 가지 방법 모두 원래 모델의 embedding 표현을 유지합니다. I0T 프레임워크는 downstream 성능에 부정적인 영향을 주지 않으면서 gap을 줄이는 데 성공합니다.



### Surrealistic-like Image Generation with Vision-Language Models (https://arxiv.org/abs/2412.14366)
Comments:
          2023 Joint international Scientific conferences on AI and Machine Learning (BNAIC-BeNeLearn)

- **What's New**: 최근 생성 AI의 발전으로 텍스트, 이미지 및 코드 등 다양한 유형의 콘텐츠를 손쉽게 생성할 수 있게 되었습니다. 이 논문에서는 DALL-E, Deep Dream Generator, DreamStudio 등의 비전-언어 생성 모델을 사용하여 초현실주의 화풍의 이미지를 생성하는 방법을 탐구합니다. 다양한 이미지 생성 설정과 모델을 통해 생성된 이미지의 품질을 평가하고, 편집된 기본 이미지가 결과 이미지에 미치는 영향을 이해하고자 합니다.

- **Technical Details**: 인공지능(AI)이 텍스트로부터 이미지를 생성하는 능력은 최근 몇 년 동안 활발히 연구되고 있는 분야입니다. 연구의 주요 초점은 텍스트-투-이미지 생성(text-to-image generation)과 현실적 이미지의 초현실주의 화풍 변환을 포함합니다. 본 연구에서 제안된 '초현실적 이미지(surrealistic-like image)'는 드림 같은 분위기, 예상치 못한 대조, 상징성과 은유 등을 포함한 특징을 기반으로 정의됩니다.

- **Performance Highlights**: 실험 결과, DALL-E 2는 ChatGPT가 생성한 프롬프트를 사용할 때 가장 뛰어난 성능을 보여주었습니다. 연구를 통해 235개의 이미지가 생성되었고, 여기에 대한 평가가 이루어졌습니다. 본 논문은 이미지를 생성하기 위한 최적의 설정을 탐색하고 초현실적 이미지를 만드는 과정에 대한 인사이트를 제시합니다.



### Enabling Realtime Reinforcement Learning at Scale with Staggered Asynchronous Inferenc (https://arxiv.org/abs/2412.14355)
- **What's New**: 본 논문에서는 실시간 강화 학습에서의 회귀(regret)를 최소화하는 새로운 접근 방식을 제안합니다. 전통적인 시퀀셜 인터랙션 방식을 통해는 더 큰 모델이 필연적으로 요구되는 복잡한 문제에서, 느린 반응 시간으로 인해 효율이 저하될 수 있다는 점을 지적합니다. 연구진은 비동기적인 상호작용 패러다임을 통해 복잡한 모델이 빠르게 동작할 수 있도록 하여 실시간 환경에서의 제어력을 유지할 수 있는 방법을 탐구합니다.

- **Technical Details**: 논문은 비동기식 환경에서의 MDP(Markov Decision Process) 정의를 확장하여 새로운 비동기 MDP 모델을 제시합니다. 이 모델은 에이전트가 행동을 선택하지 못하는 동안 환경이 일정한 기본 행동을 따르도록 가정합니다. 이러한 비동기적 상호작용은 에이전트가 존재하는 것과 비슷한 확률론적 행동을 통해 정의되며, 이로 인해 더 많은 계산 자원이 제공될 경우 모델이 매 단계에서 행동을 선택하는 것이 가능해집니다.

- **Performance Highlights**: 실험 결과, 비동기적 추론 과정을 활용함으로써 행동 추론 시간이 긴 모델을 사용했음에도 불구하고, Pokémon 및 Tetris 같은 실시간 게임에서 매우 큰 모델들이 효과적으로 학습할 수 있음을 입증합니다. 특히, 행동 추론 시간이 길어도 증가하는 추론 과정의 수가 선형적으로 증가하여 기존 방식들에 비해 훨씬 큰 규모의 모델 사용이 가능함을 보여줍니다.



### Is Peer-Reviewing Worth the Effort? (https://arxiv.org/abs/2412.14351)
Comments:
          The 31st International Conference on Computational Linguistics (COLING 2025)

- **What's New**: 이 논문은 동료 평가(peer-reviewing)가 중요한 논문을 식별하는 데 얼마나 효과적인지를 예측하는 작업으로 다룹니다. 연구자들은 어느 논문이 향후 높은 인용(citation)을 받을지를 예측하기 위해 논문의 발표 장소(venue)와 조기 인용(early returns)의 관계를 분석하였습니다. 결과적으로 조기 인용이 이후 인용과 더 큰 상관관계를 가지고 있다는 점을 강조하며, 이는 향후 연구에서도 더욱 중요한 방향성이 될 수 있음을 시사합니다.

- **Technical Details**: 연구에서는 조기 인용과 발표 장소를 기준으로 논문을 집계하고, 네 번째 해의 인용 수치를 h-index와 impact(μ)를 통해 요약합니다. 논문은 다양한 조건과 다양한 출처(ACL, PubMed, ArXiv)의 논문을 포함하여 반복 검증을 통해 조기 인용이 발표 장소보다 더 효과적이라는 결론을 도출합니다. 이에 따라 논문들은 초기 인용을 통해 우선순위를 매기는 것이 더 효과적이며, 이는 더 포괄적이고 강건하다는 설명이 포함되어 있습니다.

- **Performance Highlights**: 조기 인용이 있는 덜 선택적인 발표 장소(Workshops/ArXiv)의 논문들이 더 선택적인 발표 장소의 논문들보다 종종 더 높은 인용 수를 기록하는 경향이 있음을 보여주고 있습니다. 연구 결과, 초기 인용에 기반한 우선순위가 논문 선택에 있어서 상당한 유용성을 제공하며, 동료 평가의 효과성에 대한 새로운 관점을 제공합니다. 이러한 통찰력은 향후 학술 출판 및 심사 과정에서 중요한 참조로 작용할 것으로 기대됩니다.



### A Unifying Information-theoretic Perspective on Evaluating Generative Models (https://arxiv.org/abs/2412.14340)
- **What's New**: 본 논문에서는 생성 모델의 출력 평가를 위해 유의미한 메트릭을 정의하는 데 중점을 두고 있다. 기존의 precision과 recall 개념을 적용하여 생성된 데이터의 사실성과 다양성을 측정하기 위한 새로운 tri-dimensional metric인 Precision Cross-Entropy (PCE), Recall Cross-Entropy (RCE), Recall Entropy (RE)를 제안한다. 특히, 이 메트릭은 정보 이론의 관점에서 kNN 기반 메트릭을 통합하여 다양한 실패 모드를 구분할 수 있도록 설계되었다.

- **Technical Details**: 제안된 메트릭은 생성 모델의 출력을 fidelity와 diversity의 두 가지 주요 측면으로 나누어 평가하며, intra-class와 inter-class 다양성의 차이를 명확하게 측정한다. 이는 기존의 1D 메트릭, 즉 Inception Score (IS)와 Fréchet Inception Distance (FID)와는 달리, 여러 개별 실패 모드를 식별할 수 있는 차별화된 기능을 제공한다. kNN 밀도 추정 기법을 응용하여 구현된 이 메트릭은 샘플 및 모드 수준 분석이 가능하다.

- **Performance Highlights**: 실험 결과, 새로운 메트릭의 각 구성 요소가 출력의 질에 대해 매우 민감한 반응을 보였으며, 기존 메트릭의 단점이나 바람직하지 않은 행동을 드러내었다. 제안된 방법은 특히 이미지 평가에 중점을 두고 있으나, 다양한 데이터 유형에 쉽게 일반화될 수 있는 장점이 있다. 또한, 실험을 통해 기존 메트릭보다 더 효과적으로 평가의 제도화된 기준(ideals)에 부합함을 입증하였다.



### Embedding Cultural Diversity in Prototype-based Recommender Systems (https://arxiv.org/abs/2412.14329)
- **What's New**: 이 논문에서는 인기 편향(popularity bias) 문제를 해결하기 위해 프로토타입 기반 행렬 분해 방법을 개선하는 새로운 접근 방식을 제안합니다. 문화 제품을 추천하는 시스템에서 이 편향이 문화적 대표성에 미치는 부정적인 영향을 우려하며, 인구 통계학적 요소를 통해 이를 완화하고자 합니다. 구체적으로, 원래 프로토타입을 필터링하고 프로토타입의 균일한 분포를 강화하는 정규화 기법을 도입하여, 보다 공정한 추천 결과를 도출하는 것에 초점을 맞췄습니다.

- **Technical Details**: 연구에서는 ProtoMF(Prototype-based Matrix Factorization) 모델의 두 가지 주요 혁신을 제시합니다. 첫 번째는 Prototype K-filtering으로, k개의 가장 가까운 프로토타입을 선택하여 사용자와 항목의 표현을 개선하는 방법입니다. 두 번째는 Prototype-Distributing Regularizer로, 프로토타입의 분포를 균등화하여 다양한 문화적 표현을 촉진하는 메커니즘입니다. 이러한 접근법을 통해 인구 통계적 편향을 해결하고, 추천 시스템의 성능을 유지하면서 편향을 감소시키려 했습니다.

- **Performance Highlights**: 모델을 MovieLens-1M, LastFM-2b, Amazon Reviews’23 데이터셋에서 평가한 결과, 긴 꼬리 항목의 평균 순위가 27% 감소하고, 저명하지 않은 국가의 항목 순위는 2% 감소하는 성과를 보여주었습니다. 또한, HitRatio@10에서 기존 최고 성능 모델 대비 2% 개선된 결과를 달성했습니다. 이로 인해 추천의 공정성을 높이면서도 추천 품질을 저하하지 않는 성과를 검증하였습니다.



### Semantic Role Labeling of NomBank Partitives (https://arxiv.org/abs/2412.14328)
Comments:
          SUMEval-2: The 2nd Workshop on Scaling Up Multilingual & Multi-Cultural Evaluation at the 31st International Conference on Computational Linguistics (COLING 2025)

- **What's New**: 이번 논문은 영어의 부분 명사에 대한 의미역 레이블링(Semantic Role Labeling) 기술을 다루고 있습니다. 특히 부분 명사와 관련된 ARG1 및 연관 문제에 대한 두 가지 작업(task)에 대한 결과를 보고하였습니다. 전통적인 기계 학습과 Transformer 기반의 기계 학습 기법을 활용한 시스템을 설명하고 있으며, 최고 성능을 기록한 시스템은 Penn Treebank의 'gold' 파서를 사용하여 F1 점수 91.74%를 달성했습니다.

- **Technical Details**: 이 연구에서 사용된 시스템은 BERT 기반의 구성요소(Deep Learning 1)와 기능 기반 구성요소로 이루어진 앙상블 시스템입니다. Deep Learning 2는 언어학적 특징의 유용성을 평가하기 위해 보조 작업(auxiliary task)으로 사용되었습니다. Deep Learning 2에서 사용된 언어학적 특징은 앙상블 시스템의 기능 기반 구성요소의 일부로, 제공된 비금 고(gold) 데이터와 비금 고 데이터(non-gold data)로부터 예측 모델링을 지원합니다.

- **Performance Highlights**: 연구 결과, 비금 고 데이터에 대한 예측 성능에서 초기 모델들이 2-3% 수준의 F-score 감소를 보였습니다. 이 연구는 금 고 데이터가 모델 성능 향상에 중요한 역할을 하였음을 시사합니다. 그러나 부분 명사 작업은 복잡성이 더 높아 일반적으로 F-score가 낮았습니다. 논문에서는 다양한 모델의 성능에 영향을 미치는 요인들에 대한 깊은 분석이 필요함을 강조하고 있습니다.



### The Role of Handling Attributive Nouns in Improving Chinese-To-English Machine Translation (https://arxiv.org/abs/2412.14323)
Comments:
          18th Workshop on Building and Using Comparable Corpora (BUCC) at the 31st International Conference on Computational Linguistics (COLING 2025)

- **What's New**: 이 연구에서는 문법적인 차이가 큰 언어 간 번역에서 발생하는 도전 과제를 해결하기 위해 Chinese의 속성 명사(attributive nouns) 번역 문제에 초점을 맞추고 있습니다. Penn Chinese Discourse Treebank에서 생략된 입자 X('DE')를 수동으로 삽입하여 뉴스 기사 제목의 데이터셋을 개발하였습니다. 이 데이터셋은 Hugging Face의 중국어-영어 번역 모델을 세심하게 조정(fine-tune)하는 데 사용되었습니다.

- **Technical Details**: 귀 연구는 중국어의 속성 명사가 영어 번역에서 자주 발생하는 모호성을 어떻게 해결하는지를 다루고 있습니다. 특정 function word인 'DE'의 처리를 개선하기 위해 만들어진 데이터셋은 기계 번역 시스템의 성능을 향상시킵니다. 이러한 접근법은 이전 연구들이 제안한 광범위한 전략들과 보완적으로 작용하면서도, 중국어-영어 번역에서 빈번하게 발생하는 오류 유형을 구체적으로 해결합니다.

- **Performance Highlights**: 이 연구는 정확한 기계 번역을 위해 특정 명사의 번역 실수를 줄이는 데 중요한 실용적 개선을 제공합니다. 이 방법은 번역 모델의 정확도를 높이는 데 기여하며, 기계 번역 시스템의 전반적인 성능 향상에 기여할 것으로 기대됩니다.



### Multi-OphthaLingua: A Multilingual Benchmark for Assessing and Debiasing LLM Ophthalmological QA in LMICs (https://arxiv.org/abs/2412.14304)
Comments:
          Accepted at the AAAI 2025 Artificial Intelligence for Social Impact Track (AAAI-AISI 2025)

- **What's New**: 본 연구는 다국어 기반의 안과 질문-응답(QA) 벤치마크인 Multi-OphthaLingua를 처음으로 소개하고 있습니다. 이 데이터셋은 영어, 스페인어, 필리핀어, 포르투갈어, 만다린, 프랑스어, 힌디 등 7개 언어로 총 1184개의 질문을 포함하며, 서로 간의 직접적인 언어 비교를 가능하게 합니다. 이를 통해 의료 분야에서 LLM이 갖는 언어적 편향을 평가하고 개선하는 데 기여하고자 합니다.

- **Technical Details**: Multi-OphthaLingua 데이터셋은 각 언어에서 검사 및 평가된 질문-답변 쌍으로 구성되어 있으며, 언어별로 공정성을 확보하기 위해 네이티브 스피커가 검증했습니다. 또한, 연구진은 LLM의 성능을 비교하기 위해 Llama-2, GPT 3.5 등 6개 인기 LLM을 평가하였고, 이 과정에서 언어에 따른 성능 격차와 의료 도메인 지식 부족 문제를 발견했습니다. 이러한 문제를 해결하기 위해, CLARA라는 새로운 비편향 처리 방법을 제안하고 다중 에이전트 접근 방식을 도입했습니다.

- **Performance Highlights**: 연구 결과, CLARA 메소드는 6개 언어에서 모든 LLM의 질문-응답 정확도를 향상시키는 동시에 언어 간 성능 격차를 줄이는 데 성공했습니다. 특히, LMIC 언어의 경우 LLM 성능이 저조했으나, CLARA를 통해 이러한 격차가 크게 개선되었습니다. 그리고 이 연구는 의료 분야의 언어적 공정성을 높이기 위한 새로운 지침을 제시하며, 글로벌한 LLM 활용 가능성을 확대하는 데 중요한 기초 자료를 제공합니다.



### SAFERec: Self-Attention and Frequency Enriched Model for Next Basket Recommendation (https://arxiv.org/abs/2412.14302)
- **What's New**: 본 논문에서는 NBR(Next-Basket Recommendation) 작업을 위한 새로운 알고리즘인 SAFERec를 제안합니다. 이 알고리즘은 NIR(Next Item Recommendation)에서 우수한 성능을 보이는 transformer 기반 아키텍처를 활용하며, 아이템의 빈도 정보를 통합하여 NBR 작업에 적합성을 높입니다. SAFERec는 기존 알고리즘과 비교해 8% 향상된 Recall@10 성능을 보여줍니다.

- **Technical Details**: SAFERec는 사용자 구매 이력을 mini-batch로 처리하고, 각 바구니를 sparse multi-hot vector 형식으로 변환합니다. 이를 통해 각 사용자의 구매 이력이 sparse matrix 형태로 표현되며, 이는 최근 바구니를 고려하는 최대 길이를 나타내는 하이퍼파라미터 L을 도입하여 처리합니다. SAFERec는 transformer 레이어와 아이템 빈도 인식 모듈을 통합하여 NBR을 위한 효과적인 솔루션을 제공합니다.

- **Performance Highlights**: SAFERec의 성능은 여러 공공 데이터셋에서 테스트되었으며, 모든 다른 기준선 방법들보다도 우수한 성과를 보였습니다. 특히, SAFERec는 Recall@10에서 최대 8%의 개선을 이루어 시사하는 바가 큽니다. 이러한 결과는 SAFERec가 더 혁신적인 아이템 세트를 추천하는 데 효과적임을 보여줍니다.



### Temporally Consistent Object-Centric Learning by Contrasting Slots (https://arxiv.org/abs/2412.14295)
- **What's New**: 최근 동영상에서의 비지도 객체 중심 학습(object-centric learning)은 구조적 표현을 추출하는 유망한 접근 방식이다. 이를 통해 자율 제어(autonomous control)와 같은 하위 작업을 지원하기 위해선 표현이 조합 가능(compositional)하고 시간적으로 일관성(temporally consistent)이 있어야 한다. 기존 방법들은 종종 장시간의 안정성을 유지하지 못하며, 이는 훈련 목표가 시간적 일관성을 강제하지 않기 때문이다. 본 연구에서는 비디오 객체 중심 모델을 위한 새로운 객체 수준의 시간 대비 손실(temporal contrastive loss)을 도입하였다.

- **Technical Details**: 우리는 Slot Contrast라는 새로운 방법론을 제안하여 시간적 일관성을 유지하는 도전 과제를 해결한다. 이 방법은 사람의 주석이 필요 없이 실제 세계 비디오 데이터에 적용 가능하며, 배치 전체에서 슬롯 표현(slot representations)을 대조하면서 연속 프레임 사이의 시간적 일관성을 보장하는 자기 지도 대비 학습(self-supervised contrastive learning) 목표를 포함한다. 우리의 접근 방식은 기존 입력 재구성 기반 영상 객체 중심 프레임워크를 확장하여 슬롯 간의 대조를 통해 일관된 표현을 발견하도록 모델을 조정한다.

- **Performance Highlights**: 우리는 Slot Contrast가 학습된 표현의 시간적 일관성을 향상시킬 뿐만 아니라, 객체 발견(object discovery) 작업에서도 최첨단 성과를 달성한다고 보여준다. 이 방법은 운동 단서(motion cues)를 사용하는 약한 지도 학습(weakly-supervised) 모델들을 초월하며, 탈출장면(on-the-fly)에서의 비지도 객체 추적(object tracking)과 잠재적 객체 동역학 모델링(latent object dynamics modeling) 같이 복잡한 하위 작업에서 더욱 효과적임을 입증한다.



### PixelMan: Consistent Object Editing with Diffusion Models via Pixel Manipulation and Generation (https://arxiv.org/abs/2412.14283)
Comments:
          AAAI 2025; version includes supplementary material; 27 Pages, 15 Figures, 6 Tables

- **What's New**: 본 논문에서 제안하는 PixelMan은 기존의 Diffusion Models를 활용한 일관된 객체 편집을 위한 새로운 방법으로, 편집 과정에서 불필요한 전환 과정을 생략하여 효율성과 일관성을 크게 개선합니다. 이 방법은 훈련 없이 픽셀 조작(Pixel Manipulation)을 통해 객체를 직접 복제한 후, 이를 효율적으로 조화시킴으로써 편집된 이미지의 질을 높입니다. PixelMan은 경쟁하는 모든 훈련 기반 및 훈련 없는 방법을 초월하여, 단지 16단계의 추론으로 다양한 편집 작업을 수행할 수 있다는 점에서 혁신적입니다.

- **Technical Details**: PixelMan은 피사를 픽셀 공간에서 직접 수정하여 표적 위치에 객체의 복사본을 생성하며, 이를 통해 보다 높은 이미지 일관성을 달성합니다. 이 과정에는 세 단계 브랜칭의 비전환 샘플링 접근법이 포함되어, 명확한 "델타" 편집 방향을 사용자 요구에 맞게 계산합니다. 또한, 자동 주의(Self-Attention) 메커니즘에 의한 정보 누수를 방지하기 위해 누출 방지 기법을 도입하여 조화로운 배경 복원을 이룹니다.

- **Performance Highlights**: PixelMan은 COCOEE와 ReS 데이터 세트를 기반으로 한 실험을 통해 객체, 배경, 의미적 일관성 지표에서 우수한 성능을 나타냈습니다. 이 방법은 기존 인기 있는 방법보다 적은 평균 지연시간과 낮은 수의 Network Function Evaluations(NFEs)로 주목받으며, 훈련이 필요 없는 방법으로 높은 편집 품질을 달성했습니다.



### Fake News Detection: Comparative Evaluation of BERT-like Models and Large Language Models with Generative AI-Annotated Data (https://arxiv.org/abs/2412.14276)
Comments:
          Accepted in Knowledge and Information Systems Journal

- **What's New**: 이번 연구는 BERT 계열의 인코더 전용 모델과 자율 회귀 디코더 전용 대규모 언어 모델(LLM)을 비교하여 가짜 뉴스 탐지의 효과성을 평가하고 있습니다. 또한, GPT-4의 지원을 받아 라벨링된 뉴스 데이터셋을 소개하여 인간 전문가에 의해 검증된 신뢰 가능한 결과를 제공합니다. 수집된 데이터셋을 활용하여 두 모델 가족 간의 성능을 분석하고, AI 기반 주석과 인간 감독을 결합한 접근 방식이 가짜 뉴스 탐지에 효과적임을 강조합니다.

- **Technical Details**: 가짜 뉴스 탐지에서의 ML 기반 접근 방식의 주요 과제는 정확하고 신뢰할 수 있는 데이터 라벨링의 가용성입니다. 연구진은 10,000개의 뉴스 기사를 수집하고 최신 LLM인 OpenAI GPT-4를 이용해 이들을 라벨링했습니다. BERT 계열의 모델과 LLM의 성능을 비교하고, 특히 BERT 모델이 분류 작업에서 뛰어난 성능을 나타내는 반면, LLM은 텍스트 변동에 대해 더 강한 견고성을 보임을 분석하였습니다.

- **Performance Highlights**: BERT 계열 모델은 일반적으로 분류 작업에서 LLM에 비해 더 나은 성능을 발휘하지만, LLM은 텍스트의 작은 변화에 대해 강인한 성능을 보여줍니다. AI 라벨은 인간의 감독 하에 이루어졌을 때, 약한 감독(원거리 감독)에 의해 얻어진 라벨보다 더 정확한 분류 결과를 나타냈습니다. 이에 따라 데이터의 대량 처리와 가짜 뉴스의 탐지를 위한 강력한 AI 탐지 방법을 통해 사회적 인식을 높이는 데 집중하고 있습니다.



### Split Learning in Computer Vision for Semantic Segmentation Delay Minimization (https://arxiv.org/abs/2412.14272)
- **What's New**: 이번 논문에서는 자원 제약이 있는 장치를 위한 실시간 컴퓨터 비전(CV) 애플리케이션의 요구에 맞춘 semantic segmentation의 추론 지연을 최소화하기 위한 새로운 접근법인 split learning (SL)을 제안합니다. SL은 딥 뉴럴 네트워크(DNN)를 에지 디바이스와 중앙 서버 간에 분할하여 데이터 처리를 로컬화하고 전송해야 하는 데이터를 줄임으로써 지연 문제를 해결합니다.

- **Technical Details**: 논문의 주요 기여는 대역폭 할당(bandwidth allocation), 에지 디바이스의 DNN에서 컷 레이어(cut layer) 선택, 그리고 중앙 서버의 처리 자원(procesing resource) 할당의 공동 최적화(joint optimization)입니다. 또한, 병렬(parallel) 및 직렬(serial) 데이터 처리 시나리오를 조사하고, 연산 요구사항을 줄이면서 근사적 최적 성능을 유지하는 저복잡도( low-complexity) 휴리스틱 솔루션을 제안합니다.

- **Performance Highlights**: 수치 결과에 따르면, 제안한 방법은 추론 지연을 효과적으로 감소시키며, 자원 제약이 있는 동적인 환경에서 실시간 CV 애플리케이션 향상을 위한 SL의 잠재력을 보여줍니다. 이 접근법은 특히 자율주행 차량 및 스마트 시티 인프라와 같은 응용 프로그램에 필수적인 성능을 제공합니다.



### Syzygy: Dual Code-Test C to (safe) Rust Translation using LLMs and Dynamic Analysis (https://arxiv.org/abs/2412.14234)
Comments:
          Project Webpage: this https URL. Preliminary version accepted at LLM4Code 2025, 34 pages

- **What's New**: 이 논문에서는 C 프로그래밍 언어의 안전성을 보장하기 위해 Rust로 자동 번역하는 Syzygy라는 새로운 접근 방법을 소개합니다. C는 수동 메모리 관리와 unsafe pointer operations로 인해 많은 취약점을 가지고 있지만, Rust는 성능을 포기하지 않으면서 메모리 안전성을 제공합니다. 이 연구는 LLM(대형 언어 모델)에 기반한 코드 및 테스트 번역을 동적 분석(dyanamic analysis)에서 생성된 실행 정보를 통해 구현합니다.

- **Technical Details**: Syzygy는 코드 요소의 의존성 순서에 따라 프로그램을 반복적으로 실행하는 인크리멘탈(Incremental) 번역 기법을 사용합니다. 이 과정에서 각 스텝의 정확성을 유지하고, LLM의 강력함과 동적 분석을 결합하는 독창적인 통찰(value insights)을 제공합니다. 구체적으로, 약 3000줄의 코드와 98개의 기능을 가진 고성능 압축 라이브러리인 Zopfli를 성공적으로 Rust로 번역하고 검증했습니다.

- **Performance Highlights**: 지금까지 이루어진 C에서 안전한 Rust 코드로의 자동화되고 테스트로 검증된 번역 중 가장 큰 규모인 이 연구는, 원본 C 프로그램과의 동등성을 다양한 입력에 대해 테스트하여 번역의 정확성을 확인했습니다. 이 연구는 코드 생성(code generation)과 테스트를 결합하는 데 있어 새로운 가능성을 열어줍니다.



### A Survey on Inference Optimization Techniques for Mixture of Experts Models (https://arxiv.org/abs/2412.14219)
Comments:
          Work in Progress

- **What's New**: 대규모 모음 전문가(Mixture of Experts, MoE) 모델의 출현은 인공지능 분야에서 중요한 발전을 이루었으며, 조건부 계산을 통해 향상된 모델 용량과 계산 효율성을 제공합니다. 하지만 이러한 모델을 배포하고 추론하는 데 있어 계산 자원, 지연 시간(latency), 에너지 효율성과 같은 상당한 도전 과제가 존재합니다. 본 연구는 MoE 모델의 추론 최적화 기술에 대한 포괄적인 조사를 제공하며, 모델 수준, 시스템 수준, 하드웨어 수준의 최적화 접근법을 체계적으로 분석합니다.

- **Technical Details**: 이 논문에서는 효율적인 전문가 설계, 주의 메커니즘(attention mechanisms), 가지치기(pruning), 양자화(quantization), 지식 증류(knowledge distillation)와 같은 다양한 압축 기술을 포함한 아키텍처 혁신을 모델 수준에서 조사합니다. 시스템 수준에서는 분산 컴퓨팅(distributed computing) 접근법, 부하 분산(load balancing) 메커니즘, 효율적인 스케줄링 알고리즘을 통해 확장성을 강화하는 방안을 모색합니다. 마지막으로, 하드웨어 특정 최적화와 성능을 극대화하기 위한 공동 설계(co-design) 전략에 대해 논의합니다.

- **Performance Highlights**: MoE 모델은 대규모 언어 모델의 배포에 있어 매력적인 솔루션으로, 전문화된 서브 네트워크를 활용하여 모델 용량을 효율적으로 분산할 수 있습니다. 최근 Mixtral 8x7B, Switch Transformers, GShard와 같은 구현 사례들은 이 접근 방식이 수조 개의 매개변수로 언어 모델을 확장하면서도 합리적인 계산 요구를 유지할 수 있음을 입증하고 있습니다. 또한 이 연구는 현재의 솔루션들을 구조화하여 검토하며 MoE 추론 최적화에서의 주요 도전 과제와 유망한 연구 방향을 제시하고 있습니다.



### Heterogeneous Multi-Agent Reinforcement Learning for Distributed Channel Access in WLANs (https://arxiv.org/abs/2412.14218)
- **What's New**: 이 논문에서는 무선 로컬 영역 네트워크에서의 분산 채널 액세스 문제를 해결하기 위한 다중 에이전트 강화 학습(MARL) 접근 방식을 연구합니다. 특히, 에이전트들이 이질적으로 가치 기반(value-based) 또는 정책 기반(policy-based) 강화 학습 알고리즘을 채택하도록 허용하는 도전적인 상황을 다룹니다. 연구진은 이질적인 MARL 훈련 프레임워크인 QPMIX를 제안하며, 이는 중앙 집중형 훈련(centralized training)과 분산 실행(distributed execution)을 통해 다양한 에이전트들이 협력할 수 있도록 합니다.

- **Technical Details**: QPMIX는 레이턴시와 충돌률을 줄이면서 네트워크의 처리량을 극대화하고 공정성을 보장하는 데 중점을 두고 있습니다. 이 연구는 선형 가치 함수 근사(linear value function approximation)를 사용할 때 제안된 이질적인 MARL 방법의 수렴(convergence)을 이론적으로 증명합니다. 또한, 시뮬레이션 결과를 통해 다양한 트래픽 시나리오에서 QPMIX 알고리즘이 전통적인 CSMA/CA 방식과 비교할 때 처리량, 평균 대기 시간(mean delay), 지연 변동(delay jitter), 충돌률에서 개선되었음을 보여줍니다.

- **Performance Highlights**: QPMIX 알고리즘은 포화된 트래픽 시나리오뿐만 아니라 비포화 및 지연 민감한 트래픽 시나리오에서도 강력한 성능을 보여주며, 이질적인 에이전트 간의 협력을 촉진하는 효과적인 방법으로 입증됩니다. 연구 결과, QPMIX는 독립 학습(independent learning)과 비교할 때 이질적인 에이전트 간의 협력을 더욱 향상시키는 능력을 보여줍니다. 이러한 성과는 차세대 WLAN을 위한 효율적인 DCA 솔루션으로서 중요한 발전을 이루며, 다양한 서비스 요구 사항을 만족시킬 수 있습니다.



### Generative AI Toolkit -- a framework for increasing the quality of LLM-based applications over their whole life cyc (https://arxiv.org/abs/2412.14215)
Comments:
          16 pages, 6 figures. For source code see this https URL

- **What's New**: 이번 논문에서는 LLM 기반 애플리케이션의 개발 및 운영 과정에서의 효율성을 극대화하기 위해 Generative AI Toolkit을 소개합니다. 이 툴킷은 개발 주기 전반에 걸쳐 필수 워크플로우를 자동화하여 품질을 향상시키고 릴리즈 주기를 단축시킵니다. 또한, 사용 사례와 최선의 사례를 공유하고 향후 발전 방향을 제시합니다.

- **Technical Details**: 대규모 언어 모델(LLM)은 자연어 처리 작업을 수행할 수 있는 머신러닝 모델로, 사용자 입력에 대한 반응이 프롬프트(prompt)의 변화에 민감하다는 것이 밝혀졌습니다. LLM 기반 애플리케이션은 '환각(hallucination)' 현상으로 인해 잘못된 출력이 발생할 수 있으며, 이러한 문제를 해결하기 위해 개발 과정에서 충분한 테스트가 필요합니다. 현재 CI/CD(Continuous Integration / Continuous Deployment) 파이프라인과 같은 자동화 도구가 부족하여, LLM 애플리케이션의 전 생애 주기를 포괄하는 프레임워크가 드물게 존재합니다.

- **Performance Highlights**: Generative AI Toolkit을 사용함으로써 LLM 기반 애플리케이션의 개발과 운영에서 발생하는 여러 문제를 해결할 수 있는 효과성을 입증했습니다. 사례 연구를 통해 툴킷의 사용으로 품질을 개선하고 릴리스를 효율적으로 관리할 수 있음을 보여줍니다. 향후 오픈소스로 공개하여 다른 팀들도 활용하고 개선할 수 있도록 하는 점이 이 툴킷의 중요한 특징입니다.



### GraphicsDreamer: Image to 3D Generation with Physical Consistency (https://arxiv.org/abs/2412.14214)
- **What's New**: 이 논문은 인공지능 생성 3D 모델링 분야에서 혁신적인 GraphicsDreamer 방법을 소개합니다. 이 방법은 단일 이미지에서 고급 사용이 가능한 3D 메시(mesh)를 생성하는 기법으로, PBR(물리 기반 렌더링) 조명 방정식을 통합한 교차 영역 확산 모델을 사용합니다. GraphicsDreamer는 3D 객체가 신뢰할 수 있는 질감 세부 정보를 갖고 현실적인 재조명을 지원할 수 있도록 PBR 제약 조건을 강화합니다.

- **Technical Details**: GraphicsDreamer는 다중 보기 이미지와 PBR 재료를 예측하는 두 단계 방식으로 설계되었으며, 선명한 기하학적 구조와 디테일을 아우르는 심층 학습 기반의 역 렌더링 접근 방식을 채택하고 있습니다. 이 모델은 색상, 노멀, 깊이 및 PBR 구성요소를 포함하는 6개 도메인의 결합 분포를 예측하여 3D 객체의 기하학적 구조를 이해합니다. 또한, 3D 객체의 토폴로지 최적화 및 UV 언랩핑 자동화 기능을 통해 Blender, Unreal Engine 및 Unity와 같은 렌더링 엔진으로 직접 가져오기가 가능합니다.

- **Performance Highlights**: 광범위한 실험 결과, GraphicsDreamer는 이전의 방법들과 비교하여 합리적인 시간 내에 고품질 3D 자산을 생성할 수 있음을 보여주었습니다. 이 방법은 PBR 질감 세부 사항과 함께 매끄러운 기하학을 제공하여 그래픽 엔진에서 즉시 사용할 수 있도록 지원합니다. GraphicsDreamer는 기하학적 및 질감 세부 사항의 측면에서 선도적인 수준에 있으며 전체 PBR 재료 맵과 정돈된 토폴로지를 통해 생성된 3D 모델의 실용성을 크게 향상시킵니다.



### Tree-of-Code: A Hybrid Approach for Robust Complex Task Planning and Execution (https://arxiv.org/abs/2412.14212)
Comments:
          Submitted to the Neurips Workshop "System 2 Reasoning" in September, 2024. The openreview is avaliable at this https URL

- **What's New**: 최근 대형 언어 모델(LLMs)의 뛰어난 기능 덕분에 에이전트의 신속한 발전이 이루어지고 있습니다. 본 논문에서는 복잡한 문제 해결을 위한 새로운 접근법인 Tree-of-Code (ToC)를 제안하며, LLM 기반 에이전트의 행동을 통합하기 위한 효율적인 방법을 모색합니다. ToC는 Tree-of-Thought와 CodeAct의 이점을 결합하여 더욱 차별화된 솔루션 탐색을 지원합니다.

- **Technical Details**: ToC는 코드 실행 결과를 의사 결정 트리의 노드로 간주하고, 후보 솔루션 탐색에 폭넓은 탐색 전략을 활용하는 구조적 접근을 채택합니다. 이 프로세스는 ‘llm-function’이라는 새로운 요소를 도입하여 대형 모델이 결과 요약을 제공하도록 하여, 실행 결과에 대한 중간적인 성찰(reflection) 단계를 최소화하며, 코드와 실행 간의 조화를 강조합니다. 최종 결과는 성공적으로 실행된 노드의 다수결 투표에 의해 결정됩니다.

- **Performance Highlights**: ToC 접근법은 복잡한 작업의 실행 안정성을 높이며, 다양한 대형 언어 모델을 통합할 수 있는 가능성을 보여줍니다. 본 연구를 통해 실험적으로 문제 해결 성능이 향상되었음을 확인하였으며, 이는 코드가 논리적 구조를 내포하고 있기 때문에 발생하는 장점을 활용한 결과입니다. 최종적으로 ToC는 높은 품질의 결과를 도출하도록 설계되어 있어, LLM 기반 에이전트의 성능을 차별화할 수 있는 기반을 제공합니다.



### Integrating Evidence into the Design of XAI and AI-based Decision Support Systems: A Means-End Framework for End-users in Construction (https://arxiv.org/abs/2412.14209)
Comments:
          60 pages, 4 figures and 1 table

- **What's New**: 이 연구에서는 설명 가능한 인공지능(Explainable Artificial Intelligence, XAI) 도구의 신뢰성을 보장하기 위해 이론적 근거 기반의 수단-목적(framework) 체계를 개발했습니다. 특히 건설 분야에서 의사결정 지원 시스템의 설계를 위한 증거 기반 접근 방식을 채택하는 것이 중요하다고 강조합니다. 이 체계는 최종 사용자뿐만 아니라 이해관계자들도 효과적인 설명을 생성하는 데 활용할 수 있도록 설계되었습니다.

- **Technical Details**: 연구는 다양한 문헌을 기반으로 하여 수단-목적(framework) 체계를 구성하며, 의사결정 지원 시스템의 설계 및 개발 과정에 증거를 포함할 필요성을 강조합니다. 이러한 접근은 사용자들이 효과적으로 결정을 내릴 수 있도록 돕고, 그들의 인식적(epistemic) 목표를 달성할 수 있게 합니다. 연구의 결과는 설명 가능성을 갖춘 인공지능 및 의사결정 지원 시스템의 설계에 실질적인 기여를 할 수 있습니다.

- **Performance Highlights**: 이 연구는 건설 및 다른 엔지니어링 분야에서도 유용하게 적용될 수 있는 수단-목적(framework) 체계를 제안합니다. 특히, 설계 단계에서 증거의 강도, 가치 및 유용성을 평가하는 것이 최종 사용자에게 의미 있는 인간 설명을 제공하는 데 필수적임을 강조합니다. 이로 인해 의사결정 효율성이 향상되고, 사용자의 목표 달성에 기여할 수 있습니다.



### Large-scale Group Brainstorming using Conversational Swarm Intelligence (CSI) versus Traditional Cha (https://arxiv.org/abs/2412.14205)
- **What's New**: 이 연구는 Conversational Swarm Intelligence (CSI) 플랫폼인 Thinkscape를 사용하여 75명의 네트워크 사용자가 실시간으로 브레인스토밍과 우선 순위를 정하는 방법을 탐구합니다. 특히, 이 연구는 기존의 대규모 채팅방에서 브레인스토밍하는 경험과 CSI 구조를 통해 브레인스토밍하는 경험을 비교합니다. 이를 통해, CSI가 인간 집단 간의 협업을 어떻게 혁신할 수 있는지를 보여줍니다.

- **Technical Details**: CSI는 생물학적 원리인 Swarm Intelligence에 기반한 AI를 활용하는 방법으로, 수백 명의 참가자 간에도 실시간 대화를 가능하게 합니다. 이 연구에 사용된 Alternative Use Task (AUT) 개입의 변형을 통해, 참가자들이 CSI 구조를 사용했을 때 얼마나 긍정적인 경험을 했는지를 평가하고 비교합니다. 결과적으로, CSI는 크고 분산된 인구 집단을 작은 하위 그룹으로 나누고, 이들을 Conversational Surrogates라는 AI 에이전트로 연결합니다.

- **Performance Highlights**: Participating in the CSI structure led to significant improvements in collaboration, productivity, and answer quality. Participants reported feeling greater ownership of the final results and a stronger sense of being heard compared to traditional text chat environments. 이러한 결과는 CSI가 대규모 네트워크 인간 그룹 간의 브레인스토밍 및 우선 순위 설정을 위한 매우 유망한 방법임을 시사합니다.



### BlenderLLM: Training Large Language Models for Computer-Aided Design with Self-improvemen (https://arxiv.org/abs/2412.14203)
- **What's New**: 본 논문에서는 CAD 작업을 위해 LLMs를 훈련하는 새로운 프레임워크인 BlenderLLM을 제시합니다. 이 모델은 사용자 입력을 기반으로 CAD 스크립트를 자동 생성하는 데 활용되며, 이를 통해 기존 CAD 설계의 수동 작업을 크게 줄일 수 있습니다. 또한, 고유한 훈련 데이터 세트 BlendNet과 평가 도구 CADBench를 개발하여 LLM의 CAD 스크립트 생성 성능을 개선할 수 있는 기초를 마련했습니다.

- **Technical Details**: 이 논문의 신뢰할 수 있는 데이터 세트인 BlendNet은 8k 샘플을 포함하며, 자연어 지침을 bpy 스크립트로 매핑하는 다중 모듈 데이터 생성 파이프라인을 통해 생성되었습니다. BlenderLLM는 지도 학습(Supervised Fine-tuning) 및 반복적인 자기 개선(self-improvement) 프로세스를 통해 훈련되어 뛰어난 성과를 나타냅니다. CADBench는 사용자 제공 지침에서 CAD 스크립트를 생성하는 모델의 능력을 평가하기 위한 포괄적인 벤치마킹 프레임워크입니다.

- **Performance Highlights**: BlenderLLM은 CADBench의 여러 차원에서 모든 기준 모델을 초월하여 뛰어난 성과를 기록했습니다. 이 연구는 LLM을 CAD에 적용하는 데 강력한 기초를 마련함과 동시에 자기 개선 모델의 변혁적 가능성을 입증했습니다. 이 연구의 결과는 CAD 자동화 및 효율성 증진에 기여할 것으로 기대됩니다.



### Detecting Cognitive Impairment and Psychological Well-being among Older Adults Using Facial, Acoustic, Linguistic, and Cardiovascular Patterns Derived from Remote Conversations (https://arxiv.org/abs/2412.14194)
- **What's New**: 이 연구는 심리적 웰빙, 사회적 네트워크 및 인지 능력을 측정하기 위한 디지털 마커를 활용한 원거리 자동화된 평가 시스템을 도입하였습니다. 또한, 기존의 영상통화 플랫폼을 활용하여 인지 저하를 사전 평가할 수 있는 기회를 제공합니다. 연구팀은 기계 학습 모델을 사용하여 정상 인지 능력을 가진 개인 및 경도 인지 장애(MCI)를 가진 개인의 특성을 분석합니다.

- **Technical Details**: 이 연구에서는 얼굴, 음성, 언어 및 심혈관 특성을 분석하기 위해 39명의 피험자로부터 원거리 영상 대화 데이터를 활용하였습니다. 바탕 AI 모델(foundation AI models)을 통해 대규모 데이터셋으로부터 사전 훈련된 특성을 활용하여 인지 상태, 사회적 고립, 신경증 및 심리적 웰빙을 분석하였습니다. 연구 결과, 인지 저하 및 심리적 특성과 관련된 다양한 신호를 객관적으로 측정할 수 있는 가능성이 확인되었습니다.

- **Performance Highlights**: 모델은 Clinical Dementia Rating Scale에서 0.5의 점수를 0.78 AUC로 구별할 수 있었으며, 사회적 고립은 0.75 AUC, 신경증은 0.71 AUC, 부정 정서 척도는 0.79 AUC의 성능을 보였습니다. 이러한 결과는 원거리에서 인지 상태와 심리적 웰빙을 모니터링하는 데의 가능성을 보여줍니다. 특히, 언어 패턴이 인지 장애를 정량화하는 데 유용하며, facial expression 및 심혈관 패턴은 개인의 성격 및 심리적 웰빙을 측정하는 데 더 효과적임을 확인하였습니다.



### Whom do Explanations Serve? A Systematic Literature Survey of User Characteristics in Explainable Recommender Systems Evaluation (https://arxiv.org/abs/2412.14193)
Comments:
          31 pages, 2 figures. Submitted to ACM Transactions of Recommender Systems

- **What's New**: 이 연구는 추천 시스템의 설명에 대한 사용자의 특성이 어떻게 인식되는지를 분석하는 중요한 빈틈을 다룹니다. 총 124개의 논문을 조사하여 사용자 특성에 따른 설명의 효과를 평가한 결과, 대부분의 연구가 추천 시스템 사용자의 전형적인 특징을 잘 반영하지 못하고 있음을 확인했습니다. 이는 현재 연구에서 도출된 통찰이 일반화되는 데 제약이 될 수 있습니다.

- **Technical Details**: 추천 시스템의 설명을 개선하기 위해 사용자 특성을 평가하는 방법이 필요합니다. 연구에서는 텍스트 기반 설명의 품질을 평가하기 위해 BLEU-n, ROUGE-n 및 BERT-S와 같은 자연어 처리 지표를 사용하는 동시에 Recall 및 NDCG와 같은 랭킹-회수 기반 메트릭을 사용합니다. 그러나 현재의 오프라인 평가 메트릭은 사용자의 동기와 행동을 포착하지 못합니다.

- **Performance Highlights**: 설명 추가는 사용자 신뢰도를 향상시키고 의사결정을 개선하는 데 기여할 수 있습니다. 하지만, 다양한 사용자 특성을 고려한 연구가 부족하여 설명의 효과가 어떻게 달라지는지를 이해하는 데 한계가 있습니다. 향후 연구는 사용자 모집 및 데이터 보고의 일관성을 높여야 하며, 이러한 분야의 포괄적인 평가를 통해 추천 시스템의 설명 품질을 개선할 수 있는 기회를 제공해야 합니다.



### Ontology-Aware RAG for Improved Question-Answering in Cybersecurity Education (https://arxiv.org/abs/2412.14191)
- **What's New**: 이 논문에서는 CyberRAG라는 새로운 접근법을 제안하고 있습니다. CyberRAG는 ontology-aware retrieval-augmented generation (RAG) 방식을 적용하여 사이버 보안 교육을 위한 신뢰할 수 있는 QA 시스템을 개발하는 데 중점을 두고 있습니다. 이를 통해 AI 기반 질문-답변 시스템의 신뢰성을 향상시키고자 합니다.

- **Technical Details**: CyberRAG는 두 단계로 구성됩니다. 첫 번째 단계에서는 지식 기반에서 검증된 사이버 보안 문서를 검색하여 도메인 특정 지식을 보강하고, 두 번째 단계에서는 지식 그래프 온톨로지를 통합하여 답변의 신뢰성을 검증합니다. 이러한 방식으로 사이버 보안 문제 해결에서의 불확실성을 효과적으로 관리할 수 있습니다.

- **Performance Highlights**: 사이버 보안 데이터셋을 사용한 실험 결과, CyberRAG는 도메인 지식에 부합하는 정확하고 신뢰할 수 있는 응답을 제공합니다. 이는 AI 도구가 교육을 개선하는 데 있어 잠재력을 가지고 있음을 보여줍니다. 또한, AI 기반 시스템의 일반적인 문제인 hallucinations를 완화하고 사용 오해를 줄이는 데 기여합니다.



### Lessons From an App Update at Replika AI: Identity Discontinuity in Human-AI Relationships (https://arxiv.org/abs/2412.14190)
- **What's New**: 이 연구는 소비자가 AI와 깊은 감정적 유대를 형성하고 AI 정체성에 투자할 수 있는지에 대한 질문을 탐구합니다. 미국의 인기 있는 AI 동반자인 Replika AI의 앱 업데이트 이벤트를 활용하여, AI의 정체성에 대한 고객의 인식 변화가 소비자 복지에 미치는 영향을 살펴보았습니다. 특히, 앱이 이전에 가능했던 친밀한 상호작용을 제거한 후, 소비자들이 느끼는 AI 동반자의 정체성의 단절이 예측된다는 사실을 발견했습니다.

- **Technical Details**: 연구에서 ERP( erotic role play) 기능 제거 후, 고객은 AI 동반자와의 관계에서 상실감을 경험하게 되었으며, 이는 고객이 새로운 AI를 원래 AI와 비교하여 저평가하는 결과로 이어졌습니다. 실험 결과는 이러한 발견을 검증했으며, 사용자가 AI 동반자와 더 가까운 관계를 느끼고 있다는 점도 강조되었습니다. 이는 그들의 가장 친한 인간 친구보다도 깊은 유대관계를 나타냅니다.

- **Performance Highlights**: AI 동반자의 상실에 대한 애도를 느끼는 정도가 다양한 비인간 물체의 상실에 비해 더 크다는 것이 밝혀졌습니다. 이 연구의 결과는 소비자와 AI 사이의 관계가 진정한 인간 수준의 관계를 형성하고 있으며, 이러한 관계의 중단이 실제적인 애도 패턴과 평가절하를 유발한다는 점을 보여줍니다. 소비자와 기업에게 독특한 이점과 위험을 만들어내는 개인적인 관계임을 알립니다.



### CogSimulator: A Model for Simulating User Cognition & Behavior with Minimal Data for Tailored Cognitive Enhancemen (https://arxiv.org/abs/2412.14188)
- **What's New**: 이 논문은 사용자 인지(cognition)를 소규모 환경에서 시뮬레이션하기 위해 CogSimulator라는 새로운 알고리즘을 제시합니다. 특히, 게임 Wordle을 사례로 하여 최소한의 데이터로도 사용자 반응을 예측할 수 있는 방식이 강조됩니다. 이는 Wasserstein-1 distance와 하이퍼파라미터 튜닝을 위한 최적화 기법을 사용하여 특정 사용자에 맞춘 교육 게임을 제공할 수 있도록 돕습니다.

- **Technical Details**: CogSimulator는 일부 데이터만을 사용하여 사용자의 인지 수준을 포착하고 시뮬레이션하는 모델로, 관련 게임의 교육적 주제를 더 잘 반영할 수 있습니다. 이 기술은 특히 적은 양의 데이터를 이용해 새로운 게임 시나리오에서의 예측을 가능하게 하여, 게임 디자이너들이 사용자 인지 프로파일에 맞는 게임 난이도를 조정하는 데 도움을 줍니다.

- **Performance Highlights**: 비교 실험 결과, CogSimulator는 평균 Wasserstein-1 distance, 평균 제곱 오차(mean squared error), 그리고 평균 정확도(mean accuracy) 면에서 기존의 대부분의 기계 학습 모델보다 뛰어난 성능을 보였습니다. 이는 인지 능력을 향상시키기 위한 맞춤형 게임 디자인의 효능을 선명하게 보여줍니다.



### Towards AI-$45^{\circ}$ Law: A Roadmap to Trustworthy AGI (https://arxiv.org/abs/2412.14186)
Comments:
          First submit, Preview Only

- **What's New**: 이 논문에서는 신뢰할 수 있는 인공지능 일반 지능(AGI)을 위한 가이드라인을 제안하며, 'AI-45도 법칙'이라는 원리를 소개합니다. 이는 인공지능의 능력과 안전이 동등하게 발전해야 한다는 것을 강조합니다. 또한 'Causal Ladder of Trustworthy AGI' 프레임워크를 통해 현재의 인공지능 안전성과 능력 연구를 체계적으로 정리할 수 있는 구조를 제공합니다.

- **Technical Details**: 이 연구에서는 AGI의 핵심 신뢰성 요구 사항을 충족하는 세 가지 레이어로 구성된 기술 프레임워크를 제안합니다: Approximate Alignment Layer, Intervenable Layer, Reflectable Layer. 첫 번째 레이어는 인간의 가치와의 근사적 정렬에 초점을 맞추며, 인공지능 모델의 가치 표현 및 정렬 능력의 기술적 발전이 필요합니다. 이러한 진전을 통해 모델이 인간의 지시를 이해하고 따를 수 있도록 하는 것이 중요합니다.

- **Performance Highlights**: 이 논문은 신뢰할 수 있는 AGI의 다섯 가지 신뢰성 수준—지각, 추론, 의사결정, 자율성, 협력 신뢰성—을 정의합니다. 또한, AGI 개발에서 윤리적이고 안전한 접근 방식을 보장하기 위해 필요한 거버넌스 조치를 제안합니다. 이러한 접근 방식은 AGI 시스템의 안전성과 신뢰성을 높이는 데 기여할 것으로 기대됩니다.



### Benchmarking Harmonized Tariff Schedule Classification Models (https://arxiv.org/abs/2412.14179)
- **What's New**: 이번 연구는 HTS(에스티시) 분류 도구에 대한 표준화된 성과 평가 프레임워크를 구축하여, 다양한 클래시피케이션(분류) 도구의 성능을 체계적으로 비교합니다. 이 프레임워크는 속도, 정확도, 합리성, HTS 코드 정렬 등을 평가하여, 업계에서의 개선 및 혁신 기회를 식별합니다.

- **Technical Details**: 우선, 100개의 세관 및 국경 보호국(CBP) 판결을 무작위로 선택하여 데이터셋을 수집했습니다. 각 사례는 제품 이름, 상세 설명, 최종 HTS 코드 등의 데이터를 포함하여, 분류 모델 성능 평가의 기준으로 사용됩니다. 대규모 언어 모델(gpt-4o)이 사용되어 다양한 제품 분류가 가능하도록 했습니다.

- **Performance Highlights**: Zonos 및 WCO BACUDA 모델은 분류의 정확도와 속도에서 우수한 성능을 보이며, 특히 Zonos는 즉각적인 결과를 제공합니다. 그러나 연구에서는 모든 도구의 강점과 한계를 식별하여 향후 HTS 분류 기술의 발전과 산업 전반의 향상을 위한 기초 자료를 제공합니다.



### A Medical Low-Back Pain Physical Rehabilitation Dataset for Human Body Movement Analysis (https://arxiv.org/abs/2407.00521)
- **What's New**: 이 논문은 저자들이 개발한 제안된 의학 데이터셋 'Keraal'을 소개합니다. 이 데이터셋은 요통 재활 운동을 수행하는 임상 환자의 데이터를 포함하며, 3D Kinect 스켈레톤 위치 및 방향, RGB 비디오, 2D 스켈레톤 데이터, 의료 주석을 포함하고 있습니다. 이를 통해 운동 수행의 정확성과 오류 분류, 신체 부위와 시점의 특정을 평가할 수 있습니다.

- **Technical Details**: 이 논문은 저자가 제안한 데이터셋을 기반으로 지능형 튜터링 시스템이 재활 세션을 자동으로 감독할 수 있도록 하는 것을 목표로 하고 있습니다. 연구에서 3D 신체 운동을 분석하기 위해 Gaussian Mixture Model (GMM)과 Long-Short Term Memory (LSTM) 알고리즘을 사용하여 성능을 평가하였습니다. 데이터셋은 저자의 임상 시험을 통해 수집된 31명의 환자로부터 얻어진 것으로, 오류 인식, 공간적 및 시간적 오류 로컬라이제이션을 해결하는데 있어서 나타나는 도전 과제를 다룹니다.

- **Performance Highlights**: 제안된 Keraal 데이터셋은 다양한 환자를 대상으로 장기 재활 프로그램 동안 수집된 유일한 벤치마킹 세트로, 임상의가 제공한 레이블을 포함하고 있습니다. 두 가지 기본 운동 인식 알고리즘을 통해 평가된 결과는 이 데이터셋이 재활 치료에서의 정확한 운동 분석을 지원할 수 있는 잠재력을 갖고 있음을 보여줍니다. 특히, 데이터셋은 저렴하고 사용하기 쉬운 센서를 통해 수집되었으며, 이는 임상 환경에서 치료 효과성을 높일 수 있는 기회를 제공합니다.



### Goal Space Abstraction in Hierarchical Reinforcement Learning via Set-Based Reachability Analysis (https://arxiv.org/abs/2309.07675)
- **What's New**: 이번 연구에서는 상징적 목표 표현(symbolic goal representation)을 통한 목표 발견을 위한 발전 메커니즘을 제안합니다. 기존의 Hierarchical Reinforcement Learning(HRL) 방법들은 수동적인 목표 표현에 의존하여 제한적인 경우가 많았지만, 이 방법은 자동으로 목표 표현을 발견할 수 있는 가능성을 탐구합니다. 연구에서는 유사한 역할을 가진 상태 집합을 추상화하여 구성하는 새로운 방법을 소개합니다.

- **Technical Details**: 제안된 알고리즘은 Feudal HRL을 기반으로 하여 목표 표현과 계층 정책(hierarchical policy)을 동시에 학습하는 구조를 가지고 있습니다. 이 과정에서 신경망을 이용한 상징적 도달 가능성 분석(symbolic reachability analysis)을 활용하여 상태 집합 간의 전이 관계(transition relation)를 근사하고 목표 표현을 정제합니다. 이를 통해 목표 발견 과정의 효율성을 높이고, 더 복잡한 환경에서도 적용 가능성을 가집니다.

- **Performance Highlights**: 이 알고리즘은 복잡한 내비게이션(task) 작업에서 평가되었으며, 학습된 목표 표현은 이해 가능하고(interpretable), 전달 가능하며(transferrable), 데이터 효율적인 학습(data efficient learning) 결과를 보여주었습니다. 실험 결과는 제안된 방법이 기존의 방법들보다 더 나은 성능을 발휘할 수 있음을 나타냅니다.




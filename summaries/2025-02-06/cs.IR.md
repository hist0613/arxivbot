New uploads on arXiv(cs.CL)

### On Fairness of Unified Multimodal Large Language Model for Image Generation (https://arxiv.org/abs/2502.03429)
- **What's New**: 이번 연구에서 제시된 통합 다중 모드 대형 언어 모델(U-MLLM)은 비주얼 이해와 생성에서 놀라운 성능을 발휘하고 있습니다. 그러나 이 모델들은 생성 모델만을 위한 구조와는 달리, 성별과 인종 차별 같은 편향(bias)을 포함할 가능성이 높아 새로운 문제를 제기합니다. 우리가 제안하는 locate-then-fix 전략은 이러한 편향의 출처를 감사(audit)하고 이해하는 데 중점을 둡니다.

- **Technical Details**: U-MLLM은 이미지와 텍스트를 동시에 처리할 수 있는 구조로 되어 있으며, 이를 위해 이미지 토크나이저(image tokenizer)와 언어 모델(language model)로 구성됩니다. 본 연구에서는 VILA-U와 같은 최신 U-MLLM의 다양한 구성 요소를 분석하여 그들이 보여주는 성별 및 인종 편향의 원인을 찾아냈습니다. 구체적으로, 우리는 언어 모델에서의 편향이 주된 원인임을 발견하였습니다.

- **Performance Highlights**: 실험을 통해 우리의 접근 방식이 성별 및 인종 편향을 71.91% 감소시키는 동시에 이미지 생성의 품질을 유지하는 데 성공했음을 보여주었습니다. 예를 들어, VILA-U 모델은 초기 점수(inception score)를 12.2% 증가시켰습니다. 이러한 개선 사항은 향후 통합 MLLM의 개발에서 더욱 공정한 해석과 디바이싱 전략이 필요함을 강조합니다.



### Think or Step-by-Step? UnZIPping the Black Box in Zero-Shot Prompts (https://arxiv.org/abs/2502.03418)
Comments:
          8 pages (excluding references)

- **What's New**: 본 논문에서는 제로샷 프롬프트(zero-shot prompting) 기법의 효과성을 분석할 새로운 메트릭, ZIP 점수(Zero-shot Importance of Perturbation score)를 소개합니다. ZIP 점수는 개방형(open-source) 및 폐쇄형(closed-source) 모델에 적용 가능하며, 입력 단어의 변화에 따른 모델 성능 차이를 통해 단어의 중요성을 정량적으로 평가합니다. 기존의 그래디언트 기반(gradient-based) 및 어텐션 기반(attention-based) 해석 방법과는 달리, ZIP 점수는 내부 모델 파라미터에 접근하지 않고도 적용할 수 있습니다.

- **Technical Details**: ZIP 점수는 주어진 프롬프트에서 개인 단어가 모델 성능에 미치는 영향을 측정하여 단어의 중요성을 정량화합니다. 이 방법은 문맥에 적절한 동의어 치환과 공통 하의어(co-hyponym) 치환을 통해 가변성을 부여하며, 각 단어의 변형을 통해 필요성을 테스트합니다. 이를 통해 모델의 소명에 있어 가장 중요한 단어를 체계적으로 파악할 수 있습니다.

- **Performance Highlights**: 네 가지 최신 LLM(대형 언어 모델)과 7개의 널리 사용되는 프롬프트를 실험한 결과, 단어의 중요성을 발견하는 흥미로운 패턴이 나타났습니다. 특히, 'step-by-step'와 'think'는 모두 높은 ZIP 점수를 보였으나, 어느 것이 더 많은 영향을 미치는지는 모델과 태스크에 따라 달라지는 경향을 보였습니다. 이 연구는 LLM 행동에 대한 새로운 통찰을 제공하며, 보다 효과적인 제로샷 프롬프트 개발 및 모델 분석에 기여할 수 있습니다.



### SPRI: Aligning Large Language Models with Context-Situated Principles (https://arxiv.org/abs/2502.03397)
- **What's New**: 본 논문에서는 'Situated-PRInciples' (SPRI)라는 새로운 프레임워크를 제안하여, 최소한의 인간의 개입으로 실시간으로 각 입력 쿼리에 맞는 가이드 원칙을 자동 생성하는 방법을 다룬다. 이는 기존의 정적인 원칙이 아닌, 특정 상황에 맞춘 맞춤형 원칙을 통해 LLM의 결과를 정렬할 수 있도록 돕는다.

- **Technical Details**: SPRI는 두 단계로 구성된 알고리즘을 통해 작동한다. 첫 번째 단계에서는 기본 모델이 원칙을 생성하고, 비평 모델이 이를 반복적으로 개선하는 방식이다. 두 번째 단계에서는 생성된 원칙을 사용하여 기본 모델의 응답을 특정 사용자 입력에 맞게 조정하며, 비평 모델이 이러한 최종 응답을 평가하고 피드백을 제공한다.

- **Performance Highlights**: SPRI는 세 가지 상황에서 평가되었으며, 그 결과 복잡한 도메인 특정 작업에서도 전문가가 만든 원칙과 동등한 성능을 달성했다. 또한, SPRI가 생성한 원칙을 이용하면 LLM-judge 프레임워크에 비해 현저히 향상된 평가 기준을 제공하며, 합성 SFT 데이터 생성을 통해 진실성에서 유의미한 개선을 이끌어낼 수 있었다.



### LIMO: Less is More for Reasoning (https://arxiv.org/abs/2502.03387)
Comments:
          17 pages

- **What's New**: LIMO 모델이 제안되어 복잡한 추론 능력을 매우 적은 훈련 예시로 이끌어낼 수 있음을 보여줍니다. 기존에는 복잡한 추론 작업에 많은 양의 훈련 데이터가 필요하다고 여겨졌으나, LIMO는 단 817개의 예제로 AIME에서 57.1%, MATH에서 94.8%의 정확도를 달성했습니다. 이 결과는 신뢰할 수 있는 데이터 효율성을 보여주며, AGI(Artificial General Intelligence)에 대한 가능성을 제시합니다.

- **Technical Details**: LIMO 모델은 사전 훈련(pre-training) 시 학습된 방대한 양의 수학적 지식을 기반으로 하며, 고도로 구조화된 인지적 과정(cognitive processes)을 통해 최소한의 데모를 사용하여 복잡한 작업을 해결할 수 있습니다. 이 모델은 100배 더 많은 데이터를 사용한 전통적인 SFT(supervised fine-tuning) 모델들을 40.5% 절대 개선하여 성능을 초과 달성했습니다. 또한, 모델의 지식 기반을 효과적으로 활용할 수 있는 인지적 템플릿(cognitive templates)의 역할도 강조됩니다.

- **Performance Highlights**: LIMO는 단 1%의 훈련 데이터를 사용하여 AIME 벤치마크에서 57.1%, MATH에서 94.8%의 정확도를 기록하며 새로운 기준을 세웁니다. 이 발견은 인공지능 연구에 깊은 의미를 가지며, 최소한의 훈련 샘플로도 복잡한 추론 능력을 이끌어낼 수 있음을 시사합니다. 더욱이 LIMO는 10개의 다양한 벤치마크에서 40.5%의 성능 개선을 보여줌으로써 모델의 일반화 능력을 입증하고, 기존의 데이터 집약적 접근 방식의 필요성을 재고하게 만듭니다.



### High-Fidelity Simultaneous Speech-To-Speech Translation (https://arxiv.org/abs/2502.03382)
- **What's New**: Hibiki는 동시 음성 번역을 위한 디코더 전용 모델로, 소스와 타겟 음성을 동시에 처리하는 멀티스트림 언어 모델을 활용합니다. 이 모델은 텍스트와 오디오 토큰을 공동으로 생성하여 음성을 텍스트로 번역하고 음성에서 음성으로 번역하는 작업을 수행합니다. 또한, 이 연구는 실시간 번역을 위해 맥락을 축적할 충분한 정보를 동적으로 조정하는 방법을 제안합니다.

- **Technical Details**: Hibiki는 전 세계적으로 동시 다중 스트림 아키텍처를 활용하여 소스 음성을 수신하고 번역된 음성을 생성하는 디코더 전용 모델입니다. 이 모델은 온도 샘플링과 인과적 오디오 코덱을 결합하여 실시간으로 입력 및 출력을 처리합니다. 훈련 과정에서는 GPT와 같은 기계 번역 시스템의 혼란도를 활용하여 각 단어의 최적 지연을 식별하고, 실시간 흐름을 적응시키기 위해 적절한 침묵을 도입합니다.

- **Performance Highlights**: 프랑스-영어 동시 음성 번역 작업에서 Hibiki는 번역 품질, 화자 유사성 및 자연스러움에서 최신 성능을 입증했습니다. 이 모델은 GPU를 사용하여 실시간으로 수백 개의 시퀀스를 번역할 수 있으며, 스마트폰 상에서도 실시간 실행이 가능합니다. 인간 평가 결과, Hibiki는 인간 통역사에 가까운 경험을 제공하는 첫 번째 모델임을 보여주었습니다.



### Integrating automatic speech recognition into remote healthcare interpreting: A pilot study of its impact on interpreting quality (https://arxiv.org/abs/2502.03381)
Comments:
          to appear in the Proceedings of Translation and the Computer (TC46)

- **What's New**: 본 논문은 원격 의료 통역에서 자동 음성 인식(ASR) 기술이 통역 품질에 미치는 영향을 조사한 파일럿 연구 결과를 보고합니다. 이 연구는 네 가지 무작위 조건을 적용한 피험자 내 실험 설계를 기반으로 하며, 스크립트화된 의학 상담을 활용하여 대화 통역 작업을 시뮬레이션했습니다. 연구 결과 ASR 전사 및 ChatGPT가 생성한 요약의 이용이 통역 품질을 효과적으로 향상시켰음을 보여줍니다.

- **Technical Details**: 연구는 네 명의 중국어 및 영어를 사용하는 통역 훈련생과 함께 진행되었으며, 참여자들은 시점적 회상 보고서(cued retrospective reports) 및 반구조적 인터뷰(semi-structured interviews)를 통해 ASR 지원에 대한 경험과 인식을 수집했습니다. 초기 데이터는 다양한 형태의 ASR 출력이 통역 오류 유형의 분포에 미치는 영향을 다르게 나타냈으며, 참여자들은 ASR 전사를 선호했습니다.

- **Performance Highlights**: 이 파일럿 연구는 ASR 기술이 대화 기반 의료 통역에 적용될 가능성을 보여주며, 통역 경험과 성과를 향상시키기 위한 ASR 출력의 최적 제시 방법에 대한 통찰을 제공합니다. 그러나 이 연구의 주요 목적은 방법론을 검증하는 것이며, 이러한 발견을 확정하기 위해서는 더 큰 표본 크기를 가진 추가 연구가 필요할 것입니다.



### Demystifying Long Chain-of-Thought Reasoning in LLMs (https://arxiv.org/abs/2502.03373)
Comments:
          Preprint, under review

- **What's New**: 이 논문은 대규모 언어 모델(LLMs)에서 긴 사고의 연쇄(long chains-of-thought, CoTs) 생성 메커니즘을 체계적으로 조사하여 모델이 긴 CoT 궤적을 생성하는데 필요한 주요 요소들을 식별합니다. 주요 발견으로는 감독 세부 조정(supervised fine-tuning, SFT)과 강화 학습(reinforcement learning, RL)에 대한 연구 결과를 통해 훈련의 효율성을 개선할 수 있다는 점을 강조합니다. 이러한 연구는 LLM의 긴 CoT 추론 능력을 최적화하기 위한 실용적인 지침을 제공합니다.

- **Technical Details**: 긴 CoT 이론적 모델을 제시하며, 모델의 상태를 정의하고 SFT 및 RL 방법 전반에 걸친 개요를 포함합니다. 모델의 매개변수화와 함께, 긴 CoT를 생성하는 데 주어진 보상 신호를 필수적으로 다룹니다. 또한, 효과적인 RL 기반 인센티브 디자인을 요구하는 긴 CoT의 기원 및 도전 과제를 다룹니다.

- **Performance Highlights**: 연구에서는 SFT가 필수적이지는 않지만 훈련의 단순화와 효율성을 증가시키는 것이 중요한 발견으로 나타났습니다. 또한, RL을 통해 CoT 길이와 복잡성을 안정적으로 늘리기 위해 코사인 길이 조절 보상 및 반복 패널티를 도입하는 것이 필요하다는 점을 강조합니다. 마지막으로, 웹에서 추출된 노이즈가 포함된 솔루션을 활용한 강화 학습의 가능성을 보여줍니다.



### Minerva: A Programmable Memory Test Benchmark for Language Models (https://arxiv.org/abs/2502.03358)
- **What's New**: 이 논문에서는 LLM 기반 AI 어시스턴트가 메모리(컨텍스트)를 효율적으로 활용하여 작업을 수행할 수 있는 능력을 평가하기 위한 새로운 프레임워크를 소개합니다. 기존의 데이터 벤치마크는 정적이며 해석이 어렵고 모델의 특정 능력을 파악하는 데 한계가 있어, 논문은 메모리 관련 다양한 테스트를 자동으로 생성할 수 있는 방법을 제시합니다. 특히, 우리는 단순 검색 작업을 넘어서 원자적 작업과 복합 작업을 평가하여 모델의 메모리 사용 능력을 평가하려고 합니다.

- **Technical Details**: 이 논문에서는 메모리 사용 능력(memory-usage capabilities)을 정보 검색, 합성, 기억 회상 등의 기본 원자적 작업으로 정의합니다. 각 테스트는 원자적 작용을 격리하여 특정 능력을 평가하도록 설계되었으며, 더 복잡한 실제 시나리오를 반영하는 복합 테스트도 포함되어 있습니다. 벤치마크는 입력 데이터를 기반으로 고유하게 프로그래밍 가능한 스크립트를 활용하여 신선하고 무작위 테스트 케이스를 생성할 수 있게 설계되었습니다.

- **Performance Highlights**: 실험 결과, 모델들은 간단한 검색 작업에서는 상대적으로 좋은 성능을 보였지만, 맥락 활용 능력은 모델 간에 상당한 차이가 있음을 보여주었습니다. 특히 복합 테스트에서 모든 모델에서 성능 저하가 나타났으며, 이는 기존 모델들의 한계를 드러내고 향후 모델 훈련 및 개발을 위한 귀중한 통찰력을 제공합니다. 논문에서 제안한 프레임워크는 모델의 메모리 관련 다양한 능력에 대한 보다 구체적인 평가를 가능하게 합니다.



### ECM: A Unified Electronic Circuit Model for Explaining the Emergence of In-Context Learning and Chain-of-Thought in Large Language Mod (https://arxiv.org/abs/2502.03325)
Comments:
          Manuscript

- **What's New**: 이 논문은 Electronic Circuit Model (ECM)을 통해 인맥 학습(In-Context Learning, ICL)과 사고 과정(chain-of-Thought, CoT)의 결합된 영향을 규명하고, 이러한 이해를 통해 LLM의 성능을 개선하고 최적화하는 새로운 관점을 제시합니다. ECM은 LLM의 동작을 전자 회로로 개념화하여 ICL은 자석 자기장으로, CoT는 일련의 저항기로 모델 성능을 설명합니다. 또한, 실험 결과 ECM이 다양한 프롬프트 전략에 대해 LLM 성능을 성공적으로 예측하고 설명하는 것을 보여주었습니다.

- **Technical Details**: ECM에 따르면 모델 성능은 모델의 내재적 능력을 나타내는 기초 전압, 추론의 어려움, ICL에서 추가된 전압 등 다양한 요소에 의해 영향을 받습니다. ICL은 자석 자기장의 감소를 통해 추가 전압을 발생시키고, CoT는 여러 저항기를 사용해 각 추론 과정의 어려움을 더합니다. 이러한 관점은 ICL과 CoT의 복합적인 영향을 수치적으로 이해할 수 있는 기반을 제공합니다.

- **Performance Highlights**: ECM 활용을 통한 실험에서 LLM은 국제 정보 올림피아드(International Olympiad in Informatics, IOI) 및 국제 수학 올림피아드(International Mathematical Olympiad, IMO)와 같은 복잡한 작업에서 80% 이상의 인간 경쟁자를 초월하는 성과를 달성했습니다. 이로 인해 ECM이 LLM의 성능 향상에 기여했음을 보여주며, 이는 학술 연구 개발과 같은 탐색적 맥락에서도 최소 10% 이상의 성과 향상을 이루어냈습니다.



### Out-of-Distribution Detection using Synthetic Data Generation (https://arxiv.org/abs/2502.03323)
- **What's New**: 이번 연구에서는 OOD(Out-Of-Distribution) 탐지를 위한 혁신적인 접근 방식을 제안합니다. 특히, 대규모 언어 모델(LLM)의 생성 능력을 활용하여 외부 OOD 데이터 없이 고품질의 합성 OOD 프로시를 생성할 수 있습니다. 이를 통해 기존의 데이터 수집 문제를 해결하고, 다양한 ML(기계 학습) 작업에 적용 가능한 고유의 합성 샘플을 제공합니다.

- **Technical Details**: 이 연구에서는 LLM이 생성한 합성 데이터를 기반으로 OOD 탐지기를 훈련하는 새로운 방법을 개발했습니다. LLM의 정교한 프롬프트를 통해 가능한 분포 이동을 모방하는 합성 샘플을 생성하고, 이를 통해 신뢰성 있는 OOD 탐지를 구현합니다. 연구의 결과는 LLM이 텍스트 분류 작업에 크게 기여할 수 있음을 보여줍니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존 방법들에 비해 가짜 양성률(false positive rates)을 크게 감소시켰으며, 몇몇 경우에서 완벽한 제로를 달성했습니다. 또한, InD 데이터 및 합성 OOD 프로시를 사용해 높은 정확도를 유지하며, 다양한 모델 크기에서 좋은 성능을 보였습니다. 이 방법은 텍스트 분류 시스템의 신뢰성과 안전성을 크게 향상시킬 잠재력을 지니고 있습니다.



### MeDiSumQA: Patient-Oriented Question-Answer Generation from Discharge Letters (https://arxiv.org/abs/2502.03298)
- **What's New**: 이 논문에서는 환자 접근성을 높이기 위한 도구로서 MeDiSumQA라는 새로운 데이터셋을 소개합니다. MeDiSumQA는 MIMIC-IV의 퇴원 요약에서 자동화된 질문-답변 생성 프로세스를 통해 만들어졌습니다. 이 데이터셋은 환자 중심의 질문-답변(QA) 형식을 채택하여 건강 문서 이해도를 높이고 의료 결과를 개선하는 데 기여하고자 합니다.

- **Technical Details**: MeDiSumQA는 퇴원 요약을 기반으로 하여 생성된 고품질 질문-답변 쌍을 포함하고 있습니다. 이 데이터셋은 메디컬 LLM을 평가하기 위해 사용되며, 질문-답변 쌍의 생성 과정에서 Meta의 Llama-3 모델과 의사의 수작업 검토를 결합하였습니다. 질문-답변 페어는 증상, 진단, 치료 같은 6개의 QA 카테고리로 분류됩니다.

- **Performance Highlights**: 연구 결과, 일반적인 LLM이 생물 의학적으로 조정된 모델보다 우수한 성능을 보였습니다. 자동 평가 메트릭은 인간의 평가와 상관관계가 있음을 보여주며, MeDiSumQA를 통해 향후 LLM의 발전을 지원하고 임상 문서의 환자 친화적 이해를 증진시키겠다는 목표를 가지고 있습니다.



### ALPET: Active Few-shot Learning for Citation Worthiness Detection in Low-Resource Wikipedia Languages (https://arxiv.org/abs/2502.03292)
Comments:
          24 pages, 8 figures, 4 tables

- **What's New**: 이번 연구에서는 Citation Worthiness Detection (CWD)를 위한 ALPET 프레임워크를 도입했습니다. ALPET는 Active Learning (AL)과 Pattern-Exploiting Training (PET)을 결합하여 데이터 자원이 제한된 언어에서 CWD를 향상시키는 데 중점을 두고 있습니다. 특히, 카탈란어, 바스크어 및 알바니아어 위키백과 데이터셋에서 기존의 CCW 기준선보다 뛰어난 성능을 보이며, 필요한 레이블 데이터 양을 80\% 이상 줄일 수 있었습니다.

- **Technical Details**: ALPET는 300개의 레이블 샘플 이후 성능이 안정화되는 경향이 있어, 대규모 레이블 데이터셋이 흔하지 않은 저자원 환경에서의 적합성을 강조합니다. 특정한 Active Learning 쿼리 전략, 예를 들어 K-Means 클러스터링을 사용하는 방법은 장점을 제공할 수 있지만, 그 효율성은 항상 보장되지 않으며 작은 데이터셋에서는 무작위 샘플링에 비해 소폭의 개선에 그치는 경향이 있습니다. 이는 단순한 무작위 샘플링이 자원이 제한된 환경에서도 여전히 강력한 기준선 역할을 할 수 있음을 시사합니다.

- **Performance Highlights**: 결론적으로, ALPET은 레이블 샘플이 적더라도 높은 성능을 달성할 수 있는 능력 덕분에 저자원 언어 설정에서 온라인 콘텐츠의 검증 가능성을 높이는 데 유망한 도구가 될 것으로 평가됩니다. 연구 결과는 저자원 환경에서도 언어적 다양성을 지켜내며, 정보의 신뢰성을 향상시키기 위한 효율적인 방법론을 제시하고 있습니다.



### Token Assorted: Mixing Latent and Text Tokens for Improved Language Model Reasoning (https://arxiv.org/abs/2502.03275)
- **What's New**: 이번 연구에서는 Large Language Models (LLMs)가 chain-of-thought (CoT) 데이터를 통해 추론 능력을 극대화할 수 있음을 보여주며, 초기 추론 단계를 부분적으로 추상화한 혼합 표현(hybrid representation)을 제안합니다. VQ-VAE를 통해 생성된 잠재 이산 토큰(latent discrete tokens)을 사용하여 추론 흔적(reasoning traces)의 길이를 크게 줄이는데 성공했습니다. 이 연구는 Keys-Finding Maze 문제의 모델을 처음부터 훈련시키고, 보이지 않는 잠재 토큰(latent tokens)을 포함한 확장된 어휘로 LLM을 미세 조정하는 두 가지 시나리오를 탐구합니다.

- **Technical Details**: 제안된 방법론은 추론 과정의 초기 단계에서 불필요한 정보를 잠재 토큰으로 대체하여, 모델이 텍스트 토큰과 혼합된 새로운 토큰에 대해 신속하게 적응할 수 있도록 합니다. 훈련 과정 동안 텍스트 토큰의 수를 무작위로 변경하여 잠재 토큰으로 교체하는 전략을 채택합니다. 이를 통해, 모델은 추상적인 사고 과정의 표현과 세부적인 텍스트 설명 모두에서 학습할 수 있는 기회를 가집니다.

- **Performance Highlights**: 모델의 성능은 Keys-Finding Maze와 ProntoQA, ProsQA와 같은 다양한 벤치마크에서 평가되었으며, 텍스트 전용 추론 흔적을 사용하여 훈련 된 기준 모델을 지속적으로 능가하는 결과를 보였습니다. 다양한 수학적 추론 벤치마크(GSM8K, Math, OlympiadBench-Math)에서도 제안된 방법이 효과적으로 성능을 향상시켰습니다. 전체적으로, 여러 작업과 모델 아키텍처에서 제안된 혼합 표현 접근법이 우수한 결과를 달성했습니다.



### Efficient extraction of medication information from clinical notes: an evaluation in two languages (https://arxiv.org/abs/2502.03257)
Comments:
          Submitted to JAMIA, 17 pages, 3 figures, 2 tables and 5 supplementary tables

- **What's New**: 본 연구에서는 임상 텍스트에서 약물 정보를 추출하기 위한 새로운 자연어 처리(NLP) 방법을 제안하였습니다. 제안된 방법은 transformer 기반의 아키텍처를 사용하여 프랑스어 및 영어 임상 문서에서 효과적인 엔터티와 관계 추출을 목표로 합니다. 기존의 방법에 비해 컴퓨팅 비용을 10% 줄이면서도 높은 성능을 달성하는 점이 특징입니다.

- **Technical Details**: 연구에서 제안한 모델은 프랑스 병원 데이터와 영어 n2c2 데이터셋을 사용하여 훈련되었습니다. 약물과 관련된 정보의 추출을 위해 액자(frame) 개념을 도입하여 약물의 속성과 이력을 보다 정교하게 나타냅니다. 이 모델은 Named Entity Recognition(NER)과 Relation Extraction(RE) 작업 모두를 포함하는 end-to-end 솔루션을 제공합니다.

- **Performance Highlights**: 제안된 아키텍처는 프랑스어 및 영어 코퍼스에서 각각 F1 점수 0.69와 0.82를 기록하여 현재의 최첨단 모델과 경쟁하는 성능을 달성했습니다. 임상 텍스트 처리에서 낮은 컴퓨팅 비용으로 높은 성능을 유지하며, 일반적인 병원 IT 자원에 적합한 특징이 있습니다.



### How do Humans and Language Models Reason About Creativity? A Comparative Analysis (https://arxiv.org/abs/2502.03253)
Comments:
          CogSci 2025

- **What's New**: 이번 연구는 STEM(과학, 기술, 공학 및 수학) 분야에서의 창의성 평가에 대한 인간 전문가와 AI의 접근 방식을 비교했으며, 특히 예시를 제공할 때의 창의성 평가가 어떠한 영향을 미치는지를 분석했습니다. 연구 결과, 예시가 제공된 경우 LLM(대형 언어 모델)의 창의성 점수가 실제 점수 예측에서 크게 향상됨을 발견했습니다. 또한, 전문가와 LLM의 평가 패턴 간의 차이점과 일치점을 파악하기 위한 중요한 실험이 진행되었습니다.

- **Technical Details**: 연구는 두 가지 주요 실험을 포함하며, 첫 번째 실험에서는 72명의 과학 또는 공학 전문가가 특정 디자인 문제에 대한 창의성을 평가했습니다. 평가 요소로는 기발함(cleverness), 낯섦(remoteness), 드문 성(uncommonness) 등이 포함되었습니다. 출력 결과는 텍스트 분석(computational text analysis) 기법을 사용하여 심층적으로 분석되어 평가 과정에 영향을 미치는 인지적 요소를 규명하는 데 중점을 두었습니다.

- **Performance Highlights**: 결과적으로 LLM은 예시가 제공된 경우 창의성 점수를 더 정확하게 예측했으며, 전체적인 평가에서 점수 간의 상관 관계가 0.99 이상으로 나타났습니다. 이는 LLM이 창의성의 개별 요소들을 동질화하여 평가하는 경향이 있음을 시사합니다. 전문가와 AI 시스템 간의 창의성 판단 기준 간의 차이를 이해하고 조화시킬 수 있는 기회를 제공합니다.



### A scale of conceptual orality and literacy: Automatic text categorization in the tradition of "N\"ahe und Distanz" (https://arxiv.org/abs/2502.03252)
- **What's New**: 이 논문은 Koch와 Oesterreicher의 "Nähe und Distanz" 모델을 바탕으로 하여 텍스트의 개념적 구술성과 문해성을 평가할 수 있는 통계적 기반을 제공합니다. 기존에 이 모델은 독일 언어학에서 널리 사용되었지만 실제 언어 코퍼스 분석에서는 통계적 뒷받침이 부족했습니다. 이를 해결하기 위해 PCA(주성분 분석)를 활용하여 새로운 평가 척도를 구축하였습니다.

- **Technical Details**: 연구에서는 New High German의 두 개의 코퍼스를 활용하여 언어적 특징에 따라 작성된 텍스트의 개념적 구술성과 문해성을 평가하는 방법을 제시합니다. 논문은 자동 분석과 결합된 이 척도를 사용하여 텍스트를 다차원적으로 분류하고, 언어적 특징을 분석하는 데 있어 편견 없는 접근을 보장합니다. 또한, 기존의 평가 기준과 새로 제시된 척도를 비교하여 차별화된 분석 결과를 도출하는 방법론을 설명합니다.

- **Performance Highlights**: 주요 발견은 개념적 구술성과 문해성의 특징을 구분해야만 텍스트를 정교하게 순위를 매길 수 있다는 점입니다. 이 척도는 코퍼스 편집과 대규모 분석의 가이드로서 활용될 가능성이 높으며, Biber의 Dimension 1과 비교할 때 더욱 적절한 역할을 수행합니다. 이 연구는 텍스트 분석 분야에 기여할 수 있는 실용적인 프레임워크를 제공합니다.



### Mitigating Language Bias in Cross-Lingual Job Retrieval: A Recruitment Platform Perspectiv (https://arxiv.org/abs/2502.03220)
Comments:
          To be published in CompJobs Workshop at AAAI 2025

- **What's New**: 이 논문에서는 온라인 채용 플랫폼에서 이력서와 구인 게시물의 텍스트 구성 요소를 이해하는 것이 중요하다고 강조하고 있습니다. 기존의 연구들은 일반적으로 개별 구성 요소에 집중해왔는데, 이는 여러 전문 도구를 필요로 합니다. 그러나 이러한 단편적인 방법은 채용 텍스트 처리의 일반화 가능성을 저해할 수 있습니다. 따라서 저자들은 다중 작업 이중 인코더(multi-task dual-encoder) 프레임워크를 활용한 통합 문장 인코더를 제안합니다.

- **Technical Details**: 제안된 방법은 세 가지 주요 작업으로 구성되어 있으며, 이는 (A) 직무 제목 번역 순위 (translation ranking), (B) 직무 설명-제목 일치(matching), (C) 직무 분야 분류(classification)입니다. 이러한 작업들은 사람의 레이블 없이 온라인 사용자 생성 구인 게시물의 정보를 사용하여 훈련됩니다. 이를 통해 저자들은 다국어 문장 인코더를 개발하고, 인코더 내 언어 편향을 평가할 수 있는 새로운 지표(Language Bias Kullback-Leibler Divergence)를 제안하였습니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 작은 모델 크기에도 불구하고 기존의 최첨단 모델들보다 뛰어난 성능을 보였습니다. 특히, 저자들은 제안된 엔코더가 현저하게 낮은 언어 편향과 개선된 크로스링구얼(cross-lingual) 성능을 달성했다고 보고합니다. 이는 비영어권 사용자들에게도 효과적으로 적용될 수 있는 가능성을 시사합니다.



### iVISPAR -- An Interactive Visual-Spatial Reasoning Benchmark for VLMs (https://arxiv.org/abs/2502.03214)
- **What's New**: 최근 발표된 iVISPAR는 비전-언어 모델(VLM)의 공간 추론 능력을 평가하기 위해 고안된 인터랙티브 다중 모드 벤치마크입니다. 이 벤치마크는 고전적인 슬라이딩 타일 퍼즐의 변형으로, 논리적 계획, 공간 인지, 다단계 문제 해결 능력을 요구합니다. iVISPAR는 2D, 3D 및 텍스트 기반 입력 방식 모두를 지원하여 VLM의 전반적인 계획 및 추론 능력을 포괄적으로 평가할 수 있습니다. 연구 결과, 일부 VLM은 간단한 공간 작업에서 양호한 성능을 보였으나, 복잡한 구성에서는 어려움을 겪는다는 점이 드러났습니다.

- **Technical Details**: iVISPAR의 주요 특징 중 하나는 Sliding Geom Puzzle(SGP)로, 기존의 슬라이딩 타일 퍼즐을 고유한 색상과 모양으로 정의된 기하학적 객체로 대체했습니다. 벤치마크는 사용자가 자연어 명령어를 발행하여 보드에 대한 작업을 수행할 수 있도록 설계된 텍스트 기반 API를 지원합니다. iVISPAR는 퍼즐의 복잡도를 세밀하게 조정할 수 있으며, 다양한 기준 모델과의 성능 비교가 가능합니다. 최적의 솔루션은 A* 알고리즘을 사용하여 계산되며, 벤치마크는 보드 크기, 타일 수 및 솔루션 경로 등 다양한 요인을 조정하여 복잡도를 확장할 수 있습니다.

- **Performance Highlights**: 실험 결과, 최신 VLM들이 기본적인 공간 추론 작업을 처리할 수 있지만, 3D 환경의 보다 복잡한 시나리오에 직면할 때 상당한 어려움을 보임을 확인했습니다. VLM은 일반적으로 2D 비전에서 더 나은 성능을 보였으나, 인간 수준의 성과에는 미치지 못하는 한계를 보여주었습니다. 이러한 결과는 VLM의 현재 능력과 인간 수준의 공간 추론 간의 지속적인 격차를 강조하며, VLM 연구의 추가 발전 필요성을 시사합니다.



### Improve Decoding Factuality by Token-wise Cross Layer Entropy of Large Language Models (https://arxiv.org/abs/2502.03199)
Comments:
          NAACL 2025 Findings

- **What's New**: 이 논문에서는 Large Language Model(LLM)의 'hallucination' 문제를 해결하기 위한 새로운 접근 방식인 Cross-layer Entropy eNhanced Decoding(END)을 제안합니다. END는 추가적인 훈련 없이도 모델이 생성하는 문장의 사실성을 개선하는 데 초점을 맞추고 있으며, 각 후보 토큰의 사실적 지식량을 정량화하여 확률 분포를 조정하는 방법입니다. 이 연구는 각 토큰 레벨에서 내부 상태 변화와 출력 사실성 간의 상관관계를 분석함으로써 보다 깊이 있는 이해를 제공합니다.

- **Technical Details**: END는 모델의 여러 층에서 예측된 내부 확률의 변화를 이용하여 각 후보 토큰의 사실적 지식을 증폭시키는 기능을 수행합니다. 구체적으로, END는 각 토큰에 대해 사실성을 중요한 요소로 고려하여 최종 예측 분포를 조정합니다. 이 방법은 LLM의 여러 아키텍처에 적용될 수 있으며, 모델의 구조가 서로 다르더라도 일반화된 성능을 보여줍니다.

- **Performance Highlights**: 실험 결과, END를 통해 TruthfulQA 및 FACTOR와 같은 hallucination 벤치마크와 TriviaQA 및 Natural Questions와 같은 일반 QA 벤치마크에서 생성된 콘텐츠의 진실성과 정보성을 향상시켰습니다. 이 방법은 QA 정확도를 안정적으로 유지하면서도 생성되는 내용의 사실성을 상당히 개선하였습니다. 실험은 다양한 LLM 백본에 대한 추가 검증을 통해 END의 효용성과 일반성을 보여주었습니다.



### Euska\~nolDS: A Naturally Sourced Corpus for Basque-Spanish Code-Switching (https://arxiv.org/abs/2502.03188)
- **What's New**: 이 논문은 바스크(Basque)와 스페인어(Spanish) 간의 코드 스위칭(code-switching) 현상을 연구하기 위해 첫 번째 자연어 데이터셋인 EuskañolDS를 소개합니다. 이 데이터셋은 기존 코퍼스에서 언어 식별 모델(language identification models)을 이용해 코드 스위치된 문장을 자동으로 수집하고, 그 결과를 수동으로 검증하여 신뢰할 수 있는 샘플을 확보하는 방법론을 기반으로 하고 있습니다. 따라서 Barasko와 Spanish 언어 간의 코드 스위칭 연구를 위한 중요한 자료를 제공합니다.

- **Technical Details**: 논문에서는 언어 식별 모델을 통해 바스크어와 스페인어가 혼합된 코드 스위칭 문장을 수집하기 위한 반지도학습(semi-supervised learning) 접근 방식을 사용합니다. 최종 데이터셋은 20,008개의 자동 분류된 인스턴스와 927개의 수동 검증 인스턴스가 포함되어 있으며, 두 언어는 각기 다른 언어 계통을 가지며 많은 전형적인 차이를 갖고 있습니다. 이 데이터셋은 바스크어와 스페인어 간의 언어 접촉과 코드 스위칭 현상을 이해하는 데 중요한 자원이 될 것입니다.

- **Performance Highlights**: 이 데이터셋은 서로 다른 출처에서 수집한 코드 스위칭 인스턴스들을 포함하고 있으며, 그 특성을 정량적 분석과 질적 분석을 통해 설명합니다. 특히, 대부분의 인스턴스가 문장 간 코드 스위칭(inter-sentential CS) 형태를 보이며, 이는 바스크의 법정 의사록 및 소셜 미디어 콘텐츠에서 많이 나타납니다. 이러한 성과는 코드 스위칭이 포함된 NLP 모델 개발을 지원할 수 있는 유용한 자료로 작용할 것입니다.



### Scalable In-Context Learning on Tabular Data via Retrieval-Augmented Large Language Models (https://arxiv.org/abs/2502.03147)
Comments:
          Preprint

- **What's New**: 최근 대형 언어 모델(LLMs)을 사용한 연구는 사전 훈련 후 표 형식 데이터(tabular data)로 맞춤 설정함으로써 일반화된 표 형식 인컨텍스트 학습(TabICL) 능력을 획득할 수 있음을 보여주었습니다. 이러한 모델은 다양한 데이터 스키마(data schemas)와 작업 도메인(task domains) 간의 효과적인 전이(transfer)를 수행할 수 있게 되었습니다. 하지만 기존의 LLM 기반 TabICL 접근 방식은 시퀀스 길이 제약으로 인해 제한된 학습 사례(few-shot scenarios)에만 적용 가능한 한계가 있습니다.

- **Technical Details**: 이 논문에서는 표 형식 데이터에 적합한 검색 증강 LLM(retrieval-augmented LLMs)을 제안하며, 이는 맞춤형 검색 모듈(customized retrieval module)과 검색 기반 지시 조정(retrieval-guided instruction-tuning)을 결합한 새로운 접근 방식입니다. 이 방법은 더 큰 데이터셋을 효과적으로 활용할 수 있게 하여, 69개 널리 인식되는 데이터셋에서 성능을 크게 개선하는 동시에 확장 가능한 TabICL을 가능하게 합니다. 또한, 이 연구는 데이터를 효율적으로 활용하기 위한 비모수(non-parametric) 검색 모듈의 설계와 맞춤형 데이터셋 특정 검색 정책의 발전 가능성을 탐구합니다.

- **Performance Highlights**: 연구 결과는 LLM 기반 TabICL이 강력한 알고리즘을 발굴하고, 앙상블 다양성을 강화하며, 특정 데이터셋에서 뛰어난 성능을 보이는 데 기여할 수 있음을 보여주었습니다. 그러나 전반적인 성능에서는 잘 튜닝된 숫자 모델에 비해 여전히 뒤처진다는 사실이 밝혀졌고, 각 데이터셋의 분석 결과 LLM 기반 TabICL 접근 방식은 더 효과적인 검색 정책과 더 다양한 피처 분포를 활용함으로써 더욱 향상될 수 있음을 제시했습니다. 이러한 점에서, 본 접근 방식은 TabICL을 위한 특별한 패러다임으로서 엄청난 잠재력을 지니고 있다고 생각됩니다.



### Teaching Large Language Models Number-Focused Headline Generation With Key Element Rationales (https://arxiv.org/abs/2502.03129)
Comments:
          Pre-print for a paper accepted to findings of NAACL 2025

- **What's New**: 이번 연구에서는 뉴스 기사에서 주제, 개체, 수치 추론(Topic, Entities, Numerical reasoning, TEN) 요소를 포함한 새로운 Chain-of-Thought 프레임워크를 제안합니다. 이는 LLMs가 주제에 맞는 고품질 텍스트를 정확한 수치로 생성할 수 있는 능력을 향상시키기 위한 것입니다. 이 연구의 접근법은 강력한 Teacher LLM을 사용해 TEN 합리화를 생성하고 이를 바탕으로 Student LLM을 교육하여, 숫자가 포함된 헤드라인 생성을 자동화하는 데 중점을 두고 있습니다.

- **Technical Details**: TEN 합리화는 뉴스 기사 내에서 주요 요소에 대한 설명을 포함하며, 이는 LLMs의 수치 추론 및 주제 일치 헤드라인 생성을 증진하는 데 사용됩니다. Teacher-Student 지식 증류 프레임워크를 통해 강력한 Teacher LLM에서 Student LLM으로 지식을 전수하며, 자동으로 TEN 합리화를 생성하도록 fine-tuning 합니다. 이러한 접근 방식은 open-source LLMs(예: Mistral 7B)를 활용하여 TEN 합리화를 비용 효율적으로 생성할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식이 텍스트 품질과 수치 정확성 모두에서 기존의 강력한 기준선보다 월등한 성능을 보이는 것으로 나타났습니다. 우리의 연구는 LLMs가 숫자 중심의 헤드라인을 생성할 수 있도록 하는 데 중요한 기여를 하며, 이는 주제와 수치 정확성이 결합된 요구를 충족할 수 있는 새로운 방법론을 제시합니다. 문서에서 제안된 방식은 후속 연구와 실용적인 응용에서 큰 영향을 미칠 것으로 기대됩니다.



### Policies and Evaluation for Online Meeting Summarization (https://arxiv.org/abs/2502.03111)
Comments:
          8 pages, 1 figure

- **What's New**: 이 논문은 온라인 회의 요약(online meeting summarization)에 대한 최초의 체계적인 연구를 수행합니다. 저자들은 회의가 진행되는 동안 요약을 생성하는 새로운 방식과 정책들을 제안하며, 온라인 요약 작업의 고유한 도전과제를 논의합니다. 기존의 오프라인 요약 방식과 달리, 이 연구는 요약 품질, 지연 시간(latency), 그리고 중간 요약(intermediate summary)의 품질을 평가하는 새로운 메트릭(metric)을 도입합니다.

- **Technical Details**: 온라인 요약 시스템은 주어진 입력 토큰 스트림(input stream)으로부터 읽고(output stream) 출력 토큰을 쓰는 에이전트(agent)로 간주됩니다. 이 시스템은 사전 학습된 오프라인 요약 시스템의 상단에 구축되어, 입력 및 출력 토큰을 읽고 쓰는 시점을 결정하는 다양한 정책을 제안합니다. 특히, 이 연구에서는 질과 지연 시간의 트레이드오프를 자세히 분석하기 위한 메트릭이 개발되었습니다.

- **Performance Highlights**: AutoMin 데이터셋에서 실시한 실험 결과, 온라인 모델이 강력한 요약을 생성할 수 있으며, 제안된 메트릭을 사용하면 다양한 시스템의 품질-지연 시간 간의 트레이드오프를 분석할 수 있음을 보여주었습니다. 평가 결과, 온라인 요약 시스템이 아직 오프라인 시스템과 동일한 수준의 품질에 도달하지는 못하지만, 인간 평가에서 4점 이상의 높은 점수를 기록하며 우수한 요약을 생성할 수 있는 것으로 나타났습니다.



### Structured Token Retention and Computational Memory Paths in Large Language Models (https://arxiv.org/abs/2502.03102)
- **What's New**: 이번 논문에서 제안된 Structured Token Retention (STR) 및 Computational Memory Paths (CMP) 접근법은 기억 관리를 최적화하기 위한 새로운 패러다임입니다. STR은 고유한 계층적 선택 메커니즘을 통해 중요한 토큰을 유지하고, CMP는 프로바빌리스틱(Probabilistic) 라우팅 메커니즘을 도입하여 계산 자원을 동적으로 할당합니다. 이는 전통적인 메모리 관리 접근법과 비교하여 토큰의 생존율을 높이며, 장기적인 상관 관계를 보존하면서도 효율적인 정보 활용을 가능하게 합니다.

- **Technical Details**: STR은 토큰의 맥락 중요도를 기반으로 기억 유지 확률을 동적으로 조정하여 불필요한 메모리 소비를 완화합니다. CMP는 확률적 임베딩을 활용하여 계산 자원을 방향성 있게 조절하며, 이는 각 레이어에서의 토큰의 중요성을 재평가하는 과정을 포함합니다. 따라서 기존의 변환기 아키텍처에 부가적인 절차 없이 원활하게 통합될 수 있습니다.

- **Performance Highlights**: 실험 결과, STR 및 CMP를 구현한 오픈 소스 LLM 모델에서 긴 입력 시퀀스 처리 시 토큰 생존율을 효과적으로 증가시키고, 누적 오류 전파를 줄이는 결과를 보였습니다. 또한, 계산 오버헤드 감소와 함께 추론 속도가 향상되었고, 대규모 생성 아키텍처에서 정보 검색 효율을 최적화했습니다. 이러한 메모리 효율성 향상은 생성 텍스트 처리, 장기 맥락 이해, 확장 가능한 시퀀스 모델링에 널리 적용 가능성을 시사합니다.



### IAO Prompting: Making Knowledge Flow Explicit in LLMs through Structured Reasoning Templates (https://arxiv.org/abs/2502.03080)
Comments:
          Accepted as Oral at KnowFM @ AAAI 2025

- **What's New**: 이번 연구에서는 IAO (Input-Action-Output) 프롬프트 기법을 도입하여 대형 언어 모델(LLMs)이 지식에 접근하고 이를 적용하는 과정을 체계적으로 모델링합니다. 이 방법은 문제를 순차적인 단계로 분해하며, 각 단계에서 사용되는 입력 지식, 수행되는 행동, 생성되는 출력을 명확하게 식별합니다. 이를 통해 LLM의 지식 흐름을 추적하고, 사실 일관성을 검증하며, 지식의 공백이나 오용을 식별할 수 있는 투명성을 제공합니다.

- **Technical Details**: IAO 프롬프트는 다섯 가지 구성 요소로 구성되어 있습니다. 첫째, 서브질문(subquestion)을 통해 LLM이 주요 질문을 더 작은 지식 요소로 분해합니다. 둘째, 각 단계에서 무엇이 지식의 입력(input)인지 명확히 하며, 셋째, LLM이 그 지식을 어떻게 활용(action)하는지 설명합니다. 마지막으로, 이 지식을 적용하여 새롭게 생성되는 출력(output)을 제시합니다. 이러한 구조적 접근은 지식 흐름을 추적하고, 지식 적용을 검증하며, 공백이나 오류를 식별하는 데 도움을 줍니다.

- **Performance Highlights**: IAO 방법은 다양한 추론 작업에서 실험을 통해 효과를 입증하였습니다. 특히, 제로샷 성능(zero-shot performance)을 향상시키며, 인간 평가를 통해 지식 활용을 검증하고 환각이나 추론 오류를 탐지하는 데 기여했습니다. 이는 기존의 프롬프트 기법에 비해 더욱 투명한 지식 활용을 가능하게 합니다.



### DOLFIN -- Document-Level Financial test set for Machine Translation (https://arxiv.org/abs/2502.03053)
Comments:
          To be published in NAACL 2025 Findings

- **What's New**: 이번 연구에서는 DOLFIN이라는 새로운 문서 수준 기계 번역(MT) 테스트 세트를 제안합니다. DOLFIN은 전문 금융 문서로 구성되어 있으며, 문장 단위가 아닌 섹션 단위로 데이터가 제공되어 정보 재조직화와 같은 언어적 현상을 포함할 수 있도록 설정되었습니다. 이 데이터 세트는 5개 언어 쌍에 대해 평균 1950개의 정렬된 섹션을 포함하며, 기계 번역 평가의 새로운 기준을 제시합니다.

- **Technical Details**: DOLFIN은 PDF 문서에서 텍스트를 추출해 구축되며, 세션 레벨 번역과 문서 레벨 번역의 평가를 비교하는 새로운 접근 방식을 사용합니다. 이 데이터 세트는 정보의 재배치와 같은 고수준의 언어적 현상에 중점을 두며, 문서 내에서의 용어 일관성과 숫자 형식의 일관성을 평가할 수 있는 후속 연구의 기초를 제공합니다. DOLFIN은 기존의 문장 정렬 방식에서 벗어나, 고유한 세션 정렬 방식을 사용하고 있습니다.

- **Performance Highlights**: 모델 평가 결과, DOLFIN 테스트 세트는 문맥에 민감한 모델과 문맥 무시 모델 간의 차이를 명확히 나타냈습니다. 특히, 금융 문서 번역에서 모델의 약점을 여실히 드러내어, 문서 수준 평가의 중요성을 강조합니다. DOLFIN은 연구 및 개발 목적의 기초 데이터로 공개되어 커뮤니티에 기여할 것입니다.



### Knowledge Distillation from Large Language Models for Household Energy Modeling (https://arxiv.org/abs/2502.03034)
Comments:
          Source code is available at this https URL

- **What's New**: 이번 연구에서는 대형 언어 모델(Large Language Models, LLMs)을 에너지 모델링에 통합하여 다양한 지리적 지역의 가정 에너지 사용에 대한 현실적이고 문화적으로 민감하며 행동 특화된 데이터를 생성하는 방법을 제안합니다. 이는 스마트 그리드 연구의 진전을 가속화하고, ML 기반 전략의 채택에 대한 의구심을 해소하는 데 기여할 것으로 기대됩니다.

- **Technical Details**: 연구에서 5개의 LLM을 활용하여 가족 구조, 날씨 패턴 및 6개 나라의 가정의 일일 소비 프로필을 체계적으로 생성합니다. 4단계 방법론을 통해 문화적으로 미묘한 활동, 현실적인 날씨 범위, HVAC(Heating, Ventilation and Air Conditioning) 작동 및 독특한 소비 발자국을 포착하는 `에너지 서명'을 포함한 맥락적 일일 데이터를 종합합니다. 추가적으로, 외부 날씨 데이터셋을 직접 통합하는 대체 전략도 explored(탐구)됩니다.

- **Performance Highlights**: 이 데이터셋은 문화적, 기후적, 행동적 요인이 탄소 배출에 어떻게 영향을 미치는지를 통찰력 있게 보여주며, 시나리오 기반 에너지 최적화를 위한 비용 효율적인 접근 방식을 제공합니다. 또한, 프롬프트 엔지니어링(prompt engineering)과 지식 증류(knowledge distillation)가 결합하여 지속 가능한 에너지 연구 및 기후 완화 노력을 진전시킬 수 있는 방법을 강조합니다.



### MedBioLM: Optimizing Medical and Biological QA with Fine-Tuned Large Language Models and Retrieval-Augmented Generation (https://arxiv.org/abs/2502.03004)
- **What's New**: 이 논문에서는 MedBioLM이라는 생물의학 질문 응답 모델을 소개하고 있습니다. MedBioLM은 사실상 정확성과 신뢰성을 높이기 위해 도메인 적응 (domain-adapted) 기술을 활용하고 있으며, 기본 모델보다 높은 성능을 자랑합니다. 이 모델은 단기 및 장기 쿼리를 모두 위한 최적화된 솔루션을 제공합니다.

- **Technical Details**: MedBioLM은 사전 학습된 대형 언어 모델(LLM)을 의학 데이터셋에 맞게 미세 조정(fine-tuning)하고, 검색 강화 생성(RAG) 기술을 통합하여 구현됩니다. 이를 통해 모델은 외부 도메인 지식을 동적으로 포함시켜 사실적 정확도를 향상시키고, 의료 전문가에게 적합한 구조적인 응답을 생성할 수 있습니다. RAG는 특히 짧은 응답에서 사실적 정확성과 언어 유사성을 높이는데 중요한 역할을 합니다.

- **Performance Highlights**: MedBioLM은 MedQA와 BioASQ와 같은 다양한 생물의학 QA 데이터셋에서 정확도 88% 및 96%를 달성하여 기본 모델보다 월등한 성능을 입증하였습니다. 장기 응답 QA에서도 ROUGE-1 및 BLEU 점수가 개선되었으며, 전반적으로 LLM의 생물의학 응용 프로그램에 대한 가능성을 보여주고 있습니다. 이러한 결과는 도메인 최적화된 LLM이 생물의학 연구와 임상 의사결정 지원에 기여할 수 있음을 강조합니다.



### Training an LLM-as-a-Judge Model: Pipeline, Insights, and Practical Lessons (https://arxiv.org/abs/2502.02988)
Comments:
          accepted at WWW'25 (Industrial Track), extended version

- **What's New**: 이 논문에서는 평가 역할을 수행하는 언어 모델인 Themis를 소개하고 있다. Themis는 시나리오에 맞춰 세심하게 조정된 평가 프롬프트와 두 가지 새로운 지시 생성 방법을 통해.context-aware 평가를 제공한다. 이는 LLM의 평가 능력을 효과적으로 증류하며, 지속적인 발전을 위한 유연성을 보장한다.

- **Technical Details**: Themis는 평가를 위한 단계별 시나리오 의존 프롬프트를 사용하고, 인간-AI 협업을 통해 각 시나리오에 맞춰 평가 기준을 설계한다. 이 모델은 reference-based questioning과 role-playing quizzing 방법을 통해 지시 데이터를 생성하며, 이를 통해 LLM의 평가 능력을 더욱 향상시킨다. 지속적인 개발을 위한 유연성을 갖춘 Themis는 GPT-4와 같은 최신 LLM으로부터 평가 능력을 총체적으로 유도한다.

- **Performance Highlights**: Themis는 두 가지 인간 선호 벤치마크에서 높은 성과를 나타내며, 1% 미만의 매개변수를 사용하면서도 모든 다른 LLM을 초과하는 성능을 기록하였다. 또한, Themis의 성능은 공개된 시나리오에서 가장 잘 나타나며, 참고 답변이 평가에 미치는 영향을 분석한 결과, 폐쇄형 시나리오에서는 긍정적인 영향을 주지만 개방형 시나리오에서는 미미하거나 부정적인 영향을 미친다는 사실이 드러났다.



### Position: Editing Large Language Models Poses Serious Safety Risks (https://arxiv.org/abs/2502.02958)
- **What's New**: 이번 논문은 대규모 언어 모델(LLMs)의 지식 편집 방법(knowledge editing, KEs)이 안전 문제를 유발할 수 있음을 경고합니다. KEs는 특정 사실을 수정하는 데에 효과적이고 저비용이라는 장점이 있어 악의적인 목적에도 쉽게 사용될 수 있습니다. 저자들은 KEs의 효과를 분석할 뿐만 아니라, AI 생태계의 취약성과 사회적 인식 부족이 이러한 위험을 악화시키고 있다는 점을 강조합니다.

- **Technical Details**: LLMs의 성장과 함께 KEs는 메모리 기반(KEs), 메타 러닝 기반(KEs) 및 위치 기반 KEs로 구분됩니다. 최신 PEFT(parameter-efficient fine-tuning) 기법을 통해 KEs는 재훈련 없이 모델의 특정 사실을 정확하게 수정할 수 있습니다. 이는 KEs가 특별히 악의적 사용을 위해 유용한 도구가 될 수 있는 이유를 설명합니다.

- **Performance Highlights**: KEs는 낮은 데이터 비용과 빠른 속도로 특정 사실을 편집할 수 있는 강력한 성능을 자랑합니다. 예를 들어, KEs는 모델의 대다수의 매개변수를 수정하지 않으므로 수정된 사실과 원래 학습된 사실 간의 구분이 매우 어렵습니다. 이러한 특성들은 KEs가 전통적인 사이버 보안 위협과는 다른 새로운 위험을 초래할 수 있다는 점을 강조합니다.



### ReachAgent: Enhancing Mobile Agent via Page Reaching and Operation (https://arxiv.org/abs/2502.02955)
- **What's New**: 최근 모바일 AI 에이전트가 주목받고 있습니다. 기존 에이전트는 특정 태스크와 관련된 요소에 집중하여 로컬 최적 해법에 그치고 전체 GUI 흐름을 간과하는 경우가 많았습니다. 이를 해결하기 위해 페이지 접근과 작업 서브태스크로 분리된 훈련 데이터셋인 MobileReach를 구축했습니다.

- **Technical Details**: 저자들은 ReachAgent라는 두 단계의 프레임워크를 제안하여 태스크 완료 능력을 향상시킵니다. 첫 번째 단계에서는 페이지 접근 능력 싸인을 배우고, 두 번째 단계에서는 4단계 보상 함수를 사용하여 GUI 흐름의 선호도를 강화합니다. 또한, 액션 정렬 메커니즘으로 작업 난이도를 줄입니다.

- **Performance Highlights**: 실험 결과, ReachAgent는 기존 SOTA(State of the Art) 에이전트에 비해 단계 수준에서 IoU Acc와 Text Acc가 각각 7.12% 및 7.69% 증가하고, 태스크 수준에서도 4.72% 및 4.63% 향상되었습니다. 이러한 결과는 ReachAgent가 더욱 뛰어난 페이지 접근 및 작업 능력을 가졌음을 보여줍니다.



### LLM-KT: Aligning Large Language Models with Knowledge Tracing using a Plug-and-Play Instruction (https://arxiv.org/abs/2502.02945)
- **What's New**: 최근의 연구는 지식 추적(Knowledge Tracing) 문제에서 대규모 언어 모델(LLMs)의 활용 가능성을 제시합니다. 본 논문에서는 LLM-KT라는 새로운 프레임워크를 제안하여 LLM의 강력한 추론 능력과 전통적인 시퀀스 모델의 장점을 결합합니다. 이는 학생의 행동 패턴을 보다 정확하게 캡처하고 문제 해결 기록의 텍스트 맥락을 효율적으로 활용하는 데 중점을 둡니다.

- **Technical Details**: LLM-KT는 Plug-and-Play Instruction이라는 방법론을 사용하여 LLM과 지식 추적을 정렬합니다. 이 모델은 질문 및 개념에 특화된 토큰을 활용하여 전통적인 방법으로 학습한 여러 모달리티를 통합하는 Plug-in Context와 시퀀스 상호작용을 향상시키는 Plug-in Sequence를 설계하였습니다. 이러한 접근 방식을 통해 학생의 지식 상태를 보다 잘 이해할 수 있도록 합니다.

- **Performance Highlights**: 다양한 데이터셋에 대한 실험을 통해 LLM-KT는 약 20개의 강력한 기준선 모델과 비교하여 최첨단 성능(SOTA)을 달성했습니다. 이러한 결과는 LLM-KT 모델이 지식 추적 문제를 해결하는 데 있어 탁월한 장점을 보여줍니다. 추가적인 분석과 ablation 연구를 통해 여기에 기여하는 주요 요소들에 대한 효과도 입증되었습니다.



### LLaVAC: Fine-tuning LLaVA as a Multimodal Sentiment Classifier (https://arxiv.org/abs/2502.02938)
- **What's New**: LLaVAC는 이미지와 텍스트의 다중 모달리티를 활용하여 감정 분석을 위한 분류기를 구축하는 새로운 방법입니다. 이 방법은 Large Language and Vision Assistant (LLaVA)의 미세 조정을 통해 다중 모달 감정 레이블을 예측하는 데 중점을 두고 있습니다. LLaVAC는 기존의 방법들보다 더 향상된 성능을 보여주며, 공개적으로 구현되어 누구나 접근할 수 있도록 제공됩니다.

- **Technical Details**: LLaVAC는 미세 조정 과정에서 이미지와 텍스트 모달리티를 모두 포함하는 구조화된 프롬프트를 생성하여 감정 레이블을 예측합니다. 이를 통해 이미지, 텍스트 및 그들 간의 결합된 데이터를 독립적으로 그리고 공동으로 분류할 수 있습니다. 이 방법은 LLaVA의 강점을 활용하여 복잡한 수동 특성 공학에 의존하지 않고 다중 모달 데이터 처리를 간소화합니다.

- **Performance Highlights**: LLaVAC는 MVSA-Single 데이터셋에 대한 실험에서 8개 기준 방법들을 초월하는 성능을 기록하며, 최첨단 결과를 달성했습니다. 이 연구는 MSA(Classification)를 위한 MLLMs(다중 모달 대형 언어 모델)의 가능성을 탐구하고 있으며, LLaVA가 그러한 분류기 구축을 위한 강력한 기초로 작용할 수 있음을 보여줍니다.



### What is in a name? Mitigating Name Bias in Text Embeddings via Anonymization (https://arxiv.org/abs/2502.02903)
- **What's New**: 이 논문에서는 텍스트 임베딩 모델에서 이제까지 연구되지 않은 '이름 편향(name-bias)'을 다룹니다. 이름은 개인, 장소, 조직 등을 포함하여 문서의 내용에 중대한 영향을 미칠 수 있으며, 이는 잘못된 유사성 판단으로 이어질 수 있습니다. 후보 모델들이 동일한 의미를 지니지만 다른 캐릭터 이름을 가진 두 문서의 유사성을 인식하지 못하는 문제를 지적하고 있습니다.

- **Technical Details**: 연구에서는 전문적인 분석을 통해 텍스트에 포함된 이름이 임베딩 생성 과정에서 어떠한 편향의 원인이 되는지를 설명하고 있습니다. '텍스트 익명화(text-anonymization)' 기법을 제안하며, 이는 모델을 재훈련할 필요 없이 이름을 제거하면서 본문의 핵심을 유지할 수 있습니다. 이 기법은 두 가지 자연어 처리(NLP) 작업에서 효율성과 성능 향상을 입증하였습니다.

- **Performance Highlights**: 이름 편향을 감소시키기 위해 제안한 익명화 기법은 다양한 텍스트 임베딩 모델과 작업에서 철저한 실험을 통해 효과적임을 입증하였습니다. 이전에 존재하던 다양한 사회적 편향 연구와 비교하여, 이 연구는 특히 이름에 관련된 편향 문제를 최초로 제기하며, 이를 해결하기 위한 간단하고 직관적인 접근법을 제공합니다.



### A Benchmark for the Detection of Metalinguistic Disagreements between LLMs and Knowledge Graphs (https://arxiv.org/abs/2502.02896)
Comments:
          6 pages, 2 tables, to appear in Reham Alharbi, Jacopo de Berardinis, Paul Groth, Albert Meroño-Peñuela, Elena Simperl, Valentina Tamma (eds.), ISWC 2024 Special Session on Harmonising Generative AI and Semantic Web Technologies. this http URL (forthcoming), for associated code and data see this https URL

- **What's New**: 이 논문은 LLM(대형 언어 모델)과 KG(지식 그래프) 간의 메타 언어적 불일치가 존재할 수 있음을 제안합니다. 이는 기존의 사실 불일치 평가 방식에 새로운 관점을 추가하며, 데이터와 지식을 LLM에 통합하는 지식 그래프 공학을 위한 새로운 평가 기준을 만듭니다. 또한, LLM의 오류 원인으로 합리적이지 않은 논의 차원을 고려해야 함을 강조합니다.

- **Technical Details**: 연구팀은 T-REx 데이터셋을 사용하여 LLM의 출력에서 메타 언어적 불일치가 발생하는지를 확인하기 위한 실험을 실시했습니다. 100개의 위키피디아 초록을 샘플링하여 250개의 사실 삼중(triple)을 평가하고, LLM을 판별자로 사용하여 메타 언어적 불일치 여부를 결정했습니다. 이 과정에서 LLM 간의 처리 방식에 따른 오해 가능성도 언급되었습니다.

- **Performance Highlights**: 실험 결과, 유효한 250개의 샘플에서 메타 언어적 불일치의 비율은 평균 0.097로 나타났습니다. 초기 실험의 한계로는 표본 크기와 인간 검증 부재가 지적되었습니다. 향후 연구에서는 메타 언어적 불일치를 명확히 구분할 수 있는 벤치마크 데이터셋이 필요하다고 제안하였습니다.



### Lowering the Barrier of Machine Learning: Achieving Zero Manual Labeling in Review Classification Using LLMs (https://arxiv.org/abs/2502.02893)
Comments:
          Accepted to 2025 11th International Conference on Computing and Artificial Intelligence (ICCAI 2025)

- **What's New**: 이 논문은 대규모 언어 모델(LLM), 특히 Generative Pre-trained Transformer (GPT)와 Bidirectional Encoder Representations from Transformers (BERT)-기반 모델을 통합하여 소규모 업체 및 개인들이 감정 분류(sentiment classification) 기술을 쉽게 활용할 수 있도록 하는 새로운 접근 방식을 제안합니다. 실험 결과, 이 접근방식은 수동 레이블링(manual labeling)이나 전문가 지식 없이도 높은 분류 정확도를 유지함을 보여주며 머신 러닝 기술의 접근성을 크게 높입니다.

- **Technical Details**: 제안된 방법은 Easy Sentiment Classification Startup GPT (ESCS-GPT)와 ALBERT/RoBERTa 기반의 User Reviews Specific Language Model (URSLM)을 포함하여 여러 분류기를 결합한 것입니다. ESCS-GPT는 레이블이 있는 학습 데이터셋을 생성하고, URSLM은 텍스트 임베딩(text embeddings)을 추출하는 데 사용됩니다. 이 시스템은 고도의 전문가 지식 없이도 효율적으로 작동할 수 있도록 설계되었습니다.

- **Performance Highlights**: 제안된 방법은 수동 레이블링 없이도 높은 감정 분류 정확도를 달성할 수 있습니다. 또한, 이 접근 방식은 머신 러닝 혹은 데이터 주석(data annotation)에 대한 전문가 지식이 필요하지 않으며, 제한된 계산 리소스 환경에서도 효과적으로 운영됩니다. 이로 인해 소규모 기업과 개인들이 이 기술을 활용할 수 있는 기회가 대폭 증가했습니다.



### Achieving Operational Universality through a Turing Complete Chemputer (https://arxiv.org/abs/2502.02872)
Comments:
          18 pages, 7 figures, 28 references

- **What's New**: 이번 논문에서는 현대 컴퓨터의 기본 추상화 개념인 Turing Machine(튜링 머신)을 화학 로봇 플랫폼에 적용한 점이 주목할 만합니다. Turing completeness(튜링 완전성) 개념을 활용하여 복잡한 화합물을 합성하기 위한 단위 연산(unit operations)을 프로그램할 수 있는 XDL이라는 화학적 언어를 소개합니다. 이 연구는 기존의 화학적 접근 방식을 넘어, 화학 공정의 자동화 및 정교화를 목표로 하고 있습니다.

- **Technical Details**: 연구진은 화학 과정에서의 프로그래밍 가능성을 높이기 위해 XDL 언어를 사용하여 화학적 인지(chemically-aware) 프로그래밍을 지원하는 로봇 플랫폼을 개발했습니다. 이를 통해 색상 영역(RGB)에서 1670만 가지 조합을 이끌어내고, 각 단계에서 7800만 개의 가능한 상태를 생성함으로써 화학 공간 탐색(conceptual, chemical space exploration)의 새로운 기준을 마련하였습니다. 이러한 접근은 자동 합성 기계(automated synthesis machines)의 화합물 합성 가능성을 증가시킵니다.

- **Performance Highlights**: 이번 연구의 결과는 Turing completeness를 시연하는 인터랙티브한 데모를 통해 나타났습니다. 10개의 관심 영역(ROIs)와 조건부 논리(conditional logic)를 활용하여 설계된 실험 결과는 복잡한 화학 합성을 위한 논리 연산의 정확성과 오류 수정 가능성(error correction)을 보장하는 미래의 화학 프로그래밍 언어 개발에 중요한 기초 자료로 작용할 것입니다. 이는 자동화 및 자율적 화학 합성의 발전을 위한 길잡이가 될 것으로 기대됩니다.



### Position: Multimodal Large Language Models Can Significantly Advance Scientific Reasoning (https://arxiv.org/abs/2502.02871)
- **What's New**: 이번 논문은 Multimodal Large Language Models (MLLMs)의 통합이 과학적 추론 (scientific reasoning)에 혁신적인 발전을 가져올 수 있음을 주장합니다. 기존의 과학적 추론 모델들이 여러 분야에서의 일반화에 어려움을 겪는 점을 지적하며, MLLMs가 텍스트, 이미지 등 다양한 형태의 데이터를 통합하고 추론할 수 있는 능력을 강조합니다. 이로 인해 수학, 물리학, 화학, 생물학 등 여러 분야에서의 과학적 추론이 크게 향상될 수 있음을 제안합니다.

- **Technical Details**: MLLMs는 다양한 타입의 데이터를 처리하고, 이를 바탕으로 추론 (reasoning)을 수행할 수 있는 능력을 갖추고 있습니다. 연구 로드맵은 네 단계로 구성되어 있으며, 이 단계들은 과학적 추론의 다양한 능력을 개발하는 데 필수적입니다. 또한, MLLM의 과학적 추론에서 현황을 요약하고, 나아가 이들이 직면한 주요 도전 과제를 설명합니다.

- **Performance Highlights**: MLLMs의 도입으로 인해 과학적 추론의 응용 가능성이 커지고 있으며, 특히 다양한 데이터 타입을 통합하여 보다 정교한 결과를 도출할 수 있습니다. 그러나 MLLM이 완전한 잠재력을 발휘하기 위해 해결해야 할 주요 도전 과제가 존재합니다. 이러한 문제에 대한 실행 가능한 통찰 및 제안을 통해 AGI (Artificial General Intelligence)를달성하는 데 기여할 수 있는 방안을 모색합니다.



### CAMI: A Counselor Agent Supporting Motivational Interviewing through State Inference and Topic Exploration (https://arxiv.org/abs/2502.02807)
- **What's New**: 이 논문에서는 CAMI라는 새로운 자동 상담사를 소개하며, 이는 Motivational Interviewing(MI)라는 클라이언트 중심의 상담 접근법에 기반을 두고 있습니다. CAMI는 클라이언트의 상태 추론, 동기 주제 탐색, 그리고 응답 생성을 결합한 STAR 프레임워크를 활용하여 다양한 배경을 가진 클라이언트의 상담 결과를 향상시킵니다. 기존의 연구들과 달리, CAMI는 MI의 원리에 부합하는 변화 대화를 유도하도록 설계되었습니다.

- **Technical Details**: CAMI는 클라이언트의 마음 상태를 추론하고, 실시간으로 동기를 탐색하며, 응답을 생성하는 모듈로 구성된 STAR 프레임워크를 채택하고 있습니다. 특히, 트랜스 이론적 모델(Transtheoretical Model)을 기반으로 클라이언트의 상태를 모델링하여 개인화된 상담을 제공합니다. 또한, 주제 트리를 활용하여 다양한 동기 주제를 탐색하며, 각 전략에 대해 생성된 후보 응답을 순위 기반으로 평가함으로써 전략 선택에서의 편향을 줄입니다.

- **Performance Highlights**: CAMI는 자동 및 수동 평가를 통해 MI 기술의 경쟁력, 클라이언트 상태 추론 정확도, 주제 탐색 능력 및 전체 상담 성공률에서 기존의 여러 최첨단 방법보다 뛰어난 성과를 보였습니다. 실험 결과, CAMI는 더 현실감 있는 상담사 행동을 나타내며, 상태 추론 및 주제 탐색의 중요성을 강조하는 실험에서도 긍정적인 결과를 보여주었습니다.



### Consistent Client Simulation for Motivational Interviewing-based Counseling (https://arxiv.org/abs/2502.02802)
- **What's New**: 본 연구에서는 심리 상담에 대한 클라이언트 시뮬레이션을 위한 새로운 프레임워크를 제안합니다. 이 프레임워크는 시뮬레이션된 클라이언트의 다양한 심리 상태와 행동을 추적하고 제어하여, 클라이언트의 동기, 신념, 변화 선호 계획 및 수용성을 기반으로 일관된 행동을 생성합니다. 이를 통해 여러 상담 시나리오에 적합한 일관된 시뮬레이션 클라이언트를 효과적으로 생성할 수 있습니다.

- **Technical Details**: 제안된 클라이언트 시뮬레이션 프레임워크는 네 가지 주요 모듈로 구성되어 있습니다: 상태 전이(state transition), 행동 선택(action selection), 정보 선택(information selection), 그리고 응답 생성(response generation) 모듈입니다. 이 모듈들은 클라이언트의 행동 문제, 초기 상태, 최종 상태, 동기, 신념, 선호 변화 계획, 그리고 수용성에 대한 프로필을 입력으로 사용하여, 상담 과정 중의 상태와 행동을 세밀하게 제어합니다.

- **Performance Highlights**: 실험 결과, 본 연구의 클라이언트 시뮬레이션 방법이 기존 방법보다 높은 일관성을 보이며, 더욱 인간과 유사한 클라이언트를 생성함을 보여줍니다. 자동 평가와 전문가 평가 모두에서, 생성된 상담 세션의 일관성이 평균적으로 더 높은 것으로 확인되었습니다. 이는 상담사의 교육과 평가에 있어 효과적인 훈련 도구가 될 수 있는 가능성을 제시합니다.



### Speculative Prefill: Turbocharging TTFT with Lightweight and Training-Free Token Importance Estimation (https://arxiv.org/abs/2502.02789)
- **What's New**: 이 논문은 SpecPrefill이라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 대규모 언어 모델(LLM)의 추론에서 시간-첫 토큰 (TTFT)을 개선하여, 다양한 길이의 쿼리에 적용 가능하도록 설계되었습니다. SpecPrefill은 사전 훈련 없이 로컬에서 중요한 토큰을 추측하여 전체 계산량을 줄이고, 더 높은 QPS(Queries Per Second)를 달성할 수 있습니다.

- **Technical Details**: SpecPrefill은 경량 모델을 사용하여 주어진 문맥에 기반하여 중요 토큰을 선택적으로 예측합니다. 이 토큰들은 이후 메인 모델에 전달되어 처리됩니다. 이러한 접근 방식은 모델의 FLOPS(부동 소수점 연산 수)를 줄여주며, 모델의 훈련 데이터나 마무리 작업 없이도 쉽게 배포 가능합니다.

- **Performance Highlights**: SpecPrefill은 Llama-3.1-405B-Instruct-FP8 모델의 경우 최대 7배의 QPS 향상과 7.66배의 TTFT 감소를 견인했습니다. 다양한 실제 및 합성 데이터 세트를 대상으로 한 평가를 통해 이 프레임워크의 효과성과 한계를 검증하였습니다. 또한, SpecPrefill은 기존 기술들과 결합하여 더 큰 모델에게도 확장 가능하다는 장점을 가지고 있습니다.



### SimMark: A Robust Sentence-Level Similarity-Based Watermarking Algorithm for Large Language Models (https://arxiv.org/abs/2502.02787)
Comments:
          15 pages, 5 tables, 6 figures

- **What's New**: 이 논문은 대규모 언어 모델(LLM)로 생성된 텍스트를 추적 가능한 방법을 제시하며, SimMark라는 후처리 수위 알고리즘을 소개합니다. 이 알고리즘은 모델의 내부 logits에 접근하지 않고도 작동할 수 있어, API-only 모델을 포함한 다양한 LLM과 호환됩니다. SimMark는 의미론적 문장 임베딩의 유사성과 거부 샘플링(rejection sampling)을 활용하여 인간에게는 인식할 수 없는 통계적 패턴을 부여합니다.

- **Technical Details**: SimMark는 텍스트 생성 이후에 작동하는 후처리(posthoc) 수위 알고리즘으로, 문장 임베딩 유사성을 기반으로 합니다. 이 알고리즘은 거부 샘플링을 사용해 인접 문장의 임베딩 유사성이 특정 범위에 들어올 때까지 여러 번 LLM에 질의합니다. 검사할 때, 통계 검정 방법인 one-proportion soft-z-test를 사용하여 인간이 작성한 텍스트와 LLM이 생성한 텍스트를 구분합니다.

- **Performance Highlights**: 실험 결과, SimMark는 LLM이 생성한 내용의 견고한 수위관리를 위한 새로운 기준을 세웠습니다. 이 알고리즘은 이전의 문장 수준 수위기술보다 강인성과 샘플링 효율성이 뛰어나며 다양한 도메인에서 보다 넓은 적용성을 제공합니다. 특히, 인간 작성 텍스트에 대한 낮은 잘못된 긍정률(false positive rate)을 유지하면서 높은 품질의 텍스트를 보장합니다.



### SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Mod (https://arxiv.org/abs/2502.02737)
- **What's New**: 이 논문은 SmolLM2라는 최신 작은 언어 모델을 개발한 내용을 문서화합니다. SmolLM2는 17억 개의 파라미터를 가진 모델로, 11조 개의 토큰 데이터를 활용하여 훈련되었습니다. 또한 기존 데이터셋에 비해 문제적이었던 부분을 해결하기 위해 FineMath, Stack-Edu, SmolTalk와 같은 새로운 데이터셋을 도입하였습니다.

- **Technical Details**: 모델 훈련은 멀티 스테이지 훈련 프로세스를 사용하여 웹 텍스트, 전문 수학, 코드 및 지침 데이터가 혼합된 방식으로 수행되었습니다. 기존 데이터셋의 문제점을 극복하기 위해 수동 세분화 및 재조정 과정을 통해 훈련 데이터의 배합 비율을 조정하여 성능을 최적화했습니다. SmolLM2는 Qwen2.5-1.5B 및 Llama3.2-1B와 같은 최근의 작은 언어 모델을 능가하는 성능을 보여주고 있습니다.

- **Performance Highlights**: SmolLM2는 다양한 중요한 작업에서 만족스러운 성능을 제공하는 작은 모델로, 모바일을 포함한 다양한 장치에서 실행할 수 있는 경제적 장점이 있습니다. 특히, 개발된 지침 조정 변형은 더욱 향상된 성능을 보여줍니다. 이번 연구를 통해 SmolLM2와 모든 데이터셋이 공개되어 향후 연구에 기여할 수 있는 기반이 마련되었습니다.



### Cross-Lingual Transfer for Low-Resource Natural Language Processing (https://arxiv.org/abs/2502.02722)
Comments:
          Doctoral Thesis: University of the Basque Country UPV/EHU

- **What's New**: 이번 논문은 Natural Language Processing (NLP) 분야에서 발생한 큰 진전을 다루고 있습니다. 특히, 높은 자원이 있는 언어들, 예를 들어 영어에서 주로 이익을 본 대형 언어 모델(Large Language Models)의 출현에 주목합니다. 하지만 자원이 부족한 언어들은 여전히 훈련 데이터와 계산 자원의 부족으로 인해 상당한 어려움에 직면해 있습니다.

- **Technical Details**: 이 연구는 cross-lingual transfer learning(언어 간 전이 학습)에 초점을 맞추고 있습니다. 구체적으로, Named Entity Recognition, Opinion Target Extraction, Argument Mining과 같은 Sequence Labeling 작업을 다룹니다. 연구는 세 가지 주요 목표로 구성되며, 데이터 기반 전이 학습 방법 개선, 최신 다국어 모델을 활용한 모델 기반 전이 학습 접근법 개발, 실제 문제에의 적용 및 향후 연구를 위한 오픈 소스 리소스 제작을 포함합니다.

- **Performance Highlights**: 이 논문에서는 T-Projection이라는 새로운 방법을 제시하여 데이터 기반 전이를 개선합니다. T-Projection은 텍스트-투-텍스트 다국어 모델과 기계 번역 시스템을 활용한 주석 프로젝션 방법으로, 기존 방법보다 현저히 더 우수한 성능을 보입니다. 또한, 제약 디코딩 알고리즘을 소개하여 제로샷 세팅에서 언어 간 Sequence Labeling을 향상시키고, 최초의 다국어 텍스트-투-텍스트 의료 모델인 Medical mT5를 개발하여 실질적인 응용에 대한 연구의 영향을 보여줍니다.



### Developing multilingual speech synthesis system for Ojibwe, Mi'kmaq, and Malis (https://arxiv.org/abs/2502.02703)
- **What's New**: 이번 연구는 Ojibwe, Mi'kmaq, Maliseet 등 세 가지 북미 원주민 언어를 위한 경량화된 multilingual TTS 시스템을 소개합니다. 연구 결과, 세 가지 유사한 언어에서 훈련된 다국어 TTS 모델이 데이터가 부족할 때 단일 언어 모델보다 성능이 우수함을 보여주었습니다. 또한, Attention-free architectures는 self-attention 아키텍처와 비교하여 메모리 효율성을 높였습니다.

- **Technical Details**: 이 시스템은 Matcha-TTS를 기반으로 하여 조건부 flow matching 기술을 사용하고 있으며, TTS 모델은 텍스트 인코더, 지속 시간 예측기 및 flow matching 디코더로 구성됩니다. 저자는 단일 화자를 위한 전통적인 Matcha-TTS 모델을 다국어 음성 합성에 맞게 수정하여 고유한 화자 및 언어 임베딩을 추가했습니다. 또한, 효율적인 배포를 위해 self-attention 대신 Mamba2와 Hydra와 같은 attention-free 레이어를 탐구했습니다.

- **Performance Highlights**: 연구에서는 3개 언어에 대한 TTS 모델이 공동체 중심의 음성 녹음 프로세스를 통해 수집된 데이터를 기반으로 하여 높은 성능을 나타낸다고 강조합니다. 특히, Mamba2와 Hydra로 대체된 attention 모듈이 모든 언어에서 기대되는 성능을 유지하면서도 파라미터 수를 줄였음을 보여주었습니다. 이러한 접근 방식은 자원 부족 환경에서도 효과적인 언어 기술 개선을 가능하게 합니다.



### How Inclusively do LMs Perceive Social and Moral Norms? (https://arxiv.org/abs/2502.02696)
Comments:
          Accepted at NAACL 2025 Findings

- **What's New**: 이 논문은 언어 모델(LM)이 다양한 집단의 사회적 및 도덕적 규범을 어떻게 이해하고 반영하는지를 다룹니다. 기존의 100명의 인간 주석자의 응답과 비교하여 11개의 LM으로부터의 판단을 분석하였습니다. 특히, 이러한 연구는 소외된 관점의 대표성이 부족할 수 있다는 문제를 제기하며, 더 많은 포용성을 갖춘 LM 개발의 필요성을 강조합니다.

- **Technical Details**: 이 연구에서는 사회적 및 도덕적 주제와 관련된 규칙(RoT)을 사용하여 11개의 LM을 평가하였습니다. 각 모델의 응답과 인간 주석자의 응답 사이의 정렬을 분석하기 위해 Absolute Distance Alignment Metric (ADA-Met)을 도입했습니다. 이 지표는 주관적인 응답 간의 특정 거리를 측정하여 모델과 인간 주석자 간의 관계를 분석하는 데 도움을 줍니다.

- **Performance Highlights**: 연구 결과, 높은 소득 집단과 젊은 연령대의 LM 응답이 인간의 규범과 더 잘 일치하는 것으로 나타났습니다. 반면, LM의 응답에서는 소외된 집단의 목소리가 충분히 반영되지 않았으며, 이는 사회적 불평등을 심화시킬 수 있는 우려 요소로 작용합니다. 이러한 발견은 LM이 다양한 인간 가치를 포용하는 방향으로 개선되어야 함을 강조합니다.



### Transformers Boost the Performance of Decision Trees on Tabular Data across Sample Sizes (https://arxiv.org/abs/2502.02672)
Comments:
          12 pages, 6 figures

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)과 TabPFN을 그래디언트 부스팅 결정 트리(GBDT)와 결합하여, 데이터셋의 크기가 작은 경우와 큰 경우 모두에서 각각의 성능을 넘는 새로운 모델 LLM-Boost 및 PFN-Boost를 제안합니다. 이 새로운 방법론은 자연어 이해 및 미리 학습된 트랜스포머의 이점을 활용하여 대규모 데이터셋에서도 효과적으로 작동합니다. 특히, LLM-Boost와 PFN-Boost는 두 가지 방법 모두 기존의 독립형 모델보다 우수한 성능을 보여줍니다.

- **Technical Details**: LLM-Boost는 LLM의 예측 결과를 초기값으로 삼고, 이를 통해 GBDT 모델을 조정하여 잔여 오류를 학습하는 방식으로 작동합니다. 이 방법은 열 헤더를 활용하여 강력한 사전 지식을 적용하고, 결정 트리 알고리즘의 유도 편향과 확장성을 결합합니다. PFN-Boost는 TabPFN과 GBDT의 조합을 통해 유사한 성과를 확보하였으나, 열 헤더를 사용하지 않는 점에서 차별화됩니다.

- **Performance Highlights**: 실험 결과 LLM-Boost와 PFN-Boost는 다양한 데이터셋 크기에서 탁월한 성능을 발휘하였으며, 특히 PFN-Boost는 매우 작은 데이터셋을 제외하고는 실험한 모든 접근법 중 최고의 평균 성능을 기록했습니다. 두 방법 모두 여러 강력한 기준선 및 앙상블 알고리즘과 비교하여 최첨단 성능을 입증하며, 이를 통해 모델의 신뢰성과 효율성을 강화할 수 있습니다.



### A Training-Free Length Extrapolation Approach for LLMs: Greedy Attention Logit Interpolation (GALI) (https://arxiv.org/abs/2502.02659)
Comments:
          9 pages, under review in the conference

- **What's New**: 이번 논문에서는 Greedy Attention Logit Interpolation (GALI)이라는 새로운 training-free 길이 외삽(extrapolation) 방법을 제안합니다. GALI는 사전 훈련된 위치 간격을 최대한 활용하면서 attention logit outlier 문제를 피하는 데 중점을 두며, LLMs의 positional O.O.D. 문제를 해결하는 데 기여합니다. GALI는 기존의 state-of-the-art training-free 방법들보다 일관되게 우수한 성능을 나타내며, 단기 맥락 작업에서도 더욱 향상된 결과를 제공합니다.

- **Technical Details**: GALI는 두 가지 주요 목표를 가지고 있습니다. 첫째, 훈련 맥락(window) 내의 고유한 위치 ID를 유지하여 attention 계산의 방해 요소를 최소화하고, 둘째, attention logit interpolation 전략을 통해 outlier 문제를 완화하는 것입니다. GALI는 초기화 단계에서 훈련된 위치 간격을 최적화하고, 생성 단계에서 새로운 토큰에 대해 동적으로 보간된 위치 ID를 생성하여 위치 변동성에 대한 민감성을 높입니다.

- **Performance Highlights**: GALI는 세 가지 벤치마크를 통해 평가되었으며, 다양한 실험 결과에서 기존의 최고 수준의 training-free 방법들을 일관되게 초월하는 결과를 보여주었습니다. 특히, 모델이 위치 간격을 해석하는 데 있어 불일치성을 바탕으로 한 성능 향상 전략을 발견하였으며, 짧은 맥락 작업에서도 더 좋은 결과를 도출할 수 있음을 확인했습니다. GALI는 LLMs의 긴 텍스트 이해에 있어 중요한 진전을 이루며, 기존의 긴 맥락 프레임워크와 원활하게 통합됩니다.



### Do Large Language Model Benchmarks Test Reliability? (https://arxiv.org/abs/2502.03461)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 신뢰성을 측정하는 중요성을 강조합니다. 기존의 벤치마크들이 모델의 능력에 초점을 맞추었지만, 신뢰성 평가에는 소홀했던 점을 지적하며, 이를 보완하기 위해 'platinum benchmarks'라는 새로운 벤치마크 개념을 제안합니다. 이러한 플래티넘 벤치마크는 라벨 오류를 최소화하고 명확성을 높이기 위해 정성적으로 조정된 것입니다.

- **Technical Details**: 플래티넘 벤치마크는 기존의 벤치마크들에서 문제가 되는 라벨링 오류를 보정하고, 모델 평가 과정을 보다 효율적으로 개선합니다. 논문은 VQA v2.0, SQuAD2.0, HotPotQA와 같은 여러 질문 응답 벤치마크를 이용하여, 모델의 정확성을 측정하는 방법론을 설명합니다. 특히, 비명확한 질문을 배제하고 'yes/no' 질문으로 한정하여 샘플의 명확성을 높이는 방향으로 편집했습니다.

- **Performance Highlights**: 연구 결과, 최첨단 LLM들이 초등 수준의 수학 문제조차 해결하는 데 실패하는 경우가 많음을 발견하였습니다. 이를 통해 LLM의 성능이 예상보다 더 많은 간단한 작업에서 오류가 발생하고 있음을 확실히 보여줍니다. 연구자들은 이러한 모델 실패의 패턴을 분석하여, 언어 모델의 신뢰성을 높일 수 있는 방향을 제시하고 있습니다.



### Adapt-Pruner: Adaptive Structural Pruning for Efficient Small Language Model Training (https://arxiv.org/abs/2502.03460)
- **What's New**: 본 논문은 Adaptive Pruner라는 새로운 구조적 가지치기(Structured Pruning) 방법을 제안하여, 기존 방법보다 월등한 성능을 보입니다. 이를 통해 입력 신호에 대한 가중치 탄력성을 활용하여 각 레이어에 적합한 희소성(Sparsity)을 적용합니다. 또한, Adaptive Accel이라는 새로운 가속화 패러다임을 도입하여 가지치기와 훈련을 병행하는 방식을 보여줍니다. 최종적으로, Adapt-LLMs라는 새로운 모델 패밀리를 통해 강력한 처리 성능을 달성하였습니다.

- **Technical Details**: Adaptive Pruner는 각 레이어의 상대적인 중요도를 평가하여 가지치기를 진행하며, 적은 비율의 뉴런(약 5%)만을 제거함으로써 성능 감소를 최소화합니다. 특히, 제거 후 후속 훈련을 통해 성능 복구가 가능하며, 이는 기존의 고정된 가지치기 방식과 대조됩니다. 본 논문에서는 LLaMA-3.1-8B 모델을 활용한 실험을 통해 기존의 LLM-Pruner, FLAP, 그리고 SliceGPT보다 평균 1%-7% 더 나은 정확도를 기록하였습니다. 이 외에도 Adaptive Pruner는 MobileLLM-125M의 성능을 경쟁사 모델에 맞춰 복구했습니다.

- **Performance Highlights**: Adaptive Pruner의 성능은 MMLU 벤치마크에서 MobileLLM의 성능을 200배의 적은 토큰 수로 복구하는 데 기여했습니다. 또한, Discovering 1B 모델은 LLaMA-3.2-1B를 여러 벤치마크에서 초월하는 성과를 보였습니다. 결과적으로, Adapt-LLMs 모델은 강력한 공개 모델들에 비해 우위에 서게 되며, 이는 다양한 벤치마크에서 뛰어난 성능을 발휘하고 있음을 보여줍니다. 이러한 실험들은 작은 언어 모델을 활용한 실제 응용능력을 확대하는 데 중요한 기반이 될 것입니다.



### Harmony in Divergence: Towards Fast, Accurate, and Memory-efficient Zeroth-order LLM Fine-tuning (https://arxiv.org/abs/2502.03304)
- **What's New**: 이번 연구는 대형 언어 모델(LLM)의 미세 조정에서 메모리 효율성을 높이는 새로운 방법인 Divergence-driven Zeroth-Order (DiZO) 최적화를 제안합니다. DiZO는 레이어별로 적응형 업데이트를 통해 기존의 1차 최적화(FO) 방식처럼 다양한 크기의 업데이트를 구현하며, 이는 메모리 제한이 있는 환경에서도 효과적으로 적용될 수 있습니다. 이러한 접근법은 ZO 최적화의 성능 한계를 극복하고, 훈련 효율성을 크게 향상시킵니다.

- **Technical Details**: DiZO 최적화는 레이어 별로 적응형 업데이트를 수행하며, 이는 FO의 학습 용량과 유사한 효과를 제공합니다. 기존 ZO는 무작위 변동을 활용하여 경량화된 그래디언트 추정을 하는데 반해, DiZO는 각 레이어의 개별 최적화 요구 사항을 반영하여 정확하게 규모를 조정된 업데이트를 생성합니다. 이 과정에서 프로젝트를 활용하여 기울기를 요구하지 않으면서 효과적으로 레이어별 적응형 업데이트를 구현합니다.

- **Performance Highlights**: 실험 결과에 따르면, DiZO는 다양한 데이터셋에서 수렴을 위한 훈련 반복 횟수를 크게 줄이며, GPU 사용 시간을 최대 48%까지 절감하는 것으로 나타났습니다. 또한, RoBERTa-large, OPT 시리즈, Llama 시리즈와 같은 여러 LLM에서 대표적인 ZO 기준을 지속적으로 능가하였으며, 메모리가 많이 요구되는 FO 미세 조정 방법을 초월하는 경우도 있었습니다. 이는 DiZO가 메모리 효율성과 학습 성능을 동시에 개선할 수 있는 훌륭한 솔루션임을 증명합니다.



### SymAgent: A Neural-Symbolic Self-Learning Agent Framework for Complex Reasoning over Knowledge Graphs (https://arxiv.org/abs/2502.03283)
- **What's New**: 이번 논문에서는 SymAgent라는 혁신적인 신경-기호 에이전트 프레임워크를 소개합니다. 이 프레임워크는 Knowledge Graphs (KGs)와 Large Language Models (LLMs)의 협력적 증대를 통해 복잡한 추론 문제를 해결하는 데 중점을 두고 있습니다. KGs를 동적 환경으로 간주하여, 복잡한 추론 작업을 다단계 인터랙티브 프로세스로 변환해 LLM이 더 깊고 의미 있는 추론을 할 수 있도록 지원합니다.

- **Technical Details**: SymAgent는 두 개의 모듈인 Agent-Planner와 Agent-Executor로 구성되어 있습니다. Agent-Planner는 LLM의 유도적 추론 능력을 활용하여 KGs에서 기호 규칙을 추출하고, 효율적인 질문 분해를 안내합니다. Agent-Executor는 미리 정의된 액션 툴을 자율적으로 호출하며 KGs와 외부 문서의 정보를 통합해 KG의 불완전성 문제를 해결합니다.

- **Performance Highlights**: 실험 결과, SymAgent는 약한 LLM 백본(예: 7B 시리즈)을 사용하더라도 다양한 강력한 기준선과 비교해 더 나은 또는 동등한 성능을 보여주었습니다. 분석 결과, 에이전트는 누락된 트리플을 식별하여 자동으로 KG를 업데이트 할 수 있는 능력을 가지고 있으며, 이는 KG의 불완전성 문제에 효과적으로 대응할 수 있게 해줍니다.



### Analyze Feature Flow to Enhance Interpretation and Steering in Language Models (https://arxiv.org/abs/2502.03032)
- **What's New**: 이 논문에서는 대규모 언어 모델의 서로 다른 레이어에서 발견된 특성을 체계적으로 매핑하는 새로운 접근법을 제시합니다. 이전 연구에서는 레이어 간의 특성 연결성을 검토했으며, 최근에는 sparse autoencoder (SAE)를 활용해 이러한 특성이 어떻게 발달하는지를 분석했습니다. 데이터 없이 cosine similarity 기법을 사용하여 각 레이어에서 특정 특성이 지속되거나 변형되는 과정을 추적하는 방법이 도입되었습니다.

- **Technical Details**: 본 연구에서는 MLP, attention 및 residual 각 모듈에 걸쳐 SAE 특성을 정렬하는 데이터를 필요로 하지 않는 방법을 제안합니다. 이렇게 모은 정보를 통해 'flow graphs'라는 형태로 모델 내에서 특성이 어떻게 생성되고 전파되거나 소멸되는지를 포착할 수 있습니다. 이러한 접근은 단일 레이어 분석에서는 발견할 수 없는 특성의 생성 및 정제의 독특한 패턴을 드러냅니다.

- **Performance Highlights**: 이 연구에서는 flow graphs가 다수의 SAE 특성을 동시에 타겟팅함으로써 모델 스티어링의 품질을 향상할 수 있음을 보여줍니다. 또한 이 프레임워크는 SAE 특성을 사용한 최초의 다층 스티어링 시범을 제공합니다. 이러한 방법은 특성의 수명과 진화를 발견하고 컴퓨테이셔널 회로를 형성하는 방식을 이해하는 데 기여하는 중요한 통찰력을 제공합니다.



### Scaling Laws for Upcycling Mixture-of-Experts Language Models (https://arxiv.org/abs/2502.03009)
Comments:
          15 figures, 8 tables

- **What's New**: 이 논문은 대규모 언어 모델(LLMs)을 희소 모델인 mixture-of-experts (MoE) 구조로 업사이클링(upcycling)하는 방법을 연구합니다. 업사이클링은 작은 사전 학습(pretrained) 모델을 활용하여 더 큰 MoE 모델을 훈련하는 접근 방식입니다. 실험을 통해 데이터셋 크기와 모델 구성에 따른 성능 스케일링 법칙(empirical scaling laws)을 규명하였으며, 그러한 법칙들이 훈련의 효율성과 상관관계가 있음을 밝혔습니다.

- **Technical Details**: Dense 모델은 주로 autoregressive 언어 모델링 목표로 사전 학습된 transformer 기반 구조로, Llama2 모델과 유사합니다. MoE는 각 MLP 블록을 여러 전문가 블록으로 대체하여 구성되며, 이 블록의 출력을 비선형적으로 결합하여 다음 레이어로 전달합니다. 이 연구에서는 업사이클링을 통해 dense 모델의 가중치를 MoE에 효과적으로 재사용할 수 있는 가능성을 제시하였습니다.

- **Performance Highlights**: 연구 결과, 업사이클링을 통한 MoE 훈련은 특정 데이터셋 크기 이하에서 효과적인 것으로 나타났습니다. 그러나 데이터의 크기가 증가함에 따라, 업사이클링의 효율성은 감소하고 처음 에너지(수반 비용)도 훈련 속도에 영향을 미쳤습니다. 이러한 발견은 업사이클링이 전체 예산 제약 내에서 처음부터 훈련하는 것보다 더 나은 성과를 낼 수 있는 조건을 제시합니다.



### SPARC: Subspace-Aware Prompt Adaptation for Robust Continual Learning in LLMs (https://arxiv.org/abs/2502.02909)
- **What's New**: 본 연구에서는 대형 언어 모델(LLMs)을 위한 경량의 연속 학습 프레임워크인 SPARC를 제안합니다. SPARC는 차원 축소된 공간에서 프롬프트 튜닝을 통해 효율적으로 작업에 적응할 수 있도록 설계되었습니다. 주성분 분석(PCA)을 활용하여 훈련 데이터의 응집된 부분 공간을 식별하며, 이는 훈련 효율성을 높이는 데 기여합니다.

- **Technical Details**: SPARC의 핵심 구조는 각 작업을 입력 임베딩 공간의 부분 공간으로 표현하고, 부분 공간의 중첩을 정량화하여 새로운 작업에 기존의 프롬프트를 재사용할 수 있는지를 판단합니다. 이는 계산 오버헤드를 줄이고 관련 작업 간의 지식 전이를 촉진합니다. 이 프레임워크는 모델의 기본 구조를 변경하지 않고 소프트 프롬프트만 업데이트하여 매개변수를 절약하는 효율성을 지니고 있습니다.

- **Performance Highlights**: 실험 결과, SPARC는 연속적인 학습 환경에서도 효과적으로 지식을 유지하고, 97%의 과거 지식 보존율을 기록했습니다. 도메인 간 학습에서는 평균 잊어버린 비율이 3%에 불과하고, 작업 간 학습에서는 잊어버림이 전혀 발생하지 않는 것으로 나타났습니다. 이러한 성과는 모델의 0.04%의 매개변수만 미세 조정함으로써 이루어졌습니다.



### ScholaWrite: A Dataset of End-to-End Scholarly Writing Process (https://arxiv.org/abs/2502.02904)
Comments:
          Equal contribution: Linghe Wang, Minhwa Lee | project page: this https URL

- **What's New**: 이번 연구는 ScholWrite 데이터세트를 소개합니다. 이는 학술 작문 과정에서의 키 입력 로그를 기록한 최초의 데이터세트로, 각 키 입력에 대한 인지적 작성 의도를 철저히 주석 처리했습니다. 데이터세트는 5개의 연구 초록으로부터 수집된 LaTeX 기반의 키 입력 데이터로 구성되어 있으며, 약 62,000회의 텍스트 변경 사항이 포함되어 있습니다.

- **Technical Details**: ScholaWrite 프로젝트는 Chrome 확장을 통해 실시간으로 작성 과정을 기록하며, 데이터 접근성을 높이고 여러 연구자들로부터 LaTeX 기반의 키 입력을 수집합니다. 이 연구에서는 인지적 작문 과정에 대해 포괄적인 세분화를 제공하는 새로운 분류체계를 개발했습니다. 학술 작문에서 사용되는 인지적 의도를 이해하기 위해, 연구자들은 동료 평가를 통해 주의 깊게 주석을 단 데이터세트를 생성하였습니다.

- **Performance Highlights**: 실험 결과, ScholaWrite 데이터세트에서 파인튜닝된 Llama-8B 모델이 최종 작성물의 높은 언어적 품질을 달성하였음을 보여주었습니다. 이 데이터세트는 AI 작문 보조 도구의 발전에 기여할 가능성을 지니고 있으며, 저자의 인지적 행동을 이해하고 반영하여 학술 작문 지원을 개선할 수 있는 자원으로 활용될 수 있습니다.



### Leveraging the true depth of LLMs (https://arxiv.org/abs/2502.02790)
- **What's New**: 이 논문은 대형 언어 모델(LLM)의 깊이를 줄이는 여러 가지 방법을 탐구하고, 레이어를 그룹화하여 병렬로 평가하는 새로운 접근 방식을 제안합니다. 이러한 수정은 그래픽 구조를 최적화하여 초당 생성되는 토큰 수를 1.20배 향상시키며, 원래 정확도의 95%-99%를 유지합니다. 이를 통해 대규모 LLM 배포에서 효율성을 획기적으로 개선할 수 있습니다.

- **Technical Details**: 연구진은 프리트레인된 LLM의 컴퓨테이셔널 그래프를 수정하는 다양한 개입 전략을 분석합니다. 레이어의 셔플링, 가지치기(pruning), 병합을 통해 여러 연속 블록을 병렬 실행할 수 있는 전략을 제시하고, 이를 통해 성능 저하를 최소화하며 추론 속도를 개선합니다. 특히, 이러한 접근법은 기존의 Transformer 모델에 적용 가능하며, 재훈련 없이도 성능을 유지할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식은 모델 성능을 유지하면서 서빙 효율성을 크게 향상시킵니다. 기존 모델보다 약 1.20배 빠른 처리 속도를 가지며, 최대 1% 이내의 성능 저하로 고급 ICL(In-Context Learning)을 가능하게 합니다. 추가적인 미세 조정을 통해 일부 성능 손실을 회복하면서도 속도 향상을 유지할 수 있습니다.



### Twilight: Adaptive Attention Sparsity with Hierarchical Top-$p$ Pruning (https://arxiv.org/abs/2502.02770)
- **What's New**: 본 논문에서는 기존의 sparse attention 알고리즘이 고정 예산을 사용하여 동적인 실제 시나리오에서의 효율성과 정확도를 잘 반영하지 못하는 문제를 해결하는 새로운 접근법을 제시합니다. top-p sampling (nucleus sampling)을 sparse attention에 적용하여 적응형 예산을 가능하게 하는 Twilight라는 프레임워크를 제안하였습니다. 이 방법은 정확도를 희생하지 않으면서도 매우 높은 수준으로 중복 토큰을 제거할 수 있습니다.

- **Technical Details**: Twilight는 기존의 sparse attention 알고리즘에 적응형 sparsity를 도입하여, 세부적인 예산 조정 없이도 성능을 극대화하는 접근법을 구현합니다. 기본 알고리즘을 통해 conservatively 많은 토큰을 선택한 후, top-p 방식으로 가장 중요한 토큰 집합을 다시 정제하는 방식으로 작동합니다. 이를 통해, self-attention 연산에서 최대 98%의 중복 토큰을 능동적으로 잘라내며 성능을 크게 향상시킵니다.

- **Performance Highlights**: 실험 결과에 따르면, Twilight는 self-attention 작업에서 최대 15.4 배의 속도 향상과 기존의 sparse attention 방법에서도 최대 2.2 배의 성능 개선을 이루어냈습니다. 또한, end-to-end 토큰 지연에서도 3.9 배의 속도 향상을 보였으며, 다양한 benchmark에서 정확도 손실 없이 성능을 최적화하는 성과를 보여주었습니다.



### Peri-LN: Revisiting Layer Normalization in the Transformer Architectur (https://arxiv.org/abs/2502.02732)
Comments:
          Preprint

- **What's New**: 이 논문에서는 Transformer 아키텍처에서의 layer normalization (LN) 전략이 대규모 훈련의 안정성 및 수렴 속도에 미치는 영향을 분석합니다. 특히, 최근의 Transformer 모델에서는 'Peri-LN'이라는 새로운 LN 배치 전략이 나타났으며, 이는 서브 레이어 전후에 LN을 적용하여 모델의 성능 향상을 가져오는 것으로 보입니다. Peri-LN은 이전에 널리 사용되던 Post-LN 및 Pre-LN보다 더 나은 성능을 발휘하며, 이는 많은 연구자들이 간과하고 있던 부분입니다.

- **Technical Details**: Peri-LN 전략은 각 모듈의 입력과 출력을 정규화하여 hidden-state 변화를 조절하는데 효과적입니다. 기존의 Post-LN 및 Pre-LN 전략과는 다르게, Peri-LN을 적용할 경우 'vanishing gradients'와 'massive activations' 문제를 효과적으로 완화할 수 있습니다. 이 논문은 Peri-LN의 동작 원리와 그 이점을 탐구하며, 그 결과는 대규모 Transformer 아키텍처에서 안정성 향상에 기여할 수 있음을 보입니다.

- **Performance Highlights**: 실험 결과, Peri-LN은 gradient의 흐름을 안정적으로 유지하면서 hidden states의 varaince 성장을 보다 균형 잡힌 방식으로 야기합니다. 이는 3.2B 파라미터를 가진 대규모 Transformer 모델에서 일관된 성능 향상을 보여줍니다. Peri-LN에 대한 실험적 증거는 대규모 모델 훈련의 안정성을 높이며 최적의 LN 배치를 이해하는 데 중요한 기여를 하고 있습니다.



### A Unified Understanding and Evaluation of Steering Methods (https://arxiv.org/abs/2502.02716)
- **What's New**: 이 논문은 대형 언어 모델(LLM)에 적용되는 steering methods의 체계적 분석 및 평가를 위한 통합 프레임워크를 제시합니다. 이들 방법은 모델의 중간 활성화에 steering vectors를 적용하여 원하는 출력을 유도하는 방식으로 작동하며, 재교육 없이도 취소할 수 있는 가능성을 제공합니다. 연구자는 기존 방법들(CAA, RepE, ITI)의 원리를 정리하고, 성능 저하의 원인을 설명하며 이론적 통찰력을 제공합니다.

- **Technical Details**: steering methods는 모델의 다양한 동작을 조정하기 위한 기법으로, 중간 활성화(activation)에 steering vectors를 적용하여 모델이 바람직한 생성물을 출력하도록 유도합니다. 각 기법은 긍정적인 생성물과 부정적인 생성물로 이루어진 대조 쌍을 통해 작동하며, 예를 들어 CAA는 긍정적/부정적 예시의 임베딩 차이를 평균 내어 steering vector를 학습합니다. 연구는 case-by-case 방식으로 steering 방법의 효과를 이해하고 평가하기 위한 기준을 제공합니다.

- **Performance Highlights**: 연구 결과, 평균의 차이(Mean of differences) 방법이 PCA 기반 및 분류기 기반 접근 방식보다 우수함을 보여주었으며, 이는 여러 작업에서 일관된 성능 향상을 입증했습니다. 시각화 결과, PCA 접근법이 steering vector와 직교하는 방향으로 임베딩이 떨어질 경우 성능 저하를 겪는다는 점이 확인되었습니다. 이러한 발견은 steering 방법의 이론적 통찰력과 실제 적용 가능성을 더욱 강조해 줍니다.



### Streaming Speaker Change Detection and Gender Classification for Transducer-Based Multi-Talker Speech Translation (https://arxiv.org/abs/2502.02683)
- **What's New**: 이 논문은 스트리밍 다중 화자 음성 번역을 위한 새로운 방법론을 제안합니다. 저자들은 스피커 임베딩(speaker embeddings)을 활용하여 음성 변환 모델의 성능을 개선하고, 발화자 변화 감지 및 성별 분류를 통합하여 효율적인 실시간 번역을 목표로 합니다. 이러한 접근은 성별 정보가 음성 합성(text-to-speech, TTS) 시스템에서 화자 프로파일 선택에 도움을 줄 수 있음을 강조합니다.

- **Technical Details**: 제안된 접근 방법은 트랜스듀서(transducer) 기반의 스트리밍 음성 번역 모델을 기반으로 하며, 여기에는 세 가지 주요 구성 요소가 포함됩니다: 인코더(encoder), 예측 네트워크(prediction network), 조합 네트워크(joint network). t-vector 방법을 통해 다중 화자 음성 인식 시 스피커 임베딩을 생성하고, 이 임베딩을 사용하여 발화자 변화 및 성별 분류를 수행하는 방식을 설명합니다. 이는 다양한 언어 쌍에 대해 평가되어 높은 정확도를 보여줍니다.

- **Performance Highlights**: 실험 결과, 저자들이 제안한 방법은 발화자 변화 감지와 성별 분류 모두에서 높은 정확성을 달성했다고 보고하였습니다. 특히, 음성에서 실시간으로 정보를 처리하는 스트리밍 시나리오에서 효과적인 성능을 나타냈습니다. 이는 실시간 통신 환경에서 화자 변화 민감도가 높은 zero-shot TTS 모델에서도 적용 가능성을 시사합니다.



### On Teacher Hacking in Language Model Distillation (https://arxiv.org/abs/2502.02671)
- **What's New**: 이 논문은 언어 모델(LM)의 포스트 트레이닝에서 지식 증류(knowledge distillation)의 새로운 현상인 teacher hacking을 조사합니다. Teacher hacking은 학생 모델이 교사 모델을 모방하는 과정에서 발생할 수 있는 부정확성을 이용하는 현상입니다. 저자들은 이러한 현상이 실제로 발생하는지, 그 기준은 무엇인지, 그리고 이를 완화하기 위한 전략을 연구하는 실험 세트를 제안합니다.

- **Technical Details**: 연구팀은 오라클 모델을 기본으로 하는 제어된 실험 세트를 제안했습니다. 이 실험 세트에서는 학생 모델과 오라클 모델 간의 거리와 학생 모델과 교사 모델 간 거리로 각각 '골든 메트릭'과 '프록시 메트릭'을 정의합니다. 학생 모델의 효과적인 훈련을 위해 사용된 데이터의 다양성이 teacher hacking 방지를 위한 주요 요소로 확인되었습니다.

- **Performance Highlights**: 실험 결과, 고정된 오프라인 데이터셋을 사용할 경우 teacher hacking이 발생함을 확인하였으며, 최적화 과정이 다항 수렴 법칙에서 벗어날 때 이를 감지할 수 있음을 밝혔습니다. 반면, 온라인 데이터 생성 기법을 통해 teacher hacking을 효과적으로 완화할 수 있으며, 데이터의 다양성을 활용하는 것이 중요한 요소로 발견되었습니다.



### ParetoQ: Scaling Laws in Extremely Low-bit LLM Quantization (https://arxiv.org/abs/2502.02631)
- **What's New**: 이 논문에서는 quantization(양자화) 모델의 크기와 정확도 간의 최적의 trade-off(절충점)을 찾기 위한 통합된 프레임워크인 ParetoQ를 제안합니다. 이는 1비트, 1.58비트, 2비트, 3비트, 4비트 양자화 설정 간의 비교를 보다 rigorously(엄격하게) 수행할 수 있도록 합니다. 또한, 2비트와 3비트 간의 학습 전환을 강조하며 이는 정확한 모델 성능에 중요한 역할을 합니다.

- **Technical Details**: ParetoQ는 파라미터 수를 최소화하면서도 성능을 극대화할 수 있는 모델링 기법입니다. 실험 결과, 3비트 및 그 이상의 모델은 원래 pre-trained distribution(사전 훈련 분포)과 가까운 성능을 유지하는 반면, 2비트 이하의 네트워크는 표현 방식이 급격히 변화합니다. 또한, ParetoQ는 이전의 특정 비트 폭에 맞춘 방법들보다 우수한 성능을 보여줍니다.

- **Performance Highlights**: 작은 파라미터 수에도 불구하고, ParetoQ의 ternary 600M-parameter 모델은 기존의 SoTA(최신 기술 상태)인 3B-parameter 모델을 넘는 정확도를 기록했습니다. 다양한 실험을 통해 2비트와 3비트 양자화가 사이즈와 정확도 간의 trade-off에서 우수한 성능을 보여주며, 4비트 및 binary quantization 글리 제너럴 단위 성능이 하락하는 경향을 보였습니다.



### SEAL: Speech Embedding Alignment Learning for Speech Large Language Model with Retrieval-Augmented Generation (https://arxiv.org/abs/2502.02603)
- **What's New**: 이번 연구에서는 전통적인 두 단계 방식의 음성 검색 시스템의 한계를 극복하기 위해, Speech-to-Document 매칭을 위한 통합 임베딩 프레임워크를 제안하였습니다. 이 모델은 자동 음성 인식(ASR)과 텍스트 기반 retrieval 간의 중간 텍스트 표현 없이 직접적으로 음성을 문서 임베딩에 매핑하는 방식을 채택합니다. 이를 통해 시스템의 지연 시간을 50% 줄이고, 더 높은 검색 정확도를 달성하였습니다.

- **Technical Details**: 제안된 방법은 음성과 텍스트 각각에 대해 별도의 인코더를 사용하는 구조로, 이후 공유 Scaling 레이어를 통해 두 모달리티를 하나의 공통 임베딩 공간으로 사상합니다. 사전 훈련 단계에서는 Mean Squared Error (MSE) 손실을 최소화하여 음성과 텍스트 간의 정렬을 최적화합니다. 후속 Fine-tuning 단계에서는 다중 작업 혼합 손실을 사용하여 음성 쿼리와 관련 문서 임베딩 간의 정렬을 개선합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 기존의 두 단계 시스템에 비해 50% 더 낮은 파이프라인 지연 시간을 기록하고, 다양한 음향 조건과 화자 변동 속에서도 보다 뛰어난 강건성을 보였습니다. 이러한 성과는 실시간 음성 기반 정보 검색의 가능성을 열어 줍니다. 또한, 음성-문서 매칭의 기존 패러다임에 도전하며, 향후 연구에 대한 기초 자료를 제공합니다.



New uploads on arXiv(cs.IR)

### Investigating Corporate Social Responsibility Initiatives: Examining the case of corporate Covid-19 respons (https://arxiv.org/abs/2502.03421)
Comments:
          7 Tables

- **What's New**: 이 논문은 정책 결정자들이 방대한 양의 정보를 효과적으로 처리할 수 있는 방법을 제시합니다. Latent Dirichlet Allocation, Deep Distributed Representation, 텍스트 요약 방법, Word Based Sentence Ranking, TextRank와 같은 주제 인식 기법을 활용하여 문서의 핵심 내용을 요약할 수 있는 기술적 접근 방식을 다룹니다.

- **Technical Details**: 저자들은 Covid-19 팬데믹 초기 및 중기 시점의 기업 보도자료를 대상으로 인기 있는 NLP(Natural Language Processing) 방법론을 적용하였습니다. 이는 정보의 과부하에 대처하고, 정책 결정 과정에서 관련 문서의 의미를 파악하는 데 도움을 주기 위한 목적이었습니다.

- **Performance Highlights**: 이 연구에서 제시된 접근법은 향후 사회적 결정-making에서 유사한 방법론을 적용할 수 있는 가능성을 보여줍니다. 이는 정책 결정자들이 자원의 효율성을 높이고, 다양한 이해관계자들과의 소통을 개선하는 데 있어 중요한 기초 자료로 활용될 수 있습니다.



### DenseReviewer: A Screening Prioritisation Tool for Systematic Review based on Dense Retrieva (https://arxiv.org/abs/2502.03400)
Comments:
          Accepted at ECIR 2025

- **What's New**: 이번 논문에서는 DenseReviewer라는 새로운 스크리닝 도구를 소개합니다. 이 도구는 의료 시스템 리뷰를 위한 title 및 abstract screening을 진행하는데, 기존의 active learning 방법론을 뛰어넘는 효과적인 dense retrieval 방법을 활용합니다. 또한, 웹 기반의 스크리닝 도구와 연구자들이 새로운 active learning 방법을 개발할 수 있도록 하는 Python 라이브러리가 포함되어 있습니다.

- **Technical Details**: DenseReviewer는 두 가지 모드인 ranking mode와 focus mode를 제공합니다. 사용자는 PubMed에서 가져온 연구 데이터를 nbib 형식으로 업로드하고, PICO 쿼리를 제출하여 밀접한 연구를 효율적으로 평가할 수 있습니다. 이 도구는 스크리너의 피드백을 기반으로 Rocchio 알고리즘을 활용하여 PICO 쿼리를 업데이트하며, Docker 컨테이너로 구축되어 손쉽게 자체 호스팅할 수 있습니다.

- **Performance Highlights**: DenseReviewer는 기존의 logistic regression 및 BERT 기반의 active learning 워크플로우보다 screening에서 더 나은 성능을 보였습니다. 향후에는 PICO 쿼리를 확대하여 사용자들이 제외 기준을 표현하고, LLM을 활용한 자동화된 relevance 판단 기능을 계획하고 있습니다. 이는 검사자의 작업 부담을 줄이고, 스크리닝 프로세스를 더욱 효율적으로 만들기 위한 것입니다.



### Interactive Visualization Recommendation with Hier-SUCB (https://arxiv.org/abs/2502.03375)
- **What's New**: 이 논문에서 제안하는 상호작용형 개인화 시각화 추천(interactive personalized visualization recommendation, PVisRec) 시스템은 사용자 피드백을 통해 이전 상호작용을 학습합니다. 기존의 개인화 시각화 추천 방법은 비상호작용적이며, 새로운 사용자에 대한 초기 데이터에 의존했습니다. 이러한 제한을 극복하기 위해 Hier-SUCB라는 컨텍스트 조합 반밴딧(contextual combinatorial semi-bandit) 알고리즘을 도입하여 보다 정확하고 상호작용적인 추천을 가능하게 합니다.

- **Technical Details**: Hier-SUCB는 시각화 추천 문제에 대한 이론적 개선을 보여주며, 행동 공간(action space)을 좁히고 사용자 피드백을 유연하게 수집하기 위한 계층적 구조(hierarchical structure)를 갖춥니다. 이 알고리즘은 학습 가능한 바이어스(bias) 항을 도입하여 실제 보상(real reward)과 추정 보상(estimated reward) 간의 격차를 줄입니다. 따라서 사용자가 설정한 시각화의 구성(configuration) 및 속성(attributes) 간의 관계를 정확하게 모델링 할 수 있습니다.

- **Performance Highlights**: 다양한 실험에서 Hier-SUCB는 오프라인 방법과 비교했을 때 동등한 성능을 보였으며, 다른 반밴딧 알고리즘을 초월하여 우수한 효율성을 증명했습니다. 이 연구는 개인화 시각화 추천의 실시간 사용자 피드백 기반 학습의 중요성을 강조하며, 초기 데이터의 필요 없이도 사용자 맞춤형 추천을 가능하게 하는 시스템을 제안합니다. 새로운 사용자나 데이터에 직면했을 때도 그들의 선호도를 신속하게 반영하여 추천을 개선하는 혁신적인 접근 방식으로 주목받고 있습니다.



### Intent Representation Learning with Large Language Model for Recommendation (https://arxiv.org/abs/2502.03307)
Comments:
          11 pages, 8 figures

- **What's New**: 이번 논문은 Intent Representation Learning with Large Language Model (IRLLRec) 프레임워크를 제안하여, 텍스트 기반과 상호작용 기반의 의도(Intent)를 함수적으로 통합하고 추천 시스템의 품질을 향상시킵니다. 기존 시스템들이 다루지 못했던 다중 모드 의도와 그 내재적인 차이를 탐구하며, 이들이 가지는 모델 생성의 잠재력을 강조합니다. IRLLRec은 다중 모드의 변환을 통해 추천의 신뢰성을 높이고, 노이즈 문제를 해결하고자 합니다.

- **Technical Details**: IRLLRec 프레임워크는 듀얼 타워 구조를 사용하여 다양한 모드의 의도 표현을 학습합니다. 두 가지 정렬 전략인 쌍(pairwise) 정렬과 변환(translation) 정렬을 통해 텍스트와 상호작용 간의 차이를 최소화하고, 다양한 사용자 행동을 포착하여 추천의 질을 높입니다. 또한, 상호작용-텍스트 일치를 위한 모듈을 설계하여 잠재적인 핵심 의도를 추출하고 두 가지 모드 간의 정확한 매칭을 도모합니다.

- **Performance Highlights**: 세 가지 공공 데이터셋을 통한 실험 결과, IRLLRec 프레임워크는 기존의 기준 모델들에 비해 월등한 성능을 보였습니다. 특히, 의도 정렬 및 매칭이 추천 결과 최적화에 기여하는 방식을 분석하여, 다중 모드 의도의 잠재력을 효과적으로 활용한 것으로 나타났습니다. 이 연구는 추천 시스템의 해석 가능성과 정밀도를 한층 향상시키는 데 기여합니다.



### Data Dams: A Novel Framework for Regulating and Managing Data Flow in Large-Scale Systems (https://arxiv.org/abs/2502.03218)
- **What's New**: 이번 연구에서는 데이터 흐름을 효과적으로 관리하기 위한 새로운 프레임워크인 Data Dam을 소개합니다. 이 프레임워크는 실시간 데이터 흐름의 조절을 통해 데이터 오버플로우를 방지하고 자원 활용도를 극대화하도록 설계되었습니다. 또한, Data Dam은 물리적 댐의 사례를 바탕으로 실시간 프로세싱 아키텍처의 효율성을 증대시키는데 기여합니다.

- **Technical Details**: Data Dam은 데이터 저장소, 슬루스 제어 및 예측 분석을 활용하여 시스템 상태에 따라 동적으로 데이터 흐름을 조절합니다. 이 시스템은 데이터의 유입, 저장 및 유출을 최적화하여 평균 저장 수준을 현저히 줄이고 전체 유출량을 증가시키는 효과를 보여주었습니다. 안정적인 유출률을 유지함으로써 시스템의 효율성을 높이고 오버플로우 위험을 줄이는 방안을 제공합니다.

- **Performance Highlights**: 시뮬레이션 결과, Data Dam 프레임워크는 기존 정적 모델에 비해 평균 저장 수준을 371.68에서 426.27 단위로 줄였으며, 전체 유출량은 7999.99에서 7748.76 단위로 증가시켰습니다. 이는 데이터 흐름을 효과적으로 조절하며 시스템의 병목 현상을 방지하여 성능을 향상시키는 것을 의미합니다. Data Dam은 대규모 분산 시스템에서 동적 데이터 관리의 확장 가능 솔루션을 제시합니다.



### Scientometric Analysis of the German IR Community within TREC & CLEF (https://arxiv.org/abs/2502.03065)
- **What's New**: 이번 연구에서는 2000년부터 2022년까지의 Text Retrieval Conference (TREC) 및 Conference and Labs of the Evaluation Forum (CLEF) 캠페인에 대한 독일 정보 검색(Information Retrieval, IR) 커뮤니티의 영향을 분석했습니다. OpenAlex에서 제공하는 메타데이터 및 GROBID 프레임워크를 통해 추출한 메타데이터를 기반으로 하였습니다. 연구자 및 기관 수준에서의 분석 결과를 제시합니다.

- **Technical Details**: 분석은 기관 및 연구자 수준에서 수행되었으며, 독일 IR 커뮤니티의 기여가 CLEF에 집중됨을 발견했습니다. Lotka's Law에 의해 설정된 생산성 가정이 확인되었습니다. 이러한 데이터 분석은 학술 출판물의 전체 텍스트에서 의미 있는 메타데이터를 추출하여 이루어졌습니다.

- **Performance Highlights**: 독일 IR 커뮤니티는 CLEF에 대한 주된 기여를 표시했으며, 이로 인해 독일의 정보 검색 연구가 국제적 맥락에서 어떤 위치에 있는지를 이해할 수 있는 기회를 제공합니다. 연구 결과는 정보 검색 분야의 정책 결정 및 향후 연구 방향성 설정에 대한 중요한 통찰력을 제공합니다.



### Large Language Models Are Universal Recommendation Learners (https://arxiv.org/abs/2502.03041)
- **What's New**: 이 논문은 대규모 언어 모델(LLMs)을 사용하여 다양한 추천 작업을 통합된 입력-출력 프레임워크에서 수행할 수 있는 보편적인 추천 학습기로 기능할 수 있음을 입증합니다. 이를 통해 전문 모델 설계 없이도 여러 추천 작업을 처리할 수 있으며, 다중 모드 융합 모듈 및 효율적인 후보 생성 방법을 도입하여 LLM의 추천 성능을 향상시켰습니다. 이 연구는 대규모 산업 데이터를 적용하여 전문가 모델과 경쟁력 있는 결과를 달성했음을 보여줍니다.

- **Technical Details**: 논문에서 제안하는 보편적인 추천 모델(URM)은 추천 작업을 LLM 프롬프트 템플릿을 통해 정의하며, 이는 모든 추천 작업을 통합된 입력-출력 형식으로 처리할 수 있습니다. URM은 아이템 표현을 위해 다중 모드 융합 모듈을 활용하고, 단일 피드포워드 패스를 통해 높은 품질의 추천 세트를 생성할 수 있도록 설계되었습니다. 이 모델은 LLM의 강력한 성능을 기반으로 하여, 사용자 행동 데이터를 시퀀스 형태로 변환하여 학습합니다.

- **Performance Highlights**: 연구 결과, URM은 공개 및 산업 규모 데이터셋에서 포괄적인 실험을 수행하였으며, 정량적 및 정성적 성과 모두 훌륭한 성능을 입증했습니다. 이 모델은 제안된 프레임워크를 통해 제로샷(task transfer without retraining) 작업 전이 및 프롬프트 튜닝이 가능하도록 하여, 산업 추천 시스템 내에서의 실용성을 높였습니다. 특히, 산업 애플리케이션에서 추천 정확도를 크게 향상시킬 수 있는 가능성을 보여줍니다.



### FuXi-$\alpha$: Scaling Recommendation Model with Feature Interaction Enhanced Transformer (https://arxiv.org/abs/2502.03036)
Comments:
          Accepted by WWW2025

- **What's New**: 본 논문에서는 새로운 추천 모델인 FuXi-α를 제안합니다. 이 모델은 Adaptive Multi-channel Self-attention 메커니즘을 도입하여 시간적, 위치적, 의미적 특징을 명확히 모델링합니다. 또한 Multi-stage Feedforward Network를 통해 암시적 특징 상호작용을 개선하여 모델의 성능을 높입니다.

- **Technical Details**: FuXi-α 모델은 여러 피처 간의 상호작용을 효율적으로 처리하기 위해 Adaptive Multi-channel Self-attention (AMS) 레이어를 사용합니다. 또한, Multi-stage Feedforward Network (MFFN)을 포함하여 암시적 피처 상호작용을 극대화합니다. 이러한 설계는 모델의 표현력을 향상시키고, 데이터 세트를 기반으로 성능을 입증합니다.

- **Performance Highlights**: 오프라인 실험에서는 FuXi-α가 기존 모델들보다 뛰어난 성능을 보였고, 모델 크기가 증가할수록 성능이 지속적으로 향상되었습니다. 또한, Huawei Music 앱에서 실시한 A/B 테스트에서는 사용자당 평균 재생 곡 수가 4.76%, 평균 청취 시간이 5.10% 증가하는 성과를 보였습니다.



### Assessing Research Impact in Indian Conference Proceedings: Insights from Collaboration and Citations (https://arxiv.org/abs/2502.02997)
- **What's New**: 이번 연구는 Springer's Lecture Notes in Networks and Systems 시리즈에 색인화된 학술 대회 출판물에 대한 검토를 진행했습니다. 특히 인도에서 개최된 177개의 국제 학술 대회를 통해 총 11,066편의 논문이 발표된 것을 분석하였습니다. 연구의 주요 목적은 이러한 학술 대회 출판물의 연구 영향력을 평가하고 기여자를 규명하는 것입니다. 이 연구는 학술 분야에서 연구 품질 향상의 필요성을 강조합니다.

- **Technical Details**: 연구는 Scopus 데이터베이스에서 수집된 학술 대회와 관련된 다양한 문서 정보를 포함하고 있습니다. 총 570회의 국제 학술 대회에서 49,293개의 문서가 확인되었으며, 이 중 인도에서 개최된 학술 대회만을 필터링하여 분석하였습니다. 최종적으로 2019년부터 2024년 사이에 발표된 11,066편의 컨퍼런스 페이퍼(docs) 정보를 포함하여 교류하는 기관의 협업도 분석했습니다. 데이터 수집 과정에서 Google 검색, 학술 대회 웹사이트 및 출판사 웹사이트를 활용했습니다.

- **Performance Highlights**: 연구 결과에 따르면, 모든 발표된 논문의 평균 인용 수는 1.01로 감소세를 보였습니다. 인도 및 국제 공동 저자가 참여한 논문이 평균 인용 수 1.44로 가장 높은 영향을 나타내며, 인도 저자 단독 저자 논문은 평균 0.97로 상대적으로 낮았습니다. 특히, 사립 대학 및 대학교에서 발표된 Indian-collaborated 논문들이 주요 기여 요인으로 밝혀졌습니다. 이러한 결과는 학술 대회 출판물의 연구 영향력에 대한 보다 깊은 논의를 요구합니다.



### FACTER: Fairness-Aware Conformal Thresholding and Prompt Engineering for Enabling Fair LLM-Based Recommender Systems (https://arxiv.org/abs/2502.02966)
- **What's New**: FACTOR는 LLM 기반 추천 시스템을 위한 공정성 인식 프레임워크로, conformal prediction과 동적 프롬프트 엔지니어링을 통합합니다. 이 프레임워크는 편향 패턴이 발생할 경우 공정성 제약을 자동으로 조정하는 메커니즘을 제공합니다. 또한, 역사적 편향 사례를 분석하여 반복적인 인구통계적 편향을 줄이는 적대적 프롬프트 생성기를 개발하였습니다.

- **Technical Details**: FACTOR는 공정성 보장을 통해 semantic variance를 활용하여 추천 시스템의 공정성을 조정합니다. 비표준적 텍스트 출력의 차이를 측정하기 위해 임베딩 기반 분석을 사용하며, 추천 품목의 노출 불균형을 해결하기 위한 지속적인 프롬프트 수정을 포함합니다. Conformal prediction을 통해 예측 오류의 가능성을 관리함으로써 데이터 변동성을 고려하고, 수집된 데이터를 기반으로 적절한 공정성 기준을 설정합니다.

- **Performance Highlights**: 실험 결과, FACTER는 MovieLens와 Amazon 데이터셋에서 공정성 위반을 최대 95.5% 감소시키면서 추천 정확도를 유지하는 데 성공했습니다. 이 연구는 LLM 기반 결정 지원 시스템에 대한 공정성 개념의 확장을 보여주고, 더욱 신뢰할 수 있는 추천 모델 구축의 가능성을 제시합니다.



### Control Search Rankings, Control the World: What is a Good Search Engine? (https://arxiv.org/abs/2502.02957)
Comments:
          Accepted to Springer's AI and Ethics journal on February 4, 2025; 31 pages, 1 figure

- **What's New**: 이 논문은 '좋은 검색 엔진이란 무엇인가?'라는 윤리적 질문을 탐구합니다. 검색 엔진은 전 세계 온라인 정보의 문지기로서 중요한 역할을 함에 따라 윤리적인 기준으로 운영되어야 합니다. 특히 COVID-19와 관련하여 정보 검색의 역사와 기술적 발전을 결합해, 검색 엔진이 수행해야 할 다양한 윤리적 모델을 제시합니다.

- **Technical Details**: 이론적으로, 검색 엔진은 크롤러(crawler), 인덱스(index), 랭커(ranker), 사용자 인터페이스의 네 가지 핵심 구성 요소로 이루어져 있습니다. 사용자 관점에서는 사용자가 정보 수요를 특정한 형태로 변환하는 것이 중요하며, 이는 주로 키워드로 구성된 쿼리를 작성함으로써 이루어집니다. 검색 엔진은 사용자의 쿼리에 대해 점수 기반 원칙에 따라 관련성이 높은 문서를 우선적으로 정렬하여 반환합니다.

- **Performance Highlights**: 이 논문은 정보 검색(Information Retrieval, IR) 분야와 윤리적 관점 간의 간극을 메우기 위해 네 가지 윤리적 모델인 고객 서번트(Customer Servant), 사서(Librarian), 기자(Journalist), 교사(Teacher)를 제안합니다. 이러한 모델들은 검색 엔진의 다양한 행동을 이해하고 설명하는 데 유용하며, 특히 COVID-19 대유행 중의 웹 검색을 사례로 들어 모델의 적용을 보여줍니다. 또한, 향후 규제와 책임 문제에 대한 논의도 포함되어 있습니다.



### TD3: Tucker Decomposition Based Dataset Distillation Method for Sequential Recommendation (https://arxiv.org/abs/2502.02854)
- **What's New**: 본 논문에서는 데이터 중심 인공지능의 발전에 따라 추천 시스템 (Recommendation Systems, RS)에서 데이터 중심 접근 방식이 모델 중심 혁신보다 더 중요해졌음을 강조하고 있습니다. 새로운 방법론인 TD3는 메타 학습 프레임워크 내에서 순차 추천을 위한 고유한 데이터베이스 증류 (Dataset Distillation) 기법을 소개합니다. TD3는 원본 데이터에서 복합적이고 표현적인 합성 시퀀스 요약 (synthetic sequence summary)을 추출하는 방법을 제안하여, 기존의 추천 모델이 직면하는 데이터 양과 질이 문제를 해결하려고 합니다.

- **Technical Details**: TD3는 탁커 분해 (Tucker decomposition)를 사용하여 데이터 요약을 네 가지 요소로 분해합니다: 합성 사용자 잠재 요소 (synthetic user latent factor), 시간적 동역학 잠재 요소 (temporal dynamics latent factor), 공유 아이템 잠재 요소 (shared item latent factor), 그리고 관계 코어 (relation core)입니다. 이를 통해 시퀀스에 대한 연관 관계를 유지하면서도, 데이터의 계산 복잡성을 효과적으로 줄일 수 있습니다. 특히, 각 요소는 2차원 텐서로 표현되며, 아이템 집합 크기가 클 때에도 적합한 방법론입니다.

- **Performance Highlights**: 다양한 공개 데이터셋을 통한 실험 결과, TD3는 뛰어난 성능과 독립적인 아키텍처 일반화 가능성을 입증하였습니다. 제안된 방안은 데이터 중심 추천 시스템에서 기존의 한계를 넘어서며, 더욱 정밀한 데이터 업데이트와 피처 공간 정렬 손실 (feature space alignment loss)을 제공하여 효율성을 높입니다. 이를 통해 사용자 행동 모델링을 위해 필수적인 시퀀스 간 관계의 본질적인 정보를 유지할 수 있음을 강조하고 있습니다.



### Inducing Diversity in Differentiable Search Indexing (https://arxiv.org/abs/2502.02788)
- **What's New**: 이번 연구에서는 Differentiable Search Index (DSI)라는 새로운 정보 검색 방법론을 도입하며, 이것이 오래된 검색 기법에서 어떻게 발전했는지를 보여줍니다. DSI의 핵심은 사용자의 쿼리를 문서에 직접 매핑하는 transformer 기반의 신경망 아키텍처를 사용하여 검색 과정을 간소화하는 것입니다. 또한, 이 모델은 문서 다양성을 높이기 위해 Maximal Marginal Relevance (MMR)에서 영감을 받은 새로운 손실 함수(loss function)를 도입하여 훈련 중에도 관련성(relevance)과 다양성(diversity)을 함께 고려할 수 있습니다.

- **Technical Details**: DSI는 문서 집합에 대한 정보를 신경망의 파라미터로 인코딩하는 정보 검색의 새로운 패러다임입니다. 검색 쿼리에 대해 모델은 관련 문서의 ID를 예측하며, 검색 결과의 다양성을 높이기 위해 MMR에서 유래된 유사성을 고려한 손실 함수를 사용합니다. 이러한 모델은 훈련 중에 데이터의 다양성을 높이고, 결과적으로 관련성과 다양성을 모두 갖춘 문서를 검색할 수 있도록 설계되었습니다.

- **Performance Highlights**: 본 연구의 방법론은 NQ320K 및 MSMARCO 데이터셋을 통해 평가되었으며, 기존 DSI 기법과 비교했을 때 유의미한 성과를 보였습니다. 모델의 다양성을 증가시키는 동시에 검색 결과의 관련성에 미치는 영향이 적음을 입증하였고, 결과적으로 사용자는 보다 폭넓은 주제를 접할 수 있게 됩니다. 이러한 접근은 정보 검색 문제에서 중요성이 높은 서브 토픽 검색(sub-topic retrieval)과 같은 분야에서도 유용하게 사용될 수 있습니다.



### PalimpChat: Declarative and Interactive AI analytics (https://arxiv.org/abs/2502.03368)
- **What's New**: 이번 연구에서는 PalimpChat이라는 새로운 채팅 기반 인터페이스를 도입하여 Palimpzest와 Archytas를 통합한 시스템을 소개합니다. PalimpChat은 사용자가 자연어(Natural Language)로 복잡한 AI 파이프라인을 설계하고 실행할 수 있도록 하여, 다시 말해 비전문가도 기술적으로 접근할 수 있게 만듭니다. 이 시스템은 과학 발견, 법적 발견, 부동산 검색 등 세 가지 실제 시나리오를 제공하며, 관련 데이터셋을 쉽게 탐색하고 처리할 수 있도록 돕습니다.

- **Technical Details**: Palimpzest는 비구조화 데이터 처리 파이프라인을 자동으로 최적화하여 구축하는 선언적(declarative) 시스템으로 구성되어 있습니다. 사용자는 데이터셋과 스키마를 정의한 후, 변환을 지정하여 최종 출력을 형성할 수 있습니다. Archytas는 LLM 에이전트를 생성하여 다양한 도구와 상호작용할 수 있도록 지원하는 도구 상자(toolbox)로, 사용자가 요청을 작은 단계로 분해하여 해결책을 찾는 데 도움을 줍니다.

- **Performance Highlights**: PalimpChat은 비전문가 사용자가 침착하게 파이프라인을 설계하고 실행할 수 있는 환경을 제공합니다. 또한, 전문가 사용자는 생성된 코드를 수정하거나 직접 Palimpzest 내에서 파이프라인을 프로그래밍할 수 있는 유연성을 가지게 됩니다. 이 시스템을 통해 사용자는 과학적 연구 등 다양한 분야에서 실제 데이터를 기반으로 한 복잡한 분석 작업을 수행할 수 있는 가능성을 탐색할 수 있습니다.



New uploads on arXiv(cs.CV)

### Seeing World Dynamics in a Nutsh (https://arxiv.org/abs/2502.03465)
- **What's New**: 본 논문에서는 casually captured monocular videos를 효과적으로 공간적 및 시간적으로 일관성을 유지하면서 표현하기 위한 문제를 다룹니다. 기존의 2D 및 2.5D 기술들은 복잡한 모션, 가리기 및 기하학적 일관성을 처리하는 데 어려움을 겪고 있으며, 본 연구는 동적 3D 세계의 투영으로서 단안 비디오를 이해하고자 합니다. NutWorld라는 새로운 프레임워크를 제안하여, 단일 전방 패스를 통해 단안 비디오를 동적 3D Gaussian 표현으로 변환합니다.

- **Technical Details**: NutWorld의 핵심은 정렬된 공간-시간 Gaussians(STAG) 표현을 통해 포즈 최적화 없이 장면 모델링을 가능하게 하는 것입니다. 이 프레임워크는 효과적인 깊이와 흐름 정규화를 포함하여, 단안 비디오의 고유한 3D 형태를 모델링합니다. 세 개의 주요 구성 요소로 이루어져 있으며, 각 구성 요소는 비디오 프레임 간의 공간-시간 대응 및 동적 특성을 학습하고 변환하는 데 중점을 두고 있습니다.

- **Performance Highlights**: NutWorld는 RealEstate10K 및 MiraData 실험을 통해 비디오 재구성의 효율성을 입증하고, 다양한 비디오 다운스트림 애플리케이션에서도 실시간 성능을 유지할 수 있는 뛰어난 유연성을 보여줍니다. 이 프레임워크는 새로운 뷰 합성, 일관된 깊이 추정, 비디오 세분화 및 비디오 편집과 같은 다양한 작업을 지원하며, 비디오 표현의 범용성 가능성을 제시합니다.



### SKI Models: Skeleton Induced Vision-Language Embeddings for Understanding Activities of Daily Living (https://arxiv.org/abs/2502.03459)
- **What's New**: 본 논문에서는 Skeleton Induced 모델(SKI models)을 소개하며, 3D skeleton을 비전-언어 임베딩 공간에 통합하여 인간 행동 인식 및 비디오 캡션 생성의 소제로 학습할 수 있도록 합니다. SKI 모델은 SkeletonCLIP이라는 스켈레톤-언어 모델을 활용하여 VLM(비전 언어 모델)과 LVLM(대형 비전 언어 모델)에 스켈레톤 정보를 주입합니다. 특히, SKI 모델은 추론 중에 스켈레톤 데이터가 필요하지 않아 실제 적용에서의 강건함을 높입니다.

- **Technical Details**: SKI-VLM은 스켈레톤-언어 모델인 SkeletonCLIP Distillation (SCD)을 활용하여 3D 스켈레톤 정보를 VLM 공간에 통합한 새로운 접근 방식입니다. SKI-LVLM은 다양한 비디오 이해 능력을 향상시키기 위해 언어 기반 3D 스켈레톤 특징을 LDLM에 포함시킵니다. 이 모델들은 비디오의 다양한 모달리티를 학습하여 ADL(일상 생활 활동)에 특화된 기존의 문제를 개선할 수 있습니다.

- **Performance Highlights**: SKI-VLM은 NTU60 및 NTU120 데이터셋에서 제로 샷 행동 인식에서 우수한 성능을 보여줍니다. 또한, SKI-LVLM은 Charades 데이터셋에서 복잡한 캡션 생성을 위한 테스트에서 baseline 모델보다 더 나은 성능을 기록하여 스켈레톤 정보를 LVLM에 통합하는 중요성을 강조합니다. 본 연구는 비디오 표현 학습을 위해 스켈레톤 정보를 통합하여 비전-언어 임베딩을 강화하는 첫 번째 연구로 기록됩니다.



### Dress-1-to-3: Single Image to Simulation-Ready 3D Outfit with Diffusion Prior and Differentiable Physics (https://arxiv.org/abs/2502.03449)
Comments:
          Project page: this https URL

- **What's New**: 본 논문은 이미지에서 3D 의류 모델을 복원하는 혁신적인 파이프라인인 Dress-1-to-3를 소개합니다. 이 접근 방식은 실제 이미지에서 분리된 의류와 인간 모델을 생성하여 시뮬레이션 준비가 완료된 3D 의류를 제작하는데 초점을 맞추고 있습니다. 기존의 건조한 형상을 넘어, 사용자 맞춤화된 의류 애니메이션과 등 다양한 다운스트림 작업을 가능하게 합니다.

- **Technical Details**: 이 작업은 2D 다중 시점(diffusion) 모델과 3D 봉제 패턴 재구성을 결합하여 의류 최적화를 위한 통합된 IPC(Implicit Pattern Control) 프레임워크를 제안합니다. 이를 통해 다양한 시점에서 생성된 이미지를 사용하여 3D 봉제 패턴을 정제하고, 기존 데이터 세트의 제약을 극복할 수 있는 가능성을 보여줍니다. 최적화를 통해 향상된 기하학적 정렬로 더욱 다양하고 상세한 재구성을 달성할 수 있습니다.

- **Performance Highlights**: 실험을 통해, Dress-1-to-3는 입력 이미지를 기반으로 한 다양한 카테고리의 의류를 성공적으로 재구성할 수 있으며, 이는 이전의 방법들과 비교해 개선된 성능을 입증합니다. 본 연구는 또한 텍스처 생성 모듈과 인간 동작 생성 모듈을 통합하여, 물리적으로 타당하고 현실적인 동적 의류 시연을 생성하는 데 중점을 두고 있습니다. 이로써, 의류 산업에서 향후 활용될 가능성이 매우 높습니다.



### Masked Autoencoders Are Effective Tokenizers for Diffusion Models (https://arxiv.org/abs/2502.03444)
- **What's New**: 이 연구에서는 MAETok이라는 새로운 autoencoder를 제안하여, 기존의 diffusion 모델에서 latent space의 구조를 개선하는 방법에 대해 탐구합니다. MAETok은 mask modeling을 통해 의미론적으로 풍부한 latent space를 학습하면서도 높은 재구성 충실도를 유지합니다. 핵심 발견은, diffusion 모델의 성능이 variational constraints보다 latent space의 구조와 더 밀접하게 연관되어 있다는 것입니다. 실제 실험을 통해 MAETok은 128개의 토큰만으로도 최신 기술 성능을 달성함을 입증했습니다.

- **Technical Details**: MAETok은 Masked Autoencoder (MAE)라는 자기 지도 학습 패러다임을 적용하여, 성능 저하 없이 더 일반화되고 구분된 표현을 발견할 수 있도록 설계되었습니다. 이 구조는 encoder에서 임의로 마스킹된 이미지를 재구성하며, pixel decoder와 auxillary shallow decoders를 사용하여 숨겨진 토큰의 특징을 예측합니다. 그 결과, 높은 재구성 충실도와 함께 강력한 의미론적 표현을 학습할 수 있었습니다. 실험을 통해 MAETok은 단순한 autoencoders에 비해 훈련 속도와 생성 속도가 극적으로 개선된 것을 확인했습니다.

- **Performance Highlights**: MAETok는 ImageNet 벤치마크에서 256×256 및 512×512의 해상도에서 각각 128개의 토큰만으로도 개선된 generation FID(gFID) 성능을 보여 주었습니다. 특히, 675M 파라미터의 diffusion 모델은 512 해상도에서 gFID 1.69와 304.2 IS를 달성하며, 기존 모델을 초월하는 성능을 기록했습니다. MAETok의 결과는 diffusion 모델의 훈련 및 생성 성능을 크게 향상시켜, 효율적인 해상도 생성에 기여하고 있음을 보여줍니다.



### A Temporal Convolutional Network-Based Approach and a Benchmark Dataset for Colonoscopy Video Temporal Segmentation (https://arxiv.org/abs/2502.03430)
- **What's New**: 이번 연구에서는 콜론 비디오의 전체 프로시저를 주해할 수 있는 최초의 공개 데이터셋인 REAL-Colon을 구축하였습니다. 총 2.7백만 프레임을 포함한 60개의 전체 콜론 비디오 세트를 주석 처리하여 해부학적 위치 및 절차 단계에 대한 프레임 수준 레이블을 제공하였습니다. 또한, 새로운 모델인 ColonTCN을 제안하여 콜론 비디오에서 긴 시간 의존성을 효율적으로 포착하는 맞춤형 Temporal Convolutional Block을 활용하였습니다.

- **Technical Details**: 본 연구의 ColonTCN 아키텍처는 커스텀 Temporal Convolutional Blocks를 사용하여 콜론 비디오의 시간적 분할을 수행합니다. 이 모델은 k-fold cross-validation 평가 프로토콜을 통해 평가되었으며, 새로운 다중 센터 데이터에서 성능을 측정합니다. 또한, 모델은 두 가지 proposed k-fold cross-validation 설정에서 평가될 때 경쟁 모델을 초월하는 분류 정확도를 달성합니다.

- **Performance Highlights**: ColonTCN은 최신 모델들을 비교했을 때 뛰어난 분류 정확도를 자랑하며, 적은 수의 매개변수로도 높은 효율성을 유지합니다. 또한, ablation studies를 통해 이 작업의 도전 과제를 강조하고, 커스텀 Temporal Convolutional Blocks가 학습을 향상시키고 모델 효율성을 개선함을 보여줍니다. 이러한 발전이 콜론 절차의 시간적 분할 분야에서 중요한 진전을 나타내며, 향후 클리닉에서의 필요성을 충족할 수 있는 연구를 촉진할 것으로 기대합니다.



### TruePose: Human-Parsing-guided Attention Diffusion for Full-ID Preserving Pose Transfer (https://arxiv.org/abs/2502.03426)
- **What's New**: 본 논문에서는 Pose-Guided Person Image Synthesis (PGPIS) 분야의 한계를 해결하기 위해 새로운 접근법인 human-parsing-guided attention diffusion을 제안합니다. 기존의 diffusion-based PGPIS 방법들이 얼굴 특징을 효과적으로 보존하는 반면, 의류 세부사항을 제대로 유지하지 못하는 문제를 발견했습니다. 이를 해결하기 위해, 사전 훈련된 Siamese 네트워크 구조와 인체 파싱 정보를 활용하여 얼굴과 의류의 시각적 특징을 동시에 잘 보존할 수 있는 방법론을 소개하고 있습니다.

- **Technical Details**: 제안된 방법론은 주로 세 가지 주요 구성 요소로 이루어져 있습니다: dual identical UNets (TargetNet과 SourceNet), 인체 파싱 기반의 융합 주의(attention) 모듈인 HPFA(human-parsing-guided fusion attention), 그리고 CLIP-guided attention alignment (CAA)입니다. 이러한 구성 요소들은 이미지 생성 과정에서 사람의 얼굴과 옷의 패턴을 효과적으로 통합하여 고품질 결과를 생성하는 데 기여합니다. 특히, HPFA와 CAA 모듈은 특정 의복 패턴을 타겟 이미지 생성에 적응적으로 사용할 수 있도록 합니다.

- **Performance Highlights**: 다양한 실험을 통해 제안된 방법이 의류 리트리벌 벤치마크와 최신 인간 편집 데이터셋에서 13개의 기존 기법 대비 상당한 우위를 보여주었습니다. 결과적으로, 제안한 네트워크는 얼굴 외관과 의류 패턴을 잘 보존하며 고품질 이미지를 생성할 수 있음을 입증했습니다. 이는 패션 산업 내 저작권 보호를 위한 의류 스타일 유지 측면에서 매우 중요합니다.



### Concept Based Explanations and Class Contrasting (https://arxiv.org/abs/2502.03422)
- **What's New**: 이 논문에서는 깊은 신경망의 예측을 개념 기반으로 설명하는 새로운 방법을 제안합니다. 이 방법은 개별 클래스의 예측을 설명하며, 두 클래스 간의 대비도 가능하게 합니다. 특히, 모델이 특정 클래스를 왜 선택했는지를 이해하는 데 도움을 줍니다. 방법의 신뢰성을 테스트하기 위해 여러 개의 공개 분류 모델과 종양을 감지하기 위한 세그멘테이션 모델에서 성능을 검증하였습니다.

- **Technical Details**: 제안된 방법은 속성(attribution)을 사용하여 활성화 값을 점수화한 후, 해당 속성에 기반한 개념을 추출합니다. 예를 들어, ResNet50 모델을 대상으로 한 실험에서 모델이 클래스 'A'를 예측하지 않은 이미지들에서 추출한 크롭(crop) 이미지를 결합하여 71%의 정확도로 클래스 'A'를 예측하는 성능을 보여주었습니다. 이 과정은 기존의 CRAFT 및 ACE 방법과는 다르게, 개념을 추출한 후 점수를 매기지 않고 모든 추출된 개념이 관련 있다고 가정합니다.

- **Performance Highlights**: 제안된 방법은 ResNet50 및 ResNet34 같은 여러 모델을 평가하여 강력한 성능을 보였습니다. 예를 들어, 클래스 'A'를 묘사하는 여섯 가지 주요 특징을 시각적으로 나타내어 모델의 예측을 설명하는 데 성공했습니다. 본 연구는 현존하는 다양한 해설 방법들과 비교했을 때, 보다 정확하고 이해하기 쉬운 모델 해석을 제공해줍니다.



### Can Text-to-Image Generative Models Accurately Depict Age? A Comparative Study on Synthetic Portrait Generation and Age Estimation (https://arxiv.org/abs/2502.03420)
- **What's New**: 이 연구에서는 텍스트 기반 이미지를 생성하는 모델의 효과성을 종합적으로 분석하며, 특히 인구통계학적 특성을 반영한 합성 초상화를 제작하고 그 정확도를 검토합니다. 212개 국가, 30개의 연령대(10세에서 78세 사이), 그리고 성별을 고루 반영한 프롬프트를 사용하여 생성된 이미지를 평가하였습니다. 결과적으로 현재의 합성 데이터는 연령 관련 작업에서 요구되는 높은 정확도를 제공하기에 부족할 수 있으며, 보다 심도 있는 데이터 필터링과 큐레이션이 필요하다는 점을 시사합니다.

- **Technical Details**: 본 연구에서는 3개의 텍스트-이미지 생성 모델(FLUX.1-dev, Stable Diffusion 3.5, SDXL Epic Realism)을 사용해 12,960개의 프롬프트에 대한 이미지를 생성했습니다. 각 프롬프트는 국적, 나이 및 성별을 포함하여 1,024 픽셀의 고해상도를 유지하는 방식으로 설정되었습니다. 이 연구는 생성 모델이 인구통계적 다양성을 처리하는 방식과 관련하여 공정성, 편향 및 표현의 정확성에 대한 통찰을 제공합니다.

- **Performance Highlights**: 결과적으로 Epic Realism 모델이 낮은 평균 절대 오차(MAE)와 높은 상관계수를 달성하여 의도한 나이와의 정렬에서 우수한 성능을 보였습니다. 반면, FLUX 모델은 높은 MAE 값을 기록하며 생성된 이미지의 연령 불일치가 두드러졌습니다. 특히 청소년 및 젊은 성인 프롬프트는 낮은 절대 오차를 보였으나, 60세 이상의 연령대에서는 더 큰 도전 과제가 있음을 확인했습니다.



### Deep Learning-Based Approach for Identification of Potato Leaf Diseases Using Wrapper Feature Selection and Feature Concatenation (https://arxiv.org/abs/2502.03370)
- **What's New**: 이번 연구는 감자 식물에 영향을 미치는 Late Blight 질병을 감지하기 위해 이미지 처리(image processing)와 머신러닝(machine learning)을 활용한 자동화된 방법을 제안합니다. 이러한 접근법은 감자 재배의 수확량을 향상시키기 위한 빠르고 효과적인 솔루션을 제공할 수 있습니다. 이 연구는 여러 단계로 구성되어 있으며, 각 단계에서 기술적인 혁신이 이루어졌습니다.

- **Technical Details**: 제안된 방법은 네 가지 주요 단계로 나뉘어 있습니다. 첫 번째 단계에서는 Histogram Equalization을 통해 입력 이미지의 품질을 향상시키고, 두 번째 단계에서는 Deep CNN 모델을 사용하여 특징(feature)을 추출합니다. 세 번째 단계에서는 wrapper 기반(feature selection) 특징 선택 방법을 적용하며, 마지막 단계에서는 SVM 분류기(classifier)를 사용하여 감지 결과를 분류합니다. 이 방법은 550개의 특징을 선택하여 높은 정확도를 나타냅니다.

- **Performance Highlights**: 제안된 방법은 SVM을 사용하여 99%라는 높은 정확도를 달성했습니다. 이는 Early Blight와 Late Blight 질병의 조기 감지를 통해 감자 재배의 효율성을 크게 개선할 수 있는 가능성을 보여줍니다. 이러한 높은 성능은 특히 농업 분야에서 AI 기술의 적용 가능성을 더욱 넓힐 것입니다.



### GHOST: Gaussian Hypothesis Open-Set Techniqu (https://arxiv.org/abs/2502.03359)
Comments:
          Accepted at AAAI Conference on Artificial Intelligence 2025

- **What's New**: 본 논문에서는 Open-Set Recognition (OSR)에서 각 클래스에 대한 성과의 변동성을 강조하며, 이를 해결하기 위한 새로운 알고리즘인 Gaussian Hypothesis Open Set Technique (GHOST)를 소개합니다. GHOST는 클래스별 다변량 Gaussian 분포를 사용한 하이퍼파라미터 없는 알고리즘으로, 각 클래스의 DNN 내 특징을 따로 모델링하여 공정한 평가를 추진합니다. 또한, 이 알고리즘은 원래 모델의 예측과는 달리, 특정 클래스의 성과를 독립적으로 평가하는 데 초점을 맞춰 공정성을 향상시킵니다.

- **Technical Details**: GHOST는 DNN의 분포를 다차원 Gaussian 모델로 설명하며, 클래스별로 대각 공분산 행렬을 사용하여 각 클래스의 특성을 구별하고 관리합니다. 이러한 접근법은 네트워크가 미지의 샘플에 대해 과도한 자신감을 가지지 않도록 하여, 전체 성능을 향상시키면서도 클래스를 공정하게 평가하는 데 기여합니다. 기존 알고리즘과 달리 GHOST는 하이퍼파라미터를 필요로 하지 않기 때문에 사용자가 보다 쉽게 OSR 기법을 적용할 수 있게 합니다.

- **Performance Highlights**: GHOST는 여러 ImageNet-1K로 사전 학습된 DNN을 테스트하며, AUOSCR, AUROC, FPR95와 같은 표준 메트릭을 사용하여 통계적으로 유의미한 성과 개선을 보여줍니다. 실험 결과, GHOST는 대규모 OSR 문제에 대한 최신 기술의 기준을 높임으로써 성과를 명확하게 개선했습니다. 공정성을 평가한 최초의 분석을 통해, GHOST가 OSR에서 성과의 불균형을 해소하고 각 클래스의 문제점을 신속하게 파악할 수 있도록 돕는 것을 입증하였습니다.



### RadVLM: A Multitask Conversational Vision-Language Model for Radiology (https://arxiv.org/abs/2502.03333)
Comments:
          21 pages, 15 figures

- **What's New**: 이번 연구에서는 RadVLM이라는 다중 작업 대화형 기초 모델을 개발하여, 흉부 X-레이(CXR) 해석을 지원합니다. 이 모델은 100만 개 이상의 이미지-지침 쌍으로 구성된 대규모 지침 데이터셋을 활용하여 훈련되었습니다. 특히, 단일 턴 작업뿐만 아니라 다중 턴 대화 작업까지 지원하며, 의료 진단 과정의 효율성을 높일 수 있습니다.

- **Technical Details**: RadVLM은 시각-언어 아키텍처를 기반으로 하며, José Vaswani가 발표한 transformer 구조에 뿌리를 두고 있습니다. 이 모델은 CXR 해석을 위해 설계된 다양한 작업을 지원하며, 각 작업에 맞는 이미지를 기반으로 한 지침 쌍과 대화 쌍을 사용하여 훈련됩니다. 이를 통해 RadVLM은 단순한 사용자인터페이스 내에서 직관적이고 사용자 친화적인 대화 능력을 제공합니다.

- **Performance Highlights**: RadVLM은 대화형 작업과 시각적 접지 능력에서 최고의 성능을 발휘하며, 다른 기존의 시각-언어 모델과 비교해도 경쟁력을 가지고 있습니다. 특히, 제한된 레이블링된 데이터 환경에서 여러 작업의 공동 훈련의 이점을 강조하는 실험 결과가 나타났습니다. 이러한 결과는 RadVLM이 의료 직업에서 효과적이고 접근 가능한 진단 워크플로우를 지원하는 AI 비서로서의 잠재력을 보여줍니다.



### Deep Learning-based Event Data Coding: A Joint Spatiotemporal and Polarity Solution (https://arxiv.org/abs/2502.03285)
- **What's New**: 이 논문은 Neuromorphic vision sensors, 즉 이벤트 카메라(even cameras)의 새로운 압축 방안을 제안합니다. 기존의 손실 없는(lossless) 코딩 방식 대신, 손실 있는(lossy) DL-JEC를 사용해 단일 포인트 클라우드(single-point cloud) 표현으로 이벤트 데이터를 코딩하는 방법을 소개합니다. 이 방식은 이벤트 데이터의 시공간(spatiotemporal) 및 극성(polity) 정보 간의 상관관계를 활용할 수 있게 합니다.

- **Technical Details**: 제안된 DL-JEC는 이벤트 데이터의 상대적 극성에 따라 두 개의 포인트 클라우드를 사용하는 대신 단 하나의 포인트 클라우드를 통해 데이터를 처리합니다. 이를 통해 기존의 포인트 클라우드 코딩 솔루션을 활용할 수 있으며, 적응형 복셀 이진화(adaptive voxel binarization) 전략을 사용하여 타겟 작업에 최적화된 성능을 달성합니다. 이 접근법은 매우 높은 시각적 해상도 및 낮은 대기 시간을 요구하는 컴퓨터 비전(computer vision) 작업에 적합합니다.

- **Performance Highlights**: DL-JEC는 손실 없는 코딩 솔루션과 비교해도 시각적 성능 저하 없이 상당한 압축 성능 향상을 보여줍니다. 특히, 이벤트 분류(event classification) 작업에서 기존의 전통적인 및 딥러닝 기반의 최첨단(event data coding) 솔루션보다 월등한 성능을 획득할 수 있음을 입증하였습니다. 이러한 결과는 실시간 데이터 처리가 필요한 다양한 애플리케이션에서 DL-JEC의 유망한 활용 가능성을 시사합니다.



### ZISVFM: Zero-Shot Object Instance Segmentation in Indoor Robotic Environments with Vision Foundation Models (https://arxiv.org/abs/2502.03266)
- **What's New**: 이번 논문에서는 서비스 로봇이 비구조적 환경에서 알려지지 않은 객체를 효과적으로 인식하고 분할하도록 돕기 위해 새로운 접근법인 ZISVFM을 제안합니다. 기존의 UOIS(Unseen Object Instance Segmentation) 방법들이 직면했던 시뮬레이션과 현실 간의 격차를 극복하기 위해, 제안된 방식은 segment anything model(SAM)의 제로샷(zero-shot) 능력과 자가 감독(self-supervised) 비전 트랜스포머(ViT)의 시각적 표현을 통합합니다. 이 방법은 세 가지 단계로 구성되어 있으며, 최종적으로 더욱 정확한 객체 분할 및 인식 성능을 보장합니다.

- **Technical Details**: ZISVFM 방법론은 첫 번째로 SAM을 이용해 색칠된 깊이 이미지를 기반으로 객체 비(非)의존적인 마스크 제안(object-agnostic mask proposals)을 생성합니다. 두 번째 단계에서는 DINOv2로 훈련된 ViT의 주의(attention) 맵을 활용하여 비객체 마스크를 제거합니다. 마지막으로 K-Medoids 클러스터링을 통해 포인트 프롬프트(point prompts)를 생성하여 SAM의 정확한 객체 분할을 안내합니다. 이러한 단계를 통해 SAM 및 ViT의 강점을 효과적으로 활용하여 객체를 정확하게 분할할 수 있습니다.

- **Performance Highlights**: 실험 결과, ZISVFM은 두 개의 벤치마크 데이터셋 및 복잡한 다층 환경에서 수집한 자가 수집 데이터셋에서 기존 방법들과 비교하여 우수한 성능을 보였습니다. 특히, ZISVFM은 복잡한 환경과 물체가 혼재된 상황에서도 높은 정확도로 객체 분할을 수행하며, 이는 로봇이 실세계 환경에서 미지의 객체를 효과적으로 잡을 수 있는 가능성을 보여줍니다. 이러한 결과는 ZISVFM의 효과성과 실용성을 입증하는 중요한 사례로, 서비스 로봇의 응용 가능성을 크게 확대합니다.



### Long-tailed Medical Diagnosis with Relation-aware Representation Learning and Iterative Classifier Calibration (https://arxiv.org/abs/2502.03238)
Comments:
          This work has been accepted in Computers in Biology and Medicine

- **What's New**: 본 논문에서는 Long-tailed Medical Diagnosis (LMD) 프레임워크를 제안하여 긴 꼬리(long-tailed) 의료 데이터셋에서 균형 잡힌 의료 이미지를 분류하는 접근 방식을 소개하고 있습니다. 이 프레임워크는 Relation-aware Representation Learning (RRL) 스킴을 통해 특징 표현을 향상시키고, Iterative Classifier Calibration (ICC) 스킴을 통해 분류기를 반복적으로 보정합니다. 특히, 저수 샘플에 대한 성능 향상을 목표로 하고 있습니다.

- **Technical Details**: LMD는 첫 번째 단계에서 RRL을 통해 서로 다른 데이터 증강(augmentation)을 통해 내재적 의미적 특징을 학습할 수 있도록 인코더의 표현 능력을 향상시킵니다. 두 번째 단계에서는 Expectation-Maximization 방식으로 분류기를 보정하기 위한 ICC 스킴을 제안하며, 가상 기능의 생성 및 인코더의 미세 조정(fine-tuning)을 포함합니다. 이러한 방법을 통해 다수 클래스에서의 진단 지식은 유지하면서 소수 클래스를 보정할 수 있습니다.

- **Performance Highlights**: 세 가지 공개 긴 꼬리 의료 데이터셋에 대한 포괄적인 실험 결과, LMD 프레임워크는 최신 기술(state-of-the-art)들을 훨씬 능가하는 성능을 보였습니다. 특히, LMD는 rare disease(희귀 질병)에 대한 분류 성능을 현저히 개선하였으며, Virtual Features Compensation (VFC)와 Feature Distribution Consistency (FDC) 손실을 통해 균형 잡힌 특징 분포를 유지하며 학습합니다.



### Efficient Vision Language Model Fine-tuning for Text-based Person Anomaly Search (https://arxiv.org/abs/2502.03230)
Comments:
          Accepted by 2025 WWW Workshop on MORE

- **What's New**: HFUT-LMC 팀이 제안한 해결책은 WWW 2025의 Text-based Person Anomaly Search (TPAS) 도전과제를 해결하기 위한 것입니다. 이 도전과제의 주된 목표는 많은 보행자 이미지 라이브러리에서 정상적 혹은 비정상적인 행동을 보이는 보행자를 정확히 식별하는 것입니다. 기존의 비디오 분석 작업과는 달리, TPAS는 텍스트 설명과 시각 데이터 간의 미묘한 관계를 이해하는 데 큰 비중을 둡니다. 모델은 기초적으로 유사한 설명으로 인한 검색 결과의 차별화를 효과적으로 해결하기 위해 유사성 범위 분석(Similarity Coverage Analysis, SCA) 전략을 도입하였습니다.

- **Technical Details**: TPAS는 텍스트 설명을 질의로 사용하여 대규모 이미지 라이브러리에서 일치하는 보행자 이미지를 검색하는 것을 목표로 합니다. 이전의 Text-based Person Search (TPS)와 비교했을 때, TPAS는 행동의 비정상성을 식별하는 데 중점을 두고 있으며, 행동 정보를 간과했던 TPS의 한계를 보완합니다. SCA 전략은 유사한 텍스트 설명에 대한 모델의 인식의 어려움을 해결하기 위해 신뢰도 분석을 수행하며, 서로 유사한 답변의 신뢰도를 비교하여 효과적으로 차별화할 수 있는 능력을 향상시킵니다.

- **Performance Highlights**: TPAS 도전과제에서 제안된 솔루션은 테스트 세트에서 Recall@1 점수 85.49를 달성하였습니다. 이는 모델의 기초 성능을 고도화하고, 유사한 설명을 더 잘 처리할 수 있는 능력을 보여주는 결과입니다. 또한, X-VLM 모델을 PAB 데이터셋으로 파인 튜닝하여 차별적 인식 능력을 강화하였습니다. SCA 전략을 통해 모델의 정확성과 신뢰성을 높였으며, 실험 결과는 우리의 접근 방식의 효과성을 잘 나타냅니다.



### A Unified Framework for Semi-Supervised Image Segmentation and Registration (https://arxiv.org/abs/2502.03229)
Comments:
          Accepted for publication at IEEE International Symposium on Biomedical Imaging (ISBI) 2025

- **What's New**: 이번 논문에서는 의료 이미지 분할에 대한 새로운 반지도 학습 접근 방식을 소개합니다. 기존의 방법들은 주로 라벨링된 데이터에서 특징을 추출하는 데 집중했지만, 우리는 이미지 등록 모델을 포함하여 비라벨 데이터에 대한 고품질의 유사 라벨을 생성하는 새로운 방안을 제안합니다. 이를 통해 모델 훈련의 질을 효과적으로 개선하며, 단 1%의 라벨링된 데이터만으로도 우수한 성능을 발휘할 수 있음을 입증했습니다.

- **Technical Details**: 제안된 방법은 분할(segment) 및 등록(register) 프레임워크를 통합하여 ‘소프트 유사 마스크 생성’을 새로운 구성 요소로 추가합니다. 이 방법은 라벨링된 이미지를 기반으로 훈련된 U-Net 모델과 비지도 등록 모델이 서로를 보완하며 진화하도록 설계되었습니다. 소프트 유사 마스크는 픽셀값이 0에서 1 사이의 확신 점수로 표현되며, 반복적으로 생성되어 최종 예측에서 영향을 미치게 됩니다.

- **Performance Highlights**: 2D 뇌 데이터 세트에서의 실험 결과, 제안된 방법이 기존 반지도 학습 방법보다 뛰어난 성능을 보여주었습니다. 특히 라벨링된 데이터가 적은 상황에서도 더욱 우수한 분할 성능을 달성했습니다. 이는 소프트 유사 마스크 생성 방법이 잘 구현되었음을 나타내며, 다양한 의료 이미지 데이터에 적용 가능성을 높이고 있습니다.



### MotionAgent: Fine-grained Controllable Video Generation via Motion Field Agen (https://arxiv.org/abs/2502.03207)
- **What's New**: MotionAgent의 도입으로, 텍스트에 기반한 이미지-비디오 생성에서 세밀한 모션 제어가 가능해졌다. 이 시스템은 텍스트 프롬프트에서 모션 정보를 추출하여 명시적인 모션 영역으로 변환한다. 또한, 이 기술은 다차원적인 모션 표현을 통합하여 고품질 비디오 생성에 기여하며, 기존의 모델들과 비교하여 카메라 모션의 제어 정확도를 크게 개선하였다.

- **Technical Details**: MotionAgent는 모션 필드 에이전트를 사용하여 객체 이동과 카메라 모션을 텍스트에 기반하여 추출하고, 이를 객체 궤적(object trajectories)과 카메라 외적(camera extrinsics)으로 변환하는 과정을 포함한다. 이 두 가지 중간 표현은 3D 공간에서 통합되어 분석적 광학 흐름 구성 모듈을 통해 단일 광학 흐름(optical flow)으로 프로젝션 된다. 마지막으로, 광학 흐름 어댑터(optical flow adapter)가 통합된 흐름을 조건으로 사용하여 기본 이미지-비디오 확산 모델을 제어하여 비디오를 생성한다.

- **Performance Highlights**: MotionAgent는 공개된 I2V 생성 벤치마크에서 실험을 수행한 결과, 카메라 모션의 제어 정확도를 현저하게 개선하고, 비디오 품질에 있어 다른 고급 모델들과 대등한 수준을 달성하였다. 비디오 텍스트 모션 메트릭스에서의 성과는 생성된 비디오와 텍스트의 모션 정보 정렬 평가를 통해 입증되었으며, 사용자 연구 결과에서도 생성된 비디오가 텍스트의 모션 정보와 잘 일치하고 높은 품질을 유지하는 것으로 나타났다.



### MaxInfo: A Training-Free Key-Frame Selection Method Using Maximum Volume for Enhanced Video Understanding (https://arxiv.org/abs/2502.03183)
- **What's New**: 최근 비디오 대형 언어 모델(VLLMs)의 접근 방식인 MaxInfo는 최대 볼륨 원칙(maximum volume principle)을 기반으로 하여 입력 비디오에서 가장 대표적인 프레임을 선택하고 유지하는 새로운 방법을 제안합니다. 이 방법은 중복을 줄이면서도 정보가 많은 영역을 포함하도록 프레임 선택을 최적화합니다. MaxInfo는 기존 모델에 간편하게 통합되며 별도의 훈련이 필요 없어 실용적인 해결책입니다.

- **Technical Details**: MaxInfo는 입력 비디오의 프레임 임베딩 행렬에서 최대 볼륨을 형성하여 가장 유익한 서브스페이스를 포괄하는 프레임의 하위 집합을 선택합니다. 이를 통해 잉여 프레임은 제거되며, 중요한 정보가 담긴 프레임이 유지됩니다. MaxInfo는 기존의 균일 샘플링 방법을 강화하는 독창적인 프레임워크로, 각 씬(scene) 내의 핵심 프레임을 감지해 개선된 성능을 제공합니다.

- **Performance Highlights**: MaxInfo는 여러 평가 벤치마크에서 성능을 극대화하며, LLaVA-Video와 같은 모델에서 각각 3.28%, EgoSchema에서는 6.4%의 성능 향상을 달성했습니다. 실험 결과, MaxInfo는 기존의 균일 샘플링의 한계를 극복하고, 긴 비디오에서 중요한 정보를 효과적으로 포착할 수 있음을 증명했습니다.



### Tell2Reg: Establishing spatial correspondence between images by the same language prompts (https://arxiv.org/abs/2502.03118)
Comments:
          5 pages, 3 figures, conference paper

- **What's New**: 본 논문에서는 이미지 등록(image registration) 네트워크가 변위 필드(displacement fields)나 변환 파라미터(transformation parameters)를 예측하는 대신, 대응하는 영역을 분할(segmentation)하도록 설계된 새로운 방법을 제안합니다. 이러한 접근법은 동일한 언어 프롬프트(language prompt)를 기반으로 두 개의 서로 다른 이미지에서 대응하는 영역 쌍을 예측할 수 있도록 하며, 이는 자동화된 등록 알고리즘을 가능하게 합니다. 이를 통해 기존의 데이터 커리션(data curation) 및 라벨링(labeling)이 필요 없어져, 시간과 비용을 절감합니다.

- **Technical Details**: 제안된 방법은 고정 이미지(fixed image)와 이동 이미지(moving image) 간의 대응하는 영역을 찾아내는 문제를 해결하기 위해, GroundingDINO 및 Segment Anything Model(SAM)과 같은 사전 학습된 다중 모달 모델(multimodal models)을 이용합니다. 이 과정에서는 동일한 텍스트 프롬프트를 사용하여 두 이미지를 입력으로 하여 의미론적 영역을 인식하게 됩니다. 이 알고리즘에서는 IMfix와 Imov를 각각 고정 및 이동 이미지로 설정하고, 대응하는 ROIs를 식별하게 됩니다.

- **Performance Highlights**: Tell2Reg 알고리즘은 기존의 비지도 학습(unsupervised learning) 기반 이미지 등록 방법들과 비교하여 뛰어난 성능을 보이며, 약한 감독(weakly-supervised) 방법들과 유사한 수준의 성능을 기록합니다. 이 접근법은 다양한 이미지 등록 작업에 일반화 가능하며, 특히 프로스트 prostate MRI 이미지의 등록에서 우수한 결과를 보여 주었습니다. 추가적인 정성적 결과 또한 언어 의미(semantics)와 공간 대응(spatial correspondence) 간의 잠재적 상관관계를 제시합니다.



### Edge Attention Module for Object Classification (https://arxiv.org/abs/2502.03103)
Comments:
          11 pages

- **What's New**: 이 연구에서는 객체 분류 작업을 위해 새로운 "Edge Attention Module (EAM)"이 제안되었습니다. 기존 CNN(Covolutional Neural Networks)의 문제인 클래스 불균형과 클래스 간 유사성을 개선하기 위해, Max-Min Pooling 레이어를 포함한 자체 주의 모듈을 개발하여 엣지 정보에 집중하고 있습니다. 이를 통해 CNN 모델의 정확도와 F1-score가显著 향상된 결과를 도출하였습니다.

- **Technical Details**: Edge Attention Module은 Max-Min pooling 기술을 사용하며, 이는 이미지에서 중요한 엣지 피처만을 강조해 추출하는 방법입니다. 이 모델은 여러 표준 사전 훈련된 CNN 모델인 Caltech-101, Caltech-256, CIFAR-100, Tiny ImageNet-200 데이터셋에 적용되어 효과를 검증하였습니다. 본 연구는 기존의 CNN과 최신 기술인 Pooling-based Vision Transformer(PiT) 또는 Convolutional Block Attention Module(CBAM)과 비교하여 성능 개선을 확인하게 됩니다.

- **Performance Highlights**: 제안된 EAM 모듈을 통해 Caltech-101 데이터셋에서는 95.5%, Caltech-256 데이터셋에서는 86%의 정확도를 달성하였습니다. 이는 해당 데이터셋들에서 가장 뛰어난 성능을 기록한 것으로, 기존 모델들과 비교하여 개선된 결과로 나타났습니다. 다양한 데이터셋에서의 실험을 통해, EAM은 객체 분류 작업의 정확도를 현저히 향상시키는 데 중요한 역할을 하고 있습니다.



### Human-Aligned Image Models Improve Visual Decoding from the Brain (https://arxiv.org/abs/2502.03081)
- **What's New**: 이번 연구는 인간의 인지 과정을 고려해 개발된 이미지 인코더를 사용하여 뇌 신호로부터 이미지를 해독하는 새로운 방법을 제안합니다. 기본 이미지 인코더와 비교할 때, 인간 정렬 이미지 인코더는 감각적 속성을 보다 효과적으로 캡처하여 해독 성능을 향상시킵니다. 이는 기준선 모델보다 이미지 검색 정확도를 최대 21% 향상시키는 결과로 이어졌습니다. 이러한 결과는 다양한 EEG 아키텍처, 이미지 인코더 및 뇌 영상 모달리티에서도 일관되게 나타났습니다.

- **Technical Details**: 따라서 이 연구에서는 뇌 신호 인코더와 이미지 인코더를 조합한 아키텍처를 사용하여 뇌 활동에 기반하여 해당 이미지 데이터베이스에서 이미지를 검색하는 과정을 설명합니다. 모델은 인코더를 통해 뇌 활동을 공유 잠재 벡터 공간으로 매핑하며, 이를 통해 뇌 신호와 이미지 임베딩 간의 거리를 최적화합니다. 다중 모달 정보 최대 기대 손실(InfoNCE)을 활용하여 뇌 임베딩과 이미지 임베딩을 정렬하며, 강화된 이미지 인코더가 성능에 미치는 영향을 분석합니다.

- **Performance Highlights**: 실험 결과, 인간 정렬 이미지 인코더를 포함시키는 것이 뇌 신호에서의 시각적 해독 성능을 상당히 향상시키는 것으로 나타났습니다. 본 연구는 다양한 뇌 모달리티 및 이미지 모델을 사용하여 엄청난 성능 개선을 보여주었습니다. 이러한 개선은 이미지 유사성 데이터셋의 실험적 설계와 관련이 있으며, 뇌 신호가 빠른 시간 내에 시각 자극에 의해 유발되기 때문이라는 가설을 뒷받침합니다.



### High-frequency near-eye ground truth for event-based eye tracking (https://arxiv.org/abs/2502.03057)
- **What's New**: 이 연구는 기존의 이벤트 기반 안구 추적 데이터셋을 향상된 버전으로 제시하며, 이를 통해 알고리즘 검증과 딥 러닝 훈련에 필수적인 눈높이 주석이 추가된 데이터셋의 필요성을 해결하고자 합니다. 새로운 반자동 주석 파이프라인이 소개되며, 200Hz에서 동공 감지를 위한 주석이 과학 공동체에 제공됩니다. 이는 스마트 안경 기술에서의 효율적이고 저전력 안구 추적을 가능하게 합니다.

- **Technical Details**: 이 논문은 이벤트 기반 센서를 활용한 안구 추적 기술에 중점을 두고 있습니다. 특히, 깊이 있는 딥 러닝 모델을 사용할 수 있도록 하는 주석 생성을 위한 반자동 파이프라인이 개발되었습니다. 이 파이프라인은 200Hz 주파수에서 동공 위치를 자동으로 주석 처리하며, 사용자는 주석의 정확성을 검증하고 필요할 경우 수동으로 수정할 수 있습니다.

- **Performance Highlights**: 기존 데이터셋과 비교했을 때, 이 연구에서 제안하는 데이터셋은 200Hz의 높은 주기로 동공 중심을 추적할 수 있습니다. 이러한 고주파 주석 생성을 통해 기존 방법이 가진 한계를 극복하고, 다양한 시각적 관심사 및 인지 과정을 더욱 정확하게 모델링할 수 있습니다. 최종적으로, 이 논문의 결과는 이벤트 기반 안구 추적의 효율성을 높이는 데 기여할 것입니다.



### Driver Assistance System Based on Multimodal Data Hazard Detection (https://arxiv.org/abs/2502.03005)
- **What's New**: 이번 연구에서는 기존의 단일 모달(Modal) 도로 조건 비디오 데이터만을 사용하는 일반적인 접근 방식을 넘어, 도로 조건 비디오, 운전자의 얼굴 비디오, 오디오 데이터를 통합한 다중 모달 운전 보조 감지 시스템을 제안합니다. 이를 통해 희귀하고 예측 불가능한 운전 사건을 더 정확하게 인식할 수 있습니다. 제안된 시스템은 중간 피처 융합(intermediate fusion) 전략을 채택하여 별도의 피처 추출 없이 엔드 투 엔드(End-to-End) 학습이 가능하도록 합니다.

- **Technical Details**: 제안하는 모델은 도로 조건 비디오, 운전자의 얼굴 비디오, 오디오 데이터를 동시에 활용하여, 단일 데이터 소스로 인한 오판 가능성을 줄이고, 운전 이벤트의 긴 꼬리 분포(long-tailed distribution) 문제를 완화합니다. 특히 중간 수준에서 피처 융합을 수행함으로써, 다양한 데이터 유형 간의 상관관계를 효과적으로 캡처합니다. 이 접근 방식은 처리 파이프라인 도중 발생할 수 있는 불확실성을 줄이며, 특히 오디오 정보를 사용하는 경우에 유용합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 다중 모달 간의 상관성을 효과적으로 캡처하여 오판을 줄이고 운전 안전성을 향상시킵니다. 시뮬레이션 환경을 사용하여 수집된 새로운 세 가지 모달 데이터셋은 이 모델의 성능을 평가하기 위한 귀중한 자원으로, 이제까지 공개된 데이터셋에서는 동시에 촬영된 도로 조건 비디오, 운전자의 얼굴 비디오, 오디오 데이터가 부족했던 점을 보완합니다. 이러한 연구 결과는 향후 자율 주행 시스템의 안전성 및 신뢰성을 높이는 데 기여할 것으로 기대됩니다.



### Disentangling CLIP Features for Enhanced Localized Understanding (https://arxiv.org/abs/2502.02977)
- **What's New**: 본 논문에서는 기존의 Vision-language models (VLMs)가 이미지 분류 및 검색과 같은 거친 작업에는 잘 작동하지만, 세부적인 작업에서는 문제가 있음을 지적하고 있습니다. 이를 해결하기 위해 새로운 프레임워크인 Unmix-CLIP을 제안하며, 이는 semantic feature의 상호 연관성을 줄이고 특징을 분리하는 데 중점을 두고 있습니다.

- **Technical Details**: Unmix-CLIP은 mutual feature information (MFI) 손실을 도입하여 텍스트 특징을 서로 다른 클래스들 간의 유사성을 최소화하는 공간으로 투영합니다. 이를 통해 이미지 특징과 텍스트 특징이 서로 분리되도록 하며, multi-label recognition (MLR)을 사용하여 이미지 특징이 분리된 텍스트 특징과 정렬되도록 합니다. 이러한 방식은 다중 모달리티 간에서 특징 분리를 개선합니다.

- **Performance Highlights**: COCO-14 데이터셋에서 Unmix-CLIP은 특징 유사성을 24.9% 감소시키며, MLR 테스트에서 VOC2007에 대해 경쟁력 있는 성과를 보이고 COCO-14 데이터셋에서는 State-Of-The-Art(SOTA) 접근 방식을 초월합니다. 또한 Unmix-CLIP은 COCO 및 VOC의 기존 ZS3 방법들에서도 항상 더 좋은 성과를 보여주었습니다.



### VQA-Levels: A Hierarchical Approach for Classifying Questions in VQA (https://arxiv.org/abs/2502.02951)
- **What's New**: 본 논문은 Visual Question Answering (VQA) 시스템을 체계적으로 테스트할 수 있도록 돕는 새로운 기준 데이터셋, VQA-Levels를 제안합니다. 이 데이터셋은 질문을 7단계로 분류하여, 낮은 수준의 이미지 특징에서부터 높은 수준의 추상화에 걸쳐 다양한 질문 형식을 포함합니다. 기존 데이터셋들이 가지고 있던 모호한 질문이나 무의미한 질문의 문제를 해결하고, 자연스러운 인간의 질문 형태를 따릅니다.

- **Technical Details**: VQA-Levels 데이터셋은 질문을 1에서 7까지의 수준으로 나누어, 각 수준은 시각적 내용에 따라 특성을 달리합니다. 1단계는 직접적인 답변이 가능하며, 높은 수준의 질문인 7단계는 이미지 전체의 맥락을 고려해야 하는 질문들을 포함합니다. 각 질문은 일반적으로 하나 또는 두 개의 단어로 된 답변을 요구하며, 독특한 특성을 기반으로 합니다.

- **Performance Highlights**: 초기 테스트 결과, 기존 VQA 시스템은 1단계 및 2단계 질문에서 높은 성공률을 보였으며, 3단계 이상의 질문에서는 성능이 감소하는 경향이 나타났습니다. 특히, 3단계는 화면 텍스트에 관한 질문, 6단계는 외삽(extrapolation), 7단계는 전반적인 장면 분석을 요구하는 질문으로, 난이도가 높은 만큼 응답 정확도가 낮았습니다. 이 연구 결과는 VQA 시스템의 성능을 체계적으로 분석할 수 있는 기초 자료를 제공합니다.



### Every Angle Is Worth A Second Glance: Mining Kinematic Skeletal Structures from Multi-view Joint Cloud (https://arxiv.org/abs/2502.02936)
Comments:
          Accepted by IEEE Transactions on Visualization and Computer Graphics

- **What's New**: 이번 논문에서는 Sparse angular observations를 통한 multi-person motion capture의 문제를 해결하기 위해 Joint Cloud (JC)라는 새로운 개념을 도입하였습니다. JC는 동일한 joint type을 가진 2D joint들을 independent하게 삼각 측량하여 구성되며, 이를 통해 다양한 관점에서 정확한 2D joint 정보를 최대한 활용할 수 있습니다. 또한, Joint Cloud Selection and Aggregation Transformer (JCSAT)라는 새로운 프레임워크를 통해 3D 후보들 간의 복잡한 상관관계를 탐색하고 중복된 정보를 정리합니다.

- **Technical Details**: 제안된 JCSAT는 세 개의 연속적인 인코더로 구성되어 있으며, 이는 3D 후보들 간의 트랙 및 구조적 특징을 깊게 탐구합니다. 또한, Optimal Token Attention Path (OTAP) 모듈을 통해 중복된 후보들로부터 정보가 풍부한 특성을 선택하고 집계하여 최종적인 인간 움직임 예측을 지원합니다. 이 과정에서 masked learning mechanism을 활용해 더 높은 강건성을 유지합니다.

- **Performance Highlights**: 실험 결과, JCSAT는 새로운 multi-person motion capture 데이터셋인 BUMocap-X에서 탁월한 성능을 보였으며, 기존의 최첨단 방법들보다 모든 occlusion 시나리오에서 우수한 결과를 도출하였습니다. BUMocap-X는 복잡한 인터랙션과 심각한 occlusion이 포함된 새로운 데이터셋으로, 우리의 접근법의 강점을 입증하기 위해 제작되었습니다. 이 데이터셋과 방법론은 향후 연구에 있어 중요한 기여를 할 것으로 예상됩니다.



### Maximizing the Position Embedding for Vision Transformers with Global Average Pooling (https://arxiv.org/abs/2502.02919)
Comments:
          Accepted at AAAI 2025

- **What's New**: 최근 비전 트랜스포머(vision transformer)에서 포지션 임베딩(PE)의 효과를 극대화하는 MPVG(Method for Position-guided Vision Transformers with Global Average Pooling) 방법이 제안되었습니다. 이 방법은 기존의 클래스 토큰(class token) 대신 글로벌 평균 풀링(global average pooling, GAP) 방식을 사용할 때 발생하는 성능 저하 문제를 해결하기 위해 설계되었습니다. MPVG는 PE의 역기능을 최소화하고 각 레이어의 토큰 임베딩(token embedding)과 PE의 방향성을 유지하여 비전 트랜스포머의 성능을 향상시킵니다.

- **Technical Details**: 이 연구에서 제안된 MPVG는 레이어별 구조(layer-wise structure)에서 PE를 각 레이어에 전달하고, 레이어 정규화(Layer Normalization)를 독립적으로 적용하여 토큰 임베딩과 PE 간의 상호작용을 최적화합니다. 레이어가 깊어질수록 PE가 토큰 임베딩 값을 상호 보완함을 발견하였으며, 이는 비전 트랜스포머의 성능을 극대화하는 데 기여합니다. 또한 'Last LN'이라는 최종 레이어 정규화를 도입하여 MPVG를 통해 PE의 효과를 보다 극대화합니다.

- **Performance Highlights**: 실험 결과, MPVG는 다양한 비전 트랜스포머 아키텍처 및 작업에서 기존 방법들에 비해 우수한 성과를 기록했습니다. MPVG는 이미지 분류(image classification), 물체 탐지(object detection), 의미 분할(semantic segmentation)과 같은 여러 작업에서 일관되게 더 나은 성능을 보였습니다. 이로 인해 MPVG의 도입이 비전 트랜스포머의 효과성에 주목할 만한 영향을 미친다는 점이 강조되었습니다.



### PoleStack: Robust Pole Estimation of Irregular Objects from Silhouette Stacking (https://arxiv.org/abs/2502.02907)
- **What's New**: 본 연구에서는 여러 카메라 각도에서 수집한 실루엣 이미지를 사용하여 주요축 회전체의 회전 폴(기준축 방향)을 추정하는 알고리즘을 제안합니다. 제안된 방법은 저해상도 이미지에서도 높은 정확도를 제공하며, 심각한 표면 그림자 및 이미지 등록 오류에 대한 강인성을 입증합니다. 이 접근법은 목표 물체에 접근하는 단계와 정지 상태에서 모두 폴 추정에 적합합니다.

- **Technical Details**: 주요축 회전체에 대한 폴 추정 기술은 실루엣 스택 이미지에서 최대 대칭을 찾아내는 방식으로 구현됩니다. 이 과정에서 이산 푸리에 변환(Discrete Fourier Transform)을 적용하여 전역 중심 좌표 문제를 해결하고 강인성을 높입니다. 3D 폴 방향은 서로 다른 카메라 방향에서 수집된 두 개 이상의 측정값을 결합하여 추정합니다.

- **Performance Highlights**: 실험 결과는 저해상도 이미지에서도 가장 우수한 폴 추정 정확도를 보여주며, 접근 중 발생할 수 있는 다양한 조건 하에서도 강력한 성능을 유지합니다. 이 연구에서 제안한 방법은 특히 저해상도 조건과 복잡한 조명 상황에서도 효과적이며, 여러 이미지 간의 정렬 오류에 대한 저항력이 높습니다.



### Enhancing Quantum-ready QUBO-based Suppression for Object Detection with Appearance and Confidence Features (https://arxiv.org/abs/2502.02895)
Comments:
          8 pages for main contents, 3 pages for appendix, 3 pages for reference

- **What's New**: 이 연구는 기존 QUBO(Quadratic Unconstrained Binary Optimization) 기법의 한계를 극복하고, 객체 검출에서의 중복 예측을 정확하게 식별하는 새로운 QUBO 공식을 제안합니다. 새롭게 제안된 공식에는 이미지 유사성 기반의 appearance feature와 confidence scores의 곱으로 표시된 두 가지 특성이 포함됩니다. 이러한 접근은 부분적으로 가려진 객체를 보다 정확하게 탐지하는 데 도움을 줍니다.

- **Technical Details**: 제안한 QUBO 공식은 기존 QSQS(Quantum-soft QUBO Suppression)와 비교하여 이미지 유사성을 반영하는 appearance feature와 예측 쌍 간의 confidence scores의 곱을 통합하여 다중 예측으로 인한 중복을 효과적으로 구별합니다. 연구에서는 SSIM(Structural SIMilarity) 지표를 사용하여 appearance feature를 계산하며, 계산 비용을 줄이기 위해 divide-and-conquer 알고리즘과 GPU 병렬화 기법을 활용합니다.

- **Performance Highlights**: 실험 결과, 제안된 QUBO 공식을 사용한 객체 검출 방법은 기존 QSQS에 비해 최대 4.54점의 mAP(Mean Average Precision) 향상과 9.89점의 mAR(Mean Average Recall) 향상을 달성했습니다. 또한, 약 1GB의 GPU 메모리를 사용하는 조건에서 비혼잡 및 혼잡 장면에서 각각 6ms 및 14ms의 처리 시간을 기록하여 실행 시간을 대폭 단축했습니다.



### Expertized Caption Auto-Enhancement for Video-Text Retrieva (https://arxiv.org/abs/2502.02885)
- **What's New**: 본 논문은 Expertized Caption auto-Enhancement (ExCae) 방법을 제안하여 비디오-텍스트 검색 작업에서 캡션의 품질을 향상시키고 비서적 임무를 최소화합니다. 또한, 캡션 자동 개선 및 전문가 선택 메커니즘을 통해 비디오에 맞춤형 캡션을 제공하여 크로스 모달 매칭을 용이하게 합니다. 이 방법은 데이터 수집 및 계산 작업의 부담을 줄이며, 개인화된 매칭을 통해 자기 적응성을 향상시킵니다.

- **Technical Details**: ExCae 방법은 Caption Self-improvement (CSI) 모듈과 Expertized Caption Selection (ECS) 모듈로 구성됩니다. CSI 모듈은 다양한 관점에서 비디오를 설명하는 캡션을 생성하고, ECS 모듈은 학습 가능한 전문가를 통해 적합한 표현을 자동으로 선택합니다. 이 방법은 전통적인 텍스트 증강 방식과 달리 비디오에서 자동으로 캡션을 유도하여 비디오-텍스트 매칭을 위한 임베딩을 인코딩합니다.

- **Performance Highlights**: 저자들이 제안한 방법은 MSR-VTT에서 68.5%, MSVD에서 68.1%, DiDeMo에서 62.0%의 Top-1 recall accuracy를 달성하여 최신 방법들을 앞지르는 성능을 입증했습니다. 이 연구는 추가 데이터 없이도 비디오-텍스트 검색 작업에서 일관된 성능 향상을 보여주었으며, MSR-VTT에서의 베스트 벤치마크를 3.8% 초과 달성했습니다.



### Domain-Invariant Per-Frame Feature Extraction for Cross-Domain Imitation Learning with Visual Observations (https://arxiv.org/abs/2502.02867)
Comments:
          8 pages main, 19 pages appendix with reference. Submitted to ICML 2025

- **What's New**: 이 논문에서는 Imitation Learning (IL)에서 cross-domain(크로스 도메인) 시나리오의 복잡한 비주얼 과제를 해결하기 위해 Domain-Invariant Per-Frame Feature Extraction for Imitation Learning (DIFF-IL)이라는 새로운 방법을 제안합니다. DIFF-IL은 개별 프레임에서 도메인 불변 특징을 추출하고 이를 통해 전문가의 행동을 고립하여 복제하는 메커니즘을 도입합니다. 또한, 프레임 단위의 시간 라벨링 기법을 통해 특정 타임스텝에 따라 전문가 행동을 세분화하고, 적합한 보상을 할당함으로써 학습 성능을 향상시킵니다.

- **Technical Details**: DIFF-IL은 두 가지 주요 기여를 포함합니다: (1) 프레임 단위의 도메인 불변 특징 추출을 통해 도메인 독립적인 과제 관련 행동을 고립하고, (2) 프레임 단위 시간 라벨링을 통해 전문가 행동을 타임스텝별로 구분하여, 시간에 맞춘 보상을 할당합니다. 이러한 혁신을 통해 DIFF-IL은 소스 도메인 데이터와 전문가 행동 간의 제한된 중첩 상황에서도 정확한 도메인 정렬 및 효과적인 모방을 가능하게 합니다.

- **Performance Highlights**: DIFF-IL의 성능은 다양한 비주얼 환경에서 실험을 통해 입증되었습니다. 특히, Walker에서 Cheetah로의 전이 환경에서는 DIFF-IL이 단일 이미지 프레임의 잠재 특징을 효과적으로 조정하여 도메인 특정 아티팩트를 제거하며, 전문가 행동을 정확하게 모방하는 것을 가능하게 합니다. 이로 인해, DIFF-IL은 과제가 복잡한 조건에서도 성공적인 작업 완료를 보장할 수 있습니다.



### RS-YOLOX: A High Precision Detector for Object Detection in Satellite Remote Sensing Images (https://arxiv.org/abs/2502.02850)
- **What's New**: 이 논문에서는 위성 원격 감지 이미지의 자동 객체 탐지를 위한 향상된 YOLOX 모델인 RS-YOLOX를 제안합니다. 이 모델은 기존 원격 감지 이미지 탐지에서의 문제점을 해결하기 위해 개발되었습니다. 원활한 모델 개선을 위해 ECA(Efficient Channel Attention)와 ASFF(Adaptively Spatial Feature Fusion) 기법을 활용하였습니다.

- **Technical Details**: RS-YOLOX 모델은 YOLOX의 백본 네트워크에서 ECA를 사용하여 특징 학습 능력을 강화하고, 넥크 네트워크에서 ASFF를 결합하여 기능을 개선했습니다. 또한 Varifocal Loss 함수를 사용하여 훈련 과정에서 양성 샘플과 음성 샘플의 수를 균형 있게 조정했습니다. 최종적으로, SAHI(Slicing Aided Hyper Inference)라는 오픈 소스 프레임워크와 결합하여 고성능 원격 감지 객체 탐지기를 제작했습니다.

- **Performance Highlights**: 연구진은 DOTA-v1.5, TGRS-HRRSD 및 RSOD의 세 가지 항공 원격 감지 데이터 세트에서 RS-YOLOX 모델을 평가하였습니다. 비교 실험 결과, RS-YOLOX 모델은 원격 감지 이미지 데이터 세트에서 객체 탐지에 있어 가장 높은 정확도를 기록하였습니다.



### A Survey of Sample-Efficient Deep Learning for Change Detection in Remote Sensing: Tasks, Strategies, and Challenges (https://arxiv.org/abs/2502.02835)
Comments:
          Accepted in IEEE GRSM

- **What's New**: 최근 10년 동안 딥러닝(Deep Learning, DL) 기술의 발전은 원격 탐지 이미지(Remote Sensing Images, RSI)에서의 변화 감지(Change Detection, CD) 분야에 큰 영향을 미쳤습니다. 하지만 CD 방법들이 실제 환경에서의 적용에 있어 다양한 입력 데이터와 응용 상황으로 인해 제한적입니다. 본 논문은 DL 기반 CD 방법을 훈련하고 배포할 수 있는 다양한 방법론 및 전략을 정리하여 향후 연구에 대한 새로운 인사이트를 제공하는 것을 목표로 합니다.

- **Technical Details**: CD는 입력 이미지의 유형 및 결과의 세분화에 따라 Binary CD(BCD), Multi-class CD/Semantic CD(MCD/SCD), Time-series CD(TSCD)와 같은 여러 하위 범주로 나뉘어집니다. BCD는 최근 몇 년간 가장 많이 연구된 CD 작업으로, 초기에는 UNet과 같은 합성곱 신경망(Convolutional Neural Networks, CNN)을 활용하여 변화를 직접적으로 분할하는 방식으로 접근했습니다. 데이터의 동질성을 기반으로 시암 네트워크(Siamese networks)를 구성하여 효과적으로 시간적 특성을 활용하는 방법이 널리 받아들여지고 있습니다.

- **Performance Highlights**: 이 논문 접근법은 CD의 정확도를 90% 이상으로 향상시켜 다양한 벤치마크 데이터 세트에서 뛰어난 성능을 입증하고 있습니다. BCD의 주요 도전 과제로는 계절적 변화와 공간적 불일치 및 조명 차이를 구분하는 것이 있습니다. 최근 비전 트랜스포머(Vision Transformers, ViTs)와 같은 혁신적인 방법들이 등장하면서, RS의 변화 감지에 대한 다양한 연구가 활발히 진행되고 있습니다.



### AIoT-based smart traffic management system (https://arxiv.org/abs/2502.02821)
- **What's New**: 이번 논문은 도시 환경에서 교통 흐름을 최적화하고 혼잡을 줄이기 위해 설계된 혁신적인 AI 기반 스마트 교통 관리 시스템을 소개합니다. 기존 CCTV 카메라의 실시간 영상을 분석함으로써 추가 하드웨어의 필요성을 제거하고 배치 비용과 유지 보수 비용을 최소화합니다. 이 AI 모델은 실시간 비디오 피드를 처리하여 차량 수를 정확하게 계산하고 교통 밀도를 평가함으로써 더 높은 교통량을 가진 방향에 우선 순위를 두는 적응형 신호 제어를 가능하게 합니다.

- **Technical Details**: 제안된 시스템은 PyGame을 사용하여 다양한 교통 조건에서의 성능을 평가하기 위해 시뮬레이션되었습니다. 실시간 적응성이 보장되는 이 시스템은 교통 흐름을 원활하게 하고 운전자의 대기 시간을 최소화합니다. AI를 사용하여 교통 신호를 최적화함으로써 도시 교통 문제를 해결하는 데 중요한 역할을 할 수 있습니다.

- **Performance Highlights**: 시뮬레이션 결과, AI 기반 시스템은 전통적인 고정형 교통 신호 시스템보다 34% 더 나은 성능을 보이며 교통 흐름 효율성의 상당한 향상을 가져오는 것으로 나타났습니다. 이 혁신적인 시스템은 현대 도시를 위한 비용 효율적이며 확장 가능한 솔루션을 제공합니다. 스마트 시티 인프라 및 지능형 교통 시스템 분야에서 중요한 발전을 나타냅니다.



### 3D Foundation AI Model for Generalizable Disease Detection in Head Computed Tomography (https://arxiv.org/abs/2502.02779)
Comments:
          Under Review Preprint

- **What's New**: 본 논문에서는 일반화 가능한 질병 탐지를 위한 머리 CT 스캔을 위한 Foundation Model인 FM-CT를 소개합니다. 이 모델은 레이블 없는 3D 머리 CT 스캔 36만 1,663개로 자가 감독 학습(self-supervised learning)을 통해 학습되었으며, 이는 고품질 레이블 부족 문제를 해결하는 데 도움을 줍니다. FM-CT는 기존의 2D 처리 대신 3D 구조를 활용하여 더욱 효율적으로 머리 CT 이미지를 분석합니다.

- **Technical Details**: FM-CT는 깊은 학습 모델로서, 자기 증류(self-distillation)와 마스킹 이미지 모델링(masked image modeling)의 두 가지 자가 감독 프레임워크를 채택하여 훈련되었습니다. 이 모델은 맞춤형 비전 변환기(Transformer)를 기반으로 하는 부피 인코더(volumetric encoder)를 학습하고, 다양한 프로토콜로부터의 머리 CT 스캔을 정규화하여 일관된 입력을 제공합니다. 10개의 다양한 질병 감지 작업에 대해 평가되었으며, 전반적인 성능이 향상되었습니다.

- **Performance Highlights**: FM-CT는 내부 데이터(NYU Langone)에서 기존 모델에 비해 16.07%의 향상을 보였으며, 외부 데이터(NYU Long Island, RSNA)에서도 각각 20.86%, 12.01%의 성능 향상을 달성했습니다. 이 연구는 자가 감독 학습이 의료 영상에서 매우 효과적임을 보여주며, FM-CT 모델이 머리 CT 기반 진단에서 더 널리 적용될 잠재력을 가지고 있음을 강조합니다. 실험 결과는 실제 임상 상황에서 중요한 영향을 미칠 수 있는 모델의 일반화 가능성과 적응성을 뒷받침합니다.



### Rethinking Vision Transformer for Object Centric Foundation Models (https://arxiv.org/abs/2502.02763)
- **What's New**: FLIP(Fovea-Like Input Patching) 접근법은 전체 이미지를 처음부터 객체 중심적으로 인코딩하여 특히 작은 객체들이 많은 고해상도 장면에서 우수한 분할 성능을 보여줍니다. 기존의 Segment Anything Model(SAM) 및 FastSAM 모델 대신, FLIP은 오프 그리드(off-grid) 기반으로 이미지 입력을 선택하고 인코딩하며, 이를 통해 데이터 효율성이 개선됩니다. 이 방법은 기존의 다양한 데이터셋에서 SAM에 가까운 Intersection over Union(IoU) 점수를 달성하며, FastSAM보다 모든 IoU 측정에서 우수한 성능을 기록했습니다.

- **Technical Details**: FLIP은 고유의 fovea-inspired 패칭 방식을 가지고 있으며, 이는 객체의 크기와 공간적 특성에 따라 처리 파이프라인을 동적으로 조정합니다. 이 로직은 2D 가우시안 분포를 기반으로 하여 멀티 해상도 패치를 선택하며, 각 패치는 객체 중심에 집중되어 세부 정보를 캡처하고, 주변 맥락을 유지하기 위해 더 큰 패치를 사용합니다. 이러한 접근법은 FLIP의 성능을 극대화하기 위해 ViT(Vision Transformer) 기반 인코더와 결합되어 객체의 외관, 형태 및 위치 정보를 구분하여 인코딩합니다.

- **Performance Highlights**: FLIP은 Hypersim, KITTI-360, OpenImages 데이터셋에서 기존의 SAM 및 FastSAM에 비해 경쟁력 있는 성능을 보여주며, 특히 더 적은 매개변수 수로 더 나은 정확도를 달성합니다. 또한, FLIP은 새로운 ObjaScale 데이터셋에서 모든 다른 방법들을 초월하는 성능을 선보이며, 고해상도 실제 배경과 다양한 크기의 객체를 결합하여 세그멘테이션을 수행합니다. 이러한 결과는 FLIP이 매우 효율적인 객체 추적 응용 프로그램에 높은 잠재력을 가지고 있음을 시사합니다.



### RFMedSAM 2: Automatic Prompt Refinement for Enhanced Volumetric Medical Image Segmentation with SAM 2 (https://arxiv.org/abs/2502.02741)
- **What's New**: 본 논문에서는 기존 SAM 모델의 후속 제품인 Segment Anything Model 2(SAM 2)에 대해 다루고 있습니다. SAM 2는 이미지 및 비디오 도메인 모두에 걸쳐 우수한 0-shot 성능을 보이며, 특히 의료 이미지 분할(search segmentation)에서의 가능성도 보고되었습니다. 그러나 SAM 2 또한 이진 마스크 출력, 의미론적 라벨 추론 부족, 정확한 프롬프트에 대한 의존성 등의 제한점을 가지고 있습니다.

- **Technical Details**: 의료 이미지 분할 분야에서는 CNN과 ViT와 같은 딥러닝 기반 접근법이 자리 잡았으나, 고품질 주석의 부족으로 모델들의 훈련이 어렵습니다. 본 연구에서는 맞춤형 파인튜닝 어댑터를 통해 SAM 2의 성능 한계를 탐색하였고, BTCV 데이터셋에서 Dice Similarity Coefficient(DSC) 92.30%를 달성하여 최신의 nnUNet을 12% 초과 달성했습니다. 또한, 프롬프트 의존성을 해결하기 위해 UNet을 도입하여 예측된 마스크와 바운딩 박스를 자동으로 생성하도록 했습니다.

- **Performance Highlights**: SAM 2의 성능을 개선하기 위해 제안된 방법은 AMOS2022 데이터셋에서 최신 기술에 비해 2.9%의 Dice 향상을 보였고, BTCV 데이터셋에서는 nnUNet보다 6.4% 우수한 성능을 보여줍니다. 특히, RFMedSAM 2라는 프레임워크의 도입과 독립적인 UNet 기반의 자동 프롬프트 생성을 통해 의료 이미지 분할의 정확성을 크게 향상시켰습니다.



### Multiple Instance Learning with Coarse-to-Fine Self-Distillation (https://arxiv.org/abs/2502.02707)
- **What's New**: PathMIL은 Computational Pathology에서 Whole Slide Images (WSI) 분석을 위한 새로운 접근 방식을 제안합니다. 주목할만한 점은 instance-level (인스턴스 레벨) 감독을 도입하여 각 인스턴스에 대한 라벨을 생성하고, bag-level (백 레벨) 학습으로부터의 정보를 distill (증류) 하는 Coarse-to-Fine Self-Distillation (CFSD) 패러다임을 적용했다는 것입니다. 또한, Two-Dimensional Positional Encoding (2DPE)을 통해 가방 내 인스턴스의 공간적 배치를 효과적으로 모델링합니다.

- **Technical Details**: PathMIL의 핵심 요소는 CFSD 및 2DPE입니다. CFSD는 bag-level 정보를 이용해 높은 신뢰도의 인스턴스 레이블을 추출하고, 2DPE는 WSI의 2차원 좌표 정보를 이용하여 인스턴스 간의 문맥적 관계를 포착합니다. 이와 같은 접근 방식은 instance-level 학습을 가능하게 하여 기존의 MIL 프레임워크보다 뛰어난 성능을 보입니다.

- **Performance Highlights**: PathMIL은 TCGA-NSCLC, CAMELYON16 등 여러 벤치마크 작업에서 평가되었으며, 에스트로겐과 프로게스테론 수용체 상태 분류에서 각각 AUC 점수 0.9152 및 0.8524를 달성했습니다. 또한 subtype classification과 tumor classification에서도 각각 AUC 0.9618과 0.8634를 기록하며 기존 방법들을 초월하는 성과를 보였습니다.



### Controllable Video Generation with Provable Disentanglemen (https://arxiv.org/abs/2502.02690)
- **What's New**: 최근까지 발전된 기술에도 불구하고, 영상 생성의 정확하고 독립적인 제어는 여전히 큰 도전 과제가 남아있습니다. 본 연구에서는 Controllable Video Generative Adversarial Networks (CoVoGAN)을 제안하며, 이는 영상 내 개념을 분리하여 각 요소에 대한 효율적이고 독립적인 제어를 가능하게 합니다. CoVoGAN은 최소 변화 원칙(minimal change principle)과 충분한 변화 속성(sufficient change property)을 활용하여 정적 및 동적 잠재 변수를 분리합니다.

- **Technical Details**: CoVoGAN의 핵심은 Temporal Transition Module을 통해 동적 및 정적 요소를 구분하는 것입니다. 이 방법은 잠재 동적 변수의 차원 수를 최소화하고 시간적 조건 독립성을 부여하여, 동작과 정체성을 독립적으로 제어할 수 있게 합니다. 이러한 이론적 뒷받침과 함께 다양한 비디오 생성 기준을 통해 제안한 방식의 유효성을 실증합니다.

- **Performance Highlights**: Multiples 데이터 세트를 통해 CoVoGAN의 효과를 평가한 결과, 다른 GAN 기반의 비디오 생성 모델에 비해 생성 품질과 제어 가능성에서 현저히 개선된 성능을 보였습니다. 또한, 훈련 과정에서의 강인성과 추론 속도에서도 기존 기법들보다 우수한 결과를 나타내었습니다. 본 연구는 또한 비디오 생성 분야에서의 식별 가능성 정리를 제시하며, 향후 연구 방향을 제시합니다.



### Blind Visible Watermark Removal with Morphological Dilation (https://arxiv.org/abs/2502.02676)
- **What's New**: MorphoMod는 새로운 방식으로 눈에 띄는 워터마크를 자동으로 제거하는 방법을 제안합니다. 이 방법은 목표 이미지에 대한 사전 지식 없이 작동하는 블라인드 설정에서 작동하는 점에서 기존의 방법들과 차별화됩니다. MorphoMod는 불투명 및 투명 워터마크를 효과적으로 제거하며 의미적 내용을 보존합니다.

- **Technical Details**: MorphoMod는 형태학적 확장(morphological dilation)과 채우기(inpainting)를 결합한 새로운 프레임워크로, 세 가지 주요 단계로 구성됩니다: 1) 분할(segment), 2) 채우기(inpaint), 3) 복원(restore). 각 단계에서 선행 연구의 최적 성능을 발휘하는 분할 모델을 사용하여 이미지의 워터마크 마스크를 생성하고 이를 세밀하게 보정합니다.

- **Performance Highlights**: MorphoMod는 다양한 벤치마크 데이터셋에서 기존의 최첨단 방법에 비해 최대 50.8%의 워터마크 제거 효과성을 달성했습니다. 상세한 실험 분석을 통해 채우기 모델의 성능과 프롬프트 설계가 워터마크 제거 및 이미지 품질에 미치는 영향을 평가하였으며, 모듈을 통해 스테가노그래피 관련 문제 해결에도 효과적인 가능성을 보여주었습니다.



### Deep Learning-Based Facial Expression Recognition for the Elderly: A Systematic Review (https://arxiv.org/abs/2502.02618)
- **What's New**: 이 연구는 노인을 위한 딥러닝 기반의 얼굴 표정 인식(FER) 시스템에 대한 체계적 검토를 제공하며, 노인 인구에 적합한 기술의 필요성을 강조합니다. 특히, 노인 전용 데이터셋의 부족과 불균형한 클래스 배분, 나이에 따른 얼굴 표정 차이의 영향을 다루고 있습니다. 연구 결과, 합성곱 신경망(CNN)이 FER에서 여전히 우위를 점하고 있으며, 제한된 자원 환경에서 더 가벼운 버전의 중요성이 강조됩니다.

- **Technical Details**: 이 연구는 31개의 연구를 분석하여 딥러닝의 발전과 그에 따른 얼굴 표정 인식 시스템의 발전 상황을 조명합니다. 연구자는 심층 학습(Deep Learning) 기술, 특히 CNN이 노인 환자와의 다양한 상호작용에서 정서적 상태를 인식하는 데 어떻게 사용되는지 설명합니다. 또한, 설명 가능 인공지능(XAI)의 필요성을 강조하며, 이 기술이 FER 시스템의 투명성과 신뢰성을 높일 수 있는 방안을 제시합니다.

- **Performance Highlights**: FER 시스템은 노인 돌봄에서 중요한 역할을 할 수 있으며, 얼굴 표정을 통한 감정 모니터링 및 개별화된 치료가 가능해집니다. 그러나 노인 인구에 특화된 FER 연구가 부족하여 정서 인식의 정확성을 높이기 위한 데이터셋의 다양성과 연구 접근법의 필요성이 강하게 제기되었습니다. 이 연구는 노인 돌봄을 위한 FER 시스템의 발전 방향에 대한 의의와 함께 향후 연구에 대한 권고 사항을 포함합니다.



### MIND: Microstructure INverse Design with Generative Hybrid Neural Representation (https://arxiv.org/abs/2502.02607)
- **What's New**: 본 연구에서는 미세구조(microstructures)의 역설계(inverse design) 문제를 해결하기 위해 새로운 생성 모델을 제안합니다. 이 모델은 홀로플레인(Holoplane)이라는 고급 하이브리드 신경 표현(neural representation)과 잠재적 확산(latent diffusion)을 통합하여 기하학적(gemetric) 및 물리적 성질의 정밀한 제어를 가능하게 합니다. 이러한 혁신은 기존 방법들과 비교할 때 구조적 다양성과 설계의 유연성을 높이는 데 기여합니다.

- **Technical Details**: 제안된 생성 모델은 다양한 기하학적 형태를 포함하는 다중 클래스 데이터셋에 대해 훈련되었으며, 트러스(truss), 쉘(shell), 튜브(tube), 판(plate) 구조와 같은 여러 미세구조 클래스를 포괄합니다. 무작위화와 경계 연결성(boundary compatibility) 최적화를 지원하는 이 모델은 기하학적 유효성을 보장하며, 복잡한 조합을 통합할 수 있는 능력을 보여줍니다. 이를 통해 다양한 미세구조 유형을 생성하고 물성(target properties)을 유지하는 데 성공하고 있습니다.

- **Performance Highlights**: 실험 결과, 본 모델은 목표 물성에 부합하는 미세구조를 생성하면서 기하학적 유효성을 유지하는 뛰어난 성능을 입증하였습니다. 또한, 교차 클래스(interpolation)와 이질적 구조의 충전(infilling)과 같은 새로운 미세구조 생성 가능성을 탐구하는 데 유용합니다. 결과적으로, 본 연구는 미세구조 설계의 새로운 가능성을 제시하며, 발생할 모든 미세구조 형태와 속성을 모델링할 수 있는 잠재력을 보유하고 있습니다.



### Deep Clustering via Probabilistic Ratio-Cut Optimization (https://arxiv.org/abs/2502.03405)
Comments:
          Proceedings of the 28th International Conference on Artificial Intelligence and Statistics (AISTATS) 2025, Mai Khao, Thailand. PMLR: Volume 258

- **What's New**: 이번 연구에서는 그래프 비율 컷 최적화를 위한 새로운 방법(PRCut)을 제안합니다. 이 방법은 이진 할당을 확률 변수로 모델링하여, 온라인 환경에서 할당 변수의 매개변수를 학습합니다. 또한, PRCut 클러스터링은 라엘리 몽타주 완화 방법 및 다른 널리 사용되는 방법들보다 뛰어난 성능을 발휘함을 보여 줍니다. 이 새로운 방법은 셀프-슈퍼바이즈드(self-supervised) 표현을 활용하여 경쟁력 있는 성능을 달성할 수 있습니다.

- **Technical Details**: 본 연구에서 우리는 라플라시안(Laplacian) 행렬의 스펙트럴 분해를 사용하지 않고 비율 컷 최적화를 수행하는 방법을 개발하였습니다. 클러스터 할당을 확률 변수로 취급하며, 신경망을 활용하여 할당 확률을 파라미터화합니다. 이를 통해 스토캐스틱 경량 하강법을 사용하여 클러스터링 문제를 온라인으로 해결하고, 메모리 집약적인 스펙트럴 방법보다 더 나은 비율 컷 목표를 달성합니다. 수퍼바이즈드 유사성을 활용할 경우, 제안한 접근 방식은 수퍼바이즈드 분류기와 비슷한 성능을 보여 주는 것도 확인하였습니다.

- **Performance Highlights**: PRCut 방법은 클러스터링 결과와 제공된 유사성 측정 사이에 높은 충실도를 보여 줍니다. 또한 텍스트나 음성 트랜스포머의 사전 훈련(pre-training)에 사용될 수 있는 다른 클러스터링 방법에 대한 대체 솔루션으로 작용할 수 있습니다. 이를 통해 하향 작업(downstream tasks)에 대한 사전 훈련의 효과를 높일 수 있는 여러 가능성을 열어 줍니다.



### Ethical Considerations for the Military Use of Artificial Intelligence in Visual Reconnaissanc (https://arxiv.org/abs/2502.03376)
Comments:
          White Paper, 30 pages, 7 figures

- **What's New**: 이 백서에서는 군사적 맥락에서 인공지능(AI)을 책임 있게 배포하는 것의 중요성을 강조하고 있으며, 윤리 및 법적 기준에 대한 약속을 다루고 있습니다. 군사에서 AI의 역할은 단순한 기술적 응용을 넘어서는 만큼, 윤리적 원칙을 기반으로 한 프레임워크가 필요하다는 점을 지적하고 있습니다.

- **Technical Details**: 논문은 공정성(Fairness), 책임성(Accountability), 투명성(Transparency), 윤리(Ethics)라는 FATE 기준에 중점을 두어 윤리적 AI 원칙을 논의합니다. 또한, 군사-specific 윤리적 고려사항을 통해 정의된 전쟁 이론(Just War Theory) 및 저명한 기관들이 설정한 원칙들로부터 통찰을 얻고 있습니다.

- **Performance Highlights**: 군사 AI 응용을 위한 추가 윤리적 고려사항에는 추적 가능성(traceability), 비례성(proportionality), 거버넌스(governability), 책임성(responsibility), 신뢰성(reliability)이 포함됩니다. 이 윤리적 원칙들은 해양, 공중 및 육상 분야의 세 가지 사용 사례를 바탕으로 논의되며, 실제 시나리오와 밀접한 자동 센서 데이터 분석 및 설명 가능한 AI(XAI) 방법을 통해 그 적용이 구체화됩니다.



### Controllable GUI Exploration (https://arxiv.org/abs/2502.03330)
- **What's New**: 이 논문에서는 디퓨전 기반 접근법을 제안하여 인터페이스 스케치를 저렴하게 생성하는 방법을 소개합니다. 이 새로운 모델은 A) 프롬프트, B) 와이어프레임, C) 비주얼 플로우의 세 가지 입력을 통해 생성 프로세스를 유연하게 제어할 수 있도록 합니다. 디자이너는 이 세 가지 입력을 다양한 조합으로 제공하여 다양한 저충실도 솔루션을 신속하게 탐색할 수 있습니다.

- **Technical Details**: 제안한 모델은 고유의 두 가지 어댑터를 사용하여 GUI 요소의 위치 및 유형과 같은 지역적 속성과 전체 비주얼 플로우 방향과 같은 글로벌 속성을 제어합니다. 본 모델은 텍스트 기반 프롬프트 접근 방식의 단점을 극복하고, 비주얼 큐를 통해 더 나은 GUI 특성을 전달할 수 있습니다. 또한, 새로운 데이터셋을 생성하여 혼합된 모바일 UI 및 웹 페이지를 포함하여 GUI 생성 AI 모델 교육에 활용할 수 있는 기초 자료를 제공합니다.

- **Performance Highlights**: 모델의 성능을 평가한 결과, 제안한 모델은 입력 사양과 더욱 정밀하게 일치하며, 신속하고 다양한 GUI 대안을 탐색할 수 있다는 것을 질적으로 입증하였습니다. 이 모델은 대규모 디자인 공간을 최소한의 입력 명세로 탐색할 수 있게 해주어 UI 디자인 작업을 보다 효율적으로 수행할 수 있도록 합니다.



### MAP Image Recovery with Guarantees using Locally Convex Multi-Scale Energy (LC-MUSE) Mod (https://arxiv.org/abs/2502.03302)
- **What's New**: 본 논문은 데이터 매니폴드(data manifold) 주변의 지역적 근처에서 강한 볼록성을 가지는 다중 규모(multi-scale) 심층 에너지 모델을 제안하여 역문제(inverse problems)에 적용하고 있습니다. 특히, CNN(Convolutional Neural Network)을 통해 매개변수화된 다중 규모 에너지 모델로 음의 로그 사전분포(negative log-prior)를 표현합니다. 이 모델은 CNN의 그래디언트를 지역 단조(monotone)로 제한하여 LC-MuSE(Local Convex Multi-Scale Energy)로서의 성질을 개득하도록 합니다.

- **Technical Details**: 제안된 모델은 이미지 기반의 역문제에 사용되며, 이론적 보장으로는 해의 유일성(uniqueness), 역문제로의 최소값으로의 수렴(convergence)의 보장, 입력의 섭동에 대한 강건성(robustness)을 제공합니다. 이 논문에서는 특히 패러럴 자기공명영상(MR) 재구성의 맥락에서 상태-of-the-art의 볼록 정규화기(convex regularizers)보다 더 나은 성능을 보이는 방법을 제시하고 있습니다. 또한, 제안된 접근 방식은 대규모 문제에 쉽게 적용될 수 있도록 강한 볼록성의 제약을 지역적으로 확대하여 적용할 수 있는 가능성을 제공합니다.

- **Performance Highlights**: LC-MuSE 접근법은 색다른 이미지 복구 방식을 제안하여 동종 기술에 비해 향상된 성능을 보여줍니다. MR 이미징 적용에 초점을 맞추고 있지만, 이 프레임워크는 일반 계산적 영상 문제에 대한 일반적인 적용 가능성을 갖고 있습니다. 실험 결과로는 이미지 복원 과제에서 기존 방법들과 비교할 때 유의미한 성과를 보여줍니다.



### Conditional Prediction by Simulation for Automated Driving (https://arxiv.org/abs/2502.03286)
Comments:
          Accepted for publication at "16. Uni-DAS e.V. Workshop Fahrerassistenz und automatisiertes Fahren". Link: this https URL

- **What's New**: 이 논문에서는 협력적 계획을 가능하게 하는 예측 모델을 소개합니다. 전통적인 모듈형 자동 운전 시스템은 예측(prediction)과 계획(planning)을 별도의 작업으로 처리하였으나, 제안된 모델은 다양한 후보 궤적(candidate trajectories)을 따른 조건부 예측을 수행합니다. 이를 통해 자동 운전 차량이 주변 차들과의 상호작용을 보다 효율적으로 관리할 수 있습니다.

- **Technical Details**: 이 시스템은 역전파 강화 학습(Adversarial Inverse Reinforcement Learning, AIRL)을 통해 배운 행동 모델을 사용하여 주행 환경을 시뮬레이션합니다. 각 주변 차량의 거동을 미세한 교통 시뮬레이션으로 모델링하며, 차량 간의 상호작용을 처리하면서 예측을 진행합니다. 이를 통해 각 차의 행동이 상호 작용에 적절하게 반응하도록 합니다.

- **Performance Highlights**: 이 접근 방식은 인과적 상호작용을 포함하여 AV가 지속적으로 후보 경로를 선택하고 그 영향을 예측할 수 있게 합니다. 예측 동안 AV의 경로가 동적으로 조정되므로 복잡한 교통 상황에서도 보다 원활한 상호작용과 안전한 주행이 가능해집니다. 다양한 실제 교통 상황에서 모델이 효과적으로 작동함을 보여줍니다.



### Deep Learning Pipeline for Fully Automated Myocardial Infarct Segmentation from Clinical Cardiac MR Scans (https://arxiv.org/abs/2502.03272)
- **What's New**: 이 연구는 심근경색(MI) 분할을 위한 완전 자동화된 딥 러닝 기반 방법을 개발하고 평가하는 것을 목표로 하였습니다. 기존의 수동 LGE(지연 강화) 심장 자기 공명(MR) 이미지 분할 방식의 한계를 해결하고, 신속하고 정확한 심근경색의 양적 평가를 가능하게 하는 파이프라인을 제시합니다. 이는 임상 환경에서의 이미지 전처리에 필요한 단계를 포함하여 모든 과정을 자동으로 수행할 수 있습니다.

- **Technical Details**: 연구는 2D 및 3D CNN(합성곱 신경망)에 기반한 계단식 프레임워크를 사용하여 진행되었습니다. 144개의 LGE CMR 검사 데이터를 훈련셋으로 사용하여 AI 기반 분할 성능을 개발하였고, 152개의 검사로 구성된 테스트셋에서 AI 분할과 수동 분할을 정량적으로 비교하였습니다. CNN은 원본 데이터에서 전체 좌심실을 포함하는 작고 필요한 이미지 스택을 추출하고, 이후 다중 클래스 분할을 수행하는 방식으로 구성되었습니다.

- **Performance Highlights**: AI 기반 분할 결과는 수동으로 계산한 심근경색 볼륨과 높은 일치를 보였으며, 전문가들은 AI 측정이 실제 심근경색 범위를 더 잘 반영한다고 평가했습니다. AI 기반 분할의 정확도를 평가한 결과, 심혈관 전문가의 눈으로 볼 때 AI 출력물이 유의미하게 더 나은 평가를 받아 임상적인 가능성을 제시합니다. 반면, 미세혈관 폐쇄 MVO의 경우에는 여전히 수동 측정을 더 선호했습니다.



### When Pre-trained Visual Representations Fall Short: Limitations in Visuo-Motor Robot Learning (https://arxiv.org/abs/2502.03270)
- **What's New**: 이 연구는 사전 학습된 시각 표현(PVRs)을 비주얼-모터 로봇 학습에 통합하는 새로운 방법을 제안합니다. PVR의 고유한 한계인 시간 상호 엉킴과 손상에 대한 일반화 능력 부족을 해결하기 위해, 연구는 시간 인식과 작업 완료 감각을 결합하여 PVR 기능을 향상시킵니다. 또한, 작업 관련 지역적 특징에 선택적으로 주의를 기울이는 모듈을 도입하여 다채로운 장면에서도 로버스트한 성능을 보장합니다.

- **Technical Details**: 기존 PVR의 한계로 인해 비주얼-모터 정책 학습이 중단되는 문제를 강조하고, 이를 해결하기 위해 두 가지 주요 기법을 제안합니다. 첫 번째는 PVR 기능에 시간적 인식을 추가하여 시간적으로 엉킨 특징을 분리하는 것이며, 두 번째는 작업 관련 시각 단서를 기준으로 주목하는 모듈을 도입하여 эффективность을 높입니다. 이 과정에서 기존 PVR 생태계를 손상시키지 않고도 주요 성과를 도출했습니다.

- **Performance Highlights**: 실험 결과, 마스킹 목표로 학습된 PVR은 기존 기능을 보다 효과적으로 활용하여 성능이 유의미하게 향상되었습니다. 특히, 시각 정보가 과도하게 엉켜 있던 기존의 문제를 해결함으로써 기본적인 피컷 앤 플레이스 작업에서도 향후 성과 개선이 확인되었습니다. 연구는 PVR의 기본적인 한계를 극복하고 비주얼-모터 학습의 가능성을 상향 제시하는 중요한 기초자료를 제공하고 있습니다.



### GARAD-SLAM: 3D GAussian splatting for Real-time Anti Dynamic SLAM (https://arxiv.org/abs/2502.03228)
- **What's New**: GARAD-SLAM은 동적 장면을 위해 특별히 조정된 실시간 3DGS 기반 SLAM 시스템으로, 기존의 3DGS 기반 SLAM 시스템이 직면하는 매핑 오류와 추적 드리프트 문제를 해결하기 위해 제안되었습니다. 이 시스템은 게우시안(Gaussian)으로 동적 분할(dynamics segmentation)을 직접 수행하고 이를 전방향 네트워크에 매핑하여 동적 포인트 레이블을 획득함으로써 정밀한 동적 제거(dynamic removal)와 강력한 추적을 달성합니다.

- **Technical Details**: 기존 방법들과의 차별점은 동적 레이블이 부여된 게우시안에 렌더링 패널티(rendering penalties)를 부과하여, 간단한 가지치기로 인한 비가역적인 오류 제거를 피하는 것입니다. 또한, GARAD-SLAM은 Gaussian pyramid network를 활용하여 뛰어난 동적 추적(dynamic tracking) 능력을 보여줍니다.

- **Performance Highlights**: 실제 데이터셋에서의 결과는 GARAD-SLAM 방법이 기존의 기준 방법들과 비교하여 경쟁력이 있음을 보여주며, 아티팩트(artifacts)를 줄이고 렌더링의 품질을 향상시키는 높은 품질의 재구성을 생성합니다.



### iVISPAR -- An Interactive Visual-Spatial Reasoning Benchmark for VLMs (https://arxiv.org/abs/2502.03214)
- **What's New**: 최근 발표된 iVISPAR는 비전-언어 모델(VLM)의 공간 추론 능력을 평가하기 위해 고안된 인터랙티브 다중 모드 벤치마크입니다. 이 벤치마크는 고전적인 슬라이딩 타일 퍼즐의 변형으로, 논리적 계획, 공간 인지, 다단계 문제 해결 능력을 요구합니다. iVISPAR는 2D, 3D 및 텍스트 기반 입력 방식 모두를 지원하여 VLM의 전반적인 계획 및 추론 능력을 포괄적으로 평가할 수 있습니다. 연구 결과, 일부 VLM은 간단한 공간 작업에서 양호한 성능을 보였으나, 복잡한 구성에서는 어려움을 겪는다는 점이 드러났습니다.

- **Technical Details**: iVISPAR의 주요 특징 중 하나는 Sliding Geom Puzzle(SGP)로, 기존의 슬라이딩 타일 퍼즐을 고유한 색상과 모양으로 정의된 기하학적 객체로 대체했습니다. 벤치마크는 사용자가 자연어 명령어를 발행하여 보드에 대한 작업을 수행할 수 있도록 설계된 텍스트 기반 API를 지원합니다. iVISPAR는 퍼즐의 복잡도를 세밀하게 조정할 수 있으며, 다양한 기준 모델과의 성능 비교가 가능합니다. 최적의 솔루션은 A* 알고리즘을 사용하여 계산되며, 벤치마크는 보드 크기, 타일 수 및 솔루션 경로 등 다양한 요인을 조정하여 복잡도를 확장할 수 있습니다.

- **Performance Highlights**: 실험 결과, 최신 VLM들이 기본적인 공간 추론 작업을 처리할 수 있지만, 3D 환경의 보다 복잡한 시나리오에 직면할 때 상당한 어려움을 보임을 확인했습니다. VLM은 일반적으로 2D 비전에서 더 나은 성능을 보였으나, 인간 수준의 성과에는 미치지 못하는 한계를 보여주었습니다. 이러한 결과는 VLM의 현재 능력과 인간 수준의 공간 추론 간의 지속적인 격차를 강조하며, VLM 연구의 추가 발전 필요성을 시사합니다.



### RoboGrasp: A Universal Grasping Policy for Robust Robotic Contro (https://arxiv.org/abs/2502.03072)
- **What's New**: RoboGrasp는 기존의 로봇 잡기 기술을 발전시키기 위해 사전 훈련된 잡기 감지 모델을 통합한 범용 잡기 정책 프레임워크를 제안합니다. 이는 로봇이 복잡한 환경에서도 정확하고 안정적인 조작을 가능하게 하여, 몇 번의 학습만으로도 최대 34% 높은 성공률을 달성할 수 있게 합니다. 이 방식은 다양한 로봇 학습 패러다임에 적용할 수 있으며, 실제 환경에서의 잡기 문제를 해결하기 위한 확장 가능하고 다재다능한 솔루션을 제공합니다.

- **Technical Details**: RoboGrasp는 Grasp Detection Module과 함께 동작하며, 이는 YOLOv11-m을 활용하여 잡기 상자의 중심 좌표와 크기를 신속하게 예측합니다. 관찰 인코더는 시각적 데이터와 저차원 데이터를 결합하여 단일 잠재 표현으로 변환하며, ResNet34 기반의 특징 피라미드 인코더를 통해 다중 RGB 데이터를 처리합니다. 여기서 구현된 특이점 중 하나는 잡기 상자 특성을 관찰 데이터에 추가하여 정책의 일반화 능력을 향상시키는 것입니다.

- **Performance Highlights**: 로봇의 작업 수행 능력을 평가하기 위해 세 가지 주요 작업(PickBig, PickCup, PickGoods)을 설계했습니다. 이들은 로봇이 다양한 크기와 유형의 물체를 정확하게 잡을 수 있는 능력을 테스트합니다. 특히, PickBig 작업은 크기가 유사한 두 개의 블록 중 더 큰 것을 구분하고 잡는 능력을 평가하는데, 이는 로봇의 잡기 전략을 환경에 맞게 조정하는 데 중점을 둡니다.



### Elucidating the Preconditioning in Consistency Distillation (https://arxiv.org/abs/2502.02922)
Comments:
          Accepted at ICLR 2025

- **What's New**: 이번 연구는 consistency distillation에서의 preconditioning 설계를 처음으로 이론적으로 분석하고, teacher ODE 경로와의 연결성을 설명합니다. 이를 통해 consistency gap을 기반으로 한 'Analytic-Precond'라는 새로운 방법론을 제안하며, 이 방법이 consistency trajectory 모델의 학습을 어떻게 최적화할 수 있는지 소개합니다. 특히, 이 접근 방식은 수작업 설계나 하이퍼파라미터 조정 없이 teacher 모델에 따라 효율적으로 계산됩니다.

- **Technical Details**: Diffusion models는 고차원 데이터 분포를 가우시안 노이즈 분포로 변환하는 과정에서 stochastic differential equation (SDE)을 사용합니다. 이 과정에서, 모델은 noise에서 시작하여 probability flow (PF) ODE를 역으로 시뮬레이션하여 표본을 생성합니다. 기존의 preconditioning 방법이 손수 제작된 것에 비해 최적화되지 않을 수 있었지만, 본 연구에서는 teacher ODE의 이산화와 관련하여 preconditioning의 설계 기준을 제시함으로써 보다 효과적인 학습 구조를 구축합니다.

- **Performance Highlights**: Analytic-Precond는 다양한 데이터셋에서 consistency models (CMs)와 consistency trajectory models (CTMs)에 적용하여 검증되었습니다. 이 방법은 다단계 생성에서 2배에서 3배까지의 훈련 속도 향상을 이끌어내며, 특히 CTMs에서는 중간 점프의 효과적인 학습을 촉진합니다. 기존의 preconditioning 방법과 비교했을 때, Analytic-Precond는 모델의 성능과 안정성을 한층 더 강화하는 데 기여합니다.



### INST-Sculpt: Interactive Stroke-based Neural SDF Sculpting (https://arxiv.org/abs/2502.02891)
- **What's New**: 이번 연구에서는 Neural Signed Distance Functions (neural SDFs)을 위한 인터랙티브한도구킷 INST-Sculpt를 소개합니다. 이 툴킷은 사용자가 직관적으로 3D 형상을 조각할 수 있도록 스트로크 기반(stroke-based) 편집 기능을 제공합니다. 기존의 점 기반 편집 방식과 달리, 사용자는 자신의 브러시 프로필을 정의하고 실시간으로 표면에 적용할 수 있습니다.

- **Technical Details**: INST-Sculpt는 효율적인 튜브 샘플링(tubular sampling) 기술을 사용하여 사용자 정의 스트로크를 따라 매끄럽고 연속적인 변형을 생성합니다. 이를 통해 전통적인 조각 도구와 유사한 실시간 피드백과 직관적인 제어가 가능합니다. 또한, 다양한 사용자가 정의한 브러시 프로필을 통합할 수 있는 유연성을 제공하여 창의적인 편집의 잠재력을 높입니다.

- **Performance Highlights**: 본 연구에서 제안한 방법은 이전 지점 기반(point-based) 방법 대비 최대 16배의 속도 향상을 달성하여 편집 중 거의 즉각적인 사용자 피드백을 제공합니다. 다양한 실험을 통해 이 방법의 효율성, 유연성 및 표현력이 검증되었습니다.



### Learning Generalizable Features for Tibial Plateau Fracture Segmentation Using Masked Autoencoder and Limited Annotations (https://arxiv.org/abs/2502.02862)
Comments:
          5 pages, 6 figures

- **What's New**: 이 논문에서는 슬개골 지지 구역 골절(TPF)의 자동 분할을 위한 새로운 훈련 전략을 제시합니다. 제안된 방법은 Masked Autoencoder (MAE)를 활용하여 레이블이 없는 데이터로부터 전신 해부 구조 및 세부적인 골절 정보를 학습하여, 레이블이 있는 데이터의 사용을 최소화합니다. 이렇게 함으로써, 기존의 반지도학습 방법들이 직면한 문제를 해결하고, 다양한 골절 패턴의 일반화 및 전이 가능성을 향상시킵니다.

- **Technical Details**: 제안된 분할 네트워크는 MAE 프리트레이닝 단계와 UNETR 파인튜닝 단계로 구성됩니다. MAE는 비대칭 인코더-디코더 설계를采用하여 입력 이미지의 일부 패치를 마스킹하고, 나머지 마스크된 패치를 재구성하는 방식으로 작동합니다. UNETR은 인코더에서 생성된 피처 시퀀스를 디코더에서 해상도를 복원하는 데 사용하여 고해상도 분할 출력을 생성합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 180개의 TPF CT 스캔으로 구성된 자체 데이터세트에서 평균 Dice 유사성 계수 95.81%, 평균 대칭 표면 거리 1.91mm, 그리고 하우스도르프 거리 9.42mm를 달성하며 반지도학습 방법을 일관되게 초월합니다. 것이 주목할 점은 공용 골반 CT 데이터셋에서의 좋은 전이 가능성을 보여주며 다양한 골절 분할 작업에서의 응용 가능성을 강조합니다.



### A Decade of Action Quality Assessment: Largest Systematic Survey of Trends, Challenges, and Future Directions (https://arxiv.org/abs/2502.02817)
Comments:
          36 Pages, 20 Figures, 12 Tables

- **What's New**: 이번 논문은 Action Quality Assessment (AQA)의 최신 발전 상황을 포괄적으로 정리한 서베이를 제공합니다. AQA는 인간의 움직임과 행동의 질을 평가하는 중요한 분야로, 스포츠 훈련, 저비용 물리치료, 직무 개발 등 다양한 영역에서 활용됩니다. 이 연구에서는 200편 이상의 연구 논문을 PRISMA 프레임워크를 통해 체계적으로 검토하여 AQA의 기본 개념, 일반 프레임워크, 성능 지표 및 최신 기술 발전을 심층적으로 논의합니다.

- **Technical Details**: AQA 문제를 정의하면, 행동(action)의 질은 두 가지 주요 구성 요소로 나눌 수 있습니다: 난이도 수준(difficulty level)과 실행 품질(execution quality)입니다. 예를 들어, 피크 자세에서의 공중제비는 Tuck 자세보다 난이도가 더 높고, 실행 품질 역시 발이 서로 붙어 있을 때 더 높은 점수를 부여받습니다. 이러한 요소를 정확히 포착하기 위해 CNN과 같은 모델은 행동 시퀀스에서 특성을 식별하고 그 중요도를 평가하는 기능이 필요합니다.

- **Performance Highlights**: AQA는 사회적 형평성을 촉진하고 산업 효율성을 향상시킬 수 있는 가능성을 지닌 기술입니다. 저비용의 AI 기반 평가 시스템은 자원이 부족한 지역의 운동선수에게 맞춤형 피드백을 제공하고, 산업 훈련에서 직원의 기술 수준을 객관적으로 평가하는 데 기여합니다. AQA 시스템은 인간과 로봇 시스템 모두를 평가하여 작업장의 안전성과 효율성을 향상시키고, 공정하고 접근 가능한 평가 시스템을 개발함으로써 사회 전반에 걸쳐 긍정적 영향을 미칠 것으로 예상됩니다.



### SD++: Enhancing Standard Definition Maps by Incorporating Road Knowledge using LLMs (https://arxiv.org/abs/2502.02773)
- **What's New**: 이 연구에서는 SD(표준 정의) 맵을 HD(고해상도) 맵 수준으로 향상시키기 위한 새로운 접근 방식인 SD++를 제안합니다. LLM(대형 언어 모델)을 활용하여 도로 매뉴얼과 같은 공개적으로 사용 가능한 자료에서 정보를 통합함으로써, HD 맵 생성에 의존하지 않고도 SD 맵을 강화할 수 있는 가능성을 탐구하고 있습니다. 이 방법은 여러 지역의 도로 설계 가이드를 포함하여 일반화 가능성을 보여줍니다.

- **Technical Details**: SD++ 파이프라인은 OSM(OpenStreetMap) 데이터와 도로 설계 매뉴얼을 결합하여 지리적 요소의 정확성을 높이는 과정을 자동화합니다. 연구에서는 LLM을 활용하여 도로의 세부 지표를 추출하고, 도로 규격과 같은 정보가 HD 맵의 디테일에 얼마나 근접하게 할 수 있는지를 설명합니다. RAG(정보 검색 및 생성)는 외부 자원으로부터 정보를 통합하여 언어 모델의 출력을 개선합니다.

- **Performance Highlights**: 미국과 일본에서의 실험 결과, SD++는 다양한 도로 디자인 지침을 적용함으로써 지역 간 일반화 능력을 입증합니다. SD++는 RH(고해상도) 맵의 중요한 기능을 유지하면서도 비용 효율적이고 확장 가능한 대안을 제공합니다. 이는 기존 HD 맵 생성 방식의 한계를 극복할 수 있는 가능성을 보여줍니다.



### When are Diffusion Priors Helpful in Sparse Reconstruction? A Study with Sparse-view C (https://arxiv.org/abs/2502.02771)
Comments:
          Accepted at IEEE ISBI 2025, 5 pages, 2 figures, 1 table

- **What's New**: 이 논문에서는 이미지 생성을 위한 최신 기법인 diffusion 모델을 sparse medical image reconstruction에 활용하는 방안을 탐구합니다. 기존의 분석적 priors와 비교해, diffusion 모델은 적은 수의 관측치로도 그럴듯한 결과를 생성할 수 있으나, 이는 곧 잘못된 결과를 초래할 수 있다는 위험성을 내포하고 있습니다. 연구는 기존의 sparse 및 Tikhonov 정규화 등의 classical priors와 diffusion priors의 성능을 비교하며, 특히 저선량 흉부 CT 이미지를 통해 결과의 유용성을 평가합니다.

- **Technical Details**: 연구에서 CT 데이터셋은 935명의 환자로부터 수집되었으며, 각 이미지는 surgical mastectomy 후 방사선 치료를 받은 사례들입니다. 2D U-Net diffusion 모델을 사용하여 CT 슬라이스를 조건 없이 재구성하였으며, 다양한 regularization 및 최적화 매개변수를 탐색하여 모델의 정확도를 개선하는 데 초점을 맞추었습니다. 평가 지표로는 PSNR, SSIM과 같은 pixel 기반 및 구조 기반 메트릭과 fat content accuracy, fat localization accuracy 등의 downstream 메트릭이 사용되었습니다.

- **Performance Highlights**: 연구 결과, diffusion priors는 매우 제한적인 관측치에서도 우수한 성능을 보여주며, 몇몇 세부 사항을 이미 성공적으로 캡처할 수 있음을 발견했습니다. 그러나 모든 세부 사항, 특히 혈관 구조는 정확히 포착하지 못했습니다. 반면, 충분한 수의 프로젝션이 있을 경우 classical priors가 더 나은 성능을 발휘했으며, diffusion priors는 약 10-15개의 프로젝션으로 성능이 정체되는 경향을 보였습니다.



### Federated Low-Rank Tensor Estimation for Multimodal Image Reconstruction (https://arxiv.org/abs/2502.02761)
- **What's New**: 이번 논문은 저랭크 텐서 추정을 활용하여 노이즈가 많거나 샘플링이 부족한 조건에서의 이미지 재구성 문제를 해결하는 새로운 연합 이미지 재구성 방법을 제안합니다. 이 방법에서 Tucker decomposition(터커 분해)을 사용하여 대규모의 다중 모드 데이터를 처리하며, 개인화된 디컴포지션 랭크를 선택할 수 있도록 지원합니다.

- **Technical Details**: 제안된 방법은 공동 인수화(joint factorization)와 랜덤 스케치(randomized sketching)를 통합하여 전체 크기의 텐서를 재구성하지 않고도 데이터를 효율적으로 다룰 수 있게 합니다. 이는 또한 통신 효율성을 향상시키고 클라이언트가 미리 알거나 통신 능력에 기반하여 개인 맞춤형 랭크를 선택할 수 있게 합니다.

- **Performance Highlights**: 수치 결과는 제안된 방법이 기존 접근 방식에 비해 뛰어난 재구성 품질 및 통신 압축을 달성했음을 보여줍니다. 이러한 특징은 연합 학습(federated learning) 설정에서 다중 모드 역문제를 해결하는 데 있어 큰 잠재력을 갖고 있음을 강조합니다.



### Adaptive Voxel-Weighted Loss Using L1 Norms in Deep Neural Networks for Detection and Segmentation of Prostate Cancer Lesions in PET/CT Images (https://arxiv.org/abs/2502.02756)
Comments:
          29 pages, 7 figures, 1 table

- **What's New**: 이 연구에서는 L1 가중된 Dice Focal Loss (L1DFL)라는 새로운 손실 함수(loss function)를 제안합니다. 이 손실 함수는 분류 난이도에 따라 voxel의 가중치를 조정하여 전이성 전립선 암 병변을 자동으로 탐지하고 분할하는 데 목적을 두고 있습니다. 연구팀은 생화학적 재발 전이성 전립선 암으로 진단 받은 환자들의 PET/CT 스캔을 분석하였습니다.

- **Technical Details**: L1DFL은 L1 norm을 기반으로 하여 각 샘플의 분류 난이도와 데이터셋 내 유사 샘플의 출현 빈도를 고려한 동적 가중치 조정 전략을 채택합니다. 이 연구의 주요 기여는 새로운 L1DFL 손실 함수의 도입과 함께 Dice Loss 및 Dice Focal Loss 함수와의 성능 비교를 통해 거짓 양성 및 거짓 음성 비율에 대한 효율성을 평가한 것입니다.

- **Performance Highlights**: L1DFL은 테스트 세트에서 비교 손실 함수보다 최소 13% 높은 성능을 보였습니다. Dice Loss 및 Dice Focal Loss의 F1 점수는 L1DFL에 비해 각각 최소 6% 및 34% 낮았고, 이로 인해 L1DFL이 전이성 전립선 암 병변의 견고한 분할을 위한 가능성을 제시합니다. 또한 이 연구는 병변 특성의 변동성이 자동화된 전립선 암 탐지와 분할에 미치는 영향을 강조합니다.



### Intelligent Sensing-to-Action for Robust Autonomy at the Edge: Opportunities and Challenges (https://arxiv.org/abs/2502.02692)
- **What's New**: 이 논문은 로봇 공학, 스마트 시티 및 자율주행차에서의 자율 에지 컴퓨팅의 새로운 접근 방법을 탐구합니다. 이전의 중앙 집중식 시스템과 다르게, 센싱-투-액션(loop) 메커니즘은 하이퍼 로컬 조건에 적응하여 효율성을 높이고 리소스를 최적화합니다. 또한, 다양한 환경에서의 자율성을 향상시키기 위한 프로액티브(proactive)인 센싱 및 행동 조정을 강조합니다.

- **Technical Details**: 논문의 핵심 개념은 '센싱-투-액션(loop)'으로, 이 루프는 센서 입력과 계산 모델을 반복적으로 정렬하여 적응형 제어 전략을 구동합니다. 여러 에이전트 간의 협력을 통해 자원 사용을 최적화하며, 신경모방 컴퓨팅(neuromorphic computing)을 통한 스파이크 기반(spike-based) 이벤트 구동 처리 방식이 제시됩니다. 이러한 기술들은 자원의 제약 속에서도 연관성 있는 결정을 내릴 수 있도록 돕습니다.

- **Performance Highlights**: 에지 시스템에서는 자원의 제약으로 인해 실시간 처리에서의 지연과 부정확한 데이터가 큰 문제로 부각됩니다. 하지만, 이 루프들은 효율적인 자원 사용을 통해 높은 정확성과 적응성을 달성할 수 있는 기회를 제공합니다. 또한, 각 작업의 특성에 맞춰 리소스를 조정함으로써 효율성을 더욱 높이고, 엔드 투 엔드(end-to-end) 공동 설계를 통한 시스템 개선 전략을 강조합니다.



### SiLVR: Scalable Lidar-Visual Radiance Field Reconstruction with Uncertainty Quantification (https://arxiv.org/abs/2502.02657)
Comments:
          webpage: this https URL

- **What's New**: 이 논문에서는 lidar와 비전 데이터를 융합하여 고품질의 재구성을 생성하는 대규모 신경 방사장(NeRF) 기반 시스템인 SiLVR을 소개합니다. SiLVR은 깊이와 표면 노말에 대한 강력한 기하학적 제약을 추가하여 균일한 텍스처의 표면을 모델링하는 데 특히 유용합니다. 이 시스템은 또한 카메라와 lidar 센서 관측치를 바탕으로 각 포인트 위치의 공간 분산을 이용해 재구성의 인식 불확실성을 추정합니다.

- **Technical Details**: SiLVR은 NeRF 연구와 Nerfacto 구현을 바탕으로 하며, lidar로부터의 기하학적 제약을 추가하여 재구성 품질을 향상시킵니다. 이 방법에서는 lidar 스캔으로부터 표면 노말을 추정하여 부드러운 표면 재구성을 촉진하며, 이는 학습 기반 노말 추정 접근 방식의 입력 데이터 분포 이동 문제를 피할 수 있습니다. 또한, 공간적 분산을 재구성의 인식 불확실성 측정으로 활용하여 아티팩트를 필터링하고, 최종 재구성 정확도를 향상시킵니다.

- **Performance Highlights**: 논문에서는 옥스퍼드 스파이어 데이터셋에서 대규모 평가를 실시하여 밀리미터 단위의 정확한 3D 기준 진실과 정량적인 결과를 제시합니다. 새로운 재구성 방법은 기존 SDF 및 NeRF 기반 방법들과 비교하여 우수한 성능을 보입니다. SiLVR은 구축된 각 서브맵이 시각적 중첩을 줄여 아티팩트를 최소화하고, 전체 맵의 품질 향상에 기여한다는 것을 보여줍니다.



### ParetoQ: Scaling Laws in Extremely Low-bit LLM Quantization (https://arxiv.org/abs/2502.02631)
- **What's New**: 이 논문에서는 quantization(양자화) 모델의 크기와 정확도 간의 최적의 trade-off(절충점)을 찾기 위한 통합된 프레임워크인 ParetoQ를 제안합니다. 이는 1비트, 1.58비트, 2비트, 3비트, 4비트 양자화 설정 간의 비교를 보다 rigorously(엄격하게) 수행할 수 있도록 합니다. 또한, 2비트와 3비트 간의 학습 전환을 강조하며 이는 정확한 모델 성능에 중요한 역할을 합니다.

- **Technical Details**: ParetoQ는 파라미터 수를 최소화하면서도 성능을 극대화할 수 있는 모델링 기법입니다. 실험 결과, 3비트 및 그 이상의 모델은 원래 pre-trained distribution(사전 훈련 분포)과 가까운 성능을 유지하는 반면, 2비트 이하의 네트워크는 표현 방식이 급격히 변화합니다. 또한, ParetoQ는 이전의 특정 비트 폭에 맞춘 방법들보다 우수한 성능을 보여줍니다.

- **Performance Highlights**: 작은 파라미터 수에도 불구하고, ParetoQ의 ternary 600M-parameter 모델은 기존의 SoTA(최신 기술 상태)인 3B-parameter 모델을 넘는 정확도를 기록했습니다. 다양한 실험을 통해 2비트와 3비트 양자화가 사이즈와 정확도 간의 trade-off에서 우수한 성능을 보여주며, 4비트 및 binary quantization 글리 제너럴 단위 성능이 하락하는 경향을 보였습니다.



### Muographic Image Upsampling with Machine Learning for Built Infrastructure Applications (https://arxiv.org/abs/2502.02624)
- **What's New**: 이 논문은 연약한 인프라(aging critical infrastructure)의 비파괴 검사(non-destructive evaluation) 방법에 대한 새로운 접근법인 muography를 제안합니다. 특히 고령의 교량과 같은 구조물의 검사를 위한 non-invasive imaging(PET, CT)에 대한 심도 있는 통찰을 제공합니다. 기존 muography 기술의 한계를 해결하기 위해 두 개의 Deep Learning 모델을 사용하여 영상 개선과 세그멘테이션(semantic segmentation) 성능을 향상시켰습니다.

- **Technical Details**: 이 연구에서는 cWGAN-GP(conditional Wasserstein generative adversarial network with gradient penalty) 모델을 사용하여 undersampled muography 이미지를 예측적 업샘플링(predictive upsampling)했습니다. 이를 통해 1일 샘플 이미지가 21일 샘플 이미지와 유사한 인지적 특성을 보여주었으며, PSNR(peak signal-to-noise ratio) 개선이 31일 샘플 링과 동등함을 입증했습니다. 추가적으로, 두 번째 cWGAN-GP 모델은 세멘틱 세그멘테이션을 통해 Concrete 샘플의 특징을 정량적으로 분석했습니다.

- **Performance Highlights**: 이 논문의 결과는 muography의 영상 품질과 획득 속도를 유의미하게 개선하여, 강화 콘크리트 구조 모니터링을 위한 보다 실용적인 muography 기술의 가능성을 보여주었습니다. 세그먼테이션 정확도(Dice-Sörensen accuracy coefficients)는 각각 0.8174와 0.8663으로 나타났습니다. 이러한 발전은 muography가 산업계에 더 매력적인 기술로 자리잡을 수 있도록 기여하고 있습니다.



### Secure & Personalized Music-to-Video Generation via CHARCHA (https://arxiv.org/abs/2502.02610)
Comments:
          NeurIPS 2024 Creative AI Track

- **What's New**: 이 논문에서는 사용자가 음악 비디오 생성 과정의 공동 창작자가 될 수 있도록 돕는 완전 자동화된 파이프라인을 발표합니다. 이는 음악의 가사, 리듬 및 감정을 기반으로 개인화되고 일관되며 맥락에 맞는 비주얼을 생성하는 데 중점을 두고 있습니다. 또한 CHARCHA라는 얼굴 확인 프로토콜을 도입하여 개인의 얼굴 정보의 무단 사용을 방지하고 사용자가 제공한 이미지를 수집하여 비디오 개인화를 지원합니다.

- **Technical Details**: 우리는 다중 모달리티(모델, 오디오, 시각 언어)를 활용하여 개인화된 음악 비디오를 생성하는 파이프라인을 구축했습니다. 이 과정에서 전이 학습된 여러 모델과 API를 통합하여 오디오 필기, 텍스트-이미지 변환, 음악 감정 인식 등의 작업을 수행합니다. 또한, LoRA(저랭크 적응)를 사용하여 사용자의 이미지를 삽입하고 개인화된 경험을 제공합니다.

- **Performance Highlights**: 기존 방법들과는 달리, 본 연구는 음악의 오디오 파일만으로도 개인화된 비디오를 생성해 내며, 사용자들은 자신의 취향을 반영한 비디오 제작에 참여하게 됩니다. 우리의 자동화된 파이프라인은 다양한 음악 장르와 언어에 적응 가능하며, 지역별 시각 스타일을 포함하는 등 높은 유연성을 제공합니다. 전반적으로, 이 시스템은 음악 비디오 생성의 새로운 트렌드를 선도하며, 개인화와 보안을 두루 고려한 혁신적인 접근 방식을 제시합니다.



New uploads on arXiv(cs.AI)

### BFS-Prover: Scalable Best-First Tree Search for LLM-based Automatic Theorem Proving (https://arxiv.org/abs/2502.03438)
- **What's New**: 최근 대규모 언어 모델(LLMs)의 발전은 Lean4를 활용한 자동 정리 증명(Automatic Theorem Proving, ATP)에 대한 관심을 불러일으켰습니다. 이 논문에서는 Best-First Search (BFS)를 사용하여 대규모 정리 증명 작업에서도 경쟁력을 갖출 수 있는지 탐구합니다. 새로운 시스템인 BFS-Prover를 통해 비즈니스 모델을 개선하고 성능을 높일 수 있는 세 가지 주요 혁신을 제시합니다.

- **Technical Details**: BFS-Prover는 전략적인 데이터 필터링, 직접 선호 최적화(Direct Preference Optimization, DPO), 그리고 경로 길이 정규화(length normalization)를 통해 BFS의 샘플 효율성을 향상시키고 있습니다. DPO는 증명 상태에서 발생하는 긍정적인 전술과 부정적인 전술의 쌍을 활용하여 LLM의 정책 분포를 조정합니다. 경로 길이 정규화는 BFS가 더 깊은 증명 경로를 탐색하도록 유도하여, 복잡한 정리에 필요한 긴 전술 체인을 탐색할 수 있게 합니다.

- **Performance Highlights**: BFS-Prover는 MiniF2F 테스트 세트에서 71.31이라는 점수를 기록하며, BFS가 복잡한 트리 탐색 방식 없이도 경쟁력 있는 성과를 달성할 수 있음을 보여줍니다. 이를 통해 BFS는 MCTS와 같은 복잡한 방법들에 대한 효과적인 대안으로 자리 잡을 수 있는 가능성을 암시합니다. 이러한 결과는 BFS를 활용한 ATP의 새로운 가능성을 열어줄 것입니다.



### Learning from Active Human Involvement through Proxy Value Propagation (https://arxiv.org/abs/2502.03369)
Comments:
          NeurIPS 2023 Spotlight. Project page: this https URL

- **What's New**: 이 논문에서는 인간의 능동적인 참여를 통해 AI 에이전트를 교육하는 새로운 방법인 Proxy Value Propagation (PVP)를 제안합니다. 이 방법은 보상 없이 학습을 진행하며, 인간의 의도를 표현할 수 있는 프록시 가치 함수를 설계합니다. 인간의 시연에서 행동-상태 쌍에 높은 값을 부여하고, 인간의 개입을 받은 에이전트의 행동에는 낮은 값을 부여하여 학습 효율성을 높입니다. 실험 결과 PVP는 다양한 작업에서 높은 학습 효율성과 사용자 친화성을 보여주었습니다.

- **Technical Details**: PVP는 기존의 가치 기반 강화 학습 방법에 최소한의 수정으로 통합할 수 있는 간단하면서도 효과적인 방법입니다. TD(Temporal Difference) 학습 프레임워크를 사용하여 인간의 시연에서 레이블이 붙은 값을 에이전트의 탐색으로부터 생성된 다른 레이블 없는 데이터로 전파합니다. 이 과정에서 프록시 가치 함수는 인간의 행동을 충실히 모방하는 정책을 유도하며, 다양한 연속 및 이산 제어 작업에서 활용될 수 있습니다. 특히, 이 방법은 게임패드, 핸들, 키보드와 같은 여러 형태의 인간 제어 장치에서도 호환됩니다.

- **Performance Highlights**: PVP는 MiniGrid, MetaDrive, CARLA 및 Grand Theft Auto V(GTA V)와 같은 다양한 환경에서 우수한 성능을 보였습니다. 사용자 연구 결과, PVP는 다른 인간-루프 방법보다 더 나은 성능을 발휘하며 사용자 친화적임을 증명합니다. 이 연구는 보상 기반 방법의 단점을 극복하고, 인간의 적극적인 개입을 통해 AI의 학습 안전성을 높일 수 있는 가능성을 보여줍니다. PVP의 결과는 향후 실제 응용 프로그램 접목에 대한 중요한 통찰력을 제공합니다.



### PalimpChat: Declarative and Interactive AI analytics (https://arxiv.org/abs/2502.03368)
- **What's New**: 이번 연구에서는 PalimpChat이라는 새로운 채팅 기반 인터페이스를 도입하여 Palimpzest와 Archytas를 통합한 시스템을 소개합니다. PalimpChat은 사용자가 자연어(Natural Language)로 복잡한 AI 파이프라인을 설계하고 실행할 수 있도록 하여, 다시 말해 비전문가도 기술적으로 접근할 수 있게 만듭니다. 이 시스템은 과학 발견, 법적 발견, 부동산 검색 등 세 가지 실제 시나리오를 제공하며, 관련 데이터셋을 쉽게 탐색하고 처리할 수 있도록 돕습니다.

- **Technical Details**: Palimpzest는 비구조화 데이터 처리 파이프라인을 자동으로 최적화하여 구축하는 선언적(declarative) 시스템으로 구성되어 있습니다. 사용자는 데이터셋과 스키마를 정의한 후, 변환을 지정하여 최종 출력을 형성할 수 있습니다. Archytas는 LLM 에이전트를 생성하여 다양한 도구와 상호작용할 수 있도록 지원하는 도구 상자(toolbox)로, 사용자가 요청을 작은 단계로 분해하여 해결책을 찾는 데 도움을 줍니다.

- **Performance Highlights**: PalimpChat은 비전문가 사용자가 침착하게 파이프라인을 설계하고 실행할 수 있는 환경을 제공합니다. 또한, 전문가 사용자는 생성된 코드를 수정하거나 직접 Palimpzest 내에서 파이프라인을 프로그래밍할 수 있는 유연성을 가지게 됩니다. 이 시스템을 통해 사용자는 과학적 연구 등 다양한 분야에서 실제 데이터를 기반으로 한 복잡한 분석 작업을 수행할 수 있는 가능성을 탐색할 수 있습니다.



### SymAgent: A Neural-Symbolic Self-Learning Agent Framework for Complex Reasoning over Knowledge Graphs (https://arxiv.org/abs/2502.03283)
- **What's New**: 이번 논문에서는 SymAgent라는 혁신적인 신경-기호 에이전트 프레임워크를 소개합니다. 이 프레임워크는 Knowledge Graphs (KGs)와 Large Language Models (LLMs)의 협력적 증대를 통해 복잡한 추론 문제를 해결하는 데 중점을 두고 있습니다. KGs를 동적 환경으로 간주하여, 복잡한 추론 작업을 다단계 인터랙티브 프로세스로 변환해 LLM이 더 깊고 의미 있는 추론을 할 수 있도록 지원합니다.

- **Technical Details**: SymAgent는 두 개의 모듈인 Agent-Planner와 Agent-Executor로 구성되어 있습니다. Agent-Planner는 LLM의 유도적 추론 능력을 활용하여 KGs에서 기호 규칙을 추출하고, 효율적인 질문 분해를 안내합니다. Agent-Executor는 미리 정의된 액션 툴을 자율적으로 호출하며 KGs와 외부 문서의 정보를 통합해 KG의 불완전성 문제를 해결합니다.

- **Performance Highlights**: 실험 결과, SymAgent는 약한 LLM 백본(예: 7B 시리즈)을 사용하더라도 다양한 강력한 기준선과 비교해 더 나은 또는 동등한 성능을 보여주었습니다. 분석 결과, 에이전트는 누락된 트리플을 식별하여 자동으로 KG를 업데이트 할 수 있는 능력을 가지고 있으며, 이는 KG의 불완전성 문제에 효과적으로 대응할 수 있게 해줍니다.



### A Scalable Approach to Probabilistic Neuro-Symbolic Verification (https://arxiv.org/abs/2502.03274)
- **What's New**: 이 논문은 Neuro-Symbolic Artificial Intelligence (NeSy AI) 시스템의 확장된 약화 기반 검증 기법을 소개하여, 확률적 추론에 대한 새로운 접근 방식을 제안합니다. 기존의 순수 신경망 기반 시스템에서 형성된 검증 기술을 NeSy 설정으로 확장하여 실질적인 적용 가능성을 높였습니다. 특히, 복잡한 심볼릭 컴포넌트를 포함한 시스템의 검증을 위한 공식을 명확히 하고, 이를 통해 도출된 안전성 속성과 실제 자율주행 데이터셋에서의 적용사례를 보여줍니다.

- **Technical Details**: NeSy 시스템은 신경망(Neural Network)과 심볼릭 로직(Symbolic Logic)의 통합을 통해 확률적 추론을 수행합니다. 해당 시스템은 latent concepts를 기반으로 신경망의 출력을 심볼릭 지식과 결합하여 확률적 결과를 나타냅니다. 또한, 검증 문제의 복잡성을 NP^{#P}-hard으로 분류하고, 기존의 solver 기반 접근 방식의 한계로 인해 제안된 방법이 더욱 효율적임을 증명합니다.

- **Performance Highlights**: 제안된 방법은 기존의 solver 기반 솔루션에 비해 지수적으로 나은 성능을 보여주며, 높은 차원의 입력과 실제 네트워크 크기를 다루는 실제 문제에 적용 가능한 가능성이 높습니다. 자율주행 데이터셋을 활용하여 안전속성을 검증하며, 이는 NeSy 시스템의 신뢰할 수 있는 성능을 보장하기 위한 중요한 단계를 제공합니다. 이로 인해 비판적인 도메인에서 NeSy AI의 안전한 배포가 가능해질 전망입니다.



### CORTEX: A Cost-Sensitive Rule and Tree Extraction Method (https://arxiv.org/abs/2502.03200)
- **What's New**: 이 논문에서는 Cost-Sensitive Rule and Tree Extraction (CORTEX) 방법이 제안되어, 이는 다중 클래스 분류 문제에 적합하게 확장된 비용 민감 결정 트리(CSDT) 방법에 기반한 새로운 규칙 기반 설명 가능 인공지능(XAI) 알고리즘입니다. CORTEX는 성능이 뛰어난 규칙 추출 XAI 방법으로 평가되고, 다양한 데이터셋에서 다른 기존의 트리 및 규칙 추출 방법들과 비교되었습니다. 이 연구의 결과는 CORTEX가 다른 규칙 기반 방법들보다 우수한 성능을 가지고 있음을 보여줍니다.

- **Technical Details**: CORTEX 방법은 기본적으로 클래스 종속 비용 행렬을 유도함으로써 다중 클래스 분류 문제를 다루며, 추출된 규칙 집합은 장비 가용성 및 예측 성능을 유지하면서 보다 작고 짧은 규칙으로 구성됩니다. 이 방법은 기존 XAI 알고리즘인 C4.5-PANE, REFNE, RxREN 및 TREPAN과 비교되며, 각 방법의 규칙 추출 능력과 설명 가능성을 분석하기 위해 정량적 평가 지표가 사용됩니다. CORTEX는 또한 네트워크의 내부 요소를 사용하지 않고 재 라벨링된 목표 변수를 통해 트리 모델을 생성하는 pedagogical 접근 방식을 활용합니다.

- **Performance Highlights**: 실험 결과에 따르면, CORTEX는 다양한 클래스 수를 가진 데이터셋에서 평균적으로 더 작은 규칙 집합과 짧은 규칙을 생성하여 다른 규칙 기반 방법보다 월등한 성과를 보였습니다. 평가 지표를 통해 확인된 바와 같이, CORTEX는 설명 가능성에서 높은 정확도를 보이며, 모든 샘플을 커버하는 완전성을 제공하는 데 성공합니다. 전체적으로 CORTEX는 명확하고 인간이 이해 가능한 규칙을 생성하는 강력한 XAI 도구로서의 잠재력을 보여줍니다.



### The Cake that is Intelligence and Who Gets to Bake it: An AI Analogy and its Implications for Participation (https://arxiv.org/abs/2502.03038)
- **What's New**: 이 논문에서는 AI 시스템의 전체 생애주기를 케이크에 비유하여 다루며, 데이터의 출처와 훈련 과정 그리고 평가 및 배포의 단계를 설명합니다. 이러한 재구성은 머신러닝의 통계적 가정을 통해 소셜 임팩트를 설명하고, 기술적 기초와 사회적 결과 간의 연결을 개선할 수 있는 방법을 모색합니다. AI 전문가와 연구자들이 인식의 폭을 넓히고, 더 나아가 AI 설계 과정에 적극 참여할 수 있는 기회를 제공합니다.

- **Technical Details**: 본 논문에서는 AI 생애주기 각 단계에서 수반되는 기술적 한계를 분석하며, AI 시스템이 의존하는 데이터의 출처와 훈련 데이터셋에 대한 불투명한 관행을 비판적으로 고찰합니다. 예를 들어, 특정 사회적 맥락이 반영되지 않은 데이터 수집 방법이나, 개인의 동의 없이 개인정보가 AI 모델에 사용될 가능성을 문제삼습니다. 이렇게 데이터의 출처와 처리 과정을 명확하게 이해함으로써, AI 시스템의 설계와 윤리적 측면에서 더 나은 결정을 내릴 수 있는 기반을 마련하고자 합니다.

- **Performance Highlights**: AI 시스템의 접근성과 투명성을 제고하기 위한 여러 가지 행동 권고 사항이 마련되었습니다. 예를 들어, 공정한 데이터 수집을 위한 윤리적 기준 및 투명한 공급망을 요구하는 법 제정 등의 필요성이 강조됩니다. 이러한 권고 사항은 궁극적으로 AI 설계 및 실행 과정에서 사회적 책임을 증대시키고, 다양한 이해관계자의 포괄적인 참여를 장려하는 데 기여할 것입니다.



### FedMobileAgent: Training Mobile Agents Using Decentralized Self-Sourced Data from Diverse Users (https://arxiv.org/abs/2502.02982)
- **What's New**: 이 논문에서는 FedMobileAgent라는 협력적 학습 프레임워크를 제안하여 모바일 에이전트를 훈련시키기 위한 자가 출처 데이터의 자동 수집 방식을 탐구합니다. 이 방법론은 사용자의 일상적인 전화 사용에서 발생하는 데이터를 활용하여 고품질 데이터 집합을 구성하고, 기존의 고비용 인간 주석 방식과는 달리 자동 주석 기법인 Auto-Annotation을 도입합니다. 또한, 데이터 보호를 위해 연합 학습(Federated Learning)을 통합하여 익명의 분산 데이터 수집을 가능하게 합니다.

- **Technical Details**: FedMobileAgent 시스템은 두 가지 주요 기술을 포함하고 있습니다. 첫째, Auto-Annotation 기술을 통해 사용자의 전화 사용 중 자동으로 고품질 데이터 세트를 수집합니다. 둘째, 다양한 사용자 데이터의 비정형성을 해결하기 위해 새로운 집계 방법인 adapted global aggregation을 도입하였으며, 이는 에피소드 및 스텝별 분포를 통합하여 성능을 극대화합니다. 이러한 접근 방식은 분산 데이터 환경에서도 모바일 에이전트 훈련의 효율성을 높이는 데 기여합니다.

- **Performance Highlights**: 실험 결과 FedMobileAgent는 기존의 인간 주석 모델과 비교해 비용은 0.01% 미만이면서 우수한 성능을 보여주었습니다. Auto-Annotation은 인간 주석 데이터보다 성능이 우수하며 주석 비용을 99% 절감할 수 있음을 입증하였습니다. 마지막으로, adapted aggregation은 비IID(non-IID) 상황에서 FedAvg보다 5%의 상대적 성능 향상을 달성하여, 실제 어플리케이션에서의 활용 가능성을 크게 높였습니다.



### (Neural-Symbolic) Machine Learning for Inconsistency Measuremen (https://arxiv.org/abs/2502.02963)
- **What's New**: 이 논문에서는 기계 학습(machine learning) 기반의 접근 방식을 통해 제안된 논리 지식 베이스의 불일치 정도(inconsistency degree)를 판단하는 방법을 제시합니다. 특히, 회귀(regression) 및 신경망(neural network) 모델을 사용하여 불일치 측정값 $	ext{incmi}$와 $	ext{incat}$의 예측 값을 도출하는 기법을 다룹니다. 기존의 전통적인 방법이 복잡도가 높아 불일치 측정 결과를 도출하는 것이 어려운 점을 극복하기 위한 연구입니다.

- **Technical Details**: 불일치 측정은 정보의 불일치 정도를 정량화 할 수 있는 다양한 수단을 제공하며, 이 논문에서는 특히 제안된 지식 베이스에 대해 수치 값을 통해 불일치 정도를 예측하는 방법을 다룹니다. 불일치 측정은 수학적 성질을 통해 얻어진 규칙을 기반으로 기계 학습 모델에 제약 조건으로 통합하여 더 정확한 예측을 가능하게 합니다. 이 연구에서는 여러 실험을 통해 회귀 및 신경망 기반 모델의 성능을 평가하며, 비율에서 유도된 기호적 제약을 통합하는 방법에도 중점을 둡니다.

- **Performance Highlights**: 실험 결과에 따르면, 학습된 기계 학습 모델은 많은 상황에서 유용한 근사값을 제공하며, 비율 원칙에서 도출된 기호적 제약을 포함함으로써 예측 성능이 향상됨을 보여주었습니다. 이러한 방법론은 금융 산업이나 사기 관리와 같은 분야에서 수많은 사례를 분석해야 할 때, 계산 복잡성을 줄이면서 적시에 근사된 불일치 값을 제공할 수 있는 잠재력을 가지고 있다고 평가됩니다.



### SensorChat: Answering Qualitative and Quantitative Questions during Long-Term Multimodal Sensor Interactions (https://arxiv.org/abs/2502.02883)
Comments:
          Under review

- **What's New**: SensorChat은 멀티모달 및 고차원 센서 데이터를 포함하여 장기 센서 모니터링을 위해 설계된 최초의 종단 간(End-to-End) QA 시스템입니다. 기존 시스템은 일반적으로 단기 데이터 제한이 있는 QA 방식으로 작동하였으나, SensorChat은 고급 추론을 요구하는 질적 질문과 정확한 센서 데이터를 기반으로 한 양적 질문 모두를 처리 할 수 있습니다. 이를 통해 사용자들이 일상 속에서 센서 데이터를 쉽게 이해하고 활용하도록 지원합니다.

- **Technical Details**: SensorChat은 질의 분해(question decomposition), 센서 데이터 쿼리(sensor data query), 답변 조합(answer assembly)의 세 가지 단계로 이루어진 혁신적인 파이프라인을 사용합니다. 첫 번째 및 세 번째 단계에서는 대규모 언어 모델(Large Language Models, LLMs)을 활용하여 사용자와의 직관적인 상호작용과 센서 데이터 쿼리 과정을 안내합니다. 특히, 기존의 멀티모달 LLMs와 달리, SensorChat은 장기간의 센서 데이터에서 사실 정보를 정확하게 추출하기 위한 명시적인 쿼리 단계를 통합합니다.

- **Performance Highlights**: SensorChat은 정량적 질문에 대해 기존의 최첨단 시스템보다 최대 26% 더 높은 답변 정확성을 달성하는 결과를 보였습니다. 또한, 사용자 연구를 통해 8명의 자원봉사자와의 상호작용에서 질적 및 개방형 질문 처리의 효율성을 입증하였습니다. SensorChat은 클라우드 서버 및 엣지 플랫폼에서 실시간 상호작용이 가능하며, 개인 정보 보호를 유지하면서도 사용자들에게 실질적이고 유용한 답변을 제공할 수 있도록 설계되었습니다.



### A Decade of Action Quality Assessment: Largest Systematic Survey of Trends, Challenges, and Future Directions (https://arxiv.org/abs/2502.02817)
Comments:
          36 Pages, 20 Figures, 12 Tables

- **What's New**: 이번 논문은 Action Quality Assessment (AQA)의 최신 발전 상황을 포괄적으로 정리한 서베이를 제공합니다. AQA는 인간의 움직임과 행동의 질을 평가하는 중요한 분야로, 스포츠 훈련, 저비용 물리치료, 직무 개발 등 다양한 영역에서 활용됩니다. 이 연구에서는 200편 이상의 연구 논문을 PRISMA 프레임워크를 통해 체계적으로 검토하여 AQA의 기본 개념, 일반 프레임워크, 성능 지표 및 최신 기술 발전을 심층적으로 논의합니다.

- **Technical Details**: AQA 문제를 정의하면, 행동(action)의 질은 두 가지 주요 구성 요소로 나눌 수 있습니다: 난이도 수준(difficulty level)과 실행 품질(execution quality)입니다. 예를 들어, 피크 자세에서의 공중제비는 Tuck 자세보다 난이도가 더 높고, 실행 품질 역시 발이 서로 붙어 있을 때 더 높은 점수를 부여받습니다. 이러한 요소를 정확히 포착하기 위해 CNN과 같은 모델은 행동 시퀀스에서 특성을 식별하고 그 중요도를 평가하는 기능이 필요합니다.

- **Performance Highlights**: AQA는 사회적 형평성을 촉진하고 산업 효율성을 향상시킬 수 있는 가능성을 지닌 기술입니다. 저비용의 AI 기반 평가 시스템은 자원이 부족한 지역의 운동선수에게 맞춤형 피드백을 제공하고, 산업 훈련에서 직원의 기술 수준을 객관적으로 평가하는 데 기여합니다. AQA 시스템은 인간과 로봇 시스템 모두를 평가하여 작업장의 안전성과 효율성을 향상시키고, 공정하고 접근 가능한 평가 시스템을 개발함으로써 사회 전반에 걸쳐 긍정적 영향을 미칠 것으로 예상됩니다.



### Planning with affordances: Integrating learned affordance models and symbolic planning (https://arxiv.org/abs/2502.02768)
Comments:
          10 pages, 2 figures

- **What's New**: 이 논문에서는 기존의 업무 및 동작 계획(task and motion planning, TAMP) 프레임워크에 신뢰할 수 있는 물체의 affordance 모델을 추가하여 지능형 에이전트가 복잡한 다단계 작업을 수행할 수 있도록 합니다. 이를 통해 에이전트는 다양한 환경 설정에서 작업을 수행하면서 액션 세트를 재정의하지 않고도 학습할 수 있습니다. 우리는 물리적인 3D 가상 환경인 AI2-Thor에서 이 접근 방식을 구현하고 평가하여, 에이전트가 환경과 어떻게 상호작용하는지를 신속하게 배울 수 있음을 보여줍니다.

- **Technical Details**: 이 연구에서는 PDDLStream 프레임워크를 확장하여 하이브리드 이산 및 연속 세계에서 복잡한 다단계 작업을 수행하기 위해 학습된 conditional samplers를 기반으로 한 affordance 모델을 도입합니다. 이러한 affordance 모델은 에이전트가 특정 상태에서 어떤 행동이 가능한지와 그 행동을 수행하는 방법을 명시합니다. 이로 인해 에이전트는 안정적인 계획과 실행을 통해 주어진 목표 상태로 이동할 수 있는 계획을 수립할 수 있습니다.

- **Performance Highlights**: 논문에서 제안하는 시스템은 기존의 수작업 설계된 샘플러 대신 학습된 affordance 모델을 사용하여 연속 변수를 샘플링하는 데 효과적입니다. 실험 결과, 이 접근 방식이 시스템의 전반적인 성능을 향상시키고, 에이전트가 물체를 이동시키는 작업과 같은 실제 작업을 더욱 효율적으로 수행할 수 있음을 보여주었습니다. 이를 통해 우리는 더 복잡한 다단계 작업을 성공적으로 수행할 수 있는 기반을 마련하였습니다.



### Efficient Implementation of the Global Cardinality Constraint with Costs (https://arxiv.org/abs/2502.02688)
Comments:
          Published at the 30th International Conference on Principles and Practice of Constraint Programming (CP 2024)

- **What's New**: 이 논문에서는 cardinality constraint를 비용과 함께 다루며, 이는 all different constraint의 일반화 형태입니다. 각각의 변수가 특정 값으로 얼마나 자주 할당되어야 하는지를 명시하면서, 총 할당 비용이 제한됩니다. 또한, 새로운 접근 방식이 개발되어 shortest paths에 대한 upper bounds를 활용하여 실제적인 상황에서도 효율적으로 작업할 수 있습니다.

- **Technical Details**: 비용이 포함된 cardinality constraint는 arc consistency filtering algorithm을 통해 관리됩니다. 그러나 기존의 알고리즘은 많은 shortest paths를 체계적으로 검색해야 하여 실용성이 떨어집니다. 이 논문은 landmarks에 기반한 upper bounds를 활용한 새로운 접근 방식을 제안, 이는 preprocessing으로 볼 수 있으며, 신속하게 작업을 수행할 수 있습니다.

- **Performance Highlights**: 제안된 접근 방식은 기존의 알고리즘보다 훨씬 더 빠르며, 많은 경우에 shortest paths 계산을 생략할 수 있습니다. 이를 통해 cardinality constraint의 적용이 더 실용적이고 효율적으로 이루어질 수 있습니다.



### Fully Autonomous AI Agents Should Not be Developed (https://arxiv.org/abs/2502.02649)
- **What's New**: 이번 논문은 완전 자율 AI 에이전트를 개발하는 것에 대한 반대 입장을 제시합니다. 연구진은 다양한 AI 에이전트 수준에 대해 설명하고, 각 수준에서 존재하는 윤리적 가치와 이로 인한 잠재적 위험을 분석합니다. AI 시스템의 자율성이 높아질수록 사람에게 미치는 위험이 증가한다는 결론을 도출하였습니다.

- **Technical Details**: AI 에이전트는 다양한 기능을 수행하기 위해 LLM(대형 언어 모델)을 통합한 다기능 시스템으로 구성되어 있습니다. 이러한 시스템은 단순히 사전에 정의된 작업을 수행하는 것이 아니라, 환경의 변화에 따라 독립적으로 상황에 맞는 계획을 세우고 실행할 수 있는 능력을 지니고 있습니다. 이러한 발전은 기술적으로는 비결정론적 환경에서의 계획 수립 능력을 내포합니다.

- **Performance Highlights**: 논문은 특수한 자율성 수준에 따른 위험 요인과 안전성의 가치를 강조하며, 완전 자율 시스템의 위험을 경고합니다. 완전 자율 AI 에이전트는 코드 작성 및 실행이 가능하지만, 이는 인간의 통제를 초과하게 되어 심각한 위험을 초래할 수 있습니다. 반면, 일부 인간 통제를 유지하는 반자율 시스템은 보다 유리한 위험-이익 프로파일을 가질 수 있음을 제시합니다.



### Secure & Personalized Music-to-Video Generation via CHARCHA (https://arxiv.org/abs/2502.02610)
Comments:
          NeurIPS 2024 Creative AI Track

- **What's New**: 이 논문에서는 사용자가 음악 비디오 생성 과정의 공동 창작자가 될 수 있도록 돕는 완전 자동화된 파이프라인을 발표합니다. 이는 음악의 가사, 리듬 및 감정을 기반으로 개인화되고 일관되며 맥락에 맞는 비주얼을 생성하는 데 중점을 두고 있습니다. 또한 CHARCHA라는 얼굴 확인 프로토콜을 도입하여 개인의 얼굴 정보의 무단 사용을 방지하고 사용자가 제공한 이미지를 수집하여 비디오 개인화를 지원합니다.

- **Technical Details**: 우리는 다중 모달리티(모델, 오디오, 시각 언어)를 활용하여 개인화된 음악 비디오를 생성하는 파이프라인을 구축했습니다. 이 과정에서 전이 학습된 여러 모델과 API를 통합하여 오디오 필기, 텍스트-이미지 변환, 음악 감정 인식 등의 작업을 수행합니다. 또한, LoRA(저랭크 적응)를 사용하여 사용자의 이미지를 삽입하고 개인화된 경험을 제공합니다.

- **Performance Highlights**: 기존 방법들과는 달리, 본 연구는 음악의 오디오 파일만으로도 개인화된 비디오를 생성해 내며, 사용자들은 자신의 취향을 반영한 비디오 제작에 참여하게 됩니다. 우리의 자동화된 파이프라인은 다양한 음악 장르와 언어에 적응 가능하며, 지역별 시각 스타일을 포함하는 등 높은 유연성을 제공합니다. 전반적으로, 이 시스템은 음악 비디오 생성의 새로운 트렌드를 선도하며, 개인화와 보안을 두루 고려한 혁신적인 접근 방식을 제시합니다.



### Seeing World Dynamics in a Nutsh (https://arxiv.org/abs/2502.03465)
- **What's New**: 본 논문에서는 casually captured monocular videos를 효과적으로 공간적 및 시간적으로 일관성을 유지하면서 표현하기 위한 문제를 다룹니다. 기존의 2D 및 2.5D 기술들은 복잡한 모션, 가리기 및 기하학적 일관성을 처리하는 데 어려움을 겪고 있으며, 본 연구는 동적 3D 세계의 투영으로서 단안 비디오를 이해하고자 합니다. NutWorld라는 새로운 프레임워크를 제안하여, 단일 전방 패스를 통해 단안 비디오를 동적 3D Gaussian 표현으로 변환합니다.

- **Technical Details**: NutWorld의 핵심은 정렬된 공간-시간 Gaussians(STAG) 표현을 통해 포즈 최적화 없이 장면 모델링을 가능하게 하는 것입니다. 이 프레임워크는 효과적인 깊이와 흐름 정규화를 포함하여, 단안 비디오의 고유한 3D 형태를 모델링합니다. 세 개의 주요 구성 요소로 이루어져 있으며, 각 구성 요소는 비디오 프레임 간의 공간-시간 대응 및 동적 특성을 학습하고 변환하는 데 중점을 두고 있습니다.

- **Performance Highlights**: NutWorld는 RealEstate10K 및 MiraData 실험을 통해 비디오 재구성의 효율성을 입증하고, 다양한 비디오 다운스트림 애플리케이션에서도 실시간 성능을 유지할 수 있는 뛰어난 유연성을 보여줍니다. 이 프레임워크는 새로운 뷰 합성, 일관된 깊이 추정, 비디오 세분화 및 비디오 편집과 같은 다양한 작업을 지원하며, 비디오 표현의 범용성 가능성을 제시합니다.



### Adapt-Pruner: Adaptive Structural Pruning for Efficient Small Language Model Training (https://arxiv.org/abs/2502.03460)
- **What's New**: 본 논문은 Adaptive Pruner라는 새로운 구조적 가지치기(Structured Pruning) 방법을 제안하여, 기존 방법보다 월등한 성능을 보입니다. 이를 통해 입력 신호에 대한 가중치 탄력성을 활용하여 각 레이어에 적합한 희소성(Sparsity)을 적용합니다. 또한, Adaptive Accel이라는 새로운 가속화 패러다임을 도입하여 가지치기와 훈련을 병행하는 방식을 보여줍니다. 최종적으로, Adapt-LLMs라는 새로운 모델 패밀리를 통해 강력한 처리 성능을 달성하였습니다.

- **Technical Details**: Adaptive Pruner는 각 레이어의 상대적인 중요도를 평가하여 가지치기를 진행하며, 적은 비율의 뉴런(약 5%)만을 제거함으로써 성능 감소를 최소화합니다. 특히, 제거 후 후속 훈련을 통해 성능 복구가 가능하며, 이는 기존의 고정된 가지치기 방식과 대조됩니다. 본 논문에서는 LLaMA-3.1-8B 모델을 활용한 실험을 통해 기존의 LLM-Pruner, FLAP, 그리고 SliceGPT보다 평균 1%-7% 더 나은 정확도를 기록하였습니다. 이 외에도 Adaptive Pruner는 MobileLLM-125M의 성능을 경쟁사 모델에 맞춰 복구했습니다.

- **Performance Highlights**: Adaptive Pruner의 성능은 MMLU 벤치마크에서 MobileLLM의 성능을 200배의 적은 토큰 수로 복구하는 데 기여했습니다. 또한, Discovering 1B 모델은 LLaMA-3.2-1B를 여러 벤치마크에서 초월하는 성과를 보였습니다. 결과적으로, Adapt-LLMs 모델은 강력한 공개 모델들에 비해 우위에 서게 되며, 이는 다양한 벤치마크에서 뛰어난 성능을 발휘하고 있음을 보여줍니다. 이러한 실험들은 작은 언어 모델을 활용한 실제 응용능력을 확대하는 데 중요한 기반이 될 것입니다.



### A Schema-Guided Reason-while-Retrieve framework for Reasoning on Scene Graphs with Large-Language-Models (LLMs) (https://arxiv.org/abs/2502.03450)
- **What's New**: 이 연구에서는 SG-RwR(Schema-Guided Retrieve-while-Reason)라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 구조화된 씬 그래프(scene graphs)를 사용하여 공간적 추론(spatial reasoning) 및 계획(task planning)을 지원합니다. 기존 방법들과는 달리, 두 agente(Reasoner, Retriever)는 전체 그래프 데이터 대신 씬 그래프 스키마를 입력으로 받아 작업을 수행하여 자가 회귀(hallucination)를 감소시키고 있습니다.

- **Technical Details**: SG-RwR는 두 개의 협력하는 코드 작성 LLM 에이전트, 즉, (1) 작업 계획 및 정보 쿼리 생성을 담당하는 Reasoner와 (2) 쿼리에 따라 해당 그래프 정보를 추출하는 Retriever로 구성됩니다. 두 에이전트는 반복적으로 협력하여 sequential reasoning 및 adaptive attention을 통해 그래프 정보를 처리합니다. 이렇게 함으로써 Reasoner는 추론 흔적을 생성하며, Retriever는 스키마 이해에 기반하여 씬 그래프 데이터를 쿼리합니다.

- **Performance Highlights**: 실험 결과, SG-RwR 프레임워크는 기존 LLM 기반 접근 방식을 초월하는 성능을 보였으며, 특히 수치적 질문 답변(numerical Q&A) 및 계획(task planning) 작업에서 주목할 만한 성과를 나타냈습니다. 또한, 에이전트 수준의 시연 없이도 작업 수준의 few-shot 예제를 통해 성능을 개선할 수 있는 가능성을 보여주었습니다. 프로젝트 코드 또한 공개될 예정입니다.



### Masked Autoencoders Are Effective Tokenizers for Diffusion Models (https://arxiv.org/abs/2502.03444)
- **What's New**: 이 연구에서는 MAETok이라는 새로운 autoencoder를 제안하여, 기존의 diffusion 모델에서 latent space의 구조를 개선하는 방법에 대해 탐구합니다. MAETok은 mask modeling을 통해 의미론적으로 풍부한 latent space를 학습하면서도 높은 재구성 충실도를 유지합니다. 핵심 발견은, diffusion 모델의 성능이 variational constraints보다 latent space의 구조와 더 밀접하게 연관되어 있다는 것입니다. 실제 실험을 통해 MAETok은 128개의 토큰만으로도 최신 기술 성능을 달성함을 입증했습니다.

- **Technical Details**: MAETok은 Masked Autoencoder (MAE)라는 자기 지도 학습 패러다임을 적용하여, 성능 저하 없이 더 일반화되고 구분된 표현을 발견할 수 있도록 설계되었습니다. 이 구조는 encoder에서 임의로 마스킹된 이미지를 재구성하며, pixel decoder와 auxillary shallow decoders를 사용하여 숨겨진 토큰의 특징을 예측합니다. 그 결과, 높은 재구성 충실도와 함께 강력한 의미론적 표현을 학습할 수 있었습니다. 실험을 통해 MAETok은 단순한 autoencoders에 비해 훈련 속도와 생성 속도가 극적으로 개선된 것을 확인했습니다.

- **Performance Highlights**: MAETok는 ImageNet 벤치마크에서 256×256 및 512×512의 해상도에서 각각 128개의 토큰만으로도 개선된 generation FID(gFID) 성능을 보여 주었습니다. 특히, 675M 파라미터의 diffusion 모델은 512 해상도에서 gFID 1.69와 304.2 IS를 달성하며, 기존 모델을 초월하는 성능을 기록했습니다. MAETok의 결과는 diffusion 모델의 훈련 및 생성 성능을 크게 향상시켜, 효율적인 해상도 생성에 기여하고 있음을 보여줍니다.



### On Fairness of Unified Multimodal Large Language Model for Image Generation (https://arxiv.org/abs/2502.03429)
- **What's New**: 이번 연구에서 제시된 통합 다중 모드 대형 언어 모델(U-MLLM)은 비주얼 이해와 생성에서 놀라운 성능을 발휘하고 있습니다. 그러나 이 모델들은 생성 모델만을 위한 구조와는 달리, 성별과 인종 차별 같은 편향(bias)을 포함할 가능성이 높아 새로운 문제를 제기합니다. 우리가 제안하는 locate-then-fix 전략은 이러한 편향의 출처를 감사(audit)하고 이해하는 데 중점을 둡니다.

- **Technical Details**: U-MLLM은 이미지와 텍스트를 동시에 처리할 수 있는 구조로 되어 있으며, 이를 위해 이미지 토크나이저(image tokenizer)와 언어 모델(language model)로 구성됩니다. 본 연구에서는 VILA-U와 같은 최신 U-MLLM의 다양한 구성 요소를 분석하여 그들이 보여주는 성별 및 인종 편향의 원인을 찾아냈습니다. 구체적으로, 우리는 언어 모델에서의 편향이 주된 원인임을 발견하였습니다.

- **Performance Highlights**: 실험을 통해 우리의 접근 방식이 성별 및 인종 편향을 71.91% 감소시키는 동시에 이미지 생성의 품질을 유지하는 데 성공했음을 보여주었습니다. 예를 들어, VILA-U 모델은 초기 점수(inception score)를 12.2% 증가시켰습니다. 이러한 개선 사항은 향후 통합 MLLM의 개발에서 더욱 공정한 해석과 디바이싱 전략이 필요함을 강조합니다.



### TruePose: Human-Parsing-guided Attention Diffusion for Full-ID Preserving Pose Transfer (https://arxiv.org/abs/2502.03426)
- **What's New**: 본 논문에서는 Pose-Guided Person Image Synthesis (PGPIS) 분야의 한계를 해결하기 위해 새로운 접근법인 human-parsing-guided attention diffusion을 제안합니다. 기존의 diffusion-based PGPIS 방법들이 얼굴 특징을 효과적으로 보존하는 반면, 의류 세부사항을 제대로 유지하지 못하는 문제를 발견했습니다. 이를 해결하기 위해, 사전 훈련된 Siamese 네트워크 구조와 인체 파싱 정보를 활용하여 얼굴과 의류의 시각적 특징을 동시에 잘 보존할 수 있는 방법론을 소개하고 있습니다.

- **Technical Details**: 제안된 방법론은 주로 세 가지 주요 구성 요소로 이루어져 있습니다: dual identical UNets (TargetNet과 SourceNet), 인체 파싱 기반의 융합 주의(attention) 모듈인 HPFA(human-parsing-guided fusion attention), 그리고 CLIP-guided attention alignment (CAA)입니다. 이러한 구성 요소들은 이미지 생성 과정에서 사람의 얼굴과 옷의 패턴을 효과적으로 통합하여 고품질 결과를 생성하는 데 기여합니다. 특히, HPFA와 CAA 모듈은 특정 의복 패턴을 타겟 이미지 생성에 적응적으로 사용할 수 있도록 합니다.

- **Performance Highlights**: 다양한 실험을 통해 제안된 방법이 의류 리트리벌 벤치마크와 최신 인간 편집 데이터셋에서 13개의 기존 기법 대비 상당한 우위를 보여주었습니다. 결과적으로, 제안한 네트워크는 얼굴 외관과 의류 패턴을 잘 보존하며 고품질 이미지를 생성할 수 있음을 입증했습니다. 이는 패션 산업 내 저작권 보호를 위한 의류 스타일 유지 측면에서 매우 중요합니다.



### Lightweight Authenticated Task Offloading in 6G-Cloud Vehicular Twin Networks (https://arxiv.org/abs/2502.03403)
Comments:
          6 pages, 3 figures, IEEE Wireless Communications and Networking Conference (WCNC2025), Milan, Italy, 24-27 March 2025

- **What's New**: 이 논문은 6G 차량 네트워크에서의 작업 오프로드 관리에 있어 인증된 작업 오프로드의 효율성을 높이기 위한 통합 프레임워크를 제안합니다. 경량의 Identity-Based Cryptographic (IBC) 인증을 클라우드 기반의 6G Vehicular Twin Networks (VTNs) 내 작업 오프로드에 통합함으로써, 오프로드 성능과 자원 할당을 최적화하는 방법을 제시합니다. Proximal Policy Optimization (PPO) 알고리즘을 활용한 Deep Reinforcement Learning (DRL)이 주요 기술로 도입되어, 인증된 오프로드 결정을 최소한의 지연으로 최적화합니다.

- **Technical Details**: 이 연구에서는 차량의 복제품인 VTNs와 클라우드 또는 엣지 서버에 호스팅되는 Digital Twins (DT)를 활용하여 차량과 도로 사이드 유닛의 상태를 효과적으로 관리합니다. 제안된 시스템은 차량과 클라우드 서버 간의 데이터 통신을 통해 작업 생성, 서명, 오프로드 및 최종 수신 및 검증의 모든 과정을 최적화합니다. IBC 인증이 오프로드 효율성에 미치는 영향을 평가하기 위해 다양한 네트워크 크기, 작업 크기 및 데이터 전송 속도를 고려한 성능 평가가 수행됩니다.

- **Performance Highlights**: 성능 평가 결과, IBC 인증은 오프로드 효율성을 최대 50%까지 감소시킬 수 있으며, 네트워크 크기와 작업 크기의 증가로 인해 효율성이 최대 91.7%까지 추가로 감소할 수 있습니다. 하지만 데이터 전송 속도를 높이면 인증 오버헤드가 있는 경우에도 오프로드 성능을 최대 63% 개선할 수 있는 가능성이 보여졌습니다. 이 연구의 시뮬레이션 코드는 GitHub에서 제공되어 추가적인 참조와 재현성을 돕습니다.



### SPRI: Aligning Large Language Models with Context-Situated Principles (https://arxiv.org/abs/2502.03397)
- **What's New**: 본 논문에서는 'Situated-PRInciples' (SPRI)라는 새로운 프레임워크를 제안하여, 최소한의 인간의 개입으로 실시간으로 각 입력 쿼리에 맞는 가이드 원칙을 자동 생성하는 방법을 다룬다. 이는 기존의 정적인 원칙이 아닌, 특정 상황에 맞춘 맞춤형 원칙을 통해 LLM의 결과를 정렬할 수 있도록 돕는다.

- **Technical Details**: SPRI는 두 단계로 구성된 알고리즘을 통해 작동한다. 첫 번째 단계에서는 기본 모델이 원칙을 생성하고, 비평 모델이 이를 반복적으로 개선하는 방식이다. 두 번째 단계에서는 생성된 원칙을 사용하여 기본 모델의 응답을 특정 사용자 입력에 맞게 조정하며, 비평 모델이 이러한 최종 응답을 평가하고 피드백을 제공한다.

- **Performance Highlights**: SPRI는 세 가지 상황에서 평가되었으며, 그 결과 복잡한 도메인 특정 작업에서도 전문가가 만든 원칙과 동등한 성능을 달성했다. 또한, SPRI가 생성한 원칙을 이용하면 LLM-judge 프레임워크에 비해 현저히 향상된 평가 기준을 제공하며, 합성 SFT 데이터 생성을 통해 진실성에서 유의미한 개선을 이끌어낼 수 있었다.



### Accurate AI-Driven Emergency Vehicle Location Tracking in Healthcare ITS Digital Twin (https://arxiv.org/abs/2502.03396)
Comments:
          8 pages, 8 figures, 5th IEEE Middle East & North Africa COMMunications Conference (MENACOMM'25), Lebanon Feb 20-23, 2025

- **What's New**: 이번 연구는 Intelligent Healthcare Transportation Systems (HITS)의 Digital Twin (DT)을 개선하기 위한 새로운 접근법을 제안합니다. 특히, 인공지능(AI) 예측 모델인 Support Vector Regression (SVR)과 Deep Neural Networks (DNN)을 사용하여 구급차의 다음 위치를 예측함으로써 물리적 시스템과 가상 시스템 간의 동기화 지연을 보완하는 방법을 모색하고 있습니다. 이러한 AI 모델들은 시뮬레이션 환경에서 높은 예측 정확도를 보여 주며, 실제 응급 상황에서의 데이터 통합과 관리에 기여할 것으로 기대됩니다.

- **Technical Details**: 이 연구에서는 DT와 HITS 간의 동기화 문제를 해결하기 위해 SVR과 DNN을 포함한 AI 예측 모델을 활용하고 있습니다. 이 모델들은 과거의 지리적 GPS 데이터 세트를 기반으로 교육을 받았으며, 통신 지연이 발생하는 가운데 물리적인 위치와 가상 위치 간의 차이를 감소시키는 데 중요한 역할을 합니다. 데이터 전송 지연 문제를 해결하기 위해 Docker와 Apache Kafka를 통해 HITS의 데이터 파이프라인을 구축하여 실시간 데이터 시각화를 지원하고 있습니다.

- **Performance Highlights**: SVR와 DNN 모델의 통합을 통해 HITS에서 관찰된 지연을 약 88%에서 93%까지 줄일 수 있음을 보여줍니다. 이러한 개선은 응급 상황에서의 실시간 동기화 정확도를 크게 향상시키며, HITS 운영의 효율성을 높이는 데 기여합니다. 연구 결과는 사고 현장에 신속하게 도착해야 하는 구급차의 운용성을 개선하고, 의료 서비스 제공에 있어서 중요한 실시간 결정을 가능하게 할 것입니다.



### Benchmarking Time Series Forecasting Models: From Statistical Techniques to Foundation Models in Real-World Applications (https://arxiv.org/abs/2502.03395)
- **What's New**: 본 연구는 독일의 수천 개 레스토랑에서 수집한 실제 데이터를 활용하여 시간대별 매출 예측(time series forecasting)의 새롭고 다양한 접근 방식을 탐구합니다. 통계적(statistical), 머신러닝(machine learning, ML), 딥러닝(deep learning) 및 파운데이션 모델(foundational models)의 성능을 평가하였습니다. 특히, Chronos와 TimesFM과 같은 파운데이션 모델의 잠재력을 강조하며, 최소한의 특성 엔지니어링(feature engineering)으로 경쟁력 있는 성능을 보여줍니다.

- **Technical Details**: 연구에서 사용된 예측 솔루션은 기상 조건(weather conditions), 달력 이벤트(calendar events), 시간대 패턴(time-of-day patterns)과 같은 다양한 기능을 포함합니다. 연구는 ML 기반 메타 모델(meta-models)의 강력한 성능을 보여주며, PySpark-Pandas 하이브리드 접근법이 대규모 배포에서 수평적 확장성(horizontal scalability)을 달성하는 robust solution임을 입증합니다. 이로써 수천 개의 레스토랑에서의 대규모 분산 시스템에서도 효과적인 예측이 가능함을 보여줍니다.

- **Performance Highlights**: ML 기반 메타 모델들은 높은 예측 성능을 보여주었으며, Chronos 및 TimesFM과 같은 최신 파운데이션 모델이 특히 주목받고 있습니다. 이러한 모델들은 사전 학습된 모델(pre-trained model)을 활용하여 zero-shot inference를 통해 최소한의 데이터 처리만으로도 뛰어난 예측 성능을 발휘합니다. 연구 결과는 대량의 데이터를 다루는 호텔 산업에서의 시간대별 매출 예측을 위한 잠재력을 보여줍니다.



### LIMO: Less is More for Reasoning (https://arxiv.org/abs/2502.03387)
Comments:
          17 pages

- **What's New**: LIMO 모델이 제안되어 복잡한 추론 능력을 매우 적은 훈련 예시로 이끌어낼 수 있음을 보여줍니다. 기존에는 복잡한 추론 작업에 많은 양의 훈련 데이터가 필요하다고 여겨졌으나, LIMO는 단 817개의 예제로 AIME에서 57.1%, MATH에서 94.8%의 정확도를 달성했습니다. 이 결과는 신뢰할 수 있는 데이터 효율성을 보여주며, AGI(Artificial General Intelligence)에 대한 가능성을 제시합니다.

- **Technical Details**: LIMO 모델은 사전 훈련(pre-training) 시 학습된 방대한 양의 수학적 지식을 기반으로 하며, 고도로 구조화된 인지적 과정(cognitive processes)을 통해 최소한의 데모를 사용하여 복잡한 작업을 해결할 수 있습니다. 이 모델은 100배 더 많은 데이터를 사용한 전통적인 SFT(supervised fine-tuning) 모델들을 40.5% 절대 개선하여 성능을 초과 달성했습니다. 또한, 모델의 지식 기반을 효과적으로 활용할 수 있는 인지적 템플릿(cognitive templates)의 역할도 강조됩니다.

- **Performance Highlights**: LIMO는 단 1%의 훈련 데이터를 사용하여 AIME 벤치마크에서 57.1%, MATH에서 94.8%의 정확도를 기록하며 새로운 기준을 세웁니다. 이 발견은 인공지능 연구에 깊은 의미를 가지며, 최소한의 훈련 샘플로도 복잡한 추론 능력을 이끌어낼 수 있음을 시사합니다. 더욱이 LIMO는 10개의 다양한 벤치마크에서 40.5%의 성능 개선을 보여줌으로써 모델의 일반화 능력을 입증하고, 기존의 데이터 집약적 접근 방식의 필요성을 재고하게 만듭니다.



### Transformers and Their Roles as Time Series Foundation Models (https://arxiv.org/abs/2502.03383)
Comments:
          34 Pages, 2 Figures

- **What's New**: 이 논문은 transformers가 시계열 기초 모델로서 가지는 근사화(approximation) 및 일반화(generalization) 능력에 대한 포괄적인 분석을 제공합니다. 우리는 특정 시간 시계열에 대해 자기회귀 모델(autoregressive model)을 적합시키는 transformers의 존재를 보여주었습니다. 또한, MOIRAI라는 다변량 시계열 모델이 무제한의 공변량(covariates)을 처리할 수 있는 능력을 가지고 있음을 분석하고 증명했습니다.

- **Technical Details**: MOIRAI 모델의 설계는 특정 공변량 수에 관계없이 자동으로 자기회귀 모델을 조정할 수 있는 능력을 토대로 하고 있습니다. 이 논문에서는 Dobrushin 조건을 만족하는 경우, 시계열 모델의 사전 학습(pretraining)에 대한 일반화 경계를 설정했습니다. 여기서, 데이터가 i.i.d. 가정을 만족하지 않아도 테스트 오류가 효과적으로 제한될 수 있음을 보였습니다.

- **Performance Highlights**: 실험 결과는 이론적 발견을 뒷받침하며, transformers가 시계열 기초 모델로서의 효능을 강조합니다. 입력 시계열의 길이가 증가함에 따라 예측 오류가 감소하는 경향을 보였습니다. 이러한 결과는 현대 모델이 다양한 데이터셋에서도 탁월한 성능을 발휘하는 이유를 설명합니다.



### A Beam's Eye View to Fluence Maps 3D Network for Ultra Fast VMAT Radiotherapy Planning (https://arxiv.org/abs/2502.03360)
- **What's New**: 본 논문은 방사선 치료 기술인 Volumetric Modulated Arc Therapy (VMAT)에서 중요하게 여겨지는 fluence maps의 생성을 간소화하고, 이를 위한 새로운 딥러닝 접근 방식을 제안합니다. 제안된 3D 네트워크는 환자 데이터를 기반으로 fluence maps를 직접 예측하여, 기존의 복잡하고 반복적인 프로세스를 크게 단축시킬 수 있습니다. 또한, 새로운 데이터셋을 제작하여 기존 REQUITE 데이터셋보다 약 20배 확장한 것이 주요한 기여입니다.

- **Technical Details**: 이 연구에서는 beam's eye view (BEV)라는 새로운 3D 표현으로 입력 3D 선량 맵을 변환하여 fluence maps를 예측합니다. ConvNeXt 블록을 사용하여 모든 fluence maps를 동시에 예측하는 3D Convolutional Neural Network (CNN) 아키텍처를 채택하였습니다. 이 모델은 MLC의 동적 시퀀스를 잘 반영할 수 있도록 설계되었으며, 데이터셋의 크기를 늘리기 위해 Eclipse Scripting API (ESAPI)를 사용하여 데이터셋을 확장했습니다.

- **Performance Highlights**: 제안한 3D 네트워크 아키텍처를 사용하였을 때, PSNR에서 약 8 dB 향상이 있었고 이는 원래 REQUITE 데이터셋을 기반으로 학습된 U-Net 아키텍처와 비교한 결과입니다. 더욱이, 생성된 dose-volume histograms (DVH)는 입력된 목표 선량과 매우 유사한 결과를 보였으며, 3D fluence map 예측의 정확성을 크게 향상시켰음을 보여줍니다.



### GHOST: Gaussian Hypothesis Open-Set Techniqu (https://arxiv.org/abs/2502.03359)
Comments:
          Accepted at AAAI Conference on Artificial Intelligence 2025

- **What's New**: 본 논문에서는 Open-Set Recognition (OSR)에서 각 클래스에 대한 성과의 변동성을 강조하며, 이를 해결하기 위한 새로운 알고리즘인 Gaussian Hypothesis Open Set Technique (GHOST)를 소개합니다. GHOST는 클래스별 다변량 Gaussian 분포를 사용한 하이퍼파라미터 없는 알고리즘으로, 각 클래스의 DNN 내 특징을 따로 모델링하여 공정한 평가를 추진합니다. 또한, 이 알고리즘은 원래 모델의 예측과는 달리, 특정 클래스의 성과를 독립적으로 평가하는 데 초점을 맞춰 공정성을 향상시킵니다.

- **Technical Details**: GHOST는 DNN의 분포를 다차원 Gaussian 모델로 설명하며, 클래스별로 대각 공분산 행렬을 사용하여 각 클래스의 특성을 구별하고 관리합니다. 이러한 접근법은 네트워크가 미지의 샘플에 대해 과도한 자신감을 가지지 않도록 하여, 전체 성능을 향상시키면서도 클래스를 공정하게 평가하는 데 기여합니다. 기존 알고리즘과 달리 GHOST는 하이퍼파라미터를 필요로 하지 않기 때문에 사용자가 보다 쉽게 OSR 기법을 적용할 수 있게 합니다.

- **Performance Highlights**: GHOST는 여러 ImageNet-1K로 사전 학습된 DNN을 테스트하며, AUOSCR, AUROC, FPR95와 같은 표준 메트릭을 사용하여 통계적으로 유의미한 성과 개선을 보여줍니다. 실험 결과, GHOST는 대규모 OSR 문제에 대한 최신 기술의 기준을 높임으로써 성과를 명확하게 개선했습니다. 공정성을 평가한 최초의 분석을 통해, GHOST가 OSR에서 성과의 불균형을 해소하고 각 클래스의 문제점을 신속하게 파악할 수 있도록 돕는 것을 입증하였습니다.



### Robust Autonomy Emerges from Self-Play (https://arxiv.org/abs/2502.03349)
- **What's New**: 이번 연구에서는 self-play(셀프 플레이)가 자율주행 분야에서도 효과적인 전략임을 입증합니다. 기존의 게임, 로봇 조작 등 다양한 분야에서 제공된 데이터를 기반으로 하지 않고도, 자율주행 정책이 자연스럽고 견고하게 발전할 수 있음을 보여줍니다. 이를 통해 자율주행 시스템의 훈련 과정에서 인간 주행 데이터를 전혀 사용하지 않고도 높은 성능을 달성할 수 있음을 시사합니다.

- **Technical Details**: Gigaflow(기가플로우)는 대규모 시뮬레이션 및 훈련을 가능하게 하는 배치형 시뮬레이터입니다. 이 시스템은 단일 8-GPU 노드에서 시간당 4.4억 상태 전환을 시뮬레이션할 수 있으며, 이는 42년의 주관적 주행 경험에 해당합니다. 이 연구에서 제안된 정책은 동적인 환경을 갖춘 차량, 보행자 및 자전거 이용자 등 다양한 교통 참여자를 동시에 처리하며, self-play를 통해 여러 에이전트의 데이터를 통합하여 학습합니다.

- **Performance Highlights**: Gigaflow에서 훈련된 정책은 CARLA, nuPlan, Waymo Open Motion Dataset 등의 벤치마크에서 상태-of-the-art(최첨단) 성능을 달성했습니다. 이 정책은 다양한 환경에서 사실적인 주행을 수행하며, 1,700만 km(17.5년)의 지속 주행 중 사건 없이 평균적으로 발생했습니다. 특히, 훈련 중 인간 데이터를 전혀 사용하지 않았음에도 불구하고, 인간 주행 참조와 비교할 때 정량적 현실성 또한 높게 평가받았습니다.



### Adaptive Variational Inference in Probabilistic Graphical Models: Beyond Bethe, Tree-Reweighted, and Convex Free Energies (https://arxiv.org/abs/2502.03341)
Comments:
          This work has been submitted to the Conference on Uncertainty in Artificial Intelligence (UAI) 2025 for possible publication

- **What's New**: 이번 논문은 확률 그래픽 모델의 변분 추론에서 마진 분포(marginal distributions)와 분배 함수(partition function)의 근사화를 다룹니다. 기존의 Bethe 근사와 같은 인기 있는 방법들은 효율적이나, 복잡하고 상호작용이 강한 모델에서는 실패할 수 있습니다. 저자들은 두 가지 새로운 근사화 클래스를 제안하며, 이들로 인해 모델에 자동으로 적응할 수 있는 방식으로 구현합니다.

- **Technical Details**: 논문에서는 두 가지 근사화의 일반화된 형태인 \( \mathcal{F}_{\bm{c}} \)와 \( \mathcal{F}_{\bm{\zeta}} \)를 분석합니다. 첫 번째는 Bethe 엔트로피가 각 통계에 따라 가중치를 달리하여 정의되며, 두 번째는 모델의 상태 에너지만을 변형합니다. 이들 각각은 자가 안내 신념 전파(self-guided belief propagation)와 같은 방법을 포함하고 있으며, 이들의 근사화 특성을 분석하기 위해 매개변수를 체계적으로 변화시킵니다.

- **Performance Highlights**: 제안된 방법을 통해 ADAPT-\( \zeta \)는 밀접하게 연결된 모델에서 단일 마진(singleton marginals)을 추정하는 데 뛰어난 성능을 보였으며, ADAPT-\( c \)는 추정된 분배 함수를 몇 배 향상시켰습니다. 복잡한 문제에서 두 방식 모두 효과적임을 실험적으로 입증하며, 매개변수가 주어진 모델에 자동으로 적응함으로써 계산의 효율성을 높입니다.



### RadVLM: A Multitask Conversational Vision-Language Model for Radiology (https://arxiv.org/abs/2502.03333)
Comments:
          21 pages, 15 figures

- **What's New**: 이번 연구에서는 RadVLM이라는 다중 작업 대화형 기초 모델을 개발하여, 흉부 X-레이(CXR) 해석을 지원합니다. 이 모델은 100만 개 이상의 이미지-지침 쌍으로 구성된 대규모 지침 데이터셋을 활용하여 훈련되었습니다. 특히, 단일 턴 작업뿐만 아니라 다중 턴 대화 작업까지 지원하며, 의료 진단 과정의 효율성을 높일 수 있습니다.

- **Technical Details**: RadVLM은 시각-언어 아키텍처를 기반으로 하며, José Vaswani가 발표한 transformer 구조에 뿌리를 두고 있습니다. 이 모델은 CXR 해석을 위해 설계된 다양한 작업을 지원하며, 각 작업에 맞는 이미지를 기반으로 한 지침 쌍과 대화 쌍을 사용하여 훈련됩니다. 이를 통해 RadVLM은 단순한 사용자인터페이스 내에서 직관적이고 사용자 친화적인 대화 능력을 제공합니다.

- **Performance Highlights**: RadVLM은 대화형 작업과 시각적 접지 능력에서 최고의 성능을 발휘하며, 다른 기존의 시각-언어 모델과 비교해도 경쟁력을 가지고 있습니다. 특히, 제한된 레이블링된 데이터 환경에서 여러 작업의 공동 훈련의 이점을 강조하는 실험 결과가 나타났습니다. 이러한 결과는 RadVLM이 의료 직업에서 효과적이고 접근 가능한 진단 워크플로우를 지원하는 AI 비서로서의 잠재력을 보여줍니다.



### Controllable GUI Exploration (https://arxiv.org/abs/2502.03330)
- **What's New**: 이 논문에서는 디퓨전 기반 접근법을 제안하여 인터페이스 스케치를 저렴하게 생성하는 방법을 소개합니다. 이 새로운 모델은 A) 프롬프트, B) 와이어프레임, C) 비주얼 플로우의 세 가지 입력을 통해 생성 프로세스를 유연하게 제어할 수 있도록 합니다. 디자이너는 이 세 가지 입력을 다양한 조합으로 제공하여 다양한 저충실도 솔루션을 신속하게 탐색할 수 있습니다.

- **Technical Details**: 제안한 모델은 고유의 두 가지 어댑터를 사용하여 GUI 요소의 위치 및 유형과 같은 지역적 속성과 전체 비주얼 플로우 방향과 같은 글로벌 속성을 제어합니다. 본 모델은 텍스트 기반 프롬프트 접근 방식의 단점을 극복하고, 비주얼 큐를 통해 더 나은 GUI 특성을 전달할 수 있습니다. 또한, 새로운 데이터셋을 생성하여 혼합된 모바일 UI 및 웹 페이지를 포함하여 GUI 생성 AI 모델 교육에 활용할 수 있는 기초 자료를 제공합니다.

- **Performance Highlights**: 모델의 성능을 평가한 결과, 제안한 모델은 입력 사양과 더욱 정밀하게 일치하며, 신속하고 다양한 GUI 대안을 탐색할 수 있다는 것을 질적으로 입증하였습니다. 이 모델은 대규모 디자인 공간을 최소한의 입력 명세로 탐색할 수 있게 해주어 UI 디자인 작업을 보다 효율적으로 수행할 수 있도록 합니다.



### ECM: A Unified Electronic Circuit Model for Explaining the Emergence of In-Context Learning and Chain-of-Thought in Large Language Mod (https://arxiv.org/abs/2502.03325)
Comments:
          Manuscript

- **What's New**: 이 논문은 Electronic Circuit Model (ECM)을 통해 인맥 학습(In-Context Learning, ICL)과 사고 과정(chain-of-Thought, CoT)의 결합된 영향을 규명하고, 이러한 이해를 통해 LLM의 성능을 개선하고 최적화하는 새로운 관점을 제시합니다. ECM은 LLM의 동작을 전자 회로로 개념화하여 ICL은 자석 자기장으로, CoT는 일련의 저항기로 모델 성능을 설명합니다. 또한, 실험 결과 ECM이 다양한 프롬프트 전략에 대해 LLM 성능을 성공적으로 예측하고 설명하는 것을 보여주었습니다.

- **Technical Details**: ECM에 따르면 모델 성능은 모델의 내재적 능력을 나타내는 기초 전압, 추론의 어려움, ICL에서 추가된 전압 등 다양한 요소에 의해 영향을 받습니다. ICL은 자석 자기장의 감소를 통해 추가 전압을 발생시키고, CoT는 여러 저항기를 사용해 각 추론 과정의 어려움을 더합니다. 이러한 관점은 ICL과 CoT의 복합적인 영향을 수치적으로 이해할 수 있는 기반을 제공합니다.

- **Performance Highlights**: ECM 활용을 통한 실험에서 LLM은 국제 정보 올림피아드(International Olympiad in Informatics, IOI) 및 국제 수학 올림피아드(International Mathematical Olympiad, IMO)와 같은 복잡한 작업에서 80% 이상의 인간 경쟁자를 초월하는 성과를 달성했습니다. 이로 인해 ECM이 LLM의 성능 향상에 기여했음을 보여주며, 이는 학술 연구 개발과 같은 탐색적 맥락에서도 최소 10% 이상의 성과 향상을 이루어냈습니다.



### Out-of-Distribution Detection using Synthetic Data Generation (https://arxiv.org/abs/2502.03323)
- **What's New**: 이번 연구에서는 OOD(Out-Of-Distribution) 탐지를 위한 혁신적인 접근 방식을 제안합니다. 특히, 대규모 언어 모델(LLM)의 생성 능력을 활용하여 외부 OOD 데이터 없이 고품질의 합성 OOD 프로시를 생성할 수 있습니다. 이를 통해 기존의 데이터 수집 문제를 해결하고, 다양한 ML(기계 학습) 작업에 적용 가능한 고유의 합성 샘플을 제공합니다.

- **Technical Details**: 이 연구에서는 LLM이 생성한 합성 데이터를 기반으로 OOD 탐지기를 훈련하는 새로운 방법을 개발했습니다. LLM의 정교한 프롬프트를 통해 가능한 분포 이동을 모방하는 합성 샘플을 생성하고, 이를 통해 신뢰성 있는 OOD 탐지를 구현합니다. 연구의 결과는 LLM이 텍스트 분류 작업에 크게 기여할 수 있음을 보여줍니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존 방법들에 비해 가짜 양성률(false positive rates)을 크게 감소시켰으며, 몇몇 경우에서 완벽한 제로를 달성했습니다. 또한, InD 데이터 및 합성 OOD 프로시를 사용해 높은 정확도를 유지하며, 다양한 모델 크기에서 좋은 성능을 보였습니다. 이 방법은 텍스트 분류 시스템의 신뢰성과 안전성을 크게 향상시킬 잠재력을 지니고 있습니다.



### Simplifying Formal Proof-Generating Models with ChatGPT and Basic Searching Techniques (https://arxiv.org/abs/2502.03321)
- **What's New**: 이 논문은 ChatGPT와 기본적인 검색 기법을 통합하여 공식 증명 생성의 간소화를 탐구합니다. 특히 miniF2F 데이터셋에 중점을 두고, ChatGPT와 검증 가능성이 있는 형식 언어인 Lean의 결합이 공식 증 명 생성을 더 효율적이고 접근 가능하게 만든다는 것을 보여줍니다. 저자들은 간단함에도 불구하고 가장 뛰어난 Lean 기반 모델이 알려진 벤치마크를 초과하여 31.15%의 통과율을 기록했음을 보고합니다.

- **Technical Details**: Lean은 의존 타입 이론을 기반으로 한 인터랙티브 정리 증명기로, 수학적 객체인 정의, 정리 및 보조 정리를 사용합니다. 주요 실험에서는 GPT-4 Turbo 모델을 사용하며, 취급하는 문제는 miniF2F 데이터셋의 다양한 수학적 주제에서 파생됩니다. 해당 데이터셋은 MATH 데이터셋과 올림피아드 수준의 문제로 구성되어 있으며, 이들 문제는 복잡한 솔루션을 요구하는 특성도 포함하고 있습니다.

- **Performance Highlights**: 모델은 miniF2F 데이터셋에서 예상보다 탁월한 성능을 보여주며, 이는 다양한 수학적 도메인에서 비슷한 효과를 검증하는 데 도움을 줍니다. 특히, Llemma와 같은 다른 언어 모델에서도 유사한 실험을 통해 성능의 다양성을 강조하고 있습니다. 이러한 결과는 AI 기반 공식 증명 생성의 유망한 방향성을 제시합니다.



### Harmony in Divergence: Towards Fast, Accurate, and Memory-efficient Zeroth-order LLM Fine-tuning (https://arxiv.org/abs/2502.03304)
- **What's New**: 이번 연구는 대형 언어 모델(LLM)의 미세 조정에서 메모리 효율성을 높이는 새로운 방법인 Divergence-driven Zeroth-Order (DiZO) 최적화를 제안합니다. DiZO는 레이어별로 적응형 업데이트를 통해 기존의 1차 최적화(FO) 방식처럼 다양한 크기의 업데이트를 구현하며, 이는 메모리 제한이 있는 환경에서도 효과적으로 적용될 수 있습니다. 이러한 접근법은 ZO 최적화의 성능 한계를 극복하고, 훈련 효율성을 크게 향상시킵니다.

- **Technical Details**: DiZO 최적화는 레이어 별로 적응형 업데이트를 수행하며, 이는 FO의 학습 용량과 유사한 효과를 제공합니다. 기존 ZO는 무작위 변동을 활용하여 경량화된 그래디언트 추정을 하는데 반해, DiZO는 각 레이어의 개별 최적화 요구 사항을 반영하여 정확하게 규모를 조정된 업데이트를 생성합니다. 이 과정에서 프로젝트를 활용하여 기울기를 요구하지 않으면서 효과적으로 레이어별 적응형 업데이트를 구현합니다.

- **Performance Highlights**: 실험 결과에 따르면, DiZO는 다양한 데이터셋에서 수렴을 위한 훈련 반복 횟수를 크게 줄이며, GPU 사용 시간을 최대 48%까지 절감하는 것으로 나타났습니다. 또한, RoBERTa-large, OPT 시리즈, Llama 시리즈와 같은 여러 LLM에서 대표적인 ZO 기준을 지속적으로 능가하였으며, 메모리가 많이 요구되는 FO 미세 조정 방법을 초월하는 경우도 있었습니다. 이는 DiZO가 메모리 효율성과 학습 성능을 동시에 개선할 수 있는 훌륭한 솔루션임을 증명합니다.



### MeDiSumQA: Patient-Oriented Question-Answer Generation from Discharge Letters (https://arxiv.org/abs/2502.03298)
- **What's New**: 이 논문에서는 환자 접근성을 높이기 위한 도구로서 MeDiSumQA라는 새로운 데이터셋을 소개합니다. MeDiSumQA는 MIMIC-IV의 퇴원 요약에서 자동화된 질문-답변 생성 프로세스를 통해 만들어졌습니다. 이 데이터셋은 환자 중심의 질문-답변(QA) 형식을 채택하여 건강 문서 이해도를 높이고 의료 결과를 개선하는 데 기여하고자 합니다.

- **Technical Details**: MeDiSumQA는 퇴원 요약을 기반으로 하여 생성된 고품질 질문-답변 쌍을 포함하고 있습니다. 이 데이터셋은 메디컬 LLM을 평가하기 위해 사용되며, 질문-답변 쌍의 생성 과정에서 Meta의 Llama-3 모델과 의사의 수작업 검토를 결합하였습니다. 질문-답변 페어는 증상, 진단, 치료 같은 6개의 QA 카테고리로 분류됩니다.

- **Performance Highlights**: 연구 결과, 일반적인 LLM이 생물 의학적으로 조정된 모델보다 우수한 성능을 보였습니다. 자동 평가 메트릭은 인간의 평가와 상관관계가 있음을 보여주며, MeDiSumQA를 통해 향후 LLM의 발전을 지원하고 임상 문서의 환자 친화적 이해를 증진시키겠다는 목표를 가지고 있습니다.



### ALPET: Active Few-shot Learning for Citation Worthiness Detection in Low-Resource Wikipedia Languages (https://arxiv.org/abs/2502.03292)
Comments:
          24 pages, 8 figures, 4 tables

- **What's New**: 이번 연구에서는 Citation Worthiness Detection (CWD)를 위한 ALPET 프레임워크를 도입했습니다. ALPET는 Active Learning (AL)과 Pattern-Exploiting Training (PET)을 결합하여 데이터 자원이 제한된 언어에서 CWD를 향상시키는 데 중점을 두고 있습니다. 특히, 카탈란어, 바스크어 및 알바니아어 위키백과 데이터셋에서 기존의 CCW 기준선보다 뛰어난 성능을 보이며, 필요한 레이블 데이터 양을 80\% 이상 줄일 수 있었습니다.

- **Technical Details**: ALPET는 300개의 레이블 샘플 이후 성능이 안정화되는 경향이 있어, 대규모 레이블 데이터셋이 흔하지 않은 저자원 환경에서의 적합성을 강조합니다. 특정한 Active Learning 쿼리 전략, 예를 들어 K-Means 클러스터링을 사용하는 방법은 장점을 제공할 수 있지만, 그 효율성은 항상 보장되지 않으며 작은 데이터셋에서는 무작위 샘플링에 비해 소폭의 개선에 그치는 경향이 있습니다. 이는 단순한 무작위 샘플링이 자원이 제한된 환경에서도 여전히 강력한 기준선 역할을 할 수 있음을 시사합니다.

- **Performance Highlights**: 결론적으로, ALPET은 레이블 샘플이 적더라도 높은 성능을 달성할 수 있는 능력 덕분에 저자원 언어 설정에서 온라인 콘텐츠의 검증 가능성을 높이는 데 유망한 도구가 될 것으로 평가됩니다. 연구 결과는 저자원 환경에서도 언어적 다양성을 지켜내며, 정보의 신뢰성을 향상시키기 위한 효율적인 방법론을 제시하고 있습니다.



### STEM: Spatial-Temporal Mapping Tool For Spiking Neural Networks (https://arxiv.org/abs/2502.03287)
Comments:
          24 pages, 23 figures, under review at IEEE TC

- **What's New**: 이 논문에서는 Spiking Neural Networks (SNNs)의 뉴런 상태가 에너지 효율성에 미치는 영향을 조사하고, 이를 최신 메모리 계층 구조에 맞춰 어떻게 매핑할지를 연구합니다. STEMS라는 도구를 활용하여 SNN의 특성을 모델링하고, 데이터 이동을 최소화하기 위한 최적화를 탐색합니다. 이 연구는 SNN의 에너지 효율성을 높이기 위한 통찰력을 제공합니다.

- **Technical Details**: SNN은 생물학에 영감을 받은 신경망으로, Sparse한 Spike 기반의 계산 방식을 사용하며 시간에 따라 변하는 상태를 가진 뉴런 모델을 활용합니다. 특히 논문에서는 STEMS라는 매핑 디자인 탐색 도구를 개발하여, 공간적 및 시간적 차원에서의 최적화를 통해 뉴런 상태로 인한 데이터 이동을 최소화합니다. LIF (Leaky Integrate-and-Fire) 뉴런 모델을 사용하며, SNN의 내부 상태와 외부 데이터 간의 상호작용을 고려합니다.

- **Performance Highlights**: STEMS를 사용하여 두 개의 이벤트 기반 비전 SNN 벤치마크에서 오프칩(Off-chip) 데이터 이동을 최대 12배, 에너지를 5배 줄이는 성과를 거두었습니다. 또한 뉴런 상태 최적화를 통해 20배 이상의 뉴런 상태 감소와 1.4배의 에너지 절약을 달성하면서도 정확도 손실 없이 성능을 개선하는 결과를 보여줍니다.



### Token Assorted: Mixing Latent and Text Tokens for Improved Language Model Reasoning (https://arxiv.org/abs/2502.03275)
- **What's New**: 이번 연구에서는 Large Language Models (LLMs)가 chain-of-thought (CoT) 데이터를 통해 추론 능력을 극대화할 수 있음을 보여주며, 초기 추론 단계를 부분적으로 추상화한 혼합 표현(hybrid representation)을 제안합니다. VQ-VAE를 통해 생성된 잠재 이산 토큰(latent discrete tokens)을 사용하여 추론 흔적(reasoning traces)의 길이를 크게 줄이는데 성공했습니다. 이 연구는 Keys-Finding Maze 문제의 모델을 처음부터 훈련시키고, 보이지 않는 잠재 토큰(latent tokens)을 포함한 확장된 어휘로 LLM을 미세 조정하는 두 가지 시나리오를 탐구합니다.

- **Technical Details**: 제안된 방법론은 추론 과정의 초기 단계에서 불필요한 정보를 잠재 토큰으로 대체하여, 모델이 텍스트 토큰과 혼합된 새로운 토큰에 대해 신속하게 적응할 수 있도록 합니다. 훈련 과정 동안 텍스트 토큰의 수를 무작위로 변경하여 잠재 토큰으로 교체하는 전략을 채택합니다. 이를 통해, 모델은 추상적인 사고 과정의 표현과 세부적인 텍스트 설명 모두에서 학습할 수 있는 기회를 가집니다.

- **Performance Highlights**: 모델의 성능은 Keys-Finding Maze와 ProntoQA, ProsQA와 같은 다양한 벤치마크에서 평가되었으며, 텍스트 전용 추론 흔적을 사용하여 훈련 된 기준 모델을 지속적으로 능가하는 결과를 보였습니다. 다양한 수학적 추론 벤치마크(GSM8K, Math, OlympiadBench-Math)에서도 제안된 방법이 효과적으로 성능을 향상시켰습니다. 전체적으로, 여러 작업과 모델 아키텍처에서 제안된 혼합 표현 접근법이 우수한 결과를 달성했습니다.



### Deep Learning Pipeline for Fully Automated Myocardial Infarct Segmentation from Clinical Cardiac MR Scans (https://arxiv.org/abs/2502.03272)
- **What's New**: 이 연구는 심근경색(MI) 분할을 위한 완전 자동화된 딥 러닝 기반 방법을 개발하고 평가하는 것을 목표로 하였습니다. 기존의 수동 LGE(지연 강화) 심장 자기 공명(MR) 이미지 분할 방식의 한계를 해결하고, 신속하고 정확한 심근경색의 양적 평가를 가능하게 하는 파이프라인을 제시합니다. 이는 임상 환경에서의 이미지 전처리에 필요한 단계를 포함하여 모든 과정을 자동으로 수행할 수 있습니다.

- **Technical Details**: 연구는 2D 및 3D CNN(합성곱 신경망)에 기반한 계단식 프레임워크를 사용하여 진행되었습니다. 144개의 LGE CMR 검사 데이터를 훈련셋으로 사용하여 AI 기반 분할 성능을 개발하였고, 152개의 검사로 구성된 테스트셋에서 AI 분할과 수동 분할을 정량적으로 비교하였습니다. CNN은 원본 데이터에서 전체 좌심실을 포함하는 작고 필요한 이미지 스택을 추출하고, 이후 다중 클래스 분할을 수행하는 방식으로 구성되었습니다.

- **Performance Highlights**: AI 기반 분할 결과는 수동으로 계산한 심근경색 볼륨과 높은 일치를 보였으며, 전문가들은 AI 측정이 실제 심근경색 범위를 더 잘 반영한다고 평가했습니다. AI 기반 분할의 정확도를 평가한 결과, 심혈관 전문가의 눈으로 볼 때 AI 출력물이 유의미하게 더 나은 평가를 받아 임상적인 가능성을 제시합니다. 반면, 미세혈관 폐쇄 MVO의 경우에는 여전히 수동 측정을 더 선호했습니다.



### When Pre-trained Visual Representations Fall Short: Limitations in Visuo-Motor Robot Learning (https://arxiv.org/abs/2502.03270)
- **What's New**: 이 연구는 사전 학습된 시각 표현(PVRs)을 비주얼-모터 로봇 학습에 통합하는 새로운 방법을 제안합니다. PVR의 고유한 한계인 시간 상호 엉킴과 손상에 대한 일반화 능력 부족을 해결하기 위해, 연구는 시간 인식과 작업 완료 감각을 결합하여 PVR 기능을 향상시킵니다. 또한, 작업 관련 지역적 특징에 선택적으로 주의를 기울이는 모듈을 도입하여 다채로운 장면에서도 로버스트한 성능을 보장합니다.

- **Technical Details**: 기존 PVR의 한계로 인해 비주얼-모터 정책 학습이 중단되는 문제를 강조하고, 이를 해결하기 위해 두 가지 주요 기법을 제안합니다. 첫 번째는 PVR 기능에 시간적 인식을 추가하여 시간적으로 엉킨 특징을 분리하는 것이며, 두 번째는 작업 관련 시각 단서를 기준으로 주목하는 모듈을 도입하여 эффективность을 높입니다. 이 과정에서 기존 PVR 생태계를 손상시키지 않고도 주요 성과를 도출했습니다.

- **Performance Highlights**: 실험 결과, 마스킹 목표로 학습된 PVR은 기존 기능을 보다 효과적으로 활용하여 성능이 유의미하게 향상되었습니다. 특히, 시각 정보가 과도하게 엉켜 있던 기존의 문제를 해결함으로써 기본적인 피컷 앤 플레이스 작업에서도 향후 성과 개선이 확인되었습니다. 연구는 PVR의 기본적인 한계를 극복하고 비주얼-모터 학습의 가능성을 상향 제시하는 중요한 기초자료를 제공하고 있습니다.



### Long-tailed Medical Diagnosis with Relation-aware Representation Learning and Iterative Classifier Calibration (https://arxiv.org/abs/2502.03238)
Comments:
          This work has been accepted in Computers in Biology and Medicine

- **What's New**: 본 논문에서는 Long-tailed Medical Diagnosis (LMD) 프레임워크를 제안하여 긴 꼬리(long-tailed) 의료 데이터셋에서 균형 잡힌 의료 이미지를 분류하는 접근 방식을 소개하고 있습니다. 이 프레임워크는 Relation-aware Representation Learning (RRL) 스킴을 통해 특징 표현을 향상시키고, Iterative Classifier Calibration (ICC) 스킴을 통해 분류기를 반복적으로 보정합니다. 특히, 저수 샘플에 대한 성능 향상을 목표로 하고 있습니다.

- **Technical Details**: LMD는 첫 번째 단계에서 RRL을 통해 서로 다른 데이터 증강(augmentation)을 통해 내재적 의미적 특징을 학습할 수 있도록 인코더의 표현 능력을 향상시킵니다. 두 번째 단계에서는 Expectation-Maximization 방식으로 분류기를 보정하기 위한 ICC 스킴을 제안하며, 가상 기능의 생성 및 인코더의 미세 조정(fine-tuning)을 포함합니다. 이러한 방법을 통해 다수 클래스에서의 진단 지식은 유지하면서 소수 클래스를 보정할 수 있습니다.

- **Performance Highlights**: 세 가지 공개 긴 꼬리 의료 데이터셋에 대한 포괄적인 실험 결과, LMD 프레임워크는 최신 기술(state-of-the-art)들을 훨씬 능가하는 성능을 보였습니다. 특히, LMD는 rare disease(희귀 질병)에 대한 분류 성능을 현저히 개선하였으며, Virtual Features Compensation (VFC)와 Feature Distribution Consistency (FDC) 손실을 통해 균형 잡힌 특징 분포를 유지하며 학습합니다.



### The Other Side of the Coin: Unveiling the Downsides of Model Aggregation in Federated Learning from a Layer-peeled Perspectiv (https://arxiv.org/abs/2502.03231)
- **What's New**: 이 논문은 Federated Learning (FL)에서 모델 집계(model aggregation) 과정 중 발생하는 성능 저하(performance drop)의 원인을 심도 있게 분석한 첫 번째 연구입니다. 일반적으로 FL에서는 여러 클라이언트(client)가 지식을 공유하지만, 집계된 모델이 각 클라이언트의 로컬 데이터(local data)에서 성능을 발휘하기까지 시간이 걸리는 현상이 발생합니다. 생기는 성능 저하 현상에 대한 원인을 조사하며, 이를 경감할 수 있는 몇 가지 전략을 제안합니다.

- **Technical Details**: 저자는 다양한 데이터셋(datasets)과 모델 아키텍처(model architectures)에 걸쳐 모델 집계의 레이어(layer)-피엘 분석(layer-peeled analysis)을 수행하였습니다. 분석 결과, 성능 저하의 주요 원인은 두 가지로 나타났으며, 첫째로 딥 뉴럴 네트워크(Deep Neural Networks, DNNs)에서 특성 변동 억제(feature variability suppression)가 저해되고, 둘째로 특징(feature)들과 후속 학습 과정 간의 결합(coupling)이 약해진다는 것입니다. 이러한 발견을 바탕으로 FL에서의 모델 집계의 부정적인 영향을 완화할 수 있는 간단하고도 효과적인 전략을 제안합니다.

- **Performance Highlights**: 제안된 전략들은 모델 집계의 긍정적인 혜택을 유지하면서도 성능 저하 문제를 완화하는 데 기여할 수 있습니다. 이 연구는 FL 알고리즘의 개발에 있어 더 효과적인 방안을 모색하기 위한 기초를 마련하는 중요한 역할을 할 것으로 예상됩니다. 성과들은 클라이언트 간의 지식 공유가 결국 FL 모델의 수렴(convergence)을 가속화하는 데 도움을 줄 것으로 기대됩니다.



### iVISPAR -- An Interactive Visual-Spatial Reasoning Benchmark for VLMs (https://arxiv.org/abs/2502.03214)
- **What's New**: 최근 발표된 iVISPAR는 비전-언어 모델(VLM)의 공간 추론 능력을 평가하기 위해 고안된 인터랙티브 다중 모드 벤치마크입니다. 이 벤치마크는 고전적인 슬라이딩 타일 퍼즐의 변형으로, 논리적 계획, 공간 인지, 다단계 문제 해결 능력을 요구합니다. iVISPAR는 2D, 3D 및 텍스트 기반 입력 방식 모두를 지원하여 VLM의 전반적인 계획 및 추론 능력을 포괄적으로 평가할 수 있습니다. 연구 결과, 일부 VLM은 간단한 공간 작업에서 양호한 성능을 보였으나, 복잡한 구성에서는 어려움을 겪는다는 점이 드러났습니다.

- **Technical Details**: iVISPAR의 주요 특징 중 하나는 Sliding Geom Puzzle(SGP)로, 기존의 슬라이딩 타일 퍼즐을 고유한 색상과 모양으로 정의된 기하학적 객체로 대체했습니다. 벤치마크는 사용자가 자연어 명령어를 발행하여 보드에 대한 작업을 수행할 수 있도록 설계된 텍스트 기반 API를 지원합니다. iVISPAR는 퍼즐의 복잡도를 세밀하게 조정할 수 있으며, 다양한 기준 모델과의 성능 비교가 가능합니다. 최적의 솔루션은 A* 알고리즘을 사용하여 계산되며, 벤치마크는 보드 크기, 타일 수 및 솔루션 경로 등 다양한 요인을 조정하여 복잡도를 확장할 수 있습니다.

- **Performance Highlights**: 실험 결과, 최신 VLM들이 기본적인 공간 추론 작업을 처리할 수 있지만, 3D 환경의 보다 복잡한 시나리오에 직면할 때 상당한 어려움을 보임을 확인했습니다. VLM은 일반적으로 2D 비전에서 더 나은 성능을 보였으나, 인간 수준의 성과에는 미치지 못하는 한계를 보여주었습니다. 이러한 결과는 VLM의 현재 능력과 인간 수준의 공간 추론 간의 지속적인 격차를 강조하며, VLM 연구의 추가 발전 필요성을 시사합니다.



### A Unified and General Humanoid Whole-Body Controller for Fine-Grained Locomotion (https://arxiv.org/abs/2502.03206)
Comments:
          The first two authors contribute equally. Project page: this https URL

- **What's New**: HUGWBC는 인간의 운동능력을 로봇에 통합하기 위해 설계된 새로운 통합 제어기입니다. 이는 로봇이 다양한 자연스러운 보행 방식, 즉 걷기, 뛰기, 점프 및 홉 동작을 가능하게 하며, 각 동작에 대해 세부적인 조정이 가능합니다. HUGWBC는 고급 훈련 기술과 함께 사용되어 로봇의 각 동작을 정밀하게 제어할 수 있도록 합니다. 특히, 이 연구는 HUGWBC가 현재 로봇 시스템의 한계를 극복하는 데 중요한 역할을 할 것임을 보여줍니다.

- **Technical Details**: HUGWBC는 향상된 훈련 기술을 사용하여, 로봇이 다양한 동작을 위해 총체적인 명령 공간을 생성하는 방법을 사용합니다. 이를 통해 로봇은 걷기, 뛰기, 점핑 및 홉과 같은 다채로운 보행 스타일을 수행할 수 있습니다. HUGWBC는 강화 학습을 통해 다양한 행동을 단일 정책으로 학습하며, 상체 개입에도 적합한 구조로 설계되었습니다. 따라서 이 시스템은 매우 강력한 loco-manipulation 기능을 구현할 수 있습니다.

- **Performance Highlights**: 실험 결과, HUGWBC는 네 가지 다른 보행 스타일로 여덟 가지 다른 명령에 대해 높은 추적 정확도를 유지합니다. 또한, 이 시스템은 상체 개입 여부와 관계없이 로봇의 운동 성능을 향상시키는 데 도움이 됩니다. 제공된 명령 간의 복잡한 관계를 분석하여, 로봇이 더욱 정교한 동작을 할 수 있도록 지원합니다. HUGWBC는 유연성과 강인성을 제공하며, 향후 휴머노이드 로봇의 능력을 더욱 확장할 수 있을 것으로 기대됩니다.



### Improve Decoding Factuality by Token-wise Cross Layer Entropy of Large Language Models (https://arxiv.org/abs/2502.03199)
Comments:
          NAACL 2025 Findings

- **What's New**: 이 논문에서는 Large Language Model(LLM)의 'hallucination' 문제를 해결하기 위한 새로운 접근 방식인 Cross-layer Entropy eNhanced Decoding(END)을 제안합니다. END는 추가적인 훈련 없이도 모델이 생성하는 문장의 사실성을 개선하는 데 초점을 맞추고 있으며, 각 후보 토큰의 사실적 지식량을 정량화하여 확률 분포를 조정하는 방법입니다. 이 연구는 각 토큰 레벨에서 내부 상태 변화와 출력 사실성 간의 상관관계를 분석함으로써 보다 깊이 있는 이해를 제공합니다.

- **Technical Details**: END는 모델의 여러 층에서 예측된 내부 확률의 변화를 이용하여 각 후보 토큰의 사실적 지식을 증폭시키는 기능을 수행합니다. 구체적으로, END는 각 토큰에 대해 사실성을 중요한 요소로 고려하여 최종 예측 분포를 조정합니다. 이 방법은 LLM의 여러 아키텍처에 적용될 수 있으며, 모델의 구조가 서로 다르더라도 일반화된 성능을 보여줍니다.

- **Performance Highlights**: 실험 결과, END를 통해 TruthfulQA 및 FACTOR와 같은 hallucination 벤치마크와 TriviaQA 및 Natural Questions와 같은 일반 QA 벤치마크에서 생성된 콘텐츠의 진실성과 정보성을 향상시켰습니다. 이 방법은 QA 정확도를 안정적으로 유지하면서도 생성되는 내용의 사실성을 상당히 개선하였습니다. 실험은 다양한 LLM 백본에 대한 추가 검증을 통해 END의 효용성과 일반성을 보여주었습니다.



### Euska\~nolDS: A Naturally Sourced Corpus for Basque-Spanish Code-Switching (https://arxiv.org/abs/2502.03188)
- **What's New**: 이 논문은 바스크(Basque)와 스페인어(Spanish) 간의 코드 스위칭(code-switching) 현상을 연구하기 위해 첫 번째 자연어 데이터셋인 EuskañolDS를 소개합니다. 이 데이터셋은 기존 코퍼스에서 언어 식별 모델(language identification models)을 이용해 코드 스위치된 문장을 자동으로 수집하고, 그 결과를 수동으로 검증하여 신뢰할 수 있는 샘플을 확보하는 방법론을 기반으로 하고 있습니다. 따라서 Barasko와 Spanish 언어 간의 코드 스위칭 연구를 위한 중요한 자료를 제공합니다.

- **Technical Details**: 논문에서는 언어 식별 모델을 통해 바스크어와 스페인어가 혼합된 코드 스위칭 문장을 수집하기 위한 반지도학습(semi-supervised learning) 접근 방식을 사용합니다. 최종 데이터셋은 20,008개의 자동 분류된 인스턴스와 927개의 수동 검증 인스턴스가 포함되어 있으며, 두 언어는 각기 다른 언어 계통을 가지며 많은 전형적인 차이를 갖고 있습니다. 이 데이터셋은 바스크어와 스페인어 간의 언어 접촉과 코드 스위칭 현상을 이해하는 데 중요한 자원이 될 것입니다.

- **Performance Highlights**: 이 데이터셋은 서로 다른 출처에서 수집한 코드 스위칭 인스턴스들을 포함하고 있으며, 그 특성을 정량적 분석과 질적 분석을 통해 설명합니다. 특히, 대부분의 인스턴스가 문장 간 코드 스위칭(inter-sentential CS) 형태를 보이며, 이는 바스크의 법정 의사록 및 소셜 미디어 콘텐츠에서 많이 나타납니다. 이러한 성과는 코드 스위칭이 포함된 NLP 모델 개발을 지원할 수 있는 유용한 자료로 작용할 것입니다.



### Scalable In-Context Learning on Tabular Data via Retrieval-Augmented Large Language Models (https://arxiv.org/abs/2502.03147)
Comments:
          Preprint

- **What's New**: 최근 대형 언어 모델(LLMs)을 사용한 연구는 사전 훈련 후 표 형식 데이터(tabular data)로 맞춤 설정함으로써 일반화된 표 형식 인컨텍스트 학습(TabICL) 능력을 획득할 수 있음을 보여주었습니다. 이러한 모델은 다양한 데이터 스키마(data schemas)와 작업 도메인(task domains) 간의 효과적인 전이(transfer)를 수행할 수 있게 되었습니다. 하지만 기존의 LLM 기반 TabICL 접근 방식은 시퀀스 길이 제약으로 인해 제한된 학습 사례(few-shot scenarios)에만 적용 가능한 한계가 있습니다.

- **Technical Details**: 이 논문에서는 표 형식 데이터에 적합한 검색 증강 LLM(retrieval-augmented LLMs)을 제안하며, 이는 맞춤형 검색 모듈(customized retrieval module)과 검색 기반 지시 조정(retrieval-guided instruction-tuning)을 결합한 새로운 접근 방식입니다. 이 방법은 더 큰 데이터셋을 효과적으로 활용할 수 있게 하여, 69개 널리 인식되는 데이터셋에서 성능을 크게 개선하는 동시에 확장 가능한 TabICL을 가능하게 합니다. 또한, 이 연구는 데이터를 효율적으로 활용하기 위한 비모수(non-parametric) 검색 모듈의 설계와 맞춤형 데이터셋 특정 검색 정책의 발전 가능성을 탐구합니다.

- **Performance Highlights**: 연구 결과는 LLM 기반 TabICL이 강력한 알고리즘을 발굴하고, 앙상블 다양성을 강화하며, 특정 데이터셋에서 뛰어난 성능을 보이는 데 기여할 수 있음을 보여주었습니다. 그러나 전반적인 성능에서는 잘 튜닝된 숫자 모델에 비해 여전히 뒤처진다는 사실이 밝혀졌고, 각 데이터셋의 분석 결과 LLM 기반 TabICL 접근 방식은 더 효과적인 검색 정책과 더 다양한 피처 분포를 활용함으로써 더욱 향상될 수 있음을 제시했습니다. 이러한 점에서, 본 접근 방식은 TabICL을 위한 특별한 패러다임으로서 엄청난 잠재력을 지니고 있다고 생각됩니다.



### Gotham Dataset 2025: A Reproducible Large-Scale IoT Network Dataset for Intrusion Detection and Security Research (https://arxiv.org/abs/2502.03134)
Comments:
          16 pages, 7 figures, 4 tables. Submitted at the Data in Brief journal

- **What's New**: 이번 논문에서는 IoT 네트워크 트래픽 데이터셋을 제시합니다. 이 데이터셋은 현실적이고 이질적인 네트워크 보안 연구 환경을 제공하기 위해 설계된 Gotham 테스트베드를 활용하여 생성되었습니다. 78개의 다양한 프로토콜을 운영하는 IoT 장치가 포함됩니다.

- **Technical Details**: 데이터는 tcpdump를 사용하여 Packet Capture (PCAP) 형식으로 캡처되었습니다. 이는 정상 트래픽과 악성 트래픽을 모두 기록하며, 악성 트래픽은 다양한 공격 유형(예: Denial of Service, Telnet Brute Force 등)을 커버하는 스크립트 공격을 통해 생성되었습니다. Python을 사용하여 Tshark 도구로 기능 추출을 진행하였고 그 결과는 CSV 형식으로 변환되었습니다.

- **Performance Highlights**: 이 데이터셋은 복잡하고 대규모 IoT 환경을 위해 맞춤형 침입 탐지 시스템 및 보안 메커니즘 개발에 유용한 자원으로, 다양한 트래픽 패턴과 공격 시나리오를 제공합니다. 데이터셋은 Zenodo에서 공개 이용 가능합니다.



### Metis: A Foundation Speech Generation Model with Masked Generative Pre-training (https://arxiv.org/abs/2502.03128)
- **What's New**: 이번 논문에서는 Metis라는 통합 음성 생성의 기초 모델을 소개합니다. Metis는 대규모 비유명 음성 데이터를 활용한 마스크 생성을 통해 사전 훈련(pre-training)을 진행한 후 세부 음성 생성 작업에 맞추어 미세 조정(fine-tuning)하여 다양한 음성 생성 작업에 효율적으로 적응합니다. 특히, 두 개의 불연속 음성 표현인 SSL 토큰과 음향 토큰을 사용하여 차별화된 접근 방식을 제공합니다.

- **Technical Details**: Metis는 SSL(Speech Self-supervised Learning) 토큰을 이용하여 30만 시간의 다양한 음성 데이터로 마스크 생성 모델링(masked generative modeling)을 수행합니다. 세부 작업에 대한 조건을 통합하여 고유 음성 생성 작업에 효율적으로 적응할 수 있도록 설계되었습니다. 이 과정에서 사용된 마스크 생성 모델은 비훈련 데이터와 제한된 매개변수로도 다양한 음성 생성 작업을 가능하게 합니다.

- **Performance Highlights**: Metis는 기존 최신 작업 특정(task-specific) 모델이나 다중 작업(multi-task) 시스템보다 뛰어난 성능을 보이며, 5종의 음성 생성 작업에서 최첨단 결과를 달성했습니다. 필요한 훈련 매개변수 2천만 개 이하나 300배 적은 훈련 데이터로도 이룩한 성과입니다. 또한, 텍스트, 오디오, 비디오와 같은 다중 모드 조건 입력을 지원하여 다양한 음성 생성 작업을 분산 처리할 수 있는 능력을 제공합니다.



### Disentanglement in Difference: Directly Learning Semantically Disentangled Representations by Maximizing Inter-Factor Differences (https://arxiv.org/abs/2502.03123)
- **What's New**: 본 연구에서는 디스엔탱글먼트 인 디프런스(Disentanglement in Difference, DiD)를 제안하여 잠재 변수의 통계적 독립성과 의미적 디스엔탱글먼트 간의 불일치를 해결하고자 합니다. 기존의 방법들이 잠재 변수 간의 통계적 독립성을 증진하는 방식으로 디스엔탱글먼트를 이루어왔다면, DiD는 잠재 변수의 통계적 독립성이 반드시 의미적으로 관련이 없음을 지적하며, 직접적으로 의미적 차이를 학습하도록 설계되었습니다. 이를 통해 모델은 다양한 의미적 요인을 식별하고 분리할 수 있게 됩니다.

- **Technical Details**: DiD 접근법에서는 차이 인코더(Difference Encoder)를 설계하여 의미적 차이를 측정하며, 대조 손실 함수(contrastive loss function)를 설정하여 차원 간 비교를 촉진합니다. 이러한 요소들은 모델이 서로 다른 의미적 요인의 각기 다른 표현을 직접적으로 구분할 수 있도록 합니다. 실험 결과, dSprites 및 3DShapes 데이터셋에서 기존의 관행적인 방법들보다 우수한 성능을 보였습니다.

- **Performance Highlights**: DiD는 디스엔탱글먼트 성능에 있어 최신 기술에 상응하는 성과를 달성하였으며, 다양한 디스엔탱글먼트 지표에서 기존의 주류 방법들보다 뛰어난 결과를 나타냈습니다. 특히, 모델은 각 요인의 영향을 받는 잠재 표현 간의 거리를 극대화하여 디스엔탱글먼트를 강화함으로써, 의미론적 차를 명확히 구분할 수 있음을 입증하였습니다. 이를 통해 인공지능의 객체 이해 및 조작 가능성이 더욱 향상될 것으로 기대됩니다.



### At the Mahakumbh, Faith Met Tragedy: Computational Analysis of Stampede Patterns Using Machine Learning and NLP (https://arxiv.org/abs/2502.03120)
Comments:
          6 pages, 4 figures, 3 tables

- **What's New**: 이번 연구는 인도 대규모 종교 집회에서 발생하는 치명적인 압사사고를 머신러닝(machine learning), 역사적 분석(historical analysis), 자연어 처리(NLP)를 통해 조사하였습니다. 특히 2025년 프라야그라지에서 발생한 마하쿰브 비극과 1954년의 선행 사건을 중심으로 분석합니다. 이 연구는 재난의 시스템적 취약성을 나타내는 방식으로 군중 동역학(crowd dynamics) 및 행정 기록을 컴퓨터 모델링합니다.

- **Technical Details**: 시간적 추세 분석(temporal trend analysis)을 통해 지속적으로 발생하는 압사 지점과 좁은 강변 접근 경로가 과거의 92%의 압사 발생지점과 연결되어 있음을 발견했습니다. 또한, 자연어 처리(NLP) 기법을 사용하여 70년간의 조사 보고서를 분석하며 VIP 경로 우선 배정이 재난 발생 시 안전 자원을 분산시키는 방식이란 점도 제기됩니다. 이와 같은 연구 결과들은 의식의 급박함(ritual urgency)이 위험 인식을 능가함에 따라 발생하는 공항 패턴(panic propagation patterns)을 설명합니다.

- **Performance Highlights**: 연구 결과는 재난 대응이 예방적(preventive)이지 않고 반응적(reactionary)이라는 제도적 망각 이론(Institutional Amnesia Theory)을 지지합니다. 아카이브 패턴과 컴퓨터의 군중 행동 분석(crowd behavior analysis)을 연계함으로써 이 연구는 종교적 긴급성이 어떻게 인프라 한계와 사회적 관리 무기력(governance inertia)과 충돌하는지를 보여줍니다. 이러한 분석을 통해 예방 가능한 사망자를 정상화하는 영적 경제(spiritual economies)의 현상에 도전합니다.



### Tell2Reg: Establishing spatial correspondence between images by the same language prompts (https://arxiv.org/abs/2502.03118)
Comments:
          5 pages, 3 figures, conference paper

- **What's New**: 본 논문에서는 이미지 등록(image registration) 네트워크가 변위 필드(displacement fields)나 변환 파라미터(transformation parameters)를 예측하는 대신, 대응하는 영역을 분할(segmentation)하도록 설계된 새로운 방법을 제안합니다. 이러한 접근법은 동일한 언어 프롬프트(language prompt)를 기반으로 두 개의 서로 다른 이미지에서 대응하는 영역 쌍을 예측할 수 있도록 하며, 이는 자동화된 등록 알고리즘을 가능하게 합니다. 이를 통해 기존의 데이터 커리션(data curation) 및 라벨링(labeling)이 필요 없어져, 시간과 비용을 절감합니다.

- **Technical Details**: 제안된 방법은 고정 이미지(fixed image)와 이동 이미지(moving image) 간의 대응하는 영역을 찾아내는 문제를 해결하기 위해, GroundingDINO 및 Segment Anything Model(SAM)과 같은 사전 학습된 다중 모달 모델(multimodal models)을 이용합니다. 이 과정에서는 동일한 텍스트 프롬프트를 사용하여 두 이미지를 입력으로 하여 의미론적 영역을 인식하게 됩니다. 이 알고리즘에서는 IMfix와 Imov를 각각 고정 및 이동 이미지로 설정하고, 대응하는 ROIs를 식별하게 됩니다.

- **Performance Highlights**: Tell2Reg 알고리즘은 기존의 비지도 학습(unsupervised learning) 기반 이미지 등록 방법들과 비교하여 뛰어난 성능을 보이며, 약한 감독(weakly-supervised) 방법들과 유사한 수준의 성능을 기록합니다. 이 접근법은 다양한 이미지 등록 작업에 일반화 가능하며, 특히 프로스트 prostate MRI 이미지의 등록에서 우수한 결과를 보여 주었습니다. 추가적인 정성적 결과 또한 언어 의미(semantics)와 공간 대응(spatial correspondence) 간의 잠재적 상관관계를 제시합니다.



### Policies and Evaluation for Online Meeting Summarization (https://arxiv.org/abs/2502.03111)
Comments:
          8 pages, 1 figure

- **What's New**: 이 논문은 온라인 회의 요약(online meeting summarization)에 대한 최초의 체계적인 연구를 수행합니다. 저자들은 회의가 진행되는 동안 요약을 생성하는 새로운 방식과 정책들을 제안하며, 온라인 요약 작업의 고유한 도전과제를 논의합니다. 기존의 오프라인 요약 방식과 달리, 이 연구는 요약 품질, 지연 시간(latency), 그리고 중간 요약(intermediate summary)의 품질을 평가하는 새로운 메트릭(metric)을 도입합니다.

- **Technical Details**: 온라인 요약 시스템은 주어진 입력 토큰 스트림(input stream)으로부터 읽고(output stream) 출력 토큰을 쓰는 에이전트(agent)로 간주됩니다. 이 시스템은 사전 학습된 오프라인 요약 시스템의 상단에 구축되어, 입력 및 출력 토큰을 읽고 쓰는 시점을 결정하는 다양한 정책을 제안합니다. 특히, 이 연구에서는 질과 지연 시간의 트레이드오프를 자세히 분석하기 위한 메트릭이 개발되었습니다.

- **Performance Highlights**: AutoMin 데이터셋에서 실시한 실험 결과, 온라인 모델이 강력한 요약을 생성할 수 있으며, 제안된 메트릭을 사용하면 다양한 시스템의 품질-지연 시간 간의 트레이드오프를 분석할 수 있음을 보여주었습니다. 평가 결과, 온라인 요약 시스템이 아직 오프라인 시스템과 동일한 수준의 품질에 도달하지는 못하지만, 인간 평가에서 4점 이상의 높은 점수를 기록하며 우수한 요약을 생성할 수 있는 것으로 나타났습니다.



### Bellman Error Centering (https://arxiv.org/abs/2502.03104)
- **What's New**: 이번 논문은 최근 제안된 보상 중심화 알고리즘, 즉 단순 보상 중심화(Simple Reward Centering, SRC)와 가치 기반 보상 중심화(Value-Based Reward Centering, VRC)를 재조명합니다. SRC는 보상 중심화를 구현하지만, VRC는 본질적으로 벨만 오차 중심화(Bellman Error Centering, BEC)에 해당함을 강조하고 있습니다. 본 연구는 BEC를 기반으로하여 표 형식(value function table) 값 함수에 대한 중심 고정점(central fixed point)과 선형 값 함수 근사(approximation)에 대한 중심 TD 고정점(central TD fixed point)을 제시합니다.

- **Technical Details**: 보상 중심화는 여러 강화 학습 알고리즘으로의 확장이 용이한 방법이며, 본 논문에서는 온-정책(On-Policy) CTD 알고리즘과 오프-정책(Off-Policy) CTDC 알고리즘을 설계하였습니다. 두 알고리즘의 수렴(convergence) 증명도 포함되어 있어, 이론적인 뒷받침을 제공합니다. 또한, 베이스라인(Baseline) 모델과 C-TD 알고리즘을 통한 실험을 통해 제안된 알고리즘의 안정성(stability) 또한 검증하였습니다.

- **Performance Highlights**: 제안된 탁월한 실험 결과는 Q-러닝(Q-learning)에서의 보상 중심화와 선형 근사, 그리고 심층 Q-네트워크(Deep Q-Networks, DQN) 적용을 통해 더욱 두드러집니다. 기존 연구에서 나타난 문제점들, 즉 보상 중심화의 다른 RL 알고리즘 통합의 복잡성 및 큰 상태 공간에서의 수렴 특성에 대한 미비한 이해도 해결되었습니다. 이 연구는 효율적인 RL을 위한 새로운 접근법을 제시하는데 기여하며, 다양한 강화 학습 분야에 걸쳐 중요한 이정표를 세우고 있습니다.



### E-3SFC: Communication-Efficient Federated Learning with Double-way Features Synthesizing (https://arxiv.org/abs/2502.03092)
Comments:
          Accepted by TNNLS. arXiv admin note: text overlap with arXiv:2302.13562

- **What's New**: 이 논문에서는 Federated Learning (FL)의 모델 크기가 커짐에 따라 증가하는 통신 부담을 해결하기 위한 새로운 접근 방식을 제안합니다. 논문에서 소개된 Extended Single-Step Synthetic Features Compressing (E-3SFC) 알고리즘은 고압축 효율성과 낮은 압축 오류를 동시에 달성하는 것을 목표로 합니다. 이를 위해 모델 자체를 그래디언트 디컴프레서로 활용하는 혁신적인 기법이 포함되어 있습니다.

- **Technical Details**: E-3SFC 알고리즘은 Single-Step Synthetic Features Compressor (3SFC), 이중 압축 알고리즘, 및 통신 예산 스케줄러로 구성되어 있습니다. 3SFC는 모델 가중치와 목표 함수와 같은 훈련 선험(probabilities)을 활용하여 원시 그래디언트를 작은 합성 특징으로 압축하며, 압축 오류를 최소화하기 위해 오류 피드백(error feedback)을 통합합니다. 이론적 분석을 통해 3SFC는 강한 볼록 및 비볼록 조건 하에서 선형 및 부분 선형 수렴 속도를 달성함을 증명합니다.

- **Performance Highlights**: 실험 결과에 따르면, E-3SFC는 기존의 최첨단 방법에 비해 최대 13.4% 더 높은 성능을 보였고, 통신 비용은 111.6배 절감되었습니다. 이는 FL에서 통신 효율성을 획기적으로 개선하면서도 모델 성능을 저하시키지 않을 수 있음을 나타냅니다. 이러한 결과는 E-3SFC가 FL의 훈련 효율성 및 통신 부담을 크게 줄일 가능성이 있음을 보여줍니다.



### Implementing Large Quantum Boltzmann Machines as Generative AI Models for Dataset Balancing (https://arxiv.org/abs/2502.03086)
Comments:
          accapted at IEEE International Conference on Next Generation Information System Engineering

- **What's New**: 이 연구는 양자 기계 학습(Quantum Machine Learning, QML)의 주요 발전인 대형 양자 제약 볼츠만 머신(Quantum Restricted Boltzmann Machines, QRBMs)을 도입하여 D-Wave의 Pegasus 양자 하드웨어에서 생성 모델로 구현했습니다. QRBM은 120개의 가시 노드(visible node)와 120개의 은닉 노드(hidden node)를 가진 구조로, 기존 도구의 한계를 넘어서는 성능을 보였습니다. 이 구현을 통해 데이터셋 불균형 문제를 해결하기 위해 1.6백만 개의 공격 샘플을 생성하여 균형 잡힌 데이터셋을 만드는 데 성공했습니다.

- **Technical Details**: 양자 제약 볼츠만 머신(QRBM)은 데이터셋의 통계적 특성을 학습하고 현실적인 데이터 샘플을 생성하는 생성 모델입니다. QRBM은 양자 역학의 원리를 이용하여 고차원 데이터 공간을 효과적으로 탐색하며, 복잡한 확률 분포를 더 잘 표현할 수 있습니다. 이 연구는 D-Wave의 Pegasus 아키텍처에서 QRBM을 구현하고, 172x120 크기의 RBM을 최적화하여 효율적으로 큰 모델을 임베딩(embedding)했습니다.

- **Performance Highlights**: QRBM을 통한 데이터셋 생성은 IDS 성능 지표에서 유의미한 향상을 가져왔습니다. QRBM으로 생성된 데이터셋은 기존의 SMOTE 및 Random Oversampling 기법에 비해 정밀도(precision), 재현율(recall) 및 F1 점수에서 우수한 결과를 보여 주었습니다. 이 연구 결과는 QRBM이 IDS의 신뢰성과 견고성을 향상시키는 데 중요한 도구임을 강조하며, 데이터 전처리에서 놀라운 가능성을 지니고 있음을 나타냅니다.



### Kozax: Flexible and Scalable Genetic Programming in JAX (https://arxiv.org/abs/2502.03047)
Comments:
          5 figures, 3 tables, 1 algorithm, 10 pages

- **What's New**: Kozax는 일반 문제에 대해 기호 표현을 발전시키기 위해 개발된 새로운 유전자 프로그래밍(GP) 프레임워크입니다. 기존의 GP 프레임워크들은 특정 문제에 제한적이었지만, Kozax는 JAX를 사용하여 GPU에서 큰 데이터셋을 효율적으로 처리할 수 있습니다. 이를 통해 Kozax는 빠르고 유연한 최적화 기능을 제공하며, 여러 트리의 동시 진화를 지원합니다.

- **Technical Details**: Kozax는 parse tree를 행렬로 표현하여 CPU와 GPU에서 병렬적으로 적합도(fitness)를 평가합니다. 이 접근 방식은 기존의 GP 프레임워크들과 비교했을 때 더 낮은 계산 시간과 높은 성능을 보여줍니다. Kozax는 또한 사용자가 원하는 기능을 정의할 수 있는 많은 추가 기능을 제공하며, 이를 통해 복잡한 문제에도 적용할 수 있습니다.

- **Performance Highlights**: Kozax는 다른 GP 라이브러리들과 속도와 성능에서 경쟁하며, 기호 회귀(symbolic regression) 문제에서 성공적인 결과를 보여줍니다. 다양한 문제를 처리하는 능력이 뛰어난 Kozax는 과학적 컴퓨팅 영역에서 최적화된 화이트박스 솔루션을 위한 일반적이고 확장 가능한 라이브러리입니다.



### xai_evals : A Framework for Evaluating Post-Hoc Local Explanation Methods (https://arxiv.org/abs/2502.03014)
- **What's New**: 최근 머신러닝(Machine Learning) 및 딥러닝(Deep Learning) 모델의 복잡성이 증가하면서, 비가시적 블랙박스 시스템에 대한 의존도가 높아지고 있습니다. 이는 예측의 근거를 이해하기 어려워지는 문제를 발생시키며, 특히 의료 및 금융과 같은 고위험 분야에서는 해석 가능성이 정확도만큼이나 중요합니다. 이러한 문제를 해결하기 위해 xai_evals라는 포괄적인 파이썬 패키지가 개발되어 다양한 설명 방법을 생성하고 평가할 수 있는 통합된 프레임워크를 제공합니다.

- **Technical Details**: xai_evals는 SHAP, LIME, Grad-CAM, Integrated Gradients(IG), Backpropagation 기반의 설명 기법 등을 통합하여 제공하며, 모델 유형에 관계없이 사용 가능합니다. 이 패키지는 설명의 질을 평가할 수 있는 다양한 메트릭스인 faithfulness, sensitivity, robustness를 포함하고 있습니다. 설명 생성을 통해 기계학습 모델의 해석 가능성을 향상시키며, 안전하고 투명한 AI 시스템의 배치를 촉진합니다.

- **Performance Highlights**: xai_evals는 설명 생성 및 평가 기능을 모두 제공하여 깊은 학습 모델에 대한 신뢰와 이해를 촉진합니다. 이를 통해 연구자들과 실무자들이 다양한 설명 가능성 방법의 품질, 안정성 및 신뢰성을 평가하고 비교할 수 있는 유니파이드 프레임워크를 경험하게 됩니다. 실험 결과는 xai_evals의 효과성을 보여주며, 향후 다양한 고위험 애플리케이션에 안전하게 적용될 수 있는 가능성을 제시합니다.



### MedBioLM: Optimizing Medical and Biological QA with Fine-Tuned Large Language Models and Retrieval-Augmented Generation (https://arxiv.org/abs/2502.03004)
- **What's New**: 이 논문에서는 MedBioLM이라는 생물의학 질문 응답 모델을 소개하고 있습니다. MedBioLM은 사실상 정확성과 신뢰성을 높이기 위해 도메인 적응 (domain-adapted) 기술을 활용하고 있으며, 기본 모델보다 높은 성능을 자랑합니다. 이 모델은 단기 및 장기 쿼리를 모두 위한 최적화된 솔루션을 제공합니다.

- **Technical Details**: MedBioLM은 사전 학습된 대형 언어 모델(LLM)을 의학 데이터셋에 맞게 미세 조정(fine-tuning)하고, 검색 강화 생성(RAG) 기술을 통합하여 구현됩니다. 이를 통해 모델은 외부 도메인 지식을 동적으로 포함시켜 사실적 정확도를 향상시키고, 의료 전문가에게 적합한 구조적인 응답을 생성할 수 있습니다. RAG는 특히 짧은 응답에서 사실적 정확성과 언어 유사성을 높이는데 중요한 역할을 합니다.

- **Performance Highlights**: MedBioLM은 MedQA와 BioASQ와 같은 다양한 생물의학 QA 데이터셋에서 정확도 88% 및 96%를 달성하여 기본 모델보다 월등한 성능을 입증하였습니다. 장기 응답 QA에서도 ROUGE-1 및 BLEU 점수가 개선되었으며, 전반적으로 LLM의 생물의학 응용 프로그램에 대한 가능성을 보여주고 있습니다. 이러한 결과는 도메인 최적화된 LLM이 생물의학 연구와 임상 의사결정 지원에 기여할 수 있음을 강조합니다.



### Training an LLM-as-a-Judge Model: Pipeline, Insights, and Practical Lessons (https://arxiv.org/abs/2502.02988)
Comments:
          accepted at WWW'25 (Industrial Track), extended version

- **What's New**: 이 논문에서는 평가 역할을 수행하는 언어 모델인 Themis를 소개하고 있다. Themis는 시나리오에 맞춰 세심하게 조정된 평가 프롬프트와 두 가지 새로운 지시 생성 방법을 통해.context-aware 평가를 제공한다. 이는 LLM의 평가 능력을 효과적으로 증류하며, 지속적인 발전을 위한 유연성을 보장한다.

- **Technical Details**: Themis는 평가를 위한 단계별 시나리오 의존 프롬프트를 사용하고, 인간-AI 협업을 통해 각 시나리오에 맞춰 평가 기준을 설계한다. 이 모델은 reference-based questioning과 role-playing quizzing 방법을 통해 지시 데이터를 생성하며, 이를 통해 LLM의 평가 능력을 더욱 향상시킨다. 지속적인 개발을 위한 유연성을 갖춘 Themis는 GPT-4와 같은 최신 LLM으로부터 평가 능력을 총체적으로 유도한다.

- **Performance Highlights**: Themis는 두 가지 인간 선호 벤치마크에서 높은 성과를 나타내며, 1% 미만의 매개변수를 사용하면서도 모든 다른 LLM을 초과하는 성능을 기록하였다. 또한, Themis의 성능은 공개된 시나리오에서 가장 잘 나타나며, 참고 답변이 평가에 미치는 영향을 분석한 결과, 폐쇄형 시나리오에서는 긍정적인 영향을 주지만 개방형 시나리오에서는 미미하거나 부정적인 영향을 미친다는 사실이 드러났다.



### TGB-Seq Benchmark: Challenging Temporal GNNs with Complex Sequential Dynamics (https://arxiv.org/abs/2502.02975)
Comments:
          published at ICLR 2025

- **What's New**: 현재 연구에서는 기존의 Temporal Graph Neural Networks (GNNs)가 복잡한 시퀀스 동역학을 효과적으로 학습하지 못하는 문제를 다루고 있습니다. 이를 해결하기 위해, 연구진은 반복된 엣지를 최소화하고 unseen edges를 학습하도록 설계된 새로운 벤치마크인 Temporal Graph Benchmark with Sequential Dynamics (TGB-Seq)를 제안합니다. TGB-Seq 데이터셋은 다양한 도메인의 실제 데이터를 포함하여, 기존 방식들이 간과했던 시퀀스 동역학을 효과적으로 평가할 수 있도록 구성되었습니다.

- **Technical Details**: TGB-Seq는 전자상거래, 영화 평점, 비즈니스 리뷰, 소셜 네트워크 등 다양한 도메인의 실제 대규모 데이터셋으로 구성되어 있습니다. 이 데이터셋은 모델이 발전된 시퀀스 동역학을 일반화하고 학습하는 것을 목표로 하며, 데이터셋 내에 반복 엣지가 적도록 신중하게 설계되었습니다. 평가 기준으로 Mean Reciprocal Rank (MRR)를 사용하며, 기존의 다양한 GNN 기법과 성능 비교를 실행합니다.

- **Performance Highlights**: TGB-Seq에서 진행한 벤치마킹 실험 결과, 여러 기존의 Temporal GNN 방법들이 성능 저하를 경험하며, 특히 우수한 성과를 보였던 기존 벤치마크와 비교하여 드라마틱한 성능 격차가 발생했습니다. 이는 그들이 복잡한 시퀀스 동역학을 포착하는데 한계가 있음을 보여줍니다. 이러한 현상은 향후 연구의 새로운 기회로 삼을 수 있을 것입니다.



### FACTER: Fairness-Aware Conformal Thresholding and Prompt Engineering for Enabling Fair LLM-Based Recommender Systems (https://arxiv.org/abs/2502.02966)
- **What's New**: FACTOR는 LLM 기반 추천 시스템을 위한 공정성 인식 프레임워크로, conformal prediction과 동적 프롬프트 엔지니어링을 통합합니다. 이 프레임워크는 편향 패턴이 발생할 경우 공정성 제약을 자동으로 조정하는 메커니즘을 제공합니다. 또한, 역사적 편향 사례를 분석하여 반복적인 인구통계적 편향을 줄이는 적대적 프롬프트 생성기를 개발하였습니다.

- **Technical Details**: FACTOR는 공정성 보장을 통해 semantic variance를 활용하여 추천 시스템의 공정성을 조정합니다. 비표준적 텍스트 출력의 차이를 측정하기 위해 임베딩 기반 분석을 사용하며, 추천 품목의 노출 불균형을 해결하기 위한 지속적인 프롬프트 수정을 포함합니다. Conformal prediction을 통해 예측 오류의 가능성을 관리함으로써 데이터 변동성을 고려하고, 수집된 데이터를 기반으로 적절한 공정성 기준을 설정합니다.

- **Performance Highlights**: 실험 결과, FACTER는 MovieLens와 Amazon 데이터셋에서 공정성 위반을 최대 95.5% 감소시키면서 추천 정확도를 유지하는 데 성공했습니다. 이 연구는 LLM 기반 결정 지원 시스템에 대한 공정성 개념의 확장을 보여주고, 더욱 신뢰할 수 있는 추천 모델 구축의 가능성을 제시합니다.



### ReachAgent: Enhancing Mobile Agent via Page Reaching and Operation (https://arxiv.org/abs/2502.02955)
- **What's New**: 최근 모바일 AI 에이전트가 주목받고 있습니다. 기존 에이전트는 특정 태스크와 관련된 요소에 집중하여 로컬 최적 해법에 그치고 전체 GUI 흐름을 간과하는 경우가 많았습니다. 이를 해결하기 위해 페이지 접근과 작업 서브태스크로 분리된 훈련 데이터셋인 MobileReach를 구축했습니다.

- **Technical Details**: 저자들은 ReachAgent라는 두 단계의 프레임워크를 제안하여 태스크 완료 능력을 향상시킵니다. 첫 번째 단계에서는 페이지 접근 능력 싸인을 배우고, 두 번째 단계에서는 4단계 보상 함수를 사용하여 GUI 흐름의 선호도를 강화합니다. 또한, 액션 정렬 메커니즘으로 작업 난이도를 줄입니다.

- **Performance Highlights**: 실험 결과, ReachAgent는 기존 SOTA(State of the Art) 에이전트에 비해 단계 수준에서 IoU Acc와 Text Acc가 각각 7.12% 및 7.69% 증가하고, 태스크 수준에서도 4.72% 및 4.63% 향상되었습니다. 이러한 결과는 ReachAgent가 더욱 뛰어난 페이지 접근 및 작업 능력을 가졌음을 보여줍니다.



### VQA-Levels: A Hierarchical Approach for Classifying Questions in VQA (https://arxiv.org/abs/2502.02951)
- **What's New**: 본 논문은 Visual Question Answering (VQA) 시스템을 체계적으로 테스트할 수 있도록 돕는 새로운 기준 데이터셋, VQA-Levels를 제안합니다. 이 데이터셋은 질문을 7단계로 분류하여, 낮은 수준의 이미지 특징에서부터 높은 수준의 추상화에 걸쳐 다양한 질문 형식을 포함합니다. 기존 데이터셋들이 가지고 있던 모호한 질문이나 무의미한 질문의 문제를 해결하고, 자연스러운 인간의 질문 형태를 따릅니다.

- **Technical Details**: VQA-Levels 데이터셋은 질문을 1에서 7까지의 수준으로 나누어, 각 수준은 시각적 내용에 따라 특성을 달리합니다. 1단계는 직접적인 답변이 가능하며, 높은 수준의 질문인 7단계는 이미지 전체의 맥락을 고려해야 하는 질문들을 포함합니다. 각 질문은 일반적으로 하나 또는 두 개의 단어로 된 답변을 요구하며, 독특한 특성을 기반으로 합니다.

- **Performance Highlights**: 초기 테스트 결과, 기존 VQA 시스템은 1단계 및 2단계 질문에서 높은 성공률을 보였으며, 3단계 이상의 질문에서는 성능이 감소하는 경향이 나타났습니다. 특히, 3단계는 화면 텍스트에 관한 질문, 6단계는 외삽(extrapolation), 7단계는 전반적인 장면 분석을 요구하는 질문으로, 난이도가 높은 만큼 응답 정확도가 낮았습니다. 이 연구 결과는 VQA 시스템의 성능을 체계적으로 분석할 수 있는 기초 자료를 제공합니다.



### LLM-KT: Aligning Large Language Models with Knowledge Tracing using a Plug-and-Play Instruction (https://arxiv.org/abs/2502.02945)
- **What's New**: 최근의 연구는 지식 추적(Knowledge Tracing) 문제에서 대규모 언어 모델(LLMs)의 활용 가능성을 제시합니다. 본 논문에서는 LLM-KT라는 새로운 프레임워크를 제안하여 LLM의 강력한 추론 능력과 전통적인 시퀀스 모델의 장점을 결합합니다. 이는 학생의 행동 패턴을 보다 정확하게 캡처하고 문제 해결 기록의 텍스트 맥락을 효율적으로 활용하는 데 중점을 둡니다.

- **Technical Details**: LLM-KT는 Plug-and-Play Instruction이라는 방법론을 사용하여 LLM과 지식 추적을 정렬합니다. 이 모델은 질문 및 개념에 특화된 토큰을 활용하여 전통적인 방법으로 학습한 여러 모달리티를 통합하는 Plug-in Context와 시퀀스 상호작용을 향상시키는 Plug-in Sequence를 설계하였습니다. 이러한 접근 방식을 통해 학생의 지식 상태를 보다 잘 이해할 수 있도록 합니다.

- **Performance Highlights**: 다양한 데이터셋에 대한 실험을 통해 LLM-KT는 약 20개의 강력한 기준선 모델과 비교하여 최첨단 성능(SOTA)을 달성했습니다. 이러한 결과는 LLM-KT 모델이 지식 추적 문제를 해결하는 데 있어 탁월한 장점을 보여줍니다. 추가적인 분석과 ablation 연구를 통해 여기에 기여하는 주요 요소들에 대한 효과도 입증되었습니다.



### Large Language Model Guided Self-Debugging Code Generation (https://arxiv.org/abs/2502.02928)
- **What's New**: 이번 연구에서는 PyCapsule이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 Python 코드 생성을 위해 간단하고 효율적인 두 개의 에이전트를 사용하는 파이프라인을 채택하고, 자가 디버깅 모듈을 통합했습니다. 결과적으로, PyCapsule은 HumanEval, HumanEval-ET, BigCodeBench 등에서 현존하는 방법들과 비교해 상당한 개선율을 보였습니다.

- **Technical Details**: PyCapsule은 프로그래머 에이전트와 실행자 에이전트로 구성됩니다. 프로그래머 에이전트는 코드 생성을 담당하고, 실행자 에이전트는 코드 검증과 오류 분석을 수행합니다. 또한, 세 가지 전문 모듈인 오류 처리기, 예제 호출 탐지기 및 함수 서명 변환기를 통해 디버깅 과정을 간소화하고 성능을 극대화합니다.

- **Performance Highlights**: PyCapsule은 HumanEval에서 최대 5.7%, HumanEval-ET에서 10.3%, BigCodeBench에서 24.4%의 성공률 개선을 기록했습니다. 그러나 자가 디버깅 시도 횟수가 증가함에 따라 정규화된 성공률이 감소하는 경향이 관찰되었습니다. 이는 제한적이고 시끄러운 오류 피드백이 자가 디버깅 과정에 영향을 미친 것으로 추측됩니다.



### TopoCL: Topological Contrastive Learning for Time Series (https://arxiv.org/abs/2502.02924)
Comments:
          Submitted to TNNLS (under review)

- **What's New**: 이번 연구는 Topological Contrastive Learning (TopoCL)을 제안하여 시계열 데이터의 표현 학습에서 정보를 잃지 않도록 한다. TopoCL은 persistent homology를 활용해 데이터의 위상적 특성을 캡처하여 데이터를 변형에 대해 불변으로 유지한다. 이 방법은 시계열 데이터의 시간적 및 위상적 속성을 구별된 모달리티로 다룬다.

- **Technical Details**: TopoCL은 시계열 데이터에서 persistent homology를 계산하여 위상적 특성을 persistence diagrams로 구성하고, 이를 인코딩하는 신경망을 설계한다. 이렇게 구성된 위상적 특성을 통해 데이터 증강 과정에서 발생할 수 있는 정보 손실을 줄여주며, 시간적 의미와 위상적 특성을 통합적으로 이해하는 데 초점을 맞춘다. 이 접근 방식은 시계열 데이터의 견고한 표현 학습에 기여한다.

- **Performance Highlights**: 실험 결과, TopoCL은 시계열 분류, 이상 탐지, 예측, 전이 학습 등 네 가지 하위 작업에서 최첨단 성능을 달성하였다. TopoCL은 데이터 증강에 대한 견고성을 개선하고, 시계열의 의미 있는 표현을 효과적으로 포착함으로써 기존의 방법들보다 뚜렷한 장점을 나타내었다. 이 연구는 시계열 데이터의 위상적 속성이 잘 결합된 표현 학습의 중요성을 입증할 뿐만 아니라 새로운 연구 방향을 제시한다.



### Adaptive Budget Optimization for Multichannel Advertising Using Combinatorial Bandits (https://arxiv.org/abs/2502.02920)
- **What's New**: 이번 논문에서는 디지털 광고의 예산 할당 전략을 개선하기 위해 세 가지 핵심 기여를 제안하였다. 첫째, 다양한 채널의 광고 캠페인을 시뮬레이션할 수 있는 환경을 구축하였으며, 이 데이터는 공개적으로 이용 가능하다. 둘째, 새로운 탐색 메커니즘과 변화점 검출을 통합한 향상된 조합 밴딧 예산 할당 전략을 제안하였다. 마지막으로, 이 방법이 기존의 전략보다 높은 보상과 낮은 후회를 달성한다는 이론적 분석과 실험적 결과를 제시하였다.

- **Technical Details**: 디지털 광고 예산 할당의 비효율성을 해결하기 위해, 이 논문은 장기 비정상(non-stationary) 캠페인에 적합한 조합 밴딧(combinatorial bandit) 접근 방식을 제시한다. 이 방법은 도메인 지식을 활용하여 효율성을 평가하고, 시간에 따른 시장 변화에 동적으로 적응하도록 설계되었다. 또한 제안된 방법의 후회의 하한을 O(T𝑇√T)로 설정하여 이론적 기반을 마련하였다.

- **Performance Highlights**: 실험 결과, 제안된 예산 할당 전략은 여러 실제 캠페인 데이터에서 SOTA(baseline) 전략들보다 높은 보상과 더 낮은 후회를 기록하였다. 특히, 도메인 지식을 통한 효과적인 목표 지역 필터링과 변화점 검출이 중요한 역할을 하며, 변화하는 시장 조건에 따른 적응력을 보였다. 여기서 얻은 데이터와 모델은 향후 연구에 귀중한 자원으로 활용될 수 있다.



### Interactive Symbolic Regression through Offline Reinforcement Learning: A Co-Design Framework (https://arxiv.org/abs/2502.02917)
Comments:
          arXiv admin note: text overlap with arXiv:2402.05306

- **What's New**: Sym-Q는 기호 회귀(Symbolic Regression)를 위한 새로운 상호작용형 프레임워크로, 강화 학습(reinforcement learning)을 사용하여 대규모 기호 회귀 문제를 해결하는 데 중점을 두고 있습니다. 기존의 트랜스포머 기반 모델과는 달리, Sym-Q는 트리 인코더(tree encoder)를 통해 에이전트가 수식 발견 과정에서 도메인 전문가와의 효과적인 상호작용을 지원합니다. 이 새로운 접근 방식은 기호 회귀의 유연성과 효율성을 높이며, 전문가의 사전 지식을 통합할 수 있는 메커니즘을 제공합니다.

- **Technical Details**: Sym-Q는 기존의 표현 트리(expression tree)를 인식하고 이를 통해 최적의 작업을 결정하여 수식을 발전시키는 강화 학습 에이전트로 구성됩니다. 다양한 유형의 인코더(예: RNN, 트랜스포머)를 사용하여 트리 표현을 처리하며, 이는 계산 효율성을 크게 개선합니다. 또한, 사용자는 수식 생성 시 노드를 동적으로 수정할 수 있어 개별 작업 수준에서의 의사 결정을 가능하게 합니다.

- **Performance Highlights**: Sym-Q는 SSDNC 벤치마크에서 기존의 SR 알고리즘을 초월하는 성능을 입증했습니다. 실세계 데이터셋에서도 인터랙티브 설계 메커니즘을 통해 현저한 성과 개선을 달성하였으며, 다른 최신 모델보다 더 큰 성능 향상을 나타냈습니다. 이를 통해 Sym-Q는 기호 회귀 분야에서의 가능성을 더욱 확대하는 결과를 보여주고 있습니다.



### MobiCLR: Mobility Time Series Contrastive Learning for Urban Region Representations (https://arxiv.org/abs/2502.02912)
Comments:
          Submitted to Information Sciences (under review)

- **What's New**: 최근 도시 지역의 효과적인 표현 학습이 도시 역학을 이해하고 스마트 도시 발전에 필수적인 접근법으로 주목받고 있습니다. 기존 연구들은 이동 데이터를 활용하여 잠재 표현을 생성하고 있지만, 인간 이동 패턴의 시간적 역학과 세부 의미를 포함하는 방법은 충분히 탐구되지 않았습니다. 이를 해결하기 위해 저자들은 MobiCLR이라는 새로운 모델을 제안하였으며, 이는 유입 및 유출 이동 패턴의 의미 있는 임베딩을 포착하도록 설계되었습니다.

- **Technical Details**: MobiCLR는 인스턴스별 대조 손실(instance-wise contrastive loss)을 적용하여 이동 데이터에 내재된 시간적 역학을 학습합니다. 이 모델은 택시 여행 기록을 사용하여 생성된 시간 시계열 데이터를 기반으로 하여, 도시 지역 내 유동 패턴의 효과적인 특징을 추출하도록 설계되었습니다. 대조 학습을 통해 유입 및 유출에 대한 구별력 있는 표현을 향상시키며, 출력 특징을 특정 유동 패턴과 정렬하는 정규화 기법도 개발하여 포괄적인 이동 역학을 이해할 수 있게 합니다.

- **Performance Highlights**: 시카고, 뉴욕, 워싱턴 D.C.에서 수행된 실험 결과, MobiCLR는 단일 지표(소득, 교육 수준)뿐만 아니라 복합 지수(사회적 취약성 지수)를 예측하는 데 있어 최첨단 모델들보다 우수한 성능을 보였습니다. 이 연구는 인구 이동의 동적 패턴을 효과적으로 포착하여 도시 지역의 기능적 역학에 대한 깊은 통찰을 제공합니다. MobiCLR의 성과는 기존의 접근 방식에 비해 눈에 띄는 개선을 시사합니다.



### SPARC: Subspace-Aware Prompt Adaptation for Robust Continual Learning in LLMs (https://arxiv.org/abs/2502.02909)
- **What's New**: 본 연구에서는 대형 언어 모델(LLMs)을 위한 경량의 연속 학습 프레임워크인 SPARC를 제안합니다. SPARC는 차원 축소된 공간에서 프롬프트 튜닝을 통해 효율적으로 작업에 적응할 수 있도록 설계되었습니다. 주성분 분석(PCA)을 활용하여 훈련 데이터의 응집된 부분 공간을 식별하며, 이는 훈련 효율성을 높이는 데 기여합니다.

- **Technical Details**: SPARC의 핵심 구조는 각 작업을 입력 임베딩 공간의 부분 공간으로 표현하고, 부분 공간의 중첩을 정량화하여 새로운 작업에 기존의 프롬프트를 재사용할 수 있는지를 판단합니다. 이는 계산 오버헤드를 줄이고 관련 작업 간의 지식 전이를 촉진합니다. 이 프레임워크는 모델의 기본 구조를 변경하지 않고 소프트 프롬프트만 업데이트하여 매개변수를 절약하는 효율성을 지니고 있습니다.

- **Performance Highlights**: 실험 결과, SPARC는 연속적인 학습 환경에서도 효과적으로 지식을 유지하고, 97%의 과거 지식 보존율을 기록했습니다. 도메인 간 학습에서는 평균 잊어버린 비율이 3%에 불과하고, 작업 간 학습에서는 잊어버림이 전혀 발생하지 않는 것으로 나타났습니다. 이러한 성과는 모델의 0.04%의 매개변수만 미세 조정함으로써 이루어졌습니다.



### What is in a name? Mitigating Name Bias in Text Embeddings via Anonymization (https://arxiv.org/abs/2502.02903)
- **What's New**: 이 논문에서는 텍스트 임베딩 모델에서 이제까지 연구되지 않은 '이름 편향(name-bias)'을 다룹니다. 이름은 개인, 장소, 조직 등을 포함하여 문서의 내용에 중대한 영향을 미칠 수 있으며, 이는 잘못된 유사성 판단으로 이어질 수 있습니다. 후보 모델들이 동일한 의미를 지니지만 다른 캐릭터 이름을 가진 두 문서의 유사성을 인식하지 못하는 문제를 지적하고 있습니다.

- **Technical Details**: 연구에서는 전문적인 분석을 통해 텍스트에 포함된 이름이 임베딩 생성 과정에서 어떠한 편향의 원인이 되는지를 설명하고 있습니다. '텍스트 익명화(text-anonymization)' 기법을 제안하며, 이는 모델을 재훈련할 필요 없이 이름을 제거하면서 본문의 핵심을 유지할 수 있습니다. 이 기법은 두 가지 자연어 처리(NLP) 작업에서 효율성과 성능 향상을 입증하였습니다.

- **Performance Highlights**: 이름 편향을 감소시키기 위해 제안한 익명화 기법은 다양한 텍스트 임베딩 모델과 작업에서 철저한 실험을 통해 효과적임을 입증하였습니다. 이전에 존재하던 다양한 사회적 편향 연구와 비교하여, 이 연구는 특히 이름에 관련된 편향 문제를 최초로 제기하며, 이를 해결하기 위한 간단하고 직관적인 접근법을 제공합니다.



### Policy Abstraction and Nash Refinement in Tree-Exploiting PSRO (https://arxiv.org/abs/2502.02901)
- **What's New**: 본 논문에서는 Policy Space Response Oracles (PSRO)와 Deep Reinforcement Learning (DRL)을 결합하여 복잡한 게임의 해결 방법을 제시합니다. 특히, Tree-exploiting PSRO (TE-PSRO)는 대규모 게임 모형을 구축하여 실험적으로 효과성을 증대시키는 접근 방식을 적용하고 있습니다. 새로운 두 가지 방법론적 발전을 통해 복잡한 비대칭 정보 게임에 대한 활용 가능성을 높이고 있습니다.

- **Technical Details**: TE-PSRO 모델은 DRL을 통해 학습된 암묵적 정책에 따라 게임 트리를 구성하며, 이를 통해 전략 탐색 시 공평한 성장을 지원합니다. 또한, 우리의 방향으로, refined Nash equilibria를 활용하여 전략 탐색을 지향하는 수정된 추정 알고리즘을 도입하였습니다. 이러한 접근법은 임의의 정보 게임에서 Subgame Perfect Equilibrium (SPE)을 계산하는 모듈화된 알고리즘을 기반으로 합니다.

- **Performance Highlights**: 실험 결과, 새로운 전략이 SPE를 기반으로 생성되었을 때 TE-PSRO가 Nash equilibrium을 사용했을 때보다 더 빠르게 균형에 도달하는 것으로 나타났습니다. 본 연구는 다양한 복잡한 게임을 기반으로 하여 발전된 TE-PSRO의 효용성을 검증하고 있으며, 지속 가능한 모델 성장과 함께 시간이 지나도 합리적인 메모리 요구사항을 유지합니다.



### A Benchmark for the Detection of Metalinguistic Disagreements between LLMs and Knowledge Graphs (https://arxiv.org/abs/2502.02896)
Comments:
          6 pages, 2 tables, to appear in Reham Alharbi, Jacopo de Berardinis, Paul Groth, Albert Meroño-Peñuela, Elena Simperl, Valentina Tamma (eds.), ISWC 2024 Special Session on Harmonising Generative AI and Semantic Web Technologies. this http URL (forthcoming), for associated code and data see this https URL

- **What's New**: 이 논문은 LLM(대형 언어 모델)과 KG(지식 그래프) 간의 메타 언어적 불일치가 존재할 수 있음을 제안합니다. 이는 기존의 사실 불일치 평가 방식에 새로운 관점을 추가하며, 데이터와 지식을 LLM에 통합하는 지식 그래프 공학을 위한 새로운 평가 기준을 만듭니다. 또한, LLM의 오류 원인으로 합리적이지 않은 논의 차원을 고려해야 함을 강조합니다.

- **Technical Details**: 연구팀은 T-REx 데이터셋을 사용하여 LLM의 출력에서 메타 언어적 불일치가 발생하는지를 확인하기 위한 실험을 실시했습니다. 100개의 위키피디아 초록을 샘플링하여 250개의 사실 삼중(triple)을 평가하고, LLM을 판별자로 사용하여 메타 언어적 불일치 여부를 결정했습니다. 이 과정에서 LLM 간의 처리 방식에 따른 오해 가능성도 언급되었습니다.

- **Performance Highlights**: 실험 결과, 유효한 250개의 샘플에서 메타 언어적 불일치의 비율은 평균 0.097로 나타났습니다. 초기 실험의 한계로는 표본 크기와 인간 검증 부재가 지적되었습니다. 향후 연구에서는 메타 언어적 불일치를 명확히 구분할 수 있는 벤치마크 데이터셋이 필요하다고 제안하였습니다.



### Expertized Caption Auto-Enhancement for Video-Text Retrieva (https://arxiv.org/abs/2502.02885)
- **What's New**: 본 논문은 Expertized Caption auto-Enhancement (ExCae) 방법을 제안하여 비디오-텍스트 검색 작업에서 캡션의 품질을 향상시키고 비서적 임무를 최소화합니다. 또한, 캡션 자동 개선 및 전문가 선택 메커니즘을 통해 비디오에 맞춤형 캡션을 제공하여 크로스 모달 매칭을 용이하게 합니다. 이 방법은 데이터 수집 및 계산 작업의 부담을 줄이며, 개인화된 매칭을 통해 자기 적응성을 향상시킵니다.

- **Technical Details**: ExCae 방법은 Caption Self-improvement (CSI) 모듈과 Expertized Caption Selection (ECS) 모듈로 구성됩니다. CSI 모듈은 다양한 관점에서 비디오를 설명하는 캡션을 생성하고, ECS 모듈은 학습 가능한 전문가를 통해 적합한 표현을 자동으로 선택합니다. 이 방법은 전통적인 텍스트 증강 방식과 달리 비디오에서 자동으로 캡션을 유도하여 비디오-텍스트 매칭을 위한 임베딩을 인코딩합니다.

- **Performance Highlights**: 저자들이 제안한 방법은 MSR-VTT에서 68.5%, MSVD에서 68.1%, DiDeMo에서 62.0%의 Top-1 recall accuracy를 달성하여 최신 방법들을 앞지르는 성능을 입증했습니다. 이 연구는 추가 데이터 없이도 비디오-텍스트 검색 작업에서 일관된 성능 향상을 보여주었으며, MSR-VTT에서의 베스트 벤치마크를 3.8% 초과 달성했습니다.



### Vertical Federated Learning for Failure-Cause Identification in Disaggregated Microwave Networks (https://arxiv.org/abs/2502.02874)
Comments:
          6 pages, 7 figure, IEEE ICC 2025

- **What's New**: 이번 연구는 분산화된 마이크로웨이브 네트워크에서 Federated Learning (FL)을 적용하여 고장 원인 식별 문제를 해결하는 방법을 다룹니다. 특히, Split Neural Networks (SplitNNs)와 Gradient Boosting Decision Trees (FedTree) 기반의 Vertical Federated Learning (VFL) 기술을 활용하여 다양한 다중 공급업체 배치 시나리오에서 성능을 평가하였습니다. 이를 통해, 기존의 중앙집중형 네트워크 모델과 비슷한 수준의 성능을 유지하면서도 민감한 데이터 유출을 최소화할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 연구에서는 Federated Learning의 두 가지 주요 타입인 Horizontal Federated Learning (HFL) 및 Vertical Federated Learning (VFL)에 대해서 설명하며, 특히 VFL이 다양한 공급업체가 관리하는 분산화된 네트워크에 적합하다고 강조합니다. 이러한 FL 기술은 모델 업데이트만 중앙 서버에 공유하면서, 원본 데이터를 보호할 수 있는 장점을 가지고 있습니다. 연구 결과 GBDT 및 SplitNN VFL 기술을 통해 다중 공급업체 환경에서도 효과적인 고장 원인 식별을 실현하였음을 확인했습니다.

- **Performance Highlights**: 실험 결과, 제안한 VFL 솔루션은 중앙 집중형 시나리오와 유사한 성과를 보이며, 최대 평균 F1-Score는 96.44%로 중앙 집중형 결과인 96.49%와 매우 근접하였습니다. 이러한 결과는 VFL 기반의 접근 방식이 실제 마이크로웨이브 하드웨어 고장 데이터셋을 활용하여도 뛰어난 성능을 발휘할 수 있음을 입증합니다. 이 연구는 분산화된 마이크로웨이브 네트워크에서의 모델 협업 가능성을 확인하는 중요한 시사점을 제공합니다.



### Position: Multimodal Large Language Models Can Significantly Advance Scientific Reasoning (https://arxiv.org/abs/2502.02871)
- **What's New**: 이번 논문은 Multimodal Large Language Models (MLLMs)의 통합이 과학적 추론 (scientific reasoning)에 혁신적인 발전을 가져올 수 있음을 주장합니다. 기존의 과학적 추론 모델들이 여러 분야에서의 일반화에 어려움을 겪는 점을 지적하며, MLLMs가 텍스트, 이미지 등 다양한 형태의 데이터를 통합하고 추론할 수 있는 능력을 강조합니다. 이로 인해 수학, 물리학, 화학, 생물학 등 여러 분야에서의 과학적 추론이 크게 향상될 수 있음을 제안합니다.

- **Technical Details**: MLLMs는 다양한 타입의 데이터를 처리하고, 이를 바탕으로 추론 (reasoning)을 수행할 수 있는 능력을 갖추고 있습니다. 연구 로드맵은 네 단계로 구성되어 있으며, 이 단계들은 과학적 추론의 다양한 능력을 개발하는 데 필수적입니다. 또한, MLLM의 과학적 추론에서 현황을 요약하고, 나아가 이들이 직면한 주요 도전 과제를 설명합니다.

- **Performance Highlights**: MLLMs의 도입으로 인해 과학적 추론의 응용 가능성이 커지고 있으며, 특히 다양한 데이터 타입을 통합하여 보다 정교한 결과를 도출할 수 있습니다. 그러나 MLLM이 완전한 잠재력을 발휘하기 위해 해결해야 할 주요 도전 과제가 존재합니다. 이러한 문제에 대한 실행 가능한 통찰 및 제안을 통해 AGI (Artificial General Intelligence)를달성하는 데 기여할 수 있는 방안을 모색합니다.



### OmniRL: In-Context Reinforcement Learning by Large-Scale Meta-Training in Randomized Worlds (https://arxiv.org/abs/2502.02869)
Comments:
          Preprint

- **What's New**: OmniRL이라는 새로운 강화학습 모델이 도입되었습니다. 이 모델은 수십만 개의 다양한 태스크에서 메타 훈련(meta-training)되어 높은 일반화 가능성을 보여줍니다. OmniRL은 임시 생성된 Markov Decision Processes(MDPs)를 활용하여 복잡한 환경에서의 태스크 적응 능력을 갖추고 있습니다.

- **Technical Details**: 이 논문에서 제안된 OmniRL은 혁신적인 데이터 합성 파이프라인을 통해 다양한 행동 정책의 상호 작용 히스토리를 활용합니다. 이 모델은 imitation learning 및 reinforcement learning(RL)의 통합된 프레임워크를 사용하여 prior knowledge를 포함하며, 이를 통해 ICL(in-context learning) 기능을 강화합니다. AnyMDP라는 새로운 태스크 및 환경의 집합도 소개되어 높은 확장성을 제공합니다.

- **Performance Highlights**: OmniRL은 기존의 ICRL 프레임워크를 능가하는 성능을 보이며, 새로운 Gymnasium 환경들에서 일반화 과정을 성공적으로 수행했습니다. 또한, 메타 훈련 태스크 수가 ICRL 능력에 미치는 영향에 관한 정량적 분석도 수행하여, 특정 태스크 식별 지향의 few-shot learning과 일반적인 ICL 간의 균형을 강조했습니다. 이러한 결과는 긴 경로를 다루는 것이 일반화 성능 향상에 필수적임을 보여줍니다.



### Domain-Invariant Per-Frame Feature Extraction for Cross-Domain Imitation Learning with Visual Observations (https://arxiv.org/abs/2502.02867)
Comments:
          8 pages main, 19 pages appendix with reference. Submitted to ICML 2025

- **What's New**: 이 논문에서는 Imitation Learning (IL)에서 cross-domain(크로스 도메인) 시나리오의 복잡한 비주얼 과제를 해결하기 위해 Domain-Invariant Per-Frame Feature Extraction for Imitation Learning (DIFF-IL)이라는 새로운 방법을 제안합니다. DIFF-IL은 개별 프레임에서 도메인 불변 특징을 추출하고 이를 통해 전문가의 행동을 고립하여 복제하는 메커니즘을 도입합니다. 또한, 프레임 단위의 시간 라벨링 기법을 통해 특정 타임스텝에 따라 전문가 행동을 세분화하고, 적합한 보상을 할당함으로써 학습 성능을 향상시킵니다.

- **Technical Details**: DIFF-IL은 두 가지 주요 기여를 포함합니다: (1) 프레임 단위의 도메인 불변 특징 추출을 통해 도메인 독립적인 과제 관련 행동을 고립하고, (2) 프레임 단위 시간 라벨링을 통해 전문가 행동을 타임스텝별로 구분하여, 시간에 맞춘 보상을 할당합니다. 이러한 혁신을 통해 DIFF-IL은 소스 도메인 데이터와 전문가 행동 간의 제한된 중첩 상황에서도 정확한 도메인 정렬 및 효과적인 모방을 가능하게 합니다.

- **Performance Highlights**: DIFF-IL의 성능은 다양한 비주얼 환경에서 실험을 통해 입증되었습니다. 특히, Walker에서 Cheetah로의 전이 환경에서는 DIFF-IL이 단일 이미지 프레임의 잠재 특징을 효과적으로 조정하여 도메인 특정 아티팩트를 제거하며, 전문가 행동을 정확하게 모방하는 것을 가능하게 합니다. 이로 인해, DIFF-IL은 과제가 복잡한 조건에서도 성공적인 작업 완료를 보장할 수 있습니다.



### A Systematic Approach for Assessing Large Language Models' Test Case Generation Capability (https://arxiv.org/abs/2502.02866)
Comments:
          17 pages, 9 figures

- **What's New**: 이번 연구에서는 LLMs(대형 언어 모델)를 이용한 단위 테스트 생성의 가능성을 탐구하고, LLM 생성 테스트 케이스 평가를 위한 새로운 기준인 GBCV(Generated Benchmark from Control-Flow Structure and Variable Usage Composition)를 제안합니다. GBCV는 기본적인 제어 흐름 구조(control-flow structures)와 변수 사용(variable usage)을 활용하여 다양한 프로그램을 생성하고, 이를 통해 LLM의 테스트 생성 능력을 평가할 수 있는 유연한 프레임워크를 제공합니다. 연구결과 GPT-4o는 복잡한 프로그램 구조에서 뛰어난 성능을 보여주었으며, 모든 모델이 단순 조건의 경계 값을 효과적으로 감지하지만, 산술 계산에 어려움을 겪는 경향이 있음을 발견했습니다.

- **Technical Details**: 연구의 주요 목표는 GBCV 방법론을 소개하고, 이를 통해 기존 LLM의 테스트 생성 능력을 평가하는 것입니다. GBCV는 사용자에게 다양한 제어 흐름 구조를 개발하여 사용자 정의 변수 사용을 정의할 수 있도록 하여, 테스트 프로그램 생성을 가능하게 합니다. LLM의 단위 테스트 생성을 돕기 위해 생성된 프로그램을 프롬프트로 사용하고, 이 테스트 케이스를 프로그램에 적용하여 LLM의 기능을 평가합니다.

- **Performance Highlights**: 연구 결과에서 나타난 바와 같이, GPT-4o는 복잡한 구조에 대해 우수한 성능을 보였으며, 모든 LLM은 단순한 조건에서 경계 값을 잘 감지했습니다. 그러나 산술 계산에 있어서는 여전히 도전 과제가 남아있습니다. GBCV 방법론을 사용하여 LLM의 성능을 체계적으로 평가함으로써, 고품질 테스트 케이스 생성을 위한 LLM의 발전 가능성을 강조하고 향후 연구 방향을 제시합니다.



### OceanChat: The Effect of Virtual Conversational AI Agents on Sustainable Attitude and Behavior Chang (https://arxiv.org/abs/2502.02863)
Comments:
          21 pages, 18 figures, 2 tables

- **What's New**: 이 연구는 OceanChat이라는 대화형 시스템을 제공하며, 이는 대화형 AI 에이전트가 애니메이션된 해양 생물로 표현되어 있으며, 환경 행동 촉진(PEB)과 인식 증진을 목표로 합니다. 실험 결과, 대화형 캐릭터 내러티브는 정적 접근 방식에 비해 행동 의도 및 지속 가능한 선택 선호를 유의미하게 증가시켰습니다. 특히, 벨루가 고래 캐릭터는 감정적 참여에서 고른 점수를 보여주었습니다.

- **Technical Details**: OceanChat은 대화형 AI와 종-specific 캐릭터 디자인을 통해 인간과 해양 생물의 관계를 재구성합니다. 이 연구에서는 세 가지 조건을 비교하는 실험(N=900)을 실시했으며, 각각 정적 과학 정보, 정적 캐릭터 내러티브, 대화형 캐릭터 내러티브의 영향을 분석하였습니다. 연구 결과, 인터랙티브한 대화형 에이전트가 PEB 촉진에 효과적인 것으로 나타났습니다.

- **Performance Highlights**: 이 연구는 AI 생성 해양 캐릭터의 설계 세부사항과 PEB에 미치는 영향을 다각도로 분석하였으며, 특히 캐릭터의 인지적 특징과 감정적 연관성이 중요하다는 점이 강조됩니다. 연구 결과는 감정적으로 공감할 수 있는 캐릭터를 사용하는 것이 친환경 행동을 촉진하는 데 어떻게 기여하는지를 보여줍니다. 또한, OceanChat의 전반적인 시스템 검증을 위해 대화형 AI의 이점을 살린 고급 정보를 제공하여, 지속 가능성을 위한 인간-컴퓨터 상호작용(HCI) 연구에 기여하고 있습니다.



### Learning Generalizable Features for Tibial Plateau Fracture Segmentation Using Masked Autoencoder and Limited Annotations (https://arxiv.org/abs/2502.02862)
Comments:
          5 pages, 6 figures

- **What's New**: 이 논문에서는 슬개골 지지 구역 골절(TPF)의 자동 분할을 위한 새로운 훈련 전략을 제시합니다. 제안된 방법은 Masked Autoencoder (MAE)를 활용하여 레이블이 없는 데이터로부터 전신 해부 구조 및 세부적인 골절 정보를 학습하여, 레이블이 있는 데이터의 사용을 최소화합니다. 이렇게 함으로써, 기존의 반지도학습 방법들이 직면한 문제를 해결하고, 다양한 골절 패턴의 일반화 및 전이 가능성을 향상시킵니다.

- **Technical Details**: 제안된 분할 네트워크는 MAE 프리트레이닝 단계와 UNETR 파인튜닝 단계로 구성됩니다. MAE는 비대칭 인코더-디코더 설계를采用하여 입력 이미지의 일부 패치를 마스킹하고, 나머지 마스크된 패치를 재구성하는 방식으로 작동합니다. UNETR은 인코더에서 생성된 피처 시퀀스를 디코더에서 해상도를 복원하는 데 사용하여 고해상도 분할 출력을 생성합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 180개의 TPF CT 스캔으로 구성된 자체 데이터세트에서 평균 Dice 유사성 계수 95.81%, 평균 대칭 표면 거리 1.91mm, 그리고 하우스도르프 거리 9.42mm를 달성하며 반지도학습 방법을 일관되게 초월합니다. 것이 주목할 점은 공용 골반 CT 데이터셋에서의 좋은 전이 가능성을 보여주며 다양한 골절 분할 작업에서의 응용 가능성을 강조합니다.



### Wolfpack Adversarial Attack for Robust Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2502.02844)
Comments:
          8 pages main, 21 pages appendix with reference. Submitted to ICML 2025

- **What's New**: 이번 논문에서 제안한 Wolfpack Adversarial Attack 프레임워크는 늑대의 사냥 전략에서 영감을 받아 다수의 에이전트를 동시에 목표로 하는 독창적 공격 전략을 포함합니다. 이는 협력적인 다중 에이전트 강화 학습(MARL) 환경에서 협력을 방해하는 데 중점을 두며, 새로운 방어 메커니즘의 필요성을 강조합니다. 또한, WALL 프레임워크를 도입하여 협력 강화를 통해 공격을 방어하는 정책 훈련 방법을 선보입니다.

- **Technical Details**: Multi-agent Reinforcement Learning (MARL)에서 협력과 경쟁을 요구하는 복잡한 문제 해결을 위해 Centralized Training and Decentralized Execution (CTDE) 프레임워크를 활용하여 에이전트가 지역 관찰에 기반하여 정책을 실행합니다. 기존 CTDE 기법은 강건성 부족과 훈련 및 배치 환경 간의 부조화로 인한 예상치 못한 행동의 문제에 직면해 있습니다. 논문에서는 기존의 공격 방법들이 단일 에이전트에 국한되어 협력적 MARL의 상호 의존성을 간과한다는 점을 지적합니다.

- **Performance Highlights**: 실험 결과에 따르면 전통적인 MARL 방법은 Wolfpack 공격에 매우 취약한 것으로 나타났습니다. WALL 프레임워크는 공격 상황에서도 높은 성능을 유지하면서 기존 방법들에 비해 강건성을 크게 향상시키는 것으로 평가되었습니다. 새롭게 제안된 Wolfpack Adversarial Attack 전략은 여러 에이전트를 동시에 겨냥함으로써 정책 훈련 중 강력하고 복원력 있는 협력을 유도하는 효과가 있습니다.



### Task-Aware Virtual Training: Enhancing Generalization in Meta-Reinforcement Learning for Out-of-Distribution Tasks (https://arxiv.org/abs/2502.02834)
Comments:
          8 pages main paper, 19 pages appendices with reference, Submitted to ICML 2025

- **What's New**: 이 논문에서는 Task-Aware Virtual Training (TAVT)라는 새로운 알고리즘을 제안합니다. 이 방법은 가상 훈련(task)에서의 작업 특성을 잘 포착하여 새로운 환경에서도 성능 향상을 목표로 합니다. TAVT는 기존의 메타 강화 학습(meta-RL) 기법들이 OOD(Out-Of-Distribution) 작업에 대한 일반화의 한계를 극복하도록 돕습니다.

- **Technical Details**: TAVT는 메트릭 기반의 표현 학습(metric-based representation learning)을 사용하여 훈련 및 OOD 시나리오 모두에서 작업 특성을 정확하게 캡처합니다. Bisimulation 메트릭을 활용하여 작업 변형을 포착하고, 작업 특성을 보존하는 샘플 생성을 통해 현실적인 샘플 컨텍스트를 생성합니다. 또한, 상태 변화 환경에서의 과대 평가 오류를 완화하기 위한 상태 정규화(state regularization) 기법을 도입합니다.

- **Performance Highlights**: 실험 결과, TAVT는 다양한 MuJoCo 및 MetaWorld 환경에서 OOD 작업에 대한 일반화를 크게 향상시켰습니다. 특히, TAVT는 기존 방법들이 OOD 테스트 작업의 표현을 산란시키는 반면, 작업 특성을 더 효과적으로 반영하는 작업 잠재적(task latent)을 정렬하는 데 성공했습니다. 이러한 성과는 TAVT가 메타 강화 학습 분야에서의 새로운 진전을 이룬 것을 보여줍니다.



### Mol-LLM: Generalist Molecular LLM with Improved Graph Utilization (https://arxiv.org/abs/2502.02810)
- **What's New**: 최근의 연구들은 분자 작업에 대한 일반 LLM(대형 언어 모델)의 개발을 촉진했습니다. 이들은 분자 구조에 대한 근본적인 이해가 부족하여 진정한 일반주의 분자 LLM에 도달하지 못했습니다. 본 논문에서 제안한 Mol-LLM은 고유한 다중 모달 지침 조정(multi-modal instruction tuning)을 통해 이러한 한계를 극복하고, 다양한 분자 작업에서 경쟁력 있는 성능을 발휘합니다.

- **Technical Details**: Mol-LLM은 텍스트 기반 SMILES와 2D 분자 그래프를 모두 입력으로 사용하여 다중 작업을 수행할 수 있는 통합 모델입니다. 기존의 1D 시퀀스와 2D 그래프를 결합하는 이번 접근법은 그래프 구조를 선호하는 최적화를 포함하여 모델이 다양한 분자 작업을 효과적으로 수행할 수 있게 합니다. 이러한 기술은 지침 조정과 그래프의 조건적 사용을 통해 모델이 분자 구조를 더욱 잘 이해하게 합니다.

- **Performance Highlights**: Mol-LLM은 분자 특성 예측, 화학 반응 예측 및 분자 설명 생성 작업에서 최신 성능을 기록하며, 기존의 전문화된 LLM보다도 우수하거나 동등한 성능을 보여줍니다. 실험 결과, 이 모델은 다중 모달 및 다중 작업 교육을 통해 개선된 그래프 모달리티의 활용을 통해 다양한 분자 사전 및 반응 예측 작업에서 우수한 일반화 성능을 보여줍니다.



### Upweighting Easy Samples in Fine-Tuning Mitigates Forgetting (https://arxiv.org/abs/2502.02797)
Comments:
          49 pages, 4 figures, 12 tables. Code available at this https URL

- **What's New**: 이 논문에서는 데이터에 대한 접근이 없는 상황에서의 미세 조정(fine-tuning) 과정에서 발생하는 '재앙적 망각(catastrophic forgetting)' 문제를 해결하기 위한 새로운 방법을 제안합니다. 제안된 방법은 사전 훈련된(pre-trained) 모델의 손실(loss)에 기반한 샘플 가중치(sample weighting) 체계를 사용하는 것으로, 기존의 방법들과는 다르게 샘플 공간(sample space)에 집중합니다. 특히, 손실이 낮은 '쉬운' 샘플의 가중치를 높여주고, 반대로 손실이 높은 샘플의 가중치를 낮추는 방식으로 사전 훈련된 모델의 성능 유지에 기여합니다.

- **Technical Details**: 논문에서 제안하는 방법인 FLOW(결과 매핑을 통한 샘플 기반 손실 가중치 적용)를 통해 미세 조정 과정에서의 손실(loss)과 가중치(weight) 사이의 관계를 확립합니다. 손실이 낮은 샘플의 가중치는 exp(-ℓᵢ/τ)로 정의되며, 여기서 ℓᵢ는 특정 샘플의 사전 훈련된 손실을 나타냅니다. 이 방법은 매개변수가 필요 없는 것으로, 기존의 다양한 방법들과 비교하여 새로운 차별성을 가지고 있습니다.

- **Performance Highlights**: FLOW 방법을 사용하여 이미지 분류 및 언어 모델의 실험에서 기존 미세 조정과 비교했을 때, 평균 정확도에서 약 17% 더 높은 성능을 달성했습니다. Gemma 2 2B 모델을 수학 데이터셋에 미세 조정할 경우, 표준 미세 조정에 비해 약 4%의 성능 향상을 이루었습니다. 또한 기존의 망각 해소 방법들과의 조합을 통해 성능이 더욱 개선됨을 실험적으로 증명하였습니다.



### Speculative Prefill: Turbocharging TTFT with Lightweight and Training-Free Token Importance Estimation (https://arxiv.org/abs/2502.02789)
- **What's New**: 이 논문은 SpecPrefill이라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 대규모 언어 모델(LLM)의 추론에서 시간-첫 토큰 (TTFT)을 개선하여, 다양한 길이의 쿼리에 적용 가능하도록 설계되었습니다. SpecPrefill은 사전 훈련 없이 로컬에서 중요한 토큰을 추측하여 전체 계산량을 줄이고, 더 높은 QPS(Queries Per Second)를 달성할 수 있습니다.

- **Technical Details**: SpecPrefill은 경량 모델을 사용하여 주어진 문맥에 기반하여 중요 토큰을 선택적으로 예측합니다. 이 토큰들은 이후 메인 모델에 전달되어 처리됩니다. 이러한 접근 방식은 모델의 FLOPS(부동 소수점 연산 수)를 줄여주며, 모델의 훈련 데이터나 마무리 작업 없이도 쉽게 배포 가능합니다.

- **Performance Highlights**: SpecPrefill은 Llama-3.1-405B-Instruct-FP8 모델의 경우 최대 7배의 QPS 향상과 7.66배의 TTFT 감소를 견인했습니다. 다양한 실제 및 합성 데이터 세트를 대상으로 한 평가를 통해 이 프레임워크의 효과성과 한계를 검증하였습니다. 또한, SpecPrefill은 기존 기술들과 결합하여 더 큰 모델에게도 확장 가능하다는 장점을 가지고 있습니다.



### Inducing Diversity in Differentiable Search Indexing (https://arxiv.org/abs/2502.02788)
- **What's New**: 이번 연구에서는 Differentiable Search Index (DSI)라는 새로운 정보 검색 방법론을 도입하며, 이것이 오래된 검색 기법에서 어떻게 발전했는지를 보여줍니다. DSI의 핵심은 사용자의 쿼리를 문서에 직접 매핑하는 transformer 기반의 신경망 아키텍처를 사용하여 검색 과정을 간소화하는 것입니다. 또한, 이 모델은 문서 다양성을 높이기 위해 Maximal Marginal Relevance (MMR)에서 영감을 받은 새로운 손실 함수(loss function)를 도입하여 훈련 중에도 관련성(relevance)과 다양성(diversity)을 함께 고려할 수 있습니다.

- **Technical Details**: DSI는 문서 집합에 대한 정보를 신경망의 파라미터로 인코딩하는 정보 검색의 새로운 패러다임입니다. 검색 쿼리에 대해 모델은 관련 문서의 ID를 예측하며, 검색 결과의 다양성을 높이기 위해 MMR에서 유래된 유사성을 고려한 손실 함수를 사용합니다. 이러한 모델은 훈련 중에 데이터의 다양성을 높이고, 결과적으로 관련성과 다양성을 모두 갖춘 문서를 검색할 수 있도록 설계되었습니다.

- **Performance Highlights**: 본 연구의 방법론은 NQ320K 및 MSMARCO 데이터셋을 통해 평가되었으며, 기존 DSI 기법과 비교했을 때 유의미한 성과를 보였습니다. 모델의 다양성을 증가시키는 동시에 검색 결과의 관련성에 미치는 영향이 적음을 입증하였고, 결과적으로 사용자는 보다 폭넓은 주제를 접할 수 있게 됩니다. 이러한 접근은 정보 검색 문제에서 중요성이 높은 서브 토픽 검색(sub-topic retrieval)과 같은 분야에서도 유용하게 사용될 수 있습니다.



### Classroom Simulacra: Building Contextual Student Generative Agents in Online Education for Learning Behavioral Simulation (https://arxiv.org/abs/2502.02780)
Comments:
          26 pages

- **What's New**: 이번 연구에서는 가상 학생과의 상호작용을 통해 교육의 질을 향상시키기 위한 새로운 접근법을 제안합니다. 기존의 방법들이 강의 자료의 조절 효과를 무시한 것에 반해, 연구팀은 세부적으로 주석이 달린 데이터셋의 부족과 긴 텍스트 데이터를 처리하는 모델의 한계를 극복하기 위한 두 가지 도전을 해결했습니다.

- **Technical Details**: 연구팀은 60명의 학생을 대상으로 6주간 교육 워크숍을 실시하여, 강의 자료와 상호작용하는 학생들의 학습 행동을 기록하는 맞춤형 온라인 교육 시스템을 구축했습니다. 또한, 이들은 TIR(Transferable Iterative Reflection) 모듈을 제안하여, 프롬프트 기반 및 파인튜닝 기반의 대형 언어 모델(LLMs)을 보강하여 학습 행동을 시뮬레이션합니다.

- **Performance Highlights**: TIR 모듈을 활용한 실험 결과, 대형 언어 모델은 전통적인 딥 러닝 모델보다 더 정확한 학생 시뮬레이션을 수행할 수 있음을 보였습니다. TIR 접근법은 학습 성과의 세부적 동태성과 학생 간의 상관관계를 더 잘 포착하여, 온라인 교육을 위한 '디지털 트윈' 구축의 길을 열었습니다.



### 3D Foundation AI Model for Generalizable Disease Detection in Head Computed Tomography (https://arxiv.org/abs/2502.02779)
Comments:
          Under Review Preprint

- **What's New**: 본 논문에서는 일반화 가능한 질병 탐지를 위한 머리 CT 스캔을 위한 Foundation Model인 FM-CT를 소개합니다. 이 모델은 레이블 없는 3D 머리 CT 스캔 36만 1,663개로 자가 감독 학습(self-supervised learning)을 통해 학습되었으며, 이는 고품질 레이블 부족 문제를 해결하는 데 도움을 줍니다. FM-CT는 기존의 2D 처리 대신 3D 구조를 활용하여 더욱 효율적으로 머리 CT 이미지를 분석합니다.

- **Technical Details**: FM-CT는 깊은 학습 모델로서, 자기 증류(self-distillation)와 마스킹 이미지 모델링(masked image modeling)의 두 가지 자가 감독 프레임워크를 채택하여 훈련되었습니다. 이 모델은 맞춤형 비전 변환기(Transformer)를 기반으로 하는 부피 인코더(volumetric encoder)를 학습하고, 다양한 프로토콜로부터의 머리 CT 스캔을 정규화하여 일관된 입력을 제공합니다. 10개의 다양한 질병 감지 작업에 대해 평가되었으며, 전반적인 성능이 향상되었습니다.

- **Performance Highlights**: FM-CT는 내부 데이터(NYU Langone)에서 기존 모델에 비해 16.07%의 향상을 보였으며, 외부 데이터(NYU Long Island, RSNA)에서도 각각 20.86%, 12.01%의 성능 향상을 달성했습니다. 이 연구는 자가 감독 학습이 의료 영상에서 매우 효과적임을 보여주며, FM-CT 모델이 머리 CT 기반 진단에서 더 널리 적용될 잠재력을 가지고 있음을 강조합니다. 실험 결과는 실제 임상 상황에서 중요한 영향을 미칠 수 있는 모델의 일반화 가능성과 적응성을 뒷받침합니다.



### Cross-Modality Embedding of Force and Language for Natural Human-Robot Communication (https://arxiv.org/abs/2502.02772)
Comments:
          Under review in RSS 2025

- **What's New**: 이 논문에서는 단어와 힘 프로필(force profile)의 크로스 모달리티 임베딩(cross-modality embedding) 방법을 제시하여 언어(verbal)와 햅틱(haptic) 통신의 시너지 있는 조정을 가능하게 합니다. 두 사람이 무거운 물체를 함께 운반할 때, 자연스러운 언어와 물리적 힘의 통합으로 효과적인 조정이 이루어집니다. 이 연구는 언어와 물리적 힘 프로필이 서로 다름에도 불구하고 통합된 잠재 공간(latent space) 내에서 연결될 수 있음을 보여줍니다.

- **Technical Details**: 연구에서는 힘 프로필과 언어의 크로스 모달리티 임베딩을 위한 프레임워크를 제시하며, 이는 물리적 힘 곡선과 자연어 설명 간의 전환을 가능하게 합니다. 데이터 수집 방법론을 10명의 참가자가 힘-언어 전환을 수행하는 방식으로 수립하였고, 이러한 방법은 로봇이 의도를 효과적으로 이해하는 데 기여합니다. 또한, 힘 프로필은 방향, 크기, 지속 시간을 포함하는 요소로 정의되며, 이를 통해 힘을 양적으로 해석하는 기본 원리를 도출합니다.

- **Performance Highlights**: 실험 결과는 제안된 프레임워크의 효과성과 이전 데이터에서의 일반화를 검증하였습니다. 언어와 힘이 서로 보완하고 통합할 수 있는 가능성을 확인할 수 있었으며, 인간과 로봇 간의 상호작용을 개선하기 위해 이 모델이 중요한 역할을 할 것으로 기대됩니다. 이를 통해 다양한 작업 환경에서 로봇의 제어와 상호작용 품질이 향상될 것으로 나타났습니다.



### Adaptive Voxel-Weighted Loss Using L1 Norms in Deep Neural Networks for Detection and Segmentation of Prostate Cancer Lesions in PET/CT Images (https://arxiv.org/abs/2502.02756)
Comments:
          29 pages, 7 figures, 1 table

- **What's New**: 이 연구에서는 L1 가중된 Dice Focal Loss (L1DFL)라는 새로운 손실 함수(loss function)를 제안합니다. 이 손실 함수는 분류 난이도에 따라 voxel의 가중치를 조정하여 전이성 전립선 암 병변을 자동으로 탐지하고 분할하는 데 목적을 두고 있습니다. 연구팀은 생화학적 재발 전이성 전립선 암으로 진단 받은 환자들의 PET/CT 스캔을 분석하였습니다.

- **Technical Details**: L1DFL은 L1 norm을 기반으로 하여 각 샘플의 분류 난이도와 데이터셋 내 유사 샘플의 출현 빈도를 고려한 동적 가중치 조정 전략을 채택합니다. 이 연구의 주요 기여는 새로운 L1DFL 손실 함수의 도입과 함께 Dice Loss 및 Dice Focal Loss 함수와의 성능 비교를 통해 거짓 양성 및 거짓 음성 비율에 대한 효율성을 평가한 것입니다.

- **Performance Highlights**: L1DFL은 테스트 세트에서 비교 손실 함수보다 최소 13% 높은 성능을 보였습니다. Dice Loss 및 Dice Focal Loss의 F1 점수는 L1DFL에 비해 각각 최소 6% 및 34% 낮았고, 이로 인해 L1DFL이 전이성 전립선 암 병변의 견고한 분할을 위한 가능성을 제시합니다. 또한 이 연구는 병변 특성의 변동성이 자동화된 전립선 암 탐지와 분할에 미치는 영향을 강조합니다.



### PatchPilot: A Stable and Cost-Efficient Agentic Patching Framework (https://arxiv.org/abs/2502.02747)
- **What's New**: 이번 연구에서 제안하는 PatchPilot은 기존의 패칭(پatching) 에이전트들이 제공하는 성능과 비용 효율성 사이의 균형을 맞추는 혁신적인 접근 방식입니다. PatchPilot은 인간 기반의 패칭 워크플로우를 제안하며, 5개의 주요 구성 요소인 재현(reproduction), 위치 확인(localization), 생성(generation), 검증(validation), 정제(refinement)로 구성되어 있습니다. 특히 정제 단계는 PatchPilot만의 독창적인 과정으로, 검증 피드백에 따라 현재 패치를 반복적으로 개선합니다.

- **Technical Details**: 기존 패칭 에이전트는 주로 세 가지 구성 요소로 나뉩니다: 위치 확인, 생성, 검증. 위치 확인은 문제를 일으키는 코드 스니펫을 식별하고, 생성은 패치 후보를 생성하며, 검증은 후보 중 최종 패치를 선택하는 역할을 합니다. PatchPilot은 여러 단계의 패칭 계획(patch planning)과 이를 따르는 패치 생성을 통해 더 깊은 사고와 문제 해결 방안을 제공합니다. 이는 기존의 방법보다 더욱 효과적인 패칭 솔루션을 제공합니다.

- **Performance Highlights**: PatchPilot은 SWE-Bench-Lite와 SWE-Bench-Verified 벤치마크에서 모든 최신 오픈소스 방법들보다 뛰어난 성능을 보여주었습니다. 또한, 상대적으로 낮은 비용(1달러 미만)으로 성능을 유지하면서도 안정성 면에서도 뛰어난 결과를 얻었습니다. 특히, PatchPilot은 OpenHands와 비교했을 때 탁월한 안정성을 입증하여, 현실 세계에서의 적용 가능성을 더욱 높였습니다.



### Vision-Language Model Dialog Games for Self-Improvemen (https://arxiv.org/abs/2502.02740)
- **What's New**: 본 논문은 VLM(Dialog Games)이라는 새로운 자가 개선 프레임워크를 제안합니다. 이 방법은 두 개의 에이전트가 이미지 식별 중심의 목표 지향적인 게임에서 자가 플레이를 통해 데이터를 생성하는 방식입니다. 성공적인 게임 상호작용을 필터링하여 고품질의 이미지-텍스트 데이터셋을 자동으로 구축할 수 있습니다.

- **Technical Details**: VLM Dialog Game은 'Describer'와 'Guesser'라는 두 개의 VLM 에이전트를 사용하여 구성됩니다. Describer는 타겟 이미지에 대한 질문에 답하며, Guesser는 타겟을 분별하기 위해 특정 질문을 던집니다. 이 과정에서 생성된 데이터는 VLM을 파인튜닝하는 데 사용되어 게임 성능을 향상시키고, 더 나아가 이미지 이해 능력도 개선됩니다.

- **Performance Highlights**: 실험 결과, VLM Dialog Game 데이터를 통해 VLM을 파인튜닝할 경우, 게임 성능뿐만 아니라 관련된 이미지 이해 벤치마크에서도 상당한 성과 향상이 있음을 보여줍니다. 특히, OpenImages 및 DOCCI 데이터셋 기반의 게임을 설계했을 때, 로봇 작업 성공 감지 능력이 크게 향상되는 것을 확인했습니다.



### Peri-LN: Revisiting Layer Normalization in the Transformer Architectur (https://arxiv.org/abs/2502.02732)
Comments:
          Preprint

- **What's New**: 이 논문에서는 Transformer 아키텍처에서의 layer normalization (LN) 전략이 대규모 훈련의 안정성 및 수렴 속도에 미치는 영향을 분석합니다. 특히, 최근의 Transformer 모델에서는 'Peri-LN'이라는 새로운 LN 배치 전략이 나타났으며, 이는 서브 레이어 전후에 LN을 적용하여 모델의 성능 향상을 가져오는 것으로 보입니다. Peri-LN은 이전에 널리 사용되던 Post-LN 및 Pre-LN보다 더 나은 성능을 발휘하며, 이는 많은 연구자들이 간과하고 있던 부분입니다.

- **Technical Details**: Peri-LN 전략은 각 모듈의 입력과 출력을 정규화하여 hidden-state 변화를 조절하는데 효과적입니다. 기존의 Post-LN 및 Pre-LN 전략과는 다르게, Peri-LN을 적용할 경우 'vanishing gradients'와 'massive activations' 문제를 효과적으로 완화할 수 있습니다. 이 논문은 Peri-LN의 동작 원리와 그 이점을 탐구하며, 그 결과는 대규모 Transformer 아키텍처에서 안정성 향상에 기여할 수 있음을 보입니다.

- **Performance Highlights**: 실험 결과, Peri-LN은 gradient의 흐름을 안정적으로 유지하면서 hidden states의 varaince 성장을 보다 균형 잡힌 방식으로 야기합니다. 이는 3.2B 파라미터를 가진 대규모 Transformer 모델에서 일관된 성능 향상을 보여줍니다. Peri-LN에 대한 실험적 증거는 대규모 모델 훈련의 안정성을 높이며 최적의 LN 배치를 이해하는 데 중요한 기여를 하고 있습니다.



### Parameter Tracking in Federated Learning with Adaptive Optimization (https://arxiv.org/abs/2502.02727)
- **What's New**: 본 논문에서는 Federated Learning (FL)의 데이터 이질성 문제를 해결하기 위한 새로운 접근법으로, 제안된 파라미터 추적(Parameter Tracking, PT) 프레임워크를 소개합니다. 기존의 Gradient Tracking (GT) 기법을 일반화하여 두 가지 최신 적응형 최적화 알고리즘인 FAdamET 및 FAdamGT를 제안하며, 이들은 Adam 기반의 FL에 PT를 통합합니다. 이 연구는 비볼록(non-convex) 상황에서도 두 알고리즘의 수렴성을 철저히 분석합니다.

- **Technical Details**: FAdamET 및 FAdamGT 알고리즘은 각 클라이언트가 그들의 로컬에서 수집된 첫 번째 정보와 서버의 집계 정보를 추적합니다. 이들은 제어 변수를 사용하여 글로벌 정보를 효과적으로 추적하며, 추가적인 튜닝 없이도 비독립적(non-i.i.d.) 데이터로 인한 모델 편향을 완화하는 데 도움을 줍니다. 또한 PT의 개념을 활용하여 로컬 업데이트 과정의 다양한 단계에서 다르게 작용하는 방식을 이해하는 데 중점을 둡니다.

- **Performance Highlights**: 실험 결과, 이 두 알고리즘은 다양한 데이터 이질성 수준에서 기존 방법보다 우수한 성능을 보여줍니다. 특히 통신 비용 및 계산 비용을 종합적으로 평가한 결과, 적응형 최적화를 통해 첫 번째 정보의 수정이 효과적임을 입증합니다. CNN 및 대규모 언어 모델(LLM)을 사용한 이미지 분류 및 시퀀스 분류 작업에서 두 알고리즘은 기존 기준선보다 일관되게 높은 성능을 기록하였습니다.



### Astromer 2 (https://arxiv.org/abs/2502.02717)
Comments:
          10 pages, 17 figures

- **What's New**: Astromer 2는 별빛 곡선( light curve) 분석을 위해 특별히 설계된 파운데이션 모델로, 이전의 Astromer 1 모델의 개선된 버전입니다. 이 모델은 150만 개의 단일 밴드 빛 곡선을 사용하여 자가 지도 학습(self-supervised learning) 과제를 통해 사전 훈련되었습니다. 모델의 성능 비교와 임베딩(embeddings) 품질 분석을 통해 Astromer 2가 Astromer 1에 비해 여러 가지 시나리오에서 뛰어난 성능을 보임을 강조합니다.

- **Technical Details**: Astromer 2는 차세대 빛 곡선 분석을 위해 임베딩을 생성하는 과정에서, MLP(다층 퍼셉트론) 분류기의 F1 스코어를 기준으로 성능을 측정합니다. 특히, 가중 샘플 임베딩(weighted per-sample embeddings)을 사용하여 모델의 주의(attention) 블록에서 중간 표현을 통합함으로써 성능 향상을 이룹니다. 이 모델은 아틀라스(ATLAS) 데이터셋에서 F1 스코어 기준으로 15%의 성능 개선을 보여 주목받고 있습니다.

- **Performance Highlights**: Astromer 2는 20, 100, 500 샘플 등 다양한 제한된 데이터셋에서도 Astromer 1에 비해 월등한 성과를 거두었습니다. 특히, 모델은 단일 모달리티(single-modality) 데이터만을 처리하여 복잡성을 줄이고 높은 효율성을 달성했습니다. 이러한 성능 개선은 적은 양의 레이블 데이터로도 보다 효율적이고 확장 가능한 빛 곡선 분석을 가능하게 합니다.



### An Analysis of LLM Fine-Tuning and Few-Shot Learning for Flaky Test Detection and Classification (https://arxiv.org/abs/2502.02715)
Comments:
          10 pages

- **What's New**: 이 논문은 flaky test의 탐지 및 분류를 위한 새로운 접근 방식인 FlakyXbert를 소개합니다. Large Language Models (LLMs)을 활용한 fine-tuning과 few-shot learning (FSL) 방식의 성능과 비용을 비교 분석하며, FSL이 작은 데이터 세트에서도 효과적으로 학습할 수 있음을 강조합니다. 이를 통해 자원이 제한된 조직에서의 flaky test 탐지 및 분류 방안을 제시합니다.

- **Technical Details**: 이 연구는 flaky test의 탐지 및 분류를 위한 두 가지 방법인 fine-tuning과 FSL을 비교합니다. FlakyXbert는 Siamese 네트워크 아키텍처를 사용하여 제한된 데이터로 효율적으로 훈련될 수 있습니다. 이 연구는 FlakyCat와 IDoFT라는 두 개의 기존 flaky test 데이터 세트를 사용하여 각각의 방법의 성능을 평가합니다.

- **Performance Highlights**: 연구 결과, fine-tuning 방식은 높은 정확성을 달성할 수 있지만 FSL은 비용 효율적인 접근 방식으로 경쟁력있는 정확성을 제공합니다. 특히 FoS는 큰 annotated 데이터 세트가 부족한 상황에서 효과적으로 작용하여, 제한된 역사적 데이터만 있는 프로젝트에도 유리합니다. 이는 각 조직의 필요와 자원에 따라 두 방식 중 어떤 것이 적합한지를 결정하는 데 중요한 통찰을 제공합니다.



### Developing multilingual speech synthesis system for Ojibwe, Mi'kmaq, and Malis (https://arxiv.org/abs/2502.02703)
- **What's New**: 이번 연구는 Ojibwe, Mi'kmaq, Maliseet 등 세 가지 북미 원주민 언어를 위한 경량화된 multilingual TTS 시스템을 소개합니다. 연구 결과, 세 가지 유사한 언어에서 훈련된 다국어 TTS 모델이 데이터가 부족할 때 단일 언어 모델보다 성능이 우수함을 보여주었습니다. 또한, Attention-free architectures는 self-attention 아키텍처와 비교하여 메모리 효율성을 높였습니다.

- **Technical Details**: 이 시스템은 Matcha-TTS를 기반으로 하여 조건부 flow matching 기술을 사용하고 있으며, TTS 모델은 텍스트 인코더, 지속 시간 예측기 및 flow matching 디코더로 구성됩니다. 저자는 단일 화자를 위한 전통적인 Matcha-TTS 모델을 다국어 음성 합성에 맞게 수정하여 고유한 화자 및 언어 임베딩을 추가했습니다. 또한, 효율적인 배포를 위해 self-attention 대신 Mamba2와 Hydra와 같은 attention-free 레이어를 탐구했습니다.

- **Performance Highlights**: 연구에서는 3개 언어에 대한 TTS 모델이 공동체 중심의 음성 녹음 프로세스를 통해 수집된 데이터를 기반으로 하여 높은 성능을 나타낸다고 강조합니다. 특히, Mamba2와 Hydra로 대체된 attention 모듈이 모든 언어에서 기대되는 성능을 유지하면서도 파라미터 수를 줄였음을 보여주었습니다. 이러한 접근 방식은 자원 부족 환경에서도 효과적인 언어 기술 개선을 가능하게 합니다.



### Practically Effective Adjustment Variable Selection in Causal Inferenc (https://arxiv.org/abs/2502.02701)
Comments:
          20 pages, 8 figures

- **What's New**: 이 논문은 인과 효과 추정에서 정확도를 저하하지 않도록 변수를 선택하는 기준과 알고리즘을 제안합니다. 특히, Directed Acyclic Graphs(DAGs)와 Completed Partially Directed Acyclic Graphs(CPDAGs)에서 적용할 수 있는 특정 단계를 설명하며, CPDAGs에서 인과 효과 계산 가능성에 대한 정리를 제시합니다. 이 연구는 기존 방법들이 다루지 않았던 다가변 카테고리 변수에 대한 분석을 포함합니다.

- **Technical Details**: 논문에서는 그래픽적으로 인과 관계를 나타내기 위한 노드와 엣지 모델을 소개합니다. 변수 간의 인과적 관계는 DAG를 통해 표현되며, 알고리즘은 이러한 구조를 활용해 조정 변수를 선택하는 과정을 포함합니다. 이론적으로, 제시된 정리는 CPDAGs에서의 개입 계산 조건을 다루며, 모든 필수 교란 변수가 관찰된다고 가정합니다.

- **Performance Highlights**: 제안된 방법은 기존 데이터 및 인공 데이터를 사용하여 그 유용성을 입증합니다. 다양한 변수 집합에 대해 인과 효과 추정의 정확도를 높이는 데 기여하며, 다가변 카테고리 변수의 관계를 비선형으로 다룰 수 있는 가능성을 제시합니다. 이 알고리즘은 복잡한 시스템에서 인과적 관계를 이해하는 데 중요한 기초를 제공합니다.



### Controllable Video Generation with Provable Disentanglemen (https://arxiv.org/abs/2502.02690)
- **What's New**: 최근까지 발전된 기술에도 불구하고, 영상 생성의 정확하고 독립적인 제어는 여전히 큰 도전 과제가 남아있습니다. 본 연구에서는 Controllable Video Generative Adversarial Networks (CoVoGAN)을 제안하며, 이는 영상 내 개념을 분리하여 각 요소에 대한 효율적이고 독립적인 제어를 가능하게 합니다. CoVoGAN은 최소 변화 원칙(minimal change principle)과 충분한 변화 속성(sufficient change property)을 활용하여 정적 및 동적 잠재 변수를 분리합니다.

- **Technical Details**: CoVoGAN의 핵심은 Temporal Transition Module을 통해 동적 및 정적 요소를 구분하는 것입니다. 이 방법은 잠재 동적 변수의 차원 수를 최소화하고 시간적 조건 독립성을 부여하여, 동작과 정체성을 독립적으로 제어할 수 있게 합니다. 이러한 이론적 뒷받침과 함께 다양한 비디오 생성 기준을 통해 제안한 방식의 유효성을 실증합니다.

- **Performance Highlights**: Multiples 데이터 세트를 통해 CoVoGAN의 효과를 평가한 결과, 다른 GAN 기반의 비디오 생성 모델에 비해 생성 품질과 제어 가능성에서 현저히 개선된 성능을 보였습니다. 또한, 훈련 과정에서의 강인성과 추론 속도에서도 기존 기법들보다 우수한 결과를 나타내었습니다. 본 연구는 또한 비디오 생성 분야에서의 식별 가능성 정리를 제시하며, 향후 연구 방향을 제시합니다.



### Streaming Speaker Change Detection and Gender Classification for Transducer-Based Multi-Talker Speech Translation (https://arxiv.org/abs/2502.02683)
- **What's New**: 이 논문은 스트리밍 다중 화자 음성 번역을 위한 새로운 방법론을 제안합니다. 저자들은 스피커 임베딩(speaker embeddings)을 활용하여 음성 변환 모델의 성능을 개선하고, 발화자 변화 감지 및 성별 분류를 통합하여 효율적인 실시간 번역을 목표로 합니다. 이러한 접근은 성별 정보가 음성 합성(text-to-speech, TTS) 시스템에서 화자 프로파일 선택에 도움을 줄 수 있음을 강조합니다.

- **Technical Details**: 제안된 접근 방법은 트랜스듀서(transducer) 기반의 스트리밍 음성 번역 모델을 기반으로 하며, 여기에는 세 가지 주요 구성 요소가 포함됩니다: 인코더(encoder), 예측 네트워크(prediction network), 조합 네트워크(joint network). t-vector 방법을 통해 다중 화자 음성 인식 시 스피커 임베딩을 생성하고, 이 임베딩을 사용하여 발화자 변화 및 성별 분류를 수행하는 방식을 설명합니다. 이는 다양한 언어 쌍에 대해 평가되어 높은 정확도를 보여줍니다.

- **Performance Highlights**: 실험 결과, 저자들이 제안한 방법은 발화자 변화 감지와 성별 분류 모두에서 높은 정확성을 달성했다고 보고하였습니다. 특히, 음성에서 실시간으로 정보를 처리하는 스트리밍 시나리오에서 효과적인 성능을 나타냈습니다. 이는 실시간 통신 환경에서 화자 변화 민감도가 높은 zero-shot TTS 모델에서도 적용 가능성을 시사합니다.



### MedRAX: Medical Reasoning Agent for Chest X-ray (https://arxiv.org/abs/2502.02673)
Comments:
          11 pages, 4 figures, 2 tables

- **What's New**: 이번 연구에서는 CXR 해석을 위한 최초의 통합형 AI 에이전트인 MedRAX를 제시합니다. MedRAX는 다양한 CXR 분석 도구와 대규모 다중 모달 언어 모델을 통합하여 복잡한 의료 쿼리에 대해 추가 훈련 없이 역동적으로 활용할 수 있는 시스템입니다. 또한, 2,500개의 복잡한 의료 쿼리를 포함한 ChestAgentBench 벤치마크를 소개하여 MedRAX의 성능을 정밀하게 평가합니다.

- **Technical Details**: MedRAX는 경량 분류기부터 대규모 다중 모달 모델에 이르는 이질적인 기계 학습 모델을 통합하여 복잡한 의료 쿼리를 효과적으로 해결합니다. 이 시스템은 ReAct (Reasoning and Acting) 루프를 채택하여 사용자 쿼리를 분석하고, 필요한 작업을 결정하며, 관련 도구를 실행하는 과정을 반복합니다. 따라서 의료 이미지를 통한 다중 단계 추론이 가능하며, 각 과정에서 단기 기억을 통해 사용자와의 상호작용을 지원합니다.

- **Performance Highlights**: MedRAX는 일반 용도 모델 및 생물 의학 전문 모델과 비교하여 우수한 성능을 보여주었습니다. 실험 결과 복잡한 추론 작업에서 상당한 개선을 이루었으며, 투명한 워크플로우를 유지하면서도 임상 AI 시스템의 신뢰성을 보장합니다. 이러한 성과는 MedRAX의 자동화된 CXR 해석 시스템의 실용적 배치를 한 걸음 더 앞당기는 혁신적인 발전으로 간주됩니다.



### On Teacher Hacking in Language Model Distillation (https://arxiv.org/abs/2502.02671)
- **What's New**: 이 논문은 언어 모델(LM)의 포스트 트레이닝에서 지식 증류(knowledge distillation)의 새로운 현상인 teacher hacking을 조사합니다. Teacher hacking은 학생 모델이 교사 모델을 모방하는 과정에서 발생할 수 있는 부정확성을 이용하는 현상입니다. 저자들은 이러한 현상이 실제로 발생하는지, 그 기준은 무엇인지, 그리고 이를 완화하기 위한 전략을 연구하는 실험 세트를 제안합니다.

- **Technical Details**: 연구팀은 오라클 모델을 기본으로 하는 제어된 실험 세트를 제안했습니다. 이 실험 세트에서는 학생 모델과 오라클 모델 간의 거리와 학생 모델과 교사 모델 간 거리로 각각 '골든 메트릭'과 '프록시 메트릭'을 정의합니다. 학생 모델의 효과적인 훈련을 위해 사용된 데이터의 다양성이 teacher hacking 방지를 위한 주요 요소로 확인되었습니다.

- **Performance Highlights**: 실험 결과, 고정된 오프라인 데이터셋을 사용할 경우 teacher hacking이 발생함을 확인하였으며, 최적화 과정이 다항 수렴 법칙에서 벗어날 때 이를 감지할 수 있음을 밝혔습니다. 반면, 온라인 데이터 생성 기법을 통해 teacher hacking을 효과적으로 완화할 수 있으며, 데이터의 다양성을 활용하는 것이 중요한 요소로 발견되었습니다.



### A Training-Free Length Extrapolation Approach for LLMs: Greedy Attention Logit Interpolation (GALI) (https://arxiv.org/abs/2502.02659)
Comments:
          9 pages, under review in the conference

- **What's New**: 이번 논문에서는 Greedy Attention Logit Interpolation (GALI)이라는 새로운 training-free 길이 외삽(extrapolation) 방법을 제안합니다. GALI는 사전 훈련된 위치 간격을 최대한 활용하면서 attention logit outlier 문제를 피하는 데 중점을 두며, LLMs의 positional O.O.D. 문제를 해결하는 데 기여합니다. GALI는 기존의 state-of-the-art training-free 방법들보다 일관되게 우수한 성능을 나타내며, 단기 맥락 작업에서도 더욱 향상된 결과를 제공합니다.

- **Technical Details**: GALI는 두 가지 주요 목표를 가지고 있습니다. 첫째, 훈련 맥락(window) 내의 고유한 위치 ID를 유지하여 attention 계산의 방해 요소를 최소화하고, 둘째, attention logit interpolation 전략을 통해 outlier 문제를 완화하는 것입니다. GALI는 초기화 단계에서 훈련된 위치 간격을 최적화하고, 생성 단계에서 새로운 토큰에 대해 동적으로 보간된 위치 ID를 생성하여 위치 변동성에 대한 민감성을 높입니다.

- **Performance Highlights**: GALI는 세 가지 벤치마크를 통해 평가되었으며, 다양한 실험 결과에서 기존의 최고 수준의 training-free 방법들을 일관되게 초월하는 결과를 보여주었습니다. 특히, 모델이 위치 간격을 해석하는 데 있어 불일치성을 바탕으로 한 성능 향상 전략을 발견하였으며, 짧은 맥락 작업에서도 더 좋은 결과를 도출할 수 있음을 확인했습니다. GALI는 LLMs의 긴 텍스트 이해에 있어 중요한 진전을 이루며, 기존의 긴 맥락 프레임워크와 원활하게 통합됩니다.



### ParetoQ: Scaling Laws in Extremely Low-bit LLM Quantization (https://arxiv.org/abs/2502.02631)
- **What's New**: 이 논문에서는 quantization(양자화) 모델의 크기와 정확도 간의 최적의 trade-off(절충점)을 찾기 위한 통합된 프레임워크인 ParetoQ를 제안합니다. 이는 1비트, 1.58비트, 2비트, 3비트, 4비트 양자화 설정 간의 비교를 보다 rigorously(엄격하게) 수행할 수 있도록 합니다. 또한, 2비트와 3비트 간의 학습 전환을 강조하며 이는 정확한 모델 성능에 중요한 역할을 합니다.

- **Technical Details**: ParetoQ는 파라미터 수를 최소화하면서도 성능을 극대화할 수 있는 모델링 기법입니다. 실험 결과, 3비트 및 그 이상의 모델은 원래 pre-trained distribution(사전 훈련 분포)과 가까운 성능을 유지하는 반면, 2비트 이하의 네트워크는 표현 방식이 급격히 변화합니다. 또한, ParetoQ는 이전의 특정 비트 폭에 맞춘 방법들보다 우수한 성능을 보여줍니다.

- **Performance Highlights**: 작은 파라미터 수에도 불구하고, ParetoQ의 ternary 600M-parameter 모델은 기존의 SoTA(최신 기술 상태)인 3B-parameter 모델을 넘는 정확도를 기록했습니다. 다양한 실험을 통해 2비트와 3비트 양자화가 사이즈와 정확도 간의 trade-off에서 우수한 성능을 보여주며, 4비트 및 binary quantization 글리 제너럴 단위 성능이 하락하는 경향을 보였습니다.



### scBIT: Integrating Single-cell Transcriptomic Data into fMRI-based Prediction for Alzheimer's Disease Diagnosis (https://arxiv.org/abs/2502.02630)
Comments:
          31 pages, 5 figures

- **What's New**: 이번 연구에서는 fMRI(기능적 자기공명영상)와 single-nucleus RNA (snRNA)를 결합하여 알츠하이머병(AD) 예측을 향상시키는 새로운 방법인 scBIT를 소개합니다. 이 방법은 서로 다른 데이터 소스를 통합하여 AD 연구의 해석 가능성을 높이고 예측 모델의 성능을 개선합니다. 기존의 접근 방식과 달리 각 개인 간의 데이터의 인구 통계학적 및 유전적 유사성을 활용하여 강력한 크로스 모달(Cross-modal) 학습을 가능하게 합니다.

- **Technical Details**: scBIT는 snRNA를 보조 모달리티로 활용하여 fMRI 기반 예측 모델을 크게 향상시킵니다. 이 방법은 샘플링 전략(sampling strategy)을 사용하여 snRNA 데이터를 세포 유형별 유전자 네트워크(cell-type-specific gene networks)로 분할하며, 자기 설명 가능한 그래프 신경망(self-explainable graph neural network)을 통해 중요한 하위 그래프를 추출합니다. 이 과정에서 fMRI 데이터와 snRNA 데이터의 매칭을 위한 다양한 척도를 사용하여 각기 다른 개인의 데이터를 연결합니다.

- **Performance Highlights**: scBIT의 실험 결과, snRNA 데이터를 도입함으로써 이 모델의 분류 정확도가 크게 향상되었습니다. 이중 분류(binary classification)에서 3.39%의 정확도 향상과 다섯 분류(five-class classification)에서는 26.59%의 향상을 보여주었습니다. 이러한 성과는 알츠하이머병 연구에서 바이오마커 발견(biomarker discovery)에 새로운 통찰력을 제공하는 중요한 발전입니다.



### Graph Structure Learning for Tumor Microenvironment with Cell Type Annotation from non-spatial scRNA-seq data (https://arxiv.org/abs/2502.02629)
Comments:
          29 pages, 6 figures

- **What's New**: 이번 논문은 종양 미세환경(TME) 내 세포 이질성을 탐구하기 위해 단일 세포 RNA 시퀀싱(scRNA-seq)을 활용하는 중요성을 강조합니다. 현재의 scRNA-seq 접근 방식은 공간적 맥락이 부족하고 리간드-수용체 상호작용(ligand-receptor interactions, LRI)의 불완전한 데이터세트에 의존함으로써 세포 유형 주석(cell type annotation)과 세포 간 통신(cell-cell communication, CCC) 추론에 한계를 가지고 있습니다. 이 연구는 이러한 문제를 해결하기 위해 그래프 신경망(graph neural network, GNN)을 사용한 새로운 모델을 제시합니다.

- **Technical Details**: 제안된 모델인 scGSL은 3가지 암 유형(백혈병, 유방 침습성 암, 대장암)에서 유래한 49,020개의 세포로 구성된 데이터를 분석하였습니다. 이 모델은 세포 유형 예측(cell type prediction)과 세포 상호작용 분석(cell interaction analysis)을 향상시키는 데 중점을 두고 있습니다. scGSL 모델은 평균 정확도(accuracy) 84.83%, 정밀도(precision) 86.23%, 재현율(recall) 81.51%, F1 점수(F1 score) 80.92%를 달성하며 기존 방법에 비해 성능이 현저히 개선된 결과를 보여주었습니다.

- **Performance Highlights**: scGSL 모델은 TME 내에서 생물학적으로 의미 있는 유전자 상호작용을 비지도 방식으로 강력하게 식별할 수 있으며, 다양한 암에서 주요 유전자 쌍의 발현 차이를 통해 이를 검증하였습니다. 이는 현재의 방법들이 일반적으로 나타내는 성능보다 유의미한 발전을 나타냅니다. 본 논문에서 사용된 소스 코드와 데이터는 제공된 URL을 통해 확인할 수 있습니다.



### e-SimFT: Alignment of Generative Models with Simulation Feedback for Pareto-Front Design Exploration (https://arxiv.org/abs/2502.02628)
- **What's New**: 이 논문에서는 깊은 생성 모델이 공학 설계 문제를 해결하는데 적용할 수 있는 새로운 프레임워크인 e-SimFT를 소개합니다. 이 프레임워크는 대형 언어 모델(LLM)에서 개발된 선호 정렬(preference alignment) 방법을 적용하여, 시뮬레이션 데이터를 기반으로 한 세밀한 조정(fine-tuning)을 통해 설계 요구 사항에 맞는 솔루션을 생성합니다. 또한, epsilon-sampling이라는 새로운 샘플링 방법을 제안하여 고품질의 Pareto front를 구성하는 데 기여합니다.

- **Technical Details**: e-SimFT는 시뮬레이터를 사용하여 공학 설계에 대한 피드백을 제공함으로써, 생성 모델을 조정하는 접근법을 제시합니다. 이 연구는 특히 필요한 설계 요구 사항에 대해 어떤 방법이 더 효과적인지에 대한 통찰을 제공합니다. epsilon-sampling 방법은 고전적 최적화 알고리즘에서 사용되는 epsilon-constraint 방법에서 영감을 받아 설계되었습니다.

- **Performance Highlights**: e-SimFT는 기존의 다목적 정렬 방법들과 비교하여 더욱 뛰어난 성능을 보여줍니다. 저자들은 여러 기초선(baseline)과의 비교를 통해 e-SimFT의 전반적인 우수성을 입증하였으며, 이를 통해 설계 요구 사항에 대한 특정 성능 개선을 보여주겠다는 목표를 달성했습니다.



### Sample Complexity of Bias Detection with Subsampled Point-to-Subspace Distances (https://arxiv.org/abs/2502.02623)
- **What's New**: 이번 연구는 표본 복잡성(sample complexity) 개념을 Bias(편향) 추정에 적용하는 새로운 접근법을 제시합니다. 특히, 컴퓨터 시스템의 편향을 탐지하는 데 필요한 시간의 하한을 제시하고, 이로 인해 실제 편향 탐지의 복잡성을 다항식(polynomial) 수준으로 낮출 수 있는 방안을 모색합니다. 이를 통해 고차원 데이터 처리 시 발생하는 ‘차원의 저주(curse of dimensionality)’ 문제를 해결할 수 있는 가능성을 보여줍니다.

- **Technical Details**: Bias 탐지는 두 샘플 집합의 분포 간 거리를 추정하는 문제로 모델링됩니다. Wasserstein 거리(Wasserstein distance) 및 최대 평균 불일치(Maximum Mean Discrepancy, MMD)와 같은 다양한 거리 측정을 이용하여 편향 추정의 표본 복잡성을 분석합니다. 또한, 편향 탐지 문제를 측정 공간의 점 대 부분공간(point-to-subspace) 문제로 재구성하였고, 이를 통해 효율적으로 서브샘플링(subsampling)을 적용할 수 있음을 증명합니다.

- **Performance Highlights**: 본 연구 결과는 기존의 잘 알려진 케이스들에 대한 테스트를 통해 뒷받침됩니다. 특히, 확률적으로 약정확(pac)한 결과가 도출되어, 편향 탐지의 신뢰성을 높이는데 기여할 것으로 예상됩니다. 이는 다수의 보호 속성에 대한 서브그룹(subgroup) 검증의 필요성을 충족하며, 알고리즘의 효율성을 개선할 수 있는 실질적 방법을 제공합니다.



### Deep Learning-Based Facial Expression Recognition for the Elderly: A Systematic Review (https://arxiv.org/abs/2502.02618)
- **What's New**: 이 연구는 노인을 위한 딥러닝 기반의 얼굴 표정 인식(FER) 시스템에 대한 체계적 검토를 제공하며, 노인 인구에 적합한 기술의 필요성을 강조합니다. 특히, 노인 전용 데이터셋의 부족과 불균형한 클래스 배분, 나이에 따른 얼굴 표정 차이의 영향을 다루고 있습니다. 연구 결과, 합성곱 신경망(CNN)이 FER에서 여전히 우위를 점하고 있으며, 제한된 자원 환경에서 더 가벼운 버전의 중요성이 강조됩니다.

- **Technical Details**: 이 연구는 31개의 연구를 분석하여 딥러닝의 발전과 그에 따른 얼굴 표정 인식 시스템의 발전 상황을 조명합니다. 연구자는 심층 학습(Deep Learning) 기술, 특히 CNN이 노인 환자와의 다양한 상호작용에서 정서적 상태를 인식하는 데 어떻게 사용되는지 설명합니다. 또한, 설명 가능 인공지능(XAI)의 필요성을 강조하며, 이 기술이 FER 시스템의 투명성과 신뢰성을 높일 수 있는 방안을 제시합니다.

- **Performance Highlights**: FER 시스템은 노인 돌봄에서 중요한 역할을 할 수 있으며, 얼굴 표정을 통한 감정 모니터링 및 개별화된 치료가 가능해집니다. 그러나 노인 인구에 특화된 FER 연구가 부족하여 정서 인식의 정확성을 높이기 위한 데이터셋의 다양성과 연구 접근법의 필요성이 강하게 제기되었습니다. 이 연구는 노인 돌봄을 위한 FER 시스템의 발전 방향에 대한 의의와 함께 향후 연구에 대한 권고 사항을 포함합니다.



### PolarQuant: Quantizing KV Caches with Polar Transformation (https://arxiv.org/abs/2502.02617)
- **What's New**: 본 논문은 PolarQuant라는 새로운 정량화(quantization) 방법을 소개합니다. 이 방법은 Key-Value(KV) 캐시의 메모리 사용량을 줄이기 위해 랜덤 프리컨디셔닝(random preconditioning)과 극 좌표 변환(polar transformation)을 활용합니다. 기존의 정량화 기법에서 요구되는 명시적 정규화(normalization) 단계를 제거함으로써 메모리 저장 공간을 대폭 절약할 수 있습니다.

- **Technical Details**: PolarQuant는 KV 임베딩을 극 좌표계로 변환하기 위해 효율적인 재귀적 알고리즘을 사용합니다. 랜덤 프리컨디셔닝을 통해 극 좌표 표현에서 각(angle)의 분포를 조밀하고 안정성 있게 유지할 수 있으며, 이 정보를 이용해 최적화된 정량화 코드북을 생성합니다. 이 접근법은 정량화 오류를 최소화하고, KV 벡터에 대한 효율적인 표현을 가능하게 합니다.

- **Performance Highlights**: 긴 컨텍스트(long-context) 작업에서, PolarQuant는 기존의 최첨단 방법들과 비교하여 최상의 품질 점수를 달성하면서 KV 캐시를 4.2배 이상 압축할 수 있음을 보여줍니다. 이 성능 향상은 메모리 사용량을 줄이는 동시에 모델의 정확성을 유지하는 데 큰 기여를 합니다. 따라서 PolarQuant는 큰 언어 모델의 실용적인 적용에서 상당한 이점을 제공합니다.



### Reconstructing 3D Flow from 2D Data with Diffusion Transformer (https://arxiv.org/abs/2502.02593)
- **What's New**: 본 연구는 2D 유동 데이터에서 3D 유동장을 재구성하는 새로운 Diffusion Transformer 기반 방법을 제안합니다. 2D 플레인의 위치 정보를 모델에 포함함으로써 다양한 2D 슬라이스 조합으로부터 3D 유동장을 복원할 수 있는 가능성을 제시하며, 이는 유동 데이터를 효과적으로 활용하고 유연성을 향상시킵니다. 또한, 계산 성능을 저하시키지 않으면서도 높은 차원에서의 계산 비용을 줄이기 위해 글로벌 주의(global attention) 대신 윈도우 및 플레인 주의(window and plane attention)를 도입합니다.

- **Technical Details**: 해당 연구에서는 Transformer 기반의 Diffusion 모델을 적용하여 2D 플레인에서 3D 유동장을 재구성합니다. 2D 플레인 위치 임베딩을 활용하여 공간 정보를 캡처하고, 높은 차원 유동의 계산 복잡성을 처리하기 위해 윈도우와 플레인 주의 메커니즘을 통합합니다. 이러한 설계로 인해 2D 유동 데이터로부터 3D 유동장을 효과적으로 복원할 수 있게 되었으며, 모델은 임베드된 2D 플레인 정보를 이용하여 임의의 조합으로 3D 유동 정보를 복구할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 2D 유동 데이터로부터 3D 유동장을 효율적이고 정확하게 재구성할 수 있음을 보여주었습니다. 특히, 모델은 유동장의 실제 분포를 포착하고 현실적인 샘플을 생성하는 성능을 발휘하였습니다. 이러한 접근 방식은 PIV 및 CFD에서의 비용 절감과 더불어 여러 적용 시나리오를 확장할 수 있는 가능성을 제공합니다.



### HadamRNN: Binary and Sparse Ternary Orthogonal RNNs (https://arxiv.org/abs/2502.00047)
- **What's New**: 이 논문에서는 이진(binary) 및 희소 삼진(sparse ternary) 가중치를 사용하는 새로운 방법을 통해 vanilla RNN의 가중치를 효과적으로 학습할 수 있는 방법을 제안합니다. 특히 Hadamard 행렬의 특성을 활용하여 특정 클래스의 이진 및 희소 삼진 vanilla RNN을 생성하는 방법을 제시합니다. 이 방법은 향후 엣지 장치가 요구하는 경량화된 신경망 구축에 기여할 것으로 기대됩니다.

- **Technical Details**: 제안된 방법은 양자화된(gquantized) 가중치를 사용하여 RNN을 훈련하는데, 이는 주기적 가중치의 민감성을 해결하기 위해 Hadamard 행렬을 기반으로 합니다. 이 논문에서는 새로운 ORNN(Orthogonal RNN) 아키텍처를 통해 희소 삼진 및 이진 가중치를 가지는 RNN 모델인 HadamRNN와 lock-HadamRNN을 제안합니다. 이 모델들은 특수한 구조로 설계되어 학습 및 일반화 성능에서 이점을 제공합니다.

- **Performance Highlights**: 이 RNN 모델들은 copy task, permuted MNIST, sequential MNIST 및 IMDB 데이터셋과 같은 다양한 벤치마크에서 평가되었습니다. 이진화 또는 희소 삼진화에도 불구하고, 제안된 ORNN 모델들은 최신의 풀-정밀(full-precision) 모델과 비교할 때 경쟁력 있는 성능을 유지합니다. 특히, 1000 타임스텝 이상의 copy task를 성공적으로 수행한 첫 번째 이진 RNN 모델로 주목받고 있습니다.




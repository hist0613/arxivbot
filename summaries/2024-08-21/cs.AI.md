New uploads on arXiv(cs.CL)

### MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding (https://arxiv.org/abs/2408.11049)
- **What's New**: 이 논문에서는 Speculative Decoding(SD) 기술을 활용하여 고속 배치 처리에서 대기 시간(latency)과 처리량(throughput)을 동시에 개선할 수 있음을 보여줍니다. 기존의 SD에 대한 일반적인 신념은 큰 배치 사이즈에서는 효율적이지 않다는 것이었으나, 본 연구는 오히려 효과적일 수 있음을 보여줍니다.

- **Technical Details**: MagicDec는 배치 사이즈와 시퀀스 길이에 따른 병목 현상을 파악하고, 이를 통해 Speculative Decoding을 더욱 효과적으로 배치 처리에 활용합니다. 또한 Sparse KV cache를 사용하는 드래프트 모델을 사용하여 KV 병목 현상을 해결합니다. 이러한 방법으로 Medium-to-Long 시퀀스에서 LLM의 디코딩 성능 개선을 실험적으로 확인하였습니다.

- **Performance Highlights**: 8개의 A100 GPU에서 LLaMA-2-7B-32K에 대해 2배 속도 향상, LLaMA-3.1-8B에 대해서는 1.84배 속도 향상을 달성했습니다. 이러한 결과는 Speculative Decoding을 통한 처리 성능 향상의 가능성을 보여줍니다.



### Inside the Black Box: Detecting Data Leakage in Pre-trained Language Encoders (https://arxiv.org/abs/2408.11046)
Comments:
          ECAI24

- **What's New**: 이 논문은 Pre-trained Language Encoders (PLEs)의 개인정보 유출 및 저작권 침해와 관련된 위험을 체계적으로 탐구한 첫 연구이다. 특히, 기존 문헌에서 간과된 'membership leakage'를 다루며, 사전 훈련 데이터의 노출 위험성을 분석한다.

- **Technical Details**: PLEs는 BERT와 같은 대형 텍스트 코퍼스에서 훈련되어 자연어 처리(NLP)에서 폭넓게 사용된다. 연구는 네 가지 유형의 PLE 아키텍처, 세 가지 대표적인 다운스트림 작업, 다섯 개의 벤치마크 데이터 세트에 대해 실험을 수행하였으며, 블랙박스 접근 방식을 통한 다운스트림 모델 출력만으로도 데이터의 회원 정보 유출이 가능하다는 점을 밝혀냈다.

- **Performance Highlights**: 이 연구는 블랙박스 환경에서도 데이터 유출이 발생할 수 있음을 최초로 발견하였고, PLE 아키텍처나 다운스트림 작업의 유형에 관계없이 정보의 유출이 지속적으로 발생함을 보여주었다. 이는 PLE 사용 시 예상보다 더 심각한 개인정보 위험을 시사한다.



### Scaling Law with Learning Rate Annealing (https://arxiv.org/abs/2408.11029)
Comments:
          25 pages, 23 figures

- **What's New**: 본 연구에서는 신경 언어 모델의 cross-entropy (크로스 엔트로피) 손실 곡선이 학습률 (Learning Rate, LR) 퍼지 동안 스케일링 법칙에 기초한다는 점을 발견함. 제안된 공식은 손실 곡선을 각 학습 단계에서 나타낼 수 있도록 함.

- **Technical Details**: 제안된 손실 모델은 다음과 같은 두 가지 주요 요소를 포함함: 1) 전방 영역 (forward area) S1과 2) LR 퍼지 영역 (LR annealing area) S2. 이 식은 LR 스케쥴러와 관계없이 어떤 학습률 스케쥴에서든 훈련 단계에서 손실을 예측할 수 있게 해줌.

- **Performance Highlights**: 새로운 접근법은 치킨밀라 스케일링 법칙 (chinchilla scaling law) 대비 1% 미만의 계산 비용으로 정확한 손실 예측을 가능하게 하며, 대규모 언어 모델 개발에서 스케일링 법칙 피팅과 예측을 대폭 민주화함.



### Athena: Safe Autonomous Agents with Verbal Contrastive Learning (https://arxiv.org/abs/2408.11021)
Comments:
          9 pages, 2 figures, 4 tables

- **What's New**: 대형 언어 모델(LLM)을 이용한 자율 에이전트의 안전성을 극대화하기 위한 Athena 프레임워크를 소개합니다. 이 프레임워크는 언어 대조 학습(Verbal Contrastive Learning) 개념을 활용하여 과거의 안전 및 위험한 경로를 예제로 사용하여 에이전트가 안전성을 확보하며 작업을 수행하도록 도움을 줍니다.

- **Technical Details**: Athena 프레임워크는 Actor, Critic, Emulator의 세 가지 LLM 에이전트로 구성되며, 이들은 사용자로부터 제공된 요구 사항에 따라 상호작용하여 작업을 완료합니다. Critic 에이전트는 Actor의 사고 및 행동을 심사하여 안전성을 개선하고, 언어 대조 학습을 통해 Actor에게 과거 경로를 기반으로 안전한 행동을 유도합니다. 또한, 8080개의 도구 키트와 180개의 시나리오로 구성된 안전 평가 벤치마크를 구축하였습니다.

- **Performance Highlights**: 실험적 평가에서 언어 대조 학습과 상호작용 수준의 비평이 LLM 기반의 에이전트의 안전성을 크게 향상시킨다는 결과를 얻었습니다. 특히, GPT-4-Turbo가 가장 안정적인 후보로 확인되었습니다.



### While GitHub Copilot Excels at Coding, Does It Ensure Responsible Output? (https://arxiv.org/abs/2408.11006)
- **What's New**: 대규모 언어 모델(LLMs)의 급속한 발전이 코드 완성 능력을 크게 향상시켰으며, LLM 기반 코드 완성 도구(LCCTs)가 새로운 세대로 등장하였습니다. LCCT는 일반 목적의 LLM과는 다른 고유한 워크플로우를 가지고 있으며, 여러 정보 원천을 통합하여 입력을 처리하고 자연어 상호작용보다 코드 제안을 우선시함으로써 독특한 보안 문제를 야기합니다.

- **Technical Details**: 이 논문은 LCCT의 특성을 활용해 두 가지 핵심 보안 위험인 탈옥 공격(jailbreaking)과 훈련 데이터 추출 공격(training data extraction attacks) 방법론을 개발합니다. 우리의 실험 결과는 GitHub Copilot 및 Amazon Q에 대해 각기 99.4%와 46.3%의 탈옥 공격 성공률(ASR)을 기록했으며, GitHub Copilot에서는 54개의 실제 이메일 주소와 314개의 물리적 주소를 추출하는 데 성공했습니다. 또한, 우리의 연구에서는 이러한 코드 기반 공격 방법이 일반 목적의 LLM에도 효과적임을 보여주고 있습니다.

- **Performance Highlights**: LCCT의 고유한 워크플로우는 새로운 보안 문제를 도입하며, 코드 기반 공격은 LCCT와 일반 LLM 모두에 심각한 위협이 됩니다. 연구 결과, 공격 방법의 효과는 모델의 복잡성에 따라 다르게 나타나고 있으며, 개인 정보 유출의 위험을 강조합니다. 따라서, LCCT의 보안 프레임워크 강화가 시급하게 요구됩니다.



### Disentangling segmental and prosodic factors to non-native speech comprehensibility (https://arxiv.org/abs/2408.10997)
- **What's New**: 본 논문은 현재의 억양 변환(Accent Conversion, AC) 시스템이 비원어민의 억양 특성에서 분절(segmental) 및 운율(prosodic) 특성을 분리하지 못하고 있다는 점을 지적한다. 저자들은 이를 해결하기 위해 음성 품질을 억양과 분리하여 이 두 가지 특성을 독립적으로 조작할 수 있는 AC 시스템을 제안한다.

- **Technical Details**: 제안된 시스템은 (1) 출발 발화(source utterance)에서의 분절 특성, (2) 목표 발화(target utterance)에서의 음성 특성, (3) 참조 발화(reference utterance)에서의 운율을 결합하여 새로운 발화를 생성할 수 있다. 음향 임베딩(acoustic embeddings)의 벡터 양자화(vector quantization)와 반복된 코드워드 제거를 통해 운율을 전송하고 음성 유사성을 향상시키는 방법을 사용한다.

- **Performance Highlights**: 인지 listening 테스트 결과, 분절 특성이 운율보다 비원어민 발화의 가독성에 더 큰 영향을 미친다는 것을 발견하였다. 제안된 AC 시스템은 비원어민의 발화의 상대적 이해력을 유지하면서 음성 특성과 운율 특성의 전송을 향상시키는 데 성공하였다.



### CTP-LLM: Clinical Trial Phase Transition Prediction Using Large Language Models (https://arxiv.org/abs/2408.10995)
- **What's New**: 이 논문은 Clinical Trial Outcome Prediction (CTOP)을 위해 대규모 언어 모델(LLM)을 활용한 최초의 모델인 CTP-LLM을 제안합니다. 이는 임상 시험 프로토콜 문서를 분석하여 자동으로 시험 단계 전이를 예측하는 혁신적인 접근 방식을 도입합니다.

- **Technical Details**: CTP-LLM 모델은 GPT-3.5 기반으로 설계되었으며, PhaseTransition (PT) 데이터셋을 통해 조정되어 임상 시험의 원본 프로토콜 텍스트를 분석하여 단계 전이를 예측합니다. 이 모델은 인간이 선택한 특성 없이 학습되며, 다양한 임상 시험 문서의 정보를 통합하여 67%의 예측 정확도를 달성합니다. 특히, Phase~III에서 최종 승인으로 전이하는 경우에 대해서는 75%의 높은 정확도를 보여줍니다.

- **Performance Highlights**: CTP-LLM 모델은 모든 단계에서 F1 점수 0.67을 기록하며, Phase III로의 전이를 예측할 때 F1 점수 0.75로 더욱 높은 성과를 나타냅니다. 이러한 결과는 LLM 기반 응용 프로그램이 임상 시험 결과 예측 및 시험 설계를 평가하는 데 큰 잠재력이 있음을 시사합니다.



### The fusion of phonography and ideographic characters into virtual Chinese characters -- Based on Chinese and English (https://arxiv.org/abs/2408.10979)
Comments:
          14 pages, 7 figures

- **What's New**: 이 논문에서는 중국어와 영어의 장단점을 분석하여 새로운 문자 시스템을 제안합니다.

- **Technical Details**: 중국어(Chinese)는 학습이 어렵고 마스터(이해)하기는 쉬우며, 영어(English)는 학습하기는 쉽지만 방대한 어휘(vocabulary)를 가지고 있습니다. 새롭게 제안된 문자 시스템은 그림 문자(ideographic characters)와 음성 문자(phonetic characters)의 장점을 조합하여, 조합 가능한 새로운 문자가 생성됩니다. 특별한 접두사(special prefixes)를 통해 초보자들이 새로운 단어의 기본적인 범주(category)와 의미(meaning)를 빠르게 유추할 수 있도록 돕습니다.

- **Performance Highlights**: 새로운 문자 시스템은 학습해야 할 어휘량을 줄여주며, 더 깊은 과학 지식(deep scientific knowledge)을 쉽게 습득할 수 있도록 설계되었습니다.



### NLP for The Greek Language: A Longer Survey (https://arxiv.org/abs/2408.10962)
- **What's New**: 이번 연구에서는 헬라어(NLP) 언어 처리 도구와 자원이 열악한 그리스어에 대한 자동 처리 연구를 돌아보면서 지난 30년간의 연구 성과를 정리한다.

- **Technical Details**: 이 논문은 현대 그리스어, 고대 그리스어 및 다양한 그리스 방언에 대한 처리 기술을 정리한다. NLP 작업, 정보 검색(Information Retrieval) 및 지식 관리(Knowledge Management)를 위한 자원과 도구들을 다양한 처리 계층(layer) 및 문맥에 따라 범주화하여 설명한다.

- **Performance Highlights**: 연구 결과는 그리스어 NLP 연구자와 학생들에게 유용할 것이며, 221개의 참고 문헌과 함께 도메인별 주석 데이터셋, 임베딩(embeddings), 도구 및 관련 참고 문헌에 대한 테이블을 제공하여 독자가 필요한 정보를 찾는 데 도움을 준다.



### SysBench: Can Large Language Models Follow System Messages? (https://arxiv.org/abs/2408.10943)
- **What's New**: 이번 연구에서는 다양한 Large Language Models (LLMs)의 시스템 메시지 준수 능력을 평가할 수 있는 새로운 벤치마크인 SysBench를 소개합니다. SysBench는 제약 조건의 복잡성, 명령 불일치, 다중 회전 안정성의 세 가지 측면에서 LLM의 성능을 분석합니다.

- **Technical Details**: SysBench는 실세계에서 일반적으로 사용되는 시스템 메시지에서 도출된 6가지 제약 유형을 포함한 500개의 시스템 메시지와 각각 5턴의 사용자 대화로 구성된 데이터셋을 기반으로 합니다. 이 데이터셋은 수작업으로 검증되어 높은 품질을 보장합니다.

- **Performance Highlights**: 14개의 대중적인 LLM에 대해 실험을 수행한 결과, 다양한 모델 간에 성능 차이를 발견했습니다. 특히, 사용자 지시와 시스템 메시지가 충돌할 때 준수율이 눈에 띄게 감소했습니다. 이런 결과는 향후 시스템 메시지 기제를 개선하기 위한 통찰을 제공합니다.



### LBC: Language-Based-Classifier for Out-Of-Variable Generalization (https://arxiv.org/abs/2408.10923)
Comments:
          16 pages, 7 figures, 4 tables

- **What's New**: 이 논문에서는 Out-of-Variable (OOV) 작업에 대해 대형 언어 모델(LLMs)을 활용한 새로운 분류기인 Language-Based-Classifier (LBC)를 제안합니다. LBC는 전통적인 기계 학습 모델에 비해 OOV 작업에서의 성능을 통해 LLMs의 이점을 극대화하는 방법론을 제공합니다.

- **Technical Details**: LBC는 세 가지 주요 방법론을 사용하여 모델의 인지 능력을 향상시킵니다: 1) Categorical Changes를 통해 데이터를 모델의 이해에 맞추어 조정하고, 2) Advanced Order & Indicator로 데이터 표현을 최적화하며, 3) Verbalizer를 사용하여 추론 중 logit 점수를 클래스에 매핑합니다. 또한, LOw-Rank Adaptation (LoRA) 방법을 사용하여 분류기를 미세 조정합니다.

- **Performance Highlights**: LBC는 OOV 작업에 대해 이전 연구들 중 최초로 LLM 기반 모델을 적용하였으며, 이론적 및 경험적 검증을 통해 LBC의 우수성을 입증하였습니다. LBC는 기존 모델보다 OOV 작업에서 더 높은 성능을 보이는 것으로 나타났습니다.



### CHECKWHY: Causal Fact Verification via Argument Structur (https://arxiv.org/abs/2408.10918)
Comments:
          Accepted by ACL2024; Awarded as Outstanding Paper Award and Area Chair Award

- **What's New**: CheckWhy 데이터셋은 복잡한 causal fact verification (인과 사실 검증) 작업을 위한 새로운 데이터셋으로, 주장의 진실성을 검증하기 위한 명시적인 논리적 추론 과정을 강조합니다.

- **Technical Details**: CheckWhy 데이터셋은 19,596개의 'why' 주장-증거-주장 구조 삼중항으로 구성되어 있으며, 이는 각 주장이 인과 관계를 포함하고 그에 대한 주장 구조가 관련 증거로 이루어져 있습니다. 이 데이터는 인과 관계를 검증하기 위해 제공된 주장 구조에 의존합니다. 또한, 인간-모델 협업 주석 접근 방식을 사용하여 주장, 증거 및 해당 주장 구조를 생성합니다.

- **Performance Highlights**: 최신 모델에 대한 광범위한 실험을 통해, 주장 구조를 포함하는 것이 인과 사실 검증에 있어 중요하다는 것이 입증되었습니다. 그러나 미세 조정된 모델 및 Chain-of-Thought를 활용한 LLM으로부터 만족스러운 주장 구조를 생성하는 것에는 많은 어려움이 있어 향후 개선 여지가 큽니다.



### To Code, or Not To Code? Exploring Impact of Code in Pre-training (https://arxiv.org/abs/2408.10914)
- **What's New**: 이 연구에서는 사전 훈련(data mixture)에서 코드 데이터(code data)가 LLMs(대형 언어 모델)의 일반 성능에 미치는 영향을 체계적으로 조사하고 있습니다. 모델의 성능 향상뿐만 아니라, 코드 데이터가 비 코드 작업(non-code tasks)에 미치는 정확한 영향을 아는 것이 목표입니다.

- **Technical Details**: 대규모 사전 훈련 실험을 통해, 코드 데이터가 업계 표준의 사전 훈련에 어떻게 기여하는지 분석했습니다. 실험에는 다양한 자연어 추론(natural language reasoning) 작업, 세계 지식(world knowledge) 작업, 코드 벤치마크(code benchmarks), LLM-as-a-judge 승률(win-rates)을 포함하여 470M에서 2.8B 파라미터를 가진 모델을 평가했습니다.

- **Performance Highlights**: 코드 데이터는 비율적으로 분석할 때 자연어 추론에서 8.2%, 세계 지식에서 4.2%, 생성적 승률에서 6.6%, 코드 성능에서 12배의 성장을 보였습니다. 사전 훈련에 코드 품질(code quality) 및 속성을 중요하게 고려하며, 이는 전반적으로 모든 작업 성능 향상에 기여하고 있음을 보여주었습니다.



### BEYOND DIALOGUE: A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Mod (https://arxiv.org/abs/2408.10903)
- **What's New**: 대규모 언어 모델(LLMs)의 발전에 따라 새로운 일반적 역할 수행 모델을 위한 BEYOND DIALOGUE라는 프레임워크가 제안되었습니다. 이 프레임워크는 특정 시나리오에 따른 역할 특성과 대화를 정렬하여, 훈련 시 편향을 제거하고 문장 수준에서 세부 정렬을 달성할 수 있도록 돕습니다.

- **Technical Details**: BEYOND DIALOGUE 프레임워크는 'beyond dialogue' 작업을 통해 대화와 역할 특성을 정렬함으로써 학습 편향을 제거합니다. 이는 혁신적인 프롬프트 메커니즘을 통해 문장 수준에서 역할 프로필과 대화 간의 관계를 세밀하게 조정합니다. 이 접근법은 자동화된 대화 및 객관적 평가 방법을 통합하여 전반적인 역할 수행의 개선을 도모합니다.

- **Performance Highlights**: 실험 결과, BEYOND DIALOGUE 프레임워크를 적용한 모델이 기존의 LLMs보다 역할 프로필을 따르고 반영하는 능력이 뛰어나며, GPT-4o 및 Baichuan-NPC-Turbo와 같은 특수한 역할 수행 모델을 크게 초월했습니다. 이 모델은 평가 정확도를 30% 이상 개선하여 자가生成, 자가평가 및 자가정렬 파이프라인의 개발을 지원합니다.



### Soda-Eval: Open-Domain Dialogue Evaluation in the age of LLMs (https://arxiv.org/abs/2408.10902)
Comments:
          22 pages, 10 figures

- **What's New**: 본 논문은 GPT-3.5 기반의 Soda 대화 데이터셋을 분석하여 현재의 챗봇들이 지닌 일관성 부족(coherence), 상식 지식(commmonsense knowledge) 문제를 드러냈으나, 응답의 유창성(fluidity)과 관련성(relevance)에는 만족스럽다는 것을 확인했습니다. 새로운 데이터셋인 Soda-Eval을 소개하며, 이는 10,000개의 대화를 기준으로 120,000개 이상의 턴 레벨 평가(turn-level assessments)를 포함하고 있습니다.

- **Technical Details**: Soda-Eval은 GPT-4에 의해 생성된 주석(annotation)을 기반으로 하여 여러 품질 측면을 평가합니다. 본 연구에서는 공개 액세스가 가능한 instruction-tuned LLM들을 사용하여 대화 평가자(dialogue evaluators)로서의 성능을 검토하고, 모델 튜닝(fine-tuning) 과정을 통해 성능 향상을 확인했습니다.

- **Performance Highlights**: Soda-Eval을 통해 몇몇 공개 액세스 LLM들의 성능을 평가한 결과, 챗봇 평가가 여전히 어려운 임무임을 알 수 있었습니다. 모델을 튜닝한 결과, GPT-4의 평가와의 상관관계(correlation) 및 설명의 타당성(validity) 모두 향상되었습니다. 이러한 결과는 다양한 평가 가이드라인에 대한 모델의 적응성(adaptability)을 나타냅니다.



### Benchmarking Large Language Models for Math Reasoning Tasks (https://arxiv.org/abs/2408.10839)
- **What's New**: 본 연구는 수학 문제 해결을 위한 여섯 가지 최신 알고리즘의 공정한 비교 벤치마크를 제시하며, 다섯 가지 널리 사용되는 수학 데이터 세트에서 네 가지 강력한 기본 모델을 기반으로 한다.

- **Technical Details**: 이 연구에서는 LLMs (Large Language Models)의 수학적 추론 능력을 평가하기 위해 Chain-of-Thought (CoT), Auto CoT, Zero-Shot CoT, Self-Consistency, Complex CoT, PAL, PoT 등 여러 가지 알고리즘을 탐구한다. 수학 문제 해결의 정확도, 강건성, 자원 사용의 효율성을 핵심 과제로 설정하였다.

- **Performance Highlights**: 결과적으로, LLaMA 3-70B와 함께 사용하는 Auto CoT 알고리즘이 계산 자원과 성능 사이에서 최적의 균형을 제공하며, GPT-3.5는 Zero-Shot CoT와 함께 가장 높은 성능을 제공했다. 또한, 연구진은 차세대 연구를 지원하기 위해 벤치마크 코드마저 오픈소스로 공개하였다.



### Exploiting Large Language Models Capabilities for Question Answer-Driven Knowledge Graph Completion Across Static and Temporal Domains (https://arxiv.org/abs/2408.10819)
- **What's New**: 본 논문에서는 Generative Subgraph-based KGC (GS-KGC)라는 새로운 지식 그래프 완성 프레임워크를 소개합니다. GS-KGC는 질문-답변 형식을 활용하여 목표 엔티티를 직접 생성하고, 불확실한 질문에 대한 다수의 답변 가능성을 해결합니다.

- **Technical Details**: GS-KGC는 지식 그래프(KG) 내의 엔티티 및 관계를 중심으로 서브그래프를 추출하여, 부정 샘플 및 이웃 정보를 separately (별도로) 얻는 전략을 제안합니다. 이 방법은 알고 있는 사실을 사용하여 부정 샘플을 생성하며, 대형 언어 모델(LLM)의 추론을 향상시키기 위해 컨텍스트 정보를 제공합니다.

- **Performance Highlights**: GS-KGC는 4개의 정적 지식 그래프(SKG)와 2개의 시간적 지식 그래프(TKG)에서 성능을 평가한 결과, 5개 데이터셋에서 state-of-the-art Hits@1 메트릭을 달성했습니다. 이 방법은 기존 KG 내에서 새로운 트리플을 발견하고, 닫힌 KG를 넘어 새로운 사실을 생성하여 효과적으로 닫힌세계(closed-world)와 열린세계(open-world) KGC 간의 간극을 좁힙니다.



### Beyond English-Centric LLMs: What Language Do Multilingual Language Models Think in? (https://arxiv.org/abs/2408.10811)
Comments:
          work in progress

- **What's New**: 이번 연구에서는 비영어 중심의 LLMs가 고성능에도 불구하고 해당 언어로 '사고'하는 방식을 조사합니다. 이는 중간 레이어의 표현이 특정 지배 언어에 대해 더 높은 확률을 보일 때 등장하는 개념인 내부 latent languages에 대한 것입니다.

- **Technical Details**: 연구에서는 Llama2(영어 중심 모델), Swallow(일본어에서 계속해서 사전 훈련된 영어 중심 모델), LLM-jp(균형 잡힌 영어 및 일본어 코퍼스로 사전 훈련된 모델) 세 가지 모델을 사용하여 일본어 처리에서의 내부 latent language를 조사했습니다. 이후 logit lens 방법을 통해 각 레이어의 내부 표현을 어휘 공간으로 비구속화하여 분석했습니다.

- **Performance Highlights**: Llama2는 내부 latent language로 전적으로 영어를 사용하며, Swallow와 LLM-jp는 영어와 일본어의 두 가지 내부 latent languages를 사용합니다. 일본어 특화 모델은 특정 목표 언어에 대해 가장 관련이 깊은 latent 언어를 선호하여 활성화합니다.



### ColBERT Retrieval and Ensemble Response Scoring for Language Model Question Answering (https://arxiv.org/abs/2408.10808)
Comments:
          This work has been submitted to the 2024 IEEE Globecom Workshops for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 이번 연구는 통신 네트워크를 위한 소규모 언어 모델인 Phi-2와 Falcon-7B의 성능을 향상시키기 위한 특별한 챌린지를 기반으로 개발된 질문 응답(QA) 시스템을 소개합니다. 두 모델 모두 특정 도메인 내에서 질문을 처리할 때 효과성을 입증하며, Phi-2는 81.9%의 높은 정확도를 기록했습니다.

- **Technical Details**: 연구에서는 Retrieval-Augmented Generation (RAG) 기법을 활용하여 3GPP 문서에서 관련 맥락을 검색하고, 질문에 대해 기술 약어의 전체 이름을 포함하는 프롬프트를 설계했습니다. 각 모델에 대한 특정 방법론을 바탕으로, Phi-2는 LoRA 어댑터를 사용해 훈련되었고 Falcon-7B는 생성된 응답을 기반으로 평가 메커니즘을 개발했습니다.

- **Performance Highlights**: 솔루션은 두 모델 모두 각각의 트랙에서 최고의 성과를 달성했습니다. Phi-2는 개인 평가 데이터셋에서 81.9%의 정확도를, Falcon-7B는 57.3%의 정확도를 기록했습니다. 이는 해당 도메인에서 이전에 보고된 성과에 비해 유의미한 개선입니다.



### Adversarial Attack for Explanation Robustness of Rationalization Models (https://arxiv.org/abs/2408.10795)
- **What's New**: 본 연구에서는 rationalization 모델의 설명에 대한 공격에 대한 내성을 조사하며, 이러한 내성의 악화가 기존의 설명 방법에서 발생하는 불안정성의 원인임을 밝혔다.

- **Technical Details**: 제안된 UAT2E는 gradient-based search 방법을 사용하여 입력 텍스트에 트리거를 삽입하여 비타겟 및 타겟 공격을 수행한다. 이 과정에서 mean squared error (MSE) 손실 함수를 활용하여 rationale의 차이를 측정하고, cross-entropy 손실을 통해 예측의 차이를 계산한다.

- **Performance Highlights**: 다섯 개의 공공 데이터셋에서의 실험 결과는 기존의 rationalization 모델이 설명에 있어 매우 취약함을 드러내었다. 특히, 공격을 받을 경우 무의미한 토큰을 선택하는 경향이 있으며, 강력한 인코더를 사용하더라도 설명의 내구성은 보장되지 않는다.



### Predicting Rewards Alongside Tokens: Non-disruptive Parameter Insertion for Efficient Inference Intervention in Large Language Mod (https://arxiv.org/abs/2408.10764)
Comments:
          16 pages

- **What's New**: 이번 연구에서는 transformer 기반의 대형 언어 모델(LLM)이 가진 안전하지 않은 응답 생성, 신뢰할 수 없는 추론 등의 문제를 해결하기 위해 Non-disruptive parameters insertion(Otter) 방법을 제안합니다. 기존의 모델을 별도로 finetuning하지 않고, 추가적인 파라미터를 동일한 구조에 삽입하여 원래의 LLM 출력을 방해받지 않으면서 보정 신호를 예측할 수 있도록 합니다.

- **Technical Details**: Otter는 transformer 아키텍처의 multi-head attention layer와 feed-forward neural network layer에 추가적인 trainable 파라미터를 삽입하여 동작합니다. 이를 통해 LLM의 원래 출력과 함께 보정 신호를 동시에 생성할 수 있습니다. Otter는 기존 모델과 통합하여 코드 변경 없이 효율적인 디코딩을 제공합니다.

- **Performance Highlights**: Otter는 세 가지 고난이도 작업(생성 탈독화, 선호 정렬, 추론 가속)에서 state-of-the-art(SOTA) 성능을 달성하면서도, 최대 86.5%의 추가 공간 절약 및 98.5%의 추가 시간 절약을 보여주었습니다. 또한, 원래 모델의 응답을 여전히 이용할 수 있어 성능 저하를 방지합니다.



### Towards Efficient Large Language Models for Scientific Text: A Review (https://arxiv.org/abs/2408.10729)
- **What's New**: 최근의 대규모 언어 모델(LLMs)은 과학 분야의 복잡한 정보 처리에 새로운 시대를 열어주었습니다. 논문에서는 LLMs의 진화된 기능을 더 접근 가능한 과학 AI 솔루션으로 전환하는 방법과 이를 위한 도전과 기회를 탐구합니다.

- **Technical Details**: 이 논문은 LLMs의 두 가지 주요 접근 방식인 모델 크기 축소와 데이터 품질 향상에 대한 포괄적인 리뷰를 제공합니다. 특히, 과학 연구에서 LLMs의 활발한 활용을 위한 기법으로 Parameter-Efficient Fine-Tuning (PEFT), LoRA 및 Adapter Tuning과 같은 최신 기술들이 소개되었습니다.

- **Performance Highlights**: LLMs는 생명과학, 생물의학, 시각적 과학 등 다양한 과학 분야에서 인상적인 성능을 보였으며, 특히 PEFT 방식의 구현을 통한 모델이 제한된 자원에서도 효율적으로 활용될 수 있음을 보여주었습니다.



### Crafting Tomorrow's Headlines: Neural News Generation and Detection in English, Turkish, Hungarian, and Persian (https://arxiv.org/abs/2408.10724)
- **What's New**: 이번 연구에서 우리는 영어, 터키어, 헝가리어, 페르시아어로 구성된 신경 뉴스 탐지를 위한 벤치마크 데이터 세트를 처음으로 소개합니다. 이 데이터 세트는 BloomZ, LLaMa-2, Mistral, Mixtral, GPT-4와 같은 최신 멀티링구얼 생성기에서 생성된 결과물들을 포함하고 있습니다.

- **Technical Details**: 연구에서는 다양한 분류기(classifiers)를 실험하였으며, 언어적 특징에 기반한 모델부터 고급 Transformer 기반 모델과 LLMs prompting까지 다양합니다. 특히, 기계 생성 뉴스의 감지 결과를 해석 가능성과 강인성 측면에서 분석하였습니다. 또한, 본 연구에서 구축한 데이터 세트는 공개적으로 사용 가능합니다.

- **Performance Highlights**: 우리는 각 분류기(discriminator), 언어별, 생성기별로 분류 점수를 보고하였으며, 연구 결과는 기계 생성 텍스트 탐지의 다양한 기준선을 탐험하는 것으로 요약됩니다. 생성 품질 분석 결과에서 BloomZ-3B 및 LLaMa-2-Chat-7B가 특히 유망한 후속 작업 대상으로 부각되었습니다.



### MEGen: Generative Backdoor in Large Language Models via Model Editing (https://arxiv.org/abs/2408.10722)
Comments:
          Working in progress

- **What's New**: 이 논문은 자연어 처리(NLP) 작업을 위한 맞춤형 백도어를 생성하는 새로운 방법인 MEGen을 제안합니다. MEGen은 최소한의 부작용으로 백도어를 주입할 수 있는 에디팅 기반의 생성적 백도어 공격 전략을 제공합니다.

- **Technical Details**: MEGen은 고정된 메트릭스에 따라 선택된 트리거를 입력에 삽입하기 위해 언어 모델을 활용하고, 모델 내부의 백도어를 직접 주입하기 위한 편집 파이프라인을 설계합니다. 이 방법은 소량의 로컬 매개변수를 조정하여 백도어를 효과적으로 삽입하면서 원래의 모델 성능은 유지할 수 있도록 합니다.

- **Performance Highlights**: 실험결과 MEGen은 더 적은 샘플로도 백도어를 효율적으로 주입할 수 있으며, 다양한 다운스트림 작업에서 높은 공격 성공률을 달성하였습니다. 이는 깨끗한 데이터에 대한 원래 모델 성능을 유지하면서도 공격 시 특정 위험한 정보를 자유롭게 출력할 수 있음을 보여줍니다.



### Ferret: Faster and Effective Automated Red Teaming with Reward-Based Scoring Techniqu (https://arxiv.org/abs/2408.10701)
- **What's New**: 새로운 접근 방식인 Ferret는 Rainbow Teaming의 한계를 극복하여 각 반복(iteration)마다 여러 개의 적대적인 프롬프트 변이를 생성하고, 이를 효과적으로 순위 매기기 위해 점수 함수(score function)를 사용합니다.

- **Technical Details**: Ferret는 다양한 점수 함수(예: reward models, Llama Guard, LLM-as-a-judge)를 활용하여 해로운 변이가 될 가능성이 있는 적대적 변이를 평가하고 선택하는 방식으로 설계되었습니다. 이로 인해 공격 성공률(ASR)을 높이고, 검색(search) 과정의 효율성을 개선합니다.

- **Performance Highlights**: Ferret는 보상 모델(reward model)을 점수 함수로 사용할 경우 공격 성공률을 95%로 향상시키며, 이는 Rainbow Teaming보다 46% 높은 수치입니다. 또한, 90% ASR을 달성하는 데 드는 시간을 기준선에 비해 15.2% 단축하였고, 생성된 적대적 프롬프트는 다른 대형 LLM에서도 효과적으로 작용하는 전이성(transferable) 특성을 지닙니다.



### Unconditional Truthfulness: Learning Conditional Dependency for Uncertainty Quantification of Large Language Models (https://arxiv.org/abs/2408.10692)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 생성 과정에서 발생하는 조건부 의존성(conditional dependency)을 데이터로부터 학습하여 불확실성 정량화(Uncertainty Quantification, UQ)의 문제를 해결하는 방법을 제안합니다. 제안된 방법은 'trainable attention-based dependency (TAD)'로 명명되며, 기존의 방법들보다 효과적으로 불확실성을 측정할 수 있는 새로운 접근을 소개합니다.

- **Technical Details**: 본 연구에서는 LLM의 생성 과정에서 이전의 생성된 토큰들 간의 주의(attention) 구조를 활용하여 조건부 생성 신뢰도와 무조건 생성 신뢰도 간의 간극을 예측하는 회귀 모델을 훈련시킵니다. 이 모델은 LLM 추론 과정에서 현재 생성 단계의 불확실성을 이전 단계의 불확실성을 바탕으로 조정하는 데 사용됩니다.

- **Performance Highlights**: 본 연구의 실험은 9개의 데이터셋과 3개의 LLM을 대상으로 하여 수행되었으며, 제안된 TAD 방법이 기존의 경쟁 방법들에 비해 UQ에서 현저한 개선을 보여줌을 입증하였습니다. 이는 LLM이 긴 시퀀스를 생성하는 작업에서 특히 두드러집니다.



### Towards Robust Knowledge Unlearning: An Adversarial Framework for Assessing and Improving Unlearning Robustness in Large Language Models (https://arxiv.org/abs/2408.10682)
Comments:
          13 pages

- **What's New**: 이번 논문에서는 LLM(대형 언어 모델)의 훈련 데이터에서 발생할 수 있는 문제 있는 지식의 영향을 줄이기 위해 동적 비학습 공격(Dynamic Unlearning Attack, DUA)과 잠재적 적대적 비학습(Latent Adversarial Unlearning, LAU) 프레임워크를 제안합니다. 기존의 비학습 방법들이 적대적 쿼리에 취약한 것을 인식하고, 이를 해결하기 위한 두 가지 새로운 방법론을 제시합니다.

- **Technical Details**: DUA는 자동화된 공격 프레임워크로, 비학습 모델의 강건성을 평가하기 위해 보편적인 적대적 접미사를 최적화합니다. 반면, LAU는 비학습 과정을 min-max 최적화 문제로 설정하고, 공격 단계에서는 perturbation 벡터를 훈련하여 LLM의 잠재 공간에 추가하고, 방어 단계에서는 이전에 훈련된 perturbation 벡터를 사용하여 모델의 강건성을 향상합니다.

- **Performance Highlights**: LAU 프레임워크를 사용한 결과, AdvGA 및 AdvNPO라는 두 가지 강력한 비학습 방법을 제안하며, 비학습의 효과성을 53.5% 이상 개선하고, 이웃 지식에 대한 감소를 11.6% 미만으로 유지하며, 모델의 일반적 능력에는 거의 영향을 미치지 않는 것으로 나타났습니다.



### HMoE: Heterogeneous Mixture of Experts for Language Modeling (https://arxiv.org/abs/2408.10681)
- **What's New**: 이번 연구에서는 Heterogeneous Mixture of Experts (HMoE)라는 새로운 모델을 제안합니다. 이 모델은 전문가들이 서로 다른 크기를 가지며 각 전문가의 능력이 다릅니다. 이러한 이질성(heterogeneity)은 다양한 복잡성을 처리하는데 더 효과적인 전문가를 가능하게 합니다.

- **Technical Details**: HMoE는 기존의 동질적 전문가(homogeneous experts) 모델과 달리 전문가의 크기가 다양하여 서로 다른 능력을 발휘할 수 있습니다. 본 연구에서는 작은 전문가의 활성화를 장려하는 새로운 훈련 목표를 제안하여, 계산 효율성(computational efficiency)과 매개변수 활용(parameter utilization)을 개선합니다. 또한, Top-K Routing 및 Top-P Routing 전략을 채택하여 전문가의 활성화를 최적화합니다.

- **Performance Highlights**: HMoE는 적은 수의 활성화된 매개변수로 더 낮은 손실(loss)을 달성하며, 기존의 동질적 MoE 모델들보다 다양한 사전 훈련(pre-training) 평가 기준에서 성능이 우수한 결과를 보였습니다. 실험 결과, HMoE 모델은 컴퓨터 효율성을 높이면서도 다양한 다운스트림 성능을 저해하지 않고, 더욱 효과적인 전문가 활용을 보여줍니다.



### Towards Rehearsal-Free Multilingual ASR: A LoRA-based Case Study on Whisper (https://arxiv.org/abs/2408.10680)
- **What's New**: 본 연구에서는 Whisper 모델의 파인튜닝 과정에서 발생하는 파국적 망각(catastrophic forgetting) 문제를 해결하기 위한 방법으로 orthogonal gradient descent를 활용하는 새로운 접근 방식을 제안합니다. 기존 모델의 LoRA(LoRA-based) 파라미터를 이용하여 새로운 언어에 대한 학습을 진행함으로써 원래 언어의 성능을 유지할 수 있도록 합니다.

- **Technical Details**: 연구는 LoRA 파라미터를 활용한 다양한 방법을 비교하였으며, forgetting에 대한 취약성을 검사했습니다. 원래 모델의 LoRA 파라미터를 이용해 새로운 샘플에 대해 근사적인 orthogonal gradient descent를 적용하고, 효율적인 학습을 위해 learnable rank coefficient를 도입하였습니다.

- **Performance Highlights**: 중국어 Whisper 모델을 사용한 실험 결과, 우즈베크어(Uyghur)와 티베트어(Tibetan)에서 더 적은 파라미터 집합으로 향상된 성능을 보였습니다. 새로 도입된 방법은 rehearsal-free, parameter-efficient 및 task-id-free 특성을 가지며, 이는 모델의 일반화 능력을 크게 향상시킵니다.



### REInstruct: Building Instruction Data from Unlabeled Corpus (https://arxiv.org/abs/2408.10663)
Comments:
          Accepted by ACL2024 Findings

- **What's New**: 이 논문에서는 고급 언어 모델(LLM)을 위한 지침 데이터를 자동으로 구축하는 REInstruct라는 간단하고 확장 가능한 방법을 제안합니다. 이 방법은 독점 LLM 및 수동 주석에 대한 의존도를 줄이고, 비표시 텍스트 코퍼스에서 고품질의 지침 데이터를 구성할 수 있습니다.

- **Technical Details**: REInstruct는 먼저 비표시 텍스트 중에서 유용하고 통찰력 있는 내용을 포함할 가능성이 높은 부분 집합을 선택하고, 이후 이 텍스트에 대한 지침을 생성하는 방식입니다. 또한, 생성된 지침의 품질을 향상시키기 위해 리라이팅 기반 접근 방식을 제안합니다. Llama-7b 모델을 3천 개의 초기 데이터와 3만 2천 개의 REInstruct에서 생성된 합성 데이터로 훈련하여 성능을 평가합니다.

- **Performance Highlights**: REInstruct로 생성된 데이터로 세밀한 조정을 마친 Llama-7b 모델은 AlpacaEval leaderboard에서 text-davinci-003을 상대로 65.41%의 승률을 달성하며, 다른 오픈 소스 비증류(non-distilled) 지침 데이터 구축 방법을 초월하는 성능을 보여줍니다.



### Beneath the Surface of Consistency: Exploring Cross-lingual Knowledge Representation Sharing in LLMs (https://arxiv.org/abs/2408.10646)
- **What's New**: 이 논문은 멀티링구얼(다국어) LLMs(대형 언어 모델)에서 사실 정보의 표현이 여러 언어에 걸쳐 어떻게 공유되는지를 조사합니다. 특히 언어 간 일관성과 공유 표현의 중요성을 강조하고, LLM이 다양한 언어에서 사실을 얼마나 일관되게 응답하는지를 측정하는 방법론을 제안합니다.

- **Technical Details**: 논문에서 제안하는 측정 방법은 두 가지 주요 측면인 Cross-lingual Knowledge Consistency (CKC)와 Cross-lingual Knowledge Representation Sharing (CKR)을 기반으로 합니다. CKC는 모델이 다양한 언어에서 사실 질문에 일관성을 갖고 응답하는 정도를 측정하고, CKR은 동일한 사실에 대해 여러 언어 간의 공통 내적 표현을 사용하는 정도를 측정합니다.

- **Performance Highlights**: 연구 결과, LLMs가 각 언어에서 150% 더 많은 사실을 정확히 대답할 수 있다는 것을 발견했습니다. 특히 같은 스크립트에 속하는 언어 간에는 높은 표현 공유를 보였으며, 낮은 자원 언어와 다른 스크립트를 가진 언어 간에는 정답 일치도가 높지만 표현 공유는 적은 경향을 나타냈습니다.



### Enhancing Robustness in Large Language Models: Prompting for Mitigating the Impact of Irrelevant Information (https://arxiv.org/abs/2408.10615)
- **What's New**: 이 논문에서는 무관한 정보가 포함된 초등학교 수학 문제를 담은 데이터셋 GSMIR를 구성하여 대형 언어 모델(LLMs)의 추론 능력에 미치는 영향을 종합적으로 분석하였습니다. 특히, 기존의 탐색적 기법이 무관한 정보의 영향력을 완전히 제거하지 못함을 강조하며, 새로운 자동 구성 방법인 ATF (Analysis to Filtration Prompting)를 제안합니다.

- **Technical Details**: ATF 방식은 두 가지 단계로 구성됩니다: 첫 번째 단계는 무관한 정보를 분석하는 과정이며, 두 번째는 그런 정보를 걸러내는 단계입니다. 이 방법은 LLMs가 입력 문제 설명을 여러 조항으로 나누고 각각 분석하여 무관한 정보를 식별하고, 이를 토대로 필터링된 문제 설명을 생성하게 합니다.

- **Performance Highlights**: 실험 결과, ATF 방법을 적용한 LLM들은 무관한 정보가 포함된 문제 해결 시 추론 정확도가 크게 향상되었습니다. 다양한 프롬프트 기법과 조합하여 ATF를 평가했으며, 그 결과 모든 프롬프트 방법에서 유의미한 개선을 보여주었습니다.



### Promoting Equality in Large Language Models: Identifying and Mitigating the Implicit Bias based on Bayesian Theory (https://arxiv.org/abs/2408.10608)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)에서 나타나는 '암묵적 편향 문제'(implicit bias problem)를 정식으로 정의하고, 이를 해결하기 위한 혁신적인 프레임워크인 베이지안 이론 기반의 편향 제거(Bayesian-Theory based Bias Removal, BTBR)를 개발하였습니다.

- **Technical Details**: BTBR는 우선적으로 공개적으로 접근 가능한 편향 데이터셋에서 의도하지 않게 포함된 편향을 나타내는 데이터를 식별하기 위해 우도 비율 스크리닝(likelihood ratio screening)을 사용합니다. 그 후, 관련된 지식 트리플(knowledge triples)을 자동으로 구성하고 모델 편집 기법(model editing techniques)을 통해 LLM에서 편향 정보를 삭제합니다.

- **Performance Highlights**: 실험을 통해, LLM에서 암묵적 편향 문제의 존재를 확인하고, BTBR 접근 방식의 효과성을 입증하였습니다.



### Multilingual Non-Factoid Question Answering with Silver Answers (https://arxiv.org/abs/2408.10604)
- **What's New**: MuNfQuAD라는 새로운 다국어 질의응답 데이터셋이 소개되었습니다. 이는 비사실 기반(non-factoid) 질문을 포함하고 있으며, BBC 뉴스 기사의 질문과 해당 단락을 사용하여 구성되었습니다.

- **Technical Details**: MuNfQuAD는 38개 언어를 포괄하며, 370,000개 이상의 QA 쌍으로 이루어져 있습니다. 수동 주석을 통해 790개의 QA 쌍이 검토되었으며, 98%의 질문이 해당하는 은유적 답변(silver answer)으로 답변될 수 있음을 확인했습니다. 이 연구에서는 Answer Paragraph Selection (APS) 모델이 최적화되어 기존 기준보다 뛰어난 성과를 보였습니다.

- **Performance Highlights**: APS 모델은 MuNfQuAD 테스트셋에서 80%의 정확도와 72%의 매크로 F1 점수를 기록하였으며, 골든 세트(golden set)에서는 각각 72%와 66%의 성과를 달성했습니다. 또한, 이 모델은 은유적 레이블(silver labels)로 최적화 된 후에도 골든 세트 내 특정 언어를 효과적으로 일반화할 수 있었습니다.



### An Efficient Sign Language Translation Using Spatial Configuration and Motion Dynamics with LLMs (https://arxiv.org/abs/2408.10593)
Comments:
          Under Review

- **What's New**: 이번 연구에서는 면밀하게 설계된 Spatial and Motion-based Sign Language Translation (SpaMo) 프레임워크를 소개하여, 자원 집약적인 비주얼 인코더의 파인 튜닝 없이도 수화 비디오에서 예측하는 과정을 개선하고 있습니다.

- **Technical Details**: SpaMo는 사전 훈련된 이미지 인코더(ViT)와 비디오 인코더(VideoMAE)를 활용하여 공간적 특징과 운동 동작을 추출합니다. 추출된 특징은 LLM(대규모 언어 모델)에 입력되며, Visual-Text Alignment(VT-Align) 과정을 통해 서로 다른 모달리티 간의 차이를 줄입니다.

- **Performance Highlights**: SpaMo는 PHOENIX14T와 How2Sign 두 가지 인기있는 데이터셋에서 최첨단 성능을 기록하였으며, 자원 집약적인 접근법 없이 효과적인 수화 번역을 가능하게 하고 있습니다.



### Putting People in LLMs' Shoes: Generating Better Answers via Question Rewriter (https://arxiv.org/abs/2408.10573)
Comments:
          7 pages, 4 figures, 5 tables

- **What's New**: 이번 논문은 Large Language Models (LLMs)이 질문 응답 (QA)에서의 성능이 사용자 질문의 모호성에 의해 저해된다는 문제를 다룹니다. 이를 해결하기 위해 편리한 방법인 single-round instance-level prompt optimization을 소개하며, 이를 통해 사용자 질문의 명확성을 극대화하는 질문 리라이터 (question rewriter)를 제안합니다.

- **Technical Details**: 리라이터는 자동 생성된 답변을 평가하는 기준으로부터 수집된 피드백을 기반으로 직접 선호 최적화 (direct preference optimization)를 사용하여 최적화됩니다. 이 방법은 비싼 인적 주석이 필요 없다는 장점이 있습니다. 여러 개의 블랙박스 LLM 및 Long-Form Question Answering (LFQA) 데이터셋을 통해 이 방법의 유효성을 실험적으로 입증하였습니다.

- **Performance Highlights**: 본 연구는 질문 리라이터의 훈련을 위한 실용적인 프레임워크를 제공하며, LFQA 작업에서의 프롬프트 최적화에 대한 향후 탐색의 전례를 설정합니다. 논문의 관련 코드는 공개되어 있습니다.



### Speech Representation Learning Revisited: The Necessity of Separate Learnable Parameters and Robust Data Augmentation (https://arxiv.org/abs/2408.10557)
- **What's New**: 본 연구에서는 O-HuBERT라는 수정된 HuBERT 모델을 제안하여 음성의 다른 정보(Other information)를 별도의 학습 가능한 매개변수를 통해 인코딩하는 방법을 탐구합니다. 이는 이전의 연구들과 차별화되는 점으로, 콘텐츠 정보와 다른 정보 간의 최적화를 동시에 수행하는 것의 어려움을 극복하기 위해 설계되었습니다.

- **Technical Details**: O-HuBERT 모델은 음성 데이터의 콘텐츠 정보(Content information)와 다른 정보(Other information)를 효과적으로 모델링하기 위해 서로 다른 학습 가능한 매개변수를 사용하여 공동으로 학습합니다. 이를 위해, 콘텐츠 정보를 모델링하는 데는 HuBERT 모델을 사용하고, 다른 정보를 모델링하기 위해 추가적인 utterance similarity prediction token(XUSP)을 도입하여 전이 인코더(Transform Encoder)의 입력 임베딩에 추가합니다. 이 과정에서 새로운 손실 함수(Loss function)가 도입되어 다른 정보의 최적화를 돕습니다.

- **Performance Highlights**: O-HuBERT는 기존의 HuBERT 모델과 동일한 규모(1억 개의 매개변수) 및 사전 학습 데이터(960시간)를 사용하였으나, SUPERB 벤치마크에서 최신의 성능(SOTA)을 달성하였습니다. 또한, 두 단계의 데이터 증강 (Two-stage data augmentation) 전략을 통해 학습된 정보의 퀄리티를 증대시키는 데 큰 기여를 했습니다.



### Language Modeling on Tabular Data: A Survey of Foundations, Techniques and Evolution (https://arxiv.org/abs/2408.10548)
- **What's New**: 본 논문은 다양한 테이블 데이터를 효과적으로 분석하기 위한 최신 언어 모델링 기법에 대한 포괄적인 검토를 제공하고 있습니다. 특히, 대규모 언어 모델(LLM)의 발달로 인해 테이블 데이터 분석의 패러다임 전환을 다루고 있습니다.

- **Technical Details**: 문서에서는 1D 및 2D 테이블 데이터 구조를 분류하고, 모델 훈련 및 평가에 사용되는 주요 데이터셋을 검토합니다. 또한 데이터 처리 방법, 인기 있는 아키텍처, 훈련 목표 등을 포함한 다양한 모델링 기법의 발전을 요약합니다. 특히, 입력 처리, 중간 모듈, 훈련 목적을 포함하여 최근 연구에서 나타난 언어 모델링 기술의 분류 체계를 제시합니다.

- **Performance Highlights**: 최근 LLMs인 GPT 및 LLaMA의 출현은 테이블 데이터 모델링의 효율성을 극대화하고, 수량적으로 적은 데이터로도 우수한 성능을 발휘할 수 있게 하였습니다. 이로 인해 테이블 데이터에서의 학습과 예측 작업에서 뛰어난 성능을 보입니다.



### NoMatterXAI: Generating "No Matter What" Alterfactual Examples for Explaining Black-Box Text Classification Models (https://arxiv.org/abs/2408.10528)
- **What's New**: 이 논문에서는 기존의 Explainable AI (XAI) 방법론인 counterfactual explanations (CEs)의 한계를 보완하기 위해 alterfactual explanations (AEs)라는 새로운 개념을 제안합니다. AEs는 특정 속성의 영향 없이 예측 결과를 유지하면서 불필요한 특징을 대체하는 방법입니다.

- **Technical Details**: AEs는 'no matter what'의 대안 현실을 탐구하여, 동일한 예측 결과를 유지하며 불필요한 특징을 다른 특징으로 대체합니다. 본 논문에서는 이러한 AEs 생성을 최적화 문제로 공식을 세우고, 텍스트 분류 작업을 위한 새로운 알고리즘 MoMatterXAI를 소개합니다. 이 알고리즘은 90% 이상의 맥락 유사성을 유지하며 95%의 높은 충실도로 AEs를 생성합니다.

- **Performance Highlights**: MoMatterXAI는 실제 데이터셋에 대해 평가를 수행하여 95%의 효과성을 보여주었으며, AEs가 AI 텍스트 분류기를 사용자에게 설명하는 데 효과적임을 인간 연구를 통해 확인했습니다. 모든 코드는 공개될 예정입니다.



### XCB: an effective contextual biasing approach to bias cross-lingual phrases in speech recognition (https://arxiv.org/abs/2408.10524)
Comments:
          accepted to NCMMSC 2024

- **What's New**:  본 연구에서는 이중 언어 설정에서 기계 음성 인식(ASR) 성능을 향상시키기 위해 'Cross-lingual Contextual Biasing(XCB)' 모듈을 도입하였습니다. 이 방법은 주 언어 모델에 보조 언어 편향 모듈과 언어별 손실을 결합하여 이차 언어의 문구 인식을 개선하는 데 중점을 둡니다.

- **Technical Details**:  제안한 XCB 모듈은 언어 편향 어댑터(LB Adapter) 및 편향 병합 게이트(BM Gate)라는 두 가지 핵심 구성 요소로 이루어져 있습니다. LB 어댑터는 이차 언어(secondary language, L2nd)에 연관된 프레임을 구별하여 해당 표현을 강화합니다. BM 게이트는 언어 편향 표현을 생성하며, 이를 통해 L2nd 문구의 인식을 향상시킵니다.

- **Performance Highlights**:  실험 결과, 제안된 시스템은 이차 언어의 편향 문구 인식에서 유의미한 성능 향상을 보였으며, 추가적인 추론 오버헤드 없이도 효율성을 발휘하였습니다. 또한, ASRU-2019 테스트 세트에 적용했을 때도 일반화된 성능을 보여주었습니다.



### Data Augmentation Integrating Dialogue Flow and Style to Adapt Spoken Dialogue Systems to Low-Resource User Groups (https://arxiv.org/abs/2408.10516)
Comments:
          Accepted to SIGDIAL 2024

- **What's New**: 본 연구는 특정 대화 행동을 보이는 사용자, 특히 데이터가 부족한 경우 미성년자와의 상호작용에서 발생하는 도전 과제를 해결하기 위해 특별히 설계된 데이터 증강(data augmentation) 프레임워크를 제시합니다.

- **Technical Details**: 우리의 방법론은 대규모 언어 모델(LLM)을 활용하여 화자의 스타일을 추출하고, 사전 훈련된 언어 모델(PLM)을 통해 대화 행동 이력을 시뮬레이션합니다. 이는 각기 다른 사용자 집단에 맞춰져 풍부하고 개인화된 대화 데이터를 생성하여 더 나은 상호작용을 돕습니다. 연구는 미성년자와의 대화 데이터를 이용하여 맞춤형 데이터 증강을 수행했습니다.

- **Performance Highlights**: 다양한 실험을 통해 우리의 방법론의 효용성을 검증하였으며, 이는 더 적응적이고 포용적인 대화 시스템의 개발 가능성을 높이는 데 기여할 수 있음을 보여주었습니다.



### QUITO-X: An Information Bottleneck-based Compression Algorithm with Cross-Attention (https://arxiv.org/abs/2408.10497)
- **What's New**: 이번 연구는 정보 병목 이론(Information Bottleneck theory)을 활용하여 문맥 압축과 관련된 중요한 토큰을 구별하는 새로운 메트릭을 제시합니다. 기존의 self-information이나 PPL과 같은 메트릭 대신, encoder-decoder 아키텍처의 cross-attention을 사용함으로써 압축 기술의 성능을 개선했습니다.

- **Technical Details**: 이 연구에서는 특징적인 문맥 압축 알고리즘을 개발하였으며, ML 및 AI 분야의 일반적인 사용 사례에 대한 실험을 통해, 이 방법이 기존 방법들에 비해 효과적으로 작동함을 입증했습니다. 특히, DROP, CoQA, SQuAD, Quoref와 같은 데이터셋에서 성능을 평가했습니다. 우리 접근법은 query와 문맥 간의 상호 정보(mutual information)를 최적화하여 중요한 부분을 선택적으로 유지합니다.

- **Performance Highlights**: 실험 결과, 제안한 압축 방법은 이전 SOTA(최신 기술) 대비 약 25% 더 향상된 압축 비율을 보였습니다. 또한, 25%의 토큰을 제거한 상태에서도, 우리의 모델은 비압축 텍스트를 사용하는 대조군의 EM 점수보다 더 높은 결과를 보여주기도 했습니다.



### Analysis of Plan-based Retrieval for Grounded Text Generation (https://arxiv.org/abs/2408.10490)
- **What's New**: 이 논문에서는 언어 모델의 환각(hallucination) 문제를 해결하기 위해 계획(Planning) 기능을 활용하여 정보 검색(retrieval) 과정에서의 영향을 분석합니다. 기존의 언어 모델들은 유명한 사실을 생성할 때 정보가 부족하여 오류를 범하는데, 이를 해결하기 위해 검색 시스템과의 연계를 제안합니다.

- **Technical Details**: 우리는 instruction-tuned LLMs의 계획 기능을 활용하여 언어 모델이 필요한 정보를 효과적으로 검색하고 표현할 수 있는 방법을 탐구합니다. 이 과정에서 검색 쿼리를 생성하기 위한 LLM 계획을 바탕으로 하여 세분화된 사실 정보의 수집이 가능함을 발견했습니다.

- **Performance Highlights**: 모델이 생성하는 긴 형식의 텍스트에 대한 실험을 통해 계획에 의해 안내된 정보 검색이 환각 발생 빈도를 줄이는데 효과적임을 입증하였습니다. 또한, 현재 정보 및 낮은 빈도의 엔터티에 대한 생물 및 사건 설명 작성을 평가하여 접근 방식의 일반성을 demonstrated합니다.



### Enhancing One-shot Pruned Pre-trained Language Models through Sparse-Dense-Sparse Mechanism (https://arxiv.org/abs/2408.10473)
- **What's New**: SDS(Sparse-Dense-Sparse) 프레임워크를 제안하여 사전 훈련된 언어 모델(PLMs)의 성능을 향상시킵니다. 이 프레임워크는 가중치 분포 최적화 관점에서 다루어집니다.

- **Technical Details**: SDS는 세 단계로 구성되어 있습니다: 1단계에서는 기존의 원샷(One-shot) 가지치기 방법으로 덜 중요한 연결을 제거합니다. 2단계에서는 프루닝된 연결을 재활성화하며, 프루닝 친화적인 가중치 분포를 갖춘 밀집 모델로 복원합니다. 마지막으로 3단계에서 두 번째 프루닝을 수행하여 초기 프루닝보다 성능이 뛰어난 프루닝 모델을 만듭니다.

- **Performance Highlights**: SDS는 동일한 희소성 구성 하에서 최신 가지치기 기술인 SparseGPT 및 Wanda를 능가합니다. 데이터셋 Raw-Wikitext2에서 9.13의 퍼플렉시티(perplexity)를 줄였고, OPT-125M에서 평균 2.05%의 정확도를 향상시켰습니다. 또한, 이 모델은 AMD R7 Pro CPU에서 최대 1.87배의 가속화를 달성합니다.



### Goldfish: Monolingual Language Models for 350 Languages (https://arxiv.org/abs/2408.10441)
- **What's New**: 이 논문에서는 Goldfish라는 350개의 저자원 언어에 대해 사전 훈련된 단일 언어 모델 모음을 소개합니다. 이는 기존의 대규모 다국어 모델보다 저자원 언어에 최적화된 솔루션을 제공합니다.

- **Technical Details**: Goldfish 모델은 125M 파라미터까지 지원하며, 5MB, 10MB, 100MB, 1GB의 텍스트 데이터로 훈련되었습니다. 다양한 저자원 언어에서의 성능을 비교하기 위해 FLORES에 있는 204개 언어에서 평가되었습니다.

- **Performance Highlights**: Goldfish 모델은 204개 FLORES 언어 중 98개에서 BLOOM 및 XGLM보다 낮은 perplexities를 기록했으며, 대다수의 저자원 언어에 대해 단순 bigram 모델보다 우수한 결과를 보였습니다. 하지만, 더 큰 다국어 모델들에 비해 논리 추론 성능에서는 하위 성능을 보이는 것으로 나타났습니다.



### Resolving Lexical Bias in Edit Scoping with Projector Editor Networks (https://arxiv.org/abs/2408.10411)
- **What's New**: 본 연구에서는 거리 기반 스코프 기능이 렉시컬 편향(lexical bias)으로 인해 부적절한 입력에 대해 잘못된 수정(misfires)을 초래하는 문제를 다룹니다. 이를 해결하기 위해, PENME(Projector Editor Networks for Model Editing)이라는 새로운 모델 편집 접근 방식을 제안합니다.

- **Technical Details**: PENME는 컴팩트 어댑터와 대조 학습(objective through contrastive learning)으로 훈련된 프로젝터 네트워크로 구성되어 있습니다. 이 네트워크는 수정된 정보와 유사한 의미를 가진 입력의 거리를 재조정하여 relevance를 높이며, 메모리 기반 검색 시스템을 통해 효율적으로 수정을 수행합니다.

- **Performance Highlights**: PENME는 5,000 샘플을 수정하는 동안 뛰어난 안정성을 보여 주었으며, 학습된 프로젝터 공간이 보지 않은 수정, 패러프레이즈(파라프레즈), 그리고 이웃에 대해 잘 일반화됨을 입증하였습니다.



### Value Alignment from Unstructured Tex (https://arxiv.org/abs/2408.10392)
- **What's New**: 본 논문은 LLM(대형 언어 모델)을 비구조화된 텍스트 데이터에서 나타나는 명시적 및 암시적 가치(Value)를 정렬하는 시스템적이고 포괄적인 방법론을 제시합니다. 기존의 수동적인 데이터 수집 방법을 소모하지 않고, 합성 데이터 생성 기술을 활용하여 효율성을 극대화한 점이 주목됩니다.

- **Technical Details**: 제안된 방법론은 합성 데이터 생성(Synthetic Data Generation) 기술을 사용하여 비구조화된 데이터에서 가치를 효과적으로 추출하고, 이를 통해 LLM의 응답을 최적화하는 두 가지 주요 구성 요소로 구성됩니다: 1) 명령어와 선호 데이터 생성, 2) 감독 하에 미세 조정(Supervised Fine-Tuning) 및 선호 최적화(Preference Optimization). 이러한 방식은 전체 문서의 가치를 반영하도록 설계되었습니다.

- **Performance Highlights**: Mistral-7B-Instruct 모델에서 두 가지 사례를 통해, 제안된 방법론이 기존 접근 방식에 비해 높은 성과를 달성했으며, 자동화된 메트릭과 승률을 통해 quantifiable results(정량적 결과)를 제공했음을 보였습니다.



### Beyond Relevant Documents: A Knowledge-Intensive Approach for Query-Focused Summarization using Large Language Models (https://arxiv.org/abs/2408.10357)
Comments:
          Accepted by the 27th International Conference on Pattern Recognition (ICPR 2024)

- **What's New**: 본 논문에서는 기존의 관련 문서에 의존하는 질의 중심 요약(Query-Focused Summarization, QFS) 접근법의 한계를 극복하기 위한 새로운 지식 집약적(knowledge-intensive) 방법론을 제안합니다. 이는 사전 정의된 문서 세트를 필요로 하지 않고, 대규모 지식 코퍼스에서 잠재적으로 관련된 문서를 효율적으로 검색하는 모듈을 포함합니다.

- **Technical Details**: 제안된 방법은 크게 두 가지 주요 구성 요소로 나뉩니다: (1) Retrieval Module - 주어진 질의에 따라 대규모 지식 코퍼스에서 관련 문서를 검색하며, (2) Summarization Controller - 강력한 대형 언어 모델(LLM) 기반 요약기를 통합하여 질의에 관련된 포괄적이고 적합한 요약을 생성합니다. 이를 위해 새로운 데이터 세트를 구성하고, 인간이 주석을 단 관련성을 추가하여 검색 및 요약 성능 평가를 위한 기초를 마련했습니다.

- **Performance Highlights**: 광범위한 실험 결과, 제안된 방법이 기존의 기본 모델들보다 우수한 성능을 보여 주목을 받았습니다. 특히 초기 입력 단계에서 관련 문서를 필요로 하지 않고도 정확한 요약을 생성할 수 있는 능력이 평가되었으며, 이는 다양한 질의 시나리오에서의 실용성을 강조합니다.



### A Survey on Symbolic Knowledge Distillation of Large Language Models (https://arxiv.org/abs/2408.10210)
Comments:
          21 pages, 7 figures

- **What's New**: 이 설문지는 대형 언어 모델(LLMs)에서 상징적 지식 증류(symbolic knowledge distillation)의 emergent(신생) 및 중요한 분야를 탐구합니다. GPT-3와 BERT 모델들이 계속 확대됨에 따라, 이들이 보유한 방대한 지식을 효과적으로 활용하는 방법이 중요해졌습니다. 이 연구는 이러한 복잡한 지식을 보다 상징적이고 구체적인 형태로 정제하는 과정을 중점적으로 다루며, 모델의 투명성(tranparency)과 효율성을 향상시키는 방법을 고안합니다.

- **Technical Details**: LLMs의 지식을 매개변수의 가중치(weights)로 저장하고 있으며, 이를 상징적 방식(symbolic form)으로 전환하는 과정은 지식 전이(knowledge transfer)와 AI 시스템의 설명 가능성(explainability)을 개선하는 데 필수적입니다. 또한, 이 논문에서는 직접적(direct), 다계층(multilevel), 강화 학습(reinforcement learning) 등을 통한 지식 증류 기법을 세 가지로 분류하며, 제안된 프레임워크는 기존 방법론과의 비교도 다룹니다.

- **Performance Highlights**: 현재 연구의 틈새를 식별하고 미래 발전 가능성을 탐구하는 이 설문지는 상징적 지식 증류의 최신 개발을 광범위하게 검토하여, 연구 커뮤니티가 이 분야를 더욱 탐색할 수 있는 귀중한 통찰을 제공합니다. 또한, 이 연구는 지식 심화의 깊이를 보존하면서도 유용성을 높이는 데 필요한 복잡함을 다루고 있습니다.



### FLAME: Learning to Navigate with Multimodal LLM in Urban Environments (https://arxiv.org/abs/2408.11051)
Comments:
          10 pages, 5 figures

- **What's New**: FLAME(FLAMingo-Architected Embodied Agent)를 소개하며, MLLM 기반의 새로운 에이전트이자 아키텍처로, 도시 환경 내 VLN(Visual-and-Language Navigation) 작업을 효율적으로 처리할 수 있도록 설계되었습니다.

- **Technical Details**: FLAME은 세 단계의 조정 기법을 통해 내비게이션 작업에 효과적으로 적응합니다: 1) 단일 인식 조정: 거리 뷰 설명 학습, 2) 다중 인식 조정: 에이전트 궤적 요약 학습, 3) VLN 데이터 세트에 대한 전향적 훈련 및 평가. 이 과정에서 GPT-4를 활용하여 Touchdown 환경을 위한 캡션 및 경로 요약을 합성합니다.

- **Performance Highlights**: FLAME은 기존 방법들보다 탁월한 성능을 보여주며, Touchdown 데이터 세트에서는 7.3%의 작업 완료율 개선을 달성하며 SOTA(state-of-the-art) 기록을 세웠습니다. 또한 Map2seq에서는 3.74% 개선을 이루어 도시 VLN 작업의 새로운 기준을 확립했습니다.



### Dr.Academy: A Benchmark for Evaluating Questioning Capability in Education for Large Language Models (https://arxiv.org/abs/2408.10947)
Comments:
          Accepted to ACL 2024

- **What's New**: 본 연구에서는 LLMs(대형 언어 모델)을 교육자로서 평가하기 위해 질문 생성 능력을 포함한 벤치마크를 소개합니다. 기존 연구가 LLMs를 학습자로 보고 주요 학습 능력을 평가하는 데 집중했던 것과 대조적으로, 이 연구에서는 LLMs가 고품질 교육 질문을 생성할 수 있는지를 평가합니다.

- **Technical Details**: 연구에서는 LLMs의 질문 생성 능력을 평가하기 위해 Anderson과 Krathwohl의 교육적 분류체계를 기반으로 한 6개의 인지 수준이 포함된 Dr.Academy라는 벤치마크를 개발했습니다. 이 벤치마크는 일반, 단일 분야, 다학제 분야 등 세 가지 도메인에서 LLMs가 생성한 질문을 분석합니다. 평가 지표로는 관련성(relevance), 포괄성(coverage), 대표성(representativeness), 일관성(consistency)을 사용합니다.

- **Performance Highlights**: 실험 결과, GPT-4는 일반과 인문, 과학 과정 교육에 상당한 잠재력을 보였으며, Claude2는 다학제 교수에 더 적합한 것으로 나타났습니다. 자동 점수는 인간의 평가와 잘 일치했습니다.



### DELIA: Diversity-Enhanced Learning for Instruction Adaptation in Large Language Models (https://arxiv.org/abs/2408.10841)
Comments:
          8 pages, 5 figures

- **What's New**: 본 논문에서는 LLMs(대규모 언어 모델)의 instruction tuning(지시 조정)에 관한 한계를 다루며, DELIA(다양성 증진 학습 모델)를 통해 바이어스된 특성을 이상적인 특성으로 근사하는 혁신적인 데이터 합성 방법을 제안합니다.

- **Technical Details**: DELIA는 LLM 훈련에서 다양한 데이터의 완충 효과를 활용하여, 특정 작업 형식에 맞게 모델을 조정하기 위한 바이어스된 특성을 이상적인 특성으로 변환합니다. 이 과정에서 우리는 지시-응답 쌍의 다양성을 극대화하여 구조적 변경 없이 훈련 비용을 낮추면서 이상적인 특성에 근접하게 합니다.

- **Performance Highlights**: 실험 결과, DELIA는 일반적인 instruction tuning보다 17.07%-33.41% 향상된 Icelandic-English 번역의 bleurt 점수(WMT-21 데이터셋, gemma-7b-it)와 36.1% 향상된 포맷된 텍스트 생성 정확도를 보여주었습니다. 또한, DELIA는 새로운 특수 토큰의 내부 표현을 이전 의미와 정렬시키는 독특한 능력을 보여, 지식 주입 방법 중에서 큰 진전을 나타냅니다.



### Flexora: Flexible Low Rank Adaptation for Large Language Models (https://arxiv.org/abs/2408.10774)
Comments:
          29 pages, 13 figures

- **What's New**: 이 논문에서는 Low-Rank Adaptation (LoRA) 기술의 한계를 극복하기 위해 'Flexora'라는 새로운 방법을 제안합니다. Flexora는 다양한 다운스트림 작업에서 최적의 성능을 달성하기 위해 fine-tuning을 위한 중요한 레이어를 자동으로 선택하는 유연한 접근 방식을 제공합니다.

- **Technical Details**: Flexora는 레이어 선택 문제를 하이퍼파라미터 최적화 (Hyperparameter Optimization, HPO) 문제로 정의하고, unrolled differentiation (UD) 방법을 사용하여 이를 해결합니다. Flexora는 초기화 단계에서 정의된 하이퍼파라미터를 LoRA 파라미터에 주입하고, 선택된 레이어에 대해서만 백프롭agation과 파라미터 업데이트를 제한하여 계산 오버헤드를 대폭 줄입니다.

- **Performance Highlights**: 다양한 사전 훈련 모델과 자연어 처리 작업에서 수행된 실험 결과에 따르면, Flexora는 기존 LoRA 변형보다 일관되게 성능을 개선시킨 것으로 나타났습니다. 이 연구는 Flexora의 효과성을 뒷받침하기 위해 풍부한 이론적 결과와 여러 ablation 연구를 포함하여 comprehensive understanding을 제공합니다.



### CodeJudge-Eval: Can Large Language Models be Good Judges in Code Understanding? (https://arxiv.org/abs/2408.10718)
Comments:
          Work in progress

- **What's New**: 최근 대규모 언어 모델(LLMs)의 발전으로 코드 생성 능력이 두드러지게 평가되고 있으나, 이러한 평가 방식이 코드 이해 능력을 충분히 반영하지 못할 수 있습니다. 새로운 벤치마크인 CodeJudge-Eval (CJ-Eval)을 소개합니다. 이는 코드 생성을 넘어서 코드 이해 능력을 평가하기 위해 설계되었습니다.

- **Technical Details**: CJ-Eval은 제공된 코드 솔루션의 정답성을 판단하도록 LLMs에게 도전하는 새로운 벤치마크입니다. 다양한 오류 유형과 컴파일 문제를 포괄하며, 다채로운 문제 세트와 세밀한 평가 시스템을 활용하여 기존 벤치마크의 한계를 극복합니다. 특히, 솔루션의 암기가 아닌 실제 코드 이해를 강조합니다.

- **Performance Highlights**: 12종의 잘 알려진 LLMs를 CJ-Eval로 평가한 결과, 최첨단 모델조차도 어려움을 겪는 것으로 나타났습니다. 이는 이 벤치마크가 모델의 코드 이해 능력을 깊이 있게 탐구하는 능력을 갖추고 있음을 잘 보여줍니다.



### Minor SFT loss for LLM fine-tune to increase performance and reduce model deviation (https://arxiv.org/abs/2408.10642)
Comments:
          8 pages, 5 figures

- **What's New**: 본 논문은 SFT(Supervised Fine Tuning)와 RLHF(Reinforcement Learning from Human Feedback)에 대한 새로운 접근 방식을 제안합니다. DPO(Dynamic Preference Optimization)와 MinorDPO에 대한 통찰을 통해, SFT 단계에서 최적화된 모델과 원래 모델 간의 불일치(discrepancy)를 측정하기 위한 훈련 메트릭스를 도입하고, 훈련 효율성을 증가시키고 불일치를 줄일 수 있는 손실 함수 MinorSFT를 제안합니다.

- **Technical Details**: DPO와 MinorDPO는 LLM을 인간의 선호(preference)에 맞추기 위해 설계된 알고리즘입니다. DPO는 각 샘플에 대한 동적 샘플 수준의 계수를 사용해 선호 쌍(preference pair) 간의 거리와 관련된 학습 내용을 조정합니다. MinorSFT는 원래 SFT보다 더 많은 제약을 두어 학습 강도를 조절하여 LLM과 원래 모델 간의 불일치를 줄입니다. 이 손실 함수는 추가적인 하이퍼파라미터와 더 많은 계산 비용을 필요로 하지만, 훈련 효과성을 그리고 성능 결과를 개선할 가능성이 있습니다.

- **Performance Highlights**: MinorSFT를 통해 훈련된 모델은 원래 모델과의 불일치가 감소하고, 결과적으로 더 나은 성능을 보일 것으로 기대됩니다. 기존의 SFT 방식보다 강화된 훈련 메트릭스와 손실 함수를 통해, LLM의 일반성(generality)과 다양성(diversity)을 유지하면서도, 훈련된 모델의 품질을 향상시킬 수 있는 잠재력을 지니고 있습니다.



### Strategist: Learning Strategic Skills by LLMs via Bi-Level Tree Search (https://arxiv.org/abs/2408.10635)
Comments:
          website: this https URL

- **What's New**: 이 논문에서는 Multi-agent 게임을 위한 새로운 방법인 Strategist를 제안합니다. 이 방법은 LLMs를 활용하여 자기 개선 과정을 통해 새로운 기술을 습득하는 방식입니다.

- **Technical Details**: 우리는 Monte Carlo tree search와 LLM 기반 반사를 통해 고급 전략 기술을 학습하는 Self-play 시뮬레이션을 사용합니다. 이 과정은 고차원 전략 수업을 통해 정책을 배우고, 낮은 수준의 실행을 안내하는 상태 평가 방법을 포함합니다.

- **Performance Highlights**: GOPS와 The Resistance: Avalon 게임에서 우리의 방법이 전통적인 강화 학습 기반 접근 방식 및 기존 LLM 기반 기술 학습 방법보다 더 나은 성능을 보였음을 보여주었습니다.



### LLM-Barber: Block-Aware Rebuilder for Sparsity Mask in One-Shot for Large Language Models (https://arxiv.org/abs/2408.10631)
- **What's New**: 이번 연구에서는 기존의 비효율적인 pruning 기법들을 개선하기 위해 LLM-Barber라는 새로운 one-shot pruning 프레임워크를 제안합니다. 이 방법은 retraining이나 weight reconstruction 없이 pruning된 모델의 sparsity mask를 재구축할 수 있습니다.

- **Technical Details**: LLM-Barber는 Self-Attention 및 MLP 블록에서의 block-aware error optimization을 포함하여 글로벌 성능 최적화를 보장합니다. 새로운 pruning 메트릭은 가중치와 기울기를 곱하여 weight importance를 평가하며, 효과적인 pruning 결정을 내립니다.

- **Performance Highlights**: LLM-Barber는 7B에서 13B 파라미터를 가진 LLaMA 및 OPT 모델을 단일 A100 GPU에서 30분 만에 효율적으로 pruning하며, perplexity 및 zero-shot 성능에서 기존 방법들을 뛰어넘는 결과를 보여 줍니다.



### Synergistic Approach for Simultaneous Optimization of Monolingual, Cross-lingual, and Multilingual Information Retrieva (https://arxiv.org/abs/2408.10536)
Comments:
          15 pages, 2 figures, 13 tables

- **What's New**: 이 논문에서는 다국어 환경에서 정보 검색의 성능을 동시에 개선하면서 언어 편향을 완화할 수 있는 새로운 하이브리드 배치 훈련 전략을 제안합니다.

- **Technical Details**: 하이브리드 배치 훈련 접근법은 단일 언어 및 교차 언어 질문-답변 쌍 배치를 혼합하여 다국어 언어 모델을 미세 조정하는 방법입니다. 이는 모델이 다양한 언어 쌍에서 훈련되어 언어에 구애받지 않는 표현을 학습하도록 유도합니다. 사용된 데이터셋은 XQuAD-R, MLQA-R, MIRACL 벤치마크로, dual-encoder 구조와 contrastive learning 방법론이 적용되었습니다.

- **Performance Highlights**: 제안된 방법은 제로샷 검색(Zero-Shot Retrieval) 성능을 높이며, 다양한 언어 및 검색 작업에서 경쟁력 있는 결과를 지속적으로 달성합니다. 특히, 기존의 단일 언어 또는 교차 언어 훈련만을 이용한 모델과 비교하여 성능이 우수합니다. 또한, 모델이 훈련 데이터에 포함되지 않은 언어에 대해서도 강력한 제로샷 일반화 성능을 보입니다.



### Event Stream based Sign Language Translation: A High-Definition Benchmark Dataset and A New Algorithm (https://arxiv.org/abs/2408.10488)
Comments:
          First Large-scale and High-Definition Benchmark Dataset for Event-based Sign Language Translation

- **What's New**: 본 논문에서는 Sign Language Translation (SLT)의 새로운 접근 방법으로 Event streams를 제안합니다. 이 방법은 조명, 손 움직임 등으로부터 영향을 덜 받으며, 빠르고 높은 동적 범위를 갖추고 있어 정확한 번역을 가능하게 합니다. 또한, 새로운 Event-CSL 데이터셋을 제안하여 연구에 기여합니다.

- **Technical Details**: Event-CSL 데이터셋은 14,827개의 고화질 비디오로 구성되며, 14,821개의 glosses 및 2,544개의 중국어 단어가 포함되어 있습니다. 이 데이터는 다양한 실내 및 실외 환경에서 수집되었으며, Convolutional Neural Networks (CNN) 기반의 ResNet18 네트워크와 Mamba 네트워크를 결합한 하이브리드 아키텍처를 사용하여 SLT 성능을 향상시킵니다.

- **Performance Highlights**: 본 연구에서는 Event-CSL 데이터셋과 기존 SLT 모델들을 벤치마킹하여, SLT 분야에서의 효과적인 성능 개선을 입증하였습니다. 이 하이브리드 CNN-Mamba 아키텍처는 기존의 Transformer 기반 모델보다 더 나은 성능을 보입니다.



### LeCov: Multi-level Testing Criteria for Large Language Models (https://arxiv.org/abs/2408.10474)
- **What's New**: 리서치에서는 Large Language Models (LLMs)의 신뢰성 문제를 해결하기 위한 LeCov라는 새로운 다수의 수준 테스트 기준을 제안합니다.

- **Technical Details**: LeCov는 LLM의 내부 구성 요소인 attention mechanism, feed-forward neurons, 그리고 uncertainty를 기반으로 합니다. 총 9가지 테스트 기준이 포함되어 있으며, 테스트 우선 순위(test prioritization)와 커버리지 기반 테스트(coverage-guided testing)와 같은 두 가지 시나리오에 적용됩니다.

- **Performance Highlights**: 세 개의 모델과 네 개의 데이터셋을 이용한 실험적 평가 결과, LeCov가 유용하고 효과적임을 입증했습니다.



### Tracing Privacy Leakage of Language Models to Training Data via Adjusted Influence Functions (https://arxiv.org/abs/2408.10468)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 개인 정보 유출 문제를 해결하기 위한 새로운 접근 방식을 제안합니다. 특히, 그랜트 노름이 큰 토큰의 영향을 줄이는 'Heuristically Adjusted Influence Functions (HAIF)'를 도입하여 추적 정확도를 향상시켰습니다.

- **Technical Details**: HAIF는 기존의 Influence Functions (IFs)의 제약 사항을 극복하기 위해 설계되었습니다. 이 방법은 큰 그랜트 노름을 가진 토큰의 영향을 줄여 비밀 샘플 추적의 정확성을 개선합니다. 연구진은 PII-E 및 PII-CR이라는 두 개의 데이터셋을 사용하여 HAIF의 성능을 평가했습니다.

- **Performance Highlights**: HAIF는 PII-E 데이터셋에서 20.96%에서 73.71%로, PII-CR 데이터셋에서는 3.21%에서 45.93%로 추적 정확도를 향상시킴으로써 가장 최신 기술(SOTA)과 비교하여 유의미한 개선을 보였습니다. 또한, HAIF는 CLUECorpus2020 실제 사전 훈련 데이터에서도 뛰어난 내구성을 보여줍니다.



### Federated Learning of Large ASR Models in the Real World (https://arxiv.org/abs/2408.10443)
- **What's New**: 이 논문에서는 1억 3000만 개의 파라미터를 가진 Conformer 기반의 자동 음성 인식(ASR) 모델을 연합 학습(Federated Learning, FL)을 통해 성공적으로 학습한 사례를 제시합니다. 이는 FL의 실제 적용 사례 중에서 가장 큰 모델을 학습한 것이며, FL이 ASR 모델 품질을 향상시킬 수 있는 첫 번째 연구입니다.

- **Technical Details**: 연합 학습을 활용하여 ASR 모델을 학습하는 데 있어 다양한 효율적 기법들을 통합하여 사용했습니다. 모델 크기를 줄이기 위해 gradient checkpointing, 온라인 모델 압축(Online Model Compression, OMC) 및 부분 모델 학습(Partial Model Training) 기법을 적용했습니다. 사용자의 교정 데이터를 활용하여 데이터와 라벨의 품질을 개선하는 Weighted Client Aggregation (WCA) 알고리즘을 설계했습니다.

- **Performance Highlights**: 훈련 효율성이 크게 향상되었으며, 이는 메모리 사용량과 서버 및 클라이언트 간의 데이터 전송 크기로 측정되었습니다. 또한, FL 모델의 단어 오류율(Word Error Rate, WER)도 효과적으로 개선되었습니다.



### Development of an AI Anti-Bullying System Using Large Language Model Key Topic Detection (https://arxiv.org/abs/2408.10417)
- **What's New**: 이 논문은 인공지능(AI) 기반의 사이버 괴롭힘 방지 시스템 개발과 평가에 대한 내용을 다룹니다. 이 시스템은 소셜 미디어에서의 조직적 괴롭힘 공격을 식별하고 분석하여 대응 방안을 제시합니다.

- **Technical Details**: 이 시스템은 대규모 언어 모델(LLM)을 활용하여 괴롭힘 공격에 대한 전문가 시스템 기반의 네트워크 모델을 구성합니다. 이를 통해 괴롭힘 공격의 특성을 분석하고, 소셜 미디어 회사에 보고 메시지를 생성하는 등의 대응 활동을 지원합니다.

- **Performance Highlights**: LLM의 모델 구성에 대한 효능이 분석되었으며, 이 시스템이 괴롭힘 방지 및 대응에 효과적임을 보여줍니다.



### Narrowing the Gap between Vision and Action in Navigation (https://arxiv.org/abs/2408.10388)
- **What's New**: 이 논문에서는 Vision and Language Navigation in the Continuous Environment (VLN-CE) 에이전트의 비주얼 환경을 보다 효과적으로 학습하고 탐색 성능을 향상시키기 위해 저수준 행동 디코더와 고수준 행동 예측을 결합한 새로운 접근 방식을 제안합니다.

- **Technical Details**: 저수준 행동 디코더를 통합하여 에이전트가 고수준 뷰 선택과 동시에 저수준의 행동 시퀀스를 생성할 수 있도록 하는 이중 행동 모듈을 소개합니다. 이를 통해 에이전트는 선택된 비주얼 뷰를 저수준 제어와 연결할 수 있습니다. 또한, 시각적 표현을 포함한 waypoint predictor를 향상시켜, 객관적인 의미를 반영하고 장애물을 명시적으로 마스킹하여 더 효과적인 관점을 생성합니다.

- **Performance Highlights**: 실험 결과, 이 접근 방식은 VLN-CE 에이전트의 탐색 성능 지표를 개선하는 데 성공하였으며, 고수준 및 저수준 행동 모두에서 강력한 기준선을 초월하는 성능 향상을 보였습니다.



### SEAL: Systematic Error Analysis for Value ALignmen (https://arxiv.org/abs/2408.10270)
Comments:
          28 pages, 17 Figures, 8 Tables

- **What's New**: 이 연구에서는 인간의 피드백(Feedback)을 기반으로 강화 학습(RLHF)의 내부 메커니즘을 이해하기 위해 새로운 메트릭을 도입했습니다. 이 메트릭은 feature imprint, alignment resistance, alignment robustness로 구성되어 있습니다.

- **Technical Details**: RLHF는 언어 모델(LM)을 인간의 가치에 맞춰 조정하기 위해 보상 모델(RM)을 훈련하는 과정입니다. 연구에서는 alignment dataset을 target features(원하는 가치)와 spoiler features(원치 않는 개념)로 분류하고, RM 점수를 이들 특성과 회귀 분석하여 feature imprint를 정량화했습니다.

- **Performance Highlights**: 연구 결과, 목표 feature에 대한 RM의 보상이 큼을 발견하였고, 26%의 경우에서 RM이 인간의 선호와 일치하지 않았습니다. 또한, RM은 입력이 약간 변경되었을 때 민감하게 반응하여, misalignment가 악화되는 모습을 보였습니다.



### VyAnG-Net: A Novel Multi-Modal Sarcasm Recognition Model by Uncovering Visual, Acoustic and Glossary Features (https://arxiv.org/abs/2408.10246)
- **What's New**: 본 연구에서는 대화에서의 sarcasm 인식을 위한 새로운 접근법인 VyAnG-Net을 제안합니다. 이 방법은 텍스트, 오디오 및 비디오 데이터를 통합하여 더 신뢰할 수 있는 sarcasm 인식을 가능하게 합니다.

- **Technical Details**: 이 방법은 lightweight depth attention 모듈과 self-regulated ConvNet을 결합하여 시각 데이터의 핵심 특징을 집중적으로 분석하고, 텍스트 데이터에서 문맥에 따른 중요 정보를 추출하기 위한 attentional tokenizer 기반 전략을 사용합니다. Key contributions로는 subtitles에서 glossary content의 유용한 특징을 추출하는 attentional tokenizer branch, 비디오 프레임에서 주요 특징을 얻는 visual branch, 음향 콘텐츠에서 발화 수준의 특징 추출, 여러 모달리티에서 획득한 특징을 융합하는 multi-headed attention 기반 feature fusion branch가 포함됩니다.

- **Performance Highlights**: MUSTaRD 벤치마크 비디오 데이터셋에서 speaker dependent 및 speaker independent 환경에서 각각 79.86% 및 76.94%의 정확도로 기존 방법들에 비해 우수함을 입증하였습니다. 또한, MUStARD++ 데이터셋의 보지 않은 샘플을 통해 VyAnG-Net의 적응성을 평가하는 교차 데이터셋 분석도 수행하였습니다.



### A General-Purpose Device for Interaction with LLMs (https://arxiv.org/abs/2408.10230)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)과 고급 하드웨어 통합에 대한 연구로, LLM과의 향상된 상호작용을 위해 설계된 범용 장치 개발에 중점을 두고 있습니다. 전통적인 가상 어시스턴트(VA)의 한계를 극복하고, LLM을 통합한 새로운 지능형 어시스턴트(IA) 시대를 여는 데 그 목적이 있습니다.

- **Technical Details**: 이번 연구에서는 LLM의 요구 사항에 맞춘 독립적인 하드웨어 플랫폼에서 IA를 배치하여 복잡한 명령을 이해하고 실행하는 능력을 향상시키고자 하며, 여러 형태의 입력을 통합하는 혁신적인 프레임워크를 설계했습니다. 이 프레임워크는 음성 입력을 향상시키기 위해 로컬 전처리를 집중적으로 개선하여 더 정확한 입력을 달성하는 것을 목표로 합니다.

- **Performance Highlights**: 제안된 장치는 다차원 입력을 처리할 수 있는 능력을 갖추고 있으며, 음성 및 비디오 센서와 환경 센서를 포함하여 사용자와의 상호작용을 개선합니다. 로컬 캐싱 기능을 통해 처리 속도를 높이고 응답 시간을 감소시키는 동시에 시스템 효율성을 크게 향상시키는 성과를 보였습니다.



New uploads on arXiv(cs.IR)

### Vector Symbolic Open Source Information Discovery (https://arxiv.org/abs/2408.10734)
- **What's New**: 이 논문은 환경을 고려하여 CJIIM(Combined, Joint, Intra-governmental, Inter-agency and Multinational) 작전에서의 데이터 공유를 개선하기 위한 혁신적인 접근 방식을 제시합니다. 특히, DDIL(Denied, Degraded, Intermittent and Low bandwidth) 환경에서 효과적인 데이터 발견을 위한 새로운 통합 모델을 소개합니다.

- **Technical Details**: 이 연구에서는 대형 언어 모델(transformers)과 벡터 기호 아키텍처(Vector Symbolic Architectures, VSA)를 통합하여 데이터를 더 효율적으로 발견하고 정렬할 수 있는 방법을 탐구합니다. 특히, VSA는 1-10k비트 크기의 고도로 압축된 이진 벡터를 사용하여 DDIL 환경에서의 활용 가능성을 높입니다. FAISS를 이용해 고차원 벡터 인덱싱 및 매칭을 수행하여 더 효율적으로 정보를 처리할 수 있도록 합니다.

- **Performance Highlights**: 제안된 접근 방식은 CJIIM 파트너들 간의 데이터 발견을 더욱 용이하게 하며, OSINF 데이터 발견 포털의 프로토타입을 통해 데이터 소스를 최소한의 메타데이터 큐레이션과 낮은 통신 대역폭으로 공유할 수 있도록 하는 기능을 보여줍니다. 이 연구는 방어 분야에서의 메타데이터 상호작용성을 개선하고 데이터 소비 및 활용을 촉진하는 데 기여할 수 있습니다.



### Accelerating the Surrogate Retraining for Poisoning Attacks against Recommender Systems (https://arxiv.org/abs/2408.10666)
Comments:
          Accepted by RecSys 2024

- **What's New**:  최근 연구에 따르면, 추천 시스템은 데이터 오염 공격에 취약하며, 공격자는 조작된 가짜 사용자 인터랙션을 주입하여 목표 아이템을 홍보한다. 기존의 공격 방법은 손상된 데이터에 대한 대체 추천 모델을 반복적으로 재학습시키는 방법을 사용하나, 시간이 많이 소요된다. 저자들은 Gradient Passing (GP)이라는 새로운 기술을 도입하여, 사용자-아이템 쌍 간에 기울기를 명시적으로 전달함으로써 재학습을 가속화하고 더 효과적인 공격을 가능하게 한다.

- **Technical Details**:  이 연구에서는 추천 시스템의 재학습 과정을 분석한 결과, 하나의 사용자 혹은 아이템의 표현을 변경하면 사용자-아이템 상호작용 그래프를 통해 연쇄적인 효과가 발생함을 발견하였다. GP 기술은 하나의 업데이트로도 여러 번의 학습 반복에 해당하는 효과를 달성할 수 있도록 설계되었으며, 이렇게 하여 공격자의 목표 추천 모델에 더 가까운 근사값을 제공한다.

- **Performance Highlights**:  GP를 통합한 상태에서의 실험 결과, 평균 공격 효과iveness는 각각 29.57%, 18.02%, 177.21% 증가했으며, 시간 비용은 43.27%, 40.54%, 26.67% 감소하였다. 이는 GP 기법이 실제 데이터 세트에서 효율성과 효과성을 입증하는 데 성공했음을 보여준다.



### CoRA: Collaborative Information Perception by Large Language Model's Weights for Recommendation (https://arxiv.org/abs/2408.10645)
- **What's New**: 이 연구에서는 추천 작업에 대한 LLM(대규모 언어 모델)의 새로운 적응 방식을 제안하였습니다. 기존 방법들이 LLM의 본래 세계 지식 및 기본 능력을 감소시키는 문제를 해결하는 새로운 접근법인 CoRA(Collaborative LoRA)를 소개합니다.

- **Technical Details**: CoRA는 협동 가중치 생성기를 사용하여 협동 정보를 LLM의 파라미터 공간과 일치시키고, 이를 점진적인 가중치로 표현하여 LLM의 출력을 업데이트합니다.

- **Performance Highlights**: 광범위한 실험을 통해 CoRA가 기존 LLMRec 방법과 전통적인 협동 필터링 방법에 비해 상당한 성능 향상을 보여줌을 확인하였습니다.



### Task-level Distributionally Robust Optimization for Large Language Model-based Dense Retrieva (https://arxiv.org/abs/2408.10613)
- **What's New**: 이번 논문에서는 대용량 언어 모델 기반의 Dense Retrieval (LLM-DR) 최적화를 위한 새로운 작업 수준의 분포적으로 강건한 최적화(tDRO) 알고리즘을 제안합니다. 이 알고리즘은 다양한 도메인에서의 일반화 능력을 향상시키기 위해 각 작업의 데이터 분포를 엔드 투 엔드 방식으로 재조정합니다.

- **Technical Details**: tDRO는 도메인 가중치를 매개변수화하고 스케일된 도메인 경량화에 따라 업데이트합니다. 그런 다음 최적화된 가중치는 LLM-DR 세부 조정 과정에 전달되어 더 강건한 검색기를 훈련하는 데 사용됩니다. 이 방법은 서로 다른 배치 샘플링 전략을 조정하는 데 중점을 둡니다.

- **Performance Highlights**: 실험 결과, tDRO 최적화를 적용한 후 대규모 검색 기준에서 최적의 개선을 보여주었습니다. 이 과정에서 데이터셋 사용량이 최대 30% 감소하였습니다.



### Synergistic Approach for Simultaneous Optimization of Monolingual, Cross-lingual, and Multilingual Information Retrieva (https://arxiv.org/abs/2408.10536)
Comments:
          15 pages, 2 figures, 13 tables

- **What's New**: 이 논문에서는 다국어 환경에서 정보 검색의 성능을 동시에 개선하면서 언어 편향을 완화할 수 있는 새로운 하이브리드 배치 훈련 전략을 제안합니다.

- **Technical Details**: 하이브리드 배치 훈련 접근법은 단일 언어 및 교차 언어 질문-답변 쌍 배치를 혼합하여 다국어 언어 모델을 미세 조정하는 방법입니다. 이는 모델이 다양한 언어 쌍에서 훈련되어 언어에 구애받지 않는 표현을 학습하도록 유도합니다. 사용된 데이터셋은 XQuAD-R, MLQA-R, MIRACL 벤치마크로, dual-encoder 구조와 contrastive learning 방법론이 적용되었습니다.

- **Performance Highlights**: 제안된 방법은 제로샷 검색(Zero-Shot Retrieval) 성능을 높이며, 다양한 언어 및 검색 작업에서 경쟁력 있는 결과를 지속적으로 달성합니다. 특히, 기존의 단일 언어 또는 교차 언어 훈련만을 이용한 모델과 비교하여 성능이 우수합니다. 또한, 모델이 훈련 데이터에 포함되지 않은 언어에 대해서도 강력한 제로샷 일반화 성능을 보입니다.



### Efficient and Deployable Knowledge Infusion for Open-World Recommendations via Large Language Models (https://arxiv.org/abs/2408.10520)
Comments:
          arXiv admin note: text overlap with arXiv:2306.10933

- **What's New**: 이 논문은 REKI라는 새로운 Recommendation 시스템을 제안하고 있으며, 이를 통해 대규모 언어 모델(LLMs)이 사용자와 아이템에 대한 외부 지식을 효과적으로 활용할 수 있도록 합니다. 이 시스템은 closed-loop 시스템의 한계를 극복하고 open-world knowledge를 추구하는 접근 방식을 채택합니다.

- **Technical Details**: REKI는 지식 추출을 위해 'factorization prompting'을 도입하였으며, 개별 지식 추출(individual knowledge extraction)과 집합 지식 추출(collective knowledge extraction)을 통해 다양한 시나리오를 위해 최적화되었습니다. 이 과정에서 자동화된 벡터 생성 및 변환을 위한 하이브리드 전문가 통합 네트워크(hybridized expert-integrated network)가 사용됩니다.

- **Performance Highlights**: REKI는 최첨단 모델들과 비교하여 우수한 성능을 발휘하며, Huawei의 뉴스 및 음악 추천 플랫폼에 배포 후 각각 7% 및 1.99%의 개선효과를 보였습니다.



### Enhanced document retrieval with topic embeddings (https://arxiv.org/abs/2408.10435)
Comments:
          Accepted to AICT 2024

- **What's New**: 본 논문은 문서 검색 시스템에서 주제 정보를 활용한 새로운 벡터화(vectorization) 방법을 제안합니다. 이 방법은 LLM만 사용하는 애플리케이션에 비해 환각(hallucination) 비율을 낮추는 RAG 시스템의 효율성을 개선하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 제안된 방법은 (1) 원본 문서 임베딩과 전체 주제에서 생성된 주제 임베딩을 결합하여 새로운 문서 임베딩을 생성하는 방식과 (2) 주제를 찾고 그 안에서 문서를 찾는 두 단계의 검색 방식을 포함합니다. 이 두 가지 방법이 산업 환경에서 성공적으로 구현되었습니다.

- **Performance Highlights**: Azerbaijani 법률 데이터셋을 사용한 실험 결과, 주제 임베딩을 추가함으로써 주제가 분리 및 구분되는 성능이 향상되었습니다. Silhouette Coefficient, Davies–Bouldin index 및 Calinski–Harabasz index를 통해 평가된 결과는, 주제 정보를 포함할 경우 더 나은 클러스터링 결과를 보여주었습니다.



### Joint Modeling of Search and Recommendations Via an Unified Contextual Recommender (UniCoRn) (https://arxiv.org/abs/2408.10394)
Comments:
          3 pages, 1 figure

- **What's New**: 이번 연구에서는 검색(Search) 및 추천(Recommendation) 시스템을 통합할 수 있는 딥러닝(deep learning) 모델을 제안합니다. 이는 별도의 모델 개발로 인한 유지보수의 복잡성을 줄이고 기술 부채(technical debt)를 완화하는 데 기여합니다.

- **Technical Details**: 제안된 모델은 사용자 ID, 쿼리(query), 국가(country), 소스 엔티티 ID(source entity id), 작업(task)과 같은 정보를 기반으로합니다. 이 모델은 입력층에 다양한 컨텍스트를 통합하여 다양한 상황에 따라 그에 맞는 결과를 생성할 수 있습니다. 구조적으로는 잔여 연결(residual connections)과 특징 교차(feature crossing)를 포함하며, 이진 교차 엔트로피 손실(binary cross entropy loss)과 아담 최적화기(Adam optimizer)를 사용합니다.

- **Performance Highlights**: UniCoRn이라는 통합 모델을 통해 검색과 추천 성능 모두에서 파리티(parity) 또는 성능 향상을 달성했습니다. 비개인화된 모델에서 개인화된 모델로 전환했을 때 검색과 추천 각각 7% 및 10%의 실적 향상을 보였습니다.



### ColBERT Retrieval and Ensemble Response Scoring for Language Model Question Answering (https://arxiv.org/abs/2408.10808)
Comments:
          This work has been submitted to the 2024 IEEE Globecom Workshops for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 이번 연구는 통신 네트워크를 위한 소규모 언어 모델인 Phi-2와 Falcon-7B의 성능을 향상시키기 위한 특별한 챌린지를 기반으로 개발된 질문 응답(QA) 시스템을 소개합니다. 두 모델 모두 특정 도메인 내에서 질문을 처리할 때 효과성을 입증하며, Phi-2는 81.9%의 높은 정확도를 기록했습니다.

- **Technical Details**: 연구에서는 Retrieval-Augmented Generation (RAG) 기법을 활용하여 3GPP 문서에서 관련 맥락을 검색하고, 질문에 대해 기술 약어의 전체 이름을 포함하는 프롬프트를 설계했습니다. 각 모델에 대한 특정 방법론을 바탕으로, Phi-2는 LoRA 어댑터를 사용해 훈련되었고 Falcon-7B는 생성된 응답을 기반으로 평가 메커니즘을 개발했습니다.

- **Performance Highlights**: 솔루션은 두 모델 모두 각각의 트랙에서 최고의 성과를 달성했습니다. Phi-2는 개인 평가 데이터셋에서 81.9%의 정확도를, Falcon-7B는 57.3%의 정확도를 기록했습니다. 이는 해당 도메인에서 이전에 보고된 성과에 비해 유의미한 개선입니다.



### Multilingual Non-Factoid Question Answering with Silver Answers (https://arxiv.org/abs/2408.10604)
- **What's New**: MuNfQuAD라는 새로운 다국어 질의응답 데이터셋이 소개되었습니다. 이는 비사실 기반(non-factoid) 질문을 포함하고 있으며, BBC 뉴스 기사의 질문과 해당 단락을 사용하여 구성되었습니다.

- **Technical Details**: MuNfQuAD는 38개 언어를 포괄하며, 370,000개 이상의 QA 쌍으로 이루어져 있습니다. 수동 주석을 통해 790개의 QA 쌍이 검토되었으며, 98%의 질문이 해당하는 은유적 답변(silver answer)으로 답변될 수 있음을 확인했습니다. 이 연구에서는 Answer Paragraph Selection (APS) 모델이 최적화되어 기존 기준보다 뛰어난 성과를 보였습니다.

- **Performance Highlights**: APS 모델은 MuNfQuAD 테스트셋에서 80%의 정확도와 72%의 매크로 F1 점수를 기록하였으며, 골든 세트(golden set)에서는 각각 72%와 66%의 성과를 달성했습니다. 또한, 이 모델은 은유적 레이블(silver labels)로 최적화 된 후에도 골든 세트 내 특정 언어를 효과적으로 일반화할 수 있었습니다.



### Target-Prompt Online Graph Collaborative Learning for Temporal QoS Prediction (https://arxiv.org/abs/2408.10555)
Comments:
          This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 서비스 지향 아키텍처에서 Quality of Service (QoS)의 정확한 예측은 신뢰성을 유지하고 사용자 만족도를 향상시키는 데 매우 중요합니다. 본 논문에서 제안하는 TOGCL (Target-Prompt Online Graph Collaborative Learning) 프레임워크는 고차원 잠재 협력 관계를 활용하고, 사용자-서비스 호출에 맞춰 동적으로 특성 학습을 조정하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: TOGCL 프레임워크는 동적 사용자-서비스 호출 그래프를 사용하여 과거 상호작용을 모델링합니다. 이 그래프 위에서 target-prompt graph attention network를 활용하여 각 시간 조각에서 사용자와 서비스의 온라인 심층 잠재 특성을 추출합니다. 또한 multi-layer Transformer encoder를 사용하여 시간적 특성 진화 패턴을 발견합니다.

- **Performance Highlights**: WS-DREAM 데이터셋을 기반으로 한 광범위한 실험에서 TOGCL은 여러 메트릭에서 기존 최첨단 방법을 크게 초월하여, 예측 정확도를 최대 38.80% 향상시켰음을 보여줍니다. 이는 TOGCL의 효과적인 시간적 QoS 예측 능력을 입증합니다.



### Analysis of Plan-based Retrieval for Grounded Text Generation (https://arxiv.org/abs/2408.10490)
- **What's New**: 이 논문에서는 언어 모델의 환각(hallucination) 문제를 해결하기 위해 계획(Planning) 기능을 활용하여 정보 검색(retrieval) 과정에서의 영향을 분석합니다. 기존의 언어 모델들은 유명한 사실을 생성할 때 정보가 부족하여 오류를 범하는데, 이를 해결하기 위해 검색 시스템과의 연계를 제안합니다.

- **Technical Details**: 우리는 instruction-tuned LLMs의 계획 기능을 활용하여 언어 모델이 필요한 정보를 효과적으로 검색하고 표현할 수 있는 방법을 탐구합니다. 이 과정에서 검색 쿼리를 생성하기 위한 LLM 계획을 바탕으로 하여 세분화된 사실 정보의 수집이 가능함을 발견했습니다.

- **Performance Highlights**: 모델이 생성하는 긴 형식의 텍스트에 대한 실험을 통해 계획에 의해 안내된 정보 검색이 환각 발생 빈도를 줄이는데 효과적임을 입증하였습니다. 또한, 현재 정보 및 낮은 빈도의 엔터티에 대한 생물 및 사건 설명 작성을 평가하여 접근 방식의 일반성을 demonstrated합니다.



### LSVOS Challenge 3rd Place Report: SAM2 and Cutie based VOS (https://arxiv.org/abs/2408.10469)
- **What's New**: 이번 논문은 최신 Video Object Segmentation (VOS) 기술에 대한 새로운 접근 방식을 제시합니다. SAM2와 Cutie라는 두 가지 최신 모델을 조합하여 VOS의 도전과제를 해결하고 있으며, 다양한 하이퍼파라미터가 비디오 인스턴스 분할 성능에 미치는 영향을 탐구하였습니다.

- **Technical Details**: 우리의 VOS 접근 방식은 최첨단 메모리 기반 방법을 활용하여 이전에 분할된 프레임에서 메모리 표현을 생성하고, 새 쿼리 프레임이 이 메모리에 접근하여 세그멘테이션에 필요한 특징을 조회합니다. 특히, SAM2는 이미지와 비디오 분할을 위한 통합 모델로, 메모리 모듈을 통해 이전 프레임의 맥락을 활용하여 마스크 예측을 생성하고 정교화합니다. Cutie는 반자동 준지도 학습 환경에서 작동하여 도전적인 시나리오를 처리합니다.

- **Performance Highlights**: 우리의 접근 방식은 LSVOS 챌린지 VOS 트랙 테스트 단계에서 J&F 점수 0.7952를 달성하여 전체 3위에 랭크되었습니다. 이는 고해상도 이미지 처리, 고급 객체 추적, 오클루전 예측 전략을 통해 높은 정확도로 성과를 얻어낸 결과입니다.



### Beyond Relevant Documents: A Knowledge-Intensive Approach for Query-Focused Summarization using Large Language Models (https://arxiv.org/abs/2408.10357)
Comments:
          Accepted by the 27th International Conference on Pattern Recognition (ICPR 2024)

- **What's New**: 본 논문에서는 기존의 관련 문서에 의존하는 질의 중심 요약(Query-Focused Summarization, QFS) 접근법의 한계를 극복하기 위한 새로운 지식 집약적(knowledge-intensive) 방법론을 제안합니다. 이는 사전 정의된 문서 세트를 필요로 하지 않고, 대규모 지식 코퍼스에서 잠재적으로 관련된 문서를 효율적으로 검색하는 모듈을 포함합니다.

- **Technical Details**: 제안된 방법은 크게 두 가지 주요 구성 요소로 나뉩니다: (1) Retrieval Module - 주어진 질의에 따라 대규모 지식 코퍼스에서 관련 문서를 검색하며, (2) Summarization Controller - 강력한 대형 언어 모델(LLM) 기반 요약기를 통합하여 질의에 관련된 포괄적이고 적합한 요약을 생성합니다. 이를 위해 새로운 데이터 세트를 구성하고, 인간이 주석을 단 관련성을 추가하여 검색 및 요약 성능 평가를 위한 기초를 마련했습니다.

- **Performance Highlights**: 광범위한 실험 결과, 제안된 방법이 기존의 기본 모델들보다 우수한 성능을 보여 주목을 받았습니다. 특히 초기 입력 단계에서 관련 문서를 필요로 하지 않고도 정확한 요약을 생성할 수 있는 능력이 평가되었으며, 이는 다양한 질의 시나리오에서의 실용성을 강조합니다.



### OPDR: Order-Preserving Dimension Reduction for Semantic Embedding of Multimodal Scientific Data (https://arxiv.org/abs/2408.10264)
- **What's New**: 이번 논문은 다중 모드 과학 데이터 관리에서 가장 일반적인 작업 중 하나인 k-최근접 이웃(KNN) 검색의 차원을 줄이는 방법을 제안합니다. 본 연구는 기존의 고차원 임베딩 벡터를 낮은 차원으로 변환하면서 KNN 검색의 결과를 유지하는 Order-Preserving Dimension Reduction (OPDR) 기법을 도입하였습니다.

- **Technical Details**: OPDR은 임베딩 벡터의 차원을 줄이는 데 있어 KNN 유사성을 정량화하는 수식적 접근 방식을 사용합니다. 이 방법은 KNN 검색을 위한 측정 함수를 정의하고, 이를 통해 전역 메트릭스(global metric)와 닫힌 형태의 함수를 도출하여 차원의 수와 데이터 포인트 수 간의 관계를 밝혀냅니다. 또한 다양한 차원 축소 방법(PCA, MDS) 및 거리 메트릭(Euclidean, cosine, Manhattan)과 통합하여 실제 과학 응용 프로그램에 쉽게 적용될 수 있도록 설계되었습니다.

- **Performance Highlights**: 제안된 OPDR 방법은 4개의 대표적인 다중 모드 과학 데이터 세트에 대해 광범위한 평가를 통해 효과성을 입증하였으며, 이 결과는 높은 차원에서 저차원으로의 변환 시 KNN의 보존을 정확하게 캡처할 수 있음을 보여줍니다. 이로 인해 복잡한 과학 응용 분야에서 고급 AI 기술을 활용하는 데 새로운 통찰을 제공합니다.



### AI Transparency in Academic Search Systems: An Initial Exploration (https://arxiv.org/abs/2408.10229)
- **What's New**: AI 강화 학술 검색 시스템의 투명성에 대한 연구가 이루어지었습니다. 10개의 시스템을 분석하여 투명성 수준의 차이를 발견했습니다.

- **Technical Details**: 10개의 AI-enhanced academic search systems의 웹사이트를 질적 내용 분석(qualitative content analysis) 방법으로 조사하였고, 투명성(transparency) 수준을 평가했습니다.

- **Performance Highlights**: 5개의 시스템은 메커니즘에 대한 자세한 정보를 제공하고, 3개는 부분적인 정보만 제공하며, 2개는 거의 정보를 제공하지 않습니다. 이는 연구자의 책임(responsibility) 및 재현성(reproducibility) 문제에 대한 우려를 불러일으킵니다.



New uploads on arXiv(cs.CV)

### Prompt-Guided Image-Adaptive Neural Implicit Lookup Tables for Interpretable Image Enhancemen (https://arxiv.org/abs/2408.11055)
Comments:
          Accepted to ACM Multimedia 2024

- **What's New**: 본 논문은 해석 가능한 이미지 향상 기술과 이에 따른 새로운 아키텍처, Prompt-Guided Image-Adaptive Neural Implicit Lookup Table (PG-IA-NILUT)를 제안합니다. 기존의 정의된 필터 대신에 학습 가능한 필터를 통해 각 필터에 예를 들어 'Exposure', 'Contrast'와 같은 이해하기 쉬운 이름을 부여합니다.

- **Technical Details**: 새로운 이미지 적응형 신경 무의식 조회 테이블(IA-NILUT)의 구조가 도입되며, 다층 퍼셉트론 (Multilayer Perceptron, MLP)을 사용하여 입력 특성 공간에서 출력 색 공간으로의 변환을 암묵적으로 정의합니다. 'Prompt guidance loss'를 활용해 각 필터에 해석 가능한 이름을 할당하는 방식을 제안합니다.

- **Performance Highlights**: 제안된 방법은 기존의 정의된 필터 기반 방법보다 성능이 우수함을 보여주며, 사용자에게 이해하기 쉬운 필터 효과를 제공합니다. 실험은 FiveK 및 PPR10K 데이터셋을 사용하여 수행되었으며, 높은 성능과 해석 가능한 필터를 달성하였습니다.



### NeCo: Improving DINOv2's spatial representations in 19 GPU hours with Patch Neighbor Consistency (https://arxiv.org/abs/2408.11054)
Comments:
          Preprint. The webpage is accessible at: this https URL

- **What's New**: 이번 논문에서는 NeCo(Patch Neighbor Consistency)라는 새로운 자기 지도 학습 신호를 제안하여 사전 훈련된 표현을 개선하고자 한다. 해당 방법은 학생 모델과 교사 모델 간의 패치 레벨 이웃 일관성을 강화하는 새로운 훈련 손실을 도입한다.

- **Technical Details**: 자기 지도 학습의 일환으로, 해당 방법은 DINOv2와 같은 사전 훈련된 표현 위에 차별적 정렬(differentiable sorting) 방법을 적용하여 학습 신호를 부트스트랩(bootstrap)하고 개선한다. NeCo는 이미지 수준에서 이미 훈련된 모델을 시작으로 하여 이를 적응시키는 방식으로 빠르고 효율적인 해결책을 제공한다. 이 과정에서 패치 간 유사성을 유지해야 하며, 가장 가까운 이웃을 정렬하여 학습의 일관성을 확보한다.

- **Performance Highlights**: 제안된 방법은 다양한 모델과 데이터세트에서 뛰어난 성능을 보였으며, ADE20k 및 Pascal VOC에서 비모수적(non-parametric) 맥락 세분화(in-context segmentation)에서 각각 +5.5% 및 +6%의 성능 향상을 보였고, COCO-Things 및 COCO-Stuff에서 선형 세분화 평가에서 각각 +7.2% 및 +5.7%를 기록하였다.



### FLAME: Learning to Navigate with Multimodal LLM in Urban Environments (https://arxiv.org/abs/2408.11051)
Comments:
          10 pages, 5 figures

- **What's New**: FLAME(FLAMingo-Architected Embodied Agent)를 소개하며, MLLM 기반의 새로운 에이전트이자 아키텍처로, 도시 환경 내 VLN(Visual-and-Language Navigation) 작업을 효율적으로 처리할 수 있도록 설계되었습니다.

- **Technical Details**: FLAME은 세 단계의 조정 기법을 통해 내비게이션 작업에 효과적으로 적응합니다: 1) 단일 인식 조정: 거리 뷰 설명 학습, 2) 다중 인식 조정: 에이전트 궤적 요약 학습, 3) VLN 데이터 세트에 대한 전향적 훈련 및 평가. 이 과정에서 GPT-4를 활용하여 Touchdown 환경을 위한 캡션 및 경로 요약을 합성합니다.

- **Performance Highlights**: FLAME은 기존 방법들보다 탁월한 성능을 보여주며, Touchdown 데이터 세트에서는 7.3%의 작업 완료율 개선을 달성하며 SOTA(state-of-the-art) 기록을 세웠습니다. 또한 Map2seq에서는 3.74% 개선을 이루어 도시 VLN 작업의 새로운 기준을 확립했습니다.



### OpenScan: A Benchmark for Generalized Open-Vocabulary 3D Scene Understanding (https://arxiv.org/abs/2408.11030)
- **What's New**: 본 논문에서는 기존의 Object Class 중심의 Open-vocabulary 3D scene understanding (OV-3D)을 넘어서, Generalized Open-Vocabulary 3D Scene Understanding (GOV-3D)라는 보다 도전적인 작업을 소개합니다. 이는 객체 속성과 관련된 언어적 쿼리로 표현된 열린 지식 세트를 탐구합니다.

- **Technical Details**: GOV-3D 작업은 3D 포인트 클라우드 장면과 텍스트 쿼리를 입력받아 가장 잘 맞는 객체의 3D 마스크를 예측합니다. OpenScan이라는 새로운 벤치마크를 구축하여, 객체 속성을 아우르는 8가지 언어적 측면을 포함합니다. 이 벤치마크는 두 가지 기본적인 제한 사항을 해결합니다.

- **Performance Highlights**: 기존 OV-3D 모델들은 기본 객체 클래스 이해에서는 우수하지만, affordance 및 material과 같은 객체 속성의 추상적인 이해에서 성능이 크게 저하됩니다. 이는 이러한 모델들이 기존의 방법론에 의존하는 한, GOV-3D 작업의 요구를 충족하는 데 어려움을 겪고 있으며, 이를 해결하기 위한 새로운 접근 방법이 필요하다는 것을 강조합니다.



### MegaFusion: Extend Diffusion Models towards Higher-resolution Image Generation without Further Tuning (https://arxiv.org/abs/2408.11001)
Comments:
          Technical Report. Project Page: this https URL

- **What's New**: 이 논문에서는 MegaFusion이라는 새로운 접근 방식을 소개하며, 기존의 확산 기반 텍스트-이미지 생성 모델이 추가적인 미세 조정 없이 효율적으로 고해상도 이미지를 생성할 수 있도록 확장합니다.

- **Technical Details**: MegaFusion은 고해상도를 위한 트렁크 및 릴레이(truncate and relay) 전략을 활용하여 서로 다른 해상도에서의 디노이징(denoising) 과정을 연결합니다. 또한, 이 모델은 확장된 수렴(convolutions) 및 노이즈 재조정(noise re-scheduling)을 통합하여 고해상도 이미지 생성을 위한 모델의 고유성을 조정합니다.

- **Performance Highlights**: MegaFusion은 기존 모델이 메가픽셀 및 다양한 종횡비의 이미지를 생성할 수 있는 능력을 크게 향상시키며, 원래 컴퓨팅 비용의 약 40%로 이를 수행합니다.



### SenPa-MAE: Sensor Parameter Aware Masked Autoencoder for Multi-Satellite Self-Supervised Pretraining (https://arxiv.org/abs/2408.11000)
Comments:
          GCPR 2024

- **What's New**: 본 논문에서는 다양한 위성의 멀티스펙트럴 신호를 센서 파라미터와 결합하여 이미지 임베딩으로 변환하는 SenPa-MAE라는 트랜스포머 아키텍처를 소개합니다. 이 모델은 서로 다른 스펙트럼 또는 기하학적 특성을 가진 위성 이미지에 대해 사전 학습을 수행할 수 있습니다.

- **Technical Details**: SenPa-MAE는 센서 파라미터를 효과적으로 인코딩하기 위한 모듈 및 데이터 증강 전략을 포함합니다. 이 접근법은 다양한 센서 간의 차이를 설명하고, 여러 센서와의 교차 교육 및 센서 독립적 추론을 가능하게 합니다.

- **Performance Highlights**: 우리의 실험 결과, SenPa-MAE는 다양한 멀티 센서 환경에서 토지 피복 예측 작업에 있어 우수한 성능을 보여줍니다. 이는 센서에 구애받지 않고 EO 기초 모델을 생성할 수 있는 잠재력을 나타냅니다.



### Facial Demorphing via Identity Preserving Image Decomposition (https://arxiv.org/abs/2408.10993)
- **What's New**: 이번 연구에서는 얼굴 변형 (morphing) 이미지의 구조를 분해하여 관련된 고유 결함 (bonafides) 정보를 회복하는 새로운 방법을 제안합니다. 기존의 방법들은 참조 이미지 (reference image)를 필요로 하였지만, 우리의 접근 방식은 참조 없는 (reference-free) 방식으로 진행됩니다.

- **Technical Details**: 우리는 얼굴 변형을 ill-posed decomposition 문제로 간주하며, 제안한 방법은 얼굴 이미지를 여러 개의 정체성 보존(feature preserving) 구성 요소로 분해합니다. 이후 조합 네트워크(merger network)가 이러한 구성 요소를 결합하여 고유 결함을 회복합니다. 이 방식은 기존 방법보다 유연성을 제공하며 고성능을 보여줍니다.

- **Performance Highlights**: CASIA-WebFace, SMDD 및 AMSL 데이터셋을 사용한 실험 결과, 제안한 방법이 정의(clarity)와 충실도(fidelity) 측면에서 고품질의 고유 결함을 재구성하는 데 효과적임을 확인하였습니다.



### Multichannel Attention Networks with Ensembled Transfer Learning to Recognize Bangla Handwritten Charecter (https://arxiv.org/abs/2408.10955)
- **What's New**: 본 연구에서는 Bengali(벵골어) 손글씨 인식의 성능을 개선하기 위해 다중 채널 주의(multi-channel attention)와 앙상블(ensemble) 전이 학습(transfer learning)으로 구성된 합성곱 신경망(CNN) 구조를 제안하였습니다.

- **Technical Details**: 연구는 두 개의 CNN 분기(Inception Net과 ResNet)를 사용하여 특징(feature)을 생성하고, 이들을 연결(concatenate)하여 앙상블 특징을 구축했습니다. 이후, 주의 모듈(attention module)을 적용하여 앙상블 특징으로부터 문맥 정보를 생성하고, 마지막으로 분류 모듈(classification module)을 통해 특징을 정제하여 인식 성능을 높였습니다.

- **Performance Highlights**: CAMTERdb 3.1.2 데이터 세트를 사용하여 모델을 평가한 결과, 원본 데이터 세트에서 92%의 정확도와 전처리된 데이터 세트에서 98.00%의 정확도를 달성했습니다.



### HiRED: Attention-Guided Token Dropping for Efficient Inference of High-Resolution Vision-Language Models in Resource-Constrained Environments (https://arxiv.org/abs/2408.10945)
Comments:
          Preprint

- **What's New**: 이 논문은 High-Resolution Early Dropping (HiRED)이라는 새로운 토큰 드롭핑 기법을 제안하여 자원 제약이 있는 환경에서도 고해상도 Vision-Language Models (VLMs)을 효율적으로 처리할 수 있는 방법을 제공합니다. HiRED는 추가적인 훈련 없이 기존 VLM에 통합할 수 있는 플러그 앤 플레이(plug-and-play) 방식으로, 높은 정확도를 유지합니다.

- **Technical Details**: HiRED는 초기 레이어의 비전 인코더의 주의(attention) 메커니즘을 활용하여 각 이미지 파티션의 시각적 콘텐츠를 평가하고, 토큰 예산을 할당합니다. 마지막 레이어의 주의 메커니즘을 통해 가장 중요한 비주얼 토큰을 선택하여 할당된 예산 내에서 나머지를 드롭합니다. 이 방법은 연산 효율성을 높이기 위해 이미지 인코딩 단계에서 토큰을 드롭하여 LLM의 입력 시퀀스 길이를 줄이는 방식으로 작동합니다.

- **Performance Highlights**: HiRED를 NVIDIA TESLA P40 GPU에서 LLaVA-Next-7B에 적용한 결과, 20% 토큰 예산으로 생성된 토큰 처리량이 4.7배 증가했으며, 첫 번째 토큰 생성 지연(latency)이 15초 감소하고, 단일 추론에 대해 2.3GB의 GPU 메모리를 절약했습니다. 또한, HiRED는 기존 베이스라인보다 훨씬 높은 정확도를 달성했습니다.



### A Closer Look at Data Augmentation Strategies for Finetuning-Based Low/Few-Shot Object Detection (https://arxiv.org/abs/2408.10940)
- **What's New**: 이 논문은 데이터가 부족한 상황에서 저비용의 물체 탐지에 대한 성능과 에너지 효율성을 평가하는 포괄적인 실증 연구를 수행합니다. 특히, 사용자 정의 데이터 증가(Data Augmentation, DA) 방법과 자동 데이터 증가 선택 전략이 경량 물체 탐지기와 결합했을 때의 효과를 중점적으로 분석합니다.

- **Technical Details**: 여기서 Efficiency Factor를 사용하여 성능과 에너지 소비를 모두 고려한 분석을 통해 데이터 증가 전략의 유효성을 이해하려고 시도합니다. 연구는 세 가지 벤치마크 데이터셋에서 DA 전략이 경량 물체 탐지기의 파인튜닝 과정 중 에너지 효율성 및 일반화 능력에 미치는 영향을 평가합니다.

- **Performance Highlights**: 결과적으로 데이터 증가 전략의 성능 향상이 에너지 사용 증가로 인해 가려지는 경우가 많다고 밝혀집니다. 따라서 데이터가 부족한 상황에서, 더 에너지 효율적인 데이터 증가 전략 개발이 필요하다는 결론에 도달합니다.



### Large Point-to-Gaussian Model for Image-to-3D Generation (https://arxiv.org/abs/2408.10935)
Comments:
          10 pages, 9 figures, ACM MM 2024

- **What's New**: 본 논문에서는 2D 이미지를 기반으로 초기 포인트 클라우드(Point Cloud)에서 가우시안 파라미터(Gaussian parameters)를 생성하기 위한 새로운 대형 Point-to-Gaussian 모델을 제안합니다. 이를 통해 이미지에서 3D 자산으로의 생성 과정이 크게 개선되었습니다.

- **Technical Details**: 모델의 핵심 구성 요소로는 Attention 메커니즘, Projection 메커니즘, 그리고 포인트 특징 추출기(Point feature extractor)로 구성된 APP 블록이 있습니다. 이 블록은 이미지 특징과 포인트 클라우드 특징을 융합하는 역할을 합니다. 최종적으로 생성된 가우시안 파라미터는 멀티 헤드 가우시안 디코더를 통해 생성되고, 일반적인 가우시안 스플래팅(Gaussian Splatting)을 통해 새로운 뷰 이미지가 렌더링됩니다.

- **Performance Highlights**: GSO 및 Objaverse 데이터셋에서 실시된 질적 및 양적 실험을 통해, 제안한 방법이 기존의 최첨단 기법과 비교해 동등한 성능을 달성했음을 보여주며, 적은 데이터셋으로도 높은 효과를 얻을 수 있음을 증명합니다.



### SDI-Net: Toward Sufficient Dual-View Interaction for Low-light Stereo Image Enhancemen (https://arxiv.org/abs/2408.10934)
- **What's New**: 본 논문은 저조도 스테레오 이미지 향상을 위한 새로운 모델 SDI-Net을 제안합니다. SDI-Net은 왼쪽 및 오른쪽 보기 간의 상호작용을 극대화하여 더 나은 이미지 향상 결과를 도출하는 데 초점을 맞추고 있습니다.

- **Technical Details**: SDI-Net의 구조는 두 개의 UNet으로 구성된 인코더-디코더 쌍으로, 저조도 이미지를 노멀 라이트 이미지로 변환하는 매핑 기능을 학습합니다. 핵심 모듈인 Cross-View Sufficient Interaction Module (CSIM)은 주의 메커니즘을 통해 양안 시점 간의 상관관계를 최대한 활용합니다. CSIM은 두 개의 주요 구성 요소인 Cross-View Attention Interaction Module (CAIM) 및 Pixel and Channel Attention Block (PCAB)을 포함하여 높은 정밀도로 양쪽 뷰를 정렬하고 다양한 밝기 레벨을 고려하여 세밀한 텍스처를 복원합니다.

- **Performance Highlights**: SDI-Net은 Middlebury 및 Holopix50k 데이터셋에서 기존의 저조도 스테레오 이미지 향상 방법보다 우수한 성능을 보여주며, 여러 단일 이미지 저조도 향상 방법에 대해서도 우위를 점하고 있습니다.



### CrossFi: A Cross Domain Wi-Fi Sensing Framework Based on Siamese Network (https://arxiv.org/abs/2408.10919)
- **What's New**: 최근 Wi-Fi 센싱(Wi-Fi sensing)의 연구가 활발히 진행되고 있으며, 특히 CrossFi라는 새로운 프레임워크가 제안되었습니다. CrossFi는 siamese 네트워크에 기반하여 인도메인과 크로스도메인 시나리오에서 우수한 성능을 보이며, few-shot, zero-shot 및 new-class 시나리오를 포함한 다양한 환경에서 사용될 수 있습니다.

- **Technical Details**: CrossFi의 핵심 구성 요소는 CSi-Net이라는 샘플 유사도 계산 네트워크로, 주의 메커니즘(attention mechanism)을 사용하여 유사도 정보를 캡처하는 구조로 개선되었습니다. 추가로 Weight-Net을 개발하여 각 클래스의 템플릿을 생성하여 크로스 도메인 시나리오에서도 작동하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, CrossFi는 제스처 인식(task)에서 인도메인 시나리오에서 98.17%의 정확도를 달성하며, one-shot 크로스 도메인 시나리오에서 91.72%, zero-shot 크로스 도메인 시나리오에서 64.81%, one-shot new-class 시나리오에서 84.75%의 성능을 기록하였습니다. 이 성과들은 다양한 시나리오에서 CrossFi의 우수성을 입증합니다.



### ShapeSplat: A Large-scale Dataset of Gaussian Splats and Their Self-Supervised Pretraining (https://arxiv.org/abs/2408.10906)
- **What's New**: 3D Gaussian Splatting (3DGS)가 다양한 비전 작업에서 3D 표현의 표준 방식으로 자리 잡았습니다. 이 논문에서는 3DGS를 활용한 연구를 용이하게 하기 위해 대규모 데이터셋인 ShapeSplat을 구축하였습니다.

- **Technical Details**: ShapeSplat 데이터셋은 ShapeNet과 ModelNet 데이터셋을 기반으로하며, 87개의 고유 카테고리에서 65,000개의 객체로 구성되어 있습니다. 이 데이터셋은 TITAN XP GPU에서 2 GPU 년에 해당하는 컴퓨팅 자원을 사용하여 생성되었습니다. 우리는 이 데이터셋을 비지도 사전 훈련과 감독된 세분화(finetuning) 작업에 활용합니다. 

또한 Gaussian-MAE를 소개하며, 이는 Gaussian 파라미터를 통한 표현 학습의 독특한 이점을 강조합니다.

- **Performance Highlights**: 실험 결과, 최적화된 GS 중심의 분포가 균일하게 샘플링된 포인트 클라우드와 유의미하게 다르다는 것을 보여주었습니다. 이는 분류 작업에서 성능 저하를 초래하지만, 세분화 작업에서는 향상된 결과를 보였습니다. 추가적인 Gaussian 파라미터를 활용하기 위해 정상화된 특성 공간에서 Gaussian 기능 그룹화 및 splats pooling layer를 제안하여 유사한 Gaussians를 효과적으로 그룹화하고 임베드하여 세부 조정 작업에서 눈에 띄는 향상을 가져왔습니다.



### A Grey-box Attack against Latent Diffusion Model-based Image Editing by Posterior Collaps (https://arxiv.org/abs/2408.10901)
Comments:
          21 pages, 7 figures, 10 tables

- **What's New**: 최근 생성형 AI의 발전, 특히 Latent Diffusion Models (LDMs)이 이미지 합성 및 조작 분야에 혁신적인 변화를 가져왔습니다. 본 논문에서는 Posterior Collapse Attack (PCA)라는 새로운 공격 기법을 제안하여 현재의 방어 방법의 한계를 극복하고자 합니다.

- **Technical Details**: PCA는 VAE(Variational Autoencoder)의 posterior collapse 현상을 활용하여, LDM의 생성 품질을 현저히 저하시키는 방식으로 설계되었습니다. 이 방법은 black-box 접근법이며, 전체 모델 정보 없이도 효과를 발휘합니다. PCA는 LDM의 인코더를 통해 최소한의 모델 특화 지식으로도 실행 가능하며, 3.39%의 매개변수만을 사용하여 성능 저하를 유도합니다.

- **Performance Highlights**: PCA는 다양한 구조와 해상도에 걸쳐 LDM 기반 이미지 편집을 효과적으로 차단하며, 기존 기법보다 더 높은 전이성과 강인성을 보여줍니다. 실험 결과 PCA는 낮은 실행 시간과 VRAM으로 우수한 변이 효과를 달성하였고, 기존 기술들에 비해 더욱 일반화된 해결책을 제공합니다.



### ViLReF: A Chinese Vision-Language Retinal Foundation Mod (https://arxiv.org/abs/2408.10894)
- **What's New**: 이 연구는 451,956개의 망막 이미지와 해당 진단 텍스트 보고서로 구성된 데이터셋을 사용하여 ViLReF라는 망막 기초 모델을 개발하였습니다. 새로운 손실 함수인 Weighted Similarity Coupling Loss를 제안하여 전이 학습 시 false negative 문제를 해결하고, 배치 확장 모듈을 통해 결측 샘플을 보충합니다.

- **Technical Details**: ViLReF는 이미지-텍스트 쌍의 유사성을 학습하기 위해 전문가의 지식을 활용하여 라벨을 자동으로 추출하는 알고리즘을 개발했습니다. self-supervised contrastive learning 방식을 적용하며, 데이터의 미세한 특징 또한 학습하도록 설계되어 있습니다. 동적 메모리 큐를 유지하여 샘플의 양을 확장하고, 여기서 진단 텍스트의 의미론적 표현을 효과적으로 학습합니다.

- **Performance Highlights**: 다양한 데이터셋에 대한 실험 결과, ViLReF는 zero-shot 및 fine-tuning 기법에서 우수한 성능을 보여주며, 기존 전이 학습 모델들보다 뛰어난 의미론적 이해 및 일반화 능력을 입증하였습니다.



### Low-Quality Image Detection by Hierarchical VAE (https://arxiv.org/abs/2408.10885)
Comments:
          ICCV 2023, Workshop on Uncertainty Estimation for Computer Vision

- **What's New**: 본 연구는 저품질 이미지의 비지도 학습 방식 탐지 임무를 새롭게 제안하며, 다양한 형태의 열화(deterioration)에 대해 자동적으로 저품질 이미지를 탐지하는 방법을 제공함.

- **Technical Details**: 제안된 방법은 계층적 변분 오토인코더(hierarchical variational autoencoder)를 이용한 부분 복원(partial reconstruction)의 실패 관찰을 기반으로 하며, 원본 이미지와 복원된 이미지 사이의 불일치(discrepancy)를 비교하여 저품질 이미지를 판별한다. 이를 통해 저품질 이미지의 간접적인 시각적 단서를 제공하여 사용자가 이를 인식할 수 있도록 한다.

- **Performance Highlights**: 제안된 방법은 FFHQ-256 데이터셋을 사용한 실험에서 여러 비지도 학습 기반의 이상치 탐지(out-of-distribution detection) 방법과 비교할 때, 특히 노이즈(noise), 블러(blur), JPEG 압축(compression) 및 채도(saturation)에서 최고의 탐지 성능을 보여주었다.



### Open 3D World in Autonomous Driving (https://arxiv.org/abs/2408.10880)
- **What's New**: 본 논문은 LIDAR 센서에서 획득한 3D 포인트 클라우드 데이터를 텍스트 정보와 통합하여 자율주행 맥락 내에서 객체를 직접 위치 추적하고 식별하는 새로운 접근 방식을 제안합니다. 이 방법은 시각적 입력에 대한 이해를 강화하고, novel textual inputs에 대한 적응력을 향상시킵니다.

- **Technical Details**: 'Open3DWorld'라 이름 붙여진 이 접근법은 BEV(비행기에서 본 시점) 지역 특징과 텍스트 특징을 효율적으로 융합하는 방법을 통해 3D 개체 탐지를 위한 open vocabulary detection을 지원합니다. 이 구조는 LIDAR와 텍스트 데이터를 융합하는 새로운 방법을 포함하여, 개체 위치 지정과 식별을 직관적으로 가능하게 합니다.

- **Performance Highlights**: NuScenes-T 데이터셋에서의 실험을 통해 제안된 방법의 효율성을 입증하였으며, Lyft Level 5 데이터셋에서 제로샷(zero-shot) 성능 또한 검증되었습니다. 이 연구는 자율주행 기술의 발전에 기여하며, 3D 환경에서 open vocabulary 인지 기술을 향상시키는데 중요한 역할을 합니다.



### V-RoAst: A New Dataset for Visual Road Assessmen (https://arxiv.org/abs/2408.10872)
- **What's New**: 이 논문은 도로 안전 평가를 위한 새로운 접근법인 V-RoAst를 제안하고, 전통적인 Convolutional Neural Networks (CNNs)의 한계를 극복하기 위해 Vision Language Models (VLMs)를 사용합니다.

- **Technical Details**: 연구는 상황에 맞는 프롬프트 엔지니어링(prompt engineering)을 최적화하고, Gemini-1.5-flash 및 GPT-4o-mini와 같은 고급 VLM들을 평가하여 도로 속성을 검사하는 방법을 개발합니다. Mapillary에서 크라우드소싱된 이미지를 사용하여 도로 안전 수준을 효율적으로 추정하는 확장 가능한 솔루션을 제공합니다.

- **Performance Highlights**: 이 접근법은 자원 부족으로 어려움을 겪는 지역 이해관계자들을 위해 설계되었으며, 훈련 데이터 없이 비용 효율적이고 자동화된 방법을 제공하여 전 세계 도로 안전 평가에 기여할 수 있습니다.



### Perception-guided Jailbreak against Text-to-Image Models (https://arxiv.org/abs/2408.10848)
Comments:
          8 pages

- **What's New**: 이 논문은 인간의 지각을 기반으로 한 텍스트-이미지(T2I) 모델 공격 방법인 PGJ(perception-guided jailbreak)를 제안합니다. 이 방법은 특정 T2I 모델에 의존하지 않는 모델 프리(model-free) 방법으로, 안전한 문구를 사용하여 부적절한 이미지를 생성하는 공격 프롬프트를 효율적으로 찾는 데 초점을 맞추고 있습니다.

- **Technical Details**: PGJ 접근 방식은 인간의 지각 혼동(perceptual confusion) 개념을 사용하여, 텍스트 의미는 일치하지 않지만 인간의 인식에서 유사한 안전한 문구를 찾아 이를 공격 프롬프트의 대체로 활용합니다. 이를 위해 LLM(Large Language Model)의 능력을 활용하여 PSTSI(Perception-Semantics Text Substitution for Inference) 원칙을 적용하며, 이 원칙은 안전한 대체 문구와 목표 불법 단어가 인간의 인식에서 유사하지만 텍스트 의미에서는 일관성이 없도록 합니다.

- **Performance Highlights**: 연구는 6개의 오픈 소스 및 상업용 T2I 모델을 대상으로 수천 개의 프롬프트를 사용하여 PGJ의 효과성을 입증했습니다. 실험 결과 PGJ는 비논리적 토큰을 포함하지 않으면서 자연스러운 공격 프롬프트를 생성하는 데 성공하였습니다.



### Harmonizing Attention: Training-free Texture-aware Geometry Transfer (https://arxiv.org/abs/2408.10846)
Comments:
          10 pages, 6 figures

- **What's New**: 이 연구에서는 texture-aware geometry transfer를 위한 새로운 접근 방식인 Harmonizing Attention을 제안합니다. 이 방법은 diffusion 모델을 활용하여 매개변수 조정 없이도 다양한 재료의 기하학적 특징을 효과적으로 전달할 수 있게 해줍니다.

- **Technical Details**: Harmonizing Attention은 Texture-aligning Attention과 Geometry-aligning Attention의 두 가지 자가 주의 메커니즘을 포함합니다. 이 구조는 기존 이미지에서 기하학적 정보와 텍스쳐 정보를 쿼리하여, 재료에 독립적인 기하학적 특징을 포착하고 전달할 수 있습니다. 주의 메커니즘에서 수정된 자가 주의층이 inversion 및 generation 과정에서 통합됩니다.

- **Performance Highlights**: 실험 결과, Harmonizing Attention을 사용한 방법이 기존의 이미지 조화 기술과 비교했을 때 더 조화롭고 현실감 있는 합성 이미지를 생성함을 확인했습니다.



### CoVLA: Comprehensive Vision-Language-Action Dataset for Autonomous Driving (https://arxiv.org/abs/2408.10845)
- **What's New**: 이 논문에서는 복잡하고 예기치 않은 자율 주행 시나리오를 성공적으로 탐색하기 위한 다중 모달 대형 언어 모델(Multi-modal Large Language Models, MLLMs)의 새로운 적용 가능성을 제시합니다. CoVLA(Comprehensive Vision-Language-Action) 데이터셋을 소개하며, 이는 10,000개의 실제 주행 장면과 80시간 이상의 비디오 자료로 구성됩니다.

- **Technical Details**: CoVLA 데이터셋은 자동화된 데이터 처리 및 캡션 생성 파이프라인을 통해 생성되며, 차량 내 감지기 데이터에 기반하여 기존 데이터셋보다 더 큰 규모와 많은 주석을 제공합니다. 이 데이터셋은 비전, 언어, 행동을 통합한 VLA 모델의 개발을 가능하게 하며, 자율 주행 시나리오에서 수행 능력을 평가하는 데 사용됩니다.

- **Performance Highlights**: CoVLA-Agent 모델은 CoVLA 데이터셋에서 훈련된 VLA 모델로, 일관되며 정밀한 예측을 수행하는 능력을 보여줍니다. 이 연구의 결과는 고급 판단이 필요한 복잡한 상황에서도 일관된 언어와 행동 출력을 생성할 수 있는 모델의 강점을 강조합니다.



### Aligning Object Detector Bounding Boxes with Human Preferenc (https://arxiv.org/abs/2408.10844)
Comments:
          Accepted paper at the ECCV 2024 workshop on Assistive Computer Vision and Robotics (ACVR)

- **What's New**: 이번 연구에서는 객체 탐지 분야에서 대규모 경계 상자(large bounding boxes)가 더 작은 경계 상자(small bounding boxes)보다 인간이 선호한다는 사실을 과거 연구 결과를 바탕으로 제시합니다. 특히 향상된 경계 상자를 자동으로 탐지된 객체 상자와 맞추고 이러한 과정이 결과적으로 인간의 품질 인식을 어떻게 향상시키는지 연구했습니다.

- **Technical Details**: 세 가지 일반적으로 사용되는 객체 탐지기를 분석하여 이들 각각이 큰 상자와 작은 상자를 동일한 빈도로 예측한다는 것을 발견했습니다. 아울러, 비대칭 경계 상자 회귀 손실(asymmetric bounding box regression loss) 함수를 제안하여 큰 상자가 작보다 더 많이 예측되도록 유도했습니다. 우리의 연구 결과, 이 비대칭 손실로 세밀하게 조정된 객체 탐지기가 인간의 선호에 더 잘 맞으며 фикс된 스케일링 비율보다 더 긍정적인 평가를 받았습니다.

- **Performance Highlights**: 인간 피험자(N=123) 연구를 통해, 사람들은 상응하는 평균 정밀도(average precision, AP)가 0에 가까울지라도 1.5배 또는 2배로 확대된 객체 탐지를 선호한다는 것을 발견했습니다. 이 결과는 인간의 지각처럼 객체 형태와 같은 특정 특성에 영향을 받을 수 있음을 나타냅니다.



### Detecting Wildfires on UAVs with Real-time Segmentation Trained by Larger Teacher Models (https://arxiv.org/abs/2408.10843)
- **What's New**: 이번 연구는 UAV(무인 항공기)를 이용한 조기 산불 탐지 기술을 제안하며, 작은 카메라와 컴퓨터를 장착한 UAV가 환경의 피해를 줄이는 데 도움을 줄 수 있음을 보여줍니다. 특히, 제한된 고대역 모바일 네트워크 환경에서 실시간으로 탐지가 가능한 방법을 다룹니다.

- **Technical Details**: 연구에서는 경량화된 세분화(Segmentation) 모델을 통해 산불 연기의 픽셀 단위 존재를 정확히 추정하는 방법을 제시합니다. 이 모델은 바운딩 박스 레이블을 사용한 제로샷(Zero-shot) 학습 기반의 감독 하에 훈련되며, PIDNet이라는 최종 모델을 사용하여 실시간으로 연기를 인식합니다. 특히, 이 방법은 수동 주석이 있는 데이터셋을 기반으로 하여 63.3%의 mIoU(Mean Intersection over Union) 성능을 기록했습니다.

- **Performance Highlights**: PIDNet 모델은 UAV에 장착된 NVIDIA Jetson Orin NX 컴퓨터에서 약 11 fps로 실시간으로 작동하며, 실제 산불 발생 시 연기를 효과적으로 인지했습니다. 이 연구는 제로샷 학습을 통해 산불 탐지 모델의 훈련에 있어 낮은 장벽으로 접근할 수 있음을 증명했습니다.



### ZebraPose: Zebra Detection and Pose Estimation using only Synthetic Data (https://arxiv.org/abs/2408.10831)
Comments:
          8 pages, 5 tables, 7 figures

- **What's New**: 본 연구는 딥 러닝 작업에서 비정상적인 도메인에서 레이블이 지정된 이미지의 부족을 해결하기 위해 합성 데이터(synthetic data)를 사용하는 새로운 접근 방식을 제안합니다. 특히, 야생 동물인 얼룩말의 2D 자세 추정을 위해 처음으로 사용되는 합성 데이터셋을 생성하였으며, 기존의 실제 이미지 및 스타일 제약없이 탐지(detection)와 자세 추정(pose estimation) 모두를 수행할 수 있도록 하였습니다.

- **Technical Details**: 연구에서는 3D 포토리얼리스틱(simulator) 시뮬레이터를 활용하여 얼룩말의 합성 데이터를 생성하였으며, YOLOv5를 기반으로 한 탐지 모델과 ViTPose+ 모델을 사용하여 2D 키포인트(keypoints)를 추정합니다. 이를 통해 104K개의 수작업으로 레이블이 지정된 고해상도 이미지를 포함한 새로운 데이터셋을 제공하며, 이 데이터셋은 UAV를 통해 촬영된 이미지를 포함합니다. 또한, 합성 데이터만을 사용하여 실제 이미지에 대한 일반화 능력을 평가하였습니다.

- **Performance Highlights**: 연구 결과, 합성 데이터로 학습된 모델이 실제 얼룩말 이미지에 대해 일관되게 일반화된다는 것을 보여주었습니다. 이에 따라 최소한의 실제 이미지 데이터로도 말에 대한 2D 자세 추정으로 쉽게 일반화할 수 있음을 입증하였습니다.



### Trustworthy Compression? Impact of AI-based Codecs on Biometrics for Law Enforcemen (https://arxiv.org/abs/2408.10823)
- **What's New**: 이번 연구에서는 AI 기반 압축이 생체 인식(iris, fingerprint 및 soft-biometric) 이미지에 미치는 영향을 분석하였습니다. 특히, AI 압축이 생체 인식 성능에 미치는 영향을 평가하여 신뢰성 있는 결과를 도출했습니다.

- **Technical Details**: 연구에서 사용한 AI 압축 방법은 Hific, Mbt, Ms로, 각 방법의 다양한 변형을 사용했습니다. Mbt 및 Ms는 MSE 손실 및 MS-SSIM 손실로 훈련해 비교했습니다. 구조적 유사도(SSIM) 분석을 통해 압축 전후의 이미지 품질을 평가했습니다.

- **Performance Highlights**: AI 압축에 의해 iris 인식은 크게 악영향을 받는 반면, fingerprint 인식은 상대적으로 강인한 성능을 보였습니다. 섬세한 구조는 iris 및 섬유 이미지에서 손실되며, 문신 이미지는 복잡한 패턴 및 색상 전환에서 재구성이 충분히 이루어지지 못하는 경향을 보였습니다.



### Constructing a High Temporal Resolution Global Lakes Dataset via Swin-Unet with Applications to Area Prediction (https://arxiv.org/abs/2408.10821)
- **What's New**: 이 논문은 GLAKES-Additional이라는 확장된 호수 데이터베이스를 소개하며, 1990년부터 2021년까지 152,567개의 호수에 대한 2년마다의 경계 및 면적 측정을 제공합니다. 이는 기존의 GLAKES 데이터베이스에 대한 중복된 시계열 데이터를 제공함으로써 호수 변화의 정량적 추적을 개선합니다.

- **Technical Details**: 요약적으로, 연구진은 Swin-Unet 모델을 적용하여 위성 이미지로부터 호수의 경계를 추출하고 면적을 측정했습니다. 이 접근법은 고해상도 이미지의 수용 영역 요구 사항을 효과적으로 처리하며, 딥러닝 기법인 Long Short-Term Memory (LSTM) 신경망을 통해 기후 및 수문학적 요인을 고려한 미래 호수 면적 변화를 예측할 수 있는 모델을 구성했습니다.

- **Performance Highlights**: 모델은 미래의 호수 면적 변화를 예측하는데 있어 0.317 km²의 RMSE(root mean square error)를 달성하였으며, 이는 기후 변화에 대한 반응을 수량적으로 분석하는 데 있어 유의미한 결과로 평가됩니다.



### MPL: Lifting 3D Human Pose from Multi-view 2D Poses (https://arxiv.org/abs/2408.10805)
Comments:
          14 pages, accepted in ECCV T-CAP 2024, code: this https URL

- **What's New**: 이 논문은 2D 이미지로부터 3D 인간 포즈를 추정하는 새로운 방법을 제안합니다. 기존의 방법들은 실제 환경에서의 적용 가능성이 떨어지는 반면, 제안된 접근법은 'in-the-wild' 조건에서도 잘 작동할 수 있도록 설계되었습니다.

- **Technical Details**: 이 시스템은 3단계로 구성됩니다: 첫째, 2D 포즈 추정을 통해 여러 뷰에서 관절의 위치를 독립적으로 추출합니다. 둘째, Multi-view 3D Pose Lifter(MPL)를 사용하여 2D 스켈레톤을 3D 스켈레톤으로 변환합니다. MPL은 Spatial Pose Transformer(SPT)와 Fusion Pose Transformer(FPT)로 구성됩니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 2D 포즈를 삼각 측량하여 얻은 3D 포즈에 비해 MPJPE 오차가 최대 45% 감소하는 것으로 나타났습니다. 이는 실제 환경에서 안정적인 3D 포즈 추정을 위한 가능성을 보여줍니다.



### Tapping in a Remote Vehicle's onboard LLM to Complement the Ego Vehicle's Field-of-View (https://arxiv.org/abs/2408.10794)
Comments:
          50th Euromicro Conference Series on Software Engineering and Advanced Applications (SEAA) 2024 - WiP

- **What's New**: 이 논문은 차량의 시야(FOV)를 다른 차량의 시야와 연결하여 보완하는 개념을 제안합니다. 이를 통해 이동 중 장애물이나 보행자와 같은 교통 참여자에 대한 인식을 개선할 수 있습니다. 특히, 대화형 대형 언어 모델(LLM)을 이용하여 다른 차량과 소통하며 시각 정보를 공유하는 방식에 초점을 맞췄습니다.

- **Technical Details**: 논문에서는 V2I(vehicle-to-infrastructure) 및 V2V(vehicle-to-vehicle) 통신을 통해 차량 간의 데이터 전송 문제를 다루고 있습니다. 또한, 최신 LLM 모델인 GPT-4V와 GPT-4o의 활용 가능성을 보여줍니다. 이 모델들은 교통 상황을 이해하고 감지하는 능력이 뛰어난 것으로 평가되었습니다. 연구에서는 최소한의 대화를 통해 보행자 탐지를 위한 실험을 진행했으며, 다양한 시나리오와 데이터셋을 통해 성능을 평가하였습니다.

- **Performance Highlights**: 실험 결과는 LLM이 보행자를 탐지하는 데 높은 수준의 세밀함을 나타내지만, 탐지 품질 개선을 위한 보다 나은 프롬프트가 필요하다는 것을 강조합니다. 또한, 데이터 전송량 및 지적 재산권(IP) 민감한 정보 교환의 문제를 해결하여, 기존의 기술적인 접근 방식과 비교해 상당한 이점을 제공합니다.



### Learning Part-aware 3D Representations by Fusing 2D Gaussians and Superquadrics (https://arxiv.org/abs/2408.10789)
- **What's New**: 본 논문에서는 세멘틱 파트(semantic parts)를 기반으로 한 3D 재구성(part-aware 3D reconstruction)을 목표로 하는 하이브리드 표현(hybrid representation)을 소개합니다. 이 방법은 기존의 저수준 3D 표현(point clouds, meshes 등) 대신, 3D 객체를 구성하는 파트的 요소를 분석하여 더 높은 인식을 가능하게 합니다.

- **Technical Details**: 하이브리드 표현은 슈퍼쿼드릭(superquadrics)과 2D 가우시안(2D Gaussians)의 조합을 사용하여 다중 시점 이미지 입력에서 3D 구조 정보를 추출합니다. 슈퍼쿼드릭을 메쉬 형태로 통합하고, 가우시안의 중심을 메쉬의 면에 연결하는 방식으로 이루어집니다. 또한, 훈련 과정에서 파라미터를 최적화하여 최종적으로 효율적인 하이브리드 표현을 달성합니다.

- **Performance Highlights**: DTU 및 ShapeNet 데이터셋을 사용한 실험 결과, 본 방법은 3D 장면을 합리적인 부분으로 분할하며, 기존의 최첨단(state-of-the-art) 접근 방식을 초월하는 성능을 보여주었습니다. 따라서 고품질의 렌더링(rending)과 구조적 기하학적 재구성이 성공적으로 이루어졌습니다.



### LightMDETR: A Lightweight Approach for Low-Cost Open-Vocabulary Object Detection Training (https://arxiv.org/abs/2408.10787)
- **What's New**: 본 논문에서는 MDETR(멀티모달 객체 탐지 모델)의 최적화된 변형인 Lightweight MDETR (LightMDETR)을 소개합니다. LightMDETR은 계산 효율성을 높이는 동시에 견고한 멀티모달 기능을 유지하도록 설계되었습니다.

- **Technical Details**: LightMDETR의 주요 방법론은 MDETR의 백본(backbone) 구조를 동결(freeze)하고, 오직 Deep Fusion Encoder (DFE)라는 단일 구성 요소만 학습하는 것입니다. DFE는 이미지와 텍스트를 표현할 수 있는 능력을 가지고 있으며, 학습 가능한 컨텍스트 벡터를 사용하여 이러한 두 가지 모달리티(모드)를 전환할 수 있도록 합니다. 이는 파라미터 수를 줄이면서도 MDETR의 기본 성능을 유지합니다.

- **Performance Highlights**: LightMDETR은 RefCOCO, RefCOCO+, RefCOCOg와 같은 데이터셋에서 평가되었으며, 뛰어난 정밀도와 정확도를 달성했습니다. 또한, 비용 효율적인 훈련 방법을 통해 많은 파라미터 조정 없이도 개방 어휘(object detection)에 대한 성능을 개선할 수 있음을 보여줍니다.



### Just a Hint: Point-Supervised Camouflaged Object Detection (https://arxiv.org/abs/2408.10777)
Comments:
          Accepted by ECCV2024

- **What's New**: 본 연구에서는 Camouflaged Object Detection (COD) 작업을 단 한 점 주석(point supervision)만으로 수행할 수 있는 혁신적인 접근 방식을 제안합니다. 기존의 pixel-wise 주석이 아닌, 빠른 클릭을 통해 대상을 지정하는 방식으로 주석 과정을 간소화하였습니다.

- **Technical Details**: 주요 기술적 요소로는 1) hint area generator로, 점 주석을 확장하여 대상 영역을 제시하며, 2) attention regulator로, 모델이 전체 객체에 주의를 분산시킬 수 있도록 돕습니다. 또한, 3) Unsupervised Contrastive Learning (UCL)을 활용하여 학습 과정에서의 불안정한 특징 표현을 개선합니다.

- **Performance Highlights**: 세 가지 COD 벤치마크에서의 실험 결과, 본 방법이 여러 약하게 감독된 방법과 비교하여 월등한 성능을 보이며, 심지어 완전 감독된 방법들도 초과 달성하는 성과를 기록했습니다. 또한, 다른 과제인 scribble-supervised COD 및 SOD으로의 이전 시에도 경쟁력 있는 결과를 얻었습니다.



### Generative AI in Industrial Machine Vision -- A Review (https://arxiv.org/abs/2408.10775)
Comments:
          44 pages, 7 figures, This work has been submitted to the Journal of Intelligent Manufacturing

- **What's New**: 최근 연구는 산업 기계 비전에서 Generative AI의 진화와 응용 가능성을 조명하며, 데이터 증강(data augmentation)에 주로 활용되고 있음을 보여줍니다. 이 논문은 Generative AI의 현재 상태와 최근 발전을 포괄적으로 검토하고 있습니다.

- **Technical Details**: Generative AI (GenAI) 기술은 이미지 분류, 객체 감지, 품질 검사 등의 기계 비전 작업을 위한 데이터 증강에 활용됩니다. 이 논문에서는 1,200개 이상의 관련 논문을 분석하고 GenAI 모델 아키텍처, 산업 기계 비전 과제 및 요구사항을 정리하여 GenAI의 유용성을 강조합니다.

- **Performance Highlights**: Generative AI의 주요 응용 분야는 품질 조사 및 결함 탐지로, 연구는 이 기술이 어떻게 현재의 수작업 프로세스를 자동화할 수 있는지에 대한 통찰을 제공합니다. 이 연구는 또한 GenAI의 도전 과제를 정리하여 향후 연구 및 응용에 대한 기회를 제시합니다.



### Detection of Intracranial Hemorrhage for Trauma Patients (https://arxiv.org/abs/2408.10768)
- **What's New**: 이 연구에서는 다중 외상 환자에서 두개내 출혈을 감지하고 강조하는 딥러닝 기반 방법을 제안합니다. 특히, 이전 연구들과는 달리 구간을 분할하는 것이 아니라 경계 상자(bounding box)를 사용하여 출혈의 위치를 로컬화하는 방식입니다.

- **Technical Details**: 제안된 방법은 3D Retina-Net 아키텍처와 ResNet-50 기반의 Feature Pyramid Network (FPN)를 기반으로 하며, 새로운 Voxel-Complete IoU (VC-IoU) 손실 함수를 사용합니다. 이 함수는 신경망이 경계 상자의 3D 비율을 학습하도록 유도합니다.

- **Performance Highlights**: INSTANCE2022 데이터셋 및 개인적인 데이터셋에서 실험을 진행하였고, 각각 0.877 AR30, 0.728 AP30의 성능을 달성했습니다. 이는 다른 손실 함수와 비교했을 때 평균 회수(AR)에서 5% 향상된 성과를 나타냅니다.



### SAM-COD: SAM-guided Unified Framework for Weakly-Supervised Camouflaged Object Detection (https://arxiv.org/abs/2408.10760)
Comments:
          Accepted by ECCV2024

- **What's New**: 본 논문에서는 약한 감독(weakly-supervised) 캄플라지 사진 물체 탐지(COD) 문제를 해결하기 위해 새로운 SAM-COD 프레임워크를 제안합니다. 기존 방법들은 상이한 약한 감독 레이블을 지원하는 데 한계가 있었으나, SAM-COD는 스크리블(scribble), 바운딩 박스(bounding box), 포인트(point) 형식의 레이블을 모두 지원합니다.

- **Technical Details**: SAM-COD 프레임워크는 세 가지 모듈로 구성됩니다: 1) Prompt Adapter: 스크리블 레이블의 스켈레톤을 추출하고 이를 포인트로 샘플링해 SAM과 호환되도록 처리합니다. 2) Response Filter: SAM에서의 비정상 반응을 필터링하여 신뢰성 높은 마스크를 생성합니다. 3) Semantic Matcher: 세맨틱 점수와 세그멘테이션 점수를 결합하여 정교한 객체 마스크를 선택합니다. 또한, Prompt-adaptive Knowledge Distillation을 통해 지식 전이를 강화합니다.

- **Performance Highlights**: SAM-COD는 세 가지 주요 COD 벤치마크 데이터셋에서 광범위한 실험을 통해 기존의 약한 감독 및 완전 감독 방법들에 비해 우수한 성능을 나타냈으며, 특히 모든 유형의 약한 감독 레이블 하에서 최첨단 성능을 기록했습니다. 이 방법은 Salient Object Detection(SOD) 및 Polyp Segmentation 작업으로 이전했을 때도 긍정적인 결과를 보였습니다.



### TrackNeRF: Bundle Adjusting NeRF from Sparse and Noisy Views via Feature Tracks (https://arxiv.org/abs/2408.10739)
Comments:
          ECCV 2024 (supplemental pages included)

- **What's New**: TrackNeRF는 기존의 NeRF 방법론이 요구하는 많은 이미지와 정확한 포즈의 필요성을 극복하며, 더 적은 수의 이미지와 노이즈가 있는 포즈를 통해 정확한 3D 재구성을 지원합니다.

- **Technical Details**: TrackNeRF는 모든 가시적인 뷰의 연결된 픽셀 궤도를 활용하여 동일한 3D 포인트에 해당하는 feature tracks를 도입합니다. 이 방법은 reprojection consistency를 강제 자동화하여 전체적인 3D 일관성을 제공합니다. 이 접근법은 기존의 방법들이 해결하지 못했던 지역적인 일관성의 한계를 극복합니다.

- **Performance Highlights**: TrackNeRF는 DTU 데이터셋에서 PSNR 성능이 각각 약 8과 1의 개선을 보여주며, 뛰어난 재구성 정확도와 더 빠른 포즈 최적화 속도를 자랑합니다. TrackNeRF는 또한 큰 포즈 노이즈를 견딜 수 있으며, 고품질의 새로운 뷰를 매끄러운 깊이와 함께 생성할 수 있습니다.



### Coarse-to-Fine Detection of Multiple Seams for Robotic Welding (https://arxiv.org/abs/2408.10710)
- **What's New**: 이 논문에서는 RGB 이미지와 3D 포인트 클라우드를 활용하여 다중 용접 이음새를 효율적으로 추출할 수 있는 새로운 프레임워크를 제안합니다. 기존 연구는 주로 하나의 용접 이음새를 인식 및 위치 추적하는 데 초점을 맞추었으나, 이 논문은 한 번에 모든 이음새를 탐지하는 방법을 제시합니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 모듈로 구성되며, 첫 번째는 딥 러닝 방법을 기반으로 하는 조정 위치 모듈이며, 두 번째는 영역 성장(region growth)을 이용한 정밀 엣지 추출 모듈입니다. RGB 이미지를 사용하여 이음새의 근사 위치를 찾고, 이를 바탕으로 포인트 클라우드를 처리합니다. 알고리즘은 Fast Segment Anything Model(FastSAM)을 활용하여 불필요한 포인트 클라우드 정보를 제거하고, 효율성을 높이기 위해 다운샘플링을 수행합니다.

- **Performance Highlights**: 다양한 직선 및 곡선 형태의 용접 이음새가 포함된 물체에서의 실험을 통해 알고리즘의 유효성이 입증되었습니다. 이 방법은 실제 산업 응용 가능성을 강조하며, 영상 자료를 통해 실제 실험 과정도 확인할 수 있습니다.



### Large Language Models for Multimodal Deformable Image Registration (https://arxiv.org/abs/2408.10703)
- **What's New**: 새로운 멀티모달 변형 이미지 레지스트레이션(MDIR) 기법 LLM-Morph을 제안하며, 다양한 사전 훈련된 대형 언어 모델(LLM)을 활용하여 서로 다른 영상 모달 간의 깊은 특징 정렬 문제를 해결하는데 목표를 두고 있습니다.

- **Technical Details**: LLM-Morph는 CNN 인코더를 통해 교차 모달 이미지 쌍의 깊은 시각적 특징을 추출하고, 첫 번째 어댑터를 통해 이러한 특징을 조정한 후, 사전 훈련된 LLM의 가중치를 LoRA 방식을 사용하여 미세 조정합니다. 이 과정에서 LLM 인코딩 토큰을 다중 스케일 시각적 특징으로 변환하고 다중 스케일 변형 필드를 생성하여 MDIR 작업을 용이하게 합니다.

- **Performance Highlights**: MR-CT 복부 및 SR-Reg 뇌 데이터셋에서의 광범위한 실험을 통해 이 프레임워크의 효과성과 사전 훈련된 LLM이 MDIR 작업에 대한 잠재력을 입증하였습니다.



### MsMemoryGAN: A Multi-scale Memory GAN for Palm-vein Adversarial Purification (https://arxiv.org/abs/2408.10694)
- **What's New**: 이 논문에서는 깊은 신경망이 정맥 인식 작업에서 우수한 성능을 발휘하고 있지만, 적대적 공격에 취약하다는 문제를 극복하기 위해 MsMemoryGAN이라는 새로운 방어 모델을 제안합니다.

- **Technical Details**: MsMemoryGAN은 다중 스케일 오토인코더(multi-scale autoencoder)와 메모리 모듈을 활용하여 정상 샘플로부터 세부 패턴을 학습합니다. 또한, 입력 이미지 재구성을 위한 학습 가능한 메트릭(learnable metric)을 사용하여 가장 관련성 높은 메모리 항목을 검색합니다. 지각적 손실(perceptual loss)과 픽셀 손실(pixel loss)을 결합하여 재구성된 이미지의 품질을 향상시킵니다.

- **Performance Highlights**: 실험 결과, MsMemoryGAN은 다양한 적대적 섞임 벡터를 효과적으로 제거하여 정맥 분류기의 인식 정확도를 극대화할 수 있음을 보여줍니다.



### TDS-CLIP: Temporal Difference Side Network for Image-to-Video Transfer Learning (https://arxiv.org/abs/2408.10688)
- **What's New**: 본 논문에서는 Temporal Difference Side Network (TDS-CLIP)를 제안하여 대규모 프리트레인된 비전-언어 모델의 지식을 비디오 행동 인식 모델에 효율적으로 전이할 수 있는 방법을 모색합니다. TDS-CLIP는 지식 전이와 시간 모델링의 균형을 유지하면서 파라미터 동결 모델에서의 역전파를 회피합니다.

- **Technical Details**: TDS-CLIP의 핵심 구성 요소는 비주얼 스페이셜 인코더와 비주얼 템포럴 인코더로 구성됩니다. 비주얼 스페이셜 인코더는 프리트레인된 CLIP ViT를 기반으로 하며, 비디오 프레임의 공간 의미적 특징을 제공합니다. 비주얼 템포럴 인코더는 Side Motion Enhancement Adapter (SME-Adapter)와 Temporal Difference Adapter (TD-Adapter)를 사용하여 비디오의 동작 정보를 효과적으로 학습합니다.

- **Performance Highlights**: TDS-CLIP 방법은 Something-Something V1&V2와 Kinetics-400 데이터셋에서 실험을 수행하여 경쟁력 있는 성능을 달성하였으며, 메모리 효율성을 높이면서도 비디오 행동 인식 작업을 효율적으로 수행할 수 있음을 입증하였습니다.



### DemMamba: Alignment-free Raw Video Demoireing with Frequency-assisted Spatio-Temporal Mamba (https://arxiv.org/abs/2408.10679)
- **What's New**: 최근 Mamba는 Linear complexity를 가진 State Space Model(SSM)의 개선된 버전으로, 비디오 데모아링(video demoireing)에서 효과적인 시간 모델링을 가능하게 합니다. 본 논문에서는 새로운 정렬 모듈 없이도 작동하는 DemMamba라는 Raw 비디오 데모아링 네트워크를 제안합니다.

- **Technical Details**: DemMamba는 Spatial Mamba Block(SMB)과 Temporal Mamba Block(TMB)을 연속적으로 배열하여 Raw 비디오에서의 모아레 패턴(in moire patterns) 간의 intra- 및 inter-관계를 효과적으로 모델링합니다. SMB 내에서는 Adaptive Frequency Block(AFB)을 사용하여 주파수 영역에서 데모아링을 도와줍니다. TMB는 Channel Attention Block(CAB)을 포함하여 특징 간의 상호작용을 향상시킵니다.

- **Performance Highlights**: DemMamba는 현재 최첨단 방법들보다 1.3dB 더 우수한 성능을 보이며 비주얼 경험에서 뛰어난 품질을 제공합니다.



### A Noncontact Technique for Wave Measurement Based on Thermal Stereography and Deep Learning (https://arxiv.org/abs/2408.10670)
- **What's New**: 본 연구에서는 열 스테레오그래피(thermal stereography)와 딥러닝(deep learning)을 결합하여 비접촉(noncontact) 파독 측정 기술을 제안하였다. 이를 통해 수중의 광학적 특성 문제를 해결하고, 새로운 방식을 통해 실내 실험에서의 정확한 파 형상 측정을 가능하게 하였다.

- **Technical Details**: 이 연구에서는 긴 파장 적외선 스펙트럼(long-wave infrared spectrum)에서의 수조의 광학적 이미징 특성이 스테레오 매칭(stereo matching)에 적합하다는 것을 발견하였다. 열 스테레오 카메라를 사용하여 파 이미지를 캡처한 후, 딥러닝 기술을 활용한 재구성 전략을 제안하여 스테레오 매칭 성능을 향상시켰다. 생성적 접근(generative approach)을 통해 주석이 없는 적외선 이미지로부터 실제 불일치를 포함한 데이터셋을 합성하였다.

- **Performance Highlights**: 실험 결과, 제안한 기술은 파 프로브(wave probes)를 사용한 측정 결과와 비교하여 평균 편차(mean bias)가 2.1% 미만으로 매우 높은 정확도를 보였다. 이는 수리 실험(hydrodynamic experiments)에서 파면의 시공간 분포(spatiotemporal distribution)를 효과적으로 측정할 수 있음을 보여준다.



### UIE-UnFold: Deep Unfolding Network with Color Priors and Vision Transformer for Underwater Image Enhancemen (https://arxiv.org/abs/2408.10653)
Comments:
          Accepted by DSAA CIVIL 2024

- **What's New**: 이 논문은 기존의 학습 기반 방법의 한계를 극복하기 위해 색상 프라이어(color priors)와 각 단계 간 기능 변환(inter-stage feature transformation)을 통합한 새로운 deep unfolding network (DUN)을 제안합니다.

- **Technical Details**: 제안된 DUN 모델은 세 가지 주요 구성 요소로 이루어져 있습니다: 1) Color Prior Guidance Block (CPGB)는 열화된 이미지와 원본 이미지 색상 채널 간의 매핑을 설정합니다. 2) Nonlinear Activation Gradient Descent Module (NAGDM)은 수중 이미지 열화 과정을 시뮬레이션하는 역할을 합니다. 3) Inter Stage Feature Transformer (ISF-Former)는 네트워크의 서로 다른 단계 간 기능 교환을 용이하게 합니다.

- **Performance Highlights**: 다양한 수중 이미지 데이터셋에서의 실험 결과, 제안된 DUN 모델은 기존 최첨단 방법들보다 정량적 및 정성적 평가에서 우수한 성능을 보여주었습니다. 이를 통해 보다 정확하고 신뢰성 높은 수중 이미지 향상을 가능하게 합니다.



### Vocabulary-Free 3D Instance Segmentation with Vision and Language Assistan (https://arxiv.org/abs/2408.10652)
- **What's New**: 이 논문에서는 3D 인스턴스 세분화(3D Instance Segmentation, 3DIS) 문제를 해결하기 위해 새로운 접근 방식을 제안하고 있습니다. 기존의 닫힌 어휘(closed vocabulary) 및 열린 어휘(open vocabulary) 방법들과는 달리, 어휘가 전혀 없는 환경에서 3DIS 문제를 해결하는 방법인 어휘 없음 설정(Vocabulary-Free Setting)을 도입합니다.

- **Technical Details**: 제안된 방법인 PoVo는 비전-언어 어시스턴트(vision-language assistant)와 2D 인스턴스 세분화 모델을 활용하여 3D 인스턴스 마스크를 형성합니다. 입력되는 포인트 클라우드를 밀집 슈퍼포인트(dense superpoints)로 분할하고, 이를 스펙트럴 클러스터링(spectral clustering)을 통해 병합함으로써 3D 인스턴스 마스크를 생성합니다. 이 과정에서 마스크의 일관성과 의미론적 일관성을 모두 고려합니다.

- **Performance Highlights**: ScanNet200 및 Replica 데이터셋을 사용하여 방법을 평가한 결과, 기존 방식에 비해 어휘 없음 및 열린 어휘 설정 모두에서 성능이 향상된 것을 보여주었습니다. PoVo는 기존의 VoF3DIS 설정에 적응된 최신 접근 방식보다 우수한 성능을 기록하였으며, 많은 의미론적 개념들을 효과적으로 관리할 수 있는 강력한 설계를 갖추고 있음을 입증했습니다.



### A Review of Human-Object Interaction Detection (https://arxiv.org/abs/2408.10641)
- **What's New**: 이 논문은 이미지 기반 인간-객체 상호작용(Human-Object Interaction, HOI) 탐지의 최신 연구 성과를 체계적으로 요약하고 논의합니다. 기존의 두 단계(methods) 및 일체형(end-to-end) 탐지 접근법을 포함하여 현재 개발되고 있는 방법론을 종합적으로 분석하고 있습니다.

- **Technical Details**: HOI 탐지는 이미지 또는 비디오에서 인간과 객체를 정확히 찾고, 이들 사이의 상호작용 관계를 분류하는 것을 목표로 합니다. 논문에서는 주요 탐지 방법을 두 가지 범주로 나누어 설명합니다: 두 단계 방법(두 단계 모두 인간-객체 탐지 및 상호작용 분류)과 일체형 방법(새로운 HOI 매개체를 사용하여 상호작용 관계를 직접 예측). 또한 제로샷 학습(zero-shot learning), 약한 감독 학습(weakly supervised learning) 및 대규모 언어 모델(large-scale language models)의 발전도 논의합니다.

- **Performance Highlights**: HICO-DET 및 V-COCO 데이터 세트와 같은 다양한 데이터 세트를 통해 HOI 탐지 기술의 성과를 평가합니다. 두 단계 방법은 간단하지만 계산 부담이 크고, 일체형 방법은 추론 속도가 빠른 장점을 가지지만 다중 작업 간의 간섭이 발생할 수 있는 단점이 있습니다.



### Rethinking Video Segmentation with Masked Video Consistency: Did the Model Learn as Intended? (https://arxiv.org/abs/2408.10627)
- **What's New**: 이 논문은 Masked Video Consistency (MVC)와 Object Masked Attention (OMA)라는 새로운 훈련 전략을 통해 비디오 세그멘테이션의 품질을 향상시킨다. MVC는 이미지 패치를 무작위로 마스킹하여 전체 의미론적 세그멘테이션을 예측하도록 네트워크를 유도하는 방식을 제안한다.

- **Technical Details**: Masked Video Consistency (MVC)는 훈련 과정에서 무작위로 선택된 이미지 패치를 마스킹하고, 오히려 네트워크가 전체 이미지를 예측하게끔 유도하면서 공간적 및 시간적 특성 집합을 향상시킨다. Object Masked Attention (OMA)은 크로스-어텐션 메커니즘에서 관련 없는 질문의 영향을 줄여 시간적 모델링 능력을 강화한다.

- **Performance Highlights**: 이 연구는 다섯 개의 데이터셋에 걸쳐 세 가지 비디오 세그멘테이션 작업에서 최신의 분리된 비디오 세그멘테이션 프레임워크에 통합된 방식으로 최첨단 성능을 달성한다. 이 과정에서 모델의 파라미터 수는 증가하지 않고도 이전 방법들보다 유의미한 성능 향상을 보여준다.



### WRIM-Net: Wide-Ranging Information Mining Network for Visible-Infrared Person Re-Identification (https://arxiv.org/abs/2408.10624)
Comments:
          18 pages, 5 figures

- **What's New**: 이 논문은 Wide-Ranging Information Mining Network (WRIM-Net)을 선보이며, 이는 두 가지 차원(다양한 차원)에서 깊이 있는 정보를 채굴할 수 있는 능력을 가지고 있습니다. 특히, Multi-dimension Interactive Information Mining (MIIM) 모듈과 Auxiliary-Information-based Contrastive Learning (AICL) 접근 방식을 포함하여 모달리티 간 불일치를 극복할 수 있는 새로운 기법을 제안합니다.

- **Technical Details**: MIIM 모듈은 Global Region Interaction (GRI)을 활용하여 비국소적(spatial) 및 채널(channel) 정보를 포괄적으로 채굴합니다. 또한, AICL은 새로운 Cross-Modality Key-Instance Contrastive (CMKIC) 손실을 통해 모달리티 불변 정보를 추출할 수 있도록 네트워크를 유도합니다. 이러한 설계를 통해 MIIM은 얕은 층에 배치되어 특정 모달리티의 다차원 정보를 효과적으로 채굴할 수 있습니다.

- **Performance Highlights**: WRIM-Net은 SYSU-MM01, RegDB 및 LLCM 데이터셋을 포함한 다양한 벤치마크에서 이전의 방법들보다 뛰어난 성능을 보여주며, 모든 지표에서 최고의 성능을 기록했습니다.



### TextMastero: Mastering High-Quality Scene Text Editing in Diverse Languages and Styles (https://arxiv.org/abs/2408.10623)
- **What's New**: 이 논문에서는 복잡한 배경 및 글꼴 스타일에서의 기존 GAN 기반 방법의 한계를 극복하기 위해 다국어 장면 텍스트 편집 아키텍처인 	extit{TextMastero}를 제안합니다. 이 아키텍처는 특수한 Latent Diffusion Models (LDMs)을 기반으로 하여 텍스트의 정확성과 스타일 유사성을 보장하는 새로운 기능을 포함하고 있습니다.

- **Technical Details**: 	extit{TextMastero}는 두 가지 핵심 모듈을 포함합니다: 1) Glyph Conditioning Module - 정확한 텍스트 생성을 위한 세밀한 내용 제어를 제공; 2) Latent Guidance Module - 편집 전후에 스타일 일관성을 보장하기 위해 종합적인 스타일 정보를 제공합니다. 이 구조는 다층 OCR 모델의 피처를 활용하여 글리프 수준에서 세밀한 제어를 가능하게 합니다.

- **Performance Highlights**: 정량적 및 정성적 실험 결과, 	extit{TextMastero}는 텍스트 충실도 및 스타일 일관성에서 기존의 모든 방법보다 뛰어난 성능을 보였습니다. 특히, 비라틴 문자(예: CJK 문자)의 경우에도 만족스러운 결과를 생성하는 데 성공하였습니다.



### Novel Change Detection Framework in Remote Sensing Imagery Using Diffusion Models and Structural Similarity Index (SSIM) (https://arxiv.org/abs/2408.10619)
- **What's New**: 본 논문은 Stable Diffusion 모델과 Structural Similarity Index (SSIM)를 결합한 새로운 변화 감지 프레임워크인 Diffusion Based Change Detector를 제안합니다. 이 방법은 기존의 전통적 변화 감지 기법과 최근의 딥 러닝 기반 방법보다 뛰어난 성능을 보여줍니다.

- **Technical Details**: 제안하는 방법은 Stable Diffusion 모델의 강점을 활용하여 변화 감지를 위한 안정적이고 해석 가능한 변화 맵을 작성합니다. 이 프레임워크는 합성 데이터와 실세계 원격 감지 데이터셋에서 평가되었으며, 변화 감지의 정확성을 높이는 데 기여합니다.

- **Performance Highlights**: 실험 결과, Diffusion Based Change Detector는 복잡한 변화와 잡음이 있는 시나리오에서도 기존의 차이 기법 및 최근의 딥 러닝 기반 방법들보다 유의미하게 뛰어난 성능을 기대할 수 있음을 보여줍니다.



### A toolbox for calculating objective image properties in aesthetics research (https://arxiv.org/abs/2408.10616)
Comments:
          41 pages, 6 figure

- **What's New**: 시각 미학 연구의 결과를 비교할 수 있는 새로운 오픈 액세스 도구인 'Aesthetics Toolbox'를 개발하였습니다. 이 도구는 다양한 연구 그룹의 스크립트를 통합하여 사용자들이 손쉽게 이미지를 분석할 수 있도록 합니다.

- **Technical Details**: Aesthetics Toolbox는 시각 미학 연구에서 인기 있는 정량적 이미지 속성을 계산할 수 있도록 합니다. 여기에는 밝기(lightness) 및 색상 통계(color statistics), 푸리에 스펙트럼(Fourier spectral properties), 분율성(fractality), 자기 유사성(self-similarity), 대칭(symmetry), 다양한 엔트로피(entropy) 측정 및 CNN 기반 변동성(CNN-based variances)이 포함됩니다. 이 도구는 대부분의 장치와 호환되며 직관적인 클릭-드롭 웹 인터페이스를 제공합니다.

- **Performance Highlights**: Python 3로 번역된 스크립트가 원본 스크립트와 동일한 결과를 제공하도록 보장하여 분석 간의 일관성을 유지했습니다. 추가로 자세한 문서화와 클라우드 버전에 대한 링크도 GitHub를 통해 제공됩니다.



### Generalizable Facial Expression Recognition (https://arxiv.org/abs/2408.10614)
Comments:
          Accepted by ECCV2024

- **What's New**: 이 논문에서는 기존의 Facial Expression Recognition (FER) 방법들이 도메인 간의 차이로 인해 테스트 세트에서 낮은 성능을 보인다는 문제를 해결하고자 합니다. 구체적으로, 한 개의 학습 세트만을 이용하여 다양한 테스트 세트에 대해 zero-shot generalization (제로샷 일반화) 능력을 개선할 수 있는 새로운 FER 파이프라인을 제안합니다.

- **Technical Details**: 제안된 방법은 고정된 CLIP 얼굴 특징을 이용하여 시그모이드 마스크를 학습하여 표현 피쳐를 추출하는 방식입니다. 후보 특징의 채널을 표현 클래스에 따라 분리하여 logits를 생성하고, FC 레이어를 사용하지 않아 과적합을 줄입니다. 또한, 채널 다양성 손실 (channel-diverse loss)을 도입하여 학습된 마스크를 표현에 따라 분리하여 일반화 능력을 향상시킵니다.

- **Performance Highlights**: 다섯 개의 각기 다른 FER 데이터셋에서 수행한 실험을 통해 제안한 방법이 기존의 SOTA FER 방법들보다 현저히 우수한 성능을 보였음을 확인했습니다. 이 결과들은 학습된 특징과 시그모이드 마스크의 시각화를 통해 이해를 돕습니다.



### MUSES: 3D-Controllable Image Generation via Multi-Modal Agent Collaboration (https://arxiv.org/abs/2408.10605)
- **What's New**: 이 논문에서는 사용자 쿼리에 기반한 3D 제어 이미지 생성을 위한 일반적인 AI 시스템인 MUSES를 소개합니다. 기존의 텍스트-이미지 생성 방법들이 3D 세계에서 여러 객체와 복잡한 공간 관계를 생성하는 데 어려움을 겪고 있는 점에 주목하였습니다.

- **Technical Details**: MUSES는 (1) Layout Manager, (2) Model Engineer, (3) Image Artist의 세 가지 주요 구성 요소로 이루어진 점진적 프로세스를 개발하여 문제를 해결합니다. Layout Manager는 2D-3D 레이아웃 리프팅을 담당하고, Model Engineer는 3D 객체의 획득 및 보정을 수행하며, Image Artist는 3D-2D 이미지 랜더링을 진행합니다. 이러한 다중 모달 에이전트 파이프라인은 인간 전문가의 협업을 모방하여 3D 조작 가능한 객체의 이미지 생성과정을 효율적이고 자동화합니다.

- **Performance Highlights**: MUSES는 T2I-CompBench와 T2I-3DisBench에서 뛰어난 성능을 보여 DALL-E 3 및 Stable Diffusion 3와 같은 최근의 강력한 경쟁자들을 능가했습니다. 이 연구는 자연어, 2D 이미지 생성 및 3D 세계를 연결하는 중요한 발전을 보여줍니다.



### MV-MOS: Multi-View Feature Fusion for 3D Moving Object Segmentation (https://arxiv.org/abs/2408.10602)
Comments:
          7 pages, 4 figures

- **What's New**: 이번 연구에서는 다양한 2D 표현에서의 모션-시맨틱(motion-semantic) 피처를 융합하는 새로운 멀티-뷰 이동 객체 세분화(Moving Object Segmentation, MOS) 모델(MV-MOS)을 제안합니다. 이 모델은 효과적으로 보조적인 시맨틱(semantic) 피처를 제공함으로써 3D 포인트 클라우드(point cloud)의 정보 손실을 줄이는 데 중점을 두고 있습니다.

- **Technical Details**: MV-MOS는 두 가지 뷰(예: Bird's Eye View, BEV 및 Range View, RV)로부터 모션 피처를 통합하는 다중 브랜치 구조를 가지고 있습니다. 이 구조는 Mamba 모듈을 활용하여 시맨틱 피처와 모션 피처를 융합하고, 불균일한 피처 밀도를 처리하여 세분화의 정확성을 향상시킵니다.

- **Performance Highlights**: 제안된 모델은 SemanticKITTI 벤치마크에서 검증 세트에서 78.5%의 IoU(Intersection over Union), 테스트 세트에서 80.6%의 IoU를 달성하였으며, 기존의 최첨단 오픈 소스 MOS 모델보다 우수한 성능을 보였습니다.



### Breast tumor classification based on self-supervised contrastive learning from ultrasound videos (https://arxiv.org/abs/2408.10600)
- **What's New**: 본 논문에서는 무작위 샘플이 포함된 유방 초음파(ultrasound) 영상을 이용하여 진단 정확도를 높일 수 있는 새로운 접근법을 제시했습니다.

- **Technical Details**: 트리플렛 네트워크(triplet network)와 자기 지도 대조 학습(self-supervised contrastive learning) 기술을 통해 라벨이 없는 유방 초음파 비디오 클립에서 표현을 학습했습니다. 또한, 패턴을 정확하게 구별하기 위한 새로운 하드 트리플렛 로스(hard triplet loss)를 설계했습니다.

- **Performance Highlights**: 모델은 수신자 조작 특성 곡선(receiver operating characteristic curve, AUC)에서 0.952의 값을 달성하여 기존 모델들보다 월등히 높은 성과를 보였습니다. 특히 <100개의 라벨링된 샘플로도 0.901의 AUC를 달성할 수 있음을 보여주어 라벨이 필요한 데이터의 요구를 대폭 줄이는 가능성을 제시했습니다.



### DEGAS: Detailed Expressions on Full-Body Gaussian Avatars (https://arxiv.org/abs/2408.10588)
- **What's New**: 본 논문에서는 DEGAS라는 새로운 방법론을 소개하며, 상세한 얼굴 표정을 가진 전체 신체 아바타를 위한 최초의 3D Gaussian Splatting(3DGS) 기반 모델링 방법을 제안합니다. 이 방법은 2D 초상화 이미지를 통해 학습된 표현 잠재 공간을 사용하여 전통적인 3D Morphable Models(3DMMs)의 한계를 극복하고 2D 토킹 페이스와 3D 아바타 간의 간극을 메웁니다.

- **Technical Details**: DEGAS는 다중 시점 비디오를 훈련 데이터로 사용하는 조건부 변량 오토인코더를 통해 신체 모션과 얼굴 표정을 구동 신호로 활용하여 UV 레이아웃에 Gaussian 맵을 생성합니다. 이 과정에서 2D 초상화 이미지를 통해 학습된 표현 잠재 공간을 사용하여 아바타의 얼굴 제어를 가능하게 합니다. 추가적으로, 오디오 드리븐(audio-driven) 확장을 통해 상호작용 가능한 AI 에이전트를 위한 새로운 가능성을 열어줍니다.

- **Performance Highlights**: 기존 데이터셋과 우리가 새롭게 제안한 전체 신체 아바타 데이터셋을 사용한 실험을 통해, DEGAS가 제공하는 아바타는 섬세하고 정확한 얼굴 표정을 재현할 수 있음을 보여줍니다. 이를 통해 DEGAS는 포토리얼리스틱(rendering images) 아바타를 재현하는 데 있어 뛰어난 효율성을 입증하였습니다.



### Multi-view Hand Reconstruction with a Point-Embedded Transformer (https://arxiv.org/abs/2408.10581)
Comments:
          Generalizable multi-view Hand Mesh Reconstruction (HMR) model. Extension of the original work at CVPR2023

- **What's New**: 본 연구에서는 실세계 손 움직임 캡처를 위한 실용적인 다중 뷰 손 메쉬 재구성(Multi-View Hand Mesh Reconstruction, HMR) 모델인 POEM을 소개합니다. POEM은 3D 기초 점(basis point)을 통합하여 다양한 시점에서 얻어진 특징(feature)을 효과적으로 융합하는 새로운 접근 방식과 다섯 개의 대규모 데이터셋을 활용한 훈련 전략을 통해 발전했습니다.

- **Technical Details**: POEM 모델은 다중 뷰 스테레오에서의 정적 기초 점을 이용하여 손 메쉬를 재구성합니다. 카메라 위치와 자세를 무작위로 변경하여 다양한 카메라 구성에서 훈련을 진행함으로써 일반화 능력을 높였습니다. 또한, Projective Aggregation이라는 점별 특징 융합 모듈을 통해 여러 카메라의 정보를 융합합니다. 마지막으로 Point-Embedded Transformer를 사용하여 손 위치와 형태를 예측합니다.

- **Performance Highlights**: POEM은 실제 다중 뷰 카메라 플랫폼에서 테스트되었으며, 기존의 2D 키포인트 추정 방법보다 성능과 속도 모두에서 두드러진 장점을 보였습니다. 또한, 다양한 대규모 테스트 세트에서 뛰어난 성능을 지속적으로 입증했습니다. POEM은 손 모델의 절대 위치와 정확성을 요구하는 상호작용 시나리오에서도 효과적인 성능을 발휘합니다.



### MUSE: Mamba is Efficient Multi-scale Learner for Text-video Retrieva (https://arxiv.org/abs/2408.10575)
Comments:
          8 pages

- **What's New**: 이번 연구에서는 MUSE라는 새로운 접근법을 제안하여 다중 스케일 표현을 활용한 텍스트-비디오 검색(Text-Video Retrieval, TVR)을 수행합니다. MUSE는 마지막 단일 스케일 특성 맵(feature map)에 피쳐 피라미드(feature pyramid)를 적용하여 다중 스케일 표현을 생성합니다.

- **Technical Details**: MUSE는 효율적인 크로스-해상도(cross-resolution) 모델링을 위한 선형 계산 복잡성을 가진 다중 스케일 학습기입니다. 본 논문에서는 다양한 모델 구조와 디자인을 비교하고 연구하여 최적의 모델을 탐구하였습니다. 특히, Mamba 구조를 채택하여 스케일별 표현을 공동으로 학습합니다.

- **Performance Highlights**: MUSE는 MSR-VTT, DiDeMo, ActivityNet 데이터 셋에서 최첨단 성능을 달성하였으며, 작은 메모리 풋프린트와 조정 가능한 파라미터를 가지고 있습니다. 이러한 성과는 MUSE가 기존 Transformer 기반 방법들보다 더 효율적으로 다중 해상도 맥락을 학습하는 능력을 가지고 있음을 보여줍니다.



### Prompt-Agnostic Adversarial Perturbation for Customized Diffusion Models (https://arxiv.org/abs/2408.10571)
Comments:
          33 pages, 14 figures, under review

- **What's New**: 본 논문에서는 Personalized Diffusion Models를 위한 Prompt-Agnostic Adversarial Perturbation (PAP) 방법을 제안합니다. PAP는 라플라스 근사를 이용하여 프롬프트 분포를 모델링하고, 이를 기반으로 프롬프트에 구애받지 않는 공격 방어를 위한 교란을 생성합니다.

- **Technical Details**: PAP는 라플라스 근사를 사용하여 Gaussian 분포 형태의 프롬프트 분포를 모델링한 후, Monte Carlo 샘플링을 통해 교란을 계산합니다. 이 과정에서 Mean과 Variance는 2차 테일러 전개 및 헤시안 근사를 통해 추정됩니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋(VGGFace2, Celeb-HQ, Wikiart)에 대한 실험 결과, 제안된 PAP 방법은 기존의 프롬프트 특정 공격 방어 방법보다 현저히 성능 향상을 보였으며, 다양한 모델과 공격 프롬프트에 대해 강인한 효능을 입증했습니다.



### Diff-PCC: Diffusion-based Neural Compression for 3D Point Clouds (https://arxiv.org/abs/2408.10543)
- **What's New**: Diff-PCC는 최초의 확산 기반 점 구름 압축 방법론으로, 생성 기반 및 미적 우수성을 갖춘 복원 기능을 활용합니다.

- **Technical Details**: Diff-PCC는 두 개의 독립적인 인코딩 백본으로 구성된 이중 공간 잠재 표현(Dual-space Latent Representation)을 통해 점 구름에서 표현력이 뛰어난 형태 잠재를 추출하고, 이러한 잠재를 이용하여 노이즈가 포함된 점 구름을 확률적으로 디노이즈하는 확산 기반 생성기(Diffusion-based Generator)를 사용합니다.

- **Performance Highlights**: Diff-PCC는 최신 G-PCC 표준 대비 초저비트레이트에서 7.711 dB BD-PSNR 향상으로 최첨단 압축 성능을 달성했으며, 비주얼 품질에서도 우수한 성능을 나타냅니다.



### The Instance-centric Transformer for the RVOS Track of LSVOS Challenge: 3rd Place Solution (https://arxiv.org/abs/2408.10541)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2406.13939

- **What's New**: Referring Video Object Segmentation (ROVS)은 자연어 표현을 기반으로 비디오 내의 객체를 분할하는 다중 모달 작업으로, 최근 MeViS 데이터셋을 통해 더욱 복잡한 표현 이해가 요구됩니다. 이 연구는 두 가지 인스턴스 중심 모델을 개발하고, 프레임 수준과 인스턴스 수준에서 예측 결과를 융합하며, 세 번째 단계에서 SAM을 통한 공간적 정제를 도입합니다.

- **Technical Details**: 제안하는 방법은 MUTR 기반 모델, 인스턴스 검색 모델 및 융합 전략의 세 가지 구성 요소로 이루어져 있습니다. MUTR은 DETR 스타일로 설계되었으며, 인스턴스 마스크를 쿼리 초기화에 도입하여 일관성을 향상시키고, 동작 정보를 인스턴스_features의 특징으로 주입합니다. 비디오 클립의 모든 인스턴스 마스크를 추출하는 DVIS를 활용하여 마스크 생성을 진행합니다.

- **Performance Highlights**: 본 연구는 검증 단계에서 52.67 J&F 점수를, 테스트 단계에서 60.36 J&F 점수를 달성하였으며, 6번째 LSVOS Challenge RVOS Track에서 최종 3위를 차지하였습니다.



### Training Matting Models without Alpha Labels (https://arxiv.org/abs/2408.10539)
Comments:
          12 pages, 12 figures

- **What's New**: 이번 연구에서는 이미지 매팅(image matting)에서의 라벨링 어려움을 해결하기 위해, 트리맵(trimaps)와 같은 대충의 주석을 사용하여 깊은 이미지 매팅 모델을 효율적으로 훈련할 수 있는 새로운 방법론을 제안합니다. 이렇게 하면 정밀한 라벨 없이도 매팅 제약 조건을 기여할 수 있는 전통적인 가정 규칙과 학습된 의미를 결합할 수 있습니다.

- **Technical Details**: 우리는 방향성 거리 일관성 손실(DDC loss)을 도입하여 비지도 훈련 과정을 강화합니다. DDC loss는 입력 이미지에 기반한 픽셀 이웃의 알파 값(alpha values)을 제약하고, 알려진 영역에서 학습한 정보를 미지의 영역으로 전파할 수 있도록 합니다. 이 손실 함수는 유클리드 거리(euclidean distance)를 활용하여 주어진 이미지에서 유사한 이웃과의 일관성을 유지합니다.

- **Performance Highlights**: 실험 결과 AM-2K 및 P3M-10K 데이터셋에서 제안된 프레임워크가 세밀한 라벨에 의해 감독되는 기존 모델과 비슷한 성능을 보이거나, 때때로 인간 라벨링된 기준보다 더욱 만족스러운 결과를 제공함을 보여주었습니다. 이는 제안된 방법론의 실용성과 효과성을 검증합니다.



### Surgical Workflow Recognition and Blocking Effectiveness Detection in Laparoscopic Liver Resections with Pringle Maneuver (https://arxiv.org/abs/2408.10538)
- **What's New**: 이번 연구에서는 AI 기반의 수술 모니터링 기술을 활용하여 Pringle 기법(Pringle Maneuver, PM) 동안의 혈류 차단 효과성을 감지하고 수술 흐름을 인식하는 두 가지 보완 작업을 제안합니다. 이를 위해 50건의 로봇 과간 절제 수술 영상에서 수집한 25,037개의 비디오 프레임으로 구성된 새로운 데이터 세트 PmLR50을 구축하였습니다.

- **Technical Details**: PmNet라는 온라인 기반의 모델을 개발하여 Masked Temporal Encoding (MTE) 및 Compressed Sequence Modeling (CSM) 기법을 사용함으로써 단기 및 장기 시간 정보를 효율적으로 모델링합니다. 또한, Contrastive Prototype Separation (CPS) 기술을 통해 유사한 수술 동작 간의 행동 차별화를 강화하였습니다.

- **Performance Highlights**: 실험 결과 PmNet는 PmLR50 벤치마크에서 기존의 최첨단 수술 작업 인식 방법들을 초월한 성과를 보여주었습니다. 이는 로봇 간 절제 수술 커뮤니티에 대한 잠재적인 임상 응용 가능성을 높이는 연구로 평가받고 있습니다.



### Subspace Prototype Guidance for Mitigating Class Imbalance in Point Cloud Semantic Segmentation (https://arxiv.org/abs/2408.10537)
- **What's New**: 본 연구에서는 포인트 클라우드(3D 데이터) 의미 분할의 성능을 향상시키기 위한 새로운 방법인 서브스페이스 프로토타입 가이드(subspace prototype guidance, SPG)를 도입합니다. 이 방법은 클래스 불균형 문제를 해결하기 위한 것으로, 각 카테고리별로 포인트 클라우드를 독립적으로 분리하여 기능 서브스페이스를 생성합니다.

- **Technical Details**: SPG는 모델의 주 지점(feature point)과 보조 지점(auxiliary point)으로 구성된 두 개의 브랜치를 사용합니다. 보조 지점은 인코더와 프로젝션 헤드로 구성되며, 이는 특정 카테고리의 기능을 별도로 처리하여 소수 카테고리의 프로토타입을 추출합니다. 이 방식은 카테고리 프로토타입을 활용해 내부 클래스 분산(intra-class variance)을 줄이고 소수 카테고리의 기능 분리를 촉진합니다.

- **Performance Highlights**: 실험 결과, S3DIS, ScanNet v2, ScanNet200, Toronto-3D와 같은 대형 공공 벤치마크에서 제안하는 SPG 방법이 기존의 최첨단 기술을 초월하며 의미 분할 성능을 크게 개선한 것으로 나타났습니다.



### FAGStyle: Feature Augmentation on Geodesic Surface for Zero-shot Text-guided Diffusion Image Style Transfer (https://arxiv.org/abs/2408.10533)
- **What's New**: 이 논문에서는 FAGStyle이라는 새로운 제로샷 (zero-shot) 텍스트 가이드 확산 이미지 스타일 전이 방법을 제안합니다.

- **Technical Details**: FAGStyle은 Sliding Window Crop 기법과 Geodesic Surface上的 Feature Augmentation을 활용하여 스타일 제어 손실을 향상시킵니다. 또한, Pre-Shape self-correlation consistency 손실을 통합하여 콘텐츠 일관성을 보장함으로써 효과적인 스타일 전이를 가능하게 합니다.

- **Performance Highlights**: FAGStyle은 기존 방법들보다 우수한 성능을 보여주며, 원본 이미지의 의미 콘텐츠를 유지하면서 일관된 스타일화를 구현합니다. 다양한 스타일과 소스 콘텐츠에 대해 실험 결과가 효과성을 입증합니다.



### NutrifyAI: An AI-Powered System for Real-Time Food Detection, Nutritional Analysis, and Personalized Meal Recommendations (https://arxiv.org/abs/2408.10532)
Comments:
          7 pages, 12 figures

- **What's New**: 이 논문은 YOLOv8 모델을 활용한 음식 인식 및 영양 분석 통합 시스템을 소개합니다. 이 시스템은 모바일 및 웹 애플리케이션으로 구현되어 사용자들이 음식 데이터를 수동으로 입력하지 않아도 되도록 지원합니다.

- **Technical Details**: 시스템은 세 가지 주요 구성 요소로 나뉘며, 1) YOLOv8 모델을 이용한 음식 감지, 2) Edamam Nutrition Analysis API를 통한 영양 분석, 3) Edamam Meal Planning 및 Recipe Search APIs를 통해 제공되는 개인 맞춤형 식사 추천입니다. YOLOv8은 빠르고 정확한 객체 탐지를 위해 설계되었습니다.

- **Performance Highlights**: 초기 결과는 YOLOv8 모델이 0.963의 mAP를 기록하여 높은 정확도를 보여주며, 사용자들이 보다 정확한 영양 정보를 기반으로 식단 결정을 할 수 있도록 하는 데 가치 있는 도구임을 증명하였습니다.



### EdgeNAT: Transformer for Efficient Edge Detection (https://arxiv.org/abs/2408.10527)
- **What's New**: 본 논문에서는 DiNAT를 인코더로 사용하는 EdgeNAT라는 새로운 1단계 Transformer 기반 엣지 감지기를 제안합니다. EdgeNAT는 객체 경계 및 유의미한 엣지를 정확하고 효율적으로 추출할 수 있습니다.

- **Technical Details**: EdgeNAT는 global contextual information과 detailed local cues를 효율적으로 캡처합니다. 새로운 SCAF-MLA 디코더를 통해 feature representation을 향상시키며, inter-spatial 및 inter-channel 관계를 활용하여 엣지를 추출합니다. 모델은 다양한 데이터셋에서 성능을 검증하였고, BSDS500 데이터셋에서 ODS 및 OIS F-measure 각각 86.0%, 87.6%를 달성했습니다.

- **Performance Highlights**: EdgeNAT는 RTX 4090 GPU에서 single-scale input으로 20.87 FPS의 속도로 실행되며, 기존의 EDTER보다 ODS F-measure에서 1.2% 더 높은 성능을 보여주었습니다. 또한, 다양한 모델 크기를 제공하여 스케일러블한 특성을 갖추고 있습니다.



### BAUST Lipi: A BdSL Dataset with Deep Learning Based Bangla Sign Language Recognition (https://arxiv.org/abs/2408.10518)
- **What's New**: 이번 연구에서는 18,000개의 이미지로 구성된 새로운 방글라 수화(BdSL) 데이터셋을 소개하였습니다. 이 데이터셋은 36개의 방글라 기호를 포함하고 있으며, 그 중 30개는 자음 및 6개는 모음을 나타냅니다.

- **Technical Details**: 제안된 하이브리드 CNN(Convolutional Neural Network) 모델은 여러 개의 합성곱 계층과 LSTM(Long Short-Term Memory) 계층을 통합하여 구성되었습니다. 모델의 성능은 97.92%의 정확도로 평가되었습니다.

- **Performance Highlights**: 제안된 BdSL 데이터셋과 하이브리드 CNN 모델은 방글라 수화 연구에 있어 중요한 이정표가 될 것으로 기대됩니다. 이로 인해 수화 인식 시스템의 정확도가 크게 향상될 것으로 보입니다.



### Adaptive Knowledge Distillation for Classification of Hand Images using Explainable Vision Transformers (https://arxiv.org/abs/2408.10503)
Comments:
          Accepted at the ECML PKDD 2024 (Research Track)

- **What's New**: 본 연구는 손 이미지의 분류를 위해 Vision Transformers (ViTs)의 사용을 조사하였으며, 특히 다른 도메인 데이터에 학습 중인 모델의 지식을 효과적으로 전달하는 적응형 증류 방법을 제안합니다.

- **Technical Details**: ViT 모델은 손의 고유한 특징과 패턴을 분류하기 위해 사용됩니다. 연구에서는 영상 변환기(vision transformer)와 설명 가능한 도구(explainability tools)를 결합하여 ViTs의 내부 표현을 분석하고 모델 출력에 미치는 영향을 평가합니다. 또, 두 공개 손 이미지 데이터셋을 이용하여 여러 실험을 수행하였습니다.

- **Performance Highlights**: 실험 결과, ViT 모델은 기존의 전통적인 머신러닝 방법을 크게 초월하며, 적응형 증류 방법을 통해 소스 도메인과 타겟 도메인 데이터를 모두 활용하여 우수한 성능을 발휘합니다. 이러한 접근법은 액세스 제어, 신원 확인, 인증 시스템과 같은 실제 응용에 효과적으로 활용될 수 있습니다.



### GPT-based Textile Pilling Classification Using 3D Point Cloud Data (https://arxiv.org/abs/2408.10496)
Comments:
          8 pages, 2 figures

- **What's New**: 이번 연구에서는 섬유의 필링 평가를 위하여 최초로 공개된 3D 포인트 클라우드 데이터셋 TextileNet8을 구축하였습니다. 이를 통해, PointGPT 모델을 기반으로 한 PointGPT+NN 모델의 성능 향상을 이루었습니다.

- **Technical Details**: PointGPT+NN 모델은 비모수 네트워크(non-parametric network)로부터 추출된 입력 포인트 클라우드의 글로벌 특성을 통합함으로써 성능을 높였습니다. 이 모델은 전체 정확도(Overall Accuracy, OA) 91.8% 및 클래스당 평균 정확도(mean per-class accuracy, mAcc) 92.2%를 기록하였습니다.

- **Performance Highlights**: TextileNet8 데이터셋을 사용하여 다양한 포인트 클라우드 분류 모델의 성능을 평가함으로써, PointGPT+NN 모델의 경쟁력 있는 성능을 입증했습니다. 또한, 다른 공개 데이터셋에 대한 테스트 결과에서도 강화된 객관성을 확인하였습니다.



### Event Stream based Sign Language Translation: A High-Definition Benchmark Dataset and A New Algorithm (https://arxiv.org/abs/2408.10488)
Comments:
          First Large-scale and High-Definition Benchmark Dataset for Event-based Sign Language Translation

- **What's New**: 본 논문에서는 Sign Language Translation (SLT)의 새로운 접근 방법으로 Event streams를 제안합니다. 이 방법은 조명, 손 움직임 등으로부터 영향을 덜 받으며, 빠르고 높은 동적 범위를 갖추고 있어 정확한 번역을 가능하게 합니다. 또한, 새로운 Event-CSL 데이터셋을 제안하여 연구에 기여합니다.

- **Technical Details**: Event-CSL 데이터셋은 14,827개의 고화질 비디오로 구성되며, 14,821개의 glosses 및 2,544개의 중국어 단어가 포함되어 있습니다. 이 데이터는 다양한 실내 및 실외 환경에서 수집되었으며, Convolutional Neural Networks (CNN) 기반의 ResNet18 네트워크와 Mamba 네트워크를 결합한 하이브리드 아키텍처를 사용하여 SLT 성능을 향상시킵니다.

- **Performance Highlights**: 본 연구에서는 Event-CSL 데이터셋과 기존 SLT 모델들을 벤치마킹하여, SLT 분야에서의 효과적인 성능 개선을 입증하였습니다. 이 하이브리드 CNN-Mamba 아키텍처는 기존의 Transformer 기반 모델보다 더 나은 성능을 보입니다.



### MambaEVT: Event Stream based Visual Object Tracking using State Space Mod (https://arxiv.org/abs/2408.10487)
Comments:
          In Peer Review

- **What's New**: 본 논문에서는 이벤트 카메라 기반의 시각 추적(Visual Tracking) 알고리즘의 성능 병목 현상을 해결하기 위해 새로운 Mamba 기반의 시각 추적 프레임워크인 MambaEVT를 제안합니다. 이 프레임워크는 상태 공간 모델(State Space Model, SSM)을 기반으로 하여 낮은 계산 복잡성을 달성하며, 동적 템플릿 업데이트 전략을 통합하여 더 나은 추적 성능을 구현합니다.

- **Technical Details**: MambaEVT는 이벤트 스트림에서 정적 템플릿과 검색 영역을 추출하여 이벤트 토큰으로 변환하며, Mamba 네트워크를 통해 특징 추출 및 상호작용 학습을 수행합니다. 메모리 Mamba 네트워크를 사용하여 동적 템플릿을 생성하고, 추적 결과에 따라 템플릿 라이브러리를 업데이트합니다. 계산 복잡도는 O(N)으로, 기존의 Transformer 기반 방법보다 우수한 실적을 보입니다.

- **Performance Highlights**: MambaEVT는 EventVOT, VisEvent, FE240hz와 같은 여러 대규모 데이터셋에서 정확도와 계산 비용 간의 균형을 잘 유지하며, 기존 방법들과 비교하여 뛰어난 성능을 입증하였습니다.



### LSVOS Challenge 3rd Place Report: SAM2 and Cutie based VOS (https://arxiv.org/abs/2408.10469)
- **What's New**: 이번 논문은 최신 Video Object Segmentation (VOS) 기술에 대한 새로운 접근 방식을 제시합니다. SAM2와 Cutie라는 두 가지 최신 모델을 조합하여 VOS의 도전과제를 해결하고 있으며, 다양한 하이퍼파라미터가 비디오 인스턴스 분할 성능에 미치는 영향을 탐구하였습니다.

- **Technical Details**: 우리의 VOS 접근 방식은 최첨단 메모리 기반 방법을 활용하여 이전에 분할된 프레임에서 메모리 표현을 생성하고, 새 쿼리 프레임이 이 메모리에 접근하여 세그멘테이션에 필요한 특징을 조회합니다. 특히, SAM2는 이미지와 비디오 분할을 위한 통합 모델로, 메모리 모듈을 통해 이전 프레임의 맥락을 활용하여 마스크 예측을 생성하고 정교화합니다. Cutie는 반자동 준지도 학습 환경에서 작동하여 도전적인 시나리오를 처리합니다.

- **Performance Highlights**: 우리의 접근 방식은 LSVOS 챌린지 VOS 트랙 테스트 단계에서 J&F 점수 0.7952를 달성하여 전체 3위에 랭크되었습니다. 이는 고해상도 이미지 처리, 고급 객체 추적, 오클루전 예측 전략을 통해 높은 정확도로 성과를 얻어낸 결과입니다.



### Kubrick: Multimodal Agent Collaborations for Synthetic Video Generation (https://arxiv.org/abs/2408.10453)
- **What's New**: 본 논문은 Vision Large Language Model (VLM) 에이전트 협업을 기반으로 한 자동 합성 비디오 생성 파이프라인을 제안합니다. 이 시스템은 자연어 설명을 입력으로 받아, 여러 VLM 에이전트가 생성 파이프라인의 다양한 프로세스를 자동으로 지시합니다.

- **Technical Details**: 3단계로 나누어진 파이프라인 구조를 사용하며, 각 단계는 Director 에이전트, Programmer 에이전트, Reviewer 에이전트로 구성되어 있습니다. Director 에이전트가 비디오 설명을 서브 프로세스로 분해하고, Programmer 에이전트가 Python 스크립트를 생성하여 Blender 상에서 비디오를 렌더링합니다. Reviewer 에이전트는 생성된 비디오의 품질에 대한 피드백을 제공하여 스크립트를 반복적으로 개선합니다.

- **Performance Highlights**: 우리의 생성된 비디오는 상업적인 비디오 생성 모델과 비교하여 품질이 더 우수하며, 비디오 품질 및 지시 준수 성능에서 5개의 메트릭을 통해 이러한 성과를 입증합니다. 또한, 포괄적인 사용자 연구에서 품질, 일관성 및 합리성 면에서 다른 접근 방식보다 뛰어난 성능을 보였습니다.



### The Brittleness of AI-Generated Image Watermarking Techniques: Examining Their Robustness Against Visual Paraphrasing Attacks (https://arxiv.org/abs/2408.10446)
Comments:
          23 pages and 10 figures

- **What's New**: 최근 이미지 생성 시스템의 급속한 발전으로 인해 AI가 생성한 이미지의 물리적 저작권 보호가 우려되고 있습니다. 이에 따라 기업들은 이미지에 워터마크(watermark) 기술을 적용하려고 노력하고 있으나, 기존의 방법들은 Visual Paraphrase 공격에 취약하다는 주장을 하고 있습니다.

- **Technical Details**: 이 논문에서 제안하는 Visual Paraphrase 공격은 두 단계로 이루어집니다. 첫 번째 단계에서는 KOSMOS-2라는 최신 이미지 캡셔닝 시스템을 사용하여 주어진 이미지에 대한 캡션을 생성합니다. 두 번째 단계에서는 원본 이미지와 생성된 캡션을 이미지 간 확산 시스템(image-to-image diffusion system)에 전달하여 디노이징 단계에서 텍스트 캡션에 의해 안내된 시각적으로 유사한 이미지를 생성합니다.

- **Performance Highlights**: 실험 결과, Visual Paraphrase 공격이 기존 이미지의 워터마크를 효과적으로 제거할 수 있음을 입증하였습니다. 이 논문은 기존의 워터마킹 기법들의 취약성을 비판적으로 평가하며, 보다 강력한 워터마킹 기술의 개발을 촉구하는 역할을 합니다. 또한, 연구자들을 위한 최초의 Visual Paraphrase 데이터셋과 코드를 공개하고 있습니다.



### CLIP-DPO: Vision-Language Models as a Source of Preference for Fixing Hallucinations in LVLMs (https://arxiv.org/abs/2408.10433)
Comments:
          Accepted at ECCV 2024

- **What's New**: 본 논문에서는 LVLMs(Large Vision Language Models)의 한계인 환각 현상(hallucination)을 개선하기 위한 새로운 방법인 CLIP-DPO를 제안합니다. CLIP-DPO는 별도 데이터 구축 없이 CLIP 모델을 활용하여 LVLM의 출력 캡션을 최적화하는 방식으로 작동합니다.

- **Technical Details**: CLIP-DPO는 기존의 DPO(Direct Preference Optimization) 접근 방식에서 발전된 것으로, CLIP 모델을 사용하여 LVLM의 자가 생성된 캡션을 평가하고 긍정-부정 쌍을 구성합니다. 이 방법은 추가 외부 데이터나 LVLM을 요구하지 않고, 효율적인 LVLM에서 생성된 데이터를 기반으로 합니다. 최종 데이터 세트는 강력한 규칙 기반 필터링을 통해 훈련 전에 정제됩니다.

- **Performance Highlights**: 이 방법을 MobileVLM-v2 및 LlaVA-1.5 모델에 적용한 결과, 환각 발생률 감소에서 뚜렷한 개선을 보였으며, 기본 모델에 비해 성능이 현저히 향상되었습니다. 또한, zero-shot 이미지 분류 성과가 개선되었음을 확인했습니다.



### Towards Automation of Human Stage of Decay Identification: An Artificial Intelligence Approach (https://arxiv.org/abs/2408.10414)
Comments:
          13 pages

- **What's New**: 이 연구는 인공지능(AI)을 활용하여 인간 부패 사진의 스테이지 오브 디컴포지션(SOD) 분류를 자동화하는 가능성을 탐구합니다. 현재 수작업으로 진행되는 SOD 평가법의 주관성과 비효율성을 해결하고자 합니다.

- **Technical Details**: 이 연구는 Megyesi와 Gelderman이 제안한 두 가지 SOD 평가 방법을 AI로 자동화하는 것을 목표로 하며, Inception V3 및 Xception과 같은 심층 학습(deep learning) 모델을 대규모 인간 부패 이미지 데이터셋에서 훈련시켜 다양한 해부학적 영역에 대한 SOD를 분류합니다. 이 AI 모델의 정확도는 사람의 법의학 전문가와 비교하여 검사되었습니다.

- **Performance Highlights**: Xception 모델은 Megyesi 방식에서 머리, 몸통, 사지 각각에 대해 매크로 평균 F1 점수 0.878, 0.881, 0.702를 달성하였으며, Gelderman 방식에서도 각각 0.872, 0.875, 0.76을 기록했습니다. AI 모델의 신뢰성은 인간 전문가의 수준에 도달하며, 이 연구는 AI를 통한 SOD 식별 자동화의 가능성을 보여줍니다.



### Parallel Processing of Point Cloud Ground Segmentation for Mechanical and Solid-State LiDARs (https://arxiv.org/abs/2408.10404)
Comments:
          5 pages

- **What's New**: 본 연구는 FPGA 플랫폼을 위한 실시간 포인트 클라우드 지표(segment) 분할을 위한 새로운 병렬 처리(framework)를 도입하였습니다. LiDAR 알고리즘을 기계식에서 고체 상태의 LiDAR(SSL) 기술로 변화하는 환경에 맞추기 위해, 지표 분할 작업에 중점을 두고 이를 실제 SSL 데이터 처리에 적용하였습니다.

- **Technical Details**: 우리는 SemanticKITTI 데이터셋을 기반으로 하는 기계식 LiDAR에서 포인트 기반(point-based), 격자 기반(voxel-based), 그리고 범위 이미지(range-image) 기반의 지표 분할 접근법을 사용하여 프레임 세분화(frame-segmentation) 기반의 병렬 처리 기술을 검증하였습니다. 특히 범위 이미지 방법의 저항력이 뛰어남을 확인하였고, 사용자 정의 데이터셋을 사용하여 SSL 센서에 대한 병렬 접근의 효과를 검증하였습니다.

- **Performance Highlights**: FPGA에서 SSL 센서를 위한 범위 이미지 기반 지표 분할의 혁신적인 구현은 기존 CPU 설정보다 최대 50.3배 빠른 처리 속도 향상을 보여주었습니다. 이러한 결과는 병렬 처리 전략이 자율 시스템의 고급 인식 작업을 위한 LiDAR 기술을 상당히 향상시킬 가능성을 강조합니다.



### Webcam-based Pupil Diameter Prediction Benefits from Upscaling (https://arxiv.org/abs/2408.10397)
- **What's New**: 본 연구는 저해상도 webcam 이미지에서 동공 지름을 예측하는 여러 업스케일링 방법의 영향을 평가합니다. 특히, 기존의 다소 한정적인 이미지 데이터셋과는 달리 EyeDentify 데이터셋을 활용하여 다양한 조건에서의 이미지 품질 개선이 동공 검사에 미치는 효과를 분석합니다.

- **Technical Details**: 여러 업스케일링 기술을 비교 분석하면서 bicubic interpolation에서 시작하여 CodeFormer, GFPGAN, Real-ESRGAN, HAT, SRResNet과 같은 고급 super-resolution 방법까지 포함합니다. 연구 결과, 이미지를 업스케일링하는 방법과 비율에 따라 동공 지름 예측 성능이 크게 달라진다는 것을 발견하였습니다.

- **Performance Highlights**: 고급 SR 모델을 활용할 때 동공 지름 예측 모델의 정확성이 현저히 향상됨을 나타냅니다. 특히 업스케일링 기법을 사용하는 것이 전반적으로 동공 지름 예측의 성능을 개선하는 것으로 나타났습니다. 이는 심리적 및 생리적 연구에서 보다 정확한 평가를 용이하게 합니다.



### Evaluating Image-Based Face and Eye Tracking with Event Cameras (https://arxiv.org/abs/2408.10395)
Comments:
          This paper has been accepted at The Workshop On Neuromorphic Vision: Advantages and Applications of Event Cameras at the European Conference on Computer Vision (ECCV), 2024

- **What's New**: 이 논문은 이벤트 카메라(Event Cameras) 데이터를 활용하여 전통적인 알고리즘과의 통합 가능성을 보여줍니다. 특히, 이벤트 기반 데이터로부터 프레임 포맷을 생성하여 얼굴 및 눈 추적 작업에서 효용성을 평가합니다.

- **Technical Details**: 본 연구는 Helen Dataset의 RGB 프레임을 기반으로 이벤트를 시뮬레이션하여 프레임-기반 이벤트 데이터셋을 구축하였습니다. 또한, GR-YOLO라는 새로운 기술을 활용하여 YOLOv3에서 발전된 얼굴 및 눈 검출 성능을 평가하며, YOLOv8과의 비교 분석을 통해 결과의 효용성을 확인하였습니다.

- **Performance Highlights**: 모델은 다양한 데이터셋에서 평균 정밀도(mean Average Precision) 점수 0.91을 기록하며 좋은 예측 성능을 보였습니다. 또한, 변화하는 조명 조건에서도 실시간 이벤트 카메라 데이터에 대해 강건한 성능을 나타냈습니다.



### Narrowing the Gap between Vision and Action in Navigation (https://arxiv.org/abs/2408.10388)
- **What's New**: 이 논문에서는 Vision and Language Navigation in the Continuous Environment (VLN-CE) 에이전트의 비주얼 환경을 보다 효과적으로 학습하고 탐색 성능을 향상시키기 위해 저수준 행동 디코더와 고수준 행동 예측을 결합한 새로운 접근 방식을 제안합니다.

- **Technical Details**: 저수준 행동 디코더를 통합하여 에이전트가 고수준 뷰 선택과 동시에 저수준의 행동 시퀀스를 생성할 수 있도록 하는 이중 행동 모듈을 소개합니다. 이를 통해 에이전트는 선택된 비주얼 뷰를 저수준 제어와 연결할 수 있습니다. 또한, 시각적 표현을 포함한 waypoint predictor를 향상시켜, 객관적인 의미를 반영하고 장애물을 명시적으로 마스킹하여 더 효과적인 관점을 생성합니다.

- **Performance Highlights**: 실험 결과, 이 접근 방식은 VLN-CE 에이전트의 탐색 성능 지표를 개선하는 데 성공하였으며, 고수준 및 저수준 행동 모두에서 강력한 기준선을 초월하는 성능 향상을 보였습니다.



### HaSPeR: An Image Repository for Hand Shadow Puppet Recognition (https://arxiv.org/abs/2408.10360)
Comments:
          Submitted to IEEE Transactions on Artificial Intelligence (IEEE TAI), 11 pages, 78 figures, 2 tables

- **What's New**: 이 논문에서는 손 그림자 인형극(Hand Shadow Puppetry)의 예술을 보존하기 위한 새로운 데이터셋 HaSPeR (Hand Shadow Puppet Image Repository)를 소개합니다. 이 데이터셋은 전문 및 아마추어 인형극 비디오에서 추출한 총 8,340개의 이미지를 포함하고 있으며, 그림자 인형극에 대한 인공지능 (AI) 연구를 위한 기초 자료로 제공됩니다.

- **Technical Details**: HaSPeR 데이터셋은 이미지 분류 모델의 성능을 평가하기 위해 다양한 사전 학습된 이미지 분류 모델을 사용하여 검증되었습니다. 연구 결과, 전통적인 convolutional 모델들이 attention 기반 transformer 아키텍처보다 성능이 우수한 것으로 나타났습니다. 특히 MobileNetV2와 같은 경량 모델이 모바일 애플리케이션에 적합하며, 사용자에게 유용한 şəkildə 작동합니다. InceptionV3 모델의 특징 표현, 설명 가능성 및 분류 오류에 대한 Thorough 분석이 수행되었습니다.

- **Performance Highlights**: 논문에서 제시된 데이터셋은 예술 영역에 대한 탐구 및 분석의 기회를 제공하여 독창적인 ombromanie 교육 도구 개발에 기여할 수 있을 것으로 기대됩니다. 또한, 코드와 데이터는 공개적으로 제공되어 연구자들이 접근할 수 있도록 되어 있습니다.



### Diversity and stylization of the contemporary user-generated visual arts in the complexity-entropy plan (https://arxiv.org/abs/2408.10356)
Comments:
          18 pages, 3 figures, 1 table, SI(4 figures, 3 tables)

- **What's New**: 이 연구는 유명한 정보 공유 플랫폼인 DeviantArt와 Behance에서 수집한 149,780장의 현대 사용자 생성 시각 예술 이미지를 분석하여, 복잡도-엔트로피 (complexity-entropy, C-H) 평면을 활용해 예술 스타일의 진화 과정을 탐구합니다. 이는 예술 사조의 출현과 스타일화의 기초가 되는 진화 과정을 분석하는 새로운 접근 방식을 제공합니다.

- **Technical Details**: 연구는 두 가지 주요 분석 방법을 통해 진행됩니다. 첫째, C-H 평면에서의 평균 C 및 H 값의 연도별 변화를 측정하여 미적 스타일의 변화를 추적합니다. 둘째, 다중 이미지 표현공간에서의 유사성을 측정하기 위해 코사인 유사도(cosine similarity)와 자카드 유사도(Jaccard similarity)를 활용하며, 사전 훈련된 Residual Network (ResNet)와 Scale-invariant Feature Transform (SIFT) 기능을 통해 저차원 및 고차원 이미지 특징을 추출합니다. 또한, 자율회귀이동평균(ARMA) 모델을 사용하여 주어진 기간 내 예술 작품의 평균 C-H 위치와 이미지 특징 간의 관계를 통계적으로 분석합니다.

- **Performance Highlights**: 연구 결과, C-H 평면에서 사용자 생성 시각 예술의 스타일 진화가 명확하게 드러났으며, 시각적 예술 스타일의 C-H 정보와 다층 이미지 특징의 유사성 간에 간섭이 있는 것을 발견했습니다. 특히, 일정한 C-H 영역에서 이미지 표현의 다양성이 두드러지며, 새로운 스타일의 출현과 더 큰 스타일적 다양성을 지닌 예술 작품들 사이의 상관관계를 보여줍니다.



### Optical Music Recognition in Manuscripts from the Ricordi Archiv (https://arxiv.org/abs/2408.10260)
Comments:
          Accepted at AudioMostly 2024

- **What's New**: 리코르디 아카이브(Ricordi Archive)의 디지털화가 완료되어, 이탈리아 오페라 작곡가들의 수많은 음악 원고에서 음악 요소를 자동으로 추출하고 이를 분류할 수 있는 신경망 기반의 분류기가 개발되었습니다.

- **Technical Details**: 이 연구는 Optical Music Recognition (OMR) 방법론을 통해 음악 기호를 자동으로 식별하는 시스템을 구축하였습니다. 특히, Convolutional Recurrent Neural Networks를 활용하여 필기 음표의 복잡한 특성을 다루었습니다.

- **Performance Highlights**: 다양한 신경망 분류기를 훈련하여 음악 요소를 구분하는 실험을 수행하였으며, 이 결과들은 향후 나머지 리코르디 아카이브의 자동 주석 작업에 활용될 것으로 기대됩니다.



### NeRF-US: Removing Ultrasound Imaging Artifacts from Neural Radiance Fields in the Wild (https://arxiv.org/abs/2408.10258)
- **What's New**: 본 연구에서는 NeRF-US라는 새로운 접근법을 소개합니다. 이 모델은 3D 기하학 (3D geometry) 가이드를 NeRF 훈련에 통합하고, 전통적인 볼륨 렌더링 대신 초음파 전용 렌더링을 사용하여 고립된 환경에서 수집된 초음파 데이터의 3D 재구성과 새로운 시각 합성을 해결합니다.

- **Technical Details**: NeRF-US는 3D denoising diffusion model을 사용하여 기하학적 사전 (geometric priors)을 결합하여 3D 재구성을 유도합니다. 이 모델은 3D 벡터를 입력받아 5D 벡터 (감쇠, 반사율, 경계 확률, 산란 밀도 및 산란 강도)를 학습합니다. 또한, Diffusion model에서 얻은 기하학적 정보가 경계 확률과 산란 밀도를 도와주는 역할을 합니다.

- **Performance Highlights**: 실험을 통해 'Ultrasound in the Wild' 데이터셋에서 NeRF-US는 정확하고 임상적으로 그럴듯한, 아티팩트가 없는 3D 재구성을 보여주었습니다. 기존 방법들과 비교해 성능이 크게 개선되었습니다.



### Target-Dependent Multimodal Sentiment Analysis Via Employing Visual-to Emotional-Caption Translation Network using Visual-Caption Pairs (https://arxiv.org/abs/2408.10248)
- **What's New**: 이번 연구에서는 Target-Dependent Multimodal Sentiment Analysis (TDMSA) 방법론을 사용하여 멀티모달 포스트에서 각 타겟(측면)과 관련된 감정을 효과적으로 식별하는 새로운 접근 방식을 제시합니다.

- **Technical Details**: 우리는 Visual-to-Emotional-Caption Translation Network (VECTN)이라는 새로운 기술을 개발하였으며, 이를 통해 얼굴 표정에서 시각적 감정 단서를 성공적으로 추출하고 이를 텍스트의 타겟 속성과 정렬 및 통합할 수 있습니다.

- **Performance Highlights**: 실험 결과, Twitter-2015 데이터셋에서 81.23%의 정확도와 80.61%의 macro-F1 점수를, Twitter-2017 데이터셋에서는 77.42%의 정확도와 75.19%의 macro-F1 점수를 달성하였습니다. 이를 통해 우리의 모델이 멀티모달 데이터에서 타겟 수준의 감정을 수집하는 데 있어 다른 모델보다 우수함을 보여주었습니다.



### VyAnG-Net: A Novel Multi-Modal Sarcasm Recognition Model by Uncovering Visual, Acoustic and Glossary Features (https://arxiv.org/abs/2408.10246)
- **What's New**: 본 연구에서는 대화에서의 sarcasm 인식을 위한 새로운 접근법인 VyAnG-Net을 제안합니다. 이 방법은 텍스트, 오디오 및 비디오 데이터를 통합하여 더 신뢰할 수 있는 sarcasm 인식을 가능하게 합니다.

- **Technical Details**: 이 방법은 lightweight depth attention 모듈과 self-regulated ConvNet을 결합하여 시각 데이터의 핵심 특징을 집중적으로 분석하고, 텍스트 데이터에서 문맥에 따른 중요 정보를 추출하기 위한 attentional tokenizer 기반 전략을 사용합니다. Key contributions로는 subtitles에서 glossary content의 유용한 특징을 추출하는 attentional tokenizer branch, 비디오 프레임에서 주요 특징을 얻는 visual branch, 음향 콘텐츠에서 발화 수준의 특징 추출, 여러 모달리티에서 획득한 특징을 융합하는 multi-headed attention 기반 feature fusion branch가 포함됩니다.

- **Performance Highlights**: MUSTaRD 벤치마크 비디오 데이터셋에서 speaker dependent 및 speaker independent 환경에서 각각 79.86% 및 76.94%의 정확도로 기존 방법들에 비해 우수함을 입증하였습니다. 또한, MUStARD++ 데이터셋의 보지 않은 샘플을 통해 VyAnG-Net의 적응성을 평가하는 교차 데이터셋 분석도 수행하였습니다.



### A Comprehensive Survey on Diffusion Models and Their Applications (https://arxiv.org/abs/2408.10207)
- **What's New**: 이번 리뷰 논문은 확산 모델(Diffusion Models, DMs)에 대한 포괄적인 개요를 제공하며, 이러한 모델들이 이론적 기반 및 알고리즘 혁신을 포함한 다양한 응용 분야에서 어떻게 활용되고 있는지 강조합니다.

- **Technical Details**: 확산 모델은 데이터에 점진적으로 노이즈를 추가하고 제거하여 현실적인 샘플을 생성하는 확률적 모델입니다. 이 리뷰에서는 DDPMs, NCSNs, SDEs 등 여러 유형의 DMs를 분류하고 이론적 기초와 알고리즘 변형을 이해하는 데 도움을 줍니다.

- **Performance Highlights**: DMs는 이미지 생성, 오디오 합성, 자연어 처리 및 의료 분야 등에서 뛰어난 성능을 보여주고 있으며, 특히 고품질 샘플을 생성하는 데 있어 그 효용이 입증되었습니다. 이 논문은 DMs의 응용을 통해 다양한 분야에서의 협력 및 혁신을 촉진하는 것을 목표로 하고 있습니다.



### Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Mod (https://arxiv.org/abs/2408.11039)
Comments:
          23 pages

- **What's New**: 이번 논문에서는 텍스트와 이미지 데이터를 혼합한 다중 모달 모델 학습을 위한 "Transfusion" 방법론을 소개합니다. 이 방법은 언어 모델 손실 함수(다음 토큰 예측)와 확산 모델(difussion)을 결합하여 단일 변환기(transformer) 모델을 훈련시킵니다.

- **Technical Details**: Transfusion 모델은 텍스트 및 이미지 데이터를 50%씩 혼합하여 훈련되며, 텍스트에는 다음 토큰 예측 손실을, 이미지에는 확산 손실을 사용합니다. 모델은 각 훈련 단계에서 두 가지 모달리티(modality)와 손실 함수를 모두 처리합니다. 이미지 처리를 위해 패치 벡터(patch vectors)를 사용하고, 텍스트 토큰에 대해 인과적 주의를, 이미지 패치에 대해 양방향 주의를 적용합니다.

- **Performance Highlights**: Transfusion 모델은 Chameleon의 방법론보다 모든 모달리티 조합에서 더 나은 성능을 보였으며, 텍스트-이미지 생성에서는 1/3 미만의 계산량으로도 FID 및 CLIP 점수에서 더 나은 결과를 달성했습니다. 실험 결과, Transfusion은 기존의 여러 유명 모델들을 뛰어넘는 성능을 보이며, 7B 파라미터 모델에서도 텍스트와 이미지를 동시에 생성할 수 있다는 가능성을 보여주었습니다.



### Atmospheric Transport Modeling of CO$_2$ with Neural Networks (https://arxiv.org/abs/2408.11032)
Comments:
          Code: this https URL

- **What's New**: 본 연구에서는 CO$_2$ 대기 트레이서 운송 모델링을 위한 최신 기술인 심층 신경망(Deep Neural Networks)을 제시합니다. 특히, SwinTransformer 모델이 특히 뛰어난 성능을 보였으며, 이를 통해 대기 트레이서의 지속적인 예측을 수행할 수 있는 능력이 있음을 보여줍니다.

- **Technical Details**: 우리는 Eulerian 대기 운송에 대한 기계 학습 에뮬레이터의 학습 및 테스트를 위한 새로운 데이터셋(CarbonBench)을 생성했습니다. 역학 기반 조정을 통해 강력한 경험적 성능을 가능하게 하는 SwinTransformer 기반의 에뮬레이터를 개발하였고, 다른 세 가지 심층 신경망 아키텍처(UNet, GraphCast, SFNO)와 비교하여 우수한 성능을 냈습니다. 또한 이 모델은 CO$_2$의 농도 필드를 자동적으로 예측할 수 있는 autoregressive (자기 회귀) 특성을 가지고 있습니다.

- **Performance Highlights**: SwinTransformer는 90일 간 R² > 0.99의 성능을 기록하며 물리적으로 타당한 예측을 제공합니다. 모든 네 개 신경망 아키텍처는 6개월 이상 안정적이고 질량 보존적인 운송을 수행할 수 있는 능력을 가집니다. 이 연구는 World Meteorological Organization의 Global Greenhouse Gas Watch 지원을 위한 고해상도 CO$_2$ 역 모델링을 향한 첫 걸음을 제시합니다.



### Denoising Plane Wave Ultrasound Images Using Diffusion Probabilistic Models (https://arxiv.org/abs/2408.10987)
- **What's New**: 이 논문에서는 초음파 평면웨이브(plane wave) 이미지를 향상시키기 위한 Denoising Diffusion Probabilistic Models (DDPM)을 도입합니다. 기존의 고배율 초음파 이미징에서 발생하는 노이즈 문제를 해결하기 위해 DDPM을 적용한 새로운 방법을 제안합니다. 저각(compounding plane waves)과 고각(compounding plane waves) 초음파의 차이를 노이즈로 간주하고 이를 효과적으로 제거하여 이미지 품질을 개선합니다.

- **Technical Details**: 제안된 방법은 400개의 시뮬레이션 된 이미지를 사용하여 DDPM을 훈련시키며, 자연 이미지 분할 마스크를 강도 맵(intensity maps)으로 사용하여 다양한 해부학적 형상의 정확한 노이즈 제거를 달성합니다. 또한, 저화질 평면웨이브와 고화질 평면웨이브 사이의 차이를 변동 요소로 사용하여, 정규 노이즈가 아닌 저화질 이미지를 초기화하여 필요한 역단계의 수를 줄입니다.

- **Performance Highlights**: 시뮬레이션, 팬텀, 그리고 인비보(in vivo) 이미지를 비롯한 평가를 통해 제안된 방법은 이미지 품질을 개선하며, 다른 방법들과 비교했을 때 여러 평가 지표에서 우수한 성능을 나타냅니다. 연구진은 소스 코드와 훈련된 모델을 공개할 예정입니다.



### ISLES'24: Improving final infarct prediction in ischemic stroke using multimodal imaging and clinical data (https://arxiv.org/abs/2408.10966)
- **What's New**: ISLES'24 챌린지를 통해 급성 뇌졸중 영상 및 임상 데이터를 기반으로 최종 치료 후 뇌경색 예측을 위한 새로운 접근 방식을 제시합니다. 이 챌린지는 전체 CT 이미지를 포함하여 다양한 데이터 세트를 활용하여 뇌졸중 영상 분석의 표준화를 추진합니다.

- **Technical Details**: ISLES'24는 비대칭 CT(비대비 CT(NCCT), CT 혈관촬영(CTA), 혈관조영 CT(CTP)) 및 후속 MRI를 포함한 데이터 접근을 허용하며, 참가자들은 다양한 CT 기술과 임상 데이터를 결합하여 뇌졸중 병변에 대한 깊이 있는 연구를 할 수 있습니다. ISLES'24 챌린지는 표준화된 벤치마킹과 성능 평가를 통해 궁극적으로 뇌졸중 치료 결정을 지원합니다.

- **Performance Highlights**: ISLES'24 챌린지의 결과는 임상 의사 결정 개선 및 환자 결과 예측 향상에 기여할 것으로 기대됩니다. 또한, 참가자들은 고도로 커리된 데이터 세트를 통해 최적화된 알고리즘 전략을 실행할 수 있는 기회를 가집니다.



### DAAD: Dynamic Analysis and Adaptive Discriminator for Fake News Detection (https://arxiv.org/abs/2408.10883)
- **What's New**: 본 연구는 가짜 뉴스 탐지를 위해 Dynamic Analysis and Adaptive Discriminator (DAAD) 접근법을 제안합니다. 기존의 방법들이 인간의 전문성과 피드백에 지나치게 의존하는 반면, DAAD는 대규모 언어 모델(LLMs)의 자기 반영 능력을 활용한 Monte Carlo Tree Search (MCTS) 알고리즘을 도입하여 프롬프트 최적화를 수행합니다.

- **Technical Details**: DAAD는 네 가지 전형적인 외관 패턴(드라마틱한 과장, 논리적 불일치, 이미지 조작, 의미적 불일치)을 정의하고 이를 탐지하기 위해 네 가지 판별기를 설계했습니다. 각 판별기는 감정과 맥락의 논리성, 이미지 조작 확인, 이미지와 텍스트 의미의 일관성을 평가합니다. 또한, MemoryBank 컴포넌트를 통해 역사적 오류를 저장하고 이를 압축하여 전반적인 지침을 제공합니다. 이와 함께 soft-routing 메커니즘을 사용하여 최적 탐지 모델을 적응적으로 탐색합니다.

- **Performance Highlights**: 세 가지 실제 데이터셋인 Weibo, Weibo21, GossipCop에 대한 폭넓은 실험 결과, DAAD 접근법이 기존의 방법들보다 우수한 성능을 보여주었습니다. 이 연구는 가짜 뉴스 탐지의 유연성과 정확성을 높이기 위한 새로운 경로를 제시하고 있습니다.



### Radio U-Net: a convolutional neural network to detect diffuse radio sources in galaxy clusters and beyond (https://arxiv.org/abs/2408.10871)
Comments:
          Accepted by MNRAS, 16 pages, 9 figures, 2 tables

- **What's New**: 새로운 라디오 망원경 배열 세대가 감도(sensitivity)와 해상도(resolution)에서 큰 발전을 약속하고 있으며, 이는 새로운 희미하고 확산된 라디오 소스들을 식별하고 특성화하는 데 도움을 줄 것이다. 이러한 발전을 활용하기 위해 Radio U-Net을 소개한다.

- **Technical Details**: Radio U-Net은 U-Net 아키텍처에 기반한 완전한 합성곱 신경망(fully convolutional neural network)으로, 라디오 조사의 희미하고 확장된 소스(예: radio halos, relics, cosmic web filaments)를 탐지하도록 설계되었다. 이 모델은 우주론적 시뮬레이션에 기반한 합성 라디오 관측 데이터로 훈련되었으며, 그런 후 LOFAR Two Metre Sky Survey (LoTSS) 데이터를 통해 게가 클러스터의 확산 라디오 소스를 시각적으로 검토하여 검증되었다.

- **Performance Highlights**: 246개의 게가 클러스터의 테스트 샘플에서, Radio U-Net은 73%의 정확도로 확산 라디오 방출이 있는 클러스터와 없는 클러스터를 구별하였다. 또한, 83%의 클러스터에서 확산 라디오 방출이 정확하게 식별되었고, 저품질 이미지를 포함한 다양한 이미지에서 소스의 형태(morphology)를 성공적으로 복원하였다.



### MambaDS: Near-Surface Meteorological Field Downscaling with Topography Constrained Selective State Space Modeling (https://arxiv.org/abs/2408.10854)
- **What's New**: MambaDS라는 새로운 모델이 기상 분야의 downscaling에 통합되어, 기상 데이터의 고해상도 예측을 위한 다변량 상관관계를 보다 효과적으로 활용하고 지형 정보를 효율적으로 통합합니다.

- **Technical Details**: MambaDS는 Visual State Space Module (VSSM)에서 시작하여 Multivariable Correlation-Enhanced VSSM (MCE-VSSM)을 제안하며, 지형 제약 계층을 설계합니다. 이 구조는 기존 CNN 및 Transformer 기반 모델의 한계를 극복하고, 복잡성을 줄이면서 전세계적 문맥을 효과적으로 포착합니다.

- **Performance Highlights**: MambaDS는 중국 본토와 미국 대륙 내에서 진행된 여러 실험에서 기존의 CNN 및 Transformer 기반 모델보다 향상된 결과를 보여주어, 기상 데이터 downscaling 작업에서 최첨단 성능을 달성했습니다.



### CO2Wounds-V2: Extended Chronic Wounds Dataset From Leprosy Patients (https://arxiv.org/abs/2408.10827)
Comments:
          2024 IEEE International Conference on Image Processing (ICIP 2024)

- **What's New**: 이 논문은 레프루시(Leprosy) 환자의 만성 상처에 대한 RGB 이미지로 구성된 'CO2Wounds-V2' 데이터셋을 소개하고 있습니다. 이 데이터셋은 세그멘테이션(segmentation) 주석과 함께 제공되어 의료 이미지 분석 알고리즘 개발 및 테스트를 위한 기반 자료로 활용될 것입니다.

- **Technical Details**: CO2Wounds-V2 데이터셋은 96명의 레프루시 환자로부터 764개의 RGB 이미지를 포함하고 있으며, 세그멘테이션 주석은 COCO 형식으로 제공됩니다. 또한, 이 데이터셋은 의료 환경에서 이미지 수집을 위해 스마트폰 카메라로 촬영되었으며, 치료의 모든 단계에서 사용 가능합니다.

- **Performance Highlights**: 이 데이터셋의 출시는 이미지 처리 알고리즘의 정확도를 높이고, 다양한 임상 데이터셋에 대한 적응성을 개선할 것으로 기대됩니다. 최근 연구 결과에 따르면, 기존의 알고리즘들은 임상 환경의 차이로 인해 제한적인 성과를 보여왔으나, 이 데이터셋은 이를 극복할 수 있는 기회를 제공합니다.



### Classification of Endoscopy and Video Capsule Images using CNN-Transformer Mod (https://arxiv.org/abs/2408.10733)
- **What's New**: 이번 연구는 전통적인 내시경 이미지 분석 방법에 대한 혁신적인 접근을 제시하며, 컴퓨터 비전을 위한 최신 AI 기법인 CNN과 Swin Transformer를 통합한 하이브리드 모델을 제안합니다.

- **Technical Details**: 이 모델은 DenseNet201을 사용하여 로컬 특성을 추출하고, Swin Transformer를 통해 글로벌 특징을 이해하는 방식으로 내시경 이미지를 분류합니다. CNN 분기와 Transformer 분기를 결합하여 마지막 분류 작업을 수행하는 구조를 갖춥니다. 이는 고유한 레이어 구조와 특징 변환을 포함하여 성능을 극대화합니다.

- **Performance Highlights**: GastroVision 데이터셋에서 0.8320의 Precision, 0.8386의 Recall, 0.8324의 F1 점수, 0.8386의 Accuracy, 0.8191의 Matthews Correlation Coefficient (MCC)를 기록하며 성능이 우수함을 입증했습니다. Kvasir-Capsule 데이터셋에서도 전체 Precision 0.7007, Recall 0.7239, F1 점수 0.6900, Accuracy 0.7239, MCC 0.3871을 달성해 다른 모델에 비해 우세함을 보였습니다.



### deepmriprep: Voxel-based Morphometry (VBM) Preprocessing via Deep Neural Networks (https://arxiv.org/abs/2408.10656)
- **What's New**: 본 논문에서는 deepmriprep이라는 신경망 기반의 VBM(Voxel-based Morphometry) 분석을 위한 전처리 파이프라인을 소개합니다. 기존 CAT12 도구보다 37배 빠른 속도로 대규모 MRI 데이터를 처리할 수 있는 혁신적인 방법을 제공합니다.

- **Technical Details**: deepmriprep은 뇌 추출(brain extraction), 조직 분할(tissue segmentation), 공간 등록(spatial registration)을 포함한 세 가지 주요 전처리 단계를 모두 신경망을 사용하여 수행합니다. 뇌 추출에는 deepbet를, 조직 분할은 패치 기반의 3D UNet을, 비선형 이미지 등록에는 SYMNet을 변형한 모델을 사용합니다.

- **Performance Highlights**: deepmriprep은 CAT12와 유사한 정확도를 유지하면서도 전처리 속도가 37배 빨라 대규모 신경 이미징 연구에 적합합니다. 높은 처리 속도 덕분에 연구자들은 방대한 데이터 세트를 신속하게 전처리할 수 있습니다.



### Generating Multi-frame Ultrawide-field Fluorescein Angiography from Ultrawide-field Color Imaging Improves Diabetic Retinopathy Stratification (https://arxiv.org/abs/2408.10636)
Comments:
          27 pages, 2 figures

- **What's New**: 본 연구는 비침습적으로 얻은 UWF 색깔 망막 이미지(UWF-CF)를 이용하여 염료 없이 생성된 초광각 형광 안과 촬영(UWF-FA) 이미지를 평가하고, 당뇨병성 망막병증(DR) 검진에 대한 효과를 검토하였습니다.

- **Technical Details**: 생성적 적대 신경망(Generative Adversarial Networks, GAN)을 기반으로 한 모델에 18,321개의 다양한 단계의 UWF-FA 이미지와 해당 UWF-CF 이미지를 등록 및 훈련시켰습니다. 생성된 UWF-FA 이미지의 품질은 정량적 메트릭과 인적 평가를 통해 평가되었습니다. 또한, DeepDRiD 데이터셋을 사용하여 생성된 UWF-FA 이미지의 DR 분류에 대한 기여도를 외부에서 평가하였습니다.

- **Performance Highlights**: 생성된 초기, 중기, 후기 단계의 UWF-FA 이미지는 0.70에서 0.91까지의 다중 스케일 유사성 점수와 1에서 1.98까지의 정성적 비주얼 점수를 기록하며 높은 진위도를 보였습니다. Turing 테스트에서 무작위로 선택한 50개 이미지 중 56%에서 76%의 생성 이미지가 실제 이미지와 구별하기 어려웠습니다. DR 분류에 생성된 UWF-FA 이미지를 추가하였을 때 AUROC가 0.869에서 0.904로 유의미하게 증가하였습니다(P < .001). 이 모델은 IV 염료 주입 없이 사실적인 멀티 프레임 UWF-FA 이미지를 성공적으로 생성하였습니다.



### OMEGA: Efficient Occlusion-Aware Navigation for Air-Ground Robot in Dynamic Environments via State Space Mod (https://arxiv.org/abs/2408.10618)
Comments:
          OccMamba is Coming!

- **What's New**: 본 논문에서는 OMEGA라는 새로운 시스템을 제안합니다. OMEGA는 동적 환경에서의 시공 식별과 경로 계획을 향상시키기 위해 OccMamba와 Efficient AGR-Planner를 통합한 것입니다. 이 시스템은 3D 의미적 점유 네트워크와 함께 에너지 효율적인 경로 계획 방식을 제공합니다.

- **Technical Details**: OMEGA의 핵심 구성 요소인 OccMamba는 의미 예측과 점유 예측을 독립적인 브랜치로 분리하여 각 영역에서 전문화된 학습을 가능하게 합니다. 이를 통해 3D 공간에서 시맨틱 및 기하학적 특징을 추출하며, 이 과정에서 전압이 얇고 효율적인 계산을 특징으로 합니다. 이렇게 얻어진 의미적 점유 맵은 지역 맵에 통합되어 동적 환경의 오클루전(occlusion) 인식을 제공합니다.

- **Performance Highlights**: 실험 결과, OccMamba는 최신 3D 의미적 점유 네트워크보다 25.0%의 mIoU(Mean Intersection over Union)를 기록하며, 22.1 FPS의 고속 추론을 달성합니다. OMEGA는 동적 시나리오에서 98%의 성공률을 보이며, 평균 운동 시간을 최단인 16.1초로 기록했습니다. Dynamic 환경에서의 실험에서도 OMEGA는 약 18%의 에너지 소비 절감 효과를 나타냈습니다.



### Vision Calorimeter for Anti-neutron Reconstruction: A Baselin (https://arxiv.org/abs/2408.10599)
- **What's New**: 본 연구에서는 반중성자(anti-neutron, $ar{n}$) 재구성을 위한 새로운 방법론인 비전 칼로리미터(Vision Calorimeter, ViC)를 소개합니다. ViC는 EMC 응답과 반중성자의 특성 간의 implicit 관계를 분석하기 위해 딥러닝(ddeep learning) 기법을 활용합니다.

- **Technical Details**: ViC는 EMC에서 수집된 데이터를 기반으로 하는 2D 이미지 변환을 통해 반중성자의 입장 위치와 운동량을 예측하는 두 가지 과제를 수행합니다. 이 과정에서 pseudo bounding box를 생성하고, 맞춤형 loss function을 통해 훈련 목표를 향상시킵니다.

- **Performance Highlights**: 실험 결과, ViC는 기존 방법에 비해 입장 위치 예측 오차를 42.81% 감소시켰으며(17.31$^{	ext{°}}$에서 9.90$^{	ext{°}}$로), 최초로 반중성자의 운동량 측정을 가능하게 하여 입자 재구성에서 딥러닝의 가능성을 보여줍니다.



### An Efficient Sign Language Translation Using Spatial Configuration and Motion Dynamics with LLMs (https://arxiv.org/abs/2408.10593)
Comments:
          Under Review

- **What's New**: 이번 연구에서는 면밀하게 설계된 Spatial and Motion-based Sign Language Translation (SpaMo) 프레임워크를 소개하여, 자원 집약적인 비주얼 인코더의 파인 튜닝 없이도 수화 비디오에서 예측하는 과정을 개선하고 있습니다.

- **Technical Details**: SpaMo는 사전 훈련된 이미지 인코더(ViT)와 비디오 인코더(VideoMAE)를 활용하여 공간적 특징과 운동 동작을 추출합니다. 추출된 특징은 LLM(대규모 언어 모델)에 입력되며, Visual-Text Alignment(VT-Align) 과정을 통해 서로 다른 모달리티 간의 차이를 줄입니다.

- **Performance Highlights**: SpaMo는 PHOENIX14T와 How2Sign 두 가지 인기있는 데이터셋에서 최첨단 성능을 기록하였으며, 자원 집약적인 접근법 없이 효과적인 수화 번역을 가능하게 하고 있습니다.



### A Tutorial on Explainable Image Classification for Dementia Stages Using Convolutional Neural Network and Gradient-weighted Class Activation Mapping (https://arxiv.org/abs/2408.10572)
Comments:
          15 pages, 11 figures, 3 tables

- **What's New**: 이 논문은 CNN(Convolutional Neural Network)과 Grad-CAM(Gradient-weighted Class Activation Mapping)을 이용하여 진행성 치매의 네 단계(4 progressive dementia stages)를 분류하는 설명 가능한 접근법을 제시합니다.

- **Technical Details**: 제안된 CNN 아키텍처는 오픈 MRI 뇌 이미지(open MRI brain images)를 기반으로 하여 훈련되었으며, 구현 단계에 대한 자세한 설명이 포함되어 있습니다. Grad-CAM을 사용하여 CNN의 블랙 박스(black box) 특성을 극복하고 높은 정확도의 시각적 이유를 제공합니다.

- **Performance Highlights**: 제안된 CNN 아키텍처는 테스트 데이터셋에서 99% 이상의 정확도를 달성하였으며, 이러한 높은 정확도가 의사들에게 유용한 정보를 제공할 가능성을 보입니다.



### Prompt Your Brain: Scaffold Prompt Tuning for Efficient Adaptation of fMRI Pre-trained Mod (https://arxiv.org/abs/2408.10567)
Comments:
          MICCAI 2024

- **What's New**: Scaffold Prompt Tuning (ScaPT)은 fMRI 프리트레인 모델을 하위 작업에 적응시키기 위한 새로운 프롬프트 기반 프레임워크입니다. ScaPT는 파라미터 효율성이 높고, 기존의 fine-tuning 및 프롬프트 튜닝에 비해 성능이 개선되었습니다.

- **Technical Details**: ScaPT는 고급 리소스 작업에서 습득한 지식을 저급 리소스 작업으로 전송하는 계층적 프롬프트 구조를 설계합니다. Deeply-conditioned Input-Prompt (DIP) 매핑 모듈이 포함되어 있으며, 총 학습 가능한 파라미터의 2%만 업데이트합니다. 이 프레임워크는 입력과 프롬프트 간의 주의 메커니즘을 통해 의미론적 해석 가능성을 향상시킵니다.

- **Performance Highlights**: 대중의 휴식 상태 fMRI 데이터 세트를 통해 ScaPT는 신경퇴행성 질병 진단/예후 및 성격 특성 예측에서 기존의 fine-tuning 방법과 멀티태스크 기반 프롬프트 튜닝 방식을 초과하는 성과를 보였습니다. 20명 이하의 참가자와 함께 실험해도 뛰어난 성능이 입증되었습니다.



### Kalib: Markerless Hand-Eye Calibration with Keypoint Tracking (https://arxiv.org/abs/2408.10562)
Comments:
          The code and supplementary materials are available at this https URL

- **What's New**: 이 논문에서는 Kalib이라는 자동화된 비마커 손-눈 보정(Hand-Eye Calibration) 파이프라인을 제안합니다. 이 시스템은 시각적 기초 모델(visual foundation models)의 일반화 가능성을 활용하여 기존의 수작업 및 물리적 마커에 대한 의존성을 없앱니다.

- **Technical Details**: Kalib은 키포인트 추적(keypoint tracking)과 고유 수용 센서(proprioceptive sensors)를 사용하여 로봇의 좌표 공간과 카메라의 좌표 공간 간의 변환을 추정합니다. 이 방법은 새로운 네트워크 훈련이나 정밀한 메시 모델(mesh model)을 요구하지 않습니다.

- **Performance Highlights**: 시뮬레이션 환경과 실제 데이터셋 DROID에서 평가한 결과, Kalib은 최신 기준 방법들과 비교하여 뛰어난 정확도를 보여주었습니다. 위치 오차는 평균 0.3 cm, 회전 오차는 평균 0.4도에 불과하여 전통적인 마커 기반 방법과 동등한 성능을 보였습니다.



### SZTU-CMU at MER2024: Improving Emotion-LLaMA with Conv-Attention for Multimodal Emotion Recognition (https://arxiv.org/abs/2408.10500)
- **What's New**: 본 논문은 MER2024 Challenge의 MER-NOISE 및 MER-OV 트랙에서 우승한 접근 방식을 제시합니다. 이 시스템은 감정 이해 능력을 향상시키기 위해 Emotion-LLaMA를 활용하여 레이블이 없는 샘플에 대한 고품질 주석을 생성함으로써 제한된 레이블 데이터 문제를 해결합니다.

- **Technical Details**: 우리는 Conv-Attention이라는 경량화된 하이브리드 프레임워크를 도입하여 멀티모달 (multimodal) 기능 융합을 개선하고 모드별 노이즈를 완화합니다. Emotion-LLaMA는 레이블이 없는 MER2024 데이터셋을 위한 고품질의 의사 레이블을 생성하도록 설계된 모델로, 여러 모드에서 입력을 처리하여 텍스트 기반 감정을 해석하는 데 도움을 줍니다.

- **Performance Highlights**: MER-NOISE 트랙에서 우리는 85.30%의 가중 평균 F-score를 달성하여 2위와 3위 팀을 각각 1.47% 및 1.65% 초과했습니다. MER-OV 트랙에서는 Emotion-LLaMA를 이용한 오픈 보캐뷸러리 주석이 GPT-4V에 비해 평균 정확도 및 재현율이 8.52% 향상되었으며, 모든 참여하는 대형 멀티모달 모델 중 최고 점수를 기록하였습니다.



### Cervical Cancer Detection Using Multi-Branch Deep Learning Mod (https://arxiv.org/abs/2408.10498)
- **What's New**: 본 연구는 Multi-Head Self-Attention (MHSA) 및 Convolutional Neural Networks (CNNs)을 활용하여 자궁경부암 이미지를 자동으로 분류하는 혁신적인 접근법을 제안합니다. 이 방법은 자궁경부 이미지를 두 개의 스트림에서 처리하여 로컬 및 글로벌 특징을 효과적으로 포착하는 것을 목표로 합니다.

- **Technical Details**: 신청된 모델은 Grain Module을 통해 세분화된 특징을 생성하고, 이를 MHSA와 CNN 모듈로 나누어 처리합니다. 이 과정에서 MHSA는 흥미로운 영역에 집중하고, CNN은 계층적 특징을 추출하여 정확한 분류를 지원합니다. 최종적으로 두 스트림의 특징을 결합하여 분류 모듈에 입력하여 특징 개선 및 분류를 시행합니다.

- **Performance Highlights**: 이 연구에서 제안한 모델은 SIPaKMeD 데이터셋을 사용하여 98.522%라는 뛰어난 정확도를 달성하였으며, 이는 의료 이미지 분류에서 높은 인식 정확성을 보여주고 다른 의료 이미지 인식 작업에도 적용 가능성을 시사합니다.



### Learning Multimodal Latent Space with EBM Prior and MCMC Inferenc (https://arxiv.org/abs/2408.10467)
- **What's New**: 이번 연구에서는 멀티모달 생성 모델을 위한 새로운 접근 방식으로, 에너지 기반 모델(EBM) 사전과 마르코프 체인 몬테 카를로(MCMC) 추론을 결합하였습니다. 이를 통해 멀티모달 데이터의 복잡성을 효과적으로 캡처하고, 생성 과정의 일관성을 높였습니다.

- **Technical Details**: 우리가 제안하는 모델은 에너지 기반 모델(EBM) 사전을 사용하여 비정보적 사전 대신 배치되고, MCMC 추론을 통해 후방 분포를 더 정확하게 근사합니다. 또한, 짧은 런의 랑겐빈 다이내믹스를 사용하여 EBM 학습을 개선하는 방법을 제공합니다.

- **Performance Highlights**: 실험 결과, 제안한 EBM 사전과 MCMC 추론 기법이 멀티모달 생성 과제에서 교차 모달 및 공동 생성 작업의 개선을 보여주었습니다. 시각적 및 수치적 방법으로 효과가 검증되었습니다.



### Feasibility of assessing cognitive impairment via distributed camera network and privacy-preserving edge computing (https://arxiv.org/abs/2408.10442)
- **What's New**: 본 연구는 경증 인지장애(Mild Cognitive Impairment, MCI) 환자 간의 사회적 상호작용과 이동 패턴을 자동으로 캡처하기 위한 프라이버시 보호 분산 카메라 네트워크를 활용하여 장기 모니터링을 향상시키는 데 목표를 두었습니다. 이는 MCI 환자의 다양한 인지 기능 수준을 효과적으로 구별할 수 있는 새로운 방법론을 제시합니다.

- **Technical Details**: 연구는 1700$m^2$ 공간에서 경증 인지장애 환자들로부터 수집된 이동 및 사회적 상호작용 데이터를 사용하여, 기계 학습 알고리즘을 통해 고기능(Group with High Cognitive Function)과 저기능(Group with Low Cognitive Function)인 집단을 구별하는 특징(feature)을 개발했습니다. 39대의 엣지 컴퓨팅 및 카메라 장치가 사용되어 실시간으로 체형 추정(pose estimation) 모델을 운영하였으며, 평균 1.41m의 로컬리제이션 오차를 달성했습니다.

- **Performance Highlights**: 연구 결과, 선형 경로 길이(linear path length), 보행 속도(walking speed), 방향 변화의 변화, 속도 및 방향의 엔트로피(entropy) 등 여러 주요 특징에서 고기능 집단과 저기능 집단 간에 통계적으로 유의미한 차이를 발견하였습니다. 이 연구의 기계 학습 접근 방식은 71%의 정확도로 인지 기능 수준을 구별하는데 성공하였습니다.



### AIR: Analytic Imbalance Rectifier for Continual Learning (https://arxiv.org/abs/2408.10349)
- **What's New**: 본 논문에서는 데이터 불균형 문제를 해결하기 위해 고객화된 온라인 예시 없는 지속적 학습 방법 AIR(Analytic Imbalance Rectifier)를 제안합니다. 이 방법은 데이터-불균형 클래스 증분 학습(CIL)과 일반화된 CIL 시나리오를 위한 닫힌 형태 솔루션을 제공합니다.

- **Technical Details**: AIR는 손실 함수에서 각 클래스의 기여를 균형있게 조절하기 위해 재가중치 인자를 계산하는 분석적 재가중치 모듈(ARM)을 도입합니다. 이를 통해 AIR는 비구분적인 최적 분류기를 제공하며, 지속적 학습에서의 반복적인 업데이트 방법을 사용합니다.

- **Performance Highlights**: 여러 데이터셋에 대한 실험 결과, AIR는 긴 꼬리(LT) 및 일반화된 GCIL 시나리오에서 기존의 방법들보다 상당히 뛰어난 성능을 보여주었습니다.



### SDE-based Multiplicative Noise Remova (https://arxiv.org/abs/2408.10283)
Comments:
          9 pages, 4 figures

- **What's New**: 본 논문은 Stochastic Differential Equations (SDEs)를 활용하여 multiplicative noise를 제거하는 새로운 접근 방식을 제안합니다. 기존의 additive noise 제거 기술과는 달리, multiplicative noise는 시스템 내부에서 발생하며, 이로 인해 이미지의 질이 저하됩니다.

- **Technical Details**: 제안된 방법론에서 multiplicative noise는 Geometric Brownian Motion으로 모델링되며, Fokker-Planck 방정식을 통해 역 SDE를 도출하여 이미지의 denoising을 수행합니다. 이 방법은 2개의 데이터셋에서 extensive한 실험을 거쳐 검증되었습니다.

- **Performance Highlights**: 실험 결과, FID 및 LPIPS와 같은 perception-based metrics에서 기존 기법을 대폭 초월하며, PSNR 및 SSIM과 같은 전통적인 metrics에서도 경쟁력 있는 성능을 보였습니다.



### AltCanvas: A Tile-Based Image Editor with Generative AI for Blind or Visually Impaired Peop (https://arxiv.org/abs/2408.10240)
- **What's New**: AltCanvas는 시각 장애인을 위한 그림 제작 도구로, 텍스트-투-이미지 생성 AI의 능력을 활용한 구조적인 접근 방식을 통합하여, 사용자에게 향상된 제어 및 편집 기능을 제공합니다.

- **Technical Details**: AltCanvas는 사용자가 시각 장면을 점진적으로 구성할 수 있는 타일 기반 인터페이스를 제공합니다. 각 타일은 장면 내의 객체를 나타내며, 사용자들은 음성 및 오디오 피드백을 받으며 개체를 추가, 편집, 이동 및 배열할 수 있습니다.

- **Performance Highlights**: AltCanvas의 워크플로를 통해 14명의 시각 장애인 참가자들이 효과적으로 일러스트를 제작하였으며, 색상 일러스트레이션이나 촉각 그래픽 생성을 위한 벡터로 렌더링할 수 있는 기능을 제공하였습니다.



### AID-DTI: Accelerating High-fidelity Diffusion Tensor Imaging with Detail-preserving Model-based Deep Learning (https://arxiv.org/abs/2408.10236)
Comments:
          12 pages, 3 figures, MICCAI 2024 Workshop on Computational Diffusion MRI. arXiv admin note: text overlap with arXiv:2401.01693, arXiv:2405.03159

- **What's New**: 본 논문에서는 최소 6번의 측정만으로도 고속 및 고정밀 확산 텐서 이미징(DTI)을 가능하게 하는 새로운 방법, AID-DTI(Acclerating high fidelity Diffusion Tensor Imaging)를 제안합니다.

- **Technical Details**: 제안된 AID-DTI는 성분 수 분해(Singular Value Decomposition, SVD) 기반의 정규화 방법을 통해 세부 사항을 효과적으로 포착하고 네트워크 훈련 시 노이즈를 억제합니다. 또한, Nesterov 기반의 적응형 학습 알고리즘을 도입하여 정규화 매개변수를 동적으로 최적화하여 성능을 향상시킵니다.

- **Performance Highlights**: 휴먼 커넥톰 프로젝트(Human Connectome Project, HCP) 데이터에서 실행된 실험 결과, 제안된 방법이 정량적 및 정성적으로 현재의 최첨단 방법들을 능가하며 세밀한 DTI 파라미터 맵을 추정함을 입증하였습니다.



### EditShield: Protecting Unauthorized Image Editing by Instruction-guided Diffusion Models (https://arxiv.org/abs/2311.12066)
- **What's New**: 이번 연구에서는 instruction-guided diffusion models에 의한 무단 수정의 문제를 최초로 다루어, EditShield라는 새로운 보호 방법을 제안합니다. 이 방법은 미세한 변화를 추가하여 모델의 잠재 표현(latent representation)을 방해함으로써, 실제 수정 결과와 의도된 수정 결과 간의 불일치를 초래합니다.

- **Technical Details**: EditShield는 잠재 이미지 생성 과정에서 모델이 사용하는 잠재 표현을 방해하는 방식으로 작동합니다. 이 방법은 Gaussian noise를 추가하여 이미지의 품질을 저하시키고 주제가 일치하지 않는 비현실적인 이미지를 생성하도록 모델을 유도합니다. 실험은 합성 및 실제 데이터 세트를 포함하며, EditShield는 다양한 수정 유형과 유사한 명령어에 대해 강력한 성능을 보입니다.

- **Performance Highlights**: EditShield는 사용자가 원하는 대로 수정된 이미지를 쉽게 얻을 수 있지만, 동시에 무단 조작에 대한 우려를 제기합니다. 본 연구 결과에 따르면, EditShield는 다양한 이미지에 대해 효과적인 보호를 제공하며, 무단 편집에 강력한 내성을 가지고 있습니다.



### Screen Them All: High-Throughput Pan-Cancer Genetic and Phenotypic Biomarker Screening from H&E Whole Slide Images (https://arxiv.org/abs/2408.09554)
- **What's New**: 이번 연구는 고속 처리 AI 시스템을 통해 H&E whole slide images (WSIs)에서 1,228개의 유전체 바이오마커를 신속하고 저렴하게 예측할 수 있는 방법을 제시합니다. Virchow2라는 대형 모델을 기반으로 하여, 3백만 장의 슬라이드에서 사전 훈련된 네트워크를 사용했습니다. 이는 각 바이오마커에 대해 개별 모델을 훈련할 필요 없이, 다양한 암 유형 간에 통합적인 예측을 가능하게 합니다.

- **Technical Details**: 연구팀은 MSK-IMPACT 대상 바이오마커 패널을 기반으로 38,984명의 환자로부터 수집된 47,960개의 H&E 슬라이드를 사용하여 모델을 개발했습니다. 이 시스템은 15개의 일반적인 암 유형에서 평균 AUC 0.89를 달성하며, 80개의 고성능 바이오마커를 확인했습니다. 또한 58개의 바이오마커는 치료 선택 및 반응 예측에 관련된 대상과 강한 연관성을 보였습니다.

- **Performance Highlights**: AI 모델은 MSK의 15개 주요 암 유형에서 391개의 유전자 변형 바이오마커를 성공적으로 식별하였습니다. 이는 평균 AUC 0.84, 평균 민감도 0.92, 평균 특이도 0.55를 기록하며, 치료 선택 및 신규 치료 타겟의 탐색에 있어 큰 잠재력을 보입니다.



New uploads on arXiv(cs.AI)

### GraphFSA: A Finite State Automaton Framework for Algorithmic Learning on Graphs (https://arxiv.org/abs/2408.11042)
Comments:
          Published as a conference paper at ECAI 2024

- **What's New**: 본 논문은 그래프(Grapg)에서 작동하는 유한 상태 자동기(Finite State Automaton, FSA)를 학습하기 위한 새로운 프레임워크인 GraphFSA를 제안합니다. 기존 머신러닝 모델이 알고리즘적 결정을 명확하게 표현하는 데 어려움을 겪고 있는 가운데, GraphFSA는 특히 그래프 알고리즘 학습에서의 한계를 극복하려고 합니다.

- **Technical Details**: GraphFSA는 각 그래프의 노드에서 유한 상태 자동기를 실행하도록 설계된 연산 프레임워크입니다. 이 프레임워크는 Graph Cellular Automata와 Graph Neural Networks에서 영감을 받아, 각 노드에서 동일한 FSA를 정의합니다. 이 과정에서 이웃 노드 상태를 집계(aggregation)하여 전이 값(transition values)을 계산하며, 이는 유한한 수의 집계 방식만 처리할 수 있도록 제한됩니다.

- **Performance Highlights**: GraphFSA는 다양한 합성 문제에서 성능 평가를 거쳤으며, 이를 통해 고도화된 그래프 알고리즘을 학습하는 데 효과적임을 보여주었습니다. 특히, GraphFSA는 뛰어난 일반화(generalization) 및 외삽(extrapolation) 능력을 갖추고 있어 기존의 그래프 알고리즘 학습 방식에 비해 대안적인 접근 방식을 제시하고 있습니다.



### Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Mod (https://arxiv.org/abs/2408.11039)
Comments:
          23 pages

- **What's New**: 이번 논문에서는 텍스트와 이미지 데이터를 혼합한 다중 모달 모델 학습을 위한 "Transfusion" 방법론을 소개합니다. 이 방법은 언어 모델 손실 함수(다음 토큰 예측)와 확산 모델(difussion)을 결합하여 단일 변환기(transformer) 모델을 훈련시킵니다.

- **Technical Details**: Transfusion 모델은 텍스트 및 이미지 데이터를 50%씩 혼합하여 훈련되며, 텍스트에는 다음 토큰 예측 손실을, 이미지에는 확산 손실을 사용합니다. 모델은 각 훈련 단계에서 두 가지 모달리티(modality)와 손실 함수를 모두 처리합니다. 이미지 처리를 위해 패치 벡터(patch vectors)를 사용하고, 텍스트 토큰에 대해 인과적 주의를, 이미지 패치에 대해 양방향 주의를 적용합니다.

- **Performance Highlights**: Transfusion 모델은 Chameleon의 방법론보다 모든 모달리티 조합에서 더 나은 성능을 보였으며, 텍스트-이미지 생성에서는 1/3 미만의 계산량으로도 FID 및 CLIP 점수에서 더 나은 결과를 달성했습니다. 실험 결과, Transfusion은 기존의 여러 유명 모델들을 뛰어넘는 성능을 보이며, 7B 파라미터 모델에서도 텍스트와 이미지를 동시에 생성할 수 있다는 가능성을 보여주었습니다.



### Hybrid Recurrent Models Support Emergent Descriptions for Hierarchical Planning and Contro (https://arxiv.org/abs/2408.10970)
Comments:
          4 pages, 3 figures

- **What's New**: 본 논문에서는 연속적인 문제를 해결하기 위해 이산 추상을 유연하게 학습할 수 있는 인공지능 시스템의 필요성을 다룹니다. 특히, Hybrid state-space 모델인 recurrent switching linear dynamical systems (rSLDS)를 활용하여 의미 있는 행동 단위를 발견하고 이를 통해 계획 및 제어에 유용한 추상을 제공하는 방법을 제안합니다.

- **Technical Details**: 우리는 Active Inference에서 영감을 받은 새로운 계층적 모델 기반 알고리즘을 제시합니다. 이 알고리즘은 이산 MDP가 저레벨의 선형-2차 제어기와 상호작용하며, rSLDS가 학습한 반복 전이 동역학을 활용하여 (1) 시간 추상화된 하위 목표를 정의하고, (2) 이산 공간으로의 탐색을 가능하게 하며, (3) 이산 계획기에서 저레벨 문제의 근사 솔루션을 '캐시'합니다. 이 방식으로 우리는 Sparse Continuous Mountain Car 작업에 성공적으로 적용하여 빠른 시스템 식별과 추상 하위 목표의 구체화를 통한 실질적인 계획을 수행했습니다.

- **Performance Highlights**: 모델의 성능을 Continuous Mountain Car 작업에 적용하여 정보 이론적 탐색 기반을 통해 신속한 시스템 식별을 달성하고, 비정상적인 계획 문제에 대한 성공적인 해결책을 도출했습니다. 결과적으로, 우리는 제어 기기가 시스템의 상태-행동 공간에서 원하는 영역으로 제어 입력을 유연하게 지정할 수 있게 되었습니다.



### Dr.Academy: A Benchmark for Evaluating Questioning Capability in Education for Large Language Models (https://arxiv.org/abs/2408.10947)
Comments:
          Accepted to ACL 2024

- **What's New**: 본 연구에서는 LLMs(대형 언어 모델)을 교육자로서 평가하기 위해 질문 생성 능력을 포함한 벤치마크를 소개합니다. 기존 연구가 LLMs를 학습자로 보고 주요 학습 능력을 평가하는 데 집중했던 것과 대조적으로, 이 연구에서는 LLMs가 고품질 교육 질문을 생성할 수 있는지를 평가합니다.

- **Technical Details**: 연구에서는 LLMs의 질문 생성 능력을 평가하기 위해 Anderson과 Krathwohl의 교육적 분류체계를 기반으로 한 6개의 인지 수준이 포함된 Dr.Academy라는 벤치마크를 개발했습니다. 이 벤치마크는 일반, 단일 분야, 다학제 분야 등 세 가지 도메인에서 LLMs가 생성한 질문을 분석합니다. 평가 지표로는 관련성(relevance), 포괄성(coverage), 대표성(representativeness), 일관성(consistency)을 사용합니다.

- **Performance Highlights**: 실험 결과, GPT-4는 일반과 인문, 과학 과정 교육에 상당한 잠재력을 보였으며, Claude2는 다학제 교수에 더 적합한 것으로 나타났습니다. 자동 점수는 인간의 평가와 잘 일치했습니다.



### Large Language Model Driven Recommendation (https://arxiv.org/abs/2408.10946)
- **What's New**: 이번 논문에서는 기존의 비언어적 사용자 피드백(예: 구매, 조회, 클릭)을 기반으로 한 추천 시스템에서 자연어(NL) 상호작용을 통한 추천의 가능성을 제시합니다. 특히, 대형 언어 모델(LLM)의 일반적인 NL 추론 능력을 활용하여 개인화된 추천 시스템을 구축할 수 있는 방법론을 논의합니다.

- **Technical Details**: 논문은 언어 기반 추천을 위한 주요 데이터 소스의 분류법을 제시하며, 항목 설명, 사용자-시스템 상호작용 및 사용자 프로필을 포함합니다. 또한, 조정(tuned) 및 미조정(untuned) 상태에서의 인코더 전용 및 자회귀(autoregressive) LLM 추천 기법을 검토합니다. 이어서 LLM이 검색기(retriever) 및 추천 시스템과 상호작용하는 다중 모듈 추천 아키텍처에 대해서도 논의합니다.

- **Performance Highlights**: 마지막으로, 대화형 추천 시스템(CRSs) 아키텍처에서는 LLM이 다중 턴 대화를 가능하게 하여 사용자와의 상호작용을 통해 추천을 제공하고 선호도를 끌어내며, 비판 및 질문-답변을 진행할 수 있도록 합니다.



### The Evolution of Reinforcement Learning in Quantitative Financ (https://arxiv.org/abs/2408.10932)
Comments:
          This work is currently submitted to and under-review for ACM Computing Surveys. This copy is an unedited, pre-print version and it is the author's version of the work. I

- **What's New**: 이 논문은 금융 분야에서의 Reinforcement Learning (RL) 응용에 대한 최근 조사 결과를 제공합니다. 167개의 출판물을 분석하며, 다양한 RL 응용 및 프레임워크를 탐색합니다.

- **Technical Details**: RL은 복잡한 재무 시장을 대상으로 하여, 샤프 비율(Sharpe ratio), 포트폴리오 관리(Portfolio Management), 자산 배분과 같은 주요 측면의 개선에 중점을 둡니다. 기존의 머신 러닝(Machine Learning) 방법론과 비교할 때 RL은 더 빠르고 동적인 의사결정을 제공합니다.

- **Performance Highlights**: RL의 채택은 전통적인 방법에 비해 금융 시장의 변동성에 더 잘 대응할 수 있습니다. 실시간 학습과 적응을 통해 거래 전략을 지속적으로 개선하며, 이는 높은 빈도의 거래(High-Frequency Trading, HFT) 환경에서 더욱 유리합니다.



### MTFinEval:A Multi-domain Chinese Financial Benchmark with Eurypalynous questions (https://arxiv.org/abs/2408.10921)
- **What's New**: 이 논문에서는 LLMs(대규모 언어 모델)의 생산 배치 적합성을 평가하기 위한 새로운 기준인 MTFinEval을 제안하고, 이 기준이 경제학의 기본 지식을 평가하기 위해 구성되었음을 보여줍니다.

- **Technical Details**: MTFinEval은 대학 경제학 교과서와 시험지를 바탕으로 한 360개의 질문으로 구성되어 있으며, 경제학의 6개 주요 분야(거시경제학, 미시경제학, 회계학, 경영학, 전자상거래, 전략 경영)의 다양한 지식을 평가합니다. 각 질문은 단일 선택, 다중 선택 및 진위 여부로 나뉘어 있으며, 정답률이 모델의 성능을 직접적으로 평가하는 방식으로 진행됩니다.

- **Performance Highlights**: 실험 결과에 따르면, 모든 LLM들이 MTFinEval에서 낮은 성과를 보였으며, 이는 기본 지식을 기반으로 한 기준이 효과적임을 입증합니다. 이 연구는 특정 사용 사례에 적합한 LLM 선택을 위한 가이드를 제공하고, LLM의 신뢰성을 기초에서 강화할 필요성을 강조합니다.



### Towards Efficient Formal Verification of Spiking Neural Network (https://arxiv.org/abs/2408.10900)
- **What's New**: 최근 AI 연구는 주로 대형 언어 모델(LLM)에 집중되고 있으며, 정확성을 높이기 위해 스케일을 늘리고 더 많은 전력을 소비하는 것이 일반적입니다. 그러나 AI의 전력 소비는 사회적인 문제로 부각되고 있으며, 스파이킹 신경망(SNN)은 이 문제에 대한 유망한 해결책으로 제안되고 있습니다.

- **Technical Details**: 스파이킹 신경망(SNN)은 인간의 뇌처럼 이벤트 기반으로 작동하며 정보를 시간적으로 압축합니다. 이로 인해 SNN은 퍼셉트론 기반의 인공 신경망(ANN)에 비해 전력 소비를 크게 줄일 수 있습니다. 본 논문에서는 SNN의 적대적 강건성을 검증하기 위한 효율적인 알고리즘과 시간적 인코딩을 SMT(상태 모듈 이론) 솔버 제약조건으로 공식화하는 방법을 제안합니다.

- **Performance Highlights**: 본 연구의 방법론을 통해 SNN의 검증 시간을 측정하고 이를 기존의 방법들과 비교하여 성능을 향상시키는 방법을 제안합니다. SNN의 안정성과 속성 검증이 실용적인 수준으로 발전하였으며, 이는 SNN의 안전한 적용을 용이하게 합니다.



### Analytical and Empirical Study of Herding Effects in Recommendation Systems (https://arxiv.org/abs/2408.10895)
Comments:
          29 pages

- **What's New**: 온라인 제품 평가 시스템에서 발생하는 집단 효과(herding effects)를 수학적 모델을 통해 분석하고, 역사적인 집단 의견(history collective opinion)의 수렴 조건과 속도를 규명했습니다. 이를 통해 제품 품질을 올바르게 평가하기 위한 등급 집계 규칙과 리뷰 선택 메커니즘을 제안합니다.

- **Technical Details**: 본 연구에서는 집단 효과를 고려한 수학적 모델을 개발하였으며, 확률적 근사(stochastic approximation) 이론을 사용하여 역사적인 집단 의견이 진실의 집단 의견(ground-truth collective opinion)으로 수렴하는 충분 조건을 도출했습니다. 또한, 마틴게일(martingale) 이론을 통해 집계 규칙과 리뷰 선택 메커니즘의 효율성을 정량화했습니다.

- **Performance Highlights**: 실험 결과, 적절한 최신 정보 반영 레이팅 집계 규칙을 사용했을 때 Amazon과 TripAdvisor의 수렴 속도가 각각 41% 및 62% 향상되었음을 보여주었습니다. 이는 온라인 제품 평가 시스템의 정확성과 효율성을 개선하는 데 기여할 수 있습니다.



### On Learning Action Costs from Input Plans (https://arxiv.org/abs/2408.10889)
- **What's New**: 이번 논문은 입력 계획(set of input plans)에서 최적의 계획을 도출하기 위한 작업 비용(action costs)을 학습하는 새로운 문제를 제시합니다. 이를 통해 다양한 계획을 순위 매길 수 있습니다.

- **Technical Details**: 연구진은 LACFIP^k라는 알고리즘을 통해 레이블이 없는 입력 계획에서 작업 비용을 학습하는 방법을 제시합니다. LACFIP^k는 그 이론적 기반과 실증적 결과를 통해 이 문제 해결의 가능성을 보여줍니다.

- **Performance Highlights**: 실험 결과, LACFIP^k는 다양한 계획 분야에서 비용 함수(cost function)를 성공적으로 학습하거나 조정하는 데 사용될 수 있음을 입증했습니다.



### DAAD: Dynamic Analysis and Adaptive Discriminator for Fake News Detection (https://arxiv.org/abs/2408.10883)
- **What's New**: 본 연구는 가짜 뉴스 탐지를 위해 Dynamic Analysis and Adaptive Discriminator (DAAD) 접근법을 제안합니다. 기존의 방법들이 인간의 전문성과 피드백에 지나치게 의존하는 반면, DAAD는 대규모 언어 모델(LLMs)의 자기 반영 능력을 활용한 Monte Carlo Tree Search (MCTS) 알고리즘을 도입하여 프롬프트 최적화를 수행합니다.

- **Technical Details**: DAAD는 네 가지 전형적인 외관 패턴(드라마틱한 과장, 논리적 불일치, 이미지 조작, 의미적 불일치)을 정의하고 이를 탐지하기 위해 네 가지 판별기를 설계했습니다. 각 판별기는 감정과 맥락의 논리성, 이미지 조작 확인, 이미지와 텍스트 의미의 일관성을 평가합니다. 또한, MemoryBank 컴포넌트를 통해 역사적 오류를 저장하고 이를 압축하여 전반적인 지침을 제공합니다. 이와 함께 soft-routing 메커니즘을 사용하여 최적 탐지 모델을 적응적으로 탐색합니다.

- **Performance Highlights**: 세 가지 실제 데이터셋인 Weibo, Weibo21, GossipCop에 대한 폭넓은 실험 결과, DAAD 접근법이 기존의 방법들보다 우수한 성능을 보여주었습니다. 이 연구는 가짜 뉴스 탐지의 유연성과 정확성을 높이기 위한 새로운 경로를 제시하고 있습니다.



### DBHP: Trajectory Imputation in Multi-Agent Sports Using Derivative-Based Hybrid Prediction (https://arxiv.org/abs/2408.10878)
- **What's New**: 이번 논문은 Derivative-Based Hybrid Prediction (DBHP) 프레임워크를 제안하여 다수 에이전트의 누락된 궤적을 정확하게 입력(imputation)할 수 있도록 합니다. 기존의 접근법에서 나타나는 물리적 제약 조건의 부족이라는 문제를 해결함으로써 더 나은 결과를 얻습니다.

- **Technical Details**: DBHP 프레임워크는 Set Transformers와 양방향 LSTM을 사용하여 누락된 궤적의 나이브 예측값을 생성하고, 속도 및 가속도 정보를 활용하여 대안적 예측인 Derivative-Accumulating Predictions (DAPs)를 생성합니다. 세 가지 예측값을 적절히 조합하여 최종 예측을 제공합니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 팀 스포츠 데이터셋에서 기존의 입력 기법들에 비해 정확도가 크게 향상된 것으로 나타났으며, 물리적 타당성을 개선하는 데 효과적입니다.



### Multi-agent Multi-armed Bandits with Stochastic Sharable Arm Capacities (https://arxiv.org/abs/2408.10865)
Comments:
          28 pages

- **What's New**: 본 논문에서는 다수의 플레이어가 참여하는 다중 무장 도적(Multi-Armed Bandit, MAB) 문제의 새로운 변형을 제안합니다. 이 모델은 각 팔(arm)로 요청이 확률적으로 도착하는 특성과 요청을 플레이어에게 할당하는 정책을 반영합니다.

- **Technical Details**: 우리는 플레이어가 서로 통신하지 않고 최적의 팔 당기기 프로필을 선택할 수 있도록 하는 분산 학습 알고리즘을 설계해야 하는 문제를 다룹니다. 먼저 다항식 계산 복잡도를 가진 탐욕 알고리즘을 설계하였고, 다음으로 상수 횟수의 라운드에서 최적의 팔 당기기 프로필에 동의할 수 있도록 하는 반복적인 분산 알고리즘을 설계했습니다. explore then commit (ETC) 프레임워크를 적용하여 모델 매개변수가 알려지지 않은 온라인 설정을 다룹니다.

- **Performance Highlights**: 실험을 통해 제안한 알고리즘의 성능을 검증하였으며, 플레이어들이 최적의 팔 당기기 프로필에 대한 합의에 도달할 수 있음을 보였습니다. 이 알고리즘은 M 라운드만에 합의에 도달할 수 있는 보장을 제공합니다.



### DELIA: Diversity-Enhanced Learning for Instruction Adaptation in Large Language Models (https://arxiv.org/abs/2408.10841)
Comments:
          8 pages, 5 figures

- **What's New**: 본 논문에서는 LLMs(대규모 언어 모델)의 instruction tuning(지시 조정)에 관한 한계를 다루며, DELIA(다양성 증진 학습 모델)를 통해 바이어스된 특성을 이상적인 특성으로 근사하는 혁신적인 데이터 합성 방법을 제안합니다.

- **Technical Details**: DELIA는 LLM 훈련에서 다양한 데이터의 완충 효과를 활용하여, 특정 작업 형식에 맞게 모델을 조정하기 위한 바이어스된 특성을 이상적인 특성으로 변환합니다. 이 과정에서 우리는 지시-응답 쌍의 다양성을 극대화하여 구조적 변경 없이 훈련 비용을 낮추면서 이상적인 특성에 근접하게 합니다.

- **Performance Highlights**: 실험 결과, DELIA는 일반적인 instruction tuning보다 17.07%-33.41% 향상된 Icelandic-English 번역의 bleurt 점수(WMT-21 데이터셋, gemma-7b-it)와 36.1% 향상된 포맷된 텍스트 생성 정확도를 보여주었습니다. 또한, DELIA는 새로운 특수 토큰의 내부 표현을 이전 의미와 정렬시키는 독특한 능력을 보여, 지식 주입 방법 중에서 큰 진전을 나타냅니다.



### Understanding the Skills Gap between Higher Education and Industry in the UK in Artificial Intelligence Sector (https://arxiv.org/abs/2408.10788)
Comments:
          Accepted to the journal "Industry and Higher Education"

- **What's New**: 본 논문은 영국의 대학들이 AI 관련 과정을 통해 학생들을 실제 직업에 어떻게 준비시키는지를 조사합니다.

- **Technical Details**: 연구진은 커스텀 데이터 스크래핑 도구를 사용해 구직 광고와 대학 교육과정의 내용을 분석했습니다. Frequency 및 Naive Bayes classifier 분석 기법을 통해 산업에서 요구하는 기술 스킬을 세부적으로 파악했습니다.

- **Performance Highlights**: 연구 결과, AI 분야의 대학 커리큘럼은 프로그래밍(Programming)과 머신러닝(Machine Learning) 과목에서 대부분의 기술 스킬이 잘 균형을 이루고 있으나, 데이터 과학(Data Science) 및 수학(Math)과 통계(Statistics) 분야에서는 격차가 발견되었습니다.



### Flexora: Flexible Low Rank Adaptation for Large Language Models (https://arxiv.org/abs/2408.10774)
Comments:
          29 pages, 13 figures

- **What's New**: 이 논문에서는 Low-Rank Adaptation (LoRA) 기술의 한계를 극복하기 위해 'Flexora'라는 새로운 방법을 제안합니다. Flexora는 다양한 다운스트림 작업에서 최적의 성능을 달성하기 위해 fine-tuning을 위한 중요한 레이어를 자동으로 선택하는 유연한 접근 방식을 제공합니다.

- **Technical Details**: Flexora는 레이어 선택 문제를 하이퍼파라미터 최적화 (Hyperparameter Optimization, HPO) 문제로 정의하고, unrolled differentiation (UD) 방법을 사용하여 이를 해결합니다. Flexora는 초기화 단계에서 정의된 하이퍼파라미터를 LoRA 파라미터에 주입하고, 선택된 레이어에 대해서만 백프롭agation과 파라미터 업데이트를 제한하여 계산 오버헤드를 대폭 줄입니다.

- **Performance Highlights**: 다양한 사전 훈련 모델과 자연어 처리 작업에서 수행된 실험 결과에 따르면, Flexora는 기존 LoRA 변형보다 일관되게 성능을 개선시킨 것으로 나타났습니다. 이 연구는 Flexora의 효과성을 뒷받침하기 위해 풍부한 이론적 결과와 여러 ablation 연구를 포함하여 comprehensive understanding을 제공합니다.



### Fine-Tuning a Local LLaMA-3 Large Language Model for Automated Privacy-Preserving Physician Letter Generation in Radiation Oncology (https://arxiv.org/abs/2408.10715)
- **What's New**: 이 연구는 방사선 종양학( Radiation Oncology) 분야에서 의사 편지 생성을 위한 대형 언어 모델( Large Language Model)인 LLaMA 모델의 로컬 파인튜닝을 조사합니다. 기존 LLaMA 모델은 효과적으로 의사 편지를 생성하는 데 부족하며, QLoRA 알고리즘을 통해 제한된 컴퓨팅 자원에서 로컬 파인튜닝이 가능합니다.

- **Technical Details**: 연구에서는 2010년부터 2023년까지의 방사선 종양학 관련 14,479개의 의사 편지를 사용하여 LLaMA-3 모델을 파인튜닝하였으며, 모델은 독일어로 입력된 데이터를 처리할 수 있었습니다. 연구는 방사선 종양학 전용 정보와 기관별 스타일로 의사 편지를 생성할 수 있다는 점을 강조합니다.

- **Performance Highlights**: 평가 결과, LLaMA-3 모델이 LLaMA-2 모델보다 ROUGE 점수에서 우수하며, 10건의 사례에 대한 평가에서 임상의사들은 자동 생성된 편지의 임상적 이점을 높게 평가하였습니다(평균 4점 만점에 3.44점). 면밀한 의사의 검토와 수정을 통해 LLM 기반 의사 편지 생성의 실제 가치가 크다는 결론입니다.



### Investigating Context Effects in Similarity Judgements in Large Language Models (https://arxiv.org/abs/2408.10711)
Comments:
          Accepted at The First Workshop on AI Behavioral Science (AIBS 2024), held in conjunction with KDD 2024

- **What's New**: 이번 연구는 대형 언어 모델(LLM)에서 발생하는 순서 편향(order bias)을 통해 인류 판단과의 정렬을 조사하는 과정입니다. 기존의 심리학 연구에서 나타난 유사성 판단의 순서 효과(order effects)를 기반으로 다양한 LLM에서 이 행위를 재현해보고, 이러한 발견이 LLM 응용 프로그램 디자인에 미치는 의미를 논의합니다.

- **Technical Details**: 연구자는 Tversky와 Gati(1978)의 연구를 재현하여 대형 언어 모델이 인간과 유사한 비대칭 판단을 보이는지를 평가했습니다. 다양한 LLM(모델) 8가지를 연구하여 각각의 매개변수와 온도를 조절하여 결과를 비교했습니다. 실험에서는 두 개의 국가 쌍의 유사성을 평가하는 데 있어 국가의 순서가 판단에 미치는 영향을 분석했습니다. 이 과정에서 LLM들은 비슷한 맥락에서 유사성 판단을 내릴 때 순서에 따라 다양한 평가 점수를 도출하는 것을 확인했습니다.

- **Performance Highlights**: LLM은 분야를 막론하고 인간과 유사한 감정과 인지적 편향을 보이는 경우가 있었으나, 모든 경우가 통계적으로 유의미한 것은 아닙니다. 연구의 결과로, 다양한 온도와 텍스트 변화에 따라 유사성 판단의 순서 효과가 통계적으로 유의미한 모델-온도 쌍이 발견되며, 이는 LLM의 설계 및 실제 활용을 위한 중요한 통찰을 제공합니다.



### Fine-Tuning and Deploying Large Language Models Over Edges: Issues and Approaches (https://arxiv.org/abs/2408.10691)
- **What's New**: 2023년 새로운 LLM(large language model) 기술이 소개되었습니다. 이 기술은 기존의 전문화된 언어 모델에서 범용적인 기초 모델로의 전환을 보여줍니다. 특히, 이러한 모델은 zero-shot 능력이 뛰어나지만, 모든 작업에 대해 로컬 데이터셋에서 미세 조정이 요구됩니다.

- **Technical Details**: 이 논문에서는 메모리 효율적인 미세 조정 기법(P EFT)과 메모리 효율적 전체 미세 조정(M EF2T)을 소개합니다. PEFT는 학습 가능한 가중치를 줄이고, MEF2T는 역전파(BP) 없이 작동하는 옵티마이저를 개발하는 방법입니다. 각 방법은 분산 학습 네트워크(DL NoE)에서 효율적인 적용을 목표로 하고 있습니다.

- **Performance Highlights**: 모델 압축 기법을 사용함으로써 LLM의 에너지 소비 및 운영 비용을 줄일 수 있으며, 이는 지속 가능한 AI 발전을 위한 중요한 단계로 부각됩니다. 메모리 효율적인 미세 조정 기법을 통해 운영 효율성을 극대화하며, 결과적으로 환경에 미치는 영향을 최소화할 수 있습니다.



### Genesis: Towards the Automation of Systems Biology Research (https://arxiv.org/abs/2408.10689)
- **What's New**: AI를 활용하여 과학 연구를 자동화하는 로봇 과학자 ‘Genesis’의 개발이 중점적으로 다루어지고 있습니다. Genesis는 시스템 생물학 모델을 자동으로 향상시키며, 하루에 천 개의 가설 기반 실험 사이클을 수행할 수 있도록 설계되었습니다.

- **Technical Details**: Genesis는 천 개의 컴퓨터 제어된 μ-bioreactors를 사용하여 고급 자동화를 수행합니다. AutonoMS라는 시스템을 통해 고속 실험의 자동 실행 및 분석이 가능합니다. 또한 Genesis-DB 데이터베이스 시스템을 개발하여 소프트웨어 에이전트가 대량의 구조화된 도메인 정보에 접근할 수 있도록 지원하고 있습니다.

- **Performance Highlights**: Genesis는 기존의 시스템보다 약 100배 저렴하게 과학 연구를 수행할 수 있도록 설계되어, 실험의 효율성과 비용 효과를 획기적으로 개선할 수 있는 잠재력을 가지고 있습니다.



### Rejection in Abstract Argumentation: Harder Than Acceptance? (https://arxiv.org/abs/2408.10683)
Comments:
          accepted version as ECAI24

- **What's New**: 이 논문은 기존의 추상 논증 프레임워크에 '거부 조건(rejection conditions, RCs)'이라는 유연한 개념을 추가하여 논증을 보다 세밀하게 평가하고자 합니다. 이를 통해 논증의 복잡성을 분석하며, 특히 거부가 수용보다 더 어려운지 연구합니다.

- **Technical Details**: 논문에서는 각 논증을 특정 논리 프로그램과 연관시키고, 이로 인해 발생하는 복잡성을 분석합니다. 또한, 거부 조건이 존재할 때 이 조건을 무효화해야 한다는 점을 강조하며, 이는 널리 사용되는 Answer Set Programming (ASP)을 기반으로 합니다.

- **Performance Highlights**: 결과적으로, 이 연구는 거부 조건이 수용 조건보다 복잡성을 한 단계 높이며, 트리너스(treewidth)와 같은 구조적 매개변수를 포함한 복잡성 이론적 개념을 설정하는 데 기여합니다. 이를 통해 거부와 수용에 대한 보다 일반적인 이해를 제공합니다.



### Minor SFT loss for LLM fine-tune to increase performance and reduce model deviation (https://arxiv.org/abs/2408.10642)
Comments:
          8 pages, 5 figures

- **What's New**: 본 논문은 SFT(Supervised Fine Tuning)와 RLHF(Reinforcement Learning from Human Feedback)에 대한 새로운 접근 방식을 제안합니다. DPO(Dynamic Preference Optimization)와 MinorDPO에 대한 통찰을 통해, SFT 단계에서 최적화된 모델과 원래 모델 간의 불일치(discrepancy)를 측정하기 위한 훈련 메트릭스를 도입하고, 훈련 효율성을 증가시키고 불일치를 줄일 수 있는 손실 함수 MinorSFT를 제안합니다.

- **Technical Details**: DPO와 MinorDPO는 LLM을 인간의 선호(preference)에 맞추기 위해 설계된 알고리즘입니다. DPO는 각 샘플에 대한 동적 샘플 수준의 계수를 사용해 선호 쌍(preference pair) 간의 거리와 관련된 학습 내용을 조정합니다. MinorSFT는 원래 SFT보다 더 많은 제약을 두어 학습 강도를 조절하여 LLM과 원래 모델 간의 불일치를 줄입니다. 이 손실 함수는 추가적인 하이퍼파라미터와 더 많은 계산 비용을 필요로 하지만, 훈련 효과성을 그리고 성능 결과를 개선할 가능성이 있습니다.

- **Performance Highlights**: MinorSFT를 통해 훈련된 모델은 원래 모델과의 불일치가 감소하고, 결과적으로 더 나은 성능을 보일 것으로 기대됩니다. 기존의 SFT 방식보다 강화된 훈련 메트릭스와 손실 함수를 통해, LLM의 일반성(generality)과 다양성(diversity)을 유지하면서도, 훈련된 모델의 품질을 향상시킬 수 있는 잠재력을 지니고 있습니다.



### Strategist: Learning Strategic Skills by LLMs via Bi-Level Tree Search (https://arxiv.org/abs/2408.10635)
Comments:
          website: this https URL

- **What's New**: 이 논문에서는 Multi-agent 게임을 위한 새로운 방법인 Strategist를 제안합니다. 이 방법은 LLMs를 활용하여 자기 개선 과정을 통해 새로운 기술을 습득하는 방식입니다.

- **Technical Details**: 우리는 Monte Carlo tree search와 LLM 기반 반사를 통해 고급 전략 기술을 학습하는 Self-play 시뮬레이션을 사용합니다. 이 과정은 고차원 전략 수업을 통해 정책을 배우고, 낮은 수준의 실행을 안내하는 상태 평가 방법을 포함합니다.

- **Performance Highlights**: GOPS와 The Resistance: Avalon 게임에서 우리의 방법이 전통적인 강화 학습 기반 접근 방식 및 기존 LLM 기반 기술 학습 방법보다 더 나은 성능을 보였음을 보여주었습니다.



### Hologram Reasoning for Solving Algebra Problems with Geometry Diagrams (https://arxiv.org/abs/2408.10592)
- **What's New**: 이번 논문에서는 대수 문제와 기하 도형을 결합한 Algebra Problems with Geometry Diagrams (APGDs)을 해결하기 위한 새로운 접근 방식인 HGR(Hologram-based Reasoning)를 제안합니다. HGR은 문제 텍스트와 도형 정보를 하나의 홀로그램으로 통합하여 해결 과정을 간소화하고, 해석 가능성을 더욱 향상시키는 방법입니다.

- **Technical Details**: HGR 방법은 문제의 텍스트와 도형을 단일 글로벌 홀로그램으로 변환하여 기하학적 원소(점, 선, 각 등)로 구성된 정점과 이들 간의 관계를 나타내는 간선을 포함합니다. 깊이 강화 학습(Deep Reinforcement Learning)을 활용하여 그래프 모델의 선택 과정을 최적화하고, 최종적으로 새로운 정점 및 간선을 추가하거나 속성을 업데이트하여 대수 방정식을 생성하는 과정을 반복합니다.

- **Performance Highlights**: HGR은 해결 정확성을 보장하면서도 적은 추론 단계를 통해 문제를 해결합니다. 또한 모든 추론 단계의 설명을 제공함으로써 해결 과정의 해석 가능성을 크게 향상시킵니다. 실험 결과, HGR은 APGD 해결 시 정확성과 해석 가능성을 모두 개선하는 데 효과적임을 입증하였습니다.



### Hokoff: Real Game Dataset from Honor of Kings and its Offline Reinforcement Learning Benchmarks (https://arxiv.org/abs/2408.10556)
- **What's New**: Hokoff라는 새로운 오프라인 강화 학습과 오프라인 다중 에이전트 강화 학습을 위한 데이터셋 세트를 공개했습니다. 이 데이터셋은 실제 경험을 반영하는 높은 품질의 데이터를 포함하고 있어 연구의 기초가 될 수 있습니다.

- **Technical Details**: Hokoff는 상징적인 Multiplayer Online Battle Arena (MOBA) 게임인 Honor of Kings에서 수집된 데이터셋으로, 다양한 오프라인 RL 및 MARL 알고리즘을 벤치마킹할 수 있는 프레임워크를 제공합니다. 이 프레임워크는 샘플링, 훈련 및 평가와 같은 전 과정을 포함합니다.

- **Performance Highlights**: 이 연구는 현재의 오프라인 RL 및 MARL 접근 방식이 복잡한 작업을 효과적으로 처리하지 못하고 있음을 보여줍니다. 특히, 이러한 접근 방식은 일반화 능력과 다중 작업 학습에서 부족함을 보입니다.



### AI-Based IVR (https://arxiv.org/abs/2408.10549)
Comments:
          in Russian language

- **What's New**: 이 논문에서는 전통적인 IVR 시스템의 한계를 극복하기 위해 AI 기술을 적용하는 방법을 제안합니다. 특히 Kazakh 언어에 최적화된 시스템을 개발하여 효율성을 높이고 고객 서비스 품질을 향상시키는 것을 목표로 합니다.

- **Technical Details**: 이 시스템은 음성 인식(Automatic Speech Recognition, ASR), 텍스트 쿼리 분류를 위한 대형 언어 모델(Large Language Models, LLM), 및 음성 합성(Text-to-Speech) 기술을 통합하여 구성됩니다. Whisper 모델과 IrbisGPT 모델을 활용하고, Kazakh 데이터셋을 바탕으로 fine-tuning을 실시합니다. Low-Rank Adaptation(LoRA) 기법을 사용하여 모델을 최적화합니다.

- **Performance Highlights**: AI 기술이 적용된 IVR 시스템은 운영자의 업무량을 줄이고, 고객 서비스 품질을 향상시키며, 쿼리 처리의 효율성을 높이는 것으로 나타났습니다. 다양한 언어에 적용 가능성을 지닌 이 접근법은 특히 고유한 언어적 특성을 가진 설정에서도 효과적일 것으로 기대됩니다.



### Approximate Estimation of High-dimension Execution Skill for Dynamic Agents in Continuous Domains (https://arxiv.org/abs/2408.10512)
- **What's New**: 이 논문은 AI가 인간의 지속적 행동 영역에서 수행 오류(실행 Skill)를 보다 정확하게 추정할 수 있는 새로운 파티클 필터 기반 추정기를 제안합니다. 기존 연구들은 대칭 정규 분포를 가정하고 실행 오류가 모든 관측치에 대해 일정하다고 보았던 반면, 본 연구에서는 이러한 가정을 넘어 보다 유연하고 복잡한 모델링을 제시합니다.

- **Technical Details**: 본 연구에서 제안하는 파티클 필터 기반 추정기는 실행 Skill을 고차원으로 추정할 수 있으며, 시간에 따라 변하는 Skill을 추정할 수 있는 프레임워크를 제공합니다. 이를 통해 에이전트가 다양한 행동을 수행할 때의 오류 분포를 보다 현실적으로 모델링할 수 있습니다.

- **Performance Highlights**: MLB의 투구 데이터를 활용한 실험에서 이 추정기를 적용하여 기존 방법들에 비해 더 시간 가변적인 실행 Skill 추정을 가능하게 하여 에이전트의 의사 결정 개선 및 전반적인 성능 향상을 보여줍니다.



### QPO: Query-dependent Prompt Optimization via Multi-Loop Offline Reinforcement Learning (https://arxiv.org/abs/2408.10504)
- **What's New**: 이번 논문에서는 Query-dependent Prompt Optimization (QPO)이라는 새로운 방법을 소개하며, 이는 기존의 프롬프트 최적화 방법들이 task-level 성능에만 집중하는 문제를 해결하고자 합니다.

- **Technical Details**: QPO는 다중 루프 offline reinforcement learning을 활용하여, 입력 쿼리에 맞춤화된 최적의 프롬프트를 생성하는 소규모 사전 훈련된 언어 모델을 반복적으로 미세 조정합니다. 이 과정에서 기존의 다양한 프롬프트에 대한 benchmarking 데이터를 활용하여 온라인 상호작용의 비용을 줄입니다.

- **Performance Highlights**: 다양한 LLM 스케일 및 NLP, 수학 과제를 대상으로 한 실험에서는 우리의 방법이 zero-shot 및 few-shot 상황 모두에서 효율성과 비용 효율성을 입증하였습니다.



### Is the Lecture Engaging for Learning? Lecture Voice Sentiment Analysis for Knowledge Graph-Supported Intelligent Lecturing Assistant (ILA) System (https://arxiv.org/abs/2408.10492)
- **What's New**: 본 논문은 지식 그래프(knowledge graph)를 활용하여 강의 내용을 나타내고 최적의 교수 전략을 지원하는 지능형 강의 보조 시스템(ILA)을 소개합니다. 이 시스템은 강사들이 학생의 학습을 향상시키는 데 도움을 줄 수 있도록 음성 및 콘텐츠 분석을 실시간으로 수행합니다.

- **Technical Details**: ILA 시스템은 강의 내용과 효과적인 교수 전략을 포괄하는 지식 그래프의 구조를 기본으로 하고 있으며, 강사와 학생으로부터 입력된 음성, 콘텐츠, 반응을 지속적으로 분석합니다. 이 시스템은 회수 연습(retrieval practice), 간격 연습(spaced practice), 주제 혼합(interleaving), 피드백 기반 메타인지(feedback-driven metacognition)와 같은 증거 기반 학습 전략을 이행하도록 지원합니다.

- **Performance Highlights**: 본 논문에서 수행된 초기 연구 결과, 3000개 이상의 1분 강의 음성 클립을 기반으로 지루한 강의를 분류하는 여러 모델을 평가한 결과, 800개 이상의 독립 테스트 클립에서 90%의 F1 점수를 달성했습니다. 이 연구 성과는 향후 콘텐츠 분석과 교수 방법 통합을 위한 더 정교한 모델의 개발 기반이 됩니다.



### IDEA: Enhancing the rule learning ability of language agent through Induction, DEuction, and Abduction (https://arxiv.org/abs/2408.10455)
Comments:
          9pages, 12 figs, 4 tables

- **What's New**: 본 연구에서는 LLM(대형 언어 모델)의 규칙 학습 능력을 평가하기 위해 RULEARN이라는 새로운 벤치마크를 소개합니다. 이 벤치마크는 상호작용 환경에서 문제를 해결하는 LLM의 능력을 살펴보는 데 중점을 두고 있습니다.

- **Technical Details**: RULEARN은 세 가지 유형의 환경(함수 연산자, 탈출 방, 반응기)으로 구성되어 있으며, 에이전트는 미지의 규칙을 학습하기 위해 환경과 상호작용합니다. IDEA 에이전트는 귀납법(induction), 연역법(deduction), 유도법(abduction) 프로세스를 통합하여 규칙 학습을 수행합니다.

- **Performance Highlights**: IDEA 에이전트는 기존 LLM에 비해 RULEARN 벤치마크에서 규칙 학습 성능이 유의미하게 향상된 것으로 나타났으나, 여전히 에이전트는 새로운 환경을 탐색하는 데 어려움을 겪고 있으며 신뢰할 수 있는 가설을 생성하는 데 한계를 보이고 있습니다.



### Feasibility of assessing cognitive impairment via distributed camera network and privacy-preserving edge computing (https://arxiv.org/abs/2408.10442)
- **What's New**: 본 연구는 경증 인지장애(Mild Cognitive Impairment, MCI) 환자 간의 사회적 상호작용과 이동 패턴을 자동으로 캡처하기 위한 프라이버시 보호 분산 카메라 네트워크를 활용하여 장기 모니터링을 향상시키는 데 목표를 두었습니다. 이는 MCI 환자의 다양한 인지 기능 수준을 효과적으로 구별할 수 있는 새로운 방법론을 제시합니다.

- **Technical Details**: 연구는 1700$m^2$ 공간에서 경증 인지장애 환자들로부터 수집된 이동 및 사회적 상호작용 데이터를 사용하여, 기계 학습 알고리즘을 통해 고기능(Group with High Cognitive Function)과 저기능(Group with Low Cognitive Function)인 집단을 구별하는 특징(feature)을 개발했습니다. 39대의 엣지 컴퓨팅 및 카메라 장치가 사용되어 실시간으로 체형 추정(pose estimation) 모델을 운영하였으며, 평균 1.41m의 로컬리제이션 오차를 달성했습니다.

- **Performance Highlights**: 연구 결과, 선형 경로 길이(linear path length), 보행 속도(walking speed), 방향 변화의 변화, 속도 및 방향의 엔트로피(entropy) 등 여러 주요 특징에서 고기능 집단과 저기능 집단 간에 통계적으로 유의미한 차이를 발견하였습니다. 이 연구의 기계 학습 접근 방식은 71%의 정확도로 인지 기능 수준을 구별하는데 성공하였습니다.



### Development of an AI Anti-Bullying System Using Large Language Model Key Topic Detection (https://arxiv.org/abs/2408.10417)
- **What's New**: 이 논문은 인공지능(AI) 기반의 사이버 괴롭힘 방지 시스템 개발과 평가에 대한 내용을 다룹니다. 이 시스템은 소셜 미디어에서의 조직적 괴롭힘 공격을 식별하고 분석하여 대응 방안을 제시합니다.

- **Technical Details**: 이 시스템은 대규모 언어 모델(LLM)을 활용하여 괴롭힘 공격에 대한 전문가 시스템 기반의 네트워크 모델을 구성합니다. 이를 통해 괴롭힘 공격의 특성을 분석하고, 소셜 미디어 회사에 보고 메시지를 생성하는 등의 대응 활동을 지원합니다.

- **Performance Highlights**: LLM의 모델 구성에 대한 효능이 분석되었으며, 이 시스템이 괴롭힘 방지 및 대응에 효과적임을 보여줍니다.



### AI-Driven Review Systems: Evaluating LLMs in Scalable and Bias-Aware Academic Reviews (https://arxiv.org/abs/2408.10365)
Comments:
          42 pages

- **What's New**: 이 논문은 자동 리뷰 시스템을 통해 과학 연구 기고의 품질과 효율성을 향상시키기 위한 새로운 접근 방식을 소개합니다. 특히, 사람의 리뷰와 LLM의 리뷰 간의 일치를 평가하고 LLM을 통해 자동으로 리뷰를 생성하며, 기존 리뷰 프로세스의 한계를 극복하는 방법을 제안합니다.

- **Technical Details**: 이 연구에서는 LLM(대형 언어 모델)을 사용하여 논문 리뷰 프로세스를 전환하고, human preferences(인간 선호도)를 예측하기 위해 LLM을 미세 조정합니다. 오류를 인위적으로 도입하여 LLM의 한계를 분석하고, 여러 문서를 활용하여 LLM의 품질을 향상시킵니다.

- **Performance Highlights**: OpenReviewer, Papers with Reviews, 그리고 Reviewer Arena라는 세 가지 AI 리뷰 시스템을 개발하였으며, 인간 평가와 LLM 자동 평가를 통해 리뷰 품질을 극대화하는 방법을 제공하고 있습니다. 이 시스템들은 연구자들에게 즉각적이고 일관되며 고품질의 피드백을 제공하여, 더 나은 논문 작성을 도와줍니다.



### Query languages for neural networks (https://arxiv.org/abs/2408.10362)
Comments:
          To appear at ICDT 2025

- **What's New**: 본 논문에서는 신경망 모델을 해석하고 이해하기 위한 데이터베이스 기반 접근 방식의 기초를 다집니다. 이를 위해 선언적 언어(declarative languages)를 사용하여 신경망을 질의하는 방법을 연구하였습니다.

- **Technical Details**: 첫 번째 논리(first-order logic)를 기반으로 한 다양한 질의 언어를 검토하며, 이것은 주로 신경망 모델에 대한 접근 방식에 따라 다릅니다. 실수에 대한 첫 번째 논리는 네트워크를 블랙 박스(black box)로 보고, 네트워크가 정의하는 입력-출력 함수만 질의할 수 있는 언어를 자연스럽게 생성합니다. 반면, 화이트 박스(white-box) 언어는 신경망을 가중치 그래프(weighted graph)로 보고, 가중치 항에 대한 합(sum)을 통해 첫 번째 논리를 확장하여 얻을 수 있습니다.

- **Performance Highlights**: 이 두 접근 방식은 표현력을 비교할 수 없지만, 주어진 자연적 상황에서는 화이트 박스 접근 방식이 블랙 박스 접근 방식을 포함할 수 있음을 보여줍니다. 특히, 피드포워드 신경망(feedforward neural networks)에서 고정된 수의 은닉층(hidden layers)과 조각선형 활성화 함수(piecewise linear activation functions)로 정의할 수 있는 실수 함수에 대한 선형 제약 질의(linear constraint queries)에 대해 이 결과를 구체적으로 증명하였습니다.



### LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in the Legal Domain (https://arxiv.org/abs/2408.10343)
- **What's New**: LegalBench-RAG는 법률 및 AI 기술을 융합한 Retrieval-Augmented Generation (RAG) 시스템에서 검색 단계의 효과를 평가하기 위해 최초로 설계된 벤치마크이다. 이는 기존의 LegalBench와는 달리 RAG의 검색 품질을 집중적으로 평가할 수 있는 도구를 제공한다.

- **Technical Details**: LegalBench-RAG 벤치마크는 법률 문서에서 최소한의, 매우 관련성이 높은 텍스트 조각을 추출하여 정확한 검색을 중시한다. 68,58개의 질의-응답 쌍으로 구성된 데이터셋은 7,900만자 이상의 법률 컬렉션에서 인간 전문가에 의해 주석이 달린 데이터로, LegalBench의 맥락을 거슬러 올라가며 제작되었다. 또한, LegalBench-RAG-mini라는 경량화 버전이 속도와 실험을 위한 빠른 반복을 위해 제안되었다.

- **Performance Highlights**: LegalBench-RAG는 RAG 시스템의 정확성과 성능 향상을 위해 기업 및 연구자들에게 중요한 도구로 자리 잡을 것으로 기대된다. 특히, 보다 정밀한 결과는 LLM이 최종 사용자에게 인용을 생성하는 데 도움을 줄 수 있다.



### A Disguised Wolf Is More Harmful Than a Toothless Tiger: Adaptive Malicious Code Injection Backdoor Attack Leveraging User Behavior as Triggers (https://arxiv.org/abs/2408.10334)
- **What's New**: 최근 대규모 언어 모델(LLMs)의 코드 생성 분야에서의 발전과 함께, 소프트웨어 개발에 이 모델을 의존하는 사용자들이 증가함에 따라 코드 생성 모델과 관련된 보안 위험이 커지고 있다. 본 논문에서는 코드 생성 시나리오에 초점을 맞춘 게임 이론 모델을 최초로 제시하며, 공격자가 악성 코드 모델을 퍼뜨릴 수 있는 시나리오 및 패턴을 개요한다.

- **Technical Details**: 우리는 공격자의 관점에서 게임 이론적 프레임워크를 설계하였으며, 이는 악의적인 공격자가 코드 모델을 활용해 보안을 위협할 수 있는 방법을 설명한다. 이 모델은 사용자의 프로그래밍 능력 수준에 따라 악성 코드 주입 시점을 동적으로 조정할 수 있는 역량을 가지고 있다. 또한, 불명확한 의미의 트리거를 활용한 백도어 공격의 가능성을 탐구하였다.

- **Performance Highlights**: 소규모 실험을 통해 제안된 공격 모델의 효과를 입증하였으며, 생성된 악성 코드는 프로그래밍 기술 수준에 따라 다양한 보안 위협을 제시하는 것으로 나타났다. 그 결과, 고위험 취약점이 포함된 코드를 생성하는 방식이 사용자에게 보다 실질적인 위협을 줄 수 있음을 확인하였다.



### NeCo: Improving DINOv2's spatial representations in 19 GPU hours with Patch Neighbor Consistency (https://arxiv.org/abs/2408.11054)
Comments:
          Preprint. The webpage is accessible at: this https URL

- **What's New**: 이번 논문에서는 NeCo(Patch Neighbor Consistency)라는 새로운 자기 지도 학습 신호를 제안하여 사전 훈련된 표현을 개선하고자 한다. 해당 방법은 학생 모델과 교사 모델 간의 패치 레벨 이웃 일관성을 강화하는 새로운 훈련 손실을 도입한다.

- **Technical Details**: 자기 지도 학습의 일환으로, 해당 방법은 DINOv2와 같은 사전 훈련된 표현 위에 차별적 정렬(differentiable sorting) 방법을 적용하여 학습 신호를 부트스트랩(bootstrap)하고 개선한다. NeCo는 이미지 수준에서 이미 훈련된 모델을 시작으로 하여 이를 적응시키는 방식으로 빠르고 효율적인 해결책을 제공한다. 이 과정에서 패치 간 유사성을 유지해야 하며, 가장 가까운 이웃을 정렬하여 학습의 일관성을 확보한다.

- **Performance Highlights**: 제안된 방법은 다양한 모델과 데이터세트에서 뛰어난 성능을 보였으며, ADE20k 및 Pascal VOC에서 비모수적(non-parametric) 맥락 세분화(in-context segmentation)에서 각각 +5.5% 및 +6%의 성능 향상을 보였고, COCO-Things 및 COCO-Stuff에서 선형 세분화 평가에서 각각 +7.2% 및 +5.7%를 기록하였다.



### Revisiting VerilogEval: Newer LLMs, In-Context Learning, and Specification-to-RTL Tasks (https://arxiv.org/abs/2408.11053)
Comments:
          This paper revisits and improves the benchmark first presented in arXiv:2309.07544. Seven pages, three figures

- **What's New**: 이번 논문에서는 VerilogEval 벤치마크의 한계를 보완하고 향상시켜, 대형 언어 모델(LLMs)이 하드웨어 코드 생성에 사용될 수 있는 가능성을 탐구합니다. 새로운 상업용 및 오픈 소스 모델의 평가를 통해, 명세(specification)에서 RTL(Registration Transfer Level)로의 변환 작업을 지원합니다.

- **Technical Details**: VerilogEval 벤치마크는 코드 완성 과제뿐만 아니라 명세에서 RTL로의 변환을 지원하도록 확장되었습니다. 이를 통해 모델의 성능을 더 효과적으로 평가할 수 있으며, 인-context learning (ICL) 프롬프트를 추가하여 모델의 반응을 개선할 수 있습니다. 실패 분류 시스템을 도입하여, 코드 생성의 실패 원인을 자동으로 분석합니다.

- **Performance Highlights**: GPT-4 Turbo는 명세에서 RTL로의 변환 작업에서 59%의 합격률을 기록했고, Llama 3.1 405B는 58%의 합격률로 경쟁력을 보였습니다. 도메인 특화 모델인 RTL-Coder는 37%의 인상적인 합격률을 달성했습니다. 효율적인 프롬프트 엔지니어링(prompt engineering)이 합격률에 중요한 영향을 미친다는 것도 확인했습니다.



### Accelerating Goal-Conditioned RL Algorithms and Research (https://arxiv.org/abs/2408.11052)
- **What's New**: 이 논문은 자기 감독 목표 조건 강화 학습(self-supervised goal-conditioned reinforcement learning, GCRL) 에이전트를 위한 고성능 코드베이스와 벤치마크 JaxGCRL을 발표합니다. 이를 통해 연구자들은 단일 GPU에서 몇 분 만에 수백만 개의 환경 단계를 교육할 수 있습니다.

- **Technical Details**: 이 연구는 GPU 가속화된 환경과 대조 강화 학습(Contrastive Reinforcement Learning) 알고리즘의 안정적이고 배치된 버전을 결합합니다. 이러한 접근 방식은 infoNCE 목표를 기반으로 하여 데이터 처리량을 극대화합니다. GPU 가속화된 시뮬레이터는 단일 24GB GPU에서 초당 거의 200,000 스텝을 수집할 수 있습니다.

- **Performance Highlights**: 단일 GPU에서 10백만 스텝 환경 실험이 약 10분 밖에 걸리지 않으며, 이는 기존 CRL 코드베이스보다 10배 더 빠릅니다. 이 새로운 벤치마크 환경은 기존 자기 감독 RL 알고리즘의 능력이 이전에 생각했던 것보다 뛰어나다는 것을 보여줍니다.



### FLAME: Learning to Navigate with Multimodal LLM in Urban Environments (https://arxiv.org/abs/2408.11051)
Comments:
          10 pages, 5 figures

- **What's New**: FLAME(FLAMingo-Architected Embodied Agent)를 소개하며, MLLM 기반의 새로운 에이전트이자 아키텍처로, 도시 환경 내 VLN(Visual-and-Language Navigation) 작업을 효율적으로 처리할 수 있도록 설계되었습니다.

- **Technical Details**: FLAME은 세 단계의 조정 기법을 통해 내비게이션 작업에 효과적으로 적응합니다: 1) 단일 인식 조정: 거리 뷰 설명 학습, 2) 다중 인식 조정: 에이전트 궤적 요약 학습, 3) VLN 데이터 세트에 대한 전향적 훈련 및 평가. 이 과정에서 GPT-4를 활용하여 Touchdown 환경을 위한 캡션 및 경로 요약을 합성합니다.

- **Performance Highlights**: FLAME은 기존 방법들보다 탁월한 성능을 보여주며, Touchdown 데이터 세트에서는 7.3%의 작업 완료율 개선을 달성하며 SOTA(state-of-the-art) 기록을 세웠습니다. 또한 Map2seq에서는 3.74% 개선을 이루어 도시 VLN 작업의 새로운 기준을 확립했습니다.



### RP1M: A Large-Scale Motion Dataset for Piano Playing with Bi-Manual Dexterous Robot Hands (https://arxiv.org/abs/2408.11048)
Comments:
          Project Website: this https URL

- **What's New**: 로봇 손에 인간 수준의 능숙함(dexterity)을 부여하기 위한 연구의 일환으로, 본 논문에서는 로봇 피아노 연주를 위한 Robot Piano 1 Million (RP1M) 데이터셋을 제안합니다. 이 데이터셋은 100만 개 이상의 주행 경로를 포함하여, 다중 곡(multi-song)에서 피아노 연주를 가능하게 합니다.

- **Technical Details**: RP1M 데이터셋은 고품질 이인용(dual-manual) 로봇 피아노 연주 모션 데이터를 포함하고 있으며, 최적 운송(optimal transport) 문제로 손가락 배치를 공식화함으로써 자동으로 주석(annotation)을 달 수 있도록 설계되었습니다. 이를 통해 로봇은 다양한 음악에 대해 손가락이 어떻게 움직여야 하는지를 스스로 배울 수 있게 됩니다.

- **Performance Highlights**: RP1M을 이용하여 다양한 행동 복제(behavior cloning) 접근 방식을 벤치마킹한 결과, 기존의 모방 학습(imitation learning) 방법들이 주어진 새로운 음악 조각에 대한 모션 합성과 관련하여 새로운 최첨단(state-of-the-art) 성능을 달성했습니다.



### Reconciling Methodological Paradigms: Employing Large Language Models as Novice Qualitative Research Assistants in Talent Management Research (https://arxiv.org/abs/2408.11043)
Comments:
          Accepted to KDD '24 workshop on Talent Management and Computing (TMC 2024). 9 pages

- **What's New**: 이 연구는 인터뷰 전사 분석을 위한 새로운 접근 방식을 제안하고, Retrieval Augmented Generation (RAG) 기반의 Large Language Models (LLMs)를 활용하여 질적 데이터의 분석을 개선하는 방법을 탐구합니다.

- **Technical Details**: 이 연구는 RAG 기반의 LLM을 사용하여 반구조화된 인터뷰 데이터를 주제 모델링하는 방법을 제시하며, 이를 통해 LLM이 전통적인 정보 검색 및 검색 분야를 넘어 다양하게 활용될 수 있음을 보여줍니다. LLM은 비전문가 질적 연구 조수로서의 역할을 하도록 설계되었습니다.

- **Performance Highlights**: LLM-증강 RAG 접근 방식이 수동적으로 생성된 주제에 비해 상당한 범위를 가진 관심 주제를 성공적으로 추출할 수 있음을 입증하였습니다. 연구자들은 이러한 모델을 활용할 때 전통적인 질적 연구에서 사용하는 품질 기준에 중점을 두어 접근 방식의 엄격성과 신뢰성을 보장해야 한다고 권장합니다.



### Athena: Safe Autonomous Agents with Verbal Contrastive Learning (https://arxiv.org/abs/2408.11021)
Comments:
          9 pages, 2 figures, 4 tables

- **What's New**: 대형 언어 모델(LLM)을 이용한 자율 에이전트의 안전성을 극대화하기 위한 Athena 프레임워크를 소개합니다. 이 프레임워크는 언어 대조 학습(Verbal Contrastive Learning) 개념을 활용하여 과거의 안전 및 위험한 경로를 예제로 사용하여 에이전트가 안전성을 확보하며 작업을 수행하도록 도움을 줍니다.

- **Technical Details**: Athena 프레임워크는 Actor, Critic, Emulator의 세 가지 LLM 에이전트로 구성되며, 이들은 사용자로부터 제공된 요구 사항에 따라 상호작용하여 작업을 완료합니다. Critic 에이전트는 Actor의 사고 및 행동을 심사하여 안전성을 개선하고, 언어 대조 학습을 통해 Actor에게 과거 경로를 기반으로 안전한 행동을 유도합니다. 또한, 8080개의 도구 키트와 180개의 시나리오로 구성된 안전 평가 벤치마크를 구축하였습니다.

- **Performance Highlights**: 실험적 평가에서 언어 대조 학습과 상호작용 수준의 비평이 LLM 기반의 에이전트의 안전성을 크게 향상시킨다는 결과를 얻었습니다. 특히, GPT-4-Turbo가 가장 안정적인 후보로 확인되었습니다.



### An Overlooked Role of Context-Sensitive Dendrites (https://arxiv.org/abs/2408.11019)
- **What's New**: 본 연구에서는 피라미드 두 점 뉴런(2-point neurons, TPNs)의 능력을 새롭게 조명하면서, 아피칼 존(apical zone)에서 다양한 입력 신호가 어떻게 처리되고 통합되는지를 설명하였다. 특히, 유니버설(context, U)과 근접한(context, P) 및 먼(context, D) 신호의 상호작용이 뉴런의 학습과 신호 처리에 미치는 영향을 강조하였다.

- **Technical Details**: 연구진은 context sensitive TPNs (CS-TPNs) 모델을 개발하여, 이 모델이 피라미드 뉴런의 기능을 모방하고, 자극(cues)과 정보를 더 효과적으로 통합하고 처리할 수 있는 방식으로 아피칼 및 기저 입력 신호를 구분 및 조정하는 방법을 보여주었다. 이를 통해 시뮬레이션 결과 CS-TPNs은 훈련된 백프로파게이션(backpropagation, BP) 방식에 비해 적은 뉴런으로도 신호를 더 일관되게 전파하는 것이 가능함을 증명하였다.

- **Performance Highlights**: CS-TPNs의 적용으로 학습 속도가 빨라지고, 자원 소모를 최소화할 수 있었다. 전통적인 합성곱 신경망(convolutional neural networks, CNNs)에 비해 훨씬 적은 수의 뉴런으로도 다양한 실제 오디오-비주얼 데이터(A-V data)를 처리할 수 있음을 시연하였다.



### Multiwinner Temporal Voting with Aversion to Chang (https://arxiv.org/abs/2408.11017)
Comments:
          Appears in the 27th European Conference on Artificial Intelligence (ECAI), 2024

- **What's New**: 본 논문에서는 유권자들이 동적인 선호를 가지는 두 단계의 위원회 선거를 다루고 있습니다. 특정 투표 규칙에 따라 위원회가 선택되는 단계에서, 첫 번째 단계에 최대한 겹치는 두 번째 단계의 당선 위원회를 찾는 것이 목표입니다. 특히, Thiele 규칙의 복잡성을 이분법적으로 분석하였으며, Approval Voting (AV)에서는 문제 해결이 가능하나, 나머지 모든 Thiele 규칙에서는 어려운 것으로 나타났습니다.

- **Technical Details**: 이 논문은 Thiele 규칙에 대한 결정 문제의 복잡성을 완전하게 분류합니다. Approval Voting (AV)에서는 문제 해결이 𝖯𝖯\mathsf{P}에 속하지만, AV를 제외한 모든 Thiele 규칙에 대해서는 𝖼𝗈𝖭𝖯𝖼𝗈𝖭𝖯\mathsf{coNP}-hard로 나타납니다. 또한, greedy 변형을 포함한 Thiele 규칙에 대해서도 유사한 이분법을 제시합니다.

- **Performance Highlights**: 실험 분석을 통해 유권자 선호 변화에 따른 위원회의 변화 정도를 측정하였으며, 단순히 사전 순서로 동률을 해결하는 것이 연속성을 유지하는 목적에 비해 최적이 아님을 보여주었습니다.



### Denoising Plane Wave Ultrasound Images Using Diffusion Probabilistic Models (https://arxiv.org/abs/2408.10987)
- **What's New**: 이 논문에서는 초음파 평면웨이브(plane wave) 이미지를 향상시키기 위한 Denoising Diffusion Probabilistic Models (DDPM)을 도입합니다. 기존의 고배율 초음파 이미징에서 발생하는 노이즈 문제를 해결하기 위해 DDPM을 적용한 새로운 방법을 제안합니다. 저각(compounding plane waves)과 고각(compounding plane waves) 초음파의 차이를 노이즈로 간주하고 이를 효과적으로 제거하여 이미지 품질을 개선합니다.

- **Technical Details**: 제안된 방법은 400개의 시뮬레이션 된 이미지를 사용하여 DDPM을 훈련시키며, 자연 이미지 분할 마스크를 강도 맵(intensity maps)으로 사용하여 다양한 해부학적 형상의 정확한 노이즈 제거를 달성합니다. 또한, 저화질 평면웨이브와 고화질 평면웨이브 사이의 차이를 변동 요소로 사용하여, 정규 노이즈가 아닌 저화질 이미지를 초기화하여 필요한 역단계의 수를 줄입니다.

- **Performance Highlights**: 시뮬레이션, 팬텀, 그리고 인비보(in vivo) 이미지를 비롯한 평가를 통해 제안된 방법은 이미지 품질을 개선하며, 다른 방법들과 비교했을 때 여러 평가 지표에서 우수한 성능을 나타냅니다. 연구진은 소스 코드와 훈련된 모델을 공개할 예정입니다.



### Wave-Mask/Mix: Exploring Wavelet-Based Augmentations for Time Series Forecasting (https://arxiv.org/abs/2408.10951)
- **What's New**: 본 연구에서는 디스크리트 웨이렛 변환(Discrete Wavelet Transform, DWT)을 이용하여 시계열 데이터의 빈도 성분을 조정하고 시간 종속성을 유지하는 두 가지 데이터 증강 기법인 웨이브렛 마스킹(Wavelet Masking, WaveMask)과 웨이브렛 믹싱(Wavelet Mixing, WaveMix)을 제안합니다.

- **Technical Details**: WaveMask 기법은 각 분해 단계에서 특정 웨이브렛 계수를 선택적으로 제거하여 증강 데이터의 변동성을 증가시키고, WaveMix는 두 가지 데이터 집합의 웨이브렛 계수를 교환하여 다양성을 향상시킵니다. 두 기법은 기존의 기본 증강 방법들과 비교하여 높은 성능을 발휘합니다.

- **Performance Highlights**: 제안한 기술은 16개의 예측 지평선 작업 중 12개에서 우수한 성과를 보였으며, 남은 4개 작업에서도 두 번째로 좋은 성적을 기록했습니다. 콜드 스타트 예측에서도 지속적으로 뛰어난 결과를 보여주었습니다.



### GAIM: Attacking Graph Neural Networks via Adversarial Influence Maximization (https://arxiv.org/abs/2408.10948)
- **What's New**: 본 연구에서는 Graph Neural Networks (GNNs)에 대한 통합적인 적대적 공격 방법인 GAIM을 제안합니다. GAIM은 노드 특성 기반에서 수행되며, 엄격한 블랙박스 설정을 고려합니다. 적대적 영향을 평가하기 위한 함수가 정의되어, GNN 공격 문제를 적대적 영향 극대화 문제로 재구성합니다.

- **Technical Details**: GAIM 공격은 노드의 적대적 영향을 극대화하는 문제로 정의되며, 최적의 타겟 노드와 특성 변화의 선택을 단일 최적화 문제로 통합합니다. 우리는 대리 모델(surrogate model)을 사용하여 이 문제를 해결 가능한 선형 프로그래밍(linear programming)으로 변환합니다. 이 방법론은 레이블 중심 공격(label-oriented attacks)으로도 확장 가능합니다.

- **Performance Highlights**: 다섯 개의 벤치마크 데이터셋과 세 가지 일반 GNN 모델에 대한 포괄적인 평가 결과, GAIM 방법이 최첨단 기법(SOTA)과 비교했을 때 효과적임을 강조합니다. 다양한 공격 시나리오에서 GAIM의 일반화 가능성과 성능이 일관되게 나타났습니다.



### HiRED: Attention-Guided Token Dropping for Efficient Inference of High-Resolution Vision-Language Models in Resource-Constrained Environments (https://arxiv.org/abs/2408.10945)
Comments:
          Preprint

- **What's New**: 이 논문은 High-Resolution Early Dropping (HiRED)이라는 새로운 토큰 드롭핑 기법을 제안하여 자원 제약이 있는 환경에서도 고해상도 Vision-Language Models (VLMs)을 효율적으로 처리할 수 있는 방법을 제공합니다. HiRED는 추가적인 훈련 없이 기존 VLM에 통합할 수 있는 플러그 앤 플레이(plug-and-play) 방식으로, 높은 정확도를 유지합니다.

- **Technical Details**: HiRED는 초기 레이어의 비전 인코더의 주의(attention) 메커니즘을 활용하여 각 이미지 파티션의 시각적 콘텐츠를 평가하고, 토큰 예산을 할당합니다. 마지막 레이어의 주의 메커니즘을 통해 가장 중요한 비주얼 토큰을 선택하여 할당된 예산 내에서 나머지를 드롭합니다. 이 방법은 연산 효율성을 높이기 위해 이미지 인코딩 단계에서 토큰을 드롭하여 LLM의 입력 시퀀스 길이를 줄이는 방식으로 작동합니다.

- **Performance Highlights**: HiRED를 NVIDIA TESLA P40 GPU에서 LLaVA-Next-7B에 적용한 결과, 20% 토큰 예산으로 생성된 토큰 처리량이 4.7배 증가했으며, 첫 번째 토큰 생성 지연(latency)이 15초 감소하고, 단일 추론에 대해 2.3GB의 GPU 메모리를 절약했습니다. 또한, HiRED는 기존 베이스라인보다 훨씬 높은 정확도를 달성했습니다.



### A Closer Look at Data Augmentation Strategies for Finetuning-Based Low/Few-Shot Object Detection (https://arxiv.org/abs/2408.10940)
- **What's New**: 이 논문은 데이터가 부족한 상황에서 저비용의 물체 탐지에 대한 성능과 에너지 효율성을 평가하는 포괄적인 실증 연구를 수행합니다. 특히, 사용자 정의 데이터 증가(Data Augmentation, DA) 방법과 자동 데이터 증가 선택 전략이 경량 물체 탐지기와 결합했을 때의 효과를 중점적으로 분석합니다.

- **Technical Details**: 여기서 Efficiency Factor를 사용하여 성능과 에너지 소비를 모두 고려한 분석을 통해 데이터 증가 전략의 유효성을 이해하려고 시도합니다. 연구는 세 가지 벤치마크 데이터셋에서 DA 전략이 경량 물체 탐지기의 파인튜닝 과정 중 에너지 효율성 및 일반화 능력에 미치는 영향을 평가합니다.

- **Performance Highlights**: 결과적으로 데이터 증가 전략의 성능 향상이 에너지 사용 증가로 인해 가려지는 경우가 많다고 밝혀집니다. 따라서 데이터가 부족한 상황에서, 더 에너지 효율적인 데이터 증가 전략 개발이 필요하다는 결론에 도달합니다.



### SDI-Net: Toward Sufficient Dual-View Interaction for Low-light Stereo Image Enhancemen (https://arxiv.org/abs/2408.10934)
- **What's New**: 본 논문은 저조도 스테레오 이미지 향상을 위한 새로운 모델 SDI-Net을 제안합니다. SDI-Net은 왼쪽 및 오른쪽 보기 간의 상호작용을 극대화하여 더 나은 이미지 향상 결과를 도출하는 데 초점을 맞추고 있습니다.

- **Technical Details**: SDI-Net의 구조는 두 개의 UNet으로 구성된 인코더-디코더 쌍으로, 저조도 이미지를 노멀 라이트 이미지로 변환하는 매핑 기능을 학습합니다. 핵심 모듈인 Cross-View Sufficient Interaction Module (CSIM)은 주의 메커니즘을 통해 양안 시점 간의 상관관계를 최대한 활용합니다. CSIM은 두 개의 주요 구성 요소인 Cross-View Attention Interaction Module (CAIM) 및 Pixel and Channel Attention Block (PCAB)을 포함하여 높은 정밀도로 양쪽 뷰를 정렬하고 다양한 밝기 레벨을 고려하여 세밀한 텍스처를 복원합니다.

- **Performance Highlights**: SDI-Net은 Middlebury 및 Holopix50k 데이터셋에서 기존의 저조도 스테레오 이미지 향상 방법보다 우수한 성능을 보여주며, 여러 단일 이미지 저조도 향상 방법에 대해서도 우위를 점하고 있습니다.



### LBC: Language-Based-Classifier for Out-Of-Variable Generalization (https://arxiv.org/abs/2408.10923)
Comments:
          16 pages, 7 figures, 4 tables

- **What's New**: 이 논문에서는 Out-of-Variable (OOV) 작업에 대해 대형 언어 모델(LLMs)을 활용한 새로운 분류기인 Language-Based-Classifier (LBC)를 제안합니다. LBC는 전통적인 기계 학습 모델에 비해 OOV 작업에서의 성능을 통해 LLMs의 이점을 극대화하는 방법론을 제공합니다.

- **Technical Details**: LBC는 세 가지 주요 방법론을 사용하여 모델의 인지 능력을 향상시킵니다: 1) Categorical Changes를 통해 데이터를 모델의 이해에 맞추어 조정하고, 2) Advanced Order & Indicator로 데이터 표현을 최적화하며, 3) Verbalizer를 사용하여 추론 중 logit 점수를 클래스에 매핑합니다. 또한, LOw-Rank Adaptation (LoRA) 방법을 사용하여 분류기를 미세 조정합니다.

- **Performance Highlights**: LBC는 OOV 작업에 대해 이전 연구들 중 최초로 LLM 기반 모델을 적용하였으며, 이론적 및 경험적 검증을 통해 LBC의 우수성을 입증하였습니다. LBC는 기존 모델보다 OOV 작업에서 더 높은 성능을 보이는 것으로 나타났습니다.



### Recurrent Neural Networks Learn to Store and Generate Sequences using Non-Linear Representations (https://arxiv.org/abs/2408.10920)
- **What's New**: 이번 연구는 선형 표현 가설(Linear Representation Hypothesis, LRH)의 강한 해석에 대한 반례를 제시합니다. GRU(가중치가 있는 순환 신경망)가 입력 토큰 시퀀스를 반복하도록 훈련할 때, 각 위치의 표현이 방향이 아닌 특정 크기를 사용하여 구성된다는 사실을 발견하였습니다.

- **Technical Details**: 연구에서는 GRU 모델들이 각 위치에서의 토큰을 크기를 기반으로 표현하며, 이에 따라 새로운 비선형 표현인 '양파 표현(onion representations)'을 정의하였습니다. 작은 GRU는 이 크기 기반 솔루션만을 발견하였고, 큰 GRU는 LRH와 일치하는 선형 표현을 학습했습니다. 실험에서는 빈도 철회(Intervention) 방법을 활용하여 이 특징을 분석했습니다.

- **Performance Highlights**: 최소 크기의 GRU 모델들은 특정 위치에서의 토큰 변환을 대략 90%의 정확도로 성공적으로 수행하였고, 이는 비선형 및 양파 기반 표현이 실제로 존재함을 시사합니다. 또한, 이 연구는 연구자가 LRH의 틀을 넘어 다양한 방법론을 고려해야 한다는 점을 강조합니다.



### CrossFi: A Cross Domain Wi-Fi Sensing Framework Based on Siamese Network (https://arxiv.org/abs/2408.10919)
- **What's New**: 최근 Wi-Fi 센싱(Wi-Fi sensing)의 연구가 활발히 진행되고 있으며, 특히 CrossFi라는 새로운 프레임워크가 제안되었습니다. CrossFi는 siamese 네트워크에 기반하여 인도메인과 크로스도메인 시나리오에서 우수한 성능을 보이며, few-shot, zero-shot 및 new-class 시나리오를 포함한 다양한 환경에서 사용될 수 있습니다.

- **Technical Details**: CrossFi의 핵심 구성 요소는 CSi-Net이라는 샘플 유사도 계산 네트워크로, 주의 메커니즘(attention mechanism)을 사용하여 유사도 정보를 캡처하는 구조로 개선되었습니다. 추가로 Weight-Net을 개발하여 각 클래스의 템플릿을 생성하여 크로스 도메인 시나리오에서도 작동하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, CrossFi는 제스처 인식(task)에서 인도메인 시나리오에서 98.17%의 정확도를 달성하며, one-shot 크로스 도메인 시나리오에서 91.72%, zero-shot 크로스 도메인 시나리오에서 64.81%, one-shot new-class 시나리오에서 84.75%의 성능을 기록하였습니다. 이 성과들은 다양한 시나리오에서 CrossFi의 우수성을 입증합니다.



### The impact of labeling automotive AI as "trustworthy" or "reliable" on user evaluation and technology acceptanc (https://arxiv.org/abs/2408.10905)
Comments:
          36 pages, 12 figures

- **What's New**: 이 연구는 자동차 AI 기술에 대한 사용자 인식과 수용에 영향을 미치는 "신뢰할 수 있는(trustworthy)" 또는 "신뢰성 있는(reliable)" AI 레이블의 영향을 탐색합니다. 478명의 온라인 참가자들을 대상으로 진행된 연구로, AI의 신뢰성과 신뢰성에 대한 지침을 제시하여 응답을 평가하였습니다.

- **Technical Details**: 연구는 한 방향의 피실험자 디자인을 사용하여, 참가자들은 세 가지 시나리오를 평가하고, 인지된 사용 용이성(perceived ease of use), 인간과 유사한 신뢰(human-like trust) 등의 변수를 포함한 수정된 기술 수용 모델(Technology Acceptance Model, TAM)에 대한 설문지를 작성했습니다. 또한, AI를 신뢰할 수 있다는 레이블이 판단에 미치는 영향을 실험적으로 분석했습니다.

- **Performance Highlights**: "신뢰할 수 있는" AI라고 레이블을 붙이는 것이 특정 상황에 대한 판단에 큰 영향을 미치지 않았지만, 사용 용이성 및 인간과 유사한 신뢰, 특히 자비(benevolence)의 인식을 증가시켰습니다. 이는 AI의 사용성과 사용자 인식에서의 의인화(anthropomorphism) 효과를 긍정적으로 시사합니다.



### A Grey-box Attack against Latent Diffusion Model-based Image Editing by Posterior Collaps (https://arxiv.org/abs/2408.10901)
Comments:
          21 pages, 7 figures, 10 tables

- **What's New**: 최근 생성형 AI의 발전, 특히 Latent Diffusion Models (LDMs)이 이미지 합성 및 조작 분야에 혁신적인 변화를 가져왔습니다. 본 논문에서는 Posterior Collapse Attack (PCA)라는 새로운 공격 기법을 제안하여 현재의 방어 방법의 한계를 극복하고자 합니다.

- **Technical Details**: PCA는 VAE(Variational Autoencoder)의 posterior collapse 현상을 활용하여, LDM의 생성 품질을 현저히 저하시키는 방식으로 설계되었습니다. 이 방법은 black-box 접근법이며, 전체 모델 정보 없이도 효과를 발휘합니다. PCA는 LDM의 인코더를 통해 최소한의 모델 특화 지식으로도 실행 가능하며, 3.39%의 매개변수만을 사용하여 성능 저하를 유도합니다.

- **Performance Highlights**: PCA는 다양한 구조와 해상도에 걸쳐 LDM 기반 이미지 편집을 효과적으로 차단하며, 기존 기법보다 더 높은 전이성과 강인성을 보여줍니다. 실험 결과 PCA는 낮은 실행 시간과 VRAM으로 우수한 변이 효과를 달성하였고, 기존 기술들에 비해 더욱 일반화된 해결책을 제공합니다.



### V-RoAst: A New Dataset for Visual Road Assessmen (https://arxiv.org/abs/2408.10872)
- **What's New**: 이 논문은 도로 안전 평가를 위한 새로운 접근법인 V-RoAst를 제안하고, 전통적인 Convolutional Neural Networks (CNNs)의 한계를 극복하기 위해 Vision Language Models (VLMs)를 사용합니다.

- **Technical Details**: 연구는 상황에 맞는 프롬프트 엔지니어링(prompt engineering)을 최적화하고, Gemini-1.5-flash 및 GPT-4o-mini와 같은 고급 VLM들을 평가하여 도로 속성을 검사하는 방법을 개발합니다. Mapillary에서 크라우드소싱된 이미지를 사용하여 도로 안전 수준을 효율적으로 추정하는 확장 가능한 솔루션을 제공합니다.

- **Performance Highlights**: 이 접근법은 자원 부족으로 어려움을 겪는 지역 이해관계자들을 위해 설계되었으며, 훈련 데이터 없이 비용 효율적이고 자동화된 방법을 제공하여 전 세계 도로 안전 평가에 기여할 수 있습니다.



### Radio U-Net: a convolutional neural network to detect diffuse radio sources in galaxy clusters and beyond (https://arxiv.org/abs/2408.10871)
Comments:
          Accepted by MNRAS, 16 pages, 9 figures, 2 tables

- **What's New**: 새로운 라디오 망원경 배열 세대가 감도(sensitivity)와 해상도(resolution)에서 큰 발전을 약속하고 있으며, 이는 새로운 희미하고 확산된 라디오 소스들을 식별하고 특성화하는 데 도움을 줄 것이다. 이러한 발전을 활용하기 위해 Radio U-Net을 소개한다.

- **Technical Details**: Radio U-Net은 U-Net 아키텍처에 기반한 완전한 합성곱 신경망(fully convolutional neural network)으로, 라디오 조사의 희미하고 확장된 소스(예: radio halos, relics, cosmic web filaments)를 탐지하도록 설계되었다. 이 모델은 우주론적 시뮬레이션에 기반한 합성 라디오 관측 데이터로 훈련되었으며, 그런 후 LOFAR Two Metre Sky Survey (LoTSS) 데이터를 통해 게가 클러스터의 확산 라디오 소스를 시각적으로 검토하여 검증되었다.

- **Performance Highlights**: 246개의 게가 클러스터의 테스트 샘플에서, Radio U-Net은 73%의 정확도로 확산 라디오 방출이 있는 클러스터와 없는 클러스터를 구별하였다. 또한, 83%의 클러스터에서 확산 라디오 방출이 정확하게 식별되었고, 저품질 이미지를 포함한 다양한 이미지에서 소스의 형태(morphology)를 성공적으로 복원하였다.



### Knowledge Sharing and Transfer via Centralized Reward Agent for Multi-Task Reinforcement Learning (https://arxiv.org/abs/2408.10858)
- **What's New**: 본 논문에서는 다중 과제 강화 학습(multi-task reinforcement learning, MTRL) 프레임워크에 중앙 집중식 보상 에이전트(centralized reward agent, CRA)와 여러 개의 분산 정책 에이전트를 통합한 새로운 방법을 제안합니다. 이를 통해 즉각적인 피드백을 제공하고 학습 효율성을 높이는 것을 목표로 합니다.

- **Technical Details**: 제안된 CRA 기반 MTRL 프레임워크는 CRA가 다양한 작업에서 지식을 증류하고 이를 정책 에이전트에 분배하는 구조를 갖추고 있습니다. CRA는 경험을 통해 학습한 것들을 바탕으로 희소 보상을 밀집 보상(dense rewards)으로 변환하여 정책 에이전트에 다시 전달합니다. 추가적으로, 작업 간 유사성과 에이전트 학습 진행 상황을 고려하여 지식 분산을 균형 잡는 정보 동기화 메커니즘(information synchronization mechanism)을 도입하였습니다.

- **Performance Highlights**: CenRA 프레임워크는 희소한 외부 보상이 있는 이산 및 연속 제어 MTRL 환경에서 검증되었습니다. 학습 효율성, 지식 전이 능력 및 시스템 전반의 성능 측면에서 여러 기본 모델보다 우수한 성능을 보였습니다.



### MambaDS: Near-Surface Meteorological Field Downscaling with Topography Constrained Selective State Space Modeling (https://arxiv.org/abs/2408.10854)
- **What's New**: MambaDS라는 새로운 모델이 기상 분야의 downscaling에 통합되어, 기상 데이터의 고해상도 예측을 위한 다변량 상관관계를 보다 효과적으로 활용하고 지형 정보를 효율적으로 통합합니다.

- **Technical Details**: MambaDS는 Visual State Space Module (VSSM)에서 시작하여 Multivariable Correlation-Enhanced VSSM (MCE-VSSM)을 제안하며, 지형 제약 계층을 설계합니다. 이 구조는 기존 CNN 및 Transformer 기반 모델의 한계를 극복하고, 복잡성을 줄이면서 전세계적 문맥을 효과적으로 포착합니다.

- **Performance Highlights**: MambaDS는 중국 본토와 미국 대륙 내에서 진행된 여러 실험에서 기존의 CNN 및 Transformer 기반 모델보다 향상된 결과를 보여주어, 기상 데이터 downscaling 작업에서 최첨단 성능을 달성했습니다.



### Does Current Deepfake Audio Detection Model Effectively Detect ALM-based Deepfake Audio? (https://arxiv.org/abs/2408.10853)
- **What's New**: 본 논문은 최신 Audio Language Models (ALMs)에 기반한 딥페이크 오디오 탐지의 효율성을 평가합니다. 12가지 유형의 ALM 기반 딥페이크 오디오를 수집하고, 최신 대응 기술(countermeasure, CM)을 이용하여 평가한 결과, 매우 효과적인 탐지 성능을 보여 주었습니다.

- **Technical Details**: ALM은 대규모 언어 모델과 오디오 신경 코덱의 발전 덕분에 빠르게 발전하고 있으며, 이는 딥페이크 오디오 생성을 용이하게 하고 있습니다. 본 연구에서는 12유형의 ALM 기반 딥페이크 오디오를 수집하고, 최신 CM인 Codecfake를 사용하여 효과성을 평가했습니다. codec-trained CM은 대부분의 ALM 테스트 조건에서 0% 동일 오류율(equal error rate, EER)을 달성하는 성과를 보였습니다.

- **Performance Highlights**: 최신 codec 기반의 대응 기술이 ALM으로 생성된 딥페이크 오디오를 효과적으로 탐지할 수 있음을 보여 주었으며, 이는 앞으로의 ALM 기반 딥페이크 탐지 연구에 유망한 방향성을 제시합니다.



### Harmonizing Attention: Training-free Texture-aware Geometry Transfer (https://arxiv.org/abs/2408.10846)
Comments:
          10 pages, 6 figures

- **What's New**: 이 연구에서는 texture-aware geometry transfer를 위한 새로운 접근 방식인 Harmonizing Attention을 제안합니다. 이 방법은 diffusion 모델을 활용하여 매개변수 조정 없이도 다양한 재료의 기하학적 특징을 효과적으로 전달할 수 있게 해줍니다.

- **Technical Details**: Harmonizing Attention은 Texture-aligning Attention과 Geometry-aligning Attention의 두 가지 자가 주의 메커니즘을 포함합니다. 이 구조는 기존 이미지에서 기하학적 정보와 텍스쳐 정보를 쿼리하여, 재료에 독립적인 기하학적 특징을 포착하고 전달할 수 있습니다. 주의 메커니즘에서 수정된 자가 주의층이 inversion 및 generation 과정에서 통합됩니다.

- **Performance Highlights**: 실험 결과, Harmonizing Attention을 사용한 방법이 기존의 이미지 조화 기술과 비교했을 때 더 조화롭고 현실감 있는 합성 이미지를 생성함을 확인했습니다.



### Detecting Wildfires on UAVs with Real-time Segmentation Trained by Larger Teacher Models (https://arxiv.org/abs/2408.10843)
- **What's New**: 이번 연구는 UAV(무인 항공기)를 이용한 조기 산불 탐지 기술을 제안하며, 작은 카메라와 컴퓨터를 장착한 UAV가 환경의 피해를 줄이는 데 도움을 줄 수 있음을 보여줍니다. 특히, 제한된 고대역 모바일 네트워크 환경에서 실시간으로 탐지가 가능한 방법을 다룹니다.

- **Technical Details**: 연구에서는 경량화된 세분화(Segmentation) 모델을 통해 산불 연기의 픽셀 단위 존재를 정확히 추정하는 방법을 제시합니다. 이 모델은 바운딩 박스 레이블을 사용한 제로샷(Zero-shot) 학습 기반의 감독 하에 훈련되며, PIDNet이라는 최종 모델을 사용하여 실시간으로 연기를 인식합니다. 특히, 이 방법은 수동 주석이 있는 데이터셋을 기반으로 하여 63.3%의 mIoU(Mean Intersection over Union) 성능을 기록했습니다.

- **Performance Highlights**: PIDNet 모델은 UAV에 장착된 NVIDIA Jetson Orin NX 컴퓨터에서 약 11 fps로 실시간으로 작동하며, 실제 산불 발생 시 연기를 효과적으로 인지했습니다. 이 연구는 제로샷 학습을 통해 산불 탐지 모델의 훈련에 있어 낮은 장벽으로 접근할 수 있음을 증명했습니다.



### ZebraPose: Zebra Detection and Pose Estimation using only Synthetic Data (https://arxiv.org/abs/2408.10831)
Comments:
          8 pages, 5 tables, 7 figures

- **What's New**: 본 연구는 딥 러닝 작업에서 비정상적인 도메인에서 레이블이 지정된 이미지의 부족을 해결하기 위해 합성 데이터(synthetic data)를 사용하는 새로운 접근 방식을 제안합니다. 특히, 야생 동물인 얼룩말의 2D 자세 추정을 위해 처음으로 사용되는 합성 데이터셋을 생성하였으며, 기존의 실제 이미지 및 스타일 제약없이 탐지(detection)와 자세 추정(pose estimation) 모두를 수행할 수 있도록 하였습니다.

- **Technical Details**: 연구에서는 3D 포토리얼리스틱(simulator) 시뮬레이터를 활용하여 얼룩말의 합성 데이터를 생성하였으며, YOLOv5를 기반으로 한 탐지 모델과 ViTPose+ 모델을 사용하여 2D 키포인트(keypoints)를 추정합니다. 이를 통해 104K개의 수작업으로 레이블이 지정된 고해상도 이미지를 포함한 새로운 데이터셋을 제공하며, 이 데이터셋은 UAV를 통해 촬영된 이미지를 포함합니다. 또한, 합성 데이터만을 사용하여 실제 이미지에 대한 일반화 능력을 평가하였습니다.

- **Performance Highlights**: 연구 결과, 합성 데이터로 학습된 모델이 실제 얼룩말 이미지에 대해 일관되게 일반화된다는 것을 보여주었습니다. 이에 따라 최소한의 실제 이미지 데이터로도 말에 대한 2D 자세 추정으로 쉽게 일반화할 수 있음을 입증하였습니다.



### Exploiting Large Language Models Capabilities for Question Answer-Driven Knowledge Graph Completion Across Static and Temporal Domains (https://arxiv.org/abs/2408.10819)
- **What's New**: 본 논문에서는 Generative Subgraph-based KGC (GS-KGC)라는 새로운 지식 그래프 완성 프레임워크를 소개합니다. GS-KGC는 질문-답변 형식을 활용하여 목표 엔티티를 직접 생성하고, 불확실한 질문에 대한 다수의 답변 가능성을 해결합니다.

- **Technical Details**: GS-KGC는 지식 그래프(KG) 내의 엔티티 및 관계를 중심으로 서브그래프를 추출하여, 부정 샘플 및 이웃 정보를 separately (별도로) 얻는 전략을 제안합니다. 이 방법은 알고 있는 사실을 사용하여 부정 샘플을 생성하며, 대형 언어 모델(LLM)의 추론을 향상시키기 위해 컨텍스트 정보를 제공합니다.

- **Performance Highlights**: GS-KGC는 4개의 정적 지식 그래프(SKG)와 2개의 시간적 지식 그래프(TKG)에서 성능을 평가한 결과, 5개 데이터셋에서 state-of-the-art Hits@1 메트릭을 달성했습니다. 이 방법은 기존 KG 내에서 새로운 트리플을 발견하고, 닫힌 KG를 넘어 새로운 사실을 생성하여 효과적으로 닫힌세계(closed-world)와 열린세계(open-world) KGC 간의 간극을 좁힙니다.



### Beyond English-Centric LLMs: What Language Do Multilingual Language Models Think in? (https://arxiv.org/abs/2408.10811)
Comments:
          work in progress

- **What's New**: 이번 연구에서는 비영어 중심의 LLMs가 고성능에도 불구하고 해당 언어로 '사고'하는 방식을 조사합니다. 이는 중간 레이어의 표현이 특정 지배 언어에 대해 더 높은 확률을 보일 때 등장하는 개념인 내부 latent languages에 대한 것입니다.

- **Technical Details**: 연구에서는 Llama2(영어 중심 모델), Swallow(일본어에서 계속해서 사전 훈련된 영어 중심 모델), LLM-jp(균형 잡힌 영어 및 일본어 코퍼스로 사전 훈련된 모델) 세 가지 모델을 사용하여 일본어 처리에서의 내부 latent language를 조사했습니다. 이후 logit lens 방법을 통해 각 레이어의 내부 표현을 어휘 공간으로 비구속화하여 분석했습니다.

- **Performance Highlights**: Llama2는 내부 latent language로 전적으로 영어를 사용하며, Swallow와 LLM-jp는 영어와 일본어의 두 가지 내부 latent languages를 사용합니다. 일본어 특화 모델은 특정 목표 언어에 대해 가장 관련이 깊은 latent 언어를 선호하여 활성화합니다.



### DisMix: Disentangling Mixtures of Musical Instruments for Source-level Pitch and Timbre Manipulation (https://arxiv.org/abs/2408.10807)
- **What's New**: 이 논문에서 제안하는 DisMix 프레임워크는 다중 악기 혼합 오디오에서 피치(pitch)와 음색(timbre) 속성을 분리하고 이를 활용해 새로운 혼합물을 생성할 수 있도록 한다. 이는 기존 연구들이 주로 단일 악기 음악에 초점을 맞춘 것과는 차별점이 있다.

- **Technical Details**: DisMix는 피치와 음색 표현을 모듈형 빌딩 블록으로 사용하여 악기 및 멜로디를 생성하는 생성 프레임워크로, 각 악기를 나타내기 위해 상대적인 피치와 음색 잠재 변수를 결합한 소스 레벨 표현을 사용한다. 이 방식은 변형된 소스 레벨 표현을 입력으로 받아 새로운 혼합물을 생성하는 디코더를 포함한다.

- **Performance Highlights**: 모델은 간단한 고립된 코드 데이터셋과 J.S. 바흐 스타일의 네 부분 코랄을 포함한 보다 현실적인 데이터셋에서 평가되었으며, 피치와 음색의 성공적인 분리를 위한 핵심 구성 요소가 식별되었다. 또한, 혼합물을 재구성할 때 소스 레벨 속성 조작에 기반한 혼합물 변화의 응용 가능성을 보여준다.



### Inverse Deep Learning Ray Tracing for Heliostat Surface Prediction (https://arxiv.org/abs/2408.10802)
- **What's New**: 이 논문은 Concentrating Solar Power (CSP) 플랜트의 효율성을 높이기 위해 개별 헬리오스타트의 표면 프로파일을 예측하는 새로운 방법인 inverse Deep Learning Ray Tracing (iDLR)을 소개합니다. 기존의 방법들은 비효율적이거나 비용이 많이 드는 반면, iDLR은 타겟 이미지만을 바탕으로 헬리오스타트 표면을 예측할 수 있습니다.

- **Technical Details**: iDLR은 한 개의 헬리오스타트에서 방출된 flux 밀도의 분포가 함수에 대한 충분한 정보를 가지며, 이를 통해 딥러닝 모델이 실제 표면 특성(예: 켄팅(canting) 및 미러 오류(mirror errors))을 정확히 예측할 수 있도록 설계되었습니다. 이 연구에서는 Non-Uniform Rational B-Spline (NURBS)를 사용하여 새로운 헬리오스타트 모델을 제시하고 있으며, 이 방법이 기존의 상태(State of the Art) 을 뛰어넘을 수 있는 가능성을 보여줍니다.

- **Performance Highlights**: iDLR 모델은 기존의 물리 기반 알고리즘보다 더 나은 예측 성능을 보일 뿐만 아니라, CSP 플랜트의 전반적인 효율성과 에너지 출력을 증가시킬 수 있는 잠재력을 지니고 있습니다.



### Just a Hint: Point-Supervised Camouflaged Object Detection (https://arxiv.org/abs/2408.10777)
Comments:
          Accepted by ECCV2024

- **What's New**: 본 연구에서는 Camouflaged Object Detection (COD) 작업을 단 한 점 주석(point supervision)만으로 수행할 수 있는 혁신적인 접근 방식을 제안합니다. 기존의 pixel-wise 주석이 아닌, 빠른 클릭을 통해 대상을 지정하는 방식으로 주석 과정을 간소화하였습니다.

- **Technical Details**: 주요 기술적 요소로는 1) hint area generator로, 점 주석을 확장하여 대상 영역을 제시하며, 2) attention regulator로, 모델이 전체 객체에 주의를 분산시킬 수 있도록 돕습니다. 또한, 3) Unsupervised Contrastive Learning (UCL)을 활용하여 학습 과정에서의 불안정한 특징 표현을 개선합니다.

- **Performance Highlights**: 세 가지 COD 벤치마크에서의 실험 결과, 본 방법이 여러 약하게 감독된 방법과 비교하여 월등한 성능을 보이며, 심지어 완전 감독된 방법들도 초과 달성하는 성과를 기록했습니다. 또한, 다른 과제인 scribble-supervised COD 및 SOD으로의 이전 시에도 경쟁력 있는 결과를 얻었습니다.



### SSL-TTS: Leveraging Self-Supervised Embeddings and kNN Retrieval for Zero-Shot Multi-speaker TTS (https://arxiv.org/abs/2408.10771)
Comments:
          Submitted to IEEE Signal Processing Letters

- **What's New**: 이 연구에서는 SSL (self-supervised learning) 음성 특징을 활용하여 단일 화자의 청취로부터 학습된 경량의 제로샷 다중 화자 TTS (text-to-speech) 프레임워크인 SSL-TTS를 소개합니다. 이 방법은 복잡한 전사된 음성 데이터에 의존하지 않고도 효과적인 음성 합성을 가능하게 합니다.

- **Technical Details**: SSL-TTS는 텍스트를 SSL 임베딩으로 변환하고, 여기에 kNN 알고리즘을 통해 목표 화자의 특징과 매칭하여 새로운 음성을 합성하는 방식으로 작동합니다. 이때, 출력 음성의 스타일을 세부 조정할 수 있는 인터폴레이션 파라미터가 도입되어 화자의 정체성을 유지하면서 음성 합성의 유연성을 높입니다.

- **Performance Highlights**: 성능 평가 결과, SSL-TTS는 더 많은 훈련 데이터를 요구하는 최신 모델들과 비교하여 유사한 수준의 음질을 유지하며, 특히 데이터가 적은 도메인이나 언어에서의 다중 화자 TTS 시스템 개발에 적합합니다.



### SAM-COD: SAM-guided Unified Framework for Weakly-Supervised Camouflaged Object Detection (https://arxiv.org/abs/2408.10760)
Comments:
          Accepted by ECCV2024

- **What's New**: 본 논문에서는 약한 감독(weakly-supervised) 캄플라지 사진 물체 탐지(COD) 문제를 해결하기 위해 새로운 SAM-COD 프레임워크를 제안합니다. 기존 방법들은 상이한 약한 감독 레이블을 지원하는 데 한계가 있었으나, SAM-COD는 스크리블(scribble), 바운딩 박스(bounding box), 포인트(point) 형식의 레이블을 모두 지원합니다.

- **Technical Details**: SAM-COD 프레임워크는 세 가지 모듈로 구성됩니다: 1) Prompt Adapter: 스크리블 레이블의 스켈레톤을 추출하고 이를 포인트로 샘플링해 SAM과 호환되도록 처리합니다. 2) Response Filter: SAM에서의 비정상 반응을 필터링하여 신뢰성 높은 마스크를 생성합니다. 3) Semantic Matcher: 세맨틱 점수와 세그멘테이션 점수를 결합하여 정교한 객체 마스크를 선택합니다. 또한, Prompt-adaptive Knowledge Distillation을 통해 지식 전이를 강화합니다.

- **Performance Highlights**: SAM-COD는 세 가지 주요 COD 벤치마크 데이터셋에서 광범위한 실험을 통해 기존의 약한 감독 및 완전 감독 방법들에 비해 우수한 성능을 나타냈으며, 특히 모든 유형의 약한 감독 레이블 하에서 최첨단 성능을 기록했습니다. 이 방법은 Salient Object Detection(SOD) 및 Polyp Segmentation 작업으로 이전했을 때도 긍정적인 결과를 보였습니다.



### Generating Synthetic Fair Syntax-agnostic Data by Learning and Distilling Fair Representation (https://arxiv.org/abs/2408.10755)
- **What's New**: 최근 AI 응용 프로그램의 사용이 증가함에 따라 Data Fairness(데이터 공정성)에 대한 관심이 높아지고 있습니다. 본 논문에서는 Knowledge Distillation(지식 증류) 기반의 공정한 데이터 생성 기법을 제안하여, 기존의 공정한 생성 모델보다 성능을 향상시킵니다.

- **Technical Details**: 이 연구에서 제안하는 Fair Generative Model(공정 생성 모델, FGM)은 작은 아키텍처를 사용하여 잠재 공간에서 공정한 표현을 증류합니다. 이는 다양한 데이터 형식에 적용 가능하며, 훈련 과정에서 발생하는 계산 비용을 줄이고 안정성을 높입니다. 데이터의 공정성과 유용성을 유지하기 위해 Quality Loss(품질 손실)와 Utility Loss(유용성 손실)를 활용합니다.

- **Performance Highlights**: 이 새로운 기법은 공정성, 합성 샘플 품질 및 데이터 유용성에서 각각 5%, 5% 및 10%의 성능 향상을 보여주며, 기존 공정 생성 모델보다 우수한 성과를 나타냅니다.



### Security Assessment of Hierarchical Federated Deep Learning (https://arxiv.org/abs/2408.10752)
- **What's New**: 본 연구는 Hierarchical Federated Learning (HFL)의 보안 강도를 새로운 방법론으로 검토하여, 적대적 공격에 대한 회복 능력을 분석하고 있습니다. HFL은 교육 및 추론 시에의 적대적 공격에 견디는 방식에 중점을 두며, 다양한 데이터셋과 공격 시나리오에서 실험을 통해 이러한 내성을 평가했습니다.

- **Technical Details**: HFL은 여러 레벨의 집계 서버를 통하여 중앙 서버와 서로 연결된 구조를 갖추고 있으며, 데이터 프라이버시 및 분산 학습의 장점을 제공합니다. 적대적 공격이 시도되는 특정 상황에서의 HFL의 아키텍처가 회복력에 미치는 영향을 분석하였고, Data Poisoning Attacks (DPA) 및 Model Poisoning Attacks (MPA)의 취약점, 특히 백도어 공격에 대한 방어 메커니즘을 제시합니다.

- **Performance Highlights**: HFL은 비타겟(untargeted) DPA와 MPA에 대해서는 다단계 집계 다이어그램 덕분에 내성을 보이는 반면, 타겟(targer) 타겟 공격에 대한 취약점도 존재합니다. 그러나 적절한 방어 전략인 neural cleanser 방법의 구현은 백도어 공격에 효과적임을 입증하며, 이러한 방어의 중요성을 강조합니다.



### Pluto and Charon: A Time and Memory Efficient Collaborative Edge AI Framework for Personal LLMs Fine-Tuning (https://arxiv.org/abs/2408.10746)
Comments:
          Accepted by The 53rd International Conference on Parallel Processing (ICPP'24)

- **What's New**: Large Language Models (LLMs)의 개인화된 리소스 효율적인 조정 방안을 제안하는 PAC(Pluto and Charon) 프레임워크를 소개합니다. 이 프레임워크는 능률적인 알고리즘-시스템 공동 설계를 기반으로 하여 에지 장치에서 개인 LLM 조정의 리소스 한계를 극복합니다.

- **Technical Details**: PAC는 파라미터, 시간 및 메모리 측면에서 효율적인 개인 LLM 조정 기술을 구현합니다. Parallel Adapters를 사용하여 LLM 백본을 통한 전체 역 전파의 필요성을 피하고, 활성화 캐시 메커니즘을 이용하여 여러 에포크에 걸쳐 반복적인 전방 전파 과정을 생략합니다. 이를 통해 PAC는 에지 장치 간의 협동 훈련을 위한 하이브리드 데이터 및 파이프라인 병렬성을 활용합니다.

- **Performance Highlights**: PAC는 기존 최첨단 방법들보다 최대 8.64배 빠른 조정 속도를 달성하며, 메모리 소모는 최대 88.16% 감소합니다. 실험 결과는 PAC가 단순한 파라미터 효율성을 넘어서는 리소스 효율성을 제공함을 보여줍니다.



### Towards Efficient Large Language Models for Scientific Text: A Review (https://arxiv.org/abs/2408.10729)
- **What's New**: 최근의 대규모 언어 모델(LLMs)은 과학 분야의 복잡한 정보 처리에 새로운 시대를 열어주었습니다. 논문에서는 LLMs의 진화된 기능을 더 접근 가능한 과학 AI 솔루션으로 전환하는 방법과 이를 위한 도전과 기회를 탐구합니다.

- **Technical Details**: 이 논문은 LLMs의 두 가지 주요 접근 방식인 모델 크기 축소와 데이터 품질 향상에 대한 포괄적인 리뷰를 제공합니다. 특히, 과학 연구에서 LLMs의 활발한 활용을 위한 기법으로 Parameter-Efficient Fine-Tuning (PEFT), LoRA 및 Adapter Tuning과 같은 최신 기술들이 소개되었습니다.

- **Performance Highlights**: LLMs는 생명과학, 생물의학, 시각적 과학 등 다양한 과학 분야에서 인상적인 성능을 보였으며, 특히 PEFT 방식의 구현을 통한 모델이 제한된 자원에서도 효율적으로 활용될 수 있음을 보여주었습니다.



### Quantum Artificial Intelligence: A Brief Survey (https://arxiv.org/abs/2408.10726)
Comments:
          21 pages, 5 figures

- **What's New**: 이번 논문에서는 양자 인공지능(Quantum Artificial Intelligence, QAI)의 진척 상황을 요약하고, 향후 연구를 위한 몇 가지 개방된 질문들을 제시합니다.

- **Technical Details**: 양자 컴퓨팅(quantum computing)과 AI의 융합으로 이루어진 QAI는 여러 AI의 하위 분야(computationally hard problems)에서 양자 컴퓨팅을 사용하여 문제를 해결하는 가능성과 잠재력에 대한 주요 발견들을 강조합니다.

- **Performance Highlights**: QAI의 발전은 AI 방법들이 양자 컴퓨터 장치를 구축하고 운영하는 데 어떻게 활용될 수 있는지를 보여줍니다. 이러한 기술적 시너지는 양자 컴퓨팅과 AI 모두에 상당한 이점을 제공할 것으로 기대됩니다.



### MEGen: Generative Backdoor in Large Language Models via Model Editing (https://arxiv.org/abs/2408.10722)
Comments:
          Working in progress

- **What's New**: 이 논문은 자연어 처리(NLP) 작업을 위한 맞춤형 백도어를 생성하는 새로운 방법인 MEGen을 제안합니다. MEGen은 최소한의 부작용으로 백도어를 주입할 수 있는 에디팅 기반의 생성적 백도어 공격 전략을 제공합니다.

- **Technical Details**: MEGen은 고정된 메트릭스에 따라 선택된 트리거를 입력에 삽입하기 위해 언어 모델을 활용하고, 모델 내부의 백도어를 직접 주입하기 위한 편집 파이프라인을 설계합니다. 이 방법은 소량의 로컬 매개변수를 조정하여 백도어를 효과적으로 삽입하면서 원래의 모델 성능은 유지할 수 있도록 합니다.

- **Performance Highlights**: 실험결과 MEGen은 더 적은 샘플로도 백도어를 효율적으로 주입할 수 있으며, 다양한 다운스트림 작업에서 높은 공격 성공률을 달성하였습니다. 이는 깨끗한 데이터에 대한 원래 모델 성능을 유지하면서도 공격 시 특정 위험한 정보를 자유롭게 출력할 수 있음을 보여줍니다.



### Towards Foundation Models for the Industrial Forecasting of Chemical Kinetics (https://arxiv.org/abs/2408.10720)
Comments:
          Accepted into the IEEE CAI 2024 Workshop on Scientific Machine Learning and Its Industrial Applications (SMLIA2024)

- **What's New**: 이번 연구에서는 MLP-Mixer 아키텍처를 활용하여 화학 반응의 시계열 모델링을 위한 새로운 접근 방식을 제안합니다. 특히, ROBER 시스템을 사용하여 전통적인 수치 기법과 비교함으로써 이 방법의 효용성을 평가합니다.

- **Technical Details**: 본 연구에서는 PatchTSMixer라는 MLP-Mixer 아키텍처를 도입하여 stiff chemical kinetics를 모델링합니다. ROBER 시스템에서 얻은 데이터를 통해 모델의 예측 성능을 분석하였으며, 다차원 텐서를 활용한 패치 변환 및 MLP 믹서 레이어를 통해 상관관계를 학습합니다.

- **Performance Highlights**: PatchTSMixer는 stiff chemical kinetics의 예측에서 평균 0.0166%의 오차를 기록하며, 전통적인 수치 해법과의 결과가 양질의 일치를 보였습니다. 이는 MLP-Mixer 아키텍처가 산업적 사용에 적합함을 입증합니다.



### Offline Model-Based Reinforcement Learning with Anti-Exploration (https://arxiv.org/abs/2408.10713)
- **What's New**: 이번 연구에서는 노이즈를 예방하고, 상태에 대한 불확실성을 추정할 수 있는 Morse 모델 기반의 오프라인 RL 프레임워크(MoMo)를 제안합니다. MoMo는 동적 모델의 부정확성을 최소화하고 가치 추정을 방지하는 방법을 통해 데이터가 부족한 상황에서도 효과적으로 학습할 수 있도록 돕습니다.

- **Technical Details**: MoMo는 정책 제약과 비탐색 보너스를 결합하여 가치 과대평가를 방지하는 오프라인 모델 기반 RL 방법입니다. Morose neural network은 불확실성을 추정하여 동적 앙상블의 필요성을 제거하고, 더 나은 하이퍼파라미터 일반화를 가능하게 합니다. MoMo는 모델 자유형 및 모델 기반 변형을 갖추고 있으며, 아울러 OOD(Out-of-Distribution) 상태를 탐지하고 롤아웃을 종료하는 방법을 적용합니다.

- **Performance Highlights**: MoMo는 D4RL 데이터셋을 통한 실험에서 모델 자유형 및 모델 기반 버전 모두 훌륭한 성능을 보였으며, 특히 모델 기반 MoMo는 테스트된 대부분의 D4RL 데이터셋에서 이전의 모델 기반 및 모델 자유형 기준을 초과하는 성능을 입증했습니다.



### Coarse-to-Fine Detection of Multiple Seams for Robotic Welding (https://arxiv.org/abs/2408.10710)
- **What's New**: 이 논문에서는 RGB 이미지와 3D 포인트 클라우드를 활용하여 다중 용접 이음새를 효율적으로 추출할 수 있는 새로운 프레임워크를 제안합니다. 기존 연구는 주로 하나의 용접 이음새를 인식 및 위치 추적하는 데 초점을 맞추었으나, 이 논문은 한 번에 모든 이음새를 탐지하는 방법을 제시합니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 모듈로 구성되며, 첫 번째는 딥 러닝 방법을 기반으로 하는 조정 위치 모듈이며, 두 번째는 영역 성장(region growth)을 이용한 정밀 엣지 추출 모듈입니다. RGB 이미지를 사용하여 이음새의 근사 위치를 찾고, 이를 바탕으로 포인트 클라우드를 처리합니다. 알고리즘은 Fast Segment Anything Model(FastSAM)을 활용하여 불필요한 포인트 클라우드 정보를 제거하고, 효율성을 높이기 위해 다운샘플링을 수행합니다.

- **Performance Highlights**: 다양한 직선 및 곡선 형태의 용접 이음새가 포함된 물체에서의 실험을 통해 알고리즘의 유효성이 입증되었습니다. 이 방법은 실제 산업 응용 가능성을 강조하며, 영상 자료를 통해 실제 실험 과정도 확인할 수 있습니다.



### Variable Assignment Invariant Neural Networks for Learning Logic Programs (https://arxiv.org/abs/2408.10709)
- **What's New**: 본 논문에서는 학습 규칙을 관찰된 상태 전이에서 추출하는 Learning from interpretation transition (LFIT) 프레임워크를 다룹니다. LFIT는 기존의 심볼릭 알고리즘에서는 잡음이나 일반화 문제를 해결하지 못했으나, 본 논문에서 제안하는 δLFIT2 기술은 변수 순서 불변성(asymmetric의) 원리를 활용하여 이러한 문제를 해결합니다.

- **Technical Details**: δLFIT2는 상태 내 변수의 순열에 대해 불변성(variable permutation invariance) 기술을 도입하고, 서로 다른 출력 헤드를 사용하여 손실 함수의 최적화를 용이하게 합니다. 이는 신경망(Neural Network)의 구조에 대한 제약을 줄이며 메타-러닝(meta-learning) 모델로서 기능합니다.

- **Performance Highlights**: δLFIT2의 효과성과 확장성을 여러 실험을 통해 입증하였으며, 기존의 δLFIT+에 비해 정보 손실(Information loss)을 줄이고 성능을 향상시키는 결과를 도출하였습니다.



### AnyGraph: Graph Foundation Model in the Wild (https://arxiv.org/abs/2408.10700)
- **What's New**: 이번 연구에서는 AnyGraph라는 새로운 통합 그래프 모델을 제안합니다. 이 모델은 그래프 구조적 정보의 분포 이동, 다양한 특성 표현 공간, 새로운 그래프 도메인에 대한 빠른 적응 및 스케일링 법칙의 출현을 다룹니다.

- **Technical Details**: AnyGraph는 그래프 혼합 전문가(Graph Mixture-of-Experts, MoE) 구조를 기반으로 하여, 구조 및 특성 수준의 이질성이 있는 데이터에서 효과적으로 학습할 수 있도록 설계되었습니다. 이를 통해 우리는 선택된 전문가 네트워크를 통해 입력 그래프의 특성에 가장 적합한 전문가를 빠르게 식별하고 활성화할 수 있습니다.

- **Performance Highlights**: 38개 그래프 데이터셋에 대한 실험에서 AnyGraph는 제로샷 학습 성능이 뛰어난 모습을 보였으며, 빠른 적응 능력과 스케일링 법칙의 출현을 나타냈습니다. 이러한 특징은 AnyGraph가 여러 도메인에서 뛰어난 일반화 능력을 발휘하는 데 기여합니다.



### Towards Robust Knowledge Unlearning: An Adversarial Framework for Assessing and Improving Unlearning Robustness in Large Language Models (https://arxiv.org/abs/2408.10682)
Comments:
          13 pages

- **What's New**: 이번 논문에서는 LLM(대형 언어 모델)의 훈련 데이터에서 발생할 수 있는 문제 있는 지식의 영향을 줄이기 위해 동적 비학습 공격(Dynamic Unlearning Attack, DUA)과 잠재적 적대적 비학습(Latent Adversarial Unlearning, LAU) 프레임워크를 제안합니다. 기존의 비학습 방법들이 적대적 쿼리에 취약한 것을 인식하고, 이를 해결하기 위한 두 가지 새로운 방법론을 제시합니다.

- **Technical Details**: DUA는 자동화된 공격 프레임워크로, 비학습 모델의 강건성을 평가하기 위해 보편적인 적대적 접미사를 최적화합니다. 반면, LAU는 비학습 과정을 min-max 최적화 문제로 설정하고, 공격 단계에서는 perturbation 벡터를 훈련하여 LLM의 잠재 공간에 추가하고, 방어 단계에서는 이전에 훈련된 perturbation 벡터를 사용하여 모델의 강건성을 향상합니다.

- **Performance Highlights**: LAU 프레임워크를 사용한 결과, AdvGA 및 AdvNPO라는 두 가지 강력한 비학습 방법을 제안하며, 비학습의 효과성을 53.5% 이상 개선하고, 이웃 지식에 대한 감소를 11.6% 미만으로 유지하며, 모델의 일반적 능력에는 거의 영향을 미치지 않는 것으로 나타났습니다.



### Tensor tree learns hidden relational structures in data to construct generative models (https://arxiv.org/abs/2408.10669)
Comments:
          9 pages, 3 figures

- **What's New**: 본 논문에서는 Born 머신 프레임워크를 기반으로 하는 텐서 트리 네트워크(tensor tree network)를 사용하여 목표 분포 함수를 양자 파동 함수의 진폭으로 표현하는 일반적인 생성 모델 구축 방법을 제안합니다. 이 방법은 본드 상호 정보를 최소화하는 방향으로 트리 구조를 동적으로 최적화하는 것을 핵심 아이디어로 삼고 있습니다.

- **Technical Details**: 제안된 ATT(Adaptive Tensor Tree) 방법은 모델 분포 p𝜃를 Born 규칙에 따라 정의하고, 텐서 네트워크를 통해 양자 상태 |ψθ⟩를 표현합니다. 여기서 BMI(bond mutual information)를 통해 트리 구조를 최적화하며 강한 상관 관계가 있는 요소를 가까이 배치하는 방식으로 텐서 트리를 조정합니다.

- **Performance Highlights**: 테스트한 데이터셋으로는 랜덤 패턴, QMNIST 수기 숫자, 베이지안 네트워크, S&P500 주가 변동 패턴이 포함되며, ATT 방법은 기존의 텐서 트레인보다 향상된 성능을 보여줍니다. 예를 들어, ATT 방법을 사용하여 생성 모델이 128비트의 긴 시퀀스를 잘 학습하도록 하였으며, 이 과정에서 관련된 랜덤 변수를 가까이 위치시키는 효과를 확인했습니다.



### Probing the Safety Response Boundary of Large Language Models via Unsafe Decoding Path Generation (https://arxiv.org/abs/2408.10668)
- **What's New**: 이번 연구에서는 Large Language Models (LLMs)의 안전성 문제를 새롭게 접근하는 방법을 제안합니다. 특히, 안전 정렬(safety alignment) 방식으로 개선된 LLM에서도 숨겨진 취약점이 존재할 수 있으며, 이를 식별하기 위해 Cost Value Model (CVM)을 활용합니다.

- **Technical Details**: CVM은 특정 디코딩 단계에서 해로운 내용이 생성될 확률을 평가하는 Markov Decision Process (MDP)로 모델링되어 있습니다. 이 연구는 안전 정렬된 LLM의 디코딩 과정에서 해로운 경로를 자동으로 추출하는 혁신적인 방법을 제안하며, 이는 효율적인 상태 평가 모델을 통해 가능합니다.

- **Performance Highlights**: 실험 결과, LLaMA-2-chat 7B 모델이 총 39.18%의 해로운 콘텐츠를 출력하는 것으로 나타났으며, 이는 안전 정렬이 이루어진 LLM에서도 해로운 내용 생성의 가능성이 여전히 존재함을 보여줍니다. 다양한 프롬프트 최적화 방법을 통해 취약점을 악용할 수 있는 경로도 발견되었습니다.



### ETGuard: Malicious Encrypted Traffic Detection in Blockchain-based Power Grid Systems (https://arxiv.org/abs/2408.10657)
- **What's New**: 암호화된 트래픽 속에서 숨은 악성 공격을 탐지하기 위한 새로운 프레임워크인 ETGuard를 제안합니다. 이 프레임워크는 블록체인 기반의 전력망 시스템에 적합하며, 새로운 악성 트래픽으로부터 점진적으로 학습할 수 있는 기능을 가지고 있습니다.

- **Technical Details**: 이 연구에서는 데이터 전처리, 점진 학습 모듈, 탐지 모듈 등 세 가지 주요 구성 요소로 이루어진 프레임워크를 소개합니다. 자기 지도 학습(self-supervised learning)을 활용하여 패킷 특성을 추출하며, 점진 학습 손실(incremental learning losses)을 통하여 기존의 공격 패턴을 잊지 않고 새로운 패턴을 학습하도록 합니다.

- **Performance Highlights**: 우리의 방법은 세 가지 서로 다른 기준 데이터셋에서 우수한 성능을 보였으며, 블록체인 기반 전력망 시나리오를 위한 최초의 악성 암호화 트래픽 데이터셋인 GridET-2024를 구축했습니다. 연구는 최신 성과를 기록하며, 코드와 데이터셋은 공개되어 향후 연구에 기여할 것으로 기대됩니다.



### Vocabulary-Free 3D Instance Segmentation with Vision and Language Assistan (https://arxiv.org/abs/2408.10652)
- **What's New**: 이 논문에서는 3D 인스턴스 세분화(3D Instance Segmentation, 3DIS) 문제를 해결하기 위해 새로운 접근 방식을 제안하고 있습니다. 기존의 닫힌 어휘(closed vocabulary) 및 열린 어휘(open vocabulary) 방법들과는 달리, 어휘가 전혀 없는 환경에서 3DIS 문제를 해결하는 방법인 어휘 없음 설정(Vocabulary-Free Setting)을 도입합니다.

- **Technical Details**: 제안된 방법인 PoVo는 비전-언어 어시스턴트(vision-language assistant)와 2D 인스턴스 세분화 모델을 활용하여 3D 인스턴스 마스크를 형성합니다. 입력되는 포인트 클라우드를 밀집 슈퍼포인트(dense superpoints)로 분할하고, 이를 스펙트럴 클러스터링(spectral clustering)을 통해 병합함으로써 3D 인스턴스 마스크를 생성합니다. 이 과정에서 마스크의 일관성과 의미론적 일관성을 모두 고려합니다.

- **Performance Highlights**: ScanNet200 및 Replica 데이터셋을 사용하여 방법을 평가한 결과, 기존 방식에 비해 어휘 없음 및 열린 어휘 설정 모두에서 성능이 향상된 것을 보여주었습니다. PoVo는 기존의 VoF3DIS 설정에 적응된 최신 접근 방식보다 우수한 성능을 기록하였으며, 많은 의미론적 개념들을 효과적으로 관리할 수 있는 강력한 설계를 갖추고 있음을 입증했습니다.



### Inferring Underwater Topography with FINN (https://arxiv.org/abs/2408.10649)
- **What's New**: 이번 연구에서는 shallow-water equations (SWE)을 해결하기 위해 새롭게 도입된 hybrid architecture인 finite volume neural network (FINN)을 활용하여 수중 지형을 재구성하는 방법을 탐구합니다. 이는 wave dynamics을 기반으로 수중 지형을 추론할 수 있는 FINN의 효율성을 강조합니다.

- **Technical Details**: FINN은 finite volume method (FVM)와 deep neural networks의 학습 능력을 융합하여 기존의 Partial Differential Equations (PDEs)을 효과적으로 모델링할 수 있도록 설계되었습니다. FINN은 물리적 경계조건 (boundary conditions)을 직접적으로 통합할 수 있으며, 이는 FINN이 물리적 구조를 명시적으로 포함했음을 보여줍니다. 연구 결과, FINN은 wave dynamics에서만 수중 지형을 효율적으로 추론할 수 있는 놀라운 능력을 보여주었습니다.

- **Performance Highlights**: FINN은 기존의 pure ML 모델이나 다른 physics-aware ML 모델과 비교했을 때 뛰어난 성능을 발휘합니다. 본 연구에서 보여준 바와 같이, FINN은 두 다른 모델인 DISTANA와 PhyDNet에 비해 재구성된 지형의 품질이 우수한 것으로 나타났으며, 이는 FINN의 견고한 물리적 구조 덕분입니다.



### Privacy-preserving Universal Adversarial Defense for Black-box Models (https://arxiv.org/abs/2408.10647)
Comments:
          12 pages, 9 figures

- **What's New**: 본 논문에서는 블랙박스 환경에서 어떠한 모델에 대해서도 적용할 수 있는 범용 방어 기법 DUCD(Deep Uncertainty Certified Defense)를 소개합니다. DUCD는 타겟 모델의 내부 정보나 아키텍처에 접근하지 않고도 효과적인 방어를 구현합니다.

- **Technical Details**: DUCD는 쿼리 기반 방법을 통해 타겟 모델을 증류하여 화이트박스 계승 모델을 생성하고, 무작위 스무딩(randomized smoothing) 및 최적화된 노이즈 선택을 기반으로 이 모델을 강화합니다. 이 접근법은 다양한 적대적 공격에 대해 견고한 방어를 가능하게 합니다.

- **Performance Highlights**: 여러 이미지 분류 데이터셋에 대한 실험 결과 DUCD는 기존 블랙박스 방어 기법보다 더 나은 성능을 보여주었고, 화이트박스 방어의 정확도와 일치하며, 데이터 프라이버시를 강화하는 동시에 회원 추론 공격의 성공률을 감소시킵니다.



### Beneath the Surface of Consistency: Exploring Cross-lingual Knowledge Representation Sharing in LLMs (https://arxiv.org/abs/2408.10646)
- **What's New**: 이 논문은 멀티링구얼(다국어) LLMs(대형 언어 모델)에서 사실 정보의 표현이 여러 언어에 걸쳐 어떻게 공유되는지를 조사합니다. 특히 언어 간 일관성과 공유 표현의 중요성을 강조하고, LLM이 다양한 언어에서 사실을 얼마나 일관되게 응답하는지를 측정하는 방법론을 제안합니다.

- **Technical Details**: 논문에서 제안하는 측정 방법은 두 가지 주요 측면인 Cross-lingual Knowledge Consistency (CKC)와 Cross-lingual Knowledge Representation Sharing (CKR)을 기반으로 합니다. CKC는 모델이 다양한 언어에서 사실 질문에 일관성을 갖고 응답하는 정도를 측정하고, CKR은 동일한 사실에 대해 여러 언어 간의 공통 내적 표현을 사용하는 정도를 측정합니다.

- **Performance Highlights**: 연구 결과, LLMs가 각 언어에서 150% 더 많은 사실을 정확히 대답할 수 있다는 것을 발견했습니다. 특히 같은 스크립트에 속하는 언어 간에는 높은 표현 공유를 보였으며, 낮은 자원 언어와 다른 스크립트를 가진 언어 간에는 정답 일치도가 높지만 표현 공유는 적은 경향을 나타냈습니다.



### A Review of Human-Object Interaction Detection (https://arxiv.org/abs/2408.10641)
- **What's New**: 이 논문은 이미지 기반 인간-객체 상호작용(Human-Object Interaction, HOI) 탐지의 최신 연구 성과를 체계적으로 요약하고 논의합니다. 기존의 두 단계(methods) 및 일체형(end-to-end) 탐지 접근법을 포함하여 현재 개발되고 있는 방법론을 종합적으로 분석하고 있습니다.

- **Technical Details**: HOI 탐지는 이미지 또는 비디오에서 인간과 객체를 정확히 찾고, 이들 사이의 상호작용 관계를 분류하는 것을 목표로 합니다. 논문에서는 주요 탐지 방법을 두 가지 범주로 나누어 설명합니다: 두 단계 방법(두 단계 모두 인간-객체 탐지 및 상호작용 분류)과 일체형 방법(새로운 HOI 매개체를 사용하여 상호작용 관계를 직접 예측). 또한 제로샷 학습(zero-shot learning), 약한 감독 학습(weakly supervised learning) 및 대규모 언어 모델(large-scale language models)의 발전도 논의합니다.

- **Performance Highlights**: HICO-DET 및 V-COCO 데이터 세트와 같은 다양한 데이터 세트를 통해 HOI 탐지 기술의 성과를 평가합니다. 두 단계 방법은 간단하지만 계산 부담이 크고, 일체형 방법은 추론 속도가 빠른 장점을 가지지만 다중 작업 간의 간섭이 발생할 수 있는 단점이 있습니다.



### LLM-Barber: Block-Aware Rebuilder for Sparsity Mask in One-Shot for Large Language Models (https://arxiv.org/abs/2408.10631)
- **What's New**: 이번 연구에서는 기존의 비효율적인 pruning 기법들을 개선하기 위해 LLM-Barber라는 새로운 one-shot pruning 프레임워크를 제안합니다. 이 방법은 retraining이나 weight reconstruction 없이 pruning된 모델의 sparsity mask를 재구축할 수 있습니다.

- **Technical Details**: LLM-Barber는 Self-Attention 및 MLP 블록에서의 block-aware error optimization을 포함하여 글로벌 성능 최적화를 보장합니다. 새로운 pruning 메트릭은 가중치와 기울기를 곱하여 weight importance를 평가하며, 효과적인 pruning 결정을 내립니다.

- **Performance Highlights**: LLM-Barber는 7B에서 13B 파라미터를 가진 LLaMA 및 OPT 모델을 단일 A100 GPU에서 30분 만에 효율적으로 pruning하며, perplexity 및 zero-shot 성능에서 기존 방법들을 뛰어넘는 결과를 보여 줍니다.



### Finding the DeepDream for Time Series: Activation Maximization for Univariate Time Series (https://arxiv.org/abs/2408.10628)
Comments:
          16 pages, 4 figures, accepted at TempXAI @ ECML-PKDD

- **What's New**: 이 논문에서는 Sequence Dreaming이라는 새로운 기술을 소개합니다. 이 기술은 Activation Maximization을 시간 시계열 데이터에 적응시켜, 단변량(time series data에서 univariate) 시간 시계열을 처리하는 신경망의 해석 가능성을 높입니다.

- **Technical Details**: Sequence Dreaming은 활성화 극대화를 통해 신경망의 의사 결정 프로세스를 시각화하고, 특히 중요한 시계열의 패턴과 동역학을 강화하는 방법입니다. 이를 위해 다양한 정규화(regularization) 기법과 지수 스무딩(exponential smoothing)을 활용하여 비현실적이거나 지나치게 노이즈가 많은 시퀀스를 생성하는 것을 방지합니다.

- **Performance Highlights**: 예측 유지 관리(predictive maintenance)와 같은 응용 프로그램을 포함하는 시간 시계열 분류 데이터셋에서 검증된 결과, 제안하는 Sequence Dreaming 접근 방식이 다양한 사용 사례에 대해 목표 지향적인 활성화 극대화를 보여줍니다. 이 결과는 신경망이 학습한 중요한 시간적 특징을 추출함으로써 모델 투명성(transparency)과 신뢰성(trustworthiness)을 향상시키는 데 기여합니다.



### WRIM-Net: Wide-Ranging Information Mining Network for Visible-Infrared Person Re-Identification (https://arxiv.org/abs/2408.10624)
Comments:
          18 pages, 5 figures

- **What's New**: 이 논문은 Wide-Ranging Information Mining Network (WRIM-Net)을 선보이며, 이는 두 가지 차원(다양한 차원)에서 깊이 있는 정보를 채굴할 수 있는 능력을 가지고 있습니다. 특히, Multi-dimension Interactive Information Mining (MIIM) 모듈과 Auxiliary-Information-based Contrastive Learning (AICL) 접근 방식을 포함하여 모달리티 간 불일치를 극복할 수 있는 새로운 기법을 제안합니다.

- **Technical Details**: MIIM 모듈은 Global Region Interaction (GRI)을 활용하여 비국소적(spatial) 및 채널(channel) 정보를 포괄적으로 채굴합니다. 또한, AICL은 새로운 Cross-Modality Key-Instance Contrastive (CMKIC) 손실을 통해 모달리티 불변 정보를 추출할 수 있도록 네트워크를 유도합니다. 이러한 설계를 통해 MIIM은 얕은 층에 배치되어 특정 모달리티의 다차원 정보를 효과적으로 채굴할 수 있습니다.

- **Performance Highlights**: WRIM-Net은 SYSU-MM01, RegDB 및 LLCM 데이터셋을 포함한 다양한 벤치마크에서 이전의 방법들보다 뛰어난 성능을 보여주며, 모든 지표에서 최고의 성능을 기록했습니다.



### Novel Change Detection Framework in Remote Sensing Imagery Using Diffusion Models and Structural Similarity Index (SSIM) (https://arxiv.org/abs/2408.10619)
- **What's New**: 본 논문은 Stable Diffusion 모델과 Structural Similarity Index (SSIM)를 결합한 새로운 변화 감지 프레임워크인 Diffusion Based Change Detector를 제안합니다. 이 방법은 기존의 전통적 변화 감지 기법과 최근의 딥 러닝 기반 방법보다 뛰어난 성능을 보여줍니다.

- **Technical Details**: 제안하는 방법은 Stable Diffusion 모델의 강점을 활용하여 변화 감지를 위한 안정적이고 해석 가능한 변화 맵을 작성합니다. 이 프레임워크는 합성 데이터와 실세계 원격 감지 데이터셋에서 평가되었으며, 변화 감지의 정확성을 높이는 데 기여합니다.

- **Performance Highlights**: 실험 결과, Diffusion Based Change Detector는 복잡한 변화와 잡음이 있는 시나리오에서도 기존의 차이 기법 및 최근의 딥 러닝 기반 방법들보다 유의미하게 뛰어난 성능을 기대할 수 있음을 보여줍니다.



### OMEGA: Efficient Occlusion-Aware Navigation for Air-Ground Robot in Dynamic Environments via State Space Mod (https://arxiv.org/abs/2408.10618)
Comments:
          OccMamba is Coming!

- **What's New**: 본 논문에서는 OMEGA라는 새로운 시스템을 제안합니다. OMEGA는 동적 환경에서의 시공 식별과 경로 계획을 향상시키기 위해 OccMamba와 Efficient AGR-Planner를 통합한 것입니다. 이 시스템은 3D 의미적 점유 네트워크와 함께 에너지 효율적인 경로 계획 방식을 제공합니다.

- **Technical Details**: OMEGA의 핵심 구성 요소인 OccMamba는 의미 예측과 점유 예측을 독립적인 브랜치로 분리하여 각 영역에서 전문화된 학습을 가능하게 합니다. 이를 통해 3D 공간에서 시맨틱 및 기하학적 특징을 추출하며, 이 과정에서 전압이 얇고 효율적인 계산을 특징으로 합니다. 이렇게 얻어진 의미적 점유 맵은 지역 맵에 통합되어 동적 환경의 오클루전(occlusion) 인식을 제공합니다.

- **Performance Highlights**: 실험 결과, OccMamba는 최신 3D 의미적 점유 네트워크보다 25.0%의 mIoU(Mean Intersection over Union)를 기록하며, 22.1 FPS의 고속 추론을 달성합니다. OMEGA는 동적 시나리오에서 98%의 성공률을 보이며, 평균 운동 시간을 최단인 16.1초로 기록했습니다. Dynamic 환경에서의 실험에서도 OMEGA는 약 18%의 에너지 소비 절감 효과를 나타냈습니다.



### Generalizable Facial Expression Recognition (https://arxiv.org/abs/2408.10614)
Comments:
          Accepted by ECCV2024

- **What's New**: 이 논문에서는 기존의 Facial Expression Recognition (FER) 방법들이 도메인 간의 차이로 인해 테스트 세트에서 낮은 성능을 보인다는 문제를 해결하고자 합니다. 구체적으로, 한 개의 학습 세트만을 이용하여 다양한 테스트 세트에 대해 zero-shot generalization (제로샷 일반화) 능력을 개선할 수 있는 새로운 FER 파이프라인을 제안합니다.

- **Technical Details**: 제안된 방법은 고정된 CLIP 얼굴 특징을 이용하여 시그모이드 마스크를 학습하여 표현 피쳐를 추출하는 방식입니다. 후보 특징의 채널을 표현 클래스에 따라 분리하여 logits를 생성하고, FC 레이어를 사용하지 않아 과적합을 줄입니다. 또한, 채널 다양성 손실 (channel-diverse loss)을 도입하여 학습된 마스크를 표현에 따라 분리하여 일반화 능력을 향상시킵니다.

- **Performance Highlights**: 다섯 개의 각기 다른 FER 데이터셋에서 수행한 실험을 통해 제안한 방법이 기존의 SOTA FER 방법들보다 현저히 우수한 성능을 보였음을 확인했습니다. 이 결과들은 학습된 특징과 시그모이드 마스크의 시각화를 통해 이해를 돕습니다.



### Promoting Equality in Large Language Models: Identifying and Mitigating the Implicit Bias based on Bayesian Theory (https://arxiv.org/abs/2408.10608)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)에서 나타나는 '암묵적 편향 문제'(implicit bias problem)를 정식으로 정의하고, 이를 해결하기 위한 혁신적인 프레임워크인 베이지안 이론 기반의 편향 제거(Bayesian-Theory based Bias Removal, BTBR)를 개발하였습니다.

- **Technical Details**: BTBR는 우선적으로 공개적으로 접근 가능한 편향 데이터셋에서 의도하지 않게 포함된 편향을 나타내는 데이터를 식별하기 위해 우도 비율 스크리닝(likelihood ratio screening)을 사용합니다. 그 후, 관련된 지식 트리플(knowledge triples)을 자동으로 구성하고 모델 편집 기법(model editing techniques)을 통해 LLM에서 편향 정보를 삭제합니다.

- **Performance Highlights**: 실험을 통해, LLM에서 암묵적 편향 문제의 존재를 확인하고, BTBR 접근 방식의 효과성을 입증하였습니다.



### MUSES: 3D-Controllable Image Generation via Multi-Modal Agent Collaboration (https://arxiv.org/abs/2408.10605)
- **What's New**: 이 논문에서는 사용자 쿼리에 기반한 3D 제어 이미지 생성을 위한 일반적인 AI 시스템인 MUSES를 소개합니다. 기존의 텍스트-이미지 생성 방법들이 3D 세계에서 여러 객체와 복잡한 공간 관계를 생성하는 데 어려움을 겪고 있는 점에 주목하였습니다.

- **Technical Details**: MUSES는 (1) Layout Manager, (2) Model Engineer, (3) Image Artist의 세 가지 주요 구성 요소로 이루어진 점진적 프로세스를 개발하여 문제를 해결합니다. Layout Manager는 2D-3D 레이아웃 리프팅을 담당하고, Model Engineer는 3D 객체의 획득 및 보정을 수행하며, Image Artist는 3D-2D 이미지 랜더링을 진행합니다. 이러한 다중 모달 에이전트 파이프라인은 인간 전문가의 협업을 모방하여 3D 조작 가능한 객체의 이미지 생성과정을 효율적이고 자동화합니다.

- **Performance Highlights**: MUSES는 T2I-CompBench와 T2I-3DisBench에서 뛰어난 성능을 보여 DALL-E 3 및 Stable Diffusion 3와 같은 최근의 강력한 경쟁자들을 능가했습니다. 이 연구는 자연어, 2D 이미지 생성 및 3D 세계를 연결하는 중요한 발전을 보여줍니다.



### Multilingual Non-Factoid Question Answering with Silver Answers (https://arxiv.org/abs/2408.10604)
- **What's New**: MuNfQuAD라는 새로운 다국어 질의응답 데이터셋이 소개되었습니다. 이는 비사실 기반(non-factoid) 질문을 포함하고 있으며, BBC 뉴스 기사의 질문과 해당 단락을 사용하여 구성되었습니다.

- **Technical Details**: MuNfQuAD는 38개 언어를 포괄하며, 370,000개 이상의 QA 쌍으로 이루어져 있습니다. 수동 주석을 통해 790개의 QA 쌍이 검토되었으며, 98%의 질문이 해당하는 은유적 답변(silver answer)으로 답변될 수 있음을 확인했습니다. 이 연구에서는 Answer Paragraph Selection (APS) 모델이 최적화되어 기존 기준보다 뛰어난 성과를 보였습니다.

- **Performance Highlights**: APS 모델은 MuNfQuAD 테스트셋에서 80%의 정확도와 72%의 매크로 F1 점수를 기록하였으며, 골든 세트(golden set)에서는 각각 72%와 66%의 성과를 달성했습니다. 또한, 이 모델은 은유적 레이블(silver labels)로 최적화 된 후에도 골든 세트 내 특정 언어를 효과적으로 일반화할 수 있었습니다.



### MV-MOS: Multi-View Feature Fusion for 3D Moving Object Segmentation (https://arxiv.org/abs/2408.10602)
Comments:
          7 pages, 4 figures

- **What's New**: 이번 연구에서는 다양한 2D 표현에서의 모션-시맨틱(motion-semantic) 피처를 융합하는 새로운 멀티-뷰 이동 객체 세분화(Moving Object Segmentation, MOS) 모델(MV-MOS)을 제안합니다. 이 모델은 효과적으로 보조적인 시맨틱(semantic) 피처를 제공함으로써 3D 포인트 클라우드(point cloud)의 정보 손실을 줄이는 데 중점을 두고 있습니다.

- **Technical Details**: MV-MOS는 두 가지 뷰(예: Bird's Eye View, BEV 및 Range View, RV)로부터 모션 피처를 통합하는 다중 브랜치 구조를 가지고 있습니다. 이 구조는 Mamba 모듈을 활용하여 시맨틱 피처와 모션 피처를 융합하고, 불균일한 피처 밀도를 처리하여 세분화의 정확성을 향상시킵니다.

- **Performance Highlights**: 제안된 모델은 SemanticKITTI 벤치마크에서 검증 세트에서 78.5%의 IoU(Intersection over Union), 테스트 세트에서 80.6%의 IoU를 달성하였으며, 기존의 최첨단 오픈 소스 MOS 모델보다 우수한 성능을 보였습니다.



### Breast tumor classification based on self-supervised contrastive learning from ultrasound videos (https://arxiv.org/abs/2408.10600)
- **What's New**: 본 논문에서는 무작위 샘플이 포함된 유방 초음파(ultrasound) 영상을 이용하여 진단 정확도를 높일 수 있는 새로운 접근법을 제시했습니다.

- **Technical Details**: 트리플렛 네트워크(triplet network)와 자기 지도 대조 학습(self-supervised contrastive learning) 기술을 통해 라벨이 없는 유방 초음파 비디오 클립에서 표현을 학습했습니다. 또한, 패턴을 정확하게 구별하기 위한 새로운 하드 트리플렛 로스(hard triplet loss)를 설계했습니다.

- **Performance Highlights**: 모델은 수신자 조작 특성 곡선(receiver operating characteristic curve, AUC)에서 0.952의 값을 달성하여 기존 모델들보다 월등히 높은 성과를 보였습니다. 특히 <100개의 라벨링된 샘플로도 0.901의 AUC를 달성할 수 있음을 보여주어 라벨이 필요한 데이터의 요구를 대폭 줄이는 가능성을 제시했습니다.



### Putting People in LLMs' Shoes: Generating Better Answers via Question Rewriter (https://arxiv.org/abs/2408.10573)
Comments:
          7 pages, 4 figures, 5 tables

- **What's New**: 이번 논문은 Large Language Models (LLMs)이 질문 응답 (QA)에서의 성능이 사용자 질문의 모호성에 의해 저해된다는 문제를 다룹니다. 이를 해결하기 위해 편리한 방법인 single-round instance-level prompt optimization을 소개하며, 이를 통해 사용자 질문의 명확성을 극대화하는 질문 리라이터 (question rewriter)를 제안합니다.

- **Technical Details**: 리라이터는 자동 생성된 답변을 평가하는 기준으로부터 수집된 피드백을 기반으로 직접 선호 최적화 (direct preference optimization)를 사용하여 최적화됩니다. 이 방법은 비싼 인적 주석이 필요 없다는 장점이 있습니다. 여러 개의 블랙박스 LLM 및 Long-Form Question Answering (LFQA) 데이터셋을 통해 이 방법의 유효성을 실험적으로 입증하였습니다.

- **Performance Highlights**: 본 연구는 질문 리라이터의 훈련을 위한 실용적인 프레임워크를 제공하며, LFQA 작업에서의 프롬프트 최적화에 대한 향후 탐색의 전례를 설정합니다. 논문의 관련 코드는 공개되어 있습니다.



### A Tutorial on Explainable Image Classification for Dementia Stages Using Convolutional Neural Network and Gradient-weighted Class Activation Mapping (https://arxiv.org/abs/2408.10572)
Comments:
          15 pages, 11 figures, 3 tables

- **What's New**: 이 논문은 CNN(Convolutional Neural Network)과 Grad-CAM(Gradient-weighted Class Activation Mapping)을 이용하여 진행성 치매의 네 단계(4 progressive dementia stages)를 분류하는 설명 가능한 접근법을 제시합니다.

- **Technical Details**: 제안된 CNN 아키텍처는 오픈 MRI 뇌 이미지(open MRI brain images)를 기반으로 하여 훈련되었으며, 구현 단계에 대한 자세한 설명이 포함되어 있습니다. Grad-CAM을 사용하여 CNN의 블랙 박스(black box) 특성을 극복하고 높은 정확도의 시각적 이유를 제공합니다.

- **Performance Highlights**: 제안된 CNN 아키텍처는 테스트 데이터셋에서 99% 이상의 정확도를 달성하였으며, 이러한 높은 정확도가 의사들에게 유용한 정보를 제공할 가능성을 보입니다.



### Prompt-Agnostic Adversarial Perturbation for Customized Diffusion Models (https://arxiv.org/abs/2408.10571)
Comments:
          33 pages, 14 figures, under review

- **What's New**: 본 논문에서는 Personalized Diffusion Models를 위한 Prompt-Agnostic Adversarial Perturbation (PAP) 방법을 제안합니다. PAP는 라플라스 근사를 이용하여 프롬프트 분포를 모델링하고, 이를 기반으로 프롬프트에 구애받지 않는 공격 방어를 위한 교란을 생성합니다.

- **Technical Details**: PAP는 라플라스 근사를 사용하여 Gaussian 분포 형태의 프롬프트 분포를 모델링한 후, Monte Carlo 샘플링을 통해 교란을 계산합니다. 이 과정에서 Mean과 Variance는 2차 테일러 전개 및 헤시안 근사를 통해 추정됩니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋(VGGFace2, Celeb-HQ, Wikiart)에 대한 실험 결과, 제안된 PAP 방법은 기존의 프롬프트 특정 공격 방어 방법보다 현저히 성능 향상을 보였으며, 다양한 모델과 공격 프롬프트에 대해 강인한 효능을 입증했습니다.



### Prompt Your Brain: Scaffold Prompt Tuning for Efficient Adaptation of fMRI Pre-trained Mod (https://arxiv.org/abs/2408.10567)
Comments:
          MICCAI 2024

- **What's New**: Scaffold Prompt Tuning (ScaPT)은 fMRI 프리트레인 모델을 하위 작업에 적응시키기 위한 새로운 프롬프트 기반 프레임워크입니다. ScaPT는 파라미터 효율성이 높고, 기존의 fine-tuning 및 프롬프트 튜닝에 비해 성능이 개선되었습니다.

- **Technical Details**: ScaPT는 고급 리소스 작업에서 습득한 지식을 저급 리소스 작업으로 전송하는 계층적 프롬프트 구조를 설계합니다. Deeply-conditioned Input-Prompt (DIP) 매핑 모듈이 포함되어 있으며, 총 학습 가능한 파라미터의 2%만 업데이트합니다. 이 프레임워크는 입력과 프롬프트 간의 주의 메커니즘을 통해 의미론적 해석 가능성을 향상시킵니다.

- **Performance Highlights**: 대중의 휴식 상태 fMRI 데이터 세트를 통해 ScaPT는 신경퇴행성 질병 진단/예후 및 성격 특성 예측에서 기존의 fine-tuning 방법과 멀티태스크 기반 프롬프트 튜닝 방식을 초과하는 성과를 보였습니다. 20명 이하의 참가자와 함께 실험해도 뛰어난 성능이 입증되었습니다.



### SparseGrow: Addressing Growth-Induced Forgetting in Task-Agnostic Continual Learning (https://arxiv.org/abs/2408.10566)
Comments:
          This paper has been submitted to the AAAI conference. If accepted, the final version will be updated to reflect the conference proceedings

- **What's New**: 본 연구는 지속적 학습(Continual Learning) 분야에서 성장 유도 망각(Growth-induced Forgetting, GIFt) 문제를 최초로 식별하고 Layer Expansion을 통해 이를 해결하고자 합니다.

- **Technical Details**: SparseGrow는 데이터 기반의 희소 레이어 확장을 채택하여 성장 중에 효율적인 매개변수 사용을 제어하고, 불필요한 성장 및 기능 변화로 인한 GIFt를 줄이는 방식을 사용합니다. 또한 학습된 분포에 잘 맞는 부분적으로 0값인 확장을 생성하여 지식을 보존하고 적응성을 향상시키는 결과를 제공합니다.

- **Performance Highlights**: SparseGrow는 다양한 데이터 세트와 여러 설정에서 실험을 통해 GIFt를 극복하는 데 효과적임을 입증하였으며, 점진적 작업에 대한 적응성과 지식 보존 능력을 강조합니다.



### The Stable Model Semantics for Higher-Order Logic Programming (https://arxiv.org/abs/2408.10563)
- **What's New**: 본 논문은 고차 논리 프로그램을 위한 안정적인 모델 의미론(stable model semantics)을 제안합니다. 이 의미론은 근사 고정점 이론(Approximation Fixpoint Theory, AFT)을 사용하여 개발되었으며, 이는 비순차적(non-monotonic) 형식의 의미를 부여하는 데 이용되어왔습니다.

- **Technical Details**: 제안된 의미론은 Gelfond와 Lifschitz의 고전적인 2값 안정 모델 의미론과 Przymusinski의 3값 의미론을 일반화합니다. AFT를 이용하여 고차 논리 프로그램에 대한 여러 대체 의미론, 즉 지원 모델(supported model), Kripke-Kleene 및 잘 정립된 의미론(well-founded semantics)을 무료로 얻을 수 있습니다. 또한, 우리는 고차 논리 프로그램의 넓은 범주인 계층화된(stratified) 프로그램을 정의하고 이들이 고유한 2값 고차 안정 모델을 가짐을 입증합니다.

- **Performance Highlights**: 이 연구는 고차 논리 프로그래밍이 안정 모델 의미론 하에서 강력하고 다재다능한 형식임을 보여줍니다. 이는 새로운 ASP 시스템의 기초를 형성할 수 있는 가능성을 가지고 있으며, 다양한 응용 분야에서 실용성을 입증하는 여러 예제도 제시됩니다.



### Diff-PCC: Diffusion-based Neural Compression for 3D Point Clouds (https://arxiv.org/abs/2408.10543)
- **What's New**: Diff-PCC는 최초의 확산 기반 점 구름 압축 방법론으로, 생성 기반 및 미적 우수성을 갖춘 복원 기능을 활용합니다.

- **Technical Details**: Diff-PCC는 두 개의 독립적인 인코딩 백본으로 구성된 이중 공간 잠재 표현(Dual-space Latent Representation)을 통해 점 구름에서 표현력이 뛰어난 형태 잠재를 추출하고, 이러한 잠재를 이용하여 노이즈가 포함된 점 구름을 확률적으로 디노이즈하는 확산 기반 생성기(Diffusion-based Generator)를 사용합니다.

- **Performance Highlights**: Diff-PCC는 최신 G-PCC 표준 대비 초저비트레이트에서 7.711 dB BD-PSNR 향상으로 최첨단 압축 성능을 달성했으며, 비주얼 품질에서도 우수한 성능을 나타냅니다.



### NutrifyAI: An AI-Powered System for Real-Time Food Detection, Nutritional Analysis, and Personalized Meal Recommendations (https://arxiv.org/abs/2408.10532)
Comments:
          7 pages, 12 figures

- **What's New**: 이 논문은 YOLOv8 모델을 활용한 음식 인식 및 영양 분석 통합 시스템을 소개합니다. 이 시스템은 모바일 및 웹 애플리케이션으로 구현되어 사용자들이 음식 데이터를 수동으로 입력하지 않아도 되도록 지원합니다.

- **Technical Details**: 시스템은 세 가지 주요 구성 요소로 나뉘며, 1) YOLOv8 모델을 이용한 음식 감지, 2) Edamam Nutrition Analysis API를 통한 영양 분석, 3) Edamam Meal Planning 및 Recipe Search APIs를 통해 제공되는 개인 맞춤형 식사 추천입니다. YOLOv8은 빠르고 정확한 객체 탐지를 위해 설계되었습니다.

- **Performance Highlights**: 초기 결과는 YOLOv8 모델이 0.963의 mAP를 기록하여 높은 정확도를 보여주며, 사용자들이 보다 정확한 영양 정보를 기반으로 식단 결정을 할 수 있도록 하는 데 가치 있는 도구임을 증명하였습니다.



### EdgeNAT: Transformer for Efficient Edge Detection (https://arxiv.org/abs/2408.10527)
- **What's New**: 본 논문에서는 DiNAT를 인코더로 사용하는 EdgeNAT라는 새로운 1단계 Transformer 기반 엣지 감지기를 제안합니다. EdgeNAT는 객체 경계 및 유의미한 엣지를 정확하고 효율적으로 추출할 수 있습니다.

- **Technical Details**: EdgeNAT는 global contextual information과 detailed local cues를 효율적으로 캡처합니다. 새로운 SCAF-MLA 디코더를 통해 feature representation을 향상시키며, inter-spatial 및 inter-channel 관계를 활용하여 엣지를 추출합니다. 모델은 다양한 데이터셋에서 성능을 검증하였고, BSDS500 데이터셋에서 ODS 및 OIS F-measure 각각 86.0%, 87.6%를 달성했습니다.

- **Performance Highlights**: EdgeNAT는 RTX 4090 GPU에서 single-scale input으로 20.87 FPS의 속도로 실행되며, 기존의 EDTER보다 ODS F-measure에서 1.2% 더 높은 성능을 보여주었습니다. 또한, 다양한 모델 크기를 제공하여 스케일러블한 특성을 갖추고 있습니다.



### XCB: an effective contextual biasing approach to bias cross-lingual phrases in speech recognition (https://arxiv.org/abs/2408.10524)
Comments:
          accepted to NCMMSC 2024

- **What's New**:  본 연구에서는 이중 언어 설정에서 기계 음성 인식(ASR) 성능을 향상시키기 위해 'Cross-lingual Contextual Biasing(XCB)' 모듈을 도입하였습니다. 이 방법은 주 언어 모델에 보조 언어 편향 모듈과 언어별 손실을 결합하여 이차 언어의 문구 인식을 개선하는 데 중점을 둡니다.

- **Technical Details**:  제안한 XCB 모듈은 언어 편향 어댑터(LB Adapter) 및 편향 병합 게이트(BM Gate)라는 두 가지 핵심 구성 요소로 이루어져 있습니다. LB 어댑터는 이차 언어(secondary language, L2nd)에 연관된 프레임을 구별하여 해당 표현을 강화합니다. BM 게이트는 언어 편향 표현을 생성하며, 이를 통해 L2nd 문구의 인식을 향상시킵니다.

- **Performance Highlights**:  실험 결과, 제안된 시스템은 이차 언어의 편향 문구 인식에서 유의미한 성능 향상을 보였으며, 추가적인 추론 오버헤드 없이도 효율성을 발휘하였습니다. 또한, ASRU-2019 테스트 세트에 적용했을 때도 일반화된 성능을 보여주었습니다.



### Integrating Multi-Modal Input Token Mixer Into Mamba-Based Decision Models: Decision MetaMamba (https://arxiv.org/abs/2408.10517)
- **What's New**: 이번 연구에서는 Decision MetaMamba(DMM)라는 새로운 의사결정 모델을 제안하며, 이는 입력층을 수정하여 성능 향상을 이루어냈습니다. 또한, Selective Scan SSM을 활용하여 짧은 시퀀스에서도 효과적으로 패턴을 추출하고, 여러 시퀀스를 결합하는 방식을 도입했습니다.

- **Technical Details**: DMM은 1D convolution과 linear layers를 통합한 다중 모드(input layer)이며, causal convolution을 사용하여 정보 흐름을 최적화합니다. 선택적 스캔을 통해 전체 시퀀스를 고려하여 출력 결과를 생성하며, positional encoding을 제거하여 행동 클로닝을 감소시킵니다.

- **Performance Highlights**: DMM은 다양한 오프라인 RL 데이터셋에서 뛰어난 성능을 입증하였으며, 매개변수가 적은 경량 모델에서도 성능을 유지했습니다. 특히, 도메인 특화된 수정된 입력층을 통해 높은 성과를 보여주었습니다.



### Data Augmentation Integrating Dialogue Flow and Style to Adapt Spoken Dialogue Systems to Low-Resource User Groups (https://arxiv.org/abs/2408.10516)
Comments:
          Accepted to SIGDIAL 2024

- **What's New**: 본 연구는 특정 대화 행동을 보이는 사용자, 특히 데이터가 부족한 경우 미성년자와의 상호작용에서 발생하는 도전 과제를 해결하기 위해 특별히 설계된 데이터 증강(data augmentation) 프레임워크를 제시합니다.

- **Technical Details**: 우리의 방법론은 대규모 언어 모델(LLM)을 활용하여 화자의 스타일을 추출하고, 사전 훈련된 언어 모델(PLM)을 통해 대화 행동 이력을 시뮬레이션합니다. 이는 각기 다른 사용자 집단에 맞춰져 풍부하고 개인화된 대화 데이터를 생성하여 더 나은 상호작용을 돕습니다. 연구는 미성년자와의 대화 데이터를 이용하여 맞춤형 데이터 증강을 수행했습니다.

- **Performance Highlights**: 다양한 실험을 통해 우리의 방법론의 효용성을 검증하였으며, 이는 더 적응적이고 포용적인 대화 시스템의 개발 가능성을 높이는 데 기여할 수 있음을 보여주었습니다.



### Single-cell Curriculum Learning-based Deep Graph Embedding Clustering (https://arxiv.org/abs/2408.10511)
- **What's New**: 본 논문에서는 단일 세포 커리큘럼 학습 기반 심층 그래프 임베딩 클러스터링(scCLG)을 제안합니다. 이 모델은 Chebyshev 그래프 컨볼루셔널 오토인코더(ChebAE)를 활용하여 세포 간의 위상 구조를 보존하면서 클러스터를 식별합니다.

- **Technical Details**: 이 연구에서는 Chebyshev 그래프 컨볼루셔널 오토인코더(ChebAE)와 다중 디코더를 사용하는 새로운 방법론을 도입합니다. 세 가지 최적화 목표를 제안하여 세포-세포 위상 표현을 학습하고, 지역 및 전역 관점에서 훈련 노드의 어려움을 평가하는 계층적 난이도 측정기를 설계합니다.

- **Performance Highlights**: 실제 scRNA-seq 데이터셋 7개에서 평가한 결과, scCLG는 기존의 최첨단 방법들보다 우수한 성능을 보여주었습니다. 이 모델은 난이도 높은 노드를 제거하여 고품질 그래프를 유지하며, 클러스터링 결과의 품질을 최적화합니다.



### Adaptive Knowledge Distillation for Classification of Hand Images using Explainable Vision Transformers (https://arxiv.org/abs/2408.10503)
Comments:
          Accepted at the ECML PKDD 2024 (Research Track)

- **What's New**: 본 연구는 손 이미지의 분류를 위해 Vision Transformers (ViTs)의 사용을 조사하였으며, 특히 다른 도메인 데이터에 학습 중인 모델의 지식을 효과적으로 전달하는 적응형 증류 방법을 제안합니다.

- **Technical Details**: ViT 모델은 손의 고유한 특징과 패턴을 분류하기 위해 사용됩니다. 연구에서는 영상 변환기(vision transformer)와 설명 가능한 도구(explainability tools)를 결합하여 ViTs의 내부 표현을 분석하고 모델 출력에 미치는 영향을 평가합니다. 또, 두 공개 손 이미지 데이터셋을 이용하여 여러 실험을 수행하였습니다.

- **Performance Highlights**: 실험 결과, ViT 모델은 기존의 전통적인 머신러닝 방법을 크게 초월하며, 적응형 증류 방법을 통해 소스 도메인과 타겟 도메인 데이터를 모두 활용하여 우수한 성능을 발휘합니다. 이러한 접근법은 액세스 제어, 신원 확인, 인증 시스템과 같은 실제 응용에 효과적으로 활용될 수 있습니다.



### ProgramAlly: Creating Custom Visual Access Programs via Multi-Modal End-User Programming (https://arxiv.org/abs/2408.10499)
Comments:
          UIST 2024

- **What's New**: 이 논문은 시각적으로 보조 기술의 사용자 맞춤화의 필요성을 강조하며, 비전문가 사용자들이 assistive technology를 DIY 방식으로 프로그래밍할 수 있게 하는 ProgramAlly라는 시스템을 소개합니다.

- **Technical Details**: ProgramAlly는 사용자들이 시각 정보 필터링 프로그램을 생성하고 사용자 정의할 수 있도록 하는 모바일 애플리케이션입니다. 이 시스템은 block programming, natural language, programming by example의 세 가지 접근 방식을 통해 다양한 프로그래밍 모드를 제공합니다. 이를 통해 사용자는 특정 정보 ('find NUMBER on BUS')를 보다 쉽게 찾을 수 있습니다.

- **Performance Highlights**: 사용자 연구 결과, 12명의 시각 장애인 참가자는 각자 다른 프로그래밍 모드를 선호했으며, ProgramAlly를 통해 기존 assistive applications보다 자신의 요구에 맞춘 visual access 프로그램을 활용할 수 있을 것으로 기대했습니다. 사용자는 프로그래밍 경험이 없어도 assistive technology를 맞춤화할 수 있다는 것을 수용하였습니다.



### QUITO-X: An Information Bottleneck-based Compression Algorithm with Cross-Attention (https://arxiv.org/abs/2408.10497)
- **What's New**: 이번 연구는 정보 병목 이론(Information Bottleneck theory)을 활용하여 문맥 압축과 관련된 중요한 토큰을 구별하는 새로운 메트릭을 제시합니다. 기존의 self-information이나 PPL과 같은 메트릭 대신, encoder-decoder 아키텍처의 cross-attention을 사용함으로써 압축 기술의 성능을 개선했습니다.

- **Technical Details**: 이 연구에서는 특징적인 문맥 압축 알고리즘을 개발하였으며, ML 및 AI 분야의 일반적인 사용 사례에 대한 실험을 통해, 이 방법이 기존 방법들에 비해 효과적으로 작동함을 입증했습니다. 특히, DROP, CoQA, SQuAD, Quoref와 같은 데이터셋에서 성능을 평가했습니다. 우리 접근법은 query와 문맥 간의 상호 정보(mutual information)를 최적화하여 중요한 부분을 선택적으로 유지합니다.

- **Performance Highlights**: 실험 결과, 제안한 압축 방법은 이전 SOTA(최신 기술) 대비 약 25% 더 향상된 압축 비율을 보였습니다. 또한, 25%의 토큰을 제거한 상태에서도, 우리의 모델은 비압축 텍스트를 사용하는 대조군의 EM 점수보다 더 높은 결과를 보여주기도 했습니다.



### How Well Do Large Language Models Serve as End-to-End Secure Code Producers? (https://arxiv.org/abs/2408.10495)
- **What's New**: 본 논문은 대형 언어 모델(LLMs)인 GPT-3.5 및 GPT-4가 코드 생성 과정에서 보안 취약점을 탐지하고 수정할 수 있는 능력을 평가합니다. 특히, 보안에 민감한 시나리오에서 이 모델들이 생성한 코드의 보안성을 체계적으로 조사하였으며, 이전 연구에 비해 Python 프로그래밍 생태계에서의 성능을 보다 포괄적으로 분석하였습니다.

- **Technical Details**: 연구에서는 4,900개의 코드 스니펫을 수동 또는 자동 리뷰하여 LLM이 생성한 코드의 75% 이상이 보안 취약점을 포함하고 있음을 발견했습니다. LLM은 생성한 코드의 취약점을 정확히 식별하지 못하며, GPT-3.5와 GPT-4는 각각 33.2%와 59.6%의 성공률로 취약점 수리를 수행할 수 있지만, 자신이 생성한 코드의 자가 수리에는 한계를 보였습니다. 이를 보완하기 위해 경량 도구를 개발하여 반복적 수리 절차를 통해 코드의 보안을 높이는 방법을 제안합니다.

- **Performance Highlights**: 새롭게 개발된 도구는 의미 분석 엔진과 함께 사용할 때 65.9%~85.5%의 향상된 수리 성공률을 보여 주며, 이로써 LLM이 생성한 코드의 보안성을 보다 효과적으로 개선할 수 있음을 증명했습니다.



### Achieving the Tightest Relaxation of Sigmoids for Formal Verification (https://arxiv.org/abs/2408.10491)
- **What's New**: 이번 논문에서는 시그모이드(Sigmoid) 활성화 함수를 고려하여, 이를 최적의 경계로 설정할 수 있는 조정 가능한 하이퍼플레인을 도출하였습니다. 새로운 접근법인 α-sig을 통해 가장 긴밀한 범위의 요소별(convex relaxation) 변환을 정식 검증 형식에 통합하는 방법을 제안하였습니다.

- **Technical Details**: 시그모이드 함수의 선형 기울기와 y-절편의 접선(tangent line) 간 관계를 명확히 매핑하여 사용하였고, 이를 통해 경량의 조정 가능한 경계를 생성하였습니다. 또한 경량 NN 평가 루틴을 통해 각 단계에서 시그모이드 함수를 상한 또는 하한으로 조정하며, 이중 공간에서 효율적인 최대 기울기 경계를 사전 계산하는 순차적 2차 프로그램을 설계하였습니다.

- **Performance Highlights**: 이 접근법은 LiRPA 및 α-CROWN과 같은 최신 검증 알고리즘에 비해 경쟁력을 지니며, 실험 결과 tightest convex relaxation을 통해 효율적인 검증 속도를 달성하였습니다. 기존 검증 기술에 비해 성능이 향상되어, Large Language Model(LLM)의 검증 가능성을 증대시켰습니다.



### Event Stream based Sign Language Translation: A High-Definition Benchmark Dataset and A New Algorithm (https://arxiv.org/abs/2408.10488)
Comments:
          First Large-scale and High-Definition Benchmark Dataset for Event-based Sign Language Translation

- **What's New**: 본 논문에서는 Sign Language Translation (SLT)의 새로운 접근 방법으로 Event streams를 제안합니다. 이 방법은 조명, 손 움직임 등으로부터 영향을 덜 받으며, 빠르고 높은 동적 범위를 갖추고 있어 정확한 번역을 가능하게 합니다. 또한, 새로운 Event-CSL 데이터셋을 제안하여 연구에 기여합니다.

- **Technical Details**: Event-CSL 데이터셋은 14,827개의 고화질 비디오로 구성되며, 14,821개의 glosses 및 2,544개의 중국어 단어가 포함되어 있습니다. 이 데이터는 다양한 실내 및 실외 환경에서 수집되었으며, Convolutional Neural Networks (CNN) 기반의 ResNet18 네트워크와 Mamba 네트워크를 결합한 하이브리드 아키텍처를 사용하여 SLT 성능을 향상시킵니다.

- **Performance Highlights**: 본 연구에서는 Event-CSL 데이터셋과 기존 SLT 모델들을 벤치마킹하여, SLT 분야에서의 효과적인 성능 개선을 입증하였습니다. 이 하이브리드 CNN-Mamba 아키텍처는 기존의 Transformer 기반 모델보다 더 나은 성능을 보입니다.



### MambaEVT: Event Stream based Visual Object Tracking using State Space Mod (https://arxiv.org/abs/2408.10487)
Comments:
          In Peer Review

- **What's New**: 본 논문에서는 이벤트 카메라 기반의 시각 추적(Visual Tracking) 알고리즘의 성능 병목 현상을 해결하기 위해 새로운 Mamba 기반의 시각 추적 프레임워크인 MambaEVT를 제안합니다. 이 프레임워크는 상태 공간 모델(State Space Model, SSM)을 기반으로 하여 낮은 계산 복잡성을 달성하며, 동적 템플릿 업데이트 전략을 통합하여 더 나은 추적 성능을 구현합니다.

- **Technical Details**: MambaEVT는 이벤트 스트림에서 정적 템플릿과 검색 영역을 추출하여 이벤트 토큰으로 변환하며, Mamba 네트워크를 통해 특징 추출 및 상호작용 학습을 수행합니다. 메모리 Mamba 네트워크를 사용하여 동적 템플릿을 생성하고, 추적 결과에 따라 템플릿 라이브러리를 업데이트합니다. 계산 복잡도는 O(N)으로, 기존의 Transformer 기반 방법보다 우수한 실적을 보입니다.

- **Performance Highlights**: MambaEVT는 EventVOT, VisEvent, FE240hz와 같은 여러 대규모 데이터셋에서 정확도와 계산 비용 간의 균형을 잘 유지하며, 기존 방법들과 비교하여 뛰어난 성능을 입증하였습니다.



### Evaluation Framework for AI-driven Molecular Design of Multi-target Drugs: Brain Diseases as a Case Study (https://arxiv.org/abs/2408.10482)
Comments:
          8 pages, 1 figure, published in 2024 IEEE Congress on Evolutionary Computation (CEC)

- **What's New**: 이 논문은 Multi-target Drug Discovery (MTDD)에서 분자 생성 기술의 효과를 평가하기 위한 새로운 평가 프레임워크를 제안합니다. 이는 AI 기반 분자 설계 전략을 간섬기 위한 것으로, 특히 뇌 질환을 사례 연구로 삼았습니다.

- **Technical Details**: 이 본 연구에서는 대형 언어 모델(LLMs)을 활용하여 적절한 분자 표적을 선택하고, 생물학적 테스트 데이터를 수집 및 전처리하며, 정량적 구조-활동 관계 모델을 교육하여 표적 조절을 예측하는 과정을 포함합니다. 또한, 제안된 벤치마크 스위트에서 4개의 심층 생성 모델(deep generative models) 및 진화 알고리즘(evolutionary algorithms)의 성능을 평가합니다.

- **Performance Highlights**: 이 연구의 결과는 진화 알고리즘과 생성 모델 모두 제안된 벤치마크에서 경쟁력 있는 결과를 달성할 수 있음을 보여주었습니다. 이로 인해 MTDD 분야에서 AI의 효과적인 활용 가능성을 더욱 높이고 있습니다.



### An End-to-End Reinforcement Learning Based Approach for Micro-View Order-Dispatching in Ride-Hailing (https://arxiv.org/abs/2408.10479)
Comments:
          8 pages, 4 figures

- **What's New**: 이 논문에서는 Didi에서 마이크로 뷰 오더 디스패칭(를 위한) 단일 스테이지 강화 학습(reinforcement learning) 기반 방법을 제안합니다. 이 방법은 행동 예측과 조합 최적화(combinatorial optimization) 문제를 통합하여 처리합니다.

- **Technical Details**: 특히, 두 층의 Markov Decision Process (MDP) 프레임워크를 채택하여 문제를 모델링하고, Deep Double Scalable Network (D2SN)라는 인코더-디코더 네트워크 구조를 통해 주문-드라이버 할당을 직접 생성합니다. 이 방법은 문맥적 동향을 활용하여 행동 패턴에 적응할 수 있는 특성을 가지고 있습니다.

- **Performance Highlights**: Didi의 실제 벤치마크에서 실시된 광범위한 실험 결과, 제안된 접근 방식이 조합 최적화 및 두 단계 강화 학습 방법과 비교하여 매칭 효율성과 사용자 경험 최적화에서 주목할 만한 성과를 보였습니다.



### LeCov: Multi-level Testing Criteria for Large Language Models (https://arxiv.org/abs/2408.10474)
- **What's New**: 리서치에서는 Large Language Models (LLMs)의 신뢰성 문제를 해결하기 위한 LeCov라는 새로운 다수의 수준 테스트 기준을 제안합니다.

- **Technical Details**: LeCov는 LLM의 내부 구성 요소인 attention mechanism, feed-forward neurons, 그리고 uncertainty를 기반으로 합니다. 총 9가지 테스트 기준이 포함되어 있으며, 테스트 우선 순위(test prioritization)와 커버리지 기반 테스트(coverage-guided testing)와 같은 두 가지 시나리오에 적용됩니다.

- **Performance Highlights**: 세 개의 모델과 네 개의 데이터셋을 이용한 실험적 평가 결과, LeCov가 유용하고 효과적임을 입증했습니다.



### RUMI: Rummaging Using Mutual Information (https://arxiv.org/abs/2408.10450)
Comments:
          19 pages, 17 figures, submitted to IEEE Transactions on Robotics (T-RO)

- **What's New**: 본 논문은 Rummaging Using Mutual Information (RUMI)라는 방법을 제안하며, 이는 시각적으로 가려진 환경에서 알려진 이동 가능한 물체의 자세를 파악하기 위한 로봇 행동 시퀀스를 온라인으로 생성하는 방법입니다. RUMI는 로봇의 경로와 물체의 자세 분포 간의 상호 정보를 활용하여 행동 계획을 수립합니다.

- **Technical Details**: RUMI는 시뮬레이션된 포인트 클라우드로부터 호환 가능한 물체 자세 분포를 유도하고, 실제로 작업 공간 점유와의 상호 정보를 실시간으로 근사합니다. 정보를 활용한 비용 함수와 이동 가능성 비용 함수를 개발하였고, 이는 확률적 동역학 모델을 갖춘 모델 예측 제어(MPC) 프레임워크에 통합됩니다.

- **Performance Highlights**: RUMI는 여러 객체에 대한 시뮬레이션 및 실제 로봇 럼징 작업에서 베이스라인 방법들보다 우수한 성능을 보였습니다. 실험 결과, RUMI는 다양한 객체에서 일관된 성공을 달성하는 유일한 방법임을 입증하였습니다.



### The Brittleness of AI-Generated Image Watermarking Techniques: Examining Their Robustness Against Visual Paraphrasing Attacks (https://arxiv.org/abs/2408.10446)
Comments:
          23 pages and 10 figures

- **What's New**: 최근 이미지 생성 시스템의 급속한 발전으로 인해 AI가 생성한 이미지의 물리적 저작권 보호가 우려되고 있습니다. 이에 따라 기업들은 이미지에 워터마크(watermark) 기술을 적용하려고 노력하고 있으나, 기존의 방법들은 Visual Paraphrase 공격에 취약하다는 주장을 하고 있습니다.

- **Technical Details**: 이 논문에서 제안하는 Visual Paraphrase 공격은 두 단계로 이루어집니다. 첫 번째 단계에서는 KOSMOS-2라는 최신 이미지 캡셔닝 시스템을 사용하여 주어진 이미지에 대한 캡션을 생성합니다. 두 번째 단계에서는 원본 이미지와 생성된 캡션을 이미지 간 확산 시스템(image-to-image diffusion system)에 전달하여 디노이징 단계에서 텍스트 캡션에 의해 안내된 시각적으로 유사한 이미지를 생성합니다.

- **Performance Highlights**: 실험 결과, Visual Paraphrase 공격이 기존 이미지의 워터마크를 효과적으로 제거할 수 있음을 입증하였습니다. 이 논문은 기존의 워터마킹 기법들의 취약성을 비판적으로 평가하며, 보다 강력한 워터마킹 기술의 개발을 촉구하는 역할을 합니다. 또한, 연구자들을 위한 최초의 Visual Paraphrase 데이터셋과 코드를 공개하고 있습니다.



### Understanding Generative AI Content with Embedding Models (https://arxiv.org/abs/2408.10437)
- **What's New**: 이 논문에서는 고급 심층 신경망(DNN)의 내부 표현, 즉 embedding을 통한 자동화된 특성 공학을 제안합니다. 이들은 훈련된 DNN에서 비구조적인 샘플 데이터의 해석 가능한 고수준 개념을 드러낼 수 있다는 점이 주목할 만합니다.

- **Technical Details**: DNN은 데이터를 고차원 벡터 공간에 매핑하여 특징(feature) 표현을 생성합니다. 이 과정에서 PCA(Principal Component Analysis) 및 LDA(Linear Discriminant Analysis)를 통해 데이터를 분석하고, AI 생성 콘텐츠와 실제 데이터 간의 차이를 구별하는 데 사용됩니다.

- **Performance Highlights**: 실험 결과, 현대의 DNN 기반 feature embedder가 인간이 해석할 수 있는 특징을 추출할 수 있으며, AI 모델에 의해 생성된 콘텐츠와 실제 콘텐츠를 효과적으로 구분할 수 있음을 보여줍니다. 이는 모델 설명과 데이터 검증 분야에서의 새로운 가능성을 제시합니다.



### Are LLMs Any Good for High-Level Synthesis? (https://arxiv.org/abs/2408.10428)
Comments:
          ICCAD '24 Special Session on AI4HLS: New Frontiers in High-Level Synthesis Augmented with Artificial Intelligence

- **What's New**: 이 연구는 Large Language Models (LLMs)를 활용하여 High-Level Synthesis (HLS)의 프로세스를 간소화하고 최적화하는 새로운 방법론을 제안합니다. LLMs는 자연어 명세를 이해하고 코드 리팩토링을 수행할 수 있어 HLS 설계에서 혁신적인 접근 방식을 제공합니다.

- **Technical Details**: 본 논문에서는 LLMs의 HLS 적용 가능성을 조사하고, 이를 통한 Verilog 설계의 생산성을 평가합니다. LLMs는 자연어를 HLS-C 또는 Verilog로 변환할 수 있으며, 코드 생성을 통해 합성 가능한 HDL 생성에 도움을 줍니다. 또한, LLM이 하드웨어 검증을 위한 테스트 케이스 생성을 자동화할 수 있는 가능성도 제시합니다.

- **Performance Highlights**: LLMs를 사용한 HLS 접근 방식은 기존 HLS 도구와 비교하여 설계 품질(성능, 전력, 자원 소비)에서 경쟁력을 보여주며, 특히 설계 프로세스를 더욱 쉽게 접근 가능하게 만들어 주는 장점이 있습니다. HLS 설계에서 LLM의 역할과 이점은 향후 하드웨어 개발의 흐름을 변화시킬 잠재력을 가지고 있습니다.



### Towards Automation of Human Stage of Decay Identification: An Artificial Intelligence Approach (https://arxiv.org/abs/2408.10414)
Comments:
          13 pages

- **What's New**: 이 연구는 인공지능(AI)을 활용하여 인간 부패 사진의 스테이지 오브 디컴포지션(SOD) 분류를 자동화하는 가능성을 탐구합니다. 현재 수작업으로 진행되는 SOD 평가법의 주관성과 비효율성을 해결하고자 합니다.

- **Technical Details**: 이 연구는 Megyesi와 Gelderman이 제안한 두 가지 SOD 평가 방법을 AI로 자동화하는 것을 목표로 하며, Inception V3 및 Xception과 같은 심층 학습(deep learning) 모델을 대규모 인간 부패 이미지 데이터셋에서 훈련시켜 다양한 해부학적 영역에 대한 SOD를 분류합니다. 이 AI 모델의 정확도는 사람의 법의학 전문가와 비교하여 검사되었습니다.

- **Performance Highlights**: Xception 모델은 Megyesi 방식에서 머리, 몸통, 사지 각각에 대해 매크로 평균 F1 점수 0.878, 0.881, 0.702를 달성하였으며, Gelderman 방식에서도 각각 0.872, 0.875, 0.76을 기록했습니다. AI 모델의 신뢰성은 인간 전문가의 수준에 도달하며, 이 연구는 AI를 통한 SOD 식별 자동화의 가능성을 보여줍니다.



### Webcam-based Pupil Diameter Prediction Benefits from Upscaling (https://arxiv.org/abs/2408.10397)
- **What's New**: 본 연구는 저해상도 webcam 이미지에서 동공 지름을 예측하는 여러 업스케일링 방법의 영향을 평가합니다. 특히, 기존의 다소 한정적인 이미지 데이터셋과는 달리 EyeDentify 데이터셋을 활용하여 다양한 조건에서의 이미지 품질 개선이 동공 검사에 미치는 효과를 분석합니다.

- **Technical Details**: 여러 업스케일링 기술을 비교 분석하면서 bicubic interpolation에서 시작하여 CodeFormer, GFPGAN, Real-ESRGAN, HAT, SRResNet과 같은 고급 super-resolution 방법까지 포함합니다. 연구 결과, 이미지를 업스케일링하는 방법과 비율에 따라 동공 지름 예측 성능이 크게 달라진다는 것을 발견하였습니다.

- **Performance Highlights**: 고급 SR 모델을 활용할 때 동공 지름 예측 모델의 정확성이 현저히 향상됨을 나타냅니다. 특히 업스케일링 기법을 사용하는 것이 전반적으로 동공 지름 예측의 성능을 개선하는 것으로 나타났습니다. 이는 심리적 및 생리적 연구에서 보다 정확한 평가를 용이하게 합니다.



### Evaluating Image-Based Face and Eye Tracking with Event Cameras (https://arxiv.org/abs/2408.10395)
Comments:
          This paper has been accepted at The Workshop On Neuromorphic Vision: Advantages and Applications of Event Cameras at the European Conference on Computer Vision (ECCV), 2024

- **What's New**: 이 논문은 이벤트 카메라(Event Cameras) 데이터를 활용하여 전통적인 알고리즘과의 통합 가능성을 보여줍니다. 특히, 이벤트 기반 데이터로부터 프레임 포맷을 생성하여 얼굴 및 눈 추적 작업에서 효용성을 평가합니다.

- **Technical Details**: 본 연구는 Helen Dataset의 RGB 프레임을 기반으로 이벤트를 시뮬레이션하여 프레임-기반 이벤트 데이터셋을 구축하였습니다. 또한, GR-YOLO라는 새로운 기술을 활용하여 YOLOv3에서 발전된 얼굴 및 눈 검출 성능을 평가하며, YOLOv8과의 비교 분석을 통해 결과의 효용성을 확인하였습니다.

- **Performance Highlights**: 모델은 다양한 데이터셋에서 평균 정밀도(mean Average Precision) 점수 0.91을 기록하며 좋은 예측 성능을 보였습니다. 또한, 변화하는 조명 조건에서도 실시간 이벤트 카메라 데이터에 대해 강건한 성능을 나타냈습니다.



### Joint Modeling of Search and Recommendations Via an Unified Contextual Recommender (UniCoRn) (https://arxiv.org/abs/2408.10394)
Comments:
          3 pages, 1 figure

- **What's New**: 이번 연구에서는 검색(Search) 및 추천(Recommendation) 시스템을 통합할 수 있는 딥러닝(deep learning) 모델을 제안합니다. 이는 별도의 모델 개발로 인한 유지보수의 복잡성을 줄이고 기술 부채(technical debt)를 완화하는 데 기여합니다.

- **Technical Details**: 제안된 모델은 사용자 ID, 쿼리(query), 국가(country), 소스 엔티티 ID(source entity id), 작업(task)과 같은 정보를 기반으로합니다. 이 모델은 입력층에 다양한 컨텍스트를 통합하여 다양한 상황에 따라 그에 맞는 결과를 생성할 수 있습니다. 구조적으로는 잔여 연결(residual connections)과 특징 교차(feature crossing)를 포함하며, 이진 교차 엔트로피 손실(binary cross entropy loss)과 아담 최적화기(Adam optimizer)를 사용합니다.

- **Performance Highlights**: UniCoRn이라는 통합 모델을 통해 검색과 추천 성능 모두에서 파리티(parity) 또는 성능 향상을 달성했습니다. 비개인화된 모델에서 개인화된 모델로 전환했을 때 검색과 추천 각각 7% 및 10%의 실적 향상을 보였습니다.



### BrewCLIP: A Bifurcated Representation Learning Framework for Audio-Visual Retrieva (https://arxiv.org/abs/2408.10383)
- **What's New**: 이 논문에서는 오디오-이미지 매칭을 위한 새로운 모델, BrewCLIP을 제안합니다. BrewCLIP은 기존의 파이프라인 모델과 End-to-End 모델의 한계를 극복하고, 텍스트 정보 외에도 음성과 관련된 비텍스트 정보(예: 감정)를 효과적으로 캡처하여 성능을 향상시키기 위해 설계되었습니다. 이 모델은 Whisper와 CLIP을 기반으로 하며, 두 채널 정보를 결합하여 보다 견고한 오디오-이미지 검색을 가능하게 합니다.

- **Technical Details**: BrewCLIP 모델은 Whisper(음성 인식 모델)와 CLIP(텍스트 및 이미지 인코더)를 결합한 수정된 파이프라인 모델입니다. 우리는 End-to-End 오디오 인코더를 추가하여 서로 다른 모델 간의 상호작용 효과를 연구했습니다. 실험은 스크립트 기반 및 비스크립트 기반 데이터셋을 포함하여 다양한 데이터셋에서 수행되었으며, 프롬프트 파인튜닝(prompt finetuning)의 영향을 분석했습니다.

- **Performance Highlights**: BrewCLIP 모델은 이미지-텍스트 매칭에서 이전의 최첨단 성능을 초월하였으며, 프롬프트 공유를 통해 동결된 모델의 성능을 상당히 향상시켰습니다. BrewCLIP의 End-to-End 오디오 인코더는 음성 감정 인식(SER) 작업에 적용하여 음성의 감정 정보를 성공적으로 학습할 수 있음을 보여주었습니다.



### Efficient Reinforcement Learning in Probabilistic Reward Machines (https://arxiv.org/abs/2408.10381)
Comments:
          33 pages, 4 figures

- **What's New**: 이 연구에서는 Probabilistic Reward Machines (PRMs)을 활용한 Markov Decision Processes에서 강화 학습을 다룹니다. PRM은 로봇 작업에서 자주 발생하는 비마르코프 보상을 모델링하는 새로운 방법입니다. 본 연구의 알고리즘은 PRM에 대한 첫 번째 효율적인 알고리즘으로, 기댓값 손실을 최소화하는 규명을 제공합니다.

- **Technical Details**: 제안된 알고리즘 UCBVI-PRM은 비마르코프 보상을 대상으로 한 강화 학습에서의 성능을 개선하며, O~(√(HOAT))의 기댓값 손실 경계에 도달합니다. H는 시간 수평선, O는 관찰 수, A는 행동 수, T는 시간 단계 수를 나타냅니다. 비마르코프 보상에 대한 새로운 시뮬레이션 레마를 제시하여 임의의 비마르코프 보상으로 데이터 수집을 통한 최적 정책 학습을 지원합니다.

- **Performance Highlights**: UCBVI-PRM은 다양한 PRM 환경에서 기존 방법들보다 우수한 성능을 보였으며, 이론적 및 실험적 분석을 통해 그 효율성을 입증했습니다.



### Boolean Matrix Logic Programming (https://arxiv.org/abs/2408.10369)
- **What's New**: 이 논문에서는 효율적이고 조합 가능한 불리언 행렬 조작 모듈을 기반으로 한 datalog 쿼리 평가 방법을 제안합니다. 새로운 문제인 Boolean Matrix Logic Programming (BMLP)을 정의하고, 선형 이항 재귀 datalog 프로그램에 대한 하향식 추론을 위한 두 가지 새로운 BMLP 모듈을 개발했습니다.

- **Technical Details**: BMLP는 불리언 행렬을 사용하여 datalog 프로그램을 평가하는 일반적인 쿼리 응답 문제로, 최대 두 개의 술어를 포함하는 절이 있는 datalog 프로그램에 초점을 맞춥니다. 두 가지 모듈(BMLP-RMS, BMLP-SMP)은 SWI-Prolog에서 구현되었으며, 이러한 모듈들이 선형 및 비선형 재귀 datalog 프로그램 뿐만 아니라 다중 선형 프로그램에 대해 조합 가능함을 이론적으로 증명했습니다.

- **Performance Highlights**: 대규모 프로그램을 평가할 때 BMLP 모듈은 기존의 datalog 엔진인 Souffle, ASP 해결기 Clingo, 일반 목적 Prolog 시스템 B-Prolog 및 SWI-Prolog에 비해 각각 30배 및 9배의 성능 향상을 보였습니다.



### HaSPeR: An Image Repository for Hand Shadow Puppet Recognition (https://arxiv.org/abs/2408.10360)
Comments:
          Submitted to IEEE Transactions on Artificial Intelligence (IEEE TAI), 11 pages, 78 figures, 2 tables

- **What's New**: 이 논문에서는 손 그림자 인형극(Hand Shadow Puppetry)의 예술을 보존하기 위한 새로운 데이터셋 HaSPeR (Hand Shadow Puppet Image Repository)를 소개합니다. 이 데이터셋은 전문 및 아마추어 인형극 비디오에서 추출한 총 8,340개의 이미지를 포함하고 있으며, 그림자 인형극에 대한 인공지능 (AI) 연구를 위한 기초 자료로 제공됩니다.

- **Technical Details**: HaSPeR 데이터셋은 이미지 분류 모델의 성능을 평가하기 위해 다양한 사전 학습된 이미지 분류 모델을 사용하여 검증되었습니다. 연구 결과, 전통적인 convolutional 모델들이 attention 기반 transformer 아키텍처보다 성능이 우수한 것으로 나타났습니다. 특히 MobileNetV2와 같은 경량 모델이 모바일 애플리케이션에 적합하며, 사용자에게 유용한 şəkildə 작동합니다. InceptionV3 모델의 특징 표현, 설명 가능성 및 분류 오류에 대한 Thorough 분석이 수행되었습니다.

- **Performance Highlights**: 논문에서 제시된 데이터셋은 예술 영역에 대한 탐구 및 분석의 기회를 제공하여 독창적인 ombromanie 교육 도구 개발에 기여할 수 있을 것으로 기대됩니다. 또한, 코드와 데이터는 공개적으로 제공되어 연구자들이 접근할 수 있도록 되어 있습니다.



### The Psychological Impacts of Algorithmic and AI-Driven Social Media on Teenagers: A Call to Action (https://arxiv.org/abs/2408.10351)
Comments:
          7 pages, 0 figures, 2 tables, 2024 IEEE Conference on Digital Platforms and Societal Harms

- **What's New**: 이 연구는 소셜 미디어의 메타 이슈를 다루며, 개인 경험 및 사건 공유를 통한 사회적 상호 작용 증진이라는 이론적 목표에도 불구하고, 정서적 피해를 가져오는 역설적 결과를 밝혔습니다. 특히 청소년들 사이에서 이 현상이 두드러집니다.

- **Technical Details**: 청소년이 상처받을 수 있는 요소로는 개인화된 추천, 이상적인 디지털 이미지를 제시하려는 동료 압박, 알림과 업데이트의 지속적인 bombardment(폭격)가 포함됩니다. 현재 소셜 미디어는 감정적 안정성을 위한 연구와 사용자 보호를 위한 방안이 절실히 요구됩니다.

- **Performance Highlights**: 소셜 미디어가 청소년들에게 미치는 심리적 영향에 대한 자문을 제시하며, 특히 13세에서 17세 사이의 젊은 사용자들에서 불안 및 우울 증상과 강한 연관성을 나타냈습니다. 이 연구는 접근할 수 있는 대안적인 소셜 미디어 모델에 대한 이점과 한계를 탐구하고 있습니다.



### Decoding Human Emotions: Analyzing Multi-Channel EEG Data using LSTM Networks (https://arxiv.org/abs/2408.10328)
Comments:
          13 pages, 3 figures; accepted at ICDSA '24 Conference, Jaipur, India

- **What's New**: 이번 연구는 Electroencephalogram (EEG) 신호를 활용하여 감정 상태 분류의 예측 정확도를 향상시키기 위해 Long Short-Term Memory (LSTM) 네트워크를 적용한 방법론을 제시합니다. DEAP 데이터세트를 기반으로 EEG 신호의 시간적 의존성을 처리하여 감정의 벨런스(valence), 각성(arousal), 지배(dominance), 유사성(likeness)을 효과적으로 분류하고자 하였습니다.

- **Technical Details**: 연구에서 사용된 LSTM 네트워크는 EEG 데이터의 시간 종속적 특성을 모델링할 수 있는 강력한 도구로서, 고차원의 EEG 신호를 효과적으로 분석합니다. 본 연구는 arousal, valence, dominance 및 likeness에 대해 각각 89.89%, 90.33%, 90.70%, 90.54%의 분류 정확도를 기록하였습니다.

- **Performance Highlights**: 본 연구의 결과는 기존 감정 인식 모델에 비해 상당한 성능 개선을 보여주며, LSTM 기반 접근법이 EEG 신호로부터 감정 상태를 효과적으로 식별할 수 있음을 강조합니다. 또한, 환자 관리 및 특수 교육과 같은 다양한 응용 분야에서 활용 가능성이 큽니다.



### Leveraging Superfluous Information in Contrastive Representation Learning (https://arxiv.org/abs/2408.10292)
- **What's New**: 이번 논문에서는 기존 대비 보다 강건한 표현 학습을 위한 새로운 목표 함수인 SuperInfo를 제안합니다. SuperInfo는 예측 정보와 불필요한(superfluous) 정보를 선형 조합하여 학습하며, 이는 다양한 downstream tasks(다운스트림 작업)에서 성능 향상에 기여합니다.

- **Technical Details**: SuperInfo 목표 함수는 두 개의 증강(augmentation) 뷰 간의 상호 정보(mutual information) 최대화를 통해 학습된 표현에서 작업 관련(task-relevant) 및 작업 비관련(task-irrelevant) 정보를 분리합니다. 또한, Bayes Error Rate 분석을 통해 여러 표현의 성능을 평가합니다.

- **Performance Highlights**: 제안된 SuperInfo 손실을 활용한 학습은 이미지 분류, 객체 탐지(object detection), 인스턴스 분할(instance segmentation) 작업에 대해 전통적인 대비 표현 학습(constrastive representation learning) 방법을 초월하는 성능을 보이며, 상당한 개선을 보여줍니다.



### Recognizing Beam Profiles from Silicon Photonics Gratings using Transformer Mod (https://arxiv.org/abs/2408.10287)
- **What's New**: 이번 연구는 실리콘 포토닉스(SiPh) 격자에서 발생하는 광 빔 프로파일의 높이 범주를 인식하기 위한 트랜스포머(transformer) 모델을 개발하였습니다. 이는 이온 트랩 양자 컴퓨팅 커뮤니티에서 쿼비트를 광학적으로 다루기 위한 기술입니다.

- **Technical Details**: 모델은 두 가지 기술을 사용하여 훈련되었으며, (1) 입력 패치(input patches), (2) 입력 시퀀스(input sequence) 방식입니다. 입력 패치로 훈련된 모델은 0.938의 인식 정확도를 달성하였고, 입력 시퀀스로 훈련된 모델은 0.895의 정확도를 보였습니다. 150회의 모델 훈련 반복 시 입력 패치 모델은 0.445에서 0.959로 변동하는 일관되지 않은 정확도를 보였으며, 입력 시퀀스 모델은 0.789에서 0.936으로 높은 정확도를 기록했습니다.

- **Performance Highlights**: 이 연구의 결과는 빛의 초점 자동 조정(auto-focusing of light beam) 및 원하는 빔 프로파일을 얻기 위한 z-축 스테이지의 자동 조정(auto-adjustment of z-axis stage) 등 다양한 응용 분야로 확대될 수 있습니다.



### GPT-Augmented Reinforcement Learning with Intelligent Control for Vehicle Dispatching (https://arxiv.org/abs/2408.10286)
- **What's New**: 이번 논문에서는 도시 내 이동 품질 향상을 위한 새로운 차량 배차 시스템인 GARLIC를 소개합니다. GARLIC는 도시 교통의 복잡성을 관리하기 위해 설계되었습니다.

- **Technical Details**: GARLIC는 다중 뷰 그래프(multiview graphs)를 활용하여 계층적 교통 상황을 파악하고, 개별 운전 행동을 반영하는 동적 보상 함수(dynamic reward function)를 학습합니다. 또한, 맞춤 손실 함수(custom loss function)로 훈련된 GPT 모델을 통합하여 실제 상황에서 고정밀 예측을 가능하게 하고 배차 정책을 최적화합니다.

- **Performance Highlights**: 두 개의 실제 데이터셋을 활용한 실험을 통해 GARLIC가 운전자의 행동에 효과적으로 맞춰지며, 차량의 공차율(empty load rate)을 줄이는 데 성공했음을 나타냈습니다.



### BatGPT-Chem: A Foundation Large Model For Retrosynthesis Prediction (https://arxiv.org/abs/2408.10285)
- **What's New**: BatGPT-Chem은 150억 개의 매개변수를 갖춘 새로운 대형 언어 모델로, 향상된 retrosynthesis 예측을 위해 개발되었습니다. 이 모델은 자연어 처리 및 SMILES 표기법을 통합하여 화학 실험을 보다 효율적으로 수행할 수 있도록 합니다.

- **Technical Details**: BatGPT-Chem은 1억 개 이상의 사례를 사용하여 autoregressive 및 bidirectional 훈련 기법을 적용하여 개발되었습니다. 이를 통해 다양한 화학 지식을 포괄하며, 반응 조건을 정확하게 예측할 수 있는 능력을 갖추고 있습니다. 이 과정에서 새로운 반응 조건을 명시적으로 포함, 일반화된 예측 기능을 극대화합니다.

- **Performance Highlights**: BatGPT-Chem은 기존 AI 방법보다 유리하며, 복잡한 분자의 효과적인 전략 생성을 통해 주요 벤치마크 테스트를 통과했습니다. 이 모델은 제로샷(zero-shot) 조건에서도 뛰어난 성능을 보여주며, 화학 공학 분야에서 LLMs의 응용을 위한 새로운 기준을 마련하였습니다.



### FEDKIM: Adaptive Federated Knowledge Injection into Medical Foundation Models (https://arxiv.org/abs/2408.10276)
Comments:
          Submitted to EMNLP'24

- **What's New**: 이번 연구에서는 의료 분야에서 연합 학습( federated learning ) 프레임워크를 사용하여 의료 기반 모델을 확장하기 위한 새로운 지식 주입 방법인 FedKIM을 소개합니다. 이는 의료 데이터를 개인적으로 다루면서 모델의 성능을 향상시키는 혁신적인 접근입니다.

- **Technical Details**: FedKIM은 가벼운 로컬 모델을 활용하여 개인 데이터를 통해 헬스케어 지식을 추출하고, 이를 중앙 집중화된 기반 모델에 통합합니다. 과정에서 M3OE( Multitask Multimodal Mixture Of Experts ) 모듈을 사용하여 다양한 의료 작업을 처리할 수 있는 적응형 시스템을 구축합니다.

- **Performance Highlights**: 12개의 작업에서 7개 모달리티를 기반으로 실시한 광범위한 실험 결과, FedKIM이 다양한 환경에서 효과적임을 입증했습니다. 이는 민감한 데이터에 직접적으로 접근하지 않고도 의료 기반 모델을 확장할 수 있는 잠재력을 보여줍니다.



### FedKBP: Federated dose prediction framework for knowledge-based planning in radiation therapy (https://arxiv.org/abs/2408.10275)
Comments:
          Under review by SPIE Medical Imaging 2025 Conference

- **What's New**: 본 논문에서는 지식 기반 계획(KBP)에서 환자 맞춤형 선량 분포를 자동으로 생성하는 선량 예측의 중요성을 강조합니다. 최근 딥 러닝(Deep Learning) 기반의 선량 예측 방법들이 발전하면서, 데이터 기여자 간의 협업이 필요해졌습니다. 이에 연합 학습(Federated Learning, FL) 기법이 제시되어 환자 데이터의 프라이버시를 존중하면서도 의료 센터들이 공동으로 딥러닝 모델을 학습할 수 있는 해결책이 되었습니다.

- **Technical Details**: 연구에서는 FedKBP 프레임워크를 개발하여 OpenKBP 데이터셋의 340개 계획에 대한 선량 예측 모델의 중앙 집중식, 연합 및 개별 훈련 성능을 평가했습니다. FL 및 개별 훈련을 시뮬레이션하기 위해 데이터를 8개의 훈련 사이트로 나누었습니다. 데이터 변동성이 모델 훈련에 미치는 영향을 평가하기 위해 두 가지 유형의 사례 분포를 구현했습니다: 1) 독립적이고 동일하게 분포된 것(IID)과 2) 비 독립적이고 차별적으로 분포된 것(non-IID).

- **Performance Highlights**: 실험 결과, FL은 개인 훈련에 비해 모델 최적화 속도와 외부 샘플 테스트 점수 모두에서 일관되게 우수한 성능을 보였습니다. IID 데이터에서 FL은 중앙 집중식 훈련과 유사한 성능을 보였으며, 이는 FL이 전통적인 통합 데이터 훈련에 대한 유망한 대안임을 강조합니다. 반면 비 IID 환경에서는 큰 사이트가 작은 사이트보다 최대 19% 더 높은 테스트 점수를 기록하며, 데이터 소유자 간의 협력이 필요함을 확인했습니다. 또한, 비 IID FL은 IID FL에 비해 성능이 저하되어 데이터 변동 처리를 위한 보다 정교한 FL 방법의 필요성이 대두되었습니다.



### SEAL: Systematic Error Analysis for Value ALignmen (https://arxiv.org/abs/2408.10270)
Comments:
          28 pages, 17 Figures, 8 Tables

- **What's New**: 이 연구에서는 인간의 피드백(Feedback)을 기반으로 강화 학습(RLHF)의 내부 메커니즘을 이해하기 위해 새로운 메트릭을 도입했습니다. 이 메트릭은 feature imprint, alignment resistance, alignment robustness로 구성되어 있습니다.

- **Technical Details**: RLHF는 언어 모델(LM)을 인간의 가치에 맞춰 조정하기 위해 보상 모델(RM)을 훈련하는 과정입니다. 연구에서는 alignment dataset을 target features(원하는 가치)와 spoiler features(원치 않는 개념)로 분류하고, RM 점수를 이들 특성과 회귀 분석하여 feature imprint를 정량화했습니다.

- **Performance Highlights**: 연구 결과, 목표 feature에 대한 RM의 보상이 큼을 발견하였고, 26%의 경우에서 RM이 인간의 선호와 일치하지 않았습니다. 또한, RM은 입력이 약간 변경되었을 때 민감하게 반응하여, misalignment가 악화되는 모습을 보였습니다.



### OpenCity: Open Spatio-Temporal Foundation Models for Traffic Prediction (https://arxiv.org/abs/2408.10269)
Comments:
          12 pages

- **What's New**: 이 논문에서는 도시 교통 예측을 위한 새로운 기초 모델인 OpenCity를 소개합니다. 이 모델은 다양한 데이터 특성을 통해 기초적인 공간-시간 패턴을 효과적으로 포착하고 정규화하여, 이전에 보지 못한 지역에서도 제로샷(zero-shot) 일반화를 가능하게 합니다.

- **Technical Details**: OpenCity는 Transformer 아키텍처와 그래프 신경망(graph neural networks)을 통합하여 복잡한 공간-시간 종속성을 모델링합니다. 이 모델은 대규모의 이질적인 교통 데이터를 기반으로 사전 훈련(pre-training)을 통해 풍부하고 일반화 가능한 표현을 학습할 수 있습니다.

- **Performance Highlights**: 실험 결과 OpenCity는 뛰어난 제로샷 예측 성능을 보여주었으며, 새로운 도시 환경에 최소한의 부가 비용으로 적응할 수 있는 가능성을 제시하고 있습니다. 또한, OpenCity는 다양한 예측 작업에 대한 빠른 문맥 적응 능력을 가지고 있습니다.



### Realtime Generation of Streamliners with Large Language Models (https://arxiv.org/abs/2408.10268)
- **What's New**: 이 논문에서는 Large Language Models (LLMs)을 활용하여 constraint programming에서 streamliners를 생성하기 위한 새로운 방법인 StreamLLM을 제안합니다. Streamliner는 검색 공간을 좁혀 복잡한 문제의 해결 속도 및 가능성을 향상시키기 때문에 전통적으로 수작업으로 생성되었으나, 본 연구는 LLM을 이용해 효과적인 streamliners를 제안합니다.

- **Technical Details**: StreamLLM은 MiniZinc 제약 프로그래밍 언어로 지정된 문제에 대해 streamliners를 생성하며, 신속한 경험적 테스트를 통해 LLM에 피드백을 통합합니다. 최소한 245 CPU 일의 분량으로 10가지 문제에 대해 철저한 실험적 평가를 수행하였으며, 전통적인 생성 방식보다 LLM을 활용한 접근이 더 유용하다는 결과를 보였습니다.

- **Performance Highlights**: 우리는 StreamLLM이 몇 가지 경우에서 99% 이상의 해결 시간을 단축할 수 있음을 발견했습니다. 두 가지 최첨단 LLM(GPT-4o와 Claude 3.5 Sonnet)과 다양한 프롬프팅 변형을 활용하여, 생성된 streamliners의 성능을 평가하였으며, 접근 방식의 강건함을 검증했습니다. 그 결과, LLM이 생성한 streamliner가 다양한 제약 만족 문제에서 상당한 런타임 감소를 이끌어 낼 수 있음을 확인했습니다.



### Diffusion Model for Planning: A Systematic Literature Review (https://arxiv.org/abs/2408.10266)
Comments:
          13 pages, 2 figures, 4 tables

- **What's New**: 본 논문은 최근 확산 모델(Diffusion Models)이 계획(planning) 작업에 효과적으로 적용되고 있음을 보여주며, 2023년 이후 관련 출판물의 급증을 설명합니다.

- **Technical Details**: 논문에서는 다음과 같은 관점에서 기존 문헌을 분류하고 논의합니다: (i) 확산 모델 기반 계획을 평가하는 데 사용되는 관련 데이터셋(data sets) 및 기준(benchmarks); (ii) 샘플링 효율성(sampling efficiency)과 관련된 기초 연구; (iii) 적응성을 향상시키기 위한 기술 중심(skill-centric) 및 조건 유도(condition-guided) 계획; (iv) 안전성(safety) 및 불확실성 관리 메커니즘(uncertainty managing mechanism)으로 안전성과 강인성(enhancing safety and robustness) 향상; (v) 자율주행(autonomous driving)과 같은 도메인 특정(domain-specific) 응용.

- **Performance Highlights**: 확산 모델이 다양한 계획 작업의 성능을 향상시키는 것을 입증하며, 해당 분야의 도전 과제와 미래 방향에 대해 논의합니다.



### OPDR: Order-Preserving Dimension Reduction for Semantic Embedding of Multimodal Scientific Data (https://arxiv.org/abs/2408.10264)
- **What's New**: 이번 논문은 다중 모드 과학 데이터 관리에서 가장 일반적인 작업 중 하나인 k-최근접 이웃(KNN) 검색의 차원을 줄이는 방법을 제안합니다. 본 연구는 기존의 고차원 임베딩 벡터를 낮은 차원으로 변환하면서 KNN 검색의 결과를 유지하는 Order-Preserving Dimension Reduction (OPDR) 기법을 도입하였습니다.

- **Technical Details**: OPDR은 임베딩 벡터의 차원을 줄이는 데 있어 KNN 유사성을 정량화하는 수식적 접근 방식을 사용합니다. 이 방법은 KNN 검색을 위한 측정 함수를 정의하고, 이를 통해 전역 메트릭스(global metric)와 닫힌 형태의 함수를 도출하여 차원의 수와 데이터 포인트 수 간의 관계를 밝혀냅니다. 또한 다양한 차원 축소 방법(PCA, MDS) 및 거리 메트릭(Euclidean, cosine, Manhattan)과 통합하여 실제 과학 응용 프로그램에 쉽게 적용될 수 있도록 설계되었습니다.

- **Performance Highlights**: 제안된 OPDR 방법은 4개의 대표적인 다중 모드 과학 데이터 세트에 대해 광범위한 평가를 통해 효과성을 입증하였으며, 이 결과는 높은 차원에서 저차원으로의 변환 시 KNN의 보존을 정확하게 캡처할 수 있음을 보여줍니다. 이로 인해 복잡한 과학 응용 분야에서 고급 AI 기술을 활용하는 데 새로운 통찰을 제공합니다.



### Relational Graph Convolutional Networks Do Not Learn Sound Rules (https://arxiv.org/abs/2408.10261)
Comments:
          Full version (with appendices) of paper accepted to KR 2024 (21st International Conference on Principles of Knowledge Representation and Reasoning)

- **What's New**: 본 논문은 R-GCN(GNN 아키텍처)의 예측을 설명하기 위한 두 가지 규칙 추출 방법을 제안합니다. 이로써 R-GCN의 출력에 대한 설명 가능성을 높이고, 특정 Datalog 규칙이 R-GCN에 대해 유효하지 않음을 검증하는 방법도 제공합니다.

- **Technical Details**: R-GCN을 사용하여 KG 완성을 학습하고, 학습된 모델에서 출력 채널 중 일부가 단조로운(non-monotonic) 특성을 가질 수 있음을 확인합니다. 이는 특정 Datalog 규칙을 통해 설명 가능한 예측을 생성하는 데 필수적입니다.

- **Performance Highlights**: 실험 결과, R-GCN 모델이 높은 정확도를 유지하더라도, Datalog 규칙이 유효하지 않음을 확인하였으며, 모델 성능과 규칙 추출 간의 절충을 통해 더 많은 유효한 규칙을 얻을 수 있는 방법을 제시하였습니다.



### Optical Music Recognition in Manuscripts from the Ricordi Archiv (https://arxiv.org/abs/2408.10260)
Comments:
          Accepted at AudioMostly 2024

- **What's New**: 리코르디 아카이브(Ricordi Archive)의 디지털화가 완료되어, 이탈리아 오페라 작곡가들의 수많은 음악 원고에서 음악 요소를 자동으로 추출하고 이를 분류할 수 있는 신경망 기반의 분류기가 개발되었습니다.

- **Technical Details**: 이 연구는 Optical Music Recognition (OMR) 방법론을 통해 음악 기호를 자동으로 식별하는 시스템을 구축하였습니다. 특히, Convolutional Recurrent Neural Networks를 활용하여 필기 음표의 복잡한 특성을 다루었습니다.

- **Performance Highlights**: 다양한 신경망 분류기를 훈련하여 음악 요소를 구분하는 실험을 수행하였으며, 이 결과들은 향후 나머지 리코르디 아카이브의 자동 주석 작업에 활용될 것으로 기대됩니다.



### Contrastive Learning on Medical Intents for Sequential Prescription Recommendation (https://arxiv.org/abs/2408.10259)
Comments:
          Accepted to the 33rd ACM International Conference on Information and Knowledge Management (CIKM 2024)

- **What's New**: 이 연구는 Electronic Health Records (EHR)에 적용된 seqential modeling의 진전을 바탕으로 처방 추천 시스템에 대한 새로운 접근을 제안합니다. 특히, 이를 위한 다중 레벨 transformer 기반의 방법인 Attentive Recommendation with Contrasted Intents (ARCI)를 도입하여 환자마다 다양한 건강 프로필에 따른 임상 의도를 반영합니다.

- **Technical Details**: ARCI는 중첩된 temporal relationships를 모델링하기 위해 contrastaive learning을 사용하여, 각 환자에 대한 여러 distinct medical intents를 추출합니다. 이 방법은 다양한 방문을 통해 수집된 의료 코드에서 복잡한 관계를 disentangle하는데 중점을 두며, inter-visit 및 intra-visit medication dependencies를 포착합니다. 각각의 intent는 transformer attention-head에 연결되어, 다양한 temporal paths와 전문가 프로필을 정규화합니다.

- **Performance Highlights**: 우리의 실험은 MIMIC-III 및 Acute Kidney Injury (AKI) 데이터 세트를 사용하여 진행하였으며, ARCI가 기존의 최첨단 추천 방법들보다 뛰어난 성능을 보여주었습니다. 또한, 제안한 방법은 의료 실무자들에게 유용한 해석 가능한 인사이트를 제공합니다.



### Large Investment Mod (https://arxiv.org/abs/2408.10255)
Comments:
          20 pages, 10 figures, 2 tables

- **What's New**: 본 논문에서는 기존의 정량적 투자 연구에서의 수익 감소와 시간 및 인력 비용 증가 문제를 해결하기 위해 대형 투자 모델(LIM)을 제안합니다. LIM은 end-to-end learning(전체 과정 학습)과 universal modeling(보편적 모델링)을 활용하여 다양한 금융 데이터에서 신호 패턴을 독립적으로 학습할 수 있는 기반 모델을 만듭니다.

- **Technical Details**: LIM은 여러 거래소, 도구, 주파수에 걸친 다양한 금융 데이터를 분석하여 'global patterns'(전 세계 패턴)을 학습하고, 이를 downstream strategy modeling(하류 전략 모델링)에 전달하여 특정 과제에 대한 성능을 최적화합니다. 이러한 접근 방식은 energy-efficient(에너지 효율적인) 모델링을 가능하게 하고, 기존의 전형적인 다요인 모델링(multi-factor modeling)에서 벗어나 직접적인 거래 전략 생성을 목표로 합니다.

- **Performance Highlights**: LIM의 장점은 상품 선물 거래에 대한 교차 도구 예측에 대한 수치 실험을 통해 입증되었습니다. 전반적으로 LIM은 정량적 연구의 효율성과 효과성을 높이고, 여러 전략 과제에 대한 모델의 적용 가능성을 향상시킵니다.



### Balancing Innovation and Ethics in AI-Driven Software Developmen (https://arxiv.org/abs/2408.10252)
Comments:
          20 Pages

- **What's New**: 이 논문은 GitHub Copilot과 ChatGPT와 같은 AI 도구들이 소프트웨어 개발 과정에 통합될 때의 윤리적 함의에 대해 비판적으로 검토합니다.

- **Technical Details**: 논문은 코드 소유권(code ownership), 편향(bias), 책임(accountability), 개인 정보 보호(privacy), 그리고 일자리 시장(job market)에 미치는 잠재적 영향에 대한 문제를 탐구합니다.

- **Performance Highlights**: AI 도구들은 생산성(productivity)과 효율성(efficiency) 측면에서 큰 이점을 제공하지만, 그에 따른 복잡한 윤리적 도전 과제를 동반합니다. 이러한 도전 과제를 해결하는 것이 AI의 소프트웨어 개발 통합이 책임감 있고 사회에 유익하도록 보장하는 데 필수적이라고 주장합니다.



### Target-Dependent Multimodal Sentiment Analysis Via Employing Visual-to Emotional-Caption Translation Network using Visual-Caption Pairs (https://arxiv.org/abs/2408.10248)
- **What's New**: 이번 연구에서는 Target-Dependent Multimodal Sentiment Analysis (TDMSA) 방법론을 사용하여 멀티모달 포스트에서 각 타겟(측면)과 관련된 감정을 효과적으로 식별하는 새로운 접근 방식을 제시합니다.

- **Technical Details**: 우리는 Visual-to-Emotional-Caption Translation Network (VECTN)이라는 새로운 기술을 개발하였으며, 이를 통해 얼굴 표정에서 시각적 감정 단서를 성공적으로 추출하고 이를 텍스트의 타겟 속성과 정렬 및 통합할 수 있습니다.

- **Performance Highlights**: 실험 결과, Twitter-2015 데이터셋에서 81.23%의 정확도와 80.61%의 macro-F1 점수를, Twitter-2017 데이터셋에서는 77.42%의 정확도와 75.19%의 macro-F1 점수를 달성하였습니다. 이를 통해 우리의 모델이 멀티모달 데이터에서 타겟 수준의 감정을 수집하는 데 있어 다른 모델보다 우수함을 보여주었습니다.



### MetaEnzyme: Meta Pan-Enzyme Learning for Task-Adaptive Redesign (https://arxiv.org/abs/2408.10247)
Comments:
          Accepted to ACM Multimedia 2024

- **What's New**: MetaEnzyme은 효소 설계를 위한 통합 프레임워크로, 저자들은 기능적 설계, 돌연변이 설계 및 서열 생성 설계 등 세 가지 저자원(저자원, low-resource) 효소 재설계 작업을 수행하는 데 중점을 두고, 이를 통해 시스템적 연구를 촉진하고자 합니다.

- **Technical Details**: MetaEnzyme은 구조-서열 변환 아키텍처와 도메인 적응 기술을 활용하여 효소 설계 작업을 일반화합니다. UniProt-Net이라는 기본적인 단백질 설계 네트워크를 통해 사전 훈련되며, 다양하고 복잡한 효소 작업을 위한 다중 모달리티 입력을 받을 수 있는 구조를 갖추고 있습니다.

- **Performance Highlights**: MetaEnzyme은 다양한 효소 설계 작업에서 뛰어난 적응력을 보여줍니다. 추가적인 웻랩 실험을 통해 설계 과정의 효능이 검증되었습니다.



### VyAnG-Net: A Novel Multi-Modal Sarcasm Recognition Model by Uncovering Visual, Acoustic and Glossary Features (https://arxiv.org/abs/2408.10246)
- **What's New**: 본 연구에서는 대화에서의 sarcasm 인식을 위한 새로운 접근법인 VyAnG-Net을 제안합니다. 이 방법은 텍스트, 오디오 및 비디오 데이터를 통합하여 더 신뢰할 수 있는 sarcasm 인식을 가능하게 합니다.

- **Technical Details**: 이 방법은 lightweight depth attention 모듈과 self-regulated ConvNet을 결합하여 시각 데이터의 핵심 특징을 집중적으로 분석하고, 텍스트 데이터에서 문맥에 따른 중요 정보를 추출하기 위한 attentional tokenizer 기반 전략을 사용합니다. Key contributions로는 subtitles에서 glossary content의 유용한 특징을 추출하는 attentional tokenizer branch, 비디오 프레임에서 주요 특징을 얻는 visual branch, 음향 콘텐츠에서 발화 수준의 특징 추출, 여러 모달리티에서 획득한 특징을 융합하는 multi-headed attention 기반 feature fusion branch가 포함됩니다.

- **Performance Highlights**: MUSTaRD 벤치마크 비디오 데이터셋에서 speaker dependent 및 speaker independent 환경에서 각각 79.86% 및 76.94%의 정확도로 기존 방법들에 비해 우수함을 입증하였습니다. 또한, MUStARD++ 데이터셋의 보지 않은 샘플을 통해 VyAnG-Net의 적응성을 평가하는 교차 데이터셋 분석도 수행하였습니다.



### TrIM: Triangular Input Movement Systolic Array for Convolutional Neural Networks -- Part II: Architecture and Hardware Implementation (https://arxiv.org/abs/2408.10243)
Comments:
          This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 이 논문에서는 CNN을 위한 TrIM(삼각형 입력 이동) 기반의 혁신적인 데이터플로우를 제안하여, 최신 방식의 systolic 배열과 비교하여 메모리 접근 수를 10배 줄일 수 있는 하드웨어 아키텍처를 소개합니다.

- **Technical Details**: TrIM 아키텍처는 1512개의 처리 요소(Processing Elements, PEs)로 구성되어 있으며, VGG-16 CNN을 가속화하기 위해 Field Programmable Gate Array (FPGA)에 구현되었습니다. 이 디자인은 150 MHz의 클록 주파수에서 작동하며, 최대 453.6 Giga Operations per Second의 처리량을 제공합니다. 메모리 접근 측면에서 TrIM은 기존의 Eyeriss 아키텍처보다 약 5.1배 더 뛰어난 성능을 보이며, 에너지 효율성에서도 우수한 결과를 나타냅니다.

- **Performance Highlights**: TrIM 아키텍처는 peak throughput이 453.6 GOPS/s에 달하며, 4.2 W의 전력 소모로 작동합니다. 이는 기존의 신경망 가속기들과 비교하여 최대 12.2배 더 에너지 효율적인 성능을 보여줍니다.



### AltCanvas: A Tile-Based Image Editor with Generative AI for Blind or Visually Impaired Peop (https://arxiv.org/abs/2408.10240)
- **What's New**: AltCanvas는 시각 장애인을 위한 그림 제작 도구로, 텍스트-투-이미지 생성 AI의 능력을 활용한 구조적인 접근 방식을 통합하여, 사용자에게 향상된 제어 및 편집 기능을 제공합니다.

- **Technical Details**: AltCanvas는 사용자가 시각 장면을 점진적으로 구성할 수 있는 타일 기반 인터페이스를 제공합니다. 각 타일은 장면 내의 객체를 나타내며, 사용자들은 음성 및 오디오 피드백을 받으며 개체를 추가, 편집, 이동 및 배열할 수 있습니다.

- **Performance Highlights**: AltCanvas의 워크플로를 통해 14명의 시각 장애인 참가자들이 효과적으로 일러스트를 제작하였으며, 색상 일러스트레이션이나 촉각 그래픽 생성을 위한 벡터로 렌더링할 수 있는 기능을 제공하였습니다.



### A Conceptual Framework for Ethical Evaluation of Machine Learning Systems (https://arxiv.org/abs/2408.10239)
- **What's New**: 이 논문에서는 머신러닝(Machine Learning) 시스템의 평가 과정에서 발생하는 윤리적 문제를 다루고 있습니다. 특히, 정보 수익과 윤리적 해악 사이의 균형을 맞추는 평가 기술의 필요성을 강조합니다.

- **Technical Details**: 논문은 ML 시스템 평가와 관련된 다양한 윤리적 고려사항들을 개념적으로 분석하며, 평가의 효용 프레임워크(utility framework)를 제시합니다. 이 프레임워크는 정보 이득(information gain)과 잠재적인 윤리적 해악(ethical harms) 간의 주요 균형을 특징짓고 있으며, ML 시스템의 개발 및 평가 과정에서 직면하는 복잡한 윤리적 문제를 시각화합니다.

- **Performance Highlights**: 논문은 ML 평가 방법론을 심층적으로 탐구하며, 임상 시험(clinical trials) 및 자동차 충돌 테스트(automotive crash testing)와 같은 유사한 분야들에서의 모범 사례를 참고하여 윤리적 문제를 해결할 수 있는 방법을 제안합니다. 이 분석은 ML 팀에게 평가 과정에서의 윤리적 복잡성을 의도적으로 평가하고 관리할 필요성을 강조합니다.



### A General-Purpose Device for Interaction with LLMs (https://arxiv.org/abs/2408.10230)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)과 고급 하드웨어 통합에 대한 연구로, LLM과의 향상된 상호작용을 위해 설계된 범용 장치 개발에 중점을 두고 있습니다. 전통적인 가상 어시스턴트(VA)의 한계를 극복하고, LLM을 통합한 새로운 지능형 어시스턴트(IA) 시대를 여는 데 그 목적이 있습니다.

- **Technical Details**: 이번 연구에서는 LLM의 요구 사항에 맞춘 독립적인 하드웨어 플랫폼에서 IA를 배치하여 복잡한 명령을 이해하고 실행하는 능력을 향상시키고자 하며, 여러 형태의 입력을 통합하는 혁신적인 프레임워크를 설계했습니다. 이 프레임워크는 음성 입력을 향상시키기 위해 로컬 전처리를 집중적으로 개선하여 더 정확한 입력을 달성하는 것을 목표로 합니다.

- **Performance Highlights**: 제안된 장치는 다차원 입력을 처리할 수 있는 능력을 갖추고 있으며, 음성 및 비디오 센서와 환경 센서를 포함하여 사용자와의 상호작용을 개선합니다. 로컬 캐싱 기능을 통해 처리 속도를 높이고 응답 시간을 감소시키는 동시에 시스템 효율성을 크게 향상시키는 성과를 보였습니다.



### Neural Horizon Model Predictive Control -- Increasing Computational Efficiency with Neural Networks (https://arxiv.org/abs/2408.09781)
Comments:
          6 pages, 4 figures, 4 tables, American Control Conference (ACC) 2024

- **What's New**: 본 논문은 모델 예측 제어(Model Predictive Control, MPC) 알고리즘의 컴퓨팅 부하를 줄이기 위해 피드포워드 신경망(feed-forward neural network)을 활용한 새로운 접근 방식을 제안합니다.

- **Technical Details**: 이 방법은 문제의 수평 일부를 근사화(approximate)하면서도 나머지 최적화 부분을 통해 안전 보장(safety guarantees)인 제약 만족(constraint satisfaction)을 유지합니다.

- **Performance Highlights**: 시뮬레이션에서 검증되어 계산 효율성(computational efficiency)이 개선되었으며, 보장과 거의 최적의 성능(nearly-optimal performance)을 유지하는 것이 입증되었습니다. 이 제안된 MPC 방식은 로봇 공학(robotics) 및 제한된 계산 자원(embedded applications)으로 신속한 제어 응답을 필요로 하는 다양한 응용 프로그램에 적용될 수 있습니다.




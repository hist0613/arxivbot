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




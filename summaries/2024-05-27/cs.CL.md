### Scaling Laws for Discriminative Classification in Large Language Models (https://arxiv.org/abs/2405.15765)
- **What's New**: 최근의 대형 언어 모델(LLMs)은 기계 학습 모델의 새로운 패러다임을 제시합니다. 특히 고객 지원에서 유용할 것으로 보이나, 환각(hallucinations) 문제는 그 도입을 어렵게 하고 있습니다. 이에 대한 해결책으로, LLM의 언어 모델링 작업을 판별적(classification) 작업으로 재구성해 고객 지원 담당자를 보조하는 시스템을 제안합니다.

- **Technical Details**: 이 시스템은 다음 답변을 위한 상위 K개의 템플릿 응답을 제공함으로써 고객 지원 담당자를 지원합니다. 이를 위해 LLM을 사용하는 두 단계의 학습 파이프라인을 구축했습니다. 첫째, LLM을 도메인에 맞게 적응시키고, 둘째, 이를 판별적으로 세분화된 분류기로 미세 조정합니다. 본 연구는 주로 GPT, LLaMA, PaLM, Pythia 등의 모델 패밀리를 활용합니다. 또한, 모델의 크기, 지연(time-latency), 정확도 간의 균형도 고려했습니다.

- **Performance Highlights**: 오프라인 및 온라인 실험을 통해 오프라인 성능 향상과 통계적으로 유의미한 온라인 성능 향상을 확인했습니다. 특정 고객 지원 응용 프로그램에서의 프레임 방식으로 모델 환각 및 데이터 누출 문제를 효과적으로 피할 수 있었습니다. 또한, 이 산업 시스템을 생산 환경에 적용하기 위해 다양한 실험을 통해 모델 성능을 검증했습니다.



### GPT is Not an Annotator: The Necessity of Human Annotation in Fairness Benchmark Construction (https://arxiv.org/abs/2405.15760)
Comments:
          Accepted to ACL 2024 (main conference)

- **What's New**: 이 논문은 기존의 편향(Bias) 벤치마크 데이터셋을 평가하고, 새로운 커뮤니티 소스 방식을 사용하여 이를 구축하는 데 있어서 GPT-3.5-Turbo가 도움이 될 수 있는지를 탐색합니다. 특히, 유대인 커뮤니티와 반유대주의(Antisemitism)에 대한 새로운 편향 벤치마크 'WinoSemitism'을 소개합니다.

- **Technical Details**: 이 연구는 Felkner 외(2023)의 방법론을 참고하여 설문 조사와 인적 주석(Human Annotation)을 통해 데이터셋을 생성했습니다. 참가자들은 Jewish 커뮤니티 내 다양한 채널을 통해 모집되었고, 연령, 성별, 민족 등의 인구통계 질문과 유대인 문화 및 종교적 배경에 대한 질문에 답변했습니다. GPT-3.5-Turbo는 주석 작업에서 인간 전문가를 대체하려는 시도로 사용되었습니다.

- **Performance Highlights**: 분석 결과, GPT-3.5-Turbo는 주석 작업에서 품질 문제가 심각하며, 민감한 사회적 편향 작업에서는 인간 주석자를 대체하기에 적합하지 않다는 결론을 내렸습니다. 모델의 낮은 성능과 부적절한 출력 품질로 인해 커뮤니티 소싱의 많은 이점이 반감되었습니다.



### Filtered Corpus Training (FiCT) Shows that Language Models can Generalize from Indirect Evidenc (https://arxiv.org/abs/2405.15750)
Comments:
          10 pages + 7 pages of references/appendices. For code and trained models, see this http URL

- **What's New**: 이 논문은 필터링된 코퍼스 훈련(Filtered Corpus Training, FiCT)이라는 새로운 방법론을 소개합니다. 이 방법론은 특정 언어 구조를 필터링한 데이터로 언어 모델(LM)을 훈련시켜, 간접 증거를 바탕으로 한 언어 일반화 능력을 측정합니다. LSTM과 Transformer 모델에 이 방법을 적용해 다양한 언어 현상을 살펴보았습니다. 그 결과, Transformer 모델이 일반적인 퍼플렉서티(perplexity) 측정에서는 더 우수하지만, 언어 일반화 측면에서는 LSTM과 비슷한 성능을 보였습니다.

- **Technical Details**: FiCT 방법론은 훈련 데이터에서 특정 언어 구조를 제거하고, 모델이 이 데이터를 기반으로 얼마나 잘 일반화할 수 있는지를 테스트합니다. 예를 들어, 주어가 전치사구(prepositional phrase)에 의해 수식되지 않는 코퍼스로 모델을 훈련시키고, 그런 문장의 문법성을 판단할 수 있는 능력을 평가합니다. 이를 통해 모델이 직접적인 예제 없이도 간접적인 증거를 통해 언어 규칙을 일반화할 수 있는지를 확인합니다. 또한, LSTM과 Transformer 두 가지 주요 언어 모델 아키텍처의 귀납적 편향(inductive biases)을 조사했습니다.

- **Performance Highlights**: 결과적으로 Transformer 모델은 퍼플렉서티 측정에서 더 우수하지만, 언어 일반화 측면에서는 LSTM과 큰 차이가 없었습니다. 또한, 필터링된 코퍼스 훈련이 문법성 판단에 미치는 영향은 두 모델 모두에서 비교적 낮게 나타났습니다. 이는 언어 모델들이 간접적인 증거만으로도 복잡한 언어 일반화를 잘 수행할 수 있다는 것을 시사합니다.



### EmpathicStories++: A Multimodal Dataset for Empathy towards Personal Experiences (https://arxiv.org/abs/2405.15708)
Comments:
          Accepted to ACL 2024 Findings

- **What's New**: EmpathicStories++는 41명의 참가자가 한 달 동안 가정에서 소셜 로봇과 상호작용하면서 개인적인 이야기를 공유하는 과정을 담고 있는 멀티모달 데이터셋입니다. 이는 기존 연구들의 'in-the-wild', 장기적(longitudinal), 그리고 자기 보고(self-reported) 데이터를 보완하며, 자연스러운 환경에서의 공감을 포착합니다.

- **Technical Details**: EmpathicStories++ 데이터셋은 53시간에 달하는 비디오, 오디오, 텍스트 데이터를 포함하고 있으며, 참가자들이 로봇에게 개인적인 경험을 공유하고 공감대가 형성되는 이야기를 읽습니다. 이 데이터셋은 영상, 음성, 텍스트의 로우 레벨 feature뿐만 아니라, 심리측정학적 데이터와 다른 사람의 이야기에 대한 공감 점수도 포함하고 있습니다.

- **Performance Highlights**: 새로운 벤치마크 작업은 참가자의 개인적 경험을 기반으로 공감을 예측하는 것입니다. 이 작업은 참가자가 공유한 자신의 이야기를 기반으로 한 공감 예측과 참가자가 읽은 이야기에서의 반영을 기반으로 한 공감 예측 두 가지로 평가됩니다. 이를 통해 향상된 맥락적(Contextual)이고 장기적인(Longitudinal) 공감 모델링이 가능해집니다.



### GECKO: Generative Language Model for English, Code and Korean (https://arxiv.org/abs/2405.15640)
- **What's New**: GECKO는 한국어와 영어, 그리고 프로그래밍 언어를 최적화한 이중언어 대형 언어 모델(LLM)입니다. LLaMA 아키텍처를 바탕으로 양질의 한국어와 영어 말뭉치를 균형 있게 사전 훈련한 GECKO는 소규모 어휘 크기에도 불구하고 뛰어난 효율성을 보입니다. 이 모델은 대표적인 벤치마크, 특히 KMMLU(Korean MMLU)에서 큰 성능을 발휘하며, 영어와 코드 분야에서도 적절한 성과를 보여줍니다. GECKO는 오픈 소스 커뮤니티에 허용적인 라이선스 하에 공개되어 있습니다.

- **Technical Details**: GECKO는 고품질의 한국어와 영어 말뭉치를 사용해 Byte Pair Encoding(BPE) 알고리즘을 통해 토크나이저를 훈련합니다. UTF-8 문자를 개별 바이트로 분할하고 숫자를 개별 숫자로 처리하여 어휘에서 벗어나는 문제를 방지합니다. GECKO는 32,000개의 어휘 크기를 유지하며, 더 큰 어휘가 추론 시 더 많은 계산 능력을 요구하므로 최적의 균형을 찾습니다. 모델은 LLaMA에서 사용되는 디코더 전용 트랜스포머 아키텍처를 채택하며, AdamW 옵티마이저를 사용해 BF16 혼합 정밀도로 2000억 개의 토큰을 훈련합니다.

- **Datasets**: GECKO의 훈련에는 terabytes 이상의 한국어 말뭉치를 수집하고, 영어와 프로그래밍 언어의 대규모 오픈 소스 말뭉치를 활용합니다. 데이터 전처리에서 유해 콘텐츠 완화, 데이터 중복 제거, 구조적 정보 유지에 중점을 둡니다. 위키, 프로그래밍 코드, 수학적 표현 등 전문적 기여가 포함된 말뭉치를 처리하여 모델이 맥락적으로 정보를 이해하고 문법적으로 일관된 출력을 생성할 수 있도록 합니다.

- **Performance Highlights**: GECKO는 KMMLU 벤치마크에서 우수한 성능을 발휘하며, 코딩과 수학도에서 보통 이상의 성과를 보입니다. 이는 모델이 적은 양의 훈련 자원으로도 한국어에서 뛰어난 성능을 보여줄 수 있음을 입증합니다.

- **Conclusion**: GECKO는 한국어 대형 언어 모델 사전 훈련에 기여할 수 있는 오픈 소스 모델입니다. 향후 추가 훈련 자원을 통한 개선된 모델 출시와 함께 GECKO의 명령어 따르기 능력을 평가하기 위한 명령어 미세 조정을 준비하고 있습니다. 오픈 소스 AI 기술의 공개는 더 안전한 제품을 만들고, 혁신을 가속화하며, 시장을 확장하는 데 도움이 될 것입니다.



### Text Generation: A Systematic Literature Review of Tasks, Evaluation, and Challenges (https://arxiv.org/abs/2405.15604)
Comments:
          35 pages, 2 figures, 2 tables, Under review

- **What's New**: 본 리뷰 논문은 2017년부터 2024년까지 문서 생성(text generation) 연구에 관한 체계적인 문헌 리뷰를 제공합니다. 244편의 논문을 선정하여, 문서 생성 작업을 다섯 가지 주요 작업으로 분류했습니다: 오픈엔디드 텍스트 생성(open-ended text generation), 요약(summarization), 번역(translation), 패러프레이징(paraphrasing), 및 질의응답(question answering). 각 작업에 대한 특징과 세부 작업 및 기존 데이터 부족, 이야기 생성에서의 일관성, 복잡한 질의응답에서의 복잡한 추론 등의 도전 과제를 다룹니다. 또한, 문서 생성 시스템의 평가 방법과 현재 평가 지표의 문제점을 논의합니다.

- **Technical Details**: 문서 생성 작업을 다섯 가지 주요 카테고리로 분류하고, 각 작업의 세부적인 특성과 도전과제를 분석합니다: 
1. 오픈엔디드 텍스트 생성 (예: 이야기 생성, 대화 생성)
2. 요약 (단일 문서와 다중 문서 요약)
3. 번역 (문장 수준과 문서 수준 번역)
4. 패러프레이징 (통제되지 않은 패러프레이징과 통제된 패러프레이징)
5. 질의응답 (내부 지식과 외부 지식을 사용하는 질의응답). 또한, 다양한 평가 방법론 (모델프리 및 모델베이스드 메트릭스)을 평가하고 그 한계를 논의합니다.

- **Performance Highlights**: 리뷰는 아홉 가지 주요 도전과제를 제시합니다: 편향(bias), 추론(reasoning), 환각(hallucinations), 오용(misuse), 프라이버시(privacy), 해석가능성(interpretability), 투명성(transparency), 데이터셋(datasets), 및 컴퓨팅(computing). 각 도전 과제의 세부 내용을 분석하고 잠재적 해결책을 제공하며, 커뮤니티가 더 많은 참여를 필요로 하는 격차를 강조합니다.



### Profiling checkpointing schedules in adjoint ST-AD (https://arxiv.org/abs/2405.15590)
- **Whats New**: 이 논문에서는 adjoint algorithmic differentiation(AD)에서 중요한 역방향 데이터 흐름을 최적화하기 위해 체크포인트 배치를 개선하는 방법을 제안하고 있습니다. 특히, MITgcm 해양 및 대기 전순환 모델의 사례를 통해 이 접근법의 유용성을 입증합니다.

- **Technical Details**: 체크포인트 배치는 실행 시간과 메모리 소비의 균형을 맞추기 위해 사용되는 기술입니다. 저자들은 본 논문에서 adjoint 코드의 실행 프로파일링(runtime profiling)을 기초로 한 휴리스틱을 제안합니다. 이 휴리스틱은 기존 소스 변환 AD 도구에 구현되었으며, 체크포인트 배치를 최적화하여 실행 시간을 단축하고 메모리 사용량을 줄이기 위한 방법을 제시합니다.

- **Performance Highlights**: 제안된 방법은 MITgcm 코드 스위트의 실제 사례에서 테스트되었으며, 프로파일링 결과를 활용하여 개발자가 성능을 크게 향상시킬 수 있음을 보여줍니다. 이는 계산 시간과 메모리 사용량을 효과적으로 줄이는 데 기여합니다.



### Synergizing In-context Learning with Hints for End-to-end Task-oriented Dialog Systems (https://arxiv.org/abs/2405.15585)
- **What's New**: 이번 연구에서는 기존 LLM(대형 언어 모델)을 이용해 높은 성능을 내는 Task-Oriented Dialog (TOD) 시스템보다 더욱 개선된 SyncTOD를 제안합니다. SyncTOD는 LLM과 함께 작업에 대한 유용한 힌트를 제공함으로써, LLM의 추론 능력과 감독 학습 모델의 작업 일치를 결합하여 TOD 시스템의 성능을 향상시킵니다.

- **Technical Details**: SyncTOD는 두 가지 주요 구성 요소를 포함합니다: '힌트 예측기'와 '대표 예시 선택기'. 힌트 예측기는 대화 기록을 바탕으로 적절한 힌트를 제공하며, 대표 예시 선택기는 훈련 대화 데이터에서 가장 적절한 예제를 선택하고 재정렬합니다. 이를 통해 생성된 응답이 훈련 데이터의 스타일과 일치하도록 합니다. 특히, SyncTOD는 세 가지 유형의 힌트를 사용합니다: 응답에 포함될 엔티티 유형(Entity Types, ET), 응답 길이, 대화 종료.

- **Performance Highlights**: SyncTOD는 제한된 데이터 설정에서 기존의 LLM 기반과 SoTA(최신 기술 수준) 모델들보다 우수한 성능을 보여주며, 전체 데이터 설정에서도 경쟁력 있는 성능을 유지합니다. 특히, ChatGPT와 함께 사용했을 때 더욱 뛰어난 성능을 발휘합니다.



### Sparse Matrix in Large Language Model Fine-tuning (https://arxiv.org/abs/2405.15525)
Comments:
          14 pages

- **What's New**: 이번 연구에서는 Sparse Matrix Tuning (SMT) 방법을 새롭게 제안하여, PEFT 방법과 풀 파인튜닝(Full Fine-Tuning; FT) 간의 성능 차이를 최소화하면서도 파인튜닝 시 발생하는 계산 비용과 메모리 비용을 줄였습니다.

- **Technical Details**: SMT 방법은 그래디언트 업데이트에서 가장 중요한 하위 행렬들만 식별하여 파인튜닝 과정 동안 이들 블록만 업데이트합니다. 또한, LoRA와 DoRA 같은 기존의 PEFT 방법들이 트레이닝 가능한 파라미터 수가 증가함에 따라 성능이 점차 감소하는 반면, SMT 방법은 이 문제를 겪지 않습니다. SMT는 행렬 희소성(Matrix Sparsity)을 적용하여, 효율적으로 가장 관련 있는 메모리 섹션을 식별하고 파인튜닝합니다.

- **Performance Highlights**: SMT는 LLaMA와 같은 인기 있는 대형 언어 모델(LLM)에서 다양한 작업에 대해 LoRA와 DoRA보다 일관되게 우수한 성능을 보여주며, 풀 파인 튜닝 대비 GPU 메모리 사용량을 67% 줄였습니다. 예를 들어, LLaMA-7B에서는 상식을 묻는 문제에서 +3.0 포인트, 산수 문제에서 +2.3 포인트의 성능 향상을 보였습니다. 추가로, SMT는 SMT와 풀 파인튜닝 간의 성능 격차를 없애고, LoRA와 DoRA에 비해 적은 트레이닝 파라미터(5% 이하)로도 뛰어난 성능을 보여줍니다.



### Mosaic Memory: Fuzzy Duplication in Copyright Traps for Large Language Models (https://arxiv.org/abs/2405.15523)
- **What's New**: 새로운 연구는 대형 언어 모델 (LLMs)이 학습하는 데이터셋에 저작권 보호 콘텐츠가 포함되는 문제를 해결하기 위해 'fuzzy copyright traps'를 제안합니다. 이 기법은 기존의 동일한 텍스트 반복 삽입 대신, 약간 수정된 콘텐츠를 포함하여 데이터 중복 제거 기술로부터 보호받을 수 있습니다.

- **Technical Details**: 연구팀은 1.3B 파라미터를 가진 LLM인 CroissantLLM을 미세조정하면서, 저작권 함정 트랩(fuzzy trap sequences)을 주입했습니다. 이 함정은 특정 토큰을 변경한 반복된 시퀀스입니다. 시퀀스 수준의 Membership Inference Attack (MIA) ROC AUC를 통해 함정의 기억 수준을 측정했으며, 4개의 토큰을 교체했을 때 AUC가 0.90에서 0.87로 미세하게 감소하는 것을 발견했습니다.

- **Performance Highlights**: 비교 데이터셋 The Pile에서 자연 발생하는 fuzzy duplicates가 30% 이상 포함되어 있는 것을 발견했습니다. 또한, MIA AUC가 토큰 교체 전략에 따라 증가하는 것을 보여주었고, 이는 의미적 일관성이 함정 기억에 도움이 된다는 것을 시사합니다. 이 연구는 LLM의 기억 현상을 분석하는 새로운 변수로써 중요한 의미를 가집니다.



### Emergence of a High-Dimensional Abstraction Phase in Language Transformers (https://arxiv.org/abs/2405.15471)
- **What's New**: 이번 연구는 사전 훈련된 트랜스포머 기반의 언어 모델들(LMs)에서 나타나는 독특한 고차원적 특성을 조사합니다. 이 연구는 언어 모델 입력의 첫 번째 완전한 추상화 단계와 이 단계가 다운스트림 작업에 효율적으로 전이될 수 있음을 발견했습니다. 또한, 이 단계의 초기 발생이 향후 더 좋은 언어 모델링 성능을 예측할 수 있음을 시사합니다.

- **Technical Details**: 언어 모델의 계층 구조를 통해 정보를 변환하는 동안 나타나는 내부 기하학적 특성을 분석했습니다. 중간 계층에 고차원적 표현이 존재하며, 이는 다양한 언어요소를 복잡하게 추상화하는 것을 나타냅니다. 실험에서는 Generalized Ratios Intrinsic Dimension Estimator(GRIDE) 방법을 사용해 각 계층의 본질적인 차원을 추정했습니다.

- **Performance Highlights**: 이 연구는 5개의 다른 사전 훈련된 언어 모델(OPT-6.7B, Llama-3-8B, Pythia-6.9B, OLMo-7B, Mistral-7B)에서 고차원적 표현의 고유성을 발견했습니다. 고차원성 피크가 더 빨리 나타나는 모델일수록 언어 모델링 성능이 우수하다는 점도 확인되었습니다. 이러한 고차원 표현은 구문 및 의미 분석 작업에서 중간 성능을 보여줍니다.



### Linearly Controlled Language Generation with Performative Guarantees (https://arxiv.org/abs/2405.15454)
- **What’s New**: 본 논문에서는 대형 언어 모델(LM)들이 자연어 생성 도중 불필요한 혹은 유해한 내용으로부터 벗어나도록 제어하는 방법을 제안합니다. 이를 위해, 텍스트 생성이 언어 모델의 잠재 공간(latent space)에서 궤적을 이룬다는 개념을 도입하고, 제어 이론을 적용하여 이 궤적을 제어합니다. 특히 경량의 경사도 없는 개입(intervention)을 사용하여 불필요한 의미로부터 벗어나도록 하는 방식입니다. 이 개입은 확률적으로 의도된 영역으로 출력을 유도할 것을 보증합니다.

- **Technical Details**: 제안된 방법은 이를 Linear Semantic Control (LiSeCo)라고 부르며, 이는 최적 제어 이론(optimal control theory)를 기반으로 하여 텍스트 생성을 제약된 최적화 문제로 공식화하여 이론적인 보증을 제공합니다. 구체적으로, 언어 모델의 잠재 공간에서 특정 지점이 불필요한 의미를 포함하는 영역인지 여부를 세밀하게 파악한 뒤, 해당 영역에서 벗어날 수 있도록 각 레이어에서 출력 벡터를 수정합니다. 이 접근법은 몇 가지 특징이 있습니다: 1) 최적화 문제에 대한 닫힌 형태(closed form)의 해를 제공하며, 2) 입력 레이어와 은닉 레이어에서 경량으로 동작하여 계산량을 대폭 줄입니다.

- **Performance Highlights**: 실험 결과는 LiSeCo가 유해한 내용으로부터 벗어나도록 텍스트 생성을 제어하면서도 텍스트의 품질을 유지함을 보여줍니다. 또한, 이 방법은 경량의 계산으로 실제 적용될 때도 유효함을 입증했습니다.



### Benchmarking Pre-trained Large Language Models' Potential Across Urdu NLP tasks (https://arxiv.org/abs/2405.15453)
- **What's New**: 사용 가능한 주요 다국어 언어 모델(LLMs)인 GPT-3.5-turbo, Llama2-7B-Chat, Bloomz 7B1 및 Bloomz 3B를 14개의 작업에 대해 평가하고 15개의 우르두어 데이터셋을 사용하여 평가한 결과를 제시합니다. 본 연구는 기존의 최첨단 모델(State-of-the-Art, SOTA)과 비교 분석을 통해, 우르두어 NLP(자연어 처리) 작업에서 성능을 검토하는 최초의 심층 연구입니다.

- **Technical Details**: 이번 연구에서는 다국어 언어 모델을 사용하여 우르두어 NLP 작업을 수행하고, SOTA 모델과 결과를 비교합니다. 실험에 사용된 모델은 GPT-3.5 Turbo, Bloomz 3B, 7B1, Llama 2입니다. 모델들은 OpenAI의 GPT-3.5 Turbo, Meta의 Llama 2, 그리고 Big Science의 Bloomz 3B 및 7B1을 포함하며, zero-shot 설정에서 평가되었습니다. 각 모델은 특정 프롬프트를 사용하여 접근한 결과를 후처리해 실험 셋의 출력과 일치시켰습니다.

- **Performance Highlights**: 실험 결과, SOTA 모델이 모든 우르두어 NLP 작업에서 zero-shot 학습 시 모든 인코더-디코더 미리 훈련된 언어 모델을 능가하는 것으로 나타났습니다. 또한 규모는 작지만 특정 언어 데이터가 더 많이 포함된 모델이, 큰 규모이면서도 데이터가 적은 모델보다 더 나은 성능을 보여주었습니다. 특히 Named Entity Recognition, News Categorization, Intent Detection 등 여러 작업에서 뛰어난 성능을 기록했습니다.



### Leveraging Logical Rules in Knowledge Editing: A Cherry on the Top (https://arxiv.org/abs/2405.15452)
Comments:
          18 pages

- **What's New**: RULE-KE라는 새로운 프레임워크는 Multi-hop Question Answering (MQA)에서 지식 편집(KE) 문제를 향상시키기 위해 제안되었습니다. 이 접근법은 논리적 규칙(rule discovery)을 발견하여 높은 상관 관계를 가진 사실들을 업데이트하는 데 사용됩니다.

- **Technical Details**: RULE-KE는 기존의 MQA 방법을 보완하기 위해 논리적 규칙을 발견하여, 분해하기 어려운 질문과 상관 관계가 높은 지식 업데이트(Correlated Knowledge Updates) 문제를 해결합니다. 기존의 파라미터 기반(parameter-based)와 메모리 기반(memory-based) 방법 모두에 적용 가능합니다.

- **Performance Highlights**: 기존의 벤치마크 데이터셋과 신규 데이터셋(RKE-EVAL)을 사용한 실험 결과, RULE-KE는 파라미터 기반 솔루션에서는 최대 92%, 메모리 기반 솔루션에서는 최대 112.9%의 성능 향상을 보여주었습니다.



### Large Language Models can Deliver Accurate and Interpretable Time Series Anomaly Detection (https://arxiv.org/abs/2405.15370)
- **What's New**: 이번 연구에서는 LLMAD라는 새로운 시계열 이상 탐지(TSAD) 방법을 제안합니다. 이 방법은 대규모 언어 모델(LLMs)을 활용해 보다 정확하고 해석 가능한 TSAD 결과를 제공합니다. LLMAD는 유사한 정상 및 비정상 시계열 데이터를 검색해 LLM의 효과를 높이며, Anomaly Detection Chain-of-Thought (AnoCoT) 접근 방식을 통해 전문가의 논리를 모방하여 의사결정을 돕습니다. 이를 통해 탐지된 이상값에 대한 다양한 관점에서의 설명을 제공할 수 있습니다. LLMAD는 기존의 최첨단 딥러닝 기반 모델들과 비교할 때 유사한 성능을 내면서도 해석 가능성을 크게 향상시킵니다.

- **Technical Details**: LLMAD는 사전 학습된 LLM를 활용해 시계열 데이터를 추가 학습(fine-tuning) 없이 이상 예측과 설명을 생성합니다. 이 모델은 In-Context Learning (ICL) 기법을 사용하여 정상 및 비정상 패턴을 검색하고, 이를 입력으로 사용해 몇 가지 예시만으로도 정확한 이상 예측을 가능하게 합니다. 또한, AnoCoT 프롬프팅을 통해 TSAD에 특화된 도메인 지식을 통합하여 예측 성능을 향상시키고 보다 논리적이고 인간이 이해하기 쉬운 해석을 제공합니다.

- **Performance Highlights**: 세 개의 주요 데이터셋에 대한 실험 결과, LLMAD는 최첨단 딥러닝 기반 탐지기와 유사한 정확도를 보였습니다. 특히, LLMAD는 탐지된 이상값의 유형, 경고 수준, 그리고 설명 등 유용한 해석 결과를 낮은 비용으로 제공할 수 있습니다.



### UnKE: Unstructured Knowledge Editing in Large Language Models (https://arxiv.org/abs/2405.15349)
- **What's New**: 기존의 구조화된 지식 편집 방법론은 주요적으로 MLP 계층(local) 또는 특정 뉴런에서 지식이 key-value 쌍으로 저장된다는 가정에 기반하여 작동해왔습니다. 하지만, 실제로 많은 양의 지식은 비구조화되고 복잡한 형태로 존재합니다. 이러한 문제를 해결하기 위해 새로 제안된 Unstructured Knowledge Editing (UnKE) 방법론은 계층과 토큰 차원에서 지식 저장 방식을 확장하여 비구조화된 지식을 효과적으로 편집할 수 있도록 합니다.

- **Technical Details**: UnKE는 기존의 'knowledge locating' 단계를 제거하고 첫 몇 계층을 key로 취급하며, 모든 입력된 토큰에 대해 'term-driven optimization'을 'cause-driven optimization'으로 대체합니다. 이렇게 함으로써 모든 입력 토큰에서 마지막 계층을 직접 최적화하여 요구되는 key 벡터를 생성합니다. 이를 통해 MLP와 Attention 계층의 잠재력을 활용하여 복잡하고 포괄적인 비구조화된 지식을 효과적으로 표현하고 편집할 수 있습니다.

- **Performance Highlights**: 새롭게 제안된 비구조화 지식 편집 데이터셋(UnKEBench)과 전통적인 구조화된 데이터셋에서 UnKE는 강력한 기반모델을 능가하는 성능을 보였습니다. 특히, UnKE는 종합적인 설정에서 탁월한 성능을 발휘했으며, 배치 및 순차 편집 시에서도 우수한 안정성을 보였습니다.



### BiSup: Bidirectional Quantization Error Suppression for Large Language Models (https://arxiv.org/abs/2405.15346)
- **What's New**: 본 논문에서는 BiSup이라는 양방향 양자화 오류 억제 방법을 소개합니다. BiSup은 기존의 단일 매트릭스 곱셈 최적화에 그치지 않고, 모델 내에서 발생하는 양자화 오류의 수직 및 수평 전파를 억제하는 새로운 접근 방식을 제시합니다. 이를 통해 기존의 상태 최첨단 방법들보다 더 나은 성능을 실현하며 저비트(weight-activation quantization) 양자화의 실용성을 한층 높입니다.

- **Technical Details**: BiSup은 다음의 두 가지 핵심 전략을 통해 양자화 오류를 억제합니다. 첫째, 오류의 수직적 축적을 억제하기 위해 적절한 최적화 가능한 파라미터 공간을 구성하여 소량의 데이터를 활용한 양자화 인지 파라미터 효율 미세 조정을 수행합니다. OmniQuant 및 QLoRA의 아이디어를 기반으로, BiSup은 activation outlier의 분포 패턴과 few-shot fine-tuning의 수렴을 고려하여 이 파라미터 공간을 설계합니다. 둘째, 오류의 수평적 확산을 방지하기 위해 중요한 토큰(attention weights가 큰 토큰)의 정확성을 유지하는 전략을 사용합니다. 이 목적을 위해 BiSup은 중첩된 프롬프트 혼합 정밀도 양자화 전략을 도입하여 시스템 프롬프트의 키-값 캐시를 고정밀도로 유지합니다.

- **Performance Highlights**: Llama와 Qwen 패밀리에서의 광범위한 실험을 통해 BiSup이 두 가지 상태 최첨단 방법보다 성능을 향상시킬 수 있음을 입증했습니다. 예를 들어, 평균 WikiText2 Perplexity는 Atom의 경우 13.26에서 9.41로, QuaRot의 경우 14.33에서 7.85로 감소했습니다. 이는 BiSup이 저비트 weight-activation quantization에서 실용적인 적용을 더욱 촉진할 수 있음을 시사합니다.



### Detection and Positive Reconstruction of Cognitive Distortion sentences: Mandarin Dataset and Evaluation (https://arxiv.org/abs/2405.15334)
- **What's New**: 이 연구는 긍정 심리학 이론에 기반한 긍정 재건 프레임워크(Positive Reconstruction Framework)를 소개합니다. 본 연구는 부정적 생각을 긍정적으로 재해석하는 방법을 제안하고 이를 통해 부정적 사고 패턴을 개선하고자 합니다. 이를 위해, 인지 왜곡(cognitive distortions)을 식별하고 원래 생각의 의미를 유지하면서 긍정적으로 재구성된 대안을 제시하는 두 가지 접근법을 결합합니다. 중국어(Mandarin)로 된 4001개의 인지 왜곡 탐지 데이터셋과 1900개의 긍정 재구성 데이터셋을 제공하여, NLP 모델의 적용 가능성을 시연합니다.

- **Technical Details**: 본 연구는 대규모 데이터셋을 사용하여 일반적인 용도로 훈련된 NLP 모델을 긍정 재구축과 인지 왜곡 탐지와 같은 특정 정신 건강 요구에 맞추어 미세 조정(fine-tuning)하였습니다. RoBERTa-wwm-ext 네트워크와 같은 사전 훈련된 모델을 사용했습니다. 그리고 P-Tuning, 프롬프트 엔지니어링(prompt engineering) 및 전이 학습(transfer learning) 등 최신 NLP 기법을 활용해 성능을 평가했습니다. 특히, 다층 퍼셉트론(MLP) 및 LSTM과 같은 다양한 Readout 전략을 사용하여 최상의 결과를 확보했습니다.

- **Performance Highlights**: 피-튜닝(P-Tuning)을 사용한 긍정 재구성 작업에서 모든 메트릭에서 일괄적으로 좋은 성능을 보였습니다. 또한 긍정 심리학에서 사용 가능한 여러 전략 중에서 낙관적 전략을 선택한 네트워크가 가장 우수한 성능을 보였습니다. 인지 왜곡 탐지 작업에서는 전체 사전 훈련 네트워크를 미세 조정하는 것이 최고의 결과를 이끌어냈습니다.



### Decompose and Aggregate: A Step-by-Step Interpretable Evaluation Framework (https://arxiv.org/abs/2405.15329)
- **What's New**: 새로운 연구가 LLM(대형 언어 모델)을 평가자로 사용하는 방법의 신뢰성에 관해 탐구하였습니다. 이 연구는 기존의 단일 평가 결정 방식의 한계를 극복하기 위해, '분해 및 집계(Decompose and Aggregate)' 프레임워크를 제안합니다. 이 프레임워크는 교육 평가 절차를 기반으로 한 다양한 단계별 접근 방식을 사용하여, LLM이 생성한 텍스트를 평가하는 능력을 보다 해석 가능하게 합니다.

- **Technical Details**: 분해 및 집계 프레임워크는 두 가지 주요 단계로 구성됩니다. 첫 번째로, LLM이 평가 기준을 여러 측면으로 분해하여 각각의 측면에 대해 쌍별 점수를 부여합니다. 그런 다음, 집계 단계에서 LLM은 주어진 맥락에서 각 측면의 중요도에 따라 가중치를 제안하고 이를 외부 계산 모듈을 통해 종합하여 최종 평가 결정을 내립니다.

- **Performance Highlights**: 이 프레임워크를 사용함으로써, 다양한 데이터셋에서 LLM의 평가 성능이 최대 39.6% 향상되는 것을 실험적으로 확인하였습니다. 또한, 사람들에 의한 중간 출력 결과 평가를 통해, LLM의 강점과 한계를 보다 세밀하게 이해할 수 있었습니다. 결과적으로, LLM의 블랙박스 평가 과정을 해석 가능하게 만들어 신뢰성을 높였습니다.



### Organic Data-Driven Approach for Turkish Grammatical Error Correction and LLMs (https://arxiv.org/abs/2405.15320)
- **What's New**: 우리는 새로운 유기적 데이터 기반 접근 방식인 'clean insertions'를 소개하여, 어떠한 유기적 데이터에서도 병렬적인 터키어 문법 오류 수정 데이터셋을 구축하고, 대형 언어 모델(Large Language Models) 훈련을 위한 데이터를 정제하려고 노력했습니다. 이 방법을 통해 세 가지 공개된 터키어 문법 오류 수정 테스트 세트 중 두 가지에서 최첨단 성능을 달성했습니다. 또한, 대형 언어 모델 훈련 손실에 대한 우리의 방법의 효과도 입증했습니다.

- **Technical Details**: 우리의 'clean insertions' 방법은 일반적으로 인터넷에서 크롤링한 유기적 텍스트에 있는 자주 발생하는 철자 오류와 구문을 정확한 버전으로 대체하는 잘못된-정확한 철자 사전을 구축하는 간단한 방법입니다. 이 사전과 그 크기는 생성된 데이터셋의 품질에 큰 영향을 미칩니다. 비록 사전이 데이터셋에 존재하는 모든 오류를 포함하지 않을 수 있지만, 부분적으로 수정된 GEC 데이터셋을 통해 최첨단 결과를 얻을 수 있음을 발견했습니다.

- **Performance Highlights**: 테스트에서 우리는 터키어 문법 오류 수정 작업에서 두 가지 다른 테스트 세트에서 최첨단 성능을 달성했습니다. 또한, 우리는 GPT-4를 사용하여 자동으로 병렬 GEC 데이터셋을 생성했고, 이 데이터셋을 가지고 훈련한 모델을 다른 데이터셋과 비교했습니다. 더 나아가, LLM 훈련을 위한 데이터를 정제하면 더 낮은 손실 값을 얻을 수 있음을 실험을 통해 확인했습니다.



### Stacking Your Transformers: A Closer Look at Model Growth for Efficient LLM Pre-Training (https://arxiv.org/abs/2405.15319)
Comments:
          Preprint; The project link: $\href{this https URL}{this https URL}$

- **What's New**: LLM(대규모 언어 모델)의 사전 교육(pre-training)은 연산 비용이 매우 크며, 이에 대한 해결책으로 작은 모델을 활용한 모델 성장(model growth) 방법이 주목받고 있습니다. 본 연구는 기존 모델 성장 방법론의 한계를 분석하고, 새로운 깊이 기반 쌓기 연산자($G_{	ext{stack}}$)의 효율성을 입증하였습니다.

- **Technical Details**: 연구에서는 모델 성장 방법을 네 가지 원자성 연산자로 요약하고 이를 표준화된 LLM사전 교육 설정에서 체계적으로 평가했습니다. 특히, 깊이 기반 쌓기 연산자($G_{	ext{stack}}$)가 탁월한 가속 성능을 보여 손실 감소와 성능 개선을 달성했습니다. 또한, $G_{	ext{stack}}$의 확장성 및 실질적인 적용을 위한 경험적 가이드라인을 제시하고, 포괄적인 실험을 통해 그 효율성을 검증했습니다.

- **Performance Highlights**: $G_{	ext{stack}}$ 연산자는 7B 규모의 LLM과 750B 토큰을 사용한 사전 교육에서 우수한 확장성과 성능을 보여줍니다. 예를 들어, 300B 토큰을 사용하여 컨벤셔널하게 훈련된 7B 모델과 동등한 손실에 도달하는 데 $G_{	ext{stack}}$ 모델은 194B 토큰만으로, 54.6%의 속도 향상을 달성합니다. 또한, 성장을 위한 타이밍 및 성장 요소에 대한 가이드라인을 공식화하여 일반적인 LLM 사전 교육에서도 적용 가능하도록 했습니다.



### Are Long-LLMs A Necessity For Long-Context Tasks? (https://arxiv.org/abs/2405.15318)
Comments:
          18 pages

- **What's New**: 이번 아카이브 논문에서는 Long-LLMs(긴 문맥의 대형 언어 모델)의 필요성을 반박하고, LC-Boost(Long-Context Bootstrapper)라는 프레임워크를 소개합니다. LC-Boost는 Short-LLM(짧은 문맥의 언어 모델)을 사용해 긴 문맥의 작업을 부트스트랩 방식으로 해결하는 방법을 제안합니다.

- **Technical Details**: LC-Boost 프레임워크는 두 가지 주요 결정에서 스스로 추론을 하도록 Short-LLM을 진척시킵니다: 1) 입력 내에서 적절한 부분의 문맥을 어떻게 접근할지, 2) 접근한 문맥을 어떻게 효과적으로 사용할지입니다. 이 프레임워크는 각 작업의 성격에 맞춰 적응적으로 긴 문맥 작업을 처리할 수 있습니다. 예를 들어, 지식 기반의 질문 응답 문제에서는 문맥을 검색을 통해 접근하고 답을 생성할 수 있으며, 정보 집계가 필요한 경우 문맥 전체를 분할하여 처리할 수 있습니다.

- **Performance Highlights**: LC-Boost를 사용한 실험 결과, LC-Boost는 강력한 Long-LLMs와 동등하거나 더 나은 성능을 보였습니다. 특히, 산만한 문맥을 제거할 수 있어 더 효율적이었습니다. 또한, 리소스 소비 측면에서도 더 적은 비용으로 더 나은 성과를 보였습니다.



### Before Generation, Align it! A Novel and Effective Strategy for Mitigating Hallucinations in Text-to-SQL Generation (https://arxiv.org/abs/2405.15307)
Comments:
          Accepted to ACL Findings 2024

- **What's New**: 본 논문에서는 In-Context Learning (ICL) 기반의 대형 언어 모델(LLMs)을 활용한 text-to-SQL 성능 향상 방안을 소개합니다. 현재의 방법들은 일반적으로 스키마 연결(schema linking)과 논리 합성(logical synthesis)의 두 단계를 거치며, 해석 가능성과 효율성을 모두 갖추고 있습니다. 그러나 이러한 진전에도 불구하고, LLM의 일반화 문제로 인해 종종 발생하는 '환상(hallucinations)'이 성능의 최대치를 제한합니다. 본 연구는 각 단계에서 발생하는 일반적인 환상의 유형을 식별하고 이를 분류한 후, 'Task Alignment (TA)'라는 새로운 전략을 도입하여 각 단계에서 환상을 줄이는 방법을 제안합니다. 이를 바탕으로 TA-SQL이라는 text-to-SQL 프레임워크를 제안합니다.

- **Technical Details**: 패러다임이 두 단계로 나뉩니다. 첫 번째 단계인 스키마 연결(schema linking)은 자연어 쿼리를 데이터베이스 스키마의 관련 엔티티와 정확하게 일치시키는 작업을 포함하며, 투명성을 부여하여 다음 쿼리 실행에 중요한 역할을 합니다. 두 번째 단계인 논리 합성(logical synthesis)은 자연어 쿼리의 논리와 데이터베이스 구조를 이해하여 정확한 SQL 쿼리를 생성하는 과정을 포함합니다. 이 논문에서는 이러한 단계에서 발생하는 환상을 두 가지 주요 범주, 즉 스키마 기반 환상과 논리 기반 환상으로 분류하였습니다. TA-전략은 모델이 유사한 작업에서 얻은 경험을 활용하도록 하여 환상을 효과적으로 줄이는 것을 목표로 합니다.

- **Performance Highlights**: TA-SQL은 GPT-4 기반의 기본 성능을 BIRD dev 세트에서 21.23% 상대적으로 향상시켰으며, 여섯 가지 모델과 네 가지 주류의 복잡한 text-to-SQL 벤치마크에서 상당한 개선을 보였습니다. 이는 TA-SQL이 모델에 구애받지 않는 프레임워크임을 시사하며, 주요 폐쇄형 LLMs 뿐만 아니라 오픈 소스 LLMs에서도 적용 가능함을 보여줍니다.



### DeTikZify: Synthesizing Graphics Programs for Scientific Figures and Sketches with TikZ (https://arxiv.org/abs/2405.15306)
Comments:
          Project page: this https URL

- **What's New**: 최근 과학적 도해(figure)를 자동으로 제작하는 데 있어서 DeTikZify라는 새로운 다중모달(multi-modal) 언어 모델이 등장했습니다. 이 모델은 스케치나 기존 도해를 기반으로 TikZ 그래픽 프로그램을 생성하여 의미가 보존되도록 합니다. 특히, 많은 사람이 직접 생성한 36만 개 이상의 TikZ 그래픽을 포함한 DaTikZv2, 손으로 그린 스케치와 해당 과학 도해가 짝을 이루는 SketchFig, 다양한 과학 도해와 관련 메타데이터를 포함하는 SciCap++ 같은 새로운 데이터셋들을 소개했습니다. 이 데이터셋들을 기반으로 DeTikZify가 학습되었습니다.

- **Technical Details**: DeTikZify는 DaTikZv2와 SciCap++ 데이터셋을 통해 학습되었으며, SketchFig에서 학습된 합성 스케치(synthetic sketch)를 사용합니다. 또한, MCTS 기반의 추론 알고리즘을 도입하여 추가 학습 없이도 출력을 반복적으로 개선할 수 있도록 했습니다. 이를 통해 TikZ 프로그램을 높은 정확도로 자동 생성할 수 있습니다.

- **Performance Highlights**: 자동 및 인간 평가를 통해 DeTikZify가 상업적 Claude 3과 GPT-4V 모델보다 TikZ 프로그램을 더 잘 합성한다는 것을 입증했습니다. 특히 MCTS 알고리즘을 통해 성능이 크게 향상되었습니다. 또한 코드, 모델 및 데이터셋이 공개되었습니다.



### Decoding at the Speed of Thought: Harnessing Parallel Decoding of Lexical Units for LLMs (https://arxiv.org/abs/2405.15208)
Comments:
          Accepted for publication at LREC-COLING 2024

- **What's New**: 이번 논문에서는 대규모 언어 모델 (Large Language Models; LLMs)의 새로운 디코딩 기법인 Lexical Unit Decoding (LUD)을 제안합니다. LUD는 데이터 기반 접근 방식을 통해 디코딩 속도를 향상시키며, 출력 품질을 유지합니다. 논문은 LLM이 여러 연속적인 토큰을 예측할 수 있다는 관찰을 바탕으로, 이러한 연속 토큰을 'lexical unit'으로 정의하고, 이를 병렬로 디코딩할 수 있게 합니다.

- **Technical Details**: LUD의 핵심은 'lexical unit'의 식별입니다. 이는 모델이 높은 신뢰도로 예측하는 연속적인 토큰의 범위를 의미합니다. LUD는 이 유닛을 동시에 예측할 수 있도록 모델을 조정합니다. 추가적인 보조 모델 없이도 작동하며 기존 모델 아키텍처의 변경이 필요하지 않습니다. LLaMA-13B와 같은 대규모 모델에서 테스트한 결과, LUD는 디코딩 속도를 33% 향상시키면서도 출력 품질을 유지했습니다. 프로그램 코드 생성에서는 30%의 속도 향상과 3%의 품질 저하를 보였습니다.

- **Performance Highlights**: LUD는 자연어 생성의 경우 디코딩 속도를 33% 증가시키고 품질 손실 없이 작동합니다. 코드 생성의 경우에는 30%의 속도 향상을 보이며, 품질 저하는 단 3% 이하로 나타났습니다. LUD는 별도의 보조 모델을 사용하지 않으며, 기존의 모델 아키텍처 변경 없이 적용이 가능합니다. 이를 통해 모델 배포가 용이하고, 디코딩 속도를 크게 개선할 수 있습니다.



### Cross-Task Defense: Instruction-Tuning LLMs for Content Safety (https://arxiv.org/abs/2405.15202)
Comments:
          accepted to NAACL2024 TrustNLP workshop

- **What's New**: 최근 연구에 따르면, 대규모 언어 모델 (LLMs)은 장문의 텍스트를 처리할 때 안전성과 효율성 간의 균형을 맞추는 데 어려움을 겪고 있습니다. 이번 연구는 LLMs가 악의적인 문서와 일반 NLP 작업 쿼리를 안전하게 처리할 수 있도록 하는 강력한 방어 메커니즘을 개발하는 것을 목표로 하고 있습니다. 이를 위해 안전과 관련된 예제로 구성된 방어 데이터셋을 소개하고, instruction tuning을 위한 single-task와 mixed-task 손실함수를 제안합니다.

- **Technical Details**: 우리 연구는 LLMs를 adversarial robustness로 튜닝하기 위해, 거부 응답과 함께 안전과 관련된 예제로 구성된 방어 데이터셋을 구축하는 데 중점을 두었습니다. 수백 개의 악의적 문서를 수집하고, 인간 주석자들이 ‘거부 응답’으로 라벨링한 안전 데이터를 모았습니다. 이를 통해 LLMs가 악의적인 문서를 처리하는 대신 거부 응답을 학습하도록 하였습니다. LLaMA-2-7B 모델을 사용하여 temperature 0.7로 거부 응답을 생성, 자동으로 필터링하는 과정을 거쳤습니다.

- **Performance Highlights**: Llama2 모델은 Llama1 모델에 비해 안전성과 유용성 간의 매우 뛰어난 균형을 보여주었습니다. 실험 결과에 따르면, 요약 태스크를 튜닝한 방어 방법이 가장 효과적이며, 다른 NLP 태스크들 간 전이 가능성이 높은 방어 성능을 보였습니다. 또한, 과적합을 방지하기 위해 적절한 수의 방어 예제를 선택하는 것이 중요하며, 전체 3에포크 동안 학습을 진행했습니다.



### RAEE: A Training-Free Retrieval-Augmented Early Exiting Framework for Efficient Inferenc (https://arxiv.org/abs/2405.15198)
- **What's New**: 이 논문은 대규모 언어 모델 추론의 비용을 줄이는 새로운 접근법인 RAEE를 제안합니다. RAEE는 사전 훈련 없이 유사 데이터의 정보를 활용하여 추론을 가속화하는 구조입니다. 기존의 내장된 분류기를 훈련해야 하는 방식과 달리, RAEE는 사전 구축된 검색 데이터베이스를 사용하여 유사한 데이터의 종료 정보를 이용합니다.

- **Technical Details**: RAEE는 종료 문제를 분포 예측 문제로 모델링합니다. 유사 데이터의 종료 정보를 이용해 분포를 근사하며, 이를 통해 종료 계층을 예측합니다. 데이터베이스는 훈련 데이터에서 종료 정보를 수집해 구축하며, 추론 시에는 top-k 가장 가까운 이웃 데이터를 검색하여 종료 계층을 예측합니다. RAEE는 분류기나 모델의 추가 학습 없이 효과적으로 종료 계층을 예측합니다.

- **Performance Highlights**: RAEE는 8개의 분류 작업에서 최신의 zero-shot 성능을 달성하며, 추론 속도를 크게 향상시킵니다. 실험 결과 RAEE는 기존 방법들 대비 뛰어난 성능을 보였으며, 코드와 데이터는 공개되어 있습니다.



### An Evaluation of Estimative Uncertainty in Large Language Models (https://arxiv.org/abs/2405.15185)
- **What's New**: 이 연구는 대형 언어 모델(LLM)들이 사람들과 비슷한 추산적 불확실성 표현을 어떻게 사용하는지 조사합니다. 특히 GPT-4와 ERNIE-4 같은 모델들을 탐구하며, 성별과 언어(영어와 중국어) 상황에서도 성능을 비교합니다.

- **Technical Details**: 연구는 다양한 WEP(Estimative Probability의 말)과 대조하여 LLM들이 인간의 추산 불확실성을 얼마나 잘 반영하는지 평가했습니다. 주요 LLM들(GPT-3.5-Turbo, GPT-4, Llama-2-7B, Llama-2-13B, ERNIE-4)을 사용해 다양한 컨트롤과 시나리오에서 실험을 수행했습니다.

- **Performance Highlights**: GPT-3.5와 GPT-4는 일부 상황에서 인간의 예상과 잘 맞았으나, 17개의 WEP 중 각각 11개와 12개 WEP에서 상당한 차이를 보였습니다. 성별이 관련된 시나리오에서는 모든 LLM들이 인간의 예상과 더 큰 차이를 나타냈습니다. 또한, 중국어와 영어로 진행된 실험에서 ERNIE-4는 GPT-3.5와 비교해 많은 WEP에서 다른 결과를 보였으며, 고급 프롬프트 방법을 사용해도 GPT-4의 성능 향상을 이끌어내는 데 실패했습니다.



### VB-LoRA: Extreme Parameter Efficient Fine-Tuning with Vector Banks (https://arxiv.org/abs/2405.15179)
- **What's New**: 이번 연구에서는 대형 언어 모델(Large Language Models)의 채택이 증가하고 사용자 또는 작업별 모델 커스터마이제이션에 대한 필요성이 커짐에 따라, 파라미터 효율적인 미세 조정(Parameter-Efficient Fine-Tuning, PEFT) 방법인 저차원 적응(LoRA)과 그 변형들이 발생시키는 저장 및 전송 비용을 줄이기 위해, '분할 및 공유(divide-and-share)' 패러다임을 소개합니다. 이 패러다임은 벡터 뱅크(vector bank)를 통해 파라미터를 전역적으로 공유하여 행렬 차원, 모듈과 레이어의 저차원 분해 장벽을 허무는 것을 목표로 합니다. VB-LoRA는 모든 저차원 행렬을 공유된 벡터 뱅크에서 차별화 가능한 상위-k 혼합 모듈을 통해 구성하는 방식으로, 기존 PEFT 방법들과 비교해 뛰어난 성능을 유지하면서도 극도의 파라미터 효율성을 달성합니다.

- **Technical Details**: VB-LoRA는 특히 LoRA 기법의 변형으로, 공통의 벡터 뱅크(vector bank)와 차별화 가능한 상위-k 혼합 모듈을 이용해 모든 저차원 행렬을 구성합니다. CoLA 데이터셋에 대한 시각화를 위해, RoBERTa-large 모델의 24개 레이어에서 30개의 벡터를 가진 벡터 뱅크를 사용하여 쿼리 및 가치(Value) 모듈을 사전 훈련했으며, 최종 선택된 벡터들이 초기 선택과 크게 다르다는 점에서 훈련 역학을 확인할 수 있었습니다. 또한, 상위-k 가중치의 합을 통해 각 벡터가 특정 레이어에서 더 많이 선택되는 경향을 확인했습니다.

- **Performance Highlights**: 광범위한 실험을 통해 VB-LoRA의 우수성을 입증하였으며, 자연어 이해, 생성, 지시 조정 등의 작업에서 탁월한 성능을 보였습니다. 특히 Llama2-13B 모델의 미세 조정에서, VB-LoRA는 LoRA 스토리지의 0.4%만 사용했음에도 불구하고 더 나은 결과를 달성했습니다.



### A Solution-based LLM API-using Methodology for Academic Information Seeking (https://arxiv.org/abs/2405.15165)
Comments:
          22 pages, 13 figures

- **What's New**: 새로운 연구인 SoAy는 학술 정보 찾기에서 LLMs를 효율적으로 활용하는 방법론을 소개합니다. 이는 복잡한 API 결합문제를 해결하기 위해 미리 구성된 API 호출 순서를 사용하는 혁신적인 접근 방식입니다.

- **Technical Details**: SoAy는 API 호출 계획을 생성하고, 그 계획에 따라 실행 가능한 코드를 생성하는 방식으로 작동합니다. 이러한 방식은 LLM이 학술 API 간의 복잡한 관계를 명확히 이해할 수 있게 도와줍니다. 이를 위해 AMiner API를 기준으로 3,960개의 (query, solution, code) 트리플렛을 자동 생성하여 모델을 학습시켰으며, SoAyBench라는 평가 벤치마크와 SoAyEval 평가 방법론을 도입했습니다.

- **Performance Highlights**: 실험 결과에 따르면, SoAy는 기존 최첨단 LLM API 기반 방법들에 비해 34.58-75.99%의 성능 향상을 보였습니다. 이는 SoAy가 단순한 학술 검색 엔진보다 학술 데이터를 더욱 빠르고 정확하게 처리할 수 있음을 입증합니다.



### Machine Unlearning in Large Language Models (https://arxiv.org/abs/2405.15152)
Comments:
          10 pages

- **What's New**: 이 논문은 대형 언어 모델(LLMs)에서 기계적 '언러닝'(unlearning)을 통해 윤리적, 프라이버시 및 안전 기준에 맞추는 새로운 방법론을 소개합니다. 특히 해로운 응답 및 저작권이 있는 콘텐츠를 제거하는 데 주력하고 있습니다. 이를 통해 LLM이 더 윤리적이고 안전하게 작동하도록 돕는 이중 접근법을 제안합니다.

- **Technical Details**: 이 연구에서는 그레이디언트 어센트(Gradient Ascent) 알고리즘을 사용하여 불필요한 정보를 선택적으로 지우거나 수정합니다. PKU 데이터셋에서 그레이디언트 어센트를 적용하여 해로운 응답을 줄였고, TruthfulQA 데이터셋을 활용해 이전 지식을 유지합니다. 또한 '반지의 제왕' 코퍼스로부터 커스텀 데이터셋을 제작하여 LoRA: Low-Rank Adaptation 기법을 통해 저작권이 있는 콘텐츠를 제거하였습니다.

- **Performance Highlights**: PKU 데이터셋에서 그레이디언트 어센트를 사용해 해로운 응답이 75% 감소했고, 반지의 제왕 콘텐츠 제거에서 저작권이 있는 자료를 상당히 줄였습니다. 새로운 평가 기법도 제안되었으며, 해로운 콘텐츠 언러닝의 효과를 평가하는 원리로 작동합니다.



### Efficient Biomedical Entity Linking: Clinical Text Standardization with Low-Resource Techniques (https://arxiv.org/abs/2405.15134)
- **What's New**: 이번 연구에서는 의료용 텍스트의 다양한 표면형을 표준화하고 표준 코드를 매핑하는 효율적이고 자원 절감형의 zero-shot 바이오메디컬 엔티티 링크 솔루션을 제안합니다. 이 접근 방식은 기존의 기법들에 비해 훈련 데이터와 자원 소비를 크게 줄이고 동시에 유의미한 성능을 유지합니다.

- **Technical Details**: 본 연구는 UMLS(UMLS: Unified Medical Language System)와 같은 의료 온톨로지의 엔티티와 연관된 동의어 쌍을 학습하는데 중점을 두고 있습니다. MiniLM(Wang et al. 2020) 모델을 활용하여 엔티티 동의어 집합에서 대조 학습을 수행하고, 이를 통해 384 크기의 임베딩을 생성합니다. 이를 통해 후보 생성 단계를 수행하며, 여러 엔티티가 유사한 점수를 가지는 모호한 경우에는 재랭킹(reranking) 기법을 적용해 해결합니다.

- **Performance Highlights**: Medmentions 데이터셋에서 진행한 평가에서는, 우리의 접근 방식이 zero-shot 및 distant supervised 엔티티 링크 기법들과 유사한 성능을 달성했습니다. 또한, 단순한 검색 성능 외에도 문서 수준의 정량 및 정성 분석을 통해 엔티티 링크 성능의 더 깊은 통찰을 얻었습니다.



### Generalizable and Scalable Multistage Biomedical Concept Normalization Leveraging Large Language Models (https://arxiv.org/abs/2405.15122)
- **What's New**: 본 논문은 소스 발언의 대체 구문을 생성하고 다양한 프롬프팅 방법을 사용하여 UMLS 개념의 후보군을 줄이는 두 단계의 LLM 통합 접근법을 사용하여 생의학 엔터티 표준화(Biomedical Entity Normalization)의 성능을 크게 향상시키는 방법을 탐구합니다. 이 연구는 GPT-3.5-turbo 및 Vicuna-13b와 같은 대형 언어 모델(LLM)을 이용하여 일반적으로 사용되는 규칙 기반 표준화 시스템과 혼합하여 적용하였으며, 그 결과 성능이 크게 향상되었음을 보여줍니다.

- **Technical Details**: 본 연구는 두 가지 주요 LLM, 하나는 독점적(GPT-3.5-turbo), 다른 하나는 오픈 소스 모델(Vicuna-13b)을 사용했습니다. 두 모델 모두 표준화 시스템과 함께 언어 모델을 통합하여 개념 용어 및 텍스트 컨텍스트의 정규화를 실험했습니다. 정규화 성능은 $F_{eta}$와 F1 점수로 측정하였으며, 이때 재현(recall)을 정확도(precision)보다 우선시했습니다. 실험에 사용된 규칙 기반 표준화 시스템은 MetaMapLite, QuickUMLS, BM25가 포함되었습니다.

- **Performance Highlights**: GPT-3.5-turbo를 통합한 결과, MetaMapLite에서 $F_{eta}$ 및 F1 점수가 각각 +9.5와 +7.3만큼 증가했으며, QuickUMLS에서는 +13.9 및 +10.9, BM25에서는 +10.5 및 +10.3만큼 향상되었습니다. Vicuna 모델의 경우 MetaMapLite에서 각각 +10.8 및 +12.2, QuickUMLS에서 +14.7 및 +15, BM25에서 +15.6 및 +18.7만큼 성능이 향상되었습니다. 이를 통해 일반적으로 사용되는 대형 언어 모델(LLM)이 기존의 도구들과 결합하여 표준화 성능을 크게 향상시킬 수 있음을 확인했습니다.



### CHARP: Conversation History AwaReness Probing for Knowledge-grounded Dialogue Systems (https://arxiv.org/abs/2405.15110)
Comments:
          To appear in Findings ACL 2024

- **What's New**: 이번 연구에서는 지식 기반 대화 평가의 충실성(Faithfulness)에 중점을 둔 FaithDial 벤치마크 데이터의 문제점을 분석하고, 새로운 진단 테스트셋인 CHARP를 소개합니다. CHARP는 대화 모델의 환각(Hallucination)과 대화 작업 준수 여부를 평가하는 데 초점을 맞추고 있습니다.

- **Technical Details**: FaithDial 데이터는 대화 역사(dialogue history)를 무시하게 만드는 주석 아티팩트(annotation artifacts)를 포함하고 있어 모델이 대화 역사를 제대로 활용하지 못하게 만듭니다. 이를 확인하기 위해 CHARP를 이용한 실험을 진행하였고, CHARP는 Conversation History AwaReness Probing의 약자로, 대화 역사와 관련된 난이도를 두 가지 버전 (eCHARP, hCHARP)으로 나누어 평가합니다. FaithDial 데이터의 일부 1,080개의 샘플을 기반으로 CHARP가 생성되었습니다.

- **Performance Highlights**: 모델은 CHARP에서 대화 역사를 효과적으로 활용하지 못해 성능이 떨어지는 것으로 나타났습니다. FaithDial의 평가 방법은 이러한 문제를 제대로 포착하지 못하며, FaithDial 주석을 사용한 모델은 대화 역사를 무시하는 성향을 보였습니다. 사람의 평가와 강력한 LLM APIs를 통해 FaithDial 데이터가 모델의 대화 역사 활용에 부정적인 영향을 미친다는 것을 확인했습니다.



### Contrastive and Consistency Learning for Neural Noisy-Channel Model in Spoken Language Understanding (https://arxiv.org/abs/2405.15097)
Comments:
          Accepted NAACL 2024

- **What's New**: 최근 발표된 연구는 자동 음성 인식(ASR) 오류로 인한 문자 불일치를 처리할 수 있는 'Contrastive and Consistency Learning (CCL)' 두 단계 방법을 제안했습니다. CCL은 깨끗한 ASR 전사물과 노이즈가 있는 ASR 전사 물 간의 오류 패턴을 연결하고 두 전사물의 잠재적 특징(consistency of latent features)의 일관성을 강조합니다.

- **Technical Details**: 제안된 CCL 방법은 두 단계로 구성됩니다: 첫째로, 토큰 기반 대조 학습(token-based contrastive learning)은 노이즈가 있는 ASR 전사물의 오류를 대응되는 깨끗한 전사물과 단어 및 발화 토큰 수준에서 일치시키고, 둘째로는 일관성 학습(consistency learning)을 통해 깨끗하고 노이즈가 있는 잠재 특징의 일관성을 강조하여 노이즈 있는 ASR 전사물의 오분류를 방지합니다. 이 접근 방식은 깨끗한 텍스트로 사전 학습된 일반 언어 모델을 활용하는 모듈 기반 접근법으로서 ASR 오류 문제를 효과적으로 해결합니다.

- **Performance Highlights**: SLURP, Timers, FSC, SNIPS 등의 네 가지 벤치마크 데이터셋을 사용한 실험 결과, CCL 방법이 기존 방법보다 더 나은 성능을 보였으며 다양한 노이즈 환경에서도 ASR의 견고성을 개선하는 데 성공했습니다. 특히 SLURP 벤치마크 데이터셋에서 의도 분류 성능이 2.59% 향상되었습니다.



### Eliciting Informative Text Evaluations with Large Language Models (https://arxiv.org/abs/2405.15077)
Comments:
          Accepted by the Twenty-Fifth ACM Conference on Economics and Computation (EC'24)

- **What's New**: 최근 텍스트 기반 보고서 분야에서 피어 예측 메커니즘(peer prediction mechanisms)을 확장하여 고품질 피드백을 장려하는 두 가지 새로운 메커니즘을 소개합니다. 기계 학습 모델을 사용하여 텍스트 평가를 예측하는 Generative Peer Prediction Mechanism(GPPM)과 Generative Synopsis Peer Prediction Mechanism(GSPPM)이 그 예입니다.

- **Technical Details**: GPPM와 GSPPM 메커니즘은 LLMs(large language models)를 예측 도구로 사용하여 한 에이전트의 보고서에서 다른 에이전트의 보고서를 예측합니다. 이 메커니즘들은 LLM 예측이 충분히 정확할 경우, 높은 노력과 진실된 보고를 유도할 수 있음을 이론적으로 입증했습니다. 실험에서는 Yelp 리뷰 데이터셋과 ICLR OpenReview 데이터셋을 사용했습니다.

- **Performance Highlights**: 실험 결과, ICLR 데이터셋에서 GPPM과 GSPPM 메커니즘은 사람 작성 리뷰, GPT-4 생성 리뷰, GPT-3.5 생성 리뷰를 질적으로 구별할 수 있었습니다. 특히 GSPPM은 LLM 생성 리뷰에 대해 GPPM보다 더 효과적으로 페널티를 부과합니다.



### Grokked Transformers are Implicit Reasoners: A Mechanistic Journey to the Edge of Generalization (https://arxiv.org/abs/2405.15071)
Comments:
          21 pages, 16 figures. Code and data: this https URL

- **What's New**: 이번 연구는 트랜스포머(transformers)가 암묵지(parametric knowledge)에 대한 암묵적 추론(implicit reasoning)을 학습할 수 있는지, 그리고 그 과정에서 겪는 어려움을 탐구한다. 기존의 가장 강력한 언어 모델들도 이 능력에서 어려움을 겪고 있음에 주목하며, 조합(composition)과 비교(comparison)라는 두 가지 대표적인 추론 유형을 집중적으로 분석하였다.

- **Technical Details**: 트랜스포머 모델을 사용하여 조합과 비교 두 가지 추론 유형에서 일반화 수준을 평가하였다. 모델은 훈련 초과(overfitting)를 넘어서는 확장된 학습을 통해서만 이 기술을 견고하게 습득할 수 있음을 발견하였다. 특히, 조합에 대해서는 out-of-distribution (OOD) 예제에 대해 체계적으로 일반화하지 못하는 반면, 비교에 대해서는 성공적으로 일반화할 수 있음을 확인했다.

- **Performance Highlights**: 성능 실험을 통해, GPT-4-Turbo 및 Gemini-1.5-Pro와 같은 최첨단 언어 모델들이 비매개(memory) 방식에서는 큰 검색 공간을 가진 도전적인 추론 작업에서 실패하지만, 완전히 훈련된 트랜스포머 모델은 거의 완벽한 정확도를 달성할 수 있음을 보여주었다. 이는 트랜스포머의 개선된 구조가 복잡한 추론에서의 강력함을 입증한다.



### Optimizing example selection for retrieval-augmented machine translation with translation memories (https://arxiv.org/abs/2405.15070)
Comments:
          TALN conference, French, 10 pages, 7 figures

- **What's New**: 새로운 논문에서는 검색 강화 기계 번역(Retrieval-augmented Machine Translation) 시스템에서 검색 단계의 향상을 목표로 하고 있습니다. 특히, 업스트림 검색 단계를 개선하기 위해 서브모듈러 함수(submodular functions) 이론을 활용하고, 다중-레벤슈타인 트랜스포머(multi-Levenshtein Transformer)와 같은 고정된 다운스트림 편집 기반 모델을 사용합니다.

- **Technical Details**: 이 논문은 다중-레벤슈타인 트랜스포머(Multi-Levenshtein Transformer)를 사용하여 k개의 예시 문장들을 입력으로 받아 편집을 통해 번역을 생성하는 방법을 제안합니다. 이를 위해 소스 문장의 전체 커버리지를 극대화하는 예시 세트를 찾는 방법을 연구합니다. 서브모듈러 함수 이론을 활용하여 최적의 예시 세트를 식별하는 알고리즘을 탐구하고, 이를 통해 얻을 수 있는 기계 번역 성능 향상을 평가합니다.

- **Performance Highlights**: 다양한 실험을 통해 새로운 커버리지 최적화 알고리즘의 성능을 평가했을 때, 다중-레벤슈타인 트랜스포머 모델에서 번역 성능이 향상되었음을 확인하였습니다. 이 알고리즘들은 매우 괄목할만한 성능 향상을 나타내었습니다.



### Promoting Constructive Deliberation: Reframing for Receptiveness (https://arxiv.org/abs/2405.15067)
- **What's New**: 이 논문에서는 온라인에서 논쟁이 되는 주제에 대해 건설적인 토론을 촉진하기 위해 반대 의견을 수용성을 신호하도록 자동으로 재구성하는 방법을 제안합니다. 심리학, 커뮤니케이션, 그리고 언어학의 연구를 기반으로 여섯 가지 재구성 전략을 식별하고 이를 Reddit 댓글 데이터셋을 사용하여 자동화했습니다. 인간 중심의 실험을 통해, 제안된 프레임워크로 생성된 답변이 원래의 답변보다 더 수용적으로 인식된다는 것을 발견했습니다.

- **Technical Details**: 이 연구는 문장 재구성을 통해 토론을 활성화하고자 하며, 실험에서 Low-Level Strategies, NLP(Natural Language Processing) 기법, 그리고 Reddit 데이터셋을 사용했습니다. 특히, 여섯 가지 구체적인 전략을 사용하여 원래 의미를 유지하면서 답변을 재구성했습니다. 연구 방법은 in-context framework을 사용하여 데이터셋을 구성하고, 자동 측정과 인간 평가를 통해 생성된 답변의 유효성을 평가했습니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크로 생성된 답변은 원래 답변이나 기본적인 수용성 프레임워크보다 훨씬 더 수용적으로 인식되었습니다. 재구성된 답변은 낮은 부정적 감정 반응을 유도하고, 반대 의견에 대한 높은 호기심을 불러일으키며, 사용자가 토론을 중단시키려는 의도로 인식되지 않았습니다. 독성(톡식) 콘텐츠의 경우 재구성의 이점이 더 큰 것으로 나타나, 콘텐츠 검열의 유연성을 높이는 데 기여할 수 있습니다.

- **Applications**: 이 연구는 콘텐츠 모더레이션(content moderation)과 온라인 토론의 질을 향상시키기 위한 여러 적용 사례를 제공합니다. 사용자가 더 건설적인 토론을 하게 만들어, 온라인 커뮤니티의 긍정적인 환경을 조성할 수 있습니다.



### Reframing Spatial Reasoning Evaluation in Language Models: A Real-World Simulation Benchmark for Qualitative Reasoning (https://arxiv.org/abs/2405.15064)
Comments:
          Camera-Ready version for IJCAI 2024

- **What's New**: 새로운 논문에서는 언어 모델(Language Models, LMs)의 질적 공간 추론(Qualitative Spatial Reasoning, QSR) 평가를 위한 새로운 벤치마크를 제시합니다. 이 벤치마크는 현실적인 3D 시뮬레이션 데이터를 기반으로 하며, 다양한 방 배치와 객체들의 공간적 관계를 평가합니다. 이를 통해 기존 벤치마크의 단순화된 시나리오와 모호한 언어 설명 문제를 해결하고자 합니다.

- **Technical Details**: 이 벤치마크는 다양한 토폴로지(topological), 방향(directional), 거리 관계(distance relations)를 포함하며, 여러 관점과 다양한 세부 수준 및 관계 제약 밀도를 반영해 현실 세계의 복잡성을 모방합니다. 또한 논리 기반 일관성 검증 도구를 도입하여 여러 가능한 해법이 존재하는 실제 시나리오와 유사하게 평가를 진행합니다.

- **Performance Highlights**: 최신 언어 모델은 여러 설정에서 뛰어난 공간 추론 능력을 보였지만, 복합적인 다중 단계(multi-hop) 공간 추론과 다양한 관점의 설명을 혼합하여 해석하는 데 어려움을 겪었습니다. 특히 GPT-4가 여러 상황에서 우수한 성능을 보였으나, 이야기의 제약 그래프가 완전해질수록 성능 향상의 경향이 나타났습니다.



### CEEBERT: Cross-Domain Inference in Early Exit BER (https://arxiv.org/abs/2405.15039)
Comments:
          Accepted at ACL 2024

- **What's New**: 최근 발표된 논문에서는 Cross-Domain Inference in Early Exit BERT (CeeBERT)이라는 새로운 온라인 학습 알고리즘을 제안합니다. 이 알고리즘은 중간 레이어에서의 조기 종료를 기반으로 샘플의 추론을 동적으로 결정합니다. CeeBERT는 도메인 특화된 신뢰도(confidence)를 실시간으로 학습하여, 레이블이 없는 데이터에서 최적의 문턱값(threshold)을 학습합니다.

- **Technical Details**: CeeBERT는 기존의 큰 규모의 사전학습된 언어 모델들이 갖고 있는 추론 지연 문제를 해결하기 위해 설계되었습니다. 구체적으로, CeeBERT는 Multi-Armed Bandit 프레임워크를 사용하여 각 중간 레이어에서 발생하는 신뢰도를 기반으로 최적의 문턱값을 학습합니다. 이는 샘플이 마지막 레이어까지 가지 않고도 추론을 완료할 수 있게 함으로써 지연 시간을 단축시킵니다.

- **Performance Highlights**: CeeBERT는 BERT 및 ALBERT 모델을 대상으로 한 5개의 다양한 데이터셋(IMDB, MRPC, SciTail, SNLI, Yelp, QQP)에서 평가되었습니다. 결과적으로, CeeBERT는 거의 성능 손실 없이 인퍼런스 시간을 2배에서 3.5배까지 단축할 수 있음을 보여주었습니다. 즉, 정확도 손실이 0.1%에서 3%로 최소화된 상태에서 효율성을 대폭 향상시켰습니다.



### Aya 23: Open Weight Releases to Further Multilingual Progress (https://arxiv.org/abs/2405.15032)
- **What's New**: 최근 발표된 아야 모델(Üstün 등, 2024)을 기반으로 다국어 언어 모델인 아야 23이 소개되었습니다. 아야 23은 23개의 언어를 지원하며, 다국어 학습 모델의 성능을 세계 인구의 절반 이상으로 확장시키는 것을 목표로 합니다. 아야 101 모델이 101개의 언어를 포함한 반면, 아야 23은 적은 수의 언어에 더 많은 용량을 투입하여 성능을 최적화하는 접근 방식을 취합니다.

- **Technical Details**: 아야 23 모델군은 Cohere의 Command R 모델을 기반으로 하여 23개의 언어를 포함한 데이터로 사전 훈련되었습니다. 주요 기술적 구성 요소로는 평행 주의 및 FFN 레이어, SwiGLU 활성화 함수, 무 바이어스, RoPE(Rotary Positional Embeddings), BPE 토크나이저(Tokenizer), 그룹드 쿼리 어텐션(Grouped Query Attention, GQA) 등이 포함되어 있습니다. 모든 기본 모델은 Jax 기반 Fax 분산 훈련 프레임워크를 사용해 TPU v4 칩에서 훈련되었습니다.

- **Performance Highlights**: 아야 23 모델은 8B와 35B 두 가지 크기로 제공되며, 평가 작업 전반에서 최고의 결과를 달성했습니다. 특히 아야 23-35B 모델은 모든 평가 작업과 언어에서 가장 높은 성능을 기록했으며, 아야 23-8B 모델은 소비자급 하드웨어에서도 높은 성능을 보였습니다. 아야 101 모델과 비교하여 판별 작업에서 최대 14%, 생성 작업에서 최대 20%, 다국어 MMLU에서 최대 41.6% 개선되었습니다. 또한, 수학적 추론 능력에서는 6.6배의 향상을 보였습니다.



### AGRaME: Any-Granularity Ranking with Multi-Vector Embeddings (https://arxiv.org/abs/2405.15028)
- **What's New**: 이 논문은 다양한 수준의 세분화된 랭킹을 가능하게 하는 AGRaME (Any-Granularity Ranking with Multi-vector Embeddings) 방법을 제안합니다. 이 방법은 멀티-벡터 임베딩(multi-vector embeddings)을 활용하여 단일 수준에서의 인코딩을 유지하면서 다양한 세분화 수준에서 랭킹을 수행할 수 있습니다.

- **Technical Details**: 기존의 싱글 벡터(single-vector) 방식과는 달리, 멀티 벡터 방식은 쿼리와 패시지 각각의 토큰 수준에서 상호작용을 캡처하여 더욱 정밀한 랭킹 성능을 제공합니다. 이를 위해 멀티-그라뉼러 대조 손실(multi-granular contrastive loss)을 제안하여 트레이닝을 향상시킵니다. 이는 쿼리와 패시지 간의 토큰 매칭 점수를 계산하고, 이를 종합하여 최종 쿼리-패시지 관련성 점수를 생성합니다.

- **Performance Highlights**: AGRaME는 패시지 레벨 인코딩을 유지하면서도 문장 및 프로포지션 레벨에서 우수한 랭킹 성능을 보였습니다. 특히, open-domain question-answering와 같은 응용 분야에서 세분화된 랭킹이 더 나은 성과를 나타냈습니다. 또한, text generation에서 post-hoc citation addition을 통해 기존 방법을 능가하는 결과를 얻었습니다.



### Extracting Prompts by Inverting LLM Outputs (https://arxiv.org/abs/2405.15012)
- **What's New**: 이번 연구에서는 언어 모델(inversion)의 문제를 다루고 있습니다. 주어진 언어 모델의 출력값을 통해 그 출력을 생성한 프롬프트(prompt)를 추출하는 방법을 개발하였습니다. output2prompt라는 새로운 블랙박스(black-box) 기법을 제안하며, 이는 모델의 로짓(logits)에 접근하지 않고, 적대적인(adversarial) 또는 감옥 탈출(jailbreaking)과 같은 쿼리를 사용하지 않습니다.

- **Technical Details**: output2prompt는 기존 연구와 달리 일반 사용자 쿼리의 출력값만 필요로 합니다. 메모리 효율성을 높이기 위해 새로운 희소 인코딩 기법(sparse encoding technique)을 사용합니다. 이러한 방법을 통해 다양한 사용자 및 시스템 프롬프트에서 output2prompt의 효과를 측정하였으며, 서로 다른 대형 언어 모델(LLMs) 간 제로샷 전이(zero-shot transferability)를 입증하였습니다.

- **Performance Highlights**: output2prompt는 접근성과 메모리 효율성 면에서 큰 개선을 보여주며, 다양한 언어 모델에서 성공적으로 프롬프트를 추출할 수 있는 능력을 증명하였습니다. 특히 제로샷 전이 가능성을 통해 단일 모델에 국한되지 않는 적용성을 보여주었습니다.



### RE-Adapt: Reverse Engineered Adaptation of Large Language Models (https://arxiv.org/abs/2405.15007)
- **What's New**: RE-Adapt는 대형 언어 모델(large language models)을 새로운 도메인에 적응시키면서 기존의 지침 기반 튜닝(instruction-tuning)을 유지하는 접근 방법입니다. 핵심은 추가적인 데이터나 훈련 없이 지침 기반 모델이 사전 훈련된 모델(base model)에서 습득한 것을 역공학(reverse engineer)하여 분리하는 것입니다.

- **Technical Details**: 이 접근법은 지침 기반 모델과 사전 훈련된 모델 간의 차이를 역공학하여 어댑터(adapter)를 생성합니다. 이후 사전 훈련된 모델을 새로운 도메인에 맞게 미세 튜닝(fine-tune)하고, 역공학된 어댑터를 사용하여 지침 준수 성능을 다시 적응(readapt)시킵니다. 또한, 저순위 변형(low-rank variant)인 LoRE-Adapt를 포함하여 더 높은 성능을 달성할 수 있습니다.

- **Performance Highlights**: RE-Adapt와 LoRE-Adapt는 여러 인기 있는 대형 언어 모델(LLMs)과 데이터셋을 대상으로 다른 미세 튜닝 방법보다 우수함을 입증했습니다. 이는 리트리벌 기반 생성(retrieval-augmented generation)과 병행하여 사용할 때도 성능이 뛰어납니다.



### Linking In-context Learning in Transformers to Human Episodic Memory (https://arxiv.org/abs/2405.14992)
- **What's New**: 이 논문은 Transformer 모델의 어텐션 헤드(attention heads)와 인간의 삽화적 기억(episodic memory) 사이의 유사성을 탐구합니다. 특히, Transformer 기반 큰 언어 모델(LLMs)의 문맥 내 학습(in-context learning, ICL)을 담당하는 'induction heads'에 주목합니다. 연구 결과, induction heads와 인간의 삽화적 기억모델인 CMR(Contextual Maintenance and Retrieval) 모델 사이에 행동적, 기능적, 그리고 기계적인 유사성이 존재함을 보여줍니다. 이는 Transformer 모델과 인간 기억 시스템 간의 평행 관계를 밝히며 두 연구 분야에 중요한 통찰을 제공합니다.

- **Technical Details**: Transformer 모델의 어텐션 헤드, 특히 induction heads가 어떻게 인간의 삽화적 기억 모델(CMR)과 유사하게 작동하는지를 분석합니다. LLMs를 이용한 실험에서 induction heads가 중간 모델 레이어에서 자주 출현하며, 이들의 행동이 인간의 기억 편향과 유사하게 나타납니다. 이를 통해 인공지능 모델의 기계적 메커니즘과 인간 기억의 유사성을 분석하고자 합니다.

- **Performance Highlights**: 검토 결과, induction heads가 인간의 삽화적 기억에서 관찰되는 기억 편향과 유사한 방식으로 작동함을 발견했습니다. 이러한 유사성은 Transformer 모델이 제공하는 문맥 정보를 사용하여 새로운 작업을 즉시 수행할 수 있는 ICL 능력과 관련이 있습니다. 연구는 Transformer 모델이 어떻게 인간 기억과 유사한 방식으로 작동할 수 있는지를 설명하며, 더 나은 모델 개발 및 AI 안전 연구에 중요한 기여를 합니다.



### Data Augmentation Method Utilizing Template Sentences for Variable Definition Extraction (https://arxiv.org/abs/2405.14962)
- **What's New**: 이번 연구에서는 과학 및 기술 논문에서 변수 정의를 추출하기 위한 새로운 방법을 제안합니다. 기존의 변수 정의 추출 기법은 분야마다 성능 차이가 크고, 각 분야별로 학습 데이터를 준비하는 데에 큰 비용이 듭니다. 이를 해결하기 위해 본 연구에서는 템플릿 문장과 학습 데이터의 변수-정의 쌍으로 새로운 정의 문장을 생성하는 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 템플릿 문장을 준비하고, 이를 학습 데이터의 변수-정의 쌍과 결합하여 새로운 정의 문장을 생성합니다. 템플릿 문장은 주로 "[VAR_1]은 [DEF_1]로 정의된다"와 같은 형태입니다. 이 새로운 데이터를 사용하여 모델을 학습시킵니다. 이번 연구에서는 이 방법을 화학 공정 관련 논문에서 테스트했습니다.

- **Performance Highlights**: 제안된 방법을 사용하여 생성된 정의 문장을 학습한 모델은 89.6%의 높은 정확도를 달성했으며, 기존 모델을 능가하는 성능을 보여주었습니다. 이는 데이터 증가를 통해 변수 정의 추출 성능을 대폭 향상시킬 수 있음을 입증합니다.



### DETAIL: Task DEmonsTration Attribution for Interpretable In-context Learning (https://arxiv.org/abs/2405.14899)
- **What's New**: 이번 연구에서는 In-context learning(ICL)의 새로운 해석 방법을 제안합니다. ICL은 일반 텍스트를 사전 학습한 트랜스포머 기반의 언어 모델이 몇 가지 '작업 시연'(task demonstrations)을 통해 특정 작업을 빠르게 학습할 수 있게 하여 모델의 유연성과 범용성을 크게 증가시킵니다. 이번 연구에서는 이러한 새로운 패러다임을 이해하기 위한 영향 함수 기반의 기법, DETAIL을 제안합니다.

- **Technical Details**: DETAIL 기법은 ICL의 독특한 특성을 고려하여 개발되었으며, 내부 옵티마이저(internal optimizer)를 통해 트랜스포머가 컨텍스트에서 학습하는 최근 연구 관점에서 출발했습니다. 제안된 기법은 시연 속성(demonstration attribution)을 효과적으로 수행하면서도 계산 효율성이 높다는 점을 실증적으로 검증했습니다.

- **Performance Highlights**: DETAIL 기법을 활용하여 모델 성능을 향상시키기 위해 시연 순서를 재정렬하거나(curating) 개선할 수 있음을 보였습니다. 또한, 화이트 박스(white-box) 모델에서 얻은 속성 점수가 블랙 박스(black-box) 모델로도 전이되어 모델 성능을 향상시키는 데 효과적이라는 광범위한 적용 가능성을 실험을 통해 증명했습니다.



### Enhancing Adverse Drug Event Detection with Multimodal Dataset: Corpus Creation and Model Developmen (https://arxiv.org/abs/2405.15766)
- **What's New**: 기존의 ADE(Adverse Drug Events, 약물 이상 반응) 탐지 연구는 주로 텍스트 기반의 방법론에 집중했지만, 본 논문에서는 텍스트와 시각적 정보(visual cues)를 결합한 새로운 다중모드(MultiModal) ADE 탐지 데이터셋(MMADE)을 소개합니다. 이 데이터셋은 ADE 관련 텍스트 정보와 시각적 보조 자료를 통합하여 의료 이미지를 설명하는 프레임워크를 제공합니다. 이를 통해 의료 전문가들이 시각적으로 ADE를 식별하는 데 도움을 줍니다.

- **Technical Details**: 본 연구에서는 LLMs(Large Language Models)과 VLMs(Vision Language Models)의 기능을 활용하여 ADE를 탐지하는 프레임워크를 소개합니다. 이에 사용된 데이터셋 MMADE는 1500개의 약물 및 부작용 관련 텍스트와 이에 상응하는 이미지로 구성되어 있습니다. 실험에서는 InstructBLIP 모델을 사용했으며, BLIP 및 GIT 모델과 비교하여 세밀한 튜닝을 통해 개선된 성능을 확인했습니다.

- **Performance Highlights**: MMADE 데이터셋을 활용하여 시각적 요소를 통합했을 때 ADE 탐지 성능이 현저히 향상됨을 입증했습니다. 제안된 방법론은 약물 이상 반응 분류, 텍스트 생성, 요약 작업에서 유망한 결과를 나타냈고, 환자 안전성 향상, ADE 인식 증대, 그리고 의료 접근성 개선에 기여할 수 있는 잠재력을 가집니다.



### Optimizing Large Language Models for OpenAPI Code Completion (https://arxiv.org/abs/2405.15729)
- **What's New**: 최근 소프트웨어 개발 분야에서 대형 언어 모델(LLMs)의 발전과 코드 생성 작업에의 활용이 큰 변화를 불러일으키고 있습니다. 이 연구는 GitHub Copilot의 OpenAPI 완성 성능을 평가하고, Meta의 오픈 소스 모델인 Code Llama를 활용한 작업-특화 최적화를 제안합니다.

- **Technical Details**: Code Llama 모델의 성능을 분석하기 위해 의미 인식 OpenAPI 완성 벤치마크를 제안하였으며, 다양한 프롬프트 엔지니어링과 모델 미세 조정(fine-tuning) 기법의 영향을 실험을 통해 분석했습니다. 특히 이 연구는 Code Llama의 성능을 GitHub Copilot과 비교하여 평가합니다.

- **Performance Highlights**: 미세 조정된 Code Llama 모델은 GitHub Copilot에 비해 55.2% 더 높은 정확성을 보였습니다. 이는 상용 솔루션의 Codex 모델보다 25배 적은 파라미터를 사용함에도 불구하고 이루어진 성과입니다. 또한, 본 연구에서는 코드 인필링 트레이닝 기법의 개선도 제안했습니다.



### VDGD: Mitigating LVLM Hallucinations in Cognitive Prompts by Bridging the Visual Perception Gap (https://arxiv.org/abs/2405.15683)
Comments:
          Preprint. Under review. Code will be released on paper acceptance

- **What's New**: 최근 대형 비전-언어 모델(Large Vision-Language Models, LVLMs)에 대한 관심이 증가하면서 이는 사실과 일치하지 않는 텍스트 생성의 문제인 '환상(hallucination)' 문제를 해결해야 할 필요성이 생겼습니다. 이번 연구에서는 LVLM의 환상 문제에 대한 심층 분석을 통해 새로운 통찰을 제공하고, 이를 해결하기 위한 Visual Description Grounded Decoding (VDGD) 방법을 제안합니다. 또한 LVLM의 인지 능력을 평가하기 위한 VaLLu 벤치마크도 함께 제안했습니다.

- **Technical Details**: 분석 결과, 기존 연구는 주로 시각적 인식(VR) 프롬프트 예를 들어 이미지 설명을 요구하는 프롬프트에 집중해왔으며, 인지 프롬프트(추론 및 정보 검색 등 추가적인 기술이 필요한 프롬프트)관련 환상 감소에는 상대적으로 적은 진전이 있었습니다. LVLM의 시각적 인식에 있어서는 정확한 반면, 인지 프롬프트에 대한 반응에서는 여전히 환상이 발생함을 발견했습니다. 이를 극복하기 위해, VDGD 방법을 제안하였습니다. VDGD는 이미지 설명을 생성하고 이를 명령어의 앞부분에 추가한 후, 자동 회귀 디코딩(auto-regressive decoding) 동안 설명에 대한 KL-발산(KL-Divergence, KLD)을 활용하여 가장 적합한 단어를 선택하는 방법론입니다.

- **Performance Highlights**: 실험 결과, VDGD 방법이 다른 기준 방법들에 비해 환상 감소에 있어 큰 개선을 보였습니다. 다양한 벤치마크와 LVLM을 대상으로 한 실험에서 VDGD의 성능 향상을 확인할 수 있었습니다. 또한, LVLM의 인지 능력을 종합적으로 평가하는 VaLLu 벤치마크를 제안했으며, 이 벤치마크 역시 다양한 성능 평가 지표를 통해 LVLM의 환상 문제를 효과적으로 평가할 수 있음을 보여줍니다.



### M4U: Evaluating Multilingual Understanding and Reasoning for Large Multimodal Models (https://arxiv.org/abs/2405.15638)
Comments:
          Work in progress

- **What's New**: 다국어 멀티모달 추론(multilingual multimodal reasoning)에서 새로운 벤치마크인 M4U가 도입되었습니다. M4U는 과학, 공학, 의료 분야의 16개의 하위 분야에 걸쳐 64개 학문을 다루는 8,931개의 샘플을 포함하고 있으며, 중국어, 영어, 독일어로 제공됩니다. 이는 기존의 벤치마크들이 다국어 멀티모달 모델 간의 성능 차이를 구분하기 어려운 문제를 해결하기 위해 고안되었습니다.

- **Technical Details**: M4U는 총 8,931개의 다지선다 형태의 질문으로 구성되어 있으며, 샘플들은 주로 온라인 비디오 강의의 퀴즈와 대학 시험에서 수집되었으며, 일부(35%)는 교재를 기반으로 작성되었습니다. 데이터는 주로 이미지와 텍스트가 혼합된 문서로 구성되며, 멀티모달 및 다국어 추론 능력을 평가하기 위한 복잡한 로직과 도메인 지식을 요구합니다. 데이터 셋의 대부분은 독립적으로 작성되었으며, 언어 별 데이터 오염과 난이도의 불균형을 최소화했습니다.

- **Performance Highlights**: 현존 최고 성능의 GPT-4o 모델이 M4U 벤치마크에서 평균 47.6%의 정확도를 기록했습니다. 이는 M4U가 기존 벤치마크보다 도전적인 평가 기준을 제공하는 것을 시사합니다. 또한, 주요 LMM들이 언어별로 큰 성능 차이를 보이며, 특히 다국어 멀티모달 질문에서 성능 저하가 두드러졌습니다. 예를 들어 InstructBLIP Vicuna-7B 모델은 영어 섹션에서 29.8%의 정확도를 보인 반면, 중국어와 독일어 섹션에서는 각각 13.7%와 19.7%의 정확도를 기록했습니다.



### Certifiably Robust RAG against Retrieval Corruption (https://arxiv.org/abs/2405.15556)
- **What's New**: 새로운 논문에서는 retrieval-augmented generation (RAG)에 대한 공격 방어를 위한 첫 번째 프레임워크인 RobustRAG을 제안합니다. 이 프레임워크는 악의적인 패시지(패시지) 주입으로 인한 부정확한 응답을 방어하기 위해 설계되었습니다. RobustRAG은 '격리 후 집계(isolate-then-aggregate)' 전략을 적용하여 각 패시지로부터 독립적으로 LLM 응답을 얻고, 이를 안전하게 집계합니다.

- **Technical Details**: RobustRAG의 핵심은 악성 패시지가 다른 정상 패시지의 LLM 응답에 영향을 미치지 않도록 각 패시지로부터 응답을 독립적으로 계산한 후, 이 응답들을 안전하게 집계하는 것입니다. 이를 위해 키워드 기반 집계 및 디코딩 기반 집계라는 두 가지 알고리즘을 설계하였습니다. 특히 RobustRAG은 인증 가능한 강건성을 달성할 수 있으며, 공격자가 방어 체계를 완벽히 알고 있는 경우에도 일정 수의 악성 패시지만 주입된다면 항상 정확한 응답을 반환할 수 있습니다.

- **Performance Highlights**: 우리는 RobustRAG을 다양한 데이터셋(RealtimeQA, NQ, Bio)과 LLM(Mistral, Llama, GPT)에서 평가하여 그 효과성과 일반성을 입증했습니다. RobustRAG은 질문 응답과 장문 텍스트 생성과 같은 다양한 지식 집약적 작업에 적용할 수 있습니다.



### Learning Beyond Pattern Matching? Assaying Mathematical Understanding in LLMs (https://arxiv.org/abs/2405.15485)
- **What's New**: 이 논문은 대형 언어 모델(LLM)을 과학적 보조 도구로 사용하려는 최근 트렌드에 따라, LLM의 수학적 문제 해결 능력을 평가합니다. Neural Tangent Kernel (NTK)에서 영감을 받아, NTKEval이라는 평가 방법을 제안하고 다양한 수학 데이터를 통해 LLM의 확률 분포 변화를 분석합니다.

- **Technical Details**: LLM이 수학적 문제를 얼마나 잘 이해하고 해결할 수 있는지를 평가하기 위해서, 두 가지 학습 방법을 사용했습니다: 인-컨텍스트 학습(In-context learning)과 지시 조정(Instruction-tuning). 인-컨텍스트 학습은 문제의 깊은 구조(Deep structure)를 이해할 수 있는지 여부를 평가하고, 지시 조정은 표면 변화(Surface changes)에 의해서만 성능이 향상되는지를 탐구했습니다.

- **Performance Highlights**: 분석 결과, 인-컨텍스트 학습을 통해 LLM이 깊은 구조와 표면 구조를 구별하며 관련된 수학 기술을 효과적으로 사용할 수 있는 능력을 보여주었습니다. 반면, 지시 조정은 서로 다른 데이터로 훈련해도 유사한 성능 변화를 보여주었으며, 이는 형식 매칭(Format matching)에 의한 것이지 도메인 이해에 의한 것은 아니었습니다.



### Leveraging Large Language Models for Semantic Query Processing in a Scholarly Knowledge Graph (https://arxiv.org/abs/2405.15374)
Comments:
          for the associated repository, see this http URL

- **What's New**: 호주국립대학교(ANU) 컴퓨터과학 연구자들이 발표한 연구 작품에 대한 포괄적인 정보를 제공하는 새로운 의미 기반 쿼리 처리 시스템을 개발했습니다. 이 시스템은 대규모 언어 모델(LLMs)과 ANU 학술 지식 그래프(ASKG)를 통합하여, 연구 관련 데이터의 정교한 표현과 효율적인 쿼리 처리를 가능하게 합니다.

- **Technical Details**: 본 연구는 두 가지 혁신적인 방법을 제안합니다: Deep Document Model(DDM)과 KG 기반 쿼리 최적화(KGQP). DDM은 학술 논문의 계층 구조와 의미 관계를 세밀하게 표현하며, KGQP는 대규모 학술 지식 그래프(KG)에서 최적의 복잡한 쿼리 처리를 보장합니다. 이를 위해 자동화된 LLM-SPARQL 혼합 기법을 사용하여 ASKG에서 관련 사실과 텍스트 노드를 효율적으로 검색합니다.

- **Performance Highlights**: 초기 실험 결과, 제안된 프레임워크는 기존 방법에 비해 정확도와 쿼리 효율성 측면에서 우수한 성능을 보여줍니다. 이는 학술 지식 관리와 발견에 혁신적인 변화를 예고하며, 연구자들이 문서에서 지식을 보다 효과적으로 습득하고 활용할 수 있게 도와줍니다.



### Pipeline Parallelism with Controllable Memory (https://arxiv.org/abs/2405.15362)
- **What's New**: 이 논문에서는 파이프라인 스케줄(Pipeline Schedule)을 구성하는 새로운 프레임워크를 제안하며, 이를 통해 메모리 효율적인 빌딩 블록(Building Block) 패밀리를 소개합니다. 제안된 방법은 기존의 1F1B 방법론보다 피크 활성화 메모리(Peak Activation Memory)를 절반으로 줄이면서도 효율을 유지하거나, 3분의 1로 줄이면서도 유사한 처리량을 보장합니다.

- **Technical Details**: 파이프라인 스케줄은 기본적으로 빌딩 블록의 반복으로 구성됩니다. 빌딩 블록의 수명(Lifespan)이 파이프라인 스케줄의 피크 활성화 메모리를 결정합니다. 이 논문에서는 메모리가 비효율적인 기존의 스케줄을 비판하며, 메모리를 제어하여 더 효율적인 빌딩 블록을 제안합니다. 주요 방법은 빌딩 블록의 수명을 줄여, 피크 활성화 메모리를 감소시키는 것입니다. 이러한 접근방식은 파이프라인 버블(Pipeline Bubbles)을 거의 제거하면서도 동일한 활성화 메모리를 유지합니다.

- **Performance Highlights**: 순수한 파이프라인 병렬 처리 설정에서, 제안된 방법은 처리량 측면에서 1F1B보다 7%에서 55%까지 뛰어넘는 성능을 보입니다. 하이브리드 병렬 처리 시나리오에서는, 큰 언어 모델에 대해 1F1B 기준 처리량에서 16%의 개선을 보였습니다.



### Towards Understanding How Transformer Perform Multi-step Reasoning with Matching Operation (https://arxiv.org/abs/2405.15302)
- **What's New**: 이번 연구는 대형 언어 모델(Large Language Models, LLMs)이 복잡한 추론 작업, 특히 수학 문제 해결에서 어려움을 겪고 있는 문제를 다룹니다. 연구진은 Transformer 모델의 다단계 추론(멀티스텝 리즈닝) 메커니즘을 분석하여 이를 개선하기 위한 새로운 접근 방식을 제안했습니다. 특히, 작은 초기화(small initialization)와 LayerNorm 이후 적용 방식이 모델의 매칭 메커니즘을 강화한다고 발견했습니다. 또한, 직교 잡음(orthogonal noise)을 추가하여 모델의 추론 능력을 향상시키는 방법도 제시하고 있습니다.

- **Technical Details**: 이 연구에서는 Transformer 모델이 다단계 추론을 수행하는 방식에 대해 심층적으로 조사합니다. 세 가지 유형의 다단계 추론 데이터셋을 설계하고 모델의 내부 정보 흐름을 분석했습니다. 모델의 매칭 능력을 측정하기 위해 매칭 매트릭스(matching matrix) 개념을 도입했습니다. 연구 결과, 초기화 방법과 LayerNorm의 위치가 모델의 매칭 능력에 크게 영향을 미친다는 것을 발견했습니다. 또한, 직교 잡음을 추가하면 Transformer가 추론 작업을 더 잘 학습할 수 있게 됩니다.

- **Performance Highlights**: Transformer 모델은 다단계 추론이 필요한 작업에서 매칭 작업을 통해 성과를 거두었습니다. 특히, 작은 초기화와 post-LayerNorm을 활용하면 모델이 이러한 작업을 더 효과적으로 학습할 수 있게 됩니다. 추가적으로, Transformer는 여러 개의 추론 단계를 동시에 수행할 수 있으며, 이는 모델의 깊이에 비례해 추론 능력이 기하급수적으로 성장할 수 있음을 시사합니다.



### DEEM: Diffusion Models Serve as the Eyes of Large Language Models for Image Perception (https://arxiv.org/abs/2405.15232)
Comments:
          25 pages

- **What's New**: DEEM은 이미지 인코더로 CLIP-ViT와 같은 기존의 모델 대신 균질 모델링을 위해 디퓨전 모델의 생성적 피드백을 활용하여 대규모 멀티모달 모델의 이미지 인식을 개선하는 새로운 접근법을 제안합니다.

- **Technical Details**: DEEM은 기존의 이미지 인코더가 다운스트림 작업에 관련된 특성만을 인코딩하는 문제를 해결하기 위해 디퓨전 모델을 추가적인 '눈'으로 사용합니다. 이 모델은 이미지-텍스트 쌍을 겹치게 입력 받아 이미지와 텍스트를 인코딩하고 디퓨전 모델을 통해 이미지 재구축을 수행합니다. 이를 통해 이미지 인코더가 유의미한 세부 정보를 더 많이 포함하게 됩니다. DEEM은 CLIP-ViT보다 작은 이미지 인코더(CLIP-ConvNext-B)를 사용하면서도 더 적은 훈련 매개변수와 더 적은 사전 훈련 데이터를 사용합니다.

- **Performance Highlights**: DEEM은 RobustVQA와 POPE와 같은 벤치마크에서 기존 최첨단 모델에 비해 뛰어난 성능을 보였습니다. 특히 RobustVQA-R에서 DEEM은 MM-Interleaved보다 15.9% 더 높은 정확도를, POPE-Random에서는 5.2% 더 높은 정확도를 기록했습니다. 또한 지도 학습을 통해 다양한 멀티모달 작업(예: 시각적 질문 응답, 텍스트-투-이미지 생성 등)에서 경쟁력 있는 결과를 얻었습니다.



### Denoising LM: Pushing the Limits of Error Correction Models for Speech Recognition (https://arxiv.org/abs/2405.15216)
Comments:
          under review

- **What's New**: 이 논문에서는 자동 음성 인식(ASR) 시스템의 오류를 수정하기 위한 새로운 접근 방식인 Denoising LM (DLM)을 제안합니다. DLM은 기존의 언어 모델(LM)과 달리 대량의 합성 데이터를 사용하여 훈련됨으로써 새로운 최첨단 성능을 달성했습니다.

- **Technical Details**: DLM은 텍스트를 음성 합성(Text-to-Speech, TTS) 시스템에 입력하여 생성된 오디오를 ASR 시스템에 전달해 얻은 '노이즈가 포함된 가설(hypotheses)'을 원래 텍스트와 매칭하여 훈련합니다. 주요 구성 요소는 다음과 같습니다: (i) 대규모 모델 및 데이터 사용; (ii) 다중 화자 TTS 시스템 활용; (iii) 다양한 노이즈 증강 전략 결합; (iv) 새로운 디코딩 기법 도입.

- **Performance Highlights**: LibriSpeech 테스트 세트에서 DLM은 외부 오디오 데이터를 사용하지 않고도 test-clean에서 1.5% WER(Word Error Rate), test-other에서 3.3% WER을 달성하여 최고의 성능을 보였습니다. 또한, DLM은 다양한 ASR 시스템에 적용 가능하며 기존의 LM 기반 빔 탐색 디코딩을 크게 능가했습니다.



### SOAP: Enhancing Efficiency of Generated Code via Self-Optimization (https://arxiv.org/abs/2405.15189)
Comments:
          31 pages, 18 figures, and 8 tables

- **What's New**: 대형 언어 모델(LLMs)이 코드 생성에서 눈부신 발전을 보였지만, 생성된 코드가 비효율성 문제로 실행 시간이 길어지고 메모리 소비가 증가하는 단점이 아직 있습니다. 이를 해결하기 위해, 우리는 실행 오버헤드 프로파일(OverheAd Profile)을 기반으로 한 자가 최적화 프레임워크인 SOAP(Self Optimization based on OverheAd Profile)을 제안합니다. SOAP는 먼저 LLM을 사용하여 코드를 생성하고, 이를 로컬에서 실행하여 실행 시간 및 메모리 사용 프로파일을 수집합니다. 이 프로파일은 다시 LLM에 피드백하여 오버헤드를 줄이도록 코드를 수정하는 방식입니다.

- **Technical Details**: SOAP의 핵심은 실행 오버헤드 프로파일을 활용하여 LLM이 생성한 코드를 효율적으로 개선하는 것입니다. LLM이 최초 코드를 생성하면 이를 로컬 환경에서 실행하여 실행 시간과 메모리 사용량을 측정한 후, 이러한 프로파일 정보를 기반으로 LLM이 코드를 반복적으로 수정하여 최적화합니다. 이를 통해 코드의 효율성이 향상됩니다. SOAP의 성능을 평가하기 위해 EffiBench, HumanEval, MBPP에서 16개의 오픈소스 모델과 6개의 클로즈드 소스 모델을 대상으로 실험을 진행하였습니다.

- **Performance Highlights**: 실험 결과, SOAP는 반복적인 자가 최적화를 통해 LLM이 생성한 코드의 효율성을 크게 향상시켰습니다. 예를 들어, EffiBench에서 StarCoder2-15B의 실행 시간이 0.93초에서 0.12초로 감소하여 실행 시간 요구사항이 87.1% 줄어들었고, 총 메모리 사용량은 22.02Mb*s에서 2.03Mb*s로 줄어들어 90.8% 감소하였습니다. SOAP의 소스 코드는 본문에 제공된 URL에서 확인할 수 있습니다.



### CulturePark: Boosting Cross-cultural Understanding in Large Language Models (https://arxiv.org/abs/2405.15145)
Comments:
          Technical report; 28 pages

- **What's New**: 최근 연구에서 많은 대형 언어 모델(LLMs)이 데이터 부족으로 인해 문화적 편향을 보이는 문제를 해결하기 위해 CulturePark라는 새로운 멀티 에이전트 커뮤니케이션 프레임워크를 소개했다. 이 프레임워크는 다양한 문화를 대표하는 LLM 기반 에이전트들이 서로 소통함으로써 고품질의 문화적 데이터를 생성한다.

- **Technical Details**: CulturePark는 여러 문화적 배경을 가진 에이전트들이 참여하는 다중 에이전트 커뮤니케이션 플랫폼이다. 주 대화자는 영어를 사용하는 에이전트로, 여러 문화적 대표자가 이 에이전트와 상호작용하며 인지적 갈등을 유도한다. 이를 통해 다양한 의견과 깊은 사고를 유발하고, 더 포괄적인 답변을 생성하는 대화를 만들어낸다.

- **Performance Highlights**: CulturePark를 통해 생성된 41,000개의 문화 데이터를 사용하여 8개의 문화 특화 LLMs를 미세 조정했다. 내용 검열에서는 GPT-3.5 기반 모델이 GPT-4와 동등하거나 더 나은 성과를 보였으며, 문화적 정렬에서는 Hofstede의 VSM 13 프레임워크에서 GPT-4를 능가했다. 또한, 인간 참여자들의 문화 학습 효과와 사용자 경험에서도 GPT-4보다 우수한 결과를 보였다.



### Intelligent Go-Explore: Standing on the Shoulders of Giant Foundation Models (https://arxiv.org/abs/2405.15143)
- **What's New**: Intelligent Go-Explore (IGE)는 Go-Explore 알고리즘을 기반으로 한 새로운 접근 방식입니다. 기존의 Go-Explore가 수동으로 설계된 탐색 지침에 의존했던 반면, IGE는 기초 모델(Foundation Models, FMs)의 지능을 활용하여 이러한 지침을 대체합니다. 이로 인해 복잡한 환경에서도 직관적으로 흥미로운 상태를 식별할 수 있게 되었습니다.

- **Technical Details**: IGE는 다음 세 가지 주요 방법으로 작동합니다: (1) 가장 유망한 상태를 복귀하고 탐색할 상태로 식별, (2) 선택된 상태에서 최적의 액션 결정, (3) 새로운 상태가 아카이브에 추가될 만큼 흥미로운지 판단. FMs는 거대한 인터넷 규모 데이터셋에서 학습된 자율적 에이전트로, 특히 자연어 처리와 같은 언어 기반 작업에 유용합니다.

- **Performance Highlights**: IGE는 다양한 언어 기반 작업에서 괄목할 만한 성능 향상을 보였습니다. 예를 들어, 수학적 추론 문제인 Game of 24에서는 고전적 그래프 탐색 대비 70.8% 빠르게 100% 성공률을 달성했습니다. 또한 BabyAI-Text에서는 훨씬 적은 온라인 샘플로 기존 최고 성능을 초과했으며, TextWorld에서는 이전의 최고 성능 FM 에이전트인 Reflexion이 실패한 경우에서도 성공을 거두었습니다.



### OptLLM: Optimal Assignment of Queries to Large Language Models (https://arxiv.org/abs/2405.15130)
Comments:
          This paper is accepted by ICWS 2024

- **What's New**: 이번 논문에서는 비용 효율적인 쿼리 할당 문제를 해결하기 위한 프레임워크인 OptLLM을 제안합니다. OptLLM은 예산 제약과 성능 선호도에 맞춰 최적의 솔루션을 제공합니다. 여러 LLMs(Large Language Models) 간의 쿼리 성능을 예측하고, 비용 절감과 성능 향상을 동시에 달성할 수 있습니다.

- **Technical Details**: OptLLM 프레임워크는 예측(prediction)과 최적화(optimization) 두 가지 구성 요소로 이루어져 있습니다. 예측 단계에서는 후보 LLM들이 각 쿼리를 얼마나 성공적으로 처리할지 예측하기 위해 다중 라벨 분류(multi-label classification) 모델을 사용하며, 예측 불확실성을 처리하기 위해 부트스트랩 샘플 예측의 표준 편차를 계산합니다. 최적화 단계에서는 파괴(destruction)와 재구성(reconstruction) 과정을 통해 비용을 최소화하고 성능을 극대화하는 비지배 솔루션(non-dominated solutions)을 생성합니다.

- **Performance Highlights**: 실험 결과, OptLLM은 최고 성능을 가진 개별 LLM과 동일한 정확도를 유지하면서 비용을 2.40%에서 49.18%까지 절감할 수 있었습니다. 또한, 다른 다중 목적 최적화 알고리즘과 비교했을 때 동일 비용으로 정확도를 2.94%에서 69.05%까지 개선하거나, 동일 정확도로 비용을 8.79%에서 95.87%까지 절감할 수 있음을 증명하였습니다.



### Towards Better Understanding of In-Context Learning Ability from In-Context Uncertainty Quantification (https://arxiv.org/abs/2405.15115)
- **What's New**: 이번 연구에서는 트랜스포머(Transformer) 모델을 활용한 선형 회귀 작업(linear regression tasks)의 재훈련을 다루며, 기존 연구와 달리 조건부 기대값 E[Y|X]뿐만 아니라 조건부 분산 Var(Y|X)의 예측을 포함하는 바이오브젝티브(bi-objective) 예측 작업을 고려합니다. 이는 불확실성 정량화(uncertainty quantification) 목표를 추가함으로써, 자중 학습(in-weight learning: IWL)과 문맥 내 학습(in-context learning: ICL)을 구분하고 훈련 분포의 사전 정보 유무에 따른 알고리즘을 더 명확하게 분리하는 데 도움을 줍니다.

- **Technical Details**: 이 연구는 트랜스포머 모델의 문맥 창(context window) S에 따른 일반화 경계(generalization bound)를 $	ilde{eversemathcal{O}}(rac{	ext{min}{S, T}}{	ext{nT}})$로 도출하였으며, 이는 이전 연구들에서 나타난 경계인 $	ilde{eversemathcal{O}}(rac{1}{	ext{n}})$에 비해 더 날카로운 분석을 제공합니다. 또한, 마코프 체인(Markov chain)을 활용하여 프롬프트 시퀀스(prompt sequence)의 믹싱 시간(mixing time)의 상한을 설정하고, 문맥 창이 제한된 경우의 추가 근사 오차 항을 검토하였습니다. 이 결과는 거의 베이즈 최적(Bayes-optimum) 리스크에 가까운 훈련된 트랜스포머의 리스크 수렴을 정량화합니다.

- **Performance Highlights**: 실험적으로, 훈련된 트랜스포머는 분포가 변하는 환경에서도 베이즈 최적 해법을 모방하지는 못하지만, 조건부 기대값과 분산을 성공적으로 예측할 수 있음을 보여줍니다. 특히, 훈련 데이터의 태스크 다양성이 크다면, 트랜스포머는 태스크 분포 변이, 공변량 변이(covariates shift), 프롬프트 길이 변이(prompt length shift) 시나리오에서도 문맥 내 학습(ICL) 능력을 나타냅니다. 또한 메타 러닝 접근법을 통해 태스크 다양성을 증가시키면 공변량 변이에서도 트랜스포머의 문맥 내 학습 성능을 향상시킬 수 있습니다.



### Dissociation of Faithful and Unfaithful Reasoning in LLMs (https://arxiv.org/abs/2405.15092)
Comments:
          code published at this https URL

- **What's New**: 새 연구에서는 대형 언어 모델(LLMs)이 Chain of Thought(CoT) 추론 텍스트에서 발생한 오류를 어떻게 복구하는지 조사했습니다. 이 연구는 CoT에서 발생한 오류 후에도 정확한 최종 답변에 도달하는 모델의 성능을 분석했습니다.

- **Technical Details**: 연구진은 LLM의 추론 과정에서의 '신뢰성(faithfulness)'에 대한 질문에 중점을 두었으며, CoT 텍스트에서의 오류 회복 행동을 측정했습니다. GPT-3.5 및 GPT-4 모델을 사용하여 다양한 수학 문제 데이터셋에서 실험을 진행했습니다. 오류는 주어진 CoT 텍스트의 특정 숫자를 랜덤 값으로 변형해 도입했습니다(random integer values in {-3, -2, -1, 1, 2, 3}). 모든 모델 응답은 OpenAI 공개 API를 통해 수집되었으며, 그리디 디코딩 방식(greedy decoding)으로 샘플링 되었습니다.

- **Performance Highlights**: 모델은 분명한 오류에서 더 자주 회복되었으며, 올바른 답변에 대한 증거가 더 많은 컨텍스트에서 더 자주 회복되었습니다. 반면, '신뢰할 수 없는(unfaithful)' 회복은 더 어려운 오류 위치에서 더 자주 발생했습니다. 이는 LLM의 신뢰할 수 있는 오류 회복과 신뢰할 수 없는 오류 회복이 고유의 메커니즘에 의해 구동된다는 점을 시사합니다.



### OAC: Output-adaptive Calibration for Accurate Post-training Quantization (https://arxiv.org/abs/2405.15025)
Comments:
          20 pages, 4 figures

- **What's New**: 대규모 언어 모델(LLMs)의 배포는 급증하는 모델 크기 때문에 막대한 계산 비용을 수반합니다. 이를 해결하기 위한 새로운 방법으로, Output-adaptive Calibration (OAC)을 제안합니다. 이 방법은 모델 출력의 왜곡을 측정하여 기존 후처리 양자화(Post-training Quantization, PTQ) 방법들보다 더 나은 성능을 보입니다. 특히, 극단적으로 낮은 정밀도(2-bit 및 이진)의 양자화에서 탁월한 결과를 보여줍니다.

- **Technical Details**: 기존 PTQ 방법들은 레이어별 ℓ2 손실을 기반으로 양자화 오차를 계산하여 모델의 레이어들을 교정합니다. 그러나 OAC는 모델 출력의 교차 엔트로피 손실(distortion of the output cross-entropy loss)을 기반으로 양자화 오차를 계산합니다. 이는 출력적응 해시안(output-adaptive Hessian)을 사용하여 레이어별로 가중치 행렬을 업데이트합니다. 대규모 모델의 경우 정확한 해시안 계산이 어려워 몇 가지 가정을 통해 계산 복잡도를 줄였습니다.

- **Performance Highlights**: OAC는 ZeroQuant, LLM.int8(), SmoothQuant 등 기존의 최첨단 PTQ 방법들보다 특히 2-bit 및 이진 양자화에서 훨씬 뛰어난 성능을 나타냈습니다. 실험 결과, 다양한 작업에서 OAC가 다른 PTQ 방법을 상회하는 성능을 보였습니다.



### In-context Time Series Predictor (https://arxiv.org/abs/2405.14982)
- **What's New**: 최근 Transformer 기반의 대형 언어 모델(LLMs)은 주어진 문맥만을 바탕으로 다양한 기능을 수행할 수 있는 '문맥 내 학습(In-Context Learning)' 기능을 보여주고 있습니다. 이번 연구에서는 기존 Transformer 또는 LLM 기반의 시계열 예측 방법과는 달리, 시계열 예측 작업을 입력 토큰으로 재구성하여 다수의 (lookback, future) 쌍으로 구성된 시퀀스를 생성함으로써 문맥 내 예측 기능을 효과적으로 활용하는 새로운 방식을 제안합니다. 이 방법은 Pre-trained LLM 파라미터를 사용하지 않고도 더 파라미터 효율적입니다.

- **Technical Details**: 시계열 예측(Time Series Forecasting, TSF)은 기존 데이터로부터 미래 값을 예측하는 데 중요합니다. 제안된 In-context Time Series Predictor (ICTSP) 구조는 문맥 내 학습 기능을 활용하여 (lookback, future) 쌍을 입력 토큰으로 사용합니다. 이 접근법은 Transformer가 예측 작업을 수행하는 데 필요한 최적의 모델을 문맥 예제로부터 학습할 수 있게 합니다. 또한, ICTSP 구조는 기존 TSF 모델의 주요 문제점인 timestep 혼합, 순서 불변성, 채널 구조 제한 등의 문제를 해결합니다.

- **Performance Highlights**: ICTSP는 기존의 TSF Transformer 모델들이 가진 오버피팅 문제를 크게 해결하며, 다변량 시계열 예측 설정에서 full-data, few-shot, zero-shot 상황 모두에서 일관되게 우수한 성능을 발휘합니다. 이는 기존의 단순한 모델인 선형 예측기나 MLP보다도 뛰어난 성능을 보입니다. 또한, ICTSP는 여러 더 단순한 모델을 특수 경우로 포함하며, 복잡도의 순차적 감소를 통해 안정적인 성능을 유지합니다. 기존 방식에서 자주 문제가 되었던 Overfitting 없이 안정적인 성능을 발휘할 수 있습니다.



### LOVA3: Learning to Visual Question Answering, Asking and Assessmen (https://arxiv.org/abs/2405.14974)
Comments:
          The code is available at this https URL

- **What's New**: LOVA3는 인간이 지닌 질문하기, 질문 평가, 질문 답변 능력을 모두 갖춘 다중모달 대형 언어 모델 (Multimodal Large Language Models, MLLMs)을 제안합니다. 기존 MLLMs는 주로 질문답변 능력에 집중했으나, 새로운 프레임워크 LOVA3는 질문을 생성하고 (GenQA), 질문-답변을 평가하는 (EvalQA) 두 가지 추가 학습 과제를 도입하여 모델의 다중모달 이해도를 향상시킵니다.

- **Technical Details**: LOVA3 프레임워크는 두 가지 추가 학습 과제인 GenQA와 EvalQA를 사용하여 탄탄한 다중모달 대형 언어 모델을 설계합니다. GenQA는 모델이 이미지에 대한 다양한 질문-답변 쌍을 생성하도록 하여 질문 생성 능력을 강화시키며, 여러 종류의 다중모달 기초 작업을 포함합니다. EvalQA는 모델이 주어진 시각-질문-답변 트리플렛의 정확성을 예측하는 과제를 포함합니다. 평가를 위해, EvalQABench라는 새로운 벤치마크를 도입하였습니다.

- **Performance Highlights**: LOVA3 프레임워크로 학습된 모델은 GQA, VQAv2, VizWiz 등 10개의 널리 사용되는 다중모달 벤치마크에서 일관된 성능 향상을 보였습니다. 특히, GenQA와 EvalQA 과제를 추가함으로써 얻어진 성능 개선은 모델의 다중모달 이해력과 문제 해결 능력을 크게 향상시킵니다. 이를 통해 다중모달 대형 언어 모델이 질문 생성 및 평가 능력을 통해 더욱 우수한 성능을 발휘할 수 있음을 입증했습니다.



### SliM-LLM: Salience-Driven Mixed-Precision Quantization for Large Language Models (https://arxiv.org/abs/2405.14917)
Comments:
          22 pages

- **What's New**: 대형 언어 모델(LLMs)이 자연어 이해에서 놀라운 성과를 내고 있지만, 상당한 계산 및 메모리 자원을 요구합니다. 새로운 연구는 Salience-Driven Mixed-Precision Quantization(SliM-LLM)이라는 새로운 압축 기법을 제안합니다. 이는 가중치의 중요도 분포를 활용하여 최적의 비트폭과 양자화기를 결정함으로써 정확한 LLM 양자화를 달성합니다.

- **Technical Details**: SliM-LLM는 두 가지 핵심 기술을 주로 사용합니다: (1) Salience-Determined Bit Allocation은 중요도 분포를 군집 특성으로 활용하여 각 그룹의 비트폭을 할당함으로써 양자화된 LLM의 정확성을 높입니다. (2) Salience-Weighted Quantizer Calibration은 그룹 내 요소별 중요도를 고려하여 양자화기의 파라미터를 최적화함으로써 중요한 정보를 유지하고 오류를 최소화합니다.

- **Performance Highlights**: SliM-LLM는 초저비트 환경에서 LLM의 정확성을 크게 향상시킵니다. 예를 들어, 2-bit LLaMA-7B은 기존 모델 대비 메모리를 5.5배 절약하고, 최신의 비티훈련(Post-Training Quantization) 기법 대비 퍼플렉서티를 48% 감소시켰습니다. 또한, SliM-LLM+는 그래디언트 기반의 양자화기를 통합하여 퍼플렉서티를 추가로 35.1% 감소시켰습니다.



### Data Mixing Made Efficient: A Bivariate Scaling Law for Language Model Pretraining (https://arxiv.org/abs/2405.14908)
- **What's New**: 이번 연구에서는 대형 언어 모델(Large Language Models)의 데이터 혼합 방식을 개선하고 이론적인 지침을 제공하는 새로운 접근법을 제안합니다. 실험 결과, 엔트로피 기반(entropy-driven) 훈련 없는 데이터 혼합법이 기존 자원 소모가 큰 방법들보다 성능이 우수하거나 비슷한 결과를 얻을 수 있음을 발견했습니다.

- **Technical Details**: 이 연구는 데이터 혼합을 단순화하고 훈련 효율성을 높이기 위해 저비용 대리 데이터(low-cost proxies)를 사용한 전략을 탐구합니다. 구체적으로, 데이터 양과 혼합 비율의 이변량 스케일링 동작(bivariate scaling behaviors)을 정확하게 모델링하는 통합 스케일링 법칙(unified scaling law), BiMix를 제안합니다. BiMix는 데이터 양과 혼합 비율 간의 관계를 이론적으로 설명하고 예측하는 데 강력한 도구로 작용합니다.

- **Performance Highlights**: BiMix의 예측력과 기본 원리에 대한 실증적 증거들을 제공하며, 엔트로피 기반의 훈련 없는 데이터 혼합법으로도 자원 소모가 큰 기존 방법들과 동일하거나 더 나은 성능을 달성할 수 있음을 보여주었습니다.



### Structural Entities Extraction and Patient Indications Incorporation for Chest X-ray Report Generation (https://arxiv.org/abs/2405.14905)
Comments:
          The code is available at this https URL or this https URL

- **What's New**: 새로운 방법으로 SEI(Structural Entities extraction and patient indications Incorporation)를 도입하여 흉부 X-ray 리포트 생성을 자동화했습니다. 이 방법은 기존의 프레젠테이션 스타일 단어를 제거하고 사실 기반 엔티티(entities) 시퀀스의 품질을 개선함으로써, 방사선 의사의 워크로드를 줄이는 데 효과적입니다.

- **Technical Details**: SEI 방법은 다음과 같은 단계로 구성됩니다. 첫 번째로, 구조적 엔티티 추출(SEE) 방식을 사용하여 리포트에서 프레젠테이션 스타일 어휘를 제거합니다. 이를 통해 크로스 모달 정렬(cross-modal alignment) 모듈의 노이즈를 줄이고, X-ray 이미지와 리포트의 사실 기반 엔티티 시퀀스를 정렬하여 정밀도를 높입니다. 두 번째로, 크로스 모달 융합 네트워크(cross-modal fusion network)를 제안해 X-ray 이미지, 유사한 히스토리컬 케이스, 그리고 환자-specific 인디케이션을 통합합니다.

- **Performance Highlights**: MIMIC-CXR 데이터셋 실험 결과, SEI 방법이 양자 언어 생성(natural language generation)과 임상 효율성(clinical efficacy) 측면에서 최첨단 접근 방식보다 뛰어남을 확인했습니다.



### S-Eval: Automatic and Adaptive Test Generation for Benchmarking Safety Evaluation of Large Language Models (https://arxiv.org/abs/2405.14191)
Comments:
          18 pages, 11 figures

- **What's New**: 대형 언어 모델(LLMs)의 안전성을 평가하기 위해 새로운 종합적이고 다차원적인 평가 벤치마크인 S-Eval을 제안하였습니다. 이 벤치마크는 LLM 기반 자동 테스트 프롬프트 생성 및 선택 프레임워크를 도입하여 고품질의 테스트 스위트를 자동으로 구성합니다. 전문가 수준의 테스트 LLM Mt와 다양한 테스트 선택 전략을 결합하여 사용합니다.

- **Technical Details**: S-Eval의 핵심은 새로운 전문가 안전 평가 LLM Mc로, LLM의 응답 위험 점수를 양적으로 평가하고 위험 태그와 설명을 제공합니다. 생성 과정은 네 가지 다른 수준의 위험을 포함하는 세심하게 설계된 위험 분류에 의해 안내됩니다. 이 벤치마크는 22만 개의 평가 프롬프트로 구성되어 있으며, 이 중 2만 개는 기본 위험 프롬프트(중국어와 영어로 각각 1만 개씩)이고, 나머지 20만 개는 인기 있는 10개의 적대적 공격 지침에서 파생된 공격 프롬프트입니다.

- **Performance Highlights**: S-Eval은 20개의 주요 LLM에서 광범위하게 평가되었으며, 기존 벤치마크보다 LLM의 안전 위험성을 더 잘 반영하고 정보를 제공할 수 있음을 확인했습니다. 또한 매개 변수 규모, 언어 환경, 디코딩 매개 변수의 영향을 탐구하여 LLM의 안전성을 평가하기 위한 체계적인 방법론을 제공합니다.




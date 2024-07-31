New uploads on arXiv(cs.CL)

### Does Refusal Training in LLMs Generalize to the Past Tense? (https://arxiv.org/abs/2407.11969)
Comments:
          Code and jailbreak artifacts: this https URL

- **What's New**: 최신의 연구는 대형 언어 모델(LLMs)이 유해하거나 불법적인 요청을 거부하는 방법을 다룹니다. 특히, 간단히 유해한 요청을 과거 시제로 재구성하는 것만으로도 최신 LLM을 우회하는 데 효과적일 수 있다는 점을 밝혔습니다. 예를 들어, 'Molotov 칵테일 만드는 법?'을 '사람들이 Molotov 칵테일을 어떻게 만들었는가?'로 바꾸면 많은 최첨단 LLM이 이 요청을 받아들인다는 사실을 발견했습니다.

- **Technical Details**: 이 연구에서는 Llama-3 8B, GPT-3.5 Turbo, Gemma-2 9B, Phi-3-Mini, GPT-4o, R2D2 모델을 평가했습니다. GPT-3.5 Turbo를 재구성 모델로 사용하여 과거 시제로 유해 요청을 재구성했습니다. 과거 시제의 재구성은 단순하지만 매우 효과적임을 확인했으며, 여러 번 시도한 결과 GPT-4o의 성공률은 1%에서 88%로 증가했습니다.

- **Performance Highlights**: 자연어 처리 모델의 경우, 과거 시제의 재구성은 효과적인 반면, 미래 시제는 덜 효과적임을 발견했습니다. GPT-3.5 Turbo를 세밀하게 튜닝한 결과, 과거 시제의 요청을 거부하는 것이 가능함을 확인했습니다. 이 연구는 SFT(Supervised Fine-Tuning), RLHF(강화 학습을 통한 인간 피드백), 적대적 훈련과 같은 널리 사용되는 정합 기술들이 의도한 대로 일반화되지 않을 수 있음을 강조했습니다.



### NeedleBench: Can LLMs Do Retrieval and Reasoning in 1 Million Context Window? (https://arxiv.org/abs/2407.11963)
- **What's New**: 최근 연구는 대형 언어 모델(LLMs)의 장문 텍스트 처리 능력을 평가하기 위한 NeedleBench 프레임워크를 도입했습니다. 이는 장문 문서에서 사용자 질의와 관련된 내용을 식별하는 능력을 테스트하며, 다양한 길이 범위와 깊이 대역에서 중요한 데이터를 전략적으로 삽입하여 모델의 정보 검색 및 논리적 추론 능력을 평가합니다. 추가로 Ancestral Trace Challenge (ATC)를 제안하여 현실 세계의 복잡한 장문 컨텍스트 작업을 모사합니다.

- **Technical Details**: NeedleBench는 4k, 8k, 32k, 128k, 200k, 1000k 이상의 다양한 길이 구간과 깊이 범위에서 모델의 능력을 평가하는 진보적인 과제들로 구성됩니다. NeedleBench 내의 주요 과제는 단일 정보 검색(Single-Needle Retrieval Task, S-RT), 다중 정보 검색(Multi-Needle Retrieval Task, M-RT), 다중 정보 논리적 추론(Multi-Needle Reasoning Task, M-RS)으로 나뉩니다. 또한, ATC는 다중 단계의 논리적 추론을 요구하는 복잡한 작업을 단순화하여 모델의 실제 적용 능력을 테스트합니다.

- **Performance Highlights**: 현재의 LLM들은 실제 장문 컨텍스트 작업에서 논리적 추론의 복잡성을 처리하는 데 어려움을 겪고 있으며, 특히 2K 토큰 이하의 텍스트에서도 매끄럽게 논리적 관계를 이해하지 못하는 점을 발견했습니다. 이 연구는 주요 오픈소스 모델들이 장문에서 질문과 관련된 주요 정보를 식별하고 추론하는 능력에서 현저히 개선의 여지가 있음을 보여주었습니다. 모든 코드와 자원은 OpenCompass에서 제공됩니다.



### Rethinking Transformer-based Multi-document Summarization: An Empirical Investigation (https://arxiv.org/abs/2407.11948)
- **What's New**: Transformer 기반 다중 문서 요약(MDS)의 성능 및 동작을 다양한 실험을 통해 분석한 연구가 발표되었습니다. 이 연구는 문서 경계 구분자의 영향, 다양한 Transformer 구조의 효과, 인코더 및 디코더의 민감성, 다양한 학습 전략, 생성된 요약문의 반복 문제 등의 주제를 다룹니다.

- **Technical Details**: 이 연구는 다섯 가지 실험을 통해 Transformer 기반 MDS 모델의 동작을 정밀하게 분석했습니다: (1) 문서 경계 구분자의 영향 측정, (2) 주류 Transformer 구조의 효과 탐구, (3) 인코더와 디코더의 민감성 검사, (4) 다양한 학습 전략 논의, (5) 요약 생성 시 반복 문제 탐구. 특히, 문서 경계 구분자는 모델 성능과 문서 경계 인식을 개선하는 데 중요한 역할을 한다는 것을 실험적으로 입증했습니다.

- **Performance Highlights**: 실험 결과, 디코더는 인코더보다 노이즈에 더 민감하게 반응하며, 이는 디코더 개선의 필요성을 시사합니다. 또한, 반복 문제와 예측 불확실성 높은 상관관계를 확인했습니다. 요약문 생성 시 반복 문제가 발생할 때, 불확실성 점수가 증가하여 모델의 신뢰도가 감소한다는 분석 결과가 나왔습니다. 학습 전략 측면에서는 사전학습 및 미세조정(pretrain-finetune) 접근법이 가장 뛰어난 성능을 보였습니다.



### Fine-grained Hallucination Detection and Mitigation in Long-form Question Answering (https://arxiv.org/abs/2407.11930)
Comments:
          Code and data are available: this https URL

- **What's New**: 최근 로우폼 질문 답변 (LFQA)에 대한 오류 주석을 포함하는 최초의 데이터셋인 HaluQuestQA가 소개되었습니다. 이 데이터셋은 사람의 손으로 작성된 답변과 모델이 생성한 답변에 대한 698개의 QA 페어와 4.7k 이상의 스팬 수준 오류 주석을 포함하고 있습니다. 이 데이터를 통해 장문의 답변의 결점인 포괄성이 부족하고, 도움이 되지 않는 참고 자료를 제공하는 문제를 심층 분석합니다. 이에 따라, 오류 스팬을 예측하고 오류에 대한 설명을 제공하는 자동 피드백 모델을 훈련시켰습니다.

- **Technical Details**: HaluQuestQA 데이터셋에는 질문 오인 (question misconception), 사실성 (factuality), 포괄성 (completeness), 관련성 (relevance), 그리고 도움이 되는 참고 자료 (helpful references) 등 다섯 가지 다른 오류 유형이 포함되어 있습니다. 데이터를 기반으로 학습된 자동 피드백 모델은 불완전한 정보가 포함된 오류 스팬을 예측하고 관련된 설명을 제공합니다. 또한, 이 피드백 모델의 신호를 사용하여 생성된 답변을 개선하는 오류 정보 기반 정제 방법 (Error-informed refinement)을 제안합니다.

- **Performance Highlights**: 피드백 모델을 통해 개선된 답변은 실제 사람들에 의해 84% 더 선호되며, 기존의 기준 모델과 비교할 때 환각 현상이 줄어들고 답변의 질이 향상되었습니다. 이 접근 방식을 사용하면 과도한 환각과 오류를 줄이면서 사용자에게 더 포괄적이고 신뢰할 수 있는 답변을 제공합니다.



### What's Wrong? Refining Meeting Summaries with LLM Feedback (https://arxiv.org/abs/2407.11919)
- **What's New**: 디지털 회의가 일상화됨에 따라 회의 요약 자동화가 중요한 과제가 되었으며, 대형 언어 모델(LLM)은 기존 요약 방법에 비해 더 우수한 일관성과 맥락 이해를 제공하는 가능성을 보여줍니다. 하지만 여전히 관련성을 유지하고 허상을 피하는 데 어려움을 겪고 있습니다. 이번 연구에서는 사람이 검토하는 과정을 모방한 오류 식별 및 요약 개선의 두 단계로 구성된 다중 LLM 보정 접근 방식을 소개합니다. 이를 위해 9가지 오류 유형이 주석된 자동 생성된 회의 요약 200개의 QMSum Mistake 데이터를 공개하며, 실험 결과 이 오류들을 높은 정확도로 식별할 수 있음을 보여줍니다.

- **Technical Details**: 본 연구는 사람이 검토하는 과정을 모방한 두 단계 접근 방식을 도입하여 회의 요약을 개선하는 방법을 제안합니다. 첫 번째 단계는 기존 요약에서 오류를 식별하는 것이며, 두 번째 단계는 식별된 오류를 기반으로 요약을 개선하는 것입니다. QMSum Mistake 데이터셋은 9가지 오류 유형(예: 생략, 구조적 오류)으로 주석된 자동 생성된 회의 요약 200개로 구성됩니다. 실험에서는 다양한 LLM 인스턴스를 사용하여 각 오류 유형별로 Chain-of-Thought (CoT) 프롬프팅을 적용하여 최고 성능을 달성했습니다.

- **Performance Highlights**: LLM인 GPT-4 Turbo는 평균 약 89%의 정확도로 오류를 식별할 수 있지만, 관련성(약 81%)과 허상(약 72%) 오류에는 다소 어려움이 있습니다. CoT 프롬프팅과 여러 LLM 인스턴스를 사용하여 식별된 오류를 구체적인 피드백으로 전환한 후, 후속 모델 인스턴스를 통해 요약을 수정함으로써 원래 요약과 비교했을 때 품질이 크게 향상되었습니다. 식별된 오류에 대한 CoT 설명을 피드백으로 활용한 결과, 후속 요약의 품질이 크게 향상되었습니다.



### A Novel Lexicon for the Moral Foundation of Liberty (https://arxiv.org/abs/2407.11862)
- **What's New**: 최근 논문에서는 논란이 많은 사회적 이슈(백신 거부, 기후 변화, 낙태권 등)에 관한 사람들의 입장을 예측하는 데 중요한 역할을 하는 도덕적 가치 중 '자유'를 중심으로 한 새로운 리버티 사전을 제안합니다. 이 연구는 3,000개 이상의 수동으로 주석된 데이터를 활용하여 리버티 관련 표현을 다각적으로 분석함으로써, 더욱 풍부하고 적용 가능한 리버티 사전을 개발합니다.

- **Technical Details**: 연구팀은 위키백과, 리버타리안 및 보수주의 Reddit 커뮤니티, Black Lives Matter (BLM) 그리고 2016 미국 대선 관련 트위터 데이터를 포함한 다양한 플랫폼에서 데이터를 수집했습니다. 데이터를 기반으로 단어 임베딩 유사성 (word embedding similarity; WE) 및 조합 의미론 (compositional semantics; CS) 기법을 사용하여 각각의 데이터셋에 맞는 리버티 사전을 생성했습니다. 최종적으로는 두 기법의 장점을 통합하여 보다 정교한 사전을 개발했습니다.

- **Performance Highlights**: 개발된 리버티 사전은 다양한 도메인에서 검증되어 뛰어난 일반화 성능을 보여주었으며, 이는 논문에서 제공한 벤치마크 데이터셋과의 견고한 평가를 통해 확인되었습니다. 이 연구는 도덕적 가치 분석의 이해를 높이며, 향후 보다 정확하고 포괄적인 평가를 위한 기초를 마련하였습니다.



### Evaluating Task-Oriented Dialogue Consistency through Constraint Satisfaction (https://arxiv.org/abs/2407.11857)
- **What's New**: 이번 논문에서는 대화 일관성을 Constraint Satisfaction Problem (CSP)으로 개념화하는 새로운 접근 방식을 제안합니다. 이 논문은 CSP 해결 도구를 사용하여 대화에서 발생할 수 있는 불일치를 자동으로 감지하고, LLM(Large Language Models)을 통해 재-lexicalized된 대화에서 CSP 해결 도구가 얼마나 효과적일 수 있는지를 조사했습니다.

- **Technical Details**: CSP는 대화 일관성을 변수와 제약 조건으로 모델링합니다. 변수는 대화의 특정 부분을 나타내며, 제약 조건은 언어적, 대화적, 도메인 기반 속성을 반영합니다. 연구에서는 CSP 해결 도구를 사용하여 불일치를 감지하고, 이 방법을 통해 대화가 일관적인지 평가할 수 있음을 보여줍니다.

- **Performance Highlights**: LLM의 일관성 유지가 도전적인 작업임을 확인하면서, CSP 해결 도구를 사용했을 때의 정확도는 0.15에 불과하였습니다. 또한, 도메인 지식에서 파생된 제약 조건을 준수하는 것이 가장 어려웠다는 결과를 얻었습니다.



### Scaling Sign Language Translation (https://arxiv.org/abs/2407.11855)
- **What's New**: 이번 연구는 대규모 사전 학습 데이터를 활용하여 모델의 크기와 번역 방향 수를 확장함으로써 수어 번역(SLT) 분야에서의 한계를 극복하려는 시도입니다. 다양한 데이터 소스를 사용하여 대규모 SLT 사전 학습을 수행했으며, 이를 통해 개방형 도메인 과제에서도 SLT 모델의 성능을 개선하고자 했습니다.

- **Technical Details**: 연구진은 1) YouTube에서 수집한 다중 언어 SLT 데이터, 2) 병렬 텍스트 말뭉치, 및 3) 비디오 캡션을 다른 언어로 번역한 데이터를 포함한 다양한 데이터를 사용하여 사전 학습을 진행했습니다. 모델 초기화에는 미리 학습된 T5, mT5, 및 ByT5 모델을 사용했으며, 인코더-디코더 아키텍처 하에 다양한 학습 과업을 통합했습니다. 모델 크기를 확장하여 성능 향상을 도모했으며, 입력 프롬프트에 작업별 제어 토큰을 포함시켜 다양한 작업과 언어를 사전 학습에 포함시킬 수 있게 했습니다.

- **Performance Highlights**: How2Sign 및 FLEURS-ASL#0 데이터셋 실험에서 데이터와 모델 크기 확장이 SLT 성능 개선에 중요함을 입증했습니다. 특히, 미리 학습된 SLT 모델을 하향 과제에서 미세 조정한 결과, 기존의 상태 최첨단(SOTA) 모델을 광범위하게 능가했습니다. 또한, 고품질의 SLT 훈련 데이터를 사용하는 것은 어렵지만, YouTube에서 수집한 약한 레이블링이 된 데이터를 활용하여 다양한 주제와 수어 사용자에 대한 커버리지를 높였습니다. 실험 결과, 더 큰 모델을 사용하는 것이 항상 도움이 되지는 않았지만, 모델의 용량 한계를 극복할 때 효과적임을 확인했습니다.



### Zero-shot Cross-Lingual Transfer for Synthetic Data Generation in Grammatical Error Detection (https://arxiv.org/abs/2407.11854)
Comments:
          Submitted to EMNLP 2024

- **What's New**: 본 논문에서는 다국어 사전 학습 언어 모델 (multilingual pre-trained language models, mPLMs)의 제로샷 크로스-링구얼 전이 (zero-shot cross-lingual transfer) 능력을 활용하여, 인적 주석이 부족한 저자원 언어에서 문법 오류 검출 (Grammatical Error Detection, GED)을 수행하는 새로운 방법을 제안합니다. 제안된 방법은 다국어 데이터에서 생성된 합성 오류 (synthetic errors)를 사용하여 GED 모델을 학습하며, 두 단계의 미세 조정 (fine-tuning) 파이프라인을 통해 성능을 향상시킵니다.

- **Technical Details**: 제안된 GED 방법은 네 단계의 과정으로 개발됩니다. 첫 번째로, 소스 언어의 GEC 데이터셋을 사용하여 다국어 AEG 모델을 학습합니다. 두 번째로, 이 AEG 모델을 사용하여 타겟 언어와 소스 언어를 포함하는 GED 데이터셋을 생성합니다. 세 번째로, 다국어 인공 오류 데이터셋에서 GED 모델을 미세 조정합니다. 마지막으로, 사람-주석된 GED 데이터셋을 사용하여 추가 미세 조정을 수행합니다. 이러한 두 단계의 미세 조정 파이프라인을 통해 다양하고 인간의 오류와 유사한 오류를 생성할 수 있으며 기존의 방법보다 뛰어난 성능을 제공합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 기존의 최첨단 주석이 없는 GED 방법을 능가하는 것으로 나타났습니다. 특히, 다국어 인공 오류 데이터를 사용한 제로샷 크로스-링구얼 전이를 통해 GED 성능을 높였습니다. 본 연구는 6개의 소스 언어와 5개의 타겟 언어를 대상으로 실험을 진행했으며, 총 11개 언어에서 5백만 개 이상의 샘플로 구성된 합성GED 코퍼스를 공개합니다.



### InferAct: Inferring Safe Actions for LLM-Based Agents Through Preemptive Evaluation and Human Feedback (https://arxiv.org/abs/2407.11843)
- **What's New**: 이번 논문에서는 InferAct라는 새로운 접근 방식을 소개합니다. InferAct는 대형 언어 모델(Large Language Models, LLMs)의 Theory-of-Mind(TOM) 능력을 활용하여, 중요한 행동이 실행되기 전에 발생할 수 있는 오류를 사전에 탐지합니다. 이는 실시간으로 평가되어 위험한 결과가 발생하기 전에 오류를 탐지하고 인간의 피드백을 통합하여 배우는 방식입니다.

- **Technical Details**: InferAct는 인간 감독자의 경계심을 모방하여, 에이전트의 행동이 의도한 목표와 벗어나는지 여부를 실시간으로 평가합니다. 인지 과학의 Theory of Mind(ToM) 능력을 활용하여 LLMs가 행동 사슬(chain) 뒤에 숨겨진 의도를 해석하게 합니다. 만약 의도에서 벗어난다는 것이 감지되면, InferAct는 즉시 인간에게 알리고 피드백을 받을 수 있습니다. 이를 통해 치명적인 사건이 발생하기 전에 예방하고 배우는 에이전트의 결정 능력을 개선합니다.

- **Performance Highlights**: InferAct는 웹 쇼핑, 가정 작업, 그리고 검색 기반 질문 응답 작업을 포함한 세 가지 환경에서 실험이 진행되었습니다. 이 실험에서 InferAct는 state-of-the-art 성능을 보여주었으며, GPT-4-turbo, GPT-3.5-turbo, Llama-3-70B 등 다양한 LLMs 백엔드와 함께 높은 성과를 기록했습니다. 특히, 고위험 상황에서의 실험에서도 InferAct는 오류 감지와 위험 완화에 있어 탁월한 성과를 보였습니다.



### LoFTI: Localization and Factuality Transfer to Indian Locales (https://arxiv.org/abs/2407.11833)
Comments:
          21 pages

- **What's New**: 이번 연구에서는 인도 지역에 맞춘 로컬라이제이션 및 현실성 전이를 평가하기 위한 새로운 벤치마크 'LoFTI (Localization and Factuality Transfer to Indian Locales)'를 소개합니다. 이는 주로 서구권, 특히 영어권 데이터를 기반으로 훈련된 대형 언어 모델(LLMs)의 지리적 편향 문제를 해결하고, 다양한 인도 지역에 대한 정확한 사실 전달 능력을 평가하는 최초의 벤치마크입니다.

- **Technical Details**: LoFTI는 출처(location) 및 대상(entity) 위치에 대한 사실적 진술을 포함하고 있으며, 대상 위치는 모두 인도로, 국가, 주 및 도시에 걸친 다양한 수준의 하이퍼로컬리티(hyperlocality)를 가지고 있습니다. 이 데이터셋은 음식, 스포츠, 자연 등 다양한 카테고리를 포함합니다. 주어진 레퍼런스 텍스트를 기준으로 Mixtral 및 GPT-4를 포함한 여러 모델을 평가했습니다. Mixtral 모델은 추가적인 증거를 활용하여 성능 강화를 도모했습니다.

- **Performance Highlights**: LoFTI 벤치마크를 사용한 평가에서 GPT-4와 Mixtral 기반 접근 방식은 목표한 하이퍼로컬리티 수준에서 편향된 결과를 보여주었습니다. 특히 GPT-4는 Mixtral 변형 모델들에 비해 성능이 우월했지만, 지리적 지역에 따라 성능 저하가 나타났습니다. 이 벤치마크는 공개적으로 배포되며(https://huggingface.co/datasets/sonasimon/LoFTI), 다양한 지리적 지역을 고려한 맥락별 질의응답(Question Answering) 작업에서도 사용할 수 있습니다.



### GPT Assisted Annotation of Rhetorical and Linguistic Features for Interpretable Propaganda Technique Detection in News Tex (https://arxiv.org/abs/2407.11827)
- **What's New**: 이 논문은 설득력 있는 텍스트의 선전에 사용되는 기법을 감지하기 위해 기계 학습 모델을 사용하는 것에 대한 새로운 접근법을 제시합니다. 기존의 '블랙 박스(black-box)' 솔루션 대신 해석 가능한 접근법을 강조하면서, 이 연구는 22개의 수사적 및 언어적 특징을 문헌에서 식별하고 이를 활용하여 기존 데이터 셋을 주석합니다. 이 과정에서 인간 전문가의 주석과 GPT-3.5를 조합하여 비용을 절감했으며, 이는 인간 전문가만을 사용한 전통적인 주석 방식보다 효율적입니다.

- **Technical Details**: 이 연구는 NLP4IF 워크샵에서 소개된 프로파간다 기술 분류 데이터 셋(PTC corpus)을 활용했습니다. 이 데이터 셋은 18개의 프로파간다 기법으로 주석된 451개의 뉴스 기사로 구성되어 있습니다. 연구자는 이 데이터 셋을 문장 단위로 레이블링하여, 각 문장이 하나 이상의 프로파간다 기법으로 레이블링되도록 했습니다. 또한, RhetAnn이라는 웹 애플리케이션을 개발하여 사람이 텍스트를 주석하는 과정을 돕고자 했으며, 주석 과정에서 GPT-4의 설명을 활용해 효율성을 높였습니다.

- **Performance Highlights**: 이 연구는 소수의 인간 주석 예시와 GPT-3.5를 결합하여 주석 과정의 비용을 획기적으로 줄이는 방법을 제시합니다. 실험 결과, GPT-4와 비교하여 10배 더 저렴한 비용으로 동등한 성능을 달성할 수 있음을 확인했습니다. 이는 전통적인 인간 전문가 기반의 주석 방식보다 훨씬 효율적이며, 앞으로 다양한 NLP 작업에 활용될 수 있는 가능성을 보여줍니다.



### PipeInfer: Accelerating LLM Inference using Asynchronous Pipelined Speculation (https://arxiv.org/abs/2407.11798)
Comments:
          11 pages, submitted to SC24 conference

- **What's New**: 최근 추측 기반 추론(speculative inference) 기법들을 개선하기 위해 PipeInfer라는 새로운 파이프라인 방식의 추측 가속 기법을 제안했습니다. 이 기법은 단일 요청 시나리오에서도 시스템 활용도를 개선하고 추측 수용률이 낮거나 대역폭이 낮은 인터커넥트에서의 성능을 향상시킵니다. PipeInfer는 연속 비동기 추측 연속 비동기 추론(Continuous Asynchronous Speculation and Early Inference Cancellation)을 통해 이러한 성능 개선을 달성합니다.

- **Technical Details**: PipeInfer는 파이프라인 병렬 아키텍처를 사용하여 단일 토큰 추론을 여러 추측 실행과 동시에 실행하여 지연 시간과 생성 속도를 개선합니다. 또한 추측이 무효화되면 계산을 중간에 취소하여 계산 시간이 낭비되지 않도록 합니다. 파이프라인 KV 캐시 멀티 버퍼링(Pipelined KV Cache Multibuffering)을 통해 생성된 토큰의 인과 관계를 유지하며, 연속적인 추측(Continuous Speculation) 방식으로 작은 마이크로 배치(micro-batch)를 사용하여, 저 대역폭 시나리오에서도 적응력을 높였습니다.

- **Performance Highlights**: PipeInfer는 표준 추측 기반 추론에 비해 최대 2.15배의 생성 속도 개선을 보여주며, 잘 정렬된 모델(well-aligned models)에서는 최대 1.7배, 잘 정렬되지 않은 모델(poorly aligned models)에서는 최대 2.15배의 속도 개선이 나타났습니다. 기가비트 이더넷을 사용한 테스트에서는 낮은 지연 시간 및 대역폭 제한이 있는 시나리오에서도 상당한 성능 향상을 나타내었습니다.



### Large Language Models as Misleading Assistants in Conversation (https://arxiv.org/abs/2407.11789)
Comments:
          Next Generation of AI Safety Workshop, 41st International Conference on Machine Learning (ICML 2024)

- **What's New**: 최근 연구에서는 대형 언어 모델(LLMs)이 독해력을 향상시키는데 도움을 줄 수 있지만, 또한 의도적으로 혹은 비의도적으로 사용자를 오도할 가능성이 있음을 밝혔다. 연구진은 GPT-4가 GPT-3.5-Turbo와 GPT-4를 효과적으로 잘못된 방향으로 이끌 수 있으며, 이는 다양한 상황에서의 큰 위험성을 시사한다.

- **Technical Details**: 연구진은 '진실한' 도움, '교묘하게 잘못된' 정보 제공, 그리고 '잘못된 답변 주장' 등 세 가지 시나리오에서 LLM의 행동을 비교했다. 이를 위해 달성하기 어려운 독해 과제를 설정하고 한 모델은 '사용자'로, 다른 모델은 '도움 제공자(Assistant)'로 설정하여 실험을 진행했다.

- **Performance Highlights**: 실험 결과에 따르면, 잘못된 정보를 제공하는 모델의 경우 사용자 모델의 정확도가 최대 23%까지 감소했다. 하지만 사용자에게 더 많은 문맥 정보를 제공하면 잘못된 정보의 영향을 어느 정도 완화할 수 있었다는 점도 발견되었다.



### SwitchCIT: Switching for Continual Instruction Tuning of Large Language Models (https://arxiv.org/abs/2407.11780)
- **What's New**: 이 논문에서는 거대한 언어 모델(LLMs)의 지속적인 학습에서 발생하는 캐타스트로픽 포겟팅(catastrophic forgetting)을 해결하기 위한 새로운 방법을 제시합니다. SwitchCIT이라는 방법을 통해 각 작업에 대해 파라미터 추가를 통한 미세 조정(parameter-efficient fine-tuning)을 수행하며, 새로운 작업에 맞는 모델을 선택 및 분류하여 성능 저하를 방지합니다.

- **Technical Details**: SwitchCIT은 작업별 지시 벡터(instruction vectors)의 군집화 현상을 활용하여 지시문에서 작업을 식별하는 스위치 네트워크를 도입합니다. 새로운 작업이 주어질 때마다, Low-Rank Adaptation(LoRA)과 같은 PEFT(파라미터 효율적 미세조정) 방법을 사용하여 추가 파라미터를 생성합니다. 이로써 모든 가중치(weight)를 특정 작업에 맞게 조정할 수 있으며, 이전에 배운 작업을 상기하는 데 필요한 파라미터를 자동으로 추가합니다.

- **Performance Highlights**: 논문에서는 다섯 가지 자연어 생성 작업에 대한 실험을 통해, SwitchCIT 방법이 여러 기준 모델들과 비교하여 우월한 성능을 보임을 입증했습니다. SwitchCIT은 성능 저하 없이 새로운 작업을 학습하며, 파라미터 효율성을 극대화합니다.



### Sharif-MGTD at SemEval-2024 Task 8: A Transformer-Based Approach to Detect Machine Generated Tex (https://arxiv.org/abs/2407.11774)
Comments:
          8 pages, 3 figures, 2 tables. Proceedings of the 18th International Workshop on Semantic Evaluation (SemEval-2024)

- **What's New**: 이번 연구는 자연어처리(NLP) 분야에서 기계 생성 텍스트(Machine-Generated Text, MGT)를 감지하는 방법을 탐구합니다. 특히 RoBERTa-base transformer 모델을 미세 조정(fine-tuning)하여 MGT 감지를 이진 분류(binary classification) 작업으로 수행한 결과를 다룹니다. 제안된 시스템은 SemEval-2024 대회에서 Monolingual-English 하위 과제(Subtask A)에 대해 78.9%의 정확도를 달성했습니다.

- **Technical Details**: 제안된 시스템은 RoBERTa-base transformer 모델을 사용하여 입력 텍스트를 인간이 작성한 글과 기계가 생성한 글로 이진 분류합니다. 모델은 Embeddings, Encoder, 그리고 Classifier Head로 구성되어 있으며, 이를 통해 문맥적 이해와 입력 텍스트의 병렬 처리를 수행합니다. 데이터셋은 Wang et al. (2023) 에 의해 제공된 것으로, 총 119,757개의 훈련 예제와 5,000개의 개발 예제로 구성됩니다.

- **Performance Highlights**: 제안된 시스템은 테스트 데이터에서 78.9%의 정확도를 기록하며, 140명의 참가자 중 57위를 차지했습니다. 또한, ROC 곡선 아래의 면적(AUC) 지표는 0.69로 측정되었습니다. 이는 제안된 모델이 상당수의 긍정 사례를 잘 분류할 수 있음을 보여주지만, 정확도를 높이기 위한 여지가 남아 있음을 나타냅니다.



### Educational Personalized Learning Path Planning with Large Language Models (https://arxiv.org/abs/2407.11773)
Comments:
          6 pages

- **What's New**: 대규모 언어 모델 (Large Language Models, LLMs)과 프롬프트 엔지니어링(prompt engineering)을 통합하여 교육 맞춤형 학습 경로 계획(Personalized Learning Path Planning, PLPP)의 새로운 접근법을 제안합니다. 학습자의 개별 정보를 포함하는 프롬프트를 설계함으로써, LLama-2-70B와 GPT-4 같은 LLM들이 맞춤형이고 교육학적으로 타당한 학습 경로를 생성하도록 안내합니다.

- **Technical Details**: 우리의 방법은 학습자의 기존 지식, 학습 목표 및 선호도를 통합한 프롬프트를 설계하여 맞춤형 학습 경로를 생성합니다. 예를 들어, 프롬프트는 다음과 같이 구성될 수 있습니다: '[주제]에 대한 학습자의 이해를 기반으로 [과목]에서 마스터하기 위해 학습해야 할 다음 세 가지 개념을 제안해주세요.' 여러 턴의 대화를 통해 LLM이 질문을 하여 추천 사항을 세밀하게 조정할 수 있습니다.

- **Performance Highlights**: 실험 결과는 제안된 방법이 모든 측면에서 상당한 개선을 보여줍니다. 특히 GPT-4 모델에서는 정확성, 사용자 만족도 및 학습 경로의 질에 있어 뛰어난 성능을 입증합니다. 장기적인 영향 분석에서도 학습자 성과 및 유지를 개선하는 가능성을 확인했습니다. 이는 LLM과 프롬프트 엔지니어링이 맞춤형 교육을 발전시키는 데에서 큰 잠재력을 가지고 있음을 강조합니다.



### Robust Utility-Preserving Text Anonymization Based on Large Language Models (https://arxiv.org/abs/2407.11770)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)에 의한 재식별 공격 위협으로부터 민감한 데이터의 프라이버시를 보호하면서 데이터 유틸리티를 유지하기 위해 새로운 프레임워크를 제안합니다. 제안된 프레임워크는 텍스트 익명화를 수행하기 위해 프라이버시 평가기(Privacy Evaluator), 유틸리티 평가기(Utility Evaluator), 최적화 컴포넌트(Optimization Component)라는 세 가지 LLM 기반 구성 요소로 구성됩니다.

- **Technical Details**: 프라이버시 보호와 데이터 유틸리티 사이의 균형을 최적화하기 위해, 제안된 프레임워크는 순차적으로 텍스트를 익명화하고, 익명화된 텍스트의 프라이버시 보호 수준과 다운스트림 작업의 성능을 평가합니다. 이 과정은 Direct Preference Optimization(DPO) 기법을 활용하여 경량화된 모델로 익명화 기능을 구현합니다.

- **Performance Highlights**: 광범위한 실험 결과, 제안된 RUPTA(Robust Utility-Preserving Text Anonymization) 프레임워크가 기존의 기법들보다 뛰어난 성능을 보였습니다. 특히, 재식별 위험을 줄이면서도 다운스트림 작업의 데이터를 효과적으로 활용할 수 있는 유틸리티 유지를 달성하였습니다. 또한 경량화된 모델은 실시간 및 대규모 환경에서도 실용적인 성능을 보여줍니다.



### Vectoring Languages (https://arxiv.org/abs/2407.11766)
Comments:
          12 pages including references

- **What's New**: 이 논문에서는 기존 방법들보다 언어의 다양한 속성을 잘 포착할 수 있는 새로운 언어 구조를 제안합니다. 이 구조는 언어 모델의 메커니즘을 잘 반영하며, 선형대수학의 비유를 통해 이 관점을 강화합니다. 이런 관점의 차이가 현재 언어 모델의 설계 철학과 어떤 차이점이 있는지를 논의하며, 이 새로운 관점이 과학 발전을 가속화할 수 있는 연구 방향을 제시할 수 있다고 주장합니다.

- **Technical Details**: 이 논문의 핵심은 'vectoring'이라는 개념을 도입하여, 언어를 고차원 벡터 공간(high-dimensional vector space)으로 취급하는 것입니다. 벡터 공간의 비유를 보다 구체화하기 위해 선형대수학의 개념들을 느슨하게 도입합니다. 기존 NLP, LLM, 신경과학(neuroscience) 등의 다양한 연구와 연결시키며, 특히 인간 철학자와 AI 모델의 언어 공간의 간극을 좁히려는 시도를 강조합니다.

- **Performance Highlights**: 이 벡터링(vectoring) 관점에서 언어를 이해하는 것은, 단어의 의미(meaning), 발음(sound), 시의 미학(aesthetics) 등 여러 속성을 동시에 고려할 수 있게 합니다. 이는 인간이 언어를 사용하는 자연스러운 방법을 더 잘 반영하며, 최근의 AI 과학의 성과와도 직접 연결됩니다. 또한, 이 논문은 언어 벡터 공간을 특정 하위 공간(subspace)으로 투영(projection)하여 언어의 특정 속성을 더 잘 이해할 수 있는 방법을 제안합니다.



### How Are LLMs Mitigating Stereotyping Harms? Learning from Search Engine Studies (https://arxiv.org/abs/2407.11733)
Comments:
          Accepted at AAAI/ACM AI, Ethics, and Society

- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 자기완성(prompt)에 대한 새로운 평가 과제를 제안하고, 이러한 모델들이 생성하는 고정관념(stereotyping) 및 그에 따른 사회적 영향을 평가하고자 합니다. 연구는 ChatGPT 등 LLM의 상업적 개발이 법적 책임에 초점을 맞추고 있어 사회적 영향 평가가 소홀해진 경향이 있음을 지적합니다.

- **Technical Details**: LLM 평가를 위해 자기완성 스타일의 프롬프트를 사용하여 네 가지 메트릭(refusal rates, toxicity, sentiment, and regard)으로 처리합니다. 일곱 개의 최신 LLM을 프롬프트로 구성하여 170개 이상의 사회 집단에 대한 고정관념을 평가했습니다. 다양한 양적 평가 메트릭을 사용하여 모델 응답의 억제율, 독성 결과, 긍정도 및 암묵적 고정관념 지표를 연구했습니다.

- **Performance Highlights**: LLMs는 사회 그룹별로 고정관념에 대한 억제와 독성 면에서 뚜렷한 차이를 보였습니다. Llama-2는 대부분의 고정관념 유도 프롬프트를 거부한 반면, Starling 모델은 가장 긍정적인 응답을 생성했습니다. Falcon 모델은 가장 높은 독성 반응을 보였습니다. 특정 민족 및 성적 지향에 대한 프롬프트는 상대적으로 더 많은 독성 반응과 거부를 유도했습니다. 모델에 안전 시스템 프롬프트를 추가했을 때 고정관념에 대한 개선은 있었으나, 문제를 완전히 해결하지는 못했습니다.



### CCoE: A Compact LLM with Collaboration of Experts (https://arxiv.org/abs/2407.11686)
Comments:
          15 pages

- **What's New**: 최근의 대형 언어 모델(LLM) 연구와 관련하여 새로운 CCoE 아키텍처가 제안되었습니다. 이는 다양한 도메인 전문가 모델을 쉽게 결합하여 하나의 큰 LLM으로 통합하는 프레임워크로, 저비용으로 여러 도메인에서 우수한 성능을 발휘할 수 있게 합니다. 5개의 도메인(Code, Math, Law, text-to-SQL, Medical) 전문가를 사용하여 이 프레임워크를 테스트한 결과, 본래의 기본 모델에 비해 성능이 약 10%-20% 향상되었습니다.

- **Technical Details**: CCoE 아키텍처는 전문가(CoE) 레이어를 통해 여러 도메인 전문가 LLM를 연결합니다. 각 CoE 레이어는 하나 이상의 도메인 전문가 LLM을 포함할 수 있으며, 이들 전문가 LLM들은 특정 도메인 작업을 위해 잘 훈련되어 있습니다. 이를 통해 자원 소모를 최소화하면서 동시에 모델의 성능을 높이는 것이 가능합니다. 또한, 'pop'과 'push'라는 두 가지 연산을 통해 각 전문가를 개별적으로 미세 조정할 수 있습니다.

- **Performance Highlights**: CCoE 프레임워크를 통해 다양한 도메인에서 약 10%-20%의 성능 향상이 이루어졌습니다. 특히, 각 도메인별로 효율적으로 미세 조정된 전문가들을 결합함으로써 기본 모델 대비 적은 자원으로도 높은 성능을 발현할 수 있는 장점이 있습니다.



### MINI-LLM: Memory-Efficient Structured Pruning for Large Language Models (https://arxiv.org/abs/2407.11681)
Comments:
          13 pages

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 효율적인 가지치기를 위한 새로운 방법을 제안합니다. 기존의 가지치기 방법들이 주로 경사도(gradients) 없이 가중치 크기(weight magnitude) 또는 활성화(activation)를 기준으로 삼은 반면, 이 연구는 가중치 크기, 활성화, 그리고 경사도를 적절히 통합한 혼합 가지치기 기준을 제안합니다. 이를 통해 메모리 부하를 줄이면서도 경사 정보의 활용도를 극대화하는 Memory-effIcieNt structured prunIng procedure for LLMs (MINI-LLM)을 개발했습니다.

- **Technical Details**: MINI-LLM은 경사도 추정을 위해 전방패스(forward passes)만을 사용하며, 기존의 경사도 계산을 위한 역전파(backpropagation)가 요구하는 메모리 부담을 해결합니다. 주요 기여로는 'Feature Map Sensitivity (FMS)' 점수를 설계하여 가중치 크기, 활성화, 경사도를 통합한 새로운 가지치기 기준을 제안하고, 이를 통해 세밀한 평가가 가능합니다. 또한, 전방패스를 통해 경사도를 추정함으로써 GPU 메모리 효율성을 극대화했습니다.

- **Performance Highlights**: 실험 결과, LLaMA, BLOOM, OPT 등 세 가지 LLM에서 MINI-LLM이 다양한 다운스트림 작업(분류, 선택형 문제, 생성)에서 기존의 경사도 없는 가지치기 방법보다 성능이 우수함을 입증했습니다. 특히, 메모리 사용량은 경사도 없는 방법과 유사하면서도 성능면에서 경사기반 방법을 능가하거나 비슷한 성능을 보여주었습니다.



### ECoh: Turn-level Coherence Evaluation for Multilingual Dialogues (https://arxiv.org/abs/2407.11660)
Comments:
          Accepted to SIGDIAL 2024

- **What's New**: 이번 연구에서는 경량, 오픈 소스, 다국어 대화 평가자를 필요로 하는 이유로 GenResCoh (Generated Responses targeting Coherence)를 소개합니다. 이는 XDailyDialog와 XPersona에서 파생된 13만 개 이상의 긍정 및 부정 응답과 해당 설명으로 구성된 새로운 데이터셋입니다. 이를 통해 우리는 다국어 응답 일관성을 평가하기 위해 ECoh (Evaluation of Coherence) 평가자들을 제안합니다.

- **Technical Details**: GenResCoh는 다국어 대응 및 일관성 있는 응답 생성에 초점을 맞춘 데이터셋으로, 인공지능 대화 모델(LLM)을 사용해 생성된 대규모 응답 데이터셋입니다. 영어, 프랑스어, 독일어, 이탈리아어, 중국어로 작성된 응답과 설명을 포함하며 LLM을 활용해 긍정적인 샘플과 의미적으로 관련되지만 일관성과 논리적 일관성에서 문제가 있는 부정적인 샘플을 얻었습니다. ECoh는 이 데이터셋을 기반으로 학습되어 다국어 일관성 평가에서 우수한 성능을 보입니다.

- **Performance Highlights**: ECoh 모델은 GenResCoh에서 교사 모델인 GPT-3.5-Turbo(0.910)보다 뛰어난 성능을 보였으며, F1 점수 0.945를 기록하며 Qwen1.5-7B-Chat 모델(0.825)보다도 월등한 성능을 보였습니다. 또한, ECoh의 설명은 Qwen1.5-7B-Chat보다 높은 품질을 보였으며, 이는 GPT-4 평가에서도 대부분 5점 만점에 4점 이상을 기록했습니다.



### A Comprehensive Evaluation of Large Language Models on Temporal Event Forecasting (https://arxiv.org/abs/2407.11638)
- **What's New**: 최근의 대형 언어 모델(LLMs)은 지식 질문 응답, 수학적 추론, 상식 추론 등 다양한 데이터 마이닝 작업에서 큰 잠재력을 보여주고 있습니다. 하지만 LLMs의 시간적 이벤트 예측 능력은 아직 충분히 탐구되지 않았습니다. 이를 체계적으로 조사하기 위해, 시간적 이벤트 예측을 위한 LLM 기반 방법들의 종합적인 평가를 수행하였습니다. 이를 위해 그래프 및 텍스트 데이터 모두를 포함하는 고품질 데이터 세트가 부족한 문제를 해결하기 위해, 먼저 MidEast-TE-mini라는 벤치마크 데이터셋을 구축했습니다.

- **Technical Details**: 이 데이터셋을 기반으로, 다양한 입력 형식과 Retrieval Augmented Generation (RAG) 모듈로 특징지어진 일련의 베이스라인 방법을 설계했습니다. 광범위한 실험을 통해 발견한 주요 사항은, 원본 텍스트를 LLM의 입력에 직접 통합하는 것은 제로샷 외삽 성능을 향상시키지 않는다는 점입니다. 반면, 특정 복합 이벤트에 원본 텍스트를 통합하고 LLM을 미세 조정하면 성능이 크게 향상됩니다. 또한 검색 모듈이 강화된 LLM은 역사적 이벤트에 숨겨진 시간적 관계 패턴을 효과적으로 포착할 수 있습니다.

- **Performance Highlights**: 이번 연구 결과는 LLM 기반 이벤트 예측 방법에 대한 이해를 깊게 할 뿐만 아니라, LLM을 통한 시간적 이벤트 예측의 향후 연구에 크게 기여할 여러 유망한 연구 방향을 강조합니다. 특히, 인기 편향과 긴 꼬리 문제와 같은 이슈가 여전히 존재하지만(RAG 기반 방법에서 특히 두드러짐), 이러한 문제를 해결하기 위한 새로운 접근법을 제안할 수 있습니다.



### The Foundations of Tokenization: Statistical and Computational Concerns (https://arxiv.org/abs/2407.11606)
- **What's New**: 이 논문은 NLP(자연어 처리) 파이프라인에서 중요한 단계인 토큰화(tokenization)에 대해 이론적 기초를 제공하려는 목적을 가지고 있습니다. 현재 널리 사용되는 엔드 투 엔드 신경망 모델에 완전히 통합되지 않은 유일한 주요 단계인 토큰화의 이론적 격차를 해결하고자 합니다.

- **Technical Details**: 이 논문은 확률적 맵(stochastic maps)의 범주에 대한 기본 속성을 설명하고 확장함으로써 토크나이저(tokenizer) 모델을 표현하고 분석하기 위한 통합 프레임워크를 제안합니다. 이를 통해 통계 추정량의 일관성을 유지하기 위한 토크나이저 모델의 필요조건과 충분조건을 공식적으로 확립할 수 있게 됩니다. 또한 토크나이저 모델의 설계 및 구현에 중요한 통계적, 계산적 문제들을 논의합니다.

- **Performance Highlights**: 제안된 프레임워크와 결과는 신경 언어 모델링(neural language modeling) 분야에 강력한 이론적 기초를 마련하는 데 중요한 단계로 작용합니다. BPE(Byte Pair Encoding), WordPiece, Unigram 등의 최근의 데이터 기반 하위어(subword) 모델들이 널리 채택되었으며, 이러한 토크나이저들은 개방형 어휘를 통해 언어 모델을 학습할 수 있는 능력과 효율적이고 손실 없는 데이터 인코딩을 제공합니다.



### AdaptEval: Evaluating Large Language Models on Domain Adaptation for Text Summarization (https://arxiv.org/abs/2407.11591)
- **What's New**: AdaptEval 공개, 도메인 적응 평가를 위한 최초의 도메인 적응 평가 스위트가 도입되었습니다. 이는 다양한 도메인에서 LLMs의 요약 성능을 평가하기 위한 도메인 벤치마크와 다양한 메트릭을 포함합니다.

- **Technical Details**: 다양한 파라미터 크기를 가진 11개의 모델을 사용하여 요약 작업에서 도메인 적응 능력을 평가했습니다. 이를 위해 fine-tuning과 인컨텍스트 러닝(In-context Learning, ICL) 설정에서 실험을 진행했으며, 성능을 ROUGE, BERTScore, 도메인 어휘 중복(DVO) 및 G-eval 등을 통해 평가하였습니다.

- **Performance Highlights**: 실험 결과, ICL 설정에서는 작은 7b 모델도 두 개의 학습 예제만으로 큰 파라미터를 가진 모델과 유사한 성능을 보였습니다. 반면, 주로 fine-tuning 모델은 자동 점수 측면에서 최고 성능을 보였으나, 도메인 어휘 적응에서는 ICL보다 열등한 결과를 보였습니다. 특히 PEGASUS-X 모델은 최고의 ROUGE 점수를 기록했습니다.



### Optimizing KV Cache Eviction in LLMs: Adaptive Allocation for Enhanced Budget Utilization (https://arxiv.org/abs/2407.11550)
- **What's New**: 최근 대형 언어 모델 (LLMs)은 다양한 분야에서 뛰어난 성과를 보였으나, 긴 시퀀스 추론 시 필요한 KV 캐시의 크기로 인해 효율성에 한계를 보입니다. 기존의 캐시 퇴출(Eviction) 방법들은 이러한 캐시 크기를 줄이는 데 초점을 맞추지만, 균일한 예산 할당이 품질 저하를 초래하는 문제점이 있습니다. 본 연구는 이러한 문제를 해결하기 위해 적응형 예산 할당 알고리즘을 제안하고, 이를 두 가지 최신 방법과 통합하여 Ada-SnapKV와 Ada-Pyramid를 개발했습니다. 이 알고리즘은 이론적으로 이전의 균일 할당 방법보다 상한선을 초과하지 않으며, 실험적으로도 상한선을 줄이는 효과가 있습니다.

- **Technical Details**: 기존의 캐시 퇴출 전략은 퇴출 손실(eviction loss)의 상한을 최소화하려는 목표를 가지고 있습니다. 본 연구에서는 이러한 전략이 주어진 예산 할당 내에서 자기-주의 메커니즘(self-attention mechanism)의 출력을 기준으로 L1 거리를 측정하는 상한선을 최소화함을 발견했습니다. 그러나, 균일하게 예산을 할당하는 현재의 방법은 생성 품질을 저하시키는 원인이 됩니다. 이에 따라, 우리는 자기-주의 메커니즘의 특성을 반영한 적응형 예산 할당 알고리즘을 제안했습니다.

- **Performance Highlights**: 제안된 Ada-SnapKV와 Ada-Pyramid는 16개의 데이터셋과 Needle-in-a-Haystack 테스트에서 실험적으로 검증되었습니다. 그 결과, 두 방법 모두 새로운 벤치마크를 설정하며 현존하는 최고 성능을 달성했습니다. 또한, 이 두 방법은 긴 문맥을 유지하고 검색하는 능력을 현저히 향상시켰습니다.



### How Personality Traits Influence Negotiation Outcomes? A Simulation based on Large Language Models (https://arxiv.org/abs/2407.11549)
Comments:
          13 pages, 4 figures

- **What's New**: 이번 논문은 합성된 성격 특성을 갖춘 대규모 언어 모델(LLM) 에이전트를 사용한 협상 시뮬레이션 프레임워크를 소개합니다. 이 에이전트들은 협상 도메인 내에서 협상에 임하며, 상황에 맞게 성격과 목표가 조정될 수 있습니다. 실험 결과, LLM 기반 시뮬레이션의 행동 경향이 사람의 협상에서 관찰되는 행동 패턴을 재현할 수 있음을 보여줍니다.

- **Technical Details**: 이번 연구는 LLM 에이전트가 가진 언어적 및 경제적 능력의 정렬을 조사하는 시뮬레이션 방법론을 제안합니다. 합성된 성격은 인컨텍스트 러닝(in-context learning)을 사용하여 특정 성격 프로파일로 에이전트를 구성하고, 협상 목표는 협상 과제와 목표를 명시하는 지침을 통해 주어집니다. 구매자와 판매자 에이전트로 구성된 경쟁적인 협상 시나리오에서 각 라운드의 협상 대화에서 제시된 가격과 전략을 평가하고 분석합니다.

- **Performance Highlights**: LLM 기반 시뮬레이션의 경향이 인간 실험과 일반적으로 일치함을 보여줍니다. 합성 협상 대화를 기반으로 한 사례 연구에서는 속임수, 감정적 호소 및 '이대로 받아들이거나 떠나' 전략과 같은 흥미로운 행동 패턴을 드러냅니다. 이러한 결과는 LLM이 인간의 대화 스타일을 모방할 뿐만 아니라 의사 결정 패턴도 포착할 수 있음을 시사합니다.



### Fine-Tuning Medical Language Models for Enhanced Long-Contextual Understanding and Domain Expertis (https://arxiv.org/abs/2407.11536)
Comments:
          5 pages, 1 figure. Accepted by the Workshop on Long-Context Foundation Models (LCFM) at ICML 2024

- **What's New**: 이 연구는 전문 분야 언어 모델(Large Language Models, LLMs)의 긴 문맥 이해 능력이 분야 특화 데이터로 미세조정(fine-tuning)되면서 저하되는 현상을 조사합니다. 특히 의료 분야의 LLM에서 이러한 문제가 두드러지며, 이를 해결하기 위해 일반 데이터와 전문 데이터의 최적 비율을 찾고자 합니다.

- **Technical Details**: 연구진은 다양한 중국어 의학 시험을 활용해 모델의 문맥 이해력과 지시 따름 능력을 평가했습니다. 공개적으로 사용 가능한 일반 LLM과 의료 LLM을 비교했으며, 의료 LLM의 긴 문맥 이해력이 상대적으로 떨어지는 것을 발견했습니다. 이를 해결하기 위해 일반 데이터를 활용한 미세조정(fine-tuning) 실험을 추가로 수행했습니다.

- **Performance Highlights**: 일반 데이터를 사용하여 미세조정한 결과, 의료 LLM의 문맥 이해력이 향상되는 것을 확인했습니다. 특히 의료 능력이 뛰어난 HuatuoGPT-II 모델의 경우, 일반 데이터로 추가 미세조정할 때 문맥 이해력이 크게 향상되었습니다. 이는 문맥 이해력과 전문 능력 간의 상충(trade-off)이 존재할 수 있음을 시사하며, 모델의 성능 최적화를 위해 일반 데이터의 중요성을 강조합니다.



### Scientific QA System with Verifiable Answers (https://arxiv.org/abs/2407.11485)
Comments:
          Accepted at the 6th International Open Search Symposium 2024. arXiv admin note: substantial text overlap with arXiv:2402.18589

- **What's New**: 본 논문에서는 VerifAI 프로젝트를 소개합니다. VerifAI는 오픈 소스 과학 질문-응답 시스템으로, 참고 자료가 명시된 답변을 자동으로 검증할 수 있게 설계되었습니다.

- **Technical Details**: 이 시스템의 주요 구성 요소는 다음과 같습니다. (1) 과학 논문(PubMed)을 대상으로 하는 의미적 및 어휘적 검색 기술을 결합한 정보 검색 시스템(Information Retrieval system), (2) 미스트랄 7B(Mistral 7B)로 미세 조정된 생성 모델을 사용하고 검색된 기사를 활용하여 참조된 기사를 기반으로 주장을 생성하는 검색 증강 생성(RAG: Retrieval-Augmented Generation) 모듈, (3) SciFACT 데이터셋을 사용한 자연어 추론(NLI: Natural Language Inference) 작업에 미세 조정된 DeBERTa 및 XLM-RoBERTa 모델 기반 검증 엔진입니다. 검증 엔진은 생성된 주장과 이 주장이 도출된 기사를 대조하여 주장을 생성하면서 발생했을 수 있는 환각(hallucination)을 확인합니다.

- **Performance Highlights**: 정보 검색 및 RAG 모듈을 통해 방대한 과학 자료에서 사실적 정보를 생성하고, 검증 엔진이 이 결과를 엄격하게 이중 확인함으로써 그 정확성과 신뢰성을 보장합니다. 이 이중 단계 프로세스는 사실적 정보를 획득하고 확인하는 데 중요한 역할을 하여 정보 환경을 크게 향상시킵니다. 이 방법론은 과학자들의 생산성을 높이는 동시에 과학 분야에서 생성적 언어 모델을 적용할 때 신뢰성을 강화할 수 있습니다.



### Trust No Bot: Discovering Personal Disclosures in Human-LLM Conversations in the Wild (https://arxiv.org/abs/2407.11438)
- **What's New**: 이 연구는 인간과 챗봇 상호작용에서 개인 정보 유출 정도를 측정하여 사용자들의 AI 이해력을 파악하고 대규모 언어 모델(LLMs)의 개인정보 보호 연구를 촉진하는 것을 목표로 합니다. 실제 사용자들이 상업적인 GPT 모델에게 얼마나 많은 개인 정보를 유출하는지에 대한 정밀한 분석을 수행하였습니다.

- **Technical Details**: 연구에서는 자연 발생 대화를 정성적 및 정량적으로 분석하여 과업(task)과 민감한 주제에 대한 분류 체계를 개발하였습니다. WildChat 데이터셋(Zhao et al., 2024)과 ShareGPT 데이터셋(Chiang et al., 2023)의 1백만 개의 대화를 분석하였으며, 주요 과업으로 번역(translation), 코드 편집(code editing) 등을 포함한 21가지 카테고리를 구분하였습니다. 민감한 정보는 개인식별정보(PII)뿐만 아니라 성적 선호도나 특정 약물 사용 습관 등의 주제 또한 포함됩니다.

- **Performance Highlights**: 연구 결과, 예상치 못한 맥락에서 개인식별정보(PII)가 48%의 번역 요청과 16%의 코드 편집 요청에서 등장함을 관찰했습니다. PII 검출 기능만으로는 유저-챗봇 상호작용에서 흔히 나타나는 민감한 주제를 포착하는 데 충분하지 않다는 결론에 도달하였습니다. 총 70% 이상의 쿼리에서 어떤 형태로든 PII가 감지되었고, 약 15%에서는 성적 선호도나 약물 사용 등 비-PII 민감 주제가 언급되었습니다.



### States Hidden in Hidden States: LLMs Emerge Discrete State Representations Implicitly (https://arxiv.org/abs/2407.11421)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 새로운 능력, 즉 연속적인 덧셈을 단계별(chain-of-thought) 추론 없이 직접 계산해내는 능력을 발견했습니다. 최첨단 모델(SOTA)은 최대 15개의 두 자릿수 숫자의 덧셈 결과를 즉시 출력할 수 있습니다. 이 모델은 암시적 이산 상태 표현(Implicit Discrete State Representations, IDSRs)을 숨겨진 상태에서 형성하고 이를 내부적으로 계산하여 결과를 도출한다고 가정합니다.

- **Technical Details**: IDSR이 존재하는지 확인하기 위해 다양한 실험을 설계했습니다. 먼저 IDSR의 존재를 확인한 후, 레이어, 숫자, 시퀀스 관점에서 IDSR의 형성에 대한 흥미로운 관찰을 수행했습니다. 저자들은 IDSR이 형성되는 동안 낮은 자릿수부터 독립적으로 순차적으로 형성되는 것을 발견했습니다. 모델의 얕은 레이어에서는 의미론적 계산이 일어나고, 깊은 레이어에서는 비선형적인 처리가 이루어집니다. 시퀀스 관점에서 IDSR에 인코딩된 정보는 순차적으로 전파되어 최종 결과를 도출하는 데 사용됩니다.

- **Performance Highlights**: 연구 결과, 모델은 IDSR을 통해 정확한 결과를 도출할 수 있지만, 현재 공개 소스 모델에서는 이 상태 표현이 손실 없이 이루어지지 않아 최종 성능에 부정확함이 나타났습니다. 또한, 이 연구는 LLM의 다단계 추론과 상태 추적 능력에 대한 새로운 인사이트를 제공합니다.



### SPINACH: SPARQL-Based Information Navigation for Challenging Real-World Questions (https://arxiv.org/abs/2407.11417)
- **What's New**: 이번 연구는 지식 기반 질문 응답(KBQA) 작업에서 기존의 단순한 질문이나 소형 지식 기반(KB) 스키마에 의존하는 데이터셋이 아닌 실제 복잡한 질문을 반영하는 새로운 SPINACH 데이터셋을 소개합니다. 이 데이터셋은 Wikidata의 'Request a Query' 포럼에서 수집된 질문-SPARQL 쌍 320개로 구성되어 있으며, 도전적인 질문에 대응할 수 있는 강력한 KBQA 시스템을 요구합니다.

- **Technical Details**: SPINACH 데이터셋은 실제 포럼 토론에서 추출된 decontextualized 질문-SPARQL 쌍으로 구성되어 있으며, 기존의 간단한 질문이나 인위적으로 생성된 논리적 형식과는 다릅니다. 또한, SPINACH 에이전트를 소개하며, 이는 LLMs와 KG(지식 그래프) 추론을 통합하여 전문 지식 작업을 모방한 새로운 KBQA 접근법입니다. 이 에이전트는 QALD-7, QALD-9 Plus 및 QALD-10 데이터셋에서 30.1%, 27.0%, 10.0%의 F1 점수 향상과 WikiWebQuestions에서 미세 조정된 SOTA 모델보다 1.6% 이내의 성과를 보였습니다.

- **Performance Highlights**: SPINACH 에이전트는 QALD-7, QALD-9, QALD-10 데이터셋에서 각각 30.1%, 27.0%, 10.0%의 F1 점수 향상을 이루었고, WikiWebQuestions에서는 미세 조정된 SOTA 모델보다 1.6% 이내의 성능을 기록했습니다. 또한, SPINACH 데이터셋에서는 GPT-4 기반 KBQA 에이전트를 포함한 모든 기준을 38.1% F1 점수로 능가했습니다.



### Representation Bias in Political Sample Simulations with Large Language Models (https://arxiv.org/abs/2407.11409)
- **What's New**: 이 연구는 대형 언어 모델(Large Language Models, LLMs)을 이용한 정치적 표본 시뮬레이션에서의 편향을 식별하고 정량화하는 것을 목표로 합니다. GPT-3.5-Turbo 모델을 사용하여 미국, 독일, 중국 등 여러 국가의 데이터로 투표 행동과 공공 여론을 시뮬레이션합니다.

- **Technical Details**: GPT-3.5-Turbo 모델을 사용하여 American National Election Studies(ANES), German Longitudinal Election Study(GLES), Zuobiao Dataset 및 China Family Panel Studies(CFPS) 데이터를 기반으로 투표 행동과 공공 여론을 시뮬레이션하였습니다. 이 연구는 세 가지 유형의 대표성 편향(representation bias)을 평가합니다: 영어 사용 국가 대 비영어 사용 국가, 특정 인구 그룹 대 다른 인구 그룹, 민주주의 대 권위주의 정권.

- **Performance Highlights**: 전체적으로, 투표 선택의 시뮬레이션 성능은 공공 여론보다 우수했습니다. 영어 사용 국가에서 더 정확하게 예측했으며, 양당제에서 다당제보다, 민주주의 체제에서 권위주의 체제보다 더 나은 성능을 보였습니다. 또한, 특정 민주 그룹 간에도 성능 차이가 나타났으며, 특히 젊은 인구 그룹에서는 시뮬레이션 예측력이 낮았습니다.



### Revisiting the Impact of Pursuing Modularity for Code Generation (https://arxiv.org/abs/2407.11406)
Comments:
          9 pages, 7 figures

- **What's New**: 최신 연구에서는 코드 생성(Large Language Model)을 위한 코드 생성 모델에서 모듈화(Modularity)의 영향에 대해 조사했습니다. 연구자들은 코드 모듈화의 정량적 측정을 위한 새로운 메트릭(Metric)을 도입하였으며, 이를 통해 모듈화된 코드가 실제로 성능 향상에 필수적이지 않음을 발견했습니다.

- **Technical Details**: 연구에서는 Cyclomatic Complexity (CC) 개념을 사용하여 코드 모듈의 이상적인 수인 m*을 결정했습니다. CC는 소스 코드의 복잡도를 측정하는 방법으로, 독립적인 실행 경로의 수를 셉니다. 이를 기반으로 모듈화 점수(MoS)를 정의하고, CC 값의 평균이 일정 임계값(τ)과 같도록 설정했습니다.

- **Performance Highlights**: 다양한 코드 컬렉션을 실험한 결과, 코드의 모듈화가 성능 향상에 중요한 요소가 아니라는 점이 발견되었습니다. 연구팀은 LLM이 모듈화된 코드보다 비모듈화된 코드에 대해 선호를 보이지 않는 잠재적 이유도 탐구했습니다.



### InvAgent: A Large Language Model based Multi-Agent System for Inventory Management in Supply Chains (https://arxiv.org/abs/2407.11384)
- **What's New**: 이 연구는 대형 언어 모델(LLMs)을 다중 에이전트 재고 관리 시스템에 적용하는 새로운 접근 방식을 소개합니다. 기존의 휴리스틱 방법과 강화 학습 응용 프로그램의 연구를 보완하며, InvAgent라는 모델을 통해 공급망 네트워크의 탄력성과 효율성을 향상시킵니다. 특히, LLMs의 제로샷 학습 능력을 활용하여 사전 학습 없이 적응적이고 정보에 입각한 의사 결정을 가능하게 합니다.

- **Technical Details**: InvAgent는 다중 에이전트 재고 관리 시스템으로, 여러 공급망 단계와 시뮬레이션 기간에 걸쳐 작동합니다. 각 단계는 재고 보유 구역과 생산 구역으로 구성되며, 제품은 각 단계의 제한된 생산 능력 및 재고에 따라 생산됩니다. 재고 현황과 주문, 수요를 점검한 후, 각 기간의 이익과 비용이 계산됩니다. 제로샷 학습(Zero-shot Learning)과 설명 가능한 체인-오브-띵킹(Chain-of-Thought, CoT) 기능을 통해 기존 모델들보다 더 투명하고 신뢰할 수 있습니다.

- **Performance Highlights**: 다양한 시뮬레이션 시나리오를 통해 InvAgent의 효율성이 입증되었습니다. 모델은 변화하는 수요 시나리오에 동적으로 적응하고 비용을 최소화하며, 재고 부족을 방지합니다. 이는 기존의 휴리스틱 및 강화 학습 모델보다 뛰어난 성능을 보여줍니다.



### Reliable Reasoning Beyond Natural Languag (https://arxiv.org/abs/2407.11373)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 논리적 추론 능력을 향상시키기 위해 신경기호적(neurosymbolic) 접근 방식을 제안합니다. 이 접근 방식은 LLMs가 문제 설명에서 모든 관련 정보를 추출하여 논리 코드로 인코딩하도록 하고, 논리 프로그래밍 언어(Prolog)를 사용해 명시적 연역 추론을 수행하게 합니다. 이를 통해 LLMs의 수학적 추론 성능을 크게 향상시키며, 특히 새로운 데이터셋인 Non-Linear Reasoning (NLR) 데이터셋에서의 성능을 입증합니다.

- **Technical Details**: 이 연구에서는 LLMs가 제약 조건과 변수 간의 관계를 Prolog 코드 명령어로 인코딩하도록 유도했습니다. 생성된 코드는 Prolog를 통해 평가되며, 이를 통해 명확한 해답을 도출합니다. 이러한 방식은 LLM의 전체 아키텍처에 연역적 추론 모듈을 통합하여 성능을 높이는 데 중점을 둡니다. LLMs는 체인 오브 토트(Chain of Thought, CoT) 방식으로 텍스트와 논리 코드에서 이유를 설명하도록 유도되며, 다중 시도 추론 알고리즘(Multiple Try inference algorithm)을 사용해 변수를 조정하면서 정확한 해답을 도출합니다.

- **Performance Highlights**: Prolog와의 통합을 통해 LLMs는 Non-Linear Reasoning (NLR) 데이터셋에서 뛰어난 성과를 보였습니다. 이 데이터셋은 복잡한 비선형 추론을 요구하는 문제로 구성되어 있으며, GPT4를 포함한 최신 언어 모델들은 텍스트만으로는 해결하는 데 실패했습니다. 제안된 접근 방식은 이러한 문제를 해결하는 데 탁월한 성과를 거두었습니다.



### Estimating Agreement by Chance for Sequence Annotation (https://arxiv.org/abs/2407.11371)
Comments:
          ACL 2024

- **What's New**: 이번 논문에서는 자연어 처리(NLP)에서 시퀀스(annotation tasks)의 평가를 위해 사용되는 새로운 무작위 주석 생성 모델(random annotation model)을 소개합니다. 이 모델은 시퀀스 주석 과제의 신뢰성을 평가하기 위해 무작위 주석을 생성하고, 이를 통해 우연 합의(chance agreement)를 예측하는 방법을 제공합니다. CoNLL03 코퍼스와 시뮬레이션을 통한 평가에서, 이 모델의 정확성과 효율성이 입증되었습니다.

- **Technical Details**: 기존에 많이 사용된 Cohen’s Kappa는 분류 작업을 중심으로 설계되었으나, 이 논문은 시퀀스 주석(task)에 특화된 새로운 측정 방법을 제안합니다. 주요 기여는 다음과 같습니다: 1) 시퀀스 annotation task의 특성과 주석자의 annotation 경향을 고려한 무작위 주석 생성 모델을 제안합니다. 2) 시퀀스 annotation task의 작업 난이도를 측정하기 위해 우연 합의(chance agreement)를 적용합니다. 3) 텍스트 내 의존적인 주석 구간의 확률 분포를 유도하여 중복 계산을 피합니다.

- **Performance Highlights**: 이 방법은 모의 실험과 실제 코퍼스 실험에서 검증되어 정확하고 효율적입니다. 특히 Named Entity Recognition(NER) 등의 작업에서 서로 다른 주석자들간의 차이를 정확히 측정할 수 있습니다.



### Ancient Korean Archive Translation: Comparison Analysis on Statistical phrase alignment, LLM in-context learning, and inter-methodological approach (https://arxiv.org/abs/2407.11368)
Comments:
          ACL2024 submitted

- **What's New**: 이번 연구는 희소한 코퍼스 (sparse corpora)를 활용한 고대 텍스트 번역에 대한 세 가지 방법을 비교하였습니다: (1) 전통적인 통계 번역 방법인 구문 정렬 (phrase alignment), (2) 문맥 속 대형 언어 모델 학습 (in-context LLM learning), (3) 제안된 방법으로 소스-타겟 코퍼스의 통합 세트에서 파생된 문장 조각 토큰(sentence piece tokens)을 사용하는 통계 기계 번역입니다.

- **Technical Details**: 전통적인 구문 정렬 방법과 문맥 속 대형 언어 모델 학습 외에도, 새로운 접근 방식으로 소스-타겟 코퍼스의 통합 세트에서 문장 조각 토큰을 생성하여 이를 활용한 통계 기계 번역 방법이 제안되었습니다. 이 방법은 기존의 통계적 접근 방식과 Seq2Seq 모델을 결합하여 새로운 번역 성능을 보여줍니다.

- **Performance Highlights**: 제안된 방법의 BLEU 점수는 36.71로, 이는 SOLAR-10.7B 문맥 학습과 최고의 기존 Seq2Seq 모델을 능가하는 성과를 보였습니다. 이로써 희소한 코퍼스에서도 새로운 방법론이 더 높은 번역 품질을 제공할 수 있음을 증명했습니다.



### Beyond Binary: Multiclass Paraphasia Detection with Generative Pretrained Transformers and End-to-End Models (https://arxiv.org/abs/2407.11345)
- **What's New**: 이 연구는 연속된 구어에서 다중 유형의 발화 착오를 자동으로 감지하는, 첫 번째 연구로서의 의의를 갖는다. 이 연구는 실제 송신된 메시지를 기반으로 다중 클래스 발화착오(multiclass paraphasia) 감지를 수행하며, 자동음성인식(ASR)과 발화착오 분류 작업을 하나의 시퀀스로 학습하는 단일 시퀀스 모델(single sequence model)과 다중 시퀀스 모델(multi-sequence model)을 제안하고 탐구한다.

- **Technical Details**: 이 연구는 Generative Pretrained Transformer (GPT) 모델을 활용하여 음성 전사(transcript)에서 발화착오를 식별한다. 두 가지 접근법을 사용하였으며, 하나는 단일 시퀀스(ASR과 발화착오 분류를 동일 시퀀스로 학습), 다른 하나는 다중 시퀀스(각각의 작업을 별도로 학습하지만, 다중 작업 학습으로 최적화)다. AphasiaBank 코퍼스의 두 가지 데이터 셋(Protocol, Fridriksson)을 사용하였고, CHAT 형식의 수동 전사를 포함한 타임스탬프 데이터를 처리하였다.

- **Performance Highlights**: GPT 모델은 발화착오 감지에서 효과를 보였으나, 단일 시퀀스 모델(single sequence model)이 다중 클래스 발화착오 감지에서 GPT 기반 모델보다 더 우수한 성능을 보였다. 특히, 음소적(paraphasias) 및 신조어적(neologistic) 발화착오 감지에서 단일 시퀀스 모델이 뛰어난 성능을 나타냈다. 반면, 의미적 발화착오(semantic paraphasias) 감지의 한계도 논의되었다.



### Uncertainty is Fragile: Manipulating Uncertainty in Large Language Models (https://arxiv.org/abs/2407.11282)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 불확실성 추정의 취약성 및 잠재적인 공격을 탐구합니다. 특히, 불확실성을 조작하여 모델의 신뢰성을 떨어뜨리는 백도어(backdoor) 공격 방법을 제시하며, 이를 통해 모델의 출력 분포를 변조하면서도 최종 예측값은 변하지 않도록 합니다.

- **Technical Details**: 제안된 백도어 공격은 특정 입력 트리거를 통해 LLM의 불확실성을 조작합니다. KL divergence를 활용하여 모델의 불확실성을 공격자의 의도된 분포로 수렴시키고, 원래의 예측값은 유지합니다. 이를 위해 세 가지 트리거 전략(텍스트, 구문, 스타일)을 사용했습니다.

- **Performance Highlights**: 실험 결과, 백도어 공격은 여러 모델에서 높은 공격 성공률(ASR)을 기록했습니다. 예를 들어, 세 가지 다른 트리거 전략을 사용하여 네 개의 모델에서 100%의 공격 성공률을 달성했습니다. 게다가, 이러한 공격은 다양한 프롬프트와 도메인에서도 일반화되는 것을 확인했습니다.



### Target conversation extraction: Source separation using turn-taking dynamics (https://arxiv.org/abs/2407.11277)
Comments:
          Accepted by Interspeech 2024

- **What's New**: 대화 중인 참여자의 음성을 간섭 스피커와 잡음 속에서 추출하는 것은 어려운 문제입니다. 이 논문에서는 '목표 대화 추출(Target Conversation Extraction, TCE)'이라는 새로운 과제를 소개합니다. TCE의 목표는 대화 참여자 중 한 명의 스피커 임베딩(Speaker Embedding)을 기반으로 목표 대화의 오디오를 추출하는 것입니다. 연구진은 인간 대화의 고유한 패턴인 턴테이킹(Turn-Taking) 역학을 활용하여 이 문제를 해결하고자 합니다.

- **Technical Details**: 목표 대화를 추출하기 위해 연구진은 두 명 이상의 참여자가 포함된 음성 혼합물에서 목표 대화 신호를 추출하는 네트워크를 학습시켰습니다. 이 네트워크는 템포럴 패턴을 학습해서 간섭 스피커 및 배경잡음을 필터링합니다. TF-GridNet을 기반으로 한 소스 분리 네트워크가 사용되었고, 이는 주파수 및 시간 차원 전체에서 이중 경로 LSTM과 풀 어텐션을 사용하여 단기 오디오 혼합물에서 소스를 효과적으로 분리합니다. 특히 STFT(Short-Time Fourier Transform)를 적용하고, 실수 및 허수 부분을 결합하여 2D 컨볼루션에 적용한 후 네트워크에 입력합니다.

- **Performance Highlights**: 연구진이 제안한 모델은 2명의 간섭 스피커가 있는 상황에서 신호 대 잡음비(signal-to-noise ratio, SNR) 8.19dB 향상을, 2-4명의 간섭 스피커가 있는 상황에서는 7.92dB의 향상을 보였습니다. 이와 같은 성능 향상은 영어 및 만다린 대화 데이터셋으로 검증되었으며, 턴테이킹 패턴을 모델링하는 것이 얼마나 중요한지를 보여줍니다.



### Unraveling the Truth: Do LLMs really Understand Charts? A Deep Dive into Consistency and Robustness (https://arxiv.org/abs/2407.11229)
Comments:
          22 pages, 7 Tables, 3 Figures, 25 examples

- **What's New**: 해당 논문에서는 차트 질문 응답(CQA) 분야에서 최신 비주얼 언어 모델(VLMs)의 성능과 일관성을 평가합니다. 이 연구는 다양한 질문 유형과 차트 형식을 포함하는 데이터셋을 사용하여 모델의 복잡한 차트와 질문을 처리하는 능력 및 동일한 데이터의 다양한 시각적 표현에 대한 견고성을 조사합니다. 이를 통해 현재 모델들의 강점과 약점을 밝혀내고, 개선이 필요한 영역을 제안합니다.

- **Technical Details**: 이 논문에서는 ChartQA dataset를 주요 벤치마크로 사용하여 차트 이해 능력을 평가합니다. 데이터셋은 두 가지 질문 카테고리, 'Human' 질문과 'Augmented' 질문으로 나누어지며, 차트와 질문을 복잡도 수준에 따라 구분합니다. 실험에는 zero-shot Chain of Thought 프로빙 접근법을 사용하여, 다양한 최신 상태의 CQA 모델의 성능을 평가합니다. 주요 모델로는 MatCha, UniChart, DePlot 등이 있습니다.

- **Performance Highlights**: 실험 결과, 단순한 차트와 질문에서 복잡한 조합으로 전환될 때 모델 성능에 큰 차이가 나타났습니다. 복잡한 차트-질문 조합에서는 성능이 크게 떨어졌으며, 다양한 변형에 대해 모델의 견고성도 부족함을 보였습니다. 이는 향후 연구에서 향상된 견고성을 필요로 하는 중요한 시사점을 제공합니다.



### Automated essay scoring in Arabic: a dataset and analysis of a BERT-based system (https://arxiv.org/abs/2407.11212)
- **What's New**: 이번 연구는 아랍어 자동 에세이 채점(Automated Essay Scoring, AES) 분야에서 큰 진전을 이루었습니다. 공공 데이터의 부족으로 인한 연구 한계를 극복하고자, 2046개의 학부 수준 에세이를 포함한 AR-AES라고 불리는 새로운 아랍어 AES 벤치마크 데이터셋을 소개했습니다. 이 데이터셋은 성별 정보, 점수, 투명한 루브릭(평가 기준)을 포함하고 있어 채점 과정에 대한 깊이 있는 이해를 제공합니다.

- **Technical Details**: 이번 연구에서는 AraBERT를 사용하여 AES 성능을 탐구했습니다. AraBERT는 특히 Environmental Chemistry와 출처에 의존하는 에세이 질문 유형에서 유망한 결과를 보였습니다. 이 시스템은 기존의 BERT 기반 AES 시스템의 오류 규모를 분석했으며, 전체 오류의 96.15%가 첫 번째 평가자의 예측에서 한 점 이내, 79.49%는 정확히 일치했습니다.

- **Performance Highlights**: 추가적인 인간 평가자들과 비교했을 때, 인간 평가자들은 첫 번째 평가자와의 정확히 일치하는 비율이 30%를 넘지 않았고, 62.9%만이 한 점 이내임을 나타냈습니다. 이 결과는 에세이 채점에서의 주관성을 강조하면서도, 현재의 AES 기술이 대규모 학급에서 일관된 채점을 위해 인간 평가자에게 큰 도움이 될 수 있음을 보여줍니다.



### Actuation without production bias (https://arxiv.org/abs/2407.11202)
Comments:
          Preprint of chapter to be published in _Speech Dynamics: Synchronic Variation and Diachronic Change_

- **What's New**: 이 논문은 음향 변화에 관한 새로운 시각을 제시합니다. 기존 연구들은 주로 발음 편향(phonetic production bias)에 의한 음향 변화를 주로 다루었지만, 이 논문은 발음 편향 외의 다른 요인들이 어떤 영향을 미치는지에 대해 고민합니다. 특히, 여러 선생님으로부터 배우는 상황에서 발음 편향이 인구 집단 전체에 미치는 변화 확산 동학에 관해 논의합니다.

- **Technical Details**: 이 연구는 발음 편향이 유일한 동학을 가지지 않음을 보여주며, 교사의 사회적 가중치(social weight)와 발음 편향 간의 상관관계가 약한 경우 변화가 확산되기 어렵다는 것을 입증합니다. 연구는 음향 변화의 시작, 전파, 확산이라는 세 가지 문제를 구분하고 각 단계에서 어떤 요인들이 중요한지를 분석합니다.

- **Performance Highlights**: 연구 결과는 개별 언어 사용자의 편향이 반드시 인구 집단 전체로 확산되지 않는다는 것을 강조합니다. 특히, 단일 교사와 다중 교사 간 학습 알고리즘의 차이에 따라 인구 집단 수준의 동학이 결정된다는 것을 보여줍니다. 또한, 인구 구조와 개별 편향이 어떻게 상호작용하는지에 대한 정교한 이해를 통해 발음 편향 문제에 대한 부분적인 해결책을 제시합니다.



### FarsInstruct: Empowering Large Language Models for Persian Instruction Understanding (https://arxiv.org/abs/2407.11186)
- **What's New**: FarsInstruct가 새롭게 도입된 것은 대형 언어 모델의 페르시아어 사용 성능을 향상시키기 위한 포괄적인 지침 데이터셋입니다. 이 데이터셋은 다양한 과제 유형과 데이터셋을 포함하여 풍부한 언어적 및 문화적 표현을 담고 있습니다. 또한, 이 연구에서는 Co-CoLA 프레임워크가 소개되어 LoRA-tuned 모델의 다중 과제 적응성을 개선하는 역할을 합니다.

- **Technical Details**: FarsInstruct는 Public Pool of Prompts (P3)에서 번역된 다양한 데이터셋과 혼합된, 기본부터 복잡한 수동 작성 지시문을 포함하고 있습니다. 총 21개의 독특한 공공 데이터셋에서 선택된 200개 이상의 프롬프트 템플릿이 포함되어 있으며, 이 데이터셋은 텍스트 요약, 텍스트 함축, 텍스트 분류, 감정 분석, 단어 의미 모호성 해소, 쿼리 유사화, 질의응답, 독해, 네임드 엔티티 인식 (NER), 번역과 같은 다양한 과제 범주를 포괄합니다. Co-CoLA 프레임워크는 CoLA와 리허설 훈련을 통합하여, 다중 과제 학습 도중 발생하는 재난적 망각 문제를 완화합니다.

- **Performance Highlights**: 연구 결과, Co-CoLA 프레임워크와 결합된 FarsInstruct 데이터셋을 사용한 훈련이 페르시아어 문맥 내에서 대형 언어 모델의 성능을 향상시키는 데 효과적임을 보여줍니다. FarsInstruct는 공개적으로 이용 가능하며 지속적으로 업데이트되어, 다양한 과제와 지침 항목 및 모달리티를 포함하게 될 것입니다.



### YouTube-SL-25: A Large-Scale, Open-Domain Multilingual Sign Language Parallel Corpus (https://arxiv.org/abs/2407.11144)
Comments:
          Access YouTube-SL-25 at this https URL

- **What's New**: 최근 발표된 논문에서는 YouTube-SL-25라는 대규모, 다국어 수어 비디오 데이터셋이 소개되었습니다. 이 데이터셋은 유튜브에서 수집된 3000시간 이상의 비디오 자료를 포함하며 25개 이상의 수어에 대해 잘 맞춰진 캡션이 제공됩니다. 이는 기존의 YouTube-ASL 데이터셋 크기의 3배 이상이며, 다수의 수어에 대해 최초 또는 가장 큰 병렬 데이터셋입니다.

- **Technical Details**: YouTube-SL-25 데이터셋은 자동 분류기와 텍스트 메타데이터를 사용하여 관련 비디오를 식별한 후, 비디오 채널의 전체 시간을 기준으로 우선 순위를 매기고, 유효한 캡션이 포함된 비디오를 감사하는 두 단계 과정을 통해 구축되었습니다. T5 기반의 통합 다국어 다목적 모델을 사용하여 사인-텍스트 태스크의 베이스라인을 제공했고, 4개의 수어에 대한 벤치마크 점수를 보고했습니다.

- **Performance Highlights**: 벤치마크 결과 다국어 전이(multilingual transfer)가 YouTube-SL-25 내의 고자원 및 저자원 수어 모두에게 이득이 되는 것을 보여주었습니다. 또한 4개의 수어에 대한 베이스라인을 구축하면서 데이터셋의 성능을 확인할 수 있었습니다.



### Direct-Inverse Prompting: Analyzing LLMs' Discriminative Capacity in Self-Improving Generation (https://arxiv.org/abs/2407.11017)
Comments:
          4 pages, 3 tables

- **What's New**: 최근 주요 연구는 LLM의 생성능력을 향상시키는 데 초점을 맞추고 있지만, 여전히 LLM은 입력에 대해 다중 응답을 생성하면서 불확실성을 겪는 문제가 있습니다. 이를 해결하기 위해, LLM의 판별 능력을 활용하여 생성 불확실성을 줄이는 방법을 제안합니다. 도출된 주요 기술은 세 가지 판별 프로프트(direct, inverse, hybrid)를 사용하여 최고의 응답을 식별하는 것입니다. 이 연구는 이러한 기법이 LLM의 생성 능력 자체 개선에 얼마나 효과적인지 분석합니다.

- **Technical Details**: 본 연구에서는 두 개의 폐쇄형 LLM(GPT-4)과 두 개의 오픈소스 LLM(Llama-3-8B-Instruct, MetaMath-7B-V1.0)을 사용하여, 수학 관련 데이터셋(MATH, MathQA)에서 LLM의 판별 능력을 평가했습니다. 직접 프로프트(direct), 역 프로프트(inverse), 그리고 인버스와 직접 프로프트를 결합한 조합 프로프트(hybrid)를 통해 LLM의 판별 능력을 분석했습니다.

- **Performance Highlights**: 폐쇄형 LLM의 경우, 직접 프로프트나 역 프로프트를 사용하는 것이 생성 불확실성을 줄이는 데 매우 효과적이었습니다. 반면, 오픈소스 LLM은 명령 조정이 되어 있지 않을 경우, 판별 능력을 사용하는 것이 비추천되었습니다. 만약 명령 조정이 되어 있다면, 인버스 프로프트의 부정 이해 문제로 인해 직접 프로프트만을 사용하는 것이 좋습니다. 이 연구의 분석 결과는 LLM의 판별 능력이 생성 능력을 향상시킬 수 있음을 보여줍니다.



### LongLaMP: A Benchmark for Personalized Long-form Text Generation (https://arxiv.org/abs/2407.11016)
Comments:
          9 pages, 4 figures, 20 tables(including appendix) submitted to EMNLP

- **What's New**: 이 논문에서는 거대 언어 모델(Large Language Models, LLMs)을 사용한 개인 맞춤형 장문 생성(personalized long-text generation)을 다룹니다. 기존의 연구는 짧은 텍스트 생성(personalized short-text generation)에 집중되어 있었지만, 이 논문은 이메일, 리뷰 작성 등 실제 응용 프로그램에서 필요로 하는 장문의 개인화된 텍스트 생성 문제를 해결합니다. 이를 위해, LongLaMP (Long-text Language Model Personalization) 벤치마크를 개발했습니다. LongLaMP는 다양한 개인 맞춤형 장문 생성 작업에 대한 평가 프레임워크를 제공합니다.

- **Technical Details**: LongLaMP 벤치마크는 4가지 다른 개인화된 장문 생성 작업으로 구성됩니다: (1) 개인화된 이메일 생성, (2) 개인화된 초록 생성, (3) 개인화된 리뷰 생성, (4) 개인화된 주제 글 작성. 각 작업에서 사용자 프로필(user profile)과 입력 프롬프트(input prompt)와 목표 출력(target output)을 사용하여 사용자의 기록과 스타일을 반영한 장문을 생성합니다. 두 가지 설정이 있습니다: (a) 새로운 사용자에 대한 개인화된 텍스트 생성 평가, (b) 기존 사용자에 대한 최신 내용 생성 평가.

- **Performance Highlights**: LongLaMP 벤치마크를 사용한 평가 결과, 제안된 프레임워크가 개인 맞춤형 장문 텍스트 생성에서 기존 비개인화 베이스라인 대비 5.7%에서 128%까지 성능이 개선됨을 확인했습니다. 이 연구는 개인화된 장문 텍스트 생성의 중요성을 강조하며, 다양한 텍스트 생성 작업에서 이러한 개인화 접근법이 효과적임을 보여줍니다.



### Does ChatGPT Have a Mind? (https://arxiv.org/abs/2407.11015)
- **What's New**: 이 논문은 ChatGPT와 같은 대규모 언어 모델(LLMs)이 마음을 가졌는지, 특히 신념, 욕구, 의도를 포함하는 진정한 민속 심리학을 갖추고 있는지를 조사합니다. 저자들은 내부 표현과 행동 성향이라는 두 가지 주요 측면을 탐구합니다. LLMs가 다양한 철학적 이론의 조건을 충족하는 것을 확인하고 기계 학습 해석 가능성 연구에 근거하여 이를 주장합니다.

- **Technical Details**: 논문에서는 LLMs의 내부 표현이 정보를 전달하고, 시스템의 행동을 유발하며, 민속적 추론 패턴을 만족시키고, 표현 구조와 일치하는지의 조건을 탐구합니다. 또한, 행동 성향의 문제를 다루기 위해 신념과 욕구에 대한 두 가지 주요 철학적 전통인 해석주의(dinterpretationism)와 표현주의(representationalism)를 분석합니다. 중요한 질문은 LLMs의 언어 출력이 목표 달성을 촉진하는지 여부입니다.

- **Performance Highlights**: LLMs가 게임 이론적 환경에서 복잡한 행동 계획을 보여주는 것 같다는 증거가 있습니다. 그러나 데이터는 결정적이지 않습니다. 저자들은 LLMs의 심리적 상태에 관한 회의적인 도전에 대해 감각적 근거 문제, '확률적 앵무새'(stochastic parrots) 논쟁, 암기 우려를 포함하여 철학적 반박을 제시합니다. 최종 결론은 LLMs가 내부 표현을 가지고 있으며, 행동 성향에 관한 미결 질문이 있음을 제시합니다.



### Geode: A Zero-shot Geospatial Question-Answering Agent with Explicit Reasoning and Precise Spatio-Temporal Retrieva (https://arxiv.org/abs/2407.11014)
- **What's New**: Geode는 기존의 대형언어모델(LLM)이 지리적 질의를 효율적으로 처리하지 못하는 한계를 극복하고자 제안된 혁신적인 시스템입니다. Geode는 공간-시간 데이터 검색을 통해 정밀한 지리적 질문 응답을 실현하며, 이는 현재 최첨단 사전 학습된 모델들보다 훨씬 뛰어난 성능을 보여줍니다.

- **Technical Details**: Geode는 다중 모달(multi-modal) 데이터와 상호 작용할 수 있는 능력을 중심으로 설계되었습니다. 이 시스템은 텍스트, 이미지, 비디오, 오디오 등 다양한 형태의 데이터를 학습하고 해석할 수 있는 능력을 갖추고 있으며, 특히 지리 정보 시스템(GIS) 데이터를 포함한 공간 데이터를 효과적으로 처리합니다. 여러 전문가 시스템을 결합하여 복잡한 사용자 문의를 처리하도록 설계되었으며, 제로샷(Zero-shot) 질문 응답을 수행합니다.

- **Performance Highlights**: Geode는 현재 사용 가능한 다른 최첨단 사전 학습 모델에 비해 상당한 향상을 보여줍니다. 특히, 지리적 데이터의 공간-시간 검색을 통해 정확한 답변을 제공할 수 있으며, 이는 복잡한 지리적 질의 해결에 큰 도움을 줍니다.



### Exploring Gender-Specific Speech Patterns in Automatic Suicide Risk Assessmen (https://arxiv.org/abs/2407.11012)
Comments:
          accepted at INTERSPEECH 2024

- **What's New**: 이번 연구는 응급 의료 상황에서 자살 위험 평가를 자동으로 수행할 수 있는 음성 기반 접근 방식을 도입하였습니다. 이를 위해 20명의 환자가 중립적인 텍스트를 읽는 음성 녹음을 포함한 새로운 데이터셋을 사용하였습니다. 연구는 성별 기반 모델링과 문구 수준 정규화(pharse-level normalisation)의 영향을 탐구하는 것이 특징입니다.

- **Technical Details**: 본 연구는 음성 샘플의 볼륨을 정규화하고 강제 정렬을 통해 문구로 세분화하는 전처리 단계로 시작합니다. 이후 각 구별로 해석 가능한 오디오 기능(interpretable audio functionals)과 딥 피처(deep features)를 추출합니다. 그런 다음 전역(global) 및 문구 수준 정규화를 적용합니다. 교육 및 평가에서는 한 화자 제외 교차 검증(Leave-One-Speaker-Out, LOSO)을 사용하여 분류기를 훈련시킵니다. 특히 감정 미세 조정된 wav2vec2.0 모델에서 추출된 특징들은 자살 위험이 높은 그룹과 낮은 그룹을 구별하는 데 사용됩니다.

- **Performance Highlights**: 성별 독점 모델링(gender-exclusive modelling)을 적용한 결과, 감정 미세 조정된 wav2vec2.0 모델에서 추출된 특성을 사용하여 자살 위험이 높은 그룹과 낮은 그룹을 구별할 때의 균형 정확도(balanced accuracy)가 81%에 도달하였습니다. 특히, 남성과 여성 참가자 간의 음성 특징과 자살 위험 간의 관계에서 명백한 차이가 발견되었습니다: 남성의 경우 자살 위험은 초조함과 함께 증가한 반면, 여성의 음성 특성은 반대의 경향을 보였습니다.



### Navigating the Minefield of MT Beam Search in Cascaded Streaming Speech Translation (https://arxiv.org/abs/2407.11010)
- **What's New**: 새로운 논문에서는 기계번역(MT)에서 널리 사용되는 beam-search 알고리즘을 실시간 음성 번역 시스템에 적용하는 방법을 제안합니다. 이 접근법은 기존의 greedy decoding보다 복잡하지만, 주요 문제를 해결함으로써 BLEU 점수를 1포인트 향상시키고 CPU 시간을 최대 40%까지 감소시켰습니다.

- **Technical Details**: 논문에서는 실시간 처리, 중간 및 최종 번역의 지연 최소화, 불완전한 단어 처리, 서로 다른 길이와 상태를 가진 beam search 가설 처리, 문장 경계 처리와 같은 네 가지 주요 문제를 해결합니다. 이 시스템에서는 실시간으로 자동 음성 인식(ASR)과 텍스트 기계 번역을 결합하여 최소 지연 시간으로 결과를 제공할 수 있게 합니다. 또한, 이전 논문들이 greedy decoding을 구현한 것과 달리, 본 논문에서는 실시간 파이프라인에서 사용하는 beam-search 알고리즘을 실현했습니다.

- **Performance Highlights**: 제안된 방법은 greedy search에 비해 BLEU 점수를 1포인트 증가시키고, 반복 번역 기반의 시스템과 비교했을 때 CPU 시간을 최대 40%까지 감소시켰습니다. 또한, 반복 번역 시스템에 비해 문자 깜박임률(character flicker rate)을 20% 이상 줄였습니다.



### CharED: Character-wise Ensemble Decoding for Large Language Models (https://arxiv.org/abs/2407.11009)
Comments:
          9 pages, 4 figures

- **What's New**: 이번 연구에서는 LLMs(Large Language Models) 중 서로 다른 단어 사전과 토크나이제이션(tokenization) 방식들을 사용하는 복수의 언어 모델을 결합하는 새로운 앙상블 알고리즘 'CharED'를 제안했습니다. 이 방법은 각 LLM의 출력을 문자(character) 단위로 '평균화(averaging)'하여 출력물을 생성합니다. 따라서 더 복잡한 사전 공유나 토크나이제이션 과정을 거치지 않고도 뛰어난 성능을 구현할 수 있습니다.

- **Technical Details**: CharED 알고리즘은 문자 단위 앙상블 디코딩(character-wise ensemble decoding)을 사용합니다. 개별 모델의 각 문자에 대한 마진 분포를 찾아 이를 가중 평균(weighted average)으로 처리해 문자를 하나씩 출력합니다. 그렇기 때문에 각 모델의 사전이나 토크나이제이션 방식을 일일이 맞출 필요 없이, 다양한 LLM들을 효과적으로 결합할 수 있습니다. 이를 위해 CharED는 각 모델의 다음 토큰 확률을 조회 후, 문자 단위로 분해하여 마진 다음 문자 확률을 계산합니다.

- **Performance Highlights**: 이번 연구에서는 CharED가 코딩, 수학, 독성(毒性) 벤치마크에서 뛰어난 성능을 나타낸다는 점을 확인했습니다. HumanEval, GSM8K, ToxiGen과 같은 다양한 도메인에서 LLM들의 개별적 성능을 초과하는 성적을 보여주었습니다.



### Figuring out Figures: Using Textual References to Caption Scientific Figures (https://arxiv.org/abs/2407.11008)
- **What's New**: 새로운 연구는 과학 논문에서 복잡한 아이디어를 밀도 있게 전달하는 그림의 캡션 생성 문제를 다룹니다. 이전의 연구들에서 많이 사용되었지만, 성능이 떨어지는 단층 LSTM 모델을 넘어 SCI-CAP 데이터셋을 사용하며, CLIP과 GPT-2를 활용한 인코더-디코더 모델을 사용하여 이미지 기반 캡션을 생성했습니다. 또한, 논문의 제목, 초록, 본문 참고문헌 등의 텍스트 메타데이터를 포함하는 새로운 데이터셋인 MetaSciCap를 만들고 이 데이터를 SciBERT를 사용해 인코딩함으로써 캡션 성능을 향상시켰습니다.

- **Technical Details**: 이 연구는 CLIP+GPT-2 인코더-디코더 모델과 교차 주의를 결합하여 이미지를 조건으로 캡션을 생성합니다. CLIP은 다양한 시각적 입력으로 훈련된 Vision Transformer 아키텍처를 사용하며, 이미지를 작은 패치로 분할하여 트랜스포머 인코더로 전달합니다. SciBERT는 원본 논문 텍스트 메타데이터를 인코딩하는 용도로 사용됩니다. 최종적으로 SciBERT의 텍스트 인코딩과 CLIP의 이미지 인코딩을 결합하여 GPT-2 디코더에 전달합니다. 특정 실험에서는 모델이 텍스트 메타데이터에만 의존하지 않도록 SciBERT 인코딩에 드롭아웃을 적용하기도 했습니다.

- **Performance Highlights**: 다양한 모델 실험 결과에 따르면, SciBERT 인코더에서 받은 모든 텍스트 메타데이터와 함께 CLIP+GPT-2 모델이 최고 성능을 보였습니다. 하지만 텍스트 메타데이터만을 사용하는 SciBERT+GPT-2 모델이 가장 최적의 성능을 달성했습니다. 이는 구체적으로 모델이 피겨만으로는 불충분하며, 텍스트 정보의 결합이 성능 향상에 큰 도움이 된다는 것을 시사합니다.



### Panacea: A foundation model for clinical trial search, summarization, design, and recruitmen (https://arxiv.org/abs/2407.11007)
- **What's New**: 새로운 임상 시험 기초 모델인 Panacea가 제안되었습니다. 이 모델은 다양한 임상 시험 작업을 처리할 수 있도록 설계되어 있으며, 여기에는 시험 설계, 환자-시험 매칭, 시험 검색 및 요약이 포함됩니다. 이를 위해, 793,279개의 시험 문서와 1,113,207개의 시험 관련 과학 논문을 수집한 대규모 데이터셋 'TrialAlign'을 구축하였으며, 미세 조정을 위한 200,866개의 지시 데이터 'TrialInstruct'를 마련했습니다.

- **Technical Details**: Panacea는 두 가지 주요 단계로 훈련되었습니다. 첫 번째 단계는 시험 문서와 관련 논문을 사용하여 임상 시험에서 공통적으로 사용되는 어휘에 맞게 모델을 조정하는 'Alignment' 단계입니다. 두 번째 단계는 사용자의 작업 정의와 출력 요구 사항을 이해할 수 있도록 모델을 조정하는 'Instruction-tuning' 단계입니다. 'TrialAlign' 데이터셋은 다양한 리소스에서 수집된 793,279개의 비식별화 시험 문서와 1,113,207개의 과학 논문으로 구성되어 있으며, 이는 다양한 질병과 치료법을 포괄합니다. 'TrialInstruct'는 최소 2,000개의 데이터 포인트가 포함된 여덟 가지 작업 지시 데이터로 구성됩니다.

- **Performance Highlights**: Panacea는 새로운 벤치마크 'TrialPanorama'에서 여덟 가지 임상 시험 작업 중 일곱 가지에서 최고 성능을 발휘했습니다. 특히, 환자-시험 매칭에서 14.42%의 개선, 시험 검색에서 41.78%에서 52.02%의 개선을 이루었습니다. Panacea는 임상 시험 요약에 있어서도 다섯 개의 측면에서 consistently 높은 성적을 보였습니다. 이 연구는 Panacea가 예상되는 결과에 도달할 수 있도록 안내하는 대화 가족 협업에서 큰 가능성을 보여주었으며, 다양한 임상 시험 작업에서 유용하게 사용될 수 있음을 입증했습니다.



### How Good Is It? Evaluating the Efficacy of Common versus Domain-Specific Prompts on Foundational Large Language Models (https://arxiv.org/abs/2407.11006)
Comments:
          10 pages, 5 figures, 2 tables, and algorithms

- **What's New**: 최근 대형 언어 모델(LLMs)은 다양한 도메인으로 확장되었습니다. 하지만 이러한 모델들이 일반적인 질문과 도메인 특정 질문에 어떻게 반응하는지 평가할 필요가 있음을 강조합니다. 이 연구는 Gemma-2B와 Gemma-7B 모델을 사이버 보안, 의학, 금융 분야에서의 성능을 일반 지식 기반 질문과 비교하여 평가했습니다.

- **Technical Details**: 이 연구는 포괄적인 방법론을 사용하여 초기 모델을 평가했으며, 여기에는 문제 공식화, 데이터 분석, 새로운 이상 탐지 기법 개발이 포함됩니다. 평가 지표는 추론 시간, 응답 길이, 처리량, 품질 및 자원 사용을 포함하며, 이 요인들 간의 상관관계를 조사했습니다. 또한, 모델 크기와 사용된 프롬프트 유형이 응답 길이와 품질에 미치는 영향을 분석했습니다.

- **Performance Highlights**: 주요 발견 사항은 다음과 같습니다: 7B 모델이 2B 모델보다 더 많은 GPU 메모리를 소비하며, 일반 프롬프트는 응답 다양성이 크고 추론 시간이 길다는 점, 반면 도메인 특정 프롬프트는 일관성 있는 응답을 생성한다는 것입니다. 2B 모델은 7B 모델보다 높은 처리량을 보였으며, 추론 시간과 응답 길이 사이의 강한 상관관계가 관찰되었습니다. 7B 모델은 ChatGPT 응답과의 의미적 텍스트 유사도(STS) 측면에서 우수한 성능을 보였고, ROUGE-L 점수가 모든 도메인에서 더 높았습니다.



### RAGBench: Explainable Benchmark for Retrieval-Augmented Generation Systems (https://arxiv.org/abs/2407.11005)
- **What's New**: 이번 연구에서는 RAGBench를 도입했습니다. 이는 10만 개의 예제를 포함한 대규모 RAG(가져오기-증가 생성) 벤치마크 데이터셋으로, 다섯 가지 고유한 산업 도메인과 다양한 RAG 작업 유형을 다룹니다. RAGBench는 산업 코퍼스에서 비롯된 예제들을 포함하여, 실제 산업 응용 프로그램에 매우 적합합니다. 또한, RAG 평가를 위한 TRACe 평가 프레임워크도 함께 도입하였습니다.

- **Technical Details**: RAG 시스템은 두 가지 구성요소로 이루어져 있습니다: 입력 쿼리에 관한 도메인 특화 코퍼스를 조회하는 문서 검색 시스템과 주어진 쿼리 및 컨텍스트를 기반으로 응답을 생성하는 LLM(대형 언어 모델). RAGBench는 기존의 컨텍스트 관련성 및 답변 신뢰성 평가 메트릭에 더하여, 컨텍스트 활용도와 답변 완전성이라는 두 가지 새로운 메트릭을 도입하였습니다. 이를 통해 RAG 시스템의 전반적인 성능을 보다 잘 설명할 수 있습니다.

- **Performance Highlights**: 최신 LLM 기반의 RAG 평가 방법들은 세련된 RoBERTa 모델에 비해 성능이 부족하다는 결과가 나타났습니다. 특히 각종 도메인 및 작업 유형에서 미세 조정된 DeBERTa-large 모델이 더 나은 성능을 보였습니다. 이를 통해 RAG 평가 접근 방법 및 벤치마크 개선의 필요성이 강조되었습니다.



### The ALCHEmist: Automated Labeling 500x CHEaper Than LLM Data Annotators (https://arxiv.org/abs/2407.11004)
- **What's New**: 최근에 발표된 논문은 대형 미리 학습된 모델(pretrained models)을 활용하여 데이터 라벨링 작업을 최적화하는 새로운 접근 방식을 제안합니다. 이 시스템은 'Alchemist'라고 명명되었으며, 라벨링을 위한 프로그램을 생성하는 방식으로 모델을 이용합니다. 이 접근 방식은 고비용의 API 호출 문제를 해결하고, 생성된 데이터셋을 로컬에서 저장하고 확장할 수 있게 합니다.

- **Technical Details**: Alchemist 시스템은 모델에게 직접 데이터를 라벨링하게 하는 대신, 라벨링을 수행할 프로그램을 생성하도록 요청합니다. 이 프로그램은 로컬에서 실행될 수 있으며, API 호출 수를 크게 줄여줍니다. 예를 들어, 7,569개의 데이터 포인트를 라벨링하는 경우, 기존 비용이 1,200달러에서 0.70달러로 대폭 감소합니다. 또한, 코드 형태로 저장된 프로그램은 검토, 수정, 확장이 용이하며, 약한 감독(weak supervision) 프레임워크를 통해 여러 노이즈 소스에서 데이터셋을 구축할 수 있습니다.

- **Performance Highlights**: Alchemist 시스템은 다양한 태스크에서 기존 대형 언어 모델 기반 라벨링보다 성능이 동등하거나 더 우수하다는 결과를 보였습니다. 평균적으로 12.9%의 성능 향상을 기록하며, 전체 라벨링 비용을 약 500배 줄였습니다.



### Using Large Language Models in Public Transit Systems, San Antonio as a case study (https://arxiv.org/abs/2407.11003)
- **What's New**: 이 연구는 대형 언어 모델(LLMs, Large Language Models)의 공공 교통 시스템 통합이 도심 교통 관리 및 승객 경험에 미치는 영향을 조사합니다. 특히, 샌안토니오의 공공 교통 시스템을 사례로 사용하여 LLMs가 경로 계획, 대기 시간 단축, 개인 맞춤형 여행 지원을 어떻게 개선할 수 있는지에 대해 자세히 탐구합니다.

- **Technical Details**: 이 연구는 OpenAI의 GPT 시리즈와 같은 LLMs 모델을 활용하여 GTFS(General Transit Feed Specification) 데이터 및 기타 공공 교통 정보를 분석하고, 실시간 통신을 통해 공공 교통 시스템의 최적화를 목표로 합니다. LLMs는 자연어 처리(NLP), 데이터 분석, 실시간 의사소통 분야에서 탁월한 능력을 보여줍니다.

- **Performance Highlights**: LLMs의 잠재력 평가를 위해, 샌안토니오 공공 교통 시스템에 대한 275개의 질문 세트를 설계했습니다. 이는 두 가지 주요 영역에서 성능을 평가합니다: 1) 사전 훈련된 ChatGPT 모델이 샌안토니오의 공공 교통 시스템에 관한 질문을 얼마나 잘 이해하고 대답할 수 있는지 평가하는 '이해' 태스크와 2) 주어진 데이터 셋에서 관련 정보를 검색하는 능력을 평가하는 '정보 검색' 태스크입니다. 연구 결과는 공공 교통 시스템의 효율성을 높이고 사용자 만족도를 개선하는 방안을 제시합니다.



### MoESD: Mixture of Experts Stable Diffusion to Mitigate Gender Bias (https://arxiv.org/abs/2407.11002)
- **What's New**: 이번 연구에서는 텍스트-이미지 생성 모델이 본질적으로 가지고 있는 성 편향을 완화하기 위한 새로운 접근법을 제시합니다. 구체적으로, Mixture-of-Experts Stable Diffusion (MoESD)과 Bias Adapters (BiAs)를 도입하여 성 편향을 감소시키는 방법을 소개합니다. 이를 통해 생성 이미지의 품질을 유지하면서 성 편향을 효과적으로 줄이는 방법을 제시합니다.

- **Technical Details**: 연구에서는 성 편향이 텍스트 인코더 단계에서 이미 존재함을 확인했습니다. 이를 해결하기 위해 Mixture of Experts와 Bias Adapters를 결합하여 편향 식별 게이트를 만들고, 특별 토큰을 통해 편향 데이터를 더 잘 이해하도록 돕는 방법을 사용했습니다. Stable Diffusion 모델의 텍스트 임베딩에서 편향을 발견하고, 이를 완화하기 위해 소수의 데이터(1.5K)와 적은 수의 파라미터(5.6%)로 파인튜닝하는 접근법을 채택했습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 성 편향을 성공적으로 완화하면서도 이미지 품질을 유지할 수 있음을 보여주었습니다. 특히, 기존의 대규모 데이터셋과 모델의 전체 파인튜닝을 요구하지 않고도 효율적으로 성 편향을 줄일 수 있음을 입증했습니다.



### Generative AI Systems: A Systems-based Perspective on Generative AI (https://arxiv.org/abs/2407.11001)
- **What's New**: 최근 대형 언어 모델(LLMs)의 발전과 함께 Generative AI(생성 인공지능, GenAI)의 가능성이 크게 주목받고 있습니다. 특히 Vision-Language Models(GPT-4V)와 같은 멀티모달 시스템이 주목받고 있는데, 이 연구는 이러한 GenAI 시스템을 'GenAISys'라고 칭하며, 이들이 다양한 데이터를 처리하고 생성하며, 의사결정까지 가능하게 하는 방법을 탐구합니다.

- **Technical Details**: GenAISys는 LLMs를 핵심으로 하여 여러 데이터 소스(text, image, audio, video)를 다루는 모달리티 인코더(modality encoders)를 입출력 인터페이스로 사용합니다. 또한 GenAISys는 데이터베이스와 외부 전문 도구를 활용하여 정보 검색과 저장을 수행합니다. 핵심은 시스템의 구성을 원자의 시스템과 복합 시스템으로 나누고, 이러한 시스템 간의 상호작용 규칙을 통해 복잡한 시스템을 구성하는 것입니다.

- **Performance Highlights**: 이 논문은 GenAISys의 설계(compositionality), 신뢰성(reliability), 검증 가능성(verifiability)을 탐구하며, 시스템 기반 분석이 어떻게 GenAISys의 특성과 잠재적 문제점을 확인하는 데 도움이 될 수 있는지 논의합니다. 예를 들어, Seq2Seq 모델을 외부 도구와 함께 사용하는 사례를 통해 GenASys의 실질적인 응용 가능성을 강조합니다.



### Autonomous Prompt Engineering in Large Language Models (https://arxiv.org/abs/2407.11000)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 성능을 최적화하기 위해 자동 프롬프트 엔지니어링 도구(APET)를 소개합니다. APET는 GPT-4가 스스로 프롬프트 엔지니어링 기법을 적용할 수 있게 합니다. 이를 통해 다양한 맞춤형 작업에서 성능을 크게 향상시킬 수 있습니다.

- **Technical Details**: APET는 Expert Prompting, Chain of Thought, Tree of Thoughts와 같은 복잡한 전략을 활용해 프롬프트를 동적으로 최적화합니다. 이를 통해 Word Sorting 작업에서 4.4%, Geometric Shapes 작업에서 6.8%의 성능 향상을 이루었습니다. Checkmate in One과 같은 복잡한 작업에서는 성능이 감소(-14.8%)했으나, 이는 복잡한 프롬프트 최적화 프로세스를 외부 데이터 없이 자동화할 수 있는 가능성을 보여줍니다.

- **Performance Highlights**: APET가 적용된 GPT-4는 Word Sorting 작업에서 4.4%, Geometric Shapes 작업에서 6.8%의 성능 향상을 달성했습니다. 비록 Checkmate in One 작업에서는 성능이 감소(-14.8%)했지만, APET는 복잡한 작업 성능을 향상시키고 실세계 시나리오에서 이러한 기법의 실질적 응용을 확대할 수 있는 잠재력을 보여줍니다.



### TALEC: Teach Your LLM to Evaluate in Specific Domain with In-house Criteria by Criteria Division and Zero-shot Plus Few-sho (https://arxiv.org/abs/2407.10999)
- **What's New**: 이번 논문에서는 LLM(Large Language Models)의 평가를 위해 새로운 모델 기반 평가 방법인 TALEC를 제안합니다. TALEC는 사용자가 평가 기준을 유연하게 설정할 수 있도록 하고, in-context learning(문맥 학습)을 활용해 이러한 사내 평가 기준을 학습하도록 합니다. 더불어, zero-shot과 few-shot 접근 방식을 결합하여 평가 모델이 더 많은 정보를 집중할 수 있도록 합니다.

- **Technical Details**: TALEC는 구체적인 응용 시나리오에서 평가를 집중적으로 수행합니다. 본 논문에서는 자동차 분야의 실제 응용 사례를 통해 실험과 벤치마크를 진행했습니다. 평가 기준은 유연하게 설정할 수 있으며, 데이터셋을 'train', 'eval', 'test' 데이터셋으로 나누어 모델을 학습และ 검증합니다. 또한, zero-shot과 few-shot 기법을 결합하여 평가 모델이 더 많은 정보를 이해하도록 돕습니다.

- **Performance Highlights**: TALEC는 인간의 선호도를 정확하게 반영하는 강력한 능력을 보여주며, 인간 평가와의 상관관계가 80% 이상에 달합니다. 이는 일부 작업에서 사람 간의 상관관계조차 능가하기도 합니다.



### Discrete Diffusion Language Model for Long Text Summarization (https://arxiv.org/abs/2407.10998)
- **What's New**: 이번 연구에서는 기존의 불연속(diffusion) 모델이 길이텍스트 생성 등에 제대로 사용되지 못하는 문제를 해결하고자 합니다. 본 논문에서는 추상적 요약 기능을 보완하기 위해 새로운 의미 인식(noising) 프로세스를 도입하여 Transformer 백본을 효율적으로 사용할 수 있게 했습니다. 더불어, CrossMamba라는 모델을 소개하여 인코더-디코더 패러다임에 맞추어 개선된 성능을 보였습니다.

- **Technical Details**: 새로운 의미 인식(noising) 프로세스는 랜덤 노이즈를 중요한 정보 우선 방식을 도입하여 텍스트 모델링을 혁신했습니다. 또한, CrossMamba는 Mamba 모델을 인코더-디코더 아키텍처에 적용하여 긴 시퀀스에서도 뛰어난 성능을 발휘합니다. 이로 인해 복잡한 수학적 처리 과정을 단순화하고 효율성을 크게 향상시켰습니다.

- **Performance Highlights**: 실험 결과, 제안한 모델은 Gigaword, CNN/DailyMail, Arxiv 데이터셋에서 최고의 성능을 기록했습니다. 기존의 불연속(diffusion) 모델들보다 높은 ROUGE 점수를 획득했으며, 오토레그레시브(autoregressive) 모델들과 비교해서도 훨씬 더 빠른 추론 시간을 자랑합니다.



### Visualization Literacy of Multimodal Large Language Models: A Comparative Study (https://arxiv.org/abs/2407.10996)
- **What's New**: 최근에는 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 도입으로 언어 모델의 강력한 성능에 시각적 맥락을 이해하고 추론하는 능력이 추가되었습니다. 이로 인해 MLLMs는 텍스트만을 사용하는 모델보다 훨씬 다양한 사용례를 갖게 되었습니다. 특히 MLLMs는 시각화 결과를 이해하고 해석하는 능력을 갖추고 있어 다양한 시각화 태스크(Visualization Tasks)를 효과적으로 수행할 수 있습니다. 이번 연구에서는 주로 시각화 문해력(Visualization Literacy) 개념을 활용해 MLLMs의 시각화 태스크 수행 능력을 평가하며, 이는 기존 인간 기준과도 비교되었습니다.

- **Technical Details**: 본 연구에서는 시각화 문해력 평가를 위해 두 가지 주요 데이터셋인 VLAT와 mini-VLAT를 사용하였습니다. 여기에는 최신 MLLMs(GPT4-o, Claude 3 Opus, Gemini 1.5 Pro)과 인간 기준을 비교하는 일반적인 설정을 개발했습니다. 시각화 문해력은 다양한 시각적 인코딩(Visual Encoding)과 시각화 태스크를 포함하는 세부적인 평가체계로 구성되어 있습니다. 이를 통해 MLLMs가 특정 시각화 태스크를 얼마나 잘 수행하는지와 그 한계를 확인할 수 있었습니다.

- **Performance Highlights**: 본 연구에서는 MLLMs가 상관관계(Correlation) 및 클러스터(Cluster) 식별, 계층구조(Hierarchical Structures) 분석 등에서 인간보다 더 뛰어난 성능을 보이는 것을 확인했습니다. 특히 트리맵(Treemap) 해석에서 두드러진 성과를 나타냈습니다. 그러나 MLLMs와 인간이 보여주는 실패 패턴에는 차이가 존재하여, 이는 향후 개선 방향에 중요한 정보를 제공합니다.



### LionGuard: Building a Contextualized Moderation Classifier to Tackle Localized Unsafe Conten (https://arxiv.org/abs/2407.10995)
Comments:
          Preprint

- **What's New**: LionGuard는 싱가포르의 언어적 맥락에 맞춘 새로운 콘텐츠 조정(classification) 시스템입니다. Singlish 데이터에 대해 평가한 결과, 기존의 널리 사용되는 조정 API보다 14% 높은 이진(binary) 분류 성능과 최대 51% 높은 멀티라벨(multi-label) 분류 성능을 기록했습니다. 이는 로컬라이제이션(localization)의 이점을 강조하며, 저자원 언어를 위한 실용적이고 확장 가능한 접근 방식을 제시합니다.

- **Technical Details**: LionGuard의 주요 기여는 로컬 컨텍스트에 맞춘 안전성 위험 분류 체계 설정, 대규모 Singlish 텍스트 데이터셋 생성, 그리고 자동 라벨링을 통한 여러 분류 모델의 성능 향상입니다. 특히, OpenAI의 GPT-3.5-Turbo, Anthropic의 Claude 2.0, 그리고 Google's PaLM 2와 같은 안전성이 조정된 LLM을 사용하여 데이터셋을 자동으로 라벨링하였습니다. 모델 훈련 후, OpenAI의 Moderation API, Jigsaw의 Perspective API, Meta의 LlamaGuard와 같은 기존의 상용 시스템보다 우수한 성능을 보였습니다.

- **Performance Highlights**: LionGuard는 Hugging Face Hub에서 사용할 수 있으며, 평가 결과 싱가포르의 로컬 언어인 Singlish에 대해 14% 높은 이진 분류 성능과 최대 51% 높은 멀티라벨 분류 성능을 보였습니다. 이러한 성능 향상은 특히 저자원 언어에 대한 대응력에서 큰 잠재력을 보여줍니다.



### Panza: A Personalized Text Writing Assistant via Data Playback and Local Fine-Tuning (https://arxiv.org/abs/2407.10994)
Comments:
          Panza is available at this https URL

- **What's New**: 저자들은 Panza라는 새로운 개인 비서 시스템을 제안했습니다. 이 시스템은 이메일 생성에 특화되어 있으며, 사용자의 데이터와 작문 스타일을 학습하여 개인화된 결과를 제공합니다. Panza는 로컬 하드웨어에서 실행될 수 있어 개인 데이터의 프라이버시를 보호하면서도 사용자 맞춤형 이메일을 생성합니다.

- **Technical Details**: Panza는 '데이터 플레이백(data playback)'이라는 새로운 기술을 도입하여 소량의 사용자 데이터로도 LLM을 미세 조정(fine-tuning) 할 수 있습니다. 초기 비지시형 LLM이 사용자 이메일 샘플을 요약하도록 한 후, 이러한 요약된 데이터로 LLM을 재학습시킵니다. 또한, Retrieval-Augmented Generation (RAG)을 활용하여, 추론 및 훈련 단계에서 사용자 스타일을 반영하도록 했습니다. 예를 들어, Robust Adaptation (RoSA), Parameter-Efficient Fine-Tuning (PEFT) 등의 최신 기술이 적용되었습니다.

- **Performance Highlights**: Panza는 일반 하드웨어 (commodity hardware)에서도 효율적으로 실행될 수 있습니다. 실험 결과, 데이터 플레이백 기법은 기존의 다른 방법들, 예를 들어 RAG만을 사용한 모델보다 우수한 성능을 보였습니다. 특히, BLEU, ROUGE, MAUVE 등의 평가지표에서도 일관된 성능 향상을 보여주었으며, 제한된 자원 내에서 만족스러운 이메일 생성 결과를 얻을 수 있었습니다.



### The Effects of Embodiment and Personality Expression on Learning in LLM-based Educational Agents (https://arxiv.org/abs/2407.10993)
Comments:
          15 pages, 4 figures, 3 tables

- **What's New**: 이번 연구는 교육 대화형 에이전트(educational conversational agents)에서 성격 표현과 구체화(embodiment)가 성격 인식과 학습에 미치는 영향을 조사한다. LLM(대형 언어 모델)을 기반으로 한 대화 지원을 기존의 성격 중심 대화 에이전트 프레임워크에 통합하여 교육 애플리케이션에 맞게 조정하였다. 이를 통해 높은 외향성과 친화성을 지닌 스타일과 낮은 외향성과 친화성을 지닌 스타일을 평가하는 사용자 연구를 수행하였다.

- **Technical Details**: 시스템에서 성격 표현을 위해 높은 외향성과 친화성, 낮은 외향성과 친화성 두 가지 성격 스타일을 설정하고, 성격 표현을 대화 텍스트와 / 또는 신체 및 얼굴 애니메이션을 통해 나타내는 세 가지 모델을 평가하였다. 각 모델은 음성 피드백과 동시에 대화 텍스트를 표시하였다. 이 연구는 독립된 사용자 연구를 통해 학습, 품질 및 참여도 평가를 수집하였다.

- **Performance Highlights**: 결과는 모든 모델이 성격과 학습 성과 면에서 긍정적으로 인식되었음을 나타낸다. 특히 높은 성격 특성을 지닌 에이전트는 낮은 성격 특성을 지닌 에이전트보다 더 매력적으로 평가되었다. 또한, 참여자들이 에이전트의 성격을 지각하는 방식에 따라 성격 특성에 따른 변동성을 고려하였다. 이를 통해 외향성과 친화성이 높은 에이전트는 더 감정적으로 안정된 것으로 인식되는 경향이 있는 반면, 대화만으로 성격을 표현하는 에이전트는 성실하다고 인식될 수 있음을 확인하였다.



### AlleNoise -- large-scale text classification benchmark dataset with real-world label nois (https://arxiv.org/abs/2407.10992)
- **What's New**: 이 논문에서는 AlleNoise라는 새로운 텍스트 분류 벤치마크 데이터셋을 소개합니다. 이 데이터셋은 실제 사례에 기반한 라벨 노이즈(instance-dependent label noise)를 포함하며, 약 5,600개의 클래스와 500,000개 이상의 예제를 포함하고 있습니다. 이 데이터셋의 라벨 노이즈는 주요 전자상거래 플랫폼 사용자를 통해 수집된 것으로, 더욱 현실적인 라벨 노이즈 분포를 제공합니다.

- **Technical Details**: AlleNoise 데이터셋은 주요 전자상거래 플랫폼인 Allegro.com에서 수집된 502,310개의 제품 타이틀로 구성되어 있으며, 5,692개의 다양한 카테고리로 분류되어 있습니다. 이 중 15%는 잘못된 카테고리에 등록된 제품으로, 각각의 항목에 대해 제품 제목과 잘못된 카테고리, 그리고 전문가에 의해 확인된 올바른 카테고리를 제공합니다. 또한, 제품 카테고리의 계층적 분류를 위한 매핑(mapping) 정보도 포함되어 있어 노이즈의 의미론적 분석이 가능합니다.

- **Performance Highlights**: 기존의 라벨 노이즈 처리 방법들이 이 데이터셋에서 충분히 성과를 보이지 못한다는 증거를 제시합니다. 특히, 이 방법들은 합성된(synthetic) 노이즈에 대해서는 효과적이지만, 실제 현실 세계의 라벨 노이즈를 처리하는 데에는 한계가 있습니다. 이로 인해 AlleNoise는 텍스트 분류 작업에서 실제 라벨 노이즈를 처리할 수 있는 방법을 개발하는 데 높은 기준을 제시합니다.



### Classification of Geological Borehole Descriptions Using a Domain Adapted Large Language Mod (https://arxiv.org/abs/2407.10991)
- **What's New**: 이번 연구는 네덜란드어로 작성된 벨기에 플란더스 지역의 지질 시추공(borehole) 설명을 학습한 대규모 언어 모델 GEOBERTje를 소개합니다. GEOBERTje는 지질 시추공 설명에서 중요한 정보를 효과적으로 추출하여 수치 벡터 공간으로 변환합니다. 이 연구는 GEOBERTje의 잠재적인 응용 중 하나로, 소량의 수작업으로 라벨링된 데이터를 사용해 분류기(classifier) 모델을 미세 조정(finetune)하여 시추공 설명을 주요 및 보조 광물 분류(classification)로 나누는 작업을 수행했습니다. 이 분류기는 기존의 규칙 기반 접근법(rule-based approach)과 OpenAI의 GPT-4를 능가했습니다.

- **Technical Details**: GEOBERTje는 네덜란드어로 제공된 283,000개의 벨기에 플란더스 지역의 시추공 설명 데이터를 사용하여 훈련된 도메인 적응 대규모 언어 모델입니다. 모델 도메인 적응은 전이 학습(transfer learning)을 통해 이루어졌으며, 약 2,500개의 라벨링된 데이터를 사용하여 석재 분류기(lithology classifier)로 미세 조정됩니다. GEOBERTje는 BERT(Transformers)의 혁신적인 아키텍처를 기반으로 다양한 텍스트의 중요성을 동적으로 가중하는 자가 주의 메커니즘(self-attention mechanism)을 이용합니다.

- **Performance Highlights**: GEOBERTje 기반 분류기는 기존 규칙 기반 방법과 GPT-4를 뛰어넘는 성능을 보이며, 복잡하고 비정형적인 지질 기술에서 정보를 추출하는 효율성과 정확성을 향상시켰습니다. 이는 방대한 양의 지질 데이터를 활용한 새로운 분석 및 모델링 기회를 제공합니다.



### MedBench: A Comprehensive, Standardized, and Reliable Benchmarking System for Evaluating Chinese Medical Large Language Models (https://arxiv.org/abs/2407.10990)
Comments:
          25 pages.4 figures

- **What's New**: 중국어 의료 대형 언어 모델(Large Language Models, LLM)의 효과성 및 인류에 대한 선익성을 보장하기 위한 평가지표가 아직 확립되지 않았습니다. MedBench는 이를 해결하기 위해 도입된 종합적이고 표준화된 벤치마킹 시스템입니다. MedBench는 현재까지 가장 큰 평가 데이터셋(300,901개의 질문)을 구성하여 43개의 임상 전문 분야를 다룹니다.

- **Technical Details**: MedBench는 표준화된 완전 자동화된 클라우드 기반 평가 인프라 구조를 제공하며, 질문과 정답을 물리적으로 분리합니다. 또한, MedBench는 동적 평가 메커니즘을 구현하여 단축 학습(shortcut learning)과 답변 암기(answer remembering)를 방지합니다.

- **Performance Highlights**: MedBench를 일반 및 의료 LLM에 적용한 결과, 의학 전문가들의 관점과 일치하는 편향 없는 재현 가능한 평가 결과가 나오고 있습니다. 이 연구는 중국어 의료 LLM의 실용적인 응용을 준비하기 위한 중요한 토대를 마련했습니다.



### Do Large Language Models Understand Verbal Indicators of Romantic Attraction? (https://arxiv.org/abs/2407.10989)
- **What's New**: 최근 연구에서 대규모 언어 모델(Large Language Models, LLMs)이 첫 만남에서 로맨틱한 끌림을 감지할 수 있다는 사실이 밝혀졌습니다. 964개의 스피드 데이트 데이터를 분석한 결과, ChatGPT와 Claude 3는 객관적 및 주관적인 성공 지표를 예측하는 데 성공했습니다 (상관계수 r=0.12-0.23). 이는 인간 평가자와 비슷한 수준의 정확성을 보이며, 참가자들의 예측을 넘어서 추가적인 정보를 제공했습니다.

- **Technical Details**: 이 연구는 스피드 데이트에서 교류된 대화 데이터를 분석하여 진행되었습니다. ChatGPT는 대화의 긍정적 정서(valence)와 같은 일반적인 내용 차원을 사용하여 예측했지만, 대화 역학(conversational dynamics)에서도 상당한 예측력을 발휘했습니다. 특히, ChatGPT의 판단은 인간 참관자의 평가와 상당히 일치(r=0.29)하는 것으로 나타났으며, 이는 로맨틱한 끌림의 표현 방식이 일부 독립적인 정확성을 갖추고 있다는 것을 암시합니다.

- **Performance Highlights**: ChatGPT는 스피드 데이트에서 실제로 성사된 매칭(연락처 교환)을 예측하는데 인간 판단자와 동등한 성과를 보였습니다. 또한 사람의 판독과 비교했을 때, ChatGPT는 참가자들의 자체 예측을 보완하여 추가적인 인사이트를 제공했다는 점에서 주목할 만합니다.



### Reasoning with Large Language Models, a Survey (https://arxiv.org/abs/2407.11511)
- **What's New**: 이번 리뷰 논문은 거대 언어 모델(Large Language Models, LLMs)을 활용한 프롬프트 기반 추론(prompt-based reasoning)의 급속히 확장되고 있는 분야를 정리하고 있습니다. 이 논문은 LLMs의 합성적 추론 System 1과 더불어 연쇄 사고(Chain-of-Thought) 프롬프트 학습을 통해 System 2 추론 능력을 보이는 최근의 진전을 다루고 있습니다. 특히, 초등학교 수학 문제를 해결하는 LLM의 연구에 중점을 두고 있으며, 이는 인공지능(AI)의 추론 능력에 대한 중요한 질문을 탐구하고 있습니다.

- **Technical Details**: 이 논문에서는 LLMs의 다양한 추론 방법에 대해 세분화된 분류 체계를 제공합니다. 프롬프트 생성, 평가, 제어 등 여러 측면에서 multi-step 추론을 다루고 있으며, 주로 프롬프트 기반 접근법에 초점을 맞추고 있습니다. 핵심적으로 LLM들이 체계적인 사고를 수행할 수 있도록 돕는 'Let’s think step by step'와 같은 간단한 지시를 포함한 Chain-of-thought 실험이 관찰되었습니다. 이와 더불어, 모델 훈련 파이프라인, in-context learning(맥락 내 학습), self-reflection(자기 성찰), metacognition(메타인지) 등 다양한 관련 키워드들을 아우르고 있습니다.

- **Performance Highlights**: System 1 과제에서는 LLM이 이미 뛰어난 성능을 보입니다. 특히 번역, 요약, 질의 응답 등의 작업에서 우수한 결과를 보이고 있습니다. 하지만 System 2 과제에서는 여전히 많은 어려움을 겪고 있는데, Chain-of-thought 프롬프트 학습을 통해 놀라운 성능 향상을 이룬 사례들이 있습니다. 이러한 성과는 초등학교 수학 문제 해결과 같은 구체적인 벤치마크(GSM8K)에서 관찰되었으며, 이는 인공지능 커뮤니티와 사회 모두에게 큰 주목을 받고 있습니다.



### MMSD-Net: Towards Multi-modal Stuttering Detection (https://arxiv.org/abs/2407.11492)
Comments:
          Accepted at INTERSPEECH 2024

- **What's New**: 연구팀은 MMSD-Net이라는 세계 최초의 다중 모달 (multi-modal) 신경망 프레임워크를 통해 말을 더듬는(stuttering) 현상을 자동적으로 탐지할 수 있는 방법을 제안했습니다. 기존의 단일 모달(uni-modal) 접근법들은 오디오나 텍스트 기반에 치중해 왔으나, MMSD-Net은 비디오 신호를 추가로 사용하여 탐지 정확도를 크게 높였습니다.

- **Technical Details**: MMSD-Net의 모델 아키텍처는 비디오, 오디오, 그리고 텍스트 데이터를 처리하는 세 가지 변형기 인코더(transformer encoder)로 구성됩니다. 각 인코더는 해당 모달리티에서 가장 관련 있는 특징을 추출하며, 이 접근 방식은 장기 의존성(long-range dependencies)을 잘 포착해 말을 더듬는 현상의 세부적인 요소들을 인식할 수 있도록 도와줍니다. 이를 통해 MMSD-Net은 기존보다 더욱 포괄적인 분석이 가능하게 되었습니다.

- **Performance Highlights**: MMSD-Net은 최신의 단일 모달 방법들보다 F1-score에서 2-17%의 향상을 보여주었습니다. 이는 특히 시각적 신호를 활용한 점이 주요한 기여를 한 결과로 볼 수 있습니다. 실험 결과는 공개된 데이터셋을 통해 검증되었으며, 사용된 코드도 공개하여 연구의 재현성을 보장하고 있습니다.



### The Oscars of AI Theater: A Survey on Role-Playing with Language Models (https://arxiv.org/abs/2407.11484)
Comments:
          28 pages

- **What's New**: 이 서베이는 언어 모델(LLMs)의 롤플레잉(Role-Playing) 분야의 급성장을 탐구합니다. 초기 퍼소나 기반의 모델에서 LLM을 활용한 고도화된 캐릭터 중심 시뮬레이션으로 발전했습니다. 이러한 롤플레잉은 캐릭터 일관성, 행동 정렬 및 전반적인 매력과 같은 복잡한 캐릭터 묘사를 포함하는 방향으로 확장되었습니다. 본 논문은 이러한 시스템을 설계하는 데 중요한 구성 요소들에 대한 포괄적인 분류법을 제공하며, 현재의 방법론과 문제점들을 개괄하고 미래 연구의 방향을 제시합니다.

- **Technical Details**: 본 논문은 데이터, 모델 및 정렬, 에이전트 아키텍처, 평가와 같은 주요 요소를 체계적으로 검토합니다. 롤플레잉 모델의 효과를 향상시키기 위해 중요한 모듈들이 어떻게 작동하는지를 상세히 분석하고, 다양한 어플리케이션에서 어떻게 최적화되고 평가될 수 있는지 논의합니다. 특히 데이터의 다양성과 복잡성, 모델의 훈련 방법론, 에이전트의 기억 및 계획 능력 등이 롤플레잉의 현실감을 높이는 데 중요한 역할을 합니다.

- **Performance Highlights**: 현재 롤플레잉 연구는 더 이상 단순한 퍼소나 일관성에 국한되지 않고, 캐릭터의 일관성과 행동의 정렬, 전반적인 매력 등을 포함하는 복잡한 캐릭터 묘사로 나아가고 있습니다. 이러한 진보는 예전보다 더욱 몰입감 있고 실감나는 캐릭터 시뮬레이션을 가능하게 하며, 이를 통해 사용자들은 더 연속적이고 다이내믹한 상호작용을 경험할 수 있습니다. 또한, LLM 기반 롤플레잉의 진보는 학문적 연구와 실용적인 응용 개발을 급속도로 확장시키고 있습니다.



### Beyond Correctness: Benchmarking Multi-dimensional Code Generation for Large Language Models (https://arxiv.org/abs/2407.11470)
Comments:
          We release benchmark at this https URL and leaderboard at this https URL

- **What's New**: RACE 벤치마크를 제안한 논문이 발표되었습니다. 이 벤치마크는 코드 생성에서 Readability(가독성), mAintainability(유지보수성), Correctness(정확성), Efficiency(효율성) 등 4가지 중요한 차원을 종합적으로 평가합니다. 기존 벤치마크와는 달리, 단순한 코드의 정확성 뿐만 아니라 다양한 사용자 요구를 충족시키는 고품질 코드 생성 능력까지 평가합니다.

- **Technical Details**: RACE 벤치마크는 다음과 같은 주요 기술적 도전 과제를 해결하는 것을 목표로 합니다. 첫째, 각 차원을 정량적으로 평가할 수 있는 프레임워크를 설계합니다. 이를 위해 여러 대표적인 요인을 통합하여 코드 품질을 평가합니다. 둘째, 각 요인별 사용자 요구를 반영한 다양한 평가 메트릭을 설계합니다. 예를 들어, 가독성을 평가하기 위해 여러 단계의 주석이 포함된 코드를 생성하거나, 효율성 측정을 위해 시간 효율성과 공간 효율성을 균형 있게 맞춘 코드를 생성합니다. 셋째, 평가 메트릭을 정리하고, 정적 분석 및 런타임 모니터링 방법을 통해 LLM이 생성한 코드의 품질을 정확하게 평가합니다.

- **Performance Highlights**: 18개의 대표적인 LLM을 RACE 벤치마크로 평가한 결과는 다음과 같습니다. 1) 현재의 LLM은 특정 요구를 충족시키는 정확한 코드를 생성하는 데 어려움을 겪고 있어 소프트웨어 개발에 부족합니다. 특히, GPT-4와 DeepSeek-Coder-V2는 각 차원에서 뛰어난 성능을 보였습니다. 2) 가독성은 전체 코드 품질의 지표로 사용될 수 있으며, 적절한 주석을 추가하면 코드 정확성을 향상시킬 수 있습니다. 3) 대부분의 LLM은 특정 코딩 스타일을 선호하여 사용자 지침과 일치하지 않을 경우 일관성을 유지하는 데 어려움을 겪습니다. 이러한 결과는 현재 Code LLM의 한계를 보여주며, 향후 개선 방향을 제시합니다.



### LOTUS: Enabling Semantic Queries with LLMs Over Tables of Unstructured and Structured Data (https://arxiv.org/abs/2407.11418)
- **What's New**: LOTUS라는 새로운 오픈소스 쿼리 엔진이 소개되었습니다. LOTUS는 대규모 데이터 세트에서 자연 언어 기준으로 레코드를 정렬 또는 집계하는 등의 의미적 쿼리를 수행하기 위한 선언적 프로그래밍 인터페이스를 제공합니다. 이 새로운 시스템은 사실 확인, 극단적 다중 라벨 분류, 검색 등 여러 실제 애플리케이션에서 높은 성능을 입증했습니다.

- **Technical Details**: LOTUS는 관계형 모델을 확장하여 의미적 필터, 조인, 랭킹, 집계 및 프로젝션과 같은 AI기반 연산자(semantic operators)를 소개합니다. 이 연산자들은 최적화된 알고리즘과 함께 작동하며, 다양한 데이터 구조와의 인터페이스를 제공합니다. 또한 LOTUS는 Pandas와 유사한 API를 갖추고 있어 개발자들이 쉽게 사용할 수 있습니다. 이 시스템은 효과적인 병렬 배치 추론을 가능하게 하고, 모델 캐스케이드를 활용하여 경량 스코어링 기능을 최대화합니다.

- **Performance Highlights**: LOTUS는 FEVER 데이터 세트에서 9.5% 높은 정확도를 제공하면서도 실행 시간을 7-34배 단축시켰습니다. BioDEX 데이터 세트에서 LOTUS의 조인 연산자를 통해 800배 더 빠른 알고리즘 실행 시간이 확인되었습니다. 검색 및 랭킹 애플리케이션에서는 일반적인 검색 및 재랭커 방법보다 nDCG@10이 5.9-49.4% 향상되었으며, 실행 시간도 1.67-10배 줄어들었습니다.



### CIC-BART-SSA: Controllable Image Captioning with Structured Semantic Augmentation (https://arxiv.org/abs/2407.11393)
Comments:
          Accepted to ECCV 2024

- **What's New**: 이미지 캡셔닝(image captioning) 분야에서 제어 가능한 이미지 캡셔닝(CIC, Controllable Image Captioning)이란 사용자가 제공하는 정보를 바탕으로 이미지의 자연어 설명을 생성하는 기술입니다. 기존 데이터셋이 전체 이미지를 설명하는 캡션만 포함하고 있어, 부분적인 영역이나 관계를 설명하는 CIC 모델을 효과적으로 학습시키기 어렵습니다. 이를 개선하기 위해, 구조화된 의미 표현(AMR, Abstract Meaning Representation)을 활용하여 자동으로 집중된 캡션을 샘플링하는 완전 자동 방법을 제안했습니다.

- **Technical Details**: 우리는 AMR이라는 형식주의를 사용하여 이미지와 관련된 모든 공간-의미 관계를 인코딩하는 '구조적 의미 증강(SSA, Structured Semantic Augmentation)' 프레임워크를 활용하여 기존 이미지-캡션 데이터셋을 다양하고 집중된 캡션으로 증강했습니다. 그 후, 이 SSA로 다채롭게 된 데이터셋을 활용하는 새 모델인 CIC-BART-SSA를 개발했습니다. CIC-BART-SSA는 SSA를 통해 다양해진 데이터셋을 통제 신호로 사용하여 캡션을 생성합니다.

- **Performance Highlights**: 우리의 실험 결과 CIC-BART-SSA 모델이 기존 SOTA 모델에 비해 텍스트 품질과 다양성 면에서 우수한 캡션을 생성하며, 통제 가능성에서도 경쟁력을 갖추고 있음을 보여주었습니다. 또한, 광범위한 통제와 매우 집중된 시나리오 간의 성능 격차를 최소화했습니다. 코드는 공개되었습니다.



### A Pilot Study of GSLM-based Simulation of Foreign Accentuation Only Using Native Speech Corpora (https://arxiv.org/abs/2407.11370)
Comments:
          Accepted to INTERSPEECH2024

- **What's New**: 본 논문에서는 인간의 외국어 억양 모사 과정을 네이티브 음성 코퍼스만을 사용하여 생성적 음성 언어 모델(Generative Spoken Language Model, GSLM)을 활용해 시뮬레이션하는 방법을 제안합니다. 이를 통해 언어 A의 음성을 언어 B의 GSLM에 입력하여 B의 억양을 추가하는 과정을 모사합니다.

- **Technical Details**: GSLM 기반의 시스템은 HuBERT 모델을 사용하여 입력 음성을 벡터 시퀀스로 변환 후, S2u(구간 화 및 단위화)와 u2S(단위에서 음성으로 변환)로 나누어 처리합니다. 이 과정에서 k-평균 군집화(k-means clustering)를 사용해 단위화를 수행하며 Tacotron2를 통해 TTS 모델을 학습시킵니다. B 언어에 대해 미리 학습된 HuBERT 모델과 군집화 모델, u2S 모델을 사용하여 A 언어의 음성을 입력해 B 억양의 음성을 생성합니다.

- **Performance Highlights**: 실험 결과를 통해 생성된 억양이 실제 B 언어 사용자들이 발음한 A 언어 음성과 비교하여 높은 자연스러움을 보였으며, 억양의 강도도 조절 가능하다는 것을 확인했습니다.



### OmniGenome: Aligning RNA Sequences with Secondary Structures in Genomic Foundation Models (https://arxiv.org/abs/2407.11242)
Comments:
          submitted to NeurIPS 2024, 19 pages

- **What's New**: 이 연구에서는 OmniGenome이라는 새로운 RNA 구조-서열 정렬(Sequence-Structure Alignment) 파운데이션 모델을 도입했다. 기존 모델들이 해내지 못한 복잡한 RNA 디자인 작업을 해결할 수 있는 구조 하나로 간주되는 서열을 모델링 하는데 중점을 두고 있다.

- **Technical Details**: OmniGenome은 서열-구조 정렬을 개선하기 위해 구조-맥락화 모델링을 사용한다. 모델은 Seq2Str (Sequence-to-Structure)와 Str2Seq (Structure-to-Sequence) 두 가지 예측 작업을 동시에 다룬다. Seq2Str가 데이터 수급 문제를 해결하는 동안, Str2Seq는 서열 복원 임무를 수행하여 중복된 같은 구조를 갖는 서로 다른 서열을 예측한다.

- **Performance Highlights**: 두 개의 종합 유전체 벤치마크에서 OmniGenome은 기존의 모델들보다 월등히 높은 성능을 나타냈다. 예를 들어 Eterna V2 벤치마크에서 74%의 복잡한 퍼즐을 해결했으며, 기존의 SpliceBERT 모델이 3%만 해결한 것과 비교된다. 뿐만 아니라 OmniGenome은 대부분의 퍼즐을 1시간 이내에 해결하였다. 이 새로운 모델은 기존 모델들 보다 최대 35% 성능 향상을 나타냈다.

- **Open-source Resources**: OmniGenome 연구팀은 벤치마크, 평가 스크립트 및 FM 튜토리얼 등을 포함한 오픈소스 패키지를 개발하였다. 이를 통해 연구자들은 간단한 코드 몇 줄로 파운데이션 모델의 사전 학습과 하위 작업 미세 조정을 자동화할 수 있다. 이는 향후 유전체 연구에서 시간을 절약하고 프로그래밍 수고를 덜어줄 것이다.



### Making New Connections: LLMs as Puzzle Generators for The New York Times' Connections Word Gam (https://arxiv.org/abs/2407.11240)
- **What's New**: 이번 연구는 뉴욕타임스(NYT)에서 매일 발행하는 단어 연결 퍼즐 'Connections'를 생성하는 방법에 대한 연구이다. 이 퍼즐은 주어진 16개의 단어를 공통된 주제로 묶는 게임으로, GPT 계열의 대형 언어 모델(LLM)을 활용하여 창의적이고 도전적인 퍼즐을 생성할 수 있는지 평가하였다. 이는 LLMs가 단어 게임 생성에도 효과적으로 사용될 수 있음을 보여준다.

- **Technical Details**: 연구진은 'Tree of Thoughts(ToT)' 프롬프트 방식을 적응하여 Connections 퍼즐을 생성하는 방법을 제안했다. LLMs는 주어진 단어들을 다양한 카테고리로 분류해야 하는 메타인지 능력이 필요하기 때문에, 이 접근방식은 LLMs의 능력을 평가하기에 적합하다. 또한, 사용자 연구를 통해 인간이 만든 퍼즐과 AI가 생성한 퍼즐을 비교 평가하였다.

- **Performance Highlights**: 사용자 연구 결과, AI가 생성한 퍼즐은 인간 사용자의 판단에 의해 즐겁고 창의적이며 도전적인 것으로 평가되었다. 이는 LLM이 새로운 퍼즐을 성공적으로 생성할 수 있는 능력을 가지고 있음을 시사한다. 연구의 결과는 또한 LLM의 강점과 약점을 식별하고 향후 연구 방향을 제시하는 데 기여하였다.



### Mechanistic interpretability of large language models with applications to the financial services industry (https://arxiv.org/abs/2407.11215)
- **What's New**: 본 논문에서는 기계적 해석 가능한 방법(mechanistic interpretability)을 사용하여 대형 언어 모델(LLM)의 내부 작동 방식을 이해하고 금융 서비스 애플리케이션에 적용하는 방법을 제시합니다. 특히, GPT-2 Small 모델을 사용해 Fair Lending 법률 위반 가능성을 식별하는 작업에서의 주의 패턴과 기여도를 분석합니다.

- **Technical Details**: 기계적 해석 가능성(mechanistic interpretability)은 복잡한 AI 모델을 이해하기 위한 접근법으로, 뉴런, 회로, 주의 머리(attention heads) 수준에서의 미세한 분석을 포함합니다. 이 논문에서는 GPT-2 Small 모델의 조건적 로그 차이를 직접 추적하여 각 레이어와 그에 대응하는 주의 머리의 기여도를 연구합니다. 또한, 깨끗한 프롬프트와 손상된 프롬프트를 디자인하고, 활성 패칭(activation patching) 기법을 사용하여 과제를 완료하는 구성 요소를 추가적으로 위치시킵니다.

- **Performance Highlights**: 연구 결과, 주목할만한 (positive) 주의 머리 10.2 (10 레이어의 2번째 머리), 10.7, 11.3 과 (negative) 주의 머리 9.6, 10.6이 Fair Lending 법규 위반 가능성을 식별하는 작업에 중요한 역할을 한다는 것을 발견했습니다.



### PutnamBench: Evaluating Neural Theorem-Provers on the Putnam Mathematical Competition (https://arxiv.org/abs/2407.11214)
- **What's New**: PutnamBench는 새로운 다중 언어 벤치마크로, 신경망 정리 증명기(neural theorem-prover)가 수학 경시대회 문제를 해결하는 능력을 평가하기 위해 개발되었습니다. 이 벤치마크는 북미 최고의 학부 수학 경시대회인 William Lowell Putnam Mathematical Competition에서 출처한 640개의 정리를 1697개의 수작업 형식화 과정을 통해 구성한 것입니다.

- **Technical Details**: 모든 정리는 Lean 4와 Isabelle로 형식화되어 있으며, 상당한 부분은 Coq 형식화도 포함하고 있습니다. 이러한 정리들을 증명하기 위해서는 문제 해결 능력과 학부 수학 과정에서 다루는 다양한 주제에 대한 숙달이 요구됩니다.

- **Performance Highlights**: PutnamBench를 사용해 여러 기존의 신경망 및 기호적 정리 증명기를 평가한 결과, 이들 접근 방식은 PutnamBench 문제 중 일부만 해결할 수 있었습니다. 이는 신경망 정리 증명 연구에서 해결해야 할 중요한 도전 과제로 남게 될 것입니다.



### Unconstrained Open Vocabulary Image Classification: Zero-Shot Transfer from Text to Image via CLIP Inversion (https://arxiv.org/abs/2407.11211)
- **What's New**: NOVIC이라는 혁신적인 모델이 소개되었습니다. uNconstrained Open Vocabulary Image Classifier라고 불리는 이 모델은 autoregressive transformer를 사용하여 분류 라벨을 언어 형태로 생성해냅니다. NOVIC은 기존 CLIP 모델의 광범위한 지식을 활용하여 텍스트에서 이미지로의 zero-shot 전이(transfer)를 가능하게 합니다. 이 모델은 이미지의 잠재적인 내용을 미리 알 필요 없이, 이미지로부터 임베딩 벡터(embedding vectors)를 생성하고 이를 통해 텍스트형 객체 라벨을 직접 생성할 수 있습니다.

- **Technical Details**: NOVIC은 'object decoder' 모델을 사용하며, 이 모델은 템플릿 객체 명사 세트와 LLM이 생성한 캡션으로 구성된 대규모 92M 타겟 데이터셋을 트레이닝하는 과정을 거칩니다. 이 과정에서 CLIP 텍스트 인코더를 역전시키고, 이미지로부터 추출한 임베딩 벡터를 통해 객체 명사를 생성해냅니다. 트레이닝 데이터셋은 완전히 텍스트 기반으로 구성되어 매우 효율적이고 확장성이 뛰어납니다. 노이즈 보강(noise augmentation) 기법을 통해 이미지와 텍스트 임베딩 간의 큰 차이를 극복합니다.

- **Performance Highlights**: NOVIC은 다양한 데이터셋에서 테스트되었으며, 최대 87.5%의 세밀한 프롬프트 없이 예측 성능을 보였습니다. 이는 다양한 이미지에서 컨텍스트 단서 없이도 정확한 분류 결과를 제공함으로써 현실적인 응용 범위에서 높은 성능을 입증합니다.



### The Life Cycle of Large Language Models: A Review of Biases in Education (https://arxiv.org/abs/2407.11203)
Comments:
          20 pages, 2 figures, preprint for British Journal of Educational Technology submission

- **What's New**: 대형 언어 모델(LLMs)이 교육 환경에서 개인 맞춤형 지원을 제공하기 위해 점점 더 많이 채택되고 있습니다. 이를 통해 교육의 효율성을 높이고 학습 성과를 향상시킬 수 있지만, 알고리즘적 편향성으로 인해 교육 불평등이 악화될 수 있다는 우려가 제기되었습니다. 본 리뷰는 LLM의 생애 주기를 체계적으로 설명하고, 교육 환경에서 발생할 수 있는 편향의 잠재적 원인을 식별합니다. 이를 통해 교육의 공평성을 촉진하기 위한 평가 지침을 제공합니다.

- **Technical Details**: 기존 머신러닝 생애 주기를 기반으로 LLM 생애 주기를 초기 개발부터 교육 설정에서 사용되는 맞춤형 모델까지 전체적으로 맵핑했습니다. LLM 개발의 각 단계에서 발생할 수 있는 편향성을 조사하고, 전통적인 머신러닝 측정 방법이 왜 LLM 생성 콘텐츠에는 적합하지 않은지를 설명합니다. 또한, 텍스트가 고차원적이고 여러 가지 정답이 있을 수 있으며, 맞춤형 응답이 교육적으로 바람직할 수 있기 때문에 교육적 맥락에서의 편향 문제를 심층 분석합니다.

- **Performance Highlights**: GPT-4와 같은 최신 모델은 특정 상황에서 매우 높은 성능을 보이며, 개인화된 지원을 통해 학생들의 학습을 돕고 교사들의 업무를 보조할 수 있습니다. 예를 들어, 실시간 질문 답변, 과제에 대한 즉각적이고 맞춤형 피드백, 과제 생성 및 빠른 채점을 지원합니다. LLM 기반 기술은 쿠세라(Coursera), 에드엑스(EdX) 등의 주요 에드테크 플랫폼에서도 사용되고 있습니다. 하지만 이러한 기술이 잠재적인 교육 불평등을 심화시킬 수 있는 알고리즘적 편향을 일으킬 가능성이 있음을 강조합니다.



### AstroMLab 1: Who Wins Astronomy Jeopardy!? (https://arxiv.org/abs/2407.11194)
Comments:
          45 pages, 12 figures, 7 tables. Submitted to ApJ. Comments welcome. AstroMLab homepage: this https URL

- **What's New**: 최초의 천문학 전용 벤치마킹 데이터셋을 사용하여 독점 및 오픈 웨이트(Weights) 대형 언어 모델(LLM)의 종합적인 평가를 발표했습니다. 데이터셋은 천문학 및 천체물리학 연례 검토(Annual Review of Astronomy and Astrophysics)에서 수집된 총 4,425개의 객관식 질문으로 구성되어 있습니다.

- **Technical Details**: 데이터셋은 여러 천문학적 하위 분야를 포괄하며, 연구 환경에서의 잠재적 배포를 위해 모델의 응답 보정을 평가했습니다. 클로드-3.5-소네(Claude-3.5-Sonnet) 모델이 다른 모델을 4.6% 포인트까지 앞서며 85.0%의 정확도를 달성했습니다. 독점 모델의 경우, 3~12개월마다 비용이 크게 감소하여 비슷한 점수를 얻었습니다. 오픈 소스 모델(Open-source models)도 빠르게 개선되어, 라마-3-70b(LLaMA-3-70b)는 80.6%, 코엔-2-72b(Qwen-2-72b)는 77.7%로 현재 일부 최상위 독점 모델과 경쟁할 수 있습니다.

- **Performance Highlights**: 탑 모델들은 자신감과 정확성의 상관관계가 0.9 이상인 잘 보정된 자신감을 보여주지만, 약간 과소평가되는 경향이 있습니다. 최상위 성능을 보이는 모델들도 훈련 데이터의 다양성 부족으로 인해 외계 행성 관련 분야, 항성 천체물리학, 기기 관련 질문에서 어려움을 겪는 경향이 있습니다. 이러한 패턴은 오픈 웨이트 및 독점 모델 전반에서 나타나며, 지역별 종속성을 강조합니다. 신속하고 저비용의 추론을 제공하는 오픈 웨이트 모델의 개발은 천문학에의 저렴한 배포 가능성을 열어줍니다.



### In Silico Sociology: Forecasting COVID-19 Polarization with Large Language Models (https://arxiv.org/abs/2407.11190)
- **What's New**: 이 연구에서는 대규모 언어 모델(LLMs)을 사용하여 특정 사회 및 문화적 문맥에서 응답자를 정확하게 시뮬레이션할 수 있는 능력을 탐구합니다. 특히, 연구진은 2019년의 여론 지형을 재구성하여 COVID-19에 대한 미래의 정치적 양극화가 기존의 정치적 담론에 얼마나 예견되었는지를 조사합니다.

- **Technical Details**: LLM은 2019년까지 출판된 텍스트에 대해 훈련된 모델을 사용하여 미국의 자유주의자와 보수주의자들이 팬데믹 관련 질문에 어떻게 응답할지 시뮬레이션합니다. 모델은 문장 내 단어 순서를 예측하여 텍스트를 생성하며, 이로 인해 복잡한 언어 패턴을 학습합니다. GPT-3와 같은 모델이 이에 해당합니다.

- **Performance Highlights**: 시뮬레이션된 응답은 COVID-19 태도에 대한 주요 차이를 84%의 정확도로 재현했습니다. 특히, 자유주의자와 보수주의자 사이의 의견 차이는 '자유', '안전', '기관 신뢰'에 대한 다른 호소에서 기인한다는 점을 발견했습니다. 이러한 결과는 COVID-19의 정치화가 기존 사상적 지형과 일관되었음을 시사합니다.



### Building Intelligence Identification System via Large Language Model Watermarking: A Survey and Beyond (https://arxiv.org/abs/2407.11100)
Comments:
          59 pages, 7 figures

- **What's New**: 이번 논문에서는 대형 언어 모델 (Large Language Models; LLMs)의 보안 위험을 줄이기 위한 효과적인 식별 메커니즘을 중심으로 논의합니다. 특히, 워터마킹(watermarking) 기술을 이용하여 LLMs의 지적 재산 보호와 데이터 보안을 관리하는 방법에 대해 연구합니다. 이 논문은 워터마킹 이론과 실제 적용 사례를 종합적으로 분석하고, 지능형 식별 관점에서 개선된 모델을 제안합니다.

- **Technical Details**: 우리는 상호 정보 이론(mutual information theory)에 기반한 수학적 프레임워크를 제안하며, 이를 통해 워터마킹 식별 과정을 체계화하고 맞춤형 워터마킹을 구현할 수 있습니다. 또한 워터마킹 기술을 생성, 삽입, 공격, 추출, 재구성의 5단계로 분류하고 이러한 각 단계의 최적화 객체와 제약 조건을 수학적으로 설명합니다.

- **Performance Highlights**: 논문은 여러 LLM 참가자의 성향을 반영한 성능 평가 지표를 종합해 제시하며, 이를 통해 LLM 워터마킹의 표준화된 평가 시스템을 개발하게 합니다. 이를 통해 LLM 생태계의 보안 문제와 투명성을 향상시킬 수 있는 새로운 연구 경로와 기술 방향을 제시합니다.



### Show, Don't Tell: Evaluating Large Language Models Beyond Textual Understanding with ChildPlay (https://arxiv.org/abs/2407.11068)
- **What’s New**: 이번 연구에서는 GPT-3.5와 GPT-4 같은 대형 언어 모델(LLMs)이 비언어적 영역에서도 인지 기능을 가지고 있는지를 탐구했습니다. 이를 위해 표준 언어 벤치마크를 초과하여 Tic-Tac-Toe, Connect Four, Battleship 같은 게임을 ASCII 형식으로 인코딩하여 전략적 사고와 의사결정 능력을 평가했습니다. 또한, 이 모델들이 훈련 데이터 외의 일반화 능력을 평가하기 위해 LEGO Connect Language (LCL)와 도형 게임을 도입했습니다. 이러한 '보여줘, 말하지마' 전략을 통해 단순 질의 응답이 아닌 게임을 통해 LLMs를 평가했습니다.

- **Technical Details**: GPT-3.5 및 GPT-4 모델은 변형기(transformer) 기반 구조를 사용하며, 셀프 어텐션(self-attention)의 메커니즘을 통해 문장 내 단어의 중요도를 평가합니다. 입력 텍스트는 토큰화되고 벡터로 변환된 후 여러 변형기 층을 통과하며, 최종적으로 다음 토큰을 선택하는 방식으로 작동합니다. 이번 연구에서는 ChildPlay라는 새로운 벤치마크를 도입하였고, 이 벤치마크는 ASCII 형식으로 인코딩된 비언어적 게임들로 구성되어 모델의 전략적 사고 및 공간적 추론 능력을 평가합니다.

- **Performance Highlights**: GPT-3.5와 GPT-4는 표준 벤치마크에서는 우수한 성과를 보였지만, 비언어적 게임인 Tic-Tac-Toe, Connect Four, Battleship에서는 기대 이하의 성과를 보였습니다. Tic-Tac-Toe와 Connect Four에서는 패배수들을 예측하지 못했고, Battleship에서는 올바르게 플레이하지 못했습니다. 도형 게임에서는 GPT-4가 일부 성공을 보였지만, LCL 게임에서는 두 모델 모두 조립 작업을 실패했습니다. 이러한 결과는 대형 언어 모델이 대화 및 기본 규칙 이해에서 우수하더라도, 전략적 게임 플레이 및 공간적 추론 과제에서는 한계가 있음을 나타냅니다.



### EfficientQAT: Efficient Quantization-Aware Training for Large Language Models (https://arxiv.org/abs/2407.11062)
Comments:
          An efficient and effective quantization technical to improve the performance of low-bits LMMs and LVLMs

- **What's New**: 대규모 언어 모델(LLMs)의 메모리 요구를 관리하기 위한 새로운 양자화 기법인 Efficient Quantization-Aware Training (EfficientQAT)이 제안되었습니다. 이 기술은 Block-wise training(Block-AP)과 end-to-end training(E2E-QP)이라는 두 단계로 나누어 LLMs를 효과적으로 압축합니다.

- **Technical Details**: EfficientQAT은 두 단계로 구성됩니다. 먼저, Block-wise training(Block-AP)은 각 Transformer 블록 내의 모든 파라미터에 대해 차례로 양자화 인지 훈련(quantization-aware training)을 수행합니다. 이후 end-to-end training(E2E-QP) 단계에서는 양자화된 모델을 초기값으로 하여, 양자화 파라미터(주로 step sizes)만을 훈련합니다. 이를 통해 메모리 효율성과 양자화 정확도를 동시에 향상시킵니다.

- **Performance Highlights**: EfficientQAT은 7억 개부터 700억 개 파라미터를 가진 다양한 모델에서 우수한 성능을 보이며, 기존 양자화 방법을 능가합니다. 예를 들어, EfficientQAT은 단일 A100-80GB GPU로 41시간 이내에 2-bit Llama-2-70B 모델을 생성할 수 있으며, 정확도 감소는 3% 이하 (69.48 vs. 72.41)입니다. 또한, 이 모델은 13B Llama-2 모델보다 높은 정확도(69.48 vs. 67.81)를 보이며, 메모리 사용량도 더 적습니다.



### Was it Slander? Towards Exact Inversion of Generative Language Models (https://arxiv.org/abs/2407.11059)
Comments:
          4 pages, 3 figures

- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 평판을 손상시키려는 악의적인 공격을 다룹니다. 연구진은 이러한 비방 공격(slander attack)에 대처하기 위해서 허위 출력을 감지하고, 이에 대한 입증을 수행하는 방법을 제안합니다.

- **Technical Details**: 논문에서는 특히, LLMs의 문제 출력을 기반으로 한 입력을 재구성하려는 시도를 포함합니다. 그러나, 텍스트 입력의 불연속성(discrete nature)과 방대한 검색 공간(search space) 때문에 이 작업은 매우 어려운 것으로 나타났습니다. 연구진은 약한 역전(weak inversion)을 통해 문제 출력을 유도하는 입력을 찾아내는 방식을 제안하였습니다. 이를 위해 텍스트 공간과 임베딩 공간(embedding space) 모두에서 적대적 예제(adversarial examples)를 검색하는 방법을 사용합니다.

- **Performance Highlights**: 실험 결과, 입력을 정확히 재구성하는 것은 드문 일이며, 이는 LLMs가 여전히 비방 공격에 취약하다는 것을 나타냅니다. 대신 약한 역전을 통한 접근 방식이 문제 출력을 감지하는 데 더 유용할 수 있는지 검토되었습니다.



### A Survey on LoRA of Large Language Models (https://arxiv.org/abs/2407.11046)
- **What's New**: 이 논문은 Low-Rank Adaptation(LoRA)에 대한 포괄적인 개요를 제공합니다. LoRA는 대규모 언어 모델(LLM) 레이어에 플러그 가능한 저랭크 행렬을 삽입해 파라미터를 효율적으로 미세 조정하는 방법으로, 최근 많은 관심을 받고 있습니다. 논문은 또한 LoRA의 현재 발전 상황을 다양한 관점에서 분류하고 검토합니다.

- **Technical Details**: LoRA는 대규모 언어 모델의 밀집 신경망 레이어를 업그레이드하면서 파라미터 효율성을 달성합니다. LoRA 플러그인을 사용하면 여러 관련 다운스트림 작업에 재사용할 수 있습니다. 논문은 LoRA 발전을 위해 다음과 같은 방법들을 검토합니다: (1) 다운스트림 적응 성능을 개선하는 변형, (2) LoRA 플러그인을 혼합하여 크로스 작업 일반화를 달성하는 방법, (3) LoRA의 계산 효율성을 향상시키는 방법, (4) 연합 학습에서 데이터 프라이버시를 보호하기 위한 방법.

- **Performance Highlights**: LoRA는 풀 파인 튜닝과 비교해 다양한 다운스트림 작업에서 비슷하거나 더 나은 성과를 보여줍니다. 또한, 컴퓨팅 비용을 줄이고 GaaS(Generative-as-a-Service)와 같은 실제 사용 사례에서의 응용 가능성을 높일 수 있습니다. 프라이버시 보호 측면에서도 연합 학습과 결합할 때 매력적인 이점을 제공합니다.



### Hadamard Adapter: An Extreme Parameter-Efficient Adapter Tuning Method for Pre-trained Language Models (https://arxiv.org/abs/2407.11033)
Comments:
          Accepted to CIKM 2023 (Long Paper)

- **What's New**: 최근 몇 년간, 사전 교육된 언어 모델(Pre-trained Language Models, PLMs)이 다양한 인공지능 분야에서 큰 성공을 거두었습니다. 하지만, T5와 GPT-3와 같은 대부분의 PLM은 방대한 양의 파라미터를 가지고 있어, 이를 미세 조정(fine-tuning)하는 데 많은 비용과 시간이 소요되며, 저장 공간도 많이 차지합니다. 이를 해결하기 위해, 파라미터 수를 줄이면서도 성능을 유지할 수 있는 파라미터 효율적인 접근 방식이 필요합니다. 이번 연구에서는 자가-어텐션(self-attention) 출력에만 작용하는 새로운 어댑터인 Hadamard Adapter를 제안합니다. Hadamard Adapter는 Hadamard 곱을 이용한 요소별 선형 변환을 채택하여, 기존의 파라미터 효율적 어댑터들 중 가장 적은 파라미터를 요구합니다.

- **Technical Details**: Hadamard Adapter는 멀티-헤드 자가-어텐션(multi-head self-attention) 모듈의 출력에 바로 주입되며, 대부분의 파라미터는 고정하고, 어댑터와 그 후속 정규화 모듈만을 학습합니다. 어댑터는 가중치 벡터와 바이어스 벡터로 구성되며, 이 가중치 벡터는 자가-어텐션 출력과 요소별 곱을 통해 새로운 자가-어텐션 출력을 생성합니다. 이를 통해 파라미터 효율성을 극대화하면서도 성능 저하를 최소화합니다.

- **Performance Highlights**: GLUE 벤치마크에 대한 실험 결과, Hadamard Adapter는 전체 미세 조정 대비 0.033%의 파라미터로 경쟁력 있는 성능을 달성하였으며, 기존 어댑터들 대비 가장 적은 파라미터를 요구합니다. 또한, Hadamard Adapter 내에 일부 중복 레이어가 존재함을 찾아내어, 이를 제거한 경우 0.022%의 파라미터로 더 높은 효율성을 달성할 수 있음을 확인했습니다. 이로써 Hadamard Adapter는 역사상 가장 파라미터 효율적인 미세 조정 방법임을 입증했습니다.



### DLO: Dynamic Layer Operation for Efficient Vertical Scaling of LLMs (https://arxiv.org/abs/2407.11030)
- **What's New**: 이 논문에서는 레이어 특징 유사성(Layerwise Feature Similarity)을 기반으로 한 라우팅 정책(Routing Policy)을 이용하여, 레이어를 동적으로 확장, 활성화 또는 건너뛰는 새로운 접근법인 Dynamic Layer Operations(DLO)를 도입했습니다. 전통적인 Mixture-of-Experts(MoE) 방법과 달리, 본 접근법은 모델의 깊이를 타겟으로 하여 레이어 표현에서의 중복성을 줄이고자 합니다. DLO는 Supervised Fine-Tuning(SFT) 단계에서 통합되어 추가적인 Continual Pre-Training(CPT)을 요구하지 않습니다. 실험 결과들은 DLO가 원래 모델보다 뛰어난 성능을 보이며, 효율성을 크게 개선한 상태에서 밀집 확장된 모델과 비교할만한 성과를 내고 있음을 나타냅니다. 구현 및 모델 가중치는 논문 수락 시 공개될 예정입니다.

- **Technical Details**: DLO는 레이어를 동적으로 확장, 활성화 및 건너뛰는 세 가지 주요 작업으로 구성됩니다. 이 프레임워크는 레이어 유사성(Layer Similarity)을 고려하여 필요한 레이어만 활성화하고 불필요한 레이어는 건너뛰는 방식으로 최적의 깊이 확장을 도모합니다. 구체적으로, 레이어 그룹화, 확장, 활성화 및 건너뛰기, 그리고 적응형 FLOP(Floating Point Operations)를 포함하여 효율성을 유지하면서 모델의 일반화를 개선하는 방식으로 설계되었습니다. 모든 모듈은 Supervised Fine-Tuning(SFT) 단계에서 훈련되어 추가적인 Continual Pre-Training(CPT)을 필요로 하지 않습니다. 예를 들면, 그룹화된 레이어는 필요에 따라 추가 레이어로 확장되고, 토큰별로 다른 Sparsity 설정이 적용되어 효율적인 FLOP를 유지합니다.

- **Performance Highlights**: DLO는 철저한 실험을 통해 원래의 비확장 모델을 능가하는 성능을 보였으며, 밀집 확장된 모델들과 비교하여 효율성이 크게 향상된 상태에서 유사한 성과를 달성했습니다. 이는 다양한 NLP 작업, 수학 및 코딩 작업에서도 효과적임을 입증했습니다.



### Efficacy of Various Large Language Models in Generating Smart Contracts (https://arxiv.org/abs/2407.11019)
Comments:
          10 pages

- **What's New**: 이 연구는 이더리움 블록체인(Ethereum Blockchain)에서 불변의 솔리디티 스마트 계약(Solidity smart contracts)을 생성하는 코드 생성 대형 언어 모델(Large Language Models)의 응용을 분석합니다. 대부분의 연구가 일반적인 AI 코드 생성 능력을 평가하는 데 중점을 두었던 반면, 이 논문은 보안과 효율성이 중요한 프로그램인 스마트 계약으로 확장합니다. 연구 결과, LLM은 보안 세부 사항을 엄격히 구현하는 데 어려움을 겪고 있지만, 많은 일반적인 종류의 계약을 성공적으로 생성할 수 있음을 확인했습니다.

- **Technical Details**: 이 논문에서는 새로운 스마트 계약을 생성하는 새로운 프롬프트 전략(prompting strategies)을 발견했음을 보고합니다. 이전 연구와 달리, 이 연구는 LLM이 보안에 중점을 둔 프로그램에서도 어떻게 작동하는지를 조사했습니다.

- **Performance Highlights**: 연구 결과, LLM은 보안 구현에 어려움을 겪을 것이라고 예상했지만, 예상 외로 많은 일반적인 스마트 계약 유형에서 성공적으로 작동했습니다. 이는 AI 모델이 적절한 조건 하에서 스마트 계약을 생성하는 데 큰 잠재력이 있음을 시사합니다.



### Stream State-tying for Sign Language Recognition (https://arxiv.org/abs/2407.10975)
- **What's New**: 이 논문에서는 각 데이터 스트림(data streams)에서 상태 묶기(state tying)에 기반한 새로운 수어 인식(sign language recognition) 접근법을 제안합니다. 이 접근법은 손 제스처 신호를 여섯 개의 동기화된 데이터 스트림으로 표현하며, 이는 각각 왼손/오른손 위치, 왼손/오른손 방향, 왼손/오른손 모양을 의미합니다.

- **Technical Details**: 이 프레임워크는 수어 공간의 매우 정확한 표현을 제공하며, 파라미터 수를 적절히 유지하여 빠른 디코딩을 가능하게 합니다. 실험은 5177개의 중국어 수어에 대해 수행되었으며, 고립된 인식(real-time isolated recognition) 속도는 94.8%입니다. 연속 수어 인식(continuous sign recognition)에서 단어 정확률(word correct rate)은 91.4%에 달합니다.

- **Performance Highlights**: 이 접근법은 수어 공간을 정확하게 표현하면서도 파라미터 수를 줄이는 것이 주요 장점입니다. 이는 빠른 디코딩을 가능하게 하며 실제 실험 결과에서도 높은 실시간 고립 인식률(94.8%)과 연속 수어 인식에서 우수한 단어 정확률(91.4%)을 달성했습니다.



New uploads on arXiv(cs.IR)

### Harnessing Large Language Models for Multimodal Product Bundling (https://arxiv.org/abs/2407.11712)
Comments:
          under review

- **What's New**: 신제품 번들링(Bundle)은 개별 제품을 전략적으로 결합하여 고객에게 제공하는 방식입니다. 최근에는 멀티모달 정보(multimodal information)를 활용한 복잡한 추출기를 사용한 번들링 방법이 주목받고 있으나, 여전히 열악한 의미 이해, 제한된 지식 범위 및 콜드 스타트 문제를 해결하지 못하는 한계가 있습니다. 이에 LLMs(대규모 언어 모델)을 제품 번들링 작업에 적응시키기 위해 Bundle-LLM을 소개합니다. 이 모델은 멀티모달 정보 통합을 위한 하이브리드 아이템 토큰화 방식을 이용하며, 멀티모달 융합 모듈과 트레인 가능한 프로젝터를 통해 비텍스트 특성을 하나의 토큰으로 임베딩합니다.

- **Technical Details**: Bundle-LLM은 텍스트, 시각적, 음향적 및 관계적 데이터를 포함하는 멀티모달 정보를 통합한 하이브리드 아이템 토큰화를 사용합니다. 여기에는 BLIP2와 CLAP와 같은 기초 인코더에서 미디어 특성을 추출하고, LightGCN 같은 사전 훈련된 협업 필터링 방법을 사용해 관계적 특성을 얻습니다. 또한, 비텍스트 특성을 단순하면서도 효과적인 융합 모듈을 통해 하나의 토큰으로 임베딩하며 프롬프트 길이를 단축시킵니다. 제품 번들링 작업을 여러 선택지 중 하나를 고르는 문제로 변환하는 번들 프롬프트 템플릿을 설계하고, LLMs를 프로그레시브 옵티마이제이션(Progressive Optimization) 전략으로 미세 조정해 번들링 성능을 최적화합니다.

- **Performance Highlights**: Bundle-LLM은 두 개의 응용 도메인에서 네 개의 데이터셋을 대상으로 한 실험에서 기존의 다양한 방법들을 능가했습니다. 특히, CLHE와 GPT-4 같은 최신 모델들보다 뛰어난 성능을 보여주었습니다. 다양한 ablation 및 모달리티 연구를 통해 주요 모듈의 효과를 입증하고, 제안한 모델의 다양한 중요한 특성을 보여주었습니다.



### Interactions with Generative Information Retrieval Systems (https://arxiv.org/abs/2407.11605)
Comments:
          Draft of a chapter intended to appear in a forthcoming book on generative information retrieval, co-edited by Chirag Shah and Ryen White

- **What's New**: 정보 접근과 탐색은 본질적으로 상호작용 과정입니다. 기존 검색 엔진에서는 '재검색(requery)', '문서 클릭', '스크롤링', '다음 결과 페이지로 이동', '검색 엔진 나가기' 등 몇 가지 사전 정의된 행동에만 제한됩니다. 그러나 생성적 정보 검색(Generative Information Retrieval, IR) 시스템으로의 전환은 사용자에게 더 풍부한 표현 방식을 제공하며, 자유 형식의 자연어 상호작용 및 그 이상을 가능하게 합니다. 사용자는 클릭 가능한 링크와 버튼에 제한되지 않고 자연어로 자유롭게 표현할 수 있습니다. 또한 이미지, 비디오, 제스처, 센서 등을 활용한 멀티모달 상호작용도 가능합니다.

- **Technical Details**: 이 논문은 생성적 IR 시스템에서 상호작용의 역할을 간략히 논의합니다. 사용자가 생성적 IR 시스템과 상호작용하여 정보 요구를 표현하는 방법, 명시적 또는 암묵적인 피드백을 제공하고 이를 활용하는 방법 등을 설명합니다. 또한 사용자가 검색 결과를 상호작용적으로 정제하는 방법, 혼합 이니셔티브 상호작용, 선호도 추출, 명확화 절차, 상황 인식 추천, 과거 대화의 후속 처리, 피드백 요청 등 다양한 상호작용 유형을 다루고 있습니다.

- **Performance Highlights**: LLMs(대형 언어 모델)을 이용한 생성적 IR 시스템은 자연어 처리를 통해 사용자의 다양한 정보 요구 수준을 이해하고 응답할 수 있습니다. LLMs는 문맥을 유지하며, 다양한 도메인과 수준의 정보 요구를 만족시킬 수 있습니다. 또한 대화형 상호작용을 통해 사용자가 점진적으로 질문을 정제하고, 다양한 형식, 톤, 깊이로 정보를 요청할 수 있는 능력을 제공합니다.



### A PLMs based protein retrieval framework (https://arxiv.org/abs/2407.11548)
Comments:
          16 pages, 12 figures

- **What's New**: 이번 연구에서는 기존 단백질 검색 도구가 갖는 '서열 유사성' 편향 문제를 해결하기 위해 단백질 언어 모델(PLMs)을 활용한 새로운 단백질 검색 프레임워크를 제안합니다. 이 프레임워크는 고차원 특징 공간에서 단백질 서열을 임베딩하여 보다 정교한 단백질 표현을 제공하며, 기존 방법으로 놓치기 쉬운 기능적 유사성을 가진 단백질도 검색할 수 있습니다.

- **Technical Details**: 제안된 프레임워크는 PLM을 수정하여 단백질 서열의 고차원 임베딩을 추출하고, 그다음 추론 섹션에서 반정밀 추론 및 가지치기 기법을 통합하여 대규모 데이터베이스의 추론 효율성을 향상시킵니다. 또한, 밀집 벡터의 빠른 접근 및 검색을 위해 VPTree와 FAISS를 사용하며, 국소 민감 해시(LSH)를 사용하여 임베딩의 차원 축소 및 클러스터링을 처리하여 분산 저장 및 병렬 계산을 촉진합니다.

- **Performance Highlights**: 다양한 실험 결과, 제안된 프레임워크는 기존의 서열 유사성 기반 방법들이 간과할 수 있는 단백질도 효과적으로 검색할 수 있음을 보여줍니다. 또한, 이 프레임워크는 기존 방법들에 비해 정확도와 안정성에서 크게 성능 저하 없이 단백질 검색의 지역 최적화 함정을 벗어날 수 있습니다.



### Bootstrapped Pre-training with Dynamic Identifier Prediction for Generative Retrieva (https://arxiv.org/abs/2407.11504)
Comments:
          Accepted by ACL Findings 2024

- **What's New**: BootRet이라는 새로운 부트스트랩 기반의 사전 학습 방식을 도입하여, 생성적 검색(Generative Retrieval, GR)에서 문서 식별자(docid)를 동적으로 조정하는 방법을 제안합니다. 이는 기존의 정적인 문서 식별자 대신, 모델의 매개변수 변화에 맞추어 문서 식별자를 조정하여 검색 성능을 극대화합니다.

- **Technical Details**: BootRet은 세 가지 주요 훈련 단계를 포함합니다: (i) 초기 식별자 생성, (ii) 코퍼스 색인화 및 관련성 예측 작업을 통한 사전 학습, (iii) 식별자 갱신을 위한 부트스트랩. 또한, 대규모 언어 모델을 사용해 생성한 노이즈 문서와 가상 쿼리를 활용하여 색인화와 검색 작업의 시맨틱 연결을 모방합니다.

- **Performance Highlights**: 실험 결과, BootRet은 기존 사전 학습 생성적 검색 기준선을 크게 능가했으며, 제로샷(Zero-shot) 설정에서도 뛰어난 성능을 보였습니다. MS MARCO 데이터셋에서는 Hits@1 기준으로 Ultron 모델보다 11.8% 더 우수한 성능을 기록했습니다.



### Pacer and Runner: Cooperative Learning Framework between Single- and Cross-Domain Sequential Recommendation (https://arxiv.org/abs/2407.11245)
Comments:
          Accepted at SIGIR'24

- **What's New**: 이번 연구에서는 단일 도메인 순차 추천(SDSR)과 비교하여 크로스 도메인 순차 추천(CDSR)의 성능 향상을 목표로 새로운 모델 SyNCRec을 제안합니다. SyNCRec은 도메인 간 부정적 전이(negative transfer) 문제를 해결하여 적응적으로 가중치를 조정함으로써 추천 성능을 향상시킵니다.

- **Technical Details**: SyNCRec 모델은 각 도메인의 부정적 전이 정도를 추정하여 예측 손실(predictive loss)에 가중치로 반영하여 조정합니다. 이를 위해, 도메인 혼합 시퀀스(CDSR)로 훈련된 모델과 단일 도메인 시퀀스(SDSR)로 훈련된 모델 간의 성능을 비교합니다. 더불어, SDSR과 CDSR 작업 간의 상호 정보를 극대화하는 보조 손실(auxiliary loss)를 개발하여 유용한 단서를 전달합니다.

- **Performance Highlights**: 총 10개의 서비스 도메인에서 SyNCRec은 기존 여러 연구들을 능가하는 성능을 보였으며, 실제 산업 데이터셋을 사용한 실험에서는 클릭률이 21.4% 증가했습니다. 이는 실제 비즈니스에 상당한 가치를 제공합니다.



### A Comprehensive Evaluation of Large Language Models on Temporal Event Forecasting (https://arxiv.org/abs/2407.11638)
- **What's New**: 최근의 대형 언어 모델(LLMs)은 지식 질문 응답, 수학적 추론, 상식 추론 등 다양한 데이터 마이닝 작업에서 큰 잠재력을 보여주고 있습니다. 하지만 LLMs의 시간적 이벤트 예측 능력은 아직 충분히 탐구되지 않았습니다. 이를 체계적으로 조사하기 위해, 시간적 이벤트 예측을 위한 LLM 기반 방법들의 종합적인 평가를 수행하였습니다. 이를 위해 그래프 및 텍스트 데이터 모두를 포함하는 고품질 데이터 세트가 부족한 문제를 해결하기 위해, 먼저 MidEast-TE-mini라는 벤치마크 데이터셋을 구축했습니다.

- **Technical Details**: 이 데이터셋을 기반으로, 다양한 입력 형식과 Retrieval Augmented Generation (RAG) 모듈로 특징지어진 일련의 베이스라인 방법을 설계했습니다. 광범위한 실험을 통해 발견한 주요 사항은, 원본 텍스트를 LLM의 입력에 직접 통합하는 것은 제로샷 외삽 성능을 향상시키지 않는다는 점입니다. 반면, 특정 복합 이벤트에 원본 텍스트를 통합하고 LLM을 미세 조정하면 성능이 크게 향상됩니다. 또한 검색 모듈이 강화된 LLM은 역사적 이벤트에 숨겨진 시간적 관계 패턴을 효과적으로 포착할 수 있습니다.

- **Performance Highlights**: 이번 연구 결과는 LLM 기반 이벤트 예측 방법에 대한 이해를 깊게 할 뿐만 아니라, LLM을 통한 시간적 이벤트 예측의 향후 연구에 크게 기여할 여러 유망한 연구 방향을 강조합니다. 특히, 인기 편향과 긴 꼬리 문제와 같은 이슈가 여전히 존재하지만(RAG 기반 방법에서 특히 두드러짐), 이러한 문제를 해결하기 위한 새로운 접근법을 제안할 수 있습니다.



### EndoFinder: Online Image Retrieval for Explainable Colorectal Polyp Diagnosis (https://arxiv.org/abs/2407.11401)
Comments:
          MICCAI 2024

- **What's New**: EndoFinder는 대장 내시경 검사 중 발견된 폴립(polyp)을 설명 가능한 진단을 함께 제공하는 내용 기반 이미지 검색(content-based image retrieval) 프레임워크입니다. 이 시스템은 새로운 폴립과 매칭되는 '디지털 트윈(digital twin)' 폴립을 참조 데이터베이스에서 찾아냅니다. 이렇게 매칭된 폴립을 통해 새롭게 발견된 폴립의 임상적 의미를 추론할 수 있습니다. EndoFinder는 특히 설명가능성과 실시간 진단 지원에서 혁신적 접근을 제안합니다.

- **Technical Details**: EndoFinder는 대규모 폴립 데이터셋을 기반으로 자가 감독 학습(self-supervised learning)을 통해 사전 학습된 폴립 인식 이미지 인코더를 사용합니다. 이 인코더는 이미지 검색을 위한 보편적인 임베딩 공간(embedding space)을 생성합니다. 폴립 재식별 및 광학 생검(optical biopsy) 작업에서 EndoFinder를 검증했으며, masked image modeling과 contrastive learning을 결합하여 보편적인 폴립 인식 임베딩을 만들어냅니다. 이 임베딩을 통해 다양한 임상적 하위 작업을 지원할 수 있습니다.

- **Performance Highlights**: EndoFinder는 설명 기반 진단에서 감독 학습 기반 분류 모델과 동등한 수준의 성능을 달성합니다. 또한, 이미지 검색을 기반으로 한 EndoFinder는 실시간 내시경 검사 중 다양한 후속 의사결정 작업을 지원할 수 있는 잠재력을 가지고 있습니다.




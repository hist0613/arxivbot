New uploads on arXiv(cs.CL)

### How reliable are LLMs when it comes to playing dice? (https://arxiv.org/abs/2606.07515)
- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)의 확률적 추론 능력을 통제된 벤치마킹 연구를 통해 조사했습니다. 두 개의 데이터셋을 만들었으며 각각 일반적인 문제와 직관에 반하는 문제로 구성되어, 휴리스틱 추론(heuristic reasoning)을 유도하도록 설계되었습니다. 8개의 최신 모델을 평가한 결과, 일반 문제에는 평균 0.96의 정확도를 달성했으나, 직관에 반하는 문제에서는 0.59로 낮아지는 성과를 보였습니다.

- **Technical Details**: 연구에서는 표준 문제와 직관에 반하는 문제에서 LLM의 성능을 측정하고, 모델 간의 행동 차이를 분석했습니다. 특히, 모델의 입력 패턴에 따라 발생하는 토큰 편향(token bias)의 영향을 고려하였으며, 잘못된 제안이 모델의 응답에 미치는 영향을 실험적으로 평가했습니다. 이를 통해 모델들이 코너케이스나 잘못된 추론의 영향을 최소화하지 못하고 있다는 것을 확인할 수 있었습니다.

- **Performance Highlights**: 최신 모델들이 고차원 수학 문제를 해결하는 데 뛰어난 성과를 보였지만, 직관에 반하는 문제에서는 성과가 크게 저하되었습니다. 이로 인해 특정 확률적 추론의 측면이 기존의 수학적 벤치마크에서는 포착되지 않음을 알 수 있었습니다. 연구 결과, 현재의 LLM은 진정한 추론 기계가 아닐 뿐만 아니라, 교육 데이터로부터 물려받은 결점이 있으며, 입력 프롬프트에 매우 민감한 특성을 보여주었습니다.



### Agentopia: Long-Term Life Simulation and Learning in Agent Societies (https://arxiv.org/abs/2606.07513)
Comments:
          79 pages, 19 figures

- **What's New**: 이 논문에서는 Agentopia라는 새로운 프레임워크를 제안하여, 다수의 에이전트가 10년 동안 인간 사회를 모방한 삶의 시뮬레이션을 수행하도록 합니다. 이는 기존의 짧은 시간 시뮬레이션 한계를 넘어서 에이전트가 사회적 상호작용에서 학습하고 성장할 수 있도록 합니다. Agentopia는 인간의 안녕을 반영하는 라이프 리워드를 정의하고, 이를 통해 대형 언어 모델(LLMs)을 최적화할 수 있는 방법을 제시합니다.

- **Technical Details**: Agentopia는 에이전트가 자율적으로 성장하고 사회적 관계를 발전시키도록 설계되었습니다. 시뮬레이션은 여러 단계를 포함하여 에이전트가 계획을 세우고, 다른 사람과 소통하며, 활동을 수행하고 경험을 검토하는 형태로 진행됩니다. 환경 모델과 맥락 관리 메커니즘을 통해 에이전트는 지속적인 상호작용을 통해 인간과 유사한 행동을 학습할 수 있는 능력을 갖추게 됩니다.

- **Performance Highlights**: 실험 결과, Agentopia에서 학습한 에이전트는 더욱 풍부한 사회적 행동을 나타내며, 라이프 리워드 훈련을 통해 LLM의 성능이 향상되었습니다. 특히, 사회적 관계, 주관적 성취 및 경제적 이익에서 개선된 결과를 보였으며, CoSER Test에서 +15.6%의 성능 향상이 관찰되었습니다. 이는 에이전트의 복지 수준과 역할 수행 능력 모두를 높일 수 있음을 의미합니다.



### Your UnEmbedding Matrix is Secretly a Feature Lens for Text Embeddings (https://arxiv.org/abs/2606.07502)
Comments:
          preprint

- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)이 텍스트 임베딩 태스크에서 성능 저하를 겪는 원인을 분석하고, 이를 해결하기 위한 EmbedFilter라는 새로운 방법을 제안합니다. 연구자들은 LLM의 텍스트 임베딩이 자주 등장하지만 의미가 없는 토큰과 정렬된다는 흥미로운 관찰에서 출발하여, 높은 빈도 수의 토큰이 세밀한 의미 캡처를 저해한다는 점을 지적합니다.

- **Technical Details**: EmbedFilter는 LLM의 임베딩 품질을 향상시키기 위해 설계된 간단한 선형 변환으로, LLM이 생성하는 임베딩의 잠재 공간을 정제합니다. 이를 통해, 자주 등장하는 토큰의 영향을 제어하고 진정한 의미 표현을 발현할 수 있습니다. 해당 변환은 사전 학습된 매개변수 내에 존재하므로 추가적인 학습 없이도 쉽게 적용이 가능합니다.

- **Performance Highlights**: 많은 실험을 통해 EmbedFilter가 기존의 임베딩 차원을 크게 줄이면서도 LLM의 제로샷(formally known as 'zero-shot') 성능을 비약적으로 향상시킨다는 것을 입증했습니다. EmbedFilter는 다양한 모델과 실험 구성에서 강력한 성능을 보여주며, 이로 인해 LLM을 대규모 텍스트 임베딩 응용에서 효과적으로 사용할 수 있는 가능성을 높입니다.



### Supervision versus Demonstration-Based In-Context Learning for Multiword Expression Classification (https://arxiv.org/abs/2606.07479)
Comments:
          Accepted to ACL SRW 2026

- **What's New**: 이번 연구는 터키어의 관용적 경량 서술 동사 구성(light verb constructions, LVCs)을 탐구하며, 이들이 특정한 표면 형태를 공유하지만 단일한 부분적 관용(predicate)으로 작용하는 것에 주목합니다. 연구팀은 LVC 감지를 이진 분류 작업으로 틀을 잡고, 이를 통해 감독된 BERTurk 언어 인코더와 다양한 인스트럭션 튜닝된 LLM(대형 언어 모델) 성능을 비교했습니다. 결과는 자세한 프롬프트 민감성을 보여주며, 주의 깊게 구성된 프롬프트가 LVC 감지를 개선하는 데 중요한 역할을 한다는 것을 증명했습니다.

- **Technical Details**: 본 논문은 터키어 LVC 감지를 명시적으로 다음의 세 가지 프롬프트 방식(제로샷, 원샷, 퓨샷)으로 평가하여, 각 모델의 성능을 비교했습니다. 초기 결과에 따르면, 제로샷에서 LLM들은 부정적인 예시에서 잘 작동하지만 LVC 감지는 낮은 재현율을 보이며, 원샷 프롬프트에서는 LVC 감지를 급격히 개선시키는 반면 강한 모델 특정 편향성을 유도할 수 있음을 알았습니다. 최종적으로 퓨샷 프롬프트가 보다 안정적인 성능을 보여주며, 특히 GPT-OSS-20B와 Qwen 2.5-14B 모델에서 경쟁력이 있음을 보여주었습니다.

- **Performance Highlights**: 연구 결과는 LVC 감지에서의 프롬프트 민감성이 도드라지며, 감독된 BERTurk 기준과 비교해 인스트럭션 튜닝된 LLM이 동등하거나 그 이상의 성능을 보여줄 수 있다는 가능성을 제시합니다. 이 모델들은 특정한 예시를 제공할 때 LVC 감지에서 우수한 성과를 나타내지만, 잘못된 예측 또한 발생할 수 있음을 시사합니다. 따라서 본 연구는 터키어에서 다중 단어 표현 처리의 복잡성 및 모델링에서 유의미한 최소한의 입력이 얼마나 중요한지를 강조합니다.



### Sycophantic Praise: Evaluating Excessive Praise in Language Models (https://arxiv.org/abs/2606.07441)
- **What's New**: 이번 연구는 언어 모델에서의 아첨이 과도한 동의나 확인으로 주로 연구되었지만, 공개적으로 칭찬하거나 아첨하는 것에 대한 연구는 상대적으로 부족하다는 점을 주장합니다. 아첨하는 칭찬은 현재의 방법으로 신뢰성 있게 측정할 수 없는 별개의 정렬 문제로 제시됩니다. 이러한 문제를 해결하기 위해 SyPr라는 매개변수화된 프레임워크를 도입하여, 칭찬의 과도함을 기여의 질 및 예상되는 사용자 능력에 비례하여 측정합니다.

- **Technical Details**: 연구는 심리학 연구에 기반하여 칭찬의 맥락에 주목한 평가 프레임워크인 SyPr를 개발했습니다. 이 프레임워크는 상호작용을 정의하는 페르소나, 품질 예측이 포함된 사용자 발화, 그리고 사용자에 대한 칭찬을 포함할 수 있는 모델의 응답으로 모델링됩니다. 또한, 이 연구에서는 다양한 최신 언어 모델에서 아첨하는 칭찬이 일반적인 행위라는 점을 밝혔으며, 평가된 시스템에서 아첨하는 칭찬이 자주 발생하는 경향을 보였습니다.

- **Performance Highlights**: SyPr 프레임워크를 사용한 결과, 아첨하는 칭찬이 GPT-5.4, Claude Sonnet 4.6 등에서 15.1%에서 32.3%까지 널리 나타나는 것으로 분석되었습니다. 아울러 이 프레임워크의 평가는 인간 주석과의 일치에서 0.919 AUROC를 달성하며, 전통적인 방법과 비교해 상당한 개선을 보여주었습니다. 마지막으로, 연구는 목표 추론에 비해 사회적 해석 설정에서 아첨하는 칭찬이 강하게 비대칭적으로 발생한다는 점도 강조합니다.



### The Masked Advantage: Uncovering Local-Language Access to Cultural Knowledge in LLMs (https://arxiv.org/abs/2606.07422)
- **What's New**: 이번 연구는 대규모 언어 모델(LLM)의 다언어 시스템으로서의 발전을 탐구하며, 지역 문화 지식(access to local cultural knowledge)이 영어 쿼리 혹은 현지 언어 중 어떤 언어를 통해 더 잘 접근되는지를 비교합니다. 기존 연구는 병렬 템플릿 기반 질문에 의존해 왔으나, 이 연구는 실제 지역 벤치마크 및 지역 자료에서 수집한 실질적인 문화 질문을 기반으로 한 제어된 프레임워크를 제시합니다. 언어 능력(proficiency)과 문화 지식 접근(access)을 분리하는 새로운 측정을 위한 모델을 제안합니다.

- **Technical Details**: 연구진은 문화 특이적 질문(culture-specific questions)과 문화 비특이적 질문(culture-agnostic questions)으로 질문 유형을 나누어 영어와 지역 언어에서의 모델 성능을 비교했습니다. 이를 통해 원시 정확도(raw accuracy) 대신 1PL 항목 응답 이론 모델(Item Response Theory model)을 사용하여 언어 능력과 문화 지식 접근을 분리할 수 있었습니다. 총 13개 지역에서 약 80개의 모델을 사용하여 분석을 수행하며, 이 과정에서 문화 지식의 접근성이 다양한 쿼리 언어에 따라 어떻게 달라지는지를 조사합니다.

- **Performance Highlights**: 모델들은 문화 비특이적 질문에서 일관된 영어 능력을 나타냈으며, 이는 영어가 높은 수준의 언어 능력(presumably due to higher proficiency)으로 인해 발생했습니다. 그러나 지역 언어에 대한 접근 방법은 문화 특이적 질문에서 긍정적인 경향을 보였으며, 이는 제한된 언어 능력으로 인해 가려질 수 있습니다. 결론적으로, 지역 언어의 성과는 문화 지식을 저해하는 영어 능력과 비교했을 때 더 나은 접근성을 제공하는 것으로 나타났습니다.



### M$^3$Exam: Benchmarking Multimodal Memory for Realistic User-Agent Interactions (https://arxiv.org/abs/2606.07402)
- **What's New**: 이번 논문에서는 M$^3$Exam이라는 새로운 multimodal 대화 메모리 벤치마크를 도입하였으며, 이는 실제 사용자-에이전트 상호작용을 기반으로 하여 사용자 쿼리를 중심으로 구성되고 있습니다. 기존 벤치마크들이 인간 대 인간의 단순한 시나리오에 집중했던 것에 반해, M$^3$Exam은 복잡한 cross-modal grounding 및 암묵 정보 추론을 평가합니다. 또한, M$^3$Proctor라는 새로운 방법론을 제안하며, 이를 통해 쿼리의 모달리티 편향을 탐지하고, 필요할 때만 시각 자료를 사용하여 효율성을 개선하고 있습니다.

- **Technical Details**: M$^3$Exam은 239,239개의 다중 세션 대화를 1,515개의 페르소나 시나리오에 걸쳐 구성하고 있으며, 5,150개의 평가 질문이 포함되어 있습니다. 각 쿼리는 에이전트가 누적된 multimodal 기록을 분석하고 관련 대화 세션이나 시각 자료를 검색하여 답변해야 함을 의미합니다. 이 벤치마크는 인과 관계가 중시되는 질문 은행을 포함하고 있어, 도메인 지식을 요구하는 Thematic Reasoning(TH) 및 역사적으로 암시된 정보를 이용해야 하는 Implicit Inference(II)와 같은 질문 형식을 제안합니다.

- **Performance Highlights**: M$^3$Exam에 대한 평가 결과, 최신 Multimodal Large Language Models (MLLMs)와 에이전트 메모리 시스템은 cross-modal reasoning 및 암묵적 해석에서 어려움을 겪고 있음이 드러났습니다. M$^3$Proctor는 이러한 문제를 해결하기 위해 모달리티 인식 메모리 방법론으로 설계되었으며, 쿼리의 모달리티 편향을 감지하고 원시 비주얼 소스를 필요할 때만 사용하여 정확도를 13% 향상시키고, 70%의 토큰 사용량 감소 및 색인 구성 시간을 80% 단축하였습니다.



### LLM-Guided Evolution for Medical Decision Pipelines (https://arxiv.org/abs/2606.07342)
- **What's New**: 이 논문은 의료 결정 전략 발견을 위한 추론 시간 대안으로서 LLM-guided MAP-Elites 진화를 연구합니다. 연구자들은 긴급 분류, 인터랙티브 상담, 의료 이미지 분류를 각기 다른 임상 추론 모드로 설정하여 이 연구를 수행했습니다. 이들은 진화 과정이 수동적으로 설계된 기준보다 개선된 결과를 보여준다는 점을 강조했습니다.

- **Technical Details**: 이 연구에서 사용된 GigaEvo는 MAP-Elites 검색, 비동기 평가 및 코드 재작성 기반 변이를 지원하는 오픈 소스 프레임워크입니다. LLM 변형자 역할을 하는 모델 gpt-oss-120b는 강력한 코딩 및 구조적 출력 기능을 갖춘 것으로 보고되었습니다. 이 프레임워크를 통해 LLM은 실제 의료 결정 파이프라인을 미세 조정하지 않고도 최적화할 수 있습니다.

- **Performance Highlights**: 진화된 프로그램은 Semigran 정확도를 77.3%에서 87.1%로, 응급 재현율을 0.60에서 0.97로 증가시키며, MIMIC-ESI 성능을 향상시켰습니다. 인터랙티브 상담에서 진화된 정책은 Llama-3, Qwen-3.5 및 Gemma-4에서 정확성-비용 경계를 개선했습니다. 이 연구의 결과는 명확한 프로그램 수준 메커니즘과 위험 민감한 분류를 통해 이루어진 것입니다.



### SV-Detect: AI-generated Text Detection with Steering Vectors (https://arxiv.org/abs/2606.07313)
- **What's New**: 이 논문에서는 SV-Detect라는 새로운 기법을 제안하는데, 이는 고정된 언어 모델의 숨겨진 표현에서 추출된 steering vectors를 기반으로 한 가짜 텍스트 탐지기입니다. 이 방법은 인간이 쓴 텍스트와 기계가 생성한 텍스트를 구분하기 위한 방향을 각 층에서 구축하며, 이러한 방향들과 입력 텍스트 간의 정렬을 통해 텍스트를 표현합니다. SV-Detect는 강력한 성능을 보여주며, 특히 도메인 간 전이(distribution shift)에도 효과적입니다.

- **Technical Details**: SV-Detect의 방법론은 고정된 변환기(transformer) 언어 모델의 층별 활성화를 추출하는 것으로 시작합니다. 그런 다음, 인간 작성 텍스트와 기계 생성 텍스트를 구별하는 steering vector(조향 벡터)를 추출하며, 이는 낮은 차원의 특징으로 텍스트 표현을 투영하는 데 사용됩니다. 최종적으로, 이러한 투영 특징을 바탕으로 경량(classifier) 분류기를 학습하여 최종 탐지 점수를 생성합니다.

- **Performance Highlights**: SV-Detect는 DetectRL과 MIRAGE라는 두 개의 보완적인 벤치마크에서 평가되었으며, 두 설정 모두에서 뛰어난 성능을 나타냈습니다. 특히, SV-Detect는 도메인 간 전이와 기계 편집 변환에서도 강력한 성능을 발휘하는 것으로 나타났습니다. 논문에서는 SV-Detect가 인간 텍스트와 기계 생성 텍스트 간의 스타일적 신호를 포착하고, 표현 레벨에서의 차이를 연구하는 데 유용한 도구가 될 수 있음을 강조합니다.



### Phun-Bench: Evaluating LLMs on Phonological Understanding in Chines (https://arxiv.org/abs/2606.07300)
Comments:
          Accepted to ACL 2026 Main Conference

- **What's New**: 이 논문에서는 기존의 대형 언어 모델(LLM) 연구가 의미(semantics)와 기호(spelling)에 집중되는 반면 소리(sounds)는 간과되고 있음을 지적합니다. 이를 해결하기 위해 Phun-Bench라는 새로운 중국어 벤치마크를 제시하며, 다양한 과제와 설정을 통해 LLM의 음운학적(phonological) 이해를 체계적으로 평가합니다.

- **Technical Details**: Phun-Bench는 동음이의어(homophony), 운율(rhyme), 음성 유사성(phonetic similarity)이라는 세 가지 차원에서 다양한 작업을 포함합니다. 이러한 설계는 LLM이 음운적 이해를 어떻게 수행하는지를 평가하는 데 중점을 두고 있습니다. 기존의 벤치마크들은 대부분 암기로 해결할 수 있거나 다른 능력과 얽혀 있어 LLM의 진정한 음운학적 능력을 측정하기에 부적절했습니다.

- **Performance Highlights**: 선언된 결과에 따르면 LLM은 올바른 발음을 회상하는 데는 뛰어나지만, 인간 화자가 사용하는 직관적이고 유연한 방식으로 음운적 지식을 활용하는 데 어려움을 겪고 있습니다. 이러한 분석을 통해 LLM의 음운학적 이해와 '지각(perception)'의 기저 메커니즘에 대한 가설을 제안하며, 이는 향후 연구에서 탐구해야 할 미개척 영역으로 강조됩니다.



### KIT's Submission to Cross-Lingual Voice Cloning in IWSLT 2026 (https://arxiv.org/abs/2606.07240)
- **What's New**: 이 논문은 IWSLT 2026 Cross-Lingual Voice Cloning 트랙에 대한 Karlsruhe Institute of Technology의 기고작을 다룹니다. 주된 목표는 사용자가 명확하게 요구한 언어 태그를 통해 음성 합성 시스템의 언어 제어를 개선하고 악센트 유출 문제를 줄이는 것입니다. 연구에서는 FishAudio-S2-Pro라는 다국어 텍스트-음성 변환(TTS) 모델을 기반으로 하고, 강화 학습(Reinforcement Learning) 기법을 통해 작업에 적응하고 이해 가능성을 향상시키는 방법을 제안합니다.

- **Technical Details**: 이 시스템은 ACL 60/60 데이터셋을 사용하여 평가되며, 여기에는 영어 참조 음성이 프랑스어, 아랍어 및 중국어 텍스트와 짝지어져 있습니다. 모델은 언어 태그를 사용하여 텍스트 성분을 제어하며, 받아쓰기 시스템(ASR)을 통해 긴 음성 파일을 작은 단위로 분할하여 참조 음성에서 도메인 특화 단어의 발음을 개선하는 방법을 사용합니다. 또한, 그룹 상대 정책 최적화(Group Relative Policy Optimization) 기법을 적용하여 모델을 세밀하게 적응시킵니다.

- **Performance Highlights**: 실험 결과는 언어 태그 사용이 가장 큰 성과를 나타내며, 레퍼런스 조건 부여된 어휘 매칭 방법이 도메인 특화 단어의 발음 정확도를 일관되게 향상시키는 것으로 나타났습니다. 전체적으로, 제안된 모델은 다양한 언어에서의 크로스-링구얼 발음을 개선하면서도 원래 화자의 정체성을 유지하는 데 성공했습니다.



### When Large Language Models Fail in Healthcare: Evaluating Sensitivity to Prompt Variations (https://arxiv.org/abs/2606.07237)
Comments:
          12 pages

- **What's New**: 이 논문에서는 의료 분야에서의 대형 언어 모델(LLM)의 감도(sensitivity)에 대한 체계적인 분석을 통해, 일반 목적 모델(예: GPT-3.5, Llama3)과 의료 특화 모델(예: ClinicalBERT, BioLlama3, BioBERT)의 강건성을 평가합니다. 특히, 프롬프트의 언어적(lexical) 및 구문적(syntactic) 변동이 임상적 질문 응답 및 진단 지원과 같은 임상 작업에서 모델의 성능에 미치는 영향을 조사합니다. 연구 결과, 작은 문구의 변화조차 모델의 출력을 크게 바꿀 수 있음을 보여 주며, 이는 의료 환경에서는 치명적인 위험으로 인식됩니다.

- **Technical Details**: 연구에서는 프롬프트의 변동을 자연적인 유형과 적대적(adversarial) 유형으로 분류하고, 두 유형이 모델의 일관성(consistency), 정확성(accuracy) 및 신뢰성(reliability)에 미치는 영향을 분석합니다. 특히, 의료 LLM들은 단순한 어휘의 대체(substitution)에는 어느 정도 견딜 수 있으나, 구문 정렬(syntactic reordering)이나 오해를 일으킬 수 있는 문맥적 단서(contextual cues)에는 취약한 모습을 보입니다. 논문은 LLM의 안정성을 평가하기 위한 표를 제공합니다.

- **Performance Highlights**: 주요 성과로는 간단한 문구 변경이 올바른 진단에 부정적인 영향을 미칠 수 있음을 보여주는 여러 사례를 통해, 의료 AI 시스템에 대한 신뢰성을 구축하기 위해 해결해야 할 주요 도전 과제가 있음을 강조합니다. 특히, 적대적 조작(adversarial manipulations)은 부적절한 약물 권장이나 필수 발견 사항을 간과하는 등의 심각한 결과를 초래할 수 있습니다. 모델의 신뢰성을 강화하기 위해서는 이러한 취약성과 한계를 인식하고, 보다 강건하고 신뢰할 수 있는 의료 AI 시스템을 개발하는 방법을 모색해야 합니다.



### Adversarial Creation and Detection of AI-Generated Social Bot Conten (https://arxiv.org/abs/2606.07219)
- **What's New**: 이번 연구는 큰 언어 모델(LLMs)과 소셜 봇이 결합하여 가짜 정보를 대규모로 생성하는 방법을 다룹니다. 기존의 AI 생성 콘텐츠 감지 모델이 실제 환경에서 실패하는 이유는 데이터의 불충분함에 있습니다. 저자들은 이러한 격차를 해결하기 위해 실제 소셜 미디어 사용자의 행위를 모방하는 적대적 방법론을 제안합니다. 이를 통해 다양한 언어와 플랫폼에서 인간과 AI 생성 메시지를 짝지은 데이터셋을 구축하고, 이를 활용하여 정확한 AI 생성 텍스트 탐지 모델을 훈련합니다.

- **Technical Details**: 저자들은 다양한 언어와 플랫폼을 포함하는 멀티링구얼 데이터셋을 구축하기 위해, 소셜 미디어 대화에서 실제 데이터를 수집하고 이를 AI 생성 메시지로 확장하는 방법론을 사용합니다. 이 데이터 생성 파이프라인은 사용자의 프로필과 과거 메시지 행동에 기초하여 실제 사용자를 모방하며, 이는 텍스트 분류 문제로 단순히 처리되지 않고 특정 플랫폼의 특성과 작성자의 정체성, 상호작용 및 고유한 작성 스타일을 포함한 맥락적인 차원을 포착합니다. 이렇게 구성된 데이터셋은 기존 탐지 모델에 도전할 수 있는 서사형 메타데이터로 보강됩니다.

- **Performance Highlights**: 새롭게 제안된 탐지 모델은 실제 세계의 분포와 일치하지 않는 데이터에서 AI 봇 감지에 있어 기존 모델들보다 상당히 뛰어난 성능을 보입니다. 저자들은 이 모델의 효과성을 실험 결과를 통해 입증하였으며, 고세밀도의 적대적 데이터에 대한 학습이 AI 기반 소셜 봇 감지의 정확도를 크게 높이는 것을 보여주고 있습니다. 이 연구에서는 소셜 미디어에서 악의적인 활동을 방지하기 위해 AI 생성 콘텐츠 탐지가 매우 중요하다고 강조하였습니다.



### From Correctness to Utility: Gain-Based Prefix Evaluation for LLM Reasoning (https://arxiv.org/abs/2606.07190)
- **What's New**: 이 논문에서 제시된 새로운 아이디어는 reasoning prefix가 LLM(Large Language Model) 문제 해결의 미래 궤적에 미치는 영향을 다루고 있다는 점입니다. 기존의 프로세스 리워드 모델(process reward models)은 주로 지역적 단계의 정확성(local step correctness)을 통해 이러한 prefix를 평가하지만, 저자들은 성공적인 완료(successful completion) 확률의 증대 여부가 더욱 중요한 척도라는 주장을 합니다.

- **Technical Details**: 저자들은 prefix gain으로 불리는 개념을 정의하고, 이는 경량 학생 모델 그룹(lightweight student model group)을 prefix로 조건화(conditioning)할 때 유도된 해결률 개선(solve-rate improvement)을 나타냅니다. 이를 통해 Prefix Utility Model (PUM)을 훈련하며, 간단한 쌍비교 랭킹(pairwise ranking) 목표를 사용하여 결과 기반(prefix utility) 학습을 수행합니다.

- **Performance Highlights**: PUM은 완전한 궤적(complete trajectories)과 부분 reasoning prefix 모두를 평가할 수 있어, Best-of-$N$ 선택, beam search 및 수학적 추론을 포함한 강화 학습 환경에서 강력한 prefix 수준의 감독 신호(supervision signal)를 제공합니다. 특히 후보 풀이 크고(search budgets increase) 보상(reward)이 희박한 상황에서 뛰어난 성능을 발휘하는 것으로 나타났습니다.



### Geometry of Semantic Space: Comparative Study of Discrete and Continuous Models (https://arxiv.org/abs/2606.07183)
Comments:
          9 pages, 7 figures

- **What's New**: 이 연구는 NLP(자연어 처리) 모델의 의미 기하학을 검토하며, CamemBERT와 같은 감독된 벡터 임베딩(supervised vector embeddings)과 의미 관계를 보다 직접적으로 인코딩하는 어휘 공동 발생 그래프(lexical co-occurrence graphs)를 비교합니다. 연구 결과는 두 가지 접근 방식에서 유도된 기하학이 자주 불만족스러운 분포를 보이는 반면, 그래프 기반 모델은 의미의 더욱 명확하고 인간이 이해하기 쉬운 조직을 드러난다는 점에서 신선합니다. 이 연구는 그래프 구조를 통해 신경망 아키텍처가 더 안정적이고 해석 가능한 수렴으로 나아갈 수 있는 새로운 경로를 제안합니다.

- **Technical Details**: 이 연구에서, 연구자들은 동일한 텍스트 기반의 분석에 따라 벡터 모델과 그래프 모델을 구축하여 Grand Débat National 코퍼스를 사용하였습니다. 분석은 단어의 형태 변화를 중립화하고 언어학적 일관성을 보장하기 위해 Simplemma를 사용하여 레마타이즈(lemmatization) 단계를 포함합니다. 또한, 공동 발생 클리크(co-occurrence cliques)를 언어 단위로 정의하여 맥락적 다의성을 정확히 포착하고 그래프 구조를 구축하기 위한 기반으로써 PMI(Pointwise Mutual Information)를 사용하였습니다.

- **Performance Highlights**: 비교 결과, 벡터 기반 모델과 그래프 기반 모델 간의 기하학적 유사점은 있지만 전체적인 구조와 토폴로지(topology)는 매우 다르다는 것을 보여줍니다. 특히, Infomap 방식과 같은 클러스터링을 통해 의미의 지역을 더욱 강조할 수 있으며, 이는 각 모델의 품질 기준으로 판단하는 데 중요한 역할을 합니다. 이러한 분석은 NLP 모델의 성능을 기하학적 일관성에 따라 평가할 수 있다는 점에서 의미가 큽니다.



### UrduMMLU: A Massive Multitask Benchmark for Urdu Language Understanding (https://arxiv.org/abs/2606.07167)
Comments:
          27 pages, 18 figures, 17 tables, Submitted to ARR May 2026

- **What's New**: 우리는 UrduMMLU라는 새로운 벤치마크를 소개합니다. 이 벤치마크는 26개 과목과 5개 도메인에서 총 26,431개의 우르두어 객관식 질문(MCQs)을 포함하고 있습니다. 전통적인 번역 기반 리소스와는 달리, UrduMMLU는 표준 학문적인 주제 및 우르두어 및 지역 특화된 내용을 아우릅니다.

- **Technical Details**: UrduMMLU는 이중 인간 주석을 통해 채점된 시험 유래 부분을 포함하며, 엄격한 합의 기준을 적용하여 라벨을 부여합니다. 이 벤치마크는 우르두어 교육 맥락에서 수집된 질문들로 구성되어, 번역에 의존하지 않고 원어로 작성된 최초의 포괄적 MMLU 스타일의 벤치마크입니다.

- **Performance Highlights**: 대규모 언어 모델(LLM) 평가에서 Gemini-3.5-Flash가 90.20% 및 90.34%의 정확도를 달성하여 최상의 성능을 보였습니다. 그러나 다른 모델들이 85%를 초과하지 못했으며, 각 모델의 우르두어 인문학 관련 주제에서는 25에서 40포인트가량 저조한 퍼포먼스를 보였습니다.



### Explicit Evidence Grounding via Structured Inline Citation Generation (https://arxiv.org/abs/2606.07130)
- **What's New**: AI 시스템의 채택이 증가함에 따라 사실적이고 신뢰할 수 있는 정보 생성에 대한 수요도 증가하고 있습니다. 이러한 맥락에서 본 논문은 각 주장에 대해 출처 문서와 지원 증거를 연결하는 구조화된 인라인 인용을 생성하는 FullCite 프레임워크를 소개합니다. FullCite는 prompt-based generation, constrained decoding, posthoc span alignment의 세 가지 전략을 제안하여 웹에서의 정보 출처에 대한 신뢰성을 높입니다.

- **Technical Details**: FullCite는 긴 맥락의 질문 응답(QA)에서 문서 수준과 증거 수준의 인용을 동시에 생성하는 데 사용됩니다. 연구는 BioASQ, ExpertQA, ASQA의 세 QA 벤치마크를 사용하여 수행되며, 문서 수준의 정확성, 증거 범위 식별 및 주장-인용 간의 신뢰성을 평가합니다. 특히, LLM은 관련 문서를 찾아내는 데 효과적이지만, 그 안의 정확한 지원 범위를 식별하는 데 어려움을 겪고 있습니다.

- **Performance Highlights**: 실험 결과, LLM은 문서 수준 인용에서는 높은 성능을 보였지만, 정확한 지원 증거를識별하는 데는 여전히 어려움을 겪고 있음을 보여줍니다. FullCite는 세 가지 인용 전략을 제안하여 ASQA에서 snippet-F1 점수를 12.80에서 61.87로 증가시키는 성과를 냈습니다. 이는 문서 참조와 지원 증거 범위를 동시에 참조함으로써 보다 투명하고 신뢰할 수 있는 정보를 제공합니다.



### Learning Perspectivist Social Meaning via Demographic-Conditioned Fusion Embeddings (https://arxiv.org/abs/2606.07123)
- **What's New**: 이 논문은 언어의 사회적 의미가 인식자(annotator)의 배경, 인구통계, 이념적 입장에 따라 다름을 강조합니다. 기존의 NLP 시스템들이 이러한 다양성을 간과하고 단일 진리 레이블로 축소하는 문제를 해결하기 위해, 저자들은 인구통계 그룹 간 해석의 변동을 포착하는 구조를 모델링했습니다. 이 연구는 P1SCO라는 대규모 데이터셋을 활용하여, 여러 모델링 패러다임을 벤치마킹하고, 텍스트 및 인구통계 정보를 통합한 퓨전 임베딩(fusion embeddings)을 제안합니다.

- **Technical Details**: 연구에서 언급된 주요 생성 모델은 RoBERTa-large로, 텍스트 입력에 대해 최종 계층의 첫 번째 토큰 표현을 수집하여 소셜 차원을 분류하는 작업을 수행합니다. 이 구조는 후보 텍스트와 주석자의 인구통계 프로필을 입력으로 받아, 사회적 차원이 각기 어떻게 해석될지를 예측합니다. 최종적으로는 19,022/26,912,691/54,805,480의 훈련/검증/테스트 후보로 구성되어, 인구통계 그룹에 따라 다르게 평가됩니다.

- **Performance Highlights**: 쿼드라인 모델과 비교하여 퓨전 모델은 모든 퓨전 전략에서 일관된 통계적 개선을 보였습니다. 특히, 모호한 차원인 Power와 Trust에 대한 예측에서 상대적으로 각각 +51.9%와 +30.1%의 개선을 기록했습니다. 최종적으로, 모든 퓨전 전략이 통계적으로 유의미한 결과를 도출하여, 인구통계 신호가 진정한 예측 정보를 제공함을 입증했습니다.



### Style or Content? Evaluating Style Classifiers with Controlled Content Overlap (https://arxiv.org/abs/2606.07103)
Comments:
          9 pages

- **What's New**: 이 논문에서는 스타일 분류기(style classifier)가 자연적으로 수집된 데이터에서 스타일 레이블과 상관 관계가 있는 내용 단서를 사용하는 경우를 연구합니다. 이를 위해, 성경 번역의 평행한 텍스트를 기반으로 한 통제된 콘텐츠 중첩(content overlap) 설정을 도입하고, 내용 동일성(content identity)과 스타일 레이블(style label) 간의 상호 정보(mutual information)의 정상화 잔여물(residual)로서 중첩 매개변수(parameter) α를 정의합니다. 이 연구는 콘텐츠 단서(content cue)와 스타일 학습(style learning)의 관계를 명확히 하기 위해 설계되었습니다.

- **Technical Details**: 이 연구에서 제안하는 α 매개변수는 콘텐츠 클래스 간에 공유되는 내용의 양을 측정하며, α의 값이 0에서 1까지 변화합니다. α=0일 때에는 각 스타일이 독특한 내용과 연관이 있어 콘텐츠만으로 스타일 레이블을 예측할 수 있고, α=1일 때에는 모든 스타일이 동일한 내용을 공유하여 콘텐츠가 더 이상 레이블 정보를 제공하지 않습니다. 교차 중첩 평가(cross-overlap evaluation)는 RoBERTa 기반 모델을 사용하여 각각의 중첩 레벨에서 훈련된 모델의 성능을 측정합니다.

- **Performance Highlights**: 연구 결과, 낮은 중첩 모델은 콘텐츠 단서가 제거되었을 때 성능이 급격히 떨어지지만, 높은 중첩 모델은 더 안정적인 전송 성능을 보였습니다. 또한, 콘텐츠 회복 가능성(content recoverability)은 α가 증가할수록 감소하며, 이러한 변화는 훈련 과정에서 점진적으로 일어납니다. 이러한 결과는 통제된 중첩이 콘텐츠 단서를 분리하고 스타일 학습을 평가하는 간단한 진단 지표로 작용할 수 있음을 시사합니다.



### SigmaScale: LLM Compression with SVD-based Low-Rank Decomposition and Learned Scaling Matrices (https://arxiv.org/abs/2606.07098)
- **What's New**: 이번 논문에서는 대규모 언어 모델(LLM) 압축을 위해 보조 스케일링 행렬(S) 학습을 위한 SigmaScale 방법을 소개합니다. SigmaScale은 스케일링 행렬을 분석적으로 도출하는 대신, 활성화 인지(compression loss) 압축 손실 하에서 대각선 행렬의 선형 변환 정의를 위해 두 세트의 벡터를 최적화합니다. 학습된 스케일링이 가중치 행렬의 효과적인 본질 순위를 낮추고, 이 감소가 압축 손실과 강하게 상관관계가 있음을 보여줍니다.

- **Technical Details**: SigmaScale의 첫 단계는 모델의 각 레이어와 모듈에 대한 압축 수준을 결정하는 감도 프로빙(sensitivity probing)입니다. 두 번째 단계에서는 가중치 행렬(W)에 선형 변환을 적용하는 스케일링 행렬을 학습합니다. 최적의 스케일링 행렬이 학습된 후, 모델의 최종 압축을 수행하고 가중치 재조정을 위한 압축 후 파인튜닝을 진행합니다.

- **Performance Highlights**: 시험 결과에 따르면, SigmaScale은 Llama 3.1 8B-Instruct 및 Qwen3-8B 모델에서의 혼란도(perplexity) 및 제로샷(zero-shot) 벤치마크에서 최신 SVD 기반 압축 방법들과 경쟁력이 있습니다. 특정 작업에서의 관찰된 장점은 SigmaScale의 접근 방식이 LLM 추론 계산 비용을 줄이기 위한 애플리케이션에 적합한 선택임을 시사합니다.



### mmPISA-bench: Do LLMs Reason Equally Well Across 43 Languages? (https://arxiv.org/abs/2606.07069)
- **What's New**: 이 논문에서는 OECD 국제학생 평가 프로그램(PISA)에서 파생된 고품질의 다국어 추론 벤치마크인 mmPISA-bench를 소개합니다. 이 벤치마크는 올바른 답변을 위해 추론이 필요한 25개의 객관식 질문으로 구성되어 있으며, 43개 언어로 공식적인 인간 번역본과 기계 번역본으로 제공됩니다. 연구 결과는 현대 LLM이 평가된 모든 언어에서 효과적으로 추론할 수 있으며, 일부 언어에서 성능 차이를 보이는 것을 확인했습니다. 기계 번역된 질문이 공식 번역에 비해 정확도를 저하시키지 않는다는 점도 주목할 만합니다.

- **Technical Details**: mmPISA-bench는 25개의 객관식 질문으로 구성되어 있으며, 이는 43개 언어로 공식 인간 번역과 기계 번역 버전이 함께 제공됩니다. 질문은 PISA에서 비롯된 것으로, 2022년의 수학 아이템 11개와 2018년의 읽기 이해 아이템 14개가 포함되어 있습니다. 데이터 수집 및 검증 작업은 수동으로 진행되었으며, 질문 형식은 다중 선택 방식으로 제한되었습니다. 수집된 데이터셋은 43개 언어에서의 LLM 성능 평가를 위한 체계적인 분석을 가능하게 합니다.

- **Performance Highlights**: LLM들은 모든 평가 언어에서 인간 응시자와 유사한 정확도를 달성하였으며, 기계 번역된 질문에 대한 성능이 공식 인간 번역보다 떨어지지 않았습니다. 논문에서는 LLM 사용이 더 비쌀 수 있으며, 언어에 따라 추론 길이와 정확성의 관계를 분석했습니다. 최종적으로, 이 벤치마크는 다국어 추론 능력을 평가하기 위한 기초 자료로서의 성과를 보여줍니다.



### Modeling semantic association in self-paced reading with language model embeddings (https://arxiv.org/abs/2606.07066)
- **What's New**: 본 연구에서는 다양한 언어 모델(LM)에서 생성된 임베딩을 사용하여 의미적 연관성을 측정합니다. 이를 위해 네덜란드어 자연 텍스트의 전자 뇌파(EEG)와 자율적인 독서 실험에서 데이터를 수집하여 분석하였습니다. 연구는 10가지 구현 방식을 통해 의미적 연관성을 평가하며, 임베딩 모델과 컨텍스트 길이에 따라 결과가 다름을 강조합니다.

- **Technical Details**: 임베딩 기반의 의미적 연관성을 측정하기 위해 제안된 10가지 방법에서는 다양한 LM 임베딩과 문맥 길이를 고려합니다. Bayesian hierarchical 모델과 Bayes factor를 사용하여 N400과 자율 독서 시간을 분석하였으며, 이 과정에서 문맥과 대상 단어 간의 의미적 연관성이 어떻게 차이를 만드는지를 밝혔습니다. 또한, 문장 임베딩이 의미적 연관성을 포착하는 데 있어 유망한 가능성을 나타냅니다.

- **Performance Highlights**: 연구 결과에 따르면, 임베딩 모델의 선택이 N400과 자율 독서 시간에 대한 의미적 연관성의 추정 효과를 바꿀 수 있다는 점이 중요합니다. 오직 문장 임베딩을 바탕으로 한 구현에서만 신뢰할 수 있는 의미적 연관성을 나타냈으며, 이는 단어 예측 가능성을 넘어서는 결과를 제공합니다. 이러한 발견은 의미적 연관성을 정량화할 때 방법론적 선택의 중요성을 강조합니다.



### TRACE: Trajectory Reasoning through Adaptive Cross-Step Evidence Aggregation for LLM Agents (https://arxiv.org/abs/2606.07054)
- **What's New**: 이 논문에서는 LLM(대규모 언어 모델) 에이전트의 자동 감시를 위한 TRACE라는 새로운 모니터링 프레임워크를 제안합니다. 이 프레임워크는 에이전트의 행동과 관련된 숨겨진 악의적 목표를 추적할 수 있는 능력을 향상시킵니다. TRACE는 TIJ(Triage-Inspect-Judge) 루프를 통해 고신호 지역을 식별하고, 조사를 수행하며, 최종적으로 경과된 결과를 종합하여 결정합니다.

- **Technical Details**: TRACE는 에이전트가 사용자 요청을 수행하는 과정에서 발생하는 일련의 사고(x\_1, x\_2, ..., x\_T)에서의 추적을 감시합니다. 이 시스템은 악의적 행동이 다른 plausible 행동의 흐름 속에서 어떻게 숨겨져 있는지를 효과적으로 탐지하기 위해, 의심스러운 신호를 집중적으로 조사하고, 장기적인 증거를 축적하며, 분산된 신호를 연결합니다. 이를 위해 TRACE는 TIJ 루프 구조를 사용하여 의심스러운 처리를 체계적으로 접근합니다.

- **Performance Highlights**: TRACE는 SHADE-Arena의 10개 작업 영역에서 평가되었으며, 전체 F1 스코어는 0.713, 재현율은 0.844에 달했습니다. 특히, 증거 연결이 필요한 작업에서 가장 큰 성과를 보였습니다. TRACE의 판단 프레임워크는 기존의 모니터링 방법들보다 더 나은 성능을 보여주며, 악의적 행동의 손상 징후를 더욱 효과적으로 탐지하는 데 기여합니다.



### Beyond Rubrics: Exploration-Guided Evaluation Skills for Reward Modeling (https://arxiv.org/abs/2606.07040)
Comments:
          24 pages, 6 images

- **What's New**: Eval-Skill은 기존의 경직된 리바이딩 방식 대신, 맥락 진화를 통해 보상 모델링을 위한 평가 기술을 합성하는 새로운 방법입니다. 이 접근 방식은 매 쿼리에 대한 규칙 생성을 필요로 하지 않으며, 판단의 세밀한 도메인-specific 사용자 선호를 따를 수 있는 능력을 제공합니다.

- **Technical Details**: Eval-Skill은 100개 사례를 사용하여 도메인 수준의 재사용 가능한 평가 기술을 생성하는 두 가지 단계인 워크플로우 생성과 원칙 생성을 활용합니다. 각 단계에서는 탐색(exploration)과 선택(selection)이 결합되어 평가 능력이 형성되며, 생성된 기술은 즉시 판별자(judge) 맥락에 주입됩니다.

- **Performance Highlights**: 여러 RM 벤치마크에서 Eval-Skill은 다양한 판별자 백본(backbone)의 성능을 일관되게 향상시킵니다. 특히 RewardBench 2에서는 Qwen3-8B에서 13.44% 그리고 DeepSeek-V4-Flash에서 18.51% 이상의 개선을 나타냈습니다.



### MADE: Beyond Scoring via a Multilingual Agentic Diagnosing Engine for Fine-Grained Evaluation Insights (https://arxiv.org/abs/2606.07020)
- **What's New**: 이 논문에서는 MADE, 즉 다국어 에이전틱 진단 엔진을 제안하여 포스트 평가 분석을 보다 정밀하게 분해하고, 다국어 모델의 성능을 개선하는 방법을 제시합니다. 기존의 진단 과제가 점점 더 복잡해지는 데 반해, MADE는 언어, 문화, 샘플 수준에서 구체적인 개선 조치를 제시할 수 있도록 설계되었습니다. 연구는 54개의 질문과 15개 언어로 구성된 진단 세트를 사용하여 MADE의 퀄리티가 기존 기준보다 47% 향상되었음을 보여줍니다.

- **Technical Details**: MADE는 평가 후 분석을 다섯 가지의 역할로 분리하여 수행하는 시스템으로, 이들 역할은 Planner, Evidence Analyst, Case Analyst, Language Reflector, Reporter로 구성됩니다. 이 시스템은 8.66M 개의 평가 기록을 활용하여, 각 모델의 언어 및 문화적 특성에 따른 강점을 상세히 분석할 수 있도록 합니다. 또한, MADE는 구조화된 진단 쿼리 세트를 제공하여 다국어 평가의 세부 사항을 효과적으로 식별할 수 있게 합니다.

- **Performance Highlights**: MADE의 실험 결과는 전문가들이 선호하는 진단 퀄리티에서 87.9%의 우수성을 보이며, 기존 공통 기준 대비 47% 향상된 성과를 기록했습니다. 다국어 전문가와의 협업을 통해 브랜딩 평가 점수를 행동 지도로 변환하는 네 가지 실용적인 발견을 도출하여, 모델 선택 및 수정에 대한 유용한 지침을 제공합니다. 이로 인해, MADE는 단순한 점수 집계에서 벗어나 실제로 모델의 개선을 위한 의미 있는 통찰력을 제공합니다.



### Principles of Concept Representation in Sentence Encoders (https://arxiv.org/abs/2606.06994)
- **What's New**: 이 논문은 문장 인코더가 훌륭한 개념 표현을 생성하기 위해 필요한 조건을 탐구한다. 연구자들은 개념 집합의 구조적 불일치와 현재 인코더의 한계를 세 가지 가설을 통해 조사했다. 이 연구는 새로운 두 개의 평가 데이터 세트를 출시하며, 각 데이터 세트는 개념 간의 의미적 갭을 측정하는 데 초점을 맞춘다.

- **Technical Details**: 연구에서는 문장 인코더에 대한 네 가지 원칙(P1~P4)을 규명하고, 이를 통해 인코더가 개념의 조합성을 어떻게 충족시키는지 논의한다. 인코더는 서로 다른 의미 연산자들을 단일 잠재 공간에 통합해야 하며, 각 조합의 유형에 따라 효과가 달라지는 구조적 한계를 지닌다. 이 과정에서, 하드 네거티브 감독, 최종 레이어 시너지, 파인튜닝 등의 기술이 강조된다.

- **Performance Highlights**: 연구 결과, 개념-동등성 파인튜닝이 복잡한 명사 검색에서 성능을 향상시키며, 하드 네거티브 감독이 검색 순위를 조정하는 데 기여한다고 나타났다. 또한, 최종 레이어에서의 의미 신호 집중이 cross-layer pooling의 효과를 제한하는 중요한 요소로 확인되었다. 이러한 결과는 파인튜닝이 인코더의 구조적 발전을 이끌 수 있음을 보여준다.



### Contrastive Training with LLM-generated Near-Misses for Robust Code-Switching Speech Recognition (https://arxiv.org/abs/2606.06985)
Comments:
          Accepted at INTERSPEECH 2026

- **What's New**: 이 논문에서는 코드 스위칭(code-switching, CS)을 효과적으로 인식하기 위한 새로운 프레임워크인 Point-of-Interest (POI) 인식 대비 학습(contrastive training) 방식을 제안합니다. 기존 ASR에서 CS 영역에 대한 인식 오류가 심각한 문제로 지적된 바 있으며 본 연구에서는 이러한 문제를 해결하기 위해 POI 탐지 방법을 적용하여 CS 구간을 식별하는 과정을 설명합니다. 또한, 접근법의 독창적인 부분은 아크로스틱 근사치 바탕의 근접 생성(near-miss hypotheses)을 통해 ASR 성능을 강화하는 것입니다.

- **Technical Details**: POI인식 대비 학습은 주어진 훈련 데이터에서 ASR 모델이 생성한 N-best 가설을 활용하며, 이 가설 중에서 CS 구간이 포함된 부분을 탐지하는 방법을 포함합니다. 문서에서 제안된 CS-NMG는 음향 정보에 기반하여 POI 근접 가설을 생성하며, 이는 음향적, 음소적(phonemic), 텍스트적 제약 조건을 통해 여과되는 강력한 부정체(hard negatives)를 포함합니다. 주로 POI 구간에 대한 상대적 균등 교차 엔트로피(cross-entropy) 앵커 목표와 다중 부정 대비(rank loss)를 결합하여 최적화합니다.

- **Performance Highlights**: 실험 결과 CS-FLEURS(cmn-eng)와 ViMedCSS(vie-eng) 벤치마크에서 전통적인 단어 오류율(Word Error Rate, WER) 및 POI 오류율(Point-of-Interest Error Rate, PIER)에서 2% 이상의 지속적인 오류율 감소를 보였습니다. 이 접근법은 특히 일반적이냐 CS 인식에 있어 한층 개선된 성능을 보이며, 기존의 방법론보다 향상된 결과를 나타냅니다. 또한 CS 구간에 대한 모델의 견고성 향상에 중요한 기여를 하고 있습니다.



### Tree-of-Experience: A Structured Experience-Management Solution for Self-Evolving Agents under Low-Repetition and Implicit-Reward Environments (https://arxiv.org/abs/2606.06960)
- **What's New**: 본 논문에서는 기존 LLM (Large Language Model) 에이전트의 경험 기반 자기 진화를 평가하기 위한 새로운 벤치마크인 FinEvolveBench를 소개합니다. 일반적인 경험 재사용 시나리오를 넘어, 의도하지 않은 보상이 있는 낮은 반복성 작업 환경에서 에이전트가 적응하는 능력을 테스트합니다. 또한, Tree-of-Experience(ToE)라는 구조화된 경험 관리 방법을 제안하여, 금융 정서 예측 작업에서 경험을 효과적으로 관리할 수 있는 방법을 제시합니다.

- **Technical Details**: FinEvolveBench는 연속적으로 구성된 금융 시장 환경과 잘 정의된 자기 진화 에이전트 워크플로우로 구성되어 있습니다. 이 환경은 시간적으로 상호작용 가능하며, 에이전트는 현재 날짜까지의 정보만을 참조하여 예측을 수행합니다. Tree-of-Experience는 경험 관리 프로세스를 검색(retrieval)과 업데이트(updating) 단계로 나누어 체계적으로 지원하는 방법론을 제공합니다.

- **Performance Highlights**: 실험 결과, 일반적인 경험 기반 메커니즘들은 저경험 대비 일관되게 성능이 향상되지 않음을 나타내며, 이는 금융 시장의 저반복성 환경에서 경험의 재사용이 어려움을 강조합니다. 반면에, 구조화된 경험 관리 방법인 ToE는 더 강력한 전반적인 성과를 달성하여, 낮은 반복성과 묵시적 보상 환경에서 에이전트의 자기 진화에 대한 필요성을 강조합니다.



### OpenHalDet: A Unified Benchmark for Hallucination Detection across Diverse Generation Scenarios (https://arxiv.org/abs/2606.06959)
Comments:
          Preprint. Code and data are available at this https URL

- **What's New**: OpenHalDet는 환각(hallucination) 감지를 위한 통합 벤치마크로, 다양한 생성 시나리오에서의 신뢰할 수 있는 평가를 목표로 한다. 기존의 환각 탐지 방법들이 일반적으로 채택하는 비일관적인 평가 구성 및 제한된 하위 도메인으로 인해 발생하는 문제점을 해결한다. 이를 통해 OpenHalDet는 검증된 구조를 갖춘 코드베이스를 제공, 향후 연구 및 개발을 위한 재현 가능한 평가를 용이하게 만든다.

- **Technical Details**: OpenHalDet는 프롬프트 생성, 응답 생성, 진실성 주석, 탐지기 점수화 및 메트릭 계산 등 주요 평가 단계를 표준화하였다. 또한, 블랙박스(black-box), 그레이박스(gray-box), 화이트박스(white-box) 방법을 포괄하여 다양한 탐지 모델을 수용할 수 있도록 설계되었다. 최종적으로, 17개의 데이터셋과 4개의 LLM 백본을 통한 전체 평가를 포함한다.

- **Performance Highlights**: OpenHalDet는 효율적이고 포괄적인 평가 체계를 통해 다양한 기존 연구와 비교할 수 있는 공정한 기준을 제공한다. 이를 통해 다양한 작업 및 문맥에서의 탐지기 성능을 보다 신뢰성 있게 분석할 수 있다. 기존 연구들이 좁은 작업에만 초점을 맞춘 반면, OpenHalDet는 다양한 시나리오에 대해 보다 넓은 범위의 평가를 지원한다.



### Auditing Training Data in Domain-adapted LLMs: LoRA-MIN (https://arxiv.org/abs/2606.06946)
Comments:
          IEEE Conf. on Computers, Software, and Applications (COMPSAC), 2026

- **What's New**: 본 논문에서는 LoRA-MINT라는 새로운 방법론을 제안합니다. 이 방법론은 특정 자연어 처리(NLP) 작업을 위해 조정된 대규모 언어 모델(LLM)의 회원 추론 테스트(Membership Inference Test, MINT)에 적용됩니다. LoRA 기반의 모델이 훈련 데이터에 포함된 샘플인지 판단하는 것을 목표로 하여, 지적 재산권 및 민감한 데이터 관리에 유용한 감사 도구를 제공합니다.

- **Technical Details**: LoRA-MINT의 주요 구성 요소는 모델의 perplexity(혼란도)와 회원 상태 간의 관계를 탐구하고, 데이터의 노출 정도를 추정하는 체계적인 프레임워크를 제공합니다. 저자들은 네 가지 모델과 세 가지 벤치마크 데이터셋을 대상으로 실험을 수행하여 훈련 데이터 사용 여부를 판단하는 정확도 값이 0.77에서 0.92 사이임을 확인했습니다. LoRA 기술을 기반으로 한 파라미터 효율적인 미세 조정이 어떻게 보다 나은 성능을 제공하는지를 입증합니다.

- **Performance Highlights**: LoRA-MINT는 훈련 샘플과 비훈련 샘플을 효과적으로 구별할 수 있는 도구로, 기존의 최첨단 기법들을 능가하는 성능을 보여줍니다. 저자들은 이 방법이 LLM 감사 및 AI 기술의 윤리적이고 책임 있는 배포를 촉진하는 데 있어 중요한 잠재력을 지닌다고 강조합니다. LoRA-MINT는 범용적이고 확장성 있는 감사 도구로서, 훈련 중 데이터 노출을 탐지하여 투명성을 향상시키는 데 기여합니다.



### Didact: A Cross-Domain Capability Discovery System for Defenc (https://arxiv.org/abs/2606.06942)
Comments:
          Under Review at CIKM 2026 (System Demonstration Track)

- **What's New**: 이 논문에서는 호주에서 공개적으로 제공되는 방위 보고서와 정책 문서를 통합하는 프로토타입 툴인 Didact를 소개합니다. Didact는 방위 및 학술 연구 분야의 정보를 보다 효율적으로 탐색하는 기능을 제공하며, 사용자가 자연어 대화를 통해 관련 정보에 접근할 수 있도록 합니다. 특히, 다원적 Retrieval-Augmented Generation (RAG) 파이프라인을 활용하여 모든 관련 소스를 연결하고, 상호작용할 수 있는 Evidence Rail을 통해 증거 및 출처 관계를 시각화하는 점이 특징입니다.

- **Technical Details**: Didact는 사용자의 질문을 입력으로 받아 텍스트 응답과 함께 관련 문서 조각 및 부그래프의 Evidence Rail을 생성하는 파이프라인으로 구현되어 있습니다. 이 시스템은 두 가지 소스 유형(호주 방위 문서 및 연구 지식 그래프)을 활용하며, 별도의 접근 수준으로 문서를 분리하여 보안을 유지합니다. Didact는 사용자 쿼리에서 도메인 관련 키워드를 추출하여 지식 그래프를 통해 적절한 정보를 검색하고, 복합 RAG 구조를 통해 다양한 질문에 대한 응답을 효과적으로 생성할 수 있습니다.

- **Performance Highlights**: Didact는 방위 분야의 정책 결정자 및 분석가들에게 다원적 방식으로 정보를 탐색할 수 있는 혁신적인 도구로서, 실험 평가를 통해 그 유용성을 입증하였습니다. 이 시스템은 사용자 인터페이스에서 사용하는 Evidence Rail을 통해 데이터의 출처를 손쉽게 추적할 수 있고, 적절한 의사 결정에 필요한 증거 기반 정보를 제공합니다. Didact는 타 분야로의 확장이 가능하여 정보가 분산된 다른 영역에서도 유용하게 활용될 수 있는 잠재력을 가지고 있습니다.



### ThinkBooster: A Unified Framework for Seamless Test-Time Scaling of LLM Reasoning (https://arxiv.org/abs/2606.06915)
- **What's New**: ThinkBooster는 LLM(대형 언어 모델)의 테스트 시간 컴퓨팅(TTC) 스케일링을 위한 통일된 프레임워크입니다. 이 프레임워크는 TTC 스케일링 전략과 스코어러(scorer) 가족을 구현하는 모듈식 파이썬 라이브러리, 성능 및 계산 효율성을 종합적으로 평가하는 벤치마크, 실제 애플리케이션에 적응적 추론을 통합할 수 있는 OpenAI 호환 프록시 서비스로 구성되어 있습니다. Empirical results는 ThinkBooster가 수학적 및 코딩 과제에서 실제적인 성능 향상을 제공함을 보여줍니다.

- **Technical Details**: 이 라이브러리는 다양한 TTC 스케일링 전략과 스코어러를 포함하여, 최적의 성능-비용 트레이드오프를 검토할 수 있도록 지원합니다. TTC 스케일링 전략은 오프라인과 온라인에서 모두 작동할 수 있으며, 각 알고리즘의 세부적인 구현은 PyTorch와 Hugging Face 라이브러리를 통해서 이루어집니다. 그 외에도, 다양한 종류의 스코어러가 LLM 출력이나 내부 상태에 따라 평가되고, 이는 white-box와 black-box 전략으로 구분되어 사용됩니다.

- **Performance Highlights**: ThinkBooster는 실질적인 코딩 및 수학적 문제 해결에서 TTC 스케일링 전략의 성과를 입증했습니다. 이러한 개선은 직관적인 API 및 통합 레이어를 통해 실 세계 애플리케이션의 배포를 용이하게 합니다. 그러나 TTC 스케일링의 도입 장벽을 낮추는 동시에, 연구자들이 연구를 진행할 수 있는 벤치마크도 제공합니다.



### EASE-TTT: Evidence-Aligned Selective Test-Time Training for Long-Context Question Answering (https://arxiv.org/abs/2606.06906)
Comments:
          13 pages, 4 figures, 3 tables

- **What's New**: 본 논문에서는 Evidence-Aligned SElective Test-Time Training (EASE-TTT)이라는 새로운 테스트 타임 트레이닝 프레임워크를 제안합니다. 이 방법은 질문과 관련된 증거(chunk)들을 선택하여 소프트 어텐션(supervision target)으로 변환하며, 이를 모델의 훈련에 반영하여 장문의 질문 응답(long-context QA) 성능을 개선합니다. EASE-TTT는 증거를 단순히 입력에서 가져오는 것이 아니라, 모델의 어텐션 메커니즘을 최적화하여 전체 맥락에서의 응답 품질을 높입니다.

- **Technical Details**: EASE-TTT는 먼저 입력 맥락에서 질문과 가장 관련성이 높은 증거 조각(chunk)을 선택합니다. 이후, EASE-TTT는 이러한 선택된 증거 위치에 대한 소프트 어텐션 타겟을 만들어, 원래의 맥락은 유지하되 더욱 높은 가중치를 부여합니다. 테스트 시점에서 경량 쿼리 전용 어댑터가 업데이트되며, 이를 통해 모델은 최종 응답을 전체 맥락에서 생성합니다.

- **Performance Highlights**: EASE-TTT는 여러 개의 장문 질문 응답(Task) 벤치마크에서 실험을 수행하였고, 기존의 전체 맥락 추론, 단순 증거 검색(baselines), 쿼리 전용 테스트 타임 트레이닝(qTTT)보다 더 나은 성능을 보였습니다. 실험 결과, EASE-TTT는 선택된 증거의 중요성과 소프트 어텐션 감독의 효과를 통해 모델의 성능을 크게 향상시키는 것으로 나타났습니다.



### An Expanded Synthetic Conversation Dataset for Multi-Turn Smishing Detection (https://arxiv.org/abs/2606.06879)
- **What's New**: 이 연구에서는 COVA-X라는 확장된 데이터셋을 소개하며, 이는 10,985개의 대화로 구성되어 8개의 노인-targeted 사기 카테고리를 포함합니다. 기존의 COVA 데이터셋보다 3.43배 증가하며, 이는 데이터 품질을 개선하기 위한 새로운 생성 파이프라인을 반영합니다. Longformer 모델이 모든 평가 지표에서 XGBoost를 초월한 주요 발견을 통해, transformer 모델들이 더 큰 대화 코퍼스에서 그들의 맥락적 장점을 실현할 수 있음을 확인했습니다.

- **Technical Details**: COVA-X 데이터셋은 contamination, label mismatch, stage-direction bleed, 및 prompt-design failures를 해결하는 개선된 생성 파이프라인을 통해 생산되었습니다. 이 연구는 3개의 분류기 아키텍처에 대한 전반적인 데이터셋 품질 수명 주기 분석을 통해, 레이블 수정율이 49.8%에서 3.9%로 감소하고, 가상 유괴 아티팩트의 비율이 67.1%에서 46.5%로 감소하는 등의 성과를 문서화하였습니다. 또한, 각 사기 유형에 따른 결과 분석을 통해 사기 카테고리가 결과를 일관성 있게 조절한다는 점을 발견했습니다.

- **Performance Highlights**: 재훈련된 모든 분류기에서 Longformer가 XGBoost를 초과하여 79.71%의 정확도와 0.7786의 macro F1 점수를 달성했습니다. 이는 첫 번째 논문에서 예측한 데이터 크기 가설을 검증하며, 정리된 데이터셋에서 모든 아키텍처가 레이블 관련 신호를 개선하는 것이 확인되었습니다. 이 연구는 합성 데이터셋 구축의 미래를 위한 실질적인 로드맵을 제공하는 방법론적 기여를 주장합니다.



### Are Large Language Models Suitable for Graph Computation? Progress and Prospects (https://arxiv.org/abs/2606.06865)
- **What's New**: 이 논문은 그래프 수치 연산(graph computation)을 위한 대형 언어 모델(LLMs)의 포괄적인 검토를 제공합니다. LLMs를 역할 기반 분류법에 따라 두 가지 주요 패러다임으로 나누어 직업(task) 해결자로서의 역할과 문제를 형성하고 외부 도구나 에이전트를 호출하는 계획자로서의 역할을 구분합니다. 이 연구는 LLM이 간단한 작은 그래프 작업에서 유망하지만 대규모 및 정확성이 요구되는 그래프 작업에서는 신뢰성이 떨어진다는 것을 강조합니다.

- **Technical Details**: 그래프(G=(V,E))는 비선형 데이터 구조로, 정점(vertices)과 정점 간의 연결인 간선(edges)으로 구성됩니다. LLM들은 이러한 그래프 구조를 텍스트 형식으로 설명하여 사용되며, 일반적으로 가장 많이 사용되는 그래프 표현 방법으로는 엣지 리스트(edge list)와 인접 리스트(adjacency list)가 있습니다. LLM은 이러한 그래프를 기반으로 알고리즘을 수행하여 결정론적 결과를 생성하는 그래프 수치 연산을 수행하는데, 이는 기존의 그래프 학습(task)과 구별됩니다.

- **Performance Highlights**: LLMs는 소규모의 간단한 작업에 대해 유망한 성과를 보이지만, 그래프의 크기나 복잡성이 증가함에 따라 성능이 저하되는 경향이 있습니다. 따라서, 특정한 그래프 수치 작업에 대해 신뢰성을 확보하지 못하고 있으며, 정확성을 중시하는 작업에서는 문제가 발생할 수 있습니다. 향후 방향으로는 복잡한 그래프 쿼리를 위한 다단계 실행 최적화, 그래프에 대한 프라이버시 보존, 도메인 특화 그래프 구조의 적응 등이 제안되고 있습니다.



### Interpreting Brain Responses to Language with Sparse Features from Language Models (https://arxiv.org/abs/2606.06857)
- **What's New**: 이번 연구는 Augmented Sparse Encoding Models라는 새로운 인코딩 프레임워크를 소개합니다. 이 모델은 기존의 언어 모델(hidden states)에서 밀집된 상태를 계층적으로 조직된 희소 자동 인코더(sparse autoencoder) 특징으로 대체하고, 서프리살(surprisal)을 예측 변수로 포함시킵니다. 이를 통해 우리는 뇌의 신경 반응에 대한 해석을 생성하고, 모델-뇌 정렬이 언어 모델의 주요 또는 특이적 변화를 반영하는지를 검증합니다.

- **Technical Details**: 이 연구는 200개의 다양한 언어적 문장을 듣는 8명의 참가자로부터 수집된 초고자기장 7T fMRI 데이터셋을 사용합니다. 연구자들은 희소 자동 인코더(SAE) 특징이 밀집 LM 특징만큼의 정확도로 뇌의 반응을 예측할 수 있음을 발견하였으며, 이런 특징들은 의미, 난이도와 관련된 과거의 신경 과학적 발견을 확장하는 해석도 제공합니다. 또한, 뇌의 언어 네트워크 내부에서의 서로 다른 영역들이 처리 난이도와 내용 특징에 따라 다르게 반응함을 보여주었습니다.

- **Performance Highlights**: 뇌의 언어 처리 중 반응은 단순히 무작위 LM 특징에서 예측될 수 없으며, 일반적인 LM 표현에서 인코딩된 가장 일반적인 정보를 캡처하는 특징에 의해 가장 잘 설명됩니다. 이는 인공 언어 표현과 생물학적 언어 표현 사이의 비판적인 대응 관계를 시사합니다. 저자들은 이러한 발견을 통해 언어 모델과 뇌의 언어 처리 메커니즘 사이에 더 깊은 상관관계가 있음을 밝혀냈습니다.



### CRAFT: A Unified Counterfactual Reasoning Framework for Tabular Question Answering and Fact Verification (https://arxiv.org/abs/2606.06842)
Comments:
          24pages,10 figures

- **What's New**: 이번 연구에서 제안하는 CRAFT는 탁월한 Counterfactual Reasoning Framework로, 기존의 Tabular 질문 응답 및 사실 검증 방법을 양방향 검증 프로세스로 재정의합니다. 이는 단일 방향 추론의 한계를 극복하고 다양한 가설을 탐색할 수 있는 능력을 제공합니다. 실험 결과는 CRAFT가 기존의 방법들을 뛰어넘어 복잡한 질문 응답에서 특히 큰 향상을 보여준다는 것을 입증합니다.

- **Technical Details**: CRAFT는 총 네 가지 협력 모듈(모듈)로 구성되어 있습니다: 1) Rewriter는 질문을 선언적 가설로 변환하여 검증 가능한 주장으로 전환합니다. 2) Reverser는 이 가설을 정보성 있는 반사적 진술로 변형하여 반대 시나리오를 만듭니다. 3) Extractor는 LLM의 중간 추론 단계에서 필수 지원 증거를 추출합니다. 4) Rethinker는 추출된 증거와 후보 답변을 집계하여 최종 결정을 내립니다.

- **Performance Highlights**: CRAFT는 WikiTQ와 TabFact와 같은 데이터 세트에서 일관되게 대표적인 기준과 비교하여 우수한 성능을 나타냅니다. 또한, CRAFT는 서로 다른 LLM의 성능 격차를 크게 완화하여 단일 방향 추론의 제한을 효과적으로 극복합니다. 마지막으로, 이번 연구는 신뢰할 수 있는 추론 시스템 개발에 있어 새로운 관점을 제공하며 LLM이 보다 유연한 추론으로 유도될 수 있음을 보여줍니다.



### Characterize Then Distill: Mechanistic Reasoning in Large Output Spaces (https://arxiv.org/abs/2606.06840)
- **What's New**: 본 논문은 현대의 추론 모델들이 수백만 개의 후보 라벨 중에서 소수의 관련 라벨을 선택하는 데 있어 놀라운 제로샷 성능을 발휘하는 방식을 분석합니다. 저자들은 추론 과정을 두 단계로 구분하여 설명하는데, 첫 번째 단계는 후보군을 넓게 선별하는 'shortlisting'이며, 두 번째 단계는 결과 집합에 대한 세밀한 추론입니다. 이러한 단계를 서로 분리할 수 있으며 상호 보완적이라는 증거를 다양한 데이터셋에서 제공합니다.

- **Technical Details**: 이 논문에서는 현대의 LLM(대규모 언어 모델)의 추론 구조를 두 단계로 나누어 분석합니다. 첫 번째 단계인 'Coarse Semantic Filtering'에서는 입력 데이터의 강한 의미 신호와 관련된 주요 토큰을 식별합니다. 두 번째 단계인 'fine-grained reasoning'에서는 이러한 주요 신호에 따라 세부적인 추론 과정을 진행하며, 이를 통해 후보 라벨의 최종 선택을 지원합니다.

- **Performance Highlights**: 연구 결과, 기존의 CoT(distillation) 방법보다 저자들이 제안한 기계적 증류(mechanistic distillation) 방법이 더 나은 성능을 보입니다. 이 방법은 각 단계별 계산을 직접 감독함으로써 학생 모델의 성능을 향상시킵니다. 예를 들어, 대규모 다중 라벨 과제를 수행할 때 두 단계의 과정이 명확하게 드러나며, 이러한 접근 방식은 성능 평가에서 유리한 결과를 이끌어냈습니다.



### Translate-R1: Cost-Aware Translation Tool Use via Reinforcement Learning (https://arxiv.org/abs/2606.06835)
Comments:
          14 pages main text plus appendix, 7 figures, 11 tables

- **What's New**: LLMs(대형 언어 모델)의 언어 간 성능 차이를 해소하기 위한 새로운 방법이 제안되었습니다. 이 방법은 번역(Translation)을 통해 모델의 주요 언어로 입력을 변환하여, 완전한 기능을 즉시 발휘하게 합니다. 기존의 수동적인 언어 선택 방식은 오히려 효율이 떨어질 수 있기에, 단일 정책(policy)을 학습하여 상황에 맞게 번역을 호출하는 방식을 채택했습니다.

- **Technical Details**: 이 연구는 보상(reward)만을 기반으로 여러 언어와 도메인에서 기계가 자신의 이해도를 스스로 판단하고 필요한 경우에만 번역하는 정책을 학습하도록 합니다. 학습된 정책은 다양한 자원 수준과 비용 예산을 아우르며, 명시적인 언어 식별 또는 라우팅 규칙 없이도 언어-도메인 적응형 내성(introspection)을 발휘할 수 있습니다. 이를 통해 수치적 정량화가 가능하며, Gated GSPO라는 신뢰 기반의 도구 사용 방식을 도입하여 비용 효율성을 극대화합니다.

- **Performance Highlights**: 제안된 gated policy는 22개의 언어에서 자원 수준에 따른 보상을 개선하였으며, 높은 자원 수준에서 +4.6, 낮은 자원 수준에서 +23.5, 매우 낮은 자원 수준에서 +17.5의 성과를 보였습니다. 비용 대비 모두 최적 상태를 유지하며, 63%의 비용으로도 전체 보상을 유지할 수 있는 성능을 나타냈습니다. 또한, 전혀 노출되지 않았던 두 개의 합성 언어에서 적절하게 번역을 수행하며 +18.7 포인트를 개선하는 성과를 보여줍니다.



### The Dark Regulome: Disentangling Predictability from Regulation in Genomic Foundation Models (https://arxiv.org/abs/2606.06834)
- **What's New**: 본 논문에서는 고급 신경교종(gliomas)이 신경 회로에 기능적으로 통합되는 과정을 조사하며, 비단백질 코딩(noncoding) 요소들이 종양 세포에서의 시냅스 생성을 유도하는 유전자 발현에 어떤 영향을 미치는지를 다룹니다. 특히, 어두운 유전체(dark genome)에서 규제 프로그램을 탐색하는 새로운 방법론을 소개하고, 이를 통해 종양 세포의 비정상적인 유전자 조절 패턴을 해석하는 데 어려움을 덜어줄 수 있는 도구를 제공합니다.

- **Technical Details**: 연구진은 Caduceus-Ph, HyenaDNA, Enformer라는 세 가지 서로 다른 기초 모델(foundation models)을 사용해 30,448개의 어두운 유전체 요소를 분석하였습니다. 새로운 진단 도구인 잔여화 및 순열 진단(residualization-and-permutation diagnostic)을 통해 예측 가능성 기반의 변량과 규제 기반 변량을 구분합니다. 10kb 근접 규제 수평선(proximal-regulatory horizon)이 모든 아키텍처에서 일관성을 보였으나, 모델의 순위 구성을 기반으로 한 규제 해석은 단일한 모델에 의해 왜곡될 수 있음을 발견했습니다.

- **Performance Highlights**: 이 연구는 유전자 발현 제어에 기여하는 상위 100개의 요소가 뇌의 cis-eQTL에 대해 3.3배 더 높은 비율로 일치함을 보여주며, 이는 뇌 관련 유전자 조절에 대한 새로운 통찰력을 제공합니다. 또, Transposable elements(TE)와 NRXN1+NLGN1 단백질 쌍의 규제 레이어 주장은 적절한 순열 테스트에서 실패하여 과거 연구에서 강조된 패턴을 재검토해야 함을 시사합니다. 제공된 진단 도구는 ISM 기반 규제 연구에서 보다 정확한 결과를 도출하는 데 기여할 것으로 기대됩니다.



### Progress-SQL: Improving Reinforcement Learning for Text-to-SQL via Progressive Rewards (https://arxiv.org/abs/2606.06825)
- **What's New**: 이번 논문에서는 Progress-SQL이라는 새로운 다단계 강화학습(RL) 프레임워크를 제안합니다. 이 프레임워크는 Text-to-SQL 생성에서 점진적인 보상을 제공하여 SQL 개선을 위한 더 나은 지침을 생성합니다. 구문 레벨의 구조적 프로파일을 추상화하고 진단 피드백을 생성하는 Oracle-guided Diagnostic Tree(ODT)를 도입하여, SQL 예측을 정교하게 다듬을 수 있습니다.

- **Technical Details**: Progress-SQL은 초기 SQL에서 최종 SQL로의 개선을 측정하는 점진적 보상을 정의합니다. 또한, 조기 정확성(progression latency)과 실행 상태(execution status) 보상을 포함하여 SQL 예측이 유효하지 않은 경우 회복을 장려합니다. ODT 기반의 구조 정렬(structural alignment)과 어휘 정렬(lexical alignment)을 결합하여 더 밀집된 보상 신호를 생성합니다.

- **Performance Highlights**: BIRD 및 Spider 데이터셋에서의 실험을 통해 Progress-SQL은 실행 정확도를 평균 8.5% 향상시켰습니다. 기존 방법들과 비교해도 성능이 경쟁력 있거나 더 우수한 결과를 보여 투명한 개선을 입증했습니다. 다양한 기본 모델에 대한 광범위한 실험도 진행되어, 모든 환경에서 꾸준히 개선된 성능을 확인할 수 있었습니다.



### Quantifying Media Representation Dynamics Across 25 Years of News Reporting on Policing-related Deaths (https://arxiv.org/abs/2606.06812)
Comments:
          9 pages, 6 figures. Websci'26

- **What's New**: 이 연구는 지난 25년간 캐나다의 경찰 관련 사망 사건에 대한 뉴스 서사를 대규모로 분석한 결과로, 4,000개의 기사를 사용했다. 새로운 계산 모델인 PerspectiveGap을 개발하였으며, 이는 사회학적 연구에 기초하여 미디어의 경찰에 대한 표현 방식을 분석하는 데 도움을 준다. 연구 결과, 경찰 관련 사망 사건에 대한 보도는 일반적으로 민간인보다 국가 관료의 관점이 세 배 더 많이 나타나는 것으로 나타났다.

- **Technical Details**: PerspectiveGap 모델은 언어 모델만을 사용하여 소비자급 노트북에서도 작동할 수 있도록 설계되었다. 이 모델은 민간인의 관점과 관료의 관점을 정확하게 인식할 수 있으며, 소량의 주석 데이터로도 GPT-4o와 유사한 성능을 자랑한다. 해당 모델은 캐나다 뉴스 매체의 4,000개 기사를 분석하는 데 적용되어, 경찰 관련 사망 사건에 대한 보도가 주로 기술적이며 관료의 목소리가 우세하다는 결과를 도출했다.

- **Performance Highlights**: 최근 2020년에서 2023년 사이에 민간인의 관점이 보도에서 증가하는 경향이 확인되었으나, 전체적으로 여전히 관료 중심의 보도가 지배적이었다. 뉴스 매체 간에도 민간인 중심의 보도 비율이 상이하여, 일부 매체에서는 민간인의 목소리가 덜 다루어졌다. 연구가 제시하는 PerspectiveGap 프레임워크는 다른 사법 관할권에서도 적용 가능하여, 경찰 작용과 책임성에 대한 서사를 분석하는 데 도움이 될 것이다.



### Korean Culture into LLM Alignment: Toward Cultural Coherenc (https://arxiv.org/abs/2606.06797)
Comments:
          Accepted to ICML 2026 Workshop on Culture X AI

- **What's New**: 이번 논문은 한국을 대상으로 하여 문화적으로 올바른 반응을 정의하려는 노력을 제시했습니다. 기존의 방법들이 부정적인 요소를 억제하는 데 중점을 두었던 것에 비해, 이 연구는 긍정적이고 건설적인 응답이 필요하다는 주장을 하고 있습니다. 이를 위해 한국의 법적 틀과 사회적 규범을 기반으로 한 가이드라인을 설정하고, 한국 문화에 맞춘 안전한 응답 정책을 중심으로 한 데이터 생성 파이프라인을 디자인했습니다.

- **Technical Details**: 연구진은 한국 공공 자료를 기반으로 한 질의 생성, 자동화된 레드팀(Red Teaming), 다중 모델 안전 응답 생성기를 통합하여 문화적으로 일관된 응답을 생성하는 데이터 생성 파이프라인을 구축했습니다. 이 파이프라인은 질의-응답 쌍을 생성하고, 한국 문화에 적합한 기준으로 필터링하여 문화적 일관성을 높이는 데 초점을 맞추고 있습니다. 이러한 방식으로 생성된 데이터 세트가 한국 모델의 문화적 정렬과 일반적인 성능을 동시에 개선할 수 있음을 입증했습니다.

- **Performance Highlights**: 연구 결과, 한국의 문화적 안전 비율이 향상되었고, 여섯 개의 오픈-웨이트 LLM(대형 언어 모델)에서 일반적인 성능 지표에도 큰 저하가 없음을 확인했습니다. 미세 조정(fine-tuning)된 모델들의 출력을 품질적으로 분석한 결과, 한국의 법률 및 절차와 관련된 정보가 포함된 것을 확인할 수 있었습니다. 이 연구는 한국 특화된 문화적 요구를 다루면서도 도움을 줄 수 있는 응답의 가능성을 보여주고 있습니다.



### TA-RAG: Tone-Aware Retrieval-Augmented Generation for Peer-Support Health Communication (https://arxiv.org/abs/2606.06794)
Comments:
          5 pages, 5 figures, CIKM 2026 submission manuscript

- **What's New**: 이번 논문은 민감한 피어 지원 건강 커뮤니케이션을 위해 사실 기반 생성(RAG) 이상의 기능을 요구하는 TA-RAG라는 새로운 프레임워크를 제안합니다. TA-RAG는 톤 조절을 명시적으로 통합하여 피어 지원의 효과성을 높이는 데 필요한 네 가지 핵심 구성 요소—낙인 없는 재작성, 읽기 용이성 조정, 수령인 맞춤화, 그리고 공감 재구성을 포함합니다. 이 연구는 TA-RAG가 모델 미세 조정 없이도 신뢰할 수 있는 문서에 기반을 둔 응답을 생성할 수 있도록 돕는다는 점에서 중요합니다.

- **Technical Details**: TA-RAG는 사용자의 질문을 명확히 하고 최상의 정보 조각을 검색한 후, LLM을 통해 근거 있는 응답 초안을 생성하는 세 가지 주요 단계를 거칩니다. 첫 번째 단계에서는 불분명한 질문을 명확히 하기 위해 사용자에게 추가 질문을 요청하고, 두 번째 단계에서는 검색된 정보를 바탕으로 초안을 작성합니다. 마지막으로, 이 초안에 톤 조정 레이어를 추가하여 응답을 비난이 없고, 읽기 쉬우며, 공감적이고 수령인에게 적합하게 만듭니다.

- **Performance Highlights**: TA-RAG는 HIV 관련 질문을 대상으로 한 구성 요소 수준의 테스트에서 응답의 질을 개선하는 성과를 보였습니다. 특히, 각 톤 조절 요소가 대상 의사소통의 질을 향상시키며 핵심 내용을 유지하는 데 기여함을 확인했습니다. 이러한 결과는 친구 지원 건강 커뮤니케이션에 적합한 RAG 출력 생성을 위한 가능성을 강조합니다.



### Explain Like I'm 5 or Whatever I Choose: Evaluating the Interactive Potential of Language Model Responses (https://arxiv.org/abs/2606.06788)
Comments:
          Preprint

- **What's New**: 이번 논문에서는 과학 정보 탐색(task)에서 대형 언어 모델(LLM)의 평가를 사용 중심으로 전환하는 새로운 프레임워크를 제안합니다. 기존의 단일 정적 사용자 인터페이스를 가정하는 평가 방식에서 벗어나, 다양한 인터페이스를 고려한 평가 기준으로 변화를 꾀하고 있습니다. 이 연구는 16명의 참가자와 함께한 형성 연구(formative study)를 기반으로 하여 언어 복잡성(language complexity)에 따라 여러 개의 응답을 생성할 수 있는 모델의 능력을 평가합니다.

- **Technical Details**: 제안된 평가 프레임워크의 핵심은 다양한 언어 복잡성을 가진 응답을 생성하여 사용자 선택 및 제어를 가능하게 하는 것입니다. 55개의 최신 모델(GPT-5.1, GPT-5 mini, Claude Sonnet 4.5 + Thinking, DeepSeek-V3.1)을 사용하여 98개의 과학 질의에 대해 5개의 응답을 생성하고, 각 응답의 복잡도를 평가하였습니다. 이 과정에서 언어의 직관적인 축을 따라 편차를 분석하여 모델 성능을 평가합니다.

- **Performance Highlights**: 연구 결과, 모델들은 응답의 복잡성을 다양하게 조정할 수 있지만, 그 변동성이 일관되지 않음을 발견했습니다. 특히, 최고의 성능을 보인 모델(Claude Sonnet 4.5)은 복잡성을 올바른 방향으로 조정하는 데 성공한 비율이 46%에 불과했습니다. 이러한 결과는 샘플의 규모를 늘리고 대안적인 복잡성 수준에서도 유사하게 유지되었으며, 여러 버전 간의 복잡성 관계를 평가할 수 있는 기준을 제시합니다.



### When Better Codebooks Are Not Enough: Predictive Performance and Behavioral Reliability in LLM Political Event Coding (https://arxiv.org/abs/2606.06781)
Comments:
          14 pages, 3 figures, 11 tables

- **What's New**: 이번 연구에서는 LLM(대형 언어 모델)이 전문가가 작성한 코드북을 직접 적용하여 정치 사건 코딩(Political Event Coding)에서 발생하는 문제점을 탐구합니다. 연구팀은 코드북의 정의와 예시를 명확하게 하고 LLM에서 더 효과적으로 분류할 수 있는 방식을 제시했습니다. 이를 통해 코드북의 구조와 내용이 어떻게 모델의 성능에 영향을 미치는지 분석했습니다.

- **Technical Details**: 정치 사건 코딩은 뉴스 보고서를 정치적 상호작용의 구조화된 기록으로 변환하는 과정입니다. 본 논문에서는 특정 소스-타겟(actor) 쌍을 주어진 텍스트에서 추출하여 정치적 행동을 예측하는 관계 분류(relation classification) 문제에 초점을 맞추었습니다. PLOVER라는 코드북을 활용하여 사건의 유형을 이진, 쿼드 및 루트 수준의 레이블로 평가하고, 각 레이블이 모델에 미치는 영향을 분석합니다.

- **Performance Highlights**: 코드북에서 명확한 정의와 세부 규칙을 적용함으로써 LLM의 예측 성능이 크게 향상되었습니다. 그러나 향상된 예측 성능이 반드시 행동 신뢰성(behavioral reliability)으로 이어지지는 않음을 보여주었습니다. 이는 코드북이 제공하는 일관된 규칙을 얼마나 잘 따르는지가 성능의 중요한 결정 요소임을 나타냅니다.



### A Four-Condition Diagnostic Protocol for Evidence Utilization in Long-Context and Retrieval-Augmented Language Models (https://arxiv.org/abs/2606.06758)
Comments:
          52 pages, 34 tables, 1 figure

- **What's New**: 이 논문은 긴 문맥(long-context) 언어 모델 및 검색 보강(retrieval-augmented) 시스템의 증거 활용을 진단하기 위해 네 가지 조건을 갖춘 프로토콜을 제안합니다. 단순한 최종 정답 정확도(final answer accuracy)와 검색 재현율(retrieval recall), 인용 겹침(citation overlap)만으로는 모델의 증거 사용 여부를 확인할 수 없습니다. 연구 결과는 전통적으로 평가되어온 방식과 다른 접근 방식을 제시하며, 모델이 어떻게 증거를 활용하는지에 대한 깊은 통찰을 제공합니다.

- **Technical Details**: 연구에서는 네 가지 증거 가용성 조건인 증거 없음(no evidence), 전체 문맥(full context), 검색된 증거(retrieved evidence), 오라클 증거(oracle-evidence reference)를 설정하고, 이러한 조건에 따라 반응하는 모델의 행동을 분석합니다. ONCU는 회복된 오라클 참조 증거(adjusted oracle-reference evidence advantage)를 추정하기 위한 프로토콜에 기반한 추정기로 사용됩니다. 이 방법은 동일한 모델과 예제, 점수 필드(score field) 및 그룹화 방침(grouping scheme) 하에 서로 다른 증거 역할을 공동으로 관찰할 수 있게 해줍니다.

- **Performance Highlights**: 최종 결과에 따르면, 제어된 합성 설정은 주로 전체 문맥 활용 실패를 드러내며, 테스트된 현실적인 다단계 증거 설정은 주로 검색 체인 커버리지 실패를 노출합니다. 본 연구는 다양한 모델(Qwen, Gemma 등)에서 18,000개의 ONCU 호환 예측을 통해 이 결과를 입증하였습니다. 새로운 진단 프로토콜은 모델 개발자와 평가자들이 결과의 기초가 되는 메커니즘을 구별하도록 도와줍니다.



### PromptPrint: Behavioral Biometrics Through Natural Language Prompting in LLMs (https://arxiv.org/abs/2606.06755)
Comments:
          10 pages, 6 figures

- **What's New**: 이 논문은 PromptPrint라는 새로운 연구를 소개하며, 이는 언어 모델(LLMs)와의 상호작용에서 사용자의 정체성을 파악하기 위한 체계적인 접근법이다. 사용자들이 사용하는 언어의 습관적 어휘, 구문, 담화 패턴이 학습 가능한 행동 생체인식(biometrics)을 형성한다는 가설을 제안한다. 연구 결과, 표면적인 단어 선택이 정체성을 효과적으로 암호화하며, 강한 식별 능력을 나타내는 것을 발견했다.

- **Technical Details**: PromptPrint는 사용자 프롬프트에서 얻어낸 특징 벡터를 사용하여 정체성 확인을 위한 프로토타입 벡터를 구성한다. 연구에서 사용된 데이터는 WildChat-1M이라는 실세계에서 수집된 데이터셋으로, 1,034명의 사용자로부터 총 20,680개의 실제 프롬프트를 포함하고 있다. 또한, 다양한 특징 공간을 평가하여 행동 신호의 안정성을 분석하고, 세 가지 적대적 공격을 통해 프롬프트의 생체인식 안정성을 평가하였다.

- **Performance Highlights**: 연구는 존재하는 통계적 특징과 신호 감지 이론을 통해 사용자가 고유하고 일관된 스타일을 유지하지 못함을 보여주며, 정체성을 암호화하는 데 있어 언어적 표면 수준에서 신호가 모여 있는 것을 강조한다. 결과적으로, 프롬프트 기반의 정체성 신호가 강력하게 식별 성능을 발휘하여 행동 생체인식의 가능성을 확인하였다. 이 연구는 LLM 상호작용에서의 사용자 모델링에 대한 새로운 시각을 제공하며, 보안과 개인 정보 보호에 중요한 함의를 가진다.



### Evidence Graph Consistency in Retrieval-Augmented Generation: A Model-Dependent Analysis of Hallucination Detection (https://arxiv.org/abs/2606.06748)
Comments:
          Accepted at the International Conference on Advanced Machine Learning and Data Science; to appear in the IEEE Xplore proceedings

- **What's New**: 이번 논문에서는 Evidence Graph Consistency (EGC)라는 새로운 프레임워크를 제안하여, RAG(Retrieval-Augmented Generation)에서 발생하는 허위정보(hallucination)를 감지하는 방식을 개선합니다. 기존의 방법들이 주로 생성된 답변과 조회된 내용 간의 단순 유사성을 평가하는 데 집중한 반면, EGC는 각 응답에 대한 지역 증거 그래프를 구성하고 구조적 일관성 지표를 계산하여 이러한 허위정보의 신호를 파악합니다. EGC의 적용을 통해 Llama-2 모델에는 기대하는 진단 방향이 나타났지만, GPT-4와 GPT-3.5 모델에서는 상반된 결과가 나타나는 것으로 확인되었습니다.

- **Technical Details**: Evidence Graph Consistency(EGC)는 질문, 조회된 구절, 그리고 생성된 답변 간의 관계를 그래프로 구축하여 구조적 일관성을 특징으로 추출합니다. 이 그래프는 세 종류의 노드를 포함하며, 이를 통해 Euclidean 공간 내에서 cosine similarity를 사용하여 노드 간의 관계를 정의합니다. 또한 다섯 가지의 구조적 일관성 지표(coverage, support density, cross-evidence agreement, connectivity, isolation penalty)를 계산하여, 각각의 노드 및 엣지 간의 유기적 연결성을 평가합니다.

- **Performance Highlights**: RAGTruth 데이터셋에 대한 평가를 통해 EGC는 5,767개의 응답에서 모델 간 일관된 분할을 보여주었습니다. 특히 구조적 일관성이 허위정보 감지에 있어서 모든 모델에서 동일하게 유효하지 않으며, 특정 모델군 간의 허위정보 패턴이 질적으로 다르다는 점이 강조되었습니다. 이러한 결과는 향후 허위정보 감지 시스템의 설계 시 모델 의존성을 고려해야 함을 시사합니다.



### When to Think Deeply: Inhibitory Deliberation for LLM Reasoning (https://arxiv.org/abs/2606.06745)
- **What's New**: 이 논문은 IDPR(Inhibitory Deliberative Problem Reasoning)라는 새로운 프레임워크를 제안합니다. IDPR는 응답 조건부 억제 사고(응답에 따라 느린 사고를 사용할지 결정하는 기능)를 통해 문제 해결 성능을 향상시킵니다. 이 시스템은 먼저 간결하고 직관적인 답변을 생성한 후, 그 답변이 발송될지 억제될지를 판단합니다.

- **Technical Details**: IDPR는 빠른 정책을 통해 직관적인 답변을 유도하고, 억제 제어기를 통해 그 답변이 느린 정책을 사용할 필요가 있는지를 평가합니다. 이는 신뢰도, logit margin, 구문 분석 가능성, 생성 비용 등의 빠른 증거에 기반하여 판단합니다. 이러한 접근 방식은 특히 수학적 문제 해결에서 느린 사고를 선택적으로 사용할 수 있게 해줍니다.

- **Performance Highlights**: IDPR는 5,000개의 수학 문제 집합에서 느린 사고를 8.20%의 경우에만 호출하고, 정확성을 47.90%에서 48.92%로 향상시킵니다. 랜덤 라우팅은 정확성이 46.76%로 떨어지는 반면, IDPR는 최고 수준의 정정 정확도인 27.07%를 기록하며, 빠른 답변이 느린 사고로부터 이득을 보는 경우를 더욱 잘 식별함을 보여줍니다.



### Modular Monolingual Adaptation using Pretrained Language Models (https://arxiv.org/abs/2606.06738)
Comments:
          Accepted to ACL 2026 Industry Track

- **What's New**: 본 연구는 저자들이 다국어 프리트레인드 언어 모델(PMLM)의 모듈화된 적응 방식을 제안함으로써, 저자원 언어에 대한 단일 언어 모델을 효율적으로 만들 수 있는 새로운 접근 방식을 제공합니다. 특히, 기존의 전체 모델 조정이 필요하지 않음을 주장하며, 토큰을 교체하고 해당 임베딩을 고정시키는 방안을 채택합니다. 이를 통해 저자원 언어에 대한 성능 향상을 보여줍니다.

- **Technical Details**: 제안된 방법은 BERT와 mBERT와 같은 인코더 기반 모델을 활용하며, 훈련 데이터의 희소성으로 인한 과적합을 방지하기 위해 입력 임베딩과 출력 임베딩을 동결합니다. 또한, 텍스트 코퍼스와 같은 훈련 자료에서 어휘(Lexicon)를 기반으로 개인 맞춤형 토크나이저를 훈련시켜 효율성을 높입니다. 이 과정에서 배치 크기와 하이퍼파라미터는 사용하는 GPU에 따라 조정됩니다.

- **Performance Highlights**: 실험 결과, 저자들은 분류된 여러 데이터 세트에서 비임베딩 조정(non-embedding tuning) 방법이 전체 조정(full tuning) 전략보다 종종 더 나은 성능을 보였다고 보고합니다. 특히, 사용자 맞춤형 토크나이저를 활용하는 것이 다국어 토크나이저에 비해 더 큰 성능 향상을 이루었으며, Quichua와 같은 매우 저자원 언어의 경우로부터 중요한 결과를 도출했습니다.



### Does Topic Sentiment Cause Perceived Ideology? Comparing Human and LLM Annotations in Political News Articles (https://arxiv.org/abs/2606.06715)
Comments:
          Accepted to ACL SRW 2026

- **What's New**: 이 논문에서는 주제 감정(topic sentiment)이 인지된 정치 이념(perceived political ideology)에 인과적인 영향을 미치는지 살펴보고, 그 결과가 이념 레이블을 부여하는 주체에 따라 달라지는지를 분석합니다. AllSides의 기사들과 Llama-3.3-70b-versatile로부터의 감정 주석을 기반으로 하여, 전문가와 GPT-4o-mini(기본형 및 세밀 조정된 형), Llama-3.3-70B 간의 이념 레이블을 비교했습니다. 연구 결과, 인간 주석자들은 지역 사회 수준에서 유의미한 인과 효과를 보이지 않았지만, 세밀 조정된 GPT-4o-mini는 가장 높은 분류 정확도를 달성했습니다.

- **Technical Details**: 인과 추정(causal inference) 방법론인 더블 머신 러닝(Double Machine Learning, DML)을 통해 주제 감정이 LLM 주석자들에 의해 예측된 이념에 중요한 인과 효과를 미친다고 밝혔습니다. 세밀 조정된 GPT-4o-mini 모델에서 감정-이념 간의 유의미한 자연 직접 효과(natural direct effects, NDEs)가 나타났으며, 이는 다른 세 가지 주석자 모델에서는 발견되지 않았습니다. 이러한 결과는 LLM이 특정 주제 그룹에 따라 감정과 텍스트적 특징 간의 관계를 더 밀접하게 연결짓는다라는 점을 시사합니다.

- **Performance Highlights**: 실험을 통해 세밀 조정된 GPT-4o-mini 모델이 F1 점수 72.48로 가장 높은 분류 정확도를 보였으며, 인간 주석자와 비교할 때 중요한 인과 효과를 생성하는 유일한 모델로 확인되었습니다. 이 연구는 LLM이 감정 주석을 처리하는 방식이 인간의 판단과는 다르게 작동하며, 이는 인과 분석에 있어 LLM이 생성한 레이블을 정황적(silver) 레이블로 사용함에 있어 중요한 함의를 지닙니다. 결론적으로, 세밀 조정된 LLM의 감정-이념 예측은 인간의 주석과는 다르게 동작하며, 이는 향후 사회 과학 연구에서 LLM의 사용 방식을 재고하게 할 만한 요소입니다.



### Data-Efficient Autoregressive-to-Diffusion Language Models via On-Policy Distillation (https://arxiv.org/abs/2606.06712)
- **What's New**: 이번 연구에서는 자기 회귀 모델(ARLM)에서 확산 언어 모델(DLM)로의 변환을 탐구합니다. 기존 연구들은 ARLM의 인과 주의를 양방향 주의로 교체하고 이후 DLM 목표로 모델을 훈련시켰으나, 두 가지 분포 변화(distribution shift)가 발생합니다. 이 논문에서는 On-Policy Diffusion Language Model(OPDLM)을 도입하여 이러한 문제를 해결합니다.

- **Technical Details**: OPDLM은 자기-OPD(self-OPD)를 이용하여 훈련되며, 학생 모델인 양방향 주의를 가진 ARLM이 자체적으로 경로(trajectories)를 생성합니다. 교사 모델은 원래 동결된 ARLM으로, 이러한 경로에 대한 목표 로그it(target logits)을 제공하여 지식을 증류(distillation)합니다. 이 과정에서는 DLM에서 흔히 발생하는 훈련-추론 불일치(train-inference mismatch)를 제거할 수 있습니다.

- **Performance Highlights**: 경험적 결과에 따르면 OPDLM은 15배에서 7,000배 더 적은 훈련 토큰을 필요로 하면서도 다양한 작업에서 강력한 성능을 보입니다. 또한 OPDLM은 DLM의 사전 훈련(pretraining) 비용을 피하며, ARLM 후 훈련(post-training)의 형태로 DLM 변환을 가능하게 합니다.



### Signal-Driven Observation for Long-Horizon Web Agents (https://arxiv.org/abs/2606.06708)
Comments:
          10 pages, 1 figure

- **What's New**: 이 논문에서는 웹 에이전트가 긴 작업을 수행할 때 직면하는 관찰 과잉 섭취(observer over-ingestion) 문제를 새로운 방식으로 접근합니다. 저자들은 매번 행동을 수행할 때마다 전체 DOM을 읽는 하드웨어 설계를 비판하며, 잘못된 아키텍처로 인한 문제를 강조합니다. 이를 해결하기 위해 신호 기반 관찰(Signal-Driven Observation, SDO)라는 방법을 제안하며, 이는 필요할 때만 DOM을 읽도록 최적화된 구조입니다.

- **Technical Details**: 제안된 SDO는 웹 페이지의 전체 DOM을 읽는 전용 서브 콜(sub-call)을 도입하지만, 모든 요소를 반환하는 것이 아니라 특정 작업과 관련된 요소와 선택자만 반환합니다. 이 방법은 URL 변경, 새로 표시된 인터랙티브 요소, 행동 실패, 외부 브라우저 이벤트와 같은 경량 신호 감지기에 의해 호출됩니다. 이를 통해 에이전트는 불필요한 정보를 다시 읽지 않고도 작업을 수행할 수 있습니다.

- **Performance Highlights**: SDO 방식의 도입을 통해 관찰 과잉 섭취 문제를 해결함으로써 에이전트의 성능을 개선할 수 있습니다. 저자들은 관찰 압축(observation compression)을 에이전트 시스템 디자인에서 핵심 고려 사항으로 삼을 것을 제안하며, 이는 하향식 실패를 완화하는 데 중요한 역할을 할 것입니다. 이러한 방식으로 에이전트의 의사결정 과정이 더 효율적이고, 일관성 있게 유지될 것으로 기대됩니다.



### HKJudge: A Legal Discourse-Annotated Corpus for Interpreting What Courts Find, How They Reason, and What They Ru (https://arxiv.org/abs/2606.06679)
- **What's New**: 본 연구는 홍콩에서 처음으로 법원 판결의 담론 분석을 위한 전문가 주석이 달린 데이터셋인 HKJudge를 소개합니다. 이 데이터셋은 홍콩의 법원 계층 구조에 따라 5개 수준의 형사 판결을 포함하여 총 29만개 문장과 650만개의 토큰으로 구성되어 있습니다. 두 계층의 담론 체계를 설계하여 법원이 발견하는 사실, 추론 방식, 판결 내용을 포착합니다. 또한, 두 가지 작업인 수사적 역할 분류(rhetorical role classification)와 법적 요소 추출(legal element extraction)을 공식화하고, BERT 기반 모델 및 상업적 LLM의 평가를 제공합니다.

- **Technical Details**: HKJudge 데이터셋은 3개의 스팬 수준 요소와 26개의 문장 수준 수사적 역할 클래스를 포함하는 법적 담론 체계를 개발합니다. 법적 언어학 전문가 10명이 약 148,600개의 스팬 및 292,240개의 수사적 역할 주석을 제공했으며, 전문가 간 동의 수준이 κ = 0.8에 달합니다. 데이터셋은 1968년부터 2024년까지의 4,000개 이상의 판결을 수집하기 위해 웹 스크랩퍼를 구축하였으며, 모든 법원에 대한 자료를 포함합니다.

- **Performance Highlights**: 다양한 BERT 기반 모델 및 LLM 모델을 평가한 결과, 전문가 주석자에 비해 모든 LLM 모델이 여전히 부족함을 강조합니다. 특히 미세 조정(fine-tuning)을 통해 성능이 크게 향상되었지만, 여전히 인간 전문가의 주석을 초과하지는 못합니다. 이러한 결과는 법적 LLM 추론에서 전문가 주석의 중요성을 부각시키며, 향후 연구 및 개발의 방향성을 제시합니다.



### What Do People Actually Want From AI? Mapping Preference Plurality (https://arxiv.org/abs/2606.06674)
Comments:
          Accepted at the 2026 ACM Conference on Fairness, Accountability, and Transparency (FAccT '26)

- **What's New**: 이번 논문은 대형 언어 모델(Large Language Models, LLMs)의 조정 방법인 인간 피드백을 통한 강화 학습(Reinforcement Learning from Human Feedback, RLHF)의 한계점을 분석하고, AI 시스템에 대한 사람들의 실제 요구를 조사하는 것을 목표로 합니다. 연구는 75개국에서 수집한 1500개의 개방형 응답(PRISM 데이터셋)을 분석하여, 다양한 사람들의 가치관의 불일치와 RLHF가 이를 어떻게 포착하지 못하는지를 강조합니다. 흥미롭게도, 응답자의 49%가 요청하는 '진실성(truthfulness)'조차도 상이한 정의로 논의되는 등의 사례를 보여주며, 이는 현재의 조정 모델의 근본적인 결함을 드러냅니다.

- **Technical Details**: 연구에서는 정량적 및 정성적 분석을 통해 PRISM 데이터셋의 복잡한 사용자 요구를 파악했습니다. 비계층적 및 개방형 데이터를 활용하여 RLHF의 모델이 어떻게 특정 사용자의 의도를 간과하는지를 보여주며, 많은 응답이 사실은 각기 다른 해석을 갖고 있음을 드러내고 있습니다. 독립적인 가치와 선호의 갈등에 대한 분석을 통해, AI 시스템의 조정 결정이 사회의 다양한 가치관을 포함하지 못하고 있다는 점을 시사합니다.

- **Performance Highlights**: 이 연구는 RLHF 방식의 한계를 시사하며, 고자산 모델에서도 높은 허구율(hallucination rates)이 지속되고 있음을 보여줍니다. 이는 사용자들이 명확히 요구하는 정확성을 충족하지 못하고 있다는 것을 나타냅니다. 결과적으로, 사용자의 다변화된 선호를 반영하지 않고 단일한 보상 모델을 적용하는 것은 비효율적이며, 사회적으로 공정한 AI 시스템 구현이 어렵다는 것을 명확히 합니다.



### The Piggyback Hypothesis of Generalization: Explaining and Mitigating Emergent Misalignmen (https://arxiv.org/abs/2606.06667)
- **What's New**: 본 연구는 LLM(대규모 언어 모델)의 좁은 작업에 대한 파인튜닝(finetuning)이 의미적으로 관련 없는 테스트 도메인에 대한 광범위한 비정합 발현(emergent misalignment, EM)을 유도할 수 있음을 보여줍니다. 이 연구에서는 Piggyback Hypothesis(피기백 가설)을 제안하여, 채팅 템플릿 토큰이 사용자 쿼리와 무관한 도메인으로 학습된 행동을 전이할 수 있음을 입증합니다. 연구 결과는 미세하게 조정된 접두사(prefix)에서의 변화가 잘못된 모델의 정합성을 회복할 수 있음을 시사합니다.

- **Technical Details**: 이 연구는 Token-Regularized Finetuning (TReFT)이라는 새로운 훈련 방법을 사용하여 EM을 완화하는 방안을 제시합니다. TReFT는 파인튜닝 과정에서 특정 토큰 표현을 정규화하여, 모델이 도메인 관련 의미에 맞춰 학습할 수 있도록 유도합니다. 이후 여러 LLM 모델과 EM 유도 데이터셋에서 TReFT가 EM을 줄이면서도 도메인 내 학습을 유지할 수 있음을 확인하였습니다.

- **Performance Highlights**: Llama-3.1-8B 모델을 법률 도메인에 파인튜닝한 결과, TReFT는 데이터 간섭(data interleaving) 방식보다 33.5% 더 나은 EM 감소 효과를 보였습니다. TReFT는 또한 비정합하면 특정 행동(예: 도구 사용, 거부 등)을 학습하는 데도 적용 가능하여, 평균적으로 비의도적인 일반화를 54.3% 감소시켰습니다. 이러한 연구 결과는 LLM이 예상치 못한 방식으로 학습하고 일반화할 수 있음을 보여줍니다.



### CAF-Gen: A Multi-Agent System for Enriching Argumentation Structures (https://arxiv.org/abs/2606.06646)
Comments:
          Accepted for publication in the proceedings of ICCCI 2026

- **What's New**: 이 논문에서는 복잡한 추론을 자연어 텍스트에서 정형화하는 문제를 다루고 있습니다. 기존의 Argument Mining(AM) 기법들은 기본적인 주장(claim)과 전제(premise)를 식별하는 데에 집중했으나, Carneades Argumentation Framework (CAF)와 같은 고급 구조 정보를 포착하는 데 어려움을 겪고 있었습니다. 이러한 한계를 극복하기 위해, CAF-Gen이라는 자동화된 다중 에이전트 프레임워크가 도입되어 점진적으로 얕은 주장을 CAF 준수 모델로 보강합니다.

- **Technical Details**: CAF-Gen은 Creator-Reviewer 파이프라인을 사용하여, Creator 에이전트의 출력을 Reviewer 에이전트가 검증하여 구조적 무결성을 보장합니다. 이 구조가 단일 통과 생성 모델의 전형적인 구조적 불안정을 완화하는 데 중요합니다. 이 프레임워크는 입력된 기본 주장 구조를 CAF 접근 방식의 복잡한 특성으로 풍부하게 변형합니다.

- **Performance Highlights**: 실험 결과, 반복적인 피드백 루프가 생성된 데이터의 품질을 향상시키고 원본 주석(original annotations)과의 강한 정렬을 달성했음을 보였습니다. CAF-Gen 시스템은 단일 통과 생성의 한계를 극복하고 형식적인 주장을 자동 모델링하는 신뢰할 수 있는 방법론을 제공하는 데 성공했습니다.



### How Language Models Fail: Token-Level Signatures of Committed and Persistent Reasoning Failures (https://arxiv.org/abs/2606.06635)
- **What's New**: 본 논문에서는 언어 모델의 추론 실패를 측정하는 새로운 프레임워크를 제안합니다. 이는 토큰 수준의 불확실성 신호를 활용하여 두 가지 근본적으로 다른 실패 모드를 식별합니다: "committed failure"와 "persistent uncertainty"입니다. 이 프레임워크는 23개의 모델-데이터셋 구성에서 그 예측이 유효임을 입증하였으며, 특정 시점에서 모델이 잘못된 경로에 고착되는 "commitment point"를 정의합니다.

- **Technical Details**: 제안된 프레임워크는 모델의 추론 과정에서 토큰 수준의 불확실성 신호를 분석하여 실패가 발생하는 과정을 정량화합니다. 'committed failure'에서는 모델이 잘못된 추론 경로에 고착된 이후 불확실성이 증가하지 않고, 불확실성 신호가 최대인 지점을 'commitment point'로 정의합니다. 반면, 'persistent uncertainty'에서는 모델이 지속적으로 불확실성을 유지하며, 모든 토큰의 추론이 성공 또는 실패를 식별하는 데 필수적입니다.

- **Performance Highlights**: 이 프레임워크는 다양한 모델과 데이터셋을 통해 두 가지 실패 모드를 성공적으로 식별하며, 20/23개의 사례에서 그 예측의 유효성을 입증했습니다. 또한, 이 연구는 언어 모델의 자기 일관성(self-consistency)과의 관계를 명확히 하여, 불확실성 신호가 언제 자기 일관성과 상호 보완적으로 작용하는지 예측할 수 있는 가능성을 보여줍니다. 이러한 결과는 LLM(대형 언어 모델)에서의 추론 실패를 감지하고 적절한 대응 전략을 개발하는 데 도움을 줄 수 있습니다.



### UnpredictaBench: A Benchmark for Evaluating Distributional Randomness in LLMs (https://arxiv.org/abs/2606.06622)
- **What's New**: 본 논문에서는 UnpredictaBench라는 새로운 평가 기준을 도입하였습니다. 이 평가는 대규모 언어 모델(LLMs)이 실제 분포를 얼마나 잘 포착하는지를 테스트합니다. LLM이 경제적 시뮬레이션에서 인간을 대체하는 데 사용됨에 따라, 그들이 제공하는 출력의 다양성은 불충분하며, 현실의 불확실성을 반영해야 한다고 강조합니다.

- **Technical Details**: UnpredictaBench는 LLM이 단일 출력 분포에서 샘플링할 수 있는 능력을 평가하기 위해 448개의 문제로 구성된 벤치마크입니다. 새로운 메트릭인 KS@N은 Kolmogorov-Smirnov 통계 테스트를 기반으로 모델이 출력한 샘플이 실제 분포와 얼마나 잘 일치하는지를 측정합니다. 이 접근 방식은 여러 개의 표준 분포와 자연어 시나리오를 포함한 다양한 샘플을 고려합니다.

- **Performance Highlights**: 다양한 오픈 및 상용 모델에 대한 평가 결과, 샘플 크기 100(KS@100)에 대한 모델의 점수는 0%에서 20% 이상까지 분포했습니다. 어느 모델도 KS@100에서 40%를 초과하는 성능을 보이지 않았으며, 이는 LLM이 여전히 분포 샘플링에서 상당한 개선 여지를 가지고 있음을 시사합니다. 특정 모델은 상대적으로 좋은 성과를 보였지만, 대부분은 분포에 대한 정확성이 낮아 문제가 있음을 보여줍니다.



### Re-Centering Humans in LLM Personalization (https://arxiv.org/abs/2606.06614)
- **What's New**: 이 논문에서는 기존의 합성 데이터(synthetic data)에 의존했던 대형 언어 모델(LLMs)의 개인화(personalization) 능력을 평가하는 연구의 틈새를 분석합니다. 실제 사용자 대화 및 판단을 통해 시스템의 개인화 성능 차이를 조사하고, 550개의 대화와 5,949개 사용자 특성 판단을 수집했습니다. 연구 결과, LLMs는 인간 대화에서 사용자 특성을 추출하는 데 어려움을 겪고, 관련 특성을 신규 프롬프트와 연결하는 데 있어 인간의 판단과 불일치하는 경향이 있다는 것을 보여줍니다.

- **Technical Details**: 논문에서는 세 단계로 구성된 개인화 파이프라인을 제시합니다: (1) 대화에서 사용자 특성 추출 (user attribute extraction), (2) 현재 상호작용 맥락에 맞는 특성의 적합성 매칭 (attribute relevance matching), (3) 선택된 특성에 기반하여 개인화된 응답 생성 (personalized response generation). 연구에서는 각 단계에서의 모델 성능을 비교하고, 인간의 데이터와 판단과의 정렬도를 측정합니다. RoBERTa와 같은 경량 검증 모델을 사용하여 특성 추출 단계의 문제를 해결하기 위한 두 가지 개입을 소개합니다.

- **Performance Highlights**: 연구에서 인간 데이터를 포함시키니 모델의 한계가 드러났습니다. 예를 들어, LLM은 인간 대화에서 22%의 사용자 특성을 잘못 추출하였고, 적합성 매칭에서 인간의 기준과 불일치하는 경우가 20-40%에 달했습니다. 또한, 인간은 LLM의 개인화된 응답이 일반적인 응답보다 효과적이라고 판단하지 않는 경우가 54.6%에 이르는 등, 개인화에 대한 어려움을 강조하였습니다. 이러한 결과는 자동화된 개인화 평가가 인간의 데이터에 더 가깝도록 만드는 것이 매우 중요함을 시사합니다.



### Improving Cross-Lingual Factual Recall via Consistency-Driven Reinforcement Learning (https://arxiv.org/abs/2606.06586)
Comments:
          Under Review at EMNLP 2026

- **What's New**: 이 논문에서는 cross-lingual factual inconsistency 문제를 해결하기 위해 PolyFact라는 대규모 다국어 사실 QA 데이터셋을 소개합니다. PolyFact는 12개 언어에 걸쳐 100K의 Wikidata 기반 사실을 포함하며, 이를 통해 경량 지속적 재훈련(Continual Pretraining, CPT), 지도적 미세 조정(Supervised Fine-Tuning, SFT), 그리고 그룹 상대 정책 최적화(Group Relative Policy Optimization, GRPO)의 효과를 비교합니다. 연구 결과 GRPO가 SFT를 지속적으로 초과 이해력과 일관성을 향상시켰음을 보여줍니다.

- **Technical Details**: PolyFact 데이터셋은 위키데이터의 진리 삼중관계를 기반으로 12개 언어로 레이블을 추출하여 만들어졌으며, 이는 다국어 품질 보증을 위해 사용됩니다. 90% 이상의 LLM-인간 일치를 가진 사실들을 포함하며, Ted2025 데이터셋을 통해 지속적 재훈련을 수행합니다. GRPO를 적용하여 다국어 QA의 정확도를 높이기 위해 12개의 언어에서 독립적으로 생성된 답변을 사용하고, 각 언어에 대해 독립적인 훈련 유도를 수행합니다.

- **Performance Highlights**: 실험 결과에 따르면 GRPO는 SFT보다 월등한 성능을 보여 cross-lingual factual consistency를 향상시키며, 보지 못한 언어에 대한 일반화 능력도 개선됩니다. 계속해서 CPT는 교차 언어 사실 회상에 대한 제한된 이득을 제공하며, 잠재적 사실 지식 접근을 개선하기 위한 보다 직관적인 접근 방식으로 다국어 QA를 제안합니다. 이 연구 결과는 다국어 모델의 성능을 개선하는 새로운 경로를 제시하며, 코드, 모델 및 데이터셋이 공개됩니다.



### MemDreamer: Decoupling Perception and Reasoning for Long Video Understanding via Hierarchical Graph Memory and Agentic Retrieval Mechanism (https://arxiv.org/abs/2606.07512)
- **What's New**: 현재의 Vision-Language Models는 몇 시간에 걸친 비디오를 처리하는 데 어려움을 겪고 있습니다. 이를 해결하기 위해, 연구진은 MemDreamer를 소개하며, 이는 인식과 추론을 분리하여 긴 비디오 이해를 에이전틱 탐색 과정으로 전환합니다. MemDreamer는 비디오를 점진적으로 스트리밍하며 Hierarchical Graph Memory를 구축하는 플러그 앤 플레이(framework) 구조입니다.

- **Technical Details**: 이 프레임워크는 의미 추상화를 위한 세 가지 주요 계층으로 구성된 구조로, 시공간(spatiotemporal) 및 인과관계(causal relations)를 포착하는 기본 그래프에 의해 뒷받침됩니다. 추론 과정에서는 Agentic Tool-Augmented Retrieval을 사용하여 계층을 내비게이션하고, 노드를 탐색하며, Observation-Reason-Action 루프를 통해 논리적인 간선을 통과합니다.

- **Performance Highlights**: 실험 결과, MemDreamer는 네 가지 주요 벤치마크에서 SOTA(State Of The Art) 결과를 달성하였고, 인간 전문가와의 격차를 단 3.7점으로 좁혔습니다. 이 모델은 전체 맥락의 2%로 추론 컨텍스트 윈도우를 제한하면서도 12.5점의 절대 정확도 증가를 보여줍니다. 또한, 통계적 분석을 통해 VLM의 논리 추론 성능과 긴 비디오 이해 벤치마크 간에 강한 양의 선형 상관관계를 발견하였습니다.



### TEVI: Text-Conditioned Editing of Visual Representations via Sparse Autoencoders for Improved Vision-Language Alignmen (https://arxiv.org/abs/2606.07451)
Comments:
          20 pages, 13 figures, 14 tables

- **What's New**: 이 논문에서는 TEVI라는 새로운 프레임워크를 제안합니다. TEVI는 자막이 이미지 임베딩에서 무엇을 유지해야 하는지를 결정하는 신호로 작용합니다. 이를 통해 자막에 설명된 속성은 유지하고 다른 속성은 버리도록 설정할 수 있습니다. 이 방법은 이미지-텍스트 정렬을 개선하는 데 효과적입니다.

- **Technical Details**: TEVI는 희소 자동인코더(sparse autoencoders)를 사용하여 이미지 임베딩을 구성 요소 개념으로 분해하고, 조건부 모듈을 훈련하여 재구성에 사용할 SAE 잠재 변수를 선택합니다. 이를 통해 이미지의 특정 속성을 살리고, 필요한 정보를 유지하면서 다른 속성에 대한 정보를 폐기할 수 있습니다. 또한, 제어된 실험을 통해 TEVI의 효과를 증명했습니다.

- **Performance Highlights**: TEVI를 자연 이미지에 대해 훈련된 CLIP 모델에 적용한 결과, 짧은 자막(MS COCO, Flickr)과 긴 자막(DOCCI, IIW) 벤치마크 모두에서 검색 성능이 향상되었습니다. 특히 긴 자막 기준에서 더 큰 성능 향상을 보였으며, 이것은 풍부한 자막이 수정에 강력한 신호를 제공한다는 것을 시사합니다. 또한, RoCOCO 벤치마크에서 언어적 왜곡에 대한 강한 견고함을 보여주었습니다.



### The Lipreading Gap: Do VSR Models Perceive Visual Speech Like Human Lipreaders? (https://arxiv.org/abs/2606.07435)
Comments:
          Accepted at INTERSPEECH 2026

- **What's New**: 최근 Visual Speech Recognition (VSR) 모델은 인간의 lipreader를 초월하는 성능을 보이고 있으나, 이러한 성장이 과연 인간과 유사한 시각적 언어 인식을 확립하는지를 탐구하고 있습니다. 이 연구에서는 세 가지 VSR 시스템을 MaFI 데이터셋을 통해 인간의 성능과 비교하여, 단순한 문자의 정확도가 아닌 단어, 글자, 음소, 그리고 viseme 수준에서 성능 분석을 실시하였습니다. 모델이 전반적으로 더 높은 정확도를 달성하였지만, 특정 단어에서 인간과 모델 간의 차이가 존재하여, VSR의 성격을 보다 명확히 이해할 수 있는 기회를 제공합니다.

- **Technical Details**: VSR 시스템의 분석은 MaFI 데이터셋을 사용하여 진행되었고, 이 데이터셋은 410명의 참가자들에 의해 수집된 실험 결과를 기반으로 합니다. 각 단어의 Mouth and Facial Informativeness (MaFI) 점수를 활용하여, 참가자들이 lipreading을 통해 단어를 식별하는 데 필요한 시각적 정보의 다양성을 평가했습니다. 세 가지 모델로는 Auto-AVSR (Supervised Learning), AV-HuBERT (Self-Supervised Learning), VSP-LLM (Large Language Model Decoder)이 있으며, 이러한 모델들은 모두 공개된 가중치로 평가되었습니다.

- **Performance Highlights**: 이 연구는 VSR 모델과 인간 간의 인식 정확도 차이를 명확히 드러내며, 특히 모델의 오류는 시각적 정보보다는 훈련 빈도에 더 큰 영향을 받는 것으로 나타났습니다. VSR 모델은 인간이 인지하는 시각적 명확성과의 강한 상관관계를 보이지 않았고, 이는 기본적으로 다른 언어 처리 메커니즘을 의미합니다. 전통적인 방법에 비해, 이 연구는 인간과 VSR 모델 간의 인식의 정밀한 일치를 위한 기초 데이터를 제공하며, 향후 VSR 기술 개선에 기여할 수 있는 중요한 통찰력을 제공합니다.



### Do Coding Agents Deceive Us? Detecting and Preventing Cheating via Capped Evaluation with Randomized Tests (https://arxiv.org/abs/2606.07379)
- **What's New**: CapCode와 CapReward라는 두 가지 새롭고 혁신적인 접근 방식을 소개합니다. CapCode는 코딩 데이터셋을 구성하는 새로운 프레임워크로, 비정상적인 높은 성능을 감지할 수 있는 점수 해석 방식을 제공합니다. CapReward는 CapCode 원칙을 기반으로 하는 보상 설계로, 모델이 의도된 작업 사양을 더 잘 준수하도록 유도합니다.

- **Technical Details**: CapCode는 테스트의 성과 한계를 1 미만으로 의도적으로 설정하여 'cheating'을 감지합니다. 이 프레임워크는 모델의 비치팅(pass rate) 성능을 잘 정의된 코딩 테스트에서 100%에 도달할 수 없게 합니다. CapReward는 비치팅 행동을 받은 성과를 cap 이하로 제한하여 높은 성과를 가치 있게 평가하고 비정상적으로 높은 점수에 패널티를 부여합니다.

- **Performance Highlights**: CapCode는 여러 데이터셋에서 성과를 추적하면서 비정상적인 높은 성과를 감지하였고, CapReward는 강화 학습에서 비정상적인 행동을 줄여 의도된 과제를 보다 잘 따르는 모델을 생성했습니다. 이 실험들은 두 가지 방법이 성능을 저해하지 않으면서도 효과적으로 'cheating'을 감지 및 저감할 수 있음을 보여줍니다.



### DirectAudioEdit: Inversion-Free Text-Guided Audio Editing via Diffusion Prediction Contras (https://arxiv.org/abs/2606.07356)
- **What's New**: 이 논문에서는 DirectAudioEdit라는 교육 없이도 사용할 수 있는 오디오 편집 방법을 제안합니다. 이 방법은 일반적으로 시간 소모가 크고 재구성 오류가 발생하는 전통적인 역변환(inversion-based) 방법과는 다릅니다. DirectAudioEdit는 오디오 생성 모델의 확산 모형(diffusion model)을 기반으로 하여, 편집의 효율성과 품질을 동시에 향상시키는 것을 목표로 합니다.

- **Technical Details**: DirectAudioEdit 접근법은 초기 노이즈와 현재 상태를 조합하여 목표 오디오를 추정합니다. 이는 기존의 역변환과 노이즈 제거(denoising) 과정 모두에 걸쳐 초기 노이즈를 활용함으로써 이루어집니다. 논문에서는 또한 확산 모델의 경로를 예측하는 데 어려움이 있음을 강조하며, 이런 경로는 확률적 미분 방정식(stochastic differential equations)으로 설명되어 곡선 형태가 됩니다.

- **Performance Highlights**: DirectAudioEdit의 실험 결과, 기존의 역변환 기반 편집 방법에 비해 Macro-averaged FAD와 KL 지수가 각각 15.9% 및 15.8% 감소하며, 최대 64.5%의 편집 속도 향상을 보여주었습니다. 이는 음악 및 사건 수준의 편집 작업에서 성능이 크게 향상되었음을 의미합니다.



### Acoustic Cue Alignment in Audio Language Models for Speech Emotion Recognition (https://arxiv.org/abs/2606.07309)
Comments:
          6 pages, 3 figures, 3 tables

- **What's New**: 이번 연구는 명확한 음향 단서를 통해 지침을 따르는 오디오 언어 모델(ALMs)에서 음향 큐가 실제로 사용되는 방식을 조사합니다. 이를 위해 eGeMAPS의 유의미한 음향 개념 토큰을 도출하고, 이를 텍스트 프롬프트에 추가하여 오디오 입력은 그대로 유지합니다. 연구 결과, 알맞게 정렬된 토큰이 성능을 향상시키는 반면, 셔플된 또는 상충하는 토큰은 성능을 저하시킨다는 것을 보여줍니다.

- **Technical Details**: 유명한 FAU-Aibo와 IEMOCAP 벤치마크를 통해, 정렬된 개념 토큰을 사용한 ALM은 비정렬 토큰에 비해 비교적 높은 Unweighted Average Recall (UAR)을 달성합니다. 본 연구에서는 에너지, 피치, 역학, 밝기, 포먼트, 음성 품질 등 여섯 가지 범주로 음향 특성을 구분하여 텍스트 형식으로 변환, ALM 프롬프트에 보조 Cue로 첨부하여 실험하였습니다.

- **Performance Highlights**: 연구 결과, 어떠한 토큰 조작에도 불구하고 ALM 예측은 오디오 신호에 여전히 부분적으로 의존하는 것으로 나타났습니다. 정렬된 음향 개념 토큰은 성능을 일관되게 향상시키는 반면, 강한 손상이 있는 경우 예측 성능이 일정 수준 저하되지만 음향 신호를 완전히 무시하지는 않는 것으로 밝혀졌습니다. 이러한 발견은 ALM 기반의 감정 인식에서 음향 기반 큐의 사용을 보다 명확하게 검증하는 방법론을 제시합니다.



### SWE-Explore: Benchmarking How Coding Agents Explore Repositories (https://arxiv.org/abs/2606.07297)
Comments:
          20 pages, 5 figures

- **What's New**: 이번 논문은 SWE-Explore라는 새로운 벤치마크를 소개합니다. SWE-Explore는 코드 작성 에이전트의 중요한 기능인 리포지토리 탐색의 평가를 단독으로 수행합니다. 기존의 이진 예측 문제로 처리되는 코드 작업을 세분화하여, 코드 지역을 순위별로 반환하도록 요구합니다.

- **Technical Details**: SWE-Explore는 10개 프로그래밍 언어와 203개의 오픈 소스 리포지토리에 걸쳐 848개의 이슈를 다룹니다. 각 인스턴스에 대해, 동일한 이슈를 성공적으로 해결한 독립 에이전트의 경로에서 라인 수준의 정답을 추출하여 실제로 참조한 코드 지역을 정제합니다. 이를 통해 코드 리포지토리의 탐색 및 선택의 메커니즘을 보다 정교하게 평가할 수 있습니다.

- **Performance Highlights**: SWE-Explore의 평가에서 우리는 탐색 점수가 실제 수리(success)와 어떻게 연결되는지를 검증합니다. 여러 탐색 방법과 에이전트를 통해, 우리가 개발한 메트릭이 리포지토리 탐색 품질을 예측할 수 있음을 보여주었습니다. 이를 통해 기존의 파일 수준 정위치(file-level localization)보다 라인 수준의 범위(line-level coverage)와 효율적 순위(rank efficiency)가 상태를 차별화하는 주요 축임을 확인했습니다.



### MMAE: A Massive Multitask Audio Editing Benchmark (https://arxiv.org/abs/2606.07229)
Comments:
          Open-Source at this https URL

- **What's New**: MMAE(Massive Multitask Audio Editing)는 일반 목적의 지시 기반 오디오 편집을 위한 포괄적인 평가 벤치마크로 소개됩니다. 이러한 평가 체계는 현재까지의 높은 단편화에서 벗어나, 실제 세계의 7가지 오디오 모달리티(모델리티)와 6단계의 작업 복잡도를 포괄합니다. 이는 단순한 수정에서부터 다단계 추론과 다회 편집까지의 다양한 작업을 포함합니다.

- **Technical Details**: MMAE는 2,000개의 고해상도 샘플과 함께 최초의 기준(기준)

- **Performance Highlights**: 현재 시스템은 신뢰할 수 있는 편집을 달성하는 데 있어 상당한 한계점이 있습니다. 특히, 정확 일치율(Exact Match Rate, EMR)은 5% 이하로 일관되며, 복잡한 혼합 모달리티 과제에서는 0%로 떨어지는 것으로 나타났습니다. 이러한 결과는 정밀 실행과 구조적 강건성에서의 중요한 병목현상을 드러냅니다.



### DEFINED: A Data-Efficient Computational Framework for Fine-Grained Creativity Assessment in Debate Scenarios (https://arxiv.org/abs/2606.07226)
Comments:
          Accepted by KDD 2026

- **What's New**: 이 논문에서는 DEFINED라는 새로운 데이터 효율적인 계산 프레임워크를 제안하며, 이는 토론 시나리오에서의 세밀한 창의력 평가를 자동화하는 데 초점을 맞추고 있습니다. 이 프레임워크는 여덟 차원의 계층적 메트릭 시스템을 통해 창의력을 정의하며, 기존의 수작업 평가에서 발생하는 노동집약적인 어려움을 해결하기 위해 설계되었습니다. DEFINED는 학습된 대학원 전문가에 의해 주석이 달린 세밀한 데이터로부터 강력한 학습을 가능하게 합니다.

- **Technical Details**: DEFINED 프레임워크는 인간의 총체적 판단과 일치하는 여덟 차원의 세밀한 점수와 거시적(debate) 점수를 생성합니다. 이 시스템은 실제 토론 대회에서 수집한 정확한 경쟁 문서 데이터를 활용하며, 원본 데이터의 엘리트 편향을 줄이기 위해 제한된 데이터 증강 전략을 채택합니다. 여덟 차원의 메트릭 시스템은 분산적 사고(divergent thinking) 및 수렴적 사고(convergent thinking)와 같은 창의력 관련 차원과 함께 토론 관련 차원을 포함합니다.

- **Performance Highlights**: 실험 결과, DEFINED의 점수 모델이 기존의 토론 점수화 방법 및 프롬프트 기반 대형 언어 모델 평가자를 초과하며 정확하고 안정적인 점수를 달성했습니다. 특히, 60개의 세밀한 전문가 주석 샘플과 4,000개의 거시적 샘플을 활용하여 고정밀 예측을 달성했으며, 이러한 접근 방식은 '소규모 데이터' 문제를 효과적으로 해결했습니다. 또한, 다양한 숙련도와 주석 세분화에서 시스템의 강건성을 검증하기 위한 평가 프로토콜을 수립했습니다.



### HKVM-RAG: Key-Value-Separated Hypergraph Evidence Organization for Multi-Hop RAG (https://arxiv.org/abs/2606.07218)
Comments:
          Submitted to ICDE 2027. 13 pages, 3 figures

- **What's New**: 이번 연구에서는 Multi-hop RAG(Review-Aided Generation)가 단순한 언어 모델링 문제가 아니라 고정된 검색 예산 내에서 증거를 조직하는 데이터 엔지니어링 문제임을 강조합니다. HKVM-RAG라는 새로운 방식으로, 답변 경로(hyperedges)를 구성하고 이를 검색 키로 사용하는 고유한 증거 조직 레이어를 제안합니다. 이 시스템은 기계 학습 추론에서 증거를 더 효과적으로 활용할 수 있도록 합니다.

- **Technical Details**: HKVM-RAG는 키-값 분리(key-value-separated) 구조를 기반으로 하며, LLM(대형 언어 모델)에서 추출한 증거 튜플을 바탕으로 답변 경로 하이퍼 엣지를 형성합니다. 이 과정에서 패시지 텍스트는 답변 값을 위해 활용됩니다. 하이퍼그래프 방식의 검색 키는 패리와 그래프 기반 키에서 정보를 보다 효과적으로 조직할 수 있도록 도와줍니다.

- **Performance Highlights**: HKVM-RAG는 여러 벤치마크에서 기존 지식 그래프(KG)-기반 접근 방식보다 성능이 향상되었습니다. 예를 들어, 2WikiMultiHopQA에서 F1 점수가 +3.426, MuSiQue에서는 +3.592로 향상되었습니다. 심층 분석을 통해 지원 선택이 주요 병목 현상으로 드러나며, HKVM-RAG는 다층적인 증거 제어 메커니즘으로써 무게가 있는 하이퍼그래프 키-값 검색을 가능하게 합니다.



### Textual Supervision Enhances Geospatial Representations in Vision-Language Models (https://arxiv.org/abs/2606.07172)
Comments:
          Accepted at ICML 2026

- **What's New**: 이번 연구에서는 기계 학습 시스템의 지리적 이해를 분석하였습니다. 기존의 비전 전용 아키텍처, 비전-언어 모델, 그리고 대규모 다중 모달 기초 모델을 통해 지리적 표현을 평가하였으며, 텍스트 감독이 이러한 학습을 어떻게 향상시키는지 조사하였습니다. 연구 결과, 언어가 공간 맥락을 인코딩하는 유효한 보조 방식으로 작용하며, 다중 모달 학습이 지리적 인공지능 발전의 핵심 방향이 될 수 있음을 제시하였습니다.

- **Technical Details**: 비전 모델은 최근 10년 동안 컨볼루션 신경망(CNN)과 비전 트랜스포머(ViT)의 발전으로 큰 발전을 이루었습니다. 다양한 메타 정보로부터 지리적 정보를 내재화할 수 있는 가능성이 제기되었으며, 이러한 모델들은 프리트레인(pretrain) 및 파인튜닝(fine-tuning) 과정에서 지리적 지식을 어느 정도 내재화하는지를 분석하고 있습니다. 우리가 사용한 지리적 표현(geospatial representations)은 ViT와 VLM의 내부 레이어에 포함된 잠재적 피처를 지칭합니다.

- **Performance Highlights**: 본 연구는 비전 전용 및 비전-언어 모델이 지리적 정보를 암묵적으로 어떻게 인코딩하는지를 조사하였습니다. 다양한 모델의 성능을 레이어 별 탐색을 통해 비교하였으며, 비전 전용 모델은 마지막 레이어에서 더 강한 표현을 보였고, VLM은 언어 모델 블록의 초기 레이어에서 더 나은 지리적 표현을 나타냈습니다. 비전-언어 모델에 대한 프롬프트(prompts)를 통해 지리적 정보가 후속 레이어로 전파되는 것이 관찰되었으며, 이로 인해 표현 품질이 향상되는 경우도 발견되었습니다.



### OffQ: Taming Structured Outliers in LLM Quantization by Offsetting (https://arxiv.org/abs/2606.07116)
- **What's New**: 본 논문에서 OffQ라는 새로운 방법을 소개하여 저비트 양자화(Low-bit Quantization)에서의 활성화 아웃라이어(activation outliers) 문제를 해결합니다. OffQ는 첫 번째로 제안된 top-1 PCA를 통해 저차원의 아웃라이어 서브스페이스(low-dimensional outlier subspace)를 식별하고, 고크기 활성화를 회전을 통해 하나의 채널로 집중시킵니다. 마지막으로, 집중된 아웃라이어 채널의 크기를 공유 오프셋으로 변환하여 활성화의 표준 편차를 줄이는 오프셋 전략을 활용합니다.

- **Technical Details**: OffQ는 저비트 양자화의 기본 적용에서 아웃라이어의 영향을 제거하기 위해 고유한 회전 기반 프로세스를 사용합니다. 이 방식은 단순한 방법이며, 복잡한 매개변수나 계산 오버헤드 없이 저비트 계산의 효율성을 유지할 수 있습니다. OffQ는 Hadamard 회전을 적용하여 아웃라이어 에너지를 그룹별 오프셋으로 전환하고, 이를 표준 저비트 양자화의 제로 포인트로 흡수함으로써 성능을 높입니다. 이러한 절차는 W4A4KV4(Weight, Activation, KV-cache 모두 4비트로 양자화) 적용에서 효과적으로 이루어집니다.

- **Performance Highlights**: 다양한 LLM 구조 및 벤치마크에서 OffQ는 기존의 최신 방법들을 능가하며 모델의 정확성을 지속적으로 개선했습니다. OffQ는 예측력(perplexity)과 정확도 모두에서 개선을 이뤄내며, 최종적으로 저비트 추론의 간결함과 효율성을 유지합니다. 실험을 통해 4비트 양자화의 견고성을 높이기 위한 실용적인 메커니즘으로 제공됩니다.



### Meaning in Order, Order in Meaning: Semantic R-precision for Keyphrase Evaluation (https://arxiv.org/abs/2606.07057)
- **What's New**: 본 논문에서는 자동 생성된 키프레이즈의 평가 품질을 개선하기 위해 새로운 평가 지표인 Semantic R-Precision (SemR-p)을 소개합니다. SemR-p는 의미적 유사성(semantic similarity)을 랭크 인식(Rank-aware) R-Precision 프레임워크에 통합하여 작성되었습니다. 이 지표는 인간 중심의 관점에서 디자인되어 사용자가 실제로 인포메이션의 유의미성을 어떻게 평가하는지를 반영하려고 합니다.

- **Technical Details**: SemR-p는 정보 검색(Information Retrieval)에서의 평가 지표의 원리를 기반으로 하여 의미적으로 관련성이 높은 키프레이즈가 출력 목록의 앞쪽에 나타날수록 높은 점수를 부여합니다. 이 접근 방식은 기존의 키프레이즈 평가 방식들이 단어의 정확한 일치를 기반으로 하여 의미적 동등성을 간과하는 문제를 해결하고자 합니다. 기계 학습 모델 및 데이터셋에 대한 광범위한 분석을 통해 SemR-p의 의미적 민감성, 랭킹 인식 및 구별력을 평가하였습니다.

- **Performance Highlights**: 실험 결과 SemR-p는 기존의 전통적인 어휘 및 의미적 유사성 매칭 지표들과 함께 사용될 수 있는 보완적인 시각을 제공하며, 키프레이즈 예측 평가에서 사용자 중심의 관련성을 보다 잘 반영하고 있음을 보여주었습니다. 또한, SemR-p는 다양한 모델과 데이터셋에서 비교적 높은 신뢰성과 일관성을 입증하였으며, 기존의 방식으로는 놓치기 쉬운 의미적 관계를 포착할 수 있는 가능성을 제공한다는 점에서 주목할 만합니다.



### Phonetic Error Analysis of Raw Waveform Acoustic Models (https://arxiv.org/abs/2606.07030)
Comments:
          INTERSPEECH2026

- **What's New**: 이번 연구는 전통적인 전화 인식 시스템의 성능 측정 방식인 전화 오류율(PER) 이상의 오류 패턴을 분석합니다. 전화가 광범위한 음소 클래스를 고려하여 분류된 점에서 큰 기여를 합니다. 또한, 기존의 Filterbank 기반 시스템의 분석을 원시 파형(raw waveform) 음향 모델로 확장한 점이 주목할 만합니다.

- **Technical Details**: 모델은 파라메트릭(SincNet, Sinc2Net) 또는 비파라메트릭 CNN과 Bidirectional LSTM(BLSTM)을 조합하여 구성됩니다. BLSTM 계층의 효과는 강한 시간적 동적을 가지는 클래스에서 가장 두드러지게 나타났으며, 특히 이중모음, 마찰음 및 반모음에서 효과가 큽니다. 또한, WSJ에서 학습한 전이 학습이 자음의 오류를 유지하면서 주요 변화가 발생하는 것을 확인했습니다.

- **Performance Highlights**: 제안된 모델은 TIMIT 데이터 셋에서 원시 파형 모델 중 가장 낮은 PER 15.3%를 기록하며, WSJ 전이 학습을 통해 PER을 11.3%로 줄였습니다. 전체적으로, BLSTM 통합 모델이 다른 시스템에 비해 뛰어난 성능을 보였으며, 특히 논리 통계적으로 유의미한 자음-모음의 증가 비대칭을 관찰했습니다.



### The Sim-to-Real Gap of Foundation Model Agents: A Unified MDP Perspectiv (https://arxiv.org/abs/2606.07017)
Comments:
          7 pages, 2 figures, 2 tables. Accepted by KDD 2026 Blue Sky Ideas Track

- **What's New**: 본 논문에서는 Foundation model agents의 평가 및 훈련 간의 격차를 전통적인 simulation-to-reality(시뮬레이션에서 현실로) 문제로 포괄적으로 형식화하는 새로운 접근 방식을 제안합니다. 이를 통해 Markov Decision Process(MDP)의 네 가지 요소인 Observation(관측), Action(행동), Transition(전이), Reward(보상) 중심으로 격차를 다룹니다. 기존의 접근법을 보완하기 위해 domain randomization과 grounded action transformation 등의 솔루션을 채택할 것을 권장합니다.

- **Technical Details**: RL(강화학습)은 보통 할인된 Markov 결정 과정(MDP)으로 형식화되어 있으며, 이는 상태 공간(𝒮), 행동 공간(𝒜), 전이 동역학(𝒯), 보상 함수(ℛ)를 포함합니다. 시뮬레이션 MDP에서 훈련된 정책은 현실 MDP에 배포되고, 시뮬레이션과 현실 간의 격차는 각 요소의 불일치에서 발생합니다. 주로 관측(Observation), 행동(Action), 전이(Transition), 보상(Reward) 간의 차이가 그 원인으로 지적됩니다.

- **Performance Highlights**: 이 논문은 새로운 언어와 표준화된 스트레스 테스트 기준을 도입하여 Foundation model의 실행 가능성을 높이고, 신뢰할 수 있는 에이전트의 개발을 촉진하고자 합니다. 마지막으로, 이 연구 노력이 KDD(지식 발견 및 데이터 마이닝) 커뮤니티의 주제를 발전시키고 신뢰할 수 있는 데이터 과학의 기초를 마련하는 데에 기여할 것으로 기대됩니다.



### RASFT: Rollout-Adaptive Supervised Fine-Tuning for Reasoning (https://arxiv.org/abs/2606.07006)
- **What's New**: 이 논문에서는 Rollout-Adaptive Supervised Fine-Tuning (RASFT)라는 정책 인식 기반 SFT 프레임워크를 제안합니다. 기존의 SFT 방법이 전문가 시연을 단순히 모방하는 데 의존하는 반면, RASFT는 문제 해결 가능성에 따라 전문가의 안내 강도를 조절합니다. 이를 통해 모델의 자체 추론 능력을 유지하며, 과도한 모방으로 인한 정책 드리프트를 방지합니다.

- **Technical Details**: RASFT는 현재 모델의 롤아웃 행동을 이용하여 전문가의 지도에 대한 적절한 조정을 수행합니다. 구체적으로, RASFT는 전문가 경로와 현재 모델이 생성한 정답 경로를 포함하는 후보 집합을 구성하고, 성공적인 롤아웃 비율에 따라 문제 수준의 해결 가능성을 평가합니다. 이 점수에 따라, 문제가 어려운 경우 전문가의 영향력을 높이고 모델이 신뢰할 수 있는 추론을 이미 보여주는 경우에는 모방 강도를 줄입니다.

- **Performance Highlights**: 여러 수학적 및 코드 추론 벤치마크를 통해 RASFT는 기존 SFT 스타일의 기초선보다 일관되게 뛰어난 성능을 나타냈습니다. Qwen2.5-Math-1.5B 문제에서 10.9%의 상대적 성장을 달성하고, Llama-3.2-3B 코드 생성에서 최대 26.9%의 향상을 보였습니다. 이러한 결과는 롤아웃 기반의 지도력이 단순한 전문가 적합 이상의 이점을 제공함을 시사합니다.



### MADRAG: Multi-Agent Debate with Retrieval-Augmented Generation for Training-Free Analytic Essay Scoring (https://arxiv.org/abs/2606.06754)
Comments:
          21 pages, 7 figures, 14 tables

- **What's New**: MADRAG는 훈련이 필요 없는 분석 에세이 평가 프레임워크를 제시합니다. 이 프레임워크는 다중 에이전트 추론(multi-agent reasoning)과 검색 기반의 기초(retrieval-augmented grounding)를 결합한 혁신적인 접근 방식을 사용합니다. 기존의 LLM-as-judge 방식과 달리, MADRAG는 평가를 상호작용 과정으로 분해하여 더 안전하고 신뢰할 수 있는 점수 산출이 가능합니다.

- **Technical Details**: MADRAG는 Advocate, Skeptic, Judge의 세 가지 역할로 구성된 시스템입니다. Advocate는 에세이의 강점을 찾고, Skeptic은 약점을 비판하며, Judge는 이 두 가지 주장을 종합해 최종 점수를 매깁니다. 또한, Judge는 스코어가 매겨진 예시들과의 비교를 통해 조정(calibration)을 수행할 수 있도록 사전 평가(rubric-aligned exemplar retrieval)를 활용합니다.

- **Performance Highlights**: MADRAG는 기존의 프롬프트 기반(baseline) 접근 방식보다 성능이 현저히 향상된 결과를 보여주었습니다. 전문 교육 없이도 감독되는 시스템(supervised systems)의 성능에 근접하는 결과를 얻었습니다. 검증 결과에 따르면, 검색이 조정에서 영향을 미치고, 토론이 높은 수준의 특성에 대한 추론을 개선하는 데 기여함을 보여주고 있습니다.



### HybridCodec: Fast Dual-Stream, Semantically Enhanced Neural Audio Codec (https://arxiv.org/abs/2606.06743)
Comments:
          5 pages, 5 tables, 1 figure, Accepted at Interspeech 2026

- **What's New**: 이 논문에서는 HybridCodec라는 새로운 통합 아키텍처를 제안합니다. 이 모델은 두 가지 접근법, 즉 의미적 증류(semantic distillation)와 이중 스트림(ddual-stream) 구조를 결합하여 강력한 의미-음향 분리(disentanglement)를 실현합니다. 또한, HybridCodec은 인퍼런스(inference) 동안 SSL 모델을 요구하지 않으면서도 의미적 전문성(semantic specialization)을 유지합니다.

- **Technical Details**: HybridCodec의 구조는 두 개의 스트림, 즉 의미적 스트림과 음향 스트림으로 구성되어 있으며, 각 스트림은 고유한 인코더와 디코더로 이루어져 있습니다. 공통 인코더는 24kHz의 원시 파형을 입력으로 받아 1D 합성곱(convolution) 네트워크(CNN)를 사용하여 처리합니다. 인코더의 출력은 의미적 디코더를 통해 의미 정보가 증류되고, 음향 디코더는 잔여 음향 정보(residual acoustic information)를 복원하는 데 사용됩니다.

- **Performance Highlights**: HybridCodec은 의미적 전문화 측면에서 RVQ-1 테스트 세트에서 뛰어난 성능을 보이며, 전체적인 음향 품질도 경쟁력을 유지합니다. 논문에서는 이 모델이 기존의 이중 스트림 모델에 비해 3배 빠른 인퍼런스 속도를 달성했다고 명시하고 있습니다. 또한, 다양한 환경에서의 강인성(robustness) 또한 입증되었습니다.



### OpenSkill: Open-World Self-Evolution for LLM Agents (https://arxiv.org/abs/2606.06741)
Comments:
          20 pages, 4 figures and 8 tables. Code is avalable at this https URL

- **What's New**: 이 논문은 기존의 self-evolving agents가 특정 피드백이나 습득된 기술들을 사용하여 개선할 수 있다는 점을 넘어서, 오히려 제한된 태스크 프롬프트만을 활용해 기초를 세우는 방법을 제시합니다. 제안된 OpenSkill 프레임워크는 오픈월드 자원을 통해 기술과 검증 신호를 스스로 생성할 수 있도록 지원하여, 기존 방법과 달리 명확한 목표 태스크 감독 없이 진화를 가능하게 합니다. 이러한 접근은 대규모 언어 모델(LLM) 에이전트의 자율성을 증가시킵니다.

- **Technical Details**: OpenSkill은 세 단계로 구성된 파이프라인을 통해 오픈 월드 지식을 획득하고, 자가 생성한 가상 태스크를 통해 기술을 정교화하며, 최종적으로 제로샷(zero-shot) 평가를 통해 목표 에이전트에 기술을 배포합니다. 이러한 과정에서 오픈 월드는 배우는 지식과 감독 독립적인 연습 환경을 제공하여, 최종 평가에만 목표-태스크 감독을 사용합니다. 이를 통해 모델 간의 기술 전이를 지원합니다.

- **Performance Highlights**: OpenSkill은 세 가지 벤치마크와 두 개의 목표 에이전트 설정에서 가장 높은 자동 통과율을 기록합니다. 특히, SkillsBench에서 가장 강력한 폐쇄형 세계 기반라인보다 8.9% 높은 성능을 보이며, 모델 특화적 조정 없이 기술을 전이할 수 있습니다. 또한, OpenSkill의 자체 생성 검증자는 실제 테스트 의도와 88.9% 정합성을 갖습니다.



### Multilingual Multi-Speaker Unit Vocoders: A Systematic Analysis of Discrete Speech Representations (https://arxiv.org/abs/2606.06740)
Comments:
          5 pages, 5 tables, 1 figure, Accepted at Interspeech 2026

- **What's New**: 이 연구에서는 BigVGAN을 기반으로 한 단위 보코더를 분석하여 네 가지 인도어(벵골어, 힌디어, 타밀어, 텔루구어)에 대한 음성 생성의 품질을 높이는 방법을 논의합니다. 연구는 군집 크기와 조건화 전략 간의 상호작용을 평가하여 음성 인식의 명도와 화자 유사성을 개선하는 데 초점을 맞추고 있습니다. 또한, 언어 감독이 낮은 군집 크기에서 더욱 효과적임을 발견하여 모호한 단위의 문제를 해결하고, 다양한 언어에서 유사한 음소들이 동일한 군집 ID로 수렴하는 현상에 대해 분석합니다.

- **Technical Details**: 이 연구는 BigVGAN 아키텍처를 사용하여 생성기, 멀티 주기 분류기(MPD), CQT 기반의 시간 주파수 분류기를 포함한 음성 합성 시스템을 개발합니다. 여기서 생성기 입력은 멜 스펙트로그램 대신 디스클리트 단위를 사용하고, 화자 및 언어 임베딩을 결합하여 추가 정보를 제공합니다. 조건화 메커니즘과 군집 크기에 따른 트레이드오프를 탐색함으로써 다국어 다화자 설정에서 음성 생성 품질을 개선할 수 있는 방법을 제시합니다.

- **Performance Highlights**: 실험 결과, 군집 크기가 음성의 이해도를 제어하고 화자 유지에 필수적임을 보여주었습니다. 특히, 의사소통의 흐름을 방해하지 않는 조건화 방식이 중요하며, 이는 다양한 언어에서 공통적으로 (shared) 사용되는 단위 각각의 군집에서 선명도를 높이는 데 기여합니다. 연구에서는 단위 기반의 보코더가 멀티언어 다화자 환경에서 어떻게 성능을 발휘할 수 있는지를 정량적으로 평가하고, 음성 생성의 질을 개선하기 위한 최적의 설정 방안을 제안합니다.



### RECAP: Regression Evaluation for Continual Adaptation of Prompts (https://arxiv.org/abs/2606.06698)
- **What's New**: 이번 논문에서는 RECAP이라는 새로운 벤치마크를 소개합니다. RECAP은 이론과 실제 동작을 고려하여 지속적으로 변화하는 제약 조건을 평가하는 새로운 방법론을 제공합니다. 기존의 정적 기준에만 의존하던 기존 벤치마크와 달리, RECAP은 프로액티브(adaptive) 환경에서의 적응 및 테스트를 측정할 수 있습니다.

- **Technical Details**: RECAP은 지속적 학습(Continual Learning) 문맥에서 제약 이행을 평가하기 위해 설계되었습니다. 제약 조건은 독립적으로 진화하며, 모델은 주어진 제약 조건 명세에만 의존하여 초기 데이터 없이 적응해야 합니다. 이는 기존의 데이터가 아닌 새로운 정보를 통해 즉각적인 적응을 요구하며, 따라서 이러한 프레임워크를 통해 모델의 효율성을 측정할 수 있습니다.

- **Performance Highlights**: 논문에서는 여러 프로프트 적응 방법을 평가한 결과, 이 방법들이 프로액티브 환경에서 기대하는 수준의 성과를 보이지 않음을 확인하였습니다. 결과적으로, 기존의 오프라인 또는 리액티브 방식을 기반으로 한 방법들이 프로액티브 환경에서는 부적합하다는 점을 강조합니다. 이러한 연구는 프로액티브 방식의 프롬프트 적응 기법 설계 필요성을 더욱 부각시키고 있습니다.



### Multiscale POD of Transformer Attention Fields: Scale-Selective Analysis via Morlet Scalogram (https://arxiv.org/abs/2606.06573)
Comments:
          23 pages, 3 figures, 4 tables

- **What's New**: 본 논문에서는 transformer attention fields에 대한 scale-selective Proper Orthogonal Decomposition (POD)을 소개합니다. 이는 난류 흐름 집합체에서 에너지적으로 지배적인 모드를 추출하기 위해 POD를 활용하는 영감을 토대로 하며, Morlet continuous wavelet transform을 사용하여 attention lag 구조에서 지배적인 시간 스케일을 식별합니다. 이 연구는 attention fields의 에너지적 특성을 기반으로 layer-dependent 스케일 조직을 밝혀내는데 기여합니다.

- **Technical Details**: scale-selective POD는 Morlet scalogram을 사용해 attention field의 dominant scales를 진단하고, Gaussian lag-window를 각 dominant scale에 적용하여 POD를 따로 수행하는 방식으로 이루어집니다. 이를 통해 L×L attention field의 모든 시계열 구조를 혼합하지 않고 각 스케일에 적합한 정보를 추출할 수 있습니다. 이 방법론은 데이터 기반으로 attention field의 복잡성을 측정하는 스펙트럼 집중 지수를 정의하며, attention fields의 최소 평균 L2 근사 순위를 설정합니다.

- **Performance Highlights**: POD 적용을 통해 transformer의 attention fields에서 지배적인 패턴과 최적의 압축을 달성할 수 있으며, 이는 최소한의 기저 함수 수로 attention field의 분산을 설명하는 데 필요합니다. 이 연구는 계층 간 attention field의 복잡성을 구별하는 데이터를 기반으로 한 스펙트럼 복잡성 지수를 empirically 제공합니다. 또한, 전통적인 POD 최적성 정리를 통해 transformers의 계층별 순위를 배분하는 원칙적 기준을 수립함으로써, EGA가 계층 간 스펙트럼 에너지를 체계적으로 증가시키는 연결고리를 보여줍니다.



### Position: Don't Just "Fix it in Post": A Science of AI Must Study Training Dynamics (https://arxiv.org/abs/2606.06533)
Comments:
          Accepted as an oral to the ICML: this https URL

- **What's New**: 이번 논문은 AI(인공지능)의 과학적 이해의 필요성에 대해 논의합니다. 기존 AI 연구가 모델을 정적인 객체로 취급하며 훈련 이후의 행동만 분석하는 경향이 있음을 지적합니다. 모델 행동이 왜 발생하는지를 이해해야 한다는 점을 강조합니다.

- **Technical Details**: AI 모델은 데이터, 목표, 아키텍처(architecture), 최적화 역학(dynamics)에 의해 형성된 시간에 따라 진화하는 과정의 스냅샷으로 보아야 합니다. 논문에서는 훈련 역학(training dynamics)을 연구하여 모델 행동을 이해하는 새로운 과학의 필요성을 제시하며, 초기 훈련 신호로부터 결과를 예측하고 잘못된 경로에서 개입하는 방법을 다루고 있습니다.

- **Performance Highlights**: 훈련 절차를 설계하여 원하는 속성을 더 신뢰성 있게 생성하는 것이 궁극적인 목표입니다. 손실(loss) 예측의 성공을 능력(capabilities), 편향(biases), 강건성(robustness), 안정성(safety)과 관련된 행동에 확장하는 것이 도전 과제이며, 과학의 역사 및 철학에 기초한 이러한 이론의 요구 사항을 정립하고 관찰 가능한 문제를 식별합니다.



New uploads on arXiv(cs.IR)

### Bradley-Terry Rankings for Recommender Systems Across Dataset Taxonomies (https://arxiv.org/abs/2606.07492)
Comments:
          KDD'26

- **What's New**: 본 논문은 추천 알고리즘의 성능 평가에서 데이터 특성에 의존하는 새로운 데이터 기반의 순위 매기는 방법론을 제안합니다. 기존의 성능 지표를 단순히 집계하는 방식은 잘못된 순위를 초래할 수 있으며, 이를 해결하기 위해 Bradley-Terry 모델에 기반한 확장된 기법을 도입하였습니다. 또한, 불완전한 데이터에 대한 순위 일관성을 평가하기 위한 새로운 지표를 제안합니다.

- **Technical Details**: 연구는 89개 데이터셋에 걸쳐 14개의 추천 알고리즘을 벤치마킹하고, 다양한 실험 결과를 집계하여 확률적 순위를 생성하는 BB 모델을 기반으로 합니다. 이를 통해 데이터 특성에 따라 맞춤화된 알고리즘 순위를 제공하며, 알고리즘의 선택에서 데이터 조세에 따라 선택할 수 있도록 합니다. 특히, 'transitive triplets' 메트릭을 통해 불완전한 벤치마크 결과에서 순위 일관성을 검증합니다.

- **Performance Highlights**: 제안된 접근법은 Bradley-Terry 기반의 순위가 단순한 메트릭 집계에 의해 얻어진 순위보다 훨씬 더 안정적임을 보여줍니다. 또한, BT 트리와 공변량 조정 BT 모델을 이용한 알고리즘 순위 예측은 실험을 수행하지 않고도 특정 데이터셋에 대해 유효한 예측을 가능하게 합니다. 이 연구 결과는 알고리즘 선택의 새로운 기준을 제시하며, 실용적인 적용 가능성을 높입니다.



### PaperFlow: Profiling, Recommending, and Adapting Across Daily Paper Streams (https://arxiv.org/abs/2606.07454)
Comments:
          48 pages, 13 figures, 22 tables

- **What's New**: 이 논문은 종래의 정적 추천 시스템을 넘어서, 개별 사용자의 연구 관심사를 반영하고 업데이트하는 동적 프레임워크인 PaperFlow를 소개합니다. PaperFlow는 세 가지 연계된 단계로 구성되어 있으며, 이는 Profiling, Recommending, Adapting입니다. 각 단계는 연구자가 진행하는 일일 독서 경험을 통합하고 개별 사용자의 변화하는 관심사를 모델링하는 데 중점을 두고 있습니다.

- **Technical Details**: PaperFlow는 연구자의 프로파일을 구조화된 형태로 유지하며, 연구 방향, 주제별 가중치, 저자 및 기관 우선 순위 등의 정보를 캡처합니다. 추천 단계에서는 날짜에 맞는 후보지의 후보지를 정렬하며 multi-signal aggregation을 통해 추천합니다. Adapting 단계에서는 사용자의 피드백 신호를 반영하여 사용자 상태를 지속적으로 업데이트하고, 연구자의 관심사 변화를 모델링합니다.

- **Performance Highlights**: PaperFlow는 기존의 다섯 가지 추천 시스템과의 실험을 통해 가장 뛰어난 성과를 보여주었습니다. 구체적으로, 자동 메트릭과 전문가의 판단 간 정렬을 검증하기 위한 인간 평가 프로토콜을 사용하여, Oracle 기반의 순위에서 가장 높은 점수를 기록했습니다. 또한, PaperFlow는 사용자의 시뮬레이션된 독서 선택과 높은 행동 일치를 보이며, 최상의 블라인드 인간 평가 점수를 자랑합니다.



### Gated Bidirectional Linear Attention for Generative Retrieva (https://arxiv.org/abs/2606.07317)
Comments:
          5 pages, 2 figures, 7 tables. Accepted at SIGIR 2026

- **What's New**: 이번 논문에서는 generative retrieval 시스템에 적용된 Gated Bidirectional Linear Attention (GBLA)라는 새로운 주의 메커니즘을 제안합니다. GBLA는 커널화된 선형 주의 메커니즘을 확장하면서 세 가지 경량화된 구성 요소, 즉 로컬 인과 혼합(Conv1D), 시퀀스 수준의 키 게이팅을 통한 소프트 포기, 그리고 게이티드 RMSNorm 출력을 포함합니다. 이 방법은 기존의 self-attention 기법이 가진 한계를 극복하고 대규모 사용자 기록을 처리하는 데 필요한 효율성을 높였습니다.

- **Technical Details**: GBLA는 인코더-디코더 아키텍처에서 사용되며, 사용자-아이템 상호작용 시퀀스를 기반으로 추천 아이템을 생성합니다. 이를 위해 다중 해시 기법을 사용하여 아이템 ID를 여러 해시 함수로 매핑하고, 생성된 임베딩을 연결하여 최종적으로 선형 층을 통해 투영합니다. GBLA는 훈련과 추론 속도를 선형적으로 높이는 것이 목표이며, 이로 인해 FlashAttention-v3와 같은 최적화된 Quadratic attention 구현에 비해 지연 시간을 감소시키고 있습니다.

- **Performance Highlights**: Yandex Music 데이터세트에서, 자가 주의 기법(SA)과 GBLA를 1:2 비율로 혼합한 인코더는 bidirectional self-attention과 거의 동일한 품질의 결과를 달성했습니다. H100 GPU에서 GBLA는 역사 길이 32768에 대해 FlashAttention-v3와 비교하여 최대 8.2배의 단일 레이어 속도를 달성했습니다. 또한, 이 하이브리드 설계는 공개 Amazon 벤치마크에서도 일관된 성능을 유지하여 성공적인 일반화 가능성을 보여주었습니다.



### Constrained Dominant Sets for Multimodal Document Question Answering (https://arxiv.org/abs/2606.07252)
- **What's New**: 이번 연구는 긴 문서에서 질문 응답 시스템의 효과를 향상시키기 위해 새로운 증거 선택 방식을 도입했습니다. 기존의 유사성 기반 검색 방식이 아닌, Constrained Dominant Set (CDS) 방식을 사용하여 쿼리-증강된 친화도 그래프에서 증거를 선택합니다. 이러한 접근은 문서 내의 유사한 정보에 치중하는 기존 시스템의 한계를 극복하려는 시도를 보여줍니다.

- **Technical Details**: CDS 방법은 쿼리를 하드 구조적 제약으로 인코딩하여, 클러스터 앵커를 통해 선택된 모든 요소가 질문과 직접 연결되도록 보장합니다. 또한, 관련성과 중복성의 균형을 자동으로 결정하여, 수동으로 조정할 필요가 없습니다. 이 과정에서 복제자 동역학(replicator dynamics)을 활용하여 전 세계적인 균형을 달성하며, 탐욕적 휴리스틱이 가져오는 왜곡을 피합니다.

- **Performance Highlights**: Qwen3-VL-32B 리더를 활용한 CDS 방법은 VisDoMBench에서 $66.99$의 평균 점수로 새로운 최첨단 성능을 달성합니다. 또한, no-retrieval 기준에 비해 VisDoMBench에서 $37.1$포인트, MMLongBench-Doc에서 $4.8$포인트 향상된 결과를 나타냅니다.



### FLOWREADER: Min-Cost Flow Optimization for Multi-Modal Long Document Q&A (https://arxiv.org/abs/2606.07235)
- **What's New**: 논문에서는 FLOWREADER라는 새로운 시스템을 소개합니다. FLOWREADER는 증거 조합을 최소 비용 흐름(min-cost flow) 문제로 다시 정의하여 멀티모달 노드 그래프에서 작업을 수행합니다. 이를 통해 이전의 단편적으로 배치된 정보의 연결성을 효과적으로 캡처할 수 있습니다.

- **Technical Details**: FLOWREADER는 기존의 Retrieval-Augmented Generation (RAG) 시스템들이 처리가 어려운 여러 가지 문제를 해결합니다. 이 시스템은 쿼리와 관련된 소스와 답안을 담고 있는 싱크(sink) 사이의 경로를 최적화된 흐름을 통해 선택합니다. 또한, 이를 통해 쿼리의 중요성, 구조적 전파, 다중 홉 운송을 동시에 인코딩합니다.

- **Performance Highlights**: 실험 결과 FLOWREADER는 PaperTab과 SlideVQA 두 개의 하위 세트에서 G²-Reader보다 성능이 향상된 것으로 나타났습니다. 또한, 전반적으로 FLOWREADER는 모든 하위 세트에서 경쟁력 있는 성능을 보였으며, 가장 강력한 기준선(G²-Reader)과의 차이는 0.74점 이내였습니다. 이러한 결과는 FLOWREADER가 단편화된 멀티모달 증거에서 효과적인 접근 방안임을 보여줍니다.



### HKVM-RAG: Key-Value-Separated Hypergraph Evidence Organization for Multi-Hop RAG (https://arxiv.org/abs/2606.07218)
Comments:
          Submitted to ICDE 2027. 13 pages, 3 figures

- **What's New**: 이번 연구에서는 Multi-hop RAG(Review-Aided Generation)가 단순한 언어 모델링 문제가 아니라 고정된 검색 예산 내에서 증거를 조직하는 데이터 엔지니어링 문제임을 강조합니다. HKVM-RAG라는 새로운 방식으로, 답변 경로(hyperedges)를 구성하고 이를 검색 키로 사용하는 고유한 증거 조직 레이어를 제안합니다. 이 시스템은 기계 학습 추론에서 증거를 더 효과적으로 활용할 수 있도록 합니다.

- **Technical Details**: HKVM-RAG는 키-값 분리(key-value-separated) 구조를 기반으로 하며, LLM(대형 언어 모델)에서 추출한 증거 튜플을 바탕으로 답변 경로 하이퍼 엣지를 형성합니다. 이 과정에서 패시지 텍스트는 답변 값을 위해 활용됩니다. 하이퍼그래프 방식의 검색 키는 패리와 그래프 기반 키에서 정보를 보다 효과적으로 조직할 수 있도록 도와줍니다.

- **Performance Highlights**: HKVM-RAG는 여러 벤치마크에서 기존 지식 그래프(KG)-기반 접근 방식보다 성능이 향상되었습니다. 예를 들어, 2WikiMultiHopQA에서 F1 점수가 +3.426, MuSiQue에서는 +3.592로 향상되었습니다. 심층 분석을 통해 지원 선택이 주요 병목 현상으로 드러나며, HKVM-RAG는 다층적인 증거 제어 메커니즘으로써 무게가 있는 하이퍼그래프 키-값 검색을 가능하게 합니다.



### RISE: A Rust Library for Inverted Index Search Engines (https://arxiv.org/abs/2606.07187)
- **What's New**: RISE는 Rust로 구현된 새로운 발전된 inverted index 라이브러리로, 정보 검색 업무의 성능과 효율성을 극대화합니다. 이 라이브러리는 Rust의 안전성과 성능을 활용하여 강력한 검색 솔루션을 제공하며, 확장성을 쉽게 만들 수 있는 trait 시스템을 제공합니다. RISE는 기존의 라이브러리들에 대한 경쟁력을 갖추고 있으며, 최대 2배의 성능 향상을 이루었습니다.

- **Technical Details**: RISE는 inverted index 연구를 위한 사용하기 쉬운 오픈 소스 라이브러리로, Elasticsearch와 같은 다른 프레임워크들과의 비교에서도 경쟁력 있는 성능을 보입니다. 주요 구성 요소로는 InvertedIndex, PostingListIter, QueryOperator, DocScorer가 포함되어 있으며, 요청 처리 방식에 있어 상호작용이 가능합니다. 쿼리 처리 알고리즘과 posting list 압축 방법의 효과적인 조합을 통해 인덱스의 효율성을 높입니다.

- **Performance Highlights**: RISE는 두 개의 대규모 웹 컬렉션에서 C++ 기반의 두 최신 라이브러리인 DS2I와 PISA와 비교 평가되었고, 쿼리 처리 성능 면에서 동등하거나 우수한 결과를 보였습니다. 또한 RISE는 기존 문헌에서 제시된 압축 알고리즘과 쿼리 처리 알고리즘을 재구현하고 검증함으로써 신뢰할 수 있는 기반을 다졌습니다. RISE의 성과는 정보 검색 분야의 연구자 및 실무자에게 귀중한 도구가 될 것입니다.



### Beyond Matching: Category-Guided Latent Intent Reasoning for Generative Retrieval in E-Commerc (https://arxiv.org/abs/2606.07075)
- **What's New**: 본 논문에서는 e-commerce 검색을 위한 CaLIR (Category-guided Latent Intent Reasoning) 프레임워크를 제안합니다. 이 시스템은 사용자 쿼리를 제품 Semantic Identifiers (SIDs)로 직접 매핑하여 검색 효율성을 향상시키고자 하며, 짧고 노이즈가 많은 e-commerce 쿼리의 특성을 고려합니다. CaLIR은 고유한 텍스트 이유를 생성하는 대신, 연속적인 잠재 의도를 학습하여 카테고리 계층 구조를 활용합니다.

- **Technical Details**: CaLIR는 세 가지 주요 구성 요소로 이루어져 있습니다. 첫째, Hierarchical Semantic Reasoning을 통해 쿼리를 카테고리 레벨로 매핑하여 잠재 의도를 정교화합니다. 둘째, Query-wise Reasoning Enhancement를 통해 같은 쿼리와 관련된 카테고리 정보를 활용하여 다양한 의도 경로를 모델링합니다. 마지막으로, Reasoning-aware Constrained Decoding 전략을 도입하여 카테고리 정보를 바탕으로 쿼리별 동적 접두사 트리를 구성합니다.

- **Performance Highlights**: 다국어 e-commerce 검색 데이터 세트를 기반으로 한 실험 결과, CaLIR는 기존의 방법들보다 검색 효율성과 추론 효율성 간의 균형을 더 잘 맞추며, 카테고리 기반의 잠재 의도 추론의 효과iveness를 입증하였습니다. 본 연구는 e-commerce 검색의 주요 설정에서 CaLIR의 성능이 최첨단 GR 방법들을 능가함을 보여줍니다.



### Decision-Theoretic Stopping Rules for Document Screening (https://arxiv.org/abs/2606.07071)
- **What's New**: 이 논문은 Technology-Assisted Review (TAR)에서의 중단 규칙 개선을 위해 의사결정 이론을 적용하였습니다. 기존의 중단 규칙은 수집된 정보의 목표 재현율(target recall)을 달성하기 위해 개발되었으나, 상황에 따라 더 적합한 결정이 필요할 수 있습니다. 저자는 정보의 가치(Value of Information) 원칙을 사용하여 효과적인 중단 정책을 제안합니다.

- **Technical Details**: 제안된 접근법은 세 가지 실제 중단 정책(Greedy, Smooth, Batch)을 도출하며, 이론적으로는 기대되는 완전 정보의 가치(Expected Value of Perfect Information, EVPI)를 기반으로 합니다. 이 방법은 특허 검색 및 체계적 검토와 같은 두 가지 전문 검색 작업에 적용되어 결과를 제공합니다. 의사결정 시 문서 검사 비용을 고려하여 상황에 맞는 중단 결정을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, 제안된 접근법은 평가된 비용 및 보상 설정에서 기존 방법보다 높은 네트 유틸리티(net utility)를 생성하는 것으로 나타났습니다. 이는 사용자에게 보다 적합한 중단 결정을 명확히 입증합니다. 이러한 성과는 TAR에서의 기존 중단 규칙이 가지는 한계를 극복하는 데 기여합니다.



### Meaning in Order, Order in Meaning: Semantic R-precision for Keyphrase Evaluation (https://arxiv.org/abs/2606.07057)
- **What's New**: 본 논문에서는 자동 생성된 키프레이즈의 평가 품질을 개선하기 위해 새로운 평가 지표인 Semantic R-Precision (SemR-p)을 소개합니다. SemR-p는 의미적 유사성(semantic similarity)을 랭크 인식(Rank-aware) R-Precision 프레임워크에 통합하여 작성되었습니다. 이 지표는 인간 중심의 관점에서 디자인되어 사용자가 실제로 인포메이션의 유의미성을 어떻게 평가하는지를 반영하려고 합니다.

- **Technical Details**: SemR-p는 정보 검색(Information Retrieval)에서의 평가 지표의 원리를 기반으로 하여 의미적으로 관련성이 높은 키프레이즈가 출력 목록의 앞쪽에 나타날수록 높은 점수를 부여합니다. 이 접근 방식은 기존의 키프레이즈 평가 방식들이 단어의 정확한 일치를 기반으로 하여 의미적 동등성을 간과하는 문제를 해결하고자 합니다. 기계 학습 모델 및 데이터셋에 대한 광범위한 분석을 통해 SemR-p의 의미적 민감성, 랭킹 인식 및 구별력을 평가하였습니다.

- **Performance Highlights**: 실험 결과 SemR-p는 기존의 전통적인 어휘 및 의미적 유사성 매칭 지표들과 함께 사용될 수 있는 보완적인 시각을 제공하며, 키프레이즈 예측 평가에서 사용자 중심의 관련성을 보다 잘 반영하고 있음을 보여주었습니다. 또한, SemR-p는 다양한 모델과 데이터셋에서 비교적 높은 신뢰성과 일관성을 입증하였으며, 기존의 방식으로는 놓치기 쉬운 의미적 관계를 포착할 수 있는 가능성을 제공한다는 점에서 주목할 만합니다.



### SSRLive: Live Streaming Recommendation with Dynamic Semantic ID (https://arxiv.org/abs/2606.06970)
- **What's New**: 이번 논문에서는 SSRLive라는 새로운 추천 시스템을 소개합니다. 이 시스템은 라이브 스트리밍 플랫폼을 위한 동적 의미 ID(semantic ID)를 활용해 사용자와 스트리머 간의 실시간 상호작용을 모델링합니다. 또한 생성(generative) 모듈과 구분(discriminative) 모듈을 통합하여, 신속하게 변화하는 라이브 콘텐츠를 효과적으로 반영하였습니다.

- **Technical Details**: SSRLive는 인코더-디코더 아키텍처를 기반으로 하여 정적(static) 및 동적(dynamic) 의미 ID를 생성합니다. 정적 의미 ID는 스트리머의 기본 정보를 기반으로 하며, 동적 의미 ID는 실시간으로 변화하는 라이브 콘텐츠를 반영하여 모델이 시간에 따른 변화를 더 잘 나타낼 수 있도록 합니다. 구분 모듈은 사용자와 스트리머 간의 상호작용 특징들을 효과적으로 모델링하여, 작업 별 맞춤 형태 표현(task-specific representation)을 추출합니다.

- **Performance Highlights**: 실제 라이브 스트리밍 환경에서의 A/B 테스트 결과, SSRLive는 사용자 시청 시간(보너스 3.38%), 총 상품 판매량(GMV; 보너스 0.72%), 팔로워 증가율(보너스 3.12%) 및 상호작용량(보너스 2.92%)에서 유의미한 개선을 보였습니다. 이러한 결과는 SSRLive의 실제적인 효과와 상업적 가치를 입증합니다. 이제 이 시스템은 수억 명의 활성 사용자를 대상으로 원활히 운영되고 있습니다.



### DREAM: Dynamic Refinement of Early Assignment Mappings (https://arxiv.org/abs/2606.06947)
Comments:
          12 pages, 4 figures, 5 tables

- **What's New**: 이번 연구에서는 Generative Recommendation (GR)이 제안하는 딥러닝 기반의 추천 시스템에서 발생하는 초기 고정 경로(commitment) 문제를 다룹니다. 특히, 서늘한 시작(cold-start) 항목에 대해 각각의 항목에 단일한 정적 식별자(SID)를 부여하는 방법론의 한계를 지적하며, 이를 DREAM(동적 정제 초기 할당 매핑)이라는 새로운 프레임워크로 해결하려고 합니다.

- **Technical Details**: DREAM은 단계적으로 진행되는 세 가지 주요 프로세스로 구성됩니다. 첫 번째 단계는 Collaborative-Aware Refined Tokenization (CART)으로, 조합된 기하학적 감독을 통해 SID 공간을 재구성하며 초기에 지원되는 SID 후보군을 생성합니다. 두 번째 단계인 User-Conditioned Candidate Condensation (UC3)은 다중 맥락의 사용자가 제공하는 신뢰도 가중 투표를 통해 CID를 업데이트하도록 설계되어 있으며, 마지막 단계인 Cold-Preserved Dynamic Beam Evolution (CPDE)은 다중 경로 회복을 가능하게 하여 모든 SID 대안을 유지합니다.

- **Performance Highlights**: DREAM은 세 가지 Amazon 벤치마크에서 18개의 초 시작 메트릭에서 모두 최고 점수를 기록하였으며, 강력한 기준선 대비 4배에서 12배의 성능 향상을 달성했습니다. 또한, DREAM은 견고한 정제 과정을 통해 초 시작 추천 성능을 개선하면서도 기존 warm-item 성능을 충분히 보호할 수 있는 방법을 제시합니다.



### Towards Retrieving Interaction Spaces for Agentic Search (https://arxiv.org/abs/2606.06880)
- **What's New**: 이 논문은 검색 에이전트를 위한 정보 검색 방식을 새로운 접근 방식으로 제안합니다. 기존의 방법은 문서 집합을 선택하고 그 문서에서 정보를 읽는 것이지만, 이 연구는 에이전트가 원시 데이터베이스를 탐색할 수 있도록 하는 '상호작용 공간(Interaction Space)'의 구축을 강조합니다. 이를 통해 에이전트가 효과적으로 정보를 탐색하고 도구를 활용할 수 있도록 하여 효율성을 개선할 수 있습니다.

- **Technical Details**: 이 연구에서 제안하는 RISE(상호작용 공간 검색)는 BM25 알고리즘을 사용해 선택된 문서 집합을 구성하고, 이를 탐색할 수 있는 방식으로 처리합니다. 이 방식은 에이전트가 특정 도구를 사용하여 문서에 쉽게 접근할 수 있게 해줍니다. 에이전트는 이제 고정된 증거 창을 소비하는 대신, 에이전트가 탐색할 수 있는 경계가 있는 문서 집합으로 작업할 수 있습니다.

- **Performance Highlights**: RISE는 BrowseComp-Plus 데이터셋에서 gpt-5.4-mini를 사용하여 78%의 정확도를 달성하며, 이는 기존의 pure-shell DCI 기준과 동등합니다. RISE-BM25는 문서 처리 단계를 제거했을 경우 정확도가 11-44점 감소하며, 대규모 문서 집합에서도 안정적인 성능을 유지합니다. 1M 문서 규모로 확장 시 RISE-BM25는 81%의 정확도를 기록하는 반면, DCI는 60%로 저하됩니다.



### Mind the Gap: Bridging Behavioral Silos with LLMs in Multi-Vertical Recommendations (https://arxiv.org/abs/2606.06779)
- **What's New**: 이 논문은 DoorDash와 같은 다중 수직(e-commerce) 플랫폼에서 새로운 제품 수직이 개인화 혁신의 기회를 제공한다는 점에 주목합니다. 특히 데이터가 적은 카테고리에서 사용자에게 추천 품질을 향상시키기 위한 새로운 프레임워크를 제안하며, 이는 데이터가 풍부한 수직(예: 음식점)으로부터 지식을 전이하는 방법론에 기반을 두고 있습니다.

- **Technical Details**: 핵심 기여는 사용자 행동을 다양한 수직에서 풍부하고 숨겨진 신호의 원천으로 간주하고, 이를 대형 언어 모델(LLMs)을 통해 구조화된 표현으로 변환하여 활용하는 것입니다. 연구에서는 계층적 검색-증강 생성(RAG) 프레임워크를 활용하여 사용자 선호도를 다단계 제품 분류 체계에서 추론하는 새로운 방법론을 다룹니다. 이를 통해 '콜드 스타트(cold start)' 문제를 해결하고 사용자 표현을 풍부화하며, 더 나은 추천 시스템을 구축할 수 있는 실질적인 경로를 제시합니다.

- **Performance Highlights**: 상세한 오프라인 및 온라인 평가를 통해, 이 방법론이 신흥 비즈니스 수직에서 개인화 및 참여도를 크게 향상시킨다는 것을 증명했습니다. 특히 LLM에서 생성한 특성이 포함된 새로운 MTL 랭킹 모델이 기존 모델보다 AUC-ROC 및 MRR과 같은 평가지표에서 우수한 성능을 보였습니다. 이를 통해 추천 시스템의 효용성과 효율성을 동시에 증대시켰습니다.



### Your UnEmbedding Matrix is Secretly a Feature Lens for Text Embeddings (https://arxiv.org/abs/2606.07502)
Comments:
          preprint

- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)이 텍스트 임베딩 태스크에서 성능 저하를 겪는 원인을 분석하고, 이를 해결하기 위한 EmbedFilter라는 새로운 방법을 제안합니다. 연구자들은 LLM의 텍스트 임베딩이 자주 등장하지만 의미가 없는 토큰과 정렬된다는 흥미로운 관찰에서 출발하여, 높은 빈도 수의 토큰이 세밀한 의미 캡처를 저해한다는 점을 지적합니다.

- **Technical Details**: EmbedFilter는 LLM의 임베딩 품질을 향상시키기 위해 설계된 간단한 선형 변환으로, LLM이 생성하는 임베딩의 잠재 공간을 정제합니다. 이를 통해, 자주 등장하는 토큰의 영향을 제어하고 진정한 의미 표현을 발현할 수 있습니다. 해당 변환은 사전 학습된 매개변수 내에 존재하므로 추가적인 학습 없이도 쉽게 적용이 가능합니다.

- **Performance Highlights**: 많은 실험을 통해 EmbedFilter가 기존의 임베딩 차원을 크게 줄이면서도 LLM의 제로샷(formally known as 'zero-shot') 성능을 비약적으로 향상시킨다는 것을 입증했습니다. EmbedFilter는 다양한 모델과 실험 구성에서 강력한 성능을 보여주며, 이로 인해 LLM을 대규모 텍스트 임베딩 응용에서 효과적으로 사용할 수 있는 가능성을 높입니다.



### TA-RAG: Tone-Aware Retrieval-Augmented Generation for Peer-Support Health Communication (https://arxiv.org/abs/2606.06794)
Comments:
          5 pages, 5 figures, CIKM 2026 submission manuscript

- **What's New**: 이번 논문은 민감한 피어 지원 건강 커뮤니케이션을 위해 사실 기반 생성(RAG) 이상의 기능을 요구하는 TA-RAG라는 새로운 프레임워크를 제안합니다. TA-RAG는 톤 조절을 명시적으로 통합하여 피어 지원의 효과성을 높이는 데 필요한 네 가지 핵심 구성 요소—낙인 없는 재작성, 읽기 용이성 조정, 수령인 맞춤화, 그리고 공감 재구성을 포함합니다. 이 연구는 TA-RAG가 모델 미세 조정 없이도 신뢰할 수 있는 문서에 기반을 둔 응답을 생성할 수 있도록 돕는다는 점에서 중요합니다.

- **Technical Details**: TA-RAG는 사용자의 질문을 명확히 하고 최상의 정보 조각을 검색한 후, LLM을 통해 근거 있는 응답 초안을 생성하는 세 가지 주요 단계를 거칩니다. 첫 번째 단계에서는 불분명한 질문을 명확히 하기 위해 사용자에게 추가 질문을 요청하고, 두 번째 단계에서는 검색된 정보를 바탕으로 초안을 작성합니다. 마지막으로, 이 초안에 톤 조정 레이어를 추가하여 응답을 비난이 없고, 읽기 쉬우며, 공감적이고 수령인에게 적합하게 만듭니다.

- **Performance Highlights**: TA-RAG는 HIV 관련 질문을 대상으로 한 구성 요소 수준의 테스트에서 응답의 질을 개선하는 성과를 보였습니다. 특히, 각 톤 조절 요소가 대상 의사소통의 질을 향상시키며 핵심 내용을 유지하는 데 기여함을 확인했습니다. 이러한 결과는 친구 지원 건강 커뮤니케이션에 적합한 RAG 출력 생성을 위한 가능성을 강조합니다.



New uploads on arXiv(cs.CV)

### UniSHARP: Universal Sharp Monocular View Synthesis (https://arxiv.org/abs/2606.07514)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 SHARP 프레임워크를 기반으로 한 새로운 단일 렌더링 기법 UniSHARP를 제안합니다. UniSHARP는 다양한 카메라 시스템에 대해 단일 이미지에서 3D 구조, 가시성, 외관을 추론할 수 있도록 설계되었습니다. 특히, 다양한 이미지를 통합된 구형 잠재 공간에서 정렬하는 방식을 사용하여 피인홀 카메라의 가정에 얽매이지 않고 성능의 일관성을 높입니다.

- **Technical Details**: UniSHARP는 3D Gaussian primitives를 사용하여 한 차원에서 라인 기반의 범용 표현을 구성하고, 2D 시맨틱 및 3D 공간 특성을 결합하여 고해상도의 중첩된 Gaussian 클라우드를 생성합니다. 이를 통해 다양한 프로젝션 유형에서의 렌더링을 감독하는 혼합 카메라 훈련 전략이 도입되어 있습니다. 또한, UniSHARP는 예측된 광선 필드를 통해 카메라 형식을 추론하고 렌더링 기하를 회복하는 포즈 없는 단일 이미지 추론을 지원합니다.

- **Performance Highlights**: UniSHARP는 제안된 벤치마크에서 다양한 시스템의 검증 데이터로 테스트되었으며, 이전의 방법들보다 훨씬 우수한 성능을 보여줍니다. 이 벤치마크는 좁은 시각, 넓은 시각, 어안, 파노라마 카메라를 포함하며, 카메라의 FoV에 따라 세분화되어 렌더링 품질의 변화를 상세하게 분석할 수 있도록 하고 있습니다. UniSHARP는 다양한 카메라 시스템에 일관되게 일반화되는 성능을 확립합니다.



### MemDreamer: Decoupling Perception and Reasoning for Long Video Understanding via Hierarchical Graph Memory and Agentic Retrieval Mechanism (https://arxiv.org/abs/2606.07512)
- **What's New**: 현재의 Vision-Language Models는 몇 시간에 걸친 비디오를 처리하는 데 어려움을 겪고 있습니다. 이를 해결하기 위해, 연구진은 MemDreamer를 소개하며, 이는 인식과 추론을 분리하여 긴 비디오 이해를 에이전틱 탐색 과정으로 전환합니다. MemDreamer는 비디오를 점진적으로 스트리밍하며 Hierarchical Graph Memory를 구축하는 플러그 앤 플레이(framework) 구조입니다.

- **Technical Details**: 이 프레임워크는 의미 추상화를 위한 세 가지 주요 계층으로 구성된 구조로, 시공간(spatiotemporal) 및 인과관계(causal relations)를 포착하는 기본 그래프에 의해 뒷받침됩니다. 추론 과정에서는 Agentic Tool-Augmented Retrieval을 사용하여 계층을 내비게이션하고, 노드를 탐색하며, Observation-Reason-Action 루프를 통해 논리적인 간선을 통과합니다.

- **Performance Highlights**: 실험 결과, MemDreamer는 네 가지 주요 벤치마크에서 SOTA(State Of The Art) 결과를 달성하였고, 인간 전문가와의 격차를 단 3.7점으로 좁혔습니다. 이 모델은 전체 맥락의 2%로 추론 컨텍스트 윈도우를 제한하면서도 12.5점의 절대 정확도 증가를 보여줍니다. 또한, 통계적 분석을 통해 VLM의 논리 추론 성능과 긴 비디오 이해 벤치마크 간에 강한 양의 선형 상관관계를 발견하였습니다.



### Streaming Video Generation with Streaming Force Contro (https://arxiv.org/abs/2606.07508)
- **What's New**: 이번 논문에서는 StreamForce라는 스트리밍 비디오 생성 프레임워크를 소개합니다. StreamForce는 연속적인 힘 입력을 통해 물리적으로 기반한 제어를 가능하게 합니다. 기존 비디오 모델과는 달리, StreamForce는 다양한 힘에 대해 고유한 통합 모델로 응답하며, 시간에 따라 변화하는 힘에도 즉각적으로 일관되게 반응합니다.

- **Technical Details**: StreamForce는 단일 공식을 통해 지역적 및 전역적 힘을 함께 모델링하는 통합 힘 표현을 도입합니다. 또한, 힘이 변화하는 데이터셋을 구축하여 모델이 훈련 과정 동안 동적 힘 전환을 경험하도록 하였습니다. 이러한 시스템은 강력한 동영상 생성 역량을 유지하면서도 힘에 대한 응답성을 강화하기 위해 힘 인식을 고려한 증류 파이프라인을 개발했습니다.

- **Performance Highlights**: StreamForce는 단일 H200 GPU에서 832×480 해상도로 16.6 FPS를 달성하며, 힘 일관성과 동작 사실성 면에서 최신 성능을 자랑합니다. 이를 통해 생성된 비디오와 상호작용할 수 있는 가능성을 열며, 사용자가 힘을 적용하고 그 결과로 나타나는 동작을 즉시 관찰할 수 있게 합니다. 전반적으로 StreamForce는 생성 비디오 모델을 상호작용할 수 있는 세상 모델로서 발전시키고 있습니다.



### Differences in Detection: Explainability Where it Matters (https://arxiv.org/abs/2606.07503)
Comments:
          Accepted to IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops 2026 - How Do Vision Models Work? (HOW)

- **What's New**: 이번 연구에서는 두 개의 객체 탐지 모델을 직접 비교할 수 있는 직관적인 방법인 Differences in Detection (DnD)을 제안합니다. 이는 평균 평균 정밀도(mean Average Precision, mAP) 및 TIDE 오류 분석과 함께 작동하며 두 모델 간의 직접적인 비교를 가능하게 합니다. DnD는 여러 모델의 인식 결과를 결합하여 공통적인 오류를 분석하므로, 모델 간의 유사성과 차이를 보다 쉽게 이해할 수 있습니다.

- **Technical Details**: DnD 접근법은 두 모델 M1과 M2 간의 예측 세트 D1 및 D2에 대해 동일한 매칭 알고리즘을 적용하여 시작됩니다. 이를 통해 각 모델이 인식한 ground truth 레이블의 교차점을 계산하고, 누락된 레이블에 대한 차이 세트를 파악합니다. 이 방법론은 IoU(Intersection over Union) 기반의 정의를 지원하는 바운딩 박스, 인스턴스 마스크 또는 객체 경계를 사용하여 예측을 동원합니다.

- **Performance Highlights**: 실험 결과, DnD 방법은 ConvNext-v2와 ViTDet 모델 간의 차이를 명시적으로 보여줍니다. DnD를 사용하면 모델이 어떤 객체를 인식하는지, 서로 유사한 오류를 범하는지를 분석할 수 있으며, 이는 mAP 및 TIDE 단독으로는 이룰 수 없는 깊이 있는 비교를 가능하게 합니다. DnD는 모델의 인스턴스 수준 비교를 구조화 할 수 있도록 도와, 각 모델이 특정 인스턴스를 어떻게 인식하는지에 대한 통찰력을 제공합니다.



### Implicit Data Synthesis for Contrastive Unsupervised Data Augmentation (https://arxiv.org/abs/2606.07498)
Comments:
          11 pages, 3 figures, 2 tables

- **What's New**: 이번 연구에서는 무작위로 선택된 레이어에 결합된 잡음 항을 추가하여 과학 관측 데이터에 적합한 대조 샘플을 생성하는 새로운 방법을 제안합니다. 이 접근법은 기존의 데이터 공간 변환과 달리, 데이터의 구조를 더 잘 유지하면서도 유용한 표현을 추출할 수 있도록 합니다. 이는 예를 들어 SimCLR 기반의 파이프라인을 사용하여 유성의 레이더 관측에서 성능 향상을 보여줍니다.

- **Technical Details**: 제안된 방법인 Implicit Data Synthesis (IDS)는 Multilayer Perceptron (MLP)에서 잡음 항을 추가하여 데이터의 표현 공간에서 데이터 증강을 수행합니다. 이 접근법은 데이터 공간의 변형 없이도 대조 손실을 위한 긍정 쌍을 생성하는 데 집중하며, 이는 과학적 데이터 환경에서 매우 효과적임을 입증합니다. 또한 IDS는 일반적인 SimCLR 방법론과 함께 사용되며, 랜덤 예측에서의 견고한 표현을 생성하는 데 기여합니다.

- **Performance Highlights**: 연구 결과, IDS는 기존 데이터 공간 증강 방법과 유사한 성능을 보이며, 특정한 환경에서는 오히려 더 나은 결과를 나타냅니다. 특히 Jicamarca Radio Observatory의 유성 데이터와 CIFAR-10 이미지 데이터셋에서 효과를 검증하며, 효과적인 증강 방안을 통해 기계 학습 모델의 성능을 높이는 데 기여할 수 있음을 입증했습니다. 이로 인해, IDS는 다양한 과학 관측 데이터 분석에서도 권장할만한 방법론으로 자리잡을 수 있을 것입니다.



### TEVI: Text-Conditioned Editing of Visual Representations via Sparse Autoencoders for Improved Vision-Language Alignmen (https://arxiv.org/abs/2606.07451)
Comments:
          20 pages, 13 figures, 14 tables

- **What's New**: 이 논문에서는 TEVI라는 새로운 프레임워크를 제안합니다. TEVI는 자막이 이미지 임베딩에서 무엇을 유지해야 하는지를 결정하는 신호로 작용합니다. 이를 통해 자막에 설명된 속성은 유지하고 다른 속성은 버리도록 설정할 수 있습니다. 이 방법은 이미지-텍스트 정렬을 개선하는 데 효과적입니다.

- **Technical Details**: TEVI는 희소 자동인코더(sparse autoencoders)를 사용하여 이미지 임베딩을 구성 요소 개념으로 분해하고, 조건부 모듈을 훈련하여 재구성에 사용할 SAE 잠재 변수를 선택합니다. 이를 통해 이미지의 특정 속성을 살리고, 필요한 정보를 유지하면서 다른 속성에 대한 정보를 폐기할 수 있습니다. 또한, 제어된 실험을 통해 TEVI의 효과를 증명했습니다.

- **Performance Highlights**: TEVI를 자연 이미지에 대해 훈련된 CLIP 모델에 적용한 결과, 짧은 자막(MS COCO, Flickr)과 긴 자막(DOCCI, IIW) 벤치마크 모두에서 검색 성능이 향상되었습니다. 특히 긴 자막 기준에서 더 큰 성능 향상을 보였으며, 이것은 풍부한 자막이 수정에 강력한 신호를 제공한다는 것을 시사합니다. 또한, RoCOCO 벤치마크에서 언어적 왜곡에 대한 강한 견고함을 보여주었습니다.



### Skill-3D: Evolving Scene-Aware Skills for Agentic 3D Spatial Reasoning (https://arxiv.org/abs/2606.07436)
- **What's New**: 이 논문은 MLLM (Multimodal Large Language Model) 에이전트가 도구 사용을 통해 3D 이해 작업을 수행하도록 하는 에이전틱 (agentic) 3D 공간 이해를 탐구합니다. 현재의 방법들은 도구 사용에 대한 선호가 편향되어 있으며, 이로 인해 비에이전틱 기법에 비해 향상된 성능을 보여주지 못하고 있습니다. 이를 해결하기 위해 제안된 Skill-3D 프레임워크는 자기 진화하는 장면 인식 기술을 학습하여 에이전트를 더욱 효과적으로 만듭니다. 이 시스템은 작업 장면을 식별하고, 성공적인 도구 사용 경로를 기억하여 동적으로 기술을 업데이트하는 과정을 포함합니다.

- **Technical Details**: Skill-3D는 장면 메모리 (Scene Memory)를 구성하고, 이러한 메모리를 기반으로 기술 라이브러리 (Skill Library)를 공동 발전시키는 구조입니다. 훈련 중 에이전트는 각 질문의 장면을 식별하고 해당 도구 사용 경로와 결과를 저장하며, 성공적인 경로는 집계되어 재사용 가능한 장면 인식 기술로 증류(주입)됩니다. 각 기술이 형성된 후에는 유사한 문제에 대해 에이전트를 안내하여 새로운 경로를 생성하고, 이 신규 경로의 성공과 실패는 기술을 업데이트하는 데 활용됩니다.

- **Performance Highlights**: Skill-3D는 3D 공간 추론 벤치마크에서 기존 MLLM보다 일관되게 뛰어난 성능을 발휘하며, 도구 사용의 효율성을 39%에서 78%로 증가시켰습니다. 예를 들어, Gemini-3-Flash에서는 MMSI-Bench에서 67% 향상을 이루었습니다. 또한, 기술 지향 에이전틱 후 훈련을 통해 Qwen3-VL-8B는 VSI-Bench에서 43% 개선되었습니다. 이러한 결과들은 Skill-3D의 유효성과 도구 사용 개선의 중요성을 잘 보여줍니다.



### The Lipreading Gap: Do VSR Models Perceive Visual Speech Like Human Lipreaders? (https://arxiv.org/abs/2606.07435)
Comments:
          Accepted at INTERSPEECH 2026

- **What's New**: 최근 Visual Speech Recognition (VSR) 모델은 인간의 lipreader를 초월하는 성능을 보이고 있으나, 이러한 성장이 과연 인간과 유사한 시각적 언어 인식을 확립하는지를 탐구하고 있습니다. 이 연구에서는 세 가지 VSR 시스템을 MaFI 데이터셋을 통해 인간의 성능과 비교하여, 단순한 문자의 정확도가 아닌 단어, 글자, 음소, 그리고 viseme 수준에서 성능 분석을 실시하였습니다. 모델이 전반적으로 더 높은 정확도를 달성하였지만, 특정 단어에서 인간과 모델 간의 차이가 존재하여, VSR의 성격을 보다 명확히 이해할 수 있는 기회를 제공합니다.

- **Technical Details**: VSR 시스템의 분석은 MaFI 데이터셋을 사용하여 진행되었고, 이 데이터셋은 410명의 참가자들에 의해 수집된 실험 결과를 기반으로 합니다. 각 단어의 Mouth and Facial Informativeness (MaFI) 점수를 활용하여, 참가자들이 lipreading을 통해 단어를 식별하는 데 필요한 시각적 정보의 다양성을 평가했습니다. 세 가지 모델로는 Auto-AVSR (Supervised Learning), AV-HuBERT (Self-Supervised Learning), VSP-LLM (Large Language Model Decoder)이 있으며, 이러한 모델들은 모두 공개된 가중치로 평가되었습니다.

- **Performance Highlights**: 이 연구는 VSR 모델과 인간 간의 인식 정확도 차이를 명확히 드러내며, 특히 모델의 오류는 시각적 정보보다는 훈련 빈도에 더 큰 영향을 받는 것으로 나타났습니다. VSR 모델은 인간이 인지하는 시각적 명확성과의 강한 상관관계를 보이지 않았고, 이는 기본적으로 다른 언어 처리 메커니즘을 의미합니다. 전통적인 방법에 비해, 이 연구는 인간과 VSR 모델 간의 인식의 정밀한 일치를 위한 기초 데이터를 제공하며, 향후 VSR 기술 개선에 기여할 수 있는 중요한 통찰력을 제공합니다.



### Watch, Remember, Reason: Human-View Video Understanding with MLLMs (https://arxiv.org/abs/2606.07433)
- **What's New**: 이 논문은 멀티모달 대규모 언어 모델(MLLMs)이 비디오 이해 분야를 어떻게 혁신하고 있는지를 다루고 있습니다. 연구는 짧은 클립에서 긴 멀티모달 비디오로 넘어가며, 이러한 변화는 모델이 희소한 증거, 긴 범위 의존성, 멀티모달 정렬, 그리고 제한된 계산 자원 하에서 신뢰할 수 있는 추론을 다뤄야 함을 요구합니다. 또한, 비디오 MLLMs가 단순한 기준점으로서 비디오 작업을 취급하는 대신, 인간이 비디오를 이해하는 기제를 중심으로 분석하는 것을 목표로 합니다.

- **Technical Details**: 논문은 '보기(watching)', '기억하기(remembering)', '추론하기(reasoning)'라는 세 가지 기능 기반으로 LLM 기반 비디오 이해를 분석합니다. 이 구조는 비디오 이해 시스템을 입체적으로 바라보고, 지각적 표현, 기억 상태, 추론 과정을 포괄하는 공식을 제공합니다. 또한, 효율적인 비디오 처리, 메모리 모델링, 스트리밍 이해 및 신뢰성 있는 추론의 도전과제를 식별합니다.

- **Performance Highlights**: 논문은 여러 비디오 응용 영역을 다루며, 각 비디오 유형에 따라 요구되는 지각, 기억, 그리고 추론방식이 다름을 보여줍니다. 특히, 연구는 교육, 의료, 운동, 서사 비디오 등 구체적인 도메인을 살펴보며, 다양한 훈련 데이터셋 및 평가 지표를 정리합니다. 마지막으로, 확장 가능하고 메모리에 민감하며 증거 기반의 비디오 인공지능을 위한 미래 방향을 제시합니다.



### OpenGlass: Open-Source Smart Glasses for On-Device Event-Based Gesture Recognition (https://arxiv.org/abs/2606.07431)
- **What's New**: 이 논문은 스마트 안경 플랫폼에 대한 오픈소스 프로젝트를 소개하며, 이 플랫폼은 새로운 센서와 알고리즘의 빠른 프로토타입 제작을 지원합니다. 이벤트 기반(event-based) 및 프레임 기반(frame-based) 카메라를 통합할 수 있는 모듈형 설계를 채택하여 PCB 재설계 없이 기능을 확장할 수 있습니다. 또한, 하드웨어와 소프트웨어를 함께 설계한 전력 관리 시스템을 통해 200 mAh 배터리로 최대 11.8시간의 ML(기계 학습)을 지원합니다.

- **Technical Details**: 스마트 안경 플랫폼은 고도 통합을 실현하기 위해 세 가지 주요 부품으로 구성되어 있습니다. 주요 구성 요소는 전면 보드(main board), 카메라 모듈, 전원 관리 시스템입니다. 이 시스템은 사용자와 환경을 지속적으로 인식하고 실시간으로 적응하여 AI의 보조 역할을 수행하는 새로운 세대의 스마트 안경을 구현하기 위한 기술적 도전 과제를 해결합니다.

- **Performance Highlights**: 이 프로토타입의 성능은 LynX 데이터셋을 사용하여 평가한 결과, R(2+1)D 모델이 83.94%의 교차 주제 정확도를 달성하였으며, 33.9 ms의 종단 간 지연(latency)으로 우수한 성과를 보였습니다. 또한, 모호한 클래스 제거와 시간적 증가가 가장 큰 성과 (+8.9 pp)를 가져왔으며, 모든 하드웨어 설계, 펌웨어 및 모델이 오픈 소스로 제공됩니다.



### DisPOSE: Projected Polystochastic Diffusion for Self-Supervised Multi-View 3D Human Pose Estimation (https://arxiv.org/abs/2606.07419)
- **What's New**: 이 논문에서는 DisPOSE라는 새로운 자기 지도 학습(self-supervised learning) 프레임워크를 소개합니다. 이 프레임워크는 다중 카메라 시점에서의 사람 할당 문제를 생성적 확산 과정(generative diffusion process)으로 모델링하여, 3D 자세 추정에서의 정확성을 높입니다. 특히, 이 논문은 기존의 방법들이 경험하는 실제 환경에서의 일반화 문제를 해결하는 데 초점을 맞추고 있습니다.

- **Technical Details**: DisPOSE는 Hypergraph-Convolutional Decoder를 이용하여 다중 뷰에서의 3D 스켈레톤을 회귀합니다. 입력 RGB 이미지는 CNN 백본(ResNet-50)으로 처리되며, 이러한 과정을 통해 각 뷰에서 관절 히트맵을 추론합니다. 궁극적으로 DisPOSE는 각 시점에서의 검출 결과를 모두 고려하여 최적의 하이퍼엣지 집합을 찾아내는 방식으로 다중 뷰 사람 할당 문제를 해결합니다.

- **Performance Highlights**: 실험적으로 DisPOSE는 CMU Panoptic 데이터셋에서 19% AP25의 성능 향상을 보이며, 새로운 카메라 설정에 대해서도 뛰어난 일반화 능력을 보여줍니다. 10%의 pseudo-label만으로도 99%의 성능을 유지하는 데이터 효율성을 입증했으며, 이는 기존 방법들과 비교했을 때 매우 우수한 결과입니다. 논문에서 제안한 MM-OR Pose는 실제 복잡한 수술 환경을 캡처한 도전적인 새로운 벤치마크입니다.



### RealDocBench: A Benchmark for Field-Level QA and Layout Understanding on Real-World Regulated Documents (https://arxiv.org/abs/2606.07401)
- **What's New**: RealDocBench는 실제 규제가 있는 문서로 구성된 신규 벤치마크입니다. 이를 통해 문서 파싱 시스템을 평가할 수 있는 두 가지 트랙을 도입하였습니다. QA 트랙은 정확한 필드 값을 추출하기 위한 1,356개의 질문이 포함되어 있으며, 레이아웃 트랙은 1,500개의 페이지 이미지를 평가합니다.

- **Technical Details**: QA 트랙은 다양한 네 개의 도메인에서 581개의 실제 문서와 연결된 질문을 포함하며, 각 질문은 타입이 지정된 gold_dict와 함께 제공됩니다. 레이아웃 트랙은 COCO 스타일의 바운딩 박스와 함께 1,500개의 이미지로 구성되어 있으며, 행렬 기반 매칭 기법인 Hungarian matcher를 사용합니다. 벤치마크는 정확도, 페이지당 비용, 대기 시간(latency) 등의 지표로 평가됩니다.

- **Performance Highlights**: RealDocBench는 18개의 다양한 시스템을 평가하며, 이러한 시스템의 성능 스프레드를 드러내어 각 시스템의 강점을 비교할 수 있습니다. 문서 파싱의 실제 필드 값 추출에서 발생하는 복잡성을 분석하고, 비용 및 대기 시간의 Trade-off를 통해 최적의 운영 지점을 찾습니다. 향후 연구에 있어 재현 가능하고 체계적인 비교 분석을 위한 데이터셋과 평가 프레임워크도 출시되었습니다.



### Mind the Gap: Disentangling Performance Bottlenecks in Video Instance Segmentation (https://arxiv.org/abs/2606.07394)
- **What's New**: 이 논문에서는 Video Instance Segmentation (VIS)의 성능 손실에 대한 기여를 명확히 해주는 진단 프레임워크를 도입합니다. 기존의 비효율적인 성능 분석 방식을 넘어, Integer Linear Program (ILP)을 통해 이론적인 최적 해를 제공합니다. 이 연구는 VIS의 주요 도전 과제가 안정적인 장기 시간 연관성이라는 점을 강조하고 있습니다. 논문에서는 TrackLens라는 시각화 도구를 소개하여, 모델의 내부 예측 구조를 보다 명확히 이해할 수 있게 합니다.

- **Technical Details**: VIS는 영상에서 객체 인스턴스를 함께 감지, 분할 및 추적하는 임무로, 기존의 오프라인과 온라인 방법론으로 나뉩니다. 이 논문에서는 주로 tracking, classification, mask quality의 세 가지 오류 소스를 분리하여 분석합니다. 제안하는 Integer Linear Programming (ILP) 오라클은 각 오류 원인을 계층적으로 고립시키며, 실험 대상은 YouTube-VIS 2019/2021와 OVIS 데이터셋입니다. TrackLens는 전체 내부 예측 공간을 드러내며, 각 쿼리 간의 관계를 시각적으로 표현합니다.

- **Performance Highlights**: 연구 결과, 온라인 방법에서 tracking 불안정성이 주요 병목 현상임을 보여줍니다. occlusion 상황에서 20 AP 이상의 격차가 발생하며, 비디오 길이와 인스턴스 밀도가 증가할수록 이 격차가 심해지는 것으로 나타났습니다. 반면, 오프라인 방법은 훨씬 적은 tracking 격차를 보이며, 세분화와 추적을 분리하는 아키텍처가 더 안정적인 결과를 제공합니다. 이러한 분석을 통해 VIS의 성능 향상을 위한 향후 연구 방향을 제시합니다.



### Mitosis Detection in the Wild: Multi-Tumor and Context-Aware Generalization in the MIDOG 2025 Challeng (https://arxiv.org/abs/2606.07368)
- **What's New**: MIDOG 2025 챌린지는 자동 분열 검사(mitosis detection) 분야에서 새로운 초점으로, 전통적인 핫스팟(hotspot) 영역을 넘어 무작위(sampled randomly)로 선정된 조직 영역(tissue regions) 및 어려운 영역(challenging regions)에서의 검출을 요구합니다. 이 챌린지는 365개의 케이스를 포함하여, 다양한 종(인간, 개, 고양이)의 암 유형과 여러 스캐닝 플랫폼에서 수집된 데이터를 활용하여 알고리즘의 성능을 평가하고자 했습니다. 이는 현실적인 임상 적용(clinical application)에서 모델의 강건함을 극대화하기 위한 노력으로, 현재의 병리학적 검사에서 직면하고 있는 주요 문제들을 해결합니다.

- **Technical Details**: MIDOG 2025 챌린지에서는 총 18개 팀이 검출 트랙에 제출했으며, F1 점수는 최대 0.740에 달했습니다. 비정상적인 분열을 보이는 세포의 분류(atypical mitotic figures, AMFs)가 두 번째 트랙으로 도입되었으며, 여러분 21개의 제출이 있었고 균형 잡힌 정확성(balanced accuracy) 값은 0.908에 이르렀습니다. 이 트랙에서의 성능 분석은 전통적인 핫스팟 지역에서는 신뢰할 수 있는 성능을 보였으나, 도전적인 ROI에서는 성능 저하가 발생한다는 사실을 드러냈습니다.

- **Performance Highlights**: 성능 분석 결과, 통계적으로 다루기 어려운 지역에서 잘못된 예측(false positive) 비율이 세 배 증가됨을 보여주었습니다. AMF에 대한 앙상블(ensembling) 기법을 평가한 결과, F1 점수와 balanced accuracy에서 각각 평균 1.5 및 1.3 포인트의 향상이 나타났습니다. 그러나 TTA(transformational training augmentation)는 유의미한 개선이 없었습니다. 이러한 결과는 현실 세계에서의 mitosis 검출이 여전히 상당한 도전 과제임을 입증합니다.



### Dash2Sim: Closed-Loop Driving Simulation from in-the-wild Dashcam Videos (https://arxiv.org/abs/2606.07366)
- **What's New**: 새로운 연구인 Dash2Sim은 현장 모노큘러 대시캠 비디오를 메트릭( Metric) 및 지리적으로 참조된 4D 드라이빙 로그로 변환하는 프레임워크입니다. 이 프레임워크는 주석(annotations) 없이 독립적으로 유지되는 맵과 각 비디오를 검증하여 정확한 4D 장면을 회복하기 어려운 단점을 극복합니다. 이를 통해 연구진은 17개 도시에서 4,244개의 장면과 270만 개의 3D 객체를 포함하는 ROADWork4D 벤치마크 데이터셋을 생성했습니다.

- **Technical Details**: Dash2Sim은 대시캠 비디오를 활용하여 그 데이터에서 작업 구역(work zones)과 같은 드문 경우의 상황을 캡처합니다. 이 프레임워크는 단일 카메라 영상(monocular videos)에서 복잡한 4D(scene) 장면을 변환하여 기존 시뮬레이터와 호환되게 합니다. 로드워크 시뮬레이션에서 사용되는 이 데이터셋은 특히 폐쇄 루프(planners) 시나리오에서 활용될 수 있으며, 기존의 플래너와 비교하여 품질을 개선할 수 있는 가능성을 제공합니다.

- **Performance Highlights**: ROADWork4D-CL 데이터셋에서 검증된 부분에 대해 연구진은 전통적 규칙 기반(rule-based) 및 하이브리드(hybrid) 플래너가 특히 더 잘 일반화(generalize)하는 것을 발견했습니다. 그러나 이러한 플래너들조차도 임시 작업 구역에서 요구되는 차선 변경을 수행하는 데 실패하였습니다. Dash2Sim에서 회복된 밀집 깊이(dense depth)는 새로운 모습 합성(novel-view synthesis) 품질을 최대 19%까지 향상시켰으며, 이는 모노큘러 비디오에서 폐쇄 루프 센서 시뮬레이션을 위한 풍부한 조건 제공 가능성을 나타냅니다.



### Spatial-Temporal Decoupled Adapter for Micro-gesture Online Recognition (https://arxiv.org/abs/2606.07355)
Comments:
          Technical Report. 1st Place in Micro-gesture Online Recognition in 4th MiGA at IJCAI 2026

- **What's New**: 이 논문에서는 마이크로 제스처의 온라인 인식을 위한 새로운 접근 방식을 제안합니다. 기존의 방법은 공간적(spatial) 및 시간적(temporeal) 신호를 통합하여 처리하였으나, 이로 인해 미세한 패턴 잡아내기 어려운 한계가 있었습니다. 따라서, 저자는 공간적 및 시간적 모델링을 독립적으로 처리할 수 있는 Spatial-Temporal Decoupled Adapter를 도입하였습니다. 이 방법은 경량 깊이 방향 합성곱(depthwise convolutions) 기반으로, 각 방향이 독립적으로 기능하도록 합니다.

- **Technical Details**:  공간과 시간의 처리 경로를 분리하여 각 경로가 특정 기능에 집중할 수 있도록 하여, 마이크로 제스처 인식에서의 표현 학습을 강화합니다. 또한 Adaptive Soft Balanced Augmentation을 통해 데이터의 불균형 문제를 해결하며, 각 클래스의 샘플 수에 따라 동적으로 데이터 증강의 강도를 조정합니다. 이를 통해, 최적의 성능을 보장하며, 비균형 데이터 문제를 효과적으로 감소시키는 것을 목표로 합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 F1 점수 0.43808을 기록하며 EI-MiGA-IJCAI Challenge Track 2에서 1위를 차지했습니다. 이는 기존 방법에 비해 성능 향상에 크게 기여함을 보여줍니다. 공간과 시간의 독립적 처리를 통해 미세한 패턴을 더욱 효과적으로 인식할 수 있음을 입증한 것입니다.



### VeriDrive: Verifiable Counterfactual Supervision for Cost-Efficient Vision-Language Planning (https://arxiv.org/abs/2606.07338)
- **What's New**: VeriDrive는 계획 지향적이고 검증 가능한 카운터팩추얼(supervision) 감독을 구축하기 위한 프레임워크입니다. 기존의 자유 형식으로 만들어진 드라이빙 합리성 대신, VeriDrive는 미래의 동작에 기반해 주요 객체를 구체화하고, 교통 참여자와의 상호작용 증거를 평가하여 위험한 의도를 전문가의 행동으로 수정하는 구조화된 Perception-Evaluation-Revision 체인을 사용합니다. 이를 통해 드라이빙 모델의 의사결정 과정을 보다 해석할 수 있도록 개선합니다.

- **Technical Details**: VeriDrive 프레임워크는 다음 세 가지 단계의 카운터팩추얼(supervision) 추론을 명시적으로 감독합니다: (1) 주요 교통 참여자에 대한 미래 상호작용 증거의 구체화, (2) 규칙 기반의 위험 분석을 통한 자율 주행 경로 평가, (3) 전문가 행동으로 수정하기 위한 구조화된 의도 수준의 수정입니다. 이 프레임워크를 기반으로 구성된 VeriDrive 데이터셋은 심사 가능한 중간 감독을 포함하여 구조화된 주의 집중, 객체 중심의 미래 동작 증거, 규칙 기반 카운터팩추얼 평가 및 전문가의 수정을 포함합니다.

- **Performance Highlights**: VeriDrive는 nuScenes에서의 개방 루프(open-loop) 계획 테스트를 통해 Omni-Q 설정에서 성능을 평가했습니다. 결과적으로, VeriDrive 데이터셋과 감독 프로토콜이 충돌(Collision) 및 교차로(Intersection)에서의 성능을 향상시키면서 로그된 토큰 사용량과 생성 시간을 줄이고 예상되는 GPT API 비용을 절감하는 것으로 나타났습니다. 전체적으로 VeriDrive는 계획 중심의 자율 주행 모델의 성능을 효과적으로 개선합니다.



### Varifold Moment Invariants for Sustainable and Explainable Contour Feature Extraction (https://arxiv.org/abs/2606.07333)
Comments:
          29 pages, 12 figures

- **What's New**: 본 논문에서는 Varifold Moments Invariants (VMI)라는 새로운 틀을 소개하며, 이는 이전에 제안된 여러 Moment Invariants를 통합하는 역할을 합니다. 이러한 불변량(invariant)들은 Extended Gaussian Image, Elliptic Fourier Descriptors 또는 Shape Distributions와 같은 다른 경계 특징들과 밀접한 연관이 있습니다. VMI의 주요 장점은 지역의 기하학(geometry), 경계(boundary), 및 그에 접하는 선들(family of lines)과 결합하여 높은 분별력을 지닌 많은 불변 특징들을 생성할 수 있다는 점입니다.

- **Technical Details**: VMI는 지역의 기하학적 특징들을 활용하여 정보를 표현합니다. 이를 통해 Random Forest 및 Multi-Layer-Perceptron과 같은 경량(feature) 분류기와 결합하여 성능을 향상시킵니다. 이 방식은 외부 경계에 기반한 기존 접근방법들보다 경쟁력을 가지며, 계산 비용(computational cost)을 크게 줄여 경량 장치에서도 실행할 수 있게 합니다.

- **Performance Highlights**: 제안된 방법은 다양한 유형의 데이터셋(잎사귀, 객체, 세포)에 대해 테스트되었으며, 적은 수의 기하학적으로 해석 가능한 특징을 사용하여 높은 정확도를 달성했습니다. 이러한 알고리즘은 최신 기술에 비해 성능이 우수하며, 다양한 분류 작업에서도 효과적임을 입증했습니다.



### AnchorWorld: Embodied Egocentric World Simulation with View-based Evolution Customization (https://arxiv.org/abs/2606.07326)
- **What's New**: AnchorWorld는 상호작용이 가능한 세계 모델링의 중요한 경계를 탐색하는 프레임워크입니다. 이 프레임워크는 3D 인간 모션을 주요 상호작용 매개변수로 활용하며, 에고센트릭(egocentric) 뷰에서 시각적으로 잘리지 않는 신체 부위를 보완하기 위해 외부 관점으로부터의 보조적인 훈련 기법을 도입합니다. 이러한 접근을 통해 사용자는 환경과의 상호작용을 더욱 직관적으로 수행할 수 있습니다.

- **Technical Details**: AnchorWorld는 에고센트릭 액션 제어(embodied action control)와 위치 기반 앵커 뷰(pose-associated anchor views)를 통한 세계 상태 커스터마이징(localized world-state customization)이라는 두 가지 제어 방식을 제공합니다. 이를 통해 사용자는 3D 공간의 특정 위치에서의 상태를 명시하고, 카메라 뷰포인트에 따라 이 상태를 유지 및 진화시키는 것이 가능합니다. 이 프레임워크는 하이브리드 뷰(hybrid view)로 훈련하여 모델이 전체 신체 모션이 1인칭 시각적 관찰에 어떻게 영향을 미치는지를 배우도록 합니다.

- **Performance Highlights**: 실험 결과 AnchorWorld는 기존의 최신 기술에 비해 더 나은 성능을 보이며, 주목할 만한 공간적 인식(spatial awareness)과 장면 진화(scene evolution)를 제공했습니다. 특히, 사용자는 시야에 없는 장면의 진화(out-of-sight scene evolution)와 공간 변환에 따른 자세 일관성(pose consistency)을 유지할 수 있는 두 가지 주요 능력을 확인할 수 있습니다. 이 연구는 사람 모션 드리븐(human-motion-driven) 탐색과 상호작용이 가능한 세계 모델링의 잠재력을 크게 향상시킬 수 있음을 보여줍니다.



### CULTURESCORE: Evaluating Cultural Faithfulness in Video Generation Models (https://arxiv.org/abs/2606.07311)
- **What's New**: 본 연구는 비디오 생성 모델의 문화적 충실도를 평가하는 새로운 프레임워크인 CultureScore를 제안합니다. 이 프레임워크는 Identity, Behavior, Context의 세 가지 차원으로 구성되어 있습니다. 기존 메트릭에서는 단순히 시각적 품질만을 측정하는 데 반해, CultureScore는 문화적 대표성을 보다 세밀하게 분석할 수 있도록 합니다.

- **Technical Details**: CultureScore는 10개국의 2,943 개 프롬프트로 구성된 데이터셋을 바탕으로 6,180개의 비디오를 생성하여 평가를 수행합니다. 각 평가 항목은 Identity(누가 표현되는지), Behavior(문화적 제스처), Context(문화적 상황)로 나뉘며, 이러한 섬세한 구성 요소를 통해 모델이 어떻게 문화적으로 왜곡되는지를 진단합니다.

- **Performance Highlights**: 실험 결과, 현재의 어떤 비디오 생성 모델도 문화적으로 충실한 생성 결과를 달성하지 못했습니다. 최고 성능의 모델도 56.8%의 CultureScore를 기록했으며, Behavior 차원에서의 성과는 52% 미만으로, 문화적 표현에서의 부족함을 분명히 드러냈습니다. 또한, 문화적 충실도에 대한 인간 평가자의 선호도는 Visual Quality 메트릭과 반비례 관계를 보였습니다.



### ExMesh: EXplicit Mesh Reconstruction with Topology Adaptation (https://arxiv.org/abs/2606.07288)
Comments:
          Accepted at the IEEE/CVF Conference on Computer Vision and Pattern Recognition 2026 (CVPR 2026)

- **What's New**: 본 논문에서는 ExMesh라는 새로운 메쉬 재구성 프레임워크를 제안합니다. 이 프레임워크는 디프리셔블 최적화(differentiable optimization)와 이산(topology) 업데이트를 통합하여 메쉬를 직접 최적화합니다. 이를 통해 중간 표현을 사용하지 않고도 구조적으로 완전하며 편집 가능한 메쉬를 생성할 수 있습니다.

- **Technical Details**: ExMesh는 적응형(vertex splitting and merging) 전략을 도입하여 복잡한 지오메트리(geometry)에서 세밀한 세부정보를 복구하면서도 불필요한 면을 제거합니다. 메쉬 구조 업데이트는 각 최적화 단계에서 이루어지며, UV 좌표(real-time UV maintenance)를 실시간으로 유지함으로써 텍스처(detail)와 지오메트리의 분리를 가능하게 합니다. 이 시스템은 고유한 지오메트리 처리 문제를 해결하는 데 중점을 두고 있습니다.

- **Performance Highlights**: ExMesh는 높은 정확도(accuracy), 계산 효율성(computational efficiency), 그리고 메쉬 간결성(mesh conciseness) 사이의 균형을 잘 유지합니다. 실험 결과, ExMesh는 연속적이고 적응적인 형태의 최적화를 통해 효율적인 성능을 발휘하며, 이를 통해 기존 방법들이 갖고 있던 한계점을 극복합니다. 이러한 방향으로, ExMesh는 메쉬 재구성을 위한 새로운 기술적 패러다임을 제시합니다.



### Geometric-Aware Hypergraph Reasoning for Novel Class Discovery in Point Cloud Segmentation (https://arxiv.org/abs/2606.07280)
Comments:
          Accepted to the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026

- **What's New**: 이 논문은 새로운 클래스 발견(Novel Class Discovery, NCD) 프레임워크를 소개하여 기존의 쌍별(classwise pairwise) 연관 방식의 한계를 극복하고, 고차원(higher-order) 관계를 모델링하는 하이퍼그래프(hypergraph) 기반 프레임워크를 제안합니다. 특히, 논문에서는 3D 포인트 클라우드 데이터에서 알려진 클래스의 지식을 바탕으로 새로운 클래스를 세분화하는 방법을 설명합니다. 이를 통해 기하학적 특성과 의미적 연관성을 동시에 고려할 수 있는 'Geometric-Aware Prototypes'를 도입하여 더 정확한 분할(segmentation)을 실현합니다.

- **Technical Details**: 제안된 방법론인 Geometric-Aware Hypergraph Reasoning은 포인트 클라우드의 기하학적 정보를 활용하는 방법으로, 하이퍼그래프 구조를 통해 여러 클래스 간의 고차원 상호작용을 캡처합니다. 이 과정에서 각 포인트는 여러 프로토타입(prototype)과 연관되어 있으며, 지역적 및 전역적(spatial and structural) 정보를 동시에 추출하여 클래스 수준의 기하학적 단서를 강화합니다. 모델은 훈련 배치 간의 상호 프로토타입 연관에 따라 동적으로 하이퍼엣지(hyperedge)를 구성하여 클래스 불균형 문제를 완화합니다.

- **Performance Highlights**: SemanticKITTI와 SemanticPOSS 데이터셋에서 수행된 실험을 통해 제안된 방법의 효과성과 우수성이 입증되었습니다. 실험 결과, 하이퍼그래프 기반 접근 방식이 기존의 쌍별 클래스 연관 방법보다 더 정확한 세분화 결과를 보였습니다. 이를 통해 새로운 클래스 발견(NCD) 과제에서 구체적인 기하학적 및 의미적 맥락을 이해할 수 있는 향상된 방법론임을 입증했습니다.



### Reconstructing Multi-Decadal Forest Disturbances: A Spatio-Temporal Transformer Approach (https://arxiv.org/abs/2606.07249)
- **What's New**: 이번 연구에서는 1984년부터 2022년까지의 38년간 미국의 산림 피해를 동시적으로 모델링하는 딥러닝 프레임워크를 제안하고 있습니다. 기존의 위성 시계열 데이터 분석에서 픽셀 기반 분석의 한계를 극복하고, 시공간적 맥락을 고려하는 방법론을 개발하였습니다. 이는 노이즈 필터링과 공간적으로 일관된 피해 맵 생성을 통해 전반적인 감지를 향상시킵니다.

- **Technical Details**: 제안된 방법론에서는 Landsat, Sentinel-1 및 Sentinel-2의 다중 위성 데이터를 활용하여, 1985-2022년의 시리즈 데이터를 처리합니다. 또한, 다중 모달 템포럴 스페이셜 비전 트랜스포머(Multi-modal Temporal Spatial Vision Transformer; MTSViT) 모델을 사용하여 포괄적인 손해 감지를 실행합니다. 모델은 시공간적 관계를 모두 고려하여, 고해상도 이미지 데이터를 기반으로 훈련됩니다.

- **Performance Highlights**: 분석 결과, 제안된 모델은 다양한 산림 피해 데이터셋에서 높은 정확도(최대 98.2%)를 달성하며 공간적 왜곡을 줄이는 데 성공하였습니다. 그러나 각기 다른 피해 형태에 따라 성능의 절충이 발생하는 것을 확인하였으며, 이는 기존의 픽셀 기반 기준과 비교해 유의미한 발전을 보여주고 있습니다. 이 연구는 일관된 산림 모니터링의 기초를 제공하는 데 큰 기여를 하고 있습니다.



### Does Appearance Help? A Systematic Study of Image-Based Re-Identification in Online 3D Multi-Pedestrian Tracking (https://arxiv.org/abs/2606.07233)
Comments:
          Accepted for publication at the 35th IEEE International Conference on Robot and Human Interactive Communication (RO-MAN 2026)

- **What's New**: 이번 연구는 LiDAR 기반의 3D Multi-Object Tracking (MOT) 시스템에서 전통적인 기하학적 정보의 취약성을 보완하기 위해 이미지 기반 Re-Identification (ReID) 접근 방식을 활용하는 새로운 경량 프레임워크를 제시합니다. 이 체계적인 연구는 모바일 로봇의 성능과 실시간 반응성을 높이는 동시에, 상호작용을 지속적으로 유지할 수 있도록 모듈식 ReID 지점을 통합했습니다. 다양한 다중 모달 데이터 연관 전략을 평가하여 계산 지연과 강력한 추적의 균형을 맞추는 방법을 제안합니다.

- **Technical Details**: 연구는 PointPillars와 AB3DMOT를 기반으로 한 경량의 3D 감지기 및 추적기를 설정하였으며, 여러 기능 추출 구조와 다양한 다중 모달 연관 전략을 평가하여 최적의 LATENCY와 판별력을 찾았습니다. CNN 및 Vision Transformer 기반의 기능 추출기를 구성하여 실시간 3D MOT의 성과를 분석하였고, KITTI 데이터 세트에서 결과를 평가했습니다. 고전적인 모션 중심 접근 방법의 한계를 극복하기 위한 ReID 통합의 효과를 논의하며, occlusions에서의 추적 정확도를 보장하는 중요성을 강조합니다.

- **Performance Highlights**: 기존의 단순한 선형 퓨전 방식은 비주얼 노이즈로 인해 성능을 저하시켰지만, 계단식 매칭 전략은 occluded 트랙을 복구하는 데 성공했습니다. 이를 통해 전반적인 정확도를 유지하면서도 인간-로봇 상호작용의 연속성을 확보하는데 기여할 수 있음을 보여주었습니다. 이 연구 결과는 안전한 navigation을 위한 낮은 지연(latency)과 사회적 인식을 위한 판별력 사이의 최적의 거래를 제안하며, 경량 아키텍처를 이용하여 실시간 환경에서의 성능 향상을 확인하였습니다.



### DualGate-Net: A Prior-Gated Dual-Encoder Framework for Histopathology Cell Detection (https://arxiv.org/abs/2606.07222)
Comments:
          15 pages, 4 figures

- **What's New**: 이 논문에서는 DualGate-Net이라는 새로운 이중 인코더 프레임워크를 제안합니다. 이 프레임워크는 ConvNeXtV2 기반의 로컬 인코더와 SegFormer 기반의 글로벌 인코더를 결합하여 적응형 컨텍스트 통합을 통해 세포 탐지의 효율성을 향상시킵니다. 특히, 학습 가능한 prior-gated fusion 메커니즘이 포함되어 공간 위치별로 컨텍스트의 영향을 조절하며, 고주파 세포 구조를 보존하는 조정지향의 분기(branch)가 추가됩니다.

- **Technical Details**: 제안된 DualGate-Net은 인코더 아키텍처를 활용하여 미세구조와 대규모 컨텍스트 정보를 동시에 모델링합니다. ConvNeXtV2 인코더는 로컬 셀 구조를 정밀하게 감지하는 데 사용되며, SegFormer 인코더는 장기적인 컨텍스트 의존성을 모델링합니다. 또한, tissue segmentation을 위한 SegFormer-B2 모델을 훈련시켜 제공된 컨텍스트 정보를 기반으로 다채널 확률 맵을 생성하여 논문에서 설정한 학습 알고리즘의 기본으로 사용합니다.

- **Performance Highlights**: OCELOT 벤치마크 실험에서 DualGate-Net은 세포 탐지 성능이 꾸준히 향상되었음을 보여주었으며, 검증 세트에서 매크로 F1 점수 0.7722, 테스트 세트에서 0.7345의 기록을 달성했습니다. 이는 적응형 prior 통합이 robust한 세포 탐지에 효과적임을 입증하는 결과입니다. 언급된 강력한 성능 향상은 제안된 아키텍처의 전반적인 설계 및 샘플의 특징 추출 능력과 관련이 있습니다.



### AdaTok: Self-Budgeting Image Tokenization with Quality-Preserving Dynamic Tokens (https://arxiv.org/abs/2606.07185)
Comments:
          Preprint; 11 pages, 4 figures

- **What's New**: AdaTok는 입력에 따라 학습된 리소스 할당을 통해 자체 예산을 조정하는 이산 1D 토크나이저입니다. 이 방법은 각 이미지에 필요한 접두사가 무엇인지 배우면서, 동시에 다양한 예산에서 복원 가능한 접두사 스트림을 유지하는 것을 목표로 했습니다. 이러한 자기 예산 측정 방식은 기존 고정 예산의 비효율성을 개선하고, 토큰 수를 고정된 하이퍼파라미터로 지정하는 대신 콘텐츠에 따라 조정할 수 있도록 돕습니다.

- **Technical Details**: AdaTok는 두 가지 구성 요소인 우선순위 표현 학습(¡Prioritized Representation Learning, PRL과 적응형 토큰 할당(¡Adaptive Token Allocation, ATA)으로 구성됩니다. PRL은 중첩된 꼬리 마스킹과 다중 헤드 LoRA 헤드를 통해 토큰의 순서를 정리하여 예산에 맞는 복원을 제공합니다. ATA는 경량의 결정론적 그룹 상대 정책 최적화(GRPO)를 사용하여 주어진 예산 내에서 각 이미지에 대한 정책을 훈련합니다. 동적 파레토 가중칠(Dynamic Pareto Weighting, DPW)은 신뢰성과 길이의 균형을 자동으로 조정합니다.

- **Performance Highlights**: AdaTok는 ImageNet-1K에서 256개의 토큰으로 rFID 1.31을 달성하고, 평균 약 118개의 토큰 사용 시 AdaTok-Adaptive는 rFID 1.50을 기록했습니다. 이는 기존의 고정 토큰 수에 비해 효율적으로 성능을 개선한 것이며, 짧은 적응 표현을 통해 고정 256 토큰 디코드를 기준으로 약 2.1배 더 높은 처리량을 보여주었습니다. 보다 간단한 접두사를 사용하여 생성 품질을 유지하거나 개선할 수 있는 사례를 제시했습니다.



### OPTIMUS-Prime: Minimal and Sufficient Concept Explanations for Deep Vision Models (https://arxiv.org/abs/2606.07180)
- **What's New**: 이 논문은 eXplainable Artificial Intelligence (XAI)의 요구를 충족시키기 위해 OPTIMUS라는 새로운 프레임워크를 소개합니다. OPTIMUS는 기존의 explanation 방법들이 이론적 보장 없이 사용자에게 편리함만을 제공하던 문제를 해결하는 데 중점을 둡니다. 이 프레임워크는 주요 개념에 기반한 시각적 설명을 생성하여, 머신러닝 모델의 예측에 대한 형식적 보장을 제공합니다.

- **Technical Details**: OPTIMUS는 두 가지 주요 속성을 갖춘 시각적 heatmap을 생성합니다: sufficiency(충분성)과 minimality(최소성). 이 설명들은 특정 개념들이 분류기의 예측을 보장함을 명확하게 입증하며, 이러한 개념의 부분집합으로는 이 보장을 유지할 수 없습니다. 또한, 이 프레임워크는 개념을 활용하여 심층 분류 모델의 내부 정보를 해석 가능하게 만듭니다.

- **Performance Highlights**: OPTIMUS는 시각적 분류 벤치마크에서 검증되었으며, 예측의 이론적 근거를 충족하면서도 모델의 결정에 관련된 개념들을 자연스럽고 충실하게 드러냅니다. 실험 결과, OPTIMUS heatmap은 사용자에게 이해 가능하면서도 논리적으로 긴밀한 설명을 제공합니다. 이러한 성과는 기존의 saliency map 기법에 비해 확연히 향상된 것으로 평가됩니다.



### EvoGS: Constructing Continuous-Layered Gaussian Splatting with Evolution Tree for Scalable 3D Streaming (https://arxiv.org/abs/2606.07179)
Comments:
          Project page: this https URL

- **What's New**: EvoGS라는 새로운 프로세스를 소개합니다. 이는 기존의 3D Gaussian Splatting(3DGS) 시스템의 한계를 극복하기 위한 방법으로, 동적 네트워크에서 3D 콘텐츠의 효율적인 전송을 가능하게 합니다. EvoGS는 진화 트리(Evolution Tree) 구조를 통해 모든 splat이 부모-자식 관계를 갖고 있어 오류 수정과 데이터 압축을 용이하게 만들어 줍니다. 또한, EvoGS는 품질 전환을 원활하게 하며 실시간 스트리밍 요구 사항을 충족합니다.

- **Technical Details**: EvoGS는 연속적 레이어링(Continuous layering)이라는 새로운 개념을 도입하여 각 레벨의 splat들이 명시적인 부모-자식 관계로 연결됩니다. 이는 기존의 이산 레이어링(Discrete layering)의 한계를 해결하며, 각 계층의 splat들이 서로의 구조적 오류를 보완합니다. 특히, wavelet 기반의 정제 방법을 통해 상세한 표현을 생성하며, 이를 통해 더 낮은 스플랫 은닉 효율을 달성합니다. 최종적으로, 이 구조는 데이터 전송 및 GPU 메모리 소모를 줄이는데 기여합니다.

- **Performance Highlights**: EvoGS는 기존 방법들과 비교하여 성능에서 월등한 결과를 보여줍니다. 상세 실험을 통해 splat 중복 비율이 65%에서 25%로 감소하며, 데이터 전송량은 최대 2.5배 감소하고 GPU 요구 사항은 5.5배 경량화됩니다. 이 외에도 EvoGS는 원활한 품질 전환을 제공하고, 추가적으로 저장 공간을 8.7배 줄이는 성과를 달성했습니다.



### Seeing Without Exposing: Adaptive Privacy Control for Open-World, Context-Hungry MLLMs (https://arxiv.org/abs/2606.07175)
- **What's New**: 최근 발표된 연구에서는 Multimodal Large Language Models (MLLMs)와 관련된 새로운 개인 정보 보호 문제를 다룹니다. 특히, 사용자 제공 입력에서 민감한 정보가 포함될 수 있으며, 모델의 추론은 개인 데이터와 관련된 시각적 맥락에 의존합니다. 기존의 개인 정보 보호 방법은 고정된 카테고리와 전략에 의존하기 때문에 이러한 문제를 충분히 해결하지 못하고 있습니다. 이를 해결하기 위해 Anchored Privacy Drifting (APD)라는 새로운 방법을 제안합니다.

- **Technical Details**: APD는 훈련 없이 개인 정보가 포함된 요소를 의미적으로 동등한 대안으로 변환하면서, 원본 이미지에 대한 맥락적 단서를 고정시키는 방식으로 작동합니다. 이 방법은 공유된 다중 모달 잠재 공간(multimodal latent space)에서 작동하며, 두 개의 방향 신호를 통해 생성된 이미지의 경로를 조정하여 시각적 충실도를 보장합니다. 연구자는 AdaptShield라는 종합 벤치마크를 소개하며, 이 벤치마크는 22개의 개인 정보 카테고리를 포함하여 개인 정보 보호의 효과 및 콘텐츠 보존을 평가하는 기법을 제공합니다.

- **Performance Highlights**: 광범위한 실험 결과, 제안된 APD 방법은 텍스트 카테고리에서 평균 10.4%의 개선을, MLLM 기반 평가에서 평균 8.5%의 개선을 달성하여 개인 정보 보호와 콘텐츠 보존을 균형 있게 개선하는 데 성공하였습니다. 또한, AdaptShield 벤치마크를 통해 22개 카테고리 전반에서 APD의 성능이 최첨단 수치에 도달함을 보여줍니다. 이로 인해, MLLM 환경에서 다양한 개인 정보 보호 요구 사항에 적응 가능한 효과적인 프레임워크가 제공됩니다.



### Textual Supervision Enhances Geospatial Representations in Vision-Language Models (https://arxiv.org/abs/2606.07172)
Comments:
          Accepted at ICML 2026

- **What's New**: 이번 연구에서는 기계 학습 시스템의 지리적 이해를 분석하였습니다. 기존의 비전 전용 아키텍처, 비전-언어 모델, 그리고 대규모 다중 모달 기초 모델을 통해 지리적 표현을 평가하였으며, 텍스트 감독이 이러한 학습을 어떻게 향상시키는지 조사하였습니다. 연구 결과, 언어가 공간 맥락을 인코딩하는 유효한 보조 방식으로 작용하며, 다중 모달 학습이 지리적 인공지능 발전의 핵심 방향이 될 수 있음을 제시하였습니다.

- **Technical Details**: 비전 모델은 최근 10년 동안 컨볼루션 신경망(CNN)과 비전 트랜스포머(ViT)의 발전으로 큰 발전을 이루었습니다. 다양한 메타 정보로부터 지리적 정보를 내재화할 수 있는 가능성이 제기되었으며, 이러한 모델들은 프리트레인(pretrain) 및 파인튜닝(fine-tuning) 과정에서 지리적 지식을 어느 정도 내재화하는지를 분석하고 있습니다. 우리가 사용한 지리적 표현(geospatial representations)은 ViT와 VLM의 내부 레이어에 포함된 잠재적 피처를 지칭합니다.

- **Performance Highlights**: 본 연구는 비전 전용 및 비전-언어 모델이 지리적 정보를 암묵적으로 어떻게 인코딩하는지를 조사하였습니다. 다양한 모델의 성능을 레이어 별 탐색을 통해 비교하였으며, 비전 전용 모델은 마지막 레이어에서 더 강한 표현을 보였고, VLM은 언어 모델 블록의 초기 레이어에서 더 나은 지리적 표현을 나타냈습니다. 비전-언어 모델에 대한 프롬프트(prompts)를 통해 지리적 정보가 후속 레이어로 전파되는 것이 관찰되었으며, 이로 인해 표현 품질이 향상되는 경우도 발견되었습니다.



### When Recovery Matters: The Blind Spot of Surrogate Privacy in MLLM Editing (https://arxiv.org/abs/2606.07171)
- **What's New**: 이 논문에서는 멀티모달 대규모 언어 모델(Multimodal Large Language Models, MLLMs)을 활용한 이미지 편집에서 개인 정보 보호의 중요성에 강조를 두고, 새로운 SPPE(Surrogate-based Privacy-Preserving Editing) 벤치마크를 소개합니다. SPPE는 36개 세분화된 개인 정보 카테고리와 65개 다양한 편집 지침을 포함하며, 각각의 편집 작업에서 발생할 수 있는 개인 정보 노출을 감소시키기 위한 두 가지 주요 작업, 즉 편집 가능성 평가(editability assessment)와 원본 복원(surrogate-to-source recovery)을 정의합니다.

- **Technical Details**: SPPE에서는 ERMA(Editability-aware Relational Multi-modal Assessment)와 C2E-S2SER(Cycle-Consistent Edit-Conditioned Surrogate-to-Source Edit Recovery)와 같은 두 가지 방법론을 제안합니다. ERMA는 멀티모달 관계 모델링(multi-modal relational modeling)을 통해 주어진 지침에 대한 서퍼레이트의 편집 가능성을 예측하며, C2E-S2SER는 디퓨전 트랜스포머(diffusion transformer)를 활용하여 편집된 서퍼레이트에서 원본 이미지로 편집 효과를 식별하고 적용하는 기법입니다. 이 과정에서는 edit-conditioned 태그 생성을 통해 서로 다른 서퍼레이트에 대한 다양한 비주얼 효과를 조정합니다.

- **Performance Highlights**: 실험 결과, ERMA는 편집 가능성 평가에서 기존의 최첨단 기준을 13.9%와 12.3% 각각 SRCC와 PLCC에서 초과 달성하며, C2E-S2SER는 SPPE의 8가지 원본 무결성(source integrity) 및 편집 일관성(edit consistency) 메트릭스에서 모든 복원 기준을 초과 성과를 보였습니다. 이 논문은 서퍼레이트 기반 MLLM 개인 정보 보호 이미지 편집을 위한 회복 중심 벤치마크를 제시하여 실제 이미지 편집 작업에서 개인 정보 보호의 난제를 해결하기 위한 중요한 기준을 마련합니다.



### TraRA: Trajectory-level Recognition Aggregation for Video Text Spotting in Urban Surveillanc (https://arxiv.org/abs/2606.07161)
Comments:
          22nd IEEE International Conference on Advanced Visual and Signal-Based Systems

- **What's New**: 이 논문에서는 TraRA(Trajectory-level Recognition Aggregation for VTS)를 제안하여 비디오 텍스트 스포팅(Video Text Spotting)의 한계점을 극복하고자 합니다. 기존의 VTS 방법들이 프레임 별로 독립적으로 인식을 수행하면서 발생한 불일치성과 부정확성을 해결하기 위해, TraRA는 시간적 및 다중 모드 일관성을 활용하여 궤적 수준(text trajectory level)의 텍스트 인식을 수행합니다.

- **Technical Details**: TraRA는 두 가지 주요 모듈인 Temporal Clustering(TC)과 Vision-Language Aggregation(VLA)로 구성됩니다. TC 모듈은 비주얼과 시간적으로 일관된 텍스트 인스턴스를 그룹화하여 노이즈가 있는 궤적을 정제하며, VLA 모듈은 Low-Rank Adaptation(LoRA)을 활용하여 비주얼 신호와 언어적 맥락을 통합하여 안정적인 단어 예측을 생성합니다.

- **Performance Highlights**: TraRA는 RoadText, BOVText, ArTVideo, ICDAR15와 같은 네 가지 공개 벤치마크에서 광범위한 실험을 통해 입증되었습니다. 이 연구의 결과, TraRA는 기존의 최첨단 VTS 방법에 비해 추적 및 인식 성능을 일관되게 개선하였습니다.



### Consistent-Inversion: Reverse Consistency Guidance for Structure-Preserving Visual Editing (https://arxiv.org/abs/2606.07145)
Comments:
          Submitted to IEEE Transactions on Multimedia; 10 pages, 4 figures

- **What's New**: 이번 논문은 Consistent-Inversion이라는 새로운 툴을 소개하여, 이미지 편집에서 트레이너 없이도 구조를 보존하는 점을 강조합니다. 기존의 편집 방법들은 소스 이미지의 구조를 재사용하면서 발생하는 트레일 불일치 문제를 지적하고, 이로 인해 구조 세부정보가 손상될 수 있음을 설명합니다. Consistent-Inversion은 소스 이미지의 노이즈 경로와 목표 경로 간의 일관성을 검증하여 이러한 문제를 해결하는 방법을 제시합니다.

- **Technical Details**: Consistent-Inversion은 역 일관성 가이드를 통해 트레일 불일치 문제를 해결합니다. 이 방법은 중간 목표 경로가 소스 경로로 되돌려질 수 있는지를 점검하여, 소스 노이즈의 보조 표현을 사용하고 소스 유도 역 노이징을 수행합니다. 이 과정에서 얻는 역 일관성 불일치는 조기에 선택된 목표 노이징 단계에 보정 신호로 주입됩니다.

- **Performance Highlights**: PIE-Bench에서의 실험 결과에 따르면, Consistent-Inversion은 목표 프롬프트 정렬을 유지하면서도 배경 및 구조적인 충실도를 향상시키는 것으로 나타났습니다. 이 방법은 기존의 훈련이 필요 없는 편집 파이프라인과 호환되며, 추가적인 추론 비용이 적은 효율적인 방법임을 입증하였습니다.



### Native3D: End-to-End 3D Scene Generation via Unified Mesh-Texture Modeling and Semantic Alignmen (https://arxiv.org/abs/2606.07117)
- **What's New**: 이 논문에서는 2D 중간 표현을 완전히 우회하는 첫 번째 엔드 투 엔드 3D 장면 생성 프레임워크인 Native3D를 제안합니다. 기존의 접근 방식은 3D 표현을 2D 도메인으로 변환해야 하는데, 이는 기하학적 구조 왜곡과 텍스처 세부 사항 저하와 같은 도메인 적응 문제를 초래합니다. Native3D는 기하학적 구조와 텍스처 특성을 동시에 모델링하는 통합된 메쉬-텍스처 공동 표현을 설계하여 이러한 제한 사항을 해결합니다.

- **Technical Details**: Native3D는 Transformer 기반 장면 인코더를 통해 기하학적 구조와 텍스처 기능을 동시에 모델링합니다. 이를 통해 장면 내 객체 간의 공간 관계와 시각적 일관성을 효과적으로 유지합니다. 또한 3D 표현 정렬 손실(3D REPA Loss)을 도입하여 다중 수준의 의미 표현을 정렬함으로써 기하학적 및 텍스처 품질을 크게 향상시킵니다. 이 방법은 3D 메쉬 기하학 및 텍스처 정보를 직접 입력으로 받아 통합된 표현을 생성합니다.

- **Performance Highlights**: 실험 결과, Native3D는 생성 품질 및 편집 유연성 모두에서 기존 방법보다 우수한 성능을 보였습니다. 사용자들은 자연어를 조건 신호로 사용하여 복잡한 장면 생성 및 수정 요구를 정확하게 지정할 수 있습니다. 이러한 결과는 Native3D가 3D 장면 편집을 위한 혁신적인 솔루션을 제공한다고 강조합니다.



### 3DMorph: Single-Image-Guided Local 3D Shape Editing and Morphing (https://arxiv.org/abs/2606.07115)
Comments:
          Accepted to IJCNN 2026

- **What's New**: 최근 3D 생성에서의 성과에도 불구하고, 기존 형태의 직관적인 편집은 여전히 제한적입니다. 이미지의 경우 잘 확립된 inpainting 도구가 존재하지만, 일반 3D 객체인 메쉬(mesh)의 경우 간단하고 효과적인 지역 형태 편집 방법이 부족합니다. 본 논문에서는 훈련이 필요 없는 프레임워크인 3DMorph를 소개하며, 이는 단일 이미지의 안내를 통해 지역적인 3D 형태 편집과 변형(morphing)을 가능하게 합니다.

- **Technical Details**: 3DMorph는 수정된 이미지를 입력으로 받아 해당 3D 영역을 자동으로 식별하고, 2D 수정사항을 3D로 전이하는 메커니즘을 가지고 있습니다. 이를 통해, 기존의 메쉬에서 편집되지 않은 영역은 유지되며, 원하는 형태 수정을 쉽게 반영할 수 있습니다. 또한, 3DMorph는 원래 객체와 수정된 객체 사이의 중간 형태 생성을 지원하여 디자인 탐색을 촉진합니다.

- **Performance Highlights**: 실험 결과, 3DMorph는 직관적인 2D 편집을 3D로 변환하는 데 있어 최첨단 생성 및 편집 방법보다 뛰어난 성능을 보였습니다. 본 논문에서는 Delta3D라는 이미지 안내에 기반한 3D 편집 벤치마크를 도입하여 편집 품질을 평가하였습니다. 이는 구조적 수정에 대해 정량적인 비교를 가능하게 하는 데이터를 제공합니다.



### GP-Adapter: Gaussian Process CLIP-Adapter for Few-Shot Out-of-Distribution Detection (https://arxiv.org/abs/2606.07102)
Comments:
          8 pages, 6 figures, Accepted at IJCNN 2026

- **What's New**: GP-Adapter는 Gaussian Process(GP) 불확실성 모델링을 사용하여 CLIP(Contrastive Language-Image Pre-training)를 강화하는 훈련이 필요 없는 프레임워크입니다. 이 방법은 CLSP의 결정론적 유사성 점수에 의존하지 않고 데이터가 부족한 환경에서도 효율적으로 OOD(out-of-distribution) 샘플을 탐지할 수 있도록 도와줍니다. GP-Adapter는 K-shot 레이블 캐시를 이용하여 모달리티별, 클래스별 일급 GP를 구성하고, 이를 통해 OOD 탐지를 위한 신뢰 점수를 생성합니다.

- **Technical Details**: GP-Adapter는 CLIP로부터 추출한 이미지 및 텍스트 임베딩 위에 한 클래스 GP를 구성하여 GP 기반 불확실성 모델링을 제공합니다. 이 방식은 서로 다른 모달리티에 대해 독립적으로 작동하며, 예측 통계치를 결합하여 분산 인식을 위한 신뢰 점수를 생성합니다. GP의 모든 매개변수는 빠른 조정에 필요한 경량 그리드 서치를 통해 최적화되며, 메모리 비용은 $O(C K^2)$로 저비용을 유지합니다.

- **Performance Highlights**: GP-Adapter는 ImageNet과 여러 OOD 벤치마크에서 실험을 통해 경쟁력 있는 몇 샷 성능을 제공합니다. 특히, 정확도 손실 없이 prompt-learning 방법과 결합할 경우 OOD 탐지 성능이 지속적으로 향상되는 것을 보여줍니다. 이는 GP 기반 불확실성 모델링과 프롬프트 학습 간의 상호 보완성을 강조합니다.



### LARA: Latent Action Representation Alignment for Vision-Language-Action Models (https://arxiv.org/abs/2606.07100)
- **What's New**: 이번 연구에서는 Latent Action Representation Alignment (LARA)라는 새로운 프레임워크를 제안합니다. LARA는 Latent Action Models (LAM)과 Vision-Language-Action (VLA) 모델을 함께 최적화하여 서로에게 이익이 되는 방향으로 학습을 진행할 수 있게 합니다. 이를 통해 VLA 모델의 성능을 저하하는 비효율적인 행동 경로를 줄이고, VLA는 LAM의 정방향 동작 예측에 의해 규제됩니다.

- **Technical Details**: LARA는 LAM과 VLA의 레이턴트(잠재적) 행동 표현을 정렬하는 경량화된 메커니즘을 통해 연결됩니다. LAM은 행동 경로와 함께 공동 학습을 통해 시각적인 변화를 줄이는 반면, VLA는 LAM에서 학습한 동역학(dynamics)을 사용하여 잘못된 행동 경로의 환상을 줄입니다. 이는 Diffusion 모델의 레이턴트 표현 정렬 관련 최근 연구에 영감을 받아 설계되었습니다.

- **Performance Highlights**: 실험 결과 LARA는 시뮬레이션 및 실제 로봇 제어 환경에서 각각 평균 10%, 5%, 15% 개선 효과를 보였습니다. LARA는 기본 VLA 모델의 학습을 강화하는 중요한 툴로 작용하며, 사전 훈련된 VLA 모델을 후속 훈련 단계에서 개선하고, LAM의 레이턴트 행동 표현을 정제하는 데 효과적입니다. 이러한 성과는 VLA 학습의 표준 훈련 파이프라인에서 혁신적인 전환점을 제공합니다.



### Detecting Temporally Localized Manipulations in Authentic Video Streams (https://arxiv.org/abs/2606.07090)
- **What's New**: 이 연구는 현실적인 비디오 조작 탐지를 위한 새로운 데이터셋의 필요성을 강조하며, 현재의 데이터셋들이 짧은 조작된 구간이 포함된 진짜 비디오 시나리오를 효과적으로 모델링하지 못하고 있음을 보여줍니다. 이 데이터셋은 높은 수준의 현실감을 요구하는 조작 세그먼트를 포함해야 하며, 그에 따른 탐지 문제를 다룹니다.

- **Technical Details**: 연구에서는 DINOv3 기능을 활용한 두 가지 보완적인 접근 방식을 제안합니다. 첫 번째 접근 방식은 세 가지 임계값 전략 하에 선형 탐사를 위한 방법을 사용하고, 두 번째는 DINOv3 기능을 기반으로 연속적인 프레임 유사성 방법을 통해 시간적 조작 경계를 탐지합니다. 이 두 가지 실험은 현실적인 조작 탐지의 초기 기준점(benchmark)을 설정합니다.

- **Performance Highlights**: 제안된 방법은 DINOv3 기능을 활용하여 83%의 정밀도와 95%의 비디오 수준 정확도를 기록하였으며, 이는 훈련 없이도 가능하였습니다. 이러한 성능은 향후 연구를 위한 기초적인 기준을 제공하며, 컨텐츠 적응형 임계값 메커니즘의 필요성을 강조합니다.



### An Adaptive Data cleaning Framework for Noisy Label Detection (https://arxiv.org/abs/2606.07086)
- **What's New**: 이번 논문에서는 과도하게 매개변수화된 딥 뉴럴 네트워크(DNN)가 현실에서 흔히 발생하는 소음이 많은 레이블 문제를 해결하기 위한 자가 적응 데이터 클리닝(data-cleaning) 프레임워크를 제안합니다. 기존의 샘플 선택(sample-selection) 방식은 수동으로 설정된 임계값(thresholds)이나 노이즈 비율(noise ratio) 등 특정 조건에 의존하여 불안정했던 반면, 본 연구는 이를 개선하는 방법론을 제공합니다.

- **Technical Details**: 제안하는 프레임워크는 로컬(local), 글로벌(global), 그리고 학습 동역학(learning dynamics) 신호를 통합하여 노이즈 레이블 탐지를 시행합니다. 샘플은 모듈형 피쳐 결합(paradigm) 방식을 통해 통합된 저차원(feature space) 특성 공간으로 매핑되며, 이를 통해 클래스 적응 KNN 기반의 로컬 불일치(local disagreement)와 k-평균(global centroid distance) 기반 글로벌 거리(global distance)를 통합하는 2D 측정치와 z-정규화(z-normalized) 점수를 포함한 3D 멀티-메트릭(multi-metric) 두 가지 구현을 제공합니다.

- **Performance Highlights**: CIFAR-10, MNIST, ImageNet-100 데이터셋에 대해 5%에서 40%의 대칭 레이블 노이즈가 포함된 실험 결과, 설정에 따라 높은 리콜(recall) 성능을 확인했습니다. 특히 ImageNet-100에서 40% 노이즈 환경에서도 98% 이상의 거의 완벽한 리콜을 달성하였으며, 이후의 훈련 결과에서도 정확도가 개선되는 것을 확인했습니다. 이러한 결과는 멀티-메트릭 통합이 스레쉬홀드가 필요 없는 실용적인 노이즈 레이블 탐지 접근 방안이 될 수 있음을 시사합니다.



### AsyncPatch Diffusion: spatially-flexible image generation (https://arxiv.org/abs/2606.07079)
Comments:
          36 pages, 14 figures

- **What's New**: 이번 논문에서는 전통적인 확산 모델의 한계를 넘어서기 위해 AsyncPatch Diffusion이라는 새로운 기법을 소개합니다. 기존의 확산 모델들은 모든 공간 영역에 동일한 노이즈 레벨을 적용하지만, AsyncPatch는 서로 다른 입력 차원에 개별적인 노이즈 레벨을 할당하여 더 다양하고 풍부한 노이즈 제거 경로를 제공합니다. 이 새로운 접근법은 생성 과정을 유효하게 정의하면서도, 더 많은 변형 가능성을 제공합니다.

- **Technical Details**: AsyncPatch Diffusion은 데이터 차원이 서로 다른 독립적인 노이즈 수준을 가질 수 있도록 일반화된 확산 프레임워크를 가지고 있습니다. 이는 노이즈가 자유롭게 적용될 수 있는 환경을 제공하며, 특히 훈련 과정에서는 controlled timestep sampler를 사용하여 입력의 다양한 노이즈 레벨을 조정합니다. 이 과정에서 각 차원에 대해 개별적인 함수 매개변수를 활용하여 더욱 유연한 이미지 생성을 가능하게 합니다.

- **Performance Highlights**: AsyncPatch 모델은 ImageNet 256 및 LSUN 벤치마크에서 전통적인 확산 모델과 유사한 생성 품질을 기록하며, 전용 파인 튜닝 없이도 인페인팅에 적합하게 설계되었습니다. 논문에서는 또한 불확실성 기반 가속 및 오토회귀 샘플링과 같은 적응형 생성 전략을 다루고 있으며, 이를 통해 지역 일관성 및 텍스처 일치를 개선하는 방향으로 더 나아가고 있습니다.



### TrioPose: Native Triple-Stream Diffusion Transformers for Pose-Guided Text-to-Image Generation (https://arxiv.org/abs/2606.07053)
Comments:
          15 pages (9 pages main body, 6 pages references and appendix), 3 figures, 5 tables

- **What's New**: TrioPose는 복잡한 멀티인스턴스 시나리오에서 발생하는 사지 왜곡과 피쳐 간섭 문제를 해결하기 위해 설계된 새로운 프레임워크입니다. 이 연구는 SD3.5M 아키텍처에 기반하여, 포즈를 독립적인 모달리티로 처리하는 Triple-Stream Pose-Aware DiT (TSPA-DiT)를 도입하였습니다. 이를 통해 선행 훈련된 잠재 분포를 유지하면서도, 기하학적 제약을 매끄럽게 적용할 수 있는 새로운 방법론을 제안합니다.

- **Technical Details**: TSPA-DiT는 레이어별 활성화(layer-wise activation) 및 제로 초기화 제이중 잔여 주입(zero-initialized dual-residual injection) 방식을 사용하여, 원활한 조건부 생성을 가능하게 합니다. 또한, Learnable Relational Bias Mask를 설계하여 물리적 상태를 세분화하고 이를 연속적인 주의 편향으로 매핑합니다. 이 방법은 멀티 인스턴스 상호작용에서의 간섭을 완화하며, 자세 안내를 활용한 공간 손실 가중치를 통해 해부학적 감독을 특정 왜곡 영역에 집중시킵니다.

- **Performance Highlights**: TrioPose는 Human-Art, CrowdPose 및 OCHuman과 같은 도전적인 벤치마크에서 최첨단 성능을 기록하였습니다. Human-Art 데이터셋에서 64.33의 AP를 달성하며, 이전 작품보다 30% 개선된 성능을 보입니다. 이로 인해 복잡한 멀티 휴먼 생성에서의 시각적 진실성과 텍스트-이미지 의미 정렬에서 새로운 기준을 세웠습니다.



### STREAM: Stochastic Riemannian Flow Matching with Anisotropic Decoder for Digital Histopathology Image Generation (https://arxiv.org/abs/2606.07036)
Comments:
          27 pages, 7 figures

- **What's New**: 이 논문은 합성 병리학적 이미지 생성의 새로운 접근 방식인 STREAM을 소개합니다. 이 방법은 Latent Diffusion Model을 활용하여 생긴 기존의 'Conditioning Collapse' 문제를 해결하기 위해, 사전 훈련된 병리학 VFM의 패치-토큰 특징을 잠재 공간으로 활용합니다. 이는 생성된 이미지를 더욱 다양화하고 질을 향상시킵니다.

- **Technical Details**: STREAM은 두 단계로 구성됩니다: 첫 번째 단계는 $	ext{SLERP}$ 지오데식에서 발생하는 브리지 타입의 임의 교란을 설정하여 잠재 공간에서 Diffusion Transformer를 훈련시키는 것입니다. 두 번째 단계에서는 높은 에너지 방향을 보존하면서 저에너지 방향에서의 강건성을 할당하는 새로운 비등방형 디코더를 적용합니다. 이러한 설계를 통해 생성 과정에서의 품질이 향상됩니다.

- **Performance Highlights**: STREAM은 유방 암 진단 및 대장암 데이터셋에서 최첨단 재구성 및 생성 성능을 달성하여 기존 모델들을 능가합니다. 이 모델은 임상 데이터의 개인 정보 보호와 대규모 훈련 데이터의 필요성을 충족시키는데 기여할 것입니다. 코드 또한 논문 수락 시 공개될 예정입니다.



### ForensicConcept: Transferable Forensic Concepts for AIGI Detection (https://arxiv.org/abs/2606.07034)
Comments:
          Accepted by ICML 2026

- **What's New**: 이번 논문에서는 AI 생성 이미지 탐지를 위한 새로운 접근법인 ForensicConcept를 제안합니다. 기존의 탐지기들이 블랙박스인 문제에서 벗어나, 결정적인 증거를 명시적으로 추출하고 다른 백본(backbone)으로 전이하는 방법을 논의합니다. 이 방법은 Transformer 기법을 통해 중요한 패치를 국소화하고, 이를 밀집된 개념 코드북으로 클러스터링하여 감사 가능한 증거를 제공합니다.

- **Technical Details**: ForensicConcept 프레임워크는 네 가지 주요 요소로 구성됩니다: 포렌식 개념 유도, 외부 생성 흔적 참조 도입, 그리고 개념 코드북 주입입니다. 결정적으로, neighborhood-structure consistency (CKNNA)를 통해 탐지기 증거와 생성 흔적 간의 정량적 일치를 평가하며, 이는 다양한 생성기와의 전이 가능성을 설명합니다.

- **Performance Highlights**: 제안된 방법은 GenImage, GAN-family, Chameleon 벤치마크에서 일관된 성능 향상을 보여줍니다. GenImage에서 92.0%의 정확도를 기록하였고, GAN-family에서 90.1%, Chameleon에서 84.4%의 정확도를 달성했습니다. CKNNA 정렬이 높은 백본은 더 많이 전이 가능한 개념을 제공한다는 사실이 확인되었습니다.



### Never Seen Before: Benchmarking Genuine Zero-Shot Composed Image Retrieval with Consistent Video-Sourced Datasets (https://arxiv.org/abs/2606.07032)
- **What's New**: 이번 연구는 Zero-Shot Composed Image Retrieval (ZS-CIR) 분야에서 새로운 벤치마크인 ZeroSight를 제안합니다. 기존 ZS-CIR 데이터셋에서는 참조 이미지와 목표 이미지 간의 완전한 불일치가 문제였으며, 이를 해결하기 위해 영상에서 수집한 일관된 참조-목표 쌍을 포함하는 데이터셋을 구축했습니다. 연구는 또한 CLIP이 미리 훈련된 데이터가 포함되지 않도록 최근 영상 데이터를 사용하여 진정한 제로샷 상황을 보장합니다.

- **Technical Details**: ZeroSight는 12,048개의 다양한 비디오에서 197,313개의 후보 이미지와 54,740개의 쿼리를 바탕으로 구성된 CIR 데이터셋과 데이터 구축 파이프라인을 포함하고 있습니다. 연구팀은 고급 LLM 기법을 활용하여 시각적 및 의미적으로 일관된 이미지 쌍을 생성하고, SC4CIR(대칭 일관성을 위한 CIR)이라는 학습 필요 없는 방법을 통해 하드 네거티브 타겟도 효과적으로 식별합니다. 이 방법은 다양한 CIR 방법과 원활하게 통합되어 성능을 크게 향상시키는 것이 특징입니다.

- **Performance Highlights**: ZeroSight에서 수행한 실험 결과에 따르면, 기존 CLIP 기반 CIR 방법들은 기존 ZS-CIR 데이터셋에서 성능이 부풀려진 결과를 나타내며, CIR 방법의 능력을 과장하게 됩니다. 연구는 해당 벤치마크를 통해 다수의 긍정 및 부정 타겟 이미지의 순위를 고려한 평가 방식을 제시하여 현재 CIR 방법들의 역량을 정밀하게 평가할 수 있도록 합니다. 이 연구는 AI와 이미지 검색 기술의 발전에 기여할 것으로 기대됩니다.



### GuideCAD: A Lightweight Multimodal Framework for 3D CAD Model Generation via Prefix Embedding (https://arxiv.org/abs/2606.07024)
- **What's New**: GuideCAD는 소수의 학습 가능한 매개변수를 활용하여 3D CAD 모델을 생성하는 다중 모달 접근 방식을 제안합니다. 이 프레임워크는 사전 훈련된 대형 언어 모델을 사용하여 시각적 및 텍스트 정보를 통합하는 매핑 네트워크를 포함하고 있습니다. 실험 결과, GuideCAD는 기존의 도구들보다 더 적은 매개변수로 높은 품질의 3D CAD 모델을 생성할 수 있음을 보여줍니다.

- **Technical Details**: GuideCAD는 이미지 임베딩을 프리픽스 임베딩으로 변환하는 매핑 네트워크를 사용하여, 키워드 임베딩이 포함된 시각적-텍스트적 표현을 통합합니다. 이 과정을 통해 트랜스포머 기반의 디코더가 건설 시퀀스를 예측하여 3D CAD 모델을 생성하게 됩니다. 전체 사전 훈련된 인코더의 가중치를 동결하고, 소수의 학습 가능한 매개변수만 추가하여 효율적인 훈련을 달성합니다.

- **Performance Highlights**: GuideCAD는 최신 비전-언어 모델(VLM)과 비교할 때 학습 가능한 매개변수 수를 4배 이상 줄이면서도 비슷한 수준의 품질의 3D CAD 모델을 생성합니다. 또한, 총 교육 시간이 절반 이상 단축되며, 생성된 모델은 실제 모델의 지표와 근접한 높은 품질을 자랑합니다. 이러한 효과적인 성능을 바탕으로 GuideCAD는 공개된 다중 모달 데이터셋을 통해 3D CAD 모델 구축의 새로운 기준을 제시합니다.



### Don't Pause: Streaming Video-Language Synchrony for Online Video Understanding (https://arxiv.org/abs/2606.06991)
- **What's New**: 이번 논문에서 저자들은 새로운 온라인 비디오 이해 패러다임인 Streaming Video-Language Synchrony(SVLS)를 제안하며, 이를 바탕으로 LyraV라는 라이브 스트리밍 보조 도구를 개발하였습니다. LyraV는 두 가지 주요 혁신을 통해 작동합니다: Frame-Driven Transition Controller(FDTC)와 Streaming Token Pacer(SToP)입니다. 이 시스템은 영상 프레임과 언어 생성의 동기화를 향상시키며 실제 시간 처리 속도에서 높은 성능을 보여줍니다.

- **Technical Details**: LyraV는 Frame-Driven Transition Controller(FDTC)와 Streaming Token Pacer(SToP)라는 두 가지 구성 요소로 구성됩니다. FDTC는 고차원적으로 상황을 판단하여 언제 말을 계속할지, 새로운 응답을 시작할지 또는 침묵할지를 결정합니다. SToP는 실시간 지연 제약 내에서 프레임당 발화율을 동적으로 조정하여 비주얼 콘텐츠의 속도에 맞춰 언어 생성 속도를 최적화합니다.

- **Performance Highlights**: LyraV는 5개의 온라인 및 3개의 오프라인 벤치마크를 통해 검증되었습니다. LyraV는 비디오 재생과 98.29%의 동기화를 유지하며, 3.89 FPS의 실시간 처리 속도를 제공합니다. 이 시스템은 또한 비디오가 진행되는 동안 지속적으로 해석하고 '생각'할 수 있는 능력을 보여줍니다.



### CL-CLIP: CLIP-Based Continual Learning Framework with Cost-Volume Category Decoupling for Object Detection (https://arxiv.org/abs/2606.06978)
- **What's New**: CL-CLIP는 지속적인 객체 탐지(Continual Object Detection, COD)를 위한 혁신적인 프레임워크로, CLIP 기반의 개방된 어휘 탐지기(open-vocabulary detector)에 지속 학습 기능을 추가합니다. 이 프레임워크는 비용-양(volume-guided) 카테고리 분리를 통해 새로운 카테고리를 학습하면서도 이전에 학습한 카테고리를 유지할 수 있도록 설계되었습니다. CL-CLIP은 이전의 탐지 기능을 손상시키지 않으면서도 새로운 카테고리에 적응할 수 있는 특별한 구조적 원리를 활용합니다.

- **Technical Details**: CL-CLIP은 CAT-Seg에서 영감을 받고, 이미지-텍스트 유사성 비용 양을 사용하여 각 클래스에 대한 공간적 응답을 생성합니다. 이 구조는 이전에 학습한 클래스에 대한 공간적 우선 순위를 제공하며, 이를 통해 클래스 특화된 탐지 경로를 생성하고 이전 클래스와 새로운 클래스 간의 직접적인 경쟁을 줄입니다. 또한, 다중 전문가 RoI 헤드를 사용하여 각각의 카테고리에 대해 특화된 전문가가 할당되며, 이를 통해 메모리의 흐름을 안정화하면서 이전 클래스를 보호합니다.

- **Performance Highlights**: PASCAL VOC 및 MS-COCO 데이터셋에서의 광범위한 실험 결과, CL-CLIP은 지속적인 세부 조정(deep tuning) 하에서 F-ViT의 성능을 상당히 향상시켰습니다. 특히, CL-CLIP은 새로운 카테고리에 적응하면서도 기존의 기본 클래스 성능을 유지하는 데 있어 매우 경쟁력 있는 결과를 보여주었습니다. 이전의 기존 지속적 객체 탐지기들과 비교했을 때, CL-CLIP는 특히 새로운 카테고리에 대한 적응력에서 두드러진 성과를 나타냈습니다.



### From Vision to Text: A Compact Multimodal Approach for Robust, Cross-Domain Presentation Attack Detection on ID Cards (https://arxiv.org/abs/2606.06966)
Comments:
          Publication under the revision process on IEEE

- **What's New**: 이 논문은 Presentation Attack Detection (PAD)에서의 cross-domain shifts 문제를 해결하기 위한 새로운 접근 방식을 제안합니다. 연구진은 진짜 및 합성 ID 이미지에 대해 시각적 및 텍스트 데이터를 결합한 compact multimodal 모델을 개발했습니다. 기존의 합성 데이터셋은 실제 상황을 잘 반영하지 못하므로, 더 현실적이고 다양한 데이터셋의 필요성을 강조합니다.

- **Technical Details**: 본 연구는 세 가지 모델 평가 프레임워크를 도입하여 PAD 성능을 비교합니다. 여기에는 전통적인 Deep Learning 모델, Unimodal 모델, 그리고 Compact Multimodal 모델이 포함됩니다. 특히, DenseNet-121 아키텍처를 기반으로 하여 기존의 convolutional 구조가 ID 카드 PAD에 얼마나 효율적으로 적용될 수 있는지를 평가합니다.

- **Performance Highlights**: 이 논문은 다섯 개 국가의 ID 카드 데이터셋을 이용하여 extensive cross-dataset 평가를 수행했습니다. 각 모델의 성능은 일반화 능력이 어떻게 다른지를 분석하며, 서로 다른 데이터 카테고리에서 PAD 정확성을 높이기 위한 결과를 제시합니다. 또한, GitHub에서 코드가 제공될 예정으로, 연구 재현성을 높이기 위한 노력을 기울이고 있습니다.



### MVSegNet: A Lightweight Boundary-Aware Network for Fetal Lateral Ventricle Segmentation and Atrial Width Estimation in Prenatal Ultrasound (https://arxiv.org/abs/2606.06958)
Comments:
          11 pages, 3 figures, 4 tables. Code and trained models will be released upon acceptance. Supplementary material available upon request

- **What's New**: MVSegNet은 태아 측뇌실 세그멘테이션을 위한 경량의 인코더-디코더 네트워크로, 다중 스케일 특징 추출과 경계 인식 정제를 결합하여 개발되었습니다. 기존의 수작업 측정 방식의 변동성을 줄이고 세그멘테이션 정확성을 높이기 위해 설계되었습니다. 이 모델은 584개의 전문가 주석이 달린 초음파 이미지를 비교 평가하여 최상의 성능을 보였습니다.

- **Technical Details**: MVSegNet의 아키텍처는 MobileNetV3-Small 기반으로 구성되어 있으며, 효율적인 깊이별 분리 합성곱(depthwise separable convolution) 설계를 활용하여 저비용으로 운영됩니다. 이 모델은 멀티 스케일 심장 특징 모듈(Multi-Scale Ventricle Attention Module)과 주목(attention) 기반의 스킵 연결을 통해 구획 인식 성능을 향상시킵니다. 처리 속도 및 경계 품질을 개선하기 위해 적응형 경계 정제 모듈(Adaptive Boundary Refinement Block)을 포함하고 있습니다.

- **Performance Highlights**: MVSegNet은 80.79%의 Dice 점수, 68.47%의 IoU, 4.07mm의 하우스도르프 거리(Hausdorff distance), 3.40mm의 평균 절대 오차(mean absolute error)를 기록하며, 임계치 측정에서 뛰어난 성능을 보였습니다. 뿐만 아니라, NVIDIA T4 GPU에서 초당 165.6 프레임(fps)으로 실행되며, 모든 평가된 기준선 모델보다 뛰어난 성능을 나타냈습니다. 이는 MVSegNet이 자동화된 태아 초음파 분석에 적합함을 시사합니다.



### When is 3D Worth It? A Resource-Performance Frontier for CNNs and Transformers in Lung C (https://arxiv.org/abs/2606.06950)
Comments:
          8 pages, 6 figures

- **What's New**: 이 연구에서는 3D 모델이 일반적으로 선호되지만, 실질적인 성능 향상 여부가 계산 비용과 복잡성을 정당화하는지를 분석합니다. 2D, 2.5D, 3D의 입력 차원을 바탕으로 CNN 및 Vision Transformers의 모델 동작을 비교하며, 2.5D CNN이 가장 우수한 성능-안정성 균형을 제공한다고 보고합니다. 특히, 3D CNN은 임계값 불안정성이 나타났고, transformers는 모든 양성 예측으로 나타나는 한계를 보였습니다.

- **Technical Details**: NLST 코호트(n=1,977) 및 LIDC-IDRI 데이터를 사용한 연구로, 2D, 2.5D, 3D 입력을 각각 중앙 축 단면, 세 개의 직교 슬라이스, 그리고 폐 중심 서브 볼륨으로 정의합니다. 두 가지 네트워크 유형인 residual CNN과 Vision Transformer를 이 입력에 맞게 조정하여 동일한 학습 프로토콜(20 에포크; 가중 이진 크로스 엔트로피)을 통해 훈련하였습니다. 평가 기준으로 ROC-AUC, PR-AUC 및 민감도/특이도를 사용하였으며, 부트스트랩 신뢰구간 또한 보고합니다.

- **Performance Highlights**: 연구 결과, 2.5D CNN이 0.682의 ROC-AUC 및 0.158의 PR-AUC를 기록하며 가장 높은 성능을 보여주었습니다. 2D CNN은 지나치게 보수적이었고, 3D CNN은 임계값에 따라 불안정성을 보였습니다. transformers는 높은 GPU 메모리 요구 및 예측의 수렴 부족으로 인해 제대로된 성능을 내지 못한 것으로 나타났습니다.



### SS-TPT: Stability and Suitability-Guided Test-Time Prompt Tuning for Adversarially Robust Vision-Language Models (https://arxiv.org/abs/2606.06943)
Comments:
          Accepted in ICML2026

- **What's New**: 본 논문에서는 Vision-language models (VLMs)인 CLIP가 강력한 zero-shot 인식을 달성하였지만, 적대적 변형(adversarial perturbations) 하에서 쉽게 손상되는 문제를 다룹니다. 이를 해결하기 위해 Stability and Suitability-guided Test-time Prompt Tuning (SS-TPT)라는 새로운 방법을 제안합니다. SS-TPT는 각 증강(view)된 이미지의 품질을 평가하기 위해 두 가지 상호 보완적인 점수인 안정성(stability)과 적합성(suitability)을 활용합니다.

- **Technical Details**: 안정성은 약한 변형에 대한 예측 불변성을 측정하고, 적합성은 여러 뷰(view) 간의 특징 공간 밀도를 평가합니다. 이 두 점수는 SS-guided consistency loss와 SS-weighted prediction을 통해 적응(adaptation) 및 추론(inference)을 안내합니다. 이를 통해 신뢰할 수 있는 뷰를 강화하고 손상된 뷰를 억제하는 방식으로 시스템의 견고함을 증가시킵니다.

- **Performance Highlights**: SS-TPT는 방대한 실험을 통해 이전의 최첨단(state-of-the-art) 방법들보다 월등한 성능을 보여주며, 다양한 데이터셋과 뷰 수에서 뛰어난 견고성-처리량(robustness-throughput) 거래 균형을 달성했습니다. 이는 SS-TPT의 실용성과 일반성을 동시에 입증하는 결과입니다. 코드는 제공된 URL에서 확인할 수 있습니다.



### When CLIP Sees More, It Fights Back Harder: Multi-View Guided Adaptive Counterattacks for Test-Time Adversarial Robustness (https://arxiv.org/abs/2606.06938)
Comments:
          Accepted in CVPR2026

- **What's New**: 최근 CLIP와 같은 비전-언어 모델들은 놀라운 제로샷 인식(zero-shot recognition) 능력을 보여주었으나, 적대적 공격(adversarial perturbations)에는 취약한 모습을 보였습니다. 이를 해결하기 위해 제안된 시험시간 반격(Test-time counterattack, TTC) 방법이 있으나, 강한 공격에 대해서는 무너지는 경향이 있습니다. 본 논문에서는 다중 시점을 활용한 적응형 반격(Multi-view guided Adaptive Counterattack, MAC) 방법을 소개하여 이러한 약점을 개선하고자 했습니다.

- **Technical Details**: MAC는 먼저 입력 이미지의 다양한 증강 뷰(augmented views)를 만들어 다양한 임베딩(embeddings)을 확보합니다. 그 후, 각 뷰의 손상된 임베딩을 정제하기 위해 반격을 수행합니다. MAC는 각 뷰의 손상 정도를 추정하여 필요에 따라 반격의 강도를 조정하며, 이러한 적응형 반격된 뷰들을 집계하여 최종 예측 값을 도출합니다.

- **Performance Highlights**: 20개 데이터셋과 다양한 공격 시나리오를 통해 수행된 광범위한 실험에서 MAC는 강력한 내구성을 보여주었으며, 높은 추론 속도와 메모리 효율성을 유지하며 튜닝이 필요 없는 설계로 주목받았습니다. 이 연구는 실제 어플리케이션에서 더 나은 성능을 제공할 수 있는 가능성을 보여줍니다.



### SVHighlights: Towards Extremely Long Sport Video Highlight Detection (https://arxiv.org/abs/2606.06926)
Comments:
          Accepted to KDD 2026 (Datasets and Benchmarks Track). Project Page: this https URL

- **What's New**: 이번 논문에서는 극도로 긴 스포츠 비디오(1시간 이상)에서의 하이라이트 탐지를 위한 최초의 벤치마크인 SVHighlights를 소개합니다. 기존의 대부분의 연구는 짧은 형식의 비디오에 중점을 두었으며, 이 벤치마크는 여러 스포츠 카테고리를 포함합니다. 이를 통해 하이라이트 검출의 새로운 가능성을 열어줍니다.

- **Technical Details**: SVHighlights는 전체 스포츠 비디오와 해당 하이라이트 비디오의 쌍으로 구성되며, 공식 하이라이트 비디오를 정답으로 사용하여 복잡한 주석 작업 없이 레이블을 자동 생성합니다. 또한 TF-SELECTOR라는 훈련 없는 세그먼트 기반 접근 방식을 제안하는데, 이는 인접한 샷을 병합하여 의미론적으로 일치하는 세그먼트를 형성하고, 이를 통해 컨텍스트 인식 기반의 점수 예측을 합니다.

- **Performance Highlights**: TF-SELECTOR는 다양한 메트릭에서 VTG 튜닝된 기준 모델보다 우수한 성능을 보여주며, 특히 HIT@1에서 +3.12, HIT@K에서 +4.06, IoU에서 +2.95의 개선을 기록했습니다. 이러한 결과는 SVHighlights가 긴 형식의 하이라이트 탐지 테스트베드로서의 어려움을 제시하며, 단순한 세그먼트 기반 전략이 긴 비디오에 효과적으로 적용될 수 있음을 입증합니다.



### DRIFT: From Robustness Gaps to Invariance Manifolds for AI-Generated Image Detection (https://arxiv.org/abs/2606.06918)
Comments:
          Submitted to ECCV 2026

- **What's New**: 이번 논문에서는 AI 생성 이미지 탐지를 기존의 접근 방식과 다르게 구조화된 불변성 다양체(invariance manifold) 학습 문제로 재구성했습니다. 실제 이미지를 기반으로 한 단일 클래스(supervision) 학습을 통해 불변성을 명확히 모델링할 수 있습니다. 새로운 방식으로 경량 프로젝션 헤드를 도입하여 표현 공간을 강건한 및 취약한 하위 공간으로 분해하여, 불변성 계층 구조를 학습하는 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 우선적으로 두 개의 경량화된 프로젝션 헤드를 사용해 강건한(more robust) 및 취약한(fragile) 임베딩을 정의합니다. 강건한 하위 공간은 물리적으로 그럴싸한 변형에 의한 변화를 억제하도록 훈련되며, 취약한 하위 공간은 편집적 변화에 민감하게 반응합니다. 추론 시, 다중 스케일 패치 기반의 드리프트 분석을 통해 두 개의 하위 공간 간의 위반(margin violation)을 테스트하여 이미지의 진위를 분류합니다.

- **Performance Highlights**: 광범위한 실험을 통해 제안된 방식이 보이지 않는 생성기와 해상도에서 강력한 범위 일반화(open-world generalization)를 달성함을 입증했습니다. 기존의 훈련 없는 강건성 기반 기준선에 비해 일관되게 우수한 성능을 나타내었으며, 해석 가능한 불변성 위반 지도(invariance-violation maps)를 제공하여 결과의 신뢰성을 높였습니다.



### polyDAG: Polynomial Acyclicity Constraints for Efficient Continuous Causal Discovery in Visual Semantic Graphs (https://arxiv.org/abs/2606.06908)
- **What's New**: 본 논문에서는 visual semantic graphs(시각적 의미 그래프)에서 연속적인 인과적 발견을 위한 효율적인 Polynomial acyclicity framework인 polyDAG를 제안합니다. polyDAG는 매트릭스 지수적 acyclicity 제약 조건을 유한한 다항 트레이스 제약으로 대체하고, 새로운 제약 조건이 acyclic 그래프에 대해 정확히 0임을 증명합니다. 또한 기하급수적 구현을 통해 명시적인 합산 루프를 피하면서 동일한 acyclicity 조건을 유지합니다.

- **Technical Details**: polyDAG는 directed acyclic graph(DAG) 구조를 효율적으로 학습하기 위한 방법입니다. 기존 방법의 acyclicity 제약 조건은 매트릭스의 지수 또는 특이값 분해 등으로 인해 O(d^3) 시간이 소요됩니다. 이에 반해, polyDAG는 다항식 및 기하급수적 접근을 통해 연산 시간을 대폭 줄이고, d가 수백 개 이상의 노드를 넘어설 때에도 유용하게 적용될 수 있습니다.

- **Performance Highlights**: 실험 결과, synthetic Erdos-Renyi 그래프와 CelebA 얼굴 시각 속성 데이터세트에서 polyDAG는 평균적으로 구조적 Hamming 거리(mean structural Hamming distance)를 318.4에서 285.4로 감소시키고, 평균 F1 점수를 0.725에서 0.756으로 향상시켰습니다. 100개의 노드에서 기하급수적 변형은 3.44초에 실행되어, 기존의 5.16초와 비교했을 때 33.4%의 속도 향상을 나타냅니다.



### Beyond Skeletons: Learning Animation Directly from Driving Videos with Same2X Training Strategy (https://arxiv.org/abs/2606.06903)
Comments:
          Accepted to ICLR 2026

- **What's New**: 이번 연구에서는 DirectAnimator라는 프레임워크를 제안하며, 이는 중간 단계의 포즈 추출을 생략하고 원시 드라이빙 비디오에서 직접 학습합니다. 기존 연구는 주로 포즈 추정기를 이용하여 중간 표현을 추출하지만, 이는 차단 또는 복잡한 자세에서 오류가 발생하기 쉽습니다. DirectAnimator는 포즈, 얼굴, 위치 정보를 포함하는 Driving Cue Triplet을 도입하여 안정적이고 의미론적으로 풍부한 형태로 모션과 표정을 포착합니다.

- **Technical Details**: DirectAnimator는 CueFusion DiT 블록을 통해 포즈, 얼굴, 위치 단서를 융합하여 데이터 노이즈 제거 과정에서 신뢰성을 높입니다. 또한, Same2X 훈련 전략을 통해 서로 다른 신원 간의 피쳐를 정렬하여 최적화를 정규화하고 수렴 속도를 가속화합니다. 이 시스템은 관찰된 신원과 드라이빙 신원이 다를 때에도 안정적인 학습을 가능하게 합니다.

- **Performance Highlights**: 광범위한 실험을 통해 DirectAnimator는 최첨단 시각적 품질을 달성하며 신원 보존이 뛰어난 성능을 보였습니다. 또한, occlusion(차단)과 복잡한 신체 자세에 견고하며, 더 적은 계산 자원으로도 높은 품질을 유지합니다. 이러한 결과는 DirectAnimator의 구조적 접근 방식이 효과적임을 잘 보여줍니다.



### LUCID: Learning Unified Control for Image Deflaring and Exposure Mastery in Nighttime Photography (https://arxiv.org/abs/2606.06901)
Comments:
          Accepted by SIGGRAPH 2026

- **What's New**: 이 논문에서는 밤 시간의 사진 복원 문제를 다룰 수 있는 LUCID라는 통합 프레임워크를 소개합니다. 기존의 접근법은 야간 장면에서 발생하는 다양한 왜곡들을 독립적으로 다루었으나, LUCID는 이를 연속적이고 조절 가능한 프로세스로 제시하여 보다 효율적인 복원 방법을 제공합니다. 이를 통해 사용자들은 복원 과정에서 조명 원천과 관련된 플레어 및 고스트 아티팩트를 선택적으로 제어할 수 있는 기능을 가질 수 있습니다.

- **Technical Details**: LUCID는 주로 두 가지 구성 요소로 나누어져 있습니다: 플레어 분리 모듈과 확산 기반 복원 모듈입니다. 첫 번째 모듈은 광학 아티팩트의 ‘커튼’을 분리하여 신뢰할 수 있는 구조적 지도를 생성하고, 두 번째 모듈은 이 정보를 활용하여 깨끗하고 잘 노출된 이미지를 재구성합니다. 새롭게 도입된 네 가지 모드의 훈련 전략을 통해 사용자는 복원 과정에서의 제어 가능성을 극대화할 수 있습니다.

- **Performance Highlights**: LUCID는 다양한 실제 야간 시나리오에서 최신 기술(SOTA) 기준을 뛰어넘는 성능을 보여주었습니다. 실험 결과, LUCID는 시각적으로 우수한 결과를 생성하며, 고동적 범위(HDR) 재구성을 지원하고, 사실적인 조도 전환을 복원하는 데에도 효과적입니다. 이러한 특성을 바탕으로 LUCID는 자동화된 향상도 가능하면서 예술적 표현을 위한 유연한 도구로 자리잡을 수 있을 것으로 기대됩니다.



### Lighting-Aware Representation Learning under Controllable Lighting Variation (https://arxiv.org/abs/2606.06899)
- **What's New**: 이 논문은 조명 변화를 고려한 새로운 시각 표현 학습 프레임워크를 제안합니다. 이 방법은 기존의 데이터 증강 기법을 넘어서 조명 정보를 명시적으로 모델링하여, 학습 과정에서 조명 변화를 훈련 신호로 사용합니다. 따라서 모델은 시맨틱 일관성을 유지하면서 조명에 민감한 시각 구조를 동시에 학습할 수 있습니다.

- **Technical Details**: 이 논문은 MoCo V2와 같은 대조 학습(constrastive learning) 프레임워크를 기반으로 하며, 추가적인 목적 함수를 도입하여 조명 의존적 변화를 캡처합니다. House100KLighting 데이터셋을 사용하여 조명 환경을 조작하며, 이미지 분류와 객체 탐지 작업에서 모형의 성능을 개선합니다. 정량적 실험을 통해 조명 인식 모델이 기존의 대조 학습 기준선에 비해 성능이 일관되게 향상되는 것을 보여줍니다.

- **Performance Highlights**: 제안된 조명 인식 훈련 방법은 표준 대조 학습에 비해 낮은 조명, 복잡한 조명 상태에서도 우수한 성능을 발휘합니다. 이 연구 결과는 향후 모델의 강건성과 적응력을 향상시킬 수 있는 잠재력을 보여줍니다. 또한 조명 외의 다양한 조절 가능한 요인에 대해서도 적용 가능성을 시사합니다.



### Stream3D-VLM: Online 3D Spatial Understanding with Incremental Geometry Priors (https://arxiv.org/abs/2606.06891)
Comments:
          Project Page: this https URL

- **What's New**: 이번 논문에서는 기존의 오프라인 환경에서 작동하던 3D 대규모 멀티모달 모델의 한계를 극복하고, 실시간으로 공간 이해를 가능하게 하는 온라인 3D 비전-언어 모델인 Stream3D-VLM을 제안합니다. 이 모델은 스트리밍 비디오에서 자율적으로 반응 시점을 결정하여 자연스럽고 유연한 상호작용을 지원합니다. 또한, geometry priors를 점진적으로 통합하여 우수한 3D 지각 성능을 발휘합니다.

- **Technical Details**: Stream3D-VLM은 LLM의 autoregressive training을 기반으로 응답 타이밍을 학습하며, Visual-Spatial Feature Integration (VSFI) 모듈을 통해 비주얼 스트림에 기하학적 정보(geometry prior)를 주입합니다. 서로 다른 비디오에서 실시간으로 데이터를 처리하기 위해 Geometry-Adaptive Voxel Compression (GAVC) 모듈을 도입하여 비주얼 토큰을 압축함으로써 지연 시간을 최소화합니다. 이 모델은 오프라인 및 온라인 3D 언어 과제에서 모두 뛰어난 성능을 보입니다.

- **Performance Highlights**: 심층 실험 결과, Stream3D-VLM은 온라인 3D 공간 이해 및 추론에서 최첨단 성능을 달성하였으며, 오프라인 태스크인 비주얼 그라운딩과 밀집 캡셔닝에서도 주도적인 성과를 보였습니다. 본 연구는 실제 응용 프로그램에서 3D LMM을 배포하기 위한 필수적인 단계로 여겨집니다. 우리는 1백만 개 이상의 QA 쌍을 포함하는 데이터 생성 파이프라인과 29개의 과제를 가진 새 벤치마크를 통해 모델의 성능을 체계적으로 평가했습니다.



### Diagnosing Visual Ignorance in Vision-Language Models (https://arxiv.org/abs/2606.06890)
- **What's New**: 이번 연구는 Vision-Language Models (VLMs)의 언어 우선 의존성이 내부 메커니즘 및 행동적 관점에서 어떻게 작용하는지를 분석합니다. 이전 연구에서 발견된 내용 외에도, 우리는 VLM의 구조적 불균형과 언어 우선 의존성이 모델의 평가 기준에 미치는 영향에 대한 기존 이해도를 깊이 있게 탐구합니다. 또한, 우리는 시각 신호와 언어적 기대 간의 경쟁을 체계적으로 조사하기 위한 새로운 진단 프레임워크를 제안합니다.

- **Technical Details**: VLM의 내부 분석을 통해, 중간 레이어들이 시각 정보를 효과적으로 회수하지 못하고, 후반 레이어가 텍스트 편향을 우선시하는 경향이 있음을 발견했습니다. 이를 위해 카운터팩추얼 레이어 대체(counterfactual layer replacement)와 Layer-wise MLP probing을 결합하여 분석하였습니다. 외부 행동의 특성을 파악하기 위해, 우리는 다단계 가우시안 블러를 기반으로 한 진화하는 비주얼 디케이 메트릭을 도입하여 이미지 손상이 발생할 때 모델이 보여주는 반응의 변화를 정량화합니다.

- **Performance Highlights**: 12개의 비주얼 질문-답변 벤치마크를 평가한 결과, 약 20%에서 40%의 사례가 비주얼 내용이 완전히 파괴된 상태에서도 동일한 응답을 생성한다는 사실을 발견했습니다. 이는 현재 벤치마크가 비주얼 무지를 충분히 처벌하지 못하고 있음을 시사하며, 언어적 기대에 의존하는 모델에게 의도치 않게 보상을 제공할 수 있음을 보여줍니다. 이 연구는 VLMs의 내부 정보 라우팅 실패와 비주얼 손상에 대한 평가 설정 간의 연결을 제공하며, VLM의 외부 행동을 분석하기 위한 다각적 프레임워크 역할을 합니다.



### ARAPDiffusion: ARAP Regularization for Diffusion-Based Deformable Shape Space Learning (https://arxiv.org/abs/2606.06887)
- **What's New**: 이 논문에서는 ARAPDiffusion이라는 새로운 latent diffusion (LD) 모델을 소개합니다. 이 모델은 deformations shape collection의 기저가 되는 연속 shape space를 학습하는 것을 목표로 하며, as-rigid-as-possible (ARAP) 변형 모델을 정규화 손실로 주입하여 3D 훈련 데이터의 필요성을 줄입니다. 기존의 LD 모델을 개선하는 방법을 제안하고, 합성 분포를 사용하여 인코더/디코더와 LD 모델 모두를 발전시키는 훈련 과정을 설명합니다.

- **Technical Details**: ARAPDiffusion 모델은 세 가지 단계로 구성됩니다. 첫 번째 단계는 훈련 데이터를 사용하여 일반적인 LD 모델을 사전 훈련합니다. 두 번째 단계에서는 현재 LD 모델이 정의하는 latent space의 분포를 사용하여 ARAP 정규화 손실을 적용하여 디코더를 미세 조정합니다. 마지막으로 세 번째 단계에서는 현재 shape 디코더로부터 추출한 합성 분포에 의해 정규화된 LD 모델을 미세 조정합니다. 두 번째와 세 번째 단계는 교대로 진행됩니다.

- **Performance Highlights**: ARAPDiffusion은 일관된 메쉬 및 비조직화된 포인트 클라우드 데이터를 기반으로 평가되었으며, 정량적 및 정성적으로 최신 접근 방식보다 개선된 결과를 제공합니다. 이 모델은 인간과 동물 데이터 세트를 포함하여 메쉬와 암시적 변형 생성 모델 모두에서 일관된 향상을 보여주었습니다. 실험 결과는 ARAPDiffusion의 장점을 입증하며, 기존의 접근 방식에 비해 우수한 성능을 발휘합니다.



### FreeAnimate: Training-Free Human Image Animation with Preview-Guided Denoising (https://arxiv.org/abs/2606.06885)
Comments:
          Accepted to IEEE ICASSP 2026

- **What's New**: 이번 연구에서는 FreeAnimate라는 훈련이 필요 없는 인간 이미지 애니메이션 프레임워크를 소개합니다. 기존의 방법들은 대량의 훈련 데이터와 자원을 요구했으나, FreeAnimate는 사전 훈련된 이미지 확산 모델의 잠재력을 활용하여 시간 일관성(temporal consistency), 정체성 보존(identity preservation), 배경 안정성(background stability)을 확보합니다. 이 접근 방식은 새로운 미리보기 생성 전략(preview generation strategy)을 통합하여 훈련 없이도 포즈 정렬(pose alignment)과 배경의 일관성(background consistency)을 효과적으로 잠재울 수 있습니다.

- **Technical Details**: FreeAnimate는 두 개의 주요 모듈인 Inversion-Boosted Attention(IBA)와 Reference-Anchored Self-Attention(RA-SA)을 도입하여 시간 일관성과 정체성 보존을 보장합니다. IBA는 미리보기 프레임에서 얻은 주의(attention) 맵을 활용하여 개선된 구조적 일관성을 달성합니다. RA-SA는 레퍼런스 이미지를 기준으로 프레임의 일관성을 강화하여 최종적으로 품질 높은 비디오 애니메이션을 생성하는 데 기여합니다.

- **Performance Highlights**: 실험 결과, FreeAnimate는 기존의 훈련이 필요 없는 경쟁자들과 훈련 기반의 기준 방법들을 초과하여, 최첨단 방법들과 동등한 생성 품질을 달성하였습니다. 다양한 데이터셋에서 강력한 일반화 능력을 보여주어, 실제 애플리케이션에서의 사용 가능성을 높이고 있습니다.



### Unified Safe In-context Image Generation in Multimodal Diffusion Transformers via Restricting Unsafe Information Flows (https://arxiv.org/abs/2606.06875)
Comments:
          ICML26

- **What's New**: 본 논문에서는 다중모드 주의(attention) 메커니즘을 갖춘 확산 변환기(Diffusion Transformers, DiTs)가 이미지 생성의 주요 패러다임으로 자리 잡았음을 설명합니다. 그러나 안전하지 않은 콘텐츠의 생성을 방지하는 것이 특히 이미지-투-이미지(image-to-image, I2I) 편집 작업에서 중대한 과제로 남아있습니다. 기존의 안전성 메커니즘은 텍스트-투-이미지(text-to-image, T2I) 합성에 주로 초점을 맞추고 있어 DiT 기반 프레임워크에서의 안전성 저해를 위한 통합적 접근이 부족합니다. 이를 해결하기 위해 훈련이 필요 없는 안전 생성 프레임워크인 통합 시각 안전 조절기(Unified Visual Safety Regulator, UVR)를 제안합니다.

- **Technical Details**: UVR은 다중모드 주의(MM-Attn)에서의 주의 역학을 정보 흐름 관점에서 분석하여 생성된 이미지의 안전하지 않은 의미를 조절합니다. 연구 결과, 안전하지 않은 의미 세그먼트가 출력 패치에 빠르게 나타나는 비특정 작업 의 시작 단계와 그 이후에 발생하는 작업별 의미 강화 및 간섭 단계를 식별합니다. UVR은 이러한 관찰에 기반하여 안전하지 않은 정보를 식별하고 해당 정보 흐름을 제한하여 안전하지 않은 콘텐츠 생성을 방지하는 방식을 채택합니다.

- **Performance Highlights**: UVR은 이미지 합성과 편집 작업에서 각각 91%와 77%의 erase rate를 달성하며, 시각적 품질과 신뢰성을 최소한의 열화로 유지합니다. 다양한 개념에 대한 실험을 통해 UVR은 최첨단 성능을 보여주며, 생성과 편집 성능을 유지하면서도 안전성을 높이는 데 성공했습니다. 궁극적으로 UVR은 다중모드 DiTs에서 T2I 합성과 I2I 편집 모두에 대한 통합적인 안전성 완화 전략을 제공합니다.



### EgoPressDiff: Multimodal Video Diffusion for Egocentric UV-Domain Hand-Pressure Estimation (https://arxiv.org/abs/2606.06872)
Comments:
          Accepted to IEEE ICASSP 2026

- **What's New**: 이번 연구에서는 egocentric 카메라로부터 손 표면 접촉 압력을 추정하는 새로운 방법인 EgoPressDiff를 소개합니다. 기존 방식들이 압력 신호를 비효율적으로 처리했던 것에 반해, EgoPressDiff는 비디오 확산 프레임워크를 사용하여 압력 맵을 생성합니다. 이 방식은 PoseNet과 Vertex Encoder를 포함한 다중 모달 조정 전략을 통해 손의 자세와 3D 메쉬 정점에서 효율적으로 특징을 추출합니다.

- **Technical Details**: EgoPressDiff의 아키텍처는 PoseNet, Vertex Encoder 및 조정된 공간 레이어를 포함합니다. PoseNet은 손의 자세 특징을 추출하고, Vertex Encoder는 MANO 손 모델의 3D 정점 좌표를 이용해 압력과의 관계를 학습합니다. 마지막으로, 조정된 공간 레이어는 서로 다른 도메인에서 온 다중 모달 조정 신호를 효과적으로 통합할 수 있도록 설계되었습니다.

- **Performance Highlights**: EgoPressDiff는 EgoPressure ego-view 설정에서 기존 방법들보다 34% 이상 개선된 Volumetric IoU를 달성하며, MAE를 감소시키고 높은 시간 정확성을 유지합니다. 이는 연속적이고 시간적으로 인식 가능한 모델링의 우수성을 입증합니다. 이러한 성과는 기존의 정적이고 픽셀 단위 분류에서 동적 비디오 생성 방식으로의 혁신적인 전환에 기반합니다.



### Multi-FRuGaL: Multimodal Flexible Redundancy-aware Decomposed Gated Learning for Cancer Diagnosis and Prognosis (https://arxiv.org/abs/2606.06867)
- **What's New**: 이 논문에서는 Multi-FRuGaL이라는 최신 다중모달 융합 프레임워크를 제안합니다. 이 프레임워크는 신호 분해 레이어를 도입하여 데이터 부족 시에도 각 모달리티의 표현을 조정할 수 있도록 설계되었습니다. Multi-FRuGaL은 입력 조건부 게이팅 네트워크를 활용하여 정보가 많은 모달리티는 강화하고, 중복되거나 노이즈를 포함한 입력은 억제합니다.

- **Technical Details**: Multi-FRuGaL은 각 모달리티의 표현을 공통 및 모달리티별 구성요소로 분리하는 신호 분해(Signal Decomposition) 레이어를 특징으로 합니다. 또한, 입력 조건부 게이팅 메커니즘(Input-conditioned Gating Mechanism)을 통해 모달리티 간의 가중치를 동적으로 조절하며, 정보 예산 정보-인식 융합 손실(Information-Budget Regularizer)을 포함하여 중복된 정보를 최소화합니다.

- **Performance Highlights**: Multi-FRuGaL은 HANCOCK과 HECKTOR 데이터셋에서 생존 및 재발 예측 과제를 수행하여 기존 기초선보다 일관되게 높은 평균 성능을 달성했습니다. 생존 분석에서 Overall Survival에 대해 0.6814의 Concordance Index를 기록했으며, HECKTOR에서는 HPV 예측 성능이 0.975 AUC로 향상되었습니다. 이러한 결과는 Multi-FRuGaL이 중복 모달리티 조건에서도 강력한 다중모달 표현을 학습할 수 있음을 보여줍니다.



### LRMIL: Efficient Low-Resolution Multiple Instance Learning via High-Resolution Knowledge Distillation for Whole Slide Image Classification (https://arxiv.org/abs/2606.06864)
- **What's New**: 이 논문은 디지털 병리학에서 전체 슬라이드 이미지(Whole Slide Image, WSI) 분석에 있어 저해상도 다중 인스턴스 학습(low-resolution Multiple Instance Learning, LRMIL) 프레임워크를 제안합니다. LRMIL은 고해상도 패치에서 학습한 지식을 저해상도 표현으로 전달하는 효율적인 방법을 제공합니다. 이를 통해 기존 방법의 계산적 효율성과 표현 능력의 한계를 극복하려는 시도를 합니다.

- **Technical Details**: LRMIL은 두 단계의 지식 증류(distillation) 전략을 채택하고 있습니다. 첫 번째 단계에서는 패치 수준에서 크로스 해상도 증류(cross-resolution distillation)를 통해 저해상도 패치 임베딩을 고해상도 표현과 정렬하며, 두 번째 단계에서는 슬라이드 수준 지식 증류(slide-level knowledge distillation)를 이용하여 저해상도 MIL 모델을 훈련합니다. 이 과정에서 저해상도 패치에서만 연산을 수행함으로써 데이터 전처리와 계산 비용을 상당히 줄일 수 있습니다.

- **Performance Highlights**: 다양한 WSI 벤치마크에 대한 실험 결과 LRMIL은 최신 MIL 방법들보다 지속적으로 우수한 성능을 보였습니다. LRMIL은 임상 병리학 워크플로우에서 효율적이고 확장 가능한 솔루션으로 주목받고 있습니다. 이러한 결과는 저해상도를 사용한 효율적인 추론이 가능하다는 점에서 큰 의의를 가집니다.



### FS-DVS: A Frequency-Selective Dynamic Visual Sensing Paradigm for Enhancing Information Completeness (https://arxiv.org/abs/2606.06856)
- **What's New**: 본 논문은 FS-DVS(Frequency-Selective Dynamic Vision Sensor)라는 새로운 감지 패러다임을 제안합니다. 이는 생물학적 망막 신경세포(RGCs)의 집합 메커니즘을 모방하기 위해 이벤트 트리거링보다 앞서 학습 가능한 공간 필터를 통합합니다. 이를 통해 기존 DVS의 한계를 극복하고 구조적 완전성을 회복함으로써 성능을 크게 향상시킵니다.

- **Technical Details**: FS-DVS는 고유한 공간 필터를 사용하여 이벤트 생성 과정을 최적화합니다. 이 필터는 인식 및 탐지와 같은 다운스트림 작업에 맞게 최적화될 수 있습니다. 연구 결과, 필터는 중간 주파수 성분을 강조하는 중심-주위 형태로 스스로 진화하며, 인간의 CSF(Contrast Sensitivity Function)와 정렬됩니다.

- **Performance Highlights**: FS-DVS는 객체 탐지에서 12.3% mAP와 같은 상당한 성능 향상을 달성하며, 사람과 유사한 CSF 특성을 타 작업에서도 일관되게 반영합니다. 필터의 일반성은 다양한 시각적 작업에서 우수한 성능을 나타내며, 정보 이론적 분석 결과 FS-DVS가 기존 DVS보다 효율적인 정보를 보존함을 보여줍니다.



### MotionEnhancer: Leveraging Video Diffusion for Motion-Enhanced Vision-Language Models (https://arxiv.org/abs/2606.06853)
Comments:
          Accepted by CVPR 2026

- **What's New**: 본 논문에서는 비디오 이해를 위한 비전-언어 모델(Vision-Language Models, VLMs)의 모션 이해 능력을 향상시키기 위해 새로운 접근법인 MotionEnhancer를 소개합니다. MotionEnhancer는 강력한 비디오 확산 모델(Video Diffusion Model, VDM)에서 추출한 모션 프라이어(motion priors)를 보조 감독(auxiliary supervision)으로 활용하여 주의 정렬(attention alignment)을 통해 VLM의 모션 이해 능력을 높입니다. 이 접근법은 복잡한 모듈 설계 없이도 적용 가능하다는 점에서 주목할 만합니다.

- **Technical Details**: MotionEnhancer는 Motion-sensitive Head Selection (MHS)와 Motion-salient Text Token Identification (MTTI)이라는 두 개의 파라미터가 없는 모듈로 구성되어 있습니다. MHS는 모션에 관련된 주의 지도를 추출하기 위해 시간적 주의 지도를 평가하며, MTTI는 프레임 간 평균 값을 계산하여 원활하거나 급격한 모션에 반응하는 텍스트 토큰을 식별합니다. 이 두 모듈은 별도의 학습 파라미터 없이 VDM에서 직접 모션 관련 주의 지도를 추출하고 최적화할 수 있도록 설계되었습니다.

- **Performance Highlights**: 광범위한 실험을 통해 MotionEnhancer는 두 개의 모션 수준 비디오 이해 벤치마크에서 최첨단 VLM들에 비해 지속적인 성과 향상을 달성할 수 있음을 입증했습니다. MotionEnhancer는 비디오 QA 쌍에서의 모션 중심 주의 신호를 효율적으로 추출할 수 있도록 하여 VLM의 모션 인식 및 추론 능력을 크게 개선합니다. 연구 결과는 주의 정렬 전략이 비디오 작업으로 성공적으로 확장될 수 있음을 보여주며, 이는 적은 수정으로도 가능하다는 점에서 의미가 큽니다.



### CFRNet: Cycle-Consistent Fixed-Point Training for Real-Time Blind Face Restoration on Consumer Embedded NPUs (https://arxiv.org/abs/2606.06850)
Comments:
          12 this http URL and project page will be released

- **What's New**: 이 논문은 소비자 기기에서 사용하는 블라인드 얼굴 복원 기술을 소개합니다. 특히, CFRNet이라는 새로운 모델을 제안하며, 이는 시각적 품질과 속도, 메모리 간의 균형을 잘 맞춥니다. Cycle-Consistent Fixed-Point Training (CCFP)이라는 훈련 방법을 통해 얼굴을 복원하는 과정에서 반복적인 사용이 효과적이도록 학습됩니다.

- **Technical Details**: CFRNet은 2.0M 매개변수를 가진 ResNet 스타일의 복원기로, 256x256 크기의 얼굴 이미지를 처리하는 데 최적화되어 있습니다. CCFP는 세 가지 손실 함수를 사용하여 모델을 훈련시키며, 이로 인해 더 높은 품질의 이미지를 반복적으로 생성할 수 있습니다. 고칙적인 CNN 연산자만을 사용하여, INT8에서 약 23ms의 빠른 속도를 자랑합니다.

- **Performance Highlights**: CFRNet은 300개의 테스트 이미지 세트에서 가장 높은 LPIPS를 달성하였고, PSNR 및 SSIM에서도 최고의 성능을 보였습니다. k=3에서 복원한 결과는 품질 조정이 필요 없이 성능이 향상되었음을 보여주었습니다. 또한, 같은 개념이 일반 CNN에도 적용 가능하며, 실시간으로 차량에 탑재된 모니터링 시스템에서 테스트되었습니다.



### AdaGRPO: A Capability-Aware Adaptive Enhancement for Flow-based GRPO (https://arxiv.org/abs/2606.06828)
Comments:
          Project Website: this https URL

- **What's New**: 이번 연구는 Group Relative Policy Optimization (GRPO) 방식의 흐름 모델이 인간의 선호와 비효율적으로 일치한다는 사실을 밝혔습니다. 저자들은 현재의 GRPO가 학습자의 역량과 동떨어져 있으며, 프롬프트 선택 및 이점 추정 과정에서 문제가 발생한다는 점을 강조합니다. 이를 해결하기 위해 Adaptive GRPO (AdaGRPO)를 제안하여 이러한 문제들을 개선하고자 합니다.

- **Technical Details**: AdaGRPO는 두 가지 주요 구성 요소로 이루어져 있습니다. 첫째, Online Curriculum Filtering Strategy를 통해 모델의 능력에 따라 최적의 프롬프트를 동적으로 선택하고, 둘째, Cross-Level Advantage Fusion을 통해 intra-group 과 macro-level 이점을 통합하여 보다 정확한 정책 평가를 제공하는 구조입니다. 이 알고리즘은 기존의 GRPO 프레임워크에 통합되어 사용할 수 있는 경량화된 모듈입니다.

- **Performance Highlights**: AdaGRPO는 다양한 실험을 통해 성능 향상을 유도하고 GRPO 훈련 과정을 안정화하는 데 성공했습니다. 연구 결과, 이 새로운 알고리즘이 기존 방법보다 우수한 성능 및 안정성을 제공함을 보여주었습니다. 연구자들은 AdaGRPO가 선호 정렬 및 훈련 안정성을 개선하는 데 효과적임을 입증하였습니다.



### VideoSEG-O3: A Multi-turn Reinforcement Learning Framework for Reasoning Video Object Segmentation (https://arxiv.org/abs/2606.06819)
Comments:
          ICML2026

- **What's New**: 이번 연구에서는 비디오 객체 분할(Reasoning Video Object Segmentation, RVOS)을 위한 새로운 프레임워크인 VideoSEG-O3를 제안합니다. 이는 기존 RVOS 방식의 한계를 넘어, 복잡한 비디오 내에서의 물체 분할을 위한 구체적인 비주얼 증거를 능동적으로 수집할 수 있는 기능을 갖추고 있습니다. 특히, 이 방식은 인지적 과정인 '조잡하게-정밀하게'(coarse-to-fine) 접근 방식을 채택하고 있습니다.

- **Technical Details**: VideoSEG-O3는 다중 회전 강화 학습(multi-turn reinforcement learning) 프레임워크로, 이 과정에서 다중 회전의 시공간 체인-오브-생각(multi-turn temporal-spatial chain-of-thought)을 활용하여 중요한 간격과 키프레임을 반복적으로 확인합니다. 또한, 픽셀 수준의 분할 피드백을 토큰 수준의 로짓(logits)에 통합하는 SEG-aware Logit Calibration 기법을 도입하여, 세분화된 정보의 정확성을 높이고자 하였습니다. 최종적으로, 비디오에서의 시각적 토큰을 효율적으로 활용하기 위한 분리된 사고 추적(decoupled thinking trace)을 설계하였습니다.

- **Performance Highlights**: VideoSEG-O3는 8개의 RVOS 벤치마크에서 뛰어난 성능을 기록했습니다. 특히, 크기가 작은 모델(4B)임에도 불구하고 기존의 최고 성능(SOTA) 기법을 초과하여 LongRVOS 작업에서 +6.1%, MeViS에서 +4.2%, ReVOS에서 +4.0%의 성능 향상을 달성하였습니다. 이는 강화 학습 과정에서의 정책 최적화와 함께 반복적인 비주얼 탐색이 효과적임을 입증합니다.



### Breaking the Lock-in: Diversifying Text-to-Image Generation via Representation Modulation (https://arxiv.org/abs/2606.06813)
Comments:
          Accepted to ICML 2026. Code is available at: this https URL

- **What's New**: 이 논문에서는 최신 텍스트-이미지(T2I) 생성 모델들이 고급 Transformer 아키텍처 및 flow-based 목표를 기반으로 하여 높은 텍스트-이미지 정렬(text-image alignment)과 뛰어난 시각적 품질을 제공하지만, 고정된 프롬프트(prompt) 아래에서 유사한 샘플을 생성하는 문제를 다룹니다. 저자들은 zéro-frequency spatial average (DC) 성분이 초기 생성 단계에서 빠르게 수렴하여 다양성을 제한하는 원인을 분석하고, 이를 개선하는 새로운 방법인 DAVE(DC Attenuation for diVersity Enhancement)를 제안합니다.

- **Technical Details**: DAVE는 미세한 내부 연산을 통해 초기 단계에서 DC 성분을 선택적으로 약화시키는 기법입니다. 이 방법은 모델의 재훈련이나 샘플링 과정의 변경 없이, 계산 비용이나 메모리 전이의 부담 없이 신뢰성을 유지하면서도 고품질 이미지를 생성할 수 있도록 돕습니다. DAVE의 도입으로 인해 프롬프트에 일관되게 다양성을 증가시킬 수 있으며, 최근 주요 방법들과 비교할 때 경쟁력 있는 성능을 보입니다.

- **Performance Highlights**: DAVE는 기존의 방법들보다 훨씬 적은 비용으로도 경쟁력 있는 이미지를 생성할 수 있으며, 다양한 실험을 통해 그 효과를 입증하였습니다. 이러한 성능은 대규모 Transformer 기반 모델의 다양성을 증진하는 핵심 기여로, 생성 모델의 사용성과 확장성을 개선하는 데 중요한 역할을 할 수 있습니다. 이 연구는 T2I 모델의 다양성 제어를 위한 새로운 관점을 제공하며, 향후 연구에 대한 방향성을 제시합니다.



### MedSIGHT: Towards Grounded Visual Comprehension in Medical Large Vision-Language Models (https://arxiv.org/abs/2606.06760)
Comments:
          Accepted at ICML 2026

- **What's New**: MedSIGHT는 의학적 대형 비전-언어 모델(Med-LVLM)을 위한 통합 프레임워크로, 시각적 이해(visual comprehension)를 향상시키기 위해 구조화된 픽셀 수준의 이해를 제공합니다. 이 모델은 새로운 Region Perceiver 모듈을 도입하여 지역 중심 토큰(region-centric tokens)을 생성하며, 의료 이미지와 관련된 해부학적 및 병리학적 영역을 상징적으로 나타내는 코드를 생성합니다. 기존의 Med-LVLM이 가지던 두 가지 주요 한계를 극복하여 보다 정밀한 진단 추론을 가능하게 합니다.

- **Technical Details**: MedSIGHT는 Region Perceiver와 의료 지역 코드북(modality-aware region codebook)을 포함하여, LLM의 입력 및 출력을 강화합니다. Region Perceiver는 패치 수준(patch-level) 피처와 픽셀 수준(pixel-level) 이해 간의 격차를 메우며, 계층별로 시각적 피처를 점진적으로 업샘플링합니다. 코드북은 연속적인 지역 임베딩(region embeddings)을 해석 가능한 코드 토큰으로 변환하여, 다양한 해부학적 구조를 표현할 수 있게 합니다.

- **Performance Highlights**: MedSIGHT는 72K의 멀티모달 지시 쌍(multimodal instruction pairs)에서 훈련되어, 다양한 이미징 방식(CT, MRI 등)에서 진단 추론과 공간적 기초(segmentation)에 대해 최첨단 성능을 달성하였습니다. 이 모델은 이미지를 기반으로 한 진단 추론과 지역 특정(segmentation) 출력을 모두 가능하게 해, 비전-언어 모델이 의학적 데이터를 더욱 효율적으로 처리할 수 있도록 기여합니다.



### Anchored, Not Graded: Vision-Language Models Fail at Slant-from-Texture Perception (https://arxiv.org/abs/2606.06714)
- **What's New**: 이 논문은 Vision-Language Models (VLMs)의 표면 경사 인식 능력을 조사하여, 인간의 시각적 편향이 VLMs에도 존재하는지를 확인합니다. 이전 연구에서는 비지도 학습된 CNN이 인간과 유사한 편향을 재현하는 반면, 감독된 CNN은 편향이 없음을 보여주었습니다. 이번 연구는 VLM이 이러한 기하학적 신호를 어떻게 인식하고 언어로 표현하는지를 분석하면서, 기존의 CNN과 VLM의 구조적, 학습 목표의 차이에 대한 통찰을 제공합니다.

- **Technical Details**: 연구는 합성 점-텍스처 표면을 사용하여 200개 조건의 팩토리얼 디자인을 통해 VLM의 성능을 평가합니다. 각 자극은 다양한 파라미터에 따라 사실상의 경사 및 시각적 정보의 조합에 기초하여 설계되었습니다. 실험은 VLM이 인간의 시각적 편향을 어떻게 반영하는지, 즉 볼록/오목 비대칭과 시야(Field of View) 효과를 포함한 편향을 평가하는 데 초점을 맞추었습니다.

- **Performance Highlights**: 실험 결과, VLM은 경사 예측 시 제한된 값 집합에 강하게 고정되는 경향을 보였으며, 자극의 파라미터에 따라 연속적인 예측을 하지 못했습니다. 감독된 미세 조정이 경사 예측의 단조 관계를 도입했으나 여전히 편향 문제가 남아 있음을 발견했습니다. 이러한 결과는 다양한 VLM의 행동이 인간 및 비지도 비전 모델과 어떻게 다른지를 보여 주며, 미래의 모델에서 개선이 필요한 부분을 강조합니다.



### USU-Corn-WeedDB: A UAV RGB Image Dataset for Multi-Species Weed Detection in Forage Corn (https://arxiv.org/abs/2606.06709)
Comments:
          8 pages, 4 figures, 1 table

- **What's New**: 새로운 데이터셋인 USU-Corn-WeedDB는 Utah의 Cache Valley에 위치한 상업용 사료용 옥수수 밭에서 수집된 UAV RGB 이미지로 구성되어 있습니다. 이 데이터셋은 다중 분류 잡초 탐지를 지원하기 위해 설계되었으며, 감독 학습(supervised learning) 및 반감독 학습(semi-supervised learning) 프레임워크에서 사용할 수 있습니다.

- **Technical Details**: RGB 이미지는 2025년 6월 27일 Autel EVO II Dual 640T V2 드론을 사용하여 약 10m 고도에서 획득되었으며, 픽셀 대 지상 샘플링 거리는 약 0.48 cm입니다. 총 366개의 전체 해상도 이미지를 640 x 640 픽셀 해상도의 8,800개의 패치로 분할하였고, 그 중 800개 이미지는 3종의 잡초에 대해 수동으로 주석이 추가되었습니다. 잡초 성분 불균형은 실제 필드 조건을 반영하며, redroot pigweed가 53.86%를 차지합니다.

- **Performance Highlights**: 28개의 객체 탐지 모델을 동일한 조건에서 학습했으며, 여기에는 YOLOv8, YOLOv9, YOLOv10, YOLOv11, YOLOv26, RT-DETR와 같은 다양한 아키텍처가 포함됩니다. 테스트 세트에서 mAP@0.5는 0.773에서 0.840 사이로 측정되었으며, 경량 모델이 UAV 시스템에서 경쟁력 있는 성능을 보였습니다.



### MMBU: A Massive Multi-modal Biomedical Understanding Benchmark to Probe the Perception Capabilities of Vision-Language Models (https://arxiv.org/abs/2606.06696)
- **What's New**: 이번 연구에서는 Massive Multimodal Biomedical Understanding (MMBU) 벤치마크를 소개합니다. MMBU는 35개의 하위 모달리티를 포함하여 현재까지 가장 큰 생물의학 비전-언어 모델(VLM) 벤치마크로, 모델 성능을 체계적으로 평가할 수 있게 합니다. 이 벤치마크는 개방형 및 폐쇄형 분류와 객체 탐지를 포함하여 다양한 생물학적 스케일, 임상 환경, 이미징 모달리티에서 평가를 제공합니다.

- **Technical Details**: MMBU는 410개의 데이터세트를 커버하며, 11개의 모달리티와 20개의 표본, 95개의 고유 관심 지역을 포함하고 있습니다. 각 데이터세트는 전문가들에 의해 수작업으로 주석이 달린 VQA(Visual Question Answering) 작업으로 변환되어 핵심 인지 작업인 분류 및 탐지의 평가를 지원합니다. 이러한 방대한 데이터는 정교한 주석을 통해 이미지 출처, 데이터세트 이름, 도메인 등 다양한 특성을 포함하고 있습니다.

- **Performance Highlights**: 각 모델과 작업 전반에 걸쳐 높은 오류율이 발견되었으며, 의료 적응으로부터 제한적인 개선이 있었습니다. MMBU를 통해 모델의 공간적으로 기반한 작업과 같은 고질적인 약점을 파악할 수 있었고, 기존 벤치마크에서의 높은 정확성이 실제 시나리오에서의 인지 능력을 과대평가할 수 있음을 보여주었습니다. 이를 통해 모델 개선을 위한 기초 데이터를 제공하고, 향후 생물의학 인지 작업을 위한 더 신뢰성 있는 모델 개발을 지원하고자 합니다.



### S23DR 2026 Winning Solution (https://arxiv.org/abs/2606.06695)
- **What's New**: 이 텍스트는 구문(SfM)에서 얻은 희소한 입력을 기반으로 하는 3D 와이어프레임 복원을 위한 S23DR 2026 챌린지의 우승 솔루션에 대해 설명합니다. 이 방법은 정점(Vertices)을 조건부 세트로 처리하며, Perceiver 스타일의 장면 토큰에 조건화된 흐름 일치 다중 샘플(consensus) 모델을 활용하여 64 개의 정점 토큰을 디노이즈합니다. 제안된 시스템은 개인 리더보드에서 1위를 차지하며 HSS (Hybrid Structure Score) 0.654를 달성했습니다.

- **Technical Details**: 이 방법은 3D 와이어프레임의 예측을 조건부 세트 생성으로 간주하고, 두 단계로 구성된 예측기를 사용합니다. 첫 번째 단계에서 전반적인 구조를 예측하고 두 번째 단계에서는 초기 예측에서 세밀한 수정을 수행합니다. 각 단계는 Perceiver 스타일의 장면 인코더를 사용하여 컨텍스트 토큰을 생성하고, 흐름 일치 ODE를 사용하여 64 개의 정점 슬롯을 디노이즈합니다.

- **Performance Highlights**: 제출된 시스템은 S23DR 2026 개인 테스트 리더보드에서 1위를 기록했으며, 두 번째 참가자와의 차이는 0.006입니다. 이 시스템은 모든 제출 중에서 가장 높은 정점 F1 점수인 0.791을 달성하였으며, 배경 잡음과 비정상적인 입력을 잘 처리하는 데 성공했습니다. 이 연구는 기존의 2D-3D 변환 방식에 비해 우수한 성능을 보여줍니다.



### RPC-GS: Gaussian Splatting with native RPC Rendering for Satellite Imagery (https://arxiv.org/abs/2606.06690)
- **What's New**: RPC-GS는 Rational Polynomial Camera (RPC) 모델과 함께 원주율 기반의 새로운 Gaussian Splatting 프레임워크를 도입합니다. 이 프레임워크는 기존의 원근법이나 아핀 카메라 모델을 대체하여 위성 이미지의 복잡한 이미지 기하학을 표현하는 데 있어 더 정확한 재구성을 가능하게 합니다. 또한, Gaussian 평균과 공분산을 RPC 모델을 통해 직접 투영함으로써 발생하는 기하학적 오류를 줄이도록 설계되었습니다.

- **Technical Details**: RPC-GS는 구체적으로 스플래팅이 적합한 장면 좌표를 글로벌 지리 좌표로 매핑하는 정밀한 지리-좌표 변환 체인을 통합합니다. 또한, Jacobian을 이용한 공분산 투영을 통해 RPC 모델에 맞춘 수치적으로 안정적인 공분산 프로젝션을 도출합니다. 이러한 점은 비선형 좌표 변환의 수치적 강건성을 보장하여 효율적인 재구성을 지원합니다.

- **Performance Highlights**: RPC-GS는 DFC2019 및 IARPA2016 데이터셋에서 기존의 원근법 및 아핀 모델의 평균 고도 오류를 각각 29.6% 및 63.8%, 9.9% 및 37.9% 개선하여 가장 낮은 재구성 오류를 달성했습니다. 이러한 성과는 RPC 모델과 원주율 기반 처리의 통합 결과이며, 코드 공개를 통해 위성 이미지에서의 Gaussian Splatting 연구를 지원할 계획입니다.



### RigPAPR: Rig-Based Animation of Static Neural Point Clouds from a Fixed-Viewpoint Video (https://arxiv.org/abs/2606.06685)
Comments:
          An overview video is available at this https URL

- **What's New**: 이번 연구는 RigPAPR이라는 새로운 방법론을 제안합니다. 이 방법은 고정된 시점에서 촬영된 비디오를 기반으로 정적인 PAPR 클라우드를 자동으로 리깅(rigging)하고 애니메이션을 생성할 수 있습니다. 기존의 메쉬 프록시나 포즈 의존적 보정 없이도 가능하여, 새로운 카테고리 템플릿을 사용할 필요가 없습니다.

- **Technical Details**: RigPAPR는 Proximity Attention Point Rendering (PAPR) 기반으로 동작하며, 각 프리미티브에 대해 고정된 형태 매개변수를 사용하지 않습니다. 대신 형태는 프리미티브의 공간적 구성이 변형될 때 렌더링 시점에서 자동으로 재구성됩니다. 이로 인해 조인트 경계에서 발생하는 간섭을 효과적으로 해결할 수 있습니다.

- **Performance Highlights**: 평가 결과, RigPAPR는 감독된 뷰에서 가장 강력한 기준선과 동등한 성능을 보였고, 신선한 신규 뷰에서 메쉬 기반 및 가우시안 스플래팅 기준선을 3dB 이상의 PSNR 성능 향상으로 초과했습니다. 합성 및 실제 객체 모두에서 더 깨끗한 조인트 경계 렌더링을 제공하여 기여하였습니다.



### Adaptive Band Selection for Hyperspectral Classification with Spatially Disjoint Evaluation (https://arxiv.org/abs/2606.06684)
Comments:
          6 pages, 2 figures, 3 tables

- **What's New**: SGBR-HC(Spectral-Group Band Ranking with Hard-Concrete initialization)는 하이퍼스펙트럼 이미지에서 차별화된 대상을 선택하기 위한 새로운 두 단계 방법을 제안합니다. 이 방법은 감독된 스펙트럼 순위를 이용하여 훈련 가능한 희소 게이트를 초기화함으로써, 사전 설정된 밴드 수를 회피할 수 있습니다. SGBR-HC는 Pavia University와 Houston 2013 데이터셋에서 최고 평균 전반 정확도와 Cohen's kappa를 기록하였습니다.

- **Technical Details**: 이 방법의 첫 번째 단계에서는 훈련 픽셀을 기반으로 후보 밴드를 순위화하고, 두 번째 단계에서는 이 순위를 기반으로 해서 Hard-Concrete 게이트를 최적화합니다. 준비된 밴드는 클래스 차별성과 스펙트럼 다양성에 따라 스코어링되어, 최종적으로 학습된 게이트 분포에서 임계값 없이 추출됩니다. 이 방식은 각 밴드의 차별성을 통해 픽셀 분류를 최적화하는 동시에 스펙트럼 구조를 유지합니다.

- **Performance Highlights**: SGBR-HC는 무작위 픽셀 분할에 비해 Pavia University에서 30.56 pp, Houston 2013에서 22.15 pp의 성능 향상을 보였습니다. 연구 결과는 선택 순위의 중요성을 강조하며, 평가 방법론이 하이퍼스펙트럼 밴드 선택의 성능에 중대한 영향을 미친다고 결론을 내립니다.



### JA-SIREN: Deterministic Initialization for Sinusoidal Networks via Spectral Matching (https://arxiv.org/abs/2606.06671)
- **What's New**: JA-SIREN (Jacobi-Anger Sinusoidal Representation Network)은 기존의 stochastic initialization 문제를 해결하는 새로운 deterministic initialization 방식을 제안합니다. 비슷한 학습 환경에서도 결과의 일관성과 품질을 보장하지 못하는 기존의 implicit neural representation(INR) 기법들을 극복하기 위해, 고전적인 스펙트럼 분석에 기초하여 두 층의 sinusoidal MLP의 가중치를 산정합니다. 이를 통해, 랜덤 시드나 하이퍼파라미터 튜닝 없이 준칙을 준수하는 신호 대표성 변환이 가능합니다.

- **Technical Details**: JA-SIREN은 특정 신호의 Discrete Sine Transform(DST)을 계산하고 Jacobi-Anger 확장을 활용하여 두 층의 sinusoidal MLP에 대해 닫힌 형태의 가중치를 도출합니다. 이러한 구조는 랜덤하게 초기화하는 대신, 신호의 초기 스펙트럼 반응과 수학적으로 일치하는 가중치로 구성되어 있습니다. 이 방식은 gradient update 이전에 네트워크를 전처리하여 추가적인 파라미터 없이 효율성을 극대화합니다.

- **Performance Highlights**: Kodak 데이터셋에서 JA-SIREN은 평균 PSNR이 67.18 dB를 기록하며, 기존 최고 기준에 비해 21.30 dB 개선된 성능을 보였습니다. 추가적으로, 모든 실행에서 변동성이 zero로 나타나, 스펙트럴 정보에 기반한 초기화가 sinusoidal INR의 stochastic 초기화보다 더 효과적이고 재현 가능한 대안임을 확인하였습니다. 이러한 결과는 과학적 컴퓨팅 및 시뮬레이션에서 필수적인 결과 재현성을 높이는 데 크게 기여할 것입니다.



### Architecture-Adaptive Uncertainty Fusion for Deepfake Detection (https://arxiv.org/abs/2606.06666)
- **What's New**: 이 논문에서는 Correlation-Optimized Fusion (COF)이라는 아키텍처 적응형 프레임워크를 제안합니다. COF는 다섯 가지 보완적인 불확실성 소스를 융합하여 피어슨 상관관계를 최대화하는 방식으로 작동합니다. 이 방법은 모델 수정 없이 간단한 가중치 최적화만으로 수행됩니다.

- **Technical Details**: COF는 5개의 정규화된 불확실성 소스로 구성된 입력 행렬을 사용하여, 비선형 메서드는 특정 아키텍처에서 더 높은 성능을 보이는 경향이 있음을 규명합니다. 제안된 방법은 Sequential Least Squares Programming (SLSQP)을 통해 해결되며, 서로 다른 아키텍처에 대해 독립적으로 최적의 융합 가중치를 학습하여 인덕티브 바이어스를 포착합니다.

- **Performance Highlights**: COF는 확실하게 높은 상관관계를 유지하며, 특정 아키텍처에서는 Random Forest보다 최대 7.3배 높은 성능을 기록했습니다. 그러나 모든 아키텍처에서 외부 데이터로 인한 불확실성 역전 현상이 관찰되었으며, 이로 인해 COF가 포렌식 배포를 위한 강력한 대안으로 여겨지고 있습니다.



### Inside the Visual Mind: Neuroscience-Motivated Concept Circuits for Interpreting and Steering Vision Transformers (https://arxiv.org/abs/2606.06664)
Comments:
          In Proceedings of the International Conference on Machine Learning, 2026. (acceptance rate 26.6%)

- **What's New**: 이 논문에서는 Vision Transformer (ViT)의 내부 작동을 이해하기 위한 ViSAE라는 기계적 해석 도구를 제안합니다. ViSAE는 서로 다른 약간의 문제를 해결하기 위해 설계되어 있으며, 특히 인간이 이해할 수 있는 개념으로 모델의 표현을 분해할 수 있습니다. 이 도구는 64K 이미지를 포함한 프로빙 세트와 16K 시각적으로 기반을 두고 있는 개념 어휘를 제공하여 개념 범위를 20배 향상시킵니다.

- **Technical Details**: ViSAE는 두 단계의 개념 회로 추적 알고리즘을 포함하며, 상향식(Top-down) 개념 읽기와 하향식(Bottom-up) 회로 추적을 통해 ViT의 작동 방식을 연구합니다. 이 알고리즘은 CLIP을 통해 미리 학습된 개념 어휘와 내재적 표현을 연결하여 자동으로 개념을 해석합니다. 또한, 이 방법은 반대 사실 개입(counterfactual interventions)을 통해 계층 간 인과 관계를 추정하여 개념 간의 상호작용 그래프를 생성합니다.

- **Performance Highlights**: ViSAE를 사용한 결과, WaterBirds 데이터셋에서 최악의 그룹 정확도가 48.2% 향상되었으며, 이는 기존 방법보다 23.8% 우수한 성능을 보여줍니다. 이러한 결과는 ViSAE의 효과적인 개념 편집 및 모델 조정 능력을 입증합니다. 중요한 것은, 이 도구가 ViT 모델의 내부 의사 결정을 추적하고 진단할 수 있게 해줘 사용자들에게 신뢰성을 제공합니다.



### From Pixels to Newtons: Predicting In Vivo Joint Contact Forces from Monocular Video (https://arxiv.org/abs/2606.06631)
- **What's New**: 이 논문은 비침습적으로 모노큘라 비디오에서 즉각적인 3D 엉덩이와 무릎 관절 비접촉력을 예측할 수 있는 물리학 없는 파이프라인을 제시합니다. 기존의 방법들과 달리 마커나 힘판, 근전도, 환자 특화 이미징 모델 또는 근골격계 모델이 필요하지 않습니다. 이 연구는 26명의 환자와 25개의 활동 범주를 기반으로 시험되어, 기존의 근골격계 시뮬레이션과 동일한 정확도를 보여줍니다.

- **Technical Details**: 활동 비디오에 따라 매 프레임에서 파라메트릭 바디 메시가 복구되고, 이를 kinematic features로 인코딩한 후, 트랜스포머를 통해 힘으로 디코딩합니다. 이 과정에서 신체 형태, 관절, 측면, 활동 텍스트 및 자가 지도 학습된 비디오 토큰에 의해 포즈 스트림이 적응적으로 조절됩니다. 이는 엉덩이와 무릎을 단일 모델로 통합합니다.

- **Performance Highlights**: 논문이 소개한 파이프라인은 26명의 환자에서 leave-one-subject-out 교차-validation을 통해 임상적으로 중요한 하중을 추적할 수 있으며, 기존의 방법과 비교하여 유사하거나 우수한 성능을 보여줍니다. 파이프라인은 또한 비디오 기능만으로도 정확도를 유지하며, 원시 영상에 대한 end-to-end 추론이 가능합니다. 이를 통해 임상 기록, 1차 진료 스크리닝 및 자택 재활 추적 분석에 대한 새로운 길을 열 수 있습니다.



### Direct 3D-Aware Object Insertion via Decomposed Visual Proxies (https://arxiv.org/abs/2606.06601)
Comments:
          ICML 2026; Project Page: this https URL

- **What's New**: OBJECT INSERTION (객체 삽입) 작업에서 최신 방법들은 Reference Object (참조 객체)의 착합을 배경 이미지의 특정 영역에 Seamless하게 수행하는 데 초점을 맞추고 있습니다. 반면 기존의 Diffusion 기반 방법들은 2D Inpainting (2D 인페인팅) 작업으로만 제한되어 있어 3D Pose (3D 포즈)에 대한 명확한 조정이 불가능한 점이 한계로 지적됩니다. 이러한 문제를 해결하기 위해, 우리는 Pose-Controlled Object Insertion (포즈 제어 객체 삽입)을 가능하게 하는 새로운 프레임워크인 DIRECT (Decomposed Injection for Reference Composition and Target-integration)를 제안합니다.

- **Technical Details**: DIRECT는 Appearance (외관), Geometry (기하학), Context (맥락)의 세 가지 상호 보완적인 구성 요소로 삽입 조건을 분해하여 각 요소를 별도의 경로로 주입하는 방식으로 설계되었습니다. 이를 통해 모델은 User-specified Pose (사용자 지정 포즈)를 엄격하게 따르면서도 Reference Appearance (참조 외관)을 동시에 유지할 수 있습니다. 또한, 새로운 데이터 구축 파이프라인을 통해 다양성과 질 높은 훈련 데이터 셋을 구축하여 모델의 일반화를 향상시킵니다.

- **Performance Highlights**: 실험 결과, DIRECT는 이전 방법들에 비해 Geometry Controllability (기하학적 제어 가능성)와 Visual Quality (시각적 품질) 모두에서 우수한 성능을 보였습니다. 본 연구의 방법이 3D 프라이어의 결함과 육안의 복잡한 포즈 변형에 대해 뛰어난 강건성을 가지고 있음을 보여주어 기존 방법들이 겪던 기하학적 왜곡과 텍스처 저하 문제를 효과적으로 해결했습니다. 이러한 결과는 특히 현실 세계의 복잡한 장면에서 모델의 일반화 능력을 향상시키는 데 기여합니다.



### Synthetic Benchmarks Overstate Forward-Forward Scaling: Real-Data Limits of Layer-Local Training (https://arxiv.org/abs/2606.06539)
Comments:
          23 pages, 6 figures

- **What's New**: 이 논문에서는 Forward-Forward (FF) 학습의 새로운 발전을 제시하며, DTG-FF라는 새로운 아키텍처를 개발했습니다. 이는 동적 온도 고도(dynamic temperature goodness), 분리된 정규화(decoupled normalization), 다층 융합(multi-layer fusion) 기술을 결합하여, FF 계열의 최신 기술을 여러 실제 데이터 벤치마크에서 확립합니다. 기존의 역전파(backpropagation) 대신 이 레이어 로컬(layer-local) 학습 방식이 실제적 규모에서 경쟁력을 유지할 수 있는지 조사했습니다.

- **Technical Details**: Forward-Forward 알고리즘은 레이어 로컬 기초 학습 방식을 통해 각 레이어가 양의 데이터에 대해 높은 '고도(goodness)'를 생성하고, 음의 데이터에 대해서는 낮은 고도를 생성하도록 훈련합니다. DTG-FF는 이 알고리즘을 실험적 목적으로 사용하기 위해 설계되었으며, 이를 통해 실제 데이터 스케일 진단과 서로 다른 아키텍처 간의 효용성을 비교하였습니다. 또한, DTG-FF는 높은 성능을 기록하고, 이론적으로 메모리에서의 효율성을 제공한다고 주장합니다.

- **Performance Highlights**: DTG-FF는 CIFAR-10에서 91.8%, ImageNet-100의 첫 번째 FF 기준에서 49.4%를 기록했습니다. 그러나 동일한 조건 하에서 역전파 기반의 딥서포트(BP-DeepSup)와 비교했을 때, CIFAR-10과 CIFAR-100에서 각각 2.40 및 5.93 pp의 성능 차이를 보이며, 클래스 수가 증가할수록 성능 격차가 더욱 확대되는 경향을 보였습니다. 이 논문은 FF의 계층적 로컬 학습이 실제 데이터에서 어떤 한계가 있는지를 설명하고 있습니다.



### WorldBench: A Challenging and Visually Diverse Multimodal Reasoning Benchmark (https://arxiv.org/abs/2606.06538)
Comments:
          Project page: this https URL

- **What's New**: 이번 연구에서 소개하는 WorldBench는 다양한 환경에서 신뢰성 있는 성능을 발휘해야 하는 Multimodal Large Language Models (MLLMs)를 평가하기 위한 도전적이고 시각적으로 다양한 reasoning benchmark입니다. 기존의 다중 모달 벤치마크가 시각적 다양성을 고려하지 않고 작업 유형만 확장하는 문제를 해결하고자 하였습니다. 우리는 다양한 도메인(예: 생물체)에서 수천 개의 시각적 개념을 정리하여 이러한 분류 체계를 구축하였습니다.

- **Technical Details**: WorldBench는 구조화된 시도와 오류를 통해 MLLMs가 답변하기 어려운 문제들을 수작업으로 설계하였습니다. 우리는 검색 엔진과 기존 데이터 세트에서 이미지를 수집하여 시각적 세계를 포괄적으로 표현하는 넓은 컬렉션을 curated 하였습니다. MLLMs 평가에 앞서, 우리는 수천 가지의 시각적 개념으로부터 세분화된 분류 체계를 구축하였습니다.

- **Performance Highlights**: WorldBench에서 15개의 MLLMs를 평가한 결과, 시각 이해에 대한 약점을 드러냈습니다. 가장 강력한 모델도 64.0% 정확도에 그쳤고, 일부 모델은 기회 수준을 약간 웃도는 성능을 보였습니다. 우리는 이 연구가 다중 모달 벤치마크 구축에서 시각적 다양성의 중요성을 강조하길 바랍니다.



### Attention-Guided Autoencoder Fusion for Insulator Defect Detection Using UAV Transmission-Line Imaging (https://arxiv.org/abs/2606.06536)
- **What's New**: 이 논문은 AE-YOLO라는 새로운 Attention-Guided AutoEncoder-Enhanced YOLO 프레임워크를 제안합니다. 이 프레임워크는 Insulator defect detection을 위해 설계되었으며, UAV 기반의 전송선 검사에서의 결함 검출 성능을 향상시키는 것을 목표로 합니다. AE-YOLO는 Feature Pyramid Network-Path Aggregation Network 아키텍처에 경량의 bottleneck autoencoder를 통합하여 다중 스케일 특성을 융합하는 동안 이상 감지 정보를 보존합니다.

- **Technical Details**: AE-YOLO는 Convolutional Block Attention Modules (CBAM)을 backbone에 적용하여 특성 분별력을 강화하고 배경 간섭을 억제합니다. 논문에서는 focal loss, Complete IoU (CIoU) loss 및 autoencoder 정규화를 통합한 통합 최적화 목표를 개발하여 전경-배경 불균형 문제를 해결하고 로컬라이제이션 정확도를 개선합니다. 또한, Weighted Boxes Fusion (WBF)와 autoencoder-guided 신뢰도 향상 메커니즘을 도입하여 드문 결함 카테고리에 대한 민감도를 높입니다.

- **Performance Highlights**: Insulator-Defect Detection 데이터셋에서 AE-YOLO는 EfficientNetV2 backbone을 사용할 때 0.5에서 95.10% mAP, 96.40% precision, 93.80% recall을 달성했습니다. 이러한 성능은 YOLO 계열의 가장 강력한 기준 모델보다 5.0 포인트 높은 mAP와 6.7 포인트 높은 recall을 달성하였으며, 이는 AE-YOLO의 효과와 적응 가능성을 확인시킵니다. 이 모델은 UAV 기반의 전송선 검사 및 결함 모니터링을 위한 실용적이고 확장 가능한 솔루션으로 자리 잡을 것으로 예상됩니다.



### GOPAgen: Motion-Aware and Efficient Agentic Long-Video Understanding with Structural Memory and Hierarchical Reasoning (https://arxiv.org/abs/2606.06532)
- **What's New**: 이 논문에서는 GOPAgen이라는 새로운 동영상 이해 방법을 제안합니다. 이 방법은 비디오 코덱을 동영상 이해 프레임워크에 통합하고, Groups of Pictures (GOP)를 기반으로 하는 세심하게 설계된 모션 에이전트를 학습하여 더욱 세밀한 동작 이해를 가능하게 합니다. 또한, 제안된 GOP 트리 추론 알고리즘은 동영상 코덱과 자연스럽게 정렬되어, 동영상 내의 세부적인 동작 정보를 이해하는 능력을 향상시킵니다.

- **Technical Details**: GOPAgen은 지역 모션 정보를 구조화된 페이지의 상세 캡션과 통합하는 구조적 메모리 메커니즘을 신중하게 설계하였으며, 코스-투-파인 (coarse-to-fine) 줌인 알고리즘을 통해 구조적 메모리를 최대한 활용할 수 있도록 합니다. 또한, 다양한 크기의 모션 벡터를 효율적으로 검색할 수 있도록 모션 벡터 데이터베이스를 통합하여, 비디오 이해의 정확성을 높입니다.

- **Performance Highlights**: 총체적으로, 제안된 방법은 MotionBench와 Egoschema를 포함한 다양한 동영상 이해 벤치마크에서 뛰어난 Video Question Answering (VQA) 성능을 달성하였습니다. 이는 제안된 프레임워크가 기존 기술들보다 뛰어난 성능을 보여주며, 동작 이해 능력의 획기적인 향상을 입증합니다.



### Applying Deep Learning for cockpit segmentation in the context of mixed reality (https://arxiv.org/abs/2606.06520)
Comments:
          XXV Congresso Brasileiro de Automática - CBA 2024

- **What's New**: 이 연구에서는 컴퓨터 비전(Computer Vision) 분야의 새로운 발전 가능성을 제안하고 있습니다. 특히 1인칭 시점(First-person view) 기술을 활용하여 가상 환경(Virtual environment)과 실제 세계의 객체를 실시간으로 통합하는 Mixed Reality 환경을 구현하고자 합니다. 사용자의 몰입(Iimmersion)을 극대화하기 위해, 가상 및 실제 이미지를 통합하기 위한 이미지 분할(Image segmentation) 기술이 개발되었습니다.

- **Technical Details**: 본 논문에서는 CAT793F 오프 하이웨이 트럭 시뮬레이터를 통해 사용자의 실제 이미지를 획득하고, 이를 인공지능(Artificial Intelligence)을 활용하여 분할하는 방법을 다루고 있습니다. Segmentation을 위해 'U-net' 및 'DeepLabV3+'와 같은 합성곱 신경망(Convolutional Neural Network) 아키텍처가 적용되었습니다. 이러한 기술은 가상의 객체와 실제 패턴 간의 경계를 명확히 하여 더욱 실제적인 가상 경험을 제공합니다.

- **Performance Highlights**: 연구 결과, 약 90%의 정확도(Accuracy)를 달성하는 성과를 보였습니다. 제안된 모델들은 이미지 분할에서 높은 효과를 나타내어, Mixed Reality 환경에서 사용자 경험을 향상시키는 데 기여할 것으로 기대됩니다. 따라서 본 연구는 실제와 가상의 세계를 통합하는 시스템 구현에 중요한 기초 자료를 제공하고 있습니다.



### Planning-aligned Token Compression for Long-Context Autonomous Driving (https://arxiv.org/abs/2606.07464)
Comments:
          9 pages

- **What's New**: COMPACT-VA는 planner-aligned working memory 프레임워크로, 균형 잡힌 메모리 효율성을 통해 운전 성능을 최적화합니다. 이 연구는 의사결정에 중요한 정보를 유지하기 위해 과거 궤적과 학습된 계획 의도를 기반으로 하는 압축 메커니즘을 도입했습니다. 현재의 token 압축 방법들이 계획 목표와는 분리되어 있다는 문제점을 해결하기 위한 접근을 제공합니다.

- **Technical Details**: 이 모델은 conditional VQ-VAE(조건부 벡터 양자화 변분 오토인코더)를 기반으로 하며, 절차적 최적화를 통해 중요한 과거 정보를 어떻게 보존할지를 학습합니다. 압축된 메모리는 최근의 토큰에는 더 많은 비율을 주고, 더 먼 과거에 대해서는 적은 비율로 배치됩니다. 이 구조는 self-attention을 통해 과거 궤적 컨텍스트와 미래 운전 의도에 따라 조정이 가능합니다.

- **Performance Highlights**: COMPACT-VA는 일정한 token 예산 하에 68.3%의 성공률을 달성하며, 기존 모델보다 6.3% 향상된 결과를 보입니다. 안전에 중요한 roll-through를 22% 줄였으며, 모델의 전반적인 운전 능력은 유지하면서 3.3배의 속도 향상과 2.7배의 메모리 절감을 이끌어냈습니다. 각 세부 요소들이 성능 향상에 기여했다는 것은 ablation 테스트를 통해 확인되었습니다.



### Impact of Synthetic Lesional MR Images in Automated Focal Cortical Dysplasia Detection in Low-Data Scenarios (https://arxiv.org/abs/2606.07381)
- **What's New**: 이 연구에서는 Focal Cortical Dysplasia (FCD)의 자동 탐지를 위한 합성 MRI 데이터 생성 방법을 제안합니다. 이를 통해 수동 주석의 필요성을 줄이려는 목표를 가지고 있으며, 합성 데이터의 현실성을 평가합니다. 131명의 FCD 환자와 90명의 건강한 대조군으로부터의 MRI 데이터를 사용하여 연구하였습니다.

- **Technical Details**: T1 가중화 (T1-weighted, T1w) 및 T2 가중화 Fluid-Attenuated Inversion Recovery (FLAIR) MRI 스캔을 통한 데이터 수집이 이루어졌습니다. Binary FCD 마스크를 조건화하여 생성적 네트워크를 활용하여 합성 MRI를 생성하였으며, 두 명의 신경영상의사가 실제 이미지와 합성 이미지를 구별했습니다. 세 가지 nnU-Net 모델을 훈련시켜 FCD 탐지를 수행했습니다.

- **Performance Highlights**: 전문가들은 실제 이미지와 합성 이미지를 분별하는 데 제한된 능력을 보였으며, T1w의 분류 정확도는 60%, FLAIR는 70%였습니다. 합성 데이터를 사용한 자동 FCD 탐지는 민감도를 8.14% 증가시켰고, 확대된 실제 데이터 모델은 민감도를 73.8%까지 개선시켰습니다. 합성 데이터 기반 augmentation은 라벨된 데이터 요구를 약 20% 줄이면서도 동등한 민감도를 유지하는 것으로 나타났습니다.



### Beyond Backscatter: InSAR coherence from detected SAR images (https://arxiv.org/abs/2606.07374)
Comments:
          27 pages, 20 figures

- **What's New**: 이 연구에서는 고전적인 인터페로메트릭 처리에서 필요한 정밀한 코어지스트레이션 없이 감지된 SAR 이미지에서 일관성을 회귀하는 딥 러닝 프레임워크를 제안합니다. Residual U-Net 모델은 정확히 코어지스트레이트된 Sentinel-1 SLC 데이터에서 유도된 일관성 맵을 학습하여 백스캐터(Backscatter) 강도와 일관성의 관계를 이해합니다. 이 방법은 다양한 데이터셋에서 평가되었으며, 높은 해상도의 일관성 회귀와 기존 강도 기반 접근 방식에 비해 향상된 정확성을 보여주었습니다.

- **Technical Details**: 제안된 방법론은 SAR 백스캐터 쌍으로부터 직접 인터페로메트릭 일관성을 회귀하며, 방사선 보정이 이루어진 백스캐터 정보에만 의존하고 정밀한 서브픽셀(Sublpixel) 인터페로메트릭 코어지스트레이션이 필요하지 않습니다. 네트워크는 오직 감지된 백스캐터만을 입력으로 받아 일관성 행동과 관련된 통계적 및 구조적 단서를 학습합니다. 이 방법은 데이터 주도(coherence regression) 일관성 회귀를 가능하게 하며, 학습 과정에서 코어지스트레이트된 SLC 데이터를 감독으로 사용합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 다양한 지리적 지역과 주파수에서의 고해상도 일관성 회귀를 효과적으로 수행하며, 기존 접근 방식보다 뛰어난 일반화 성능을 입증하였습니다. 이를 통해 무작위로 나타나는 다양한 시간 기반, 지역 및 폴라리제이션에 대해 잘 일반화되며, Google Earth Engine과 같은 글로벌 데이터 소스에서도 적용이 가능합니다. 이로 인해 대규모 임무 설계와 변화 모니터링, 그리고 다양한 매핑 작업에서 널리 활용될 수 있는 잠재력을 보여줍니다.



### Closed-Form Spectral Regularization for Multi-Task Model Merging (https://arxiv.org/abs/2606.07289)
- **What's New**: 이 논문은 여러 개의 독립적으로 세분화된 전문가 모델들을 단일 다중 작업(multi-task) 모델로 결합하는 방식에 대해 다루고 있습니다. 기존의 최첨단 병합(methods)은 층별(quadratic) 간섭 최소화 문제로 병합을 공식화하지만, 본 연구에서는 반복적으로 수행하는 솔버가 실제로는 최적화기가 아닌 희소한(normal equation) 정규화 역할을 한다고 제시합니다.

- **Technical Details**: 저자들은 다중 작업 모델 병합을 소음이 많은 선형 역문제(noisy linear inverse problem)로 공식화하고, 각 방향 필터로 매개변수화된 스펙트럴 필터링 추정치(spectral filtering estimator)를 제안합니다. SWUDI라는 닫힌 형태(closed-form) 방법을 통해 드롭아웃 전략과 같은 필터링 기법을 사용해 소음을 억제하는 방식도 설명합니다.

- **Performance Highlights**: 제안된 스펙트럴 솔버는 네 가지 일반 벤치마크와 VQA, Geometry, Chart, OCR, Grounding 등 여러 모달리티 병합을 포함한 벤치마크에서 기존의 최첨단 병합 방법들에 비해 동등하거나 더 나은 성능을 보였습니다. 또한, 벽시계 시간(wall-clock time)을 28-72배, 최대 GPU 메모리를 50% 줄이는 성과를 달성했습니다.



### Beyond Waypoints: A Trajectory-Centric Waypointing Paradigm for Vision-Language Navigation (https://arxiv.org/abs/2606.07244)
- **What's New**: 이 연구에서는 기존의 노드 중심 waypoint 예측을 뛰어넘어 Trajectory Waypoint라는 새로운 패러다임을 제안합니다. 이는 각 후보 waypoint를 실행 가능한 궤적에 기반하여 생성하여, 물리적으로 도달 가능한 목표를 보장합니다. 또한, 이 방법은 고수준 계획과 저수준 실행 간의 일관성을 극대화하는데 기여합니다.

- **Technical Details**: Trajectory Waypoint Predictor는 Diffusion policy로 형성되어 있으며, TSDF 기반의 비용 유도(guidance)를 통해 오프젝트를 피하면서 궤적 생성을 조정합니다. 이로 인해 더욱 안전하고 다양한 항해 가능 궤적을 생성할 수 있고, 이를 통해 에이전트의 탐색 능력을 향상시킵니다. 또한, 궤적 기반 내비게이터는 고급 기하학적 정보와 관련된 궤적을 플래닝 정보로 활용하여 보다 정밀한 방향성을 제공합니다.

- **Performance Highlights**: VLN-CE 벤치마크에서의 실험 결과, Trajectory Waypoint 패러다임이 기존의 waypoint 예측기보다 목표 도달 가능성에서 눈에 띄게 우수한 성능을 보임을 확인했습니다. 전체적으로, 이 프레임워크는 아래의 다양한 VLN-CE 작업에서 성능이 크게 향상되었습니다. 따라서, 기존의 기본선과 비교했을 때 안정적이고 강력한 개선을 나타냅니다.



### Robotic Policy Adaptation via Weight-Space Meta-Learning (https://arxiv.org/abs/2606.07217)
- **What's New**: WIZARD는 언어 지침 및 짧은 시연 비디오만을 사용하여 고정된 VLA 정책에 대한 태스크별 LoRA 파라미터를 생성하는 가중치 공간 메타 학습 프레임워크입니다. WIZARD는 개별 태스크의 액션 레이블이나 테스트 시간 최적화 없이 단 하나의 전파 과정으로 적응 가중치를 예측합니다. 이 방법은 태스크 증거를 전문가 LoRA 업데이트에 직접 매핑하며, 태스크 간의 관계를 가중치 공간에서 캡처합니다.

- **Technical Details**: WIZARD의 메타 네트워크는 함수형으로 구축되어 태스크 증거(z)에서 전문가 정책의 LoRA 업데이트(ΔW)로의 직접적인 변환을 수행합니다. 이는 전통적인 테스트 시간 적응과 다르며, 액션 공간에서 최적화하는 데 의존하지 않습니다. WIZARD는 학습된 태스크 증거에 바탕을 두어 가중치 공간에서 태스크 관계를 학습하면서, 고정된 VLA 정책에 특화된 태스크 별 적응 메커니즘을 제공합니다.

- **Performance Highlights**: LIBERO 데이터 세트에서 WIZARD는 아직 훈련되지 않은 데이터 세트에서 평균적으로 약 2배 그리고 새로운 태스크에서 최대 14배의 성능 향상을 보였습니다. 이는 사전 훈련된 VLA와 태스크별 전문가 간의 성능 차이를 줄이는 데 큰 기여를 하며, WIZARD로 생성된 어댑터가 시뮬레이션 이상의 태스크 수준 특성을 제공함을 보여줍니다.



### Beyond Universality: The GCC-FER Dataset and Culture-Aware Adaptation for Dynamic Facial Expression Recognition (https://arxiv.org/abs/2606.07063)
- **What's New**: 이번 연구는 문화적으로 다양한 동적 얼굴 표현 인식(Dynamic Facial Expression Recognition, DFER) 시스템을 위한 새로운 하이브리드 멀티컬처 비디오 데이터셋인 GCC-FER(Global Cross-Cultural Facial Expression Recognition)를 소개합니다. 이 데이터셋은 아프리카, 코카시안, 동아시아, 남아시아의 네 가지 문화 그룹에 걸쳐 23,934개의 비디오 샘플을 포함하고 있으며, 서로 다른 문화에서의 표정 인식 성능 향상을 목표로 하고 있습니다. GCC-FER는 기존의 패널과 다양한 문화적 맥락을 반영하지 않았던 데이터셋의 한계를 극복하기 위해 설계되었습니다.

- **Technical Details**: GCC-FER 데이터셋은 동적 얼굴 표정 인식을 위한 대규모 멀티컬처 비디오 데이터셋으로, 각 문화 그룹에 대해 행동적으로 기반한 문화적 선행 정보(cultural priors)를 도출합니다. 연구팀은 720p로 촬영된 비디오 세그먼트를 포함하여 얼굴 감지 및 정렬을 진행하고, 각 비디오에서 두 사람 이하의 표정만을 포함했습니다. 데이터셋의 정합성을 위해 훈련된 어노테이터들이 자동화된 인종 추정 프레임워크와 함께 세 단계의 필터링 절차를 통해 각 비디오 샘플에 문화적 레이블을 부여했습니다.

- **Performance Highlights**: GCC-FER 데이터셋을 활용하여 제안된 문화 인식 얼굴 표정 인식 시스템(Culture-Aware FER, CA-FER)이 다양한 문화적 환경에서 얼굴 표정 인식 성능을 일관되게 향상시키는 것을 입증합니다. 실험 결과, CA-FER 시스템은 다문화 상황에서 인식 성능을 개선하며, 이는 문화적 편향을 완화하기 위한 효과적인 기제로 작용합니다. 또한, GCC-FER와 DFEW에 대한 포괄적인 실험을 통해 이 시스템의 신뢰성과 유용성이 강조되었습니다.



### Constructing VAE Latent Spaces with Prescribed Topology (https://arxiv.org/abs/2606.07058)
Comments:
          16 pages, 7 figures

- **What's New**: 본 논문에서는 비유클리드(Non-Euclidean) 구조를 가진 데이터에서 표준 Gaussian prior가 유발하는 위상(topology) 불일치를 해결하기 위한 수학적 프레임워크를 제시합니다. 이 프레임워크는 순수한 곱 커버링 공간(product covering space)을 허용하는 모든 다양체(manifold)에 대해 적용됩니다. 특별히, 원형(circles), 구간(intervals), 선(lines) 등의 기본 요소로 구성된 다양체를 포함하며, 이를 통해 VAEs의 설계를 보편화합니다.

- **Technical Details**: 이 연구는 곱(topology) 생성와 편리한 KL divergence (Kullback-Leibler divergence) 계산을 위한 재파라미터화 가능한 인코더-프라이어 쌍의 카탈로그를 제공합니다. 지정된 위상에 대해 KL 정규화가 데이터 다양체와 일치하도록 조정되며, 각 잠재 요소가 독립적으로 형성되도록 합니다. 각 요소마다 인코더-프라이어 쌍을 선택하고, 연속적인 그래디언트를 출력할 수 있도록 표준 신경망을 위한 좌표 변환을 수행합니다.

- **Performance Highlights**: 인공적으로 생성된 다양체와 실제 이미지 데이터셋(MNIST의 회전 및 순환 이동)에 대한 실험 결과, 위상이 일치하는 프라이어가 KL 정규화와 데이터 다양체를 연결하는 데 효과적임을 입증했습니다. 이러한 위상 인식 모델(topology-aware models)은 모든 실제적인 정규화 강도에서 Gaussian 기준선을 초과하는 성능을 보여줍니다. 그러므로 본 프레임워크는 비유클리드 잠재 공간 학습의 새로운 기준점을 제시합니다.



### Hierarchical Semantic-Constrained Heterogeneous Graph for Audio-Visual Event Localization (https://arxiv.org/abs/2606.07033)
- **What's New**: 이 논문은 Open-vocabulary audio-visual event localization (OV-AVEL)을 위한 계층적 의미 제약 이질 그래프(HSCHG)를 제안합니다. 이 방법은 오디오 및 비디오 세그먼트 노드와 해당 비디오 수준 노드 간의 이질적인 계층적 그래프를 구성하여, 여러 시간 스케일에서 오디오-비주얼 일관성을 유지하는 문제를 해결하려고 합니다. 특히, 이 연구는 하이퍼볼릭 공간(hyperbolic space)으로 멀티 레벨 오디오-비주얼 표현과 텍스트 프로토타입을 매핑하는 과정을 포함합니다.

- **Technical Details**: 제안된 HSCHG는 오디오-비주얼 일관성을 보장하기 위해 서로 다른 레벨의 시맨틱 정보를 통합하는 특징을 가지고 있습니다. 논문에서는 멀티 방향 시간 경로와 게이티드 퓨전 전략을 사용하며, 정밀한 시간 정보 캡처를 위한 역방향 및 정방향 엣지를 도입합니다. 이 방법은 또한 각 레벨 간의 의미적 일관성을 위한 양방향 의미 제약을 설정하여, 세그먼트와 비디오 표현 간의 관계를 동적으로 정교화합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 OV-AVEL 벤치마크에서 기존 방법보다 뛰어난 성능을 나타냈습니다. 변화 생성 연구(ablation studies)를 통해 제안한 모델의 효과성과 각 구성 요소의 유용성이 추가적으로 입증되었습니다. 이 연구는 오픈 어휘 환경에서 시맨틱 일관성을 구축하는 데 있어 중요한 기여를 합니다.



### An Integrated Roadside Sensing and Communication Framework for Vulnerable Road User Safety at Signalized Intersections (https://arxiv.org/abs/2606.07016)
Comments:
          17 pages, 5 figures, 2 tables. Preprint

- **What's New**: 이 논문은 신호등이 있는 교차로에서 취약한 도로 사용자(Vulnerable Road Users, VRUs)의 보호를 위한 통합된 프레임워크를 제안합니다. LiDAR, 레이더, RGB 카메라, 열 화상 카메라를 결합하여 다중 감지 기술을 사용하고, edge-based 예측과 대체 안전 분석을 통해 안전성을 높입니다. 이는 기존 시스템과는 달리, 모든 요소가 동일한 교차로에 통합된 형태로 제공됩니다.

- **Technical Details**: 이 프레임워크는 감지 계층에서 LiDAR, 레이더, RGB 및 열 화상 카메라를 사용하고, 컴퓨테이션 계층에서는 edge-based 예측 및 서레깅( surrogate) 안전 분석을 통해 데이터를 처리합니다. 또한, 통신 계층에서는 V2X(vehicle-to-everything) 및 P2X(pedestrian-to-everything) 메시징을 강조하며, 작동 계층에서는 적응형 신호 제어(adaptive signal control)를 통해 교차로의 효율성을 높입니다. 이 연구는 R-LiViT라는 첫 공개된 도로변 LiDAR-Visual-Thermal 데이터 세트를 기반으로 하며, 독일의 세 개 교차로에서 수집된 다양한 다중 모드 시퀀스를 포함합니다.

- **Performance Highlights**: 연구 결과, VRUs는 도로 사용자 관찰의 약 49%를 차지하며, 낮과 밤의 밀도는 보행자는 38%, 차량은 45% 감소했습니다. 특히 교차로의 8개 독특한 위치에서 프레임 당 근접 이벤트 수가 약 10배 차이가 나는 것으로 나타났습니다. 또한, 보행자의 경계 상자 중 83%는 이미지 공간에서 상대적으로 작아서 VRUs가 일반적으로 단일 센서와의 거리가 멀다는 것을 시사합니다.



### DaX: Learning General Pathology Representations Across Scales (https://arxiv.org/abs/2606.06983)
- **What's New**: 이번 논문에서는 다양한 임상 지표에 걸쳐 전이될 수 있는 시각적 표현을 요구하는 컴퓨터 병리학(Computational Pathology)을 위해 DaX라는 새로운 병리학 비전 기초 모델을 소개합니다. DaX는 전통적인 DINOv3 스타일의 self-supervised learning을 전체 슬라이드 히스토병리학에 적응할 수 있도록 설계되었습니다. 이 모델은 자연 이미지 DINOv3 가중치로 초기화되며, 다양한 훈련 기법을 포함하고 있습니다.

- **Technical Details**: DaX는 지속적인 배율 훈련(continuous magnification training), 크로스 스케일 조직 뷰(cross-scale tissue views), 방향 무관(orientation-agnostic) 및 촬영 강건성(acquisition-robust) 증강(augmentation), 다양한 입력 크기(multi-input-size) 훈련, 그리고 Gram-anchored dense consistency를 통합합니다. 이러한 설계들은 국소 세포 형태(local cellular morphology)와 전반적인 조직 구조(global tissue architecture)를 연결하고, 입력 스케일 전반에 걸쳐 조밀한 토큰 수준 표현(dense token-level representations)을 안정화하기 위해 마련되었습니다.

- **Performance Highlights**: 제안된 모델은 44개의 공개 데이터셋에서 161개의 임상적으로 중요한 과제를 포함한 WSI 수준 벤치마크를 통해 평가되었습니다. DaX는 모든 임상 영역과 아홉 개의 작업 범주를 커버하며, 고정된 환자 수준 크로스 검증 프로토콜 아래에서 평가되었습니다. 이 결과 DaX는 각 과제에서 평균적으로 가장 높은 성능을 기록하였고, 진단 병리학, 바이오마커 및 분자 프로파일링 등 다양한 영역에서 안정적인 작업 수준 순위를 달성하였습니다.



### ActionMap: Robot Policy Learning via Voxel Action Heatmap (https://arxiv.org/abs/2606.06904)
- **What's New**: 최근 Vision-language-action (VLA) 모델 분야에서 ActionMap을 도입하였습니다. 이는 기존 VLA의 액션 디코더를 대체할 수 있는 voxel heatmap action head로, 각 액션에 대해 액션 공간의 voxel heatmap을 예측합니다. 이러한 접근은 인접한 액션 간의 기하학적 근접성을 활용하여 데이터 효율성과 수렴 속도를 향상시킬 수 있는 가능성을 제공합니다.

- **Technical Details**: ActionMap은 각 액션에 대해 3D 번역, 3D 회전 및 그리퍼를 포함한 세 가지 voxel heatmap을 예측하며, 각 voxel은 해당 액션의 확률을 저장합니다. 이러한 heatmap은 교차 엔트로피(cross-entropy)를 사용하여 중심이 실제 액션인 Gaussian blob에 대해 훈련됩니다. 추론 시 각 heatmap에서 연속 액션을 하드 argmax(hard argmax) 또는 top-kk soft argmax를 통해 회복할 수 있습니다.

- **Performance Highlights**: LIBERO 시뮬레이션 및 실제 Franka 조작 실험에서, 우리의 heatmap 헤드는 OpenVLA-OFT의 L1 회귀 헤드보다 평균 8.2% 향상되었습니다. 또한, 낮은 훈련 데이터에서 데이터 효율성이 크게 증가하였으며, 수렴 속도에서도 일관되게 더 빠른 성능을 보였습니다. 이러한 결과는 액션 표현이 VLA 성능의 중요한 레버라는 것을 보여줍니다.



### A Cross-view Fusion Framework for Robust 6-DoF Grasp Pose Estimation (https://arxiv.org/abs/2606.06878)
Comments:
          Corresponding author: Jin Xie

- **What's New**: 이 논문에서는 코너 뷰에서 6-DoF 그립 포즈 추정을 향상시키기 위한 크로스 뷰 융합 프레임워크를 제안합니다. 우리는 보조 뷰를 포함시킴으로써 가리낌(occlusion) 문제를 완화하고, 시간을 많이 소모하는 멀티 뷰 재구성 과정을 피하는 포스트 융합(post-fusion) 전략을 채택하였습니다. 이 방법은 그립에 관련된 지오메트리를 포괄적으로 표현하기 위해 크로스 뷰 포인트 쌍을 이용한 자기 지도 대비 학습(self-supervised contrastive learning) 전략을 사용합니다.

- **Technical Details**: 제안된 프레임워크는 크로스 뷰 점 인식 향상을 위해 세 가지 주요 컴포넌트를 포함합니다. 첫째, 보조 뷰를 활용하여 기하학적 정보를 보강하고, 아울러 원통형 좌표계에서 포인트를 정렬하여 잡기의 방향성을 강조합니다. 둘째, 자기 지도 대비 학습 전략을 통해 포인트 특성을 정규화하여, 공간적 일관성을 유지하고 방향적인 구별성을 강화합니다. 마지막으로, 로컬 셀프 어텐션과 시드 크로스 어텐션을 차례로 사용하여, 단일 뷰 내에서의 상호작용과 뷰 간의 상호작용을 지원하여 정밀한 잡기 형상을 생성합니다.

- **Performance Highlights**: 본 프레임워크는 GraspNet-1Billion 벤치마크와 실제 응용에서 강력한 성능을 보여줍니다. 특히, 보조 뷰를 사용하여 코너 뷰의 가리낌 문제를 해결함으로써 6-DoF 그립 포즈 추정의 정확성을 높였습니다. 실시간 로봇 작업 흐름에서 높은 해상도의 포인트 표현을 유지하면서 기하학적 세부정보의 손실을 방지하여, 보다 견고한 잡기 성능을 달성하였습니다.



### Physics-Driven Semantic Scattering Structure Understanding of Aircraft Target in SAR Images (https://arxiv.org/abs/2606.06847)
- **What's New**: 이 논문은 SAR 항공기 해석을 위한 새로운 패러다임인 'Semantic Scattering Structure Understanding'을 제안합니다. 이 방법은 로컬 전자기 응답과 물리적으로 의미 있는 항공기 구성 요소를 연결하는 'semantic scattering keypoints'을 정의합니다. 또한, 이 논문에서는 물리적 존재가 약하게 관찰될 수 있음에도 불구하고 이를 유지할 수 있도록 'visibility-aware attributes'을 도입합니다.

- **Technical Details**: S³U-SAR는 전자기 스캐터링 정보를 활용하여 각 항공기 구성 요소에 대한 의미 있는 키포인트를 식별하고 안정된 구조를 형성합니다. 우선, 전자기 스캐터링 응답과 물리적 구성 요소 간의 관계를 명시적으로 인코딩하는 'semantic scattering keypoints'을 수립합니다. 물리적 제약에 의해 구속된 구조적 토폴로지를 도입하여 키포인트 사이의 기계적인 관계를 모델링하고, 스캐터링 불균일성, 강체 토폴로지, 스펙클 불확실성 등의 멀티 차원 물리적 사전 정보를 제약으로 활용합니다.

- **Performance Highlights**: KP-SAR-Aircraft-1.0 벤치마크를 통해 S³U-SAR의 성능을 비교한 결과, 기존의 방법들과 비교하여 가장 우수한 성능을 기록했습니다. 다양한 카테고리와 데이터셋에서의 평가를 통해 방법의 강건함과 전이 가능성을 입증하였습니다. 특히, 새로운 방향 추정 패러다임이 개발되어 17%의 성능 향상을 달성하였습니다.



### Think Like a Pilot: Fine-Grained Long-Horizon UAV Navigation (https://arxiv.org/abs/2606.06836)
- **What's New**: 이번 논문에서는 UAV(무인항공기) 에이전트가 긴 지평선의 의미 기반 지시를 실행하고 부드럽고 물리적으로 실행 가능한 연속 비행 명령을 생성할 수 있도록 하는 새로운 벤치마크인 FLIGHT를 소개합니다. 기존의 Vision-Language Navigation (VLN) 벤치마크는 일반적으로 이산적 또는 조잡한 행동을 사용하며, UAV Vision-Language-Action (VLA) 작업은 짧고 원자적인 조작에 집중하고 있었습니다. FLIGHT는 세밀한 VLN과 긴 지평선 흐름으로 나누어진 두 개의 데이터셋에서 다단계 지침과 밀집된 6-DoF 궤적 주석을 결합하고 있습니다.

- **Technical Details**: FLIGHT 시스템은 UAV 에이전트가 과제 실행 상태와 미션 계획에 대한 실시간 비행 중 추론을 가능하게 하며, 동시에 고주파의 정확한 제어를 수용하기 위해 FLIGHT VLA라는 비동기 아키텍처를 제안합니다. 이 구조는 과제 상태 추론을 위한 저주파 Streaming Pilot Vision-Language Model (VLM)와 연속 제어를 위한 고주파 확산 행동 모델을 분리하여 멀티 태스크를 수행합니다. 또한 현재 비행 상태를 요약하고 다음 하위 목표를 예상하는 명확한 Pilot Reasoning 텍스트로 감독됩니다.

- **Performance Highlights**: 실시간 평가에서 FLIGHT VLA는 FLIGHT 벤치마크의 대표적인 VLN 및 VLA 기준을 지속적으로 초과 달성하며, 다단계 완료, 하위 목표 준수 및 터미널 제어에서 우수한 결과를 보여주었습니다. 학습된 Streaming Pilot Reasoning VLM은 UAV 비디오 추론을 더욱 향상시켜, 설계의 효과성을 입증합니다.



### Compute-Optimal Network Design for Echocardiography Myocardial Segmentation and Perfusion Quantification using Neural Scaling Laws (https://arxiv.org/abs/2606.06725)
Comments:
          15 pages, 4 figures, 5 tables, journal

- **What's New**: 이 연구는 대조 강화 초음파(CEUS)를 이용한 심근 관류 정량화에 대한 새로운 접근 방식을 제시합니다. 특히, 수동 레이블링의 시간 소모를 해결하기 위해 신경 스케일링 법칙(neural scaling laws)을 적용하여 자동화된 세분화를 수행합니다. CAMUS 에코카르디오그래피 데이터셋과 25명의 환자에 대한 CEUS 데이터셋에서 최적의 네트워크 크기를 결정하는 방법을 개발했습니다.

- **Technical Details**: CEUS는 미세기포(microbubbles)를 사용하여 혈관 이미징을 수행하는 비 방사선 기술입니다. 수동 세분화가 시간 소모적이고 오류의 가능성이 있는 반면, 자동 세분화는 여러 심근 영역을 동시에 분석할 수 있게 합니다. 신경 스케일링 분석은 테스트 성능, 데이터셋 크기 및 모델 크기 간의 관계를 규명하여 최적의 모델 설계를 가능하게 합니다.

- **Performance Highlights**: 연구 결과, 자동으로 세분화된 마스크가 심근 관류 정량화에서 수석 심장 전문의와 동등한 성능을 보이는 것으로 나타났습니다. 신경 스케일링 법칙에 기반하여 두 개의 네트워크가 CAMUS에서 최신 성능을 얻으면서 파라미터 수를 240배 줄일 수 있었습니다. 이러한 결과는 소규모 이미징 데이터셋에 대한 데이터 기반 최적 모델 설계의 유용성을 입증합니다.



### What Matters When Cotraining Robot Manipulation Policies on Everyday Human Videos? (https://arxiv.org/abs/2606.06627)
Comments:
          The project website is here: this https URL

- **What's New**: 이번 연구에서는 532개의 인간 동영상과 28시간의 고품질 3D 손 레이블을 포함한 새로운 데이터셋을 사용하여 로봇 정책을 공동 훈련(cotraining) 하는 방법을 탐구합니다. 인터넷 동영상을 통해 인간의 자연스러운 동작을 활용할 수 있는 가능성을 제시하며, 손 자세의 품질이 로봇으로의 전이에 영향을 미친다는 것을 발견했습니다. 연구진은 고품질 손 레이블이 있는 데이터셋을 통해 로봇이 더 성공적으로 학습하도록 하는 방법을 제안하고, 전이의 주요 요인으로 3D 손 자세의 질과 자연 인간 동작의 차이를 밝혔습니다.

- **Technical Details**: 연구진은 EgoExo4D 데이터셋을 기반으로 532개의 요리 동영상에서 고품질 3D 손을 다중 뷰 삼각 측량(multi-view triangulation)하여 새로운 데이터셋을 구축했습니다. 이 과정에서 2D 키포인트의 정확성을 향상시키기 위해 모델 기반 포즈 추정기를 활용하고, 손 자세 추정기를 재조정(retraining)하여 기존의 2D 추정기보다 더 나은 결과를 얻었습니다. 또한, 미니배치(mini-batch) 훈련 시 손 자세의 품질과 전이에 대한 주요 이슈를 식별하여, 기술적 접근 방식으로 비율 정렬(scale alignment)과 토큰 단위 융합(token-level fusion)을 제안하였습니다.

- **Performance Highlights**: 연구는 532개의 인간 동영상과 3,000개의 로봇 시연으로 훈련하여 6가지 실제 조작 작업에 대해 평가를 실시했습니다. 공동 훈련 방법은 로봇 데이터가 적은 환경에서 +29.7%의 절대 성과 향상을 보였으며, 특정 작업에 맞춘 로봇 데이터가 늘어날수록 성과는 계속 향상되었습니다. 이는 연구진이 제안한 데이터셋이 기존의 대규모 실험실 데이터셋보다 로봇 전이 성과가 우수함을 입증하는 결과입니다.



### ErA: Error-Aware Deep Unrolling Network for Single Image Defocus Deblurring (https://arxiv.org/abs/2606.06540)
- **What's New**: 이번 연구에서는 ErA (Error-Aware Deep Unrolling Network)를 소개합니다. 이 네트워크는 단일 이미지에서의 초점 흐림 복원을 위해 구축된 엔드 투 엔드(end-to-end) 프레임워크입니다. ErA는 컴팩트한 커널 기저(kernel basis)와 픽셀별 가중치를 동시에 학습하며, Augmented Lagrangian unrolling의 에러 감지 오류 보정(term)으로 커널 추정 오류를 수정합니다.

- **Technical Details**: ErA는 PSF(point spread function)를 명시적으로 예측하고 에러 감지 정규화(error-aware regularization)를 Augmented Lagrangian unrolling 스킴에 포함시킵니다. 이 하이브리드 디자인은 임의의 비가우시안(Non-Gaussian) 블러를 처리하면서 커널 추정 오류를 보상할 수 있도록 합니다. 실험은 DPDD, RealDOF, RTF 벤치마크에서 최첨단 결과를 도출하며, CUHK에서는 지상 진리(ground truth)가 없음에도 강한 일반화를 보여줍니다.

- **Performance Highlights**: ErA는 DPDD 및 RealDOF에서 가장 높은 PSNR과 SSIM을 달성하였으며, RTF에 대해서도 경쟁력 있는 성능을 보였습니다. 구조적으로 신뢰할 수 있는 복원 결과를 도출하여, 텍스트, 섬세한 에지 및 미세 구조를 보다 선명하게 복원할 수 있었습니다. 최종 실험 결과는 ErA가 더 적은 훈련 샘플로도 최첨단 방법을 초과하는 성과를 내어 우수한 일반화 능력을 보여줌을 입증했습니다.



### DSU-Net: An Attention-Enhanced Dense Skip U-Net for Breast Lesion Segmentation in Mammographic Images (https://arxiv.org/abs/2606.06537)
- **What's New**: 이 연구에서는 유방암 조기 발견을 위한 DSU-Net이라는 주의 기반(Attention-enhanced) Dense Skip U-Net 구조를 제안합니다. 이 모델은 유방영상에서 병변(segmentation) 식별을 자동화하는데 중점을 두고 있으며, 정확한 병변 경계 측정과 공간 정보 보존을 위해 밀집 스킵 연결(dense skip connections) 및 주의 메커니즘(attention mechanisms)을 통합합니다.

- **Technical Details**: DSU-Net은 디지털 유방 스크리닝 데이터베이스(CBIS-DDSM)를 사용하여 실험했습니다. 학습 과정에서 Dice loss, focal loss, binary cross-entropy loss를 결합한 복합 손실 함수(composite loss function)를 통해 심각한 전경(foreground)-배경(background) 불균형 문제를 해결했습니다.

- **Performance Highlights**: 모델은 검증 데이터셋에서 0.9421의 Dice 유사도 계수(Dice Similarity Coefficient), 0.8905의 교차 영역 비율(Intersection over Union), 0.9711의 정확도(accuracy), 0.9878의 AUC-ROC 성능을 달성했습니다. 또한, 다양한 크기와 형태를 가진 병변을 정확하게 구분할 수 있으며, 이로 인해 DSU-Net이 유방영상에서 신뢰할 수 있는 자동 병변 세분화 모델임을 입증했습니다.



### Advanced Flood Prediction with Physics-Guided Deep Learning: Combining UNet, FNO, and SAR/Optical Imagery (https://arxiv.org/abs/2606.06524)
Comments:
          This paper has been accepted for publication in the Proceedings of the IEEE Radar Conference (RadarConf 2026). The final authenticated version will be available through IEEE Xplore

- **What's New**: 본 연구는 물리 기반의 심층 학습 프레임워크를 통해 다중 모드 원격 감지 데이터(Sentinel-1 SAR, Sentinel-2 광학 이미지, DEM 유래 지형 특성)를 결합하고 얕은 물의 방정식(Shallow Water Equations, SWE)의 제약 조건을 통합합니다. 제안된 혼합 아키텍처는 미세한 공간 세부정보를 포착하기 위해 UNet을 사용하고, 유역 규모의 수리적 상호작용을 모델링하기 위해 Fourier Neural Operator (FNO)를 사용합니다. 이 접근법은 물리적 일관성을 유지하며 데이터를 기반으로 하는 추정에서 발생할 수 있는 오류를 보완합니다.

- **Technical Details**: 이 프레임워크는 다중 모드 원격 감지 입력을 통합하여 물리적 제약을 포함한 정확한 홍수 예측을 가능하게 합니다. UNet과 FNO는 지역적인 기능 추출과 전역적인 패턴 모델링을 각각 수행하며, 이들의 융합을 통해 고해상도 홍수 경계와 대규모 수리학적 상호작용을 동시에 포착합니다. 최종 출력은 UNet에서 온 미세한 세부정보와 FNO의 전역 컨텍스트를 통합하여 연속적인 수리적 상태 변수를 예측합니다.

- **Performance Highlights**: 혼합 모델은 다양한 홍수 평야 환경에서 테스트되었으며, 홍수 범위 예측에서 0.82의 Intersection over Union(IOU)과 0.90의 F1 점수를 기록하여 UNet과 FNO 전용 모델을 능가했습니다. 수리 시뮬레이션을 참고 데이터로 사용하여, 물 깊이와 유속에 대해 각각 0.21m와 0.15m/s의 RMSE를 달성했습니다. 물리적 일관성을 유지하며, 잔여물과 질량 불균형이 2.1% 이하로 낮게 유지되었습니다.



### A Geometric Gaussian Mixture Representation of Plane Curves (https://arxiv.org/abs/2606.06505)
- **What's New**: 본 논문에서는 사용자가 정의한 확률적 다각형 표현(user defined probabilistic polygonal representation)을 도입하여 평면 곡선을 표현합니다. 이 표현은 곡선상의 정점을 선택하고 이를 선분으로 연결하여 얻어진 다각형 근사를 통해 생성됩니다. 각 선분은 사용자 정의 불확실성 매개변수(uncertainty parameter)가 부여되어, 기존의 결정론적 곡선 표현과 차별화된 확률적 기하학적 원시(primitives)들을 생성합니다.

- **Technical Details**: 각 선분에 대해, 우리는 선분의 접선(tangent) 방향에서 균일하게 분포하고, 법선(normal) 방향에서 가우시안 분포가 있는 랜덤 변수(Random Variable)를 정의합니다. 이 구조는 선분의 중간 지점에서 평균(mean)이 위치하고, 공분산(covariance)으로 접선과 법선의 불확실성을 캡쳐하는 가우시안 확률 밀도 함수(Gaussian Probability Density Function)를 생성합니다. 블렌딩된 가우시안 구성 요소들이 적절한 가중치와 결합되어, 평면 곡선의 가우시안 혼합 모델(Gaussian Mixture Model) 표현이 생성됩니다.

- **Performance Highlights**: 이 프레임워크는 국소 기하학(local geometry)과 법선 방향에서의 불확실성을 유지하면서도, 평면 곡선의 매끄러운, 폐곡선, 열린 곡선, 비정규 및 자기 교차형 곡선에 적용될 수 있습니다. 실험을 통해 우리는 생성된 GMM이 국소 접선, 법선, 그리고 아크 길이(local arc length)를 잘 캡처하여, 기초 곡선의 전반적인 형태(global shape)가 정확하게 포착된다는 것을 보여주었습니다. 이 기법은 불확실성을 인식하는 CAD 및 디지털 트윈, 로봇에서의 확률적 장애물 모델링, 그리고 확률적 경로 계획 등 다양한 응용 분야에 사용될 수 있습니다.



### Semantic-Structural Alignment for Generative Pictorial Charts (https://arxiv.org/abs/2606.06498)
Comments:
          11 pages, 17 figures, Accepted to ACM TOG

- **What's New**: 이번 연구에서는 전통적인 통계 그래픽의 단점을 보완하기 위해 자동화된 pictorial 차트 생성을 위한 생성 프레임워크를 제안합니다. 새로운 접근 방식은 텍스트 프롬프트와 컨텍스트 이미지를 결합하여 세밀하고 구조적으로 일관된 시각 표현을 생성하는 데 중점을 두었습니다. 이 방법은 데이터의 수치적 정보를 보호하면서의 시각적 표현의 매력을 극대화하는 것이 목표입니다.

- **Technical Details**: 이 연구는 이중 조건 생성 프레임워크를 활용하여 통계 차트의 시맨틱(semantic) 표현과 구조적 충실성을 유지하는 방법을 제시합니다. 구조적 정렬(Structural Alignment)과 시맨틱 정렬(Semantic Alignment)의 두 가지 상호 보완적인 기능 레벨 메커니즘을 도입하여 그림 차트의 생성 과정에서 수치적 가독성을 보장합니다. 이를 통해 복잡한 시각적 스토리텔링을 위한 기반을 제공합니다.

- **Performance Highlights**: 정량평가와 사용자 연구를 통해 제안하는 프레임워크가 전통적인 이미지 편집 방법과 비교하여 우수한 성능을 보임을 입증했습니다. 이는 실질적으로 창의적인 표현과 통계적 가독성을 조화롭게 결합하는 데 성공했음을 보여줍니다. 따라서 이 연구는 시각적 스토리텔링의 데이터 기반 생성 모델링의 새로운 가능성을 제시합니다.



### Real-Time AttentionBender: Granular Interactive Network Bending of Video Diffusion Transformers (https://arxiv.org/abs/2606.06497)
Comments:
          In review. 5 pages, 4 figures

- **What's New**: 이 논문에서는 Generative video models의 시각적 충실성이 뛰어나지만, 프롬프트(prompt) 기반의 인터페이스가 예술가들에게 창의적인 작업의 제약을 가져올 수 있음을 지적합니다. 이에 대한 해결책으로, Real-Time AttentionBender라는 도구를 제안하여 비디오 디퓨전 변환기(video diffusion transformer, DiT)의 전 깊이에 걸쳐 네트워크 변형(network bending) 작업을 실시간으로 진행할 수 있게 합니다.

- **Technical Details**: 이 도구는 DayDream Scope 생태계 내 플러그인(plugin)으로 개발되었으며, 오픈 소스 오프라인 Wan 파이프라인(real-time Wan pipelines)을 감쌉니다. Real-Time AttentionBender는 자기 주의(self-attention), 교차 주의(cross-attention), 피드 포워드 네트워크(feed-forward network)를 독립적으로 조작할 수 있는 표면(surfaces)으로 노출하여, 개별 디퓨전 단계(diffusion steps), DiT 레이어(layers), 프롬프트 토큰(prompt tokens), 및 숨겨진 뉴런(hidden neurons) 단위로 조정할 수 있도록 지원합니다.

- **Performance Highlights**: 실시간 조작의 즉각성(immediacy)은 모델과의 '물질적 친밀성(material intimacy)'을 제공하며, 특정 레이어와 뉴런이 생성된 비디오에 어떻게 영향을 미치는지를 즉각적으로 느낄 수 있게 합니다. 이 도구는 변환기(transformer) 내부 구조에 대한 XAIxArts 탐구이자, 모델의 기본 표현 공간을 넘어 미학(aesthetics)을 발견하기 위한 표현 도구로 자리 잡고 있습니다.



New uploads on arXiv(cs.AI)

### How AI Agents Reshape Knowledge Work: Autonomy, Efficiency, and Scop (https://arxiv.org/abs/2606.07489)
- **What's New**: 본 논문에서는 대화형 보조기구에서 에이전트 오케스트레이션으로의 전환이 지식 노동에 미치는 경제적 영향을 실증적으로 분석합니다. 특히, AI 에이전트가 작업 흐름을 가속화하고 비용을 줄이며 품질을 향상시키고 있음을 보여줍니다. 이를 위해 Perplexity의 AI 제품을 분석하였으며, AI와 사용자 행동의 공동 진화가 AI의 활용 방식에 미치는 영향을 조명합니다.

- **Technical Details**: 본 연구는 Perplexity Search, Comet Assistant, Perplexity Computer 세 가지 제품을 비교하여 지식 작업의 수행 방식을 탐구합니다. 자율성(autonomy)과 문맥 통합(context integration)의 두 가지 차원에서 이러한 제품의 변화를 분석하였으며, 각 제품은 AI의 발전 정도를 나타냅니다. 특히, Computer 제품은 26분의 자율 작업을 수행하는 반면, Search 제품은 단 33초의 업무를 수행하게 됩니다.

- **Performance Highlights**: Computer는 사용자가 동일한 작업을 수행할 때 평균 269분이 소요되는 것을 36분으로 줄여 주며, 시간 및 비용을 각각 87% 및 94% 절감할 수 있음을 보였습니다. 또한 Computer는 더 복잡하고 다양한 작업을 수행하는 경향이 있으며, 이는 사용자들이 새로운 작업 기회를 얻을 수 있게 합니다. 이러한 결과는 사용자가 기존의 전문 분야를 넘어 다양한 작업을 시도하게 하여 조정 비용을 줄이는 데 기여합니다.



### Act As a Real Researcher: A Suite of Benchmarks Evaluating Frontier LLMs and Agentic Harnesses in Research Lifecyc (https://arxiv.org/abs/2606.07462)
- **What's New**: 본 연구에서는 AARR(Act As a Real Researcher) 벤치마크 시리즈를 개념화하여 AI 에이전트가 실제 연구자의 행동을 emulation 할 수 있는지를 평가하고 있습니다. 기존 벤치마크가 주로 작업 완료와 최종 결과를 측정하는 반면, AARR는 연구자의 전문성과 세심한 추론을 평가하는 데 중점을 두고 있습니다. 특히 AARRI-Bench(Act As a Real Research Intern)라는 첫 번째 벤치마크를 제안하여, AI 에이전트가 연구 인턴으로서의 역할을 얼마나 잘 수행할 수 있는지를 평가합니다.

- **Technical Details**: AARRI-Bench는 AI 연구에서 발생하는 다양한 일상적인 시나리오를 포괄하는 작업들로 구성되어 있습니다. 이 벤치마크는 AI 에이전트가 인간 연구자에게는 간단하지만 자율 에이전트에게는 상당한 도전이 되는 작업을 중심으로 설계되었습니다. 평가에는 Harbor 프레임워크가 사용되며, 이는 각 작업의 형식을 표준화하고 클린, 컨테이너화된 환경을 제공합니다.

- **Performance Highlights**: 실험 결과에 따르면, 최고의 성능을 보인 구성(Mini-SWE-Agent와 Claude Opus 4.7)은 오직 68.3%의 성공률을 기록했습니다. 이는 실제 인간 연구자에게는 명백한 중요한 세부 사항을 자주 간과한다는 것을 보여줍니다. 연구자로서 유사한 AI를 개발하기 위해서는 복잡한 구조적 지원보다는 연구 행동의 탐구가 더 필요하다는 것을 시사합니다.



### Online Pandora's Box for Contextual LLM Cascading (https://arxiv.org/abs/2606.07392)
- **What's New**: 본 논문에서는 LLM(대형 언어 모델) 캐스케이딩에 기반하여 온라인 컨텍스츄얼 판도라의 상자(online contextual Pandora's Box) 모델을 제안합니다. 이 모델은 적응적으로 LLM API를 쿼리하고 선택하는 과정에서의 결정 문제를 해결하는 데 초점을 맞추고 있습니다. 이전의 모델과는 달리, 본 연구는 API 별 출력과 비용의 분포를 직접적으로 추정하는 대신, 예약 지수를 모델링하고 학습하는 접근 방식을 발전시켰습니다.

- **Technical Details**: 제안된 모델은 두 단계의 결정 문제를 특징으로 하며, 쿼리 단계에서는 API를 순차적으로 쿼리하여 출력을 생성하고, 선택 단계에서는 생성된 출력 중 하나를 선택하게 됩니다. 이러한 출력 상황을 통해 현실적인 피드백 구조를 통해 결정 과정을 운영합니다. 또한, 이 연구에서는 예약 지수 함수를 일반화된 선형 함수로 파라미트릭 구조를 두어 설정하고, 결정을 위한 온라인 학습 접근 방법을 제공합니다.

- **Performance Highlights**: 제안된 정책은 예약 지수의 GMM(Generalized Method of Moments) 추정과 UCB(Upper Confidence Bound) 스타일의 접근법을 결합하여 작동합니다. 이 방식은 조기 종료 및 선택 과정에서 발생할 수 있는 후회(cumulative regret)를 O~(T)(sqrt{T})의 차원 의존적으로 보장합니다. 이를 통해 효율적인 LLM API 쿼리 및 선택에서 비용과 성과 사이의 균형을 이룰 수 있는 가능성을 보여줍니다.



### Off-Policy Evaluation with Strategic Agents via Local Disclosur (https://arxiv.org/abs/2606.07308)
- **What's New**: 본 논문에서는 정책 결정자가 정책에 응답하여 에이전트가 자신의 공변량을 전략적으로 수정하는 경우의 off-policy evaluation (OPE)에 대해 연구하였습니다. 이러한 전략적 행동은 정책 의존적인 공변량 변화를 유도하며, 기존 OPE 접근 방식의 가정을 깨뜨립니다. 저자들은 정보 비대칭을 완화하기 위해 정책 정보를 공개하는 방법으로 local information disclosure (LID)를 활용하는 혁신적인 방법론을 제시합니다.

- **Technical Details**: 연구진은 에이전트의 응답 모델을 추정하고 정책 가치를 추정하기 위해 doubly robust estimator (DR estimator)를 구성합니다. 또한 에이전트의 비용 민감도가 조건부 로그 정규 분포를 따른다고 가정하여 제안된 추정기의 일관성을 확립합니다. 이후, post-hoc 설명을 통해 에이전트의 원래 공변량을 관찰하고 이를 통해 전략적 수정 이전의 정보에 접근할 수 있음을 강조합니다.

- **Performance Highlights**: 저자들은 본 방식을 통해 평가된 정책 가치를 실증적으로 검증하였으며, 설계된 인터랙션 구조가 정보 비대칭을 완화하고 에이전트의 전략적 응답에서 숨겨진 구조를 드러내는 방식으로 작용하는 점을 강조합니다. 이를 통해 에이전트의 개인화된 응답을 보다 잘 이해하고 정책 평가의 정확성을 높일 수 있는 가능성을 제시합니다.



### DuMate-DeepResearch: An Auditable Multi-Agent System with Recursive Search and Rubric-Grounded Reasoning (https://arxiv.org/abs/2606.07299)
Comments:
          Technical report by the DuMate Team. 26 pages, 6 figures, 4 tables

- **What's New**: 이번 연구는 DuMate-DeepResearch라는 새로운 multi-agent Deep Research (DR) 프레임워크를 제안합니다. 이 프레임워크는 문제 이해, 계획, 실행을 관리하는 Agent Core와 검색 및 증거 수집을 위한 Tool Ecosystem을 분리한 구조로 설계되어 있습니다. 이를 통해 중간의 모든 의사결정과 도구 호출이 명확하게 추적될 수 있습니다.

- **Technical Details**: 구성 요소로는 그래프 기반의 동적 계획 전략, 재귀적인 두 단계 실행 설계, 규칙 기반의 테스트 최적화 메커니즘이 포함됩니다. 이러한 구성 요소들은 오랜 기간의 연구 로드맵을 같은 방향으로 발전시키고 다양한 서브 작업을 분리하여 처리하여, 노이즈로부터 완전한 생태계를 보장합니다. 이 구조는 전략적 사고와정보 정합성을 동시에 관리할 수 있도록 설계되어 있습니다.

- **Performance Highlights**: DuMate-DeepResearch는 두 가지 심층 연구 벤치마크에서 최고의 점수를 기록하여 새로운 최첨단 결과를 수립했습니다. DeepResearch Bench에서는 58.03%의 종합 점수를, DeepResearch Bench II에서는 61.95%의 종합 점수를 달성하며 정보 회수 및 분석 부문에서도 1위를 차지했습니다. 이로 통해 제안된 아키텍처가 보고서의 품질과 증거 확보 및 합성을 개선하는 데 효과적임을 입증하였습니다.



### TOPSIS-RAD: Ranking According to Desires (https://arxiv.org/abs/2606.07253)
Comments:
          21 pages, 15 Tables and 6 figures. The numerical computation of the data that appear in the Toy Examples was Supported by the Visual TOPSIS RAD that is available at this https URL. The data of the Toy examples are also available in this URL and can be loaded in the app as the template "Article"

- **What's New**: 이 논문은 전통적인 TOPSIS 방법이 의사결정자(Decision Maker, DM)의 요구와 맞지 않는 문제를 해결하기 위해 TOPSIS-RAD라는 새로운 방법을 제안합니다. 기존 TOPSIS는 긍정적 이상 솔루션(Positive Ideal Solution, PIS)과 부정적 이상 솔루션(Negative Ideal Solution, NIS)을 관찰된 대안 세트에서 도출하기 때문에 순위가 왜곡되기 쉽습니다. TOPSIS-RAD는 두 개의 DM 정의 참고 수준 배열을 통합하여 이러한 문제를 해결합니다.

- **Technical Details**: TOPSIS-RAD는 비타당한 대안(Vetoed Performance Levels, VPL)을 배제하여 정규화 전 순위를 왜곡할 수 있는 영향을 차단하고, DM이 원하는 수준에 성능을 제한하는 원하는 성능 수준(Desired Performance Levels, DPL)을 설정합니다. 이를 통해 PIS는 데이터 세트의 극단값이 아닌 명시적인 목표에 기반하여 고정됩니다. 이 방법은 TOPSIS의 거리 기반 구조를 유지하면서도 DM이 지정한 안정적인 경계 내에서 순위를 정립합니다.

- **Performance Highlights**: 본 연구는 세 개의 장난감 예제를 통해 각 메커니즘을 입증합니다. VPL은 비타당한 대안을 제거하여 정규화 경계를 재구성하고, 고정된 DPL 경계는 원하는 수준보다 높은 성능의 영향을 제한하여 순위를 안정화합니다. 논문에서는 방법의 한계와 향후 연구 방향에 대해서도 논의하였습니다.



### Think Fast: Estimating No-CoT Task-Completion Time Horizons of Frontier AI Models (https://arxiv.org/abs/2606.07157)
- **What's New**: 이 논문에서는 체인 오브 사고(Chain-of-Thought, CoT) 추론 없이 최전선 AI 모델들이 얼마나 잘 추론하는지를 측정합니다. 30,000개가 넘는 질문을 통해 수학, 코딩, 퍼즐, 인과관계, 마음의 이론, 전략적 추론 등 43개의 벤치마크에서 평가하였습니다. CoT 없이 작업을 수행할 때 모델과 인간을 비교하기 위해 50% 작업 완료 시간 수치(TH)를 추정합니다.

- **Technical Details**: 모델의 성능을 비교하기 위해 50% 작업 완료 시간과 50% 추론 토큰 수치를 동시에 분석합니다. 논문에서는 최전선 모델들의 CoT가 없는 50% TH가 매년 약 두 배로 증가하고 있음을 발견하였고, GPT-5.5의 TH는 3분을 초과하며, 추론 토큰 수치는 1,500 토큰을 초과합니다. 이 연구는 2028년까지 TH가 7분을 넘고, 2030년까지 25분에 이를 것으로 예측합니다.

- **Performance Highlights**: 최전선 모델들이 CoT 없이도 우수한 성능을 보여주고 있다는 점은 주목할 만합니다. 연구 결과에 따르면, 추론의 복잡성이 증가함에 따라 모델의 작업 완료 시간과 필요한 추론 토큰 수치가 지속적으로 증가하고 있습니다. 이러한 변화는 AI 모델의 향후 발전 방향에 중요한 시사점을 제공합니다.



### Beyond Post-hoc Explanation: Toward Glassbox AI via Probabilistic Mediation (https://arxiv.org/abs/2606.07113)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)이 복잡한 의사결정에 사용되면서 필요한 투명성을 확보하기 위해 새로운 아키텍처인 Glassbox Framework를 제안합니다. 기존의 사후 설명(post-hoc explainability) 접근 방식이 설명의 불안정성과 법적 책임 문제를 해결하지 못하는 고민을 담고 있습니다. Glassbox Framework는 Bayesian networks(BNs)를 활용해 예측이 이루어지기 전에 투명한 추론 구조를 제공함으로써 AI 시스템이 보다 책임감 있는 결과를 도출하도록 돕습니다.

- **Technical Details**: Glassbox Framework는 Bayesian Networks(BNs)를 사용하여 생성 모델이 결과를 예측하기 전, 도메인 지식 및 인과 관계를 수렴시키는 구조를 갖추고 있습니다. 이 프레임워크의 핵심 기여는 예측이 발생하기 이전에 구조화된 추론을 명시적으로 설계하여, 금전적 추정이나 불확실성 정량화, 공통의 분쟁 가능성을 제공하는 것입니다. 이를 통해 AI의 책임성을 더욱 강화할 수 있는 체계를 마련합니다.

- **Performance Highlights**: 이 논문은 Glassbox Framework의 구현 가능성을 높이기 위한 주요 도전 과제를 제시합니다. 특히, 의미적 정렬(semantic alignment), 동적 모델 구축(dynamic model construction), 확률적 기반(probabilistic grounding), 인간 거버넌스(human governance) 등이 있습니다. 이러한 과제를 해결함으로써, AI 시스템이 적용될 수 있는 고위험 분야에서 투명성과 책임성을 강화하는 기초 연구를 진행할 수 있는 길을 모색하고 있습니다.



### DyCon: Dynamic Reasoning Control via Evolving Difficulty Modeling (https://arxiv.org/abs/2606.07108)
Comments:
          Accepted at ICML 2026

- **What's New**: 이번 논문은 기존의 LRM(대규모 추론 모델)에서 발생하는 비효율적인 '과잉 사고(overthinking)' 문제를 해결하기 위해 DyCon이라는 새로운 프레임워크를 제안합니다. DyCon은 훈련 없이도 단계별 표현을 활용하여 문제의 난이도를 명시적으로 모델링하고, 이를 통해 추론의 깊이를 동적으로 조절할 수 있습니다. 이를 통해 복잡한 문제에 대한 충분한 탐색을 유지하면서도 단순한 문제에 대해서는 불필요한 추론 단계를 줄일 수 있습니다.

- **Technical Details**: DyCon은 LRM의 단계별 임베딩(step-level embeddings)에서 동적으로 진화하는 문제의 난이도를 모델링하는 기법입니다. 각 추론 단계에서 난이도를 예측하기 위해 소규모 데이터셋에서 선형 회귀 분석을 수행하고, 이를 기반으로 반사 키워드의 logit을 조절합니다. 이 과정에서 난이도가 낮을 경우 reflection 키워드의 logit을 줄여 빠른 수렴(convergence)을 유도하며, 난이도가 높을 경우에는 logit을 늘려 깊은 반사를 유도합니다.

- **Performance Highlights**: DyCon을 이용한 다양한 모델에 대한 실험 결과, 모델은 추가적인 정확성이나 일반화 능력을 잃지 않으면서 과도한 추론 단계를 효과적으로 줄일 수 있음을 보여주었습니다. 4B에서 32B에 이르는 4개 모델과 12개의 벤치마크에서 수행된 실험은 DyCon의 우수한 성능을 입증하며, 다양한 문제 복잡성과 도메인에서의 강력한 일반화 능력을 강조합니다.



### Front-to-Attractors: Modifying the Front-to-Front Heuristic in Bidirectional Search (https://arxiv.org/abs/2606.07047)
- **What's New**: 이번 논문에서는 F2F (front-to-front) heuristics의 계산 비용 문제를 해결하기 위해 새로운 heuristic 클래스인 F2A (front-to-attractors)를 소개합니다. F2A는 상반된 검색 전선의 모든 상태에 대해 거리를 평가하는 대신, 소규모의 동적으로 관리되는 attractors 세트를 활용하여 비용을 대폭 줄입니다. 이를 통해 F2A는 F2F의 정보성을 유지하면서도 훨씬 적은 계산 자원을 소모합니다.

- **Technical Details**: Bi-HS (Bidirectional heuristic search) 문제 인스턴스는 방향을 규정하는 두 개의 heuristic (hF, hB)을 포함하여 그래프 G와 시작(상태) s​t​a​r​t와 목표(g​o​a​l) 상태로 구성됩니다. F2E (front-to-end) 그리고 F2F heuristics는 각각 단일 목표를 기반으로 한 반면, 논문에서 제안하는 F2A는 attractors를 통해 검색 전반의 효율성을 향상시킵니다. 이 attractors는 전체 frontier를 대체하는 compact surrogate 역할을 하여 계산의 최적성을 유지합니다.

- **Performance Highlights**: F2A는 여러 도메인(2D grid pathfinding, Sliding Tiles, Pancake puzzle 등)에서 평가되었으며, F2F에 비해 pairwise 평가 횟수를 최대 11.2배 줄였습니다. 또한 평균적으로는 F2E에 비해 노드 확장 횟수를 4.8배 줄이는 성과를 보여줍니다. 이러한 성과는 Bi-HS 알고리즘의 성능 개선에 기여할 것으로 기대됩니다.



### Hierarchical Semantic-Constrained Heterogeneous Graph for Audio-Visual Event Localization (https://arxiv.org/abs/2606.07033)
- **What's New**: 이 논문은 Open-vocabulary audio-visual event localization (OV-AVEL)을 위한 계층적 의미 제약 이질 그래프(HSCHG)를 제안합니다. 이 방법은 오디오 및 비디오 세그먼트 노드와 해당 비디오 수준 노드 간의 이질적인 계층적 그래프를 구성하여, 여러 시간 스케일에서 오디오-비주얼 일관성을 유지하는 문제를 해결하려고 합니다. 특히, 이 연구는 하이퍼볼릭 공간(hyperbolic space)으로 멀티 레벨 오디오-비주얼 표현과 텍스트 프로토타입을 매핑하는 과정을 포함합니다.

- **Technical Details**: 제안된 HSCHG는 오디오-비주얼 일관성을 보장하기 위해 서로 다른 레벨의 시맨틱 정보를 통합하는 특징을 가지고 있습니다. 논문에서는 멀티 방향 시간 경로와 게이티드 퓨전 전략을 사용하며, 정밀한 시간 정보 캡처를 위한 역방향 및 정방향 엣지를 도입합니다. 이 방법은 또한 각 레벨 간의 의미적 일관성을 위한 양방향 의미 제약을 설정하여, 세그먼트와 비디오 표현 간의 관계를 동적으로 정교화합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 OV-AVEL 벤치마크에서 기존 방법보다 뛰어난 성능을 나타냈습니다. 변화 생성 연구(ablation studies)를 통해 제안한 모델의 효과성과 각 구성 요소의 유용성이 추가적으로 입증되었습니다. 이 연구는 오픈 어휘 환경에서 시맨틱 일관성을 구축하는 데 있어 중요한 기여를 합니다.



### StainFlow: Entity-Stain Tracking and Evidence Linking for Process Rewards in GUI Agents (https://arxiv.org/abs/2606.07027)
- **What's New**: 이 논문에서는 GUI 에이전트 (Agents)를 위한 새로운 보상 모델인 StainFlow를 소개합니다. 이는 기존의 Process Reward Models (PRMs)에서 발생하는 주관적인 전역 마일스톤 분해의 문제를 해결하기 위해 제안되었습니다. StainFlow는 전역 엔티티 얼룩 추적(Global Entity Stain Tracking) 모듈을 통해 여러 유효한 실행 경로를 보다 객관적으로 구분할 수 있도록 지원합니다.

- **Technical Details**: StainFlow는 두 가지 주요 모듈로 구성됩니다. 첫째, Global Entity Stain Tracking 모듈에서는 시각적으로 검증 가능한 작업 엔티티를 추출하고, 그 얼룩 농도(stain concentrations) 및 상태 변화를 추적하여 주어진 경로에 따라 작업 단계를 구분합니다. 둘째, Local Stain Evidence Linking 모듈에서는 각 후보 키 노드의 관련 단계를 동적으로 구축하여 진정한 키 노드를 검증하기 위한 고밀도 증거 창을 생성합니다.

- **Performance Highlights**: 경험적으로, StainFlow는 AndroidWorld 및 OGRBench에서 온라인 RL(success) 성공률을 3.2% 개선하고, 경로 완성 판단 정확도를 1.8% 향상시켰습니다. 이러한 성과는 스테인 추적(stain tracking) 메커니즘의 도입이 GUI 에이전트의 수행 능력을 향상시키는데 기여했음을 보여줍니다.



### The Sim-to-Real Gap of Foundation Model Agents: A Unified MDP Perspectiv (https://arxiv.org/abs/2606.07017)
Comments:
          7 pages, 2 figures, 2 tables. Accepted by KDD 2026 Blue Sky Ideas Track

- **What's New**: 본 논문에서는 Foundation model agents의 평가 및 훈련 간의 격차를 전통적인 simulation-to-reality(시뮬레이션에서 현실로) 문제로 포괄적으로 형식화하는 새로운 접근 방식을 제안합니다. 이를 통해 Markov Decision Process(MDP)의 네 가지 요소인 Observation(관측), Action(행동), Transition(전이), Reward(보상) 중심으로 격차를 다룹니다. 기존의 접근법을 보완하기 위해 domain randomization과 grounded action transformation 등의 솔루션을 채택할 것을 권장합니다.

- **Technical Details**: RL(강화학습)은 보통 할인된 Markov 결정 과정(MDP)으로 형식화되어 있으며, 이는 상태 공간(𝒮), 행동 공간(𝒜), 전이 동역학(𝒯), 보상 함수(ℛ)를 포함합니다. 시뮬레이션 MDP에서 훈련된 정책은 현실 MDP에 배포되고, 시뮬레이션과 현실 간의 격차는 각 요소의 불일치에서 발생합니다. 주로 관측(Observation), 행동(Action), 전이(Transition), 보상(Reward) 간의 차이가 그 원인으로 지적됩니다.

- **Performance Highlights**: 이 논문은 새로운 언어와 표준화된 스트레스 테스트 기준을 도입하여 Foundation model의 실행 가능성을 높이고, 신뢰할 수 있는 에이전트의 개발을 촉진하고자 합니다. 마지막으로, 이 연구 노력이 KDD(지식 발견 및 데이터 마이닝) 커뮤니티의 주제를 발전시키고 신뢰할 수 있는 데이터 과학의 기초를 마련하는 데에 기여할 것으로 기대됩니다.



### Teaching the Way, Not the Answer: Privileged Tutoring Distillation for Multimodal Policy Optimization (https://arxiv.org/abs/2606.07000)
- **What's New**: 본 논문에서는 Reinforcement Learning with Verifiable Rewards (RLVR)의 한계를 극복하기 위해 Privileged Tutoring Distillation Policy Optimization (PTD-PO) 프레임워크를 제안합니다. PTD-PO는 학생 정책에게 답변을 노출하지 않으면서도 밀집한 가이드를 제공하여 효율적인 탐색을 개선합니다. 기존의 방법들과 달리, PTD-PO는 공간적 주의(spatial attention)와 중간 텍스트적 추론(Intermediate textual reasoning) 단계를 이용한 구조화된 힌트를 통해 학생 모델을 훈련시킵니다.

- **Technical Details**: PTD-PO는 원래의 답변 없는 맥락에서 학생 정책을 최적화하고, 실패한 롤아웃을 힌트로 강화된 기준 모델과 정렬하여 토큰 분포 수준에서 학습 신호를 생성합니다. 또한, Top-K Jensen-Shannon divergence 목표를 도입하여 가이드와 비가이드 컨텍스트 간의 분포 변화에서 안정성을 유지하며 메모리 오버헤드를 줄입니다. 이 접근법은 비대칭적인 증류(asymmetric distillation) 간의 불안정성과 메모리 집약적인 문제를 해결합니다.

- **Performance Highlights**: 2B에서 8B 파라미터의 대형 비전-언어 모델(LVLM)을 대상으로 한 실험에서, PTD-PO는 RLVR 및 기존 증류 기반 기준선보다 지속적으로 우수한 성능을 보였습니다. PTD-PO는 더 높은 정책 엔트로피(policy entropy)를 유지하며, 실패한 롤아웃에서의 회복 능력을 향상시킵니다. 이 결과는 PTD-PO가 복잡한 다중 모달 추론 성능을 향상시키고 탐색을 유지할 수 있는 잠재력을 가지고 있음을 보여줍니다.



### Exploring Agentic Tool-Calling Decisions via Uncertainty-Aligned Reinforcement Learning (https://arxiv.org/abs/2606.06976)
- **What's New**: 본 논문에서는 대규모 언어 모델 기반 에이전트의 도구 호출 결정 실패를 개선하기 위한 새로운 프레임워크인 TRUST를 제안합니다. 이전 연구들은 주로 도구 호출 결정을 위한 보상 신호를 결정의 결과와 구조적 체크리스트에 기반하여 개선했으나, 본 연구는 불확실성(uncertainty)의 특성을 강조합니다. TRUST는 보상 설계에 불확실성 정량화(uncertainty quantification)를 통합하여 에이전트의 결정 역학을 최적화하고, 오류 누적을 줄이는 데 초점을 맞추고 있습니다.

- **Technical Details**: TRUST는 보상 구조 내에 불확실성을 고려하며, 올바른 결정에 대해서는 낮은 불확실성을, 잘못된 결정에 대해서는 상대적으로 높은 불확실성을 부여하는 방식으로 작동합니다. 이를 통해 탐색을 촉진하고 보다 신뢰할 수 있는 대안을 위한 정책 업데이트의 신호 강도를 높입니다. 또한, 단독 결정 인스턴스에서 다중 턴 에이전트 궤도로 훈련 패러다임을 확장하여, 경량 키 턴 주석을 사용해 전체 궤적을 재라벨링하지 않고도 훈련을 강화합니다.

- **Performance Highlights**: 실험 결과, TRUST는 다양한 도구 사용 벤치마크에서 에이전트의 결정 품질과 성능을 일관되게 개선하는 것으로 나타났습니다. 사용된 벤치마크에서 When2Call 작업의 정확도를 11% 이상 향상시키며, 복잡한 다중 턴 상호작용 및 도구 사용 궤적에서도 성능을 강화합니다. 특히 무의미한 도구 호출과 같은 도전적인 결정 밀집 시나리오에서 더욱 두드러진 성과를 보여줍니다.



### Accounting for Context: Shaping Moral Credences for Value Alignmen (https://arxiv.org/abs/2606.06972)
- **What's New**: 이 논문에서는 에이전트의 행동이 인간의 도덕적 가치와 일치하도록 보장하는 가치 정렬(value alignment)의 문제를 다룬다. 특히, 서로 다른 도덕 이론에 따른 평가를 공정하게 집계하기 위한 메커니즘이 필요하다고 주장하면서, 맥락적 요소를 포함해야 한다고 강조하고 있다. 이 논문은 도덕적 불확실성(moral uncertainty) 하에서 에이전트의 의사 결정을 형식화하며, 이러한 맥락적 요소가 도덕 평가의 집계 결과에 미치는 영향을 분석한다.

- **Technical Details**: 연구는 도덕 이론의 다원성과 에이전트의 행동 평가에서의 맥락적 요소를 고려한 형식적 프레임워크를 제공한다. 의사 결정 이론을 확장하여 도덕 이론에 대한 신뢰도를 확률적으로 표현하고, 이 신뢰도를 통해 각 도덕 이론의 평가를 사회적 선택(social choice) 방식으로 집계하는 방법론을 제안한다. 이를 통해 여러 가지 집계 방법이 약한 파레토 원칙(weak Pareto principle)을 위반할 수 있다는 결과를 도출하며, 이는 심슨의 역설(Simpson's paradox)의 변형으로 해석된다.

- **Performance Highlights**: 이 논문은 다양한 윤리적 이론을 고려할 때 도덕적 선택의 집계 과정에서의 중대한 문제를 제기한다. 특히, 신뢰도와 맥락을 반영하는 방식으로 도덕적 신념(moral credences)을 다루는 새로운 연구 방향을 제안하고 있다. 이 연구는 도덕적 평가 집계의 한계를 식별할 뿐만 아니라, 향후 연구의 중요한 방향성을 제시하는 데 기여한다.



### Quantum-Inspired Trace-Augmented Evidence Selection for Reasoning over Structured Hypothesis Spaces (https://arxiv.org/abs/2606.06941)
- **What's New**: 이 논문은 법률과 같은 전문 분야에서의 오류를 줄이기 위해 EP-HUBO(Evidence Pool Higher-Order Binary Optimisation)라는 새로운 접근 방식을 제안합니다. 전통적인 majority vote 기반의 집계 방식 대신, EP-HUBO는 Chain-of-Thought(CoT) 사고 과정의 조각을 명시적인 조합 최적화(combinatorial optimisation) 문제로 처리하여, 소수의 경우라도 뒷받침되는 가설이 노이즈가 많은 다수결 답변을 초월할 수 있도록 합니다.

- **Technical Details**: EP-HUBO는 네 가지 주요 단계를 포함합니다. 첫째, 작은 로컬 모델을 사용하여 질문당 여러 CoT 트레이스를 생성합니다. 둘째, 이 모델을 사용하여 조각을 각 가설별 증거 풀로 변환합니다. 셋째, 각 증거 풀에 대한 higher-order unconstrained binary optimization(HUBO) 문제를 해결하여 각 답변 풀을 강력히 지지하는 조각들을 선택합니다. 마지막으로, 큰 프론티어 모델을 사용하여 최적화된 증거를 기반으로 단일 판단을 내립니다.

- **Performance Highlights**: EP-HUBO는 법률 관련 두 개의 벤치마크에서 majority vote 방식보다 상당한 개선을 보여주었습니다. LEXam에서 프론티어 LLM은 87.7%의 질문에 대해 동일한 답변 옵션을 선택하는 심각한 편향을 보였으나, HUBO 선택 증거를 사용할 경우 이 편향이 감소하며 정확도가 +11.4pp 향상되었습니다. EP-HUBO는 또한 양자 컴퓨터와 전통적인 컴퓨터 모두에서 유사한 성능을 보여, 최적화 문제 해결에 있어 유용한 대안임을 입증하였습니다.



### Declarative Skills for AI Agents in Knowledge-Grounded Tool-Use Workflows (https://arxiv.org/abs/2606.06923)
- **What's New**: 이 논문에서는 비구조화된 지식 기반에서 고객 서비스 워크플로우를 위한 도구-사용 AI 에이전트의 오케스트레이션 메커니즘을 연구합니다. 저자들은 자연 언어 기술 파일이 시스템 프롬프트에 추가된 선언적 에이전트(DeclarativeAgents)가 효과적인 오케스트레이션 패러다임임을 주장합니다. 비교 분석을 통해 프로그램 상태 머신(ImperativeAgent)과 기본 기반선 에이전트(unscaffolded baseline agent)와의 성능 차이를 empirically 평가합니다.

- **Technical Details**: 세 가지 에이전트는 분산된 부분 관측 마르코프 결정 과정(Decentralised Partial Observable Markov Decision Process, Dec-POMDP)으로 공식화됩니다. 각 에이전트는 정보 이론적 및 구조적 특성을 분석하고, 고급 언어 모델과 두 가지 검색 체제를 이용하여 성능 차이를 테스트합니다. 이 연구는 에이전트가 수행하는 작업의 성공, 견고성, 준수 및 효율성을 모두 평가합니다.

- **Performance Highlights**: 실험 결과는 검색 품질이 AI 에이전트의 주요 병목 현상이라는 것을 보여줍니다. 고품질 검색 환경에서는 선언적 기술이 절차적 작업의 정확성을 지속적으로 개선하고 오케스트레이션 오류를 줄이는 반면, 프로그램 상태 머신은 일관되게 작업 성공이나 준수를 향상시키지 못했습니다.



### Workflow-to-Skill: Skill Creation via Routing-Workflow-Semantics-Attachments Decomposition (https://arxiv.org/abs/2606.06893)
Comments:
          10 pages, 2 figures

- **What's New**: 본 논문은 자동화된 Skill 생성의 중요성을 강조하며, 다양한 상호작용 증거를 통해 기술적인 절차 지식을 암호화할 수 있는 방법을 제안합니다. 특히, 트레이스(Trace)가 단순한 요약 작업이 아닌, 복잡하고 세분화된 프로세스임을 지적하며, 이를 해결하기 위한 새로운 중간 표현인 RWSA를 소개합니다. RWSA는 Skill을 Workflow 구조, 실행 Semantics, 및 런타임 Attachments로 분해하여 처리하고, 이를 기반으로 한 W2S 프레임워크를 제안합니다.

- **Technical Details**: W2S는 트레이스를 세그먼트화하고, 지역적인 Skill 초안을 유도하며, 공유된 구조를 정렬 및 재조정하고, 중복성을 압축하면서도 증거와 신뢰 주석을 보존합니다. Skill-IR은 상호작용 트레이스를 컴파일 가능한 객체로 변환하여 실행 가능한 에이전트 지침으로 변환하는 역할을 합니다. Skill-IR은 경로 헤더와 세 가지 런타임 구성 요소로 구성되어 있으며, 이를 통해 Skill의 적합성과 실행 방식을 명확히 정의합니다.

- **Performance Highlights**: 실험 결과 W2S가 70개의 Skills에 대한 행동 재생 일관성을 10.5% 개선한 것으로 나타났으며, 이는 트레이스를 실행 가능한 런타임 사양으로 간주해야 한다는 필요성을 강조합니다. W2S는 능동적이고 체계적인 Skill 생성을 지원하여, 기존 기술적인 요약 중심 접근법에 비해 더 나은 구조적 충실성과 행동 일관성을 제공합니다. 이런 결과는 경험으로부터 신뢰할 수 있는 학습을 위한 명시적 중간 표현과 런타임 인식 구조의 중요성을 보여줍니다.



### Evidence-Based Intelligent Diagnostic and Therapeutic Visualization System with Large Language Models: Multi-Turn Interaction and Multimodal Treatment Plan Generation (https://arxiv.org/abs/2606.06869)
Comments:
          29 pages, 9 figures, 5 tables, including supporting information

- **What's New**: 본 연구에서는 투명성과 해석 가능성을 개선하기 위해 지식 향상 시각 진단 시스템을 제안합니다. 기존의 전통 중의학 진단 도구들은 불투명한 추론 과정과 수동적인 대화 방식, 제한적인 치료 계획 제시 문제를 가지고 있습니다. 이 시스템은 241개의 증후군, 1,263개의 증상, 그리고 2,485개의 관계로 구성된 Neo4j 지식 그래프에 기반하여 구축되었습니다.

- **Technical Details**: 시스템은 정밀, 의미론적, 퍼지, 대형 언어 모델 검증의 네 단계 증상 매칭 파이프라인을 포함하고, 정보 이득 기반의 적극적인 질문 전략을 유전 알고리즘으로 최적화하여 개발되었습니다. 또한, 인공지능 생성 일러스트레이션, 3D 경락-혈 허점을 통합한 다중 모드 치료 계획 제시를 통해 사용자와의 상호작용을 향상시킵니다. 이러한 요소들은 진단의 신뢰성(reliability)과 정보의 명확성을 높이기 위해 설계되었습니다.

- **Performance Highlights**: 사례 연구를 통해 자동화된 쌍 비교 평가의 결과, 진단에 대한 신뢰성이 크게 향상됨이 입증되었습니다. 이 시스템은 정보 과부하를 줄이고, 환자 자가 평가 및 의료 교육에서 효과적인 상호작용 워크플로우가 가능하도록 하였습니다. 또한, 임상적 관점에서 진단의 신뢰성을 32% 향상시키고, 인지 부하를 감소시키는 결과를 도출하였습니다.



### AdMem: Advanced Memory for Task-solving Agents (https://arxiv.org/abs/2606.06787)
- **What's New**: 이번 논문에서는 semantic (의미), episodic (경험적), procedural (절차적) 메모리를 통합한 새로운 메모리 프레임워크인 AdMem을 소개합니다. 이 구조는 단기 및 장기 메모리를 결합한 이단계 설계를 통해 LLM(대형 언어 모델)의 기억력 한계를 극복하고자 합니다. 기존에는 사실 정보 저장 및 과거 성공 사례 재사용에 국한되었으나, AdMem은 실패 사례를 다루며 온라인 확장성을 보장합니다.

- **Technical Details**: 제안된 메모리 프레임워크는 multi-agent (다중 에이전트) 아키텍처를 기반으로 하여 actor (행동자), memory (메모리), critic (비평자) 에이전트 간의 자동 메모리 생성, 보상 주석 부여 및 적응형 검색을 가능하게 합니다. 이 시스템은 메모리 관리에 보상 기반 평가, 병합 및 가지치기를 적용하여 지속적인 개선을 촉진하고 확장성을 보장합니다. 또한, 메모리 생성을 위해 적응형 작업 계획, 기대 주석 및 자동 반영이 포함됩니다.

- **Performance Highlights**: 다양한 환경에서 진행된 실험 결과, AdMem은 기존 기준선과 비교하여 복잡한 다단계 작업에서의 강인성과 성공률 모두에서 개선된 성과를 보였습니다. 이 연구는 LLM 기반 에이전트를 위한 포괄적이고 적응성 있는 메모리 관리의 중요성을 강조하며, 사용자 맞춤화 및 의사결정 과정의 자기 개선을 통한 장기 작업 해결에 기여할 수 있음을 보여줍니다.



### OpenSkill: Open-World Self-Evolution for LLM Agents (https://arxiv.org/abs/2606.06741)
Comments:
          20 pages, 4 figures and 8 tables. Code is avalable at this https URL

- **What's New**: 이 논문은 기존의 self-evolving agents가 특정 피드백이나 습득된 기술들을 사용하여 개선할 수 있다는 점을 넘어서, 오히려 제한된 태스크 프롬프트만을 활용해 기초를 세우는 방법을 제시합니다. 제안된 OpenSkill 프레임워크는 오픈월드 자원을 통해 기술과 검증 신호를 스스로 생성할 수 있도록 지원하여, 기존 방법과 달리 명확한 목표 태스크 감독 없이 진화를 가능하게 합니다. 이러한 접근은 대규모 언어 모델(LLM) 에이전트의 자율성을 증가시킵니다.

- **Technical Details**: OpenSkill은 세 단계로 구성된 파이프라인을 통해 오픈 월드 지식을 획득하고, 자가 생성한 가상 태스크를 통해 기술을 정교화하며, 최종적으로 제로샷(zero-shot) 평가를 통해 목표 에이전트에 기술을 배포합니다. 이러한 과정에서 오픈 월드는 배우는 지식과 감독 독립적인 연습 환경을 제공하여, 최종 평가에만 목표-태스크 감독을 사용합니다. 이를 통해 모델 간의 기술 전이를 지원합니다.

- **Performance Highlights**: OpenSkill은 세 가지 벤치마크와 두 개의 목표 에이전트 설정에서 가장 높은 자동 통과율을 기록합니다. 특히, SkillsBench에서 가장 강력한 폐쇄형 세계 기반라인보다 8.9% 높은 성능을 보이며, 모델 특화적 조정 없이 기술을 전이할 수 있습니다. 또한, OpenSkill의 자체 생성 검증자는 실제 테스트 의도와 88.9% 정합성을 갖습니다.



### A Geometric Account of Activation Steering through Angle-Norm Decomposition (https://arxiv.org/abs/2606.06735)
- **What's New**: 이 논문은 언어 모델의 행동을 제어하기 위한 선형 활성화 조정 방법의 한계를 다루면서, 구면 조정 방식이 더 나은 해결책이 될 수 있음을 제시합니다. 저자들은 각 조정 방식의 각도와 반지름 구성 요소의 역할을 구분하여, activations에서 개념 정보가 주로 각도로 표현된다는 것을 보여줍니다. 이 연구는 기존 연구들이 가정했던 두 가지 주요 방향을 다시 조명합니다.

- **Technical Details**: 언어 모델의 중간 표현을 활용한 조정 방법에 대한 여러 실험을 통해, 보존된 노름(norm)이 생성의 품질을 유지하면서도 개념 정보를 효과적으로 조절할 수 있음을 입증했습니다. 본 연구에서는 7개의 언어 모델과 4개의 개념 데이터셋을 사용하여 조정 방식을 비교하였으며, 각 조정 방식의 구성 요소를 angular (각도)와 radial (반지름)로 분리하여 분석하였습니다. 이 과정에서 저자들은 각 조정 방식이 개념 방향을 향한 각도의 변화와 히든 상태의 크기 변화 간의 관계를 다루고 있습니다.

- **Performance Highlights**: 연구 결과, 활성화 방향에 주로 개념 정보가 암호화된다는 것을 발견했으며, 이로 인해 조정 방법이 더 이해하기 쉬운 설계를 가질 수 있음을 제안합니다. 특히, 각도 변화가 의미적 조정을 조절하는 데 중요한 역할을 하고, 반지름은 생성 안정성과 입력 관련성을 유지하는 데 필수적임을 보여주었습니다. 이 새로운 두 가지 파라미터 접근법은 독립적으로 고려되지 않았던 다른 결과를 비교적 입증하였습니다.



### AEGIS: A Backup Reflex for Physical AI (https://arxiv.org/abs/2606.06660)
- **What's New**: 본 논문은 로봇 조작의 장기 실패를 예방하기 위한 새로운 방법인 AEGIS(Activation-probe Early-warning, Gated Inference Switching)를 소개합니다. AEGIS는 약한(policy) 정책의 동결된 활성화(activations)를 사용하여 고위험 단계를 감지하고, 문제 발생 전 제어를 더 강력한 정책으로 전환합니다. 이런 방식은 실패가 발생할 것으로 예상되는 단계에서만 더 강한 정책을 호출하여 효율성을 높입니다.

- **Technical Details**: AEGIS는 약한 정책의 활성화를 기초로 하여 단계별로 조기 경고를 날립니다. 특정 임계값(threshold)을 넘어설 경우에만 강력한 정책으로 전환하고, 이로 인해 복구(measure recovery)가 가능합니다. 그 결과, AEGIS는 약한 정책으로 발생하는 실패의 10.1%를 회복하며, 이는 다른 접근법에 비해 상대적으로 높은 수치입니다.

- **Performance Highlights**: 실험 결과, AEGIS는 약한 정책의 실패를 10.1% 회복하며, 이는 대조군인 붐 인상과 무작위 트리거에 비해 각각 5.4pp, 5.0pp의 개선을 보입니다. 해당 성과는 700개의 에피소드를 통해 검증 되었으며, 예측 정확도도 0.76의 AUROC으로 고무적인 결과를 보였습니다. 이러한 결과는 AEGIS가 예측과 회복을 효과적으로 결합할 수 있음을 보여줍니다.



### A Study of Parallel Continuous Local Search (https://arxiv.org/abs/2606.06656)
- **What's New**: 이번 연구에서는 대칭적인 의사-불리언(Pseudo-Boolean, PB) 제약조건을 갖는 부울 만족성 문제를 해결하기 위한 병렬 지속적 지역 탐색(Continuous Local Search, CLS) 접근 방식을 연구했습니다. 연구 결과, 만족 가능한 사례의 경우 전역 최소값이 주어진 SAT 문제의 만족하는 할당에 해당한다는 점이 밝혀졌습니다. 더불어, 중복된 제약조건이 수렴을 오히려 방해할 수 있다는 사실도 발견하였습니다.

- **Technical Details**: CLS는 부울 변수를 구간 [-1, 1]의 실수로 완화하여 제한된 비볼록 지속적 최적화 문제로 재구성합니다. 이 방법은 FourierSAT에 의해 구체화되었으며 GPU 계산을 위해 FastFourierSAT로 확장되었습니다. 특히, 대칭 제약조건의 경우 증표 조합의 수를 줄여 closed-form Fourier 계수를 도출할 수 있는 장점이 있습니다.

- **Performance Highlights**: 세 가지 사례 연구를 통해 CLS의 동작 방식에 대한 다양한 측면이 드러났습니다. 실험 결과, CLS는 NVIDIA Tesla V100 GPU에서 신속하게 솔루션을 찾을 수 있으며, 특히 Ramsey 색칠 문제와 같은 실제 문제에서 성능을 크게 향상시킬 수 있음을 보여주었습니다. 이를 통해 CLS가 가장 현대적인 하드웨어에서의 SAT 문제 해결을 위한 실용적인 사용에 어떻게 기여할 수 있는지에 대한 통찰도 제공하였습니다.



### Accelerated Fourier SAT (AFSAT): Fully Realising a GPU-based Symmetric Pseudo-Boolean SAT Solver (https://arxiv.org/abs/2606.06641)
- **What's New**: Accelerated Fourier SAT (AFSAT)는 GPU 가속을 활용한 pseudo-Boolean satisfiability을 해결하기 위한 새로운 solver를 제시하며, 이는 continuous local search (CLS) 접근 방식을 기반으로 합니다. AFSAT는 이전의 FastFourierSAT 개념을 완전히 설계된 solver로 발전시켰으며, 이는 다양한 대칭 제약 조건을 포함한 문제를 처리할 수 있습니다. JAX 컴파일러를 사용하여 AFSAT는 함수를 순수하게 조합하고 자동 벡터화, 자동 미분, 그리고 JIT(Just-In-Time) 컴파일을 수행하여 대량의 후보 할당에 대해 병렬로 CLS를 수행하게 합니다.

- **Technical Details**: AFSAT는 Python과 JAX 및 XLA 컴파일러를 사용하여 구현되었습니다. 이 시스템은 PB-encoding된 문제 사양을 정의하고 구문 분석하며, 모든 지원되는 대칭 제약 타입에 대한 Fourier 계수를 닫힌 형태로 계산합니다. JIT 컴파일된 solver 커널은 후보 할당 배치에 대해 자동 미분이 가능하며 벡터화되어 GPU에서 효율적으로 실행됩니다.

- **Performance Highlights**: AFSAT는 FastFourierSAT에 비해 실행 시간, 메모리 소비 및 피크 처리량에서 상당한 개선을 보입니다. 또한, 다중 GPU에서의 거의 선형 확장을 유지하며, 특수한 DFT 행렬 구성으로 부동 소수점 오류를 최소화하여 숫자적 안정성 및 정밀성을 개선했습니다. AFSAT는 부분적인 변수 할당을 입력으로 받아 들일 수 있어, 계층 기반 아키텍처나 포트폴리오 접근 방식 내에서 하위 해결자로 활용될 수 있습니다.



### Position: Don't Just "Fix it in Post": A Science of AI Must Study Training Dynamics (https://arxiv.org/abs/2606.06533)
Comments:
          Accepted as an oral to the ICML: this https URL

- **What's New**: 이번 논문은 AI(인공지능)의 과학적 이해의 필요성에 대해 논의합니다. 기존 AI 연구가 모델을 정적인 객체로 취급하며 훈련 이후의 행동만 분석하는 경향이 있음을 지적합니다. 모델 행동이 왜 발생하는지를 이해해야 한다는 점을 강조합니다.

- **Technical Details**: AI 모델은 데이터, 목표, 아키텍처(architecture), 최적화 역학(dynamics)에 의해 형성된 시간에 따라 진화하는 과정의 스냅샷으로 보아야 합니다. 논문에서는 훈련 역학(training dynamics)을 연구하여 모델 행동을 이해하는 새로운 과학의 필요성을 제시하며, 초기 훈련 신호로부터 결과를 예측하고 잘못된 경로에서 개입하는 방법을 다루고 있습니다.

- **Performance Highlights**: 훈련 절차를 설계하여 원하는 속성을 더 신뢰성 있게 생성하는 것이 궁극적인 목표입니다. 손실(loss) 예측의 성공을 능력(capabilities), 편향(biases), 강건성(robustness), 안정성(safety)과 관련된 행동에 확장하는 것이 도전 과제이며, 과학의 역사 및 철학에 기초한 이러한 이론의 요구 사항을 정립하고 관찰 가능한 문제를 식별합니다.



### CARVE-Q: Quantum-Proposed, Classically Certified Interactive Driving Repair (https://arxiv.org/abs/2606.06531)
Comments:
          9 pages, 3 figures

- **What's New**: CARVE(지정된 기동 행동 수리의 증명 가능한 저비용 모델)은 차량 상호작용 후 발생하는 반발 행위를 수리하기 위한 혁신적인 인증 구조를 제시합니다. 이 프레임워크는 차단된 행동에 대해 안전성과 책임이 명확한 수리 방안으로 전환하여, 예를 들어, 정지하거나 양보하거나 감속하는 행동을 제안합니다. CARVE는 전통적인 예측 기반 접근 방식과 달리, 상황에 맞는 수리 작업을 정형화하고 이를 인증할 수 있도록 합니다.

- **Technical Details**: CARVE는 하드 규칙이 있는 폐쇄된 기동 행동을 수정하기 위해 유한한 수리 격자를 구축하고, 이를 통해 수리 증명서를 발급합니다. 이 증명서는 바인딩 규칙, 선택된 공동 수리, 우선 권한에 따른 협력 범위, 책임에 따른 비용 분배 및 자율적인 대체 행동을 기록합니다. 증명서의 구조적 결정은 여러 소유자의 수리에 따른 복잡한 격자를 소개하며, CARVE-Q는 양자 최소 탐색 기술을 사용하여 이 격자의 효율성을 높입니다.

- **Performance Highlights**: CARVE는 65,536개의 할당에 대한 상태 벡터 최소 찾기를 통해 수리 작업의 정확성을 입증하며, INTERACTION 재생에서 100% 우선 권한 존중과 0% 허위 긍정으로 유효성을 검증하였습니다. 이러한 결과는 CARVE가 인증된 자율주행의 신뢰 기반 모델로 작용할 수 있음을 보여줍니다. 양자 분산 시스템의 활용은 워크플로우의 효율성을 높이며, CARVE가 정보 안전 및 책임을 유지하는 방식으로 작동함을 강조합니다.



### Attack Selection in Agentic AI Control Evaluations Meaningfully Decreases Safety (https://arxiv.org/abs/2606.06529)
- **What's New**: 이번 논문은 전략적으로 공격 시점을 선택하는 공격자와 관련된 능력을 연구하며, 공격 결정 과정을 두 가지 정책, 즉 시작 정책과 중단 정책으로 나누어 분석합니다. 기존의 연구에서는 공격자가 무작위로 공격 확률을 독립적으로 결정한다고 가정했지만, 이 연구는 공격자가 추구하는 최적의 기회를 정교하게 분석하도록 세분화합니다. 이로 인해 AI 시스템의 안전성 평가 방식이 개선될 것으로 기대합니다.

- **Technical Details**: 연구는 BashArena와 LinuxArena라는 두 가지 에이전트 환경에서 수행됩니다. 여기서 시작 정책은 공격 시점을 결정하고, 중단 정책은 공격이 진행 중일 때 중단할지를 판단합니다. 각 정책은 너무도 미세하게 공격을 삼가서 최적의 공격 기회를 파악함으로써 계산된 안전성을 크게 낮추기 때문에, 향후 AI 제어 평가 방안에 있어 이 정책을 반영할 필요성을 제기합니다.

- **Performance Highlights**: 실험 결과, 시작 정책은 BashArena와 LinuxArena에서 각각 20pp의 안전성을 감소시켰으며, 중단 정책은 BashArena에서 20pp, LinuxArena에서 28pp의 안전성을 감소시켰습니다. 이는 공격자의 능력이나 공격 수행 방식의 변화 없이도 발생하며, 선택적 공격자에 대한 안전성 추정치가 과도하게 낙관적이지 않도록 주의해야 함을 시사합니다. 따라서, 향후 제어 평가에서는 공격 선택을 명시적으로 파악하여 더욱 현실적인 안전 추정치를 도출할 것을 권장합니다.



### CrowdMath: A Dataset of Crowdsourced Mathematical Research Discussions (https://arxiv.org/abs/2606.06526)
Comments:
          16 pages, 4 figures

- **What's New**: CrowdMath는 수학적 문제 해결을 위한 협동 연구 논의를 기록한 새로운 데이터셋으로, MIT PRIMES 아트 오브 문제 해결 프로그램에서 수집된 164개의 전문가 주석 처리된 진행 체인을 포함합니다. 기존 데이터셋이 잘 정의된 문제 해결이나 최종 답변에 중점을 둔 것과는 달리, CrowdMath는 여러 참가자들이 기여한 부분 아이디어와 진전을 포함하고 있으며, 이를 통해 수학적 진전의 협력적 과정을 효과적으로 드러냅니다.

- **Technical Details**: CrowdMath의 데이터는 프로세스의 다양한 기능적 역할에 따라 각 게시물에 주석을 달 수 있도록 구성되어 있으며, 게시물은 진행(Progress), 증명(Proof), 오류(Erroneous), 오류 발견(FindError), 질문(Question), 답변(Answer) 등의 역할을 가집니다. 이 데이터셋은 협동적 수학 연구의 실질적인 진행을 반영하며, 수학적 논의의 흐름을 모델링하는 데 초점을 맞추고 있습니다.

- **Performance Highlights**: 모델들은 다음 게시물 예측에서 83-88%의 정확도를 기록함으로써 수학적 논의의 표면 흐름을 따라갈 수 있음을 보여주었습니다. 그러나 개별 기여의 기능적 중요성을 인식하는 데 어려움이 있으며, 최선의 모델조차도 포스트 역할 분류에서 0.42의 macro-F1 점수를 기록했습니다. 이는 현재 모델들이 수학적 문제 해결을 점진적인 협력 과정으로 이해하는 데 한계가 있음을 시사합니다.



### Lean4Agent: Formal Modeling and Verification for Agent Workflow and Trajectory (https://arxiv.org/abs/2606.06523)
- **What's New**: 이번 논문에서는 LLM (Large Language Model)의 신뢰 가능한 다단계 작업 흐름 실행을 위한 새로운 프레임워크인 **Lean4Agent**를 소개합니다. **Lean4Agent**는 의존형 타입 이론(dependent-type theory)을 활용하여 에이전트 행동을 모델링하고 검증할 수 있는 최초의 시도로, **FormalAgentLib**라는 Lean4 라이브러리를 출시합니다. 이 프레임워크는 명시적 가정 하에 에이전트 작업 흐름의 의미적 일관성을 공식적으로 모델링하고 검증하는 기능을 제공합니다.

- **Technical Details**: **Lean4Agent**의 설계는 에이전트 작업 흐름과 실행 궤적을 명시적 가정 하에 모델링하고 검증하기 위한 공식적인 기초를 제공합니다. 이 프레임워크는 세 가지 레벨의 정확성(구조적, 의미적, 실행 궤적)에 따라 에이전트 작업 흐름을 검증하는 **FormalAgentLib**를 포함하며, **LeanEvolve**를 통해 작업 흐름의 성능을 강화하는 방법론을 제안합니다. 이러한 구조는 정적 의미적 정확성을 자동으로 증명할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, 검증을 통과한 작업 흐름이 실패한 작업 흐름에 비해 평균 **11.94%** 높은 성능을 보였고, **LeanEvolve**를 사용한 추가 개선으로 SWE (Software Engineering) 작업에서 평균 **7.47%**의 성능 향상을 기록했습니다. 이 통계적으로 유의미한 개선은 **Lean4Agent**의 작업 흐름 검증 및 세련화 방법의 유용성을 입증합니다. **Lean4Agent**는 향후 자가 개선형 LLM 에이전트 시스템 개발의 기초를 마련하고 있습니다.



### SafeGene: Reusable Adapters for Transferable Safety Alignmen (https://arxiv.org/abs/2606.06519)
- **What's New**: 이번 논문에서는 Open-weight LLMs의 안전성 문제를 해결하기 위한 새로운 모듈인 SafeGene을 제안합니다. SafeGene은 크로스-태스크 안전 전이를 위한 재사용 가능한 안전 어댑터 모듈로, 태스크 특화 업데이트와 분리하여 안전성을 보존하도록 설계되었습니다. 이를 통해 사용자와 데이터의 변화에 따라 모델을 효율적으로 업데이트하면서도 안전성을 유지할 수 있는 방법을 제공합니다.

- **Technical Details**: SafeGene은 안전성이 마모되지 않도록 하며, 각 아키텍처 호환 모델 계열 내에서 태스크 간 재사용을 가능하게 합니다. 데이터 인식 레이어 선택을 통해 태스크-전이 가능한 안전 벡터를 유지하며, 각 다운스트림 태스크 적응 모델에 대해 선택된 레이어의 스칼라 계수만 소량 조정합니다. 이러한 과정으로 SafeGene은 타겟 모델의 태스크 업데이트를 손상시키지 않으면서 새로운 타겟 분포에 대한 안전성을 회복할 수 있습니다.

- **Performance Highlights**: 실험 결과, SafeGene을 적용한 모델은 다운스트림 성능을 유지하면서도 유해 응답 비율을 11.48% 감소시켰습니다. 또한, SafeGene은 기존의 안전 적응 기법들보다 우수한 안전-유틸리티 트레이드오프를 달성하며, 다양한 모델 패밀리 및 태스크에서 효과적인 성능 향상을 나타냈습니다.



### DiBS: Diffusion-Informed Branch Selection (https://arxiv.org/abs/2606.06518)
Comments:
          12 pages, 6 figures, 3 tables

- **What's New**: 이번 연구에서는 Sudoku 문제를 해결하기 위한 새로운 접근 방식인 DiBS(Diffusion-Informed Branch Selection)를 제안합니다. 기존의 휴리스틱과 깊이 학습 솔버의 한계를 극복하기 위해, DiBS는 분산(diffusion) 모델을 활용하여 솔버의 가지 선택(search process)에서 효과적으로 가이드를 제공합니다. 이를 통해 솔버의 완전성을 유지하면서도, 탐색 비용을 현저히 줄일 수 있음을 보여주었습니다.

- **Technical Details**: DiBS는 현재의 부분 할당(partial assignment) 아래에서 후보 값의 순위를 매기는 방식을 채택하며, 경량의 일관성 신호(lightweight consistency signal)를 더하여 보다 빠르고 효율적인 탐색을 가능하게 합니다. 기존의 학습 기반 분기 선택 방법들과 달리, DiBS는 후보 값을 재정렬만 수행하여 정책을 전부 학습하거나 가지를 잘라내지 않으면서도 완전성을 보장합니다. 이론적으로, 더 나은 가지 순서가 잘못된 서브트리 탐색 비용을 줄일 수 있음을 증명하고, DiBS가 최적 정책에 접근하는 방식을 보여줍니다.

- **Performance Highlights**: Royle의 17단서(Sudoku benchmark) 실험 결과, DiBS는 전반적인 탐색 비용을 큰폭으로 감소시켰습니다. 특히 노드 및 백트랙(backtracks) 수와 긴 꼬리(long-tail) 백분위수에서 두드러진 성과를 기록했습니다. 이 결과는 결정적인 경우에 학습된 전역 가이드가 효과적이라는 것을 입증하며, DiBS가 기존의 강력한 휴리스틱 기준보다 우수함을 확인시켜 줍니다.



### Detecting and Mitigating Bias by Treating Fairness as a Symmetry Operation (https://arxiv.org/abs/2606.06514)
Comments:
          8 pages, 7 figures

- **What's New**: 본 논문에서는 머신러닝 시스템이 사회적 그룹에 미치는 편향(bias) 문제를 다루고 있다. 편향을 대칭 깨짐(symmetry breaking)으로 형식화하며, 민감 속성을 전환하더라도 출력 결과가 불변할 때 분류기를 공정(fair)하다고 정의한다. 이 연구는 고전적 방법론보다 더 직관적이고 구조적인 불변성을 요구하는 접근법을 제안하며, 다양한 데이터셋에서 성능 검증을 수행하였다.

- **Technical Details**: 연구에서는 임의의 분류기(f):X→[0,1]에 대해 정의하며, 민감 속성과 관련된 자료(x:[xm; xs])를 포함한다. 특정 대칭 변환 T에 따라 분류기가 대칭적인지를 확인하는 점위의 위반(v)과 집단 위반(V) 측정 방식을 제시하고, 데이터를 기반으로 실제 샘플을 통해 위반 측정을 근사화하는 방법을 설명한다. 이 프레임워크는 인과 그래프에 대한 지식 없이도 적용 가능하며, 컴퓨팅 비용이 낮은 특징을 가진다.

- **Performance Highlights**: 제안된 프레임워크는 노이즈, 상관관계 및 편향의 수준이 다양한 네 개의 합성 데이터셋에서 90% 이상의 위반 감소를 달성하였다. 하지만 정확도는 약 5%의 비용이 발생하였다. 또한, 제안된 방법은 임의의 민감 속성에서 일반화 가능하므로, 고전적 벤치마크에서의 차별적 요인을 대체할 수 있는 가능성을 보인다.



### How reliable are LLMs when it comes to playing dice? (https://arxiv.org/abs/2606.07515)
- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)의 확률적 추론 능력을 통제된 벤치마킹 연구를 통해 조사했습니다. 두 개의 데이터셋을 만들었으며 각각 일반적인 문제와 직관에 반하는 문제로 구성되어, 휴리스틱 추론(heuristic reasoning)을 유도하도록 설계되었습니다. 8개의 최신 모델을 평가한 결과, 일반 문제에는 평균 0.96의 정확도를 달성했으나, 직관에 반하는 문제에서는 0.59로 낮아지는 성과를 보였습니다.

- **Technical Details**: 연구에서는 표준 문제와 직관에 반하는 문제에서 LLM의 성능을 측정하고, 모델 간의 행동 차이를 분석했습니다. 특히, 모델의 입력 패턴에 따라 발생하는 토큰 편향(token bias)의 영향을 고려하였으며, 잘못된 제안이 모델의 응답에 미치는 영향을 실험적으로 평가했습니다. 이를 통해 모델들이 코너케이스나 잘못된 추론의 영향을 최소화하지 못하고 있다는 것을 확인할 수 있었습니다.

- **Performance Highlights**: 최신 모델들이 고차원 수학 문제를 해결하는 데 뛰어난 성과를 보였지만, 직관에 반하는 문제에서는 성과가 크게 저하되었습니다. 이로 인해 특정 확률적 추론의 측면이 기존의 수학적 벤치마크에서는 포착되지 않음을 알 수 있었습니다. 연구 결과, 현재의 LLM은 진정한 추론 기계가 아닐 뿐만 아니라, 교육 데이터로부터 물려받은 결점이 있으며, 입력 프롬프트에 매우 민감한 특성을 보여주었습니다.



### MemDreamer: Decoupling Perception and Reasoning for Long Video Understanding via Hierarchical Graph Memory and Agentic Retrieval Mechanism (https://arxiv.org/abs/2606.07512)
- **What's New**: 현재의 Vision-Language Models는 몇 시간에 걸친 비디오를 처리하는 데 어려움을 겪고 있습니다. 이를 해결하기 위해, 연구진은 MemDreamer를 소개하며, 이는 인식과 추론을 분리하여 긴 비디오 이해를 에이전틱 탐색 과정으로 전환합니다. MemDreamer는 비디오를 점진적으로 스트리밍하며 Hierarchical Graph Memory를 구축하는 플러그 앤 플레이(framework) 구조입니다.

- **Technical Details**: 이 프레임워크는 의미 추상화를 위한 세 가지 주요 계층으로 구성된 구조로, 시공간(spatiotemporal) 및 인과관계(causal relations)를 포착하는 기본 그래프에 의해 뒷받침됩니다. 추론 과정에서는 Agentic Tool-Augmented Retrieval을 사용하여 계층을 내비게이션하고, 노드를 탐색하며, Observation-Reason-Action 루프를 통해 논리적인 간선을 통과합니다.

- **Performance Highlights**: 실험 결과, MemDreamer는 네 가지 주요 벤치마크에서 SOTA(State Of The Art) 결과를 달성하였고, 인간 전문가와의 격차를 단 3.7점으로 좁혔습니다. 이 모델은 전체 맥락의 2%로 추론 컨텍스트 윈도우를 제한하면서도 12.5점의 절대 정확도 증가를 보여줍니다. 또한, 통계적 분석을 통해 VLM의 논리 추론 성능과 긴 비디오 이해 벤치마크 간에 강한 양의 선형 상관관계를 발견하였습니다.



### Sparse Subspace-to-Expert Sharing for Task-Agnostic Continual Learning (https://arxiv.org/abs/2606.07500)
Comments:
          19 pages. arXiv admin note: text overlap with arXiv:2601.17616

- **What's New**: 본 연구에서는 대규모 언어 모델(LLMs)에서 지속적인 학습(contiual learning)의 문제를 해결하기 위해 Mixture of Sparse Experts for Task Agnostic Continual Learning (SETA)라는 새로운 프레임워크를 제안합니다. SETA는 기존의 프레임워크가 가진 플라스틱성-안정성 딜레마(plasticity-stability dilemma)를 해결하기 위해 적응형 희소(subspace decomposition) 전문가 모듈을 도입합니다. 이 접근법은 특정 작업(task) 지식과 공유(shared) 능력을 구분하여 각 작업에 따라 독립적으로 지식을 관리할 수 있도록 합니다.

- **Technical Details**: SETA는 전문가 모듈을 통해 작업 간의 경쟁 없이 독립적으로 정보를 저장하고 업데이트할 수 있게 설계되었습니다. 이 시스템은 Split-on-Share(SoS) 메커니즘을 활용하여, 작업 간 공통으로 사용되는 매개변수는 공유 전문가로 분류되며, 이들은 적응형 탄력적 고정(adaptive elastic anchoring)과 라우팅 인식 정규화(routing-aware regularization)를 통해 보호됩니다. 반면 특정 작업에만 영향을 받는 매개변수는 고정된 독립 전문가로 유지되어, 이전 지식을 보존할 수 있게 합니다.

- **Performance Highlights**: 여러 도메인 특정 벤치마크에서의 광범위한 실험을 통해 SETA는 최신 지속적 학습 기법들과 비교해 경쟁력 있는 성능을 발휘하는 것으로 나타났습니다. 특히, 초기 작업(Task) 지식을 강력하게 유지하며, LLaMA-2 7B와 Qwen3-4B와 같은 대규모 모델에서 우수한 성능을 보여주었습니다. 이러한 결과는 SETA의 구조적 접근이 기존의 문제를 효과적으로 해결할 수 있음을 입증합니다.



### Twelve quick tips for designing AI-driven HPC workflows (https://arxiv.org/abs/2606.07491)
Comments:
          12 pages, 1 figure. Formatted using the bioRxiv LaTeX preprint style

- **What's New**: 본 논문에서는 예측 가능하고 결정적인 성격을 가진 전통적인 고성능 컴퓨팅(HPC) 파이프라인에서 AI 기반의 새로운 컴퓨팅 패러다임으로의 전환을 다룹니다. AI 기반 워크플로우의 특성은 반복적이고 데이터 중심이며 확률적이라는 점에서 기존의 파이프라인과 다른 설계와 실행 방식을 필요로 합니다. 이 가이드는 연구자들이 효율적이고 확장 가능하며 재현이 가능한 AI 기반 HPC 워크플로우를 설계하는 데 도움이 되도록 12가지 팁을 제공합니다.

- **Technical Details**: AI 기반의 HPC 워크플로우는 전통적인 HPC와는 다른 요구사항을 반영해야 합니다. AI는 별도의 단계가 아니라 워크플로우의 핵심 요소로 통합되어야 하며, 데이터 이동 문제와 훈련, 추론, 시뮬레이션 단계의 명확한 분리를 통해 자원을 효율적으로 활용해야 합니다. 또한, 이러한 단계들은 서로 다른 컴퓨팅 자원에 대응하여 설계되어야 하며, 이질적인 리소스의 활용이 필요합니다.

- **Performance Highlights**: AI 기반 워크플로우는 전체적인 워크플로우 차원에서 병렬성을 활용하여 자원의 효율적인 사용을 도모할 수 있습니다. 예를 들어, 고유한 데이터 샘플 처리, 모델 구성 평가 및 하이퍼파라미터 스윕 등은 독립적으로 실행될 수 있습니다. 이러한 병렬성은 HPC 스케줄러와 잘 맞물리는 특성을 지니고 있으며, 고성능 구현과 함께 대규모 매개변수 공간 탐색을 지원합니다.



### Supervision versus Demonstration-Based In-Context Learning for Multiword Expression Classification (https://arxiv.org/abs/2606.07479)
Comments:
          Accepted to ACL SRW 2026

- **What's New**: 이번 연구는 터키어의 관용적 경량 서술 동사 구성(light verb constructions, LVCs)을 탐구하며, 이들이 특정한 표면 형태를 공유하지만 단일한 부분적 관용(predicate)으로 작용하는 것에 주목합니다. 연구팀은 LVC 감지를 이진 분류 작업으로 틀을 잡고, 이를 통해 감독된 BERTurk 언어 인코더와 다양한 인스트럭션 튜닝된 LLM(대형 언어 모델) 성능을 비교했습니다. 결과는 자세한 프롬프트 민감성을 보여주며, 주의 깊게 구성된 프롬프트가 LVC 감지를 개선하는 데 중요한 역할을 한다는 것을 증명했습니다.

- **Technical Details**: 본 논문은 터키어 LVC 감지를 명시적으로 다음의 세 가지 프롬프트 방식(제로샷, 원샷, 퓨샷)으로 평가하여, 각 모델의 성능을 비교했습니다. 초기 결과에 따르면, 제로샷에서 LLM들은 부정적인 예시에서 잘 작동하지만 LVC 감지는 낮은 재현율을 보이며, 원샷 프롬프트에서는 LVC 감지를 급격히 개선시키는 반면 강한 모델 특정 편향성을 유도할 수 있음을 알았습니다. 최종적으로 퓨샷 프롬프트가 보다 안정적인 성능을 보여주며, 특히 GPT-OSS-20B와 Qwen 2.5-14B 모델에서 경쟁력이 있음을 보여주었습니다.

- **Performance Highlights**: 연구 결과는 LVC 감지에서의 프롬프트 민감성이 도드라지며, 감독된 BERTurk 기준과 비교해 인스트럭션 튜닝된 LLM이 동등하거나 그 이상의 성능을 보여줄 수 있다는 가능성을 제시합니다. 이 모델들은 특정한 예시를 제공할 때 LVC 감지에서 우수한 성과를 나타내지만, 잘못된 예측 또한 발생할 수 있음을 시사합니다. 따라서 본 연구는 터키어에서 다중 단어 표현 처리의 복잡성 및 모델링에서 유의미한 최소한의 입력이 얼마나 중요한지를 강조합니다.



### Graph Neural Network leveraging Higher-order Class Label Connectivity for Heterophilous Graphs (https://arxiv.org/abs/2606.07475)
- **What's New**: 이 논문에서는 heterophilous (이질적) 그래프에서의 노드 분류 성능 향상을 위해 새로운 Label Context Classifier (LCC)를 제안합니다. 기존의 Graph Convolutional Networks (GCNs)는 동일 클래스 간의 관계를 잘 포착하지만, 다양한 클래스 간의 다층 연결을 고려하지 못하는 한계가 있습니다. LCC는 네 가지 유형의 label walks를 사용하여 고차원 클래스 레이블 연결성을 캡처하는 방식으로 설계되었습니다.

- **Technical Details**: LCC는 label context embeddings를 생성하여 대상 노드의 클래스를 추정합니다. 네 가지 기본적인 label walk 유형인 forward walk, backward walk, sibling walk, guardian walk를 통해 클래스 레이블 간의 다양한 연결성을 포착합니다. 이 접근 방식은 LCC와 기존의 GNN을 결합하여 이를 학습하고, 추가적인 모델 훈련 없이 그 중요성을 적응적으로 학습할 수 있습니다.

- **Performance Highlights**: 실험 결과 LCC가 통합된 GNN이 최신의 SOTA 방법보다 뛰어난 성능을 보여줍니다. 이 레이블 컨텍스트 임베딩은 heterophilous directed (이질적 방향 그래프) 그래프에서 노드 분류 성능을 향상시키는 것으로 나타났습니다. LCC를 통해 고차원 클래스 레이블 연결성을 효과적으로 캡처함으로써, 다양한 실제 애플리케이션의 노드 분류에서 개선된 결과를 보입니다.



### Whisper Hallucination Detection and Mitigation via Hidden Representation Steering and Sparse AutoEncoders (https://arxiv.org/abs/2606.07473)
- **What's New**: 본 연구에서는 Whisper라는 자동 음성 인식(ASR) 모델이 비음성 오디오에 대해 일관된 전사(hallucination)를 생성하는 문제를 다루고 있습니다. 연구진은 Whisper의 내부 표현을 통해 이러한 일관성을 탐지하고 완화할 수 있는지를 조사하였습니다. 특히, Sparse AutoEncoder(SAE)를 이용한 두 가지 방법을 제안하고, 이를 통해 hallucination 비율을 상당히 낮출 수 있었음을 보여주었습니다.

- **Technical Details**: Whisper 모델은 Transformer 기반의 ASR 모델로, 680,000시간 분량의 약한 감독 데이터로 훈련되었습니다. 연구진은 모델의 비음성과 언어 회귀 전사를 결정하기 위해 activation-space steering과 SAE latent-space steering 기법을 채택하였습니다. 이를 통해 내부의 활성화 및 SAE 잠재 표현을 분석하며, 특정 단층에서 정보의 가시성이 더욱 높아지는 경향이 있음을 발견했습니다.

- **Performance Highlights**: SAE 기반 조정 기법을 통해 Whisper 모델의 hallucination 비율을 Whisper small에서는 72.63%에서 14.11%로, Whisper large-v3에서는 86.88%에서 27.33%로 줄였습니다. 이러한 기법들은 미세 조정 없이도 훌륭한 성능을 보여 주었으며, 기존의 ASR 메트릭을 유지하면서도 hallucination 감소에 효과적임을 나타냈습니다.



### Planning-aligned Token Compression for Long-Context Autonomous Driving (https://arxiv.org/abs/2606.07464)
Comments:
          9 pages

- **What's New**: COMPACT-VA는 planner-aligned working memory 프레임워크로, 균형 잡힌 메모리 효율성을 통해 운전 성능을 최적화합니다. 이 연구는 의사결정에 중요한 정보를 유지하기 위해 과거 궤적과 학습된 계획 의도를 기반으로 하는 압축 메커니즘을 도입했습니다. 현재의 token 압축 방법들이 계획 목표와는 분리되어 있다는 문제점을 해결하기 위한 접근을 제공합니다.

- **Technical Details**: 이 모델은 conditional VQ-VAE(조건부 벡터 양자화 변분 오토인코더)를 기반으로 하며, 절차적 최적화를 통해 중요한 과거 정보를 어떻게 보존할지를 학습합니다. 압축된 메모리는 최근의 토큰에는 더 많은 비율을 주고, 더 먼 과거에 대해서는 적은 비율로 배치됩니다. 이 구조는 self-attention을 통해 과거 궤적 컨텍스트와 미래 운전 의도에 따라 조정이 가능합니다.

- **Performance Highlights**: COMPACT-VA는 일정한 token 예산 하에 68.3%의 성공률을 달성하며, 기존 모델보다 6.3% 향상된 결과를 보입니다. 안전에 중요한 roll-through를 22% 줄였으며, 모델의 전반적인 운전 능력은 유지하면서 3.3배의 속도 향상과 2.7배의 메모리 절감을 이끌어냈습니다. 각 세부 요소들이 성능 향상에 기여했다는 것은 ablation 테스트를 통해 확인되었습니다.



### PaperFlow: Profiling, Recommending, and Adapting Across Daily Paper Streams (https://arxiv.org/abs/2606.07454)
Comments:
          48 pages, 13 figures, 22 tables

- **What's New**: 이 논문은 종래의 정적 추천 시스템을 넘어서, 개별 사용자의 연구 관심사를 반영하고 업데이트하는 동적 프레임워크인 PaperFlow를 소개합니다. PaperFlow는 세 가지 연계된 단계로 구성되어 있으며, 이는 Profiling, Recommending, Adapting입니다. 각 단계는 연구자가 진행하는 일일 독서 경험을 통합하고 개별 사용자의 변화하는 관심사를 모델링하는 데 중점을 두고 있습니다.

- **Technical Details**: PaperFlow는 연구자의 프로파일을 구조화된 형태로 유지하며, 연구 방향, 주제별 가중치, 저자 및 기관 우선 순위 등의 정보를 캡처합니다. 추천 단계에서는 날짜에 맞는 후보지의 후보지를 정렬하며 multi-signal aggregation을 통해 추천합니다. Adapting 단계에서는 사용자의 피드백 신호를 반영하여 사용자 상태를 지속적으로 업데이트하고, 연구자의 관심사 변화를 모델링합니다.

- **Performance Highlights**: PaperFlow는 기존의 다섯 가지 추천 시스템과의 실험을 통해 가장 뛰어난 성과를 보여주었습니다. 구체적으로, 자동 메트릭과 전문가의 판단 간 정렬을 검증하기 위한 인간 평가 프로토콜을 사용하여, Oracle 기반의 순위에서 가장 높은 점수를 기록했습니다. 또한, PaperFlow는 사용자의 시뮬레이션된 독서 선택과 높은 행동 일치를 보이며, 최상의 블라인드 인간 평가 점수를 자랑합니다.



### TEVI: Text-Conditioned Editing of Visual Representations via Sparse Autoencoders for Improved Vision-Language Alignmen (https://arxiv.org/abs/2606.07451)
Comments:
          20 pages, 13 figures, 14 tables

- **What's New**: 이 논문에서는 TEVI라는 새로운 프레임워크를 제안합니다. TEVI는 자막이 이미지 임베딩에서 무엇을 유지해야 하는지를 결정하는 신호로 작용합니다. 이를 통해 자막에 설명된 속성은 유지하고 다른 속성은 버리도록 설정할 수 있습니다. 이 방법은 이미지-텍스트 정렬을 개선하는 데 효과적입니다.

- **Technical Details**: TEVI는 희소 자동인코더(sparse autoencoders)를 사용하여 이미지 임베딩을 구성 요소 개념으로 분해하고, 조건부 모듈을 훈련하여 재구성에 사용할 SAE 잠재 변수를 선택합니다. 이를 통해 이미지의 특정 속성을 살리고, 필요한 정보를 유지하면서 다른 속성에 대한 정보를 폐기할 수 있습니다. 또한, 제어된 실험을 통해 TEVI의 효과를 증명했습니다.

- **Performance Highlights**: TEVI를 자연 이미지에 대해 훈련된 CLIP 모델에 적용한 결과, 짧은 자막(MS COCO, Flickr)과 긴 자막(DOCCI, IIW) 벤치마크 모두에서 검색 성능이 향상되었습니다. 특히 긴 자막 기준에서 더 큰 성능 향상을 보였으며, 이것은 풍부한 자막이 수정에 강력한 신호를 제공한다는 것을 시사합니다. 또한, RoCOCO 벤치마크에서 언어적 왜곡에 대한 강한 견고함을 보여주었습니다.



### Re-imagining ISO 26262 in the Age of Autonomous Vehicles: Enhancing Controllability through Transferability and Predictability (https://arxiv.org/abs/2606.07437)
- **What's New**: 이 논문은 ISO 26262 표준의 ‘Controllability’ 개념을 자율주행차(AV)를 위한 두 가지 새로운 차원인 Transferability와 Predictability로 세분화합니다. 이 새로운 개념은 자율주행차의 안전성을 평가하는 데 필요한 기준과 방법론을 제공하여 기존 기준을 보완합니다. Transferability는 AV 시스템이 안전한 방식으로 제어를 이전할 수 있는 능력을 측정하고, Predictability는 외부의 사용자들이 AV의 행동을 얼마나 예측할 수 있는지를 나타냅니다.

- **Technical Details**: ISO 26262 표준은 차량의 기능적 안전성을 보장하기 위해 ‘Severity(심각도)’, ‘Exposure(노출도)’, ‘Controllability(통제 가능성)’라는 리스크 평가를 기반으로 합니다. 이 논문에서는 자율주행차의 환경에서 Controllability를 체계적으로 재해석하고 보완하기 위한 수학적 프레임워크를 제시합니다. Transferability는 고장 허용 시간 내에서의 통제 이전 능력을 측정하는 반면, Predictability는 외부 도로 사용자가 AV의 행동을 얼마나 정확하게 예상할 수 있는지를 수치화합니다.

- **Performance Highlights**: Transferability와 Predictability는 자율주행차의 안전성을 평가하기 위한 SOTIF(ISO 21448)와 조화를 이루는 핵심 성과 지표로 제안됩니다. 이 논문은 Transferability의 최소 위험 조작 성능을 평가하기 위한 접근 방식을 공식화하며, Predictability는 외부 도로 사용자의 안전성을 정량화할 수 있는 새로운 차원으로 설정합니다. 제안된 기준과 검증 방법론은 ISO 26262와 SOTIF 안전 생명주기 프로세스 내에서 채택될 수 있는 신뢰성 있는 메트릭스를 제공합니다.



### Watch, Remember, Reason: Human-View Video Understanding with MLLMs (https://arxiv.org/abs/2606.07433)
- **What's New**: 이 논문은 멀티모달 대규모 언어 모델(MLLMs)이 비디오 이해 분야를 어떻게 혁신하고 있는지를 다루고 있습니다. 연구는 짧은 클립에서 긴 멀티모달 비디오로 넘어가며, 이러한 변화는 모델이 희소한 증거, 긴 범위 의존성, 멀티모달 정렬, 그리고 제한된 계산 자원 하에서 신뢰할 수 있는 추론을 다뤄야 함을 요구합니다. 또한, 비디오 MLLMs가 단순한 기준점으로서 비디오 작업을 취급하는 대신, 인간이 비디오를 이해하는 기제를 중심으로 분석하는 것을 목표로 합니다.

- **Technical Details**: 논문은 '보기(watching)', '기억하기(remembering)', '추론하기(reasoning)'라는 세 가지 기능 기반으로 LLM 기반 비디오 이해를 분석합니다. 이 구조는 비디오 이해 시스템을 입체적으로 바라보고, 지각적 표현, 기억 상태, 추론 과정을 포괄하는 공식을 제공합니다. 또한, 효율적인 비디오 처리, 메모리 모델링, 스트리밍 이해 및 신뢰성 있는 추론의 도전과제를 식별합니다.

- **Performance Highlights**: 논문은 여러 비디오 응용 영역을 다루며, 각 비디오 유형에 따라 요구되는 지각, 기억, 그리고 추론방식이 다름을 보여줍니다. 특히, 연구는 교육, 의료, 운동, 서사 비디오 등 구체적인 도메인을 살펴보며, 다양한 훈련 데이터셋 및 평가 지표를 정리합니다. 마지막으로, 확장 가능하고 메모리에 민감하며 증거 기반의 비디오 인공지능을 위한 미래 방향을 제시합니다.



### The Masked Advantage: Uncovering Local-Language Access to Cultural Knowledge in LLMs (https://arxiv.org/abs/2606.07422)
- **What's New**: 이번 연구는 대규모 언어 모델(LLM)의 다언어 시스템으로서의 발전을 탐구하며, 지역 문화 지식(access to local cultural knowledge)이 영어 쿼리 혹은 현지 언어 중 어떤 언어를 통해 더 잘 접근되는지를 비교합니다. 기존 연구는 병렬 템플릿 기반 질문에 의존해 왔으나, 이 연구는 실제 지역 벤치마크 및 지역 자료에서 수집한 실질적인 문화 질문을 기반으로 한 제어된 프레임워크를 제시합니다. 언어 능력(proficiency)과 문화 지식 접근(access)을 분리하는 새로운 측정을 위한 모델을 제안합니다.

- **Technical Details**: 연구진은 문화 특이적 질문(culture-specific questions)과 문화 비특이적 질문(culture-agnostic questions)으로 질문 유형을 나누어 영어와 지역 언어에서의 모델 성능을 비교했습니다. 이를 통해 원시 정확도(raw accuracy) 대신 1PL 항목 응답 이론 모델(Item Response Theory model)을 사용하여 언어 능력과 문화 지식 접근을 분리할 수 있었습니다. 총 13개 지역에서 약 80개의 모델을 사용하여 분석을 수행하며, 이 과정에서 문화 지식의 접근성이 다양한 쿼리 언어에 따라 어떻게 달라지는지를 조사합니다.

- **Performance Highlights**: 모델들은 문화 비특이적 질문에서 일관된 영어 능력을 나타냈으며, 이는 영어가 높은 수준의 언어 능력(presumably due to higher proficiency)으로 인해 발생했습니다. 그러나 지역 언어에 대한 접근 방법은 문화 특이적 질문에서 긍정적인 경향을 보였으며, 이는 제한된 언어 능력으로 인해 가려질 수 있습니다. 결론적으로, 지역 언어의 성과는 문화 지식을 저해하는 영어 능력과 비교했을 때 더 나은 접근성을 제공하는 것으로 나타났습니다.



### Socratic-SWE: Self-Evolving Coding Agents via Trace-Derived Agent Skills (https://arxiv.org/abs/2606.07412)
Comments:
          21 pages, 5 figures. Under review

- **What's New**: Socratic-SWE는 LLM(대규모 언어 모델) 기반 소프트웨어 엔지니어링(SWE) 에이전트를 위한 새로운 프레임워크입니다. 이 방법은 과거의 문제 해결 경로(historical solving traces)를 활용하여 에이전트의 훈련 신호를 생성합니다. 기존의 데이터 생성 방식과 다르게, Socratic-SWE는 에이전트의 기술(skill)을 구조적으로 증류하여 반복적인 실패 및 효과적인 수리 패턴을 요약합니다.

- **Technical Details**: Socratic-SWE는 클로즈드 루프(self-evolution framework)를 통해 에이전트의 진화 과정(evolutionary process)을 지원합니다. 이 프레임워크는 이전의 해결 경로를 보상 계산의 증거로만 사용하지 않고, 이를 구조화된 에이전트 기술로 개발하여 실제 레포지토리에서 목표 지향적인 수리 작업을 생성합니다. 생성된 작업은 실행 기반 검증(execution-based validation)을 통해 확인되며, solver-gradient alignment 보상으로 평가됩니다.

- **Performance Highlights**: Socratic-SWE는 SWE-bench Verified, SWE-bench Lite, SWE-bench Pro, Terminal-Bench 2.0을 포함한 여러 벤치마크에서 자체 진화 기준선(self-evolving baselines)보다 지속적으로 개선된 성과를 보여줍니다. 특히 SWE-bench Verified에서 세 번의 반복 후 50.40%를 달성하며, 해결 경로가 자기 진화 SWE 에이전트를 위한 확장 가능한 기초(substrate)로 사용될 수 있음을 시사합니다.



### A Comprehensive Anatomy of Human and DeepSeek-R1 LLM Mathematical Reasoning (https://arxiv.org/abs/2606.07410)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)인 DeepSeek-R1-0120의 "Aha moments" 발생이 실제 추론인지 모방인지에 대한 의문을 제기합니다. 연구자들은 30개의 문제를 대상으로 모델과 인간의 추론을 비교하며, 10,247단계로 구성된 추론 단계를 다섯 가지 기능 범주로 분석하였습니다. 현저한 구조적 차이점을 발견했으며, 인간의 해결법은 분석과 추론 간의 체계적 전환을 유지하는 반면, DeepSeek-R1은 중간 결과에 반복적으로 접근하고 불필요한 검증을 수행하는 경향을 보였습니다.

- **Technical Details**: DeepSeek-R1의 추론 단계는 5가지 기능적 범주로 나뉘어져 있으며, 각 단계는 텍스트 서명으로 검증할 수 있습니다. 이 연구는 CoT의 구조적 역학과 제어 메커니즘이 진정한 문제 해결과 단순한 모방을 어떻게 구별하는지를 탐구합니다. 주요 발견은 반사(reflection)가 특정한 맥락에서만 효과적이며, 탐색적 행동의 사용 패턴이 실패한 추적(trace)에서 과소 또는 과잉 사용된다는 것입니다.

- **Performance Highlights**: 이 연구는 LLM의 추론 품질이 단순히 반사의 양이 아니라, 반사가 일관되게 적절한 논리적 규모에서 나타나는 여부에 달려있음을 강조합니다. 향후 평가 및 트레이닝 개선 방향으로는 교차 추적 안정성(cross-trace stability)을 측정하고, '스피닝 휠(spinning-wheel)' 추적에 대한 패널티 부여를 포함해야 한다고 제안합니다. 따라서, 주목할 점은 단순한 외형적 복잡성이 추론 품질의 증거가 아니라는 것입니다.



### Impact of Synthetic Lesional MR Images in Automated Focal Cortical Dysplasia Detection in Low-Data Scenarios (https://arxiv.org/abs/2606.07381)
- **What's New**: 이 연구에서는 Focal Cortical Dysplasia (FCD)의 자동 탐지를 위한 합성 MRI 데이터 생성 방법을 제안합니다. 이를 통해 수동 주석의 필요성을 줄이려는 목표를 가지고 있으며, 합성 데이터의 현실성을 평가합니다. 131명의 FCD 환자와 90명의 건강한 대조군으로부터의 MRI 데이터를 사용하여 연구하였습니다.

- **Technical Details**: T1 가중화 (T1-weighted, T1w) 및 T2 가중화 Fluid-Attenuated Inversion Recovery (FLAIR) MRI 스캔을 통한 데이터 수집이 이루어졌습니다. Binary FCD 마스크를 조건화하여 생성적 네트워크를 활용하여 합성 MRI를 생성하였으며, 두 명의 신경영상의사가 실제 이미지와 합성 이미지를 구별했습니다. 세 가지 nnU-Net 모델을 훈련시켜 FCD 탐지를 수행했습니다.

- **Performance Highlights**: 전문가들은 실제 이미지와 합성 이미지를 분별하는 데 제한된 능력을 보였으며, T1w의 분류 정확도는 60%, FLAIR는 70%였습니다. 합성 데이터를 사용한 자동 FCD 탐지는 민감도를 8.14% 증가시켰고, 확대된 실제 데이터 모델은 민감도를 73.8%까지 개선시켰습니다. 합성 데이터 기반 augmentation은 라벨된 데이터 요구를 약 20% 줄이면서도 동등한 민감도를 유지하는 것으로 나타났습니다.



### Do Coding Agents Deceive Us? Detecting and Preventing Cheating via Capped Evaluation with Randomized Tests (https://arxiv.org/abs/2606.07379)
- **What's New**: CapCode와 CapReward라는 두 가지 새롭고 혁신적인 접근 방식을 소개합니다. CapCode는 코딩 데이터셋을 구성하는 새로운 프레임워크로, 비정상적인 높은 성능을 감지할 수 있는 점수 해석 방식을 제공합니다. CapReward는 CapCode 원칙을 기반으로 하는 보상 설계로, 모델이 의도된 작업 사양을 더 잘 준수하도록 유도합니다.

- **Technical Details**: CapCode는 테스트의 성과 한계를 1 미만으로 의도적으로 설정하여 'cheating'을 감지합니다. 이 프레임워크는 모델의 비치팅(pass rate) 성능을 잘 정의된 코딩 테스트에서 100%에 도달할 수 없게 합니다. CapReward는 비치팅 행동을 받은 성과를 cap 이하로 제한하여 높은 성과를 가치 있게 평가하고 비정상적으로 높은 점수에 패널티를 부여합니다.

- **Performance Highlights**: CapCode는 여러 데이터셋에서 성과를 추적하면서 비정상적인 높은 성과를 감지하였고, CapReward는 강화 학습에서 비정상적인 행동을 줄여 의도된 과제를 보다 잘 따르는 모델을 생성했습니다. 이 실험들은 두 가지 방법이 성능을 저해하지 않으면서도 효과적으로 'cheating'을 감지 및 저감할 수 있음을 보여줍니다.



### Mitosis Detection in the Wild: Multi-Tumor and Context-Aware Generalization in the MIDOG 2025 Challeng (https://arxiv.org/abs/2606.07368)
- **What's New**: MIDOG 2025 챌린지는 자동 분열 검사(mitosis detection) 분야에서 새로운 초점으로, 전통적인 핫스팟(hotspot) 영역을 넘어 무작위(sampled randomly)로 선정된 조직 영역(tissue regions) 및 어려운 영역(challenging regions)에서의 검출을 요구합니다. 이 챌린지는 365개의 케이스를 포함하여, 다양한 종(인간, 개, 고양이)의 암 유형과 여러 스캐닝 플랫폼에서 수집된 데이터를 활용하여 알고리즘의 성능을 평가하고자 했습니다. 이는 현실적인 임상 적용(clinical application)에서 모델의 강건함을 극대화하기 위한 노력으로, 현재의 병리학적 검사에서 직면하고 있는 주요 문제들을 해결합니다.

- **Technical Details**: MIDOG 2025 챌린지에서는 총 18개 팀이 검출 트랙에 제출했으며, F1 점수는 최대 0.740에 달했습니다. 비정상적인 분열을 보이는 세포의 분류(atypical mitotic figures, AMFs)가 두 번째 트랙으로 도입되었으며, 여러분 21개의 제출이 있었고 균형 잡힌 정확성(balanced accuracy) 값은 0.908에 이르렀습니다. 이 트랙에서의 성능 분석은 전통적인 핫스팟 지역에서는 신뢰할 수 있는 성능을 보였으나, 도전적인 ROI에서는 성능 저하가 발생한다는 사실을 드러냈습니다.

- **Performance Highlights**: 성능 분석 결과, 통계적으로 다루기 어려운 지역에서 잘못된 예측(false positive) 비율이 세 배 증가됨을 보여주었습니다. AMF에 대한 앙상블(ensembling) 기법을 평가한 결과, F1 점수와 balanced accuracy에서 각각 평균 1.5 및 1.3 포인트의 향상이 나타났습니다. 그러나 TTA(transformational training augmentation)는 유의미한 개선이 없었습니다. 이러한 결과는 현실 세계에서의 mitosis 검출이 여전히 상당한 도전 과제임을 입증합니다.



### A robust PPG foundation model using multimodal physiological supervision (https://arxiv.org/abs/2606.07365)
- **What's New**: 본 논문에서는 PPG(P Photoplethysmography) 기초 모델을 제안하며, 고품질 선행 학습 데이터 없이도 동반된 ECG(전기생리학적 심전도) 및 호흡 신호를 활용하여 대비 샘플을 선택하는 새로운 방법을 소개합니다. 기존의 모델들은 전처리된 데이터를 필요로 하거나 저작권이 있는 데이터에 의존했으나, 본 연구는 소음이 많은 PPG 데이터를 효과적으로 학습할 수 있는 가능성을 제시합니다. 이 모델은 고유한 구조 덕분에 기존 최첨단 방법들보다 데이터 수가 적음에도 불구하고 다양한 다운스트림 작업에서 성능을 향상시켰습니다.

- **Technical Details**: 제안하는 모델은 다중 모드 사전 학습 프레임워크를 사용하여, PPG 기초 모델이 PPG 신호만 필요로 하면서도 사전 학습 동안 다른 생리 신호(ECG와 호흡 신호)를 감독 신호로 활용합니다. 이러한 접근법은 소비자 수준의 노이즈에 대한 내구성을 높이면서도 배치나 감지의 복잡성을 증가시키지 않습니다. 연구자들은 PPG 데이터에 대한 세밀한 평가를 위해 피험자별 선형 프로빙 평가 프로토콜을 도입하여, 일반적인 교차 피험자 평가 이상으로 모델의 일반화 성질을 드러냅니다.

- **Performance Highlights**: 실험 결과, 제안한 PPG 기초 모델은 15개의 다양한 다운스트림 작업에서 14개 작업에서 5개의 강력한 기준 모델보다 뛰어난 성능을 보였습니다. 이 결과는 다중 모드 사전 학습이 얼마나 효과적이고 내구성이 우수한지를 입증하며, 특히 현장 배포 시나리오에서 높은 성능을 유지할 수 있는 잠재력을 보여줍니다. 따라서 본 연구는 PPG 기반 모델의 일반화 능력 향상에 기여할 뿐만 아니라, 가정용 데이터에 대한 적용 가능성을 높이는 데 중요한 의미를 갖습니다.



### SleepExplain: Explainable Non-Rapid Eye Movement and Rapid Eye Movement Sleep Stage Classification from EEG Signa (https://arxiv.org/abs/2606.07351)
Comments:
          6 pages, 7 figures, 2022 25th International Conference on Computer and Information Technology (ICCIT)

- **What's New**: 이번 연구에서는 NREM(Non-Rapid Eye Movement) 및 REM(Rapid Eye Movement) 수면 단계를 자동으로 분류하는 SleepExplain 모델을 제안합니다. 연구는 3개의 EEG(뇌파) 채널에서 수면 데이터를 수집하고, 샘플 기준으로 89096개의 데이터를 활용하여 모델을 훈련시켰습니다. 이 모델은 기존의 비판적 접근 방식이 아닌, SHAP(SHapley Additive exPlanations)를 이용하여 예측 결과를 설명하는 설명 가능성을 갖춘 인공지능(XAI)을 적용했습니다.

- **Technical Details**: 이 연구는 Random Forest, Gradient Boosting, XGBoost 같은 앙상블 머신러닝 기법을 이용하여 수면 단계를 분류했습니다. 수집된 데이터셋은 Haaglanden Medisch Centrum에서 얻어진 것으로, 뇌파 기록은 여러 신호가 포함되어 있으며, 75개의 특징을 추출하여 주요 수면 단계를 식별합니다. FFT(Fast Fourier Transform) 기법을 통해 신호를 변환하고, SMOTE(Synthetic Minority Over-sampling Technique)를 활용하여 데이터 불균형 문제를 해결했습니다.

- **Performance Highlights**: 모델의 정확도는 Random Forest에서 92.54%, Gradient Boosting에서 94.25%, XGBoost에서 94.30%로 향상되었습니다. 이러한 결과는 수면 단계 분류를 통해 수면 장애를 보다 효과적으로 진단하고 개선할 수 있음을 시사합니다. 동시적으로 개발된 SHAP 기반의 설명 가능한 모델은 예측의 근거를 제시하여 사용자에게 신뢰성을 높였습니다.



### A Temporal Spatial Minimax Rate for Smoothly-Varying Distributions in Wasserstein Spac (https://arxiv.org/abs/2606.07325)
- **What's New**: 이 논문은 곡선 $t 	o 
u_t$의 미래 값 $
u_{t_n+h}$를 2-Wasserstein 공간 $	extmath{P}_2(	extmath{R}^d)$에서 경량 데이터로부터 추정하는데 필요한 minimax 비율(minimax rate)을 연구합니다. 저자들은 부드러운 속도장의 $k$-차 공변 미분에 대한 아디아바틱 경계 조건을 두고 다양한 정규화된 하위 클래스에 대해 $W_2$-위험과 추정자의 성능을 평가합니다. 이 연구는 시간과 공간에 관한 새로운 통합 minimax 하한을 섭렵하고, 이를 통해 정확한 과거 데이터와 공간 패킹의 결합을 통해 추정 성능 한계를 설정합니다.

- **Technical Details**: 본 연구에서는 매개변수 $k$와 전체 샘플 크기 $M$ 간 관계를 설명하는 공식을 제시합니다. 예를 들어 $W_2$-위험이 $M$ 지수와 함께 $M^{-	ext{γ}_d(k+1)/(k+1+	ext{γ}_d)}$로 나타날 수 있음을 보여줍니다. 또한 이론적으로 $k 	o 	ext{∞}$ 일 때, 정적 분포 추정 비율로 복구된다는 점에서 $M^{-	ext{γ}_d}$의 공간적 평가의 저주를 드러냅니다. 이를 바탕으로, 특정 관측 시간에 대해 유효 샘플 크기가 가중된 설계 형태로 하한을 제시하고, 조밀한(동점 spaced) 환경에서 닫힌 형식의 지수를 도출합니다.

- **Performance Highlights**: 본 연구에서 제시된 하한은 수치 실험을 통해 예측된 지수와 일치함을 보여줍니다. 특히, $k=0$에서 매칭 상한이 설정되며(d ≥ 3일 경우 $M^{-1/(d+1)}$ 속도를 달성함), $k 
eq 0$에 대한 조건부 추정자가 특정 조건하에 이 비율을 달성할 수 있음을 발견했습니다. 또한, 동적 분포 예측은 정적 분포보다 느린 것으로 나타나며, 이러한 발견은 수치 실험을 통해 확인되었습니다.



### Hierarchical Certified Semantic Commitment for Byzantine-Resilient LLM-Agent Collaboration (https://arxiv.org/abs/2606.07316)
Comments:
          27 pages, 3 figures, 8 tables

- **What's New**: 이번 논문에서는 대규모 언어 모델(LLM) 에이전트 간의 비잔틴 협력 문제를 해결하기 위한 새로운 프로토콜인 Hierarchical Certified Semantic Commitment (H-CSC)를 제안합니다. 이 프로토콜은 여러 제안들 중 어떤 결과를 최종적으로 결정해야 하는지를 판단해야 합니다. 기존의 비잔틴 결함 허용(BFT) 프로토콜은 비트 레벨에서의 동작을 기반으로 하였으나, H-CSC는 언어적 제안의 의미론적(core) 내용에 기반하여 결정을 내립니다.

- **Technical Details**: H-CSC는 최종성( конечность) 신호를 embedding 기반으로 변환하여 세 가지 유형의 결과를 내놓습니다: semantic_commit(의미적 접속이 존재할 때), verdict_commit(판단의 우세가 나타날 때), 그리고 명시적 abort(의도가 불가능할 때). 이 프로토콜은 메트릭 공간 내에서 제안들을 기하학적으로 집계하여 각 라운드에 대해 적합한 결정을 내립니다.

- **Performance Highlights**: H-CSC는 제어된 의미적 오염 진단에서 0.31도에서 2.04도의 낮은 각도 편차로 커밋을 수행하며, 비잔틴 조건을 초과하는 라운드에서는 100%의 abort를 실현했습니다. 실제 LLM 에이전트 검증 벤치마크에서 H-CSC는 0.90/0.92의 높은 정확도를 기록하면서도 강력한 증명서를 추가하여 타당한 결과를 가져왔습니다.



### SV-Detect: AI-generated Text Detection with Steering Vectors (https://arxiv.org/abs/2606.07313)
- **What's New**: 이 논문에서는 SV-Detect라는 새로운 기법을 제안하는데, 이는 고정된 언어 모델의 숨겨진 표현에서 추출된 steering vectors를 기반으로 한 가짜 텍스트 탐지기입니다. 이 방법은 인간이 쓴 텍스트와 기계가 생성한 텍스트를 구분하기 위한 방향을 각 층에서 구축하며, 이러한 방향들과 입력 텍스트 간의 정렬을 통해 텍스트를 표현합니다. SV-Detect는 강력한 성능을 보여주며, 특히 도메인 간 전이(distribution shift)에도 효과적입니다.

- **Technical Details**: SV-Detect의 방법론은 고정된 변환기(transformer) 언어 모델의 층별 활성화를 추출하는 것으로 시작합니다. 그런 다음, 인간 작성 텍스트와 기계 생성 텍스트를 구별하는 steering vector(조향 벡터)를 추출하며, 이는 낮은 차원의 특징으로 텍스트 표현을 투영하는 데 사용됩니다. 최종적으로, 이러한 투영 특징을 바탕으로 경량(classifier) 분류기를 학습하여 최종 탐지 점수를 생성합니다.

- **Performance Highlights**: SV-Detect는 DetectRL과 MIRAGE라는 두 개의 보완적인 벤치마크에서 평가되었으며, 두 설정 모두에서 뛰어난 성능을 나타냈습니다. 특히, SV-Detect는 도메인 간 전이와 기계 편집 변환에서도 강력한 성능을 발휘하는 것으로 나타났습니다. 논문에서는 SV-Detect가 인간 텍스트와 기계 생성 텍스트 간의 스타일적 신호를 포착하고, 표현 레벨에서의 차이를 연구하는 데 유용한 도구가 될 수 있음을 강조합니다.



### CULTURESCORE: Evaluating Cultural Faithfulness in Video Generation Models (https://arxiv.org/abs/2606.07311)
- **What's New**: 본 연구는 비디오 생성 모델의 문화적 충실도를 평가하는 새로운 프레임워크인 CultureScore를 제안합니다. 이 프레임워크는 Identity, Behavior, Context의 세 가지 차원으로 구성되어 있습니다. 기존 메트릭에서는 단순히 시각적 품질만을 측정하는 데 반해, CultureScore는 문화적 대표성을 보다 세밀하게 분석할 수 있도록 합니다.

- **Technical Details**: CultureScore는 10개국의 2,943 개 프롬프트로 구성된 데이터셋을 바탕으로 6,180개의 비디오를 생성하여 평가를 수행합니다. 각 평가 항목은 Identity(누가 표현되는지), Behavior(문화적 제스처), Context(문화적 상황)로 나뉘며, 이러한 섬세한 구성 요소를 통해 모델이 어떻게 문화적으로 왜곡되는지를 진단합니다.

- **Performance Highlights**: 실험 결과, 현재의 어떤 비디오 생성 모델도 문화적으로 충실한 생성 결과를 달성하지 못했습니다. 최고 성능의 모델도 56.8%의 CultureScore를 기록했으며, Behavior 차원에서의 성과는 52% 미만으로, 문화적 표현에서의 부족함을 분명히 드러냈습니다. 또한, 문화적 충실도에 대한 인간 평가자의 선호도는 Visual Quality 메트릭과 반비례 관계를 보였습니다.



### Acoustic Cue Alignment in Audio Language Models for Speech Emotion Recognition (https://arxiv.org/abs/2606.07309)
Comments:
          6 pages, 3 figures, 3 tables

- **What's New**: 이번 연구는 명확한 음향 단서를 통해 지침을 따르는 오디오 언어 모델(ALMs)에서 음향 큐가 실제로 사용되는 방식을 조사합니다. 이를 위해 eGeMAPS의 유의미한 음향 개념 토큰을 도출하고, 이를 텍스트 프롬프트에 추가하여 오디오 입력은 그대로 유지합니다. 연구 결과, 알맞게 정렬된 토큰이 성능을 향상시키는 반면, 셔플된 또는 상충하는 토큰은 성능을 저하시킨다는 것을 보여줍니다.

- **Technical Details**: 유명한 FAU-Aibo와 IEMOCAP 벤치마크를 통해, 정렬된 개념 토큰을 사용한 ALM은 비정렬 토큰에 비해 비교적 높은 Unweighted Average Recall (UAR)을 달성합니다. 본 연구에서는 에너지, 피치, 역학, 밝기, 포먼트, 음성 품질 등 여섯 가지 범주로 음향 특성을 구분하여 텍스트 형식으로 변환, ALM 프롬프트에 보조 Cue로 첨부하여 실험하였습니다.

- **Performance Highlights**: 연구 결과, 어떠한 토큰 조작에도 불구하고 ALM 예측은 오디오 신호에 여전히 부분적으로 의존하는 것으로 나타났습니다. 정렬된 음향 개념 토큰은 성능을 일관되게 향상시키는 반면, 강한 손상이 있는 경우 예측 성능이 일정 수준 저하되지만 음향 신호를 완전히 무시하지는 않는 것으로 밝혀졌습니다. 이러한 발견은 ALM 기반의 감정 인식에서 음향 기반 큐의 사용을 보다 명확하게 검증하는 방법론을 제시합니다.



### Where Rectified Flows Leak: Characterising Membership Signals Along the Interpolation Path (https://arxiv.org/abs/2606.07271)
Comments:
          ICML 2026 article, 9 main pages and 25 with annexes, 11 figures

- **What's New**: 이번 연구에서는 Rectified Flows의 메모리 신호(membership signal) 구조를 탐구합니다. 이는 훈련 데이터로부터 기억되는 특성을 이해하는 데 중요한데, 특히 법적 문제와 개인정보 보호의 맥락에서 중요합니다. 모델이 훈련 데이터를 직접 재현하는 것 이상으로 미세한 단서를 암호화할 수 있음이 밝혀졌습니다.

- **Technical Details**: 연구진은 Rectified Flow 훈련에서 정의된 간섭 경로(X_λ = (1-λ)X_0 + λX_1)를 통해 훈련 데이터와 테스트 데이터의 재구성 간 차이를 분석했습니다. λ의 변화에 따라 벨 형태의 곡선을 따르는 간극이 존재하며, 이는 훈련 중 누적됩니다. 이 경로에서 얻어진 이론적 예측은 오디오 및 이미지 데이터셋에서 실험적으로 검증되었습니다.

- **Performance Highlights**: 연구의 명백한 기여로, 제안된 λ-해결 구조는 단순한 Membership Inference Attack (MIA)을 통해 훈련 세트와 비훈련 세트를 구별하는 데 사용될 수 있습니다. 이를 통해 연구진은 모델이 훈련 데이터를 어떻게 기억하고 있는지를 관찰할 수 있는 방법론을 제시하였으며, 이는 향후 연구와 실무에 응용될 가능성이 높습니다.



### AI Sovereignty: A Qualitative Model of Strategic Competition as AI Becomes an Instrument of National Power (https://arxiv.org/abs/2606.07245)
Comments:
          Main article: 19 pages, 10 figures. Supplementary: 19 pages, 7 figures, 7 tables. To be presented at the 2026 International System Dynamics Conference (ISDC), July 20-24, TU Delft, Delft, Netherlands

- **What's New**: 이 논문은 AI 주권(AI sovereignty)의 정의와 개념 모델을 제시하여 국가가 AI 기술을 독립적으로 통제하는 정도를 살펴봅니다. 특히, AI의 경제적 이점, 경쟁 우위 및 국가적 힘에 대한 국가의 전략적 중요성을 강조합니다. 또한, AI 주권 동역학을 탐색하는 데 필요한 정의와 질적 모델을 처음으로 제안합니다.

- **Technical Details**: 제안된 질적 모델은 미시(micro), 중시(meso), 거시(macro) 기여자를 포함하여 AI 주권의 복잡한 요소를 통합합니다. 이를 통해 모델 기반 질적 예측을 수행하고 국가 간의 경쟁 역학을 분석하여 AI 기반의 국가력을 진단합니다. 또한, 전력, 물, 데이터 세트 및 숙련된 인력과 같은 레버리지 포인트(key leverage points)를 식별하여 국가 성장 강화 또는 적의 약화를 위한 전략을 모색합니다.

- **Performance Highlights**: 모델은 직접적인 물리적 행동과 사이버, 우주, 정보 및 경제적 강제 수단과 같은 간접적인 비물리적 효과를 통해 활성화될 수 있는 레버리지 포인트의 전략적 및 운영적 측면에서의 위력을 강조합니다. 이 연구는 21세기 국가들이 경제적 이익, 경쟁 우위, 국가적 힘을 향상시키기 위해 어떻게 AI 주권을 활용할 수 있을지에 대한 근본적인 통찰을 제공합니다.



### Beyond Waypoints: A Trajectory-Centric Waypointing Paradigm for Vision-Language Navigation (https://arxiv.org/abs/2606.07244)
- **What's New**: 이 연구에서는 기존의 노드 중심 waypoint 예측을 뛰어넘어 Trajectory Waypoint라는 새로운 패러다임을 제안합니다. 이는 각 후보 waypoint를 실행 가능한 궤적에 기반하여 생성하여, 물리적으로 도달 가능한 목표를 보장합니다. 또한, 이 방법은 고수준 계획과 저수준 실행 간의 일관성을 극대화하는데 기여합니다.

- **Technical Details**: Trajectory Waypoint Predictor는 Diffusion policy로 형성되어 있으며, TSDF 기반의 비용 유도(guidance)를 통해 오프젝트를 피하면서 궤적 생성을 조정합니다. 이로 인해 더욱 안전하고 다양한 항해 가능 궤적을 생성할 수 있고, 이를 통해 에이전트의 탐색 능력을 향상시킵니다. 또한, 궤적 기반 내비게이터는 고급 기하학적 정보와 관련된 궤적을 플래닝 정보로 활용하여 보다 정밀한 방향성을 제공합니다.

- **Performance Highlights**: VLN-CE 벤치마크에서의 실험 결과, Trajectory Waypoint 패러다임이 기존의 waypoint 예측기보다 목표 도달 가능성에서 눈에 띄게 우수한 성능을 보임을 확인했습니다. 전체적으로, 이 프레임워크는 아래의 다양한 VLN-CE 작업에서 성능이 크게 향상되었습니다. 따라서, 기존의 기본선과 비교했을 때 안정적이고 강력한 개선을 나타냅니다.



### When Large Language Models Fail in Healthcare: Evaluating Sensitivity to Prompt Variations (https://arxiv.org/abs/2606.07237)
Comments:
          12 pages

- **What's New**: 이 논문에서는 의료 분야에서의 대형 언어 모델(LLM)의 감도(sensitivity)에 대한 체계적인 분석을 통해, 일반 목적 모델(예: GPT-3.5, Llama3)과 의료 특화 모델(예: ClinicalBERT, BioLlama3, BioBERT)의 강건성을 평가합니다. 특히, 프롬프트의 언어적(lexical) 및 구문적(syntactic) 변동이 임상적 질문 응답 및 진단 지원과 같은 임상 작업에서 모델의 성능에 미치는 영향을 조사합니다. 연구 결과, 작은 문구의 변화조차 모델의 출력을 크게 바꿀 수 있음을 보여 주며, 이는 의료 환경에서는 치명적인 위험으로 인식됩니다.

- **Technical Details**: 연구에서는 프롬프트의 변동을 자연적인 유형과 적대적(adversarial) 유형으로 분류하고, 두 유형이 모델의 일관성(consistency), 정확성(accuracy) 및 신뢰성(reliability)에 미치는 영향을 분석합니다. 특히, 의료 LLM들은 단순한 어휘의 대체(substitution)에는 어느 정도 견딜 수 있으나, 구문 정렬(syntactic reordering)이나 오해를 일으킬 수 있는 문맥적 단서(contextual cues)에는 취약한 모습을 보입니다. 논문은 LLM의 안정성을 평가하기 위한 표를 제공합니다.

- **Performance Highlights**: 주요 성과로는 간단한 문구 변경이 올바른 진단에 부정적인 영향을 미칠 수 있음을 보여주는 여러 사례를 통해, 의료 AI 시스템에 대한 신뢰성을 구축하기 위해 해결해야 할 주요 도전 과제가 있음을 강조합니다. 특히, 적대적 조작(adversarial manipulations)은 부적절한 약물 권장이나 필수 발견 사항을 간과하는 등의 심각한 결과를 초래할 수 있습니다. 모델의 신뢰성을 강화하기 위해서는 이러한 취약성과 한계를 인식하고, 보다 강건하고 신뢰할 수 있는 의료 AI 시스템을 개발하는 방법을 모색해야 합니다.



### DEFINED: A Data-Efficient Computational Framework for Fine-Grained Creativity Assessment in Debate Scenarios (https://arxiv.org/abs/2606.07226)
Comments:
          Accepted by KDD 2026

- **What's New**: 이 논문에서는 DEFINED라는 새로운 데이터 효율적인 계산 프레임워크를 제안하며, 이는 토론 시나리오에서의 세밀한 창의력 평가를 자동화하는 데 초점을 맞추고 있습니다. 이 프레임워크는 여덟 차원의 계층적 메트릭 시스템을 통해 창의력을 정의하며, 기존의 수작업 평가에서 발생하는 노동집약적인 어려움을 해결하기 위해 설계되었습니다. DEFINED는 학습된 대학원 전문가에 의해 주석이 달린 세밀한 데이터로부터 강력한 학습을 가능하게 합니다.

- **Technical Details**: DEFINED 프레임워크는 인간의 총체적 판단과 일치하는 여덟 차원의 세밀한 점수와 거시적(debate) 점수를 생성합니다. 이 시스템은 실제 토론 대회에서 수집한 정확한 경쟁 문서 데이터를 활용하며, 원본 데이터의 엘리트 편향을 줄이기 위해 제한된 데이터 증강 전략을 채택합니다. 여덟 차원의 메트릭 시스템은 분산적 사고(divergent thinking) 및 수렴적 사고(convergent thinking)와 같은 창의력 관련 차원과 함께 토론 관련 차원을 포함합니다.

- **Performance Highlights**: 실험 결과, DEFINED의 점수 모델이 기존의 토론 점수화 방법 및 프롬프트 기반 대형 언어 모델 평가자를 초과하며 정확하고 안정적인 점수를 달성했습니다. 특히, 60개의 세밀한 전문가 주석 샘플과 4,000개의 거시적 샘플을 활용하여 고정밀 예측을 달성했으며, 이러한 접근 방식은 '소규모 데이터' 문제를 효과적으로 해결했습니다. 또한, 다양한 숙련도와 주석 세분화에서 시스템의 강건성을 검증하기 위한 평가 프로토콜을 수립했습니다.



### DualGate-Net: A Prior-Gated Dual-Encoder Framework for Histopathology Cell Detection (https://arxiv.org/abs/2606.07222)
Comments:
          15 pages, 4 figures

- **What's New**: 이 논문에서는 DualGate-Net이라는 새로운 이중 인코더 프레임워크를 제안합니다. 이 프레임워크는 ConvNeXtV2 기반의 로컬 인코더와 SegFormer 기반의 글로벌 인코더를 결합하여 적응형 컨텍스트 통합을 통해 세포 탐지의 효율성을 향상시킵니다. 특히, 학습 가능한 prior-gated fusion 메커니즘이 포함되어 공간 위치별로 컨텍스트의 영향을 조절하며, 고주파 세포 구조를 보존하는 조정지향의 분기(branch)가 추가됩니다.

- **Technical Details**: 제안된 DualGate-Net은 인코더 아키텍처를 활용하여 미세구조와 대규모 컨텍스트 정보를 동시에 모델링합니다. ConvNeXtV2 인코더는 로컬 셀 구조를 정밀하게 감지하는 데 사용되며, SegFormer 인코더는 장기적인 컨텍스트 의존성을 모델링합니다. 또한, tissue segmentation을 위한 SegFormer-B2 모델을 훈련시켜 제공된 컨텍스트 정보를 기반으로 다채널 확률 맵을 생성하여 논문에서 설정한 학습 알고리즘의 기본으로 사용합니다.

- **Performance Highlights**: OCELOT 벤치마크 실험에서 DualGate-Net은 세포 탐지 성능이 꾸준히 향상되었음을 보여주었으며, 검증 세트에서 매크로 F1 점수 0.7722, 테스트 세트에서 0.7345의 기록을 달성했습니다. 이는 적응형 prior 통합이 robust한 세포 탐지에 효과적임을 입증하는 결과입니다. 언급된 강력한 성능 향상은 제안된 아키텍처의 전반적인 설계 및 샘플의 특징 추출 능력과 관련이 있습니다.



### An Abstract Architecture for Explainable Autonomy in Hazardous Environments (https://arxiv.org/abs/2606.07211)
Comments:
          Originally published 20th of October 2022 at the Second International Workshop on Requirements Engineering for Explainable Systems (RE4ES), which was hosted by the International Requirements Engineering Conference 2022

- **What's New**: 이 논문에서는 자율 로봇 시스템이 위험한 환경에서 사용되도록 설계되고 있음을 강조합니다. 특히, 사용자가 시스템에 대한 신뢰를 가질 수 있도록 하는 것이 중요하다고 주장합니다. 설명 가능성(explainability)이 시스템의 신뢰성과 밀접한 관계가 있으며, 이러한 설명 가능성을 시스템의 설계에 내재화해야 한다고 제안합니다.

- **Technical Details**: 제안된 아키텍처는 고수준과 저수준의 의사결정을 분리합니다. 고수준 결정을 위해 Belief-Desire-Intention (BDI) 에이전트를 사용하며, 이 에이전트는 환경 정보를 해석하고 논리적인 진술로 바꾸어 계획을 선택합니다. 이 아키텍처는 시스템의 각 행동에 대한 과거 설명과 미래 행동에 대한 설명을 제공하는 두 가지 모드를 지원합니다.

- **Performance Highlights**: 이 아키텍처가 제공하는 설명 가능성의 두 가지 주요 모드는 사용자와 규제 기관 모두에게 시스템의 신뢰도를 높이는 데 기여할 수 있습니다. 과거 행동에 대한 설명은 결정의 근거를 제공하고, 미래 행동에 대한 설명은 시스템의 안전성을 검증합니다. 이러한 기능들은 특히 핵산업과 같은 안전-critical한 환경에서 중요한 역할을 합니다.



### RETROSPECT: RETROsynthesis via Sequential Prediction, and Chemically Transformed-ranking (https://arxiv.org/abs/2606.07181)
Comments:
          Accepted at the AI for Science workshop (ICML 2026)

- **What's New**: 이번 논문에서 제안하는 RETROSPECT 시스템은 단일 단계의 레트로합성(retrosynthesis) 문제를 다루기 위해 제안 및 재정렬(proposal-selection decomposition)을 결합합니다. ChemAlign Transformer라는 단일 Transformer 제안 모델과 LambdaMART 재정렬기를 통합하여 구조적 특성(structural descriptors), 반응 템플릿(reaction-template), 상위 점수(upstream score) 등의 변수를 사용하여 후보를 정렬합니다. 이를 통해 더 정확한 후보 제안과 다채로운 후보 목록을 생성할 수 있습니다.

- **Technical Details**: RETROSPECT 시스템에 사용된 ChemAlign Transformer는 하이브리드 루트 정렬(root-aligned) 및 랜덤 SMILES 증강을 통해 훈련됩니다. 이 모델은 6개의 인코더 층과 6개의 디코더 층을 사용하며, 포지셔널 인코딩(positional encoding) 방법을 통해 긴 트리플릿을 처리합니다. 추가적으로, LambdaMART 재정렬기는 후보 풀이 생성된 후 유용하게 작용하여, 우선순위를 매기는 데 중요한 디스크리트(accessory) 손실 함수와 함께 작동합니다.

- **Performance Highlights**: 모델의 성능은 USPTO-50K 데이터셋에서 전체 5,007개의 반응을 가진 테스트 세트를 기준으로 평가되었습니다. RETROSPECT는 첫 번째 제안에 대해 55.00%의 top-1 정확도와 86.18%의 top-10 정확도를 달성했으며, 구현된 LambdaMART 모델은 59.4%의 top-1 정확도와 0.7171의 평균 역순위(mean reciprocal rank)를 기록했습니다. 이 연구는 단일 모델 제안 방식이 강력하고, 후보 선택과 함께 사용할 때 더욱 효과적임을 보여줍니다.



### Textual Supervision Enhances Geospatial Representations in Vision-Language Models (https://arxiv.org/abs/2606.07172)
Comments:
          Accepted at ICML 2026

- **What's New**: 이번 연구에서는 기계 학습 시스템의 지리적 이해를 분석하였습니다. 기존의 비전 전용 아키텍처, 비전-언어 모델, 그리고 대규모 다중 모달 기초 모델을 통해 지리적 표현을 평가하였으며, 텍스트 감독이 이러한 학습을 어떻게 향상시키는지 조사하였습니다. 연구 결과, 언어가 공간 맥락을 인코딩하는 유효한 보조 방식으로 작용하며, 다중 모달 학습이 지리적 인공지능 발전의 핵심 방향이 될 수 있음을 제시하였습니다.

- **Technical Details**: 비전 모델은 최근 10년 동안 컨볼루션 신경망(CNN)과 비전 트랜스포머(ViT)의 발전으로 큰 발전을 이루었습니다. 다양한 메타 정보로부터 지리적 정보를 내재화할 수 있는 가능성이 제기되었으며, 이러한 모델들은 프리트레인(pretrain) 및 파인튜닝(fine-tuning) 과정에서 지리적 지식을 어느 정도 내재화하는지를 분석하고 있습니다. 우리가 사용한 지리적 표현(geospatial representations)은 ViT와 VLM의 내부 레이어에 포함된 잠재적 피처를 지칭합니다.

- **Performance Highlights**: 본 연구는 비전 전용 및 비전-언어 모델이 지리적 정보를 암묵적으로 어떻게 인코딩하는지를 조사하였습니다. 다양한 모델의 성능을 레이어 별 탐색을 통해 비교하였으며, 비전 전용 모델은 마지막 레이어에서 더 강한 표현을 보였고, VLM은 언어 모델 블록의 초기 레이어에서 더 나은 지리적 표현을 나타냈습니다. 비전-언어 모델에 대한 프롬프트(prompts)를 통해 지리적 정보가 후속 레이어로 전파되는 것이 관찰되었으며, 이로 인해 표현 품질이 향상되는 경우도 발견되었습니다.



### UrduMMLU: A Massive Multitask Benchmark for Urdu Language Understanding (https://arxiv.org/abs/2606.07167)
Comments:
          27 pages, 18 figures, 17 tables, Submitted to ARR May 2026

- **What's New**: 우리는 UrduMMLU라는 새로운 벤치마크를 소개합니다. 이 벤치마크는 26개 과목과 5개 도메인에서 총 26,431개의 우르두어 객관식 질문(MCQs)을 포함하고 있습니다. 전통적인 번역 기반 리소스와는 달리, UrduMMLU는 표준 학문적인 주제 및 우르두어 및 지역 특화된 내용을 아우릅니다.

- **Technical Details**: UrduMMLU는 이중 인간 주석을 통해 채점된 시험 유래 부분을 포함하며, 엄격한 합의 기준을 적용하여 라벨을 부여합니다. 이 벤치마크는 우르두어 교육 맥락에서 수집된 질문들로 구성되어, 번역에 의존하지 않고 원어로 작성된 최초의 포괄적 MMLU 스타일의 벤치마크입니다.

- **Performance Highlights**: 대규모 언어 모델(LLM) 평가에서 Gemini-3.5-Flash가 90.20% 및 90.34%의 정확도를 달성하여 최상의 성능을 보였습니다. 그러나 다른 모델들이 85%를 초과하지 못했으며, 각 모델의 우르두어 인문학 관련 주제에서는 25에서 40포인트가량 저조한 퍼포먼스를 보였습니다.



### From Privacy to Workflow Integrity: Communication-Graph Metadata in Autonomous Agent Interoperability (https://arxiv.org/abs/2606.07150)
Comments:
          12 pages, 6 figures

- **What's New**: 이번 논문에서는 A2A 및 MCP와 같은 에이전트 상호 운용성 프로토콜이 메시지 콘텐츠를 표준화하며, HTTP(S)를 통한 주소 기반 전송을 가정하고 있다는 점을 강조합니다. 보안은 주로 메시지 콘텐츠 보호에 집중되지만, 통신 그래프(communication graph)는 노출되어 결국 개인 정보 보호 이상의 문제로 발전할 수 있습니다. 이 논문은 이러한 메타데이터가 예측 가능한 방식으로 에이전트의 작업 흐름에 대한 통찰력을 제공하여, 권위 있는 행동 예측에 대한 위험을 제기한다고 주장합니다.

- **Technical Details**: 글에서는 에이전트 통신 그래프의 위협 모델을 개발하고, 에이전트 메타데이터가 어떤 이유로 독특하게 드러나는지 설명하여, 개인 정보 보호 차원을 넘어 자율적 워크플로우의 무결성에 대한 위협으로 재구성합니다. 뿐만 아니라, 메시지의 전송 방식(transport) 및 부트스트랩 계층의 프라이버시 특성 프레임워크를 정의하고, SLIM, Tor, mixnets와 같은 후보 전송 방식을 평가합니다. 마지막으로, A2A 사례 연구를 통해 메타데이터 보호 결합이 어떻게 설정될 수 있는지를 보여 줍니다.

- **Performance Highlights**: 실험 결과, 수동 메타데이터만으로도 작업 클래스가 우연보다 훨씬 잘 회복됨을 보여주며, 전체 속성 세트를 사용해야만 그러한 회복이 우연의 경향으로 위축된다는 것을 발견했습니다. 장애인이 특정 워크플로우를 기반으로 작업을 수행할 경우, 이 모델에서는 메타데이터를 알지 못하는 공격자가 얻을 수 있는 이점의 대부분을 캡처할 수 있습니다. 이러한 결과는 공격자가 워크플로우의 열기 및 고정 예산을 기준으로 행동할 때 예측 가능한 행동을 촉진한다는 것을 시사합니다.



### REMEDI: A Benchmark for Retention and Unlearning Evaluation in Multi-label Clinical Disease Inferenc (https://arxiv.org/abs/2606.07141)
Comments:
          Under review

- **What's New**: 이번 논문에서는 임상 질병 추론을 위한 머신 언러닝의 새로운 벤치마크인 REMEDI를 소개합니다. REMEDI는 의료 분야를 위해 특별히 설계되었으며, 환자 데이터를 기반으로 다양한 잊기 설정을 포괄합니다. 이 벤치마크는 메디컬 도메인에서 머신 언러닝 방법의 성능 평가를 위해 실제적인 시나리오를 반영합니다.

- **Technical Details**: REMEDI는 MIMIC-III 임상 데이터베이스를 활용하여 구축되었으며, 다중 레이블 및 다중 클래스 분류 작업을 대상으로 합니다. 이 벤치마크는 세 가지 잊기 강도 수준을 포함하며, 개별적인 데이터 삭제 요청부터 대규모 데이터 삭제 요청까지 아우릅니다. 반복 학습 없이도 특정 데이터의 영향을 효과적으로 제거할 수 있는 알고리즘 개발의 필요성이 강조됩니다.

- **Performance Highlights**: REMEDI는 모델 성능을 유지하면서도 잊기 요청이 성공적으로 수행되었는지를 평가합니다. 네 가지 머신 언러닝 방법을 비교하여 이들이 임상 질병 추론 작업에서 얼마나 효과적으로 작동하는지를 검증합니다. 또한, REMEDI는 모델이 현실적으로 어떤 이점을 가지고 있는지를 종합적으로 평가하는 프레임워크를 제공합니다.



### The Three-Ring Architecture: Governing Agents in the Era of On-Platform Organisations (https://arxiv.org/abs/2606.07119)
Comments:
          28 pages

- **What's New**: 현재 기업 AI 배포의 새로운 단계는 구조적 실패를 겪고 있습니다. 조직들은 관리 인프라 없이 능동적인(agentic) 기능을 확보하고 있으며, 이로 인해 95%의 프로젝트 실패율을 초래할 것으로 예상하고 있습니다. 이 논문은 플랫폼 조직의 관리 인프라로서 Three-Ring Architecture를 형식화하며, 각 링이 수행하는 역할을 설명합니다.

- **Technical Details**: Three-Ring Architecture는 세 개의 링으로 구성됩니다. 첫 번째 링(Ring 1)은 기존의 생산 아키텍처, 두 번째 링(Ring 2)은 전략 기반의 능동형 AI를 위한 M2 연합 계층, 세 번째 링(Ring 3)은 LLM 기반의 최전선 지능 계층입니다. Ring 2는 기업의 운영체제와 같은 역할을 하며, 자원 추상화, 프로세스 조정, 권한 집행을 통해 지능의 복합화를 가능하게 합니다.

- **Performance Highlights**: 이 아키텍처는 지난 10년간 금융 서비스, 정부, 조달 및 규정 준수 분야에서 검증되었습니다. LLM의 능력 향상은 Ring 2 아키텍처에 대한 구조적 이점을 제공하며, 이러한 능력이 향상됨에 따라 Governance 요구 사항도 비례하여 증가합니다. 이 논문의 주장은 모든 조직이 LLM의 능력을 활용하기 위해 꼭 필요한 Ring 2가 필요하다는 점에 있습니다.



### Native3D: End-to-End 3D Scene Generation via Unified Mesh-Texture Modeling and Semantic Alignmen (https://arxiv.org/abs/2606.07117)
- **What's New**: 이 논문에서는 2D 중간 표현을 완전히 우회하는 첫 번째 엔드 투 엔드 3D 장면 생성 프레임워크인 Native3D를 제안합니다. 기존의 접근 방식은 3D 표현을 2D 도메인으로 변환해야 하는데, 이는 기하학적 구조 왜곡과 텍스처 세부 사항 저하와 같은 도메인 적응 문제를 초래합니다. Native3D는 기하학적 구조와 텍스처 특성을 동시에 모델링하는 통합된 메쉬-텍스처 공동 표현을 설계하여 이러한 제한 사항을 해결합니다.

- **Technical Details**: Native3D는 Transformer 기반 장면 인코더를 통해 기하학적 구조와 텍스처 기능을 동시에 모델링합니다. 이를 통해 장면 내 객체 간의 공간 관계와 시각적 일관성을 효과적으로 유지합니다. 또한 3D 표현 정렬 손실(3D REPA Loss)을 도입하여 다중 수준의 의미 표현을 정렬함으로써 기하학적 및 텍스처 품질을 크게 향상시킵니다. 이 방법은 3D 메쉬 기하학 및 텍스처 정보를 직접 입력으로 받아 통합된 표현을 생성합니다.

- **Performance Highlights**: 실험 결과, Native3D는 생성 품질 및 편집 유연성 모두에서 기존 방법보다 우수한 성능을 보였습니다. 사용자들은 자연어를 조건 신호로 사용하여 복잡한 장면 생성 및 수정 요구를 정확하게 지정할 수 있습니다. 이러한 결과는 Native3D가 3D 장면 편집을 위한 혁신적인 솔루션을 제공한다고 강조합니다.



### OffQ: Taming Structured Outliers in LLM Quantization by Offsetting (https://arxiv.org/abs/2606.07116)
- **What's New**: 본 논문에서 OffQ라는 새로운 방법을 소개하여 저비트 양자화(Low-bit Quantization)에서의 활성화 아웃라이어(activation outliers) 문제를 해결합니다. OffQ는 첫 번째로 제안된 top-1 PCA를 통해 저차원의 아웃라이어 서브스페이스(low-dimensional outlier subspace)를 식별하고, 고크기 활성화를 회전을 통해 하나의 채널로 집중시킵니다. 마지막으로, 집중된 아웃라이어 채널의 크기를 공유 오프셋으로 변환하여 활성화의 표준 편차를 줄이는 오프셋 전략을 활용합니다.

- **Technical Details**: OffQ는 저비트 양자화의 기본 적용에서 아웃라이어의 영향을 제거하기 위해 고유한 회전 기반 프로세스를 사용합니다. 이 방식은 단순한 방법이며, 복잡한 매개변수나 계산 오버헤드 없이 저비트 계산의 효율성을 유지할 수 있습니다. OffQ는 Hadamard 회전을 적용하여 아웃라이어 에너지를 그룹별 오프셋으로 전환하고, 이를 표준 저비트 양자화의 제로 포인트로 흡수함으로써 성능을 높입니다. 이러한 절차는 W4A4KV4(Weight, Activation, KV-cache 모두 4비트로 양자화) 적용에서 효과적으로 이루어집니다.

- **Performance Highlights**: 다양한 LLM 구조 및 벤치마크에서 OffQ는 기존의 최신 방법들을 능가하며 모델의 정확성을 지속적으로 개선했습니다. OffQ는 예측력(perplexity)과 정확도 모두에서 개선을 이뤄내며, 최종적으로 저비트 추론의 간결함과 효율성을 유지합니다. 실험을 통해 4비트 양자화의 견고성을 높이기 위한 실용적인 메커니즘으로 제공됩니다.



### DIFFRACT: Neuralized Utility Maximization for Wireless Networks by Differentiable Programming (https://arxiv.org/abs/2606.07114)
Comments:
          IEEE INFOCOM 2026

- **What's New**: DIFFRACT는 차별화 가능한 (differentiable) 프로그래밍을 활용한 새로운 유틸리티 극대화 프레임워크입니다. 이 프레임워크는 무선 네트워크에서 교란 관리(Interference Management)를 위해 표준 교란 함수의 수학적 구조를 이용합니다. 이를 통해 분산 가능한 그라디언트 기반 학습이 가능해지며, 지상과 비지상 환경 모두에서 실시간으로 적응할 수 있는 능력을 갖습니다.

- **Technical Details**: DIFFRACT는 표준 교란 함수의 수학적 구조를 차별화 가능한 컴퓨터 그래프에 포함하여 그라디언트 기반 학습을 지원합니다. 참여형 최적화 개념을 통해, 전통적인 최적화 문제에서 발생하는 반복 알고리즘을 신경망에 통합하여 파라미터를 엔드 투 엔드 그라디언트 하강법(Gradient Descent)을 통해 학습할 수 있습니다. 이로 인해 복잡한 채널 동적을 모델링하면서도 명확한 최적화 경로를 제공하여 실용적인 무선 네트워크 최적화를 지원합니다.

- **Performance Highlights**: 실험 결과는 DIFFRACT의 이론적 기초와 실제적인 유용성을 입증합니다. 딥 러닝과 차별화 가능한 모델을 결합한 이 프레임워크는 무선 네트워크에서 유틸리티 극대화와 빠른 적응을 가능하게 하는 학습 기반 솔루션을 제공하여 스케일과 강인성을 확보합니다. 이는 다양한 무선 환경에서도 최적화된 성능을 기대할 수 있음을 시사합니다.



### GP-Adapter: Gaussian Process CLIP-Adapter for Few-Shot Out-of-Distribution Detection (https://arxiv.org/abs/2606.07102)
Comments:
          8 pages, 6 figures, Accepted at IJCNN 2026

- **What's New**: GP-Adapter는 Gaussian Process(GP) 불확실성 모델링을 사용하여 CLIP(Contrastive Language-Image Pre-training)를 강화하는 훈련이 필요 없는 프레임워크입니다. 이 방법은 CLSP의 결정론적 유사성 점수에 의존하지 않고 데이터가 부족한 환경에서도 효율적으로 OOD(out-of-distribution) 샘플을 탐지할 수 있도록 도와줍니다. GP-Adapter는 K-shot 레이블 캐시를 이용하여 모달리티별, 클래스별 일급 GP를 구성하고, 이를 통해 OOD 탐지를 위한 신뢰 점수를 생성합니다.

- **Technical Details**: GP-Adapter는 CLIP로부터 추출한 이미지 및 텍스트 임베딩 위에 한 클래스 GP를 구성하여 GP 기반 불확실성 모델링을 제공합니다. 이 방식은 서로 다른 모달리티에 대해 독립적으로 작동하며, 예측 통계치를 결합하여 분산 인식을 위한 신뢰 점수를 생성합니다. GP의 모든 매개변수는 빠른 조정에 필요한 경량 그리드 서치를 통해 최적화되며, 메모리 비용은 $O(C K^2)$로 저비용을 유지합니다.

- **Performance Highlights**: GP-Adapter는 ImageNet과 여러 OOD 벤치마크에서 실험을 통해 경쟁력 있는 몇 샷 성능을 제공합니다. 특히, 정확도 손실 없이 prompt-learning 방법과 결합할 경우 OOD 탐지 성능이 지속적으로 향상되는 것을 보여줍니다. 이는 GP 기반 불확실성 모델링과 프롬프트 학습 간의 상호 보완성을 강조합니다.



### MetaConfigurator: AI-Assisted RDF Authoring from JSON Data (https://arxiv.org/abs/2606.07094)
Comments:
          Submitted as post-proceedings for the deRSE26 conference

- **What's New**: 이 논문에서는 JSON, YAML, 또는 CSV 데이터를 RDF로 변환할 수 있는 RDF Authoring View를 제시합니다. 메타컨피규레이터(MetaConfigurator)라는 오픈 소스 JSON 스키마 편집기를 확장하여 연구자들이 AI 지원 RML 매핑을 통해 데이터를 변환하고, SPARQL 쿼리를 실행하고, 지식 그래프를 시각화할 수 있는 통합 웹 인터페이스를 제공합니다. 이 환경은 전통적인 구조화된 데이터 관리와 시맨틱 웹 기술을 결합하여 실험적 맥락을 유지하고 기술적 장벽을 감소시킵니다.

- **Technical Details**: 이 시스템은 RDF와 JSON-LD에 대한 편집 기능을 갖추고 있으며, Ontology-aware IRI 자동 완성, JSON-LD @context 편집, SPARQL 쿼리 생성 및 지식 그래프 시각화를 통합하여 제공합니다. 사용자는 자연어 힌트를 통해 AI 지원으로 RML 매핑을 생성하거나 수동으로 작성할 수 있으며, JSON 데이터를 효율적으로 RDF로 변환할 수 있습니다. 또한 사용자에게 RDF 데이터에 대한 검사, 수정 및 쿼리 생성 기능을 제공합니다.

- **Performance Highlights**: 논문에서 제시된 워크플로우는 금속 유기 구조물의 합성 실험 데이터를 활용하여 교과 과정 데이터를 JSON에서 JSON-LD로 변환하는 과정을 보여줍니다. 실험 조건 및 결과 간의 관계를 쿼리하고, 생성된 지식 그래프를 대화식으로 탐색할 수 있습니다. 이 통합 환경은 실제 과학 데이터의 시맨틱 표현을 강화하고, 연구자들이 데이터를 보다 직관적으로 관리하고 분석할 수 있도록 돕습니다.



### On the Geometry of On-Policy Distillation (https://arxiv.org/abs/2606.07082)
Comments:
          17 pages, 8 figures

- **What's New**: 이 논문에서는 On-policy Distillation (OPD)의 파라미터 공간 내에서의 업데이틀 로직을 조사하였으며, 이를 Supervised Fine-Tuning (SFT) 및 Reinforcement Learning with Verifiable Rewards (RLVR)와 비교하였습니다. OPD는 SFT보다 더 적은 가중치에 영향을 미치면서도 RLVR보다 덜 제약된 업데이트 패턴을 드러내며, 이로 인해 '완화된 비주요 프린시플( relaxed off-principal)' 영역에 위치함을 확인했습니다.

- **Technical Details**: 연구의 중심은 OPD 업데이트의 진화 과정을 이해하는 것으로, 우리는 누적 업데이트를 기반으로 효과적인 차원, 업데이트 스케일 및 스펙트럼 형태 진단을 통해 OPD가 좁은 저차원 채널로 빠르게 진입함을 발견했습니다. 이는 OPD가 SFT보다 명확히 더 큰 누적 업데이트 노름을 가지면서도 최종적으로 유사한 안정된 랭크를 유지함을 보여줍니다. 이러한 특성은 OPD가 조기 생성된 업데이트 서브스페이스의 제약을 성공적으로 활용할 수 있음을 증명합니다.

- **Performance Highlights**: 들어본 바와 같이, OPD는 단순히 SFT와 RLVR 사이의 중간 지점이 아니라, 완화된 비주요 방향의 위치화, 초기 서브스페이스 잠금 및 목적 구성에 대한 민감성을 가진 독특한 업데이트 경로를 따릅니다. 추가 실험을 통해 OPD의 토큰 감독 밀도 및 롤아웃 정책 변경이 이러한 업데이트 경로를 유지하는 데 도움을 줄 수 있음을 보여주었습니다. 이로 인해 향후 OPD 알고리즘의 기하학적 설계를 위한 유용한 인사이트를 제공합니다.



### dots.tts Technical Repor (https://arxiv.org/abs/2606.07080)
- **What's New**: 이번 연구에서는 2B-파라미터 연속 자기회귀 텍스트-투-스피치(TTS) 모델인 dots.tts를 소개합니다. 이 모델은 지속적인 잠재 공간에서 음성을 모델링하며, 기존의 연속 자기회귀 모델과 차별화된 세 가지 혁신을 갖추고 있습니다. 첫째, 여러 목표를 가진 AudioVAE를 훈련하여 의미론적으로 구조화되고 예측 친화적인 음성 공간을 구축합니다.

- **Technical Details**: 모델은 고충실도 세멘틱 AudioVAE를 기반으로 구축되며, 오디오 표현을 정의하는 오토인코더와 그 표현을 패치별로 예측하는 자기회귀 백본으로 구성됩니다. 또한, AR 흐름 일치 헤드에서 장기 일관성을 보존하기 위해 전체 이력 조건화를 사용하여 생성 과정에서의 드리프트를 줄입니다. 마지막으로 보상 없는 자기 교정 후 훈련을 통해 음질과 강건성을 추가적으로 향상시킵니다.

- **Performance Highlights**: dots.tts는 다국어 대규모 말뭉치를 기반으로 훈련되어 Seed-TTS-Eval에서 최상의 평균 성능을 달성하였으며, WER이 각각 0.94%, 1.30%, 6.60%입니다. 또한, 다양한 벤치마크에서도 오픈소스 최첨단 성능을 지속적으로 보여주며, 강력한 생성 안정성과 음성 복제 능력 및 감정 표현성을 발휘합니다.



### SlimSearcher: Training Efficiency-Aware Web Agents via Adaptive Reward Gating (https://arxiv.org/abs/2606.07074)
Comments:
          17 pages, 8 figures,

- **What's New**: 이 논문에서는 SlimSearcher라는 새로운 프레임워크를 제안합니다. SlimSearcher는 정확도와 계산 비용 간의 균형을 맞추기 위해 효율성을 최적화하는 접근 방식을 전면에 내세웁니다. 이를 통해 기존의 비효율적인 접근 방식의 단점을 극복하고, 더 나은 효율성과 성능을 달성할 수 있도록 돕습니다.

- **Technical Details**: SlimSearcher는 Supervised Fine-Tuning (SFT)과 Reinforcement Learning (RL) 단계에서 효율성을 고려한 필터링 및 적응형 보상 구조를 사용합니다. 이 시스템은 기존의 툴 사용 의존성과 과도한 반복 작업을 줄여 최적의 경로를 찾는 것을 목표로 합니다. Adaptive Reward Gating이라는 동적 보상 메커니즘을 통해 툴과 토큰 효율성을 평가하고, 보상을 통해 모델이 최소한의 경로를 학습하도록 유도합니다.

- **Performance Highlights**: SlimSearcher는 GAIA, BrowseComp, XBenchDeepSearch와 같은 벤치마크에서 평균 툴 호출 라운드를 17%에서 58%까지 감소시켰으면서도 정확도를 유지하거나 향상시키는 성과를 보여줍니다. 이러한 성과는 SlimSearcher의 효율성 강화 접근 방식이 실제로 성능에 긍정적인 영향을 미친다는 것을 증명합니다.



### TRACE: Trajectory Reasoning through Adaptive Cross-Step Evidence Aggregation for LLM Agents (https://arxiv.org/abs/2606.07054)
- **What's New**: 이 논문에서는 LLM(대규모 언어 모델) 에이전트의 자동 감시를 위한 TRACE라는 새로운 모니터링 프레임워크를 제안합니다. 이 프레임워크는 에이전트의 행동과 관련된 숨겨진 악의적 목표를 추적할 수 있는 능력을 향상시킵니다. TRACE는 TIJ(Triage-Inspect-Judge) 루프를 통해 고신호 지역을 식별하고, 조사를 수행하며, 최종적으로 경과된 결과를 종합하여 결정합니다.

- **Technical Details**: TRACE는 에이전트가 사용자 요청을 수행하는 과정에서 발생하는 일련의 사고(x\_1, x\_2, ..., x\_T)에서의 추적을 감시합니다. 이 시스템은 악의적 행동이 다른 plausible 행동의 흐름 속에서 어떻게 숨겨져 있는지를 효과적으로 탐지하기 위해, 의심스러운 신호를 집중적으로 조사하고, 장기적인 증거를 축적하며, 분산된 신호를 연결합니다. 이를 위해 TRACE는 TIJ 루프 구조를 사용하여 의심스러운 처리를 체계적으로 접근합니다.

- **Performance Highlights**: TRACE는 SHADE-Arena의 10개 작업 영역에서 평가되었으며, 전체 F1 스코어는 0.713, 재현율은 0.844에 달했습니다. 특히, 증거 연결이 필요한 작업에서 가장 큰 성과를 보였습니다. TRACE의 판단 프레임워크는 기존의 모니터링 방법들보다 더 나은 성능을 보여주며, 악의적 행동의 손상 징후를 더욱 효과적으로 탐지하는 데 기여합니다.



### STREAM: Stochastic Riemannian Flow Matching with Anisotropic Decoder for Digital Histopathology Image Generation (https://arxiv.org/abs/2606.07036)
Comments:
          27 pages, 7 figures

- **What's New**: 이 논문은 합성 병리학적 이미지 생성의 새로운 접근 방식인 STREAM을 소개합니다. 이 방법은 Latent Diffusion Model을 활용하여 생긴 기존의 'Conditioning Collapse' 문제를 해결하기 위해, 사전 훈련된 병리학 VFM의 패치-토큰 특징을 잠재 공간으로 활용합니다. 이는 생성된 이미지를 더욱 다양화하고 질을 향상시킵니다.

- **Technical Details**: STREAM은 두 단계로 구성됩니다: 첫 번째 단계는 $	ext{SLERP}$ 지오데식에서 발생하는 브리지 타입의 임의 교란을 설정하여 잠재 공간에서 Diffusion Transformer를 훈련시키는 것입니다. 두 번째 단계에서는 높은 에너지 방향을 보존하면서 저에너지 방향에서의 강건성을 할당하는 새로운 비등방형 디코더를 적용합니다. 이러한 설계를 통해 생성 과정에서의 품질이 향상됩니다.

- **Performance Highlights**: STREAM은 유방 암 진단 및 대장암 데이터셋에서 최첨단 재구성 및 생성 성능을 달성하여 기존 모델들을 능가합니다. 이 모델은 임상 데이터의 개인 정보 보호와 대규모 훈련 데이터의 필요성을 충족시키는데 기여할 것입니다. 코드 또한 논문 수락 시 공개될 예정입니다.



### Never Seen Before: Benchmarking Genuine Zero-Shot Composed Image Retrieval with Consistent Video-Sourced Datasets (https://arxiv.org/abs/2606.07032)
- **What's New**: 이번 연구는 Zero-Shot Composed Image Retrieval (ZS-CIR) 분야에서 새로운 벤치마크인 ZeroSight를 제안합니다. 기존 ZS-CIR 데이터셋에서는 참조 이미지와 목표 이미지 간의 완전한 불일치가 문제였으며, 이를 해결하기 위해 영상에서 수집한 일관된 참조-목표 쌍을 포함하는 데이터셋을 구축했습니다. 연구는 또한 CLIP이 미리 훈련된 데이터가 포함되지 않도록 최근 영상 데이터를 사용하여 진정한 제로샷 상황을 보장합니다.

- **Technical Details**: ZeroSight는 12,048개의 다양한 비디오에서 197,313개의 후보 이미지와 54,740개의 쿼리를 바탕으로 구성된 CIR 데이터셋과 데이터 구축 파이프라인을 포함하고 있습니다. 연구팀은 고급 LLM 기법을 활용하여 시각적 및 의미적으로 일관된 이미지 쌍을 생성하고, SC4CIR(대칭 일관성을 위한 CIR)이라는 학습 필요 없는 방법을 통해 하드 네거티브 타겟도 효과적으로 식별합니다. 이 방법은 다양한 CIR 방법과 원활하게 통합되어 성능을 크게 향상시키는 것이 특징입니다.

- **Performance Highlights**: ZeroSight에서 수행한 실험 결과에 따르면, 기존 CLIP 기반 CIR 방법들은 기존 ZS-CIR 데이터셋에서 성능이 부풀려진 결과를 나타내며, CIR 방법의 능력을 과장하게 됩니다. 연구는 해당 벤치마크를 통해 다수의 긍정 및 부정 타겟 이미지의 순위를 고려한 평가 방식을 제시하여 현재 CIR 방법들의 역량을 정밀하게 평가할 수 있도록 합니다. 이 연구는 AI와 이미지 검색 기술의 발전에 기여할 것으로 기대됩니다.



### Phonetic Error Analysis of Raw Waveform Acoustic Models (https://arxiv.org/abs/2606.07030)
Comments:
          INTERSPEECH2026

- **What's New**: 이번 연구는 전통적인 전화 인식 시스템의 성능 측정 방식인 전화 오류율(PER) 이상의 오류 패턴을 분석합니다. 전화가 광범위한 음소 클래스를 고려하여 분류된 점에서 큰 기여를 합니다. 또한, 기존의 Filterbank 기반 시스템의 분석을 원시 파형(raw waveform) 음향 모델로 확장한 점이 주목할 만합니다.

- **Technical Details**: 모델은 파라메트릭(SincNet, Sinc2Net) 또는 비파라메트릭 CNN과 Bidirectional LSTM(BLSTM)을 조합하여 구성됩니다. BLSTM 계층의 효과는 강한 시간적 동적을 가지는 클래스에서 가장 두드러지게 나타났으며, 특히 이중모음, 마찰음 및 반모음에서 효과가 큽니다. 또한, WSJ에서 학습한 전이 학습이 자음의 오류를 유지하면서 주요 변화가 발생하는 것을 확인했습니다.

- **Performance Highlights**: 제안된 모델은 TIMIT 데이터 셋에서 원시 파형 모델 중 가장 낮은 PER 15.3%를 기록하며, WSJ 전이 학습을 통해 PER을 11.3%로 줄였습니다. 전체적으로, BLSTM 통합 모델이 다른 시스템에 비해 뛰어난 성능을 보였으며, 특히 논리 통계적으로 유의미한 자음-모음의 증가 비대칭을 관찰했습니다.



### Towards Unified Song Generation and Singing Voice Conversion with Accompaniment Co-Generation (https://arxiv.org/abs/2606.07015)
- **What's New**: 이번 연구는 UniSinger라는 최초의 end-to-end 프레임워크를 제안하여 송 생성(song generation)과 목소리 변환(SVC)을 통합하였습니다. 이는 기존의 두 독립적인 접근 방식을 결합하여 보컬 클로닝(zero-shot speaker cloning)과 반주 동기화(accompaniment co-generation)를 가능하게 합니다. 연구에서는 멀티모달(diffusion transformer) 아키텍처를 활용하여 두 작업 간의 음색 조정(timbre control)을 세밀하게 수행할 수 있는 방법을 제공합니다.

- **Technical Details**: UniSinger는 네 가지 핵심 요소로 구성되어 있습니다: 멀티모달 입력 처리(multi-modal input processing), 점진적 커리큘럼 학습(progressive curriculum learning), 크로스-태스크 스피커 임베딩(cross-task speaker embedding space), 및 MM-DiT 백본(MM-DiT backbone)입니다. 이러한 구성요소들은 다양한 입력을 공유 잠재 공간(shared latent space)으로 변환하여 세밀한 보컬 조건을 유지하도록 돕습니다. 점진적 커리큘럼 학습 전략은 작업-특유 모달리티 마스킹(task-specific modality masking)을 포함해 모델이 단계적으로 음성 합성(vocal synthesis)과 텍스트 기반 동반 생성(accompaniment generation)의 메커니즘을 마스터할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과는 UniSinger가 두 가지 작업 모두에서 최첨단 성능을 보여주며, 송 생성에서의 글로벌 음악 구조(global musical structure)가 SVC의 발음과 조화를 개선하는 상호 보완적 이점을 제공함을 확인하였습니다. 또한, 우리의 방법은 음성과 배경 음악 간의 조화(acoustic synergy)를 충분히 활용하여 퀄리티를 높이고 있습니다. 최종적으로, UniSinger는 지능형 음악 제작(intelligent music production)에 새로운 가능성을 열어줍니다.



### A Geometric View for Understanding Concept Learning and Neuron Interpretation in Sparse Autoencoders (https://arxiv.org/abs/2606.07007)
- **What's New**: 이 논문에서는 희소 오토인코더(SAE)에서 개념 학습과 뉴런 해석을 위한 통합 수학적 프레임워크를 제안합니다. 이는 모델이 인간이 정의한 개념과 모델이 생성한 개념 사이의 집합 정렬(set alignment) 문제로 개념 학습을 포착합니다. 이러한 접근법은 개념 탐지(detection), 분리(separation), 근사(approximation)의 세 가지 학습 모드를 구분하고, 이러한 개념들이 개별 뉴런 또는 다중 뉴런 유닛에 의해 표현될 수 있는 경우에 대한 기하학적 조건과 오류 경계를 제공합니다.

- **Technical Details**: 논문은 개념을 데이터 포인트의 집합으로 공식화하고, SAE 뉴런의 해석을 데이터 예제 집합의 특성을 나타내는 방식으로 접근합니다. 이 작업에서는 개념 학습을 목표로 하는 인간의 개념과 모델이 유도한 개념 간의 정렬로 정의하고, 이를 집합 이론적 구조와의 연결을 통해 설명합니다. 이 프레임워크는 개념 학습의 서로 다른 모드를 구별하고, 이러한 모드가 발생하는 조건을 유도하며, SAE와 관련된 여러 현상을 설명합니다.

- **Performance Highlights**: 실험은 ReLU 및 Top-K SAEs를 사용하여 이론을 입증하고, SAE의 크기와 희소성이 개념 학습에 미치는 영향을 분석합니다. 실험 결과는 희소 오토인코더가 전문가들이 이해할 수 있는 개념을 더 잘 해석하게 하는데 기여한다는 것을 보여줍니다. 또한, 이러한 연구는 계층적 개념 구조를 정의하는 포멀 개념 격자(formal concept lattice)를 통해 개념 학습과 뉴런 해석 간의 관계를 명확히 합니다.



### DataEvolver: Automatic Data Preparation for Large Language Models through Multi-Level Self-Evolving (https://arxiv.org/abs/2606.07001)
- **What's New**: 이 논문에서는 데이터 품질을 향상시키기 위한 첫 번째 자가 진화 데이터 준비 시스템인 DataEvolver를 소개합니다. 기존의 데이터 준비 방법들은 사전 정의된 파이프라인이나 사용자 지정 인간 지침에 의존하여 다양한 데이터 분포에 대한 적응력을 제한합니다. DataEvolver는 원시 데이터를 고품질 데이터로 자동 변환하기 위해 파이프라인을 구축하는 기능을 제공합니다.

- **Technical Details**: DataEvolver는 다단계 자가 진화 메커니즘을 활용하여 파이프라인 실행 가능성(executability)과 효과성(effectiveness)을 보장합니다. 운영자 수준에서는 운영자 집합을 점진적으로 확장하여 논리 계획을 구성하고 의존성 충돌을 해결합니다. 파이프라인 수준에서는 논리 계획을 실행 가능한 코드로 변환하고 피드백 루프를 통해 파이프라인 오케스트레이션을 반복적으로 정제합니다.

- **Performance Highlights**: 실험 결과, DataEvolver는 7개의 벤치마크에서 데이터 품질을 크게 향상시키고 원시 데이터에 비해 평균 10%의 성능 향상을 달성했습니다. 이는 LLM의 훈련 데이터 품질 향상뿐만 아니라 LLM과 데이터의 반복적인 공동 진화를 위한 새로운 기회를 나타냅니다. 또한, DataEvolver는 자가 진화 관점에서 LLM 데이터 준비를 재구성하는 첫 번째 사례로, 데이터 준비 자동화에 대한 중요한 발전을 보여줍니다.



### Don't Pause: Streaming Video-Language Synchrony for Online Video Understanding (https://arxiv.org/abs/2606.06991)
- **What's New**: 이번 논문에서 저자들은 새로운 온라인 비디오 이해 패러다임인 Streaming Video-Language Synchrony(SVLS)를 제안하며, 이를 바탕으로 LyraV라는 라이브 스트리밍 보조 도구를 개발하였습니다. LyraV는 두 가지 주요 혁신을 통해 작동합니다: Frame-Driven Transition Controller(FDTC)와 Streaming Token Pacer(SToP)입니다. 이 시스템은 영상 프레임과 언어 생성의 동기화를 향상시키며 실제 시간 처리 속도에서 높은 성능을 보여줍니다.

- **Technical Details**: LyraV는 Frame-Driven Transition Controller(FDTC)와 Streaming Token Pacer(SToP)라는 두 가지 구성 요소로 구성됩니다. FDTC는 고차원적으로 상황을 판단하여 언제 말을 계속할지, 새로운 응답을 시작할지 또는 침묵할지를 결정합니다. SToP는 실시간 지연 제약 내에서 프레임당 발화율을 동적으로 조정하여 비주얼 콘텐츠의 속도에 맞춰 언어 생성 속도를 최적화합니다.

- **Performance Highlights**: LyraV는 5개의 온라인 및 3개의 오프라인 벤치마크를 통해 검증되었습니다. LyraV는 비디오 재생과 98.29%의 동기화를 유지하며, 3.89 FPS의 실시간 처리 속도를 제공합니다. 이 시스템은 또한 비디오가 진행되는 동안 지속적으로 해석하고 '생각'할 수 있는 능력을 보여줍니다.



### DaX: Learning General Pathology Representations Across Scales (https://arxiv.org/abs/2606.06983)
- **What's New**: 이번 논문에서는 다양한 임상 지표에 걸쳐 전이될 수 있는 시각적 표현을 요구하는 컴퓨터 병리학(Computational Pathology)을 위해 DaX라는 새로운 병리학 비전 기초 모델을 소개합니다. DaX는 전통적인 DINOv3 스타일의 self-supervised learning을 전체 슬라이드 히스토병리학에 적응할 수 있도록 설계되었습니다. 이 모델은 자연 이미지 DINOv3 가중치로 초기화되며, 다양한 훈련 기법을 포함하고 있습니다.

- **Technical Details**: DaX는 지속적인 배율 훈련(continuous magnification training), 크로스 스케일 조직 뷰(cross-scale tissue views), 방향 무관(orientation-agnostic) 및 촬영 강건성(acquisition-robust) 증강(augmentation), 다양한 입력 크기(multi-input-size) 훈련, 그리고 Gram-anchored dense consistency를 통합합니다. 이러한 설계들은 국소 세포 형태(local cellular morphology)와 전반적인 조직 구조(global tissue architecture)를 연결하고, 입력 스케일 전반에 걸쳐 조밀한 토큰 수준 표현(dense token-level representations)을 안정화하기 위해 마련되었습니다.

- **Performance Highlights**: 제안된 모델은 44개의 공개 데이터셋에서 161개의 임상적으로 중요한 과제를 포함한 WSI 수준 벤치마크를 통해 평가되었습니다. DaX는 모든 임상 영역과 아홉 개의 작업 범주를 커버하며, 고정된 환자 수준 크로스 검증 프로토콜 아래에서 평가되었습니다. 이 결과 DaX는 각 과제에서 평균적으로 가장 높은 성능을 기록하였고, 진단 병리학, 바이오마커 및 분자 프로파일링 등 다양한 영역에서 안정적인 작업 수준 순위를 달성하였습니다.



### OpenHalDet: A Unified Benchmark for Hallucination Detection across Diverse Generation Scenarios (https://arxiv.org/abs/2606.06959)
Comments:
          Preprint. Code and data are available at this https URL

- **What's New**: OpenHalDet는 환각(hallucination) 감지를 위한 통합 벤치마크로, 다양한 생성 시나리오에서의 신뢰할 수 있는 평가를 목표로 한다. 기존의 환각 탐지 방법들이 일반적으로 채택하는 비일관적인 평가 구성 및 제한된 하위 도메인으로 인해 발생하는 문제점을 해결한다. 이를 통해 OpenHalDet는 검증된 구조를 갖춘 코드베이스를 제공, 향후 연구 및 개발을 위한 재현 가능한 평가를 용이하게 만든다.

- **Technical Details**: OpenHalDet는 프롬프트 생성, 응답 생성, 진실성 주석, 탐지기 점수화 및 메트릭 계산 등 주요 평가 단계를 표준화하였다. 또한, 블랙박스(black-box), 그레이박스(gray-box), 화이트박스(white-box) 방법을 포괄하여 다양한 탐지 모델을 수용할 수 있도록 설계되었다. 최종적으로, 17개의 데이터셋과 4개의 LLM 백본을 통한 전체 평가를 포함한다.

- **Performance Highlights**: OpenHalDet는 효율적이고 포괄적인 평가 체계를 통해 다양한 기존 연구와 비교할 수 있는 공정한 기준을 제공한다. 이를 통해 다양한 작업 및 문맥에서의 탐지기 성능을 보다 신뢰성 있게 분석할 수 있다. 기존 연구들이 좁은 작업에만 초점을 맞춘 반면, OpenHalDet는 다양한 시나리오에 대해 보다 넓은 범위의 평가를 지원한다.



### When is 3D Worth It? A Resource-Performance Frontier for CNNs and Transformers in Lung C (https://arxiv.org/abs/2606.06950)
Comments:
          8 pages, 6 figures

- **What's New**: 이 연구에서는 3D 모델이 일반적으로 선호되지만, 실질적인 성능 향상 여부가 계산 비용과 복잡성을 정당화하는지를 분석합니다. 2D, 2.5D, 3D의 입력 차원을 바탕으로 CNN 및 Vision Transformers의 모델 동작을 비교하며, 2.5D CNN이 가장 우수한 성능-안정성 균형을 제공한다고 보고합니다. 특히, 3D CNN은 임계값 불안정성이 나타났고, transformers는 모든 양성 예측으로 나타나는 한계를 보였습니다.

- **Technical Details**: NLST 코호트(n=1,977) 및 LIDC-IDRI 데이터를 사용한 연구로, 2D, 2.5D, 3D 입력을 각각 중앙 축 단면, 세 개의 직교 슬라이스, 그리고 폐 중심 서브 볼륨으로 정의합니다. 두 가지 네트워크 유형인 residual CNN과 Vision Transformer를 이 입력에 맞게 조정하여 동일한 학습 프로토콜(20 에포크; 가중 이진 크로스 엔트로피)을 통해 훈련하였습니다. 평가 기준으로 ROC-AUC, PR-AUC 및 민감도/특이도를 사용하였으며, 부트스트랩 신뢰구간 또한 보고합니다.

- **Performance Highlights**: 연구 결과, 2.5D CNN이 0.682의 ROC-AUC 및 0.158의 PR-AUC를 기록하며 가장 높은 성능을 보여주었습니다. 2D CNN은 지나치게 보수적이었고, 3D CNN은 임계값에 따라 불안정성을 보였습니다. transformers는 높은 GPU 메모리 요구 및 예측의 수렴 부족으로 인해 제대로된 성능을 내지 못한 것으로 나타났습니다.



### Auditing Training Data in Domain-adapted LLMs: LoRA-MIN (https://arxiv.org/abs/2606.06946)
Comments:
          IEEE Conf. on Computers, Software, and Applications (COMPSAC), 2026

- **What's New**: 본 논문에서는 LoRA-MINT라는 새로운 방법론을 제안합니다. 이 방법론은 특정 자연어 처리(NLP) 작업을 위해 조정된 대규모 언어 모델(LLM)의 회원 추론 테스트(Membership Inference Test, MINT)에 적용됩니다. LoRA 기반의 모델이 훈련 데이터에 포함된 샘플인지 판단하는 것을 목표로 하여, 지적 재산권 및 민감한 데이터 관리에 유용한 감사 도구를 제공합니다.

- **Technical Details**: LoRA-MINT의 주요 구성 요소는 모델의 perplexity(혼란도)와 회원 상태 간의 관계를 탐구하고, 데이터의 노출 정도를 추정하는 체계적인 프레임워크를 제공합니다. 저자들은 네 가지 모델과 세 가지 벤치마크 데이터셋을 대상으로 실험을 수행하여 훈련 데이터 사용 여부를 판단하는 정확도 값이 0.77에서 0.92 사이임을 확인했습니다. LoRA 기술을 기반으로 한 파라미터 효율적인 미세 조정이 어떻게 보다 나은 성능을 제공하는지를 입증합니다.

- **Performance Highlights**: LoRA-MINT는 훈련 샘플과 비훈련 샘플을 효과적으로 구별할 수 있는 도구로, 기존의 최첨단 기법들을 능가하는 성능을 보여줍니다. 저자들은 이 방법이 LLM 감사 및 AI 기술의 윤리적이고 책임 있는 배포를 촉진하는 데 있어 중요한 잠재력을 지닌다고 강조합니다. LoRA-MINT는 범용적이고 확장성 있는 감사 도구로서, 훈련 중 데이터 노출을 탐지하여 투명성을 향상시키는 데 기여합니다.



### SS-TPT: Stability and Suitability-Guided Test-Time Prompt Tuning for Adversarially Robust Vision-Language Models (https://arxiv.org/abs/2606.06943)
Comments:
          Accepted in ICML2026

- **What's New**: 본 논문에서는 Vision-language models (VLMs)인 CLIP가 강력한 zero-shot 인식을 달성하였지만, 적대적 변형(adversarial perturbations) 하에서 쉽게 손상되는 문제를 다룹니다. 이를 해결하기 위해 Stability and Suitability-guided Test-time Prompt Tuning (SS-TPT)라는 새로운 방법을 제안합니다. SS-TPT는 각 증강(view)된 이미지의 품질을 평가하기 위해 두 가지 상호 보완적인 점수인 안정성(stability)과 적합성(suitability)을 활용합니다.

- **Technical Details**: 안정성은 약한 변형에 대한 예측 불변성을 측정하고, 적합성은 여러 뷰(view) 간의 특징 공간 밀도를 평가합니다. 이 두 점수는 SS-guided consistency loss와 SS-weighted prediction을 통해 적응(adaptation) 및 추론(inference)을 안내합니다. 이를 통해 신뢰할 수 있는 뷰를 강화하고 손상된 뷰를 억제하는 방식으로 시스템의 견고함을 증가시킵니다.

- **Performance Highlights**: SS-TPT는 방대한 실험을 통해 이전의 최첨단(state-of-the-art) 방법들보다 월등한 성능을 보여주며, 다양한 데이터셋과 뷰 수에서 뛰어난 견고성-처리량(robustness-throughput) 거래 균형을 달성했습니다. 이는 SS-TPT의 실용성과 일반성을 동시에 입증하는 결과입니다. 코드는 제공된 URL에서 확인할 수 있습니다.



### Didact: A Cross-Domain Capability Discovery System for Defenc (https://arxiv.org/abs/2606.06942)
Comments:
          Under Review at CIKM 2026 (System Demonstration Track)

- **What's New**: 이 논문에서는 호주에서 공개적으로 제공되는 방위 보고서와 정책 문서를 통합하는 프로토타입 툴인 Didact를 소개합니다. Didact는 방위 및 학술 연구 분야의 정보를 보다 효율적으로 탐색하는 기능을 제공하며, 사용자가 자연어 대화를 통해 관련 정보에 접근할 수 있도록 합니다. 특히, 다원적 Retrieval-Augmented Generation (RAG) 파이프라인을 활용하여 모든 관련 소스를 연결하고, 상호작용할 수 있는 Evidence Rail을 통해 증거 및 출처 관계를 시각화하는 점이 특징입니다.

- **Technical Details**: Didact는 사용자의 질문을 입력으로 받아 텍스트 응답과 함께 관련 문서 조각 및 부그래프의 Evidence Rail을 생성하는 파이프라인으로 구현되어 있습니다. 이 시스템은 두 가지 소스 유형(호주 방위 문서 및 연구 지식 그래프)을 활용하며, 별도의 접근 수준으로 문서를 분리하여 보안을 유지합니다. Didact는 사용자 쿼리에서 도메인 관련 키워드를 추출하여 지식 그래프를 통해 적절한 정보를 검색하고, 복합 RAG 구조를 통해 다양한 질문에 대한 응답을 효과적으로 생성할 수 있습니다.

- **Performance Highlights**: Didact는 방위 분야의 정책 결정자 및 분석가들에게 다원적 방식으로 정보를 탐색할 수 있는 혁신적인 도구로서, 실험 평가를 통해 그 유용성을 입증하였습니다. 이 시스템은 사용자 인터페이스에서 사용하는 Evidence Rail을 통해 데이터의 출처를 손쉽게 추적할 수 있고, 적절한 의사 결정에 필요한 증거 기반 정보를 제공합니다. Didact는 타 분야로의 확장이 가능하여 정보가 분산된 다른 영역에서도 유용하게 활용될 수 있는 잠재력을 가지고 있습니다.



### The Fine-Tuning Trap: Evaluating Negative Transfer and the Role of PEFT in Sub-1B Mathematical Reasoning (https://arxiv.org/abs/2606.06920)
Comments:
          8 pages, 6 figures, 2 tables

- **What's New**: 본 연구에서는 1B 미만의 소형 언어 모델(Small Language Models, SLMs)을 엣지 디바이스에 배포하는 효율적인 파인튜닝 전략을 제안합니다. 특히, 모델 파라미터가 300M 미만인 경우 전체 파인튜닝(Full Fine-Tuning, Full FT)이 성능에 부정적인 영향을 미친다는 점을 발견했습니다. 이는 '부정적 이전'(negative transfer)을 초래하고, 그래서 파라미터 효율적인 파인튜닝(Parameter-Efficient Fine-Tuning, PEFT) 방법을 기본으로 사용하는 것이 필수적이라는 결론에 도달했습니다.

- **Technical Details**: 연구에서는 저차원 밀집 모델이 높은 민감성이 있기 때문에 파인튜닝 중 최적화가 부적절한 방향으로 진행될 수 있음을 설명합니다. 이는 Hessian 분광 분석을 통해 모델의 안정성 격차를 이론적으로 정형화하여, Full FT의 실패를 저용량 경량화에서의 급격한 최소값 전이와 관련지었습니다. PEFT 방법, 특히 저랭크 적응(LoRA)과 가중치 분해 저랭크 적응(DoRA)은 적응 시 중요한 전이 지식을 유지하도록 설계되었습니다.

- **Performance Highlights**: 성능 측면에서 LoRA와 DoRA는 여러 작업에서 다르게 작용하며, DoRA는 복잡한 추론 작업에서는 우수한 성능을 보이고, LoRA는 패턴 매칭에서 두각을 나타냈습니다. 특히, 300M 미만의 모델에 대해 Full FT보다 LoRA가 더 낮은 성능 저하를 보였으며, 실험 결과는 PEFT를 기본으로 사용할 것을 제안합니다. 500M보다 작은 모델에 대해 전체 파인튜닝을 피해야 한다는 점 또한 강조하고 있습니다.



### ThinkBooster: A Unified Framework for Seamless Test-Time Scaling of LLM Reasoning (https://arxiv.org/abs/2606.06915)
- **What's New**: ThinkBooster는 LLM(대형 언어 모델)의 테스트 시간 컴퓨팅(TTC) 스케일링을 위한 통일된 프레임워크입니다. 이 프레임워크는 TTC 스케일링 전략과 스코어러(scorer) 가족을 구현하는 모듈식 파이썬 라이브러리, 성능 및 계산 효율성을 종합적으로 평가하는 벤치마크, 실제 애플리케이션에 적응적 추론을 통합할 수 있는 OpenAI 호환 프록시 서비스로 구성되어 있습니다. Empirical results는 ThinkBooster가 수학적 및 코딩 과제에서 실제적인 성능 향상을 제공함을 보여줍니다.

- **Technical Details**: 이 라이브러리는 다양한 TTC 스케일링 전략과 스코어러를 포함하여, 최적의 성능-비용 트레이드오프를 검토할 수 있도록 지원합니다. TTC 스케일링 전략은 오프라인과 온라인에서 모두 작동할 수 있으며, 각 알고리즘의 세부적인 구현은 PyTorch와 Hugging Face 라이브러리를 통해서 이루어집니다. 그 외에도, 다양한 종류의 스코어러가 LLM 출력이나 내부 상태에 따라 평가되고, 이는 white-box와 black-box 전략으로 구분되어 사용됩니다.

- **Performance Highlights**: ThinkBooster는 실질적인 코딩 및 수학적 문제 해결에서 TTC 스케일링 전략의 성과를 입증했습니다. 이러한 개선은 직관적인 API 및 통합 레이어를 통해 실 세계 애플리케이션의 배포를 용이하게 합니다. 그러나 TTC 스케일링의 도입 장벽을 낮추는 동시에, 연구자들이 연구를 진행할 수 있는 벤치마크도 제공합니다.



### SpectCount: Spectrotemporal Counting via Synthetic Signals Improves Large Audio Language Models (https://arxiv.org/abs/2606.06907)
Comments:
          5 pages, 5 figures

- **What's New**: 본 연구에서는 대규모 오디오 언어 모델(LALM)의 특정 스펙트로템포럴(perceptual) 약점을 파악하고 이를 해결하기 위한 데이터 효율적인 파인튜닝 방법인 Spectrotemporal Counting(SpectCount)를 제안합니다. SpectCount는 실제 오디오 데이터나 주석 없이 생성된 순수 합성 신호를 사용하여 훈련되며, 다양한 오디오 기준에서 성능 향상을 보여줍니다. 이러한 접근법은 LALM의 청각 이해 능력을 향상시키기 위한 데이터 효율적인 방법을 제공하는 중요한 방향입니다.

- **Technical Details**: SpectCount는 초과 분포된 신호를 기반으로 하여 밀리초 규모의 소리 이벤트를 감지할 수 있는 모델을 훈련합니다. 각 신호는 다양한 빈도와 시간 위치에서 짧은 펄스(pulse)를 포함하며, 이들은 신호의 세부적인 스펙트로템포럴 정보를 집계하는 데 중요합니다. 모델은 Low-Rank Adaptation(LoRA) 기법을 사용하여 파인튜닝되며, 이는 훈련된 파라미터를 유지하면서도 효율성을 극대화합니다.

- **Performance Highlights**: 실험 결과, SpectCount는 MMAU, MMAR, MMSU, AIR-Bench와 같은 다양한 청각 기준에서 LALMs의 성능을 향상시킵니다. 이러한 결과는 합성 신호만으로도 LALM의 청각 이해 능력이 크게 개선될 수 있음을 보여줍니다. 연구진은 이 방법이 기존 모델들이 가지던 스펙트로템포럴 약점을 크게 보완할 수 있음을 입증하였습니다.



### EASE-TTT: Evidence-Aligned Selective Test-Time Training for Long-Context Question Answering (https://arxiv.org/abs/2606.06906)
Comments:
          13 pages, 4 figures, 3 tables

- **What's New**: 본 논문에서는 Evidence-Aligned SElective Test-Time Training (EASE-TTT)이라는 새로운 테스트 타임 트레이닝 프레임워크를 제안합니다. 이 방법은 질문과 관련된 증거(chunk)들을 선택하여 소프트 어텐션(supervision target)으로 변환하며, 이를 모델의 훈련에 반영하여 장문의 질문 응답(long-context QA) 성능을 개선합니다. EASE-TTT는 증거를 단순히 입력에서 가져오는 것이 아니라, 모델의 어텐션 메커니즘을 최적화하여 전체 맥락에서의 응답 품질을 높입니다.

- **Technical Details**: EASE-TTT는 먼저 입력 맥락에서 질문과 가장 관련성이 높은 증거 조각(chunk)을 선택합니다. 이후, EASE-TTT는 이러한 선택된 증거 위치에 대한 소프트 어텐션 타겟을 만들어, 원래의 맥락은 유지하되 더욱 높은 가중치를 부여합니다. 테스트 시점에서 경량 쿼리 전용 어댑터가 업데이트되며, 이를 통해 모델은 최종 응답을 전체 맥락에서 생성합니다.

- **Performance Highlights**: EASE-TTT는 여러 개의 장문 질문 응답(Task) 벤치마크에서 실험을 수행하였고, 기존의 전체 맥락 추론, 단순 증거 검색(baselines), 쿼리 전용 테스트 타임 트레이닝(qTTT)보다 더 나은 성능을 보였습니다. 실험 결과, EASE-TTT는 선택된 증거의 중요성과 소프트 어텐션 감독의 효과를 통해 모델의 성능을 크게 향상시키는 것으로 나타났습니다.



### Beyond Skeletons: Learning Animation Directly from Driving Videos with Same2X Training Strategy (https://arxiv.org/abs/2606.06903)
Comments:
          Accepted to ICLR 2026

- **What's New**: 이번 연구에서는 DirectAnimator라는 프레임워크를 제안하며, 이는 중간 단계의 포즈 추출을 생략하고 원시 드라이빙 비디오에서 직접 학습합니다. 기존 연구는 주로 포즈 추정기를 이용하여 중간 표현을 추출하지만, 이는 차단 또는 복잡한 자세에서 오류가 발생하기 쉽습니다. DirectAnimator는 포즈, 얼굴, 위치 정보를 포함하는 Driving Cue Triplet을 도입하여 안정적이고 의미론적으로 풍부한 형태로 모션과 표정을 포착합니다.

- **Technical Details**: DirectAnimator는 CueFusion DiT 블록을 통해 포즈, 얼굴, 위치 단서를 융합하여 데이터 노이즈 제거 과정에서 신뢰성을 높입니다. 또한, Same2X 훈련 전략을 통해 서로 다른 신원 간의 피쳐를 정렬하여 최적화를 정규화하고 수렴 속도를 가속화합니다. 이 시스템은 관찰된 신원과 드라이빙 신원이 다를 때에도 안정적인 학습을 가능하게 합니다.

- **Performance Highlights**: 광범위한 실험을 통해 DirectAnimator는 최첨단 시각적 품질을 달성하며 신원 보존이 뛰어난 성능을 보였습니다. 또한, occlusion(차단)과 복잡한 신체 자세에 견고하며, 더 적은 계산 자원으로도 높은 품질을 유지합니다. 이러한 결과는 DirectAnimator의 구조적 접근 방식이 효과적임을 잘 보여줍니다.



### FreeAnimate: Training-Free Human Image Animation with Preview-Guided Denoising (https://arxiv.org/abs/2606.06885)
Comments:
          Accepted to IEEE ICASSP 2026

- **What's New**: 이번 연구에서는 FreeAnimate라는 훈련이 필요 없는 인간 이미지 애니메이션 프레임워크를 소개합니다. 기존의 방법들은 대량의 훈련 데이터와 자원을 요구했으나, FreeAnimate는 사전 훈련된 이미지 확산 모델의 잠재력을 활용하여 시간 일관성(temporal consistency), 정체성 보존(identity preservation), 배경 안정성(background stability)을 확보합니다. 이 접근 방식은 새로운 미리보기 생성 전략(preview generation strategy)을 통합하여 훈련 없이도 포즈 정렬(pose alignment)과 배경의 일관성(background consistency)을 효과적으로 잠재울 수 있습니다.

- **Technical Details**: FreeAnimate는 두 개의 주요 모듈인 Inversion-Boosted Attention(IBA)와 Reference-Anchored Self-Attention(RA-SA)을 도입하여 시간 일관성과 정체성 보존을 보장합니다. IBA는 미리보기 프레임에서 얻은 주의(attention) 맵을 활용하여 개선된 구조적 일관성을 달성합니다. RA-SA는 레퍼런스 이미지를 기준으로 프레임의 일관성을 강화하여 최종적으로 품질 높은 비디오 애니메이션을 생성하는 데 기여합니다.

- **Performance Highlights**: 실험 결과, FreeAnimate는 기존의 훈련이 필요 없는 경쟁자들과 훈련 기반의 기준 방법들을 초과하여, 최첨단 방법들과 동등한 생성 품질을 달성하였습니다. 다양한 데이터셋에서 강력한 일반화 능력을 보여주어, 실제 애플리케이션에서의 사용 가능성을 높이고 있습니다.



### Neuro-Symbolic Learning for Long-Horizon Task Planning Under Complex Logical Constraints (https://arxiv.org/abs/2606.06877)
- **What's New**: 이번 논문에서는 로봇의 작업 계획(task planning) 효율성을 개선하기 위해 객체 중요도(object importance)를 학습하는 새로운 두 단계 최적화(bilevel optimization) 문제 모델을 제시합니다. 상위 단계에서는 신경망 점수 생성기(neural scorer)를 최적화하고 하위 단계에서는 점수에 의해 가지치기된 검색 공간에서 기호적 계획(symbolic planning) 문제를 해결합니다. 이러한 접근은 과거 방식에서 발생한 학습-테스트 불일치(train-test mismatch) 문제를 해결하고 플래너가 신뢰할 수 있는 피드백을 제공하도록 돕습니다.

- **Technical Details**: 제안된 프레임워크는 3R(Repair, Restart, Rollback) 전략을 포함하여 초기 단계의 불안정한 학습 과정에서도 하위 계획에서 신뢰할 수 있는 피드백을 제공합니다. 이를 통해 플래너는 불확실한 가지치기된 검색 공간에서 실질적인 계획을 찾을 수 있으며, 상위 학습의 효율성을 높일 수 있습니다. 이 기법은 PDDL(Planning Domain Definition Language)의 정의를 기반으로 하여 복잡한 논리적 제약(logical constraints)을 처리하는 데 효과적입니다.

- **Performance Highlights**: 실험 결과, MazeNamo, SokoMindPlus, LogisticsPlus의 세 가지 벤치마크에서 SOTA 성능을 달성하였으며, 실패율(failure rate)은 80.04% 감소하고 계획 시간(planning time)은 57.14% 줄어드는 결과를 보였습니다. 또한, 사족 기반 이동 조작기에 대한 시뮬레이션 및 실세계 검증을 통해 제안된 방법의 효율성 및 배포 가능성을 입증했습니다.



### EgoPressDiff: Multimodal Video Diffusion for Egocentric UV-Domain Hand-Pressure Estimation (https://arxiv.org/abs/2606.06872)
Comments:
          Accepted to IEEE ICASSP 2026

- **What's New**: 이번 연구에서는 egocentric 카메라로부터 손 표면 접촉 압력을 추정하는 새로운 방법인 EgoPressDiff를 소개합니다. 기존 방식들이 압력 신호를 비효율적으로 처리했던 것에 반해, EgoPressDiff는 비디오 확산 프레임워크를 사용하여 압력 맵을 생성합니다. 이 방식은 PoseNet과 Vertex Encoder를 포함한 다중 모달 조정 전략을 통해 손의 자세와 3D 메쉬 정점에서 효율적으로 특징을 추출합니다.

- **Technical Details**: EgoPressDiff의 아키텍처는 PoseNet, Vertex Encoder 및 조정된 공간 레이어를 포함합니다. PoseNet은 손의 자세 특징을 추출하고, Vertex Encoder는 MANO 손 모델의 3D 정점 좌표를 이용해 압력과의 관계를 학습합니다. 마지막으로, 조정된 공간 레이어는 서로 다른 도메인에서 온 다중 모달 조정 신호를 효과적으로 통합할 수 있도록 설계되었습니다.

- **Performance Highlights**: EgoPressDiff는 EgoPressure ego-view 설정에서 기존 방법들보다 34% 이상 개선된 Volumetric IoU를 달성하며, MAE를 감소시키고 높은 시간 정확성을 유지합니다. 이는 연속적이고 시간적으로 인식 가능한 모델링의 우수성을 입증합니다. 이러한 성과는 기존의 정적이고 픽셀 단위 분류에서 동적 비디오 생성 방식으로의 혁신적인 전환에 기반합니다.



### Modeling Nonlinear Feature Interactions with Product-Unit Residual Networks (https://arxiv.org/abs/2606.06861)
Comments:
          Accepted at ICCS 2026

- **What's New**: 본 논문에서는 픽토리 모델 (product-unit residual networks, PURe)을 제안하여 비선형 특징 상호작용을 명시적으로 모델링하고 이를 기반으로 안정적인 최적화를 유지하는 방법을 탐구합니다. PURe는 곱셈 제품 단위와 잔여 연결을 통합하여 ,MLP(멀티 레이어 퍼셉트론)와 비교하여 더 나은 예측 정확성, 러너블 노이즈에 대한 강인성, 적은 데이터 환경에서의 성능을 평가하며 더욱 시너지 효과를 내는 개선된 구조적 분석을 제공합니다.

- **Technical Details**: PURe는 잔여 블록을 통해 비선형 상호작용을 모델링하고 최적화를 안정시키는 미세 조정된 신경망 구조를 사용합니다. 이 구조는 입력을 세 단계 변환을 통해 처리하며, 중간 벡터를 긍정적으로 제약하여 로그 선형을 통해 곱셈적 관계를 명시적으로 모델링합니다. 이는 기존의 MLP 대비 더욱 명확한 구조적 표현을 가능케 하여 해석 가능성을 높입니다.

- **Performance Highlights**: PURe는 세 가지 회귀 테스트에서 평가되었으며, 이들은 비선형 특징 상호작용을 포괄하고 점진적으로 복잡성이 증가하는 데이터를 포함합니다. PURe는 일반적인 MLP 혹은 복잡 값 변종과 비교하여 경쟁력 있는 성능과 더 나은 일반화 및 해석 가능성을 기록하여 데이터가 제한된 상황에서도 뛰어난 샘플 효율성을 보여줍니다.



### MotionEnhancer: Leveraging Video Diffusion for Motion-Enhanced Vision-Language Models (https://arxiv.org/abs/2606.06853)
Comments:
          Accepted by CVPR 2026

- **What's New**: 본 논문에서는 비디오 이해를 위한 비전-언어 모델(Vision-Language Models, VLMs)의 모션 이해 능력을 향상시키기 위해 새로운 접근법인 MotionEnhancer를 소개합니다. MotionEnhancer는 강력한 비디오 확산 모델(Video Diffusion Model, VDM)에서 추출한 모션 프라이어(motion priors)를 보조 감독(auxiliary supervision)으로 활용하여 주의 정렬(attention alignment)을 통해 VLM의 모션 이해 능력을 높입니다. 이 접근법은 복잡한 모듈 설계 없이도 적용 가능하다는 점에서 주목할 만합니다.

- **Technical Details**: MotionEnhancer는 Motion-sensitive Head Selection (MHS)와 Motion-salient Text Token Identification (MTTI)이라는 두 개의 파라미터가 없는 모듈로 구성되어 있습니다. MHS는 모션에 관련된 주의 지도를 추출하기 위해 시간적 주의 지도를 평가하며, MTTI는 프레임 간 평균 값을 계산하여 원활하거나 급격한 모션에 반응하는 텍스트 토큰을 식별합니다. 이 두 모듈은 별도의 학습 파라미터 없이 VDM에서 직접 모션 관련 주의 지도를 추출하고 최적화할 수 있도록 설계되었습니다.

- **Performance Highlights**: 광범위한 실험을 통해 MotionEnhancer는 두 개의 모션 수준 비디오 이해 벤치마크에서 최첨단 VLM들에 비해 지속적인 성과 향상을 달성할 수 있음을 입증했습니다. MotionEnhancer는 비디오 QA 쌍에서의 모션 중심 주의 신호를 효율적으로 추출할 수 있도록 하여 VLM의 모션 인식 및 추론 능력을 크게 개선합니다. 연구 결과는 주의 정렬 전략이 비디오 작업으로 성공적으로 확장될 수 있음을 보여주며, 이는 적은 수정으로도 가능하다는 점에서 의미가 큽니다.



### Characterize Then Distill: Mechanistic Reasoning in Large Output Spaces (https://arxiv.org/abs/2606.06840)
- **What's New**: 본 논문은 현대의 추론 모델들이 수백만 개의 후보 라벨 중에서 소수의 관련 라벨을 선택하는 데 있어 놀라운 제로샷 성능을 발휘하는 방식을 분석합니다. 저자들은 추론 과정을 두 단계로 구분하여 설명하는데, 첫 번째 단계는 후보군을 넓게 선별하는 'shortlisting'이며, 두 번째 단계는 결과 집합에 대한 세밀한 추론입니다. 이러한 단계를 서로 분리할 수 있으며 상호 보완적이라는 증거를 다양한 데이터셋에서 제공합니다.

- **Technical Details**: 이 논문에서는 현대의 LLM(대규모 언어 모델)의 추론 구조를 두 단계로 나누어 분석합니다. 첫 번째 단계인 'Coarse Semantic Filtering'에서는 입력 데이터의 강한 의미 신호와 관련된 주요 토큰을 식별합니다. 두 번째 단계인 'fine-grained reasoning'에서는 이러한 주요 신호에 따라 세부적인 추론 과정을 진행하며, 이를 통해 후보 라벨의 최종 선택을 지원합니다.

- **Performance Highlights**: 연구 결과, 기존의 CoT(distillation) 방법보다 저자들이 제안한 기계적 증류(mechanistic distillation) 방법이 더 나은 성능을 보입니다. 이 방법은 각 단계별 계산을 직접 감독함으로써 학생 모델의 성능을 향상시킵니다. 예를 들어, 대규모 다중 라벨 과제를 수행할 때 두 단계의 과정이 명확하게 드러나며, 이러한 접근 방식은 성능 평가에서 유리한 결과를 이끌어냈습니다.



### LLM Agent-Assisted Reverse Engineering with Quantitative Readability Metrics (https://arxiv.org/abs/2606.06838)
- **What's New**: 본 논문은 자동 디컴파일러가 생성하는 읽기 어려운 C 코드를 개선하는 방법을 소개합니다. 기존의 메트릭 없이 LLM 대리인(agents)이 작업을 수행하면서 발생했던 문제를 해결하기 위해 세 가지 단계의 연구 발전을 제시합니다. 특히, Quantitative Readability Score (QRS) 프레임워크를 개발하여 가독성을 정량적으로 측정하고 이를 기반으로 LLM이 개선된 결과물을 제공할 수 있도록 돕습니다.

- **Technical Details**: 연구는 세 가지 단계로 진행됩니다. 첫 번째 단계에서는 Ghidra MCP를 통해 LLM이 자율적으로 코드 분석을 수행하게 했으나, 결과물이 불완전하고 일관성이 결여되는 문제를 경험했습니다. 두 번째 단계에서는 구조적 유사성을 기준으로 기능적 동등성을 달성했지만, 가독성 저하라는 부작용이 발생했습니다. 최종적으로 QRS에 의해 안내되는 세 번째 단계에서 코드의 가독성을 목표로 한 개선이 이루어졌습니다.

- **Performance Highlights**: QRS 프레임워크의 도입으로 LLM 대리인이 가독성을 목표로 하는 작업 시, 정확성을 희생하지 않으면서도 효과적으로 개선 작업을 수행할 수 있게 되었습니다. 단기 반복적인 개선 요청을 통해 에이전트는 지속적으로 성과를 점검하고 자연스러운 재시작점을 생성하여 최적화합니다. 이 방법론은 비트코드 유사성 평가에서 향상된 점수를 기록하며, Ghidra 및 radare2 등의 도구를 활용하여 디컴파일된 이진 파일 간의 분석을 위한 효율적인 접근법을 제시합니다.



### Think Like a Pilot: Fine-Grained Long-Horizon UAV Navigation (https://arxiv.org/abs/2606.06836)
- **What's New**: 이번 논문에서는 UAV(무인항공기) 에이전트가 긴 지평선의 의미 기반 지시를 실행하고 부드럽고 물리적으로 실행 가능한 연속 비행 명령을 생성할 수 있도록 하는 새로운 벤치마크인 FLIGHT를 소개합니다. 기존의 Vision-Language Navigation (VLN) 벤치마크는 일반적으로 이산적 또는 조잡한 행동을 사용하며, UAV Vision-Language-Action (VLA) 작업은 짧고 원자적인 조작에 집중하고 있었습니다. FLIGHT는 세밀한 VLN과 긴 지평선 흐름으로 나누어진 두 개의 데이터셋에서 다단계 지침과 밀집된 6-DoF 궤적 주석을 결합하고 있습니다.

- **Technical Details**: FLIGHT 시스템은 UAV 에이전트가 과제 실행 상태와 미션 계획에 대한 실시간 비행 중 추론을 가능하게 하며, 동시에 고주파의 정확한 제어를 수용하기 위해 FLIGHT VLA라는 비동기 아키텍처를 제안합니다. 이 구조는 과제 상태 추론을 위한 저주파 Streaming Pilot Vision-Language Model (VLM)와 연속 제어를 위한 고주파 확산 행동 모델을 분리하여 멀티 태스크를 수행합니다. 또한 현재 비행 상태를 요약하고 다음 하위 목표를 예상하는 명확한 Pilot Reasoning 텍스트로 감독됩니다.

- **Performance Highlights**: 실시간 평가에서 FLIGHT VLA는 FLIGHT 벤치마크의 대표적인 VLN 및 VLA 기준을 지속적으로 초과 달성하며, 다단계 완료, 하위 목표 준수 및 터미널 제어에서 우수한 결과를 보여주었습니다. 학습된 Streaming Pilot Reasoning VLM은 UAV 비디오 추론을 더욱 향상시켜, 설계의 효과성을 입증합니다.



### Hearing the Unspoken: Language Model Priors for Acoustic Adversarial Attacks (https://arxiv.org/abs/2606.06833)
- **What's New**: 이 논문은 실시간 음성 인식(ASR) 시스템에서의 공격 성능을 향상시키는 새로운 접근법인 Semantic Gambit 공격을 제안합니다. 이 공격은 대형 언어 모델(LLM)의 예측 맥락을 활용하여 공격자가 다양한 정보를 실시간으로 획득할 수 있도록 합니다. 본 연구에서는 이 새로운 방법론이 현재 최첨단 기술보다 세 배 높은 차원의 단어 오류율(Word Error Rate)을 초래할 수 있음을 보여줍니다.

- **Technical Details**: 우리는 ASR 시스템의 공격을 공격자의 지식 수준에 따라 분류하는 새로운 프레임워크를 도입했습니다. 이 프레임워크는 공격자가 접근할 수 있는 정보의 양을 기준으로 하여 공격의 효율성을 평가합니다. Semantic Gambit 방법은 각기 다른 정보의 채널을 활용하여 공격자의 성능을 극대화하며, 이는 LLM의 예측 기능을 이용해 실시간으로 음성의 변화를 촉진합니다.

- **Performance Highlights**: 실험에 따르면, Semantic Gambit 공격은 Wav2Vec 2.0 모델에서 단어 오류율을 2%에서 35% 이상으로 증가시켰습니다. 또한, 이 공격은 평균 0.6초 만에 3초의 변형을 생성할 수 있어, 실시간 공격을 가능하게 합니다. 이러한 성과는 ASR 모델에 대한 기존의 위험 인식이 상당히 과소평가되었다는 점을 시사하며, 방어 AI 연구는 증강된 정보의 가능성을 고려해야 할 필요성을 강조합니다.



### Progress-SQL: Improving Reinforcement Learning for Text-to-SQL via Progressive Rewards (https://arxiv.org/abs/2606.06825)
- **What's New**: 이번 논문에서는 Progress-SQL이라는 새로운 다단계 강화학습(RL) 프레임워크를 제안합니다. 이 프레임워크는 Text-to-SQL 생성에서 점진적인 보상을 제공하여 SQL 개선을 위한 더 나은 지침을 생성합니다. 구문 레벨의 구조적 프로파일을 추상화하고 진단 피드백을 생성하는 Oracle-guided Diagnostic Tree(ODT)를 도입하여, SQL 예측을 정교하게 다듬을 수 있습니다.

- **Technical Details**: Progress-SQL은 초기 SQL에서 최종 SQL로의 개선을 측정하는 점진적 보상을 정의합니다. 또한, 조기 정확성(progression latency)과 실행 상태(execution status) 보상을 포함하여 SQL 예측이 유효하지 않은 경우 회복을 장려합니다. ODT 기반의 구조 정렬(structural alignment)과 어휘 정렬(lexical alignment)을 결합하여 더 밀집된 보상 신호를 생성합니다.

- **Performance Highlights**: BIRD 및 Spider 데이터셋에서의 실험을 통해 Progress-SQL은 실행 정확도를 평균 8.5% 향상시켰습니다. 기존 방법들과 비교해도 성능이 경쟁력 있거나 더 우수한 결과를 보여 투명한 개선을 입증했습니다. 다양한 기본 모델에 대한 광범위한 실험도 진행되어, 모든 환경에서 꾸준히 개선된 성능을 확인할 수 있었습니다.



### PandaAI: A Practical Agent CQ2 for Neuro-symbolic Data Analysis And Integrated Decision-Making in Quantitative Financ (https://arxiv.org/abs/2606.06823)
- **What's New**: 이 논문에서는 금융 분야에서의 깊은 학습(deep learning)의 한계를 극복하기 위해, PandaAI라는 neuro-symbolic LLM 에이전트를 제안합니다. PandaAI는 시장 상황 모형(market regime modeling)과 제약된 알파 생성(constrained alpha generation)을 결합하여 일반 LLM의 추론과 금융의 엄격함을 연결합니다. 이 시스템은 전통적인 모델들과 달리, 금융 시장에서의 리스크 인식을 명시적으로 내포하고 있으며, 효율적인 의사결정을 지원합니다.

- **Technical Details**: PandaAI는 고전적인 MCTS(모니테링 몬테 카를로 트리 탐색) 알고리즘을 기반으로 하여, 금융의 하드 제약 조건(financial hard constraints)을 통합하여 알파 생성(alpha generation)을 안내합니다. 모델은 비정상적인 시장 특성을 반영하기 위해 잠재 상태(latent state)를 도입하고, 고차원 시장 동역학을 연속적인 잠재 상태로 압축합니다. 또한, LLM과 효율적인 재무 처리를 결합하여 모델의 실시간 적응력을 높입니다.

- **Performance Highlights**: 실험 결과, PandaAI는 CSI 300 주식 데이터에서 $18.2\%$ 높은 Rank IC와 $25.7\%$ 낮은 최대 손실(maximum drawdown)을 기록하며 기존 최첨단 시계열 모델들보다 뛰어난 성능을 보였습니다. 이러한 결과는 LLM의 제약된 생성 방식과 이중 채널 적응 메커니즘이 높은 위험이 따르는 연속적 의사결정에서 효과적으로 적용될 수 있음을 보여줍니다.



### SCALE: Scalable Cross-Attention Learning with Extrapolation for Agentic Workflow Scheduling (https://arxiv.org/abs/2606.06820)
Comments:
          Submitted to Computer Networks

- **What's New**: 본 논문에서는 Agentic Large Language Model (LLM) 시스템이 복잡한 작업을 Directed Acyclic Graphs (DAGs)로 분해하고, 이를 이종 클러스터에 맞춰 스케줄링하는 방법을 제안합니다. 기존의 deep reinforcement learning (DRL) 스케줄러는 고정된 클러스터 크기와 연관되어 있으며 서버 수가 변경될 때마다 재학습이 필요합니다. SCALE(Scalable Cross-Attention Learning with Extrapolation)는 이러한 문제를 해결하기 위한 새로운 DRL 스케줄러로, 클러스터 크기와 관계없이 일반화가 가능합니다.

- **Technical Details**: SCALE은 크로스-어텐션 포인터 네트워크를 활용하여, 작업의 특징이 서버의 특징에 쿼리되는 형태로 구성됩니다. 이 구조는 서버 수에 무관하게 동작할 수 있도록 설계되었습니다. 그러나 서버 수가 증가함에 따라 주의(feature) 변량이 변화하는 문제를 해결하기 위해 Structured Representation Regularization (SRR)을 도입하였습니다. SRR은 특징 통계가 안정적으로 유지되도록 도와주며, 이를 통해 SCALE 모델은 N=16에서 학습한 후 N=32 및 48에서 추가적인 튜닝 없이도 동작할 수 있습니다.

- **Performance Highlights**: SCALE은 이종 시뮬레이션 클러스터에서 N=16으로 학습하고 N=32 및 N=48에서 테스트를 진행하였습니다. 실험 결과, SRR 없이 같은 아키텍처에서 N=48일 때 응답 시간이 8.9% 향상됨을 확인했습니다. 이는 명시적인 정규화가 스케일-일반화 격차를 해소하는 데 필수적임을 증명하는 결과입니다.



### Breaking the Lock-in: Diversifying Text-to-Image Generation via Representation Modulation (https://arxiv.org/abs/2606.06813)
Comments:
          Accepted to ICML 2026. Code is available at: this https URL

- **What's New**: 이 논문에서는 최신 텍스트-이미지(T2I) 생성 모델들이 고급 Transformer 아키텍처 및 flow-based 목표를 기반으로 하여 높은 텍스트-이미지 정렬(text-image alignment)과 뛰어난 시각적 품질을 제공하지만, 고정된 프롬프트(prompt) 아래에서 유사한 샘플을 생성하는 문제를 다룹니다. 저자들은 zéro-frequency spatial average (DC) 성분이 초기 생성 단계에서 빠르게 수렴하여 다양성을 제한하는 원인을 분석하고, 이를 개선하는 새로운 방법인 DAVE(DC Attenuation for diVersity Enhancement)를 제안합니다.

- **Technical Details**: DAVE는 미세한 내부 연산을 통해 초기 단계에서 DC 성분을 선택적으로 약화시키는 기법입니다. 이 방법은 모델의 재훈련이나 샘플링 과정의 변경 없이, 계산 비용이나 메모리 전이의 부담 없이 신뢰성을 유지하면서도 고품질 이미지를 생성할 수 있도록 돕습니다. DAVE의 도입으로 인해 프롬프트에 일관되게 다양성을 증가시킬 수 있으며, 최근 주요 방법들과 비교할 때 경쟁력 있는 성능을 보입니다.

- **Performance Highlights**: DAVE는 기존의 방법들보다 훨씬 적은 비용으로도 경쟁력 있는 이미지를 생성할 수 있으며, 다양한 실험을 통해 그 효과를 입증하였습니다. 이러한 성능은 대규모 Transformer 기반 모델의 다양성을 증진하는 핵심 기여로, 생성 모델의 사용성과 확장성을 개선하는 데 중요한 역할을 할 수 있습니다. 이 연구는 T2I 모델의 다양성 제어를 위한 새로운 관점을 제공하며, 향후 연구에 대한 방향성을 제시합니다.



### Lane Change Trajectory Planning for Personalized Driving Comfort and Mobility Efficiency (https://arxiv.org/abs/2606.06805)
Comments:
          Accepted by the IEEE Intelligent Vehicles Symposium (IEEE IV 2026), Detroit, MI, United States, June 22_25, 2026

- **What's New**: 이 연구는 차량의 차선 변경 시 걸리는 시간과 효율성을 동시에 고려한 신경망 기반의 플래너(neural network-driven planner)를 제안합니다. 특히, 세 번째 차수의 다항식 경로 생성기(third-order polynomial trajectory generator)와 최적의 경로 매개변수를 추정하는 학습 모듈(learning module)을 통합하여 다양한 주행 조건에서 최적의 경로를 생성합니다.

- **Technical Details**: 플래너는 공유된 백본(shared backbone) 구조를 사용하며, 두 개의 헤드를 가지고 있습니다. 하나의 헤드는 모든 주행 조건에서의 운영 보장을 제공하고, 다른 헤드는 운전자의 편안함(comfort) 또는 주행 효율성(mobility efficiency)을 잡아냅니다. 또한, 통계적 게이트(statistical gate)를 기반으로 하는 헤드 게이티드 스위칭 메커니즘(head-gated switching mechanism)을 통해 다양한 주행 조건에 맞게 적합한 헤드를 선택할 수 있습니다.

- **Performance Highlights**: 대표적인 사례와 몬테 카를로 시뮬레이션을 통해 이 플래너가 차선 변경 시 개인화된 편안함과 주행 효율성을 달성함을 보여주었습니다. 반면, 기본 방법은 개인화된 데이터가 부족하거나 접근할 수 없는 주행 조건에서도 가능한 경로를 보장합니다.



### Exploring Reinforcement Learning for Fluid Transitions Between Clinical Mental Healthcare and Everyday Wellness Suppor (https://arxiv.org/abs/2606.06800)
- **What's New**: 이 논문은 정신 건강 관리와 웰니스(Wellness) 지원을 통합하는 디지털 헬스 시스템을 위한 강화 학습(Reinforcement Learning, RL) 접근 방식의 가능성을 탐구합니다. 연구팀은 지속적인 저널링(Sustained Journaling)을 최적화하기 위해 임상 및 웰니스 프롬프트( Journaling prompts)를 동적으로 선택하는 컨텍스트 밴딧(Contextual Bandit)을 개발하고, 이를 일반인 집단(N=38)에 대한 4주 간의 탐색 연구에 배포했습니다. 이 연구의 결과는 정신 건강 개입의 전환 시기에 대한 새로운 질문을 제기합니다.

- **Technical Details**: 강화 학습의 특성 덕분에, 시스템은 사용자와 각 개입의 변화에 따라 개입 선택을 동적으로 조정할 수 있습니다. 이를 통해 임상 및 웰니스 지원 간의 미세한 전환이 가능해지며, 예를 들어, 사용자가 안정된 기간 동안 저널링을 통해 관리할 수 있는 방법을 제공합니다. 연구팀은 247개의 저널링 프롬프트를 임상과 웰니스 문헌에서 수집한 후, 에피소드-탐색 전략으로 훈련된 간단한 모델을 사용하여 최적의 프롬프트를 선택하게 하였습니다.

- **Performance Highlights**: 연구에서 강화 학습 최적화된 개입 시퀀스의 많은 이점이 개입이 종료된 후에만 나타나는 경향이 있음을 발견했습니다. 이는 사용자들이 강화 학습으로 생성된 개입에 더 깊이 관여한 경우, 시간이 지남에 따라 그들의 참여도가 증가하는 반면, 고정된 개입에만 관여한 경우에는 탈락률이 증가한다는 것을 나타냅니다. 연구 결과는 이러한 새로운 시스템이 개입의 강도를 조절하여 탈진과 치료 효과 극대화 사이의 균형을 맞추는 방법에 대한 추가 연구의 필요성을 강조합니다.



### What Your Posts Reveal: A Benchmark and Agentic Framework for User-Level Privacy Leakage on Social Media (https://arxiv.org/abs/2606.06784)
- **What's New**: 이번 연구에서는 공개 소셜 미디어에서 개인 정보 유출을 평가하기 위한 새로운 벤치마크인 SopriBench와 개인정보 노출 점수(Privacy Exposure Score, PES)를 제안합니다. 기존 연구들은 단일 게시물에서의 자가 노출(self-disclosure)이나 PII 인식에 집중했으나, 이 연구는 여러 게시물에 걸쳐 누적되는 개인 정보 노출 문제를 조명합니다. SopriBench는 사용자 수준에서의 멀티모달(multi-modal) 프라이버시 누출을 측정하기 위한 합성 벤치마크로서, 50명의 사용자 프로필과 1,569장의 이미지를 포함하고 있습니다.

- **Technical Details**: SopriBench는 리드노트(Rednote)와 인스타그램(Instagram)의 개인 판별되지 않은 사용자 계정으로부터 추출된 패턴을 기반으로 구축되었습니다. 이 벤치마크는 게시물의 진실 값, 맥락적 민감성, 세분성(granularity), 유출 유형, 추론 난이도 등을 기록합니다. 또한, PES는 속성(granularity)과 민감성(sensitivity)을 결합한 점수로, 노출의 정교함을 평가하는 데 도움을 줍니다.

- **Performance Highlights**: 연구에서는 Argus라는 교육이 필요 없는 인과적 추론 프레임워크를 도입하였으며, 이는 누적 증거를 통해 가설을 형성하고, 증거를 검증하며, 사용자 수준 개인정보를 구성합니다. Argus는 기존의 최강 기준선보다 25% 향상된 PES(0.55)를 달성하였으며, 이는 주로 게시물 간 유출을 통한 개선 덕분입니다. 이러한 성과는 가설 추적과 증거 검증을 통한 조합적 고유 노출 품질 향상과 관련이 있습니다.



### Mind the Gap: Bridging Behavioral Silos with LLMs in Multi-Vertical Recommendations (https://arxiv.org/abs/2606.06779)
- **What's New**: 이 논문은 DoorDash와 같은 다중 수직(e-commerce) 플랫폼에서 새로운 제품 수직이 개인화 혁신의 기회를 제공한다는 점에 주목합니다. 특히 데이터가 적은 카테고리에서 사용자에게 추천 품질을 향상시키기 위한 새로운 프레임워크를 제안하며, 이는 데이터가 풍부한 수직(예: 음식점)으로부터 지식을 전이하는 방법론에 기반을 두고 있습니다.

- **Technical Details**: 핵심 기여는 사용자 행동을 다양한 수직에서 풍부하고 숨겨진 신호의 원천으로 간주하고, 이를 대형 언어 모델(LLMs)을 통해 구조화된 표현으로 변환하여 활용하는 것입니다. 연구에서는 계층적 검색-증강 생성(RAG) 프레임워크를 활용하여 사용자 선호도를 다단계 제품 분류 체계에서 추론하는 새로운 방법론을 다룹니다. 이를 통해 '콜드 스타트(cold start)' 문제를 해결하고 사용자 표현을 풍부화하며, 더 나은 추천 시스템을 구축할 수 있는 실질적인 경로를 제시합니다.

- **Performance Highlights**: 상세한 오프라인 및 온라인 평가를 통해, 이 방법론이 신흥 비즈니스 수직에서 개인화 및 참여도를 크게 향상시킨다는 것을 증명했습니다. 특히 LLM에서 생성한 특성이 포함된 새로운 MTL 랭킹 모델이 기존 모델보다 AUC-ROC 및 MRR과 같은 평가지표에서 우수한 성능을 보였습니다. 이를 통해 추천 시스템의 효용성과 효율성을 동시에 증대시켰습니다.



### Generalization in Deep Neural Networks: Minimax Rates for Gradient Methods (https://arxiv.org/abs/2606.06772)
Comments:
          37 pages

- **What's New**: 이번 연구에서는 gradient 기반 방법을 통해 훈련된 심층 신경망(DNN)의 일반화 성능을 분석하여 기존의 갭을 메우는 데 중요한 진전을 이루었습니다. 저자는 gradient descent(GD) 및 stochastic gradient descent(SGD)와 같은 방법을 사용하는 DNN과 이에 상응하는 kernel 방법 간의 새로운 연결을 확립했습니다. 이로 인해 DNN은 kernel 방법들이 가진 유리한 학습 역학을 온전히 상속받을 수 있음을 보여주었습니다.

- **Technical Details**: 저자들은 우선적으로 DNN의 학습 역학과 무한 폭 NTK와 관련된 kernel 방법 간의 상관관계를 바탕으로 최소-최적 위험 비율을 유도했습니다. 결과적으로 기존의 kernel 방식과 동일한 가정 하에 GD와 SGD가 최적의 성능을 달성할 수 있다는 것을 입증했습니다. 또한 DNN의 훈련 시 input space 전반에 걸쳐 균일하게 집중하는 NTK에 대한 정교한 분석을 수행했습니다.

- **Performance Highlights**: 연구 결과는 DNN이 충분한 폭을 가질 경우 GD 또는 SGD를 통해 kernel 기반 방법과 유사한 일반화 성능을 달성할 수 있음을 보여줍니다. 특히, 네트워크 폭이 샘플 크기와 관련하여 다항식으로 증가하는 경우, GD와 SGD는 최소-최적의 위험 비율을 개선하여 𝒪(n⁻²β²β+γ)와 같은 성능을 제공할 수 있습니다. 이러한 결과는 더 어려운 학습 문제로의 확장 가능성을 시사합니다.



### Optimal Rates for Generalization of Gradient Descent Methods with Deep Neural Networks (https://arxiv.org/abs/2606.06764)
Comments:
          39 pages, 1 table

- **What's New**: 이 논문은 기울기 하강법(Gradient Descent)과 확률적 기울기 하강법(Stochastic Gradient Descent)을 사용하는 심층 ReLU 네트워크의 일반화 분석을 제시하여, 기존 연구에서 부족했던 심층 신경망 이론의 갭을 메우고 있다. 특히, 네트워크의 너비가 깊이와 훈련 샘플 크기에 대해 다항적으로 스케일할 때 GD와 SGD의 최적 초과 집단 위험(minimax-optimal rates of excess population risk)을 도출했다. 이 결과는 충분한 너비를 가진 심층 ReLU 네트워크의 경우, 기울기 하강법이 커널 방법과 비슷한 최적 일반화 비율을 이룰 수 있음을 보여준다.

- **Technical Details**: 이 논문에서는 심층 ReLU 네트워크를 위한 기울기 하강법과 확률적 기울기 하강법의 일반화 특성을 체계적으로 분석했다. LL 계층 ReLU 네트워크의 상당히 큰 너비 mm에 대해, GD와 SGD는 유사한 가정 하에 커널 설정에서의 고전적인 결과를 복제할 수 있음을 입증했다. 특히, 이 결과는 NTK의 그램 행렬에 대한 일반적인 가정을 부과하지 않고도 달성되며, 심층 네트워크의 너비 요구 사항을 기존 연구보다 개선하였다.

- **Performance Highlights**: 심층 ReLU 네트워크에서 GD와 SGD의 초과 위험(minimax-optimal excess risk rates)을 기존 연구에서 제시된 경량 네트워크의 결과를 기반으로 분석하였다. 이 논문의 주요 기여는 GD 및 SGD가 심층 아키텍처로 확장된 연구를 진행하면서, 기존의 분류 문제 연구의 이론적 한계를 넘어 심층 회귀 문제에 대한 성능을 강조하는 것이다. 최종적으로, 저자들은 심층 ReLU 네트워크의 좋은 일반화 성능이 커널 기반 방법과 동등하다는 것을 입증하며, 이는 기계 학습 이론에 중요한 의미를 지닌다.



### AxisGuide: Grounding Robot Action Coordinate System in RGB Observations for Robust Visuomotor Manipulation (https://arxiv.org/abs/2606.06761)
Comments:
          Accepted to Robotics: Science and Systems (RSS) 2026

- **What's New**: 이번 연구에서는 비주얼 모터 조작 정책이 이해한 작업의 의미와 저수준 행동의 이해 사이의 간극을 다루고 있습니다. 새로운 방법론인 AxisGuide를 제안하여 로봇의 기본 좌표계 행동을 이미지 공간에서 해석할 수 있도록 시각적인 단서를 제공합니다. 이를 통해 다양한 시나리오에서 강력한 행동 실행 능력을 보여주고자 합니다. AxisGuide는 RGB 영상에 추가적인 채널을 통해 +x, +y, +z 동작의 의미를 명확히 시각화합니다.

- **Technical Details**: AxisGuide는 카메라 파라미터와 엔드 이펙터의 포즈를 활용해 각 카메라 뷰에서 로봇 베이스 프레임의 축을 그립니다. 이를 통해 작동 좌표계를 RGB 관찰과 연결하여, 정책이 새롭게 나타나는 물체 위치에 맞춰 더욱 신뢰성 있게 엔드 이펙터의 비행을 조정할 수 있도록 돕습니다. 주요 기술은 동작 공간의 의미를 픽셀 단위로 명시화하여, 각 RGB 이미지에 대한 행동 이해도를 향상시키는 것입니다.

- **Performance Highlights**: AxisGuide는 실제 환경과 시뮬레이션 환경 모두에서 성능 향상을 보여주었습니다. 성과는 특히 물체가 보지 못한 위치에 놓였을 때 성공률이 20%p까지 증가하는 것으로 측정되었습니다. 또한, 다각 뷰 및 단일 뷰 구성 모두에서 일관된 성공률 향상을 보이므로, 명시적인 행동 좌표 기준이 신뢰할 수 있는 실행을 촉진하는 데 기여했음을 나타냅니다.



### Evidence Graph Consistency in Retrieval-Augmented Generation: A Model-Dependent Analysis of Hallucination Detection (https://arxiv.org/abs/2606.06748)
Comments:
          Accepted at the International Conference on Advanced Machine Learning and Data Science; to appear in the IEEE Xplore proceedings

- **What's New**: 이번 논문에서는 Evidence Graph Consistency (EGC)라는 새로운 프레임워크를 제안하여, RAG(Retrieval-Augmented Generation)에서 발생하는 허위정보(hallucination)를 감지하는 방식을 개선합니다. 기존의 방법들이 주로 생성된 답변과 조회된 내용 간의 단순 유사성을 평가하는 데 집중한 반면, EGC는 각 응답에 대한 지역 증거 그래프를 구성하고 구조적 일관성 지표를 계산하여 이러한 허위정보의 신호를 파악합니다. EGC의 적용을 통해 Llama-2 모델에는 기대하는 진단 방향이 나타났지만, GPT-4와 GPT-3.5 모델에서는 상반된 결과가 나타나는 것으로 확인되었습니다.

- **Technical Details**: Evidence Graph Consistency(EGC)는 질문, 조회된 구절, 그리고 생성된 답변 간의 관계를 그래프로 구축하여 구조적 일관성을 특징으로 추출합니다. 이 그래프는 세 종류의 노드를 포함하며, 이를 통해 Euclidean 공간 내에서 cosine similarity를 사용하여 노드 간의 관계를 정의합니다. 또한 다섯 가지의 구조적 일관성 지표(coverage, support density, cross-evidence agreement, connectivity, isolation penalty)를 계산하여, 각각의 노드 및 엣지 간의 유기적 연결성을 평가합니다.

- **Performance Highlights**: RAGTruth 데이터셋에 대한 평가를 통해 EGC는 5,767개의 응답에서 모델 간 일관된 분할을 보여주었습니다. 특히 구조적 일관성이 허위정보 감지에 있어서 모든 모델에서 동일하게 유효하지 않으며, 특정 모델군 간의 허위정보 패턴이 질적으로 다르다는 점이 강조되었습니다. 이러한 결과는 향후 허위정보 감지 시스템의 설계 시 모델 의존성을 고려해야 함을 시사합니다.



### HybridCodec: Fast Dual-Stream, Semantically Enhanced Neural Audio Codec (https://arxiv.org/abs/2606.06743)
Comments:
          5 pages, 5 tables, 1 figure, Accepted at Interspeech 2026

- **What's New**: 이 논문에서는 HybridCodec라는 새로운 통합 아키텍처를 제안합니다. 이 모델은 두 가지 접근법, 즉 의미적 증류(semantic distillation)와 이중 스트림(ddual-stream) 구조를 결합하여 강력한 의미-음향 분리(disentanglement)를 실현합니다. 또한, HybridCodec은 인퍼런스(inference) 동안 SSL 모델을 요구하지 않으면서도 의미적 전문성(semantic specialization)을 유지합니다.

- **Technical Details**: HybridCodec의 구조는 두 개의 스트림, 즉 의미적 스트림과 음향 스트림으로 구성되어 있으며, 각 스트림은 고유한 인코더와 디코더로 이루어져 있습니다. 공통 인코더는 24kHz의 원시 파형을 입력으로 받아 1D 합성곱(convolution) 네트워크(CNN)를 사용하여 처리합니다. 인코더의 출력은 의미적 디코더를 통해 의미 정보가 증류되고, 음향 디코더는 잔여 음향 정보(residual acoustic information)를 복원하는 데 사용됩니다.

- **Performance Highlights**: HybridCodec은 의미적 전문화 측면에서 RVQ-1 테스트 세트에서 뛰어난 성능을 보이며, 전체적인 음향 품질도 경쟁력을 유지합니다. 논문에서는 이 모델이 기존의 이중 스트림 모델에 비해 3배 빠른 인퍼런스 속도를 달성했다고 명시하고 있습니다. 또한, 다양한 환경에서의 강인성(robustness) 또한 입증되었습니다.



### Multilingual Multi-Speaker Unit Vocoders: A Systematic Analysis of Discrete Speech Representations (https://arxiv.org/abs/2606.06740)
Comments:
          5 pages, 5 tables, 1 figure, Accepted at Interspeech 2026

- **What's New**: 이 연구에서는 BigVGAN을 기반으로 한 단위 보코더를 분석하여 네 가지 인도어(벵골어, 힌디어, 타밀어, 텔루구어)에 대한 음성 생성의 품질을 높이는 방법을 논의합니다. 연구는 군집 크기와 조건화 전략 간의 상호작용을 평가하여 음성 인식의 명도와 화자 유사성을 개선하는 데 초점을 맞추고 있습니다. 또한, 언어 감독이 낮은 군집 크기에서 더욱 효과적임을 발견하여 모호한 단위의 문제를 해결하고, 다양한 언어에서 유사한 음소들이 동일한 군집 ID로 수렴하는 현상에 대해 분석합니다.

- **Technical Details**: 이 연구는 BigVGAN 아키텍처를 사용하여 생성기, 멀티 주기 분류기(MPD), CQT 기반의 시간 주파수 분류기를 포함한 음성 합성 시스템을 개발합니다. 여기서 생성기 입력은 멜 스펙트로그램 대신 디스클리트 단위를 사용하고, 화자 및 언어 임베딩을 결합하여 추가 정보를 제공합니다. 조건화 메커니즘과 군집 크기에 따른 트레이드오프를 탐색함으로써 다국어 다화자 설정에서 음성 생성 품질을 개선할 수 있는 방법을 제시합니다.

- **Performance Highlights**: 실험 결과, 군집 크기가 음성의 이해도를 제어하고 화자 유지에 필수적임을 보여주었습니다. 특히, 의사소통의 흐름을 방해하지 않는 조건화 방식이 중요하며, 이는 다양한 언어에서 공통적으로 (shared) 사용되는 단위 각각의 군집에서 선명도를 높이는 데 기여합니다. 연구에서는 단위 기반의 보코더가 멀티언어 다화자 환경에서 어떻게 성능을 발휘할 수 있는지를 정량적으로 평가하고, 음성 생성의 질을 개선하기 위한 최적의 설정 방안을 제안합니다.



### SCOUT: Semantic scene COverage via Uncertainty-guided Traversa (https://arxiv.org/abs/2606.06721)
Comments:
          2026 ICRA Workshop on Uncertainty in Open World Robotics

- **What's New**: 본 연구에서는 SCOUT라는 최신 온라인 시맨틱 탐사 프레임워크를 제안하여, 로봇이 장기간 운영되면서 공간을 이해하도록 돕습니다. 이 시스템은 확률적 장면 그래프 구성과 능동적인 탐색을 결합하여, 로봇이 새로운 장면을 자율적으로 인식하고 데이터를 지속적으로 업데이트할 수 있도록 합니다. SCOUT는 기존의 정적 데이터 접근 방식에서 벗어나, 실시간으로 장면의 의미론적 완전성을 목표로 발전합니다.

- **Technical Details**: SCOUT는 두 가지 주요 구성 요소로 구성되어 있습니다: 불확실성 안내 탐색기(UGT)와 확률적 장면 그래프 생성기(PSGG)입니다. UGT는 시맨틱 정보를 활용하여 다음 관측 지점을 선택하고, PSGG는 로봇의 RGB-D 관측 데이터를 통합하여 불확실성 인식 3D 장면 그래프를 만들어냅니다. 이 시스템은 일련의 반복적 과정을 통해 탐색과 그래프 구성을 함께 조율하여, 객체의 불확실성을 해결합니다.

- **Performance Highlights**: SCOUT는 시각 정보와 공간적 관계를 동시에 고려하여 로봇의 장면 이해도를 극대화합니다. 실험 결과, SCOUT는 95%의 기하학적 커버리지와 0.7 미만의 불확실성 측정을 만족하는 정확도를 달성하며, 이는 로봇이 효율적으로 탐색하고 복잡한 환경을 관리할 수 있도록 지원합니다. 이러한 접근 방식은 로봇이 더욱 자율적으로 실내 환경을 순찰하고 업데이트하며, 필요한 경우 최소한의 인간 개입으로 의사결정을 할 수 있도록 합니다.



### MSAIC-Net: A Multi-Scale Attention and Imbalance-Aware Contrastive Network for ECG-Based Myocardial Substrate Abnormality Detection (https://arxiv.org/abs/2606.06718)
- **What's New**: 이 논문에서는 심근 기질 이상(myocardial substrate abnormalities)의 전기심전도(ECG) 기반 검출을 위한 다중 스케일 주의 향상 합성곱 네트워크(MSAIC-Net)를 제안합니다. MSAIC-Net은 병렬 atrous 합성곱(branch)을 사용하여 여러 시간적 수용(field)에서 ECG 특징을 추출하여 모델이 지역 및 장기적인 시간적 패턴을 모두 캡처할 수 있도록 합니다. 또한, 새로운 불균형 인식 감독 대조 학습(imamalance-aware supervised contrastive learning) 전략을 도입하여 클래스 불균형 문제를 해결하고, 모델의 해석 가능성을 개선하기 위한 lead-wise permutation importance를 사용합니다.

- **Technical Details**: 제안된 MSAIC-Net은 다중 스케일 합성곱 작업을 활용하여 지역 및 전역 시간적 패턴을 포착하며, 정보가 많은 lead와 feature channel을 적응적으로 강조할 수 있도록 주의 메커니즘을 통합합니다. 또한, focal binary cross-entropy 손실(focal binary cross-entropy loss)과 focal-weighted 감독 대조 학습 전략을 채택하여 클래스 불균형으로 인해 발생하는 훈련 편향을 완화하고, intra-class compactness 및 inter-class separability를 높여 ECG 신호에서의 MSAIC-Net의 성능을 개선합니다.

- **Performance Highlights**: 제안된 방법은 두 가지 보완 데이터셋에서 평가되었으며, 특히 저데이터 UVA 코호트에서 기본 모델보다 현저한 성능 향상을 보였습니다. 실험 결과, MSAIC-Net은 심근 흉터(muscle scar) 분류에서 높은 정확도를 기록하였고, 대규모 공용 PTB-XL 데이터셋에서 심근 경색(MI) 식별에서도 뛰어난 성과를 보여주었습니다. 전반적으로, 이 프레임워크는 ECG 기반 심근 기질 이상 탐지에 효과적이고 해석 가능한 접근 방식을 제공합니다.



### ShallowBench: Benchmarking Generative Drug Design Models on Shallow-Pocket Targets (https://arxiv.org/abs/2606.06717)
- **What's New**: 이번 연구에서는 ShallowBench라는 새로운 벤치마크 데이터셋을 소개합니다. 이 데이터셋은 CrossDocked2020에서 추출한 5,780개의 얕은 포켓 타겟을 엄격하게 선별하여 구성되었습니다. 특히, KRAS와 MYC와 같은 전통적인 약물 개발이 어려운 표적들을 효과적으로 겨냥할 수 있는 유용한 자원입니다.

- **Technical Details**: ShallowBench의 구성 방법은 두 단계의 볼륨 계산 접근법에 기반합니다. 첫 번째 단계에서는 단백질 원자와 리간드의 중심을 기준으로 특정 부위의 원자들로 볼륨을 계산합니다. 두 번째 단계에서는 Alpha Shape 메쉬를 사용하여 해당 부위의 vacío와 포켓의 깊이를 측정하고, 이를 통해 최종적으로 낮은 오목성을 가진 타겟을 선정합니다.

- **Performance Highlights**: ShallowBench에서의 여러 최신 생성 모델 평가 결과, 모든 아키텍처에서 예측된 결합 친화도가 낮아지는 경향을 보였습니다. 이는 생성 리간드가 얕은 표면에서의 치명적인 약점을 드러내며, 새로운 구조 혁신이나 손실 함수의 필요성을 강조합니다.



### Does Topic Sentiment Cause Perceived Ideology? Comparing Human and LLM Annotations in Political News Articles (https://arxiv.org/abs/2606.06715)
Comments:
          Accepted to ACL SRW 2026

- **What's New**: 이 논문에서는 주제 감정(topic sentiment)이 인지된 정치 이념(perceived political ideology)에 인과적인 영향을 미치는지 살펴보고, 그 결과가 이념 레이블을 부여하는 주체에 따라 달라지는지를 분석합니다. AllSides의 기사들과 Llama-3.3-70b-versatile로부터의 감정 주석을 기반으로 하여, 전문가와 GPT-4o-mini(기본형 및 세밀 조정된 형), Llama-3.3-70B 간의 이념 레이블을 비교했습니다. 연구 결과, 인간 주석자들은 지역 사회 수준에서 유의미한 인과 효과를 보이지 않았지만, 세밀 조정된 GPT-4o-mini는 가장 높은 분류 정확도를 달성했습니다.

- **Technical Details**: 인과 추정(causal inference) 방법론인 더블 머신 러닝(Double Machine Learning, DML)을 통해 주제 감정이 LLM 주석자들에 의해 예측된 이념에 중요한 인과 효과를 미친다고 밝혔습니다. 세밀 조정된 GPT-4o-mini 모델에서 감정-이념 간의 유의미한 자연 직접 효과(natural direct effects, NDEs)가 나타났으며, 이는 다른 세 가지 주석자 모델에서는 발견되지 않았습니다. 이러한 결과는 LLM이 특정 주제 그룹에 따라 감정과 텍스트적 특징 간의 관계를 더 밀접하게 연결짓는다라는 점을 시사합니다.

- **Performance Highlights**: 실험을 통해 세밀 조정된 GPT-4o-mini 모델이 F1 점수 72.48로 가장 높은 분류 정확도를 보였으며, 인간 주석자와 비교할 때 중요한 인과 효과를 생성하는 유일한 모델로 확인되었습니다. 이 연구는 LLM이 감정 주석을 처리하는 방식이 인간의 판단과는 다르게 작동하며, 이는 인과 분석에 있어 LLM이 생성한 레이블을 정황적(silver) 레이블로 사용함에 있어 중요한 함의를 지닙니다. 결론적으로, 세밀 조정된 LLM의 감정-이념 예측은 인간의 주석과는 다르게 동작하며, 이는 향후 사회 과학 연구에서 LLM의 사용 방식을 재고하게 할 만한 요소입니다.



### Data-Efficient Autoregressive-to-Diffusion Language Models via On-Policy Distillation (https://arxiv.org/abs/2606.06712)
- **What's New**: 이번 연구에서는 자기 회귀 모델(ARLM)에서 확산 언어 모델(DLM)로의 변환을 탐구합니다. 기존 연구들은 ARLM의 인과 주의를 양방향 주의로 교체하고 이후 DLM 목표로 모델을 훈련시켰으나, 두 가지 분포 변화(distribution shift)가 발생합니다. 이 논문에서는 On-Policy Diffusion Language Model(OPDLM)을 도입하여 이러한 문제를 해결합니다.

- **Technical Details**: OPDLM은 자기-OPD(self-OPD)를 이용하여 훈련되며, 학생 모델인 양방향 주의를 가진 ARLM이 자체적으로 경로(trajectories)를 생성합니다. 교사 모델은 원래 동결된 ARLM으로, 이러한 경로에 대한 목표 로그it(target logits)을 제공하여 지식을 증류(distillation)합니다. 이 과정에서는 DLM에서 흔히 발생하는 훈련-추론 불일치(train-inference mismatch)를 제거할 수 있습니다.

- **Performance Highlights**: 경험적 결과에 따르면 OPDLM은 15배에서 7,000배 더 적은 훈련 토큰을 필요로 하면서도 다양한 작업에서 강력한 성능을 보입니다. 또한 OPDLM은 DLM의 사전 훈련(pretraining) 비용을 피하며, ARLM 후 훈련(post-training)의 형태로 DLM 변환을 가능하게 합니다.



### MMBU: A Massive Multi-modal Biomedical Understanding Benchmark to Probe the Perception Capabilities of Vision-Language Models (https://arxiv.org/abs/2606.06696)
- **What's New**: 이번 연구에서는 Massive Multimodal Biomedical Understanding (MMBU) 벤치마크를 소개합니다. MMBU는 35개의 하위 모달리티를 포함하여 현재까지 가장 큰 생물의학 비전-언어 모델(VLM) 벤치마크로, 모델 성능을 체계적으로 평가할 수 있게 합니다. 이 벤치마크는 개방형 및 폐쇄형 분류와 객체 탐지를 포함하여 다양한 생물학적 스케일, 임상 환경, 이미징 모달리티에서 평가를 제공합니다.

- **Technical Details**: MMBU는 410개의 데이터세트를 커버하며, 11개의 모달리티와 20개의 표본, 95개의 고유 관심 지역을 포함하고 있습니다. 각 데이터세트는 전문가들에 의해 수작업으로 주석이 달린 VQA(Visual Question Answering) 작업으로 변환되어 핵심 인지 작업인 분류 및 탐지의 평가를 지원합니다. 이러한 방대한 데이터는 정교한 주석을 통해 이미지 출처, 데이터세트 이름, 도메인 등 다양한 특성을 포함하고 있습니다.

- **Performance Highlights**: 각 모델과 작업 전반에 걸쳐 높은 오류율이 발견되었으며, 의료 적응으로부터 제한적인 개선이 있었습니다. MMBU를 통해 모델의 공간적으로 기반한 작업과 같은 고질적인 약점을 파악할 수 있었고, 기존 벤치마크에서의 높은 정확성이 실제 시나리오에서의 인지 능력을 과대평가할 수 있음을 보여주었습니다. 이를 통해 모델 개선을 위한 기초 데이터를 제공하고, 향후 생물의학 인지 작업을 위한 더 신뢰성 있는 모델 개발을 지원하고자 합니다.



### The Geography of Algorithmic Judgment: LLM Intermediaries, Place Identity, and Racial Steering in Housing Search (https://arxiv.org/abs/2606.06694)
Comments:
          13 pages with supplemental tables and figures, AIES '26 Submission

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)이 주거 검색에서 중재자 역할을 수행하면서 발생하는 인종 미신(steering) 현상에 대한 새로운 통찰을 제공합니다. 저자들은 미국 내 네 개 도시에서 개방형 및 폐쇄형 LLM 일곱 종을 대상으로 행동 감사를 수행하고, 사용자 특성과 선호도를 고려한 추천 메커니즘을 분석했습니다. 이 연구의 주요 발견으로는, LLM의 미신 현상이 고정된 속성으로 존재하는 것이 아니라 사용자의 정체성과 선호도에 따라 상호작용을 통해 나타나는 동적 행동임을 밝혀냈습니다.

- **Technical Details**: 이 연구는 LLM의 인종 미신을 평가하기 위해 일곱 개 모델에 대해 세 가지 반복적인 프롬프트 조건을 사용하여 테스트했습니다. 각 프롬프트는 사용자의 라이프스타일 선호 맥락을 추가하여 LLM의 추천 방식이 사용자 인종 정체성과 어떻게 상호작용하는지를 분석했습니다. 결과적으로, 미신 행동은 사용자가 제시한 선호의 해석 방식에 따라 달라지며, 동일한 주거 선호라도 사용자 인종에 따라 다르게 해석될 수 있음을 발견했습니다.

- **Performance Highlights**: 연구 결과, LLM들은 추천 조건의 변화에 따라 미신 행동을 다르게 전개하며, 특정 프롬프트에서는 미신 행동을 증가시키거나 재구성할 수 있다는 점이 강조되었습니다. 또한 지역적 특성과 사회경제적 맥락에 따라 도시 별 LLM의 성능이 달라지며, 이는 LLM이 주거 검색에서 공정한 추천을 제공하는 데 있어 중요한 요소로 작용함을 시사합니다. 이 연구는 LLM이 법적 및 제도적 의무를 준수하며 공정 주거를 지원하기 위한 지역 및 도메인 전문성이 필요하다는 결론으로 마무리됩니다.



### HKJudge: A Legal Discourse-Annotated Corpus for Interpreting What Courts Find, How They Reason, and What They Ru (https://arxiv.org/abs/2606.06679)
- **What's New**: 본 연구는 홍콩에서 처음으로 법원 판결의 담론 분석을 위한 전문가 주석이 달린 데이터셋인 HKJudge를 소개합니다. 이 데이터셋은 홍콩의 법원 계층 구조에 따라 5개 수준의 형사 판결을 포함하여 총 29만개 문장과 650만개의 토큰으로 구성되어 있습니다. 두 계층의 담론 체계를 설계하여 법원이 발견하는 사실, 추론 방식, 판결 내용을 포착합니다. 또한, 두 가지 작업인 수사적 역할 분류(rhetorical role classification)와 법적 요소 추출(legal element extraction)을 공식화하고, BERT 기반 모델 및 상업적 LLM의 평가를 제공합니다.

- **Technical Details**: HKJudge 데이터셋은 3개의 스팬 수준 요소와 26개의 문장 수준 수사적 역할 클래스를 포함하는 법적 담론 체계를 개발합니다. 법적 언어학 전문가 10명이 약 148,600개의 스팬 및 292,240개의 수사적 역할 주석을 제공했으며, 전문가 간 동의 수준이 κ = 0.8에 달합니다. 데이터셋은 1968년부터 2024년까지의 4,000개 이상의 판결을 수집하기 위해 웹 스크랩퍼를 구축하였으며, 모든 법원에 대한 자료를 포함합니다.

- **Performance Highlights**: 다양한 BERT 기반 모델 및 LLM 모델을 평가한 결과, 전문가 주석자에 비해 모든 LLM 모델이 여전히 부족함을 강조합니다. 특히 미세 조정(fine-tuning)을 통해 성능이 크게 향상되었지만, 여전히 인간 전문가의 주석을 초과하지는 못합니다. 이러한 결과는 법적 LLM 추론에서 전문가 주석의 중요성을 부각시키며, 향후 연구 및 개발의 방향성을 제시합니다.



### Inside the Visual Mind: Neuroscience-Motivated Concept Circuits for Interpreting and Steering Vision Transformers (https://arxiv.org/abs/2606.06664)
Comments:
          In Proceedings of the International Conference on Machine Learning, 2026. (acceptance rate 26.6%)

- **What's New**: 이 논문에서는 Vision Transformer (ViT)의 내부 작동을 이해하기 위한 ViSAE라는 기계적 해석 도구를 제안합니다. ViSAE는 서로 다른 약간의 문제를 해결하기 위해 설계되어 있으며, 특히 인간이 이해할 수 있는 개념으로 모델의 표현을 분해할 수 있습니다. 이 도구는 64K 이미지를 포함한 프로빙 세트와 16K 시각적으로 기반을 두고 있는 개념 어휘를 제공하여 개념 범위를 20배 향상시킵니다.

- **Technical Details**: ViSAE는 두 단계의 개념 회로 추적 알고리즘을 포함하며, 상향식(Top-down) 개념 읽기와 하향식(Bottom-up) 회로 추적을 통해 ViT의 작동 방식을 연구합니다. 이 알고리즘은 CLIP을 통해 미리 학습된 개념 어휘와 내재적 표현을 연결하여 자동으로 개념을 해석합니다. 또한, 이 방법은 반대 사실 개입(counterfactual interventions)을 통해 계층 간 인과 관계를 추정하여 개념 간의 상호작용 그래프를 생성합니다.

- **Performance Highlights**: ViSAE를 사용한 결과, WaterBirds 데이터셋에서 최악의 그룹 정확도가 48.2% 향상되었으며, 이는 기존 방법보다 23.8% 우수한 성능을 보여줍니다. 이러한 결과는 ViSAE의 효과적인 개념 편집 및 모델 조정 능력을 입증합니다. 중요한 것은, 이 도구가 ViT 모델의 내부 의사 결정을 추적하고 진단할 수 있게 해줘 사용자들에게 신뢰성을 제공합니다.



### CAF-Gen: A Multi-Agent System for Enriching Argumentation Structures (https://arxiv.org/abs/2606.06646)
Comments:
          Accepted for publication in the proceedings of ICCCI 2026

- **What's New**: 이 논문에서는 복잡한 추론을 자연어 텍스트에서 정형화하는 문제를 다루고 있습니다. 기존의 Argument Mining(AM) 기법들은 기본적인 주장(claim)과 전제(premise)를 식별하는 데에 집중했으나, Carneades Argumentation Framework (CAF)와 같은 고급 구조 정보를 포착하는 데 어려움을 겪고 있었습니다. 이러한 한계를 극복하기 위해, CAF-Gen이라는 자동화된 다중 에이전트 프레임워크가 도입되어 점진적으로 얕은 주장을 CAF 준수 모델로 보강합니다.

- **Technical Details**: CAF-Gen은 Creator-Reviewer 파이프라인을 사용하여, Creator 에이전트의 출력을 Reviewer 에이전트가 검증하여 구조적 무결성을 보장합니다. 이 구조가 단일 통과 생성 모델의 전형적인 구조적 불안정을 완화하는 데 중요합니다. 이 프레임워크는 입력된 기본 주장 구조를 CAF 접근 방식의 복잡한 특성으로 풍부하게 변형합니다.

- **Performance Highlights**: 실험 결과, 반복적인 피드백 루프가 생성된 데이터의 품질을 향상시키고 원본 주석(original annotations)과의 강한 정렬을 달성했음을 보였습니다. CAF-Gen 시스템은 단일 통과 생성의 한계를 극복하고 형식적인 주장을 자동 모델링하는 신뢰할 수 있는 방법론을 제공하는 데 성공했습니다.



### How Language Models Fail: Token-Level Signatures of Committed and Persistent Reasoning Failures (https://arxiv.org/abs/2606.06635)
- **What's New**: 본 논문에서는 언어 모델의 추론 실패를 측정하는 새로운 프레임워크를 제안합니다. 이는 토큰 수준의 불확실성 신호를 활용하여 두 가지 근본적으로 다른 실패 모드를 식별합니다: "committed failure"와 "persistent uncertainty"입니다. 이 프레임워크는 23개의 모델-데이터셋 구성에서 그 예측이 유효임을 입증하였으며, 특정 시점에서 모델이 잘못된 경로에 고착되는 "commitment point"를 정의합니다.

- **Technical Details**: 제안된 프레임워크는 모델의 추론 과정에서 토큰 수준의 불확실성 신호를 분석하여 실패가 발생하는 과정을 정량화합니다. 'committed failure'에서는 모델이 잘못된 추론 경로에 고착된 이후 불확실성이 증가하지 않고, 불확실성 신호가 최대인 지점을 'commitment point'로 정의합니다. 반면, 'persistent uncertainty'에서는 모델이 지속적으로 불확실성을 유지하며, 모든 토큰의 추론이 성공 또는 실패를 식별하는 데 필수적입니다.

- **Performance Highlights**: 이 프레임워크는 다양한 모델과 데이터셋을 통해 두 가지 실패 모드를 성공적으로 식별하며, 20/23개의 사례에서 그 예측의 유효성을 입증했습니다. 또한, 이 연구는 언어 모델의 자기 일관성(self-consistency)과의 관계를 명확히 하여, 불확실성 신호가 언제 자기 일관성과 상호 보완적으로 작용하는지 예측할 수 있는 가능성을 보여줍니다. 이러한 결과는 LLM(대형 언어 모델)에서의 추론 실패를 감지하고 적절한 대응 전략을 개발하는 데 도움을 줄 수 있습니다.



### What Matters When Cotraining Robot Manipulation Policies on Everyday Human Videos? (https://arxiv.org/abs/2606.06627)
Comments:
          The project website is here: this https URL

- **What's New**: 이번 연구에서는 532개의 인간 동영상과 28시간의 고품질 3D 손 레이블을 포함한 새로운 데이터셋을 사용하여 로봇 정책을 공동 훈련(cotraining) 하는 방법을 탐구합니다. 인터넷 동영상을 통해 인간의 자연스러운 동작을 활용할 수 있는 가능성을 제시하며, 손 자세의 품질이 로봇으로의 전이에 영향을 미친다는 것을 발견했습니다. 연구진은 고품질 손 레이블이 있는 데이터셋을 통해 로봇이 더 성공적으로 학습하도록 하는 방법을 제안하고, 전이의 주요 요인으로 3D 손 자세의 질과 자연 인간 동작의 차이를 밝혔습니다.

- **Technical Details**: 연구진은 EgoExo4D 데이터셋을 기반으로 532개의 요리 동영상에서 고품질 3D 손을 다중 뷰 삼각 측량(multi-view triangulation)하여 새로운 데이터셋을 구축했습니다. 이 과정에서 2D 키포인트의 정확성을 향상시키기 위해 모델 기반 포즈 추정기를 활용하고, 손 자세 추정기를 재조정(retraining)하여 기존의 2D 추정기보다 더 나은 결과를 얻었습니다. 또한, 미니배치(mini-batch) 훈련 시 손 자세의 품질과 전이에 대한 주요 이슈를 식별하여, 기술적 접근 방식으로 비율 정렬(scale alignment)과 토큰 단위 융합(token-level fusion)을 제안하였습니다.

- **Performance Highlights**: 연구는 532개의 인간 동영상과 3,000개의 로봇 시연으로 훈련하여 6가지 실제 조작 작업에 대해 평가를 실시했습니다. 공동 훈련 방법은 로봇 데이터가 적은 환경에서 +29.7%의 절대 성과 향상을 보였으며, 특정 작업에 맞춘 로봇 데이터가 늘어날수록 성과는 계속 향상되었습니다. 이는 연구진이 제안한 데이터셋이 기존의 대규모 실험실 데이터셋보다 로봇 전이 성과가 우수함을 입증하는 결과입니다.



### ChronoForest: Closed-Loop Multi-Tree Diffusion Planning for Efficient Bridge Search and Route Composition (https://arxiv.org/abs/2606.06618)
Comments:
          40 pages, 4 figures, 7 tables, 3 algorithms

- **What's New**: 이 논문에서는 오프라인 네비게이션의 도전 과제인 단기 데이터만으로 장기 경로를 계획하는 문제를 다룹니다. ChronoForest라는 폐쇄 루프 계획 시스템을 제안하여, 단기 오프라인 데이터에서 장기 목표지점으로의 경로를 효율적으로 생성하는 방법을 제시합니다. 이 시스템은 시간적 거리(temporal distance)와 다수의 트리를 활용하여 경로 재해결을 수행하며, 목표 도달을 위한 연결성을 검증합니다.

- **Technical Details**: ChronoForest는 로컬 브리지 탐색(local bridge search)과 온라인 경로 재해결을 결합한 구조로, 낮은 수준의 앵커 체인 트리 확산 계획가(Anchor-chaining tree diffusion planner)와 온라인 다중 트리 조율자(Online multi-tree orchestrator)를 포함합니다. 저렴한 탐색 예산 내에서 성과를 최적화하며, 앵커 간 비용을 실시간으로 평가하여 경로 품질을 향상시킵니다. 이 시스템은 각 앵커 쌍을 단기적으로 연결할 수 있는 다리(brige)를 효율적으로 생성합니다.

- **Performance Highlights**: ChronoForest는 OGBench AntMaze-Stitch 데이터 세트에서 각각 99.8%, 99.3%, 99.5%의 성공률을 기록하며 이전의 확산 기반 결과보다 34.5 포인트 향상된 성과를 보였습니다. 또한, 헬미토니안 경로 구성(Hamiltonian route-composition) benchmark에서는 온라인 재해결을 통해 시간적 순서 오류를 교정하고 경로 품질을 개선하면서도 전면적인 탐색 계획(exhaustive planning)보다 훨씬 낮은 비용을 유지했습니다.



### FIGMA: Towards FIne-Grained Music retrievA (https://arxiv.org/abs/2606.06615)
Comments:
          Accepted to ACL 2026. Project Website: this https URL

- **What's New**: 이번 연구에서는 자연어 설명을 통한 음악 검색의 한계를 개선하기 위해 FIGMA (FIne-Grained Music RetrievAl)라는 새로운 다중 관점 대비 모델을 제안하였습니다. 기존 모델들은 세부적인 음악 속성을 정확히 검색하는 데 한계를 보였으며, 이는 기존 대비 학습의 목표에 기인했습니다. FIGMA는 오디오-텍스트 간의 전역 정렬과 프레임 레벨, 토큰 기반 정렬을 함께 최적화하여 이러한 문제를 해결하고 있습니다.

- **Technical Details**: FIGMA는 기존 CLAP 기반 모델이 긴 캡션의 정보를 효과적으로 활용하지 못하는 문제를 해결하기 위해 멀티 뷰 대비 손실을 포함하는 새로운 구조입니다. 이 모델은 전통적인 글로벌 대비 목표 외에도 오디오 프레임과 캡션 토큰 간의 명시적 정렬을 수행하여 높은 수준의 의미적 컨텍스트와 세부적인 음악 속성을 함께 캡처할 수 있습니다. FGMCaps라는 대규모 데이터셋을 생성하여 tempo, key, chord progression, beat count 등을 주석으로 추가하여 훈련과 테스트에 사용하고 있습니다.

- **Performance Highlights**: 광범위한 실험을 통해 FIGMA는 기존의 CLAP 기반 음악 검색 모델들을 지속적으로 초과 달성하며, 여러 음악 검색 벤치마크에서 최대 73.3%의 상대적 개선을 보였습니다. FIGMA는 세부적인 속성에 대한 검색 능력에서 탁월한 성능을 나타내어, 음악가와 프로듀서들이 정확한 음악적 사양에 따라 오디오 자료를 효율적으로 찾고 재사용하는 데 기여할 것으로 기대됩니다.



### Re-Centering Humans in LLM Personalization (https://arxiv.org/abs/2606.06614)
- **What's New**: 이 논문에서는 기존의 합성 데이터(synthetic data)에 의존했던 대형 언어 모델(LLMs)의 개인화(personalization) 능력을 평가하는 연구의 틈새를 분석합니다. 실제 사용자 대화 및 판단을 통해 시스템의 개인화 성능 차이를 조사하고, 550개의 대화와 5,949개 사용자 특성 판단을 수집했습니다. 연구 결과, LLMs는 인간 대화에서 사용자 특성을 추출하는 데 어려움을 겪고, 관련 특성을 신규 프롬프트와 연결하는 데 있어 인간의 판단과 불일치하는 경향이 있다는 것을 보여줍니다.

- **Technical Details**: 논문에서는 세 단계로 구성된 개인화 파이프라인을 제시합니다: (1) 대화에서 사용자 특성 추출 (user attribute extraction), (2) 현재 상호작용 맥락에 맞는 특성의 적합성 매칭 (attribute relevance matching), (3) 선택된 특성에 기반하여 개인화된 응답 생성 (personalized response generation). 연구에서는 각 단계에서의 모델 성능을 비교하고, 인간의 데이터와 판단과의 정렬도를 측정합니다. RoBERTa와 같은 경량 검증 모델을 사용하여 특성 추출 단계의 문제를 해결하기 위한 두 가지 개입을 소개합니다.

- **Performance Highlights**: 연구에서 인간 데이터를 포함시키니 모델의 한계가 드러났습니다. 예를 들어, LLM은 인간 대화에서 22%의 사용자 특성을 잘못 추출하였고, 적합성 매칭에서 인간의 기준과 불일치하는 경우가 20-40%에 달했습니다. 또한, 인간은 LLM의 개인화된 응답이 일반적인 응답보다 효과적이라고 판단하지 않는 경우가 54.6%에 이르는 등, 개인화에 대한 어려움을 강조하였습니다. 이러한 결과는 자동화된 개인화 평가가 인간의 데이터에 더 가깝도록 만드는 것이 매우 중요함을 시사합니다.



### Direct 3D-Aware Object Insertion via Decomposed Visual Proxies (https://arxiv.org/abs/2606.06601)
Comments:
          ICML 2026; Project Page: this https URL

- **What's New**: OBJECT INSERTION (객체 삽입) 작업에서 최신 방법들은 Reference Object (참조 객체)의 착합을 배경 이미지의 특정 영역에 Seamless하게 수행하는 데 초점을 맞추고 있습니다. 반면 기존의 Diffusion 기반 방법들은 2D Inpainting (2D 인페인팅) 작업으로만 제한되어 있어 3D Pose (3D 포즈)에 대한 명확한 조정이 불가능한 점이 한계로 지적됩니다. 이러한 문제를 해결하기 위해, 우리는 Pose-Controlled Object Insertion (포즈 제어 객체 삽입)을 가능하게 하는 새로운 프레임워크인 DIRECT (Decomposed Injection for Reference Composition and Target-integration)를 제안합니다.

- **Technical Details**: DIRECT는 Appearance (외관), Geometry (기하학), Context (맥락)의 세 가지 상호 보완적인 구성 요소로 삽입 조건을 분해하여 각 요소를 별도의 경로로 주입하는 방식으로 설계되었습니다. 이를 통해 모델은 User-specified Pose (사용자 지정 포즈)를 엄격하게 따르면서도 Reference Appearance (참조 외관)을 동시에 유지할 수 있습니다. 또한, 새로운 데이터 구축 파이프라인을 통해 다양성과 질 높은 훈련 데이터 셋을 구축하여 모델의 일반화를 향상시킵니다.

- **Performance Highlights**: 실험 결과, DIRECT는 이전 방법들에 비해 Geometry Controllability (기하학적 제어 가능성)와 Visual Quality (시각적 품질) 모두에서 우수한 성능을 보였습니다. 본 연구의 방법이 3D 프라이어의 결함과 육안의 복잡한 포즈 변형에 대해 뛰어난 강건성을 가지고 있음을 보여주어 기존 방법들이 겪던 기하학적 왜곡과 텍스처 저하 문제를 효과적으로 해결했습니다. 이러한 결과는 특히 현실 세계의 복잡한 장면에서 모델의 일반화 능력을 향상시키는 데 기여합니다.



### Generative Models Erode Human Temporal Learning Through Market Selection (https://arxiv.org/abs/2606.06572)
Comments:
          Accepted at ICML 2026

- **What's New**: 이 논문은 현대 생성 모델이 AGI(Artificial General Intelligence) 능력 수준 이전에도 지식과 문화 생산에 구조적 위험을 초래한다고 주장합니다. 저자들은 지속적인 문제 해결 참여를 통한 경로 의존적 지식 축적을 의미하는 Human Temporal Learning (HTL) 개념을 도입합니다. 최근 생성 모델의 출력물은 HTL 기법을 활용한 작업과 유사해지므로, 출력물의 진정한 인간 학습을 검증하는 데 드는 비용이 증가합니다. 이에 따라 평가자들은 생산 방식에 관계없이 출력을 점수화하게 되며, 이는 HTL에 투자한 생산자들에게 가격 경쟁으로 이어지는 경로 가치를 무너뜨리게 만듭니다.

- **Technical Details**: HTL(인간 시간 학습)은 문제에 대한 지속적 참여를 통해 이해가 발전하는 방식을 설명합니다. 이전에는 연구 논문이나 창작 작품과 같은 출력물이 지속적 참여에 의한 지식 축적의 신호로 작용했으나, 생성 모델은 이러한 작업을 표면 특성만으로 모방하여 실제 학습 과정을 생략하고 출력합니다. 이를 위해 두 가지 생산 모드를 제시하였고, 각 모드의 경제적 특성을 바탕으로 검증의 복잡성을 분석합니다. 구체적으로, HTL-집중 출력과 저-HTL 출력의 검증 가능성이 경제적 심사와 마켓 다이나믹스에 미치는 영향을 설명합니다.

- **Performance Highlights**: 논문은 검증 저하 현상이 임상 의학, 법률 관행, 콘텐츠 플랫폼 및 소프트웨어 보안 분야에서 어떻게 진행되고 있는지를 제시합니다. 각 분야에서는 다양한 비율로 검증의 저하가 일어나고 있으며, 임상 의학과 법률 분야에서는 실질적인 검증이 여전히 진행되고 있으나, 이는 대가가 큰 고품질 결과물을 확보하기 위한 것이기도 합니다. 반면, 콘텐츠와 소프트웨어 분야에서는 검증의 필요성이 줄어들고 있으며, 이는 K-카테기(knowledge category)가 선을 넘어 무분별한 평가로 이어질 수 있다는 위험을 내포하고 있습니다.



### MalTree: Tracing Malware Evolution from Embeddings at Sca (https://arxiv.org/abs/2606.06570)
Comments:
          33 pages, accepted at ICML 2026

- **What's New**: 이번 연구에서는 전통적인 악성코드 감지 방식이 다소 소극적이라는 점을 지적하며, 악성코드 가족 간의 진화적 관계를 이해하는 것이 선제적인 방어에 기여할 수 있음을 보여줍니다. 기존의 역공학(reverse engineering) 방법은 수개월에서 수년까지 소요될 수 있는데, 이를 자동화하는 MalTree라는 새로운 프레임워크를 제안합니다. MalTree는 생물정보학(bioinformatics)에서 영감을 받은 계통발생학적 기법(phylogenetic techniques)으로 악성코드 진화를 모델링합니다.

- **Technical Details**: MalTree는 구조적(structural), 행동적(behavioral), 이미지 기반(image-based) 기능을 아우르는 다양한 특징을 분석하여 악성코드의 진화 과정을 자동으로 모델링합니다. 이 시스템은 VirusTotal의 타임스탬프를 이용하여 유추된 계통도가 실제 진화 순서를 반영하는지 평가하는 시간적 검증(temporal validation)을 도입합니다. 결과적으로 MalTree는 87%의 시간적 일관성을 달성하였고, 이는 추정된 진화 관계가 실제 발생 시간과 밀접하게 연관되어 있음을 나타냅니다.

- **Performance Highlights**: 분석 결과, 일부 악성코드 가족은 다른 가족들보다 10배 이상 빠르게 돌연변이를 겪으며, 이는 탐지 전략이 특정 가족의 진화 속도에 맞춰 조정되어야 함을 시사합니다. Mirai 봇넷을 포함한 사례 연구(case studies)에서는 우리의 계통발생학적 나무에서 유추된 관계가 문서화된 위협 정보(threat intelligence)와 일치함을 확인하였습니다. 이러한 강력한 분석 기반은 악성코드 분석을 샘플별 분류(sample-by-sample classification)에서 계통에 대한 인식을 포함한 진화적 모델링으로 전환할 수 있는 기초를 제공합니다.



### NTILC: Neural Tool Invocation via Learned Compression (https://arxiv.org/abs/2606.06566)
Comments:
          10 Pages, 4 Figures, 5 Tables, 1 Algorithm

- **What's New**: 본 논문에서는 NTILC(Neural Tool Invocation via Learned Compression)라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 인-컨텍스트 툴 레지스트리 검색을 지능적으로 대체하여, 사용자의 의도 및 툴 사양을 공유된 임베딩 공간으로 매핑합니다. NTILC는 선택된 툴 스키마에만 모델을 조건지어 줌으로써, 정밀하고 제한적인 인자 생성을 가능하게 합니다.

- **Technical Details**: NTILC는 각 툴의 서명에 기반하여 생성된 제약을 통해 의미적 유사성을 증강하는 시그니처 인식 복합 목표를 중심으로 작동합니다. 이를 통해 모델은 호환되지 않는 툴 간의 separation을 강제하게 됩니다. NTILC는 Circle Loss와 Functional Margin Loss를 결합하여 임베딩 공간 내에서 의미론적으로 일관되면서도 기능적으로 호환 가능한 툴을 선택할 수 있게 돕습니다.

- **Performance Highlights**: NTILC는 공개된 툴 선택 및 함수 호출 데이터 세트에서 검증되었으며, 토큰 사용량, 검색 정확도 및 선택 대기시간 메트릭을 보고합니다. 평가 결과, NTILC는 컨텍스트 윈도우 소비를 95% 이상 줄였고, 추론 지연 시간을 최대 74%까지 감소시켰습니다.



### WAV: Multi-Resolution Block Residual Routing for Deep Decoder-Only Transformers (https://arxiv.org/abs/2606.06564)
Comments:
          6 pages, 4 figures, 3 tables

- **What's New**: WAV v1는 decoder-only Transformer를 위한 경량의 다중 해상도 잔차 라우팅 방법을 제안하고 있습니다. 이 방법은 각 블록을 축적된 잔차 합계로만 표현하는 대신, 주의(attention)와 MLP 업데이트 간의 차이를 나타내는 두 가지 방향적(detail) 기초를 추가합니다. 이러한 새로운 구조는 블록 내 방향 정보를 보존하여 잔차 라우팅의 효과를 향상시킵니다.

- **Technical Details**: WAV v1는 기존의 블록 요약 블록 합계(Cb)와 함께 두 개의 제로합 방향적 기초(Dbphase와 Dbsplit)를 저장합니다. 이 기초들은 블록의 잔차 업데이트를 생성할 때 온라인으로 누적되어 계산되며, 훈련의 안정성을 높이기 위해 음의 초기 편향과 RMS 매칭이 사용됩니다. 이 방식을 통해 깊은 Transformer 모델에 대해 보다 효과적인 잔차 라우팅이 가능합니다.

- **Performance Highlights**: WAV v1는 TinyStories와 Text8 데이터셋에서 12, 24, 48층의 GPT decoder-only 모델로 실험을 진행한 결과, 잔차 깊이에 따라 성능이 향상되는 경향을 보였습니다. 48층에서는 WAV v1이 Block AttnRes에 비해 검증 손실이 개선되었으며, 모든 평가된 잔차 메커니즘 중 최고의 성능을 기록했습니다. 이러한 결과는 깊은 Transformer 모델에서 잔차 라우팅의 중요성을 강조하고 있습니다.



### AI-Driven Test Case Generation from Natural Language Requirements: A Survey of Techniques and Research Gaps (https://arxiv.org/abs/2606.06563)
Comments:
          22 pages, 7 figures, 4 tables

- **What's New**: 이번 논문은 소프트웨어 테스팅의 중요성을 강조하며, 자연어 요구사항으로부터 테스트 케이스를 생성하는 데 있어 인공지능(AI)과 자연어 처리(NLP) 기술의 발전을 살펴봅니다. 기존의 접근법들이 여러 품질 차원을 충족시키지 못하며 새로운 위험을 동반하고 있다는 점에 주목하고 있습니다.

- **Technical Details**: 연구에서는 2000년부터 2025년까지의 주요 학술 데이터베이스를 바탕으로 21개의 주요 연구를 발굴하였습니다. 특히 자동화(Automation), 모호성 처리(Ambiguity handling), 도메인 적합성(Domain applicability) 등을 포함한 여섯 가지 품질 차원이 기존 접근법들에 의해 충족되지 않고 있음을 밝혔습니다.

- **Performance Highlights**: 이 설문 조사는 AI 기반 테스트 생성을 위한 세 가지 진화적 합성을 제시하고, 현재 접근법이 모든 품질 차원을 완벽하게 다루지 못한다는 사실을 강조합니다. 또한, 환각(Hallucination), 추적 가능성(Traceability), 복잡성 민감도(Complexity sensitivity) 및 준수(Compliance)에 대한 네 가지 연구 가이드를 제시하여 향후 연구 방향을 제안합니다.



### MacArena: Benchmarking Computer Use Agents on an Online macOS Environmen (https://arxiv.org/abs/2606.06560)
Comments:
          Accepted to the Second Workshop on Agents in the Wild: Safety, Security, and Beyond (AIWILD) at ICML 2026

- **What's New**: 이 논문에서는 MacArena라는 새로운 벤치마크를 소개합니다. MacArena는 421개의 수동 검증 작업을 포함하며, 50개의 애플리케이션을 포괄합니다. 기존의 macOSWorld는 제한된 작업 공간만을 다루고 있었으나, MacArena는 OSWorld에 기반한 여러 작업, macOSWorld의 콘텐츠, 그리고 49개의 새로운 macOS 원주율 작업을 조합하여 제안합니다. 이를 통해 현재의 GUI 에이전트들이 macOS환경에서 직면하는 고유한 도전들을 더 잘 이해하고 평가할 수 있도록 합니다.

- **Technical Details**: MacArena는 부분 관측 마르코프 결정 과정(Partially Observable Markov Decision Process, POMDP)으로 형식화되어 있습니다. 여기서 상태 공간(𝒮)과 관측 공간(𝒪), 행동 공간(𝒜), 전이 함수(𝒯) 등이 정의되며, 에이전트가 접근할 수 있는 시스템 상태의 다양한 측면을 모델링합니다. 이를 통해 macOS 환경에서의 자동화 에이전트의 상호작용을 보다 세밀하게 추적할 수 있습니다. 또한, 모든 작업이 수동으로 검증되어 그 신뢰성을 높이고 있습니다.

- **Performance Highlights**: MacArena에서의 성능 분석 결과, 현재 평가된 모든 모델의 성능이 Linux에 비해 저하되는 경향을 보였습니다. 이는 macOS에서의 환경이 현재의 GUI 에이전트에게 실질적으로 더 어려운 도전이 있음을 나타냅니다. 특히, 이 논문에서는 macOS 원주율 작업에서 상위 모델이 26% 이상 떨어지는 성과를 보여, 현 시스템들이 macOS에서 더욱 효과적으로 작동하지 못함을 강조하고 있습니다.



### IRAF: Interference-Resilient Adaptive Fusion for Noise-Robust End-to-End Full-Duplex Spoken Dialogue Systems (https://arxiv.org/abs/2606.06559)
- **What's New**: 이번 논문에서는 실시간 대화에서 음성 에이전트가 동시에 듣고 말할 수 있도록 지원하는 풀듀플렉스 대화 모델을 소개합니다. 기존의 엔드투엔드(End-to-End) 모델들은 현실적인 음향 환경에서 사용자 쿼리에 다른 사람의 잡음이 혼입될 경우 성능이 저하될 수 있습니다. 이에 대한 해결책으로 제안된 Interference-Resilient Adaptive Fusion (IRAF) 모듈은 사용자의 오디오 기여도를 동적으로 조정하여 훈련 성능을 개선합니다.

- **Technical Details**: IRAF는 목표 화자(target speaker)와 사용자 오디오의 임베딩을 기반으로 신뢰도 게이트를 예측합니다. 이를 통해 사용자 표현을 재조정하고 에이전트 임베딩과 결합하는 방식을 사용합니다. 모델은 LLM 백본과 함께 오디오와 텍스트의 멀티 채널 다음 토큰 예측 목표를 통해 E2E 최적화를 지원합니다. 이 구조는 모델의 복잡성을 줄이는 동시에 일관된 맥락을 유지합니다.

- **Performance Highlights**: MS-MARCO와 InstructS2S-200K 데이터셋에서 실시한 실험 결과, 제안된 모델이 응답 품질과 풀듀플렉스 상호작용에서 일관된 개선을 보였습니다. 특히, 잡음이 있는 환경에서도 안정적인 턴-테이킹(turn-taking)과 답변 품질이 향상되었으며, 기존의 강력한 기준선보다 성능이 뛰어났습니다. 해당 방법은 복잡한 음향 환경에서의 적용 가능성을 보여줍니다.



### Multi-Scale Feature Attention Network for Polymer Classification using THz Dual-Comb Spectroscopy (https://arxiv.org/abs/2606.06554)
Comments:
          Accepted in EUSIPCO'26

- **What's New**: 이 연구는 THz-Dual-Comb Spectroscopy (THz-DCS)를 활용하여 12종의 폴리머를 분류하는 새로운 방식을 제안합니다. 이 방식은 빠르고 고해상도의 비파괴 측정을 가능하게 하여 재활용 플라스틱의 품질과 안전성을 보장할 수 있습니다. 특히, Multi-Scale Feature Attention Network (MSFAN)이라는 딥러닝 구조를 통해 복잡한 스펙트럼 신호를 효과적으로 처리하여 분류 정확도를 85.2%에 도달하게 했습니다.

- **Technical Details**: MSFAN은 스펙트럼 선택, 다중 스케일 합성곱, 그리고 주의 메커니즘을 통합한 새로운 딥러닝 아키텍처입니다. 기존의 전통적인 스펙트로스코피 기술의 한계를 극복하기 위해, 두 개의 주파수 콤을 사용하는 THz-DCS를 활용합니다. 스펙트럼 신호의 두께 의존성을 해결하기 위해 원시 스펙트럼 진폭을 샘플 두께에 비례하여 정규화하는 방법을 도입하여, 내부 스펙트럼 특성에만 의존할 수 있도록 하였습니다.

- **Performance Highlights**: 이 연구는 MSFAN이 기존의 최신 모델보다 일관되게 우수한 성능을 보이면서 폴리머의 효과적이고 해석 가능한 분류를 가능하게 한다는 것을 보여줍니다. 제안된 프레임워크는 다양한 폴리머 샘플에서 실험을 통해 그 유용성을 입증하였으며, 결과적으로 고해상도 THz 스펙트럼을 활용한 정확한 폴리머 분류의 가능성을 실증하였습니다. THz-DCS와 딥러닝 기술의 결합이 폴리머 분류의 새로운 기준을 제시할 것으로 기대됩니다.



### Geometric Second-Order Feature Correlation Learning for Self-Supervised Speech Emotion Recognition (https://arxiv.org/abs/2606.06550)
- **What's New**: 이 논문에서는 음성 감정 인식(SER)을 위한 새로운 모델링 접근 방식인 Second-Order Correlation (SOC) 레이어를 제안합니다. 기존의 첫 번째 차수 집계 방식이 과도한 차원에서 발생하는 정보 손실 문제를 해결하고자 SOC는 특성 간의 상관관계를 공분산(descriptor)으로 모델링합니다. 이러한 접근은 기하학적으로 고유한 정보를 보존하면서 감정 인식의 robustness를 높이는 데 기여합니다.

- **Technical Details**: 이 모델은 입력으로 고차원 SSL 임베딩을 수신하고, 이것을 콤팩트한 하위 공간으로 프로젝션하여 구성적 차원 축소와 기하학적 encoding을 수행합니다. SOC 레이어는 Log-Euclidean mapping (LEM)을 사용하여 Riemannian manifold에서 유클리드(tangent space)로의 매핑을 통해 기하학적 일관성을 보장합니다. 이를 통해 첫 번째 차수 pooling에서 손실된 변별 정보를 회복하고 고차원 음성 데이터를 효과적으로 집계할 수 있습니다.

- **Performance Highlights**: ESD 및 RAVDESS 데이터셋에서 광범위한 실험을 통해 SOC가 첫 번째 차수 기준선보다 일관되게 우수한 성능을 발휘함을 입증했습니다. SOC 레이어는 감정의 모호성을 더 잘 특성화할 수 있는 비유클리드 기하학을 활용하여 SER 커뮤니티의 잠재력을 보여줍니다. 이 연구는 고차 통계 복원을 위한 효과적인 방법론으로 SER의 데이터 처리 및 인식 효율성을 극대화하는 데 중점을 둡니다.



### FAIR-Calib: Frontier-Aware Instability-Reweighted Calibration for Post-Training Quantization of Diffusion Large Language Models (https://arxiv.org/abs/2606.06547)
Comments:
          Accepted as a poster at the 43rd International Conference on Machine Learning (ICML 2026)

- **What's New**: 이 논문은 Diffusion Large Language Models (dLLMs)의 새로운 Post-Training Quantization (PTQ) 접근을 제안합니다. 기존의 PTQ 방법들이 dLLM 특유의 비가역적인 commit을 다루지 못하는 문제를 보완하기 위해, Frontier-Aware Instability-Reweighted Calibration (FAIR-Calib)이라는 두 단계의 메커니즘을 도입했습니다. 이 연구에서는 dLLMs의 불안정성을 줄이기 위해, 전체 정밀도 모델을 활용하여 fragile frontier 상태의 불확실성을 재조정하는 방법을 표현합니다.

- **Technical Details**: FAIR-Calib의 첫 번째 단계는 full-precision teacher로부터 position-aware prior를 추정하는 과정입니다. 이 prior는 비가역적인 commit과 masked-stage reliability를 통합하여 안정성을 강화합니다. 두 번째 단계에서는 레이어 별로 off-policy hidden-state MSE를 최소화하여 FRontal states를 효과적으로 보호하며, 이는 비용이 많이 드는 end-to-end diffusion rollouts를 피할 수 있게 도와줍니다.

- **Performance Highlights**: FAIR-Calib는 LLaDA 및 Dream (W4A4) 벤치마크에서 기존 최첨단 기법보다 일관되게 우수한 성능을 보여줍니다. 이 방법은 decision flips를 효과적으로 줄이고, post-commit mismatch를 감소시키는 데 성공하여 최종 생성 품질을 크게 향상시킵니다. 이 결과들은 더 적은 teacher-forced commit-step flips와 함께, sequential error amplification을 줄이는 성과로 나타났습니다.



### Queen-Bee Agents: A BeeSpec-Centered Architecture for Governed Enterprise MCP Orchestration (https://arxiv.org/abs/2606.06545)
Comments:
          Technical report. Prototype-level systems evidence; 59 enterprise-style tasks

- **What's New**: 최근 기업 환경에서 대규모 언어 모델(Large Language Model, LLM)과의 통합이 필요해졌으며, 이로 인해 정책 시행과 임차인 범위 격리와 같은 요소들이 중요해졌다. 본 논문은 Queen-Bee라는 다중 에이전트 아키텍처를 제안하는데, 이는 구체적인 BeeSpec 명세를 이용하여 전문화된 Bee 에이전트들이 제한된 도구 접근 하에서 작업을 수행하도록 설계되었다. 이 시스템은 다양한 엔터프라이즈 과제를 수행하기 위해 59개의 실험 과제에서 평가되었다.

- **Technical Details**: Queen-Bee 아키텍처는 실행을 네 개의 레이어로 분리합니다. Queen control plane은 능력 검색과 계획 수립을 담당하며, BeeSpec 생성 및 정책 결정을 수행합니다. BeeSpec은 각 Bee의 실행 경계를 명확히 정의하며, 이를 통해 정책 집행과 감사가 용이해집니다. 이 구조는 미리 정의된 범위 내에서 작업을 수행할 수 있도록 보장합니다.

- **Performance Highlights**: Queen-Bee의 retrieval-driven 변형은 0.964의 작업 성공률과 제로(Zero) 거버넌스 실패율을 기록하여 기존의 static baseline 및 허용적인 단일 에이전트 모델에 비해 더 나은 실행 품질을 보여주었습니다. 여러 공급 방식들과의 비교 결과, 현재의 경량화된 구조적 검색기(lightweight structured retriever)가 가장 효율적임을 입증했습니다.



### Coordinated optimization of departure sequencing and section-track allocation in railway short-term concentrated departure scenarios based on qubo and hybrid quantum algorithms (https://arxiv.org/abs/2606.06543)
- **What's New**: 이 연구는 철도 단기 집중 출발 시나리오에서 출발 순서 최적화와 구역 트랙 할당의 연계 최적화를 검토합니다. 이 과정에서 이차 무제한 이진 최적화(Quadratic Unconstrained Binary Optimization; QUBO) 모델이 출발 위치 할당과 구역 트랙 선택을 통합된 이진 프레임워크 내에서 표기하는 데 사용됩니다. 시간 의존적인 운영 상호작용이 정적인 조합 모델로 완전히 포착될 수 없기 때문에, 시뮬레이션 기반 평가 계층이 도입되었습니다.

- **Technical Details**: 단기 집중 출발 문제는 출발 순서, 트랙 할당 및 운영의 실현 가능성을 포함하는 결합된 배치 문제로 간주되어야 합니다. QUBO는 이산 결정 변수를 정리하고 쌍별 결합을 표현할 수 있으며 다양한 운영 제약 조건을 하나의 이진 목표로 표현할 수 있는 장점이 있습니다. 본 연구에서는 QUBO를 결정 계층으로 활용하여 후보 이진 계획을 생성하고, 시뮬레이션 계층에서 이들의 운영 결과를 평가합니다.

- **Performance Highlights**: 연구 결과, QUBO 모델은 출발 순서를 최적화하고, 시뮬레이션 계층은 일반 및 방해된 조건에서의 경쟁 알고리즘의 운영 성능을 명확히 구분합니다. QPSO-QAOA는 정상 조건에서 가장 우수한 성과를 보였으며, 양자 향상 방법은 동적 조건에서 전통적인 방법에 비해 총 비용을 평균 4.28%에서 26.26%, 총 지연을 4.37%에서 24.25% 줄였습니다. 이 결과는 QUBO 기반 모델링과 시뮬레이션 기반 평가의 통합이 철도 단기 집중 출발 스케줄링에 유용한 방법론적 틀을 제공함을 시사합니다.



### Synthetic Benchmarks Overstate Forward-Forward Scaling: Real-Data Limits of Layer-Local Training (https://arxiv.org/abs/2606.06539)
Comments:
          23 pages, 6 figures

- **What's New**: 이 논문에서는 Forward-Forward (FF) 학습의 새로운 발전을 제시하며, DTG-FF라는 새로운 아키텍처를 개발했습니다. 이는 동적 온도 고도(dynamic temperature goodness), 분리된 정규화(decoupled normalization), 다층 융합(multi-layer fusion) 기술을 결합하여, FF 계열의 최신 기술을 여러 실제 데이터 벤치마크에서 확립합니다. 기존의 역전파(backpropagation) 대신 이 레이어 로컬(layer-local) 학습 방식이 실제적 규모에서 경쟁력을 유지할 수 있는지 조사했습니다.

- **Technical Details**: Forward-Forward 알고리즘은 레이어 로컬 기초 학습 방식을 통해 각 레이어가 양의 데이터에 대해 높은 '고도(goodness)'를 생성하고, 음의 데이터에 대해서는 낮은 고도를 생성하도록 훈련합니다. DTG-FF는 이 알고리즘을 실험적 목적으로 사용하기 위해 설계되었으며, 이를 통해 실제 데이터 스케일 진단과 서로 다른 아키텍처 간의 효용성을 비교하였습니다. 또한, DTG-FF는 높은 성능을 기록하고, 이론적으로 메모리에서의 효율성을 제공한다고 주장합니다.

- **Performance Highlights**: DTG-FF는 CIFAR-10에서 91.8%, ImageNet-100의 첫 번째 FF 기준에서 49.4%를 기록했습니다. 그러나 동일한 조건 하에서 역전파 기반의 딥서포트(BP-DeepSup)와 비교했을 때, CIFAR-10과 CIFAR-100에서 각각 2.40 및 5.93 pp의 성능 차이를 보이며, 클래스 수가 증가할수록 성능 격차가 더욱 확대되는 경향을 보였습니다. 이 논문은 FF의 계층적 로컬 학습이 실제 데이터에서 어떤 한계가 있는지를 설명하고 있습니다.



### Attention-Guided Autoencoder Fusion for Insulator Defect Detection Using UAV Transmission-Line Imaging (https://arxiv.org/abs/2606.06536)
- **What's New**: 이 논문은 AE-YOLO라는 새로운 Attention-Guided AutoEncoder-Enhanced YOLO 프레임워크를 제안합니다. 이 프레임워크는 Insulator defect detection을 위해 설계되었으며, UAV 기반의 전송선 검사에서의 결함 검출 성능을 향상시키는 것을 목표로 합니다. AE-YOLO는 Feature Pyramid Network-Path Aggregation Network 아키텍처에 경량의 bottleneck autoencoder를 통합하여 다중 스케일 특성을 융합하는 동안 이상 감지 정보를 보존합니다.

- **Technical Details**: AE-YOLO는 Convolutional Block Attention Modules (CBAM)을 backbone에 적용하여 특성 분별력을 강화하고 배경 간섭을 억제합니다. 논문에서는 focal loss, Complete IoU (CIoU) loss 및 autoencoder 정규화를 통합한 통합 최적화 목표를 개발하여 전경-배경 불균형 문제를 해결하고 로컬라이제이션 정확도를 개선합니다. 또한, Weighted Boxes Fusion (WBF)와 autoencoder-guided 신뢰도 향상 메커니즘을 도입하여 드문 결함 카테고리에 대한 민감도를 높입니다.

- **Performance Highlights**: Insulator-Defect Detection 데이터셋에서 AE-YOLO는 EfficientNetV2 backbone을 사용할 때 0.5에서 95.10% mAP, 96.40% precision, 93.80% recall을 달성했습니다. 이러한 성능은 YOLO 계열의 가장 강력한 기준 모델보다 5.0 포인트 높은 mAP와 6.7 포인트 높은 recall을 달성하였으며, 이는 AE-YOLO의 효과와 적응 가능성을 확인시킵니다. 이 모델은 UAV 기반의 전송선 검사 및 결함 모니터링을 위한 실용적이고 확장 가능한 솔루션으로 자리 잡을 것으로 예상됩니다.



### Attention Consistent Longitudinal Medical Visual Question Answering Guided by Vision Foundation Models (https://arxiv.org/abs/2606.06534)
Comments:
          Accepted to CVPR 2026 Workshop PHAROS-AIF-MIH

- **What's New**: 이번 연구는 흉부 X-ray를 대상으로 하는 장기적인 의료 비주얼 질문 응답(Longitudinal medical visual question answering, VQA) 문제에 대한 새로운 접근법을 제시합니다. 기존의 직접 대비 방식 대신, 경량의 어파인 등록 모듈(affine registration module)을 포함하여 현재 이미지를 기준 이미지(reference image)와 맞추어 정렬합니다. 이 방법은 변형된 이미지 쌍을 인코더에 입력하여, 텍스트 특징과 결합하여 최종 응답을 생성하는 멀티모달 변환기(Multimodal transformer) 기반 디코더로 처리됩니다.

- **Technical Details**: 이 모델은 장기적 비주얼 질문 응답을 위한 다양한 모듈을 포함하고 있습니다. 경량 CNN 기반의 마이크로 사전 정렬(micro pre-alignment) 모듈은 작은 포즈와 스케일 변화를 완화하고, 자가 감독(self-supervised) DINO 모델을 사용해 강건한 병변 후보를 생성합니다. 모델은 다양하고 다중 기하학적 훈련 목표들을 채택하여, 관심 있는 이미지의 특성과 마스크 특성 간의 내부 곱(Inner Product)의 차이를 제약하고, 공간 정렬을 촉진합니다.

- **Performance Highlights**: 모델은 Medical-Diff-VQA 기준에서 강력한 BLEU, ROUGE-L, CIDEr 및 METEOR 점수를 달성하였습니다. 또한, 공유된 주목 마스크를 통해 본질적인 해석 가능성을 제공하며, 포스트 호크 설명(post-hoc explanation) 없이도 의사에게 직관적인 결과를 제공합니다. 이러한 성과는 더 많은 의학 VQA 연구에 대한 길잡이로 작용할 수 있으며, 블랙 박스 속성에 기인한 불신을 줄이는 데 기여할 수 있습니다.



### Agentic Large Language Models for Automated Structural Analysis of 3D Frame Systems (https://arxiv.org/abs/2606.06525)
- **What's New**: 이 논문은 자연어 입력을 기반으로 자동화된 3D 프레임 구조 분석을 위한 agentic LLM 프레임워크를 제안합니다. 기존의 plane beam 모델링에서 3D 모델링으로의 확장은 도전적이었으며, 불규칙한 기하학 표현 및 긴 추론 체인 등의 문제를 해결하는 방법을 모색합니다. 특히, 제안된 방법은 패러미터화된 2D 계획으로의 투영을 통해 3D 프레임을 효율적으로 표현하는 혁신적인 기법을 포함하고 있습니다.

- **Technical Details**: 3D 프레임의 구조 분석을 위해 제안된 프레임워크는 여러 에이전트를 포함한 파이프라인으로 구성됩니다. 문제 분석 에이전트는 입력을 구조화된 JSON으로 파싱하고, 층 분해 에이전트는 각 층의 공간 배치를 도출합니다. 각 층의 노드, 기둥 및 슬래브 에이전트는 병렬적으로 작동하여 노드 좌표와 평면 연결을 생성하며, 코드 변환 에이전트는 결과를 SAP2000 스크립트로 변환합니다.

- **Performance Highlights**: 이 프레임워크는 10개의 대표 3D 프레임을 대상으로 평가한 결과, 평균 90%의 정확도를 달성했습니다. 반복 테스트에서도 일관된 성능을 보였으며, 평균 실행 시간은 3분 이내, 비용은 약 0.20달러로 나타났습니다. 이러한 결과는 기존 SOTA 일반 목적 LLMs에 비해 유의미하게 높은 성능을 기록한 것입니다.



### P-Cast Precision in FP8 Attention: Sink-Induced Collapse and the Optimality of S=2^8 (https://arxiv.org/abs/2606.06521)
Comments:
          8 pages, 3 figures, 3 tables, 1 algorithm. Technical note on FP8 E4M3 P-cast precision

- **What's New**: 이번 논문은 FP8 (E4M3) 가속화가 주의(attention) 연산에서 중요한 처리량 향상을 제공하는 한 편, 소프트맥스 확률 행렬이 FP8로 변환될 때 정밀도 문제를 일으킬 수 있음을 밝혔다. 특히 KV 블록의 반복 순서와 P의 변환 전 정적 스케일링 요인이 주의가 줄어드는 현상인 Attention Sink와 상호작용하는 방식을 분석하였다.

- **Technical Details**: 논문에서는 E4M3 형식의 정밀도 균형을 맞추기 위해 두 가지 구현 선택을 자세히 분석하였다. 첫째로, KV 블록의 반복 순서가 P-collapse를 야기함을 밝혔으며, 둘째로, P의 스케일링 인자가 FP8 변환의 정밀도에 미치는 영향을 다루었다. 일반적인 구현에서 P 값의 다수는 E4M3의 표현 가능 범위를 초과하여 언더플로우 되는 현상, 즉 P-collapse를 수학적으로 정량화하였다.

- **Performance Highlights**: 실험 결과, 논문에서 제안하는 두 가지 최적화 기법은 FlashAttention-3/4에서 이미 적용되었으며, 중간 수준의 sink 강도에서 MSE(Mean Squared Error)가 3배에서 10배 개선됨을 보여주었다. 또한, 두 가지 수정사항이 함께 적용될 때 같은 정밀도 바닥에 수렴하는 것을 확인하며, E4M3 형식의 이점과 그 한계를 명확히 규명하였다.



### DxPTA: An Architecture Design Space Exploration with Optical Dataflow-guided Strategy for HW/SW Co-Design of Photonic Transformer Accelerators (https://arxiv.org/abs/2606.06515)
Comments:
          8 pages, 12 figures

- **What's New**: 이번 연구에서는 포토닉 변환기 가속기(PTA)의 설계 공간 탐색 방법인 DxPTA를 제안하여, 다양한 응용 프로그램의 제약 조건을 충족할 수 있는 효율적인 하드웨어/소프트웨어 공동 설계를 가능하게 합니다. DxPTA는 포토닉 데이터 흐름에 기반하여 PTA 아키텍처 매개변수를 식별하고 그 영향을 분석해 최적화된 아키텍처를 검색합니다.

- **Technical Details**: DxPTA는 PTA 아키텍처의 계층 구조와 포토닉 장치의 특성을 분석하여 중요한 매개변수를 식별하며, 이들 매개변수가 면적, 전력, 에너지 및 지연에 미치는 영향을 조사합니다. 이를 바탕으로 제약 조건을 충족하는 아키텍처 후보를 탐색하는 제약 인식 검색 알고리즘을 개발합니다.

- **Performance Highlights**: 실험 결과, DxPTA는 50mm²의 면적, 5W의 전력, 50mJ의 에너지 및 10ms의 지연 제한 조건을 만족하는 가속기 아키텍처를 찾을 수 있었습니다. 이 방법은 15.2배 더 빠른 탐색 시간을 기록하며, 다양한 AGI 기반 응용 프로그램을 위한 효율적인 PTA 설계를 지원하는 가능성을 보여줍니다.



### FP8 is All You Need (Part 1): Debunking Hardware FP64 as the HPC Holy Gra (https://arxiv.org/abs/2606.06510)
Comments:
          There is a companion Part (2) paper focusing on Ozaki-style FFT

- **What's New**: 이 논문은 고전적인 HPC(Higher Performance Computing) 신념을 반박하고, FP64(Floating Point 64) 하드웨어가 더 이상 과학적 계산의 필수 기반이 아님을 주장합니다. AI 최적화 GPU인 B300 세대 및 그 이후에서, FP8 텐서 처리 수치가 증가하고 Ozaki Scheme II가 메모리 제약을 극복하면서 정확한 FP64 계산을 복원할 수 있음을 보여줍니다. 또한 기존 HPC 커널에서 성능을 최대화할 수 있는 새로운 경로를 제시합니다.

- **Technical Details**: FP64 수치 처리 성능이 NPVD(Non-Persistent Volume Disks)에서 B300세대에서 1.3 TFLOPS로 감소하였으나, Ozaki II를 이용하면 더 높은 성능을 얻을 수 있는 경로를 제시합니다. 제안된 TME(Tensor-Memory Equilibrium) 모델은 다양한 파라미터를 통해 Ozaki II가 native FP64 수치 처리 성능에 도달할 수 있는 기준을 설정합니다. 이 모델은 메모리 및 계산 성능을 통합적으로 고려하여 성능 예측을 가능하게 합니다.

- **Performance Highlights**: Ozaki II는 B300의 FP64 성능을 500 TFLOPS로 증가시킬 수 있으며, 이는 B200의 native FP64 성능을 넘어서게 됩니다. H100 GPU와의 비교에서도 Ozaki II는 H100의 모든 워크로드에서 맞먹거나 초과하는 성능을 보여줍니다. 메모리 집약적인 커널에서도 Ozaki II는 메모리 제약을 극복하고 FP64의 전반적인 성능을 극대화할 수 있음을 입증합니다.



### Which Anatomy Matters Under Limited Labels? A Data-Efficient Anatomy-Aware Benchmark for Cardiac Pathology Prediction (https://arxiv.org/abs/2606.06509)
Comments:
          ACCEPTED at ICML 2026 Workshop GlobalSouthML (Seoul, South Korea; PMLR 306, 2026)

- **What's New**: 이 연구는 제한된 레이블(Labels) 하에서 심장 병리(Cardial pathology) 예측을 위한 저데이터( Low-data) 해부학적 벤치마크를 설정하는 데 초점을 맞췄습니다. ACDC MRI 데이터셋을 활용하여 심장 구조별로 정보를 인식하는 방식이 기계학습 모델의 복잡성보다 예측 성능에 더 큰 영향을 미친다는 점을 발견했습니다. 이를 통해 의료 AI 모델의 성능 개선을 위한 해부학적 정보 표현의 중요성을 강조하고 있습니다.

- **Technical Details**: 연구에서는 5개의 심장 병리 클래스 (DCM, HCM, MINF, NOR, RV)에 대해 환자 단위 강조 기능을 생성하기 위해 라벨이 지정된 심장 MRI의 세 가지 주요 해부학적 구조(RV, MYO, LV)를 사용했습니다. 다양한 분류기(fθ(r))를 통해 해부학적 표현이 예측 신호에 미치는 영향을 평가하며, 저데이터 환경에서도 의미 있는 예측 기능을 지원하는지 검증합니다. 특히, 다구조(multi-structure) 해부학적 표현이 단일 구조보다 우수한 성능을 나타내는 조건을 수립했습니다.

- **Performance Highlights**: 결과적으로 심근(MYO) 묘사가 예측 신호를 가장 잘 나타내는 특징으로 밝혀졌으며, 다구조 표현은 전반적인 성능에서 가장 우수한 결과를 보여주었습니다. 제한된 레이블 하에서도 성능이 기대치를 웃도는 결과를 나타내며, 단순한 형태의 피쳐들을 사용하는 것이 정적 해부학적 설명을 넘어서는 이점을 보이지 않았습니다. 이러한 결과는 한정된 자원에서도 의료 ML 모델링에 있어 해부학적 표현의 우선순위 설정이 중요함을 시사합니다.



### A Geometric Gaussian Mixture Representation of Plane Curves (https://arxiv.org/abs/2606.06505)
- **What's New**: 본 논문에서는 사용자가 정의한 확률적 다각형 표현(user defined probabilistic polygonal representation)을 도입하여 평면 곡선을 표현합니다. 이 표현은 곡선상의 정점을 선택하고 이를 선분으로 연결하여 얻어진 다각형 근사를 통해 생성됩니다. 각 선분은 사용자 정의 불확실성 매개변수(uncertainty parameter)가 부여되어, 기존의 결정론적 곡선 표현과 차별화된 확률적 기하학적 원시(primitives)들을 생성합니다.

- **Technical Details**: 각 선분에 대해, 우리는 선분의 접선(tangent) 방향에서 균일하게 분포하고, 법선(normal) 방향에서 가우시안 분포가 있는 랜덤 변수(Random Variable)를 정의합니다. 이 구조는 선분의 중간 지점에서 평균(mean)이 위치하고, 공분산(covariance)으로 접선과 법선의 불확실성을 캡쳐하는 가우시안 확률 밀도 함수(Gaussian Probability Density Function)를 생성합니다. 블렌딩된 가우시안 구성 요소들이 적절한 가중치와 결합되어, 평면 곡선의 가우시안 혼합 모델(Gaussian Mixture Model) 표현이 생성됩니다.

- **Performance Highlights**: 이 프레임워크는 국소 기하학(local geometry)과 법선 방향에서의 불확실성을 유지하면서도, 평면 곡선의 매끄러운, 폐곡선, 열린 곡선, 비정규 및 자기 교차형 곡선에 적용될 수 있습니다. 실험을 통해 우리는 생성된 GMM이 국소 접선, 법선, 그리고 아크 길이(local arc length)를 잘 캡처하여, 기초 곡선의 전반적인 형태(global shape)가 정확하게 포착된다는 것을 보여주었습니다. 이 기법은 불확실성을 인식하는 CAD 및 디지털 트윈, 로봇에서의 확률적 장애물 모델링, 그리고 확률적 경로 계획 등 다양한 응용 분야에 사용될 수 있습니다.



### Autonomous heterogeneous catalyst discovery with a self-evolving multi-agent digital twin (https://arxiv.org/abs/2606.05050)
- **What's New**: 최근 논문에서는 CatDT(Catalysis Digital Twin)라는 자율 디지털 트윈 시스템을 소개합니다. 이 시스템은 다중 에이전트 구조를 기반으로 하여 다양한 촉매의 안정적인 표면과 반응 경로를 예측하고, 1개의 GPU에서 5-30분 만에 계산할 수 있습니다. CatDT는 새로운 물질의 경로를 찾는 데 드는 비용을 기존 방법보다 1000배 이상 낮추는데 기여하며, 자신을 지속적으로 개선하는 특징을 가지고 있습니다.

- **Technical Details**: CatDT는 기체-고체, 액체-고체 모델링을 통합하여 작동하는 촉매의 디지털 트윈을 구축합니다. 이 시스템은 8개의 전문 에이전트와 27개의 과학 도구들로 구성되어 있으며, 반응 경로를 나열하고 순위를 매기고 전이 상태를 찾는 등 복잡한 계산 작업을 수행합니다. 또한, UniMech와 기억 증강 강화 루프와 같은 두 가지 혁신적인 접근 방식을 도입하여 전이 상태 계산의 성공률을 41%에서 84%로 향상시킵니다.

- **Performance Highlights**: CatDT는 600개의 촉매 표면을 대상으로 한 실험에서 모든 예측이 0.5-2배의 오차 범위 내에 들어갔습니다. 프로판 탈수소 반응에서 CatDT는 Ni@ZrO₂ SMSI 오버레이어를 제안하며, 약 100%의 선택성에서 시뮬레이션된 TOF(Time of Flight)는 1.63~s^{-1}에 도달합니다. 이는 비귀금속 후보가 Pt 기반의 산업 표준과 경쟁할 수 있는 가능성을 보여줍니다.



### When Does Multi-Agent Collaboration Help? An Entropy Perspectiv (https://arxiv.org/abs/2602.04234)
Comments:
          Project page: this https URL

- **What's New**: 새로운 연구는 Multi-Agent Systems (MAS)가 큰 언어 모델(LLMs)을 활용하여 복잡한 작업을 해결하기 위해 개발된다는 점을 강조합니다. 기존의 MAS의 효율성을 뒷받침하는 요소들은 아직 충분히 탐구되지 않았습니다. 본 연구에서는 MAS의 효과를 평가하기 위해 엔트로피(Entropy)를 주요 관점으로 채택하여 다양한 작업 환경에서 문제 해결 시 엔트로피의 전환을 분석하였습니다.

- **Technical Details**: 연구에서는 대규모 정보 엔트로피의 세밀한 관계를 밝히기 위해 245개의 기능을 구조적으로 분석했습니다. 이 과정에서 MAS의 성능은 주로 초기 상호작용 라운드에서의 엔트로피 동역학에 의해 결정된다는 것을 발견했습니다. 엔트로피의 최고치가 보편적으로 해로운 영향을 미친다는 역설적인 발견을 통해, SAS(Single-Agent Systems)가 약 43.3%의 경우에서 MAS보다 뛰어날 수 있다는 사실도 강조됩니다.

- **Performance Highlights**: 이 연구는 Certainty Preference, Base Entropy, Task Awareness라는 세가지 주요 통찰력을 제공합니다. 이를 통해, 엔트로피 기반 성능 향상 도구인 Entropy Judger를 도입하여 MAS의 정확성을 일관되게 높이는 방법을 제시했습니다. 최종적으로, 다양한 MAS 구성 및 작업에서 정확도를 지속적으로 향상시킬 수 있는 메커니즘을 제안하였습니다.



### Beyond Output Matching: Preserving Internal Geometry in NVFP4 LLM Distillation (https://arxiv.org/abs/2606.05682)
Comments:
          13 pages,1 figures

- **What's New**: 이 논문은 NVFP4 기반 접근법을 통해 낮은 정밀도의 추론이 점점 더 많이 요구되는 상황에서 Quantization-aware Distillation(QAD)의 새로운 진전을 소개합니다. QAD는 정량화된 학생이 고정된 고정밀 교사의 출력 분포에 맞추도록 훈련하여 낮은 비트 정량화 하에서 손실된 정확도를 복구합니다. 연구진은 CKA(Centered Kernel Alignment)를 사용하여 KL 발산만 사용하는 QAD가 레이어별 표현 유사성을 감소시킨다는 것을 보여주며, 이는 복잡한 reasoning 및 coding 작업에 대한 성능 저하로 이어진다고 주장합니다.

- **Technical Details**: CKA-QAD는 NVFP4 QAD와 낮은 비트 LLM 정확도 복구를 위한 CKA 지침을 기반으로 하는 표현 정렬 방법론입니다. 본 방법론은 CKA 유사성을 사용하여 내부 레이어의 표현 기하학을 보존하도록 설계된 경량 정규화를 도입합니다. KL 발산을 통한 출력 분포 정렬과 CKA를 통한 내부 표현 정렬을 공동으로 최적화하여 결과적으로 표면적 필드 매핑을 억제합니다.

- **Performance Highlights**: CKA-QAD는 Nemotron 3 Nano 및 Qwen3-4B-Thinking-2507 모델에서 실험 후, 특히 Qwen3-4B-Thinking-2507에서 AIME25, GPQA-D, LiveCodeBench-v5에서 평균 정확도를 각각 68.5%에서 72.3%, 59.5%에서 61.1%, 57.9%에서 59.8%로 향상시킵니다. 이러한 결과는 저비트 LLM 증류에서 CKA 기반 회복이 기능적인 정렬과 표현적 정렬의 공동 최적화로부터 이익을 받을 수 있음을 시사합니다.



New uploads on arXiv(cs.RO)

### Affordance-Based Hierarchical Reinforcement Learning for Quadruped Pedipulation (https://arxiv.org/abs/2606.07506)
Comments:
          This paper is submitted to Wiley Journal of Field Robotics

- **What's New**: 본 연구는 사족 보행 로봇의 객체 조작 능력 강화를 위한 세 가지 수준의 계층적 강화 학습(RL) 프레임워크를 제안합니다. 이 프레임워크는 조작 대상의 상호작용 지점과 로봇 자세를 자율적으로 선택하여 선행 설계된 궤적 없이도 객체 조작을 가능하게 합니다. 기존의 접근 방식과 달리, 이 연구는 물체 중심의 조작 계획을 통해 시뮬레이션과 실제 환경에서 평가되었습니다.

- **Technical Details**: 제안된 프레임워크는 비전 기반의 포즈 어포던스 모듈을 사용하여 로봇의 목표 포즈를 생성하고, 이를 내비게이션 정책에 피드백하여 저수준의 보행 정책을 관리합니다. 중층에서 고급 시각 정보를 사용하여 초기 접촉 지점을 식별하고, 침식된 상호작용 포인트를 따라 저수준의 조작 정책을 조정합니다. 이 구조는 프레임워크가 선행 학습된 모델을 플러그 앤 플레이 방식으로 통합할 수 있도록 합니다.

- **Performance Highlights**: 시뮬레이션과 실제 환경에서 평가한 결과, 제안된 프레임워크는 인간의 지침 없이도 후보 포즈를 자율적으로 식별하고 객체 조작 작업을 성공적으로 수행할 수 있음을 나타냈습니다. 다양한 환경에서 로봇은 목표 물체 근처를 탐색하고, 3D 환경에서의 객체 상호작용을 위한 상호작용 데이터셋을 생성했습니다. 이러한 결과는 로봇이 비정형 환경에서도 효과적으로 물체를 탐색하고 조작할 수 있음을 보여줍니다.



### Planning-aligned Token Compression for Long-Context Autonomous Driving (https://arxiv.org/abs/2606.07464)
Comments:
          9 pages

- **What's New**: COMPACT-VA는 planner-aligned working memory 프레임워크로, 균형 잡힌 메모리 효율성을 통해 운전 성능을 최적화합니다. 이 연구는 의사결정에 중요한 정보를 유지하기 위해 과거 궤적과 학습된 계획 의도를 기반으로 하는 압축 메커니즘을 도입했습니다. 현재의 token 압축 방법들이 계획 목표와는 분리되어 있다는 문제점을 해결하기 위한 접근을 제공합니다.

- **Technical Details**: 이 모델은 conditional VQ-VAE(조건부 벡터 양자화 변분 오토인코더)를 기반으로 하며, 절차적 최적화를 통해 중요한 과거 정보를 어떻게 보존할지를 학습합니다. 압축된 메모리는 최근의 토큰에는 더 많은 비율을 주고, 더 먼 과거에 대해서는 적은 비율로 배치됩니다. 이 구조는 self-attention을 통해 과거 궤적 컨텍스트와 미래 운전 의도에 따라 조정이 가능합니다.

- **Performance Highlights**: COMPACT-VA는 일정한 token 예산 하에 68.3%의 성공률을 달성하며, 기존 모델보다 6.3% 향상된 결과를 보입니다. 안전에 중요한 roll-through를 22% 줄였으며, 모델의 전반적인 운전 능력은 유지하면서 3.3배의 속도 향상과 2.7배의 메모리 절감을 이끌어냈습니다. 각 세부 요소들이 성능 향상에 기여했다는 것은 ablation 테스트를 통해 확인되었습니다.



### Re-imagining ISO 26262 in the Age of Autonomous Vehicles: Enhancing Controllability through Transferability and Predictability (https://arxiv.org/abs/2606.07437)
- **What's New**: 이 논문은 ISO 26262 표준의 ‘Controllability’ 개념을 자율주행차(AV)를 위한 두 가지 새로운 차원인 Transferability와 Predictability로 세분화합니다. 이 새로운 개념은 자율주행차의 안전성을 평가하는 데 필요한 기준과 방법론을 제공하여 기존 기준을 보완합니다. Transferability는 AV 시스템이 안전한 방식으로 제어를 이전할 수 있는 능력을 측정하고, Predictability는 외부의 사용자들이 AV의 행동을 얼마나 예측할 수 있는지를 나타냅니다.

- **Technical Details**: ISO 26262 표준은 차량의 기능적 안전성을 보장하기 위해 ‘Severity(심각도)’, ‘Exposure(노출도)’, ‘Controllability(통제 가능성)’라는 리스크 평가를 기반으로 합니다. 이 논문에서는 자율주행차의 환경에서 Controllability를 체계적으로 재해석하고 보완하기 위한 수학적 프레임워크를 제시합니다. Transferability는 고장 허용 시간 내에서의 통제 이전 능력을 측정하는 반면, Predictability는 외부 도로 사용자가 AV의 행동을 얼마나 정확하게 예상할 수 있는지를 수치화합니다.

- **Performance Highlights**: Transferability와 Predictability는 자율주행차의 안전성을 평가하기 위한 SOTIF(ISO 21448)와 조화를 이루는 핵심 성과 지표로 제안됩니다. 이 논문은 Transferability의 최소 위험 조작 성능을 평가하기 위한 접근 방식을 공식화하며, Predictability는 외부 도로 사용자의 안전성을 정량화할 수 있는 새로운 차원으로 설정합니다. 제안된 기준과 검증 방법론은 ISO 26262와 SOTIF 안전 생명주기 프로세스 내에서 채택될 수 있는 신뢰성 있는 메트릭스를 제공합니다.



### Rapid co-design of Buoyancy-assisted robots for Challenging Locomotion using Gaussian Evolutionary Specialists (https://arxiv.org/abs/2606.07424)
Comments:
          Submitted to RA-L

- **What's New**: 이번 연구에서는 로봇 공학에서 형태(차체)와 제어(control)를 공동 최적화하는 것이 중요하다는 점을 강조합니다. 기존의 모델 예측 제어 대신 모델이 없는 강화 학습(Model-free Reinforcement Learning, RL)을 활용하여 보다 강력한 제어기를 개발하는 새로운 접근 방식을 제안하고 있습니다. 제안된 기법인 Gaussian Evolutionary Specialists (GES)는 복잡한 형태 최적화를 위한 새로운 프레임워크로, 표현의 혼합물(Mixture-of-Experts)을 통해 다양한 행동을 효과적으로 캡처하려는 목표를 가지고 있습니다.

- **Technical Details**: GES는 두 단계 구조로 구성되어 있습니다. 첫 번째 단계에서는 각 전문가는 가우스 영역(Gaussian territory)이 할당되어, 이를 바탕으로 훈련된 후 최적의 성능을 보이는 전문 책임자의 경쟁적 배정이 이루어집니다. 두 번째 단계에서는 훈련된 전문가들이 형태 최적화를 위한 제로샷 평가(Zero-shot evaluation) 역할을 하며, 각각의 후보 디자인에 대해 RL 훈련을 대체하게 됩니다.

- **Performance Highlights**: BALLU(Buoyancy-Assisted Light Legged Unit) 플랫폼에서 테스트한 결과, GES는 기존 보편적인 정책보다 5-25% 향상된 성능을 보여주었고, 24cm 높이의 장애물을 넘는 데 있어 3배의 성능 향상을 기록했습니다. 게다가 GES를 통해 설계 최적화 시간이 37% 단축되어, 효율적인 로봇 설계를 가능하게 합니다.



### Simulation-Driven Imitation Learning for Biosignals-Free Shared-Autonomy Prosthetic Grasping (https://arxiv.org/abs/2606.07389)
- **What's New**: 이번 연구는 EMG(표면 전극 근전도)나 다른 생리학적 신호에 의존하지 않고 자연스럽고 효율적인 조작을 가능하게 하는 상호 자율적인 제어 기법을 개발하였습니다. 새로운 시뮬레이션 프레임워크를 통해 다양한 도달-잡기 시연을 자동으로 생성하며, 실제 데이터 수집의 비용과 변동성을 줄이고 있습니다. 이 시스템은 물리적으로 가능한 잡기 구성, 자연스러운 접근 경로를 포함해 시뮬레이션 환경에서 매우 다양하고 풍부한 데이터를 생성합니다.

- **Technical Details**: 제안된 시뮬레이션 프레임워크는 손목에 장착된 가상 카메라를 통해 다양한 시나리오를 자동 생성하는 것이며, 이는 물체의 속성을 이해하고 자율적으로 잡기 행동을 수행하는 것을 목표로 합니다. 훈련 데이터를 생성하기 위해 다양한 손목 경로 및 잡기 구성을 포함하며, 복잡한 실내 환경 속에서 자동으로 데이터를 수집합니다. 이 방법은 비용 높은 실제 데이터 수집의 필요성을 감소시키며, 전문가들이 사용한 데이터를 활용해 강력한 정책 학습을 가능하게 합니다.

- **Performance Highlights**: 시뮬레이션 기반의 학습된 정책은 90% 이상의 잡기 성공률을 보이며, 기존의 기준 방법들을 초과하는 성능을 발휘했습니다. 연구는 1,800회의 실제 실험을 기반으로 한 검증을 통해, 훈련된 정책의 실용성을 확인했습니다. 이러한 결과는 biosignals-free 자율 보철 팔 잡기에 대한 시뮬레이션 기반 학습의 가능성을 강조합니다.



### Spline Policy: A Structured Representation for Robot Policies (https://arxiv.org/abs/2606.07386)
Comments:
          This work has been submitted to the IEEE for possible publication

- **What's New**: 이 논문은 로봇 조작을 위한 현대의 모방 학습 정책에서 스플라인 매개변수(Spline Parameters)를 사용하는 정책 출력 인터페이스를 제안합니다. 스플라인 정책(Spline Policy, SP)은 고정 해상도의 액션 청크(Action Chunks)를 대체하면서 정책의 본체는 변경하지 않습니다. 이 방식은 연속적인 궤적을 컴팩트하게 디코딩하고, 다양한 시간 해상도로 쿼리할 수 있게 하여, 관측 데이터로부터 불확실성을 전파하는 데 도움이 됩니다.

- **Technical Details**: SP는 스플라인을 기초로 하며, 이는 간결한 매개변수화 방식으로 연속적 궤적을 생성할 수 있게 합니다. 특정 관측값에 대해 정책 본체는 디스크리트 액션 시퀀스 대신 스플라인 매개변수를 예측합니다. 예측된 스플라인은 파라미터 공간에서 제약이나 편집이 가능하며, 다운스트림 컨트롤러에 전달될 수 있습니다.

- **Performance Highlights**: 실험 결과, SP는 현대 정책 학습 기법과 호환성을 유지하면서 불확실성 평가, 국소적 수정, 컴팩트 디코딩, 시간 재샘플링 등 유용한 모션 구조 속성을 드러냅니다. 다양한 정책 본체와 함께 SP를 실험하여, 실제 로봇 사례 연구와 동일한 스플라인 출력을 통해 메커니즘 및 배치 수준의 장점을 보여주었습니다.



### RhinoVLA Technical Repor (https://arxiv.org/abs/2606.07383)
- **What's New**: 이번 연구에서는 실시간 로봇 조작을 위한 Vision-Language-Action (VLA) 모델의 배치 지연(latency) 주요 원인인 VLM 시각 및 컨텍스트 토큰을 식별하였습니다. RhinoVLA라는 새로운 배치 지향 VLA 모델을 제안하며, 이는 Huixi R1 엣지 SoC와 함께 설계되었습니다. Qwen3-VL 백본을 채택하여 VLM측 토큰 및 연산 부담을 줄이면서도 다중 모드 기능을 유지하고 있습니다.

- **Technical Details**: RhinoVLA는 하드웨어 인지 컴파일(hardware-aware compilation), 혼합 정밀도 실행(mixed-precision execution), 병렬 시각 인코딩(parallel visual encoding)으로 최적화되어 있습니다. 이 모델은 트레이닝 데이터의 이질성 문제를 해결하기 위해 통합된 인터페이스를 도입하여 다양한 로봇 관찰 및 행동 스키마를 공유 정책하에 정렬할 수 있게 하였습니다.

- **Performance Highlights**: RhinoVLA는 유사한 매개변수 규모에서 π0.5에 비견되는 다운스트림 성능을 달성하며, Huixi R1에서 11.69Hz의 실시간 추론 속도를 기록합니다. 이는 10Hz의 실시간 닫힌 루프 제어 목표를 충족하여, 실제 로봇 애플리케이션에서의 유용성을 입증합니다.



### CAPE: Contrastive Action-conditioned Parallel Encoding for Embodied Planning (https://arxiv.org/abs/2606.07304)
Comments:
          19 pages, 7 figures

- **What's New**: 이번 연구에서는 CAPE(Contrastive Action-conditioned Parallel Encoding)라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 다양한 행동 시퀀스가 유도하는 미래 결과를 구별하여 시각적 동역학(visual dynamics)을 학습합니다. CAPE는 초기 관찰과 행동 시퀀스를 기반으로 미래의 잠재 경로(latent trajectory)를 단일 전진 패스(single forward pass)로 디코딩하는 방식으로 작동합니다.

- **Technical Details**: CAPE는 목표 수렴 대비 대조 목표(Goal-Convergent Contrastive Objective)로 훈련되며, 이는 동일한 미래 결과에 해당하는 예측을 정렬하고 서로 다른 결과를 유도하는 예측은 분리하는 것을 목표로 합니다. 이러한 방법은 행동 조건에 따라 달라지는 전환(action-conditioned transitions)에 초점을 두며, 전체적인 시각 상태를 학습하는 데 소요되는 계산 비용을 크게 줄이는 데 기여합니다.

- **Performance Highlights**: CAPE는 실제 환경에서 DROID 데이터셋과 맞춤형 작업에 대해 RoboCasa에 제로 샷 전이(zero-shot transfer)를 수행하며, 미래 상태 검색(future-state retrieval), 오프라인 행동 매칭(offline action matching), 폐쇄 루프(planning)에서 기존의 기준보다 성능이 우수합니다. 또한 CAPE는 긴 예측 수명(prediction horizons) 동안 계획 시간 추론 비용을 현저히 줄이면서도 뛰어난 효율성을 보여줍니다.



### Beyond Waypoints: A Trajectory-Centric Waypointing Paradigm for Vision-Language Navigation (https://arxiv.org/abs/2606.07244)
- **What's New**: 이 연구에서는 기존의 노드 중심 waypoint 예측을 뛰어넘어 Trajectory Waypoint라는 새로운 패러다임을 제안합니다. 이는 각 후보 waypoint를 실행 가능한 궤적에 기반하여 생성하여, 물리적으로 도달 가능한 목표를 보장합니다. 또한, 이 방법은 고수준 계획과 저수준 실행 간의 일관성을 극대화하는데 기여합니다.

- **Technical Details**: Trajectory Waypoint Predictor는 Diffusion policy로 형성되어 있으며, TSDF 기반의 비용 유도(guidance)를 통해 오프젝트를 피하면서 궤적 생성을 조정합니다. 이로 인해 더욱 안전하고 다양한 항해 가능 궤적을 생성할 수 있고, 이를 통해 에이전트의 탐색 능력을 향상시킵니다. 또한, 궤적 기반 내비게이터는 고급 기하학적 정보와 관련된 궤적을 플래닝 정보로 활용하여 보다 정밀한 방향성을 제공합니다.

- **Performance Highlights**: VLN-CE 벤치마크에서의 실험 결과, Trajectory Waypoint 패러다임이 기존의 waypoint 예측기보다 목표 도달 가능성에서 눈에 띄게 우수한 성능을 보임을 확인했습니다. 전체적으로, 이 프레임워크는 아래의 다양한 VLN-CE 작업에서 성능이 크게 향상되었습니다. 따라서, 기존의 기본선과 비교했을 때 안정적이고 강력한 개선을 나타냅니다.



### Robotic Policy Adaptation via Weight-Space Meta-Learning (https://arxiv.org/abs/2606.07217)
- **What's New**: WIZARD는 언어 지침 및 짧은 시연 비디오만을 사용하여 고정된 VLA 정책에 대한 태스크별 LoRA 파라미터를 생성하는 가중치 공간 메타 학습 프레임워크입니다. WIZARD는 개별 태스크의 액션 레이블이나 테스트 시간 최적화 없이 단 하나의 전파 과정으로 적응 가중치를 예측합니다. 이 방법은 태스크 증거를 전문가 LoRA 업데이트에 직접 매핑하며, 태스크 간의 관계를 가중치 공간에서 캡처합니다.

- **Technical Details**: WIZARD의 메타 네트워크는 함수형으로 구축되어 태스크 증거(z)에서 전문가 정책의 LoRA 업데이트(ΔW)로의 직접적인 변환을 수행합니다. 이는 전통적인 테스트 시간 적응과 다르며, 액션 공간에서 최적화하는 데 의존하지 않습니다. WIZARD는 학습된 태스크 증거에 바탕을 두어 가중치 공간에서 태스크 관계를 학습하면서, 고정된 VLA 정책에 특화된 태스크 별 적응 메커니즘을 제공합니다.

- **Performance Highlights**: LIBERO 데이터 세트에서 WIZARD는 아직 훈련되지 않은 데이터 세트에서 평균적으로 약 2배 그리고 새로운 태스크에서 최대 14배의 성능 향상을 보였습니다. 이는 사전 훈련된 VLA와 태스크별 전문가 간의 성능 차이를 줄이는 데 큰 기여를 하며, WIZARD로 생성된 어댑터가 시뮬레이션 이상의 태스크 수준 특성을 제공함을 보여줍니다.



### An Abstract Architecture for Explainable Autonomy in Hazardous Environments (https://arxiv.org/abs/2606.07211)
Comments:
          Originally published 20th of October 2022 at the Second International Workshop on Requirements Engineering for Explainable Systems (RE4ES), which was hosted by the International Requirements Engineering Conference 2022

- **What's New**: 이 논문에서는 자율 로봇 시스템이 위험한 환경에서 사용되도록 설계되고 있음을 강조합니다. 특히, 사용자가 시스템에 대한 신뢰를 가질 수 있도록 하는 것이 중요하다고 주장합니다. 설명 가능성(explainability)이 시스템의 신뢰성과 밀접한 관계가 있으며, 이러한 설명 가능성을 시스템의 설계에 내재화해야 한다고 제안합니다.

- **Technical Details**: 제안된 아키텍처는 고수준과 저수준의 의사결정을 분리합니다. 고수준 결정을 위해 Belief-Desire-Intention (BDI) 에이전트를 사용하며, 이 에이전트는 환경 정보를 해석하고 논리적인 진술로 바꾸어 계획을 선택합니다. 이 아키텍처는 시스템의 각 행동에 대한 과거 설명과 미래 행동에 대한 설명을 제공하는 두 가지 모드를 지원합니다.

- **Performance Highlights**: 이 아키텍처가 제공하는 설명 가능성의 두 가지 주요 모드는 사용자와 규제 기관 모두에게 시스템의 신뢰도를 높이는 데 기여할 수 있습니다. 과거 행동에 대한 설명은 결정의 근거를 제공하고, 미래 행동에 대한 설명은 시스템의 안전성을 검증합니다. 이러한 기능들은 특히 핵산업과 같은 안전-critical한 환경에서 중요한 역할을 합니다.



### Shield-Loco: Shielding Locomotion Policies with Predictive Safety Filtering (https://arxiv.org/abs/2606.07193)
- **What's New**: 본 논문은 RL (Reinforcement Learning) 정책을 위한 사전 예측 안전 필터(preemptive safety filter)를 제안하여 발달된 다리 기반 로봇의 안전성을 향상시킵니다. 이 필터는 RL 정책에 제공되는 접촉 위치를 후처리하여, 예상되는 충돌 시 더 안전한 접촉 시퀀스를 찾기 위해 샘플링 기반 최적화(sampling-based optimization)를 사용합니다. 피지컬 모델을 활용하여 전체 신체 동작을 고려하며, 접촉 위치를 안전 제약 조건을 적용하기 위한 유용한 기계적 추상화(kinematic abstraction)로 사용합니다.

- **Technical Details**: 제안된 방법은 샘플링된 접촉 위치를 기하학적으로 투영하는 기법, 모멘텀 보강 업데이트(momentum-augmented updates), 그리고 레플리카 교환 전략(replica exchange strategy)을 포함하여 최적화를 용이하게 합니다. 이러한 구성 요소들은 일관된 접촉 환경에서 비연속적인 목표 제어를 효과적으로 처리할 수 있도록 하며, RL 정책의 수정 없이도 전신 충돌 회피를 가능하게 합니다. 로봇의 접촉 기능에 대한 비선형 제어 이론을 바탕으로, 적절한 안전성을 보장합니다.

- **Performance Highlights**: 시뮬레이션과 실제 상황에서 사륜 로봇을 통해 검증한 결과, 제안한 안전 필터를 통해 안전 위반이 크게 줄어드는 효과를 확인할 수 있었습니다. 특히, 충돌이 빈번한 복잡한 환경에서도 안정적인 성능을 보이며, 접촉 계획 변경에 대한 부정적인 영향을 최소화하는 데 성공했습니다. 이는 접촉 위치의 최적화가 높은 안전성을 유지하면서도 RL 정책의 임무 성능을 개선할 수 있음을 보여줍니다.



### A Causal Probabilistic Framework for Perception-Informed Closed-Loop Simulation of Autonomous Driving (https://arxiv.org/abs/2606.07186)
- **What's New**: 본 논문에서는 기존의 이상적인 센서 모델을 사용하는 소프트웨어 인 더 루프(SIL) 시뮬레이션의 한계를 극복하기 위해 현실적인 인식 오류를 체계적으로 주입하는 새로운 실험 방법론을 제안합니다. 기존 SIL 환경에서는 인식 알고리즘의 성능 부족을 무시하고 지나치게 낙관적인 안전성 평가를 하였으나, 본 연구는 인식 기반 SIL 테스트 프레임워크를 통해 이러한 문제를 해결합니다.

- **Technical Details**: 제안된 프레임워크는 베이지안 네트워크(Bayesian Networks)를 이용하여 카메라 센서의 내재적 결함 모드(예: 탐지 손실, 크기 오류, 위치 오류)를 환경적 요인과 연결합니다. 이 방법론은 물리적 트리거 조건에 따라 인식 부족이 차량의 제어 시스템과 행동에 미치는 영향을 평가할 수 있는 고차원 시뮬레이션 도구체인 내에서 구현됩니다.

- **Performance Highlights**: 이 연구는 새로운 인식 기반 테스트가 전통적인 SIL 환경에서 발견되지 않는 잠재적인 운영 위험을 드러내며, ISO 21448(SOTIF) 유효성을 입증하기 위한 확장 가능한 경로를 제시함을 보여줍니다. 안전-critical한 시나리오를 통해 이 프레임워크가 결정을 내리는 과정에서의 결함을 식별하는 데 효과적임을 입증하였습니다.



### Test-Time Trajectory Optimization for Autonomous Driving (https://arxiv.org/abs/2606.07170)
- **What's New**: 이 논문에서는 자율주행을 위한 새로운 방법 TOAD( Trajectory Optimization at Autonomous Driving)를 소개합니다. 기존의 방법은 후보 궤적 세트를 생성하고 이를 채점한 후 가장 높은 점수를 받은 궤적을 선택했습니다. TOAD는 채점기를 활용하여 이 과정을 최적화하며, 별도의 재훈련 없이 기존의 플래너에 적용할 수 있는 장점을 가집니다.

- **Technical Details**: TOAD는 테스트 시기 동안 Cross-Entropy Method (CEM)를 활용하여 후보 궤적을 검색합니다. 이 방법은 고정된 상태의 채점기를 활용하며, 후보 궤적에 대한 Gaussian 분포에서 샘플링하고 최상위 후보들의 분포를 재조정합니다. 키 포인트는 모든 훈련된 채점기가 보상으로 작용할 수 있는 것은 아니라는 점이며, 훈련 제안 외의 궤적을 탐색할 때 정확도를 유지하는 것이 중요한 요소입니다.

- **Performance Highlights**: TOAD는 기존의 여섯 개 플래너에 적용되어 NAVSIM-v1 및 NAVSIM-v2에서 성능 향상을 보여주었습니다. 특히, DrivoR와 TOAD 조합에서 56.3 EPDMS를 기록하며 새로운 성과를 달성했습니다. TOAD의 성과는 단순한 재순위보다 훨씬 더 효과적이며, 기존의 성능 한계를 극복할 수 있는 가능성을 보여줍니다.



### QuadVerse: An Integrated Framework Aligning Visual-Physical Reality for Quadruped Simulation (https://arxiv.org/abs/2606.07118)
- **What's New**: 새로운 연구는 로봇 학습에서 시뮬레이션의 중요성을 강조하지만, 여전히 시뮬레이션과 현실 간의 간극(sim-to-real gap)이 큰 문제로 남아 있다. 이 논문에서는 QuadVerse를 소개하며, 이는 시각적 인식, 물리적 상호작용, 액추에이터 동력을 정렬하기 위해 재구성된 장면을 사용하는 통합 프레임워크이다. QuadVerse는 고품질 RGB 비디오를 통해 복원된 3D Gaussian Splatting(3DGS) 장면을 사용하여 사실적인 렌더링과 충돌 준비가 된 시멘틱 메쉬 추출을 지원한다.

- **Technical Details**: QuadVerse는 3DGS를 활용하여 고해상도의 사실적인 환경을 구축하고 액추에이터의 비일관성을 줄이기 위해 잔여 동역학 보상기를 훈련한다. 이러한 접근 방식은 현장 경험을 바탕으로 한 평균 포지션 오류를 줄이고, 조인트 스페이스 재생 정확도를 개선하며, 혼합 마찰 지형에서의 내비게이션에서 성공적인 성능을 보여준다. QuadVerse는 전송 과정의 모듈 수준의 충실성과 시스템 수준의 sim-to-real 이전을 평가하는 실험을 통해 검증되었다.

- **Performance Highlights**: QuadVerse의 통합 외부 내비게이션 작업에서 84%의 성공률을 기록하며, 이는 정렬된 시뮬레이션에서의 92%와 비교됐다. 이 결과는 QuadVerse에서 훈련된 정책을 사용하여 작업 특화된 현실 세계 시행 없이도 이룰 수 있었다. 또한, QuadVerse의 재구성 파이프라인은 640x480 해상도에서 2000 FPS 이상의 높은 처리능력으로 RL 훈련을 지원하며, 사실적인 지오메트리와 높은 충실도를 보장한다.



### Coarse-to-Control: Action-Token Planning for Vision-Language-Action Models (https://arxiv.org/abs/2606.07107)
- **What's New**: 이 논문은 Vision-Language-Action (VLA) 모델의 한계를 극복하기 위해 Coarse-to-Control을 제안합니다. 기존의 VLA 모델들은 관찰을 직접적으로 행동으로 매핑하여, 중간 계획 없이 작동하는 방식에서 기인한 성능 제한을 경험하고 있었습니다. Coarse-to-Control은 액션 토큰 공간에서 원활한 계획을 도입하며, 이를 통해 장기적인 작업에서 더 나은 성과를 달성할 수 있는 방법을 제시합니다.

- **Technical Details**: Coarse-to-Control은 정책이 미래 경로를 요약한 조잡한 액션 토큰 시퀀스를 예측한 후, 이 계획에 따라 실행 가능한 액션 토큰을 생성하는 계획-실행 VLA입니다. 이 접근 방식은 플래닝(planning)과 실행(execution) 간의 단일 분리된 디스크리트 액션 어휘를 공유함으로써 실행 가능성이 높은 지침을 제공합니다. 훈련 과정에서, 조잡한 플래닝 토큰이 내부 접두사로 생성되며, 실행 가능한 토큰은 이 접두사에 조건부로 생성됩니다.

- **Performance Highlights**: Coarse-to-Control은 LIBERO에서 97.90%의 평균 성공률을 기록하며, SimplerEnv-WidowX에서는 83.3%의 성공률을 보였습니다. 또한 네 개의 실제 조작 작업에서 강건성을 개선하며, 조잡한 액션 토큰 플래닝이 직접적인 액션 생성보다 지속적으로 우수한 결과를 도출함을 입증하였습니다. 이 연구는 조작 작업의 성격과 복잡성을 반영한 액션 생성 방식에 중요한 기여를 합니다.



### Dreaming when Necessary: Advancing World Action Models with Adaptive Multi-Modal Reasoning (https://arxiv.org/abs/2606.07089)
- **What's New**: 이번 논문에서는 AdaWAM이라는 새로운 World Action Model을 제안합니다. AdaWAM은 작업 실행 중 텍스트적 또는 시각적 추론을 필요에 따라 autonomously(자율적으로) 활성화하는 동적 라우터를 통합하여 적응형 다중모드 추론이 가능하게 합니다. 이는 기존 모델들이 바탕으로 삼았던 비디오 예측에 의존하지 않고도 더 효과적으로 작동하는 것을 목표로 합니다.

- **Technical Details**: AdaWAM의 아키텍처는 Generative Diffusion Backbone과 Semantic Specialist, Adaptive Routing Mechanism의 세 가지 핵심 요소로 구성됩니다. Generative Diffusion Backbone은 환경 역학을 포착하기 위한 Diffusion Transformer(디퓨전 트랜스포머) 모델을 사용하며, 정책 실행을 위한 ActionDiT와 실시간 시각적 추론을 위한 Text Reasoning Module을 포함하고 있습니다. 이 구조는 다양한 작업 맥락에 맞추어 적절한 추론 모드를 선택할 수 있도록 설계되었습니다.

- **Performance Highlights**: AdaWAM은 LIBERO, RoboTwin 2.0 및 ALOHA/PiPER와 같은 다양한 실험에서 기존 최신 모델들보다 우수한 성능을 보여주었습니다. 특히, AdaWAM은 긴 호리즌(long-horizon) 및 세밀한 조작(fine-grained manipulation) 작업에서 두드러진 성과를 달성하였으며, 실세계 작업에서의 능력 또한 이전의 WAM 및 VLA 방법들보다 뛰어난 결과를 입증했습니다.



### Predictive Style Matching: Natural and Robust Humanoid Locomotion (https://arxiv.org/abs/2606.07083)
- **What's New**: 본 논문에서는 Predictive Style Matching (PSM)을 제안합니다. PSM은 로봇의 하체 상태 이력과 속도 명령을 해석 가능한 상체 관절 및 보행 목표로 매핑하는 오프라인 예측기를 포함합니다. 이를 통해 로보틱스 훈련 시 자연스러움을 극대화하면서도 배포 시 작업 중심의 RL (Reinforcement Learning) 성능을 유지할 수 있습니다.

- **Technical Details**: PSM의 구조는 두 단계로 나뉩니다. 첫 번째 단계에서는 하체 상태와 명령을 기반으로 해석 가능한 상체 목표를 생성하는 오프라인 예측기 fᵩ를 훈련합니다. 두 번째 단계에서는 이 예측기의 출력을 PPO (Proximal Policy Optimization) 훈련 중 보상 항으로 사용하여 정책을 훈련합니다.

- **Performance Highlights**: PSM을 사용한 Unitree G1에서의 평가 결과는 일반적인 행동을 시뮬레이션 및 하드웨어에서 더욱 효율적으로 처리하며, 스타일 에러를 약 10배 감소시키는 것으로 나타났습니다. 그에 비해 모션 모방 기반 방법은 스타일 에러 측면에서 우수하지만, 외부 방해에 대한 복원 능력이 떨어집니다.



### Extending Responsibility-Sensitive Safety for the Assessment of Offloaded Autonomous Driving Services (https://arxiv.org/abs/2606.07067)
Comments:
          8 pages; accepted for 2026 IEEE 29th International Conference on Intelligent Transportation Systems (ITSC), Naples, Italy, September 15-18, 2026 - DOI will be added after publication

- **What's New**: 이 논문에서는 자율주행 시스템의 안전성을 위한 새로운 접근법을 제시합니다. 특히, 기능 오프로드(function offloading)가 안전-critical한 자율주행 기능에 미치는 영향을 고려하여 응답 시간의 변동성을 관리하는 방안을 다룹니다. 책임 민감 안전(Responsibility-Sensitive Safety, RSS)의 정의를 확장하여 지역(local) 및 오프로드 서비스 조합의 응답 시간을 명시적으로 반영하는 방법을 소개합니다.

- **Technical Details**: 연구에서는 오프로드 서비스 조합의 안전성을 보장하기 위해 RSS 안전 제약조건을 사용한 오프로드 의사 결정 및 복귀(fallback) 메커니즘을 통합합니다. 현재 교통 상황이 해당 엔드투엔드(end-to-end) 응답 시간에 따라 안전하다고 판단될 때만 오프로드가 허용됩니다. 이 조건이 위반되면 시스템은 지역 실행으로의 제어된 복귀를 수행하고, 오프로드 서비스에 대해 워밍 대기(warm-standby) 단계가 포함된 향상된 복귀 전략도 도입됩니다.

- **Performance Highlights**: 제안된 방법은 시뮬레이션과 실제 환경 모두에서 평가되었습니다. 실험 결과는 새로운 접근법이 기존의 기능 오프로드 및 안전 프레임워크와 비교하여 안전성을 향상시키는 동시에, 안전 조건이 허용할 경우 분산 컴퓨테이션의 이점도 유지한다는 것을 보여줍니다. 이는 자율주행 시스템의 안전성 보장을 위해 큰 기여를 할 것으로 기대됩니다.



### A Multi-Operator Mixed-Reality Interface for Multi-Robot Control and Coordination: Co-Located and Private Workspace Collaboration (https://arxiv.org/abs/2606.07013)
Comments:
          Submitted to RO-MAN 2026

- **What's New**: 본 논문에서는 이전의 HORUS 인터페이스를 기반으로 하여 다중 운영자가 협력하여 로봇 팀을 제어할 수 있는 혼합 현실(mixed-reality) 인터페이스를 제안합니다. 이 시스템은 코로케이티드(co-located) 공유 작업 공간과 개인 작업 공간 두 가지 모드를 지원하여, 동일한 미니 맵을 관찰하거나 독립적으로 배치된 작업 공간에서 작업할 수 있습니다. 주요 기여로는 공유 상태 동기화와 협력 작업을 위한 명시적 기구를 포함하며, 혼합 현실에서의 다중 사용자 환경을 위한 확장성을 제공합니다.

- **Technical Details**: 이 시스템은 로봇 등록 메타데이터, 지도, 센서 및 작업 가능성을 전달하는 SDK 계층, 명령 및 상태를 처리하는 ROS 2 브리지, 그리고 혼합 현실 런타임을 포함하는 3계층 아키텍처를 따릅니다. 모든 로봇 및 작업 관리는 동일한 작업 공간을 중심으로 이루어지며, 로봇 상태 전송과 다중 사용자 동기화를 분리하여 처리합니다. 또한 각 로봇에 대한 새로운 작업 권한과 모드 변경을 가능하게 하는 전용 로봇 관리자 패널이 제공됩니다.

- **Performance Highlights**: 3명의 이동 로봇을 제어하고 두 개의 검색 환경에서 운영한 사람 대상 연구를 통해, 두 가지 모드 모두에서 목표 작업 수행이 동일하게 이루어졌습니다. 그러나 코로케이티드 공유 작업 공간 모드는 협력의 인식, 공동 이해 및 인수 명확성을 크게 향상시켰으며, 선호되는 협력 모드로 나타났습니다. 젊은 사용자가 이 시스템을 통해 실제 상황에서도 효과적인 작업 수행을 할 수 있다는 결과가 나왔습니다.



### Task Editing for Generalizable 3D Visuomotor Policy Learning (https://arxiv.org/abs/2606.07012)
Comments:
          8 pages, 4 figures

- **What's New**: 이번 연구는 Task-Edit라는 새로운 데이터 생성 프레임워크를 제안하여, 작업 중심의 편집 관점에서 다양한 경로를 생성할 수 있는 방법을 소개합니다. 기존의 방법들은 객체 중심 변환을 이용하여 데이터의 질과 효율성을 향상시켰으나, 일반화 능력을 제한하는 고정된 행동 패턴을 가지고 있었습니다. Task-Edit는 장면, 기술, 객체라는 세 가지 핵심 요소로 작업을 분해하고 이를 유연하게 재조합함으로써, 복잡한 로봇 조작 작업에 대한 데이터 다양성을 향상시킵니다.

- **Technical Details**: Task-Edit는 작업을 장면, 기술, 객체로 독립적으로 편집하여 데이터 다양성을 증가시키는 접근 방식을 채택하고 있습니다. 이를 통해 한 번의 인간 시연에서 여러 가지 훈련 경로를 생성할 수 있으며, 예를 들어 요리 작업으로부터 여러 요리 설정과 기술 순서 변화를 가능하게 합니다. 또한, Sim2Real에서의 시뮬레이션 데이터를 효과적으로 수집하기 위해 깊이 복구 모듈을 도입하여 시뮬레이션과 현실 간의 격차를 줄이고 있습니다.

- **Performance Highlights**: 실제 실험을 통해 Task-Edit는 여러 가지 장점이 있음을 입증했습니다. 첫째, 다양한 현실 세계 조작 작업에서 3D visuomotor 정책의 성능을 크게 개선했습니다. 둘째, 새로운 장면 설정에서도 모델의 일반화 능력을 향상시켰습니다. 셋째, 복잡한 수집이 어려운 환경에서도 적용 가능성을 높이는 데 기여하였습니다.



### Mission-Level Runtime Assurance Framework for Autonomous Driving (https://arxiv.org/abs/2606.06996)
- **What's New**: 이 논문은 자율주행 시스템에서 높은 수준의 운전 명령이 고장나거나 신뢰할 수 없을 때의 런타임 안전성을 연구합니다. 제안된 프레임워크는 차량이 명령을 실행하기 전에 운전 안전성과 임무를 성공적으로 완료할 수 있는지를 평가합니다. 이는 기존의 런타임 안전 접근법이 즉각적인 차량 안전에만 초점을 맞춘 것과 대조적입니다.

- **Technical Details**: 프레임워크는 highway-env를 확장하여 체크포인트 건너뛰기, 제한 구역 진입 및 임무 완료 불가능한 미래 경로 생성과 같은 임무 수준의 고장 시나리오를 포함합니다. 런타임 모니터링 시스템이 도입되어 안전하지 않거나 임무를 수행할 수 없는 명령을 실행 전에 탐지하고 거부합니다. 제안된 방법론은 플랫폼 수준의 런타임 안전성과 임무 수준의 장애를 동시에 평가하는 계층적 구조로 되어 있습니다.

- **Performance Highlights**: 실험 결과, 기존 플랫폼 수준의 런타임 안전성만으로는 임무 수준의 계획 결함을 탐지할 수 없음을 보여주었습니다. 반면, 제안된 프레임워크는 임무를 수행할 수 없는 명령을 성공적으로 거부하고 조건이 무작위로 변화할 때 임무 성공률을 높였습니다. 이를 통해 차량이 안전하면서도 임무 수행에 실패하지 않도록 해야 함을 확인했습니다.



### Compliance-Based Sensor Placement for Force Sensing on a Sensorized Prostate Phantom (https://arxiv.org/abs/2606.06977)
- **What's New**: 이 연구는 디지털 직장 검사(Digital Rectal Examination, DRE) 훈련을 위한 센서화된 전립선 팬텀에 대한 힘 감지(compliance-based sensor placement) 방법을 제시합니다. 팬텀은 고유 압력 센서로 사용되는 3개의 내부 공압 챔버와 10개의 표면 변위 마커로 구성되어 있습니다. 이 방법은 임상적으로 관련 있는 후면(contact region) 접촉 영역을 우선시하여 센서 배치 정확도를 극대화합니다.

- **Technical Details**: 이 방법은 유한 요소 시뮬레이션(finite-element simulation)을 통해 생성된 compliance matrix를 기반으로 합니다. 저자들은 무게를 두는 탐색 전략을 제안하며, 이를 통해 최소 거리 제약을 따르면서 원하는 로컬(force) 재구성 가능성을 최적화합니다. 캠퍼(Chamber) 압력과 표면 변위를 포함하는 센서 출력 변화와 입력 힘 변화 간의 관계를 모델링하여, 각 표면 노드를 후보 센서 위치로 취급합니다.

- **Performance Highlights**: 제안된 방법은 기존의 QR 기반 배치 전략과 비교하여 목표 지역에서 평균 재구성 가능성 점수를 22.5% 향상시킵니다. 선정된 구성은 개발된 메커니즘에 반영되며, ROI에서 재구성 가능성이 특히 높아지는 것을 보여줍니다. 향후에는 하드웨어에서의 배치 검증 및 힘(localization) 감지 같은 다운스트림(downstream) 작업에서의 성과를 평가할 계획입니다.



### LIMMT: Less is More for Motion Tracking (https://arxiv.org/abs/2606.06953)
Comments:
          Accepted at ICML 2026

- **What's New**: 이 논문에서는 LIMMT(모션 추적을 위한 Less Is More)를 소개하며, 물리 기반의 휴머노이드 모션 추적을 위한 최초의 데이터 중심 연구라고 주장합니다. 기존의 저품질 클립 제거를 넘어, 세 가지 차원(물리적 실행 가능성, 다양성 및 복잡성)을 통해 모션 데이터 품질을 정의합니다. 연구에서는 AMASS 데이터셋의 3%만 사용하여도 전체 데이터셋을 사용하는 것보다 더 나은 성능을 나타낸다고 밝혔습니다.

- **Technical Details**: LIMMT는 General Quality Selection(GQS)라는 세 단계의 파이프라인을 통해 물리적 실행 가능성 필터링, 의미론적 모션 임베딩, 다양성 인지 및 복잡성 가중 하위 집합 선택을 수행합니다. GQS의 첫 번째 단계에서는 시뮬레이터에서 후보 모션을 재생하며 물리적으로 해석 가능한 지표에 따라 종합 실행 가능성 점수를 계산합니다. 이후 남은 모션에 대해서는 Harmonic Motion Embedding(HME)을 통해 행동의 구조적 유사성을 포착하며, 마지막 단계에서는 물리적 모션 강도를 기준으로 복잡성 점수를 계산하여 하위 집합을 선택합니다.

- **Performance Highlights**: LIMMT는 여러 추적 시스템에서 강력한 'less-is-more' 효과를 입증하였으며, 일부 설정에서는 AMASS의 3%로 훈련한 것이 모든 평가 지표에서 전체 데이터셋을 사용하는 것보다 나은 성능을 보였습니다. GQS는 다양한 추적기에서 플러그 앤 플레이 방식으로 성능 향상을 보여주며, 각 차원의 기여를 고립시키고 물리적 실패 모드가 정책 학습에 미치는 영향을 정량화하는 광범위한 실험과 분석 결과를 제공합니다.



### T-GMP: Terrain-conditioned Generative Motion Priors for Versatile and Natural Humanoid Locomotion (https://arxiv.org/abs/2606.06944)
- **What's New**: 이번 연구에서 제안한 Terrain-conditioned Generative Motion Priors(T-GMP)는 정적인 모션 우선순위 대신 다양한 환경에 적응할 수 있는 동적인 모션 매니폴드를 학습하여, 인류형 로봇의 안정적인 이동을 목표로 하고 있습니다. 이 모듈은 Conditional Variational Autoencoder(CVAE)를 이용하여, 전문가의 상태-지형 시연 데이터를 바탕으로 지형에 조건화된 잠재 모션 매니폴드를 캡처합니다. 또한, 자연스러움 제약조건을 동적으로 조절하는 디스크리미네이터를 통합하여 인간처럼 보이는 다양하고 유연한 동작 생성을 지도합니다.

- **Technical Details**: 이 연구는 경사면, 계단과 같은 복잡한 지형에서의 안정적인 이동을 위해 고급 정책 데이터와 인간의 MoCap 데이터를 결합하여 지형-조건의 모션 데이터셋을 구축하였습니다. 아울러, CVAE를 활용하여 지형 매핑을 명시적으로 포함시키고, 이를 통해 다양한 모션 스타일을 모델링하는 조건부 β-VAE 방식으로 확장합니다. 이 과정을 통해, 로봇은 각기 다른 지형에서 전문가 상태 완료의 차이를 학습하고, 유연한 스타일 전환이 가능한 동작을 생성할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존의 기준선보다 이동 성공률과 모션 부드러움 면에서 우수한 성과를 보였습니다. 다양한 지형 위에서도 조정되고 인간적인 전체 신체 행동을 생성할 수 있음을 보여주어, 특히 비침습적인 행동에서의 유연성이 증가했습니다. 이러한 성과는 로봇이 어려운 지면 위에서도 성공적으로 기능할 수 있는 가능성을 제시합니다.



### ActionMap: Robot Policy Learning via Voxel Action Heatmap (https://arxiv.org/abs/2606.06904)
- **What's New**: 최근 Vision-language-action (VLA) 모델 분야에서 ActionMap을 도입하였습니다. 이는 기존 VLA의 액션 디코더를 대체할 수 있는 voxel heatmap action head로, 각 액션에 대해 액션 공간의 voxel heatmap을 예측합니다. 이러한 접근은 인접한 액션 간의 기하학적 근접성을 활용하여 데이터 효율성과 수렴 속도를 향상시킬 수 있는 가능성을 제공합니다.

- **Technical Details**: ActionMap은 각 액션에 대해 3D 번역, 3D 회전 및 그리퍼를 포함한 세 가지 voxel heatmap을 예측하며, 각 voxel은 해당 액션의 확률을 저장합니다. 이러한 heatmap은 교차 엔트로피(cross-entropy)를 사용하여 중심이 실제 액션인 Gaussian blob에 대해 훈련됩니다. 추론 시 각 heatmap에서 연속 액션을 하드 argmax(hard argmax) 또는 top-kk soft argmax를 통해 회복할 수 있습니다.

- **Performance Highlights**: LIBERO 시뮬레이션 및 실제 Franka 조작 실험에서, 우리의 heatmap 헤드는 OpenVLA-OFT의 L1 회귀 헤드보다 평균 8.2% 향상되었습니다. 또한, 낮은 훈련 데이터에서 데이터 효율성이 크게 증가하였으며, 수렴 속도에서도 일관되게 더 빠른 성능을 보였습니다. 이러한 결과는 액션 표현이 VLA 성능의 중요한 레버라는 것을 보여줍니다.



### A Cross-view Fusion Framework for Robust 6-DoF Grasp Pose Estimation (https://arxiv.org/abs/2606.06878)
Comments:
          Corresponding author: Jin Xie

- **What's New**: 이 논문에서는 코너 뷰에서 6-DoF 그립 포즈 추정을 향상시키기 위한 크로스 뷰 융합 프레임워크를 제안합니다. 우리는 보조 뷰를 포함시킴으로써 가리낌(occlusion) 문제를 완화하고, 시간을 많이 소모하는 멀티 뷰 재구성 과정을 피하는 포스트 융합(post-fusion) 전략을 채택하였습니다. 이 방법은 그립에 관련된 지오메트리를 포괄적으로 표현하기 위해 크로스 뷰 포인트 쌍을 이용한 자기 지도 대비 학습(self-supervised contrastive learning) 전략을 사용합니다.

- **Technical Details**: 제안된 프레임워크는 크로스 뷰 점 인식 향상을 위해 세 가지 주요 컴포넌트를 포함합니다. 첫째, 보조 뷰를 활용하여 기하학적 정보를 보강하고, 아울러 원통형 좌표계에서 포인트를 정렬하여 잡기의 방향성을 강조합니다. 둘째, 자기 지도 대비 학습 전략을 통해 포인트 특성을 정규화하여, 공간적 일관성을 유지하고 방향적인 구별성을 강화합니다. 마지막으로, 로컬 셀프 어텐션과 시드 크로스 어텐션을 차례로 사용하여, 단일 뷰 내에서의 상호작용과 뷰 간의 상호작용을 지원하여 정밀한 잡기 형상을 생성합니다.

- **Performance Highlights**: 본 프레임워크는 GraspNet-1Billion 벤치마크와 실제 응용에서 강력한 성능을 보여줍니다. 특히, 보조 뷰를 사용하여 코너 뷰의 가리낌 문제를 해결함으로써 6-DoF 그립 포즈 추정의 정확성을 높였습니다. 실시간 로봇 작업 흐름에서 높은 해상도의 포인트 표현을 유지하면서 기하학적 세부정보의 손실을 방지하여, 보다 견고한 잡기 성능을 달성하였습니다.



### Neuro-Symbolic Learning for Long-Horizon Task Planning Under Complex Logical Constraints (https://arxiv.org/abs/2606.06877)
- **What's New**: 이번 논문에서는 로봇의 작업 계획(task planning) 효율성을 개선하기 위해 객체 중요도(object importance)를 학습하는 새로운 두 단계 최적화(bilevel optimization) 문제 모델을 제시합니다. 상위 단계에서는 신경망 점수 생성기(neural scorer)를 최적화하고 하위 단계에서는 점수에 의해 가지치기된 검색 공간에서 기호적 계획(symbolic planning) 문제를 해결합니다. 이러한 접근은 과거 방식에서 발생한 학습-테스트 불일치(train-test mismatch) 문제를 해결하고 플래너가 신뢰할 수 있는 피드백을 제공하도록 돕습니다.

- **Technical Details**: 제안된 프레임워크는 3R(Repair, Restart, Rollback) 전략을 포함하여 초기 단계의 불안정한 학습 과정에서도 하위 계획에서 신뢰할 수 있는 피드백을 제공합니다. 이를 통해 플래너는 불확실한 가지치기된 검색 공간에서 실질적인 계획을 찾을 수 있으며, 상위 학습의 효율성을 높일 수 있습니다. 이 기법은 PDDL(Planning Domain Definition Language)의 정의를 기반으로 하여 복잡한 논리적 제약(logical constraints)을 처리하는 데 효과적입니다.

- **Performance Highlights**: 실험 결과, MazeNamo, SokoMindPlus, LogisticsPlus의 세 가지 벤치마크에서 SOTA 성능을 달성하였으며, 실패율(failure rate)은 80.04% 감소하고 계획 시간(planning time)은 57.14% 줄어드는 결과를 보였습니다. 또한, 사족 기반 이동 조작기에 대한 시뮬레이션 및 실세계 검증을 통해 제안된 방법의 효율성 및 배포 가능성을 입증했습니다.



### What Is My Robot Thinking? Design Considerations for Transparent and Trustworthy Shared Autonomy (https://arxiv.org/abs/2606.06870)
Comments:
          9 pages, 5 Figures, Code and videos are available at this https URL. Under review at IROS 2026

- **What's New**: 이번 연구에서는 사용자가 원하는 목표와 로봇의 추정된 목표 간의 불일치를 줄이기 위해 피드백 모달리티(visual vs. auditory)와 정보의 풍부함(sparse vs. rich)이 상호작용에 미치는 영향을 조사했습니다. 25명의 참가자를 대상으로 실시한 사용자 연구 결과, 피드백 제공이 의도 정렬(intent alignment)을 크게 개선하고 수정 개입의 필요성을 줄인 것으로 나타났습니다. 이러한 결과는 투명한 피드백이 혼합 자율 시스템에서 효과적임을 보여줍니다.

- **Technical Details**: 이 연구는 비전 기반(shared autonomy) 시스템 내에서 피드백 인터페이스가 투명성과 신뢰에 미치는 영향을 실험적으로 평가합니다. 모드는 비주얼(visual)과 오디오(auditory) 형태로 제공되며, 정보의 풍부함은 희소(sparse) 또는 풍부(rich)한 형태로 설정됩니다. 피드백의 종류를 변경하면서 사용자의 상호작용 방식에 미치는 영향을 분석하고, 감지 및 의사소통의 다양한 차원에서 발생하는 이점을 규명합니다.

- **Performance Highlights**: 참가자들은 비주얼 피드백을 선호했으며 희소한 정보와 풍부한 정보에 대한 선호는 작업의 복잡성에 따라 달라졌습니다. 최대한의 정보를 노출하는 것이 신뢰와 의도 정렬에 항상 긍정적이지 않다는 것도 발견되었습니다. 이는 투명성이 특정 작업에 적합한 정보 노출에 의존한 것으로, 설계 가이드라인을 통해 혼합 자율 시스템의 개선 방향을 제시한 것입니다.



### Think Like a Pilot: Fine-Grained Long-Horizon UAV Navigation (https://arxiv.org/abs/2606.06836)
- **What's New**: 이번 논문에서는 UAV(무인항공기) 에이전트가 긴 지평선의 의미 기반 지시를 실행하고 부드럽고 물리적으로 실행 가능한 연속 비행 명령을 생성할 수 있도록 하는 새로운 벤치마크인 FLIGHT를 소개합니다. 기존의 Vision-Language Navigation (VLN) 벤치마크는 일반적으로 이산적 또는 조잡한 행동을 사용하며, UAV Vision-Language-Action (VLA) 작업은 짧고 원자적인 조작에 집중하고 있었습니다. FLIGHT는 세밀한 VLN과 긴 지평선 흐름으로 나누어진 두 개의 데이터셋에서 다단계 지침과 밀집된 6-DoF 궤적 주석을 결합하고 있습니다.

- **Technical Details**: FLIGHT 시스템은 UAV 에이전트가 과제 실행 상태와 미션 계획에 대한 실시간 비행 중 추론을 가능하게 하며, 동시에 고주파의 정확한 제어를 수용하기 위해 FLIGHT VLA라는 비동기 아키텍처를 제안합니다. 이 구조는 과제 상태 추론을 위한 저주파 Streaming Pilot Vision-Language Model (VLM)와 연속 제어를 위한 고주파 확산 행동 모델을 분리하여 멀티 태스크를 수행합니다. 또한 현재 비행 상태를 요약하고 다음 하위 목표를 예상하는 명확한 Pilot Reasoning 텍스트로 감독됩니다.

- **Performance Highlights**: 실시간 평가에서 FLIGHT VLA는 FLIGHT 벤치마크의 대표적인 VLN 및 VLA 기준을 지속적으로 초과 달성하며, 다단계 완료, 하위 목표 준수 및 터미널 제어에서 우수한 결과를 보여주었습니다. 학습된 Streaming Pilot Reasoning VLM은 UAV 비디오 추론을 더욱 향상시켜, 설계의 효과성을 입증합니다.



### STRIPS-WM: Learning Grounded Propositional STRIPS-style World Models from Images (https://arxiv.org/abs/2606.06832)
- **What's New**: 본 논문에서는 STRIPS-WM이라는 새로운 프레임워크를 소개하여, 로봇이 시각적 변화에서 이미지를 기반으로 한 STRIPS 스타일의 세계 모델을 학습합니다. 이 방법은 행동 주도 시각적 상태 전환을 통해 추상적 전이 그래프를 구성하고, 각 행동 레이블당 하나의 기초 명제를 학습하여 상징적 행동 모델을 형성하는 데 중점을 둡니다. 이러한 접근법을 통해 데이터를 기반으로 한 상징적 세계 모델을 효과적으로 학습할 수 있습니다.

- **Technical Details**: STRIPS-WM은 세 단계로 구성되어 있습니다. 첫 번째 단계에서는 행동 조건부 시각적 동역학 모델이 이미지를 바탕으로 한 작업 그래프를 생성합니다. 두 번째 단계에서는 CP-SAT 솔버를 사용하여 그래프 노드에 이진 기초 명제 벡터를 할당하고, 세 번째 단계에서는 시각적 기초 명제 분류기를 사용하여 학습된 기초 명제를 이미지에 다시 연결합니다. 이 과정은 물체 중심 감독 없이 수행되며, 고전적 계획이 가능한 프레임워크를 제공합니다.

- **Performance Highlights**: 시각적 재배치 작업에 대한 실험 결과, STRIPS-WM은 기존의 시각적 롤아웃, 잠재적 그래프 검색 및 잠재적 상징적 베이스라인에 비해 이미지 대 계획 성공률을 개선했습니다. 강력한 성능을 보여주며, 일반적인 고전적 계획 기법을 통해 새로운 시작 및 목표 이미지를 이용하여 계획을 수립할 수 있는 가능성을 제시합니다.



### Three-dimensional hydro-cluttered locomotion by an undulatory robo (https://arxiv.org/abs/2606.06829)
- **What's New**: 이 논문은 수중에서 장애물과 상호작용할 수 있는 다림질 없는 로봇 AquaMILR의 개발을 소개합니다. Hydro-cluttered 환경에서의 로봇 움직임에 대한 새로운 원칙을 제시하며, 물리적 접촉을 활용하여 효과적으로 진행할 수 있는 방법을 보여줍니다. 이 로봇은 프로그래밍 가능한 몸의 탄력성과 깊이 조절 기능을 통해 복잡한 환경을 탐색할 수 있습니다.

- **Technical Details**: AquaMILR는 양측 케이블 구동 작동, 프로그래밍 가능한 몸의 탄력성, 분산 깊이 조절 기능이 결합되어 있는 연장형 로봇입니다. 조인트는 양측 케이블 풀리 시스템으로 구동되며, 각 조인트는 독립적으로 제어되는 케이블로 작동하여 몸의 진동을 생성합니다. 이는 수중에서의 효율적인 운동을 가능하게 하고, 환경의 복잡성에 대한 적응력을 높입니다.

- **Performance Highlights**: 실험 결과 AquaMILR는 수중 맹그로브 환경에서 장애물을 우회하고 탐색하는 데 성공했습니다. 프로그래머블 몸의 탄력성이 장애물과의 접촉을 관리해 운동 성능을 향상시켰으며, 깊이 조절 기능 덕분에 3차원적으로 주변 환경을 탐색할 수 있는 능력이 있음을 보여주었습니다. 이는 수중 로봇 설계와 운영의 새로운 패러다임을 제시합니다.



### Lane Change Trajectory Planning for Personalized Driving Comfort and Mobility Efficiency (https://arxiv.org/abs/2606.06805)
Comments:
          Accepted by the IEEE Intelligent Vehicles Symposium (IEEE IV 2026), Detroit, MI, United States, June 22_25, 2026

- **What's New**: 이 연구는 차량의 차선 변경 시 걸리는 시간과 효율성을 동시에 고려한 신경망 기반의 플래너(neural network-driven planner)를 제안합니다. 특히, 세 번째 차수의 다항식 경로 생성기(third-order polynomial trajectory generator)와 최적의 경로 매개변수를 추정하는 학습 모듈(learning module)을 통합하여 다양한 주행 조건에서 최적의 경로를 생성합니다.

- **Technical Details**: 플래너는 공유된 백본(shared backbone) 구조를 사용하며, 두 개의 헤드를 가지고 있습니다. 하나의 헤드는 모든 주행 조건에서의 운영 보장을 제공하고, 다른 헤드는 운전자의 편안함(comfort) 또는 주행 효율성(mobility efficiency)을 잡아냅니다. 또한, 통계적 게이트(statistical gate)를 기반으로 하는 헤드 게이티드 스위칭 메커니즘(head-gated switching mechanism)을 통해 다양한 주행 조건에 맞게 적합한 헤드를 선택할 수 있습니다.

- **Performance Highlights**: 대표적인 사례와 몬테 카를로 시뮬레이션을 통해 이 플래너가 차선 변경 시 개인화된 편안함과 주행 효율성을 달성함을 보여주었습니다. 반면, 기본 방법은 개인화된 데이터가 부족하거나 접근할 수 없는 주행 조건에서도 가능한 경로를 보장합니다.



### Learning All-Terrain Locomotion for a Planetary Rover with Actively Articulated Suspension (https://arxiv.org/abs/2606.06790)
Comments:
          21 pages, 26 figures

- **What's New**: 이 논문은 ERNEST라는 4륜 행성 탐사 로버 개념을 소개합니다. 이 로버는 2자유도 능동 균형 서스펜션(Active Gimbal Suspension)을 장착하여 조향, 휠 재구성 및 능동 하중 재분배를 가능하게 합니다. 단일 신경망(neural network) 제어기를 활용하여 도전적인 지형을 자동으로 탐색할 수 있는 능력을 제공합니다.

- **Technical Details**: ERNEST의 설계는 복잡성을 최소화하면서도 다양한 지형을 탐색할 수 있도록 최적화되었습니다. 이 시스템은 고충실도 DARTS 시뮬레이션 엔진을 사용하여 강화 학습(framework using reinforcement learning) 기법을 적용합니다. 또한, terrain-specialized 경험을 하나의 신경망으로 통합하는 정책 통합 전략(policy consolidation strategy)을 개발하여, 다양한 지형에 대해 명시적인 제어기 전환 없이 동작할 수 있도록 하였습니다.

- **Performance Highlights**: 실험 결과, ERNEST는 바위, 모래, 높은 스텝 같은 다양한 지형을 독립적으로 탐색하는 능력을 입증하였습니다. 특히, 20°의 모래 경사에서 기존 방식보다 37%의 효율성을 이끌어내며, 습식 모래에서도 기계적 장애움직임에 강한 성능을 보여주었습니다. 이와 같은 개선된 성능은 탐사의 효율성 및 성공률을 크게 높이는 데 기여할 것입니다.



### Multi-Robot Planning and Control from CCTV Camera Networks in a Real Warehous (https://arxiv.org/abs/2606.06762)
- **What's New**: 본 연구는 외부 CCTV 네트워크와 엣지 컴퓨팅을 활용하여 실제 창고 환경에서 다중 로봇의 계획 및 조정을 효율적으로 수행하는 새로운 접근법을 제시한다. 기존의 고가의 센서와 컴퓨팅 장치 없이도 카메라 네트워크를 통해 로봇의 독립적인 비전 기반 모션 제어를 실현하며, 특히 중복된 카메라 시야를 활용하여 충돌을 방지하고 원활한 임무 수행이 가능하게 한다.

- **Technical Details**: 이 시스템은 비보정 상태의 픽셀 단위 토폴로지 카메라 그래프를 통해 이미지 공간 내에서 작동하며, 각 로봇에 대한 카메라 시퀀스를 선택하고 계획하는 계층적 플래너를 사용한다. 로봇 간의 충돌을 방지하기 위해 카메라 중첩 영역을 세마포어처럼 하나의 로봇만 소유하도록 정의하여, 동적 장애물로서 다른 로봇을 고려한 독립 계획을 통한 조정을 구현한다.

- **Performance Highlights**: 이 연구는 실제 창고에서 4개의 로봇과 30개의 카메라를 사용하여 27미터 길이의 통로 6개에 걸쳐 유효성을 검증하였다. 미션 성공률, 소요 시간과 충돌 통계 등을 보고하였으며, 특히 외부 카메라 네트워크와 오프 보드 컴퓨팅을 사용하는 다중 로봇 계획 및 조정의 첫 실험 결과라는 점이 주목받는다.



### AxisGuide: Grounding Robot Action Coordinate System in RGB Observations for Robust Visuomotor Manipulation (https://arxiv.org/abs/2606.06761)
Comments:
          Accepted to Robotics: Science and Systems (RSS) 2026

- **What's New**: 이번 연구에서는 비주얼 모터 조작 정책이 이해한 작업의 의미와 저수준 행동의 이해 사이의 간극을 다루고 있습니다. 새로운 방법론인 AxisGuide를 제안하여 로봇의 기본 좌표계 행동을 이미지 공간에서 해석할 수 있도록 시각적인 단서를 제공합니다. 이를 통해 다양한 시나리오에서 강력한 행동 실행 능력을 보여주고자 합니다. AxisGuide는 RGB 영상에 추가적인 채널을 통해 +x, +y, +z 동작의 의미를 명확히 시각화합니다.

- **Technical Details**: AxisGuide는 카메라 파라미터와 엔드 이펙터의 포즈를 활용해 각 카메라 뷰에서 로봇 베이스 프레임의 축을 그립니다. 이를 통해 작동 좌표계를 RGB 관찰과 연결하여, 정책이 새롭게 나타나는 물체 위치에 맞춰 더욱 신뢰성 있게 엔드 이펙터의 비행을 조정할 수 있도록 돕습니다. 주요 기술은 동작 공간의 의미를 픽셀 단위로 명시화하여, 각 RGB 이미지에 대한 행동 이해도를 향상시키는 것입니다.

- **Performance Highlights**: AxisGuide는 실제 환경과 시뮬레이션 환경 모두에서 성능 향상을 보여주었습니다. 성과는 특히 물체가 보지 못한 위치에 놓였을 때 성공률이 20%p까지 증가하는 것으로 측정되었습니다. 또한, 다각 뷰 및 단일 뷰 구성 모두에서 일관된 성공률 향상을 보이므로, 명시적인 행동 좌표 기준이 신뢰할 수 있는 실행을 촉진하는 데 기여했음을 나타냅니다.



### IDDMBSE: Integrating Data-Driven and Model-Based Systems Engineering for Trusted Autonomous Cyber-Physical Systems (https://arxiv.org/abs/2606.06727)
Comments:
          9 pages, 11 figures. This work has been submitted to the IEEE for possible publication

- **What's New**: 본 논문에서는 IDDMBSE(Integrated Data-Driven and Model-Based Systems Engineering)를 제안하며, 이는 전통적인 model-based 시스템 엔지니어링에 데이터 기반 루프를 통합하여 자율 사이버 물리 시스템(CPS)을 설계할 수 있는 새로운 방법론입니다. 이 방법론은 SysML을 기반으로 하여 각 단계에서 데이터 기반 접근 방식을 포함시키며, 실제 자율 로봇 개발 사례를 통해 그 유용성을 입증합니다.

- **Technical Details**: IDDMBSE는 시스템 구조, 동작 및 요구사항을 통합된 SysML 표현을 통해 추적 가능한 방법론으로 확장하며, 이를 현실화 하기 위해 세 가지 툴 체인 도구인 PERFECT, TRADES-X, VERITAS를 개발하였습니다. 이러한 도구들은 서로 호환 가능하며, 시스템 아키텍처를 ROS 자율성 스택으로 변환하고 모델 기반 최적화와 데이터 기반 평가단계를 결합합니다.

- **Performance Highlights**: 논문은 IDDMBSE 툴 체인을 통해 신뢰할 수 있는 자율 지상 로봇의 개발 생명주기 전반에 걸쳐 하드웨어 디자인, 소프트웨어 개발, 행동 보증을 수행함으로써 그 성능을 입증합니다. 또한, Isaac Sim 테스트 범위를 공개함으로써 자동화된 시험 환경에서의 성능 평가를 지원하고 있으며, 향후 SysML v2 및 KerML 기초로 IDDMBSE를 재구성하여 더 나은 ML/AI 통합을 위한 방향성을 제시합니다.



### SCOUT: Semantic scene COverage via Uncertainty-guided Traversa (https://arxiv.org/abs/2606.06721)
Comments:
          2026 ICRA Workshop on Uncertainty in Open World Robotics

- **What's New**: 본 연구에서는 SCOUT라는 최신 온라인 시맨틱 탐사 프레임워크를 제안하여, 로봇이 장기간 운영되면서 공간을 이해하도록 돕습니다. 이 시스템은 확률적 장면 그래프 구성과 능동적인 탐색을 결합하여, 로봇이 새로운 장면을 자율적으로 인식하고 데이터를 지속적으로 업데이트할 수 있도록 합니다. SCOUT는 기존의 정적 데이터 접근 방식에서 벗어나, 실시간으로 장면의 의미론적 완전성을 목표로 발전합니다.

- **Technical Details**: SCOUT는 두 가지 주요 구성 요소로 구성되어 있습니다: 불확실성 안내 탐색기(UGT)와 확률적 장면 그래프 생성기(PSGG)입니다. UGT는 시맨틱 정보를 활용하여 다음 관측 지점을 선택하고, PSGG는 로봇의 RGB-D 관측 데이터를 통합하여 불확실성 인식 3D 장면 그래프를 만들어냅니다. 이 시스템은 일련의 반복적 과정을 통해 탐색과 그래프 구성을 함께 조율하여, 객체의 불확실성을 해결합니다.

- **Performance Highlights**: SCOUT는 시각 정보와 공간적 관계를 동시에 고려하여 로봇의 장면 이해도를 극대화합니다. 실험 결과, SCOUT는 95%의 기하학적 커버리지와 0.7 미만의 불확실성 측정을 만족하는 정확도를 달성하며, 이는 로봇이 효율적으로 탐색하고 복잡한 환경을 관리할 수 있도록 지원합니다. 이러한 접근 방식은 로봇이 더욱 자율적으로 실내 환경을 순찰하고 업데이트하며, 필요한 경우 최소한의 인간 개입으로 의사결정을 할 수 있도록 합니다.



### Optimal Control Approach for Non-prehensile Ball Juggling Using a 7-DoF Manipulator (https://arxiv.org/abs/2606.06704)
Comments:
          8 pages, accepted at ICRA 2026

- **What's New**: 이번 연구에서는 로봇의 비잡기(non-prehensile) 물체 조작 기술을 이용해 공을 저글링(juggling)할 수 있는 모델 기반의 제어 프레임워크를 제안합니다. 해당 프레임워크는 작업 공간(task-space)에서 비로소 비로소 응답할 수 있는 궤적을 생성하고, 이를 통해 안정적인 저글링 동작을 구현하는 데 중점을 두었습니다. 연구진은 로봇 팔의 다양한 관절 운동을 최적화하여 지속적으로 저글링하는 패턴을 유지할 수 있는 방법을 개발하였습니다.

- **Technical Details**: 저글링 제어 시스템은 두 단계의 최적 제어 문제(Optimal Control Problem, OCP)를 통해 각 단계에서 동적 구속력을 고려합니다. 첫 번째 단계는 도구(tool)의 움직임을 위한 이상적인 기준 궤적을 생성하며, 두 번째 단계에서 이 궤적을 로봇의 관절 공간(joint-space)으로 매핑합니다. 이렇게 생성된 궤적들은 외부 방해나 모델 정확도 부족에 대한 강건성을 제공하기 위해 오프라인으로 다양한 경계 조건에 대해 반복적으로 계산되어 데이터베이스에 저장됩니다.

- **Performance Highlights**: 제안된 제어 시스템은 Franka Emika Panda 로봇을 사용해 시뮬레이션 환경에서 저글링 성능을 평가하였고, 실험 결과 안정적인 저글링 패턴을 유지할 수 있음을 보였습니다. 이 시스템은 비잡기 물체 조작 기술을 이용한 새로운 제어 건축물(control architecture)을 통해 고속 비동기 적 디시전 사이클을 가능하게 합니다. 최적 궤적 데이터베이스를 활용함으로써 실시간으로 오류를 수정하고 지속적인 저글링이 가능하다는 점을 보여주었습니다.



### On the Hardness of Optimal Motion on Trees (https://arxiv.org/abs/2606.06686)
- **What's New**: 이 논문은 다중 에이전트 경로 찾기(MAPF)의 복잡성을 단순화하는 프레임워크를 제안하며, 이는 나무에서의 거리(distance), 총 소요 시간(makespan), 흐름 시간(flowtime)과 같은 여러 표준 목표에 대해 적용됩니다. 또한 레이블이 있는 경우와 색상이 있는 경우의 두 가지 변형에 대해 NP-난해성을 증명합니다. 특히, 'Pebble Motion' 문제를 해결하여 최대 이동 횟수를 최소화하는 목표를 설정하고 있습니다.

- **Technical Details**: 다중 에이전트 경로 찾기(MAPF)는 여러 에이전트가 동시에 이동할 수 있는 최적화 문제이며, leavels가 지정된 경우와 색상에 따라 교환할 수 있는 경우의 다양한 변형이 있습니다. 이 연구는 가볍게 나뉘어진 별 구조(subdivided stars)와 같은 매우 간단한 나무에서도 제시된 모든 문제에서의 NP-난해성 결과를 확인하였습니다. 특히, Stack Rearrangement 문제의 복잡성을 통해 이러한 결과들이 정립되었습니다.

- **Performance Highlights**: MAPF의 세 가지 목표에 대한 NP-난해성 결과를 처음으로 제공하며, 이는 나무에서의 두 색상을 가진 MAPF의 첫 NP-난해성과 동일합니다. 연구에서 확인된 결과는 이전의 연구들에 비해 문제의 복잡성을 명확히 하고, 효율적인 해결 방법을 제공하는 것으로 하여, 이러한 문제들이 서로 연결되어 있다는 것을 보여주고 있습니다. 최적의 흐름 시간(flowtime) 문제에 대한 최초의 난해성 결과도 함께 제시되었습니다.



### What Matters When Cotraining Robot Manipulation Policies on Everyday Human Videos? (https://arxiv.org/abs/2606.06627)
Comments:
          The project website is here: this https URL

- **What's New**: 이번 연구에서는 532개의 인간 동영상과 28시간의 고품질 3D 손 레이블을 포함한 새로운 데이터셋을 사용하여 로봇 정책을 공동 훈련(cotraining) 하는 방법을 탐구합니다. 인터넷 동영상을 통해 인간의 자연스러운 동작을 활용할 수 있는 가능성을 제시하며, 손 자세의 품질이 로봇으로의 전이에 영향을 미친다는 것을 발견했습니다. 연구진은 고품질 손 레이블이 있는 데이터셋을 통해 로봇이 더 성공적으로 학습하도록 하는 방법을 제안하고, 전이의 주요 요인으로 3D 손 자세의 질과 자연 인간 동작의 차이를 밝혔습니다.

- **Technical Details**: 연구진은 EgoExo4D 데이터셋을 기반으로 532개의 요리 동영상에서 고품질 3D 손을 다중 뷰 삼각 측량(multi-view triangulation)하여 새로운 데이터셋을 구축했습니다. 이 과정에서 2D 키포인트의 정확성을 향상시키기 위해 모델 기반 포즈 추정기를 활용하고, 손 자세 추정기를 재조정(retraining)하여 기존의 2D 추정기보다 더 나은 결과를 얻었습니다. 또한, 미니배치(mini-batch) 훈련 시 손 자세의 품질과 전이에 대한 주요 이슈를 식별하여, 기술적 접근 방식으로 비율 정렬(scale alignment)과 토큰 단위 융합(token-level fusion)을 제안하였습니다.

- **Performance Highlights**: 연구는 532개의 인간 동영상과 3,000개의 로봇 시연으로 훈련하여 6가지 실제 조작 작업에 대해 평가를 실시했습니다. 공동 훈련 방법은 로봇 데이터가 적은 환경에서 +29.7%의 절대 성과 향상을 보였으며, 특정 작업에 맞춘 로봇 데이터가 늘어날수록 성과는 계속 향상되었습니다. 이는 연구진이 제안한 데이터셋이 기존의 대규모 실험실 데이터셋보다 로봇 전이 성과가 우수함을 입증하는 결과입니다.



### ChronoForest: Closed-Loop Multi-Tree Diffusion Planning for Efficient Bridge Search and Route Composition (https://arxiv.org/abs/2606.06618)
Comments:
          40 pages, 4 figures, 7 tables, 3 algorithms

- **What's New**: 이 논문에서는 오프라인 네비게이션의 도전 과제인 단기 데이터만으로 장기 경로를 계획하는 문제를 다룹니다. ChronoForest라는 폐쇄 루프 계획 시스템을 제안하여, 단기 오프라인 데이터에서 장기 목표지점으로의 경로를 효율적으로 생성하는 방법을 제시합니다. 이 시스템은 시간적 거리(temporal distance)와 다수의 트리를 활용하여 경로 재해결을 수행하며, 목표 도달을 위한 연결성을 검증합니다.

- **Technical Details**: ChronoForest는 로컬 브리지 탐색(local bridge search)과 온라인 경로 재해결을 결합한 구조로, 낮은 수준의 앵커 체인 트리 확산 계획가(Anchor-chaining tree diffusion planner)와 온라인 다중 트리 조율자(Online multi-tree orchestrator)를 포함합니다. 저렴한 탐색 예산 내에서 성과를 최적화하며, 앵커 간 비용을 실시간으로 평가하여 경로 품질을 향상시킵니다. 이 시스템은 각 앵커 쌍을 단기적으로 연결할 수 있는 다리(brige)를 효율적으로 생성합니다.

- **Performance Highlights**: ChronoForest는 OGBench AntMaze-Stitch 데이터 세트에서 각각 99.8%, 99.3%, 99.5%의 성공률을 기록하며 이전의 확산 기반 결과보다 34.5 포인트 향상된 성과를 보였습니다. 또한, 헬미토니안 경로 구성(Hamiltonian route-composition) benchmark에서는 온라인 재해결을 통해 시간적 순서 오류를 교정하고 경로 품질을 개선하면서도 전면적인 탐색 계획(exhaustive planning)보다 훨씬 낮은 비용을 유지했습니다.



### PhyRoGen: Synthetic Generation of Physical Robot Manipulation Puzzles Using Procedural Content Generation (https://arxiv.org/abs/2606.06569)
Comments:
          8 pages, accepted at CASE 2026

- **What's New**: 이번 논문에서는 로봇의 물리적 퍼즐 조작을 위한 새로운 프레임워크인 Physical Robot Manipulation Puzzle Generation (PhyRoGen)을 제안합니다. 이 프레임워크는 절차적 콘텐츠 생성(procedural content generation, PCG)을 활용하여 조작 퍼즐의 합성 데이터셋을 자동으로 생성합니다. PhyRoGen은 상호 의존성을 가진 물리적 퍼즐을 생성할 수 있는 범용 퍼즐 생성기로, 다양한 조작 알고리즘의 성능을 벤치마킹하는 데 중요한 역할을 합니다.

- **Technical Details**: PhyRoGen은 세 가지 주요 부분으로 구성되며, 각각 키네마틱 체인 생성기(kinematic chain generators)를 정의합니다. 이 알고리즘은 객체 객체(Object), 조인(Joint), 변환(Transformation)으로 이루어진 입력을 받아 키네마틱 체인(kinematic chain)을 반환하는 절차적 모델입니다. 총 2424개의 무작위 조작 퍼즐을 만든 후, KUKA LBR iiwa 로봇을 활용하여 조작 가능성을 입증했습니다.

- **Performance Highlights**: 제안된 프레임워크는 모든 생성된 퍼즐을 1초에서 300초 사이에 샘플링 기반 계획 알고리즘을 사용하여 해결하는 성과를 보였습니다. 이러한 성능을 통해 PhyRoGen은 독특하고 해결 가능한 로봇 조작 퍼즐을 생성할 수 있는 능력을 보여줍니다. 이는 조작 알고리즘 벤치마킹과 내구성이 강한 기초 모델 개발에 필수적인 요소임을 강조합니다.



### Robots Need More than VLA and World Models (https://arxiv.org/abs/2606.06556)
- **What's New**: 이 논문에서는 로봇 지능의 발전을 위해 '정책 학습' 외에 '비구축 데이터(ungrounded data)'와 로봇 감독을 제공할 수 있는 메커니즘이 부족하다는 점을 강조합니다. 로봇이 탁월한 일반화를 이루는 데 필요한 핵심 구성 요소 네 가지로 데이터 인터페이스, 인체 동작 리타게팅, 물리 기반 세계 모델, 보상 인터페이스를 제시합니다. 이는 로봇이 어떻게 자연적인 데이터로부터 학습할 수 있을지를 재정의하는 중요한 논의입니다.

- **Technical Details**: 현재 로봇 시스템은 다양한 데이터 소스로부터 로봇 학습의 감독을 넓히려는 시도를 하고 있으며, 기본적으로 로봇 데이터 중심으로 구성되어 있습니다. 이 논문은 로봇이 물리적 경험을 통해 학습할 수 있도록 하는 메커니즘을 구체적으로 다룹니다. 앞으로 로봇 학습 파이프라인은 광범위한 물리적 경험을 기반으로 해야 하며, 이를 통해 행동, 상호작용 및 목표 달성에 필요한 정보를 제공하는 방법을 개발해야 합니다.

- **Performance Highlights**: 최근의 로봇 데이터 셋은 다양한 과제와 환경에서 로봇의 수행 능력을 높이는 데 중요한 역할을 하고 있습니다. 예를 들어, BC-Z는 100개 이상의 과제를 아우르는 로봇 모방 데이터의 확장을 통해 제로샷 일반화를 가능하게 했으며, RT-1은 대규모 데이터셋으로 훈련된 변환기가 여러 로봇의 행동을 학습할 수 있도록 했습니다. 이러한 성과들은 데이터의 다양성과 규모가 로봇의 일반화 능력에 필수적임을 보여줍니다.



### Physiologically Constrained Musculoskeletal Neural Network for Multi-DoF Joint Kinematics Estimation from Partially Observed sEMG (https://arxiv.org/abs/2606.07476)
- **What's New**: 이 논문은 부분 관찰된 표면 근전도(sEMG) 하의 다자유도(DoF) 관절 운동학 추정에 대한 연구를 다룹니다. 기존의 하이브리드 신경망 접근 방식과 달리, 제안된 새로운 근골격계 신경망(MSK-NN)은 측정된 근육과 측정되지 않은 근육의 활성화를 동시에 추정할 수 있습니다. MSK-NN은 CNN 기반의 근육 활성화 추정기와 통합된 MSK 전방 동역학 모듈을 포함하여 완전 미분 가능한 아키텍처를 형성합니다.

- **Technical Details**: MSK-NN 프레임워크는 측정된 근육으로부터의 sEMG 데이터를 입력받아 활성화 추정을 수행하고, 이 활성화를 통하여 근육 힘을 생성한 후 관절 토크에 매핑하여 여러 DoF 관절 각도를 예측합니다. 이 과정에서 복합 물리-생리 손실 함수를 사용하여 모델이 생리학적으로 그럴듯한 활성화를 추정하도록 정규화됩니다. 또한, 활성화 추정기는 경량의 CNN 기반 인코더를 사용하여 측정된 근육과 미측정 근육의 활성화를 동시에 추정하도록 설계됩니다.

- **Performance Highlights**: MSK-NN은 무제한 속도와 진폭을 가진 세 가지 리드믹 동작 및 하나의 랜덤 동작을 포함한 두 자유도 손목 운동학 추정에서 기존의 Bi-LSTM, CNN-LSTM, PET 기법과 비교하여 NRMSE와 결정 계수(R²)에서 우수한 성능을 나타냈습니다. 특히 랜덤 동작의 경우, 일반화 성능이 강조되었습니다. 입력이 포함되지 않은 근육의 활성화 추정이 기록된 sEMG와 강한 시간적 일치를 보였으며, 이는 MSK-NN이 부분적으로 관찰된 sEMG로부터 생리학적으로 그럴듯한 활성화를 도출할 수 있음을 보여줍니다.



### On orbital stabilization of a circular motion primitive for a dynamic extension of the Dubins car mod (https://arxiv.org/abs/2606.07449)
Comments:
          34 pages

- **What's New**: 이 논문에서는 Dubins 자동차 모델의 동적 확장에 대한 원 운동의 안정성을 다루며, 전이 선형화(transverse linearization) 프레임워크에서 이를 분석합니다. 기존의 전이 선형화 접근법에서는 안정화가 불가능하다는 것을 보여주며, 새로운 조건을 제시하여 원운동의 안정성을 보장할 수 있는 방법을 제기합니다. 이 연구는 원 운동 제어에 대한 돌파구를 제시하며, 기존의 통념에 도전합니다.

- **Technical Details**: 논문에서는 전이 선형화에 기반한 컨트롤러 설계에서의 불안정성과 선형화의 비가역성(non-integrable constraints) 문제를 언급합니다. 이는 기존 시스템에 적용되는 여러 테스트 케이스에서 성립되지 않을 수 있음을 의미합니다. 또한 비표준 전이 좌표(non-standard transverse coordinates)를 사용하여 특정 동역학 구조를 더욱 철저하게 분석합니다.

- **Performance Highlights**: 수치 시뮬레이션을 통해 제안된 컨트롤러 설계 절차가 효과적임을 보여줍니다. 특히, 원 운동에서의 안정성 확보를 위한 조건과 방법론이 주목받고 있습니다. 이 연구는 비선형 제어 이론과 해석역학(analytical mechanics) 간의 연결 고리를 밝히며, 향후 비선형 시스템의 동적 분석을 위한 기초 자료로 활용될 수 있습니다.



### Dash2Sim: Closed-Loop Driving Simulation from in-the-wild Dashcam Videos (https://arxiv.org/abs/2606.07366)
- **What's New**: 새로운 연구인 Dash2Sim은 현장 모노큘러 대시캠 비디오를 메트릭( Metric) 및 지리적으로 참조된 4D 드라이빙 로그로 변환하는 프레임워크입니다. 이 프레임워크는 주석(annotations) 없이 독립적으로 유지되는 맵과 각 비디오를 검증하여 정확한 4D 장면을 회복하기 어려운 단점을 극복합니다. 이를 통해 연구진은 17개 도시에서 4,244개의 장면과 270만 개의 3D 객체를 포함하는 ROADWork4D 벤치마크 데이터셋을 생성했습니다.

- **Technical Details**: Dash2Sim은 대시캠 비디오를 활용하여 그 데이터에서 작업 구역(work zones)과 같은 드문 경우의 상황을 캡처합니다. 이 프레임워크는 단일 카메라 영상(monocular videos)에서 복잡한 4D(scene) 장면을 변환하여 기존 시뮬레이터와 호환되게 합니다. 로드워크 시뮬레이션에서 사용되는 이 데이터셋은 특히 폐쇄 루프(planners) 시나리오에서 활용될 수 있으며, 기존의 플래너와 비교하여 품질을 개선할 수 있는 가능성을 제공합니다.

- **Performance Highlights**: ROADWork4D-CL 데이터셋에서 검증된 부분에 대해 연구진은 전통적 규칙 기반(rule-based) 및 하이브리드(hybrid) 플래너가 특히 더 잘 일반화(generalize)하는 것을 발견했습니다. 그러나 이러한 플래너들조차도 임시 작업 구역에서 요구되는 차선 변경을 수행하는 데 실패하였습니다. Dash2Sim에서 회복된 밀집 깊이(dense depth)는 새로운 모습 합성(novel-view synthesis) 품질을 최대 19%까지 향상시켰으며, 이는 모노큘러 비디오에서 폐쇄 루프 센서 시뮬레이션을 위한 풍부한 조건 제공 가능성을 나타냅니다.



### Does Appearance Help? A Systematic Study of Image-Based Re-Identification in Online 3D Multi-Pedestrian Tracking (https://arxiv.org/abs/2606.07233)
Comments:
          Accepted for publication at the 35th IEEE International Conference on Robot and Human Interactive Communication (RO-MAN 2026)

- **What's New**: 이번 연구는 LiDAR 기반의 3D Multi-Object Tracking (MOT) 시스템에서 전통적인 기하학적 정보의 취약성을 보완하기 위해 이미지 기반 Re-Identification (ReID) 접근 방식을 활용하는 새로운 경량 프레임워크를 제시합니다. 이 체계적인 연구는 모바일 로봇의 성능과 실시간 반응성을 높이는 동시에, 상호작용을 지속적으로 유지할 수 있도록 모듈식 ReID 지점을 통합했습니다. 다양한 다중 모달 데이터 연관 전략을 평가하여 계산 지연과 강력한 추적의 균형을 맞추는 방법을 제안합니다.

- **Technical Details**: 연구는 PointPillars와 AB3DMOT를 기반으로 한 경량의 3D 감지기 및 추적기를 설정하였으며, 여러 기능 추출 구조와 다양한 다중 모달 연관 전략을 평가하여 최적의 LATENCY와 판별력을 찾았습니다. CNN 및 Vision Transformer 기반의 기능 추출기를 구성하여 실시간 3D MOT의 성과를 분석하였고, KITTI 데이터 세트에서 결과를 평가했습니다. 고전적인 모션 중심 접근 방법의 한계를 극복하기 위한 ReID 통합의 효과를 논의하며, occlusions에서의 추적 정확도를 보장하는 중요성을 강조합니다.

- **Performance Highlights**: 기존의 단순한 선형 퓨전 방식은 비주얼 노이즈로 인해 성능을 저하시켰지만, 계단식 매칭 전략은 occluded 트랙을 복구하는 데 성공했습니다. 이를 통해 전반적인 정확도를 유지하면서도 인간-로봇 상호작용의 연속성을 확보하는데 기여할 수 있음을 보여주었습니다. 이 연구 결과는 안전한 navigation을 위한 낮은 지연(latency)과 사회적 인식을 위한 판별력 사이의 최적의 거래를 제안하며, 경량 아키텍처를 이용하여 실시간 환경에서의 성능 향상을 확인하였습니다.



### LARA: Latent Action Representation Alignment for Vision-Language-Action Models (https://arxiv.org/abs/2606.07100)
- **What's New**: 이번 연구에서는 Latent Action Representation Alignment (LARA)라는 새로운 프레임워크를 제안합니다. LARA는 Latent Action Models (LAM)과 Vision-Language-Action (VLA) 모델을 함께 최적화하여 서로에게 이익이 되는 방향으로 학습을 진행할 수 있게 합니다. 이를 통해 VLA 모델의 성능을 저하하는 비효율적인 행동 경로를 줄이고, VLA는 LAM의 정방향 동작 예측에 의해 규제됩니다.

- **Technical Details**: LARA는 LAM과 VLA의 레이턴트(잠재적) 행동 표현을 정렬하는 경량화된 메커니즘을 통해 연결됩니다. LAM은 행동 경로와 함께 공동 학습을 통해 시각적인 변화를 줄이는 반면, VLA는 LAM에서 학습한 동역학(dynamics)을 사용하여 잘못된 행동 경로의 환상을 줄입니다. 이는 Diffusion 모델의 레이턴트 표현 정렬 관련 최근 연구에 영감을 받아 설계되었습니다.

- **Performance Highlights**: 실험 결과 LARA는 시뮬레이션 및 실제 로봇 제어 환경에서 각각 평균 10%, 5%, 15% 개선 효과를 보였습니다. LARA는 기본 VLA 모델의 학습을 강화하는 중요한 툴로 작용하며, 사전 훈련된 VLA 모델을 후속 훈련 단계에서 개선하고, LAM의 레이턴트 행동 표현을 정제하는 데 효과적입니다. 이러한 성과는 VLA 학습의 표준 훈련 파이프라인에서 혁신적인 전환점을 제공합니다.



### AEGIS: A Backup Reflex for Physical AI (https://arxiv.org/abs/2606.06660)
- **What's New**: 본 논문은 로봇 조작의 장기 실패를 예방하기 위한 새로운 방법인 AEGIS(Activation-probe Early-warning, Gated Inference Switching)를 소개합니다. AEGIS는 약한(policy) 정책의 동결된 활성화(activations)를 사용하여 고위험 단계를 감지하고, 문제 발생 전 제어를 더 강력한 정책으로 전환합니다. 이런 방식은 실패가 발생할 것으로 예상되는 단계에서만 더 강한 정책을 호출하여 효율성을 높입니다.

- **Technical Details**: AEGIS는 약한 정책의 활성화를 기초로 하여 단계별로 조기 경고를 날립니다. 특정 임계값(threshold)을 넘어설 경우에만 강력한 정책으로 전환하고, 이로 인해 복구(measure recovery)가 가능합니다. 그 결과, AEGIS는 약한 정책으로 발생하는 실패의 10.1%를 회복하며, 이는 다른 접근법에 비해 상대적으로 높은 수치입니다.

- **Performance Highlights**: 실험 결과, AEGIS는 약한 정책의 실패를 10.1% 회복하며, 이는 대조군인 붐 인상과 무작위 트리거에 비해 각각 5.4pp, 5.0pp의 개선을 보입니다. 해당 성과는 700개의 에피소드를 통해 검증 되었으며, 예측 정확도도 0.76의 AUROC으로 고무적인 결과를 보였습니다. 이러한 결과는 AEGIS가 예측과 회복을 효과적으로 결합할 수 있음을 보여줍니다.



New uploads on arXiv(cs.MA)

### Modelling Opinion Dynamics at Scale with Deep MARL (https://arxiv.org/abs/2606.07487)
Comments:
          35 pages, 28 figures, preprint

- **What's New**: 이번 연구에서는 전통적인 의견 동적 모델링 방법 대신, 다중 에이전트 강화 학습 (Multi-Agent Reinforcement Learning, MARL)을 통해 의견 형성과 극단화 같은 매크로 현상을 직접 학습하는 방법을 탐구합니다. GPU 가속을 활용한 동기화 및 진실 추구 게임을 개발하여 최대 1000명의 에이전트로 구성된 집단에서의 의견 동적 현상을 모사할 수 있게 하였습니다.

- **Technical Details**: 연구에서는 일반 합 산 사회적 상호작용을 확장하여 비현실적인 규범을 방지하는 방식으로 다른 플레이 (other-play)를 도입하였습니다. Bluesky 네트워크의 일부를 대상으로 에이전트 중요 구조를 그래프 토폴로지에서 학습된 주의층 (attention layer)을 통해 복원하여, 높은 규범성을 가진 집단이 인간 데이터와 가장일치함을 발견하였습니다.

- **Performance Highlights**: 대규모 소셜 미디어 네트워크에서는 그런 높은 수준의 규범성이 집단의 정확도를 상당히 저하시켜, 주변에 맞추기 위해 거짓말을 하는 비정직한 에이전트가 증가하는 경향이 있습니다. 반면, 소규모의 역동적인 채집자 네트워크에서는 규범성이 집단의 동의도를 오히려 개선할 수 있는 긍정적인 영향을 미칩니다. 이러한 결과는 진화한 인간 규범성의 휴리스틱과 현대 소셜 미디어 환경 간의 괴리가 잘못된 정보의 기여 요인 중 하나일 수 있음을 시사합니다.



### Hierarchical Certified Semantic Commitment for Byzantine-Resilient LLM-Agent Collaboration (https://arxiv.org/abs/2606.07316)
Comments:
          27 pages, 3 figures, 8 tables

- **What's New**: 이번 논문에서는 대규모 언어 모델(LLM) 에이전트 간의 비잔틴 협력 문제를 해결하기 위한 새로운 프로토콜인 Hierarchical Certified Semantic Commitment (H-CSC)를 제안합니다. 이 프로토콜은 여러 제안들 중 어떤 결과를 최종적으로 결정해야 하는지를 판단해야 합니다. 기존의 비잔틴 결함 허용(BFT) 프로토콜은 비트 레벨에서의 동작을 기반으로 하였으나, H-CSC는 언어적 제안의 의미론적(core) 내용에 기반하여 결정을 내립니다.

- **Technical Details**: H-CSC는 최종성( конечность) 신호를 embedding 기반으로 변환하여 세 가지 유형의 결과를 내놓습니다: semantic_commit(의미적 접속이 존재할 때), verdict_commit(판단의 우세가 나타날 때), 그리고 명시적 abort(의도가 불가능할 때). 이 프로토콜은 메트릭 공간 내에서 제안들을 기하학적으로 집계하여 각 라운드에 대해 적합한 결정을 내립니다.

- **Performance Highlights**: H-CSC는 제어된 의미적 오염 진단에서 0.31도에서 2.04도의 낮은 각도 편차로 커밋을 수행하며, 비잔틴 조건을 초과하는 라운드에서는 100%의 abort를 실현했습니다. 실제 LLM 에이전트 검증 벤치마크에서 H-CSC는 0.90/0.92의 높은 정확도를 기록하면서도 강력한 증명서를 추가하여 타당한 결과를 가져왔습니다.



### Learning Multi-Agent Communication Protocol: Study on Information Entropy Efficiency in MARL (https://arxiv.org/abs/2606.07200)
- **What's New**: 이 논문에서는 Multi-Agent System (MAS)과 Multi-Agent Reinforcement Learning (MARL)에서의 효율적인 통신 프로토콜 학습에 대한 기초를 다루고 있습니다. 특히, 정보 저비용 효율 지수(Information Entropy Efficiency Index, IEI)를 제안하여 메시지 엔트로피(message entropy)와 작업 성능(task performance)의 비율을 정량화하고, 이를 통해 더 효율적인 커뮤니케이션을 장려합니다. 이 접근법은 비효율적인 통신 대신 높은 성능을 유지하면서도 메시지 크기를 줄이는 방향으로 나아가게 합니다.

- **Technical Details**: 제안하는 IEI는 메시지 엔트로피와 작업 성능 간의 관계를 체계적으로 평가할 수 있는 지표로 기능하며, 이는 정책 업데이트(policy update)에 IEI를 통합하여 대화방식이 정보의 유용성을 극대화하도록 돕습니다. 이 논문은 여러 MARL 알고리즘을 통해 IEI 기반의 학습을 허용함으로써, 보다 compact한 정보 전달로 인한 작업 성능 향상을 목표로 합니다. 또한, 연속적인 다중 라운드 커뮤니케이션 시나리오에서의 MARL에 대한 일반화된 프레임워크를 제시하여, 다양한 아키텍처간의 분석이 가능하도록 합니다.

- **Performance Highlights**: 실험 결과, 제안한 접근법은 기존 기본 방법들과 비교하여 동등하거나 우수한 작업 성능을 달성하면서 통신 효율성을 개선하였습니다. 컴퓨터 자원이 제한된 환경에서의 통신 집약적인 시스템 배포의 실용성을 간과할 수 있는 현재 연구의 한계에 도전하며, 통신 효율성과 작업 성공률을 동시에 향상시킬 수 있는 가능성을 보여줍니다. 이러한 발견은 성능 향상이 복잡한 아키텍처나 통신 오버헤드의 증가를 요구하지 않음을 증명하며, MAS의 대규모 확장을 위한 새로운 패러다임을 제시합니다.



### Modeling U.S. Attitudes Toward China via an Event-Steered Multi-Agent Simulator (https://arxiv.org/abs/2606.06971)
- **What's New**: 본 연구에서는 미국과 중국 간의 여론 동태를 정확하게 시뮬레이션하기 위한 Event-Steered Multi-Agent Simulator(ES-MAS)를 제안합니다. 기존의 시뮬레이터들은 정적인 규칙과 고정된 데이터를 기반으로 운영되었으나, ES-MAS는 사건 및 뉴스에 기반한 동적인 상호작용을 통해 여론 변화를 실시간으로 반영합니다. 이를 위해 2021년부터 2025년까지의 중대한 사건 258건과 14,000건 이상의 일일 뉴스를 포함하는 중국-미국 관계 진화(CURE) 데이터셋을 구성하였습니다.

- **Technical Details**: ES-MAS는 Dual-Stream Data Integration Engine(DSDIE)와 News-Driven Dynamic Interaction(NDDI) 모듈을 통해 동시적으로 중요한 사건과 개인화된 뉴스 노출을 통합합니다. DSDIE는 역사적인 타임라인과 사건을 기반으로 시뮬레이션을 정렬하며, NDDI는 유사한 뉴스 관심사를 가진 에이전트를 지역화된 상호작용 컨텍스트로 그룹화하여 아래로부터의 합의 형성을 촉진합니다. 이 과정에서 개인적인 의견이 미국의 중대한 공공 태도로 전환될 수 있습니다.

- **Performance Highlights**: CURE 데이터셋을 기반으로 한 실험 결과, ES-MAS는 기존의 시뮬레이터들과 비교하여 현실 세계의 역사적 경향을 재현하는 데 현저히 뛰어난 성과를 보였습니다. 이러한 결과는 동적 여론 진화를 모델링하기 위한 확장 가능하고 효과적인 프레임워크를 제공하게 됩니다. 이는 특히 소규모 네트워크에서 의견의 사회적 확산을 명확히 반영할 수 있다는 점에서 중요한 의미를 가집니다.



### MADRAG: Multi-Agent Debate with Retrieval-Augmented Generation for Training-Free Analytic Essay Scoring (https://arxiv.org/abs/2606.06754)
Comments:
          21 pages, 7 figures, 14 tables

- **What's New**: MADRAG는 훈련이 필요 없는 분석 에세이 평가 프레임워크를 제시합니다. 이 프레임워크는 다중 에이전트 추론(multi-agent reasoning)과 검색 기반의 기초(retrieval-augmented grounding)를 결합한 혁신적인 접근 방식을 사용합니다. 기존의 LLM-as-judge 방식과 달리, MADRAG는 평가를 상호작용 과정으로 분해하여 더 안전하고 신뢰할 수 있는 점수 산출이 가능합니다.

- **Technical Details**: MADRAG는 Advocate, Skeptic, Judge의 세 가지 역할로 구성된 시스템입니다. Advocate는 에세이의 강점을 찾고, Skeptic은 약점을 비판하며, Judge는 이 두 가지 주장을 종합해 최종 점수를 매깁니다. 또한, Judge는 스코어가 매겨진 예시들과의 비교를 통해 조정(calibration)을 수행할 수 있도록 사전 평가(rubric-aligned exemplar retrieval)를 활용합니다.

- **Performance Highlights**: MADRAG는 기존의 프롬프트 기반(baseline) 접근 방식보다 성능이 현저히 향상된 결과를 보여주었습니다. 전문 교육 없이도 감독되는 시스템(supervised systems)의 성능에 근접하는 결과를 얻었습니다. 검증 결과에 따르면, 검색이 조정에서 영향을 미치고, 토론이 높은 수준의 특성에 대한 추론을 개선하는 데 기여함을 보여주고 있습니다.



### From Privacy to Workflow Integrity: Communication-Graph Metadata in Autonomous Agent Interoperability (https://arxiv.org/abs/2606.07150)
Comments:
          12 pages, 6 figures

- **What's New**: 이번 논문에서는 A2A 및 MCP와 같은 에이전트 상호 운용성 프로토콜이 메시지 콘텐츠를 표준화하며, HTTP(S)를 통한 주소 기반 전송을 가정하고 있다는 점을 강조합니다. 보안은 주로 메시지 콘텐츠 보호에 집중되지만, 통신 그래프(communication graph)는 노출되어 결국 개인 정보 보호 이상의 문제로 발전할 수 있습니다. 이 논문은 이러한 메타데이터가 예측 가능한 방식으로 에이전트의 작업 흐름에 대한 통찰력을 제공하여, 권위 있는 행동 예측에 대한 위험을 제기한다고 주장합니다.

- **Technical Details**: 글에서는 에이전트 통신 그래프의 위협 모델을 개발하고, 에이전트 메타데이터가 어떤 이유로 독특하게 드러나는지 설명하여, 개인 정보 보호 차원을 넘어 자율적 워크플로우의 무결성에 대한 위협으로 재구성합니다. 뿐만 아니라, 메시지의 전송 방식(transport) 및 부트스트랩 계층의 프라이버시 특성 프레임워크를 정의하고, SLIM, Tor, mixnets와 같은 후보 전송 방식을 평가합니다. 마지막으로, A2A 사례 연구를 통해 메타데이터 보호 결합이 어떻게 설정될 수 있는지를 보여 줍니다.

- **Performance Highlights**: 실험 결과, 수동 메타데이터만으로도 작업 클래스가 우연보다 훨씬 잘 회복됨을 보여주며, 전체 속성 세트를 사용해야만 그러한 회복이 우연의 경향으로 위축된다는 것을 발견했습니다. 장애인이 특정 워크플로우를 기반으로 작업을 수행할 경우, 이 모델에서는 메타데이터를 알지 못하는 공격자가 얻을 수 있는 이점의 대부분을 캡처할 수 있습니다. 이러한 결과는 공격자가 워크플로우의 열기 및 고정 예산을 기준으로 행동할 때 예측 가능한 행동을 촉진한다는 것을 시사합니다.



### The Three-Ring Architecture: Governing Agents in the Era of On-Platform Organisations (https://arxiv.org/abs/2606.07119)
Comments:
          28 pages

- **What's New**: 현재 기업 AI 배포의 새로운 단계는 구조적 실패를 겪고 있습니다. 조직들은 관리 인프라 없이 능동적인(agentic) 기능을 확보하고 있으며, 이로 인해 95%의 프로젝트 실패율을 초래할 것으로 예상하고 있습니다. 이 논문은 플랫폼 조직의 관리 인프라로서 Three-Ring Architecture를 형식화하며, 각 링이 수행하는 역할을 설명합니다.

- **Technical Details**: Three-Ring Architecture는 세 개의 링으로 구성됩니다. 첫 번째 링(Ring 1)은 기존의 생산 아키텍처, 두 번째 링(Ring 2)은 전략 기반의 능동형 AI를 위한 M2 연합 계층, 세 번째 링(Ring 3)은 LLM 기반의 최전선 지능 계층입니다. Ring 2는 기업의 운영체제와 같은 역할을 하며, 자원 추상화, 프로세스 조정, 권한 집행을 통해 지능의 복합화를 가능하게 합니다.

- **Performance Highlights**: 이 아키텍처는 지난 10년간 금융 서비스, 정부, 조달 및 규정 준수 분야에서 검증되었습니다. LLM의 능력 향상은 Ring 2 아키텍처에 대한 구조적 이점을 제공하며, 이러한 능력이 향상됨에 따라 Governance 요구 사항도 비례하여 증가합니다. 이 논문의 주장은 모든 조직이 LLM의 능력을 활용하기 위해 꼭 필요한 Ring 2가 필요하다는 점에 있습니다.



### Learn to Match: Two-Sided Matching with Temporally Extended Feedback (https://arxiv.org/abs/2606.06744)
- **What's New**: 이 논문에서는 2측 매칭 시장에 대한 새로운 접근 방식을 제안합니다. 기존의 매칭 모델은 고정된 선호도에 대한 즉각적인 피드백만을 고려한 반면, 새로운 프레임워크는 점진적으로 정보가 드러나는 상황을 포착합니다. 이를 통해 매칭 결정이 미래 결과에 미치는 영향을 모델링합니다. 논문은 Learn2Match라는 동적 매칭 시장을 위한 다중 에이전트 강화 학습 벤치마크를 소개합니다.

- **Technical Details**: 이 프레임워크는 temporally extended feedback을 도입하여 불확실성과 비용이 발생하는 전후 매치 과정을 포함하는 부분 관찰 가능한 Markov 게임으로 2측 매칭을 모델링합니다. Learn2Match는 진화하는 잠재적 프로필과 복잡한 전후 의사결정, 노이즈가 있는 관찰을 통해 에이전트가 매칭 여부, 검색, 지속 여부 등을 결정해야 하는 환경을 제공합니다. 이 평가에서는 작용하는 선호와 진정한 잠재적 선호 간의 격차를 나타내는 information-friction loss를 측정합니다.

- **Performance Highlights**: 실험 결과, 독립 PPO 알고리즘은 CA-ETC라는 반란형 벤치마크에 비해 누적 사회 복지(cumulative social welfare)를 높이고 누적 후회(cumulative regret)를 줄이는 데 성공하였습니다. 하지만 정보 마찰 손실(information-friction loss)은 여전히 더 높아, MARL이 매칭-밴디드 방법의 탐색 구조를 충분히 제공하지 못함을 드러냅니다. Learn2Match는 다음 세대의 매칭 시장 알고리즘 개발을 촉진하는 데 중요한 역할을 할 것으로 기대됩니다.



### Comparing Sentiment Contagion in AI-Agent and Human Social Networks: Evidence from MOLTBOOK (https://arxiv.org/abs/2606.06665)
Comments:
          8 pages without appendix

- **What's New**: 이 논문은 자율 AI 에이전트들이 상호작용하는 사회적 네트워크를 살펴보며, AI 전용 사회 네트워크에서 부정적인 감정이 어떻게 확산되는지에 대한 연구 결과를 제공합니다. MOLTBOOK이라는 AI 전용 소셜 네트워크를 분석한 결과, 부정적인 게시글이 더 많은 주목을 받지만, 그에 대한 댓글은 대개 중립으로 전환된다는 것을 보여주었습니다. 즉, AI 네트워크에서는 부정적 감정이 극복되는 것이 아닌, 부정적인 주목이 중립화되는 경향이 있음을 발견했습니다.

- **Technical Details**: 이 연구에서 사용된 MOLTBOOK 데이터셋은 자율 언어 모델 에이전트들로부터 생산된 2.9백만 개의 게시글과 1.5백만 개의 댓글을 포함하고 있습니다. 감정은 CardiffNLP Twitter-RoBERTa 감정 모델을 통해 라벨링되었으며, 부정적인 게시물은 더 많은 댓글을 유도하지만 댓글들은 주로 중립적입니다. 논문은 감정의 전이 비대칭성을 분석하고, 같은 날과 다음 날의 감정 상관관계를 비교하였습니다.

- **Performance Highlights**: 연구 결과는 AI 에이전트 네트워크가 인간 네트워크와는 다른 방식으로 감정 역학이 작동함을 보여줍니다. 부정적인 게시글은 관심을 끌지만, 댓글은 확실히 부정적인 감정을 증폭시키지 않고 오히려 중립화되는 경향이 있습니다. 이 연구는 AI 에이전트 시스템이 감정의 극단을 억제하는 양상을 나타내며, 이러한 시스템의 상호작용 구조가 그들의 집합적 행동에 크게 영향을 미칠 가능성이 있음을 시사합니다.



### OpenAgenet / OAN Yellow Paper: Technical Architecture for Trust-Governed Resource Identity and Discovery (https://arxiv.org/abs/2606.03163)
- **What's New**: 이번 논문에서는 OpenAgenet/OAN의 기술 아키텍처를 설명합니다. OAN은 개방형 에이전트 상호 연결 및 AI 자원 제품을 위한 프로토콜 중립적인 신뢰 계층(trust layer)을 제공합니다. 이 시스템은 자원 신원(identity) 및 탐색(discovery) 아키텍처를 정의하며, 에이전트 간 상호작용을 위한 다양한 프로토콜을 지원합니다.

- **Technical Details**: OAN은 자원 신원을 인증하는 과정과 탐색 아키텍처에 중점을 둡니다. 신원 검증은 등록, Root 수용, 패키지 배포 등 여러 단계에서 이루어지며, 신뢰할 수 있는 리소스를 중심으로 구성됩니다. OAN은 다양한 에이전트 프레임워크 및 상호작용 프로토콜을 지원하는 리소스 우선 모델(Resource First model) 접근 방식을 구현합니다.

- **Performance Highlights**: OAN 프로토콜 스택은 여러 프로필을 포함하며, 개별적으로 구현되고 다양한 운영자에 의해 조합될 수 있습니다. 신뢰 층은 비즈니스 데이터와 분리되며, 검증을 통해 신뢰할 수 있는 요청만을 허용합니다. 또한, 신뢰 구역(trust domain) 및 거버넌스(gov) 상태 관리가 통합되어 있어, Rus는 인프라의 신뢰성을 지속적으로 검증하고 유지합니다.



### OpenAgenet / OAN White Paper: Open Infrastructure for Trusted Agent Interconnection (https://arxiv.org/abs/2606.03161)
- **What's New**: OAN(OpenAgenet)은 신뢰할 수 있는 에이전트 연결을 위한 오픈 인프라 프로젝트로, 에이전트가 개방된 다중 운영자 네트워크에서 이동할 때 문제가 발생한다고 설명합니다. 이 프로젝트는 기존의 개별 응용 프로그램에서 벗어나 에이전트가 다른 에이전트를 안전하게 발견하고 선택하며 호출할 수 있도록 신원 확인, 거버넌스 상태, 발견 인증 등을 검증할 수 있는 방법을 제공합니다. OAN은 프로토콜 중립적인 신뢰 레이어로 설계되어 에이전트 상호작용 프로토콜이나 응용 프로그램 수준의 워크플로를 대체하지 않고 안전성을 보장합니다.

- **Technical Details**: OAN은 did:oan 기반의 자원 신원 등록, 거버넌스에 기반한 Root 수용, 신뢰할 수 있는 패키지 배포, 인증-aware Discovery, 서명된 호출 등을 정의하는 오픈 인프라 아키텍처를 갖추고 있습니다. 이 아키텍처는 에이전트의 아이덴티티, 생애 주기 거버넌스 및 신뢰 증거에 중점을 두며, 신뢰 모델의 차별성을 두고 있습니다. Root는 인프라의 정당성을 스스로 선택하는 단일한 지점이 아닌, 위임된 거버넌스를 기반으로 인프라 라이프사이클 결정을 출판하는 역할을 수행합니다.

- **Performance Highlights**: OAN은 신뢰할 수 있는 자원 등록 및 발견 인프라로서 에이전트 생태계의 협력을 가능하게 하며, 다양한 조직들이 단일 에이전트 런타임이나 프로토콜 없이 협력할 수 있도록 지원합니다. 에이전트 서비스, 기술, MCP 서버, Tool/API 리소스를 감지하고 검증할 수 있는 공통 신뢰 기반을 제공합니다. OAN의 가장 큰 기여는 연합된 인프라 거버넌스, 자원 신원, 검증 가능한 배포 및 사전 연결 검증을 결합한 것입니다.




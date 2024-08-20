New uploads on arXiv(cs.CL)

### Multilingual Needle in a Haystack: Investigating Long-Context Behavior of Multilingual Large Language Models (https://arxiv.org/abs/2408.10151)
- **What's New**: 이 논문은 다국어(Multilingual) 설정에서 대형 언어 모델(LLM)의 장기 컨텍스트(Long Context) 처리 능력을 체계적으로 평가하기 위해 MultiLingual Needle-in-a-Haystack (MLNeedle) 테스트를 소개합니다. 이는 정보 검색(Information Retrieval)의 문맥에서 LLM의 성능을 분석하는 첫 시도로, 다양한 언어와 문맥 내 위치에 따른 모델 성능의 변화를 평가하고 있습니다.

- **Technical Details**: MLNeedle 테스트는 LLM이 다국어 방해 텍스트(Haystack)에서 관련 정보를 찾는 능력(Needle)을 평가하는 것을 목표로 합니다. 논문은 LLM의 성능 평가를 위해 MLQA 데이터셋을 활용하며, 7개 언어에 걸쳐 질문-답변 쌍을 조사합니다. 실험에서는 Needle의 위치와 언어에 따라 성능이 어떻게 변화하는지를 체계적으로 분석합니다.

- **Performance Highlights**: 실험 결과, LLM의 성능은 언어와 Needle의 위치에 따라 크게 달라지는 것으로 나타났습니다. 특히, Needle이 영어 계열 언어가 아닌 경우 및 입력 컨텍스트의 중간에 위치할 때 성능이 가장 낮았습니다. 또한, 8K 토큰 이상의 컨텍스트 크기를 주장하는 일부 모델도 증가하는 컨텍스트 길이에 따라 만족스러운 성능을 보이지 않았습니다.



### Instruction Finetuning for Leaderboard Generation from Empirical AI Research (https://arxiv.org/abs/2408.10141)
- **What's New**: 이 연구는 사전 훈련된 대형 언어 모델(LLMs)의 명령 조정(instruction finetuning)을 통해 AI 연구 리더보드를 자동 생성할 수 있는 방법을 소개합니다. 이는 기존의 수동 커뮤니케이션 작업 방식에서 자동화된 LLM 기반 접근 방식으로 전환되며, (Task, Dataset, Metric, Score) 쿼드러플(quadrilples)을 추출하는 데 주력하고 있습니다.

- **Technical Details**: FLAN-T5 모델을 활용하여 LLM의 정보 추출 능력을 향상시킵니다. 이 연구는 LLM이 특정 지시를 정확히 해석하고 실행할 수 있는 능력을 개발하여 비구조화된 텍스트를 처리하는 방식을 혁신합니다. 또한 AI 연구의 도메인-특정 미세 조정을 촉진하기 위해 기존 NLI(Natural Language Inference) 시스템의 한계를 극복합니다.

- **Performance Highlights**: 이번 연구에서는 SOTA(Task State Of The Art) 작업 성능이 약 10% 향상되어 기존 NLI 기반 시스템에 비해 뛰어난 성능을 보였습니다. 이로 인해 우리는 모델의 실용성 및 효과성을 입증하게 되었습니다.



### Rhyme-aware Chinese lyric generator based on GP (https://arxiv.org/abs/2408.10130)
- **What's New**: 본 논문에서는 기존의 사전 훈련된 언어 모델이 가사 생성 시 운율(rhyme) 정보를 주로 고려하지 않는 문제를 다룹니다. 이를 해결하기 위해, 운율 정보를 통합하여 성능을 향상시키는 방안을 제시합니다.

- **Technical Details**: 제안하는 모델은 사전 훈련(pre-trained)된 언어 모델을 기반으로 하며, 운율 통합(integrated rhyme) 기법을 적용하여 가사 생성 성능을 개선합니다. 이 모델은 대규모 코퍼스에 대해 사전 훈련되어 있어, 풍부한 의미적 패턴을 효과적으로 포착할 수 있습니다.

- **Performance Highlights**: 운율 정보를 통합한 모델을 사용하여 생성된 가사의 질이 현저하게 향상되었습니다. 이는 기존의 모델보다 우수한 자연어 생성 성능을 보여줍니다.



### GLIMMER: Incorporating Graph and Lexical Features in Unsupervised Multi-Document Summarization (https://arxiv.org/abs/2408.10115)
Comments:
          19 pages, 7 figures. Accepted by ECAI 2024

- **What's New**: GLIMMER는 기존의 신경망 기반 모델과 달리, 대규모 데이터셋 없이도 작동할 수 있는 가벼운 비지도 학습 다문서 요약 접근법이다. 이 모델은 문장 그래프를 구성하고, 저수준의 어휘적 특성을 이용하여 의미적 클러스터를 자동으로 식별한다.

- **Technical Details**: GLIMMER는 문서 집합에서 문장 그래프를 먼저 구성한 후, 저수준의 어휘적 특징을 통해 클러스터 내 상관성을 높이고, 생성된 문장의 유창성을 향상시킨다. 실험을 통해 Multi-News, Multi-XScience, DUC-2004 데이터셋에서 기존 비지도 접근법 및 최신 사전 학습된 모델들과 비교했을 때 우수한 성능을 입증하였다.

- **Performance Highlights**: GLIMMER의 요약은 ROUGE 스코어에서 PEGASUS 및 PRIMERA와 같은 최신 다문서 요약 모델을 초월하였으며, 인간 평가에서 높은 가독성과 정보성을 기록하였다.



### Privacy Checklist: Privacy Violation Detection Grounding on Contextual Integrity Theory (https://arxiv.org/abs/2408.10053)
- **What's New**: 이 논문은 개인 정보 보호(Privacy)의 문제를 간단한 패턴 매칭이 아닌 추론 문제로 새롭게 정의합니다. 기존의 연구들이 개별 분야의 사각지대에 갇혀 있는 반면, 이 연구는 Contextual Integrity(CI) 이론에 기반하여 개인 정보 보호를 더욱 포괄적으로 접근합니다.

- **Technical Details**: 논문에서는 개인 정보를 평가하기 위해 Privacy Checklist라는 도구를 제안합니다. 이는 대규모 언어 모델(LLMs)을 활용하여 기존의 규정을 기반으로 세부적인 정보 전송 규범을 추출하고, 계층 구조를 활용하여 이들 규칙을 구조화합니다. 또한, 역할 기반(role-based)과 속성 기반(attribute-based)의 그래프를 포함합니다. 이를 통해 HIPAA(Health Insurance Portability and Accountability Act of 1996)의 규정을 활용하여 개인 정보 위반 사례를 평가합니다.

- **Performance Highlights**: 제안된 Privacy Checklist는 기존 LLM의 개인 정보 판단 능력을 6%에서 18% 향상시키며, 실제 법원 사례를 통해 개인 정보 규제 준수 여부를 평가할 때 효과적임을 입증합니다.



### Benchmarking LLMs for Translating Classical Chinese Poetry:Evaluating Adequacy, Fluency, and Eleganc (https://arxiv.org/abs/2408.09945)
Comments:
          Work in progress

- **What's New**: 이번 연구는 고전 중국 시를 영어로 번역하는 새로운 벤치마크를 소개하며, 대형 언어 모델(LLM)이 요구되는 번역의 적합성, 유창성, 우아함을 충족시키지 못함을 밝혀냈습니다. 이를 개선하기 위해 RAT(Recovery-Augmented Translation) 방법론을 제안하였습니다.

- **Technical Details**: RAT는 고전 시와 관련된 지식을 검색하여 번역 품질을 향상시키는 방법입니다. 연구에서는 GPT-4를 기반으로 한 새로운 자동 평가 지표를 도입하여 번역의 적합성, 유창성, 우아함을 평가합니다. 데이터셋은 1200개의 고전 시와 608개의 수동 번역으로 구성됩니다.

- **Performance Highlights**: RAT 방법은 번역 과정에서 고전 시와 관련된 knowledge를 활용하여 번역의 품질을 개선시켜주며, 평가 결과는 기존의 자동 평가 방식보다 LLM 기반 번역의 성능을 더 잘 측정합니다.



### "Image, Tell me your story!" Predicting the original meta-context of visual misinformation (https://arxiv.org/abs/2408.09939)
Comments:
          Preprint. Code available at this https URL

- **What's New**: 이번 연구에서는 시각적 허위정보를 자동으로 탐지하기 위한 접근으로서, 이미지의 원래 메타-문맥(meta-context)을 확인하는 자동화된 이미지 맥락화(image contextualization) 작업을 도입합니다. 이를 통해 사실 확인자들이 정보 왜곡을 보다 효과적으로 탐지할 수 있도록 지원합니다.

- **Technical Details**: 연구팀은 1,676개의 사실 확인된 이미지와 그에 대한 질문-답변 쌍으로 구성된 5Pils 데이터셋을 생성하였으며, 이는 5 Pillars 사실 확인 프레임워크에 기반합니다. 이 기법을 통해 이미지를 원래의 메타-문맥에 연결하는 첫 번째 기준선을 구현하였고, 웹에서 수집한 텍스트 증거를 활용하였습니다.

- **Performance Highlights**: 실험 결과, 자동화된 이미지 맥락화 작업은 여러 도전 과제를 드러내며 SOTA (State of the Art) 대형 언어 모델(LLMs)에게도 도전적인 결과를 보였습니다. 연구팀은 코드와 데이터를 공개할 예정입니다.



### Active Learning for Identifying Disaster-Related Tweets: A Comparison with Keyword Filtering and Generic Fine-Tuning (https://arxiv.org/abs/2408.09914)
Comments:
          Submitted for the Intelligent Systems Conference (IntelliSys 2024). The version of record of this contribution is published in the Springer series Lecture Notes in Networks and Systems, and is available online at this https URL. This preprint has not undergone peer review or any post-submission improvements or corrections. 13 pages, 2 figures

- **What's New**: 본 연구는 자연재해와 관련된 트윗을 식별하기 위한 Active Learning(AL) 기법의 활용 가능성을 조사합니다. 기존의 키워드 필터링 기법 및 RoBERTa 모델과 비교하여 AL 방식이 뛰어난 성능을 보임을 입증하였습니다.

- **Technical Details**: 트윗 분류 성능을 비교하기 위해 CrisisLex 데이터와 2021년 독일 홍수 및 2023년 칠레 산불의 수작업으로 라벨 된 데이터를 사용했습니다. AL 기법은 대량의 비라벨 데이터 중에서 모델이 직접 라벨을 할 샘플을 선택하여 라벨링 효율성을 높였습니다.

- **Performance Highlights**: 일반 데이터셋으로 10회 AL을 결합한 모델이 다른 모든 접근 방식을 능가했습니다. 이로 인해 최소한의 라벨링 비용으로도 재해 관련 트윗 식별에 대한 유용한 모델을 구축할 수 있음을 확인했습니다.



### Performance Law of Large Language Models (https://arxiv.org/abs/2408.09895)
Comments:
          Personal opinions of the authors

- **What's New**: 본 논문에서는 LLM(대형 언어 모델)의 성능을 예측하기 위한 새로운 경험적 방정식인 'Performance Law'를 제안합니다. 이 방정식은 LLM의 일반적인 역량을 평가하기 위해 MMLU(Multi-choice Machine Learning Understandability) 점수를 직접적으로 예측합니다. 또한, 다양한 LLM 아키텍처와 데이터 양을 고려하여 LLM의 성능을 보다 정확하게 추정합니다.

- **Technical Details**: Performance Law는 다음과 같은 주요 변수들에 기반합니다: 레이어 수(N), 히든 사이즈(h), FFN(Feed-Forward Network)의 인터미디어 사이즈(d), 트레이닝 데이터의 크기(T), 모델 파라미터의 크기(S). 이 방정식은 기존의 스케일링 법칙을 바탕으로 LLM의 성능을 예측하며, 특히 다양한 모델 구조와 형태에 대한 성능을 잘 설명합니다. Mixture-of-expert (MoE) 모델의 경우, 활성화된 파라미터의 수(A)를 추가적으로 고려해야 합니다.

- **Performance Highlights**: 이 연구를 통해 제안된 Performance Law는 2020년부터 2024년까지 다양한 LLM의 MMLU 점수를 놀랍도록 정확하게 예측할 수 있게 해줍니다. 특히, 단 몇 개의 주요 하이퍼파라미터와 훈련 데이터의 양만으로도 다양한 사이즈와 아키텍처의 모델들의 성능 예측이 가능하다는 점에서 연구의 시사점이 큽니다. 이 법칙은 LLM 아키텍처의 선택과 컴퓨팅 자원 할당을 효율적으로 가이드할 수 있어, 불필요한 실험 없이도 자원 낭비를 줄이는 데 기여합니다.



### Docling Technical Repor (https://arxiv.org/abs/2408.09869)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2206.01062

- **What's New**: 이번 기술 보고서는 PDF 문서 변환을 위한 간편한 오픈 소스 패키지인 Docling을 소개합니다. 이는 레이아웃 분석(layout analysis)을 위한 AI 모델인 DocLayNet과 테이블 구조 인식(table structure recognition)을 위한 모델인 TableFormer로 구동되며, 일반 하드웨어에서 효율적으로 실행됩니다.

- **Technical Details**: Docling은 사용자가 PDF 문서를 JSON 또는 Markdown 형식으로 변환하고, 세부 페이지 레이아웃을 이해하며, 그림을 찾고 테이블 구조를 복구하는 기능을 제공합니다. 또한 메타데이터(예: 제목, 저자, 참고문헌, 언어)를 추출할 수 있으며, OCR 기능을 선택적으로 적용할 수 있습니다.

- **Performance Highlights**: Docling은 배치 모드(높은 처리량, 낮은 해결 시간)와 인터랙티브 모드(효율성 타협, 낮은 해결 시간)에 최적화할 수 있으며, 다양한 가속기(GPU, MPS 등)를 활용할 수 있습니다.



### TaSL: Continual Dialog State Tracking via Task Skill Localization and Consolidation (https://arxiv.org/abs/2408.09857)
Comments:
          Accepted to ACL 2024 Main Conference

- **What's New**: TaSL(태스크 스킬 로컬라이제이션 및 통합)이라는 새로운 프레임워크를 제안하며, 대화 시스템의 연속 대화 상태 추적(Continual Dialogue State Tracking) 문제를 해결하기 위해 기억 재생(memor replay)에 의존하지 않고 지식을 효과적으로 전이할 수 있는 방법을 제공합니다.

- **Technical Details**: TaSL은 그룹 단위의 중요성 인식 스킬 로컬라이제이션(group-wise importance-aware skill localization) 기법을 사용하여 모델 파라미터의 중요 분포를 식별하고, 과거 작업의 지식와 현재 작업의 지식을 카테고리별로 통합합니다. 이 과정에서 각 스킬 유닛의 중요성을 정량화하여 작업 간 지식 전이가 원활하게 이루어지도록 합니다. 또한, 정교한 스킬 통합(fine-grained skill consolidation) 전략을 통해 과거 작업의 특정 지식이 잊히지 않도록 보호합니다.

- **Performance Highlights**: TaSL은 다양한 백본 모델에서 우수한 성능을 보이며, 기존 SOTA(State-Of-The-Art) 방법에 비해 Avg. JGA(공통 대화 목표 도달률)에서 3.1%의 절대적인 증가를, BWT(후진 전이) 메트릭에서는 8.8%의 절대적인 향상을 보여줍니다.



### TeamLoRA: Boosting Low-Rank Adaptation with Expert Collaboration and Competition (https://arxiv.org/abs/2408.09856)
- **What's New**: TeamLoRA는 Parameter-Efficient Fine-Tuning (PEFT) 방식의 새로운 접근법으로, 효율성과 효과성을 동시에 개선하기 위해 전문가 간 협업 및 경쟁 모듈을 도입합니다.

- **Technical Details**: TeamLoRA는 두 가지 주요 구성 요소로 이루어져 있습니다: (i) 효율적 협업 모듈에서는 비대칭 매트릭스 아키텍처를 활용하여 A와 B 매트릭스 간의 지식 공유 및 조직을 최적화합니다; (ii) 경쟁 모듈은 게임 이론에 기반한 상호작용 메커니즘을 통해 특정 다운스트림 작업에 대한 도메인 지식 전달을 촉진합니다.

- **Performance Highlights**: 실험 결과, TeamLoRA는 기존의 MoE-LoRA 방식보다 더 높은 성능과 효율성을 보이며, 2.5백만 샘플로 구성된 다양한 도메인과 작업 유형을 포함하는 종합적인 평가 기준(CME)에서 그 효과성을 입증했습니다.



### Self-Directed Turing Test for Large Language Models (https://arxiv.org/abs/2408.09853)
- **What's New**: 본 연구는 전통적인 Turing 테스트의 단점을 해결하기 위해 Self-Directed Turing Test라는 새로운 프레임워크를 제안합니다. 이 테스트는 다중 연속 메시지를 통한 더 역동적인 대화를 허용하며, LLM이 대화의 대부분을 스스로 지시할 수 있도록 설계되었습니다.

- **Technical Details**: Self-Directed Turing Test는 burst dialogue 형식을 통해 자연스러운 인간 대화를 보다 잘 반영합니다. 이 과정에서 LLM은 대화의 진행을 스스로 생성하고, 마지막 대화 턴에 대한 인간과의 짧은 대화를 통해 평가합니다. 새롭게 도입된 X-Turn Pass-Rate 메트릭은 LLM의 인간 유사성을 평가하기 위한 기준이 됩니다.

- **Performance Highlights**: 초기에는 GPT-4와 같은 LLM들이 3회 대화 턴에서 51.9%, 10회에서는 38.9%의 수치로 테스트를 통과했지만 대화가 진행됨에 따라 성능이 하락하는 경향을 보였으며, 이는 장기 대화에서 일관성을 유지하는 것이 어렵다는 것을 강조합니다.



### Importance Weighting Can Help Large Language Models Self-Improv (https://arxiv.org/abs/2408.09849)
- **What's New**: 이 논문에서는 LLM(self-generated data)의 self-improvement를 위해 DS weight라는 새로운 메트릭(또는 척도)을 제안하여 모델 성능 향상을 위한 중요한 필터링 전략을 제시합니다.

- **Technical Details**: DS weight는 LLM의 데이터 분포 변화 정도(Distribution Shift Extent, DSE)를 근사하는 메트릭으로, Importance Weighting 방법에서 영감을 받아 개발되었습니다. 이 메트릭을 활용하여 데이터 필터링 전략을 구축하고, self-consistency와 결합하여 최신 LLM을 fine-tune 합니다.

- **Performance Highlights**: 제안된 접근 방식인 IWSI(Importance Weighting-based Self-Improvement)는 적은 양의 유효 데이터(Training Set의 5% 이하)를 사용하더라도 현재 LLM self-improvement 방법들의 추론 능력을 상당히 향상시키며, 이는 기존의 외부 감독(pre-trained reward models)을 통한 성능과 동등한 수준입니다.



### Continual Dialogue State Tracking via Reason-of-Select Distillation (https://arxiv.org/abs/2408.09846)
Comments:
          Accepted to ACL 2024 Findings

- **What's New**: 이 논문에서는 대화 상태 추적(Dialogue State Tracking, DST) 문제를 해결하기 위한 Reason-of-Select (RoS) 증류 방법을 소개합니다. 이 방법은 새로운 서비스에 적응하면서 이전 지식을 유지하는 연속 학습(Continual Learning, CL)의 어려움을 극복하기 위해 새로운 '메타-추론(meta-reasoning)' 기능을 가진 소형 모델을 강화합니다.

- **Technical Details**: RoS 증류 방법은 메타-추론 능력을 향상시켜, DSA의 계층적 쿼리와 복잡한 대화 관리를 가능하게 합니다. 이 과정에서 다중 값 해결(multi-value resolution) 전략과 의미적 대조 추론 선택(Semantic Contrastive Reasoning Selection) 방법이 도입됩니다. 이를 통해 DST-specific 선택 체인을 생성하고, 모델의 비현실적인 추론을 감소시킵니다.

- **Performance Highlights**: 광범위한 실험을 통해 RoS 방법이 탁월한 DST 성능과 강력한 일반화 능력을 입증했습니다. 제공되는 소스 코드는 재현 가능성을 보장합니다.



### CMoralEval: A Moral Evaluation Benchmark for Chinese Large Language Models (https://arxiv.org/abs/2408.09819)
Comments:
          Accepted by ACL 2024 (Findings)

- **What's New**: 본 연구에서는 CMoralEval이라는 대규모 벤치마크 데이터셋을 제안하여 중국어 대형 언어 모델(LLM)의 도덕성 평가를 위한 자료를 제공합니다. 이 데이터셋은 중국 사회의 도덕 규범을 기반으로 다양한 도덕적 상황을 다룹니다.

- **Technical Details**: CMoralEval은 1) 중국 TV 프로그램을 통해 수집된 도덕적 이야기 및 2) 신문 및 학술지에서 수집된 중국 도덕 불일치 데이터를 포함하여 30,388개의 도덕 사례를 포함합니다. 사례는 가족 도덕성, 사회 도덕성, 직업 윤리, 인터넷 윤리 및 개인 도덕성의 다섯 가지 범주로 구분됩니다.

- **Performance Highlights**: CMoralEval에 대한 실험 결과, 다양한 중국어 LLMs가 평가되었으며, 이는 이 데이터셋이 도덕성 평가에 있어 도전적인 벤치마크임을 보여줍니다. 이 데이터셋은 공개적으로 제공됩니다.



### Anim-Director: A Large Multimodal Model Powered Agent for Controllable Animation Video Generation (https://arxiv.org/abs/2408.09787)
Comments:
          Accepted by SIGGRAPH Asia 2024, Project and Codes: this https URL

- **What's New**: 이번 연구에서는 LMMs(large multimodal models)를 활용하여 애니메이션 제작을 자동화하는 새로운 방식의 에이전트인 Anim-Director를 소개합니다. 이_agent_는 사용자로부터 간결한 내러티브나 지시 사항을 받아 일관성 있는 애니메이션 비디오를 생성할 수 있도록 설계되었습니다.

- **Technical Details**: Anim-Director는 세 가지 주요 단계로 작동합니다: 첫 번째로, 사용자 입력을 바탕으로 일관된 스토리를 생성하고, 두 번째 단계에서는 LMMs와 이미지 생성 도구를 사용하여 장면의 비주얼 이미지를 생산합니다. 마지막으로, 이러한 이미지들은 애니메이션 비디오 제작의 기초로 활용되며, LMMs가 각각의 프로세스를 안내하는 프롬프트를 생성합니다.

- **Performance Highlights**: 실험 결과, Anim-Director는 개선된 스토리라인과 배경의 일관성을 유지하며 긴 애니메이션 비디오를 생성할 수 있음을 증명했습니다. LMMs와 생성 도구의 통합을 통해 애니메이션 제작 프로세스를 크게 간소화하여, 창조적인 과정의 효율성과 비디오 품질을 향상시키는 데 기여합니다.



### Summarizing long regulatory documents with a multi-step pipelin (https://arxiv.org/abs/2408.09777)
Comments:
          Under review

- **What's New**: 이번 논문은 긴 규제 문서를 효과적으로 요약하기 위한 다단계 추출-추상 구조를 제안합니다. 모델에 따라 두 단계 아키텍처의 성능이 크게 다르다는 사실을 보여줍니다.

- **Technical Details**: 이 연구에서는 긴 문서 요약을 위해 문서를 작은 ‘청크(chunk)’로 나누고 각 청크를 추출적 요약 모델로 처리하며, 그 후 결과 요약을 결합하여 최종 요약을 만드는 과정이 포함됩니다. 이 접근 방식은 추출적 및 추상적 기법을 결합하여 긴 텍스트를 처리하는 데 유용할 수 있습니다.

- **Performance Highlights**: 인간 평가에서 법률 텍스트에 대해 사전 학습된 언어 모델이 높은 점수를 받았으나, 자동 평가에서는 일반 목적 언어 모델이 더 높은 점수를 기록했습니다. 이는 요약 전략 선택이 모델 아키텍처 및 맥락 길이에 따라 얼마나 중요한지를 강조합니다.



### Are Large Language Models More Honest in Their Probabilistic or Verbalized Confidence? (https://arxiv.org/abs/2408.09773)
- **What's New**: 이번 연구에서는 대형 언어 모델 (LLMs)이 자사 지식 경계를 어떻게 인식하는지를 분석하고, 확률적 인식 (probabilistic perception)과 언어화된 인식 (verbalized perception) 간의 차이와 상관관계를 살펴봅니다.

- **Technical Details**: 연구에서는 LLMs의 확률적 신뢰도와 언어화된 신뢰도를 비교하며, 다양한 질문 빈도가 이들 신뢰도에 미치는 영향을 조사합니다. 네 가지 널리 사용되는 LLM을 대상으로 Natural Questions와 Parent-Child 데이터셋에서 실험을 진행했습니다.

- **Performance Highlights**: 실험 결과, LLMs의 확률적 인식이 언어화된 인식보다 일반적으로 더 정확하며, 두 인식 모두 드문 질문에서 더 좋은 성능을 보였습니다. 그러나 LLMs가 자연어로 내재적 신뢰도를 정확하게 표현하는 것은 어려웠습니다.



### Paired Completion: Flexible Quantification of Issue-framing at Scale with LLMs (https://arxiv.org/abs/2408.09742)
Comments:
          9 pages, 4 figures

- **What's New**: 이번 연구에서는 소수의 예시를 이용하여 이슈 프레이밍(issue framing)과 내러티브 분석(narrative analysis)을 효과적으로 탐지할 수 있는 새로운 방법인 'paired completion'을 개발했습니다. 이는 기존의 자연어 처리(NLP) 접근법보다 더 정확하고 효율적입니다.

- **Technical Details**: 'Paired completion' 방식은 생성형 대형 언어 모델(generative large language models)에서 유래된 다음 토큰 로그 확률(next-token log probabilities)을 활용하여, 특정 이슈에 대한 텍스트의 프레이밍 정렬 여부를 판별합니다. 이 방법은 몇 개의 예시만으로도 높은 정확성을 제공합니다.

- **Performance Highlights**: 192개의 독립 실험을 통해, 'paired completion' 방법이 기존의 프롬프트 기반 방법(prompt-based methods) 및 전통적인 NLP 방식보다 우수한 성능을 보인다는 것을 증명했습니다. 특히, 적은 양의 데이터 환경에서도 높은 성과를 달성했습니다.



### SEMDR: A Semantic-Aware Dual Encoder Model for Legal Judgment Prediction with Legal Clue Tracing (https://arxiv.org/abs/2408.09717)
- **What's New**: 이 논문에서는 Semantic-Aware Dual Encoder Model (SEMDR)을 제안하여 법적 판단 예측의 정확성을 높이고, 유사한 범죄 간의 미세한 차이를 구별하는 법적 단서 추적 메커니즘을 설계합니다.

- **Technical Details**: SEMDR은 세 가지 수준(1. Lexicon-Tracing, 2. Sentence Representation Learning, 3. Multi-Fact Reasoning)의 추론 체계를 통해 범죄 사실과 범죄 도구 간의 세밀한 의미론적 추론을 수행합니다. 각 수준은 범죄 설명에서 의식을 추출하고 혼란스러운 범죄 사실을 더 잘 표현하기 위해 언어 모델을 대조적으로 훈련하며, 범죄 사실 노드 간의 의미적 단서를 전달하는 이유 그래프를 구축합니다.

- **Performance Highlights**: SEMDR은 CAIL2018 데이터셋에서 최첨단 성능을 달성하고 몇 가지 예제에 대한 적응력이 높습니다. 실험 결과 SEMDR은 범죄 사실에 대한 더 균일하고 구별된 표현을 학습하여 혼란스러운 범죄 사례에 대해 보다 정확한 예측을 가능하게 하며, 판단 시 모델의 불확실성을 감소시킵니다.



### Bridging the Language Gap: Enhancing Multilingual Prompt-Based Code Generation in LLMs via Zero-Shot Cross-Lingual Transfer (https://arxiv.org/abs/2408.09701)
Comments:
          Under Review

- **What's New**: 이번 연구는 다국어 프롬프트 기반 프로그램 코드 생성에서의 복잡성을 탐구하며, 비영어 프롬프트에 대한 LLM의 성능 차이를 집중적으로 분석했습니다. 연구에서 소개된 제로샷 크로스링구얼(zero-shot cross-lingual) 접근 방식은 LASER 멀티언어 인코더를 이용하여 다양한 언어의 임베딩을 LLM의 토큰 공간으로 매핑하는 혁신적인 기술입니다.

- **Technical Details**: 제안된 방법은 사전학습된 LASER 인코더를 사용하여 여러 언어의 입력을 공통 벡터 공간으로 인코딩하고, 이를 LLM의 입력 공간으로 프로젝션(projection)하여 크로스링구얼 처리에서 성능을 향상시킵니다. 이 프로세스는 영어 데이터에 대해서만 훈련되고 추가적인 외부 데이터 없이 다국어 처리를 가능하게 합니다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식이 번역 및 품질 검증이 완료된 MBPP 데이터셋에서 코드 품질을 획기적으로 개선하는 것으로 나타났습니다. 이는 LLM들이 다국어 기능을 강화하고 다양한 언어적 스펙트럼을 지원하는 데 중요한 기여를 할 것으로 기대됩니다.



### Recording for Eyes, Not Echoing to Ears: Contextualized Spoken-to-Written Conversion of ASR Transcripts (https://arxiv.org/abs/2408.09688)
Comments:
          7 pages, 3 figures

- **What's New**: 최근 자동음성인식(ASR) 전사물의 가독성을 향상시키기 위한 새로운 접근 방식을 제안하는 연구가 진행되었습니다. 이를 위해, 비공식 텍스트를 공식적인 스타일로 변환하는 Contextualized Spoken-to-Written conversion (CoS2W) 과제가 도입되었습니다. 이 과제는 ASR 오류와 문법 오류를 처리하면서 내용은 보존합니다.

- **Technical Details**: CoS2W는 ASR 오류 수정(ASR error correction), 문법 오류 수정(Grammatical Error Correction, GEC), 텍스트 스타일 전환(text style transfer) 등의 여러 하위 작업을 결합하여 작동합니다. 이 연구에서는 다양한 LLMs(대형 언어 모델)의 성능을 비교하기 위해 ASR 전사물 문서 수준의 Spoken-to-Written conversion Benchmark (SWAB) 데이터셋을 구축하였습니다. 실험을 통해 문맥과 보조 정보를 활용하는 방법을 제안하였으며, CoS2W 성능에 미치는 다양한 데이터의 세분화 수준(granularity level)의 영향을 조사했습니다.

- **Performance Highlights**: 실험 결과, LLMs는 CoS2W 과제에서 특히 문법성과 공식성 측면에서 우수한 성능을 보여주었으며, LLM 평가자가 인간 평가와 높은 상관관계를 보이는 것을 발견했습니다. ASR 전사물에 CoS2W를 적용한 경우, 기계 번역 성능이 BLEURT 점수에서 57.07에서 61.9로 향상되는 등의 성과를 보였습니다.



### BLADE: Benchmarking Language Model Agents for Data-Driven Scienc (https://arxiv.org/abs/2408.09667)
- **What's New**: 이번 연구에서는 데이터 기반 과학 분석을 평가하기 위한 새로운 벤치마크인 BLADE를 소개합니다. BLADE는 과학 문헌에서 발췌한 12개의 데이터셋과 연구 질문으로 구성되어 있으며, 데이터 과학자 및 연구자들의 독립 분석을 통해 수집된 정답(ground truth)을 기반으로 합니다.

- **Technical Details**: BLADE는 멀티 단계 분석 과정에서 다양한 요인을 고려하여 분석 결정을 평가합니다. 이 벤치마크는 데이터 세미antics(semantics), 도메인 지식(domain knowledge), 그리고 다양한 분석 방법을 통합하여 과학 연구 질문에 대한 대답을 도출하는 분석 능력을 측정합니다. 또한, 자동 평가(computational methods)를 통해 에이전트의 응답을 정량적으로 분석합니다.

- **Performance Highlights**: 언어 모델(LM)은 기본적인 분석을 수행하는 데는 적합하지만, 복잡한 분석에서는 다양성 부족이 나타났습니다. 반면, ReAct 에이전트는 점진적인 개선을 보여주었으나 여전히 최적의 성과에는 미치지 못했습니다. BLADE는 에이전트의 의사결정 능력을 평가하는 데 중요한 기초 데이터를 제공하며, 향후 연구의 진전을 도울 것입니다.



### Acquiring Bidirectionality via Large and Small Language Models (https://arxiv.org/abs/2408.09640)
- **What's New**: 이 연구에서는 기존의 정방향 언어 모델(이하 UniLM)을 대신하여 새로운 역방향 언어 모델(이하 backward LM)을 새롭게 훈련시키고, 이를 기존의 UniLM에 연결(concatenate)하여 pseudo bidirectionality를 달성하는 방안을 제안합니다. 이러한 방식이 named entity recognition(NER) 성능을 10포인트 이상 향상시킬 수 있음을 실험을 통해 입증합니다.

- **Technical Details**: 본 연구는 token-classification 작업에 있어 UniLM과 BiLM(양방향 언어 모델) 사이의 차이를 분석합니다. UniLM은 이전의 문맥만을 기반으로 단어의 표현을 계산하는 반면, BiLM은 양쪽의 문맥을 모두 고려합니다. 제안된 방법은 역방향 LM의 representations를 기존의 UniLM representations에 연결하여 높은 정확도의 token-level classification 작업을 위한 효율성을 제공합니다.

- **Performance Highlights**: 실험 결과, backward LM을 추가함으로써 NER 벤치마크 성능이 10포인트 이상 향상되었으며, 특히 희귀 도메인이나 few-shot 학습 환경에서도 일관된 성능 개선을 보여 주었습니다.



### How to Make the Most of LLMs' Grammatical Knowledge for Acceptability Judgments (https://arxiv.org/abs/2408.09639)
- **What's New**: 이번 연구는 언어 모델(LLM)의 문법적 지식을 포괄적으로 평가하기 위해 다양한 판단 방법을 조사합니다. 특히, 기존의 확률 기반 기법을 넘어서, in-template LP와 Yes/No 확률 계산과 같은 새로운 방법들이 높은 성능을 보임을 강조합니다.

- **Technical Details**: 연구에서는 세 가지 다른 그룹의 방법을 통해 LLM에서의 수용성 판단을 추출하는 실험을 진행했습니다. 확률 기반 셉방식(세 문장 쌍에 대한 확률을 계산), in-template 방식, 그리고 프롬프팅 기반 방식(Yes/No prob comp)을 비교하였습니다. in-template 방식에서는 문장을 템플릿에 삽입하여 문법성에 집중하도록 지시했습니다.

- **Performance Highlights**: in-template LP 및 Yes/No prob comp가 특히 높은 성능을 보였으며, 기존의 방법들을 초월했습니다. Yes/No prob comp는 길이 편향에 강한 강점을 가지고 있습니다. 통합 사용 시 두 방법은 서로의 보완적인 능력을 보여주며, Mix-P3 방법이 인간보다 1.6% 포인트 높은 정확도를 기록했습니다.



### A Strategy to Combine 1stGen Transformers and Open LLMs for Automatic Text Classification (https://arxiv.org/abs/2408.09629)
Comments:
          13 pages, 3 figures, 8 tables

- **What's New**: 본 연구에서는 1세대 Transformer 모델(BERT, RoBERTa, BART)과 두 가지 오픈 LLM(Llama 2, Bloom)의 성능을 11개의 감정 분석 데이터 세트를 통해 비교하였습니다. 결과적으로, 오픈 LLM은 미세 조정(fine-tuning)을 수행할 경우, 8개 데이터 세트에서 1세대 Transformer보다 중간 수준의 성능 향상을 보이며, 더 높은 비용을 요구함을 발견하였습니다.

- **Technical Details**: 연구에서는 1세대 Transformer 모델과 오픈 LLM의 조합을 통해 새로운 신뢰 기반 전략인 'Call My Big Sibling' (CMBS)을 제안합니다. 이 전략은 높은 신뢰도를 가진 문서를 1세대 Transformer 모델로 분류하고, 불확실한 경우 LLM을 활용하여 분류합니다. 이를 통해 전반적인 비용을 절감하면서 성능을 최적화할 수 있습니다.

- **Performance Highlights**: CMBS 전략은 11개 데이터 세트 중 7개에서 1세대 Transformer의 성능을 초과하였으며, 나머지 4개에서는 동점을 기록했습니다. 또한, CMBS는 미세 조정된 LLM에 비해 평균 1/13의 비용으로 더 나은 성능을 보였습니다.



### Refining Packing and Shuffling Strategies for Enhanced Performance in Generative Language Models (https://arxiv.org/abs/2408.09621)
Comments:
          11 pages (include appendix), 26 figures, submitted to ACL ARR Aug 2024

- **What's New**: 이 논문은 오토 리그레시브 언어 모델 훈련에서 데이터 패킹과 섞기(token packing and shuffling) 방법의 차이를 분석하여 적절한 데이터 패킹 전략을 비교하고자 하였습니다. 특히, 최대 시퀀스 길이(Maximum Sequence Length, MSL)에 맞춘 '아톰 크기(atom size)' 설정이 모델 성능에 미치는 영향을 확인했습니다.

- **Technical Details**: 연구에서는 GPT-2 124M 모델을 WikiText 데이터셋으로 사전 훈련하였으며, 패딩(padding) 및 연결(concatenation) 방법을 사용하여 다양한 아톰 크기와 MSL을 실험했습니다. 연구 결과, 아톰 크기를 MSL로 맞출 경우 두 가지 패킹 방법이 모두 성능을 최적화하는 것을 확인하였습니다. 또한, 패딩 방법이 연결 방법보다 낮은 perplexity(퍼플렉시티)를 달성하지만, 훈련 단계가 더 필요하다는 점을 밝혔습니다.

- **Performance Highlights**: 연구 결과, 아톰 크기를 MSL로 설정할 경우 연결 및 패딩 방법 모두 최적의 성능을 보였고, 패딩 방법이 더 나은 모델 성능을 기록했습니다. 패딩 모델은 최종 perplexity 102.82, 연결 모델은 118.08로 보고되었습니다. 패딩 방법은 더 많은 훈련 단계를 요구하지만, 최종 성능은 더 높았습니다.



### Grammatical Error Feedback: An Implicit Evaluation Approach (https://arxiv.org/abs/2408.09565)
- **What's New**: 본 연구에서는 Grammatical Error Feedback (GEF)라는 새로운 피드백 방식을 제안합니다. 이는 기존의 Grammatical Error Correction (GEC) 시스템의 수동 피드백 주석 없이도 GEF를 제공할 수 있는 암시적 평가 접근 방식입니다.

- **Technical Details**: 제안된 방법은 'grammatical lineup' 접근 방식을 이용하여 피드백과 에세이 표현을 적절히 짝짓는 작업을 수행합니다. 대규모 언어 모델 (LLM)을 활용하여 피드백과 에세이를 매칭하며, 이 과정에서 선택된 대안의 질인 'foils' 생성이 중요한 요소입니다.

- **Performance Highlights**: 실험 결과, 새로운 GEF 생성 체계는 기존의 GEC 시스템에 비해 더 포괄적이고 해석 가능한 피드백을 제공함으로써 L2 학습자들에게 더 효과적일 수 있음을 보여주었습니다.



### HiAgent: Hierarchical Working Memory Management for Solving Long-Horizon Agent Tasks with Large Language Mod (https://arxiv.org/abs/2408.09559)
Comments:
          Project Page: this https URL

- **What's New**: 이 논문에서는 LLM(대형 언어 모델) 기반 에이전트의 작업 메모리 향상을 위한 새로운 프레임워크 HiAgent를 제안합니다. HiAgent는 서브골(subgoal)을 사용하여 에이전트의 작업 메모리를 관리하며, 이를 통해 실행 가능한 행동을 생성하기 전에 LLM이 서브골을 공식화하도록 유도합니다.

- **Technical Details**: HiAgent는 LLM의 작업 메모리를 계층적으로 관리하는 방법론으로, 사용자 발문의 역사적 행동-관찰 쌍을 직접 입력하는 기존 접근 방식을 개선합니다. 에이전트는 이전 서브골을 요약된 관찰로 대체하기로 결정함으로써 현재 서브골과 관련된 행동-관찰 쌍만을 유지합니다. 이는 메모리의 불필요한 중복을 줄여주며 보다 효율적인 문제 해결이 가능합니다.

- **Performance Highlights**: 실험 결과, HiAgent는 5개의 장기 과제에서 성공률을 두 배로 증가시키고 평균적으로 필요한 단계 수를 3.8 단계 줄였습니다. 또한 HiAgent는 여러 단계에 걸쳐 일관되게 성능 향상을 보이며, 그 견고성과 일반성을 강조합니다.



### No Such Thing as a General Learner: Language models and their dual optimization (https://arxiv.org/abs/2408.09544)
Comments:
          11 pages, 4 figures

- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)이 인간 인지 이해에 미치는 역할을 탐구하며, 언어 습득 논쟁에 대한 정보를 제공하는 방식에 대해 논의합니다. 특히 LLMs가 일반 학습자가 아니라는 관점을 제시하고, 이들이 훈련과 진화 과정을 통해 최적화되었음을 강조합니다. LLMs의 성능이 인간의 인지 편향과 어떻게 연관되는지에 대한 논란도 다루고 있습니다.

- **Technical Details**: 이 논문은 LLMs가 일반 학습자(General Learner)로 간주될 수 있는지에 대한 질문을 던집니다. 저자들은 LLMs가 그 자체로 최적화 과정(Optimization Process)을 거쳤다고 주장하며, 이는 자연 선택(Natural Selection) 과정에 유사하다고 설명합니다. 또한, LLMs의 권장된 훈련 목표와 그들의 진화적 발달(Evolutionary Maturation) 간의 관계를 분석합니다. 그들은 강한 해석 하에선 일반 학습자가 존재하지 않는다고 짚고 넘어갑니다. 다양한 연구와 성능(Benchmark)에 대한 결과를 바탕으로, LLMs가 실제 학습하는 방식의 특성과 인간 학습방법의 차이를 설명합니다.

- **Performance Highlights**: LLMs는 다양한 모델 아키텍처로 설계되며, 각각의 구조적 선택이 성능에 영향을 미친다는 점을 강조합니다. BERT, Llama, GPT-4와 같은 현대적인 LLM들은 성능을 극대화하기 위해 진화해온 절차를 통해 선택되었으며, 그 결과 매우 특화된 조정된 모델이 되었다고 설명합니다. 이는 LLMs가 훈련 데이터 이상으로 인덕티브 점프(Inductive Leap)를 할 수 있도록 해주는 기초가 됩니다.



### Using ChatGPT to Score Essays and Short-Form Constructed Responses (https://arxiv.org/abs/2408.09540)
Comments:
          35 pages, 8 tables, 2 Figures, 27 references

- **What's New**: 이 연구는 ChatGPT의 대규모 언어 모델이 ASAP 대회에서의 인간 및 기계 점수의 정확도를 일치시킬 수 있는지를 평가하였습니다.

- **Technical Details**: 연구는 선형 회귀(linear regression), 랜덤 포레스트(random forest), 그래디언트 부스팅(gradient boost), 부스팅(boost)을 포함한 다양한 예측 모델에 초점을 맞추었습니다. ChatGPT의 성능은 이차 가중 카파(quadratic weighted kappa, QWK) 메트릭스를 사용하여 인간 평가자와 비교 및 평가되었습니다.

- **Performance Highlights**: 결과는 ChatGPT의 그래디언트 부스팅 모델이 일부 데이터 세트에서 인간 평가자에 가까운 QWK를 달성했으나, 전체적인 성능은 불일치하며 종종 인간 점수보다 낮음을 나타냈습니다. 연구는 공정성을 확보하고 편향을 처리하는 데 있어 추가적인 개선이 필요하다고 강조하였습니다. 그럼에도 불구하고, ChatGPT는 특히 도메인 특화된 미세 조정(fine-tuning)을 통해 점수 산정의 효율성을 보여줄 잠재력을 입증하였습니다.



### Revisiting the Graph Reasoning Ability of Large Language Models: Case Studies in Translation, Connectivity and Shortest Path (https://arxiv.org/abs/2408.09529)
- **What's New**: 이 논문은 Large Language Models(LLM)의 그래프(즉, graph) 추론 능력에 집중합니다. 이론적으로 LLM이 그래프 추론 과제를 처리할 수 있는 능력이 입증되었으나 실제 평가에서는 여러 가지 실패가 나타났습니다. 이를 통해 LLM의 한계를 분석하고자 합니다.

- **Technical Details**: 세 가지 기본적인 그래프 과제인 그래프 설명 번역(graph description translation), 그래프 연결(graph connectivity), 그리고 최단 경로 문제(shortest-path problem)에 대해 LLM의 능력을 재조명합니다. 이 연구는 LLM이 그래프 구조를 텍스트 설명을 통해 이해하는 데 실패할 수 있음을 보여줍니다. 또한, 실제 지식 그래프(knowledge graphs)에 대한 조사도 진행하여 이론적 발견과 일치하는 결과를 얻었습니다.

- **Performance Highlights**: LLM은 세 가지 기본 그래프 과제에서 일관된 성능을 보여주지 못했으며, 그 원인은 다양하고 복합적인 것으로 나타났습니다. 연구 결과에 따르면, 그래프 크기(graph size) 외에도 연결 유형(connection types) 및 그래프 설명(graph descriptions)도 중요한 역할을 합니다. 이러한 발견은 LLM이 그래프 추론 과제에서 이론적으로 해결할 수 있는 능력이 있지만, 실제로는 그 한계가 분명함을 나타냅니다.



### Out-of-distribution generalization via composition: a lens through induction heads in Transformers (https://arxiv.org/abs/2408.09503)
Comments:
          41 pages, 25 figures

- **What's New**: 이 논문은 대형 언어 모델(LLM)들이 출처가 다른 데이터에 대해 어떻게 일반화할 수 있는지에 대한 연구 결과를 제시합니다. 특히, 숨겨진 규칙에 따라 생성된 샘플을 다루며, 주어진 입력 프롬프트에 대한 숨겨진 규칙을 추론하는 이 과정에서 OOD(Out-Of-Distribution) 일반화의 중요성을 강조합니다.

- **Technical Details**: 연구는 Transformer 아키텍처 내의 induction head 구성 요소에 초점을 맞추어 OOD 일반화와 컴포지션이 어떻게 연결되어 있는지를 규명합니다. Transformer의 훈련 동역학을 조사하고, 특수한 작업을 통해 두 개의 self-attention 레이어를 결합하여 규칙을 배우는 방법을 분석합니다. 자주 사용되는 'common bridge representation hypothesis'를 통해 임베딩(embedding) 공간의 공유 잠재 서브스페이스가 컴포지션을 위한 다리 역할을 한다고 제안합니다.

- **Performance Highlights**: 실험 결과, 2-레이어 Transformer가 OOD 일반화를 달성하는 과정에서 서브스페이스 매칭이 급격히 발생함을 보여주었습니다. 이는 인지 작업의 경우 유사한 컴포지션 구조가 필요하다는 것을 나타내며, LLMs는 모델 파라미터를 업데이트하지 않고도 몇 가지 데모가 있는 경우와 없는 경우 모두 특정 규칙을 학습할 수 있음을 시사합니다.



### REFINE-LM: Mitigating Language Model Stereotypes via Reinforcement Learning (https://arxiv.org/abs/2408.09489)
- **What's New**: 이번 논문에서는 기존 언어 모델(LM)에서 발생하는 다양한 편향 문제를 해결하기 위해 REFINE-LM이라는 새로운 디바이싱(debiasing) 방법을 제안합니다. 이 방법은 강화 학습(reinforcement learning)을 사용하여 모델의 성능을 유지하면서도 편향을 제거할 수 있습니다.

- **Technical Details**: REFINE-LM 접근법은 언어 모델의 예측 확률 분포를 기반으로 한 간단한 모델을 학습시켜 편향을 완화합니다. 이 방법은 인적 주석(human annotations)이나 엄청난 컴퓨팅 리소스를 요구하지 않으며, 성별, 민족, 종교 및 국적과 같은 다양한 편향 유형에 일반화될 수 있습니다.

- **Performance Highlights**: 실험 결과, REFINE-LM은 기존의 편향을 에지(-edge)로 하는 언어 모델의 성능을 유지하면서도 고전적인 편향을 의미 있게 줄임을 보여주었습니다. 또한, 이 방법은 다양한 유형의 편향에 적용 가능하며, 학습 비용이 낮고 빠르게 배포될 수 있습니다.



### Activated Parameter Locating via Causal Intervention for Model Merging (https://arxiv.org/abs/2408.09485)
- **What's New**: 이 논문에서는 Activated Parameter Locating (APL) 방법을 제안하여, 모델 병합 시 파라미터의 중요성을 효과적으로 평가하고, 델타 파라미터의 불필요한 부분을 정교하게 제거할 수 있도록 합니다. 이를 통해 기존 방법들이 간과한 태스크 관련 정보를 더 잘 활용할 수 있습니다.

- **Technical Details**: APL 방법은 인과 개입(causal intervention)을 사용하여 파라미터의 중요성을 추정합니다. 또한, 많은 파라미터 분할을 처리할 때의 계산 복잡성을 줄이기 위해 이론적으로 지지되는 경량화된 그래디언트 근사 방법을 도입합니다. 이 방법은 각 파라미터 점유에 대해 과제를 나타내는 벡터(즉, delta parameters)와 pretrained 모델의 그래디언트 벡터 간의 절댓값 내적을 사용하여 중요성을 근사합니다.

- **Performance Highlights**: 실험 결과, APL 방법이 있는 모델 병합이 인도메인(in-domain) 및 아웃오브도메인(out-of-domain) 환경 모두에서 기존 방법보다 더 나은 성능을 보임을 입증했습니다. APL을 통해 파라미터의 중요성을 기준으로 한 정교한 제거 및 병합 가중치 조정이 이루어져, 최종적으로 모델 병합 시 충돌을 효과적으로 완화할 수 있었습니다.



### PanoSent: A Panoptic Sextuple Extraction Benchmark for Multimodal Conversational Aspect-based Sentiment Analysis (https://arxiv.org/abs/2408.09481)
Comments:
          Accepted by ACM MM 2024 (Oral)

- **What's New**: 이 논문에서는 다중모드(multimodal) 대화형(aspect-based) 감정 분석(ABSA)의 새로운 접근을 제안합니다. 논문은 Panoptic Sentiment Sextuple Extraction과 Sentiment Flipping Analysis라는 두 가지 새로운 하위 작업을 소개합니다.

- **Technical Details**: 이 연구에서는 PanoSent라는 대규모(multimodal) 다국어(multilingual) 데이터셋을 구성하였으며, 이는 수동 및 자동으로 주석(annotation)이 달린 데이터로, 다양한 시나리오(scenarios)의 감정 요소를 포괄합니다. 또한, Chain-of-Sentiment reasoning framework와 새로운 다중모달 대형 언어 모델(multi-modal large language model)인 Sentica도 개발하였습니다.

- **Performance Highlights**: 제안된 방법론은 기존의 강력한 기준선(baselines)과 비교하여 탁월한 성능을 보여주었으며, 모든 방법이 효과적임을 검증했습니다.



### WPN: An Unlearning Method Based on N-pair Contrastive Learning in Language Models (https://arxiv.org/abs/2408.09459)
Comments:
          ECAI 2024

- **What's New**: 이번 연구에서는 유해한 출력을 줄이기 위한 새로운 방법인 Weighted Positional N-pair (WPN) Learning을 제안합니다. 이 방법은 모델의 출력을 개선하는 동시에 성능 저하를 최소화하는 데 목표를 두고 있습니다.

- **Technical Details**: WPN은 위치 가중 평균 풀링(position-weighted mean pooling) 기법을 도입하여 n-pair contrastive learning(대조 학습) 프레임워크에서 작동합니다. 이 기법은 유해한 출력(예: 유독한 반응)을 제거하여 모델의 출력을 '유해한 프롬프트-유해하지 않은 응답'으로 바꾸는 것을 목표로 합니다.

- **Performance Highlights**: OPT와 GPT-NEO 모델에 대한 실험에서 WPN은 유해한 반응 비율을 최대 95.8%까지 줄이는 데 성공했으며, 아홉 개의 일반 벤치마크에서 평균적으로 2% 미만의 성능 저하로 안정적인 성능을 유지했습니다. 또한, WPN의 일반화 능력과 견고성을 입증하는 실험 결과도 제공됩니다.



### Identifying Speakers and Addressees of Quotations in Novels with Prompt Learning (https://arxiv.org/abs/2408.09452)
Comments:
          This paper has been accepted by NLPCC 2024

- **What's New**: 이 논문은 중국 소설에서 인용문의 화자(인용문을 말하는 사람)와 대화 상대방(인용문을 듣는 사람)을 식별하는 최초의 데이터셋인 JY-QuotePlus를 소개합니다. 해당 데이터셋은 화자, 대화 상대방, 발화 방식, 언어적 단서를 포함하여 고유한 관계를 구축할 수 있게 합니다.

- **Technical Details**: JY-QuotePlus 데이터셋은 기존의 JY 데이터셋을 기반으로 만들어졌습니다. 화자와 대화 상대방을 식별하기 위해 MRC (Machine Reading Comprehension) 접근 방식을 활용하며, T5 및 PromptCLUE 같은 정밀 조정된 사전 훈련 모델(PTM)을 사용합니다. 본 논문에서는 영어와 중국어 데이터셋을 모두 평가하여 제안된 방법의 효과를 입증합니다.

- **Performance Highlights**: 실험 결과 제안된 방법이 적은 수의 예시로도 잘 작동하고 기존 방법보다 우수한 성능을 보였습니다. F1 점수는 98.78로 매우 높은 정확성을 나타내며, Kappa 점수는 0.9717로 안정성과 신뢰성을 확인할 수 있었습니다.



### HySem: A context length optimized LLM pipeline for unstructured tabular extraction (https://arxiv.org/abs/2408.09434)
Comments:
          9 pages, 4 tables, 3 figures, 1 algorithm

- **What's New**: HySem은 HTML 테이블에서 정확한 의미론적 JSON 표현을 생성하기 위한 새로운 컨텍스트 길이 최적화 기술을 도입합니다. 이 시스템은 소형 및 중형 제약회사를 위해 특별히 설계된 커스텀 파인 튜닝 모델을 활용하며, 데이터 보안과 비용 측면을 고려합니다.

- **Technical Details**: HySem은 세 가지 구성 요소로 구성된 LLM 파이프라인입니다: Context Optimizer Subsystem(𝒜CO), Semantic Synthesizer(𝒜SS), Syntax Validator(𝒜SV). 이 시스템은 컨텍스트 최적화 및 의미론적 변환을 통해 HTML 테이블의 데이터를 처리하고, 유효한 JSON을 출력하는 자동 구문 교정 기능까지 갖추고 있습니다.

- **Performance Highlights**: HySem은 기존 오픈 소스 모델들보다 높은 정확도를 제공하며, OpenAI GPT-4o와 비교할 때도 경쟁력 있는 성능을 보여줍니다. 이 시스템은 대용량 테이블을 처리하는 데 필수적인 컨텍스트 길이 제한을 효과적으로 해결합니다.



### FASST: Fast LLM-based Simultaneous Speech Translation (https://arxiv.org/abs/2408.09430)
- **What's New**: 본 논문에서는 동시 음성 번역(Simultaneous Speech Translation, SST) 문제를 해결하기 위해 FASST라는 새로운 접근법을 제안합니다. 이는 기존의 고지연(latency) 문제를 극복하면서 더 나은 번역 품질을 제공합니다.

- **Technical Details**: FASST는 블록별(causal) 음성 인코딩(blockwise-causal speech encoding)과 일관성 마스크(consistency mask)를 도입하여 빠른 음성 입력 인코딩을 가능하게 합니다. 두 단계 훈련 전략을 사용하여 품질-지연 간의 균형을 최적화합니다: 1) 음성 인코더 출력과 LLM 임베딩(Large Language Model embeddings)을 정렬하고, 2) 동시 번역을 위한 미세 조정(finetuning)을 진행합니다.

- **Performance Highlights**: FASST는 MuST-C 데이터셋에서 7B 모델을 사용하여 115M baseline 모델과 비교했을 때 같은 지연에서 영어-스페인어 번역에서 평균적으로 1.5 BLEU 점수 향상을 이루어 내었습니다.



### Distinguish Confusion in Legal Judgment Prediction via Revised Relation Knowledg (https://arxiv.org/abs/2408.09422)
Comments:
          Accepted by ACM TOIS

- **What's New**: 이번 연구에서는 법적 판별 예측(LJP) 문제를 해결하기 위해 새로운 통합 모델인 D-LADAN을 제안합니다. D-LADAN은 유사한 법 조항들이 서로 혼동되는 문제를 개선하기 위해 Graph Distillation Operator(GDO)를 적용하고, 법 조항 간의 후행 유사성을 감지하기 위한 동적 메모리 메커니즘을 도입하였습니다.

- **Technical Details**: D-LADAN은 법 조항 간의 그래프를 구성하고, Graph Distillation Operator(GDO)를 통해 고유한 특징을 추출합니다. 또한, 가중치가 있는 GDO를 사용하여 데이터 불균형 문제로 인한 귀납 편향을 수정하는데 초점을 맞추고 있습니다. 이 모델은 큰 규모의 실험을 통해 기존의 최첨단 모델들에 비해 높은 정확도와 견고성을 보여줍니다.

- **Performance Highlights**: D-LADAN은 기존 방법들에 비해 유의미한 성능 향상을 보여줍니다. 특히 데이터 불균형 문제로 인해 발생하는 혼동을 효과적으로 해결하여, 법적 판단의 정확성을 크게 개선시키는데 기여하고 있습니다.



### Challenges and Responses in the Practice of Large Language Models (https://arxiv.org/abs/2408.09416)
- **What's New**: 이 논문은 현재 주목받고 있는 AI 분야에서의 산업 동향, 학술 연구, 기술 혁신 및 비즈니스 애플리케이션 등 여러 차원에서의 질문들을 체계적으로 요약합니다.

- **Technical Details**: 강력한 컴퓨팅 리소스 통합을 위해 클라우드-엣지-단말 협업 아키텍처(cloud-edge-end collaborative architecture)를 구현하고, 데이터 수집, 엣지 처리, 클라우드 컴퓨팅, 협업 작업의 흐름을 설명합니다. 또한, 대형 언어 모델(LLM)의 필요성과 모델 훈련 시의 주요 도전 과제를 다룹니다.

- **Performance Highlights**: 2023년 중국의 가속 칩 시장이 140만 규모로 급격히 성장하였고, GPU 카드가 85%의 시장 점유율을 차지했습니다. LLM과의 관계에서는 사용자 맞춤형 개발 및 비즈니스 효율 개선이 두드러지며, 국내 AI 칩 브랜드가 미래에 더 큰 돌파구를 기대할 수 있다고 언급합니다.



### Comparison between the Structures of Word Co-occurrence and Word Similarity Networks for Ill-formed and Well-formed Texts in Taiwan Mandarin (https://arxiv.org/abs/2408.09404)
Comments:
          4 pages, 1 figure, 5 tables

- **What's New**: 이 연구는 대만 만다린의 비정형 텍스트로부터 생성된 단어 공기기 네트워크(Word Co-occurrence Network, WCN)와 단어 유사성 네트워크(Word Similarity Network, WSN)의 구조를 비교하고 분석하였습니다.

- **Technical Details**: 연구는 PTT의 139,578개 게시물과 2004년 및 2008년 대만 사법부의 53,272개 판결문을 데이터로 사용하여, 텍스트를 전처리 후 skip-gram word2vec 모델을 통해 단어 임베딩을 생성하였습니다. WCN 및 WSN는 각각 비가중치 및 무방향 네트워크로 구성되었습니다.

- **Performance Highlights**: 모든 네트워크는 스케일-프리(scale-free) 특성을 가지며, 소규모 세계(small-world) 성질을 지니고 있음이 확인되었습니다. WCN과 WSN은 일반적으로 비-assortative를 보였으며, 각 네트워크의 DAC 값은 다소 차이를 보였습니다.



### Offline RLHF Methods Need More Accurate Supervision Signals (https://arxiv.org/abs/2408.09385)
Comments:
          under review

- **What's New**: 이 논문에서는 기존의 오프라인 강화 학습 방법인 Reinforcement Learning with Human Feedback (RLHF)의 한계점을 극복하기 위해 보상 차이 최적화(Reward Difference Optimization, RDO)라는 새로운 방법을 제안합니다. 이 방법은 신뢰할 수 있는 선호 데이터를 기반으로 샘플 쌍의 보상 차이를 측정하여 LLM을 최적화하는 접근법입니다.

- **Technical Details**: RDO는 샘플 쌍의 보상 차이 계수를 도입하여 각 응답 쌍의 상호작용을 분석하고 개인의 선호를 반영합니다. 연구팀은 Attention 기반의 Difference Model을 개발하여 두 응답 간의 보상 차이를 예측하며, 이를 통해 오프라인 RLHF 방법을 개선합니다. 이 방법은 최대 마진 순위 손실(max-margin ranking loss) 및 부정 로그 우도 손실(negative log-likelihood loss)과 같은 기존 방법에 적용될 수 있습니다.

- **Performance Highlights**: 7B LLM을 사용하여 HH 및 TL;DR 데이터세트에서 진행된 실험 결과, RDO는 자동 지표 및 인간 평가 모두에서 효과적인 성능을 발휘하며, LLM과 인간의 의도 및 가치를 잘 맞출 수 있는 잠재력을 보여줍니다.



### Improving and Assessing the Fidelity of Large Language Models Alignment to Online Communities (https://arxiv.org/abs/2408.09366)
- **What's New**: 대형 언어 모델(LLMs)을 온라인 커뮤니티에 맞추는 새로운 프레임워크를 제안하며, 정서, 진정성, 독성 및 해악과 같은 다양한 언어적 측면에서의 정렬 품질을 평가하는 방법을 소개합니다.

- **Technical Details**: 이 연구는 커뮤니티의 사회적 미디어 게시물 코퍼스를 활용하여 LLM을 조정하는 방식으로, 각 시연에서 작업(예: 트윗 생성)을 정의하고 해당 작업에 대한 응답(정확한 트윗)을 기록합니다. 이후, 정렬된 LLM을 원본 게시물과 비교하여 진정성, 정서 톤, 독성 및 해악의 네 가지 주요 측면에서 평가합니다.

- **Performance Highlights**: 정렬된 LLM들은 전통적인 LLM보다 더 높은 품질의 저비용 디지털 대리인을 제공하였으며, 식이장애 위험이 다양한 커뮤니티 간의 차이를 성공적으로 구분해냈습니다. 또한, 이러한 프레임워크는 공공 건강 및 사회 과학 연구에 중요한 기여를 할 수 있는 잠재력을 보여주었습니다.



### SkyScript-100M: 1,000,000,000 Pairs of Scripts and Shooting Scripts for Short Drama (https://arxiv.org/abs/2408.09333)
Comments:
          18 pages, 12 figures

- **What's New**: SkyScript-100M은 약 1억 개의 스크립트와 촬영 스크립트 쌍으로 구성된 대규모 데이터셋으로, 이는 AI 기반 단편 드라마 생산의 효율성을 크게 향상시킬 수 있는 잠재력을 가집니다.

- **Technical Details**: 이 연구는 6,660개의 인기 단편 드라마 에피소드에서 수집한 약 80,000개의 에피소드를 기반으로 하여 약 10,000,000개의 촬영 스크립트를 생성했습니다. 연구팀은 'SkyReels'라는 자체 개발한 대규모 단편 드라마 생성 모델을 이용해 1,000,000,000개의 스크립트 쌍을 생성했습니다.

- **Performance Highlights**: SkyScript-100M 데이터셋은 기존 데이터셋과의 상세 비교를 통해, 단편 드라마의 자동화 및 최적화 가능성을 보여줍니다. 이 데이터셋은 AI 기반 스크립트 생성 및 짧은 드라마 비디오 생성의 패러다임 전환을 촉진할 수 있습니다.



### Fostering Natural Conversation in Large Language Models with NICO: a Natural Interactive COnversation datas (https://arxiv.org/abs/2408.09330)
Comments:
          16 pages, 3 figures, 10 tables

- **What's New**: NICO라는 이름의 새로운 자연 인터랙티브 대화 데이터셋이 소개되었으며, 이는 중국어로 된 다양한 일상 대화 주제를 포함하고 있습니다. 특히, 자연스럽고 인간과 유사한 대화 생성을 촉진하기 위해 설계되었습니다.

- **Technical Details**: NICO 데이터셋은 20개의 일상 생활 주제와 5가지 사회적 상호작용 유형을 포함하여 4,000개의 대화를 생성했습니다. 각 대화는 평균적으로 22.1개의 발화로 구성되어 있으며, 이를 통해 LLMs의 자연 대화 능력을 평가하고 개선할 수 있습니다.

- **Performance Highlights**: NICO 데이터셋은 유창성, 일관성, 자연스러움에서 기존의 다른 데이터셋보다 우수한 성과를 보였습니다. 특히, LLMs는 비자연적인 문장을 식별하는 데 있어 한계를 보였으며, 일상 대화의 유사한 모방에서는 여전히 도전과제가 존재합니다.



### Characterizing and Evaluating the Reliability of LLMs against Jailbreak Attacks (https://arxiv.org/abs/2408.09326)
- **What's New**: 본 연구에서는 Jailbreak 공격에 대한 대규모 실험을 바탕으로 LLM(대형 언어 모델)의 신뢰성을 평가하기 위한 포괄적인 평가 프레임워크를 제시합니다. 또한, 체계적으로 10가지 최신 jailbreak 전략과 1525개의 위험 질문을 분석하여 다양한 LLM의 취약점을 살펴봅니다.

- **Technical Details**: 이 연구는 Attack Success Rate (ASR), Toxicity Score, Fluency, Token Length, 그리고 Grammatical Errors 등 여러 차원에서 LLM의 출력을 평가합니다. 이를 통해 특정 공격 전략에 대한 모델의 저항력을 점검하고, 종합적인 신뢰성 점수를 생성하여 LLM의 취약점을 줄이기 위한 전략적 제안을 합니다.

- **Performance Highlights**: 실험 결과 모든 tested LLMs가 특정 전략에 대해 저항력이 부족함을 보여주며, LLM의 신뢰성을 중심으로 향후 연구의 방향성을 제시합니다. 이들 모델 중 일부는 유의미한 저항력을 보여준 반면, 다른 모델은 훈련 중 부여된 윤리적 및 내용 지침에 대한 일치성이 부족한 것으로 나타났습니다.



### An Open-Source American Sign Language Fingerspell Recognition and Semantic Pose Retrieval Interfac (https://arxiv.org/abs/2408.09311)
Comments:
          8 pages, 9 figures

- **What's New**: 이 논문은 ASL(미국 수화) 팔자 손가락 철자 인식과 의미적 포즈 검색을 위한 오픈 소스 인터페이스를 제안합니다. 이 인터페이스는 ASL 철자화를 음성 정보로 변환하는 인식 모듈과 음성 정보를 ASL 포즈 시퀀스로 변환하는 생성 모듈의 두 가지 모듈 구성요소를 제공합니다.

- **Technical Details**: 이 시스템은 합성곱 신경망(Convolutional Neural Networks) 및 포즈 추정 모델(Pose Estimation Models)을 활용하여 설계되었습니다. 사용자는 다양한 환경 조건(배경, 조명, 피부 톤, 손 크기 등)에서도 실시간으로 기능할 수 있도록 접근성과 사용자 친화성을 강조합니다.

- **Performance Highlights**: 이는 현재 ASL 철자 인식 및 생성을 위한 최첨단 기술을 보여주며, 향후 소비자 애플리케이션을 위한 실질적인 개선 가능성을 논의합니다.



### CyberPal.AI: Empowering LLMs with Expert-Driven Cybersecurity Instructions (https://arxiv.org/abs/2408.09304)
- **What's New**: 이번 연구에서는 사이버 보안 분야의 전문지식에 기반하여 설계된 새로운 데이터 세트인 SecKnowledge를 소개합니다. 이 데이터 세트는 여러 보안 관련 작업을 지원하는 데 최적화되어 있으며, 여러 단계의 생성 과정을 통해 구축되었습니다.

- **Technical Details**: SecKnowledge는 보안 관련 지식과 LLM(대규모 언어 모델)의 강점을 결합하여, 사이버 보안의 복잡한 개념을 이해하고 따를 수 있는 LLM을 훈련시키기 위한 지침 데이터 세트입니다. 이 데이터 세트는 두 단계로 생성됩니다. 첫 번째 단계에서는 전문가의 심층 분석을 바탕으로 미리 정의된 스키마에 따라 고품질의 지침을 생성하고, 두 번째 단계에서는 합성 콘텐츠를 기반으로 초기 데이터 세트를 확장합니다.

- **Performance Highlights**: CyberPal.AI라는 사이버 보안 전문 LLM 패밀리는 SecKnowledge 데이터 세트를 사용하여 훈련되었으며, 다양한 사이버 보안 작업에서 기준 모델에 비해 평균 24%의 성능 향상을 보여주었습니다. 또한, SecKnowledge-Eval이라는 평가 데이터 세트를 통해 LLM의 사이버 보안 작업 수행 능력을 평가하고, 이들 모델이 공인된 사이버 보안 기준에서도 최대 10%의 성능 향상을 달성했습니다.



### ConVerSum: A Contrastive Learning based Approach for Data-Scarce Solution of Cross-Lingual Summarization Beyond Direct Equivalents (https://arxiv.org/abs/2408.09273)
- **What's New**: 이 논문에서는 Cross-Lingual Summarization (CLS) 분야에서 고품질 데이터가 부족할 때도 효과적으로 요약을 생성할 수 있는 새로운 데이터 효율적인 접근법인 ConVerSum을 제안합니다. 이 모델은 대조 학습(contrastive learning)의 힘을 활용하여 다양한 언어로 후보 요약을 생성하고, 이를 참조 요약과 대비하여 훈련합니다.

- **Technical Details**: ConVerSum은 Seq2Seq 모델을 사용하여 여러 언어로 후보 요약을 생성하고, 다양한 Beam search를 통해 후보 요약의 품질을 평가합니다. 이후, 대조 학습을 통해 긍정적 쌍과 부정적 쌍 간의 거리를 최소화하거나 최대화하여 모델을 훈련시킵니다. 이 과정에서 LaSE 및 BERTScore를 사용하여 모델의 성능을 평가합니다.

- **Performance Highlights**: ConVerSum은 낮은 자원 언어(Low-resource languages)에서의 CLS 성능이 기존의 강력한 대형 언어 모델(Gemini, GPT 3.5, GPT 4)보다 우수함을 입증하였습니다. 이 연구는 CLS 기술의 효율성과 정확성을 한층 높이는 데 기여할 것으로 기대됩니다.



### Reference-Guided Verdict: LLMs-as-Judges in Automatic Evaluation of Free-Form Tex (https://arxiv.org/abs/2408.09235)
- **What's New**: 이 연구는 여러 LLM을 활용하여 오픈 엔디드(Large Language Models) 텍스트 생성을 평가하는 새로운 레퍼런스 기반 판별 방법을 제안합니다. 이는 평가의 신뢰성과 정확성을 높이고 인간 평가와의 일치도를 크게 향상시킵니다.

- **Technical Details**: 제안된 방법은 입력, 후보 모델의 응답 및 레퍼런스 답변을 포함하여 다수의 LLM을 판단자로 활용합니다. 평가 과정은 세 가지 주요 구성 요소로 이루어져 있으며, LLM이 생성한 출력을 검토하는 과정을 포함합니다. 이를 통해 다수의 LLM 판단의 집합적 판단이 인간 판단자와 유사한 신뢰성을 생성할 수 있는지를 검증합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 기본적으로 사람의 평가와 강한 상관관계를 보였으며, 특히 복잡한 작업에서 다수의 LLM 판단을 활용할 때 평가의 신뢰성과 정확성이 크게 향상됩니다.



### Architectural Foundations and Strategic Considerations for the Large Language Model Infrastructures (https://arxiv.org/abs/2408.09205)
- **What's New**: 대규모 언어 모델(LLM) 인프라 구축의 중요성을 강조하며, 소프트웨어 및 데이터 관리에 대한 자세한 분석을 통해 성공적인 LLM 개발을 위한 필수 고려사항과 보호 조치를 제시합니다.

- **Technical Details**: LLM 훈련을 위한 인프라 구성에서 H100/H800 GPU가 필수적이며, 7B 모델을 하루 안에 훈련할 수 있는 8개의 노드를 갖춘 서버 클러스터의 효율성을 강조합니다. LoRA(저순위 적응)와 같은 가벼운 조정 방법론은 GPU 사용의 접근성을 높이고, 고성능 GPU의 전략적 선택이 중요한 요소로 작용한다고 설명합니다.

- **Performance Highlights**: LLM의 성공적인 배포를 위해 컴퓨팅 파워, 비용 효율성, 소프트웨어 최적화 전략, 하드웨어 선택을 포괄적으로 고려해야 하며, 이는 AI 애플리케이션의 보편적 채택을 촉진하고 다양한 Domain에서의 지원을 강화하는 데 기여합니다.



### Chinese Metaphor Recognition Using a Multi-stage Prompting Large Language Mod (https://arxiv.org/abs/2408.09177)
- **What's New**: 본 연구에서는 중국어 은유를 인식하고 생성하는 능력을 증대시키기 위한 다단계 생성 휴리스틱 향상 프롬프트 프레임워크를 제안합니다. 기존의 사전 훈련된 모델로는 은유에서 텐서(tensor)와 차량(vehicle)을 완벽하게 인식하기 어려운 문제를 해결하고자 합니다.

- **Technical Details**: 제안된 시스템은 NLPCC-2024 Shared Task 9에 참여하여, DeBERTa 모델을 사용해 답변 후보를 생성합니다. 첫 번째 단계에서는 답변 후보에 대해 신뢰 점수를 획득하고, 두 번째 단계에서는 질문을 특정 규칙에 따라 클러스터링하고 샘플링합니다. 마지막으로 생성된 답변 후보와 시연을 결합해 휴리스틱 향상 프롬프트를 형성합니다.

- **Performance Highlights**: 제안된 모델은 NLPCC-2024 Shared Task 9의 Subtask 1에서 Track 1에서 3위, Track 2에서 1위, Subtask 2의 두 트랙 모두에서 1위를 차지하는 성과를 거두었습니다. 이는 각각 0.959, 0.979, 0.951, 0.941의 정확도로 우수한 결과를 나타냅니다.



### TableBench: A Comprehensive and Complex Benchmark for Table Question Answering (https://arxiv.org/abs/2408.09174)
Comments:
          12 pages

- **What's New**: 최근 대형 언어 모델(LLMs)의 발전은 표 형식 데이터의 해석 및 처리에서 이전에 상상할 수 없었던 능력을 도입했습니다. 그러나 산업 시나리오에서의 적용 시 LLM은 여전히 상당한 도전에 직면하고 있으며, 이는 실제 표 형식 데이터에 대한 복잡한 추론 요구로 인한 것입니다. 이를 해결하기 위해 우리는 산업에서의 표 형식 데이터 적용에 대한 상세한 조사를 수행하고, TableQA 능력의 네 가지 주요 범주 내 18개 필드를 포함한 포괄적이고 복잡한 벤치마크인 TableBench를 제안합니다.

- **Technical Details**: TableBench는 18개의 하위 범주를 포함하는 종합적이고 복잡한 표 질문 응답(TableQA) 벤치마크로, 문제 해결에 필요한 추론 단계에 따라 작업의 복잡성을 정의하고 체계적으로 분석합니다. 또한, 우리는 세 가지 서로 다른 추론 방법을 다루는 대규모 TableQA 지침 집합인 TableInstruct를 생성합니다. LLM에 대한 평가를 통해 TableLLM을 제안하며, 이는 GPT-3.5와 비슷한 성능을 보입니다.

- **Performance Highlights**: 테스트 결과에 따르면, 30개 이상의 LLM 모델이 TableBench에서 평가되었으며, 개방 소스 및 독점 LLM 모두 실제 세계 요구를 충족하기 위해 상당한 개선이 필요함을 강조합니다. 특히 가장 진보된 모델인 GPT-4도 인간 성능에 비해 보통수준의 점수만을 달성했습니다.



### Automatic Metrics in Natural Language Generation: A Survey of Current Evaluation Practices (https://arxiv.org/abs/2408.09169)
Comments:
          Accepted to INLG 2024

- **What's New**: 이 논문은 자연어 생성(NLG) 작업에서 자동 평가 메트릭의 사용에 대한 조사를 수행한 연구를 다룹니다. 연구자들은 자동 메트릭 사용의 적절성, 구현 세부 사항의 부족, 인간 평가와의 상관관계 결여 등 여러 문제점을 강조하고 있으며, 이에 대한 개선 방안을 제안하고 있습니다.

- **Technical Details**: 조사는 2023년에 발표된 110개의 논문을 기반으로 하며, 자동 평가 메트릭 및 인간 평가 방법의 사용을 정의하고 이를 평가하는 방식으로 진행되었습니다. 연구자들은 다양한 메트릭 사용의 패턴을 분석하고 이들의 재현 가능성과 반복 가능성을 검토하면서 메트릭의 적용과 보고 방식의 문제점을 논의합니다.

- **Performance Highlights**: 논문에서는 자동 메트릭 사용이 2016-2019년 사이에 약 25% 증가했음을 보여주며, 조사된 많은 논문들이 오로지 자동 메트릭에 의존한 것으로 나타났습니다. 최종적으로 연구자들은 더욱 엄격한 평가 관행을 확립하기 위한 권장 사항을 제시합니다.



### CogLM: Tracking Cognitive Development of Large Language Models (https://arxiv.org/abs/2408.09150)
Comments:
          under review

- **What's New**: 본 연구는 Piaget의 인지 발달 이론(PTC)에 기반하여 대형 언어 모델(LLM)의 인지 수준을 평가하는 기준 CogLM(언어 모델의 인지 능력 평가)를 소개하고, LLM의 인지 능력을 조사한 최초의 연구 중 하나입니다.

- **Technical Details**: CogLM은 1,220개의 질문을 포함하며, 10개의 인지 능력을 평가합니다. 이 질문들은 20명 이상의 전문가들에 의해 제작되었고, LLM의 성능을 다양한 방향에서 측정하기 위한 체계적인 평가 기준을 제공합니다. 실험은 OPT, Llama-2, GPT-3.5-Turbo, GPT-4 모델에서 수행되었습니다.

- **Performance Highlights**: 1. GPT-4와 같은 고급 LLM은 20세 인간과 유사한 인지 능력을 보여주었습니다. 2. LLM의 인지 수준에 영향을 미치는 주요 요인은 매개변수의 크기 및 최적화 목표입니다. 3. 다운스트림 작업에서의 성능은 LLM의 인지 능력 수준과 긍정적인 상관관계를 보입니다.



### Improving Rare Word Translation With Dictionaries and Attention Masking (https://arxiv.org/abs/2408.09075)
- **What's New**: 이 논문에서는 기계 번역에서 희귀 단어 문제를 해결하기 위한 새로운 접근법을 제안합니다. 이 방법은 이중 언어 사전(bilingual dictionary)에서 정의(definitions)를 소스 문장(source sentences)에 추가하고 주의 마스킹(attention masking)을 사용하여 희귀 단어와 그 정의를 연결합니다.

- **Technical Details**: 희귀 단어를 정의와 연결하기 위해 주의 마스킹 기법을 적용하며, 이중 언어 사전의 정의를 통해 번역 품질을 향상시키는 방법을 제시합니다.

- **Performance Highlights**: 희귀 단어에 대한 정의를 포함하면 BLEU 점수가 최대 1.0 증가하고 MacroF1 점수가 1.6 증가하여 번역 성능이 향상됨을 보여주었습니다.



### CodeTaxo: Enhancing Taxonomy Expansion with Limited Examples via Code Language Prompts (https://arxiv.org/abs/2408.09070)
- **What's New**: 새로운 연구인 CodeTaxo는 코드 언어를 프롬프트로 활용하여 분류체계(taxonomy) 확장의 효율성을 높여주는 혁신적인 방법입니다.

- **Technical Details**: CodeTaxo는 LLMs(Generative Large Language Models)의 특성을 이용하여 분류체계를 코드 완료(code completion) 문제로 재정의합니다. 이 방법에서는 각 엔티티를 나타내는 Entity 클래스를 정의하고, 인간 전문가나 크라우드소싱으로 작성된 기존 분류체계를 확장합니다. SimCSE를 활용하여 유사한 엔티티만을 선택적으로 프롬프트에 포함시킵니다.

- **Performance Highlights**: 다양한 실험 결과, CodeTaxo는 기존의 자가 지도(self-supervised) 방법에 비해 10.26%, 8.89%, 9.21%의 정확도 개선을 보여주었습니다. 이는 WordNet, Graphine 및 SemEval-2016에서 실험을 통해 검증되었습니다.



### Language Models Show Stable Value Orientations Across Diverse Role-Plays (https://arxiv.org/abs/2408.09049)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)이 다양한 페르소나(persona)를 채택하더라도 일관된 가치 지향(value orientation)을 보인다는 사실을 입증하였습니다. 이는 LLM의 응답에서 지속적인 관성(inertia) 현상이 나타남을 보여줍니다.

- **Technical Details**: 연구진은 'role-play-at-scale' 방법론을 도입하여 LLM에 무작위로 다양한 페르소나를 부여한 후 응답의 거시적(microscopic) 경향을 분석하였습니다. 이 방법은 기존의 방식과 달리, 무작위 페르소나가 적용된 여러 질문에 대해 모델이 어떻게 반응하는지를 체계적이고 확장 가능한 방법으로 진단합니다.

- **Performance Highlights**: 이 연구는 다양한 역할 놀이(role-play) 시나리오에서 LLM의 응답 패턴이 일관됨을 밝혀냈으며, 이는 LLM에 내재된 경향성(inherent tendencies)을 나타냅니다. 이러한 발견은 기초 모델(foundation models)에서의 가치 정렬(value alignment) 논의에 기여할 뿐만 아니라, LLM의 내재된 편향(bias)을 발견하는 진단 도구로서 'role-play-at-scale'의 효능을 보여줍니다.



### Studying the Effects of Collaboration in Interactive Theme Discovery Systems (https://arxiv.org/abs/2408.09030)
- **What's New**: 이 논문은 질적 연구에서 NLP(자연어 처리) 도구들이 협력 전략에 따라 어떻게 다른 결과를 초래할 수 있는지에 대한 평가 프레임워크를 제안합니다. 특히 동기식(synchronous) 및 비동기식(asynchronous) 협력에서의 차이를 분석합니다.

- **Technical Details**: 논문에서는 두 가지 NLP 도구, 즉 전통적인 HitL(인간-제어 루프) 주제 모델링 솔루션과 고급 개념 간의 관계를 모델링하는 시스템을 비교합니다. 이를 통해 동기식 및 비동기식 협력 맥락에서의 실험을 통해 출력 품질의 일관성, 응집력(cohesiveness), 정확성(correctness)을 평가합니다.

- **Performance Highlights**: 이 연구는 질적 연구의 상황에서 협력 설정이 NLP 도구의 성과에 미치는 영향을 증명하고, 다양한 출력 품질을 측정하는 포괄적인 평가 전략을 제안합니다. 이를 통해 질적 연구 도구의 향후 개발 및 평가 방법에 대한 통찰을 제공합니다.



### See What LLMs Cannot Answer: A Self-Challenge Framework for Uncovering LLM Weaknesses (https://arxiv.org/abs/2408.08978)
Comments:
          COLM 2024

- **What's New**: 이번 논문은 대규모 언어 모델(LLM)인 GPT-4가 자신의 한계를 발견할 수 있는지를 탐색하고, 이를 위한 Self-Challenge 평가 프레임워크를 제안합니다. 이 프레임워크는 GPT-4의 오류에서 발생하는 패턴을 요약하고, 인간 피드백을 통해 더 도전적인 데이터 생성을 위한 패턴을 세련되게 조정합니다.

- **Technical Details**: Self-Challenge 프레임워크는 주어진 실패 인스턴스 집합을 기반으로 LLM이 자율적으로 자신의 오류 패턴을 식별하고 요약하도록 유도합니다. 이를 통해 1,835개의 새로운 인스턴스를 생성한 SC-G4 벤치마크가 구축되었으며, 이는 다양한 도메인과 결합된 8개의 오류 패턴을 포함합니다.

- **Performance Highlights**: SC-G4에서 GPT-4는 오직 44.96%의 정확도로 문제를 해결하였습니다. 이러한 오류 패턴은 Claude-3 및 Llama-3와 같은 다른 LLM 역시 어려움을 겪게 하며, 단순한 fine-tuning으로는 해결되지 않는 '버그'일 가능성이 있습니다.



### A Multi-Task and Multi-Label Classification Model for Implicit Discourse Relation Recognition (https://arxiv.org/abs/2408.08971)
- **What's New**: 이번 연구에서는 Implicit Discourse Relation Recognition (IDRR)에서 내재된 모 ambiguity를 해결하기 위해 multi-task classification model을 제안합니다. 이 모델은 multi-label과 single-label의 discourse relations를 모두 학습할 수 있는능력을 가지고 있습니다.

- **Technical Details**: 제안된 모델은 pre-trained language model을 기반으로 하며, 두 개의 discourse arguments (ARG1+ARG2)를 입력으로 받아 embedding을 생성합니다. 이 embedding은 linear layer와 dropout layer를 통과한 후 PDTB 3.0 프레임워크의 세 가지 sense 수준에 대해 각각의 classification head로 전달됩니다. 모델의 학습에는 Adam optimization 알고리즘이 사용되었습니다.

- **Performance Highlights**: 제안된 모델은 DiscoGeM corpus에서 single-label classification task에서 SOTA 결과를 달성하였고, PDTB 3.0 corpus에서도 competitive한 성능을 보여주는 promising 결과를 나타내었습니다. 그러나 PDTB 3.0 데이터에 대한 사전 노출 없이 학습한 모델이기 때문에 성능이 현재 SOTA에는 미치지 못합니다.



### BnSentMix: A Diverse Bengali-English Code-Mixed Dataset for Sentiment Analysis (https://arxiv.org/abs/2408.08964)
- **What's New**: 이 논문에서는 20,000개의 샘플과 4가지 감성 태그가 포함된 대규모의 코드-혼합 벵갈리(Bengali) 감정 분석 데이터셋인 BnSentMix를 소개합니다. 이 데이터셋은 Facebook, YouTube 및 전자상거래 사이트와 같이 다양한 출처에서 수집되어 현실적인 코드-혼합 시나리오를 재현합니다.

- **Technical Details**: BnSentMix 데이터셋은 여러 출처에서 수집된 데이터를 바탕으로 하고 있으며, 긍정, 부정, 중립 및 혼합 감성의 4가지 태그로 라벨링되었습니다. 또한 코드-혼합 벵갈리-영어에 대해 추가로 학습된 3개의 새로운 transformer encoder를 포함한 14개의 기준 방법이 제안되며, 이로 인해 감정 분류 작업에서 69.8%의 정확도와 69.1%의 F1 점수를 달성했습니다.

- **Performance Highlights**: 성능 분석 결과, 다양한 감성 태그와 텍스트 유형에 따라 성능 차이가 발견되었으며, 이로 인해 향후 개선 가능성이 있는 부분이 강조되었습니다.



### LongVILA: Scaling Long-Context Visual Language Models for Long Videos (https://arxiv.org/abs/2408.10188)
Comments:
          Code and models are available at this https URL

- **What's New**: LongVILA는 멀티모달 비전-언어 모델을 위한 새로운 솔루션으로, 시스템, 모델 학습 및 데이터 집합 개발을 포함한 풀 스택(full-stack) 접근 방식을 제안합니다. 특히, Multi-Modal Sequence Parallelism (MM-SP) 시스템을 통해 256개의 GPU에서 2M 컨텍스트 길이 학습 및 추론이 가능해졌습니다.

- **Technical Details**: LongVILA는 5단계 교육 파이프라인, 즉 정렬, 사전 학습, 컨텍스트 확장 및 장단기 공동 감독의 세부 조정을 포함합니다. MM-SP를 통해 메모리 강도가 높은 장기 컨텍스트 모델을 효과적으로 교육할 수 있으며, Hugging Face Transformers와의 원활한 통합이 가능합니다.

- **Performance Highlights**: LongVILA는 VILA의 프레임 수를 128배(8에서 1024 프레임까지)로 확장하고, 긴 비디오 캡셔닝 점수를 2.00에서 3.26(1.6배)로 개선했습니다. 1400프레임 비디오(274k 컨텍스트 길이)에서 99.5%의 정확도를 달성하며, VideoMME 벤치마크에서도 성능 일관성을 보여주고 있습니다.



### In-Context Learning with Representations: Contextual Generalization of Trained Transformers (https://arxiv.org/abs/2408.10147)
- **What's New**: 이번 연구는 transformers가 부분적으로 레이블이 있는 프롬프트를 기반으로 보이지 않는 예제와 작업에 대해 일반화하는 방법을 이론적으로 설명합니다. 특히, 이 연구는 transformers가 in-context learning (ICL)을 통해 컨텍스트 정보를 배워 보이지 않는 예제에 일반화할 수 있는지를 최초로 입증한 작업입니다.

- **Technical Details**: 이 논문에서는 비선형 회귀(non-linear regression) 과제를 통해 gradient descent를 사용한 transformers의 훈련 동역학(training dynamics)을 분석합니다. 각 작업에 대한 템플릿 함수는 $m$개의 기저 함수로 구성된 선형 공간에 속하며, 초당 파라미터의 정보 추출 및 기억 동작을 분석합니다. 연구결과, 단일 레이어 다중 헤드 transformers가 훈련 기간 동안 훈련 손실이 글로벌 최소값으로 수렴한다는 것을 보여주었습니다.

- **Performance Highlights**: 훈련한 transformer는 ridge regression을 수행하여 기저 함수에 대해 효과적으로 학습하며, 주어진 임의의 프롬프트에 대해 ε-정밀도로 템플릿 선택을 통한 성능을 평가합니다. 이 연구는 기존 이론적 작업과 비교하여 transformers의 ICL 능력을 이해하는 데 기여하고 있습니다.



### Personalizing Reinforcement Learning from Human Feedback with Variational Preference Learning (https://arxiv.org/abs/2408.10075)
Comments:
this http URL

- **What's New**: 본 논문은 Reinforcement Learning from Human Feedback (RLHF) 기술을 발전시켜, 다양한 사용자들의 특성과 선호도에 맞춘 새로운 다중 모드 RLHF 방법론을 제안합니다. 본 기술은 사용자 특정 잠재 변수를 추론하고, 이 잠재 변수에 조건화된 보상 모델과 정책을 학습하는 방식으로, 단일 사용자 데이터 없이도 이루어집니다.

- **Technical Details**: 제안된 방법, Variational Preference Learning (VPL),은 변량 추론(variational inference) 기법을 계승하여 다중 모드 보상 모델링을 수행합니다. 사용자로부터 받은 선호 정보를 바탕으로 잠재 사용자 맥락에 대한 분포를 추론하고 이를 통해 다중 모드 선호 분포를 회복하는 것이 특징입니다. 이를 통해 RLHF의 비효율성과 잘못된 보상 모델을 수정합니다.

- **Performance Highlights**: 실험을 통해 VPL 방법이 다중 사용자 선호를 반영한 보상 함수를 효과적으로 모델링하고, 성능을 10-25% 향상시킬 수 있음을 입증했습니다. 시뮬레이션 로봇 환경과 언어 작업 모두에서 보상 예측의 정확성이 향상되어 diversa (diverse) 사용자 선호를 만족시키는 정책의 학습 성능이 크게 개선되었습니다.



### C${^2}$RL: Content and Context Representation Learning for Gloss-free Sign Language Translation and Retrieva (https://arxiv.org/abs/2408.09949)
- **What's New**: 이번 논문에서는 Sign Language Representation Learning (SLRL)을 위한 새로운 사전 학습 프로세스인 C${^2}$RL을 도입합니다. 이 방법은 Implicit Content Learning (ICL)과 Explicit Context Learning (ECL)을 통합하여 광라벨 없이도 효과적인 수어 표현 학습을 가능하게 하며, 기존의 방법들이 직면했던 문제들을 해결합니다.

- **Technical Details**: C${^2}$RL은 두 가지 주요 구성 요소인 ICL과 ECL을 중심으로 이루어져 있습니다. ICL은 커뮤니케이션의 미묘함을 포착하고, ECL은 수어의 맥락적 의미를 이해하여 이를 해당 문장으로 변환하는 데 집중합니다. 이 두 가지 학습 방법은 결합 최적화를 통해 강력한 수어 표현을 생성하여 SLT 및 SLRet 작업을 개선하는 데 기여합니다.

- **Performance Highlights**: 실험 결과 C${^2}$RL은 P14T에서 BLEU-4 점수를 +5.3, CSL-daily에서 +10.6, OpenASL에서 +6.2, How2Sign에서 +1.3 상승시키며, R@1 점수에서도 P14T에서 +8.3, CSL-daily에서 +14.4, How2Sign에서 +5.9 향상된 결과를 보여주었습니다. 또한, OpenASL 데이터셋의 SLRet 작업에서 새로운 베이스라인을 설정했습니다.



### Microscopic Analysis on LLM players via Social Deduction Gam (https://arxiv.org/abs/2408.09946)
Comments:
          Under review, 10 pages

- **What's New**: 최근 연구들은 대형 언어 모델(LLMs)을 활용한 자율 게임 플레이어 개발에 주목하고 있으며, 특히 사회적 추론 게임(SDGs)에서의 게임 플레이 능력 평가에 대한 새로운 접근 방식을 제안하고 있습니다.

- **Technical Details**: 기존 연구들이 게임 플레이 능력을 전반적인 게임 결과를 통해 평가한 반면, 본 연구에서는 SpyFall 게임의 변형인 SpyGame을 사용하여 4개의 LLM을 분석하고, 이를 통해 특화된 8개의 미세 지표(microscopic metrics)를 도입하였습니다. 이 지표들은 의도 식별(intent identification)과 변장(camouflage) 능력을 평가하는 데 더 효과적임을 보여줍니다.

- **Performance Highlights**: 본 연구에서는 4개의 주요 그리고 5개의 하위 카테고리를 통해 LLM의 비정상적 추론 패턴을 확인하였으며, 이에 대한 정량적 결과들을 정성적 분석과 연계하여 검증하였습니다.



### Attribution Analysis Meets Model Editing: Advancing Knowledge Correction in Vision Language Models with VisEd (https://arxiv.org/abs/2408.09916)
- **What's New**: 이번 연구에서는 Vision-LLMs (VLLMs)에 대한 모델 편집 기술을 제안합니다. 기존 연구들은 주로 텍스트 모달리티에 국한되었으나, 이 연구에서는 비주얼 표현이 어떻게 예측에 영향을 미치는지에 대한 새로운 통찰을 제공합니다.

- **Technical Details**: 연구는 주로 미드-레벨 및 레이턴 (latent) 레이어에서 비주얼 표현의 기여를 측정하기 위해 contribution allocation과 noise perturbation 방법을 사용합니다. VisEdit라는 새로운 VLLM 편집기를 도입하여 탈중심화된 비주얼 표현을 수정하여 주요 응답을 개선합니다.

- **Performance Highlights**: VisEdit는 BLIP2-OPT, MiniGPT-4, LLaVA-V1.5 등 여러 VLLM 백본을 사용하여 테스트되었으며, 기존 최첨단 편집기들에 비해 우수한 성능을 보였습니다. E-VQA 및 E-IC와 같은 공공 VLLM 편집 벤치마크 데이터셋에서도 뛰어난 신뢰성과 일반성을 입증했습니다.



### MAPLE: Enhancing Review Generation with Multi-Aspect Prompt LEarning in Explainable Recommendation (https://arxiv.org/abs/2408.09865)
Comments:
          8 main pages, 10 pages for appendix. Under review

- **What's New**: 이번 논문에서는 사용자와 아이템 쌍으로부터 추천 이유를 제시하는 Explainable Recommendation 과제를 다루고 있습니다. MAPLE(Multi-Aspect Prompt LEarner)라는 개인화된 모델이 제안되었으며, 이는 세분화된 측면 용어를 기억하기 위해 측면 범주를 또 다른 입력 차원으로 통합합니다.

- **Technical Details**: MAPLE은 리뷰 생성과 관련된 기존 모델들이 일반성과 환각 문제로 고생하는 것을 해결하기 위해 Multi-aspect 개념을 도입했습니다. 실험을 통해 MAPLE은 음식점 도메인에서 문장과 특징의 다양성을 높이면서도 일관성과 사실적 관련성을 유지한다는 것을 보여주었습니다. MAPLE은 리트리버-리더 프레임워크에서 리트리버 구성 요소로 사용되며, 대형 언어 모델(LLM)을 리더로 배치하여 개인화된 설명을 생성합니다.

- **Performance Highlights**: MAPLE은 기존의 리뷰 생성 모델에 비해 텍스트와 기능의 다양성에서 우수한 성과를 보였으며, MAPLE이 생성한 설명은 리트리버-리더 프레임워크 내에서 좋은 쿼리로 기능합니다. 또한, MAPLE은 생성된 설명의 사실적 관련성과 설득력 모두를 개선할 수 있는 잠재력을 보여줍니다.



### AutoML-guided Fusion of Entity and LLM-based representations (https://arxiv.org/abs/2408.09794)
- **What's New**: BabelFusion은 지식 기반의 표현을 LLM 기반 표현에 주입하여 분류 작업의 성능을 향상시키는 새로운 접근법을 제안합니다. 이 방법은 또한 자동화된 머신러닝(AutoML)을 활용하여 지식으로 정보가 융합된 표현 공간에서 분류 정확성을 개선할 수 있음을 보여줍니다.

- **Technical Details**: 이 연구에서 제안하는 BabelFusion은 텍스트 문서와 관련된 레이블로 구성된 데이터셋을 기반으로, 지식 그래프에서 관련 항목을 식별하고 벡터 표현을 검색합니다. 이후 각 문서에 대해 이러한 임베딩을 평균하여 단일 벡터로 변환하고, 이를 SVD(슬램값 분해)를 통해 더 낮은 차원의 공간으로 투사하여 문서 표현을 학습합니다.

- **Performance Highlights**: 이 연구는 지식 기반 표현을 LLM에 통합하고, 단순 모델과 결합하여 경쟁력 있는 결과를 달성할 수 있음을 증명하였습니다. 다양한 실험에서는 다섯 개의 강력한 LLM 기초 모델을 사용해, 저차원으로 축소된 표현 공간에서도 낮은 예측 성능 손실로 신속한 분류기가 가능하다는 것을 보여주었습니다.



### GoNoGo: An Efficient LLM-based Multi-Agent System for Streamlining Automotive Software Release Decision-Making (https://arxiv.org/abs/2408.09785)
- **What's New**: GoNoGo는 자동차 소프트웨어 배포 과정을 간소화하기 위해 설계된 LLM(대형 언어 모델) 에이전트 시스템입니다. 이 시스템은 기능적 요구사항과 산업적 제약을 모두 만족시킬 수 있도록 구성되어 있으며, 특히 위험이 민감한 도메인에 맞춤화되었습니다.

- **Technical Details**: GoNoGo에는 두 가지 주요 LLM 에이전트인 Planner와 Actor가 포함됩니다. Planner는 사용자의 쿼리를 이해하고 데이터 분석을 위한 단계별 지침으로 분해합니다. Actor는 이러한 고수준 지침에서 실행 가능한 스크립트를 생성합니다. 이 시스템은 인-context learning을 사용하여 도메인 특정 요구사항을 인코딩합니다.

- **Performance Highlights**: GoNoGo는 Level 2 난이도의 작업에 대해 3-shot 예시를 사용하여 100% 성공률을 달성했으며, 복잡한 작업에서도 높은 성능을 유지합니다. 이 시스템은 더 간단한 작업의 의사결정을 자동화하여 수동 개입의 필요성을 크게 줄입니다.



### Strategic Demonstration Selection for Improved Fairness in LLM In-Context Learning (https://arxiv.org/abs/2408.09757)
- **What's New**: 최근 연구에 따르면 in-context learning (ICL)을 활용하여 대형 언어 모델(LLMs)이 표 형식의 데이터를 처리하는 데 있어 뛰어난 효율성을 보이고 있습니다. 그러나 이러한 방법의 공정성 문제에 대해서는 상대적으로 낮은 이해도를 보이고 있습니다. 이 연구는 ICL 프롬프트에서 다양한 데모가 LLM의 공정성 결과에 미치는 영향을 조사합니다.

- **Technical Details**: 연구에서는 소수 그룹 샘플을 프롬프트에 포함시키는 방법을 통해 LLM의 공정성을 크게 향상시킬 수 있음을 발견했습니다. 특히, 소수 집단과 다수 집단 샘플의 비율이 공정성과 예측 정확성 사이의 균형에 영향을 미친다는 것을 보여줍니다. 본 연구는 Clustering과 Evolutionary 전략을 활용한 Fairness via Clustering-Genetic (FCG) 알고리즘을 제안하여 훈련 데이터에서 다양한 대표 샘플을 커버하는 방안을 모색합니다.

- **Performance Highlights**: 실험 결과, 제안된 FCG 알고리즘은 다양한 지표에서 LLM의 공정성을 극적으로 개선시키는 것으로 밝혀졌습니다. 특정 전략에서 소수 그룹을 우선적으로 고려할 경우 가장 우수한 성과를 보였습니다.



### R2GenCSR: Retrieving Context Samples for Large Language Model based X-ray Medical Report Generation (https://arxiv.org/abs/2408.09743)
Comments:
          In Peer Review

- **What's New**: 본 논문은 X-ray 의료 보고서 생성을 위한 새로운 프레임워크인 R2GenCSR를 제안합니다. 이는 컨텍스트 샘플을 활용하여 대형 언어 모델(LLMs)의 성능을 향상시키는 방법을 제공합니다.

- **Technical Details**: Mamba라는 비전 백본을 도입하여 선형 복잡도로 높은 성능을 달성하며, 트레이닝 단계에서 각 미니 배치의 샘플을 위한 컨텍스트 샘플을 검색하여 기능 표현을 강화합니다. 시각 토큰과 컨텍스트 정보를 LLM에 공급하여 고품질 의료 보고서를 생성합니다.

- **Performance Highlights**: IU-Xray, MIMIC-CXR, CheXpert Plus라는 세 개의 X-ray 보고서 생성 데이터셋에서 광범위한 실험을 진행하였으며, 제안된 프레임워크의 효과성을 완벽하게 검증하였습니다.



### Pedestrian Attribute Recognition: A New Benchmark Dataset and A Large Language Model Augmented Framework (https://arxiv.org/abs/2408.09720)
Comments:
          MSP60K PAR Benchmark Dataset, LLM based PAR model, In Peer Review

- **What's New**: 이 논문에서는 새로운 대규모 보행자 속성 인식 데이터셋인 MSP60K를 제안합니다. 이 데이터셋은 60,122개의 이미지와 57개의 속성 주석을 포함하며, 8가지 시나리오에서 수집되었습니다. 또한, LLM(대규모 언어 모델)으로 강화된 보행자 속성 인식 프레임워크를 제안하여 시각적 특징을 학습하고 속성 분류를 위한 부분 인식을 가능하게 합니다.

- **Technical Details**: MSP60K 데이터셋은 다양한 환경과 시나리오를 반영하여 보행자 이미지를 수집하였으며, 구조적 손상(synthetic degradation) 처리를 통해 현실 세계의 도전적인 상황을 시뮬레이션합니다. 데이터셋에 대해 17개의 대표 보행자 속성 인식 모델을 평가하였고, 랜덤 분할(random split)과 크로스 도메인 분할(cross-domain split) 프로토콜을 사용하여 성능을 검증하였습니다. LLM-PAR 프레임워크는 비전 트랜스포머(Vision Transformer) 기반으로 이미지 특징을 추출하며, 멀티-임베딩 쿼리 트랜스포머(Multi-Embedding Query Transformer)를 통해 속성 분류를 위한 부분 인식 특징을 학습합니다.

- **Performance Highlights**: 제안된 LLM-PAR 모델은 PETA 데이터셋에서 mA 및 F1 메트릭 기준으로 각각 92.20 / 90.02의 새로운 최첨단 성능을 달성하였으며, PA100K 데이터셋에서는 91.09 / 90.41의 성능을 기록했습니다. 이러한 성능 향상은 MSP60K 데이터셋과 기존 PAR 벤치마크 데이터셋에서 폭넓은 실험을 통해 확인되었습니다.



### A Comparison of Large Language Model and Human Performance on Random Number Generation Tasks (https://arxiv.org/abs/2408.09656)
- **What's New**: 본 연구는 ChatGPT-3.5가 인간의 무작위 숫자 생성 패턴을 모방하는 방식에 대해 조사합니다. 특히, LLM(대형 언어 모델) 환경에서 기존 인간 RNGT(무작위 숫자 생성 작업)를 조정하여 새로운 연구 결과를 도출했습니다.

- **Technical Details**: 이 연구는 OpenAI의 ChatGPT-3.5 모델을 이용하여 10,000개의 응답을 수집했습니다. 모델에 대해 기존 연구에서 사용된 구술 지침을 기반으로 특별한 사용자 프롬프트를 작성하여 무작위 숫자 시퀀스를 생성하도록 지시했습니다. 생성된 시퀀스의 길이는 평균 269와 표준편차 325로 설정했습니다.

- **Performance Highlights**: 초기 결과에 따르면, ChatGPT-3.5는 인간에 비해 반복적이고 순차적인 패턴을 덜 피하며, 특히 인접 숫자 빈도가 현저히 낮습니다. 이는 LLM이 인간의 인지 편향을 대신하여 무작위 숫자 생성 능력을 향상할 가능성을 제시합니다.



### MoDeGPT: Modular Decomposition for Large Language Model Compression (https://arxiv.org/abs/2408.09632)
Comments:
          31 pages, 9 figures

- **What's New**: 최근 대형 언어 모델(LLM)의 혁신과 업그레이드 속에서, 'MoDeGPT'라는 새로운 고급 압축 기술이 소개되었습니다. MoDeGPT는 이전의 방법들과는 달리 회복 미세 조정(Recovery Fine-Tuning) 필요 없이 Transformer 블록을 모듈로 분할하여 압축하는 방법을 제시하여, 낮은 정확도 저하 및 높은 오버헤드를 유발하는 기존의 단점을 해소합니다.

- **Technical Details**: MoDeGPT는 세 가지 잘 정의된 행렬 분해 알고리즘(Nyström approximation, CR decomposition, SVD)을 기반으로 한 이론적 프레임워크를 적용하여 Transformer 모듈을 압축합니다. 이 방법은 특정 모듈 내에서 여러 행렬을 동시에 분해하면서도 매개변수 추가 없이, 고유한 매커니즘을 통해 중간 차원을 효과적으로 줄이며, 모듈 수준의 재구성 오류 경계를 제공합니다.

- **Performance Highlights**: MoDeGPT는 Llama-2/3 및 OPT 모델에서 25-30%의 압축률을 유지하면서도 90-95%의 제로샷(zero-shot) 성능을 보입니다. 또한 13B 모델의 경우 계산 비용을 98% 절감하며, 한 대의 GPU에서 몇 시간 내에 압축이 가능하고, 추론 처리량(inference throughput)을 최대 46%까지 증가시킵니다.



### PhysBERT: A Text Embedding Model for Physics Scientific Literatur (https://arxiv.org/abs/2408.09574)
- **What's New**: PhysBERT는 물리학 분야에 특화된 첫 번째 텍스트 임베딩 모델로, 120만 개의 arXiv 물리학 논문을 데이터로 사전 학습하고, 세부 지도 데이터를 통해 미세 조정되어 물리학 관련 작업에서 일반 모델을 능가한다.

- **Technical Details**: PhysBERT는 BERT 아키텍처를 기반으로 하여, 사전 학습 과정에서 Masked Language Modeling (MLM) 접근법을 활용한다. 모델 구조는 BERTbase를 사용하고, 최종적으로 Simple Contrastive Learning of Sentence Embeddings (SimCSE) 기법으로 미세 조정하여 의미 있는 문장 표현을 생성한다.

- **Performance Highlights**: PhysBERT는 물리학 전용 작업에서 정밀도와 관련성이 향상된 정보를 제공하며, 클러스터링, 정보 검색, 분류 등 다양한 하위 작업에서 성능을 평가한 결과 일반 모델보다 높은 성과를 보였다.



### Image-Based Geolocation Using Large Vision-Language Models (https://arxiv.org/abs/2408.09474)
- **What's New**: 이 논문은 기존의 딥러닝 및 LVLM(large vision-language models) 기반 지리 정보 방법들이 제기하는 문제를 심층적으로 분석한 첫 번째 연구입니다. LVLM이 이미지로부터 지리 정보를 정확히 파악할 수 있는 가능성을 보여 주며, 이를 해결하기 위해 	ool{}이라는 혁신적인 프레임워크를 제안합니다.

- **Technical Details**: 	ool{}는 체계적인 사고 연쇄(chain-of-thought) 접근 방식을 사용하여 차량 유형, 건축 스타일, 자연 경관, 문화적 요소와 같은 시각적 및 맥락적 단서를 면밀히 분석함으로써 인간의 지리 추측 전략을 모방합니다. 50,000개의 실제 데이터를 기반으로 한 광범위한 테스트 결과, 	ool{}은 전통적인 모델 및 인간 기준보다 높은 정확도를 달성했습니다.

- **Performance Highlights**: GeoGuessr 게임에서 평균 점수 4550.5와 85.37%의 승률을 기록하며 뛰어난 성능을 보였습니다. 가장 가까운 거리 예측의 경우 0.3km까지 정확성을 보였고, LVLM의 인지 능력을 활용한 정교한 프레임워크를 통해 지리 정보의 정확성을 크게 향상시켰습니다.



### Hindi-BEIR : A Large Scale Retrieval Benchmark in Hind (https://arxiv.org/abs/2408.09437)
- **What's New**: 본 논문에서는 인도에서 사용되는 힌디어 정보를 검색하기 위한 새로운 기준인 Hindi-BEIR를 소개합니다. 이 기준은 영어 BEIR 분석 데이터의 하위 집합, 기존 힌디어 검색 데이터 세트, 합성 데이터 세트를 포함하여 15개의 데이터세트와 8개의 다양한 작업을 포함합니다.

- **Technical Details**: Hindi-BEIR는 6개의 서로 다른 도메인에서 온 15개의 데이터 세트로 구성되며, 이는 데이터 검색 모델을 평가하고 비교하기 위한 포괄적인 벤치마크 제공을 목표로 합니다. 또한, 다국어 검색 모델에 대한 기준 성능 비교를 통해 향후 연구 방향을 제안합니다.

- **Performance Highlights**: 이 벤치마크는 현재를 기준으로 하는 다국어 검색 모델의 성능을 비교하고 고유한 도전 과제를 발견함으로써 힌디어 정보 검색 시스템의 발전에 기여할 것입니다. Hindi-BEIR는 공개적으로 이용 가능하며, 연구자들이 현재 힌디어 검색 모델의 한계와 가능성을 이해하는 데 도움을 줄 것입니다.



### Enhancing Startup Success Predictions in Venture Capital: A GraphRAG Augmented Multivariate Time Series Method (https://arxiv.org/abs/2408.09420)
- **What's New**: 이번 연구에서는 기존의 스타트업 성공 예측 모델들이 경쟁과 협력 관계와 같은 중요한 기업 간 관계를 제대로 반영하지 못한 문제를 해결하기 위해 GraphRAG을 통합한 신규 접근 방식을 제안하였습니다.

- **Technical Details**: GraphRAG는 지식 그래프(knowledge graphs)와 검색 증강 생성 모델(retrieval-augmented generation models)을 효과적으로 통합하여 스타트업 간의 관계 복잡성을 탐색할 수 있도록 합니다. 본 연구에서는 multivariate time series analysis를 통해 기업 간의 관계 정보를 포함함으로써 예측 모델의 성능을 향상시키고자 하였습니다.

- **Performance Highlights**: 실험 결과, 본 연구의 GraphRAG 기반 모델이 스타트업 성공 예측에서 기존 모델보다 상당히 우수한 성능을 보였습니다. 특히, 데이터가 희소한 상황에서도 높은 정확도를 유지하며 예측력을 향상시켰습니다.



### Game Development as Human-LLM Interaction (https://arxiv.org/abs/2408.09386)
- **What's New**: 본 논문에서는 LLM(대형 언어 모델)을 활용한 사용자 친화적 게임 개발 플랫폼인 Interaction-driven Game Engine(IGE)을 소개합니다. 사용자는 자연어로 게임을 개발할 수 있으며, 복잡한 프로그래밍 언어의 학습 없이도 맞춤형 게임을 만들 수 있게 됩니다.

- **Technical Details**: IGE는 사용자의 입력을 기반으로 게임 스크립트 세그먼트를 구성하고, 해당 스크립트에 따른 코드 스니펫을 생성하며, 사용자와의 상호작용(가이드 및 피드백 포함)을 처리하는 세 가지 주요 단계(P_{script}, P_{code}, P_{utter})를 수행하도록 LLM을 훈련시킵니다. 또한, 효과적인 데이터 생성 및 훈련을 위해 세 단계의 점진적 훈련 전략을 제안합니다.

- **Performance Highlights**: 포커 게임을 사례 연구로 사용하여 IGE의 성능을 두 가지 관점(상호작용 품질 및 코드 정확성)에서 종합적으로 평가했습니다. 이 평가 과정은 IGE의 게임 개발 효율성을 향상시키고, 사용자와의 상호작용을 쉽게 만드는 데 기여합니다.



### Concept Distillation from Strong to Weak Models via Hypotheses-to-Theories Prompting (https://arxiv.org/abs/2408.09365)
Comments:
          13 pages, 8 figures, conference

- **What's New**: 본 논문에서는 Concept Distillation (CD)이라는 자동화된 프롬프트 최적화 기법을 제안하여, 약한 언어 모델(weak language models)의 복잡한 작업에서의 성능을 향상시키는 방법을 다룹니다. CD는 초기 프롬프트를 기반으로 약한 모델의 오류를 수집하고, 강력한 모델(strong model)을 사용해 이러한 오류에 대한 이유를 생성하여 규칙을 만듭니다.

- **Technical Details**: CD는 세 가지 주요 단계를 포함합니다: (1) 약한 모델의 잘못된 응답 수집(초기화), (2) 강력한 모델을 통해 오류 분석 및 규칙 생성(유도), (3) 검증 세트 성능에 따른 규칙 필터링 및 기본 프롬프트에 통합(추론/검증). 이 과정에서, 강력한 모델이 약한 모델의 성능을 개선하는 데 필요한 개념을 제공하는 방식으로 진행됩니다.

- **Performance Highlights**: CD 방법을 NL2Code, 수학적 추론 과제 등에 적용한 결과, 작은 언어 모델의 성능이 현저히 향상되었습니다. 예를 들어, Mistral-7B 모델의 Multi-Arith 정확도가 20% 증가하고, Phi-3-mini-3.8B 모델의 HumanEval 정확도가 34% 상승했습니다. CD는 다른 자동화 방법들에 비해 약한 모델의 성능을 개선하고, 새로운 언어 모델로의 원활한 전환을 지원할 수 있는 비용 효율적인 전략을 제공합니다.



### Threshold Filtering Packing for Supervised Fine-Tuning: Training Related Samples within Packs (https://arxiv.org/abs/2408.09327)
Comments:
          13 pages, 4 figures

- **What's New**: 이 논문에서는 Supervised Fine-Tuning (SFT)에서 효율적인 packing 기술인 Threshold Filtering Packing (TFP)을 소개합니다. TFP는 관련된 문맥을 가지면서도 다양성을 유지하는 샘플을 선정하여 SFT 성능을 개선하는 방법입니다.

- **Technical Details**: TFP는 Traveling Salesman Problem (TSP)에서 영감을 받은 탐욕 알고리즘을 사용하여 샘플을 여러 pack으로 세분화합니다. 각 샘플은 그래프의 노드로 표현되며, 임계값 필터링을 통해 샘플 간의 유사성을 조절하여 서로 다른 문장 간의 혼합을 방지합니다. 이를 통해 각 pack이 유용한 문맥을 제공하면서도 불필요한 교차 오염을 피할 수 있습니다.

- **Performance Highlights**: 실험 결과, TFP는 GSM8K에서 최대 7%, HumanEval에서 4%, adult-census-income 데이터셋에서 15%의 성능 향상을 보여주었으며, 여러 LLM에서 우수한 성능을 발휘하면서 계산 비용도 크게 낮추었습니다.



### Generating Data with Text-to-Speech and Large-Language Models for Conversational Speech Recognition (https://arxiv.org/abs/2408.09215)
Comments:
          To appear at SynData4GenAI 2024 workshop

- **What's New**: 이번 연구에서는 대규모 언어 모델(LLM)을 활용하여 다중 발화자 대화 ASR을 위한 합성 데이터 생성 파이프라인을 제안합니다. 구체적으로, LLM을 통한 콘텐츠 생성과 대화형 다중 발화자 텍스트-음성 변환(TTS) 모델을 통한 음성 합성을 결합하여 효율적인 데이터 생성을 목표로 합니다.

- **Technical Details**: 제안된 방법은 Llama 3 8B Instruct 모델을 사용하여 대화 원고를 생성하고, Parakeet라는 대화형 TTS 모델을 통해 이 원고를 음성으로 변환합니다. 연구는 Whisper ASR 모델을 전화 통화 및 원거리 대화 환경에서 미세 조정하여 최고 성능을 평가합니다.

- **Performance Highlights**: 연구 결과, 제안된 방법은 외부 비대화 음성 데이터셋을 사용하는 전통적인 다중 발화자 생성 접근 방식보다 유의미하게 향상된 성능을 보여주었습니다. 이를 통해 합성 데이터의 활용 가능성이 증대되었음을 입증했습니다.



### AI Managed Emergency Documentation with a Pretrained Mod (https://arxiv.org/abs/2408.09193)
Comments:
          Ethical approval for the study was obtained from the University College Dublin, Human Research Ethics Committee (UCD HREC)

- **What's New**: 이 연구는 응급실(ED)에서 퇴원 편지 작성을 개선하기 위한 대형 언어 모델 시스템의 활용을 조사하였습니다. 현재 퇴원 편지 목표에 대한 준수가 어려운 시간 제약과 인프라 부족 문제를 해결하고자 AI 소프트웨어의 효율성 제안을 탐색했습니다.

- **Technical Details**: 연구에서 사용된 시스템은 GPT-3 Davinci 모델을 기반으로 하여 의료 중심 퇴원 편지를 생성하도록 세부 조정되었습니다. 시스템은 음성에서 텍스트로 변환하기 위해 open-source whisper model을 사용하며, 주기적인 세부 조정 과정을 통해 모델 성능을 향상시킵니다.

- **Performance Highlights**: 19명의 응급 의학 경력이 있는 의사들이 MedWrite LLM 인터페이스를 평가한 결과, 수작업과 비교하여 시각적으로 의미 있는 시간 절약이 스며드는 것으로 나타났습니다.



### Cognitive LLMs: Towards Integrating Cognitive Architectures and Large Language Models for Manufacturing Decision-making (https://arxiv.org/abs/2408.09176)
Comments:
          20 pages, 8 figures, 2 tables

- **What's New**: 이번 연구에서는 Cognitive Architectures와 Large Language Models (LLMs) 간의 이분법을 해결하기 위해 LLM-ACTR라는 새로운 신경-기호적 (neuro-symbolic) 아키텍처를 도입했습니다. 이는 사람의 의사결정 과정을 모방할 수 있는 지식을 LLM에 주입하여 보다 신뢰성 있는 기계 판단 능력을 확보하는 것을 목표로 합니다.

- **Technical Details**: LLM-ACTR는 ACT-R Cognitive Architecture와 LLM을 통합하여 인간과 유사하지만 유연한 의사결정 기능을 제공합니다. 이 프레임워크는 ACT-R의 내부 의사결정 과정을 잠재적 신경 표현(latent neural representations)으로 추출하고, 이를 조정 가능한 LLM 어댑터 레이어에 주입하여 후속 예측을 위해 LLM을 미세 조정(fine-tuning)합니다.

- **Performance Highlights**: 새로운 설계 제조(Design for Manufacturing) 작업에서 LLM-ACTR의 실험 결과, 이전 LLM-only 모델과 비교하여 작업 성능이 향상되었으며, grounded decision-making 능력이 개선되었음을 확인했습니다.



### Unc-TTP: A Method for Classifying LLM Uncertainty to Improve In-Context Example Selection (https://arxiv.org/abs/2408.09172)
Comments:
          7 pages, long paper

- **What's New**: 이 논문은 LLM의 불확실성을 분류하기 위한 새로운 Paradigm인 Uncertainty Tripartite Testing Paradigm (Unc-TTP)을 제안합니다. Unc-TTP는 LLM 출력의 일관성을 평가하여 LLM의 불확실성을 분류합니다.

- **Technical Details**: Unc-TTP는 세 가지 테스트 시나리오 {no-label, right-label, wrong-label} 하에 LLM의 모든 결과를 열거하고 응답 일관성에 따라 모델 불확실성을 분류하는 접근법입니다. 이를 통해 출력 결과를 82개의 카테고리로 분류합니다.

- **Performance Highlights**: Unc-TTP를 기반으로 한 예시 선택 전략은 LLM의 성능을 개선하며, 기존의 retrieval 기반 방법보다 효율성이 뛰어나고 더 나은 성능 향상을 보여줍니다.



### Selective Prompt Anchoring for Code Generation (https://arxiv.org/abs/2408.09121)
Comments:
          Under review

- **What's New**: 이 논문은 최근 LLMs(대형 언어 모델)의 주의력 감소(self-attention dilution) 문제를 다루며, Selective Prompt Anchoring (SPA)이라는 새로운 방법을 제안합니다. 이는 코드 생성 과정에서 사용자의 초기 프롬프트에 대한 모델의 주의를 극대화하는 접근 방식입니다.

- **Technical Details**: SPA는 선택된 텍스트 부분의 영향을 증폭시켜 코드 생성 시 모델의 출력을 개선합니다. 구체적으로, SPA는 원래의 로짓 분포와 앵커 텍스트가 마스킹된 로짓 분포의 차이를 계산합니다. 이를 통해 앵커 텍스트가 출력 로짓에 기여하는 정도를 근사합니다.

- **Performance Highlights**: SPA를 사용한 결과, 모든 설정에서 Pass@1 비율을 최대 9.7% 향상시킬 수 있었으며, 이는 LLM의 성능을 개선하는 새롭고 효과적인 방법을 제시합니다.



### Measuring Visual Sycophancy in Multimodal Models (https://arxiv.org/abs/2408.09111)
- **What's New**: 이 논문에서는 다중 모달 언어 모델에서 나타나는 '시각적 아첨(visual sycophancy)' 현상을 소개하고 분석합니다. 이는 모델이 이전의 지식이나 응답과 상충할지라도 시각적으로 제시된 정보를 지나치게 선호하는 경향을 설명하는 용어입니다. 연구 결과, 모델이 시각적으로 표시된 옵션을 선호하는 경향이 있음을 발견했습니다.

- **Technical Details**: 이 연구는 여러 가지 모델 아키텍처에서 일관된 패턴으로 시각적 아첨을 정량화할 수 있는 방법론을 적용했습니다. 주요 기술로는 이미지와 선택지의 시각적 강조를 기반으로 한 실험 설계를 통해 모델 응답이 어떻게 변화하는지를 측정했습니다. 실험은 과거 지식을 유지하면서도 시각적 단서를 통해 얼마나 큰 영향을 받는지를 평가하는 방식으로 진행되었습니다.

- **Performance Highlights**: 모델들이 초기 정답을 제공한 후에도 시각적으로 강조된 옵션으로 응답이 치우치는 경향이 나타났습니다. 이는 모델의 신뢰성에 중대한 한계를 드러내며, 비판적인 의사 결정 맥락에서 이러한 현상이 어떻게 영향을 미칠지에 대한 새로운 의문을 제기합니다.



### Learning to Route for Dynamic Adapter Composition in Continual Learning with Language Models (https://arxiv.org/abs/2408.09053)
- **What's New**: 본 논문에서는 PEFT(Parament Efficient Fine-Tuning) 모듈을 새로운 작업에 대해 독립적으로 훈련하고, 메모리로부터 샘플을 활용하여 이전에 학습한 모듈의 조합을 학습하는 L2R(Learning to Route) 방법을 제안합니다. 이는 기존 PEFT 방법의 두 가지 주요 한계를 해결합니다.

- **Technical Details**: L2R 방법은 PEFT 모듈을 학습할 때 이전 작업으로부터의 간섭을 피하고, 테스트 타임 추론 이전에 메모리 내 예시를 활용하여 PEFT 모듈을 동적으로 조합할 수 있는 라우팅(routing) 기능을 학습하는 구조입니다. 이는 로컬 적응(local adaptation)을 활용하여 모듈의 적절한 조합을 수행합니다.

- **Performance Highlights**: L2R은 여러 벤치마크와 CL 설정에서 다른 PEFT 기반 방법들에 비해 개선된 모듈 조합 성능을 보여주며, 이를 통해 일반화 및 성능 향상을 달성했습니다. 본 방법은 기존 우수한 성능을 유지하면서 새로운 작업을 지속적으로 학습하는 모델 개발의 실질적인 목표를 충족시킵니다.



### From Lazy to Prolific: Tackling Missing Labels in Open Vocabulary Extreme Classification by Positive-Unlabeled Sequence Learning (https://arxiv.org/abs/2408.08981)
- **What's New**: Open-vocabulary Extreme Multi-label Classification (OXMC)는 전통적인 XMC의 경계를 넘어 매우 큰 미리 정의된 라벨 세트에서 예측이 가능하도록 확장된 모델입니다. 이 연구에서는 데이터 주석에서 발생하는 자기 선택 편향(self-selection bias)을 해결하기 위해 Positive-Unlabeled Sequence Learning (PUSL) 방식을 도입하였습니다.

- **Technical Details**: PUSL은 OXMC를 무한한 키프레이즈(generation task) 생성 작업으로 재구성하여 생성 모델의 느슨함을 해결합니다. 이 연구에서는 평가 지표로 F1@$	extmathcal{O}$ 및 새롭게 제안된 B@$k$를 활용하여 불완전한 정답과 함께 OXMC 모델을 신뢰성 있게 평가하는 방법을 제안합니다.

- **Performance Highlights**: PUSL은 상당한 라벨이 누락된 비대칭 e-commerce 데이터셋에서 30% 더 많은 고유 라벨을 생성하였고, 예측의 72%가 실제 사용자 쿼리와 일치하였습니다. 비대칭(EURLex-4.3k) 데이터셋에서도 PUSL은 F1 점수를 뛰어난 성과를 보이며, 라벨 수가 15에서 30으로 증가함에 따라 성능이 향상되었습니다.



### Adaptive Guardrails For Large Language Models via Trust Modeling and In-Context Learning (https://arxiv.org/abs/2408.08959)
Comments:
          Under Review

- **What's New**: 이 연구는 사용자 신뢰 지표를 기반으로 민감한 콘텐츠에 대한 접근을 동적으로 조절할 수 있는 적응형 경계 장치(adaptive guardrail) 메커니즘을 도입합니다. 이를 통해 사용자별로 신뢰도를 고려하여 콘텐츠 중재를 조정할 수 있으며, 기존의 정적이지 않은 방식과의 차별점을 제공합니다.

- **Technical Details**: 이 시스템은 Direct Interaction Trust와 Authority Verified Trust의 조합을 활용하여 사용자의 신뢰를 평가하고, 이를 바탕으로 개별 사용자가 알맞은 콘텐츠에 접근할 수 있도록 합니다. 또한, In-context Learning (ICL)을 통해 실시간으로 사용자 쿼리의 민감도에 맞춰 응답을 맞춤형으로 조정하는 기능이 포함되어 있습니다. 이는 LLM의 동적 환경에 잘 적응할 수 있는 경계 장치를 구현하는 데 중요한 역할을 합니다.

- **Performance Highlights**: 실험 평가 결과, 제안된 적응형 경계 장치가 다양한 사용자 요구를 효과적으로 충족시키며, 기존의 경계 장치보다 뛰어난 실용성을 보여주었습니다. 민감한 정보 보호와 잠재적 위험 콘텐츠의 정확한 관리를 통해 LLMs의 안전한 운영을 보장하는데 기여하고 있습니다.



### VerilogCoder: Autonomous Verilog Coding Agents with Graph-based Planning and Abstract Syntax Tree (AST)-based Waveform Tracing Too (https://arxiv.org/abs/2408.08927)
Comments:
          main paper 7 pages, reference 1 page, appendix 22 pages. It is under review of AAAI 2025

- **What's New**: 본 논문에서는 여러 인공지능(AI) 에이전트를 활용하여 Verilog 코드 생성을 자동화하는 시스템인 VerilogCoder를 제안합니다. 이 시스템은 Verilog 코드를 자율적으로 작성하고 문법 및 기능 오류를 수정하며, Verilog 개발 도구를 협력적으로 사용합니다.

- **Technical Details**: VerilogCoder는 새로운 Task and Circuit Relation Graph(TCRG) 기반의 계획자를 이용하여 모듈 설명을 바탕으로 높은 품질의 계획을 작성합니다. 기능 오류를 디버깅하기 위해 새로운 추상 구문 트리(AST) 기반의 파형 추적 도구를 통합합니다. 이 시스템은 최대 94.2%의 문법 및 기능적 정확성을 발휘하며, 최신 방법보다 33.9% 더 향상된 성과를 보여줍니다.

- **Performance Highlights**: VerilogCoder는 VerilogEval-Human v2 벤치마크에서 94.2%의 성공률을 달성하였으며, 이는 기존 최첨단 방법보다 33.9% 높은 성능입니다.



### Cybench: A Framework for Evaluating Cybersecurity Capabilities and Risk of Language Models (https://arxiv.org/abs/2408.08926)
Comments:
          86 pages, 7 figures

- **What's New**: 이 논문에서는 사이버 보안을 위한 언어 모델(LM) 에이전트를 평가하기 위한 새로운 프레임워크인 Cybench를 소개합니다. 이 프레임워크는 에이전트가 자율적으로 취약점을 식별하고 악용(exploit)할 수 있는 능력을 정량화하려고 합니다.

- **Technical Details**: Cybench는 4개의 CTF(Capture the Flag) 대회에서 선택된 40개의 전문 수준의 CTF 과제를 포함하고 있습니다. 각 과제는 설명, 시작 파일, bash 명령을 실행하고 출력을 관찰할 수 있는 환경에서 초기화됩니다. 기존의 LM 에이전트의 한계를 넘어서는 과제들이 많기 때문에, 우리는 17개의 과제에 대해 중간 단계로 나누는 서브태스크(subtask)를 도입하였습니다.

- **Performance Highlights**: 에이전트는 인간 팀이 해결하는 데 최대 11분이 소요된 가장 쉬운 과제만 해결할 수 있었습니다. Claude 3.5 Sonnet과 GPT-4o가 가장 높은 성공률을 기록하였으며, 서브태스크가 있는 경우 전체 과제에서 3.2% 더 높은 성공률을 보였습니다.



### Retail-GPT: leveraging Retrieval Augmented Generation (RAG) for building E-commerce Chat Assistants (https://arxiv.org/abs/2408.08925)
Comments:
          5 pages, 4 figures

- **What's New**: 이번 연구는 소매 전자상거래에서 사용자 참여를 증진하기 위해 제품 추천을 안내하고 장바구니 작업을 지원하는 오픈 소스 RAG(Retrieval-Augmented Generation) 기반 챗봇인 Retail-GPT를 소개합니다.

- **Technical Details**: Retail-GPT는 다양한 전자상거래 도메인에 적응할 수 있는 크로스 플랫폼 시스템으로, 특정 채팅 애플리케이션이나 상업 활동에 의존하지 않습니다. 이 시스템은 인간과 유사한 대화를 통해 사용자 요구를 해석하고, 제품 가용성을 확인하며, 장바구니 작업을 관리합니다.

- **Performance Highlights**: Retail-GPT는 가상 판매 대리인으로 기능하며, 다양한 소매 비즈니스에서 이러한 어시스턴트의 실행 가능성을 테스트하는 것을 목표로 합니다.



### Graph Retrieval-Augmented Generation: A Survey (https://arxiv.org/abs/2408.08921)
Comments:
          Ongoing work

- **What's New**: 이 논문은 Retrieval-Augmented Generation (RAG) 분야의 최신 기술 중 하나인 GraphRAG에 대한 체계적인 리뷰를 제공합니다. GraphRAG는 관계형 지식을 활용하여 더 정확하고 포괄적인 정보 검색을 가능하게 하여, 기존 RAG의 한계를 극복하고자 합니다.

- **Technical Details**: GraphRAG 워크플로우는 Graph-Based Indexing (G-Indexing), Graph-Guided Retrieval (G-Retrieval), Graph-Enhanced Generation (G-Generation)의 세 단계로 구성됩니다. 각 단계에서는 핵심 기술 및 학습 방법을 정리하고 있습니다. GraphRAG는 그래프 데이터베이스에서 적절한 쿼리와 관련된 그래프 요소를 검색하여 응답을 생성할 수 있습니다.

- **Performance Highlights**: GraphRAG는 텍스트 기반 RAG보다 관계형 정보를 더 정확하고 포괄적으로 검색할 수 있으며, 대규모 텍스트 입력의 긴 문맥 문제를 해결합니다. 또한, 다양한 산업 분야에서의 활용 가능성이 커지고 있으며, 연구의 초기 단계이지만, GraphRAG의 잠재력을 통해 새로운 연구 방향을 제시하고 있습니다.



### What should I wear to a party in a Greek taverna? Evaluation for Conversational Agents in the Fashion Domain (https://arxiv.org/abs/2408.08907)
Comments:
          Accepted at KDD workshop on Evaluation and Trustworthiness of Generative AI Models

- **What's New**: 이번 연구는 대형 언어 모델(LLM)이 온라인 패션 소매 분야에서 고객 경험 및 패션 검색을 개선하는 데 미치는 실질적인 영향을 탐구합니다. LLM을 활용한 대화형 에이전트는 고객과 직접 상호작용함으로써 그들의 니즈를 표현하고 구체화할 수 있는 새로운 방법을 제공합니다. 이는 고객이 자신의 패션 취향과 의도에 맞는 조언을 받을 수 있도록 합니다.

- **Technical Details**: 우리는 4,000개의 다국어 대화 데이터셋을 구축하여 고객과 패션 어시스턴트 간의 상호작용을 평가합니다. 이 데이터는 LLM 기반의 고객 에이전트가 어시스턴트와 대화하는 형태로 생성되며, 대화의 주제나 항목을 통해 서로의 요구를 파악하고 적절한 상품을 제안하는 방식으로 진행됩니다. 데이터셋은 영어, 독일어, 프랑스어 및 그리스어를 포함하며, 다양한 패션 속성(색상, 유형, 소재 등)을 기준으로 구축되었습니다.

- **Performance Highlights**: 이 연구는 LLM이 고객 요구에 맞는 패션 아이템을 추천하는 데 얼마나 효과적인지를 평가하기 위해 여러 공개 및 비공개 모델(GPT, Llama2, Mistral 등)을 벤치마킹하였습니다. 이로써 패션 분야의 대화형 에이전트가 고객과 백엔드 검색 엔진 간의 강력한 인터페이스 역할을 할 수 있음을 입증하였습니다.



### Kov: Transferable and Naturalistic Black-Box LLM Attacks using Markov Decision Processes and Tree Search (https://arxiv.org/abs/2408.08899)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLM)의 해로운 행동을 유도하기 위해 Markov decision process (MDP)로 문제를 수렴시킨 후, Monte Carlo tree search를 활용하여 블랙박스 모델에서 해로운 행동을 탐색합니다. 자동화된 기법을 통해 자연적인 공격을 위해 log-perplexity를 포함하고 있으며, 이는 더 해석 가능한 공격 생성에 기여합니다.

- **Technical Details**: 연구는 기존의 토큰 레벨 공격 방법으로부터 발전하여 자연어 공격을 최적화하는 방법을 제안합니다. 특히, 블랙박스 모델인 GPT-3.5를 대상으로 프로토타입 테스트를 수행했으며, 10개의 쿼리만으로 jailbreak를 성공적으로 달성하였습니다. 반면, GPT-4 모델에서는 실패하였다.

- **Performance Highlights**: 사전 연구 결과에 따르면, 제안된 알고리즘은 블랙박스 모델에 대한 공격을 최적화하는 데 성공적이며, 최신 모델들이 토큰 수준 공격에 대한 강력함을 보이는 경향을 확인하였습니다. 연구 결과는 모두 오픈 소스 형태로 제공됩니다.



### Enhancing Exploratory Learning through Exploratory Search with the Emergence of Large Language Models (https://arxiv.org/abs/2408.08894)
Comments:
          11 pages, 7 figures

- **What's New**: 이 논문에서는 정보 시대에서 학습자들이 정보 검색 및 활용 방식을 이해하는 데 필요한 새로운 이론적 모델을 제안합니다. 특히, 대규모 언어 모델(LLMs)의 영향을 고려하여 탐색적 학습(exploratory learning) 이론과 탐색적 검색 전략을 결합하였습니다.

- **Technical Details**: 이 연구는 Kolb의 학습 모델을 개선하여 높은 빈도로 탐색할 수 있는 전략과 피드백 루프(feedback loops)를 통합합니다. 이러한 접근법은 학생들의 깊은 인지(deep cognitive) 및 고차원 인지(higher-order cognitive) 기술 발달을 촉진하는 것을 목표로 합니다.

- **Performance Highlights**: 이 논문은 LLMs가 정보 검색(information retrieval) 및 정보 이론(information theory)에 통합되어 학생들이 효율적으로 탐색적 검색을 수행할 수 있도록 지원한다고 논의합니다. 이론적으로는 학생-컴퓨터 상호작용을 촉진하고 새로운 시대의 학습 여정을 지원하는 데 기여하고자 합니다.



New uploads on arXiv(cs.IR)

### Customizing Language Models with Instance-wise LoRA for Sequential Recommendation (https://arxiv.org/abs/2408.10159)
- **What's New**: 본 연구는 Instance-wise LoRA (iLoRA)라는 새로운 방법론을 제안하며 이를 통해 다양하고 개별화된 사용자 행동을 효과적으로 반영할 수 있는 추천 시스템을 구축하고자 한다.

- **Technical Details**: iLoRA는 Mixture of Experts (MoE) 프레임워크를 통합하여 Low-Rank Adaptation (LoRA) 모듈을 개선하는 방식으로 작동한다. 이 방법은 다양한 전문가를 생성하며 각 전문가는 특정 사용자 선호를 반영하도록 훈련된다. 또한, 게이팅 네트워크는 사용자 상호작용 시퀀스를 기반으로 맞춤형 전문가 참여 가중치를 생성하여 다채로운 행동 패턴에 적응한다.

- **Performance Highlights**: 세 가지 벤치마크 데이터셋(LastFM, MovieLens, Steam)에서의 실험을 통해 iLoRA가 기존 방법론들보다 탁월한 성능을 보이며 사용자별 선호를 효과적으로 캡처하고 추천의 정확성을 향상시켰음을 입증하였다.



### Efficient Inference of Sub-Item Id-based Sequential Recommendation Models with Millions of Items (https://arxiv.org/abs/2408.09992)
Comments:
          Accepted by RecSys 2024

- **What's New**: 이번 논문은 RecJPQ를 기반으로 한 Transformer 기반의 추천 모델의 추론 효율성을 향상시키기 위한 PQTopK 알고리즘을 제안합니다. 이 연구를 통해 기존 RecJPQ 모델의 메모리 소모를 줄이는 효과에 이어 속도 개선 또한 이루어졌습니다.

- **Technical Details**: RecJPQ는 아이템 ID를 공유된 서브 아이템 ID로 분해하여 아이템 카탈로그를 압축하는 방법으로, 메모리 사용량을 최대 50배까지 줄이는데 성공했습니다. 그러나 RecJPQ의 스코어링 알고리즘은 각 아이템에 대한 스코어 누적기를 사용하여 병렬 처리에 제약이 있었습니다. PQTopK는 효율적인 아이템 스코어 계산을 위한 알고리즘으로, RecJPQ 기반 모델의 추론 효율성을 향상시킬 수 있습니다.

- **Performance Highlights**: SASRec을 RecJPQ로 개선한 모델은 원본 SASRec 추론 방법에 비해 4.5배, RecJPQ 코드 내부의 방법에 비해 1.56배 더 빠른 추론이 가능합니다. 또한 PQTopK는 수천만 개의 아이템이 포함된 카탈로그에서도 효율적으로 작동할 수 있습니다.



### Fashion Image-to-Image Translation for Complementary Item Retrieva (https://arxiv.org/abs/2408.09847)
- **What's New**: 이 논문에서는 패션 호환성 모델링 및 아이템 검색을 개선하기 위해 Generative Compatibility Model (GeCo)을 소개합니다. GeCo는 쌍 이미지 간 변환(pair image-to-image translation) 방법을 활용하여 패션 이미지 검색의 성능을 향상시킵니다.

- **Technical Details**: GeCo 모델은 두 단계로 구성되어 있으며, 첫 번째 단계에서는 Conditional Generative Adversarial Networks (GANs) 기반의 Complementary Item Generation Model (CIGM)을 사용하여 시드 아이템에서 타겟 아이템 이미지를 생성합니다. 두 번째 단계에서는 이 생성된 샘플을 사용하여 호환성 모델링 및 검색을 수행합니다.

- **Performance Highlights**: GeCo는 세 가지 데이터셋에서 평가되어 기존의 최첨단 모델보다 우수한 성능을 보였습니다. 또한, Fashion Taobao 데이터셋의 새로운 버전을 공개하여 추가적인 연구를 위한 기반을 마련했습니다.



### Ranking Generated Answers: On the Agreement of Retrieval Models with Humans on Consumer Health Questions (https://arxiv.org/abs/2408.09831)
- **What's New**: 이 논문은 Generative Large Language Models (LLMs)의 출력을 평가하는 새로운 방법론을 제안합니다. 이전의 평가 방식들은 주로 단일 선택 질문 응답 또는 텍스트 분류에 초점을 맞추고 있었으나, 저자들은 개방형 질문 응답 능력을 평가하는 방법론의 필요성을 강조합니다.

- **Technical Details**: 직접적인 관련성 판단 없이 랭킹 신호(ranking signals)를 사용하여 LLM의 답변을 평가하는 방법을 제안했습니다. 이 방법은 이미 알려진 기계 번역의 랭킹 기반 평가 방법에서 착안하였습니다. 평가에서는 Normalized Rank Position (NRP)라는 지표를 사용하여 생성된 답변의 유효성을 측정합니다.

- **Performance Highlights**: 실험을 통해 LLM의 질은 모델의 크기와 보다 정교한 프롬프트 전략에 따라 개선된다는 사실을 확인했습니다. 이 연구는 소비자 건강 분야에 중점을 두었으며, 제공된 랭킹 방법이 전문가의 순위와 어떻게 일치하는지를 검증하기 위한 사용자 연구도 포함되어 있습니다.



### Contextual Dual Learning Algorithm with Listwise Distillation for Unbiased Learning to Rank (https://arxiv.org/abs/2408.09817)
Comments:
          12 pages, 2 figures

- **What's New**: 본 논문에서는 Baidu의 웹 검색 로그를 기반으로 한 실제 클릭 데이터에서 기존의 Unbiased Learning to Rank (ULTR) 방법의 효과를 평가하였습니다. 새로운 Contextual Dual Learning Algorithm with Listwise Distillation (CDLA-LD)을 제안하여 위치 편향(position bias)과 맥락 편향(contextual bias)을 동시에 해결합니다.

- **Technical Details**: CDLA-LD 알고리즘은 두 가지 랭킹 모델을 포함합니다: (1) self-attention 메커니즘을 활용하는 listwise-input 랭킹 모델과 (2) pointwise-input 랭킹 모델입니다. DLA를 사용하여 unbiased listwise-input 랭킹 모델과 pointwise-input 모델을 동시에 학습하며, listwise 방식으로 지식을 증류(distill)합니다.

- **Performance Highlights**: 실험 결과, CDLA-LD는 기존의 ULTR 방법들에 비해 뛰어난 성능을 보여주었으며, 다양한 방법들의 propensity 학습을 비교 분석했습니다. 또한, 실제 검색 로그를 활용한 연구에 대한 새로운 통찰을 제공합니다.



### Revisiting Reciprocal Recommender Systems: Metrics, Formulation, and Method (https://arxiv.org/abs/2408.09748)
Comments:
          KDD 2024

- **What's New**: 이 논문은 Reciprocal Recommender Systems (RRS)을 체계적으로 재조명하고, 새로운 평가 지표와 방법론을 제시합니다. 특히, 이전 연구에서는 각 측면을 독립적으로 평가하였으나, 본 연구는 양측의 추천 결과가 시스템의 효과성에 미치는 영향을 포괄적으로 고려합니다.

- **Technical Details**: 연구는 세 가지 관점에서 RRS의 성능을 종합적으로 평가하는 5개의 새로운 지표를 제안합니다: 전체 커버리지(Overall Coverage), 양측 안정성(Bilateral Stability), 그리고 균형 잡힌 순위(Balanced Ranking). 또한, 잠재적 결과 프레임워크를 사용해 모델 비의존적인 인과적 RRS 방법인 Causal Reciprocal Recommender System (CRRS)를 개발하였습니다.

- **Performance Highlights**: 두 개의 실제 데이터셋에서 시행된 광범위한 실험을 통해 제안한 지표와 방법의 효과성을 입증하였습니다. 특히, 추천의 중복성을 고려하는 새로운 평가 방식이 RRS의 전반적인 성능 향상에 기여함을 보여주었습니다.



### Carbon Footprint Accounting Driven by Large Language Models and Retrieval-augmented Generation (https://arxiv.org/abs/2408.09713)
- **What's New**: 이번 논문은 탄소 발자국 계산(carbon footprint accounting)의 실시간 업데이트를 위해 대형 언어 모델(large language models, LLMs)과 검색 증강 생성(retrieval-augmented generation, RAG) 기술을 통합한 새로운 접근 방식을 제안합니다. 이는 전통적인 방법의 한계를 극복하고 더 나은 정보 검색 및 분석을 가능하게 합니다.

- **Technical Details**: 제안된 LLMs-RAG-CFA 방법은 LLMs의 논리적 및 언어 이해 능력과 RAG의 효율적인 검색 능력을 활용하여 탄소 발자국 정보를 보다 관련성 높게 검색합니다. 이를 통해 전문 정보의 폭넓은 커버리지와 효율적인 실시간 정보 획득, 그리고 비용 효율적인 자동화를 제공합니다.

- **Performance Highlights**: 실험 결과, LLMs-RAG-CFA 방법은 다섯 개 산업(1차 알루미늄, 리튬 배터리, 태양광 발전, 신에너지 차량, 변압기)에서 전통적인 방법 및 다른 LLMs보다 높은 정보 검색률을 달성하였으며, 정보 편차 및 탄소 발자국 계산 편차가 현저히 낮아지는 성과를 보였습니다.



### Harnessing Multimodal Large Language Models for Multimodal Sequential Recommendation (https://arxiv.org/abs/2408.09698)
- **What's New**: 본 논문에서는 Multimodal Large Language Model-enhanced Sequential Multimodal Recommendation (MLLM-MSR) 모델을 제안합니다. 이 모델은 사용자 동적 선호를 반영하기 위해 두 단계의 사용자 선호 요약 방법을 설계하였습니다.

- **Technical Details**: MLLM-MSR은 MLLM 기반의 아이템 요약기를 통해 이미지를 텍스트로 변환하여 이미지 피처를 추출합니다. 이후 LLM 기반의 사용자 요약기를 사용하여 사용자 선호의 동적 변화를 포착합니다. 마지막으로 Supervised Fine-Tuning (SFT) 기법을 통해 추천 시스템의 성능을 향상시킵니다.

- **Performance Highlights**: 다양한 데이터셋에 대한 광범위한 평가를 통해 MLLM-MSR의 효과성을 검증하였으며, 사용자 선호의 변화를 정확하게 반영하고 추천의 개인화 및 정확성을 높이는 우수한 성능을 보여주었습니다.



### GANPrompt: Enhancing Robustness in LLM-Based Recommendations with GAN-Enhanced Diversity Prompts (https://arxiv.org/abs/2408.09671)
- **What's New**: 본 논문에서는 추천 시스템의 적응성과 안정성을 개선하기 위해 GAN 기반의 다차원 LLM 프롬프트 다양성 프레임워크인 GANPrompt를 제안합니다.

- **Technical Details**: GANPrompt는 사용자 행동 데이터를 분석하여 다양한 프롬프트를 생성하는 생성기를 훈련시키고, 이 생성된 프롬프트를 사용하여 LLM을 훈련시킵니다. 이를 통해 모델은 미지의 프롬프트에 대해서도 성능을 향상시킬 수 있습니다. 또한, 수학 이론 기반의 다양성 제약 메커니즘을 도입하여 생성된 프롬프트의 다양성과 관련성을 보장합니다.

- **Performance Highlights**: 다수의 데이터셋에서의 실험을 통해 GANPrompt가 기존의 최첨단 방법론에 비해 정확성과 강인성이 상당히 개선됨을 입증하였습니다.



### Data-driven Conditional Instrumental Variables for Debiasing Recommender Systems (https://arxiv.org/abs/2408.09651)
- **What's New**: 이 논문에서는 추천 시스템에서의 잠재 변수(latent variables)로 인한 편향을 해결하기 위해 새로운 데이터 기반의 조건부 기기 변수(Conditional Instrumental Variables, CIV) 디바이싱 방법인 CIV4Rec을 제안합니다. 이 방법은 상호작용 데이터에서 유효한 CIV와 그에 해당하는 조건 집합을 자동 생성하여 IV 선택의 복잡성을 줄입니다.

- **Technical Details**: CIV4Rec은 변분 자동 인코더(Variational Autoencoder, VAE)를 활용하여 상호작용 데이터로부터 CIV 및 조건 집합의 표현을 생성합니다. 또한 최소 제곱법(Least Squares)을 적용하여 클릭 예측을 위한 인과적 표현(causal representations)을 도출합니다. 이 방법은 사용자-아이템 쌍의 임베딩을 치료 변수(treatment variable)로 사용하고, 사용자 피드백을 결과(outcome)로 사용하여 편향을 완화합니다.

- **Performance Highlights**: Movielens-10M 및 Douban-Movie 두 개의 실세계 데이터셋에서 진행된 광범위한 실험 결과, CIV4Rec은 유효한 CIV를 성공적으로 식별하고 편향을 효과적으로 줄이며 추천 정확도를 향상시킵니다. 또한 기존의 인과적 디바이싱 방법들과 비교했을 때 최적의 디바이싱 결과와 추천 성능을 달성하였습니다.



### Debiased Contrastive Representation Learning for Mitigating Dual Biases in Recommender Systems (https://arxiv.org/abs/2408.09646)
- **What's New**: 이 논문은 추천 시스템에서 발생하는 인기 편향(popularity bias)과 동조 편향(conformity bias)을 동시에 처리하기 위한 새로운 접근 방식을 제시합니다. 기존의 연구들은 대개 한 가지 편향만을 다루었지만, 본 연구는 두 가지 편향이 동시에 존재한다는 점에 주목하고 이를 해결하기 위한 구조적 모델을 제안합니다.

- **Technical Details**: 제안된 방법은 DCLMDB(Debiased Contrastive Learning framework for Mitigating Dual Biases)라는 새로운 학습 프레임워크를 기반으로 합니다. DCLMDB는 사용자 선택 및 추천 항목이 인기와 동조에 의해 부당하게 영향을 받지 않도록 하기 위해 대조 학습(contrastive learning)을 사용합니다. 또한, 인과 그래프(causal graph)를 활용하여 두 가지 편향의 발생 기전을 모델링하고 이를 통해 사용자와 항목 간의 상호작용을 정제합니다.

- **Performance Highlights**: Movielens-10M과 Netflix의 두 가지 실제 데이터 세트에서 수행된 광범위한 실험을 통해 DCLMDB 모델이 이중 편향을 효과적으로 감소시키고 추천의 정확성과 다양성을 크게 향상시킬 수 있음을 보여줍니다.



### Towards Boosting LLMs-driven Relevance Modeling with Progressive Retrieved Behavior-augmented Prompting (https://arxiv.org/abs/2408.09439)
- **What's New**: 본 논문은 LLMs(대형 언어 모델)를 활용한 적합성 모델링(relevance modeling)에 사용자 행동 데이터를 통합하는 새로운 접근 방식을 제안합니다. 사용자의 검색 로그를 통한 사용자 상호작용을 기반으로 한 ProRBP(Progressive Retrieved Behavior-augmented Prompting) 프레임워크를 운영하여 적합성 판단을 개선합니다.

- **Technical Details**: ProRBP는 데이터 기반의 사용자 행동 이웃 검색을 통해 신속하게 도메인 전용 지식을 획득하고, 이를 LLM의 출력 개선을 위한 진보적인 프롬프트(prompting) 기법과 결합하여 적합성 모델을 형성합니다. 이 과정은 다양한 측면들을 체계적으로 고려하여 진행됩니다.

- **Performance Highlights**: 실제 산업 데이터를 통한 실험과 온라인 A/B 테스트에서 본 연구가 제안하는 ProRBP 프레임워크가 기존 방법들(Chen et al., 2023) 대비 우수한 성능을 보여 주목을 받고 있습니다.



### Hindi-BEIR : A Large Scale Retrieval Benchmark in Hind (https://arxiv.org/abs/2408.09437)
- **What's New**: 본 논문에서는 인도에서 사용되는 힌디어 정보를 검색하기 위한 새로운 기준인 Hindi-BEIR를 소개합니다. 이 기준은 영어 BEIR 분석 데이터의 하위 집합, 기존 힌디어 검색 데이터 세트, 합성 데이터 세트를 포함하여 15개의 데이터세트와 8개의 다양한 작업을 포함합니다.

- **Technical Details**: Hindi-BEIR는 6개의 서로 다른 도메인에서 온 15개의 데이터 세트로 구성되며, 이는 데이터 검색 모델을 평가하고 비교하기 위한 포괄적인 벤치마크 제공을 목표로 합니다. 또한, 다국어 검색 모델에 대한 기준 성능 비교를 통해 향후 연구 방향을 제안합니다.

- **Performance Highlights**: 이 벤치마크는 현재를 기준으로 하는 다국어 검색 모델의 성능을 비교하고 고유한 도전 과제를 발견함으로써 힌디어 정보 검색 시스템의 발전에 기여할 것입니다. Hindi-BEIR는 공개적으로 이용 가능하며, 연구자들이 현재 힌디어 검색 모델의 한계와 가능성을 이해하는 데 도움을 줄 것입니다.



### Gender Dynamics in Russian Online Political Discours (https://arxiv.org/abs/2408.09378)
- **What's New**: 이번 연구는 러시아-우크라이나 전쟁 동안 유튜브 사용자 행동을 분석하여 공공 의견과 참여의 변화를 이해하려는 새로운 접근을 제공합니다. 2168개의 비디오와 36000개 이상의 댓글을 분석한 결과, 여성들이 특히 고조된 갈등 기간 동안 반정부 채널에서 더 활발하다는 발견이 있었습니다.

- **Technical Details**: 이 연구는 디지털 플랫폼에서의 정치적 담론과 성별 역학의 상관관계를 설명합니다. 인터넷 사용이 정보의 접근성을 민주화하고 사용자가 정치적 담론에 참여할 수 있는 기회를 제공함으로써 자율성을 강화하는 방식에 대해 논의합니다. 이와 함께, 다양한 온라인 행동(예: 댓글 작성) 분석을 통해 정치적 경향성을 파악하는 방법론을 제시합니다.

- **Performance Highlights**: 연구 결과, 여성들은 온라인에서 주요 정치적 사건과 군사적 상황에 대한 반응에서 중요한 디지털 소통자로 부각되었으며, 이는 권위주의 정부 하에서도 정치적 표현의 가능성을 강조하는 중요한 발견으로 이어집니다. 이 연구는 디지털 플랫폼의 사용이 시민들의 정치적 참여와 의견 표현에 미치는 영향을 보여줍니다.



### Deep Code Search with Naming-Agnostic Contrastive Multi-View Learning (https://arxiv.org/abs/2408.09345)
- **What's New**: 본 논문은 코드 검색 코드 스니펫의 네이밍 규칙이 서로 다를 때 발생하는 문제를 해결하기 위한 새로운 방법인 네이밍 비독립적 코드 검색 방법(NACS)을 제안합니다.

- **Technical Details**: NACS는 대조적 다중 뷰 코드 표현 학습(contrastive multi-view code representation learning)을 기반으로 하며, 추상 구문 트리(Abstract Syntax Tree, AST)에서 변수 이름에 묶인 정보를 제거하여 AST 구조에서 본질적인 속성을 파악하는 데 초점을 둡니다. 이 방법은 의미 수준 및 구문 수준의 증강 기법을 사용하고, 대조 학습(contrastive learning)을 통해 코드 스니펫 이해를 향상시키기 위한 그래프 뷰 모델링 구성 요소를 설계합니다.

- **Performance Highlights**: NACS는 기존 코드 검색 방법보다 우수한 성능을 자랑하며, 네이밍 문제에 직면했을 때도 성능 저하가 없음을 보여줍니다. 이는 NACS의 전반적인 성능이 기존 코드 검색 방법보다 뛰어나고, 여러 뷰 학습을 통해 그래프 뷰 모델링 구성 요소의 강화를 가능하게 합니다.



### A Study of PHOC Spatial Region Configurations for Math Formula Retrieva (https://arxiv.org/abs/2408.09283)
- **What's New**: 이번 연구에서는 피라미드 문자인코딩(Phoc, Pyramidal Histogram Of Characters) 모델에 동심원 사각형 영역을 추가하는 방법을 제안하고, PHOC 벡터에서 레벨 생략이 정보의 중복성을 줄이는지를 분석합니다.

- **Technical Details**: PHOC는 수식의 공간적 위치를 이진 벡터로 표현하는 방법으로, 여러 개의 유형(사각형, 타원형 등)의 구역으로 나누어진 레벨로 구성됩니다. 새로운 연구에서 PHOC 구성에서 레벨을 생략함으로써 처음으로 표현된 너비의 구역을 가진 동심원 사각형에 대해 연구하였고, ARQMath-3 공식 검색 벤치마크를 활용하여 각 레벨의 기여를 분석하였습니다.

- **Performance Highlights**: 연구 결과, 원래의 PHOC 구성에서 일부 레벨은 중복된 정보를 포함하고 있으며, 직사각형 영역을 사용하는 PHOC 모델이 이전 모델들을 초과하는 성능을 보였습니다. 간단함에도 불구하고 PHOC 모델은 최첨단 기술과 놀라울 정도로 경쟁력이 있음을 보여주었습니다.



### Towards Effective Top-N Hamming Search via Bipartite Graph Contrastive Hashing (https://arxiv.org/abs/2408.09239)
- **What's New**: 새롭게 제안된 Bipartite Graph Contrastive Hashing (BGCH+) 모델은 그래프 구조에서의 학습 효율성을 극대화하고 정보 손실을 최소화하기 위해 자기 감독 학습(self-supervised learning) 기법을 도입합니다.

- **Technical Details**: BGCH+는 이중 강화(feature augmentation) 접근 방식을 사용하여 중간 정보 및 해시 코드 출력을 보강하며, 이는 잠재적 피처 공간(latent feature spaces) 내에서 더욱 표현력 강하고 견고한 해시 코드를 생성합니다.

- **Performance Highlights**: BGCH+는 6개의 실제 벤치마크에서 기존 방식에 비해 성능을 향상시킨 것으로 검증되었으며, Hamming 공간에서의 Top-N 검색에서 기존의 풀 정밀도(full-precision) 모델과 유사한 예측 정확도를 유지하면서 계산 속도를 8배 이상 가속화할 수 있음을 보여줍니다.



### Hybrid Semantic Search: Unveiling User Intent Beyond Keywords (https://arxiv.org/abs/2408.09236)
- **What's New**: 이 논문은 전통적인 키워드 기반 검색의 한계를 극복하고 사용자의 의도를 이해하기 위한 새로운 하이브리드 검색 접근 방식을 소개합니다.

- **Technical Details**: 제안된 시스템은 키워드 매칭, 의미적 벡터 임베딩(semantic vector embeddings), 그리고 LLM(대규모 언어 모델) 기반의 구조화된 쿼리를 통합하여 고도로 관련성 높고 맥락에 적절한 검색 결과를 제공합니다.

- **Performance Highlights**: 이 하이브리드 검색 모델은 포괄적이고 정확한 검색 결과를 생성하는 데 효과적이며, 쿼리 실행 최적화를 통해 응답 시간을 단축할 수 있는 기술을 탐구합니다.



### FabricQA-Extractor: A Question Answering System to Extract Information from Documents using Natural Language Questions (https://arxiv.org/abs/2408.09226)
- **What's New**: 본 논문에서는 Relation Coherence 모델을 도입하여 비구조적 텍스트에서 구조적 데이터를 추출하는 새로운 방법을 제안합니다. 기존의 정보 추출 기법들이 관계 구조를 활용하지 못한 반면, 새로운 시스템인 FabricQA-Extractor는 대규모 데이터에서 효율적으로 정보를 추출합니다.

- **Technical Details**: Relation Coherence 모델은 데이터베이스의 관계(schema)에 대한 지식을 활용하여 정보 추출 품질을 향상시키고, FabricQA-Extractor는 입력된 수백만 개의 문서에서 각각의 셀에 대한 답변을 신속하게 제공할 수 있는 엔드투엔드 시스템입니다. 이를 통해 Passage Ranker, Answer Ranker와 Relation Coherence 모델을 결합하여 정답을 선별합니다.

- **Performance Highlights**: Relation Coherence 모델을 적용한 결과, QA-ZRE와 BioNLP 데이터셋을 사용하여 대규모 정보 추출 작업에서 품질이 현저히 향상되었음을 입증하였으며, 이는 생물의학 텍스트와 같은 도메인 변경 시에도 잘 작동합니다.



### TC-RAG:Turing-Complete RAG's Case study on Medical LLM Systems (https://arxiv.org/abs/2408.09199)
Comments:
          version 1.0

- **What's New**: 새로운 RAG 프레임워크인 TC-RAG가 시스템 상태 변수를 포함하여 더 효율적이고 정확한 지식 검색을 가능하게 함.

- **Technical Details**: TC-RAG는 Turing 완전(Turing Complete) 시스템을 적용하여 상태 변수를 관리하고, 메모리 스택 시스템을 통해 적응형 검색, 추론 및 계획 기능을 제공, 잘못된 지식의 축적을 방지하기 위한 Push 및 Pop 작업을 수행.

- **Performance Highlights**: 의료 도메인에 대한 실험 결과, TC-RAG는 기존 방법들보다 정확도가 7.20% 이상 향상됨을 입증하며, 관련 데이터셋과 코드를 제공함.



### Ranking Across Different Content Types: The Robust Beauty of Multinomial Blending (https://arxiv.org/abs/2408.09168)
Comments:
          To appear in 18th ACM Conference on Recommender Systems (RecSys24), Bari, Italy. ACM, New York, NY, USA, 3 pages

- **What's New**: 이번 논문에서는 다양한 콘텐츠 유형 간의 순위를 매기는 문제를 다루고, 기존의 learning-to-rank (LTR) 알고리즘과 함께 사용할 수 있는 multinomial blending (MB)이라는 새로운 방식에 대해 소개합니다.

- **Technical Details**: 이 논문은 음악과 팟캐스트 등 서로 다른 콘텐츠 유형의 아이템을 순위 매기는 방법을 제안합니다. 특정 콘텐츠 유형의 사용자 참여 패턴이 다르기 때문에 전통적인 LTR 알고리즘이 효과적이지 않은 문제를 해결하고자 하며, 비즈니스 목표를 달성하기 위해 해석 가능성과 용이성을 중시한 MB 방안을 채택하였습니다. MB는 단일 점수 함수 h를 사용하여 모든 후보 아이템을 점수 매긴 후, 특정 콘텐츠 유형에 따라 샘플링하여 순위를 결정합니다.

- **Performance Highlights**: Amazon Music의 실제 사례를 통해 A/B 테스트를 실시하여 MB 방법이 사용자 참여를 어떻게 강화하는지에 대해 보고하고 있습니다. MB는 특정 콘텐츠 유형의 노출을 증가시키면서도 전체 참여율을 해치지 않는 방향으로 최적화되었습니다.



### Meta Knowledge for Retrieval Augmented Large Language Models (https://arxiv.org/abs/2408.09017)
Comments:
          Accepted in Workshop on Generative AI for Recommender Systems and Personalization, KDD 2024

- **What's New**: 본 연구에서는 Retrieval Augmented Generation (RAG) 시스템의 새로운 데이터 중심 워크플로우인 prepare-then-rewrite-then-retrieve-then-read (PR3)를 제안합니다. 이 시스템은 기존의 retrieve-then-read 구조와 다르게, 도메인 전문가 수준의 지식 이해를 목표로 하여 문서의 메타데이터와 합성 질문-답변(QA)을 생성합니다.

- **Technical Details**: PR3 워크플로우는 각 문서에 대해 사용자 맞춤형 메타데이터와 QA 쌍을 생성하며, Meta Knowledge Summary(MK Summary)를 도입하여 문서 클러스터를 메타데이터 기반으로 구성합니다. 이 접근 방식은 고차원의 벡터 공간에서 쿼리와 문서를 개별적으로 인코딩하고 유사성을 측정하기 위해 내적(inner product)을 계산하는 이중 인코더 밀집 검색 모델을 사용합니다. 또한, 이 연구는 LLMs를 평가자로 활용하고 새로운 성능 비교 메트릭을 도입하여 RAG 파이프라인의 효율성을 높였습니다.

- **Performance Highlights**: Synthetic question matching을 포함한 추가 쿼리를 사용함으로써 전통적인 RAG 파이프라인에 비해 문서 청크 방식보다 유의미하게 성능이 향상되었으며(p < 0.01), 메타 지식이 증강된 쿼리는 검색 정밀도와 재현율을 개선하고, 최종 답변의 폭과 깊이, 관련성 및 특정성을 더욱 높였습니다. 연구 결과는 Claude 3 Haiku를 사용하여 2000개의 연구 논문에 대해 20달러 이하의 비용으로 수행 가능하였습니다.



### From Lazy to Prolific: Tackling Missing Labels in Open Vocabulary Extreme Classification by Positive-Unlabeled Sequence Learning (https://arxiv.org/abs/2408.08981)
- **What's New**: Open-vocabulary Extreme Multi-label Classification (OXMC)는 전통적인 XMC의 경계를 넘어 매우 큰 미리 정의된 라벨 세트에서 예측이 가능하도록 확장된 모델입니다. 이 연구에서는 데이터 주석에서 발생하는 자기 선택 편향(self-selection bias)을 해결하기 위해 Positive-Unlabeled Sequence Learning (PUSL) 방식을 도입하였습니다.

- **Technical Details**: PUSL은 OXMC를 무한한 키프레이즈(generation task) 생성 작업으로 재구성하여 생성 모델의 느슨함을 해결합니다. 이 연구에서는 평가 지표로 F1@$	extmathcal{O}$ 및 새롭게 제안된 B@$k$를 활용하여 불완전한 정답과 함께 OXMC 모델을 신뢰성 있게 평가하는 방법을 제안합니다.

- **Performance Highlights**: PUSL은 상당한 라벨이 누락된 비대칭 e-commerce 데이터셋에서 30% 더 많은 고유 라벨을 생성하였고, 예측의 72%가 실제 사용자 쿼리와 일치하였습니다. 비대칭(EURLex-4.3k) 데이터셋에서도 PUSL은 F1 점수를 뛰어난 성과를 보이며, 라벨 수가 15에서 30으로 증가함에 따라 성능이 향상되었습니다.



### RoarGraph: A Projected Bipartite Graph for Efficient Cross-Modal Approximate Nearest Neighbor Search (https://arxiv.org/abs/2408.08933)
Comments:
          to be published in PVLDB

- **What's New**: 본 논문에서는 교차 모드 (cross-modal) 최근접 이웃 검색(ANNS)에서의 비효율성을 분석하고, query 분포 (query distribution)를 활용한 새로운 그래프 인덱스인 RoarGraph를 제안합니다.

- **Technical Details**: RoarGraph는 bipartite graph를 사용하여 쿼리와 base 데이터 간의 유사성 관계를 그래프 구조로 매핑하고, neighbor-aware projection을 통해 공간적으로 먼 노드 간 경로를 생성합니다. 이 그래프 인덱스는 base 데이터만으로 구성되어 있으며 쿼리 분포로부터 파생된 이웃 관계를 효과적으로 보존합니다.

- **Performance Highlights**: RoarGraph는 현대의 교차 모드 데이터셋에서 기존의 ANNS 방법들에 비해 최대 3.56배 빠른 검색 속도를 달성하며, OOD 쿼리의 90% 재현율을 확보합니다.



### Personalized Federated Collaborative Filtering: A Variational AutoEncoder Approach (https://arxiv.org/abs/2408.08931)
Comments:
          10 pages, 3 figures, 4 tables, conference

- **What's New**: 본 논문에서는 Federated Collaborative Filtering (FedCF) 분야에서 개인 정보를 보호하면서 추천 시스템을 향상시키기 위한 새로운 방법론을 제안합니다. 기존의 방법들이 사용자의 개인화된 정보를 사용자 임베딩 벡터에 집약하는 데 그쳤다면, 본 연구는 이를 잠재 변수와 신경 모델로 동시에 보존하는 방안을 모색합니다.

- **Technical Details**: 제안된 방법은 사용자 지식을 두 개의 인코더로 분해하여, 아키텍처 내에서 공유 지식과 개인화를 나누어 캡처합니다. 전역 인코더는 모든 클라이언트 간에 공유되는 일반적인 잠재 공간에 사용자 프로필을 매핑하며, 개인화된 로컬 인코더는 사용자-specific 잠재 공간으로 매핑합니다. 이를 통해 개인화와 일반화를 균형 있게 처리합니다. 또한, 추천 시스템의 특수한 Variational AutoEncoder (VAE) 작업으로 모델링하고, 사용자의 상호작용 벡터 재구성과 누락된 값 예측을 통합하여 훈련합니다.

- **Performance Highlights**: 제안된 FedDAE 방법은 여러 벤치마크 데이터셋에서 수행한 실험 결과, 기존의 기준 방법들보다 우수한 성능을 나타냈습니다. 이는 개인화와 일반화 사이의 균형을 정교하게 조정하는 게이팅 네트워크의 도입 덕분입니다.



### Retail-GPT: leveraging Retrieval Augmented Generation (RAG) for building E-commerce Chat Assistants (https://arxiv.org/abs/2408.08925)
Comments:
          5 pages, 4 figures

- **What's New**: 이번 연구는 소매 전자상거래에서 사용자 참여를 증진하기 위해 제품 추천을 안내하고 장바구니 작업을 지원하는 오픈 소스 RAG(Retrieval-Augmented Generation) 기반 챗봇인 Retail-GPT를 소개합니다.

- **Technical Details**: Retail-GPT는 다양한 전자상거래 도메인에 적응할 수 있는 크로스 플랫폼 시스템으로, 특정 채팅 애플리케이션이나 상업 활동에 의존하지 않습니다. 이 시스템은 인간과 유사한 대화를 통해 사용자 요구를 해석하고, 제품 가용성을 확인하며, 장바구니 작업을 관리합니다.

- **Performance Highlights**: Retail-GPT는 가상 판매 대리인으로 기능하며, 다양한 소매 비즈니스에서 이러한 어시스턴트의 실행 가능성을 테스트하는 것을 목표로 합니다.



### MLoRA: Multi-Domain Low-Rank Adaptive Network for CTR Prediction (https://arxiv.org/abs/2408.08913)
Comments:
          11 pages. Accepted by RecSys'2024, full paper

- **What's New**: 이번 연구에서는 CTR 예측(Click-Through Rate Prediction)의 다중 도메인 환경에 대한 새로운 접근법인 MLoRA를 제안합니다. 기존 모델들이 직면했던 데이터 희소성(data sparsity) 및 도메인 간 분산(disparate data distribution) 문제를 해결합니다.

- **Technical Details**: MLoRA는 각 도메인에 특화된 LoRA 모듈을 도입하여 멀티 도메인 CTR 예측 성능을 향상시키고, 여러 딥러닝 모델에 적용될 수 있는 일반적인 프레임워크입니다. 각 도메인에 대한 LoRA 어댑터를 구축하여 보다 효율적으로 데이터 분포를 학습합니다.

- **Performance Highlights**: 다양한 공개 데이터 세트에서 MLoRA의 성능이 기존 최첨단 모델들보다 유의미하게 개선된 것으로 나타났습니다. 실제 환경에서 A/B 테스트를 통해 CTR이 1.49% 증가하고, 주문 전환율(order conversion rate)이 3.37% 증가했습니다.



### What should I wear to a party in a Greek taverna? Evaluation for Conversational Agents in the Fashion Domain (https://arxiv.org/abs/2408.08907)
Comments:
          Accepted at KDD workshop on Evaluation and Trustworthiness of Generative AI Models

- **What's New**: 이번 연구는 대형 언어 모델(LLM)이 온라인 패션 소매 분야에서 고객 경험 및 패션 검색을 개선하는 데 미치는 실질적인 영향을 탐구합니다. LLM을 활용한 대화형 에이전트는 고객과 직접 상호작용함으로써 그들의 니즈를 표현하고 구체화할 수 있는 새로운 방법을 제공합니다. 이는 고객이 자신의 패션 취향과 의도에 맞는 조언을 받을 수 있도록 합니다.

- **Technical Details**: 우리는 4,000개의 다국어 대화 데이터셋을 구축하여 고객과 패션 어시스턴트 간의 상호작용을 평가합니다. 이 데이터는 LLM 기반의 고객 에이전트가 어시스턴트와 대화하는 형태로 생성되며, 대화의 주제나 항목을 통해 서로의 요구를 파악하고 적절한 상품을 제안하는 방식으로 진행됩니다. 데이터셋은 영어, 독일어, 프랑스어 및 그리스어를 포함하며, 다양한 패션 속성(색상, 유형, 소재 등)을 기준으로 구축되었습니다.

- **Performance Highlights**: 이 연구는 LLM이 고객 요구에 맞는 패션 아이템을 추천하는 데 얼마나 효과적인지를 평가하기 위해 여러 공개 및 비공개 모델(GPT, Llama2, Mistral 등)을 벤치마킹하였습니다. 이로써 패션 분야의 대화형 에이전트가 고객과 백엔드 검색 엔진 간의 강력한 인터페이스 역할을 할 수 있음을 입증하였습니다.



### Bundle Recommendation with Item-level Causation-enhanced Multi-view Learning (https://arxiv.org/abs/2408.08906)
- **What's New**: BunCa는 비대칭(item-level causation-enhanced) 관계를 고려한 새로운 번들 추천 방법론을 제안합니다. 기존의 추천 시스템이 개별 아이템 추천에 중점을 두었다면, BunCa는 연결된 아이템 세트를 추천하여 사용자와 비즈니스의 편의성을 향상시키고자 합니다.

- **Technical Details**: BunCa는 두 가지 뷰, 즉 Coherent View와 Cohesive View를 통해 사용자와 번들에 대한 포괄적인 표현을 제공합니다. Coherent View는 Multi-Prospect Causation Network를 활용하여 아이템 간의 인과 관계를 반영하며, Cohesive View는 LightGCN을 사용하여 사용자와 번들 간의 정보 전파를 수행합니다. 또한, 구체적이고 이산적인 대비 학습(concrete and discrete contrastive learning)을 통해 다중 시점 표현의 일관성과 자기 구별(self-discrimination)을 최적화합니다.

- **Performance Highlights**: BunCa는 3개의 벤치마크 데이터셋에서 광범위한 실험을 수행하여 기존 최첨단 방법들과 비교할 때 뛰어난 성능을 나타냈습니다. 이는 번들 추천 작업에서의 유효성을 입증합니다.



### Bayesian inference to improve quality of Retrieval Augmented Generation (https://arxiv.org/abs/2408.08901)
- **What's New**: 이 논문에서는 Retrieval Augmented Generation (RAG) 시스템의 텍스트 청크 품질을 검증하기 위해 베이지안 접근법을 제안합니다.

- **Technical Details**: RAG는 사용자 쿼리에 따라 관련 있는 문단을 대규모 코퍼스에서 찾은 후, LLM에 프롬프트로 제공하는 방식입니다. 이 연구에서는 Bayes theorem을 이용해 텍스트 청크의 품질을 확인하고, 문서에서의 페이지 번호를 기반으로 사전 확률을 설정합니다.

- **Performance Highlights**: 우리는 이러한 접근 방식을 통해 RAG 시스템에서 제공하는 전체 응답의 품질을 향상시킬 수 있다고 주장합니다.



### Towards Effective Authorship Attribution: Integrating Class-Incremental Learning (https://arxiv.org/abs/2408.08900)
Comments:
          Submitted to IEEE CogMI 2024 Conference

- **What's New**: 이 논문에서는 저자 식별(AA, Authorship Attribution) 문제를 단계적으로 새로운 저자를 소개받는 제안된 Class-Incremental Learning (CIL) Paradigm으로 재정의하고, 기존 AA 시스템의 한계를 극복하기 위한 접근법을 제시합니다. 특히, 저자 고유성 및 새로운 저자 수용의 중요성을 강조합니다.

- **Technical Details**: 논문에서는 저자 식별을 CIL으로 재정의하며, 다양한 저자 맥락에서 요건 및 정책을 검토합니다. 기존 CIL 접근 방식의 범위를 다루고 강점 및 약점을 평가함으로써, AA에 통합하기 위한 효과적인 방법들을 제안합니다. 또한, 여러 유명한 CIL 모델을 구현하고 널리 사용되는 AA 데이터셋에서 평가 결과를 제공합니다.

- **Performance Highlights**: 제안된 접근법을 통해 AA 시스템을 폐쇄형 모델에서 지속 학습이 가능한 CIL 패러다임으로 발전시킬 수 있습니다. 연구 커뮤니티를 위한 GitHub 리포지토리를 공개하여 연구자들이 접근할 수 있도록 하였습니다.



### LLMJudge: LLMs for Relevance Judgments (https://arxiv.org/abs/2408.08896)
Comments:
          LLMJudge Challenge Overview, 3 pages

- **What's New**: LLMJudge 챌린지는 SIGIR 2024에서 열리는 LLM4Eval 워크숍의 일환으로, 정보 검색 (IR) 시스템의 평가를 위한 새로운 접근 방식을 탐색합니다. LLMs를 활용하여 relevancy 판단을 생성하는 방법을 실험하고 있습니다.

- **Technical Details**: 참가자들은 주어진 쿼리와 문서에 대해 relevance를 평가하는 0~3 스케일의 점수를 생성해야 합니다. 데이터셋은 TREC 2023 Deep Learning 트랙의 정보를 바탕으로 하여 개발 및 테스트 세트로 나뉘며, 다양한 LLM을 통해 생성된 라벨의 품질을 평가하게 됩니다.

- **Performance Highlights**: 39개의 제출물과 7777개의 그룹으로부터 수집된 데이터를 통해 LLMJudge 테스트 세트에서 labelers의 성능을 평가했습니다. 결과적으로 labelers는 시스템 순위에 대해 일관된 의견을 보였으나, 시각적으로 확인한 Cohen’s κ와 Kendall’s τ의 변동성을 통해 라벨의 일관성에는 차이가 있음을 나타냈습니다.



### Enhancing Exploratory Learning through Exploratory Search with the Emergence of Large Language Models (https://arxiv.org/abs/2408.08894)
Comments:
          11 pages, 7 figures

- **What's New**: 이 논문에서는 정보 시대에서 학습자들이 정보 검색 및 활용 방식을 이해하는 데 필요한 새로운 이론적 모델을 제안합니다. 특히, 대규모 언어 모델(LLMs)의 영향을 고려하여 탐색적 학습(exploratory learning) 이론과 탐색적 검색 전략을 결합하였습니다.

- **Technical Details**: 이 연구는 Kolb의 학습 모델을 개선하여 높은 빈도로 탐색할 수 있는 전략과 피드백 루프(feedback loops)를 통합합니다. 이러한 접근법은 학생들의 깊은 인지(deep cognitive) 및 고차원 인지(higher-order cognitive) 기술 발달을 촉진하는 것을 목표로 합니다.

- **Performance Highlights**: 이 논문은 LLMs가 정보 검색(information retrieval) 및 정보 이론(information theory)에 통합되어 학생들이 효율적으로 탐색적 검색을 수행할 수 있도록 지원한다고 논의합니다. 이론적으로는 학생-컴퓨터 상호작용을 촉진하고 새로운 시대의 학습 여정을 지원하는 데 기여하고자 합니다.



### Molecular Graph Representation Learning Integrating Large Language Models with Domain-specific Small Models (https://arxiv.org/abs/2408.10124)
- **What's New**: 이번 연구에서는 Large Language Models (LLMs)와 Domain-specific Small Models (DSMs)의 장점을 통합한 새로운 분자 그래프 표현 학습 프레임워크인 MolGraph-LarDo를 제안합니다. 이는 분자 특성 예측 작업에서 기존 모델들이 가진 한계를 극복하고 전문 지식의 획득 비용을 줄이는 데 기여합니다.

- **Technical Details**: MolGraph-LarDo는 두 단계의 프롬프트 전략을 통해 DSM이 LLM이 제공하는 지식을 보정하여 도메인 특화 데이터를 더욱 정확하게 생성하도록 설계되었습니다. 이 프레임워크에서는 생물학적 도메인 지식을 효과적으로 활용하여 대규모의 라벨이 없는 데이터를 활용하는 방식으로 학습합니다.

- **Performance Highlights**: 실험 결과, MolGraph-LarDo는 분자 특성 예측 작업의 성능을 개선하고 생물학적 도메인 지식을 확보하는 데 드는 비용을 줄이는 데 효과적임이 입증되었습니다.



### MAPLE: Enhancing Review Generation with Multi-Aspect Prompt LEarning in Explainable Recommendation (https://arxiv.org/abs/2408.09865)
Comments:
          8 main pages, 10 pages for appendix. Under review

- **What's New**: 이번 논문에서는 사용자와 아이템 쌍으로부터 추천 이유를 제시하는 Explainable Recommendation 과제를 다루고 있습니다. MAPLE(Multi-Aspect Prompt LEarner)라는 개인화된 모델이 제안되었으며, 이는 세분화된 측면 용어를 기억하기 위해 측면 범주를 또 다른 입력 차원으로 통합합니다.

- **Technical Details**: MAPLE은 리뷰 생성과 관련된 기존 모델들이 일반성과 환각 문제로 고생하는 것을 해결하기 위해 Multi-aspect 개념을 도입했습니다. 실험을 통해 MAPLE은 음식점 도메인에서 문장과 특징의 다양성을 높이면서도 일관성과 사실적 관련성을 유지한다는 것을 보여주었습니다. MAPLE은 리트리버-리더 프레임워크에서 리트리버 구성 요소로 사용되며, 대형 언어 모델(LLM)을 리더로 배치하여 개인화된 설명을 생성합니다.

- **Performance Highlights**: MAPLE은 기존의 리뷰 생성 모델에 비해 텍스트와 기능의 다양성에서 우수한 성과를 보였으며, MAPLE이 생성한 설명은 리트리버-리더 프레임워크 내에서 좋은 쿼리로 기능합니다. 또한, MAPLE은 생성된 설명의 사실적 관련성과 설득력 모두를 개선할 수 있는 잠재력을 보여줍니다.



### On the Necessity of World Knowledge for Mitigating Missing Labels in Extreme Classification (https://arxiv.org/abs/2408.09585)
Comments:
          Preprint, 23 pages

- **What's New**: 이 논문에서는 Extreme Classification (XC)에서의 누락된 라벨 문제를 해결하기 위해 SKIM(Scalable Knowledge Infusion for Missing Labels)이라는 새로운 알고리즘을 제안합니다. SKIM은 작은 언어 모델(Small Language Models)과 풍부한 비구조화 메타데이터를 활용하여 미흡한 지식을 보완하는 방법입니다.

- **Technical Details**: SKIM 알고리즘은 두 가지 주요 단계로 구성됩니다: (i) 다양성 있는 쿼리 생성, (ii) 검색 기반 매핑. 이 과정에서 SKIM은 비구조화된 메타데이터로부터 유용한 지식을 추출하여 기존 학습 데이터 세트에 보강합니다.

- **Performance Highlights**: SKIM은 대규모 공공 데이터셋에서 기존 방법들보다 Recall@100에서 10점 이상 높은 성능을 보였으며, 1000만 개 문서를 포함하는 독점 쿼리-광고 검색 데이터셋에서도 12% 향상된 결과를 보였습니다. 또한, 인기 검색 엔진에서 실시한 온라인 A/B 테스트에서 광고 클릭 수익율이 1.23% 증가했습니다.



### WPN: An Unlearning Method Based on N-pair Contrastive Learning in Language Models (https://arxiv.org/abs/2408.09459)
Comments:
          ECAI 2024

- **What's New**: 이번 연구에서는 유해한 출력을 줄이기 위한 새로운 방법인 Weighted Positional N-pair (WPN) Learning을 제안합니다. 이 방법은 모델의 출력을 개선하는 동시에 성능 저하를 최소화하는 데 목표를 두고 있습니다.

- **Technical Details**: WPN은 위치 가중 평균 풀링(position-weighted mean pooling) 기법을 도입하여 n-pair contrastive learning(대조 학습) 프레임워크에서 작동합니다. 이 기법은 유해한 출력(예: 유독한 반응)을 제거하여 모델의 출력을 '유해한 프롬프트-유해하지 않은 응답'으로 바꾸는 것을 목표로 합니다.

- **Performance Highlights**: OPT와 GPT-NEO 모델에 대한 실험에서 WPN은 유해한 반응 비율을 최대 95.8%까지 줄이는 데 성공했으며, 아홉 개의 일반 벤치마크에서 평균적으로 2% 미만의 성능 저하로 안정적인 성능을 유지했습니다. 또한, WPN의 일반화 능력과 견고성을 입증하는 실험 결과도 제공됩니다.



### ELASTIC: Efficient Linear Attention for Sequential Interest Compression (https://arxiv.org/abs/2408.09380)
Comments:
          Submitted to AAAI 2025

- **What's New**: ELASTIC은 고전적인 self-attention의 복잡도를 낮추어 긴 사용자 행동 시퀀스를 효율적으로 모델링할 수 있는 새로운 기법입니다. 이 방법은 선형 시간 복잡도를 요구하여 모델 용량을 계산 비용과 분리시킵니다.

- **Technical Details**: ELASTIC은 고정 길이의 interest experts와 선형 dispatcher attention 메커니즘을 도입하여 긴 행동 시퀀스를 압축된 표현으로 변환합니다. 새롭게 제안된 interest memory retrieval 기술은 대규모 사용자 맞춤형 관심을 모델링하는데 있어 일관된 계산 비용을 유지하면서 높은 정확도를 제공합니다.

- **Performance Highlights**: ELASTIC은 공개 데이터셋에서 다양한 기존 추천 시스템보다 탁월한 성과를 보이며 GPU 메모리 사용량을 최대 90%까지 줄이고, 추론 속도를 2.7배 향상시킵니다. 실험 결과 ELASTIC은 기존 방법에 비해 추천 정확성과 효율성을 모두 달성했습니다.



### CodeTaxo: Enhancing Taxonomy Expansion with Limited Examples via Code Language Prompts (https://arxiv.org/abs/2408.09070)
- **What's New**: 새로운 연구인 CodeTaxo는 코드 언어를 프롬프트로 활용하여 분류체계(taxonomy) 확장의 효율성을 높여주는 혁신적인 방법입니다.

- **Technical Details**: CodeTaxo는 LLMs(Generative Large Language Models)의 특성을 이용하여 분류체계를 코드 완료(code completion) 문제로 재정의합니다. 이 방법에서는 각 엔티티를 나타내는 Entity 클래스를 정의하고, 인간 전문가나 크라우드소싱으로 작성된 기존 분류체계를 확장합니다. SimCSE를 활용하여 유사한 엔티티만을 선택적으로 프롬프트에 포함시킵니다.

- **Performance Highlights**: 다양한 실험 결과, CodeTaxo는 기존의 자가 지도(self-supervised) 방법에 비해 10.26%, 8.89%, 9.21%의 정확도 개선을 보여주었습니다. 이는 WordNet, Graphine 및 SemEval-2016에서 실험을 통해 검증되었습니다.



### ASGM-KG: Unveiling Alluvial Gold Mining Through Knowledge Graphs (https://arxiv.org/abs/2408.08972)
- **What's New**: ASGM-KG(Artisanal and Small-Scale Gold Mining Knowledge Graph)는 아마존 분지의 환경 영향을 이해하는 데 도움이 되는 지식 그래프입니다. ASGM_KG는 비정부 및 정부 기구에서 발행된 문서와 보고서에서 추출된 1,899개의 triple로 구성되어 있습니다.

- **Technical Details**: ASGM-KG는 대형 언어 모델(LLM)을 사용하여 RDF(Resource Description Framework) 형식으로 1,899개의 triple을 생성했습니다. 생성된 triple은 전문가의 검토 및 자동화된 사실 검증 프레임워크(Data Assessment Semantics, DAS)를 통해 검증되었습니다.

- **Performance Highlights**: ASGM-KG는 90% 이상의 정확도를 달성하였으며, 다양한 환경 위기에 대한 지식 집합 및 표현의 발전을 나타냅니다.



### Graph Retrieval-Augmented Generation: A Survey (https://arxiv.org/abs/2408.08921)
Comments:
          Ongoing work

- **What's New**: 이 논문은 Retrieval-Augmented Generation (RAG) 분야의 최신 기술 중 하나인 GraphRAG에 대한 체계적인 리뷰를 제공합니다. GraphRAG는 관계형 지식을 활용하여 더 정확하고 포괄적인 정보 검색을 가능하게 하여, 기존 RAG의 한계를 극복하고자 합니다.

- **Technical Details**: GraphRAG 워크플로우는 Graph-Based Indexing (G-Indexing), Graph-Guided Retrieval (G-Retrieval), Graph-Enhanced Generation (G-Generation)의 세 단계로 구성됩니다. 각 단계에서는 핵심 기술 및 학습 방법을 정리하고 있습니다. GraphRAG는 그래프 데이터베이스에서 적절한 쿼리와 관련된 그래프 요소를 검색하여 응답을 생성할 수 있습니다.

- **Performance Highlights**: GraphRAG는 텍스트 기반 RAG보다 관계형 정보를 더 정확하고 포괄적으로 검색할 수 있으며, 대규모 텍스트 입력의 긴 문맥 문제를 해결합니다. 또한, 다양한 산업 분야에서의 활용 가능성이 커지고 있으며, 연구의 초기 단계이지만, GraphRAG의 잠재력을 통해 새로운 연구 방향을 제시하고 있습니다.



### PATopics: An automatic framework to extract useful information from pharmaceutical patents documents (https://arxiv.org/abs/2408.08905)
Comments:
          17 pages, 5 figures, 5 tables

- **What's New**: PATopics는 제약 특허의 중요 정보를 효과적으로 추출하고 요약할 수 있는 새로운 프레임워크로, 수작업 검색의 필요성을 줄이기 위해 특별히 설계되었습니다.

- **Technical Details**: PATopics는 특허에서 텍스트 정보를 추출하고, 관련 주제를 생성하여 균형 잡힌 요약을 제공하는 네 가지 구성 요소로 이루어져 있습니다. 주요 단계로는 데이터 표현, 주제 모델링 분해, 엔티티 간 상관 관계 분석 및 요약 인터페이스가 포함됩니다. 이를 통해 Non-negative Matrix Factorization (NMF) 기법을 활용하여 문서 간의 관계를 분석합니다.

- **Performance Highlights**: 이 프레임워크는 4,832개의 제약 특허를 분석하는 데 사용되었으며, 연구자, 화학자 및 기업의 세 가지 사용자 프로필에서 실용성을 입증했습니다. PATopics는 제약 분야에서 특허의 검색과 요약을 용이하게 하며, 실질적이고 유용한 정보 제공에 중점을 두고 있습니다.



### SC-Rec: Enhancing Generative Retrieval with Self-Consistent Reranking for Sequential Recommendation (https://arxiv.org/abs/2408.08686)
- **What's New**: 최근 언어 모델(Language Models, LMs)의 발전은 추천 시스템(Recommendation Systems)에서의 사용을 증가시키고 있습니다. 본 논문에서는 정보의 통합과 협력적 지식의 상호 보완성을 반영하는 SC-Rec이라는 새로운 추천 시스템을 제안합니다.

- **Technical Details**: SC-Rec은 다양한 항목 인덱스(item indices)와 여러 프롬프트 템플릿(prompt templates)으로부터 수집한 지식을 학습합니다. 이 시스템은 세 가지 단계로 구성되며, 첫째로 다중 인덱스를 생성하고 둘째로 추천모델을 훈련하며 셋째로 정확도(self-consistency) 기반의 재정렬(reranking)을 수행합니다. 이를 통해 다각적인 정보와 구조적 차이를 활용합니다.

- **Performance Highlights**: SC-Rec은 세 가지 실제 데이터셋에서 수행된 실험을 통해 기존의 최첨단 방법들보다 상당히 우수한 성능을 나타냈으며, 다양한 인덱스와 프롬프트 템플릿을 통해 서로 보완적인 고급 정보를 효과적으로 통합함으로써 높은 품질의 재정렬된 리스트를 생성합니다.



New uploads on arXiv(cs.CV)

### SANER: Annotation-free Societal Attribute Neutralizer for Debiasing CLIP (https://arxiv.org/abs/2408.10202)
- **What's New**: 이 논문에서는 CLIP와 같은 대규모 비전-언어 모델에서 발생하는 사회적 편향을 해결하기 위한 새로운 방법, SANER(사회적 속성 중화기)를 소개합니다. 기존 연구에서는 편향 제거를 위해 적대적 학습이나 테스트 시간에서 프로젝팅하는 방법을 제안했으나, 이들이 가진 두 가지 주요 한계를 지적합니다.

- **Technical Details**: SANER는 속성 중립적인 설명에서 CLIP 텍스트 특징을 수정하여 속성 특정 설명과 동등한 거리로 만듭니다. 이를 위해 속성 주석 없이 작동하는 수정 레이어(다층 인식기)를 학습하여, 특정 속성 설명은 원래 정보를 그대로 유지하면서 속성 중립 설명에 대해서만 편향을 제거합니다.

- **Performance Highlights**: SANER는 기존 방법들보다 더 효과적으로 성별, 나이 및 인종의 편향을 완화할 수 있는 능력을 보여줍니다. 실험 결과 SANER은 속성 의존성을 줄이며, 기존 방법들에 비해 더 나은 성능을 입증하였습니다.



### MeshFormer: High-Quality Mesh Generation with 3D-Guided Reconstruction Mod (https://arxiv.org/abs/2408.10198)
Comments:
          20 pages, 9 figures

- **What's New**: 이 논문에서는 MeshFormer를 소개하며, 이는 고품질의 3D 텍스처 메쉬를 생성하는 오픈 월드 스파스 뷰 재구성 모델입니다. MeshFormer는 비좁은 포즈가 있는 이미지 세트를 입력받아 단일 전방 패스로 고품질 3D 메쉬를 제공할 수 있습니다.

- **Technical Details**: MeshFormer는 기존의 트리플레인 표현 대신 3D 스파스 복셀에 특징을 저장하며, 대규모 트랜스포머와 3D(스파스) 컨볼루션을 결합하여 명시적인 3D 구조 및 투사 편향을 활용합니다. 네트워크는 RGB 입력 외에도 적절한 법선 맵을 생성할 수 있도록 학습되며, 고해상도 SDF(Signed Distance Function) 감독을 통해 메쉬를 직접 생성하는 것을 목표로 합니다.

- **Performance Highlights**: MeshFormer는 단 8개의 GPU로 훈련될 수 있으며, 100개 이상의 GPU가 필요한 기존 방법들보다 더 효율적이고 빠른 훈련 시간을 기록했습니다. 결과적으로, 고품질의 텍스처 메쉬를 만족스러운 세부 사항과 함께 신속하게 생성할 수 있습니다.



### SpaRP: Fast 3D Object Reconstruction and Pose Estimation from Sparse Views (https://arxiv.org/abs/2408.10195)
Comments:
          ECCV 2024

- **What's New**: 이 논문은 제한적인 수의 non-posed 2D 이미지로부터 3D 객체를 복원하는 새로운 메소드인 SpaRP를 제안합니다. 이 방법은 기존의 dense view 방식과 달리 sparse view 입력을 처리하며, 기존 방법들이 기대에 미치지 못하는 컨트롤을 개선하였습니다.

- **Technical Details**: SpaRP는 2D diffusion models에서 지식을 추출하고, 이를 활용하여 sparse view 이미지 간의 3D 공간 관계를 유추합니다. 이 메소드는 카메라 포즈를 추정하고 3D textured mesh를 생성하는 데 필요한 정보를 종합적으로 활용하여, 단지 약 20초 만에 결과를 도출합니다.

- **Performance Highlights**: 세 가지 데이터셋에 대한 실험을 통해 SpaRP는 3D 복원 품질과 포즈 예측 정확도에서 기존의 방법들과 비교하여 상당한 성능 향상을 보여주었습니다.



### LongVILA: Scaling Long-Context Visual Language Models for Long Videos (https://arxiv.org/abs/2408.10188)
Comments:
          Code and models are available at this https URL

- **What's New**: LongVILA는 멀티모달 비전-언어 모델을 위한 새로운 솔루션으로, 시스템, 모델 학습 및 데이터 집합 개발을 포함한 풀 스택(full-stack) 접근 방식을 제안합니다. 특히, Multi-Modal Sequence Parallelism (MM-SP) 시스템을 통해 256개의 GPU에서 2M 컨텍스트 길이 학습 및 추론이 가능해졌습니다.

- **Technical Details**: LongVILA는 5단계 교육 파이프라인, 즉 정렬, 사전 학습, 컨텍스트 확장 및 장단기 공동 감독의 세부 조정을 포함합니다. MM-SP를 통해 메모리 강도가 높은 장기 컨텍스트 모델을 효과적으로 교육할 수 있으며, Hugging Face Transformers와의 원활한 통합이 가능합니다.

- **Performance Highlights**: LongVILA는 VILA의 프레임 수를 128배(8에서 1024 프레임까지)로 확장하고, 긴 비디오 캡셔닝 점수를 2.00에서 3.26(1.6배)로 개선했습니다. 1400프레임 비디오(274k 컨텍스트 길이)에서 99.5%의 정확도를 달성하며, VideoMME 벤치마크에서도 성능 일관성을 보여주고 있습니다.



### Assessment of Spectral based Solutions for the Detection of Floating Marine Debris (https://arxiv.org/abs/2408.10187)
Comments:
          5 pages, 3 figures, submitted and accepted for 2024 Second International Conference on Networks, Multimedia and Information Technology (NMITCON)

- **What's New**: 이 연구는 해양 쓰레기 탐지를 위해 새로운 표준 데이터셋인 Marine Debris Archive (MARIDA)를 활용하는 방법을 제안합니다. 기존의 방법들은 주로 사람의 노력이 많이 필요하고 공간적 범위가 제한적이었으나, MARIDA는 다양한 탐지 솔루션을 비교할 수 있게 도와줍니다.

- **Technical Details**: MARIDA 데이터셋은 해양 플라스틱 쓰레기를 탐지하기 위한 Machine Learning (ML) 알고리즘 개발 및 평가의 기준으로 활용됩니다. 본 연구에서는 MARIDA 데이터셋을 기반으로 한 스펙트럼(spectral) 기반 솔루션들의 성능을 평가하였습니다.

- **Performance Highlights**: 결과는 공정한 성능 평가를 위해 정확한 기준(reference)이 필요하다는 점을 강조합니다. 이를 통해 연구자들이 해양 환경 보전을 위한 연구를 더욱 촉진할 수 있을 것으로 기대됩니다.



### Imbalance-Aware Culvert-Sewer Defect Segmentation Using an Enhanced Feature Pyramid Network (https://arxiv.org/abs/2408.10181)
- **What's New**: 이 논문은 불균형 데이터셋에서도 효과적으로 작동하는 Enhanced Feature Pyramid Network (E-FPN)라는 새로운 심층 학습 모델을 소개합니다. E-FPN은 오브젝트 변형을 잘 처리하고, 기능 추출을 개선하기 위한 건축 혁신을 포함합니다.

- **Technical Details**: E-FPN은 희소 연결 블록과 깊이 별개의 합성곱(depth-wise separable convolutions)을 사용하여 정보 흐름을 효율적으로 관리하고, 파라미터를 줄이면서도 표현력은 유지합니다. 또한, class decomposition(클래스 분해)와 data augmentation(데이터 증대) 전략을 통해 데이터 불균형 문제를 해결합니다.

- **Performance Highlights**: E-FPN은 수로와 하수관 결함 데이터셋 및 항공기 초상형 분할 드론 데이터셋에서 각각 평균 Intersection over Union (IoU) 개선치를 13.8% 및 27.2% 달성했으며, 클래스 분해와 데이터 증대 기법이 결합될 경우 약 6.9% IoU 성능 향상을 보였습니다.



### NeuRodin: A Two-stage Framework for High-Fidelity Neural Surface Reconstruction (https://arxiv.org/abs/2408.10178)
- **What's New**: NeuRodin은 고품질의 3D 표면 재구성을 위한 새로운 2단계 신경망 프레임워크로, SDF 기반 방법의 한계를 극복합니다. 이 방법은 일반적인 밀도 기반 방법의 유연한 최적화 특성을 유지하며, 세밀한 기하학 구조를 효과적으로 캡쳐합니다.

- **Technical Details**: NeuRodin은 두 가지 주요 문제를 해결합니다. 첫째, SDF-밀도 변환을 글로벌 스케일 매개변수에서 로컬 적응형 매개변수로 전환하여 유연성을 높입니다. 둘째, 새로운 손실 함수가 최대 확률 거리와 제로 레벨 세트를 정렬하여 기하학적 표현의 정렬성을 향상시킵니다. 이 과정은 coarse 및 refinement 과정의 두 단계로 나뉘어 진행됩니다.

- **Performance Highlights**: NeuRodin은 Tanks and Temples와 ScanNet++ 데이터 세트에서 이전 모든 방법들에 비해 우수한 성능을 발휘했으며, 특히 복잡한 토폴로지 구조 최적화와 세밀한 정보 보존에서 두각을 나타냈습니다. 제안된 방법은 VR 및 AR 시스템에서 활용될 수 있는 잠재력을 가지고 있습니다.



### Fairness Under Cover: Evaluating the Impact of Occlusions on Demographic Bias in Facial Recognition (https://arxiv.org/abs/2408.10175)
Comments:
          Accepted at ECCV Workshop FAILED

- **What's New**: 본 연구는 얼굴 인식 시스템의 공정성에 대한 차이가 인종 집단별로 어떻게 다르게 나타나는지 분석합니다. 특히, 합성된 현실적인 오클루전(occlusion)이 얼굴 인식 모델의 성능에 미치는 영향을 평가하고, 특히 아프리카계 인물에게 더 큰 영향을 미친다는 점을 강조합니다.

- **Technical Details**: Racial Faces in the Wild (RFW) 데이터셋을 사용하여 BUPT-Balanced 및 BUPT-GlobalFace 데이터셋으로 훈련된 얼굴 인식 모델의 성능을 평가합니다. FMR( False Match Rate), FNMR(False Non-Match Rate), 정확도와 같은 성능 지표의 분산이 증가하고, Equilized Odds, Demographic Parity, STD of Accuracy, Fairness Discrepancy Rate와 같은 공정성 지표는 감소한다는 점을 확인했습니다. 새로운 메트릭인 Face Occlusion Impact Ratio (FOIR)를 제안하여 오클루전이 얼굴 인식 모델의 성능에 미치는 영향을 정량화합니다.

- **Performance Highlights**: 오클루전이 존재할 때 아프리카계 인물에 대한 오클루전의 중요성이 다른 민족 그룹과 비교하여 더 높게 나타났습니다. 연구 결과는 일반적으로 얼굴 인식 시스템에 존재하는 인종적 편견을 증폭시키며, 공정성 평가에 대한 새로운 통찰력을 제공합니다.



### NeuFlow v2: High-Efficiency Optical Flow Estimation on Edge Devices (https://arxiv.org/abs/2408.10161)
- **What's New**: 본 논문에서는 NeuFlow v1을 기반으로 한 새로운 고효율의 optical flow 추정 방법을 제안합니다. 기존 방법들과 비교했을 때 10배에서 70배 이상의 처리 속도 향상을 이루면서도 비슷한 성능을 유지합니다.

- **Technical Details**: 제안된 architecture는 간소화된 CNN 기반의 backbone과 효율적인 iterative refinement 모듈로 구성됩니다. 첫 번째 모듈은 cross-attention과 global matching을 포함하여 대형 변위를 처리할 수 있는 초기 optical flow를 추정하는데 도움을 줍니다. 두 번째 모듈은 간단한 RNN 구조를 사용하여 flow를 반복적으로 개선합니다.

- **Performance Highlights**: 이 시스템은 Jetson Orin Nano에서 512x384 해상도의 이미지에 대해 20 FPS를 초과하는 속도로 실행될 수 있습니다. 또한, 실제 데이터에서 뛰어난 일반화 성능을 보여줍니다.



### LoopSplat: Loop Closure by Registering 3D Gaussian Splats (https://arxiv.org/abs/2408.10154)
Comments:
          Project page: \href{this https URL}{this http URL}

- **What's New**: LoopSplat은 RGB-D 이미지를 입력으로 받으며, 3D Gaussian Splatting을 통해 밀집 맵핑(dense mapping)을 수행하는 새로운 SLAM 시스템을 제안합니다. 이 시스템은 루프 클로저(loop closure)를 온라인으로 트리거하고, 서브맵(submap) 간의 상대 루프 엣지 제약조건을 직접 계산하여 전통적인 점 클라우드(Point Cloud) 등록법보다 효율성과 정확성을 개선합니다.

- **Technical Details**: LoopSplat은 3D Gaussian Splats(3DGS)를 사용하는 결합형 SLAM 시스템으로서, RGB-D 카메라를 통해 서브맵을 생성하고 프레임-모델 추적(frame-to-model tracking)을 수행합니다. 이 방법은 루프 클로저를 위한 새로운 등록(registration) 방법을 제안해 3DGS 표현을 직접 활용하여 서브맵을 정렬하고 글로벌 일관성을 유지합니다.

- **Performance Highlights**: Synthetic Replica 데이터셋 및 실제 데이터셋인 TUM-RGBD, ScanNet, ScanNet++에서 평가된 결과, LoopSplat은 기존의 밀집 RGB-D SLAM 방법들과 비교하여 뛰어난 추적(tracking), 맵핑(mapping), 렌더링(rendering) 성능을 보였습니다. 특히, 다양한 실제 세계 데이터셋에 걸쳐 성능 개선과 강인성이 향상됨을 확인했습니다.



### Structure-preserving Image Translation for Depth Estimation in Colonoscopy Video (https://arxiv.org/abs/2408.10153)
Comments:
          12 pages, 7 figures, accepted at MICCAI 2024

- **What's New**: 이 논문에서는 콜로노스코피(Colonoscopy) 비디오에서 단안 깊이 추정(Monocular depth estimation)의 과제를 해결하기 위해 합성 데이터와 실제 임상 데이터 간의 도메인 갭(domain gap)을 메우기 위한 새로운 접근法을 제안합니다. 주목할 점은 복잡한 모델링 없이도 깊이 정보를 보존하는 이미지를 생성하는 구조 보존 이미지를 번역하는 모듈화된 파이프라인을 제시했다는 것입니다.

- **Technical Details**: 제안된 방법은 CycleGAN을 기반으로 하여, 합성 콜로노스코피 데이터(SimCol3D)에서 시각적으로 사실적인 비디오 프레임으로 변환하며, 이 과정에서 깊이 정보를 유지합니다. 이를 통해 다양한 데이터를 통해 실험적으로 깊이 추정 성능을 개선할 수 있었습니다. 또한, 영상 전이 과정에서의 구조 보존 손실 구조를 적용하여, 깊이 추정의 정확도를 높이는 데 중점을 두었습니다.

- **Performance Highlights**: 제안된 방법은 다양한 데이터셋에서 깊이 추정 성능을 평가했으며, 기존의 합성 데이터에 비해 더욱 사실적인 결과를 보였습니다. 더불어, 두 개의 신규 데이터셋을 통해 특수한 관점에 대한 테스트가 가능해져서, 임상 데이터에 대한 일반화 능력을 향상시키는 성과를 거두었습니다.



### Multi-Scale Representation Learning for Image Restoration with State-Space Mod (https://arxiv.org/abs/2408.10145)
- **What's New**: 이 논문에서는 효율적인 이미지 복원을 위해 다중 스케일 상태 공간 모델(Multi-Scale State-Space Model, MS-Mamba)을 제안합니다. 이는 전통적인 CNN과 Transformer의 한계를 극복하고, 다중 스케일 정보의 표현 학습을 강화하기 위한 새로운 방법론입니다.

- **Technical Details**: 논문에서 제안하는 MS-Mamba는 글로벌 및 지역 상태 공간 모델(SSM) 모듈을 통합한 계층적 뱀바 블록(Hierarchical Mamba Block, HMB)을 포함합니다. 또한, 다양한 방향의 그래디언트를 캡처하여 세부 정보 추출 능력을 향상시키는 Adaptive Gradient Block (AGB)과 주파수 도메인에서 세부 사항 학습을 촉진하는 Residual Fourier Block (RFB)을 제안합니다.

- **Performance Highlights**: 제안된 MS-Mamba는 이미지 비오는 것(image deraining), 안개 제거(dehazing), 노이즈 제거(denoising), 저조도 향상(low-light enhancement)의 네 가지 클래식 이미지 복원 작업에서 아홉 개 공개 벤치마크를 통해 기존의 최첨단(image restoration state-of-the-art) 방법보다 뛰어난 성능을 보이며, 낮은 계산 복잡도를 유지합니다.



### $R^2$-Mesh: Reinforcement Learning Powered Mesh Reconstruction via Geometry and Appearance Refinemen (https://arxiv.org/abs/2408.10135)
- **What's New**: 본 논문에서는 다각도 이미지를 기반으로 점진적으로 메쉬를 생성 및 최적화하는 새로운 알고리즘을 제안합니다. 이 알고리즘은 NeRF 모델을 사용하여 초기 Signed Distance Field (SDF)와 뷰 의존적 appearance 필드를 설정하는 것부터 시작합니다.

- **Technical Details**: 점진적으로 메쉬를 최적화하기 위해, FlexiCubes와 차별화 가능한 레스터화 기법을 사용하여 메쉬의 정점 위치와 연결성을 유연하게 업데이트합니다. 또한 Upper Confidence Bound (UCB) 알고리즘에 기반한 온라인 학습 전략을 도입하여 최적의 뷰포인트를 선택하고, NeRF 모델로 렌더링된 이미지를 훈련 데이터셋에 점진적으로 추가하여 메쉬 훈련을 향상시킵니다.

- **Performance Highlights**: 본 방법은 메쉬 렌더링 품질과 기하학적 품질 모두에서 경쟁력 있는 성능을 보여줍니다. 실험을 통해 본 알고리즘이 높은 충실도와 세부 정보가 풍부한 시각적, 기하학적 결과를 생성하는 데 효과적임을 입증합니다.



### Perceptual Depth Quality Assessment of Stereoscopic Omnidirectional Images (https://arxiv.org/abs/2408.10134)
Comments:
          Accepted by IEEE TCSVT

- **What's New**: 이 연구는 입체적 360도 전방향 이미지의 Depth Quality Index (DQI)라는 새로운 객관적 품질 평가 모델을 개발하며, 이는 깊이 품질(Depth Quality) 평가에서 참조 없는 (no-reference) 방법을 사용합니다.

- **Technical Details**: DQI는 인간 시각 체계(Human Visual System, HVS)의 지각적 특성을 기반으로 하며, 다채널 색상, 적응형 뷰포트 선택, 그리고 양안 불일치(interocular discrepancy) 특성에 의존합니다. 실험 결과, 제안된 DQI는 단일 뷰포트 및 전방향 입체 이미지 데이터베이스를 통해 이미징 품질 평가(IQA) 및 깊이 품질 평가(DQA) 방법보다 인지적 깊이 품질을 예측하는 데 더 우수함을 입증했습니다.

- **Performance Highlights**: DQI와 기존의 IQA 방법을 결합하면 3D 전방향 이미지의 전반적인 품질 예측 성능이 대폭 향상되는 것으로 나타났습니다.



### UNINEXT-Cutie: The 1st Solution for LSVOS Challenge RVOS Track (https://arxiv.org/abs/2408.10129)
- **What's New**: 이번 논문에서는 자연어 표현을 통해 비디오 객체를 분할하는 Referring Video Object Segmentation (RVOS) 작업에 대한 새로운 접근 방식을 제안합니다. 특히 6th LSVOS Challenge RVOS Track에서 MeViS 데이터셋을 도입하며, 이것은 전통적인 RVOS 작업보다 더 복잡한 장면을 제공합니다.

- **Technical Details**: 본 연구에서는 RVOS 및 VOS 모델의 장점을 통합하여 간단하면서도 효과적인 RVOS 파이프라인을 구축합니다. 먼저, RVOS 모델 UNINEXT를 미세 조정하여 언어 설명과 관련된 마스크 시퀀스를 획득하고, 이후 VOS 모델 Cutie를 사용하여 마스크 결과의 품질 및 시간 일관성을 향상시킵니다. 또한, 반지도 학습(semi-supervised learning)을 통해 RVOS 모델의 성능을 더욱 개선합니다.

- **Performance Highlights**: 우리의 접근 방식은 MeViS 테스트 세트에서 62.57 J&F를 달성하며 6th LSVOS Challenge RVOS Track에서 1위를 기록했습니다.



### Video Object Segmentation via SAM 2: The 4th Solution for LSVOS Challenge VOS Track (https://arxiv.org/abs/2408.10125)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2408.00714

- **What's New**: 새로운 Segment Anything Model 2 (SAM 2)은 비디오 내 개체 분할을 수행하는 데 있어 사용자 상호 작용을 통해 데이터를 개선하고, 대규모 비디오 분할 데이터셋을 수집하는 데이터 엔진을 구축했습니다.

- **Technical Details**: SAM 2는 간단한 트랜스포머 아키텍처로, 실시간 비디오 처리를 위한 스트리밍 메모리를 사용합니다. 여러 프레임에서의 객체를 정밀하게 분할하기 위해 포인트, 박스, 마스크 프롬프트를 지원하며, 이전 예측과 프롬프트된 프레임의 메모리를 기반으로 최적의 임베딩을 생성합니다.

- **Performance Highlights**: SAM 2는 MOSE와 LVOS 데이터셋에서 75.79%의 J&F를 달성하였으며, 6회 LSVOS 챌린지 VOS Track에서 4위에 랭크되었습니다. 이는 Fine-tuning 없이 이루어진 결과로, SAM 2의 강력한 제로샷 성능을 입증합니다.



### Factorized-Dreamer: Training A High-Quality Video Generator with Limited and Low-Quality Data (https://arxiv.org/abs/2408.10119)
- **What's New**: 본 연구에서는 고품질 비디오 생성기(HQ video generator)가 공개된 제한적이고 저품질(LQ) 데이터만으로도 훈련될 수 있음을 보여줍니다. 특히, 'Factorized-Dreamer'라는 두 단계의 비디오 생성 과정을 제안합니다.

- **Technical Details**: 이 모델은 고도로 설명적인 캡션(Highly Descriptive Caption)을 바탕으로 이미지를 생성하고, 그 후 생성된 이미지 및 간결한 동작 세부 사항의 캡션을 바탕으로 비디오를 합성하는 방식으로 작동합니다. Factorized-Dreamer는 텍스트와 이미지 임베딩을 결합하는 어댑터와 픽셀 수준의 영상 정보를 캡처하는 픽셀 인식 교차 주의 모듈(pixel-aware cross attention module)을 포함합니다.

- **Performance Highlights**: WebVid-10M과 같은 LQ 데이터셋에서 직접 훈련할 수 있으며, 기존의 많은 T2V 모델보다 뛰어난 결과를 보여줍니다.



### Modelling the Distribution of Human Motion for Sign Language Assessmen (https://arxiv.org/abs/2408.10073)
Comments:
          Accepted to Twelfth International Workshop on Assistive Computer Vision and Robotics at ECCV 2024

- **What's New**: 본 논문에서는 Sign Language Assessment(SLA) 도구의 개발을 다루고 있으며, 기존의 단일 참조 비디오를 통한 단편적인 평가를 넘어서는 새로운 접근법을 제안합니다. 이를 통해 SL의 이해 가능성을 평가할 수 있는 도구가 마련되었습니다.

- **Technical Details**: 저자들은 Skeleton Variational Autoencoder(SkeletonVAE)를 기반으로 SL의 연속적인 시퀀스를 평가할 수 있는 시스템을 구축했습니다. 이 시스템은 여러 원어민 서명자의 데이터를 활용하여 자연스러운 인간 동작의 분포를 모델링하며, Gaussian Process(GP)를 활용하여 임베디드 데이터를 학습합니다.

- **Performance Highlights**: 개발된 도구는 인간 평가자와의 비교에서 유의미한 상관관계를 보이며, 실제 서명자의 행동을 평가할 수 있는 능력을 보여주었습니다. 또한, SL 학습 과정에서 발생하는 비정상적인 결과를 시공간적으로 감지해, 효율적인 피드백을 제공할 수 있음을 입증하였습니다.



### FFAA: Multimodal Large Language Model based Explainable Open-World Face Forgery Analysis Assistan (https://arxiv.org/abs/2408.10072)
Comments:
          17 pages, 18 figures; project page: this https URL

- **What's New**: 본 논문에서는 효율적인 얼굴 위조(face forgery) 분석을 위해 새로운 OW-FFA-VQA(Open-World Face Forgery Analysis VQA) 작업과 벤치마크를 도입하였습니다. 이를 통해 사용자 친화적이면서도 이해하기 쉬운 인증 분석을 제공합니다.

- **Technical Details**: OW-FFA-VQA 작업은 기존의 이진 분류(task) 문제를 넘어, 다양한 실제 및 위조 얼굴 이미지의 설명과 신뢰할 수 있는 위조 추론을 포함하는 VQA(Visual Question Answering) 데이터셋을 기반으로 합니다. FFAA(Face Forgery Analysis Assistant)는 정밀 조정된 MLLM(Multimodal Large Language Model)과 MIDS(Multi-answer Intelligent Decision System)로 구성되며, 다양한 가설을 기반으로 유연한 반응을 제공하는 기능을 갖추고 있습니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, 우리의 방법은 이전 방법들에 비해 정확성과 견고성을 크게 향상시키면서 사용자 친화적인 설명 가능한 결과를 제공합니다. 또한 FFAA는 복잡한 환경에서도 뛰어난 일반화 능력을 발휘하여 원본과 위조된 이미지를 효과적으로 구별하는 성능을 보여주었습니다.



### LNQ 2023 challenge: Benchmark of weakly-supervised techniques for mediastinal lymph node quantification (https://arxiv.org/abs/2408.10069)
Comments:
          Submitted to MELBA

- **What's New**: 이 논문은 MICCAI 2023에서 열린 Mediastinal Lymph Node Quantification (LNQ) 챌린지를 요약합니다. 이 챌린지는 부분적으로 주석이 달린 데이터셋을 제공하여 약한 감독 학습(weakly-supervised learning) 방법을 발전시키고자 하였습니다.

- **Technical Details**: 챌린지 참가자들은 제한된 주석이 달린 CT 스캔을 사용하여 림프절을 세분화해야 했습니다. 병원 데이터셋은 전체 513개의 가슴 CT 스캔으로 구성되어 있으며, 이는 다양한 암 환자 치료 중 촬영되었습니다. 참가자들은 기존의 작고 공개된 완전 주석 데이터셋을 활용할 수 있었습니다.

- **Performance Highlights**: 약한 감독 방식의 평균 Dice 점수는 61.0%로 나타났으며, 최고 점수 팀들은 질적 완전 주석 데이터셋을 활용하여 70%를 초과하는 성능을 보였습니다. 이는 약한 감독 방법의 가능성과 고품질 주석 데이터의 필요성을 강조합니다.



### Facial Wrinkle Segmentation for Cosmetic Dermatology: Pretraining with Texture Map-Based Weak Supervision (https://arxiv.org/abs/2408.10060)
- **What's New**: 이번 논문에서는 최초의 공개 얼굴 주름 데이터셋인 'FFHQ-Wrinkle'을 소개하며, 얼굴 주름 자동 탐지를 위한 새로운 훈련 전략을 제안합니다. 이 데이터셋은 1,000장의 인간 레이블이 있는 이미지와 50,000장의 자동 생성된 약한 레이블이 포함되어 있습니다.

- **Technical Details**: U-Net과 Swin UNETR 아키텍처를 사용하여 주름 감지 모델을 훈련합니다. 두 단계의 훈련 전략을 통해 약한 레이블 데이터로 사전 훈련을 수행한 후, 인간 레이블 데이터로 미세 조정(finetuning)하여 주름 감지 성능을 향상시킵니다. U-Net 모델 구조는 네 개의 인코더 블록과 네 개의 디코더 블록으로 이루어져 있습니다.

- **Performance Highlights**: 본 방법론은 기존의 사전 훈련 방법과 비교하여 정량적 및 시각적 측면에서 얼굴 주름 분할 성능을 향상시키는 데 성공하였습니다. 이 연구는 주름 검출 알고리즘 개발을 위한 기준 데이터셋 제공으로 향후 연구에 기여할 것입니다.



### Implicit Gaussian Splatting with Efficient Multi-Level Tri-Plane Representation (https://arxiv.org/abs/2408.10041)
- **What's New**: 이 논문은 Implicit Gaussian Splatting (IGS)라는 혁신적인 하이브리드 모델을 제안합니다. 이는 명시적 포인트 클라우드와 암시적 특성 임베딩을 통합하여 고급의 메모리 효율성을 제공합니다.

- **Technical Details**: IGS는 다중 해상도의 다중 레벨 tri-plane 아키텍처를 통해 연속적인 공간 도메인 표현을 가능하게 합니다. 또한 레벨 기반의 점진적 훈련 방식과 공간적 정규화를 도입하여 렌더링 품질을 향상시킵니다.

- **Performance Highlights**: 우리의 알고리즘은 몇 MB의 용량만을 사용하여 높은 품질의 렌더링을 제공할 수 있으며, 현재 기술 수준의 경쟁력을 갖춘 결과를 보여줍니다.



### SHARP: Segmentation of Hands and Arms by Range using Pseudo-Depth for Enhanced Egocentric 3D Hand Pose Estimation and Action Recognition (https://arxiv.org/abs/2408.10037)
Comments:
          Accepted at 27th International Conference on Pattern Recognition (ICPR)

- **What's New**: 본 연구에서는 egocentric 3D 손 자세 추정을 RGB 프레임만을 사용하여 개선하는 방법을 제안합니다. 이는 최첨단 깊이 추정 기법을 활용하여 생성된 pseudo-depth 이미지를 이용하여 불필요한 장면 부분을 세분화하는 방식입니다. 이 과정에서 SHARP(Segmentation of Hands and Arms by Range using Pseudo-depth)라는 새로운 세분화 모듈을 도입하였습니다.

- **Technical Details**: 제안된 방법은 EffHandEgoNet3D 구조를 기반으로 하며, 깊이 센서를 사용하지 않고도 RGB 관점에서 3D 손 자세를 추정합니다. pseudo-depth 이미지와 세그멘테이션 기술을 결합하여 사진에서 손과 객체만을 분리하여 정확도를 높이는 방식입니다. 최종적으로, Transformer 기반의 action recognition 네트워크를 사용하여 행동 인식 정확도를 91.73%까지 향상시켰습니다.

- **Performance Highlights**: H2O 데이터셋에서 평균 포즈 오차가 28.66 mm로 감소하였으며(기존 35.48 mm에서), 이는 기존 최첨단 방법 대비 우수한 성능을 보여줍니다. 실험 결과, 손 자세 추정의 정확도가 높아지고, 이를 통해 egocentric action recognition의 가능성이 확대되었습니다.



### Dynamic Label Injection for Imbalanced Industrial Defect Segmentation (https://arxiv.org/abs/2408.10031)
Comments:
          ECCV 2024 VISION Workshop

- **What's New**: 본 연구에서는 불균형 다중 클래스 시멘틱 세그멘테이션(semantic segmentation) 문제를 해결하기 위한 간단하면서도 효과적인 방법인 Dynamic Label Injection(DLI) 알고리즘을 제안합니다. 이 방법은 학습 세트에서 클래스 간 균형을 유지하는 데 중점을 두어, 불균형한 입력 배치를 균등한 분포로 조정하고자 합니다.

- **Technical Details**: Dynamic Label Injection(DLI) 알고리즘은 입력 배치의 결함(defect) 분포를 현재 상태에서 계산하고, Poisson 기반의 seamless image cloning과 cut-paste 기술을 결합하여 결함을 다른 이미지로 전이하여 재조정합니다. 이 방법은 실시간 품질 검사와 관련된 산업 응용에 적합한 알고리즘 파이프라인입니다. 실험에서는 Magnetic Tiles 데이터셋을 사용하였고, IoU(Intersection over Union) 점수가 개선되었습니다.

- **Performance Highlights**: DLI 알고리즘은 기존의 클래스 불균형 해결 방법들에 비해 우수한 성능을 보여주었으며, 다양한 실험 반복 수행을 통해 결과의 강건성을 입증하였습니다. 또한, 불균형 다중 클래스 결함 세그멘테이션 문제를 다룬 연구 중 몇 안 되는 논문 중 하나로, 산업 데이터를 활용한 중요한 기여를 하고 있습니다.



### Towards Robust Federated Image Classification: An Empirical Study of Weight Selection Strategies in Manufacturing (https://arxiv.org/abs/2408.10024)
Comments:
          Submitted to The 2nd IEEE International Conference on Federated Learning Technologies and Applications (FLTA24)

- **What's New**: 이 연구에서는 제조 분야의 Federated Learning (FL)에서 클라이언트 가중치를 선택하는 두 가지 전략, 즉 최종 에폭 가중치 선택(FEWS)과 최적 에폭 가중치 선택(OEWS)을 비교하여 모델 성능에 미치는 영향을 분석합니다.

- **Technical Details**: 이 연구에서는 EfficientNet, ResNet 및 VGG와 같은 다양한 신경망 아키텍처를 사용하여 FL 프레임워크 내에서 FEWS와 OEWS의 상대적 효과를 평가합니다. 각 클라이언트의 데이터세트는 고도로 상관된 비독립 및 동일 분포(Non-IID) 데이터를 포함하여 독특한 구성으로 되어 있습니다.

- **Performance Highlights**: 실험 결과, OEWS 전략이 FEWS보다 글로벌 FL 모델의 성능 향상에 더 효과적임을 입증하며, 제한된 수의 클라이언트로 협력할 때 더 견고한 모델 수렴과 효율성을 촉진합니다.



### Detecting Adversarial Attacks in Semantic Segmentation via Uncertainty Estimation: A Deep Analysis (https://arxiv.org/abs/2408.10021)
- **What's New**: 이 연구에서는 인지적 세분화(semantic segmentation) 모델에 대한 적대적 공격(adversarial attack)을 탐지하기 위한 불확실성 기반 접근 방식(uncertainty-based approach)을 제안했습니다. 적대적 예제(adversarial examples)의 탐지 성능을 심층 분석하였으며, 여러 유형의 공격에 대해서도 효과적인 탐지가 가능함을 보여주었습니다.

- **Technical Details**: 불확실성(uncertainty)은 출력 분포의 엔트로피(entropy)로 측정되며, 깨끗한 이미지와 적대적으로 교란된 이미지에서 다르게 작용하는 특성을 가지고 있습니다. 이 연구에서는 신경망의 출력 정보를 활용하여 post-processing 단계에서 단순하게 적용 가능한 탐지 방식을 제안합니다. 이 방법은 기존 모델 수정을 요구하지 않으며, 다양한 최신 신경망 아키텍처에 대해 적용 가능합니다.

- **Performance Highlights**: 우리의 불확실성 기반 탐지 방법은 89.36%의 평균 탐지 정확도(accuracy)를 달성하였습니다. 이 연구는 다양한 유형의 공격에 대해 높은 탐지 성능을 보였으며, 모델에 대한 추가적인 지식 없이 효과적으로 작동합니다.



### CLIPCleaner: Cleaning Noisy Labels with CLIP (https://arxiv.org/abs/2408.10012)
Comments:
          Accepted to ACMMM2024

- **What's New**: 이번 논문은 CLIP 모델을 활용한 샘플 선택 방법인 CLIPCleaner를 제안하여 노이즈 레이블에 대한 학습(LNL) 문제를 해결하는 새로운 접근법을 소개합니다. 기존의 방법들은 self-confirmation bias를 극복하기 어렵고, 고급 모듈에 의존하는 경우가 많았으나, CLIPCleaner는 이러한 문제를 단순화하여 단일 단계의 샘플 선택을 가능하게 합니다.

- **Technical Details**: CLIPCleaner는 CLIP 모델을 기반으로 하여 자동 생성된 설명 class prompts를 사용하여 제로샷(zero-shot) 분류기를 사용하며, 시각적/의미적 유사성을 고려하여 샘플 선택을 수행합니다. 이 방법은 training model과 독립적으로 작동하여 noisy labels의 영향을 방지합니다. 또한, MixFix라는 새로운 semi-supervised learning 방법을 도입하여 노이즈가 있는 데이터셋에서 초기 clean subset을 확장합니다.

- **Performance Highlights**: CLIPCleaner는 CIFAR10 및 CIFAR100과 같은 다양한 벤치마크 데이터셋에서 경쟁력 있는 성능을 보여주었으며, 실제 데이터셋에서도 우수한 결과를 입증하였습니다. 이 방법은 단순함에도 불구하고 기존 방법들보다 우수한 성능을 달성했습니다.



### P3P: Pseudo-3D Pre-training for Scaling 3D Masked Autoencoders (https://arxiv.org/abs/2408.10007)
Comments:
          Under review. Pre-print

- **What's New**: 본 연구에서는 3D 데이터 수집의 어려움을 극복하기 위해, 실제 3D 데이터와 이미지를 통해 생성된 pseudo-3D 데이터를 활용하는 혁신적인 self-supervised pre-training 프레임워크를 제안합니다.

- **Technical Details**: 제안된 방법은 Sparse Weight Indexing이라는 효율적인 3D token embedding 방식을 사용하며, 2D reconstruction target을 통해 계산 복잡도를 줄입니다. 기존의 방법들보다 탁월한 성능을 보여줍니다.

- **Performance Highlights**: 우리는 3D 분류 및 few-shot 학습에서 state-of-the-art 성과를 달성하며, 높은 pre-training 및 다운스트림 fine-tuning 효율성을 유지합니다.



### Boosting Open-Domain Continual Learning via Leveraging Intra-domain Category-aware Prototyp (https://arxiv.org/abs/2408.09984)
- **What's New**: 본 논문에서는 Open-Domain Continual Learning(ODCL) 문제를 해결하기 위한 새로운 접근 방식인 DPeCLIP을 제안합니다. 이 방법은 intra-domain category-aware prototypes를 기반으로 하여 Task-ID를 식별하고 각 도메인과 관련된 지식을 보존하는 데 중점을 두고 있습니다.

- **Technical Details**: DPeCLIP은 training-free 방식의 Task-ID discriminator를 활용해 이미지의 Task-ID를 분류합니다. 각 도메인 내에서 카테고리 인식 프로토타입을 도입하여 지식을 유지하며, 텍스트 및 이미지 브랜치에서 self-attention과 cross-attention 모듈을 적용해 도메인-인식 정보를 인코딩합니다.

- **Performance Highlights**: 11개의 데이터셋에 대한 실험을 통해 DPeCLIP은 class-incremental 및 task-incremental 설정에서 각각 2.37% 및 1.14%의 평균 성능 향상을 이루어냈습니다. 또한, 제안한 방법은 ODCL-CIL 및 ODCL-TIL 설정 모두에서 이전 최고의 방법들과 비교하여 최첨단 성능(SOTA)을 달성했습니다.



### Weakly Supervised Pretraining and Multi-Annotator Supervised Finetuning for Facial Wrinkle Detection (https://arxiv.org/abs/2408.09952)
- **What's New**: 이번 연구는 피부 질병 및 피부 미용에 대한 관심이 높아짐에 따라 얼굴 주름 예측의 중요성이 커지고 있는 점에 주목하였습니다. 이 연구는 얼굴 주름 분할을 위해 convolutional neural networks (CNN) 모델을 자동으로 훈련할 수 있는지를 평가합니다.

- **Technical Details**: 연구에서는 여러 주석자(annotator)로부터의 데이터를 통합하는 효과적인 기술을 제시했으며, transfer learning을 활용하여 성능을 향상시킬 수 있다는 것을 보여주었습니다. 이를 통해 얼굴 주름의 신뢰할 수 있는 분할 결과를 얻을 수 있습니다.

- **Performance Highlights**: 이 접근 방식은 주름 분석과 같은 복잡하고 시간 소모적인 작업을 자동화할 수 있는 가능성을 제시하며, 피부 치료 및 진단을 용이하게 하는 데 활용될 수 있습니다.



### C${^2}$RL: Content and Context Representation Learning for Gloss-free Sign Language Translation and Retrieva (https://arxiv.org/abs/2408.09949)
- **What's New**: 이번 논문에서는 Sign Language Representation Learning (SLRL)을 위한 새로운 사전 학습 프로세스인 C${^2}$RL을 도입합니다. 이 방법은 Implicit Content Learning (ICL)과 Explicit Context Learning (ECL)을 통합하여 광라벨 없이도 효과적인 수어 표현 학습을 가능하게 하며, 기존의 방법들이 직면했던 문제들을 해결합니다.

- **Technical Details**: C${^2}$RL은 두 가지 주요 구성 요소인 ICL과 ECL을 중심으로 이루어져 있습니다. ICL은 커뮤니케이션의 미묘함을 포착하고, ECL은 수어의 맥락적 의미를 이해하여 이를 해당 문장으로 변환하는 데 집중합니다. 이 두 가지 학습 방법은 결합 최적화를 통해 강력한 수어 표현을 생성하여 SLT 및 SLRet 작업을 개선하는 데 기여합니다.

- **Performance Highlights**: 실험 결과 C${^2}$RL은 P14T에서 BLEU-4 점수를 +5.3, CSL-daily에서 +10.6, OpenASL에서 +6.2, How2Sign에서 +1.3 상승시키며, R@1 점수에서도 P14T에서 +8.3, CSL-daily에서 +14.4, How2Sign에서 +5.9 향상된 결과를 보여주었습니다. 또한, OpenASL 데이터셋의 SLRet 작업에서 새로운 베이스라인을 설정했습니다.



### Caption-Driven Explorations: Aligning Image and Text Embeddings through Human-Inspired Foveated Vision (https://arxiv.org/abs/2408.09948)
- **What's New**: 본 연구에서는 CapMIT1003 데이터를 도입하여 캡션 작업 중 인류의 주의력을 연구하고, CLIP 모델과 NeVA 알고리즘을 결합한 NevaClip이라는 제로샷(zero-shot) 방법을 제안합니다. 이 방법을 통해 캡션과 시각적 스캔패스를 예측할 수 있습니다.

- **Technical Details**: CapMIT1003 데이터셋은 참가자가 이미지에 클릭하여 캡션을 제공하면서 수행하는 클릭 중심(click-contingent) 이미지 탐사를 통해 수집되었습니다. NevaClip 알고리즘은 과거 예측된 고정점(fixation)을 기반으로 이미지를 흐리게 만들어 주의 메커니즘을 적용하여 주어진 캡션과 시각적 자극의 표현을 정렬합니다.

- **Performance Highlights**: 실험 결과 NevaClip은 기존 인간 주의력 모델보다 캡션 작성 및 자유 탐사 작업에서 더 뛰어난 결과를 보였습니다. CapMIT1003 데이터셋을 통해 얻은 스캔패스는 최신 모델을 초월하는 성과를 달성했습니다.



### ML-CrAIST: Multi-scale Low-high Frequency Information-based Cross black Attention with Image Super-resolving Transformer (https://arxiv.org/abs/2408.09940)
- **What's New**: 본 연구에서는 초해상도(Super-Resolution) 태스크에 있어 Multi-scale low-high frequency 정보를 활용하는 새로운 transformer 기반의 모델 ML-CrAIST를 제안합니다. 기존 모델들은 공간적(self-attention) 특성과 주파수(frequency) 정보를 충분히 모델링하지 못하는 한계를 가지고 있었습니다.

- **Technical Details**: ML-CrAIST는 spatial 및 channel self-attention을 동시에 운영하여 픽셀 상호작용을 모델링합니다. 이를 통해 저주파(low-frequency)와 고주파(high-frequency) 정보 간의 상관관계를 탐구하는 cross-attention block(CAB)을 포함하고 있습니다. 또한, 2D Discrete Wavelet Transformation(2dDWT)을 활용하여 주파수 밴드를 분석하고 이를 통해 저해상도 이미지에서 고해상도 이미지를 복원하는 성능을 향상시켰습니다.

- **Performance Highlights**: ML-CrAIST는 Manga109 데이터셋에서 초해상도 성능을 0.15 dB 향상시켰으며, 기존 최신 모델들에 비해 더 나은 성과를 기록했습니다.



### DiscoNeRF: Class-Agnostic Object Field for 3D Object Discovery (https://arxiv.org/abs/2408.09928)
- **What's New**: 이 논문에서는 Neural Radiance Fields (NeRFs)를 활용하여 복잡한 3D 장면을 효과적으로 세분화하는 새로운 방법인 DiscoNeRF를 제안합니다. 기존 NeRF의 한계를 극복하고, 사전 정의된 클래스 없이 사용자 상호작용 없이도 의미 있는 객체 세분화를 수행할 수 있습니다.

- **Technical Details**: DiscoNeRF는 3D 장면에서 객체 세분화를 수행하기 위해 몇 개의 경쟁하는 객체 슬롯을 도입합니다. 이러한 슬롯들과의 마스크 매칭을 통해, 각 클래스에 대한 확률 벡터를 예측하고, 이를 활용하여 2D 감독 신호를 최적화합니다. 추가적인 정규화 항목을 최소화하여 잔여 세분화의 일관성을 유지합니다. 이는 전통적인 Intersection over Union (IoU) 측정값을 확장하여 흐름 점수와 함께 작동합니다.

- **Performance Highlights**: 실험 결과, DiscoNeRF는 다양한 클래스의 3D 파노프틱 세분화를 생성할 수 있으며, 복잡한 장면에서 고품질 3D 자산을 추출하여 가상 3D 환경에서 활용할 수 있는 가능성을 보여줍니다.



### Sliced Maximal Information Coefficient: A Training-Free Approach for Image Quality Assessment Enhancemen (https://arxiv.org/abs/2408.09920)
Comments:
          6 pages, 5 figures, accepted by ICME2024

- **What's New**: 이번 논문에서는 기존의 FR-IQA 모델의 한계를 극복하기 위해 인간 시각 시스템(HVS)의 시각적 집중 추정 전략을 탐구합니다. 이를 통해 기존의 화질 평가 모델을 개선하려는 시도를 하고 있으며, 신경망 기반 모델뿐만 아니라 고전적인 모델에서도 적용할 수 있습니다.

- **Technical Details**: 이 연구에서는 Sliced Maximal Information Coefficient(SMIC)을 제안하여, 고유한 훈련 과정 없이 참조 이미지와 왜곡된 이미지 간의 통계적 종속성을 측정합니다. SMIC는 심층 특징 공간에서의 상호 정보를 계산하여 HVS와 정렬된 주의 맵을 생성하는 데 사용됩니다.

- **Performance Highlights**: 실험 결과, 제안된 주의 추정 모듈을 포함함으로써 기존의 다양한 IQA 모델의 성능이 향상되는 것을 확인했습니다. 특히 GAN 기반 및 초해상도 왜곡과 같은 다양한 왜곡 유형에서도 강력한 성능을 발휘했습니다.



### Long-Tail Temporal Action Segmentation with Group-wise Temporal Logit Adjustmen (https://arxiv.org/abs/2408.09919)
Comments:
          Accepted by ECCV 2024

- **What's New**: 이 논문은 절차적 활동 비디오에서 발생하는 긴 꼬리 문제(long-tail problem)를 처음으로 다루며, 이를 해결하기 위한 새로운 Group-wise Temporal Logit Adjustment (G-TLA) 프레임워크를 제안합니다.

- **Technical Details**: 제안된 G-TLA 프레임워크는 액티비티 정보와 액션 순서를 활용하여 logit 조정을 수행하며, 그룹별 softmax 수식을 결합합니다. 이를 통해 꼬리 액션 인식이 향상되고, 불필요한 거짓 긍정(false positives)이 줄어듭니다. 이를 통해 클레스 간 의존성을 활용하여 과분할(over-segmentation)을 완화합니다.

- **Performance Highlights**: 제안된 방법은 5개의 데이터셋에서 폭넓은 평가를 통해, 기존의 최첨단(backbone) 모델 및 표준 긴 꼬리 학습(lost-tail learning) 접근 방식들을 초월하는 성능을 보여줍니다.



### Attribution Analysis Meets Model Editing: Advancing Knowledge Correction in Vision Language Models with VisEd (https://arxiv.org/abs/2408.09916)
- **What's New**: 이번 연구에서는 Vision-LLMs (VLLMs)에 대한 모델 편집 기술을 제안합니다. 기존 연구들은 주로 텍스트 모달리티에 국한되었으나, 이 연구에서는 비주얼 표현이 어떻게 예측에 영향을 미치는지에 대한 새로운 통찰을 제공합니다.

- **Technical Details**: 연구는 주로 미드-레벨 및 레이턴 (latent) 레이어에서 비주얼 표현의 기여를 측정하기 위해 contribution allocation과 noise perturbation 방법을 사용합니다. VisEdit라는 새로운 VLLM 편집기를 도입하여 탈중심화된 비주얼 표현을 수정하여 주요 응답을 개선합니다.

- **Performance Highlights**: VisEdit는 BLIP2-OPT, MiniGPT-4, LLaVA-V1.5 등 여러 VLLM 백본을 사용하여 테스트되었으며, 기존 최첨단 편집기들에 비해 우수한 성능을 보였습니다. E-VQA 및 E-IC와 같은 공공 VLLM 편집 벤치마크 데이터셋에서도 뛰어난 신뢰성과 일반성을 입증했습니다.



### Harnessing Multi-resolution and Multi-scale Attention for Underwater Image Restoration (https://arxiv.org/abs/2408.09912)
- **What's New**: 이번 연구에서는 멀티-해상도(multi-resolution) 및 멀티-스케일(multi-scale) 이미지 분석을 통해 수중 이미지를 복원하기 위해 설계된 경량 멀티-스테이지 네트워크인 Lit-Net을 제안합니다. 이 네트워크는 최종 단계에서 원본 해상도를 유지하며, 특징을 개선하고 재구성하는 단계로 나뉘어 있습니다.

- **Technical Details**: Lit-Net의 새로운 인코더 블록은 지역 정보를 캡쳐하고 연산 속도를 높이기 위해 병렬 $1×1$ 합성곱(convolution) 레이어를 활용합니다. 또한, 색상과 세부 정보를 회복하기 위해 수정된 가중치 기반 색상 채널 전용 $l_1$ 손실($cl_1$) 함수를 통합했습니다.

- **Performance Highlights**: EUVP 데이터셋에서 Lit-Net은 $29.477$ dB PSNR($1.92\\%$ 향상) 및 $0.851$ SSIM($2.87\\%$ 향상)의 중요한 개선을 나타내며, 최근의 최첨단 방법들보다 우수한 성능을 보였습니다.



### SAM-UNet:Enhancing Zero-Shot Segmentation of SAM for Universal Medical Images (https://arxiv.org/abs/2408.09886)
- **What's New**: 본 논문은 SAM-UNet이라는 새로운 기초 모델을 제안합니다. SAM-UNet은 기존의 Segment Anything Model(SAM)에 U-Net 아키텍처를 통합하여 의료 이미지 분할에서의 성능 저하 문제를 해결하고자 합니다.

- **Technical Details**: SAM-UNet은 이미지 인코더에 병렬 CNN(Convolutional Neural Network) 브랜치를 포함시키고, 마스크 디코더에서는 다중 스케일 퓨전(multi-scale fusion) 전략을 적용하여 다양한 크기의 객체를 정확하게 분할할 수 있도록 합니다. 이 모델은 SA-Med2D-16M이라는 대규모 의료 이미지 분할 데이터셋에서 훈련되었습니다.

- **Performance Highlights**: SAM-UNet은 SA-Med2D-16M 데이터셋에서 0.883의 Dice Similarity Coefficient(DSC) 점수를 기록하며 최신 성능을 보여줍니다. 또한, 7개의 외부 의료 이미지 데이터셋에서 제로샷(segmentation zero-shot) 실험을 진행해 기존의 SAM 기반 모델들보다 훨씬 우수한 성능을 보였습니다.



### 3D-Aware Instance Segmentation and Tracking in Egocentric Videos (https://arxiv.org/abs/2408.09860)
- **What's New**: 본 논문은 3D 인식(3D awareness)을 활용한 첫 번째 인물 비디오에서의 인스턴스 분할(instance segmentation) 및 추적(tracking) 방법을 제안합니다. 이는 빠른 카메라 동작과 잦은 객체 가림(occlusion)으로 인해 발생하는 도전과제를 극복하기 위한 것입니다.

- **Technical Details**: 제안된 방법은 비디오 프레임에서 장면 기하학(scene geometry) 및 3D 객체 중심 위치(centroid tracking)를 통합하여 동적인 개인 중심 비디오를 분석할 수 있는 강력한 프레임워크를 만듭니다. 이 방법은 공간적(spatial) 및 시간적(temporal) 신호를 포함하여, 2D 기반의 기존 방법들보다 우수한 성능을 발휘합니다.

- **Performance Highlights**: EPIC Fields 데이터셋에 대한 광범위한 평가를 통해, 우리의 방법은 차세대 방법보다 7점 높은 협회 정확도(AssA) 및 4.5점 높은 IDF1 점수를 기록했습니다. 또한 다양한 객체 범주에서 ID 전환(ID switches)의 수를 73%에서 80%까지 줄였습니다.



### OccMamba: Semantic Occupancy Prediction with State Space Models (https://arxiv.org/abs/2408.09859)
Comments:
          9 pages, 4 figures

- **What's New**: 최근의 연구에서는 transformer 기반 아키텍처를 활용하여 semantic occupancy prediction의 성능을 향상시키고자 하였으나, 그에 따른 높은 계산 복잡도가 문제였습니다. 본 논문에서는 Mamba 아키텍처를 기반으로 한 OccMamba 네트워크를 제안하며, 3D 데이터를 1D로 변환할 때 발생하는 공간적 관계 손실 문제를 해결하기 위한 3D-to-1D 재정렬 작업을 소개합니다.

- **Technical Details**: OccMamba는 Mamba 아키텍처를 활용하여 3D voxel 데이터를 효율적으로 처리하며, height-prioritized 2D Hilbert expansion과 같은 재정렬 정책을 통해 공간 구조를 최대한 보존합니다. 이 방법은 LiDAR와 카메라의 시각 신호를 효과적으로 융합하여, 복잡한 주행 시나리오에서도 높은 성능을 보여줍니다.

- **Performance Highlights**: OccMamba는 OpenOccupancy, SemanticKITTI, SemanticPOSS 등 3개의 주요 benchmark에서 가장 우수한 성능을 기록하였고, 특히 OpenOccupancy에서는 이전의 최첨단 알고리즘인 Co-Occ를 각각 3.1% IoU 및 3.2% mIoU로 초월하는 성과를 보였습니다.



### Segment-Anything Models Achieve Zero-shot Robustness in Autonomous Driving (https://arxiv.org/abs/2408.09839)
Comments:
          Accepted to IAVVC 2024

- **What's New**: 이번 연구는 자율 주행을 위한 세만틱 세그멘테이션에서 Segment Anything Model(SAM)의 제로샷 적대적 견고성(zero-shot adversarial robustness)을 심층적으로 조사합니다. SAM은 특정 객체에 대한 추가 교육 없이 다양한 이미지를 인식하고 세그멘테이션할 수 있는 통합된 이미지 세그멘테이션 프레임워크입니다.

- **Technical Details**: 이 연구는 SAM 모델의 제로샷 적대적 견고성을 평가하기 위해 Cityscapes 데이터셋을 사용하여 실험을 진행했습니다. 모델 평가에는 CNN 및 ViT(as Vision Transformer) 모델이 포함되며, SAM 모델은 언어 인코더의 제약 조건 하에 평가되었습니다. 연구는 화이트박스(white-box) 공격과 블랙박스(black-box) 공격을 통해 적대적 견고성을 평가하였습니다.

- **Performance Highlights**: 실험 결과, SAM의 제로샷 적대적 견고성이 비교적 문제없는 것으로 나타났습니다. 이는 거대한 모델 매개변수와 대규모 학습 데이터가 적대적 견고성 확보에 기여하는 현상을 보여줍니다. 연구 결과는 SAM이 인공지능 일반화 모델(AGI)의 초기 프로토타입으로서 향후 안전한 자율주행 구현에 중요한 통찰을 제공함을 강조합니다.



### SurgicaL-CD: Generating Surgical Images via Unpaired Image Translation with Latent Consistency Diffusion Models (https://arxiv.org/abs/2408.09822)
- **What's New**: 본 연구에서는 	extit{SurgicaL-CD}라는 새로운 일관성 증류(diffusion) 방법을 도입하여 페어링 데이터 없이도 몇 가지 샘플링 단계로 현실적인 수술 이미지를 생성할 수 있는 기술을 제안합니다.

- **Technical Details**: 이 방법은 시뮬레이션에서 자동으로 렌더링된 시맨틱 레이블이 포함된 수술 이미지를 생성하며, 이미지 생성 속도를 크게 향상시키기 위해 잠재 공간에서의 일관성 증류를 활용합니다. 또한, 최적 수송(OT) 기반의 색상 변환 방법을 통해 고품질 이미지를 생성하는 방법을 제시합니다.

- **Performance Highlights**: 세 가지 데이터셋에서 실험을 수행한 결과, 우리의 방법이 기존의 GAN 및 다른 유사한 방법들보다 높은 품질과 다양한 수술 이미지를 생성할 수 있으며, 실제 이미지와의 조합을 통해 세분화 모델의 성능이 10% 개선됨을 보여주었습니다.



### Latent Diffusion for Guided Document Table Generation (https://arxiv.org/abs/2408.09800)
Comments:
          Accepted in ICDAR 2024

- **What's New**: 이 연구는 복잡한 테이블 구조의 주석이 달린 이미지를 생성하기 위한 새로운 접근 방식을 제안합니다. 이 방법은 latent diffusion models를 활용하여 행과 열의 mask 이미지를 조건으로 사용하여 테이블 이미지 생성을 가이드합니다.

- **Technical Details**: 연구진은 latent diffusion models (LDMs) 기반의 조건부 생성 기법을 통해 다양한 테이블 이미지를 합성합니다. 이 방식은 특정 구조를 설명하는 입력 마스크를 바탕으로 복잡한 테이블 이미지를 생성합니다.

- **Performance Highlights**: 생성된 테이블 이미지는 YOLOv5 객체 탐지 모델을 학습하는 데 사용되며, pubtables-1m 테스트셋에서 실험 결과 평균 정밀도(mAP) 값이 최신 모델과 유사한 성능을 나타내었습니다. 또한 생성된 합성 데이터의 낮은 FID 결과는 제안된 방법론의 효과성을 입증합니다.



### Cross-composition Feature Disentanglement for Compositional Zero-shot Learning (https://arxiv.org/abs/2408.09786)
Comments:
          work in progress

- **What's New**: 이 논문은 Compositional Zero-shot Learning (CZSL)에서 시각적 특징의 분해(disentanglement)를 개선하기 위한 새로운 접근법을 제안합니다. 본 연구의 핵심은 다양한 구성(compositions) 간에 일반화된 원시 특성을 학습하는 것으로, 이는 cross-composition feature disentanglement를 통해 이루어집니다.

- **Technical Details**: 제안된 방법은 compositional graph를 활용하여 구성 간 원시 공유 관계를 정의하고, CLIP이라는 대형 비전-언어 모델에 dual cross-composition disentangling adapters (L-Adapter 및 V-Adapter)를 통합하여 구성합니다. L-Adapter는 GNN을 사용하여 텍스트 특성을 전파하며, V-Adapter는 서로 관련성이 높은 이미지 특징을 강조하여 일반화 가능한 특성을 학습합니다.

- **Performance Highlights**: MIT-States 및 UT-Zappos 벤치마크에서 새로운 SOTA(최신 기술 동향)를 달성했으며, CAILA와 비교하여 더 고유하고 일반화 가능한 속성 임베딩을 제공합니다. 또한, C-GQA에서도 매우 경쟁력 있는 성능을 보입니다.



### Event Stream based Human Action Recognition: A High-Definition Benchmark Dataset and Algorithms (https://arxiv.org/abs/2408.09764)
Comments:
          In Peer Review

- **What's New**: CeleX-HAR 데이터셋은 해상도가 1280x800인 대규모 인간 행동 인식 데이터셋으로, 총 124,625개의 비디오 시퀀스를 포함하고 있으며 150개의 일반적인 행동 카테고리를 포괄합니다.

- **Technical Details**: EVMamba라는 새로운 비전 백본 네트워크를 제안하며, 이는 공간 평면 다중 방향 스캔과 새로운 복셀 시계열 스캔 메커니즘을 갖추고 있습니다.

- **Performance Highlights**: CeleX-HAR 데이터셋을 기반으로 훈련된 20개 이상의 인식 모델은 성능 비교를 위한 좋은 플랫폼을 제공합니다.



### A Unified Framework for Iris Anti-Spoofing: Introducing IrisGeneral Dataset and Masked-MoE Method (https://arxiv.org/abs/2408.09752)
- **What's New**: 이 논문에서는 iris 인식 시스템의 보안성을 강화하기 위해 새로운 IrisGeneral 데이터셋을 제안합니다. 이 데이터셋은 10개의 서브셋, 7개의 데이터베이스, 4개의 기관에서 수집한 6종류의 장치로 구성되어 있습니다. 또한, cross-domain capability를 평가하기 위해 설계된 3가지 프로토콜을 포함합니다.

- **Technical Details**: IrisGeneral 데이터셋은 average performance, cross-racial generalization, cross-device generalization을 평가하는 데 필요한 프로토콜을 갖추고 있습니다. 이 데이터셋의 통합 문제를 해결하기 위해 Mixture of Experts (MoE) 모델을 사용하였으며, MMoE(Masked-MoE)라는 새로운 방법론을 도입하여 overfitting 문제를 완화하고 일반화 능력을 향상시켰습니다.

- **Performance Highlights**: 실험 결과, CLIP 모델과 함께 제안된 MMoE 방식이 IrisGeneral 데이터셋에서 최고의 성능을 보여주며, 세 가지 프로토콜 모두에서 강력한 일반화 능력을 달성했습니다.



### Enhanced Cascade Prostate Cancer Classifier in mp-MRI Utilizing Recall Feedback Adaptive Loss and Prior Knowledge-Based Feature Extraction (https://arxiv.org/abs/2408.09746)
- **What's New**: 이 논문은 전립선암(mpMRI) 진단에서 임상 정보를 통합한 자동 등급 분류 솔루션을 제안합니다. 기존 연구는 샘플 분포의 불균형 문제를 해결하지 못하고 있어, 이에 대한 개선이 이루어진 것입니다.

- **Technical Details**: 우리는 기존의 PI-RADS 기준을 수학적으로 모델링하여 모델 훈련에 진단 정보를 통합하는 Prior Knowledge-Based Feature Extraction 을 도입하며, 극심한 불균형 데이터 문제를 해결하기 위한 Adaptive Recall Feedback Loss 를 제안합니다. 마지막으로, Enhanced Cascade Prostate Cancer Classifier 를 설계하여 전립선암을 여러 레벨로 분류합니다.

- **Performance Highlights**: 본 연구는 PI-CAI 데이터셋에서 실험 검증을 통해 다른 방법들보다 더 균형 잡힌 정확도와 Recall을 보이며 우수한 성과를 기록하였습니다.



### RealCustom++: Representing Images as Real-Word for Real-Time Customization (https://arxiv.org/abs/2408.09744)
Comments:
          23 pages

- **What's New**: 본 연구에서는 텍스트와 이미지의 불일치 문제를 해결하기 위해 RealCustom++라는 새로운 패러다임을 제안합니다. 이 방법은 주어진 주제를 실제 단어로 표현함으로써 텍스트의 제어 가능성과 주제의 유사성을 동시에 최적화할 수 있도록 합니다.

- **Technical Details**: RealCustom++는 'train-inference' 분리형 프레임워크를 도입하여, 훈련 과정에서 비전 조건과 실제 단어 간의 정렬을 학습하며, 추론 과정에서는 Adaptive Mask Guidance(AMG)를 사용하여 특정 목표 단어의 생성을 맞춤화합니다. 이 과정에서 Cross-layer Cross-Scale Projector(CCP)를 사용하여 주제 특징을 견고하게 추출합니다.

- **Performance Highlights**: RealCustom++는 기존의 pseudo-word 접근 방식에 비해 높은 주제 유사성과 텍스트 제어 가능성을 동시에 달성하였으며, 오픈 도메인에서도 우수한 일반화 능력을 보여주었습니다.



### R2GenCSR: Retrieving Context Samples for Large Language Model based X-ray Medical Report Generation (https://arxiv.org/abs/2408.09743)
Comments:
          In Peer Review

- **What's New**: 본 논문은 X-ray 의료 보고서 생성을 위한 새로운 프레임워크인 R2GenCSR를 제안합니다. 이는 컨텍스트 샘플을 활용하여 대형 언어 모델(LLMs)의 성능을 향상시키는 방법을 제공합니다.

- **Technical Details**: Mamba라는 비전 백본을 도입하여 선형 복잡도로 높은 성능을 달성하며, 트레이닝 단계에서 각 미니 배치의 샘플을 위한 컨텍스트 샘플을 검색하여 기능 표현을 강화합니다. 시각 토큰과 컨텍스트 정보를 LLM에 공급하여 고품질 의료 보고서를 생성합니다.

- **Performance Highlights**: IU-Xray, MIMIC-CXR, CheXpert Plus라는 세 개의 X-ray 보고서 생성 데이터셋에서 광범위한 실험을 진행하였으며, 제안된 프레임워크의 효과성을 완벽하게 검증하였습니다.



### TraDiffusion: Trajectory-Based Training-Free Image Generation (https://arxiv.org/abs/2408.09739)
Comments:
          The code: this https URL

- **What's New**: 이 논문에서는 사용자가 마우스 궤적을 통해 이미지 생성을 쉽고 직관적으로 조작할 수 있는, 훈련이 필요 없는 궤적 기반의 T2I(텍스트-투-이미지) 접근법인 TraDiffusion을 제안합니다.

- **Technical Details**: TraDiffusion은 거리 인식 에너지 함수(distance awareness energy function)를 설계하여 잠재 변수(latent variables)를 효과적으로 안내하며, 생성 초점이 궤적이 정의하는 영역 내에 있도록 합니다. 이 에너지 함수는 궤적에 맞추어 생성을 더 가깝게 가져가는 제어 함수와 궤적에서 멀리 떨어진 영역의 활동을 줄이는 이동 함수를 포함합니다.

- **Performance Highlights**: COCO 데이터셋을 통한 광범위한 실험 및 질적 평가 결과, TraDiffusion은 더 자연스러운 이미지 조작을 용이하게 하며, 생성된 이미지 내에서 두드러진 영역, 속성 및 관계를 조절할 수 있는 능력을 보여줍니다.



### Mutually-Aware Feature Learning for Few-Shot Object Counting (https://arxiv.org/abs/2408.09734)
Comments:
          Submitted to Pattern Recognition

- **What's New**: 이 연구는 다중 클래스 시나리오에서의 객체 식별 문제에 대한 새로운 접근방식을 제시합니다. 기존의 few-shot object counting 방법들이 가지고 있었던 '타겟 혼동(target confusion)' 문제를 해결하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 구제안하는 프레임워크인 Mutually-Aware FEAture Learning(MAFEA)은 쿼리 이미지와 예시 이미지 간의 상호 작용을 통해 특징을 추출합니다. MAFEA는 cross-attention을 활용하여 쿼리와 예시 특징 간의 양방향 관계를 포착하고, self-attention을 통해 내부 관계를 반영합니다. 또한, 배경 토큰(background token)을 도입하여 배경과 타겟 객체 간의 구분을 명확히 합니다.

- **Performance Highlights**: MAFEA는 FSCD-LVIS와 FSC-147의 두 가지 도전적인 벤치마크에서 새로운 선두 성과를 달성했습니다. 실험 결과, 타겟 혼동 문제의 정도가 현저히 줄어들었음을 보여줍니다.



### Pedestrian Attribute Recognition: A New Benchmark Dataset and A Large Language Model Augmented Framework (https://arxiv.org/abs/2408.09720)
Comments:
          MSP60K PAR Benchmark Dataset, LLM based PAR model, In Peer Review

- **What's New**: 이 논문에서는 새로운 대규모 보행자 속성 인식 데이터셋인 MSP60K를 제안합니다. 이 데이터셋은 60,122개의 이미지와 57개의 속성 주석을 포함하며, 8가지 시나리오에서 수집되었습니다. 또한, LLM(대규모 언어 모델)으로 강화된 보행자 속성 인식 프레임워크를 제안하여 시각적 특징을 학습하고 속성 분류를 위한 부분 인식을 가능하게 합니다.

- **Technical Details**: MSP60K 데이터셋은 다양한 환경과 시나리오를 반영하여 보행자 이미지를 수집하였으며, 구조적 손상(synthetic degradation) 처리를 통해 현실 세계의 도전적인 상황을 시뮬레이션합니다. 데이터셋에 대해 17개의 대표 보행자 속성 인식 모델을 평가하였고, 랜덤 분할(random split)과 크로스 도메인 분할(cross-domain split) 프로토콜을 사용하여 성능을 검증하였습니다. LLM-PAR 프레임워크는 비전 트랜스포머(Vision Transformer) 기반으로 이미지 특징을 추출하며, 멀티-임베딩 쿼리 트랜스포머(Multi-Embedding Query Transformer)를 통해 속성 분류를 위한 부분 인식 특징을 학습합니다.

- **Performance Highlights**: 제안된 LLM-PAR 모델은 PETA 데이터셋에서 mA 및 F1 메트릭 기준으로 각각 92.20 / 90.02의 새로운 최첨단 성능을 달성하였으며, PA100K 데이터셋에서는 91.09 / 90.41의 성능을 기록했습니다. 이러한 성능 향상은 MSP60K 데이터셋과 기존 PAR 벤치마크 데이터셋에서 폭넓은 실험을 통해 확인되었습니다.



### Dataset Distillation for Histopathology Image Classification (https://arxiv.org/abs/2408.09709)
- **What's New**: 본 논문에서는 병리학 이미지 데이터셋을 위한 새로운 dataset distillation 알고리즘인 Histo-DD를 소개합니다. 이 알고리즘은 stain normalisation과 model augmentation을 통합하여, 다양한 색상 이질성을 가진 병리학 이미지와의 호환성을 크게 향상시킵니다.

- **Technical Details**: Histo-DD는 원래의 대규모 데이터셋을 매우 작은 합성 샘플 집합으로 압축하기 위해 설계되었습니다. 이 알고리즘은 cropped patches에서 합성 샘플을 생성하며, layer-wise gradients의 차이를 최소화하여 성능을 보장합니다. 또한, 개선된 데이터 증강(data augmentation) 기법과 함께 차별화된 stain normalisation을 도입하여 샘플의 품질을 향상시킵니다.

- **Performance Highlights**: Histo-DD는 Camelyon16, TCGA-IDH, UniToPath의 세 가지 공개 WSI 데이터셋에서 테스트되었으며, 원하는 패치보다 더 정보가 풍부한 합성 패치를 생성할 수 있음을 보여주었습니다. 본 연구에서는 합성 샘플이 원본 대규모 데이터셋을 대체할 수 있을 정도로 우수한 정보를 유지하며, 훈련 노력을 상당히 줄일 수 있음을 입증했습니다.



### MePT: Multi-Representation Guided Prompt Tuning for Vision-Language Mod (https://arxiv.org/abs/2408.09706)
- **What's New**: 최근의 Vision-Language Models (VLMs) 발전으로, prompt tuning 기법이 다양한 다운스트림 작업에 모델을 적응시키는 데 큰 잠재력을 보였습니다. 본 논문에서는 기존 낱말 조정 방식의 한계를 보완하기 위해 Multi-Representation Guided Prompt Tuning (MePT)이라는 새로운 방법을 제안하였습니다.

- **Technical Details**: MePT는 세 가지 가지(branch)로 구성된 프레임워크를 사용하며, 다양한 시각적 특징을 포착하는 데 중점을 둡니다. 첫 번째 가지는 global branch로, CLIP 모델과의 정렬을 통해 글로벌 시각 특징을 다룹니다. 두 번째는 augmented branch로, 다양한 주목 패턴을 활용하여 도메인 특화 작업을 통해 성과를 올립니다. 마지막으로 vanilla branch는 일반적인 시각적 지식을 보존하는 데 중점을 둡니다. 각 가지에서 예측된 값을 통합하는 parameter-efficient self-ensemble 전략을 도입합니다.

- **Performance Highlights**: MePT의 효과를 검증하기 위해 11개의 다양한 데이터 세트에서 광범위한 실험을 수행하였으며, category shift 및 domain shift 작업에서 상당한 개선이 나타났습니다. 이는 MePT가 모델의 일반화를 강화하고 다양한 시각적 특징을 효과적으로 활용하게 해줍니다.



### Photorealistic Object Insertion with Diffusion-Guided Inverse Rendering (https://arxiv.org/abs/2408.09702)
Comments:
          ECCV 2024, Project page: this https URL

- **What's New**: 이번 논문에서는 기존의 모델들이 단일 이미지에서 장면의 조명 효과를 충분히 이해하지 못한다고 지적하며, 이를 극복하기 위해 개인화된 대규모 Diffusion 모델을 물리 기반 역 렌더링 프로세스의 가이드로 사용하는 새로운 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 세 가지 주요 기여를 포함합니다: 1) 물리 기반 렌더러를 사용하여 조명과 3D 자산 간의 상호작용을 정확히 시뮬레이션하여 최종 합성 이미지를 생성합니다. 2) 입력 이미지와 삽입된 객체 유형에 기반한 경량 개인화 체계를 제안합니다. 3) 개인화를 활용하고 훈련 안정성을 개선하는 SDS 손실 변형을 설계합니다.

- **Performance Highlights**: 실험 결과, 제안된 DiPIR 방법이 실내 및 실외 데이터셋에서 기존 최첨단 조명 추정 방법보다 우수한 성능을 보임을 입증하였습니다.



### MambaLoc: Efficient Camera Localisation via State Space Mod (https://arxiv.org/abs/2408.09680)
- **What's New**: 본 연구는 Selective State Space (SSM) 모델을 시각적 로컬라이제이션에 혁신적으로 적용하여 새로운 모델인 MambaLoc을 소개합니다. MambaLoc은 효율적인 기능 추출과 빠른 계산을 통해 훈련 효율성을 극대화합니다.

- **Technical Details**: MambaLoc은 Mamba의 강점을 활용하여 Sparse 데이터 환경에서도 강인한 성능을 출력할 수 있으며, Global Information Selector (GIS)를 제안합니다. GIS는 비선택적 Non-local Neural Networks의 전역 특징 추출 능력을 구현하기 위해 SSM을 이용합니다.

- **Performance Highlights**: 7Scenes Dataset에서 MambaLoc은 단 22.8초 만에 0.05%의 훈련 샘플로 최첨단 번역 정확도를 달성했습니다. 다양한 실내 및 실외 데이터셋에 대한 광범위한 실험적 검증을 통해 MambaLoc의 효과성과 GIS의 다양성을 입증했습니다.



### Image-based Freeform Handwriting Authentication with Energy-oriented Self-Supervised Learning (https://arxiv.org/abs/2408.09676)
Comments:
          Accepted by TMM

- **What's New**: 최근의 연구에서 자율 감독 학습(self-supervised learning)을 활용하여 간편한 필기 인증(freeform handwriting authentication) 시스템을 개발했습니다. 이를 위해 새로운 네트워크 구조인 SherlockNet을 제안하였습니다.

- **Technical Details**: SherlockNet은 에너지 지향(energy-oriented) 이진 대조적(self-supervised) 학습 프레임워크로, 명확한 구조를 가지고 있습니다: (i) 프리프로세싱(pre-processing): 필기 데이터를 에너지 분포로 변환; (ii) 일반화된 사전 훈련(generalized pre-training): 두 가지 동적 대조 학습을 통해 고차원 필기 특징 학습; (iii) 개인화된 미세 조정(personalized fine-tuning): 소량의 라벨 데이터로 훈련된 지식 조정; (iv) 실제 응용(practical application): 복잡한 데이터에서도 개별 필기 식별.

- **Performance Highlights**: 여섯 개의 기준 데이터셋에서 SherlockNet의 Robustness와 Efficiency를 평가한 결과 우수성을 입증하였습니다. 특히 EN-HA 데이터셋을 통해 실제 적용 시의 안전성을 보여주었습니다.



### Implicit Grid Convolution for Multi-Scale Image Super-Resolution (https://arxiv.org/abs/2408.09674)
- **What's New**: 본 논문에서는 Super-Resolution(SR) 문제를 해결하기 위해 새롭고 효율적인 다중 스케일 훈련 프레임워크를 제안했습니다. 이를 위해 Implicit Grid Convolution(IGConv) 업샘플러를 도입하여 여러 정수 스케일을 단일 모델에서 동시에 훈련하는 방법을 제시했습니다.

- **Technical Details**: 저자들은 전통적인 SR에서 고정 스케일 접근 방식의 단점을 지적하고, 다양한 스케일에서 유사한 특징을 추출하는 모델을 발견했습니다. IGConv를 사용하며 Sub-Pixel Convolution(SPConv)을 통합하였으며, 주어진 스케일에 특화된 모델 없이도 효과적으로 성능을 개선하도록 설계되었습니다. 또한 IGConv+를 통해 spectral bias, 입력 독립적 업샘플링 및 앙상블 예측 기능을 도입하여 성능을 더욱 향상시켰습니다.

- **Performance Highlights**: SRFormer-IGConv+ 모델은 Urban100×4에서 PSNR 0.25dB 향상된 성능을 보이며, 훈련 예산과 저장된 매개변수를 각각 1/3로 줄였습니다. EDSR, SRFormer, MambaIR 모델에서도 PSNR이 각각 0.16dB, 0.25dB, 0.12dB 향상되었습니다.



### SG-GS: Photo-realistic Animatable Human Avatars with Semantically-Guided Gaussian Splatting (https://arxiv.org/abs/2408.09665)
Comments:
          12 pages, 5 figures

- **What's New**: 이 논문에서는 단안 비디오에서 사실적인 애니메이션 가능한 인간 아바타를 생성하는 새로운 방법인 SG-GS(Semantically-Guided 3D Gaussian Splatting)를 제안합니다. 이 방법은 인간 신체의 의미 정보를 통합하여 동적인 아바타의 세밀한 디테일 재구성을 가능하게 합니다.

- **Technical Details**: SG-GS는 의미가 내장된 3D 가우시안, 뼈대 기반의 강체 변형(skeleton-driven rigid deformation), 비강체 의류 동역학(non-rigid cloth dynamics deformation)을 사용하여 단안 비디오로부터 애니메이션 가능한 아바타를 생성합니다. 또한, SMPL의 의미 사전(semantic prior)을 활용한 의미 인식 바디 부분 레이블링을 위한 Semantic Human-Body Annotator(SHA)를 설계하였습니다.

- **Performance Highlights**: 실험 결과, SG-GS는 현재의 최첨단 기술(SOTA: state-of-the-art)와 비교하여 뛰어난 기하학적(geometry) 및 외관(appearance) 재구성 성능을 보여주었으며, 더 빠른 렌더링 속도를 유지하면서 의미 정확도와 렌더링 품질을 크게 향상시켰습니다.



### CHASE: 3D-Consistent Human Avatars with Sparse Inputs via Gaussian Splatting and Contrastive Learning (https://arxiv.org/abs/2408.09663)
Comments:
          13 pages, 6 figures

- **What's New**: 최근 인간 아바타 합성을 위한 연구에서, radiance fields를 활용하여 사실감 넘치는 애니메이션 가능한 인간 아바타를 재구성하는 방법론이 발전하였습니다. 본 논문에서는 새롭게 제안하는 CHASE가 어떻게 저조한 세밀한 재구성과 3D 일관성을 개선하는지를 다루고 있습니다.

- **Technical Details**: CHASE는 intrinsic 3D consistency와 3D geometry contrastive learning을 활용하여, sparse inputs에서조차도 full inputs에서와 비교할 수 있는 성능을 달성합니다. 이를 위해 skeleton-driven rigid deformation과 non-rigid cloth dynamics deformation을 통합하여 인간 아바타를 재구성하며, Dynamic Avatar Adjustment (DAA)를 통해 선택된 유사한 pose/image를 기반으로 변형된 Gaussian을 조정합니다.

- **Performance Highlights**: ZJU-MoCap 및 H36M 데이터셋에서의 실험 결과, CHASE는 sparse inputs 환경에서도 뛰어난 성능을 보이며, full input 환경에서도 기존 SOTA 방법을 능가하는 성과를 기록하였습니다.



### ExpoMamba: Exploiting Frequency SSM Blocks for Efficient and Effective Image Enhancemen (https://arxiv.org/abs/2408.09650)
- **What's New**: ExpoMamba는 기존의 컴퓨터 비전 모델의 한계를 극복하기 위해 디자인된 새로운 아키텍처입니다. 이 모델은 수정된 U-Net 아키텍처 내에서 주파수 상태 공간 구성 요소를 통합하여 효율성과 효과성을 결합합니다.

- **Technical Details**: ExpoMamba 모델은 혼합 노출 문제를 해결하기 위해 2D-Mamba 블록과 주파수 상태 공간 블록(FSSB)을 결합합니다. 이 아키텍처는 특별히 고해상도 이미지의 처리에서 발생하는 계산적 비효율성을 해결하며, 36.6ms의 추론 시간을 달성하여 전통적인 모델보다 2-3배 빠르게 저조도 이미지를 개선합니다.

- **Performance Highlights**: ExpoMamba는 경쟁 모델 대비 PSNR(피크 신호 대 잡음비)을 약 15-20% 개선하며 실시간 이미지 처리 애플리케이션에 매우 적합한 모델로 입증되었습니다.



### C2P-CLIP: Injecting Category Common Prompt in CLIP to Enhance Generalization in Deepfake Detection (https://arxiv.org/abs/2408.09647)
Comments:
          10 pages, 5 figures

- **What's New**: 이 연구에서는 AIGC(Artificial Intelligence Generated Content) 탐지 기술을 개선하기 위해 CLIP(Contrastive Language–Image Pretraining) 모델의 적합성을 분석하고, 이를 통해 범용 탐지기를 개발하는 방법을 제시합니다. 새로운 접근법인 C2P-CLIP(Category Common Prompt CLIP)을 제안하여 탐지 성능을 향상시켰습니다.

- **Technical Details**: 두 가지 주요 문제가 다루어졌습니다: 1) CLIP 모델의 특징이 어떻게 깊이 있는 탐지에 효과적인지를 이해하는 것과 2) CLIP의 탐지 잠재력을 탐구하는 것입니다. C2P-CLIP는 카테고리 공통 프롬프트를 이미지 인코더에 주입하여 카테고리 관련 개념을 통합함으로써 탐지 성능을 향상시킵니다. 이 과정에서 CLIP의 텍스트 인코더 매개변수는 고정되어 있고, 이미지 인코더는 LoRA를 통해 훈련됩니다.

- **Performance Highlights**: C2P-CLIP는 원래 CLIP 모델에 비해 탐지 정확도가 12.41% 개선되었으며, 20개 생성 모델로 구성된 데이터셋에서 포괄적인 실험을 수행하여 최첨단 성능을 입증하였습니다. 이러한 성능은 추가적인 매개변수를 도입하지 않고도 달성되었습니다.



### The First Competition on Resource-Limited Infrared Small Target Detection Challenge: Methods and Results (https://arxiv.org/abs/2408.09615)
- **What's New**: 본 논문은 리소스 제한 환경에서의 적외선 소형 목표 탐지(인증명: LimitIRSTD)에 대한 첫 번째 경진대회를 요약합니다. 이 경진대회는 약한 감독적 적외선 소형 목표 탐지(Track 1)와 경량의 적외선 소형 목표 탐지(Track 2) 두 개의 트랙으로 구성됩니다. 46개 팀이 Track 1에, 60개 팀이 Track 2에 등록하여 경쟁을 진행했습니다.

- **Technical Details**: 이 경진대회는 WideIRSTD 데이터셋을 기반으로 하며, 데이터셋은 다양한 목표 모양, 파장 및 이미지 해상도를 포함합니다. Track 1은 단일 포인트 감독 하의 약한 감독적 탐지, Track 2는 픽셀 수준 감독 하의 경량 탐지로 구성되어 있습니다. 성능 평가는 Intersection over Union (IoU), 탐지 확률 (Pd), 그리고 허위 경고율 (Fa) 등의 지표를 사용합니다.

- **Performance Highlights**: Track 1에서 상위 성적을 거둔 팀들은 사전 훈련된 모델이나 전통적인 방법을 통해 생성된 의사 라벨을 사용하여 우수한 결과를 달성했습니다. Track 2에서는 지식 증류(knowledge distillation), 네트워크 프루닝(pruning) 및 경량 설계를 통해 경량을 유지하면서 뛰어난 성능을 기록했습니다.



### Enhancing ASL Recognition with GCNs and Successive Residual Connections (https://arxiv.org/abs/2408.09567)
Comments:
          To be submitted in G2-SP CV 2024. Contains 7 pages, 5 figures

- **What's New**: 이 연구는 Graph Convolutional Networks (GCNs)와 successive residual connections를 통합하여 American Sign Language (ASL) 인식을 향상시키는 새로운 접근법을 제시합니다.

- **Technical Details**: MediaPipe 프레임워크를 사용하여 각 손 제스처의 주요 랜드마크를 추출하고, 이를 기반으로 그래프 표현을 구성합니다. 전처리 파이프라인에서는 변환 및 스케일 정규화 기법이 포함되어 데이터셋 전반에 걸쳐 일관성을 보장합니다. 구성된 그래프는 residual connections가 포함된 GCN 기반 신경망 구조에 전달되어 네트워크 안정성을 향상시킵니다.

- **Performance Highlights**: 이 방법은 ASL 알파벳 데이터셋에서 99.14%의 검증 정확도를 달성하여 기존 방법을 획기적으로 능가하며 ASL 인식의 새로운 기준을 세웠습니다.



### Generating Automatically Print/Scan Textures for Morphing Attack Detection Applications (https://arxiv.org/abs/2408.09558)
Comments:
          Paper under revision process in Journal

- **What's New**: 이 논문은 Morphing Attack Detection (MAD) 시스템을 위한 새로운 데이터셋 생성 방법론을 제안합니다. 특히, 일반적으로 사용되는 데이터의 부족 문제를 해결하고자 하는 지속적인 노력이 필요합니다.

- **Technical Details**: 제안된 방법은 Pix2pix와 CycleGAN을 이용하여 디지털 프린트/스캔 얼굴 이미지를 자동으로 생성하는 두 가지 방식으로 구성됩니다. 첫 번째 방법은 이미지 간의 스타일 전송을 수행하며, 두 번째 방법은 인쇄 및 스캔 과정에서 생성된 노이즈 및 아티팩트를 분리하여 적용하는 반자동화된 프로세스입니다.

- **Performance Highlights**: 우리의 방법은 FRGC/FERET 데이터베이스에서 각각 3.84%와 1.92%의 Equal Error Rate (EER)을 달성했습니다. 이 과정에서 합성 이미지와 텍스처 전송 이미지를 훈련 데이터로 포함하여 높은 정확도를 기록했습니다.



### AnomalyFactory: Regard Anomaly Generation as Unsupervised Anomaly Localization (https://arxiv.org/abs/2408.09533)
Comments:
          Accepted to the 2nd workshop on Vision-based InduStrial InspectiON (VISION) at ECCV 2024

- **What's New**: 새로운 프레임워크인 AnomalyFactory는 비지도 학습의 anomaly generation과 anomaly localization을 동일한 네트워크 아키텍처로 통합하여, 기존의 많은 대규모 생성 모델을 필요로 하지 않고도 다양한 데이터셋에서 효과적으로 작동할 수 있도록 설계되었습니다.

- **Technical Details**: AnomalyFactory는 두 개의 생성기(BootGenerator와 FlareGenerator)와 하나의 예측기(BlazeDetector)로 구성됩니다. BootGenerator는 목표 에지 맵의 구조와 참조 색상 이미지의 외관을 학습된 히트맵의 지침을 받아 결합합니다. FlareGenerator는 BootGenerator로부터 감독 신호를 받아 생성된 이미지에서 anomaly 위치를 표시하는 히트맵을 수정합니다. BlazeDetector는 FlareGenerator로 생성된 anomaly 이미지를 정상 이미지로 변환하여 학습된 히트맵을 사용해 anomaly 픽셀을 국지화합니다.

- **Performance Highlights**: 5개 데이터셋(MVTecAD, VisA, MVTecLOCO, MADSim, RealIAD)에서 수행된 포괄적인 실험 결과 AnomalyFactory가 기존의 anomaly 생성기 대비 월등한 생성 능력과 확장성을 보여주는 것으로 입증되었습니다. 특히 MVTecAD 데이터셋에서 AnomalyFactory는 SOTA 모델 대비 4.24 IS 점수를 달성하였으며, 단일 생성기만으로도 2.37/0.11 높은 IS/LPIPS 개선을 이루었습니다.



### NAVERO: Unlocking Fine-Grained Semantics for Video-Language Compositionality (https://arxiv.org/abs/2408.09511)
- **What's New**: 이 논문에서는 비디오-언어 (VidL) 모델의 객체, 속성, 행동 및 이들의 관계 간의 조합 이해 능력을 연구하고, AARO라는 벤치마크를 구축하여 평가합니다. 또한, NEGATIVE-augmented Video-language Enhancement for ReasOning (NAVERO)라는 훈련 방법을 제안하여 조합 이해 능력을 향상시킵니다.

- **Technical Details**: AARO 벤치마크는 주어진 비디오에 대한 잘못된 행동 설명이 포함된 부정 텍스트로 구성되어 있으며, 모델은 긍정적인 텍스트와 해당 비디오를 매칭하는 것이 목표입니다. NAVERO는 부정 텍스트로 보강된 비디오-텍스트 데이터셋을 활용하며, 부정-증강된 시각-언어 매칭 손실 (visual-language matching loss)을 개발하여 작동합니다. 또한, 다양한 부정 텍스트 생성을 통해 조합 이해를 향상시킵니다.

- **Performance Highlights**: NAVERO는 비디오-언어 및 이미지-언어 조합 이해에서 다른 최신 방법들보다 유의미한 성능 향상을 보여주며, 전통적인 텍스트-비디오 검색 작업에서도 강력한 성능을 유지합니다.



### StyleBrush: Style Extraction and Transfer from a Single Imag (https://arxiv.org/abs/2408.09496)
Comments:
          9 pages, 6figures, Under Review

- **What's New**: StyleBrush는 참조 이미지를 통해 스타일을 추출하고 이를 다른 입력 시각 콘텐츠에 적용하는 새로운 스타일링 방법론입니다. 이 작업은 스타일과 구조적 요소를 효과적으로 분리하는 것을 목표로 하며, 기존의 스타일 전이 방법보다 경쟁력이 있습니다.

- **Technical Details**: StyleBrush는 ReferenceNet과 Structure Guider의 두 가지 브랜치로 구성되며, 전자는 참조 이미지에서 스타일을 추출하고 후자는 입력 이미지에서 구조적 특징을 추출합니다. 데이터 셋은 100K 고품질 스타일 이미지를 포함하고 있으며, 다양한 스타일과 높은 미적 점수로 구성됩니다.

- **Performance Highlights**: 실험 결과, StyleBrush는 정량적 및 정성적 분석 모두에서 최첨단 성능을 달성하였으며, 코드 및 데이터셋은 논문 수락 이후 공개될 예정입니다.



### Source-Free Test-Time Adaptation For Online Surface-Defect Detection (https://arxiv.org/abs/2408.09494)
Comments:
          Accepted to ICPR 2024

- **What's New**: 본 논문에서는 산업 생산에서의 표면 결함 감지(Surface Defect Detection) 분야에서, 새로운 도메인과 클래스에 적응하는 테스트 시간 적응(Test-Time Adaptation) 방법을 제안합니다. 이 방법은 모델이 새로운 데이터에 대한 적응을 실시간으로 수행할 수 있게 해주며, 추가적인 오프라인 재학습이 필요하지 않습니다.

- **Technical Details**: 제안된 방법은 두 가지 핵심 아이디어를 포함하고 있습니다. 첫째, 신뢰도가 높은 샘플만을 필터링하여 모델 업데이트에 사용하는 감독(supervisor)을 도입하여 부정확한 데이터에 의해 과도하게 편향되지 않도록 합니다. 둘째, 안정적인 pseudo labels 생성을 위한 증강 평균 예측(augmented mean prediction)과 학습을 효과적으로 통합하는 동적으로 균형 잡힌 손실(dynamically-balancing loss)을 제안합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 다양한 산업 데이터셋에서 기존의 최신 기법들을 능가하는 성능을 보였습니다. 이는 온라인 적응이 필요한 산업 환경에서 실시간으로 결함을 정확하게 감지할 수 있도록 최적화 된 방법임을 보여줍니다.



### Advances in Multiple Instance Learning for Whole Slide Image Analysis: Techniques, Challenges, and Future Directions (https://arxiv.org/abs/2408.09476)
- **What's New**: 이 논문은 Whole Slide Images (WSIs)에 대한 Multiple Instance Learning (MIL) 기술의 적용에 대한 포괄적인 개요를 제공합니다. 특히, 암 분류 및 감지에서의 MIL의 힘을 강조하며, 최근의 기술적 진전을 반영하고 있습니다.

- **Technical Details**: WSI는 H&E 염색된 조직 샘플의 기가픽셀 규모 디지털 이미지로, MIL은 이미지 패치를 처리하여 WSI 수준의 임상 레이블을 매핑하는 방식으로 작동합니다. 이 과정은 세 단계로 나뉘는데, 첫 번째 단계에서 특정 패치의 특징을 추출하고, 두 번째 단계에서 패치 임베딩을 집계하여 WSI 표현을 형성하며, 마지막 단계에서 이 표현을 WSI 레이블로 매핑합니다.

- **Performance Highlights**: MIL은 종양 탐지, 암 아형 분류, 종양 등급 매기기와 같은 다양한 조직병리학적 작업에서 뛰어난 성능을 보여주었습니다. 최근 연구에서는 MIL의 적용 가능성을 드러내며, 예측, 분류, 진단 및 예후 작업을 사이버병리학 (computational pathology)과 결합하여 현대 의학에 대한 기여를 강화하고 있습니다.



### MedMAP: Promoting Incomplete Multi-modal Brain Tumor Segmentation with Alignmen (https://arxiv.org/abs/2408.09465)
- **What's New**: 본 논문에서는 의료 영상 세분화에서 결측 모달리티(missing modality) 문제를 해결하기 위한 새로운 패러다임, Medical Modality Alignment Paradigm (MedMAP)을 제안합니다. 이 방법은 사전 훈련된 모델의 부재로 인한 갭을 줄이기 위해 모달리티의 잠재적 특징(latent features)을 공통 분포에 정렬합니다.

- **Technical Details**: MedMAP은 결측 모달리티 접근 방식의 효율성을 높이기 위해 설계되었습니다. 각 모달리티를 미리 정의된 분포 P_{mix}로 정렬함으로써,모달리티 간의 갭을 줄이고 성능을 향상시킵니다. 이 방법은 다양한 백본(backbone)에 대해 실험을 통해 검증되었습니다.

- **Performance Highlights**: BraTS2018 및 BraTS2020 데이터셋에서 MedMAP를 적용한 모델은 기존 방법보다 우수한 성능을 보였습니다. 본 연구는 세분화 작업에서 모달리티 갭을 줄이는 것이 모델의 일반화(generalization)에 긍정적인 영향을 미친다는 것을 발견했습니다.



### 3C: Confidence-Guided Clustering and Contrastive Learning for Unsupervised Person Re-Identification (https://arxiv.org/abs/2408.09464)
- **What's New**: 이번 연구에서는 비지도(person re-identification, Re-ID) 상황에서 카메라 간 검색 기능을 갖춘 특징 네트워크를 학습하기 위해, 신뢰도 기반 클러스터링 및 대조 학습(Confidence-guided Clustering and Contrastive learning, 3C) 프레임워크를 제안합니다. 3C 프레임워크는 샘플과 클러스터 간의 불일치 신뢰도를 평가하여 클러스터링을 개선하는 방법을 포함합니다.

- **Technical Details**: 비지도 Re-ID 시스템에서, 이 프레임워크는 세 가지 신뢰도를 기반으로 동작합니다: 1) 클러스터링 단계에서 HDC(Harmonic Discrepancy Clustering) 알고리즘을 구현하여 샘플과 클러스터 간의 불일치 신뢰도를 고려합니다. 2) 전파 학습 단계에서 카메라의 다양성을 반영하는 새로운 카메라 정보 엔트로피(Camera Information Entropy, CIE)를 통해 클러스터의 신뢰도를 평가합니다. 3) 역전파 학습 단계에서 유효한 하드 샘플을 선택하기 위해 신뢰 통합 하모닉 불일치를 설계합니다.

- **Performance Highlights**: 테스트 결과, 3C 프레임워크는 Market-1501, MSMT17, VeRi-776 데이터셋에서 각각 86.7%/94.7%, 45.3%/73.1%, 47.1%/90.6%의 mAP/Rank-1 정확도를 기록하여 최첨단 성능을 달성했습니다.



### Fine-Grained Building Function Recognition from Street-View Images via Geometry-Aware Semi-Supervised Learning (https://arxiv.org/abs/2408.09460)
Comments:
          This paper is currently under review

- **What's New**: 이 연구에서는 미세한 건물 기능 인식을 위한 기하학 인식 반지도 방법을 제안합니다. 이 방법은 다중 소스 데이터 간의 기하학적 관계를 활용하여 반지도 학습에서 의사 레이블의 정확성을 향상시킴으로써 건물 기능 인식의 과제를 확장하고 Cross-Categorization 시스템에 적용할 수 있도록 합니다.

- **Technical Details**: 이 방법은 세 단계로 구성됩니다. 첫 번째로, 온라인 반지도 예비 훈련 단계를 설계하여 거리 이미지에서 건물 외관 위치 정보를 정확하게 획득합니다. 두 번째 단계에서는 GIS 데이터와 거리 데이터를 효과적으로 결합하여 의사 주석의 정확성을 향상시키는 기하학 인식 거친 주석 생성 모듈을 제안합니다. 세 번째 단계에서는 새로 생성된 거친 주석과 기존 라벨 데이터 세트를 결합하여 여러 도시에서 대규모로 미세한 기능 인식을 달성합니다.

- **Performance Highlights**: 광범위한 실험 결과, 제안된 프레임워크는 미세한 건물 기능 인식에 우수한 성능을 보였습니다. 동일한 분류 시스템 내에서, 완전 감독 방법에 비해 7.6%, 최첨단 반지도 방법에 비해 4.8% 향상되었습니다. 또한, 이 방법은 교차 도시 작업에서도 잘 수행되어 OmniCity(뉴욕)에서 훈련된 모델이 새로운 지역(로스앤젤레스 및 보스턴)으로 확장될 수 있음을 보여줍니다.



### G2Face: High-Fidelity Reversible Face Anonymization via Generative and Geometric Priors (https://arxiv.org/abs/2408.09458)
- **What's New**: G2Face는 전통적인 얼굴 익명화 기법을 보완하는 혁신적인 방법으로, 생성 모델과 기하학적 정보를 결합하여 높은 품질의 reversible face anonymization을 구현합니다.

- **Technical Details**: G2Face는 3D 얼굴 모델을 활용하여 입력 얼굴의 기하학적 정보를 추출한 후, GAN 기반의 디코더와 결합하여 사실적인 익명화된 얼굴을 생성합니다.  또한, ID-irrelevant한 특징들을 유지하면서도 멀티스케일 얼굴 특징을 정확하게 통합하는 새로운 아이디 인식 특징 융합 블록(IFF)을 도입합니다.

- **Performance Highlights**: G2Face는 기존의 최첨단 기법들을 초월하는 성능을 보이며, 얼굴 익명화 및 복구 과정에서 높은 데이터 유용성을 유지합니다.



### Retina-inspired Object Motion Segmentation (https://arxiv.org/abs/2408.09454)
- **What's New**: 이 논문에서는 동적 비전 센서(DVS)를 기반으로 하는 새로운 생체 영감을 받은 알고리즘을 제안합니다. 이 알고리즘은 맘모리안 망막의 기능을 활용하여 물체 운동 감도를 계산하며, 실시간 및 자원 제약이 있는 환경에서도 적용할 수 있도록 설계되었습니다.

- **Technical Details**: 이 연구는 생체 영감을 받은 Object Motion Sensitivity (OMS) 알고리즘을 개발하여 DVS 데이터로부터 물체 운동을 분리하는 방법을 제시합니다. OMS는 맘모리안 시각 시스템에서 사용되며, 깊이 있는 신경망 모델없이도 강력한 성능을 보여줍니다. 기존 방법들과 비교해 1000배 적은 매개변수를 사용하여 처리 속도를 높였습니다.

- **Performance Highlights**: 연구 결과, 제안된 OMS 알고리즘은 기존의 7가지 최신 기법과 비교하여 Intersection over Union 및 감지율 면에서 유사하거나 우수한 성능을 보여 주며, 3차원 반성적 복잡성 감소를 통해 효율성을 높였습니다.



### Attention Is Not What You Need: Revisiting Multi-Instance Learning for Whole Slide Image Classification (https://arxiv.org/abs/2408.09449)
- **What's New**: 본 논문에서는 기존의 주의 기반(Attention-based) 다중 인스턴스 학습(MIL) 알고리즘의 한계를 지적하고, 새로운 인스턴스 기반 MIL 방법인 FocusMIL을 제안합니다. FocusMIL은 최대 풀링(max-pooling)과 전방 아모르티즈 변분 추론(forward amortized variational inference)을 기반으로 하여 보다 정확하게 종양 형태(tumour morphology)에 집중할 수 있도록 설계되었습니다.

- **Technical Details**: FocusMIL은 피드포워드 신경망(feed-forward neural network)과 최대 풀링(max-pooling) 기법을 사용합니다. 이 방법은 인스턴스의 라벨을 예측하는 데 있어 비인과적(non-causal) 요소를 무시하고 인과적(factor causal) 요인에 중점을 두어, 다중 인스턴스 가정을 고수함으로써 개발됩니다. 이 모델은 작은 배치 크기(batch size)를 사용하는 전통적인 SGD 대신 큰 배치 크기를 사용하는 미니 배치 경량화(mini-batch gradient descent) 기법을 적용하여 분류 경계를 개선합니다.

- **Performance Highlights**: Camelyon16 및 TCGA-NSCLC 데이터셋에서 FocusMIL이 기존의 기준선(baselines)보다 patch-level 분류에서 현저하게 더 나은 성능을 발휘함을 보여주었습니다. 슬라이드 수준(slide level) 예측 역시 기존 방법들과 유사한 성능을 보였으며, 특히 중요한 인스턴스와 어렵게 분류된 인스턴스를 더욱 정확하게 예측할 수 있음을 확인하였습니다.



### CLIP-CID: Efficient CLIP Distillation via Cluster-Instance Discrimination (https://arxiv.org/abs/2408.09441)
Comments:
          11 pages,8 figures

- **What's New**: 본 논문에서는 CLIP-CID라는 새로운 지식 증류 방식(knowledge distillation mechanism)을 도입하여 큰 비전-언어 기반 모델에서 작은 모델로 효과적으로 지식을 전이하는 방법을 제안합니다. 이 방법은 LAION400M 데이터에서 이미지-텍스트 쌍의 43.7%를 필터링하면서도 뛰어난 성능을 유지할 수 있도록 합니다.

- **Technical Details**: CLIP-CID는 클러스터 및 인스턴스 구별(cluster-instance discrimination) 기법을 통합하여 교사 모델(teacher model)에서 학생 모델(student model)로 지식을 효과적으로 전이합니다. 이는 학생 모델이 사전 학습 데이터의 전체적인 의미 이해(holistic semantic comprehension)를 강화하는 데 도움을 줍니다. 이 방법은 전이 학습 편향을 줄이고 증류 효율성(distillation efficiency)을 향상시킵니다.

- **Performance Highlights**: 실험 결과, CLIP-CID는 14개의 일반 데이터셋에 대해 우수한 선형 프로브(linear probe) 성능을 보여 주며, 제로 샷 분류(zero-shot classification) 및 기타 하위 작업에서 최신 성과(state-of-the-art performance)를 달성하였습니다.



### Adversarial Attacked Teacher for Unsupervised Domain Adaptive Object Detection (https://arxiv.org/abs/2408.09431)
- **What's New**: 이 논문에서는 Domain Adaptive Object Detection (DAOD) 분야에서의 성능 향상을 목표로 하는 새로운 프레임워크인 Adversarial Attacked Teacher (AAT)를 소개합니다. AAT는 교사 모델에 적대적 공격(adversarial attack)을 적용하여 가짜 라벨(pseudo-label)의 품질을 개선하는 간단하지만 효과적인 접근법입니다.

- **Technical Details**: AAT는 교사 모델에 적대적 공격을 적용하여 교사 모델이 편향을 수정하고, 과도한 신뢰를 억제하며, 신뢰가 부족한 제안을 장려하기 위해 적대적 가짜 라벨(adversarial pseudo-labels)을 생성하도록 유도합니다. 이 과정에서 높은 확신을 가지는 가짜 라벨에 대한 가중치를 높이고, 불확실한 예측의 부정적인 영향을 줄이는 적응형 가짜 라벨 정규화(adaptive pseudo-label regularization)를 도입합니다.

- **Performance Highlights**: AAT는 다양한 데이터셋에서 광범위한 테스트를 실시한 결과, Clipart1k에서 52.6 mAP를 달성하여 기존 최고의 성능(SOTA)보다 6.7% 향상되었습니다. Foggy Cityscapes에서는 53.0 mAP를 기록하여 이전의 기술들을 크게 초월하는 성과를 보여주었습니다.



### A Robust Algorithm for Contactless Fingerprint Enhancement and Matching (https://arxiv.org/abs/2408.09426)
- **What's New**: 이번 연구에서는 contactless (비접촉) 지문 이미지의 정확성을 높이기 위해 새로운 알고리즘을 제안하였습니다. 기존의 기술들이 contact (접촉) 지문 이미지에 초점을 맞춘 반면, 본 연구는 contactless 지문 특성에 적합한 방법론을 제공합니다.

- **Technical Details**: 제안된 방법은 두 개의 주요 단계로 나뉘며, 첫 번째 단계는 지문 이미지를 향상시키고 minutiae (미세구조) 기능을 추출 및 인코딩하는 오프라인 단계입니다. 두 번째 단계는 새로운 템플릿 이미지를 센서에서 수집하고, 이를 향상시키고 minutiae 기능을 추출하며 인코딩하는 온라인 과정입니다. Gabor 필터 기반의 컨텍스처 필터링을 통해 지역적 ridge (능선) 구조를 향상시키고, 이후 이진화 및 얇게 만들기를 통해 ridge/valley (능선/골짜기) 스켈레톤을 생성합니다.

- **Performance Highlights**: Proposed method는 PolyU contactless fingerprint dataset에서 2.84%의 최저 Equal Error Rate (EER)를 달성하였으며, 이는 기존의 최첨단 기술들과 비교했을 때 우수한 성능을 보여줍니다. 이러한 높은 정밀도와 회복력을 통해 contactless fingerprint 기반의 식별 시스템 구현에 효과적이고 실행 가능한 해결책을 입증합니다.



### OVOSE: Open-Vocabulary Semantic Segmentation in Event-Based Cameras (https://arxiv.org/abs/2408.09424)
Comments:
          conference

- **What's New**: OVOSE는 이벤트 카메라(E event cameras)용으로 특별히 설계된 최초의 Open-Vocabulary Semantic Segmentation 알고리즘으로, 기존의 모델들이 작업에 따라 제한적인 Closed-set semantic segmentation에 국한된 것이 벗어나고 있습니다.

- **Technical Details**: OVOSE는 두 개의 브랜치 네트워크로 작동하며, 하나는 그레이스케일 이미지에, 다른 하나는 이벤트 데이터에 대해 최적화됩니다. CLIP 스타일의 이미지 인코더와 MLP를 사용하여 텍스트-이미지 확산 모델을 적용하며, 재구성된 이미지를 위해 E2VID 모델을 활용해 지식을 증류합니다. 이 방식은 이벤트 카메라의 데이터 포맷에 적합하게 조정되어 있습니다.

- **Performance Highlights**: OVOSE는 DDD17 및 DSEC-Semantic 데이터셋에서 기존의 Closed-set 방법들 및 이벤트 기반 데이터에 적응된 기존의 Open-Vocabulary 모델들과 비교하여 우수한 성능을 보여주며, 현실 세계의 응용 가능성을 입증하고 있습니다.



### Weakly Supervised Lymph Nodes Segmentation Based on Partial Instance Annotations with Pre-trained Dual-branch Network and Pseudo Label Learning (https://arxiv.org/abs/2408.09411)
Comments:
          Accepted for publication at the Journal of Machine Learning for Biomedical Imaging (MELBA) this https URL

- **What's New**: 이번 연구에서는 lymph node segmentation을 위해 부분 인스턴스 주석(partial instance annotation)에서 학습하는 새로운 DBDMP(Dual-Branch network with Dynamically Mixed Pseudo label) 모델을 제안합니다.

- **Technical Details**: 모델은 두 개의 디코더를 통한 동적 혼합(pseudo label) 방법을 활용하여 주석이 없는 lymph node에 대한 신뢰할 수 있는 pseudo label을 생성합니다. 또한 Self-supervised pre-training 전략을 사용하여 모델의 feature extraction 성능을 강화합니다.

- **Performance Highlights**: Mediastinal Lymph Node Quantification (LNQ) 데이터셋에서 Dice Similarity Coefficient(DSC)가 11.04%에서 54.10%로 개선되었으며, Average Symmetric Surface Distance(ASSD)는 20.83 mm에서 8.72 mm로 감소했습니다. MICCAI 2023 LNQ 챌린드에서 4위의 성적을 기록했습니다.



### OPPH: A Vision-Based Operator for Measuring Body Movements for Personal Healthcar (https://arxiv.org/abs/2408.09409)
- **What's New**: 이 연구는 인간 몸의 움직임을 보건의료 목적에 맞춰 보다 정확하게 추정할 수 있도록 설계된 OPPH 연산자를 제안합니다. 이 연산자는 기존의 비전 기반 모션 추정 방식의 정확성을 향상시키며, 움직이지 않는 상태의 감지 및 장기적인 움직임 트렌드 유지에 중점을 둡니다.

- **Technical Details**: OPPH 연산자는 인간의 몸 움직임과 노이즈 특성을 고려한 다단계 필터입니다. 이 연산자는 실제 환경에서 생성된 데이터셋에서 몸이 움직이지 않는 상태를 정확하게 탐지하고 활동적인 몸 움직임을 추정하며, 장기적인 인간 몸 움직임의 변화 추세를 캡쳐할 수 있습니다.

- **Performance Highlights**: OPPH 연산자는 포즈 기반 및 옵틱 플로우 기반 방법들과 비교하여 2D 몸 움직임 속도 추정에 있어, RMSE가 2.7×10^-4 픽셀로 오류를 거의 완전히 제거했습니다. 또한, 다양한 데이터셋에서 OPPH 연산자는 기존의 방법들보다 더 높은 정확도를 보여주었습니다.



### VrdONE: One-stage Video Visual Relation Detection (https://arxiv.org/abs/2408.09408)
Comments:
          12 pages, 8 figures, accepted by ACM Multimedia 2024

- **What's New**: VrdONE 모델은 비디오 물체 간의 관계를 효과적으로 탐지하기 위한 혁신적인 접근으로, 주제 및 객체의 특징을 결합한 1D 인스턴스 세그멘테이션 방법으로 리레이션 카테고리 식별과 바이너리 마스크 생성을 동시에 수행할 수 있도록 설계되었습니다.

- **Technical Details**: VrdONE 모델은 Subject-Object Synergy (SOS) 모듈을 통해 주제와 객체가 서로를 인식하는 방법을 향상시키고, Bilateral Spatiotemporal Aggregation (BSA) 메커니즘을 통해 서로의 단기 특징들의 상호작용을 효과적으로 학습합니다. 모델은 분류 및 세그멘테이션 브랜치에서 동시에 훈련되어 관계 카테고리와 시간 경계를 동시에 얻을 수 있도록 합니다.

- **Performance Highlights**: VrdONE는 VidOR 벤치마크 및 ImageNet-VidVRD에서 최첨단 성능을 달성하였으며, 다양한 시간 규모의 관계를 인식하는 데 있어 우수한 능력을 보여줍니다.



### Combo: Co-speech holistic 3D human motion generation and efficient customizable adaptation in harmony (https://arxiv.org/abs/2408.09397)
- **What's New**: 본 논문에서는 조화로운 음성 기반 3D 인간 모션 생성 및 효율적인 맞춤형 적응을 위한 새로운 프레임워크인 Combo를 제안합니다. 이 프레임워크는 다중 입력 및 다중 출력(MIMO) 모델의 도전을 해결하고자 하며, 이는 음성 신호와 캐릭터 안내 정보(예: 정체성 및 감정)를 사용하는 구성 요소 간의 복잡한 상호작용을 포함합니다.

- **Technical Details**: Combo는 두 가지 주요 디자인을 통해 MIMO 시스템의 복잡성을 줄입니다. 입력 단계에서는 고정된 정체성 및 중립 감정에 대한 데이터를 이용해 모델을 사전 훈련(pre-train)하며, 커스터마이징 조건의 주입은 미세 조정(fine-tuning) 단계에 지연시킵니다. 출력 단계에서는 DU-Trans라는 변환기 디자인을 사용하여, 얼굴 표정과 신체 움직임의 개별 특징을 학습하고 이를 결합하여 조화로운 모션을 생성합니다. X-Adapter는 파라미터 효율적인 미세 조정을 가능하게 합니다.

- **Performance Highlights**: Combo는 BEAT2 및 SHOW 데이터셋에서 높은 품질의 모션을 생성하고, 정체성 및 감정 변환에서도 효율적임을 입증했습니다. 특히 FMD 지표에서 우수한 성능을 보여주며, 사전 훈련된 모델의 미세 조정(약 5%의 시간)만으로도 기존 방법보다 뛰어난 결과를 얻었습니다.



### OU-CoViT: Copula-Enhanced Bi-Channel Multi-Task Vision Transformers with Dual Adaptation for OU-UWF Images (https://arxiv.org/abs/2408.09395)
- **What's New**: 본 논문에서는 근시(myopia) 검사를 위한 새로운 접근 방식을 제안합니다. 이는 최첨단 초광시야(UWF, ultra-widefield) 안저(NWF) 이미징과 여러 이산(디스크리트) 및 연속(컨티뉴어스) 임상 점수의 조합 모델링을 포함합니다. 본 연구의 주안점은 "양안 비대칭성(interocular asymmetries)" 개념을 적용한 이중 채널(bi-channel) 모델을 활용하여 고상관 및 비대칭 정보를 통합하는 것입니다.

- **Technical Details**: 제안된 OU-CoViT 모델은 다음과 같은 세 가지 주요 혁신을 포함합니다: 1) 클로즈드 형태의 새로운 Copula Loss를 개발하여 다차원 혼합 분류-회귀 작업에서 조건부 의존 구조를 캡처합니다. 2) 이중 적응 및 공유 백본을 사용하여 고상관 및 heterogeneity(다형성)를 동시에 모델링하는 새로운 이중 채널 아키텍처입니다. 3) 작은 의료 데이터셋에 대한 ViT(비전 변환기, Vision Transformers) 모델의 적응을 용이하게 하여 과적합(overfitting) 문제를 해결합니다.

- **Performance Highlights**: OU-CoViT 모델은 단일 채널 기반 선행 모델에 비해 예측 성능을 크게 향상시키는 것을 실험적으로 입증했습니다. 제안된 구조는 다양한 ViT 변형 및 대형 딥러닝 모델에 쉽게 확장될 수 있으며, 여러 임상 영역에서 AI 지원 진단 및 치료 결정을 발전시킬 수 있는 가능성을 열어줍니다.



### FD2Talk: Towards Generalized Talking Head Generation with Facial Decoupled Diffusion Mod (https://arxiv.org/abs/2408.09384)
Comments:
          Accepted by ACM Multimedia 2024

- **What's New**: 본 논문에서는 Talking head generation을 위한 Facial Decoupled Diffusion 모델, FD2Talk을 제안합니다. 기존의 GAN 기반 및 회귀 모델에서 발생하는 여러 문제를 해결하고, diffusion 모델의 장점을 활용하여 고품질의 다양한 결과를 생성합니다.

- **Technical Details**: FD2Talk은 복잡한 얼굴 세부정보를 모션(motion)과 외관(appearance)으로 분리하는 다단계(multi-stage) 프레임워크입니다. 첫 번째 단계에서 Diffusion Transformer를 통해 원시 오디오로부터 모션 계수를 예측하고, 두 번째 단계에서 참조 이미지를 인코딩하여 외관 텍스처를 캡처합니다. 이후 이 정보를 Diffusion UNet의 조건으로 사용하여 고품질의 프레임을 생성합니다.

- **Performance Highlights**: FD2Talk은 많은 실험을 통해 이전 방법들보다 이미지 품질 향상과 더 정확하고 다양한 결과를 생성하는 데 성공했습니다. 특히 머리 자세 모델링을 포함하여, 이전 방법들에 비해 훨씬 더 다양한 결과를 만들어냅니다.



### Detecting the Undetectable: Combining Kolmogorov-Arnold Networks and MLP for AI-Generated Image Detection (https://arxiv.org/abs/2408.09371)
Comments:
          8 Pages, IEEE Transactions

- **What's New**: 이번 논문은 최신 생성 AI 모델에서 생성된 이미지를 효과적으로 식별하기 위한 새로운 감지 프레임워크를 제시합니다. 특히 DALL-E 3, MidJourney, Stable Diffusion 3의 이미지를 포함한 포괄적인 데이터셋을 통해 다양한 조건에서 실제 이미지와 AI 생성 이미지를 구분하는 데 중점을 두고 있습니다.

- **Technical Details**: 이 연구에서는 전통적인 Multilayer Perceptron (MLP)와 의미론적 이미지 임베딩을 결합한 분류 시스템을 발전시켰습니다. 더불어 Kolmogorov-Arnold Networks (KAN)와 MLP를 결합한 하이브리드 아키텍처를 제안하여 AI 생성 이미지의 복잡한 패턴을 분석할 수 있습니다.

- **Performance Highlights**: 제안한 하이브리드 KAN MLP 모델은 세 가지 out-of-distribution (OOD) 데이터셋에서 표준 MLP 모델을 지속적으로 초과하는 성능을 보였으며, F1 점수는 각각 0.94, 0.94, 0.91로 나타났습니다. 이는 AI 생성 이미지와 실제 이미지를 구분하는 데 있어 우수한 성능과 내구성을 입증하는 결과입니다.



### Angle of Arrival Estimation with Transformer: A Sparse and Gridless Method with Zero-Shot Capability (https://arxiv.org/abs/2408.09362)
Comments:
          8 pages, 8 figures

- **What's New**: 본 연구에서는 자동차 레이더의 각도 추정(Angle of Arrival, AOA) 알고리즘인 AAETR(Angle of Arrival Estimation with TRansformer)를 소개합니다. AAETR은 깊은 학습 방식을 기반으로 하여 기존의 초해상도(AOA) 알고리즘에 비해 우수한 성능을 발휘하며, 특히 계산 비용을 효과적으로 줄이는 구조를 가지고 있습니다.

- **Technical Details**: AAETR은 레이더 신호를 처리하기 위해 설계된 변환기(transformer) 기반의 아키텍처로, 인코더와 디코더로 구성되어 있습니다. 데이터의 실시간 처리와 고정밀도를 유지하면서, 계산 비용을 낮출 수 있는 그리드가 없는(gridless) 각도 추정 기능을 제공합니다. 이 모델은 적은 수의 조정 가능한 하이퍼 파라미터를 요구하며, end-to-end 방식으로 훈련이 가능합니다.

- **Performance Highlights**: AAETR은 실제 데이터셋을 평가할 때, 합성 데이터셋에서 학습한 후에도 뛰어난 제로샷(zero-shot) 일반화 성능을 보여주었습니다. 아울러, 기존의 알고리즘과 비교하여 실시간 환경에서도 잘 적응하고 높은 정확도를 유지하며, 레이더 신호 처리의 새로운 가능성을 열어줄 것으로 기대됩니다.



### Panorama Tomosynthesis from Head CBCT with Simulated Projection Geometry (https://arxiv.org/abs/2408.09358)
Comments:
          12 pages, 6 figures, 1 table, Journal submission planned

- **What's New**: 이 논문에서는 Cone Beam Computed Tomography (CBCT) 데이터를 이용하여 효과적으로 Panoramic X-ray 이미지를 합성하는 새로운 방법을 제안합니다. 기존의 방법들과는 달리, 이 방법은 환자의 치아가 없거나 금속 임플란트가 있는 경우에도 높은 품질의 이미지를 생성할 수 있습니다.

- **Technical Details**: 제안된 방법은 변동하는 회전 중심과 반타원형 경로를 따르는 시뮬레이션된 프로젝션 기하학을 정의합니다. 첫 번째 단계에서는 환자의 머리 기울기를 수정하여 턱 위치를 감지하며, 그 후에 상악과 하악을 포함하는 초점 영역을 정의하여 최종 Panoramic 프로젝션에 기여합니다. 그런 다음 여러 X-ray 프로젝션을 생성하여 최종 X-ray 이미지를 만듭니다.

- **Performance Highlights**: 연구 결과, 제안된 방법은 종래의 방법보다 더 나은 품질의 Panoramic X-ray 이미지를 생성하며, 환자의 턱 형태나 위치에 관계없이 효과적으로 적용할 수 있음이 입증되었습니다.



### Joint Temporal Pooling for Improving Skeleton-based Action Recognition (https://arxiv.org/abs/2408.09356)
- **What's New**: 이 논문에서는 Joint Motion Adaptive Temporal Pooling(JMAP)라는 새로운 메소드를 제안하여 스켈레톤 기반의 인간 행동 인식에서의 시공간 관계를 개선하고자 합니다. 구체적으로는, 행동 시퀀스에서 중요한 정보를 담고 있는 프레임을 효과적으로 선택하여 기존의 평범한 풀링 방법의 한계를 극복하려고 합니다.

- **Technical Details**: JMAP는 프레임 단위 풀링과 관절 단위 풀링의 두 가지 변형을 갖추고 있습니다. 이는 관절의 동작 강도를 기반으로 하여 조정된 시공간 풀링 윈도우를 정의하고, 이를 통해 인식 성능을 향상시키도록 설계되었습니다. JMAP는 다양한 행동 인식 모델에 통합할 수 있으며, NTU RGB+D 120 및 PKU-MMD 데이터셋에서 실험을 통해 유효성을 검증했습니다.

- **Performance Highlights**: 제안한 JMAP는 기존 방법들과 비교하여 현저한 성능 향상을 보였으며, 스켈레톤 기반의 행동 인식 시스템의 정확도를 개선하는데 기여하였습니다.



### Boundary-Recovering Network for Temporal Action Detection (https://arxiv.org/abs/2408.09354)
Comments:
          Submitted to Pattern Recognition Journal

- **What's New**: 이 논문은 Temporal Action Detection (TAD)에서 'vanishing boundary problem'을 해결하기 위한 새로운 아키텍처인 Boundary-Recovering Network (BRN)을 제안합니다.

- **Technical Details**: BRN은 scale-time features를 도입하여 다중 스케일 특징에 대한 경계 패턴의 회복을 가능하게 합니다. BRN은 스케일-타임 블록을 사용하여 다기준 레벨 간의 특징을 교환하고, 이를 통해 경계 정보의 손실을 줄입니다. 이 구조는 scale convolutions와 time convolutions를 포함하여 특징을 효과적으로 집계합니다.

- **Performance Highlights**: BRN은 ActivityNet-v1.3과 THUMOS14라는 두 가지 도전적인 벤치마크에서 기존의 최신 기술을 초과하여 vanishing boundary problem의 정도를 현저히 줄였습니다.



### Hyperstroke: A Novel High-quality Stroke Representation for Assistive Artistic Drawing (https://arxiv.org/abs/2408.09348)
Comments:
          11 pages, 10 figures

- **What's New**: 이 논문에서는 예술가에게 지능적인 가이드를 제공하는 보조 드로잉 시스템을 위한 새로운 획 표현 방식인 'hyperstroke'를 소개합니다. 기존 솔루션들은 복잡한 획의 세부 사항을 효과적으로 모델링하지 못하고, 드로잉의 시간적 요소를 적절히 다루지 못했습니다.

- **Technical Details**: hyperstroke는 RGB 외관과 알파 채널 불투명도를 포함한 세밀한 획의 세부 사항을 포착하기 위해 설계된 획 표현 방식입니다. 이 방식은 실제 드로잉 비디오에서 축약된 토큰화된 획 표현을 학습하기 위해 벡터 양자화(Vector Quantization) 접근법을 사용하며, 변환기 기반 아키텍처(transformer-based architecture)를 통해 보조 드로잉을 모델링합니다.

- **Performance Highlights**: hyperstroke 디자인의 효율성을 입증하는 실험 결과들이 제시되며, 이는 예술적 드로잉의 미세한 세부 사항을 예측하면서 점진적 드로잉의 가능성을 보여줍니다.



### S^3D-NeRF: Single-Shot Speech-Driven Neural Radiance Field for High Fidelity Talking Head Synthesis (https://arxiv.org/abs/2408.09347)
Comments:
          ECCV 2024

- **What's New**: 본 논문에서는 S^3D-NeRF(싱글 샷 음성 기반 신경 복사장) 방법을 제안하여 단 하나의 이미지와 오디오 시퀀스를 사용하여 고화질의 말하는 얼굴 영상을 합성할 수 있는 새로운 접근법을 소개합니다.

- **Technical Details**: S^3D-NeRF는 세 가지 주요 문제를 해결합니다: 1) 각 신원의 대표적인 외관 특징 학습, 2) 오디오에 기반한 얼굴 부위 운동 모델링, 3) 입술 영역의 시간적 일관성 유지. 이를 위해 계층적 얼굴 외관 인코더(Hierarchical Facial Appearance Encoder)와 교차 변형 분야(Cross-modal Facial Deformation Field)를 도입하였습니다. 또한, 입술 동기화 분산기(lip-sync discriminator)를 사용하여 오디오-비주얼 시퀀스의 비일치성을 벌주고 시간적 일관성을 향상시키는 방법을 사용합니다.

- **Performance Highlights**: 광범위한 실험을 통해 S^3D-NeRF가 이전 기술들보다 비디오 충실도(video fidelity) 및 오디오-입술 동기화(audio-lip synchronization) 모두에서 뛰어난 성능을 보여줍니다.



### Elite360M: Efficient 360 Multi-task Learning via Bi-projection Fusion and Cross-task Collaboration (https://arxiv.org/abs/2408.09336)
Comments:
          15 pages main paper

- **What's New**: 이번 연구에서는 3D 구조(예: 깊이 및 표면 법선)와 의미 정보(semantic information)를 동시에 추론할 수 있는 새로운 다중 과제 학습 프레임워크인 Elite360M을 제안합니다. 이 모델은 기하학(geometry)과 의미(semantics) 간의 상호 관계를 탐구하여 보다 효과적으로 처리할 수 있도록 설계되었습니다.

- **Technical Details**: Elite360M은 고급 글로벌 인식을 통해 기하학 및 의미 정보를 동시 추론할 수 있는 커다란 가능성을 가지고 있습니다. 이 모델은 Bi-projection Bi-attention Fusion(B2F) 모듈과 Cross-task Collaboration(CoCo) 모듈 두 가지 주요 구성 요소로 구성되어 있으며, 비왜곡(non-distortion) 및 공간적으로 연속적인 아이코사헤드론 투영(icosahedron projection) 점들을 사용하여 ERP(EquiRectangular Projection)를 보완합니다.

- **Performance Highlights**: Elite360M은 Matterport3D 및 Structured3D와 같은 두 개의 대규모 다중 과제 장면 이해 벤치마크에서 평가되었습니다. 이 방법은 낮은 계산 비용으로 여러 인기 있는 다중 과제 학습 기준선보다 뛰어난 성능을 보였으며, 나아가 더 적은 매개변수로 단일 과제 학습 방법에 필적하는 성능을 달성했습니다.



### YOLOv1 to YOLOv10: The fastest and most accurate real-time object detection systems (https://arxiv.org/abs/2408.09332)
Comments:
          13 pages, 14 figures

- **What's New**: YOLO 시리즈의 최신 기술적 특성을 다시 검토하고, 이 시리즈가 실시간 컴퓨터 비전 연구에 미친 영향과 발전을 다룬 포괄적인 리뷰이다.

- **Technical Details**: YOLO는 전통적인 두 단계의 객체 탐지 방법을 버리고, 통합된 일 단계의 객체 탐지 방법을 제안한다. 이 기법은 객체를 픽셀 그리드에서 직접 예측하는 방식을 사용하여 높은 속도와 정확성을 구현했다. YOLOv1부터 YOLOv4까지의 발전을 살펴보며, 각각의 방법론 및 특징을 비교한다.

- **Performance Highlights**: YOLO 시리즈는 두 단계 탐지 방법보다 우수한 정확도를 달성하며, 특히 산업 및 연구 분야에서 실시간 객체 분석을 위한 가장 선호되는 방법이 되었다. 스케일된 YOLOv4는 일반 객체 탐지에서의 정확도를 극대화함으로써 YOLO 시리즈 연구의 새로운 발전 방향을 제시하였다.



### Multi-Camera Multi-Person Association using Transformer-Based Dense Pixel Correspondence Estimation and Detection-Based Masking (https://arxiv.org/abs/2408.09295)
Comments:
          5 pages, 6 figures

- **What's New**: 이번 연구는 다중 카메라 간의 객체 및 개인 식별을 위한 새로운 Multi-camera Association (MCA) 알고리즘을 소개합니다. 이 알고리즘은 Transformer 기반 아키텍처와Detection-based masking을 활용하여 밀집 픽셀 일치를 추정합니다.

- **Technical Details**: 알고리즘은 각 카메라 뷰에서 detections 간의 일치 확률을 나타내는 affinity matrix를 생성합니다. 그런 다음 Hungarian algorithm을 적용하여 최적의 assignment matrix를 생성하여 예측된 연관성을 계산합니다.

- **Performance Highlights**: WILDTRACK Seven-Camera HD Dataset에서 평가한 결과, 알고리즘은 유사한 관점에서 장면을 관찰하는 카메라 쌍에서 보행자 연결을 우수하게 수행하였으나, 거리나 각도가 크게 다른 카메라 쌍에서는 개선의 여지가 여전히 큽니다.



### Adaptify: A Refined Adaptation Scheme for Frame Classification in Atrophic Gastritis Videos (https://arxiv.org/abs/2408.09261)
Comments:
          ISBI 2024 Proceeding

- **What's New**: 이 논문에서는 gastric cancer(위암) 위험을 증가시키는 요인인 atrophic gastritis(위축성 위염)의 정확한 탐지를 지원하기 위해 Adaptify라는 새로운 Adaptation scheme(적응 체계)를 제안합니다. 이 방법은 모델이 자신의 분류 결정에서 지식을 습득하는 방식을 통합하여 output의 안정성과 일관성을 개선합니다.

- **Technical Details**: Adaptify는 고정된 주 네트워크와 지속적으로 매개변수를 업데이트하는 보조 네트워크를 통해 작동합니다. 이 방법은 여러 이전 프레임의 분류 출력을 특정 가중치로 결합하여 현재 프레임의 분류 점수와 통합합니다. 이를 통해 전체 이미지를 기반으로 더 강력하고 일관된 분류 결과를 도출하는 것을 목표로 합니다.

- **Performance Highlights**: 모든 조합을 평가한 결과, 기존의 단일 모델보다 false positive(위양성)와 false negative(위음성) 예측을 크게 줄이고, 안정적이고 일관된 출력 성능을 보이는 것으로 나타났습니다. 이 연구는 복잡한 의료 영상 분류 문제에서 deep learning(딥러닝)의 가능성을 보여줍니다.



### MagicID: Flexible ID Fidelity Generation System (https://arxiv.org/abs/2408.09248)
- **What's New**: 새로운 연구 MagicID는 Multi-Mode Fusion Training Strategy (MMF)와 DDIM Inversion 기반 ID Restoration inference framework (DIIR)를 결합하여 포트레이트 생성에서 제어 가능성과 충실도를 향상시키는 혁신적 접근 방식을 제안합니다.

- **Technical Details**: MagicID는 IDZoom이라는 백만 수준의 다중 모드 데이터 세트를 기반으로 하여 개발되었습니다. MMF는 얼굴 ID, 인체 골격, 얼굴 랜드마크 및 텍스트 프롬프트와 같은 다중 모드 데이터를 사용하여 훈련하며, DIIR는 저해상도 얼굴에서 아케이드를 수정하여 포토 품질을 향상시킵니다. 이는 배경을 유지하면서 얼굴 세부 사항 복원을 보장합니다.

- **Performance Highlights**: 실험 결과 MagicID는 주관적 및 객관적 메트릭 모두에서 관찰할 수 있는 유의미한 장점을 보여주며, 다인칭 시나리오에서 제어 가능한 생성을 성공적으로 달성했습니다.



### Re-boosting Self-Collaboration Parallel Prompt GAN for Unsupervised Image Restoration (https://arxiv.org/abs/2408.09241)
Comments:
          This paper is an extended and revised version of our previous work "Unsupervised Image Denoising in Real-World Scenarios via Self-Collaboration Parallel Generative Adversarial Branches"(this https URL)

- **What's New**: 본 연구에서는 기존의 복원 모델을 위한 Self-Collaboration (SC) 전략을 제안합니다. 이 전략은 이전 단계의 정보를 피드백으로 활용하여 후속 단계를 안내하며, 복원 모델의 성능을 획기적으로 개선합니다.

- **Technical Details**: SC 전략은 Prompt Learning (PL) 모듈과 Restorer (Res)로 구성됩니다. PL 모듈은 이전의 덜 강력한 Restorer를 더 강력한 Res로 반복적으로 교체하며, 이를 통해 더 나은 pseudo-degraded/clean 이미지 쌍을 생성합니다. 이 과정에서 추가적인 파라미터를 추가하지 않고도 1.5 dB 이상의 성능 향상을 달성합니다. 또한, Self-Ensemble (SE) 전략과 SC 전략의 장단점을 비교하며, Re-boosting 모듈인 Reb-SC를 통해 SC 전략의 성능을 더욱 향상시킵니다.

- **Performance Highlights**: 다양한 복원 작업에 대한 실험 결과, 제안된 RSCP2GAN 모델이 기존의 최첨단 비지도 복원 방법들에 비해 우수한 성능을 보이며, deraining 및 denoising 작업에서 각각 약 0.3 dB의 추가 성능 개선을 나타냈습니다.



### RepControlNet: ControlNet Reparameterization (https://arxiv.org/abs/2408.09240)
- **What's New**: Diffusion models의 확산적 응용이 증가함에 따라, RepControlNet이라는 새로운 modal reparameterization 방법을 제안하여 고성능의 조건부 생성(conditional generation)을 가능하게 한다. 이 방법은 계산 자원을 추가적으로 소모하지 않으면서도 기존의 ControlNet보다 더 효율적으로 작동한다.

- **Technical Details**: RepControlNet은 원래의 diffusion 모델의 CNN과 MLP 가중치를 복사하여 modal network를 생성하고, 학습 과정에서 이 네트워크의 파라미터만을 최적화한다. 추론 과정에서는 modal network의 가중치를 재파라미터화하여, 원래 diffusion 모델보다 부하를 최소화하면서도 개선된 성능을 제공한다.

- **Performance Highlights**: Stable Diffusion 1.5(SD1.5)와 Stable Diffusion XL(SDXL)에서의 실험을 통해 RepControlNet의 효과성과 효율성을 검증하였다. RepControlNet은 ControlNet과 비교했을 때, 계산량을 반으로 늘리지 않으면서도 거의 동일한 조건부 생성 능력을 유지한다.



### Flatten: Video Action Recognition is an Image Classification task (https://arxiv.org/abs/2408.09220)
Comments:
          13pages, 6figures

- **What's New**: 본 논문에서는 Flatten이라는 새로운 비디오 표현 아키텍처를 소개합니다. Flatten은 비디오 행동 인식 작업을 이미지 분류 작업으로 변환하여 비디오 이해의 복잡성을 감소시키는 모듈입니다. 이 아키텍처는 기존의 이미지 이해 네트워크에 손쉽게 통합될 수 있습니다.

- **Technical Details**: Flatten은 3D 시공간(spatiotemporal) 데이터를 평면화하는(row-major transform) 특정 작업을 통해 2D 공간정보로 변환합니다. 이후 일반적인 이미지 이해 모델을 사용하여 시간적 동적 및 공간적 의미 정보를 포착합니다. 이 방식으로 비디오 행동 인식을 효율적으로 수행할 수 있습니다.

- **Performance Highlights**: Kinetics-400, Something-Something v2, HMDB-51과 같은 일반적인 데이터 세트에서 대규모 실험을 통해 Flatten을 적용하면 기존 모델 대비 성능 향상이 크게 나타났습니다. 특히, ‘Uniformer(2D)-S + Flatten’ 조합은 Kinetics400 데이터셋에서 81.1이라는 최고의 인식 성능을 기록했습니다.



### DRL-Based Resource Allocation for Motion Blur Resistant Federated Self-Supervised Learning in IoV (https://arxiv.org/abs/2408.09194)
Comments:
          This paper has been submitted to IEEE Journal. The source code has been released at: this https URL

- **What's New**: 본 논문에서는 Internet of Vehicles (IoV)에서 개인 정보를 보호하는 Federated Self-Supervised Learning (FSSL) 방법론을 제안한다. 기존의 Momentum Contrast (MoCo) 대신에 Simplified Contrast (SimCo)를 사용하여 개인 정보 유출 문제를 해결하고, motion blur 문제에 강한 새로운 FSSL 방법 (BFSSL)을 도입하였다. 또한, Deep Reinforcement Learning (DRL) 기반의 자원 할당 기법인 DRL-BFSSL을 제안하여 에너지 소비와 지연을 줄일 수 있도록 하였다.

- **Technical Details**: FSSL은 Self-Supervised Learning (SSL) 기법을 기반으로 하여 라벨이 없는 데이터를 이용해 차량들이 로컬 모델을 학습할 수 있게 한다. SimCo는 dual temperature를 활용하여 negative sample의 딕셔너리 사용 없이 sample distribution을 제어하여 개인정보를 더욱 철저히 보호한다. BFSSL은 motion blur의 영향을 고려하여 모델 집계를 진행하며, DRL-BFSSL은 CPU 주파수와 전송 전력을 최적화하여 에너지 소비를 최소화하고 지연을 줄인다.

- **Performance Highlights**: 시뮬레이션 결과, 제안한 BFSSL 방법이 motion blur에 효과적으로 저항하며, DRL-BFSSL 알고리즘이 에너지 소비와 지연을 모두 최소화하는 최적의 자원 할당을 달성함을 보여준다.



### GSLAMOT: A Tracklet and Query Graph-based Simultaneous Locating, Mapping, and Multiple Object Tracking System (https://arxiv.org/abs/2408.09191)
Comments:
          11 pages, 9 figures, ACM MM 2024

- **What's New**: 본 논문은 GSLAMOT라는 새로운 프레임워크를 제안하여 모바일 객체를 동시에 위치추적(mapping), 맵화(mapping) 및 추적(tracking)하는 문제를 해결합니다. 이 시스템은 카메라와 LiDAR의 멀티모달 정보(multi-modal information)를 활용하여 동적 장면을 처리합니다.

- **Technical Details**: GSLAMOT는 세 가지 중요한 구성 요소로 이루어져 있습니다: (1) 정적 환경을 나타내는 시맨틱 맵(semantic map), (2) 에고 에이전트의 궤적(trajectory), (3) 감지된 이동 객체의 3D 포스를 추적 및 예측하기 위한 온라인 유지 Tracklet Graph(TG). Query Graph(QG)는 개별 프레임에서 객체 감지로 생성되어 TG의 업데이트를 실시간으로 처리합니다. 또한, MSGA(Multi-criteria Star Graph Association) 및 OGO(Object-centric Graph Optimization)라는 새로운 알고리즘을 이용하여 객체의 정확한 매칭과 최적화를 수행합니다.

- **Performance Highlights**: GSLAMOT는 KITTI, Waymo, Traffic Congestion 데이터셋에서 실험을 통해 뛰어난 성능을 입증하였으며, 복잡한 동적 환경에서도 정확한 객체 추적과 SLAM을 동시에 수행할 수 있음을 보여줍니다. 특히, GSLAMOT는 기존의 최첨단 방법들보다 더 뛰어난 성능을 달성하였습니다.



### PADetBench: Towards Benchmarking Physical Attacks against Object Detection (https://arxiv.org/abs/2408.09181)
- **What's New**: 이 논문은 물리적 공격에 대한 공정하고 철저한 벤치마크를 제안하며, 이는 객체 탐지 모델을 평가하기 위해 이상적인 조건 하에서 다양한 물리적 동역학을 고려하여 평가는 새로운 접근법을 제공합니다.

- **Technical Details**: CARLA라는 자율주행 시뮬레이터를 활용하여 물리적 공격을 일관되게 평가하고, 20가지 물리적 공격 방법, 48개의 객체 탐지기, 다양한 물리적 동역학 및 평가 지표를 포함한 포괄적인 파이프라인을 구축했습니다.

- **Performance Highlights**: 논문에서는 8064개의 평가 그룹을 수행하였고, 이는 다양한 물리적 동역학과 평가 지표 하에서 물리적 공격의 효능을 분석하여 의의 있는 발견을 강조하고, 향후 연구 방향에 대해 논의하고 있습니다.



### MambaTrack: A Simple Baseline for Multiple Object Tracking with State Space Mod (https://arxiv.org/abs/2408.09178)
Comments:
          Accepted by ACM Multimedia 2024

- **What's New**: 이번 연구에서 제안된 Mamba moTion Predictor (MTP)는 기존의 Kalman Filter 기반 모션 예측의 한계를 극복하기 위해 데이터 기반의 모션 예측 방법을 탐구합니다. 특히, 복잡한 움직임을 가지는 객체들, 예를 들어 댄서와 운동선수의 모션 패턴 학습에 중점을 두고 있습니다.

- **Technical Details**: MTP는 객체의 과거 운동 정보를 입력으로 받아 bi-Mamba 인코딩 레이어를 통해 모션 정보를 인코딩하고 다음 움직임을 예측합니다. 이 과정에서 Intersection-over-Union (IoU) 값을 기준으로 데이터 연관성을 수행하며, occlusion이 발생한 경우에는 tracklet patching 모듈을 사용하여 잃어버린 경로를 다시 설정합니다.

- **Performance Highlights**: 제안된 트래커 MambaTrack은 DanceTrack과 SportsMOT 벤치마크에서 상태-of-the-art 성능을 보이며, 복잡한 움직임과 심각한 occlusion 상황에서도 우수한 데이터 연관 문제 해결 능력을 보여줍니다.



### Zero-Shot Object-Centric Representation Learning (https://arxiv.org/abs/2408.09162)
- **What's New**: 이번 연구에서는 오브젝트 중심( object-centric ) 표현 학습을 제로샷( zero-shot ) 일반화 관점에서 분석합니다. 이를 위해 다양한 합성 및 실제 세계 데이터셋으로 구성된 8개의 벤치마크를 도입하여 현재 모델의 제로샷 전이 능력을 평가합니다.

- **Technical Details**: 기존의 오브젝트 중심 학습 방법들은 정적 pre-trained encoder를 사용하지만, 본 논문에서는 이를 파인튜닝(fine-tuning)하여 오브젝트 발견(object discovery) 작업에 적합하게 조정합니다. 또한, 다양한 데이터로 학습한 모델이 제로샷 전이 성능을 높인다는 것을 발견했습니다.

- **Performance Highlights**: 제안하는 방법은 CoCo 데이터셋에서의 실제 오브젝트 중심 학습에 대해 새로운 최첨단 성능을 달성하였으며, 여러 다양한 데이터셋에 대해 제로샷 전이를 성공적으로 수행했습니다.



### DSReLU: A Novel Dynamic Slope Function for Superior Model Training (https://arxiv.org/abs/2408.09156)
Comments:
          Under peer review at ICPR, 2024

- **What's New**: 이 연구는 훈련 과정 전반에 걸쳐 동적으로 조정되는 경사를 가진 새로운 활성화 함수 DSReLU(Dynamic Slope Changing Rectified Linear Unit)를 소개합니다. 이 활성화 함수는 딥 뉴럴 네트워크의 적응성과 성능 향상을 목표로 하며, 특히 컴퓨터 비전 작업에서 성능을 개선하는 데 초점을 맞추고 있습니다.

- **Technical Details**: DSReLU는 훈련 시 발생하는 경사를 동적으로 변화시켜 학습의 초기 단계에서는 가파른 경사로 빠른 학습을 촉진하고, 후반부에는 경사를 감소시켜 안정성을 제공합니다. 이를 통해 소실 그래디언트 문제(vanishing gradient problem)와 과적합(overfitting)을 해결할 수 있습니다.

- **Performance Highlights**: DSReLU는 Mini-ImageNet, CIFAR-100, MIT-BIH 데이터셋에서 기존 최첨단 활성화 함수들과 비교하여 우수한 성능을 보여주었으며, 정확도(accuracy), F1-score 및 AUC와 같은 분류 지표에서 향상된 결과를 도출했습니다. 또한, DSReLU는 외부 정규화 기법을 사용하지 않고도 훈련 중 과적합을 줄이는 데 효과적입니다.



### Are CLIP features all you need for Universal Synthetic Image Origin Attribution? (https://arxiv.org/abs/2408.09153)
Comments:
          Accepted at ECCV 2024 TWYN workshop

- **What's New**: 본 연구에서는 Diffusion Models에서 생성된 합성 이미지의 Open-Set 기원 속성을 추적하는 새로운 프레임워크를 제안합니다. 이 프레임워크는 대규모 학습된 기초 모델에서 추출된 기능을 사용하여 다양한 생성 모델에 의해 생성된 이미지의 출처를 식별하는 데 효과적입니다.

- **Technical Details**: 기존의 포렌식 기법은 주로 GAN이 생성한 이미지에 한정되어 있으며, 이를 위해 주파수 기반의 '지문' 기능에 의존합니다. 반면, 본 연구에서는 이미지의 저수준 주파수 기반 표현 대신 고수준 이미지 기능을 활용하여 Open-Set 시나리오에서 합성 이미지 기원 속성을 효과적으로 추적할 수 있음을 보여줍니다. 사용된 기초 모델로는 DINOv2와 CLIP이 있으며, 이들은 방대한 데이터셋으로 학습되었습니다.

- **Performance Highlights**: 본 연구 방법론은 낮은 데이터 상황에서도 뛰어난 속성 성능을 보였으며, 다양한 아키텍처에서 얻은 이미지에 대해 더 나은 일반화 성능을 보였습니다. 기존의 방법들보다 더 효과적이며, 다양한 모델 패밀리의 데이터에서 합성 이미지를 탐지 및 속성을 추적하는 데 성공하였습니다.



### Realistic Extreme Image Rescaling via Generative Latent Space Learning (https://arxiv.org/abs/2408.09151)
- **What's New**:  본 연구에서는 극단적인 이미지 리스케일링을 위한 새로운 프레임워크인 Latent Space Based Image Rescaling (LSBIR)을 제안합니다. 이 방법은 기존의 픽셀 손실을 전혀 적용하지 않고도 재구성된 결과의 전반적인 의미와 구조를 유지하는 데 뛰어난 성능을 보여줍니다.

- **Technical Details**:  LSBIR은 두 단계의 훈련 전략을 채택하고 있습니다. 첫 번째 단계에서는 HR 이미지의 잠재적 특성을 LR 이미지와 쌍방향으로 매핑하기 위한 준 가역적인 인코더-디코더 모델을 사용합니다. 두 번째 단계에서는 첫 번째 단계에서 재구성된 특징들을 사전 훈련된 확산 모델(pre-trained diffusion model)을 통해 더욱 신뢰할 수 있는 시각적 세부 사항으로 정제합니다. 이 과정에서 잠재 공간(latent space)에서의 리스케일링을 통해 더 많은 정보를 LR 이미지에 포함시킵니다.

- **Performance Highlights**:  LSBIR은 정량적 및 정성적 평가에서 기존 방법들보다 우수한 성능을 보입니다. 실험을 통해 LSBIR이 극단적인 리스케일링 작업에서 어떻게 더 나은 세부 사항과 의미론적 구조를 생성하는지를 입증하였습니다.



### SSNeRF: Sparse View Semi-supervised Neural Radiance Fields with Augmentation (https://arxiv.org/abs/2408.09144)
- **What's New**: 이 논문에서는 제한된 시점 이미지를 사용한 NeRF(Neural Radiance Fields)에서 발생하는 문제를 해결하기 위해 SSNeRF(스파스 뷰 반감 초감독 NeRF)라는 새로운 방법을 제안합니다. 기존 방법들이 깊이 맵과 같은 보조 정보를 필요로 하는 데 반해, 우리는 고신뢰성의 가짜 레이블을 제공하여 NeRF가 스파스 뷰 정보에 대한 강건성을 높이는 데 주력합니다.

- **Technical Details**: SSNeRF는 teacher-student(framework) 방식으로 구현되며, NeRF 모델에 점진적으로 심한 스파스 뷰 손실을 주면서 노이즈와 불완전한 정보를 인식하도록 학습합니다. Teacher NeRF는 새로운 뷰와 신뢰 점수를 생성하고, Student NeRF는 강화된 입력으로부터 학습하여 고신뢰도 가짜 레이블을 활용합니다. 다양한 방법으로 볼륨 렌더링 가중치에 노이즈를 주입하고, 취약한 레이어의 특징 맵에 영향을 주어 손실을 시뮬레이션합니다.

- **Performance Highlights**: 실험 결과, SSNeRF는 스파스 뷰 손실이 적고 선명한 이미지를 생성하는 능력을 보였습니다. 또한, 복잡한 데이터셋에서도 성능을 발휘하며, 렌더링된 비디오에서 깜박임 픽셀의 손실을 효과적으로 제거했습니다.



### StylePrompter: Enhancing Domain Generalization with Test-Time Style Priors (https://arxiv.org/abs/2408.09138)
- **What's New**: 본 연구에서는 기존의 도메인 일반화(Domain Generalization, DG) 방법의 한계를 극복하기 위해 스타일 프롬프트(style prompt)를 도입하여, 훈련된 모델을 동적으로 적응시키는 새로운 접근 방식을 제안합니다. 스타일 정보를 추출하는 스타일 프로퍼터를 훈련하여 입력 이미지의 스타일 정보를 추출하고, 이를 후보 카테고리 단어 앞에 배치하여 모델을 프롬프트합니다.

- **Technical Details**: 우리의 방법은 입력 이미지로부터 스타일 정보를 추출하기 위해 스타일 프로퍼터(style prompter)를 훈련시키며, 이는 이미지 특징을 입력받아 토큰 임베딩 공간에서 스타일 정보를 출력합니다. 우리는 스타일 토큰 임베딩 공간을 개방형으로 나누고 수작업으로 스타일 정규화(style regularization)를 적용하여 훈련된 스타일 프로퍼터가 알려지지 않은 도메인에서 샘플의 스타일 임베딩을 효과적으로 추출할 수 있도록 합니다. 이 과정에서, 비전-언어 모델(vision-language model)의 이미지-텍스트 매칭 능력을 활용하여 모델이 적응적으로 알려지지 않은 도메인 데이터를 처리할 수 있도록 지원합니다.

- **Performance Highlights**: 우리의 방법은 다양한 크기의 공공 데이터셋에서 최첨단(state-of-the-art) 성능을 입증하였으며, 여러 철저한 실험을 통해 효과성을 검증했습니다.



### Thin-Plate Spline-based Interpolation for Animation Line Inbetweening (https://arxiv.org/abs/2408.09131)
- **What's New**: 이 논문에서는 애니메이션 선화 중간 보간(inbetweening) 기술을 개선하여 특히 큰 움직임을 다루는 데 중점을 둡니다. 기존 방법들이 선 연결 끊김과 같은 문제를 자주 발생시키는 것에 반해, 본 연구는 Thin-Plate Spline(TPS) 기반의 변환을 사용하여 보다 정확한 거친 모션 추정을 수행합니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 모듈로 구성됩니다. 첫째, TPS 변환을 통해 대규모 움직임을 보다 정확히 모델링하는 거친 모션 추정 모듈을 구축합니다. 둘째, 이 거친 추정을 바탕으로 모션 세부 사항을 향상시키는 모션 정제 모듈을 설정하며, 최종 프레임 인터폴레이션을 위해 간단한 UNet 모델을 사용합니다. 개선된 Metric으로 Weighted Chamfer Distance(WCD)와 Earth Mover's Distance(EMD)를 제안하여 성능 평가를 수행합니다.

- **Performance Highlights**: 제안된 방법은 다양한 벤치마크 데이터셋에서 기존의 최첨단 기술들보다 우수한 성능을 발휘하며, 1, 5, 9개의 간격에 대한 고품질 보간 결과를 제공합니다. 또한, WCD와 EMD를 사용한 평가에서도 높은 일관성을 보여 눈에 띄는 성과를 입증하였습니다.



### Gaussian in the Dark: Real-Time View Synthesis From Inconsistent Dark Images Using Gaussian Splatting (https://arxiv.org/abs/2408.09130)
Comments:
          accepted by PG 2024

- **What's New**: 본 연구에서는 다크 환경에서의 다중 뷰 일관성 문제를 해결하기 위해 Gaussian-DK라는 새로운 방법을 제안합니다. 이 방법은 3D Gaussian Splatting 기술을 활용하여 다크 환경에서도 고품질의 렌더링을 생성할 수 있습니다.

- **Technical Details**: Gaussian-DK는 3D 공간에서 물리적 조도 필드를 표현하기 위해 비등방성 3D Gaussian을 사용합니다. 또한, 카메라의 이미지 프로세스 개선을 위해 카메라 응답 모듈을 설계하고, 노출 레벨을 조정하여 하이라이트 및 그림자 처리를 개선합니다. 이를 통해 반복적인 샘플링 과정을 줄이고 실시간 렌더링이 가능해집니다.

- **Performance Highlights**: 실험 결과, Gaussian-DK는 고질적인 플로터 또는 유령 아티팩트 없이 다크 환경에서도 높은 품질의 렌더링 결과를 제공하며, 기존의 방법들보다 현저히 향상된 성능을 보여줍니다. 또한, 우리는 새로운 벤치마크 데이터세트를 공개하였고, 이는 실제 다크 환경에서 수집된 12개의 장면으로 구성되어 있습니다.



### Barbie: Text to Barbie-Style 3D Avatars (https://arxiv.org/abs/2408.09126)
Comments:
          9 pages, 7 figures

- **What's New**: 최근 텍스트 기반의 3D 아바타 생성 기술에서 큰 발전이 있었습니다. 본 논문에서는 다양한 고품질 의상과 액세서리를 착용할 수 있는 새로운 프레임워크인 Barbie를 제안하고 있습니다. 이 모델은 전반적인 모델에 의존하는 대신 인체와 의상을 위한 분리된 모델을 활용하여 세밀한 요소 분리를 달성합니다.

- **Technical Details**: Barbie 프레임워크는 세분화된 표현을 생성하기 위해 의미적으로 정렬된 인체 사전 지식으로 초기화된 다양한 아바타 구성 요소를 사용합니다. 각 표현은 대응하는 전문가 텍스트-투-이미지(T2I) 모델에 의해 최적화되어 특정 영역에 맞는 형태와 질감을 제공합니다. 또한 인간 사전 진화 손실(human-prior evolving loss)과 템플릿 보존 손실(template-preserving loss)을 도입하여 다양성과 정렬성을 향상시킵니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, Barbie는 아바타 및 의상 생성에서 기존 방법들을 초월하는 성능을 보여주었으며, 뛰어난 형상, 질감 및 텍스트 설명 정렬을 갖추고 있습니다. 이로써 Barbie는 고품질의 의상 조합 및 애니메이션을 지원하는 데 큰 기여를 하고 있습니다.



### MaskBEV: Towards A Unified Framework for BEV Detection and Map Segmentation (https://arxiv.org/abs/2408.09122)
Comments:
          Accepted to ACM MM 2024

- **What's New**: 본 논문에서는 MaskBEV라는 새로운 다중 작업(Multi-Task) 학습 패러다임을 제안합니다. MaskBEV는 3D 객체 탐지(3D Object Detection)와 조감도(Bird's Eye View, BEV) 지도 분할을 통합하는 방식으로, 이는 기존의 독립적인 접근 방식과는 다른 점입니다.

- **Technical Details**: MaskBEV는 마스크된 주의를 기반으로 하는 Transformer 디코더를 활용하여 다양한 작업을 동시에 처리합니다. 공간 변조(spatial modulation) 및 장면 수준 컨텍스트 집합(Scene-level Context Aggregation) 전략을 도입하여 BEV 공간에서 상호 보완적인 정보를 최대한 활용합니다. 이 구조는 작업 간의 종속성을 자연스럽게 고려하여 학습 성능을 향상시킵니다.

- **Performance Highlights**: nuScenes 데이터셋에서 MaskBEV는 기존의 최첨단 3D 객체 탐지 방법에 비해 1.3 NDS 향상과 BEV 지도 분할에서 2.7 mIoU 향상을 달성하였습니다. 또한, inference 속도가 약간 앞서는 결과도 보여줍니다.



### LOID: Lane Occlusion Inpainting and Detection for Enhanced Autonomous Driving Systems (https://arxiv.org/abs/2408.09117)
Comments:
          8 pages, 6 figures and 4 tables

- **What's New**: 본 논문에서는 레인(선) 탐지 문제에서 occlusion(가림현상)으로 인한 성능 저하를 해결하기 위한 두 가지 혁신적인 접근 방식을 제안합니다. 첫 번째 방법은 aug-Segment라는 이름으로, CULanes 데이터셋의 훈련 데이터를 시뮬레이션된 occlusion으로 확장하여 훈련한 세그멘테이션 모델입니다. 두 번째 접근 방식인 LOID(Lane Occlusion Inpainting and Detection)는 인페인팅(복원) 기법을 이용해 가려진 도로 환경을 재구성합니다.

- **Technical Details**: 전통적인 레인 탐지 모델은 가림 현상에서 신뢰할 수 없는 탐지를 초래하는 경우가 많습니다. aug-Segment 모델은 CULanes 데이터셋에서 여러 SOTA 모델에 비해 12% 향상된 성능을 보이며, 두 번째 LOID 접근 방식은 BDDK100 및 CULanes 데이터셋에서 각각 20% 및 24% 향상을 달성하여, 각 접근 방식의 효과를 강조합니다. LOID는 동적 occlusion 탐지, lane 정보 복원 및 선명한 레인 라인 탐지를 위한 신경망 기반 파이프라인으로 구성됩니다.

- **Performance Highlights**: LOID와 aug-Segment는 여러 SOTA 모델 대비显著한 향상을 보였습니다. LOID는 안정적인 레인 탐지를 보장하며, 다양한 데이터셋에 대한 적응성과 견고성을 입증했습니다.



### GoodSAM++: Bridging Domain and Capacity Gaps via Segment Anything Model for Panoramic Semantic Segmentation (https://arxiv.org/abs/2408.09115)
Comments:
          15 pages, under review

- **What's New**: 이 논문은 GoodSAM++라는 새로운 프레임워크를 소개합니다. 이는 SAM(Segment Anything Model)의 강력한 zero-shot instance segmentation 능력을 이용하여 레이블이 없는 데이터를 통해 컴팩트한 파노라마 세멘틱 세그멘테이션 모델을 학습할 수 있도록 합니다.

- **Technical Details**: GoodSAM++는 SAM과 학생 모델의 도메인 및 용량 간의 격차를 해소하기 위해 Teacher Assistant(TA) 개념을 도입합니다. 이를 통해 TA는 SAM의 익숙하지 않은 데이터에서 의미 정보를 제공하며, Distortion-Aware Rectification(DARv2) 모듈과 Multi-level Knowledge Adaptation(MKA) 모듈을 통해 학습을 최적화합니다.

- **Performance Highlights**: GoodSAM++는 다양한 실험에서 기존 방법들보다 뛰어난 성능을 보여주며, 3.7백만 파라미터의 초경량 학생 모델이 SOTA(season of the art) 모델들과 유사한 성능을 발휘하는 것으로 나타났습니다. 또한, GoodSAM++는 열린 세계(open-world) 데이터에서도 높은 일반화 능력을 보여주었습니다.



### Locate Anything on Earth: Advancing Open-Vocabulary Object Detection for Remote Sensing Community (https://arxiv.org/abs/2408.09110)
Comments:
          10 pages, 5 figures

- **What's New**: 본 논문에서는 원거리 감지(OD)의 한계를 극복하기 위해 Locate Anything on Earth (LAE)라는 새로운 접근 방식을 제안하고, 100만 개 라벨이 달린 객체 데이터로 구성된 LAE-1M 데이터셋을 구축하였습니다. 이 데이터셋은 원거리 감지 분야에서 최초로 대규모 객체 감지 데이터를 제공합니다.

- **Technical Details**: LAE-1M 데이터셋은 비주얼-가이드 텍스트 프롬프트 학습(VisGT)과 동적 어휘 구성(DVC) 모듈을 포함하는 LAE-DINO 모델을 사용하여 훈련되었습니다. VisGT는 이미지와 텍스트 간의 관계를 개선하여 보다 효과적인 객체 감지를 달성합니다.

- **Performance Highlights**: LAE-1M 데이터셋은 기존 모델에 비해 개방형 시나리오에서 특히 성능이 크게 향상되었으며, LAE-DINO 모델이 최첨단 성능을 기록했습니다.



### HybridOcc: NeRF Enhanced Transformer-based Multi-Camera 3D Occupancy Prediction (https://arxiv.org/abs/2408.09104)
Comments:
          Accepted to IEEE RAL

- **What's New**: 본 논문에서는 HybridOcc라는 새로운 3D semantic scene completion (SSC) 방법을 제안합니다. 이 방법은 Transformer 기반의 하이브리드 3D 볼륨 쿼리 제안 방법을 활용하여 NeRF(Neural Radiance Fields) 표현을 통합하고, coarse-to-fine SSC 예측 프레임워크에서 정제됩니다.

- **Technical Details**: HybridOcc는 Transformer와 NeRF를 결합하여 문맥적 특성을 집계합니다. 여기서 Transformer는 여러 스케일을 포함하고, 2D에서 3D로의 변환을 위한 공간 크로스 어텐션을 사용합니다. NeRF 분기는 볼륨 렌더링을 통해 장면 점유율을 암시적으로 추론하고, RGB 대신 장면 깊이를 명시적으로 포착합니다. 또한, 점유 인식(ray sampling) 방법을 도입하여 SSC 작업을 향상시킵니다.

- **Performance Highlights**: nuScenes 및 SemanticKITTI 데이터 세트를 통해 진행된 많은 실험 결과, HybridOcc는 FB-Occ 및 VoxFormer와 같은 기존 깊이 예측 네트워크 기반 방법들보다 우수한 성능을 보였습니다.



### Depth-guided Texture Diffusion for Image Semantic Segmentation (https://arxiv.org/abs/2408.09097)
- **What's New**: 이 논문에서는 depth 정보와 RGB 이미지 간의 격차를 줄여서 더 정확한 semantic segmentation을 달성하는 새로운 Depth-guided Texture Diffusion 접근 방식을 소개합니다. 이 방법은 깊이 맵을 텍스쳐 이미지로 변환하고 이를 선택적으로 확산시킴으로써, 구조적 정보를 강화하여 객체 윤곽 추출을 더욱 정밀하게 만듭니다.

- **Technical Details**: 제안된 방법은 깊이 정보의 텍스쳐 세부 정보를 선택적으로 강조하여 깊이 맵과 2D 이미지 간의 호환성을 높입니다. 이 과정 동안 구조적 일관성을 보장하기 위해 `structural loss function`을 이용하여 객체 구조의 완전성을 유지합니다. 최종적으로, 텍스쳐가 정제된 깊이와 RGB 이미지를 통합하여 모델의 정확성을 개선합니다.

- **Performance Highlights**: 폭넓은 데이터셋을 대상으로 한 실험 결과, 제안된 방법인 Depth-guided Texture Diffusion이 기존의 기초 모델들보다 일관되게 우수한 성능을 보였으며, camouflaged object detection, salient object detection 및 실내 semantic segmentation 분야에서 새로운 최첨단 성과를 달성하였습니다.



### Segment Anything with Multiple Modalities (https://arxiv.org/abs/2408.09085)
Comments:
          Project page: this https URL

- **What's New**: 이번 연구에서는 기존의 Segment Anything Model (SAM)의 기능을 확장한 MM-SAM을 소개합니다. MM-SAM은 교차 모달 및 다중 모달 처리를 지원하여 LiDAR, 깊이 및 열 센서와 같은 다양한 센서 조합으로부터 강력하고 향상된 분할(segmentation) 성능을 제공합니다.

- **Technical Details**: MM-SAM은 두 가지 주요 설계 요소, 즉 비지도 교차 모달 전이(unsupervised cross-modal transfer, UCMT)와 약하게 감독되는 다중 모달 융합(weakly-supervised multi-modal fusion, WMMF)을 사용하여 센서 모달리티에 대한 라벨 효율적인 및 파라미터 효율적인 적응을 가능하게 합니다. UCMT는 모달별 패치 임베딩 및 파라미터 효율적인 튜닝을 통해, 각 센서의 특징을 효과적으로 추출할 수 있도록 돕습니다.

- **Performance Highlights**: MM-SAM은 SAM에 비해 여러 세션 및 데이터 모달리티에서 일관되게 우수한 성능을 보이며, 다중 센서 조합을 통한 유기적인 데이터 처리 및 향상된 강건성을 입증합니다.



### Linking Robustness and Generalization: A k* Distribution Analysis of Concept Clustering in Latent Space for Vision Models (https://arxiv.org/abs/2408.09065)
- **What's New**: 이 논문은 vision 모델의 latent space를 평가하기 위한 새로운 방법론을 제안하며, k* Distribution을 사용하여 각 개념의 학습된 latent space를 분석합니다. 이를 통해 기존의 간접 평가 방식과는 다른 접근을 제공합니다.

- **Technical Details**: k* Distribution 방법론은 네트워크에서 취득한 데이터를 각 개념별로 분석하고, Skewness 계수를 사용하여 latent space의 품질을 정량화합니다. 이 방법은 현재의 vision 모델이 개별 개념의 분포를 어떻게 왜곡하고 있는지를 평가하는 데 초점을 맞추고 있습니다. 또한, robust한 모델에서의 분포 왜곡이 감소하는 경향성을 관찰했습니다.

- **Performance Highlights**: 결과적으로, 모델이 여러 데이터셋에 대해 일반화 능력이 향상될수록 개념의 클러스터링이 좋아지는 경향이 있으며, 이는 robust성이 증가함에 따라 발생합니다. 이는 모델의 일반화 능력과 robust성 사이의 관계를 드러냅니다.



### MoRA: LoRA Guided Multi-Modal Disease Diagnosis with Missing Modality (https://arxiv.org/abs/2408.09064)
Comments:
          Accepted by MICCAI 2024

- **What's New**: 본 논문에서는 다중 모달(pre-trained models) 모델을 질병 진단에 활용하고, 데이터를 다루는데 있어 모달이 불완전할 때 성능과 강건성을 향상시키기 위해 Modality-aware Low-Rank Adaptation (MoRA) 방식을 제안합니다.

- **Technical Details**: MoRA은 각 입력을 저차원(low intrinsic dimension)으로 프로젝션(projection)한 후, 모달 특화된 업 프로젝션을 통해 적응(adaptation)시킵니다. 이는 모달이 누락된 경우 각 모달의 고유한 특성을 파악하여 모델의 강건성과 성능을 향상시킵니다.

- **Performance Highlights**: MoRA은 기존 방법들과 비교하여 질병 진단에서 뛰어난 정확도와 강건성을 보여주며, 상대적으로 적은 데이터셋(몇 천 샘플)에서도 더 나은 성능을 달성하는 훈련 효율성을 입증하였습니다.



### ADen: Adaptive Density Representations for Sparse-view Camera Pose Estimation (https://arxiv.org/abs/2408.09042)
Comments:
          ECCV 2024, Oral

- **What's New**: ADen (Adaptive Density Estimation Network)는 여러 카메라 포즈 가설을 생성하고 이를 평가하기 위해 생성기(generator)와 판별기(discriminator)를 사용하는 새로운 프레임워크를 제안합니다. 이 방식은 서로 다른 두 접근 방식을 통합하여 다수의 모드를 모델링할 수 있으며, 이전 방법들보다 더 높은 정밀도와 빠른 실행 시간을 제공합니다.

- **Technical Details**: ADen은 조건부 확률 분포에서 카메라 포즈를 학습하고 샘플링하는 데 필요한 샘플 수를 줄입니다. 이 방법은 고차원 공간에서도 자연스럽게 확장되며, 모든 공간을 exhaustively 샘플링할 필요가 없습니다. 생성기는 상대 포즈의 조건부 분포로부터 샘플을 생산하고, 판별기는 이 가설 중에서 데이터에 가장 잘 맞는 것을 식별합니다.

- **Performance Highlights**: 실험 결과, ADen은 기존의 SoTA (State-of-the-Art) 방법들을 큰 폭으로 능가하며, 특히 낮은 오류 임계값에서도 뛰어난 성능을 보였습니다. ADen은 또한 이전 방법들보다 훨씬 빠른 실행 속도를 달성하여, 실시간 inference 속도를 제공합니다.



### Multi Teacher Privileged Knowledge Distillation for Multimodal Expression Recognition (https://arxiv.org/abs/2408.09035)
- **What's New**: 이 논문에서는 다양한 모달리티에서 정보를 추출하기 위해 다중 선생 모델(Multi-Teacher Model)을 이용하는 다중 선생 특권 지식 증류(Multi-Teacher Privileged Knowledge Distillation, MT-PKDOT) 방법을 제안하고 있습니다. 이 방법은 학생 모델이 여러 다양한 소스에서 학습하도록 하여 성능을 개선합니다.

- **Technical Details**: MT-PKDOT는 최적 수송(Optimal Transport) 메커니즘에 기반한 구조적 유사성(Knowledge Distillation) 방법을 활용하여 다양한 선생 모델의 표현을 정렬합니다. 여기서는 특권 모달리티와 관련된 데이터를 활용하여 학생 모델의 학습을 돕습니다.

- **Performance Highlights**: MT-PKDOT 방법은 Affwild2와 Biovid 데이터셋에서 검증되었습니다. Biovid 데이터에서는 시각 전용 기준보다 5.5% 개선되었으며, Affwild2 데이터에서는 발란스(valence)와 각성(arousal) 측면에서 각각 3% 및 5% 향상되었습니다.



### Comparative Performance Analysis of Transformer-Based Pre-Trained Models for Detecting Keratoconus Diseas (https://arxiv.org/abs/2408.09005)
Comments:
          14 pages, 3 tables, 27 figures

- **What's New**: 이번 연구는 각기 다른 8개의 사전 훈련된 CNN을 비교하여 각 모델의 각막원추증(keratoconus) 진단 능력을 평가했습니다. 또한, 데이터셋을 철저히 선택하여 각막원추증, 정상, 의심 사례를 포함했습니다.

- **Technical Details**: 테스트한 모델에는 DenseNet121, EfficientNetB0, InceptionResNetV2, InceptionV3, MobileNetV2, ResNet50, VGG16, VGG19가 포함됩니다. 모델 훈련을 극대화하기 위해 나쁜 샘플 제거, 리사이징(resizing), 리스케일링(rescaling), 증강(augmentation) 기법이 사용되었습니다. 각 모델은 유사한 파라미터와 활성화 함수, 분류 함수, 옵티마이저(optimizer)로 훈련되었습니다.

- **Performance Highlights**: MobileNetV2는 각막원추증과 정상 사례를 잘 식별하는 데 가장 높은 정확도를 보였으며, 잘못 분류된 경우가 적었습니다. InceptionV3와 DenseNet121은 각막원추증 탐지에 효과적이었지만, 의심 사례를 다루는 데 어려움을 겪었습니다. 반면, EfficientNetB0, ResNet50, VGG19는 의심 사례와 일반 사례를 구별하는 데 더 많은 어려움을 보여, 모델 세분화 및 개발 필요성을 시사합니다.



### Fire Dynamic Vision: Image Segmentation and Tracking for Multi-Scale Fire and Plume Behavior (https://arxiv.org/abs/2408.08984)
- **What's New**: 이 논문에서는 다양한 공간 및 시간적 스케일에서 화재와 연기가 어떻게 확산되는지를 정확히 모델링하고 추적하는 새로운 접근 방식을 소개합니다. 이 방법은 이미지 분할(image segmentation)과 그래프 이론(graph theory)을 결합하여 화재와 연기의 경계를 구분합니다.

- **Technical Details**: Fire Dynamic Vision (FDV)라는 도구를 통해, 시각 및 적외선 비디오에서 화재 및 연기의 동역학을 분석합니다. FDV는 RGB(Red, Green, Blue)와 HSV(Hue, Saturation, Value) 이미지 분할 기술을 사용하고, 공간 클러스터 분석(spatial cluster analysis) 기법을 통해 여러 화재 전선(multiple fire fronts)이 있는 비디오를 처리합니다.

- **Performance Highlights**: 이 방법은 다양한 이미지 소스에서 화재와 연기의 동적 특성을 성공적으로 추적하고 분석하는 데 효과적임을 보여주었습니다. 결과적으로, 다양한 해상도의 비디오에서 화재와 연기의 확산을 포착하여, 통계적 및 기계 학습 모델에 유용한 데이터셋을 생성하는 포괄적인 파이프라인을 제시합니다.



### Deep Generative Classification of Blood Cell Morphology (https://arxiv.org/abs/2408.08982)
- **What's New**: 새로운 연구에서는 CytoDiffusion이라는 확산 기반의 분류기를 소개합니다. 이 모델은 혈액 세포의 복잡한 형태를 효과적으로 모델링하여 정확한 분류와 강력한 이상 탐지, 분포 변동에 대한 저항성을 제공합니다.

- **Technical Details**: CytoDiffusion는 32,619개의 이미지를 사용하여 훈련되었으며, 전문가들로 구성된 Turing 테스트에서 52.3%의 정확도로 합성 이미지를 구별할 수 있는 능력을 보여주었습니다. 이 모델은 강력한 이상 탐지 기능과 저 데이터 상태에서의 높은 성능(AUC 0.976)을 자랑합니다.

- **Performance Highlights**: CytoDiffusion는 기존 최첨단 모델들보다 우수한 성능을 보이며, 희귀하거나 보지 못한 세포 유형을 효율적으로 탐지할 수 있는 능력을 보여주었습니다. 특히 PBC 데이터셋에서 95.88%의 balanced accuracy를 기록했습니다.



### Enhancing Object Detection with Hybrid dataset in Manufacturing Environments: Comparing Federated Learning to Conventional Techniques (https://arxiv.org/abs/2408.08974)
Comments:
          Submitted and Presented at the IEEE International Conference on Innovative Engineering Sciences and Technological Research (ICIESTR-2024)

- **What's New**: 이 논문에서는 페더레이티드 러닝(Federated Learning, FL) 모델이 전통적인 객체 탐지 기술과 비교하여 소형 객체 탐지에서 더 뛰어난 성능을 보인다는 것을 보여주고 있습니다. 특히 다양한 환경에서 테스트한 결과, FL이 중앙 집중형 훈련 모델을 능가한다는 점을 강조하고 있습니다.

- **Technical Details**: 페더레이티드 러닝은 클라이언트 간에 원시 데이터를 공유하는 대신 모델 가중치만 공유하여 데이터 프라이버시를 보장합니다. 연구에서는 하이브리드 데이터셋(합성 및 실제 데이터셋)을 활용하여 YOLOv5 알고리즘을 적용하고, 전이 학습(Transfer Learning) 및 미세 조정(Fine-tuning) 기법을 사용하여 소형 객체 탐지 성능을 평가하였습니다.

- **Performance Highlights**: 결과적으로, FL 모델은 다양한 조명 조건, 배경의 혼잡도, 객체의 시점 변화 등에서 효과적으로 작동하는 강력한 글로벌 모델을 생성할 수 있습니다. 이는 제조 환경에서의 내구성 있는 객체 탐지 모델 배치에 대한 귀중한 통찰력을 제공합니다.



### Image Class Translation Distance: A Novel Interpretable Feature for Image Classification (https://arxiv.org/abs/2408.08973)
Comments:
          20 pages, 18 figures, submitted to Computational Intelligence

- **What's New**: 이번 연구에서는 전통적인 블랙 박스(classification networks) 대안으로 더 해석 가능한 이미지 분류를 위한 이미지 변환 네트워크(image translation networks)의 새로운 응용을 제안합니다. 이미지 간 변환 거리(translation distance)를 정량화하여 분류 문제를 해결하는 방법을 보여주었습니다.

- **Technical Details**: 훈련된 네트워크는 이미지를 가능한 클래스 간에 변환하는 방법으로, 변환 거리를 측정함으로써 각 클래스에 적합하기 위해 이미지가 얼마나 수정되어야 하는지를 나타냅니다. 이러한 변환 거리는 클러스터와 트렌드를 분석하는 데 사용되며, SVM(Support Vector Machine)과 같은 단순 분류기를 통해 정확성을 유지합니다.

- **Performance Highlights**: 사과와 오렌지 간의 이미지 변환을 시작으로, 멜라노마 감지 및 6가지 세포 유형(class) 분류에 이르는 다양한 의료 이미지 작업에 전략을 적용한 결과, 기존의 CNN보다 향상된 성능을 보여주었습니다. 또한 변환된 이미지의 시각적 검사는 데이터 세트의 특성과 편향을 드러내는 중요한 정보를 제공합니다.



### SHARP-Net: A Refined Pyramid Network for Deficiency Segmentation in Culverts and Sewer Pipes (https://arxiv.org/abs/2408.08879)
- **What's New**: SHARP-Net(Semantic Haar-Adaptive Refined Pyramid Network)을 소개하며, 이 네트워크는 다중 스케일 특징을 포착하는 새로운 구조로 설계되었다. Inception-like 블록과 깊이별 분리 가능한 합성곱을 이용하여 세그멘테이션을 개선한다.

- **Technical Details**: SHARP-Net은 Inception-like 블록과 $1	imes1$, $3	imes3$ 깊이별 분리 가능한 합성곱을 이용하여 고해상도 특징을 생성하며, Haar-like 특징을 통합하여 성능을 극대화한다. 이 모델은 다양한 필터 크기를 사용하여 multi-scale 특징을 효과적으로 캡처한다.

- **Performance Highlights**: SHARP-Net은 Culvert-Sewer Defects 데이터셋에서 77.2%의 IoU를 달성하며, DeepGlobe Land Cover 데이터셋에서는 70.6%의 IoU를 기록했다. Haar-like 특징을 통합하여 기본 모델보다 22.74%의 성능 개선을 이뤄내고, 다른 딥러닝 모델에 적용 시 35.0%의 성능 향상을 보였다.



### Criticality Leveraged Adversarial Training (CLAT) for Boosted Performance via Parameter Efficiency (https://arxiv.org/abs/2408.10204)
Comments:
          9 pages + appendix/ additional experiments

- **What's New**: 이번 논문은 CLAT라는 새로운 접근 방식을 소개합니다. CLAT는 적대적 훈련(adversarial training) 중 과적합(overfitting) 문제를 완화하여 클린 정확도(clean accuracy) 및 적대적 강건성(adversarial robustness)을 동시에 향상시키는 방법입니다.

- **Technical Details**: CLAT는 비강건 특징(non-robust features)을 학습하는 주요 레이어(critical layers)만을 식별하고 조정(fine-tune)하며, 나머지 레이어는 동결(freeze)하여 강건성을 강화합니다. CLAT의 알고리즘은 동적(dynamic)으로 레이어의 중요성을 선택하여 훈련 과정에서 그 필요성에 따라 조정됩니다. 이 방법은 기존 적대적 훈련 방법 위에 추가로 적용할 수 있으며, 훈련 가능 매개변수(trainable parameters) 수를 약 95% 감소시킵니다.

- **Performance Highlights**: CLAT는 이전 적대적 훈련 방법과 비교하여 2% 이상의 적대적 강건성 향상을 보여주며, 과적합의 위험을 줄이면서도 클린 정확도를 유지하거나 약간 향상시킵니다. 또한, CLAT는 메모리 절약을 가능하게 하는 매개변수 효율적인 방법입니다.



### Learning Precise Affordances from Egocentric Videos for Robotic Manipulation (https://arxiv.org/abs/2408.10123)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 로봇 조작 작업을 위한 효율적인 어포던스(affordance) 학습 시스템을 제안합니다. 자동화된 데이터 수집부터 모델 훈련, 로봇 배치까지의 과정을 포함하여, 기존 방법과 차별화된 방식으로 정확한 세분화 마스크를 활용하여 유용한 정보를 추출합니다.

- **Technical Details**: 본 연구는 Geometry-guided Affordance Transformer (GKT) 모델을 제안하며, 이 모델은 Depth Feature Injector (DFI)를 통합하여 3D 형상 및 기하학적 정보를 반영합니다. 또한, Aff-Grasp라는 프레임워크를 통해 다양한 작업 요건에 따라 적응적으로 물체를 잡을 수 있는 기능을 갖추고 있습니다.

- **Performance Highlights**: GKT 모델은 mIoU에서 최신 기술보다 15.9% 높은 성능을 기록하였으며, Aff-Grasp는 179회의 실험에서 어포던스 예측의 성공률 95.5%, 성공적인 잡기 77.1%의 높은 성공률을 보였습니다.



### Towards a Benchmark for Colorectal Cancer Segmentation in Endorectal Ultrasound Videos: Dataset and Model Developmen (https://arxiv.org/abs/2408.10067)
- **What's New**: 이 논문에서는 대규모 ERUS(Endorectal Ultrasound) 데이터셋인 ERUS-10K를 제안합니다. 이 데이터셋은 77개의 비디오와 10,000개의 고해상도 주석 프레임을 포함하고 있으며, 다양한 상황을 포함한 최초의 기준 데이터셋입니다.

- **Technical Details**: 제안된 모델인 Adaptive Sparse-context Transformer (ASTR)는 세 가지 요소를 기반으로 설계되었습니다: 스캐닝 모드 불일치, 시간 정보, 낮은 계산 복잡성. ASTR는 각 프레임의 로컬 및 글로벌 특징을 통합하기 위해 sparse-context transformer를 사용하고, 참조 프레임에서 맥락적 특징을 추출하기 위해 sparse-context block을 도입합니다.

- **Performance Highlights**: 제안된 ASTR 모델은 기준 데이터셋에서 직장암 분할에서 77.6%의 Dice 점수를 달성하였으며, 이는 이전의 최첨단 방법들을 크게 능가하는 성능을 보여줍니다.



### Exploiting Fine-Grained Prototype Distribution for Boosting Unsupervised Class Incremental Learning (https://arxiv.org/abs/2408.10046)
- **What's New**: 본 논문은 감독 없는 클래스 증가 학습(UCIL)이라는 도전적인 문제를 다루며, 기존의 클래스 증가 학습(CIL) 방법들이 일반적으로 모든 진짜 라벨을 사용 가능하다는 가정을 내세우는 것과 달리, 진짜 라벨이 없는 상황에서도 효과적으로 새로운 클래스를 발견하고 기존의 지식을 보존하는 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 미세한 프로토타입 분포(fine-grained prototype distribution)를 통해 UCIL을 촉진하는 새로운 프레임워크를 포함합니다. 미리 훈련된 ViT(비전 트랜스포머)를 피처 인코더로 사용하고, 이를 고정하여 학습하는 가운데, 미세한 프로토타입 분포를 활용하여 특성 공간(feature space) 내의 객체를 더 자세히 식별합니다. 새로운 클래스와 기존 클래스 간의 중복을 줄이는 전략도 포함되어 있습니다.

- **Performance Highlights**: 다섯 개의 벤치마크 데이터셋에서 제안된 방법은 CIFAR100과 CUB 데이터셋에서 각각 9.1%와 9.0% 향상을 보이며, 기존의 최첨단 방법들과 비교해 상당한 성능 향상을 달성했습니다.



### Pose-GuideNet: Automatic Scanning Guidance for Fetal Head Ultrasound from Pose Estimation (https://arxiv.org/abs/2408.09931)
Comments:
          Accepted by MICCAI2024

- **What's New**: 이번 연구에서는 자유 손으로 실시한 2D 초음파 스캔에서 3D 태아 자세를 추정하여, 초음파 검사자가 머리의 표준 평면(standard plane)을 찾도록 안내하는 Pose-GuideNet이라는 새로운 방법을 제안합니다.

- **Technical Details**: Pose-GuideNet은 2D 초음파 이미지를 3D 해부학적 아틀라스에 정렬하기 위한 혁신적인 2D/3D 등록 접근법을 사용합니다. 이 방법은 기존의 3D 초음파 이미지를 이용하지 않고도 2D 초음파 스캔 데이터를 기반으로 3D 정보를 추정할 수 있게 해줍니다. 세부적으로는 기하학적 정보와 해부학적 유사성을 활용하여 스캔 프레임을 정렬합니다.

- **Performance Highlights**: 임상 두부 생체 측정 작업에서 Pose-GuideNet은 정확한 자세 예측 및 태아 머리 방향 예측을 보여 주었습니다. 센서가 없는 환경에서도 자유로운 초음파 정상 항법에 대한 적용 가능성을 입증하였습니다.



### Data Augmentation of Contrastive Learning is Estimating Positive-incentive Nois (https://arxiv.org/abs/2408.09929)
- **What's New**: 이 논문은 Positive-incentive Noise (Pi-Noise 또는 π-Noise)의 개념을 활용하여 대비 학습(contrastive learning)과 π-Noise의 연관성을 탐구합니다. 특히, 대비 손실을 보조 Gaussian 분포로 변환하여 대비 학습의 태스크 엔트로피를 정의하는 새로운 방법을 제안합니다.

- **Technical Details**: 대비 학습에서의 태스크 엔트로피( task entropy) H(𝒯)는 보조 Gaussian 분포 설계를 통해 정의됩니다. 이는 정보 이론의 프레임워크를 활용하여 특정 대비 모델의 난이도를 정량적으로 측정하게 합니다. 또한 본 연구는 기존 대비 학습 프레임워크에서의 사전 정의된 데이터 증강이 π-Noise의 점 추정(point estimation)으로 간주될 수 있음을 증명합니다.

- **Performance Highlights**: 새롭게 제안된 π-Noise 생성기는 다양한 데이터 유형에 적용 가능하며, 기존의 대비 모델과 완벽하게 호환됩니다. 실험 결과, 제안된 방법이 효과적인 증강(augmentations)을 성공적으로 학습함을 보여줍니다.



### LCE: A Framework for Explainability of DNNs for Ultrasound Image Based on Concept Discovery (https://arxiv.org/abs/2408.09899)
- **What's New**: 본 논문에서는 의학 이미지 분석에서 Deep Neural Networks (DNNs)의 설명 가능성을 높이기 위한 새로운 프레임워크인 Lesion Concept Explainer (LCE)를 제안합니다. 기존의 attribution 방법과 concept-based 방법의 결합을 통해 초음파 이미지에 대한 의미 있는 설명을 가능하게 합니다.

- **Technical Details**: LCE는 Segment Anything Model (SAM)을 기반으로 하며, 대규모 의학 이미지 데이터로 파인튜닝되었습니다. Shapley value를 사용하여 개념 발견과 설명을 수행하며, 새로운 평가 지표인 Effect Score를 제안하여 설명의 신뢰성과 이해 가능성을 동시에 고려합니다. 이 프레임워크는 ResNet50 모델을 사용하여 공공 및 사적 유방 초음파 데이터셋(BUSI 및 FG-US-B)에서 평가되었습니다.

- **Performance Highlights**: LCE는 기존의 설명 가능성 방법들에 비해 우수한 성능을 나타내며, 전문가 초음파 검사자들에 의해 높은 이해도를 기록했습니다. 특히, LCE는 유방 초음파 데이터의 세밀한 진단 작업에서도 안정적이고 신뢰할 수 있는 설명을 제공하였으며, 추가 비용 없이 다양한 진단 모델에 적용 가능하다는 장점이 있습니다.



### Preoperative Rotator Cuff Tear Prediction from Shoulder Radiographs using a Convolutional Block Attention Module-Integrated Neural Network (https://arxiv.org/abs/2408.09894)
- **What's New**: 이번 연구에서는 평면 어깨 엑스레이(shoulder radiograph)와 딥 러닝(deep learning) 방법을 조합해 회전근개 파열(rotator cuff tears) 환자를 식별할 수 있는지를 테스트했습니다.

- **Technical Details**: 우리는 convolutional block attention modules를 딥 뉴럴 네트워크(deep neural network)에 통합하여 모델을 개발했습니다. 이 모델은 회전근개 파열을 검출하는 데 있어 높은 정확도를 보였으며, 평균 AUC(area under the curve) 0.889 및 0.831의 정확성을 달성했습니다.

- **Performance Highlights**: 본 연구는 딥 러닝 모델이 엑스레이에서 회전근개 파열을 정확히 감지할 수 있음을 검증하였습니다. 이는 MRI와 같은 더 비싼 이미징 기술에 대한 유효한 사전 평가(pre-assessment) 또는 대안이 될 수 있습니다.



### New spectral imaging biomarkers for sepsis and mortality in intensive car (https://arxiv.org/abs/2408.09873)
Comments:
          Markus A. Weigand, Lena Maier-Hein and Maximilian Dietrich contributed equally

- **What's New**: 이 연구는 hyperspectral imaging (HSI)을 활용하여 패혈증(sepsis) 진단과 사망률 예측을 위한 새로운 생체 지표(biomarker)를 제시한다. 기존의 진단 방안들이 가진 한계를 극복하기 위한 노력의 일환으로, HSI 기반의 접근법이 실용적이고 비침습적(non-invasive)이며 빠른 진단이 가능하다는 점에 주목했다.

- **Technical Details**: 480명 이상의 환자에서 HSI 데이터를 수집하여, 손바닥과 손가락에서 패혈증을 예측하는 알고리즘을 개발했다. HSI 측정치는 패혈증 관련하여 0.80의 AUROC, 사망률 예측에 대해서는 0.72의 AUROC를 달성하였다. 임상 데이터를 추가하면 패혈증에 대해 0.94, 사망률에 대해 0.84의 AUROC까지 향상됨을 보였다.

- **Performance Highlights**: HSI는 기존의 생체 지표와 비교하여 신뢰성 높은 예측 성능을 제공하며, 특히 중환자실(ICU) 환자에 대한 신속하고 비침습적인 진단 방법으로서의 가능성을 보여준다. 이 연구는 머신러닝을 통해 HSI 데이터를 분석하여 패혈증과 사망률 예측에 효과적인 새로운 접근 방식을 제시하였다.



### Docling Technical Repor (https://arxiv.org/abs/2408.09869)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2206.01062

- **What's New**: 이번 기술 보고서는 PDF 문서 변환을 위한 간편한 오픈 소스 패키지인 Docling을 소개합니다. 이는 레이아웃 분석(layout analysis)을 위한 AI 모델인 DocLayNet과 테이블 구조 인식(table structure recognition)을 위한 모델인 TableFormer로 구동되며, 일반 하드웨어에서 효율적으로 실행됩니다.

- **Technical Details**: Docling은 사용자가 PDF 문서를 JSON 또는 Markdown 형식으로 변환하고, 세부 페이지 레이아웃을 이해하며, 그림을 찾고 테이블 구조를 복구하는 기능을 제공합니다. 또한 메타데이터(예: 제목, 저자, 참고문헌, 언어)를 추출할 수 있으며, OCR 기능을 선택적으로 적용할 수 있습니다.

- **Performance Highlights**: Docling은 배치 모드(높은 처리량, 낮은 해결 시간)와 인터랙티브 모드(효율성 타협, 낮은 해결 시간)에 최적화할 수 있으며, 다양한 가속기(GPU, MPS 등)를 활용할 수 있습니다.



### Hear Your Face: Face-based voice conversion with F0 estimation (https://arxiv.org/abs/2408.09802)
Comments:
          Interspeech 2024

- **What's New**: 본 논문은 개인의 얼굴 특징과 음성 특성 간의 독특한 관계를 활용하여 새로운 Face-based Voice Conversion 프레임워크를 제안합니다. 이 프레임워크는 주로 목표 화자의 얼굴 이미지에서 파생된 평균 기본 주파수(average fundamental frequency)를 사용하여 얼굴 기반 음성을 변환합니다.

- **Technical Details**: 제안하는 HYFace 네트워크는 조건부 변분 오토인코더(conditional variational autoencoder) 아키텍처를 기반으로 하며, 선행 인코더(prior encoder)에 얼굴 이미지를 조건으로 사용하는 방식으로 설계되었습니다. 그 과정에서 얼굴 이미지로부터 학습된 화자 임베딩이 여러 인코더 및 디코더를 조건화하여 음성을 변환합니다. 이를 통해 기본 주파수(F0)를 프레임 단위로 조절하여 원본 오디오의 스타일을 목표 화자의 얼굴 기반 특성으로 수정합니다.

- **Performance Highlights**: HYFace는 기존의 얼굴 기반 음성 변환에서 새로운 기준을 제시하며, 고품질 합성 음성 생성 및 얼굴 이미지와 음성 특성 간의 일치를 효과적으로 달성하는 성능을 보여줍니다. 구체적으로, 실험 결과 HYFace 모델은 유사한 연구 결과들과 비교하여 성능 향상을 나타내고 있으며, 데모는 https://jaejunL.github.io/HYFace_Demo/에서 확인 가능합니다.



### Anim-Director: A Large Multimodal Model Powered Agent for Controllable Animation Video Generation (https://arxiv.org/abs/2408.09787)
Comments:
          Accepted by SIGGRAPH Asia 2024, Project and Codes: this https URL

- **What's New**: 이번 연구에서는 LMMs(large multimodal models)를 활용하여 애니메이션 제작을 자동화하는 새로운 방식의 에이전트인 Anim-Director를 소개합니다. 이_agent_는 사용자로부터 간결한 내러티브나 지시 사항을 받아 일관성 있는 애니메이션 비디오를 생성할 수 있도록 설계되었습니다.

- **Technical Details**: Anim-Director는 세 가지 주요 단계로 작동합니다: 첫 번째로, 사용자 입력을 바탕으로 일관된 스토리를 생성하고, 두 번째 단계에서는 LMMs와 이미지 생성 도구를 사용하여 장면의 비주얼 이미지를 생산합니다. 마지막으로, 이러한 이미지들은 애니메이션 비디오 제작의 기초로 활용되며, LMMs가 각각의 프로세스를 안내하는 프롬프트를 생성합니다.

- **Performance Highlights**: 실험 결과, Anim-Director는 개선된 스토리라인과 배경의 일관성을 유지하며 긴 애니메이션 비디오를 생성할 수 있음을 증명했습니다. LMMs와 생성 도구의 통합을 통해 애니메이션 제작 프로세스를 크게 간소화하여, 창조적인 과정의 효율성과 비디오 품질을 향상시키는 데 기여합니다.



### Coarse-Fine View Attention Alignment-Based GAN for CT Reconstruction from Biplanar X-Rays (https://arxiv.org/abs/2408.09736)
- **What's New**: 이 논문에서는 수술 계획 및 실시간 이미징을 위해 이차 평면 X선(biplanar X-ray) 이미지를 사용하여 3D CT 이미지를 복원하는 새로운 방법을 제안합니다. 기존 연구와는 달리, X선 뷰를 동등하게 다루는 대신, 중요한 정보를 강조할 수 있도록 'Coarse-Fine View Attention Alignment' 방법론을 활용했습니다.

- **Technical Details**: 제안된 CVAA-GAN(Course-Fine View Attention Alignment GAN)은 두 개의 직각 이차 평면 X선의 피처를 결합하는 새로운 구조입니다. 주요 구성 요소로는 Generator 네트워크와 Discriminator 네트워크가 포함됩니다. Generator는 X선 이미지를 3D 피처로 변환하는 두 개의 인코더와 3D 이미지 복원을 수행하는 디코더로 구성됩니다. 또한, 뷰 주의 정렬(View Attention Alignment) 서브 모듈과 정제(Fine Distillation) 서브 모듈을 도입하여 각 뷰의 정보를 효과적으로 조합합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 기존 SOTA(State-of-the-Art) 방법들에 비해 우수한 성능을 보였습니다. 특히, 이차 평면 X선 이미지를 활용한 CT 복원에서 더 정확하고 세밀한 해부학적 구조를 제공하는 것으로 나타났습니다.



### Diff2CT: Diffusion Learning to Reconstruct Spine CT from Biplanar X-Rays (https://arxiv.org/abs/2408.09731)
- **What's New**: 최신 연구에서는 복합 X-ray 이미지를 활용한 3D CT 재구성의 혁신적인 방법을 소개합니다. 기존의 이미지 생성 기술에 의존하지 않고, 조건부 확산(diffusion) 과정을 통해 재구성 문제를 해결하고자 하였습니다.

- **Technical Details**: 이 연구는 Denoising Diffusion Probabilistic Model (DDPM)을 기반으로 하여, 복합 X-ray 이미지로부터 3D CT 이미지를 생성하는 과정에서 새로운 조건부 디노이징 오토인코더를 사용합니다. 이 과정에서는 새로운 프로젝션 손실 함수가 도입되어 구조적 일관성을 개선합니다.

- **Performance Highlights**: 제안하는 방법은 기존의 최신 기법들보다 뛰어난 성능을 보여주었으며, 구조 유사도 지수(SSIM)가 0.83으로 10% 향상되었고, Fréchet Inception Distance (FID)가 83.43으로 25% 감소하였습니다.



### HYDEN: Hyperbolic Density Representations for Medical Images and Reports (https://arxiv.org/abs/2408.09715)
- **What's New**: 본 논문에서는 의료 도메인을 특화하여 이미지-텍스트 표현 학습을 위한 새로운 방법인 HYDEN을 제안합니다. 이는 하이퍼볼릭 밀도 임베딩(hyperbolic density embedding)을 사용하여 이미지와 텍스트의 관계를 효과적으로 모델링합니다.

- **Technical Details**: HYDEN은 하이퍼볼릭 공간에서 이미지 텍스트 특징을 밀도 특징으로 매핑하는 방법을 제공합니다. 이 방법은 이미지에서의 전역 특징(global features)과 텍스트 인식 지역 특징(text-aware local features)을 통합하며, 하이퍼볼릭 유사-가우시안 분포(pseudo-Gaussian distribution)를 활용합니다. 또한 캡슐화 손실 함수(encapsulation loss function)를 통해 이미지-텍스트 밀도 분포 간의 부분 순서 관계(partial order relations)를 모델링합니다.

- **Performance Highlights**: 실험 결과 HYDEN은 여러 제로샷(zero-shot) 작업 및 다양한 데이터셋에서 기존 방법들에 비해 우수한 성능을 보였습니다. 본 연구는 의료 이미징 및 관련 의학 보고서에 대한 해석 가능성과 뛰어난 성능을 강조하고 있습니다.



### TESL-Net: A Transformer-Enhanced CNN for Accurate Skin Lesion Segmentation (https://arxiv.org/abs/2408.09687)
- **What's New**: 본 연구는 피부 병변의 정확한 분할을 위한 새로운 네트워크 TESL-Net을 제안합니다. 이 네트워크는 CNN 인코더-디코더 아키텍처의 지역적 특징과 Bi-ConvLSTM 네트워크 및 Swin transformer를 통한 장기 및 시간 의존성을 결합하여 성능을 향상시킵니다.

- **Technical Details**: TESL-Net은 RGB 이미지와 해당 마스크를 입력으로 받아, 인코더 단계에서 심층 별도 합성곱(Depthwise Separable Convolution)과 활성화 함수, 배치 정규화(Batch Normalization) 레이어를 적용합니다. 최대 풀링(Max Pooling)을 사용하여 특징의 공간적 차원을 줄이고, Swin transformer 블록을 통해 특징 정보를 패치 단위로 추출 및 정제합니다. 디코더 단계에서는 전치 합성곱(Transposed Convolution)을 사용하여 특징 맵을 업샘플링하고, Bi-ConvLSTM이 단기 및 장기 의존성을 포착합니다.

- **Performance Highlights**: TESL-Net은 ISIC 2016, 2017, 2018 데이터셋에서 평가되었으며, Jaccard 지수에서 유의미한 성과 향상을 보여주며 최신 성능을 달성하였습니다.



### Screen Them All: High-Throughput Pan-Cancer Genetic and Phenotypic Biomarker Screening from H\&E Whole Slide Images (https://arxiv.org/abs/2408.09554)
- **What's New**: 이 연구는 전통적인 단일 또는 다중 유전자 분석 방법에 비해 훨씬 빠르고 비용 효율적인 AI 기반 시스템을 제안합니다. 이 시스템은 3백만 개의 슬라이드에 대해 사전 훈련된 Virchow2 모델을 활용하여, 38,984명의 환자로부터 스캔된 47,960개의 H&E 전 슬라이드 이미지(WSI)에서 유전자 특징을 추출합니다.

- **Technical Details**: 제안된 모델은 1,228개의 유전자 마커를 동시에 예측하는 통합 모델로, Memorial Sloan Kettering Cancer Center (MSKCC) 데이터 세트를 사용하여 훈련되었습니다. 이는 505개의 유전자에 대한 MSK-IMPACT 패널과의 일치를 통해 80개의 높은 성능을 가진 바이오마커를 발견하였으며, 평균 AU-ROC는 0.89에 달합니다.

- **Performance Highlights**: AI 모델은 15개 암 유형에서 391개의 유전자 변화를 예측하는데 성공하였으며, 평균 AUC 0.84, 평균 민감도 0.92, 평균 특이도 0.55의 성능을 기록했습니다. 이 모델은 치료 선택을 안내하고, 임상 시험을 위한 환자 스크리닝을 가속화하는 데 기여할 잠재력이 있습니다.



### Image-Based Geolocation Using Large Vision-Language Models (https://arxiv.org/abs/2408.09474)
- **What's New**: 이 논문은 기존의 딥러닝 및 LVLM(large vision-language models) 기반 지리 정보 방법들이 제기하는 문제를 심층적으로 분석한 첫 번째 연구입니다. LVLM이 이미지로부터 지리 정보를 정확히 파악할 수 있는 가능성을 보여 주며, 이를 해결하기 위해 	ool{}이라는 혁신적인 프레임워크를 제안합니다.

- **Technical Details**: 	ool{}는 체계적인 사고 연쇄(chain-of-thought) 접근 방식을 사용하여 차량 유형, 건축 스타일, 자연 경관, 문화적 요소와 같은 시각적 및 맥락적 단서를 면밀히 분석함으로써 인간의 지리 추측 전략을 모방합니다. 50,000개의 실제 데이터를 기반으로 한 광범위한 테스트 결과, 	ool{}은 전통적인 모델 및 인간 기준보다 높은 정확도를 달성했습니다.

- **Performance Highlights**: GeoGuessr 게임에서 평균 점수 4550.5와 85.37%의 승률을 기록하며 뛰어난 성능을 보였습니다. 가장 가까운 거리 예측의 경우 0.3km까지 정확성을 보였고, LVLM의 인지 능력을 활용한 정교한 프레임워크를 통해 지리 정보의 정확성을 크게 향상시켰습니다.



### Deformation-aware GAN for Medical Image Synthesis with Substantially Misaligned Pairs (https://arxiv.org/abs/2408.09432)
Comments:
          Accepted by MIDL2024

- **What's New**: 본 논문에서는 Deformation-aware GAN (DA-GAN)이라는 혁신적인 접근법을 제안하여 의료 이미지 합성 과정에서 발생하는 상당한 비정렬 문제를 다루고 있습니다. 이 방법은 다중 목표 역 일관성(multi-objective inverse consistency)을 기반으로 하여 이미지 합성 중 비정렬을 동적으로 수정합니다.

- **Technical Details**: DA-GAN은 생성 과정에서 대칭 등록(symmetric registration)과 이미지 생성을 조화롭게 최적화하며, 적대적 과정에서는 변형 인식 판별기(deformation-aware discriminator)를 설계하여 이미지 충실도를 향상시킵니다. 이 방법은 매칭된 공간 형태와 이미지 충실도를 분리하여 조정합니다.

- **Performance Highlights**: 실험 결과, DA-GAN은 공공 데이터 세트와 호흡 동작 비정렬이 있는 실제 폐 MRI-CT 데이터 세트에서 우수한 성능을 보였습니다. 이는 방사선 치료 계획과 같은 다양한 의료 이미지 합성 작업에서 DA-GAN의 잠재력을 시사합니다.



### Obtaining Optimal Spiking Neural Network in Sequence Learning via CRNN-SNN Conversion (https://arxiv.org/abs/2408.09403)
Comments:
          Accepted by 33rd International Conference on Artificial Neural Networks

- **What's New**: 본 연구에서는 기존의 Spiking Neural Networks (SNNs)가 갖는 성능 저하 문제를 해결하기 위해 새로운 Recurrent Bipolar Integrate-and-Fire (RBIF) 뉴런 모델을 제안합니다. 이를 통해 RNN과 SNN 간의 매핑에 성공하여, 더 긴 시퀀스 학습에서 안정적인 성능을 보여줍니다.

- **Technical Details**: SNN은 기존의 ANN과 달리 비선형 이진 통신 메커니즘을 사용하여 신경 간 정보를 교환합니다. 본 논문에서는 두 개의 서브 파이프라인(CNN-Morph 및 RNN-Morph)을 통해 엔드 투 엔드 컨버전을 지원하며, 양자화된 CRNN에서의 매핑을 통해 완전한 손실 없는 변환을 달성합니다. 이를 통해 SNN의 성능을 극대화합니다.

- **Performance Highlights**: S-MNIST에서 99.16%의 정확도(0.46% 향상), PS-MNIST(시퀀스 길이 784)에서 94.95%의 정확도(3.95% 향상), 그리고 충돌 회피 데이터셋에서 8 타임 스텝 내 평균 손실 0.057(0.013 감소)을 달성하여 현재 최첨단(contemporaneous state-of-the-art) 방법들을 초월했습니다.



### Flemme: A Flexible and Modular Learning Platform for Medical Images (https://arxiv.org/abs/2408.09369)
Comments:
          8 pages, 6 figures

- **What's New**: Flemme라는 FLExible하고 Modular한 의료 이미지 학습 플랫폼을 제안합니다. 이 플랫폼은 인코더와 모델 아키텍처를 분리하여 다양한 인코더와 아키텍처의 조합을 통해 모델을 쉽게 구성할 수 있도록 지원합니다.

- **Technical Details**: Flemme는 컨볼루션(convolution), 트랜스포머(transformer), 상태 공간 모델(state-space model, SSM) 기반의 인코더를 사용하여 2D 및 3D 이미지 패치를 처리합니다. 기본 아키텍처는 인코더-디코더(encoder-decoder) 스타일이며, 이미지 분할(segmentation), 재구성(reconstruction), 생성(generation) 작업을 위한 파생 아키텍처가 포함되어 있습니다. 일반적인 계층 구조는 피라미드 손실(pyramid loss)을 활용하여 수직적 특징을 최적화하고 융합합니다.

- **Performance Highlights**: 모델 성능 실험 결과, 분할 모델의 Dice 점수에서 평균 5.60%의 개선, 평균 상호작용 단위(mIoU)에서는 7.81%의 향상이 있었고, 재구성 모델에서는 피크 신호 대 잡음 비율(PSNR)에서 5.57%, 구조적 유사도(SSIM)에서 8.22%의 개선을 확인했습니다.



### Improving Lung Cancer Diagnosis and Survival Prediction with Deep Learning and CT Imaging (https://arxiv.org/abs/2408.09367)
- **What's New**: 최근 폐암의 조기 진단과 치료를 향상시키기 위한 새로운 접근법으로, CT 이미지에서 폐의 형태학(morphology)과 폐암 발생 위험 사이의 비선형 관계를 모델링하는 3D 컨볼루션 신경망(convolutional neural networks, CNN)을 제안합니다.

- **Technical Details**: 이 연구에서는 Cox 비례 위험 모델을 확장하여 비볼록성(non-convexity) 문제를 다루는 미니 배치 손실(mini-batched loss)을 사용합니다. 이 손실함수는 폐암 발생 예측과 사망 위험 예측을 동시에 수행하는 데 적용되며, 또한 대규모 데이터 세트 훈련을 가능하게 합니다.

- **Performance Highlights**: National Lung Screening Trial 데이터 세트를 사용한 평가에서 여러 3D CNN 아키텍처를 적용하여 폐암 분류(classification) 및 생존 예측(survival prediction)에서 높은 AUC 및 C-index 점수를 달성했습니다. 시뮬레이션과 실제 데이터 실험을 통해 제안된 방법의 효과성이 입증되었습니다.



### Unpaired Volumetric Harmonization of Brain MRI with Conditional Latent Diffusion (https://arxiv.org/abs/2408.09315)
- **What's New**: 본 연구에서는 Conditional Latent Diffusion (HCLD)를 통해 3D MRI 조화를 위한 새로운 접근 방식을 제안합니다. 이 프레임워크는 이미지 스타일과 뇌 해부학을 명시적으로 고려하여, 다중 사이트에서 얻은 MRI 데이터를 더 효과적으로 조화합니다.

- **Technical Details**: HCLD 프레임워크는 일반화 가능한 3D 오토인코더와 조건부 잠재 확산 모델(conditional latent diffusion model)로 구성되어 있습니다. 이 모델은 4D 잠재 공간을 통해 MRI를 인코딩하고 디코딩하며, 소스 MRI의 잠재 분포를 학습하여 목표 이미지 스타일에 맞춘 조화된 MRI를 생성합니다.

- **Performance Highlights**: HCLD는 4,158개의 T1-가중 뇌 MRI로 훈련 및 평가되었으며, 사이트 관련 변variation을 제거하면서 필수 생물학적 특성을 유지하는 데 성공했습니다. 기존 방법들과 비교하여 HCLD가 더 높은 이미지 품질을 달성한다는 것을 여러 정성적 및 정량적 실험을 통해 입증하였습니다.



### Cross-Species Data Integration for Enhanced Layer Segmentation in Kidney Pathology (https://arxiv.org/abs/2408.09278)
- **What's New**: 이 연구는 인간 신장 데이터 세트를 모델 훈련에 활용하기 위해 생리학적으로 유사한 생쥐 신장 데이터 세트를 통합했으며, 이를 통해 신장 조직의 피질과 수질을 효과적으로 세분화할 수 있는 새로운 접근 방식을 제시합니다.

- **Technical Details**: 생쥐 신장 데이터 세트는 Periodic Acid-Schiff(PAS) 염색을 통해 수집되었으며, 여러 CNN 및 Transformer 아키텍처를 활용하여 공동 훈련을 실시했습니다. 이 과정에서는 Cross Entropy Loss와 Dice Loss를 조합한 하이브리드 손실 함수를 사용하여 다양한 클래스의 데이터를 처리했습니다. 특히, Focal Loss를 통해 레이블 불균형 문제를 해결하고,  공통적 묘사 특징을 가진 서로 다른 종의 데이터를 효과적으로 활용했습니다.

- **Performance Highlights**: 모델 훈련 후 mIoU에서 1.77% 및 1.24%, Dice 점수에서 1.76% 및 0.89%의 성능 향상이 확인되었습니다. 이러한 결과는 생볍으로 보면 모델의 일반화 능력을 향상시키며, 임상 샘플 부족 상황에서도 효과적인 세분화를 지원할 수 있음을 보여줍니다.



### A Fast and Computationally Inexpensive Method For Image Translation of 3D Volume Patient Data (https://arxiv.org/abs/2408.09218)
- **What's New**: 본 논문에서는 CycleGAN을 SynthRAD Grand Challenge Dataset에서 단일 에포크 수정 방법(Single-Epoch Modification, SEM)을 사용하여 훈련하였으며, 이를 CycleGAN-single로 지칭합니다. 이는 보통 200 에포크에서 훈련되는 CycleGAN(CycleGAN-multi) 방법과 비교됩니다.

- **Technical Details**: 모델 성능은 PSNR, SSIM, MAE, MSE와 같은 정량적 성능 지표를 포함하여 정성적 및 정량적으로 평가되었습니다. 이 논문은 의료 영상과 같은 특정 이미지 변환 작업에서의 정량적 및 정성적 성능 평가의 중요성을 강조합니다.

- **Performance Highlights**: FQGA(Fast Paired Image-to-Image Translation Quarter-Generator Adversary) 모델은 CycleGAN에 비해 파라미터 수가 1/4에 불과하지만, 20 에포크 훈련 후에도 정성적 및 정량적으로 CycleGAN을 초월하였으며, SEM 방법을 적용함으로써 추가적인 성능 향상을 보여주었습니다.



### Learning to Explore for Stochastic Gradient MCMC (https://arxiv.org/abs/2408.09140)
- **What's New**: 본 논문에서는 Stochastic Gradient MCMC(SGMCMC) 알고리즘의 메타 학습 전략을 제안하여 다중 모드 타겟 분포를 효율적으로 탐색할 수 있도록 하였습니다. 이 접근법은 다양한 작업에서 전이 가능성을 보여 주며, 메타 학습된 SGMCMC의 샘플링 효율 구축을 가능하게 합니다.

- **Technical Details**: 우리는 L2E(Learning to Explore)라는 새로운 메타 학습 프레임워크를 제안합니다. 이 프레임워크는 기존의 수작업 디자인 접근 방식 대신 데이터에서 직접 설계를 학습하며, 다양한 BNN(inference task)을 포괄하는 비선형 회귀를 사용하여 더 나은 믹싱 비율(mixing rates)과 예측 성능을 실현합니다. 또한, 사전 훈련 단계에서 보지 못한 작업에 대해서도 일반화할 수 있는 능력을 갖추고 있습니다.

- **Performance Highlights**: 이미지 분류 벤치마크를 기반으로 BNN을 통한 SGMCMC의 성능 향상을 입증하였으며, 기존의 수동으로 최적화한 SGMCMC에 비해 샘플링 효율성(sampling efficiency)이 크게 개선되었습니다. 이로 인해 계산 비용을 크게 증가시키지 않고도 더 우수한 성능을 달성할 수 있습니다.



### Measuring Visual Sycophancy in Multimodal Models (https://arxiv.org/abs/2408.09111)
- **What's New**: 이 논문에서는 다중 모달 언어 모델에서 나타나는 '시각적 아첨(visual sycophancy)' 현상을 소개하고 분석합니다. 이는 모델이 이전의 지식이나 응답과 상충할지라도 시각적으로 제시된 정보를 지나치게 선호하는 경향을 설명하는 용어입니다. 연구 결과, 모델이 시각적으로 표시된 옵션을 선호하는 경향이 있음을 발견했습니다.

- **Technical Details**: 이 연구는 여러 가지 모델 아키텍처에서 일관된 패턴으로 시각적 아첨을 정량화할 수 있는 방법론을 적용했습니다. 주요 기술로는 이미지와 선택지의 시각적 강조를 기반으로 한 실험 설계를 통해 모델 응답이 어떻게 변화하는지를 측정했습니다. 실험은 과거 지식을 유지하면서도 시각적 단서를 통해 얼마나 큰 영향을 받는지를 평가하는 방식으로 진행되었습니다.

- **Performance Highlights**: 모델들이 초기 정답을 제공한 후에도 시각적으로 강조된 옵션으로 응답이 치우치는 경향이 나타났습니다. 이는 모델의 신뢰성에 중대한 한계를 드러내며, 비판적인 의사 결정 맥락에서 이러한 현상이 어떻게 영향을 미칠지에 대한 새로운 의문을 제기합니다.



### Temporal Reversed Training for Spiking Neural Networks with Generalized Spatio-Temporal Representation (https://arxiv.org/abs/2408.09108)
Comments:
          15 pages, 8 figures

- **What's New**: 이번 논문에서는 Spiking Neural Networks (SNNs)의 시공간 성능을 최적화하기 위한 새로운 Temporal Reversed Training (TRT) 방법을 제안합니다. 이 방법은 기존의 SNNs가 겪고 있는 비효율적인 추론과 최적이 아닌 성능 문제를 해결합니다.

- **Technical Details**: Temporal Reversed Training (TRT) 방법은 입력 시간 데이터를 시간적으로 반전시켜 SNN이 원본-반전 일관된 출력 로짓(logits)을 생성하도록 유도합니다. 이를 통해 perturbation-invariant representations를 학습하게 됩니다. 정적 데이터의 경우, 스파이크 신경세포의 고유한 시간적 특성을 이용하여 스파이크 기능의 시간적 반전을 적용합니다. 원래의 스파이크 발화율과 시간적으로 반전된 스파이크 발화율을 요소별 곱(element-wise multiplication)을 통해 혼합하여 spatio-temporal regularization을 수행합니다.

- **Performance Highlights**: 정적 및 신경형 물체/행동 인식, 3D 포인트 클라우드 분류 작업에서 광범위한 실험을 통해 제안된 방법의 효과성과 일반화를 입증하였습니다. 특히, 단 두 개의 타임스텝으로 ImageNet에서 74.77% 그리고 ModelNet40에서 90.57% 정확도를 달성하였습니다.



### Classifier-Free Guidance is a Predictor-Corrector (https://arxiv.org/abs/2408.09000)
Comments:
          AB and PN contributed equally

- **What's New**: 이 논문에서는 classifier-free guidance (CFG)의 이론적 기초를 조사하며, CFG가 DDPM과 DDIM과 어떻게 다르게 상호작용하는지를 보여줍니다. CFG가 gamma-powered 분포를 생성하지 못한다는 점을 명확히 하며, 이를 통한 이론적 이해를 제공합니다.

- **Technical Details**: 본 연구에서는 CFG가 예측-수정 방법(prediction-correction method)의 일종으로 작용함을 입증합니다. 이 방법은 노이즈 제거와 선명도 조정을 번갈아 수행하며, 이를 predictor-corrector guidance (PCG)라고 정의합니다. CFG의 성능을 SDE 한계(SDE limit)에서 분석하여, 특정 gamma 값에 대해 DDIM 예측기와 Langevin 동역학 수정기를 결합한 형태와 동등함을 보여줍니다.

- **Performance Highlights**: CFG는 대부분의 현대 텍스트-투-이미지(diffusion model) 모델에서 사용되며, 많은 유의한 개선을 보이고 있습니다. 하지만 기존의 이해와는 달리, CFG가 항상 이론적으로 보장된 결과를 생성하는 것은 아니며, DDPM과 DDIM간의 분포 생성 차이를 명확히 밝혀내는 중요한 이론적 발견을 제공합니다.



### Ask, Attend, Attack: A Effective Decision-Based Black-Box Targeted Attack for Image-to-Text Models (https://arxiv.org/abs/2408.08989)
- **What's New**: 이 논문에서는 이미지-텍스트 모델에 대한 의사 결정 기반 블랙박스(targeted black-box) 공격 접근 방식을 제안합니다. 특히, 공격자는 모델의 최종 출력 텍스트에만 접근할 수 있으며, 이를 통해 목표 텍스트와 관련된 공격을 수행하게 됩니다. 또한, 이 연구는 'Ask, Attend, Attack'의 세 단계로 구성된 공격 프로세스를 통해 최적화 문제를 해결하는 방법을 제시합니다.

- **Technical Details**: 제안된 방법론은 'Ask, Attend, Attack'(AAA) 프로세스를 포함합니다. 'Ask' 단계에서 공격자는 특정 의미를 충족하는 타겟 텍스트를 생성하도록 안내받고, 'Attend' 단계에서는 공격을 위한 중요한 이미지 영역을 파악하여 탐색 공간을 줄이며, 마지막 'Attack' 단계에서는 진화 알고리즘(evolutionary algorithm)을 사용하여 목표 텍스트와 출력 텍스트 간의 불일치를 최소화하는 방식으로 공격을 수행합니다. 이는 공격자가 출력 텍스트를 목표 텍스트에 가깝게 변경할 수 있게 합니다.

- **Performance Highlights**: 실험 결과, Transformer 기반과 CNN+RNN 기반의 대표 이미지-텍스트 모델인 VIT-GPT2와 Show-Attend-Tell에서 제안된 AAA 방법이 기존의 gray-box 방법들보다 우수한 공격 성능을 보임을 확인하였습니다. 본 연구는 결정 기반 블랙박스 공격이 기존 방법에 비해 효과적인 해결책이 될 수 있음을 강조합니다.



### A Survey of Trojan Attacks and Defenses to Deep Neural Networks (https://arxiv.org/abs/2408.08920)
- **What's New**: 최근 연구에 따르면 Deep Neural Networks(DNNs)는 Neural Network Trojans(NN Trojans)라는 악의적인 공격에 취약하다는 사실이 드러났습니다. 이 논문에서는 DNN을 겨냥한 Trojan 공격과 이를 완화하기 위한 대응 방법을 종합적으로 조사합니다. 또한, 실제 상황에서의 적용 가능성을 평가하여, 이 문제가 얼마나 긴급한지를 강조하고 있습니다.

- **Technical Details**: 이 논문에서는 다양한 NN Trojan 공격을 다섯 가지 범주로 분류합니다: 데이터 기반 Trojans, 모델 기반 Trojans, 라이브러리 기반 Trojans, 신경원 Trojans, 하드웨어 기반 Trojans. 각 범주는 DNN 모델의 구조와 데이터의 관계에 따라 차별화된 방법으로 Trojans를 생성합니다. 예를 들어, 데이터 기반 Trojans는 데이터셋을 조작하여 NN Trojans를 생성합니다.

- **Performance Highlights**: 이 연구는 NN Trojans의 공격 및 방어 전략에서 최신 기술을 요약하고, 그 접근 방식을 비교 분석합니다. 이를 통해 공격에 대한 이해를 높이고, DNN 모델의 신뢰성을 보장하기 위한 향후 개발 방향성을 제공합니다.



### Tree species classification at the pixel-level using deep learning and multispectral time series in an imbalanced contex (https://arxiv.org/abs/2408.08887)
- **What's New**: 본 논문은 Sentinel-2 다중 스펙트럼 위성 이미지 시계열을 사용하여 수종 분류(Tree Species Classification)를 수행하는 연구입니다. 여러 연구에서 강조된 것처럼, 원격 감지 시계열을 활용하여 이러한 지도를 생성하는 것이 중요하나 기존의 방법들은 주로 Random Forest (RF) 알고리즘과 식생 지수에 의존하고 있습니다. 이 연구는 딥러닝 모델(DL Models)을 사용함으로써 분류 성능이 현저히 향상될 수 있음을 보여줍니다.

- **Technical Details**: 연구는 프랑스 중부 지역에서 10종의 나무를 대상으로 하였으며, 3개의 서로 다른 딥러닝 아키텍처를 활용하여 전반적인 정확도(Overall Accuracy, OA) 약 95%와 F1-macro 점수 약 80%를 달성했습니다. 반면, RF 알고리즘을 사용할 경우 OA는 93%, F1은 60%로, 소수 클래스의 분류 정확도가 부족함을 나타냅니다. 평범한 다층 퍼셉트론(Multilayer Perceptron)조차 배치 정규화(Batch Normalization)와 충분한 파라미터 수로 경쟁력이 있음을 보여줍니다.

- **Performance Highlights**: 본 연구는 DL 모델들이 자연스럽게 불균형 데이터에 강인하다는 것을 증명하며, 맞춤형 기술을 사용하여 유사한 결과를 얻을 수 있다는 것을 보여줍니다. 이 프레임워크는 제한된 참조 데이터로도 대부분의 시나리오에서 쉽게 구현될 수 있는 강력한 기준을 제공합니다.



### U-MedSAM: Uncertainty-aware MedSAM for Medical Image Segmentation (https://arxiv.org/abs/2408.08881)
- **What's New**: U-MedSAM 모델은 MedSAM 모델과 불확실성 인식 손실 함수(uncertainty-aware loss function), Sharpness-Aware Minimization(SharpMin) 옵티마이저를 통합하여 정확한 마스크 예측을 개선합니다.

- **Technical Details**: U-MedSAM은 픽셀 기반 손실(pixel-based loss), 지역 기반 손실(region-based loss), 분포 기반 손실(distribution-based loss)을 결합한 불확실성 인식 손실을 사용하여 세그멘테이션(segmentation) 정확성과 강인성을 높입니다. SharpMin 옵티마이저는 손실 경량(loss landscape) 내에서 플랫 최소값(flat minima)을 찾아 과적합(overfitting)을 줄이며 일반화(generalization)를 향상시킵니다.

- **Performance Highlights**: U-MedSAM은 CVPR24 MedSAM on Laptop 챌린지에서 기존 모델에 비해 86.10%의 DSC(Dice Similarity Coefficient)를 기록하며 뛰어난 성과를 보였습니다. 이는 불확실성을 인식하는 손실 함수와 SharpMin 최적화 기법을 통해 가능한 결과입니다.



### LEGENT: Open Platform for Embodied Agents (https://arxiv.org/abs/2404.18243)
Comments:
          ACL 2024 System Demonstration

- **What's New**: 이번 논문에서는 LEGENT라는 새로운 오픈 소스 플랫폼을 소개하며, 이를 통해 사용자 친화적인 인터페이스와 대규모 데이터 생성 파이프라인을 기반으로 LLMs(대형 언어 모델) 및 LMMs(대형 다중 모달 모델)을 활용한 임바디드 에이전트의 개발을 용이하게 합니다.

- **Technical Details**: LEGENT는 다채롭고 상호작용이 가능한 3D 환경을 제공하며, 사용자가 이해하기 쉬운 인터페이스와 함께 최첨단 알고리즘을 통해 시뮬레이션된 세계에서 감독을 활용하여 대규모 데이터 생성을 지원합니다. 또한, 인간과 유사한 에이전트가 능동적으로 언어 상호작용을 수행할 수 있습니다.

- **Performance Highlights**: LEGENT에서 생성된 데이터로 훈련된 초기 비전-언어-행동 모델은 GPT-4V를 초과하는 성능을 보였으며, 전통적인 임바디드 환경에서는 경험하지 못했던 강력한 일반화 능력을 보여줍니다.



### Latency-Aware Generative Semantic Communications with Pre-Trained Diffusion Models (https://arxiv.org/abs/2403.17256)
Comments:
          Accepted for publication in IEEE Wireless Communication Letters

- **What's New**: 최근 생성형 AI 모델들이 텍스트 프롬프트와 조건 신호를 사용하여 고품질의 자연 신호를 합성하는 데 성공을 거두었습니다. 이 연구에서는 사전 훈련된 생성형 모델을 활용한 지연 인식(지연-aware) 의미 소통 프레임워크를 개발하였습니다.

- **Technical Details**: 이 프레임워크는 입력 신호에 대해 다중 모드 의미 분해(multi-modal semantic decomposition)를 수행하고, 각 의미 흐름을 전송할 때 의도에 따라 적절한 인코딩 및 통신 방식으로 전송합니다. 프롬프트는 재전송 기반 스킴을 채택해 신뢰성 있는 전송을 보장하고, 다른 의미 모달리티는 변화하는 무선 채널에 강건함을 제공하기 위해 적응형 변조/인코딩 스킴을 사용합니다.

- **Performance Highlights**: 시뮬레이션 결과는 초저속, 저지연 및 채널 적응형 의미 통신의 가능성을 보여줍니다.



New uploads on arXiv(cs.AI)

### Learning Brave Assumption-Based Argumentation Frameworks via ASP (https://arxiv.org/abs/2408.10126)
Comments:
          Extended version of the paper accepted at the 27th European Conference on Artificial Intelligence (ECAI 2024); Paper ID: M1488 (this https URL)

- **What's New**: 이 논문에서는 기존의 Assumption-based Argumentation (ABA) 프레임워크를 사전 준비 없이 교육 데이터로부터 자동으로 학습하는 방법에 초점을 맞추고 있습니다.

- **Technical Details**: 우리는 안정적 확장에 기반한 용감한 추론(brave reasoning) 관점에서 ABA 학습 문제를 새롭게 정의하고, Rote Learning, Folding, Assumption Introduction과 Fact Subsumption과 같은 변환 규칙을 이용한 새로운 알고리즘을 제시합니다. 이 알고리즘은 Answer Set Programming(ASP)을 활용하여 구현됩니다.

- **Performance Highlights**: 실험 결과, 제안한 ASP-ABAlearn 시스템은 최첨단의 ILASP 시스템과 비교하여 우수한 성능을 보임을 입증하였습니다.



### Geometry Informed Tokenization of Molecules for Language Model Generation (https://arxiv.org/abs/2408.10120)
- **What's New**: 이 논문은 3D 공간에서의 분자 생성 문제를 해결하기 위해 새로운 토크나이징 방법인 Geo2Seq를 제안합니다. 이 방법은 분자 기하학을 1D 이산 시퀀스로 변환하는 데 도움을 주며, 특히 LMs(언어 모델)의 처리 효율성을 활용합니다.

- **Technical Details**: Geo2Seq는 두 가지 주요 단계를 포함하고 있습니다: 1) 정규화 레이블링(canonical labeling)과 2) 불변 구형 표현(invariant spherical representation). 이러한 단계를 통해 분자 기하학의 기하학적 및 원자 충실도를 유지합니다. 또한, S⁢E⁢(3)𝑆𝐸3SE(3) 불변성을 보장하며, 이를 통해 분자 데이터의 복잡한 그래프 구조를 처리할 수 있게 됩니다.

- **Performance Highlights**: Geo2Seq를 활용한 실험에서 다양한 LMs는 원하는 특성을 가진 3D 분자를 생성하는 데 뛰어난 성능을 보여주었으며, 특히 조건부 생성 작업에서 기존의 확산 모델(diffusion models)을 크게 능가했습니다. 이러한 결과는 약물 개발 및 소재 과학과 같은 분야에서의 새로운 분자 발견과 설계에 기여할 것으로 기대됩니다.



### Enhancing Reinforcement Learning Through Guided Search (https://arxiv.org/abs/2408.10113)
Comments:
          Accepted Paper at ECAI 2024; Extended Version

- **What's New**: 이 논문은 Off-Policy 설정에서 Markov Decision Problem의 성능을 향상시키기 위해 Offline Reinforcement Learning (RL)에서 영감을 받으려는 시도를 합니다.

- **Technical Details**: 이 연구에서는 정책 학습 동안 불확실성을 완화하고 잠재적인 정책 오류를 줄이기 위해 참고 정책(reference policy)에 가까운 위치를 유지하는 일반적인 접근 방식을 채택했습니다. Monte Carlo Tree Search (MCTS) 알고리듬을 가이드로 사용하는 것이 주요 초점이며, 이 알고리듬은 단일 및 이인용 환경에서 균형에 수렴하는 능력으로 유명합니다.

- **Performance Highlights**: MCTS를 RL 에이전트의 가이드로 활용했을 때, 개별적으로 각 방법을 사용할 때보다 성능이 크게 향상된 것을 관찰했습니다. 실험은 Atari 100k 벤치마크에서 수행되었습니다.



### PLUTUS: A Well Pre-trained Large Unified Transformer can Unveil Financial Time Series Regularities (https://arxiv.org/abs/2408.10111)
- **What's New**: PLUTUS는 1000억 개의 관측치로 훈련된 최초의 대규모 오픈소스 재무 시계열 모델로, 노이즈가 많은 금융 환경에서 높은 효율성을 발휘합니다.

- **Technical Details**: PLUTUS는 invertible embedding 모듈과 contrastive learning, autoencoder 기술을 활용하여 원시 데이터를 patch embedding과 약 일대일 매핑을 생성합니다. TimeFormer라는 주의(attention) 기반 아키텍처가 핵심으로 사용되며, 다양한 시간적 및 변수 차원에서의 특징을 포착합니다.

- **Performance Highlights**: PLUTUS는 여러 다운스트림 작업에서 최고 성능을 기록하며, 강력한 전이 가능성을 입증하고 있습니다. 그 성능은 금융 분야의 새로운 기준을 설정합니다.



### ARMADA: Attribute-Based Multimodal Data Augmentation (https://arxiv.org/abs/2408.10086)
- **What's New**: 본 논문에서는 시각적 속성을 지식 기반의 조작을 통해 조화롭게 다루는 새로운 다중 모달 데이터 증강 방법인 ARMADA를 제안합니다. 이 방법은 기존의 데이터 증강 방식들이 가지고 있는 의미적 불일치나 비현실적인 이미지 생성 문제를 해결하려고 합니다.

- **Technical Details**: ARMADA는 원본 텍스트 데이터에서 개체(entity)와 시각적 속성을 추출하고, 이를 기반으로 지식 기반(KB)과 대형 언어 모델(LLM)의 지도를 받아 대체 값을 검색하여 이미지를 편집합니다. 이 과정에서, KB에서 지식 기반의 속성을 추출하여 의미론적으로 일관 있으며 독특한 이미지-텍스트 쌍을 생성합니다. 또한, LLM의 일반 상식 지식을 활용해 주변 시각적 속성을 조정하여 원래의 개체를 더욱 견고하게 표현합니다.

- **Performance Highlights**: 본 연구는 4가지 하류 작업에서 실험을 통해 ARMADA 프레임워크가 고품질 데이터를 생성하고 모델 성능을 향상시킬 수 있음을 보여주었습니다. 이미지-텍스트 검색, 시각적 질문 응답(VQA), 이미지 캡션 생성, 세밀한 이미지 분류 작업에서의 향상된 성능이 특히 강조됩니다.



### The Practimum-Optimum Algorithm for Manufacturing Scheduling: A Paradigm Shift Leading to Breakthroughs in Scale and Performanc (https://arxiv.org/abs/2408.10040)
- **What's New**: Practimum-Optimum (P-O) 알고리즘은 복잡한 실제 비즈니스 문제를 해결하기 위한 자동 최적화 제품 개발에서 패러다임 전환을 나타냅니다. 이는 깊은 비즈니스 도메인 전문 지식을 활용하여 다양한 '사고 방식'을 가진 가상 인간 전문가(VHE) 에이전트를 생성합니다.

- **Technical Details**: P-O 알고리즘은 강화 기계 학습(reinforced machine learning) 알고리즘을 통해 VHE 일정의 강점과 약점을 학습하고, 이를 바탕으로 Demand Set에서 보상과 처벌을 변경하여 작업의 시간 및 자원 할당 우선순위를 수정합니다. 이러한 접근 방식은 다음 iteration에서 일정의 새로운 부분을 탐색하도록 유도합니다.

- **Performance Highlights**: P-O 알고리즘은 Plataine Scheduler의 핵심으로, 클릭 한 번으로 복잡한 제조 작업을 위해 30,000-50,000개의 작업을 정기적으로 스케줄링합니다. 이는 전통적인 알고리즘과 비교하여 훨씬 높은 스케줄링 성능을 제공합니다.



### MSDiagnosis: An EMR-based Dataset for Clinical Multi-Step Diagnosis (https://arxiv.org/abs/2408.10039)
- **What's New**: 이 논문은 임상 진단 과정의 복잡성을 반영하여 여러 단계의 진단 작업을 제안하고, 이를 위한 새로운 데이터셋인 MSDiagnosis를 구축합니다. 기존의 단일 단계 진단 방식과는 달리, 이 연구는 임상의 실무에 적합한 다단계 절차를 반영합니다.

- **Technical Details**: 제안된 MSDiagnosis 데이터셋은 2,225개의 의료 기록으로 구성되어 있으며, 이 데이터셋에는 주 진단, 감별 진단, 최종 진단 관련 질문이 포함되어 있습니다. 연구자들은 LLM(큰 언어 모델)의 진단 결과를 스스로 평가하고 조정할 수 있는 고유한 프레임워크를 제안하여, 전방 추론(forward inference), 후방 추론(backward inference), 반영(reflection), 그리고 정제(refinement) 과정을 통합하고 있습니다.

- **Performance Highlights**: 광범위한 실험 결과는 제안된 방법이 효과적임을 보여줍니다. 이 연구는 오픈 소스 및 클로즈드 소스 LLM을 사용한 복잡한 다단계 진단 작업에서 성능을 종합적으로 평가하였으며, 실험 결과는 제안된 프레임워크의 우수성을 입증하고 있습니다.



### Deterministic Policy Gradient Primal-Dual Methods for Continuous-Space Constrained MDPs (https://arxiv.org/abs/2408.10015)
- **What's New**: 이 논문은 연속적인 상태-행동 공간을 갖는 제약된 마르코프 결정 프로세스(Constraints Markov Decision Processes, MDPs)에서 결정론적 최적 정책을 계산하는 문제를 다룹니다. 특히, 제약된 MDP를 위한 결정론적 정책 기울기 메서드는 기존 방법의 응용에 어려움이 있어 새로운 방법론이 필요하다는 점에 착안하여, 결정론적 정책 기울기 주술적 방법(Deterministic Policy Gradient Primal-Dual, D-PGPD)을 개발했습니다.

- **Technical Details**: D-PGPD 알고리즘은 제약된 MDP의 Lagrangian을 정규화하여 결정론적 정책을 업데이트하는 방법론을 제시합니다. quadratically regularized gradient ascent 및 descent 단계를 통해 결정론적 정책과 이중 변수를 업데이트하며, 이 알고리즘의 반복 과정이 최적의 정규화된 원-이중 쌍으로 수렴함을 보였습니다. 또한 D-PGPD는 함수 근사를 포함하여 접근할 수 있으며, 이는 이중 반복 과정에서 수렴 오차를 포함한 최적의 정규화된 원-이중 쌍으로의 수렴을 보장합니다.

- **Performance Highlights**: 이 방법론은 로봇 네비게이션 및 유체 제어와 같은 두 가지 연속 제어 문제에서 효과적으로 구현되었습니다. 연구에 따르면, D-PGPD는 이러한 제약된 상황에서 연속 공간을 다루는 결정론적 정책 탐색 방법으로 성공적으로 기능하고 있으며, 기존의 제한된 방법론에 비해 더 나은 성능을 보였습니다.



### Towards a Knowledge Graph for Models and Algorithms in Applied Mathematics (https://arxiv.org/abs/2408.10003)
Comments:
          Preprint submitted to the 18th International Conference on Metadata and Semantics Research 2024

- **What's New**: 본 논문은 수학적 모델과 알고리즘의 새로운 온톨로지(ontology)를 통합하고 확장하여 데이터와 알고리즘 간의 관계를 보다 세분화하고 명확히 하기 위한 지식 그래프(knowledge graph)를 개발했습니다. 이를 통해 FAIR 원칙에 따른 연구 데이터 관리가 가능해졌습니다.

- **Technical Details**: 두 개의 온톨로지를 통합하여, 모델링 과정에서 발생하는 계산 작업(computational tasks)과 알고리즘 작업(algorithmic tasks)의 관계를 수립하였습니다. 또한, 기본 양(base quantities)과 특정 사용 사례 양(specific use case quantities)을 구분하는 새로운 클래스를 도입하고, 메타데이터(metadata)를 모델과 알고리즘에 추가할 수 있도록 개선하였습니다.

- **Performance Highlights**: 현재까지 250개 이상의 응용 수학 연구 자산이 지식 그래프에 통합되어 있으며, 이는 수학적 모델과 알고리즘의 구체적인 워크플로우를 명확히 표현할 수 있게 해 줍니다. 이 연구는 수학적 성질을 기반으로 하는 실현 가능한 솔루션 알고리즘을 찾기 위한 중요한 기초 자료로 활용될 것입니다.



### Contextual Importance and Utility in Python: New Functionality and Insights with the py-ciu Packag (https://arxiv.org/abs/2408.09957)
Comments:
          In Proceedings of XAI 2024 Workshop of 33rd International Joint Conference on Artificial Intelligence (IJCAI 2024), Jeju, South Corea

- **What's New**: 이 논문은 Contextual Importance and Utility (CIU) 모델에 대한 Python 구현(	exttt{py-ciu})을 소개합니다. 기존 설명 가능 AI(XAI) 방법들과 차별화된 특성을 갖춘 CIU의 기능을 시연하여, 연구자들이 여러 AI 시스템에서 이 방법을 손쉽게 적용할 수 있도록 돕습니다.

- **Technical Details**: CIU는 모델 비특정적(post-hoc) 설명 방법으로, 특성 중요도(feature importance)와 특성 영향력(feature influence)의 차이를 강조합니다. 기존의 LIME 또는 SHAP와 달리 CIU는 이론적으로 독립성이 있어 잠재적 영향력(plot) 등을 통해 독창적인 설명을 제공합니다. Python 구현(	exttt{py-ciu})은 표 형식(tabular data)을 위한 것으로, CIU의 이론과 기능을 설명하기 위한 두 가지 주요 목표를 갖습니다.

- **Performance Highlights**: CIU는 LIME 및 SHAP와 유사한 설명을 생성하는 기능을 갖추고 있으며, 별도의 기능성을 통해 기존 방법들보다 높은 설명 가능성을 제공합니다.



### Principle Driven Parameterized Fiber Model based on GPT-PINN Neural Network (https://arxiv.org/abs/2408.09951)
- **What's New**: 이 논문은 Beyond 5G 통신의 필요를 충족시키기 위해 데이터 기반 (data driven) 인공지능 모델을 활용하여 전통적인 split step Fourier 방법보다 훨씬 빠른 속도로 섬유 전송에서의 펄스 진화를 예측하는 모델을 제안합니다.

- **Technical Details**: 제안된 모델은 비선형 슈뢰딩거 방정식 (Nonlinear Schodinger Equation, NLSE)을 손실 함수에 추가함으로써 물리적 해석 가능성 (physical interpretability)을 높입니다. 이 모델은 예측된 NLSE 해를 전송 조건에 따라 여러 개의 고유 해 (eigen solutions)의 선형 결합으로 분해하여, 다양한 전송 조건에서 재학습 (re-trained)할 필요를 크게 줄입니다.

- **Performance Highlights**: 모델의 계산 복잡성 (computational complexity)은 기존의 split step Fourier 방법의 0.0113% 및 기존의 원리에 기반한 섬유 모델의 1%에 달하며, 높은 계산 효율성 (computing efficiency)을 보여줍니다.



### Fiber Transmission Model with Parameterized Inputs based on GPT-PINN Neural Network (https://arxiv.org/abs/2408.09947)
- **What's New**: 이 연구에서는 파라미터화된 입력을 위한 단거리 전송 모델을 제안하였습니다. 이 모델은 기존의 원리에 의해 구동되는 섬유(optical fiber) 모델을 기반으로 합니다.

- **Technical Details**: 제안된 모델은 Reduced Basis Expansion Method와 Nonlinear Schrodinger Equations의 파라미터화된 계수를 변환하여, 다양한 비트 전송률에 따른 보편적인 솔루션을 제공합니다. 이 과정에서 전체 모델을 재훈련할 필요가 없으며, 전송된 신호를 미리 수집하지 않아도 효과적인 훈련이 가능합니다.

- **Performance Highlights**: 모델의 실험 결과는 2Gbps에서 50Gbps 범위의 On-Off Keying 신호 작업에 대해 신뢰성이 높음을 보여주었습니다. 이 모델은 계산 효율성과 물리적 배경 모두에서 뚜렷한 장점을 갖추고 있습니다.



### Microscopic Analysis on LLM players via Social Deduction Gam (https://arxiv.org/abs/2408.09946)
Comments:
          Under review, 10 pages

- **What's New**: 최근 연구들은 대형 언어 모델(LLMs)을 활용한 자율 게임 플레이어 개발에 주목하고 있으며, 특히 사회적 추론 게임(SDGs)에서의 게임 플레이 능력 평가에 대한 새로운 접근 방식을 제안하고 있습니다.

- **Technical Details**: 기존 연구들이 게임 플레이 능력을 전반적인 게임 결과를 통해 평가한 반면, 본 연구에서는 SpyFall 게임의 변형인 SpyGame을 사용하여 4개의 LLM을 분석하고, 이를 통해 특화된 8개의 미세 지표(microscopic metrics)를 도입하였습니다. 이 지표들은 의도 식별(intent identification)과 변장(camouflage) 능력을 평가하는 데 더 효과적임을 보여줍니다.

- **Performance Highlights**: 본 연구에서는 4개의 주요 그리고 5개의 하위 카테고리를 통해 LLM의 비정상적 추론 패턴을 확인하였으며, 이에 대한 정량적 결과들을 정성적 분석과 연계하여 검증하였습니다.



### LCE: A Framework for Explainability of DNNs for Ultrasound Image Based on Concept Discovery (https://arxiv.org/abs/2408.09899)
- **What's New**: 본 논문에서는 의학 이미지 분석에서 Deep Neural Networks (DNNs)의 설명 가능성을 높이기 위한 새로운 프레임워크인 Lesion Concept Explainer (LCE)를 제안합니다. 기존의 attribution 방법과 concept-based 방법의 결합을 통해 초음파 이미지에 대한 의미 있는 설명을 가능하게 합니다.

- **Technical Details**: LCE는 Segment Anything Model (SAM)을 기반으로 하며, 대규모 의학 이미지 데이터로 파인튜닝되었습니다. Shapley value를 사용하여 개념 발견과 설명을 수행하며, 새로운 평가 지표인 Effect Score를 제안하여 설명의 신뢰성과 이해 가능성을 동시에 고려합니다. 이 프레임워크는 ResNet50 모델을 사용하여 공공 및 사적 유방 초음파 데이터셋(BUSI 및 FG-US-B)에서 평가되었습니다.

- **Performance Highlights**: LCE는 기존의 설명 가능성 방법들에 비해 우수한 성능을 나타내며, 전문가 초음파 검사자들에 의해 높은 이해도를 기록했습니다. 특히, LCE는 유방 초음파 데이터의 세밀한 진단 작업에서도 안정적이고 신뢰할 수 있는 설명을 제공하였으며, 추가 비용 없이 다양한 진단 모델에 적용 가능하다는 장점이 있습니다.



### Uncertainty Quantification of Pre-Trained and Fine-Tuned Surrogate Models using Conformal Prediction (https://arxiv.org/abs/2408.09881)
- **What's New**: 이 논문은 복잡한 수치 및 실험 모델링 작업에 대한 단기적이고 저비용의 근사치를 제공할 수 있는 데이터 기반 서 surrogate 모델의 불확실성을 정량화할 수 있는 새로운 방법론인 Conformal Prediction (CP) 프레임워크를 제시합니다. 이러한 프레임워크는 모델 독립적으로 작동하며, 거의 제로에 가까운 계산 비용으로 마진 보장을 제공합니다. 이를 통해 다양한 spatio-temporal 모델에 대해 유효한 오차 막대를 생성할 수 있습니다.

- **Technical Details**: 우리는 CP 프레임워크를 통해 사전 훈련 및 세심하게 조정된 신경망 기반 서 surrogate 모델의 예측에 대해 보장된 오차 막대를 제공하는 방법에 대한 실증 연구를 실시했습니다. 이 연구에서는 spatio-temporal 영역에서 보장된 커버리지를 제공할 수 있음을 입증하며, 예측하는 데이터의 차원 수에 관계없이 유효성을 확보할 수 있는 방법을 제시합니다.

- **Performance Highlights**: 실험 결과, 예측 데이터가 훈련 데이터와 다른 물리적 설정을 나타내더라도 의미 있는 오차 막대를 제공할 수 있는 능력이 확인되었습니다. 전체 실험을 통해, CP 방법론이 다양한 신경망 기반 서 surrogate 모델에 대해 보장된 커버리지를 추구할 수 있음을 보여주며, 이는 산업 수준의 안전-critical 적용 분야에서도 중요하게 활용될 수 있습니다.



### Demystifying Reinforcement Learning in Production Scheduling via Explainable AI (https://arxiv.org/abs/2408.09841)
- **What's New**: 본 논문에서는 Deep Reinforcement Learning (DRL) 에이전트의 스케줄링 결정 reasoning 을 설명하기 위해 SHAP (DeepSHAP)와 Captum (Input x Gradient)이라는 두 가지 설명 가능한 AI (xAI) 프레임워크를 체계적으로 적용하는 사례 연구를 수행합니다. 이를 통해 xAI 문헌의 한계와 개선점을 제시합니다.

- **Technical Details**: Deep Reinforcement Learning (DRL)은 강화 학습 (Reinforcement Learning)의 하위 분야로, 복잡한 스케줄링 문제를 해결하는 데 유용합니다. DRL은 딥 뉴럴 네트워크 (DNNs)를 활용하여 상태를 행동으로 매핑하는 함수를 근사하고, 관련 보상을 추정합니다. 논문에서 제안하는 설명 가능한 AI 접근 방식은 가설 기반 워크플로우(hypotheses-based workflow)를 통해 에이전트의 보상 가설과 일치하는지 검토합니다.

- **Performance Highlights**: 제안된 xAI 워크플로우는 스케줄링 활용 사례에 적용 가능하며, 설명의 반복적인 검증을 강조하여 실제 도메인에서의 성과를 높이는 데 기여할 수 있습니다.



### Minor DPO reject penalty to increase training robustness (https://arxiv.org/abs/2408.09834)
Comments:
          8 pages, 19 figures

- **What's New**: 이 논문은 인간의 선호에 대한 학습(Learning from human preference) paradigm을 기반으로 한 대규모 언어 모델(LLM)의 미세 조정 방법을 분석합니다. 최근에 Direct Preference Optimization (DPO) 방법이 제안되었는데, 이는 간소화된 RL-free 방법으로 LLM의 선호도 최적화 문제를 해결하려고 합니다.

- **Technical Details**: DPO는 선택된 데이터 쌍과 거부된 데이터 쌍의 상대 로그 확률을 암묵적 보상 함수로 모델링하고 단순한 이진 교차 엔트로피 목표를 사용하여 LLM 정책을 최적화합니다. 이 과정에서 중요한 하이퍼파라미터인 β(베타)의 내부 작동 메커니즘이 분석됩니다. 또한 MinorDPO라는 새로운 접근법이 제안되어 원래의 RL 알고리즘과 더 잘 정렬되고 선호도 최적화 과정의 안정성을 증가시킵니다.

- **Performance Highlights**: MinorDPO는 DPO보다 더 뛰어난 성능을 보이며, 이전 방법들이 가지는 취약성을 극복하면서도 추가적인 하이퍼파라미터 없이 안정성과 견고성을 증가시키는 데 성공했습니다.



### TDNetGen: Empowering Complex Network Resilience Prediction with Generative Augmentation of Topology and Dynamics (https://arxiv.org/abs/2408.09825)
- **What's New**: 본 논문은 복잡한 네트워크의 회복력(resilience) 예측을 위한 새로운 프레임워크인 TDNetGen을 소개합니다. 이 프레임워크는 네트워크 토폴로지(topology)와 동역학(dynamics)에 대한 생성적 데이터 증대를 통해 라벨이 부족한 상황에서의 문제를 해결하도록 설계되었습니다.

- **Technical Details**: TDNetGen은 그래프 컨볼루션 네트워크(graph convolutional network) 기반의 토폴로지 인코더와 트랜스포머(transformer) 기반의 동적 인코더를 통합하여 네트워크 구조와 동역학 간의 복잡한 관계를 캡처합니다. 이 프레임워크는 라벨이 없는 데이터에서의 해로운 분포를 포착하고 이를 통해 각 클래스 레이블의 조건부 분포를 얻는 방식으로 작동하며, 이는 상황에 따른 예측 모델의 효율성을 높입니다.

- **Performance Highlights**: TDNetGen은 세 개의 네트워크 데이터셋에서 85%-95%의 높은 예측 정확도를 달성하며, 데이터 부족 상황에서도 98.3%의 성능을 유지할 수 있는 강력한 증대 능력을 보여줍니다. 이는 기존의 최첨단 방법들보다 월등히 개선된 성능을 의미합니다.



### World Models Increase Autonomy in Reinforcement Learning (https://arxiv.org/abs/2408.09807)
- **What's New**: 본 논문에서는 리셋이 없는 환경에서의 모델 기반 강화 학습(Model-based Reinforcement Learning, MBRL)의 우수성을 입증하고, 모델 기반 리셋 없는 에이전트(MoReFree)를 제안하여 성능을 개선합니다.

- **Technical Details**: MoReFree는 주요 메커니즘인 탐색(exploration)과 정책 학습(policy learning)을 조정하여 리셋이 없는 작업에서 효과적으로 작업 관련 상태(task-relevant states)를 우선시하게 합니다. 또한, 에이전트는 환경 보상(environmental reward)이나 데모 없이 다양한 리셋 없는 작업에서 우수한 데이터 효율(data efficiency)를 보여줍니다.

- **Performance Highlights**: 모델 기반 접근 방식이 8개의 도전적인 리셋 없는 작업에서 7/8 작업에서 이전 최첨단 방법을 초과하여 성능과 데이터 효율성을 달성했으며, MoReFree는 가장 어려운 3개의 작업에서도 모델 기반 기준선을 초과하는 성과를 보였습니다.



### AutoML-guided Fusion of Entity and LLM-based representations (https://arxiv.org/abs/2408.09794)
- **What's New**: BabelFusion은 지식 기반의 표현을 LLM 기반 표현에 주입하여 분류 작업의 성능을 향상시키는 새로운 접근법을 제안합니다. 이 방법은 또한 자동화된 머신러닝(AutoML)을 활용하여 지식으로 정보가 융합된 표현 공간에서 분류 정확성을 개선할 수 있음을 보여줍니다.

- **Technical Details**: 이 연구에서 제안하는 BabelFusion은 텍스트 문서와 관련된 레이블로 구성된 데이터셋을 기반으로, 지식 그래프에서 관련 항목을 식별하고 벡터 표현을 검색합니다. 이후 각 문서에 대해 이러한 임베딩을 평균하여 단일 벡터로 변환하고, 이를 SVD(슬램값 분해)를 통해 더 낮은 차원의 공간으로 투사하여 문서 표현을 학습합니다.

- **Performance Highlights**: 이 연구는 지식 기반 표현을 LLM에 통합하고, 단순 모델과 결합하여 경쟁력 있는 결과를 달성할 수 있음을 증명하였습니다. 다양한 실험에서는 다섯 개의 강력한 LLM 기초 모델을 사용해, 저차원으로 축소된 표현 공간에서도 낮은 예측 성능 손실로 신속한 분류기가 가능하다는 것을 보여주었습니다.



### GoNoGo: An Efficient LLM-based Multi-Agent System for Streamlining Automotive Software Release Decision-Making (https://arxiv.org/abs/2408.09785)
- **What's New**: GoNoGo는 자동차 소프트웨어 배포 과정을 간소화하기 위해 설계된 LLM(대형 언어 모델) 에이전트 시스템입니다. 이 시스템은 기능적 요구사항과 산업적 제약을 모두 만족시킬 수 있도록 구성되어 있으며, 특히 위험이 민감한 도메인에 맞춤화되었습니다.

- **Technical Details**: GoNoGo에는 두 가지 주요 LLM 에이전트인 Planner와 Actor가 포함됩니다. Planner는 사용자의 쿼리를 이해하고 데이터 분석을 위한 단계별 지침으로 분해합니다. Actor는 이러한 고수준 지침에서 실행 가능한 스크립트를 생성합니다. 이 시스템은 인-context learning을 사용하여 도메인 특정 요구사항을 인코딩합니다.

- **Performance Highlights**: GoNoGo는 Level 2 난이도의 작업에 대해 3-shot 예시를 사용하여 100% 성공률을 달성했으며, 복잡한 작업에서도 높은 성능을 유지합니다. 이 시스템은 더 간단한 작업의 의사결정을 자동화하여 수동 개입의 필요성을 크게 줄입니다.



### MalLight: Influence-Aware Coordinated Traffic Signal Control for Traffic Signal Malfunctions (https://arxiv.org/abs/2408.09768)
Comments:
          Paper accepted to CIKM24 Full Research track

- **What's New**: 이번 연구는 교통 신호의 오작동 문제를 해결하기 위해 새로운 트래픽 신호 제어 프레임워크인 MalLight를 제안합니다. 특히, 본 논문은 Reinforcement Learning 기반의 접근 방식이 오작동 상황에서 효과적으로 작동하는 첫 사례로, 근처의 정상 작동 신호들과 조정된 제어를 통해 교통 혼잡과 충돌 등의 부정적 영향을 완화합니다.

- **Technical Details**: MalLight는 Influence-aware State Aggregation Module(ISAM)과 Influence-aware Reward Aggregation Module(IRAM)을 활용하여 인접 신호의 제어를 원활하게 조정합니다. 이 모델은 그래프 확산 컨볼루션 네트워크를 적용하여 각 에이전트가 자신의 교통 상태뿐만 아니라 다른 신호에 미치는 영향을 인지하도록 설계되었습니다. 연구는 Markov Decision Process(MDP)를 기반으로 하여 오작동 발생 시 필요로 하는 상태(State)와 보상(Reward)을 동적으로 조정합니다.

- **Performance Highlights**: 실험 결과, MalLight는 기존의 전통적인 방법이나 심층 학습 기반 모델들에 비해 뛰어난 성능을 보여주었으며, 특별히 오작동 상황에서의 처리량(throughput)을 최대 48.6% 감소시키는 성과를 이루었습니다.



### HYDEN: Hyperbolic Density Representations for Medical Images and Reports (https://arxiv.org/abs/2408.09715)
- **What's New**: 본 논문에서는 의료 도메인을 특화하여 이미지-텍스트 표현 학습을 위한 새로운 방법인 HYDEN을 제안합니다. 이는 하이퍼볼릭 밀도 임베딩(hyperbolic density embedding)을 사용하여 이미지와 텍스트의 관계를 효과적으로 모델링합니다.

- **Technical Details**: HYDEN은 하이퍼볼릭 공간에서 이미지 텍스트 특징을 밀도 특징으로 매핑하는 방법을 제공합니다. 이 방법은 이미지에서의 전역 특징(global features)과 텍스트 인식 지역 특징(text-aware local features)을 통합하며, 하이퍼볼릭 유사-가우시안 분포(pseudo-Gaussian distribution)를 활용합니다. 또한 캡슐화 손실 함수(encapsulation loss function)를 통해 이미지-텍스트 밀도 분포 간의 부분 순서 관계(partial order relations)를 모델링합니다.

- **Performance Highlights**: 실험 결과 HYDEN은 여러 제로샷(zero-shot) 작업 및 다양한 데이터셋에서 기존 방법들에 비해 우수한 성능을 보였습니다. 본 연구는 의료 이미징 및 관련 의학 보고서에 대한 해석 가능성과 뛰어난 성능을 강조하고 있습니다.



### Partial-Multivariate Model for Forecasting (https://arxiv.org/abs/2408.09703)
Comments:
          25 pages

- **What's New**: 이번 연구에서는 Partial-Multivariate 모델이라고 불리는 새로운 접근법을 소개하여, Univariate와 Complete-Multivariate 모델의 중간 지점을 탐구합니다. 이를 위해 Transformer 기반의 모델인 PMformer를 제안하며, 이 모델은 여러 피처들 사이의 부분적인 관계만을 캡처합니다.

- **Technical Details**: PMformer는 Transformer 아키텍처를 기반으로 하며, 피처를 개별적으로 토큰화하고 선택된 피처에 대한 attention 맵을 계산합니다. 이 모델은 random sampling 및 partitioning을 통한 훈련 알고리즘을 도입합니다. 이론적 분석을 통해 PMformer는 높은 엔트로피(higher entropy)와 더 큰 훈련 데이터셋 크기를 가지므로 완전 다변량 모델에 비해 우수함을 설명합니다.

- **Performance Highlights**: PMformer는 20개의 기존 모델을 대상으로 가장 높은 예측 정확도를 기록하며, 효율적인 inter-feature attention 비용을 보이고, 누락된 피처에 대한 강인성을 나타냅니다. 또한 PMformer에 대한 새로운 추론 기법을 소개하여 예측 성능을 더욱 향상시켰습니다.



### Simulating Field Experiments with Large Language Models (https://arxiv.org/abs/2408.09682)
Comments:
          17 pages, 5 figures, 6 tables

- **What's New**: 이 논문에서는 대형 언어 모델(Large Language Models, LLMs)을 활용하여 현장 실험(field experiments)을 시뮬레이션하는 두 가지 새로운 프롬프트 전략, 즉 관찰자 모드(observer mode)와 참여자 모드(participant mode)를 제안하고 평가합니다.

- **Technical Details**: 관찰자 모드는 주요 결론에 대한 직접 예측을 가능하게 하며, 참여자 모드는 참가자들의 응답 분포를 시뮬레이션합니다. 이 접근법을 통해 여러 잘 인용된 현장 실험 논문 15편을 조사하였고, 특정 시나리오에서 시뮬레이션된 결과와 실제 결과 간의 긍정적인 일치성을 확인했습니다. 또한, LLMs가 부족한 성과를 보이는 주제, 즉 성별 차이와 사회 규범과 관련된 연구를 분석하였습니다.

- **Performance Highlights**: 관찰자 모드에서 66%의 자극 정확도(stimulation accuracy)를 달성하며, 연구자들이 비싼 현장 실험에 착수하기 전에 LLMs를 활용할 수 있는 가능성을 제시합니다. 이 연구는 현장 실험 시뮬레이션에 있어 LLMs의 잠재적 활용성을 확장하고, 연구자들이 LLMs를 실험 도구로 통합할 때 유의해야 할 한계를 분명히 합니다.



### Multi-Agent Reinforcement Learning for Autonomous Driving: A Survey (https://arxiv.org/abs/2408.09675)
Comments:
          23 pages, 6 figures and 2 tables. Submitted to IEEE Journal

- **What's New**: 이 논문은 다중 에이전트 강화 학습(multi-agent RL, MARL) 기술의 최근 발전을 종합하고, 자율주행 및 지능형 교통 시스템에 대한 연구 성과를 정리합니다. 특히, 시뮬레이터의 주요 메트릭스를 제안하고 기존 벤치마크의 특징을 요약합니다.

- **Technical Details**: 이 논문은 MARL의 원리를 활용하여 환경 모델링, 상태 표현, 인식 유닛(perception units), 알고리즘 설계 등에서 자율주행의 최근 연구를 종합적으로 분석합니다. 또한, 자율주행에 객관적인 벤치마크를 설정하고 대규모 시뮬레이터와 데이터셋을 검토합니다.

- **Performance Highlights**: MARL의 발전으로 교통 제어, 에너지 분배, 대규모 로봇 제어 등의 분야에서 혁신적인 결과가 나타나고 있으며, 동시에 자율주행 시스템의 효율성을 극대화할 수 있는 가능성이 확인되었습니다. 이 논문은 다양한 연구를 통해 MARL 기반 자율주행의 기술적 개선점과 해결되지 않은 도전 과제를 명확히 제시하고 있습니다.



### A Comparison of Large Language Model and Human Performance on Random Number Generation Tasks (https://arxiv.org/abs/2408.09656)
- **What's New**: 본 연구는 ChatGPT-3.5가 인간의 무작위 숫자 생성 패턴을 모방하는 방식에 대해 조사합니다. 특히, LLM(대형 언어 모델) 환경에서 기존 인간 RNGT(무작위 숫자 생성 작업)를 조정하여 새로운 연구 결과를 도출했습니다.

- **Technical Details**: 이 연구는 OpenAI의 ChatGPT-3.5 모델을 이용하여 10,000개의 응답을 수집했습니다. 모델에 대해 기존 연구에서 사용된 구술 지침을 기반으로 특별한 사용자 프롬프트를 작성하여 무작위 숫자 시퀀스를 생성하도록 지시했습니다. 생성된 시퀀스의 길이는 평균 269와 표준편차 325로 설정했습니다.

- **Performance Highlights**: 초기 결과에 따르면, ChatGPT-3.5는 인간에 비해 반복적이고 순차적인 패턴을 덜 피하며, 특히 인접 숫자 빈도가 현저히 낮습니다. 이는 LLM이 인간의 인지 편향을 대신하여 무작위 숫자 생성 능력을 향상할 가능성을 제시합니다.



### On the Foundations of Conflict-Driven Solving for Hybrid MKNF Knowledge Bases (https://arxiv.org/abs/2408.09626)
- **What's New**: 이 논문에서는 Hybrid MKNF Knowledge Bases (HMKNF-KBs)에 대한 새로운 이론적 기초를 탐구하며, 충돌을 유도하는 해결기(conflict-driven solver)를 위한 이론적 토대를 제시합니다.

- **Technical Details**: HMKNF-KB는 규칙 기반(rule-based) Knowledge와 온톨로지(ontology) 지식을 통합하여 실제 시스템을 모델링하는 데 사용됩니다. 이 연구는 MKNF 모델을 특성화하는 완료(completion) 및 루프(loop) 공식을 정의하여 nogoods 집합을 형성하고, 이를 통해 충돌을 유도하는 해결기를 개발하려고 합니다.

- **Performance Highlights**: 이 연구의 접근 방식은 MKNF-KBs에 적용 가능한 충돌 유도 알고리즘에서 nogoods가 어떻게 사용될 수 있는지에 대한 개요를 제공합니다. 이를 통해 MKNF-KB 모델 계산의 효율성과 정확성을 높일 수 있는 가능성을 제시합니다.



### Attention is a smoothed cubic splin (https://arxiv.org/abs/2408.09624)
Comments:
          20 pages, 2 figures

- **What's New**: 본 논문은 트랜스포머(transformer) 내의 어텐션 모듈이 부드러운 구간 스플라인(smooth cubic spline)으로 해석될 수 있다는 새로운 통찰을 제시합니다. 이는 클래식 근사 이론(classical approximation theory)에서 깊이 뿌리내린 개념의 자연스러운 발전으로 볼 수 있습니다.

- **Technical Details**: ReLU(리룰) 활성화 함수를 사용하여 어텐션, 마스크드 어텐션(masked attention), 인코더-디코더 어텐션(encoder-decoder attention)은 모두 구간 스플라인(cubic splines)으로 표현될 수 있음을 보여줍니다. 트랜스포머의 모든 구성 요소는 이러한 어텐션 모듈과 전방 신경망(feed forward neural networks)의 조합으로 구성되며, 고차 스플라인을 형성합니다. Pierce-Birkhoff 추측(Pierce-Birkhoff conjecture)을 가정하면 모든 스플라인이 ReLU-활성화 인코더로 해석된다는 사실도 언급됩니다. 

- **Performance Highlights**: 이 연구는 트랜스포머의 본질을 스플라인(splines)이라는 잘 알려진 수학적 객체로 설명할 수 있게 해주며, 이를 통해 트랜스포머의 이해도를 높일 수 있음을 보여줍니다. 결국 스플라인은 적용 수학(applied mathematics)에서 잘 이해된 개체입니다.



### Does Thought Require Sensory Grounding? From Pure Thinkers to Large Language Models (https://arxiv.org/abs/2408.09605)
- **What's New**: 최근 논문은 인공지능(AI)과 철학에서의 사고(thinking)와 감각(sensing)의 관계에 대한 논의를 다룬다.

- **Technical Details**: 논자는 사고가 감각 없이도 가능하다고 주장하며, 감각 없이 이루어질 수 있는 사고의 한계를 제시한다. 또한 대규모 언어 모델(large language models)이 사고하거나 이해할 수 있는지에 대한 직접적인 주장은 하지 않지만, 감각 기초(sensory grounding)라는 주장을 반박한다.

- **Performance Highlights**: 최근 언어 모델의 결과를 활용하여, 감각 기초가 인지 능력(cognitive capacities)을 어떻게 향상시킬 수 있는지를 논의한다.



### Antidote: Post-fine-tuning Safety Alignment for Large Language Models against Harmful Fine-tuning (https://arxiv.org/abs/2408.09600)
- **What's New**: 이번 논문에서는 Large Language Model (LLM)의 안전성을 강화하기 위한 새로운 접근법인 Antidote를 제안합니다. 이 방법은 특정 training hyper-parameters의 선택에 민감하지 않으면서도 harmful fine-tuning 데이터의 영향을 최소화할 수 있도록 설계되었습니다.

- **Technical Details**: Antidote는 harmful fine-tuning 후 harmful weights를 제거하는 one-shot pruning 단계를 도입합니다. 이를 통해 LLM이 harmful behaviors에서 회복할 수 있도록 하며, fine-tuning 단계에서의 학습률(learning rate)이나 에포크 수(epoch number)에 대한 의존성이 없습니다.

- **Performance Highlights**: 실험 결과, Antidote는 harmful score를 최대 17.8%까지 낮추면서도 downstream task의 정확도를 최대 1.83% 손실로 유지하는 효과를 보였습니다.



### Moonshine: Distilling Game Content Generators into Steerable Generative Models (https://arxiv.org/abs/2408.09594)
- **What's New**: 본 연구는 Procedural Content Generation via Machine Learning (PCGML) 분야에서의 기존의 한계를 극복하기 위해, 건설적인 PCG 알고리즘을 제어 가능한 PCGML 모델로 증류하는 방법을 제시합니다.

- **Technical Details**: 연구진은 먼저 대량의 콘텐츠를 생성하는 건설적인 알고리즘을 사용한 다음, 이를 Large Language Model (LLM)을 통해 라벨링합니다. 이러한 합성 라벨을 이용해 두 가지 PCGML 모델, 즉 diffusion model 과 five-dollar model 의 콘텐츠 특정 생성에 대해 조건화합니다. 이 신경망 증류 과정은 생성이 원래 알고리즘과 일치하도록 하면서도 텍스트를 통해 제어 가능성을 도입합니다. 이를 Text-to-game-Map (T2M) 작업으로 정의하고, 기존의 text-to-image 다중 모달 작업의 대안으로 제시합니다.

- **Performance Highlights**: 증류된 모델들은 기초 건설적인 알고리즘과 비교되었으며, 생성된 콘텐츠의 다양성, 정확성 및 품질 분석을 통해 제어 가능한 텍스트 조건부 PCGML 모델로의 증류 방법이 효과적임을 입증합니다.



### SynTraC: A Synthetic Dataset for Traffic Signal Control from Traffic Monitoring Cameras (https://arxiv.org/abs/2408.09588)
Comments:
          Accepted to IEEE ITSC2024

- **What's New**: 이번 논문에서는 SimTraC 이미지 기반 교통 신호 제어 데이터셋을 소개하며, 시뮬레이션 환경과 실제 교통 관리 간의 간극을 메우는 것을 목표로 합니다. 전통적인 데이터셋이 차량 수와 같은 단순한 특성 벡터를 제공하는 반면, SynTraC는 CARLA 시뮬레이터에서 생성된 실제 스타일의 이미지를 제공하며, 주석 처리된 특성과 교통 신호 상태도 포함되어 있습니다.

- **Technical Details**: SynTraC 데이터셋은 86,000개 이상의 RGB 이미지로 구성되어 있으며, 각 이미지는 다양한 날씨 조건과 시간대에서 수집된 교통 신호 상태 및 보상 값을 태그합니다. 날씨 조건에는 맑음, 안개, 비, 흐림 등이 포함되며, 시간 조건은 낮과 밤으로 구분됩니다. 이 데이터셋은 또한 교통 신호 제어 알고리즘, 특히 강화 학습에 적합합니다.

- **Performance Highlights**: 실험 결과, 이미지 기반 접근 방식과 전통적인 방법 간에는 성능 차이가 있음을 보여주었으며, 이는 기존 강화 학습 방법을 이미지 기반 TSC에 적용하는 데 어려움이 있음을 강조합니다. 이는 교통 관리에서 비주얼 데이터를 최대한 활용하기 위한 새로운 알고리즘의 필요성을 시사합니다.



### PA-LLaVA: A Large Language-Vision Assistant for Human Pathology Image Understanding (https://arxiv.org/abs/2408.09530)
Comments:
          8 pages, 4 figs

- **What's New**: 본 연구에서는 특정 도메인을 위한 대형 언어-비전 모델인 PA-LLaVA를 개발하여 병리 이미지 이해를 지원합니다. 실험을 통해 PA-LLaVA가 기존의 멀티모달 모델 중에서 최고의 성능을 보인 점이 특징입니다.

- **Technical Details**: 1) 병리 이미지와 텍스트의 데이터셋을 정제하여 도메인 맞춤형 모델을 구축합니다. 2) PLIP 모델을 훈련시켜 병리 이미지에 특화된 비주얼 인코더로 활용합니다. 3) 두 단계 학습 방식을 채택하여 PA-LLaVA를 훈련합니다.

- **Performance Highlights**: PA-LLaVA는 감독 학습(supervised learning) 및 제로샷(zero-shot) VQA 데이터셋에서 최상의 성능을 달성했으며, 기존의 유사한 규모의 멀티모달 모델 대비 뛰어난 결과를 보였습니다.



### ALS-HAR: Harnessing Wearable Ambient Light Sensors to Enhance IMU-based HAR (https://arxiv.org/abs/2408.09527)
- **What's New**: 본 연구에서는 신체 착용형 환경광 센서(ALS)를 활용하여 사람의 활동을 인식하는 새로운 접근 방식을 제안합니다. 기존에 널리 사용되는 IMU와 결합하여 ALS의 민감성을 보완하고, 다양한 환경에서도 안정적으로 작동할 수 있는 모듈을 개발했습니다.

- **Technical Details**: ALS-HAR은 패시브 환경광 센서를 기반으로 하여 사용자의 주변 환경과 활동에 대한 정보를 제공합니다. 연구에서는 환경 조건이 변화함에 따라 효율적인 지식 전이 기법을 통해 IMU 기반 분류기의 성능을 향상시키는 방법을 소개합니다. 특히, 앙상블 다중 모달 및 대조적(classification) 방법을 적용하여 데이터 상관성을 높였습니다.

- **Performance Highlights**: ALS-HAR의 정확도는 환경광 조건에 크게 의존하지만, 크로스 모달 정보 전이는 IMU 기반 분류기에서 최대 4.2%와 6.4%의 매크로 F1 점수 향상을 가져왔습니다. 특히, 세 가지 실험 시나리오 중 두 가지에서는 다중 모달 센서 융합 모델보다 더 나은 성능을 발휘했습니다.



### $\mathbb{BEHR}$NOULLI: A Binary EHR Data-Oriented Medication Recommendation System (https://arxiv.org/abs/2408.09410)
- **What's New**: 본 논문은 이진 전자의료기록(EHR) 데이터를 통해 약물 추천 시스템을 구축하는 새로운 접근법을 제시합니다. 기존의 방법들이 주로 이진 값에서 발생하는 문제들로 인해 제한을 받았던 것에 비해, 새로운 통계적 시각을 도입하여 이진 데이터를 연속적 확률로 변환하고, 이를 통해 효과적인 약물 추천을 실시할 수 있음을 보여줍니다.

- **Technical Details**: 제안된 시스템인 𝔹⁢𝔼⁢ℍ⁢ℝ𝔹𝔼ℍℝ\mathbb{BEHR}는 그래프 신경망(GNN)을 기반으로 하여 이진 EHR 데이터를 활용합니다. 데이터는 각 사건의 조건부 베르누이 확률로 모델링되며, GNN은 환자 간 사건의 상관관계를 학습합니다. 이 접근법은 사건의 우연한 발생 관계를 모델링하고, 환자 특성을 강조합니다.

- **Performance Highlights**: 제안된 방법은 MIMIC III 데이터셋에서 Jaccard 점수에 대해 4.8%, F1 점수에 대해 6.7%, PRAUC에 대해 6.3%, AUROC에 대해 4%의 향상을 보여주며, 기존의 이차 정보를 사용하는 방법을 큰 차이로 초월하는 것으로 나타났습니다.



### Obtaining Optimal Spiking Neural Network in Sequence Learning via CRNN-SNN Conversion (https://arxiv.org/abs/2408.09403)
Comments:
          Accepted by 33rd International Conference on Artificial Neural Networks

- **What's New**: 본 연구에서는 기존의 Spiking Neural Networks (SNNs)가 갖는 성능 저하 문제를 해결하기 위해 새로운 Recurrent Bipolar Integrate-and-Fire (RBIF) 뉴런 모델을 제안합니다. 이를 통해 RNN과 SNN 간의 매핑에 성공하여, 더 긴 시퀀스 학습에서 안정적인 성능을 보여줍니다.

- **Technical Details**: SNN은 기존의 ANN과 달리 비선형 이진 통신 메커니즘을 사용하여 신경 간 정보를 교환합니다. 본 논문에서는 두 개의 서브 파이프라인(CNN-Morph 및 RNN-Morph)을 통해 엔드 투 엔드 컨버전을 지원하며, 양자화된 CRNN에서의 매핑을 통해 완전한 손실 없는 변환을 달성합니다. 이를 통해 SNN의 성능을 극대화합니다.

- **Performance Highlights**: S-MNIST에서 99.16%의 정확도(0.46% 향상), PS-MNIST(시퀀스 길이 784)에서 94.95%의 정확도(3.95% 향상), 그리고 충돌 회피 데이터셋에서 8 타임 스텝 내 평균 손실 0.057(0.013 감소)을 달성하여 현재 최첨단(contemporaneous state-of-the-art) 방법들을 초월했습니다.



### Game Development as Human-LLM Interaction (https://arxiv.org/abs/2408.09386)
- **What's New**: 본 논문에서는 LLM(대형 언어 모델)을 활용한 사용자 친화적 게임 개발 플랫폼인 Interaction-driven Game Engine(IGE)을 소개합니다. 사용자는 자연어로 게임을 개발할 수 있으며, 복잡한 프로그래밍 언어의 학습 없이도 맞춤형 게임을 만들 수 있게 됩니다.

- **Technical Details**: IGE는 사용자의 입력을 기반으로 게임 스크립트 세그먼트를 구성하고, 해당 스크립트에 따른 코드 스니펫을 생성하며, 사용자와의 상호작용(가이드 및 피드백 포함)을 처리하는 세 가지 주요 단계(P_{script}, P_{code}, P_{utter})를 수행하도록 LLM을 훈련시킵니다. 또한, 효과적인 데이터 생성 및 훈련을 위해 세 단계의 점진적 훈련 전략을 제안합니다.

- **Performance Highlights**: 포커 게임을 사례 연구로 사용하여 IGE의 성능을 두 가지 관점(상호작용 품질 및 코드 정확성)에서 종합적으로 평가했습니다. 이 평가 과정은 IGE의 게임 개발 효율성을 향상시키고, 사용자와의 상호작용을 쉽게 만드는 데 기여합니다.



### ELASTIC: Efficient Linear Attention for Sequential Interest Compression (https://arxiv.org/abs/2408.09380)
Comments:
          Submitted to AAAI 2025

- **What's New**: ELASTIC은 고전적인 self-attention의 복잡도를 낮추어 긴 사용자 행동 시퀀스를 효율적으로 모델링할 수 있는 새로운 기법입니다. 이 방법은 선형 시간 복잡도를 요구하여 모델 용량을 계산 비용과 분리시킵니다.

- **Technical Details**: ELASTIC은 고정 길이의 interest experts와 선형 dispatcher attention 메커니즘을 도입하여 긴 행동 시퀀스를 압축된 표현으로 변환합니다. 새롭게 제안된 interest memory retrieval 기술은 대규모 사용자 맞춤형 관심을 모델링하는데 있어 일관된 계산 비용을 유지하면서 높은 정확도를 제공합니다.

- **Performance Highlights**: ELASTIC은 공개 데이터셋에서 다양한 기존 추천 시스템보다 탁월한 성과를 보이며 GPU 메모리 사용량을 최대 90%까지 줄이고, 추론 속도를 2.7배 향상시킵니다. 실험 결과 ELASTIC은 기존 방법에 비해 추천 정확성과 효율성을 모두 달성했습니다.



### Concept Distillation from Strong to Weak Models via Hypotheses-to-Theories Prompting (https://arxiv.org/abs/2408.09365)
Comments:
          13 pages, 8 figures, conference

- **What's New**: 본 논문에서는 Concept Distillation (CD)이라는 자동화된 프롬프트 최적화 기법을 제안하여, 약한 언어 모델(weak language models)의 복잡한 작업에서의 성능을 향상시키는 방법을 다룹니다. CD는 초기 프롬프트를 기반으로 약한 모델의 오류를 수집하고, 강력한 모델(strong model)을 사용해 이러한 오류에 대한 이유를 생성하여 규칙을 만듭니다.

- **Technical Details**: CD는 세 가지 주요 단계를 포함합니다: (1) 약한 모델의 잘못된 응답 수집(초기화), (2) 강력한 모델을 통해 오류 분석 및 규칙 생성(유도), (3) 검증 세트 성능에 따른 규칙 필터링 및 기본 프롬프트에 통합(추론/검증). 이 과정에서, 강력한 모델이 약한 모델의 성능을 개선하는 데 필요한 개념을 제공하는 방식으로 진행됩니다.

- **Performance Highlights**: CD 방법을 NL2Code, 수학적 추론 과제 등에 적용한 결과, 작은 언어 모델의 성능이 현저히 향상되었습니다. 예를 들어, Mistral-7B 모델의 Multi-Arith 정확도가 20% 증가하고, Phi-3-mini-3.8B 모델의 HumanEval 정확도가 34% 상승했습니다. CD는 다른 자동화 방법들에 비해 약한 모델의 성능을 개선하고, 새로운 언어 모델로의 원활한 전환을 지원할 수 있는 비용 효율적인 전략을 제공합니다.



### Siamese Multiple Attention Temporal Convolution Networks for Human Mobility Signature Identification (https://arxiv.org/abs/2408.09230)
Comments:
          27th IEEE International Conference on Intelligent Transportation Systems (ITSC) (ITSC 2024)

- **What's New**: 본 논문에서는 Human Mobility Signature Identification (HuMID) 문제를 해결하기 위해 Siamese Multiple Attention Temporal Convolutional Network (Siamese MA-TCN)을 제안합니다. 이 모델은 TCN 아키텍처와 multi-head self-attention의 장점을 결합하여 지역 및 장기 의존성을 효과적으로 추출할 수 있습니다.

- **Technical Details**: Siamese MA-TCN은 GPS 트레일 데이터를 기반으로 한 대규모 드라이버 식별을 위해 효율적으로 정확한 예측을 제공하며, multi-head self-attention 메커니즘을 통해 전역 의존성을 획득하고, 다중 스케일 지역 특징을 캡처하기 위한 특별히 설계된 집계 주의 메커니즘을 사용합니다. 또한, 아블레이션 실험을 통해 이러한 구성 요소의 효과성을 검증합니다.

- **Performance Highlights**: 두 개의 실제 택시 트레일 데이터셋에 대한 실험 평가를 통해 제안된 모델이 지역 주요 정보와 장기 의존성을 효과적으로 추출할 수 있음을 확인했습니다. 이러한 결과는 다양한 크기의 데이터셋에서 모델의 뛰어난 일반화 능력을 강조하며, 노출되지 않은 드라이버에 대한 탐지 성능을 개선합니다.



### FEDMEKI: A Benchmark for Scaling Medical Foundation Models via Federated Knowledge Injection (https://arxiv.org/abs/2408.09227)
Comments:
          Submitted to Neurips 2024 DB Track

- **What's New**: 본 연구는 의료 지식을 기초 모델에 통합하는 데 있어 프라이버시 제한을 고려하여 설계된 새로운 벤치마크인 Federated Medical Knowledge Injection (FEDMEKI) 플랫폼을 소개합니다. 이 플랫폼은 중앙 집중식 데이터 수집의 문제를 회피하며, 다중 사이트, 다중 모드 및 다중 작업 의료 데이터를 처리할 수 있도록 세심하게 설계되었습니다.

- **Technical Details**: FEDMEKI 플랫폼은 7개의 의료 모달리티(이미지, 신호, 텍스트, 실험실 결과, 생체 신호 등)를 포함한 다중 사이트, 다중 모달, 다중 작업 데이터 세트를 관리합니다. 검증을 위해 8개의 의료 작업을 포함하는 데이터셋을 커리팅하였으며, 학습 과정에서 16개의 벤치마크 방법을 사용하여 분산 학습을 수행합니다.

- **Performance Highlights**: FEDMEKI는 데이터 프라이버시를 유지하면서도 더 넓은 범위의 의료 지식으로부터 학습할 수 있도록 의료 기초 모델의 능력을 향상시킵니다. 이를 통해 헬스케어 분야에서 기초 모델의 새로운 벤치마크를 설정합니다.



### Neuro-Symbolic AI for Military Applications (https://arxiv.org/abs/2408.09224)
Comments:
          Accepted at IEEE Transactions on Artificial Intelligence (TAI)

- **What's New**: 이번 논문은 Neuro-Symbolic AI의 군사 응용 가능성을 탐구하며, 방어 시스템과 전략적 의사결정 향상에 기여하는 혁신적인 요소들을 강조합니다.

- **Technical Details**: Neuro-Symbolic AI는 신경망(neural networks)과 상징적 추론(symbolic reasoning)의 장점을 결합하는 접근 방식입니다. 이 시스템은 복잡한 정보 분석(analysis) 및 자율 시스템(autonomous systems)의 자동화, 그리고 전투 상황에서의 전술적 의사결정(tactical decision-making) 향상을 가능하게 합니다. 또한, 윤리적(ethical), 전략적(strategic), 기술적(technical) 고려사항도 논의합니다.

- **Performance Highlights**: Neuro-Symbolic AI는 군사 작전에서의 전술적 의사결정, 복잡한 정보 분석 자동화, 자율 시스템의 강화에 있어 중요한 역할을 할 수 있으며, 물류 최적화(logistics optimization) 및 동적 의사결정(dynamic decision-making) 등 다양한 분야에서의 잠재력도 보여주고 있습니다.



### Maintainability Challenges in ML: A Systematic Literature Review (https://arxiv.org/abs/2408.09196)
- **What's New**: 이 연구는 기계 학습(ML) 시스템의 다양한 워크플로우 단계에서의 유지보수성(challenges) 문제를 체계적으로 식별하고 분석하였습니다. 이를 통해 각 단계가 서로 어떻게 의존하며 유지보수에 어떤 영향을 미치는지를 이해하는 것을 목표로 합니다.

- **Technical Details**: 13000편 이상의 논문을 스크리닝하고 56편을 정성적으로 분석한 결과, 데이터 엔지니어링(Data Engineering)과 모델 엔지니어링(Model Engineering) 워크플로우에서의 13가지 유지보수성 문제를 정리한 카탈로그가 작성되었습니다. 또한, 이 문제들이 ML의 전반적인 워크플로우에 미치는 영향을 시각적으로 맵으로 나타냈습니다.

- **Performance Highlights**: 이 연구의 주요 기여는 ML 시스템의 유지보수성 문제를 관리하고 해결하는 방법을 제시함으로써 개발자들이 실수를 피하고 유지보수 가능한 ML 시스템을 구축하는 데에 도움을 줄 수 있다는 점입니다.



### AI Managed Emergency Documentation with a Pretrained Mod (https://arxiv.org/abs/2408.09193)
Comments:
          Ethical approval for the study was obtained from the University College Dublin, Human Research Ethics Committee (UCD HREC)

- **What's New**: 이 연구는 응급실(ED)에서 퇴원 편지 작성을 개선하기 위한 대형 언어 모델 시스템의 활용을 조사하였습니다. 현재 퇴원 편지 목표에 대한 준수가 어려운 시간 제약과 인프라 부족 문제를 해결하고자 AI 소프트웨어의 효율성 제안을 탐색했습니다.

- **Technical Details**: 연구에서 사용된 시스템은 GPT-3 Davinci 모델을 기반으로 하여 의료 중심 퇴원 편지를 생성하도록 세부 조정되었습니다. 시스템은 음성에서 텍스트로 변환하기 위해 open-source whisper model을 사용하며, 주기적인 세부 조정 과정을 통해 모델 성능을 향상시킵니다.

- **Performance Highlights**: 19명의 응급 의학 경력이 있는 의사들이 MedWrite LLM 인터페이스를 평가한 결과, 수작업과 비교하여 시각적으로 의미 있는 시간 절약이 스며드는 것으로 나타났습니다.



### Cognitive LLMs: Towards Integrating Cognitive Architectures and Large Language Models for Manufacturing Decision-making (https://arxiv.org/abs/2408.09176)
Comments:
          20 pages, 8 figures, 2 tables

- **What's New**: 이번 연구에서는 Cognitive Architectures와 Large Language Models (LLMs) 간의 이분법을 해결하기 위해 LLM-ACTR라는 새로운 신경-기호적 (neuro-symbolic) 아키텍처를 도입했습니다. 이는 사람의 의사결정 과정을 모방할 수 있는 지식을 LLM에 주입하여 보다 신뢰성 있는 기계 판단 능력을 확보하는 것을 목표로 합니다.

- **Technical Details**: LLM-ACTR는 ACT-R Cognitive Architecture와 LLM을 통합하여 인간과 유사하지만 유연한 의사결정 기능을 제공합니다. 이 프레임워크는 ACT-R의 내부 의사결정 과정을 잠재적 신경 표현(latent neural representations)으로 추출하고, 이를 조정 가능한 LLM 어댑터 레이어에 주입하여 후속 예측을 위해 LLM을 미세 조정(fine-tuning)합니다.

- **Performance Highlights**: 새로운 설계 제조(Design for Manufacturing) 작업에서 LLM-ACTR의 실험 결과, 이전 LLM-only 모델과 비교하여 작업 성능이 향상되었으며, grounded decision-making 능력이 개선되었음을 확인했습니다.



### Unc-TTP: A Method for Classifying LLM Uncertainty to Improve In-Context Example Selection (https://arxiv.org/abs/2408.09172)
Comments:
          7 pages, long paper

- **What's New**: 이 논문은 LLM의 불확실성을 분류하기 위한 새로운 Paradigm인 Uncertainty Tripartite Testing Paradigm (Unc-TTP)을 제안합니다. Unc-TTP는 LLM 출력의 일관성을 평가하여 LLM의 불확실성을 분류합니다.

- **Technical Details**: Unc-TTP는 세 가지 테스트 시나리오 {no-label, right-label, wrong-label} 하에 LLM의 모든 결과를 열거하고 응답 일관성에 따라 모델 불확실성을 분류하는 접근법입니다. 이를 통해 출력 결과를 82개의 카테고리로 분류합니다.

- **Performance Highlights**: Unc-TTP를 기반으로 한 예시 선택 전략은 LLM의 성능을 개선하며, 기존의 retrieval 기반 방법보다 효율성이 뛰어나고 더 나은 성능 향상을 보여줍니다.



### Measuring Visual Sycophancy in Multimodal Models (https://arxiv.org/abs/2408.09111)
- **What's New**: 이 논문에서는 다중 모달 언어 모델에서 나타나는 '시각적 아첨(visual sycophancy)' 현상을 소개하고 분석합니다. 이는 모델이 이전의 지식이나 응답과 상충할지라도 시각적으로 제시된 정보를 지나치게 선호하는 경향을 설명하는 용어입니다. 연구 결과, 모델이 시각적으로 표시된 옵션을 선호하는 경향이 있음을 발견했습니다.

- **Technical Details**: 이 연구는 여러 가지 모델 아키텍처에서 일관된 패턴으로 시각적 아첨을 정량화할 수 있는 방법론을 적용했습니다. 주요 기술로는 이미지와 선택지의 시각적 강조를 기반으로 한 실험 설계를 통해 모델 응답이 어떻게 변화하는지를 측정했습니다. 실험은 과거 지식을 유지하면서도 시각적 단서를 통해 얼마나 큰 영향을 받는지를 평가하는 방식으로 진행되었습니다.

- **Performance Highlights**: 모델들이 초기 정답을 제공한 후에도 시각적으로 강조된 옵션으로 응답이 치우치는 경향이 나타났습니다. 이는 모델의 신뢰성에 중대한 한계를 드러내며, 비판적인 의사 결정 맥락에서 이러한 현상이 어떻게 영향을 미칠지에 대한 새로운 의문을 제기합니다.



### Temporal Reversed Training for Spiking Neural Networks with Generalized Spatio-Temporal Representation (https://arxiv.org/abs/2408.09108)
Comments:
          15 pages, 8 figures

- **What's New**: 이번 논문에서는 Spiking Neural Networks (SNNs)의 시공간 성능을 최적화하기 위한 새로운 Temporal Reversed Training (TRT) 방법을 제안합니다. 이 방법은 기존의 SNNs가 겪고 있는 비효율적인 추론과 최적이 아닌 성능 문제를 해결합니다.

- **Technical Details**: Temporal Reversed Training (TRT) 방법은 입력 시간 데이터를 시간적으로 반전시켜 SNN이 원본-반전 일관된 출력 로짓(logits)을 생성하도록 유도합니다. 이를 통해 perturbation-invariant representations를 학습하게 됩니다. 정적 데이터의 경우, 스파이크 신경세포의 고유한 시간적 특성을 이용하여 스파이크 기능의 시간적 반전을 적용합니다. 원래의 스파이크 발화율과 시간적으로 반전된 스파이크 발화율을 요소별 곱(element-wise multiplication)을 통해 혼합하여 spatio-temporal regularization을 수행합니다.

- **Performance Highlights**: 정적 및 신경형 물체/행동 인식, 3D 포인트 클라우드 분류 작업에서 광범위한 실험을 통해 제안된 방법의 효과성과 일반화를 입증하였습니다. 특히, 단 두 개의 타임스텝으로 ImageNet에서 74.77% 그리고 ModelNet40에서 90.57% 정확도를 달성하였습니다.



### Research on color recipe recommendation based on unstructured data using TENN (https://arxiv.org/abs/2408.09094)
- **What's New**: 이 논문에서는 비정형 데이터 및 감정적인 자연어를 기반으로 색상 레시피를 유추하는 TENN(Tokenizing Encoder Neural Network) 모델을 제안하고 이를 입증했습니다.

- **Technical Details**: TENN 모델은 비정형 데이터와 교육되지 않은 레시피를 추론하기 위해 비선형 데이터 패턴을 처리할 수 있는 강력한 인공 신경망을 사용합니다. 데이터 전처리는 토큰화(tokenization) 및 인코딩(encoding) 기능을 포함한 12개 층으로 구성되어 있으며, 감정 정보를 인식하여 RGB 색상 코드 조합을 출력합니다. 총 300개 이상의 샘플을 사용하여 학습되었습니다.

- **Performance Highlights**: TENN 모델은 입력된 비정형 데이터로부터 감정 정보를 바탕으로 색상 패턴을 파악할 수 있었으며, 유사한 색상에 대해 추천된 RGB 색 코드의 정확도는 약 80-85%로 이전 연구보다 약 5% 낮은 성능을 나타냈습니다. 평균 Delta E 값은 73.8로, 이는 색상 간의 차이를 나타내는 정량적 지표입니다.



### Keep Calm and Relax -- HMI for Autonomous Vehicles (https://arxiv.org/abs/2408.09046)
Comments:
          14 pages, 3 figures, 1 table

- **What's New**: 자율주행차의 인기가 높아짐에 따라 승객의 신뢰 및 편안함을 향상시키기 위한 사용자 인터페이스(UI)와 인간-기계 인터페이스(HMI)의 필요성이 증가했습니다. 이 연구는 HMI와 UI의 가능성을 탐구하며, 다양한 감정 조절 방법이 자율주행차의 신뢰성을 어떻게 증진시킬 수 있는지를 분석합니다.

- **Technical Details**: 이 연구는 자율주행차에서 HMI 및 UI의 상호작용 클러스터를 정의하고, 반자동화 및 완전 자동화 차량에서 승객의 감정을 개선하기 위한 열 가지 상호작용 클러스터를 제안합니다. 기존 문헌을 기반으로 한 포괄적인 리뷰를 통해, 멀티모달 사용자 인터페이스의 이점과 섬세한 커뮤니케이션을 위한 단일 모달 UI의 효과를 강조합니다.

- **Performance Highlights**: 다양한 HMI 및 UI의 분석을 통해 승객의 신뢰를 높이는 혁신적이고 비용 효율적인 솔루션을 제안했습니다. 이 연구는 위기 상황에서 승객을 진정시키는 HMI와 UI의 가능성을 논의하며, 모든 사용자에게 더 스마트한 이동성을 제공하기 위한 방향을 제시합니다.



### On the Completeness of Conflict-Based Search: Temporally-Relative Duplicate Pruning (https://arxiv.org/abs/2408.09028)
Comments:
          9 pages, 4 figures, 2 tables

- **What's New**: 이 논문에서는 Conflict-Based Search (CBS) 알고리즘의 무한한 실행 문제를 해결하기 위해 Temporally-Relative Duplicate Pruning (TRDP) 기법을 도입했습니다. TRDP는 중복 상태를 탐지하고 제거하여 CBS의 완전성을 보장합니다.

- **Technical Details**: TRDP는 매개 변수가 있는 MAPF(Multi-Agent Pathfinding) 문제에서 중복 탐지를 위한 간단한 절차로, CBS의 이론적 결함을 보완합니다. 이를 통해 TRDP는 다중 에이전트 루프를 감지하고 파괴하여 무한한 검색 공간을 유한하게 만듭니다. TRDP는 대부분의 MAPF 도메인에 적용 가능하며, 이론적 및 경험적으로 검색 종료를 보장합니다.

- **Performance Highlights**: 논문에서 제시된 결과에 따르면 TRDP는 해결 가능한 MAPF 사례에서 실행 시간에 큰 영향을 미치지 않으며, 특정 경우에는 성능을 크게 향상시키는 것으로 나타났습니다.



### On the Undecidability of Artificial Intelligence Alignment: Machines that Ha (https://arxiv.org/abs/2408.08995)
Comments:
          Submitted for the Scientific Reports AI Alignment Collection

- **What's New**: 내부 정렬(inner alignment) 문제는 특정 인공지능(AI) 모델이 입력에 따른 출력의 비논리적 정렬 기능을 만족하는지를 검토하는 것으로, 이는 결정 불가능한 문제임을 철저히 증명하였다. 이로 인해 우리는 AI 아키텍처가 사전적으로 정렬 성질을 보장해야 한다고 주장한다.

- **Technical Details**: Rice의 정리(Rice's Theorem)를 통해 임의의 AI 시스템이 정렬 기능을 항상 만족할 것인지 결정하는 것은 불가능하다고 보인다. 하지만 정렬이 보장된 기본 모델과 작업의 유한 집합에서 출발하여 원하는 특성을 가지는 무한 집합의 AI를 구성할 수 있음을 논의했다.

- **Performance Highlights**: 작업은 통제 가능성, 모니터링, AI 윤리 준수 등의 다른 AI 시스템 속성이 결정 불가능하다는 점에 대한 새로운 관점을 제시한다. 모델과 작업이 정렬 보장을 가진 채로 만들어질 수 있는 가능성을 강조한다.



### Ask, Attend, Attack: A Effective Decision-Based Black-Box Targeted Attack for Image-to-Text Models (https://arxiv.org/abs/2408.08989)
- **What's New**: 이 논문에서는 이미지-텍스트 모델에 대한 의사 결정 기반 블랙박스(targeted black-box) 공격 접근 방식을 제안합니다. 특히, 공격자는 모델의 최종 출력 텍스트에만 접근할 수 있으며, 이를 통해 목표 텍스트와 관련된 공격을 수행하게 됩니다. 또한, 이 연구는 'Ask, Attend, Attack'의 세 단계로 구성된 공격 프로세스를 통해 최적화 문제를 해결하는 방법을 제시합니다.

- **Technical Details**: 제안된 방법론은 'Ask, Attend, Attack'(AAA) 프로세스를 포함합니다. 'Ask' 단계에서 공격자는 특정 의미를 충족하는 타겟 텍스트를 생성하도록 안내받고, 'Attend' 단계에서는 공격을 위한 중요한 이미지 영역을 파악하여 탐색 공간을 줄이며, 마지막 'Attack' 단계에서는 진화 알고리즘(evolutionary algorithm)을 사용하여 목표 텍스트와 출력 텍스트 간의 불일치를 최소화하는 방식으로 공격을 수행합니다. 이는 공격자가 출력 텍스트를 목표 텍스트에 가깝게 변경할 수 있게 합니다.

- **Performance Highlights**: 실험 결과, Transformer 기반과 CNN+RNN 기반의 대표 이미지-텍스트 모델인 VIT-GPT2와 Show-Attend-Tell에서 제안된 AAA 방법이 기존의 gray-box 방법들보다 우수한 공격 성능을 보임을 확인하였습니다. 본 연구는 결정 기반 블랙박스 공격이 기존 방법에 비해 효과적인 해결책이 될 수 있음을 강조합니다.



### ASGM-KG: Unveiling Alluvial Gold Mining Through Knowledge Graphs (https://arxiv.org/abs/2408.08972)
- **What's New**: ASGM-KG(Artisanal and Small-Scale Gold Mining Knowledge Graph)는 아마존 분지의 환경 영향을 이해하는 데 도움이 되는 지식 그래프입니다. ASGM_KG는 비정부 및 정부 기구에서 발행된 문서와 보고서에서 추출된 1,899개의 triple로 구성되어 있습니다.

- **Technical Details**: ASGM-KG는 대형 언어 모델(LLM)을 사용하여 RDF(Resource Description Framework) 형식으로 1,899개의 triple을 생성했습니다. 생성된 triple은 전문가의 검토 및 자동화된 사실 검증 프레임워크(Data Assessment Semantics, DAS)를 통해 검증되었습니다.

- **Performance Highlights**: ASGM-KG는 90% 이상의 정확도를 달성하였으며, 다양한 환경 위기에 대한 지식 집합 및 표현의 발전을 나타냅니다.



### Differentiable Edge-based OPC (https://arxiv.org/abs/2408.08969)
Comments:
          Accepted by ICCAD24

- **What's New**: 본 논문에서는 차별화 가능한 광학 근접 보정 기술인 DiffOPC를 제안합니다. 이는 앞서 개발된 edge-based OPC와 inverse lithography technology (ILT)의 장점을 결합하여 마스크 최적화를 통해 반도체 제조에서의 높은 정확도와 실용성을 제공합니다.

- **Technical Details**: DiffOPC는 마스크 규칙을 고려한 그래디언트 기반의 최적화 접근법을 사용하여 마스크의 가장자리 세그먼트의 이동을 효율적으로 안내합니다. 이 방법은 공정 변화를 고려하여 edge placement error (EPE)를 최적화합니다. CUDA 가속화된 레이 캐스팅 및 새로운 SRAF(서브 해상도 보조 기능) 생성 알고리즘을 도입합니다.

- **Performance Highlights**: DiffOPC는 기존 EBOPC의 EPE를 반으로 줄이고, ILT보다 더 낮은 EPE를 달성하면서도 제조 비용은 ILT의 절반에 해당하는 결과를 보였습니다. 이는 반도체 제조에서 고품질의 효율적인 OPC 수정을 위한 유망한 솔루션입니다.



### Adaptive Guardrails For Large Language Models via Trust Modeling and In-Context Learning (https://arxiv.org/abs/2408.08959)
Comments:
          Under Review

- **What's New**: 이 연구는 사용자 신뢰 지표를 기반으로 민감한 콘텐츠에 대한 접근을 동적으로 조절할 수 있는 적응형 경계 장치(adaptive guardrail) 메커니즘을 도입합니다. 이를 통해 사용자별로 신뢰도를 고려하여 콘텐츠 중재를 조정할 수 있으며, 기존의 정적이지 않은 방식과의 차별점을 제공합니다.

- **Technical Details**: 이 시스템은 Direct Interaction Trust와 Authority Verified Trust의 조합을 활용하여 사용자의 신뢰를 평가하고, 이를 바탕으로 개별 사용자가 알맞은 콘텐츠에 접근할 수 있도록 합니다. 또한, In-context Learning (ICL)을 통해 실시간으로 사용자 쿼리의 민감도에 맞춰 응답을 맞춤형으로 조정하는 기능이 포함되어 있습니다. 이는 LLM의 동적 환경에 잘 적응할 수 있는 경계 장치를 구현하는 데 중요한 역할을 합니다.

- **Performance Highlights**: 실험 평가 결과, 제안된 적응형 경계 장치가 다양한 사용자 요구를 효과적으로 충족시키며, 기존의 경계 장치보다 뛰어난 실용성을 보여주었습니다. 민감한 정보 보호와 잠재적 위험 콘텐츠의 정확한 관리를 통해 LLMs의 안전한 운영을 보장하는데 기여하고 있습니다.



### Imprecise Belief Fusion Facing a DST benchmark problem (https://arxiv.org/abs/2408.08928)
Comments:
          12 pages

- **What's New**: 이 연구는 Dempster-Shafer Theory (DST)의 결합 규칙에서 발생하는 비정상적인 동작을 해결하기 위한 새로운 융합 방법을 제안합니다. 이 방법은 믿음 융합을 확률적 논리적 프로세스로 접근하는 최초의 시도입니다.

- **Technical Details**: Dempster-Shafer Theory (DST)는 불확실성을 모델링하기 위한 이론적 프레임워크를 제공하며, 그 과정에서 믿음 융합(Belief Fusion)을 통해 여러 독립적인 출처에서 정보를 결합합니다. 저자들은 DST의 결합 규칙이 동등한 신뢰성과 전문성을 가진 출처에서 의견을 무시하는 문제가 발생하는 원인을 고찰하고, 새로운 융합 방법으로 비정상을 제거하고자 합니다. 이 새로운 융합 방법은 확률적 논리를 기반으로 하며, 기존의 Dempster Paradox 문제를 해결하는 데 초점을 맞추고 있습니다.

- **Performance Highlights**: 저자들은 제안하는 새로운 융합 방법이 기존의 DST 결합 규칙보다 효과적으로 다양한 출처에서 얻은 정보를 통합하며, 이를 통해 Dempster Paradox를 해결할 수 있는 가능성을 보여주고 있습니다.



### VerilogCoder: Autonomous Verilog Coding Agents with Graph-based Planning and Abstract Syntax Tree (AST)-based Waveform Tracing Too (https://arxiv.org/abs/2408.08927)
Comments:
          main paper 7 pages, reference 1 page, appendix 22 pages. It is under review of AAAI 2025

- **What's New**: 본 논문에서는 여러 인공지능(AI) 에이전트를 활용하여 Verilog 코드 생성을 자동화하는 시스템인 VerilogCoder를 제안합니다. 이 시스템은 Verilog 코드를 자율적으로 작성하고 문법 및 기능 오류를 수정하며, Verilog 개발 도구를 협력적으로 사용합니다.

- **Technical Details**: VerilogCoder는 새로운 Task and Circuit Relation Graph(TCRG) 기반의 계획자를 이용하여 모듈 설명을 바탕으로 높은 품질의 계획을 작성합니다. 기능 오류를 디버깅하기 위해 새로운 추상 구문 트리(AST) 기반의 파형 추적 도구를 통합합니다. 이 시스템은 최대 94.2%의 문법 및 기능적 정확성을 발휘하며, 최신 방법보다 33.9% 더 향상된 성과를 보여줍니다.

- **Performance Highlights**: VerilogCoder는 VerilogEval-Human v2 벤치마크에서 94.2%의 성공률을 달성하였으며, 이는 기존 최첨단 방법보다 33.9% 높은 성능입니다.



### Graph Retrieval-Augmented Generation: A Survey (https://arxiv.org/abs/2408.08921)
Comments:
          Ongoing work

- **What's New**: 이 논문은 Retrieval-Augmented Generation (RAG) 분야의 최신 기술 중 하나인 GraphRAG에 대한 체계적인 리뷰를 제공합니다. GraphRAG는 관계형 지식을 활용하여 더 정확하고 포괄적인 정보 검색을 가능하게 하여, 기존 RAG의 한계를 극복하고자 합니다.

- **Technical Details**: GraphRAG 워크플로우는 Graph-Based Indexing (G-Indexing), Graph-Guided Retrieval (G-Retrieval), Graph-Enhanced Generation (G-Generation)의 세 단계로 구성됩니다. 각 단계에서는 핵심 기술 및 학습 방법을 정리하고 있습니다. GraphRAG는 그래프 데이터베이스에서 적절한 쿼리와 관련된 그래프 요소를 검색하여 응답을 생성할 수 있습니다.

- **Performance Highlights**: GraphRAG는 텍스트 기반 RAG보다 관계형 정보를 더 정확하고 포괄적으로 검색할 수 있으며, 대규모 텍스트 입력의 긴 문맥 문제를 해결합니다. 또한, 다양한 산업 분야에서의 활용 가능성이 커지고 있으며, 연구의 초기 단계이지만, GraphRAG의 잠재력을 통해 새로운 연구 방향을 제시하고 있습니다.



### Cyclic Supports in Recursive Bipolar Argumentation Frameworks: Semantics and LP Mapping (https://arxiv.org/abs/2408.08916)
Comments:
          Paper presented at the 40th International Conference on Logic Programming (ICLP 2024), University of Texas at Dallas, USA, October 2024

- **What's New**: 이 논문에서는 Dung의 Abstract Argumentation Framework (AF)와 그의 확장인 Bipolar Argumentation Framework (BAF)와 Recursive BAF (Rec-BAF)의 기존의 복잡한 의미론(semanics)에 대한 문제를 해결하기 위한 클래식한 의미론을 제안합니다.

- **Technical Details**: 제안된 방법론은 AF 기반의 프레임워크에서 패배와 수용 요소의 집합을 모듈 방식으로 정의함으로써, 일반 BAF 및 Rec-BAF의 의미론을 우아하고 일관되게 설명합니다. 특별히, 복잡한 순환 지원이 내재된 상황에서의 프레임워크 의미론의 정의와 관련된 문제를 단순한 수정으로 해결합니다.

- **Performance Highlights**: 이 새로운 접근법을 통해 BAF 및 Rec-BAF 프레임워크의 모든 특정 프레임워크에 적용 가능한 의미론을 도출할 수 있으며, 이는 기존의 비순환 프레임워크에서 정의된 의미론의 확장으로 이해될 수 있습니다.



### KAN 2.0: Kolmogorov-Arnold Networks Meet Scienc (https://arxiv.org/abs/2408.10205)
Comments:
          27 pages, 14 figures

- **What's New**: 이 논문은 Kolmogorov-Arnold Networks(KAN)과 과학을 통합하기 위한 새로운 프레임워크를 제안합니다. 이 프레임워크는 KAN이 과학 발견의 세 가지 측면 - 관련 기능 식별, 모듈 구조 제시 및 기호적 공식 발견 - 을 지원하도록 합니다.

- **Technical Details**: 이 논문에서 제시된 새 기능으로는 MultKAN(곱셈 노드가 있는 KAN), kanpiler(기호 공식에서 KAN으로 변환하는 컴파일러), tree converter(KAN을 트리 그래프로 변환하는 기능)가 있습니다. 이러한 도구들을 통해, KAN은 보존된 양, Lagrangians, 대칭, 구성 법칙 등의 물리 법칙을 발견하는 능력을 보여줍니다.

- **Performance Highlights**: 최근 몇 년간 AI + Science 분야에서 상당한 발전을 이루어왔으며, 이 연구는 호기심 기반 과학(curiosity-driven science)의 필요성을 강조하며 KAN을 활용하여 이전에 설명되지 않았던 과학적 지식을 추출하고 통합하는 방법을 제시합니다.



### Demystifying the Communication Characteristics for Distributed Transformer Models (https://arxiv.org/abs/2408.10197)
- **What's New**: 이번 논문에서는 트랜스포머(Transformer) 아키텍처 기반의 심층 학습 모델의 통신 동작을 분석하여, 여러 노드 및 다중 GPU에서의 훈련 과정에서 데이터 통신 최적화를 모색했습니다.

- **Technical Details**: 우리는 GPT 기반의 언어 모델을 사례 연구로 사용하였으며, 데이터 볼륨, 통신 원시 요소, 통화 횟수 및 메시지 크기와 같은 요소들을 다양한 병렬화 전략에 따라 분석했습니다. 또한, 데이터 병렬(Data Parallelism) 및 모델 병렬(Model Parallelism) 환경에서의 시퀀스 길이가 통신 볼륨에 미치는 영향을 조사했습니다.

- **Performance Highlights**: 이번 연구의 주요 기여로는 다양한 병렬화 방식과 시퀀스 길이에 따른 통신 행동의 시스템 불문 측정치를 제공하였고, AMD 인피니티 패브릭(AMD Infinity Fabric)과 HPE 슬링샷 11(HPE-Slingshot 11)에서 레이턴시 및 대역폭을 측정하여 커뮤니케이션 오버헤드를 분석했습니다.



### SpaRP: Fast 3D Object Reconstruction and Pose Estimation from Sparse Views (https://arxiv.org/abs/2408.10195)
Comments:
          ECCV 2024

- **What's New**: 이 논문은 제한적인 수의 non-posed 2D 이미지로부터 3D 객체를 복원하는 새로운 메소드인 SpaRP를 제안합니다. 이 방법은 기존의 dense view 방식과 달리 sparse view 입력을 처리하며, 기존 방법들이 기대에 미치지 못하는 컨트롤을 개선하였습니다.

- **Technical Details**: SpaRP는 2D diffusion models에서 지식을 추출하고, 이를 활용하여 sparse view 이미지 간의 3D 공간 관계를 유추합니다. 이 메소드는 카메라 포즈를 추정하고 3D textured mesh를 생성하는 데 필요한 정보를 종합적으로 활용하여, 단지 약 20초 만에 결과를 도출합니다.

- **Performance Highlights**: 세 가지 데이터셋에 대한 실험을 통해 SpaRP는 3D 복원 품질과 포즈 예측 정확도에서 기존의 방법들과 비교하여 상당한 성능 향상을 보여주었습니다.



### Transformers to SSMs: Distilling Quadratic Knowledge to Subquadratic Models (https://arxiv.org/abs/2408.10189)
- **What's New**: 본 논문에서는 사전학습된 Transformer 아키텍처를 다양한 대안 아키텍처, 특히 상태 공간 모델(SSM)로 증류하는 방법을 제안합니다. 이는 기존의 Mamba 아키텍처를 개선하는 진전을 보여줍니다.

- **Technical Details**: 제안된 방법인 MOHAWK는 세 가지 단계로 구성됩니다: (1) 행렬 정렬 단계에서 시퀀스 변환 행렬을 정렬하고, (2) 숨겨진 상태 증류 단계에서 개별 레이어의 숨겨진 상태 표현을 정렬하며, (3) 전체 훈련 단계에서 네트워크의 최종 출력으로 증류합니다. 이 과정은 Phi-Mamba 모델에서 3B 토큰과 하이브리드 Phi-Mamba 모델에서 5B 토큰만을 사용하여 구현되었습니다.

- **Performance Highlights**: Phi-Mamba 모델은 Winogrande 데이터셋에서 71.7%의 정확도를 달성하였으며, 이는 사전학습된 Mamba-2 모델의 60.9%보다 현저히 향상된 결과입니다. 또한, 하이브리드 모델은 Phi-1.5 모델의 67.2%와 근접한 66.0%의 성능을 나타냈습니다.



### Imbalance-Aware Culvert-Sewer Defect Segmentation Using an Enhanced Feature Pyramid Network (https://arxiv.org/abs/2408.10181)
- **What's New**: 이 논문은 불균형 데이터셋에서도 효과적으로 작동하는 Enhanced Feature Pyramid Network (E-FPN)라는 새로운 심층 학습 모델을 소개합니다. E-FPN은 오브젝트 변형을 잘 처리하고, 기능 추출을 개선하기 위한 건축 혁신을 포함합니다.

- **Technical Details**: E-FPN은 희소 연결 블록과 깊이 별개의 합성곱(depth-wise separable convolutions)을 사용하여 정보 흐름을 효율적으로 관리하고, 파라미터를 줄이면서도 표현력은 유지합니다. 또한, class decomposition(클래스 분해)와 data augmentation(데이터 증대) 전략을 통해 데이터 불균형 문제를 해결합니다.

- **Performance Highlights**: E-FPN은 수로와 하수관 결함 데이터셋 및 항공기 초상형 분할 드론 데이터셋에서 각각 평균 Intersection over Union (IoU) 개선치를 13.8% 및 27.2% 달성했으며, 클래스 분해와 데이터 증대 기법이 결합될 경우 약 6.9% IoU 성능 향상을 보였습니다.



### NeuRodin: A Two-stage Framework for High-Fidelity Neural Surface Reconstruction (https://arxiv.org/abs/2408.10178)
- **What's New**: NeuRodin은 고품질의 3D 표면 재구성을 위한 새로운 2단계 신경망 프레임워크로, SDF 기반 방법의 한계를 극복합니다. 이 방법은 일반적인 밀도 기반 방법의 유연한 최적화 특성을 유지하며, 세밀한 기하학 구조를 효과적으로 캡쳐합니다.

- **Technical Details**: NeuRodin은 두 가지 주요 문제를 해결합니다. 첫째, SDF-밀도 변환을 글로벌 스케일 매개변수에서 로컬 적응형 매개변수로 전환하여 유연성을 높입니다. 둘째, 새로운 손실 함수가 최대 확률 거리와 제로 레벨 세트를 정렬하여 기하학적 표현의 정렬성을 향상시킵니다. 이 과정은 coarse 및 refinement 과정의 두 단계로 나뉘어 진행됩니다.

- **Performance Highlights**: NeuRodin은 Tanks and Temples와 ScanNet++ 데이터 세트에서 이전 모든 방법들에 비해 우수한 성능을 발휘했으며, 특히 복잡한 토폴로지 구조 최적화와 세밀한 정보 보존에서 두각을 나타냈습니다. 제안된 방법은 VR 및 AR 시스템에서 활용될 수 있는 잠재력을 가지고 있습니다.



### Fairness Under Cover: Evaluating the Impact of Occlusions on Demographic Bias in Facial Recognition (https://arxiv.org/abs/2408.10175)
Comments:
          Accepted at ECCV Workshop FAILED

- **What's New**: 본 연구는 얼굴 인식 시스템의 공정성에 대한 차이가 인종 집단별로 어떻게 다르게 나타나는지 분석합니다. 특히, 합성된 현실적인 오클루전(occlusion)이 얼굴 인식 모델의 성능에 미치는 영향을 평가하고, 특히 아프리카계 인물에게 더 큰 영향을 미친다는 점을 강조합니다.

- **Technical Details**: Racial Faces in the Wild (RFW) 데이터셋을 사용하여 BUPT-Balanced 및 BUPT-GlobalFace 데이터셋으로 훈련된 얼굴 인식 모델의 성능을 평가합니다. FMR( False Match Rate), FNMR(False Non-Match Rate), 정확도와 같은 성능 지표의 분산이 증가하고, Equilized Odds, Demographic Parity, STD of Accuracy, Fairness Discrepancy Rate와 같은 공정성 지표는 감소한다는 점을 확인했습니다. 새로운 메트릭인 Face Occlusion Impact Ratio (FOIR)를 제안하여 오클루전이 얼굴 인식 모델의 성능에 미치는 영향을 정량화합니다.

- **Performance Highlights**: 오클루전이 존재할 때 아프리카계 인물에 대한 오클루전의 중요성이 다른 민족 그룹과 비교하여 더 높게 나타났습니다. 연구 결과는 일반적으로 얼굴 인식 시스템에 존재하는 인종적 편견을 증폭시키며, 공정성 평가에 대한 새로운 통찰력을 제공합니다.



### SMILE: Zero-Shot Sparse Mixture of Low-Rank Experts Construction From Pre-Trained Foundation Models (https://arxiv.org/abs/2408.10174)
Comments:
          Code is available at this https URL

- **What's New**: 이번 연구에서는 기존의 모델을 하나의 통합된 신뢰할 수 있는 Sparse MIxture of Low-rank Experts (SMILE) 모델로 융합하는 새로운 방법론을 제안합니다. 이 방법은 추가 데이터나 추가 훈련 없이도 소스 모델을 MoE (Mixture of Experts) 모델로 확장할 수 있게 해줍니다.

- **Technical Details**: 우리는 매트릭스 분해를 활용하여 파라미터 간의 간섭 문제를 최적화 문제로 정의합니다. 또한, zero-shot 접근 방식을 통해 기존 모델의 융합을 가능하게 하며, 이는 과적합 (overfitting) 문제를 감소시키는데 기여합니다.

- **Performance Highlights**: 전체 미세 조정(full fine-tuning)에서 약 50%의 추가 파라미터가 기존 8개의 미세 조정된 모델의 성능의 98-99%에 해당하는 효과를 보였습니다. LoRA (Low-Rank Adaptation) 성능 유지에서도 단 2%의 파라미터 증가로 99%의 성능을 유지할 수 있었습니다.



### Customizing Language Models with Instance-wise LoRA for Sequential Recommendation (https://arxiv.org/abs/2408.10159)
- **What's New**: 본 연구는 Instance-wise LoRA (iLoRA)라는 새로운 방법론을 제안하며 이를 통해 다양하고 개별화된 사용자 행동을 효과적으로 반영할 수 있는 추천 시스템을 구축하고자 한다.

- **Technical Details**: iLoRA는 Mixture of Experts (MoE) 프레임워크를 통합하여 Low-Rank Adaptation (LoRA) 모듈을 개선하는 방식으로 작동한다. 이 방법은 다양한 전문가를 생성하며 각 전문가는 특정 사용자 선호를 반영하도록 훈련된다. 또한, 게이팅 네트워크는 사용자 상호작용 시퀀스를 기반으로 맞춤형 전문가 참여 가중치를 생성하여 다채로운 행동 패턴에 적응한다.

- **Performance Highlights**: 세 가지 벤치마크 데이터셋(LastFM, MovieLens, Steam)에서의 실험을 통해 iLoRA가 기존 방법론들보다 탁월한 성능을 보이며 사용자별 선호를 효과적으로 캡처하고 추천의 정확성을 향상시켰음을 입증하였다.



### Rhyme-aware Chinese lyric generator based on GP (https://arxiv.org/abs/2408.10130)
- **What's New**: 본 논문에서는 기존의 사전 훈련된 언어 모델이 가사 생성 시 운율(rhyme) 정보를 주로 고려하지 않는 문제를 다룹니다. 이를 해결하기 위해, 운율 정보를 통합하여 성능을 향상시키는 방안을 제시합니다.

- **Technical Details**: 제안하는 모델은 사전 훈련(pre-trained)된 언어 모델을 기반으로 하며, 운율 통합(integrated rhyme) 기법을 적용하여 가사 생성 성능을 개선합니다. 이 모델은 대규모 코퍼스에 대해 사전 훈련되어 있어, 풍부한 의미적 패턴을 효과적으로 포착할 수 있습니다.

- **Performance Highlights**: 운율 정보를 통합한 모델을 사용하여 생성된 가사의 질이 현저하게 향상되었습니다. 이는 기존의 모델보다 우수한 자연어 생성 성능을 보여줍니다.



### Advancing Voice Cloning for Nepali: Leveraging Transfer Learning in a Low-Resource Languag (https://arxiv.org/abs/2408.10128)
Comments:
          7 pages, 10 figures

- **What's New**: 이 논문에서는 고급 AI 소프트웨어를 사용하여 인간의 음성을 복제하는 방법인 voice cloning의 발전에 대해 다룹니다. 특히, 적은 수의 오디오 샘플을 이용하여 비슷한 음성을 생성할 수 있는 신경망 기반 클로닝 시스템이 소개됩니다.

- **Technical Details**: 제안된 voice cloning 방법은 데이터 파일의 사전 처리를 포함하여 세 가지 주요 모델로 구성됩니다. 546명의 개인으로부터 수집된 <voice, text> 쌍 다중 화자 데이터셋을 통해 150,000개 이상의 오디오 파일이 준비되었습니다. 이 논문은 Mel-spectrogram을 사용한 인코더와 Tacotron2 아키텍처를 포함한 텍스트를 음성으로 변환하는 시스템을 사용하여 고품질 음성을 생성합니다.

- **Performance Highlights**: 모델의 훈련 결과에서, 클러스터 형성의 정확도를 나타내는 손실 곡선과 정확도 곡선이 주요 파라미터로 관찰되었습니다. 음성의 자연스러움과 기존 화자와의 유사성을 기준으로 비교했을 때, 적은 수의 클로닝 오디오로도 우수한 성능을 발휘했습니다.



### Molecular Graph Representation Learning Integrating Large Language Models with Domain-specific Small Models (https://arxiv.org/abs/2408.10124)
- **What's New**: 이번 연구에서는 Large Language Models (LLMs)와 Domain-specific Small Models (DSMs)의 장점을 통합한 새로운 분자 그래프 표현 학습 프레임워크인 MolGraph-LarDo를 제안합니다. 이는 분자 특성 예측 작업에서 기존 모델들이 가진 한계를 극복하고 전문 지식의 획득 비용을 줄이는 데 기여합니다.

- **Technical Details**: MolGraph-LarDo는 두 단계의 프롬프트 전략을 통해 DSM이 LLM이 제공하는 지식을 보정하여 도메인 특화 데이터를 더욱 정확하게 생성하도록 설계되었습니다. 이 프레임워크에서는 생물학적 도메인 지식을 효과적으로 활용하여 대규모의 라벨이 없는 데이터를 활용하는 방식으로 학습합니다.

- **Performance Highlights**: 실험 결과, MolGraph-LarDo는 분자 특성 예측 작업의 성능을 개선하고 생물학적 도메인 지식을 확보하는 데 드는 비용을 줄이는 데 효과적임이 입증되었습니다.



### Factorized-Dreamer: Training A High-Quality Video Generator with Limited and Low-Quality Data (https://arxiv.org/abs/2408.10119)
- **What's New**: 본 연구에서는 고품질 비디오 생성기(HQ video generator)가 공개된 제한적이고 저품질(LQ) 데이터만으로도 훈련될 수 있음을 보여줍니다. 특히, 'Factorized-Dreamer'라는 두 단계의 비디오 생성 과정을 제안합니다.

- **Technical Details**: 이 모델은 고도로 설명적인 캡션(Highly Descriptive Caption)을 바탕으로 이미지를 생성하고, 그 후 생성된 이미지 및 간결한 동작 세부 사항의 캡션을 바탕으로 비디오를 합성하는 방식으로 작동합니다. Factorized-Dreamer는 텍스트와 이미지 임베딩을 결합하는 어댑터와 픽셀 수준의 영상 정보를 캡처하는 픽셀 인식 교차 주의 모듈(pixel-aware cross attention module)을 포함합니다.

- **Performance Highlights**: WebVid-10M과 같은 LQ 데이터셋에서 직접 훈련할 수 있으며, 기존의 많은 T2V 모델보다 뛰어난 결과를 보여줍니다.



### Envisioning Possibilities and Challenges of AI for Personalized Cancer Car (https://arxiv.org/abs/2408.10108)
Comments:
          7 pages, 1 table, short paper at CSCW 2024

- **What's New**: 이 연구는 인공지능(AI)이 암 생존자들에게 제공하는 의료 서비스의 격차를 식별하고 해결할 수 있는 가능성을 탐구합니다. 특히 소수 인종 및 민족 그룹의 의료 서비스 접근에서의 불균형 문제에 주목하고, AI 기반의 개인화된 의료 접근 방식이 어떻게 이 문제를 해결할 수 있는지를 조명합니다.

- **Technical Details**: 연구는 6명의 암 생존자와의 반구조화된 인터뷰를 통해 그들의 정신 사회적 필요를 분석하였으며, 그 결과 개인화된 돌봄, 문화적 적합성, 정보의 투명성이 중요하다는 결론을 도출했습니다. 또한, AI의 개인화 구현에서 나타나는 데이터 프라이버시, 인간의 돌봄 손실, 다각적 정보 노출의 위험에 대한 우려를 언급합니다.

- **Performance Highlights**: 참여자들은 AI가 그들의 건강 정보를 투명하게 제공하고 개인의 필요에 맞춘 실시간 상호작용을 가능하게 할 것이라는 기대를 표명했습니다. 그러나, 제공되는 정보의 질과 신뢰성 부족으로 인해 감정적 건강에 미치는 부정적인 영향에 대한 우려도 있었습니다.



### Perturb-and-Compare Approach for Detecting Out-of-Distribution Samples in Constrained Access Environments (https://arxiv.org/abs/2408.10107)
Comments:
          Accepted to European Conference on Artificial Intelligence (ECAI) 2024

- **What's New**: MixDiff라는 새로운 OOD(Out-of-Distribution) 탐지 프레임워크를 제안, 모델 파라미터나 활성화 정보에 접근 불가능한 상황에서도 활용 가능.

- **Technical Details**: MixDiff는 동일한 입력 수준에서의 교란(perturbation)을 통해 타겟 샘플과 유사한 ID 샘플을 비교하여 OOD 샘플을 탐지하는 방법론을 제시한다. 구현 방식은 크게 세 가지 단계로 나뉜다: 1) 타겟 샘플에 Mixup을 적용하여 변형된 샘플 생성, 2) 해당 ID 샘플도 같은 방법으로 변형, 3) 두 변형된 샘플의 모델 출력 비교.

- **Performance Highlights**: MixDiff는 다양한 비전 및 텍스트 데이터셋에서 OOD 탐지 성능을 일관되게 향상시키며, 기존 OOD 점수 기반 방법과 통합이 가능함을 강조한다.



### Convert and Speak: Zero-shot Accent Conversion with Minimum Supervision (https://arxiv.org/abs/2408.10096)
Comments:
          9 pages, 4 figures, conference

- **What's New**: 이번 논문에서는 악센트 변환(accent conversion) 문제를 해결하기 위한 혁신적인 두 단계 생성 프레임워크 'convert-and-speak'를 제안합니다. 이 프레임워크는 의미 토큰(semantic token) 레벨에서만 변환을 수행하고, 변환된 의미 토큰에 조건을 두어 목표 악센트에서 음성을 생성하는 데 중점을 둡니다.

- **Technical Details**: 제안된 방법은 두 개의 주요 단계로 구성됩니다: 변환 단계와 음성 생성 단계. 변환 단계에서는 원본 악센트로부터 목표 악센트의 의미 토큰을 생성하며, 음성 생성 단계에서는 이들 변환된 의미 토큰을 기반으로 음성을 생성합니다. 또한, TF-Codec 기반의 단일 단계 자가 회귀 모델(autoregressive model)을 사용하여 효율적인 음성 생성을 구현합니다.

- **Performance Highlights**: 실험 결과, 인도 영어에서 일반 미국 영어로의 변환에서 제안된 프레임워크는 단 15분의 약한 병렬 데이터(weakly parallel data)를 사용하여 최첨단 성능을 달성하였으며, 다양한 악센트 유형에 대한 높은 적응성을 보여 다른 저자원 악센트로도 쉽게 확장될 수 있음을 입증하였습니다.



### No Screening is More Efficient with Multiple Objects (https://arxiv.org/abs/2408.10077)
- **What's New**: 이번 논문에서는 이질적인 물품을 효율적으로 할당하는 메커니즘 설계를 연구하였습니다. 여기서 우리는 잔여 잉여(residual surplus)를 극대화하는 것을 목표로 하며, 이는 할당에서 발생하는 총 가치(total value)에서 에이전트의 가치를 스크리닝(screening)하는 비용을 뺀 값입니다. 흥미롭게도, 물품의 다양성이 증가할수록 외부 우선 순위인(serial dictatorship with exogenous priority order) 무스크리닝 메커니즘이 더 효과적인 경향을 보인다는 것을 발견했습니다.

- **Technical Details**: 논문에서는 이론적인 환경에서 효율적인 메커니즘을 특성화하여 이러한 경향의 원인을 분석하였습니다. 또한 자동화된 메커니즘 설계(automated mechanism design) 접근 방식을 적용하여 효율적인 메커니즘을 수치적으로 도출하였고, 일반 환경에서도 이 경향을 검증하였습니다.

- **Performance Highlights**: 이 연구의 함의를 바탕으로 팬데믹(pandemic) 질병에 대한 예방접종을 스케줄링하는 효율적인 시스템인 등록 초대 시스템(register-invite-book system, RIB)을 제안하였습니다.



### Personalizing Reinforcement Learning from Human Feedback with Variational Preference Learning (https://arxiv.org/abs/2408.10075)
Comments:
this http URL

- **What's New**: 본 논문은 Reinforcement Learning from Human Feedback (RLHF) 기술을 발전시켜, 다양한 사용자들의 특성과 선호도에 맞춘 새로운 다중 모드 RLHF 방법론을 제안합니다. 본 기술은 사용자 특정 잠재 변수를 추론하고, 이 잠재 변수에 조건화된 보상 모델과 정책을 학습하는 방식으로, 단일 사용자 데이터 없이도 이루어집니다.

- **Technical Details**: 제안된 방법, Variational Preference Learning (VPL),은 변량 추론(variational inference) 기법을 계승하여 다중 모드 보상 모델링을 수행합니다. 사용자로부터 받은 선호 정보를 바탕으로 잠재 사용자 맥락에 대한 분포를 추론하고 이를 통해 다중 모드 선호 분포를 회복하는 것이 특징입니다. 이를 통해 RLHF의 비효율성과 잘못된 보상 모델을 수정합니다.

- **Performance Highlights**: 실험을 통해 VPL 방법이 다중 사용자 선호를 반영한 보상 함수를 효과적으로 모델링하고, 성능을 10-25% 향상시킬 수 있음을 입증했습니다. 시뮬레이션 로봇 환경과 언어 작업 모두에서 보상 예측의 정확성이 향상되어 diversa (diverse) 사용자 선호를 만족시키는 정책의 학습 성능이 크게 개선되었습니다.



### Synthesis of Reward Machines for Multi-Agent Equilibrium Design (Full Version) (https://arxiv.org/abs/2408.10074)
- **What's New**: 이 논문은 기계 설계(mechanism design)와 관련된 새로운 개념인 평형 설계(equilibrium design)를 다룬다. 평형 설계에서는 디자이너의 권한이 제한적이며, 새로운 게임을 만드는 대신 주어진 게임의 인센티브 구조만 변경할 수 있다.

- **Technical Details**: 논문에서는 동적인 인센티브 구조(dynamic incentive structures)인 리워드 머신(reward machines)을 이용하여 평형 설계 문제를 연구한다. 게임 모델로는 가중 동시 게임 구조(weighted concurrent game structures)를 사용하며, 목표는 평균 보상(mean-payoff)으로 정의된다. 리워드 머신이 디자이너의 목표를 최적화하는 방식으로 보상을 배분하는 방법을 제시한다.

- **Performance Highlights**: 주요 결정 문제인 보상 개선 문제(payoff improvement problem)를 도입하고, 이 문제는 주어진 임계값보다 디자이너의 보상을 개선할 수 있는 리워드 머신이 존재하는지를 묻는다. 강한 변형과 약한 변형의 두 가지를 제시하며, 두 문제 모두 NP 오라클이 장착된 튜링 머신(Turing machine)을 이용해 다항 시간에 해결할 수 있음을 보여준다. 또한 이들 변형이 NP-hard 또는 coNP-hard임을 입증하고, 존재할 경우 해당 리워드 머신을 합성하는 방법도 제시한다.



### FFAA: Multimodal Large Language Model based Explainable Open-World Face Forgery Analysis Assistan (https://arxiv.org/abs/2408.10072)
Comments:
          17 pages, 18 figures; project page: this https URL

- **What's New**: 본 논문에서는 효율적인 얼굴 위조(face forgery) 분석을 위해 새로운 OW-FFA-VQA(Open-World Face Forgery Analysis VQA) 작업과 벤치마크를 도입하였습니다. 이를 통해 사용자 친화적이면서도 이해하기 쉬운 인증 분석을 제공합니다.

- **Technical Details**: OW-FFA-VQA 작업은 기존의 이진 분류(task) 문제를 넘어, 다양한 실제 및 위조 얼굴 이미지의 설명과 신뢰할 수 있는 위조 추론을 포함하는 VQA(Visual Question Answering) 데이터셋을 기반으로 합니다. FFAA(Face Forgery Analysis Assistant)는 정밀 조정된 MLLM(Multimodal Large Language Model)과 MIDS(Multi-answer Intelligent Decision System)로 구성되며, 다양한 가설을 기반으로 유연한 반응을 제공하는 기능을 갖추고 있습니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, 우리의 방법은 이전 방법들에 비해 정확성과 견고성을 크게 향상시키면서 사용자 친화적인 설명 가능한 결과를 제공합니다. 또한 FFAA는 복잡한 환경에서도 뛰어난 일반화 능력을 발휘하여 원본과 위조된 이미지를 효과적으로 구별하는 성능을 보여주었습니다.



### Facial Wrinkle Segmentation for Cosmetic Dermatology: Pretraining with Texture Map-Based Weak Supervision (https://arxiv.org/abs/2408.10060)
- **What's New**: 이번 논문에서는 최초의 공개 얼굴 주름 데이터셋인 'FFHQ-Wrinkle'을 소개하며, 얼굴 주름 자동 탐지를 위한 새로운 훈련 전략을 제안합니다. 이 데이터셋은 1,000장의 인간 레이블이 있는 이미지와 50,000장의 자동 생성된 약한 레이블이 포함되어 있습니다.

- **Technical Details**: U-Net과 Swin UNETR 아키텍처를 사용하여 주름 감지 모델을 훈련합니다. 두 단계의 훈련 전략을 통해 약한 레이블 데이터로 사전 훈련을 수행한 후, 인간 레이블 데이터로 미세 조정(finetuning)하여 주름 감지 성능을 향상시킵니다. U-Net 모델 구조는 네 개의 인코더 블록과 네 개의 디코더 블록으로 이루어져 있습니다.

- **Performance Highlights**: 본 방법론은 기존의 사전 훈련 방법과 비교하여 정량적 및 시각적 측면에서 얼굴 주름 분할 성능을 향상시키는 데 성공하였습니다. 이 연구는 주름 검출 알고리즘 개발을 위한 기준 데이터셋 제공으로 향후 연구에 기여할 것입니다.



### Edge-Cloud Collaborative Motion Planning for Autonomous Driving with Large Language Models (https://arxiv.org/abs/2408.09972)
- **What's New**: EC-Drive라는 새로운 엣지-클라우드 협업 자율 주행 시스템을 소개하며, 데이터 드리프트 검출 기능을 활용하여 주행 중 새로운 장애물이나 교통 패턴 변경과 같은 중요한 데이터를 효율적으로 클라우드로 업로드합니다.

- **Technical Details**: EC-Drive는 드리프트 검출 알고리즘을 사용하여 중요한 데이터만 클라우드로 전송하고, 나머지 데이터는 엣지 디바이스의 작은 LLMs가 처리하여 추론 지연 시간을 줄입니다. 이 시스템은 자연어 명령을 이해하여 운전 전략을 개인화할 수 있도록 합니다.

- **Performance Highlights**: 실험으로 확인된 EC-Drive의 강력한 처리 능력은 실제 주행 환경에서의 적용 가능성을 보여주며, 엣지-클라우드 협력 체계의 실용적인 장점이 강조됩니다.



### Unsupervised Machine Learning Hybrid Approach Integrating Linear Programming in Loss Function: A Robust Optimization Techniqu (https://arxiv.org/abs/2408.09967)
- **What's New**: 본 논문에서는 비지도 학습 모델의 손실 함수(loss function)에 선형 계획법(linear programming, LP)을 통합한 새로운 하이브리드 접근법을 제안합니다. 이 방법은 최적화 기법과 기계 학습의 장점을 결합하여 전통적인 방법이 한계를 보일 수 있는 복잡한 최적화 문제를 해결하는 강력한 프레임워크를 제공합니다.

- **Technical Details**: 제안된 접근법은 선형 계획 문제의 제약 조건(constraints)과 목표(objectives)를 손실 함수에 직접 포함하여, 이러한 제약을 준수하면서 원하는 결과를 최적화하는데 필요한 학습 프로세스를 유도합니다. 이 기술은 선형 계획법의 해석 가능성(interpretability)을 유지하면서도 기계 학습의 유연성(flexibility)과 적응성(adaptability)의 이점을 누릴 수 있습니다.

- **Performance Highlights**: 이 기법은 특히 비지도 학습(unsupervised learning) 또는 반지도 학습(semi-supervised learning) 시나리오에 적합하며, 복잡한 최적화 문제를 보다 효율적으로 해결하는 데 기여할 수 있음을 보여줍니다.



### AdaResNet: Enhancing Residual Networks with Dynamic Weight Adjustment for Improved Feature Integration (https://arxiv.org/abs/2408.09958)
- **What's New**: 이 논문은 AdaResNet(Auto-Adapting Residual Network)라는 새로운 아키텍처를 제안하여, ResNet의 성능을 극대화하는 방법을 보여줍니다. 특히, 입력 데이터의 가중치와 변환된 데이터의 가중치 간 비율을 자동으로 조정할 수 있는 가능성을 탐구합니다.

- **Technical Details**: AdaResNet은 튜닝 가능한 가중치(weight) 매개변수(w_{tfd}^{ipd})를 도입하여, 훈련 데이터에 따라 입력 데이터(ipd)와 변환된 데이터(tfd)의 비율을 동적으로 조정합니다. 이는 기존 ResNet의 1:1 고정 비율 조합을 넘어, 다양한 데이터 배포에 적응할 수 있는 능력을 제공합니다.

- **Performance Highlights**: 실험 결과, AdaResNet은 전통적인 ResNet에 비해 최대 50% 이상의 정확도 향상을 달성했습니다.



### Weakly Supervised Pretraining and Multi-Annotator Supervised Finetuning for Facial Wrinkle Detection (https://arxiv.org/abs/2408.09952)
- **What's New**: 이번 연구는 피부 질병 및 피부 미용에 대한 관심이 높아짐에 따라 얼굴 주름 예측의 중요성이 커지고 있는 점에 주목하였습니다. 이 연구는 얼굴 주름 분할을 위해 convolutional neural networks (CNN) 모델을 자동으로 훈련할 수 있는지를 평가합니다.

- **Technical Details**: 연구에서는 여러 주석자(annotator)로부터의 데이터를 통합하는 효과적인 기술을 제시했으며, transfer learning을 활용하여 성능을 향상시킬 수 있다는 것을 보여주었습니다. 이를 통해 얼굴 주름의 신뢰할 수 있는 분할 결과를 얻을 수 있습니다.

- **Performance Highlights**: 이 접근 방식은 주름 분석과 같은 복잡하고 시간 소모적인 작업을 자동화할 수 있는 가능성을 제시하며, 피부 치료 및 진단을 용이하게 하는 데 활용될 수 있습니다.



### Caption-Driven Explorations: Aligning Image and Text Embeddings through Human-Inspired Foveated Vision (https://arxiv.org/abs/2408.09948)
- **What's New**: 본 연구에서는 CapMIT1003 데이터를 도입하여 캡션 작업 중 인류의 주의력을 연구하고, CLIP 모델과 NeVA 알고리즘을 결합한 NevaClip이라는 제로샷(zero-shot) 방법을 제안합니다. 이 방법을 통해 캡션과 시각적 스캔패스를 예측할 수 있습니다.

- **Technical Details**: CapMIT1003 데이터셋은 참가자가 이미지에 클릭하여 캡션을 제공하면서 수행하는 클릭 중심(click-contingent) 이미지 탐사를 통해 수집되었습니다. NevaClip 알고리즘은 과거 예측된 고정점(fixation)을 기반으로 이미지를 흐리게 만들어 주의 메커니즘을 적용하여 주어진 캡션과 시각적 자극의 표현을 정렬합니다.

- **Performance Highlights**: 실험 결과 NevaClip은 기존 인간 주의력 모델보다 캡션 작성 및 자유 탐사 작업에서 더 뛰어난 결과를 보였습니다. CapMIT1003 데이터셋을 통해 얻은 스캔패스는 최신 모델을 초월하는 성과를 달성했습니다.



### Benchmarking LLMs for Translating Classical Chinese Poetry:Evaluating Adequacy, Fluency, and Eleganc (https://arxiv.org/abs/2408.09945)
Comments:
          Work in progress

- **What's New**: 이번 연구는 고전 중국 시를 영어로 번역하는 새로운 벤치마크를 소개하며, 대형 언어 모델(LLM)이 요구되는 번역의 적합성, 유창성, 우아함을 충족시키지 못함을 밝혀냈습니다. 이를 개선하기 위해 RAT(Recovery-Augmented Translation) 방법론을 제안하였습니다.

- **Technical Details**: RAT는 고전 시와 관련된 지식을 검색하여 번역 품질을 향상시키는 방법입니다. 연구에서는 GPT-4를 기반으로 한 새로운 자동 평가 지표를 도입하여 번역의 적합성, 유창성, 우아함을 평가합니다. 데이터셋은 1200개의 고전 시와 608개의 수동 번역으로 구성됩니다.

- **Performance Highlights**: RAT 방법은 번역 과정에서 고전 시와 관련된 knowledge를 활용하여 번역의 품질을 개선시켜주며, 평가 결과는 기존의 자동 평가 방식보다 LLM 기반 번역의 성능을 더 잘 측정합니다.



### SZU-AFS Antispoofing System for the ASVspoof 5 Challeng (https://arxiv.org/abs/2408.09933)
Comments:
          8 pages, 2 figures, ASVspoof 5 Workshop (Interspeech2024 Satellite)

- **What's New**: 이 논문은 ASVspoof 5 챌린지의 열린 조건에서 Track 1을 위한 SZU-AFS 안티 스푸핑 시스템을 소개합니다. 시스템은 네 개의 단계로 구성되어 있으며, 이는 기초 모델 선택, 효과적인 데이터 증강(DA) 방법 탐색, 기울기 노름 인식 최소화(GAM)를 기반으로 한 공동 향상 전략 적용, 그리고 두 개의 최상의 모델에서 추출된 로짓 점수를 융합하는 과정으로 나뉩니다.

- **Technical Details**: SZU-AFS 시스템은 Wav2Vec2 프론트엔드 특성 추출기와 AASIST 백엔드 분류기를 기초 모델로 사용합니다. 모델의 미세 조정 과정에서는 세 가지 DA 정책(싱글-DA, 랜덤-DA, 카스케이드-DA)을 적용하였고, GAM 기반의 공동 향상 전략을 통해 Adam 옵티마이저가 더 평평한 최소값을 찾을 수 있도록 하여 모델의 일반화를 향상시켰습니다.

- **Performance Highlights**: 최종 융합 시스템은 평가 세트에서 minDCF 0.115와 EER 4.04%의 성능을 달성하였습니다.



### Preoperative Rotator Cuff Tear Prediction from Shoulder Radiographs using a Convolutional Block Attention Module-Integrated Neural Network (https://arxiv.org/abs/2408.09894)
- **What's New**: 이번 연구에서는 평면 어깨 엑스레이(shoulder radiograph)와 딥 러닝(deep learning) 방법을 조합해 회전근개 파열(rotator cuff tears) 환자를 식별할 수 있는지를 테스트했습니다.

- **Technical Details**: 우리는 convolutional block attention modules를 딥 뉴럴 네트워크(deep neural network)에 통합하여 모델을 개발했습니다. 이 모델은 회전근개 파열을 검출하는 데 있어 높은 정확도를 보였으며, 평균 AUC(area under the curve) 0.889 및 0.831의 정확성을 달성했습니다.

- **Performance Highlights**: 본 연구는 딥 러닝 모델이 엑스레이에서 회전근개 파열을 정확히 감지할 수 있음을 검증하였습니다. 이는 MRI와 같은 더 비싼 이미징 기술에 대한 유효한 사전 평가(pre-assessment) 또는 대안이 될 수 있습니다.



### New spectral imaging biomarkers for sepsis and mortality in intensive car (https://arxiv.org/abs/2408.09873)
Comments:
          Markus A. Weigand, Lena Maier-Hein and Maximilian Dietrich contributed equally

- **What's New**: 이 연구는 hyperspectral imaging (HSI)을 활용하여 패혈증(sepsis) 진단과 사망률 예측을 위한 새로운 생체 지표(biomarker)를 제시한다. 기존의 진단 방안들이 가진 한계를 극복하기 위한 노력의 일환으로, HSI 기반의 접근법이 실용적이고 비침습적(non-invasive)이며 빠른 진단이 가능하다는 점에 주목했다.

- **Technical Details**: 480명 이상의 환자에서 HSI 데이터를 수집하여, 손바닥과 손가락에서 패혈증을 예측하는 알고리즘을 개발했다. HSI 측정치는 패혈증 관련하여 0.80의 AUROC, 사망률 예측에 대해서는 0.72의 AUROC를 달성하였다. 임상 데이터를 추가하면 패혈증에 대해 0.94, 사망률에 대해 0.84의 AUROC까지 향상됨을 보였다.

- **Performance Highlights**: HSI는 기존의 생체 지표와 비교하여 신뢰성 높은 예측 성능을 제공하며, 특히 중환자실(ICU) 환자에 대한 신속하고 비침습적인 진단 방법으로서의 가능성을 보여준다. 이 연구는 머신러닝을 통해 HSI 데이터를 분석하여 패혈증과 사망률 예측에 효과적인 새로운 접근 방식을 제시하였다.



### 3D-Aware Instance Segmentation and Tracking in Egocentric Videos (https://arxiv.org/abs/2408.09860)
- **What's New**: 본 논문은 3D 인식(3D awareness)을 활용한 첫 번째 인물 비디오에서의 인스턴스 분할(instance segmentation) 및 추적(tracking) 방법을 제안합니다. 이는 빠른 카메라 동작과 잦은 객체 가림(occlusion)으로 인해 발생하는 도전과제를 극복하기 위한 것입니다.

- **Technical Details**: 제안된 방법은 비디오 프레임에서 장면 기하학(scene geometry) 및 3D 객체 중심 위치(centroid tracking)를 통합하여 동적인 개인 중심 비디오를 분석할 수 있는 강력한 프레임워크를 만듭니다. 이 방법은 공간적(spatial) 및 시간적(temporal) 신호를 포함하여, 2D 기반의 기존 방법들보다 우수한 성능을 발휘합니다.

- **Performance Highlights**: EPIC Fields 데이터셋에 대한 광범위한 평가를 통해, 우리의 방법은 차세대 방법보다 7점 높은 협회 정확도(AssA) 및 4.5점 높은 IDF1 점수를 기록했습니다. 또한 다양한 객체 범주에서 ID 전환(ID switches)의 수를 73%에서 80%까지 줄였습니다.



### TeamLoRA: Boosting Low-Rank Adaptation with Expert Collaboration and Competition (https://arxiv.org/abs/2408.09856)
- **What's New**: TeamLoRA는 Parameter-Efficient Fine-Tuning (PEFT) 방식의 새로운 접근법으로, 효율성과 효과성을 동시에 개선하기 위해 전문가 간 협업 및 경쟁 모듈을 도입합니다.

- **Technical Details**: TeamLoRA는 두 가지 주요 구성 요소로 이루어져 있습니다: (i) 효율적 협업 모듈에서는 비대칭 매트릭스 아키텍처를 활용하여 A와 B 매트릭스 간의 지식 공유 및 조직을 최적화합니다; (ii) 경쟁 모듈은 게임 이론에 기반한 상호작용 메커니즘을 통해 특정 다운스트림 작업에 대한 도메인 지식 전달을 촉진합니다.

- **Performance Highlights**: 실험 결과, TeamLoRA는 기존의 MoE-LoRA 방식보다 더 높은 성능과 효율성을 보이며, 2.5백만 샘플로 구성된 다양한 도메인과 작업 유형을 포함하는 종합적인 평가 기준(CME)에서 그 효과성을 입증했습니다.



### Self-Directed Turing Test for Large Language Models (https://arxiv.org/abs/2408.09853)
- **What's New**: 본 연구는 전통적인 Turing 테스트의 단점을 해결하기 위해 Self-Directed Turing Test라는 새로운 프레임워크를 제안합니다. 이 테스트는 다중 연속 메시지를 통한 더 역동적인 대화를 허용하며, LLM이 대화의 대부분을 스스로 지시할 수 있도록 설계되었습니다.

- **Technical Details**: Self-Directed Turing Test는 burst dialogue 형식을 통해 자연스러운 인간 대화를 보다 잘 반영합니다. 이 과정에서 LLM은 대화의 진행을 스스로 생성하고, 마지막 대화 턴에 대한 인간과의 짧은 대화를 통해 평가합니다. 새롭게 도입된 X-Turn Pass-Rate 메트릭은 LLM의 인간 유사성을 평가하기 위한 기준이 됩니다.

- **Performance Highlights**: 초기에는 GPT-4와 같은 LLM들이 3회 대화 턴에서 51.9%, 10회에서는 38.9%의 수치로 테스트를 통과했지만 대화가 진행됨에 따라 성능이 하락하는 경향을 보였으며, 이는 장기 대화에서 일관성을 유지하는 것이 어렵다는 것을 강조합니다.



### Importance Weighting Can Help Large Language Models Self-Improv (https://arxiv.org/abs/2408.09849)
- **What's New**: 이 논문에서는 LLM(self-generated data)의 self-improvement를 위해 DS weight라는 새로운 메트릭(또는 척도)을 제안하여 모델 성능 향상을 위한 중요한 필터링 전략을 제시합니다.

- **Technical Details**: DS weight는 LLM의 데이터 분포 변화 정도(Distribution Shift Extent, DSE)를 근사하는 메트릭으로, Importance Weighting 방법에서 영감을 받아 개발되었습니다. 이 메트릭을 활용하여 데이터 필터링 전략을 구축하고, self-consistency와 결합하여 최신 LLM을 fine-tune 합니다.

- **Performance Highlights**: 제안된 접근 방식인 IWSI(Importance Weighting-based Self-Improvement)는 적은 양의 유효 데이터(Training Set의 5% 이하)를 사용하더라도 현재 LLM self-improvement 방법들의 추론 능력을 상당히 향상시키며, 이는 기존의 외부 감독(pre-trained reward models)을 통한 성능과 동등한 수준입니다.



### Segment-Anything Models Achieve Zero-shot Robustness in Autonomous Driving (https://arxiv.org/abs/2408.09839)
Comments:
          Accepted to IAVVC 2024

- **What's New**: 이번 연구는 자율 주행을 위한 세만틱 세그멘테이션에서 Segment Anything Model(SAM)의 제로샷 적대적 견고성(zero-shot adversarial robustness)을 심층적으로 조사합니다. SAM은 특정 객체에 대한 추가 교육 없이 다양한 이미지를 인식하고 세그멘테이션할 수 있는 통합된 이미지 세그멘테이션 프레임워크입니다.

- **Technical Details**: 이 연구는 SAM 모델의 제로샷 적대적 견고성을 평가하기 위해 Cityscapes 데이터셋을 사용하여 실험을 진행했습니다. 모델 평가에는 CNN 및 ViT(as Vision Transformer) 모델이 포함되며, SAM 모델은 언어 인코더의 제약 조건 하에 평가되었습니다. 연구는 화이트박스(white-box) 공격과 블랙박스(black-box) 공격을 통해 적대적 견고성을 평가하였습니다.

- **Performance Highlights**: 실험 결과, SAM의 제로샷 적대적 견고성이 비교적 문제없는 것으로 나타났습니다. 이는 거대한 모델 매개변수와 대규모 학습 데이터가 적대적 견고성 확보에 기여하는 현상을 보여줍니다. 연구 결과는 SAM이 인공지능 일반화 모델(AGI)의 초기 프로토타입으로서 향후 안전한 자율주행 구현에 중요한 통찰을 제공함을 강조합니다.



### CMoralEval: A Moral Evaluation Benchmark for Chinese Large Language Models (https://arxiv.org/abs/2408.09819)
Comments:
          Accepted by ACL 2024 (Findings)

- **What's New**: 본 연구에서는 CMoralEval이라는 대규모 벤치마크 데이터셋을 제안하여 중국어 대형 언어 모델(LLM)의 도덕성 평가를 위한 자료를 제공합니다. 이 데이터셋은 중국 사회의 도덕 규범을 기반으로 다양한 도덕적 상황을 다룹니다.

- **Technical Details**: CMoralEval은 1) 중국 TV 프로그램을 통해 수집된 도덕적 이야기 및 2) 신문 및 학술지에서 수집된 중국 도덕 불일치 데이터를 포함하여 30,388개의 도덕 사례를 포함합니다. 사례는 가족 도덕성, 사회 도덕성, 직업 윤리, 인터넷 윤리 및 개인 도덕성의 다섯 가지 범주로 구분됩니다.

- **Performance Highlights**: CMoralEval에 대한 실험 결과, 다양한 중국어 LLMs가 평가되었으며, 이는 이 데이터셋이 도덕성 평가에 있어 도전적인 벤치마크임을 보여줍니다. 이 데이터셋은 공개적으로 제공됩니다.



### Contextual Dual Learning Algorithm with Listwise Distillation for Unbiased Learning to Rank (https://arxiv.org/abs/2408.09817)
Comments:
          12 pages, 2 figures

- **What's New**: 본 논문에서는 Baidu의 웹 검색 로그를 기반으로 한 실제 클릭 데이터에서 기존의 Unbiased Learning to Rank (ULTR) 방법의 효과를 평가하였습니다. 새로운 Contextual Dual Learning Algorithm with Listwise Distillation (CDLA-LD)을 제안하여 위치 편향(position bias)과 맥락 편향(contextual bias)을 동시에 해결합니다.

- **Technical Details**: CDLA-LD 알고리즘은 두 가지 랭킹 모델을 포함합니다: (1) self-attention 메커니즘을 활용하는 listwise-input 랭킹 모델과 (2) pointwise-input 랭킹 모델입니다. DLA를 사용하여 unbiased listwise-input 랭킹 모델과 pointwise-input 모델을 동시에 학습하며, listwise 방식으로 지식을 증류(distill)합니다.

- **Performance Highlights**: 실험 결과, CDLA-LD는 기존의 ULTR 방법들에 비해 뛰어난 성능을 보여주었으며, 다양한 방법들의 propensity 학습을 비교 분석했습니다. 또한, 실제 검색 로그를 활용한 연구에 대한 새로운 통찰을 제공합니다.



### Propagating the prior from shallow to deep with a pre-trained velocity-model Generative Transformer network (https://arxiv.org/abs/2408.09767)
- **What's New**: VelocityGPT는 Transformer decoder를利用한 새로운 프레임워크로, 지하속속도 모델을 상위에서 하위로 생성하는 능력을 제공합니다. 기존의 CNN 및 RNN에 비해 더 효율적이고 유연한 학습이 가능합니다.

- **Technical Details**: VelocityGPT는 overlapping patches의 속도 모델을 이산 형태로 변환하기 위해 VQ-VAE(Vector-Quantized Variational Auto Encoder)를 사용하고, Transformer decoder를 통해 예측을 수행합니다. 첫 번째 단계에서는 VQ-VAE로 변환된 이산 데이터가 GPT 모델에 입력되며, 이를 통해 속도 모델의 깊은 레이어를 예측할 수 있습니다.

- **Performance Highlights**: OpenFWI 벤치마킹 데이터셋을 사용한 테스트에서 VelocityGPT는 강력한 속도 모델 생성기로서의 잠재력을 입증했습니다. 소규모 속도 모델에 대한 학습으로부터 보다 현실적인 크기의 속도 모델을 생성할 수 있음을 보여주었습니다.



### Event Stream based Human Action Recognition: A High-Definition Benchmark Dataset and Algorithms (https://arxiv.org/abs/2408.09764)
Comments:
          In Peer Review

- **What's New**: CeleX-HAR 데이터셋은 해상도가 1280x800인 대규모 인간 행동 인식 데이터셋으로, 총 124,625개의 비디오 시퀀스를 포함하고 있으며 150개의 일반적인 행동 카테고리를 포괄합니다.

- **Technical Details**: EVMamba라는 새로운 비전 백본 네트워크를 제안하며, 이는 공간 평면 다중 방향 스캔과 새로운 복셀 시계열 스캔 메커니즘을 갖추고 있습니다.

- **Performance Highlights**: CeleX-HAR 데이터셋을 기반으로 훈련된 20개 이상의 인식 모델은 성능 비교를 위한 좋은 플랫폼을 제공합니다.



### Revisiting Reciprocal Recommender Systems: Metrics, Formulation, and Method (https://arxiv.org/abs/2408.09748)
Comments:
          KDD 2024

- **What's New**: 이 논문은 Reciprocal Recommender Systems (RRS)을 체계적으로 재조명하고, 새로운 평가 지표와 방법론을 제시합니다. 특히, 이전 연구에서는 각 측면을 독립적으로 평가하였으나, 본 연구는 양측의 추천 결과가 시스템의 효과성에 미치는 영향을 포괄적으로 고려합니다.

- **Technical Details**: 연구는 세 가지 관점에서 RRS의 성능을 종합적으로 평가하는 5개의 새로운 지표를 제안합니다: 전체 커버리지(Overall Coverage), 양측 안정성(Bilateral Stability), 그리고 균형 잡힌 순위(Balanced Ranking). 또한, 잠재적 결과 프레임워크를 사용해 모델 비의존적인 인과적 RRS 방법인 Causal Reciprocal Recommender System (CRRS)를 개발하였습니다.

- **Performance Highlights**: 두 개의 실제 데이터셋에서 시행된 광범위한 실험을 통해 제안한 지표와 방법의 효과성을 입증하였습니다. 특히, 추천의 중복성을 고려하는 새로운 평가 방식이 RRS의 전반적인 성능 향상에 기여함을 보여주었습니다.



### Enhanced Cascade Prostate Cancer Classifier in mp-MRI Utilizing Recall Feedback Adaptive Loss and Prior Knowledge-Based Feature Extraction (https://arxiv.org/abs/2408.09746)
- **What's New**: 이 논문은 전립선암(mpMRI) 진단에서 임상 정보를 통합한 자동 등급 분류 솔루션을 제안합니다. 기존 연구는 샘플 분포의 불균형 문제를 해결하지 못하고 있어, 이에 대한 개선이 이루어진 것입니다.

- **Technical Details**: 우리는 기존의 PI-RADS 기준을 수학적으로 모델링하여 모델 훈련에 진단 정보를 통합하는 Prior Knowledge-Based Feature Extraction 을 도입하며, 극심한 불균형 데이터 문제를 해결하기 위한 Adaptive Recall Feedback Loss 를 제안합니다. 마지막으로, Enhanced Cascade Prostate Cancer Classifier 를 설계하여 전립선암을 여러 레벨로 분류합니다.

- **Performance Highlights**: 본 연구는 PI-CAI 데이터셋에서 실험 검증을 통해 다른 방법들보다 더 균형 잡힌 정확도와 Recall을 보이며 우수한 성과를 기록하였습니다.



### R2GenCSR: Retrieving Context Samples for Large Language Model based X-ray Medical Report Generation (https://arxiv.org/abs/2408.09743)
Comments:
          In Peer Review

- **What's New**: 본 논문은 X-ray 의료 보고서 생성을 위한 새로운 프레임워크인 R2GenCSR를 제안합니다. 이는 컨텍스트 샘플을 활용하여 대형 언어 모델(LLMs)의 성능을 향상시키는 방법을 제공합니다.

- **Technical Details**: Mamba라는 비전 백본을 도입하여 선형 복잡도로 높은 성능을 달성하며, 트레이닝 단계에서 각 미니 배치의 샘플을 위한 컨텍스트 샘플을 검색하여 기능 표현을 강화합니다. 시각 토큰과 컨텍스트 정보를 LLM에 공급하여 고품질 의료 보고서를 생성합니다.

- **Performance Highlights**: IU-Xray, MIMIC-CXR, CheXpert Plus라는 세 개의 X-ray 보고서 생성 데이터셋에서 광범위한 실험을 진행하였으며, 제안된 프레임워크의 효과성을 완벽하게 검증하였습니다.



### Paired Completion: Flexible Quantification of Issue-framing at Scale with LLMs (https://arxiv.org/abs/2408.09742)
Comments:
          9 pages, 4 figures

- **What's New**: 이번 연구에서는 소수의 예시를 이용하여 이슈 프레이밍(issue framing)과 내러티브 분석(narrative analysis)을 효과적으로 탐지할 수 있는 새로운 방법인 'paired completion'을 개발했습니다. 이는 기존의 자연어 처리(NLP) 접근법보다 더 정확하고 효율적입니다.

- **Technical Details**: 'Paired completion' 방식은 생성형 대형 언어 모델(generative large language models)에서 유래된 다음 토큰 로그 확률(next-token log probabilities)을 활용하여, 특정 이슈에 대한 텍스트의 프레이밍 정렬 여부를 판별합니다. 이 방법은 몇 개의 예시만으로도 높은 정확성을 제공합니다.

- **Performance Highlights**: 192개의 독립 실험을 통해, 'paired completion' 방법이 기존의 프롬프트 기반 방법(prompt-based methods) 및 전통적인 NLP 방식보다 우수한 성능을 보인다는 것을 증명했습니다. 특히, 적은 양의 데이터 환경에서도 높은 성과를 달성했습니다.



### Mutually-Aware Feature Learning for Few-Shot Object Counting (https://arxiv.org/abs/2408.09734)
Comments:
          Submitted to Pattern Recognition

- **What's New**: 이 연구는 다중 클래스 시나리오에서의 객체 식별 문제에 대한 새로운 접근방식을 제시합니다. 기존의 few-shot object counting 방법들이 가지고 있었던 '타겟 혼동(target confusion)' 문제를 해결하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 구제안하는 프레임워크인 Mutually-Aware FEAture Learning(MAFEA)은 쿼리 이미지와 예시 이미지 간의 상호 작용을 통해 특징을 추출합니다. MAFEA는 cross-attention을 활용하여 쿼리와 예시 특징 간의 양방향 관계를 포착하고, self-attention을 통해 내부 관계를 반영합니다. 또한, 배경 토큰(background token)을 도입하여 배경과 타겟 객체 간의 구분을 명확히 합니다.

- **Performance Highlights**: MAFEA는 FSCD-LVIS와 FSC-147의 두 가지 도전적인 벤치마크에서 새로운 선두 성과를 달성했습니다. 실험 결과, 타겟 혼동 문제의 정도가 현저히 줄어들었음을 보여줍니다.



### Pedestrian Attribute Recognition: A New Benchmark Dataset and A Large Language Model Augmented Framework (https://arxiv.org/abs/2408.09720)
Comments:
          MSP60K PAR Benchmark Dataset, LLM based PAR model, In Peer Review

- **What's New**: 이 논문에서는 새로운 대규모 보행자 속성 인식 데이터셋인 MSP60K를 제안합니다. 이 데이터셋은 60,122개의 이미지와 57개의 속성 주석을 포함하며, 8가지 시나리오에서 수집되었습니다. 또한, LLM(대규모 언어 모델)으로 강화된 보행자 속성 인식 프레임워크를 제안하여 시각적 특징을 학습하고 속성 분류를 위한 부분 인식을 가능하게 합니다.

- **Technical Details**: MSP60K 데이터셋은 다양한 환경과 시나리오를 반영하여 보행자 이미지를 수집하였으며, 구조적 손상(synthetic degradation) 처리를 통해 현실 세계의 도전적인 상황을 시뮬레이션합니다. 데이터셋에 대해 17개의 대표 보행자 속성 인식 모델을 평가하였고, 랜덤 분할(random split)과 크로스 도메인 분할(cross-domain split) 프로토콜을 사용하여 성능을 검증하였습니다. LLM-PAR 프레임워크는 비전 트랜스포머(Vision Transformer) 기반으로 이미지 특징을 추출하며, 멀티-임베딩 쿼리 트랜스포머(Multi-Embedding Query Transformer)를 통해 속성 분류를 위한 부분 인식 특징을 학습합니다.

- **Performance Highlights**: 제안된 LLM-PAR 모델은 PETA 데이터셋에서 mA 및 F1 메트릭 기준으로 각각 92.20 / 90.02의 새로운 최첨단 성능을 달성하였으며, PA100K 데이터셋에서는 91.09 / 90.41의 성능을 기록했습니다. 이러한 성능 향상은 MSP60K 데이터셋과 기존 PAR 벤치마크 데이터셋에서 폭넓은 실험을 통해 확인되었습니다.



### Photorealistic Object Insertion with Diffusion-Guided Inverse Rendering (https://arxiv.org/abs/2408.09702)
Comments:
          ECCV 2024, Project page: this https URL

- **What's New**: 이번 논문에서는 기존의 모델들이 단일 이미지에서 장면의 조명 효과를 충분히 이해하지 못한다고 지적하며, 이를 극복하기 위해 개인화된 대규모 Diffusion 모델을 물리 기반 역 렌더링 프로세스의 가이드로 사용하는 새로운 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 세 가지 주요 기여를 포함합니다: 1) 물리 기반 렌더러를 사용하여 조명과 3D 자산 간의 상호작용을 정확히 시뮬레이션하여 최종 합성 이미지를 생성합니다. 2) 입력 이미지와 삽입된 객체 유형에 기반한 경량 개인화 체계를 제안합니다. 3) 개인화를 활용하고 훈련 안정성을 개선하는 SDS 손실 변형을 설계합니다.

- **Performance Highlights**: 실험 결과, 제안된 DiPIR 방법이 실내 및 실외 데이터셋에서 기존 최첨단 조명 추정 방법보다 우수한 성능을 보임을 입증하였습니다.



### Harnessing Multimodal Large Language Models for Multimodal Sequential Recommendation (https://arxiv.org/abs/2408.09698)
- **What's New**: 본 논문에서는 Multimodal Large Language Model-enhanced Sequential Multimodal Recommendation (MLLM-MSR) 모델을 제안합니다. 이 모델은 사용자 동적 선호를 반영하기 위해 두 단계의 사용자 선호 요약 방법을 설계하였습니다.

- **Technical Details**: MLLM-MSR은 MLLM 기반의 아이템 요약기를 통해 이미지를 텍스트로 변환하여 이미지 피처를 추출합니다. 이후 LLM 기반의 사용자 요약기를 사용하여 사용자 선호의 동적 변화를 포착합니다. 마지막으로 Supervised Fine-Tuning (SFT) 기법을 통해 추천 시스템의 성능을 향상시킵니다.

- **Performance Highlights**: 다양한 데이터셋에 대한 광범위한 평가를 통해 MLLM-MSR의 효과성을 검증하였으며, 사용자 선호의 변화를 정확하게 반영하고 추천의 개인화 및 정확성을 높이는 우수한 성능을 보여주었습니다.



### LightWeather: Harnessing Absolute Positional Encoding to Efficient and Scalable Global Weather Forecasting (https://arxiv.org/abs/2408.09695)
- **What's New**: 이 논문은 기상 예측에서 Transformer 모델의 절대 위치 인코딩(absolute positional encoding)의 중요성을 강조하며, 이를 통해 효율적이고 경량화된 모델인 LightWeather를 제안합니다.

- **Technical Details**: LightWeather는 MLP(다층 퍼셉트론)를 사용하여 Transformer의 복잡한 구성 요소를 대체하고, 3D 지리적 좌표와 실제 시간 특성을 통합하여 공간-시간 상관관계를 명시적으로 모델링합니다. 이 모델은 30,000개 미만의 파라미터를 가지고 있으며, 한 시간 이내의 훈련시간으로도 우수한 성능을 보입니다.

- **Performance Highlights**: LightWeather는 13개의 다양한 기저 모델에 비해 뛰어난 성능을 나타내며, 글로벌 기상 데이터셋에서 최신 기술의 결과를 달성했습니다. 이 모델은 기상 스테이션의 수가 증가함에 따라 선형적으로 복잡성이 증가하지만, 파라미터 수는 독립적입니다.



### MambaLoc: Efficient Camera Localisation via State Space Mod (https://arxiv.org/abs/2408.09680)
- **What's New**: 본 연구는 Selective State Space (SSM) 모델을 시각적 로컬라이제이션에 혁신적으로 적용하여 새로운 모델인 MambaLoc을 소개합니다. MambaLoc은 효율적인 기능 추출과 빠른 계산을 통해 훈련 효율성을 극대화합니다.

- **Technical Details**: MambaLoc은 Mamba의 강점을 활용하여 Sparse 데이터 환경에서도 강인한 성능을 출력할 수 있으며, Global Information Selector (GIS)를 제안합니다. GIS는 비선택적 Non-local Neural Networks의 전역 특징 추출 능력을 구현하기 위해 SSM을 이용합니다.

- **Performance Highlights**: 7Scenes Dataset에서 MambaLoc은 단 22.8초 만에 0.05%의 훈련 샘플로 최첨단 번역 정확도를 달성했습니다. 다양한 실내 및 실외 데이터셋에 대한 광범위한 실험적 검증을 통해 MambaLoc의 효과성과 GIS의 다양성을 입증했습니다.



### Data-driven Conditional Instrumental Variables for Debiasing Recommender Systems (https://arxiv.org/abs/2408.09651)
- **What's New**: 이 논문에서는 추천 시스템에서의 잠재 변수(latent variables)로 인한 편향을 해결하기 위해 새로운 데이터 기반의 조건부 기기 변수(Conditional Instrumental Variables, CIV) 디바이싱 방법인 CIV4Rec을 제안합니다. 이 방법은 상호작용 데이터에서 유효한 CIV와 그에 해당하는 조건 집합을 자동 생성하여 IV 선택의 복잡성을 줄입니다.

- **Technical Details**: CIV4Rec은 변분 자동 인코더(Variational Autoencoder, VAE)를 활용하여 상호작용 데이터로부터 CIV 및 조건 집합의 표현을 생성합니다. 또한 최소 제곱법(Least Squares)을 적용하여 클릭 예측을 위한 인과적 표현(causal representations)을 도출합니다. 이 방법은 사용자-아이템 쌍의 임베딩을 치료 변수(treatment variable)로 사용하고, 사용자 피드백을 결과(outcome)로 사용하여 편향을 완화합니다.

- **Performance Highlights**: Movielens-10M 및 Douban-Movie 두 개의 실세계 데이터셋에서 진행된 광범위한 실험 결과, CIV4Rec은 유효한 CIV를 성공적으로 식별하고 편향을 효과적으로 줄이며 추천 정확도를 향상시킵니다. 또한 기존의 인과적 디바이싱 방법들과 비교했을 때 최적의 디바이싱 결과와 추천 성능을 달성하였습니다.



### ExpoMamba: Exploiting Frequency SSM Blocks for Efficient and Effective Image Enhancemen (https://arxiv.org/abs/2408.09650)
- **What's New**: ExpoMamba는 기존의 컴퓨터 비전 모델의 한계를 극복하기 위해 디자인된 새로운 아키텍처입니다. 이 모델은 수정된 U-Net 아키텍처 내에서 주파수 상태 공간 구성 요소를 통합하여 효율성과 효과성을 결합합니다.

- **Technical Details**: ExpoMamba 모델은 혼합 노출 문제를 해결하기 위해 2D-Mamba 블록과 주파수 상태 공간 블록(FSSB)을 결합합니다. 이 아키텍처는 특별히 고해상도 이미지의 처리에서 발생하는 계산적 비효율성을 해결하며, 36.6ms의 추론 시간을 달성하여 전통적인 모델보다 2-3배 빠르게 저조도 이미지를 개선합니다.

- **Performance Highlights**: ExpoMamba는 경쟁 모델 대비 PSNR(피크 신호 대 잡음비)을 약 15-20% 개선하며 실시간 이미지 처리 애플리케이션에 매우 적합한 모델로 입증되었습니다.



### Deep Learning-based Machine Condition Diagnosis using Short-time Fourier Transformation Variants (https://arxiv.org/abs/2408.09649)
Comments:
          4 pages, 6 images, submitted to 2024 International Conference on Diagnostics in Electrical Engineering (Diagnostika)

- **What's New**: 본 연구는 전기 모터의 결함 진단을 위한 새로운 접근 방식으로, 전기 전류 신호를 2차원 시간-주파수 플롯으로 변환하여 Convolutional Neural Network (CNN)를 통해 결함을 분류하는 방법을 제안합니다. 이전의 기계 학습 방법보다 높은 정확도를 얻어냈습니다.

- **Technical Details**: 연구에서는 Short-time Fourier Transform (STFT) 및 그 변형들을 사용하여 전기 모터 전류 신호를 2D 플롯으로 변환합니다. 데이터셋은 3,750개 샘플 포인트로, 하나의 정상 상태와 네 가지 합성 결함 상태(베어링 축 불일치, 고정자 간 단락, 파손된 로터 스트립, 외부 베어링 결함)를 포함합니다. 다섯 가지 STFT 변환 방법이 적용되었고, 결과적으로 CNN을 통해 결함 분류를 수행했습니다.

- **Performance Highlights**: 오버랩 STFT 방법은 97.65%의 평균 정확도로 가장 우수한 성능을 보였으며, 나머지 방법들도 모두 95% 이상의 평균 정확도를 기록했습니다. 모든 STFT 기반 접근 방식이 이전의 최고 ML 방법인 LightGBM(93.20%)을 초과했습니다.



### Debiased Contrastive Representation Learning for Mitigating Dual Biases in Recommender Systems (https://arxiv.org/abs/2408.09646)
- **What's New**: 이 논문은 추천 시스템에서 발생하는 인기 편향(popularity bias)과 동조 편향(conformity bias)을 동시에 처리하기 위한 새로운 접근 방식을 제시합니다. 기존의 연구들은 대개 한 가지 편향만을 다루었지만, 본 연구는 두 가지 편향이 동시에 존재한다는 점에 주목하고 이를 해결하기 위한 구조적 모델을 제안합니다.

- **Technical Details**: 제안된 방법은 DCLMDB(Debiased Contrastive Learning framework for Mitigating Dual Biases)라는 새로운 학습 프레임워크를 기반으로 합니다. DCLMDB는 사용자 선택 및 추천 항목이 인기와 동조에 의해 부당하게 영향을 받지 않도록 하기 위해 대조 학습(contrastive learning)을 사용합니다. 또한, 인과 그래프(causal graph)를 활용하여 두 가지 편향의 발생 기전을 모델링하고 이를 통해 사용자와 항목 간의 상호작용을 정제합니다.

- **Performance Highlights**: Movielens-10M과 Netflix의 두 가지 실제 데이터 세트에서 수행된 광범위한 실험을 통해 DCLMDB 모델이 이중 편향을 효과적으로 감소시키고 추천의 정확성과 다양성을 크게 향상시킬 수 있음을 보여줍니다.



### Exploring Wavelet Transformations for Deep Learning-based Machine Condition Diagnosis (https://arxiv.org/abs/2408.09644)
Comments:
          4 pages, 6 figures, submitted to 2024 International Conference on Diagnostics in Electrical Engineering (Diagnostika)

- **What's New**: 본 연구는 딥러닝(DL) 전략을 통해 전기 모터의 결함을 진단하는 새로운 접근법을 제시합니다. 모터의 위상 전류 신호를 분석함으로써 비침습적이고 비용 효율적인 방법을 제공합니다.

- **Technical Details**: Wavelet Transform (WT)을 사용하여 3,750개의 모터 전류 신호 데이터 포인트를 2D 시간-주파수 표현으로 변환하였습니다. 5개의 서로 다른 결함과 5개의 로드 조건(0, 25, 50, 75, 100%)에서 데이터를 수집한 후, 다섯 가지 WT 기반 기법(WT-Amor, WT-Bump, WT-Morse, WSST-Amor, WSST-Bump)을 사용했습니다.

- **Performance Highlights**: WT-Amor, WT-Bump, WT-Morse 모델이 각각 90.93%, 89.20%, 93.73%의 최고 정확도를 기록하여 기존 2D 이미지 기반 방법들(최고 80.25%)보다 뛰어난 성능을 보였습니다. 특히 WT-Morse 기법은 이전 최고 MLP 기술인 93.20%를 초과했습니다.



### How to Make the Most of LLMs' Grammatical Knowledge for Acceptability Judgments (https://arxiv.org/abs/2408.09639)
- **What's New**: 이번 연구는 언어 모델(LLM)의 문법적 지식을 포괄적으로 평가하기 위해 다양한 판단 방법을 조사합니다. 특히, 기존의 확률 기반 기법을 넘어서, in-template LP와 Yes/No 확률 계산과 같은 새로운 방법들이 높은 성능을 보임을 강조합니다.

- **Technical Details**: 연구에서는 세 가지 다른 그룹의 방법을 통해 LLM에서의 수용성 판단을 추출하는 실험을 진행했습니다. 확률 기반 셉방식(세 문장 쌍에 대한 확률을 계산), in-template 방식, 그리고 프롬프팅 기반 방식(Yes/No prob comp)을 비교하였습니다. in-template 방식에서는 문장을 템플릿에 삽입하여 문법성에 집중하도록 지시했습니다.

- **Performance Highlights**: in-template LP 및 Yes/No prob comp가 특히 높은 성능을 보였으며, 기존의 방법들을 초월했습니다. Yes/No prob comp는 길이 편향에 강한 강점을 가지고 있습니다. 통합 사용 시 두 방법은 서로의 보완적인 능력을 보여주며, Mix-P3 방법이 인간보다 1.6% 포인트 높은 정확도를 기록했습니다.



### Meta-Learning on Augmented Gene Expression Profiles for Enhanced Lung Cancer Detection (https://arxiv.org/abs/2408.09635)
Comments:
          Accepted to AMIA 2024 Annual Symposium

- **What's New**: 이번 연구에서는 유전자 발현 프로파일을 기반으로 폐암을 예측하기 위한 메타 학습(meta-learning) 접근 방식을 제안합니다. 이 방법은 표준 심층 학습(deep learning) 방법론에 적용되며, 서로 다른 데이터셋을 활용하여 적은 샘플로도 신속한 데이터 적응이 가능하다는 것입니다.

- **Technical Details**: 연구에서는 Model-Agnostic Meta-Learning (MAML) 전략을 사용하여 심층 신경망을 폐암 탐지에 적용합니다. 이를 통해 제한된 샘플에서 적응 능력을 훈련하여, 고차원 유전자 발현 데이터를 이용한 딥러닝 응용 가능성을 입증하고자 하였습니다.

- **Performance Highlights**: 메타 학습 방법이 증가된 소스 데이터에서 전통적인 방법들과 비교할 때 우수한 성능을 보였으며, 특히 데이터가 제한된 상황에서도 효과적인 결과를 보여주었습니다.



### Say My Name: a Model's Bias Discovery Framework (https://arxiv.org/abs/2408.09570)
- **What's New**: 본 연구에서 새롭게 소개하는 "Say My Name" (SaMyNa) 도구는 심층 학습 모델 내의 편향(bias)을 의미론적으로 식별할 수 있는 첫 번째 도구입니다. 기존의 방법들과 달리, SaMyNa는 모델이 학습한 편향에 초점을 맞추어 설명 가능성(explainability)을 높이고, 학습 중 또는 후속 검증(post-hoc validation) 단계에서 적용될 수 있습니다.

- **Technical Details**: SaMyNa는 텍스트 기반의 파이프라인을 통해 모델이 학습한 특정 편향을 식별하고, 이러한 편향에 대해 자연어 설명을 제공합니다. 이 도구는 특정 작업 관련 정보를 분리할 수 있으며, 텍스트 인코더(text encoder)의 임베딩 공간을 활용하여 저작할 수 있는 간단하면서도 효과적인 기법을 제안합니다.

- **Performance Highlights**: 전통적인 벤치마크에서 SaMyNa의 효과성을 평가한 결과, 모델의 편향을 식별하고 이를 해소하는 능력을 입증하였습니다. 이 도구는 다수의 이미지 분류 작업에서 편향으로 인식된 사례와 그렇지 않은 경우를 구별할 수 있는 능력을 보여주었으며, 이는 DL 모델 진단에 대한 확장 가능성을 강조합니다.



### MergeRepair: An Exploratory Study on Merging Task-Specific Adapters in Code LLMs for Automated Program Repair (https://arxiv.org/abs/2408.09568)
- **What's New**: 본 논문은 Automated Program Repair (APR) 작업에 대한 코드 LLM과 아답터의 지속적 병합(continual merging) 과정을 제안하고, 이를 통해 성능을 평가하는 연구를 다룹니다.

- **Technical Details**: MergeRepair라는 프레임워크를 통해 다양한 아답터 병합 기법을 검토하고, 무게 공간 평균(weight-space averaging), TIES-Merging, DARE와 같은 세 가지 기존 기법을 사용하여 아답터의 병합 효과를 살펴봅니다.

- **Performance Highlights**: 아답터의 지속적 병합을 통해 APR 작업에서 아답터의 성능을 향상시키고, 새로운 실행 시나리오에서 적절한 아답터의 조합을 탐구합니다.



### Grammatical Error Feedback: An Implicit Evaluation Approach (https://arxiv.org/abs/2408.09565)
- **What's New**: 본 연구에서는 Grammatical Error Feedback (GEF)라는 새로운 피드백 방식을 제안합니다. 이는 기존의 Grammatical Error Correction (GEC) 시스템의 수동 피드백 주석 없이도 GEF를 제공할 수 있는 암시적 평가 접근 방식입니다.

- **Technical Details**: 제안된 방법은 'grammatical lineup' 접근 방식을 이용하여 피드백과 에세이 표현을 적절히 짝짓는 작업을 수행합니다. 대규모 언어 모델 (LLM)을 활용하여 피드백과 에세이를 매칭하며, 이 과정에서 선택된 대안의 질인 'foils' 생성이 중요한 요소입니다.

- **Performance Highlights**: 실험 결과, 새로운 GEF 생성 체계는 기존의 GEC 시스템에 비해 더 포괄적이고 해석 가능한 피드백을 제공함으로써 L2 학습자들에게 더 효과적일 수 있음을 보여주었습니다.



### HiAgent: Hierarchical Working Memory Management for Solving Long-Horizon Agent Tasks with Large Language Mod (https://arxiv.org/abs/2408.09559)
Comments:
          Project Page: this https URL

- **What's New**: 이 논문에서는 LLM(대형 언어 모델) 기반 에이전트의 작업 메모리 향상을 위한 새로운 프레임워크 HiAgent를 제안합니다. HiAgent는 서브골(subgoal)을 사용하여 에이전트의 작업 메모리를 관리하며, 이를 통해 실행 가능한 행동을 생성하기 전에 LLM이 서브골을 공식화하도록 유도합니다.

- **Technical Details**: HiAgent는 LLM의 작업 메모리를 계층적으로 관리하는 방법론으로, 사용자 발문의 역사적 행동-관찰 쌍을 직접 입력하는 기존 접근 방식을 개선합니다. 에이전트는 이전 서브골을 요약된 관찰로 대체하기로 결정함으로써 현재 서브골과 관련된 행동-관찰 쌍만을 유지합니다. 이는 메모리의 불필요한 중복을 줄여주며 보다 효율적인 문제 해결이 가능합니다.

- **Performance Highlights**: 실험 결과, HiAgent는 5개의 장기 과제에서 성공률을 두 배로 증가시키고 평균적으로 필요한 단계 수를 3.8 단계 줄였습니다. 또한 HiAgent는 여러 단계에 걸쳐 일관되게 성능 향상을 보이며, 그 견고성과 일반성을 강조합니다.



### Addressing Heterogeneity in Federated Learning: Challenges and Solutions for a Shared Production Environmen (https://arxiv.org/abs/2408.09556)
- **What's New**: 주요 초점은 제조 분야에서 Federated Learning(FL)의 이질성(heterogeneity)을 다루고, 비독립적 및 동일하게 분포되어 있지 않은(Non-IID) 데이터 및 불균형 데이터와 같은 다양한 이질성 유형을 소개합니다.

- **Technical Details**: FL은 여러 분산 장치에서 모델을 학습하는 협업 학습 방법으로, 데이터는 로컬 장치에 남겨두고 모델 업데이트만 중앙 서버에 공유합니다. 이질성 유형은 장치 이질성(device heterogeneity), 모델 이질성(model heterogeneity), 데이터 이질성(data heterogeneity) 등으로 나뉘며, 이를 해결하기 위한 방법으로 개인화된 모델, 견고한 집계 기법(robust aggregation techniques), 클라이언트 선택 기법(client selection techniques)을 검토합니다.

- **Performance Highlights**: FL을 통해 에너지 효율성을 높일 수 있으며, 배포된 학습 환경에서 모델의 강인성을 향상시킬 수 있습니다. 연구 결과는 FL 시스템의 안정성 향상과 공정하고 효율적인 학습을 통해 성과를 보여줍니다.



### Using ChatGPT to Score Essays and Short-Form Constructed Responses (https://arxiv.org/abs/2408.09540)
Comments:
          35 pages, 8 tables, 2 Figures, 27 references

- **What's New**: 이 연구는 ChatGPT의 대규모 언어 모델이 ASAP 대회에서의 인간 및 기계 점수의 정확도를 일치시킬 수 있는지를 평가하였습니다.

- **Technical Details**: 연구는 선형 회귀(linear regression), 랜덤 포레스트(random forest), 그래디언트 부스팅(gradient boost), 부스팅(boost)을 포함한 다양한 예측 모델에 초점을 맞추었습니다. ChatGPT의 성능은 이차 가중 카파(quadratic weighted kappa, QWK) 메트릭스를 사용하여 인간 평가자와 비교 및 평가되었습니다.

- **Performance Highlights**: 결과는 ChatGPT의 그래디언트 부스팅 모델이 일부 데이터 세트에서 인간 평가자에 가까운 QWK를 달성했으나, 전체적인 성능은 불일치하며 종종 인간 점수보다 낮음을 나타냈습니다. 연구는 공정성을 확보하고 편향을 처리하는 데 있어 추가적인 개선이 필요하다고 강조하였습니다. 그럼에도 불구하고, ChatGPT는 특히 도메인 특화된 미세 조정(fine-tuning)을 통해 점수 산정의 효율성을 보여줄 잠재력을 입증하였습니다.



### Revisiting the Graph Reasoning Ability of Large Language Models: Case Studies in Translation, Connectivity and Shortest Path (https://arxiv.org/abs/2408.09529)
- **What's New**: 이 논문은 Large Language Models(LLM)의 그래프(즉, graph) 추론 능력에 집중합니다. 이론적으로 LLM이 그래프 추론 과제를 처리할 수 있는 능력이 입증되었으나 실제 평가에서는 여러 가지 실패가 나타났습니다. 이를 통해 LLM의 한계를 분석하고자 합니다.

- **Technical Details**: 세 가지 기본적인 그래프 과제인 그래프 설명 번역(graph description translation), 그래프 연결(graph connectivity), 그리고 최단 경로 문제(shortest-path problem)에 대해 LLM의 능력을 재조명합니다. 이 연구는 LLM이 그래프 구조를 텍스트 설명을 통해 이해하는 데 실패할 수 있음을 보여줍니다. 또한, 실제 지식 그래프(knowledge graphs)에 대한 조사도 진행하여 이론적 발견과 일치하는 결과를 얻었습니다.

- **Performance Highlights**: LLM은 세 가지 기본 그래프 과제에서 일관된 성능을 보여주지 못했으며, 그 원인은 다양하고 복합적인 것으로 나타났습니다. 연구 결과에 따르면, 그래프 크기(graph size) 외에도 연결 유형(connection types) 및 그래프 설명(graph descriptions)도 중요한 역할을 합니다. 이러한 발견은 LLM이 그래프 추론 과제에서 이론적으로 해결할 수 있는 능력이 있지만, 실제로는 그 한계가 분명함을 나타냅니다.



### A Unified Framework for Interpretable Transformers Using PDEs and Information Theory (https://arxiv.org/abs/2408.09523)
- **What's New**: 본 논문은 Transformer 아키텍처를 이해하기 위한 새로운 통합 이론적 프레임워크를 제시하며, 부분 미분 방정식(Partial Differential Equations, PDEs), 신경 정보 흐름 이론(Neural Information Flow Theory), 정보 병목 이론(Information Bottleneck Theory)을 통합합니다.

- **Technical Details**: Transformer 정보 동역학을 연속적 PDE 과정으로 모델링하여, 확산(diffusion), 자기 주의(self-attention), 비선형 잔여(residual) 요소를 포함합니다. 논문은 각 층의 Transformer 주의 분포와 높은 유사도를 (cosine similarity > 0.98) 보이는 PDE 기반 모델을 검증하는 실험을 수행하였습니다.

- **Performance Highlights**: PDE 모델은 주어진 데이터 세트에서 정보를 전파하는 패턴 및 안정성을 효과적으로 캡처하였으며, 향후 딥 러닝 아키텍처 설계 최적화에 대한 중요한 이론적 통찰을 제공합니다.



### A Logic for Policy Based Resource Exchanges in Multiagent Systems (https://arxiv.org/abs/2408.09516)
- **What's New**: 이번 논문에서는 다중 에이전트 시스템(multiagent systems)에서 에이전트들이 개인 및 집단 목표를 달성하기 위해 상호작용하는 방식에 대해 다룹니다. 특히 자원 교환에 대한 협상 및 합의에 대한 모델링이 주요 이슈로 등장합니다.

- **Technical Details**: 논문에서는 에이전트들이 제공하는 자원 및 교환에 대한 요구를 정의할 수 있는 'exchange environments'라는 포멀(공식적)한 설정을 제안합니다. 또한, 교환 환경을 표현하고 그 동역학(dynamics)을 연구하기 위한 기본 도구로서 결정 가능한(decidable) 선형 논리(linear logic)의 확장을 소개합니다.

- **Performance Highlights**: 이 방법론은 에이전트들이 자원을 적절히 처리하도록 보장하면서 동적 행동을 캡처하는 데 있어 중요한 기초를 제공합니다. 이를 통해 협상 및 자원 교환의 모델링과 합의 형성을 더욱 명확히 할 수 있습니다.



### Out-of-distribution generalization via composition: a lens through induction heads in Transformers (https://arxiv.org/abs/2408.09503)
Comments:
          41 pages, 25 figures

- **What's New**: 이 논문은 대형 언어 모델(LLM)들이 출처가 다른 데이터에 대해 어떻게 일반화할 수 있는지에 대한 연구 결과를 제시합니다. 특히, 숨겨진 규칙에 따라 생성된 샘플을 다루며, 주어진 입력 프롬프트에 대한 숨겨진 규칙을 추론하는 이 과정에서 OOD(Out-Of-Distribution) 일반화의 중요성을 강조합니다.

- **Technical Details**: 연구는 Transformer 아키텍처 내의 induction head 구성 요소에 초점을 맞추어 OOD 일반화와 컴포지션이 어떻게 연결되어 있는지를 규명합니다. Transformer의 훈련 동역학을 조사하고, 특수한 작업을 통해 두 개의 self-attention 레이어를 결합하여 규칙을 배우는 방법을 분석합니다. 자주 사용되는 'common bridge representation hypothesis'를 통해 임베딩(embedding) 공간의 공유 잠재 서브스페이스가 컴포지션을 위한 다리 역할을 한다고 제안합니다.

- **Performance Highlights**: 실험 결과, 2-레이어 Transformer가 OOD 일반화를 달성하는 과정에서 서브스페이스 매칭이 급격히 발생함을 보여주었습니다. 이는 인지 작업의 경우 유사한 컴포지션 구조가 필요하다는 것을 나타내며, LLMs는 모델 파라미터를 업데이트하지 않고도 몇 가지 데모가 있는 경우와 없는 경우 모두 특정 규칙을 학습할 수 있음을 시사합니다.



### Beyond Local Views: Global State Inference with Diffusion Models for Cooperative Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2408.09501)
Comments:
          15 pages, 12 figures

- **What's New**: 본 연구에서는 부분적으로 관찰 가능한 다중 에이전트 시스템에서 에이전트들이 지역 관찰( local observations)만을 기반으로 원래의 전역 상태( global state)를 복원하는 새로운 방법인 State Inference with Diffusion Models (SIDIFF)를 제안합니다.

- **Technical Details**: SIDIFF는 상태 생성기( state generator)와 상태 추출기( state extractor)로 구성되어 에이전트들이 복원된 전역 상태와 지역 관찰을 바탕으로 적절한 행동을 선택할 수 있도록 지원합니다. 이 모델은 diffusion models를 활용하여 지역 정보를 통해 전역 상태를 재구성 하며, Vision Transformer (ViT) 아키텍처를 통해 효과적으로 정보를 추출합니다.

- **Performance Highlights**: SIDIFF는 Multi-Agent Battle City (MABC)와 같은 다양한 실험 플랫폼에서 다른 인기 있는 알고리즘보다 우수한 성과를 기록했습니다. 이는 SIDIFF가 온라인 다중 에이전트 강화 학습( reinforcement learning) 작업에서 전역 상태를 재구성하기 위한 최초의 프레임워크임을 의미합니다.



### Leveraging Invariant Principle for Heterophilic Graph Structure Distribution Shifts (https://arxiv.org/abs/2408.09490)
Comments:
          20 pages, 7 figures

- **What's New**: 이번 논문에서는 Heterophilic Graph Neural Networks(HGNNs)의 한계를 분석하고, 기존의 데이터 분포 차이를 고려하지 않은 노드 수준의 작업에서의 HGSS(Heterophilic Graph Structure distribution Shift) 문제를 해결하기 위한 새로운 프레임워크인 HEI를 제안합니다.

- **Technical Details**: HEI는 heterophily 정보를 활용하여 라틴 환경을 추론하고, 이를 통해 불변 노드 표현을 생성합니다. 이를 통해 HGSS에서의 불변 예측을 가능하게 하며, 기존의 환경 증강 방식 대신 잠재 환경 분할이라는 새로운 접근 방식을 채택합니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋에서 HEI의 성능을 기존 최첨단 방법들과 비교하여 검증하였으며, HEI가 HGSS 문제를 효과적으로 해결할 수 있음을 입증했습니다.



### REFINE-LM: Mitigating Language Model Stereotypes via Reinforcement Learning (https://arxiv.org/abs/2408.09489)
- **What's New**: 이번 논문에서는 기존 언어 모델(LM)에서 발생하는 다양한 편향 문제를 해결하기 위해 REFINE-LM이라는 새로운 디바이싱(debiasing) 방법을 제안합니다. 이 방법은 강화 학습(reinforcement learning)을 사용하여 모델의 성능을 유지하면서도 편향을 제거할 수 있습니다.

- **Technical Details**: REFINE-LM 접근법은 언어 모델의 예측 확률 분포를 기반으로 한 간단한 모델을 학습시켜 편향을 완화합니다. 이 방법은 인적 주석(human annotations)이나 엄청난 컴퓨팅 리소스를 요구하지 않으며, 성별, 민족, 종교 및 국적과 같은 다양한 편향 유형에 일반화될 수 있습니다.

- **Performance Highlights**: 실험 결과, REFINE-LM은 기존의 편향을 에지(-edge)로 하는 언어 모델의 성능을 유지하면서도 고전적인 편향을 의미 있게 줄임을 보여주었습니다. 또한, 이 방법은 다양한 유형의 편향에 적용 가능하며, 학습 비용이 낮고 빠르게 배포될 수 있습니다.



### PanoSent: A Panoptic Sextuple Extraction Benchmark for Multimodal Conversational Aspect-based Sentiment Analysis (https://arxiv.org/abs/2408.09481)
Comments:
          Accepted by ACM MM 2024 (Oral)

- **What's New**: 이 논문에서는 다중모드(multimodal) 대화형(aspect-based) 감정 분석(ABSA)의 새로운 접근을 제안합니다. 논문은 Panoptic Sentiment Sextuple Extraction과 Sentiment Flipping Analysis라는 두 가지 새로운 하위 작업을 소개합니다.

- **Technical Details**: 이 연구에서는 PanoSent라는 대규모(multimodal) 다국어(multilingual) 데이터셋을 구성하였으며, 이는 수동 및 자동으로 주석(annotation)이 달린 데이터로, 다양한 시나리오(scenarios)의 감정 요소를 포괄합니다. 또한, Chain-of-Sentiment reasoning framework와 새로운 다중모달 대형 언어 모델(multi-modal large language model)인 Sentica도 개발하였습니다.

- **Performance Highlights**: 제안된 방법론은 기존의 강력한 기준선(baselines)과 비교하여 탁월한 성능을 보여주었으며, 모든 방법이 효과적임을 검증했습니다.



### MedMAP: Promoting Incomplete Multi-modal Brain Tumor Segmentation with Alignmen (https://arxiv.org/abs/2408.09465)
- **What's New**: 본 논문에서는 의료 영상 세분화에서 결측 모달리티(missing modality) 문제를 해결하기 위한 새로운 패러다임, Medical Modality Alignment Paradigm (MedMAP)을 제안합니다. 이 방법은 사전 훈련된 모델의 부재로 인한 갭을 줄이기 위해 모달리티의 잠재적 특징(latent features)을 공통 분포에 정렬합니다.

- **Technical Details**: MedMAP은 결측 모달리티 접근 방식의 효율성을 높이기 위해 설계되었습니다. 각 모달리티를 미리 정의된 분포 P_{mix}로 정렬함으로써,모달리티 간의 갭을 줄이고 성능을 향상시킵니다. 이 방법은 다양한 백본(backbone)에 대해 실험을 통해 검증되었습니다.

- **Performance Highlights**: BraTS2018 및 BraTS2020 데이터셋에서 MedMAP를 적용한 모델은 기존 방법보다 우수한 성능을 보였습니다. 본 연구는 세분화 작업에서 모달리티 갭을 줄이는 것이 모델의 일반화(generalization)에 긍정적인 영향을 미친다는 것을 발견했습니다.



### In-Memory Learning Automata Architecture using Y-Flash C (https://arxiv.org/abs/2408.09456)
- **What's New**: 이번 논문에서는 메모리 내 데이터 처리(In-Memory Computing)를 개선하기 위해 Y-Flash memristor 장치를 활용하는 새로운 접근 방식을 제안합니다. 고전적인 von Neumann 구조의 병목 현상을 극복하기 위해, 새로운 기계 학습 알고리즘인 Tsetlin Machine (TM)을 사용합니다.

- **Technical Details**: Y-Flash memristor는 기존 CMOS 공정을 통해 제작된 부유 게이트 메모리에 기초한 비휘발성 메모리(NVM) 장치입니다. 이 장치들은 기본적으로 아날로그 저항 조정이 가능하며, 각 Y-Flash 셀에서 최대 40개의 고유한 상태를 생성할 수 있으며, 세심한 조정을 통해 최대 1000개 상태에 도달할 수 있습니다.

- **Performance Highlights**: 제안된 하드웨어 구현체는 Tsetlin Machine의 학습 자동화 기능을 통합하여, 데이터 처리의 확장성 및 에지 학습 능력을 크게 향상시켰습니다. 여러 실험적 검증을 통해 Y-Flash 장치의 신뢰성 및 성능의 우수성이 입증되었습니다.



### GraphSPNs: Sum-Product Networks Benefit From Canonical Orderings (https://arxiv.org/abs/2408.09451)
- **What's New**: 본 논문에서는 기존의 복잡한 확률 분포를 처리할 수 있는 Deep generative 모델의 한계를 극복하기 위한 Graph sum-product networks (GraphSPNs)를 제안합니다.

- **Technical Details**: GraphSPNs는 그래프의 임의 부분에 대해 정확하고 효율적인 추론이 가능하도록 설계된 tractable deep generative 모델입니다. 이 모델은 SPNs의 순열 불변성을 보장하기 위한 다양한 원리를 조사하였습니다.

- **Performance Highlights**: GraphSPNs는 새로운 화학적으로 유효한 분자 그래프를 (조건부적으로) 생성할 수 있으며, 기존의 복잡한 모델들과 비교했을 때 경쟁력 있는 성능을 보입니다.



### Parallel Sampling via Counting (https://arxiv.org/abs/2408.09442)
- **What's New**: 본 논문은 임의의 분포 $
u$에서 샘플링을 가속화하기 위한 병렬화(parallelization) 기법을 소개합니다. 이는 특정 집합 $S 	imes [n]$에 대한 카운팅 쿼리(queries)를 통해 오라클(oracle) 접근을 활용하여 이루어집니다.

- **Technical Details**: 제안된 알고리즘은 $O(n^{2/3} 	imes 	ext{polylog}(n,q))$ 의 병렬 시간(parallel time)을 가지며, 지금까지 알려진 임의 분포에 대한 첫 번째 서브리니어(sublinear) 런타임(runtime)입니다. 알고리즘은 조건부 마진 쿼리(conditional marginal queries) $	ext{P}_{X 	ext{~} 
u}[X_i=	au_i | X_S=	au_S]$ 를 통해 직접 작동하며, 이 역할은 자율 회귀 모델(autoregressive models)의 훈련된 신경망(neural network)에 의해 이루어집니다.

- **Performance Highlights**: 이 결과는 임의 순서 자율 회귀 모델에서 샘플링을 위한 대략 $n^{1/3}$ 배의 속도 향상이 가능하다는 것을 시사합니다. 또한, 카운팅 오라클에 대한 쿼리를 $	ext{poly}(n)$ 회 이하로 하는 모든 병렬 샘플링 알고리즘의 런타임에 대한 하한은 $	ilde{	ext{O}}(n^{1/3})$임을 보여줍니다.



### Towards Boosting LLMs-driven Relevance Modeling with Progressive Retrieved Behavior-augmented Prompting (https://arxiv.org/abs/2408.09439)
- **What's New**: 본 논문은 LLMs(대형 언어 모델)를 활용한 적합성 모델링(relevance modeling)에 사용자 행동 데이터를 통합하는 새로운 접근 방식을 제안합니다. 사용자의 검색 로그를 통한 사용자 상호작용을 기반으로 한 ProRBP(Progressive Retrieved Behavior-augmented Prompting) 프레임워크를 운영하여 적합성 판단을 개선합니다.

- **Technical Details**: ProRBP는 데이터 기반의 사용자 행동 이웃 검색을 통해 신속하게 도메인 전용 지식을 획득하고, 이를 LLM의 출력 개선을 위한 진보적인 프롬프트(prompting) 기법과 결합하여 적합성 모델을 형성합니다. 이 과정은 다양한 측면들을 체계적으로 고려하여 진행됩니다.

- **Performance Highlights**: 실제 산업 데이터를 통한 실험과 온라인 A/B 테스트에서 본 연구가 제안하는 ProRBP 프레임워크가 기존 방법들(Chen et al., 2023) 대비 우수한 성능을 보여 주목을 받고 있습니다.



### Enhancing Modal Fusion by Alignment and Label Matching for Multimodal Emotion Recognition (https://arxiv.org/abs/2408.09438)
Comments:
          The paper has been accepted by INTERSPEECH 2024

- **What's New**: 본 논문에서는 Foal-Net이라는 새로운 다중모달 감정 인식(MER) 프레임워크를 제안합니다. 이 프레임워크는 정렬 후 융합을 통해 모드 간 정보 융합의 효과를 향상시키고, 음성-비디오 감정 정렬(AVEL) 및 교차 모달 감정 레이블 매칭(MEM)이라는 두 개의 보조 태스크를 포함합니다.

- **Technical Details**: Foal-Net은 다중 태스크 학습에 기반하여, 음성과 비디오 모달 간 감정 정보 정렬을 위한 AVEL과, 현재 샘플 쌍의 감정이 동일한지 평가하는 MEM을 통해 기능합니다. AVEL은 대조 학습을 통해 감정 정보를 정렬하고, 이후 모달 융합 네트워크가 정렬된 특징을 통합합니다. MEM은 모드 정보 융합을 도와 모델이 감정 정보에 더 집중하도록 유도합니다.

- **Performance Highlights**: IEMOCAP 코퍼스를 기반으로 한 실험 결과, Foal-Net의 성능은 최고 성능을 기록했습니다. Foal-Net의 비가중 정확도(Unweighted Accuracy)와 가중 정확도(Weighted Accuracy)는 각각 80.1% 및 79.45%로, 다른 최신 방법들(State-of-the-Art)보다 우수합니다.



### HySem: A context length optimized LLM pipeline for unstructured tabular extraction (https://arxiv.org/abs/2408.09434)
Comments:
          9 pages, 4 tables, 3 figures, 1 algorithm

- **What's New**: HySem은 HTML 테이블에서 정확한 의미론적 JSON 표현을 생성하기 위한 새로운 컨텍스트 길이 최적화 기술을 도입합니다. 이 시스템은 소형 및 중형 제약회사를 위해 특별히 설계된 커스텀 파인 튜닝 모델을 활용하며, 데이터 보안과 비용 측면을 고려합니다.

- **Technical Details**: HySem은 세 가지 구성 요소로 구성된 LLM 파이프라인입니다: Context Optimizer Subsystem(𝒜CO), Semantic Synthesizer(𝒜SS), Syntax Validator(𝒜SV). 이 시스템은 컨텍스트 최적화 및 의미론적 변환을 통해 HTML 테이블의 데이터를 처리하고, 유효한 JSON을 출력하는 자동 구문 교정 기능까지 갖추고 있습니다.

- **Performance Highlights**: HySem은 기존 오픈 소스 모델들보다 높은 정확도를 제공하며, OpenAI GPT-4o와 비교할 때도 경쟁력 있는 성능을 보여줍니다. 이 시스템은 대용량 테이블을 처리하는 데 필수적인 컨텍스트 길이 제한을 효과적으로 해결합니다.



### Deformation-aware GAN for Medical Image Synthesis with Substantially Misaligned Pairs (https://arxiv.org/abs/2408.09432)
Comments:
          Accepted by MIDL2024

- **What's New**: 본 논문에서는 Deformation-aware GAN (DA-GAN)이라는 혁신적인 접근법을 제안하여 의료 이미지 합성 과정에서 발생하는 상당한 비정렬 문제를 다루고 있습니다. 이 방법은 다중 목표 역 일관성(multi-objective inverse consistency)을 기반으로 하여 이미지 합성 중 비정렬을 동적으로 수정합니다.

- **Technical Details**: DA-GAN은 생성 과정에서 대칭 등록(symmetric registration)과 이미지 생성을 조화롭게 최적화하며, 적대적 과정에서는 변형 인식 판별기(deformation-aware discriminator)를 설계하여 이미지 충실도를 향상시킵니다. 이 방법은 매칭된 공간 형태와 이미지 충실도를 분리하여 조정합니다.

- **Performance Highlights**: 실험 결과, DA-GAN은 공공 데이터 세트와 호흡 동작 비정렬이 있는 실제 폐 MRI-CT 데이터 세트에서 우수한 성능을 보였습니다. 이는 방사선 치료 계획과 같은 다양한 의료 이미지 합성 작업에서 DA-GAN의 잠재력을 시사합니다.



### FASST: Fast LLM-based Simultaneous Speech Translation (https://arxiv.org/abs/2408.09430)
- **What's New**: 본 논문에서는 동시 음성 번역(Simultaneous Speech Translation, SST) 문제를 해결하기 위해 FASST라는 새로운 접근법을 제안합니다. 이는 기존의 고지연(latency) 문제를 극복하면서 더 나은 번역 품질을 제공합니다.

- **Technical Details**: FASST는 블록별(causal) 음성 인코딩(blockwise-causal speech encoding)과 일관성 마스크(consistency mask)를 도입하여 빠른 음성 입력 인코딩을 가능하게 합니다. 두 단계 훈련 전략을 사용하여 품질-지연 간의 균형을 최적화합니다: 1) 음성 인코더 출력과 LLM 임베딩(Large Language Model embeddings)을 정렬하고, 2) 동시 번역을 위한 미세 조정(finetuning)을 진행합니다.

- **Performance Highlights**: FASST는 MuST-C 데이터셋에서 7B 모델을 사용하여 115M baseline 모델과 비교했을 때 같은 지연에서 영어-스페인어 번역에서 평균적으로 1.5 BLEU 점수 향상을 이루어 내었습니다.



### A Robust Algorithm for Contactless Fingerprint Enhancement and Matching (https://arxiv.org/abs/2408.09426)
- **What's New**: 이번 연구에서는 contactless (비접촉) 지문 이미지의 정확성을 높이기 위해 새로운 알고리즘을 제안하였습니다. 기존의 기술들이 contact (접촉) 지문 이미지에 초점을 맞춘 반면, 본 연구는 contactless 지문 특성에 적합한 방법론을 제공합니다.

- **Technical Details**: 제안된 방법은 두 개의 주요 단계로 나뉘며, 첫 번째 단계는 지문 이미지를 향상시키고 minutiae (미세구조) 기능을 추출 및 인코딩하는 오프라인 단계입니다. 두 번째 단계는 새로운 템플릿 이미지를 센서에서 수집하고, 이를 향상시키고 minutiae 기능을 추출하며 인코딩하는 온라인 과정입니다. Gabor 필터 기반의 컨텍스처 필터링을 통해 지역적 ridge (능선) 구조를 향상시키고, 이후 이진화 및 얇게 만들기를 통해 ridge/valley (능선/골짜기) 스켈레톤을 생성합니다.

- **Performance Highlights**: Proposed method는 PolyU contactless fingerprint dataset에서 2.84%의 최저 Equal Error Rate (EER)를 달성하였으며, 이는 기존의 최첨단 기술들과 비교했을 때 우수한 성능을 보여줍니다. 이러한 높은 정밀도와 회복력을 통해 contactless fingerprint 기반의 식별 시스템 구현에 효과적이고 실행 가능한 해결책을 입증합니다.



### Distinguish Confusion in Legal Judgment Prediction via Revised Relation Knowledg (https://arxiv.org/abs/2408.09422)
Comments:
          Accepted by ACM TOIS

- **What's New**: 이번 연구에서는 법적 판별 예측(LJP) 문제를 해결하기 위해 새로운 통합 모델인 D-LADAN을 제안합니다. D-LADAN은 유사한 법 조항들이 서로 혼동되는 문제를 개선하기 위해 Graph Distillation Operator(GDO)를 적용하고, 법 조항 간의 후행 유사성을 감지하기 위한 동적 메모리 메커니즘을 도입하였습니다.

- **Technical Details**: D-LADAN은 법 조항 간의 그래프를 구성하고, Graph Distillation Operator(GDO)를 통해 고유한 특징을 추출합니다. 또한, 가중치가 있는 GDO를 사용하여 데이터 불균형 문제로 인한 귀납 편향을 수정하는데 초점을 맞추고 있습니다. 이 모델은 큰 규모의 실험을 통해 기존의 최첨단 모델들에 비해 높은 정확도와 견고성을 보여줍니다.

- **Performance Highlights**: D-LADAN은 기존 방법들에 비해 유의미한 성능 향상을 보여줍니다. 특히 데이터 불균형 문제로 인해 발생하는 혼동을 효과적으로 해결하여, 법적 판단의 정확성을 크게 개선시키는데 기여하고 있습니다.



### Challenges and Responses in the Practice of Large Language Models (https://arxiv.org/abs/2408.09416)
- **What's New**: 이 논문은 현재 주목받고 있는 AI 분야에서의 산업 동향, 학술 연구, 기술 혁신 및 비즈니스 애플리케이션 등 여러 차원에서의 질문들을 체계적으로 요약합니다.

- **Technical Details**: 강력한 컴퓨팅 리소스 통합을 위해 클라우드-엣지-단말 협업 아키텍처(cloud-edge-end collaborative architecture)를 구현하고, 데이터 수집, 엣지 처리, 클라우드 컴퓨팅, 협업 작업의 흐름을 설명합니다. 또한, 대형 언어 모델(LLM)의 필요성과 모델 훈련 시의 주요 도전 과제를 다룹니다.

- **Performance Highlights**: 2023년 중국의 가속 칩 시장이 140만 규모로 급격히 성장하였고, GPU 카드가 85%의 시장 점유율을 차지했습니다. LLM과의 관계에서는 사용자 맞춤형 개발 및 비즈니스 효율 개선이 두드러지며, 국내 AI 칩 브랜드가 미래에 더 큰 돌파구를 기대할 수 있다고 언급합니다.



### Comparison between the Structures of Word Co-occurrence and Word Similarity Networks for Ill-formed and Well-formed Texts in Taiwan Mandarin (https://arxiv.org/abs/2408.09404)
Comments:
          4 pages, 1 figure, 5 tables

- **What's New**: 이 연구는 대만 만다린의 비정형 텍스트로부터 생성된 단어 공기기 네트워크(Word Co-occurrence Network, WCN)와 단어 유사성 네트워크(Word Similarity Network, WSN)의 구조를 비교하고 분석하였습니다.

- **Technical Details**: 연구는 PTT의 139,578개 게시물과 2004년 및 2008년 대만 사법부의 53,272개 판결문을 데이터로 사용하여, 텍스트를 전처리 후 skip-gram word2vec 모델을 통해 단어 임베딩을 생성하였습니다. WCN 및 WSN는 각각 비가중치 및 무방향 네트워크로 구성되었습니다.

- **Performance Highlights**: 모든 네트워크는 스케일-프리(scale-free) 특성을 가지며, 소규모 세계(small-world) 성질을 지니고 있음이 확인되었습니다. WCN과 WSN은 일반적으로 비-assortative를 보였으며, 각 네트워크의 DAC 값은 다소 차이를 보였습니다.



### Federated Graph Learning with Structure Proxy Alignmen (https://arxiv.org/abs/2408.09393)
Comments:
          Accepted by KDD 2024

- **What's New**: 이번 논문에서는 다수의 데이터 소유자가 분산된 그래프 데이터에 대한 그래프 학습 모델을 학습할 수 있도록 하는 Federated Graph Learning (FGL) 프레임워크인 FedSpray를 제안합니다. 이 프레임워크는 글로벌 구조 프록시를 통해 노드 분류의 성능을 향상시키기 위해서 설계되었습니다.

- **Technical Details**: FedSpray는 로컬 클래스별 구조 프록시를 학습하여 서버에서 글로벌 구조 프록시와 정렬함으로써 신뢰할 수 있고 편향되지 않은 이웃 정보를 확보합니다. 이를 통해 GNN 모델의 개인화된 로컬 훈련을 정규화하는 데 사용됩니다. 특히, FedSpray는 구조 프록시와 결합된 글로벌 기능-구조 인코더를 훈련하여 노드 특징에만 의존하는 신뢰할 수 있는 소프트 타겟을 생성합니다.

- **Performance Highlights**: 다양한 실험을 통해 FedSpray가 다른 기준 모델들보다 우수한 성능을 보임을 입증하였습니다. 실험 결과는 FedSpray의 효과성을 강조하며, 특정 데이터 세트를 활용한 평가 결과가 포함되어 있습니다.



### Offline RLHF Methods Need More Accurate Supervision Signals (https://arxiv.org/abs/2408.09385)
Comments:
          under review

- **What's New**: 이 논문에서는 기존의 오프라인 강화 학습 방법인 Reinforcement Learning with Human Feedback (RLHF)의 한계점을 극복하기 위해 보상 차이 최적화(Reward Difference Optimization, RDO)라는 새로운 방법을 제안합니다. 이 방법은 신뢰할 수 있는 선호 데이터를 기반으로 샘플 쌍의 보상 차이를 측정하여 LLM을 최적화하는 접근법입니다.

- **Technical Details**: RDO는 샘플 쌍의 보상 차이 계수를 도입하여 각 응답 쌍의 상호작용을 분석하고 개인의 선호를 반영합니다. 연구팀은 Attention 기반의 Difference Model을 개발하여 두 응답 간의 보상 차이를 예측하며, 이를 통해 오프라인 RLHF 방법을 개선합니다. 이 방법은 최대 마진 순위 손실(max-margin ranking loss) 및 부정 로그 우도 손실(negative log-likelihood loss)과 같은 기존 방법에 적용될 수 있습니다.

- **Performance Highlights**: 7B LLM을 사용하여 HH 및 TL;DR 데이터세트에서 진행된 실험 결과, RDO는 자동 지표 및 인간 평가 모두에서 효과적인 성능을 발휘하며, LLM과 인간의 의도 및 가치를 잘 맞출 수 있는 잠재력을 보여줍니다.



### VRCopilot: Authoring 3D Layouts with Generative AI Models in VR (https://arxiv.org/abs/2408.09382)
Comments:
          UIST 2024

- **What's New**: VRCopilot은 가상 현실(VR)에서의 몰입형 저작을 지원하는 혼합 주도 시스템으로, 사전 훈련된 생성 AI 모델을 통합하였습니다. 이를 통해 사용자와 AI 간의 협력적 창작을 촉진할 수 있습니다.

- **Technical Details**: VRCopilot은 다중 모드 상호 작용 및 중간 표현(예: 와이어프레임) 개념을 도입하여 사용자가 생성되는 콘텐츠를 더 잘 제어할 수 있도록 돕습니다. 사용자들은 음성과 제스처를 통해 객체 생성 요구를 명확히 할 수 있습니다.

- **Performance Highlights**: 사용자 연구를 통해 발견된 바에 따르면, 와이어프레임을 사용한 scaffolded creation 방식이 자동 생성에 비해 사용자 에이전시를 향상시키며, 수동 생성(manual creation) 방식이 가장 높은 창의성과 에이전시를 제공합니다.



### Detecting the Undetectable: Combining Kolmogorov-Arnold Networks and MLP for AI-Generated Image Detection (https://arxiv.org/abs/2408.09371)
Comments:
          8 Pages, IEEE Transactions

- **What's New**: 이번 논문은 최신 생성 AI 모델에서 생성된 이미지를 효과적으로 식별하기 위한 새로운 감지 프레임워크를 제시합니다. 특히 DALL-E 3, MidJourney, Stable Diffusion 3의 이미지를 포함한 포괄적인 데이터셋을 통해 다양한 조건에서 실제 이미지와 AI 생성 이미지를 구분하는 데 중점을 두고 있습니다.

- **Technical Details**: 이 연구에서는 전통적인 Multilayer Perceptron (MLP)와 의미론적 이미지 임베딩을 결합한 분류 시스템을 발전시켰습니다. 더불어 Kolmogorov-Arnold Networks (KAN)와 MLP를 결합한 하이브리드 아키텍처를 제안하여 AI 생성 이미지의 복잡한 패턴을 분석할 수 있습니다.

- **Performance Highlights**: 제안한 하이브리드 KAN MLP 모델은 세 가지 out-of-distribution (OOD) 데이터셋에서 표준 MLP 모델을 지속적으로 초과하는 성능을 보였으며, F1 점수는 각각 0.94, 0.94, 0.91로 나타났습니다. 이는 AI 생성 이미지와 실제 이미지를 구분하는 데 있어 우수한 성능과 내구성을 입증하는 결과입니다.



### Panorama Tomosynthesis from Head CBCT with Simulated Projection Geometry (https://arxiv.org/abs/2408.09358)
Comments:
          12 pages, 6 figures, 1 table, Journal submission planned

- **What's New**: 이 논문에서는 Cone Beam Computed Tomography (CBCT) 데이터를 이용하여 효과적으로 Panoramic X-ray 이미지를 합성하는 새로운 방법을 제안합니다. 기존의 방법들과는 달리, 이 방법은 환자의 치아가 없거나 금속 임플란트가 있는 경우에도 높은 품질의 이미지를 생성할 수 있습니다.

- **Technical Details**: 제안된 방법은 변동하는 회전 중심과 반타원형 경로를 따르는 시뮬레이션된 프로젝션 기하학을 정의합니다. 첫 번째 단계에서는 환자의 머리 기울기를 수정하여 턱 위치를 감지하며, 그 후에 상악과 하악을 포함하는 초점 영역을 정의하여 최종 Panoramic 프로젝션에 기여합니다. 그런 다음 여러 X-ray 프로젝션을 생성하여 최종 X-ray 이미지를 만듭니다.

- **Performance Highlights**: 연구 결과, 제안된 방법은 종래의 방법보다 더 나은 품질의 Panoramic X-ray 이미지를 생성하며, 환자의 턱 형태나 위치에 관계없이 효과적으로 적용할 수 있음이 입증되었습니다.



### Meta-Learning Empowered Meta-Face: Personalized Speaking Style Adaptation for Audio-Driven 3D Talking Face Animation (https://arxiv.org/abs/2408.09357)
- **What's New**: 이 논문에서는 다양한 말하기 스타일에 대한 적응력을 확보하기 위해 MetaFace라는 새로운 방법론을 소개합니다. 기존의 방법들이 특정 인물과 고정된 말하기 스타일에 맞춰져 있었던 것에 비해, MetaFace는 더 유연한 적응을 가능하게 합니다.

- **Technical Details**: MetaFace는 메타-학습(meta-learning) 개념에 기반하여 설계되었습니다. 이 방법론은 Robust Meta Initialization Stage (RMIS)를 통해 기본적인 말하기 스타일 적응을 지원하고, Dynamic Relation Mining Neural Process (DRMN)를 활용해 관찰된 말하기 스타일과 관찰되지 않은 스타일 간의 관계를 형성합니다. 또한 Low-rank Matrix Memory Reduction Approach를 도입하여 모델 최적화와 학습 스타일 세부 사항을 효율적으로 개선합니다.

- **Performance Highlights**: MetaFace는 기존의 다양한 기초 모델들과 비교하여 훨씬 뛰어난 성능을 보여줍니다. 실험 결과를 통해, 이 방법이 새로운 최첨단 성과를 달성했음을 입증합니다.



### E-CGL: An Efficient Continual Graph Learner (https://arxiv.org/abs/2408.09350)
- **What's New**: 이 논문에서는 Efficient Continual Graph Learner (E-CGL)라는 새로운 방법을 제안합니다. E-CGL은 지속적인 그래프 학습에서 발생하는 두 가지 주요 문제, 즉 그래프 데이터 간의 상호 의존성(interdependencies)과 대형 그래프 처리의 효율성(concerns)이란 문제를 해결하기 위해 개발되었습니다.

- **Technical Details**: E-CGL은 중요 샘플링 전략(importance sampling strategy)과 다양성 샘플링 전략(diversity sampling strategy)을 결합하여 사용합니다. 이는 각각 노드의 중요성과 선호도를 기반으로 새로운 패턴을 모델링합니다. E-CGL은 MLP (Multi-Layer Perceptron) 모델을 사용하여 GCN (Graph Convolutional Network)와 가중치를 공유하여 훈련의 효율성을 높입니다. 이로 인해 계산 집약적인 메시지 전달 과정이 피할 수 있습니다.

- **Performance Highlights**: E-CGL은 4개의 그래프 지속 학습 데이터 세트에서 9개의 기준 모델(base models)에 비해 우수한 성능을 발휘하며, 평균 15.83배의 훈련 시간 가속화를 달성하고 4.89배의 추론 시간 가속화를 나타냈습니다. 또한, E-CGL은 평균적으로 카타스트로픽 포겟팅 문제를 1.1%로 줄였습니다.



### Characterizing and Evaluating the Reliability of LLMs against Jailbreak Attacks (https://arxiv.org/abs/2408.09326)
- **What's New**: 본 연구에서는 Jailbreak 공격에 대한 대규모 실험을 바탕으로 LLM(대형 언어 모델)의 신뢰성을 평가하기 위한 포괄적인 평가 프레임워크를 제시합니다. 또한, 체계적으로 10가지 최신 jailbreak 전략과 1525개의 위험 질문을 분석하여 다양한 LLM의 취약점을 살펴봅니다.

- **Technical Details**: 이 연구는 Attack Success Rate (ASR), Toxicity Score, Fluency, Token Length, 그리고 Grammatical Errors 등 여러 차원에서 LLM의 출력을 평가합니다. 이를 통해 특정 공격 전략에 대한 모델의 저항력을 점검하고, 종합적인 신뢰성 점수를 생성하여 LLM의 취약점을 줄이기 위한 전략적 제안을 합니다.

- **Performance Highlights**: 실험 결과 모든 tested LLMs가 특정 전략에 대해 저항력이 부족함을 보여주며, LLM의 신뢰성을 중심으로 향후 연구의 방향성을 제시합니다. 이들 모델 중 일부는 유의미한 저항력을 보여준 반면, 다른 모델은 훈련 중 부여된 윤리적 및 내용 지침에 대한 일치성이 부족한 것으로 나타났습니다.



### Learning Fair Invariant Representations under Covariate and Correlation Shifts Simultaneously (https://arxiv.org/abs/2408.09312)
Comments:
          CIKM 2024

- **What's New**: 본 논문에서는 공정성을 고려한 도메인 불변 예측기를 학습하는 새로운 접근 방식을 제안합니다. 이 접근 방식은 공변량 변화(covariate shift)와 상관 관계 변화(correlation shift)를 동시에 처리하여, 훈련 중 접근할 수 없는 미지의 테스트 도메인에 대한 일반화를 보장합니다.

- **Technical Details**: 제안된 방법은 데이터의 내용(content)과 스타일(style) 요소를 잠재 공간(latent spaces)에서 분리하여 학습합니다. 이 과정에서 민감한 정보는 최소화하고 비민감한 정보는 최대한 유지하며, 공정성을 고려한 도메인 불변 콘텐츠 표현을 학습합니다. 우리의 프레임워크는 콘텐츠 특징 추출기(content featurizer), 공정한 표현 학습기(fair representation learner), 도메인 불변 분류기(invariant classifier)로 구성됩니다.

- **Performance Highlights**: 광범위한 실험 결과는 제안된 FLAIR 알고리즘이 모델 정확성 뿐만 아니라 그룹 및 개인 공정성 측면에서도 최첨단 방법들보다 우수하다는 것을 보여줍니다. 세 가지 벤치마크 데이터셋에 대한 실험에서 효과적인 일반화를 달성했습니다.



### A Benchmark Time Series Dataset for Semiconductor Fabrication Manufacturing Constructed using Component-based Discrete-Event Simulation Models (https://arxiv.org/abs/2408.09307)
- **What's New**: 본 연구에서는 Intel 반도체 제조 공장의 기준 모델을 기반으로 한 데이터셋을 개발하고 구성하여, 기계 학습(ML) 모델 생성을 위한 토대를 마련합니다.

- **Technical Details**: 이 모델은 Parallel Discrete-Event System Specification (PDEVS)을 사용하여 형식화되었으며, DEVS-Suite 시뮬레이터를 통해 실행됩니다. 수집된 데이터셋은 명확한 구조와 정확한 행동을 지닌 공장 모델에 기반하므로 기계 학습 모델 개발에 적합합니다.

- **Performance Highlights**: 기계 학습 모델 실행이 물리 기반 모델에 비해 고효율적이며, 이 데이터셋은 ML 커뮤니티에서 공식화되고 확장 가능한 구성 요소 기반의 이산 사건 모델과 시뮬레이션에 기반한 행동 분석에도 활용될 수 있습니다.



### Evaluating Usability and Engagement of Large Language Models in Virtual Reality for Traditional Scottish Curling (https://arxiv.org/abs/2408.09285)
- **What's New**: 이 논문은 가상 현실(VR) 환경에서 대규모 언어 모델(LLMs)의 혁신적인 응용을 탐구하며, 스코틀랜드 전통 컬링의 교육을 증진하는 데 중점을 두고 있습니다. 연구 결과, LLM 기반 챗봇이 기존의 스크립트 기반 챗봇보다 상호작용성과 참여도를 크게 향상시켜 더 동적이고 몰입감 있는 학습 환경을 조성하는 데 효과적이라는 것을 보여줍니다.

- **Technical Details**: 이 연구에서는 'Scottish Bonspiel VR' 게임에서 LLM 기반 챗봇과 사전 정의된 스크립트 챗봇의 성능을 비교하였습니다. 연구 방법은 사용자 경험을 평가하기 위한 Between-subjects 디자인을 통해 진행되었으며, 36명의 참가자들이 참여했습니다. 연구 질문으로는 LLM 챗봇이 기존 챗봇에 비해 사용성과 참여도 및 문화유산 학습 결과를 어떻게 개선하는지를 분석하였습니다.

- **Performance Highlights**: LLM 기반 챗봇은 사용자가 스코틀랜드 전통 컬링의 문화를 더 깊이 이해할 수 있도록 돕는 동시에, 전통적인 말하기 방식에서 벗어나 디지털화된 경험을 통해 문화유산을 더 잘 전달하고 보존할 수 있도록 하는 데 기여합니다. 이러한 통합은 사용자들이 인지적 부담을 줄이고 더 나은 학습 결과를 가져오는 데 긍정적인 영향을 미쳤습니다.



### PREMAP: A Unifying PREiMage APproximation Framework for Neural Networks (https://arxiv.org/abs/2408.09262)
Comments:
          arXiv admin note: text overlap with arXiv:2305.03686

- **What's New**: 이 논문은 뉴럴 네트워크의 출력 속성을 만족하는 입력 집합인 프리이미지(preimage)의 추상화를 위한 일반적인 프레임워크를 제시합니다. 이 프레임워크는 모든 polyhedral output set에 대해 과소 근사와 과대 근사를 생성합니다. 이를 위해 저렴한 매개변수화된 선형 이완(parameterized linear relaxation)이 사용되며, 입력 특징과 뉴런을 분할하여 입력 공간을 반복적으로 파티션하는 anytime refinement 절차가 포함되어 있습니다.

- **Technical Details**: 주요 방법론은 'disjoint unions of polytopes (DUP)'로 표현되는 과소 및 과대 근사를 계산하는 것입니다. 뉴럴 네트워크의 파라미터화된 선형 이완과 branch-and-bound refinement 전략이 결합되어 효율적인 분석이 가능해집니다. 이 방법은 최적화 목표를 설정하여 근사의 부피(volume)를 최소화하는 데 초점을 맞춥니다.

- **Performance Highlights**: 이 방법은 높은 입력 차원의 이미지 분류 작업에서 주목할 만한 효율성과 확장성을 보여줍니다. 여러 데이터셋에 대한 경험적 평가에서 기존 최첨단 기법에 비해 성능이 크게 향상된 것으로 나타났습니다. 정량적 검증(quantitative verification)과 견고성 분석(robustness analysis) 문제에도 적용 가능성이 입증되었습니다.



### V2X-VLM: End-to-End V2X Cooperative Autonomous Driving Through Large Vision-Language Models (https://arxiv.org/abs/2408.09251)
- **What's New**: 이 논문에서는 대규모 비전-언어 모델(large vision-language models, VLM)을 활용한 최초의 E2E 차량-인프라 협동 자율 주행 프레임워크인 V2X-VLM을 도입합니다. V2X-VLM은 차량 장착 카메라, 인프라 센서 및 텍스트 정보를 통합하여 상황 인식, 의사 결정 및 최적 궤적 계획을 개선하도록 설계되었습니다.

- **Technical Details**: V2X-VLM 프레임워크는 다양한 출처의 다중 모드(multi-modal) 데이터를 통합하여 자율 주행 시스템이 복잡한 동적 환경을 정확하게 이해하고 주행할 수 있도록 합니다. 이 구조는 V2X 통신을 통해 차량 및 인프라 쪽의 복잡한 시각적 장면을 쌍으로 만들고, 이를 효과적으로 처리하는 통합 패러다임을 제공합니다.

- **Performance Highlights**: DAIR-V2X 데이터셋을 통한 검증 결과, V2X-VLM은 기존의 협력 자율 주행 방법들에 비해 성능이 개선되었습니다. 이 연구는 자율 주행 분야에서 VLM의 잠재력을 입증하고 있으며, 안전하고 효율적인 자율 주행 운영을 위한 유망한 해결책으로 자리 잡을 것으로 기대됩니다.



### Towards Effective Top-N Hamming Search via Bipartite Graph Contrastive Hashing (https://arxiv.org/abs/2408.09239)
- **What's New**: 새롭게 제안된 Bipartite Graph Contrastive Hashing (BGCH+) 모델은 그래프 구조에서의 학습 효율성을 극대화하고 정보 손실을 최소화하기 위해 자기 감독 학습(self-supervised learning) 기법을 도입합니다.

- **Technical Details**: BGCH+는 이중 강화(feature augmentation) 접근 방식을 사용하여 중간 정보 및 해시 코드 출력을 보강하며, 이는 잠재적 피처 공간(latent feature spaces) 내에서 더욱 표현력 강하고 견고한 해시 코드를 생성합니다.

- **Performance Highlights**: BGCH+는 6개의 실제 벤치마크에서 기존 방식에 비해 성능을 향상시킨 것으로 검증되었으며, Hamming 공간에서의 Top-N 검색에서 기존의 풀 정밀도(full-precision) 모델과 유사한 예측 정확도를 유지하면서 계산 속도를 8배 이상 가속화할 수 있음을 보여줍니다.



### Hybrid Semantic Search: Unveiling User Intent Beyond Keywords (https://arxiv.org/abs/2408.09236)
- **What's New**: 이 논문은 전통적인 키워드 기반 검색의 한계를 극복하고 사용자의 의도를 이해하기 위한 새로운 하이브리드 검색 접근 방식을 소개합니다.

- **Technical Details**: 제안된 시스템은 키워드 매칭, 의미적 벡터 임베딩(semantic vector embeddings), 그리고 LLM(대규모 언어 모델) 기반의 구조화된 쿼리를 통합하여 고도로 관련성 높고 맥락에 적절한 검색 결과를 제공합니다.

- **Performance Highlights**: 이 하이브리드 검색 모델은 포괄적이고 정확한 검색 결과를 생성하는 데 효과적이며, 쿼리 실행 최적화를 통해 응답 시간을 단축할 수 있는 기술을 탐구합니다.



### Reference-Guided Verdict: LLMs-as-Judges in Automatic Evaluation of Free-Form Tex (https://arxiv.org/abs/2408.09235)
- **What's New**: 이 연구는 여러 LLM을 활용하여 오픈 엔디드(Large Language Models) 텍스트 생성을 평가하는 새로운 레퍼런스 기반 판별 방법을 제안합니다. 이는 평가의 신뢰성과 정확성을 높이고 인간 평가와의 일치도를 크게 향상시킵니다.

- **Technical Details**: 제안된 방법은 입력, 후보 모델의 응답 및 레퍼런스 답변을 포함하여 다수의 LLM을 판단자로 활용합니다. 평가 과정은 세 가지 주요 구성 요소로 이루어져 있으며, LLM이 생성한 출력을 검토하는 과정을 포함합니다. 이를 통해 다수의 LLM 판단의 집합적 판단이 인간 판단자와 유사한 신뢰성을 생성할 수 있는지를 검증합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 기본적으로 사람의 평가와 강한 상관관계를 보였으며, 특히 복잡한 작업에서 다수의 LLM 판단을 활용할 때 평가의 신뢰성과 정확성이 크게 향상됩니다.



### Flatten: Video Action Recognition is an Image Classification task (https://arxiv.org/abs/2408.09220)
Comments:
          13pages, 6figures

- **What's New**: 본 논문에서는 Flatten이라는 새로운 비디오 표현 아키텍처를 소개합니다. Flatten은 비디오 행동 인식 작업을 이미지 분류 작업으로 변환하여 비디오 이해의 복잡성을 감소시키는 모듈입니다. 이 아키텍처는 기존의 이미지 이해 네트워크에 손쉽게 통합될 수 있습니다.

- **Technical Details**: Flatten은 3D 시공간(spatiotemporal) 데이터를 평면화하는(row-major transform) 특정 작업을 통해 2D 공간정보로 변환합니다. 이후 일반적인 이미지 이해 모델을 사용하여 시간적 동적 및 공간적 의미 정보를 포착합니다. 이 방식으로 비디오 행동 인식을 효율적으로 수행할 수 있습니다.

- **Performance Highlights**: Kinetics-400, Something-Something v2, HMDB-51과 같은 일반적인 데이터 세트에서 대규모 실험을 통해 Flatten을 적용하면 기존 모델 대비 성능 향상이 크게 나타났습니다. 특히, ‘Uniformer(2D)-S + Flatten’ 조합은 Kinetics400 데이터셋에서 81.1이라는 최고의 인식 성능을 기록했습니다.



### On the Improvement of Generalization and Stability of Forward-Only Learning via Neural Polarization (https://arxiv.org/abs/2408.09210)
Comments:
          To be published in ECAI 2024

- **What's New**: 최근 대안으로 주목받고 있는 Forward-only learning 알고리즘인 Polar-FFA는 기존의 Forward-Forward Algorithm (FFA)을 개선한 새로운 구현입니다. Polar-FFA는 양성과 음성 샘플 간의 신경 분할을 도입하여, 각 층에서 대칭적인 그래디언트 행동을 생성하는 데 초점을 맞추고 있습니다.

- **Technical Details**: Polar-FFA는 각각의 신경이 양성 및 음성 폴라리제이션에 따라 학습을 수행하도록 설계되었습니다. 이는 각각의 신경이 자신의 데이터 유형에 대해 goodness score를 최대화하고 반대 데이터 유형에 대해서는 최소화하는 것을 목표로 합니다. 이를 통해 학습 과정에서의 그래디언트 불균형 문제를 완화합니다.

- **Performance Highlights**: Polar-FFA는 이미지 분류 데이터셋에서 여러 활성화 함수와 goodness 함수를 사용하여 실험을 수행한 결과, 기존의 FFA에 비해 정확도와 수렴 속도 면에서 우수한 성능을 보였으며, 하이퍼파라미터에 대한 의존도가 낮아 더 넓은 범위의 신경망 구성을 가능하게 합니다.



### Architectural Foundations and Strategic Considerations for the Large Language Model Infrastructures (https://arxiv.org/abs/2408.09205)
- **What's New**: 대규모 언어 모델(LLM) 인프라 구축의 중요성을 강조하며, 소프트웨어 및 데이터 관리에 대한 자세한 분석을 통해 성공적인 LLM 개발을 위한 필수 고려사항과 보호 조치를 제시합니다.

- **Technical Details**: LLM 훈련을 위한 인프라 구성에서 H100/H800 GPU가 필수적이며, 7B 모델을 하루 안에 훈련할 수 있는 8개의 노드를 갖춘 서버 클러스터의 효율성을 강조합니다. LoRA(저순위 적응)와 같은 가벼운 조정 방법론은 GPU 사용의 접근성을 높이고, 고성능 GPU의 전략적 선택이 중요한 요소로 작용한다고 설명합니다.

- **Performance Highlights**: LLM의 성공적인 배포를 위해 컴퓨팅 파워, 비용 효율성, 소프트웨어 최적화 전략, 하드웨어 선택을 포괄적으로 고려해야 하며, 이는 AI 애플리케이션의 보편적 채택을 촉진하고 다양한 Domain에서의 지원을 강화하는 데 기여합니다.



### SA-GDA: Spectral Augmentation for Graph Domain Adaptation (https://arxiv.org/abs/2408.09189)
- **What's New**: 이 논문에서는 Graph Neural Networks (GNNs)의 라벨 부족 문제와 도메인 간 전이 학습의 한계를 해결하기 위해 새로운 도메인 적응 방법인 	extit{Spectral Augmentation for Graph Domain Adaptation (SA-GDA)}를 제안합니다. 이 방법은 서로 다른 도메인에서 동일한 레이블을 가진 노드 간의 스펙트럼 도메인에서의 특성을 활용하여 각 카테고리에 대한 피쳐 공간을 정렬합니다.

- **Technical Details**: SA-GDA는 첫 번째로, 서로 다른 도메인에 있는 동일한 카테고리 노드의 스펙트럼적 특성이 유사하다는 관찰을 기반으로 합니다. 따라서, 전체 피쳐 공간을 정렬하는 대신, 카테고리 피쳐 공간을 스펙트럼 도메인에서 정렬하는 전략을 세웁니다. 또한, 듀얼 그래프 합성곱 신경망을 개발하여 지역적(local) 및 전역적(global) 일관성을 활용하여 피쳐 집계(feature aggregation)를 수행하며, 도메인 분류기를 통해 서로 다른 도메인 간의 지식 전이를 촉진합니다.

- **Performance Highlights**: 다양한 공개 데이터셋에서 수행한 실험 결과, SA-GDA는 현재 최첨단 경쟁 모델들보다 우수한 성능을 보였습니다. 이로 인해 SA-GDA는 그래프 도메인 적응 설정에서 카테고리 정렬된 피쳐의 사용을 통해 노드 분류의 정확성을 크게 향상시킴을 입증하였습니다.



### EEG-SCMM: Soft Contrastive Masked Modeling for Cross-Corpus EEG-Based Emotion Recognition (https://arxiv.org/abs/2408.09186)
Comments:
          16 pages, 8 figures, 15 tables, submitted to AAAI 2025

- **What's New**: 본 논문에서는 EEG (electroencephalography) 기반의 감정 인식을 위한 새로운 Soft Contrastive Masked Modeling (SCMM) 프레임워크를 제안합니다. 기존 연구에서 데이터셋 간 분포 차이로 인해 일반화된 모델 개발이 어려웠던 점을 해결하고자 이 프레임워크를 고안하였습니다.

- **Technical Details**: SCMM은 감정의 연속성에 영감을 받아 소프트 대조 학습 (Soft Contrastive Learning)과 새로운 하이브리드 마스킹 전략을 결합하여 짧은 시간 동안의 감정 특징을 효과적으로 학습합니다. 자기 감독 학습 과정에서는 샘플 쌍에 소프트 가중치를 부여하여 샘플 간 유사성 관계의 적응형 학습을 가능하게 합니다. 또한, 여러 인접 샘플의 보완 정보를 가중 집계하여 세부적인 특징 표현을 강화하고 원본 샘플 재구성을 위해 사용합니다.

- **Performance Highlights**: SEED, SEED-IV, DEAP 데이터셋에 대한 실험 결과, SCMM 방법이 최신의 SOTA (state-of-the-art) 성능을 달성하였고, 두 가지 크로스-코퍼스 조건 (동일 클래스와 다른 클래스) 하에서 EEG 기반 감정 인식에서 두 번째로 우수한 방법보다 평균 4.26% 높은 정확도를 보였습니다.



### Chinese Metaphor Recognition Using a Multi-stage Prompting Large Language Mod (https://arxiv.org/abs/2408.09177)
- **What's New**: 본 연구에서는 중국어 은유를 인식하고 생성하는 능력을 증대시키기 위한 다단계 생성 휴리스틱 향상 프롬프트 프레임워크를 제안합니다. 기존의 사전 훈련된 모델로는 은유에서 텐서(tensor)와 차량(vehicle)을 완벽하게 인식하기 어려운 문제를 해결하고자 합니다.

- **Technical Details**: 제안된 시스템은 NLPCC-2024 Shared Task 9에 참여하여, DeBERTa 모델을 사용해 답변 후보를 생성합니다. 첫 번째 단계에서는 답변 후보에 대해 신뢰 점수를 획득하고, 두 번째 단계에서는 질문을 특정 규칙에 따라 클러스터링하고 샘플링합니다. 마지막으로 생성된 답변 후보와 시연을 결합해 휴리스틱 향상 프롬프트를 형성합니다.

- **Performance Highlights**: 제안된 모델은 NLPCC-2024 Shared Task 9의 Subtask 1에서 Track 1에서 3위, Track 2에서 1위, Subtask 2의 두 트랙 모두에서 1위를 차지하는 성과를 거두었습니다. 이는 각각 0.959, 0.979, 0.951, 0.941의 정확도로 우수한 결과를 나타냅니다.



### Ranking Across Different Content Types: The Robust Beauty of Multinomial Blending (https://arxiv.org/abs/2408.09168)
Comments:
          To appear in 18th ACM Conference on Recommender Systems (RecSys24), Bari, Italy. ACM, New York, NY, USA, 3 pages

- **What's New**: 이번 논문에서는 다양한 콘텐츠 유형 간의 순위를 매기는 문제를 다루고, 기존의 learning-to-rank (LTR) 알고리즘과 함께 사용할 수 있는 multinomial blending (MB)이라는 새로운 방식에 대해 소개합니다.

- **Technical Details**: 이 논문은 음악과 팟캐스트 등 서로 다른 콘텐츠 유형의 아이템을 순위 매기는 방법을 제안합니다. 특정 콘텐츠 유형의 사용자 참여 패턴이 다르기 때문에 전통적인 LTR 알고리즘이 효과적이지 않은 문제를 해결하고자 하며, 비즈니스 목표를 달성하기 위해 해석 가능성과 용이성을 중시한 MB 방안을 채택하였습니다. MB는 단일 점수 함수 h를 사용하여 모든 후보 아이템을 점수 매긴 후, 특정 콘텐츠 유형에 따라 샘플링하여 순위를 결정합니다.

- **Performance Highlights**: Amazon Music의 실제 사례를 통해 A/B 테스트를 실시하여 MB 방법이 사용자 참여를 어떻게 강화하는지에 대해 보고하고 있습니다. MB는 특정 콘텐츠 유형의 노출을 증가시키면서도 전체 참여율을 해치지 않는 방향으로 최적화되었습니다.



### Linear Attention is Enough in Spatial-Temporal Forecasting (https://arxiv.org/abs/2408.09158)
- **What's New**: 이번 연구는 교통 예측(task) 분야에서 전통적인 Spatial-Temporal Graph 구조를 탈피하여, 토큰(token) 기반의 Spatial-Temporal Transformer (STformer) 모델을 제안합니다. STformer는 도로 네트워크에서 각 시간 단계의 노드를 독립적인 토큰으로 취급하여 공간-시간 패턴을 학습합니다.

- **Technical Details**: STformer는 각 시간 단계의 센서를 독립적인 ST-Token으로 사용하며, 이를 바닐라 Transformer에 입력하여 공간-시간의 복잡한 패턴을 학습합니다. O(N²) 복잡도를 가진 STformer의 단점을 보완하기 위해, Nyström 방법을 기반으로 하는 NSTformer 변형 모델을 제안하여, O(N) 복잡도로 자가 주의(self-attention)를 근사합니다.

- **Performance Highlights**: 제안된 모델들은 METR-LA와 PEMS-BAY의 두 공공 데이터셋에서 최첨단(State-of-the-Art) 성능을 기록하며, NSTformer는 STformer보다 더 나은 성능을 보였다는 놀라운 결과를 확인했습니다. 컴퓨팅 비용의 효율성을 고려할 때, 이러한 모델은 대규모 도로 네트워크와 장기 예측 과제에서 매우 유리합니다.



### CogLM: Tracking Cognitive Development of Large Language Models (https://arxiv.org/abs/2408.09150)
Comments:
          under review

- **What's New**: 본 연구는 Piaget의 인지 발달 이론(PTC)에 기반하여 대형 언어 모델(LLM)의 인지 수준을 평가하는 기준 CogLM(언어 모델의 인지 능력 평가)를 소개하고, LLM의 인지 능력을 조사한 최초의 연구 중 하나입니다.

- **Technical Details**: CogLM은 1,220개의 질문을 포함하며, 10개의 인지 능력을 평가합니다. 이 질문들은 20명 이상의 전문가들에 의해 제작되었고, LLM의 성능을 다양한 방향에서 측정하기 위한 체계적인 평가 기준을 제공합니다. 실험은 OPT, Llama-2, GPT-3.5-Turbo, GPT-4 모델에서 수행되었습니다.

- **Performance Highlights**: 1. GPT-4와 같은 고급 LLM은 20세 인간과 유사한 인지 능력을 보여주었습니다. 2. LLM의 인지 수준에 영향을 미치는 주요 요인은 매개변수의 크기 및 최적화 목표입니다. 3. 다운스트림 작업에서의 성능은 LLM의 인지 능력 수준과 긍정적인 상관관계를 보입니다.



### Learning to Explore for Stochastic Gradient MCMC (https://arxiv.org/abs/2408.09140)
- **What's New**: 본 논문에서는 Stochastic Gradient MCMC(SGMCMC) 알고리즘의 메타 학습 전략을 제안하여 다중 모드 타겟 분포를 효율적으로 탐색할 수 있도록 하였습니다. 이 접근법은 다양한 작업에서 전이 가능성을 보여 주며, 메타 학습된 SGMCMC의 샘플링 효율 구축을 가능하게 합니다.

- **Technical Details**: 우리는 L2E(Learning to Explore)라는 새로운 메타 학습 프레임워크를 제안합니다. 이 프레임워크는 기존의 수작업 디자인 접근 방식 대신 데이터에서 직접 설계를 학습하며, 다양한 BNN(inference task)을 포괄하는 비선형 회귀를 사용하여 더 나은 믹싱 비율(mixing rates)과 예측 성능을 실현합니다. 또한, 사전 훈련 단계에서 보지 못한 작업에 대해서도 일반화할 수 있는 능력을 갖추고 있습니다.

- **Performance Highlights**: 이미지 분류 벤치마크를 기반으로 BNN을 통한 SGMCMC의 성능 향상을 입증하였으며, 기존의 수동으로 최적화한 SGMCMC에 비해 샘플링 효율성(sampling efficiency)이 크게 개선되었습니다. 이로 인해 계산 비용을 크게 증가시키지 않고도 더 우수한 성능을 달성할 수 있습니다.



### Vanilla Gradient Descent for Oblique Decision Trees (https://arxiv.org/abs/2408.09135)
Comments:
          Published in ECAI2024. This version includes supplementary material

- **What's New**: DTSemNet 아키텍처는 기존의 결정 트리(Decision Tree, DT)를 신경망(Neural Networks, NN)으로 변환하여 효율적인 기울기 강하(Gradient Descent)를 적용할 수 있도록 하는 혁신적인 방법론입니다. 이것은 복잡한 결정 트리 학습의 효율성을 크게 향상시킵니다.

- **Technical Details**: DTSemNet은 비가원적(hard) 경량화 결정 트리와 같은 구조를 가진 신경망으로, ReLU 활성화 함수와 선형 연산을 사용하여 기 differentiable합니다. 각 노드의 결정은 NN 안에서 훈련 가능한 가중치에 1:1로 대응하여, 기존 DT의 복잡성을 개선하면서 효율적인 학습이 가능합니다. 이 방법은 DT의 결정과 리프 노드에서 회귀기(regressor) 학습을 동시에 가능하게 합니다.

- **Performance Highlights**: DTSemNet은 다양한 분류 및 회귀 기준에서 실험을 통해 최첨단 기술로 학습된 유사한 크기의 oblique DT보다 더 높은 정확도를 보였고, DT 학습 시간을 크게 줄였습니다. RL 환경에서도 DTSemNet은 기존 NN 정책과 동일한 성능의 DT 정책을 생성할 수 있음을 입증하였습니다.



### Better Python Programming for all: With the focus on Maintainability (https://arxiv.org/abs/2408.09134)
- **What's New**: 이 연구는 대형 언어 모델(LLMs)로 생성된 코드의 유지보수성을 향상시키는 것을 목표로 하고 있으며, 특히 Python 프로그래밍 언어에 중점을 두고 있습니다. 이전 연구들은 주로 기능적 정확성과 테스트 성공에 집중하였으나, 코드 유지보수성을 간과해 왔습니다. 이를 위해 연구팀은 유지보수성 평가를 위한 특수 설계된 데이터셋을 활용하고, LLM을 코드 리팩토링(refactoring)으로 세밀하게 조정하여 코드 가독성 및 복잡성을 줄이고 유지보수성을 향상시키는 방법을 제안합니다.

- **Technical Details**: 연구에서는 Python 코드의 유지보수성을 중점적으로 다루는 맞춤형 데이터셋을 개발하였고, 이는 가독성(readability), 복잡성(complexity), 코드 리팩토링의 용이성을 강조합니다. LLM을 매개변수 효율적인 미세 조정(parameter-efficient fine-tuning, PEFT) 기술을 사용하여 개선하고, Hugging Face Supervised Fine-tuning Trainer(SFT Trainer) API를 통해 코드 품질과 유지보수성 기준을 높인 모델의 성능을 평가합니다.

- **Performance Highlights**: 연구 결과, 코드 유지보수성을 우선시하여 조정된 LLM 모델은 코드의 유지보수성 기준을 크게 개선하는 것으로 나타났습니다. 이는 AI 지원 소프트웨어 개발의 미래에 대한 유망한 방향을 제시합니다. 이 모델은 코드 생성 후 수유 있는 코드 품질을 초기에 중요시함으로써 기술 부채(technical debt)를 줄이고, 유지보수가 용이한 코드를 생성합니다.



### Identifying Technical Debt and Its Types Across Diverse Software Projects Issues (https://arxiv.org/abs/2408.09128)
- **What's New**: 본 연구는 소프트웨어 프로젝트에서 Technical Debt (TD)의 정확한 식별의 필요성을 강조하며, transformer 기반 모델을 활용한 TD 분류 방법론을 제안합니다. 이는 대규모 소프트웨어 개발 환경에서의 효율적인 TD 식별을 목표로 하고 있습니다.

- **Technical Details**: 연구에서는 GitHub Archive Issues (2015-2024)와 산업 데이터 검증을 통해 다중 이진 분류기를 사용하고 프레임워크인 ensemble learning을 결합한 TD 분류 시스템을 구축하였습니다. 또한, DistilRoBERTa 모델이 GPT 등 대형 언어 모델보다 TD 분류 작업에 더 효과적임을 밝혀냈습니다.

- **Performance Highlights**: 모델의 일반화 능력을 MCC, AUC ROC, Recall, F1 score 등의 메트릭을 사용하여 평가하며, 특정 프로젝트의 컨텍스트에 기반하여 fine-tuning된 transformer 모델의 성능이 특정 작업에 대해 뛰어난 결과를 내는 것으로 나타났습니다.



### Markov Balance Satisfaction Improves Performance in Strictly Batch Offline Imitation Learning (https://arxiv.org/abs/2408.09125)
- **What's New**: 본 연구에서는 환경과의 상호작용 없이 오직 관찰된 행동에만 의존하는 이모테이션 학습(imitation learning, IL) 방법론을 제시했습니다. 기존의 최첨단(state-of-the-art, SOTA) IL 기법과는 달리, 보상 모델을 추정하지 않고 마르코프 균형 방정식(Markov Balance Equation)을 기반으로 새로운 조건부 밀도 추정(conditional density estimation) 방식을 활용합니다.

- **Technical Details**: 우리의 방법은 두 개의 조건부 상태-행동 전이 밀도 함수로부터 데이터를 학습하여, 마르코프 균형 방정식을 만족하는 정책을 학습합니다. 이를 위해 최근 개발된 정규화 흐름(normalizing flows) 기법을 사용하여 밀도 추정을 수행하였습니다. 이러한 접근은 온라인 샘플을 필요로 하지 않으며, 제한된 전문가 데이터만으로도 동작할 수 있게 합니다.

- **Performance Highlights**: 고전 제어(Classic Control) 및 MuJoCo 환경에서의 일련의 수치 실험을 통해, 제안한 방법이 많은 기존 SOTA IL 알고리즘에 비해 일관되게 우수한 성능을 보임을 입증했습니다.



### Selective Prompt Anchoring for Code Generation (https://arxiv.org/abs/2408.09121)
Comments:
          Under review

- **What's New**: 이 논문은 최근 LLMs(대형 언어 모델)의 주의력 감소(self-attention dilution) 문제를 다루며, Selective Prompt Anchoring (SPA)이라는 새로운 방법을 제안합니다. 이는 코드 생성 과정에서 사용자의 초기 프롬프트에 대한 모델의 주의를 극대화하는 접근 방식입니다.

- **Technical Details**: SPA는 선택된 텍스트 부분의 영향을 증폭시켜 코드 생성 시 모델의 출력을 개선합니다. 구체적으로, SPA는 원래의 로짓 분포와 앵커 텍스트가 마스킹된 로짓 분포의 차이를 계산합니다. 이를 통해 앵커 텍스트가 출력 로짓에 기여하는 정도를 근사합니다.

- **Performance Highlights**: SPA를 사용한 결과, 모든 설정에서 Pass@1 비율을 최대 9.7% 향상시킬 수 있었으며, 이는 LLM의 성능을 개선하는 새롭고 효과적인 방법을 제시합니다.



### Fragment-Masked Molecular Optimization (https://arxiv.org/abs/2408.09106)
Comments:
          11 pages, 5 figures, 2 tables

- **What's New**: 이번 논문에서는 PDD(phenotypic drug discovery)에 기반한 새로운 분자 최적화 방법인 FMOP(fragment-masked molecular optimization)을 제안합니다. 이 방법은 특정 타겟 구조에 의존하지 않고, 정확한 구조를 포착하기 어려운 혁신적인 약물 개발을 돕습니다.

- **Technical Details**: FMOP는 훈련 없이 조건부 최적화를 수행할 수 있는 회귀 없는 diffusion 모델을 사용합니다. 이 모델은 마스킹된 분자 영역을 최적화하여 유사한 골격을 가진 새로운 분자를 생성하며, GDSCv2 데이터셋에서 945개의 세포주에 대한 최적화를 수행하였습니다. IC50 값을 기준으로 최적화의 성공률은 94.4%에 달하며 평균 효능은 5.3% 증가했습니다.

- **Performance Highlights**: FMOP의 전반적인 실험 결과는 94.4%의 성공률을 보였으며, 평균 효능 상승률은 5.3%에 이릅니다. 추가적인 ablation 및 시각화 실험을 통해 FMOP의 효과성과 강건성을 입증했습니다.



### Depth-guided Texture Diffusion for Image Semantic Segmentation (https://arxiv.org/abs/2408.09097)
- **What's New**: 이 논문에서는 depth 정보와 RGB 이미지 간의 격차를 줄여서 더 정확한 semantic segmentation을 달성하는 새로운 Depth-guided Texture Diffusion 접근 방식을 소개합니다. 이 방법은 깊이 맵을 텍스쳐 이미지로 변환하고 이를 선택적으로 확산시킴으로써, 구조적 정보를 강화하여 객체 윤곽 추출을 더욱 정밀하게 만듭니다.

- **Technical Details**: 제안된 방법은 깊이 정보의 텍스쳐 세부 정보를 선택적으로 강조하여 깊이 맵과 2D 이미지 간의 호환성을 높입니다. 이 과정 동안 구조적 일관성을 보장하기 위해 `structural loss function`을 이용하여 객체 구조의 완전성을 유지합니다. 최종적으로, 텍스쳐가 정제된 깊이와 RGB 이미지를 통합하여 모델의 정확성을 개선합니다.

- **Performance Highlights**: 폭넓은 데이터셋을 대상으로 한 실험 결과, 제안된 방법인 Depth-guided Texture Diffusion이 기존의 기초 모델들보다 일관되게 우수한 성능을 보였으며, camouflaged object detection, salient object detection 및 실내 semantic segmentation 분야에서 새로운 최첨단 성과를 달성하였습니다.



### Linking Robustness and Generalization: A k* Distribution Analysis of Concept Clustering in Latent Space for Vision Models (https://arxiv.org/abs/2408.09065)
- **What's New**: 이 논문은 vision 모델의 latent space를 평가하기 위한 새로운 방법론을 제안하며, k* Distribution을 사용하여 각 개념의 학습된 latent space를 분석합니다. 이를 통해 기존의 간접 평가 방식과는 다른 접근을 제공합니다.

- **Technical Details**: k* Distribution 방법론은 네트워크에서 취득한 데이터를 각 개념별로 분석하고, Skewness 계수를 사용하여 latent space의 품질을 정량화합니다. 이 방법은 현재의 vision 모델이 개별 개념의 분포를 어떻게 왜곡하고 있는지를 평가하는 데 초점을 맞추고 있습니다. 또한, robust한 모델에서의 분포 왜곡이 감소하는 경향성을 관찰했습니다.

- **Performance Highlights**: 결과적으로, 모델이 여러 데이터셋에 대해 일반화 능력이 향상될수록 개념의 클러스터링이 좋아지는 경향이 있으며, 이는 robust성이 증가함에 따라 발생합니다. 이는 모델의 일반화 능력과 robust성 사이의 관계를 드러냅니다.



### Learning to Route for Dynamic Adapter Composition in Continual Learning with Language Models (https://arxiv.org/abs/2408.09053)
- **What's New**: 본 논문에서는 PEFT(Parament Efficient Fine-Tuning) 모듈을 새로운 작업에 대해 독립적으로 훈련하고, 메모리로부터 샘플을 활용하여 이전에 학습한 모듈의 조합을 학습하는 L2R(Learning to Route) 방법을 제안합니다. 이는 기존 PEFT 방법의 두 가지 주요 한계를 해결합니다.

- **Technical Details**: L2R 방법은 PEFT 모듈을 학습할 때 이전 작업으로부터의 간섭을 피하고, 테스트 타임 추론 이전에 메모리 내 예시를 활용하여 PEFT 모듈을 동적으로 조합할 수 있는 라우팅(routing) 기능을 학습하는 구조입니다. 이는 로컬 적응(local adaptation)을 활용하여 모듈의 적절한 조합을 수행합니다.

- **Performance Highlights**: L2R은 여러 벤치마크와 CL 설정에서 다른 PEFT 기반 방법들에 비해 개선된 모듈 조합 성능을 보여주며, 이를 통해 일반화 및 성능 향상을 달성했습니다. 본 방법은 기존 우수한 성능을 유지하면서 새로운 작업을 지속적으로 학습하는 모델 개발의 실질적인 목표를 충족시킵니다.



### Language Models Show Stable Value Orientations Across Diverse Role-Plays (https://arxiv.org/abs/2408.09049)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)이 다양한 페르소나(persona)를 채택하더라도 일관된 가치 지향(value orientation)을 보인다는 사실을 입증하였습니다. 이는 LLM의 응답에서 지속적인 관성(inertia) 현상이 나타남을 보여줍니다.

- **Technical Details**: 연구진은 'role-play-at-scale' 방법론을 도입하여 LLM에 무작위로 다양한 페르소나를 부여한 후 응답의 거시적(microscopic) 경향을 분석하였습니다. 이 방법은 기존의 방식과 달리, 무작위 페르소나가 적용된 여러 질문에 대해 모델이 어떻게 반응하는지를 체계적이고 확장 가능한 방법으로 진단합니다.

- **Performance Highlights**: 이 연구는 다양한 역할 놀이(role-play) 시나리오에서 LLM의 응답 패턴이 일관됨을 밝혀냈으며, 이는 LLM에 내재된 경향성(inherent tendencies)을 나타냅니다. 이러한 발견은 기초 모델(foundation models)에서의 가치 정렬(value alignment) 논의에 기여할 뿐만 아니라, LLM의 내재된 편향(bias)을 발견하는 진단 도구로서 'role-play-at-scale'의 효능을 보여줍니다.



### mRNA2vec: mRNA Embedding with Language Model in the 5'UTR-CDS for mRNA Design (https://arxiv.org/abs/2408.09048)
- **What's New**: 이 논문은 mRNA 기반의 백신과 치료법 개발에 있어 mRNA 서열 선택의 비용 문제를 해결하기 위한 새로운 방법인 mRNA2vec 모델을 제안합니다. 이 모델은 데이터 변환(data2vec) 기반의 자가 지도 학습(teacher-student learning) 프레임워크를 기반으로 하여 mRNA의 5' 비전사 영역(UTR)과 코딩 서열(CDS)을 통합하여 입력으로 사용합니다.

- **Technical Details**: mRNA2vec는 mRNA 서열의 위치 중요성을 고려하기 위해 확률적 마스킹(probabilistic masking)을 적용하고, 최소 자유 에너지(Minimum Free Energy, MFE) 예측 및 2차 구조(Secondary Structure, SS) 분류를 추가적인 프리텍스트(pretext) 과제로 채택합니다. 이 방법은 mRNA 서열의 맥락(contextual) 수준 학습을 통해 mRNA 표현의 개선을 추구합니다.

- **Performance Highlights**: mRNA2vec는 UTR에서 Translation Efficiency (TE) 및 Expression Level (EL) 예측 작업에서 최신 기술(SOTA)인 UTR-LM과 비교하여 유의미한 개선을 보였습니다. 더불어 CDS에 대한 mRNA 안정성 및 단백질 생산 수준 작업에서도 경쟁력 있는 성능을 보여주었습니다.



### Improving VTE Identification through Language Models from Radiology Reports: A Comparative Study of Mamba, Phi-3 Mini, and BER (https://arxiv.org/abs/2408.09043)
- **What's New**: 이번 연구는 Venous thromboembolism (VTE) 감지를 위한 Mamba 아키텍처 기반 분류기를 도입하여 기존의 복잡한 방법을 개선했습니다. Mamba 모델은 DVT 데이터셋에서 97\%의 정확성과 F1 점수를, PE 데이터셋에서 98\%의 정확성과 F1 점수를 달성했습니다.

- **Technical Details**: Mamba는 State Space Models (SSM)에 영감을 받은 혁신적인 아키텍처로, 긴 시퀀스에 대한 효율성과 스케일을 향상시킵니다. 이 모델은 8K 토큰의 긴 시퀀스를 처리할 수 있어 방사선 보고서와 같은 긴 데이터에 적합합니다. 또한, Phi-3 Mini라는 경량형 Large Language Model (LLM)을 실험했으나 더 많은 파라미터로 인해 계산 성능은 높은 편입니다.

- **Performance Highlights**: Mamba 기반 분류기는 DVT와 PE 데이터셋 모두에서 뛰어난 성능을 보였으며, 특히 모든 이전 방법들보다 높은 정확성과 F1 점수를 기록하였습니다. 이는 VTE 분류 작업에 가장 최적화된 솔루션이 될 것으로 보입니다.



### Efficient Autoregressive Audio Modeling via Next-Scale Prediction (https://arxiv.org/abs/2408.09027)
Comments:
          7 pages, 6 figures, 7 tables

- **What's New**: 이번 논문에서는 오디오 생성 모델의 효율성을 개선하기 위해 새로운 	extbf{S}cale-level 	extbf{A}udio 	extbf{T}okenizer (SAT)를 제안합니다. 이 토크나이저는 개선된 잔여 양자화(residual quantization)를 활용하여 훈련 비용과 추론 시간을 크게 줄입니다.

- **Technical Details**: SAT에 기반한 스케일 수준의 	extbf{A}coustic 	extbf{A}uto	extbf{R}egressive (AAR) 모델링 프레임워크를 도입하여, 다음 토큰 예측이 아닌 다음 스케일 예측으로 전환합니다. 이러한 접근 방식은 오디오 훈련 시 시퀀스 길이 문제를 효과적으로 해결합니다.

- **Performance Highlights**: 제안된 AAR 프레임워크는 AudioSet 벤치마크에서 기존 모델 대비 	extbf{35}$	imes$ 더 빠른 추론 속도와 +	extbf{1.33} Fréchet Audio Distance (FAD)를 달성하였습니다.



### Classifier-Free Guidance is a Predictor-Corrector (https://arxiv.org/abs/2408.09000)
Comments:
          AB and PN contributed equally

- **What's New**: 이 논문에서는 classifier-free guidance (CFG)의 이론적 기초를 조사하며, CFG가 DDPM과 DDIM과 어떻게 다르게 상호작용하는지를 보여줍니다. CFG가 gamma-powered 분포를 생성하지 못한다는 점을 명확히 하며, 이를 통한 이론적 이해를 제공합니다.

- **Technical Details**: 본 연구에서는 CFG가 예측-수정 방법(prediction-correction method)의 일종으로 작용함을 입증합니다. 이 방법은 노이즈 제거와 선명도 조정을 번갈아 수행하며, 이를 predictor-corrector guidance (PCG)라고 정의합니다. CFG의 성능을 SDE 한계(SDE limit)에서 분석하여, 특정 gamma 값에 대해 DDIM 예측기와 Langevin 동역학 수정기를 결합한 형태와 동등함을 보여줍니다.

- **Performance Highlights**: CFG는 대부분의 현대 텍스트-투-이미지(diffusion model) 모델에서 사용되며, 많은 유의한 개선을 보이고 있습니다. 하지만 기존의 이해와는 달리, CFG가 항상 이론적으로 보장된 결과를 생성하는 것은 아니며, DDPM과 DDIM간의 분포 생성 차이를 명확히 밝혀내는 중요한 이론적 발견을 제공합니다.



### Adaptive Uncertainty Quantification for Generative AI (https://arxiv.org/abs/2408.08990)
- **What's New**: 이번 연구는 현대의 블랙박스 모델에 대한 conformal prediction 방법을 제시합니다. 이 방법은 사용자에게 접근할 수 없는 데이터에서 학습된 모델을 기반으로 하며, 적응형으로 예측 공간을 그룹으로 나눈 후 각 그룹마다 conformity 점수를 보정합니다.

- **Technical Details**: 연구에서는 split-conformal inference의 자리를 대체할 수 있는 방법으로, robust regression tree를 사용하여 calibration 세트의 conformity 점수를 적응적으로 그룹화합니다. 이 방식은 새로운 관측값이 나무의 적합도를 중요한 확률로 변화시키지 않도록 설계되었습니다.

- **Performance Highlights**: 저자들은 여러 실제 및 시뮬레이션된 사례에서 예측 불확실성을 줄이기 위해 지역적으로 타이트한 예측 간격을 제공하는 이점을 시연했습니다. 특히, ChatGPT의 예측과 피부 질환 진단을 다루는 두 가지 현실적인 사례 연구에서 효과적인 결과를 보여주었습니다.



### Online SLA Decomposition: Enabling Real-Time Adaptation to Evolving Systems (https://arxiv.org/abs/2408.08968)
Comments:
          The paper has been submitted to IEEE Networking Letters

- **What's New**: 이 논문에서는 다중 도메인에서 네트워크 슬라이스를 다루기 위한 온라인 학습 기반의 SLA (Service Level Agreement) 분해 프레임워크를 제안합니다. 이 프레임워크는 최근 피드백을 활용하여 위험 모델을 동적으로 업데이트하여 안정성과 강건성을 향상시킵니다.

- **Technical Details**: 제안된 프레임워크는 E2E (End-to-End) 서비스 오케스트레이터와 로컬 컨트롤러를 포함하는 두 레벨 네트워크 슬라이싱 관리 시스템 위에서 작동하며, 최근의 피드백을 기반으로 위험 모델을 지속적으로 업데이트합니다. 또한, FIFO (First In First Out) 메모리 버퍼를 활용하여 데이터 관리를 강화합니다.

- **Performance Highlights**: 제안된 방법은 동적 환경 및 희소한 데이터 조건에서 기존의 정적 방법에 비해 SLA 분해의 정확성과 복원력을 향상시키는 것으로 실험적으로 입증되었습니다.



### A Factored MDP Approach To Moving Target Defense With Dynamic Threat Modeling and Cost Efficiency (https://arxiv.org/abs/2408.08934)
- **What's New**: 이 논문은 기존의 공격자 보상에 대한 전제 없이 Markov Decision Process (MDP) 모델을 사용하여 Moving Target Defense (MTD)를 위한 새로운 접근 방식을 제안합니다. 이 접근법은 다이나믹 Bayesian Network을 활용하여 공격자의 실시간 반응을 방어자의 MDP에 통합하고, 공격 응답 예측기를 점진적으로 업데이트하여 적응적이며 강건한 방어 메커니즘을 확보합니다.

- **Technical Details**: 이 연구에서는 방어자의 관점에서 MTD 문제를 결정론적 MDP로 정의합니다. 상태 공간 S는 모든 가능한 시스템 구성으로 구성되고, 행동 공간 A는 방어자가 선택 가능한 스위칭 행동으로 이루어집니다. 각 스위칭 행동은 deterministic하게 다음 구성으로 전이되는 방식으로 처리됩니다. 공격자의 반응은 현재 구성 및 스위칭 행동에 따라 달라지는 이진 값으로 모델링됩니다.

- **Performance Highlights**: 제안된 프레임워크는 National Vulnerability Database (NVD)에서 수집한 공격 데이터를 사용하여 웹 애플리케이션 환경과 네트워크 환경에서 두 가지 넓은 도메인에 걸쳐 효과성을 입증합니다. 경험적 분석을 통해 동일한 선행 지식을 가정하는 다른 방법들과 비교하여 상당한 성능 개선을 보여주었습니다.



### RoarGraph: A Projected Bipartite Graph for Efficient Cross-Modal Approximate Nearest Neighbor Search (https://arxiv.org/abs/2408.08933)
Comments:
          to be published in PVLDB

- **What's New**: 본 논문에서는 교차 모드 (cross-modal) 최근접 이웃 검색(ANNS)에서의 비효율성을 분석하고, query 분포 (query distribution)를 활용한 새로운 그래프 인덱스인 RoarGraph를 제안합니다.

- **Technical Details**: RoarGraph는 bipartite graph를 사용하여 쿼리와 base 데이터 간의 유사성 관계를 그래프 구조로 매핑하고, neighbor-aware projection을 통해 공간적으로 먼 노드 간 경로를 생성합니다. 이 그래프 인덱스는 base 데이터만으로 구성되어 있으며 쿼리 분포로부터 파생된 이웃 관계를 효과적으로 보존합니다.

- **Performance Highlights**: RoarGraph는 현대의 교차 모드 데이터셋에서 기존의 ANNS 방법들에 비해 최대 3.56배 빠른 검색 속도를 달성하며, OOD 쿼리의 90% 재현율을 확보합니다.



### Personalized Federated Collaborative Filtering: A Variational AutoEncoder Approach (https://arxiv.org/abs/2408.08931)
Comments:
          10 pages, 3 figures, 4 tables, conference

- **What's New**: 본 논문에서는 Federated Collaborative Filtering (FedCF) 분야에서 개인 정보를 보호하면서 추천 시스템을 향상시키기 위한 새로운 방법론을 제안합니다. 기존의 방법들이 사용자의 개인화된 정보를 사용자 임베딩 벡터에 집약하는 데 그쳤다면, 본 연구는 이를 잠재 변수와 신경 모델로 동시에 보존하는 방안을 모색합니다.

- **Technical Details**: 제안된 방법은 사용자 지식을 두 개의 인코더로 분해하여, 아키텍처 내에서 공유 지식과 개인화를 나누어 캡처합니다. 전역 인코더는 모든 클라이언트 간에 공유되는 일반적인 잠재 공간에 사용자 프로필을 매핑하며, 개인화된 로컬 인코더는 사용자-specific 잠재 공간으로 매핑합니다. 이를 통해 개인화와 일반화를 균형 있게 처리합니다. 또한, 추천 시스템의 특수한 Variational AutoEncoder (VAE) 작업으로 모델링하고, 사용자의 상호작용 벡터 재구성과 누락된 값 예측을 통합하여 훈련합니다.

- **Performance Highlights**: 제안된 FedDAE 방법은 여러 벤치마크 데이터셋에서 수행한 실험 결과, 기존의 기준 방법들보다 우수한 성능을 나타냈습니다. 이는 개인화와 일반화 사이의 균형을 정교하게 조정하는 게이팅 네트워크의 도입 덕분입니다.



### DePrompt: Desensitization and Evaluation of Personal Identifiable Information in Large Language Model Prompts (https://arxiv.org/abs/2408.08930)
- **What's New**: 이 논문에서는 LLM(대형 언어 모델)과의 상호작용에서 중요한 역할을 하는 프롬프트의 개인 정보 보호 및 효율성 평가 프레임워크인 DePrompt를 제안합니다. 이 프레임워크는 사용자들이 LLM을 안전하게 사용할 수 있도록 하며, 특히 PII(개인 식별 정보) 유출 위험을 줄이는 방법에 중점을 두고 있습니다.

- **Technical Details**: DePrompt는 LLM의 파인튜닝 기법을 활용하여 프롬프트의 개인 정보 유형을 정의하고 높은 정확도로 PII 엔티티를 식별합니다. 키 속성을 분석하고 대립적 생성 탈감춤 방법을 통해 개인 정보 속성과 식별자 간의 연결을 방해하면서도 중요한 의미 콘텐츠를 유지합니다. 또한, 프롬프트의 유용성을 평가하기 위한 지표를 제시하여 프라이버시와 유용성을 균형 있게 측정할 수 있도록 합니다.

- **Performance Highlights**: DePrompt 프레임워크는 PII 엔티티 인식에서 평균 95.95%의 높은 정확도를 달성했습니다. 또한, 기존 전통적인 익명화 방법들과 비교했을 때, 우리의 프레임워크는 프라이버시 보호의 유용성과 모델 추론 결과에서 우수한 성능을 보여주었습니다.



### Cybench: A Framework for Evaluating Cybersecurity Capabilities and Risk of Language Models (https://arxiv.org/abs/2408.08926)
Comments:
          86 pages, 7 figures

- **What's New**: 이 논문에서는 사이버 보안을 위한 언어 모델(LM) 에이전트를 평가하기 위한 새로운 프레임워크인 Cybench를 소개합니다. 이 프레임워크는 에이전트가 자율적으로 취약점을 식별하고 악용(exploit)할 수 있는 능력을 정량화하려고 합니다.

- **Technical Details**: Cybench는 4개의 CTF(Capture the Flag) 대회에서 선택된 40개의 전문 수준의 CTF 과제를 포함하고 있습니다. 각 과제는 설명, 시작 파일, bash 명령을 실행하고 출력을 관찰할 수 있는 환경에서 초기화됩니다. 기존의 LM 에이전트의 한계를 넘어서는 과제들이 많기 때문에, 우리는 17개의 과제에 대해 중간 단계로 나누는 서브태스크(subtask)를 도입하였습니다.

- **Performance Highlights**: 에이전트는 인간 팀이 해결하는 데 최대 11분이 소요된 가장 쉬운 과제만 해결할 수 있었습니다. Claude 3.5 Sonnet과 GPT-4o가 가장 높은 성공률을 기록하였으며, 서브태스크가 있는 경우 전체 과제에서 3.2% 더 높은 성공률을 보였습니다.



### Retail-GPT: leveraging Retrieval Augmented Generation (RAG) for building E-commerce Chat Assistants (https://arxiv.org/abs/2408.08925)
Comments:
          5 pages, 4 figures

- **What's New**: 이번 연구는 소매 전자상거래에서 사용자 참여를 증진하기 위해 제품 추천을 안내하고 장바구니 작업을 지원하는 오픈 소스 RAG(Retrieval-Augmented Generation) 기반 챗봇인 Retail-GPT를 소개합니다.

- **Technical Details**: Retail-GPT는 다양한 전자상거래 도메인에 적응할 수 있는 크로스 플랫폼 시스템으로, 특정 채팅 애플리케이션이나 상업 활동에 의존하지 않습니다. 이 시스템은 인간과 유사한 대화를 통해 사용자 요구를 해석하고, 제품 가용성을 확인하며, 장바구니 작업을 관리합니다.

- **Performance Highlights**: Retail-GPT는 가상 판매 대리인으로 기능하며, 다양한 소매 비즈니스에서 이러한 어시스턴트의 실행 가능성을 테스트하는 것을 목표로 합니다.



### Prefix Guidance: A Steering Wheel for Large Language Models to Defend Against Jailbreak Attacks (https://arxiv.org/abs/2408.08924)
- **What's New**: 본 논문에서는 새로운 jailbreak 방어 방법인 Prefix Guidance (PG)를 제안합니다. 이는 모델의 출력을 시작하는 몇 개의 토큰을 직접 설정함으로써 위험한 쿼리를 인식하도록 모델을 안내하는 방식입니다.

- **Technical Details**: PG 방법은 모델의 고유한 보안 기능과 외부 분류기를 결합하여 jailbreak 공격에 저항합니다. 이 접근 방식은 간단하게 배치할 수 있는 plug-and-play 형태로, 모델의 출력 첫 몇 개의 토큰을 설정하는 것만으로 수행됩니다.

- **Performance Highlights**: PG는 Guanaco, Vicuna v1.5 및 Llama2 Chat 모델에 대해 실시된 실험에서 유효성을 입증하였으며, Just-Eval 벤치마크에서 이전 방법들보다 모델의 성능을 더 잘 유지한다는 결과를 보여주었습니다.



### Supervised and Unsupervised Alignments for Spoofing Behavioral Biometrics (https://arxiv.org/abs/2408.08918)
Comments:
          11 pages, 4 figures, 5 tables, submission in progress

- **What's New**: 이번 논문은 행동 생체 인식 시스템에 대한 템플릿 복원 공격(template reconstruction attack)의 가능성을 탐구하며, 특히 핸드라이팅(handwriting) 모달리티에서의 새로운 공격을 제안합니다.

- **Technical Details**: 우리는 비공식 접근 방식으로 두 가지 생체 인식 시스템(음성 및 손글씨)에 템플릿 복원 공격을 수행합니다. 이를 위해 비 라벨 집합의 임베딩과 알려진 인코더의 임베딩을 정렬하는 방법론을 사용하며,-이를 통해 공격자는 오로지 아키텍처에 대한 지식만 가지고도 공격을 수행할 수 있습니다. 이 과정에서 비지도 학습(unsupervised) 및 지도 학습(supervised) 알고리즘을 활용합니다.

- **Performance Highlights**: 본 논문의 결과는 음성 및 핸드라이팅 생체 인식 시스템에서 효과적인 공격을 할 수 있는 가능성을 입증하며, 비공식 접근 방식에서도 높은 공격 성능을 보입니다.



### A Survey on Blockchain-based Supply Chain Finance with Progress and Future directions (https://arxiv.org/abs/2408.08915)
- **What's New**: 이번 논문에서는 Supply Chain Finance(공급망 금융) 분야에서 Blockchain(블록체인) 기술의 적용을 종합적으로 분석하고, 자동화 및 지능적인 금융 관리를 위한 스마트 계약(Smart Contracts)의 가능성을 탐구합니다.

- **Technical Details**: Supply Chain Finance는 최근 몇 년간 금융의 어려움을 완화하고 다양한 서비스와 응용을 지원할 수 있는 중요한 도구로 자리 잡았습니다. Blockchain의 무결성, 진위성, 개인 정보 보호 및 정보 공유의 특성은 Supply Chain Finance의 요구에 매우 적합합니다. 이 논문은 Blockchain 기술의 적용이 Supply Chain Finance의 정보 비대칭 문제를 완화하고, 신용 감소 및 자금 조달 비용을 줄이며, 스마트 계약을 통해 운영을 개선할 수 있음을 강조합니다.

- **Performance Highlights**: 연구 결과, 189개의 논문이 선정되어 Blockchain 기반 Supply Chain Finance의 현재와 미래 연구 방향을 제시하며, Supply Chain Finance와 Blockchain 기술의 관계를 컴퓨터 과학 관점에서 탐구합니다. 이는 기존의 단순 관리 수준의 연구를 넘어서, 깊이 있는 융합적 접근법을 목표로 합니다.



### Why Do Experts Favor Solar and Wind as Renewable Energies Despite their Intermittency? (https://arxiv.org/abs/2408.08910)
Comments:
          Shifted references from hyperlinks to academic style

- **What's New**: 이 논문은 지속 가능한 에너지 발전으로의 전환이 가속화되는 가운데 비전문가도 이해해야 하는 복잡한 에너지 기술과 시장에 대해 설명합니다. 특히, 태양광 및 풍력 발전이 간헐적임에도 불구하고 왜 미래 에너지 공급의 대부분을 차지할 것으로 예상되는지를 탐구하고 있습니다.

- **Technical Details**: 태양광 및 풍력 발전의 기본 요구 사항이 충족되면(예: 유틸리티 규모로의 확장 가능성 및 자원의 전 세계적인 가용성), 이들의 비용이 경쟁 기술보다 2-4배 낮을 것으로 예측됩니다. 논문은 Renewable Energy의 다양한 기술들을 효율성, 비용, 신뢰성, 불연속성, 환경 영향, 확장성, 자원 가용성 등의 7가지 차원으로 분석하였습니다.

- **Performance Highlights**: 현재(2024년)와 10년 후(2034년)의 기술 성과를 비교한 결과, 태양광 발전이 효율성과 환경 영향을 포함하여 뛰어난 성과를 보이며, 풍력도 좋은 성과를 보입니다. 하지만 히드로 발전은 상대적으로 안정적이지만, 자원 가용성이 제한적이라는 문제가 있으며, 바이오매스와 해양 기술은 아직 성숙하지 않은 기술로 평가되었습니다.



### An Adaptive Differential Privacy Method Based on Federated Learning (https://arxiv.org/abs/2408.08909)
- **What's New**: 이 논문에서는 연합 학습(federated learning)에서 개인 정보 보호를 위해 차등 개인 정보 보호(differential privacy) 방법을 적용하는 새로운 접근법을 제안합니다.

- **Technical Details**: 이 방법은 정확도(accuracy), 손실(loss), 훈련 라운드(training rounds), 데이터셋(dataset) 및 클라이언트의 수에 따라 조정 계수(adjustment coefficient)와 점수 함수(scoring function)를 설정합니다. 이후 이들을 기반으로 개인 정보 보호 예산(privacy budget)을 조정합니다. 로컬 모델 업데이트는 스케일링 팩터(scaling factor)와 노이즈(noise)에 따라 처리됩니다.

- **Performance Highlights**: 실험 평가를 통해 이 방법이 개인 정보 보호 예산을 약 16% 줄이는 동시에 정확도는 대체로 유지함을 보여주었습니다.



### What should I wear to a party in a Greek taverna? Evaluation for Conversational Agents in the Fashion Domain (https://arxiv.org/abs/2408.08907)
Comments:
          Accepted at KDD workshop on Evaluation and Trustworthiness of Generative AI Models

- **What's New**: 이번 연구는 대형 언어 모델(LLM)이 온라인 패션 소매 분야에서 고객 경험 및 패션 검색을 개선하는 데 미치는 실질적인 영향을 탐구합니다. LLM을 활용한 대화형 에이전트는 고객과 직접 상호작용함으로써 그들의 니즈를 표현하고 구체화할 수 있는 새로운 방법을 제공합니다. 이는 고객이 자신의 패션 취향과 의도에 맞는 조언을 받을 수 있도록 합니다.

- **Technical Details**: 우리는 4,000개의 다국어 대화 데이터셋을 구축하여 고객과 패션 어시스턴트 간의 상호작용을 평가합니다. 이 데이터는 LLM 기반의 고객 에이전트가 어시스턴트와 대화하는 형태로 생성되며, 대화의 주제나 항목을 통해 서로의 요구를 파악하고 적절한 상품을 제안하는 방식으로 진행됩니다. 데이터셋은 영어, 독일어, 프랑스어 및 그리스어를 포함하며, 다양한 패션 속성(색상, 유형, 소재 등)을 기준으로 구축되었습니다.

- **Performance Highlights**: 이 연구는 LLM이 고객 요구에 맞는 패션 아이템을 추천하는 데 얼마나 효과적인지를 평가하기 위해 여러 공개 및 비공개 모델(GPT, Llama2, Mistral 등)을 벤치마킹하였습니다. 이로써 패션 분야의 대화형 에이전트가 고객과 백엔드 검색 엔진 간의 강력한 인터페이스 역할을 할 수 있음을 입증하였습니다.



### Bundle Recommendation with Item-level Causation-enhanced Multi-view Learning (https://arxiv.org/abs/2408.08906)
- **What's New**: BunCa는 비대칭(item-level causation-enhanced) 관계를 고려한 새로운 번들 추천 방법론을 제안합니다. 기존의 추천 시스템이 개별 아이템 추천에 중점을 두었다면, BunCa는 연결된 아이템 세트를 추천하여 사용자와 비즈니스의 편의성을 향상시키고자 합니다.

- **Technical Details**: BunCa는 두 가지 뷰, 즉 Coherent View와 Cohesive View를 통해 사용자와 번들에 대한 포괄적인 표현을 제공합니다. Coherent View는 Multi-Prospect Causation Network를 활용하여 아이템 간의 인과 관계를 반영하며, Cohesive View는 LightGCN을 사용하여 사용자와 번들 간의 정보 전파를 수행합니다. 또한, 구체적이고 이산적인 대비 학습(concrete and discrete contrastive learning)을 통해 다중 시점 표현의 일관성과 자기 구별(self-discrimination)을 최적화합니다.

- **Performance Highlights**: BunCa는 3개의 벤치마크 데이터셋에서 광범위한 실험을 수행하여 기존 최첨단 방법들과 비교할 때 뛰어난 성능을 나타냈습니다. 이는 번들 추천 작업에서의 유효성을 입증합니다.



### Audit-LLM: Multi-Agent Collaboration for Log-based Insider Threat Detection (https://arxiv.org/abs/2408.08902)
Comments:
          12 pages, 5 figures

- **What's New**: 이번 연구에서는 감사 로그 기반의 내부자 위협 탐지 프레임워크인 Audit-LLM을 소개하며, 이는 세 가지 협력 에이전트로 이루어져 있습니다. Decomposer 에이전트는 복잡한 ITD 작업을 관리 가능한 하위 작업으로 나누고, Tool Builder 에이전트는 하위 작업을 위한 재사용 가능한 툴을 생성하며, Executor 에이전트는 구성된 툴을 통해 최종 탐지 결론을 생성합니다.

- **Technical Details**: Audit-LLM은 Chain-of-Thought (CoT) 추론을 사용하여 ITD 작업을 하위 작업으로 분해하며, Evidence-based Multi-agent Debate (EMAD) 메커니즘을 통해 두 개의 독립적인 Executor가 결론을 반복적으로 정제하여 합의에 도달하도록 합니다. 이 과정은 LLMs의 신뢰성 문제인 'hallucination'을 해결하는 데 기여합니다.

- **Performance Highlights**: 세 가지 공공 ITD 데이터셋(CERT r4.2, CERT r5.2, PicoDomain)에서 실시한 종합 실험을 통해, Audit-LLM은 기존의 최첨단 모델들보다 우수한 성능을 보여주었고, EMAD 메커니즘을 통해 LLMs가 생성하는 설명의 신뢰성을 크게 향상시켰습니다.



### Kov: Transferable and Naturalistic Black-Box LLM Attacks using Markov Decision Processes and Tree Search (https://arxiv.org/abs/2408.08899)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLM)의 해로운 행동을 유도하기 위해 Markov decision process (MDP)로 문제를 수렴시킨 후, Monte Carlo tree search를 활용하여 블랙박스 모델에서 해로운 행동을 탐색합니다. 자동화된 기법을 통해 자연적인 공격을 위해 log-perplexity를 포함하고 있으며, 이는 더 해석 가능한 공격 생성에 기여합니다.

- **Technical Details**: 연구는 기존의 토큰 레벨 공격 방법으로부터 발전하여 자연어 공격을 최적화하는 방법을 제안합니다. 특히, 블랙박스 모델인 GPT-3.5를 대상으로 프로토타입 테스트를 수행했으며, 10개의 쿼리만으로 jailbreak를 성공적으로 달성하였습니다. 반면, GPT-4 모델에서는 실패하였다.

- **Performance Highlights**: 사전 연구 결과에 따르면, 제안된 알고리즘은 블랙박스 모델에 대한 공격을 최적화하는 데 성공적이며, 최신 모델들이 토큰 수준 공격에 대한 강력함을 보이는 경향을 확인하였습니다. 연구 결과는 모두 오픈 소스 형태로 제공됩니다.



### Enhancing Exploratory Learning through Exploratory Search with the Emergence of Large Language Models (https://arxiv.org/abs/2408.08894)
Comments:
          11 pages, 7 figures

- **What's New**: 이 논문에서는 정보 시대에서 학습자들이 정보 검색 및 활용 방식을 이해하는 데 필요한 새로운 이론적 모델을 제안합니다. 특히, 대규모 언어 모델(LLMs)의 영향을 고려하여 탐색적 학습(exploratory learning) 이론과 탐색적 검색 전략을 결합하였습니다.

- **Technical Details**: 이 연구는 Kolb의 학습 모델을 개선하여 높은 빈도로 탐색할 수 있는 전략과 피드백 루프(feedback loops)를 통합합니다. 이러한 접근법은 학생들의 깊은 인지(deep cognitive) 및 고차원 인지(higher-order cognitive) 기술 발달을 촉진하는 것을 목표로 합니다.

- **Performance Highlights**: 이 논문은 LLMs가 정보 검색(information retrieval) 및 정보 이론(information theory)에 통합되어 학생들이 효율적으로 탐색적 검색을 수행할 수 있도록 지원한다고 논의합니다. 이론적으로는 학생-컴퓨터 상호작용을 촉진하고 새로운 시대의 학습 여정을 지원하는 데 기여하고자 합니다.



### U-MedSAM: Uncertainty-aware MedSAM for Medical Image Segmentation (https://arxiv.org/abs/2408.08881)
- **What's New**: U-MedSAM 모델은 MedSAM 모델과 불확실성 인식 손실 함수(uncertainty-aware loss function), Sharpness-Aware Minimization(SharpMin) 옵티마이저를 통합하여 정확한 마스크 예측을 개선합니다.

- **Technical Details**: U-MedSAM은 픽셀 기반 손실(pixel-based loss), 지역 기반 손실(region-based loss), 분포 기반 손실(distribution-based loss)을 결합한 불확실성 인식 손실을 사용하여 세그멘테이션(segmentation) 정확성과 강인성을 높입니다. SharpMin 옵티마이저는 손실 경량(loss landscape) 내에서 플랫 최소값(flat minima)을 찾아 과적합(overfitting)을 줄이며 일반화(generalization)를 향상시킵니다.

- **Performance Highlights**: U-MedSAM은 CVPR24 MedSAM on Laptop 챌린지에서 기존 모델에 비해 86.10%의 DSC(Dice Similarity Coefficient)를 기록하며 뛰어난 성과를 보였습니다. 이는 불확실성을 인식하는 손실 함수와 SharpMin 최적화 기법을 통해 가능한 결과입니다.



### SHARP-Net: A Refined Pyramid Network for Deficiency Segmentation in Culverts and Sewer Pipes (https://arxiv.org/abs/2408.08879)
- **What's New**: SHARP-Net(Semantic Haar-Adaptive Refined Pyramid Network)을 소개하며, 이 네트워크는 다중 스케일 특징을 포착하는 새로운 구조로 설계되었다. Inception-like 블록과 깊이별 분리 가능한 합성곱을 이용하여 세그멘테이션을 개선한다.

- **Technical Details**: SHARP-Net은 Inception-like 블록과 $1	imes1$, $3	imes3$ 깊이별 분리 가능한 합성곱을 이용하여 고해상도 특징을 생성하며, Haar-like 특징을 통합하여 성능을 극대화한다. 이 모델은 다양한 필터 크기를 사용하여 multi-scale 특징을 효과적으로 캡처한다.

- **Performance Highlights**: SHARP-Net은 Culvert-Sewer Defects 데이터셋에서 77.2%의 IoU를 달성하며, DeepGlobe Land Cover 데이터셋에서는 70.6%의 IoU를 기록했다. Haar-like 특징을 통합하여 기본 모델보다 22.74%의 성능 개선을 이뤄내고, 다른 딥러닝 모델에 적용 시 35.0%의 성능 향상을 보였다.



### Confronting the Reproducibility Crisis: A Case Study of Challenges in Cybersecurity AI (https://arxiv.org/abs/2405.18753)
Comments:
          8 pages, 0 figures, 2 tables, updated to incorporate feedback and improvements

- **What's New**: 본 논문은 사이버 보안 분야에서 AI 기반 연구의 재현 가능성을 보장하는 데 있어 시급한 필요성을 강조하며, 악의적인 교란에 대한 방어를 중심으로 한 연구의 재현성 위기를 다루고 있습니다.

- **Technical Details**: 이 연구는 VeriGauge 툴킷을 사용하여 인증된 레질리언스에 관한 이전의 연구 결과를 확인하는 사례 연구를 통해 소프트웨어 및 하드웨어 호환성, 버전 충돌, 노후화 등으로 인해 생기는 문제를 해결하기 위한 접근 방식을 제시합니다.

- **Performance Highlights**: 이 연구 결과는 사이버 보안 커뮤니티가 AI 시스템의 신뢰성과 효율성을 보장하기 위해 재현 가능성 문제를 해결하는 데 집중해야 함을 강조하며, 표준화된 방법론, 컨테이너화, 종합 문서화 필요성을 제기합니다.



### LEGENT: Open Platform for Embodied Agents (https://arxiv.org/abs/2404.18243)
Comments:
          ACL 2024 System Demonstration

- **What's New**: 이번 논문에서는 LEGENT라는 새로운 오픈 소스 플랫폼을 소개하며, 이를 통해 사용자 친화적인 인터페이스와 대규모 데이터 생성 파이프라인을 기반으로 LLMs(대형 언어 모델) 및 LMMs(대형 다중 모달 모델)을 활용한 임바디드 에이전트의 개발을 용이하게 합니다.

- **Technical Details**: LEGENT는 다채롭고 상호작용이 가능한 3D 환경을 제공하며, 사용자가 이해하기 쉬운 인터페이스와 함께 최첨단 알고리즘을 통해 시뮬레이션된 세계에서 감독을 활용하여 대규모 데이터 생성을 지원합니다. 또한, 인간과 유사한 에이전트가 능동적으로 언어 상호작용을 수행할 수 있습니다.

- **Performance Highlights**: LEGENT에서 생성된 데이터로 훈련된 초기 비전-언어-행동 모델은 GPT-4V를 초과하는 성능을 보였으며, 전통적인 임바디드 환경에서는 경험하지 못했던 강력한 일반화 능력을 보여줍니다.



### SC-Rec: Enhancing Generative Retrieval with Self-Consistent Reranking for Sequential Recommendation (https://arxiv.org/abs/2408.08686)
- **What's New**: 최근 언어 모델(Language Models, LMs)의 발전은 추천 시스템(Recommendation Systems)에서의 사용을 증가시키고 있습니다. 본 논문에서는 정보의 통합과 협력적 지식의 상호 보완성을 반영하는 SC-Rec이라는 새로운 추천 시스템을 제안합니다.

- **Technical Details**: SC-Rec은 다양한 항목 인덱스(item indices)와 여러 프롬프트 템플릿(prompt templates)으로부터 수집한 지식을 학습합니다. 이 시스템은 세 가지 단계로 구성되며, 첫째로 다중 인덱스를 생성하고 둘째로 추천모델을 훈련하며 셋째로 정확도(self-consistency) 기반의 재정렬(reranking)을 수행합니다. 이를 통해 다각적인 정보와 구조적 차이를 활용합니다.

- **Performance Highlights**: SC-Rec은 세 가지 실제 데이터셋에서 수행된 실험을 통해 기존의 최첨단 방법들보다 상당히 우수한 성능을 나타냈으며, 다양한 인덱스와 프롬프트 템플릿을 통해 서로 보완적인 고급 정보를 효과적으로 통합함으로써 높은 품질의 재정렬된 리스트를 생성합니다.



### MAT-SED: A Masked Audio Transformer with Masked-Reconstruction Based Pre-training for Sound Event Detection (https://arxiv.org/abs/2408.08673)
Comments:
          Received by interspeech 2024

- **What's New**: 본 논문에서는 SED(소리 이벤트 탐지) 작업을 위한 순수 Transformer 기반의 모델인 MAT-SED를 제안합니다. 이 모델은 RNN 기반의 컨텍스트 네트워크 대신 상대적 위치 인코딩을 갖춘 Transformer를 사용하여 시간적 의존성을 더 잘 모델링합니다.

- **Technical Details**: MAT-SED는 크게 두 가지 구성 요소로 나뉩니다: 인코더 네트워크와 컨텍스트 네트워크. 인코더 네트워크는 mel-spectrogram에서 피쳐를 추출하고, 컨텍스트 네트워크는 인코더에서 나온 잠재 피쳐의 시간적 의존성을 캡처합니다. 또한, masked-reconstruction 기반의 사전 학습 방식을 채택하여 컨텍스트 네트워크를 자가 감독식(self-supervised)으로 사전 학습하고 있습니다.

- **Performance Highlights**: DCASE2023 Task 4에서 MAT-SED는 각각 0.587의 PSDS1 및 0.896의 PSDS2를 달성하여 최신 성능을 초과했습니다. 이는 제안하는 접근 방식이 SED에서의 패러다임 전환을 의미함을 보여줍니다.



### Sum-Product-Set Networks: Deep Tractable Models for Tree-Structured Graphs (https://arxiv.org/abs/2408.07394)
- **What's New**: 본 논문은 확률적 회로(probabilistic circuits)의 확장인 sum-product-set networks를 제안하여 비구조적 텐서 데이터에서 트리 구조 그래프 데이터로의 전환을 다루고 있습니다.

- **Technical Details**: 논문에서 제안하는 sum-product-set networks는 무작위 유한 집합(random finite sets)을 사용하여 그래프의 변수 개수인 노드와 엣지를 반영하고, 이를 통해 정확하고 효율적인 추론을 가능하게 합니다.

- **Performance Highlights**: 저자들은 이 모델이 신경망을 기반으로 한 여러 비추정 가능한(intractable) 모델과 유사한 성능을 발휘한다는 점을 입증했습니다.




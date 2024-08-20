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




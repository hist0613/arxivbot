### Improving LLMs for Recommendation with Out-Of-Vocabulary Tokens (https://arxiv.org/abs/2406.08477)
- **What's New**: 본 논문에서는 LLM(대형언어모델, Large Language Models)을 기반으로 한 추천 시스템에서 사용자 및 아이템을 효과적으로 토크나이징(tokenizing)하는 방법을 제안합니다. 특히, 기존의 in-vocabulary 토큰 외에 out-of-vocabulary(OOV) 토큰을 활용하여 사용자와 아이템 간의 상관관계를 더 잘 반영하고, 다변성을 확보해 추천 성능을 향상시키고자 합니다.

- **Technical Details**: 논문에서 제안하는 META ID(META-path-guided Identifier) 프레임워크는 메타-패스(meta-path)를 사용하여 사용자-아이템 상호작용을 표현하고, 이를 바탕으로 생성된 스킵그램 모델(skip-gram model)에서 사용자 및 아이템의 표현을 얻습니다. 연산된 표현들을 클러스터링하여 유사한 속성을 가진 사용자/아이템 조합이 동일한 OOV 토큰을 공유하도록 합니다. 이를 통해 학습된 OOV 토큰을 LLM의 어휘에 통합하여 사용자와 아이템 간의 상관관계를 보다 잘 포착할 수 있습니다.

- **Performance Highlights**: 실험 결과, META ID는 기존의 최첨단 방법들보다 다양한 추천 작업에서 더 나은 성능을 보였습니다. 특히, 메모리제이션(memorization)과 다변성(diversity) 점수가 향상되어 추천 작업의 성능 향상을 가져왔습니다.



### Wiki Entity Summarization Benchmark (https://arxiv.org/abs/2406.08435)
- **What's New**: Google Research 팀은 새로운 엔티티 요약(이하 ES) 벤치마크인 WikES를 소개합니다. WikES는 기존의 몇 백 개 엔티티에 국한되지 않고, 광범위한 지식 그래프 구조를 포함한 포괄적인 벤치마크입니다.

- **Technical Details**: WikES는 엔티티와 그 요약, 그리고 이들의 연결을 포함합니다. 특히, WikES는 인간의 주석(annotate)을 필요로 하지 않기 때문에 비용 효율적이며, 여러 도메인에서 일반화가 가능한 방법론을 제안합니다. 그래프 알고리즘(graph algorithms)과 자연어 처리 모델(NLP models)을 사용하여 다양한 데이터 소스를 결합함으로써 이를 달성했습니다. 또한, WikES는 데이터셋 생성기를 제공하여 지식 그래프의 다양한 영역에서 ES 알고리즘을 테스트할 수 있습니다.

- **Performance Highlights**: WikES는 기존 데이터셋과의 비교를 통해 ES 메서드의 유용성을 확인하는 여러 실증 연구를 포함하고 있습니다. 이를 통해 WikES의 확장 가능성과 복잡한 지식 그래프의 위상과 의미론을 포착할 수 있는 능력을 증명했습니다. 해당 데이터, 코드, 모델은 제공된 URL에서 확인할 수 있습니다.



### Boosting Multimedia Recommendation via Separate Generic and Unique Awareness (https://arxiv.org/abs/2406.08270)
- **What's New**: 멀티미디어 추천 시스템에서 개별 모달리티(modality) 특성과 공통 모달리티 특성을 동시에 학습하는 새로운 프레임워크인 SAND(Separate Alignment aNd Distancing)가 제안되었습니다. 이는 단순히 모달리티 간의 정렬에만 집중하는 기존 방법의 한계를 극복하고자 합니다.

- **Technical Details**: SAND 프레임워크는 모달-고유(modal-unique)와 모달-일반(modal-generic) 특성을 별도로 학습합니다. 이를 위해 각 모달 특성을 공통 부분과 고유 부분으로 나눈 후, 다양한 모달리티의 의미 정보를 더 잘 통합하기 위해 SoloSimLoss를 설계했습니다. 고유 모달리티와 일반 모달리티 간의 거리를 두어 각 모달리티가 고유의 보완 정보를 유지하도록 합니다. 평가 모듈에서는 mutual information minimization(상호 정보 최소화)와 단순한 negative ℓ2 거리(negative ℓ2 distance)를 선택적인 기술 옵션으로 제공하여 자원 환경에 맞게 적용할 수 있습니다.

- **Performance Highlights**: 세 가지 대표적인 데이터 세트를 사용한 실험 결과, SAND 프레임워크는 기존 방법에 비해 높은 효과성과 일반화를 보여주었습니다. 이는 모달리티 간의 고유 정보를 잘 반영함으로써 멀티미디어 추천의 성능을 크게 향상시켰음을 의미합니다.



### GPT4Rec: Graph Prompt Tuning for Streaming Recommendation (https://arxiv.org/abs/2406.08229)
Comments:
          Accepted by SIGIR 2024. arXiv admin note: text overlap with arXiv:2303.11700 by other authors

- **What's New**: GPT4Rec을 선보입니다, 개인 맞춤 추천 시스템에서 지속적으로 변화하는 사용자 선호도와 새로운 사용자 및 아이템에 대한 대응을 위한 혁신적인 방법입니다. 이 모델은 기존의 데이터 재재생이나 모델 분리 전략의 한계를 뛰어넘어, 지속적인 그래프 학습을 통해 스트리밍 추천 시스템의 문제를 해결합니다.

- **Technical Details**: GPT4Rec은 사용자-아이템 상호작용 그래프를 여러 뷰로 분해하여 특정 관계 패턴을 분리합니다. 노드 수준의 프롬프트(node-level prompts)는 개별 노드의 속성 변화를 다루고, 구조 수준의 프롬프트(structure-level prompts)는 그래프 내 전반적인 연결 패턴을 조절합니다. 마지막으로 뷰 수준의 프롬프트(view-level prompts)는 분리된 뷰에서 정보를 종합하여 일관된 그래프 표현을 유지하게 합니다.

- **Performance Highlights**: 네 가지 실제 데이터셋에 대한 실험 결과, GPT4Rec이 기존의 방법들보다 효과적이고 효율적임을 증명하였습니다. 특히 지속적으로 변하는 데이터 환경에서도 중요한 역사적 정보를 보존하면서 새로운 데이터에 효과적으로 적응할 수 있음을 보여주었습니다.



### Graph Bottlenecked Social Recommendation (https://arxiv.org/abs/2406.08214)
Comments:
          Accepted by KDD 2024

- **What's New**: 소셜 네트워크의 등장으로 소셜 추천이 개인화된 서비스에서 중요한 기술로 자리 잡았습니다. 최근 그래프 기반 소셜 추천이 고차의 소셜 인플루언스를 반영해 유망한 결과를 보였지만, 실제 세계의 소셜 네트워크는 잡음이 많아(불필요한 소셜 관계) 정확한 사용자 선호도 파악에 방해가 될 수 있습니다. 이 논문에서는 정보를 보틀넥 관점에서 학습하여 소셜 구조를 노이즈 제거하고 추천 작업을 촉진하기 위해 새로운 Graph Bottlenecked Social Recommendation(GBSR) 프레임워크를 제안합니다.

- **Technical Details**: GBSR는 모델-독립적인 소셜 디노이징 프레임워크로, 디노이즈된 소셜 그래프와 추천 레이블 사이의 상호 정보를 최대화하고, 디노이즈된 소셜 그래프와 원래의 소셜 그래프 사이의 상호 정보를 최소화하는 것을 목표로 합니다. 기술적으로 GBSR은 두 가지 구성 요소로 구성되어 있습니다: 1) 사용자 선호를 유도하는 소셜 그래프 정제, 2) HSIC(힐버트-슈미트 독립 기준)에 기반한 보틀넥 학습.

- **Performance Highlights**: 광범위한 실험 결과, 제안된 GBSR은 높은 성능과 다양한 백본과의 호환성을 포함하여 우수한 성능을 보여주었습니다. GBSR은 불필요한 소셜 관계를 효과적으로 줄여 소셜 추천을 향상시킵니다.



### Prediction of the Realisation of an Information Need: An EEG Study (https://arxiv.org/abs/2406.08105)
- **What's New**: 정보 검색(IR)의 기본 목표 중 하나인 정보 요구(IN)의 실현을 실시간으로 예측하는 방식으로 뇌파 데이터(EEG)를 활용한 연구가 소개되었습니다. 지난 연구들에서는 EEG 데이터를 통해 신경 프로세스를 실시간으로 분석해왔으나, 실제 검색 경험에 마지막으로 활용하는 단계까지 이루어지지 않았습니다. 이번 연구에서는 14명의 피험자를 대상으로 질문-응답(Q/A) 작업을 수행하며 EEG 데이터를 분석해 정보 요구의 실현 여부를 예측하는 데 주목하였습니다.

- **Technical Details**: EEG 데이터는 14명의 피험자로부터 40개의 전극을 사용해 캡처되었으며, 데이터 샘플링은 500Hz로 이루어졌습니다. Q/A 작업은 TREC-8 및 TREC-2001에서 가져온 일반 지식 질문으로 구성되었습니다. 질문의 난이도는 독립적인 평가자 두 명이 평가하였으며 최종적으로 120개의 질문이 선정되었습니다. 피험자들은 질문이 한 단어씩 화면에 표시되는 동안 EEG 신호를 기록하였고, 질문이 모두 표시된 후에 올바른 답변을 선택하거나 '모르겠다' 옵션을 선택하도록 하였습니다. 머신 러닝 모델을 통해 EEG 데이터를 분석하여 정보 요구의 실현 여부를 예측하였습니다.

- **Performance Highlights**: 연구 결과, EEG 데이터를 사용하여 모든 피험자에 대해 정보 요구의 실현 여부를 73.5%의 정확도로 예측할 수 있었으며, 개별 피험자에 대해 90.1%의 정확도를 기록하였습니다. 또한, Q/A 작업 중 정보 요구의 실현 여부를 강하게 나타내는 문장의 구간을 식별하였습니다.



### A Self-boosted Framework for Calibrated Ranking (https://arxiv.org/abs/2406.08010)
Comments:
          KDD 2024

- **What's New**: 이번 연구에서는 Calibrated Ranking의 새로운 접근 방식인 Self-Boosted Calibrated Ranking 프레임워크 (SBCR)를 제안합니다. SBCR은 기존의 다목적 접근 방식에서 발생하는 두 가지 주요 문제를 해결하고, 광고 클릭율 (CTR)과 같은 실세계 애플리케이션의 성능을 향상시키기 위해 설계되었습니다.

- **Technical Details**: 기존의 다목적 CR 접근 방식은 데이터 셔플링 (data shuffling)과 충돌하고, 단일 확률 예측에 대해 두 개의 상충하는 손실 함수를 동시에 적용함으로 인해 최적의 트레이드오프를 이루기 어렵습니다. 이러한 문제를 해결하기 위해, 우리는 Self-Boosted pairwise loss와 Calibration 모듈을 소개합니다. Self-Boosted pairwise loss는 데이터 셔플링을 가능하게 하며, Calibrated Ranking 접근 방식에서 발생하는 희생을 줄입니다.

- **Performance Highlights**: 제안된 SBCR 프레임워크는 Kuaishou 비디오 검색 시스템에서 검증되었으며, 오프라인 실험에서 높은 성능을 보였습니다. 온라인 A/B 테스트에서는 기존의 프로덕션 베이스라인을 능가하며, CTR과 사용자 시청 시간 측면에서 상당한 향상을 이끌어냈습니다.



### Counteracting Duration Bias in Video Recommendation via Counterfactual Watch Tim (https://arxiv.org/abs/2406.07932)
Comments:
          Accepted by KDD 2024

- **What's New**: 이 논문은 동영상 추천 시스템에서 사용자 관심사를 더 정확하게 반영하기 위해 '반사실적 시청 시간(Counterfactual Watch Time, CWT)'이라는 개념을 도입했습니다. CWT는 사용자가 동영상이 충분히 길다면 얼마나 오래 시청할지를 나타내는 잠재적 시청 시간입니다. 이를 통해, 기존 시청 기록이 사용자 관심을 정확히 반영하지 못하는 '지속 시간 편향(Duration Bias)' 문제를 해결할 수 있습니다.

- **Technical Details**: 제안된 방법은 '반사실적 시청 모델(Counterfactual Watch Model, CWM)'을 중심으로 하여, 경제적 관점에서 사용자의 시청 행동을 모델링합니다. CWM은 사용자가 동영상을 시청하면서 얻는 보상(marginal rewards)과 투자된 시청 시간(watching cost)을 고려하여, 사용자가 최대 이득을 얻는 시점을 CWT로 정의합니다. 또한, CWM은 비용 기반 변환 함수(cost-based transform function)를 사용하여 CWT를 사용자 관심의 추정치로 변환합니다. 이 모델은 관찰된 사용자 시청 시간을 바탕으로 정의된 반사실적 우도 함수(counterfactual likelihood function)를 최적화함으로써 학습됩니다.

- **Performance Highlights**: 세 개의 실제 동영상 추천 데이터셋과 온라인 A/B 테스트를 통해, 제안된 CWM이 동영상 추천의 정확도를 효과적으로 향상시키고, 지속 시간 편향을 상쇄하는 데 성공했습니다. 특히, 완전히 재생된 기록에서의 사용자 관심을 더욱 정밀하게 측정할 수 있음을 입증했습니다.



### Bridging the Gap: Unravelling Local Government Data Sharing Barriers in Estonia and Beyond (https://arxiv.org/abs/2406.08461)
- **What's New**: 에스토니아의 디지털 정부 성공에도 불구하고, 지역 수준에서의 공공데이터(OGD) 개방 이니셔티브가 여전히 지속적인 도전에 직면하고 있습니다. 이 연구는 에스토니아 지방 자치단체들이 OGD를 개방하는 것을 막는 장애물들을 조사하고 있습니다. 연구는 개선된 인식, 데이터 거버넌스 프레임워크 향상, 중앙 및 지방 당국 간의 협력 강화 등의 실질적인 권고안을 제안합니다.

- **Technical Details**: 이 연구는 에스토니아 지방 자치단체와의 인터뷰를 통해 질적 접근 방법을 사용하였으며, OGD에 적응된 'Innovation Resistance Theory' 모델을 기반으로 합니다. 이를 통해 데이터 공유를 방해하는 장벽을 밝히고, 지방 정부 맥락에 맞춘 수정된 이론 모델을 제안합니다.

- **Performance Highlights**: 이 연구는 에스토니아의 국가 정책과 지방 이행 간의 격차를 메우기 위한 실천 가능한 권고안을 제공함으로써, 더욱 탄력적이고 지속 가능한 오픈 데이터 생태계에 기여합니다. 정책 입안자와 실무자에게 지역 OGD 이니셔티브를 우선시할 필요성을 강조합니다.



### "It answers questions that I didn't know I had": Ph.D. Students' Evaluation of an Information Sharing Knowledge Graph (https://arxiv.org/abs/2406.07730)
- **What's New**: 본 연구는 학제간 박사 과정 학생들을 위한 정보 교환 및 의사결정을 돕는 지식 그래프(Knowledge Graph) 모델을 개발하였습니다. 지식 그래프는 여러 출처에서 추출한 중요한 정보와 그 관계를 포함하며, 특히 새로운 학생들에게 불확실성과 학업 스트레스를 줄이는 데 도움이 됩니다.

- **Technical Details**: 이 지식 그래프는 주요 카테고리와 그 관계를 포함하여 여러 출처에서 정보를 추출하여 만들어집니다. 연구는 참여적 설계를 통해 지식 그래프의 사용성을 평가하였으며, 사용자들이 정보 탐색과 의사결정에 도움을 주는 것에 중점을 두었습니다. 특히, 학생-교수 네트워크 탐색, 마일스톤 추적, 집계 데이터에 대한 빠른 접근, 그리고 군중 소싱된 동료 학생 활동에 대한 통찰을 제공하는 기능들이 강조되었습니다.

- **Performance Highlights**: 지식 그래프와의 상호작용은 박사 과정 학생들에게 불확실성과 스트레스를 감소시키는 데 큰 도움을 주었고, 특히 학제간 환경에서의 협업 선택 시 의사결정에 큰 효과를 보였습니다. 맞춤형 정보를 제공함으로써 정보 발견 및 의사결정 과정의 효율성을 크게 개선할 잠재력을 가지고 있습니다. 타 분야 및 프로그램에서도 이 모델을 적용할 수 있습니다.




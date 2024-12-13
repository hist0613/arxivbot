New uploads on arXiv(cs.CL)

### AgentTrek: Agent Trajectory Synthesis via Guiding Replay with Web Tutorials (https://arxiv.org/abs/2412.09605)
Comments:
this https URL

- **What's New**: 이번 연구에서는 GUI(그래픽 사용자 인터페이스) 에이전트를 자동화하기 위한 새로운 데이터 합성 파이프라인인 AgentTrek을 제안합니다. 기존의 접근 방식은 비싼 인력 자원에 의존하지만, 본 연구에서는 자동으로 웹 튜토리얼에서 정보를 수집하여 지속 가능한 데이터 생성을 가능하게 합니다. 이러한 혁신은 GUI 에이전트의 훈련을 대규모로 진행할 수 있는 가능성을 열어줍니다.

- **Technical Details**: AgentTrek은 웹 튜토리얼과 유사한 텍스트를 자동으로 수집하여, 이를 단계별 지침을 포함하는 작업 목표로 변환합니다. 또한, Visual-Language Model(VLM) 에이전트를 사용하여 실제 디지털 환경에서 이러한 작업을 수행합니다. 마지막으로, VLM 기반 평가자가 생성된 경로의 정확성을 검증합니다.

- **Performance Highlights**: 이 연구에서 생성된 합성 경로로 훈련된 GUI 에이전트는 기존 모델에 비해 기초(Grounding) 및 계획(Planning) 성능이 크게 향상됨을 입증했습니다. 또한, 본 방법은 전통적인 인력 주도의 주석 방법에 비해 비용 효율성이 더 높습니다. 이러한 결과는 웹 튜토리얼을 활용한 가이드 재생이 대규모 GUI 에이전트 훈련을 위한 현실적인 전략임을 강조합니다.



### OpenNER 1.0: Standardized Open-Access Named Entity Recognition Datasets in 50+ Languages (https://arxiv.org/abs/2412.09587)
- **What's New**: OpenNER 1.0은 다양한 언어와 개체 유형을 포함한 명명된 개체 인식(NER) 데이터셋의 표준화된 컬렉션을 제공합니다. 34개의 데이터셋은 51개 언어를 아우르며, 이들은 다양한 명명된 개체 온톨로지(ontology)로 주석이 달려 있습니다. OpenNER의 주요 목적은 다국적 및 다온톨로지 NER 연구를 위한 통합된 리소스를 제공하는 것입니다.

- **Technical Details**: OpenNER의 데이터셋은 모두 유효한 BIO 'CoNLL' 형식으로 변환되었으며, 개체 유형 이름을 표준화하여 다국어 평가를 쉽게 지원합니다. 데이터의 수집 및 주석은 명확한 주석 가이드라인에 따라 수동으로 이루어졌으며, 전통적인 명명된 개체인 인물, 장소, 조직 등을 중심으로 합니다. OpenNER은 개체 유형의 일관성을 요구하지 않으며, 원래 데이터셋에서 주석된 모든 유형을 포함합니다.

- **Performance Highlights**: OpenNER은 최근 모델과 비교를 위한 3개의 사전 훈련된 다국어 언어 모델을 사용하여 기준 모델을 제공합니다. 이는 NER 연구를 촉진하고, 다국어 NER 모델 개발을 지원하는 데 기여할 것입니다. 이와 함께 OpenNER은 기존의 다양한 데이터셋들을 활용할 수 있는 기회를 제공하여, NER 성능 향상을 위한 기반을 마련하고 있습니다.



### DiverseAgentEntropy: Quantifying Black-Box LLM Uncertainty through Diverse Perspectives and Multi-Agent Interaction (https://arxiv.org/abs/2412.09572)
- **What's New**: 이 논문에서는 새로운 방법인 DiverseAgentEntropy를 제안하여 큰 언어 모델(LLMs)의 불확실성을 평가합니다. 기존의 자기 일관성(self-consistency) 기반 방법들이 모델의 진짜 불확실성을 정확히 포착하지 못하는 문제를 해결하고자 합니다. 우리는 다수의 에이전트 상호작용을 통해 모델이 동일한 원본 쿼리에 대한 응답을 다양한 각도에서 일관되게 회상해야 한다고 가정합니다.

- **Technical Details**: DiverseAgentEntropy는 원본 쿼리에 대한 최종 에이전트 응답의 가중 엔트로피(weighted entropy)를 통한 신뢰할 수 있는 불확실성 측정을 활용합니다. 이 방법은 원본 쿼리에 대한 응답의 일관성을 평가하며, 불확실성이 높을 때 응답을 보류하는 절차(abstention policy)를 채택합니다. 이를 통해 우리는 모델 신뢰성과 허위 응답을 효과적으로 평가하며, 자기 일관성 기반 기법들보다 AUROC 점수를 뛰어넘는 성능을 보입니다.

- **Performance Highlights**: 우리의 접근 방식은 기존 방법에 비해 질문에 대한 정확성이 2.5% 향상되었으며, 다양한 QA 작업에서 효과적으로 작동합니다. 연구 결과, LLM이 정확한 정보를 보유하고 있음에도 불구하고, 다양한 각도에서의 질문에 대해 일관된 응답을 제공하지 못하는 경향이 있다는 사실이 드러났습니다. 이는 파라메트릭 지식(retrieval of parametric knowledge)의 향상이 필요함을 강조합니다.



### JuStRank: Benchmarking LLM Judges for System Ranking (https://arxiv.org/abs/2412.09569)
- **What's New**: 이번 연구에서는 LLM 기반의 평가자가 시스템 간의 상대적 질을 평가하는 새로운 벤치마크인 JuStRank를 소개합니다. JuStRank는 모델의 정확한 순위를 매기는 능력을 통해 평가자를 비교하며, 더불어 평가자의 시스템 편향 성향을 정량화합니다. 이 연구는 시스템 수준에서의 평가자의 행동과 품질을 탐구하는 첫 번째 대규모 연구가 됩니다.

- **Technical Details**: 연구에서 LLM 기반 평가자가 다양한 시스템의 출력을 평가하여 생성된 시스템 점수를 기준으로 한 정확한 모델 성능 순위를 도출하는 방법론을 제안합니다. 기존 연구들은 인스턴스 기반의 성능 측정에 집중해왔지만, 본 연구는 시스템 레벨 평가의 중요성을 강조합니다. 이를 통해 평가자가 각기 다른 모델을 비교하고 순위를 매길 때 발생하는 편향과 다면적인 행동 특성들을 분석합니다.

- **Performance Highlights**: JuStRank는 4848개의 최신 LLM들과 보상 모델들을 포괄하는 대규모 벤치마크로, 평가의 일관성과 정확성을 크게 향상시킬 가능성을 제시합니다. 이 연구를 통해서 시스템 수준의 평가자가 일관성을 높이고 우수한 모델과 열등한 모델 간의 간극을 더욱 두드러지게 만드는 성향을 가질 수 있음을 보여줍니다. 또한, 시스템 모델에 대한 평가자의 편향을 규명함으로써 더욱 정확한 순위 산출을 꾀할 수 있게 됩니다.



### The Impact of Copyrighted Material on Large Language Models: A Norwegian Perspectiv (https://arxiv.org/abs/2412.09460)
Comments:
          pre-print, under review

- **What's New**: 이번 논문은 대규모 언어 모델(LLMs) 교육에 있어 저작권이 보호된 자료의 사용이 미치는 영향을 평가하기 위한 프레임워크를 제안합니다. 특히 노르웨이 언어에 대해 연구가 진행되었으며, 책과 신문 자료는 긍정적인 성과를 보인 반면, 허구(fiction) 작품은 성과 저하를 일으킬 가능성이 있음을 발견했습니다. 이 연구 결과는 AI 발전에 기여한 저작물에 대한 보상 체계 수립에 기여할 수 있습니다.

- **Technical Details**: 본 연구에서는 노르웨이 언어 데이터를 포함한 다양한 코퍼스를 구성하여 LLM을 훈련합니다. 이를 통해 저작권이 보호된 자료와 비보호 자료의 기여도를 정량적으로 평가하고, 텍스트 생성, 번역, 요약, 질문-답변, 감정 분석 등 다양한 자연어 처리 작업에서 모델 성능을 비교합니다. 평가 프레임워크는 생성 능력 및 자연어 이해를 중심으로 다양한 NLP 메트릭을 사용하여 수행됩니다.

- **Performance Highlights**: 실험 결과, 저작권이 보호된 자료가 포함된 모델이 높은 품질의 생성을 보여주었으며, 이는 노르웨이 기준에 따라 정성적 및 정량적 평가로 분석되었습니다. 연구에 사용된 데이터셋은 저작권이 보호되지 않은 자료보다 높은 성능을 기록했으며, 향후 저작권 관련 정책 및 보상 방안 설계에 대한 중요한 통찰력을 제공할 것으로 기대됩니다.



### Unifying AI Tutor Evaluation: An Evaluation Taxonomy for Pedagogical Ability Assessment of LLM-Powered AI Tutors (https://arxiv.org/abs/2412.09416)
Comments:
          8 pages

- **What's New**: 본 논문에서는 최신 대형 언어 모델(LLMs)이 교육 대화에서 AI 튜터로서 효과적인지, 그리고 효과적인 AI 튜터링을 위한 교육적 능력을 보여주는지 조사합니다. 기존의 평가 노력들은 주관적인 프로토콜과 벤치마크에 국한되었으며, 본 연구에서는 학습 과학의 주요 원칙에 기반한 8개의 교육적 차원으로 구성된 통합 평가 분류체계를 제안합니다. MRBench라는 새로운 평가 벤치마크를 출시하여 LLM과 인간 튜터의 192 개 대화 및 1,596 개 응답을 포함하였습니다.

- **Technical Details**: 통합 평가 분류체계는 학생의 실수 수정에 관련된 8개의 차원으로 구성되어 있습니다: (1) 실수 식별, (2) 실수 위치, (3) 답변 공개, (4) 가이드 제공, (5) 행동 가능성, (6) 일관성, (7) 튜터의 어조, (8) 인간유사성. 각 차원은 교육적 가치를 평가하는 데 강하게 연계되어 있으며, 이전 연구에서 사용된 다양한 분류체계를 통합합니다. MRBench의 각 인스턴스는 학생의 실수 또는 혼란이 발생할 때까지의 부분적인 대화를 포함하고 있으며, LLM 기반 튜터가 생성한 응답을 평가하기 위한 인간 및 LLM 기반 평가가 진행되었습니다.

- **Performance Highlights**: 최신 LLM 모델들이 효과적인 질문-응답 시스템으로 작용하지만, 일부 모델이 학생 튜터로서의 능력은 제한적임을 발견했습니다. 본 연구는 AI 튜터의 교육적 능력을 평가하기 위한 새로운 기준을 제공하여 향후 연구의 발전을 도울 것으로 기대됩니다. 또한, LLM 평가의 신뢰성을 분석하고 인간의 판단과의 상관관계를 탐구하여 그 유용성을 제시합니다.



### Text Generation Models for Luxembourgish with Limited Data: A Balanced Multilingual Strategy (https://arxiv.org/abs/2412.09415)
Comments:
          Accepted at VarDial 2025

- **What's New**: 이 논문은 희소한 자원을 가진 언어 모델 개발의 도전 과제에 대해 다루고 있으며, 특히 룩셈부르크어에 초점을 맞추고 있습니다. 룩셈부르크어는 다국어 맥락에서 디지털 데이터 부족 문제에 직면해 있습니다. T5 아키텍처를 바탕으로 한 새로운 텍스트 생성 모델을 제안하며, 독일어 및 프랑스어 데이터와 결합하여 룩셈부르크어 모델의 성능 향상을 기대합니다.

- **Technical Details**: 모델 훈련에는 룩셈부르크어, 독일어, 프랑스어가 사용되며, 이를 통해 다국어 간 전이 학습(capacity)을 개선하고자 합니다. LuxGen이라는 텍스트 생성 벤치마크가 도입되며, 이 벤치마크는 룩셈부르크어를 위한 최초의 데이터셋입니다. 이 연구에서는 몬로링구얼(단일 언어) 및 멀티링구얼(다국어) 훈련 접근법이 룩셈부르크어 생성에 어떤 이점을 가져오는지 평가합니다.

- **Performance Highlights**: 룩셈부르크어에 대한 기존의 다국어 모델들이 많은 성공을 거두었지만, 여전히 더 많은 데이터와 최적화된 접근법이 필요합니다. LuxemBERT 및 LuxGPT와 같은 기존 모델은 중간 정도의 성능을 보였지만, 제안된 모델들은 독일어 및 프랑스어 데이터를 통합함으로써 더 나은 성능을 발휘할 것으로 예상됩니다. 연구 결과는 소수 자원을 가진 언어에서 효율적인 모델 개발에 유용한 통찰력을 제공할 것입니다.



### Neural Text Normalization for Luxembourgish using Real-Life Variation Data (https://arxiv.org/abs/2412.09383)
Comments:
          Accepted at VarDial 2025

- **What's New**: 이번 연구에서는 룩셈부르크어의 비표준 철자 변형을 정규화하는 첫 번째 시퀀스-투-시퀀스(normalization) 모델을 제안합니다. ByT5 및 mT5 아키텍처를 사용하며, 실제 텍스트 변형 데이터로부터 훈련 데이터를 확보하여 모델의 성능을 극대화하고자 했습니다. 연구는 룩셈부르크어의 특성과 변형을 반영한 정교한 평가 방식을 적용하여 모델의 강점과 약점을 분석합니다.

- **Technical Details**:  본 연구는 룩셈부르크어의 정규화를 위해 실생활에서 수집된 단어 기반 변형 데이터를 활용하여 훈련 데이터를 생성합니다. AI 모델인 ByT5와 mT5는 훈련되어 다양한 비표준 및 표준 텍스트 변형의 정규화 작업을 수행합니다. 우리는 모델 성능을 평가하기 위해 정량적 지표와 함께 언어적 맥락에 따른 질적 평가를 병행합니다.

- **Performance Highlights**: 제안된 시퀀스 모델은 실제 변형 데이터를 기반으로 하여 룩셈부르크어 정규화에 매우 효과적으로 작용함을 보여주었습니다. 텍스트 정규화를 통해 NLP의 후속 작업, 예를 들어 POS 태깅과 NER의 성능을 개선할 수 있습니다. 이 연구는 룩셈부르크어의 정규화에 있어 중요한 이정표가 될 것이며, 관련 분야의 다른 연구자들에게도 기초 자료로 활용될 수 있을 것입니다.



### Word Sense Linking: Disambiguating Outside the Sandbox (https://arxiv.org/abs/2412.09370)
- **What's New**: 본 연구에서는 Word Sense Disambiguation (WSD)의 새로운 작업인 Word Sense Linking (WSL)을 소개합니다. WSD는 주어진 맥락에서 단어의 적절한 의미를 선택하는 작업으로, 최근에는 성능이 향상되었음에도 불구하고 하위 응용 프로그램을 찾는 데 어려움을 겪고 있습니다. WSL은 입력 텍스트와 참조 의미 목록이 주어진 경우, 의미를 명확하게 구별하고 연결하는 새로운 접근 방식을 제공합니다.

- **Technical Details**: WSL의 주요 특징은 세 가지 서브태스크로 나눌 수 있는 점입니다: Concept Detection (CD), Candidate Generation (CG), 그리고 Word Sense Disambiguation (WSD)입니다. CD는 입력 텍스트에서 구별할 구간을 찾고, CG는 각 구간에 대해 의미 후보 목록을 생성합니다. 이 과정은 최신 Transformer 기반 아키텍처를 활용하여 구현되며, WSD 시스템의 가정들을 완화하면서 평가됩니다.

- **Performance Highlights**: WSL에 적용된 모델은 실세계 환경에서 WSD 시스템을 확장할 때 직면하는 여러 문제를 드러냅니다. 기존의 WSD 시스템을 WSL로 간단히 확장할 경우 성능 저하가 발생하는 반면, 본 연구의 모델은 더 높은 견고성을 유지하며 성능에서 상당한 우위를 보입니다. 이 연구는 WSL의 평가 데이터셋과 새로운 아키텍처를 제공함으로써 이 분야의 진전을 촉진하는 것을 목표로 하고 있습니다.



### Falcon-UI: Understanding GUI Before Following User Instructions (https://arxiv.org/abs/2412.09362)
Comments:
          18 pages, 14 figures

- **What's New**: 이 논문에서는 기존 연구가 사용자 지시 사항을 따르는 능력에 촛점을 맞추는 경향이 있었던 것에 반해, GUI 맥락을 이해하는 중요성을 강조합니다. 본 연구는 Insight-UI Dataset라는 지시 없이도 GUI 탐색을 가능하게 하는 데이터셋을 도입하였으며, 이 데이터셋은 312K 도메인에서 자동으로 생성되었습니다. 또한, 새로운 GUI 에이전트 모델인 Falcon-UI를 개발하여 이 데이터셋으로 사전 훈련한 후 모바일 및 웹 GUI 작업에 맞추어 미세 조정하였습니다.

- **Technical Details**: Insight-UI Dataset는 다양한 GUI 환경에 대한 이해도를 높이기 위해 여러 플랫폼(iOS, Android, Windows, Linux)의 스크린샷을 포함하여 434K 에피소드를 구성합니다. 모델은 GUI 에이전트로서, GUI 컨텍스트를 인식하고 사용자 지시를 이해할 수 있는 두 가지 필수 능력이 필요합니다. Falcon-UI는 70억 개의 파라미터를 가지며, AITZ 데이터셋에서 720억 개의 파라미터를 가진 Qwen2VL과 비슷한 정확도를 달성합니다.

- **Performance Highlights**: Falcon-UI 모델은 Insight-UI Dataset로 초기 사전 훈련 후, 다양한 모바일 및 웹 GUI 데이터셋(AITW, AITZ 등)에 대해 향상된 성능을 보였습니다. 이에 따라 이 모델은 다양한 GUI 시나리오에 대한 적응성과 일반화 능력을 보여주며, 지시 없이도 높은 성능을 달성함으로써 GUI 컨텍스트에 대한 이해와 에이전트 성능 간의 관계를 입증합니다. 연구팀은 개발한 코드와 데이터셋을 오픈소스로 제공하여 향후 연구에 기여할 계획입니다.



### Training LayoutLM from Scratch for Efficient Named-Entity Recognition in the Insurance Domain (https://arxiv.org/abs/2412.09341)
Comments:
          Coling 2025 workshop (FinNLP)

- **What's New**: 본 연구에서는 금융 및 보험 도메인과 같은 전문 분야에 적합한 사전 학습(pre-trained) 모델의 성능을 개선하기 위한 전략들을 비교합니다. 특히, 맞춤형 데이터셋인 Payslips를 사용하여 보험 관련 재무 문서의 네임드 엔티티 인식(named-entity recognition, NER) 문제를 해결하는 데 초점을 맞추고 있습니다. 도메인 관련 문서를 사용함으로써 성과를 크게 개선할 수 있는 가능성을 시사합니다.

- **Technical Details**: LayoutLM 모델은 텍스트를 2D 문서로 나타내는 입력 방식을 채택하여, 단어를 토큰화하고 2D 위치 임베딩을 통해 각 단어에 대한 정보를 표현합니다. 이러한 구조는 테이블 형식의 문서에서 매우 효과적이며, 페이지 내에서의 단어 위치를 인식하는 데 유리합니다. 본 연구에서는 레이어 수를 절반으로 줄여도 성능 저하 없이 효율적인 추론(inference)을 달성할 수 있는 방법론을 제안합니다.

- **Performance Highlights**: 망각점 법칙(Mask Language Modeling, MLM) 손실을 사용해 모델을 사전 훈련했으며, 기존 LayoutLM 모델 대비 적은 수의 레이어로도 경쟁력 있는 F1 점수를 기록했습니다. 결과적으로, 적은 양의 훈련 데이터로도 성능이 향상되었으며, 연구진은 데이터와 코드를 공개하여 널리 사용될 수 있도록 하였습니다. 새로운 Payslips 데이터셋의 출범은 보험 업계에서의 정보 추출 작업을 지원하는 데 중요한 기여를 할 것으로 기대됩니다.



### Benchmarking LLMs for Mimicking Child-Caregiver Language in Interaction (https://arxiv.org/abs/2412.09318)
- **What's New**: 이번 연구에서 LLMs(대규모 언어 모델)의 아동-양육자 상호작용 능력을 평가했습니다. 연구는 LLMs가 아동과 양육자의 대화에서 보이는 독특한 언어적 특징을 얼마나 잘 포착할 수 있는지를 살펴보았습니다. 발견된 바에 따르면, 최신 LLM인 Llama 3 및 GPT-4o는 단어와 발화 수준에서는 적절한 반응을 생성할 수 있지만, 아동과 양육자의 담화 패턴을 재현하는 데는 한계가 있었습니다.

- **Technical Details**: 연구진은 CHILDES 데이터셋을 활용하여 2세에서 5세 아동과 양육자의 대화를 분석했습니다. 데이터는 발화-응답 쌍으로 재구성되어 6,600개의 상호작용 쌍을 포함하게 되었습니다. 연구는 단일 회차(single-turn) 및 다중 회차(multi-turn) 테스트 방식으로 LLM의 반응을 비교하며, 제로샷(zero-shot) 및 몇 샷(few-shot) 테스트를 진행했습니다.

- **Performance Highlights**: 평가 결과, LLM들은 아동의 발화에 적절히 반응하는 데 있는 다양한 언어적 비정통성을 처리하는 데 어려움을 겪었습니다. 아동-양육자 간 대화에서의 반응은 인간과의 비교에서 구조적 지표에서 한계를 드러냈습니다. 이러한 결과는 아동과의 상호작용에 대한 LLM의 응용 가능성을 높이기 위한 포괄적인 기준 개발의 필요성을 강조하고 있습니다.



### Learning to Solve Domain-Specific Calculation Problems with Knowledge-Intensive Programs Generator (https://arxiv.org/abs/2412.09280)
Comments:
          Under review

- **What's New**: 이 논문은 도메인 특화된 수학 문제 해결을 위한 새로운 파이프라인인 KIPG(Key-Intensive Programs Generator)를 제안합니다. KIPG는 도메인 지식에 따라 지식 집약적인 프로그램을 생성하고, 이를 통해 복잡한 조건을 만족하는 계산 문제를 해결합니다. 실험을 통해 법률 도메인에서의 효과적인 계산 능력을 입증하였습니다.

- **Technical Details**: KIPG 시스템은 Key Variables를 추출하여 도메인 지식에 기반한 출력을 계산합니다. 이 시스템은 여러 모듈 구조를 포함하고, 전문 지식을 자동으로 습득하여 새로운 도메인에 적용할 수 있는 능력을 가지고 있습니다.

- **Performance Highlights**: KIPG는 기존 기법들에 비해 뛰어난 성능을 보이며, 법률 도메인과 같은 복잡한 데이터셋에서 거의 모든 경우에 대해 기존 기준선보다 높은 정확도를 기록했습니다. 이 코드는 다른 도메인으로의 일반화 가능성도 보여주어 응용 범위가 넓습니다.



### Towards Understanding the Robustness of LLM-based Evaluations under Perturbations (https://arxiv.org/abs/2412.09269)
Comments:
          Accepted at ICON 2024

- **What's New**: 본 논문에서는 BLEU 및 ROUGE와 같은 기존 평가 메트릭이 생성된 텍스트의 미세한 품질을 포착하는 데 한계가 있음을 지적하고, Google Gemini 1과 같은 Large Language Models (LLMs)를 비표준화된 메트릭의 자동 평가자로 활용할 가능성을 탐구합니다. 특히 본 연구는 LLM의 품질 평가 기능을 조사하며, 인간 평가자와의 일치를 비교하고 약화된 입력에 대한 평가의 강인성도 살펴봅니다. 이는 NLG 태스크의 평가 방법 개선에 기여할 수 있는 새로운 통찰을 제공합니다.

- **Technical Details**: NLG에서 BLEU, ROUGE와 같은 전통적인 메트릭은 참고 자료와의 토큰 겹침에 의존하므로, 다중 유효 출력이 존재하는 abstractive summarization 및 dialog 평가에 적합하지 않습니다. 본 연구는 Google Gemini를 기반으로 서로 다른 프롬프트 전략을 적용하여 요약 및 대화 평가에서 LLM의 성능을 평가하는 방법론을 제안합니다. SummEval 및 USR 데이터셋을 사용하여 LLM이 산출한 점수와 정당성을 검토하고, 각 메트릭에 대한 점수를 생성하는 네 가지 프롬프트 전략을 검토합니다.

- **Performance Highlights**: 실험 결과, LLM 평가자와 인간 평가자 간의 일치는 제한적이며 LLM의 강인성이 떨어진다는 점을 발견했습니다. NLG 태스크에 필요한 주관적 메트릭에서 LLM을 신뢰할 수 있는 평가자로 사용하기 위해서는 상당한 개선이 요구됩니다. 연구 결과는 LLM 기반 평가자가 인간 평가자에 비해 더 일관되고 효율적인 평가를 제공할 수 있는 잠재력이 있으나, 실질적인 응용을 위해서는 더욱 발전이 필요함을 시사합니다.



### First Train to Generate, then Generate to Train: UnitedSynT5 for Few-Shot NLI (https://arxiv.org/abs/2412.09263)
Comments:
          14 pages

- **What's New**: 이번 연구에서는 자연어 추론(NLI) 모델을 개선하기 위해 합성 데이터 증강(synthetic data augmentation) 방식을 도입한 UnitedSynT5라는 새로운 접근법을 제안합니다. 기존의 Entailment Few-Shot Learning (EFL) 모델을 기반으로 하여 T5 기반 생성기를 활용해 추가적인 전제-가설 쌍을 생성했습니다. 이렇게 생성된 데이터는 엄격하게 정리되어 학습 데이터에 통합되며, 이를 통해 모델의 일반화 능력을 향상시킬 수 있습니다.

- **Technical Details**: UnitedSynT5는 EFL 프레임워크를 활용하여, T5 생성기를 통해 합성된 데이터 세트를 사용합니다. 전제-가설 쌍은 의미적 일관성을 유지하며 데이터 세트의 다양성과 복잡성을增强시키기 위해 필터링 과정을 거칩니다. 이 기술은 GTR-T5-XL 모델에서 확장된 데이터 세트로 훈련되어, NLI 모델의 성능을 획기적으로 향상시키는 결과를 가져왔습니다.

- **Performance Highlights**: 이 연구 결과는 SNLI 데이터 세트에서 94.7%의 새로운 기준을 달성했으며, E-SNLI와 MultiNLI 데이터 세트에서도 각각 94.01%와 92.57%의 정확도를 기록했습니다. 이러한 성과는 합성 데이터 증강의 잠재력을 보여주며, 향후 자연어 이해(natural language understanding) 작업의 발전에 기여할 것으로 기대됩니다.



### Make Satire Boring Again: Reducing Stylistic Bias of Satirical Corpus by Utilizing Generative LLMs (https://arxiv.org/abs/2412.09247)
Comments:
          Accepted to BUCC2025 Workshop @COLING2025

- **What's New**: 이 연구는 풍자 탐지( satire detection )의 성능을 높이기 위해 편향 방지(debiasing) 접근법을 제안합니다. 특히, 훈련 데이터의 다양한 샘플을 생성하기 위해 Generative Large Language Models (LLMs)를 활용합니다. 이 접근법은 터키어와 영어에서 자가 교차 검증(cross-validation) 방식으로 평가되었으며, 결과적으로 모델의 강건성과 일반화 능력을 강화했습니다.

- **Technical Details**: 풍자 탐지에서 발생하는 스타일 편향 문제를 해결하기 위해, 데이터 중심의 접근법에 의존하여 Generative LLMs를 활용하는 방안을 제안합니다. 제안된 방법은 편향된 말뭉치에서 생성된 풍자적 텍스트를 통해 더 자연스러운 스타일로 변환하여 비풍자적 말뭉치와 유사하게 만듭니다. 이 연구의 핵심은 터키어 풍자 뉴스 데이터셋을 수집하고, 이에 대한 상세한 인간 주석(human annotations)을 제공합니다.

- **Performance Highlights**: 실험 결과, 제안한 편향 방지 방법이 터키어와 영어의 풍자 및 아이러니 탐지 업무에서 모델의 성능을 향상시키는 것으로 나타났습니다. 그러나 Llama-3.1 같은 인과 언어 모델에 미치는 영향은 제한적이라는 점도 발견되었습니다. 이 연구는 또한 풍자 뉴스 탐지를 위한 터키어 데이터셋의 효용성을 검토하였으며, 분류(classification), 편향 방지 및 설명 가능성의 사례 연구를 포함합니다.



### CleanComedy: Creating Friendly Humor through Generative Techniques (https://arxiv.org/abs/2412.09203)
- **What's New**: 이 논문에서는 CleanComedy라는 새로운 데이터셋을 소개하며, 독성은 필터링된 영어와 러시아어 농담을 수집했습니다. 이 데이터셋은 무작위 농담들이 아니라, 정확한 주제와 유머 스타일을 학습하도록 준비된 것입니다. 또한, 연구자들은 LLM(대규모 언어 모델)을 활용하여 새로운 유머 샘플을 생성하고, 이를 인간 평가자를 통해 측정했습니다.

- **Technical Details**: 연구에서는 유머 파일터링 방법론에 대해 설명합니다. 데이터셋 수집은 영어와 러시아어를 포함하며, 매력적인 유머를 보존하기 위해 중복 및 유독한 데이터를 제거했습니다. 초기 전처리에서는 라틴 문자와 구두점 외의 모든 기호를 포함한 사례들을 제거하고, 결과적으로 50자에서 150자사이의 예시만 유지하여 데이터를 정제했습니다.

- **Performance Highlights**: 결과적으로 연구자들은 두 단계 세부 조정(Supervised Fine-Tuning and Alignment)을 통해 легкий 유머 모델이 생성될 수 있음을 보여줍니다. 이 연구는 코믹한 콘텐츠 생성에서 책임성과 효과성을 높일 수 있는 가능성을 시사합니다. 그러나 생성 유머는 여전히 열린 연구 문제로 남아 있으며, 기존 모델과의 비교를 통해 그 성능을 검토하였습니다.



### ReFF: Reinforcing Format Faithfulness in Language Models across Varied Tasks (https://arxiv.org/abs/2412.09173)
Comments:
          Accepted to AAAI 2025

- **What's New**: 이 논문에서는 FormatBench라는 새로운 벤치마크를 제안하여 Large Language Models (LLMs)의 format faithfulness(형식 충실도)를 평가하는 데 중점을 두고 있습니다. 이전의 형식 관련 벤치마크들보다 다양한 응용 장면과 상호작용 스타일, 형식 유형을 포함하고 있으며, 각 작업에 형식 검사기 프로그램이 부착되어 있습니다. 또한, ReFF(형식 충실도 강화)를 통해 LLM들이 지시된 형식의 출력을 효과적으로 생성할 수 있도록 하는 방법을 제시하고 있습니다.

- **Technical Details**: FormatBench는 전통적인 NLP 작업, 창의적 작업, 자율 에이전시 작업 등 다양한 작업을 포함하여 형식 충실도를 평가합니다. 각 작업에 대한 형식 검사를 위해, 비정형 데이터를 활용한 reinforcement learning(RL) 기반의 ReFF 방법론을 도입하여 형식 충실도를 크게 개선할 수 있음을 보여줍니다. 예를 들어, ReFF를 적용한 경우 LLaMA3 모델의 형식 충실도는 21.6%에서 95.0%로 향상되었습니다.

- **Performance Highlights**: FormatBench를 통해 수행된 실험 결과는 최신 LLM들이 형식 요구 사항이 간단한 작업에서도 상당한 도전을 받고 있음을 보여줍니다. ReFF는 레이블이 없는 데이터로도 형식 충실도와 일반 품질 모두를 동시에 개선할 수 있는 가능성을 보여줍니다. 레이블이 있는 훈련 데이터를 결합하면 형식 충실도는 21.6%에서 75.5%로, 일반 품질은 F1 점수가 47.3에서 61.6으로 상향 조정되었습니다.



### When Text Embedding Meets Large Language Model: A Comprehensive Survey (https://arxiv.org/abs/2412.09165)
Comments:
          Work in progress

- **What's New**: 이 논문은 자연어 처리(NLP)에서의 텍스트 임베딩의 역할을 심층적으로 조사하며, LLM(대형 언어 모델)이 텍스트 임베딩 기법과 어떻게 상호 작용하는지를 세 가지 주요 주제로 나누어 설명합니다. 특히 LLM이 기존의 텍스트 임베딩 방법을 보강하거나 자체적으로 텍스트 임베딩 생성에 어떻게 활용되는지를 다루고 있습니다. 이 연구는 다양한 연구 및 응용 분야의 기여를 조직적으로 정리하고, LLM과 PLM(사전 학습 언어 모델) 시대의 남아 있는 도전 과제를 강조합니다.

- **Technical Details**: 텍스트 임베딩 학습은 자연어 처리를 위한 기초 작업으로, 주어진 텍스트에서 유용한 특성을 추출하는 것을 목표로 합니다. LLM은 탐색, 텍스트 분류, 기계 번역 등 다양한 다운스트림 작업에서 탁월한 일반화 및 전이 능력을 보여줍니다. 본 논문에서는 LLM이 데이터 주석 및 모델 기초로서 높은 질의 텍스트 표현을 생성하는 두 가지 방법으로 기존 임베딩 학습 환경을 변화시켰음을 강조합니다.

- **Performance Highlights**: 텍스트 임베딩 분야에서 최근 LLM의 등장은 진화의 새로운 방향을 제시하였으며, 특히 정보 추출, 유사성 측정 등 여러 분야에서 장기적인 기대 효과를 생성하고 있습니다. 다양한 전통적 및 새롭게 등장한 다운스트림 작업들에 대해 LLM이 어떻게 기여할 수 있음을 보여주며, 기존 방법들의 한계와 LLM으로 인해 새롭게 발생한 도전 과제를 함께 다루고 있습니다. 이 연구는 앞으로의 텍스트 임베딩 발전 방향에 대해 이론적 그리고 실천적 기회를 탐구하며 지속적인 발전을 장려합니다.



### PolyIPA -- Multilingual Phoneme-to-Grapheme Conversion Mod (https://arxiv.org/abs/2412.09102)
- **What's New**: 이 논문은 다국어 음소(phoneme)에서 그래프음(grapheme)으로 변환하는 새로운 모델인 PolyIPA를 소개합니다. 이 모델은 다국어 이름의 음역(transliteration), 이름학 연구(onamastic research), 정보 검색을 위한 목적으로 설계되었습니다. PolyIPA는 데이터 증가(data augmentation)를 위해 식별된 두 개의 보조 모델, IPA2vec와 similarIPA를 활용하며, 여러 언어와 문자 시스템을 테스트하여 매우 낮은 Character Error Rate와 높은 BLEU 점수를 달성했습니다.

- **Technical Details**: PolyIPA의 구현은 음소-그래프음(P2G) 변환의 역방향 문제에 초점을 맞추고 있습니다. 이 모델은 기존의 G2P 데이터셋과 모델을 활용하여 두 개의 보조 모델을 개발하였으며, IPA2vec는 언어 간의 유사 발음을 찾고, similarIPA는 다양한 음소 기호 표기에 대처합니다. 최종 데이터셋은 약 1964만의 고유한 언어-그래프음-음소 쌍으로 구성되어 있으며, 기계 학습(neural network)을 통해 이 데이터셋이 강화되었습니다.

- **Performance Highlights**: PolyIPA 모델은 테스트 세트에서 평균 Character Error Rate 0.055와 BLEU 점수 0.914를 달성하였으며, 특히 얕은 철자법(shallow orthography)을 가진 언어에서 뛰어난 성능을 보였습니다. 추가적으로, 빔 탐색(beam search)을 구현하여 상위 3개의 후보를 활용할 때, 효과적인 오류율을 52.7% 감소시킬 수 있었습니다. 이러한 결과는 다국어 음역 및 정보 검색 응용 프로그램에서의 PolyIPA 모델의 효과를 보여줍니다.



### Filter-then-Generate: Large Language Models with Structure-Text Adapter for Knowledge Graph Completion (https://arxiv.org/abs/2412.09094)
Comments:
          COLING 2025 Main Conference

- **What's New**: 이번 연구에서는 Knowledge Graph Completion (KGC) 문제에 대한 새로운 접근 방식으로 instruction-tuning 기반의 FtG 방법을 제안합니다. 기존의 대형 언어 모델(LLMs)이 KGC 작업에서 낮은 성능을 보인 문제를 해결하기 위해, 필터-생성(filter-then-generate) 패러다임을 도입하여 후보 엔티티를 효과적으로 축소합니다. 또한, 그래프 구조 정보를 LLMs에 통합하기 위해 구조 인식 프롬프트를 설계했습니다.

- **Technical Details**: FtG 방법은 KGC 작업을 다중 선택 질문 형식으로 변환하여 LLMs가 타겟 엔티티를 생성하도록 유도하는 방식입니다. 이 방법은 먼저 전통적인 KGC 기법을 이용하여 불가능한 엔티티 후보를 제거한 후, 남은 상위 k 개의 후보로부터 최종 엔티티를 선택하도록 합니다. 그 밖에도, 구조와 텍스트 정보를 맵핑하는 가벼운 구조-텍스트 어댑터를 도입하여 LLMs가 KG의 복잡한 구조를 이해하도록 돕습니다.

- **Performance Highlights**: 실험 결과, 제안된 FtG 방법이 기존의 최첨단 KGC 방법들과 비교하여 상당한 성능 향상을 보여주었습니다. 특히, 여러 표준 벤치마크에서 KGC 작업 수행 시, FtG가 기존의 구조 기반 방법보다 더 나은 성과를 기록했습니다. 이러한 결과는 FtG 방법이 기존 KGC 접근 방식을 개선하는 데 효과적이며, 플러그 앤 플레이(plug-and-play) 방식으로도 활용 가능함을 시사합니다.



### Evaluating Pixel Language Models on Non-Standardized Languages (https://arxiv.org/abs/2412.09084)
Comments:
          Accepted at COLING 2025

- **What's New**: 이 논문은 표준 언어에서 방언으로의 전이 학습을 위한 픽셀 기반 모델의 가능성을 탐구하고 있습니다. 연구진은 텍스트를 이미지로 변환하여 개발된 모델이 방언 데이터에서 발생할 수 있는 out-of-vocabulary 단어 문제를 해결하는 데 유용하다고 보고하였습니다. 독일어를 사례로 하여 픽셀 기반 모델과 토큰 기반 모델의 성능을 비교하였으며, 특정 작업에서 픽셀 기반 모델이 토큰 기반 모델보다 뛰어난 결과를 거두었다는 점이 주목할 만합니다.

- **Technical Details**: 픽셀 기반 모델은 텍스트를 RGB 이미지로 변환하고 이를 16x16 픽셀의 패치로 나누어 처리합니다. 이 모델은 12층 트랜스포머 아키텍처를 사용하며, 전체 8600만 개의 파라미터를 가지고 있습니다. 주요 특징 중 하나는 패치의 연속적인 표현을 활용하여 여러 언어 및 스크립트를 처리할 수 있다는 것입니다. 또한 픽셀 기반 모델은 전통적인 토큰화 방식보다 큰 어휘를 필요로 하지 않아 계산 비용을 줄일 수 있습니다.

- **Performance Highlights**: 실험 결과는 픽셀 기반 모델이 방언 평가에서 포지션 태깅, 의존 구문 분석, 의도 탐지 작업에서 최대 26% 높은 성능을 보임을 보여주었습니다. 그러나 주제 분류에서는 픽셀 기반 모델이 토큰 기반 모델에 비해 성능이 부족했습니다. 이 연구는 다양한 방언 데이터를 처리할 수 있는 픽셀 기반 모델의 가능성을 강조하며, 추가 연구가 필요하다는 결론을 내렸습니다.



### Forest-of-Thought: Scaling Test-Time Compute for Enhancing LLM Reasoning (https://arxiv.org/abs/2412.09078)
Comments:
          Preprint

- **What's New**: 이 논문에서는 복잡한 논리 문제를 해결하기 위한 Forest-of-Thought (FoT)라는 새로운 추론 프레임워크를 제안합니다. FoT는 여러 개의 추론 트리를 통합하여 집단적 의사 결정을 활용하며, 스파스 활성화 전략을 통해 가장 관련성이 높은 경로를 선택함으로써 효율성과 정확성을 향상시킵니다. 또한 실시간 오류 수정 및 과거 학습을 통해 동적 자기 수정 전략을 도입하여 복잡한 문제 해결에서의 성능을 크게 향상시킵니다.

- **Technical Details**: FoT 프레임워크는 여러 개의 추론 트리를 활용하여 모델의 추론 능력을 확장시키며, 각 트리의 가장 관련성 높은 경로를 선택하기 위해 스파스 활성화 전략을 채택합니다. 이와 함께 합의 기반의 의사 결정 전략을 포함하여, 모델이 필요할 때만 추론 과정을 계속 진행하도록 최적화합니다. 이러한 방법은 LLM이 복잡한 작업을 보다 정밀하고 효율적으로 해결할 수 있도록 지원합니다.

- **Performance Highlights**: 실험 결과는 FoT 프레임워크가 제공하는 다양한 전략과 결합하여 LLM의 추론 성능을 크게 향상시키는 데 기여함을 보여줍니다. 연구에 따르면 FoT는 특히 복잡한 과제를 해결하는 데 있어 더욱 뛰어난 정확성과 효율성을 제공합니다. 이를 통해 LLM은 일반적인 문제 해결뿐 아니라 고급 추론 요구사항에도 효과적으로 대응할 수 있는 가능성을 보여줍니다.



### Dial-In LLM: Human-Aligned Dialogue Intent Clustering with LLM-in-the-loop (https://arxiv.org/abs/2412.09049)
- **What's New**: 이 논문에서는 고객 대화에서 의도를 발견하는 데 있어 기존의 텍스트 클러스터링 방법의 한계를 극복하기 위해 대규모 언어 모델(LLM)의 이해 능력을 활용합니다. 전통적인 방법들이 embedding distance에서 semantic distance로의 전환으로 인해 인간의 인식과 일치하지 않는 반면, LLM을 기반으로 한 새로운 의도 클러스터링 알고리즘이 제안됩니다. 이를 통해 의도 클러스터의 품질 평가 또한 더 정확하고 직관적으로 수행할 수 있습니다.

- **Technical Details**: 연구의 초점은 LLM을 사용한 의도 클러스터링 과정의 발전에 있습니다. 특히, fine-tuned LLM의 활용을 통해 semantic coherence를 평가하고 클러스터에 대한 명확한 레이블을 제공합니다. 게다가, iterative clustering 알고리즘을 통해 고품질 의도 클러스터를 지속적으로 발견할 수 있도록 하고 다양한 샘플링 전략에서의 성능을 입증합니다.

- **Performance Highlights**: 제안된 기법은 1,507개의 의도 클러스터를 포함하는 대규모 산업 데이터 세트에서 실험을 통해 그 효과를 입증했습니다. 이 방법은 기존 방법들에 비해 6.25%의 정량적 지표 향상과 함께 애플리케이션 성능이 12% 증가하는 결과를 가져왔습니다. 이러한 성과는 고객 지원 시스템에서 의도 분류기를 구축하는 데 있어 큰 진전을 나타냅니다.



### Multi-Task Learning with LLMs for Implicit Sentiment Analysis: Data-level and Task-level Automatic Weight Learning (https://arxiv.org/abs/2412.09046)
Comments:
          11 pages, 6 figures, and 6 tables

- **What's New**: 이번 연구에서는 새로운 다중 작업 학습(Multi-task Learning, MTL) 프레임워크인 MT-ISA를 도입하여 암묵적 감정 분석(Implicit Sentiment Analysis, ISA)을 개선합니다. MT-ISA는 대규모 언어 모델(Large Language Models, LLMs)의 생성 및 추론 기능을 활용하여 데이터 및 작업 수준의 불확실성을 처리하는 자동 MTL 접근법을 적용합니다. 이를 통해 모델들이 보다 효과적으로 감정을 인식하고 분석할 수 있도록 지원합니다.

- **Technical Details**: MT-ISA는 생성적 LLM을 사용하여 보조 작업을 구성하고, 자동 가중치 학습(Automatic Weight Learning, AWL)을 통해 데이터와 작업 간의 관계를 동적으로 학습합니다. 데이터 수준의 AWL은 데이터의 신뢰성을 높이고, 작업 수준의 AWL은 모델의 추론 능력에 따라 세밀한 가중치를 적응적으로 학습하도록 설정됩니다. 이러한 과정은 모델의 성능을 높이는 효과적인 방법으로 입증되었습니다.

- **Performance Highlights**: 대규모 실험을 통해 MT-ISA는 다양한 모델 크기에서 주 감정 예측과 보조 작업 간의 최적의 균형을 이룰 수 있음을 보여주었습니다. 제안된 방법은 ISA 분야에서 최신 기술(state-of-the-art) 결과를 달성하며, 데이터 수준 및 작업 수준의 AWL 전략을 통한 성능 개선을 확인하였습니다. 이 연구의 결과는 LLM의 활용 가능성을 더욱 확장시키는 중요한 시사점을 제공합니다.



### Mining Word Boundaries from Speech-Text Parallel Data for Cross-domain Chinese Word Segmentation (https://arxiv.org/abs/2412.09045)
Comments:
          COLING 2025

- **What's New**: 본 연구에서는 음성과 텍스트의 병렬 데이터를 활용하여, 처음으로 CWS(중국어 단어 분할)의 단어 경계를 명확히 추출하는 방법을 제안합니다. Montreal Forced Aligner(MFA) 도구를 사용하여 음성-텍스트 데이터에서 글자 수준의 정렬을 수행하고 Pause를 후보 단어 경계로 활용합니다. 불확실한 단어 경계를 필터링하기 위한 확률 기반 전략을 제시하며, 효과적인 CTT(complete-then-train) 전략도 포함되어 있습니다.

- **Technical Details**: 두 가지 주요 도메인, 즉 ZX와 AISHELL2에서 교차 도메인 CWS 실험을 진행하였으며, AISHELL2의 평가 데이터로 약 1,000 문장을 수동으로 주석 처리하였습니다. GMM-HMM 모델을 사용하여 문자 수준의 음성-텍스트 정렬을 수행하고, 수집된 Pause에 대한 상세 분석을 통해 신뢰할 수 있는 Pause를 단어 경계로 giữ기 위한 필터링 전략을 제안합니다. 음성 신호와 관련된 정확한 문자 정렬을 얻는 것이 이 아이디어를 구현하는 데 주요 도전 과제입니다.

- **Performance Highlights**: 실험 결과, ZX와 AISHELL2 모두에서 제안한 접근법이 효과적임을 입증하였습니다. 연구팀은 현재 보다 큰 데이터 세트인 Emilia 데이터를 기반으로 추가 실험을 진행 중이며, 결과는 arXiv 버전에서 추가로 보고될 예정입니다. 본 연구의 코드는 GitHub에 공개되어 있으며, 데이터의 새로운 주석도 함께 제공됩니다.



### ZigZagkv: Dynamic KV Cache Compression for Long-context Modeling based on Layer Uncertainty (https://arxiv.org/abs/2412.09036)
- **What's New**: 이 논문은 Layer Uncertainty를 활용하여 Key-Value (KV) cache를 동적으로 압축하는 새로운 방법인 \'zigzagkv\'를 제안합니다. 기존 방법들은 모든 레이어에 대해 균일한 예산 크기를 할당했지만, 각 레이어의 중요 정보 유지에 필요한 예산 크기가 다름을 발견했습니다. 이 연구는 정보 손실 최소화를 위해 각 레이어에 대해 동적으로 예산을 할당하는 방식을 도입하였습니다.

- **Technical Details**: 논문에서는 LLM의 추론 과정에서 KV cache의 크기가 입력 길이에 비례해 증가하며, 이로 인해 메모리 소비 및 Out-of-Memory 문제가 발생할 수 있음을 설명합니다. KV cache를 유지하기 위한 예산 크기는 각 레이어 및 모델에 따라 다르며, 이 연구는 이러한 차이를 분석합니다. \'zigzagkv\' 방법은 각 레이어의 불확실성(uncertainty)을 기반으로 예산 크기를 조정하여 정보 손실을 최소화합니다.

- **Performance Highlights**: 실험 결과, \'zigzagkv\' 방법이 기존의 KV cache 압축 방법들보다 우수한 성능을 보이며, 두 가지 주요 벤치마크에서 성능을 검증했습니다. 이 방법은 Full KV 추론에 비해 KV cache의 메모리 사용량을 약 20%로 줄이면서도 거의 무손실에 가까운 성능을 유지합니다. 이를 통해 \'zigzagkv\'는 KV cache 압축에서의 새로운 접근법으로 자리매김할 가능성을 보이고 있습니다.



### Dialogue Language Model with Large-Scale Persona Data Engineering (https://arxiv.org/abs/2412.09034)
- **What's New**: 이 논문에서는 오픈 도메인 대화 시스템에서 인물(Persona) 일관성을 유지하기 위한 새로운 접근 방식, PPDS(Pre-trained Persona Dialogue System)를 소개합니다. 기존 데이터 세트의 한계를 극복하기 위해, 광범위한 생성적 사전 훈련을 활용하여 인물 대화 데이터를 자동으로 생성하는 인물 추출 모델을 개발했습니다. 또한, 데이터 세트에 내재된 잘못된 인물 편향을 해결하기 위한 혁신적인 인물 증강 기법도 제시합니다.

- **Technical Details**: PPDS는 대규모 인물 대화 데이터 세트를 구축하기 위해 기존의 대화 자연어 추론(NLI) 데이터 세트를 기반으로 한 인물 추출 모델을 사용하여 설계되었습니다. 이를 통해 Reddit 댓글과 같은 대규모 대화 데이터에서 인물을 자동으로 추출하고 정확하게 요약하는 방법을 이용합니다. 이후 Transformer 기반 대규모 모델을 훈련하여 인물 일관성을 높이는 데 초점을 맞추고, PMF(전이 학습, 미세 조정, 인물 증강)의 역할을 분석합니다.

- **Performance Highlights**: 정량적 평가와 인간 평가 모두에서 제안한 모델이 비슷한 기준의 모델들에 비해 우수한 응답 품질과 인물 일관성을 나타내었다고 보고합니다. 미세 조정을 통해 인물 일관성을 유지하는 강화 학습 기법이 효과적임을 입증하며, 새로운 데이터 세트의 규모와 다양성이 기존 대화 모델의 질을 개선할 수 있도록 돕는다고 강조합니다.



### Shiksha: A Technical Domain focused Translation Dataset and Model for Indian Languages (https://arxiv.org/abs/2412.09025)
- **What's New**: 이 논문에서는 과학, 기술 및 교육 분야에 대한 번역 데이터셋의 부족을 해결하기 위해 8개 인도 언어의 영어-인디크 및 인디크-인디크 고품질 번역 쌍이 280만 개 이상 포함된 다국어 병렬 말뭉치를 생성했습니다. 이 데이터셋은 NPTEL 비디오 강의의 인간 번역 전사(Transcriptions)를 비텍스트 마이닝(Bitext Mining)하여 얻은 것입니다. 그들의 데이터셋을 사용하여 NMT 모델을 세밀 조정(Finetune)하고 평가하며, 특정 도메인 작업에서 공개된 다른 모델들을 초월했습니다.

- **Technical Details**: 논문은 8개 인도 언어로 2.8백만 개의 문장 쌍을 포함하는 병렬 텍스트 리소스를 만드는 과정을 설명합니다. NPTEL의 비디오 전사 데이터를 사용하여 문장을 NLP 라이브러리(nltk 및 indic-nlp)로 필터링했습니다. 그 후 SentAlign과 LABSE를 이용하여 고품질의 문장 쌍을 추출하며, n-m 문장 쌍 매칭을 활용하여 데이터를 더욱 세밀하게 분석했습니다.

- **Performance Highlights**: 모델 평가에서, 우리는 FLORES+ 벤치마크에서 평균 2 BLEU 이상의 성능 향상을 보여주며 일반 도메인 번역 작업에 대한 일반화 가능성을 증명했습니다. 이번 연구를 통해 우리는 인도 언어의 기술 도메인 번역 작업에서 NMT 모델의 성능이 크게 개선될 수 있음을 보여주었습니다. 또한, 생성된 데이터셋과 모델은 널리 배포되어 인도 학생들에게 교육적 혜택을 제공할 것입니다.



### Improvement in Sign Language Translation Using Text CTC Alignmen (https://arxiv.org/abs/2412.09014)
- **What's New**: 본 논문에서는 수화 번역(SLT)을 개선하기 위한 새로운 방법을 제안합니다. 이 방법은 Joint CTC (Connectionist Temporal Classification)과 Attention 메커니즘을 통합하여, 단순한 gloss 기반 감독 방식의 한계를 극복합니다. 특히 CTC와 Attention을 결합함으로써 비선형 정렬(non-monotonic alignment)을 효율적으로 처리할 수 있습니다.

- **Technical Details**: 이 연구에서는 CTC와 Attention 메커니즘을 활용하여 수화 영상과 음성 텍스트 간의 복잡한 정렬 문제를 해결합니다. CTC는 계층적 인코딩을 통해 수화의 길이 조정과 재구성을 가능하게 하고, Attention 메커니즘은 디코딩 과정에서 노출/라벨 편향을 줄이는 데 기여합니다. 특히, Transfer Learning을 통해 비전(vision)과 언어 간의 모달리티 차이를 효과적으로 연결합니다.

- **Performance Highlights**: 실험 결과는 RWTH-PHOENIX-Weather 2014 T 및 CSL-Daily라는 두 가지 널리 사용되는 벤치마크에서 발표된 결과에 기반하여, 제안된 방법이 최신 기술과 유사한 성능을 달성했음을 보여줍니다. 특히 기존의 pure-attention 기반 방법보다 뛰어난 성능을 발휘하여, 고유한 수화의 언어적 특성을 보다 잘 이해하고 반영할 수 있음을 입증합니다.



### What Makes Cryptic Crosswords Challenging for LLMs? (https://arxiv.org/abs/2412.09012)
Comments:
          COLING 2025

- **What's New**: 이 논문에서는 현대의 NLP 모델, 특히 대규모 언어 모델(LLMs)이 난해한 크립틱 크로스워드를 해결하는 데 어려움을 겪는 이유를 탐구합니다. 연구 결과, Gemma2, LLaMA3 및 ChatGPT와 같은 LLM의 성능이 여전히 인간의 성과에 비해 상당히 낮음을 보여주었으며, 이를 통해 이들 모델의 해석 가능성을 분석하고자 합니다. 또한, 새로운 데이터셋과 코드를 공개하여 연구의 재현성을 높입니다.

- **Technical Details**: 크립틱 크로스워드는 일반적인 정의나 동의어 대신 단어 놀이와 수수께끼를 포함하는 퍼즐입니다. 이 논문에서는 모델의 추론 능력을 평가하기 위해 클루의 정의 부분 추출, 단어 놀이 유형 식별, 모델의 내부 논리 설명을 통한 세 가지 보조 작업을 수행합니다. 또한, 데이터셋은 다양한 예제를 포함하고 있으며, 특정 단어 놀이 유형을 레이블링한 소규모 새 데이터셋을 추가하여 모델 학습을 지원합니다.

- **Performance Highlights**: 이 연구는 기존 LLM들이 크립틱 크로스워드 문제에서 인간 전문가에 비해 여전히 낮은 성능을 나타낸다는 기존 문헌을 확장합니다. 최근 연구에서는 LLM을 사용한 몇 가지 프롬프트 기술을 적용했지만, 그 성과는 20% 수준에 그쳤습니다. 이로 인해 모델의 문제 해결 능력을 향상시키기 위한 추가적인 연구가 필요하다는 결론을 내리고 있습니다.



### Assessing the Robustness of Retrieval-Augmented Generation Systems in K-12 Educational Question Answering with Knowledge Discrepancies (https://arxiv.org/abs/2412.08985)
Comments:
          10 pages

- **What's New**: EduKDQA는 K-12 교육 분야에서의 Retrieval-Augmented Generation(RAG) 시스템의 성능을 평가하기 위한 새로운 데이터셋으로, 3,005개의 질문을 포함하고 있습니다. 이 데이터셋은 교과서와 대형 언어 모델(LLM) 간의 지식 불일치를 시뮬레이션하기 위해 가상의 지식 업데이트를 적용하여 설계되었습니다. 연구진은 다양한 질문 유형을 통해 LLM의 맥락 활용과 지식 통합 능력을 평가했습니다.

- **Technical Details**: EduKDQA 데이터셋은 중학교 커리큘럼에 속하는 물리, 화학, 생물학, 지리, 역사 분야의 질문을 포함하고 있으며, 각 질문은 교재의 문단에서 파생됩니다. 더불어, LLM의 지식 불일치시 맥락 활용(Context Utilization)과 지식 통합(Knowledge Integration) 능력을 평가하기 위한 질문 유형을 설계했습니다. 질문 유형은 Simple Direct, Multi-hop Direct, Multi-hop Distant, Multi-hop Implicit 등으로 나뉘며, 각 유형에 따라 LLM의 성능을 세밀하게 분석하고 있습니다.

- **Performance Highlights**: 실험 결과, 기존 RAG 시스템은 지식 불일치를 겪을 경우 성능이 22-27% 감소하는 경향을 보였습니다. Lexical 기반의 전통적인 검색 방법이 높은 성능을 발휘하는 반면, LLM들은 Multi-hop Implicit 질문에서 뚜렷한 성능 격차를 나타내었고, 특히 작은 오픈 소스 모델들은 고급 모델에 비해 성능 저하가 두드러졌습니다. 이러한 결과는 지식 통합이 지식 불일치 상황에서 LLM에 큰 도전 과제가 됨을 시사합니다.



### RuleArena: A Benchmark for Rule-Guided Reasoning with LLMs in Real-World Scenarios (https://arxiv.org/abs/2412.08972)
Comments:
          Data and Codes are available at this https URL

- **What's New**: 이번 논문에서는 RuleArena라는 새로운 벤치마크를 소개합니다. 이 벤치마크는 LLMs (Large Language Models)가 복잡한 실제 규칙을 따르는 능력을 평가하기 위해 설계되었습니다. 항공사 수하물 요금, NBA 거래, 세금 규정 등 세 가지 실제 도메인을 다루며, LLM의 장기적 이해, 논리적 추론 및 수학적 계산 능력을 평가합니다.

- **Technical Details**: RuleArena는 기존의 규칙 기반 추론 벤치마크와 차별화되는 두 가지 주요 속성을 가지고 있습니다. 첫째, 첫 번째 주문 논리 표현을 넘어서는 확장성을 제공하며, 둘째, 진짜 상황에 기반하여 LLMs이 실제 응용 프로그램에서의 적절성과 신뢰성을 평가하는 데 도움을 줍니다.

- **Performance Highlights**: 실험 결과에 따르면, 현재 기술적으로 가장 진보된 LLM들, 예를 들어 GPT-4o 및 Claude-3.5 Sonnet은 복잡한 규칙 기반 추론 작업에서 대부분 실패하는 것으로 나타났습니다. LLM은 여러 규칙이나 사실을 통합하는 데 어려움을 겪으며, 종종 무관한 정보에 의해 방해받는 경향이 있습니다.



### Reasoning-Aware Query-Focused Summarization over Multi-Table Data (https://arxiv.org/abs/2412.08970)
- **What's New**: 이 논문은 다중 테이블 데이터에 대한 질의 중심 요약(query-focused summarization)을 위한 새로운 접근 방식을 제안합니다. 기존 방법의 복잡한 전처리 단계를 제거하고, 대형 언어 모델(LLM)을 활용하여 직접적으로 질의와 관련된 요약을 생성하는 방식을 채택합니다. 이를 통해 구조화된 데이터에서의 정보 추출 효율성을 높이며, 다양한 도메인에서의 일반성과 복잡한 질의 처리 능력을 강화하였습니다.

- **Technical Details**: 제안된 방법은 다중 테이블에서 주어진 질의에 기반하여 요약을 생성하기 위해 사전 훈련된 LLM을 이용합니다. 모델은 질의와 일치하는 테이블 특정 태스크를 포함한 훈련 과정을 통해 테이블 기반 추론 능력을 강화하며, 중간 직렬화 단계를 통해 정보 손실을 줄입니다. 이 모델은 질의와 함께 제공된 테이블 집합으로부터 요약을 생성하며, 조건부 가능성을 극대화하는 훈련 목표를 가지고 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 QueryTableSummarizer++는 BLEU, ROUGE 및 F1-score 등의 평가 지표에서 최신 기준선을 뛰어넘는 성능을 입증했습니다. 특히, 질의 복잡성을 처리하는 데 있어 10% 이상의 향상을 보여주었습니다. 인간 평가를 통해 생성된 요약의 품질과 실용성을 검증하여 다중 테이블 요약 작업을 위한 강력한 솔루션으로 자리잡았습니다.



### Align, Generate, Learn: A Novel Closed-Loop Framework for Cross-Lingual In-Context Learning (https://arxiv.org/abs/2412.08955)
- **What's New**: 이번 연구에서는 크로스-링구얼 인-컨텍스트 학습(XICL)을 위한 새로운 자기 감독(self-supervised) 프레임워크를 제안합니다. 이는 기존의 외부 검색기(retriever) 또는 작업 특정 미세 조정(task-specific fine-tuning)에 의존하지 않고, 언어 모델(LLM)의 생성 능력을 활용하여 내부적으로 작업 관련 예제를 선택합니다. 이 방법은 검색-생성 정렬 손실(retrieval-generation alignment loss)과 의미 일관성 손실(semantic coherence loss)이라는 두 가지 주요 목표를 도입하여 성능을 최적화합니다.

- **Technical Details**: 제안된 프레임워크는 대규모 언어 모델(LLM)의 생성 능력을 이용하여 효과적인 XICL을 구현합니다. 모델은 입력 시퀀스와 작업 관련 예제를 기반으로 출력 시퀀스의 조건부 확률 분포를 모델링하며, 이 과정에서 외부 검색기 없이 예제를 내부적으로 선택하고 활용합니다. 학습 목표는 검색과 생성을 정렬하게 하여 모델의 예제 선택과 활용 메커니즘을 최적화합니다.

- **Performance Highlights**: 광범위한 다국어 벤치마크를 통해 평가된 결과, 이 접근 방식은 기존의 방법들보다 월등히 우수한 성과를 발휘했습니다. 특히, 저자원 언어와 다양한 언어 계열에서의 일반화 가능성과 적응성을 입증하며, 새로운 기준에서도 최고 성능(state-of-the-art performance)를 기록했습니다. 인간 평가를 통해 모델이 생성한 결과물의 유창성, 관련성 및 의미론적 정확성을 확인할 수 있었습니다.



### From Text to Trajectory: Exploring Complex Constraint Representation and Decomposition in Safe Reinforcement Learning (https://arxiv.org/abs/2412.08920)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 이 논문에서는 Safe Reinforcement Learning(RL)에서 자연어로 제공된 제약을 활용하는 새로운 방식을 소개합니다. Trajectory-level Textual Constraints Translator(TTCT)를 도입하여 수동으로 설계된 비용 함수(cost function)를 대체함으로써, 제약을 더 유연하게 그리고 직관적으로 처리할 수 있게 됩니다. TTCT는 텍스트 제약을 이해하고 이를 정책(policy) 학습 신호로 활용하여, 더 낮은 위반율(violation rate)을 달성할 수 있음을 보여줍니다.

- **Technical Details**: TTCT는 RL 에이전트의 과거 상태 및 행동을 인식하는 것을 통해 텍스트 제약을 평가하는 방법을 제안합니다. 이 과정에서, 텍스트와 트라젝토리 사이의 임베딩 유사성을 최대화하고 비일치 쌍의 유사성을 최소화하는 대조 학습(contrastive learning) 접근 방식을 사용합니다. 또한, 시간적 신뢰 할당(temporal credit assignment) 방법을 도입하여 트라젝토리 내의 각 상태-행동 쌍에 대한 더 밀집된 비용 신호를 할당하여, 안전성과 학습 성능을 향상시킵니다.

- **Performance Highlights**: 실험에 따르면, TTCT 방식으로 학습된 에이전트는 기존의 비용 함수로 훈련된 에이전트에 비해 위반률이 최대 4.0배 낮아졌으며, 보상(reward) 성능은 유사한 수준을 유지했습니다. 더불어, 제약 환경이 변화하는 상황에서도 미세 조정 없이도 적응할 수 있는 제로샷(zero-shot) 능력이 있음을 보여주었습니다. 이러한 성과는 TTCT가 복잡한 다차원 제약을 처리하는 데 있어 매우 유연하고 일반화 가능한 시스템임을 입증합니다.



### Phi-4 Technical Repor (https://arxiv.org/abs/2412.08905)
- **What's New**: phi-4는 140억 개의 파라미터를 가진 언어 모델로, 데이터 품질 중심의 학습 방식을 채택하여 훈련되었습니다. 기존의 언어 모델들과 달리, phi-4는 훈련 과정 전반에 걸쳐 합성 데이터(synthetic data)를 전략적으로 통합하여 놀라운 성과를 올렸습니다. 특히, phi-4는 STEM 관련 질의응답(STEM-focused QA) 능력에서 GPT-4를 능가하며, 이는 데이터 생성(data-generation) 및 후속 훈련(post-training) 기술이 기존 방식보다 한층 발전했음을 보여줍니다.

- **Technical Details**: phi-4는 주로 합성 데이터로 훈련되며, 멀티 에이전트 프롬프팅(multi-agent prompting), 자기 수정 워크플로우(self-revision workflows), 지시 반전(instruction reversal) 등의 다양한 기법을 통해 생성됩니다. 이러한 방법들은 모델이 더 강한 추론 및 문제 해결 능력을 갖추도록 설계된 데이터셋을 만드는 데 기여합니다. 또한, 훈련 과정에서의 커리큘럼 최적화와 포스트 트레이닝(post-training) 혁신도 주요하게 다루어집니다.

- **Performance Highlights**: phi-4는 훈련 후 평가에서 새로운 데이터로 테스트했을 때 뛰어난 성과를 보였습니다. 특히 전통적인 벤치마크에서 GPT-4의 성능을 크게 초과하며, MATH(수학 대회) 및 GPQA(대학원 STEM Q&A) 등에서 두드러진 성과를 기록하였습니다. 이러한 성과는 phi-4의 훈련 방식이 과적합(overfitting) 문제를 잘 관리하고 있음을 나타내며, 새로운 테스트 데이터에서의 강력한 승인 결과가 이를 뒷받침합니다.



### AI-assisted Knowledge Discovery in Biomedical Literature to Support Decision-making in Precision Oncology (https://arxiv.org/abs/2412.08900)
Comments:
          Accepted at AMIA Annual Symposium 2024

- **What's New**: 이번 연구에서는 암 환자에게 적절한 타겟 테라피(targeted therapy)를 제공하기 위해 필수적인 분자 프로파일링(molecular profiling)과 임상 특성을 분석하는 과정에서 자연어 처리(natural language processing)의 기여 가능성을 평가하였습니다. 특히, 생의학 문헌에서 지식 발견(knowledge discovery)을 지원하는 다양한 모델들을 테스트하였습니다.

- **Technical Details**: Bidirectional Encoder Representations from Transformers(BERT) 계열의 두 가지 모델, 두 가지 대형 언어 모델(Large Language Models) 및 PubTator 3.0의 성능을 비교 분석하였습니다. 이 연구는 명명된 개체 인식(named entity recognition, NER)과 관계 추출(relation extraction, RE) 작업을 수행하는 능력을 중심으로 진행되었습니다.

- **Performance Highlights**: PubTator 3.0과 BioBERT 모델이 NER 작업에서 각각 0.93과 0.89의 최상의 F1 점수를 기록하며 가장 우수한 성능을 보였습니다. BioBERT는 RE 작업에서 다른 솔루션들을 초월하는 성능을 보였으며, 특정 사용 사례에서도 거의 모든 개체 언급(entity mentions)과 대부분의 관계를 성공적으로 인식하였습니다.



### A Graph-Based Synthetic Data Pipeline for Scaling High-Quality Reasoning Instructions (https://arxiv.org/abs/2412.08864)
- **What's New**: 이 논문에서는 고품질의 추론 데이터 생성을 위한 경제적이고 확장 가능한 프레임워크인 Graph-based Synthetic Data Pipeline (GSDP)를 제안합니다. 기존의 합성 방법들은 고비용과 확장성 부족의 문제를 겪고 있는데, 비해 GSDP는 지식 그래프에서 영감을 받아 독창적인 지식 점 관계 그래프(Knowledge Point Relationships Graph, KPRG)를 구축합니다. 이를 통해 최대 255배의 데이터 확장을 이루며, GPT-4와 유사한 품질의 데이터 생성을 가능하게 합니다.

- **Technical Details**: GSDP는 네 가지 주요 단계로 구성됩니다: 첫째, 전문 수학 모델을 사용하여 시드 데이터에서 지식 점(Knowledge Points, KPs)을 추출합니다. 둘째, KPRG를 구성하여 KPs 간의 관계를 명시적 및 암묵적 관계로 명확히 정의합니다. 셋째, KPRG에서 선택된 KPs의 조합을 입력으로 하여 새로운 문제와 해결책을 생성합니다. 마지막으로, 여러 고급 오픈소스 모델을 이용해 생성된 문제와 해결책의 품질을 평가합니다.

- **Performance Highlights**: GSDP-MATH 데이터셋은 191만 개 이상의 수학 문제 쌍을 포함하고 있으며, Mistral-7B 기반의 모델인 GSDP-7B는 MATH에서 37.7%, GSM8K에서 78.4%의 정확도를 기록합니다. GSDP는 다른 방법보다 100배 낮은 비용으로 데이터 합성을 수행하면서도 성능이 우수함을 입증했습니다. 이러한 결과는 GSDP가 수학적 추론 작업에서의 효과성을 갖추었음을 보여줍니다.



### Exploring Large Language Models on Cross-Cultural Values in Connection with Training Methodology (https://arxiv.org/abs/2412.08846)
- **What's New**: 이번 논문에서는 오픈소스 대규모 언어 모델(Large Language Model, LLM)이 다양한 국가의 문화적 가치에 대한 판단을 어떻게 수행하는지를 탐구했습니다. 특히 모델 크기, 훈련 코퍼스, 정렬(alignment) 등 훈련 방법론과의 관계를 분석하였습니다. LLM은 인간과 유사하게 사회문화적 규범을 판단하지만, 사회 체제나 진보에 대해서는 다소 차이를 보입니다. LLM의 문화적 가치 판단은 서구 문화에 편향된 경향이 있으며, 다국어 코퍼스에 대한 훈련을 통해 이를 개선할 수 있습니다.

- **Technical Details**: 연구에서는 World Value Survey(WVS) 데이터셋을 활용하여 LLM의 문화적 가치 이해도를 측정했습니다. WVS는 55개 국가에서 12개 범주에 걸쳐 총 209개 사회 가치 관련 질문을 포함하고 있습니다. 각 질문은 다중 선택형 태스크로 변환되며, LLM과 인간의 응답 선택 간의 상관관계를 측정하는 방법론이 제시됩니다. LLM에서 생성된 정답 후보에 기반하여 확률 분포를 정의하고, 평균 점수를 계산하여 인간의 응답과 비교합니다.

- **Performance Highlights**: 분석 결과, LLM은 인간과 유사한 방식으로 사회문화적 규범을 판단할 수 있으며, 더 큰 모델일수록 문화적 인식이 더 강하게 나타났습니다. 또한 작은 모델은 합성 데이터(synthetic data)를 통해 문화적 지식을 강화할 수 있는 가능성이 있습니다. LLM을 인간처럼 정렬하는 것이 인간과의 유사성을 높이는 데 기여할 수 있음을 확인했습니다. 최종적으로 LLM의 설계 방법론 개선을 위한 중요한 통찰력이 제공되었습니다.



### Large Concept Models: Language Modeling in a Sentence Representation Spac (https://arxiv.org/abs/2412.08821)
Comments:
          49 pages

- **What's New**: 이번 연구는 기존의 LLM(대규모 언어 모델) 접근 방식에서 벗어나, 고차원 의미 표현인 'concept'를 기반으로 한 새로운 아키텍처를 제안합니다. 이 모델은 언어 및 모달리티에 관계없이 고차원 아이디어나 행동을 표현하며, 입력을 토큰 수준이 아닌 개념 수준에서 처리합니다. 본 연구에서는 이 개념이 문장에 해당한다고 가정하고, SONAR(200개 언어 지원)라는 기존의 문장 임베딩 공간을 활용합니다.

- **Technical Details**: 제안된 Large Concept Model(대규모 개념 모델, LCM)은 1.6B 파라미터 모델에서 1.3T 토큰의 훈련 데이터를 사용하여 여러 세대 생성 작업을 수행합니다. 모델은 MSE 회귀, 확산 기반 생성의 변형 및 양자화된 SONAR 공간에서 운용되는 모델과 같은 다양한 접근 방식을 탐구합니다. 저자들은 LCM을 통해 추론을 수행하며, 이를 통해 새로운 시퀀스의 생성이 가능함을 보여줍니다.

- **Performance Highlights**: LCM 모델은 여러 언어에서 저코드 손실 제너레이션을 활용하여 뛰어난 제로 샷 일반화 성능을 보였으며, 동일한 크기의 기존 LLM보다 더 우수한 성능을 발휘했습니다. 연구팀은 모델의 훈련 코드를 무료로 제공하고 있으며, 이는 접근성과 연구 커뮤니티와의 공유 측면에서 중요한 발전을 나타냅니다.



### jina-clip-v2: Multilingual Multimodal Embeddings for Text and Images (https://arxiv.org/abs/2412.08802)
Comments:
          21 pages, 1-10 main paper, 10-12 refs, 12-21 benchmarks

- **What's New**: 이번 연구에서는 jina-clip-v1 모델을 기반으로 한 향상된 프레임워크인 jina-clip-v2를 제안합니다. 이 모델은 여러 언어에서 다중 작업 및 다중 단계의 대조 학습(multi-task, multi-stage contrastive learning)을 활용하여 텍스트 전용 검색 성능을 개선합니다. ML 모델의 한계로서 명시된 다국어 지원 부족 및 복잡한 비주얼 문서 이해에서의 성능 저하 문제를 해결합니다.

- **Technical Details**: jina-clip-v2 모델은 듀얼 인코더 아키텍처(dual encoder architecture)를 사용하여 텍스트와 이미지를 동일한 임베딩 공간에서 인코딩합니다. 텍스트 인코더는 사전 훈련된 Jina-XLM-RoBERTa 모델을 초기화하고, 이미지 인코더는 EVA02 계열의 ViT 모델을 선택하였습니다. 모델 학습을 위해 다양한 다국어 데이터셋을 구축하였고, 하드 네거티브(hard negative) 샘플을 포함한 훈련 방법을 채택했습니다.

- **Performance Highlights**: jina-clip-v2는 텍스트 전용 및 다중 모드 작업에서 이전 모델에 비해 성능이 크게 향상되었습니다. 이 모델은 다국어 텍스트 검색 벤치마크 및 시각적으로 풍부한 문서 검색 벤치마크에서 국가 최우수 성능에 필적하는 결과를 보여줍니다. 또한, Matryoshka Representation Learning을 활용하여 벡터 저장 비용을 절감하며, 성능 저하 없이 출력 벡터 차원을 축소 가능합니다.



### Coverage-based Fairness in Multi-document Summarization (https://arxiv.org/abs/2412.08795)
- **What's New**: 이 논문은 다중 문서 요약(multi-document summarization, MDS)에서 공정성을 측정하는 새로운 방법을 제안하고 있습니다. 제안된 공정성 척도는 이전의 Proportional Representation(비례 대표성) 방식을 개선하여, 정보의 중복성을 고려한 Equal Coverage와 코퍼스 전체의 공정성을 평가하는 Coverage Parity를 포함합니다. 이를 통해 다양한 사회적 속성을 가진 문서를 공정하게 요약할 수 있는 시스템을 평가하였습니다.

- **Technical Details**: Equal Coverage는 문서 집합에서 각 문서가 요약에 포함될 확률이 사회 속성과 독립적이어야 한다는 원칙을 기반으로 하고 있습니다. 또한 Coverage Parity는 다양한 사회 속성을 가진 문서들이 과대표현 또는 과소 대표되는 일이 없도록 평가합니다. 이러한 두 가지 공정성 척도는 LLM(Large Language Models)에서의 출력 결과를 기반으로 검증되었습니다.

- **Performance Highlights**: 실험 결과, 여러 LLM을 평가한 결과 Claude3-sonnet이 가장 공정하다는 것을 발견했습니다. 또한 대부분의 LLM은 특정 사회적 속성을 과대표현하는 경향이 있음을 알 수 있었습니다. 이러한 결과는 사용자들이 LLM으로 생성된 요약을 사용할 때 더욱 섬세한 판단을 할 수 있도록 도와줄 것입니다.



### BDA: Bangla Text Data Augmentation Framework (https://arxiv.org/abs/2412.08753)
- **What's New**: 본 논문은 Bangla 텍스트 데이터 증강(BDA) 프레임워크를 소개하여, 고급 데이터 증강 기법을 활용해 데이터 부족 문제를 해결합니다. 기존의 데이터 세트는 제한된 크기와 낮은 어휘 다양성 문제를 갖고 있기 때문에, 효과적인 증강 방법이 필요합니다. BDA 프레임워크는 미리 훈련된 모델과 규칙 기반 방법을 혼합하여 의미를 보존하면서 텍스트의 변형을 생성합니다.

- **Technical Details**: BDA 프레임워크는 다양한 증강 기법을 적용하여 원본 데이터의 의미를 유지하면서 새로운 샘플을 생성합니다. 동의어 대체(Synonym Replacement), 무작위 교환(Random Swap) 등의 기법을 채택하여 Bangla 텍스트의 품질과 다양성을 높입니다. 각 문장에서 비정지 단어를 무작위로 선택하여 동의어로 대체하는 방식이 사용되며, 이는 감정 분석 및 텍스트 분류 작업에서 효과적입니다.

- **Performance Highlights**: BDA 프레임워크는 다섯 개의 상이한 데이터 세트에서 F1 점수를 크게 향상시켰습니다. 100%의 데이터로 훈련된 모델과 유사한 성과를 50%의 훈련 데이터로 달성할 수 있었습니다. 데이터 부족 문제를 점진적으로 해결하면서 BDA를 통해 상당한 성과 개선을 이끌어냈습니다.



### In-Context Learning with Topological Information for Knowledge Graph Completion (https://arxiv.org/abs/2412.08742)
- **What's New**: 이번 연구는 Knowledge Graph Completion(KGC) 분야에서 인-context learning 기법을 활용하여 성능을 향상시키는 새로운 접근법을 제안합니다. 특히, 대형 언어 모델(LLMs)의 사전 훈련된 지식을 통해 그래프의 토폴로지 정보와 온톨로지를 통합하여 빠진 정보를 추론하는 데 중점을 두고 있습니다. 이러한 접근법은 기존의 KGC 방법들에 비해 더욱 효과적인 성과를 보여줍니다.

- **Technical Details**: 제안된 방법은 두 단계로 구성됩니다. 첫 번째 단계에서는 LLM의 도메인 이해를 이용해 Knowledge Graph에서 온톨로지를 구축하여 그래프 내 노드의 타입과 관계를 포착합니다. 두 번째 단계에서는 그래프의 구조화된 정보를 활용하여, 사라진 지식 트리플과 기존 그래프 트리플 간의 중복 노드를 이용해 후보 솔루션을 생성하며, 이 과정에서 복잡한 그래프의 토폴로지 구조를 이용합니다.

- **Performance Highlights**: 실험 결과, ILPC-small 및 ILPC-large 데이터셋에서 제안된 방법이 기존의 최첨단 기준 모델들에 비해 월등히 높은 성능을 나타내었습니다. 이러한 성과는 전이학습(transductive) 및 유도(inductive) 환경 모두에서 관찰되었으며, 추가적인 훈련 없이도 높은 효율성을 기록하였습니다.



### LatentQA: Teaching LLMs to Decode Activations Into Natural Languag (https://arxiv.org/abs/2412.08686)
Comments:
          Project page is at this https URL

- **What's New**: 이번 논문에서는 LatentQA라는 새로운 작업을 소개하여 자연어로 모델 활성화에 대한 개방형 질문에 답변할 수 있도록 합니다. 이를 통해 모델의 해석 가능성(interpretability)을 개선하고, 모델의 잠재적인 문제를 드러내는 방식으로 사용될 수 있습니다. Latent Interpretation Tuning(LIT)이라는 새로운 방법을 통해, 모델의 활성화 데이터와 질문-답변 쌍으로 디코더 LLM을 미세 조정(fine-tuning)하여 LatentQA를 수행할 수 있도록 합니다.

- **Technical Details**: Latent Interpretation Tuning(LIT)은 자연어 레이블과 쌍을 이루는 활성화 데이터셋을 기반으로 디코더를 미세 조정합니다. 이 디코더는 특정 프롬프트에 대한 활성화를 제공하고, 결과 모델 생성의 질적 속성을 예측하는 훈련을 받습니다. LatentQA 시스템은 모델의 활성화에 대한 입력을 받고, 어떤 자연어 질문에 대답하는 형태로 구성되어 있으며, 그를 통해 모델 경향(예: 고정관념 또는 스타일 선택)을 이해할 수 있도록 합니다.

- **Performance Highlights**: LIT는 이전에 연구된 latent attribute extraction 작업에서 성능을 향상시켰으며, LatentQA 외에도 해로운 지식을 유출할 수 있는 감사를 위한 기법으로도 활용됩니다. 우리는 우리 디코더가 편향을 감소시키는 데 있어 통계적으로 유의미한 성과를 보였고, 감정 생성을 제어하는 작업에서도 평균적으로 41%의 성능 향상을 달성했습니다. 미래 연구에서는 LatentQA가 다양한 형태의 데이터로 학습되며 새로운 응용 프로그램을 열 수 있는 잠재력을 가지고 있다고 제안합니다.



### Context Canvas: Enhancing Text-to-Image Diffusion Models with Knowledge Graph-Based RAG (https://arxiv.org/abs/2412.09614)
Comments:
          Project Page: this https URL

- **What's New**: 이 논문에서는 그래프 기반의 Retrieval-Augmented Generation (RAG)을 사용하여 텍스트-이미지 생성 모델(T2I)의 기능을 향상시키는 새로운 접근 방식을 소개합니다. 이 시스템은 지식 그래프에서 동적으로 세부 캐릭터 정보와 관계 데이터를 검색하여 시각적으로 정확하고 맥락이 풍부한 이미지를 생성합니다. Context Canvas는 이러한 기술을 활용하여 고충실도(context-aware) 다면적 이미지를 생성하는 첫 번째 응용 프로그램입니다.

- **Technical Details**: Context Canvas는 텍스트-이미지 확산 모델을 개선하기 위해 구조화된 지식 그래프에서 관련 텍스트 정보를 동적으로 검색합니다. 이 접근 방식은 훈련 데이터 이상의 개념을 생성하고, 더 높은 정확성, 일관성 및 해석 가능성을 제공합니다. 또한, RAG 기반의 자기 수정 메커니즘을 제안하여 다양한 맥락을 기반으로 시각적 출력을 iteratively 조정함으로써, 여러 측면에서 일관된 이미지를 생성하도록 돕습니다.

- **Performance Highlights**: 정성적 및 정량적 실험 결과, Context Canvas는 Flux, Stable Diffusion 및 DALL-E와 같은 인기 모델의 성능을 현저히 향상시킵니다. ControlNet 기능도 강화되어 보다 세밀한 이미지 편집 작업에서 효율적으로 작동합니다. 이 연구에서는 문화적으로 구체적이고 복합적인 개념을 생성하는 데 있어 기존 모델의 한계를 해결하고 정확한 시각적 내러티브를 실현하는 데 기여합니다.



### Olympus: A Universal Task Router for Computer Vision Tasks (https://arxiv.org/abs/2412.09612)
Comments:
          Technical Report

- **What's New**: Olympus는 Multimodal Large Language Models (MLLMs)를 활용하여 다양한 컴퓨터 비전 작업을 수행하도록 통합된 프레임워크를 제안합니다. 이는 이미지를 포함한 20가지 전문 작업을 처리하는 데 사용되는 특정 모듈에 대한 작업을 위임하는 방식으로 구성되어 있습니다. 라이팅(is instruction-based routing) 시스템을 통해 복잡한 워크플로우를 구현할 수 있으며, 기존의 MLLMs와 쉽게 통합되어 성능을 향상시킵니다.

- **Technical Details**: Olympus는 446.3K개 고품질 트레이닝 데이터와 49.6K개 평가 데이터를 포함하는 OlympusInstruct 및 OlympusBench라는 데이터 세트를 통해 설계되었습니다. 각 전문 작업을 위한 특정 라우팅 토큰을 만들어 사용자 명령 내에서 여러 작업을 체인화하여 수행할 수 있는 능력을 가지고 있습니다. 이 시스템은 MLLM으로서의 기능을 활용하여 외부 모듈로 작업을 위임하는 방식으로 작동합니다.

- **Performance Highlights**: 실험 결과, Olympus는 20개 개별 작업에서 평균 94.75%의 라우팅 정확성을 달성했으며, 체인 액션 시나리오에서는 91.82%의 정밀도를 기록했습니다. 이는 Olympus가 다양한 컴퓨터 비전 작업을 효과적으로 처리할 수 있는 보편적 작업 라우터로 기능할 가능성을 강조합니다. 또한 Olympus는 단일 명령 내에서 최대 5개의 작업을 해결할 수 있는 능력을 보여줍니다.



### TimeRefine: Temporal Grounding with Time Refining Video LLM (https://arxiv.org/abs/2412.09601)
- **What's New**: 이번 연구에서는 비디오 템포럴 그라운딩(Video Temporal Grounding, VTG)의 문제를 해결하기 위해 TimeRefine이라는 새로운 접근 방식을 제안합니다. 기존의 비디오 LLM(Large Language Models)은 타임스탬프를 직접 예측하는 방법에 의존했으나, TimeRefine는 초기 예측 후 오프셋(offset)을 다시 예측하는 프로세스를 통해 정확성을 향상시킵니다. 이 방법은 모델의 셀프-수정(self-correct) 기능을 강조하며, 모델이 그라운드 트루스(ground truth)에서 얼마나 멀리 떨어져 있는지에 따라 페널티를 부여하는 보조 예측 헤드(auxiliary prediction head)를 추가하여 모델의 시간 인식 능력을 강화합니다.

- **Technical Details**: TimeRefine의 핵심 원리는 타임스탬프 예측을 직접 수행하는 대신 점진적인 정제 작업으로 재구성하는 것입니다. 모델은 먼저 대략적인 타임스탬프를 예측하고, 이후 이를 기반으로 오프셋을 예측하여 최종 예측에 도달합니다. 이 과정은 여러 차례 반복되며, 모델은 이전 예측에서 발생한 오류를 스스로 수정합니다. 또한, L1 손실(L1 loss)을 사용하여 예측이 그라운드 트루스에서 얼마나 멀리 떨어져 있는지에 따라 더 많은 페널티를 부여하여 모델의 학습을 유도합니다.

- **Performance Highlights**: 실험 결과 TimeRefine를 적용한 VTimeLLM은 ActivityNet 캡션과 Charades-STA 데이터세트에서 각각 3.6% 및 5.0%의 mIoU(mean Intersection over Union) 개선을 보여주었습니다. TimeRefine는 기존의 LLM 기반 VTG 방법과 쉽게 통합할 수 있어, 다양한 모델에서 성능 향상에 기여할 수 있습니다. 이러한 결과는 비디오 LLM의 시간 인식 능력을 획기적으로 향상시킬 잠재력을 제시합니다.



### InternLM-XComposer2.5-OmniLive: A Comprehensive Multimodal System for Long-term Streaming Video and Audio Interactions (https://arxiv.org/abs/2412.09596)
Comments:
          Github Repo: this https URL

- **What's New**: 본 연구는 인간의 인지 방식을 모방하여 장기간 동안 환경과 상호작용할 수 있는 AI 시스템 개발을 목표로 하고 있습니다. 기존의 MLLMs(다중모달 대형 언어 모델)은 실시간으로 입력을 처리하고 출력을 생성하는 기능이 제한적입니다. 이러한 한계를 극복하기 위해 연구팀은 전혀 새로운 구조의 InternLM-XComposer2.5-OmniLive(IXC2.5-OL) 시스템을 제안하였으며, 이는 스트리밍 비디오 및 오디오와 실시간으로 상호작용할 수 있는 능력을 갖추고 있습니다.

- **Technical Details**: IXC2.5-OL 시스템은 스트리밍 인지 모듈, 다중모달 장기 기억 모듈 및 추론 모듈의 세 가지 주요 구성 요소로 구성되어 있습니다. 스트리밍 인지 모듈은 비디오와 오디오의 스트림을 실시간으로 처리하여 곧바로 기억에 중요한 세부정보를 저장합니다. 다중모달 장기 기억 모듈은 단기 기억 정보를 압축하여 장기 기억으로 변환함으로써 정보 검색의 효율성과 정확성을 증가시킵니다.

- **Performance Highlights**: IXC2.5-OL 시스템은 오디오 및 비디오 벤치마크에서 뛰어난 성능을 보여주었습니다. 특히 비디오 이해에서 10B 이하의 모델 중에서 최첨단 성능을 기록했으며, StreamingBench에서 73.79%의 SOTA(상태 최적화 기술) 결과를 달성했습니다. 모델 및 소스 코드는 Github을 통해 공개되어 있어 다중 모달 스트리밍 상호작용 커뮤니티의 발전에 기여할 것입니다.



### DISHONEST: Dissecting misInformation Spread using Homogeneous sOcial NEtworks and Semantic Topic classification (https://arxiv.org/abs/2412.09578)
- **What's New**: 이번 연구는 COVID-19 팬데믹에 따른 온라인 플랫폼의 잘못된 정보 확산을 분석하며, 이러한 현상이 '에코 챔버' 개념과 어떻게 연결되는지를 탐구합니다. 이 연구에서는 사용자 간의 사회적 상호작용과 트윗 콘텐츠 분석을 결합하여 이 두 차원을 연결짓는 새로운 방법론을 제안합니다. 특히, 사용자가 사회 네트워크를 통해 이동하는 속도를 측정하는 새로운 지표인 'node speed'를 개발하였습니다.

- **Technical Details**: 연구 방법론은 두 주요 경로로 구성됩니다. 첫 번째로, Twitter 사용자의 사회적 네트워크를 나타내는 그래프를 구성하고 분석합니다. 두 번째로, Top2Vec를 사용하여 사용자가 트윗하는 주제를 모델링하고, 다양한 주제에 대한 경향성을 추론합니다. 데이터 출처로는 1,235,833개의 백신 망설임 관련 트윗을 포함한 Avax 데이터셋을 활용하였습니다.

- **Performance Highlights**: 팬데믹 관련 잘못된 정보에 대한 사회적 행동과 트윗 내용 간의 상관관계를 분석하는 결과, 에코 챔버 현상에 대한 일반적인 직관을 뒷받침하는 증거가 제시되었습니다. 연구는 사회적 상호작용의 속도와 사용자 주제 다양성 간의 연결을 강조하며, 하위 커뮤니티에서도 잘못된 정보가 여전히 활성화되어 있음을 보여줍니다.



### Does Representation Matter? Exploring Intermediate Layers in Large Language Models (https://arxiv.org/abs/2412.09563)
Comments:
          Accepted to 2024 NeurIPs Workshop on Machine Learning and Compression

- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs)의 중간 표현(intermediate representations) 품질을 여러 아키텍처에서 조사했습니다. 연구 결과, 중간 계층이 최종 계층보다 다운스트림 작업에 더 유용한 표현을 제공함을 보여주었습니다. 또한, 대표성 품질을 측정하기 위해 기존의 다양한 메트릭을 적응하여 적용했습니다.

- **Technical Details**: 저자들은 LLM의 구조인 Transformer와 State Space Models(SSM)을 비교하고, 입력의 무작위성(randomness) 및 프롬프트 길이에 따라 표현 품질이 어떻게 변화하는지를 분석합니다. 연구에서 프롬프트 엔트로피(prompt entropy), 곡률(curvature), 증강 불변성(augmentation-invariance) 메트릭을 적용하여 중간 표현의 품질을 정량화합니다. 이 연구는 다양한 설정에서 메트릭의 변화를 살펴보며, Transformer와 SSM 간의 행동 차이도 밝혀냅니다.

- **Performance Highlights**: 전체적으로, 중간 계층의 표현은 LLM의 내부 메커니즘을 이해하는 데 중요한 역할을 합니다. 연구 결과는 아키텍처 최적화 및 훈련 전략을 형성하는 데 실질적인 지침을 제공하며, 중간 계층의 복잡성을 밝히는 데 기여합니다. 이러한 통찰은 LLM 표현의 효율적 활용을 위한 더 나은 아키텍처와 훈련 전략으로 이어질 수 있습니다.



### Foundational Large Language Models for Materials Research (https://arxiv.org/abs/2412.09560)
- **What's New**: 이 논문에서는 LLaMat이라는 재료 과학에 최적화된 Large Language Models (LLMs) 라이브러리를 소개합니다. 재료 과학 문헌과 결정 구조 데이터에 대한 지속적인 사전 학습을 통해 개발된 이 모델은 텍스트 및 구조적 정보 추출에 강점을 보이며, 일반적인 언어적 능력도 유지하고 있습니다. 특히 LLaMat-CIF 변형 모델은 주기율표에서 안정적인 결정 생성 예측에 있어 놀라운 성과를 보여줍니다.

- **Technical Details**: LLaMat 모델은 재료 과학 특정 자연어 처리 (NLP) 및 구조적 정보 추출에 있어 뛰어난 성능을 발휘하며, LLaMA-2에 비해 LLaMA-3가 더 나은 성능을 보임에도 불구하고 LLaMat-2는 다양한 재료 과학 작업에서 예기치 않게 향상된 도메인 특정 성능을 나타냅니다. 이는 과도한 훈련에서 오는 적응성 강직성(adaptation rigidity) 때문일 수 있습니다. 이를 통해 우리는 특히 결정 구조 생성과 관련된 과제에서 LLaMat 모델의 우수성을 확인하였습니다.

- **Performance Highlights**: LLaMat 모델은 재료 과학에서 실질적인 응용이 가능한 LLM 동반자(copilot) 개발에 효과적임을 증명합니다. 도메인 전환과 관련하여, 모델 선택, 훈련 방법론 및 도메인 특화 성능 등이 전문 과학 AI 시스템의 개발에 중요한 영향을 미칠 수 있다는 점을 강조합니다. 이러한 연구는 재료 연구의 가속화에 기여하는 LLM의 잠재력을 드러내고 있습니다.



### Audios Don't Lie: Multi-Frequency Channel Attention Mechanism for Audio Deepfake Detection (https://arxiv.org/abs/2412.09467)
- **What's New**: 이 연구는 멀티 주파수 채널 주의 메커니즘(MFCA)과 2D 이산 코사인 변환(DCT)을 기반으로 하는 오디오 딥페이크 탐지 방법을 제안합니다. 이 방법은 오디오 신호를 멜 스펙트로그램(melspectrogram)으로 처리하고, MobileNet V2를 이용해 깊은 특징을 추출하며, MFCA 모듈과 결합하여 오디오 신호의 다양한 주파수 채널에 가중치를 부여합니다. 이를 통해 오디오 신호의 세밀한 주파수 영역 특징을 효과적으로 포착하고, 딥페이크 오디오의 분류 능력을 향상 시킵니다.

- **Technical Details**: 제안된 방법은 먼저 입력된 wav 포맷의 오디오 파일에서 오디오 신호를 추출하고, 이를 멜 스펙트로그램으로 변환하여 주파수 영역 특징을 캡처합니다. 이후, MobileNet V2 모델을 활용하여 딥페이크 탐지를 위한 특징을 효율적으로 추출하며, MFCA를 통해 주파수 특징과 시간 종속성을 융합합니다. 이러한 접근법은 오디오 딥페이크 탐지에서 정확성과 견고성을 향상시키는 데 기여합니다.

- **Performance Highlights**: 실험 결과, 본 연구에서 제안한 모델은 기존의 전통적인 방법들과 비교하여 정확도(accuracy), 정밀도(precision), 재현율(recall), F1 점수 등에서 현저한 장점을 보였습니다. 특히 복잡한 오디오 환경에서 강력한 견고성과 일반화 능력을 보여주며, 오디오 딥페이크 탐지에 대한 새로운 아이디어를 제공하고 중대한 실용적 응용 가치를 갖고 있습니다.



### From Intention To Implementation: Automating Biomedical Research via LLMs (https://arxiv.org/abs/2412.09429)
- **What's New**: 이 논문에서는 BioResearcher라는 최초의 엔드 투 엔드(end-to-end) 자동화 시스템을 소개합니다. 이 시스템은 생물의학 연구의 모든 단계를 간소화하며, 특히 dry lab 실험을 포함하는 연구 과정을 자동화합니다. BioResearcher는 검색, 문헌 처리, 실험 디자인 및 프로그래밍을 위한 전문 에이전트를 통합하는 모듈식 다중 에이전트 아키텍처를 채택했습니다.

- **Technical Details**: BioResearcher는 다학제적 기술 세트를 통합하기 위해 LLMs를 기반으로 하는 모듈식 다중 에이전트 아키텍처를 사용합니다. 이 시스템은 연구 목표에 따라 관련 문헌을 조사하고 적절한 실험 프로토콜을 설계하며, 이를 구현하기 위한 프로그램을 작성하는 기능을 가지고 있습니다. 또한, 실험 프로토콜의 품질을 평가하기 위한 새로운 메트릭을 제안했습니다.

- **Performance Highlights**: BioResearcher는 8가지 이전에 해결되지 않았던 연구 목표에 대한 평균 실행 성공률 63.07%를 달성했습니다. 제안된 품질 메트릭을 기준으로 보통 에이전트 시스템보다 평균 22.0% 더 나은 성과를 기록했습니다. 이 시스템은 연구자들의 작업 부담을 줄이고 생물의학 발견을 가속화할 수 있는 상당한 잠재력을 보여줍니다.



### Imitate, Explore, and Self-Improve: A Reproduction Report on Slow-thinking Reasoning Systems (https://arxiv.org/abs/2412.09413)
Comments:
          Technical Report on Slow Thinking with LLMs: Part II

- **What's New**: 최근 슬로우-씽킹( slow-thinking) 추론 시스템인 o1이 복잡한 문제 해결에서 뛰어난 능력을 보여주고 있습니다. 이러한 시스템은 쿼리에 응답하기 전에 긴 사고 과정을 가지고 있어 더 철저하고 정확한 솔루션을 생성할 수 있습니다. 본 논문에서는 o1 유사 시스템을 구현하기 위한 '모방, 탐색 및 자기 개선' 프레임워크를 제안하고, 과거 연구를 기반으로 이러한 시스템을 재현하는 접근법을 설명합니다.

- **Technical Details**: 저자들은 o1 같은 추론 시스템을 개발하기 위해 세 가지 주요 훈련 단계를 제시합니다: 모방(imitate), 탐색(explore), 그리고 자기 개선(self-improve)입니다. 초기 단계에서는 장기적 사고(long-form thought) 데이터를 미세 조정하여 모델이 슬로우-씽킹 모드를 활성화하도록 하고, 이후 문제를 탐색하며 여러 개의 확률적 경로(rollouts)를 생성합니다. 이러한 방식을 통해 모델은 반복적으로 훈련 데이터를 개선하고, 이전의 제약을 극복하여 다양한 도메인에서 일반화할 수 있는 시스템을 목표로 합니다.

- **Performance Highlights**: 연구 결과, 제안된 접근 방식은 MATH-OAI, AIME, GPQA와 같은 여러 벤치마크에서 산업 수준의 시스템과 경쟁할 수 있는 성능을 달성했습니다. 실험적으로 3,900개의 예시를 사용하는 경우, 우리의 시스템은 일부 산업 시스템에 가까운 성능을 보여주었으며, 1,100개의 증류된 데이터만으로도 긍정적인 결과를 얻었습니다. 이를 통해 제안한 프레임워크가 실제 문제 해결에 효과적인지 검증되었습니다.



### From Bench to Bedside: A Review of Clinical Trialsin Drug Discovery and Developmen (https://arxiv.org/abs/2412.09378)
Comments:
          11 pages

- **What's New**: 이번 논문에서는 임상 시험의 다양한 단계와 각 단계의 특성 및 상관관계를 강조하여 신약 개발 과정에서의 임상 시험의 중요성을 논의합니다. 또한 임상 시험에서의 주요 도전 과제와 그 해결 방법도 제시되며, 인공지능(artificial intelligence), 빅데이터(big data), 디지털화(digitalization)와 같은 혁신 기술의 역할 역시 다룹니다. 이는 신약 개발과 임상 적용 간의 간극을 연결하는 필수적인 과정임을 강조합니다.

- **Technical Details**: 임상 시험은 일반적으로 네 가지 단계로 나뉘며, 각각 Phase I(안전성 평가), Phase II(초기 효능 평가), Phase III(대규모 검증), Phase IV(판매 후 감시)를 포함합니다. 각 단계는 신약 후보의 안전성, 효능 및 최적 사용 방식을 엄격히 검증합니다. 임상 시험의 성공 여부는 이후 단계로의 진행 및 시장 승인 여부에 직접적인 영향을 미칩니다.

- **Performance Highlights**: 임상 시험은 제약 회사 및 과학자들에게 중요한 역할을 할 뿐만 아니라, 환자의 건강 및 삶의 질에 직접적으로 영향을 미칩니다. 특히 주요 질병 치료에 있어 성공적인 임상 시험은 전 세계 수백만 환자에게 새로운 치료 옵션을 제공합니다. 기술의 발전과 함께 임상 시험의 효율성 및 데이터 품질이 향상되고 있으며, 이는 혁신적인 신약 개발에 중요한 기여를 하고 있습니다.



### Causal Graphical Models for Vision-Language Compositional Understanding (https://arxiv.org/abs/2412.09353)
- **What's New**: 최근 연구에 따르면 Vision-Language Models (VLMs)는 인간 언어의 조합적 속성을 완전히 이해하는 데 어려움을 겪고 있으며, 이는 일반적으로 이미지 캡션을 "bag of words"로 모델링하고 있기 때문입니다. 이 연구에서는 Causal Graphical Model (CGM)을 사용하여 텍스트 및 시각적 토큰 간의 의존 관계를 모델링하고, VLM 비주얼 인코더에 의해 조건화된 디코더를 훈련하는 방식을 제안합니다. 우리의 접근 방식은 조합 작업에서 VLM의 성능을 크게 향상시키는 데 기여합니다.

- **Technical Details**: 이 연구에서 제안한 방법은 전통적인 순차적(autoregressive) 예측 대신 반순차적(semi-parallel) 예측 전략을 사용하여 CGM 구조를 따릅니다. 이를 통해 디코더는 문장 내 주요 인과적 의존 관계만 학습하고, 불필요한 상관관계를 배제할 수 있습니다. 특히, 의존 파서를 통해 작성된 의존 트리를 기반으로 CGM을 구축하여 이미지 패치와 텍스트 토큰 간의 의존 관계를 설명합니다.

- **Performance Highlights**: 다양한 실험을 통해 제안된 방법이 기존의 모든 최첨단 조합 접근 방식을 상당히 능가함을 보여주며, 평가된 모든 벤치마크에서 새로운 최첨단 성과를 기록합니다. 또한, 훨씬 적은 데이터로 훈련되었음에도 불구하고 Cap 및 CapPa 방식에서도 성능 개선을 보여줍니다.



### CRVQ: Channel-relaxed Vector Quantization for Extreme Compression of LLMs (https://arxiv.org/abs/2412.09282)
Comments:
          5 figures, 4 tables

- **What's New**: 본 논문은 Channel-Relaxed Vector Quantization (CRVQ)라는 새로운 기술을 제안하여 Post-Training Quantization (PTQ) 방법의 성능을 크게 향상시킵니다. 이 기술은 중요 채널을 신중하게 선택하고 재정렬하는 동시에, 여러 코드북을 활용하여 비트 수를 최소한으로 늘리고 있습니다. CRVQ는 강력한 서브 2비트 PTQ 기준선에 비해 38.9% 성능 향상을 이끌어내며, 1비트 압축을 거의 무손실에 가깝게 달성합니다.

- **Technical Details**: CRVQ는 두 가지 주요 혁신을 통해 성과를 달성합니다: 첫째, 중요한 가중치 채널을 신중하게 선택하고 재배치하여 유사한 중요도를 가지는 채널끼리 그룹화합니다. 둘째, 몇 가지 핵심 채널을 선택하고 추가 코드북을 사용하여 이들의 표현력을 향상시킵니다. 이 접근 방식은 LLaMA2-7B 모델에서 perplexity를 38.9% 감소시키고, 제로샷 정확도를 12.9% 향상시키는 결과를 보여줍니다.

- **Performance Highlights**: CRVQ는 특히 다양한 모델 크기(125M에서 13B까지)에서 일관된 성능을 보여주며, 자원 제약이 있는 장치에서도 우수한 양자화 성능을 달성할 수 있음을 입증합니다. 기존 AQLM과 비교했을 때, CRVQ는 1비트 PTQ에서 수익성을 크게 향상시키며, 비트 너비의 추가 비용은 매우 미미합니다. 이로 인해, 이동 장치와 같은 다양한 하드웨어 플랫폼에서도 유연한 배포 옵션을 제공합니다.



### Mojito: Motion Trajectory and Intensity Control for Video Generation (https://arxiv.org/abs/2412.08948)
- **What's New**: 이 논문에서는 텍스트를 기반으로 비디오를 생성하는 새로운 확산 모델인 Mojito를 소개합니다. Mojito는 방향성 모션 제어(Directional Motion Control) 모듈과 모션 강도 조절(Motion Intensity Modulator) 모듈을 통합하여 사용자 요구에 맞는 모션 방향과 강도를 제어할 수 있는 기능을 갖추고 있습니다. 이를 통해 추가 교육 없이도 생성된 물체의 모션을 효율적으로 유도할 수 있으며, 비디오의 자연스러운 동적 흐름을 실현합니다.

- **Technical Details**: Mojito는 두 가지 혁신적인 모듈을 포함하고 있어 비디오 생성 시 모션의 방향과 강도를 정밀하게 조절할 수 있습니다. 첫 번째 모듈인 DMC는 크로스 어텐션을 활용하여 객체의 동작 경로를 조정하며, 두 번째 모듈인 MIM은 비디오에서 생성된 옵티컬 플로우 맵을 기반으로 모션의 다양한 강도를 안내합니다. 이러한 설계는 사용자 요구에 맞게 조정할 수 있는 유연성과 효율성을 제공합니다.

- **Performance Highlights**: Mojito는 실험을 통해 목표한 모션 방향과 강도를 정확히 구현하면서도 높은 계산 효율성을 달성한 것으로 나타났습니다. 기존의 최첨단 모델과 비교할 때, Mojito는 고품질 비디오 콘텐츠를 생산하며 효과적이고 효율적인 모션 제어를 제공합니다. 이 연구는 향후 모션을 강조한 비디오 생성 모델의 발전을 위한 중요한 통찰력을 제공합니다.



### MoSLD: An Extremely Parameter-Efficient Mixture-of-Shared LoRAs for Multi-Task Learning (https://arxiv.org/abs/2412.08946)
Comments:
          Accept by COLING 2025

- **What's New**: 최근 LoRA(Low-Rank Adaptation)가 대형 사전 학습 모델을 미세 조정하는 데 중요한 기술로 부상하였으나, 다중 작업 학습(multi-task learning) 시 성능이 저조한 점이 드러났습니다. 이에 비해 MoE(Mixture of Experts) 아키텍처는 이러한 문제를 자연스럽게 해결할 수 있는 방법으로 주목받고 있습니다. 그러나 MoE는 데이터 간 상호 간섭(mutual interference) 및 다양한 작업의 지식 망각(knowledge forgetting) 같은 도전 과제를 가져옵니다. 이를 해결하기 위해 본 논문에서는 MoSLD(mixture-of-shared-LoRAs)라는 모델을 제안하고, drop out 전략을 활용하여 이러한 문제를 극복합니다.

- **Technical Details**: MoSLD는 LoRA의 상위 프로젝션 매트릭스를 다양한 전문가 간에 공유하여 여러 작업 간의 일반 지식을 학습하도록 유도합니다. 기본적으로 이 모델은 LoRA의 상위 프로젝션 매트릭스(A)와 하위 프로젝션 매트릭스(B)로 구성되어 있으며, 상위 매트릭스는 서로 다른 전문가 간에 공유됩니다. Dropout 전략이 적용되어, 하위 매트릭스의 특정 특징을 유지하는 동시에 매개변수 매트릭스의 불균형 업데이트를 완화하고 과적합(parameter overfitting)을 줄여줍니다.

- **Performance Highlights**: 다양한 실험을 통해 MoSLD 모델이 단일 작업(sing-task) 및 다중 작업(multi-task) 시나리오에서 뛰어난 성능을 보여주었습니다. 특히, 본 모델은 out-of-domain 데이터에 대해 강력한 일반화(generalization) 능력을 발휘하여, 실제 환경에서의 적용 가능성을 높였습니다. 이러한 결과는 MoSLD가 다양한 작업 간의 지식 전이 및 데이터 간 불균형 문제를 효과적으로 해결할 수 있는 잠재력을 가지고 있음을 입증합니다.



### Multi-Scale Heterogeneous Text-Attributed Graph Datasets From Diverse Domains (https://arxiv.org/abs/2412.08937)
- **What's New**: 이 논문에서는 다양한 도메인과 크기에 걸쳐 있는 Heterogeneous Text-Attributed Graphs (HTAGs)를 위한 새로운 벤치마크 데이터셋을 소개합니다. 현재의 연구는 주로 동질적인 그래프에만 집중하고 있어 복잡한 HTAGs에 대한 이해가 부족했습니다. 새로운 데이터셋은 영화, 커뮤니티 Q&A, 학술, 문학 및 특허 네트워크를 포함하여, 머신러닝 모델의 실용적이고 재현 가능한 평가를 목적으로 구성되었습니다.

- **Technical Details**: HTAG 데이터셋은 여러 크기로 제공되며, 작은 그래프(24K 노드, 104K 엣지)부터 큰 그래프(5.6M 노드, 29.8M 엣지)로 구성되어 있습니다. 각 데이터셋은 시간 기반으로 분할되어 좀 더 현실적이고 의미 있는 평가를 제공하며, 모든 소스 데이터, 데이터셋 구축 코드, 처리된 HTAGs, 데이터 로더, 벤치마크 코드, 평가 설정이 GitHub와 Hugging Face를 통해 공개되었습니다. 이 연구는 HTAGs에 대한 학습에서 노드 속성과 헤테로지니어스 그래프의 토폴로지를 효과적으로 통합하는 데 중점을 둡니다.

- **Performance Highlights**: 논문에서는 다양한 Graph Neural Networks(GNNs)를 사용하여 이러한 데이터셋에서 벤치마크 실험을 수행하였습니다. 이 데이터셋을 통해 더 나은 모델 성능을 평가할 수 있으며, 개방형 코드 제공으로 연구자들이 더 크고 복잡한 HTAG 데이터셋을 구축할 수 있도록 지원합니다. 이는 HTAGs의 연구 및 발전에 기여할 뿐만 아니라 다양한 응용 프로그램을 탐구하는 데 필요한 도구를 제공합니다.



### Euclid: Supercharging Multimodal LLMs with Synthetic High-Fidelity Visual Descriptions (https://arxiv.org/abs/2412.08737)
Comments:
          33 pages, 22 figures, 5 tables, 7 algorithms

- **What's New**: 최근 멀티모달 대형 언어 모델(MLLMs)은 비약적인 발전을 이루었지만, 여전히 저수준 시각 인식(LLVP)에서는 어려움을 겪고 있습니다. 본 논문에서는 'Geoperception'이라는 벤치마크를 소개하여 MLLM의 2D 기하학적 정보 전사 능력을 평가합니다. 이 벤치마크를 통해 기존 MLLM의 한계를 드러내고, 기하학적 작업 성능을 향상시키기 위한 전략을 연구하게 됩니다.

- **Technical Details**: Geoperception 벤치마크에서는 MLLM들이 이미지에서 기하학적 세부사항을 정확하게 설명하는 능력을 평가합니다. 연구 결과, 특정 모델 아키텍처, 훈련 기법, 데이터 전략이 기하학적 작업을 향상시키는 데 도움이 된다는 것을 발견했습니다. 특히, 데이터 커리큘럼(data curriculum)을 통해 모델은 처음부터 학습하지 못하는 어려운 기하학적 이해 작업을 학습할 수 있음을 보여줍니다.

- **Performance Highlights**: Euclid이라는 모델 계열이 강력한 저수준 기하학적 인식을 위해 최적화되었습니다. Euclid는 순전히 합성된 멀티모달 데이터로 훈련되었음에도 불구하고, 새로운 기하학적 형태에 대한 일반화 능력이 뛰어난 성능을 보여줍니다. 예를 들어, Euclid는 Geoperception 벤치마크의 특정 작업에서 최고의 비공식 모델인 Gemini-1.5-Pro보다 최대 58.56% 향상된 성능을 기록하고, 전체 작업 평균으로는 10.65% 더 나은 성과를 보입니다.



### Enhancing Code-Switching ASR Leveraging Non-Peaky CTC Loss and Deep Language Posterior Injection (https://arxiv.org/abs/2412.08651)
Comments:
          SLT 2024

- **What's New**: 이 논문은 코드 스위칭(code-switching) 문제를 해결하기 위한 새로운 접근 방식을 제안합니다. 특히, 다국어 사용자가 대화 중 언어를 번갈아 사용하는 상황에서 발생하는 음향적(acoustic) 및 의미적(semantic) 혼란을 극복하기 위한 방법론입니다. 이 연구는 언어 식별(language identification) 정보와 언어 경계 정렬(loss) 기법을 활용하여 음성 인식 성능을 개선하는 데 중점을 두고 있습니다.

- **Technical Details**: 제안된 접근 방식에서는 인코더의 여러 중간 레이어에 언어 식별 정보를 통합하여 출력 임베딩(embeddings)에 더 많은 언어 정보를 강화하는 데 초점을 맞추었습니다. 또한, 언어 경계 정렬 손실(language boundary alignment loss)을 통해 ASR 모듈이 내부 언어 포스터리어(internal language posteriors)에 대한 지식을 더 효과적으로 활용하도록 하였습니다. 마지막으로, 공유 인코더와 언어별 인코더 간의 깊은 상호작용(deep interaction)을 촉진하기 위해 언어 포스터리어 사용 가능성을 탐구하였습니다.

- **Performance Highlights**: SEAME 코퍼스(SEAME corpus)에서 시행한 종합적인 실험을 통해 제안된 방법이 이전의 disentangle based mixture-of-experts (D-MoE) 방법보다 우수하다는 것을 입증하였습니다. 이 연구는 인코더의 언어 감수성을 더욱 향상시키는 방향으로 나아가고 있으며, 최종적으로는 다양한 언어 간의 인식 능력을 높이는 데 기여하고자 합니다.



New uploads on arXiv(cs.IR)

### SPRec: Leveraging Self-Play to Debias Preference Alignment for Large Language Model-based Recommendations (https://arxiv.org/abs/2412.09243)
- **What's New**: 이번 논문에서는 추천 시스템에서의 대형 언어 모델(LLMs)에 대한 새로운 접근법인 SPRec를 제안합니다. 이 방법은 직접 선호 최적화(Direct Preference Optimization, DPO)가 내재적편향을 초래하는 문제를 해결하고, 추가 데이터나 수동 개입 없이도 공정성을 개선할 수 있도록 설계되었습니다. SPRec는 자기 학습(self-play) 프레임워크를 사용하여 추천 품목의 초과 추천을 완화합니다.

- **Technical Details**: SPRec는 매 자기 학습 반복(iteration)에서 먼저 감독된 미세 조정(Supervised Fine-Tuning, SFT) 단계를 수행한 후 DPO 단계를 진행합니다. 이 과정에서 오프라인 상호작용 데이터는 긍정적 샘플로 취급되며, 이전 반복의 예측 출력은 부정적 샘플로 사용됩니다. 이렇게 하여 DPO 손실 함수를 모델의 로짓을 기반으로 가중을 다시 조정하여 편향된 항목을 효과적으로 억제합니다.

- **Performance Highlights**: 다양한 현실 세계의 데이터셋을 활용한 실험 결과, SPRec는 추천 정확도를 향상시키고 공정성 문제를 해결하는 데 효과적임을 보여주었습니다. 본 논문은 SPRec의 구조와 셀프 플레이 학습 과정이 어떻게 편향을 억제하고 긍정적 샘플과의 정렬을 유지하는지에 대한 실증적 증거를 제시합니다.



### MOPI-HFRS: A Multi-objective Personalized Health-aware Food Recommendation System with LLM-enhanced Interpretation (https://arxiv.org/abs/2412.08847)
- **What's New**: 이번 연구는 건강 친화적인 음식 추천 시스템에 대한 두 가지 대규모 개인화된 벤치마크를 최초로 수립하여, 사용자의 특정 건강 상태를 기반으로 한 추천의 개인화 가능성을 탐색합니다. 특히, 새로운 프레임워크인 Multi-Objective Personalized Interpretable Health-aware Food Recommendation System (MOPI-HFRS)을 개발하여, 사용자 선호도, 개인화된 건강성, 영양 다양성을 균형 있게 고려한 추천을 제공합니다. 또한, 대규모 언어 모델(LLM)을 통한 해석 기능을 통해 건강한 식단에 대한 인식을 높이려는 노력을 강조하고 있습니다.

- **Technical Details**: MOPI-HFRS는 건강 관련 정보를 훈련 과정에 동적으로 포함시키는 건강 친화적 그래프 구조 학습 모듈과, 사용자 선호도 및 개인화된 건강성, 영양 다양성을 균형 있게 최적화하는 파레토 멀티 목표 학습을 활용하여 기능합니다. 연구는 National Health and Nutrition Examination Survey (NHANES) 데이터셋을 기반으로 하여, 사용자의 의료 정보를 통합한 개인화된 추천의 새로운 가능성을 선보입니다. 또한, 생성적인 LLM을 통해 추천 과정에서 얻은 지식을 해석에 사용할 수 있도록 설계하여 사용자들이 추천 결과를 이해할 수 있도록 돕습니다.

- **Performance Highlights**: MOPI-HFRS는 설정된 벤치마크에 대한 Extensive 실험을 통해 기존 최첨단 시스템을 초과하는 성과를 나타냈습니다. 이는 다목적 추천 작업과 하위 이유 제공 작업 모두에서 이루어졌으며, 건강한 식단 선택을 지원하는 데 있어 효과적인 도구로 자리매김할 잠재력을 보여줍니다. 따라서 본 연구는 개인의 건강 상태와 식이 선호를 동시에 고려하는 맞춤형 추천 시스템에 기여하고 있으며, 건강한 식습관 인식을 향상시키는 데 큰 가능성을 제시합니다.



### Reducing Popularity Influence by Addressing Position Bias (https://arxiv.org/abs/2412.08780)
- **What's New**: 이 논문에서는 추천 시스템에서 지속적으로 발생하는 위치 편향(position bias) 문제를 다룹니다. 위치 편향의 완화 방법들이 단기적으로는 사용자 참여도(user engagement)나 순위의 관련성(rate relevance)의 눈에 띄는 향상을 가져오지 않을 수 있다는 점을 강조합니다. 저자들은 위치 편향의 감소가 아이템의 인기도(popularity)를 보다 고르게 분산시킬 수 있다는 대안을 제시하고 있습니다.

- **Technical Details**: 논문에서는 추천 시스템이 사용자 요청에 맞추어 추천 아이템을 표시하는 과정을 설명합니다. 이 과정에서 특정 아이템의 인기도는 주어진 아이템이 받은 상호작용(interactions)의 비율을 의미하며, 이러한 인기도 분포는 일반적으로 긴 꼬리 분포(long-tailed distribution)로 근사될 수 있습니다. 위치 편향이 존재하지 않는 경우와 존재하는 경우를 비교하며, 후자의 경우 데이터에는 피드백 루프(feedback loop)가 존재하여 인기도의 왜곡이 발생함을 논의합니다.

- **Performance Highlights**: 실험을 통해 위치 편향 완화가 이뤄질 경우, 사용자 참여도와 재정적 지표(financial metrics)에는 악영향을 주지 않으면서도 추천 시스템의 다양성(assortment utilization)을 눈에 띄게 향상시킬 수 있다는 것을 증명합니다. 이는 장기적으로 파트너나 콘텐츠 제공자를 더 유치할 수 있도록 도와주며, 고객과 비즈니스 모두에게 긍정적인 영향을 미칩니다.



### Foundational Large Language Models for Materials Research (https://arxiv.org/abs/2412.09560)
- **What's New**: 이 논문에서는 LLaMat이라는 재료 과학에 최적화된 Large Language Models (LLMs) 라이브러리를 소개합니다. 재료 과학 문헌과 결정 구조 데이터에 대한 지속적인 사전 학습을 통해 개발된 이 모델은 텍스트 및 구조적 정보 추출에 강점을 보이며, 일반적인 언어적 능력도 유지하고 있습니다. 특히 LLaMat-CIF 변형 모델은 주기율표에서 안정적인 결정 생성 예측에 있어 놀라운 성과를 보여줍니다.

- **Technical Details**: LLaMat 모델은 재료 과학 특정 자연어 처리 (NLP) 및 구조적 정보 추출에 있어 뛰어난 성능을 발휘하며, LLaMA-2에 비해 LLaMA-3가 더 나은 성능을 보임에도 불구하고 LLaMat-2는 다양한 재료 과학 작업에서 예기치 않게 향상된 도메인 특정 성능을 나타냅니다. 이는 과도한 훈련에서 오는 적응성 강직성(adaptation rigidity) 때문일 수 있습니다. 이를 통해 우리는 특히 결정 구조 생성과 관련된 과제에서 LLaMat 모델의 우수성을 확인하였습니다.

- **Performance Highlights**: LLaMat 모델은 재료 과학에서 실질적인 응용이 가능한 LLM 동반자(copilot) 개발에 효과적임을 증명합니다. 도메인 전환과 관련하여, 모델 선택, 훈련 방법론 및 도메인 특화 성능 등이 전문 과학 AI 시스템의 개발에 중요한 영향을 미칠 수 있다는 점을 강조합니다. 이러한 연구는 재료 연구의 가속화에 기여하는 LLM의 잠재력을 드러내고 있습니다.



### When Text Embedding Meets Large Language Model: A Comprehensive Survey (https://arxiv.org/abs/2412.09165)
Comments:
          Work in progress

- **What's New**: 이 논문은 자연어 처리(NLP)에서의 텍스트 임베딩의 역할을 심층적으로 조사하며, LLM(대형 언어 모델)이 텍스트 임베딩 기법과 어떻게 상호 작용하는지를 세 가지 주요 주제로 나누어 설명합니다. 특히 LLM이 기존의 텍스트 임베딩 방법을 보강하거나 자체적으로 텍스트 임베딩 생성에 어떻게 활용되는지를 다루고 있습니다. 이 연구는 다양한 연구 및 응용 분야의 기여를 조직적으로 정리하고, LLM과 PLM(사전 학습 언어 모델) 시대의 남아 있는 도전 과제를 강조합니다.

- **Technical Details**: 텍스트 임베딩 학습은 자연어 처리를 위한 기초 작업으로, 주어진 텍스트에서 유용한 특성을 추출하는 것을 목표로 합니다. LLM은 탐색, 텍스트 분류, 기계 번역 등 다양한 다운스트림 작업에서 탁월한 일반화 및 전이 능력을 보여줍니다. 본 논문에서는 LLM이 데이터 주석 및 모델 기초로서 높은 질의 텍스트 표현을 생성하는 두 가지 방법으로 기존 임베딩 학습 환경을 변화시켰음을 강조합니다.

- **Performance Highlights**: 텍스트 임베딩 분야에서 최근 LLM의 등장은 진화의 새로운 방향을 제시하였으며, 특히 정보 추출, 유사성 측정 등 여러 분야에서 장기적인 기대 효과를 생성하고 있습니다. 다양한 전통적 및 새롭게 등장한 다운스트림 작업들에 대해 LLM이 어떻게 기여할 수 있음을 보여주며, 기존 방법들의 한계와 LLM으로 인해 새롭게 발생한 도전 과제를 함께 다루고 있습니다. 이 연구는 앞으로의 텍스트 임베딩 발전 방향에 대해 이론적 그리고 실천적 기회를 탐구하며 지속적인 발전을 장려합니다.



### Predicting Quality of Video Gaming Experience Using Global-Scale Telemetry Data and Federated Learning (https://arxiv.org/abs/2412.08950)
Comments:
          22 pages, 11 figures, 6 tables

- **What's New**: 이번 연구에서는 FPS(Frames Per Second)가 비디오 게임 경험에 미치는 영향을 다루며, 게임에 대한 정확한 FPS 예측이 플레이어와 개발자 모두에게 이익이 된다는 점을 강조합니다. FPS 예측의 정확성을 높이기 위해 다양한 장치에서의 게임 성능 예측을 위한 연합 학습 기반 모델을 제안합니다. 이 모델은 사용자 데이터를 보호하면서 FPS 성능을 예측할 수 있는 방법으로, 각 플레이어와 게임에 고유한 학습 가능한 지식 커널(Learnable Knowledge Kernel, LKK)을 사용하여 개인화합니다.

- **Technical Details**: 연구는 FPS에 영향을 미치는 다양한 요인들을 종합적으로 분석하여 게임 성능 예측을 위한 새로운 모델을 설계합니다. 이 모델은 224개 국가 및 지역에서 수집된 100,000명의 사용자 데이터를 포함하여 835개의 다양한 비디오 게임과 관련된 76.4백만 개의 게임 프로세스를 기록하였습니다. 이러한 훈련 과정은 웨이제르스타인 거리(Wasserstein distance) 등의 지표를 이용하여 예측 성능을 평가하며, 모델의 정확도를 높이기 위해 동적 커널 적용 기법을 도입합니다.

- **Performance Highlights**: 제안된 모델은 예측된 FPS 분포와 실제 FPS 분포 간의 평균 Wasserstein 거리가 0.469에 달하며 기존의 모든 기준 방법들을 초월하는 성과를 보여주었습니다. 또한, 고유한 LKK를 통한 FPS 예측 정확도 향상을 통해 콜드 스타트 문제를 해결하였으며, 이로 인해 Wasserstein 거리가 7.57% 감소하는 효과를 가져왔습니다. 이 모델은 FPS를 보다 정확히 예측할 수 있어 게임 사용자들에게 더 나은 경험을 제공할 수 있을 것으로 기대됩니다.



### A Flexible Plug-and-Play Module for Generating Variable-Length (https://arxiv.org/abs/2412.08922)
- **What's New**: 이번 논문에서는 Nested Hash Layer (NHL)라는 새로운 모듈을 제안하여, 기존의 심층 감독 해싱(deep supervised hashing) 모델에서 해시 코드의 다양한 길이를 동시에 생성할 수 있도록 합니다. 기존 모델들은 특정 길이의 해시 코드 생성에만 집중하여 그 길이에 따른 비효율성과 효과성의 거래 관계를 해결하지 못했습니다. NHL 프레임워크는 다중 학습 목표에서 발생하는 최적화 충돌을 해결하기 위해 동적 가중치 조정 전략을 도입합니다.

- **Technical Details**: NHL은 단일 훈련 세션을 통해 다양한 길이의 해시 코드를 생성할 수 있도록 설계되었습니다. NHL의 기본 구조는 나중에 길어진 해시 코드가 짧은 해시 코드의 보조 설명으로 기능할 수 있다는 점에 기반하고 있습니다. 이를 통해 해시 코드의 길이에 따라 필요한 모델을 반복적으로 훈련하는 대신, 해시 코드 생성 과정에서 구조적 관계를 활용하여 효율성을 높일 수 있습니다.

- **Performance Highlights**: NHL은 훈련 과정을 가속화하고 다양한 심층 해싱 모델에서 우수한 검색 성능을 달성하는 것으로 나타났습니다. 실험 결과, NHL은 약 5-8배 정도의 훈련 속도 개선을 보이면서도 효과적인 검색 결과를 보장합니다. 이는 대규모 이미지 데이터베이스에서 해시 코드를 활용한 저장 및 검색 효율성을 극대화할 수 있는 중요한 발전으로 기여합니다.



### Goal-Conditioned Supervised Learning for Multi-Objective Recommendation (https://arxiv.org/abs/2412.08911)
- **What's New**: 이 논문은 Multi-Objective Goal-Conditioned Supervised Learning (MOGCSL) 프레임워크를 소개합니다. MOGCSL은 기존의 Goal-Conditioned Supervised Learning (GCSL) 방법을 여러 목표를 처리하도록 확장하며, 목표를 일차원 스칼라로부터 다차원 벡터로 재정의합니다. 이를 통해 복잡한 아키텍처와 최적화 제약 조건을 자연스럽게 제거할 수 있습니다.

- **Technical Details**: MOGCSL은 오프라인 시퀀셜 데이터로부터 여러 목표를 자동으로 달성하기 위해 설계되었습니다. 유익하지 않거나 노이즈가 있는 인스턴스를 필터링하고, '높이' 달성 가능한 목표를 선택하는 새로운 목표 선택 알고리즘을 포함합니다. 이 시스템은 상업적인 추천 시스템에서 다음 행동 예측 문제에 적용되는 데 중점을 두고 있으며, 대량의 노이즈 데이터에 강인함을 가지고 있습니다.

- **Performance Highlights**: MOGCSL은 실제 추천 데이터셋에서 시행된 광범위한 실험을 통해 높은 성능을 입증했습니다. 특히, 추천 시스템에서 훈련 데이터의 노이즈가 있는 부분을 배제하는 데 강력한 능력을 보여줍니다. 이 연구는 MOGCSL이 효율성 및 효과성을 고려할 때 매우 뛰어난 성능을 보였다는 것을 강조합니다.



### jina-clip-v2: Multilingual Multimodal Embeddings for Text and Images (https://arxiv.org/abs/2412.08802)
Comments:
          21 pages, 1-10 main paper, 10-12 refs, 12-21 benchmarks

- **What's New**: 이번 연구에서는 jina-clip-v1 모델을 기반으로 한 향상된 프레임워크인 jina-clip-v2를 제안합니다. 이 모델은 여러 언어에서 다중 작업 및 다중 단계의 대조 학습(multi-task, multi-stage contrastive learning)을 활용하여 텍스트 전용 검색 성능을 개선합니다. ML 모델의 한계로서 명시된 다국어 지원 부족 및 복잡한 비주얼 문서 이해에서의 성능 저하 문제를 해결합니다.

- **Technical Details**: jina-clip-v2 모델은 듀얼 인코더 아키텍처(dual encoder architecture)를 사용하여 텍스트와 이미지를 동일한 임베딩 공간에서 인코딩합니다. 텍스트 인코더는 사전 훈련된 Jina-XLM-RoBERTa 모델을 초기화하고, 이미지 인코더는 EVA02 계열의 ViT 모델을 선택하였습니다. 모델 학습을 위해 다양한 다국어 데이터셋을 구축하였고, 하드 네거티브(hard negative) 샘플을 포함한 훈련 방법을 채택했습니다.

- **Performance Highlights**: jina-clip-v2는 텍스트 전용 및 다중 모드 작업에서 이전 모델에 비해 성능이 크게 향상되었습니다. 이 모델은 다국어 텍스트 검색 벤치마크 및 시각적으로 풍부한 문서 검색 벤치마크에서 국가 최우수 성능에 필적하는 결과를 보여줍니다. 또한, Matryoshka Representation Learning을 활용하여 벡터 저장 비용을 절감하며, 성능 저하 없이 출력 벡터 차원을 축소 가능합니다.



New uploads on arXiv(cs.CV)

### Doe-1: Closed-Loop Autonomous Driving with Large World Mod (https://arxiv.org/abs/2412.09627)
Comments:
          Code is available at: this https URL

- **What's New**: 새로 제안된 Doe-1은 자율주행을 위한 폐쇄 루프(closed-loop) 프레임워크를 제공하며, 기존의 방법들이 처한 여러 문제점들을 해결하고자 한다. 이 모델은 perceptio (인식), prediction (예측), planning (계획)을 통합하여 다중 모달(multi-modal) 토큰을 사용하여 다양한 작업을 수행한다. 특히, 자유형 텍스트(free-form texts)와 RGB 이미지 토큰을 활용하여 보다 효율적이고 정확한 데이터를 수집할 수 있는 가능성을 가지고 있다.

- **Technical Details**: Doe-1은 관측(observation), 설명(description), 행동(action) 토큰으로 장면을 표현하며, 이러한 토큰 간의 전이를 통해 일반적인 인식, 계획 및 예측 작업을 모델링한다. 이미지는 벡터 양자화된 변분 오토인코더(vector-quantized variational autoencoder)를 통해 토큰화되며, 행동은 새의 눈(view)에서의 이동(displacement)으로 표현된다. 이 통합된 접근 방식은 자율주행을 위해 비전 중심(vision-centric)으로 최적화되어 있으며, 각 모달의 작성을 효율적으로 실현한다.

- **Performance Highlights**: Doe-1은 nuScenes 데이터셋에서 다양한 작업을 수행하며 그 효과를 입증하였다. 실험을 통해, 시각적 질문 응답(visual question-answering), 행동 조건 영상 생성(action-conditioned video generation), 그리고 모션 계획(motion planning) 등의 작업에서 우수한 성능을 보여준다. 특히, fine-tuning 없이 여러 작업을 동시에 수행할 수 있는 가능성을 가지고 있어, 자율주행 분야에서 강력한 도구로 자리 잡을 것으로 기대된다.



### FreeScale: Unleashing the Resolution of Diffusion Models via Tuning-Free Scale Fusion (https://arxiv.org/abs/2412.09626)
Comments:
          Project Page: this http URL

- **What's New**: 이번 논문에서는 FreeScale이라는 새로운 tuning-free inference paradigm을 제안하여, 고해상도 이미지 생성을 가능하게 합니다. 기존의 diffusion 모델들이 고해상도 생성에 어려움을 겪어온 문제를 해결하고자 정보를 다양한 receptive scale에서 처리하여 원하는 주파수 성분을 융합합니다. 또한, FreeScale은 최초로 8k 해상도의 이미지를 생성할 수 있는 가능성을 열었습니다.

- **Technical Details**: FreeScale은 convolutional receptive field의 제약을 극복하기 위해 tailored self-cascade upscaling과 restrained dilated convolution을 도입합니다. 이 방법은 고해상도 생성에서 기본적인 시각 구조를 얻고 품질을 유지하는 데 기여합니다. 추가적으로, 다양한 receptive scale에서 정보를 처리하고 이를 주파수 성분에 따라 융합하여, 시각적 콘텐츠의 구조와 품질을 동시에 보장합니다.

- **Performance Highlights**: FreeScale은 이미지 및 비디오 모델 모두에서 실험을 통해 그 효과를 검증하였습니다. 실험 결과, FreeScale은 기존의 최첨단 tuning-free 방법과 비교했을 때, 시각적 품질에서 뛰어난 성능을 보여주며 짧은 추론 시간으로 새로운 경계를 넘는 이미지를 생성할 수 있음을 입증했습니다.



### Illusion3D: 3D Multiview Illusion with 2D Diffusion Priors (https://arxiv.org/abs/2412.09625)
Comments:
          Project page: this https URL

- **What's New**: 이번 연구에서는 텍스트 프롬프트나 이미지를 기반으로 3D 멀티뷰 환상을 생성하는 새로운 접근 방식을 제시합니다. 기존 방법들은 2D 이미지에 국한되어 있었고, 예술적 표현이나 실용성이 제한되어 있었습니다. 이에 반해, 본 연구는 사전 훈련된 텍스트-투-이미지(diffusion) 모델을 활용하여 사용자의 입력에 따라 더 복잡한 3D 환상을 제작할 수 있게 합니다.

- **Technical Details**: 제안하는 방법은 차별 가능한 렌더링(differentiable rendering)을 통해 신경망 3D 표현의 텍스처(texture)와 기하학(geometry)을 최적화합니다. 이 과정은 멀티 앵글(view)에서 바라볼 때 서로 다른 해석이 가능하게 하는 특징을 가지고 있습니다. 여러 기법을 개발하여 생성된 3D 멀티뷰 환상의 품질을 향상시키는 방법도 소개됩니다.

- **Performance Highlights**: 광범위한 실험을 통해 우리의 접근 방식의 효과성을 입증했습니다. 다양한 3D 형태로 환상을 생성하는 능력이 뛰어나며, 이는 예술적이고 창의적인 응용에 적합합니다. 결과적으로, 제안하는 기법은 3D 환상 생성 분야에서 중요한 발전을 보여줍니다.



### GenEx: Generating an Explorable World (https://arxiv.org/abs/2412.09624)
Comments:
          Website: this http URL

- **What's New**: 이번 연구에서 우리는 GenEx라는 플랫폼을 소개합니다. GenEx는 단일 RGB 이미지에서 시작하여 전체 3D 환경을 생성하고, 파노라마 비디오 스트림을 통해 이를 실시간으로 구현합니다. 이 시스템은 AI 에이전트가 탐험하고 상호작용할 수 있는 무한한 공간을 제공합니다.

- **Technical Details**: GenEx는 두 가지 주요 구성 요소인 상상 세계와 구현된 에이전트로 구성되어 있습니다. 상상 세계는 탐험을 위한 3D 환경을 동적으로 생성하며, 구현된 에이전트는 이 환경과 상호작용하여 이해를 정교화합니다. 에이전트는 예측 기대(predictive expectations)를 활용하여 물리적 세계의 보지 못한 부분을 탐색합니다.

- **Performance Highlights**: GenEx는 고품질 세계 생성, 긴 경로에서의 강력한 루프 일관성 및 일관성을 유지하며 활발한 3D 매핑을 수행하는 능력을 보여줍니다. 이 플랫폼은 목표가 없는 탐험 및 목표 중심 내비게이션과 같은 복잡한 임무 수행을 위한 AI 에이전트의 역할을 강화합니다.



### OmniDrag: Enabling Motion Control for Omnidirectional Image-to-Video Generation (https://arxiv.org/abs/2412.09623)
- **What's New**: OmniDrag는 몰입감 있는 전방향 비디오(ODV) 생성을 위한 첫 번째 방법으로, 장면과 객체 수준의 모션 제어 기능을 제공합니다. 이전 텍스트 기반 ODV 생성의 한계점을 극복하고 정확하고 고품질 비디오 생성을 가능하게 합니다. 이 방법은 사전 학습된 동영상 확산 모델을 기반으로 하여 복잡한 구형 운동을 효과적으로 처리할 수 있는 새로운 제어 모듈을 도입합니다.

- **Technical Details**: OmniDrag에서는 장면 수준과 객체 수준의 제어를 가능하게 하는 구형 운동 추정기(Spherical Motion Estimator, SME)를 개발하였습니다. SME는 중요한 움직임을 균일하고 정확하게 포착하고, 사용자가 핸들과 목표점을 단순히 드로잉함으로써 ODV 생성을 가능하게 합니다. 이와 함께, Move360이라는 새로운 ODV 데이터셋을 소개하며, 이는 큰 장면과 객체 움직임을 포함하는 1,500개 이상의 비디오 클립으로 구성되어 있습니다.

- **Performance Highlights**: 실험 결과 OmniDrag는 ODV 생성에서 장면 수준 및 객체 수준의 제어에 있어 현저한 우수성을 입증하였습니다. 이는 사용자에게 직관적이고 높은 품질의 동적 비디오 생성 경험을 제공합니다. 또한, Move360 데이터셋에서 훈련된 모델은 더 큰 운동 범위를 효과적으로 제어할 수 있어 실제 응용 가능성을 높입니다.



### LoRACLR: Contrastive Adaptation for Customization of Diffusion Models (https://arxiv.org/abs/2412.09622)
Comments:
          Project page: this https URL

- **What's New**: LoRACLR는 개별 LoRA 모델을 통합하여 다중 개념 이미지 생성을 가능하게 하는 새로운 접근 방식을 제시합니다. 이 방법은 특별한 미세 조정 없이 각 모델의 가중치 공간을 정렬 및 병합하는 대조적(dcontrastive) 목표를 사용하여 호환성을 보장합니다. 또한, LoRACLR은 기존 LoRA 모델을 활용하면서 추가적인 재훈련 없이도 다중 개념의 이미지 합성을 가능하게 합니다.

- **Technical Details**: LoRACLR는 기존에 훈련된 LoRA 모델을 통합하기 위해 최적화 기반 병합을 사용합니다. 각 모델의 가중치 공간을 정렬하고 상호 간섭을 최소화하여 공동 구성 내에서 고충실도를 유지하는 것이 핵심입니다. 또한, LoRA는 대형 모델을 저계(rank) 행렬을 추가하여 미세 조정하며, 안정적 확산 모델의 크로스 어텐션 층에 적용되어 계산 효율성을 높입니다.

- **Performance Highlights**: LoRACLR의 평가는 기존 방법들에 비해 시각적 품질 및 구성 일관성에서 상당한 개선을 이룸을 보여줍니다. 질적 및 양적 실험을 통해, LoRACLR은 각 개념의 충실도와 정체성을 유지하며, 복잡성이 증가해도 일반적인 특징 간섭 문제를 피하고 있습니다. 이는 가상 콘텐츠 생성, 개인화된 스토리텔링, 디지털 아트 제작 등 여러 분야에서 적용 가능성을 높입니다.



### Stereo4D: Learning How Things Move in 3D from Internet Stereo Videos (https://arxiv.org/abs/2412.09621)
- **What's New**: 이번 연구는 인터넷의 스테레오 와이드 앵글 비디오에서 동적 3D 장면을 복원하기 위한 새로운 시스템을 제안합니다. 이 시스템은 카메라 포즈 추정, 스테레오 깊이 추정, 시간적 추적 방법의 결과를 융합하고 필터링하여 고품질의 3D 복원을 수행합니다. 이를 통해 장기적인 모션 궤적을 가진 월드-일관성(pseudo-metric) 3D 포인트 클라우드를 생성할 수 있습니다.

- **Technical Details**: 연구진은 동적 장면을 위한 구조-from-모션 구조를 최적화하여 온라인 스테레오 피쉬아이 영상에서 고성능 3D 데이터를 추출하는 파이프라인을 구축했습니다. 이 시스템은 깊이 맵, 카메라 포즈, 이미지와 같은 중간 결과물 및 2D 대응 관계를 포함하여 100K 이상의 비디오 시퀀스를 추출할 수 있습니다. DynaDUSt3R라는 새로운 방법을 통해 실제 이미지 쌍에서 3D 구조와 모션을 예측할 수 있습니다.

- **Performance Highlights**: DynaDUSt3R 모델을 활용하여 여러 현실 세계의 이미지 쌍을 기반으로 3D 구조와 모션 예측의 성능을 입증했습니다. 이 데이터셋은 다양한 동적 장면에 대해 일반화할 수 있는 잠재력을 보여주며, 새로운 3D 작업의 발전을 촉진하는 데 기여할 것으로 기대됩니다. 연구팀은 이 방법이 실제 다양한 환경에서 효과적으로 작동함을 강조합니다.



### Learning Camera Movement Control from Real-World Drone Videos (https://arxiv.org/abs/2412.09620)
- **What's New**: 본 논문에서는 기존 피사체를 매력적인 비디오로 촬영하기 위한 카메라 움직임 제어를 자동화하는 방법을 제안합니다. 드론 비디오를 테스트 케이스로 선택하여 복잡한 카메라 경로와 고유한 시청 각도를 활용하며, 이는 기존 AI 비디오 그래피의 한계를 극복하려는 시도입니다. 특히, 99,000개의 카메라 궤적을 수집하고 카메라 경로를 예측하는 DVGFormer라는 새로운 모델을 도입합니다.

- **Technical Details**: 이 연구는 DroneMotion-99k 데이터셋을 통해 3D 카메라 경로를 자동으로 생성하는 파이프라인을 구축하였습니다. Colmap을 사용하여 온라인 비디오로부터 3D 카메라 포즈를 재구성하고, 카메라 궤적을 추출하여 낮은 품질의 데이터를 거릅니다. DVGFormer는 과거 프레임의 카메라 경로 및 이미지를 바탕으로 다음 프레임의 카메라 움직임을 예측하며, 이는 기존의 틀을 넘어서는 혁신적인 접근입니다.

- **Performance Highlights**: 제안된 시스템은 38개의 합성 자연 장면과 7개의 실제 도시 3D 스캔을 통해 평가되었습니다. 이 시스템은 장애물을 내비게이션하고, 낮은 고도를 유지하여 인식된 속도를 높이며, 타워와 건물 주위를 회전하는 등 고품질 비디오 촬영을 위한 도전적인 카메라 움직임을 성공적으로 학습하였습니다. 결과적으로, 이 모델은 사용자 선호도에서 우수성을 보였고, 낮은 충돌률과 향상된 움직임 부드러움을 기록했습니다.



### SnapGen: Taming High-Resolution Text-to-Image Models for Mobile Devices with Efficient Architectures and Training (https://arxiv.org/abs/2412.09619)
- **What's New**: 이 논문에서는 기존의 텍스트-이미지(T2I) 확산 모델이 가진 여러 한계를 극복하기 위해 매우 작고 빠른 T2I 모델을 개발했습니다. 이 모델은 모바일 플랫폼에서 고해상도와 고품질 이미지를 생성하는 것을 목표로 합니다. 그 결과, 모바일에서 1.4초 만에 1024x1024 px 이미지를 생성할 수 있는 SnapGen 모델이 탄생했습니다.

- **Technical Details**: 모델의 성능을 개선하기 위해 여러 기술을 도입했습니다. 첫째, 네트워크 아키텍처의 설계 선택을 체계적으로 검토하여 모델 매개변수와 대기시간을 줄임과 동시에 고품질 생성을 보장합니다. 둘째, 생성 품질을 더욱 향상시키기 위해 훨씬 더 큰 모델로부터의 cross-architecture knowledge distillation을 적용하여 다층적으로 모델 학습을 지도합니다. 셋째, 적대적 가이드를 도입하여 몇 단계의 생성을 가능하게 합니다.

- **Performance Highlights**: 모델은 단 372M 매개변수로 ImageNet-1K에서 256x256 px 생성에 대해 2.06의 FID를 달성했습니다. T2I 벤치마크(예: GenEval 및 DPG-Bench)에서 379M 매개변수로도 수십억 개의 매개변수를 가진 대형 모델을 초월했습니다. 이 모델은 SDXL보다 7배, IF-XL보다 14배 더 작습니다.



### EasyRef: Omni-Generalized Group Image Reference for Diffusion Models via Multimodal LLM (https://arxiv.org/abs/2412.09618)
Comments:
          Tech report

- **What's New**: 본 논문에서는 EasyRef라는 새로운 적응 기법을 소개합니다. 이 방법은 확산 모델이 여러 개의 참조 이미지와 텍스트 프롬프트에 조건화될 수 있도록 합니다. 기존의 방법들이 개별 이미지의 임베딩을 평균화하여 참조 이미지를 처리한 반면, EasyRef는 여러 이미지 간의 일관된 시각적 요소를 효과적으로 캡처하는 능력을 지닌 멀티모달 대형 언어 모델을 활용합니다.

- **Technical Details**: EasyRef는 Denoising Diffusion Probabilistic Models (DDPMs)와 함께 작동하며, 훈련 과정에서 가우시안 노이즈를 단계별로 데이터에 추가합니다. 이 모델은 노이즈 제거 과정을 통해 훈련된 파라미터 모델 pθ를 학습하여 주어진 노이즈 샘플에서 원본 데이터를 복원합니다. 또한, MLLM의 표현을 확산 과정에 주입하여 미지의 도메인에서도 일반화할 수 있습니다.

- **Performance Highlights**: 실험 결과는 EasyRef가 IP-Adapter와 LoRA와 같은 튜닝 기반 방법을 초월하여 다양한 도메인에서 우수한 심미적 품질과 강력한 제로샷 일반화를 달성함을 보여줍니다. 새로운 다중 참조 이미지 생성 벤치마크인 MRBench도 소개되어, 앞으로의 연구 방향을 제시하고 성능 평가에 사용될 것입니다.



### V2PE: Improving Multimodal Long-Context Capability of Vision-Language Models with Variable Visual Position Encoding (https://arxiv.org/abs/2412.09616)
Comments:
          The code and models will be available at this https URL

- **What's New**: 저자들은 Variable Visual Position Encoding (V2PE)라는 새로운 위치 인코딩 접근 방식을 제안합니다. 이는 시각적 토큰을 처리하기 위한 변수를 사용할 뿐만 아니라 더 작은 증가량을 통해 긴 멀티모달 시퀀스를 더 효율적으로 관리할 수 있도록 합니다. V2PE는 기존 위치 인코딩 방식의 한계를 극복하고 VLMs의 긴 컨텍스트 이해 능력을 향상시키는 데 중점을 둡니다.

- **Technical Details**: 연구자들은 멀티모달 데이터를 훈련하고 평가하기 위해 대규모의 긴 컨텍스트 멀티모달 데이터셋을 구축했습니다. 특히, 기존 데이터셋의 시퀀스 길이를 32K 또는 256K 토큰으로 확장하고, 결과로 얻은 V2PE를 사용하여 모델이 시각적 토큰의 위치를 효과적으로 처리하게 설계되었습니다. 실험 결과, V2PE는 VLMs가 긴 컨텍스트 멀티모달 시퀀스에서 더 잘 이해하고 추론할 수 있음을 보여주었습니다.

- **Performance Highlights**: V2PE를 활용하여 재조정된 open-source VLM, InternVL2-2B는 일반 멀티모달 벤치마크 뿐만 아니라 긴 컨텍스트 멀티모달 작업에서도 뛰어난 성능을 발휘합니다. 특히 훈련 데이터의 시퀀스 길이를 256K 토큰으로 증가시키자, 모델은 최대 1M 토큰에 이르는 멀티모달 시퀀스를 처리할 수 있게 되었으며, 이는 실제 복잡한 멀티모달 데이터 적용 시 뛰어난 결과를 보일 가능성을 암시합니다.



### Context Canvas: Enhancing Text-to-Image Diffusion Models with Knowledge Graph-Based RAG (https://arxiv.org/abs/2412.09614)
Comments:
          Project Page: this https URL

- **What's New**: 이 논문에서는 그래프 기반의 Retrieval-Augmented Generation (RAG)을 사용하여 텍스트-이미지 생성 모델(T2I)의 기능을 향상시키는 새로운 접근 방식을 소개합니다. 이 시스템은 지식 그래프에서 동적으로 세부 캐릭터 정보와 관계 데이터를 검색하여 시각적으로 정확하고 맥락이 풍부한 이미지를 생성합니다. Context Canvas는 이러한 기술을 활용하여 고충실도(context-aware) 다면적 이미지를 생성하는 첫 번째 응용 프로그램입니다.

- **Technical Details**: Context Canvas는 텍스트-이미지 확산 모델을 개선하기 위해 구조화된 지식 그래프에서 관련 텍스트 정보를 동적으로 검색합니다. 이 접근 방식은 훈련 데이터 이상의 개념을 생성하고, 더 높은 정확성, 일관성 및 해석 가능성을 제공합니다. 또한, RAG 기반의 자기 수정 메커니즘을 제안하여 다양한 맥락을 기반으로 시각적 출력을 iteratively 조정함으로써, 여러 측면에서 일관된 이미지를 생성하도록 돕습니다.

- **Performance Highlights**: 정성적 및 정량적 실험 결과, Context Canvas는 Flux, Stable Diffusion 및 DALL-E와 같은 인기 모델의 성능을 현저히 향상시킵니다. ControlNet 기능도 강화되어 보다 세밀한 이미지 편집 작업에서 효율적으로 작동합니다. 이 연구에서는 문화적으로 구체적이고 복합적인 개념을 생성하는 데 있어 기존 모델의 한계를 해결하고 정확한 시각적 내러티브를 실현하는 데 기여합니다.



### PVC: Progressive Visual Token Compression for Unified Image and Video Processing in Large Vision-Language Models (https://arxiv.org/abs/2412.09613)
- **What's New**: 이번 논문에서는 이미지와 비디오를 통합할 수 있는 새로운 방식, 즉 Progressive Visual Token Compression (PVC)을 제안합니다. 기존의 이미지와 비디오 처리 방식은 각기 다른 token 압축 전략을 사용해 왔으나, PVC는 두 매체의 token 압축을 통합하여 더 효율적으로 처리할 수 있습니다. 이는 비디오의 동적 변화를 포착함과 동시에 이미지의 공간적 세부정보를 보존하도록 설계되었습니다.

- **Technical Details**: PVC는 각 이미지를 "정적" 비디오로 확장하고, 이전 프레임에서 추출되지 않은 새로운 정보만을 인코딩하는 progressive encoding 방식을 채택합니다. 또한, AdaLN(Adaptive Layer Normalization) 모듈을 사용하여 프레임 간 중복을 피하고 다소 복잡한 정보 추출을 가능하게 하였습니다. 이러한 접근 방식은 시간적으로 인접한 프레임 간의 중복성을 최소화하여 압축 효율을 높입니다.

- **Performance Highlights**: 이 모델은 다양한 비디오 이해 기준에서 최첨단 성능을 보여주며, 그것은 시간적 변화 및 세부 사항을 효과적으로 포착하는 데 성공했습니다. 특히, 긴 비디오 작업과 섬세한 짧은 비디오 작업에서 뛰어난 성능을 보이며, 이미지 기준에서도 성능 손실 없이 동작합니다. 이는 PVC 방식이 기존의 방식보다 우수하게 시각적 작업을 처리함을 의미합니다.



### Olympus: A Universal Task Router for Computer Vision Tasks (https://arxiv.org/abs/2412.09612)
Comments:
          Technical Report

- **What's New**: Olympus는 Multimodal Large Language Models (MLLMs)를 활용하여 다양한 컴퓨터 비전 작업을 수행하도록 통합된 프레임워크를 제안합니다. 이는 이미지를 포함한 20가지 전문 작업을 처리하는 데 사용되는 특정 모듈에 대한 작업을 위임하는 방식으로 구성되어 있습니다. 라이팅(is instruction-based routing) 시스템을 통해 복잡한 워크플로우를 구현할 수 있으며, 기존의 MLLMs와 쉽게 통합되어 성능을 향상시킵니다.

- **Technical Details**: Olympus는 446.3K개 고품질 트레이닝 데이터와 49.6K개 평가 데이터를 포함하는 OlympusInstruct 및 OlympusBench라는 데이터 세트를 통해 설계되었습니다. 각 전문 작업을 위한 특정 라우팅 토큰을 만들어 사용자 명령 내에서 여러 작업을 체인화하여 수행할 수 있는 능력을 가지고 있습니다. 이 시스템은 MLLM으로서의 기능을 활용하여 외부 모듈로 작업을 위임하는 방식으로 작동합니다.

- **Performance Highlights**: 실험 결과, Olympus는 20개 개별 작업에서 평균 94.75%의 라우팅 정확성을 달성했으며, 체인 액션 시나리오에서는 91.82%의 정밀도를 기록했습니다. 이는 Olympus가 다양한 컴퓨터 비전 작업을 효과적으로 처리할 수 있는 보편적 작업 라우터로 기능할 가능성을 강조합니다. 또한 Olympus는 단일 명령 내에서 최대 5개의 작업을 해결할 수 있는 능력을 보여줍니다.



### FluxSpace: Disentangled Semantic Editing in Rectified Flow Transformers (https://arxiv.org/abs/2412.09611)
Comments:
          Project Page: this https URL

- **What's New**: 이 논문에서는 이미지 편집을 위한 새로운 프레임워크인 FluxSpace를 소개합니다. FluxSpace는 rectified flow transformers를 기반으로 하여 명시적인 조작을 가능하게 하고 세분화된 이미지 편집 및 예술적 창작 작업을 지원합니다. 특히, 모델의 출력 공간 내에서 정밀한 탐색이 가능하다는 점에서 기존 연구와 차별화됩니다.

- **Technical Details**: FluxSpace는 transformer 블록에서 학습된 표현을 활용하여 높은 의미 해석 가능한 표현을 제공합니다. 이를 통해 세분화된 편집 작업을 지원하고, 텍스트 프롬프트를 기반으로 한 이미지 생성 및 편집에서 발생하는 얽힌 수정 문제를 해결합니다. 추가적으로, 본 연구는 joint transformer 블록에서 인코딩된 고도로 분리된 의미 정보를 통해 지속적인 콘텐츠 정제를 가능하게 합니다.

- **Performance Highlights**: FluxSpace는 기존의 방법들에 비해 더욱 정교한 편집을 지원하며, 웃음 추가와 같은 세밀한 수정과 스타일화와 같은 큰 수준의 변경을 가능하게 합니다. 실험 결과, 제안된 방법은 다양한 최신 기법들과의 비교에서 효과성을 입증하며, 실제 이미지는 물론 생성된 이미지의 편집도 지원합니다.



### Representing Long Volumetric Video with Temporal Gaussian Hierarchy (https://arxiv.org/abs/2412.09608)
Comments:
          SIGGRAPH Asia 2024 (TOG). Project page: this https URL

- **What's New**: 이 논문은 다중 뷰 RGB 비디오에서 긴 볼륨 비디오를 재구성하는 도전에 초점을 맞추고 있습니다. 최근의 동적 뷰 합성(dynamic view synthesis) 방법은 고품질 렌더링 결과를 달성하기 위해 강력한 4D 표현을 활용하지만, 대부분이 짧은 클립에 국한되어 있습니다. 이를 극복하기 위해 제안된 Temporal Gaussian Hierarchy는 긴 볼륨 비디오를 효과적으로 모델링할 수 있습니다.

- **Technical Details**: Temporal Gaussian Hierarchy는 서로 다른 속도로 변화하는 장면 영역을 구분하여 다중 수준의 4D Gaussian 원시(primitives) 구조를 생성합니다. 이 계층 구조는 장면을 동적 성분 변화에 따라 효율적으로 설명하고, 고정된 GPU 메모리 사용을 보장하면서 훈련 및 렌더링 성능을 최적화합니다. 또한, 하이브리드 외관 모델(hybrid appearance model)을 통해 Gaussian 원시의 과적합(overfitting) 문제를 완화하면서 동적 뷰 의존적으로 높은 표현 능력을 유지합니다.

- **Performance Highlights**: 제안된 방법은 훈련 비용, 렌더링 속도 및 저장 공간 측면에서 기존 방법들보다 우수한 결과를 보입니다. RTX 4090 GPU를 사용하여 18,000 프레임의 비디오를 450 FPS로 렌더링할 수 있으며, 고해상도(1080p)의 비디오 처리에서도 뛰어난 성능을 제공합니다. 다양한 데이터셋에서의 실험을 통해, 본 방법은 기존 방법들보다 월등한 시각적 품질과 렌더링 속도를 유지하면서도 훈련 비용과 메모리 사용을 크게 줄이는 점을 입증했습니다.



### Spectral Image Tokenizer (https://arxiv.org/abs/2412.09607)
- **What's New**: 이 논문에서는 이미지 생성의 autoregressive transformer 기반 방법에 필수적인 역할을 하는 이미지 토크나이저(image tokenizer)의 개선된 방식을 제안합니다. 기존의 토크나이저는 입력 이미지의 공간적 위치에 따라 순서가 매겨져 있었으나, 우리는 이산 웨이블릿 변환(discrete wavelet transform, DWT)을 이용해 이미지 스펙트럼을 토큰화하여 더 나은 표현 방식을 구현하고자 합니다.

- **Technical Details**: 제안하는 토크나이저는 이미지의 주파수 영역에서 자연 이미지가 높은 주파수에서 더 압축 가능하다는 점을 활용합니다. 또한, 다양한 해상도의 이미지를 재훈련 없이도 수용하고 재구성할 수 있으며, 다음 토큰 예측을 위한 조건 설정을 개선해줍니다. 이를 통해 초기 몇 개의 토큰으로 이미지의 대략적인 버전을 재구성할 수 있는 부분적 디코딩(partial decoding)과 이미지 업샘플링(image upsampling)을 가능하게 합니다.

- **Performance Highlights**: 실험을 통해 토크나이저의 재구성 메트릭과 다중 스케일 이미지 생성, 텍스트 기반 이미지 업샘플링, 편집 성능을 평가했습니다. 결과적으로 제안된 방법은 기존 기법보다 개선된 성능을 보여주며, 이미지 생성 및 수정 작업에 대해 보다 효율적인 접근 방식을 제공함을 입증합니다.



### Feat2GS: Probing Visual Foundation Models with Gaussian Splatting (https://arxiv.org/abs/2412.09606)
Comments:
          Project Page: this https URL

- **What's New**: 이 논문에서는 기존의 비주얼 파운데이션 모델(Visual Foundation Models, VFM)이 3D 세계를 얼마나 잘 이해하는지 평가하기 위한 새로운 방법, Feat2GS를 소개합니다. Feat2GS는 비모양 이미지에서 추출한 VFM 특징으로부터 3D Gaussian 속성을 읽어들이며, 이는 3D 데이터 없이도 기하학(geometry)과 텍스처(texture)에 대한 이해도를 탐색할 수 있게 해줍니다. 기존 3D 프로빙 작업이 무시했던 텍스처 인식의 중요성을 강조합니다.

- **Technical Details**: Feat2GS는 기하학적 파라미터($\boldsymbol{x}, \alpha, \Sigma$)와 텍스처($\boldsymbol{c}$)를 분리하여 분석할 수 있도록 설계되었습니다. 이로 인해 사용자는 3D 인식의 다양한 요소를 독립적으로 연구할 수 있으며, 새로운 시점 합성(novel view synthesis) 작업도 지원합니다. 이를 통해 기존의 제한된 2D 뷰에 대한 의존성을 극복하는 방법론이 제시됩니다.

- **Performance Highlights**: 논문에서는 여러 VFM을 대상으로 Feat2GS를 통해 3D 인식을 탐색하는 실험을 수행하였으며, 3D 인식이 잘 이루어지는 VFM의 구성 요소를 찾아냈습니다. 이러한 발견을 바탕으로 다양한 데이터셋에서 최첨단 성능을 달성하는 다양한 변형을 개발했습니다. Feat2GS는 VFM 프로빙 및 새로운 뷰 합성을 위한 유용한 기준선(baseline)으로 자리매김할 것으로 기대됩니다.



### SynerGen-VL: Towards Synergistic Image Understanding and Generation with Vision Experts and Token Folding (https://arxiv.org/abs/2412.09604)
- **What's New**: 이번 논문에서는 SynerGen-VL이라는 새로운 Multimodal Large Language Model (MLLM)을 제안했습니다. 이 모델은 인코더 없이 이미지 이해 및 생성을 동시에 수행할 수 있는 단순하면서도 강력한 구조를 가지고 있습니다. SynerGen-VL은 특히 이미지의 고해상도 이해를 지원하기 위해 토큰 접기(token folding) 메커니즘과 비전 전문화 진행 정렬(pretraining) 전략을 도입했습니다.

- **Technical Details**: SynerGen-VL은 Multimodal Mixture-of-Experts (MMoE) 구조에서 영감을 받아, 이미지 표현을 위한 추가 매개변수를 갖춘 비전 전문가를 도입합니다. 이를 통해 LLM의 사전 훈련된 지식을 최대한 보호하면서 비전 기능을 통합합니다. 모델의 토큰 시퀀스는 고해상도 이미지를 효과적으로 지원하기 위해 압축되며, 이미지 생성 동안 세부 이미지를 재구성하는 추가 디코더를 사용합니다.

- **Performance Highlights**: SynerGen-VL은 대규모 혼합 이미지-텍스트 데이터에서 훈련되었으며 다양한 이미지 이해 및 생성 벤치마크에서 평가되었습니다. 실험 결과, SynerGen-VL은 기존의 인코더 없는 통합 MLLM과 비교할 때 유사하거나 더 작은 파라미터 크기로도 뛰어난 성능을 달성하였고, 작업별 최첨단(state-of-the-art) 모델과의 격차를 줄였습니다. 특히, 2.4B의 활성화된 파라미터로도 8B의 Emu3와 동급의 성능을 보여주어 차세대 통합 MLLM으로 높은 잠재력을 입증했습니다.



### Do Multimodal Large Language Models See Like Humans? (https://arxiv.org/abs/2412.09603)
Comments:
          Project page: this https URL

- **What's New**: 이번 연구에서는 MLLMs(다중 모달 대형 언어 모델)이 인간의 시각 정보를 어떻게 인식하는지를 판단하기 위한 새로운 벤치마크인 HVSBench를 소개합니다. HVSBench는 85K 이상의 다중 모달 샘플로 구성되어 있으며, 13개 카테고리와 5개 분야에서 인간의 시각 시스템(HVS)과 MLLMs 간의 정렬 정도를 평가하기 위해 설계되었습니다. 본 벤치마크는 MLLMs가 인간처럼 시각 정보를 처리하는지를 평가하는 새로운 기준을 제공합니다.

- **Technical Details**: HVSBench는 Prominence, Subitizing, Prioritizing, Free-Viewing, Searching를 포함한 HVS의 5개 분야에서 85K 이상의 질문과 관련된 이미지를 제공합니다. HVS에 따른 질문은 13가지 유형으로 십분화되어 있으며, 각 질문은 다중 선택형, 정수 예측, 정렬 및 스캔패스 예측을 포함합니다. 평가 프로토콜은 인간의 인지를 기반으로 하여 MLLMs의 성능을 보다 정확하게 평가할 수 있도록 설계되었습니다.

- **Performance Highlights**: HVSBench를 통해 13개의 SOTA MLLMs를 평가한 결과, 최상의 모델조차도 여전히 개선 여지가 크고 대부분이 보통 수준의 성능만을 보였습니다. 연구 결과는 MLLMs와 인간의 HVS 간의 현격한 차이를 강조하며, 인간과 정렬된 MLLMs의 개발을 위한 중요한 통찰력을 제공합니다. 또한, HVS에 정렬된 MLLMs가 다운스트림 애플리케이션에서 성능을 향상시킬 수 있다는 것을 보여주는 방법도 제시되었습니다.



### Hidden Biases of End-to-End Driving Datasets (https://arxiv.org/abs/2412.09602)
Comments:
          Technical report for the CVPR 2024 Workshop on Foundation Models for Autonomous Systems. Runner-up of the track 'CARLA Autonomous Driving Challenge' in the 2024 Autonomous Grand Challenge (this https URL)

- **What's New**: 본 연구는 CARLA Leaderboard 2.0에서 엔드 투 엔드(End-to-End) 자율주행을 위한 최초의 시도를 제시합니다. 기존의 아키텍처 대신 훈련 데이터셋(training dataset) 분석에 중점을 두어, 전문가의 운전 스타일이 정책 성능에 미치는 영향과 간단한 기준으로 프레임을 가중치 부여하는 것의 문제점을 밝혀냈습니다. 이 연구의 결과로 제안된 새로운 데이터 필터링 방법은 데이터셋의 크기를 대폭 줄일 수 있으며, 모델은 2024 CARLA Challenge에서 두 개의 트랙에서 각각 1위 및 2위로 평가받았습니다.

- **Technical Details**: 연구에서는 PDM-Lite라는 오픈소스로 제공되는 플래너를 활용하여 CARLA Leaderboard 2.0 시나리오에서 데이터를 수집했습니다. 수집된 데이터는 RGB 이미지, LiDAR 포인트 클라우드, 경로 체크포인트 및 주변 물체에 대한 경계 상자 예측을 포함합니다. 이와 더불어 TransFuser++라는 기존의 Imitation Learning 모델을 약간 수정하여 훈련에 사용하였습니다.

- **Performance Highlights**: 제안된 모델은 다양한 도시 주행 시나리오를 안전하게 처리하며, 2024 CARLA Challenge에서 두 번째로 높은 순위를 기록했습니다. 또한 Bench2Drive 테스트 경로에서 1위를 차지하며, 기존의 평가 메트릭의 설계 결함을 밝혀냈습니다. 이러한 성과는 훈련 데이터셋의 중요성을 강조하며, 향후 도전과제를 위한 메트릭 개선 방안을 제안합니다.



### TimeRefine: Temporal Grounding with Time Refining Video LLM (https://arxiv.org/abs/2412.09601)
- **What's New**: 이번 연구에서는 비디오 템포럴 그라운딩(Video Temporal Grounding, VTG)의 문제를 해결하기 위해 TimeRefine이라는 새로운 접근 방식을 제안합니다. 기존의 비디오 LLM(Large Language Models)은 타임스탬프를 직접 예측하는 방법에 의존했으나, TimeRefine는 초기 예측 후 오프셋(offset)을 다시 예측하는 프로세스를 통해 정확성을 향상시킵니다. 이 방법은 모델의 셀프-수정(self-correct) 기능을 강조하며, 모델이 그라운드 트루스(ground truth)에서 얼마나 멀리 떨어져 있는지에 따라 페널티를 부여하는 보조 예측 헤드(auxiliary prediction head)를 추가하여 모델의 시간 인식 능력을 강화합니다.

- **Technical Details**: TimeRefine의 핵심 원리는 타임스탬프 예측을 직접 수행하는 대신 점진적인 정제 작업으로 재구성하는 것입니다. 모델은 먼저 대략적인 타임스탬프를 예측하고, 이후 이를 기반으로 오프셋을 예측하여 최종 예측에 도달합니다. 이 과정은 여러 차례 반복되며, 모델은 이전 예측에서 발생한 오류를 스스로 수정합니다. 또한, L1 손실(L1 loss)을 사용하여 예측이 그라운드 트루스에서 얼마나 멀리 떨어져 있는지에 따라 더 많은 페널티를 부여하여 모델의 학습을 유도합니다.

- **Performance Highlights**: 실험 결과 TimeRefine를 적용한 VTimeLLM은 ActivityNet 캡션과 Charades-STA 데이터세트에서 각각 3.6% 및 5.0%의 mIoU(mean Intersection over Union) 개선을 보여주었습니다. TimeRefine는 기존의 LLM 기반 VTG 방법과 쉽게 통합할 수 있어, 다양한 모델에서 성능 향상에 기여할 수 있습니다. 이러한 결과는 비디오 LLM의 시간 인식 능력을 획기적으로 향상시킬 잠재력을 제시합니다.



### Owl-1: Omni World Model for Consistent Long Video Generation (https://arxiv.org/abs/2412.09600)
Comments:
          Code is available at: this https URL

- **What's New**: 본 논문에서는 일관성 있는 장기 비디오 생성을 위한 Omni World modeL인 Owl-1을 제안합니다. 전통적인 비디오 생성 모델들이 단기 정보를 바탕으로 비디오를 생성하는 데 한계를 보이는 반면, Owl-1은 잠재 상태 변수를 이용하여 장기적이고 포괄적인 조건을 형성하여 더 일관된 비디오를 생성합니다. 이 모델은 비디오가 진화하는 세계의 관찰로 간주하고, 세계의 동적인 변화를 반영하는 방법으로 설계되었습니다.

- **Technical Details**: Owl-1은 잠재 상태 변수를 사용하여 현재와 과거의 세계 정보를 인코딩함으로써, 비디오 클립으로 디코딩됩니다. 이 모델은 세계의 미래 역학을 예측하여 상태 변수를 업데이트하고, 이를 통해 긴 비디오의 일관성을 높이는 동시에 내용의 다양성도 확보합니다. 특히, 대규모 다중모달 모델(LMM)과 비디오 확산 모델을 사용하여 강력한 생성 성능을 달성합니다.

- **Performance Highlights**: Owl-1은 VBench-I2V 및 VBench-Long 벤치마크 테스트에서 최신 기술(SOTA) 방법들과 비견되는 성능을 보여 주었습니다. 이를 통해 고품질 비디오 관찰을 생성할 수 있는 능력을 입증한 것으로 평가받고 있습니다. Owl-1의 접근 방식은 비디오 생성 모델의 새로운 가능성을 여는 데 기여할 것입니다.



### RatBodyFormer: Rodent Body Surface from Keypoints (https://arxiv.org/abs/2412.09599)
- **What's New**: 이 논문은 자동으로 쥐의 행동을 분석하기 위해 RatDome이라고 하는 새로운 멀티카메라 시스템과 RatBodyFormer이라는 네트워크를 소개합니다. RatDome은 쥐의 3D 신체 표면 포인트를 밀집하게 샘플링할 수 있도록 대규모 데이터셋을 생성합니다. 특히, 이 시스템은 쥐 몸의 텍스처가 없기 때문에 일반적인 키포인트 탐지 방법으로는 포착할 수 없었던 정보들을 수집할 수 있는 혁신적인 방법을 제공합니다.

- **Technical Details**: RatDome은 색상이 있는 비드를 잠시 쥐에 부착하여 이들을 추적하고, 다중 뷰 기하학을 이용하여 3D 신체 표면 포인트를 회복하는 데이터셋을 수집합니다. RatBodyFormer는 탐지된 3D 키포인트를 입력으로 받아 3D 신체 표면 포인트로 변환하는 트랜스포머 기반 모델입니다. 이 모델은 훈련 데이터에서 3D 신체 표면 포인트의 정확한 위치에 무관하며, 마스크 학습(masked-learning) 기법으로 훈련됩니다.

- **Performance Highlights**: 실험을 통해 RatBodyFormer는 각기 다른 자세나 형태에도 불구하고 평균 6mm의 L2 오차로 신체 표면을 정확하게 추정할 수 있음을 보여줍니다. 또한 쥐의 과거 움직임을 바탕으로 현재의 신체 움직임을 예측하는 능력을 입증하여 여러 쥐의 각기 다른 상호작용을 분석하는 데 유용한 도구로 작용할 수 있음을 시사합니다. 궁극적으로, 이 연구는 쥐 행동 분석의 자동화를 위한 기초 도구를 제공하며, 생물 의학 및 신경 과학 연구에 광범위한 영향을 미칠 수 있을 것으로 기대됩니다.



### LiftImage3D: Lifting Any Single Image to 3D Gaussians with Video Generation Priors (https://arxiv.org/abs/2412.09597)
Comments:
          Project page: this https URL

- **What's New**: 이 논문은 단일 이미지 기반의 3D 재구성을 위한 새로운 프레임워크인 LiftImage3D를 제안합니다. 이 프레임워크는 Latent Video Diffusion Models (LVDMs)의 생성적 선행 지식을 효과적으로 활용하여 3D 일관성을 유지합니다. 이러한 접근 방식은 큰 카메라 모션에도 품질 저하 없이 동영상을 생성할 수 있는 능력을 보여줍니다.

- **Technical Details**: LiftImage3D는 큰 카메라 모션을 작은 제어 가능한 단계로 나누는 관절 궤적 전략을 설계하여 3D 재구성을 위한 고급 프레임 생성을 가능하게 합니다. 또한 강력한 신경 매칭 방법인 MASt3R를 사용하여 생성된 프레임의 카메라 자세를 보정하고, 왜곡 인식 3D 가우시안 스플래팅 표현을 통해 프레임 간 독립적인 왜곡 학습을 구현합니다. 이 절차를 통해 3D 일관성을 보장할 수 있습니다.

- **Performance Highlights**: LiftImage3D는 LLFF, DL3DV 및 Tanks and Temples와 같은 세 가지 어려운 데이터 세트에서 최첨단 성능을 달성하며, 실제 환경에서 다양한 이미지에 대해 잘 일반화됩니다. 이 방법은 만화 일러스트에서 복잡한 실제 장면까지 폭넓은 입력에 대해 필요한 3D 정보를 제공합니다. 실험 결과, LiftImage3D는 이전의 방법들보다 뛰어난 시각적 품질과 3D 일관성을 보여주었습니다.



### InternLM-XComposer2.5-OmniLive: A Comprehensive Multimodal System for Long-term Streaming Video and Audio Interactions (https://arxiv.org/abs/2412.09596)
Comments:
          Github Repo: this https URL

- **What's New**: 본 연구는 인간의 인지 방식을 모방하여 장기간 동안 환경과 상호작용할 수 있는 AI 시스템 개발을 목표로 하고 있습니다. 기존의 MLLMs(다중모달 대형 언어 모델)은 실시간으로 입력을 처리하고 출력을 생성하는 기능이 제한적입니다. 이러한 한계를 극복하기 위해 연구팀은 전혀 새로운 구조의 InternLM-XComposer2.5-OmniLive(IXC2.5-OL) 시스템을 제안하였으며, 이는 스트리밍 비디오 및 오디오와 실시간으로 상호작용할 수 있는 능력을 갖추고 있습니다.

- **Technical Details**: IXC2.5-OL 시스템은 스트리밍 인지 모듈, 다중모달 장기 기억 모듈 및 추론 모듈의 세 가지 주요 구성 요소로 구성되어 있습니다. 스트리밍 인지 모듈은 비디오와 오디오의 스트림을 실시간으로 처리하여 곧바로 기억에 중요한 세부정보를 저장합니다. 다중모달 장기 기억 모듈은 단기 기억 정보를 압축하여 장기 기억으로 변환함으로써 정보 검색의 효율성과 정확성을 증가시킵니다.

- **Performance Highlights**: IXC2.5-OL 시스템은 오디오 및 비디오 벤치마크에서 뛰어난 성능을 보여주었습니다. 특히 비디오 이해에서 10B 이하의 모델 중에서 최첨단 성능을 기록했으며, StreamingBench에서 73.79%의 SOTA(상태 최적화 기술) 결과를 달성했습니다. 모델 및 소스 코드는 Github을 통해 공개되어 있어 다중 모달 스트리밍 상호작용 커뮤니티의 발전에 기여할 것입니다.



### Neural LightRig: Unlocking Accurate Object Normal and Material Estimation with Multi-Light Diffusion (https://arxiv.org/abs/2412.09593)
Comments:
          Project page: this https URL

- **What's New**: 본 연구에서는 Neural LightRig라는 새로운 프레임워크를 제안하여 단일 이미지로부터 물체의 내재적인 형상과 재료 추정을 향상시켰습니다. 이 모델은 다중 조명 조건에서의 2D diffusion 모델을 활용하여 고품질의 재조명 이미지를 생성합니다. 더욱이, G-buffer 모델을 사용해 표면 법선과 PBR 재료를 정확하게 예측합니다.

- **Technical Details**: Neural LightRig는 다중 조명 확산 모델(multi-light diffusion model)과 대규모 예측 모델을 결합합니다. 이 과정에서 Blender를 활용해 합성 재조명 데이터셋(synthetic relighting dataset)을 만들어 모델 학습을 지원합니다. U-Net 아키텍처를 통한 G-buffer 예측은 픽셀 수준의 엔드 투 엔드 감독을 통해 수행됩니다.

- **Performance Highlights**: 상세한 실험 결과에 따르면, Neural LightRig는 기존의 접근 방식에 비해 표면 법선 추정 및 PBR 재료 추정에서 탁월한 성능을 보입니다. 우리의 방법은 단일 이미지 재조명에서도 독보적인 결과를 나타내며, 프로젝트 페이지를 통해 제공되는 코드와 데이터셋을 통해 검증되었습니다.



### Gaze-LLE: Gaze Target Estimation via Large-Scale Learned Encoders (https://arxiv.org/abs/2412.09586)
- **What's New**: 본 논문에서는 시선 타겟 추정(gaze target estimation) 문제를 다루며, 사람의 주목하는 방향을 예측하는 데 있어 DINOv2 인코더를 활용한 새로운 Gaze-LLE 아키텍처를 제안합니다. 기존의 복잡한 다중 분기 설계와 달리, Gaze-LLE는 성능을 크게 향상시키면서 학습 가능한 매개변수를 일관되게 줄일 수 있습니다. 이를 통해 시선 추정 작업이 반복적인 학습 구조에서 벗어나 더 간단하고 효율적으로 진행될 수 있음을 보여줍니다.

- **Technical Details**: Gaze-LLE 구조는 단일 특징 표현을 활용하여 사람의 시선을 예측하기 위한 개인적 위치 프롬프트(person-specific positional prompt)를 통합합니다. 이 새로운 아키텍처는 이전에 존재하던 복잡한 다중 인코더 방식을 대체하며, 대형 모집단 인코더(frozen large-scale encoder)를 활용하여 더욱 정교한 예측을 가능하게 합니다. DINOv2을 기반으로 한 Gaze-LLE의 설계는 눈에 띄게 단순해지며, 훈련 과정에서 학습 가능한 매개변수 수가 1-2배 감소하였습니다.

- **Performance Highlights**: Gaze-LLE 모델은 기존의 다른 모델들보다 약 5%의 학습 가능한 매개변수를 사용하면서도, 여러 시선 추정 벤치마크에서 최첨단 성능을 달성했습니다. 또한, 파인튜닝 없이도 다양한 데이터셋에서 강력한 성능을 보이며, 훈련 시간도 1.5 GPU 시간 이하로 빠르게 소요됩니다. 연구팀은 Gaze-LLE 코드를 공개하여 더 강력한 시선 추정기를 개발할 수 있도록 할 계획입니다.



### OLA-VLM: Elevating Visual Perception in Multimodal LLMs with Auxiliary Embedding Distillation (https://arxiv.org/abs/2412.09585)
Comments:
          Project Page: this https URL

- **What's New**: 본 연구에서는 자연어 감독(natural language supervision)만으로는 MLLM의 시각적 이해 능력을 최적화할 수 없음을 제기하고, 시각적 표현을 직접 최적화할 수 있는 기회를 제안합니다. OLA-VLM은 라벨이 매겨진 시각적 표현(target visual representations) 세트로부터 LLM의 숨겨진 표현(hidden representations)에 지식을 전달하는 최초의 접근 방식입니다. 또한, MLLMs의 미세 조정 과정에서 얼마만큼의 시각적 정보를 사용할 수 있는지에 대한 새로운 통찰을 제공합니다.

- **Technical Details**: 우리는 OLA-VLM을 통해 프리트레이닝 단계에서 예측 시각 임베딩(predictive visual embedding)과 다음 텍스트 토큰 예측(next text-token prediction) 간의 결합 최적화를 수행합니다. 이 과정은 특정 작업에 대한 목표 시각 정보(target visual information)를 LLM의 입력 시퀀스에 통합하여 implicit visual chain of thought를 형성하도록 설계되었습니다. 이를 통해 효율성과 성능의 균형을 이루며 모델의 시각적 이해 능력을 향상시킵니다.

- **Performance Highlights**: 우리의 OLA-VLM 모델은 다양한 벤치마크에서 단일 및 다중 인코더 베이스라인을 능가했으며, 특히 CV-Bench의 Depth 과제에서 평균 8.7% 향상을 보였습니다. 이 연구는 MLLM에서 시각적 표현의 질과 모델의 성능 간의 긍정적 상관관계를 강조하며, 모델이 훈련 데이터를 통해 점진적으로 시각적 이해 능력을 향상시킴을 보여줍니다.



### FreeSplatter: Pose-free Gaussian Splatting for Sparse-view 3D Reconstruction (https://arxiv.org/abs/2412.09573)
Comments:
          Project page: this https URL

- **What's New**: 이번 연구에서는 FreeSplatter라는 새로운 피드포워드(Fedd-forward) 복원 프레임워크를 소개합니다. 이 모델은 미보정된 스파스 뷰 이미지를 사용하여 고품질 3D Gaussian을 생성하고, 카메라 파라미터를 몇 초 내에 복구하는 능력이 있습니다. 기존의 복원 모델들이 필수적인 카메라 포즈를 요구하는 것에 반해, FreeSplatter는 이러한 제약에서 벗어나 스파스 뷰 복원을 수행할 수 있도록 설계되었습니다.

- **Technical Details**: FreeSplatter는 다중 뷰 이미지 토큰 간 정보 교환을 지원하는 자기 주의 블록을 포함하는 간소화된 트랜스포머 아키텍처에 기반하고 있습니다. 이 프레임워크는 입력 카메라 포즈나 내부 파라미터 없이 각 픽셀에 대한 Gaussian을 예측하여, 이를 고충실도 3D 장면 표현으로 활용합니다. 또한 자유로운 감지 구조로, 표준 솔버를 사용하여 초 빠른 카메라 외부 및 내부 파라미터 추정을 가능하게 합니다.

- **Performance Highlights**: FreeSplatter는 두 가지 모델 변형을 통해 객체 중심(object-centric) 및 장면(level) 복원 시나리오에서의 성능 개선을 입증했습니다. 실험 결과, FreeSplatter는 기존의 스파스 뷰 복원 방법에 비해 재구성 품질 및 포즈 추정 정확도에서 탁월한 성능을 보였으며, 특히 FreeSplatter-O는 기존의 포즈 의존적 대규모 복원 모델들에 비해 월등한 성능 차이를 보였습니다. 이 기술은 텍스트/이미지에서 3D 콘텐츠 생성과 같은 다운스트림 애플리케이션의 생산성을 향상시키는 데 도움을 줄 것으로 기대됩니다.



### Video Creation by Demonstration (https://arxiv.org/abs/2412.09551)
Comments:
          Project page at this https URL

- **What's New**: 이 논문에서는 비디오 생성의 새로운 경험으로서 '비디오 제작을 통한 시연(Video Creation by Demonstration)'을 제안합니다. 사용자는 시연 비디오와 특정 환경을 담고 있는 초기 장면 이미지를 제공함으로써 동작 개념을 자연스럽게 이어지는 비디오로 생성할 수 있습니다. 기존의 명시적 신호를 기반으로 하는 대부분의 비디오 생성 접근 방식과 달리, 우리는 암묵적인 잠재 조작 방식을 채택하여 일반 비디오에서 최대한의 유연성과 표현력을 확보합니다.

- **Technical Details**: 이 연구에서는 'δ-Diffusion'이라는 자기 지도 학습(self-supervised training) 접근 방식을 제안하며, 주어진 비디오의 동작 정보를 바탕으로 미래 프레임을 예측합니다. 두 단계의 훈련 과정으로 구성되어 있으며, 첫 번째 단계에서 시연 비디오의 시공간적 의미 표현을 추출하고, 두 번째 단계에서는 그 표현을 사용하여 현실적인 모션을 생성하는 데 필요한 동작 잠재 정보를 조정합니다. 이러한 접근법은 관련된 모든 이론적 기초를 제공하여 일반 비디오 생성에서의 새로운 능력을 개방합니다.

- **Performance Highlights**: 실험 결과, δ-Diffusion은 인간 선호도 및 대규모 기계 평가에서 관련 기준선을 초과하는 성능을 보여주었습니다. 비디오 생성 과정에서 더 나은 제어 가능성과 개념 전이 능력을 인정받았으며, 다양한 동작 개념을 포함한 고충실도의 비디오 생성을 가능하게 합니다. 우리는 또한 서로 결합된 여러 시연 비디오를 통해 연속적인 일관된 시퀀스를 생성할 수 있는 가능성을 보여주었으며, 이는 대화형 환경을 생성하는 대안으로 활용될 수 있습니다.



### Exemplar Masking for Multimodal Incremental Learning (https://arxiv.org/abs/2412.09549)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 다중 모드(class-incremental learning) 증거 replay의 효율성을 높이기 위해 exemplar masking 프레임워크를 제안합니다. 이는 중요하지 않은 토큰을 attention weights 기반으로 마스킹하여 저장 크기를 줄이고 기존 지식을 유지합니다. 또한, 다양한 모드를 통해 old knowledge를 효과적으로 재생하기 위한 데이터 증강 기법을 설계했습니다.

- **Technical Details**: 논문에서 제안된 exemplar masking 방법은 주요 토큰만을 유지하는 것으로, 각 클래스의 샘플에서 필요한 정보만 저장함으로써 메모리 사용량을 최소화합니다. 이 접근법은 기존의 iCaRL 방법을 기반으로 하지만, 각 데이터 유형의 특성을 고려하여 더 효율적인 exemplars 선택 및 저장을 가능하게 합니다. 또한, parameter-efficient tuning 기법을 활용하여 모델 학습 시의 부하를 경감합니다.

- **Performance Highlights**: 실험 결과, 제안된 exemplar masking 프레임워크는 기존의 baseline 방법들보다 더 효율적이고 견고하며, 제한된 메모리 환경에서도 학습 능력을 유지함을 보여주었습니다. 특히, ImageNet-R 데이터셋을 확장하여 생성된 캡션을 사용한 다중 모드 데이터셋에서도 유의미한 성과를 달성하였습니다.



### SimAvatar: Simulation-Ready Avatars with Layered Hair and Clothing (https://arxiv.org/abs/2412.09545)
Comments:
          Project website: this https URL

- **What's New**: SimAvatar는 텍스트 프롬프트로부터 시뮬레이션에 적합한 3D 인간 아바타를 생성하는 프레임워크입니다. 이 방법은 기존에 사용되는 아바타 생성 기법들이 가지는 머리카락과 의복의 복잡한 기하학적 구조 문제를 해결하여, 쉽게 애니메이션화할 수 있는 아바타를 생산할 수 있도록 합니다. 이 연구는 시뮬레이션을 염두에 두고 설계된 첫 번째 아바타 생성 방법론으로, 실제 이미지를 기반으로 한 이미지 디퓨전 모델의 사전 지식을 효과적으로 활용합니다.

- **Technical Details**: SimAvatar 프레임워크는 3D 가우시안을 사용하여 다양한 인체 부위를 나타내고, 텍스트 CONDITION된 3D 생성 모델을 통해 의상 메시, 신체 형태 및 머리카락을 생성합니다. 다음 단계에서는 물리 기반 시뮬레이터를 사용하여 의상 메시와 머리카락을 애니메이션 처리한 후, 이들 움직임을 3D 가우시안으로 전이합니다. 이러한 방식으로 생성된 아바타는 생동감 있는 질감과 사실적인 동적 모션을 특징으로 하며, 각 부위에 대한 애니메이션이 용이합니다.

- **Performance Highlights**: SimAvatar는 현재의 접근 방식들과 비교했을 때, 사실적인 외관과 동적인 모션을 가진 고도화된 품질의 아바타를 생성합니다. 이 방법론을 통해 많은 사용자들이 3D 아바타 생성 과정을 쉽게 수행할 수 있게 될 것으로 기대됩니다. 결과적으로, 생성된 아바타는 텍스처 및 동작에서 매우 뛰어난 품질을 제공하여 다양한 응용 프로그램에서 사용될 수 있습니다.



### Dynamic-VLM: Simple Dynamic Visual Token Compression for VideoLLM (https://arxiv.org/abs/2412.09530)
- **What's New**: 이 연구에서는 기존의 LVLMs에 부족한 비디오 데이터셋을 보완하기 위해 대규모 합성 데이터셋을 개발하였다. 이 합성 데이터셋은 특허 모델에서 생성되었으며, 다양한 질문을 해결하기 위한 프롬프트가 신중하게 설계되었다. 또한 비디오 길이에 따른 유동적인 비주얼 토큰 압축 구조를 탐구하며 계산 효율성과 성능 간의 균형을 맞추고자 하였다.

- **Technical Details**: 제안된 보편적인 비주얼 토큰 압축기는 다양한 길이의 비디오를 수용할 수 있도록 설계되었다. 기존의 고정된 토큰 수 대신 각 이미지에 대해 가변 수의 토큰을 활용하여 비디오의 다양한 길이를 처리하는 유연성을 제공한다. 이를 통해 Open-ended VideoQA, Multiple-choice VideoQA 및 다양한 Multi-image QA 태스크에서 Dynamic-VLM의 성능을 전반적으로 평가하였다.

- **Performance Highlights**: 실험 결과, \\model{}는 VideoMME에서 LLaVA-OneVision보다 2.7%의 절대적인 성능 향상을 달성하였으며, MuirBench에서는 10.7%의 향상 결과를 보였다. Dynamic-VLM은 다양한 비디오 태스크에서 최첨단 성과를 달성하고 뛰어난 일반화 능력을 보여주었다. 이러한 결과는 새로운 멀티 이미지 이해 기준을 설정함으로써, VideoLLM 분야에 긍정적인 영향을 미칠 것으로 예상된다.



### Can Modern LLMs Act as Agent Cores in Radiology~Environments? (https://arxiv.org/abs/2412.09529)
Comments:
          22 pages,7 figures

- **What's New**: 현재 대형 언어 모델(LLMs)의 발전으로 LLM 기반 에이전트 시스템이 다양한 분야에서 더 높은 정확성과 해석 가능성을 제공할 수 있게 되었습니다. 방사선학은 이러한 복잡한 분석 요구사항을 해결할 수 있는 이상적인 분야로, 본 논문은 현대의 LLM이 방사선학 환경에서 에이전트 핵심으로 작용할 수 있는지를 탐구합니다. 이를 위해 RadABench를 소개하고, 방사선학에서의 LLM 기반 에이전트 평가를 위한 포괄적인 데이터셋과 평가 플랫폼을 제안합니다.

- **Technical Details**: RadABench는 세 가지 주요 기여로 구성됩니다: 첫째, 6개의 해부학적 영역, 5개의 영상 방식, 10개의 도구 카테고리 및 11개의 방사선학 과제를 포함한 포괄적인 합성 평가 데이터셋인 RadABench-Data를 제시합니다. 둘째, LLM 기반 방사선학 에이전트를 평가하기 위한 새로운 플랫폼인 RadABench-EvalPlat을 개발하였습니다. 셋째, 7개의 최신 LLM의 성능을 5가지 관점에서 평가하여 데이터셋과 코드를 오픈 소스로 제공하며, 이 데이터는 방사선학의 복잡성을 반영하고 있습니다.

- **Performance Highlights**: 저자들은 이러한 평가를 통해 최신 LLM들이 일부 간단한 작업에서 뛰어난 성능을 보이는 반면, 더 복잡한 임상 시나리오에서는 상당한 성능 격차가 존재함을 발견했습니다. 이는 현재 LLM의 능력이 실제 방사선학 애플리케이션의 엄격한 요구를 충족하지 못하고 있음을 강조합니다. 연구 결과는 방사선학에서의 에이전트 기반 시스템의 발전을 가속화하기 위한 통찰을 제공하며, 에이전트 시스템을 실제 임상에서 효과적으로 적용하는 방법에 대한 지침을 제시합니다.



### Efficient and Comprehensive Feature Extraction in Large Vision-Language Model for Clinical Pathology Analysis (https://arxiv.org/abs/2412.09521)
- **What's New**: 이 논문에서는 병리 진단의 효율성과 정확성을 높이기 위해 두 가지 혁신적인 전략을 제안합니다. 첫 번째는 mixed task-guided feature enhancement (MTGFE)로, 이는 병변과 관련된 세부 사항을 다양한 스케일에서 효과적으로 추출하도록 기능을 안내합니다. 두 번째는 prompt-guided detail feature completion (PGDFC)로, 특정 프롬프트에 기반하여 WSI에서 세밀한 특징을 통합하여 추론 속도를 유지하면서도 진단 정보의 손실을 방지합니다.

- **Technical Details**: 고해상도 전 슬라이드 이미지 (WSI)를 기반으로 한 종합 데이터세트를 사용하여 490,000개의 샘플을 수집하였으며, 이를 통해 병리학 특화 대형 비전 언어 모델 (LVLM)인 OmniPath를 훈련시켰습니다. MTGFE와 PGDFC 전략을 통해 모델이 병변 세부 정보를 보다 잘 인식하고, 다양한 병리 진단 분석 작업에서 필요한 멀티스케일 특징을 완벽하게 지원할 수 있게 됩니다. 이 과정에서, 특정 병리 개념 탐지를 위한 지침 데이터도 모델에 추가되어 정확성을 높였습니다.

- **Performance Highlights**: OmniPath는 기존의 병리 LVLM보다 진단 정확도 및 효율성에서 현저히 우수한 성과를 보여주었습니다. 이 모델은 다양한 병리 응용 분야에서 보조 진단을 위한 상호 작용적이고 임상에 적합한 접근 방식을 제공합니다. 실험 결과는 OmniPath가 병리학의 다양한 진단 작업에서 기존 방법들을 뛰어넘는 성능을 보였음을 입증하며, 임상 병리 진단의 실제 요구에 더 적합한 솔루션으로 자리매김하였습니다.



### Agent-based Video Trimming (https://arxiv.org/abs/2412.09513)
- **What's New**: 이 논문에서는 사용자가 생성한 긴 비디오를 효과적으로 요약하고 정리를 위한 새로운 작업인 Video Trimming (VT)를 소개합니다. 기존에는 특정 시간 구간을 선택하는 데 초점을 두었지만, VT는 불필요한 영상을 필터링하고 의미 있는 세그먼트를 선별하여 논리적인 이야기 구성으로 변환하는 데 중점을 둡니다. 이를 통해 비디오의 관계성과 일관성을 고려한 혁신적인 접근을 제시합니다. 본 연구에서 제안하는 Agent-based Video Trimming (AVT) 알고리즘은 이러한 VT 작업을 효과적으로 수행할 수 있도록 세 가지 단계로 설계되었습니다.

- **Technical Details**: AVT는 Video Structuring, Clip Filtering, Story Composition의 세 단계를 포함합니다. Video Structuring 단계에서는 비디오를 작은 단위로 나누고, Video Captioning Agent가 이를 구조화된 텍스트 설명으로 변환하여 각 세그먼트에 대한 세부적인 의미 분석을 가능하게 합니다. Clip Filtering 단계에서는 저품질의 클립을 동적으로 필터링하여 유용한 클립을 선택하고, Story Composition 단계에서는 선택된 클립들을 논리적이고 일관된 최종 비디오로 조합합니다.

- **Performance Highlights**: 최신 벤치마크 데이터셋을 사용한 평가 결과, AVT는 사용자 연구를 통해 긍정적인 평가를 받았으며, YouTube Highlights, TVSum 및 본 데이터셋에서 highlight detection 작업에서 뛰어난 mAP 및 precision 성능을 나타냈습니다. 또한, 인간 평가와 함께 진행된 비디오 평가를 통해 AVT의 효과성을 입증하였으며, 새로운 비디오 트리밍 벤치마크를 구축하여 평가의 신뢰성을 높였습니다.



### GEAL: Generalizable 3D Affordance Learning with Cross-Modal Consistency (https://arxiv.org/abs/2412.09511)
Comments:
          22 pages, 8 figures, 12 tables; Project Page at this https URL

- **What's New**: 이 논문에서는 GEAL이라는 새로운 프레임워크를 통해 3D affordance learning의 일반화 및 강인성을 향상시키고자 합니다. 기존의 메소드들은 한정된 주석 데이터로 인해 일반화에 어려움을 겪었지만, GEAL은 대규모로 사전 훈련된 2D 모델을 활용하여 이러한 문제를 해결합니다. 또한, 3D 포인트 클라우드와 2D 표현 간의 일관된 매핑을 확립하기 위해 Gaussian splatting(가우시안 스플래팅) 기법을 사용합니다.

- **Technical Details**: GEAL은 듀얼 브랜치 아키텍처를 채택하여 3D 데이터와 2D 데이터 간의 상호작용을 극대화합니다. 가변적 크기의 시각 및 텍스트 특징을 통합할 수 있는 granularity-adaptive fusion module(변화성 적응형 융합 모듈)과 2D-3D 일관성 정렬 모듈을 통해 다중 모달 간의 지식 전이 및 정렬을 보장합니다. 이 프레임워크는 또한 PIAD-C 및 LASO-C라는 두 가지 새로운 파손 기반 벤치마크를 제공합니다.

- **Performance Highlights**: GEAL은 다양한 표준 및 부패 기반 벤치마크에서 강력한 성능을 입증하였습니다. 실험 결과, GEAL은 이미 알려진 데이터에서 새로운 데이터를 효과적으로 전이하며, 파손된 데이터에서도 높은 성능을 유지하는 것으로 나타났습니다. 이로 인해 GEAL은 다양한 조건에서도 강력하고 적응력 있는 affordance 예측능력을 보여줍니다.



### Vision Transformers for Efficient Indoor Pathloss Radio Map Prediction (https://arxiv.org/abs/2412.09507)
Comments:
          Work partly supported by the RA Science Committee grant No. 22rl-052 (DISTAL) and the EU under Italian National Recovery and Resilience Plan of NextGenerationEU on "Telecommunications of the Future" (PE00000001 - program "RESTART")

- **What's New**: 이 연구는 Vision Transformers (ViTs)를 활용하여 실내 경로 손실(Pathloss) 라디오 맵 예측 문제를 해결하는 새로운 접근 방식을 제시합니다. 기존의 방법론인 합성곱 신경망(CNN)에서 비전 트랜스포머로 전환함으로써 더 복잡한 실내 환경에서의 모델 일반화 능력을 평가하고자 합니다. ICASSP 2025에서 개최되는 제1회 실내 경로 손실 라디오 맵 예측 챌린지를 위해 특별히 설계된 연구입니다.

- **Technical Details**: 연구에 사용된 데이터셋은 레이 트레이싱(ray-tracing) 알고리즘을 기반으로 생성된 경로 손실 라디오 맵으로 구성됩니다. 이 데이터셋은 25개의 다양한 실내 구조와 3개의 주파수 대역, 5개의 안테나 방사 패턴을 포함하여, 신경망의 일반화 능력을 평가하는 기준이 됩니다. 입력 데이터는 경로 손실을 효과적으로 나타내기 위해 두 가지 투명도 채널과 거리 정보를 포함한 3채널을 사용합니다.

- **Performance Highlights**: 데이터 증강(data augmentation) 기법을 통해 훈련 데이터를 효과적으로 증가시킴으로써 모델의 일반화 능력을 강화합니다. 이 연구는 MixUp 및 회전 증강(rotation augmentation) 기법을 통해 입력 데이터의 다양성을 증대시키고, 실시간 환경을 보다 잘 반영할 수 있는 기술적 기반을 제공합니다. 결과적으로, 다양한 시나리오에서도 강력한 성능을 발휘할 수 있는 모델 개발에 성공하였습니다.



### Lyra: An Efficient and Speech-Centric Framework for Omni-Cognition (https://arxiv.org/abs/2412.09501)
Comments:
          Tech report

- **What's New**: 이번 논문은 Lyra라는 새로운 Multi-modal Large Language Model(MLLM)을 소개합니다. 기존의 omni-model들이 음성을 소홀히 다뤘던 반면, Lyra는 멀티모달 능력을 향상시키고, 긴 음성 이해와 소리 인식, 그리고 원활한 음성 상호작용을 지원합니다. 또한 효율성을 위해 데이터 요구 사항을 줄이고, 훈련 비용을 최소화하는 여러 전략을 채택했습니다.

- **Technical Details**: Lyra는 세 가지 주요 전략을 통해 효율성과 음성 중심의 능력을 구현합니다. 첫째, 기존의 오픈소스 대형 모델을 활용하여 최소한의 훈련 데이터로 이러한 모델의 강점을 보존합니다. 둘째, 잠재적 다중 모달 정규화기(latent multi-modality regularizer)와 추출기(latent multi-modality extractor)를 사용하여 음성과 다른 모달리티 간의 관계를 강화합니다. 셋째, 1.5M 멀티모달 데이터 샘플과 12K 긴 스피치 샘플을 포함하는 고품질 데이터셋을 구축하여 Lyra의 처리 성능을 향상시킵니다.

- **Performance Highlights**: Lyra는 다양한 비전-언어, 비전-스피치, 스피치-언어 벤치마크에서 최첨단 성능을 달성했습니다. 특히, 다른 모델에 비해 적은 계산 자원과 훈련 데이터로 강력한 Omni-comprehension 능력을 보여줍니다. Lyra의 이러한 특성은 사용자의 다양한 요구에 부합하며, 효율적이고 강력한 다중 모달 인터랙션을 가능하게 합니다.



### New keypoint-based approach for recognising British Sign Language (BSL) from sequences (https://arxiv.org/abs/2412.09475)
Comments:
          International Conference on Computer Vision (ICCV) - HANDS Workshop

- **What's New**: 이 논문은 British Sign Language (BSL) 단어를 인식하기 위해 설계된 새로운 keypoint 기반의 분류 모델을 소개합니다. BOBSL 데이터셋을 사용하여 성능을 평가한 결과, keypoint 기반 접근 방식이 RGB 기반 모델보다 계산 효율성과 메모리 사용 측면에서 뛰어난 성능을 보였습니다. 이 모델은 기존의 연구와 비교할 때, BSL 단어 분류를 위한 keypoint 기반 모델의 최초 적용이라고 할 수 있습니다.

- **Technical Details**: 본 연구에서는 2D keypoints를 사용하여 얼굴, 오른손, 왼손 및 자세의 특정 지점을 표현합니다. 이러한 접근 방식의 주요 장점으로는 정보 제어가 용이하고, 조명과 의류와 같은 불필요한 요소를 제거할 수 있으며, 프레임 속도로 키포인트를 계산할 수 있어 실시간 모델 실행이 가능하다는 점이 있습니다. Mediapipe를 사용하여 실시간으로 키포인트를 추출하고, Transformer-based 모델에 입력으로 제공하여 BSL 단어를 인식합니다.

- **Performance Highlights**: 우리의 모델은 keypoint 기반 접근 방식이 RGB 기반 모델보다 뛰어난 계산 효율성과 메모리 사용, 훈련 속도를 나타낸다는 것을 보여줍니다. BOBSL 데이터셋에서는 총 8,162개의 단어 카테고리를 갖는 분류 문제가 발생하였으며, 여기에 대한 훈련의 결과로 모델의 성능이 향상되었음을 확인하였습니다. 다양한 메소드와 keypoint 수에 대한 실험을 통해 모델의 적합성을 평가했습니다.



### OFTSR: One-Step Flow for Image Super-Resolution with Tunable Fidelity-Realism Trade-offs (https://arxiv.org/abs/2412.09465)
- **What's New**: 최근 확산(diffusion) 및 흐름(flow) 기반 생성 모델은 이미지 복원(image restoration) 작업에서 기존의 딥러닝 방법들보다 뛰어난 성능을 보여주었습니다. 그러나 이러한 방법들은 고품질 이미지를 생성하기 위해 많은 샘플링 단계(sampling steps)를 필요로 하여 계산 오버헤드(computational overhead)가 크거나, 고정된 신뢰도-현실성(fidelity-realism) 균형을 요구하는 모델 증류(model distillation)에 의존합니다. 본 논문에서는 OFTSR을 소개하며, 이는 신뢰도와 현실성을 조정 가능한 한 단계 이미지 초해상도(one-step image super-resolution)를 생성하는 새로운 흐름 기반(framework) 프레임워크입니다.

- **Technical Details**: OFTSR은 두 단계의 훈련 파이프라인을 채택합니다. 첫 번째 단계에서는 조건부 흐름 기반(super-resolution model)을 훈련하여 교사 모델(teacher model)로 사용합니다. 두 번째 단계에서는 특수한 제약을 적용하여 학생 모델(student model)로 하여금 교사 모델의 동일한 샘플링 ODE 궤적(sampling ODE trajectory)에서 예측하도록 강제합니다. 이 정렬(alignment) 과정은 학생 모델의 단일 단계 예측이 교사의 예측과 일치하도록 보장합니다.

- **Performance Highlights**: 다양한 도전적인 데이터셋(FFHQ, DIV2K, ImageNet)을 통해 OFTSR이 기존의 다양한 방법들과 비교해 한 단계 초해상도 작업에서 최첨단 성능(state-of-the-art performance)을 달성하는 것을 입증했습니다. OFTSR은 틀에 박히지 않은 방식으로 신뢰도-현실성 균형을 유연하게 조정할 수 있는 능력을 가지고 있습니다. 코드와 사전 훈련된 모델은 해당 링크에서 제공됩니다.



### ATPrompt: Textual Prompt Learning with Embedded Attributes (https://arxiv.org/abs/2412.09442)
Comments:
          Technical Report. Project Page: this https URL

- **What's New**: 이번 연구에서는 비전-언어 모델(VLMs) 간 이미지를 정체불명의 카테고리와 효과적으로 연관 지을 수 있는 새로운 접근법인 ATPrompt를 제안합니다. 기존의 소프트 프롬프트 방식에서 한계를 나타내던 점을 보완하여, 다차원 속성 레벨로 학습 공간을 확장하는 방법을 소개하고 있습니다. 이 방법은 보편적인 속성 토큰을 통합하여 소프트 프롬프트가 카테고리 특정 뿐만 아니라 속성과 관련된 일반 표현을 학습하도록 유도합니다.

- **Technical Details**: ATPrompt는 소프트 프롬프트가 고정된 보편적 속성 토큰과 결합되어 기존의 원차원 카테고리 수준에서 다차원 속성 수준으로 이동할 수 있도록 설계되었습니다. 이 방식은 특히 낯선 카테고리에 대한 이미지를 더 잘 정렬할 수 있도록 도와줍니다. 또한, LLM을 활용하여 후보 풀에서 적합한 속성을 학습하고 선택하는 차별화 가능한 속성 검색 방법을 도입하여, 모델 훈련 과정에서 정확한 속성 조합을 사용할 수 있게 합니다.

- **Performance Highlights**: 11개의 다양한 데이터셋에 대한 실험을 통해 ATPrompt가 기존의 텍스트 기반 방법들과 원활하게 통합되며, 추가적인 계산 비용이 거의 없이 성능 향상을 이룰 수 있음을 보여주었습니다. 이는 이미지를 정체불명의 카테고리와 더 잘 연관 지을 수 있는 가능성을 제시하며, VLMs의 적용 범위를 넓힐 수 있는 새로운 방향을 제공합니다.



### Towards Robust and Fair Vision Learning in Open-World Environments (https://arxiv.org/abs/2412.09439)
Comments:
          PhD Dissertation

- **What's New**: 이번 논문은 비전 학습에서의 공정성(fairness)과 강건성(robustness)을 향상시키기 위한 네 가지 주요 기여를 제시합니다. 첫째, 대규모 데이터 요구 문제를 해결하기 위해 새롭고 혁신적인 Fairness Domain Adaptation 접근법을 소개합니다. 둘째, 개방형 세계(open-world) 모델링이 가능한 비전 학습을 위해 Open-world Fairness Continual Learning Framework를 제안합니다.

- **Technical Details**: 이 연구의 중요한 구성 요소는 Bijective Maximum Likelihood와 Fairness Adaptation Learning에서 파생된 새로운 연구 발견들입니다. 또한 시각 데이터를 다양한 카메라 뷰에서 수집되기 때문에, 이 논문에서는 Geometry-based Cross-view Adaptation 프레임워크를 통해 뷰 간에 강건한 특징 표현을 학습하는 방법을 제시합니다. 마지막으로, Transformer 기반의 접근법을 사용하여 대규모 영상 및 다중 모드 데이터에 대한 강건한 특징 표현을 개선합니다.

- **Performance Highlights**: 이 논문의 이론적 분석 및 실험 결과는 제안된 접근법의 효과성을 입증하며, 기존 연구들에 비해 뛰어난 성능을 보여줍니다. 연구의 기여는 기계 비전 학습의 공정성과 강건성을 한층 더 발전시켰습니다. 이러한 결과는 비전 학습 분야에서 새로운 가능성을 열어 줄 것으로 기대됩니다.



### Multimodal Music Generation with Explicit Bridges and Retrieval Augmentation (https://arxiv.org/abs/2412.09428)
- **What's New**: 본 논문은 다양한 입력 방식(모달리티)으로부터 음악을 생성하는 다중모달 음악 생성 기술에 대해 다룹니다. 특히, 텍스트와 음악을 명확한 연결 고리로 활용하여 모달리티 정렬의 문제를 해결하는 새로운 방법인 Visuals Music Bridge (VMB)를 소개합니다. VMB는 시각적 입력을 텍스트 설명으로 변환하여 음악 생성을 돕는 Multimodal Music Description Model과, 음악 조각을 넓고 특정하게 검색할 수 있는 Dual-track Music Retrieval 모듈을 포함하고 있습니다.

- **Technical Details**: VMB는 다중 모달 입력을 처리하기 위해 세 가지 핵심 구성요소로 구성됩니다. 첫째, Multimodal Music Description Model은 영상 입력을 음악 설명으로 변환합니다. 둘째, Dual-track Music Retrieval 모듈은 감정 및 주제 정렬을 위한 광범위 검색과 특정 악기와 장르 조정을 위한 목표 검색 전략을 사용합니다. 마지막으로, Explicitly Conditioned Music Generation 프레임워크는 위 두 다리를 활용하여 최종 음악을 생성합니다.

- **Performance Highlights**: 실험 결과, VMB는 기존 방식들에 비해 음악의 품질을 크게 향상시켰으며, 입력 모달리티와 생성된 음악 간의 정렬을 개선하는 데 기여했습니다. 또한, 사용자 맞춤형 음악 생성이 가능하여 뛰어난 조작성을 제공합니다. VMB는 다양한 멀티미디어 분야에서 적용 가능한 해석 가능하고 표현력이 풍부한 다중모달 음악 생성의 새로운 기준을 설정했습니다.



### MultiEYE: Dataset and Benchmark for OCT-Enhanced Retinal Disease Recognition from Fundus Images (https://arxiv.org/abs/2412.09402)
Comments:
          Accepted at IEEE TMI

- **What's New**: 이 연구는 "OCT-enhanced disease recognition from fundus images"라는 새로운 설정을 제안하여, 교육 단계에서 비쌍(multimodal unpaired) 데이터를 사용하고, 시험 단계에서는 널리 사용되는 fundus 사진을 활용할 수 있도록 합니다. 본 연구의 주요 기여 중 하나는 눈 질환 진단을 위한 대규모 데이터셋인 MultiEYE를 구축한 것입니다. 이 데이터셋은 fundus 및 OCT 이미지를 포함하며, 두 이미지는 반드시 동일한 환자에서 비롯되지 않아도 됩니다.

- **Technical Details**: 연구진은 OCT-CoDA(Conceptual Distillation Approach)를 제안하였으며, 이는 OCT 이미지를 통해 질병 관련 지식을 추출하고 이를 fundus 모델에 결합하는 방식입니다. 이 방법에서는 특정 질병의 증명을 설명하는 개념을 생성하여 지식을 전달하는 데 있어 강력한 언어 모델과 비전 언어 모델을 사용합니다. 지식 증류 과정에서는 전역 프로토타입 증류와 지역 대조 증류의 두 가지 유형의 손실 함수를 도입하여, 질병 수준에서 개념 유사성을 전이합니다.

- **Performance Highlights**: MultiEYE 데이터셋을 기반으로 한 광범위한 실험을 통해, OCT-CoDA는 단일 모드(single modal) 및 다중 모드(multi-modal) 네트워크보다 뛰어난 효율성과 설명 가능성을 보여줍니다. 연구 결과는 fundus 모델의 진단 성능을 크게 향상시키며 임상 응용 가능성을 증명합니다. 이렇게 개선된 성능은 fundus 이미지만으로도 높은 진단 정확도를 유지할 수 있음을 나타냅니다.



### SLAM3R: Real-Time Dense Scene Reconstruction from Monocular RGB Videos (https://arxiv.org/abs/2412.09401)
- **What's New**: 본 논문에서는 SLAM3R라는 새로운 단안 RGB SLAM 시스템을 소개합니다. SLAM3R는 실시간 밀집 3D 재구성을 위한 효과적인 솔루션으로, 피드포워드 신경망을 통해 로컬 3D 재구성과 글로벌 좌표 등록을 통합합니다. 이 시스템은 입력 비디오를 슬라이딩 윈도우 메커니즘을 사용하여 겹치는 클립으로 변환하고, RGB 이미지에서 직접 3D 포인트 맵을 회귀하여 전 세계적으로 일관된 장면 재구성을 진행합니다.

- **Technical Details**: SLAM3R는 두 가지 계층 구조로 구성되어 있으며, 첫 번째로 입력 비디오의 짧은 클립에서 로컬 3D 기하 구조를 재구성합니다. 두 번째로, 이 로컬 재구성들을 점진적으로 등록하여 전 세계적으로 일관된 3D 장면을 생성합니다. 특히, SLAM3R는 별도의 카메라 매개변수 추정 없이 3D 포인트를 재구성할 수 있도록 설계되었습니다.

- **Performance Highlights**: SLAM3R는 여러 벤치마크에서 기존 밀집 SLAM 시스템보다 우수한 성능을 보이며, 20 FPS 이상의 속도로 고품질 장면 재구성을 실현합니다. 이 시스템은 재구성의 정확도와 완전성을 모두 보장하면서도 실시간으로 처리할 수 있는 혁신적인 접근 방식을 제공합니다.



### UFO: Enhancing Diffusion-Based Video Generation with a Uniform Frame Organizer (https://arxiv.org/abs/2412.09389)
Comments:
          Code:this https URL

- **What's New**: 최근 확산 기반의 비디오 생성 모델에서 중요한 진전을 이루었지만, 기존 모델들은 일관성과 이미지 품질 저하 등의 문제를 겪고 있습니다. 이를 해결하기 위해 새로운 비침습적 플러그인인 Uniform Frame Organizer (UFO)를 제안합니다. UFO는 모든 확산 기반 비디오 생성 모델과 호환 가능하며, 동적 요소와 정적 정확성을 유지하며 향상된 일관성을 제공합니다.

- **Technical Details**: UFO는 조정 가능한 강도를 가지는 일련의 적응형 어댑터로 구성되어 있습니다. 이 어댑터는 영상의 배경과 전경 간의 일관성을 크게 높이며, 모델의 원래 매개변수를 변경하지 않고도 이미지 품질을 개선할 수 있도록 설계되었습니다. UFO는 훈련 데이터로 적은 양의 비디오 프레임이나 이미지를 사용할 때 강도를 최대치로 설정하여 모델 출력이 정적 비디오에 가까워지도록 합니다.

- **Performance Highlights**: UFO는 공공 비디오 생성 벤치마크 Vbench에서 비디오 일관성과 품질을 크게 향상시키는 것으로 입증되었습니다. 사용자는 매우 적은 리소스와 비용으로 UFO를 훈련할 수 있으며, 비디오-텍스트 쌍 없이도 일관성을 높일 수 있습니다. 이러한 효율성 덕분에 사용자는 고품질 비디오 생성의 부담을 크게 줄일 수 있습니다.



### All You Need in Knowledge Distillation Is a Tailored Coordinate System (https://arxiv.org/abs/2412.09388)
- **What's New**: 이번 논문에서는 Knowledge Distillation (KD)의 새로운 접근 방식인 Tailored Coordinate System (TCS)을 제안합니다. TCS는 기존의 KD 방법이 요구하는 작업 특화된 teacher 모델 없이도 스스로 선행 학습된(self-supervised learning, SSL) 모델의 어두운 지식을 효과적으로 추출할 수 있도록 설계되었습니다. 이 방법은 하나의 포워드 패스를 통해 teacher의 feature를 기반으로 좌표계를 형성하여 student 네트워크를 최적화할 수 있게 합니다.

- **Technical Details**: TCS는 주로 Principal Component Analysis (PCA)를 이용하여 teacher의 feature에서 좌표계를 생성합니다. 이 과정에서 데이터 증강을 필요로 하지 않아 오직 하나의 포워드 패스만으로도 비용을 최소화할 수 있습니다. 이후 student의 feature를 이 좌표계에 맞추어 조정하는 과정은 iterative feature selection 기법에 의해 지원되며, 이는 TCS의 효율성을 높이는 데 기여합니다.

- **Performance Highlights**: TCS 방법은 최신 KD 기술에 비해 훨씬 더 높은 정확도를 보여주면서도 훈련 시간과 GPU 메모리 비용을 약 절반으로 줄일 수 있음을 실험적으로 입증하였습니다. 이 방식은 다양한 아키텍처에 적용 가능하며, few-shot learning을 포함한 다양한 작업에서도 효과적으로 작동합니다. TCS는 교육받은 teacher 없이도 학생 네트워크를 초고속으로 훈련할 수 있는 가능성을 보여줍니다.



### Causal Graphical Models for Vision-Language Compositional Understanding (https://arxiv.org/abs/2412.09353)
- **What's New**: 최근 연구에 따르면 Vision-Language Models (VLMs)는 인간 언어의 조합적 속성을 완전히 이해하는 데 어려움을 겪고 있으며, 이는 일반적으로 이미지 캡션을 "bag of words"로 모델링하고 있기 때문입니다. 이 연구에서는 Causal Graphical Model (CGM)을 사용하여 텍스트 및 시각적 토큰 간의 의존 관계를 모델링하고, VLM 비주얼 인코더에 의해 조건화된 디코더를 훈련하는 방식을 제안합니다. 우리의 접근 방식은 조합 작업에서 VLM의 성능을 크게 향상시키는 데 기여합니다.

- **Technical Details**: 이 연구에서 제안한 방법은 전통적인 순차적(autoregressive) 예측 대신 반순차적(semi-parallel) 예측 전략을 사용하여 CGM 구조를 따릅니다. 이를 통해 디코더는 문장 내 주요 인과적 의존 관계만 학습하고, 불필요한 상관관계를 배제할 수 있습니다. 특히, 의존 파서를 통해 작성된 의존 트리를 기반으로 CGM을 구축하여 이미지 패치와 텍스트 토큰 간의 의존 관계를 설명합니다.

- **Performance Highlights**: 다양한 실험을 통해 제안된 방법이 기존의 모든 최첨단 조합 접근 방식을 상당히 능가함을 보여주며, 평가된 모든 벤치마크에서 새로운 최첨단 성과를 기록합니다. 또한, 훨씬 적은 데이터로 훈련되었음에도 불구하고 Cap 및 CapPa 방식에서도 성능 개선을 보여줍니다.



### DisPose: Disentangling Pose Guidance for Controllable Human Image Animation (https://arxiv.org/abs/2412.09349)
- **What's New**: 이번 연구에서는 DisPose라는 모듈을 제안하여 희소한 스켈레톤 포즈를 모션 필드 가이드와 키포인트 대응 관계로 분리합니다. 이 방법은 추가적인 밀도 높은 입력 없이 더 일반적이고 효과적인 제어 신호를 발굴하여 생성된 비디오의 품질을 향상시킵니다.

- **Technical Details**: DisPose는 희소한 모션 필드를 기반으로 한 밀도 높은 모션 필드를 생성하여 지역 레벨의 밀도 높은 가이드를 제공합니다. 또한, 참조 이미지에서 추출한 확산 특징을 목표 포즈에 전달하여 독특한 정체성을 부여합니다. 이러한 처리 과정은 ControlNet와 유사한 구조로 기존 모델에 통합되어 영상의 질과 일관성을 개선합니다.

- **Performance Highlights**: 광범위한 실험 결과, DisPose가 기존 방법들보다 우수한 성능을 보이는 것을 입증했습니다. 이 연구는 복잡한 모션을 처리하면서도 스켈레톤 포즈의 키포인트를 고려하여 외형 일관성을 유지하면서도 제어 신호의 효율성을 극대화하는 새로운 접근 방식을 제공합니다.



### MaskTerial: A Foundation Model for Automated 2D Material Flake Detection (https://arxiv.org/abs/2412.09333)
Comments:
          9 pages, 5 figures

- **What's New**: 이 논문에서는 Optical microscope 이미지에서 외부로 박리된 2D 물질 조각을 탐지하고 분류할 수 있는 Deep Learning 모델인 MaskTerial을 제안합니다. 이 모델은 Instance Segmentation Network를 사용하여 2D 물질 조각을 신뢰성 있게 식별하며, Synthetic Data Generator를 통해 합성 데이터에 대한 대규모 사전 훈련을 수행합니다. 특히, 저대비(low-contrast) 재료에 대한 탐지 성능을 개선하여, 최소 5~10개의 이미지로도 새로운 재료에 빠르게 적응할 수 있습니다.

- **Technical Details**: MaskTerial의 아키텍처는 두 개의 Deep Learning 모델을 결합합니다. 첫 번째 모델인 Instance Prediction Model은 입력 이미지에서 관심 있는 모든 조각을 예측하고, 두 번째 모델인 Arbitrary Mixture Model은 조각의 대비에 따라 조각의 클래스를 할당합니다. 이 구조는 새로운 재료를 추가할 때 후자의 모델만 재훈련하면 되어 효율성을 높입니다.

- **Performance Highlights**: 이 모델은 5가지 서로 다른 2D 재료를 포함하는 8가지 데이터셋에서 평가되었으며, 기존 기술에 비해 저대비 재료인 육각 붕소 질화물(hexagonal boron nitride)을 탐지하는 데에서 현저한 개선을 보였습니다. 특히, Gaussian 기반의 클래스 조건부 모델을 통해 불확실성 추정을 통해 보다 신뢰성 있는 예측을 제공하며, 모형의 해석 가능성을 높이는 점이 유의미합니다.



### Are Conditional Latent Diffusion Models Effective for Image Restoration? (https://arxiv.org/abs/2412.09324)
Comments:
          16 pages, 12 figures, submitted to IEEE / CVF Computer Vision and Pattern Recognition Conference (CVPR 2025)

- **What's New**: 최근 이미지 복원(Image Restoration, IR) 분야에서 Conditional Latent Diffusion Models (CLDMs)의 활용이 증가하고 있습니다. 그러나 본 연구에서는 CLDM이 IR 작업에 효과적이지 않을 수 있음을 제기하고, 이들 모델이 저수준 표현(low-level representation)을 사용하여 손상된 이미지와 실제 이미지의 관계를 모델링하는 데 어려움을 겪고 있음을 보여줍니다. 특히, CLDM 모델은 시각적 세부 사항을 충실히 복원하는 데 한계가 있음을 실험을 통해 입증합니다.

- **Technical Details**: 이미지 복원은 손상된 이미지 I_y로부터 실제 이미지 I_x를 복구하는 클래식한 역문제(inverse problem)입니다. 본 논문에서는 CLDM을 기반으로 한 이미지 복원 파이프라인을 설명하며, 기존의 복원 모델이 저품질 이미지에서 노이즈와 압축 아티팩트를 완화하는 첫 번째 단계로 활용됩니다. CLDM은 처음 복원된 이미지 I_reg에서 시작하여, 잠재 공간(latent space)에서의 작업을 통해 정보 손실을 줄이려는 목표를 가지고 있습니다.

- **Performance Highlights**: 실험 결과, CLDM 기반 모델은 확장성(scalaibility)에서는 장점을 보이지만, 작은 손상이 있는 경우에는 전통적인 방식보다 높은 왜곡과 의미적 편차(semantic deviation)를 보이는 것으로 나타났습니다. 이러한 성과는 CLDM 모델의 복원 능력이 낮은 세부 사항의 복원에 미흡하다는 점을 강조합니다. 이는 향후 CLDM 기반의 이미지 복원 솔루션에 대한 재검토 및 새로운 평가 지표의 개발 필요성을 시사합니다.



### T-SVG: Text-Driven Stereoscopic Video Generation (https://arxiv.org/abs/2412.09323)
Comments:
          5 pages, 4 figures

- **What's New**: 스테레오 비디오(Stereoscopic Video)의 제작은 기술적으로 복잡하여 도전 과제가 많습니다. 이를 해결하기 위해 제안된 T-SVG(Text-driven Stereoscopic Video Generation) 시스템은 사용자 친화적인 zero-shot 방법론을 통해 텍스트 프롬프트를 기반으로 참조 비디오를 생성합니다. 이 방법은 3D 포인트 클라우드(3D Point Cloud)로 전환되어 자연스러운 입체 시각 효과를 달성합니다.

- **Technical Details**: T-SVG 시스템은 텍스트 입력을 활용하여 비디오 쌍을 생성하고, 깊이 추정(Depth Estimation) 및 비디오 인페인팅(Video Inpainting) 기법을 적용하여 고품질의 결과물을 제공합니다. 초기 비디오는 RGB-D 이미지로 변환되고, 이들을 3D 점 구름 모델로 변환하여 인간의 두 눈의 시각을 모방합니다. 이후 좌우 시점의 변환 행렬을 적용하여 스테레오 효과가 있는 비디오를 생성합니다.

- **Performance Highlights**: T-SVG는 최첨단의 훈련이 필요 없는 기법을 통합하여 스테레오 콘텐츠 제작의 효율성을 높입니다. 시스템의 유연한 아키텍처는 최신 모델로의 원활한 업데이트를 가능하게 하여, 비전문가도 쉽게 활용할 수 있는 환경을 조성합니다. 이러한 점에서 T-SVG는 스테레오 비디오 제작의 접근성을 크게 향상시키며, 향후 추가적인 응용 가능성을 제시합니다.



### FAMNet: Frequency-aware Matching Network for Cross-domain Few-shot Medical Image Segmentation (https://arxiv.org/abs/2412.09319)
Comments:
          Accepted by the 39th Annual AAAI Conference on Artificial Intelligence (AAAI-25)

- **What's New**: 본 논문에서는 기존의 적은 샘플을 기반으로 한 의료 이미지 세분화(Few-Shot Medical Image Segmentation, FSMIS)가 가진 한계를 극복하기 위해 교차 도메인 적은 샘플 의료 이미지 세분화(Cross-Domain Few-Shot Medical Image Segmentation, CD-FSMIS)를 제안합니다. 특히 의료 영상의 다양한 특성으로 인해 발생하는 도메인 변화 문제를 해결하기 위해 주파수 기반의 특성을 활용한 Frequency-aware Matching Network(FAMNet)를 개발했습니다. FAMNet은 두 가지 주요 모듈인 Frequency-aware Matching(FAM) 모듈과 Multi-Spectral Fusion(MSF) 모듈로 구성되어 있습니다.

- **Technical Details**: 제안된 모델은 FAM 모듈을 통해 지원-쿼리 간의 매칭을 수행하여 내부 도메인와 외부 도메인 간의 변동성을 감소시킵니다. FAM 모듈은 고주파 및 저주파 대역에서 발생하는 변화에 대응하고, 동시에 중주파 대역의 유사성을 활용하여 보다 견고한 세분화 성능을 제공합니다. 이어서 MSF 모듈은 FAM 모듈에서 분리된 특성을 융합하여 도메인 변형 정보를 억제하며, 모델의 일반화 능력을 강화합니다.

- **Performance Highlights**: 제안된 FAMNet은 세 가지 교차 도메인 데이터셋에서 기존의 FSMIS 모델들보다 우수한 성능을 보여주며, 최신 기술 수준의 성능을 달성했습니다. 수차례의 ablation study와 시각화를 통해 모델의 효과성과 우수성이 더욱 확인되었습니다. FAMNet은 적은 수의 주석 샘플로 새로운 클래스의 세분화를 가능하게 하여 의료 이미지 세분화 분야의 발전에 기여할 것으로 기대됩니다.



### Advancing Attribution-Based Neural Network Explainability through Relative Absolute Magnitude Layer-Wise Relevance Propagation and Multi-Component Evaluation (https://arxiv.org/abs/2412.09311)
Comments:
          30 pages, 16 figures, 13 tables, ACM Transactions on Intelligence Systems and Technology

- **What's New**: 최근 딥 뉴럴 네트워크(Deep Neural Networks) 기술의 발전으로 인해 설명 가능성과 투명성이 중요한 다양한 분야에서 블랙박스 모델의 사용이 제한되고 있습니다. 본 논문은 기존의 Layer-Wise Relevance Propagation (LRP) 기법의 한계를 극복하고, 입력 뉴런의 관련성을 계층별로 전파하여 새로운 접근 방법을 제시합니다. 또한, 최근 개발된 Vision Transformer 아키텍처에 이 방법을 적용하여 두 개의 이미지 분류 데이터셋인 ImageNet과 PascalVOC에서 성능을 평가했습니다.

- **Technical Details**: 본 연구에서는 Relative Absolute Magnitude Layer-Wise Relevance Propagation (absLRP)라는 새로운 규칙을 개발하여, 동일 계층 내에서 서로 다른 절대 크기를 가진 활성화 뉴런 간의 잘못된 상대적 기여 문제를 해결합니다. 세 가지 다른 아키텍처(VGG, ResNet50, ViT-Base)를 사용하여 absLRP를 적용하고, 기존의 기법들과 비교하여 확실한 장점을 입증했습니다. 또한, Global Attribution Evaluation (GAE)이라는 새로운 평가 방법을 제안하여, 기여도 평가의 신뢰성과 강인성을 함께 평가할 수 있는 종합적인 점수를 도출합니다.

- **Performance Highlights**: 실험 결과, absLRP는 기존의 여러 최신 기법들에 비해 뛰어난 성능을 보였으며, 각각의 기법의 강점과 약점을 비교 분석할 수 있는 기회를 제공했습니다. 두 개의 유명한 이미지 분류 데이터셋에서 수행된 실험은 우리 방법론의 우수성을 명확히 드러냈습니다. 이러한 결과는 다양한 기여 기반 방법을 평가하는 데 매우 유용한 인사이트를 제공합니다.



### GoHD: Gaze-oriented and Highly Disentangled Portrait Animation with Rhythmic Poses and Realistic Expression (https://arxiv.org/abs/2412.09296)
Comments:
          Accepted by AAAI 2025

- **What's New**: GoHD는 오디오 기반으로 매우 사실적이고 표현력이 풍부한 초상화 비디오를 생성하기 위한 강력한 프레임워크입니다. 이 모델은 오디오와 시각 데이터를 원활하게 통합하며, 세 가지 핵심 모듈을 통해 동작과 신원의 고 분해능을 달성합니다. 특히, 향상된 일반화 능력을 가지고 있어 보지 못한 입력 스타일에도 잘 작동합니다.

- **Technical Details**: GoHD는 세 가지 주요 모듈로 구성됩니다: 첫째, 잠재 탐색(latent navigation)을 활용한 애니메이션 모듈은 입력 스타일에 대한 일반화 기능을 개선합니다. 둘째, 조건부 확산 모델을 사용하여 음성의 억양(prosody)을 인식하면서 머리 자세를 보장합니다. 셋째, 두 단계의 학습 방식을 통해 오디오와 리드십이 동기화된 현실적인 표현을 추정합니다.

- **Performance Highlights**: 광범위한 실험을 통해 GoHD의 우수한 일반화 기능이 입증되며, 이는 임의의 주제에 대한 사실적인 화상 결과를 생성할 수 있습니다. 이 시스템은 기존의 방법들과 달리 자연스러운 안구 움직임을 교정하고, 동기화를 통해 사실적인 표현을 생성하는 데 성공하였습니다. 또한, 복잡한 동작의 조합을 효과적으로 처리하여 다양한 얼굴 표정을 정교하게 만들어냅니다.



### InstanceCap: Improving Text-to-Video Generation via Instance-aware Structured Caption (https://arxiv.org/abs/2412.09283)
- **What's New**: 이번 논문에서는 텍스트-비디오 생성의 품질 향상을 위한 새로운 프레임워크인 InstanceCap을 제안합니다. 이것은 인스턴스 수준의 세밀한 비디오 캡션 생성을 최초로 가능하게 하며, 두 가지 주요 과제를 해결합니다: 캡션과 비디오 간의 높은 일관성과 정확한 내용 표현입니다. InstanceCap은 멀티모달 큰 언어 모델(MLLM)을 통해 모호한 정보 없이 정확한 비디오 설명을 생성하도록 설계되었습니다.

- **Technical Details**: InstanceCap은 글로벌 비디오를 로컬 인스턴스의 고유한 정보로 변환하기 위해 보조 모델 클러스터(AMC)를 활용합니다. 이러한 방식으로 캡션의 명확성을 높이고, 복잡한 텍스트로부터 구조화된 구문으로 변환하여 세분화된 비디오 설명을 생성합니다. 또한, 22K InstanceVid 데이터셋을 구성하여 고해상도 비디오 훈련에 사용합니다.

- **Performance Highlights**: 다양한 실험 결과에 따르면, InstanceCap은 기존 모델을 능가하며 캡션과 비디오 간의 높은 일관성을 유지합니다. 또한, 훈련 후 T2V 모델은 세부사항과 동작을 더욱 정확하게 추적할 수 있는 능력을 증명하였습니다. 이러한 결과는 InstanceCap의 효과성과 텍스트-비디오 생성에서의 진보된 가능성을 보여줍니다.



### Towards a Multimodal Large Language Model with Pixel-Level Insight for Biomedicin (https://arxiv.org/abs/2412.09278)
Comments:
          Accepted by AAAI2025

- **What's New**: 최근 다중모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 발전으로, 지능형 생의학 보조 도구 개발이 가능하게 되었습니다. 그러나 기존 생의학 MLLMs는 이미지 수준의 이해에 주로 초점을 맞추고, 텍스트 명령어로의 상호작용에 제한됨으로써 그 활용성을 제한하고 있습니다. 본 논문에서는 픽셀 수준의 이해가 가능한 새로운 End-to-End MLLM인 MedPLIB를 소개하며, 이는 추천 인식(vizual question answering, VQA), 다양한 픽셀 기반 프롬프트와 픽셀 수준의 그라운딩을 지원합니다.

- **Technical Details**: 우리는 새로운 Mixture-of-Experts(MoE) 다단계 훈련 전략을 제안하며, 이를 통해 시각-언어 전문가 모델과 픽셀 그라운딩 전문가 모델을 분리하여 훈련한 뒤, MoE를 사용하여 미세 조정합니다. 이 전략은 여러 작업의 학습을 효과적으로 조정하면서, 추론 시 단일 전문가 모델의 계산 비용 수준을 유지합니다. 또한, Medical Complex Vision Question Answering Dataset (MeCoVQA)를 소개하여 복잡한 의료 이미징 질문에 대한 8가지 모달리티를 포함합니다.

- **Performance Highlights**: 실험 결과, MedPLIB는 여러 의료 시각 언어 작업에서 최첨단 성능을 달성했습니다. 특히, 픽셀 그라운딩 작업의 제로샷 평가에서 MedPLIB는 mDice 메트릭 기준으로 가장 작은 및 큰 모델에 대해 각각 19.7, 15.6의 차이로 우위를 점하고 있습니다. 또한, 연구 커뮤니티를 위해 데이터, 코드 및 모델 체크포인트를 공개할 예정입니다.



### Text-Video Multi-Grained Integration for Video Moment Montag (https://arxiv.org/abs/2412.09276)
- **What's New**: 이 연구에서는 Video Moment Montage(VMM)라는 새로운 작업을 제안합니다. VMM은 주어진 내레이션 스크립트에 기반하여 적절한 비디오 세그먼트를 자동으로 찾고 조합하여 비디오 몽타주를 생성합니다. 이 작업은 스크립트 문장과 비디오 클립 간의 세부적인 문맥 관계를 학습해야 하는 도전 과제가 있습니다. 이를 해결하기 위해, 저자들은 Text-Video Multi-Grained Integration(TV-MGI) 방법을 소개합니다.

- **Technical Details**: TV-MGI는 텍스트 특성과 비디오 특성을 다층적으로 융합하여 세부적인 멀티모달 이해 및 예측을 가능하게 합니다. 이 방법은 프레임-샷-텍스트 기능 융합 메커니즘을 통해 다양한 소스의 비디오를 처리하고, 비디오의 내용을 스크립트와 정밀하게 맞추는 작업을 수행합니다. 또한, 신뢰할 수 있는 비디오 자료를 생성하는 합성 데이터 증강 모듈도 제안되어 모델의 일반화 능력을 향상시킵니다.

- **Performance Highlights**: MSSD(Multiple Sentences with Shots Dataset)라는 대규모 데이터셋이 수집되었으며, 이 데이터셋은 VMM 작업에 맞게 설계되었습니다. 실험을 통해, 제안된 프레임워크는 기존의 기준 모델 대비 뛰어난 효과를 보여주었습니다. 이는 VMM 작업이 단순한 비디오 세그먼트 검색을 넘어 비디오 편집 요구 사항에 부합하는 새로운 기준을 제공한다는 것을 의미합니다.



### LatentSync: Audio Conditioned Latent Diffusion Models for Lip Sync (https://arxiv.org/abs/2412.09262)
- **What's New**: 이번 논문에서는 LatentSync라는 새로운 lip sync 프레임워크를 제안합니다. 이 프레임워크는 오디오가 조건화된 잠재적 확산 모델(latent diffusion models)을 기반으로 하며, 중간 모션 표현 없이도 작동합니다. LatentSync는 Stable Diffusion의 강력한 생성 능력을 활용하여 복잡한 오디오-비주얼 상관관계를 직접 모델링할 수 있도록 설계되었습니다.

- **Technical Details**: LatentSync의 주요 특징 중 하나는 템포럴(REPresentation) 정렬(TREPA) 기법입니다. TREPA는 대규모 자기 지도 자기 비디오 모델(VideoMAE-v2)을 통해 추출된 시간적 표현을 사용하여 생성된 프레임과 실제 프레임의 정렬을 개선합니다. 프레임 간의 시간적 일관성을 확보하고 lip-sync 정확성을 유지하는 데 도움을 줍니다.

- **Performance Highlights**: LatentSync는 HDTF 테스트 세트에서 SyncNet의 정확성을 91%에서 94%로 크게 향상시켰습니다. 다양한 메트릭을 통해 HDTF 및 VoxCeleb2 데이터 세트에서 최신 lip sync 방법을 초월하는 성능을 보였습니다. 이러한 경험은 SyncNet을 활용하는 다른 lip sync 및 오디오 기반 포트레이트 애니메이션 방법에도 적용 가능성을 가지고 있습니다.



### FD2-Net: Frequency-Driven Feature Decomposition Network for Infrared-Visible Object Detection (https://arxiv.org/abs/2412.09258)
Comments:
          This work is accepted by AAAI 2025

- **What's New**: 본 논문에서는 적외선-가시 물체 탐지(IVOD)를 위한 새로운 Frequency-Driven Feature Decomposition Network(FD2-Net)를 제안합니다. 이 네트워크는 적외선과 가시 이미지의 보완 정보를 이용하여 복잡한 환경에서 탐지 성능을 향상시킵니다. 특히, 기존 방법들이 무시했던 주파수 특성을 활용하여 고주파와 저주파 특성을 효율적으로 분리하고 처리합니다.

- **Technical Details**: FD2-Net은 고주파 유닛(HFU)과 저주파 유닛(LFU)으로 구성된 특징 분해 인코더를 사용합니다. HFU는 이산 코사인 변환(discrete cosine transform)을 활용해 고주파 특징을 포착하고, LFU는 다중 스케일 컨볼루션 커널을 사용해 저주파 구조를 모델링합니다. 이러한 유닛들은 서로의 강점을 통합하는 매개변수 없는 보완 강도 전략(complementary strengths strategy)을 통해 멀티모달 특징을 강화합니다.

- **Performance Highlights**: FD2-Net은 LLVIP, FLIR, M3FD와 같은 다양한 IVOD 벤치마크에서 최신 기술(SOTA) 모델을 초월하는 성능을 보여주었습니다. 각각 96.2% mAP, 82.9% mAP 및 83.5% mAP라는 높은 정확도를 기록하여, 적외선 및 가시 이미지의 보완 정보를 효과적으로 활용하는 데 성공했습니다.



### VLMs meet UDA: Boosting Transferability of Open Vocabulary Segmentation with Unsupervised Domain Adaptation (https://arxiv.org/abs/2412.09240)
- **What's New**: 본 논문은 다양한 도메인에서 세분화(segmentation) 정확도를 향상시키기 위해 Vision-Language reasoning과 Unsupervised Domain Adaptation (UDA) 기법을 통합하는 새로운 접근 방식을 제안합니다. 기존의 Vision-Language Models (VLMs)와 합성 데이터 기반 방법의 한계를 극복하기 위해, 제안하는 Foundational-Retaining Open Vocabulary Semantic Segmentation (FROVSS) 프레임워크는 다중 규모의 컨텍스트 데이터를 활용하여 세분화 능력을 개선하고 있습니다. 또한 UDA와 함께 이 기술을 적용하여 새로운 카테고리에 대해 더 효과적으로 적응할 수 있는 모델을 개발했습니다.

- **Technical Details**: 제안된 FROVSS 프레임워크는 VLM의 세부 정보를 보존하면서 안정적인 교육을 위한 distillation 기법과 cross-domain mixed sampling을 채택합니다. 이를 통해 데이터 요구사항을 줄이고, VLM의 글로벌 지식을 유지하면서 미세 조정(fine-tuning) 과정에서의 재앙적 망각(catastrophic forgetting)을 방지합니다. 또한, 텍스트 관계 개선을 위해 대규모 언어 모델(LLMs)을 활용한 개념 수준의 프롬프트 증강 전략을 적용하여, 다양한 데이터셋 간의 카테고리 인식 개선을 목표로 하고 있습니다.

- **Performance Highlights**: FROVSS 및 UDA-FROVSS를 통해 학습된 모델은 여러 세그멘테이션 데이터셋에서 우수한 성능을 나타내며, 각 벤치마크에서 개선된 결과를 보여주고 있습니다. 특히 Cityscapes 데이터셋에서 22.1% mIoU 향상이 이루어졌으며, Synthia-to-Cityscapes 설정에서 새로운 UDA 기준을 수립하여 이전의 최신 방법에 비해 8% 이상 개선되었습니다. 이러한 결과는 다양한 데이터셋에 걸쳐 FROVSS의 성능과 일반화 능력이 크게 향상되었음을 시사합니다.



### Foundation Models and Adaptive Feature Selection: A Synergistic Approach to Video Question Answering (https://arxiv.org/abs/2412.09230)
- **What's New**: 이번 논문에서는 비디오 질문-답변(Video QA)의 새로운 접근 방식을 제안합니다. LGQAVE(Local-Global Question Aware Video Embedding)라는 모델을 통해 질문과 비디오 프레임, 그리고 의미적 객체 수준 추상화를 효과적으로 통합할 수 있는 방안을 모색합니다. 기존의 방법이 프레임 샘플링에서 한계를 보인 반면, 이 방법은 교차 주의(cross-attention) 메커니즘을 사용하여 질문과 가장 관련성이 높은 프레임을 정확히 식별합니다.

- **Technical Details**: LGQAVE는 질문을 인식하여 비디오 프레임을 선택하기 위해 학습 가능한 교차 주의 모듈을 활용합니다. 이 모듈은 질문의 의미와 가장 일치하는 프레임을 효율적으로 고립시켜, 선택된 프레임의 객체 간 상호작용을 모델링하기 위해 spatial 및 temporal 그래프를 생성합니다. 이 그래프는 Dynamic Graph Transformer(또는 DGT-Q 모델이라고 부르며)로 입력되어 프레임 특성을 정제하는 데 쓰입니다.

- **Performance Highlights**: LGQAVE는 NextVQA, TGIF-QA, Causal VidQA, MSRVTT-QA와 같은 여러 벤치마크에서 기존 모델에 비해 유의미한 성과를 보였습니다. 특히, 고차원의 다중 선택 및 개방형 질문에 대한 정확성을 크게 향상시켜, 비디오 질문-답변의 범위를 넓히는 데 기여하고 있습니다. 이러한 성과는 질문의 의미와 관련된 비디오 프레임에 대한 정량적 해석이 가능하다는 점에서 주목할 만합니다.



### UADet: A Remarkably Simple Yet Effective Uncertainty-Aware Open-Set Object Detection Framework (https://arxiv.org/abs/2412.09229)
Comments:
          Under review

- **What's New**: 본 연구에서는 Open-Set Object Detection (OSOD) 문제를 해결하기 위한 새로운 접근법인 UADet를 제안합니다. UADet는 알려지지 않은 객체와 배경을 구분하기 위해 appearance uncertainty와 geometric uncertainty를 접목하여, 과거 방법들이 가지던 한계를 극복합니다. 이 방법은 주석이 없는 데이터를 효과적으로 활용하여 높은 성능을 유지하면서도 알려지지 않은 객체의 회수율을 1.8배 향상시킵니다.

- **Technical Details**: UADet는 두 가지 유형의 불확실성, 즉 appearance uncertainty(객체의 전경일 가능성)와 geometric uncertainty(알려진 객체와의 겹침 정도)를 고려합니다. 이를 통해 정보의 손실을 최소화하며, 주석이 없는 객체를 효과적으로 활용합니다. 기존 두 단계(Faster R-CNN의 강력한 모델을 기반으로 한)에 기반한 설계를 활용하여 OSOD 문제를 해결하며, 알려지지 않은 객체에 대한 회수율을 개선합니다.

- **Performance Highlights**: UADet는 OSOD와 Open World Object Detection (OWOD) 작업 모두에서 최고의 성능을 달성했습니다. 특히 M-OWODB 및 S-OWODB 벤치마크에서 알려지지 않은 객체 인식의 평균 회수율이 각각 13.8% 및 6.9% 향상되었습니다. 이러한 결과는 UADet이 두 가지 작업 모두에서 새로운 최고 성능 기준을 세웠음을 보여줍니다.



### DASK: Distribution Rehearsing via Adaptive Style Kernel Learning for Exemplar-Free Lifelong Person Re-Identification (https://arxiv.org/abs/2412.09224)
Comments:
          in Proceedings of the 39th AAAI Conference on Artificial Intelligence (AAAI-25)

- **What's New**: 이번 연구에서는 Lifelong Person Re-identification (LReID) 문제를 해결하기 위한 새로운 패러다임인 DASK(Distribution Rehearsing via Adaptive Style Kernel Learning)를 제안합니다. 기존 LReID 접근법이 데이터 재생(data replay)과 지식 증류(knowledge distillation) 방법에 의존했던 반면, DASK는 어떠한 역사적 사례도 저장하지 않고도 강력한 잊어버림 방지 성능을 발휘합니다. 특히, DASK는 새로운 데이터 학습 중에 이전 도메인의 분포를 모델링하고 연습하는 방식을 도입하여 지식 통합을 강화합니다.

- **Technical Details**: DASK는 크게 두 가지 구성 요소로 나뉘며, 첫째는 Distribution Rehearser Learning (DRL), 둘째는 Distribution Rehearsing-driven LReID Training (DRRT)입니다. DRL은 임의의 분포 데이터를 현재 데이터 스타일로 변환하는 과정을 스스로 학습하며, AKPNet(Adaptive Kernel Prediction network)을 통해 각 인스턴스에 특정한 분포 전이 커널을 예측합니다. DRRT는 새로운 데이터와 이전 학습 단계에서 얻은 AKPNet 모델을 함께 이용하여 새로운 지식과 오래된 지식을 통합한 모델을 구축합니다.

- **Performance Highlights**: 실험 결과, DASK는 기존 방법들보다 잊어버림 방지 성능이 3.6%-6.8%, 일반화 능력에서는 4.5%-6.5% 향상되었습니다. 이러한 성능 향상은 DASK의 분포 연습 메커니즘이 효과적으로 구식 데이터 스타일로의 변환을 가능하게 하고, 새로운 데이터와 기존 데이터를 동시에 활용하여 지식을 통합하는 데에 기인합니다. 따라서 DASK는 현재의 최고의 비사례(exemplar-free) LReID 방법들 중 하나로 자리 잡고 있습니다.



### USDRL: Unified Skeleton-Based Dense Representation Learning with Multi-Grained Feature Decorrelation (https://arxiv.org/abs/2412.09220)
Comments:
          Accepted by AAAI 2025

- **What's New**: 본 연구에서는 특징 비상관(feature decorrelation) 기반의 통합 스켈레톤 밀집 표현 학습 프레임워크인 USDRL을 제안합니다. 이는 시간, 공간, 인스턴스 도메인에서 밀도 있게 특성을 비상관화하여 표현의 차원 간 중복성을 줄이고 정보 추출을 극대화합니다. 기존의 대조 학습 방법이 데이터 의존성이 강한 점을 해결하며, 밀집 예측 작업을 위한 세밀한 로컬 표현 학습에 초점을 맞추고 있습니다.

- **Technical Details**: USDRL은 시간적, 공간적, 인스턴스 도메인에서의 다면적 특성 비상관화를 사용하여 표현의 일관성과 차별성을 확보합니다. 또한, 밀집 시공간 인코더(DSTE)를 설계하여 세밀한 동작 표현을 캡처하고, Convolutional Attention(CA) 및 Dense Shift Attention(DSA) 모듈을 통해 로컬 특성과 밀집 종속성을 모델링합니다.

- **Performance Highlights**: NTU-60, NTU-120, PKU-MMD I 및 PKU-MMD II와 같은 다양한 데이터셋에서 시행된 실험 결과, 제안된 방법이 최신 기술(SOTA)을 현저히 웃도는 성능을 보여주었습니다. 이를 통해 동작 인식, 검색, 감지 등 여러 하위 작업에서의 유용성을 입증하였습니다.



### Enhancing Implicit Neural Representations via Symmetric Power Transformation (https://arxiv.org/abs/2412.09213)
Comments:
          Accepted by AAAI 2025

- **What's New**: 이번 연구에서는 삼성적인 전력 변환(symmetric power transformation) 방법을 제안하여 암묵 신경 표현(Implicit Neural Representation, INR)의 용량을 데이터 변환(data transformation) 관점에서 향상시킵니다. 기존의 무작위 순열(random permutation) 또는 인덱스 재배치(index rearrangement)와는 달리, 이 방법은 추가 저장이 필요 없는 reversible operation을 특징으로 합니다. 연구팀은 특정 범위와 대칭이 INR의 표현 능력을 향상시킬 수 있다는 Range-Defined Symmetric Hypothesis를 제안하며 이 이론에 따라 비선형 대칭 전력 변환(nonlinear symmetric power transformation)을 개발하였습니다.

- **Technical Details**: 대칭 전력 변환은 데이터를 특정 범위로 스케일링(scaling)하고 대칭 분포를 재분배(redistribution)하여 INR의 훈련을 개선하는 것을 목표로 합니다. 연구팀은 데이터 변환이 신호의 연결성을 단절시키지 않도록 보장하기 위해 deviation-aware calibration과 adaptive soft boundary를 추가 설계하여 극단적인 편차 극대화와 경계에서의 연속성 중단 문제를 해결합니다. 이로 인해 변환의 견고성을 향상시키고 자연 신호의 복잡성을 관리하여 더 나은 성능을 보장하게 됩니다.

- **Performance Highlights**: 제안된 대칭 전력 변환 방법은 기존의 다른 데이터 변환 방법들과 비교하여 두드러진 성능 향상을 보였습니다. 다양한 실험, 특히 1D 오디오, 2D 이미지, 3D 비디오 적합 작업을 통해 방법의 효과성과 적용 가능성이 입증되었습니다. 가장 큰 장점은 대칭 전력 변환이 추가 저장 공간을 소모하지 않고 신호를 손실 없이 복원할 수 있다는 점으로, 이는 기존 방법들과 비교할 때 상당한 메리트입니다.



### eCARLA-scenes: A synthetically generated dataset for event-based optical flow prediction (https://arxiv.org/abs/2412.09209)
- **What's New**: 이 연구에서는 event-based vision과 Spiking Neural Networks (SNNs)의 결합이 로봇 공학에 미치는 영향력에 대해 논의하고 있습니다. 특히, eWiz라는 포괄적인 라이브러리와 eCARLA-scenes라는 합성 event-based optical flow 데이터셋을 소개하여, 다양한 환경을 효과적으로 시뮬레이션하고 SNNs의 활용 가능성을 제시합니다. 이러한 접근 방법은 데이터의 다양성과 조작성을 높이기 위해 필수적입니다.

- **Technical Details**: 이 논문은 event-based 카메라 (EBC)와 optical flow 예측을 위한 새로운 데이터 생성 파이프라인을 다룹니다. EBC는 전통적인 프레임 기반 카메라와 달리 비동기적으로 동작하여 빠르게 변화하는 조명을 효과적으로 포착할 수 있습니다. 또한, eWiz 라이브러리는 데이터 로딩, 변형, 증대, 시각화 및 훈련 데이터 생성을 위한 다양한 도구들을 제공하여 연구자들이 이벤트 기반 데이터를 효율적으로 처리할 수 있게 지원합니다.

- **Performance Highlights**: eCARLA-scenes 데이터셋은 CARLA 시뮬레이터를 활용해 다양한 날씨 조건 하에서 환경을 시뮬레이션하여 생성되었습니다. 이 데이터셋은 기존의 real-world 데이터셋이 갖는 한계를 극복하고, SNNs와 같은 신경형 하드웨어에서 효과적으로 활용될 수 있는 가능성을 제공합니다. eWiz와 eCARLA-scenes는 연구자들이 event-based 데이터 처리의 필요성을 충족시킬 수 있도록 설계되었습니다.



### Temporal Action Localization with Cross Layer Task Decoupling and Refinemen (https://arxiv.org/abs/2412.09202)
Comments:
          AAAI 2025

- **What's New**: 이 논문에서는 Cross Layer Task Decoupling and Refinement (CLTDR) 방법을 제안하여 데이터에 있는 두 가지 주요 작업인 분류(action classification) 및 위치 지정(action localization)을 효과적으로 분리하고 정제하는 접근 방식을 소개한다. CLTDR 전략은 비디오의 피라미드 특성을 활용하여 높은 레이어에서의 의미론적 강한 특징과 낮은 레이어에서의 경계 인식(boundary-aware) 특징을 통합하여 작업의 일관성을 높인다. 또한, 경량화된 Gated Multi-Granularity (GMG) 모듈을 통해 즉각적, 지역적, 전역적 시간적 세분화를 고려하여 비디오 특성을 종합적으로 추출하는 방법도 제안한다.

- **Technical Details**: CLTDR는 각 레이어의 특징들을 통합하여 분류 및 위치 지정을 동시에 고려한다. 이 과정에서 두 개의 서로 다른 작업을 완화하는 데 도움이 되는 attention 메커니즘이 사용된다. GMG 모듈은 1D 깊이-wise convolution과 Fast Fourier Transform (FFT)를 이용하여 다양한 시간적 세분화의 정보를 포괄적으로 집계하며, 여러 가지 가지 네트워크를 통해 효율적인 정보 추출을 수행한다.

- **Performance Highlights**: 본 방법은 THUMOS14, MultiTHUMOS, EPIC-KITCHENS-100, ActivityNet-1.3, HACS와 같은 다섯 가지 벤치마크에서 최신 성능을 달성하였다. CLTDR와 GMG 모듈의 조합 덕분에 기존 방법에 비해 성능이 크게 향상되었음을 보인다. 실험 결과는 제안된 방법이 TAL(performance on Temporal Action Localization)에서 우수함을 강조한다.



### MVC-VPR: Mutual Learning of Viewpoint Classification and Visual Place Recognition (https://arxiv.org/abs/2412.09199)
Comments:
          8 pages

- **What's New**: 본 논문에서는 Visual Place Recognition (VPR)에서 서로 다른 시점에서 촬영된 이미지 간의 인식 성능 저하 문제를 해결하기 위해, 뷰포인트(views) 자체 분류의 상호 학습(mutable learning) 기법을 제안합니다. 기존의 지도 학습 방식에서 벗어나, 지리적 좌표를 기반으로 한 거칠고 미세한 분류 방식을 통해 데이터셋을 자동으로 구분하게 됩니다. 이러한 접근법은 손으로 정의한 규칙이나 참조 레이블 없이, 데이터셋 클래스를 매번 업데이트하며 성능을 향상시키는 데 기여합니다.

- **Technical Details**: 제안한 방법은 먼저 UTM(Universal Transverse Mercator) 좌표계에 따라 데이터를 거칠게 분류하고, 이후 K-Means 클러스터링을 통해 각 그룹 내에서 미세한 분류를 수행합니다. 다양한 시점에서 촬영된 이미지의 feature를 추출하여, 이러한 features를 기반으로 서로 다른 뷰포인트에 대한 클러스터를 생성합니다. 이 과정은 VPR 분류 목표와의 훈련을 통해 상호 강화되는 방식을 채택하여, 안정적인 모델 훈련을 지원합니다.

- **Performance Highlights**: 실험 결과, 본 방법이 거의 완벽하게 데이터셋을 뷰포인트에 따라 분할하는 성능을 보여주었으며, 기존의 ground truth 레이블에 의존하는 방법들보다 우수한 성과를 달성했습니다. K-Means 클러스터링을 활용하여 효율성과 높은 성능을 동시에 만족하는 방식을 제안함으로써, VPR 분야에서 새로운 연구 방향을 제시하고 있습니다.



### ExpRDiff: Short-exposure Guided Diffusion Model for Realistic Local Motion Deblurring (https://arxiv.org/abs/2412.09193)
Comments:
          Project website: this https URL

- **What's New**: 이 논문에서는 ExpRDiff라는 새로운 이미지 복원 네트워크를 제안하여 로컬 블러(blur) 제거를 효과적으로 수행합니다. 기존 방법들이 주로 전역 블러(global blur)에 중점을 두었음에도 불구하고, 본 연구는 이동 물체로 인한 블러를 효과적으로 식별하고 복원하기 위해 문맥 기반의 로컬 블러 탐지 모듈을 개발하였습니다. 또한, 짧은 노출 시간의 이미지를 활용하여 구조적 세부 정보를 포함한 블러 제거 방법도 개발하였습니다.

- **Technical Details**: ExpRDiff는 세 가지 주요 모듈로 구성됩니다: 문맥 기반 로컬 블러 탐지 모듈, 블러리-어웨어(blur-aware) 가이드 이미지 복원 모듈, 및 짧은 노출 가이드 확산(diffusion) 모델입니다. 문맥 기반 모듈은 추가적인 문맥 정보를 통해 블러 지역을 식별하며, 가이드 이미지 복원 모듈은 블러 지역과 명확한 지역을 동시에 처리합니다. 마지막으로, 확산 모델은 신뢰할 수 있는 구조적 세부 사항에 집중하여 현실적인 이미지를 복원합니다.

- **Performance Highlights**: 실험 결과 ExpRDiff는 사용 가능한 로컬 블러 데이터셋과 실제 신뢰할 수 있는 상황에서 최신 기술보다 우수한 성능을 보였습니다. 이를 통해 Blurry-aware guided image restoration 및 short-exposure guided diffusion model의 조합이 현저한 블러 제거 및 복원 성과를 가져온다는 것을 입증하였습니다. 이로 인해, 다양한 응용 분야에서 더 나은 이미지 품질을 제공할 수 있을 것으로 기대됩니다.



### RAD: Region-Aware Diffusion Models for Image Inpainting (https://arxiv.org/abs/2412.09191)
- **What's New**: 이번 논문에서는 전통적인 diffusion 모델을 사용해 이미지 인페인팅을 위한 새로운 방법인 지역 인식 diffusion 모델(Region-Aware Diffusion Models, RAD)을 제안합니다. RAD는 각 픽셀에 서로 다른 노이즈 스케줄(noise schedule)을 적용하여 지역별로 비동기식(asynchronous) 생성이 가능하도록 하여, 전체 이미지 컨텍스트를 고려합니다. 이 접근법은 추가적인 구성 요소 없이도 빠른 추론 속도를 달성할 수 있습니다.

- **Technical Details**: RAD는 지역별로 다른 노이즈 스케줄을 정의하여 각 픽셀의 노이즈를 비율적으로 조절합니다. 기존 방법들이 추가적인 모듈이나 Nested Loop를 요구할 경우, RAD는 간단한 수정으로 이루어져 있으며, Reshape된 Fully Connected 레이어를 1×1 convolution으로 바꾸는 것만으로도 우수한 성능을 달성합니다. 또한, Perlin noise를 활용하여 효율적인 훈련을 위해 효과적인 노이즈 스케줄을 생성하는 등 더 효율적인 훈련 방법도 소개합니다.

- **Performance Highlights**: 실험 결과, RAD는 FFHQ, LSUN Bedroom, ImageNet 데이터셋에서 기존의 최첨단(in-state-of-the-art) 방법들보다 최대 100배 빠른 추론 시간을 기록하며 이미지 품질 또한 개선되었습니다. FID와 LPIPS 점수 모두에서 팬데믹 이미지 인페인팅 분야에서 최고 성능을 나타내었고, ablation study를 통해 지역 변동 노이즈 스케줄 및 공간 노이즈 임베딩 기법이 RAD의 성공에 필수적임을 증명했습니다.



### On the effectiveness of Rotation-Equivariance in U-Net: A Benchmark for Image Segmentation (https://arxiv.org/abs/2412.09182)
- **What's New**: 이 논문은 Convolutional Neural Networks (CNNs)에서 rotation equivariance의 통합에 초점을 맞춥니다. 특히, 이 연구는 이미지 분할을 위해 U-Net 아키텍처에 rotation equivariance를 접목하여 기존의 연구가 다루지 않았던 다양한 과제를 포함한 더 포괄적인 평가를 제공합니다. 기존 연구들은 주로 분류 작업에 집중했기 때문에 이 연구는 이미지 분할에 적용될 수 있는 새로운 통찰을 제공합니다.

- **Technical Details**: U-Net 아키텍처는 인코더-디코더 구조를 기반으로 하며 skip connections를 포함합니다. 인코더는 입력 이미지에서 계층적 특징을 추출하고, 디코더는 세분화 마스크를 재구성하는 역할을 합니다. 이 구조는 특히 경계 구분에 효과적이며, 다양한 형태의 객체에 대해 세분화 마스크를 생성할 수 있습니다.

- **Performance Highlights**: 이 연구는 rotation-equivariant U-Net 모델이 전통적인 U-Net 아키텍처에 비해 정확성을 높이거나 계산 자원을 절약할 수 있는지를 분석합니다. 평가에는 Kvasir-SEG와 같은 특정 데이터셋뿐만 아니라 COCO-Stuff과 같이 더 일반적인 세분화 과제에 대한 데이터셋이 포함되어, rotation equivariance의 더 넓은 적용 가능성을 탐구합니다.



### Weighted Poisson-disk Resampling on Large-Scale Point Clouds (https://arxiv.org/abs/2412.09177)
Comments:
          Accepted to AAAI 2025

- **What's New**: 본 논문에서는 대규모 포인트 클라우드 처리에 대한 새로운 방법으로 가중 Poisson-disk (WPD) 재샘플링 기법을 제안합니다. WPD 기법은 전통적인 Poisson-disk 재샘플링을 개선한 것으로, voxel 기반 추정 전략을 통해 포인트 클라우드의 기하학적 일관성을 유지하면서 포인트 수와 밀도를 효율적으로 조정할 수 있게 합니다. 또한, 이 방법은 가파른 특성을 보존하는 기능을 갖추고 있어 다양한 응용 프로그램에서의 활용성이 높습니다.

- **Technical Details**: 제안된 WPD 재샘플링 방법은 두 가지 단계로 구성됩니다. 초기 Poisson 재샘플링 단계에서는 voxel 기반 분석을 통해 포인트 수에 대한 더 정확한 Poisson-disk 반경을 추정합니다. 이어서 가중 탄젠트 평활화 단계를 통해 포인트 분포 최적화와 함께 국소 이웃을 제어하며, 이 과정에서 등방성 분포를 효과적으로 설정합니다.

- **Performance Highlights**: 실험 결과, WPD 재샘플링 방법이 대규모 포인트 클라우드의 재샘플링 성능을 크게 향상시킴을 보여주었습니다. 이 방법은 또한 다양한 응용 프로그램에 대해서도 높은 품질의 기하학적 일관성을 제공하며, 특히 시각화 및 의미론적 특성 학습에 대한 요구 사항을 잘 균형 잡을 수 있습니다.



### DECOR:Decomposition and Projection of Text Embeddings for Text-to-Image Customization (https://arxiv.org/abs/2412.09169)
- **What's New**: 이번 논문에서는 Text-to-image (T2I) 모델의 커스터마이징(CUSTOMIZATION) 능력을 향상시키기 위한 새로운 접근방식인 DECOR를 제안합니다. 이전의 방법에서는 적은 수의 참조 이미지를 사용할 경우 오버피팅(overfitting)이 발생하는 문제가 있었으나, DECOR는 이러한 문제를 해결하기 위해 텍스트 임베딩(text embedding)을 분석하고 개선합니다.

- **Technical Details**: DECOR는 텍스트 임베딩 매트릭스를 분해하고 구성 요소 분석을 통해 임베딩 공간의 기하학을 이해합니다. 이 과정에서 바람직하지 않은 토큰 벡터와 직교(orthogonal)인 벡터 공간으로 텍스트 임베딩을 투영하여 원치 않는 의미의 영향을 감소시킵니다. 이러한 기술적 접근은 기존의 low-rank adaptations (LoRA) 방식의 한계를 극복하는 데 중점을 둡니다.

- **Performance Highlights**: 실험 결과 DECOR는 최신 커스터마이즈 모델들과 비교하여 우수한 성능을 보였으며, 텍스트와 시각적 정렬 평가 메트릭에서 Pareto frontier 성능을 달성했습니다. 또한, 입력 프롬프트에 보다 충실한 이미지를 생성함으로써 텍스트-이미지 커스터마이징의 효과성을 입증하였습니다.



### Pinpoint Counterfactuals: Reducing social bias in foundation models via localized counterfactual generation (https://arxiv.org/abs/2412.09160)
- **What's New**: 이 논문에서는 기존의 편향을 분석할 수 있는 생성 방법의 한계를 극복하기 위해, 이미지의 맥락을 보존하면서 특정 속성 관련 영역에서만 수정이 이루어지도록 자동 마스킹과 가이드 인페인팅을 이용한 새로운 지역화된 반사실 생성 방법을 제안합니다. 이는 성별 편향을 생성하기 위해 Conceptual Captions 데이터셋에 적용되었으며, 기존 방법들보다 높은 시각적 및 의미적 정확도를 보여주었습니다.

- **Technical Details**: 제안된 방법은 기본적으로 속성 관련 영역을 자동으로 마스킹하고, 그 지역에서만 수정이 이뤄지도록 제약을 두는 방식입니다. 이를 통해 추가적인 편향을 유입하지 않으면서도 원본 이미지의 맥락을 유지할 수 있습니다. 성별 등 보호 속성에 따라 이미지를 수정하는 과정에서, 주요 속성 변경을 초점으로 하여, 다른 요소들인 의상, 자세, 환경 등이 가진 편향이 최소화됩니다.

- **Performance Highlights**: 제안된 방법은 기존의 생성적 접근법들보다 편향 프로파일링의 정확성을 증명하였으며, 생성된 데이터셋을 활용한 모델 파인튜닝에서도 여러 메트릭에서 편향 감소를 보여주었습니다. 또한, ImageNet 제로샷 성능을 유지하면서도 성별 분류 불균형과 개인 선호 점수를 개선하였습니다.



### Evaluating Adversarial Attacks on Traffic Sign Classifiers beyond Standard Baselines (https://arxiv.org/abs/2412.09150)
Comments:
          Accepted for publication at ICMLA 2024

- **What's New**: 최근 연구에서는 교통 신호 분류 모델에 대한 adversarial 공격이 주목받고 있습니다. 이 연구는 기본 모델(LISA-CNN, GTSRB-CNN)에서 패턴을 반복하는 데 한정된 기존 연구와는 달리, 모델 아키텍처와 데이터세트를 분리하여 일반 모델에서의 비교를 실시합니다. 또한, inconspicuous(눈에 띄지 않는) 및 visible(눈에 띄는) 공격 설정을 비교하여 새로운 공격 평가 방식을 제안합니다.

- **Technical Details**: 본 연구는 기존의 두 가지 간극을 메우고자 합니다. 첫째로, 일반 모델과 함께 기초 모델을 평가하여 더 공정한 비교를 가능하게 하고, 둘째로, 공격 설정을 비교하여 공격의 효과와 눈에 띄는 정도 사이의 상관 관계를 탐구합니다. 이 연구에서는 주어진 대상 클래스에 대해 딥러닝 모델의 출력을 기반으로 공격을 실시합니다.

- **Performance Highlights**: 실험 결과, LISA-CNN 및 GTSRB-CNN과 같은 표준 기초 모델이 일반 모델보다 공격에 더 취약하다는 것을 확인했습니다. 본 연구는 앞으로 새로운 공격을 평가할 때 더 다양한 기초 모델을 고려해야 함을 강조합니다. 이를 통해 기존의 연구 한계를 극복하고 교통 신호 인식 시스템의 강화된 안전성을 도모할 수 있을 것입니다.



### LVMark: Robust Watermark for latent video diffusion models (https://arxiv.org/abs/2412.09122)
- **What's New**: 이번 연구에서는 LVMark라는 새로운 워터마킹 방법을 제안합니다. 이 방법은 비디오 확산 모델에 워터마크를 직접 삽입하여 생성 모델의 소유권을 보호하는 데 중점을 두고 있습니다. 기존의 비디오 워터마킹 기법들이 비디오 내용에 워터마크를 삽입하는 데 그쳤던 것에 비해, LVMark는 생성 모델 자체에 워터마크를 삽입하여 모델 도난 시 소유권verification이 가능하도록 하였습니다.

- **Technical Details**: LVMark의 핵심은 선택적 가중치 조정(strategy)으로, 이는 생성하는 비디오의 품질을 유지하면서 비디오 확산 모델에 워터마크 메시지를 효율적으로 포함시킵니다. 또한, 새로운 워터마크 디코더는 시공간(spatio-temporal) 정보를 활용하여 메시지를 디코딩하며, 이를 위해 3D wavelet 변환을 사용합니다. 이 방법은 비디오의 정합성과 H.264 압축과 같은 비디오 특유의 공격에 강한 Robustness를 제공합니다.

- **Performance Highlights**: 실험 결과, LVMark는 기존의 모든 비교 기준을 초과하는 성능을 입증하였습니다. 방법이 확산 모델의 구조를 변경하지 않기 때문에, 비공식 사용자가 워터마킹 과정을 우회하는 것을 방지할 수 있습니다. LVMark는 AI 생성 비디오를 식별하고 소유권을 검증할 수 있는 강력한 도구로, 향후 비디오 생성 모델의 소유권 보호를 위한 중요한 기술로 자리잡을 것입니다.



### ResFlow: Fine-tuning Residual Optical Flow for Event-based High Temporal Resolution Motion Estimation (https://arxiv.org/abs/2412.09105)
Comments:
          10 pages, 8 figures

- **What's New**: 본 논문에서는 이벤트 카메라를 활용한 고속(HTR) 광학 흐름 추정의 새로운 알고리즘을 제안합니다. 기존 방법들은 HTR 추정에서 발생하는 정보 부족으로 인해 성능이 저하되는 문제를 해결하지 못했습니다. 이 논문은 흐름 예측을 잔여(prediction)로 변환하는 새로운 접근 방식을 소개하여, HTR 흐름을 LTR에서 유도하고 효과적으로 추정합니다.

- **Technical Details**: 제안된 방법은 HTR 추정을 위해 두 단계의 잔여 기반 프레임워크를 구성합니다. 첫 번째 단계는 전역 선형 운동을 추정하고, 두 번째 단계는 HTR 잔여 흐름을 개선하여 event 데이터의 내재적 희소성 문제를 완화합니다. 훈련 과정에서는 공유된 잔여 정제기를 사용하여 LTR 감독을 통해 HTR 흐름을 추정하고, 지역 노이즈를 추가하여 LTR에서 HTR 흐름으로의 적응을 돕도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 기존의 LTR 및 HTR 지표에서 최첨단 성능을 달성했습니다. 특히, 유연한 훈련 전략과 강력한 흐름 보간 방법이 결합되어 결과의 신뢰성을 높였습니다. 이러한 연구는 이벤트 카메라의 가능성을 재정의하고, 동적이고 도전적인 환경에서의 운동 분석을 변화시킬 수 있는 잠재력을 지니고 있습니다.



### Towards Long-Horizon Vision-Language Navigation: Platform, Benchmark and Method (https://arxiv.org/abs/2412.09082)
Comments:
          A novel Vision-Language Navigation task: Long-Horizon Vision-Language Navigation

- **What's New**: 이번 연구에서는 장기 계획과 의사 결정 일관성을 강조하는 새로운 Vision-Language Navigation (VLN) 작업인 Long-Horizon Vision-Language Navigation (LH-VLN)을 제안합니다. 이를 위해 복잡한 작업 구조를 가진 데이터셋을 자동 생성하는 플랫폼인 NavGen을 개발하여 LH-VLN 지원에 중점을 두었습니다. 또한, 3,260개의 다양한 작업으로 구성된 Long-Horizon Planning and Reasoning in VLN (LHPR-VLN) 벤치마크를 통해 복잡한 VLN 작업을 평가할 수 있는 기틀을 마련했습니다.

- **Technical Details**: NavGen 플랫폼은 다단계 작업을 위한 복잡한 데이터셋을 생성하는 데 필요한 자동화된 데이터 생성 기능을 갖추고 있습니다. 이 플랫폼은 양방향의 다중 Granularity 생성 방식을 통해 작업의 다양성을 풍부하게 하고 데이터의 유용성을 높입니다. 또한, Multi-Granularity Dynamic Memory (MGDM) 모듈을 통해 단기 및 장기 메모리를 통합하여 다이나믹 환경에서의 유연한 탐색이 가능하도록 합니다.

- **Performance Highlights**: LH-VLN 태스크에서 새로운 State-of-the-Art 성능을 달성했으며, 이는 지속적인 의사 결정과 긴 다단계 작업에서의 강력한 탐색 능력을 보여줍니다. 이번 연구의 기여는 LH-VLN 작업을 통해 복잡한 다단계 네비게이션 과제를 평가할 수 있는 새로운 기준을 제시하는 것입니다. 또한, 새로운 평가 메트릭인 독립 성공률(Independent Success Rate, ISR)과 조건부 성공률(Conditional Success Rate, CSR)을 도입하여 성과 측정의 세밀함을 추가했습니다.



### DomCLP: Domain-wise Contrastive Learning with Prototype Mixup for Unsupervised Domain Generalization (https://arxiv.org/abs/2412.09074)
Comments:
          Code page: this https URL

- **What's New**: 이번 연구에서는 Self-supervised learning (SSL) 방법의 주요 한계를 극복하기 위한 새로운 접근법인 DomCLP, 즉 Domain-wise Contrastive Learning with Prototype Mixup을 제안합니다. 기존의 SSL 모델은 unseen-domain 데이터에 대한 효과적인 표현을 생성하는 데 어려움을 겪는 반면, DomCLP는 도메인에 무관한 공통 특징을 강화하여 도메인 일반화 성능을 향상시킵니다. 이 접근법은 InfoNCE의 적용에서 나타나는 부정적인 영향과 도메인 관련 특징의 증폭 현상을 분석하여, 이를 해결하기 위한 전략을 소개합니다.

- **Technical Details**: DomCLP는 Domain-wise Contrastive Learning (DCon)과 Prototype Mixup Learning (PMix)의 두 가지 구성 요소로 나뉘어 있습니다. DCon은 도메인 무관의 공통 특징을 강화하여 표현 학습을 수행하고, PMix은 각 도메인에서의 공통 특징을 보간하여 여러 도메인 전반에 걸쳐 일반화합니다. 이는 k-means 클러스터링을 통해 특징의 프로토타입을 추출하고, mixup을 활용하여 혼합 프로토타입으로 모델을 훈련시키는 방식으로 진행됩니다.

- **Performance Highlights**: 제안된 방법은 PACS 및 DomainNet 데이터셋에서 전체 최신 기술을 초월하는 성능을 보여줍니다. 예를 들어, PACS 1% 레이블 분수에서 최대 11.3%의 성능 향상을 달성하였고, DomainNet 1% 레이블 분수에서는 12.32%의 향상을 기록했습니다. 이러한 결과는 DomCLP가 도메인 무관의 공통 특징을 효과적으로 향상하고 일반화할 수 있음을 입증합니다.



### SVasP: Self-Versatility Adversarial Style Perturbation for Cross-Domain Few-Shot Learning (https://arxiv.org/abs/2412.09073)
- **What's New**: 이번 연구에서는 Cross-Domain Few-Shot Learning(CD-FSL)에서 발생하는 그래디언트 불안정성과 지역 최적화 문제를 해결하기 위해 Self-Versatility Adversarial Style Perturbation(SVasP)이라는 새로운 방법을 제안합니다. SVasP는 다양한 입력 패턴과 지역화된 크롭 스타일 그래디언트를 통합하여 그래디언트의 안정성을 강화하고, 손실 경관에서 평평한 최적점을 얻어 모델의 전이 가능성을 높입니다. 이를 통해 기존의 방식보다 모델의 일반화 성능이 현저히 개선됨이 입증되었습니다.

- **Technical Details**: SVasP는 이미지 내에서 글로벌 스타일과 크롭 스타일 그래디언트를 결합하여 입력의 다양성을 증진시키는 구조로 설계되었습니다. 이 방법은 타겟 이미지의 글로벌 스타일을 안정적으로 조정하기 위해 선형 및 비선형 반복 과정을 포함하며, 각 과정에서 다양한 입력 배치를 통해 스타일 그래디언트를 생성하고 통합해 나갑니다. 이에 따라 트레이닝 과정에서 그래디언트의 진동 문제가 효과적으로 완화되므로, 모델은 날카로운 최소값을 피하고 더 나은 일반화 성능을 발휘할 수 있습니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋에서 실시한 실험 결과, SVasP는 최신의 다른 방법들에 비해 모델의 일반화 능력을 크게 향상시킴을 보여줍니다. 특히, 신선한 스타일 perturbation을 적용함으로써 기존의 방식이 가진 한계를 극복하고, 훨씬 더 높은 성능을 발휘할 수 있음을 확인했습니다. 이 연구의 결과는 CD-FSL 분야에서 스타일 기반 접근법의 효과성을 새로운 차원으로 끌어올리는 기초가 될 것입니다.



### Cross-View Completion Models are Zero-shot Correspondence Estimators (https://arxiv.org/abs/2412.09072)
Comments:
          Project Page: this https URL

- **What's New**: 이번 연구는 cross-view completion 학습에 대한 새로운 관점을 탐구하며, 이는 self-supervised correspondence learning과 유사성을 기반으로 하고 있습니다. 본 논문은 cross-attention map이 인코더 또는 디코더의 피처보다 효과적인 대응 관계를 포착함을 보여주고, 이를 통해 geometric downstream task에서 성능을 향상시킬 수 있음을 밝혔다.

- **Technical Details**: 이 연구는 cross-view completion(CVC) 모델의 cross-attention map이 encoder와 decoder의 특징을 시각화하여 얻은 pixel-wise cosine similarity 점수를 통해 분석되었습니다. 특히 CroCo-v2를 중심으로 이 세 가지 상관 관계 중 cross-attention map이 더욱 선명하고 노이즈가 적은 반면, 기존의 피처 설명자는 상대적으로 품질이 떨어진다고 설명하고 있습니다.

- **Performance Highlights**: ZeroCo라는 zero-shot 추론 기법을 도입하여 cross-attention map을 이용한 대응 관계의 강화를 강조했습니다. 검증 결과, cross-attention map은 상태-of-the-art 성능을 달성하며, 밀집 대응(dense correspondence)과 다중 프레임 깊이 추정(multi-frame depth estimation)에서 경쟁력 있는 결과를 보였습니다.



### An Efficient Framework for Enhancing Discriminative Models via Diffusion Techniques (https://arxiv.org/abs/2412.09063)
Comments:
          Accepted by AAAI2025

- **What's New**: 본 논문은 인간의 뇌에서의 신호 인식 과정을 바탕으로, 디퓨전 모델(Generative model)과 분별 모델(Discriminative model)을 통합한 새로운 프레임워크인 DBMEF(Diffusion-Based Discriminative Model Enhancement Framework)를 제안합니다. 이 접근 방식은 초기 예측을 통해 불확실한 예측을 재평가할 수 있는 능력을 부여하여, 이미지 분류의 정확도와 일반화 능력을 향상시키는 데 중점을 두고 있습니다. DBMEF는 훈련 과정 없이 쉽게 통합될 수 있는 플러그 앤 플레이 구조를 가지고 있습니다.

- **Technical Details**: DBMEF는 분별 모델이 테스트 입력에 대한 초기 예측을 수행한 후, 예측의 불확실성이 높을 경우 디퓨전 모델에도 테스트 샘플을 전달하여 재검토 과정을 수행합니다. 이는 인간 뇌의 신속한 경로 및 느린 경로의 상호작용을 모방하며, 다양한 딥 뉴럴 네트워크에 대해 효과적인 결과를 도출합니다. 저자들은 ResNet-50 모델에 대해 ImageNet 데이터셋에서 1.51%의 성능 개선을 기록하며, ImageNet-A 데이터셋에서는 3.02%의 향상을 달성했습니다.

- **Performance Highlights**: 실험 결과는 본 프레임워크가 딥 뉴럴 네트워크의 분류 정확도와 일반화 능력을 상당히 증가시킬 수 있음을 보여줍니다. 단순한 네트워크 아키텍처인 ResNet, VGG, ViT 모두에서 우수한 결과를 나타내며, 기존의 디퓨전 분류기와 비교하여 훨씬 적은 시간으로도 성능이 향상된다는 점이 두드러집니다. 이러한 결과는 DBMEF가 다양한 데이터셋 및 신경망에서 안정적인 성능 개선을 이룰 수 있는 새로운 패러다임을 제시함을 의미합니다.



### Hyperbolic-constraint Point Cloud Reconstruction from Single RGB-D Images (https://arxiv.org/abs/2412.09055)
Comments:
          Accepted by AAAI25

- **What's New**: 이번 논문에서는 3D point cloud reconstruction에 하이퍼볼릭 공간 (hyperbolic space)을 도입하여 점군의 복잡한 계층 구조를 왜곡 없이 표현하고 이해하는 방식을 제안합니다. 기존의 메소드들은 비싼 CAD 모델과 복잡한 기하학적 이전조건에 의존했으나, 새로운 접근 방식은 이들 문제를 해결할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 하이퍼볼릭 차원에서 하이퍼볼릭 샴퍼 거리 (hyperbolic Chamfer distance)와 정규화된 트리플렛 손실 (regularized triplet loss)을 제안하여 부분 및 전체 점군의 관계를 개선했습니다. 또한, 3D 구조의 이해와 복원을 향상시키기 위해 적응형 경계 조건 (adaptive boundary conditions)을 설계하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 모델이 기존의 대부분의 모델보다 뛰어난 성능을 보이며, feature extraction 능력이 크게 향상되었습니다. 특히, ablation 연구를 통해 모델의 구성 요소들이 갖는 중요성을 강조하고 있습니다.



### ContextHOI: Spatial Context Learning for Human-Object Interaction Detection (https://arxiv.org/abs/2412.09050)
Comments:
          in proceedings of the 39th AAAI Conference on Artificial Intelligence (AAAI-25)

- **What's New**: 이번 논문에서는 Human-Object Interaction (HOI) 인식에 있어 공간적 맥락의 중요성을 강조하며, 이를 개선하기 위한 새로운 프레임워크 ContextHOI를 제안합니다. 기존의 HOI 감지기가 사물 탐지기에서 발전하였지만, 공간적 맥락을 충분히 활용하지 않아 제한된 시각 단서에서의 인식을 어려워하는 문제를 해결하고자 합니다. ContextHOI는 객체 감지 기능과 공간적 맥락을 동시에 포착하여 HOI 감지의 정확도를 높이는 것을 목표로 합니다.

- **Technical Details**: ContextHOI는 이중 분기 구조를 가지고 있으며, 특히 배경 레이블을 추가로 요구하지 않고도 유용한 공간적 맥락을 추출할 수 있도록 훈련됩니다. 이를 위해 우리는 컨텍스트 추출기와 공간적 감시, 의미적 감시를 도입하여 관련 없는 노이즈를 필터링하고 의미 있는 맥락을 포착합니다. 덕분에 이 모델은 HOI 검출에서 뛰어난 성능을 발휘하며 HICO-DET 및 v-coco 벤치마크에서 최첨단 성능을 기록합니다.

- **Performance Highlights**: ContextHOI는 HICO-DET 및 v-coco에서 경쟁력 있는 성과를 보이며, 특히 HICO-DET (ambiguity) 벤치마크에서 이전의 HOI 감지기들보다 더 높은 견고성을 입증하였습니다. 이를 통해 불분명하거나 가려진 전경의 HOI 장면을 포함한 검출 능력을 향상시킨 것으로, fully occluded objects와 blurred objects에 대한 인식에서 강화된 성능을 보여주고 있습니다.



### Motif Guided Graph Transformer with Combinatorial Skeleton Prototype Learning for Skeleton-Based Person Re-Identification (https://arxiv.org/abs/2412.09044)
Comments:
          Accepted by AAAI 2025. Codes are available at this https URL

- **What's New**: 이 논문에서는 인물 재식별(person re-ID)을 위한 3D 스켈레톤 데이터 기반의 접근 방식을 제안합니다. 기존의 방법들이 모든 관절 간의 가상의 운동 연관성을 가정했던 반면, 본 연구에서는 주요 신체 구조와 보행 패턴을 중점적으로 분석하여 효과적인 스켈레톤 표현을 학습하는 새로운 방식을 제시합니다. 특히, 모티프 유도 그래프 변환기(Motif guided graph transformer, MGT)와 조합형 스켈레톤 프로토타입 학습(Combinatorial skeleton prototype learning, CSP) 방법을 통해 스켈레톤 관찰의 심화 학습을 목표로 합니다.

- **Technical Details**: 이 연구의 핵심 기술은 모티프 유도 그래프 변환기(MGT)와 조합형 스켈레톤 프로토타입 학습(CSP)입니다. MGT는 계층적 구조 모티프와 보행 협업 모티프를 통합하여 신체 관절 간의 관계 학습을 강화합니다. CSP는 랜덤한 공간-시간 조합을 활용하여 다양한 서브 스켈레톤 및 서브 트랙렛 표현을 생성하며, 이를 통해 각 개체의 대표적인 스켈레톤 특징을 학습합니다.

- **Performance Highlights**: 실험 결과, MoCos는 기존의 최첨단 모델에 비해 우수한 성능을 보였습니다. 다섯 개의 공개 데이터셋에서의 평가를 통해, MoCos는 RGB로 추정된 스켈레톤 및 다양한 그래프 모델링, 비지도 학습 시나리오에서 효과적으로 적용될 수 있음을 입증하였습니다. 이로 인해 MoCos는 스켈레톤 기반 인물 재식별 분야에서 유망한 접근으로 자리매김할 것으로 기대됩니다.



### DrivingRecon: Large 4D Gaussian Reconstruction Model For Autonomous Driving (https://arxiv.org/abs/2412.09043)
- **What's New**: 본 논문에서는 자율주행을 위한 Large 4D Gaussian Reconstruction Model(DrivingRecon)을 소개합니다. 이 모델은 surround view 비디오에서 직접 4D Gaussian을 예측하는 방식으로, 기존 방법들이 갖는 시간 소모적인 반복 과정을 피하고 효율성을 높입니다. DrivingRecon은 Prune and Dilate Block(PD-Block)을 사용하여 인접 뷰 간 중복된 Gaussian 점을 제거하고 배경 점을 줄입니다.

- **Technical Details**: DrivingRecon은 2D 인코더를 사용해 surround-view 이미지에서 특징을 추출하고, DepthNet 모듈이 깊이를 추정하여 세계 좌표를 도출합니다. 이러한 좌표와 이미지 특징은 시간적 교차 주의 메커니즘에 입력되어, PD-Block을 통해 멀티뷰 통합을 강화합니다. 이러한 과정에서, PD-Block은 복잡한 객체를 위한 Gaussian 점의 확장을 학습하여 풍부한 장면 재구성을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, DrivingRecon은 기존 방법에 비해 장면 재구성 품질과 새로운 뷰 합성에서 유의미한 개선을 보여줍니다. 모델 전훈, 차량 적응, 장면 편집 같은 응용 분야에서도 효과적임을 검증하며, 실제 환경에서의 다양한 상황에 대한 일반화 성능 또한 향상되었습니다.



### Video Anomaly Detection with Motion and Appearance Guided Patch Diffusion Mod (https://arxiv.org/abs/2412.09026)
Comments:
          Accept by AAAI2025

- **What's New**: 최근 비디오 이상 탐지(Video Anomaly Detection, VAD)의 새로운 접근 방식은 diffusion 모델을 활용하여 정상 패턴을 생성하는 문제로 구성하였습니다. 기존 모델들이 특성 수준에서 정상 샘플을 예측하는 데 집중했던 반면, 본 연구에서는 이상 현상이 외관과 동작 양 측면에서 나타나는 것을 고려하였습니다. 이를 위해 상세한 지역 정보를 캡처하는 패치 기반(patch-based) diffusion 모델을 제안하며, 이상 탐지를 위한 완벽한 솔루션을 도출할 것을 주장합니다.

- **Technical Details**: 우리는 비디오 프레임을 외관과 동작 성분으로 분해하며, 처음 프레임은 외관의 모든 시각적 측면을 포함하여 예측의 어려움을 증가시킵니다. 우리는 첫 번째로 외관 인코더와 메모리 블록을 통합해 외관 정보를 이해할 수 있도록 하고, 두 번째로 파라미터 차이를 활용한 사전 동작 전략을 제안하여 동작의 미세한 차이를 포착합니다. 이러한 패치 기반 기법을 통해 더 정밀한 로컬라이제이션을 실현하고, 두 가지 조건을 융합하는 혁신적인 방법을 개발하였습니다.

- **Performance Highlights**: 본 연구는 네 가지 도전적인 VAD 데이터셋에서 실험을 수행하여 제안된 접근 방식의 유효성을 입증하였습니다. 결과적으로, 우리의 모델은 대부분의 기존 방법보다 일관되게 우수한 성능을 보여주었으며, 정상 및 비정상 동작 패턴을 구분하는 데 효과적임을 나타냅니다. 이는 비디오 분석에서 다양한 이상 패턴을 보다 효과적으로 감지하고 구분할 수 있도록 해줍니다.



### STEAM: Squeeze and Transform Enhanced Attention Modu (https://arxiv.org/abs/2412.09023)
- **What's New**: 본 논문에서는 CNN(Convolutional Neural Network)의 채널(Channel) 및 공간(Spatial) 주의(attention) 메커니즘을 효과적으로 모델링하기 위한 새로운 방법인 STEAM(Squeeze and Transform Enhanced Attention Module)을 소개합니다. 이 모듈은 파라미터를 최소화하면서도 깊은 신경망의 표현 능력을 향상시키는 데 중점을 두고 있습니다. 특히, 기존의 그래프 기반 접근에서 영감을 받아 채널과 공간 주의를 동시에 다루는 최초의 방법을 제안합니다.

- **Technical Details**: STEAM 모듈은 그래프의 관계 모델링(relational modeling) 원리를 기반으로 하여 채널 및 공간 주의를 통합하는 상수 파라미터 모듈을 구현합니다. 또한, Output Guided Pooling (OGP) 기법을 도입하여 공간적 맥락(context)을 효과적으로 포착하여 공간 주의를 강화합니다. 이러한 접근법을 통해 효율적인 특징(context) 모델링을 구현하면서도 파라미터 수와 계산량을 줄이는 데 성공했습니다.

- **Performance Highlights**: STEAM은 대규모 이미지 분류(image classification), 객체 탐지(object detection), 인스턴스 분할(instance segmentation) 등 여러 기준 데이터셋에서 광범위하게 평가되었습니다. 평가 결과, STEAM은 표준 ResNet-50 모델에 비해 2%의 정확도 향상을 달성했으며, GFLOPs 증가량 또한 매우 미미했습니다. 또한, STEAM은 정확도 측면에서 ECA 및 GCT와 같은 선도적인 모듈을 초월하면서 GFLOPs는 세 배 감소하는 성과를 보였습니다.



### Arbitrary-steps Image Super-resolution via Diffusion Inversion (https://arxiv.org/abs/2412.09013)
Comments:
          16 pages, 9 figures. Project: this https URL

- **What's New**: 이 연구는 이미지 슈퍼 해상도(Super-Resolution, SR)를 위한 새로운 기술을 제안하며, 이는 확산 모델(diffusion models)의 최적 노이즈 맵을 추정하여 SR 성능을 향상시키고자 합니다. Partial noise Prediction(PnP) 전략을 통해 확산 모델의 중간 상태를 구성하며, 이를 시작 샘플링 지점으로 사용합니다. 기존 방법들과 달리, 우리의 접근 방식은 확산 네트워크에 대한 수정 없이 최적화된 노이즈 맵을 찾는 데 초점을 맞추고 있습니다.

- **Technical Details**: 본 연구는 SR을 위한 확산 반전(diffusion inversion) 기술을 재정립합니다. 노이즈 예측기(noise predictor)를 도입하여 주어진 저해상도 이미지에서 노이즈 맵을 추정하고, PnP 전략을 통해 중간 상태를 구성합니다. 이 접근 방식은 확산 모델의 전진 프로세스에 따라 저해상도 이미지에 노이즈를 추가함으로써 구현됩니다.

- **Performance Highlights**: 제안된 방법은 유연하고 효율적인 샘플링 메커니즘을 제공하여 샘플링 단계의 수를 1에서 5까지 조절할 수 있습니다. 단일 샘플링 단계에서도 최근의 최첨단 방법들과 동등하거나 우수한 성능을 보이는 결과를 나타냅니다. 이는 SR에서의 다양한 저하 조건에 대처할 수 있는 유연성을 제공하여, 각기 다른 정도의 저하 상태에 따라 샘플링 단계를 조정할 수 있게 합니다.



### MS2Mesh-XR: Multi-modal Sketch-to-Mesh Generation in XR Environments (https://arxiv.org/abs/2412.09008)
Comments:
          IEEE AIxVR 2025

- **What's New**: 이 논문에서는 MS2Mesh-XR이라는 혁신적인 다중 모드 스케치-중첩 생성 파이프라인을 제안합니다. 이 시스템은 사용자들이 손으로 그린 스케치와 음성 입력을 통해 확장 현실(XR) 환경에서 사실적인 3D 객체를 생성할 수 있도록 지원합니다. MS2Mesh-XR은 20초 이내에 고품질 3D 메시를 생성하여 몰입형 시각화와 조작을 가능하게 합니다.

- **Technical Details**: 제안된 파이프라인은 사용자로부터 수집된 다중 모드 정보를 활용하여 고해상도 이미지를 생성하고, 이를 Convolutional Reconstruction Model을 사용하여 세부적인 3D 메시로 재구성합니다. 주요 기술로는 ControlNet이 있으며, 이는 손으로 그린 스케치에서 기하학적 컨텍스트를 추출하고 음성 입력에서 텍스트 프롬프트를 해석하는 단계가 포함됩니다. 이러한 과정은 각기 다른 단계에서 이미지 디퓨전, 3D 재구성 및 메시 세부 조정을 포함합니다.

- **Performance Highlights**: 두 가지 사용 사례를 통해 제안한 파이프라인의 실용성을 입증했습니다. 가상 현실(VR) 모드에서 자산 생성과 혼합 현실(MR) 모드에서 인테리어 디자인을 수행하여 다양한 XR 시나리오에서 우리의 방법의 효과성을 보여주었습니다. 이 접근 방식은 몰입감을 증대시키고 사용자 상호작용을 지원하여 XR 기반 창작 생산성을 크게 향상시킵니다.



### Enhancing Facial Consistency in Conditional Video Generation via Facial Landmark Transformation (https://arxiv.org/abs/2412.08976)
- **What's New**: 이번 논문에서는 3D Morphable Model (3DMM)을 기반으로 한 얼굴 랜드마크 변환 방법을 제안하여, 조건부 비디오 생성에서 얼굴 특징의 일관성을 개선합니다. 특히, 복잡한 동작을 포함한 캐릭터 애니메이션 생성에서 참조 이미지와 일치하는 얼굴 특징을 유지하는 것을 목표로 합니다. 이를 통해 기존의 접근 방식에서 발생하는 얼굴 특징 불일치를 효과적으로 해결할 수 있습니다.

- **Technical Details**: 제안하는 방법은 먼저 소스 비디오에서 3D 얼굴을 재구성한 후, 참조 이미지의 얼굴 특징에 맞추기 위해 3DMM의 파라미터를 조정합니다. 조정된 3D 얼굴 모델에서 새로 변환된 얼굴 랜드마크를 추출하여 비디오 생성 모델로 입력합니다. 이 과정은 플러그 앤 플레이 방식으로 다양한 비디오 생성 프레임워크에 쉽게 통합이 가능합니다.

- **Performance Highlights**: 제안된 방법은 생성된 비디오와 참조 이미지 간의 얼굴 일관성을 개선하며, 얼굴 특징의 일관성을 주요 성과로 시연합니다. 기존의 방법들과 비교하여 더욱 높은 정밀도로 개인의 얼굴 특징을 재현하는 데 성공하였으며, 비디오 생성의 전반적인 품질을 향상시켰습니다.



### Elevating Flow-Guided Video Inpainting with Reference Generation (https://arxiv.org/abs/2412.08975)
Comments:
          AAAI 2025

- **What's New**: 이번 연구는 Video Inpainting (VI)을 위한 새로운 프레임워크인 RGVI(Reference-Guided Video Inpainting)를 제안합니다. 이 시스템은 강력한 generative model(생성 모델)을 활용하여 레퍼런스 생성을 하고, 고급 픽셀 전파 알고리즘과 결합하여 기존의 흐름 기반 접근 방식의 한계를 극복합니다. RGVI는 텍스트 프롬프트를 기반으로 하여 누락된 영역에서 새로운 콘텐츠를 생성할 수 있는 능력을 갖추고 있습니다.

- **Technical Details**: RGVI는 픽셀 전파의 오류 누적을 피하고 sub-pixel precision(서브 픽셀 정밀도)을 유지하기 위해 새로운 픽셀 전파 알고리즘을 소개합니다. 이 알고리즘은 흐름 추적(flow tracing)과 그리드 워핑(grid warping)을 결합하여 고해상도 비디오에서도 뛰어난 품질을 제공합니다. 또한, RGVI는 불안정한 전파 영역을 감지하는 전파 검증 메서드를 도입하여 전파의 신뢰성을 높입니다.

- **Performance Highlights**: RGVI는 HQVI 데이터셋을 사용하여 다양한 VI 방법을 평가할 수 있는 고품질 VI 벤치마크를 제공합니다. 공개 벤치마크 테스트에서 RGVI는 뛰어난 시각적 품질과 시간적 일관성을 보여주었으며, 2K 해상도 이상의 고해상도 비디오 처리에 대한 우수성을 강조합니다. 정량적인 평가는 RGVI의 효율성을 더욱 뒷받침하고, 기존 솔루션들에 비해 현저히 더 높은 결과를 보여줍니다.



### Is Contrastive Distillation Enough for Learning Comprehensive 3D Representations? (https://arxiv.org/abs/2412.08973)
Comments:
          Under review

- **What's New**: 본 논문에서는 현재의 contrastive distillation 방법의 한계를 이론적으로 분석하고, 이러한 문제를 해결하기 위해 CMCR이라는 새로운 프레임워크를 제안합니다. 기존 방법들이 modality-specific features를 미흡하게 다루고 있음을 지적하며, 본 방법은 modality 공유와 특정 기능을 통합하는 데 초점을 맞추고 있습니다. 특히 masked image modeling과 occupancy estimation 작업을 도입하여 더 나은 학습을 지원합니다.

- **Technical Details**: CMCR 프레임워크는 modality-specific 및 modality-shared features를 동시에 학습할 수 있도록 설계되었습니다. 다중 모달 통합 코드북을 통해 서로 다른 모달리티 간의 효과적인 임베딩 공간을 형성하고, geometry-enhanced masked image modeling을 통해 3D 표현 학습을 향상시킵니다. 이러한 접근을 통해, 다양한 실제 과제에서 기존 방법들을 일관되게 초월하는 성능을 입증합니다.

- **Performance Highlights**: 실험 결과, CMCR 프레임워크는 3D semantic segmentation, object detection 및 panoptic segmentation 등의 다운스트림 작업에서 뛰어난 성능을 보였습니다. 기존의 self-supervised learning 기법들과 비교하여 높은 적응성과 성능을 발휘하여, 3D 표현 학습에서의 한계를 효과적으로 극복합니다. 코드 및 실험 데이터는 논문에서 제공되며, 이를 통해 연구 결과를 보다 쉽게 검증할 수 있습니다.



### AFFAKT: A Hierarchical Optimal Transport based Method for Affective Facial Knowledge Transfer in Video Deception Detection (https://arxiv.org/abs/2412.08965)
Comments:
          Accepted by AAAI 2025

- **What's New**: 본 논문에서는 비디오 속임수 탐지를 위한 새로운 방법인 AFFAKT를 제안합니다. 이 방법은 심리학 이론에 착안하여, 대규모 표정 데이터 셋에서 유용하고 상관된 지식을 전이하여 분류 성능을 개선하는 데 초점을 맞추고 있습니다. AFFAKT는 두 개의 주요 과제를 해결하는 데 중점을 두어, 비디오 속임수 탐지 모델의 성능을 향상시킵니다.

- **Technical Details**: AFFAKT는 Hierarchical Optimal Transport Knowledge Transfer (H-OTKT) 모듈과 Sample-specific Re-weighting Knowledge Bank (SRKB) 모듈을 통합하여, 표정 클래스와 속임수 샘플 간의 최적 연결을 정량화합니다. H-OTKT는 카테고리별 지식의 전이량을 결정하고, SRKB는 샘플별 가중치 조정 전략을 사용하여 상관 프로토타입을 통해 전이된 지식을 정교하게 조정합니다. 이 구조는 속임수 탐지에서의 그루비션 성능을 높이기 위한 기반을 제공합니다.

- **Performance Highlights**: 실험 결과, AFFAKT는 두 개의 비디오 속임수 탐지 데이터 세트에서 기존 방법들보다 우수한 성능을 보였습니다. 또한, 해석 가능성 연구를 통해 속임수 행동과 부정적인 감정 간의 높은 연관성을 발견했으며, 이는 심리학 이론과 일치하는 결과입니다. 이상적인 지식 전이로 인해 속임수 탐지의 정확도가 개선되었음을 확인할 수 있습니다.



### Multimodal Industrial Anomaly Detection by Crossmodal Reverse Distillation (https://arxiv.org/abs/2412.08949)
- **What's New**: 이번 논문에서는 Crossmodal Reverse Distillation (CRD)이라는 새로운 방법론을 제안하여 다중 모달 산업 이상 탐지 분야에서의 효과적인 이상 탐지 및 로컬라이제이션을 실현하고 있습니다. CRD는 각 모달리티에 독립적인 브랜치를 할당하여 각 모달리티 내에서의 이상 탐지를 더욱 정교하게 할 수 있도록 설계되었습니다. 또한, Crossmodal Filter와 Amplifier를 도입하여 모달리티 간 상호 작용을 증진시킵니다.

- **Technical Details**: CRD는 Multi-branch Distillation (MBD) 설계를 기반으로 하여, 각 모달리티의 이상 탐지를 강화하고 모달리티 융합 과정을 개선합니다. 이 과정을 통해 각각의 모달리티에서 선생 네트워크의 피처를 사용하여 학생 네트워크가 정상 특성을 더 잘 학습할 수 있도록 합니다. 이를 통해 이상 스코어는 모든 브랜치의 이상 맵을 통합하여 계산됩니다.

- **Performance Highlights**: MVTec 3D-AD 데이터셋에서의 실험 결과, 우리의 제안한 CRD 방법이 다중 모달 이상 탐지 및 로컬라이제이션에서 최첨단 성능을 달성함을 입증하였습니다. 이 연구는 다중 모달 이상 탐지에서 기존의 방법들이 갖는 한계를 극복하고, 다양한 산업 환경에서의 실제 적용 가능성을 높이고 있습니다.



### Mojito: Motion Trajectory and Intensity Control for Video Generation (https://arxiv.org/abs/2412.08948)
- **What's New**: 이 논문에서는 텍스트를 기반으로 비디오를 생성하는 새로운 확산 모델인 Mojito를 소개합니다. Mojito는 방향성 모션 제어(Directional Motion Control) 모듈과 모션 강도 조절(Motion Intensity Modulator) 모듈을 통합하여 사용자 요구에 맞는 모션 방향과 강도를 제어할 수 있는 기능을 갖추고 있습니다. 이를 통해 추가 교육 없이도 생성된 물체의 모션을 효율적으로 유도할 수 있으며, 비디오의 자연스러운 동적 흐름을 실현합니다.

- **Technical Details**: Mojito는 두 가지 혁신적인 모듈을 포함하고 있어 비디오 생성 시 모션의 방향과 강도를 정밀하게 조절할 수 있습니다. 첫 번째 모듈인 DMC는 크로스 어텐션을 활용하여 객체의 동작 경로를 조정하며, 두 번째 모듈인 MIM은 비디오에서 생성된 옵티컬 플로우 맵을 기반으로 모션의 다양한 강도를 안내합니다. 이러한 설계는 사용자 요구에 맞게 조정할 수 있는 유연성과 효율성을 제공합니다.

- **Performance Highlights**: Mojito는 실험을 통해 목표한 모션 방향과 강도를 정확히 구현하면서도 높은 계산 효율성을 달성한 것으로 나타났습니다. 기존의 최첨단 모델과 비교할 때, Mojito는 고품질 비디오 콘텐츠를 생산하며 효과적이고 효율적인 모션 제어를 제공합니다. 이 연구는 향후 모션을 강조한 비디오 생성 모델의 발전을 위한 중요한 통찰력을 제공합니다.



### Selective Visual Prompting in Vision Mamba (https://arxiv.org/abs/2412.08947)
Comments:
          in Proceedings of the 39th AAAI Conference on Artificial Intelligence (AAAI-25)

- **What's New**: 이번 연구에서는 Vision Mamba (Vim) 모델에 대한 처음의 시각적 프로프트 기술인 Selective Visual Prompting (SVP) 방법을 도입했습니다. 기존의 비주얼 프롬프팅 기법은 주로 Vision Transformer (ViT) 모델에 맞춰져 있어 Vim의 고유한 특성을 고려하지 않았습니다. SVP는 입력 종속적으로 토큰 수준의 프롬프트를 생성하여 불필요한 정보를 제거하고 차별적인 특징을 효과적으로 전파하는 데 중점을 둡니다.

- **Technical Details**: SVP는 경량 프롬프터를 사용하여 입력에 따라 동적으로 토큰 프롬프트를 생성하며, 이를 통해 Vim 모델의 갱신 및 잊기 게이트를 선택적으로 활성화하여 정보 전파를 촉진합니다. 또한, Cross-Prompting과 Inner-Prompting이라는 이중 경로 구조를 통해 이를 구분된 파라미터로 구현하여 계층 간 공유 정보와 각 계층 내 특정 정보를 최적화합니다. 이는 두 가지 정보 유형 간의 균형 있는 활용을 보장합니다.

- **Performance Highlights**: 광범위한 실험 결과, SVP 방법이 기존의 비주얼 프롬프팅 기법보다 현저히 더 나은 성능을 보였습니다. 모델 크기와 사전 학습 데이터셋은 동일하지만, SVP를 통해 차별적인 정보의 효과적인 전파 및 유지가 가능하다는 것이 입증되었습니다. 이러한 결과는 Vim 모델이 차세대의 모델 아키텍처로 자리매김함을 의미합니다.



### Dynamic Contrastive Knowledge Distillation for Efficient Image Restoration (https://arxiv.org/abs/2412.08939)
- **What's New**: 이번 논문에서는 이미지 복원을 위한 새로운 동적 대조 지식 증류(dynamic contrastive knowledge distillation, DCKD) 프레임워크를 제안합니다. 기존의 KD 방법들이 학습된 학생 모델의 상태를 반영하지 못하고 고정된 솔루션 공간에 의존했던 문제를 해결하고자 합니다. 최초로 학생의 학습 상태를 감지하고, 대조 학습을 통해 증류되는 솔루션 공간을 동적으로 조정하는 방식을 도입합니다.

- **Technical Details**: DCKD는 동적 대조 정규화(dynamic contrastive regularization)와 분포 매핑 모듈(distribution mapping module, DMM)을 통해 교사 모델과 학생 모델 간의 픽셀 수준의 카테고리 분포를 추출하고 정렬합니다. 이 방법은 L1 타입의 손실(L1 loss)만을 이용했던 기존 이미지 복원 증류 방법보다 더 효과적으로 카테고리 분포 정보(distibution information)를 도입하여 성능을 향상시킵니다. DCKD는 다양한 네트워크 백본(backbone)에 적응할 수 있으며, 상한 제약을 최적화하는 방법과 결합될 수 있습니다.

- **Performance Highlights**: DCKD는 이미지 슈퍼 해상도(super-resolution), 디블러링(deblurring), 비 오는 이미지 복원(deraining) 등 여러 이미지 복원 작업에서 기존의 최첨단 KD 방법들을 크게 능가하는 성능을 보여줍니다. 본 연구를 통해 동적 하한 제약을 도입하고 카테고리 분포 정보를 활용하여 이전의 이미지 복원 KD 방법들이 놓쳤던 점을 보완하고 있습니다. 폭넓은 실험을 통해 DCKD의 효과가 확인되었습니다.



### Deep clustering using adversarial net based clustering loss (https://arxiv.org/abs/2412.08933)
- **What's New**: 본 연구에서는 깊은 클러스터링(deep clustering) 기법을 적대적 네트워크(adversarial net)를 활용하여 재구성하였습니다. 이 과정에서 전통적인 KL 발산(KL divergence)을 사용하여 손실 함수를 최적화하며, 디스크리미네이터(discriminator)와 인코더(encoder) 간의 상호 작용을 통해 성능을 개선합니다. 이 방법은 기존의 깊은 클러스터링 방식보다 더 유연한 접근 방식을 제공하며, 효과적으로 학습할 수 있는 새로운 기회를 열어줍니다.

- **Technical Details**: 깊은 클러스터링은 잠재 공간(latent space)에서 샘플이 진리 클러스터 중심으로부터 이탈하는 것에 대해 패널티를 부과하는 손실 함수를 바탕으로 작동합니다. 기존 방식에서는 KL 발산을 사용하여 확률적 접근을 했으나, 본 연구에서는 적대적 네트워크를 통해 공동으로 학습할 수 있는 새로운 손실 함수를 제안합니다. 이를 통해 인코더는 데이터를 적절히 군집화할 수 있도록 최적화되고, 디스크리미네이터는 잘못 분류된 샘플에 대해 패널티를 부여하여 학습 과정이 강화됩니다.

- **Performance Highlights**: 제안된 방식은 SVHN, USPS, MNIST 및 CIFAR10과 같은 잘 알려진 데이터셋에서 실험을 통해 state-of-the-art 깊은 클러스터링 기법들과 대등하거나 향상된 성능을 달성하였습니다. 특히, 새롭게 제안된 DCAN(Deep Clustering using Adversarial Net)은 클러스터링 문제에서 탁월한 성능을 보이며, 데이터 분포를 효과적으로 모델링할 수 있는 가능성을 나타냅니다.



### CAPrompt: Cyclic Prompt Aggregation for Pre-Trained Model Based Class Incremental Learning (https://arxiv.org/abs/2412.08929)
Comments:
          in Proceedings of the 39th AAAI Conference on Artificial Intelligence (AAAI-25)

- **What's New**: 최근 CIL(Class Incremental Learning) 분야에서 프롬프트 튜닝 방법들이 유망한 성과를 보여주고 있습니다. 본 논문에서는 기존의 작업 ID 예측에 의존하지 않고, 여러 프롬프트의 지식을 순환적으로 집계하는 Cyclic Prompt Aggregation (CAPrompt) 방식을 제안합니다. 이 방법은 훈련 및 추론 단계에서 각각의 태스크 확률을 추정하여, 이를 기반으로 집계 프롬프트를 생성하여 지식의 일관성을 유지합니다.

- **Technical Details**: CAPrompt는 네트워크의 예측이 프롬프트 매개변수에 대해 오목한(condave) 조건을 만족하는 경우, 집계된 프롬프트가 단일 태스크 전용 프롬프트보다 예측 오류가 낮다는 것을 보장합니다. 이를 위해 오목 및 선형 제약 조건을 도입하며, 프롬프트 가중치를 주기적으로 조정하는 순환적인 가중치 예측 전략을 개발하였습니다. 이로 인해 각 태스크에 대한 초기 가중치가 균등하게 설정되고, 이후 적절한 가중치 값으로 자동으로 조정됩니다.

- **Performance Highlights**: 실험 결과, 제안한 CAPrompt 방법은 다양한 데이터셋에서 기존 최첨단 방법들에 비해 2%-3% 더 나은 성능을 나타내며, 이를 통해 CIL에서 프롬프트 기반 학습의 가능성을 더욱 끌어올리는 데 기여합니다. 이 연구는 CIL의 다양한 구현 방식과 모델의 기능을 확장하는 데 중요한 기초 자료로 작용할 것으로 기대됩니다.



### A Flexible Plug-and-Play Module for Generating Variable-Length (https://arxiv.org/abs/2412.08922)
- **What's New**: 이번 논문에서는 Nested Hash Layer (NHL)라는 새로운 모듈을 제안하여, 기존의 심층 감독 해싱(deep supervised hashing) 모델에서 해시 코드의 다양한 길이를 동시에 생성할 수 있도록 합니다. 기존 모델들은 특정 길이의 해시 코드 생성에만 집중하여 그 길이에 따른 비효율성과 효과성의 거래 관계를 해결하지 못했습니다. NHL 프레임워크는 다중 학습 목표에서 발생하는 최적화 충돌을 해결하기 위해 동적 가중치 조정 전략을 도입합니다.

- **Technical Details**: NHL은 단일 훈련 세션을 통해 다양한 길이의 해시 코드를 생성할 수 있도록 설계되었습니다. NHL의 기본 구조는 나중에 길어진 해시 코드가 짧은 해시 코드의 보조 설명으로 기능할 수 있다는 점에 기반하고 있습니다. 이를 통해 해시 코드의 길이에 따라 필요한 모델을 반복적으로 훈련하는 대신, 해시 코드 생성 과정에서 구조적 관계를 활용하여 효율성을 높일 수 있습니다.

- **Performance Highlights**: NHL은 훈련 과정을 가속화하고 다양한 심층 해싱 모델에서 우수한 검색 성능을 달성하는 것으로 나타났습니다. 실험 결과, NHL은 약 5-8배 정도의 훈련 속도 개선을 보이면서도 효과적인 검색 결과를 보장합니다. 이는 대규모 이미지 데이터베이스에서 해시 코드를 활용한 저장 및 검색 효율성을 극대화할 수 있는 중요한 발전으로 기여합니다.



### Sensing for Space Safety and Sustainability: A Deep Learning Approach with Vision Transformers (https://arxiv.org/abs/2412.08913)
Comments:
          To be published in the 12th Annual IEEE International Conference on Wireless for Space and Extreme Environments (WiSEE 2024)

- **What's New**: 저자는 최근 저궤도(Low Earth Orbit, LEO)에서의 소형 위성의 급격한 증가가 디지털 서비스를 혁신적으로 변화시킬 수 있음을 언급합니다. 하지만 우주 환경이 동적이기 때문에 이를 관리하기 위한 다양한 도전 과제가 있습니다. 이러한 문제 해결을 위해 저자들은 새로운 위성 객체 탐지(Satellite Object Detection, SOD) 모델을 제안하며, 기존의 YOLOv9 모델을 능가하는 성과를 기록했다고 설명합니다.

- **Technical Details**: 본 논문에서는 두 가지 새로운 딥러닝 모델, GELAN-ViT와 GELAN-RepViT를 제안합니다. 이들 모델은 Generalized Efficient Layer Aggregation Network (GELAN) 아키텍처에 비전 트랜스포머(Vision Transformer, ViT)를 통합하였고, 컨볼루션 신경망(Convolutional Neural Network, CNN)과 ViT 경로를 분리하여 한계점을 극복하였습니다. 이를 통해 고속 탐지성과 높은 정밀도를 유지하며, 자원 효율적인 접근을 제공합니다.

- **Performance Highlights**: 제안된 모델은 SOD 데이터셋에서 약 95% 평균 정확도(Mean Average Precision, mAP50)를 달성하며, giga-floating point operations (GFLOPs)는 5.0 이상 감소되었습니다. VOC 2012 데이터셋에서는 60.7% 이상의 mAP50을 달성하면서 GFLOPs는 5.2 이상 감소하여 뛰어난 성능을 입증하였습니다.



### Reversing the Damage: A QP-Aware Transformer-Diffusion Approach for 8K Video Restoration under Codec Compression (https://arxiv.org/abs/2412.08912)
Comments:
          12 pages, 8 figures

- **What's New**: 이 논문에서는 DiQP라는 새로운 Transformer-Diffusion 모델을 소개하여, 코덱 압축으로 인해 저하된 8K 비디오 품질을 복원합니다. 기존의 Denoising Diffusion 접근 방식을 통해 다양한 코덱(AV1, HEVC)에서 발생하는 아티팩트를 효과적으로 다루는 첫 번째 모델로, 추가적인 노이즈를 고려하지 않고 복원 작업을 수행합니다. 이를 통해 복잡하고 비가우시안적인 압축 아티팩트를 모델링하고, 이를 효과적으로 되돌리는 방법을 제시합니다.

- **Technical Details**: DiQP의 구조는 Transformer의 장거리 의존성을 포착하는 능력을 활용하고, 픽셀 그룹 내에서 시공간 컨텍스트를 보존하는 강화된 창(windowed) 메커니즘을 결합합니다. 모델은 'Look Ahead'와 'Look Around' 모듈을 통합하여 복원 과정을 더욱 향상시키며, 주변 프레임 및 미래 프레임 정보를 활용해 세밀한 디테일을 재구성합니다. 이를 통해 고해상도 비디오(4K 및 8K)의 복원 효과를 극대화합니다.

- **Performance Highlights**: 실험 결과, DiQP는 다양한 데이터 세트에서 기존 최첨단 방법들보다 우수한 성능을 보여주며, 특히 4K 및 8K와 같은 고해상도 비디오에서 뛰어난 복원 능력을 입증합니다. 이는 비디오의 시각적 품질을 높이고, 압축된 소스에서 만족스러우면서도 복원된 비디오를 생성하는 데 효과적입니다. DiQP의 접근 방식은 전통적인 비디오 복원 방법들과는 차별화된 성능을 보이며, 8K 콘텐츠 복원에 특화되어 있습니다.



### GaGA: Towards Interactive Global Geolocation Assistan (https://arxiv.org/abs/2412.08907)
- **What's New**: 이 논문에서는 GaGA라는 새로운 상호작용형 글로벌 지리 위치 보조 시스템을 소개합니다. GaGA는 대규모 비전-언어 모델(LVLMs)을 기반으로 하여 이미지 내의 지리적 단서를 발견한 후 LVLM에 내장된 광범위한 세계 지식과 결합하여 위치를 판별합니다. 또한 이 시스템은 예측 결과에 대한 정당성과 설명을 제공하여 사용자에게 더 나은 경험을 제공합니다.

- **Technical Details**: GaGA는 현재까지의 전통적인 정적 추론 방법을 초월하는 새로운 상호작용 지리 위치 결정 방법을 설계하였습니다. 사용자는 예측에 개입하거나 수정할 수 있는 기능을 통해 모델의 유연성과 실용성을 강화합니다. GaGA는 5백만 개의 고품질 이미지-텍스트 쌍으로 구성된 Multi-modal Global Geolocation (MG-Geo) 데이터셋을 바탕으로 개발되었습니다.

- **Performance Highlights**: GaGA는 GWS15k 데이터셋에서 최고 성능을 기록하며 국가 수준에서 4.57%, 도시 수준에서 2.92% 정확도를 향상시켰습니다. 또한, 사용자가 효과적인 지침을 제공할 경우, GaGA의 위치 결정 정확도가 상당히 개선되는 것을 보여주며, 이는 기존 모델의 단점을 극복하는 중요한 진전을 나타냅니다.



### LV-CadeNet: Long View Feature Convolution-Attention Fusion Encoder-Decoder Network for Clinical MEG Spike Detection (https://arxiv.org/abs/2412.08896)
- **What's New**: 본 연구에서는 MEG에서의 자동 뇌전증 스파이크 탐지를 위한 새로운 딥러닝 프레임워크인 LV-CadeNet을 소개하고 있습니다. LV-CadeNet은 임상 시험 데이터에서의 훈련 데이터 분포 불균형 문제를 해결하고, 인간 전문가의 분석 과정을 모방하여 긴 형태학적 입력 데이터를 구축하는 방식으로 설계되었습니다. 이를 통해 실제 임상 환경에서의 MEG 데이터에 대한 탐지 성능을 높이고자 하였습니다.

- **Technical Details**: LV-CadeNet은 컨볼루션(attention)과 공간 및 시간적 특징 추출을 위한 고급 모듈을 통합하여 설계되었습니다. 이 모델은 한 번의 MEG 신호를 긴 시간에 걸쳐 분석하여 스파이크 탐지의 정확도를 진전시킵니다. 특히, 반지도 학습(semi-supervised learning) 전략을 적용하여 훈련 데이터의 분포를 조정하고, 모델 미세 조정(process)에서 지식을 증류하는 방법을 채택하였습니다.

- **Performance Highlights**: LV-CadeNet은 Sanbo Brain Hospital Capital Medical University에서 수집된 새로운 임상 데이터셋에서 MEG 스파이크 탐지 정확도를 42.31%에서 54.88%로 향상시켰습니다. 이는 특히 불균형한 긍정 및 부정 샘플 분포를 가진 데이터셋에서 탁월한 성능을 보여주고, 임상 적용 가능성을 높입니다. 이 연구는 MEG 데이터 자동 탐지 기술의 개발에 크게 기여할 것으로 기대됩니다.



### Video Repurposing from User Generated Content: A Large-scale Dataset and Benchmark (https://arxiv.org/abs/2412.08879)
Comments:
          Accepted by AAAI2025

- **What's New**: 최근 소셜 미디어 플랫폼에서 짧은 형식의 비디오 제작 수요가 급증하고 있습니다. 본 연구는 10,000개 이상의 비디오와 120,000개 이상의 주석이 달린 클립을 포함하는 "Repurpose-10K"라는 대규모 데이터세트를 제안합니다. 이는 길이 제한을 두고 원본 비디오에서 매력적인 짧은 클립으로 재편집하는 과정을 다룬 최초의 데이터세트로, 다양한 비디오 콘텐츠를 이해하고 활용하는 단계적 솔루션을 제공합니다.

- **Technical Details**: 비디오 재편집 작업을 수행하기 위해, 우리는 음성 인식 모델을 통한 자막 개선 모듈과 다중 모달 정렬 가이더를 포함하는 종단 간 트랜스포머 인코더-디코더 아키텍처를 제안합니다. 각 모달리티의 정보를 최대한 활용하여 세그먼트를 적절히 선택하도록 유도하며, 이 과정에서 비디오, 오디오 및 자막의 통합을 보장합니다. 3단계(annotation) 접근 방식을 통해 각 클립의 기초 편집본을 생성하고, 사용자 피드백을 바탕으로 최종 주석 데이터를 수집합니다.

- **Performance Highlights**: 제안한 모델은 영상 하이라이트 탐지 태스크에서 우수한 성능을 발휘하며, 기존의 최신 모델들과의 비교를 통해 실제 적용 가능성을 입증합니다. 대규모 데이터세트의 활용과 진화된 아키텍처는 길이가 긴 비디오를 매력적인 짧은 클립으로 효과적으로 변환하는 데 기여합니다. 궁극적으로 우리는 이 연구가 비디오 재편집 분야에서 혁신적인 연구를 촉발하길 기대합니다.



### Inference-Time Diffusion Model Distillation (https://arxiv.org/abs/2412.08871)
Comments:
          Code: this https URL

- **What's New**: 이 연구에서는 Distillation++라는 새로운 inference-time distillation 프레임워크를 도입하여 teacher 모델과 student 모델 간의 성능 격차를 줄이고자 합니다. 기존의 방법들과 달리, 이 프레임워크는 샘플링 프로세스 전반에 걸쳐 teacher의 지도를 지속적으로 통합하여 데이터 없이도 denoising 프로세스를 실시간으로 개선할 수 있습니다. 또한, Distillation++는 다양한 student 모델과의 호환성을 가지고 있으며, 다양한 solver에 대한 일반적인 적용 가능성을 보여줍니다.

- **Technical Details**: Diffusion 모델은 고품질 샘플을 생성하기 위해 초기 노이즈 벡터를 점진적으로 denoise하는 iterative refinement 프로세스를 사용합니다. 이 모델은 확률 흐름 확률 미분 방정식(Probability Flow ODE, PF-ODE) 경로를 따라 적분을 직접 추정함으로써 teacher 모델을 distill하는 방식으로 연산 비용을 절감할 수 있습니다. 또한, Distillation++는 샘플링 경로를 score distillation sampling loss (SDS)를 통해 정규화하는 방식으로 학생 모델의 샘플링을 근접 최적화 문제로 재구성합니다.

- **Performance Highlights**: Distillation++는 특히 초기 샘플링 단계에서의 성능을 크게 향상시켜 주목받고 있으며, 이는 총 비용 없이 student 모델의 성능을 개선할 수 있음을 보여줍니다. 이 접근 방식은 기존의 가장 진보된 distillation 기법들에 비해 우수한 성능을 나타내며, 최종적으로 diffusion distillation 모델을 위한 강력한 guided sampling 프로세스를 제시합니다. Empirical 결과들은 Distillation++가 신뢰할 수 있는 post-training 솔루션으로 자리잡을 가능성을 증명합니다.



### ViUniT: Visual Unit Tests for More Robust Visual Programming (https://arxiv.org/abs/2412.08859)
- **What's New**: 이 논문은 ViUniT라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 시각적 프로그램의 신뢰성을 높이기 위해 자동으로 유닛 테스트를 생성하도록 설계되었습니다. ViUniT는 주어진 쿼리에 대해 생성된 프로그램의 논리적 정확성을 검증하기 위해 이미지 및 예상 답변 쌍으로 유닛 테스트를 표현합니다.

- **Technical Details**: ViUniT는 언어 모델을 활용하여 이미지 설명과 예상 답변의 형태로 유닛 테스트를 생성하고, 이에 상응하는 이미지를 생성하기 위해 이미지 합성 기술을 사용합니다. 이 방법은 주석이 없는 완전 비지도 방식으로 작동하며, 프로그램 입력 및 출력의 커버리지를 극대화하는 최적화 기준을 수립합니다. 또한, 유닛 테스트의 결과를 기반으로 개선된 프로그램 생성을 위한 다양한 메커니즘을 탐색합니다.

- **Performance Highlights**: ViUniT는 세 가지 데이터 세트와 두 가지 모델을 사용한 실험에서 모델 성능을 11.4% 향상시켰습니다. 특히, 7B 개방형 소스 모델이 gpt-4o-mini보다 평균 7.7% 더 뛰어난 성능을 발휘할 수 있도록 하였으며, 잘못된 이유로 옳은 프로그램의 비율을 40% 감소시켰습니다. 이로 인해 비주얼 프로그래밍 접근 방식의 신뢰성과 강건성이 향상될 것으로 기대됩니다.



### Labits: Layered Bidirectional Time Surfaces Representation for Event Camera-based Continuous Dense Trajectory Estimation (https://arxiv.org/abs/2412.08849)
Comments:
          24 pages, 12 figures, 9 tables

- **What's New**: 이 논문에서는 기존의 이벤트 카메라 기반 수치 표현의 한계를 극복하기 위해 Labits: Layered Bidirectional Time Surfaces 라는 새로운 표현 방법을 제안합니다. Labits는 모든 중요한 특성을 보존하면서 이벤트 카메라의 비동기적 특성을 인지하고 이를 활용하는 혁신적인 솔루션입니다. 이로 인해 기존 기술에 비해 트레일리감 엔드포인트 오차(Trajectory End-Point Error, TEPE)를 49% 줄이는 성과를 올렸습니다.

- **Technical Details**: 이벤트 카메라는 비상동형 동시 비전 센서로, 사전 정의된 임계치 이상의 밝기 변화가 감지되면 즉시 이벤트를 발생시킵니다. 각 이벤트는 위치(x,y), 순간적인 시간(t), 그리고 이진 극성(p) 정보를 기록하며, 이를 통해 정밀한 이동 객체 추적이 가능합니다. 그러나 기존의 데이터 변환 방식인 GNN(Graph Neural Network)이나 SNN(Spiking Neural Network)의 한계로 인해 이벤트 기반 비전의 널리 사용에 제약이 있었습니다.

- **Performance Highlights**: Labits를 적용한 결과, TEPE의 13% 개선이 있었으며, 추가적인 APLOF(Active Pixel Local Optical Flow) 모듈을 도입함으로써 오류를 추가로 30% 줄일 수 있었습니다. 이러한 성과는 이벤트 카메라의 강점을 최대한 활용한 결과로, 자동차나 로봇 등 다양한 분야에서의 응용 가능성을 높였습니다.



### DALI: Domain Adaptive LiDAR Object Detection via Distribution-level and Instance-level Pseudo Label Denoising (https://arxiv.org/abs/2412.08806)
- **What's New**: 본 논문에서는 Domain Adaptive LIdar (DALI) 객체 감지 프레임워크를 소개하며, 이를 통해 객체 검출에서 발생하는 노이즈 문제를 해결하고자 합니다. 본 연구는 주로 보정된 크기 노멀라이제이션(post-training size normalization, PTSN)과 두 가지 종류의 의사 포인트 클라우드 생성(pseudo point clouds generation, PPCG) 전략을 도입합니다. 이러한 접근 방식은 성능 향상에 기여하며, 기존의 최첨단 기법에 비해 뛰어난 결과를 보여줍니다.

- **Technical Details**: DALI 프레임워크는 모델 훈련 후에 발생하는 의사 라벨 크기 분포의 편향을 줄이기 위해 PTSN 전략을 사용합니다. 또한, 두 가지 PPCG 전략, 즉 레이 기반 제약(ray-constrained)과 제약 없는(constraint-free) 방식을 사용하여 각 인스턴스에 대해 일관성 있는 의사 포인트 클라우드를 생성합니다. 이를 통해 훈련 과정에서 의사 라벨과 포인트 클라우드 간의 불일치를 해결하고, 최적의 바이어스 없는 스케일을 식별함으로써 문제를 완화합니다.

- **Performance Highlights**: KITTI, Waymo, nuScenes와 같은 널리 사용되는 데이터셋을 통해 실험한 결과, DALI 프레임워크는 대부분의 도메인 적응 작업에서 기존의 최신 기법들을 초월하는 성능을 기록했습니다. 본 연구의 방법론은 소스 도메인과 타겟 도메인 모두에서 경쟁력 있는 결과를 달성하면서 노이즈 처리 문제를 효과적으로 해결함을 입증하였습니다. 성능 향상은 객체 인식 정확도 및 효율성에 기여하며, 자율 주행과 같은 관련 응용 프로그램에 실질적인 이점을 제공합니다.



### Generative Modeling with Explicit Memory (https://arxiv.org/abs/2412.08781)
Comments:
          10 pages, 5 figures, 4 tables

- **What's New**: 이 논문에서는 GMem(Generative Modeling with Explicit Memory)이라는 새로운 접근법을 제시합니다. 이 방법은 외부 메모리 뱅크를 통해 데이터 분포에서 의미론적 정보를 보존하면서 다양한 데이터 세트에서 학습 및 일반화할 수 있는 능력을 향상시킵니다. GMem은 큰 신경망의 용량에 대한 의존성을 줄이고, 효율적인 학습 및 샘플링을 가능하게 합니다.

- **Technical Details**: GMem은 훈련 및 샘플링 과정에서 외부 메모리 뱅크를 활용하여 생성 모델의 성능을 극대화합니다. 이 접근법은 신경망 안에서 데이터를 암기하지 않고도 진정한 데이터 분포를 근사할 수 있도록 하여, 훈련 단계에서 상당한 신경망 매개변수와 계산 자원을 필요로 하지 않게 만듭니다. 결과적으로, 모델은 샘플링 과정에서 가우시안 노이즈를 내부 의미론적 분포로 변환한 뒤 데이터를 재구성하는 방식을 취합니다.

- **Performance Highlights**: GMem은 ImageNet 데이터셋에서 SiT(Semi-implicit Transforms) 모델을 사용할 때 약 46.7배 이상 가속화할 수 있으며, 150K의 스텝으로 7M 스텝에서 얻는 것과 같은 성능을 달성합니다. 또한, GMem은 최적의 기존 방법인 REPA보다도 16배 빠르며 약 250K 스텝 내에 FID 스코어 5.75를 기록했습니다. GMem은 2M 스텝 훈련 후 FID 스코어 3.56을 달성하며, classifier-free guidance를 사용하지 않고도 최첨단 성능을 나타냅니다.



### ProtoOcc: Accurate, Efficient 3D Occupancy Prediction Using Dual Branch Encoder-Prototype Query Decoder (https://arxiv.org/abs/2412.08774)
Comments:
          Accepted to AAAI Conference on Artificial Intelligence 2025, 9 pages, 5 figures

- **What's New**: 이 논문에서는 3D 점유 예측을 새로운 차원으로 끌어올리는 ProtoOcc 모델을 소개합니다. ProtoOcc는 Dual Branch Encoder(DBE)와 Prototype Query Decoder(PQD)라는 두 가지 주요 구성 요소로 이루어져 있으며, 이들 구조는 3D voxel과 BEV(Bird’s-Eye View) 표현을 통해 장면을 이해합니다. 따라서, ProtoOcc는 단일 단계에서 3D 점유 예측을 가능하게 하며 반복적인 Transformer 디코딩을 제거함으로써 높은 효율성을 자랑합니다.

- **Technical Details**: ProtoOcc의 Dual Branch Encoder(DBE)는 다중 스케일을 통해 3D voxel과 BEV 표현을 결합하여 새로운 3D voxel 표현을 생성합니다. 이 구조는 BEV 표현의 큰 수용 영역을 제공하는 동시에 voxel 표현을 위한 작은 수용 영역을 유지하여 성능과 계산 효율성을 향상시킵니다. Prototype Query Decoder(PQD)는 Scene-Adaptive Prototypes와 Scene-Agnostic Prototypes를 활용하여 디코딩 과정을 가속화하고 Robust Prototype Learning 방법을 통해 노이즈를 제거합니다.

- **Performance Highlights**: ProtoOcc는 Occ3D-nuScenes 벤치마크에서 45.02%의 mIoU를 달성하며 최신의 성능을 보여줍니다. 단일 프레임 방법으로는 39.56%의 mIoU를 기록하고, NVIDIA RTX 3090에서 12.83 FPS의 추론 속도를 자랑합니다. 이러한 성과 덕분에 ProtoOcc는 이미지에서 3D 점유 예측을 수행하는 데 필요한 계산 복잡성을 대폭 줄이면서도 높은 정확도를 유지하는 데 성공했습니다.



### LLaVA-Zip: Adaptive Visual Token Compression with Intrinsic Image Information (https://arxiv.org/abs/2412.08771)
- **What's New**: 이 논문에서는 시각 토큰의 과부하 문제를 해결하기 위해 DFMR(Dynamic Feature Map Reduction)라는 새로운 방법을 제안합니다. DFMR는 이미지를 기반으로 시각 토큰을 동적으로 압축하여 최대 토큰 한도를 효율적으로 관리합니다. 이는 자원 제약이 있는 학술 환경과 산업 환경 모두에 적용 가능하여, 멀티 이미지 및 비디오 시나리오를 처리하는 데 유리합니다.

- **Technical Details**: DFMR는 LLaVA-1.5 모델에 통합되어 시각 토큰의 압축 비율을 이미지의 내재적 정보에 따라 조정합니다. 이 모델의 구조는 비전 인코더와 프로젝터 사이에 DFMR 모듈이 삽입된 형태로 설계되었습니다. 각 입력 이미지는 비전 인코더를 통해 시각 토큰으로 변환된 후, DFMR 모듈을 지나 압축되어 최종적으로 LLM에 전달됩니다.

- **Performance Highlights**: 실험 결과, DFMR를 적용한 LLaVA-1.5는 다양한 멀티모달 평가 기준에서 성능이 개선되었음을 보여줍니다. 특히, DFMR은 압축된 시각 토큰을 활용할 때 모든 여덟 개의 업무에서 성능이 향상되었습니다. 이러한 결과는 DFMR이 시각 토큰 과부하 문제를 해결하는 유망한 방법임을 시사합니다.



### Beyond Knowledge Silos: Task Fingerprinting for Democratization of Medical Imaging AI (https://arxiv.org/abs/2412.08763)
- **What's New**: 현재 의료 이미징 AI 분야는 급속한 변화를 겪고 있으며, 연구 결과가 임상 실무로 점차 구체화되고 있습니다. 최근의 연구는 지식 고립 문제를 강조하며, 데이터 공유와 협업의 어려움을 해결하기 위해 새로운 프레임워크를 제안하고 있습니다. 이 프레임워크의 핵심은 데이터셋의 '지문(fingerprints)'을 활용하여 작업 유사성을 정량화할 수 있는 구조적 표현을 만드는 것입니다.

- **Technical Details**: 제안된 프레임워크는 지식 클라우드(knowledge cloud)를 중심으로 구성되며, 모델 훈련 경험과 이를 통해 생성된 작업 지문(task fingerprint)를 저장합니다. 사용자는 자신의 작업 지문을 생성하여 기존의 관련 지식이나 데이터를 쿼리할 수 있습니다. 주요 요소인 binned Kullback-Leibler Divergence(bKLD)은 추출된 이미지 특성을 바인딩하여 작업 간 유사성을 계산하는 효율적이고 일반화 가능한 방법입니다.

- **Performance Highlights**: 71개의 서로 다른 작업과 12개의 의료 이미징 모달리티에 대한 포괄적인 분석을 통해, 제안된 방법이 기존의 전통적인 방법보다 우수하다는 결과를 도출했습니다. 협동적 모델 훈련을 통해 67%의 검증 작업에서 성능 향상을 보였으며, 특정 조건에서 최대 90%의 작업 개선률을 달성했습니다. 이 프레임워크는 의료 이미징 분야에서 AI의 민주화를 촉진하며, 과학 발전 가속화에 기여할 수 있는 중요한 도구로 자리잡을 것입니다.



### Proactive Adversarial Defense: Harnessing Prompt Tuning in Vision-Language Models to Detect Unseen Backdoored Images (https://arxiv.org/abs/2412.08755)
- **What's New**: 이 논문은 백도어 공격(Backdoor attacks)을 탐지하기 위한 새로운 방법을 제안합니다. 기존의 연구들은 이러한 공격을 방지하기 위해 모델의 파라미터를 조정하는 데 집중했지만, 본 연구는 백도어 공격으로부터 모델을 사전 예방적으로 보호하는 알고리즘을 개발했습니다. 이를 통해 훈련 및 추론 과정에서 보이지 않는 백도어 이미지를 효과적으로 분류할 수 있습니다.

- **Technical Details**: 이 방법은 Vision Language Models(VLMs)의 성공적인 프롬프트 튜닝(prompt tuning)을 활용하여 작동합니다. 백도어 공격을 탐지하기 위해 학습 가능한 텍스트 프롬프트를 훈련시켜, 정상 이미지와 백도어 트리거가 숨겨진 이미지를 구별합니다. 이 과정에서 기존의 공격 패턴을 분석하고, 보다 정교한 데이터 셋을 통해 불순 이미지들을 사전에 제거하는 방식을 채택합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 두 개의 유명한 데이터 세트에서 백도어 트리거 탐지 정확도 86%를 달성했습니다. 이는 이전의 방어 기법들에 비해 획기적인 성과로, 백도어 방어의 새로운 기준을 설정합니다. 이 연구는 객체 인식 시스템 보안 강화에 큰 기여를 할 것으로 기대됩니다.



### DocVLM: Make Your VLM an Efficient Reader (https://arxiv.org/abs/2412.08746)
- **What's New**: DocVLM은 OCR 정보(OCR information)를 VLMs에 효율적으로 통합하여 문서 인식(document understanding) 기능을 향상시키는 방법입니다. 이 방법은 OCR 인코더(OCR encoder)를 사용하여 텍스트와 레이아웃(layout) 정보를 캡처하고 이를 압축된 쿼리 세트(learned queries)로 변환합니다. 이를 통해 고해상도 이미지에 대한 의존도를 크게 낮출 수 있습니다.

- **Technical Details**: DocVLM은 문서 인식 정확도를 유지하면서 저해상도 입력을 사용하는 VLM의 읽기 능력을 향상시킵니다. 이 모델은 64개의 학습된 쿼리(learned queries)를 사용하여 OCR 정보를 압축하고 이를 VLM에 통합하여 시각적 특징과 함께 처리합니다. 기존의 VLM 가중치를 보존하여 다양한 모델 아키텍처에 쉽게 통합될 수 있도록 설계되었습니다.

- **Performance Highlights**: DocVLM은 다양한 VLM 아키텍처에서 실험을 통해 뛰어난 성능을 보여주었습니다. 특히, 입력 토큰 수가 제한된 상황에서(448x448) InternVL2와 결합한 경우 DocVQA 결과가 56.0%에서 86.6%로, Qwen2-VL과의 조합에서는 84.4%에서 91.2%로 향상되었습니다. 또한, LLaVA-OneVision을 사용하는 경우, 이미지 토큰을 80% 줄이면서도 성능을 향상시키며 멀티페이지 문서 처리에서도 효과적입니다.



### Euclid: Supercharging Multimodal LLMs with Synthetic High-Fidelity Visual Descriptions (https://arxiv.org/abs/2412.08737)
Comments:
          33 pages, 22 figures, 5 tables, 7 algorithms

- **What's New**: 최근 멀티모달 대형 언어 모델(MLLMs)은 비약적인 발전을 이루었지만, 여전히 저수준 시각 인식(LLVP)에서는 어려움을 겪고 있습니다. 본 논문에서는 'Geoperception'이라는 벤치마크를 소개하여 MLLM의 2D 기하학적 정보 전사 능력을 평가합니다. 이 벤치마크를 통해 기존 MLLM의 한계를 드러내고, 기하학적 작업 성능을 향상시키기 위한 전략을 연구하게 됩니다.

- **Technical Details**: Geoperception 벤치마크에서는 MLLM들이 이미지에서 기하학적 세부사항을 정확하게 설명하는 능력을 평가합니다. 연구 결과, 특정 모델 아키텍처, 훈련 기법, 데이터 전략이 기하학적 작업을 향상시키는 데 도움이 된다는 것을 발견했습니다. 특히, 데이터 커리큘럼(data curriculum)을 통해 모델은 처음부터 학습하지 못하는 어려운 기하학적 이해 작업을 학습할 수 있음을 보여줍니다.

- **Performance Highlights**: Euclid이라는 모델 계열이 강력한 저수준 기하학적 인식을 위해 최적화되었습니다. Euclid는 순전히 합성된 멀티모달 데이터로 훈련되었음에도 불구하고, 새로운 기하학적 형태에 대한 일반화 능력이 뛰어난 성능을 보여줍니다. 예를 들어, Euclid는 Geoperception 벤치마크의 특정 작업에서 최고의 비공식 모델인 Gemini-1.5-Pro보다 최대 58.56% 향상된 성능을 기록하고, 전체 작업 평균으로는 10.65% 더 나은 성과를 보입니다.



### VisionArena: 230K Real World User-VLM Conversations with Preference Labels (https://arxiv.org/abs/2412.08687)
- **What's New**: 비전-언어 모델(Vision-Language Models, VLMs)의 채택이 증가함에 따라, 실제 사용자와 VLM 간의 상호작용을 포착할 수 있는 벤치마크의 필요성이 대두되었습니다. 이를 해결하기 위해 VisionArena라는 데이터셋을 개발하였습니다. 이 데이터셋은 230,000개의 실제 대화로 구성되어 있으며, 73,000명의 고유 사용자가 45개의 VLM과 138가지 언어로 이루어진 대화를 포함하고 있습니다.

- **Technical Details**: VisionArena는 세 가지 하위 집합으로 나뉘어 있습니다: VisionArena-Chat은 사용자와 VLM 간의 20만 건의 단일 및 다중 턴 대화로 구성되며, VisionArena-Battle은 두 개의 익명의 VLM을 비교하는 30,000개의 대화와 사용자 선호 투표로 구성됩니다. 마지막으로 VisionArena-Bench는 500개의 다양한 사용자 프롬프트로 구성된 자동 벤치마크입니다. 이 데이터셋은 실제 Chatbot Arena 모델 순위를 효율적으로 근사합니다.

- **Performance Highlights**: 사용자 질문 유형, 응답 스타일의 선호도에 대한 영향, 모델의 실패 영역 등을 강조합니다. 개방형 작업인 캡셔닝(captioning)과 유머에 대한 스타일 의존성이 높다는 것을 발견하였고, 현재 VLM은 공간 추론(spatial reasoning)과 계획 작업에서 어려움을 겪고 있습니다. 비록 VisionArena-Chat에서 동일한 기본 모델의 파인튜닝(fine-tuning)이 Llava-Instruct-158K를 능가하고, MMMU에서 17점, WildVision 벤치마크에서 46점 향상된 성과를 보였습니다.



### ChatDyn: Language-Driven Multi-Actor Dynamics Generation in Street Scenes (https://arxiv.org/abs/2412.08685)
- **What's New**: 이 논문에서는 ChatDyn이라는 새로운 시스템을 소개합니다. ChatDyn은 언어 지침을 기반으로 다양한 교통 참가자의 상호작용적이고 사실적인 동적 행동을 생성하는 최초의 시스템입니다. 또한, 다양한 참가자들의 복잡한 상호작용을 고려하여 높은 수준의 계획과 낮은 수준의 제어를 결합한 방법론을 개발했습니다.

- **Technical Details**: ChatDyn은 다중 LLM 에이전트 역할 수행 방식을 사용하여 사용자 언어 지침을 해석하고 고급 계획을 수립합니다. PedExecutor와 VehExecutor라는 두 가지 새로운 실행 프로그램을 통해 보행자와 차량 동작을 사실적으로 생성합니다. PedExecutor는 여러 계획 작업을 수행할 수 있는 통합된 다중 작업 실행기로, VehExecutor는 물리적 전이 과정을 기반으로 차량 동작을 생성하는 제어 정책입니다.

- **Performance Highlights**: 실험 결과, ChatDyn은 여러 차량과 보행자가 포함된 현실적인 주행 장면 동적 행동을 생성하며, 이전 방법 대비 매우 우수한 성능을 보였습니다. 이런 성과는 높은 상호작용성, 높은 제어 가능성 및 사실적인 결과를 달성하는 데 기여하였습니다. 코드와 모델은 제공된 링크를 통해 이용할 수 있습니다.



### Coherent3D: Coherent 3D Portrait Video Reconstruction via Triplane Fusion (https://arxiv.org/abs/2412.08684)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2405.00794

- **What's New**: 최근 단일 이미지 3D 초상화 재구성 분야에서 획기적인 발전이 이루어졌습니다. 이를 통해 원거리에서 사람들을 얼굴을 맞대어 만날 수 있는 telepresence 시스템이 가능해졌습니다. 하지만 기존 시스템은 시간이 지남에 따라 사용자 외형을 잊고 일관성이 결여된 3D 재구성을 보여주는 문제가 있었습니다. 이 논문에서는 동적 퍼프레임 외형을 유지하면서 일관된 정체성을 유지하는 새로운 솔루션을 제안합니다.

- **Technical Details**: 제안하는 방법은 기준 뷰에서의 canonical 3D prior와 동적 외형을 결합하여, 사용자 외형의 무결성을 유지하며 시간적으로 안정적인 3D 비디오를 생성하는 fusion-based 방법입니다. 이모션 조건부 3D GAN을 사용해 생성된 합성 데이터를 통해 훈련된 이 encoder 기반 방법은 stae-of-the-art 3D 재구성과 시간 일관성을 동시에 달성합니다. 또한, LP3D를 사용하여 사용자에 대한 canonical triplane prior를 구성하고, 비디오 재구성 중 각 입력 프레임을 raw triplane으로 처리합니다.

- **Performance Highlights**: 제안하는 방법은 다수의 데이터셋에서 in-studio 및 in-the-wild 평가를 통해 시간적 일관성과 재구성 정확성을 모두 향상시킵니다. 기존의 방법들은 이 두 가지 특성을 동시에 만족하지 못했던 반면, 우리의 방법은 사용자 외형의 조명 및 표현을 충실히 재구성하며 탁월한 성능을 보여줍니다. 특히, 우리의 triplane fusion 방법은 predefined canonical triplane과 동적 정보의 정확한 결합을 통해 현실감 넘치는 3D 초상화 비디오를 생성하는 가능성을 보여줍니다.



### A Deep Semantic Segmentation Network with Semantic and Contextual Refinements (https://arxiv.org/abs/2412.08671)
Comments:
          Accept by tmm

- **What's New**: 이 논문은 Semantic Refinement Module (SRM)과 Contextual Refinement Module (CRM)을 도입하여 효과적인 semantic segmentation을 개선하는 방법을 제안합니다. SRM은 업샘플링된 feature maps의 각 픽셀에 대한 transformations 오프셋을 배우고 이웃 픽셀의 오프셋 영향을 고려하여 객체 경계 주변의 semantic label 지정을 향상시킵니다. CRM은 공간과 채널 차원 전반에 걸쳐 전역 컨텍스트 정보를 포착하기 위해 attention 메커니즘을 활용합니다.

- **Technical Details**: SRM은 feature map의 각 픽셀에 대한 오프셋을 추정하기 위해 고해상도 feature maps와 이웃 오프셋 정보를 사용합니다. 이는 높은 정확도로 객체 경계의 픽셀에 semantic label을 부여하는 데 기여합니다. CRM은 채널과 공간의 차원에서 픽셀 간 의존성을 탐구하여 전역 컨텍스트 정보를 효과적으로 포착합니다. 이 논문은 SRM과 CRM을 다양한 segmentation 네트워크에 적용하여 그 효율성을 검증합니다.

- **Performance Highlights**: 제안된 모듈은 Cityscapes, Bdd100K, ADE20K 데이터셋에서 최고의 성능을 보여줍니다. Cityscapes 검증 세트에서 단 137.9 GFLOPs의 계산량으로 82.5%의 mIoU를 달성하며, lightweight segmentation 네트워크에 효과적으로 확장됩니다. 이러한 결과는 SRM과 CRM이 semantic segmentation 성능을 개선하는 데 중요한 역할을 함을 나타냅니다.



### A feature refinement module for light-weight semantic segmentation network (https://arxiv.org/abs/2412.08670)
Comments:
          Accept by icip 2023

- **What's New**: 이번 논문에서는 경량화된 네트워크의 표현 능력을 강화하기 위한 새로운 의미 분할(semantic segmentation) 방법을 제안합니다. 특히, 다단계 특징 맵에서 정보 추출 및 비근접적(contextual) 정보를 포착하는 feature refinement module (FRM)을 도입하여 정확성과 속도를 동시에 확보할 수 있도록 합니다. 실험 결과는 Cityscapes와 Bdd100K 데이터셋에서 성능이 향상됨을 보여줍니다.

- **Technical Details**: 제안된 방법은 인코더-디코더 구조에 기반하고 있으며, 네트워크의 네 가지 단계에서 출력된 특징을 집계하는 feature refinement module (FRM)을 포함합니다. FRM은 다단계 정보를 동일한 크기로 풀링한 후, 이를 연결하여 풍부한 의미 정보를 추출합니다. 또한, disentangled non-local block (DNL)을 사용하여 특징 맵의 서로 다른 위치 간의 상관관계를 평가하여 문맥 정보를 포착합니다.

- **Performance Highlights**: 이 방법은 Cityscapes 테스트 세트에서 80.4%의 mIoU를 달성하며, 214.82 GFLOPs의 계산 비용을 요구하는 훌륭한 성능을 보여줍니다. 또한, 이 방법은 경량 네트워크와 기존 방법들 간의 정확성과 계산 비용에서 더 나은 균형을 이루고 있습니다. Bdd100K 데이터셋에서도 우수한 성능을 기록하여 방법의 유효성을 확보했습니다.



### Neptune: The Long Orbit to Benchmarking Long Video Understanding (https://arxiv.org/abs/2412.09582)
- **What's New**: 이 논문은 긴 동영상을 이해하기 위한 도전적인 질문-답변-혼란 집합(Question-Answer-Decoy Sets)을 생성하는 반자동 파이프라인을 소개합니다. 기존의 비디오 데이터셋은 주로 10초에서 30초의 짧은 클립에 중점을 두고 있어, 긴 비디오 이해 측면에서의 한계를 극복할 필요가 있었습니다. 본 연구에서는 VLMs(비디오 언어 모델)와 LLMs(대규모 언어 모델)를 활용하여 자동으로 밀도가 높은 시간 정렬 비디오 캡션 및 도전적인 질문 답변 세트를 생성하는 확장 가능한 데이터셋 생성 파이프라인을 제안합니다.

- **Technical Details**: 제안된 파이프라인은 YouTube의 모든 비디오에 적용 가능하며, 자동 캡션 생성, 자동 음성 인식(ASR), 샷 경계 및 비디오 메타데이터 추출 과정을 포함합니다. 이 과정에서 LLM에게 체계적인 프롬프트를 통해 다단계 사고 과정을 거쳐 이미지 캡션과 결합하여 질문-답변-혼란 집합(QAD)을 만듭니다. 대다수의 파이프라인은 자동화되어 있으며, 마지막 단계에서 품질을 보증하기 위한 rater 검증 단계를 포함합니다. 최종적으로 Neptune이라는 데이터셋이 생성되며, 다양한 비디오와 여러 모드의 평가를 제공합니다.

- **Performance Highlights**: Neptune 데이터셋은 2,405개 비디오에 대해 3,268개의 QAD 주석을 포함하여 평가 전용으로 설계되었습니다. 이 데이터셋은 시간 순서, 카운팅 및 상태 변화와 같은 질문들에서 현재 공개된 긴 비디오 모델들이 부족한 성능을 보인다는 점이 부각되었습니다. 또한 새로운 공개 소스 모델 기반 메트릭인 GEM(Gemma Equivalence Metric)을 도입하여 개방형 질문 응답의 평가 지표를 제공합니다. 연구 결과는 기존 공개 도구와 비공식 모델 간의 성능 차이를 명확하게 나타내어, 더 발전된 모델 개발을 촉진할 수 있는 계기가 될 것입니다.



### Meshtron: High-Fidelity, Artist-Like 3D Mesh Generation at Sca (https://arxiv.org/abs/2412.09548)
Comments:
          Project page: this https URL

- **What's New**: Meshtron은 64K 면을 갖고 1024 수준의 좌표 해상도로 메쉬를 생성할 수 있는 새로운 오토 회귀(mesh generation) 모델입니다. 이는 기존 최첨단 방법보다 면 수가 40배, 좌표 해상도가 8배 더 높습니다. 이 모델은 아트웍에서 생성된 메쉬와 유사한 높은 품질의 메쉬 생성을 가능하게 해주어, 애니메이션, 게임 및 가상 환경을 위한 보다 현실적이고 세밀한 3D 자산 생성을 열어줍니다.

- **Technical Details**: Meshtron은 네 가지 주요 구성 요소에 의해 확장됩니다: (1) hourglass neural architecture, (2) truncated sequence training, (3) sliding window inference, (4) robust sampling strategy입니다. 이러한 요소들은 훈련 메모리를 50% 이상 줄이고, 처리 속도를 2.5배 높이며, 기존 모델보다 일관성을 더 높입니다. 이로 인해 Neural Network(NN) 기반의 메쉬 생성을 보다 효율적으로 수행할 수 있습니다.

- **Performance Highlights**: Meshtron은 예술가 품질의 메쉬 생성을 자랑하며, 자세한 기하학, 높은 품질의 토폴로지, 그리고 다양한 형태를 제공합니다. 손쉬운 배포와 데이터 병렬화 설정을 통해 11억 개의 매개변수를 가진 모델을 훈련하였으며, 이는 큰 메쉬를 생산하는 데 있어 신속함과 효율성을 높여줍니다. 이 모델은 훨씬 더 대규모의 트레이닝 데이터를 통해 작동하여, 기존의 메쉬 생성 모델들보다 높은 성능을 인증받았습니다.



### Video Seal: Open and Efficient Video Watermarking (https://arxiv.org/abs/2412.09492)
Comments:
          Code available at this https URL

- **What's New**: 이 논문에서는 Video Seal이라는 포괄적인 프레임워크를 소개하며, 이는 신경망 기반의 비디오 워터마킹(neural video watermarking)을 위한 공개 모델로 활발한 연구를 촉진하는 것을 목표로 하고 있습니다. 기존의 비디오 워터마킹 기법의 한계를 극복하기 위해 이 모델은 임베더(embedder)와 이엑스트랙터(extractor)를 함께 훈련시켜 높은 강인성(robustness)을 유지합니다. 또한, 이 논문에서는 임베딩을 위한 새로운 기법인 temporal watermark propagation을 도입하여 모든 고해상도 프레임에 워터마크를 추가하지 않고도 효율적인 비디오 워터마킹을 가능하게 합니다.

- **Technical Details**: 기법적으로, Video Seal은 2D 차원에서 작동하여 스트리밍(streamability)을 보장하고 추출을 간소화하며 유연성을 제공합니다. 이 모델은 고해상도 및 다양한 길이의 비디오에 효과적으로 적용될 수 있도록 설계되었으며, 임베더는 효율적인 U-Net 아키텍처를 사용하고 이엑스트랙터는 비전 변환기(vision transformer)를 기반으로 구성되어 유연한 추론을 가능하게 합니다. 모델은 이미지 사전 훈련(image pre-training) 및 하이브리드 후 훈련(hybrid post-training)을 포함한 다단계 훈련 방식을 적용하여 다양한 비디오 변환 및 고압축이 일어나는 상황에서도 높은 성능을 유지합니다.

- **Performance Highlights**: 실험 결과는 Video Seal이 기존의 강력한 베이스라인(like MBRS, TrustMark, and WAM)보다 높은 강인성을 달성하며, 특히 지리적 변형 및 비디오 압축과 같은 도전에 잘 대응함을 보여줍니다. 또한, PSNR을 통해 품질 저하 없이 비트 정확성과 강인성을 향상시킬 수 있는 이엑스트랙터 미세 조정(extractor fine-tuning)의 중요성이 강조되었습니다. Released artifacts, including 모델 체크포인트와 평가 코드들은 연구자들이 비디오 워터마킹 기술의 진전을 촉진할 수 있도록 공유되고 있습니다.



### Embeddings are all you need! Achieving High Performance Medical Image Classification through Training-Free Embedding Analysis (https://arxiv.org/abs/2412.09445)
Comments:
          15 pages, 7 figures, 3 tables

- **What's New**: 이 연구에서는 전통적인 훈련 절차를 치환할 수 있는 임베딩 기반 접근 방식을 제안하고 있습니다. 이는 의학 이미지를 간결하고 의미 있는 표현으로 변환하여 기존의 자원 소모를 줄이면서도 유사한 혹은 더 나은 진단 성능을 달성하는 방법입니다. 이 방식은 CNN(Convolutional Neural Networks) 및 CLIP(Contrastive Language-Image Pre-training)와 같은 사전 훈련된 모델을 활용합니다.

- **Technical Details**: 연구에서는 여러 개의 의학 이미지 모달리티—망막 이미지, 유방 촬영술, 피부과 사진, 흉부 엑스레이—를 사용하여 임베딩을 생성하고, 이 임베딩에 간단한 선형 분류기를 적용했습니다. 결과적으로, 임베딩 기반 모델은 전통적인 방법으로 훈련된 벤치마크 모델들보다 AUC-ROC(Receiver Operating Characteristic) 점수가 최대 87% 향상되었습니다.

- **Performance Highlights**: 특히, CLIP 임베딩 모델이 가장 높은 AUC-ROC 점수를 달성하여 뛰어난 분류 성능을 보였으며, 계산 요구 사항도 크게 줄였습니다. 이 연구는 임베딩을 사용한 접근 방식이 의학 이미지 분석에서 일반적인 자원 집약적인 훈련과 테스트 절차를 효과적으로 대체할 수 있음을 입증했습니다.



### MOS: Model Surgery for Pre-Trained Model-Based Class-Incremental Learning (https://arxiv.org/abs/2412.09441)
Comments:
          Accepted to AAAI 2025. Code is available at: this https URL

- **What's New**: 이번 논문에서는 MOdel Surgery (MOS)라는 새로운 접근법을 통해 Class-Incremental Learning (CIL)에서 발생하는 과거 지식의 망각 문제를 해결하고자 합니다. MOS는 사전 훈련된 모델(Pre-trained Models, PTMs)을 기반으로, 파라미터 레벨과 검색(retrieval) 레벨 모두에서 지식 손실을 효과적으로 방지하는 방법을 제공합니다. MOS는 특히 다양한 다운스트림 작업에 대한 적합성을 강조하며, 적응형 변환기를 통하여 새로운 작업을 지속적으로 조정합니다.

- **Technical Details**: MOS는 교육(training) 및 추론(inference) 단계로 나뉘어 있습니다. 교육 단계에서는 작업 특화된 적응기를 학습하여 파라미터 수준의 망각을 최소화하는 어댑터 병합(adaptive merging) 접근 방식을 제시합니다. 반면, 추론 단계에서는 훈련이 필요 없는 자기 정제(self-refining) 어댑터 검색 메커니즘을 도입하여, 모델의 내재적 능력을 활용하여 더 나은 어댑터 검색을 수행합니다.

- **Performance Highlights**: MOS는 7개의 벤치마크 데이터셋에서 실시된 실험을 통해 최첨단 성능을 입증하였습니다. 이 시스템은 여러 단계에서의 모델의 능력을 통합할 수 있는 앙상블 방법을 제공합니다. 또한, 자기 정제 어댑터 검색 메커니즘의 시각화 결과, MOS가 다양한 다운스트림 작업에 대한 어댑터 검색을 효과적으로 학습한다는 것을 보여줍니다.



### A Plug-and-Play Algorithm for 3D Video Super-Resolution of Single-Photon LiDAR data (https://arxiv.org/abs/2412.09427)
Comments:
          14 pages, 10 figures

- **What's New**: 본 논문은 단일광자 법선 다이오드(SPAD) 데이터를 통해 동적 장면의 3D 이미지를 개선하기 위한 새로운 계산된 이미징 알고리즘을 제안합니다. 이 알고리즘은 모션 블러를 처리하고 센서의 고유 해상도를 높이는 방법을 통합하여 보다 정확한 이미징 결과를 도출합니다. 대칭 비디오 슈퍼 해상도 및 정밀한 이미지 재정렬을 위한 최적화 기법을 활용하여 다양한 환경에서 실험을 진행하였습니다.

- **Technical Details**: 제안된 방법은 높은 속도의 SPAD 이벤트를 활용하고 저속도의 기존 카메라로부터 고해상도 깊이 이미지를 생성하는 과정을 포함합니다. 논문에서는 SPAD와 기존 카메라의 데이터를 융합하여 비 블러 고해상도 이미지를 만드는 것이 목표입니다. 이를 위해 PnP(Plug-and-Play) 방식을 활용하여 알고리즘의 각 단계를 최적화하며, 모션 추정 및 깊이 정보의 정렬이 이루어집니다.

- **Performance Highlights**: 실험 결과는 다양한 신호 대 잡음 비율(SNR) 및 광자 수준에서 이미지 해상도가 크게 향상됨을 보여줍니다. 또한, 이 방법은 실제 환경에서의 동적 물체를 대상으로 한 실험에서도 인상적인 성능을 발휘했습니다. 저해상도 SPAD 데이터와 실내외 다양한 조건에서의 성능 검증 결과, 모션 속도와 잡음 상태가 나쁘더라도 효율적인 3D 이미징이 가능함을 입증하였습니다.



### Learned Compression for Compressed Learning (https://arxiv.org/abs/2412.09405)
Comments:
          Accepted as paper to 2025 IEEE Data Compression Conference

- **What's New**: 이번 논문에서는 WaLLoC(Wavelet Learned Lossy Compression)이라는 새로운 신경 코덱 아키텍처를 소개하며, 이는 선형 변환 인코딩(linear transform coding)과 비선형 차원 축소 오토인코더(nonlinear dimensionality-reducing autoencoders)를 결합하여 압축된 도메인 학습(compressed-domain learning)을 지원합니다. WaLLoC는 고주파 세부 정보를 효과적으로 나타낼 수 있는 성능을 제공하며, RGB 이미지 및 스테레오 오디오를 포함한 다양한 매개체에 적합합니다. 기존의 압축 방식들은 효율성이나 정보를 효과적으로 전달하는 데 한계가 있었으나, WaLLoC는 이러한 문제를 해결합니다.

- **Technical Details**: WaLLoC는 신경망 구성 요소 사이에 비가역 웨이블릿 패킷 변환(invertible wavelet packet transform)을 삽입하여 신호의 중복성을 효과적으로 이용하며, 인코딩 단계에서는 거의 모든 선형 연산으로 구성되어 있습니다. 이 접근 방법은 인코딩 비용을 5% 미만으로 줄이며, 고차원 신호 패치를 저차원 잠재 표현으로 변환하여 차원 축소를 극대화합니다. 또한, 훈련 중에 엔트로피 병목(entropy bottleneck)을 적용하여 복잡한 양자화 과정을 통해 왜곡을 줄이는 방식으로 고도로 압축된 잠재 벡터를 생성합니다.

- **Performance Highlights**: WaLLoC는 이미지 분류(image classification), 색상화(colorization), 문서 이해(document understanding), 음악 소스 분리(music source separation)와 같은 다양한 과제에서 압축된 도메인 학습을 통해 현저한 성능 향상을 보여줍니다. 특히, WaLLoC는 안정적인 확산 모델(Stable Diffusion)에서 사용되는 변분 오토인코더(VAE)보다 약 6배 높은 압축 비율을 달성하며, 유사한 품질의 데이터를 유지합니다. 이로 인해 WaLLoC는 자원 제약이 있는 모바일 컴퓨팅 및 원격 감지 애플리케이션에도 적합한 효율적인 솔루션을 제공합니다.



### Multi-Stage Segmentation and Cascade Classification Methods for Improving Cardiac MRI Analysis (https://arxiv.org/abs/2412.09386)
Comments:
          Cardiac MRI, heart pathology, deep learning, segmentation, Gaussian smoothing, classification, cascade

- **What's New**: 이번 연구에서는 심장 자기공명영상(cardiac magnetic resonance imaging)의 분할(segmentation) 및 분류(classification)에서 새로운 심층 학습(deep learning) 기반 접근 방식을 제안합니다. 이 방법은 기존의 기법들이 겪고 있는 정확도 및 일반화(generalizability) 문제를 해결하는 데 중점을 두고 있습니다. U-Net 및 ResNet 모델을 활용한 다단계(process) 접근법이 특징입니다.

- **Technical Details**: 연구에서는 U-Net 및 ResNet 모델을 사용하여 심장 영상을 효과적으로 분할하고, 이를 Gaussian smoothing으로 후처리하여 정확도를 개선했습니다. 결과적으로 왼쪽 심실(left ventricle)에서 0.974, 오른쪽 심실(right ventricle)에서 0.947의 Dice 계수를 기록하며 높은 분할 정확도를 달성했습니다. 심장 질환(distinguish heart conditions) 분류는 심층 학습(classifiers)의 연속적인 계층(cascade)을 통해 이루어졌습니다.

- **Performance Highlights**: 이 방법은 비대성 심근병증(hypertrophic cardiomyopathy), 심근경색(myocardial infarction), 확장성 심근병증(dilated cardiomyopathy) 등을 포함한 질환들을 구분하며 평균 97.2%의 정확도를 기록했습니다. 제안된 접근 방식은 기존 모델들보다 뛰어난 성능을 보였으며, 임상 응용(clinical applications)에서의 가능성을 보여줍니다. 그러나 다양한 영상 프로토콜(image protocols)에 대한 추가 검증(validation)과 해석이 필요합니다.



### Quantitative Evaluation of Motif Sets in Time Series (https://arxiv.org/abs/2412.09346)
- **What's New**: 이 논문에서는 Time Series Motif Discovery (TSMD) 분야에서 정량적 평가를 위한 새로운 메트릭인 PROM을 소개합니다. 이 메트릭은 기존의 방법들이 가진 한계를 극복하고, 다양한 데이터 세트에 걸쳐 범용적으로 사용할 수 있도록 설계되었습니다. 또한, TSMD-Bench라는 새로운 벤치마크를 제안하여 현재의 메트릭과의 성능 비교를 더 효과적으로 수행할 수 있도록 합니다.

- **Technical Details**: PROM은 발견된 motif 세트를 실제 'ground truth'와 비교하여 평가하는 방식으로 작동합니다. 이는 각각의 motif가 얼마나 정확하게 발견되었는지와 발견된 motif 세트가 ground truth와 얼마나 잘 일치하는지를 고려합니다. 여러 가지 기존의 정량적 평가 방법들이 자주 사용하는 가정들이 이 예시에서는 위배됨을 보여줍니다.

- **Performance Highlights**: PROM과 TSMD-Bench를 통해 수행된 실험 결과는 PROM이 기존의 메트릭보다 포괄적인 평가를 제공함을 보여주고, TSMD-Bench가 이전 벤치마크보다 더 도전적이라는 것을 확인했습니다. 이 조합은 TSMD 방법 간의 상대적인 성능을 이해하는 데 도움을 주며, 대규모의 체계적인 성능 비교를 가능하게 합니다.



### Physics-Driven Autoregressive State Space Models for Medical Image Reconstruction (https://arxiv.org/abs/2412.09331)
Comments:
          10 pages, 4 figures

- **What's New**: 본 논문에서는 MambaRoll이라는 새로운 물리 기반 자회귀 상태 공간 모델을 소개합니다. 이 모델은 독창적인 피사계 모듈을 사용하여 여러 공간 스케일에서의 이미지 복원을 가능하게 하며, 의료 이미징에서의 왜곡 감소에 큰 기여를 합니다. MambaRoll은 기존의 물리 기반 방법들과는 달리, 각 스케일 간의 연속적인 예측을 통해 다중 스케일의 컨텍스트를 포착합니다.

- **Technical Details**: MambaRoll은 물리 기반 상태 공간 모듈(Physics-Driven State Space Modules, PSSM)을 활용하여 고해상도 피처 맵을 점진적으로 재구성합니다. 이 모듈은 컨텍스트를 효율적으로 집계하며, 수집된 데이터에 대한 일관성을 유지합니다. PSSM은 인코더, 셔플된 SSM 블록, 디코더와 잔여 데이터 일관성 블록으로 구성되어 있습니다.

- **Performance Highlights**: MambaRoll은 가속화된 MRI 및 희소 보기 CT 재구성에서 최첨단 물리 기반 방법들을 능가하는 성능을 보여주었습니다. 이 모델은 다중 스케일의 컨텍스트 기능을 포착하여 이미지 왜곡을 효과적으로 감소시킵니다. 연구 결과, MambaRoll은 기존의 컨볼루션, 트랜스포머 및 전통적인 SSM 모듈 기반의 방법들과 비교하여 우수한 성능을 발휘하는 것으로 나타났습니다.



### Computer-Aided Osteoporosis Diagnosis Using Transfer Learning with Enhanced Features from Stacked Deep Learning Modules (https://arxiv.org/abs/2412.09330)
- **What's New**: 이 연구에서는 무릎 골다공증(Knee Osteoporosis)을 진단하기 위한 컴퓨터 보조 진단(CAD) 시스템을 제안합니다. 이 시스템은 전이 학습(Transfer Learning)과 깊은 학습 블록(Deep Learning Blocks)의 스택(stacked)을 결합하여 초기 X-ray 이미지에서 뼈 조직의 복잡한 패턴을 효과적으로 학습합니다.

- **Technical Details**: 제안된 시스템의 첫 단계로, 무릎 X-ray 이미지를 전처리한 후, 사전 학습된 합성곱 신경망(Convolutional Neural Network, CNN)을 사용하여 특징(feature)을 추출합니다. 이 후, 다섯 개의 연속적인 Conv-RELU-MaxPooling 블록을 통해 이 특징들이 향상되며, Conv2D 층은 저수준 특징을 감지하고, ReLU 활성화 함수는 비선형성을 도입하여 복잡한 패턴을 학습할 수 있도록 합니다.

- **Performance Highlights**: 제안된 모델은 OKX Kaggle 이진(Binary), KXO-Mendeley 다중 클래스(Multi-Class), OKX Kaggle 다중 클래스 및 결합 데이터셋(combined dataset)에서 각각 97.32%, 98.24%, 97.27%, 98.00%의 정확도로 기존 방법보다 약 2% 향상된 성능을 보여줍니다. 이러한 결과는 무릎 골다공증의 조기 진단에 기여할 것으로 기대됩니다.



### Multimodal Sentiment Analysis based on Video and Audio Inputs (https://arxiv.org/abs/2412.09317)
Comments:
          Presented as a full paper in the 15th International Conference on Emerging Ubiquitous Systems and Pervasive Networks (EUSPN 2024) October 28-30, 2024, Leuven, Belgium

- **What's New**: 이번 연구는 비디오와 오디오 입력을 활용한 감정 인식 모델의 유용성을 입증하는 것을 목표로 하고 있습니다. 최신 연구들은 transformer 기반의 접근법을 통해 감정 분석의 정확도를 높여왔으나, 본 논문에서는 이러한 접근법들 간의 균형을 찾아 더 나은 결과를 도출하고자 하였습니다. 여러 감정 인식 데이터셋인 CREMA-D와 RAVDESS를 활용하여 모델을 훈련하며, 다양한 메소드와 평가 기법들을 적용하였습니다.

- **Technical Details**: 연구에서 사용된 주요 데이터셋은 CREMA-D와 RAVDESS입니다. CREMA-D는 7442개의 음성 클립과 91명의 배우로 구성되어 있으며, 각 클립은 분노, 혐오, 두려움, 행복, 슬픔, 중립 등의 감정을 포함하고 있습니다. RAVDESS 데이터셋은 7356개의 영상 파일로, 감정의 정확성과 강도를 평가받았습니다. 최종적으로 Facebook의 wav2vec2-large와 Google의 vivit-b-16x2-kinetics400 모델을 선택하여 오디오 및 비디오 분류 작업을 수행하였습니다.

- **Performance Highlights**: 연구의 초기 결과는 두 모델의 조합을 통한 감정 예측의 신뢰성을 높이는 방향으로 이루어졌습니다. 여러 테스트 프레임워크를 통해 최상의 정확도를 기록한 모델을 선택하였으며, 이는 향후 연구에 긍정적인 결과를 가져올 것으로 기대됩니다. 본 프로젝트의 제한된 접근 방식에도 불구하고 감정 분석의 효과적 성과를 이끌어 냈으며, 이는 후속 연구에서도 유효한 방향으로 평가될 것입니다.



### Accuracy Improvements for Convolutional and Differential Distance Function Approximations (https://arxiv.org/abs/2412.09200)
- **What's New**: 이번 논문에서는 내부 점에서 경계까지의 거리 함수를 추정하는 방법을 다룹니다. 특히, convolutional 및 differential distance estimation 스킴을 비교하고 각각의 정확도를 개선하기 위한 방법을 제안합니다. 개선된 방법은 Laplace 적분의 비대칭성(asymptotics)과 Taylor 급수(extrapolations)를 활용합니다.

- **Technical Details**: 연구에서는 거리 함수의 근사화를 위한 다양한 기법을 제안합니다. 특히, convolution 기반의 거리 변환(convolution distance transforms)과 비볼록 변분 문제(non-convex variational problems)에서의 Laplace 근사를 연결 지어 개선된 결과를 도출합니다. 또한, heat method와의 결합을 통해 거리 함수 근사화의 효율성을 높일 수 있음을 보여줍니다.

- **Performance Highlights**: 제안된 방법들은 기존의 거리 함수에 대한 추정 정확도를 개선하는 데 기여합니다. 특히, 다양한 경우에 대해 거리 함수를 효과적으로 근사할 수 있는 가능성을 제시합니다. 추가적으로, flat domains에 대한 Taylor 기반의 확장 기법은 실제 적용 가능성에서 중요한 기여를 할 것으로 보입니다.



### YingSound: Video-Guided Sound Effects Generation with Multi-modal Chain-of-Thought Controls (https://arxiv.org/abs/2412.09168)
Comments:
          16 pages, 4 figures

- **What's New**: 최근 몇 년간 오디오와 음악 생성 기술이 비약적으로 발전했습니다. YingSound는 적은 라벨 데이터로도 고품질 사운드를 생성할 수 있는 비디오 유도 사운드 생성 모델로, 고급 오디오 비주얼 통합과 함께 멀티모달 체인 오브 사고 (multi-modal chain-of-thought) 접근 방식을 사용하여 더욱 정교한 소리 효과를 생성합니다. 새로운 V2A 데이터셋이 다양한 현실적인 시나리오를 포함하여 제시되었습니다.

- **Technical Details**: YingSound는 두 가지 주요 모듈로 구성됩니다. 첫 번째 모듈은 조건부 플로우 매칭 변환기(conditional flow matching transformer)를 사용하여 오디오와 비주얼 간의 효과적인 의미적 정렬을 달성합니다. 두 번째 모듈은 멀티모달 비주얼-오디오 체인 오브 사고(Multi-modal Visual-Audio Chain-of-Thought) 접근 방식을 통해 고품질 오디오를 생성하는 기능을 제공합니다.

- **Performance Highlights**: 자동 평가와 인간 연구를 통해 YingSound는 다양한 조건 입력에 대해 고품질의 동기화된 사운드를 효과적으로 생성하는 것으로 나타났습니다. 이 모델은 영화, 게임 등 다양한 산업적 장면에서의 소리 효과 생성에 적합하다는 것을 보여줍니다. 기존의 수작업을 최소화하는 방법으로, 효율적인 오디오 생성이 가능하다는 장점을 가지고 있습니다.



### Vision CNNs trained to estimate spatial latents learned similar ventral-stream-aligned representations (https://arxiv.org/abs/2412.09115)
Comments:
          29 pages, 20 figures, ICLR 2025

- **What's New**: 이번 연구는 유인원의 ventral visual stream의 기능이 객체 인식(object recognition)뿐만 아니라 spatial latents를 추정하는 데도 최적화되어 있을 가능성을 탐구합니다. 전통적으로는 이 두 기능이 분리되어 있다고 여겨졌지만, 연구 결과에 따르면 공간적 정보도 중요한 역할을 하는 것으로 보입니다. 연구자는 3D 그래픽 엔진을 활용하여 다양한 공간적 변수(latent variables)를 포함하는 합성 이미지 데이터셋을 생성했습니다.

- **Technical Details**: 연구에서는 CNN(Convolutional Neural Networks)을 훈련시켜 객체의 위치(position)와 자세(pose) 같은 공간적 변수를 추정하는 것을 목적으로 하였습니다. 총 1억 장에 달하는 이미지를 포함하는 대규모 합성 데이터셋을 만들고, 다양한 spatial latent variables의 예측 성능을 평가하여 각 모델의 내부 표현(internal representations)과 primate ventral stream의 신경 응답(neural responses)과의 정렬(neural alignment)을 비교하였습니다. 특히, 모델들은 서로 다른 목표를 학습했음에도 불구하고 매우 유사한 표현을 학습했습니다.

- **Performance Highlights**: 모델이 spatial latents를 추정하기 위해 훈련되었을 때, 자연 이미지 데이터에 훈련된 모델들과 비슷한 신경 정렬 점수를 달성했습니다. 또한, 단 몇 개의 공간적 변수를 추정하는 모델들이 수백 개의 카테고리로 훈련된 모델들과 유사한 성능을 보여주었으며, 이들의 신경 정렬과 공간적 성능 간에는 강한 상관관계가 있었습니다. 연구 결과는 training objective가 유사한 모델을 만들 수 있다는 것을 제안하며, ventral stream의 기능이 단순한 객체 인식에 국한되지 않음을 시사합니다.



### A Wander Through the Multimodal Landscape: Efficient Transfer Learning via Low-rank Sequence Multimodal Adapter (https://arxiv.org/abs/2412.08979)
Comments:
          Accepted at AAAI 2025

- **What's New**: 이 논문에서는 loW-rank sequence multimodal adapter (Wander)를 제안하여 다중 모달 전이 학습(efficient transfer learning)에서의 두 가지 주요 문제를 해결합니다. 기존의 전이 학습 기술은 주로 비전-언어 모델에 국한되어 있으며, 두 개의 모달 이상에서는 제대로 확장되지 않습니다. Wander는 파라미터 효율적으로 서로 다른 모달의 시퀀스 간의 상호작용을 가능하게 하여 이러한 문제점을 해결합니다.

- **Technical Details**: Wander는 외적(outer product)을 활용하여 서로 다른 모달의 정보를 효과적으로 융합합니다. CP 분해(CP decomposition)를 사용하여 텐서를 저차원(rank-one) 성분으로 분해하며, 이로 인해 파라미터 수를 대폭 줄입니다. 이러한 설계를 통해 Wander는 모달 간의 보다 세분화된 특징(token-level features)과 시퀀스 관계(sequence relationships)를 추출합니다.

- **Performance Highlights**: Wander는 다양한 모달 수를 가진 데이터셋에서 광범위한 실험을 통해 기존의 최첨단 전이 학습 방법에 비해 일관되게 뛰어난 성능을 보였습니다. 특히, 파라미터 수가 적은 상태에서 효과성 및 효율성을 입증하였으며, 이는 Wander의 보편성을 강조합니다. 이 결과는 Wander가 다중 모달 모델 처리에 있어 실질적인 진전을 이룬 것을 의미합니다.



### Optimized Gradient Clipping for Noisy Label Learning (https://arxiv.org/abs/2412.08941)
Comments:
          Accepted by AAAI2025

- **What's New**: 이번 논문에서는 손실 함수의 그래디언트 조정 기법인 Optimized Gradient Clipping (OGC)을 제안합니다. 이는 노이즈 그래디언트와 클린 그래디언트의 비율에 기반하여 클리핑 임계값을 동적으로 조정합니다. 기존의 고정된 임계값을 사용하는 방법과는 달리, OGC는 학습의 각 단계에서 변화하는 그래디언트 분포에 효과적으로 대응하여 모델의 내구성을 발휘합니다.

- **Technical Details**: OGC는 클린 및 노이즈 샘플의 손실 분포를 모델링하기 위해 2-컴포넌트 가우시안 혼합 모델 (2-GMM)을 활용합니다. 클리핑 이후의 노이즈 그래디언트와 클린 그래디언트 비율을 추정하여 각 학습 단계에서 적절한 클리핑 임계값을 결정합니다. 이로써 그래디언트의 영향을 제대로 제한하여, 기존의 비 강건 손실 함수인 Cross Entropy (CE)를 다양한 유형의 레이블 노이즈에 대해 강건하게 만듭니다.

- **Performance Highlights**: 제안된 OGC 방법은 다양한 레이블 노이즈 (대칭, 비대칭, 인스턴스 종속, 실제 데이터 노이즈)에 대해 효과성을 입증하는 폭넓은 실험을 통해 성공적으로 평가되었습니다. 또한, 본 연구에서 제공하는 통계 분석은 OGC의 노이즈 내성 능력을 보증합니다. 연구 결과는 디지털 형식으로 제공되며, 출판 시 오픈 소스화 예정입니다.



### Deep Clustering using Dirichlet Process Gaussian Mixture and Alpha Jensen-Shannon Divergence Clustering Loss (https://arxiv.org/abs/2412.08940)
- **What's New**: 이 논문에서는 딥 클러스터링에서의 주요 문제를 해결하기 위해 새로운 접근법을 제안합니다. 기존의 Kullback-Leibler divergence(KLD)의 비대칭성을 개선하기 위해 Jensen-Shannon divergence(JSD)를 사용하고, Dirichlet process Gaussian mixture model을 도입하여 클러스터 수를 고정하지 않고도 클러스터링과 모델 선택을 동시에 수행합니다. 이러한 방식은 사전 지식 없이도 최적의 클러스터 수에 접근할 수 있습니다.

- **Technical Details**: 논문에서는 먼저 기본적인 Autoencoder 기반의 클러스터링 방법을 설명하고, Type II 정규화를 통해 이미지 클러스터링의 차원 축소와 클러스터링을 함께 최적화합니다. JSD는 이전 논문에서 제안된 αJSD를 통해 닫힌 형태(close form)의 솔루션을 제공하며, Dirichlet process mixture(DPM)를 사용하여 무한 클러스터 표현을 가능하게 합니다. 이 접근은 클러스터 수를 사전 정의하지 않고도 최적의 클러스터 수에 도달하도록 안내합니다.

- **Performance Highlights**: 저자들은 MIT67 및 CIFAR100과 같은 대규모 데이터셋에 대해 제안한 딥 모델 선택 방법을 전통적인 모델 선택 방법과 비교합니다. 실험 결과, 제안된 접근법이 기존의 KLD 기반 방법뿐만 아니라 전통적인 클러스터링 기법보다 우수한 성능을 보임을 확인하였습니다. 이로 인해, 딥 클러스터링의 응용 가능성이 더욱 확장될 것으로 기대됩니다.



### jina-clip-v2: Multilingual Multimodal Embeddings for Text and Images (https://arxiv.org/abs/2412.08802)
Comments:
          21 pages, 1-10 main paper, 10-12 refs, 12-21 benchmarks

- **What's New**: 이번 연구에서는 jina-clip-v1 모델을 기반으로 한 향상된 프레임워크인 jina-clip-v2를 제안합니다. 이 모델은 여러 언어에서 다중 작업 및 다중 단계의 대조 학습(multi-task, multi-stage contrastive learning)을 활용하여 텍스트 전용 검색 성능을 개선합니다. ML 모델의 한계로서 명시된 다국어 지원 부족 및 복잡한 비주얼 문서 이해에서의 성능 저하 문제를 해결합니다.

- **Technical Details**: jina-clip-v2 모델은 듀얼 인코더 아키텍처(dual encoder architecture)를 사용하여 텍스트와 이미지를 동일한 임베딩 공간에서 인코딩합니다. 텍스트 인코더는 사전 훈련된 Jina-XLM-RoBERTa 모델을 초기화하고, 이미지 인코더는 EVA02 계열의 ViT 모델을 선택하였습니다. 모델 학습을 위해 다양한 다국어 데이터셋을 구축하였고, 하드 네거티브(hard negative) 샘플을 포함한 훈련 방법을 채택했습니다.

- **Performance Highlights**: jina-clip-v2는 텍스트 전용 및 다중 모드 작업에서 이전 모델에 비해 성능이 크게 향상되었습니다. 이 모델은 다국어 텍스트 검색 벤치마크 및 시각적으로 풍부한 문서 검색 벤치마크에서 국가 최우수 성능에 필적하는 결과를 보여줍니다. 또한, Matryoshka Representation Learning을 활용하여 벡터 저장 비용을 절감하며, 성능 저하 없이 출력 벡터 차원을 축소 가능합니다.



### A Hybrid Framework for Statistical Feature Selection and Image-Based Noise-Defect Detection (https://arxiv.org/abs/2412.08800)
Comments:
          23 pages, 17 figures

- **What's New**: 이 논문은 복잡한 환경에서 노이즈를 포함한 산업 이미지에서 표면 결함을 정확하게 탐지하고 구분하기 위한 하이브리드 프레임워크를 제안합니다. 통계적 특성 선택(statistical feature selection)과 분류(classification) 기법을 통합하여 결함 탐지 정확도를 개선하고 허위 양성(false positive)을 최소화하는 것이 이 시스템의 핵심 동기입니다. 약 55개의 차별화된 특성이 추출되어 통계적 방법을 통해 분석되며, 이로 인해 진짜 결함과 노이즈 간의 분리 최대화를 목표로 합니다.

- **Technical Details**: 이 연구에서는 피셔 분리(Fisher separation), 카이제곱 검정(chi-squared test), 분산 분석(variance analysis)과 같은 통계적 방법을 활용하여 산업 이미지에서 가장 차별화된 특성을 추출합니다. 이러한 기술은 진짜 결함(TP)과 허위 결함(FP) 간의 분리를 극대화하고, 피셔 기준(Fisher's criterion)을 통해 자동화 시스템의 실시간 성능을 보장합니다. 이 프레임워크는 기존 분류기에 적용할 수 있는 블랙박스 모듈로 구현될 수 있어 유연한 머신 러닝 애플리케이션과 통합되며, 다양한 결함과 환경에서의 정확성과 일반성을 향상시킵니다.

- **Performance Highlights**: 제안된 프레임워크는 복잡한 노이즈 환경에서도 결함 탐지의 정확도를 개선시키고, 허위 양성과 잘못된 분류를 줄이는 데 크게 기여합니다. 본 논문의 접근 방식은 노이즈와 결함이 모두 동적 범위 내에서 매우 작은 부분을 차지하고 있어 본질적으로 대조가 낮다는 점에 주목하여 이를 극복합니다. 통계적 직관에 기반하여 통합된 방법론은 실제 환경에서의 신뢰성을 보장하며, 다양한 응용 분야에 유용하게 적용될 수 있습니다.



### Emotional Vietnamese Speech-Based Depression Diagnosis Using Dynamic Attention Mechanism (https://arxiv.org/abs/2412.08683)
Comments:
          9 Page, 5 Figures

- **What's New**: 본 연구에서는 음성 신호를 분석하여 감정을 분류하기 위한 Attention-GRU 네트워크에 동적 합성곱 블록 주의 모듈(Dynamic-CBAM)을 도입하였습니다. 이 모듈을 통해 우울증이 의심되는 환자를 진단하고 조기 치료 및 예방에 기여할 수 있는 가능성을 제시합니다. 음성의 불편함, 슬픔, 감정의 결여 등의 신호를 파악하고, 이를 바탕으로 우울증의 징후를 빠르게 감지할 수 있음을 강조하고 있습니다.

- **Technical Details**: Dynamic-CBAM은 Convolutional Block Attention Module(CBAM)을 수정하여 Omni-Dimensional Dynamic Convolution(ODConv)과 결합한 새로운 방법론입니다. CBAM은 중요한 특징에 집중하고 불필요한 특징을 억제하는 주의 메커니즘을 적용하여 CNN의 성능을 향상시키는 역할을 합니다. ODConv는 전통적인 합성곱과 달리 동적으로 커널 가중치를 생성하여, 더 복잡하고 다양한 패턴을 추출할 수 있게 해주는 특징적인 방법입니다.

- **Performance Highlights**: 실험 결과, 본 연구에서 제안한 모델은 VNEMOS 데이터셋에서 Unweighted Accuracy(UA) 0.87, Weighted Accuracy(WA) 0.86, F1 점수 0.87이라는 뛰어난 성능을 달성하였습니다. 이러한 결과는 음성 신호의 주의 깊은 분석이 우울증 진단에 효과적으로 활용될 수 있는 가능성을 보여줍니다. 향후 우울증 치료 접근 방식의 발전에 기여할 것으로 기대됩니다.



### Detecting Visual Triggers in Cannabis Imagery: A CLIP-Based Multi-Labeling Framework with Local-Global Aggregation (https://arxiv.org/abs/2412.08648)
Comments:
          This project was initiated in September 2023

- **What's New**: 본 연구는 캠퍼스 식용 대마 제품과 관련된 온라인 논의에서 시각적 요소와 텍스트적 요소의 상호작용을 조사하며, 사용자 참여에 미치는 영향을 분석합니다. CLIP 모델을 활용하여 42,743개의 이미지를 분석하였고, 색상과 밝기와 같은 이미지 속성이 사용자 상호작용에 미치는 영향을 평가했습니다. 또한, BART 모델을 사용하여 텍스트 분석을 진행하고, 구조적 주제 모델링에서 도출된 10개의 주제를 분류하여 사용자 참여와의 관계를 조사하였습니다.

- **Technical Details**: 연구 과정에서 Crowd Tangle API를 사용해 수집된 데이터는 2021년 3월 1일부터 8월 31일 사이의 Facebook 포스트입니다. 데이터 필터링을 위해 비관련 콘텐츠를 제외하기 위한 블랙리스트를 설정하여 식용 대마와 관련된 마케팅 포스트를 엄선했습니다. CLIP 모델은 각 이미지에서 시각적 요소를 자동으로 탐지하며, 이는 다중 레이블 분류 문제로 설정되어, 각 이미지에서 여러 요소가 동시에 포함될 수 있도록 구성되었습니다.

- **Performance Highlights**: 선형 회귀 분석 결과, 과일, 사탕, 제빵 관련 시각적 요소가 사용자 참여 점수와 긍정적인 상관관계를 보인 반면, 이미지의 색상 감도와 특정 텍스트 주제와는 부정적인 연관성이 발견되었습니다. 데이터 구성에서 약 75%의 이미지 텍스트가 Null로 나타났고, 거의 80%의 메시지가 100단어 이하로 요약됐다는 점이 주목할 만합니다. 연구 결과는 정책 입안자와 규제 기관이 경고 레이블 및 마케팅 전략을 설계하는 데 유용한 통찰력을 제공합니다.



New uploads on arXiv(cs.AI)

### The Parameters of Educability (https://arxiv.org/abs/2412.09480)
Comments:
          13 pages

- **What's New**: 이번 논문에서는 인간이 가진 독특한 인지 능력을 설명하기 위해 제안된 교육 가능성 모델(educability model)에 대해 다루고 있습니다. 이 모델은 기계가 인간의 능력을 모사할 수 있는 잠재력을 설명하는 것을 목표로 합니다. 교육 가능성은 지식을 습득하고 적용할 수 있는 능력으로 정의되며, 이는 인간과 기계 모두에게 적용될 수 있습니다.

- **Technical Details**: 교육 가능성 모델은 수치적 매개변수와 비수치적 매개변수가 모두 포함된, 더 복잡한 개념입니다. 이 모델은 기계의 설계와 교육에 필요한 다양한 매개변수를 반영하고, 교육 정책과 같이 정보를 신뢰할 출처에 대한 정책도 포함합니다. 이러한 매개변수는 인간 교육뿐만 아니라 기계의 인간 능력 모사 가능성을 이해하는 데 중요합니다.

- **Performance Highlights**: 논문에서는 유니버설 튜링 기계와 같이 정확하게 정의된 수학적 사양을 요구하는 확장 가능하고 강력한 계산 모델의 중요성을 강조합니다. 여기서 교육 가능성 모델은 특히 새로운 사례에 대한 정확한 일반화를 위한 능력을 다루며, PAC(Probably Approximately Correct) 학습 모델과 유사한 특성을 가집니다. 이러한 접근법은 기계의 현재 능력과 앞으로 기대할 수 있는 가능성을 이해하는 데 유용한 통찰력을 제공합니다.



### Imitate, Explore, and Self-Improve: A Reproduction Report on Slow-thinking Reasoning Systems (https://arxiv.org/abs/2412.09413)
Comments:
          Technical Report on Slow Thinking with LLMs: Part II

- **What's New**: 최근 슬로우-씽킹( slow-thinking) 추론 시스템인 o1이 복잡한 문제 해결에서 뛰어난 능력을 보여주고 있습니다. 이러한 시스템은 쿼리에 응답하기 전에 긴 사고 과정을 가지고 있어 더 철저하고 정확한 솔루션을 생성할 수 있습니다. 본 논문에서는 o1 유사 시스템을 구현하기 위한 '모방, 탐색 및 자기 개선' 프레임워크를 제안하고, 과거 연구를 기반으로 이러한 시스템을 재현하는 접근법을 설명합니다.

- **Technical Details**: 저자들은 o1 같은 추론 시스템을 개발하기 위해 세 가지 주요 훈련 단계를 제시합니다: 모방(imitate), 탐색(explore), 그리고 자기 개선(self-improve)입니다. 초기 단계에서는 장기적 사고(long-form thought) 데이터를 미세 조정하여 모델이 슬로우-씽킹 모드를 활성화하도록 하고, 이후 문제를 탐색하며 여러 개의 확률적 경로(rollouts)를 생성합니다. 이러한 방식을 통해 모델은 반복적으로 훈련 데이터를 개선하고, 이전의 제약을 극복하여 다양한 도메인에서 일반화할 수 있는 시스템을 목표로 합니다.

- **Performance Highlights**: 연구 결과, 제안된 접근 방식은 MATH-OAI, AIME, GPQA와 같은 여러 벤치마크에서 산업 수준의 시스템과 경쟁할 수 있는 성능을 달성했습니다. 실험적으로 3,900개의 예시를 사용하는 경우, 우리의 시스템은 일부 산업 시스템에 가까운 성능을 보여주었으며, 1,100개의 증류된 데이터만으로도 긍정적인 결과를 얻었습니다. 이를 통해 제안한 프레임워크가 실제 문제 해결에 효과적인지 검증되었습니다.



### Uncommon Belief in Rationality (https://arxiv.org/abs/2412.09407)
Comments:
          The 39th Annual AAAI Conference on Artificial Intelligence (AAAI-25)

- **What's New**: 이 논문은 대리인 간의 상호작용을 분석할 때 일반적으로 가정되는 합리성의 동기적 (common knowledge) 전제를 넘어, 대리인들이 가질 수 있는 더 복잡한 고차 신념 구조를 포착하기 위한 그래프 기반 언어를 제안합니다. 기존의 이론에서는 대리인의 합리성이 공통 지식으로 가정되었으나, 본 연구는 이 가정이 꼭 진리인 것은 아니라는 점을 강조합니다. 또한, 주어진 신념 구조를 기반으로 한 추론 과정을 포착하는 해법 개념과 고유 최소 형태로의 신념 구조 압축을 위한 효율적인 알고리즘을 제시합니다.

- **Technical Details**: 이 연구에서는 대리인의 합리성 및 타인에 대한 신념을 전이하여 인식하는 신념 계층 (belief hierarchy)의 개념을 도입합니다. 이 신념 계층은 유향 그래프(directed graph)로 나타낼 수 있으며, 신념 계층의 각각의 노드는 대리인의 합리성과 신념의 상관관계를 시각적으로 나타냅니다. 연구에서 제안된 'RBR 그래프'는 대리인의 전략적 행동을 분석하기 위한 기초가 되며, 이는 대리인의 주관적 확률 분포 (subjective probability distributions) 대신 동적 경과를 포착하는 데 주안점을 둡니다.

- **Performance Highlights**: 이 모델은 수많은 대리인 간의 상호작용을 분석할 때 더 높은 신뢰성과 정확성을 부여합니다. 이를 통해 에이전트가 자신의 합리성을 인식하는 방식과 타인의 합리성을 추론하는 과정을 동시에 살펴볼 수 있습니다. 따라서, 복잡한 신념 구조를 쉽게 이해할 수 있는 방법을 제공하며, 게임 이론 및 다중 대리인 시스템의 연구에 기여할 수 있습니다.



### AI Predicts AGI: Leveraging AGI Forecasting and Peer Review to Explore LLMs' Complex Reasoning Capabilities (https://arxiv.org/abs/2412.09385)
Comments:
          47 pages, 8 figures, 17 tables, appendix with data and code

- **What's New**: 이 논문은 16개의 최첨단 대규모 언어 모델(LLMs)을 이용하여 2030년까지 인공지능 일반 지능(AGI) 출현 가능성을 추정하는 작업을 진행했습니다. 모델들의 추정치는 3%에서 47.6%까지 다양했으며, 그 중간값은 12.5%로 나타났습니다. 이러한 결과는 최근 전문가 설문조사에서 제시된 2027년까지 AGI 출현 가능성이 10%라는 예측과 밀접하게 관련되어 있습니다.

- **Technical Details**: 자동화된 동료 검토 과정인 LLM-PR을 통해 LLM의 예측 품질을 평가했습니다. 각 모델 간의 점수 일관성을 나타내는 높은 Intraclass Correlation Coefficient (ICC = 0.79)는 LLM-PR 프로세스의 신뢰성을 입증합니다. Pplx-70b-online 모델이 가장 우수한 성능을 보였고, Gemini-1.5-pro-api가 최하위로 평가되었습니다.

- **Performance Highlights**: LLM의 순위는 외부 벤치마크(LMSYS Chatbot Arena)와 비교했을 때 일관성을 보였으며, 이는 현재의 벤치마크가 AGI 예측에 필요한 일부 기술을 포괄하지 않을 가능성을 시사합니다. 외부 벤치마크를 기반으로 한 가중치 스킴의 활용이 LLM 예측과 인간 전문가 예측 간의 정렬을 최적화하는 데 기여했습니다. 이 분석을 통해 AGI 관련 작업의 성능 차이를 강조하는 새로운 'AGI 벤치마크'가 개발되었습니다.



### Beware of Metacognitive Laziness: Effects of Generative Artificial Intelligence on Learning Motivation, Processes, and Performanc (https://arxiv.org/abs/2412.09315)
- **What's New**: 최근 기술과 교육 혁신의 발전에 따라, 학습자들은 교사, 동료, 교육 기술 및 생성적 인공지능인 ChatGPT와 같은 다양한 에이전트의 지원을 받을 수 있게 되었습니다. 하이브리드 지능(hybrid intelligence)의 개념은 여전히 초기 단계에 있으며, AI 및 인간 전문가와의 공생 관계를 통해 학습자가 어떻게 혜택을 받을 수 있는지는 아직 불확실합니다. 본 연구는 이러한 격차를 해소하기 위해 무작위 실험 연구를 수행하였습니다.

- **Technical Details**: 117명의 대학생을 대상으로 ChatGPT, 인간 전문가, 작문 분석 도구 및 추가 도구 없이 지원을 받는 여러 그룹을 비교하여 학습 동기, 자기 조절 학습 과정 및 작문 과업 수행 결과를 분석하였습니다. 연구 결과, 학습자는 동기에 차이가 없었으나, 자기 조절 학습 과정의 빈도 및 순서에서는 유의미한 차이를 보였습니다. 특히, ChatGPT를 활용한 그룹이 에세이 점수 향상에서 우수한 성과를 보였지만, 지식 습득 및 전이에 있어서는 유의미한 차이가 없었습니다.

- **Performance Highlights**: AI 기술인 ChatGPT는 학습자의 기술 의존성을 증가시키고 메타 인지적 태만(metacognitive laziness)을 유발할 가능성이 있다는 점이 주목할 만합니다. 본 연구는 다양한 지원을 받는 학습자들이 동기에서는 차이가 없으나, 자기 조절 학습 과정이 다름으로써 각기 다른 성과를 보였음을 발견하였습니다. 이러한 결과는 하이브리드 인공지능 학습의 메커니즘과 결과에 대한 깊이 있는 이해를 필요로 함을 보여줍니다.



### Speeding up approximate MAP by applying domain knowledge about relevant variables (https://arxiv.org/abs/2412.09264)
Comments:
          16 pages, 7 figures

- **What's New**: 이 논문에서는 베이지안 네트워크에서 MAP 문제를 해결하기 위한 'Most Frugal Explanation' 헤uristic 접근법을 다시 탐구하고 있습니다. 이 접근법은 중간 변수 세트를 관련 변수와 비관련 변수로 분리하여 해결책을 찾기 위해 시도되었습니다. 또한, 특정 쿼리에 대해 어떤 변수가 관련 있는지에 대한 지식이 계산을 얼마나 가속화하는지 탐험하였습니다.

- **Technical Details**: MAP 문제는 일반적으로 다루기 어려운(Intractable) 문제입니다. 본 연구에서 저자들은 관련 변수(relevant variables)와 비관련 변수(irrelavant variables)로 나누어 접근하는 방법을 사용합니다. 이 과정에서, 비관련 변수는 그들의 도메인에서 샘플 값을 할당받고, 관련 변수는 마진화(marginalization)됩니다.

- **Performance Highlights**: 연구 결과는 결론이 뚜렷하지 않지만, 특이적으로 MAP 쿼리의 세부사항, 특히 MAP 변수의 수에 따라 결과가 달라질 수 있음을 보여주었습니다. 이는 정확한 MAP 및 근사 MAP보다 더 빠른 계산을 가능하게 할 수 있는지에 관한 의문을 제기하고 있습니다.



### LMAgent: A Large-scale Multimodal Agents Society for Multi-user Simulation (https://arxiv.org/abs/2412.09237)
- **What's New**: 이번 논문에서는 멀티모달 대화형 AI 에이전트를 기반으로 한 아주 크고 다양한 에이전트 사회인 LMAgent를 소개합니다. LMAgent는 사용자가 자유롭게 대화하고 제품을 탐색, 구매 및 리뷰를 하고, 심지어 라이브 스트리밍 전자상거래도 수행할 수 있도록 설계되어 있습니다. 이 시스템은 기존의 다중 에이전트 시스템보다 더 향상된 의사 결정 성능을 제공하며, 에이전트의 멀티모달 능력을 증대시키기 위해 자기 일관성 프롬프트 메커니즘을 도입했습니다.

- **Technical Details**: LMAgent는 멀티모달 LLM을 활용하여, 사용자 행동을 보다 정확하게 시뮬레이션할 수 있는 멀티모달 에이전트의 분석 능력을 강화합니다. 이 논문에서 제안하는 자기 일관성 프롬프트 메커니즘은 복잡한 멀티모달 시나리오에서 에이전트의 의사 결정을 개선하여 시뮬레이션 성능을 더욱 향상시키고, 시스템의 효율성을 높이기 위해 빠른 메모리 메커니즘을 도입하여 40% 정도 시스템 부하를 줄입니다.

- **Performance Highlights**: LMAgent의 실험 결과는 이 시스템이 행동 지표에서 인간과 유사한 성능을 보이면서 신뢰할 수 있는 대규모 사회 행동 시뮬레이션에 대한 잠재력을 보여줍니다. 특히, 소비자 행동 시뮬레이션에서는 실제 사용자 데이터와 유사한 공동 구매 패턴을 생성하며, 사회적 요인이 에이전트 행동에 미치는 영향도 조사하였습니다. 10,000명 이상의 에이전트를 동시에 시뮬레이션할 수 있는 기능은 LMAgent의 신뢰성을 높이는 또 다른 요소로 작용합니다.



### Goal-Driven Query Answering over First- and Second-Order Dependencies with Equality (https://arxiv.org/abs/2412.09125)
Comments:
          47 pages

- **What's New**: 이 논문에서는 처음으로 제시된 목표 주도형(query answering) 쿼리 응답 기술을 다룹니다. 이 기술은 일차 및 이차 종속성(first-order and second-order dependencies)과 동등성(reasoning with equality)을 모두 다루며, 기존의 비효율적인 모델 생성 방식을 보완합니다. 목표에 맞춘 접근 방식은 필요 없는 추론(inference) 생성을 피함으로써 성능을 향상시킬 수 있는 가능성을 보여줍니다.

- **Technical Details**: 논문에서 제안하는 기술은 입력 종속성을 변형하여 쿼리에 불필요한 추론이 발생하지 않도록 설계되었습니다. 이 과정은 세 가지 주요 기술로 나뉘며, 첫째로 Marnette의 단수화(singularisation) 기법의 변형을 소개합니다. 둘째, 쿼리 응답에 기여하지 않는 종속성을 제거하는 관련성 분석(relevance analysis) 기법이 포함됩니다. 셋째, 동등성을 고려한 이차 종속성 처리에 적합한 매직 세트(magic sets) 알고리즘의 변형이 소개됩니다.

- **Performance Highlights**: 광범위한 실험 평가 결과, 목표 주도형 쿼리 응답이 전체 유니버설 모델 컴퓨팅보다 몇 배 빠를 수 있음을 보여주었습니다. 이러한 성능 향상은 특히 고정 쿼리에 대해서 많은 추론을 피할 수 있다는 점에서 매우 의미가 있습니다. 논문에서 제안하는 기법은 실제 응용에서 매우 유용하게 사용될 수 있으며, 이는 다양한 도메인에서 효율성을 높일 것입니다.



### In-Dataset Trajectory Return Regularization for Offline Preference-based Reinforcement Learning (https://arxiv.org/abs/2412.09104)
Comments:
          7 pages, Proceedings of the 39th AAAI Conference on Artificial Intelligence (AAAI-25)

- **What's New**: 오프라인 선호 기반 강화 학습(PbRL)을 위한 새로운 방법인 In-Dataset Trajectory Return Regularization(DTR)를 제안합니다. DTR은 보상 편향을 감소시키기 위해 조건부 시퀀스 모델링(conditional sequence modeling, CSM)과 TD-Learning(TD 학습)을 통합합니다. 이를 통해 트랜지스터를 활용한 보상 모델의 견고성을 높이고, 행동 정책의 충실성을 유지하면서 최적의 행동을 선택하는 균형을 이룹니다.

- **Technical Details**: DTR의 핵심 요소로는 (1) Decision Transformer(DT) 구조를 활용하여 각 개별 궤적과 RTG(return-to-go) 토큰을 연계하는 정책, (2) 높은 단계별 보상을 가진 최적 행동을 선택하는 TDL 모듈, (3) 보상 차별화와 정확성 간의 균형을 위한 앙상블 표준화 기법이 포함됩니다. 이러한 요소들이 결합되어 오프라인 PbRL에서의 보상 편향의 위험을 경감하며 성능 향상을 이루어냅니다.

- **Performance Highlights**: 다양한 벤치마크에서의 실험은 DTR이 최신 기술 기준인 타 방법들보다 우수한 성능을 발휘함을 입증합니다. DTR은 오프라인 PbRL에서 보상 모델의 영향으로 인한 TD 학습 단계의 문제를 해결함으로써 정책 학습의 효율성을 높이고 있습니다. 이 연구는 대규모 언어 모델과 로봇 제어와 같은 분야에서의 개선 가능성을 제시하며, DTR의 기여가 오프라인 PbRL의 발전에 중대한 기여를 할 것임을 보여줍니다.



### Temporal Numeric Planning with Patterns (https://arxiv.org/abs/2412.09101)
Comments:
          Accepted at the 39th Annual AAAI Conference on Artificial Intelligence (AAAI-25)

- **What's New**: 이번 논문에서는 PDDL2.1 레벨 3로 표현된 시간적 수치 계획 문제(temporal numeric planning problems) $
Pi$에 대해 논의하고, SMT(Satisfiability Modulo Theory) 공식을 생성하는 방법을 제시합니다. 이 공식은 유효한 계획(valid plans)과 최근 제안된 패턴 기법을 시간적 경우로 확장합니다. 연구진은 이 접근 방법의 정확성(correctness)과 완전성(completeness)을 증명하고, 10개의 도메인에서 우수한 성능을 보였음을 나타냅니다.

- **Technical Details**: 논문은 PDDL2.1에 기반하여 시간적 수치 계획 문제를 정의하고, 유한 집합의 불리언(Boolean) 및 수치(numeric) 변수, 초기 상태(initial state), 목표(goals)를 포함하는 이 문제를 설명합니다. 각 행동은 사전(preconditions) 및 결과(effects)가 명시된 연산으로 구성되며, 여러 행동이 동시에 실행될 수 있습니다. 이로 인해 유효한 계획을 찾는 것이 훨씬 어려워지며, 일반적인 경우에는 결정 불가능(undecidable)할 수 있습니다.

- **Performance Highlights**: 제안된 기법은 10개의 시간적 도메인에서 공개된 모든 시간적 계획 도구와 비교하여 최고 성능을 보였습니다. 연구 결과에 따르면, 우리의 계획 도구는 10개의 도메인 중 9개에서 가장 높은 문제 해결 문제 수를 기록했으며, 다른 기계들에 비해 모든 문제에서 유효한 계획을 찾는 데 필요한 경계(bound)가 낮았습니다. 따라서, 이 접근 방식은 시간적 수치 계획 문제에 대한 가장 발전된 방법으로 평가됩니다.



### A Context-Enhanced Framework for Sequential Graph Reasoning (https://arxiv.org/abs/2412.09056)
Comments:
          Appeared at IJCAI 2024

- **What's New**: 이 논문은 그래프 구조 데이터에 대한 순차적 추론을 연구하고 있으며, 이는 자동화된 수학 문제 해결 및 신경망 그래프 알고리즘 학습과 같은 여러 인기 있는 분야에서 중요한 작업으로 자리 잡고 있습니다. 기존 아키텍처를 일반화하고, 각 단계의 추론이 이전 단계의 결과뿐만 아니라 더 많은 역사적 결과의 정보를 집합하여 활용하는 맥락 강화 프레임워크(context-enhanced framework)를 제안합니다.

- **Technical Details**: 제안된 프레임워크는 순차적 그래프 추론의 각 단계가 전통적인 seq-to-seq 작업보다 서로 강한 내부 연결을 가진다는 점을 관찰한 데서 비롯되었습니다. 이 덕분에 각 단계의 출력을 종합적이고 효율적으로 처리할 수 있는 방법을 찾게 되었고, 기존 방법들과 통합되어 그들의 추론 능력을 향상시킬 수 있습니다. 에mpirical 평가를 통해 CLRS Reasoning Benchmark에서 이 프레임워크가 기존 아키텍처의 성능을 획기적으로 개선할 수 있음을 보여주었습니다.

- **Performance Highlights**: 제안된 프레임워크는 CLRS Reasoning Benchmark의 다양한 데이터세트에서 대부분의 경우 최첨단 성과를 달성하며, 기존 아키텍처에 비해 성능을 상당히 향상시켰습니다. 이러한 결과는 순차적 추론을 위한 새로운 접근 방식이 성공적으로 적용될 수 있음을 시사합니다. 논문에 제안된 방법은 각 단계의 연관성을 최대한 활용하여 더 정확한 추론을 가능하게 하는 점에서 중요한 의미가 있습니다.



### Neural Interactive Proofs (https://arxiv.org/abs/2412.08897)
Comments:
          42 pages, 17 figures

- **What's New**: 이 논문에서는 신뢰할 수 있지만 계산적으로 제한된 에이전트(검증자)가 강력하지만 불신할 수 있는 에이전트(증명자)와 상호작용하여 주어진 작업을 해결하는 방법을 다룹니다. 특히, 에이전트를 신경망으로 표현하고 이 문제의 해결책을 '신경 상호작용 증명(neural interactive proofs)'이라고 지칭합니다. 이 작업의 일환으로, 기존 프로토콜을 일반화한 통합 프레임워크를 도입하고 여러 새로운 프로토콜을 제안합니다.

- **Technical Details**: 신경 상호작용 증명은 에이전트 간의 상호작용을 통해 문제를 솔루션할 수 있도록 설계됩니다. 검증자와 증명자는 각각 신뢰성과 강력함을 갖춘 신경망으로 모델링됩니다. 이를 통해 다양한 문제를 해결할 수 있는 가능성이 열리며, 제안된 프로토콜은 제로 지식 증명(zero-knowledge proofs)도 지원합니다.

- **Performance Highlights**: 이론 및 실험을 통해, 논문은 두 개의 도메인에서 신경 상호작용 증명의 응용성을 보여줍니다. 하나는 장난감 그래프 동형성 문제(toy graph isomorphism problem)로, 주요 아이디어를 설명하며, 다른 하나는 코드 검증 작업입니다. 이러한 연구는 더 안전한 AI 시스템 구축을 위한 신경 상호작용 증명에 대한 기초를 마련하는 데 중점을 둡니다.



### Structural Entropy Guided Probabilistic Coding (https://arxiv.org/abs/2412.08841)
- **What's New**: 이 논문에서는 기존의 정보 병목 (Information Bottleneck) 원칙을 기반으로 한 확률적 임베딩 (Probabilistic Embedding) 방법론에 구조적 엔트로피 정규화 손실을 접목한 새로운 모델인 SEPC를 제안합니다. 기존의 방법들이 잠재 변수의 개별적 제약만을 고려했던 것에 반해, SEPC는 잠재 변수 간의 구조적 정보를 포함합니다. 또한, 회귀 작업에서의 효과적인 구조적 엔트로피 활용 방안을 제시하여, 특정 문제를 해결하고자 했습니다.

- **Technical Details**: SEPC 모델은 잠재 변수 간의 관계를 최적화 문제에 통합하기 위해 구조적 엔트로피 정규화 손실을 도입합니다. 이를 위해 먼저 임베딩 간의 유사성 기반의 인접 행렬을 구성하고, 유도된 그래프의 구조적 엔트로피를 극대화합니다. 또한 회귀 작업에 맞춘 확률적 인코딩 트리를 설계하여, 소프트 레이블로 전환한 후 이를 구조적 엔트로피 손실에 적용합니다.

- **Performance Highlights**: 12개의 자연어 처리 작업에 대한 실험 결과, SEPC는 다른 최신 모델들보다 효과성, 일반화 능력, 그리고 레이블 노이즈에 대한 강인함에서 우수한 성능을 보였습니다. SEPC는 처음으로 구조적 엔트로피를 정규화 손실로 활용하는 방법을 제안하였고, 이 방식이 분류 및 회귀 작업에서 SOTA 성능을 기록하게 하였습니다. 이러한 결과는 SEPC 모델이 다양한 상황에서도 뛰어난 성과를 낸다는 것을 입증합니다.



### Autoformalizing and Simulating Game-Theoretic Scenarios using LLM-augmented Agents (https://arxiv.org/abs/2412.08805)
Comments:
          code: this https URL

- **What's New**: 이번 논문은 게임 이론적 시뮬레이션을 자동화하는 새로운 프레임워크를 제안합니다. 이 프레임워크는 대형 언어 모델(LLM)로 강화된 에이전트를 활용하여 자연어 시나리오 설명을 실행 가능한 논리 프로그램으로 변환합니다. 이를 통해 게임의 규칙을 정의하고, 생성된 프로그램의 구문적 정확성을 검증합니다.

- **Technical Details**: 저자들은 이전 연구를 기반으로 2x2 동시 이동 게임을 모델링하는 자연어 시나리오의 자동 형식화(autoformalization) 모듈을 통합하여 에이전트 모델을 개선했습니다. 이 모델은 생성된 게임 코드의 기능과 의미적 정확성을 검증하는 방식으로 사용됩니다. 이를 통해 다양한 전략을 탐색하고 비교할 수 있는 능력을 제공합니다.

- **Performance Highlights**: 55개의 자연어 설명에 대한 실험 결과, 생성된 게임 규칙의 구문적 정확도가 96%, 의미적 정확도가 87%임을 보여줍니다. 또한, LLK 강화 에이전트가 게임 플레이를 위해 전략을 자동 형식화하는 능력을 평가하고 다양한 게임 이론적 맥락에서의 전략 효율성을 평가하는 데 기여합니다.



### Doe-1: Closed-Loop Autonomous Driving with Large World Mod (https://arxiv.org/abs/2412.09627)
Comments:
          Code is available at: this https URL

- **What's New**: 새로 제안된 Doe-1은 자율주행을 위한 폐쇄 루프(closed-loop) 프레임워크를 제공하며, 기존의 방법들이 처한 여러 문제점들을 해결하고자 한다. 이 모델은 perceptio (인식), prediction (예측), planning (계획)을 통합하여 다중 모달(multi-modal) 토큰을 사용하여 다양한 작업을 수행한다. 특히, 자유형 텍스트(free-form texts)와 RGB 이미지 토큰을 활용하여 보다 효율적이고 정확한 데이터를 수집할 수 있는 가능성을 가지고 있다.

- **Technical Details**: Doe-1은 관측(observation), 설명(description), 행동(action) 토큰으로 장면을 표현하며, 이러한 토큰 간의 전이를 통해 일반적인 인식, 계획 및 예측 작업을 모델링한다. 이미지는 벡터 양자화된 변분 오토인코더(vector-quantized variational autoencoder)를 통해 토큰화되며, 행동은 새의 눈(view)에서의 이동(displacement)으로 표현된다. 이 통합된 접근 방식은 자율주행을 위해 비전 중심(vision-centric)으로 최적화되어 있으며, 각 모달의 작성을 효율적으로 실현한다.

- **Performance Highlights**: Doe-1은 nuScenes 데이터셋에서 다양한 작업을 수행하며 그 효과를 입증하였다. 실험을 통해, 시각적 질문 응답(visual question-answering), 행동 조건 영상 생성(action-conditioned video generation), 그리고 모션 계획(motion planning) 등의 작업에서 우수한 성능을 보여준다. 특히, fine-tuning 없이 여러 작업을 동시에 수행할 수 있는 가능성을 가지고 있어, 자율주행 분야에서 강력한 도구로 자리 잡을 것으로 기대된다.



### Olympus: A Universal Task Router for Computer Vision Tasks (https://arxiv.org/abs/2412.09612)
Comments:
          Technical Report

- **What's New**: Olympus는 Multimodal Large Language Models (MLLMs)를 활용하여 다양한 컴퓨터 비전 작업을 수행하도록 통합된 프레임워크를 제안합니다. 이는 이미지를 포함한 20가지 전문 작업을 처리하는 데 사용되는 특정 모듈에 대한 작업을 위임하는 방식으로 구성되어 있습니다. 라이팅(is instruction-based routing) 시스템을 통해 복잡한 워크플로우를 구현할 수 있으며, 기존의 MLLMs와 쉽게 통합되어 성능을 향상시킵니다.

- **Technical Details**: Olympus는 446.3K개 고품질 트레이닝 데이터와 49.6K개 평가 데이터를 포함하는 OlympusInstruct 및 OlympusBench라는 데이터 세트를 통해 설계되었습니다. 각 전문 작업을 위한 특정 라우팅 토큰을 만들어 사용자 명령 내에서 여러 작업을 체인화하여 수행할 수 있는 능력을 가지고 있습니다. 이 시스템은 MLLM으로서의 기능을 활용하여 외부 모듈로 작업을 위임하는 방식으로 작동합니다.

- **Performance Highlights**: 실험 결과, Olympus는 20개 개별 작업에서 평균 94.75%의 라우팅 정확성을 달성했으며, 체인 액션 시나리오에서는 91.82%의 정밀도를 기록했습니다. 이는 Olympus가 다양한 컴퓨터 비전 작업을 효과적으로 처리할 수 있는 보편적 작업 라우터로 기능할 가능성을 강조합니다. 또한 Olympus는 단일 명령 내에서 최대 5개의 작업을 해결할 수 있는 능력을 보여줍니다.



### Hidden Biases of End-to-End Driving Datasets (https://arxiv.org/abs/2412.09602)
Comments:
          Technical report for the CVPR 2024 Workshop on Foundation Models for Autonomous Systems. Runner-up of the track 'CARLA Autonomous Driving Challenge' in the 2024 Autonomous Grand Challenge (this https URL)

- **What's New**: 본 연구는 CARLA Leaderboard 2.0에서 엔드 투 엔드(End-to-End) 자율주행을 위한 최초의 시도를 제시합니다. 기존의 아키텍처 대신 훈련 데이터셋(training dataset) 분석에 중점을 두어, 전문가의 운전 스타일이 정책 성능에 미치는 영향과 간단한 기준으로 프레임을 가중치 부여하는 것의 문제점을 밝혀냈습니다. 이 연구의 결과로 제안된 새로운 데이터 필터링 방법은 데이터셋의 크기를 대폭 줄일 수 있으며, 모델은 2024 CARLA Challenge에서 두 개의 트랙에서 각각 1위 및 2위로 평가받았습니다.

- **Technical Details**: 연구에서는 PDM-Lite라는 오픈소스로 제공되는 플래너를 활용하여 CARLA Leaderboard 2.0 시나리오에서 데이터를 수집했습니다. 수집된 데이터는 RGB 이미지, LiDAR 포인트 클라우드, 경로 체크포인트 및 주변 물체에 대한 경계 상자 예측을 포함합니다. 이와 더불어 TransFuser++라는 기존의 Imitation Learning 모델을 약간 수정하여 훈련에 사용하였습니다.

- **Performance Highlights**: 제안된 모델은 다양한 도시 주행 시나리오를 안전하게 처리하며, 2024 CARLA Challenge에서 두 번째로 높은 순위를 기록했습니다. 또한 Bench2Drive 테스트 경로에서 1위를 차지하며, 기존의 평가 메트릭의 설계 결함을 밝혀냈습니다. 이러한 성과는 훈련 데이터셋의 중요성을 강조하며, 향후 도전과제를 위한 메트릭 개선 방안을 제안합니다.



### TimeRefine: Temporal Grounding with Time Refining Video LLM (https://arxiv.org/abs/2412.09601)
- **What's New**: 이번 연구에서는 비디오 템포럴 그라운딩(Video Temporal Grounding, VTG)의 문제를 해결하기 위해 TimeRefine이라는 새로운 접근 방식을 제안합니다. 기존의 비디오 LLM(Large Language Models)은 타임스탬프를 직접 예측하는 방법에 의존했으나, TimeRefine는 초기 예측 후 오프셋(offset)을 다시 예측하는 프로세스를 통해 정확성을 향상시킵니다. 이 방법은 모델의 셀프-수정(self-correct) 기능을 강조하며, 모델이 그라운드 트루스(ground truth)에서 얼마나 멀리 떨어져 있는지에 따라 페널티를 부여하는 보조 예측 헤드(auxiliary prediction head)를 추가하여 모델의 시간 인식 능력을 강화합니다.

- **Technical Details**: TimeRefine의 핵심 원리는 타임스탬프 예측을 직접 수행하는 대신 점진적인 정제 작업으로 재구성하는 것입니다. 모델은 먼저 대략적인 타임스탬프를 예측하고, 이후 이를 기반으로 오프셋을 예측하여 최종 예측에 도달합니다. 이 과정은 여러 차례 반복되며, 모델은 이전 예측에서 발생한 오류를 스스로 수정합니다. 또한, L1 손실(L1 loss)을 사용하여 예측이 그라운드 트루스에서 얼마나 멀리 떨어져 있는지에 따라 더 많은 페널티를 부여하여 모델의 학습을 유도합니다.

- **Performance Highlights**: 실험 결과 TimeRefine를 적용한 VTimeLLM은 ActivityNet 캡션과 Charades-STA 데이터세트에서 각각 3.6% 및 5.0%의 mIoU(mean Intersection over Union) 개선을 보여주었습니다. TimeRefine는 기존의 LLM 기반 VTG 방법과 쉽게 통합할 수 있어, 다양한 모델에서 성능 향상에 기여할 수 있습니다. 이러한 결과는 비디오 LLM의 시간 인식 능력을 획기적으로 향상시킬 잠재력을 제시합니다.



### Owl-1: Omni World Model for Consistent Long Video Generation (https://arxiv.org/abs/2412.09600)
Comments:
          Code is available at: this https URL

- **What's New**: 본 논문에서는 일관성 있는 장기 비디오 생성을 위한 Omni World modeL인 Owl-1을 제안합니다. 전통적인 비디오 생성 모델들이 단기 정보를 바탕으로 비디오를 생성하는 데 한계를 보이는 반면, Owl-1은 잠재 상태 변수를 이용하여 장기적이고 포괄적인 조건을 형성하여 더 일관된 비디오를 생성합니다. 이 모델은 비디오가 진화하는 세계의 관찰로 간주하고, 세계의 동적인 변화를 반영하는 방법으로 설계되었습니다.

- **Technical Details**: Owl-1은 잠재 상태 변수를 사용하여 현재와 과거의 세계 정보를 인코딩함으로써, 비디오 클립으로 디코딩됩니다. 이 모델은 세계의 미래 역학을 예측하여 상태 변수를 업데이트하고, 이를 통해 긴 비디오의 일관성을 높이는 동시에 내용의 다양성도 확보합니다. 특히, 대규모 다중모달 모델(LMM)과 비디오 확산 모델을 사용하여 강력한 생성 성능을 달성합니다.

- **Performance Highlights**: Owl-1은 VBench-I2V 및 VBench-Long 벤치마크 테스트에서 최신 기술(SOTA) 방법들과 비견되는 성능을 보여 주었습니다. 이를 통해 고품질 비디오 관찰을 생성할 수 있는 능력을 입증한 것으로 평가받고 있습니다. Owl-1의 접근 방식은 비디오 생성 모델의 새로운 가능성을 여는 데 기여할 것입니다.



### InternLM-XComposer2.5-OmniLive: A Comprehensive Multimodal System for Long-term Streaming Video and Audio Interactions (https://arxiv.org/abs/2412.09596)
Comments:
          Github Repo: this https URL

- **What's New**: 본 연구는 인간의 인지 방식을 모방하여 장기간 동안 환경과 상호작용할 수 있는 AI 시스템 개발을 목표로 하고 있습니다. 기존의 MLLMs(다중모달 대형 언어 모델)은 실시간으로 입력을 처리하고 출력을 생성하는 기능이 제한적입니다. 이러한 한계를 극복하기 위해 연구팀은 전혀 새로운 구조의 InternLM-XComposer2.5-OmniLive(IXC2.5-OL) 시스템을 제안하였으며, 이는 스트리밍 비디오 및 오디오와 실시간으로 상호작용할 수 있는 능력을 갖추고 있습니다.

- **Technical Details**: IXC2.5-OL 시스템은 스트리밍 인지 모듈, 다중모달 장기 기억 모듈 및 추론 모듈의 세 가지 주요 구성 요소로 구성되어 있습니다. 스트리밍 인지 모듈은 비디오와 오디오의 스트림을 실시간으로 처리하여 곧바로 기억에 중요한 세부정보를 저장합니다. 다중모달 장기 기억 모듈은 단기 기억 정보를 압축하여 장기 기억으로 변환함으로써 정보 검색의 효율성과 정확성을 증가시킵니다.

- **Performance Highlights**: IXC2.5-OL 시스템은 오디오 및 비디오 벤치마크에서 뛰어난 성능을 보여주었습니다. 특히 비디오 이해에서 10B 이하의 모델 중에서 최첨단 성능을 기록했으며, StreamingBench에서 73.79%의 SOTA(상태 최적화 기술) 결과를 달성했습니다. 모델 및 소스 코드는 Github을 통해 공개되어 있어 다중 모달 스트리밍 상호작용 커뮤니티의 발전에 기여할 것입니다.



### Neptune: The Long Orbit to Benchmarking Long Video Understanding (https://arxiv.org/abs/2412.09582)
- **What's New**: 이 논문은 긴 동영상을 이해하기 위한 도전적인 질문-답변-혼란 집합(Question-Answer-Decoy Sets)을 생성하는 반자동 파이프라인을 소개합니다. 기존의 비디오 데이터셋은 주로 10초에서 30초의 짧은 클립에 중점을 두고 있어, 긴 비디오 이해 측면에서의 한계를 극복할 필요가 있었습니다. 본 연구에서는 VLMs(비디오 언어 모델)와 LLMs(대규모 언어 모델)를 활용하여 자동으로 밀도가 높은 시간 정렬 비디오 캡션 및 도전적인 질문 답변 세트를 생성하는 확장 가능한 데이터셋 생성 파이프라인을 제안합니다.

- **Technical Details**: 제안된 파이프라인은 YouTube의 모든 비디오에 적용 가능하며, 자동 캡션 생성, 자동 음성 인식(ASR), 샷 경계 및 비디오 메타데이터 추출 과정을 포함합니다. 이 과정에서 LLM에게 체계적인 프롬프트를 통해 다단계 사고 과정을 거쳐 이미지 캡션과 결합하여 질문-답변-혼란 집합(QAD)을 만듭니다. 대다수의 파이프라인은 자동화되어 있으며, 마지막 단계에서 품질을 보증하기 위한 rater 검증 단계를 포함합니다. 최종적으로 Neptune이라는 데이터셋이 생성되며, 다양한 비디오와 여러 모드의 평가를 제공합니다.

- **Performance Highlights**: Neptune 데이터셋은 2,405개 비디오에 대해 3,268개의 QAD 주석을 포함하여 평가 전용으로 설계되었습니다. 이 데이터셋은 시간 순서, 카운팅 및 상태 변화와 같은 질문들에서 현재 공개된 긴 비디오 모델들이 부족한 성능을 보인다는 점이 부각되었습니다. 또한 새로운 공개 소스 모델 기반 메트릭인 GEM(Gemma Equivalence Metric)을 도입하여 개방형 질문 응답의 평가 지표를 제공합니다. 연구 결과는 기존 공개 도구와 비공식 모델 간의 성능 차이를 명확하게 나타내어, 더 발전된 모델 개발을 촉진할 수 있는 계기가 될 것입니다.



### A Theoretical Analysis of Soft-Label vs Hard-Label Training in Neural Networks (https://arxiv.org/abs/2412.09579)
Comments:
          Main Body of the Paper is under Review at L4DC 2025

- **What's New**: 이 논문은 저자들이 소프트 레이블(soft-label) 학습의 효율성을 강조하는 새로운 실험 결과를 제시합니다. 특히, 소프트 레이블 학습이 하드 레이블(hard-label) 학습보다 신경망(neural network)의 뉴런(neuron) 수를 현저히 줄여주는 이유를 규명하려고 합니다. 이러한 발견은 데이터셋이 더 어려워질수록 소프트 레이블 학습의 성능 차이가 더욱 두드러진다는 것을 보여줍니다.

- **Technical Details**: 연구진은 이론적 기여를 통해 소프트 레이블 학습의 뉴런 필요 개수가 $O(\frac{1}{\gamma^2 \epsilon})$로 줄어드는 것을 증명했습니다. 여기서 $eta$는 한계 커널(limiting kernel)의 분리 마진(separation margin)을 나타내며, 이는 고차원의 분류 문제에서 소프트 레이블 학습이 어떻게 효과적으로 뉴런 수를 감소시킬 수 있는지를 설명합니다. 반면, 하드 레이블 학습은 훨씬 더 많은 뉴런을 요구하여, 데이터셋이 복잡해질수록 그 차이가 명확해집니다.

- **Performance Highlights**: 심층 신경망(deep neural networks) 및 이론적 실험 결과들은 소프트 레이블 학습이 하드 레이블 학습에 비해 더 낮은 분류 손실(classification loss)을 달성하는 데 필요한 뉴런 수가 적다는 것을 입증합니다. 특히, 데이터셋이 분류하기 어렵거나 복잡할 경우 소프트 레이블 접근 방식이 제공하는 성능 향상이 두드러집니다. 이 연구는 소프트 레이블 학습의 활용 가능성을 실질적으로 보여주며, 나아가 네트워크 디자인 및 학습 전략에 중요한 영향을 미칠 것입니다.



### DISHONEST: Dissecting misInformation Spread using Homogeneous sOcial NEtworks and Semantic Topic classification (https://arxiv.org/abs/2412.09578)
- **What's New**: 이번 연구는 COVID-19 팬데믹에 따른 온라인 플랫폼의 잘못된 정보 확산을 분석하며, 이러한 현상이 '에코 챔버' 개념과 어떻게 연결되는지를 탐구합니다. 이 연구에서는 사용자 간의 사회적 상호작용과 트윗 콘텐츠 분석을 결합하여 이 두 차원을 연결짓는 새로운 방법론을 제안합니다. 특히, 사용자가 사회 네트워크를 통해 이동하는 속도를 측정하는 새로운 지표인 'node speed'를 개발하였습니다.

- **Technical Details**: 연구 방법론은 두 주요 경로로 구성됩니다. 첫 번째로, Twitter 사용자의 사회적 네트워크를 나타내는 그래프를 구성하고 분석합니다. 두 번째로, Top2Vec를 사용하여 사용자가 트윗하는 주제를 모델링하고, 다양한 주제에 대한 경향성을 추론합니다. 데이터 출처로는 1,235,833개의 백신 망설임 관련 트윗을 포함한 Avax 데이터셋을 활용하였습니다.

- **Performance Highlights**: 팬데믹 관련 잘못된 정보에 대한 사회적 행동과 트윗 내용 간의 상관관계를 분석하는 결과, 에코 챔버 현상에 대한 일반적인 직관을 뒷받침하는 증거가 제시되었습니다. 연구는 사회적 상호작용의 속도와 사용자 주제 다양성 간의 연결을 강조하며, 하위 커뮤니티에서도 잘못된 정보가 여전히 활성화되어 있음을 보여줍니다.



### JuStRank: Benchmarking LLM Judges for System Ranking (https://arxiv.org/abs/2412.09569)
- **What's New**: 이번 연구에서는 LLM 기반의 평가자가 시스템 간의 상대적 질을 평가하는 새로운 벤치마크인 JuStRank를 소개합니다. JuStRank는 모델의 정확한 순위를 매기는 능력을 통해 평가자를 비교하며, 더불어 평가자의 시스템 편향 성향을 정량화합니다. 이 연구는 시스템 수준에서의 평가자의 행동과 품질을 탐구하는 첫 번째 대규모 연구가 됩니다.

- **Technical Details**: 연구에서 LLM 기반 평가자가 다양한 시스템의 출력을 평가하여 생성된 시스템 점수를 기준으로 한 정확한 모델 성능 순위를 도출하는 방법론을 제안합니다. 기존 연구들은 인스턴스 기반의 성능 측정에 집중해왔지만, 본 연구는 시스템 레벨 평가의 중요성을 강조합니다. 이를 통해 평가자가 각기 다른 모델을 비교하고 순위를 매길 때 발생하는 편향과 다면적인 행동 특성들을 분석합니다.

- **Performance Highlights**: JuStRank는 4848개의 최신 LLM들과 보상 모델들을 포괄하는 대규모 벤치마크로, 평가의 일관성과 정확성을 크게 향상시킬 가능성을 제시합니다. 이 연구를 통해서 시스템 수준의 평가자가 일관성을 높이고 우수한 모델과 열등한 모델 간의 간극을 더욱 두드러지게 만드는 성향을 가질 수 있음을 보여줍니다. 또한, 시스템 모델에 대한 평가자의 편향을 규명함으로써 더욱 정확한 순위 산출을 꾀할 수 있게 됩니다.



### Sail into the Headwind: Alignment via Robust Rewards and Dynamic Labels against Reward Hacking (https://arxiv.org/abs/2412.09544)
Comments:
          46 pages, 3 figures

- **What's New**: 이 논문에서는 인간의 선호를 반영한 AI 시스템의 보상 해킹(reward hacking) 문제를 조사합니다. 특히, 오프라인(preference optimization)에서 보상 해킹의 두 가지 유형, 즉 Type I과 Type II을 식별하고 이로 인해 발생하는 문제점들을 분석합니다. 이를 해결하기 위해 새로운 방법인 POWER와 POWER-DL을 제안하여 보상 해킹을 완화하는 방안을 모색합니다.

- **Technical Details**: Type I 보상 해킹은 불리한 선택이 우선적으로 더 긍정적으로 평가받는 문제로, 고통 포함(choice coverage) 파악의 결여에서 비롯됩니다. 반면 Type II 보상 해킹은 좋은 선택이 상대적으로 더 부정적으로 평가되는 현상입니다. 이 문제를 해결하기 위해, POWER는 Guiaşu의 가중 엔트로피(wighted entropy)를 사용하여 안정적인 보상 최대화를 목표로 하며, 동적 레이블 업데이트(dynamic labels)를 통해 Type II 문제를 해결합니다.

- **Performance Highlights**: POWER-DL은 기존의 최첨단 방법들과 비교하여 각종 설정에서 일관되게 우수한 성능을 발휘합니다. AlpacaEval 2.0과 Arena-Hard에서 DPO 대비 각각 최대 13.0 및 11.5 포인트의 성능 개선을 보여줍니다. 또한, 수학적 추론 및 지시 이행과 같은 하위 작업에서도 성능을 유지하거나 향상시킴으로써, 보상 해킹에 대한 견고성을 입증합니다.



### Efficient and Comprehensive Feature Extraction in Large Vision-Language Model for Clinical Pathology Analysis (https://arxiv.org/abs/2412.09521)
- **What's New**: 이 논문에서는 병리 진단의 효율성과 정확성을 높이기 위해 두 가지 혁신적인 전략을 제안합니다. 첫 번째는 mixed task-guided feature enhancement (MTGFE)로, 이는 병변과 관련된 세부 사항을 다양한 스케일에서 효과적으로 추출하도록 기능을 안내합니다. 두 번째는 prompt-guided detail feature completion (PGDFC)로, 특정 프롬프트에 기반하여 WSI에서 세밀한 특징을 통합하여 추론 속도를 유지하면서도 진단 정보의 손실을 방지합니다.

- **Technical Details**: 고해상도 전 슬라이드 이미지 (WSI)를 기반으로 한 종합 데이터세트를 사용하여 490,000개의 샘플을 수집하였으며, 이를 통해 병리학 특화 대형 비전 언어 모델 (LVLM)인 OmniPath를 훈련시켰습니다. MTGFE와 PGDFC 전략을 통해 모델이 병변 세부 정보를 보다 잘 인식하고, 다양한 병리 진단 분석 작업에서 필요한 멀티스케일 특징을 완벽하게 지원할 수 있게 됩니다. 이 과정에서, 특정 병리 개념 탐지를 위한 지침 데이터도 모델에 추가되어 정확성을 높였습니다.

- **Performance Highlights**: OmniPath는 기존의 병리 LVLM보다 진단 정확도 및 효율성에서 현저히 우수한 성과를 보여주었습니다. 이 모델은 다양한 병리 응용 분야에서 보조 진단을 위한 상호 작용적이고 임상에 적합한 접근 방식을 제공합니다. 실험 결과는 OmniPath가 병리학의 다양한 진단 작업에서 기존 방법들을 뛰어넘는 성능을 보였음을 입증하며, 임상 병리 진단의 실제 요구에 더 적합한 솔루션으로 자리매김하였습니다.



### Vision Transformers for Efficient Indoor Pathloss Radio Map Prediction (https://arxiv.org/abs/2412.09507)
Comments:
          Work partly supported by the RA Science Committee grant No. 22rl-052 (DISTAL) and the EU under Italian National Recovery and Resilience Plan of NextGenerationEU on "Telecommunications of the Future" (PE00000001 - program "RESTART")

- **What's New**: 이 연구는 Vision Transformers (ViTs)를 활용하여 실내 경로 손실(Pathloss) 라디오 맵 예측 문제를 해결하는 새로운 접근 방식을 제시합니다. 기존의 방법론인 합성곱 신경망(CNN)에서 비전 트랜스포머로 전환함으로써 더 복잡한 실내 환경에서의 모델 일반화 능력을 평가하고자 합니다. ICASSP 2025에서 개최되는 제1회 실내 경로 손실 라디오 맵 예측 챌린지를 위해 특별히 설계된 연구입니다.

- **Technical Details**: 연구에 사용된 데이터셋은 레이 트레이싱(ray-tracing) 알고리즘을 기반으로 생성된 경로 손실 라디오 맵으로 구성됩니다. 이 데이터셋은 25개의 다양한 실내 구조와 3개의 주파수 대역, 5개의 안테나 방사 패턴을 포함하여, 신경망의 일반화 능력을 평가하는 기준이 됩니다. 입력 데이터는 경로 손실을 효과적으로 나타내기 위해 두 가지 투명도 채널과 거리 정보를 포함한 3채널을 사용합니다.

- **Performance Highlights**: 데이터 증강(data augmentation) 기법을 통해 훈련 데이터를 효과적으로 증가시킴으로써 모델의 일반화 능력을 강화합니다. 이 연구는 MixUp 및 회전 증강(rotation augmentation) 기법을 통해 입력 데이터의 다양성을 증대시키고, 실시간 환경을 보다 잘 반영할 수 있는 기술적 기반을 제공합니다. 결과적으로, 다양한 시나리오에서도 강력한 성능을 발휘할 수 있는 모델 개발에 성공하였습니다.



### Video Seal: Open and Efficient Video Watermarking (https://arxiv.org/abs/2412.09492)
Comments:
          Code available at this https URL

- **What's New**: 이 논문에서는 Video Seal이라는 포괄적인 프레임워크를 소개하며, 이는 신경망 기반의 비디오 워터마킹(neural video watermarking)을 위한 공개 모델로 활발한 연구를 촉진하는 것을 목표로 하고 있습니다. 기존의 비디오 워터마킹 기법의 한계를 극복하기 위해 이 모델은 임베더(embedder)와 이엑스트랙터(extractor)를 함께 훈련시켜 높은 강인성(robustness)을 유지합니다. 또한, 이 논문에서는 임베딩을 위한 새로운 기법인 temporal watermark propagation을 도입하여 모든 고해상도 프레임에 워터마크를 추가하지 않고도 효율적인 비디오 워터마킹을 가능하게 합니다.

- **Technical Details**: 기법적으로, Video Seal은 2D 차원에서 작동하여 스트리밍(streamability)을 보장하고 추출을 간소화하며 유연성을 제공합니다. 이 모델은 고해상도 및 다양한 길이의 비디오에 효과적으로 적용될 수 있도록 설계되었으며, 임베더는 효율적인 U-Net 아키텍처를 사용하고 이엑스트랙터는 비전 변환기(vision transformer)를 기반으로 구성되어 유연한 추론을 가능하게 합니다. 모델은 이미지 사전 훈련(image pre-training) 및 하이브리드 후 훈련(hybrid post-training)을 포함한 다단계 훈련 방식을 적용하여 다양한 비디오 변환 및 고압축이 일어나는 상황에서도 높은 성능을 유지합니다.

- **Performance Highlights**: 실험 결과는 Video Seal이 기존의 강력한 베이스라인(like MBRS, TrustMark, and WAM)보다 높은 강인성을 달성하며, 특히 지리적 변형 및 비디오 압축과 같은 도전에 잘 대응함을 보여줍니다. 또한, PSNR을 통해 품질 저하 없이 비트 정확성과 강인성을 향상시킬 수 있는 이엑스트랙터 미세 조정(extractor fine-tuning)의 중요성이 강조되었습니다. Released artifacts, including 모델 체크포인트와 평가 코드들은 연구자들이 비디오 워터마킹 기술의 진전을 촉진할 수 있도록 공유되고 있습니다.



### Regression and Classification with Single-Qubit Quantum Neural Networks (https://arxiv.org/abs/2412.09486)
Comments:
          21 pages, 7 figures, 6 tables

- **What's New**: 이번 연구는 매개변수화된 양자 회로(Parameterized Quantum Circuits, PQC)를 활용하여 단일 큐비트 양자 신경망(Single-Qubit Quantum Neural Network, SQQNN)을 제안하는 것입니다. 이 SQQNN는 회귀(regression) 및 분류(classification) 작업에서 높은 효율성을 보여주며, 새로운 훈련 방법을 통해 빠른 학습 속도를 자랑합니다. 특히, MNIST 데이터셋에서의 수행 능력은 거의 오류가 없음을 입증하며, 현재의 근단계 양자 기기에서의 사용 가능성을 강조합니다.

- **Technical Details**: SQQNN의 핵심은 매개변수화된 단일 큐비트 유니터리 연산자와 양자 측정을 활용하여 효율적인 학습을 수행하는 것입니다. 이 네트워크는 그래디언트 강하(gradient descent) 방식을 통해 회귀 작업을 훈련하고, 테일러 급수(Taylor series)를 바탕으로 한 새로운 훈련 방법으로 분류 작업을 수행합니다. 이로 인해 훈련시간이 획기적으로 단축되어 반복적인 방법보다 더 빠른 학습 성능을 제공합니다.

- **Performance Highlights**: 여러 응용 프로그램에서 SQQNN은 회귀 및 분류 작업에서 매우 뛰어난 성능을 보여주었습니다. 특히, 로지스틱 게이트 평가와 MNIST 데이터셋에서 낮은 오류율을 기록하며, 고차원 리얼 월드 데이터 문제 처리에서도 효과적임을 증명했습니다. 이러한 결과는 SQQNN의 강력한 다목적성과 확장성이 뛰어나다는 것을 잘 보여줍니다.



### New keypoint-based approach for recognising British Sign Language (BSL) from sequences (https://arxiv.org/abs/2412.09475)
Comments:
          International Conference on Computer Vision (ICCV) - HANDS Workshop

- **What's New**: 이 논문은 British Sign Language (BSL) 단어를 인식하기 위해 설계된 새로운 keypoint 기반의 분류 모델을 소개합니다. BOBSL 데이터셋을 사용하여 성능을 평가한 결과, keypoint 기반 접근 방식이 RGB 기반 모델보다 계산 효율성과 메모리 사용 측면에서 뛰어난 성능을 보였습니다. 이 모델은 기존의 연구와 비교할 때, BSL 단어 분류를 위한 keypoint 기반 모델의 최초 적용이라고 할 수 있습니다.

- **Technical Details**: 본 연구에서는 2D keypoints를 사용하여 얼굴, 오른손, 왼손 및 자세의 특정 지점을 표현합니다. 이러한 접근 방식의 주요 장점으로는 정보 제어가 용이하고, 조명과 의류와 같은 불필요한 요소를 제거할 수 있으며, 프레임 속도로 키포인트를 계산할 수 있어 실시간 모델 실행이 가능하다는 점이 있습니다. Mediapipe를 사용하여 실시간으로 키포인트를 추출하고, Transformer-based 모델에 입력으로 제공하여 BSL 단어를 인식합니다.

- **Performance Highlights**: 우리의 모델은 keypoint 기반 접근 방식이 RGB 기반 모델보다 뛰어난 계산 효율성과 메모리 사용, 훈련 속도를 나타낸다는 것을 보여줍니다. BOBSL 데이터셋에서는 총 8,162개의 단어 카테고리를 갖는 분류 문제가 발생하였으며, 여기에 대한 훈련의 결과로 모델의 성능이 향상되었음을 확인하였습니다. 다양한 메소드와 keypoint 수에 대한 실험을 통해 모델의 적합성을 평가했습니다.



### STORM: A Spatio-Temporal Factor Model Based on Dual Vector Quantized Variational Autoencoders for Financial Trading (https://arxiv.org/abs/2412.09468)
- **What's New**: 이번 연구에서는 금융 거래에서 효과적으로 시간적 패턴을 포착하지 못하는 기존의 변동 오토인코더 기반(latent factor models) 모델의 한계를 극복하고자 합니다. 우리는 dual vector quantized variational autoencoders (VQ-VAE) 기반의 Spatio-Temporal factOR Model(STORM)을 제안하여 주식의 시공간적 특성을 추출하고 이를 다차원 임베딩(multi-dimensional embeddings)으로 표현합니다. 이를 통해 기존 모델의 낮은 품질과 다양성을 극복하고, 금융 거래에서의 팩터 선택을 가능하게 합니다.

- **Technical Details**: STORM은 cross-sectional(교차단면) 및 time-series(시계열) 요소를 포착하기 위해 dual VQ-VAE 아키텍처를 설계합니다. 이 모델은 팩터 임베딩의 다양성을 보장하고, 각 요소 간의 독립성을 확보하기 위해 diversity loss와 orthogonality loss를 도입하여 팩터 구분 및 선택 과정을 투명하게 만듭니다. 팩터들은 코드북의 클러스터 센터를 사용해 명확하게 구분되며, 이는 금융 데이터의 복잡성과 비선형성을 포착하는 데 기여합니다.

- **Performance Highlights**: 두 가지 주식 데이터셋에 대한 포트폴리오 관리 및 여섯 개 특정 주식에 대한 개별 거래 작업을 통해 STORM의 성능을 검증했습니다. 광범위한 실험 결과, STORM은 기본 모델들 대비 뛰어난 성능을 보였으며, 2가지 예측 메트릭과 6가지 표준 금융 메트릭에서 모두 우수한 결과를 기록했습니다. 이러한 결과는 STORM이 하위 작업에 대한 적응력이 뛰어남을 시사합니다.



### Solving Multiagent Path Finding on Highly Centralized Networks (https://arxiv.org/abs/2412.09433)
- **What's New**: 이번 논문에서는 Multiagent Path Finding (MAPF) 문제를 규명하고, 알고리즘의 행동을 체계적으로 연구하는 최근의 결과를 보완합니다. 특히, 스타 형태의 네트워크 또는 11개의 잎을 가진 트리에서 MAPF가 NP-hard임을 보여줍니다. 이 연구 결과는 MAPF 문제의 다루기 쉬운 성질에 대한 이해를 확대하는 데 기여합니다.

- **Technical Details**: 본 연구에서는 접근 방식으로 exact algorithms를 설계하여 최적의 솔루션을 보장하는 것을 목표로 합니다. 전통적인 MAPF 문제의 컴퓨터 복잡성을 감안할 때, 일반적인 사례에서 해결이 불가능하지만, 구조적 파라미터에 대한 분석을 통해 이는 가능하다고 주장합니다. 여기서 제안하는 첫 번째 결과는 유한한 vertex cover number를 갖는 그래프에서 MAPF가 다루기 힘든 문제임을 보여줍니다.

- **Performance Highlights**: 마지막으로, 고밀도 그래프에서 MAPFC 문제를 해결하는 효율적인 알고리즘을 제시합니다. 이 알고리즘은 실제 애플리케이션에서 대각선 접속 지점이 적은 경우에도 동작할 수 있는 이점을 제공합니다. 연구 결과는 MAPF 문제의 다양한 구조적 접근 방식을 제시하며, 실제 적용 가능성을 높이는 데 기여할 것으로 기대됩니다.



### From Intention To Implementation: Automating Biomedical Research via LLMs (https://arxiv.org/abs/2412.09429)
- **What's New**: 이 논문에서는 BioResearcher라는 최초의 엔드 투 엔드(end-to-end) 자동화 시스템을 소개합니다. 이 시스템은 생물의학 연구의 모든 단계를 간소화하며, 특히 dry lab 실험을 포함하는 연구 과정을 자동화합니다. BioResearcher는 검색, 문헌 처리, 실험 디자인 및 프로그래밍을 위한 전문 에이전트를 통합하는 모듈식 다중 에이전트 아키텍처를 채택했습니다.

- **Technical Details**: BioResearcher는 다학제적 기술 세트를 통합하기 위해 LLMs를 기반으로 하는 모듈식 다중 에이전트 아키텍처를 사용합니다. 이 시스템은 연구 목표에 따라 관련 문헌을 조사하고 적절한 실험 프로토콜을 설계하며, 이를 구현하기 위한 프로그램을 작성하는 기능을 가지고 있습니다. 또한, 실험 프로토콜의 품질을 평가하기 위한 새로운 메트릭을 제안했습니다.

- **Performance Highlights**: BioResearcher는 8가지 이전에 해결되지 않았던 연구 목표에 대한 평균 실행 성공률 63.07%를 달성했습니다. 제안된 품질 메트릭을 기준으로 보통 에이전트 시스템보다 평균 22.0% 더 나은 성과를 기록했습니다. 이 시스템은 연구자들의 작업 부담을 줄이고 생물의학 발견을 가속화할 수 있는 상당한 잠재력을 보여줍니다.



### Reinforcement Learning Within the Classical Robotics Stack: A Case Study in Robot Soccer (https://arxiv.org/abs/2412.09417)
Comments:
          Submitted to ICRA 2025

- **What's New**: 본 논문에서는 부분적으로 관측 가능한 다중 에이전트 환경에서의 로봇 의사결정 문제를 해결하기 위한 새로운 접근법을 제시합니다. 전통적인 로봇 스택에 강화 학습(RL)을 통합하고, 행위를 학습된 서브 행위로 분해하는 아키텍처를 개발했습니다. 이 시스템은 2024 RoboCup SPL 챌린지에서 우승하는 결과를 가져왔습니다.

- **Technical Details**: 제안된 아키텍처는 인식, 상태 추정, 행동 및 제어를 모듈로 분리하여 구성됩니다. RL의 주요 역할은 각 로봇의 고수준 의사결정을 담당하는 행동 모듈에 통합됩니다. 이 과정에서 다중 충실도(sim2real) 훈련 접근법을 채택하여 서로 다른 신뢰성을 가진 시뮬레이터를 활용하여 행동을 효과적으로 훈련하고, 서브 행위 간의 휴리스틱 선택을 통해 전략을 부여합니다.

- **Performance Highlights**: 팀 WisTex United는 8게임 중 7게임에서 승리했으며, 총 점수는 39대 7로 나타났습니다. 또한, 우리의 아키텍처에 대한 주요 디자인 선택의 중요성을 실험을 통해 검증하여, 이들이 2024 SPL 챌린지에서의 승리에 어떻게 기여했는지를 분석했습니다. 본 연구는 RL의 고수준 의사결정 적용 가능성을 고무적으로 보여주었습니다.



### UFO: Enhancing Diffusion-Based Video Generation with a Uniform Frame Organizer (https://arxiv.org/abs/2412.09389)
Comments:
          Code:this https URL

- **What's New**: 최근 확산 기반의 비디오 생성 모델에서 중요한 진전을 이루었지만, 기존 모델들은 일관성과 이미지 품질 저하 등의 문제를 겪고 있습니다. 이를 해결하기 위해 새로운 비침습적 플러그인인 Uniform Frame Organizer (UFO)를 제안합니다. UFO는 모든 확산 기반 비디오 생성 모델과 호환 가능하며, 동적 요소와 정적 정확성을 유지하며 향상된 일관성을 제공합니다.

- **Technical Details**: UFO는 조정 가능한 강도를 가지는 일련의 적응형 어댑터로 구성되어 있습니다. 이 어댑터는 영상의 배경과 전경 간의 일관성을 크게 높이며, 모델의 원래 매개변수를 변경하지 않고도 이미지 품질을 개선할 수 있도록 설계되었습니다. UFO는 훈련 데이터로 적은 양의 비디오 프레임이나 이미지를 사용할 때 강도를 최대치로 설정하여 모델 출력이 정적 비디오에 가까워지도록 합니다.

- **Performance Highlights**: UFO는 공공 비디오 생성 벤치마크 Vbench에서 비디오 일관성과 품질을 크게 향상시키는 것으로 입증되었습니다. 사용자는 매우 적은 리소스와 비용으로 UFO를 훈련할 수 있으며, 비디오-텍스트 쌍 없이도 일관성을 높일 수 있습니다. 이러한 효율성 덕분에 사용자는 고품질 비디오 생성의 부담을 크게 줄일 수 있습니다.



### All You Need in Knowledge Distillation Is a Tailored Coordinate System (https://arxiv.org/abs/2412.09388)
- **What's New**: 이번 논문에서는 Knowledge Distillation (KD)의 새로운 접근 방식인 Tailored Coordinate System (TCS)을 제안합니다. TCS는 기존의 KD 방법이 요구하는 작업 특화된 teacher 모델 없이도 스스로 선행 학습된(self-supervised learning, SSL) 모델의 어두운 지식을 효과적으로 추출할 수 있도록 설계되었습니다. 이 방법은 하나의 포워드 패스를 통해 teacher의 feature를 기반으로 좌표계를 형성하여 student 네트워크를 최적화할 수 있게 합니다.

- **Technical Details**: TCS는 주로 Principal Component Analysis (PCA)를 이용하여 teacher의 feature에서 좌표계를 생성합니다. 이 과정에서 데이터 증강을 필요로 하지 않아 오직 하나의 포워드 패스만으로도 비용을 최소화할 수 있습니다. 이후 student의 feature를 이 좌표계에 맞추어 조정하는 과정은 iterative feature selection 기법에 의해 지원되며, 이는 TCS의 효율성을 높이는 데 기여합니다.

- **Performance Highlights**: TCS 방법은 최신 KD 기술에 비해 훨씬 더 높은 정확도를 보여주면서도 훈련 시간과 GPU 메모리 비용을 약 절반으로 줄일 수 있음을 실험적으로 입증하였습니다. 이 방식은 다양한 아키텍처에 적용 가능하며, few-shot learning을 포함한 다양한 작업에서도 효과적으로 작동합니다. TCS는 교육받은 teacher 없이도 학생 네트워크를 초고속으로 훈련할 수 있는 가능성을 보여줍니다.



### Distributed Intelligent System Architecture for UAV-Assisted Monitoring of Wind Energy Infrastructur (https://arxiv.org/abs/2412.09387)
Comments:
          Wind turbine inspection, UAV, intelligent systems, distributed architecture, defect detection, renewable energy maintenance, automated monitoring

- **What's New**: 본 논문은 녹색 에너지(green energy)의 급속한 발전에 발맞추어, 풍력 터빈(wind turbines)의 효율성과 신뢰성을 향상시키기 위한 혁신적인 지능형 시스템 아키텍처(intelligent system architecture)를 소개합니다. 이 시스템은 드론(unmanned aerial vehicles, UAVs)을 활용하여 비주얼(visual) 및 열(thermal) 센서를 통합하여 실시간으로 데이터를 수집하고 처리합니다.

- **Technical Details**: 제안된 시스템은 분산 프레임워크(distributed framework) 내에서 고급 알고리즘(algorithms)을 사용하여 점검 정확도(inspection accuracy)와 효율성을 향상시킵니다. 실험은 우크라이나의 'Staryi Sambir-1' 풍력 발전소에서 진행되었으며, 이 과정에서 전통적인 방법 대비 결함 탐지 정확도를 94%까지 향상시켰습니다.

- **Performance Highlights**: 검사 시간(inspection time)은 터빈당 1.5시간으로 줄여졌습니다. 이러한 결과는 제안된 시스템이 풍력 터빈 유지보수에 있어 확장 가능하고 신뢰할 수 있는 솔루션을 제공하며, 재생 가능 에너지 인프라의 내구성(durability) 및 성능(performance) 향상에 기여함을 보여줍니다.



### Diffusion Model with Representation Alignment for Protein Inverse Folding (https://arxiv.org/abs/2412.09380)
- **What's New**: 이번 연구에서는 단백질의 역 접힘 문제를 해결하기 위해 새로운 방법인 확산 모델과 표현 정렬(Representation Alignment) 기법을 결합한 DMRA(Diffusion Models with Representation Alignment)를 제안합니다. 이 방법은 기존의 방법들이 어려움을 겪는 잔여간 상호작용을 효과적으로 캡처하며, 전체 단백질 구조에서 맥락 정보를 집계하여 각 잔여체에 선택적으로 분배합니다. 또한, 노이즈가 있는 숨겨진 표현과 깔끔한 의미론적 표현을 정렬하여, AA(아미노산) 시퀀스의 복원을 향상시킵니다.

- **Technical Details**: DMRA는 단백질 구조를 K-최근접 이웃 그래프로 변환하여 압축한 후, 메시지 패싱을 이용하여 잔여체 간의 정보를 효과적으로 전달합니다. 이를 통해 기존의 동브리 고정 방식에서 벗어나, 공통 센터를 설정하여 잔여체 간의 의사소통을 극대화합니다. 이와 함께, 아미노산 타입에 대한 사전 정의된 의미적 표현을 사용하여 각 잔여체의 숨겨진 표현을 조정함으로써 고차원적인 특성을 잘 반영할 수 있도록 합니다.

- **Performance Highlights**: CATH4.2 데이터셋에서 DMRA는 기존의 선도적인 방법들을 초월하여 최첨단 성능을 달성했으며, TS50 및 TS500 데이터셋에서도 뛰어난 일반화 성능을 보여주었습니다. 이를 통해 제안된 방법이 단백질 시퀀스 복원 및 단백질 구조 해석에서 높은 효과를 나타내는 것을 확인했습니다. 연구의 성공적인 결과는 기존 방법들이 놓치고 있는 표현적 피드백을 활용하고, 잔여체 간의 상호작용을 효과적으로 캡처함으로써 가능하였습니다.



### Word Sense Linking: Disambiguating Outside the Sandbox (https://arxiv.org/abs/2412.09370)
- **What's New**: 본 연구에서는 Word Sense Disambiguation (WSD)의 새로운 작업인 Word Sense Linking (WSL)을 소개합니다. WSD는 주어진 맥락에서 단어의 적절한 의미를 선택하는 작업으로, 최근에는 성능이 향상되었음에도 불구하고 하위 응용 프로그램을 찾는 데 어려움을 겪고 있습니다. WSL은 입력 텍스트와 참조 의미 목록이 주어진 경우, 의미를 명확하게 구별하고 연결하는 새로운 접근 방식을 제공합니다.

- **Technical Details**: WSL의 주요 특징은 세 가지 서브태스크로 나눌 수 있는 점입니다: Concept Detection (CD), Candidate Generation (CG), 그리고 Word Sense Disambiguation (WSD)입니다. CD는 입력 텍스트에서 구별할 구간을 찾고, CG는 각 구간에 대해 의미 후보 목록을 생성합니다. 이 과정은 최신 Transformer 기반 아키텍처를 활용하여 구현되며, WSD 시스템의 가정들을 완화하면서 평가됩니다.

- **Performance Highlights**: WSL에 적용된 모델은 실세계 환경에서 WSD 시스템을 확장할 때 직면하는 여러 문제를 드러냅니다. 기존의 WSD 시스템을 WSL로 간단히 확장할 경우 성능 저하가 발생하는 반면, 본 연구의 모델은 더 높은 견고성을 유지하며 성능에서 상당한 우위를 보입니다. 이 연구는 WSL의 평가 데이터셋과 새로운 아키텍처를 제공함으로써 이 분야의 진전을 촉진하는 것을 목표로 하고 있습니다.



### Causal Graphical Models for Vision-Language Compositional Understanding (https://arxiv.org/abs/2412.09353)
- **What's New**: 최근 연구에 따르면 Vision-Language Models (VLMs)는 인간 언어의 조합적 속성을 완전히 이해하는 데 어려움을 겪고 있으며, 이는 일반적으로 이미지 캡션을 "bag of words"로 모델링하고 있기 때문입니다. 이 연구에서는 Causal Graphical Model (CGM)을 사용하여 텍스트 및 시각적 토큰 간의 의존 관계를 모델링하고, VLM 비주얼 인코더에 의해 조건화된 디코더를 훈련하는 방식을 제안합니다. 우리의 접근 방식은 조합 작업에서 VLM의 성능을 크게 향상시키는 데 기여합니다.

- **Technical Details**: 이 연구에서 제안한 방법은 전통적인 순차적(autoregressive) 예측 대신 반순차적(semi-parallel) 예측 전략을 사용하여 CGM 구조를 따릅니다. 이를 통해 디코더는 문장 내 주요 인과적 의존 관계만 학습하고, 불필요한 상관관계를 배제할 수 있습니다. 특히, 의존 파서를 통해 작성된 의존 트리를 기반으로 CGM을 구축하여 이미지 패치와 텍스트 토큰 간의 의존 관계를 설명합니다.

- **Performance Highlights**: 다양한 실험을 통해 제안된 방법이 기존의 모든 최첨단 조합 접근 방식을 상당히 능가함을 보여주며, 평가된 모든 벤치마크에서 새로운 최첨단 성과를 기록합니다. 또한, 훨씬 적은 데이터로 훈련되었음에도 불구하고 Cap 및 CapPa 방식에서도 성능 개선을 보여줍니다.



### Does Low Spoilage Under Cold Conditions Foster Cultural Complexity During the Foraging Era? -- A Theoretical and Computational Inquiry (https://arxiv.org/abs/2412.09335)
- **What's New**: 이 연구는 인간 문화의 복잡성이 자연 환경의 변화와 자원 관리 능력과 어떻게 연관될 수 있는지를 형식적이고 다학제적인 접근법을 통해 검토한다. 특히 낮은 저장 식품 손실률을 가진 환경이 문화적 복잡성을 촉진하는 조건이 될 수 있음을 보여주기 위해 수학적 모델과 강화 학습 시뮬레이션을 도입하였다. 이를 통해 문화적 활동에 보다 많은 시간을 할애할 수 있는 조건을 수학적으로 증명하였다.

- **Technical Details**: 이 연구는 수학적 모델을 통해 음식의 수확량(yield), 부패율(spoilage), 자원 관리 기술(resource management skill), 그리고 문화적 복잡성(cultural complexity) 사이의 관계를 수립하였다. 수학적 프레임워크를 바탕으로 그룹이 환경에서 자원 관리 기술을 발전시키면서 낮은 부패와 높은 수확량을 달성할 수 있음을 설명하고, 이를 통해 문화 활동을 위한 여유 시간이 증가한다고 주장한다. 강화 학습( reinforcement learning) 알고리즘을 사용하여 다양한 상황에서 요인 간의 상관관계를 분석한다.

- **Performance Highlights**: 시뮬레이션 결과, 낮은 식품 손실률과 높은 수확량을 가진 환경에서 문화적 활동에 더 많은 시간이 할애됨을 관찰하였다. 이 연구는 환경적 안정성이 문화적 복잡성을 유도하는 가능성을 제시하며, 이는 긴 시간 동안 누적됨으로써 문화가 더욱 번창할 수 있음을 나타낸다. 이 연구는 고고학적 또는 인류학적 요소와 수학적 접근을 통합함으로써 문화의 기원에 대한 새로운 통찰을 제공한다.



### Towards Open-Vocabulary Video Semantic Segmentation (https://arxiv.org/abs/2412.09329)
Comments:
          13 pages, 7 figures

- **What's New**: 본 논문에서는 Open Vocabulary Video Semantic Segmentation (OV-VSS)라는 새로운 작업을 소개합니다. 이는 전통적인 비디오 시맨틱 세분화 모델이 직면하는 문제를 해결하고, 훈련 데이터에 없는 새로운 카테고리에 대해서도 정확한 픽셀 세분화를 목표로 합니다. 본 연구는 공간-시간 융합 모듈과 랜덤 프레임 강화 모듈을 포함하는 OV2VSS라는 새로운 기준 모델을 제안하여, 비디오 전체에서 의미적인 맥락을 넓히는 데 중점을 두고 있습니다.

- **Technical Details**: OV-VSS는 모든 비디오 프레임 내 픽셀을 분류하여 새로운 카테고리를 포함할 수 있도록 설계된 모델입니다. 기존 이미지 기반 방법 대신, CLIP 모델의 이미지와 텍스트 인코더를 활용하여 픽셀 수준의 시맨틱 정보에 집중하는 방법론을 제안합니다. 이를 통해 모델은 많은 객체나 배경 요소에 대해 공간적 및 시간적 정보를 활용하여 정확한 라벨링을 구현할 수 있습니다.

- **Performance Highlights**: VSPW 데이터셋을 통해 수행된 실험에서는, OV-VSS 모델이 보이지 않는 클래스에 대해 17.99%의 mIoU를 달성했으며, 이는 기존 이미지 기반의 오픈 빈도 세분화 방법보다 4% 향상된 결과입니다. 이러한 성과는 OV-VSS의 우수성과 일반화 능력을 증명하며, 다양한 비디오 데이터셋에서의 성능 향상 가능성을 보여줍니다.



### Auto-Regressive Moving Diffusion Models for Time Series Forecasting (https://arxiv.org/abs/2412.09328)
Comments:
          no comment

- **What's New**: 이번 논문에서는 기존의 전통적인 diffusion 기반 모델의 한계를 극복하기 위해 Auto-Regressive Moving Diffusion (ARMD) 모델을 제안합니다. ARMD 모델은 시간의 연속적 진행성을 활용하여, 과거 데이터를 기반으로 미래 값을 예측하는 새로운 접근 방식을 취하고 있습니다. 이 모델은 중간 상태 정보를 통해 예측의 정확성과 안정성을 향상시키며, 기존의 noise 기반 조건 생성을 불필요하게 만듭니다.

- **Technical Details**: ARMD 모델은 시간 시계를 연속적인 데이터 포인트의 진행으로 간주하며, 과거 데이터와의 종속성을 기반으로 설계되었습니다. 모델은 체인 기반의 diffusion을 사용하여, 역사적 시계열 데이터를 초기 상태로 설정하고 미래의 시계열을 최종 상태로 정의합니다. 이러한 과정에서, sliding 기술을 통해 중간 시리즈를 생성하고, 이를 통해 예측 정확도를 높입니다.

- **Performance Highlights**: 실험 결과, ARMD 모델은 7개의 유명한 데이터 세트에서 기존의 diffusion 기반 TSF 모델들보다 우수한 성능을 보였습니다. 모델의 설계가 예측 목표와 샘플링 과정을 일치시켜 안정성을 높이고, 예측 성능을 강화했습니다. 이러한 성과는 ARMD가 차세대 TSF 모델로 자리 잡을 수 있는 가능성을 보여줍니다.



### Benchmarking LLMs for Mimicking Child-Caregiver Language in Interaction (https://arxiv.org/abs/2412.09318)
- **What's New**: 이번 연구에서 LLMs(대규모 언어 모델)의 아동-양육자 상호작용 능력을 평가했습니다. 연구는 LLMs가 아동과 양육자의 대화에서 보이는 독특한 언어적 특징을 얼마나 잘 포착할 수 있는지를 살펴보았습니다. 발견된 바에 따르면, 최신 LLM인 Llama 3 및 GPT-4o는 단어와 발화 수준에서는 적절한 반응을 생성할 수 있지만, 아동과 양육자의 담화 패턴을 재현하는 데는 한계가 있었습니다.

- **Technical Details**: 연구진은 CHILDES 데이터셋을 활용하여 2세에서 5세 아동과 양육자의 대화를 분석했습니다. 데이터는 발화-응답 쌍으로 재구성되어 6,600개의 상호작용 쌍을 포함하게 되었습니다. 연구는 단일 회차(single-turn) 및 다중 회차(multi-turn) 테스트 방식으로 LLM의 반응을 비교하며, 제로샷(zero-shot) 및 몇 샷(few-shot) 테스트를 진행했습니다.

- **Performance Highlights**: 평가 결과, LLM들은 아동의 발화에 적절히 반응하는 데 있는 다양한 언어적 비정통성을 처리하는 데 어려움을 겪었습니다. 아동-양육자 간 대화에서의 반응은 인간과의 비교에서 구조적 지표에서 한계를 드러냈습니다. 이러한 결과는 아동과의 상호작용에 대한 LLM의 응용 가능성을 높이기 위한 포괄적인 기준 개발의 필요성을 강조하고 있습니다.



### Multimodal Sentiment Analysis based on Video and Audio Inputs (https://arxiv.org/abs/2412.09317)
Comments:
          Presented as a full paper in the 15th International Conference on Emerging Ubiquitous Systems and Pervasive Networks (EUSPN 2024) October 28-30, 2024, Leuven, Belgium

- **What's New**: 이번 연구는 비디오와 오디오 입력을 활용한 감정 인식 모델의 유용성을 입증하는 것을 목표로 하고 있습니다. 최신 연구들은 transformer 기반의 접근법을 통해 감정 분석의 정확도를 높여왔으나, 본 논문에서는 이러한 접근법들 간의 균형을 찾아 더 나은 결과를 도출하고자 하였습니다. 여러 감정 인식 데이터셋인 CREMA-D와 RAVDESS를 활용하여 모델을 훈련하며, 다양한 메소드와 평가 기법들을 적용하였습니다.

- **Technical Details**: 연구에서 사용된 주요 데이터셋은 CREMA-D와 RAVDESS입니다. CREMA-D는 7442개의 음성 클립과 91명의 배우로 구성되어 있으며, 각 클립은 분노, 혐오, 두려움, 행복, 슬픔, 중립 등의 감정을 포함하고 있습니다. RAVDESS 데이터셋은 7356개의 영상 파일로, 감정의 정확성과 강도를 평가받았습니다. 최종적으로 Facebook의 wav2vec2-large와 Google의 vivit-b-16x2-kinetics400 모델을 선택하여 오디오 및 비디오 분류 작업을 수행하였습니다.

- **Performance Highlights**: 연구의 초기 결과는 두 모델의 조합을 통한 감정 예측의 신뢰성을 높이는 방향으로 이루어졌습니다. 여러 테스트 프레임워크를 통해 최상의 정확도를 기록한 모델을 선택하였으며, 이는 향후 연구에 긍정적인 결과를 가져올 것으로 기대됩니다. 본 프로젝트의 제한된 접근 방식에도 불구하고 감정 분석의 효과적 성과를 이끌어 냈으며, 이는 후속 연구에서도 유효한 방향으로 평가될 것입니다.



### Advancing Attribution-Based Neural Network Explainability through Relative Absolute Magnitude Layer-Wise Relevance Propagation and Multi-Component Evaluation (https://arxiv.org/abs/2412.09311)
Comments:
          30 pages, 16 figures, 13 tables, ACM Transactions on Intelligence Systems and Technology

- **What's New**: 최근 딥 뉴럴 네트워크(Deep Neural Networks) 기술의 발전으로 인해 설명 가능성과 투명성이 중요한 다양한 분야에서 블랙박스 모델의 사용이 제한되고 있습니다. 본 논문은 기존의 Layer-Wise Relevance Propagation (LRP) 기법의 한계를 극복하고, 입력 뉴런의 관련성을 계층별로 전파하여 새로운 접근 방법을 제시합니다. 또한, 최근 개발된 Vision Transformer 아키텍처에 이 방법을 적용하여 두 개의 이미지 분류 데이터셋인 ImageNet과 PascalVOC에서 성능을 평가했습니다.

- **Technical Details**: 본 연구에서는 Relative Absolute Magnitude Layer-Wise Relevance Propagation (absLRP)라는 새로운 규칙을 개발하여, 동일 계층 내에서 서로 다른 절대 크기를 가진 활성화 뉴런 간의 잘못된 상대적 기여 문제를 해결합니다. 세 가지 다른 아키텍처(VGG, ResNet50, ViT-Base)를 사용하여 absLRP를 적용하고, 기존의 기법들과 비교하여 확실한 장점을 입증했습니다. 또한, Global Attribution Evaluation (GAE)이라는 새로운 평가 방법을 제안하여, 기여도 평가의 신뢰성과 강인성을 함께 평가할 수 있는 종합적인 점수를 도출합니다.

- **Performance Highlights**: 실험 결과, absLRP는 기존의 여러 최신 기법들에 비해 뛰어난 성능을 보였으며, 각각의 기법의 강점과 약점을 비교 분석할 수 있는 기회를 제공했습니다. 두 개의 유명한 이미지 분류 데이터셋에서 수행된 실험은 우리 방법론의 우수성을 명확히 드러냈습니다. 이러한 결과는 다양한 기여 기반 방법을 평가하는 데 매우 유용한 인사이트를 제공합니다.



### Learning Novel Skills from Language-Generated Demonstrations (https://arxiv.org/abs/2412.09286)
- **What's New**: 본 연구에서는 자연어 지시에서 로봇이 새로운 기술을 습득하도록 하는 스킬 학습 프레임워크를 제안합니다. 기존의 로봇 학습 알고리즘은 데모 데이터셋이나 환경 상호작용에 의존하여 높은 노동 비용과 잠재적 안전 위험을 초래했습니다. 본 프레임워크는 비전-언어 모델(vision-language models, VLM)을 활용하여 새로운 기술의 데모 비디오를 생성하며, 이 비디오는 역 동역학 모델(inverse dynamics model, IDM)으로 처리되어 비 라벨 데모로부터 동작을 추출합니다. 이를 통해 로봇들은 효과적으로 새로운 기술을 학습할 수 있습니다.

- **Technical Details**: 제안된 파이프라인은 네 가지 모듈로 구성되어 로봇이 언어 설명으로부터 직접적으로 새로운 작업을 배울 수 있도록 지원합니다. 첫 번째로, VLM이 작업 설명을 보강하여 DVG(데모 비디오 생성기)로 전달됩니다. DVG는 작업별 비디오 데모를 합성하며, 이후 IDM이 이 데모로부터 상태-동작 쌍을 추출합니다. 마지막으로, ILM(모방 학습 모델)이 환경 상태를 동작으로 매핑하여 로봇이 새로운 기술을 습득하게 합니다.

- **Performance Highlights**: 실험 결과, 제안된 파이프라인은 정밀한 데모 비디오를 생성하며 다양한 스킬 학습 알고리즘이 새로운 작업에서 원래의 세 배에 달하는 성취율을 기록하였습니다. 이러한 결과는 로봇 학습에 대한 새로운 접근 방식을 강조하며, 직관적이고 지능적인 새로운 로봇 기술 습득의 기초를 제공합니다. 특히 생성된 비디오는 다양한 로봇 기술의 효율적인 학습을 가능하게 하여, 실질적으로 로봇의 자율성을 크게 향상시킵니다.



### InstanceCap: Improving Text-to-Video Generation via Instance-aware Structured Caption (https://arxiv.org/abs/2412.09283)
- **What's New**: 이번 논문에서는 텍스트-비디오 생성의 품질 향상을 위한 새로운 프레임워크인 InstanceCap을 제안합니다. 이것은 인스턴스 수준의 세밀한 비디오 캡션 생성을 최초로 가능하게 하며, 두 가지 주요 과제를 해결합니다: 캡션과 비디오 간의 높은 일관성과 정확한 내용 표현입니다. InstanceCap은 멀티모달 큰 언어 모델(MLLM)을 통해 모호한 정보 없이 정확한 비디오 설명을 생성하도록 설계되었습니다.

- **Technical Details**: InstanceCap은 글로벌 비디오를 로컬 인스턴스의 고유한 정보로 변환하기 위해 보조 모델 클러스터(AMC)를 활용합니다. 이러한 방식으로 캡션의 명확성을 높이고, 복잡한 텍스트로부터 구조화된 구문으로 변환하여 세분화된 비디오 설명을 생성합니다. 또한, 22K InstanceVid 데이터셋을 구성하여 고해상도 비디오 훈련에 사용합니다.

- **Performance Highlights**: 다양한 실험 결과에 따르면, InstanceCap은 기존 모델을 능가하며 캡션과 비디오 간의 높은 일관성을 유지합니다. 또한, 훈련 후 T2V 모델은 세부사항과 동작을 더욱 정확하게 추적할 수 있는 능력을 증명하였습니다. 이러한 결과는 InstanceCap의 효과성과 텍스트-비디오 생성에서의 진보된 가능성을 보여줍니다.



### Towards a Multimodal Large Language Model with Pixel-Level Insight for Biomedicin (https://arxiv.org/abs/2412.09278)
Comments:
          Accepted by AAAI2025

- **What's New**: 최근 다중모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 발전으로, 지능형 생의학 보조 도구 개발이 가능하게 되었습니다. 그러나 기존 생의학 MLLMs는 이미지 수준의 이해에 주로 초점을 맞추고, 텍스트 명령어로의 상호작용에 제한됨으로써 그 활용성을 제한하고 있습니다. 본 논문에서는 픽셀 수준의 이해가 가능한 새로운 End-to-End MLLM인 MedPLIB를 소개하며, 이는 추천 인식(vizual question answering, VQA), 다양한 픽셀 기반 프롬프트와 픽셀 수준의 그라운딩을 지원합니다.

- **Technical Details**: 우리는 새로운 Mixture-of-Experts(MoE) 다단계 훈련 전략을 제안하며, 이를 통해 시각-언어 전문가 모델과 픽셀 그라운딩 전문가 모델을 분리하여 훈련한 뒤, MoE를 사용하여 미세 조정합니다. 이 전략은 여러 작업의 학습을 효과적으로 조정하면서, 추론 시 단일 전문가 모델의 계산 비용 수준을 유지합니다. 또한, Medical Complex Vision Question Answering Dataset (MeCoVQA)를 소개하여 복잡한 의료 이미징 질문에 대한 8가지 모달리티를 포함합니다.

- **Performance Highlights**: 실험 결과, MedPLIB는 여러 의료 시각 언어 작업에서 최첨단 성능을 달성했습니다. 특히, 픽셀 그라운딩 작업의 제로샷 평가에서 MedPLIB는 mDice 메트릭 기준으로 가장 작은 및 큰 모델에 대해 각각 19.7, 15.6의 차이로 우위를 점하고 있습니다. 또한, 연구 커뮤니티를 위해 데이터, 코드 및 모델 체크포인트를 공개할 예정입니다.



### Towards Understanding the Robustness of LLM-based Evaluations under Perturbations (https://arxiv.org/abs/2412.09269)
Comments:
          Accepted at ICON 2024

- **What's New**: 본 논문에서는 BLEU 및 ROUGE와 같은 기존 평가 메트릭이 생성된 텍스트의 미세한 품질을 포착하는 데 한계가 있음을 지적하고, Google Gemini 1과 같은 Large Language Models (LLMs)를 비표준화된 메트릭의 자동 평가자로 활용할 가능성을 탐구합니다. 특히 본 연구는 LLM의 품질 평가 기능을 조사하며, 인간 평가자와의 일치를 비교하고 약화된 입력에 대한 평가의 강인성도 살펴봅니다. 이는 NLG 태스크의 평가 방법 개선에 기여할 수 있는 새로운 통찰을 제공합니다.

- **Technical Details**: NLG에서 BLEU, ROUGE와 같은 전통적인 메트릭은 참고 자료와의 토큰 겹침에 의존하므로, 다중 유효 출력이 존재하는 abstractive summarization 및 dialog 평가에 적합하지 않습니다. 본 연구는 Google Gemini를 기반으로 서로 다른 프롬프트 전략을 적용하여 요약 및 대화 평가에서 LLM의 성능을 평가하는 방법론을 제안합니다. SummEval 및 USR 데이터셋을 사용하여 LLM이 산출한 점수와 정당성을 검토하고, 각 메트릭에 대한 점수를 생성하는 네 가지 프롬프트 전략을 검토합니다.

- **Performance Highlights**: 실험 결과, LLM 평가자와 인간 평가자 간의 일치는 제한적이며 LLM의 강인성이 떨어진다는 점을 발견했습니다. NLG 태스크에 필요한 주관적 메트릭에서 LLM을 신뢰할 수 있는 평가자로 사용하기 위해서는 상당한 개선이 요구됩니다. 연구 결과는 LLM 기반 평가자가 인간 평가자에 비해 더 일관되고 효율적인 평가를 제공할 수 있는 잠재력이 있으나, 실질적인 응용을 위해서는 더욱 발전이 필요함을 시사합니다.



### First Train to Generate, then Generate to Train: UnitedSynT5 for Few-Shot NLI (https://arxiv.org/abs/2412.09263)
Comments:
          14 pages

- **What's New**: 이번 연구에서는 자연어 추론(NLI) 모델을 개선하기 위해 합성 데이터 증강(synthetic data augmentation) 방식을 도입한 UnitedSynT5라는 새로운 접근법을 제안합니다. 기존의 Entailment Few-Shot Learning (EFL) 모델을 기반으로 하여 T5 기반 생성기를 활용해 추가적인 전제-가설 쌍을 생성했습니다. 이렇게 생성된 데이터는 엄격하게 정리되어 학습 데이터에 통합되며, 이를 통해 모델의 일반화 능력을 향상시킬 수 있습니다.

- **Technical Details**: UnitedSynT5는 EFL 프레임워크를 활용하여, T5 생성기를 통해 합성된 데이터 세트를 사용합니다. 전제-가설 쌍은 의미적 일관성을 유지하며 데이터 세트의 다양성과 복잡성을增强시키기 위해 필터링 과정을 거칩니다. 이 기술은 GTR-T5-XL 모델에서 확장된 데이터 세트로 훈련되어, NLI 모델의 성능을 획기적으로 향상시키는 결과를 가져왔습니다.

- **Performance Highlights**: 이 연구 결과는 SNLI 데이터 세트에서 94.7%의 새로운 기준을 달성했으며, E-SNLI와 MultiNLI 데이터 세트에서도 각각 94.01%와 92.57%의 정확도를 기록했습니다. 이러한 성과는 합성 데이터 증강의 잠재력을 보여주며, 향후 자연어 이해(natural language understanding) 작업의 발전에 기여할 것으로 기대됩니다.



### VLMs meet UDA: Boosting Transferability of Open Vocabulary Segmentation with Unsupervised Domain Adaptation (https://arxiv.org/abs/2412.09240)
- **What's New**: 본 논문은 다양한 도메인에서 세분화(segmentation) 정확도를 향상시키기 위해 Vision-Language reasoning과 Unsupervised Domain Adaptation (UDA) 기법을 통합하는 새로운 접근 방식을 제안합니다. 기존의 Vision-Language Models (VLMs)와 합성 데이터 기반 방법의 한계를 극복하기 위해, 제안하는 Foundational-Retaining Open Vocabulary Semantic Segmentation (FROVSS) 프레임워크는 다중 규모의 컨텍스트 데이터를 활용하여 세분화 능력을 개선하고 있습니다. 또한 UDA와 함께 이 기술을 적용하여 새로운 카테고리에 대해 더 효과적으로 적응할 수 있는 모델을 개발했습니다.

- **Technical Details**: 제안된 FROVSS 프레임워크는 VLM의 세부 정보를 보존하면서 안정적인 교육을 위한 distillation 기법과 cross-domain mixed sampling을 채택합니다. 이를 통해 데이터 요구사항을 줄이고, VLM의 글로벌 지식을 유지하면서 미세 조정(fine-tuning) 과정에서의 재앙적 망각(catastrophic forgetting)을 방지합니다. 또한, 텍스트 관계 개선을 위해 대규모 언어 모델(LLMs)을 활용한 개념 수준의 프롬프트 증강 전략을 적용하여, 다양한 데이터셋 간의 카테고리 인식 개선을 목표로 하고 있습니다.

- **Performance Highlights**: FROVSS 및 UDA-FROVSS를 통해 학습된 모델은 여러 세그멘테이션 데이터셋에서 우수한 성능을 나타내며, 각 벤치마크에서 개선된 결과를 보여주고 있습니다. 특히 Cityscapes 데이터셋에서 22.1% mIoU 향상이 이루어졌으며, Synthia-to-Cityscapes 설정에서 새로운 UDA 기준을 수립하여 이전의 최신 방법에 비해 8% 이상 개선되었습니다. 이러한 결과는 다양한 데이터셋에 걸쳐 FROVSS의 성능과 일반화 능력이 크게 향상되었음을 시사합니다.



### Foundation Models and Adaptive Feature Selection: A Synergistic Approach to Video Question Answering (https://arxiv.org/abs/2412.09230)
- **What's New**: 이번 논문에서는 비디오 질문-답변(Video QA)의 새로운 접근 방식을 제안합니다. LGQAVE(Local-Global Question Aware Video Embedding)라는 모델을 통해 질문과 비디오 프레임, 그리고 의미적 객체 수준 추상화를 효과적으로 통합할 수 있는 방안을 모색합니다. 기존의 방법이 프레임 샘플링에서 한계를 보인 반면, 이 방법은 교차 주의(cross-attention) 메커니즘을 사용하여 질문과 가장 관련성이 높은 프레임을 정확히 식별합니다.

- **Technical Details**: LGQAVE는 질문을 인식하여 비디오 프레임을 선택하기 위해 학습 가능한 교차 주의 모듈을 활용합니다. 이 모듈은 질문의 의미와 가장 일치하는 프레임을 효율적으로 고립시켜, 선택된 프레임의 객체 간 상호작용을 모델링하기 위해 spatial 및 temporal 그래프를 생성합니다. 이 그래프는 Dynamic Graph Transformer(또는 DGT-Q 모델이라고 부르며)로 입력되어 프레임 특성을 정제하는 데 쓰입니다.

- **Performance Highlights**: LGQAVE는 NextVQA, TGIF-QA, Causal VidQA, MSRVTT-QA와 같은 여러 벤치마크에서 기존 모델에 비해 유의미한 성과를 보였습니다. 특히, 고차원의 다중 선택 및 개방형 질문에 대한 정확성을 크게 향상시켜, 비디오 질문-답변의 범위를 넓히는 데 기여하고 있습니다. 이러한 성과는 질문의 의미와 관련된 비디오 프레임에 대한 정량적 해석이 가능하다는 점에서 주목할 만합니다.



### CSSDH: An Ontology for Social Determinants of Health to Operational Continuity of Care Data Interoperability (https://arxiv.org/abs/2412.09223)
Comments:
          6 pages, 3 figures, conference-The 25th International Conference on Intelligent Data Engineering and Automated Learning

- **What's New**: 디지털 플랫폼의 발전으로 인해 기술 기반의 가정 의료 솔루션에 대한 의존도가 높아지고 있으며, 개인들이 자신의 건강을 모니터링하고 필요에 따라 건강 관리 전문가와 정보를 공유할 수 있게 되었습니다. 그러나 효율적인 치료 계획 관리 시스템을 구축하기 위해서는 병원 요약과 전자 건강 기록(EHR) 분석 이상의 노력이 필요합니다. 개인 사용자 요구와 건강의 사회적 결정 요인들이 고려되어야 합니다.

- **Technical Details**: 이 논문에서는 다양한 보건 의료 시스템과 애플리케이션 간의 상호운용성(interoperability)을 확립하는 것이 중요하다고 강조합니다. European Interoperability Framework (EIF)는 환자 중심의 건강 데이터 접근 및 통제의 필요성을 강조하고 있으며, 이를 바탕으로 ISO/DIS 13940:2024 ContSys와 WHO Social Determinants of Health를 결합한 통합적인 온톨로지 모델인 Common Semantic Data Model for Social Determinants of Health (CSSDH)를 제안합니다. CSSDH는 지속적인 치료 네트워크(Continuity of Care Network) 내에서 상호운용성을 달성하는 것을 목표로 합니다.

- **Performance Highlights**: 보건 의료 관리 시스템에서의 다양한 스키마 복잡성과 용어 다양성은 큰 도전 과제가 됩니다. 개인 건강 기록과 전자 건강 기록 간의 효율적인 데이터 흐름을 위해 CSSDH 모델은 이러한 문제를 해결하는 데 기여할 것으로 기대됩니다. 또한, 본 연구는 환자 중심의 데이터 관리 체계를 통해 보다 효율적이고 일관된 건강 관리 솔루션을 제공할 수 있는 가능성을 보여줍니다.



### When Text Embedding Meets Large Language Model: A Comprehensive Survey (https://arxiv.org/abs/2412.09165)
Comments:
          Work in progress

- **What's New**: 이 논문은 자연어 처리(NLP)에서의 텍스트 임베딩의 역할을 심층적으로 조사하며, LLM(대형 언어 모델)이 텍스트 임베딩 기법과 어떻게 상호 작용하는지를 세 가지 주요 주제로 나누어 설명합니다. 특히 LLM이 기존의 텍스트 임베딩 방법을 보강하거나 자체적으로 텍스트 임베딩 생성에 어떻게 활용되는지를 다루고 있습니다. 이 연구는 다양한 연구 및 응용 분야의 기여를 조직적으로 정리하고, LLM과 PLM(사전 학습 언어 모델) 시대의 남아 있는 도전 과제를 강조합니다.

- **Technical Details**: 텍스트 임베딩 학습은 자연어 처리를 위한 기초 작업으로, 주어진 텍스트에서 유용한 특성을 추출하는 것을 목표로 합니다. LLM은 탐색, 텍스트 분류, 기계 번역 등 다양한 다운스트림 작업에서 탁월한 일반화 및 전이 능력을 보여줍니다. 본 논문에서는 LLM이 데이터 주석 및 모델 기초로서 높은 질의 텍스트 표현을 생성하는 두 가지 방법으로 기존 임베딩 학습 환경을 변화시켰음을 강조합니다.

- **Performance Highlights**: 텍스트 임베딩 분야에서 최근 LLM의 등장은 진화의 새로운 방향을 제시하였으며, 특히 정보 추출, 유사성 측정 등 여러 분야에서 장기적인 기대 효과를 생성하고 있습니다. 다양한 전통적 및 새롭게 등장한 다운스트림 작업들에 대해 LLM이 어떻게 기여할 수 있음을 보여주며, 기존 방법들의 한계와 LLM으로 인해 새롭게 발생한 도전 과제를 함께 다루고 있습니다. 이 연구는 앞으로의 텍스트 임베딩 발전 방향에 대해 이론적 그리고 실천적 기회를 탐구하며 지속적인 발전을 장려합니다.



### Enhancing Modality Representation and Alignment for Multimodal Cold-start Active Learning (https://arxiv.org/abs/2412.09126)
Comments:
          11 pages, ACMMM Asia 2024, Oral Presentation

- **What's New**: 본 연구는 Multi-Modal Cold-Start Active Learning (MMCSAL)이라는 새로운 접근법을 제안하여 개선된 데이터 선택 과정을 통해 멀티모달 데이터의 라벨링 비용을 줄이는 것을 목표로 합니다. 두 단계로 구성된 이 방법은 각 모달리티 간의 간극을 해소하기 위한 uni-modal prototypes를 도입하고, cross-modal alignment를 강화하여 멀티모달 데이터 쌍의 품질을 향상시킵니다. 이러한 해결책은 기존 연구들에서 간과된 멀티모달 데이터의 특징을 활용합니다.

- **Technical Details**: 제안된 모델 MMCSAL은 두 단계로 이루어져 있습니다. 첫 번째 단계에서는 self-supervised learning (SSL) 기법을 통해 멀티모달 데이터의 표현을 생성하고, 모달리티 간 거리 계산에서 발생하는 'modality gap'을 줄이기 위해 uni-modal prototypes를 활용합니다. 두 번째 단계에서는 cross-modal alignment 강화를 위해 정규화 항을 도입하여, 선택된 데이터 쌍의 일치성을 높이고, 이는 Downstream tasks의 성능을 개선하는 데 기여합니다.

- **Performance Highlights**: 실험 결과, MMCSAL은 Food101, KineticsSound 및 VGGSound와 같은 세 가지 멀티모달 분류 데이터셋에서 더 나은 멀티모달 데이터 쌍 선택 능력을 보여주었습니다. 이 방법은 cold-start 멀티모달 active learning의 성능을 효과적으로 개선하며, 일반적으로 사용되는 고전적인 AL 접근법에서 발생하는 문제를 해결합니다. MMCSAL의 도입으로 초기 라벨링이 부족한 상황에서도 효율적인 데이터 선택이 가능해졌습니다.



### PolyIPA -- Multilingual Phoneme-to-Grapheme Conversion Mod (https://arxiv.org/abs/2412.09102)
- **What's New**: 이 논문은 다국어 음소(phoneme)에서 그래프음(grapheme)으로 변환하는 새로운 모델인 PolyIPA를 소개합니다. 이 모델은 다국어 이름의 음역(transliteration), 이름학 연구(onamastic research), 정보 검색을 위한 목적으로 설계되었습니다. PolyIPA는 데이터 증가(data augmentation)를 위해 식별된 두 개의 보조 모델, IPA2vec와 similarIPA를 활용하며, 여러 언어와 문자 시스템을 테스트하여 매우 낮은 Character Error Rate와 높은 BLEU 점수를 달성했습니다.

- **Technical Details**: PolyIPA의 구현은 음소-그래프음(P2G) 변환의 역방향 문제에 초점을 맞추고 있습니다. 이 모델은 기존의 G2P 데이터셋과 모델을 활용하여 두 개의 보조 모델을 개발하였으며, IPA2vec는 언어 간의 유사 발음을 찾고, similarIPA는 다양한 음소 기호 표기에 대처합니다. 최종 데이터셋은 약 1964만의 고유한 언어-그래프음-음소 쌍으로 구성되어 있으며, 기계 학습(neural network)을 통해 이 데이터셋이 강화되었습니다.

- **Performance Highlights**: PolyIPA 모델은 테스트 세트에서 평균 Character Error Rate 0.055와 BLEU 점수 0.914를 달성하였으며, 특히 얕은 철자법(shallow orthography)을 가진 언어에서 뛰어난 성능을 보였습니다. 추가적으로, 빔 탐색(beam search)을 구현하여 상위 3개의 후보를 활용할 때, 효과적인 오류율을 52.7% 감소시킬 수 있었습니다. 이러한 결과는 다국어 음역 및 정보 검색 응용 프로그램에서의 PolyIPA 모델의 효과를 보여줍니다.



### Filter-then-Generate: Large Language Models with Structure-Text Adapter for Knowledge Graph Completion (https://arxiv.org/abs/2412.09094)
Comments:
          COLING 2025 Main Conference

- **What's New**: 이번 연구에서는 Knowledge Graph Completion (KGC) 문제에 대한 새로운 접근 방식으로 instruction-tuning 기반의 FtG 방법을 제안합니다. 기존의 대형 언어 모델(LLMs)이 KGC 작업에서 낮은 성능을 보인 문제를 해결하기 위해, 필터-생성(filter-then-generate) 패러다임을 도입하여 후보 엔티티를 효과적으로 축소합니다. 또한, 그래프 구조 정보를 LLMs에 통합하기 위해 구조 인식 프롬프트를 설계했습니다.

- **Technical Details**: FtG 방법은 KGC 작업을 다중 선택 질문 형식으로 변환하여 LLMs가 타겟 엔티티를 생성하도록 유도하는 방식입니다. 이 방법은 먼저 전통적인 KGC 기법을 이용하여 불가능한 엔티티 후보를 제거한 후, 남은 상위 k 개의 후보로부터 최종 엔티티를 선택하도록 합니다. 그 밖에도, 구조와 텍스트 정보를 맵핑하는 가벼운 구조-텍스트 어댑터를 도입하여 LLMs가 KG의 복잡한 구조를 이해하도록 돕습니다.

- **Performance Highlights**: 실험 결과, 제안된 FtG 방법이 기존의 최첨단 KGC 방법들과 비교하여 상당한 성능 향상을 보여주었습니다. 특히, 여러 표준 벤치마크에서 KGC 작업 수행 시, FtG가 기존의 구조 기반 방법보다 더 나은 성과를 기록했습니다. 이러한 결과는 FtG 방법이 기존 KGC 접근 방식을 개선하는 데 효과적이며, 플러그 앤 플레이(plug-and-play) 방식으로도 활용 가능함을 시사합니다.



### Understanding Opportunities and Risks of Synthetic Relationships: Leveraging the Power of Longitudinal Research with Customised AI Tools (https://arxiv.org/abs/2412.09086)
Comments:
          This is a "Position paper accepted for CONVERSATIONS 2024 - the 8th International Workshop on Chatbots and Human-Centred AI, hosted by CERTH, Thessaloniki, Greece, December 4-5, 2024." The original publication is available on the workshop website: this https URL . This document is identical to the original and is mainly available here for accessibility and discoverability

- **What's New**: 이 논문은 맞춤형 AI 도구를 이용한 장기적 행동 연구의 장점에 대해 논의합니다. 특히, 인간과 AI 도구 간의 지속적인 상호작용인 합성 관계(synthetic relationships)에 초점을 맞추고 있습니다. 합성 관계는 AI 도구가 인간의 생각, 감정, 행동에 영향을 미치는 새로운 형태의 관계로 정의됩니다.

- **Technical Details**: 연구는 기존의 발견을 보완하는 방법론적 접근법을 제안합니다. 여기서는 스스로 조립된 AI 에이전트(self-assembled AI agents)를 이용한 장기 연구 설계(longitudinal research designs)를 통해 상세한 행동 데이터(behavioural data)와 자기 보고 데이터(self-reported data)의 통합을 가능하게 하는 방안을 모색합니다.

- **Performance Highlights**: 합성 관계는 건강, 교육 및 직장에서의 기회를 향상시킬 수 있는 잠재력을 지니고 있지만, 미세한 조작(manipulation) 및 개인 정보 및 자율성에 대한 우려를 동반합니다. 이 연구는 이러한 기회를 활용하고 위험을 완화하기 위한 구체적인 방법론을 제공하여 앞으로의 연구 방향을 제시합니다.



### Forest-of-Thought: Scaling Test-Time Compute for Enhancing LLM Reasoning (https://arxiv.org/abs/2412.09078)
Comments:
          Preprint

- **What's New**: 이 논문에서는 복잡한 논리 문제를 해결하기 위한 Forest-of-Thought (FoT)라는 새로운 추론 프레임워크를 제안합니다. FoT는 여러 개의 추론 트리를 통합하여 집단적 의사 결정을 활용하며, 스파스 활성화 전략을 통해 가장 관련성이 높은 경로를 선택함으로써 효율성과 정확성을 향상시킵니다. 또한 실시간 오류 수정 및 과거 학습을 통해 동적 자기 수정 전략을 도입하여 복잡한 문제 해결에서의 성능을 크게 향상시킵니다.

- **Technical Details**: FoT 프레임워크는 여러 개의 추론 트리를 활용하여 모델의 추론 능력을 확장시키며, 각 트리의 가장 관련성 높은 경로를 선택하기 위해 스파스 활성화 전략을 채택합니다. 이와 함께 합의 기반의 의사 결정 전략을 포함하여, 모델이 필요할 때만 추론 과정을 계속 진행하도록 최적화합니다. 이러한 방법은 LLM이 복잡한 작업을 보다 정밀하고 효율적으로 해결할 수 있도록 지원합니다.

- **Performance Highlights**: 실험 결과는 FoT 프레임워크가 제공하는 다양한 전략과 결합하여 LLM의 추론 성능을 크게 향상시키는 데 기여함을 보여줍니다. 연구에 따르면 FoT는 특히 복잡한 과제를 해결하는 데 있어 더욱 뛰어난 정확성과 효율성을 제공합니다. 이를 통해 LLM은 일반적인 문제 해결뿐 아니라 고급 추론 요구사항에도 효과적으로 대응할 수 있는 가능성을 보여줍니다.



### EmbedGenius: Towards Automated Software Development for Generic Embedded IoT Systems (https://arxiv.org/abs/2412.09058)
- **What's New**: 이 논문에서는 EmbedGenius라는 완전 자동화된 소프트웨어 개발 플랫폼을 소개합니다. 이 플랫폼은 일반 목적의 임베디드 IoT 시스템 개발을 위한 것이며, 대규모 언어 모델(LLMs)의 추론 능력과 임베디드 시스템 전문 지식을 활용하여 하드웨어 통합 개발 프로세스를 자동화합니다. EmbedGenius는 코드 생성의 정확도를 95.7%로 달성하고, 작업 완료율은 86.5%에 달하는 성과를 보여줍니다. 이는 기존의 사람 개입 방식보다 15.6%에서 37.7%, 25.5%에서 53.4% 더 뛰어난 성과입니다.

- **Technical Details**: EmbedGenius는 여러 가지 혁신적인 방법을 통해 기존의 임베디드 시스템 개발 과정을 간소화합니다. 이 시스템은 하드웨어 종속성을 해결하기 위한 컴포넌트 인식 라이브러리 해상도 방법, LLMs에 유틸리티 도메인 지식을 주입하는 라이브러리 지식 생성 방법, 성공적인 배포를 보장하는 자동 프로그래밍 방법을 포함합니다. 이는 개발자가 수동으로 처리해야 했던 복잡한 개발 단계를 자동화하여, 시간과 노력을 절감할 수 있게 합니다.

- **Performance Highlights**: EmbedGenius는 71개의 모듈과 4개의 주요 임베디드 개발 플랫폼에서 350개 이상의 IoT 작업에 대한 성과를 평가했습니다. 실험 결과, 평균 코드 생성 정확도는 95.7%에 달하며, 평균 작업 완료율은 86.5%입니다. 또한, 환경 모니터링 시스템과 원격 제어 시스템 개발에 대한 사례 연구를 통해 EmbedGenius의 실용적 이점을 입증했습니다. 환경 모니터링 시스템은 2.6분, 원격 제어 시스템은 3.1분 만에 개발할 수 있었습니다.



### Multi-Task Learning with LLMs for Implicit Sentiment Analysis: Data-level and Task-level Automatic Weight Learning (https://arxiv.org/abs/2412.09046)
Comments:
          11 pages, 6 figures, and 6 tables

- **What's New**: 이번 연구에서는 새로운 다중 작업 학습(Multi-task Learning, MTL) 프레임워크인 MT-ISA를 도입하여 암묵적 감정 분석(Implicit Sentiment Analysis, ISA)을 개선합니다. MT-ISA는 대규모 언어 모델(Large Language Models, LLMs)의 생성 및 추론 기능을 활용하여 데이터 및 작업 수준의 불확실성을 처리하는 자동 MTL 접근법을 적용합니다. 이를 통해 모델들이 보다 효과적으로 감정을 인식하고 분석할 수 있도록 지원합니다.

- **Technical Details**: MT-ISA는 생성적 LLM을 사용하여 보조 작업을 구성하고, 자동 가중치 학습(Automatic Weight Learning, AWL)을 통해 데이터와 작업 간의 관계를 동적으로 학습합니다. 데이터 수준의 AWL은 데이터의 신뢰성을 높이고, 작업 수준의 AWL은 모델의 추론 능력에 따라 세밀한 가중치를 적응적으로 학습하도록 설정됩니다. 이러한 과정은 모델의 성능을 높이는 효과적인 방법으로 입증되었습니다.

- **Performance Highlights**: 대규모 실험을 통해 MT-ISA는 다양한 모델 크기에서 주 감정 예측과 보조 작업 간의 최적의 균형을 이룰 수 있음을 보여주었습니다. 제안된 방법은 ISA 분야에서 최신 기술(state-of-the-art) 결과를 달성하며, 데이터 수준 및 작업 수준의 AWL 전략을 통한 성능 개선을 확인하였습니다. 이 연구의 결과는 LLM의 활용 가능성을 더욱 확장시키는 중요한 시사점을 제공합니다.



### Motif Guided Graph Transformer with Combinatorial Skeleton Prototype Learning for Skeleton-Based Person Re-Identification (https://arxiv.org/abs/2412.09044)
Comments:
          Accepted by AAAI 2025. Codes are available at this https URL

- **What's New**: 이 논문에서는 인물 재식별(person re-ID)을 위한 3D 스켈레톤 데이터 기반의 접근 방식을 제안합니다. 기존의 방법들이 모든 관절 간의 가상의 운동 연관성을 가정했던 반면, 본 연구에서는 주요 신체 구조와 보행 패턴을 중점적으로 분석하여 효과적인 스켈레톤 표현을 학습하는 새로운 방식을 제시합니다. 특히, 모티프 유도 그래프 변환기(Motif guided graph transformer, MGT)와 조합형 스켈레톤 프로토타입 학습(Combinatorial skeleton prototype learning, CSP) 방법을 통해 스켈레톤 관찰의 심화 학습을 목표로 합니다.

- **Technical Details**: 이 연구의 핵심 기술은 모티프 유도 그래프 변환기(MGT)와 조합형 스켈레톤 프로토타입 학습(CSP)입니다. MGT는 계층적 구조 모티프와 보행 협업 모티프를 통합하여 신체 관절 간의 관계 학습을 강화합니다. CSP는 랜덤한 공간-시간 조합을 활용하여 다양한 서브 스켈레톤 및 서브 트랙렛 표현을 생성하며, 이를 통해 각 개체의 대표적인 스켈레톤 특징을 학습합니다.

- **Performance Highlights**: 실험 결과, MoCos는 기존의 최첨단 모델에 비해 우수한 성능을 보였습니다. 다섯 개의 공개 데이터셋에서의 평가를 통해, MoCos는 RGB로 추정된 스켈레톤 및 다양한 그래프 모델링, 비지도 학습 시나리오에서 효과적으로 적용될 수 있음을 입증하였습니다. 이로 인해 MoCos는 스켈레톤 기반 인물 재식별 분야에서 유망한 접근으로 자리매김할 것으로 기대됩니다.



### Speech-Forensics: Towards Comprehensive Synthetic Speech Dataset Establishment and Analysis (https://arxiv.org/abs/2412.09032)
- **What's New**: 이 논문에서는 신뢰할 수 있는 스피치 분석을 위해 Speech-Forensics 데이터셋을 제안합니다. 이 데이터셋은 진짜 목소리, 합성된 목소리, 그리고 부분적으로 변조된 목소리 샘플을 포함하여 다양한 합성 알고리즘에 의한 다수의 세그먼트를 광범위하게 커버합니다. 이는 기존 데이터셋의 제한을 극복하고 포괄적인 연구를 지원하기 위해 설계되었습니다.

- **Technical Details**: 본 연구에서는 TEST라고 불리는 임시 스피치 로컬라이제이션 네트워크를 소개합니다. 이 네트워크는 진위 감지(authenticity detection), 다수의 가짜 세그먼트 로컬라이제이션(localization), 합성 알고리즘 인식(recognition) 기능을 복잡한 후처리 없이 동시에 수행하도록 설계되었습니다. TEST는 LSTM(Long Short-Term Memory)과 Transformer를 효과적으로 통합하여 시계열 스피치 표현을 강화하고, 다중 스케일 피라미드 특징에서 밀집 예측을 통해 합성된 범위를 추정합니다.

- **Performance Highlights**: 모델은 발화 수준에서 평균 mAP(mean Average Precision) 83.55%와 EER(Equal Error Rate) 5.25%를 달성했습니다. 세그먼트 수준에서는 EER 1.07%와 92.19% F1 점수를 기록하여, 본 모델의 합성 스피치에 대한 강력한 분석 능력을 입증합니다. 이러한 결과는 향후 연구 및 실제 응용 분야에서 매우 유망한 가능성을 보여줍니다.



### RingFormer: A Ring-Enhanced Graph Transformer for Organic Solar Cell Property Prediction (https://arxiv.org/abs/2412.09030)
Comments:
          12 pages, 4 figures. This is the extended version of the paper accepted at AAAI 2025, which includes all technical appendices and additional experimental details

- **What's New**: 본 논문에서는 Organic Solar Cells (OSCs)의 고유한 링 구조를 캡처할 수 있도록 개발된 새로운 그래프 트랜스포머 프레임워크인 RingFormer를 소개합니다. 기존의 방법들이 OSC 분자의 링 시스템을 효과적으로 모델링하지 못하는 한계를 극복하기 위해, RingFormer는 원자 수준, 링 수준 및 각각의 관계를 고려한 계층적인 그래프 구조를 구축합니다. 이를 통해 높은 정밀도의 OSC 속성 예측이 가능해졌습니다.

- **Technical Details**: RingFormer는 세 가지 수준의 계층적인 그래프를 활용하여 OSC 분자의 구조를 효과적으로 모델링합니다. 원자 레벨 그래프는 원자 간 결합을 설명하고, 링 레벨 그래프는 링 구조와 그 상호 연결을 강조하며, inter-level 그래프는 링과 원자의 관계를 모델링합니다. 이 구조는 메시지 패싱(local message passing) 및 글로벌 어텐션(global attention) 메커니즘을 결합하여 각 수준의 구조적 패턴을 포착하고, 최종적으로 OSC 속성의 정밀한 예측을 위한 표현을 학습합니다.

- **Performance Highlights**: RingFormer의 성능은 다섯 개의 OSC 분자 데이터 셋에 대한 광범위한 실험을 통해 평가되었습니다. 특히, CEPDB 데이터셋에서는 기존의 최근 경쟁자에 비해 22.77%의 상대적 향상을 달성하며, 이는 RingFormer가 기존의 방법들보다 지속적으로 우수하다는 것을 보여줍니다. 이러한 성과는 OSC 속성 예측의 효율화를 통해 유망한 에너지 생산 기술로서 OSC의 발전에 기여할 것으로 기대됩니다.



### Shiksha: A Technical Domain focused Translation Dataset and Model for Indian Languages (https://arxiv.org/abs/2412.09025)
- **What's New**: 이 논문에서는 과학, 기술 및 교육 분야에 대한 번역 데이터셋의 부족을 해결하기 위해 8개 인도 언어의 영어-인디크 및 인디크-인디크 고품질 번역 쌍이 280만 개 이상 포함된 다국어 병렬 말뭉치를 생성했습니다. 이 데이터셋은 NPTEL 비디오 강의의 인간 번역 전사(Transcriptions)를 비텍스트 마이닝(Bitext Mining)하여 얻은 것입니다. 그들의 데이터셋을 사용하여 NMT 모델을 세밀 조정(Finetune)하고 평가하며, 특정 도메인 작업에서 공개된 다른 모델들을 초월했습니다.

- **Technical Details**: 논문은 8개 인도 언어로 2.8백만 개의 문장 쌍을 포함하는 병렬 텍스트 리소스를 만드는 과정을 설명합니다. NPTEL의 비디오 전사 데이터를 사용하여 문장을 NLP 라이브러리(nltk 및 indic-nlp)로 필터링했습니다. 그 후 SentAlign과 LABSE를 이용하여 고품질의 문장 쌍을 추출하며, n-m 문장 쌍 매칭을 활용하여 데이터를 더욱 세밀하게 분석했습니다.

- **Performance Highlights**: 모델 평가에서, 우리는 FLORES+ 벤치마크에서 평균 2 BLEU 이상의 성능 향상을 보여주며 일반 도메인 번역 작업에 대한 일반화 가능성을 증명했습니다. 이번 연구를 통해 우리는 인도 언어의 기술 도메인 번역 작업에서 NMT 모델의 성능이 크게 개선될 수 있음을 보여주었습니다. 또한, 생성된 데이터셋과 모델은 널리 배포되어 인도 학생들에게 교육적 혜택을 제공할 것입니다.



### What Makes Cryptic Crosswords Challenging for LLMs? (https://arxiv.org/abs/2412.09012)
Comments:
          COLING 2025

- **What's New**: 이 논문에서는 현대의 NLP 모델, 특히 대규모 언어 모델(LLMs)이 난해한 크립틱 크로스워드를 해결하는 데 어려움을 겪는 이유를 탐구합니다. 연구 결과, Gemma2, LLaMA3 및 ChatGPT와 같은 LLM의 성능이 여전히 인간의 성과에 비해 상당히 낮음을 보여주었으며, 이를 통해 이들 모델의 해석 가능성을 분석하고자 합니다. 또한, 새로운 데이터셋과 코드를 공개하여 연구의 재현성을 높입니다.

- **Technical Details**: 크립틱 크로스워드는 일반적인 정의나 동의어 대신 단어 놀이와 수수께끼를 포함하는 퍼즐입니다. 이 논문에서는 모델의 추론 능력을 평가하기 위해 클루의 정의 부분 추출, 단어 놀이 유형 식별, 모델의 내부 논리 설명을 통한 세 가지 보조 작업을 수행합니다. 또한, 데이터셋은 다양한 예제를 포함하고 있으며, 특정 단어 놀이 유형을 레이블링한 소규모 새 데이터셋을 추가하여 모델 학습을 지원합니다.

- **Performance Highlights**: 이 연구는 기존 LLM들이 크립틱 크로스워드 문제에서 인간 전문가에 비해 여전히 낮은 성능을 나타낸다는 기존 문헌을 확장합니다. 최근 연구에서는 LLM을 사용한 몇 가지 프롬프트 기술을 적용했지만, 그 성과는 20% 수준에 그쳤습니다. 이로 인해 모델의 문제 해결 능력을 향상시키기 위한 추가적인 연구가 필요하다는 결론을 내리고 있습니다.



### The AI Interface: Designing for the Ideal Machine-Human Experience (Editorial) (https://arxiv.org/abs/2412.09000)
Comments:
          8 pages

- **What's New**: 본 논문에서는 일상 생활에 점점 더 깊이 통합되고 있는 인공지능(AI)의 맥락에서 직관적이며 신뢰할 수 있고 감정적으로 공감할 수 있는 AI-인간 인터페이스 디자인의 중요성을 다룹니다. "AI 경험 디자인"의 심리학을 탐색하는 특집호를 소개하며, 인간과 기계 간의 원활한 협업을 어떻게 도울 수 있는지를 중심으로 논의합니다.

- **Technical Details**: 논문은 헬스케어, 소비자 기술, 직장 역학 및 문화 분야 등 다양한 분야의 통찰을 기반으로 하여 인간-AI 상호작용에서 신뢰, 투명성 및 감정적 민감성의 복잡성을 조명합니다. 핵심 주제로는 사용자 인식 및 기대와 일치하는 AI 시스템 디자인, 투명성과 신뢰를 통해 저항 극복, 사용자 불안을 줄이기 위한 AI 능력의 프레이밍 등이 포함됩니다.

- **Performance Highlights**: 여덟 개의 다양한 연구를 종합하여 AI 인터페이스가 효율성과 공감을 균형 있게 조화시킬 필요성을 강조합니다. 기능적 차원과 감정적 차원 모두를 다루는 인간 중심 디자인을 통해 AI 시스템이 인간의 삶을 개선할 수 있도록 실천 가능한 프레임워크의 필요성을 제기합니다.



### Is Contrastive Distillation Enough for Learning Comprehensive 3D Representations? (https://arxiv.org/abs/2412.08973)
Comments:
          Under review

- **What's New**: 본 논문에서는 현재의 contrastive distillation 방법의 한계를 이론적으로 분석하고, 이러한 문제를 해결하기 위해 CMCR이라는 새로운 프레임워크를 제안합니다. 기존 방법들이 modality-specific features를 미흡하게 다루고 있음을 지적하며, 본 방법은 modality 공유와 특정 기능을 통합하는 데 초점을 맞추고 있습니다. 특히 masked image modeling과 occupancy estimation 작업을 도입하여 더 나은 학습을 지원합니다.

- **Technical Details**: CMCR 프레임워크는 modality-specific 및 modality-shared features를 동시에 학습할 수 있도록 설계되었습니다. 다중 모달 통합 코드북을 통해 서로 다른 모달리티 간의 효과적인 임베딩 공간을 형성하고, geometry-enhanced masked image modeling을 통해 3D 표현 학습을 향상시킵니다. 이러한 접근을 통해, 다양한 실제 과제에서 기존 방법들을 일관되게 초월하는 성능을 입증합니다.

- **Performance Highlights**: 실험 결과, CMCR 프레임워크는 3D semantic segmentation, object detection 및 panoptic segmentation 등의 다운스트림 작업에서 뛰어난 성능을 보였습니다. 기존의 self-supervised learning 기법들과 비교하여 높은 적응성과 성능을 발휘하여, 3D 표현 학습에서의 한계를 효과적으로 극복합니다. 코드 및 실험 데이터는 논문에서 제공되며, 이를 통해 연구 결과를 보다 쉽게 검증할 수 있습니다.



### RuleArena: A Benchmark for Rule-Guided Reasoning with LLMs in Real-World Scenarios (https://arxiv.org/abs/2412.08972)
Comments:
          Data and Codes are available at this https URL

- **What's New**: 이번 논문에서는 RuleArena라는 새로운 벤치마크를 소개합니다. 이 벤치마크는 LLMs (Large Language Models)가 복잡한 실제 규칙을 따르는 능력을 평가하기 위해 설계되었습니다. 항공사 수하물 요금, NBA 거래, 세금 규정 등 세 가지 실제 도메인을 다루며, LLM의 장기적 이해, 논리적 추론 및 수학적 계산 능력을 평가합니다.

- **Technical Details**: RuleArena는 기존의 규칙 기반 추론 벤치마크와 차별화되는 두 가지 주요 속성을 가지고 있습니다. 첫째, 첫 번째 주문 논리 표현을 넘어서는 확장성을 제공하며, 둘째, 진짜 상황에 기반하여 LLMs이 실제 응용 프로그램에서의 적절성과 신뢰성을 평가하는 데 도움을 줍니다.

- **Performance Highlights**: 실험 결과에 따르면, 현재 기술적으로 가장 진보된 LLM들, 예를 들어 GPT-4o 및 Claude-3.5 Sonnet은 복잡한 규칙 기반 추론 작업에서 대부분 실패하는 것으로 나타났습니다. LLM은 여러 규칙이나 사실을 통합하는 데 어려움을 겪으며, 종종 무관한 정보에 의해 방해받는 경향이 있습니다.



### AFFAKT: A Hierarchical Optimal Transport based Method for Affective Facial Knowledge Transfer in Video Deception Detection (https://arxiv.org/abs/2412.08965)
Comments:
          Accepted by AAAI 2025

- **What's New**: 본 논문에서는 비디오 속임수 탐지를 위한 새로운 방법인 AFFAKT를 제안합니다. 이 방법은 심리학 이론에 착안하여, 대규모 표정 데이터 셋에서 유용하고 상관된 지식을 전이하여 분류 성능을 개선하는 데 초점을 맞추고 있습니다. AFFAKT는 두 개의 주요 과제를 해결하는 데 중점을 두어, 비디오 속임수 탐지 모델의 성능을 향상시킵니다.

- **Technical Details**: AFFAKT는 Hierarchical Optimal Transport Knowledge Transfer (H-OTKT) 모듈과 Sample-specific Re-weighting Knowledge Bank (SRKB) 모듈을 통합하여, 표정 클래스와 속임수 샘플 간의 최적 연결을 정량화합니다. H-OTKT는 카테고리별 지식의 전이량을 결정하고, SRKB는 샘플별 가중치 조정 전략을 사용하여 상관 프로토타입을 통해 전이된 지식을 정교하게 조정합니다. 이 구조는 속임수 탐지에서의 그루비션 성능을 높이기 위한 기반을 제공합니다.

- **Performance Highlights**: 실험 결과, AFFAKT는 두 개의 비디오 속임수 탐지 데이터 세트에서 기존 방법들보다 우수한 성능을 보였습니다. 또한, 해석 가능성 연구를 통해 속임수 행동과 부정적인 감정 간의 높은 연관성을 발견했으며, 이는 심리학 이론과 일치하는 결과입니다. 이상적인 지식 전이로 인해 속임수 탐지의 정확도가 개선되었음을 확인할 수 있습니다.



### Predicting Quality of Video Gaming Experience Using Global-Scale Telemetry Data and Federated Learning (https://arxiv.org/abs/2412.08950)
Comments:
          22 pages, 11 figures, 6 tables

- **What's New**: 이번 연구에서는 FPS(Frames Per Second)가 비디오 게임 경험에 미치는 영향을 다루며, 게임에 대한 정확한 FPS 예측이 플레이어와 개발자 모두에게 이익이 된다는 점을 강조합니다. FPS 예측의 정확성을 높이기 위해 다양한 장치에서의 게임 성능 예측을 위한 연합 학습 기반 모델을 제안합니다. 이 모델은 사용자 데이터를 보호하면서 FPS 성능을 예측할 수 있는 방법으로, 각 플레이어와 게임에 고유한 학습 가능한 지식 커널(Learnable Knowledge Kernel, LKK)을 사용하여 개인화합니다.

- **Technical Details**: 연구는 FPS에 영향을 미치는 다양한 요인들을 종합적으로 분석하여 게임 성능 예측을 위한 새로운 모델을 설계합니다. 이 모델은 224개 국가 및 지역에서 수집된 100,000명의 사용자 데이터를 포함하여 835개의 다양한 비디오 게임과 관련된 76.4백만 개의 게임 프로세스를 기록하였습니다. 이러한 훈련 과정은 웨이제르스타인 거리(Wasserstein distance) 등의 지표를 이용하여 예측 성능을 평가하며, 모델의 정확도를 높이기 위해 동적 커널 적용 기법을 도입합니다.

- **Performance Highlights**: 제안된 모델은 예측된 FPS 분포와 실제 FPS 분포 간의 평균 Wasserstein 거리가 0.469에 달하며 기존의 모든 기준 방법들을 초월하는 성과를 보여주었습니다. 또한, 고유한 LKK를 통한 FPS 예측 정확도 향상을 통해 콜드 스타트 문제를 해결하였으며, 이로 인해 Wasserstein 거리가 7.57% 감소하는 효과를 가져왔습니다. 이 모델은 FPS를 보다 정확히 예측할 수 있어 게임 사용자들에게 더 나은 경험을 제공할 수 있을 것으로 기대됩니다.



### Selective Visual Prompting in Vision Mamba (https://arxiv.org/abs/2412.08947)
Comments:
          in Proceedings of the 39th AAAI Conference on Artificial Intelligence (AAAI-25)

- **What's New**: 이번 연구에서는 Vision Mamba (Vim) 모델에 대한 처음의 시각적 프로프트 기술인 Selective Visual Prompting (SVP) 방법을 도입했습니다. 기존의 비주얼 프롬프팅 기법은 주로 Vision Transformer (ViT) 모델에 맞춰져 있어 Vim의 고유한 특성을 고려하지 않았습니다. SVP는 입력 종속적으로 토큰 수준의 프롬프트를 생성하여 불필요한 정보를 제거하고 차별적인 특징을 효과적으로 전파하는 데 중점을 둡니다.

- **Technical Details**: SVP는 경량 프롬프터를 사용하여 입력에 따라 동적으로 토큰 프롬프트를 생성하며, 이를 통해 Vim 모델의 갱신 및 잊기 게이트를 선택적으로 활성화하여 정보 전파를 촉진합니다. 또한, Cross-Prompting과 Inner-Prompting이라는 이중 경로 구조를 통해 이를 구분된 파라미터로 구현하여 계층 간 공유 정보와 각 계층 내 특정 정보를 최적화합니다. 이는 두 가지 정보 유형 간의 균형 있는 활용을 보장합니다.

- **Performance Highlights**: 광범위한 실험 결과, SVP 방법이 기존의 비주얼 프롬프팅 기법보다 현저히 더 나은 성능을 보였습니다. 모델 크기와 사전 학습 데이터셋은 동일하지만, SVP를 통해 차별적인 정보의 효과적인 전파 및 유지가 가능하다는 것이 입증되었습니다. 이러한 결과는 Vim 모델이 차세대의 모델 아키텍처로 자리매김함을 의미합니다.



### MoSLD: An Extremely Parameter-Efficient Mixture-of-Shared LoRAs for Multi-Task Learning (https://arxiv.org/abs/2412.08946)
Comments:
          Accept by COLING 2025

- **What's New**: 최근 LoRA(Low-Rank Adaptation)가 대형 사전 학습 모델을 미세 조정하는 데 중요한 기술로 부상하였으나, 다중 작업 학습(multi-task learning) 시 성능이 저조한 점이 드러났습니다. 이에 비해 MoE(Mixture of Experts) 아키텍처는 이러한 문제를 자연스럽게 해결할 수 있는 방법으로 주목받고 있습니다. 그러나 MoE는 데이터 간 상호 간섭(mutual interference) 및 다양한 작업의 지식 망각(knowledge forgetting) 같은 도전 과제를 가져옵니다. 이를 해결하기 위해 본 논문에서는 MoSLD(mixture-of-shared-LoRAs)라는 모델을 제안하고, drop out 전략을 활용하여 이러한 문제를 극복합니다.

- **Technical Details**: MoSLD는 LoRA의 상위 프로젝션 매트릭스를 다양한 전문가 간에 공유하여 여러 작업 간의 일반 지식을 학습하도록 유도합니다. 기본적으로 이 모델은 LoRA의 상위 프로젝션 매트릭스(A)와 하위 프로젝션 매트릭스(B)로 구성되어 있으며, 상위 매트릭스는 서로 다른 전문가 간에 공유됩니다. Dropout 전략이 적용되어, 하위 매트릭스의 특정 특징을 유지하는 동시에 매개변수 매트릭스의 불균형 업데이트를 완화하고 과적합(parameter overfitting)을 줄여줍니다.

- **Performance Highlights**: 다양한 실험을 통해 MoSLD 모델이 단일 작업(sing-task) 및 다중 작업(multi-task) 시나리오에서 뛰어난 성능을 보여주었습니다. 특히, 본 모델은 out-of-domain 데이터에 대해 강력한 일반화(generalization) 능력을 발휘하여, 실제 환경에서의 적용 가능성을 높였습니다. 이러한 결과는 MoSLD가 다양한 작업 간의 지식 전이 및 데이터 간 불균형 문제를 효과적으로 해결할 수 있는 잠재력을 가지고 있음을 입증합니다.



### From Text to Trajectory: Exploring Complex Constraint Representation and Decomposition in Safe Reinforcement Learning (https://arxiv.org/abs/2412.08920)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 이 논문에서는 Safe Reinforcement Learning(RL)에서 자연어로 제공된 제약을 활용하는 새로운 방식을 소개합니다. Trajectory-level Textual Constraints Translator(TTCT)를 도입하여 수동으로 설계된 비용 함수(cost function)를 대체함으로써, 제약을 더 유연하게 그리고 직관적으로 처리할 수 있게 됩니다. TTCT는 텍스트 제약을 이해하고 이를 정책(policy) 학습 신호로 활용하여, 더 낮은 위반율(violation rate)을 달성할 수 있음을 보여줍니다.

- **Technical Details**: TTCT는 RL 에이전트의 과거 상태 및 행동을 인식하는 것을 통해 텍스트 제약을 평가하는 방법을 제안합니다. 이 과정에서, 텍스트와 트라젝토리 사이의 임베딩 유사성을 최대화하고 비일치 쌍의 유사성을 최소화하는 대조 학습(contrastive learning) 접근 방식을 사용합니다. 또한, 시간적 신뢰 할당(temporal credit assignment) 방법을 도입하여 트라젝토리 내의 각 상태-행동 쌍에 대한 더 밀집된 비용 신호를 할당하여, 안전성과 학습 성능을 향상시킵니다.

- **Performance Highlights**: 실험에 따르면, TTCT 방식으로 학습된 에이전트는 기존의 비용 함수로 훈련된 에이전트에 비해 위반률이 최대 4.0배 낮아졌으며, 보상(reward) 성능은 유사한 수준을 유지했습니다. 더불어, 제약 환경이 변화하는 상황에서도 미세 조정 없이도 적응할 수 있는 제로샷(zero-shot) 능력이 있음을 보여주었습니다. 이러한 성과는 TTCT가 복잡한 다차원 제약을 처리하는 데 있어 매우 유연하고 일반화 가능한 시스템임을 입증합니다.



### Goal-Conditioned Supervised Learning for Multi-Objective Recommendation (https://arxiv.org/abs/2412.08911)
- **What's New**: 이 논문은 Multi-Objective Goal-Conditioned Supervised Learning (MOGCSL) 프레임워크를 소개합니다. MOGCSL은 기존의 Goal-Conditioned Supervised Learning (GCSL) 방법을 여러 목표를 처리하도록 확장하며, 목표를 일차원 스칼라로부터 다차원 벡터로 재정의합니다. 이를 통해 복잡한 아키텍처와 최적화 제약 조건을 자연스럽게 제거할 수 있습니다.

- **Technical Details**: MOGCSL은 오프라인 시퀀셜 데이터로부터 여러 목표를 자동으로 달성하기 위해 설계되었습니다. 유익하지 않거나 노이즈가 있는 인스턴스를 필터링하고, '높이' 달성 가능한 목표를 선택하는 새로운 목표 선택 알고리즘을 포함합니다. 이 시스템은 상업적인 추천 시스템에서 다음 행동 예측 문제에 적용되는 데 중점을 두고 있으며, 대량의 노이즈 데이터에 강인함을 가지고 있습니다.

- **Performance Highlights**: MOGCSL은 실제 추천 데이터셋에서 시행된 광범위한 실험을 통해 높은 성능을 입증했습니다. 특히, 추천 시스템에서 훈련 데이터의 노이즈가 있는 부분을 배제하는 데 강력한 능력을 보여줍니다. 이 연구는 MOGCSL이 효율성 및 효과성을 고려할 때 매우 뛰어난 성능을 보였다는 것을 강조합니다.



### Phi-4 Technical Repor (https://arxiv.org/abs/2412.08905)
- **What's New**: phi-4는 140억 개의 파라미터를 가진 언어 모델로, 데이터 품질 중심의 학습 방식을 채택하여 훈련되었습니다. 기존의 언어 모델들과 달리, phi-4는 훈련 과정 전반에 걸쳐 합성 데이터(synthetic data)를 전략적으로 통합하여 놀라운 성과를 올렸습니다. 특히, phi-4는 STEM 관련 질의응답(STEM-focused QA) 능력에서 GPT-4를 능가하며, 이는 데이터 생성(data-generation) 및 후속 훈련(post-training) 기술이 기존 방식보다 한층 발전했음을 보여줍니다.

- **Technical Details**: phi-4는 주로 합성 데이터로 훈련되며, 멀티 에이전트 프롬프팅(multi-agent prompting), 자기 수정 워크플로우(self-revision workflows), 지시 반전(instruction reversal) 등의 다양한 기법을 통해 생성됩니다. 이러한 방법들은 모델이 더 강한 추론 및 문제 해결 능력을 갖추도록 설계된 데이터셋을 만드는 데 기여합니다. 또한, 훈련 과정에서의 커리큘럼 최적화와 포스트 트레이닝(post-training) 혁신도 주요하게 다루어집니다.

- **Performance Highlights**: phi-4는 훈련 후 평가에서 새로운 데이터로 테스트했을 때 뛰어난 성과를 보였습니다. 특히 전통적인 벤치마크에서 GPT-4의 성능을 크게 초과하며, MATH(수학 대회) 및 GPQA(대학원 STEM Q&A) 등에서 두드러진 성과를 기록하였습니다. 이러한 성과는 phi-4의 훈련 방식이 과적합(overfitting) 문제를 잘 관리하고 있음을 나타내며, 새로운 테스트 데이터에서의 강력한 승인 결과가 이를 뒷받침합니다.



### Radiology Report Generation via Multi-objective Preference Optimization (https://arxiv.org/abs/2412.08901)
Comments:
          11 pages,3 figures

- **What's New**: 이번 연구에서는 자동 방사선 보고서 생성(RRG) 방식에서 다중 목표 최적화(MPO)를 통한 인간의 다양한 선호를 반영하는 새로운 방법을 제안합니다. 기존의 RRG 모델은 단일 평가 지표에 의존하여 보고서를 생성하기 때문에 방사선 전문의들의 다양한 선호와 일치하지 않는 문제가 있었습니다. 본 연구는 다차원 보상 함수를 활용하여 선호를 최적화하고 강화 학습 기법을 통해 이를 해결하고자 합니다.

- **Technical Details**: 제안된 방법은 두 가지 새로운 모듈인 선호 벡터 융합(PVF) 네트워크와 다중 목표 최적화(MOO) 모듈을 포함합니다. 이 구조는 인코더와 디코더 사이에 위치하고, 다중 헤드 어텐션 메커니즘을 통해 선호 벡터와 인코딩된 이미지 특징을 융합합니다. 이렇게 함으로써 RRG 모델은 특정 선호와 조건부로 보고서를 생성할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 두 개의 공개 데이터셋에서 다양한 선호를 가진 데이터를 처리할 수 있는 단일 모델로서 최신 성능을 달성했습니다. 추가적인 파인튜닝 없이도 특정 선호에 맞춰 보고서를 생성할 수 있는 능력을 보여주었습니다. 이는 RRG 분야에서 선호 다각화를 위한 최초의 시도로, 의료 영상 분야에서의 활용 가능성을 크게 확장합니다.



### AI-assisted Knowledge Discovery in Biomedical Literature to Support Decision-making in Precision Oncology (https://arxiv.org/abs/2412.08900)
Comments:
          Accepted at AMIA Annual Symposium 2024

- **What's New**: 이번 연구에서는 암 환자에게 적절한 타겟 테라피(targeted therapy)를 제공하기 위해 필수적인 분자 프로파일링(molecular profiling)과 임상 특성을 분석하는 과정에서 자연어 처리(natural language processing)의 기여 가능성을 평가하였습니다. 특히, 생의학 문헌에서 지식 발견(knowledge discovery)을 지원하는 다양한 모델들을 테스트하였습니다.

- **Technical Details**: Bidirectional Encoder Representations from Transformers(BERT) 계열의 두 가지 모델, 두 가지 대형 언어 모델(Large Language Models) 및 PubTator 3.0의 성능을 비교 분석하였습니다. 이 연구는 명명된 개체 인식(named entity recognition, NER)과 관계 추출(relation extraction, RE) 작업을 수행하는 능력을 중심으로 진행되었습니다.

- **Performance Highlights**: PubTator 3.0과 BioBERT 모델이 NER 작업에서 각각 0.93과 0.89의 최상의 F1 점수를 기록하며 가장 우수한 성능을 보였습니다. BioBERT는 RE 작업에서 다른 솔루션들을 초월하는 성능을 보였으며, 특정 사용 사례에서도 거의 모든 개체 언급(entity mentions)과 대부분의 관계를 성공적으로 인식하였습니다.



### SMMF: Square-Matricized Momentum Factorization for Memory-Efficient Optimization (https://arxiv.org/abs/2412.08894)
- **What's New**: 본 논문에서는 SMMF (Square-Matricized Momentum Factorization)이라는 메모리 효율적인 최적화 기법을 제안합니다. 이 기법은 Adam과 같은 적응형 학습률 최적화 기법의 메모리 요구량을 최대 96%까지 줄입니다. SMMF는 임의의 순위(rank)와 형태(shape)를 가지는 모멘텀 텐서를 효율적으로 분해할 수 있도록 설계되었습니다. 이는 CNN 및 Transformers와 같은 다양한 딥 모델 아키텍처에 적용 가능하게 합니다.

- **Technical Details**: SMMF는 첫 번째 및 두 번째 모멘텀 텐서를 행렬으로 변환하여 처리하는 독창적인 방법을 사용합니다. 이는 메모리 공간을 크게 줄이면서도 기존의 메모리 효율적 최적화 기법과 유사한 최적화 성능을 제공합니다. SMMF는 NNMF (Non-Negative Matrix Factorization)를 통해 두 개의 벡터로 모멘텀 텐서를 분해하며, 이러한 과정을 통해 메모리 사용을 최소화합니다. 또한, 이 방법은 이론적인 수렴 보장 분석을 통해 경쟁력 있는 최적화 기법으로서의 기초를 제공합니다.

- **Performance Highlights**: 실험 결과, SMMF는 Adafactor, CAME, SM3와 같은 최신 메모리 효율 최적화 기법에 비해 최대 96% 적은 메모리를 사용하면서도 비슷한 성능을 달성하였습니다. 이는 특히 Raspberry Pi와 같은 극히 제한된 메모리를 가진 시스템에서도 모델을 효과적으로 학습할 수 있게 합니다. 따라서 SMMF는 다양한 딥러닝 작업에서 실질적인 메모리 절약과 성능을 동시에 제공합니다.



### Efficient Reinforcement Learning for Optimal Control with Natural Images (https://arxiv.org/abs/2412.08893)
- **What's New**: 이번 연구는 자연 이미지 시퀀스에서 최적 제어(optimal control) 문제를 다루고 있습니다. 연구자들은 이미지가 최적 정책을 구현하는 데 충분한 조건을 파악하고, 특정 유형의 이미지 표현이 강화 학습(reinforcement learning)에서 효율적임을 보여줍니다. 새로운 벤치마크를 개발하여 다양한 이미지 표현의 성능과 효율성을 비교할 수 있도록 하였습니다.

- **Technical Details**: 자연 이미지는 높은 차원의 데이터 세트로, 과제가 요구하는 정보가 포함되어 최적 정책을 결정할 수 있는 가능성이 있습니다. 연구자는 이러한 이미지 표현을 사용해 Markov 결정 과정(Markov decision process)과 최적 제어 과제를 효율적으로 해결하기 위한 일반적인 프레임워크를 개발했습니다. 또한, 과도 표현(overcomplete representation)을 사용하는 스파스 코드(sparse code)를 통해 더욱 효율적인 정책 찾기가 가능하다는 점을 강조하였습니다.

- **Performance Highlights**: 연구 결과에 따르면, 스파스 코드는 이미지 표현에서 최적 정책을 찾는 데 필요한 계산 자원을 크게 줄일 수 있습니다. 벤치마크는 상태와 시간 간격을 쉽게 확장할 수 있도록 설계되어, 긴 시간 동안의 작업 및 많은 상태를 해결하는 데 유리합니다. 이러한 접근 방식은 강화 학습 분야에서 실세계 데이터의 활용을 확대하는 데 기여할 것으로 예상됩니다.



### Residual Channel Boosts Contrastive Learning for Radio Frequency Fingerprint Identification (https://arxiv.org/abs/2412.08885)
Comments:
          5 pages, 4 figures

- **What's New**: 본 연구는 미리 훈련된 모델을 새로운 환경에 배치하는 데 있어 제한된 데이터 샘플 문제를 해결하기 위해, Radio Frequency Fingerprint Identification (RFFI)에 대한 잔여 채널 기반의 데이터 증강 전략을 제안합니다. 가벼운 SimSiam 대조 학습 프레임워크와 결합하여, 서로 다른 잔여 채널 효과를 가진 신호를 생성합니다. 이를 통해 모델이 더욱 효과적인 표현을 학습할 수 있도록 하고, 새로운 환경에서 1% 샘플로 모델을 세밀하게 조정합니다.

- **Technical Details**: 이 연구에서는 least square (LS) 및 최소 평균 제곱 오차 (MMSE) 채널 추정 기법을 적용하여 잔여 채널 신호를 생성합니다. 이를 통해 다양한 잔여 채널에서의 특성을 모델이 자동으로 추출하게 됩니다. 또한, 이러한 원시 신호를 평등화하여, 영어 및 월드내(두 개의 주파수 프레임워크에 대한 평등화) 기술을 활용하여 실제 월드 서비스에 대한 강력한 성능을 달성하도록 설계하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 특성 추출 능력과 일반화 성능을 크게 향상시키면서 적은 샘플과 짧은 시간 내에 이루어졌습니다. 따라서 이 방법은 실제 무선 보안 응용 프로그램에 적합하며, 기존의 프리트레인 모델과 비슷한 성능을 유지할 수 있음을 보여주었습니다. 더불어, 메모리 요구 사항과 컴퓨팅 리소스의 요구도 현저히 줄었습니다.



### Towards modeling evolving longitudinal health trajectories with a transformer-based deep learning mod (https://arxiv.org/abs/2412.08873)
- **What's New**: 이 연구에서는 전국 규모의 전자 건강 기록(Health Records) 기반으로 개인의 건강 경로(health trajectory)를 변화하는 방식으로 분석하는 Transformer 기반의 심층 학습 모델을 제안합니다. 모델은 단일 진단 예측을 넘어, 시간에 따라 변화하는 연속적인 예측을 수행할 수 있는 능력을 가지고 있습니다. 이러한 접근은 EHR 데이터의 복잡한 특성을 효과적으로 다루며, 질병 발생을 예측하는 데 유용할 것으로 기대됩니다.

- **Technical Details**: Transformer 아키텍처를 기반으로 한 이 모델은 다수의 multi-head self-attention 레이어를 사용하여 입력 시퀀스의 각 요소를 병렬로 처리합니다. 입력 데이터에는 의료 관련 코드, 나이, 그리고 시퀀스 내의 위치 정보가 포함됩니다. 여기에서 우리는 예측 기간까지 남은 연수를 나타내는 추가 매개변수(t2f)를 도입하여 보다 정교한 분석을 가능하게 합니다.

- **Performance Highlights**: 모델은 기존의 bi-directional transformer 모델과 유사한 수준의 기본 예측 성능을 보여주면서도, 건강 경로의 동적 변화를 모델링하는 데 있어 유망한 특성을 지니고 있습니다. 변화를 감지하는 다양한 방식으로 모델을 사용하며, 이는 향후 질병 발생을 예측하는 데 도움을 줄 수 있습니다. 이 연구는 개인의 건강 모니터링을 위한 지속적인 개입을 가능하게 할 수 있는 방법을 탐구하고 있습니다.



### Inference-Time Diffusion Model Distillation (https://arxiv.org/abs/2412.08871)
Comments:
          Code: this https URL

- **What's New**: 이 연구에서는 Distillation++라는 새로운 inference-time distillation 프레임워크를 도입하여 teacher 모델과 student 모델 간의 성능 격차를 줄이고자 합니다. 기존의 방법들과 달리, 이 프레임워크는 샘플링 프로세스 전반에 걸쳐 teacher의 지도를 지속적으로 통합하여 데이터 없이도 denoising 프로세스를 실시간으로 개선할 수 있습니다. 또한, Distillation++는 다양한 student 모델과의 호환성을 가지고 있으며, 다양한 solver에 대한 일반적인 적용 가능성을 보여줍니다.

- **Technical Details**: Diffusion 모델은 고품질 샘플을 생성하기 위해 초기 노이즈 벡터를 점진적으로 denoise하는 iterative refinement 프로세스를 사용합니다. 이 모델은 확률 흐름 확률 미분 방정식(Probability Flow ODE, PF-ODE) 경로를 따라 적분을 직접 추정함으로써 teacher 모델을 distill하는 방식으로 연산 비용을 절감할 수 있습니다. 또한, Distillation++는 샘플링 경로를 score distillation sampling loss (SDS)를 통해 정규화하는 방식으로 학생 모델의 샘플링을 근접 최적화 문제로 재구성합니다.

- **Performance Highlights**: Distillation++는 특히 초기 샘플링 단계에서의 성능을 크게 향상시켜 주목받고 있으며, 이는 총 비용 없이 student 모델의 성능을 개선할 수 있음을 보여줍니다. 이 접근 방식은 기존의 가장 진보된 distillation 기법들에 비해 우수한 성능을 나타내며, 최종적으로 diffusion distillation 모델을 위한 강력한 guided sampling 프로세스를 제시합니다. Empirical 결과들은 Distillation++가 신뢰할 수 있는 post-training 솔루션으로 자리잡을 가능성을 증명합니다.



### Key Safety Design Overview in AI-driven Autonomous Vehicles (https://arxiv.org/abs/2412.08862)
- **What's New**: 이 논문은 자율주행 SAE 레벨 3 및 4의 증가하는 존재와 함께, 인공지능 소프트웨어를 포함한 복잡한 기술적 과제를 다룹니다. 기능 안전성(functional safety) 및 견고한 소프트웨어 설계의 중요성을 강조하며, 자동차 소프트웨어와 하드웨어를 위한 안전 아키텍처(safety architecture)와 체계적 접근 방식(systematic approach)을 제시합니다.

- **Technical Details**: 논문은 자동차 안전 완전성 수준(ASIL D, 가장 높은 안전 완전성 등급)의 실패 소프트(fail soft) 처리, 인공지능(AI)과 머신러닝(ML)의 통합(integration) 등 필수 안전 아키텍처 요소를 설명합니다. 또한 AI 기반 자동차 소프트웨어의 고유한 도전 과제를 해결하기 위해 다양한 기법을 제안하고, 소프트웨어 신뢰성(reliability) 향상에서 AI의 역할을 논의합니다.

- **Performance Highlights**: 안전 실패 분석(safety failure analysis) 및 완화 전략(mitigation strategies)의 도입을 통해 자동차 소프트웨어의 안전성과 신뢰성을 보장하는 방안이 제시됩니다. 데이터 라이프사이클(data lifecycle) 전반에서 AI의 역할이 강조되며, 이러한 접근 방식이 고급 운전 보조 시스템(ADAS) 애플리케이션에서의 성능 평가(performance evaluation)에 미치는 영향을 분석합니다.



### Quantum Kernel-Based Long Short-term Memory for Climate Time-Series Forecasting (https://arxiv.org/abs/2412.08851)
Comments:
          arXiv admin note: text overlap with arXiv:2411.13225

- **What's New**: 본 논문에서는 양자 커널 기반 Long Short-Term Memory (QK-LSTM) 네트워크를 제안합니다. QK-LSTM은 고전적인 LSTM 아키텍처에 양자 커널 방법을 통합하여 기후 시계열 예측의 정확성과 계산 효율성을 향상시킵니다. AQI 예측과 같은 기후 관련 예측 작업에서 QK-LSTM의 효율성을 강조하고 있습니다.

- **Technical Details**: QK-LSTM은 고전적인 입력을 고차원 양자 특징 공간에 임베딩하여 비선형 종속성과 시간적 동역학을 포착합니다. 이런 방식은 학습 가능한 매개변수를 줄이면서도 복잡한 패턴과 장기 의존성을 효과적으로 모델링할 수 있도록 합니다. 양자 커널 방법을 활용하여 양자 공간에서의 내적(product) 계산을 효율적으로 수행하고, 복잡한 양자 회로 기반 모델의 전통적인 계산 문제를 해결합니다.

- **Performance Highlights**: 실험 결과 QK-LSTM은 AQI 예측에서 고전적인 LSTM 네트워크보다 더 나은 성능을 보여주었습니다. 이는 환경 모니터링 및 자원 제약이 있는 상황에서의 잠재력을 강조하며, 대규모 기후 데이터셋을 다루는 데 있어 양자 강화 머신러닝 프레임워크의 보다 넓은 응용 가능성을 보여줍니다.



### Labits: Layered Bidirectional Time Surfaces Representation for Event Camera-based Continuous Dense Trajectory Estimation (https://arxiv.org/abs/2412.08849)
Comments:
          24 pages, 12 figures, 9 tables

- **What's New**: 이 논문에서는 기존의 이벤트 카메라 기반 수치 표현의 한계를 극복하기 위해 Labits: Layered Bidirectional Time Surfaces 라는 새로운 표현 방법을 제안합니다. Labits는 모든 중요한 특성을 보존하면서 이벤트 카메라의 비동기적 특성을 인지하고 이를 활용하는 혁신적인 솔루션입니다. 이로 인해 기존 기술에 비해 트레일리감 엔드포인트 오차(Trajectory End-Point Error, TEPE)를 49% 줄이는 성과를 올렸습니다.

- **Technical Details**: 이벤트 카메라는 비상동형 동시 비전 센서로, 사전 정의된 임계치 이상의 밝기 변화가 감지되면 즉시 이벤트를 발생시킵니다. 각 이벤트는 위치(x,y), 순간적인 시간(t), 그리고 이진 극성(p) 정보를 기록하며, 이를 통해 정밀한 이동 객체 추적이 가능합니다. 그러나 기존의 데이터 변환 방식인 GNN(Graph Neural Network)이나 SNN(Spiking Neural Network)의 한계로 인해 이벤트 기반 비전의 널리 사용에 제약이 있었습니다.

- **Performance Highlights**: Labits를 적용한 결과, TEPE의 13% 개선이 있었으며, 추가적인 APLOF(Active Pixel Local Optical Flow) 모듈을 도입함으로써 오류를 추가로 30% 줄일 수 있었습니다. 이러한 성과는 이벤트 카메라의 강점을 최대한 활용한 결과로, 자동차나 로봇 등 다양한 분야에서의 응용 가능성을 높였습니다.



### Exploring Large Language Models on Cross-Cultural Values in Connection with Training Methodology (https://arxiv.org/abs/2412.08846)
- **What's New**: 이번 논문에서는 오픈소스 대규모 언어 모델(Large Language Model, LLM)이 다양한 국가의 문화적 가치에 대한 판단을 어떻게 수행하는지를 탐구했습니다. 특히 모델 크기, 훈련 코퍼스, 정렬(alignment) 등 훈련 방법론과의 관계를 분석하였습니다. LLM은 인간과 유사하게 사회문화적 규범을 판단하지만, 사회 체제나 진보에 대해서는 다소 차이를 보입니다. LLM의 문화적 가치 판단은 서구 문화에 편향된 경향이 있으며, 다국어 코퍼스에 대한 훈련을 통해 이를 개선할 수 있습니다.

- **Technical Details**: 연구에서는 World Value Survey(WVS) 데이터셋을 활용하여 LLM의 문화적 가치 이해도를 측정했습니다. WVS는 55개 국가에서 12개 범주에 걸쳐 총 209개 사회 가치 관련 질문을 포함하고 있습니다. 각 질문은 다중 선택형 태스크로 변환되며, LLM과 인간의 응답 선택 간의 상관관계를 측정하는 방법론이 제시됩니다. LLM에서 생성된 정답 후보에 기반하여 확률 분포를 정의하고, 평균 점수를 계산하여 인간의 응답과 비교합니다.

- **Performance Highlights**: 분석 결과, LLM은 인간과 유사한 방식으로 사회문화적 규범을 판단할 수 있으며, 더 큰 모델일수록 문화적 인식이 더 강하게 나타났습니다. 또한 작은 모델은 합성 데이터(synthetic data)를 통해 문화적 지식을 강화할 수 있는 가능성이 있습니다. LLM을 인간처럼 정렬하는 것이 인간과의 유사성을 높이는 데 기여할 수 있음을 확인했습니다. 최종적으로 LLM의 설계 방법론 개선을 위한 중요한 통찰력이 제공되었습니다.



### Quantum-Train-Based Distributed Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2412.08845)
- **What's New**: 이번 논문에서 우리는 전통적인 Reinforcement Learning(RL)의 규모 확장 문제를 퀀텀 컴퓨팅 원리를 통합하여 해결하는 새로운 접근법인 Quantum-Train 기반 분산 다중 에이전트 강화 학습(Dist-QTRL)을 소개합니다. 본 연구는 파라미터화된 퀀텀 회로를 활용하여 효율적으로 신경망 파라미터를 생성하고, 이는 훈련 가능한 파라미터의 차원을 폴리로지 수준으로 줄여줍니다. 또한, 이 프레임워크는 여러 개의 Quantum Processing Units(QPUs)가 병렬로 작동하는 분산 다중 에이전트 환경을 위해 설계되었습니다.

- **Technical Details**: Dist-QTRL 프레임워크는 클래식 신경망의 파라미터 축소를 위한 분산 퀀텀 훈련을 활용하여 고성능 컴퓨팅(HPC) 환경으로 확장을 지원합니다. 이를 통해 퀀텀 얽힘을 활용한 데이터 표현 개선과 신속한 수렴을 가능하게 합니다. 논문은 Dist-QTRL 프레임워크의 수학적 정식화와 수렴 특성을 탐구하며, 경험적 결과를 뒷받침합니다.

- **Performance Highlights**: 실험 결과, 우리의 프레임워크는 전통적인 QTRL 모델 대비 성능 개선을 보여주었습니다. 특히 분산 컴퓨팅 환경에서 모델 정확도를 저해하지 않으면서 병렬화로 인한 상당한 속도 향상이 이루어졌습니다. 이 연구는 실제 애플리케이션에서 대규모 및 고차원 작업을 해결하기 위한 확장 가능한 퀀텀 강화 학습 시스템을 위한 가능성을 제시합니다.



### Kajal: Extracting Grammar of a Source Code Using Large Language Models (https://arxiv.org/abs/2412.08842)
Comments:
          9 pages, 6 figures, 1 table, preprint

- **What's New**: 이번 논문에서는 Kajal이라는 새로운 접근 방식을 소개합니다. Kajal은 도메인 특화 언어(DSL)의 문법을 자동으로 유추하는 방법으로, 대규모 언어 모델(LLMs)을 활용하여 프롬프트 엔지니어링과 몇 샷 학습(few-shot learning) 기술을 결합했습니다. 이 과정은 LLM이 코드 스니펫을 기반으로 문법을 생성하도록 안내하며, 반복적인 피드백을 통해 이를 개선합니다.

- **Technical Details**: Kajal의 핵심 기능은 피드백 기반의 반복 프로세스와 몇 샷 프롬프트 기술을 통합한 것입니다. 초기 추론 이후, 문법이 입력 코드 스니펫을 성공적으로 구문 분석(parse)할 수 있는지 테스트하고, 실패 시에는 오류 메시지를 LLM에 피드백하여 프롬프트를 수정합니다. 이 과정은 최대 10회 반복되며, 올바른 문법이 생성될 때까지 계속 진행됩니다.

- **Performance Highlights**: Kajal은 몇 샷 학습을 활용할 경우 60%의 정확도를, 사용하지 않을 경우 45%의 정확도를 달성했습니다. 이를 통해 이 도구가 DSL 문법 추출의 효율성을 크게 향상시킬 수 있는 가능성을 보여줍니다. 향후 연구에서는 소형 오픈 소스 LLM을 활용하고, 대규모 데이터셋을 통해 Kajal의 성능을 추가로 검증할 계획입니다.



### HadaCore: Tensor Core Accelerated Hadamard Transform Kern (https://arxiv.org/abs/2412.08832)
- **What's New**: HadaCore는 최신 GPU 하드웨어에 있는 Tensor Cores를 활용하여 최적화된 Fast Walsh-Hadamard Transform (FWHT) 알고리즘입니다. 기존 FWHT의 재귀 구조를 따라 동일한 비대칭 런타임 복잡성을 유지하면서도 Tensor Core 가속의 이점을 활용합니다. 이를 통해 계산 및 데이터 교환에서의 병목현상을 줄였습니다.

- **Technical Details**: HadaCore는 Hadamard 행렬에 기반하여 재귀적으로 구성된 Walsh-Hadamard 행렬을 사용합니다. 이 알고리즘은 m×n 크기의 행렬에 대해 O(mn log(n))의 시간을 가지며, 일반적인 행렬 곱셈보다 빠른 속도로 작동합니다. 최신 GPU에서의 병렬 처리를 통해 내부 루프들 간의 오버랩 없이 효율적으로 동작하고, CUDA 프로그래밍을 통해 스레드 간의 동기화를 최소화하였습니다.

- **Performance Highlights**: Nvidia A100 및 H100 GPU에서 HadaCore는 기존 알고리즘에 비해 1.1-1.4배, 최대 3.5배의 속도를 향상시켰습니다. FP16 또는 BF16을 사용할 경우에도 높은 수치적 정확성을 유지하면서 MMLU 벤치마크에서 상대적인 정확도를 제공합니다. 이로 인해 고속게산을 필요로 하는 LLM의 응용에서 HadaCore의 성능이 돋보입니다.



### Efficient Dynamic Attributed Graph Generation (https://arxiv.org/abs/2412.08810)
Comments:
          14 pages,10 figures. Accepted by IEEE ICDE2025

- **What's New**: 이번 논문에서는 VRDAG(Variational Recurrent Dynamic Attributed Graph Generator)를 소개합니다. 기존의 그래프 생성 방식으로는 동적 구조와 노드 속성을 동시에 생성하는 데 한계가 있었으나, VRDAG는 비양방향 메시지 전송 메커니즘을 통해 이를 해결합니다. 또한, 특정 조건부 변량 베이즈 방법을 사용해 이웃 타임스텝에서 새로운 스냅샷을 생성할 수 있도록 설계되었습니다.

- **Technical Details**: VRDAG는 GRU 기반의 서브모듈을 활용하여 생성된 시퀀스의 시간적 의존성을 캡처합니다. 이 모델은 고차원 잠재 변수를 샘플링하여 구조와 노드 속성의 진화 과정을 효과적으로 포착합니다. 이러한 잠재 변수들은 이전 그래프의 숨겨진 상태를 기반으로 하여 새롭고 현실적인 그래프를 생성하는 데 도움을 줍니다.

- **Performance Highlights**: 광범위한 실험 결과, VRDAG는 기존의 최신 모델들과 비교했을 때 매우 효과적이고 효율적인 동적 속성 그래프 생성을 보여주었습니다. 생성 속도가 상당히 빨라졌으며, 실제 네트워크에서 관찰된 공동 진화 패턴을 제대로 모델링할 수 있습니다. 이를 통해 그래프 마이닝 커뮤니티에 중요한 기여를 할 수 있는 잠재력이 있습니다.



### Coverage-based Fairness in Multi-document Summarization (https://arxiv.org/abs/2412.08795)
- **What's New**: 이 논문은 다중 문서 요약(multi-document summarization, MDS)에서 공정성을 측정하는 새로운 방법을 제안하고 있습니다. 제안된 공정성 척도는 이전의 Proportional Representation(비례 대표성) 방식을 개선하여, 정보의 중복성을 고려한 Equal Coverage와 코퍼스 전체의 공정성을 평가하는 Coverage Parity를 포함합니다. 이를 통해 다양한 사회적 속성을 가진 문서를 공정하게 요약할 수 있는 시스템을 평가하였습니다.

- **Technical Details**: Equal Coverage는 문서 집합에서 각 문서가 요약에 포함될 확률이 사회 속성과 독립적이어야 한다는 원칙을 기반으로 하고 있습니다. 또한 Coverage Parity는 다양한 사회 속성을 가진 문서들이 과대표현 또는 과소 대표되는 일이 없도록 평가합니다. 이러한 두 가지 공정성 척도는 LLM(Large Language Models)에서의 출력 결과를 기반으로 검증되었습니다.

- **Performance Highlights**: 실험 결과, 여러 LLM을 평가한 결과 Claude3-sonnet이 가장 공정하다는 것을 발견했습니다. 또한 대부분의 LLM은 특정 사회적 속성을 과대표현하는 경향이 있음을 알 수 있었습니다. 이러한 결과는 사용자들이 LLM으로 생성된 요약을 사용할 때 더욱 섬세한 판단을 할 수 있도록 도와줄 것입니다.



### LLaVA-Zip: Adaptive Visual Token Compression with Intrinsic Image Information (https://arxiv.org/abs/2412.08771)
- **What's New**: 이 논문에서는 시각 토큰의 과부하 문제를 해결하기 위해 DFMR(Dynamic Feature Map Reduction)라는 새로운 방법을 제안합니다. DFMR는 이미지를 기반으로 시각 토큰을 동적으로 압축하여 최대 토큰 한도를 효율적으로 관리합니다. 이는 자원 제약이 있는 학술 환경과 산업 환경 모두에 적용 가능하여, 멀티 이미지 및 비디오 시나리오를 처리하는 데 유리합니다.

- **Technical Details**: DFMR는 LLaVA-1.5 모델에 통합되어 시각 토큰의 압축 비율을 이미지의 내재적 정보에 따라 조정합니다. 이 모델의 구조는 비전 인코더와 프로젝터 사이에 DFMR 모듈이 삽입된 형태로 설계되었습니다. 각 입력 이미지는 비전 인코더를 통해 시각 토큰으로 변환된 후, DFMR 모듈을 지나 압축되어 최종적으로 LLM에 전달됩니다.

- **Performance Highlights**: 실험 결과, DFMR를 적용한 LLaVA-1.5는 다양한 멀티모달 평가 기준에서 성능이 개선되었음을 보여줍니다. 특히, DFMR은 압축된 시각 토큰을 활용할 때 모든 여덟 개의 업무에서 성능이 향상되었습니다. 이러한 결과는 DFMR이 시각 토큰 과부하 문제를 해결하는 유망한 방법임을 시사합니다.



### Integrating Optimization Theory with Deep Learning for Wireless Network Design (https://arxiv.org/abs/2412.08761)
Comments:
          Accepted for publication in IEEE Communications Magazine

- **What's New**: 이 논문은 전통적인 최적화 이론과 딥러닝 기법을 통합하여 기존 연구들의 이론적 기반 부족 문제를 해결하는 혁신적인 구조적 및 분석적 프레임워크를 소개합니다. 이 메소드는 최적화 이론 기반 솔루션의 블록 다이어그램을 구성하고, 최적성 조건 및 반복 프로세스와 관련된 핵심 구성 요소를 식별하는 것으로 시작됩니다. 그런 다음 특정 핵심 요소를 다양한 수준에서 DNN(Deep Neural Networks)으로 교체하여 시스템의 적응성과 해석 가능성을 Enhanced시킵니다.

- **Technical Details**: 최적화 이론 기반 접근은 의사결정 변수와 극대화하거나 최소화할 목표 함수를 명시하는 것으로 시작합니다. 시스템 성능에 대한 정밀한 수학적 모델이 필요하며, 이는 일반적으로 도메인 전문 지식에 기반하여 복잡한 문제를 발생시킵니다. 반면, 딥러닝 기반 접근은 DNN을 사용하여 시스템 매개변수에 따른 최적 자원 배분을 수행합니다. 딥러닝 모델은 데이터를 통해 훈련되며, 이 과정에서 필요한 교육 데이터의 양을 줄이기 위해 다양한 사전 훈련 전략이 사용됩니다.

- **Performance Highlights**: 시뮬레이션 결과, 이 하이브리드 접근 방식은 최적화 이론 기반 접근 방식에 비해 실행 시간을 줄이는 동시에 정확도와 수렴률을 유의미하게 향상시킵니다. 이를 통해 순수 딥러닝 모델을 초과하는 성과를 보여 주며, 최적화 알고리즘의 적응성과 해석 가능성을 크게 개선합니다. 또한, 분석적 블록 다이어그램을 DNN 아키텍처에 통합함으로써 해결책의 해석 가능성을 높이고, 전통적인 블랙박스 모델에 의존하지 않습니다.



### Proactive Adversarial Defense: Harnessing Prompt Tuning in Vision-Language Models to Detect Unseen Backdoored Images (https://arxiv.org/abs/2412.08755)
- **What's New**: 이 논문은 백도어 공격(Backdoor attacks)을 탐지하기 위한 새로운 방법을 제안합니다. 기존의 연구들은 이러한 공격을 방지하기 위해 모델의 파라미터를 조정하는 데 집중했지만, 본 연구는 백도어 공격으로부터 모델을 사전 예방적으로 보호하는 알고리즘을 개발했습니다. 이를 통해 훈련 및 추론 과정에서 보이지 않는 백도어 이미지를 효과적으로 분류할 수 있습니다.

- **Technical Details**: 이 방법은 Vision Language Models(VLMs)의 성공적인 프롬프트 튜닝(prompt tuning)을 활용하여 작동합니다. 백도어 공격을 탐지하기 위해 학습 가능한 텍스트 프롬프트를 훈련시켜, 정상 이미지와 백도어 트리거가 숨겨진 이미지를 구별합니다. 이 과정에서 기존의 공격 패턴을 분석하고, 보다 정교한 데이터 셋을 통해 불순 이미지들을 사전에 제거하는 방식을 채택합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 두 개의 유명한 데이터 세트에서 백도어 트리거 탐지 정확도 86%를 달성했습니다. 이는 이전의 방어 기법들에 비해 획기적인 성과로, 백도어 방어의 새로운 기준을 설정합니다. 이 연구는 객체 인식 시스템 보안 강화에 큰 기여를 할 것으로 기대됩니다.



### Sampling-based Continuous Optimization with Coupled Variables for RNA Design (https://arxiv.org/abs/2412.08751)
- **What's New**: 본 연구는 RNA 디자인 문제를 다루며, 기존의 휴리스틱 방법 대신 연속 최적화(continuous optimization) 접근 방식을 제안합니다. 이를 통해 무작위 후보 시퀀스의 분포를 시작점으로 설정하고, 기울기 하강(gradient descent) 방법을 사용하여 목적 함수의 기대값을 향상시킵니다. 또한, 유효하지 않은 시퀀스를 제거하고 뉴클레오타이드 간 상관관계를 모델링하기 위해 새로운 분포를 정의합니다.

- **Technical Details**: 연구에서는 미리 정해진 후보군이 아닌, 모든 후보 시퀀스에 대한 분포에서 시작하여 기울기 계산을 통해 최적 해결책을 찾아가는 방법을 사용합니다. softmax parameterization을 통해 각 뉴클레오타이드의 확률을 정의하고, 그 기울기를 계산하고 다른 경우(예: 쌍, 불일치)에도 이를 확장합니다. 이러한 기울기 계산 및 확률 정의는 RNA 구조의 목표 함수에 대한 응용 확장성을 높이는 데 기여합니다.

- **Performance Highlights**: 본 연구는 Eterna100 벤치마크에서 기존의 최첨단 방법들과 비교하여 일관되게 높은 성능을 보였습니다. 특정 지표인 Boltzmann 확률(Boltzmann probability), 집합 결함(ensemble defect), 에너지 갭(energy gap)에서 특히 우수한 결과를 나타내며, 특히 긴 시퀀스와 설계하기 어려운 구조에도 효과적입니다. 연구팀은 이 코드를 오픈 소스로 제공하여 연구자들이 활용할 수 있도록 하고 있습니다.



### In-Context Learning with Topological Information for Knowledge Graph Completion (https://arxiv.org/abs/2412.08742)
- **What's New**: 이번 연구는 Knowledge Graph Completion(KGC) 분야에서 인-context learning 기법을 활용하여 성능을 향상시키는 새로운 접근법을 제안합니다. 특히, 대형 언어 모델(LLMs)의 사전 훈련된 지식을 통해 그래프의 토폴로지 정보와 온톨로지를 통합하여 빠진 정보를 추론하는 데 중점을 두고 있습니다. 이러한 접근법은 기존의 KGC 방법들에 비해 더욱 효과적인 성과를 보여줍니다.

- **Technical Details**: 제안된 방법은 두 단계로 구성됩니다. 첫 번째 단계에서는 LLM의 도메인 이해를 이용해 Knowledge Graph에서 온톨로지를 구축하여 그래프 내 노드의 타입과 관계를 포착합니다. 두 번째 단계에서는 그래프의 구조화된 정보를 활용하여, 사라진 지식 트리플과 기존 그래프 트리플 간의 중복 노드를 이용해 후보 솔루션을 생성하며, 이 과정에서 복잡한 그래프의 토폴로지 구조를 이용합니다.

- **Performance Highlights**: 실험 결과, ILPC-small 및 ILPC-large 데이터셋에서 제안된 방법이 기존의 최첨단 기준 모델들에 비해 월등히 높은 성능을 나타내었습니다. 이러한 성과는 전이학습(transductive) 및 유도(inductive) 환경 모두에서 관찰되었으며, 추가적인 훈련 없이도 높은 효율성을 기록하였습니다.



### VEL: A Formally Verified Reasoner for OWL2 EL Prof (https://arxiv.org/abs/2412.08739)
- **What's New**: 이번 연구에서는 기존 OWL의 한계를 보완하기 위해 VEL이라는 기계 검증 가능성을 갖춘 ℰℒ++ (EL++) 추론기를 개발했습니다. 이 추론기는 입력에 대한 유효성 보장을 보장하는 머신-체크 가능(correctness proofs) 정확성 증명을 기반으로 하고 있습니다. VEL은 Baader et al.의 알고리즘을 기초로 하며, OCaml 코드로 실행 가능하게 변환되었습니다.

- **Technical Details**: VEL의 형식화는 Baader et al.의 알고리즘에 기반하고 있으며, Coq 증명 보조기의 추출 기능을 통해 실행 가능한 코드로 구현되었습니다. 연구 중에 발견된 원래 적절성 증명의 두 가지 오류는 알고리즘 변경으로 이어졌습니다. 또한, ℰℒ++ (EL++) 기술적 어휘를 활용하여, 보다 정교한 추론 알고리즘을 기계화하였습니다.

- **Performance Highlights**: 공식적으로 검증된 알고리즘을 기반으로 한 VEL은 높은 정확성과 성능을 자랑하며, 명확한 공급망 증명을 통해 의료와 같은 중요한 분야에서 신뢰할 수 있는 결과를 보장합니다. 기존의 추론기들과의 비교를 통해, VEL이 기계적 구현과 이론적 정합성을 모두 충족하는 최초의 사례임을 입증하였습니다.



### Euclid: Supercharging Multimodal LLMs with Synthetic High-Fidelity Visual Descriptions (https://arxiv.org/abs/2412.08737)
Comments:
          33 pages, 22 figures, 5 tables, 7 algorithms

- **What's New**: 최근 멀티모달 대형 언어 모델(MLLMs)은 비약적인 발전을 이루었지만, 여전히 저수준 시각 인식(LLVP)에서는 어려움을 겪고 있습니다. 본 논문에서는 'Geoperception'이라는 벤치마크를 소개하여 MLLM의 2D 기하학적 정보 전사 능력을 평가합니다. 이 벤치마크를 통해 기존 MLLM의 한계를 드러내고, 기하학적 작업 성능을 향상시키기 위한 전략을 연구하게 됩니다.

- **Technical Details**: Geoperception 벤치마크에서는 MLLM들이 이미지에서 기하학적 세부사항을 정확하게 설명하는 능력을 평가합니다. 연구 결과, 특정 모델 아키텍처, 훈련 기법, 데이터 전략이 기하학적 작업을 향상시키는 데 도움이 된다는 것을 발견했습니다. 특히, 데이터 커리큘럼(data curriculum)을 통해 모델은 처음부터 학습하지 못하는 어려운 기하학적 이해 작업을 학습할 수 있음을 보여줍니다.

- **Performance Highlights**: Euclid이라는 모델 계열이 강력한 저수준 기하학적 인식을 위해 최적화되었습니다. Euclid는 순전히 합성된 멀티모달 데이터로 훈련되었음에도 불구하고, 새로운 기하학적 형태에 대한 일반화 능력이 뛰어난 성능을 보여줍니다. 예를 들어, Euclid는 Geoperception 벤치마크의 특정 작업에서 최고의 비공식 모델인 Gemini-1.5-Pro보다 최대 58.56% 향상된 성능을 기록하고, 전체 작업 평균으로는 10.65% 더 나은 성과를 보입니다.



### From MLP to NeoMLP: Leveraging Self-Attention for Neural Fields (https://arxiv.org/abs/2412.08731)
Comments:
          Preprint. Source code: this https URL

- **What's New**: 최근 Neural fields (NeFs)은 다양한 모달리티의 시공간 신호를 인코딩하는 혁신적인 방법으로 떠오르고 있습니다. 그러나 NeFs는 파라미터 공간의 복잡성과 기본 대칭성으로 인해 분류(classification)나 분할(segmentation)과 같은 다운스트림 작업에서 효과적으로 사용되기 어려운 문제점이 있었습니다. 본 연구에서는 연결주의(connectionism) 원리를 바탕으로 MLP(다층 퍼셉트론) 기반의 새로운 아키텍처, 즉 NeoMLP를 설계하였습니다.

- **Technical Details**: NeoMLP는 MLP를 그래프(graph)로 보고, 이를 멀티 파르티트 그래프(multi-partite graph)에서 입력, 숨겨진(hidden), 출력 노드로 구성된 완전 그래프(complete graph)로 변환합니다. 이 과정에서 고차원 특징(high-dimensional features)을 활용하여 메시지 전송(message passing)을 수행하고, 모든 노드 간의 자기 주의(self-attention)를 통해 가중치 공유(weight-sharing)를 이루어냅니다. 또한 NeoMLP는 숨겨진 및 출력 노드를 통해 조건화(conditioning) 메커니즘을 내장하고 있어, 이는 잠재 코드(latent codes)의 집합으로 작용할 수 있습니다.

- **Performance Highlights**: 높은 해상도의 신호를 나타내는 데이터에 대해 본 방법의 효과를 입증하였으며, 다중 모달 오디오-비주얼 데이터(multimodal audio-visual data)를 포함한 다양한 데이터셋에 대한 고도화된 적합(fitting)을 성공적으로 수행하였습니다. 또한 단일 백본 아키텍처(backbone architecture)를 통해 인스턴스 별(latent codes) 코드를 학습하여 다운스트림 작업에서 최신의 방법들을 능가하는 성능을 보였습니다. 이 연구의 소스 코드는 오픈 소스로 제공됩니다.



### A quantum-classical reinforcement learning model to play Atari games (https://arxiv.org/abs/2412.08725)
Comments:
          10 + 13 pages

- **What's New**: 최근 강화학습(Reinforcement Learning) 분야에서 파라미터화된 양자 회로(parametrized quantum circuits)를 기반으로 한 양자 학습 모델의 가능성이 보여지고 있습니다. 이 연구는 Atari 게임과 같은 복잡한 문제에서도 양자 근접 강화학습(Quantum Reinforcement Learning) 기술을 적용할 수 있는 가능성을 모색합니다. 새로운 하이브리드 모델(hybrid model)을 제안하여 고전적인 특성 인코딩(classical feature encoding) 및 후처리(post-processing) 레이어와 결합했습니다.

- **Technical Details**: 제안된 하이브리드 모델은 고전 모델(classical model)과 유사한 구조적 제약을 받으며, 이를 통해 Pong 환경을 해결하고 Breakout 게임에서 고전적인 참조 모델(reference model)과 유사한 점수를 달성했습니다. 또한, 연구 결과는 양자 구성 요소(quantum components)와 고전 구성 요소(classical components) 간의 상호작용에 중요한 영향을 미치는 하이퍼파라미터(hyperparameter) 설정과 설계 선택에 대한 통찰력을 제공합니다.

- **Performance Highlights**: 본 연구에서 개발된 하이브리드 모델은 Pong 환경을 성공적으로 해결한 결과를 보여주었으며, Breakout 게임에서도 고전 모델과 유사한 성능을 입증했습니다. 이러한 결과는 향후 양자 학습 모델의 실제 RL(강화학습) 시나리오에서의 활용 가능성을 높이고, 양자 및 고전적 방법의 조합을 통한 문제 해결의 새로운 길을 제시합니다.



### Learning Physics Informed Neural ODEs With Partial Measurements (https://arxiv.org/abs/2412.08681)
- **What's New**: 본 논문에서는 부분적으로 측정된 시스템의 동역학을 학습하기 위한 새로운 방법론을 제안합니다. 이는 측정되지 않는 상태를 포함한 동역학을 모델링하기 위해 선택적 최적화(framework) 기법을 활용합니다. 기존의 Neural ODE와 물리 지식 기반(Physics Informed) 접근 방식을 결합하여 데이터에서 직접 복잡한 상관관계를 학습합니다.

- **Technical Details**: 제안된 방법은 데이터에서 알 수 없는 동역학을 학습할 때 사용되며, 이 과정에서 파라메트릭 모델을 활용하고 물리학에 대한 지식을 손실(term)로 통합하여 모델 훈련을 보조합니다. 이를 통해 점진적이고 순차적인 최적화 기법을 도입했으며, 이는 상태와 모델 파라미터를 효과적으로 학습하는 데 기여합니다. 이 방법은 특히 측정이 불완전한 상황에서 효율적인 NODE 학습을 가능하게 합니다.

- **Performance Highlights**: 제안된 접근법은 실제 데이터셋과 수치 시뮬레이션을 통해 성능을 입증하였으며, 기존의 기본 모델들과 비교해 개선된 성능을 보였습니다. 특히 물리학적 지식을 통합한 손실 함수를 통해 상태 추정과 모델 매개변수 재조정을 위한 성과를 얻었습니다. 이로 인해 대규모 모델에 적용 가능성이 커졌습니다.



### Distinguishing Scams and Fraud with Ensemble Learning (https://arxiv.org/abs/2412.08680)
- **What's New**: 본 논문은 LLM(대형 언어 모델) 기반 웹 챗봇이 스캠 방어를 위한 사용자 요청에 어떻게 대응하는지를 평가합니다. CFPB(소비자 금융 보호국) 불만 데이터베이스를 활용하여 스캠과 비스캠 사기를 구분하는 LLM 앙상블 접근방식을 개발하였으며, 초기 결과를 통해 LLM의 강점과 약점을 분석하였습니다.

- **Technical Details**: CFPB 데이터베이스에서 스캠과 사기를 구별하기 위해 300개의 complaints를 수집하고 라벨링하는 과정을 진행했습니다. 이 과정에서 각 complaint는 '스캠' 또는 '비스캠'으로 분류되었으며, LLM의 성능을 향상시키기 위해 사용자 정의 프롬프트를 반복적으로 개선하였습니다. 연구에서는 Gemini와 GPT-4를 주로 사용하여 이들 모델의 예측 정확도를 평가하였습니다.

- **Performance Highlights**: 결과적으로, 다양한 프롬프트를 사용하여 Gemini와 GPT-4의 앙상블 모델을 통해 스캠을 효과적으로 식별하는 성능을 확보하였습니다. 연구진은 LLM의 성능과 complaint 길이, 편집 내용, 회사 이름 사이의 관계에 대한 초기 관찰 결과를 보고하였으며, 향후 연구에서는 더 큰 스캠 내러티브 집합에서 이러한 관찰을 평가할 계획입니다.



### A Behavior Tree-inspired programming language for autonomous agents (https://arxiv.org/abs/2412.08654)
- **What's New**: 이번 논문에서는 자율 에이전트를 위한 함수형 프로그래밍 언어의 설계를 제안합니다. 이는 Behavior Trees (BTs)의 개념과 동기를 바탕으로 하며, BTs가 로봇공학 및 AI에서 에이전트의 행동을 설계하는 인기 있는 모델로 사용되고 있음을 강조합니다. BTs의 성장과 복잡성이 증가하면서 기존의 간단한 모델이 제한적이게 되었고, BTs의 기능성을 발전시켜 자체 프로그래밍 언어로 발달시키려는 방향으로 나아가고 있습니다.

- **Technical Details**: BT 모델은 자율 에이전트를 위한 보다 발전된 기능을 갖추기 위해 확장을 필요로 하며, 몇 가지 중요한 문제를 해결해야 한다고 제안합니다. 여기에는 'reactive' selection, 'monitoring' 안전 중요 조건, 그리고 액션 간 데이터 전송 등이 포함됩니다. 논문에서는 이러한 문제들이 복잡하며, 현재의 BT 접근 방식이 모듈성과 일치하게 처리하지 못한다고 지적합니다. 이를 해결하기 위해 우리는 모듈형 프로그래밍 기본 요소의 간단한 집합을 제시하고, 이들을 조합하여 복잡한 프로그램을 구축하는 방법을 보여줍니다.

- **Performance Highlights**: 우리는 BT에 영감을 받은 언어에 대한 전체 사양을 제시하고, 함수형 프로그래밍 언어 Haskell로 구현한 사례를 다룹니다. 또한, 복잡한 BT를 간단하고 명확한 프로그램으로 변환하는 예제를 통해 우리의 언어의 유용성을 입증합니다. 이 과정에서 BT의 모듈성과 반응성을 강조하며, 자율 에이전트 개발에 있어 새로운 가능성을 제시합니다.



### What AI evaluations for preventing catastrophic risks can and cannot do (https://arxiv.org/abs/2412.08653)
- **What's New**: 이번 연구에서는 AI 평가(AI evaluations)가 AI 시스템의 안전성을 보장하는 데 있어 어떤 유용성과 한계를 지니고 있는지에 대해 다루고 있습니다. 평가들은 AI 능력을 평가하고, 특정한 오용 리스크(misuse risks)를 식별하는 데 중요한 역할을 하지만, 근본적인 한계로 인해 AI 안전성을 확보하기 위한 주요 수단으로 의존해서는 안 된다고 주장합니다. 평가의 결과는 미래의 AI 모델 능력을 예측하거나 자율 AI 시스템의 위험을 안정적으로 평가하는 데 필요한 상위 경계를 설정할 수 없습니다.

- **Technical Details**: 연구에서는 AI 평가의 두 가지 주요 차원, 즉 시기(timing)와 리스크 유형(risk type)을 기준으로 AI 평가의 효용과 한계를 분석합니다. 특히, 현재의 모델 능력 평가와 미래 모델 능력 예측을 분리하며, 인간의 오용 위험과 AI 자율 행동에 따른 위험을 명확히 구분합니다. 이러한 분석을 통해 AI 평가가 AI 안전성에 기여하는 방식과 그 내재된 한계를 제시합니다.

- **Performance Highlights**: AI 평가는 AI 시스템의 현재 능력을 명확히 보여주는 강력한 증거를 제공합니다. 예를 들어, 사이버 보안 관련 문제를 해결하는 AI의 능력이 평가를 통해 입증될 수 있으며, 이를 기반으로 안전 조치를 판단하는 데 활용될 수 있습니다. 그러나 이러한 하한선(lower bounds)은 필수적인 안전 요구 사항을 규명하는 데 유용하지만, 그것만으로는 충분하지 않다는 점을 강조합니다.



### A Mathematical Framework for Consciousness in Neural Networks (https://arxiv.org/abs/1704.01148)
- **What's New**: 이 논문은 의식(consciousness)과 그 물리적 상관관계 사이의 설명적 간극(explanatory gap)을 연결하기 위한 새로운 수학적 프레임워크를 제안합니다. 특히, 우리는 qualia가 신경망(neural network) 토폴로지의 수학적 표현에서 특이점(singularities)에 대응한다고 주장합니다. 이 접근 방식은 철학, 수학, 인지 신경과학(cognitive neuroscience) 및 인공지능(AI)의 통찰을 통합합니다.

- **Technical Details**: 우리는 qualia가 특이점(singularities)이라고 주장하지 않으며, 특이점이 qualia가 왜 그렇게 느껴지는지를 "설명한다"고도 주장하지 않습니다. 대신, 특이점은 시스템 동역학(dynamics)의 순수 수치적(Numerical) 설명이 원리적으로 한계에 도달하는 지점을 알려주는 원칙적(principled), 좌표 불변(coordinate-invariant) 마커로 작용한다고 제안합니다. 이는 의식의 물리적 상관관계 모델에 이러한 비축약성(irreducibility)의 형식적 마커를 통합하여, 복잡성(complexity)이나 정보(information)로 환원되지 않는 qualia를 현상으로 인식하는 프레임워크를 마련합니다.

- **Performance Highlights**: 이론적으로 주로 구성된 이러한 통찰은 향후 AI 및 인공지능(consciousness)에 대한 연구의 발전 가능성을 열고 있습니다. 우리는 비축약적(topological features) 특성을 인식하고 활용하는 것이 점진적(scale-based) 개선을 넘어서는 데 중요한 열쇠가 될 수 있다고 제안합니다. 궁극적으로 이는 인공지능 일반(AGI) 및 인공지능 의식(AC) 연구로 나아가는 길을 제시합니다.



### Learning About Algorithm Auditing in Five Steps: Scaffolding How High School Youth Can Systematically and Critically Evaluate Machine Learning Applications (https://arxiv.org/abs/2412.06989)
- **What's New**: 이 연구는 젊은 세대가 AI/ML 시스템에 대해 비판적으로 평가할 수 있도록 지원하는 방법에 대한 문헌이 부족한 가운데, 알고리즘 감사를 통한 체계적인 평가 방안을 제시합니다. 연구진은 청소년들이 참여하는 워크샵을 통해 동료들이 설계한 TikTok 필터를 감사하는 과정을 다섯 단계로 나누어 설명하고, 이를 통해 코드와 시스템의 불투명한 작동 방식 및 그 외부적 영향을 이해하도록 돕고자 합니다.

- **Technical Details**: 연구는 기존의 알고리즘 감사 방법론을 기반으로, 청소년들이 알고리즘 감사를 통해 AI/ML 시스템을 평가하는 학습 활동을 설계하는 데 중점을 두며, 알고리즘 감사의 다섯 단계를 제안합니다. 감사를 위한 단계는 (1) 가설 개발, (2) 입력 생성, (3) 테스트 실행, (4) 데이터 분석, (5) 결과 보고 등으로 구성되며, 각 단계가 청소년의 참여와 이해를 증진시키기 위한 방법으로 연결됩니다.

- **Performance Highlights**: 이 연구의 핵심 성과는 청소년들이 알고리즘 감사에 참여하며 경험한 구체적인 사례와 그들이 각 단계에서 어떻게 상호작용했는지를 제시한 점입니다. 이를 통해 알고리즘 감사가 청소년들에게 AI/ML 시스템을 비판적으로 평가하고 개인의 경험과 연결할 수 있는 유용한 도구가 될 수 있음을 보여줍니다.




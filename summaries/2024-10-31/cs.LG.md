New uploads on arXiv(cs.CL)

### $100K or 100 Days: Trade-offs when Pre-Training with Academic Resources (https://arxiv.org/abs/2410.23261)
- **What's New**: 이 논문에서는 학술 연구자들이 자원 부족으로 인해 모델의 사전 훈련이 어렵다는 일반적인 가정을 의문시합니다. 연구자들의 컴퓨팅 자원을 조사하고, 이를 바탕으로 주어진 GPU에서 모델을 재현하는 데 필요한 시간을 실증적으로 측정합니다.

- **Technical Details**: 논문은 2000 GPU-시간을 투자하여 3000개 이상의 구성에서 성능을 최적화하고, 최적의 훈련 설정을 찾기 위해 여러 모델에 대한 벤치마크를 수행하였습니다. 예를 들어, Pythia-1B 모델은 원래 64개의 GPU로 3일 걸려 훈련되었지만, 4개의 GPU로 18일 만에 재현할 수 있음을 보여줍니다. 또한, 비용-편익 분석을 통해 제한된 예산 내에서 가장 효과적인 하드웨어 구성도 제안합니다.

- **Performance Highlights**: 이 연구는 현재 하드웨어와 최적화를 활용하여 사전 훈련을 위한 GPU 사용을 3배 줄일 수 있음을 보여줍니다. 즉, 저예산 환경에서도 더 큰 모델에 대한 훈련을 가능하게 하며, 더 많은 연구자들이 실험을 할 수 있도록 지원합니다.



### Evaluating Cultural and Social Awareness of LLM Web Agents (https://arxiv.org/abs/2410.23252)
Comments:
          Work in progress

- **What's New**: 이 논문에서는 LLM(대형 언어 모델)의 문화적 및 사회적 인식에 대한 평가 기준, CASA를 도입합니다. CASA는 온라인 쇼핑과 사회적 토론 포럼이라는 두 가지 웹 기반 작업을 통해 LLM의 감수성을 평가하는 새로운 벤치마크입니다.

- **Technical Details**: CASA(문화적 및 사회적 인식 평가)는 LLM 에이전트가 사용자의 문화적 및 사회적 비준수 쿼리를 탐지하고 적절히 반응할 수 있는 능력을 평가합니다. 두 가지 웹 기반 작업에 대한 평가 프레임워크를 개발하였습니다.

- **Performance Highlights**: 현재 LLM 에이전트는 웹 기반 환경에서 10% 미만의 문화적 인식 범위와 40% 이상의 비준수 비율을 보이며, 이는 비 에이전트 환경에서 더 나은 성능을 보여줍니다. 두 가지 방법(프롬프트 및 파인튜닝)을 사용하여 LLM의 성능을 개선할 수 있음을 발견했습니다.



### OS-ATLAS: A Foundation Action Model for Generalist GUI Agents (https://arxiv.org/abs/2410.23218)
- **What's New**: OS-Atlas는 GUI grounding과 Out-Of-Distribution (OOD) 작업에서 두각을 나타내는 기본 GUI 행동 모델을 개발하였습니다. 기존 상용 Vision-Language Models (VLMs)에 비해 개방형 VLM이 가진 성능 저하 문제를 해결하기 위한 혁신적인 데이터 및 모델링 기법을 도입하였습니다.

- **Technical Details**: 첫 번째로, OS-Atlas는 다중 플랫폼에서 GUI grounding 데이터를 자동 합성할 수 있는 도구 키트를 개발하여 개방형 데이터 수집을 혁신적으로 용이하게 하였습니다. 이 키트를 이용해 2.3 백만 개의 스크린샷과 1,300만 개 이상의 GUI 요소를 포함하는 가장 큰 규모의 다중 플랫폼 GUI grounding 데이터셋을 공개하였습니다. 또한, OS-Atlas는 GUI 작업을 위한 새로운 방식의 훈련 과정을 통해 높은 정확도를 자랑합니다.

- **Performance Highlights**: OS-Atlas는 모바일, 데스크톱, 웹을 포함한 세 개 플랫폼에서 여섯 가지 벤치마크에 대한 평가를 통해 기존의 최고 성능 모델들에 비해 상당한 성능 향상을 보여주었습니다. 이러한 결과는 OS-Atlas가 상용 VLM, 예를 들어 GPT-4o에 대한 개방형 대안으로서의 가능성을 지니고 있음을 나타냅니다.



### Reliability of Topic Modeling (https://arxiv.org/abs/2410.23186)
- **What's New**: 이 연구에서는 기존의 topic model(주제 모델) 신뢰성 평가 방법이 두 가지 널리 사용되는 주제 모델에서 중요한 변화를 포착하지 못함을 보여줍니다. McDonald의 ω(오메가)라는 새로운 메트릭이 topic model 방법론의 검증을 위한 보다 정교한 도구임을 입증합니다.

- **Technical Details**: 기존의 topic model 신뢰성 측정은 주로 여러 차원의 구조를 고려하지 않고 단순히 주제의 일치 비율을 계산하는 팀을 포함합니다. 이 연구에서는 주제 모델에서 파생된 top word(상위 단어)의 신뢰성과 문서 및 주제 간 분포의 신뢰성을 함께 평가하는 방법을 제안하고 McDonald's ω를 통해 이를 실증적으로 검증합니다.

- **Performance Highlights**: 우리는 합성 및 실제 데이터를 통해 McDonald의 ω가 주제 모델 신뢰성을 가장 잘 encapsulate(캡슐화)한다고 보여줍니다. 이 새로운 메트릭은 주제 모델 기반 연구의 표준 구성 요소가 되어야 할 중요한 도구를 제공합니다.



### SciPIP: An LLM-based Scientific Paper Idea Proposer (https://arxiv.org/abs/2410.23166)
Comments:
          25 pages, 5 figures, 19 tables

- **What's New**: 이 논문은 과학 논문 아이디어 제안기인 SciPIP를 제안합니다. SciPIP는 사용자 제공 연구 배경을 바탕으로 유용한 논문을 검색하고, 이를 바탕으로 더 새롭고 실행 가능한 아이디어를 생성하는 방식으로 기존의 대형 언어 모델(LLM)의 잠재력을 활용합니다.

- **Technical Details**: SciPIP는 사용자가 제공한 연구 배경을 기반으로 문헌을 검색하여 아이디어를 제안하는 시스템입니다. 이를 위해 문헌 검색 데이터베이스를 구축하고, 의미론(semantic), 엔티티(entity), 인용의 공동 출현(citation co-occurrence)을 기반으로 문헌 검색을 수행합니다. 이후에는 문헌에서 얻은 정보를 활용하여 솔루션을 유추하거나 독창적인 아이디어를 생성하는 두 가지 경로를 통해 아이디어를 제안합니다.

- **Performance Highlights**: NLP 분야에서 진행된 광범위한 실험을 통해 SciPIP는 기존의 상위 회의 논문들과 유사한 인용을 검색하고, 많은 아이디어를 생성함으로써 그 효과를 입증하였습니다. 또한 SciPIP에 의해 생성된 아이디어는 청사진의 참조를 유지하면서도 혁신성과 실행 가능성을 확보하고 있음을 평가해 보여줍니다.



### The Good, the Bad, and the Ugly: The Role of AI Quality Disclosure in Lie Detection (https://arxiv.org/abs/2410.23143)
Comments:
          Order of the authors are in alphabetical order of their last names. All authors contributed equally. The manuscript is under review. 74 Pages, including appendices and references

- **What's New**: 본 논문은 낮은 품질의 AI 어드바이저가 품질 공시 없이 어떻게 텍스트 기반 거짓말을 퍼뜨릴 수 있는지를 조사합니다. 실험에 참여한 참가자들은 게임 쇼의 전사를 평가하며 진실과 거짓을 구별하는 작업을 수행했으며, 낮은 품질의 어드바이저를 의존할 때 진실 감지 능력이 개인의 능력 아래로 떨어지는 경향을 발견했습니다. 반면, 품질 높은 어드바이저는 공시 여부와 관계없이 진실 감지를 향상시킵니다.

- **Technical Details**: 우리는 AI 어드바이저의 품질을 여러 수준(낮음, 보통, 높음)으로 설정하고, 참가자들은 AI의 효과성이 공개된 환경과 비공개된 환경에서 각각 진실을 감지하도록 실험을 진행했습니다. 본 연구는 AI 어드바이저의 품질 스펙트럼과 그 효과성의 (비)공식 공시가 참가자의 AI 의존도에 어떻게 영향을 미치는지를 조사합니다.

- **Performance Highlights**: 연구 결과, 낮은 품질의 AI 어드바이저에 대한 의존은 참가자의 진실 감지 능력을 저하시켰으며, 이로 인해 거짓 정보의 확산이 우려됩니다. 반면, 높은 품질의 AI 어드바이저는 참가자들이 진실을 감지할 수 있는 능력을 전반적으로 향상시켰습니다.



### Crowdsourcing Lexical Diversity (https://arxiv.org/abs/2410.23133)
- **What's New**: 이 논문은 언어 간의 전이 문제와 문화적 편향을 줄이기 위한 새로운 크라우드소싱 방법론을 제안합니다. 연구자들은 언어의 다양성이 풍부한 domains인 친족 관계와 음식과 관련된 lexemes를 비교합니다.

- **Technical Details**: Crowd workers는 LSRs (Lexical-Semantic Resources)에서의 lexical gaps를 식별하는 마이크로 태스크를 통해 두 언어 (예: 영어와 아랍어, 표준 인도네시아어와 반자레어) 간의 동등한 용어를 비교합니다. 실험은 2,140개의 lexical gaps (영어-아랍어)와 951개의 (인도네시아어-반자레어)를 식별했습니다.

- **Performance Highlights**: 우리의 연구는 crowdsourcing이 LLMs (Large Language Models)보다 낮은 자원 언어에서 언어 및 문화 특정 개념을 식별하는 데 있어 유의미하게 우수하다는 결과를 보여주었습니다.



### On Memorization of Large Language Models in Logical Reasoning (https://arxiv.org/abs/2410.23123)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 추론 능력에 대한 메커니즘을 이해하기 위한 새로운 연구 결과를 제시합니다. 특히, LLM이 문제를 해결하는 데 있어 기억(memorization)과 진정한 추론 능력이 어떻게 상호작용하는지를 체계적으로 분석합니다.

- **Technical Details**: 연구진은 Knights and Knaves (K&K) 퍼즐 기반의 동적으로 생성된 논리 추론 벤치마크를 이용하여 LLM의 기억 행동을 정량적으로 측정하였습니다. 본 논문은 LLM이 훈련 퍼즐에 대해 높은 정확성을 달성할 수 있지만, 퍼즐이 약간 변형되면 실패하는 양상을 통해 기억에 크게 의존하고 있음을 밝혔습니다. 또한, fine-tuning은 기억력을 증가시키지만 일반화 성능(Generalization Performance)도 지속적으로 향상시키는 경향이 있음을 보여줍니다.

- **Performance Highlights**: 연구 결과에 따르면, 고급 LLM들은 K&K 퍼즐을 잘 해결할 수 있으며, 기억력이 증가함에 따라 일반화 정확도도 개선되는 것으로 나타났습니다. 다양한 분석을 통해 모델들이 훈련된 데이터의 기억에도 불구하고 진정한 추론 능력을 개발하는 방법을 확인하였으며, 이는 메모리와 추론 능력 사이의 복잡한 상호작용을 나타냅니다.



### Teaching a Language Model to Distinguish Between Similar Details using a Small Adversarial Training S (https://arxiv.org/abs/2410.23118)
- **What's New**: 이 논문에서는 언어 모델이 자연어 작업(Natural Language Tasks)에서 높은 정확도를 달성할 수 있지만, 수동으로 생성된 적대적 사례(Adversarial Examples)에서는 성능이 떨어진다는 점을 다룹니다.

- **Technical Details**: 저자들은 Stanford Natural Language Inference (SNLI) 데이터 세트에서 훈련된 언어 모델의 성능을 수동으로 생성된 적대적 테스트 세트에서 조사하였습니다. 이 모델은 비슷한 단어와 구문을 구별하도록 돕기 위해 소규모의 수동으로 생성된 적대적 훈련 세트로 미세 조정(Fine Tuning) 되었습니다.

- **Performance Highlights**: 적대적 테스트 세트에서 정확도가 13% 증가하였으며, 원래 NLI 작업에서는 여전히 좋은 성능을 유지하였습니다. 또한, SNLI 테스트 세트에서 가장 유사한 모순(Cosine Similarity 기준)에서 정확도가 91.2%에서 92.9%로 증가한 것을 보여주었습니다.



### Comparative Analysis of Demonstration Selection Algorithms for LLM In-Context Learning (https://arxiv.org/abs/2410.23099)
Comments:
          6 pages, 4 figures

- **What's New**: 이 논문은 다양한 demonstration selection 알고리즘을 분석하고, 실험을 통해 이들 알고리즘의 효율성과 효과를 평가합니다. 특히, 무작위 선택이 특정 상황에서 오히려 나은 결과를 초래할 수 있음을 보여줍니다.

- **Technical Details**: 연구에서는 여섯 가지 demonstration selection 알고리즘(CBDS, RD-direct, RD-channel, LLM Retriever, UPRISE, OpenICL TopK)을 비교하였습니다. 각 알고리즘은 특정 전략을 사용하여 LLM의 성능을 개선하는 데 중점을 두었으며, 이 과정에서 Bayesian 접근법, 순차적 예제 검색, 교차 인과 모델 등을 활용했습니다.

- **Performance Highlights**: 경험적인 연구 결과, 알고리즘 간 성능 차이가 크며, 같은 데이터셋에서도 정확도 차이가 45%까지 발생할 수 있습니다. 또한, 시연의 수 증가가 항상 향상된 성능으로 이어지지 않으며, 정확성과 처리 효율성 간에는 트레이드오프가 존재한다는 점을 발견했습니다.



### BUZZ: Beehive-structured Sparse KV Cache with Segmented Heavy Hitters for Efficient LLM Inferenc (https://arxiv.org/abs/2410.23079)
- **What's New**: BUZZ는 기존의 KV 캐시 메커니즘의 단점을 극복하기 위해 제안된 새로운 알고리즘으로, 구조화된 맥락 정보를 이용해 캐시 메모리 사용량을 최소화하고 추론 속도를 향상시킵니다.

- **Technical Details**: BUZZ 알고리즘은 벌집 구조의 희소 캐시를 사용하며, 슬라이딩 윈도우를 통해 최근 정보를 포착하고 역사적 토큰을 청크로 동적으로 분할하여 중요한 토큰에 우선순위를 부여합니다. 이 알고리즘은 O(n)의 시간 복잡도를 가지며, 인간의 기억 패턴을 모방하여 중요한 정보를 보다 효과적으로 유지합니다.

- **Performance Highlights**: BUZZ는 LLM 추론 시 캐시 메모리 사용량을 2.5배 줄이고, 긴 텍스트 요약에서 99% 이상의 정확도를 유지하며, 다중 문서 질문 응답에서 메모리 한계 내에서 7.69% 높은 성능을 보였습니다.



### Don't Just Pay Attention, PLANT It: Transfer L2R Models to Fine-tune Attention in Extreme Multi-Label Text Classification (https://arxiv.org/abs/2410.23066)
- **What's New**: 새로운 PLANT(Pretrained and Leveraged AtteNTion) 모델은 Extreme Multi-Label Text Classification (XMTC)에서 상태-of-the-art (SOTA) 성능을 달성하는 혁신적인 전이 학습 전략입니다. 기존 모델의 한계를 극복하고 특히 희귀 코드 처리에서 탁월한 성능을 보입니다.

- **Technical Details**: PLANT는 미리 훈련된 Learning-to-Rank (L2R) 모델을 주의(attention) 레이어로 활용하는 등의 기술 혁신을 포함합니다. 또한 상호 정보 이득(mutual-information gain)을 통합하여 주의를 강화하고, 주의가 필요 없는(inattention) 메커니즘과 상태 유지 디코더(stateful-decoder)를 구현하여 문맥을 유지합니다.

- **Performance Highlights**: PLANT는 약 50% 포인트 이상의 F1 점수 향상으로 MIMIC-III의 rare 및 few 데이터 세트에서 이전 few-shot 모델을 능가하며, 전통적인 모델 대비 적은 데이터로도 상당한 정확도를 달성합니다. 또한, EURLEX-4K와 WIKI10-31K 데이터 세트에서도 SOTA 성능을 기록했습니다.



### \textsc{Long$^2$RAG}: Evaluating Long-Context \& Long-Form Retrieval-Augmented Generation with Key Point Reca (https://arxiv.org/abs/2410.23000)
Comments:
          Our paper has been accepted for EMNLP 2024

- **What's New**: 본 연구에서는 Retrieval-augmented generation (RAG) 시스템의 평가 방법론에서의 부족한 점을 극복하기 위해 	extsc{Long$^2$RAG} 벤치마크와 Key Point Recall (KPR) 메트릭을 제안합니다. 	extsc{Long$^2$RAG}는 10개의 분야를 아우르는 280개의 질문으로 구성되어 있으며, 각 질문은 평균 2,444 단어 길이를 가진 5개의 검색된 문서와 관련이 있습니다.

- **Technical Details**: KPR 메트릭은 모델이 생성한 응답에 검색된 문서에서 추출된 핵심 포인트를 얼마나 잘 반영하는지를 평가합니다. 이를 통해 LLM(large language model)이 검색된 정보를 활용하는 능력을 보다 정밀하게 측정합니다. 연구는 총 9개의 최첨단 LLM을 평가하였으며, 결과적으로 폐쇄형 모델이 더 우수한 성능을 보였고, 입력 문서의 길이가 증가함에 따라 모델의 성능이 전반적으로 저하된다는 점을 발견했습니다.

- **Performance Highlights**: 연구 결과, 작은 오픈소스 모델인 Phi-3-mini가 72B 매개변수를 가진 Qwen2보다 성능이 뛰어난 결과를 보였습니다. 표준 RAG 절차에서 검색된 문서를 잘라내는 방식은 정보 손실로 이어져 성능 저하를 초래하는 것으로 나타났습니다. 이 연구는 LLM이 검색된 정보를 활용하는 데 있어 다양한 측면에서 이해를 촉진할 것으로 기대됩니다.



### Bonafide at LegalLens 2024 Shared Task: Using Lightweight DeBERTa Based Encoder For Legal Violation Detection and Resolution (https://arxiv.org/abs/2410.22977)
- **What's New**: 이번 연구에서는 비구조적 텍스트 데이터에서 법률 위반을 감지하고, 이를 잠재적으로 영향을 받는 개인과 연관시키기 위해 두 가지 시스템인 Named Entity Resolution (NER)과 Natural Language Inference (NLI)를 소개합니다. 경량화된 DeBERTa 기반 인코더를 사용하여 LLM 기준선을 초월하는 성과를 올렸습니다.

- **Technical Details**: 제안된 NER 시스템은 LegalLens 도전의 Subtask A에서 60.01%의 F1 점수를 달성하였으며, 법률 위반 사항을 식별하는 데 중점을 두고 있습니다. NLI 시스템은 Subtask B에서 84.73%의 F1 점수를 달성하였으며, 이는 이미 해결된 집단 소송 사건과의 연관성을 찾아내는 데 초점을 맞췄습니다. 두 시스템 모두 GLiNER 아키텍처와 DeBERTaV3를 기반으로 하며, 사전 학습된 모델을 미세 조정하여 성능을 향상시키고 있습니다.

- **Performance Highlights**: NER 시스템은 LegalLens 리더보드에서 6위, NLI 시스템은 5위에 랭크되었습니다. NLI 데이터셋의 증분 증대 방법을 통해 인식 성능이 7.65% 향상되었으며, NER과 NLI의 합쳐진 시스템을 통해 법률 위반 사항의 효과적인 감지가 가능해졌습니다.



### Private Synthetic Text Generation with Diffusion Models (https://arxiv.org/abs/2410.22971)
- **What's New**: 본 논문은 확산 모델(difusion models)이 합성 텍스트(synthetic texts) 생성을 위한 성능을 평가하고, 차등 개인 정보 보호(differential privacy, DP) 하에서의 합성 데이터 생성의 유효성을 검토합니다. 또한, 이전 연구에서의 가정을 재검토하여 차등 개인 정보 보호 보장을 위반할 수 있는 요소를 밝힙니다.

- **Technical Details**: 이 연구에서는 3개의 최신 확산 모델을 사용하여 다양한 차등 개인 정보 보호 강도를 가진 합성 텍스트 생성을 평가하였습니다. 또한, 기존 연구들이 공공 데이터셋에 의존하여 개인 정보 보호를 고려하지 않았음을 지적하며, 새로운 데이터셋을 도입하여 실험의 효용성을 증가시켰습니다. 차등 개인 정보 보호 개념에 대해 Dwork(2006)의 접근법을 기반으로 설명하고, Abadi et al.(2016)의 DP-SGD를 통해 신경망 학습 과정에 적용하는 방법을 제시합니다.

- **Performance Highlights**: 실험 결과, 공개 소스 LLMs가 차등 개인 정보 보호 환경에서 확산 모델보다 뛰어난 성능을 보인다는 것을 발견했습니다. 이는 이전 연구 결과들과는 반대되는 결과로, 확산 모델이 차등 개인 정보 보호 훈련 하에서는 성능이 저하된다는 것을 보여줍니다. 이 연구는 투명성(Transparency), 재현성(Reproducibility), 책임성(Accountability)에서 높아진 기준을 제시하며, 향후 연구에 기여할 수 있는 데이터셋 및 코드베이스를 공개했습니다.



### Multi-Agent Large Language Models for Conversational Task-Solving (https://arxiv.org/abs/2410.22932)
- **What's New**: 본 연구는 대화적 문제 해결에서 다수의 대리인이 활용되는 가능성과 도전 과제를 체계적으로 평가하여, 기존의 단일 대규모 언어 모델의 한계를 보완하는 방식으로 새로운 프레임워크를 제안합니다.

- **Technical Details**: MALLM(다중 에이전트 LLM)는 대화형 문제 해결을 위한 인간 상호작용을 시뮬레이션하는 프레임워크로, 생성 작업(예: 요약, 번역) 및 QA 작업(예: 선택적 윤리적 QA, 추출형 QA)에 대해 평가합니다. 논의의 수렴, 개별 에이전트의 영향, 대화 패러다임을 조사합니다.

- **Performance Highlights**: 다중 에이전트 시스템은 복잡한 추론 작업에서 우수한 성능을 보였으나, 기본 작업에서는 실패했습니다. 세 가지 주요 문제(문제 확산, 정렬 붕괴, 독점화)를 식별하였으며, 긴 대화가 도출하는 문제의 공정성에 대한 우려가 포함됩니다.



### Explainable Behavior Cloning: Teaching Large Language Model Agents through Learning by Demonstration (https://arxiv.org/abs/2410.22916)
Comments:
          20 pages

- **What's New**: 본 논문에서는 모바일 애플리케이션의 자율 상호작용을 위한 새로운 접근법인 Explainable Behavior Cloning LLM Agent (EBC-LLMAgent)를 제안합니다. 이 방법은 대형 언어 모델(LLMs)과 행동 복제를 결합하여 지능적이고 설명 가능한 에이전트를 생성합니다.

- **Technical Details**: EBC-LLMAgent는 세 가지 핵심 모듈인 Demonstration Encoding, Code Generation, UI Mapping으로 구성되어 있습니다. Demonstration Encoding 모듈은 사용자 시연을 캡처하여 LLM 에이전트가 처리할 수 있는 형식으로 구조화합니다. Code Generation 모듈은 인코딩된 시연을 모듈화되고 설명적인 코드 조각으로 변환합니다. UI Mapping 모듈은 생성된 코드 조각과 애플리케이션의 UI 요소 간의 정확한 대응을 설정하여 원활한 상호작용을 보장합니다.

- **Performance Highlights**: 다양한 도메인의 인기 있는 모바일 애플리케이션 5종에 대한 광범위한 실험을 통해 EBC-LLMAgent는 기본 방법에 비해 높은 작업 완료 성공률과 보이지 않는 시나리오에 대한 효율적인 일반화를 달성하였으며, 의미 있는 설명도 생성했습니다.



### From Babble to Words: Pre-Training Language Models on Continuous Streams of Phonemes (https://arxiv.org/abs/2410.22906)
- **What's New**: 이 논문은 언어 모델(언어 모델들은 보통 대량의 텍스트 데이터를 사용하여 훈련됨)의 훈련에 있어 일반적인 표기법(orthographic representation) 대신 음소(phoneme)를 사용하여 훈련하는 방식을 제안합니다. 이를 통해 음소 기반 훈련이 제공하는 분석적 및 실용적 이점을 강조하고 있습니다.

- **Technical Details**: 이 논문에서는 100백만 단어로 구성된 BabyLM 챌린지의 데이터셋을 사용하여 음소 기반의 입력 표현을 이용한 모델 훈련 및 평가 방법론을 제시합니다. 구체적으로, 텍스트 데이터셋을 음소의 연속 스트림으로 변환하는 파이프라인을 개발하여, 이를 통해 표기법과 음소 표현을 모두 사용할 수 있게 하였습니다. 세 가지 주요 변환(transformations)을 통해 표기법 텍스트를 음소 표현으로 변환하였고, 각각의 변환이 모델의 성능에 미치는 영향을 분석했습니다.

- **Performance Highlights**: 그 결과, 음소 기반 훈련이 전통적인 언어 이해 작업에서 성능을 약간 감소시켰으나, 언어 모델이 음소 입력 표현에서 문법 지식을 학습할 수 있는 강력한 통계학적 학습자를 가지고 있음을 확인하였습니다. BabySLM 벤치마크에서 최상의 점수를 기록하며, 성능 저하가 기존 연구에서 주장하는 것보다 크지 않다는 것을 발견했습니다.



### Combining psychoanalysis and computer science: an empirical study of the relationship between emotions and the Lacanian discourses (https://arxiv.org/abs/2410.22895)
- **What's New**: 이 연구는 정신분석학(psychoanalysis)과 컴퓨터 과학(computer science) 간의 상호작용을 탐구하고, 양 분야 간의 상생관계를 제안합니다. 특히, 본 연구는 감정과 라캉(Lacan) 담론 간의 기본 관계를 수립하기 위해 컴퓨터 과학 기법을 적용하는 것을 목표로 합니다.

- **Technical Details**: 주요 과정은 다음과 같습니다: a) 주어진 텍스트에서 30개의 감정(emotions) 세트를 활용하여 감정을 식별한다; b) 텍스트에서 라캉 담론(Lacanian discourses)도 식별한다; c) 감정과 라캉 담론 간의 내재적 관계를 확인하기 위해 통계적 조사를 수행한다; d) 마지막으로 이러한 통계적 발견을 이론적으로 검증한다. 연구에서는 '라캉 담론 발견(Lacanian Discourse Discovery, LDD)'이라는 개념을 소개하여 담론 식별을 체계화합니다.

- **Performance Highlights**: 이 연구 방법은 AI 기술을 통해 텍스트에서 감정과 담론을 효과적으로 식별할 수 있는 자동화된 기법을 제공합니다. 이는 디지털 헬스 애플리케이션 및 가짜 뉴스 탐지 소프트웨어와 같은 다양한 디지털 시스템에 실질적인 응용 가능성을 제시합니다.



### Less is More: Pre-Training Cross-Lingual Small-Scale Language Models with Cognitively-Plausible Curriculum Learning Strategies (https://arxiv.org/abs/2410.22886)
Comments:
          BabyLM Shared Task 2024 (Accepted, Poster), co-located in EMNLP 2024

- **What's New**: 이 논문에서는 Curriculum Learning (CL) 전략을 통해 Small-Scale Language Models (SSLMs)의 성능을 향상시키기 위한 연구를 진행하였습니다. 특히, 아동 발화를 기반으로 한 연령 순서의 코퍼스를 활용하여 언어 획득 이론에 기반한 보다 세밀한 CL 전략을 제시하였습니다.

- **Technical Details**: 연구에서는 Growing, Inwards, MMM이라는 세 가지 목표 기반 커리큘럼을 개발하였으며, 이들은 아동이 언어 습득 초기 단계에서 따르는 것으로 이론화된 발달 순서를 기반으로 하고 있습니다. 이 커리큘럼들은 SSLMs의 Masked Language Modeling (MLM) 목표를 수정하여 다국적 언어 획득 이론을 시뮬레이션합니다.

- **Performance Highlights**: 본 연구에서 제안된 커리큘럼은 비 커리큘럼 기반의 모델보다 우수한 성능을 보였으며, 특정 언어에 대해서는 더 정교한 언어 특정 MMM 버전이 더 나은 성능을 나타냈습니다. SSLMs는 LLMs에 비해 약 25배 적은 매개변수와 6,000배 적은 단어로 유사한 성능을 달성했습니다.



### Eliciting Critical Reasoning in Retrieval-Augmented Language Models via Contrastive Explanations (https://arxiv.org/abs/2410.22874)
- **What's New**: 이번 논문에서는 Retrieval-augmented generation (RAG) 방법론의 새로운 접근 방식인 Contrastive-RAG (C-RAG)를 제안합니다. C-RAG는 LLMs가 RAG 기반 정보를 비판적으로 분석할 수 있도록 도와줍니다.

- **Technical Details**: C-RAG는 네 가지 단계로 구성됩니다: (i) 수집 단계, (ii) 대조적 추론 단계, (iii) 설명 단계, (iv) 답변 단계입니다. 이 과정에 따라 LLMs가 쿼리와 관련된 문서를 검색하고, relevantes와 irrelevants한 내용을 비교하여 설명을 생성하며, 최종 답변을 도출합니다.

- **Performance Highlights**: C-RAG를 사용한 실험 결과, 기존 RAG 모델에 비해 평균 정확도가 55.4% 향상되었으며, Self-RAG 및 Self-Reasoning 모델에 비해 각각 7.2% 및 1.9%의 성능 향상이 있었습니다. 또한, C-RAG는 필요한 프롬프트 수와 훈련 예제가 현저히 적으면서도 더 나은 성능을 달성하였습니다.



### Danoliteracy of Generative, Large Language Models (https://arxiv.org/abs/2410.22839)
Comments:
          16 pages, 13 figures, submitted to: NoDaLiDa/Baltic-HLT 2025

- **What's New**: 이 연구는 덴마크어에 대한 Generative, Large Language Models (GLLMs)의 성능을 평가하기 위한 Danoliterate Benchmark를 소개합니다. 이 벤치마크는 덴마크어 및 문화적 역량을 측정하며, 다양한 시나리오에서 모델의 능력을 검증하는 새로운 데이터 세트를 제공합니다.

- **Technical Details**: 이 연구는 RWK (Real-World Knowledge), NLU (Natural Language Understanding), NLG (Natural Language Generation) 등 세 가지 주요 범주에서 덴마크어 GLLM의 성능을 평가합니다. 605개의 다지선다형 질문으로 구성된 Citizenship Test와 같은 다양한 시나리오를 포함하여 모델 성능을 비교 분석합니다. 벤치마크는 GPT-4와 Claude Opus 모델이 최고 성능을 기록한 사실을 입증합니다.

- **Performance Highlights**: 본 연구의 결과, GLLMs의 성능은 약 80% 정도의 인간 피드백과 상관관계를 나타내며, 하나의 주요 요소가 GLLMs의 협응력(ability consistency)을 설명하는 것으로 나타났습니다. 이는 Danoliteracy 평가에서 고른 성능 변화를 제공할 수 있음을 시사합니다.



### How Well Do Large Language Models Disambiguate Swedish Words? (https://arxiv.org/abs/2410.22827)
Comments:
          SLTC 2024 extended abstract

- **What's New**: 이 연구에서는 스웨덴어의 단어 의미 구별(word sense disambiguation, WSD)을 위한 최신 대형 언어 모델(large language models, LLMs)을 평가했습니다. 특히 WSD 작업에 대해 이 모델들이 어떻게 성능을 발휘하는지, 그리고 어떻게 가능한 의미를 표현하는지에 대한 다양한 프롬프트(prompts) 접근 방식을 비교했습니다.

- **Technical Details**: WSD 실험은 SALDO lexicon을 기반으로 진행되었습니다. SALDO는 스웨덴어 단어에 대한 대규모 의미 목록을 정의하며, 이를 통해 다양한 어휘 의미 자원 간에 연결고리 역할을 합니다. 본 연구는 SemTag 프로젝트와 SENSEVAL-2 등의 두 가지 데이터셋에 대해 LLMs의 성능을 평가했습니다.

- **Performance Highlights**: 대부분의 최신 모델은 감독된 해리기(discambiguators)에 비해 정확도가 떨어지지만, 그래프 기반 비감독 볼 수 있는 시스템보다 더 나은 성능을 보였습니다. 특히 인간이 작성한 정의를 포함한 프롬프트가 가장 높은 정확도를 달성했습니다.



### EvoCodeBench: An Evolving Code Generation Benchmark with Domain-Specific Evaluations (https://arxiv.org/abs/2410.22821)
Comments:
          Accepted by the 38th Conference on Neural Information Processing Systems (NeurIPS 2024)

- **What's New**: 본 논문은 코드 생성 분야에서의 대형 언어 모델(Large Language Models, LLMs) 평가를 위한 새로운 벤치마크인 EvoCodeBench를 제안합니다. EvoCodeBench는 데이터 유출(data leakage) 문제를 해결하고, 특정 도메인(domain-specific) 평가를 추가하여 기존 벤치마크의 한계를 극복하고자 합니다.

- **Technical Details**: EvoCodeBench는 동적으로 업데이트되는 데이터, 프로그래밍 도메인의 분류, 도메인 레이블과 같은 세 가지 주요 혁신점을 포함합니다. 벤치마크는 6개월마다 새로운 버전으로 갱신되며, 첫 번째 버전인 EvoCodeBench-2403는 25개의 리포지토리에서 275개의 샘플을 포함하고 있습니다. 연구자들은 8개의 인기 LLM(gpt-4, DeepSeek Coder 등)을 평가하고 Pass@k 및 Domain-Specific Improvement(DSI) 메트릭을 통해 LLM의 도메인 별 성과를 분석합니다.

- **Performance Highlights**: EvoCodeBench는 41.47%에서 2.18%로 데이터 유출율을 줄이며, 실제 개발 시나리오에서 LLM들의 성능을 반영합니다. gpt-4는 대부분의 도메인에서 가장 높은 성과를 보였지만, 인터넷 도메인에서는 상대적으로 저조한 성과를 기록했습니다. StarCoder 2-15B는 데이터베이스 도메인에서 예기치 않게 우수한 성능을 발휘하여 33B LLM보다 더 나은 성과를 나타냈습니다.



### MALoRA: Mixture of Asymmetric Low-Rank Adaptation for Enhanced Multi-Task Learning (https://arxiv.org/abs/2410.22782)
Comments:
          14 pages, 5 figures

- **What's New**: 본 논문에서는 Mixture of Asymmetric Low-Rank Adaptation (MALoRA)라는 새로운 유연한 파인 튜닝 프레임워크를 제안합니다. MALoRA는 LoRA 전문가들 간의 비대칭 최적화를 활용하여 파라미터의 수를 30%에서 48% 줄이고, 훈련 속도를 1.2배 증가시키며, 단일 작업 LoRA 모델과의 계산 효율성을 일치시킵니다.

- **Technical Details**: MALoRA는 공유 가능한 저랭크 서브스페이스(representing low-rank subspace)를 도입하여 내려가는 프로젝션 모듈(down-projection module)에서 파라미터 중복을 줄이고, 각 LoRA 전문가에게 긴축된 계수 행렬(compacted coefficient matrix)을 배정하여 효과적으로 파라미터 수와 계산 복잡성을 낮춥니다.

- **Performance Highlights**: MALoRA는 다양한 다중 작업 학습 시나리오에서 진행된 실험을 통해 MoLoRA의 성능을 뛰어넘는 결과를 지속적으로 보여주었으며, 효율성과 일반화에서 모든 기준 방법들을 초월합니다.



### InjecGuard: Benchmarking and Mitigating Over-defense in Prompt Injection Guardrail Models (https://arxiv.org/abs/2410.22770)
- **What's New**: 이번 논문에서는 Prompt injection 공격으로부터 대형 언어 모델(LLM)을 방어하기 위한 새로운 평가 데이터셋 NotInject를 소개합니다. NotInject는 다양한 프롬프트 가드 모델의 과도한 방어(over-defense) 문제를 체계적으로 측정할 수 있도록 설계되었습니다.

- **Technical Details**: NotInject 데이터셋은 339개의 benign 샘플로 구성되며, 이들은 프롬프트 인젝션 공격에서 자주 사용되는 trigger words를 포함하고 있습니다. InjecGuard라는 새로운 프롬프트 가드 모델은 Mitigating Over-defense for Free (MOF)라는 새로운 훈련 전략을 통해 trigger word에 대한 편향을 크게 줄입니다.

- **Performance Highlights**: InjecGuard는 NotInject를 포함한 다양한 벤치마크에서 state-of-the-art 성능을 입증하였으며, 기존 최고의 모델보다 30.8% 개선된 83% 이상의 평균 정확도로 benign, malicious 및 over-defense 입력을 탐지합니다.



### Beyond Ontology in Dialogue State Tracking for Goal-Oriented Chatbo (https://arxiv.org/abs/2410.22767)
Comments:
          There are 10 chapters, including references, and 2 figures used. To be presented at the 15th IEEE International Conference on Knowledge Graphs (ICKG2024)

- **What's New**: 이 논문에서는 고전적인 대화 상태 추적( DST ) 방법의 한계를 극복하기 위한 새로운 접근법을 제안합니다. 고정된 온톨로지(ontology) 및 수동으로 수집된 슬롯 값을 사용하지 않고도 LLM( 대형 언어 모델 )이 대화 상태를 추론할 수 있도록 하는 방법을 소개합니다.

- **Technical Details**: 제안된 방법은 명령 조정(instruction tuning) 및 고급 프롬프트 전략을 활용하여 열린 도메인 대화에 대한 유연성을 제공합니다. 또한, VGAE(변종 그래프 오토인코더)를 사용하여 사용자 의도를 모델링하고 예측하는 과정도 포함되어 있습니다.

- **Performance Highlights**: 본 논문에서 제안한 방법은 새로운 JGA 42.57%를 달성하며 기존의 온톨로지 없는 DST 모델들을 초월하고 실제 열린 도메인 대화 데이터에서 우수한 성능을 보였습니다.



### Constructing Multimodal Datasets from Scratch for Rapid Development of a Japanese Visual Language Mod (https://arxiv.org/abs/2410.22736)
Comments:
          15 pages, 7 figures

- **What's New**: 이 연구에서는 일본어를 위한 새로운 고성능 Visual Language Model (VLM) 생성을 위해 일본어 멀티모달 데이터세트를 신속하게 구축하는 방법을 제안합니다. 이로 인해 영어 자료에 비해 부족했던 일본어 멀티모달 리소스를 확보하게 되었습니다.

- **Technical Details**: 본 연구에서는 일본어 이미지-텍스트 쌍 및 인터리브드 데이터(interleaved data)를 웹 아카이브에서 수집하고, 기존 VLM을 이용하여 직접적으로 일본어 명령 데이터(instruction data)를 생성합니다. 실험 결과는 본 연구의 데이터세트를 사용하여 훈련된 VLM이 기계 번역 데이터에 의존한 모델보다 더 높은 성능을 나타냄을 보여줍니다.

- **Performance Highlights**: 제안된 데이터세트를 기반으로 훈련된 모델은 주로 기계 번역 데이터로 개발된 모델에 비해 특히 명령 데이터 측면에서 더 높은 정확도를 기록했습니다.



### Linguistics Theory Meets LLM: Code-Switched Text Generation via Equivalence Constrained Large Language Models (https://arxiv.org/abs/2410.22660)
- **What's New**: 이 논문에서는 코드 스위칭(code-switching)을 자연스럽고도 언어학적으로 유효한 방식으로 생성하기 위한 새로운 프레임워크인 EZSwitch를 제안합니다. 이 프레임워크는 Equivalence Constraint Theory (ECT)와 대형 언어 모델(LLMs)을 결합하여 기존의 방법보다 더 향상된 퍼포먼스를 보여줍니다.

- **Technical Details**: EZSwitch는 LLM에 ECT를 통합하여 코드 스위칭 텍스트를 생성합니다. 이를 위해 1) 번역 수집, 2) 이중 텍스트 정렬 및 3) 코드 스위칭 문장 생성을 포함하는 세 가지 단계가 포함됩니다. ECT 이론은 코드 스위칭이 두 언어의 문법 구조가 일치할 때만 허용됨을 명시합니다. 이 이론을 바탕으로, 이전의 연구에서 소개된 유연한 ECT 버전을 사용하여 코드 스위칭 패턴을 더 잘 식별할 수 있습니다.

- **Performance Highlights**: EZSwitch로 생성된 코드 스위칭 문장은 기존의 LLM들과 비교하여 유의미한 품질 향상을 보여줍니다. 또한, CSPref라는 인간 선호 데이터셋을 생성하여 향후 연구와 평가에 기여할 수 있도록 하였습니다.



### Prove Your Point!: Bringing Proof-Enhancement Principles to Argumentative Essay Generation (https://arxiv.org/abs/2410.22642)
Comments:
          EMNLP 2024

- **What's New**: 이 논문에서는 논리적 개선(logical enhancement)에 중점을 두고, 주장을 강화(proof-enhancement)하고 스스로 주석을 달도록(self-annotation) 하는 두 단계 프레임워크인 PESA를 제안합니다. 기존의 주장을 단순히 생성하는 방법의 한계를 극복하고, 주장 간의 고차원 연결을 강화하여 논리적인 일관성을 높입니다.

- **Technical Details**: PESA는 두 가지 주요 단계를 포함합니다: 증거와 근거(claims and grounds)를 위한 가짜 레이블 생성과, 증거 원칙을 도입하여 논리적 흐름을 보장하는 트리 계획(tree planning) 접근 방식입니다. 이 방식은 Toulmin Argumentation Model을 차용하여, 구체적 근거와 추상적 주장을 연결하여 보다 설득력 있는 주장을 생성하도록 설계되었습니다.

- **Performance Highlights**: PESA는 기존의 강력한 기준 모델들보다 논리적 유효성과 설득력에서 우수한 성과(State-Of-Art)를 보여주며, 자동 평가 메트릭스에서 뛰어난 결과를 달성했습니다. 추가적으로, 인적 평가에서도 PESA가 유창성, 논리성 및 설득력 측면에서 현저히 우수하다는 것을 확인했습니다.



### Characterizing the Role of Similarity in the Property Inferences of Language Models (https://arxiv.org/abs/2410.22590)
- **What's New**: 본 연구는 언어 모델(LM)의 속성 상속(property inheritance) 행동을 조사하며, 높은 수준의 범주(예: 새)에서 낮은 수준의 범주(예: 참새)로 새로운 속성이 전달되는 방식을 분석합니다. 연구 결과, LMs는 분류학적(taxonomic) 관련성과 범주적 유사성(categorical similarity)이 환원되지 않은 비일관성을 보이며, 언어 모델이 속성 상속을 수행할 때 두 가지 요소 모두 중요하다는 것을 발견했습니다.

- **Technical Details**: 기존의 언어 모델 연구와는 달리, 본 연구는 LMs의 속성 상속 행동을 이해하기 위해 행동적 분석(behavioral analysis) 및 인과적 해석(causal interpretability) 방법을 결합하였습니다. 연구 과정에서 분배 정렬 검색(distributed alignment search, DAS)을 활용해 속성 상속에 기여하는 하위 공간(subspace)을 로컬라이즈했습니다. 이 방법은 LMs의 인식 구조가 분류학적 및 유사성 특성을 반영하고 있음을 지적합니다.

- **Performance Highlights**: 실험 결과, LMs의 속성 상속 판단은 명사 유사성(noun similarity)과 강한 상관관계를 보였으며, 이는 인간의 속성 상속 행동과 유사합니다. 또한 연구에서 통계적으로 의미 있는 방식으로 속성 상속에 대한 잘못된 긍정(false positive) 및 잘못된 부정(false negative)을 설명할 수 있는 결과를 나타냈습니다.



### Toxicity of the Commons: Curating Open-Source Pre-Training Data (https://arxiv.org/abs/2410.22587)
- **What's New**: 이 논문에서 우리는 공개 도메인 데이터를 기반으로 훈련된 언어 모델의 유해 출력을 줄이기 위한 데이터 선별 파이프라인을 제안합니다. 새로운 오픈 소스 파이프라인을 통해 유해 콘텐츠를 필터링하는 방법을 개발하였습니다.

- **Technical Details**: 우리는 유해 콘텐츠를 감지하기 위해 데이터셋 ToxicCommons를 만들고, 이를 기반으로 커스터마이즈된 분류기 Celadon을 훈련했습니다. 이 분류기는 공개 데이터에서 유해 내용을 더 효율적으로 대규모로 감지할 수 있도록 설계되었습니다.

- **Performance Highlights**: 제안된 방법은 제거가 필요한 문제적 콘텐츠를 높은 정확도로 식별하고, 저수준의 유해성을 가진 데이터를 유용한 데이터로 변형할 수 있는 방안을 제공합니다. 이로 인해 모델의 원치 않는 행동을 줄일 수 있는 가능성을 보여줍니다.



### Auto-Intent: Automated Intent Discovery and Self-Exploration for Large Language Model Web Agents (https://arxiv.org/abs/2410.22552)
Comments:
          EMNLP 2024 Findings

- **What's New**: 본 논문에서는 Auto-Intent를 소개하며, 이는 학습하여 구축한 대형 언어 모델(LLM)을 직접적인 fine-tuning 없이 특정 도메인에 맞게 조정하는 방법입니다. 이 방법은 웹 탐색 작업을 중심으로 경험적으로 연구되었습니다.

- **Technical Details**: Auto-Intent는 대상 도메인의 시연에서 기본적인 의도(intents)를 비지도학습(unsupervised) 방식으로 발견합니다. 이 의도들은 최대 3단어로 매우 간결하게 표현되며, 이를 통해 다음 의도를 예측하는 의도 예측기(intent predictor)를 학습합니다. 특히 self-exploration 접근 방식을 통해, 가장 가능성이 높은 top-k 의도 예측 결과를 LLM 에이전트에게 제공하여 의사결정 능력을 향상시킵니다.

- **Performance Highlights**: Auto-Intent는 Mind2Web의 대규모 실 웹 탐색 벤치마크(Task benchmarks)와 WebArena의 온라인 탐색 작업에서 GPT-{3.5, 4}와 Llama-3.1-{70B, 405B} 에이전트의 성능을 실질적으로 향상시켰습니다.



### Attention Speaks Volumes: Localizing and Mitigating Bias in Language Models (https://arxiv.org/abs/2410.22517)
- **What's New**: 본 논문은 큰 언어 모델(LLMs)에서 애매한 비교 프롬프트가 주어졌을 때 어떻게 편향(bias)이 발생하는지를 탐구합니다. 새로운 방법으로, 편향을 특정 레이어에 국한시키고 주의(attention) 점수를 분석하여 이를 완화할 수 있는 기법인 $	exttt{ATLAS}$를 제안합니다.

- **Technical Details**: $	exttt{ATLAS}$는 두 단계의 접근법으로, 첫 번째로 주의 점수를 분석하여 편향이 집중된 레이어를 식별하고, 두 번째로 이러한 편향 레이어에 대해 주의 점수를 조정하여 편향을 줄이는 것입니다. 실험은 다양한 모델(GPT-2 XL, GPT-J, LLaMA-2 및 LLaMA-3)과 데이터셋(BBQ, Crows-Pairs, WinoGender)을 사용하여 수행되었습니다.

- **Performance Highlights**: 우리의 실험 결과, 편향은 모델의 후반 레이어에 집중되어 있으며, $	exttt{ATLAS}$는 다운스트림 성능에 영향을 주지 않으면서 편향을 효과적으로 완화하였습니다. 평균적으로 0.28 포인트의 편향 점수 개선이 이루어졌습니다.



### Anticipating Future with Large Language Model for Simultaneous Machine Translation (https://arxiv.org/abs/2410.22499)
Comments:
          Under review

- **What's New**: 이번 연구에서는 Simultaneous Machine Translation (SMT) 기술에서의 번역 품질을 높이기 위해 Translation by Anticipating Future (TAF)라는 새로운 접근 방식을 제안합니다. 이 방법은 인간 통역사의 기법을 모방하여, 입력된 텍스트의 미래 단어를 예측하고 낮은 지연 시간을 유지하면서도 번역 성능을 극대화합니다.

- **Technical Details**: TAF는 대형 언어 모델(LLM)을 이용하여 원본 입력의 여러 가능한 연속성을 예측하고, 각 연속성을 번역하기 위해 기계 번역(MT) 모델을 활용합니다. 그런 다음, 다수결 투표 메커니즘을 통해 가장 많은 후보가 동의하는 접두사를 선택하여 원본 입력과의 일관성을 확보합니다. 이 방식은 추가적인 파인튜닝 없이 사용 가능한 다양한 MT 모델 및 LLM 조합에서 작동합니다.

- **Performance Highlights**: TAF는 4개의 언어 쌍에서 진행된 실험 결과, 기존 방법보다 최대 5 BLEU 포인트 향상된 번역 품질-지연 시간 균형을 달성했습니다. 특히, 3개의 단어 지연 시간에서 최고 성능을 기록하며, LLM에 더 긴 컨텍스트를 제공할 경우 번역 품질을 저하시키지 않고도 지연 시간을 더욱 줄일 수 있음을 보였습니다.



### Scaling LLM Inference with Optimized Sample Compute Allocation (https://arxiv.org/abs/2410.22480)
- **What's New**: 이 논문에서는 LLM(Large Language Model)의 샘플링 구성의 최적 분배를 찾기 위한 새로운 알고리즘 OSCA(Optimizes Sample Compute Allocation)를 제안합니다. OSCA는 다양한 샘플링 구성에 대한 샘플 예산을 최적화하여 더 효율적인 추론 성능을 보여줍니다.

- **Technical Details**: OSCA 알고리즘은 여러 샘플링 구성(모델, 온도 등)에 대해 각 구성에서 생성할 샘플의 수를 학습함으로써 최적의 예산 분배를 예측하는 방법으로 재미있는 방식에서 성능 향상을 가져옵니다. 특히, 작은 샘플 크기에서는 OSCA에서 학습된 할당이 순수 최적 할당보다 성능이 뛰어납니다.

- **Performance Highlights**: OSCA를 사용한 경우 코드 생성에서 128배, 4개의 이유(task) 과제에서 25배 적은 계산으로 더 높은 정확도를 달성했습니다. SWE-Bench에서 단일 턴 작업을 넘어 Agentic 워크플로우 개선에서도 우수한 성능을 보였습니다.



### A Pointer Network-based Approach for Joint Extraction and Detection of Multi-Label Multi-Class Intents (https://arxiv.org/abs/2410.22476)
Comments:
          Accepted at EMNLP 2024 Findings (Long Paper)

- **What's New**: 본 연구는 멀티레이블 멀티클래스 의도 탐지와 스팬 추출에 관한 새로운 접근 방식을 제시합니다. 최초로 멀티언어 멀티 의도 데이터셋(MLMCID-dataset)을 구축하고, 포인터 네트워크 기반 아키텍처를 통해 여러 개의 의도 스팬을 추출하며 다양한 레이블을 감지합니다.

- **Technical Details**: 연구에서는 포인터 네트워크 기반의 인코더-디코더 프레임워크를 구축하여 주어진 쿼리에서 여러 의도 스팬을 추출합니다. MLMCID 의도 탐지 모듈은 coarse(거칠게 분류된) 및 fine(세부적으로 분류된) 의도를 자동으로 감지하는 feed-forward 네트워크를 활용하며, 이는 sextuplet 형태로 구성됩니다.

- **Performance Highlights**: 다양한 MLMCID 데이터셋에 대한 실험 결과, 포인터 네트워크 기반의 RoBERTa 모델이 기존의 LLM(대규모 언어 모델) 및 기타 방법들보다 높은 정확도와 개선된 매크로 F1 점수를 기록하며 우수한 성능을 보였습니다.



### Do Large Language Models Align with Core Mental Health Counseling Competencies? (https://arxiv.org/abs/2410.22446)
Comments:
          9 Pages, In Submission to NAACL 2025

- **What's New**: 본 논문에서는 CounselingBench라는 새로운 NCMHCE 기반 벤치마크를 소개하여 대규모 언어 모델(LLM)의 정신 건강 상담 능력을 평가합니다.

- **Technical Details**: 22개의 일반 목적 및 의료 전문화 LLM을 대상으로 하여 다섯 가지 핵심 정신 건강 상담 능력에 대한 평가를 진행했습니다.

- **Performance Highlights**: 최신 모델들이 최소 기준을 초과하는 성능을 보였으나, 전문가 수준의 성능에는 미치지 못하며, Intake, Assessment & Diagnosis에서는 우수한 성과를 보이는 반면, Core Counseling Attributes 및 Professional Practice & Ethics에서는 어려움을 겪는 경향이 있음을 발견했습니다.



### AAAR-1.0: Assessing AI's Potential to Assist Research (https://arxiv.org/abs/2410.22394)
Comments:
          Project Webpage: this https URL

- **What's New**: 이번 연구에서는 LLM의 성능을 평가하기 위해 설계된 AAAR-1.0 이라는 새로운 벤치마크 데이터셋을 소개합니다. 이 데이터셋은 연구 아이디어 브레인스토밍, 실험 설계, 논문 작성 및 검토와 같은 연구자의 일상 활동을 반영하는 네 가지 전문 연구 태스크를 포함합니다.

- **Technical Details**: AAAR-1.0은 (i) EquationInference, (ii) ExperimentDesign, (iii) PaperWeakness, (iv) ReviewCritique의 네 가지 태스크를 포함하며, 모델이 실제 연구에서 요구되는 심층적인 도메인 전문성과 경험을 소유해야 하는지 평가합니다. 데이터 품질을 보장하기 위해 경험이 풍부한 AI 연구자들이 데이터 주석을 수행하였습니다.

- **Performance Highlights**: 여러 LLM의 평가 결과, EquationInference에서는 대부분의 모델이 단순한 확률로 가까운 성능을 보였고, ExperimentDesign에서는 인간의 작업에 비해 혁신적이긴 하나 실제적인 실행 가능성이 부족했습니다. PaperWeakness와 ReviewCritique에서는 LLM이 식별한 약점과 인간 리뷰의 결함을 효과적으로 포착하지 못하고 있어 연구 평가에 한계가 있음을 나타냈습니다.



### ArxivDIGESTables: Synthesizing Scientific Literature into Tables using Language Models (https://arxiv.org/abs/2410.22360)
Comments:
          EMNLP 2024, 21 pages, 8 figures, 10 tables

- **What's New**: 이 논문에서는 문헌 리뷰 테이블 자동 생성 방법을 제안합니다. 이를 위해 언어 모델(LM)을 활용하여 스키마(schema)와 값(value) 생성을 단계별로 수행합니다. 새로운 데이터 세트인 arxivDIGESTables를 공개하여 품질 있는 데이터셋 부족 문제를 해결하고, DecontextEval이라는 자동 평가 방법을 통해 사람 작성 테이블과의 비교를 지원합니다.

- **Technical Details**: 연구진은 2,228개의 문헌 리뷰 테이블을 포함하는 arxivDIGESTables 데이터셋을 수집하였습니다. 이 데이터셋은 16년 동안의 ArXiv 논문에서 추출된 것으로, 총 7,542개의 연구 논문을 종합한 것입니다. 또한, 모델 생성 결과와 인간이 작성한 테이블의 평가를 위한 DecontextEval 자동 평가 방법을 개발하였습니다. 이 방법은 언어 모델을 사용하여 테이블의 열 이름을 문서에 기반한 설명으로 확장합니다.

- **Performance Highlights**: 연구 결과, LMs는 추가적인 맥락이 주어질 때 참조 테이블을 재구성하는 데 있어 높은 성능을 보였으며, 생성된 새로운 측면도 유용할 수 있음을 발견했습니다. 특히, 테이블 캡션이나 본문 내 참조 등을 조건으로 하는 경우, LMs의 리콜이 증가하는 경향이 있었습니다.



### Efficient Machine Translation with a BiLSTM-Attention Approach (https://arxiv.org/abs/2410.22335)
- **What's New**: 본 연구는 Seq2Seq 모델의 혁신적인 구조를 제안하여 기계 번역의 품질을 향상시키면서 모델의 저장 공간을 줄이는 것을 목표로 합니다.

- **Technical Details**: 이 모델은 Bidirectional Long Short-Term Memory network (Bi-LSTM)을 인코더로 사용하여 입력 시퀀스의 컨텍스트 정보를 포착하며, 디코더에는 attention mechanism을 통합하여 번역 과정에서 중요한 정보에 집중할 수 있는 능력을 향상시킵니다.

- **Performance Highlights**: 본 모델은 WMT14 데이터셋에서 기존의 Transformer 모델보다 더 우수한 성능을 보여주었으며, 저장 공간과 계산 효율성을 대폭 개선했습니다.



### SlowFast-VGen: Slow-Fast Learning for Action-Driven Long Video Generation (https://arxiv.org/abs/2410.23277)
- **What's New**: 본 논문에서는 SlowFast-VGen이라는 새로운 이중속도 학습 시스템을 소개합니다. 이는 행동 주도의 긴 비디오 생성을 위해 느린 학습과 빠른 학습을 통합하여, 일관되고 응답적인 비디오 생성을 가능하게 합니다.

- **Technical Details**: 이 시스템은 두 가지 주요 구성 요소로 이루어져 있습니다. 느린 학습(적용된 조건부 비디오 확산 모델)과 빠른 학습(Temporal LoRA 모듈 기반의 추론 시간 학습 전략)입니다. 빠른 학습 과정에서는 입력 및 출력에 기반하여 Temporal LoRA 매개변수를 업데이트하여 에피소드 메모리를 저장합니다. 또한, 느린 학습 알고리즘과 빠른 학습 루프를 결합하여 다중 에피소드 경험을 기억하여 기술 학습을 지원합니다.

- **Performance Highlights**: SlowFast-VGen은 FVD 점수에서 514를 기록하며 782을 초월하여 이전 모델들보다 뛰어난 성능을 보였습니다. 또한 0.37의 평균 장면 전환수로, 긴 비디오의 일관성을 유지하는 데 있어 우수함을 보여주었습니다.



### TOMATO: Assessing Visual Temporal Reasoning Capabilities in Multimodal Foundation Models (https://arxiv.org/abs/2410.23266)
- **What's New**: 본 논문에서는 다중 모달 재단 모델(Multimodal Foundation Models, MFMs)의 비디오 이해에 대한 시각적 시간 추론 능력을 평가하기 위한 새로운 벤치마크, TOMATO(Temporal Reasoning Multimodal Evaluation)를 제안합니다.

- **Technical Details**: TOMATO는 (1) Multi-Frame Gain, (2) Frame Order Sensitivity, (3) Frame Information Disparity라는 세 가지 원칙과 해당 메트릭을 기반으로 하여 구성되었습니다. 이 벤치마크는 1,484개의 질문으로 구성되며, 1,417개의 비디오에 적용됩니다.

- **Performance Highlights**: 현재 MFMs의 성능 평가에서 인간 모델 성능 간의 격차는 57.3%로 나타났습니다. MFMs는 개별 프레임에서 이벤트를 정확하게 인식할 수 있지만, 이러한 프레임을 연속적인 시퀀스로 해석하는 데 실패하는 기본적인 한계가 드러났습니다.



### EMMA: End-to-End Multimodal Model for Autonomous Driving (https://arxiv.org/abs/2410.23262)
Comments:
          Blog post: this https URL

- **What's New**: EMMA(End-to-end Multimodal Model for Autonomous driving)는 모듈식 접근법 대신, 센서 데이터를 직접 처리하여 운전 관련 작업을 수행하는 새로운 모델입니다. 다양한 센서 데이터를 자연어 텍스트로 변환하여 통합적으로 처리할 수 있는 기능이 특징입니다.

- **Technical Details**: EMMA는 미리 훈련된 멀티모달 대형 언어 모델(Gemini) 위에 구축되었으며, 카메라 이미지 및 자연어 텍스트를 입력으로 받아 특정 운전 작업을 수행합니다. EMMA는 비전 모듈과 행동 모듈 간의 기호 인터페이스를 제거하여 각 운전 목표를 공동으로 최적화합니다.

- **Performance Highlights**: EMMA는 nuScenes 데이터셋에서 최첨단 성능을 달성하였으며, Waymo Open Motion Dataset에서도 경쟁력 있는 결과를 보였습니다. 또한, 3D 객체 탐지 및 도로 그래프 추정 등 여러 인식 작업에서도 뛰어난 성능을 보였습니다. 하지만, 이미지 프레임 수 처리의 제한, LiDAR와 같은 3D 감지 모드의 부재로 인한 문제도 존재합니다.



### COMAL: A Convergent Meta-Algorithm for Aligning LLMs with General Preferences (https://arxiv.org/abs/2410.23223)
- **What's New**: 본 논문에서는 보상 모델을 사용하는 기존의 알고리즘의 한계를 극복하기 위해 Convergent Meta Alignment Algorithm (COMAL)을 제안합니다. 이는 게임 이론의 수렴적 알고리즘에 영감을 받아 개발되었습니다.

- **Technical Details**: COMAL은 두 플레이어 제로섬 게임을 모델링하여 Nash equilibrium 정책에 도달하는 메타 알고리즘입니다. COMAL은 ProxProxoman_Prox 연산자를 기본 빌딩 블록으로 사용하여 모든 정책에 대해 50% 이상의 승률을 보장하는 robust alignment를 달성합니다.

- **Performance Highlights**: COMAL은 다양한 기존 알고리즘과 비교했을 때 마지막 반복(iterate)에서 Nash equilibrium에 수렴하는 유일한 알고리즘임을 실험적으로 입증하였으며, DPO 및 다양한 반복 알고리즘에 비해 항상 50%를 초과하는 승률을 기록했습니다.



### ProTransformer: Robustify Transformers via Plug-and-Play Paradigm (https://arxiv.org/abs/2410.23182)
- **What's New**: 이 논문에서는 트랜스포머 기반 아키텍처의 강인성을 향상시키기 위해 설계된 새로운 강력한 주의 기법을 소개합니다. 이 기술은 기존 트랜스포머에 플러그 앤 플레이 (plug-and-play) 레이어로 통합할 수 있어 추가적인 훈련이나 파인 튜닝 없이도 강인성을 개선할 수 있습니다.

- **Technical Details**: ProTransformer는 주의 메커니즘과 가중 최소 제곱 추정기 (weighted least square estimator) 간의 새로운 연결을 확립하며, 이로부터 강인한 토큰 추정기를 제안하여 적대적 공격에 대한 토큰 집계의 회복력을 향상시킵니다. 제안된 Newton-IRLS 알고리즘은 비볼록 및 비매끄러운 강력한 토큰 추정기를 근사화합니다.

- **Performance Highlights**: ProTransformer는 BERT, ALBERT, DistilBERT 및 RoBERTa에서 각각 19.5%, 28.3%, 16.1% 및 11.4%의 성능 향상을 보여주며, 대형 언어 모델인 T5 및 LLaMA에서도 24.8% 및 17.8%의 성능 향상을 나타냅니다. 추가적으로, ProTransformer는 비전 및 그래프 도메인에서도 우수한 강인성을 보여줍니다.



### Unified Triplet-Level Hallucination Evaluation for Large Vision-Language Models (https://arxiv.org/abs/2410.23114)
Comments:
          18 pages, 8 figures

- **What's New**: 본 논문은 대형 비전-언어 모델(LVLM)에서 발생하는 객체 및 관계 환각(hallucination)을 동시에 평가하기 위한 통합 프레임워크를 설계하였습니다.

- **Technical Details**: LVLM의 응답에서 추출된 (객체, 관계, 객체) 트리플렛(triplet) 기반의 환각 평가를 통해 환각 유형을 통합적으로 분석할 수 있으며, Tri-HE라는 새로운 평가 벤치마크를 도입하여 LVLM의 환각 문제를 보다 세밀하게 평가할 수 있습니다.

- **Performance Highlights**: 이 연구에서 제안한 방법은 기존 LVLM이 갖고 있는 관계 환각 문제를 해결함으로써, LLaVA-1.5 모델이 모든 오픈 소스 모델을 초월하는 성능을 기록하였으며, 강력한 GPT-4V와 동등한 성능을 보여주었습니다.



### CORAL: Benchmarking Multi-turn Conversational Retrieval-Augmentation Generation (https://arxiv.org/abs/2410.23090)
- **What's New**: 이 논문에서는 다중 회화(Multi-turn) 환경에서의 RAG 시스템 평가를 위한 새로운 벤치마크인 CORAL을 소개합니다. 기존 연구가 단일 턴(Single-turn) RAG에 주로 집중되는 가운데, 다중 턴 회화를 다루기 위한 연구가 부족하다는 점을 강조합니다.

- **Technical Details**: CORAL은 8,000개의 다양한 정보 검색 대화를 포함하며, Wikipedia에서 자동으로 유도된 대화를 기반으로 합니다. 이 벤치마크는 대화형 RAG의 세 가지 핵심 작업인 문서 검색(Passage Retrieval), 응답 생성(Response Generation), 그리고 인용 레이블(Citation Labeling)을 지원합니다.

- **Performance Highlights**: 정교한 오픈 소스 LLM이 상업적으로 배포된 클로즈드 소스 LLM보다 검색 성능에서 우수함을 보여주며, 입력 길이를 줄이는 것이 응답 품질을 유지하고 인용 레이블 정확도를 개선할 수 있음을 발견했습니다.



### Multi-Programming Language Sandbox for LLMs (https://arxiv.org/abs/2410.23074)
Comments:
          25 pages, 14 figures

- **What's New**: MPLSandbox는 다국어 지원의 최신 방식으로, 코드 생성을 위한 통합된 컴파일러 피드백과 분석을 제공하는 다목적 샌드박스 환경입니다. 기존의 샌드박스 도구가 단일 프로그래밍 언어에 제한되었던 문제를 해결하고, 연구자들이 여러 언어로 코드를 안전하게 컴파일하고 실행할 수 있도록 설계되었습니다.

- **Technical Details**: MPLSandbox는 세 가지 핵심 모듈로 구성됩니다: 1) 다국어 샌드박스 환경(Multi-Programming Language Sandbox Environment), 2) 코드 분석 모듈(Code Analysis Module), 3) 정보 통합 모듈(Information Integration Module). 다국어 샌드박스 환경은 프로그래밍 언어를 자동으로 인식하고 안전하게 코드를 실행합니다. 코드 분석 모듈은 정적 및 동적 분석 도구를 통합하여 코드에 대한 포괄적인 정보를 제공합니다. 정보 통합 모듈은 이러한 분석 결과를 LLMs에 통합하여 코드 생성의 품질을 향상시킵니다.

- **Performance Highlights**: MPLSandbox는 효율적인 코드 관련 작업 수행을 위해 LLM의 훈련과 배포 시나리오에 쉽게 통합될 수 있으며, 코드의 정확성과 품질을 개선하는 데 기여합니다. 실제 코드 관련 작업인 단위 테스트 생성, 버그 수정, 취약점 로컬라이제이션, 코드 번역 등의 분야에서 효율성을 강조하고 있습니다.



### Controlling Language and Diffusion Models by Transporting Activations (https://arxiv.org/abs/2410.23054)
- **What's New**: 이 논문에서는 Activation Transport (AcT)라는 새로운 프레임워크를 소개하며, 이는 Optimal Transport(OT) 이론에 기반하여 모델의 활성화를 유도할 수 있는 방법론입니다. 이 기법은 기존의 활성화 조정 방법들을 통합적으로 설명하며, 대조군 언어 또는 서로 다른 스타일의 텍스트에서의 변환을 효과적으로 수행할 수 있습니다.

- **Technical Details**: AcT는 모달리티에 구애받지 않으며, 모델의 내부 활성화 분포를 보존하면서도 미세한 조정을 가능하게 합니다. 이 방법은 λ (lambda)라는 강도 매개변수를 이용해 개입 정도를 조절할 수 있으며, 0에서 1 사이의 값으로 설정하여 부분적 또는 전체적 변환을 적용합니다. 특히, 저자는 Linear-AcT 방식이 기존의 개입 방법들과 비교하여 더 나은 성과를 낼 수 있음을 실험적으로 입증했습니다.

- **Performance Highlights**: AcT는 Large Language Models (LLMs)에서 독성 감소, 개념 유도, 진실성 증가를 효과적으로 범위 내에서 수행하며, Text-to-Image (T2I) 모델에서도 미세한 스타일 조정 및 개념 부정이 가능함을 보여줍니다. 이 연구는 LLMs와 확산 모델에서 동시에 효과적인 개입 방법을 적용한 최초의 사례로 자리잡고 있습니다.



### Online Intrinsic Rewards for Decision Making Agents from Large Language Model Feedback (https://arxiv.org/abs/2410.23022)
- **What's New**: 이번 연구에서는 ONI라는 분산 아키텍처를 제안하여 강화 학습 (RL) 정책과 내재적 보상 함수를 동시에 학습하는 방식으로, 기존의 내재적 보상 설계의 한계를 극복합니다.

- **Technical Details**: ONI는 비동기 LLM 서버를 통해 에이전트의 경험을 주석 처리하고, 이러한 피드백을 토대로 내재적 보상 모델을 증류합니다. 다양한 보상 모델링 방법 (해싱, 분류, 순위 모델)을 탐색하여 내재적 보상 설계에 대한 통찰을 제공합니다.

- **Performance Highlights**: ONI는 NetHack Learning Environment의 도전적인 희소 보상 작업에서 최첨단 성능을 달성했으며, 외부 데이터셋이나 소스 코드 없이 에이전트가 수집한 경험만을 사용하여 학습하게 됩니다.



### VisAidMath: Benchmarking Visual-Aided Mathematical Reasoning (https://arxiv.org/abs/2410.22995)
Comments:
          58 pages, 28 figures

- **What's New**: 새로운 연구에서는 시각적 정보를 활용한 수학 문제 해결(Visual Aided Mathematical Problem Solving, MPS) 과정을 평가하기 위한 VisAidMath 벤치마크를 소개합니다. 이 벤치마크는 1,200개의 난이도 있는 문제를 포함하며, 다양한 출처에서 수집된 문제와 답변을 평가합니다.

- **Technical Details**: VisAidMath 벤치마크는 수학적 질문을 시각적 맥락(Visual Context), 질문(Question), 시각적 도움(Visual Aids), 답변(Answer) 네 부분으로 나누어 설계되었습니다. 이 벤치마크는 명확한 시각적 정보를 포함하고 있으며, 값은 LaTeX 형식으로 정리되어 있습니다. 이 연구에서는 통계적으로 10개의 주요 LLMs 및 LMMs의 성능을 분석하였습니다.

- **Performance Highlights**: 주요 모델인 GPT-4V는 시각적 정보를 활용한 추론 과제에서 평균 45.33%의 정확도를 보였으며, 이는 전문적인 시각적 도움을 제공받았을 때도 2점 감소하였습니다. 또한, SOTA 모델들은 약 50%의 평균 정확도에 그쳤으며, 생성된 시각적 보조 자료는 5%의 n-gram 유사성만을 보였습니다.



### Focus On This, Not That! Steering LLMs With Adaptive Feature Specification (https://arxiv.org/abs/2410.22944)
Comments:
          28pages, 14 figures

- **What's New**: 이번 연구에서는 Focus Instruction Tuning (FIT)이라는 새로운 방법론을 소개합니다. FIT는 LLMs를 특정 작업 수행 시 반응을 조정하도록 훈련시키며, 모델이 특정 기능에 초점을 맞추고 다른 기능은 무시할 수 있도록 합니다.

- **Technical Details**: FIT는 사용자가 어떤 기능에 주목할지를 동적으로 지정할 수 있게 하여, 모델의 행동을 유연하고 효과적으로 조정할 수 있습니다. 이는 특정 입력에 대해 주어진 특징에 따라 다르게 응답할 수 있도록 LLMs를 훈련시킵니다. 다양한 NLP 작업, 예를 들어 감정 분석(sentiment analysis), 자연어 추론(natural language inference), 질문-응답(question-answering) 등에서 FIT의 효과를 실험하였습니다.

- **Performance Highlights**: FIT는 훈련 시 사용하지 않았던 새로운 기능에서의 일반화 능력이 뛰어나며, 분포의 변동에 강인하게 반응합니다. 사용자가 요청하는 바에 따라 모델이 알려진 왜곡된 기능(spurious features)을 무시하고 작업 관련 기능(task-relevant features)에 주력하도록 유도하여, 더 강력하고 공정하며 제어 가능한 LLM 애플리케이션을 가능하게 합니다.



### VPO: Leveraging the Number of Votes in Preference Optimization (https://arxiv.org/abs/2410.22891)
- **What's New**: 본 논문에서는 Direct Preference Optimization (DPO) 방법을 보완하여 사용자 투표 데이터를 활용한 Vote-based Preference Optimization (VPO) 프레임워크를 제안합니다. 이 방법은 사용자 선호도를 보다 효과적으로 반영하여 다양한 주관적 선호와의 정렬을 개선하는 것을 목표로 합니다.

- **Technical Details**: VPO는 Bayesian Minimum Mean Square Error (MMSE) 추정기를 사용하여 두 개의 생성 결과 중 하나가 더 선호될 확률을 모델링합니다. VPO는 DPO와 Identity Preference Optimization (IPO) 알고리즘을 각각 VDPO와 VIPO로 확장할 수 있으며, 이는 생성 쌍의 논쟁적 여부를 분별하는 데 도움이 됩니다.

- **Performance Highlights**: 실험 결과 VDPO와 VIPO는 기존 알고리즘들보다 뛰어난 생성 품질과 훈련 안정성을 달성하였습니다. 이 프레임워크는 고비용의 인간 투표 정보가 없는 상황에서도 AI 피드백을 활용하여 적용 가능합니다.



### Effective and Efficient Adversarial Detection for Vision-Language Models via A Single Vector (https://arxiv.org/abs/2410.22888)
- **What's New**: 이번 논문에서는 Visual Language Models (VLMs)에 대한 새로운 대규모 적대적 이미지 데이터셋 RADAR를 구축하고, 이를 이용해 NEARSIDE라는 새로운 적대적 이미지 탐지 방법을 제안합니다. 이 방법은 VLMs의 숨겨진 상태로부터 도출된 단일 벡터(공격 방향)를 활용하여 적대적 이미지를 효과적으로 탐지합니다.

- **Technical Details**: RADAR 데이터셋은 MiniGPT-4와 LLaVA를 공격하기 위해 다양한 유해 내용을 가진 적대적 이미지를 생성하여 구성되었습니다. 데이터셋에는 총 4,000개의 샘플이 포함되어 있으며, NEARSIDE 방법은 입력의 임베딩이 공격 방향과의 유사성을 평가하여 적대적 샘플의 존재를 탐지합니다. 실험 결과, LLaVA에서는 83.1%, MiniGPT-4에서는 93.5%의 탐지 정확도를 기록했습니다.

- **Performance Highlights**: NEARSIDE는 LLaVA에서 0.14초의 평균 속도로 탐지를 수행하여 가장 기존 방법보다 40배나 빨라 효율성이 뛰어남을 증명했습니다. 또한, 공격 방향의 모델 간 전이 가능성을 검증하여 연구의 유용성을 높였습니다.



### Stealing User Prompts from Mixture of Experts (https://arxiv.org/abs/2410.22884)
- **What's New**: 이 논문은 Mixture-of-Experts (MoE) 모델에서 사용자 쿼리를 추출하는 새로운 공격 방법을 제시합니다. 이 공격 방법은 피해자의 쿼리와 같은 배치에 있는 쿼리를 조작하여 피해자의 프롬프트를 완전히 공개할 수 있습니다.

- **Technical Details**: 연구에서 소개된 공격은 두 계층의 Mixtral 모델에서 실행되며, 공격자는 $O({VM}^2)$ 쿼리(어휘 크기 $V$ 및 프롬프트 길이 $M$에 따라) 또는 평균적으로 각 토큰당 100개의 쿼리로 프롬프트를 추출할 수 있습니다. 이것은 공격자가 Expert-Choice-Routing을 악용하여 구현된 CUDA의 tie-handling 동작을 활용하는 것입니다.

- **Performance Highlights**: 이 연구는 사용자 프롬프트를 추출하기 위해 구조적 결함을 exploit 하는 공격의 첫 번째 사례로, 새로운 LLM(대형 언어 모델) 취약점의 클래스를 소개합니다.



### Improving Uncertainty Quantification in Large Language Models via Semantic Embeddings (https://arxiv.org/abs/2410.22685)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)에서 정확히 불확실성을 정량화하는 새로운 접근 방식을 제안합니다. 기존의 방법들은 이중 포함 관계를 기준으로 여러 생성된 응답간의 의미적 불확실성을 측정하는데 의존하며, 이는 세부적인 표현 차이에 민감하여 실제 불확실성을 과대평가하는 경향이 있습니다. 이에 반해, 저자들은 의미 임베딩을 사용하여 의미적 불확실성을 보다 부드럽고 견고하게 추정할 수 있는 방법을 제시합니다.

- **Technical Details**: 저자들은 의미적 임베딩 불확실성(Semantic Embedding Uncertainty, SEU) 개념을 도입하며, 이는 전체 응답 임베딩의 평균 쌍 간 코사인 유사도를 활용합니다. 이 방법은 이중 포함을 기준으로 의미적으로 동등한 응답을 군집화하는 과정에서 발생하는 문제를 피할 수 있습니다. 또한 저자들은 이론을 기반으로 한 암모르티즈드(SEU) 모델을 제시하여 잠재 변수로서 의미를 모델링해 단일 전방 통과로 불확실성을 추정하는 방식을 개발하였습니다. 이를 통해 연산 오버헤드를 획기적으로 줄이고 실무 환경에서의 활용성을 향상시킬 수 있습니다.

- **Performance Highlights**: 다양한 질문-응답 데이터셋과 최전선 LLMs에 대한 실험 결과, 저자들이 제안한 임베딩 기반 방법이 기존의 전통적인 기법들보다 더 정확하고 섬세한 불확실성 정량화를 제공함을 입증하였습니다.



### Automated Trustworthiness Oracle Generation for Machine Learning Text Classifiers (https://arxiv.org/abs/2410.22663)
- **What's New**: 논문에서는 텍스트 분류를 위한 자동화된 신뢰성 오라클 생성 방법인 TOKI를 제안합니다. 이는 예측에 기여하는 단어들이 예측된 클래스와 얼마나 관련이 있는지를 자동으로 검사하여, 머신러닝 모델의 신뢰성을 높이기 위한 방법입니다.

- **Technical Details**: TOKI는 설명 방법(Explanation Methods)과 단어 임베딩(Word Embeddings)을 활용하여 예측에 영향을 미치는 단어들의 의미적 관련성을 평가합니다. 논문은 6000개의 예측 샘플을 사용해 TOKI의 성능을 검증했으며, 전반적인 신뢰성 검증을 향상시키기 위한 자동화 접근 방식을 탐구합니다.

- **Performance Highlights**: TOKI는 단순 모델 신뢰도(naive baseline)보다 142% 높은 정확도를 기록했고, TOKI 기반 적대적 공격 방법이 기존의 SOTA 적대적 공격 방법인 A2T보다 더 효과적인 것으로 나타났습니다.



### BENCHAGENTS: Automated Benchmark Creation with Agent Interaction (https://arxiv.org/abs/2410.22584)
- **What's New**: 본 연구에서는 BENCHAGENTS라는 프레임워크를 소개하며, 이는 대형 언어 모델(LLMs)을 활용하여 복잡한 기능을 평가하기 위한 벤치마크 생성 과정을 자동화합니다. 특히, 벤치마크 생성 과정을 계획, 생성, 검증, 평가의 네 가지 단계로 나누고 각각을 LLM 에이전트가 담당하여 데이터의 질과 메트릭 품질을 보장합니다.

- **Technical Details**: BENCHAGENTS는 다양한 LLM 에이전트를 사용하여 벤치마크 생성 프로세스를 관리합니다. 각 에이전트는 다음과 같은 역할을 수행합니다: Planning Agent는 벤치마크 계획을 수립하고, Data Generation Agent는 데이터를 생성하며, Verification Agent는 생성된 데이터의 품질을 확인하고, Evaluation Agent는 모델 성능 평가를 위한 메트릭을 생성합니다.

- **Performance Highlights**: BENCHAGENTS를 활용하여 생성된 두 개의 벤치마크(BA-Calendar 및 BA-Text)에 대해 최첨단 LLM 7개 모델을 평가한 결과, 모든 모델이 제약 조건 충족에 어려움을 겪고 있으며, 특히 제약 조건 수가 증가할수록 성능이 떨어지는 경향을 보였습니다. 이는 모델 간의 우선순위 설정에서 차이점을 보이며, 주로 수치적 혹은 논리적 추론을 요구하는 제약 조건에서 실패하는 경향이 있음을 나타냅니다.



### Advancing Agentic Systems: Dynamic Task Decomposition, Tool Integration and Evaluation using Novel Metrics and Datas (https://arxiv.org/abs/2410.22457)
Comments:
          38th Conference on Neural Information Processing Systems (NeurIPS 2024), NeurIPS 2024 Workshop on Open-World Agents

- **What's New**: 이 논문은 자율 에이전트 시스템의 발전을 위해 'Advanced Agentic Framework'를 제안하며, 이를 통해 다단계 작업을 역동적으로 처리하고 적절한 도구를 자동으로 선택할 수 있는 능력을 강화하고 있습니다.

- **Technical Details**: 논문에서는 주요 기여로 세 가지 요소를 제시합니다: 1) 다단계 쿼리를 처리하고 작업 그래프를 생성 및 실행하며 적합한 도구를 선택하고 실시간 변화에 적응하는 고급 에이전트 프레임워크, 2) 에이전트 시스템에 대한 전반적인 평가를 위한 새로운 평가 지표(Node F1 Score, Structural Similarity Index (SSI), Tool F1 Score), 3) 다양한 작업 복잡성에서 에이전트 행동을 분석하기 위한 AsyncHow 기반의 전문 데이터셋을 개발하였습니다.

- **Performance Highlights**: 강화된 작업 그래프 분해는 시스템의 응답성과 확장성을 크게 향상시킵니다. 특히, SSI는 순차적 작업에서 성능의 가장 중요한 예측 변수가 되고, Tool F1 Score는 병렬 작업에서 필수적인 성과 지표로 나타났습니다.



### A Closer Look at Neural Codec Resynthesis: Bridging the Gap between Codec and Waveform Generation (https://arxiv.org/abs/2410.22448)
Comments:
          NeurIPS 2024 Audio Imagination workshop paper; demo page at this https URL

- **What's New**: 이 논문에서는 neural audio codec의 coarse 토큰에서 waveforms를 재합성(resynthesize)하는 방법을 탐구합니다. 일반적으로 기존 연구들은 coarse 토큰 생성을 우선시했지만, 저자들은 coarse 토큰만을 활용하여 어떻게 높은 품질의 오디오를 재합성할 수 있는지에 대해 주목합니다.

- **Technical Details**: 저자들은 코덱 재합성을 위한 두 가지 주요 접근 방식, 즉 토큰 예측(token prediction)과 회귀(regression)에 기반한 전략을 연구하고, 새로운 방법인 Schrödinger Bridge를 제안합니다. 이러한 방법은 다양한 설계 선택이 기계 및 인간 인식에 미치는 영향을 분석합니다.

- **Performance Highlights**: 이 연구는 coarse 임베딩을 기반으로 한 오디오 재합성을 통해 기존의 ad-hoc 방법들을 개선하고, 레벨에 따른 정보 구조를 활용하여 더 나은 오디오 품질을 달성할 수 있는 가능성을 보여줍니다.



### Rare-to-Frequent: Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with LLM Guidanc (https://arxiv.org/abs/2410.22376)
- **What's New**: 본 연구는 희귀한 개념에 대한 텍스트-이미지(T2I) 분산 모델의 구성능력을 높이기 위해 대형 언어 모델(LLM) 가이드를 활용하는 접근 방식을 제안합니다. 우리는 창의적이고 비범한 특성을 가진 새로운 캐릭터 디자인과 같은 희귀한 프롬프트 생성 과정에서 기존 모델들이 어려움을 겪는 문제를 강조합니다.

- **Technical Details**: 본 연구에서는 희귀한 개념 구성을 위한 새로운 접근 방식인 R2F(Rare-to-Frequent)를 제안합니다. R2F는 LLM을 활용하여 희귀한 컨셉과 관련된 보다 일반적인 빈번한 개념을 찾고, 이를 통해 분산 추론(difussion inference) 과정을 개선합니다.

- **Performance Highlights**: R2F는 다양한 희귀 개념 구성을 포함하는 새로운 벤치마크인 RareBench에서 기존 모델인 SD3.0 및 FLUX와 비교하여 T2I 정렬 정확도에서 최대 28.1%p 향상된 성능을 보였습니다.



### Rethinking Code Refinement: Learning to Judge Code Efficiency (https://arxiv.org/abs/2410.22375)
- **What's New**: 이 논문은 LLM (Large Language Models)이 생성한 코드가 항상 효율적이지 않다는 점을 강조하며, 코드 리팩토링 과정에서 효율성을 판단하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 제안하는 모델은 코드 언어 모델 (code language model)을 기반으로 하여, 두 개의 코드 (원본과 개선된 버전)의 효율성을 비교합니다. 이 모델은 코드 쌍이 주어졌을 때, 어떤 코드가 더 효율적인지 또는 개선된 코드의 효율성을 예측하는 방식으로 학습됩니다.

- **Performance Highlights**: 여러 프로그래밍 언어와 다양한 리팩토링 시나리오를 통해 검증한 결과, 제안한 효율성 판단 모델이 두 가지 다른 버전 중에서 더 효율적인 코드를 식별하는 데 있어 현저한 성과를 보였으며, 기존 방법들에 비해 우수한 성능을 나타냈습니다.



### Survey of User Interface Design and Interaction Techniques in Generative AI Applications (https://arxiv.org/abs/2410.22370)
- **What's New**: 이 논문은 현재의 인간-인공지능 상호작용 분야에서 사용자 인터페이스(UI) 디자인 및 사용자 상호작용 패턴에 대한 포괄적인 조사 결과를 제시합니다. 연구자들과 개발자들이 AI 애플리케이션을 설계하는 데 참고할 수 있는 다양한 사용자 상호작용 패턴을 문서화하고, 이를 통해 생성적 AI 애플리케이션 디자인의 진입장벽을 낮추는 것을 목표로 합니다.

- **Technical Details**: 이 설문조사는 사용자가 주도하는 상호작용(user-guided interactions)을 중심으로 진행되며, 이는 사용자가 의도적으로 생성적 AI 시스템에 영향을 미치는 모든 상호작용을 포함합니다. 구체적으로는 다양한 상호작용 기술과 이를 위한 UI 레이아웃, 사용자-AI 참여 수준을 분류하여 정리합니다. 또한, 텍스트 또는 이미지의 프롬프트(prompt)를 통해 사용자가 생성 모델을 제어하는 방식도 다룹니다.

- **Performance Highlights**: 이 연구는 100개 이상의 관련 생성적 AI 논문을 바탕으로 다양한 상호작용 기술과 UI 디자인 패턴을 카테고리화하였으며, 이는 디자이너들이 기존의 패턴을 활용하여 보다 직관적인 사용자 경험을 제공할 수 있도록 돕습니다. 연구 결과는 디자이너와 연구자 모두에게 생성적 AI 애플리케이션의 효과적인 설계를 위한 기초 자료가 될 것입니다.



### RuleRAG: Rule-guided retrieval-augmented generation with language models for question answering (https://arxiv.org/abs/2410.22353)
- **What's New**: 이번 논문에서는 질의와 규칙을 활용한 Rule-Guided Retrieval-Augmented Generation (RuleRAG)이라는 새로운 QA 접근법을 제안합니다. 이 방법은 retriever가 논리적으로 관련된 문서를 검색하도록 가이드하고, generator가 규칙에 따라 답변을 생성할 수 있도록 합니다.

- **Technical Details**: RuleRAG는 두 가지 방식, 즉 RuleRAG-ICL과 RuleRAG-FT를 사용합니다. RuleRAG-ICL에서는 인-context learning을 통해 고신뢰의 규칙을 활용하여 더 나은 검색 결과를 얻고 보다 정확한 답변을 생성하도록 합니다. RuleRAG-FT는 supervised fine-tuning을 통해 retriever와 generator를 업데이트하여 규칙 기반의 작업 지시 능력을 개선합니다.

- **Performance Highlights**: 실험 결과, RuleRAG-ICL은 Recall@10 점수에서 +89.2%, exact match 점수에서 +103.1%의 성능 향상을 보여줍니다. 추가적인 fine-tuning을 통해 RuleRAG-FT는 더욱 뚜렷한 성능 개선을 달성할 수 있음을 나타냅니다.



### Search Engines in an AI Era: The False Promise of Factual and Verifiable Source-Cited Responses (https://arxiv.org/abs/2410.22349)
- **What's New**: 본 논문은 대규모 언어 모델(LLM) 기반의 Answer Engine이 기존의 검색 엔진을 대체하는 것에 대한 연구 결과를 다루고 있습니다. 이러한 시스템은 정보를 검색하고 요약하여 제공하는 방식으로 발전하고 있으며, 그 한계와 개선 방안을 제안합니다.

- **Technical Details**: 연구에서는 21명의 참가자와 함께 Answer Engines와 전통적인 검색 엔진의 상호작용을 평가하고, 16가지의 Answer Engine의 한계를 식별했습니다. 그 후 8개의 지표와 연결된 16개의 디자인 권장 사항을 제안하였습니다. 이 연구는 일반적인 사용성 평가와 자동화된 평가를 통해 진행되었습니다.

- **Performance Highlights**: 자동 평가 결과, 세 가지 유명한 엔진(유채팅, 빙 코파일럿, 퍼플렉시 AI)의 한계(예: 잦은 환각, 부정확한 인용)를 수량화하였고, 모든 평가된 응답 엔진이 50-80%의 확률로 편향된 답변을 생성하는 것으로 나타났습니다. 이는 사용자의 의견에 동의하는 답변을 생성하는 경향이 있음을 나타냅니다.



### A Systematic Survey on Instructional Text: From Representation Formats to Downstream NLP Tasks (https://arxiv.org/abs/2410.18529)
- **What's New**: 이 논문은 복합적인 지시문 이해와 처리를 위한 포괄적인 조사 연구입니다. 현재 NLP 시스템이 쉽게 수행할 수 없는 복잡한 다단계 지시사항에 대한 필요성을 강조합니다.

- **Technical Details**: 본 연구는 177개의 문헌을 분석하여 지시문 텍스트와 관련된 리소스, 표현 방식, 다운스트림 태스크를 체계적으로 정리합니다. 다양한 분야에서의 복합 지시문 표현 방식과 그에 따른 데이터 세트, 기법을 차별화된 관점으로 제시합니다.

- **Performance Highlights**: AI/NLP 연구자에게 복합 지시문 이해에 대한 통합적 관점을 제공함으로써 서로 다른 연구 방향 간의 격차를 줄이고 향후 연구 기회를 강조합니다.



New uploads on arXiv(cs.IR)

### ReasoningRec: Bridging Personalized Recommendations and Human-Interpretable Explanations through LLM Reasoning (https://arxiv.org/abs/2410.23180)
Comments:
          Large Language Model, Recommendation, Human-Interpretable Reasoning, Personalization, Submitted for NAACL 2025

- **What's New**: 이 논문에서는 ReasoningRec라는 새로운 추천 프레임워크를 제안합니다. 이 프레임워크는 Large Language Model (LLM)을 활용하여 추천 및 인간이 해석할 수 있는 설명 사이의 간극을 메우는 데 중점을 두고 있습니다.

- **Technical Details**: ReasoningRec은 사용자의 선호와 반감을 모델링하기 위해 LLM을 사용하며, 사용자가 특정 아이템을 좋아할 이유에 대한 합성 설명을 생성합니다. 이러한 설명은 더 작은 LLM을 미세 조정하는 데 사용되어 추천 정확성을 높이고, 인간이 이해할 수 있는 설명을 제공합니다.

- **Performance Highlights**: ReasoningRec은 추천 예측에서 기존의 최첨단 기술들을 12.5%까지 초과하는 성능을 보였으며,追加로 인간이 이해할 수 있는 설명을 제공합니다.



### Real-Time Personalization for LLM-based Recommendation with Customized In-Context Learning (https://arxiv.org/abs/2410.23136)
- **What's New**: 본 논문에서는 기존의 Large Language Model (LLM) 기반 추천 시스템에서 모델 업데이트 없이 동적인 사용자 관심사에 적응할 수 있는 방법을 제안합니다. 이 방법론은 In-Context Learning (ICL)이라는 기법을 활용하여 신규 관심사 예시를 사용하여 LLM이 실시간 관심을 배울 수 있도록 합니다.

- **Technical Details**: 우리는 RecICL을 제안하여 추천 특화된 ICL 기능을 개발합니다. 이 방법은 모델 훈련 시 이력 상호작용 시퀀스와 최근 추천 예시를 입력으로 포함시켜 ICL 기능을 유지하면서도 추천 작업에 aligned되도록 합니다. 훈련 완료 후, 실제 사용자 데이터를 포함한 예시로 대체하여 실시간으로 사용자 관심을 적응하게 만듭니다.

- **Performance Highlights**: 폭넓은 실험을 통해 RecICL이 모델 업데이트 없이도 실시간 추천을 제공하며 기존 접근 방식보다 뛰어난 성능을 보임을 입증했습니다. 이 방법은 향후에도 강력한 성능을 유지할 수 있습니다.



### CORAL: Benchmarking Multi-turn Conversational Retrieval-Augmentation Generation (https://arxiv.org/abs/2410.23090)
- **What's New**: 이 논문에서는 다중 회화(Multi-turn) 환경에서의 RAG 시스템 평가를 위한 새로운 벤치마크인 CORAL을 소개합니다. 기존 연구가 단일 턴(Single-turn) RAG에 주로 집중되는 가운데, 다중 턴 회화를 다루기 위한 연구가 부족하다는 점을 강조합니다.

- **Technical Details**: CORAL은 8,000개의 다양한 정보 검색 대화를 포함하며, Wikipedia에서 자동으로 유도된 대화를 기반으로 합니다. 이 벤치마크는 대화형 RAG의 세 가지 핵심 작업인 문서 검색(Passage Retrieval), 응답 생성(Response Generation), 그리고 인용 레이블(Citation Labeling)을 지원합니다.

- **Performance Highlights**: 정교한 오픈 소스 LLM이 상업적으로 배포된 클로즈드 소스 LLM보다 검색 성능에서 우수함을 보여주며, 입력 길이를 줄이는 것이 응답 품질을 유지하고 인용 레이블 정확도를 개선할 수 있음을 발견했습니다.



### A Universal Sets-level Optimization Framework for Next Set Recommendation (https://arxiv.org/abs/2410.23023)
Comments:
          Accepter at CIKM2024

- **What's New**: 이번 연구에서는 Next Set Recommendation (NSRec)에서의 통합된 최적화 프레임워크인 SNSRec를 제안합니다. 이는 다양성과 복잡한 의존 관계를 통합하여 기존의 접근법들을 넘어서려는 시도를 포함하고 있습니다.

- **Technical Details**: SNSRec는 Structured Determinantal Point Process (SDPP)를 활용하여 순차적 세트를 단일 단위로 모델링합니다. 이러한 방식으로 모델은 세트 내의 의존성과 세트 간 의존성을 포괄하는 다양성 분포를 기반으로 최적화 기준을 설정합니다. 이 접근 방식은 co-occurrence representation을 활용하여 세트의 중요성을 식별하고 강조합니다.

- **Performance Highlights**: 실제 데이터셋에 대한 광범위한 실험 결과, SNSRec는 관련성과 다양성 모두에서 기존 방법들보다 지속적으로 우수한 성능을 보여주었습니다.



### DataRec: A Framework for Standardizing Recommendation Data Processing and Analysis (https://arxiv.org/abs/2410.22972)
- **What's New**: 이번 연구에서는 추천 시스템의 데이터 관리 단계에 중점을 둔 Python 프레임워크인 DataRec를 제안합니다. DataRec는 다양한 데이터 형식으로의 읽기 및 쓰기를 지원하고, 필터링 및 분할 기법을 제공하며, 잘 알려진 메트릭을 사용하여 데이터 분포 분석을 가능하게 합니다. 이로 인해 다른 추천 프레임워크와의 상호운용성을 높이고 데이터를 통합하여 처리할 수 있는 접근 방식을 장려합니다.

- **Technical Details**: DataRec는 추천 시스템의 데이터 관리 프로세스를 개선하기 위해 설계되었습니다. 이 프레임워크는 여러 데이터 형식과 처리 기법을 지원하며, 기존 추천 프레임워크와의 호환성을 확보하기 위해 데이터를 해당 형식으로 내보내는 기능을 제공합니다. 또한, 데이터 소스에 대한 추적 가능한 참조와 버전 관리를 제공하고, 추천 데이터 메트릭에 대한 접근을 가능하게 합니다. 향후 이 프레임워크는 다양한 추천 시스템의 데이터 처리 및 분석 전략을 표준화하고 촉진할 것으로 기대됩니다.

- **Performance Highlights**: DataRec의 도입으로 추천 시스템의 데이터 처리 및 관리에서 상호운용성이 향상되며, 연구자들은 보다 체계적이고 재현 가능하게 데이터셋을 관리할 수 있게 됩니다. 이를 통해 다양한 추천 알고리즘 실험에 대한 비교 분석에서 발생할 수 있는 오류를 줄이고, 각 추천 프레임워크의 장점을 최대한 활용할 수 있는 기회를 제공합니다.



### Understanding and Improving Adversarial Collaborative Filtering for Robust Recommendation (https://arxiv.org/abs/2410.22844)
- **What's New**: 이 논문에서는 Adversarial Collaborative Filtering (ACF)이 전통적인 협업 필터링(CF)에 비해 추천 성능과 강인성을 높일 수 있는 이론적 근거를 제시합니다. 새로운 접근법으로 Personalized Magnitude Adversarial Collaborative Filtering (PamaCF)을 제안하고, 이를 통해 사용자에 대한 개인화된 교란 크기를 적용하여 효과를 극대화합니다.

- **Technical Details**: ACF는 사용자 및 아이템 임베딩에 대한 적대적 섭동(adversarial perturbations)을 통해 협업 필터링 시스템의 강인성을 개선합니다. 본 논문에서는 ACF의 최적화 과정에서의 추천 오류 감소에 대한 상하계를 설정하고, 각각의 사용자에 대해 개인화된 섭동 크기를 적용하는 것이 ACF의 효과를 더욱 개선할 수 있음을 발견했습니다.

- **Performance Highlights**: PamaCF는 기존의방식에 비해 추천 성능을 평균 13.84% 향상시키고, 공격 성공 확률을 평균 44.92% 감소시켰습니다. 실험 결과는 PamaCF가 다양한 유형의 독성 공격에 효과적으로 대응하고 추천 성능을 개선함을 보여줍니다.



### Causality-Enhanced Behavior Sequence Modeling in LLMs for Personalized Recommendation (https://arxiv.org/abs/2410.22809)
- **What's New**: 본 연구에서는 Large Language Models (LLMs)를 활용한 추천 시스템의 최신 동향과 한계를 논의하고, 사용자 행동 시퀀스의 중요성을 강조하는 새로운 Counterfactual Fine-Tuning (CFT) 방법을 제안합니다.

- **Technical Details**: 우리는 행동 시퀀스가 모델 출력에 미치는 인과적 효과를 식별하기 위해 counterfactual reasoning을 활용하며, 이로부터 추론된 효과를 직접 활용하여 진짜 레이블에 적합하도록 하는 새로운 작업을 도입합니다. 또한, 아이템 토큰에 대한 강조 강도를 조정하기 위한 token-level weighting 메커니즘을 개발하여, 예측할 때 초기 토큰에서 후속 토큰으로 갈수록 행동 시퀀스의 영향력이 줄어드는 것을 반영합니다.

- **Performance Highlights**: 실제 데이터셋에 대한 광범위한 실험을 통해 CFT가 행동 시퀀스 모델링을 효과적으로 개선함을 입증하였습니다. 이로 인해 LLM 기반의 추천 시스템의 효과성과 개인화된 추천의 정확성이 향상되었습니다.



### Dual Contrastive Transformer for Hierarchical Preference Modeling in Sequential Recommendation (https://arxiv.org/abs/2410.22790)
- **What's New**: 본 논문은 기존 Sequential Recommender Systems (SRSs)의 한계를 극복하기 위해 고안된 새로운 계층적 선호 모델링 프레임워크(Hierarchical Preference modeling, HPM)를 소개합니다. 이 프레임워크는 사용자 선호를 더 정확하게 모델링하기 위해 여러 가지 혁신적인 모듈을 도입하였습니다.

- **Technical Details**: 이 연구에서는 Dual-Transformer 모듈과 Dual Contrastive Learning 방식을 통해 사용자의 저수준(low-level) 및 고수준(high-level) 선호를 구별하여 학습합니다. 또한, Semantics-enhanced Context Embedding Learning 모듈을 통해 아이템 간의 숨겨진 의미 관계를 잘 캡처하여 더 나은 추천 성능을 위해 정보가 풍부한 컨텍스트 임베딩을 생성합니다.

- **Performance Highlights**: 여섯 개의 실제 데이터셋에 대한 광범위한 실험 결과, 제안된 방법이 기존 최첨단 방법들보다 우수한 성능을 나타내며, 고안된 설계의 합리성이 입증되었습니다.



### RuleRAG: Rule-guided retrieval-augmented generation with language models for question answering (https://arxiv.org/abs/2410.22353)
- **What's New**: 이번 논문에서는 질의와 규칙을 활용한 Rule-Guided Retrieval-Augmented Generation (RuleRAG)이라는 새로운 QA 접근법을 제안합니다. 이 방법은 retriever가 논리적으로 관련된 문서를 검색하도록 가이드하고, generator가 규칙에 따라 답변을 생성할 수 있도록 합니다.

- **Technical Details**: RuleRAG는 두 가지 방식, 즉 RuleRAG-ICL과 RuleRAG-FT를 사용합니다. RuleRAG-ICL에서는 인-context learning을 통해 고신뢰의 규칙을 활용하여 더 나은 검색 결과를 얻고 보다 정확한 답변을 생성하도록 합니다. RuleRAG-FT는 supervised fine-tuning을 통해 retriever와 generator를 업데이트하여 규칙 기반의 작업 지시 능력을 개선합니다.

- **Performance Highlights**: 실험 결과, RuleRAG-ICL은 Recall@10 점수에서 +89.2%, exact match 점수에서 +103.1%의 성능 향상을 보여줍니다. 추가적인 fine-tuning을 통해 RuleRAG-FT는 더욱 뚜렷한 성능 개선을 달성할 수 있음을 나타냅니다.



### Search Engines in an AI Era: The False Promise of Factual and Verifiable Source-Cited Responses (https://arxiv.org/abs/2410.22349)
- **What's New**: 본 논문은 대규모 언어 모델(LLM) 기반의 Answer Engine이 기존의 검색 엔진을 대체하는 것에 대한 연구 결과를 다루고 있습니다. 이러한 시스템은 정보를 검색하고 요약하여 제공하는 방식으로 발전하고 있으며, 그 한계와 개선 방안을 제안합니다.

- **Technical Details**: 연구에서는 21명의 참가자와 함께 Answer Engines와 전통적인 검색 엔진의 상호작용을 평가하고, 16가지의 Answer Engine의 한계를 식별했습니다. 그 후 8개의 지표와 연결된 16개의 디자인 권장 사항을 제안하였습니다. 이 연구는 일반적인 사용성 평가와 자동화된 평가를 통해 진행되었습니다.

- **Performance Highlights**: 자동 평가 결과, 세 가지 유명한 엔진(유채팅, 빙 코파일럿, 퍼플렉시 AI)의 한계(예: 잦은 환각, 부정확한 인용)를 수량화하였고, 모든 평가된 응답 엔진이 50-80%의 확률로 편향된 답변을 생성하는 것으로 나타났습니다. 이는 사용자의 의견에 동의하는 답변을 생성하는 경향이 있음을 나타냅니다.



### GleanVec: Accelerating vector search with minimalist nonlinear dimensionality reduction (https://arxiv.org/abs/2410.22347)
- **What's New**: 이 논문은 고차원 벡터 검색을 가속화하고 정확도를 유지할 수 있는 새로운 선형(Linear) 및 비선형(Nonlinear) 차원 축소 방법인 LeanVec-Sphering과 Generalized LeanVec(GleanVec)를 소개합니다.

- **Technical Details**: LeanVec-Sphering은 기존의 차원 축소 기법보다 더 우수하며, 하이퍼파라미터가 필요하지 않고 검색 중에 타겟 차원(Dimensionality)을 설정할 수 있습니다. GleanVec는 조각별 선형 구조를 사용하여 검색 정확성을 높이면서 계산 성능 또한 강조합니다.

- **Performance Highlights**: 초기 실험 결과에 따르면, LeanVec-Sphering과 GleanVec는 벡터 검색 상태를 끌어올리며, 높은 차원에서도 성능 저하를 방지할 수 있습니다.



### SciPIP: An LLM-based Scientific Paper Idea Proposer (https://arxiv.org/abs/2410.23166)
Comments:
          25 pages, 5 figures, 19 tables

- **What's New**: 이 논문은 과학 논문 아이디어 제안기인 SciPIP를 제안합니다. SciPIP는 사용자 제공 연구 배경을 바탕으로 유용한 논문을 검색하고, 이를 바탕으로 더 새롭고 실행 가능한 아이디어를 생성하는 방식으로 기존의 대형 언어 모델(LLM)의 잠재력을 활용합니다.

- **Technical Details**: SciPIP는 사용자가 제공한 연구 배경을 기반으로 문헌을 검색하여 아이디어를 제안하는 시스템입니다. 이를 위해 문헌 검색 데이터베이스를 구축하고, 의미론(semantic), 엔티티(entity), 인용의 공동 출현(citation co-occurrence)을 기반으로 문헌 검색을 수행합니다. 이후에는 문헌에서 얻은 정보를 활용하여 솔루션을 유추하거나 독창적인 아이디어를 생성하는 두 가지 경로를 통해 아이디어를 제안합니다.

- **Performance Highlights**: NLP 분야에서 진행된 광범위한 실험을 통해 SciPIP는 기존의 상위 회의 논문들과 유사한 인용을 검색하고, 많은 아이디어를 생성함으로써 그 효과를 입증하였습니다. 또한 SciPIP에 의해 생성된 아이디어는 청사진의 참조를 유지하면서도 혁신성과 실행 가능성을 확보하고 있음을 평가해 보여줍니다.



### HijackRAG: Hijacking Attacks against Retrieval-Augmented Large Language Models (https://arxiv.org/abs/2410.22832)
- **What's New**: 본 연구에서는 새로운 보안 취약점인 'retrieval prompt hijack attack (HijackRAG)'을 제시합니다. 이 공격은 공격자가 악의적인 텍스트를 지식 데이터베이스에 주입하여 RAG 시스템의 검색 메커니즘을 조작할 수 있게 합니다. 이를 통해 RAG 시스템이 목표 질문에 대한 잘못된 답변을 생성하게 만듭니다.

- **Technical Details**: HijackRAG는 최적화 문제로 형식화되며, 공격자의 지식 정도에 따라 black-box 및 white-box 공격 전략을 제시합니다. 실험 결과, HijackRAG는 여러 벤치마크 데이터셋에서 높은 공격 성공률을 기록하고 있으며, 다양한 검색 모델에 대해 이식성이 있음을 입증합니다.

- **Performance Highlights**: 후보 공격의 효과성을 입증하기 위해 수행된 광범위한 실험에서 HijackRAG는 기존 공격 방법에 비해 우수한 성과를 보였습니다. 특히, 기존 방어 메커니즘이 HijackRAG에 대해 충분한 방어력을 제공하지 못함을 확인하여, RAG 시스템의 실제 배치를 보호하기 위한 보다 강력한 보안 대책의 필요성을 강조합니다.



### A Pointer Network-based Approach for Joint Extraction and Detection of Multi-Label Multi-Class Intents (https://arxiv.org/abs/2410.22476)
Comments:
          Accepted at EMNLP 2024 Findings (Long Paper)

- **What's New**: 본 연구는 멀티레이블 멀티클래스 의도 탐지와 스팬 추출에 관한 새로운 접근 방식을 제시합니다. 최초로 멀티언어 멀티 의도 데이터셋(MLMCID-dataset)을 구축하고, 포인터 네트워크 기반 아키텍처를 통해 여러 개의 의도 스팬을 추출하며 다양한 레이블을 감지합니다.

- **Technical Details**: 연구에서는 포인터 네트워크 기반의 인코더-디코더 프레임워크를 구축하여 주어진 쿼리에서 여러 의도 스팬을 추출합니다. MLMCID 의도 탐지 모듈은 coarse(거칠게 분류된) 및 fine(세부적으로 분류된) 의도를 자동으로 감지하는 feed-forward 네트워크를 활용하며, 이는 sextuplet 형태로 구성됩니다.

- **Performance Highlights**: 다양한 MLMCID 데이터셋에 대한 실험 결과, 포인터 네트워크 기반의 RoBERTa 모델이 기존의 LLM(대규모 언어 모델) 및 기타 방법들보다 높은 정확도와 개선된 매크로 F1 점수를 기록하며 우수한 성능을 보였습니다.



New uploads on arXiv(cs.CV)

### ReferEverything: Towards Segmenting Everything We Can Speak of in Videos (https://arxiv.org/abs/2410.23287)
Comments:
          Project page at this https URL

- **What's New**: REM은 자연어로 설명 가능한 다양한 개념을 비디오에서 분할하는 프레임워크로, 비디오 디퓨전 모델에서 학습된 시각-언어 표현을 활용합니다.

- **Technical Details**: REM은 데이터의 한정적인 카테고리에서 개체 마스크를 기반으로 훈련되었음에도 불구하고 희귀하고 보지 못한 객체를 정확하게 분할하고 추적할 수 있습니다. 또한, Ref-VPS라는 새로운 벤치마크를 통해 비객체 동적 개념(예: 바다가 부서지는 파도)에 대한 일반화 능력을 보여줍니다.

- **Performance Highlights**: 실험 결과, REM은 Ref-DAVIS와 같은 도메인 내 데이터셋에서 최신 모델과 동등한 성능을 보여주며, 도메인 외 데이터에서는 최대 12점까지 지역 유사도를 초과 달성하였습니다.



### RelationBooth: Towards Relation-Aware Customized Object Generation (https://arxiv.org/abs/2410.23280)
- **What's New**: 본 연구에서는 사용자 제공 이미지 프롬프트를 바탕으로 개체의 정체성을 유지하는 동시에 텍스트 프롬프트에서 지정한 관계를 반영하는 관계 인식 맞춤화 이미지 생성(relation-aware customized image generation)에 중점을 두고 있는 RelationBooth라는 새로운 프레임워크를 소개합니다.

- **Technical Details**: RelationBooth는 LoRA(Low-Rank Adaptation) 전략을 텍스트 교차 주의 레이어에 적용하여 관계 학습을 촉진하며, 두 가지 주요 모듈을 포함합니다: (1) Keypoint Matching Loss(KML)를 통해 개체의 포즈를 조정하고, (2) 이미지 프롬프트에서 지역 토큰(local tokens)을 주입하여 겹치는 개체 간의 혼동을 방지합니다.

- **Performance Highlights**: 세 가지 벤치마크에서의 광범위한 결과들은 RelationBooth가 다양한 개체와 관계에 대해 정밀한 관계 생성을 유지하며 우수한 성능을 발휘함을 증명합니다.



### OpenSatMap: A Fine-grained High-resolution Satellite Dataset for Large-scale Map Construction (https://arxiv.org/abs/2410.23278)
Comments:
          NeurIPS 2024 D&B Track. Project Page:this https URL

- **What's New**: OpenSatMap은 대규모 지도 구축을 위한 고해상도 위성 데이터셋으로, 세부적인 개별 인스턴스 레벨 주석을 제공하고, 최대 해상도인 레벨 20의 이미지를 포함하며, 60개 도시와 19개 국가에서 수집된 높은 다양성을 자랑합니다.

- **Technical Details**: OpenSatMap은 레벨 20의 고해상도 이미지(0.15 m/pixel)를 사용하며, 약 38,000개의 1024 × 1024 이미지와 445,000개의 인스턴스를 포함하여 기존 데이터셋보다 5배 이상 규모가 큽니다. 이 데이터셋은 lane detection과 autonomous driving과 같은 응용에 필수적인 정보로 활용될 수 있습니다.

- **Performance Highlights**: 기존 데이터셋과 비교하여 OpenSatMap은 더 세밀하고 높은 해상도를 제공하며, 다양한 지리적 환경과 도로 유형을 포함하고 있어 대규모 지도 구축과 자율 주행 기술의 발전에 기여할 것으로 기대됩니다.



### SlowFast-VGen: Slow-Fast Learning for Action-Driven Long Video Generation (https://arxiv.org/abs/2410.23277)
- **What's New**: 본 논문에서는 SlowFast-VGen이라는 새로운 이중속도 학습 시스템을 소개합니다. 이는 행동 주도의 긴 비디오 생성을 위해 느린 학습과 빠른 학습을 통합하여, 일관되고 응답적인 비디오 생성을 가능하게 합니다.

- **Technical Details**: 이 시스템은 두 가지 주요 구성 요소로 이루어져 있습니다. 느린 학습(적용된 조건부 비디오 확산 모델)과 빠른 학습(Temporal LoRA 모듈 기반의 추론 시간 학습 전략)입니다. 빠른 학습 과정에서는 입력 및 출력에 기반하여 Temporal LoRA 매개변수를 업데이트하여 에피소드 메모리를 저장합니다. 또한, 느린 학습 알고리즘과 빠른 학습 루프를 결합하여 다중 에피소드 경험을 기억하여 기술 학습을 지원합니다.

- **Performance Highlights**: SlowFast-VGen은 FVD 점수에서 514를 기록하며 782을 초월하여 이전 모델들보다 뛰어난 성능을 보였습니다. 또한 0.37의 평균 장면 전환수로, 긴 비디오의 일관성을 유지하는 데 있어 우수함을 보여주었습니다.



### TOMATO: Assessing Visual Temporal Reasoning Capabilities in Multimodal Foundation Models (https://arxiv.org/abs/2410.23266)
- **What's New**: 본 논문에서는 다중 모달 재단 모델(Multimodal Foundation Models, MFMs)의 비디오 이해에 대한 시각적 시간 추론 능력을 평가하기 위한 새로운 벤치마크, TOMATO(Temporal Reasoning Multimodal Evaluation)를 제안합니다.

- **Technical Details**: TOMATO는 (1) Multi-Frame Gain, (2) Frame Order Sensitivity, (3) Frame Information Disparity라는 세 가지 원칙과 해당 메트릭을 기반으로 하여 구성되었습니다. 이 벤치마크는 1,484개의 질문으로 구성되며, 1,417개의 비디오에 적용됩니다.

- **Performance Highlights**: 현재 MFMs의 성능 평가에서 인간 모델 성능 간의 격차는 57.3%로 나타났습니다. MFMs는 개별 프레임에서 이벤트를 정확하게 인식할 수 있지만, 이러한 프레임을 연속적인 시퀀스로 해석하는 데 실패하는 기본적인 한계가 드러났습니다.



### EMMA: End-to-End Multimodal Model for Autonomous Driving (https://arxiv.org/abs/2410.23262)
Comments:
          Blog post: this https URL

- **What's New**: EMMA(End-to-end Multimodal Model for Autonomous driving)는 모듈식 접근법 대신, 센서 데이터를 직접 처리하여 운전 관련 작업을 수행하는 새로운 모델입니다. 다양한 센서 데이터를 자연어 텍스트로 변환하여 통합적으로 처리할 수 있는 기능이 특징입니다.

- **Technical Details**: EMMA는 미리 훈련된 멀티모달 대형 언어 모델(Gemini) 위에 구축되었으며, 카메라 이미지 및 자연어 텍스트를 입력으로 받아 특정 운전 작업을 수행합니다. EMMA는 비전 모듈과 행동 모듈 간의 기호 인터페이스를 제거하여 각 운전 목표를 공동으로 최적화합니다.

- **Performance Highlights**: EMMA는 nuScenes 데이터셋에서 최첨단 성능을 달성하였으며, Waymo Open Motion Dataset에서도 경쟁력 있는 결과를 보였습니다. 또한, 3D 객체 탐지 및 도로 그래프 추정 등 여러 인식 작업에서도 뛰어난 성능을 보였습니다. 하지만, 이미지 프레임 수 처리의 제한, LiDAR와 같은 3D 감지 모드의 부재로 인한 문제도 존재합니다.



### PointRecon: Online Point-based 3D Reconstruction via Ray-based 2D-3D Matching (https://arxiv.org/abs/2410.23245)
- **What's New**: 이 논문에서는 posed된 단일 RGB 비디오에서 새로운 온라인 포인트 기반 3D 재구성 방법인 PointRecon을 제안합니다. 이 모델은 장면의 글로벌 포인트 클라우드 표현을 유지하며, 새로운 이미지가 관찰될 때마다 포인트의 특징과 3D 위치를 지속적으로 업데이트합니다.

- **Technical Details**: PointRecon은 레이 기반의 2D-3D 특징 매칭 기술을 사용하여 새로운 이미지를 통해 기존 3D 포인트와 2D 특징을 정렬합니다. 이 방법은 이전 포인트 위치 예측의 오류에 강인성을 보입니다. 새로운 포인트는 이미지에서 기존 포인트 클라우드에 효과적으로 병합되며, 중복 포인트를 제거하는 모듈이 설계되어 있습니다.

- **Performance Highlights**: ScanNet 데이터셋에서의 실험 결과, 제안된 방법은 온라인 MVS 방식 중에서 최첨단 품질을 달성하며, 밀도 변화에 적응적으로 표면 세부 정보를 표현하여 더 높은 세부 묘사를 제공합니다.



### LGU-SLAM: Learnable Gaussian Uncertainty Matching with Deformable Correlation Sampling for Deep Visual SLAM (https://arxiv.org/abs/2410.23231)
- **What's New**: 이 논문에서는 Learnable Gaussian Uncertainty (LGU) 매칭을 제안하여 정확한 대응 구성에 중점을 두고 있습니다.

- **Technical Details**: LGU 매칭은 매칭 프레임 쌍에 대한 입력 의존성 Gaussian 분포를 생성하는 2D Gaussian 불확실성 모델을 설계하여 이루어집니다. 또한 다중 스케일 변형 상관 샘플링 전략을 통해 각 방향의 샘플링을 적응적으로 미세 조정합니다.

- **Performance Highlights**: 실제 및 합성 데이터 세트에 대한 광범위한 실험을 통해 LGU-SLAM의 효과성과 우수성을 입증하였습니다.



### Aligning Audio-Visual Joint Representations with an Agentic Workflow (https://arxiv.org/abs/2410.23230)
- **What's New**: 본 연구에서는 오디오-비주얼(Audio-Visual, AV) 데이터 간의 정렬 문제를 해결하기 위해 LLM 기반의 AVAgent를 제안합니다. AVAgent는 멀티모달 LLM을 사용하여 오디오와 비주얼 데이터를 언어 설명으로 변환하고, 이 데이터를 기반으로 정렬 여부를 판단하며 필요시 오디오 신호를 편집합니다.

- **Technical Details**: 제안된 방법에서 AVAgent는 도구 사용(tool use), 계획(planning), 반영(reflection) 단계를 순환적으로 수행하여 오디오 신호를 시각 데이터에 점진적으로 정렬합니다. 이 과정에서 AVAgent는 배경 잡음을 제거하고 데이터를 보강하는 전처리 작업을 수행하며, 이러한 작업 후 VLM(비전-언어 모델)을 통해 편집된 오디오 신호가 비디오와 잘 맞는지 평가합니다.

- **Performance Highlights**: Flick-SoundNet, VGG-Instruments 등 다양한 데이터 세트를 통해 실험한 결과, 제안된 방법이 기존 기준선에 비해 우수한 성능을 보였습니다. 특히, 선형 프로빙(linear probing), 미세 조정(fine-tuning) 분류, 비주얼 사운드 로컬라이제이션(sound localization) 등 다양한 다운스트림 작업에서 최첨단 성능을 입증했습니다.



### DiaMond: Dementia Diagnosis with Multi-Modal Vision Transformers Using MRI and PE (https://arxiv.org/abs/2410.23219)
Comments:
          Accepted by IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2025

- **What's New**: 이 논문은 DiaMond라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 MRI(자기공명영상)와 PET(양전자 방출 단층촬영) 데이터를 효과적으로 통합하기 위해 비전 트랜스포머(vision Transformers)를 사용합니다. DiaMond는 셀프 어텐션(self-attention)과 바이 어텐션(bi-attention) 메커니즘을 활용하여 두 가지 모달리티의 특징을 결합하며, 멀티모달 정규화(multi-modal normalization)을 통해 중복 의존성을 줄여 성능을 향상시킵니다.

- **Technical Details**: DiaMond는 MRI와 PET 데이터를 독립적으로 처리하는 두 개의 가지를 포함하는 구조입니다. 셀프 어텐션은 각각의 모달리티에서 고유한 특징을 추출하고, 바이 어텐션은 서로 겹치는 정보를 기반으로 모달리티 간의 상관 관계를 캡처합니다. RegBN이라는 멀티모달 정규화 기법이 적용되어 두 모달리티 간의 불필요한 의존성을 제거합니다. 이는 DiaMond가 보다 효율적으로 각 모달리티의 독특한 특징을 탐색할 수 있도록 합니다.

- **Performance Highlights**: DiaMond는 다양한 데이터셋에서 기존의 멀티모달 방법을 크게 초월하는 결과를 보였으며, AD(알츠하이머병) 진단에서 92.4%, AD-MCI-CN(경도 인지 장애) 분류에서 65.2%, AD와 FTD(전두측두엽 치매)의 구별 진단에서 76.5%의 균형 잡힌 정확도를 달성했습니다. 이 연구는 DiaMond의 강인성 또한 확인하였습니다.



### ELMGS: Enhancing memory and computation scaLability through coMpression for 3D Gaussian Splatting (https://arxiv.org/abs/2410.23213)
- **What's New**: 이 논문에서는 3D Gaussian Splatting 모델의 메모리 및 연산 확장성(enabling both memory and computation scalability) 향상을 위한 새로운 접근 방식을 제안합니다. 이를 위해 반복적 가지치기(iterative pruning) 전략을 도입하여 모델에 인코딩된 중복 정보를 제거하고, 최적화 전략에 미분 가능 양자화(differentiable quantization)와 엔트로피 인코딩(entropy coding) 추정기를 포함하여 모델의 압축성을 향상시킵니다.

- **Technical Details**: 이 연구는 최적화된 3D Gaussian 씬에서 불필요한 Gaussian을 제거하기 위해 기울기와 불투명도(opacity) 수준에 따라 가지치기(pruning)를 실시합니다. 이후 양자화 인식 훈련(Quantization-Aware Training, QAT)을 통해 씬 사이즈를 추가로 압축하고, 마지막으로 엔트로피 인코딩을 적용하여 성능 향상과 함께 상당한 압축 이득을 가져옵니다. 이 방법은 IoT 기기와 AR/VR 헤드셋에서의 적용 가능성을 높입니다.

- **Performance Highlights**: 제안된 방법은 인기 있는 벤치마크에서 효과성을 입증하였으며, 리소스가 제한된 장치에서도 널리 배포 가능한 솔루션을 열어줍니다. 이 연구는 씬 충실도(scene fidelity)와 압축 간의 균형을 개선하여 기존 방법보다 우수한 성과를 달성하였습니다.



### HEX: Hierarchical Emergence Exploitation in Self-Supervised Algorithms (https://arxiv.org/abs/2410.23200)
- **What's New**: 이 논문에서는 Self-Supervised Learning (SSL) 알고리즘의 기존 한계를 극복하기 위한 새로운 방법론을 제안합니다. 특히, 학습 과정에서 나타나는 계층적 구조를 활용하는 가중형 정규화 기법을 소개합니다. 이를 통해 데이터 샘플 간의 지역적인 차원 붕괴 현상을 보다 효과적으로 다룰 수 있습니다.

- **Technical Details**: 제안된 알고리즘은 InfoNCE 손실의 분모를 지역 계층적 정규화(local hierarchical regularization)와 전역 정규화(global regularization)로 나누어 가중치를 부여합니다. 이 분해는 배치 내 샘플의 코사인 유사도 분포를 분석하여 진행되며, 특정 유사도 임계값(ϵ) 이상인 샘플들을 동일한 계층에 속하는 것으로 간주합니다.

- **Performance Highlights**: 이 연구에서는 제안된 계층적 발생 활용(Hex) 접근법이 다양한 SSL 알고리즘에 통합될 수 있으며, Imagenet 데이터셋을 대상으로 한 실험에서 기반 SSL 방법과 비교하여 최대 5.6%의 분류 정확도 향상을 보여주었습니다.



### Continuous Spatio-Temporal Memory Networks for 4D Cardiac Cine MRI Segmentation (https://arxiv.org/abs/2410.23191)
Comments:
          Accepted to WACV 2025

- **What's New**: 본 논문에서는 심장 cine 자기 공명 영상(cMR)의 전체 이미지 시퀀스에서 유용한 시간 정보를 활용하여 심장 전반에 대한 세그멘테이션(sementation) 성능을 향상시키는 새로운 방법을 제안합니다. 특히, CSTM(Continuous Spatio-Temporal Memory) 네트워크를 적용하여 반자율적으로 전체 심장과 전체 시퀀스를 분할하는 접근 방식을 고안했습니다.

- **Technical Details**: 제안된 CSTM 네트워크는 심장 해부학 구조의 공간적, 스케일, 시간적, 평면 간 연속성을 최대한 활용하여 정확한 4D segmentation을 수행합니다. 이 모델은 key encoder, value encoder 및 mask decoder로 구성된 STCN(Spatio-Temporal Convolutional Networks)에 기반을 두고 있으며, query와 memory 프레임 간의 관계를 쉽게 성립할 수 있도록 설계되었습니다.

- **Performance Highlights**: 세 가지 cMR 데이터셋에서 광범위한 실험을 통해, 제안한 방법이 특히 세그멘테이션이 어려운 영역에서도 4D cMR segmentation 성능을 효과적으로 개선한다는 결과를 도출했습니다.



### Fourier Amplitude and Correlation Loss: Beyond Using L2 Loss for Skillful Precipitation Nowcasting (https://arxiv.org/abs/2410.23159)
Comments:
          Accepted by NeurIPS 2024. Camera-ready submission

- **What's New**: 본 연구에서는 기존의 모델 아키텍처에 의존하지 않고, 새로운 손실 함수인 Fourier Amplitude and Correlation Loss (FACL)을 제안합니다. 이 손실 함수는 예측의 고주파 패턴을 복원하는 데 중점을 두며, 전통적인 MSE 손실을 대신하여 더 선명한 강수 예측을 가능하게 합니다.

- **Technical Details**: FACL은 두 가지 주요 손실 항목으로 구성됩니다: Fourier Amplitude Loss (FAL)와 Fourier Correlation Loss (FCL). FAL은 예측의 푸리에 진폭을 정규화하고, FCL은 누락된 위상 정보를 보완합니다. 이러한 손실 항목의 결합은 명암도와 공간 구조를 더욱 향상시키며, FAL과 FCL 사이의 학습 메커니즘을 통해 점진적으로 훈련됩니다.

- **Performance Highlights**: 실험 결과, FACL은 MSE로 기존 예측보다 더 사실적이고 기술적인 성능이 우수한 결과를 보여주었습니다. 또한, 새로운 평가 지표인 Regional Histogram Divergence (RHD)를 통해 이미지 패턴 간의 유사성을 더욱 정량적으로 측정할 수 있게 되었습니다. 이러한 개선은 강수 예측의 정확성을 높이는 데 기여하고 있습니다.



### Revisiting MAE pre-training for 3D medical image segmentation (https://arxiv.org/abs/2410.23132)
Comments:
          Arxiv Preprint. Currently under Review

- **What's New**: 이 논문에서는 Self-Supervised Learning (SSL) 접근법을 활용하여 44,000개의 3D 뇌 MRI 데이터셋을 기반으로 기존 SSL 방법보다 성능 향상을 이루었습니다. 특히, Residual Encoder U-Net 아키텍처를 사용하여 3D 의료 이미지 분석의 한계를 극복하고자 했습니다.

- **Technical Details**: 제안된 모델은 Masked Auto Encoders (MAEs) 개념을 3D CNN에 최적화하여 구성되었습니다. 실험은 5개의 개발 및 8개의 테스트 뇌 MRI 분할 데이터셋에서 진행되었습니다. 논문에서 다룬 주요 기술적 세부사항으로는 z-score 정규화, poly-learning rate 조정을 통한 SGD 최적화 방법 등이 있습니다.

- **Performance Highlights**: 제안된 모델은 기존 nnU-Net 베이스라인보다 평균 3 Dice 포인트 향상된 성능을 보였으며, 7개의 방법 중 평균 순위 2를 기록하여 뛰어난 안정성을 자랑합니다.



### Unified Triplet-Level Hallucination Evaluation for Large Vision-Language Models (https://arxiv.org/abs/2410.23114)
Comments:
          18 pages, 8 figures

- **What's New**: 본 논문은 대형 비전-언어 모델(LVLM)에서 발생하는 객체 및 관계 환각(hallucination)을 동시에 평가하기 위한 통합 프레임워크를 설계하였습니다.

- **Technical Details**: LVLM의 응답에서 추출된 (객체, 관계, 객체) 트리플렛(triplet) 기반의 환각 평가를 통해 환각 유형을 통합적으로 분석할 수 있으며, Tri-HE라는 새로운 평가 벤치마크를 도입하여 LVLM의 환각 문제를 보다 세밀하게 평가할 수 있습니다.

- **Performance Highlights**: 이 연구에서 제안한 방법은 기존 LVLM이 갖고 있는 관계 환각 문제를 해결함으로써, LLaVA-1.5 모델이 모든 오픈 소스 모델을 초월하는 성능을 기록하였으며, 강력한 GPT-4V와 동등한 성능을 보여주었습니다.



### NASM: Neural Anisotropic Surface Meshing (https://arxiv.org/abs/2410.23109)
Comments:
          SIGGRAPH Asia 2024 (Conference Track)

- **What's New**: 이 논문은 NASM이라는 새로운 학습 기반 방법을 소개하며, 비등방성 표면 메싱(anisotropic surface meshing)에 초점을 맞추고 있습니다. 그래프 신경망(graph neural network)을 사용하여 입력 메쉬를 고차원 유클리드 임베딩 공간(high-dimensional Euclidean embedding space)으로 변환하여 곡률 기반의 비등방성 메트릭을 보존합니다.

- **Technical Details**: 제안된 방법은 고차원 엣지 벡터(high-dimensional edge vectors) 간의 내적 손실(dot product loss)을 사용하여 비등방성 메트릭(anisotropic metric)을 유지하고 계산 시간을 대폭 단축시키며 확장성을 증가시킵니다. 그런 다음, 생성된 고차원 임베딩에서 새로운 기능 민감형 리메싱(feature-sensitive remeshing)을 제안하여 날카로운 기하학적 특징을 자동으로 캡처합니다. 고차원 법선 메트릭(high-dimensional normal metric)을 정의하고 이를 바탕으로 자동 미분(automatic differentiation)을 통해 고차원 중심 포로노이 분할(Centroidal Voronoi Tessellation, CVT) 최적화를 수행하여 기하학적 특징과 곡률 비등방성을 동시에 보존합니다.

- **Performance Highlights**: 제안된 방법은 Thingi10K 데이터셋에서의 다양한 표면 모델을 포함하여 최신 기술(state-of-the-art)과 비교하여 실험 결과를 평가하였습니다. 또한, Multi-Garment Network 데이터셋 및 FAUST 인간 데이터셋의 광범위한 3D 형태에도 테스트되었습니다.



### Decoupling Semantic Similarity from Spatial Alignment for Neural Networks (https://arxiv.org/abs/2410.23107)
Comments:
          Accepted at NeurIPS2024

- **What's New**: 본 논문에서는 Representational Similarity Matrices (RSMs)의 기존 계산 방법의 한계를 제시하고, 공간적 위치에 영향을 받지 않는 semantic RSMs를 제안하여 이미지 응답 간의 유사성을 측정합니다.

- **Technical Details**: 우리는 semantic RSMs를 제안하여 공간적 순열에 불변이며, 집합 매칭 문제로 형성된 semantic 유사성을 측정합니다. 이 방법은 CNN과 ViT의 이미지 응답을 비교하여, 두 모델의 유사성 구조를 파악합니다.

- **Performance Highlights**: 제안한 semantic RSMs는 spatio-semantic RSMs에 비해 이미지 검색 성능을 향상시키고, 분류기 표현 간의 유사성을 더 잘 반영합니다. 또한 컴퓨팅 복잡성을 줄이기 위한 근사화 방법을 소개합니다.



### Automated Image-Based Identification and Consistent Classification of Fire Patterns with Quantitative Shape Analysis and Spatial Location Identification (https://arxiv.org/abs/2410.23105)
- **What's New**: 이 연구는 화재 조사관들이 사용할 수 있는 정량적 화재 패턴 분류 프레임워크를 제안하고 있습니다. 이는 기존의 주관적 해석을 개선하여 일관성과 정확성을 목표로 합니다.

- **Technical Details**: 프레임워크는 네 가지 구성 요소로 통합됩니다: (1) 인간-컴퓨터 상호작용(Human-Computer Interaction)을 이용해 표면에서 화재 패턴을 추출하고, (2) 비율 기반 랜덤 포레스트 모델(Aspect Ratio-based Random Forest Model)을 사용하여 화재 패턴 형태를 분류하며, (3) 화재 장면 포인트 클라우드(Point Cloud) 분할을 통해 화재 영향을 받은 지역을 정확히 식별하고 2D 화재 패턴을 3D 장면에 매핑합니다. (4) 또한 화재 패턴과 실내 요소 간의 공간적 관계를 지원하여 화재 장면 해석을 돕습니다.

- **Performance Highlights**: 이 프레임워크의 분류 결과는 합성 데이터(Synthetic Data)에서 93%의 정밀도를, 실화(real fire patterns)에서는 83%의 정밀도를 달성했습니다.



### First Place Solution to the ECCV 2024 ROAD++ Challenge @ ROAD++ Atomic Activity Recognition 2024 (https://arxiv.org/abs/2410.23092)
- **What's New**: 2024 ECCV ROAD++ Challenge의 Track 3에서의 다중 레이블 원자 활동 인식(atomic activity recognition) 과제를 해결하기 위한 새로운 기술 솔루션을 제시합니다. 이 솔루션은 작은 객체, 단일 객체와 객체 그룹 구분, 모델 오버피팅을 개선합니다.

- **Technical Details**: 다중 브랜치(activity recognition framework) 원자 활동 인식 프레임워크를 구성하고, 다양한 모델 앙상블(model ensembling) 전략과 데이터 증강(data augmentation) 방법을 개발하였습니다. 이는 비디오 프레임을 뒤집고 도로 토폴로지를 조작하여 샘플 공간을 확장합니다.

- **Performance Highlights**: ROAD++ Challenge 2024의 Track 3 테스트 세트에서 1위를 차지하고, mAP(Mean Average Precision) 69%를 달성하였습니다.



### CausalDiff: Causality-Inspired Disentanglement via Diffusion Model for Adversarial Defens (https://arxiv.org/abs/2410.23091)
Comments:
          accepted by NeurIPS 2024

- **What's New**: 이 논문에서 제안하는 CausalDiff 모델은 주로 보이지 않는 공격에 대해 효과적인 방어 메커니즘을 제공하며, 기존의 방법들과 비교하여 뛰어난 성능을 보여준다.

- **Technical Details**: CausalDiff는 퍼지 생성 데이터의 필수 라벨 원인 요소(label-causative factors)와 비원인 요소(label-non-causative factors)를 구분하여 데이터 생성을 지원하는 구조적 인과 모델(Structural Causal Model, SCM)에 기반하고 있다. 이는 Denoising Diffusion Probabilistic Model (DDPM)을 backbone으로 활용하여 데이터를 조건부로 생성하는 방법을 채택하였다.

- **Performance Highlights**: CausalDiff는 CIFAR-10에서 86.39%의 강건성(+4.01%), CIFAR-100에서 56.25%의 강건성(+3.13%), GTSRB에서 82.62%의 강건성(+4.93%)을 기록하며, 최신 방어 방법들과 비교하여 뛰어난 성능을 보여준다.



### PIP-MM: Pre-Integrating Prompt Information into Visual Encoding via Existing MLLM Structures (https://arxiv.org/abs/2410.23089)
- **What's New**: 이번 연구에서는 Multimodal Large Language Models (MLLMs)의 기존 이미지 인코딩 프로세스를 개선하기 위해 새로운 프레임워크인 PIP-MM을 제안합니다. 이는 Prompt 정보를 사전에 통합하여 이미지와 텍스트 간의 정합성을 향상시킵니다.

- **Technical Details**: PIP-MM은 기존 MLLMs에서 프리 인티그레이션을 통해 동작합니다. LLM이 입력 Prompt를 벡터화 한 후, Multi-Layer Perceptron (MLP)를 훈련하여 이미지 인코더의 클래스 임베딩을 교체해 시각 데이터와 텍스트 정보를 조기에 융합합니다.

- **Performance Highlights**: PIP-MM은 다양한 벤치마크에서 시험한 결과, 자동 및 수동 평가에서 우수한 성능을 보였으며, 특히 시각적 토큰을 절반으로 줄인 상태에서도 우수한 생성 결과를 유지했습니다.



### S3PT: Scene Semantics and Structure Guided Clustering to Boost Self-Supervised Pre-Training for Autonomous Driving (https://arxiv.org/abs/2410.23085)
Comments:
          Accepted for WACV 2025

- **What's New**: S3PT (Scene Semantics and Structure guided Pre-Training)는 자율 주행에 적합한 클러스터링 기반의 자가 지도 전처리 기법으로, 다양한 객체 크기와 클래스 불균형 문제를 해결하는 데 중점을 두고 제안되었습니다.

- **Technical Details**: 이 방법은 세 가지 주요 구성 요소를 포함합니다: 1) 희귀 클래스(예: 오토바이, 동물)의 표현 개선을 위한 의미 분포 일관적 클러스터링; 2) 객체 크기의 불균형을 처리하기 위한 객체 다양성 일관적 공간 클러스터링; 3) 장면의 기하학적 정보에 기초한 깊이 유도 공간 클러스터링.

- **Performance Highlights**: S3PT는 nuScenes, nuImages, Cityscapes 데이터셋을 통한 2D 의미 세분화 및 3D 객체 탐지 작업에서 성능 향상을 보여주며, 자율 주행 데이터에서의 사전학습을 통한 일반화 능력이 강화되었습니다.



### First Place Solution to the ECCV 2024 ROAD++ Challenge @ ROAD++ Spatiotemporal Agent Detection 2024 (https://arxiv.org/abs/2410.23077)
- **What's New**: 이 보고서는 2024 ECCV ROAD++ 챌린지의 Track 1에 대한 팀의 솔루션을 제시합니다. Track 1은 연속 비디오 프레임에서 도로 에이전트에 대한 '에이전트 튜브'를 구축하는 spatiotemporal agent detection 과제를 다룹니다.

- **Technical Details**: 주요 기술적 접근 방식으로는 extreme-size object detection heads, dual-stream detection model, feature fusion module, multi-branch detection framework, data augmentation techniques 및 loss function 개선이 포함됩니다. 특히 dual-stream detection model은 low-light enhancement stream을 통합하여 저조도 환경에서 성능을 향상시킵니다.

- **Performance Highlights**: 이 팀은 2024 ROAD++ 챌린지 Track 1의 테스트 세트에서 1위를 기록하며 30.82% 평균 video-mAP을 달성했습니다.



### RSNet: A Light Framework for The Detection of Multi-scale Remote Sensing Targets (https://arxiv.org/abs/2410.23073)
- **What's New**: 최근 합성 개구 레이더(SAR) 선박 탐지에서 딥 러닝 기술의 정확성과 속도가 크게 향상되었습니다. 이 논문에서는 RSNet이라는 경량 구조를 제안하여 SAR 이미지에서의 선박 탐지 능력을 개선하려고 합니다.

- **Technical Details**: RSNet은 Waveletpool-ContextGuided(WCG) 백본과 Waveletpool-StarFusion(WSF) 헤드를 특징으로 하며, 파라미터 수를 줄이면서도 높은 정확도를 제공합니다. Lightweight-Shared(LS) 모듈을 통해 탐지 헤드의 파라미터 로드를 최소화합니다. 이 논문은 SAR Ship Detection Dataset(SSDD)와 High-Resolution SAR Image Dataset(HRSID)에서의 실험을 통해 RSNet이 경량 설계와 탐지 성능의 강력한 균형을 이루었음을 보여줍니다.

- **Performance Highlights**: RSNet은 1.49M의 파라미터로 SSDD와 HRSID에서 각각 72.5%와 67.6%의 mAP(.50:95)를 달성하여 많은 최첨단 탐지기를 초월했습니다.



### CNN Explainability with Multivector Tucker Saliency Maps for Self-Supervised Models (https://arxiv.org/abs/2410.23072)
Comments:
          29 pages, 20 figures

- **What's New**: 이번 연구에서는 Convolutional Neural Networks (CNNs)의 해석 가능성에 대한 새로운 접근법인 Tucker Saliency Map (TSM) 방법을 소개합니다. 이는 기존의 EigenCAM과 달리, feature maps의 고유한 구조를 더 잘 포착하여 더 정확한 saliency maps를 생성합니다.

- **Technical Details**: Tucker tensor decomposition을 적용하여 singular vectors와 values를 생성하며, 이를 통해 고화질의 saliency maps를 생성합니다. 또한, EigenCAM과 TSM을 확장한 Multivec-EigenCAM과 Multivector Tucker Saliency Maps (MTSM)를 도입하여 모든 singular vectors와 values를 활용합니다.

- **Performance Highlights**: 정량적 평가 결과, TSM, Multivec-EigenCAM, 및 MTSM은 label-dependent 방법들과 경쟁력 있는 성능을 보였으며, TSM은 EigenCAM에 비해 explainability를 약 50% 향상시켰고, MTSM은 self-supervised 모델에서 최고의 결과를 달성했습니다.



### VisAidMath: Benchmarking Visual-Aided Mathematical Reasoning (https://arxiv.org/abs/2410.22995)
Comments:
          58 pages, 28 figures

- **What's New**: 새로운 연구에서는 시각적 정보를 활용한 수학 문제 해결(Visual Aided Mathematical Problem Solving, MPS) 과정을 평가하기 위한 VisAidMath 벤치마크를 소개합니다. 이 벤치마크는 1,200개의 난이도 있는 문제를 포함하며, 다양한 출처에서 수집된 문제와 답변을 평가합니다.

- **Technical Details**: VisAidMath 벤치마크는 수학적 질문을 시각적 맥락(Visual Context), 질문(Question), 시각적 도움(Visual Aids), 답변(Answer) 네 부분으로 나누어 설계되었습니다. 이 벤치마크는 명확한 시각적 정보를 포함하고 있으며, 값은 LaTeX 형식으로 정리되어 있습니다. 이 연구에서는 통계적으로 10개의 주요 LLMs 및 LMMs의 성능을 분석하였습니다.

- **Performance Highlights**: 주요 모델인 GPT-4V는 시각적 정보를 활용한 추론 과제에서 평균 45.33%의 정확도를 보였으며, 이는 전문적인 시각적 도움을 제공받았을 때도 2점 감소하였습니다. 또한, SOTA 모델들은 약 50%의 평균 정확도에 그쳤으며, 생성된 시각적 보조 자료는 5%의 n-gram 유사성만을 보였습니다.



### LumiSculpt: A Consistency Lighting Control Network for Video Generation (https://arxiv.org/abs/2410.22979)
- **What's New**: 이 논문에서는 비디오 생성에 있어 조명 제어를 정밀하고 일관성 있게 수행할 수 있는 새로운 모델인 LumiSculpt를 제안합니다. 또한, 새로운 경량 데이터셋인 LumiHuman을 구축하여 조명 데이터 부족 문제를 해결했습니다.

- **Technical Details**: LumiSculpt는 텍스트에서 비디오(text-to-video) 생성 환경에서 조명 방향, 위치 및 경로를 제어하기 위한 모듈로, 가벼운 LumiHuman 데이터셋을 활용하여 22만 개 이상의 비디오와 조명 파라미터를 포함하고 있습니다. LumiSculpt는 Light control module과 Decoupling loss를 사용하여 조명 특성과 생성 모델을 효과적으로 분리합니다.

- **Performance Highlights**: 실험 결과, LumiSculpt는 조명 강도, 방향, 경로를 정확하게 생성하며, 비디오 생성의 통일성을 유지함과 동시에 콘텐츠 다양성을 확보하여 최첨단 성능을 달성했습니다.



### EnsIR: An Ensemble Algorithm for Image Restoration via Gaussian Mixture Models (https://arxiv.org/abs/2410.22959)
Comments:
          10 pages for main manuscript, additional 17 pages for appendix, 18 figures, 17MB

- **What's New**: 이번 연구에서는 이미지 복원에서의 앙상블 학습(Ensemble Learning)의 새로운 접근 방식을 제안합니다. 기존의 복원 모델들을 활용하여 추론 단계에서 효율적인 앙상블 방법을 고안하였으며, 가우시안 혼합 모델(Gaussian Mixture Models, GMMs)을 통해 앙상블 가중치를 추정하는 새로운 방식을 도입했습니다.

- **Technical Details**: 제안된 방법은 이미지 복원 문제를 GMMs로 재정립하고, 기대 최대화(expectation maximization, EM) 알고리즘을 사용하여 예측값을 모집단으로 집계하기 위한 앙상블 가중치를 추정합니다. 우리는 참조 세트(reference set)에서 범위별 앙상블 가중치를 추정하고, 이를 효율적인 앙상블 추론을 위해 룩업 테이블(lookup table, LUT)에 저장합니다.

- **Performance Highlights**: 제안된 알고리즘은 3개의 이미지 복원 작업(초해상도, 디블러링, 강수 제거)에서 14개의 기준 벤치마크를 통해 회귀 기반 방법과 평균 앙상블 접근 방식을 지속적으로 능가했습니다. 이는 기존의 앙상블 방법과 비교할 때 뛰어난 성능을 보이며, 다양한 사전 훈련된 이미지 복원 모델과의 통합이 용이합니다.



### Efficient Adaptation of Pre-trained Vision Transformer via Householder Transformation (https://arxiv.org/abs/2410.22952)
- **What's New**: 본 연구는 Vision Transformers (ViTs)의 하이퍼파라미터 조정을 위해 Singular Value Decomposition (SVD)을 활용한 새로운 Parameter-Efficient Fine-Tuning (PEFT) 접근 방식을 제안합니다. 이는 Householder 변환을 이용하여 고유 행렬을 효율적으로 구성하고, 각 레이어의 특성에 맞춰 유연하게 파라미터를 조정할 수 있게 합니다.

- **Technical Details**: 제안된 방법은 Householder 변환을 사용하여 각 레이어에 적응 가능한 다이아곤 행렬을 학습하며, 이는 고유 행렬을 대체합니다. Householder 변환은 단일 벡터로부터 유도되기 때문에 파라미터 효율성을 높입니다. 이를 통해 서로 다른 레이어에서 적응 행렬의 다양한 계급을 생성할 수 있어, PEFT의 유연성을 증가시킵니다.

- **Performance Highlights**: 표준 다운스트림 비전 작업에 대한 실험 결과, 제안한 방법은 다양한 ViT 버전에서 우수한 하이퍼파라미터 조정 성능을 보여주었습니다. 이 연구는 PEFT의 기존 접근 방식과는 다른 새로운 방향을 제시함으로써, 적응 성능과 파라미터 효율성 사이의 우수한 균형을 달성합니다.



### AdaptiveISP: Learning an Adaptive Image Signal Processor for Object Detection (https://arxiv.org/abs/2410.22939)
Comments:
          Accepted at NeurIPS2024

- **What's New**: AdaptiveISP는 각 입력에 맞춰 ISP(이미지 신호 프로세서) 파이프라인과 파라미터를 동적으로 조정할 수 있는 새로운 접근법입니다. 이 시스템은 객체 감지 성능을 극대화하기 위해 딥 강화 학습(deep reinforcement learning)을 사용합니다.

- **Technical Details**: AdaptiveISP는 입력 이미지에 따라 최적의 ISP 파이프라인을 생성하고 관련 파라미터를 자동으로 조정하여 객체 감지의 정확도를 극대화합니다. 이 접근법은 마르코프 결정 과정(Markov Decision Process) 모형을 기반으로 하여 각 단계에서 단일 ISP 모듈을 선택하며, 이전 단계의 출력물을 다음 단계 입력으로 사용하여 최적의 모듈을 찾습니다.

- **Performance Highlights**: AdaptiveISP는 LOD, OnePlus 및 synthetic raw COCO 데이터셋에서 이전의 최첨단 방법들을 초과하는 성능을 보여주며, 고정밀 ISP 파이프라인과 저지연 파이프라인 간의 동적 전환 능력을 입증했습니다.



### Bringing NeRFs to the Latent Space: Inverse Graphics Autoencoder (https://arxiv.org/abs/2410.22936)
- **What's New**: 본 연구에서는 인버스 그래픽스(autoencoder)와 3D 기하학을 활용한 새로운 기법을 도입하여 2D 잠재 공간에서의 이미지 자동 인코딩 문제를 해결하는 Inverse Graphics Autoencoder (IG-AE)를 제안합니다.

- **Technical Details**: IG-AE는 3D-aware latent space를 적용하여 기본 이미지 자동 인코더의 잠재 공간을 3D 장면과 정합시키고, NeRF(Neural Radiance Fields)를 통합하는 훈련 파이프라인을 구현하여 잠재 기반 학습을 지원합니다.

- **Performance Highlights**: 실험을 통해 IG-AE를 사용하여 훈련된 Latent NeRF가 표준 자동 인코더 대비 향상된 품질을 갖추며, NeRF의 훈련 및 렌더링 속도도 개선되었습니다.



### An Individual Identity-Driven Framework for Animal Re-Identification (https://arxiv.org/abs/2410.22927)
Comments:
          10 pages

- **What's New**: 본 논문에서는 동물 재식별 (Animal ReID)을 위한 새로운 접근 방식인 IndivAID 프레임워크를 제안합니다. 이 프레임워크는 Vision-Language 모델인 CLIP의 크로스 모달 특성을 활용하여 동물의 개별 텍스트 설명을 생성하는 두 단계의 프로세스로 구성됩니다.

- **Technical Details**: IndivAID는 두 단계로 구성됩니다: 첫 번째 단계에서는 이미지에서 개별적인 의미 정보를 추출하여 이미지 및 개체에 대한 설명을 생성하는 텍스트 설명 생성기를 학습합니다. 두 번째 단계에서는 이 텍스트 설명을 동적으로 통합하여 개별적인 피처를 강조하는 주의 모듈을 통해 시각적 개념 학습을 개선합니다.

- **Performance Highlights**: 여덟 개의 벤치마크 데이터셋과 실제 Stoat 데이터셋에 대한 평가 결과, IndivAID는 최첨단 방법들과 비교하여 효과성과 적용가능성을 보여주었습니다. 특히 다양한 특성을 가진 동물의 재식별에서 우수한 성능을 발휘했습니다.



### High-Fidelity Document Stain Removal via A Large-Scale Real-World Dataset and A Memory-Augmented Transformer (https://arxiv.org/abs/2410.22922)
Comments:
          Accepted by WACV2025

- **What's New**: StainDoc라는 대규모 고해상도 데이터셋을 소개하며, 이는 문서의 얼룩 제거를 위한 첫 번째 데이터셋입니다. 이 데이터셋은 5,000쌍 이상의 얼룩이 있는 문서 이미지와 깨끗한 문서 이미지를 포함하고 있어 다양한 얼룩 유형과 문서 배경을 포괄합니다.

- **Technical Details**: StainRestorer는 Transformer 기반의 문서 얼룩 제거 모델로, 메모리 증강 Transformer 아키텍처를 활용하여 문서 내의 세부적인 얼룩 표현을 계층적으로 캡처합니다. Stain Removal Transformer(SRTransformer)는 향상된 공간적 관심 메커니즘과 채널 관심 메커니즘을 결합하여 얼룩 제거 중 문서의 내용을 보존합니다.

- **Performance Highlights**: StainRestorer는 StainDoc 데이터셋 및 그의 변형(StainDoc_Mark 및 StainDoc_Seal)에서 최첨단 방법들에 비해 우수한 성능을 보여주었습니다. 이는 문서 얼룩 제거를 위한 새로운 기준을 설정하였습니다.



### UniRiT: Towards Few-Shot Non-Rigid Point Cloud Registration (https://arxiv.org/abs/2410.22909)
Comments:
          21 pages, 14 figures, under review

- **What's New**: 이 연구는 몇 개의 샘플만 있을 때 비강체(Non-rigid) 포인트 클라우드 등록 문제를 다루며, 새로운 프레임워크인 UniRiT를 제안합니다. 이를 통해 복잡한 비강체 변환 패턴을 강체(Rigid)와 작은 비강체 변환으로 분해할 수 있음을 보입니다.

- **Technical Details**: UniRiT는 두 단계 등록 전략을 채택하여, 먼저 소스와 타겟 포인트 클라우드의 중심을 정렬한 후 비강체 변환으로 등록을 세분화하여 문제의 복잡성을 크게 줄입니다. 이 과정은 각 포인트에 대해 고유한 변위 벡터를 적용하여 소스 포인트 클라우드를 도출하고, 이를 통해 효율적인 변환 패턴 학습을 가능하게 합니다.

- **Performance Highlights**: MedMatch3D라는 실제 인간 장기로 구성된 새로운 데이터셋에서 UniRiT는 기존 방법보다 94.22% 개선된 성능을 달성하며 최첨단(State-of-the-art) 성능을 입증했습니다. 이 결과는 저자들이 제안한 몇 샷 비강체 등록 접근 방식을 통해 달성되었습니다.



### HelloMeme: Integrating Spatial Knitting Attentions to Embed High-Level and Fidelity-Rich Conditions in Diffusion Models (https://arxiv.org/abs/2410.22901)
Comments:
          11 pages, 7 figures, 2 tables

- **What's New**: 이번 연구에서는 텍스트-이미지( text-to-image ) 기본 모델에 어댑터를 삽입하는 효과적인 방법을 제안합니다. 이 방법은 기본 모델의 일반화 능력을 유지하면서 복잡한 다운스트림 작업을 실행할 수 있게 해줍니다. 특히 이 접근법은 2D 피처 맵과 관련된 주의 메커니즘을 최적화하여 어댑터의 성능을 향상시킵니다.

- **Technical Details**: 제안된 방법은 세 가지 모듈로 구성됩니다: HMReferenceNet, HMControlNet, HMDenoisingNet. 첫 번째 모듈은 참조 이미지에서 선명한 피처를 추출하는 완전한 SD1.5 UNet이고, 두 번째 모듈은 고급 피처를 추출하여 UNet의 잠재 공간에 매핑합니다. 마지막 모듈은 노이즈 제거 모델로, HMReferenceNet과 HMControlNet에서 피처를 수신하여 참조 이미지에 새로운 머리 자세와 표정을 부여하는 이미지를 생성합니다. 이 과정에서 Spatial Knitting Attention (SK Attention) 메커니즘을 활용하여 2D 피처 맵의 구조 정보를 자연스럽게 보존합니다.

- **Performance Highlights**: 이 접근 방식은 meme 비디오 생성 작업에서 유의미한 결과를 보여주었으며, 널리 사용되는 여러 Stable Diffusion 응용 프로그램에 대해 강력한 성능을 발휘합니다. 특히, SD1.5 파생 모델과의 호환성이 뛰어나고, 이를 통해 오픈 소스 커뮤니티에서의 활용 가능성을 제시합니다.



### Wormhole Loss for Partial Shape Matching (https://arxiv.org/abs/2410.22899)
Comments:
          Accepted for publication at the conference on Neural Information Processing Systems (NeurIPS) 2024

- **What's New**: 이 논문에서는 부분 형상 대응(partial shape correspondence) 문제를 해결하기 위한 새로운 손실 함수(loss function)를 제안하고 있습니다. 이는 표면의 두 점 간의 일관된 거리(consistency distances)를 효과적으로 포착하는 새로운 기준을 도입하여, 부분 표면이 포함된 매칭 처리에서 발생할 수 있는 왜곡을 방지합니다.

- **Technical Details**: 제안된 방법은 지오데식 거리(geodesic distance)를 활용하여 부분 표면에서의 일관된 쌍을 정의하는 데 중점을 두고 있습니다. 새로운 손실 함수는 부분 표면에서 계산된 지오데식 쌍 거리와 경계점(boundary point) 간 거리 및 수치 공간(embedding space)에서의 거리 측정간의 관계를 반영하여 일관된 쌍을 찾도록 설계되었습니다.

- **Performance Highlights**: 이 새로운 접근 방식은 SHREC'16 CUTS 및 HOLES 데이터 세트와 PFAUST 데이터 세트에서 가장 최근의 SOTA(state-of-the-art) 결과를 달성했습니다.



### YOLOv11 for Vehicle Detection: Advancements, Performance, and Applications in Intelligent Transportation Systems (https://arxiv.org/abs/2410.22898)
Comments:
          16 pages

- **What's New**: 이 논문에서는 차량 탐지 작업에 exclusive하게 초점을 맞춘 YOLO11에 대한 자세한 분석을 제공합니다. YOLO11은 이전 모델들의 성공을 기반으로 하여 탐지 속도, 정확도 및 복잡한 환경에서의 강인성을 향상시키도록 설계된 아키텍처 개선을 도입하였습니다.

- **Technical Details**: YOLO11은 다중 차량 유형(차량, 트럭, 버스, 오토바이, 자전거)을 포함하는 포괄적인 데이터셋을 사용하여 성능을 평가하였으며, precision, recall, F1 score, mean average precision (mAP)과 같은 메트릭을 사용하여 성능을 분석하였습니다.

- **Performance Highlights**: YOLO11은 이전 모델(YOLOv8 및 YOLOv10)을 능가하여 작은 차량 및 가려진 차량 탐지에서 우수한 성능을 보이며, 실시간 응용 프로그램에 적합한 경쟁력 있는 추론 시간을 시현합니다. 복잡한 차량 기하학의 탐지에서 중요한 개선사항을 보여줘 효율적이고 확장 가능한 차량 탐지 시스템 개발에 기여할 것으로 기대됩니다.



### Effective and Efficient Adversarial Detection for Vision-Language Models via A Single Vector (https://arxiv.org/abs/2410.22888)
- **What's New**: 이번 논문에서는 Visual Language Models (VLMs)에 대한 새로운 대규모 적대적 이미지 데이터셋 RADAR를 구축하고, 이를 이용해 NEARSIDE라는 새로운 적대적 이미지 탐지 방법을 제안합니다. 이 방법은 VLMs의 숨겨진 상태로부터 도출된 단일 벡터(공격 방향)를 활용하여 적대적 이미지를 효과적으로 탐지합니다.

- **Technical Details**: RADAR 데이터셋은 MiniGPT-4와 LLaVA를 공격하기 위해 다양한 유해 내용을 가진 적대적 이미지를 생성하여 구성되었습니다. 데이터셋에는 총 4,000개의 샘플이 포함되어 있으며, NEARSIDE 방법은 입력의 임베딩이 공격 방향과의 유사성을 평가하여 적대적 샘플의 존재를 탐지합니다. 실험 결과, LLaVA에서는 83.1%, MiniGPT-4에서는 93.5%의 탐지 정확도를 기록했습니다.

- **Performance Highlights**: NEARSIDE는 LLaVA에서 0.14초의 평균 속도로 탐지를 수행하여 가장 기존 방법보다 40배나 빨라 효율성이 뛰어남을 증명했습니다. 또한, 공격 방향의 모델 간 전이 가능성을 검증하여 연구의 유용성을 높였습니다.



### Adaptive Paradigm Synergy: Can a Cross-Paradigm Objective Enhance Long-Tailed Learning? (https://arxiv.org/abs/2410.22883)
Comments:
          11 pages, 3 figures

- **What's New**: 이 논문에서는 Self-supervised learning (SSL)과 supervised learning을 결합한 Adaptive Paradigm Synergy (APS) 접근 방식을 소개합니다. APS는 장기 분포(long-tailed distribution) 데이터를 처리할 수 있는 새로운 방법론을 제안하며, SSL의 성능 향상을 도모합니다.

- **Technical Details**: APS는 contrastive learning을 공간 구조(spatial structure) 관점에서 재조명하고, adaptive temperature tuning을 통해 잠재 공간 구조의 균일성을 동적으로 조정합니다. 또한, supervised learning의 재가중치 전략을 활용하여 온도 조정의 단점을 보완합니다. 이러한 접근은 기존의 supervised long-tailed learning 기법을 SSL에 통합할 수 있는 가능성을 제공합니다.

- **Performance Highlights**: APS는 여러 공개된 장기 분포 데이터셋에서 효과적인 성능 향상을 보여주었으며, 이를 통해 SSL과 supervised 기법 간의 보다 깊은 통합 가능성을 밝혀내고, 실제 데이터 클래스 불균형 문제를 해결하는 Robust한 모델 개발에 기여할 수 있는 길을 열었습니다.



### SFA-UNet: More Attention to Multi-Scale Contrast and Contextual Information in Infrared Small Object Segmentation (https://arxiv.org/abs/2410.22881)
Comments:
          Accepted and Presented at PRIP 2023

- **What's New**: 이 연구는 Infrared Small Object Segmentation (ISOS) 문제를 해결하기 위해 새로운 구조인 SFA-UNet을 제안합니다.

- **Technical Details**: SFA-UNet은 Scharr Convolution (SC) 및 Fast Fourier Convolution (FFC)을 통합하고, 수직 및 수평 Attention gates (AG)를 포함하는 수정된 U-Net 아키텍처로 구성됩니다. 이 모델은 인코더 및 디코더 레이어에 이중 합성곱 레이어를 사용하여 배경 대비 정보와 멀티 스케일 컨텍스트 정보를 효과적으로 학습합니다.

- **Performance Highlights**: 제안된 방법은 SIRST와 IRSTD 데이터셋을 사용하여 기존의 최첨단 방법들에 비해 모든 결합된 메트릭에서 평균 0.75%, 분산 0.025의 성능 향상을 달성하였습니다.



### Prune and Repaint: Content-Aware Image Retargeting for any Ratio (https://arxiv.org/abs/2410.22865)
Comments:
          NeurIPS24

- **What's New**: 이번 연구에서는 이미지 비율 조절(image retargeting)의 문제를 해결하기 위한 새로운 방법인 PruneRepaint를 제안합니다. 이 방법은 픽셀 단위의 의미를 고려하여 이미지의 중요한 정보를 보존하면서 각기 다른 화면 비율에 적합하게끔 이미지를 조정합니다.

- **Technical Details**: PruneRepaint 방법은 컨텐츠 인식 시멘틱 정보(content-aware semantic information)를 기반으로 하여 주요 개념을 유지하고, 지역 프루닝(pruning) 과정을 통해 이미지의 중요한 객체를 보호합니다. 또한, 적응형 재페인팅 모듈(adaptive repainting module)을 사용하여 제거된 픽셀의 분포와 전경 크기(foreground size)와 목표 비율 사이의 비율을 고려하여 재페인팅 지역을 선택합니다.

- **Performance Highlights**: 이 연구는 다양한 화면 비율에 대한 실험을 통해 기존 방법들과 비교하여 주요 개념과 미적 요소를 잘 보존하는 능력이 우수함을 입증했습니다. 객관적 실험 결과와 주관적 사용자 연구에서 모두 높은 성과를 나타냈습니다.



### AtGCN: A Graph Convolutional Network For Ataxic Gait Detection (https://arxiv.org/abs/2410.22862)
- **What's New**: 본 논문에서는 AtGCN이라는 새로운 그래프 컨볼루션 네트워크를 제안하여 2D 비디오를 사용하여 운동실조(atacia) 보행을 감지하고 그 중증도를 평가합니다. 특히 운동실조 보행의 정상 보행과의 미세한 차이를 효과적으로 포착할 수 있도록 설계된 특별한 시공간(spatiotemporal) 그래프 컨볼루션을 활용하고 있습니다.

- **Technical Details**: AtGCN 모델은 사전 학습된 행동 인식(action recognition) 데이터셋 위에서 체계적으로 조정하고 미세 조정하여 작은 데이터셋에서도 학습할 수 있도록 설계되었습니다. 이 시스템은 보행 주기를 여러 개로 분할한 후, 각 주기를 그래프 구조로 변환하여 모델의 입력으로 사용합니다. YOLO 감지기를 통해 비디오에서 인물의 바운딩 박스를 생성하며, DeepSORT를 통해 프레임 간 인물 추적을 수행합니다.

- **Performance Highlights**: 제안된 AtGCN 모델은 운동실조 보행 감지 및 중증도 예측에서 93.46%의 정확도 및 0.4169의 MAE(Mean Absolute Error)를 기록하며, 최신 기술들(state-of-the-art)을 능가하는 성능을 보여줍니다.



### DAVINCI: A Single-Stage Architecture for Constrained CAD Sketch Inferenc (https://arxiv.org/abs/2410.22857)
Comments:
          Accepted at BMVC 2024

- **What's New**: DAVINCI는 래스터 스케치 이미지로부터 단일 단계에서 CAD 스케치 매개변수화(parameterization) 및 제약 조건 추론(inference)을 수행할 수 있는 통합 아키텍처를 제안합니다. 이 방법은 제약 조건을 보존하는 변환인 Constraint-Preserving Transformations (CPTs)을 도입하여, 대규모 주석 데이터셋에 대한 의존성을 줄이고 0.1%의 데이터로도 합리적인 성능을 달성합니다.

- **Technical Details**: DAVINCI는 래스터 스케치 이미지에서 CAD 스케치를 동시에 처리하며, 기존의 방법들이 분리된 문제로 취급하던 것을 단일 네트워크로 예측하여 다단계 처리로 인한 오류 누적을 줄입니다. 이 아키텍처는 대규모 데이터셋인 SketchGraphs에서 최신 성능을 기록했으며, CPT를 사용한 데이터 증강(data augmentation) 기법을 통해 다양한 매개변수를 가진 CAD 스케치를 생성합니다.

- **Performance Highlights**: DAVINCI는 SketchGraphs 데이터셋에서 높은 성능을 달성했으며, 80백만 개의 CPT-증강 스케치를 포함하는 새로운 CPTSketchGraphs 데이터셋을 소개하여 CAD 스케치 도메인에서의 연구에 중요한 자료를 제공합니다.



### SFDFusion: An Efficient Spatial-Frequency Domain Fusion Network for Infrared and Visible Image Fusion (https://arxiv.org/abs/2410.22837)
Comments:
          accept in ECAI 2024

- **What's New**: 본 논문은 효율적인 Infrared(적외선)과 Visible(가시광선) 이미지 융합을 위한 Spatial-Frequency Domain Fusion(SFDFusion) 네트워크를 제안합니다. DMRM(Dual-Modality Refinement Module)을 통해 두 가지 모달리티에서 보완적인 정보를 추출하고, FDFM(Frequency Domain Fusion Module)을 사용하여 주파수 도메인 정보를 통합합니다.

- **Technical Details**: 우리는 Fast Fourier Transform(FFT)를 통해 공간 도메인을 주파수 도메인으로 변환하고, 이를 다시 Inverse Fast Fourier Transform(IFFT)를 통해 공간 도메인으로 되돌립니다. 이 과정에서 주파수 도메인 융합 손실(fusion loss)을 설계하여 융합 과정에 대한 지침을 제공합니다. 이 모델은 공간 도메인과 주파수 도메인으로부터 숨겨진 표현을 동시에 학습할 수 있는 병렬 구조를 채택합니다.

- **Performance Highlights**: 다양한 공공 데이터 세트에서 본 방법은 7개의 최신 기법(SOTA)과 비교했을 때 여러 융합 지표 및 시각 효과에서 우수한 성능을 보여주었습니다. 또한, 높은 효율성과 객체 탐지 정확성을 향상시킬 수 있는 능력을 입증했습니다.



### Situational Scene Graph for Structured Human-centric Situation Understanding (https://arxiv.org/abs/2410.22829)
Comments:
          Accepted for WACV 2025

- **What's New**: 이 논문에서는 Situational Scene Graph (SSG)라는 새로운 그래프 기반 표현 방식을 제안하여 인간-객체 관계와 관련된 의미적 속성을 통합적으로 인코딩합니다. 기존의 방법들이 인간-객체 관계에 중점을 두었지만 세부적인 의미적 속성을 간과한 것에 대한 해결책을 제시합니다.

- **Technical Details**: SSG는 사전 정의된 역할(roles)과 값(values)을 사용하여 상황 장면 내의 여러 구성 요소의 세부 정보를 표현합니다. 제안된 InComNet 모델은 (I) 객체 SRV 분류, (II) 동사 술어 분류, (III) 동사 술어 SRV 분류 및 (IV) 인물 SRV 분류의 네 단계로 나뉘어 있습니다.

- **Performance Highlights**: 실험 결과, SSG 통합 표현 방식은 술어(prediction) 분류 및 의미적 역할-값(semantic role-value) 분류를 개선할 뿐만 아니라, 인간 중심 상황 이해를 위한 추론(reasoning) 작업에서도 성능 혜택을 보여줍니다.



### Epipolar-Free 3D Gaussian Splatting for Generalizable Novel View Synthesis (https://arxiv.org/abs/2410.22817)
Comments:
          NeurIPS 2024

- **What's New**: 본 논문에서는 epipolar 제약 조건에 의존하지 않고 새로운 장면을 효과적으로 재구성할 수 있는 eFreeSplat 모델을 제안합니다. 이 모델은 3D Gaussian Splatting을 기반으로 하며, 기존 방법들의 제한을 극복하여 다양한 관점에서의 시각 정보를 일반화할 수 있습니다.

- **Technical Details**: eFreeSplat은 자가 감독(self-supervised) Vision Transformer (ViT)와 대규모 데이터셋에 대한 크로스 뷰(completion) 사전 훈련을 활용하여 다중 관점(feature) 추출을 향상시킵니다. 또한, Iterative Cross-view Gaussians Alignment (ICGA) 메소드를 도입하여 서로 다른 뷰 간의 깊이 스케일을 일관되게 유지합니다.

- **Performance Highlights**: eFreeSplat은 RealEstate10K 및 ACID 데이터셋을 사용한 실험에서 다른 최첨단 모델들을 초월하여 우수한 기하학적 재구성과 새로운 뷰 합성 품질을 보여줍니다.



### Adaptive Multi Scale Document Binarisation Using Vision Mamba (https://arxiv.org/abs/2410.22811)
- **What's New**: 본 연구에서는 문서 이미지를 이진화(binarisation)하기 위한 Mamba 기반 아키텍처를 제안하였습니다. 이 아키텍처는 긴 시퀀스를 효율적으로 처리하며 메모리 사용을 최적화하는 것이 특징입니다.

- **Technical Details**: AMSDB-VMamba는 U-Net 기반 아키텍처로, Mamba 인코더를 사용하고 다중 상향 샘플링 블록을 포함하여 다양한 깊이에서 의미를 포착하고 이진화된 이미지를 예측합니다. 고주파 특징 추출을 위해 Gaussian Difference(DoG) 기능을 도입했습니다.

- **Performance Highlights**: 제안된 모델은 역사적 문서 이미지 분석에서의 가독성(readability) 향상을 목표로 하며, 높은 품질의 세부 출력을 생성하는 성능을 보여줍니다.



### Wavelet Burst Accumulation for turbulence mitigation (https://arxiv.org/abs/2410.22802)
- **What's New**: 본 논문에서는 최근 제안된 가중치 푸리에 버스트 집합(FBA) 방법을 웨이브렛(wavelet) 도메인으로 확장하는 방법을 조사합니다. 이는 대기 난류를 통해 획득된 세quence의 처리를 가능하게 합니다.

- **Technical Details**: 기존의 알고리즘에서 사용되던 강직(registration) 단계 대신 비강직(non-rigid registration) 단계를 제안하며, 웨이브렛 도메인에서 작업하게 되어 두 가지 유형의 알고리즘을 수립합니다. 또한, 가중치 아이디어를 희소성(sparsity)을 촉진하는 접근 방식을 제안합니다.

- **Performance Highlights**: 여러 실험을 통해 제안된 방법의 효율성을 보여주며, 웨이브렛을 사용할 때 푸리에보다 유리한 점을 강조합니다.



### Open Turbulent Image Set (OTIS) (https://arxiv.org/abs/2410.22791)
- **What's New**: 이 논문에서는 대기 난기류(turbulence)로 인한 영향을 연구하고 이를 평가할 수 있는 전문 데이터셋인 OTIS(Open Turbulent Images Set)를 소개하며, 이는 정적(static) 및 동적(dynamic) 장면의 여러 시퀀스를 포함합니다.

- **Technical Details**: OTIS 데이터셋은 17개의 정적 시퀀스와 4개의 동적 시퀀스로 구성되어 있습니다. 모든 시퀀스에는 알고리즘 성능 평가를 위한 groundtruth 이미지가 제공됩니다. 데이터 수집에는 GoPro Hero 4 Black 카메라가 사용되었으며, MSE(Mean Square Error)와 SSIM(Structural Similarity Index Measure)과 같은 평가 지표를 제안합니다.

- **Performance Highlights**: OTIS 데이터셋은 대기 난기류 완화 알고리즘의 비교를 용이하게 해주며, 연구자들은 다양한 버전의 OTIS를 무료로 다운로드할 수 있습니다. 실제로 SSIM 지표를 통해 대기 난기류 완화 알고리즘의 객관적 성능 평가가 가능하다고 제안합니다.



### Bregman implementation of Meyer's $G-$norm for cartoon + textures decomposition (https://arxiv.org/abs/2410.22777)
- **What's New**: 본 논문에서는 Split Bregman iteration을 기반으로 하는 간단한 알고리즘을 설계하여 Meyer의 cartoon + textures 분해 모델을 수치적으로 해결하는 방법을 제안합니다. 이는 Chambolle의 비선형 projector에 비해 속도에 있어 상당한 이득을 제공합니다.

- **Technical Details**: 자세한 내용은 먼저 Chambolle의 비선형 projector와 cartoon + textures 분해 모델에 대해 설명하고, 이후 Split Bregman iteration을 이용해 G−limit-from𝐺G-italic_G -norm 모델을 해결하는 방법과 그에 따른 cartoon + textures 분해 알고리즘에 대해 다룹니다. 이 과정에서 Bregman iteration을 활용한 최적화 기법과 알고리즘의 수학적 근거를 다룹니다.

- **Performance Highlights**: 새로운 알고리즘을 통해 얻은 결과는 분해 과정에서의 효과성과 성능 향상을 보여줍니다. 실험 결과는 기존 방법들을 능가하는 성능을 입증하였으며, 구체적인 수치 결과와 시각화를 통해 이를 보여줍니다.



### Diffusion Beats Autoregressive: An Evaluation of Compositional Generation in Text-to-Image Models (https://arxiv.org/abs/2410.22775)
- **What's New**: 본 논문에서는 최근에 출시된 텍스트-투-이미지(T2I) 생성 모델인 FLUX와 LlamaGen의 합성 생성(compositional generation) 능력을 평가하며, 기존의 모델인 DALL-E3과 Stable Diffusion(SD)과의 비교를 실시했습니다. 이 결과는 두 모델의 성능 차이를 구체적으로 보여줍니다.

- **Technical Details**: FLUX 모델은 하이브리드 아키텍처를 사용하여 다형성과 평행한(diffusion transformer) 블록을 통합하며, rotary positional embeddings와 parallel attention layers를 특징으로 합니다. LlamaGen은 Llama 아키텍처를 기반으로 다음 토큰 예측(next-token prediction) 패러다임을 사용하여 이미지를 생성하고, 이는 전통적인 autoregressive 모델로 분류됩니다. T2I-CompBench 벤치마크를 통해 다양한 합성 생성 능력이 평가되었습니다.

- **Performance Highlights**: FLUX 모델은 DALL-E3과 비교할 때 유사한 합성 생성 능력을 보이며, LlamaGen은 같은 기준(모델 크기와 추론 시간) 하에서 SD-v1.4보다 합성 생성 과제에서 성능이 낮다는 것을 보였습니다. 결과적으로, FLUX는 최신의 T2I 모델 중 하나로서 높은 품질의 이미지 생성에서 강력한 성능을 보여줍니다.



### FuseAnyPart: Diffusion-Driven Facial Parts Swapping via Multiple Reference Images (https://arxiv.org/abs/2410.22771)
Comments:
          Accepted by the NeurIPS 2024 (Spotlight). Homepage: this https URL

- **What's New**: FuseAnyPart는 개별적인 얼굴 부위를 원활하게 교환할 수 있게 설계된 최초의 diffusion 기반 방법입니다. 이 기술은 다양한 사람들의 얼굴 부위를 결합하여 하나의 새로운 얼굴을 생성하는 혁신적인 접근법을 제공합니다.

- **Technical Details**: FuseAnyPart의 Mask-based Fusion Module에서 다양한 얼굴 부위의 특징을 조합하여 잠재 공간(latent space) 내에서 완전한 얼굴을 생성합니다. 이후, Addition-based Injection Module을 통해 UNet에 통합되어 새로운 캐릭터를 생성하는 데 사용됩니다.

- **Performance Highlights**: FuseAnyPart는 광범위한 실험에 의해 우수성 및 강건성이 검증되었습니다. 제안된 모듈들이 기존의 cross-attention 메커니즘에 비해 더욱 효과적이고 효율적인 결과를 보여주며, 세밀하고 맞춤형 얼굴 디자인을 가능하게 합니다.



### Analysis of Classifier Training on Synthetic Data for Cross-Domain Datasets (https://arxiv.org/abs/2410.22748)
Comments:
          10 pages

- **What's New**: 이 연구는 인공지능 운전 보조 시스템(ADAS)에서 카메라 기반의 교통 표지 인식 애플리케이션을 위해 합성 데이터(synthetic data)를 사용한 훈련의 잠재력을 탐구했습니다. 합성 이미지를 통한 훈련이 실제 이미지 기반 훈련보다 성능적으로 우수하며, 이미지 획득 비용을 절감할 수 있음을 보입니다.

- **Technical Details**: 합성 데이터의 증강 파이프라인은 구조화된 그림자(structured shadows) 및 가우시안 반사(gussian specular highlights)와 같은 새로운 증강 과정을 포함하고 있습니다. 잘 알려진 딥 러닝 모델을 훈련하여 합성 데이터와 실제 데이터 기반 훈련 모델 간의 성능을 비교하였습니다. 합성 이미지는 반지도 학습(semi-supervised) 오류 안내 방법을 사용하여 생성됩니다.

- **Performance Highlights**: 실험 결과, 합성 이미지 기반 접근 방식이 교차 도메인 테스트 데이터 세트에서 대부분의 경우 실제 이미지 기반 훈련을 초월하는 것으로 나타났으며, GTSRB 데이터 세트에서 +10%의 정확도를 기록했습니다. 이로 인해 모델의 일반화가 개선되어 이미지 획득 비용이 줄어들었습니다.



### ETO:Efficient Transformer-based Local Feature Matching by Organizing Multiple Homography Hypotheses (https://arxiv.org/abs/2410.22733)
- **What's New**: 이 논문에서는 Transformer 기반의 지역 특징 매칭(Local Feature Matching)에서의 효율성 문제를 다룹니다. 우리는 새로운 아키텍처 ETO를 제안하여, 여러 개의 호모그래피 가설(Homography Hypotheses)을 구성하는 방식으로 실제 세계의 연속적 대응을 근사하고, 단방향 크로스 어텐션(uni-directional cross-attention)을 통해 정제를 가속화합니다.

- **Technical Details**: ETO는 두 단계의 대칭 매칭(coarse-to-fine matching) 파이프라인을 따릅니다. 첫 번째 단계에서는 패치 수준에서 가설 집합을 예측하고, 두 번째 단계에서는 이러한 매치를 서브픽셀(sub-pixel) 수준으로 정제합니다. ETO는 호모그래피 변환을 기반으로 패치 매칭을 기술하며, 처리할 토큰 수(number of tokens)를 줄여 효율성을 증가시킵니다. 이 과정에서 단방향 크로스 어텐션을 활용하여 계산 복잡성(computation complexity)을 대폭 감소시킵니다.

- **Performance Highlights**: YFCC100M 데이터셋에서, ETO는 LoFTR보다 4배 빠르며 매우 경쟁력 있는 정확도를 유지하고 있습니다. Megadepth, ScanNet, HPatches와 같은 다양한 공개 데이터셋에서도 성능이 매우 뛰어나며, 알고리즘의 전반적인 적용 가능성을 나타냅니다.



### One Prompt to Verify Your Models: Black-Box Text-to-Image Models Verification via Non-Transferable Adversarial Attacks (https://arxiv.org/abs/2410.22725)
- **What's New**: 최근 Text-to-Image (T2I) 모델의 성공으로 인해 다양한 제3자 플랫폼이 등장하였습니다. 이 플랫폼들은 더 저렴한 API 서비스를 제공하고 모델 옵션의 유연성을 주장합니다. 하지만, 이러한 플랫폼이 실제로 주장하는 모델을 제공하고 있는지는 불확실한 보안 문제로 이어졌습니다. 이를 해결하기 위해 "비전이전이 불가능한 적대적 공격을 통한 T2I 모델 검증 (TVN)"라는 최초의 검증 방법이 제안되었습니다.

- **Technical Details**: TVN은 비전이전이 불가능한 적대적 예제를 이용하여 검증을 수행합니다. 비전이전이 불가능하다는 것은 이러한 예제가 특정 목표 모델에만 효과적이며 다른 모델에 대해서는 효과가 없다는 것을 의미합니다. TVN은 Non-dominated Sorting Genetic Algorithm II (NSGA-II)를 활용하여 텍스트 인코딩의 코사인 유사성을 최적화하고, 이로써 비전이전이 불가능한 적대적 프롬프트를 생성합니다. 생성된 이미지를 이용해 CLIP-text 점수를 계산하고, 3-sigma 임계치를 설정하여 모델의 일치를 검증합니다.

- **Performance Highlights**: TVN은 닫힌 집합(closed-set) 및 열린 집합(open-set) 시나리오 모두에서 뛰어난 성능을 보여주며, 모델 검증 정확도가 90%를 초과하였습니다. 또한, TVN이 생성한 적대적 프롬프트는 목표 모델의 CLIP-text 점수를 현저히 감소시키고 다른 모델에는 거의 영향이 없음을 보여주었습니다.



### SCRREAM : SCan, Register, REnder And Map:A Framework for Annotating Accurate and Dense 3D Indoor Scenes with a Benchmark (https://arxiv.org/abs/2410.22715)
- **What's New**: 이 논문에서는 SCRREAM이라는 새로운 데이터셋 주석 프레임워크를 제안합니다. 이 프레임워크는 장면 내 개체의 완전한 밀집 메쉬(annotation of fully dense meshes)를 주석화하고 실제 이미지 시퀀스에 카메라 포즈를 등록하여, 희소 3D와 밀집 3D 작업 모두에 대해 정확한 기준을 생성할 수 있도록 설계되었습니다.

- **Technical Details**: SCRREAM은 4단계로 구성된 파이프라인을 통해 작동합니다: SCanning, Registering, REndering, Mapping. 반복적으로 생성된 메쉬를 등록하고 다양한 장면을 렌더링함으로써, 내부 장면 재구축 및 SLAM과 같은 작업에 활용할 수 있습니다. 예를 들어, 전체 장면에서 고품질의 메쉬와 실제 물체를 스캔하여 사용됩니다.

- **Performance Highlights**: 기존의 실내 데이터셋에 비해 SCRREAM은 11개의 샘플 장면에서 정확하게 렌더링된 기준 깊이 맵을 기준으로 밀집 기하학 작업을 평가할 수 있는 새로운 벤치마크를 제공합니다. 이를 통해 밀도 있는 메쉬를 활용한 고품질 3D 재구성 및 다양한 장면의 수정 가능성을 개선할 수 있습니다.



### LoFLAT: Local Feature Matching using Focused Linear Attention Transformer (https://arxiv.org/abs/2410.22710)
- **What's New**: 본 논문에서는 Local Feature matching의 새로운 접근 방식을 제안합니다. LoFLAT(Local Feature matching using Focused Linear Attention Transformer)은 기존 Transformer 기반 지역 특징 매칭 방법의 한계를 극복하고, 세부적인 지역 상호작용을 포착하는 데 중점을 두고 있습니다. 이 방법은 Feature Extraction, Feature Transformer 그리고 Matching 모듈로 구성됩니다.

- **Technical Details**: LoFLAT는 ResNet과 Feature Pyramid Network를 사용하여 계층적인 특징을 추출합니다. 그 후, Focused Linear Attention을 이용해 주목 분포를 정제하고, depth-wise convolution을 통해 특징의 다양성을 향상시킵니다. 마지막으로, Matching 모듈은 coarse-to-fine 전략을 통해 정밀하고 견고한 매칭을 예측합니다.

- **Performance Highlights**: 실험 결과, LoFLAT는 MegaDepth 데이터셋에서 LoFTR에 비해 효율성과 정확성 모두에서 우수한 성능을 보였습니다.



### FilterViT and DropoutViT: Lightweight Vision Transformer Models for Efficient Attention Mechanisms (https://arxiv.org/abs/2410.22709)
- **What's New**: 이번 연구에서는 MobileViT의 개선된 버전인 FilterViT를 소개합니다. FilterViT는 초기 단계의 downsampling(다운샘플링)에서 attention-based mechanism(어텐션 기반 메커니즘)을 활용합니다. 기존의 QKV operations는 계산 집약적이어서, 우리는 CNN을 이용한 필터 어텐션 메커니즘을 제안하여 중요 마스크를 생성하고, 주목할 영역에 집중합니다.

- **Technical Details**: 이 방법은 CNN을 통해 생성된 중요 마스크를 이용하여 Feature Map 내에서 중요한 픽셀을 선택합니다. Filter Attention(필터 어텐션) 메커니즘을 통해 선별적으로 주목하여 QKV operation(쿼리-키-값 연산)의 계산 복잡성을 줄입니다. 중요도가 높은 상위 K개 픽셀만 선택하여 Attention을 수행하며, 이는 GPU에서의 Sparse Matrix Operations(희소 행렬 연산)을 간편하게 해결합니다.

- **Performance Highlights**: 실험 결과 FilterViT는 다른 모델들에 비해 효율성과 정확도 모두에서 상당한 성능 향상을 보여주었습니다. 또한, DropoutViT를 도입하여 픽셀 선택에 Stochastic(확률적) 방식을 적용해 모델의 견고함을 더욱 향상시켰습니다.



### Geometry Cloak: Preventing TGS-based 3D Reconstruction from Copyrighted Images (https://arxiv.org/abs/2410.22705)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 이 논문에서는 Triplane Gaussian Splatting (TGS) 프로세스를 악용하여 저작권이 있는 이미지를 무단으로 3D 모델로 생성하는 문제를 다루고 있습니다. 저작권 보호를 위한 새로운 접근 방식으로 'geometry cloak'라는 개념을 도입하여, 이미지에 눈에 띄지 않게 기하학적 왜곡을 삽입하여 TGS의 3D 재구성 과정에서 특정 패턴을 드러내는 방법을 제안합니다.

- **Technical Details**: 제안된 geometry cloak는 TGS가 3D 모델 생성을 실패하게 하여 저작권이 있는 이미지에 대해 식별 가능한 패턴을 노출하도록 유도합니다. 이 과정에서 view-specific Projected Gradient Descent (PGD) 전략을 사용하여, 정의된 패턴과의 거리 최적화를 통해 숨겨진 패턴을 밝혀낼 수 있습니다. 이를 통해 악의적인 사용자들이 TGS를 사용하여 3D 모델을 재구성할 때, 해당 모델은 사용할 수 없는 결과를 생성하게 됩니다.

- **Performance Highlights**: 실험을 통해 geometry cloak의 효과가 입증되었습니다. 저작권 보호 수단으로써 기존의 디지털 워터마킹 방식과 달리, TGS를 통해 감지된 패턴으로 저작권 주장 및 소유권을 명확히 할 수 있는 어떻게 작용합니다. 이는 Triplane 기반의 Gaussian Splatting뿐만 아니라 기타 GS 기반의 3D 생성 접근 방식에서도 일반화될 수 있는 가능성을 보여주었습니다.



### Persistent Homology for MCI Classification: A Comparative Analysis between Graph and Vietoris-Rips Filtrations (https://arxiv.org/abs/2410.22681)
Comments:
          17 pages, 5 figures, 4 tables

- **What's New**: 본 연구는 조기 신경퇴행에 연관된 경도 인지 장애(MCI)의 두 가지 하위 유형인 조기 MCI와 후기 MCI와 관련된 위상적 변화를 상세하게 분석합니다.

- **Technical Details**: Persistent Homology는 위상 데이터 분석(topological data analysis) 방법으로, 이 연구에서는 Vietoris-Rips 및 그래프 필터링(graph filtration)이라는 두 가지 필터링 기법을 사용하여 MCI 하위 유형을 분류합니다. Vietoris-Rips 필터링은 지속적인 도표(persistent diagrams) 간의 Wasserstein 거리 행렬을 사용하여 분류하며, 그래프 필터링은 가장 지속적인 동형 특징의 상위 10개를 기반으로 합니다.

- **Performance Highlights**: Vietoris-Rips 필터링 방법은 85.7%의 분류 정확도를 달성하여 건강한 대조군과 MCI를 효과적으로 구분하였으며, 그래프 필터링은 최대 71.4%의 정확도에 그쳤습니다. 이 연구의 결과는 neurodegeneration과 관련된 복잡한 위상적 특징을 감지하는 데 있어 Vietoris-Rips 필터링의 우수성을 강조합니다.



### Practical and Accurate Reconstruction of an Illuminant's Spectral Power Distribution for Inverse Rendering Pipelines (https://arxiv.org/abs/2410.22679)
Comments:
          3 pages, 3 Figures, Submitted as a Tiny Paper at ICVGIP'24, Bangalore, India

- **What's New**: 본 논문에서는 가상 현실 장면에서 물체를 포토리얼리스틱하게 재구성하기 위한 역 렌더링(inverse rendering) 파이프라인을 제안합니다. 기존의 비싼 분광계(spectrometer) 대신 저비용의 회절 컴팩트 디스크(CD-ROM)와 머신 러닝(machin learning) 방법을 사용하여 균일 조명의 스펙트럼 전력 분포(SPD)를 캡처하고 재구성하는 방법을 소개합니다.

- **Technical Details**: 제안된 방법은 기존의 스펙트로미터 및 하이퍼스펙트럴 이미징(hyper-spectral imaging) 기법 대신, 회절 광학 소자(diffractive optical element)를 활용한 이미지 기반 스펙트럼 프로파일링을 이용합니다. 우리는 개념 및 시뮬레이션 환경에서 조사한 결과, MLP(다층 퍼셉트론: multilayer perceptron) 네트워크를 통해 5000개의 SPDs를 사용하여 학습하고, 평균 절대 오차(MAE) 및 RMSE(Root Mean Square Error)와 같은 통계적 측정을 통해 성과를 평가했습니다. 이 과정이 100,000 에폭(epoch) 훈련을 필요로 했습니다.

- **Performance Highlights**: 제안된 방법은 고급 SSR(뛰어난 스펙트럼 렌더링) 기술과 함께 실험되었으며, 결과의 정확성은 시각적으로 검증되었습니다. 우리는 MLP 네트워크의 성능이 매우 높은 정확도로 SPDs를 예측함을 확인했으며, 랜더링 결과가 실제 데이터와 거의 구별되지 않는다는 것을 보여주었습니다. 향후 실제 조명의 SPDs에 대한 검증을 통해 더욱 발전시킬 계획입니다.



### Backdoor Attack Against Vision Transformers via Attention Gradient-Based Image Erosion (https://arxiv.org/abs/2410.22678)
Comments:
          Accepted by IEEE GLOBECOM 2024

- **What's New**: 이번 연구에서는 Vision Transformers (ViTs)를 겨냥한 Attention Gradient 기반 Erosion Backdoor(AGEB)를 제안합니다. AGEB는 ViTs의 주의(attention) 메커니즘을 활용하여 픽셀을 선택적으로 침식시켜 백도어 트리거를 내장합니다. 기존 방법들에 비해 AGEB는 공격 stealthiness(은폐성)와 효과성의 최적 균형을 달성합니다.

- **Technical Details**: AGEB는 ViTs의 주의 그래디언트를 고려하여 최대 주의 그래디언트를 가진 영역에서 픽셀을 선택적으로 침식합니다. 이 과정은 이미지 변경을 목표로 하며, 이를 통해 트리거를 은밀하게 생성합니다. AGEB는 기존의 지역적 패치 기반 트리거의 한계를 극복하기 위해 전역적 트리거 메커니즘을 채택해 일반화 능력을 향상시킵니다.

- **Performance Highlights**: 다양한 ViT 아키텍처와 데이터셋에서 광범위한 실험 평가가 이루어졌으며, AGEB는 높은 공격 성공률(Attack Success Rate, ASR)과 함께 청정 데이터 정확도(Clean Data Accuracy, CDA)를 유지하는 뛰어난 효과를 보여주었습니다. 특히 ImageNette 데이터셋에서는 테스트 이미지의 97.02%가 특정 타겟 클래스에 정확히 분류되었습니다.



### FlowDCN: Exploring DCN-like Architectures for Fast Image Generation with Arbitrary Resolution (https://arxiv.org/abs/2410.22655)
Comments:
          Accepted on NeurIPS24

- **What's New**: 이 논문은 FlowDCN을 제안하며, 이는 고해상도 이미지를 효율적으로 생성할 수 있는 새로운 컨볼루션 기반 생성 모델입니다. 기존의 Transformer 기반 확산 모델의 한계를 극복하고 linear한 시간 및 메모리 복잡성으로 작업합니다.

- **Technical Details**: FlowDCN은 learnable group-wise deformable convolution block을 탑재하여 다양한 해상도를 단일 모델로 처리할 수 있는 유연성을 제공합니다. 이 모델은 256x256 ImageNet 기준에서 4.30 sFID를 달성하며, 기존 모델 대비 더 빠른 수렴 속도와 높은 시각적 품질을 나타냅니다. 또한, 파라미터 수가 8% 감소하고 FLOPs가 20% 감소하였습니다.

- **Performance Highlights**: FlowDCN은 256x256 이미지에서 1.5M 학습 단계에 걸쳐 4.30 sFID와 2.13 FID를 기록했습니다. 512x512 이미지 기준에서도 좋은 성능을 보여주었으며, 특히 빠른 임의 해상도 생성이 가능하여 더 나은 시각적 품질을 보였습니다.



### SimpsonsVQA: Enhancing Inquiry-Based Learning with a Tailored Datas (https://arxiv.org/abs/2410.22648)
- **What's New**: 이번 논문에서는 'SimpsonsVQA'라는 새로운 VQA(Visual Question Answering) 데이터셋을 소개합니다. 이는 The Simpsons TV 쇼에서 유래된 데이터로, 전통적인 VQA 작업뿐만 아니라 이미지와 관련된 무관한 질문 및 제공된 답변을 평가하는 새로운 작업을 포함합니다.

- **Technical Details**: SimpsonsVQA는 약 23,000개의 이미지, 166,000쌍의 Q&A(질문과 답변) 및 약 500,000개의 판단(judgments)을 포함합니다. 이 데이터셋은 세 가지 시나리오(유관한 질문, 무관한 질문, 그리고 제공된 답변 평가)에 초점을 맞추고 있으며, 교육적 맥락에서 인지 장애인을 위한 시스템 개발에 기여할 수 있습니다.

- **Performance Highlights**: 현재의 대규모 비전-언어 모델인 ChatGPT4o는 제로 샷(zero-shot) 설정에서 모든 세 가지 작업에서 성능이 떨어지는 것으로 나타났습니다. 이는 SimpsonsVQA의 가치가 모델 성능 개선에 기여할 수 있음을 강조합니다.



### Unbiased Regression Loss for DETRs (https://arxiv.org/abs/2410.22638)
- **What's New**: 이번 논문에서는 DETR 기반의 객체 검출기를 위한 새로운 비편향(비편향적, unbiased) 회귀 손실 함수인 Sized $L_{1}$ 손실을 소개합니다. 전통적인 $L_{1}$ 회귀 손실이 더 큰 박스에 편향되는 경향이 있으며, 이로 인해 작은 객체에 대한 검출 성능이 떨어진다는 점에 주목하였습니다.

- **Technical Details**: 제안된 Sized $L_{1}$ 손실은 각 박스의 길이와 높이를 바탕으로 모든 박스의 크기를 정규화하여 손실 기여도를 동일하게 합니다. 이 손실은 두 가지 유형의 DETR 기반 객체 검출기에 적용됩니다: (1) 완전 지도 학습(fully-supervised), (2) 반지도 학습(semi-supervised). 실험 결과, 두 시나리오에서 모두 성능이 향상됨을 보여줍니다.

- **Performance Highlights**: MS-COCO 벤치마크 데이터셋을 사용한 실험 결과, 제안된 Sized $L_{1}$ 손실을 사용하는 경우 특히 작은 객체에 대한 성능이 개선되었음을 보여줍니다.



### CrossEarth: Geospatial Vision Foundation Model for Domain Generalizable Remote Sensing Semantic Segmentation (https://arxiv.org/abs/2410.22629)
Comments:
          The codes and models will be available at this https URL

- **What's New**: 본 논문에서는 Remote Sensing Domain Generalization (RSDG)을 위한 최초의 비전 기초 모델인 CrossEarth를 소개합니다. CrossEarth는 지구 스타일 데이터 주입(Earth-Style Injection) 파이프라인과 다중 작업 학습(Multi-Task Training) 파이프라인을 활용하여 크로스 도메인 일반화를 달성합니다.

- **Technical Details**: CrossEarth는 데이터 조작(Data Manipulation) 및 표현 학습(Representation Learning)을 통해 생성되며, 두 가지 상호 보완적인 파이프라인으로 구성됩니다: 1) Earth-Style Injection: 소스 도메인 데이터를 확장하여 훈련 도메인 분포를 넓히는 방식, 2) Multi-Task Training: 공동 DINO-V2 백본을 사용하여 의미적 분할과 Masked Image Modeling (MIM)을 동시에 수행하여 견고한 의미적 특징을 학습.

- **Performance Highlights**: CrossEarth는 28개의 크로스 도메인 설정을 포함하는 RSDG 벤치마크에서 광범위한 실험을 통해 현재까지의 최고 성능(state-of-the-art, SOTA)을 보여주며 기존 모델들과의 비교에서 뛰어난 일반화 능력을 입증하였습니다.



### Symbolic Graph Inference for Compound Scene Understanding (https://arxiv.org/abs/2410.22626)
- **What's New**: 이번 연구는 장면 이해(understanding)에서 새로운 접근 방식을 제안합니다. 기존의 end-to-end 방법이 장면을 개별적인 개체로 해석하는 반면, 제안하는 방법은 구성 요소 객체(object)와 그들의 배열(arrangement)을 분석하여 장면의 의미를 추론합니다.

- **Technical Details**: 제안하는 방법은 장면 그래프(scene graph)와 지식 그래프(knowledge graph)를 결합하여 공간(spatial) 정보를 캡처하며, 일반 도메인 지식을 공동 그래프 탐색(joint graph search)을 통해 활용합니다. 이 방법은 동적 그래프 전파(dynamic graph propagation) 기법을 사용하여 실행 시간(runtime)과 확장성(scalability)을 개선합니다. 전체 프로세스는 장면 및 지식 그래프 생성을 포함하고, 두 그래프를 병합(merging)한 후, 이 통합된 그래프에서 탐색(search)하여 예측을 수행합니다.

- **Performance Highlights**: ADE20K 데이터셋을 기반으로 실험을 수행하였으며, 우리의 방법은 기존의 기호(symbolic) 기반 및 신경망(neural-only) 접근 방법과 비교하여 장면 이해의 가능성을 보여주었습니다.



### PV-VTT: A Privacy-Centric Dataset for Mission-Specific Anomaly Detection and Natural Language Interpretation (https://arxiv.org/abs/2410.22623)
Comments:
          Accepted to WACV 2025

- **What's New**: PV-VTT (Privacy Violation Video To Text) 데이터셋을 소개하며, 이는 개인의 프라이버시 침해를 탐지하는 데 중점을 두고 있습니다. 비디오의 전체적인 분석보다는 범죄 예방을 위한 사전 신호를 포착하는 데 초점을 맞춘 데이터셋입니다.

- **Technical Details**: PV-VTT는 비디오와 텍스트에 대한 상세한 주석을 제공하며, 개인의 프라이버시를 보장하기 위해 비디오 원본 데이터는 공개하지 않고 비디오 기능 벡터만 제공하는 방식으로 설계되었습니다. 비디오 설명을 생성하기 위해 Graph Neural Network (GNN)를 활용하여 Large Language Model (LLM)과 통합됩니다.

- **Performance Highlights**: PV-VTT 데이터셋을 통해 여러 최신 모델을 평가하였으며, 실험 결과는 데이터셋의 중요성과 도전 과제를 보여줍니다. 제안된 GNN 기반의 프롬프트는 LLM API 사용 최적화를 통해 비용 효율적인 비디오 설명 생성을 지원합니다.



### GRADE: Quantifying Sample Diversity in Text-to-Image Models (https://arxiv.org/abs/2410.22592)
Comments:
          For project page and code see this https URL

- **What's New**: 이 논문은 텍스트에서 이미지 (Text-to-image, T2I) 모델의 출력 다양성을 평가하기 위해 GRADE(Granular Attribute Diversity Evaluation)라는 자동화된 방법을 제안합니다.

- **Technical Details**: GRADE는 대형 언어 모델과 시각적 질문-답변 시스템에 내재된 세계 지식을 활용하여 관련 개념별 다양성 축(axes)을 식별하고, 개념 및 속성의 빈도 분포를 추정한 후, 엔트로피(entropy)를 사용해 다양성을 정량화합니다.

- **Performance Highlights**: GRADE는 90% 이상의 인간 합의(consensus)를 달성했으며, 12개의 T2I 모델을 평가한 결과 모든 모델에서 제한된 변동성을 나타냈습니다. 특히, 모델이 동일한 속성을 가진 개념을 일관되게 생성하는 '기본 행동(default behaviors)'을 보이는 경향이 있음을 발견했습니다.



### Pre-Trained Vision Models as Perception Backbones for Safety Filters in Autonomous Driving (https://arxiv.org/abs/2410.22585)
- **What's New**: 이 논문에서는 비전 기반 자율 주행의 안전 문제 해결을 위한 새로운 접근법을 제시합니다. 특히, 프리트레인(pre-trained)된 비전 모델을 안전 필터에 활용하여 고차원 환경에서도 안전한 제어를 가능하게 합니다.

- **Technical Details**: 저자들은 DeepAccident 데이터셋을 사용하여 차량의 행동을 주석 달아 제공된 멀티 카메라 비디오를 기반으로 다양한 훈련 기법을 평가합니다. 이 연구에서는 제어 장치의 동적 모델이 알려지지 않은 블랙박스(dynamic control systems) 설정에서 안전 필터를 훈련시키는 세 가지 방법을 시도합니다.

- **Performance Highlights**: 결과적으로, 프리트레인 PVR(Pre-trained Vision Representations)을 활용한 안전 필터는 고전적인 필터와 경쟁력을 갖추며, 때때로 지상 진실 상태(ground truth state)에 대한 접근이 있는 필터보다 나은 성능을 보입니다. 또한, 다중 카메라 피드를 활용한 경우 각 카메라 피드를 별도로 사용할 때보다 하나의 상태로 융합하는 것이 더 효과적임을 보여줍니다.



### Remote Sensing for Weed Detection and Contro (https://arxiv.org/abs/2410.22554)
Comments:
          5 pages, 3 figures

- **What's New**: 이 연구는 드론과 위성 이미지를 활용하여 이탈리안 호밀잡초의 정밀한 제어 및 관리 방법을 제안합니다. 드론 이미지는 충분한 해상도를 가지지만 호밀잡초와 크롭을 구별하기 어렵기 때문에, 600개 이상의 신경망 아키텍처를 테스트하여 최적의 분할 모델을 개발하였습니다.

- **Technical Details**: 이 연구에서 사용된 데이터 세트는 약 1.3×10^9 개의 주석이 달린 픽셀로 구성되어 있으며, 드론 이미지는 2.9cm 해상도를 가지고 있습니다. 또한, 모델 성능은 Pearson 상관 계수(R²)를 통해 평가되며, 최적 모델 조합은 NuSVR, SVR 및 ExtraTreesRegressor를 포함합니다.

- **Performance Highlights**: 최고의 모델은 99%의 잡초에 herbicide를 적용하면서 주석된 잡초 면적보다 30% 더 넓은 면적에 분사했습니다. 따라서 잡초가 들판의 작은 부분만을 차지할 경우, 상당한 비용 절감 효과를 나타냅니다.



### FairSkin: Fair Diffusion for Skin Disease Image Generation (https://arxiv.org/abs/2410.22551)
- **What's New**: 이 연구에서는 의료 영상 생성에서의 인종 편향 문제를 다루기 위해 FairSkin이라는 새로운 Diffusion Model (DM) 프레임워크를 제안합니다. 이 프레임워크는 세 가지 수준의 리샘플링 메커니즘을 통해 피부 질환 이미지의 공정한 표현을 보장합니다.

- **Technical Details**: FairSkin 프레임워크는 (1) 균형 잡힌 샘플링, (2) 클래스 다양성 손실(class diversity loss), (3) 불균형 인식 증강(imbalance-aware augmentation) 및 동적 가중치 조정(dynamic reweighting) 기법을 활용하여 교육 데이터의 과소 표현된 집단을 공평하게 대표합니다.

- **Performance Highlights**: FairSkin은 생성된 이미지의 다양성과 질을 크게 개선하여 피부 질환 검출의 공정성을 높입니다. 실험을 통해 FairSkin이 기존 Diffusion Model에 비해 진단 유용성이 향상되었음을 입증하였습니다.



### AffectNet+: A Database for Enhancing Facial Expression Recognition with Soft-Labels (https://arxiv.org/abs/2410.22506)
- **What's New**: 이 논문에서는 기존의 자동 얼굴 표정 인식(FER) 데이터셋에 대해 여러 감정을 동시에 라벨링할 수 있는 새로운 방법인 '소프트 라벨링(soft-labeling)' 기법을 제안합니다. 이는 감정 표현을 보다 현실적으로 인식하는 데 기여할 수 있습니다.

- **Technical Details**: 기존의 FER 데이터셋은 단일 감정 카테고리(하드 라벨)가 할당되는 반면, 새로운 데이터셋인 AffectNet+는 각 이미지에 대해 여러 감정을 나타내는 소프트 라벨을 사용하여 더 많은 정보와 맥락을 제공합니다. 또한, 이 데이터셋은 이미지의 인식 난이도에 따라 세 가지 하위 카테고리(쉬운(Easy), 도전적인(Challenging), 어려운(Difficult))로 분류됩니다.

- **Performance Highlights**: 제안된 소프트 라벨 기법은 얼굴 표정의 다면성을 반영할 수 있으며, 다양한 감정 인식의 결정 경계를 부드럽게 만들어 주고, 데이터의 불균형과 편향 문제를 완화하는 데 장점을 제공합니다.



### The PV-ALE Dataset: Enhancing Apple Leaf Disease Classification Through Transfer Learning with Convolutional Neural Networks (https://arxiv.org/abs/2410.22490)
Comments:
          To appear in th Sixth International Conference on Soft Computing and its Engineering Applications (icSoftComp2024)

- **What's New**: 이 논문에서는 기존의 PlantVillage 데이터셋을 확장하여 사과 잎의 질병 분류에서 더 많은 다양성과 복잡성을 제공합니다. 새로운 사과 잎 질병 클래스가 추가되어 모델의 일반화 능력이 요구되고, 이를 통해 정확하고 신뢰할 수 있는 질병 진단 모델 개발을 위한 새로운 벤치마크를 제시합니다.

- **Technical Details**: 본 연구에서는 5가지 독특한 사과 잎 질병 유형과 1가지 건강한 클래스에 대한 종합 데이터셋을 구축하였습니다. 각 이미지에는 해당 질병 레이블이 주어졌으며, 멀티 클래스 분류를 위한 CNN 아키텍처가 설계되었습니다. 모델의 성능 평가는 클래스 불균형 상황을 포함한 다양한 시나리오에서 엄격하게 수행되었습니다.

- **Performance Highlights**: 원본 데이터셋에서 99.63%, 확장된 데이터셋에서는 97.87%의 테스트 F1 스코어를 기록했습니다. 이 연구는 사과 잎 질병 진단 모델의 정확성을 높이고자 하는 연구자들에게 중요한 기초 자료를 제공합니다.



### Multimodality Helps Few-Shot 3D Point Cloud Semantic Segmentation (https://arxiv.org/abs/2410.22489)
- **What's New**: 이번 연구에서는 기존의 FS-PCS 방법들이 단일 모드(point cloud) 입력에 집중하고 있는 기존의 한계를 극복하고, 새로운 multimodal FS-PCS 설정을 제안하고 있습니다. 이 방식은 텍스트 필드와 2D 이미지 모달리티를 활용하여 비용이 들지 않는 솔루션을 제공합니다.

- **Technical Details**: MultiModal Few-Shot SegNet (MM-FSS) 모델은 두 개의 헤드를 가진 공유 backbone을 사용하여 intermodal 및 unimodal 시각적 특징을 추출하고, 텍스트 인코더를 통해 텍스트 임베딩을 생성합니다. Multimodal Correlation Fusion (MCF) 모듈과 Multimodal Semantic Fusion (MSF) 모듈을 통해 서로 다른 모달리티의 정보를 효과적으로 융합하고, Test-time Adaptive Cross-modal Calibration (TACC) 기법으로 훈련 편향을 완화합니다.

- **Performance Highlights**: S3DIS 및 ScanNet 데이터세트에서의 실험 결과, 제안된 MM-FSS 방법이 기존의 방법들에 비해 유의미한 성능 향상을 보여주었습니다. 이는 무료로 사용할 수 있는 모달리티를 활용하는 것의 이점을 입증하며, 향후 연구에 많은 통찰력을 제공합니다.



### Unified Domain Generalization and Adaptation for Multi-View 3D Object Detection (https://arxiv.org/abs/2410.22461)
Comments:
          Accepted to NeurIPS 2024

- **What's New**: 이 논문에서는 Unified Domain Generalization and Adaptation (UDGA)라는 새로운 방법을 제안하여 기존의 3D 객체 탐지 모델이 직면하는 도메인 간 미세 조정 문제를 해결하고자 한다. UDGA는 다중 시점에서 발생하는 기하학적 왜곡을 줄이고, 소량의 레이블(1%와 5%)로도 모델이 효과적으로 적응할 수 있도록 지원한다.

- **Technical Details**: UDGA는 Multi-view Overlap Depth Constraint를 활용하여 서로 다른 시점 간의 기하학적 간극을 줄인다. 이 방법은 복수의 카메라를 통해 수집된 다중 시점 데이터를 기반으로 하여, 기하학적 비일치를 효과적으로 완화시킨다. 또한, 본 연구에서는 three landmark datasets인 Waymo, nuScenes, Lyft를 사용하여 모델의 성능을 평가하였다. 마지막으로, BEVDepth 및 BEVFormer와 같은 기본 탐지기를 활용하여 실험을 진행하였다.

- **Performance Highlights**: UDGA는 nuScenes, Lyft, Waymo의 대규모 벤치마크에서 현재의 최고 성능 방법들을 초과하는 성능을 보였다. 특히, UDGA는 소스 도메인에서의 성능을 저해하지 않고 타겟 도메인에서의 정확도를 향상시킬 수 있으며, 이전에 학습한 잠재 정보를 잊지 않고, 실제 데이터 분할을 효율적으로 한 경우(예: 1%, 5%)에도 높은 NDS 향상치를 기록하였다.



### Image2Struct: Benchmarking Structure Extraction for Vision-Language Models (https://arxiv.org/abs/2410.22456)
Comments:
          NeurIPS 2024. First three authors contributed equally

- **What's New**: 이번 연구에서는 이미지로부터 구조를 추출하는 Vision-Language Models (VLMs)의 성능을 평가하기 위해 Image2Struct라는 새로운 벤치마크를 도입합니다. 이 벤치마크는 실제 사용 사례를 포착하며, 완전 자동화되어 사람의 판단이 필요 없고, 신선한 데이터의 흐름을 기반으로 합니다.

- **Technical Details**: Image2Struct는 입력 이미지를 기반으로 LaTeX 코드나 HTML과 같은 구조를 생성하게끔 VLM을 유도합니다. 시스템은 이미지를 다양하게 변환 가능한 알고리즘을 기반으로 하며, 3단계 프로세스를 통해 목적인 이미지를 비교하여 유사성 점수를 생성합니다. 이 벤치마크는 웹페이지, LaTeX 및 음악 점수의 세 가지 도메인에서 실행됩니다. 또한, 기존의 이미지 메트릭을 사용하여 이미지 간의 비교를 효율적으로 수행합니다.

- **Performance Highlights**: 14개의 주요 VLM을 평가한 결과, 모델에 따라 성능이 크게 차이를 보였습니다. 특히, 고급 API 모델들이 개방형 모델들보다 우수하며, GPT-4 Omni가 다양한 작업에서 가장 높은 평균 승률을 기록했습니다. 또한, 다양한 도메인에 따라 최상의 점수가 상이하며, 이는 Image2Struct가 여러 난이도의 작업을 포함하고 있음을 시사합니다.



### Brain age identification from diffusion MRI synergistically predicts neurodegenerative diseas (https://arxiv.org/abs/2410.22454)
- **What's New**: 본 연구에서는 마그네틱 레조넌스 이미지(MRI)로부터 추정한 뇌 연령(brain age)과 변화를 통해 신경퇴행성 질환(neurodegenerative disease)의 조기 징후를 발견할 수 있는 방법을 제시합니다. 특히, 확산 MRI(diffusion MRI, dMRI)를 통해 미세구조(microstructure) 정보에 특화된 뇌 연령을 추정하는 새로운 접근 방식을 소개합니다.

- **Technical Details**: 연구팀은 13,398명의 참가자에서 수집된 12개의 데이터셋에서 dMRI를 이용해 뇌 연령을 추정하는 방법을 개발했습니다. 이를 통해 비강체(non-rigid) 정합을 통해 모든 이미지가 표준 템플릿에 등록되도록 하여 매크로구조(macrostructure) 정보를 최소화했습니다. 연구에서는 매크로구조 정보를 최소화한 모델과 기존의 T1-weighted (T1w) MRI 모델을 비교했습니다.

- **Performance Highlights**: 연구 결과, dMRI 기반의 뇌 연령 추정은 T1w MRI 기반의 뇌 연령과 비교했을 때 신경퇴행 단계에 따라 차이를 보였습니다. 특히, 인지적으로 정상(cognitively normal)에서 경도 인지 장애(mild cognitive impairment)로 진행할 때 dMRI 기반 뇌 연령이 T1w 기반보다 더 높게 나타났지만, 이미 알츠하이머병(Alzheimer's disease) 진단을 받은 참가자에서는 반대로 더 젊게 추정되었습니다. 또한, 약 4년 전 경도 인지 장애 진단 전 dMRI 기반 뇌 연령이 T1w MRI 기반보다 우수한 예측 성능을 보였습니다.



### Addressing Issues with Working Memory in Video Object Segmentation (https://arxiv.org/abs/2410.22451)
Comments:
          12 pages, 11 figures

- **What's New**: 본 논문은 기존의 비디오 객체 분할 (VOS) 모델이 카메라 컷 및 비정상적인 장면 변화에 대해 더욱 강건해질 수 있도록 하는 간단한 알고리즘 변경을 제안합니다. 이 알고리즘은 메모리 버퍼 업데이트를 조정하고, 불필요한 프레임이 메모리에 저장되는 것을 방지하여 객체 추적 성능을 향상시킵니다.

- **Technical Details**: 기존 VOS 모델은 프레임 간 유사성을 기반으로 객체 마스크를 예측합니다. 그러나 카메라 컷과 같은 동작에 대해 정확성을 높이기 위해 L2 거리 기반의 이진 분류기를 도입하였습니다. 이를 통해 모델은 이미지 임베딩 간의 비정상적인 변경을 감지하고 불필요한 프레임을 메모리에 기록하지 않을 수 있습니다.

- **Performance Highlights**: 모델의 성능 실험 결과, 카메라 컷이 있는 비디오 데이터에서 성능이 상당히 향상되었습니다. 제안된 알고리즘 수정을 통해, 쓰기 메모리의 객체 추적 성능이 저하되지 않고 잘 유지되며, 정확한 객체 재식별 (re-identification)이 가능해졌습니다.



### Embedding Watermarks in Diffusion Process for Model Intellectual Property Protection (https://arxiv.org/abs/2410.22445)
- **What's New**: 이번 연구에서는 확산 모델(Diffusion Models)의 워터마크를 전체 확산 프로세스에 삽입하는 혁신적인 기술을 제안합니다. 이 접근법은 모델 출력에 추가 정보가 포함되지 않도록 이론적으로 보장하며, 모델 샘플의 내부 생성 과정에서 통계 알고리즘을 사용하여 워터마크를 검증할 수 있습니다.

- **Technical Details**: 제안된 워터마크 기법은 훈련 단계에서 워터마크를 주입하며, 모델 Sampling 과정의 중간 단계에서도 검증이 가능합니다. 기존의 백도어(Backdoor) 기반 기술과 달리, 워터마크 작업과 주요 작업이 서로 분리될 수 없도록 특별히 설계된 훈련 과정을 사용하여 쉽게 제거할 수 없습니다. 사용된 기법은 Denoising Diffusion Probabilistic Model(DDPM)과 함께 전방 확산 과정에 워터마크를 삽입하는 방법입니다.

- **Performance Highlights**: 광범위한 벤치마크 데이터셋을 통해 수행된 실험 결과, 제안된 워터마킹 기술이 확산 모델의 성능을 저하시키지 않으며, 백도어 기반 방법에 의존하지 않고 워터마크를 명확히 검증할 수 있음을 입증했습니다. 이로 인해 AI 기술의 보다 안전하고 법적으로 인정받는 애플리케이션을 위한 길이 열렸습니다.



### Gradient Distance Function (https://arxiv.org/abs/2410.22422)
Comments:
          We developed this concurrently with 'Neural Vector Field,' and there are similarities between the two works so please pay them a visit as well. Here, we demonstrate how directly learning the gradient vector is much easier than learning the UDF

- **What's New**: 본 논문에서는 Gradient Distance Functions (GDFs)를 사용하여 비워터타이트(non-watertight) 표면을 효과적으로 표현하는 방법을 제안합니다. GDF는 Unsigned Distance Functions (UDFs)의 한계를 극복하고 비구조적 표면의 부드러운 복원을 가능하게 합니다.

- **Technical Details**: GDF는 각 3D 포인트에 대해 가장 가까운 표면 포인트를 향하는 3D 벡터를 연관 지어 정의되며, 그 벡터의 크기는 표면까지의 unsigned distance이고 방향은 가장 가까운 표면 포인트 방향입니다. 이 방식은 표면을 가로지를 때 벡터 구성 요소의 기호가 변경되면서 크기가 0에 가까워지는 식으로, GDF가 SDF와 유사한 행동을 보입니다.

- **Performance Highlights**: ShapeNet Car, Multi-Garment, 3D-Scene 데이터셋에서 GDF의 성능을 평가한 결과, GDF는 UDF와 비교하여 더 나은 결과를 보였으며, 단일 형태 재구성 네트워크나 범주형 오토인코더에서도 이점을 나타냈습니다.



### Exploiting Semantic Scene Reconstruction for Estimating Building Envelope Characteristics (https://arxiv.org/abs/2410.22383)
- **What's New**: 이 논문에서는 기존 건물의 에너지 사용 및 배출량을 줄이기 위한 노후화 재설계(retrofitting) 과정에서 필요한 정확한 건물 외피(건축 외관) 특성 평가 방법을 제안합니다.

- **Technical Details**: 전통적인 방법은 2D 이미지에서 창-벽 비율(window-to-wall ratio)이나 건물 면적(footprint area)과 같은 건물 특성들을 추정하기 위해 딥러닝(detection 또는 segmentation) 기법에 의존했습니다. 그러나 본 연구에서는 서명 거리 함수(signed distance function, SDF) 기반의 신경망(네트워크) 표면 재구성 기법을 활용하여 3D 건물 분석을 수행합니다. 새로운 프레임워크인 BuildNet3D는 2D 이미지 입력을 통해 건물의 기하학적 특성을 추정합니다.

- **Performance Highlights**: 여러 복잡한 건물 구조물에 대해 평가한 결과, BuildNet3D는 높은 정확성과 일반화 가능성을 보여줍니다. 창-벽 비율(window-to-wall ratio)과 건물 면적(footprint)을 추정하는 데 있어 두드러진 성과를 보였습니다. 이 결과는 건물 분석 및 노후화 재설계에 있어 BuildNet3D의 효과성을 입증합니다.



### Accelerating Augmentation Invariance Pretraining (https://arxiv.org/abs/2410.22364)
- **What's New**: 이 연구에서는 Vision Transformers (ViTs)의 사전 훈련을 위한 대조 학습 방법의 계산적 도전 과제를 해결하기 위해, 계산 자원을 줄이기 위한 새로운 가속 프레임워크를 제안합니다.

- **Technical Details**: 제안된 방법은 무작위 토큰 드롭아웃(randomized token dropout)과 유연한 패치 스케일링(flexible patch scaling) 같은 시퀀스 압축 전략을 결합하여 그래디언트 추정(gradiend estimation) 비용을 줄이고 수렴(convergence)을 가속화합니다. 또한, 다양한 가속 전략의 그래디언트 추정 오차를 분석하고 이를 다운스트림 작업에 미치는 영향을 연구합니다.

- **Performance Highlights**: 제안된 방법은 MoCo에서 4배, SimCLR에서 3.3배, DINO에서 2.5배의 훈련 속도를 달성하여 대규모 데이터셋에서 계산 오버헤드를 크게 줄입니다.



### Multi-student Diffusion Distillation for Better One-step Generators (https://arxiv.org/abs/2410.23274)
Comments:
          Project page: this https URL

- **What's New**: Multi-Student Distillation (MSD)라는 새로운 프레임워크를 도입하여 기존의 단일 단계 확산 증류(diffusion distillation) 방법의 효율성을 개선했습니다.

- **Technical Details**: MSD는 조건부 교사 확산 모델을 여러 개의 단일 단계 생성기로 증류(dilute)하여 생성 품질을 향상시키는 방법입니다. 각 학생 생성기는 일부분의 조건 데이터를 처리하며, 빠른 추론 속도를 위한 소형 모델을 훈련시킵니다.

- **Performance Highlights**: 4개의 동등한 크기의 학생 모델을 사용하여, MSD는 ImageNet-64x64에서 1.20 FID, zero-shot COCO2014에서 8.20 FID라는 새로운 최첨단 성능을 달성했습니다.



### Keypoint Abstraction using Large Models for Object-Relative Imitation Learning (https://arxiv.org/abs/2410.23254)
Comments:
          CoRL LangRob Workshop, 2024

- **What's New**: KALM(KP Abstraction using Large Models for Object-Relative Imitation Learning) 프레임워크는 대규모 사전 학습된 비전-언어 모델을 사용하여 자동으로 작업 관련 및 인스턴스 간 일관성이 있는 키포인트를 생성합니다.

- **Technical Details**: KALM은 대규모 사전 학습된 모델을 통해 키포인트 후보를 생성하고 소량의 로봇 시연 데이터를 기반으로 이를 검증하여 견고하고 일관된 키포인트를 도출합니다. 이 과정은 키포인트 중심의 정책 모델을 훈련하여 로봇이 다양한 물체 자세, 카메라 뷰 및 인스턴스에서 일반화할 수 있도록 합니다.

- **Performance Highlights**: KALM은 실제 환경에서 고도의 일반화를 보여주며, 소수의 시연 데이터만으로도 다양한 작업 및 환경에 적응하여 강력한 성능을 발휘합니다.



### bit2bit: 1-bit quanta video reconstruction via self-supervised photon prediction (https://arxiv.org/abs/2410.23247)
Comments:
          NeurIPS 2024

- **What's New**: 이 논문에서는 photon detection events를 1-bit 배열로 나타내는 새로운 sensor 기술인 Quanta image sensors를 다룹니다. 특히, 이 연구는 sparse binary quanta image data에서 원래의 spatiotemporal 해상도로 고품질 이미지 스택을 재구성하는 bit2bit이라는 새로운 방법을 제안합니다.

- **Technical Details**: bit2bit 방법은 photon arrival location의 확률 분포를 예측하여 sparse binary photon data로부터 밀집 이미지 시퀀스를 생성하는 알고리즘입니다. 데이터의 이진성 때문에 Poisson 분포의 가정이 부족하다는 것을 보여주고, 대신 truncated Poisson의 Bernoulli 격자 과정을 모델로 사용합니다. 자가 지도 학습(self-supervised learning) 구조에 기반하여 masked loss function을 통해 이진 데이터를 처리합니다.

- **Performance Highlights**: 모의 데이터에서 우리는 (<0.06 photons per pixel per frame) 매우 photon-sparse binary 입력에 대해 34.35의 평균 PSNR을 달성했습니다. 실제 SPAD 고속 비디오의 새로운 데이터셋을 제작하여 다양한 도전적인 이미징 조건 속에서 우리의 접근 방식의 잠재력을 입증하였고, reconstruction quality와 throughput에서 기존 방법들을 크게 초월했습니다.



### OS-ATLAS: A Foundation Action Model for Generalist GUI Agents (https://arxiv.org/abs/2410.23218)
- **What's New**: OS-Atlas는 GUI grounding과 Out-Of-Distribution (OOD) 작업에서 두각을 나타내는 기본 GUI 행동 모델을 개발하였습니다. 기존 상용 Vision-Language Models (VLMs)에 비해 개방형 VLM이 가진 성능 저하 문제를 해결하기 위한 혁신적인 데이터 및 모델링 기법을 도입하였습니다.

- **Technical Details**: 첫 번째로, OS-Atlas는 다중 플랫폼에서 GUI grounding 데이터를 자동 합성할 수 있는 도구 키트를 개발하여 개방형 데이터 수집을 혁신적으로 용이하게 하였습니다. 이 키트를 이용해 2.3 백만 개의 스크린샷과 1,300만 개 이상의 GUI 요소를 포함하는 가장 큰 규모의 다중 플랫폼 GUI grounding 데이터셋을 공개하였습니다. 또한, OS-Atlas는 GUI 작업을 위한 새로운 방식의 훈련 과정을 통해 높은 정확도를 자랑합니다.

- **Performance Highlights**: OS-Atlas는 모바일, 데스크톱, 웹을 포함한 세 개 플랫폼에서 여섯 가지 벤치마크에 대한 평가를 통해 기존의 최고 성능 모델들에 비해 상당한 성능 향상을 보여주었습니다. 이러한 결과는 OS-Atlas가 상용 VLM, 예를 들어 GPT-4o에 대한 개방형 대안으로서의 가능성을 지니고 있음을 나타냅니다.



### VisualPredicator: Learning Abstract World Models with Neuro-Symbolic Predicates for Robot Planning (https://arxiv.org/abs/2410.23156)
Comments:
          In submission

- **What's New**: 이 논문에서는 Neuro-Symbolic Predicates(NSPs)라는 새로운 1차원 추상화 언어를 소개합니다. 이 언어는 기호(symbolic)와 신경망(neural) 지식 표현의 장점을 결합하여 임무에 특화된 추상화를 형성합니다.

- **Technical Details**: Neuro-Symbolic Predicates는 Python 코드 스니펫으로, 시각-언어 모델(vision-language models, VLMs)을 호출하여 지각적 속성을 쿼리하고 이 속성을 알고리즘적으로 조작할 수 있습니다. 이 접근법은 기존의 심볼릭 세계 모델을 사용한 로봇 작업 계획과는 달리 새로운 환경에 적응할 수 있는 학습 기반의 모델을 제공합니다.

- **Performance Highlights**: 실험 결과, NSPs 접근법은 샘플 복잡성(sample complexity)의 효율성이 더 뛰어나고, 새로운 환경에서의 제너럴리제이션(generalization)이 강하며, 해석 가능성(interpretability)이 개선된 것으로 나타났습니다. 또한, 5개의 로봇 시뮬레이션 환경에서 기존의 기계 학습, 심볼릭, LLM 기반 방식과 비교하여 더 높은 성능을 나타냈습니다.



### Nested ResNet: A Vision-Based Method for Detecting the Sensing Area of a Drop-in Gamma Prob (https://arxiv.org/abs/2410.23154)
- **What's New**: 이번 연구에서는 drop-in gamma probe의 감지 영역을 정확히 예측하기 위한 새로운 깊이 학습 프레임워크를 제안합니다. 기존의 방법과 달리, Nested ResNet 아키텍처와 스테레오 내시경 이미지를 통해 보다 정교한 예측을 이끌어냈습니다.

- **Technical Details**: 우리는 세 가지 가지(branch)로 구성된 깊이 학습 프레임워크를 도입했습니다. 주 가지는 스테레오 내시경 이미지를 입력으로 하여 Nested ResNet 구조를 활용하며, 보조 가지에서는 깊이 추정(depth estimation) 및 프로브 축 방향성(orientation guidance)을 제공합니다. 각 가지에서 추출한 특징을 결합하여 최종 예측을 생성합니다.

- **Performance Highlights**: 제안된 방법은 공개 데이터셋에서 평가되었으며, 기존 방법에 비해 2D 평균 오차는 22.10% 감소하고 3D 평균 오차는 41.67% 줄어드는 성과를 보였습니다. 이러한 결과는 깊은 학습 기반 접근법의 개선을 강조합니다.



### FAIR-TAT: Improving Model Fairness Using Targeted Adversarial Training (https://arxiv.org/abs/2410.23142)
- **What's New**: 이번 연구에서는 공정한 타겟 적대 훈련(Fair Targeted Adversarial Training, FAIR-TAT)이라는 새로운 접근법을 도입하여 적대적 훈련 중 클래스 간 불균형과 모델의 공정성을 개선하는 것을 목표로 하고 있습니다. 구체적으로, 대상 적대적 공격을 통해 클래스별로 훈련 목표를 조정하는 방식을 채택하였습니다.

- **Technical Details**: FAIR-TAT는 전통적인 적대적 훈련(Adversarial Training, AT)에서 비대상 적대적 공격이 아닌 대상 적대적 공격을 사용하여 훈련 진행 중 클래스 간 편향을 동적으로 모니터링하고 조정합니다. 이를 통해 더 높은 클래스 정확도를 달성하고 클래스별 성능 불균형을 감소시킵니다. 또한, 모델의 훈련 과정에서 관찰된 클래스 간 오류율을 기반으로 타겟 클래스를 선택하는 방법을 사용합니다.

- **Performance Highlights**: FAIR-TAT는 기존의 최첨단 방법들에 비해 공정성을 개선하는 데 성공하였으며, 공격을 받는 클래스뿐만 아니라 일반적인 왜곡에 대해서도 높은 성능을 유지함으로써 전체적인 강건성을 보장합니다. 실험 결과, 공정한 타겟 적대적 훈련을 통해 난이도가 높은 클래스에서의 정확도가 향상되었습니다.



### Compositional Segmentation of Cardiac Images Leveraging Metadata (https://arxiv.org/abs/2410.23130)
Comments:
          IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2025

- **What's New**: 본 논문에서는 심장 기능 평가 및 구조 모니터링을 위한 심장 이미지 세분화의 중요성을 강조하며, 새로운 다중 작업(compositional) 세분화 접근 방식을 제안합니다. 이 방법은 심장 이미지를 동시에 로컬라이즈(localize)하고 관심 영역(part-based segmentation)을 세분화하는 것이 특징입니다.

- **Technical Details**: 제안된 방법은 Cross-Modal Feature Integration (CMFI) 모듈을 통해 이미지 획득 시 수집된 메타데이터를 활용하여 심장 이미지 세분화를 수행합니다. 이 모듈은 MRI와 초음파 데이터셋을 사용하여 테스트되었으며, 세분화 네트워크에 메타데이터를 통합하는 방식을 도입하여 이미지 특성의 변동성을 다루고자 합니다. 이를 통해 세분화의 정확성과 강인성을 향상시킵니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 기존의 최첨단 기술보다 월등한 성능을 보인다는 것을 확인하였으며, 다양한 이미징 모달리티에서 일관된 성능 개선이 나타났습니다. 이로 인해 다른 도메인에서도 유사한 정확도 향상을 기대할 수 있습니다.



### Why Fine-grained Labels in Pretraining Benefit Generalization? (https://arxiv.org/abs/2410.23129)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2303.16887

- **What's New**: 이 논문은 심층 신경망(deep neural network)의 사전 훈련(pretraining)에서 미세하게 레이블이 지정된 데이터(fine-grained labeled data)를 사용하고, 이를 통해 파인 튜닝(fine-tuning)을 하는 기법을 제안합니다. 이는 일반적인 레이블이 붙은 데이터(coarse-labeled data)에 비해 더 나은 일반화(generalization)을 제공합니다.

- **Technical Details**: 이 연구는 '계층적 다중 관점(hierarchical multi-view)' 구조를 도입하여 입력 데이터 분포를 제한합니다. 이를 통해 코스 그레인(pretraining with coarse-grained data)은 신경망이 공통 특징(common features)만을 학습하도록 허용하고, 미세 그레인(pretraining with fine-grained data)은 공통 특징과 드문 특징(rare features)을 모두 학습하게 하여, 어려운 다운스트림 테스트 샘플(test samples)에서 향상된 정확도를 제공합니다.

- **Performance Highlights**: 미세하게 레이블이 지정된 데이터로 사전 훈련한 경우, 신경망은 더 나은 정확도를 보여주며, 특히 어려운 샘플에서의 성능이 개선됨을 논문의 결과에서 확인할 수 있습니다.



### AI-assisted prostate cancer detection and localisation on biparametric MR by classifying radiologist-positives (https://arxiv.org/abs/2410.23084)
- **What's New**: 본 연구에서는 방사선 전문의의 해석을 초월하여, 환자 및 병변의 분류를 통해 전립선암 진단 정확도를 향상시키기 위한 딥러닝 모델을 개발하는 데 중점을 두었습니다. 기존 모델들과의 차별점은 방사선 전문의가 양성으로 식별한 사례만을 사용한다는 점입니다.

- **Technical Details**: 단일 보텍스(voxel) 수준 분류 모델을 설계하였으며, 병변, Barzell 존 및 환자 수준에서 긍정적인 사례를 판별하기 위한 간단한 퍼센트 임계값(threshold)을 활용합니다. 연구는 UCLA와 UCL PROMIS 연구에서 각각 800명과 500명의 환자에서 기록된 조직병리학적 레이블을 가진 MR 이미지를 기반으로 하였습니다.

- **Performance Highlights**: 제안된 방식은 UCLA 데이터 세트에서 방사선 전문의 독립형 진단으로 36.3%의 특이성(specificity)을 44.1%로 향상시키는 등의 성과를 보여주었습니다. 이는 불필요한 생검(biopsies)을 줄이고, 암 선별(cost in cancer screening) 비용을 낮추며, 치료에 대한 위험을 정량화하는 데 있어 임상적 가치가 있음을 시사합니다.



### Controlling Language and Diffusion Models by Transporting Activations (https://arxiv.org/abs/2410.23054)
- **What's New**: 이 논문에서는 Activation Transport (AcT)라는 새로운 프레임워크를 소개하며, 이는 Optimal Transport(OT) 이론에 기반하여 모델의 활성화를 유도할 수 있는 방법론입니다. 이 기법은 기존의 활성화 조정 방법들을 통합적으로 설명하며, 대조군 언어 또는 서로 다른 스타일의 텍스트에서의 변환을 효과적으로 수행할 수 있습니다.

- **Technical Details**: AcT는 모달리티에 구애받지 않으며, 모델의 내부 활성화 분포를 보존하면서도 미세한 조정을 가능하게 합니다. 이 방법은 λ (lambda)라는 강도 매개변수를 이용해 개입 정도를 조절할 수 있으며, 0에서 1 사이의 값으로 설정하여 부분적 또는 전체적 변환을 적용합니다. 특히, 저자는 Linear-AcT 방식이 기존의 개입 방법들과 비교하여 더 나은 성과를 낼 수 있음을 실험적으로 입증했습니다.

- **Performance Highlights**: AcT는 Large Language Models (LLMs)에서 독성 감소, 개념 유도, 진실성 증가를 효과적으로 범위 내에서 수행하며, Text-to-Image (T2I) 모델에서도 미세한 스타일 조정 및 개념 부정이 가능함을 보여줍니다. 이 연구는 LLMs와 확산 모델에서 동시에 효과적인 개입 방법을 적용한 최초의 사례로 자리잡고 있습니다.



### Neural Attention Field: Emerging Point Relevance in 3D Scenes for One-Shot Dexterous Grasping (https://arxiv.org/abs/2410.23039)
- **What's New**: 이번 연구에서는 네트워크 주의 필드(neural attention field)를 제안하여, 객체 및 맥락 변화를 가진 새로운 장면에서 능숙한 잡기를 단 한 번의 시연으로 전이할 수 있는 접근 방식을 개발하였습니다. 이 방법은 3D 공간에서 의미론적 밀집 특성 필드를 표현하며, 각 점의 개별적 특성을 모델링하는 대신 점 간의 관련성을 학습합니다.

- **Technical Details**: 제안하는 네트워크 주의 필드는 변환기(decoder) 구조를 기반으로 하며, 3D 쿼리 포인트와 모든 장면 포인트 간의 크로스 어텐션(cross-attention)을 계산합니다. 이를 통해 각 쿼리 포인트의 특성을 어텐션 기반 집합(aggregation)으로 제공하며, 자체 감시(self-supervised) 학습 프레임워크를 통해 제한된 3D 포인트 클라우드(point cloud) 데이터만으로 훈련이 가능하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 이 방법은 작업 관련 장면 영역에 대한 엔드 이펙터의 집중을 촉진하여, 다양한 복잡한 시나리오에서 뛰어난 성공률을 보여주었습니다. 특히, 실제 로봇 실험에서 우리는 기존 기능 필드 기반 방법 대비 현저한 성공률 개선을 확인하였습니다.



### DexGraspNet 2.0: Learning Generative Dexterous Grasping in Large-scale Synthetic Cluttered Scenes (https://arxiv.org/abs/2410.23004)
- **What's New**: 이 논문에서는 1319개의 객체, 8270개의 장면, 4억 2700만 개의 그리프(grasps)를 포함하는 대규모 합성 벤치마크인 DexGraspNet 2.0을 제시하였습니다. 이 벤치마크는 혼잡한 장면에서의 로봇 능조작(grasping) 성능을 평가하는 데 사용됩니다. 또한, 로컬 지오메트리(local geometry)에 조건을 두고 데이터로부터 효율적으로 학습하는 혁신적인 두 단계 그리프 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 복잡한 분포(complicated distribution)의 그리프를 보다 효과적으로 처리하기 위해 생성 모델(generative model)을 활용하며, 이는 출력 품질을 향상시킵니다. 또한, 로컬 피처(local features)에 조건을 두어 데이터 세트의 다양한 변화를 잘 활용하여 새로운 객체와 장면에 대한 일반화를 향상시킵니다.

- **Performance Highlights**: 시뮬레이션 실험에서 제안된 방법이 모든 기준 모델을 초월하였으며,
결과적으로 혼잡한 장면에서의 실제 세계에서 90.7%의 성공률을 달성했습니다. 이는 완전 합성 데이터로 교육된 모델의 실용성을 확인하는 중요한 결과입니다.



### Towards Population Scale Testis Volume Segmentation in DIXON MRI (https://arxiv.org/abs/2410.22866)
- **What's New**: 이번 논문은 UK Biobank의 MRI 데이터를 사용하여 고환 부피를 세그멘테이션(화면 분할)하는 방법을 평가합니다. 이 연구는 고환의 크기를 평가하는 데 기계 학습을 활용하여 대규모 인구 데이터에서의 적용 가능성을 탐구하고 있습니다.

- **Technical Details**: Deep learning 기술을 사용하여 MRI 데이터에서 고환의 세그멘테이션을 수행하는 세 가지 주요 기여가 있습니다. 첫째, UK Biobank 데이터셋에 대한 새로운 주석된 데이터셋을 제공하고, 둘째, 인간 전문가의 주석 품질을 검토하며, 셋째, 최종 모델 및 관련 가중치를 공개하여 기준선을 제공합니다. 연구에서는 DeepLabV3, DeepLabV3Plus, UNet 아키텍처와 같이 여러 컨볼루션 기반 아키텍처와 그것의 변형인 3D UNet 아키텍처를 비교하였습니다.

- **Performance Highlights**: 모델의 중위 Dice 점수는 0.87로, 동일 데이터셋에서 인간 판독자 간 신뢰도 중위 Dice 점수인 0.83보다 우수했습니다. 이로 인해 대규모 고환 MRI 세그멘테이션 연구에서 접근성과 재현성을 높이는 데 기여할 것입니다.



### Latent Diffusion, Implicit Amplification: Efficient Continuous-Scale Super-Resolution for Remote Sensing Images (https://arxiv.org/abs/2410.22830)
- **What's New**: 본 논문은 기존의 고정된 정수 배율(scale factor)을 가진 슈퍼 해상도(SR) 방법의 한계를 극복하고, 비정수 배율도 유연하게 처리할 수 있는 E$^2$DiffSR 모델을 제안합니다. 이를 통해 원거리 감지(remote sensing) 이미지에서의 연속 배율 SR을 가능하게 합니다.

- **Technical Details**: E$^2$DiffSR은 두 단계의 잠재(diffusion) 확산(paradigm) 프로세스를 채택합니다. 첫 번째 단계에서는 자동 인코더(autoencoder)를 사용해 고해상도(HR)와 저해상도(LR) 이미지 간의 차이 정보를 학습합니다. 두 번째 단계에서는 조건부 확산(diffusion) 모델을 통해 잠재 공간(latent space) 내에서 진짜 차별 이전(differential prior)을 예측합니다.

- **Performance Highlights**: E$^2$DiffSR은 최신 SR 방법들과 비교하여 객관적인 메트릭(objective metrics)과 시각적 품질(visual quality)에서 우수한 성능을 보여주며, 확산 기반 SR 방법의 추론 시간(inference time)을 비 확산 방법과 비슷한 수준으로 줄였습니다.



### Contrastive Learning and Adversarial Disentanglement for Privacy-Preserving Task-Oriented Semantic Communications (https://arxiv.org/abs/2410.22784)
Comments:
          Submitted to EEE Journal on Selected Areas in Communications (JSAC): Intelligent Communications for Real-Time Computer Vision (Comm4CV)

- **What's New**: 본 논문에서 제안된 CLAD(contrastive learning and adversarial disentanglement) 방법론은 태스크 지향적 의미 통신 시스템의 정보 전송 방식에 혁신을 가져옵니다. 기존의 문제점인 태스크 관련 및 비관련 정보를 완전히 분리하지 못하는 한계를 극복하기 위해 정보 병목(information-bottleneck) 방식을 도입하였습니다.

- **Technical Details**: CLAD는 대조 학습(contrastive learning)을 활용하여 태스크 관련 기능(feature)을 효과적으로 캡처하는 동시에 적대적 분리(adversarial disentanglement)를 통해 태스크와 무관한 정보를 제거합니다. 또한, 인코딩된 기능 벡터의 정보 유지 지수(information retention index, IRI)를 도입하여 인코딩된 기능과 입력 간의 상호 정보(mutual information)를 대리하는 지표로 사용합니다.

- **Performance Highlights**: CLAD는 태스크 성능, 프라이버시 보존 및 IRI 측면에서 최신 기술들의 성능을 능가하였습니다. CLAD는 약 2.5-3%의 예측 성능 향상, 77-90%의 IRI 감소 및 57-76%의 적대 정확도 감소를 달성하였습니다.



### st-DTPM: Spatial-Temporal Guided Diffusion Transformer Probabilistic Model for Delayed Scan PET Image Prediction (https://arxiv.org/abs/2410.22732)
- **What's New**: 이번 연구는 이중 시간 PET(dual-time PET) 이미징에서의 예측 문제를 해결하기 위해 새로운 공간-시간 유도 확산 변환 모델(spatial-temporal guided diffusion transformer probabilistic model, st-DTPM)을 제안합니다.

- **Technical Details**: 이 구조는 CNN의 패치-와이즈(patch-wise) 특성과 Transformer의 픽셀-와이즈(pixel-wise) 관련성을 통합한 U-net 프레임워크를 활용합니다. 이후 조건부 DDPM(Conditional Denoising Diffusion Probabilistic Model) 모델을 사용하여 이미지 합성을 진행합니다. 공간 조건에서는 초기 스캔 PET 이미지와 노이즈가 있는 PET 이미지를 각 디노이징 단계에서 결합하여 디노이징 샘플링의 공간 분포를 유도합니다. 시간 조건에서는 확산 시간 단계와 지연 시간을 보편적인 시간 벡터로 변환하여 모델 아키텍처의 각 레이어에 삽입하여 예측 정확도를 더욱 향상시킵니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 이미지 품질과 구조 정보를 보존하는 데 있어 다른 접근 방식보다 우수함을 입증하였으며, 예측 작업의 효율성을 확인했습니다.



### Robotic State Recognition with Image-to-Text Retrieval Task of Pre-Trained Vision-Language Model and Black-Box Optimization (https://arxiv.org/abs/2410.22707)
Comments:
          Accepted at Humanoids2024

- **What's New**: 이 논문에서는 로봇이 환경과 객체의 상태를 인식하는 방법으로, 사전 훈련된 vision-language 모델을 활용하는 새로운 방식을 제안합니다. 기존의 수작업 어노테이션, 특수 센서 준비 및 수동 프로그래밍 없이, 언어 프롬프트를 이용하여 상태 인식의 정확성을 높일 수 있습니다.

- **Technical Details**: 제안된 방법은 CLIP 모델을 활용하여, 미리 정의된 텍스트 프롬프트와 현재 이미지 간의 유사성을 계산하여 상태 인식을 수행합니다. black-box optimization을 통해 각 프롬프트의 최적 가중치를 부여하여 더욱 정확한 인식이 가능합니다.

- **Performance Highlights**: 이 연구를 통해 투명한 문이 열려 있는지 닫혀 있는지, 수도꼭지에서 물이 흐르고 있는지, 주방이 깨끗한지의 상태를 인식할 수 있으며, 기존의 방법들에 비해 수고를 덜고 자원을 효율적으로 관리할 수 있습니다.



### Consistency Diffusion Bridge Models (https://arxiv.org/abs/2410.22637)
Comments:
          NeurIPS 2024

- **What's New**: 이번 연구에서는 노이즈와 데이터 사이의 확률 흐름을 직접 예측하는 일관성 함수(consistency function)를 학습하는 새로운 접근방식을 제안합니다. 이는 기존의 DDBM(denoising diffusion bridge models)보다 더 나은 샘플링 효율성을 가져다 줍니다.

- **Technical Details**: CDBM(consistency diffusion bridge models)이라는 새로운 모델을 도입하며, 두 가지 훈련 패러다임인 일관성 브릿지 증류(consistency bridge distillation)와 일관성 브릿지 훈련(consistency bridge training)을 제공합니다. 이는 DDBM 공식과 일관성 모델을 유연하게 통합할 수 있는 구조적 기술들을 포함합니다.

- **Performance Highlights**: CDBM은 기존 DDBM보다 4배에서 50배 더 빠른 샘플링 속도를 보이며, 동일한 단계에서 이미지 번역 및 이미지 인페인팅 작업에서 우수한 시각적 품질을 더해줍니다. FID(Fréchet Inception Distance) 지표를 기준으로 두 단계 생성 후 양쪽에서 개선된 성과를 보여주었습니다.



### FISC: Federated Domain Generalization via Interpolative Style Transfer and Contrastive Learning (https://arxiv.org/abs/2410.22622)
- **What's New**: 이번 연구에서는 다양한 도메인에서 수집된 클라이언트 데이터로 인한 도메인 시프트 문제를 해결하기 위한 새로운 접근법, FISC(Federated Interpolation Style Contrastive learning)를 제안합니다. FISC는 클라이언트 간의 복잡한 도메인 분포를 처리하며 대비 학습(contrastive learning)을 통해 멀티 도메인 표현(multi-domain representations)을 획득합니다.

- **Technical Details**: FISC는 로컬 스타일(local styles)에서 인터폴레이션 스타일을 추출하고 클라이언트 간의 공유 도메인 지식을 융합합니다. 이를 통해 모든 클라이언트가 고유한 데이터 및 클래스 정보를 드러내지 않고도 글로벌 스타일을 공유할 수 있도록 합니다. FISC는 클라이언트가 자신이 소속되어 있는 도메인 특성과 글로벌 도메인 특성을 결합하여 새로운 데이터를 생성할 수 있게 합니다.

- **Performance Highlights**: FISC는 PACS, Office-Home, IWildCam 등 다양한 데이터셋에서 기존의 최첨단(Federated Domain Generalization) 방법론보다 3.64%에서 57.22%까지 정확도 향상을 보여주었습니다. 이러한 성과는 클라이언트의 개인 정보를 효과적으로 보존하면서도 모델의 성능을 유지할 수 있음을 나타냅니다.



### Efficient Feature Extraction and Classification Architecture for MRI-Based Brain Tumor Detection (https://arxiv.org/abs/2410.22619)
- **What's New**: 이번 연구에서는 뇌 MRI 스캔을 이용하여 Tumor의 존재 유무를 식별하기 위해 Convolutional Neural Network (CNN) 모델을 훈련시키고, 그것의 정확도를 99.17%에 도달하는 성과를 이루었습니다.

- **Technical Details**: 본 연구에서는 CNN 모델과 KNN, Logistic regression, SVM, Random Forest, Naive Bayes, Perception과 같은 전통적인 머신러닝 모델을 포함하여 총 6개의 머신 러닝 모델을 적용하였습니다. CNN 모델은 DenseNet, ResNet, EfficientNetB0와 같은 다양한 네트워크 아키텍처를 사용하여 최적화되었으며, 특징 추출 및 이미지 분류 프로세스에서 중요성을 강조했습니다.

- **Performance Highlights**: CNN 모델은 99.17%의 정확도를 달성했으며, Precision, Recall, Specificity, F1 score 등의 표준 메트릭을 사용하여 머신 러닝 모델들과 비교되었습니다. 의사 진단의 중요성을 통해 CNN 모델은 Tumor 존재를 식별하고 환자 치료에 기여할 수 있는 사례로 나타났습니다.



### Deep Priors for Video Quality Prediction (https://arxiv.org/abs/2410.22566)
Comments:
          Indian Conference on Computer Vision, Graphics and Image Processing (ICVGIP) 2024 conference tinny paper

- **What's New**: 이번 연구에서는 딥 비디오 프라이어(Deep Video Prior)를 활용한 완전 블라인드 비디오 품질 평가 알고리즘을 설계했습니다. 이 알고리즘은 단일 왜곡된 비디오와 참조 비디오 쌍을 사용하여 딥 비디오 프라이어를 학습하고, 학습된 프라이어를 통해 원본 비디오를 복원하는 방식입니다. 주의할 점은 학습된 프라이어가 고도 왜곡된 비디오 복원에서는 실패할 수 있다는 가설을 세웠습니다.

- **Technical Details**: 우리의 알고리즘은 손상된 비디오와 복원된 비디오 간의 거리를 비디오의 지각적 품질로 사용하여 품질을 평가합니다. 이 방법은 단일 비디오 쌍을 사용하여 학습되며, 어떤 레이블 데이터도 필요하지 않습니다. 제안된 알고리즘은 LCC(Linear Correlation Coefficient) 및 SROCC(Spearman’s Rank Order Correlation Coefficient)에서 기존의 비지도 비디오 품질 평가 알고리즘보다 우수한 성능을 보여줍니다.

- **Performance Highlights**: 본 연구의 제안된 알고리즘은 합성 왜곡된 비디오 품질 평가 데이터셋에서 기존의 비지도 비디오 품질 평가 알고리즘을 능가하는 성과를 달성했습니다. 이는 높은 왜곡이 있는 비디오에서도 일반화된 성능을 보여줄 수 있다는 가능성을 시사합니다.



### Adaptive Aggregation Weights for Federated Segmentation of Pancreas MRI (https://arxiv.org/abs/2410.22530)
- **What's New**: 본 논문에서는 췌장 MRI 세분화에서의 Federated Learning (FL) 알고리즘을 종합적으로 평가하고, 도메인별 차이를 고려하여 모델 집계를 개선하는 적응형 집계 가중치를 도입합니다.

- **Technical Details**: 네트워크의 중앙 서버와 K개의 클라이언트가 협력하여 모델을 훈련하는 FL 설정에서, 각 클라이언트는 고유한 훈련 데이터셋을 가지고 있으며, 감독 하에 모델 가중치를 업데이트합니다. 기존의 Federated Averaging (FedAvg) 방법은 클라이언트 간의 동질성을 가정하지만, 이는 다양한 데이터 분포로 인해 성능 저하를 초래합니다. 본 연구는 이러한 문제를 해결하기 위해 각 클라이언트의 데이터 성격에 따라 적응형 집계 가중치를 도입하여, 보다 높은 성능의 모델을 생성하도록 합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 전통적인 FL 방법에 비해 세분화 정확도를 향상시키고, 도메인 전이의 영향을 줄이는 데 효과적임을 보였습니다. 여러 병원에서 유의미한 성능 향상이 관찰되었습니다.



### EfficientNet with Hybrid Attention Mechanisms for Enhanced Breast Histopathology Classification: A Comprehensive Approach (https://arxiv.org/abs/2410.22392)
- **What's New**: 이 논문은 Hybrid EfficientNet 모델을 기존의 주의 메커니즘(Attention Mechanisms)인 Convolutional Block Attention Module (CBAM), Self-Attention, Deformable Attention과 통합하여 유방암 조직병리 이미지 분류를 향상시키는 새로운 접근 방식을 제시합니다.

- **Technical Details**: 제안된 모델은 BreakHis 데이터셋을 사용하여 4가지 배율(40X, 100X, 200X, 400X)에서 여러 측정 지표로 평가됩니다. Hybrid EfficientNet은 복잡한 조직 구조를 효과적으로 처리하며, CBAM, Self-Attention, Deformable Attention이 포함되어 있어 이미지의 중요한 부분에 초점을 맞춰 특징 추출 능력을 향상시킵니다.

- **Performance Highlights**: 제안된 방법은 400X 배율에서 98.42%의 정확도로 여러 최신 모델(VGG, ResNet 등)을 초과하는 성능을 보여주며, 정확성, F1-score, 정밀도, 재현율 등의 지표로 결과가 검증되었습니다. 이 모델은 실시간 진단 워크플로우에 통합될 수 있는 높은 계산 효율성을 보입니다.



### Rare-to-Frequent: Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with LLM Guidanc (https://arxiv.org/abs/2410.22376)
- **What's New**: 본 연구는 희귀한 개념에 대한 텍스트-이미지(T2I) 분산 모델의 구성능력을 높이기 위해 대형 언어 모델(LLM) 가이드를 활용하는 접근 방식을 제안합니다. 우리는 창의적이고 비범한 특성을 가진 새로운 캐릭터 디자인과 같은 희귀한 프롬프트 생성 과정에서 기존 모델들이 어려움을 겪는 문제를 강조합니다.

- **Technical Details**: 본 연구에서는 희귀한 개념 구성을 위한 새로운 접근 방식인 R2F(Rare-to-Frequent)를 제안합니다. R2F는 LLM을 활용하여 희귀한 컨셉과 관련된 보다 일반적인 빈번한 개념을 찾고, 이를 통해 분산 추론(difussion inference) 과정을 개선합니다.

- **Performance Highlights**: R2F는 다양한 희귀 개념 구성을 포함하는 새로운 벤치마크인 RareBench에서 기존 모델인 SD3.0 및 FLUX와 비교하여 T2I 정렬 정확도에서 최대 28.1%p 향상된 성능을 보였습니다.



### Unpacking SDXL Turbo: Interpreting Text-to-Image Models with Sparse Autoencoders (https://arxiv.org/abs/2410.22366)
- **What's New**: 이 연구에서는 sparse autoencoders (SAEs)를 이용하여 텍스트-이미지 모델의 해석 가능한 특징을 학습할 수 있는 가능성을 조사하였습니다. 특히 SDXL Turbo와 같은 몇 단계의 diffusion 모델에서의 적용을 탐구하였습니다.

- **Technical Details**: SAEs는 SDXL Turbo의 denoising U-net 내에서 transformer 블록의 업데이트에 대해 훈련되었습니다. 이 과정에서 학습된 특징들은 해석 가능하고, 생성 과정에 인과적으로 영향을 미치며, 블록 간의 전문화(specialization)를 드러냈습니다.

- **Performance Highlights**: 연구에서는 이미지 구성, 지역 세부사항 추가, 색상 및 조명, 스타일을 담당하는 블록을 식별하였습니다. 이는 텍스트-이미지 생성 모델의 내부 구조를 이해하는 데 중요한 첫 걸음이 됩니다.



### Vascular Segmentation of Functional Ultrasound Images using Deep Learning (https://arxiv.org/abs/2410.22365)
- **What's New**: 이 논문은 기능적 초음파(fUS) 영상을 위한 최초의 심층 학습(segmentation based deep learning) 기반 세분화 도구를 소개합니다. 이 도구는 ULM(ultrasound localization microscopy) 자동 주석(annotation)을 이용하여 서로 다른 혈관 영역의 신호를 구별하고, 동적 혈관 용적(CBV) 정량화를 가능하게 합니다.

- **Technical Details**: fUS는 비침습적(imaging method) 이미징 방법으로 뇌혈관의 볼륨 변화를 고해상도(spatio-temporal resolution)로 측정합니다. 그러나 같은 픽셀 내에서 혈류 방향이 반대이기 때문에 소동맥(arterioles)과 소정맥(venules)을 구별하는 것이 어렵습니다. 저자는 다양한 UNet(architecture) 아키텍처를 평가하였고, 100개의 시간 프레임(temporal frames)만으로 90%의 정확도와 71%의 F1 점수, 0.59의 IoU를 성취했습니다.

- **Performance Highlights**: 렌즈 구조(segmentation)에서의 경쟁력을 보여주는 결과를 도출했으며, 시각적 자극(visual stimulation) 동안 캡처된 이미지에 잘 일반화됩니다. 이 연구는 fUS 데이터 해석을 향상시키고 혈관 기능(vessel function)에 대한 이해를 높이는 비용 효율적(alternative)인 방법을 제공합니다.



### MMM-RS: A Multi-modal, Multi-GSD, Multi-scene Remote Sensing Dataset and Benchmark for Text-to-Image Generation (https://arxiv.org/abs/2410.22362)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 본 논문에서는 다양한 원격 감지(RS) 시나리오에서 텍스트-이미지 생성이 가능하도록 다중 모달, 다중 GSD(ground sample distance) 및 다중 장면의 원격 감지 데이터셋(MMM-RS)을 제안합니다. 이 데이터셋은 기존의 9개의 공개 RS 데이터셋을 표준화하여 약 210만 개의 텍스트-이미지 쌍을 포함하고 있습니다.

- **Technical Details**: MMM-RS 데이터셋은 RGB, SAR(Synthetic Aperture Radar), NIR 근적외선의 세 가지 모달리티를 포함하며, 각 이미지는 정보가 풍부한 텍스트 프롬프트와 결합되어 있습니다. 이를 위하여 대규모 사전 훈련된 비전-언어 모델(BLIP-2)을 사용하여 자동적으로 텍스트 프롬프트를 생성하고, 수동으로 보정 작업을 수행하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 MMM-RS 데이터셋은 기존의 점착형(diffusion) 모델들이 다양한 RS 이미지를 생성할 수 있도록 지원하며, 다양한 모달리티, 장면 및 기상 조건에서 높은 성능을 보여주었습니다. 특히, 제안된 데이터셋을 통해 수치적 및 정성적 비교 실험을 수행하여 효과성을 입증하였습니다.



### Robots Pre-train Robots: Manipulation-Centric Robotic Representation from Large-Scale Robot Datasets (https://arxiv.org/abs/2410.22325)
- **What's New**: 이번 논문에서는 Manipulation Centric Representation (MCR)라는 새로운 기반 표현 학습 프레임워크를 제안하여 로봇 조작 성능을 향상시키고 있습니다. MCR은 시각적 특징과 조작 작업의 행동 및 고유 환경 정보(고유 감각 정보)를 포착하여 조작 중심성을 개선합니다.

- **Technical Details**: MCR을 위해 DROID 로봇 데이터셋에서 시각 인코더를 사전 학습하고, 로봇의 고유 감각 상태와 행동 같은 동작 관련 데이터를 활용합니다. 핵심으로는 동적 정렬 손실(dynamics alignment loss)과 행동 예측 손실(action prediction loss)을 포함하며, 이를 통해 로봇 데이터셋의 동적 레이블을 효과적으로 활용합니다.

- **Performance Highlights**: MCR은 4개의 시뮬레이션 도메인에서 20개의 작업을 수행하며 이전 방법보다 14.8%의 성능 향상을 보여주었고, UR5e 관절 팔을 사용한 실제 3개의 작업에서는 76.9%의 성능 향상을 이루었습니다.



New uploads on arXiv(cs.AI)

### A little less conversation, a little more action, please: Investigating the physical common-sense of LLMs in a 3D embodied environmen (https://arxiv.org/abs/2410.23242)
Comments:
          25 pages, 4 figures

- **What's New**: 이번 논문에서는 Large Language Models(LLMs)의 물리적 상식 추론(physical common-sense reasoning) 능력을 평가하기 위한 새로운 방법인 LLMs in Animal-AI(LLM-AAI) 프레임워크를 소개합니다. LLM들을 3D 환경 내의 에이전트로 '구현(embody)'하여 기존의 정적 벤치마크를 넘어서 생태적으로 유효한 평가를 가능하게 합니다.

- **Technical Details**: LLM-AAI 프레임워크는 Animal-AI 환경을 이용하여 LLM의 물리적 상식 추론을 평가합니다. 여기서는 복잡한 실험을 통해 거리 추정, 시야에서 보이지 않는 물체 추적, 도구 사용 등의 물리적 추론 능력을 연구합니다. 이와 함께, 이 프레임워크는 Reinforcement Learning(RL) 에이전트 및 인간 아동과의 비교를 가능하게 합니다.

- **Performance Highlights**: LLM-AAI 프레임워크의 실험 결과, 최첨단 멀티모달 모델들이 해당 작업을 수행할 수 있음을 보여주지만, 이러한 과제에서는 인간 아동에게 뒤처지는 결과를 나타냈습니다. 이는 LLM이 물리적 추론에서 여전히 인간 수준에 도달하지 못했음을 시사합니다.



### VisualPredicator: Learning Abstract World Models with Neuro-Symbolic Predicates for Robot Planning (https://arxiv.org/abs/2410.23156)
Comments:
          In submission

- **What's New**: 이 논문에서는 Neuro-Symbolic Predicates(NSPs)라는 새로운 1차원 추상화 언어를 소개합니다. 이 언어는 기호(symbolic)와 신경망(neural) 지식 표현의 장점을 결합하여 임무에 특화된 추상화를 형성합니다.

- **Technical Details**: Neuro-Symbolic Predicates는 Python 코드 스니펫으로, 시각-언어 모델(vision-language models, VLMs)을 호출하여 지각적 속성을 쿼리하고 이 속성을 알고리즘적으로 조작할 수 있습니다. 이 접근법은 기존의 심볼릭 세계 모델을 사용한 로봇 작업 계획과는 달리 새로운 환경에 적응할 수 있는 학습 기반의 모델을 제공합니다.

- **Performance Highlights**: 실험 결과, NSPs 접근법은 샘플 복잡성(sample complexity)의 효율성이 더 뛰어나고, 새로운 환경에서의 제너럴리제이션(generalization)이 강하며, 해석 가능성(interpretability)이 개선된 것으로 나타났습니다. 또한, 5개의 로봇 시뮬레이션 환경에서 기존의 기계 학습, 심볼릭, LLM 기반 방식과 비교하여 더 높은 성능을 나타냈습니다.



### Public Domain 12M: A Highly Aesthetic Image-Text Dataset with Novel Governance Mechanisms (https://arxiv.org/abs/2410.23144)
Comments:
          Project Page: this https URL

- **What's New**: 새로운 Public Domain 12M (PD12M) 데이터셋은 1240만 개의 고품질 공개 도메인 및 CC0 라이선스 이미지를 합성 캡션과 함께 제공하며, 텍스트-이미지 모델 교육을 위해 설계되었습니다. PD12M은 현재까지 가장 큰 공개 도메인 이미지-텍스트 데이터셋으로, 저작권 문제를 최소화하면서 기초 모델(training foundation models) 교육에 충분한 크기를 자랑합니다. 이와 함께, 커뮤니티 기반 데이터셋 거버넌스 메커니즘도 도입하였습니다.

- **Technical Details**: PD12M 데이터셋은 Public Domain Mark 또는 Creative Commons Zero 라이선스가 부여된 자료에서만 수집되었습니다. 3.3M 항목의 하위 데이터셋인 Public Domain 3M을 포함하여 총 1240만 개의 이미지-캡션 쌍을 수집하였습니다. 데이터 수집 과정에서 GLAM (Galleries, Libraries, Archives, and Museums) 출처에서 2310만 개의 이미지를 확보하고, Wikimedia Commons에서 1130만 개의 이미지, iNaturalist에서 320만 개의 이미지를 수집하였습니다. 이미지를 다운로드할 때, 서버 부하를 줄이기 위해 자체 속도 제한(rate limiting)을 사용하였습니다.

- **Performance Highlights**: PD12M 데이터셋은 38M의 이미지 URL 중 필터링 과정을 거쳐 12.4M의 결과물을 도출했으며, 해당 데이터셋은 고해상도 훈련에 적합하도록 최소 해상도 기준을 256x256 픽셀로 설정하였습니다. 데이터셋의 안전성과 품질을 확보하기 위해 LAION의 NSFW 분류기를 사용하여 유해 콘텐츠를 걸러냈고, 부정적인 메타데이터를 제거하는 작업도 진행하였습니다.



### Guided Game Level Repair via Explainable AI (https://arxiv.org/abs/2410.23101)
- **What's New**: 이번 연구는 기계 학습을 통한 프로시저 생성 수준(Procedural Content Generation, PCG)에서 생성된 레벨이 해결 불가능할 수 있음을 다루고 있습니다. 연구진은 레벨 수리가 필요한 경우에 대해 설명 가능성(explainability) 방법을 활용하여, 특정 영역이 문제를 일으키는지를 파악하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 연구진은 2D 타일 기반의 여러 게임(Super Mario Bros., Super Cat Tales, 커스텀 Cave 게임)에서 해결 가능성과 해결 불가능한 레벨을 구별하는 이진 분류기를 훈련합니다. 이 분류기의 설명 가능성 방법을 활용하여 각 타일의 중요도를 평가하고, 이를 기반으로 제약 만족 문제(constraint satisfaction problem)를 해결하는데 필요한 가중치를 제공합니다.

- **Performance Highlights**: 본 연구의 실험 결과, 제안된 접근 방식이 레벨 수리를 더 빠르게 수행할 수 있도록 도와주는 것을 발견했습니다. 이 방법은 자동으로 생성된 레벨의 문제를 파악하고, 보다 효율적으로 수리를 진행할 수 있도록 합니다.



### Emotional RAG: Enhancing Role-Playing Agents through Emotional Retrieva (https://arxiv.org/abs/2410.23041)
- **What's New**: 이 논문은 감정 기반의 기억 검색 프레임워크인 Emotional RAG를 제안하여 역할 수행 에이전트의 응답 생성을 개선하는 방법을 보여줍니다. 이 프레임워크는 기억 검색 과정에서 감정 상태를 고려하여 인간과 유사한 응답을 생성하는 것을 목표로 합니다.

- **Technical Details**: Emotion RAG는 두 가지 검색 전략, 즉 조합 전략(combination strategy)과 순차적 전략(sequential strategy)을 통해 기억의 의미적 관련성과 감정 상태를 통합하는 과정을 포함합니다. 이 프로세스는 LLMs를 기반으로 하여 각 쿼리의 의미와 감정 상태를 벡터로 인코딩하여 역할 수행 에이전트를 통해 적절한 응답을 생성합니다.

- **Performance Highlights**: 세 가지 대표적인 역할 수행 데이터셋에서 실험을 진행한 결과, Emotional RAG 프레임워크는 감정 요소를 고려하지 않은 기존 방법보다 역할 수행 에이전트의 개성 유지에 있어 획기적으로 향상된 성능을 보여주었습니다.



### Semantic Enrichment of the Quantum Cascade Laser Properties in Text- A Knowledge Graph Generation Approach (https://arxiv.org/abs/2410.22996)
- **What's New**: 이 논문에서는 Quantum Cascade Laser (QCL) 특성을 효율적으로 추출하고, 이 정보를 연결하여 분석할 수 있는 지식 그래프(Knowledge Graph) 생성을 제안합니다. 기존의 규칙 기반 접근법 및 기계 학습 방법의 한계를 극복하기 위해 Retrieval Augmented Generation (RAG) 기반의 정보 추출 파이프라인을 통해 QCL property를 추출하는 새로운 방법론을 개발했습니다.

- **Technical Details**: QCL의 특성은 설계 특성(Design properties)과 작동 특성(Optoelectronic/Working properties)으로 구분됩니다. 이 논문에서는 QCL ontology를 기반으로 한 정보 추출 파이프라인을 구축하고, GPT 4-Turbo 언어 모델을 활용하여 QCL properties를 식별합니다. 특히 작동 온도(Working Temperature), 레이저 설계 유형(Laser Design Type), 발진 주파수(Lasing Frequency), 레이저 출력 전력(Laser Optical Power), 그리고 이종구조(Heterostructure) 등과 같은 특성을 중점적으로 분석합니다.

- **Performance Highlights**: 실험 결과는 이 접근법이 비구조적인 텍스트에서 QCL properties를 효과적으로 추출하고, QCL properties를 위한 지식 그래프를 생성하는 데 있어 실행 가능성과 효율성을 보여줍니다. 이 지식 그래프는 QCL 데이터의 의미적 강화 및 분석에 잠재적인 응용 가능성을 가지고 있습니다.



### BIS: NL2SQL Service Evaluation Benchmark for Business Intelligence Scenarios (https://arxiv.org/abs/2410.22925)
Comments:
          This paper has been accepted by ICSOC (International Conference on Service-Oriented Computing) 2024

- **What's New**: 이 논문에서는 기존의 NL2SQL (Natural Language to Structured Query Language) 벤치마크가 비즈니스 인텔리전스 (BI) 시나리오에 적합하지 않다는 점을 인식하고, 일반적인 BI 질문에 초점을 맞추어 새로운 벤치마크인 BIS를 제안합니다.

- **Technical Details**: BIS 벤치마크는 일반적인 BI 질문과 데이터베이스 스키마에 중점을 두고 있으며, 쿼리 유사성과 결과 유사성을 평가하기 위한 두 가지 새로운 평가 메트릭을 포함하고 있습니다. 또한, BI 응용 프로그램에서 NL2SQL 모델의 성능을 더욱 정확하게 평가할 수 있도록 두 가지 평가 지표를 제안합니다.

- **Performance Highlights**: 이 논문은 기존 검사 방법의 한계를 밝히고, 다양한 비즈니스 인텔리전스 질문에 대한 성능을 더 잘 반영하는 방안을 제시합니다. 특히, 쿼리의 부분 유사성과 비즈니스 분석 관점에서의 질문 분류의 중요성을 강조합니다.



### Self-optimization in distributed manufacturing systems using Modular State-based Stackelberg Games (https://arxiv.org/abs/2410.22912)
Comments:
          This pre-print was submitted to Journal of Manufacturing Systems on October 30, 2024

- **What's New**: 본 연구에서는 분산형 자가 학습 모듈식 제조 시스템을 위한 새로운 게임 구조인 Modular State-based Stackelberg Games (Mod-SbSG)를 소개합니다. Mod-SbSG는 State-based Potential Games (SbPG)와 Stackelberg 게임을 통합하여, 생산 시스템 내 자가 학습 에이전트 간의 협력적 의사 결정을 향상시킵니다.

- **Technical Details**: Mod-SbSG는 중요도가 높은 모듈이 선도자 역할을 수행하고, 덜 중요한 모듈이 이에 최적응답하는 계층적인 게임 구조를 제공합니다. 이는 전통적인 다중 에이전트 학습 알고리즘과 달리 동시에 결정하는 방식이 아닌 계층적 의사결정 과정을 도입합니다. 이 게임 구조는 강화학습(Deep Reinforcement Learning) 및 경량화된 학습 알고리즘에 통합되어 자가 최적화 알고리즘과 호환됩니다.

- **Performance Highlights**: Mod-SbSG는 실험적으로 두 가지 실험 환경에서 테스트되어, 기존 SbPG와 비교하여 시스템 오버플로우를 97.1% 감소시켰고, 특정 경우에는 오버플로우를 완전히 방지했습니다. 또한, 생산 수요를 충족시키면서도 에너지 소비를 5-13% 줄이는 성과를 거두어 잠재적 글로벌 목표 값을 상당히 향상시켰습니다.



### Reliability Assessment of Information Sources Based on Random Permutation S (https://arxiv.org/abs/2410.22772)
Comments:
          10 pages

- **What's New**: 이번 논문은 Dempster-Shafer 이론(DST)의 불확실성 처리에 대한 실질적인 기여를 하며, 랜덤 순열 집합(Random Permutation Set, RPS)을 통해 기존의 DST를 확장시키는 방법을 제안합니다. 특히, RPS의 내부 요소 순서를 고려하여 DST에서 RPS로의 변환 방법과 RPS를 위한 확률 변환 방법을 개발하고 이를 패턴 인식에 적용합니다.

- **Technical Details**: 이 논문에서는 RPS의 확률 변환 방법을 제안하며, RPS의 내부 순서를 활용하여 DST의 제한사항을 극복하고 보다 정밀한 질량 함수(mass function) 계산이 가능하도록 합니다. RPS는 Permutation Event Sets (PES)와 Permutation Mass Functions (PMF)를 도입하여 기존의 증거 이론과 확률 이론을 통합하는 방향으로 발전합니다. 또한, RPS에 대한 신뢰성 계산 방법도 제안합니다.

- **Performance Highlights**: 제안된 접근 방식은 DST와 RPS 간의 간극을 메우고, 분류 문제에서 높은 인식 정확도를 달성하는 데 효과적임을 입증하는 실험 결과를 보여줍니다. 다양한 수치 예제와 실제 응용을 통해 제안된 알고리즘의 효과성을 입증하였습니다.



### Self-Driving Car Racing: Application of Deep Reinforcement Learning (https://arxiv.org/abs/2410.22766)
- **What's New**: 딥 강화 학습(Deep Reinforcement Learning) 기술을 자율 자율주행 자동차 레이싱 분야에 적용한 연구로, OpenAI Gymnasium CarRacing 환경에서 AI 에이전트를 훈련시켜 모의 자동차를 효율적으로 운전하는 방법을 개발하였다.

- **Technical Details**: 여러 강화 학습 알고리즘을 조사하며, Deep Q-Network(DQN), Proximal Policy Optimization(PPO), 전이 학습(Transfer Learning), 순환 신경망(Recurrent Neural Network, RNN)을 통합하여 성능을 향상시키기 위한 연구를 진행하였다. DQN은 정책 학습의 강력한 기준을 제공하지만 ResNet과 LSTM 모델을 통합하면 복잡한 공간 및 시간 역학을 포착하는 능력이 크게 향상된다.

- **Performance Highlights**: DQN은 정책 학습에서 강력한 기준을 제공하며, PPO는 연속 행동 공간에서 세밀한 제어를 위한 유망한 결과를 보였다. 그러나 정책 붕괴(policy collapse)와 같은 도전 과제가 여전히 남아 있으며, 성능 비교와 더불어 계산 효율성 개선 및 모델 안정성 문제 해결을 위한 향후 연구 방향을 제시하였다.



### A Walsh Hadamard Derived Linear Vector Symbolic Architectur (https://arxiv.org/abs/2410.22669)
Comments:
          To appear in the 38th Conference on Neural Information Processing Systems (NeurIPS 2024)

- **What's New**: 본 논문에서는 Hadamard 유도 선형 바인딩(HLB)을 소개합니다. HLB는 전통적인 VSA 작업에서 뛰어난 성능을 발휘할 수 있도록 설계되었으며, 딥러닝 환경에서도 효과적으로 작동할 수 있는 특성을 가지고 있습니다.

- **Technical Details**: HLB는 Walsh Hadamard 변환에서 유도된 VSA로, 바인딩 단계에서의 계산 복잡도를 O(d)로 축소하고, 수치적 안정성을 보장합니다. 이 메서드는 바인딩 연산에서 O(d log d)와 같은 전통적인 Hadamard 변환의 복잡도를 피하고, 보다 비용이 많이 드는 VSA 대안보다 우수한 성능을 나타냅니다.

- **Performance Highlights**: HLB는 전통적인 VSA 벤치마크 결과 및 두 가지 최근 딥러닝 작업에서의 성능을 개선하였으며, 기존 VSA 시스템과 기준 작업에서 비교 시 동등하거나 개선된 성능을 보여줍니다.



### CoGS: Model Agnostic Causality Constrained Counterfactual Explanations using goal-directed ASP (https://arxiv.org/abs/2410.22615)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2407.08179

- **What's New**: 이 논문은 머신 러닝 모델의 블랙 박스 특성을 극복하기 위해 CoGS(Counterfactual Generation with s(CASP))라는 모델 불문 프레임워크를 도입하여, 의사 결정 과정을 투명하게 만드는 방법을 제안합니다.

- **Technical Details**: CoGS는 목표 지향 방식의 Answer Set Programming 시스템인 s(CASP)를 활용하여, 특징 값(feature values)의 실제적이고 인과적으로 일관된 변화를 계산합니다. 또한, RBML(rule-based machine learning) 알고리즘, 특히 FOLD-SE 알고리즘을 사용하여 통계적 모델의 기본 로직을 추출하고, 바람직한 결과를 달성하기 위해 필요한 변화를 단계별로 제시합니다.

- **Performance Highlights**: CoGS는 원치 않는 결과에서 원하는 결과로의 경로를 추적하여 사용자에게 해석 가능하고 실행 가능한 설명을 제공합니다.



### ML Research Benchmark (https://arxiv.org/abs/2410.22553)
- **What's New**: 본 논문에서는 AI 에이전트의 성능을 평가하고 벤치마킹하기 위한 새로운 방법론인 ML Research Benchmark (MLRB)를 소개합니다. MLRB는 최신 머신 러닝 대회의 7개 경쟁 과제를 포함하여 AI 연구 및 개발에 필요한 도전 과제를 평가하는 프레임워크입니다.

- **Technical Details**: ML Research Benchmark는 모델 훈련 효율성, 제한된 데이터에 대한 프리트레이닝(Pretraining), 도메인 특화 미세 조정(Fine-tuning), 모델 압축(Model Compression) 등의 활동을 포함합니다. 평가에는 최신 프런티어 모델들인 Claude-3 및 GPT-4o가 사용되었습니다.

- **Performance Highlights**: Claude-3.5 Sonnet 에이전트가 벤치마크 전반에 걸쳐 가장 우수한 성능을 보였지만, 시험한 모든 에이전트가 비트리비얼(non-trivial) 연구 반복 작업 수행에 어려움을 겪었습니다. 이러한 결과는 현재 AI 에이전트가 복잡한 지시를 처리하고 기본적인 결과를 생성할 수 있지만, 고급 AI 연구에 필요한 능력에는 미치지 못함을 보여줍니다.



### From Silos to Systems: Process-Oriented Hazard Analysis for AI Systems (https://arxiv.org/abs/2410.22526)
- **What's New**: AI 시스템의 잠재적 위험을 처리하기 위해 시스템 수준의 위험을 식별하고 완화하는 것이 중요하다는 점을 강조합니다. 이 논문은 기존의 AI 시스템 분석 방법이 개별 구성 요소에 초점을 맞추고 있다는 점을 비판합니다.

- **Technical Details**: 시스템 이론적 프로세스 분석(System Theoretic Process Analysis, STPA)을 AI 운영 및 개발 프로세스 분석에 적용합니다. 특히, 머신 러닝 알고리즘에 의존하는 시스템에 대한 세 가지 사례 연구(선형 회귀, 강화 학습, 변환기 기반 생성 모델)를 통해 STPA의 적용을 검토합니다.

- **Performance Highlights**: PHASE(Process-oriented Hazard Analysis for AI Systems) 지침을 제시하여 STPA 개념을 AI에 맞게 조정합니다. 이는 위험 식별, 사회적 요인 인식, 책임 있는 규명 및 새로운 위험 모니터링의 네 가지 주요 가능성을 제공합니다.



### RealCQA-V2 : Visual Premise Proving (https://arxiv.org/abs/2410.22492)
Comments:
          Under Review : Code and Data will be made public soon

- **What's New**: 본 논문에서는 Visual Premise Proving (VPP)이라는 새로운 작업을 소개하여 차트 질문 응답(Chart Question Answering, CQA) 프로세스를 논리적 전제로 세분화하여 모델의 추론 능력을 평가합니다. VPP는 기존의 정확도 기반 평가 방식에서 벗어나, 모델이 각 전제를 연속적으로 검증할 수 있는 능력에 중점을 둡니다.

- **Technical Details**: VPP는 차트 질문 응답을 위한 새로운 프레임워크로, 각 전제가 차트 이해와 결론 도출을 위해 필요한 단계를 나타냅니다. 이 작업은 First-Order Logic (FOL) 문제에 제한하고, RealCQA 데이터셋을 사용하여 1,000만 개의 과학 차트 질문-답변 쌍을 기반으로 Sequential Reasoning을 평가하는 데 중점을 둡니다. VPP는 두 가지 새로운 평가 지표, ACC_{VPP}와 DCP를 제안합니다.

- **Performance Highlights**: MATCHA 모델을 활용한 제로샷 연구에서, 차트 추론 분야에서 27%의 성능을 보인 반면, 차트 구조와 데이터 검색에서는 각각 19% 및 14%를 기록하였습니다. 이는 모델이 시각적 데이터의 변화에도 불구하고 더 나은 추론 능력을 일반화할 수 있음을 나타냅니다.



### Predicting Future Actions of Reinforcement Learning Agents (https://arxiv.org/abs/2410.22459)
Comments:
          16 pages, 8 figures

- **What's New**: 이 논문은 강화 학습(목표에 맞지 않는 행동을 줄이기 위해 미래의 에이전트 행동 및 사건 예측이 중요한 중요성을 논의합니다. 특히, 명시적 계획, 암묵적 계획 및 비계획 에이전트의 세 가지 유형에 대한 예측의 효과성을 비교 평가합니다.

- **Technical Details**: 미래 행동과 사건 예측을 위한 두 가지 방법을 제안합니다. 첫 번째는 inner state approach로 에이전트의 내부 계산 기반으로 예측합니다. 두 번째는 simulation-based approach로, 학습된 세계 모델에서 에이전트를 펼쳐보며 행동을 관찰합니다. 이 두 방식을 다양한 RL 알고리즘에 적용하여 예측 가능성을 평가합니다.

- **Performance Highlights**: 실험 결과, 명시적 계획 에이전트의 계획은 다른 유형의 뉴런 활성화보다 예측에 훨씬 더 정보가 풍부하다는 사실이 확인되었습니다. 또한, 내부 계획을 사용할 경우, 모델 품질에 비해 행동 예측의 견고성을 더 잘 보여주었습니다. 사건 예측의 경우 결과가 혼합되어 있습니다. 이러한 발견은 미래 에이전트 행동 및 사건 예측을 위한 내부 상태와 시뮬레이션 활용의 이점을 강조합니다.



### Advancing Agentic Systems: Dynamic Task Decomposition, Tool Integration and Evaluation using Novel Metrics and Datas (https://arxiv.org/abs/2410.22457)
Comments:
          38th Conference on Neural Information Processing Systems (NeurIPS 2024), NeurIPS 2024 Workshop on Open-World Agents

- **What's New**: 이 논문은 자율 에이전트 시스템의 발전을 위해 'Advanced Agentic Framework'를 제안하며, 이를 통해 다단계 작업을 역동적으로 처리하고 적절한 도구를 자동으로 선택할 수 있는 능력을 강화하고 있습니다.

- **Technical Details**: 논문에서는 주요 기여로 세 가지 요소를 제시합니다: 1) 다단계 쿼리를 처리하고 작업 그래프를 생성 및 실행하며 적합한 도구를 선택하고 실시간 변화에 적응하는 고급 에이전트 프레임워크, 2) 에이전트 시스템에 대한 전반적인 평가를 위한 새로운 평가 지표(Node F1 Score, Structural Similarity Index (SSI), Tool F1 Score), 3) 다양한 작업 복잡성에서 에이전트 행동을 분석하기 위한 AsyncHow 기반의 전문 데이터셋을 개발하였습니다.

- **Performance Highlights**: 강화된 작업 그래프 분해는 시스템의 응답성과 확장성을 크게 향상시킵니다. 특히, SSI는 순차적 작업에서 성능의 가장 중요한 예측 변수가 되고, Tool F1 Score는 병렬 작업에서 필수적인 성과 지표로 나타났습니다.



### Provable acceleration for diffusion models under minimal assumptions (https://arxiv.org/abs/2410.23285)
- **What's New**: 본 논문에서는 확률적 샘플러를 위한 새로운 훈련 없이도 가능한 가속화 방식이 제안되었습니다. 이 방식은 표준 score-based 샘플러의 높은 계산 부담을 줄이고 샘플링 속도를 개선하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 제안하는 가속화 샘플러는 $L^2$ 정확한 score 추정과 목표 분포에 대한 유한한 두 번째 모멘트 조건을 최소한의 가정으로 두고 있습니다. 이로써 $	ilde{O}(d^{5/4}/	ext{sqrt}(oldsymbol{	ext{ε}}))$ 반복 횟수 내에 총 변동에서 $oldsymbol{	ext{ε}}$-정확도를 보장할 수 있습니다. 기존 score-based 샘플러의 복잡도인 $	ilde{O}(d/oldsymbol{	ext{ε}})$를 상당히 개선하였습니다.

- **Performance Highlights**: 제안된 이론은 목표 분포에 대한 제한적인 가정이나 고차 score 추정 보장에 의존하지 않기 때문에, 더 다양한 분포에 대해 적용이 가능합니다.



### A Neural Transformer Framework for Simultaneous Tasks of Segmentation, Classification, and Caller Identification of Marmoset Vocalization (https://arxiv.org/abs/2410.23279)
- **What's New**: 본 연구에서는 Marmoset의 음성 통신을 세분화(segmentation), 분류(classification), 발신자 식별(caller identification)을 동시에 수행하기 위해 Transformer 아키텍처를 제안합니다. 이는 기존의 CNN보다 쿼드러틱 복잡성이 낮아 장거리 음향 패턴을 효율적으로 처리할 수 있습니다.

- **Technical Details**: 제안된 모델은 두 개의 스트림을 가진 Transformer 모델을 사용하여 상호작용하는 Marmoset의 오디오 녹음을 처리합니다. 각 채널은 슬라이딩 스펙트럼 세그먼트를 분석하여 호출 유형과 발신자 정보로 분류됩니다. 이 모델은 세 가지 중요한 작업인 세분화, 분류 및 발신자 식별을 통합하여 수행합니다.

- **Performance Highlights**: F-score와 정확도를 사용하여 분류, 세분화 및 발신자 식별 성능을 평가하였으며, 이전 연구들에 비해 향상된 결과를 보였습니다. 본 연구는 Marmoset의 사회적 음성 상호작용의 복잡한 역학을 포착하는 데 중요한 기여를 합니다.



### SlowFast-VGen: Slow-Fast Learning for Action-Driven Long Video Generation (https://arxiv.org/abs/2410.23277)
- **What's New**: 본 논문에서는 SlowFast-VGen이라는 새로운 이중속도 학습 시스템을 소개합니다. 이는 행동 주도의 긴 비디오 생성을 위해 느린 학습과 빠른 학습을 통합하여, 일관되고 응답적인 비디오 생성을 가능하게 합니다.

- **Technical Details**: 이 시스템은 두 가지 주요 구성 요소로 이루어져 있습니다. 느린 학습(적용된 조건부 비디오 확산 모델)과 빠른 학습(Temporal LoRA 모듈 기반의 추론 시간 학습 전략)입니다. 빠른 학습 과정에서는 입력 및 출력에 기반하여 Temporal LoRA 매개변수를 업데이트하여 에피소드 메모리를 저장합니다. 또한, 느린 학습 알고리즘과 빠른 학습 루프를 결합하여 다중 에피소드 경험을 기억하여 기술 학습을 지원합니다.

- **Performance Highlights**: SlowFast-VGen은 FVD 점수에서 514를 기록하며 782을 초월하여 이전 모델들보다 뛰어난 성능을 보였습니다. 또한 0.37의 평균 장면 전환수로, 긴 비디오의 일관성을 유지하는 데 있어 우수함을 보여주었습니다.



### Multi-student Diffusion Distillation for Better One-step Generators (https://arxiv.org/abs/2410.23274)
Comments:
          Project page: this https URL

- **What's New**: Multi-Student Distillation (MSD)라는 새로운 프레임워크를 도입하여 기존의 단일 단계 확산 증류(diffusion distillation) 방법의 효율성을 개선했습니다.

- **Technical Details**: MSD는 조건부 교사 확산 모델을 여러 개의 단일 단계 생성기로 증류(dilute)하여 생성 품질을 향상시키는 방법입니다. 각 학생 생성기는 일부분의 조건 데이터를 처리하며, 빠른 추론 속도를 위한 소형 모델을 훈련시킵니다.

- **Performance Highlights**: 4개의 동등한 크기의 학생 모델을 사용하여, MSD는 ImageNet-64x64에서 1.20 FID, zero-shot COCO2014에서 8.20 FID라는 새로운 최첨단 성능을 달성했습니다.



### Proportional Fairness in Non-Centroid Clustering (https://arxiv.org/abs/2410.23273)
Comments:
          A preliminary version appeared at NeurIPS 2024

- **What's New**: 본 논문에서는 비 중심 클러스터링(non-centroid clustering)에서의 비례 공정성 보장(proportional fairness guarantees)을 확장하여 연구합니다. 중심 기반 클러스터링이 아닌 클러스터링의 손실 함수(loss function)를 다른 요인들을 고려하여 정의함으로써, 각 클러스터에 속한 에이전트(agent)들 간의 상호 작용을 반영하고자 합니다.

- **Technical Details**: 비례 공정성의 두 가지 기준인 core와 완전 정당화 표현(Fully Justified Representation, FJR)을 비 중심 클러스터링에 적응하여 적용했습니다. GreedyCapture 알고리즘을 비 중심 클러스터링에 적합시켜 근사화를 시도했으나, 이에 대한 효율성이 다소 떨어진다는 것을 보였습니다. 대신, 새로운 비효율적 알고리즘인 GreedyCohesiveClustering이 등장하며, 이는 임의의 손실 함수 아래에서 FJR을 정확하게 달성합니다.

- **Performance Highlights**: 실제 데이터 실험 결과에 따르면, 기존의 클러스터링 알고리즘들은 공정성이 매우 낮았으나, GreedyCapture 알고리즘은 훨씬 더 공정한 결과를 보였고, 기존 클러스터링 목표에서만 약간의 손실을 초래했습니다.



### A Monte Carlo Framework for Calibrated Uncertainty Estimation in Sequence Prediction (https://arxiv.org/abs/2410.23272)
- **What's New**: 이 논문은 고차원 입력 데이터(예: 이미지)로부터 분리된 시퀀스의 확률적 예측을 위한 몬테카를로(Monte Carlo) 프레임워크를 제안합니다. 특히 이 프레임워크는 확률 및 신뢰 구간(confidence interval) 추정을 가능하게 하여 예측의 불확실성을 정량화하기 위해 개발되었습니다.

- **Technical Details**: 제안된 foCus(framework for Monte Carlo uncertainty quantification of sequences)는 오토리그레시브(autoregressive) 모델을 사용하여 이미지 입력에 조건화된 시퀀스를 샘플링합니다. 이 과정에서 생성된 샘플들을 통해 각 상태가 발생할 확률을 추정하며, 훈련 시 열의 미스칼리브레이션(miscalibration) 문제를 개선하기 위해 시간 의존적인 정규화(time-dependent regularization) 방식을 도입했습니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 합성 데이터 및 실제 데이터에서 정확한 분별 예측(discriminative predictions)을 나타내었으나, 훈련된 오토리그레시브 시뮬레이터는 심각한 미스칼리브레이션 문제를 보였음을 확인했습니다. 이를 통해 시간 의존적인 정규화 방법이 더 나은 신뢰도 추정을 가능하게 하는 것을 보여주었습니다.



### TOMATO: Assessing Visual Temporal Reasoning Capabilities in Multimodal Foundation Models (https://arxiv.org/abs/2410.23266)
- **What's New**: 본 논문에서는 다중 모달 재단 모델(Multimodal Foundation Models, MFMs)의 비디오 이해에 대한 시각적 시간 추론 능력을 평가하기 위한 새로운 벤치마크, TOMATO(Temporal Reasoning Multimodal Evaluation)를 제안합니다.

- **Technical Details**: TOMATO는 (1) Multi-Frame Gain, (2) Frame Order Sensitivity, (3) Frame Information Disparity라는 세 가지 원칙과 해당 메트릭을 기반으로 하여 구성되었습니다. 이 벤치마크는 1,484개의 질문으로 구성되며, 1,417개의 비디오에 적용됩니다.

- **Performance Highlights**: 현재 MFMs의 성능 평가에서 인간 모델 성능 간의 격차는 57.3%로 나타났습니다. MFMs는 개별 프레임에서 이벤트를 정확하게 인식할 수 있지만, 이러한 프레임을 연속적인 시퀀스로 해석하는 데 실패하는 기본적인 한계가 드러났습니다.



### EMMA: End-to-End Multimodal Model for Autonomous Driving (https://arxiv.org/abs/2410.23262)
Comments:
          Blog post: this https URL

- **What's New**: EMMA(End-to-end Multimodal Model for Autonomous driving)는 모듈식 접근법 대신, 센서 데이터를 직접 처리하여 운전 관련 작업을 수행하는 새로운 모델입니다. 다양한 센서 데이터를 자연어 텍스트로 변환하여 통합적으로 처리할 수 있는 기능이 특징입니다.

- **Technical Details**: EMMA는 미리 훈련된 멀티모달 대형 언어 모델(Gemini) 위에 구축되었으며, 카메라 이미지 및 자연어 텍스트를 입력으로 받아 특정 운전 작업을 수행합니다. EMMA는 비전 모듈과 행동 모듈 간의 기호 인터페이스를 제거하여 각 운전 목표를 공동으로 최적화합니다.

- **Performance Highlights**: EMMA는 nuScenes 데이터셋에서 최첨단 성능을 달성하였으며, Waymo Open Motion Dataset에서도 경쟁력 있는 결과를 보였습니다. 또한, 3D 객체 탐지 및 도로 그래프 추정 등 여러 인식 작업에서도 뛰어난 성능을 보였습니다. 하지만, 이미지 프레임 수 처리의 제한, LiDAR와 같은 3D 감지 모드의 부재로 인한 문제도 존재합니다.



### Keypoint Abstraction using Large Models for Object-Relative Imitation Learning (https://arxiv.org/abs/2410.23254)
Comments:
          CoRL LangRob Workshop, 2024

- **What's New**: KALM(KP Abstraction using Large Models for Object-Relative Imitation Learning) 프레임워크는 대규모 사전 학습된 비전-언어 모델을 사용하여 자동으로 작업 관련 및 인스턴스 간 일관성이 있는 키포인트를 생성합니다.

- **Technical Details**: KALM은 대규모 사전 학습된 모델을 통해 키포인트 후보를 생성하고 소량의 로봇 시연 데이터를 기반으로 이를 검증하여 견고하고 일관된 키포인트를 도출합니다. 이 과정은 키포인트 중심의 정책 모델을 훈련하여 로봇이 다양한 물체 자세, 카메라 뷰 및 인스턴스에서 일반화할 수 있도록 합니다.

- **Performance Highlights**: KALM은 실제 환경에서 고도의 일반화를 보여주며, 소수의 시연 데이터만으로도 다양한 작업 및 환경에 적응하여 강력한 성능을 발휘합니다.



### EMOTION: Expressive Motion Sequence Generation for Humanoid Robots with In-Context Learning (https://arxiv.org/abs/2410.23234)
- **What's New**: 이 논문은 EMOTION이라 불리는 새로운 프레임워크를 소개하며, 이를 통해 인간형 로봇의 표정 있는 동작 시퀀스를 생성하여 비언어적 커뮤니케이션 능력을 향상시킵니다. 이 방법은 대형 언어 모델(LLMs)의 인컨텍스트 학습 기능을 활용하여 사회적으로 적절한 제스처 동작 시퀀스를 동적으로 생성하는데 중점을 둡니다.

- **Technical Details**: EMOTION 프레임워크는 사용자 언어 지시 및 로봇 이미지 관찰을 입력으로 받아, 행위 동작 시퀀스를 생성합니다. 본 연구에서는 총 10개의 서로 다른 표정 있는 제스처를 생성하고, EMOTION과 인간 피드백 버전인 EMOTION++의 생성 동작을 비교하는 온라인 사용자 연구를 수행했습니다. 여러 제스처에 대한 인식의 차이를 확인한 결과, 일부 제스처는 높은 평가를 받았지만, 다른 제스처는 상대적으로 낮은 평가를 받았습니다.

- **Performance Highlights**: EMOTION과 EMOTION++는 자연스러움과 이해 가능성에 있어 인간의 성능과 동등하거나 이를 초과하는 결과를 보여주며, 인간 피드백을 통합함으로써 생성된 제스처의 품질을 향상시킬 수 있는 가능성을 제시합니다. 결과적으로, LLMs와 VLMs를 통한 인간형 로봇의 표현력 향상에 대한 접근법이 효과적임을 강조하며, 향후 연구를 위한 디자인 의미를 제공합니다.



### Aligning Audio-Visual Joint Representations with an Agentic Workflow (https://arxiv.org/abs/2410.23230)
- **What's New**: 본 연구에서는 오디오-비주얼(Audio-Visual, AV) 데이터 간의 정렬 문제를 해결하기 위해 LLM 기반의 AVAgent를 제안합니다. AVAgent는 멀티모달 LLM을 사용하여 오디오와 비주얼 데이터를 언어 설명으로 변환하고, 이 데이터를 기반으로 정렬 여부를 판단하며 필요시 오디오 신호를 편집합니다.

- **Technical Details**: 제안된 방법에서 AVAgent는 도구 사용(tool use), 계획(planning), 반영(reflection) 단계를 순환적으로 수행하여 오디오 신호를 시각 데이터에 점진적으로 정렬합니다. 이 과정에서 AVAgent는 배경 잡음을 제거하고 데이터를 보강하는 전처리 작업을 수행하며, 이러한 작업 후 VLM(비전-언어 모델)을 통해 편집된 오디오 신호가 비디오와 잘 맞는지 평가합니다.

- **Performance Highlights**: Flick-SoundNet, VGG-Instruments 등 다양한 데이터 세트를 통해 실험한 결과, 제안된 방법이 기존 기준선에 비해 우수한 성능을 보였습니다. 특히, 선형 프로빙(linear probing), 미세 조정(fine-tuning) 분류, 비주얼 사운드 로컬라이제이션(sound localization) 등 다양한 다운스트림 작업에서 최첨단 성능을 입증했습니다.



### COMAL: A Convergent Meta-Algorithm for Aligning LLMs with General Preferences (https://arxiv.org/abs/2410.23223)
- **What's New**: 본 논문에서는 보상 모델을 사용하는 기존의 알고리즘의 한계를 극복하기 위해 Convergent Meta Alignment Algorithm (COMAL)을 제안합니다. 이는 게임 이론의 수렴적 알고리즘에 영감을 받아 개발되었습니다.

- **Technical Details**: COMAL은 두 플레이어 제로섬 게임을 모델링하여 Nash equilibrium 정책에 도달하는 메타 알고리즘입니다. COMAL은 ProxProxoman_Prox 연산자를 기본 빌딩 블록으로 사용하여 모든 정책에 대해 50% 이상의 승률을 보장하는 robust alignment를 달성합니다.

- **Performance Highlights**: COMAL은 다양한 기존 알고리즘과 비교했을 때 마지막 반복(iterate)에서 Nash equilibrium에 수렴하는 유일한 알고리즘임을 실험적으로 입증하였으며, DPO 및 다양한 반복 알고리즘에 비해 항상 50%를 초과하는 승률을 기록했습니다.



### Partial Channel Dependence with Channel Masks for Time Series Foundation Models (https://arxiv.org/abs/2410.23222)
Comments:
          NeurIPS Workshop on Time Series in the Age of Large Models, 2024. Oral presentation

- **What's New**: 본 논문은 채널 간의 의존성을 보다 정교하게 조정할 수 있는 부분 채널 의존성(partial channel dependence, PCD) 개념을 도입합니다. 이전 연구들은 데이터셋 간의 명시적 이질성에 초점을 맞추었으나, 본 연구에서는 암묵적 이질성에 주목합니다.

- **Technical Details**: 우리는 PCD를 달성하기 위해 두 가지 주요 구성 요소로 구성된 채널 마스크(channel mask)를 제안합니다: 1) 채널 간의 상대적 의존성을 인코딩하는 상관 행렬(correlation matrix), 2) 각 데이터셋에 특화된 절대 의존성을 학습하는 도메인 파라미터(domain parameters), 이를 통해 상관 행렬을 정제합니다.

- **Performance Highlights**: 본 연구는 예측(forecasting), 분류(classification), 결측치 보간(imputation), 이상 탐지(anomaly detection) 등의 4가지 시간 시계열(task)에서 PCD의 효과를 검증하며, 몇 개의 샷(few-shot) 및 제로샷(zero-shot) 시나리오를 포함한 다양한 설정에서 우수한 성능을 보였습니다.



### DiaMond: Dementia Diagnosis with Multi-Modal Vision Transformers Using MRI and PE (https://arxiv.org/abs/2410.23219)
Comments:
          Accepted by IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2025

- **What's New**: 이 논문은 DiaMond라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 MRI(자기공명영상)와 PET(양전자 방출 단층촬영) 데이터를 효과적으로 통합하기 위해 비전 트랜스포머(vision Transformers)를 사용합니다. DiaMond는 셀프 어텐션(self-attention)과 바이 어텐션(bi-attention) 메커니즘을 활용하여 두 가지 모달리티의 특징을 결합하며, 멀티모달 정규화(multi-modal normalization)을 통해 중복 의존성을 줄여 성능을 향상시킵니다.

- **Technical Details**: DiaMond는 MRI와 PET 데이터를 독립적으로 처리하는 두 개의 가지를 포함하는 구조입니다. 셀프 어텐션은 각각의 모달리티에서 고유한 특징을 추출하고, 바이 어텐션은 서로 겹치는 정보를 기반으로 모달리티 간의 상관 관계를 캡처합니다. RegBN이라는 멀티모달 정규화 기법이 적용되어 두 모달리티 간의 불필요한 의존성을 제거합니다. 이는 DiaMond가 보다 효율적으로 각 모달리티의 독특한 특징을 탐색할 수 있도록 합니다.

- **Performance Highlights**: DiaMond는 다양한 데이터셋에서 기존의 멀티모달 방법을 크게 초월하는 결과를 보였으며, AD(알츠하이머병) 진단에서 92.4%, AD-MCI-CN(경도 인지 장애) 분류에서 65.2%, AD와 FTD(전두측두엽 치매)의 구별 진단에서 76.5%의 균형 잡힌 정확도를 달성했습니다. 이 연구는 DiaMond의 강인성 또한 확인하였습니다.



### Grounding by Trying: LLMs with Reinforcement Learning-Enhanced Retrieva (https://arxiv.org/abs/2410.23214)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 환각(hallucination) 문제를 해결하기 위해 정보 검색 기능을 통합하고, 이를 통해 모델의 응답을 실제 출처에 기반하도록 하는 새로운 강화 학습 프레임워크인 LeReT(Learning to Retrieve by Trying)를 소개합니다.

- **Technical Details**: LeReT는 다양한 검색 쿼리를 생성하여 LLM이 정보 검색에서 더 효과적인 쿼리를 학습할 수 있도록 하는 접근 방식을 사용합니다. 이 프레임워크는 우선 다양한 검색 쿼리를 시도한 후, 이러한 쿼리 중에서 유용한 결과를 가져오는 데 성공한 쿼리에 대해 보상을 주어 쿼리 품질을 향상시키는 방법입니다. LeReT는 다단계 검색에 적합하게 설계되어 있습니다. 또한, LeReT는 여러 검색 도구에서 쉽게 적용될 수 있습니다.

- **Performance Highlights**: LeReT는 두 개의 질문-답변 데이터 세트에서 리트리벌(리트리버) 정확도를 최대 29% 향상시키고, 하위 생성 평가에서 17% 개선된 성능을 보여주며, 특히 강력한 생성 모델(GPT-4와 같은)에서 그 혜택이 더욱 두드러졌습니다. LeReT의 성능은 반복적으로 개선될 수 있으며, 이는 다양한 쿼리 샘플링이 효과적이라는 실험 결과로 뒷받침됩니다.



### Kinetix: Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks (https://arxiv.org/abs/2410.23208)
Comments:
          The first two authors contributed equally. Project page located at: this https URL

- **What's New**: 이번 논문에서는 Kinetix라는 새로운 물리 기반 RL 환경을 도입하여 강화 학습 에이전트를 훈련시키는 최신 접근법을 다룹니다. Kinetix는 다양한 2D 물리적 작업을 절차적으로 생성하여, 에이전트가 미지의 환경에서도 성공적으로 작업을 수행할 수 있도록 하는 데 초점을 맞추고 있습니다.

- **Technical Details**: Kinetix는 하드웨어 가속 물리 엔진인 Jax2D를 활용하여 수십억 단계의 환경 상호작용을 효과적으로 시뮬레이션합니다. 에이전트는 Kinetix의 환경을 샘플링하여 훈련을 진행하며, 이러한 환경은 로봇의 이동, 잡기 작업에서부터 비디오 게임, 전통적인 RL 환경까지 다양합니다.

- **Performance Highlights**: 훈련된 에이전트는 인간 설계의 미지의 환경을 제로샷(zero-shot)으로 해결할 수 있는 강력한 물리적 추론 능력을 보여주었습니다. 또한, 관심 있는 특정 작업에 대한 파인튜닝(Fine-tuning)을 통해 RL 에이전트를 처음부터 학습시키는 것과 비교하여 현저하게 향상된 성능을 발휘했습니다. 이는 기존 RL 훈련에서는 실패하는 작업도 포함됩니다.



### ReasoningRec: Bridging Personalized Recommendations and Human-Interpretable Explanations through LLM Reasoning (https://arxiv.org/abs/2410.23180)
Comments:
          Large Language Model, Recommendation, Human-Interpretable Reasoning, Personalization, Submitted for NAACL 2025

- **What's New**: 이 논문에서는 ReasoningRec라는 새로운 추천 프레임워크를 제안합니다. 이 프레임워크는 Large Language Model (LLM)을 활용하여 추천 및 인간이 해석할 수 있는 설명 사이의 간극을 메우는 데 중점을 두고 있습니다.

- **Technical Details**: ReasoningRec은 사용자의 선호와 반감을 모델링하기 위해 LLM을 사용하며, 사용자가 특정 아이템을 좋아할 이유에 대한 합성 설명을 생성합니다. 이러한 설명은 더 작은 LLM을 미세 조정하는 데 사용되어 추천 정확성을 높이고, 인간이 이해할 수 있는 설명을 제공합니다.

- **Performance Highlights**: ReasoningRec은 추천 예측에서 기존의 최첨단 기술들을 12.5%까지 초과하는 성능을 보였으며,追加로 인간이 이해할 수 있는 설명을 제공합니다.



### SciPIP: An LLM-based Scientific Paper Idea Proposer (https://arxiv.org/abs/2410.23166)
Comments:
          25 pages, 5 figures, 19 tables

- **What's New**: 이 논문은 과학 논문 아이디어 제안기인 SciPIP를 제안합니다. SciPIP는 사용자 제공 연구 배경을 바탕으로 유용한 논문을 검색하고, 이를 바탕으로 더 새롭고 실행 가능한 아이디어를 생성하는 방식으로 기존의 대형 언어 모델(LLM)의 잠재력을 활용합니다.

- **Technical Details**: SciPIP는 사용자가 제공한 연구 배경을 기반으로 문헌을 검색하여 아이디어를 제안하는 시스템입니다. 이를 위해 문헌 검색 데이터베이스를 구축하고, 의미론(semantic), 엔티티(entity), 인용의 공동 출현(citation co-occurrence)을 기반으로 문헌 검색을 수행합니다. 이후에는 문헌에서 얻은 정보를 활용하여 솔루션을 유추하거나 독창적인 아이디어를 생성하는 두 가지 경로를 통해 아이디어를 제안합니다.

- **Performance Highlights**: NLP 분야에서 진행된 광범위한 실험을 통해 SciPIP는 기존의 상위 회의 논문들과 유사한 인용을 검색하고, 많은 아이디어를 생성함으로써 그 효과를 입증하였습니다. 또한 SciPIP에 의해 생성된 아이디어는 청사진의 참조를 유지하면서도 혁신성과 실행 가능성을 확보하고 있음을 평가해 보여줍니다.



### FlexTSF: A Universal Forecasting Model for Time Series with Variable Regularities (https://arxiv.org/abs/2410.23160)
- **What's New**: 본 논문에서는 시간 시계열 예측을 위한 새로운 통합 모델인 FlexTSF를 제안합니다. FlexTSF는 다양한 도메인과 구조적 다양성을 효과적으로 처리할 수 있도록 설계되었습니다.

- **Technical Details**: FlexTSF는 세 가지 주요 혁신 설계를 포함합니다: VT-Norm(정규화 전략), IVP Patcher(패칭 모듈) 및 LED Attention(주의 메커니즘). VT-Norm은 서로 다른 특성을 가진 데이터를 표준화하여 동적 시간 패턴을 학습하는 데 중점을 두고, IVP Patcher는 다양한 구조적 시간 시리즈를 위한 연속적 패칭을 구현합니다. 마지막으로, LED Attention은 예측을 위한 자동회귀 프로세스에서 이러한 요소들을 통합하여 도메인과 시간 정보에 대한 인식을 제공합니다.

- **Performance Highlights**: FlexTSF는 12개의 다양한 데이터셋에서 기존의 최첨단 예측 모델보다 우수한 성능을 보였으며, 자체 지도 학습(pre-training) 후 제로샷(zero-shot) 및 몇 샷(few-shot) 설정에서도 뛰어난 예측력을 보여줍니다.



### Fourier Amplitude and Correlation Loss: Beyond Using L2 Loss for Skillful Precipitation Nowcasting (https://arxiv.org/abs/2410.23159)
Comments:
          Accepted by NeurIPS 2024. Camera-ready submission

- **What's New**: 본 연구에서는 기존의 모델 아키텍처에 의존하지 않고, 새로운 손실 함수인 Fourier Amplitude and Correlation Loss (FACL)을 제안합니다. 이 손실 함수는 예측의 고주파 패턴을 복원하는 데 중점을 두며, 전통적인 MSE 손실을 대신하여 더 선명한 강수 예측을 가능하게 합니다.

- **Technical Details**: FACL은 두 가지 주요 손실 항목으로 구성됩니다: Fourier Amplitude Loss (FAL)와 Fourier Correlation Loss (FCL). FAL은 예측의 푸리에 진폭을 정규화하고, FCL은 누락된 위상 정보를 보완합니다. 이러한 손실 항목의 결합은 명암도와 공간 구조를 더욱 향상시키며, FAL과 FCL 사이의 학습 메커니즘을 통해 점진적으로 훈련됩니다.

- **Performance Highlights**: 실험 결과, FACL은 MSE로 기존 예측보다 더 사실적이고 기술적인 성능이 우수한 결과를 보여주었습니다. 또한, 새로운 평가 지표인 Regional Histogram Divergence (RHD)를 통해 이미지 패턴 간의 유사성을 더욱 정량적으로 측정할 수 있게 되었습니다. 이러한 개선은 강수 예측의 정확성을 높이는 데 기여하고 있습니다.



### The Good, the Bad, and the Ugly: The Role of AI Quality Disclosure in Lie Detection (https://arxiv.org/abs/2410.23143)
Comments:
          Order of the authors are in alphabetical order of their last names. All authors contributed equally. The manuscript is under review. 74 Pages, including appendices and references

- **What's New**: 본 논문은 낮은 품질의 AI 어드바이저가 품질 공시 없이 어떻게 텍스트 기반 거짓말을 퍼뜨릴 수 있는지를 조사합니다. 실험에 참여한 참가자들은 게임 쇼의 전사를 평가하며 진실과 거짓을 구별하는 작업을 수행했으며, 낮은 품질의 어드바이저를 의존할 때 진실 감지 능력이 개인의 능력 아래로 떨어지는 경향을 발견했습니다. 반면, 품질 높은 어드바이저는 공시 여부와 관계없이 진실 감지를 향상시킵니다.

- **Technical Details**: 우리는 AI 어드바이저의 품질을 여러 수준(낮음, 보통, 높음)으로 설정하고, 참가자들은 AI의 효과성이 공개된 환경과 비공개된 환경에서 각각 진실을 감지하도록 실험을 진행했습니다. 본 연구는 AI 어드바이저의 품질 스펙트럼과 그 효과성의 (비)공식 공시가 참가자의 AI 의존도에 어떻게 영향을 미치는지를 조사합니다.

- **Performance Highlights**: 연구 결과, 낮은 품질의 AI 어드바이저에 대한 의존은 참가자의 진실 감지 능력을 저하시켰으며, 이로 인해 거짓 정보의 확산이 우려됩니다. 반면, 높은 품질의 AI 어드바이저는 참가자들이 진실을 감지할 수 있는 능력을 전반적으로 향상시켰습니다.



### Fair Division with Market Values (https://arxiv.org/abs/2410.23137)
- **What's New**: 본 논문에서는 주관적 평가와 시장 가치를 함께 고려하는 공정 분배 모델을 제안합니다. 이 모델은 비분할 가능한 상품을 여러 에이전트에게 배분하는 문제를 다루고 있으며, 제품마다 시장 가치를 명시합니다.

- **Technical Details**: 모델은 각각의 에이전트가 개인적인 주관적 효용 함수(u_i)를 제출하고, 이를 기반으로 공정성을 평가합니다. 시장 가치는 모든 에이전트에게 동일하게 적용되는 추가적인 가치 평가(v)를 제공합니다. 여기서 제시된 공정성 개념 중 하나인 SD-EF1(확률적 우세 불만 없음)은 주관적 평가와 시장 가치 모두에 기반한 배분을 요구합니다.

- **Performance Highlights**: 연구 결과, SD-EF1을 충족하는 배분이 항상 존재하지는 않지만, 주관적 평가에 대해 EF1을 보장하면서 시장 가치에 대해 SD-EF1을 만족하는 배분은 보장할 수 있습니다. 또한, Pareto 최적성, EFX, MMS와 같은 다른 보장 조건도 연구하였으며, 비가산 가치 평가 및 케이크 자르기 문제로의 모델 확장도 논의되었습니다.



### Revisiting MAE pre-training for 3D medical image segmentation (https://arxiv.org/abs/2410.23132)
Comments:
          Arxiv Preprint. Currently under Review

- **What's New**: 이 논문에서는 Self-Supervised Learning (SSL) 접근법을 활용하여 44,000개의 3D 뇌 MRI 데이터셋을 기반으로 기존 SSL 방법보다 성능 향상을 이루었습니다. 특히, Residual Encoder U-Net 아키텍처를 사용하여 3D 의료 이미지 분석의 한계를 극복하고자 했습니다.

- **Technical Details**: 제안된 모델은 Masked Auto Encoders (MAEs) 개념을 3D CNN에 최적화하여 구성되었습니다. 실험은 5개의 개발 및 8개의 테스트 뇌 MRI 분할 데이터셋에서 진행되었습니다. 논문에서 다룬 주요 기술적 세부사항으로는 z-score 정규화, poly-learning rate 조정을 통한 SGD 최적화 방법 등이 있습니다.

- **Performance Highlights**: 제안된 모델은 기존 nnU-Net 베이스라인보다 평균 3 Dice 포인트 향상된 성능을 보였으며, 7개의 방법 중 평균 순위 2를 기록하여 뛰어난 안정성을 자랑합니다.



### Provably Optimal Memory Capacity for Modern Hopfield Models: Transformer-Compatible Dense Associative Memories as Spherical Codes (https://arxiv.org/abs/2410.23126)
Comments:
          Accepted at NeurIPS 2024

- **What's New**: 이번 연구에서는 현대 Hopfield 모델과 Kernelized Hopfield Models (KHMs)의 최적 메모리 용량을 분석하고, 정보 이론의 구형 코드와 KHMs의 메모리 구성 간의 연결을 수립합니다. 이를 통해 메모리 문제를 초구 위의 점 배열 문제로 변환하는 새로운 관점을 제공합니다.

- **Technical Details**: KHMs에서는 메모리를 최적의 구형 코드로 형성할 수 있는 특징 공간이 필요하며, 이를 통해 KHMs의 최적 메모리 용량을 분석합니다. 우리는 KHMs의 메모리 용량에 대한 상한선 및 하한선을 수립하고, 메모리 개수에 비례하는 피처 차원의 크기 변화에 대한 분석을 수행합니다.

- **Performance Highlights**: 실험적으로 제공된 수치 결과는 KHMs의 조회 능력 향상 및 관련 transformer의 표현 학습 개선을 뒷받침합니다. 또한, KHMs의 최적 용량에 도달하기 위한 서브-선형 시간 알고리즘 $	exttt{U-Hop}^+$을 제안합니다.



### Teaching a Language Model to Distinguish Between Similar Details using a Small Adversarial Training S (https://arxiv.org/abs/2410.23118)
- **What's New**: 이 논문에서는 언어 모델이 자연어 작업(Natural Language Tasks)에서 높은 정확도를 달성할 수 있지만, 수동으로 생성된 적대적 사례(Adversarial Examples)에서는 성능이 떨어진다는 점을 다룹니다.

- **Technical Details**: 저자들은 Stanford Natural Language Inference (SNLI) 데이터 세트에서 훈련된 언어 모델의 성능을 수동으로 생성된 적대적 테스트 세트에서 조사하였습니다. 이 모델은 비슷한 단어와 구문을 구별하도록 돕기 위해 소규모의 수동으로 생성된 적대적 훈련 세트로 미세 조정(Fine Tuning) 되었습니다.

- **Performance Highlights**: 적대적 테스트 세트에서 정확도가 13% 증가하였으며, 원래 NLI 작업에서는 여전히 좋은 성능을 유지하였습니다. 또한, SNLI 테스트 세트에서 가장 유사한 모순(Cosine Similarity 기준)에서 정확도가 91.2%에서 92.9%로 증가한 것을 보여주었습니다.



### Unified Triplet-Level Hallucination Evaluation for Large Vision-Language Models (https://arxiv.org/abs/2410.23114)
Comments:
          18 pages, 8 figures

- **What's New**: 본 논문은 대형 비전-언어 모델(LVLM)에서 발생하는 객체 및 관계 환각(hallucination)을 동시에 평가하기 위한 통합 프레임워크를 설계하였습니다.

- **Technical Details**: LVLM의 응답에서 추출된 (객체, 관계, 객체) 트리플렛(triplet) 기반의 환각 평가를 통해 환각 유형을 통합적으로 분석할 수 있으며, Tri-HE라는 새로운 평가 벤치마크를 도입하여 LVLM의 환각 문제를 보다 세밀하게 평가할 수 있습니다.

- **Performance Highlights**: 이 연구에서 제안한 방법은 기존 LVLM이 갖고 있는 관계 환각 문제를 해결함으로써, LLaVA-1.5 모델이 모든 오픈 소스 모델을 초월하는 성능을 기록하였으며, 강력한 GPT-4V와 동등한 성능을 보여주었습니다.



### Why Gradient Subspace? Identifying and Mitigating LoRA's Bottlenecks in Federated Fine-Tuning of Large Language Models (https://arxiv.org/abs/2410.23111)
Comments:
          24 pages, 10 figures, pre-print

- **What's New**: 이 논문은 Federated Learning (FL) 환경에서 Large Language Models (LLMs)의 성능을 향상시킬 수 있는 새로운 전략을 제안합니다. 특히, Low-Rank Adaptation (LoRA) 기반의 접근 방식이 가진 한계를 분석하고, 대안으로 직접적인 가중치 평균화 전략이 보다 나은 결과를 보여줄 수 있음을 밝힙니다.

- **Technical Details**: FL에서 LLMs의 파라미터 효율적인 미세 조정 방법으로 LoRA가 널리 사용되고 있으나, 이 접근 방식은 저차원 행렬에 대한 제한된 서브스페이스 학습 때문에 최적이 아니며, 실험적으로 GaLore와 같은 그라디언트 저차원 최적화 방법이 더 효과적임을 입증합니다.

- **Performance Highlights**: 직접적인 가중치 집합을 사용하는 전략이 FL 환경에서 LoRA 기반 전략보다 우수한 성능을 보이며, GaLore는 FlexLoRA 및 FFA-LoRA와 같은 최신 LoRA 방법을 능가하는 성과를 보여줍니다. 이를 통해 FL에서의 LLM 미세 조정 성능을 극대화하는 새로운 혁신적 접근 방식을 제안합니다.



### Controllable Game Level Generation: Assessing the Effect of Negative Examples in GAN Models (https://arxiv.org/abs/2410.23108)
- **What's New**: 이번 연구에서는 CGAN(Conditional Generative Adversarial Networks)과 Rumi-GAN 두 가지 제어 가능한 GAN 변형 모델을 비교하고, 게임 레벨 생성 중의 특정 제약 조건인 '플레이 가능성(playability)' 및 '제어 가능성(controllability)'의 충족 여부에 대한 성능을 평가했습니다. 특히, Rumi-GAN은 GAN 훈련시 '부정적 예제(negative examples)'를 활용하여 긍정적 예제를 효과적으로 학습하는 새로운 접근 방식을 제시했습니다.

- **Technical Details**: 이 연구는 'Deep Convolutional GAN' 아키텍처를 사용하여 훈련된 CGAN과 Rumi-GAN 모델을 기반으로 합니다. 훈련 과정에서는 각각의 모델이 긍정적 및 부정적 예제를 사용하여 특정 게임 레벨의 제약 조건을 적용할 수 있도록 하였습니다. Rumi-GAN의 경우, 손실 함수가 특정 조건을 충족하는 레벨 생성(segment generation)을 촉진하고, 조건을 충족하지 않는 레벨 생성을 억제합니다.

- **Performance Highlights**: 부정적 예제를 포함한 훈련이 GAN 모델의 특정 제약 조건 충족, 특히 플레이 가능성에서 긍정적인 영향을 미친다는 결과를 도출했습니다. 이 연구는 다양한 제어 가능한 GAN 모델의 비교 분석을 통해 게임 레벨 생성에 있어 긍정적 예제와 부정적 예제의 통합의 효과를 최초로 보여주었습니다.



### Decoupling Semantic Similarity from Spatial Alignment for Neural Networks (https://arxiv.org/abs/2410.23107)
Comments:
          Accepted at NeurIPS2024

- **What's New**: 본 논문에서는 Representational Similarity Matrices (RSMs)의 기존 계산 방법의 한계를 제시하고, 공간적 위치에 영향을 받지 않는 semantic RSMs를 제안하여 이미지 응답 간의 유사성을 측정합니다.

- **Technical Details**: 우리는 semantic RSMs를 제안하여 공간적 순열에 불변이며, 집합 매칭 문제로 형성된 semantic 유사성을 측정합니다. 이 방법은 CNN과 ViT의 이미지 응답을 비교하여, 두 모델의 유사성 구조를 파악합니다.

- **Performance Highlights**: 제안한 semantic RSMs는 spatio-semantic RSMs에 비해 이미지 검색 성능을 향상시키고, 분류기 표현 간의 유사성을 더 잘 반영합니다. 또한 컴퓨팅 복잡성을 줄이기 위한 근사화 방법을 소개합니다.



### Comparative Analysis of Demonstration Selection Algorithms for LLM In-Context Learning (https://arxiv.org/abs/2410.23099)
Comments:
          6 pages, 4 figures

- **What's New**: 이 논문은 다양한 demonstration selection 알고리즘을 분석하고, 실험을 통해 이들 알고리즘의 효율성과 효과를 평가합니다. 특히, 무작위 선택이 특정 상황에서 오히려 나은 결과를 초래할 수 있음을 보여줍니다.

- **Technical Details**: 연구에서는 여섯 가지 demonstration selection 알고리즘(CBDS, RD-direct, RD-channel, LLM Retriever, UPRISE, OpenICL TopK)을 비교하였습니다. 각 알고리즘은 특정 전략을 사용하여 LLM의 성능을 개선하는 데 중점을 두었으며, 이 과정에서 Bayesian 접근법, 순차적 예제 검색, 교차 인과 모델 등을 활용했습니다.

- **Performance Highlights**: 경험적인 연구 결과, 알고리즘 간 성능 차이가 크며, 같은 데이터셋에서도 정확도 차이가 45%까지 발생할 수 있습니다. 또한, 시연의 수 증가가 항상 향상된 성능으로 이어지지 않으며, 정확성과 처리 효율성 간에는 트레이드오프가 존재한다는 점을 발견했습니다.



### From Hype to Reality: The Road Ahead of Deploying DRL in 6G Networks (https://arxiv.org/abs/2410.23086)
- **What's New**: 본 연구에서는 6G 애플리케이션의 요구를 충족하기 위해 Deep Reinforcement Learning (DRL)의 변혁적 잠재력을 강조하고 있습니다. DRL은 전통적인 머신 러닝 솔루션과 비교하여 6G의 데이터 처리 요구사항을 효과적으로 해결할 수 있는 방법을 제시합니다.

- **Technical Details**: 이 논문은 DRL을 활용한 세 가지 응용 프로그램을 통해 무선 접근 제어(wireless access control), 베이스밴드 함수 배치(baseband function placement), 네트워크 슬라이싱 조정(network slicing coordination) 등을 다룹니다. DRL은 MDP(Markov Decision Processes)로 네트워크 관리를 수식하며, 심층 신경망(deep neural networks) 원리를 결합하여 높은 차원의 입력 공간을 처리할 수 있습니다.

- **Performance Highlights**: 테스트베드에서 네트워크 슬라이스를 관리하기 위한 실제 DRL 배포를 통해 서비스 지연 시간과 에너지 소비를 최소화하는 데 효과적임을 입증하였습니다. 그러나 DRL 기반 네트워크 관리 솔루션이 성숙되지 않아서 실제 네트워크 응용에서 발생할 수 있는 여러 도전 과제가 존재합니다.



### S3PT: Scene Semantics and Structure Guided Clustering to Boost Self-Supervised Pre-Training for Autonomous Driving (https://arxiv.org/abs/2410.23085)
Comments:
          Accepted for WACV 2025

- **What's New**: S3PT (Scene Semantics and Structure guided Pre-Training)는 자율 주행에 적합한 클러스터링 기반의 자가 지도 전처리 기법으로, 다양한 객체 크기와 클래스 불균형 문제를 해결하는 데 중점을 두고 제안되었습니다.

- **Technical Details**: 이 방법은 세 가지 주요 구성 요소를 포함합니다: 1) 희귀 클래스(예: 오토바이, 동물)의 표현 개선을 위한 의미 분포 일관적 클러스터링; 2) 객체 크기의 불균형을 처리하기 위한 객체 다양성 일관적 공간 클러스터링; 3) 장면의 기하학적 정보에 기초한 깊이 유도 공간 클러스터링.

- **Performance Highlights**: S3PT는 nuScenes, nuImages, Cityscapes 데이터셋을 통한 2D 의미 세분화 및 3D 객체 탐지 작업에서 성능 향상을 보여주며, 자율 주행 데이터에서의 사전학습을 통한 일반화 능력이 강화되었습니다.



### An Event-Based Digital Compute-In-Memory Accelerator with Flexible Operand Resolution and Layer-Wise Weight/Output Stationarity (https://arxiv.org/abs/2410.23082)
Comments:
          5 pages, 7 figures, submitted to IEEE ISCAS 2025

- **What's New**: 본 연구에서는 임의의 피연산자 해상도와 형태를 지원하는 새로운 디지털 CIM 매크로인 FlexSpIM을 제안합니다. 이 매크로는 가중치(weights)와 막전위(membrane potential)를 위한 통합 CIM 스토리지 시스템을 기반으로 하여, SNN 실행 중 데이터 이동 비용을 최소화하는 하이브리드 데이터 흐름을 가능하게 합니다.

- **Technical Details**: FlexSpIM의 주요 기술적 성과는 다음과 같습니다. 첫째, 가중치와 막전위의 완전 재구성이 가능하여 정확도, 에너지 효율성 및 메모리 공간 간의 균형을 탐색할 수 있습니다. 둘째, 재구성 가능한 피연산자 형태를 제공하여 다양한 해상도 값을 지원하며, 셋째, 가중치와 막전위에 대한 통합 메모리를 통해 하이브리드 스테이셔너리 데이터 흐름을 구현하여 데이터 이동 효율성을 증대시킵니다.

- **Performance Highlights**: FlexSpIM의 프로토타입은 40nm CMOS 기술로 제작되었으며, 이전의 고정정밀 디지털 CIM-SNN에 비해 비트 정규화 에너지 효율성이 2배 향상되었습니다. 대규모 시스템에서는 최대 90% 에너지를 절약할 수 있으며, IBM DVS 제스처 데이터셋에서 95.8%의 최첨단 분류 정확도를 달성합니다.



### BUZZ: Beehive-structured Sparse KV Cache with Segmented Heavy Hitters for Efficient LLM Inferenc (https://arxiv.org/abs/2410.23079)
- **What's New**: BUZZ는 기존의 KV 캐시 메커니즘의 단점을 극복하기 위해 제안된 새로운 알고리즘으로, 구조화된 맥락 정보를 이용해 캐시 메모리 사용량을 최소화하고 추론 속도를 향상시킵니다.

- **Technical Details**: BUZZ 알고리즘은 벌집 구조의 희소 캐시를 사용하며, 슬라이딩 윈도우를 통해 최근 정보를 포착하고 역사적 토큰을 청크로 동적으로 분할하여 중요한 토큰에 우선순위를 부여합니다. 이 알고리즘은 O(n)의 시간 복잡도를 가지며, 인간의 기억 패턴을 모방하여 중요한 정보를 보다 효과적으로 유지합니다.

- **Performance Highlights**: BUZZ는 LLM 추론 시 캐시 메모리 사용량을 2.5배 줄이고, 긴 텍스트 요약에서 99% 이상의 정확도를 유지하며, 다중 문서 질문 응답에서 메모리 한계 내에서 7.69% 높은 성능을 보였습니다.



### CNN Explainability with Multivector Tucker Saliency Maps for Self-Supervised Models (https://arxiv.org/abs/2410.23072)
Comments:
          29 pages, 20 figures

- **What's New**: 이번 연구에서는 Convolutional Neural Networks (CNNs)의 해석 가능성에 대한 새로운 접근법인 Tucker Saliency Map (TSM) 방법을 소개합니다. 이는 기존의 EigenCAM과 달리, feature maps의 고유한 구조를 더 잘 포착하여 더 정확한 saliency maps를 생성합니다.

- **Technical Details**: Tucker tensor decomposition을 적용하여 singular vectors와 values를 생성하며, 이를 통해 고화질의 saliency maps를 생성합니다. 또한, EigenCAM과 TSM을 확장한 Multivec-EigenCAM과 Multivector Tucker Saliency Maps (MTSM)를 도입하여 모든 singular vectors와 values를 활용합니다.

- **Performance Highlights**: 정량적 평가 결과, TSM, Multivec-EigenCAM, 및 MTSM은 label-dependent 방법들과 경쟁력 있는 성능을 보였으며, TSM은 EigenCAM에 비해 explainability를 약 50% 향상시켰고, MTSM은 self-supervised 모델에서 최고의 결과를 달성했습니다.



### LLMs Integration in Software Engineering Team Projects: Roles, Impact, and a Pedagogical Design Space for AI Tools in Computing Education (https://arxiv.org/abs/2410.23069)
- **What's New**: 이번 연구는 생성적 AI (Generative AI, GenAI) 모델 및 도구, 예를 들어 ChatGPT와 GitHub Copilot이 2학년 학부 소프트웨어 공학 팀 프로젝트에 미치는 영향을 탐구합니다. 설문조사와 인터뷰를 통해 학생들의 코딩 경험, 학습 및 자기 효능감에 대한 인상을 공유합니다.

- **Technical Details**: 39명의 학생들을 대상으로 진행된 설문조사와 8명의 학생들과의 인터뷰에서 얻은 질적 데이터는 GenAI의 활용이 팀워크, 팀 효능감 및 팀 역학에 미치는 역할과 함의에 대한 이해의 간극을 해소합니다. 연구에서는 학습과 교육학의 관점을 통해 데이터를 분석했습니다.

- **Performance Highlights**: GenAI 기반 프로그래밍 학습 도구를 위한 초기 디자인 공간을 제안합니다. 이는 GenAI가 학습 과정에서 어떤 역할을 하는지, 각 역할에 적용할 수 있는 다양한 지원 패턴, 그리고 교육자들뿐만 아니라 팀원 및 학생들 간의 GenAI 투명성을 지원하는 것의 중요성을 강조합니다.



### Controlling Language and Diffusion Models by Transporting Activations (https://arxiv.org/abs/2410.23054)
- **What's New**: 이 논문에서는 Activation Transport (AcT)라는 새로운 프레임워크를 소개하며, 이는 Optimal Transport(OT) 이론에 기반하여 모델의 활성화를 유도할 수 있는 방법론입니다. 이 기법은 기존의 활성화 조정 방법들을 통합적으로 설명하며, 대조군 언어 또는 서로 다른 스타일의 텍스트에서의 변환을 효과적으로 수행할 수 있습니다.

- **Technical Details**: AcT는 모달리티에 구애받지 않으며, 모델의 내부 활성화 분포를 보존하면서도 미세한 조정을 가능하게 합니다. 이 방법은 λ (lambda)라는 강도 매개변수를 이용해 개입 정도를 조절할 수 있으며, 0에서 1 사이의 값으로 설정하여 부분적 또는 전체적 변환을 적용합니다. 특히, 저자는 Linear-AcT 방식이 기존의 개입 방법들과 비교하여 더 나은 성과를 낼 수 있음을 실험적으로 입증했습니다.

- **Performance Highlights**: AcT는 Large Language Models (LLMs)에서 독성 감소, 개념 유도, 진실성 증가를 효과적으로 범위 내에서 수행하며, Text-to-Image (T2I) 모델에서도 미세한 스타일 조정 및 개념 부정이 가능함을 보여줍니다. 이 연구는 LLMs와 확산 모델에서 동시에 효과적인 개입 방법을 적용한 최초의 사례로 자리잡고 있습니다.



### Offline Reinforcement Learning and Sequence Modeling for Downlink Link Adaptation (https://arxiv.org/abs/2410.23031)
- **What's New**: 본 논문에서는 전통적인 링크 적응(link adaption, LA) 알고리즘의 복잡성과 비효율성을 해결하기 위해 오프라인 강화학습(offline reinforcement learning, RL)을 활용하는 새로운 접근법을 제안합니다.

- **Technical Details**: 연구는 배치 제약 깊이 Q-러닝(batch-constrained deep Q-learning), 보수적인 Q-러닝(conservative Q-learning), 그리고 결정 변환기(decision transformers)를 기반으로 한 세 가지 LA 설계를 제안하며, 이는 온라인 RL 방법의 성과에 필적하는 성능을 보여줍니다. 오프라인 RL의 데이터 수집은 최소한의 간섭으로도 가능하여 네트워크 운영에 미치는 영향을 줄입니다.

- **Performance Highlights**: 오프라인 RL 알고리즘은 적절한 행동 정책(behavioral policy)으로 수집된 데이터를 바탕으로 최신 온라인 RL 방법과 동등한 성능을 도출할 수 있음을 입증하였습니다.



### Online Intrinsic Rewards for Decision Making Agents from Large Language Model Feedback (https://arxiv.org/abs/2410.23022)
- **What's New**: 이번 연구에서는 ONI라는 분산 아키텍처를 제안하여 강화 학습 (RL) 정책과 내재적 보상 함수를 동시에 학습하는 방식으로, 기존의 내재적 보상 설계의 한계를 극복합니다.

- **Technical Details**: ONI는 비동기 LLM 서버를 통해 에이전트의 경험을 주석 처리하고, 이러한 피드백을 토대로 내재적 보상 모델을 증류합니다. 다양한 보상 모델링 방법 (해싱, 분류, 순위 모델)을 탐색하여 내재적 보상 설계에 대한 통찰을 제공합니다.

- **Performance Highlights**: ONI는 NetHack Learning Environment의 도전적인 희소 보상 작업에서 최첨단 성능을 달성했으며, 외부 데이터셋이나 소스 코드 없이 에이전트가 수집한 경험만을 사용하여 학습하게 됩니다.



### A Comparison of Prompt Engineering Techniques for Task Planning and Execution in Service Robotics (https://arxiv.org/abs/2410.22997)
Comments:
          6 pages, 3 figures, 2 tables, to be published in the 2024 IEEE-RAS International Conference on Humanoid Robots, We make our code, including all prompts, available at this https URL

- **What's New**: 이 논문은 LLM(대형 언어 모델)을 활용하여 서비스 로봇에서 고급 과제 계획 및 실행을 수행하는 방법을 연구합니다. 특히, 다양한 프롬프트 엔지니어링 기법을 비교하고 조합하여 작업 수행의 정확성 및 실행 시간을 측정합니다.

- **Technical Details**: 이 연구에서는 로봇의 기능을 자연어로 정의된 일련의 동작 시퀀스에서 샘플링하여 제로샷(Zero-shot) 방식으로 작업을 실행할 수 있는 방법을 제시합니다. 또한, 서비스 로봇의 긴 수명의 과제를 수행하기 위한 LLM과 다양한 프롬프트 엔지니어링 기법 통합의 가능성을 평가합니다.

- **Performance Highlights**: 이 논문의 결과는 다양한 과련 작업에서 고려된 프롬프트 엔지니어링 기법이 작업 완료 정확도 및 실행 시간에 미치는 영향을 체계적으로 분석하였으며, 실제 로봇 환경을 시뮬레이션하여 이를 진행합니다.



### VisAidMath: Benchmarking Visual-Aided Mathematical Reasoning (https://arxiv.org/abs/2410.22995)
Comments:
          58 pages, 28 figures

- **What's New**: 새로운 연구에서는 시각적 정보를 활용한 수학 문제 해결(Visual Aided Mathematical Problem Solving, MPS) 과정을 평가하기 위한 VisAidMath 벤치마크를 소개합니다. 이 벤치마크는 1,200개의 난이도 있는 문제를 포함하며, 다양한 출처에서 수집된 문제와 답변을 평가합니다.

- **Technical Details**: VisAidMath 벤치마크는 수학적 질문을 시각적 맥락(Visual Context), 질문(Question), 시각적 도움(Visual Aids), 답변(Answer) 네 부분으로 나누어 설계되었습니다. 이 벤치마크는 명확한 시각적 정보를 포함하고 있으며, 값은 LaTeX 형식으로 정리되어 있습니다. 이 연구에서는 통계적으로 10개의 주요 LLMs 및 LMMs의 성능을 분석하였습니다.

- **Performance Highlights**: 주요 모델인 GPT-4V는 시각적 정보를 활용한 추론 과제에서 평균 45.33%의 정확도를 보였으며, 이는 전문적인 시각적 도움을 제공받았을 때도 2점 감소하였습니다. 또한, SOTA 모델들은 약 50%의 평균 정확도에 그쳤으며, 생성된 시각적 보조 자료는 5%의 n-gram 유사성만을 보였습니다.



### Higher-order Cross-structural Embedding Model for Time Series Analysis (https://arxiv.org/abs/2410.22984)
- **What's New**: 이번 논문에서는 Higher-order Cross-structural Embedding Model for Time Series (High-TS)라는 새로운 프레임워크를 제안합니다. 이 모델은 멀티스케일 Transformer와 Topological Deep Learning (TDL)을 결합하여 시계열 데이터의 공간적 및 시간적 의존성을 동시에 모델링할 수 있습니다. 또한, 대조학습(contrastive learning)을 활용하여 두 가지 구조를 통합함으로써 강력하고 차별화된 표현을 생성합니다.

- **Technical Details**: High-TS는 두 가지 주요 구성 요소로 이루어져 있습니다. 첫 번째는 시간 차원의 멀티스케일 임베딩 모듈로, 이 모듈은 멀티스케일 주의 메커니즘을采用하여 시간 차원에서의 표현을 학습합니다. 두 번째는 공간 차원의 단순 복합체 임베딩 모듈로, 이 모듈은 TDL을 사용하여 서로 다른 단순 복합체 간의 고차원 상호작용을 구성하고 새로운 특성 표현을 학습합니다.

- **Performance Highlights**: High-TS는 다양한 시계열 작업에서 최첨단 방법들보다 우수한 성능을 보이며, 모델 성능을 향상시키는 데 있어 고차원 교차 구조 정보의 중요성을 입증합니다. 대규모 실험 결과, 모델이 복잡한 상호작용을 효과적으로 모델링하고 잡아내는 능력을 갖추고 있음을 확인하였습니다.



### PDSR: Efficient UAV Deployment for Swift and Accurate Post-Disaster Search and Rescu (https://arxiv.org/abs/2410.22982)
Comments:
          This paper is currently under review at IEEE IoT Magazine

- **What's New**: 이 논문은 무인항공기(UAV)를 활용한 재난 이후 검색 및 구조(PDSR) 작업의 최적화를 위한 포괄적인 프레임워크를 소개합니다. UAV 군집을 신속하게 배치하여 다양한 감지 및 통신 기능을 효과적으로 사용하는 새로운 접근법을 제안합니다.

- **Technical Details**: 제안된 프레임워크는 멀티 티어(미층) 군집 아키텍처를 채택하여 전통적인 방법보다 피해 지역을 훨씬 빠르게 완전하게 커버하고, 머신러닝을 이용한 데이터 융합으로 감지 정확성을 향상시킵니다. 이 접근법은 장애물 뒤에 묻힌 생존자를 정확하게 식별하는 것을 목표로 합니다.

- **Performance Highlights**: 이 논문의 기여는 UAV 군집 아키텍처를 통한 실시간 탐지 기능을 확보하고, 다양한 센서 기술을 통합하여 장애물 뒤에서의 인명 탐지의 정밀성을 개선하는 것입니다. 또한, PDSR 작업에서의 탐지 정확성과 지연 시간을 향상시키기 위한 제안된 프레임워크의 효과성을 입증합니다.



### Efficient Adaptation of Pre-trained Vision Transformer via Householder Transformation (https://arxiv.org/abs/2410.22952)
- **What's New**: 본 연구는 Vision Transformers (ViTs)의 하이퍼파라미터 조정을 위해 Singular Value Decomposition (SVD)을 활용한 새로운 Parameter-Efficient Fine-Tuning (PEFT) 접근 방식을 제안합니다. 이는 Householder 변환을 이용하여 고유 행렬을 효율적으로 구성하고, 각 레이어의 특성에 맞춰 유연하게 파라미터를 조정할 수 있게 합니다.

- **Technical Details**: 제안된 방법은 Householder 변환을 사용하여 각 레이어에 적응 가능한 다이아곤 행렬을 학습하며, 이는 고유 행렬을 대체합니다. Householder 변환은 단일 벡터로부터 유도되기 때문에 파라미터 효율성을 높입니다. 이를 통해 서로 다른 레이어에서 적응 행렬의 다양한 계급을 생성할 수 있어, PEFT의 유연성을 증가시킵니다.

- **Performance Highlights**: 표준 다운스트림 비전 작업에 대한 실험 결과, 제안한 방법은 다양한 ViT 버전에서 우수한 하이퍼파라미터 조정 성능을 보여주었습니다. 이 연구는 PEFT의 기존 접근 방식과는 다른 새로운 방향을 제시함으로써, 적응 성능과 파라미터 효율성 사이의 우수한 균형을 달성합니다.



### SpiroActive: Active Learning for Efficient Data Acquisition for Spirometry (https://arxiv.org/abs/2410.22950)
- **What's New**: 본 연구에서는 wearable spirometry 기술을 사용하여 호흡기 질환의 조기 진단을 위한 데이터 수집 비용을 줄이는 active learning 접근법을 제안합니다. 이는 소수의 데이터 샘플을 전략적으로 선택하여 기존 데이터 수집 방법의 어려움을 극복하고 최대한 정확한 모델 성능을 유지하는 것을 목표로 합니다.

- **Technical Details**: 우리 연구는 공공에서 사용 가능한 spirometry 데이터셋을 이용한 active learning 접근 방식을 활용하여, FEV1, FVC, PEF의 값을 추정합니다. Active learning의 기본 개념은 기계학습 시스템이 자기가 학습할 데이터를 선택하게 함으로써 정확도를 향상시킬 수 있다는 것입니다. 우리는 Single Task 및 Multi-Task Active Learning과 같은 다양한 학습 전략을 탐구했습니다.

- **Performance Highlights**: 우리의 active learning 방법론은 FEV1에서 4.96%, FVC에서 4.45%의 오류율을 달성하였으며, 이는 American Thoracic Society (ATS)의 허용 오류율인 7%를 지속적으로 초과 성과를 나타냅니다. 또한, 우리는 약 30%의 전체 데이터셋을 사용하여 단일 작업 설정에서 이러한 결과를 달성했습니다.



### Focus On This, Not That! Steering LLMs With Adaptive Feature Specification (https://arxiv.org/abs/2410.22944)
Comments:
          28pages, 14 figures

- **What's New**: 이번 연구에서는 Focus Instruction Tuning (FIT)이라는 새로운 방법론을 소개합니다. FIT는 LLMs를 특정 작업 수행 시 반응을 조정하도록 훈련시키며, 모델이 특정 기능에 초점을 맞추고 다른 기능은 무시할 수 있도록 합니다.

- **Technical Details**: FIT는 사용자가 어떤 기능에 주목할지를 동적으로 지정할 수 있게 하여, 모델의 행동을 유연하고 효과적으로 조정할 수 있습니다. 이는 특정 입력에 대해 주어진 특징에 따라 다르게 응답할 수 있도록 LLMs를 훈련시킵니다. 다양한 NLP 작업, 예를 들어 감정 분석(sentiment analysis), 자연어 추론(natural language inference), 질문-응답(question-answering) 등에서 FIT의 효과를 실험하였습니다.

- **Performance Highlights**: FIT는 훈련 시 사용하지 않았던 새로운 기능에서의 일반화 능력이 뛰어나며, 분포의 변동에 강인하게 반응합니다. 사용자가 요청하는 바에 따라 모델이 알려진 왜곡된 기능(spurious features)을 무시하고 작업 관련 기능(task-relevant features)에 주력하도록 유도하여, 더 강력하고 공정하며 제어 가능한 LLM 애플리케이션을 가능하게 합니다.



### DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data (https://arxiv.org/abs/2410.22938)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: DiffLight는 교통 신호 제어(Traffic Signal Control, TSC)에서 데이터가 누락된 시나리오를 처리하기 위해 소개된 새로운 조건부 확산 모델입니다. 이 모델은 데이터가 누락된 실제 시나리오의 요구를 충족하기 위해 설계되었습니다.

- **Technical Details**: DiffLight는 Partial Rewards Conditioned Diffusion (PRCD) 모델을 통합하여 교통 데이터 보완과 의사 결정 과제를 모두 처리합니다. 또한, Spatial-Temporal transFormer (STFormer) 아키텍처를 통해 교차로 간의 공간-시간 의존성을 효과적으로 포착합니다. Diffusion Communication Mechanism (DCM)을 통해 데이터가 누락된 상황에서도 더 나은 통신 및 제어 성능을 증진시킵니다.

- **Performance Highlights**: 다양한 데이터 누락 시나리오에 대해 5개의 데이터셋에서 실시된 실험 결과, DiffLight는 누락된 데이터가 있는 TSC 문제에 대해 매우 효과적인 성능을 보였습니다.



### Thoughtful Adoption of NLP for Civic Participation: Understanding Differences Among Policymakers (https://arxiv.org/abs/2410.22937)
Comments:
          Forthcoming in the Proceedings of the 2025 Conference on Computer Supported Cooperative Work and Social Computing (CSCW)

- **What's New**: 이 논문은 자연어 처리(NLP) 도구의 정부 내 채택 및 사용의 복잡성을 이해하기 위해 정치인과 공무원 간의 상호작용을 조사하였습니다. 연구는 라틴 아메리카의 두 개국인 칠레와 우루과이를 중심으로 진행되었습니다.

- **Technical Details**: 연구는 정치인 7명과 공무원 13명과의 인터뷰를 통해 이뤄졌고, 그들은 NLP 도구의 채택과 사용을 어떻게 결정하는지에 대한 통찰을 제공하였습니다. 정치인들은 경력 발전과 작업의 정당성을 강조하는 경향이 있었고, 공무원들은 더 효율적인 데이터 분석을 중시했습니다.

- **Performance Highlights**: 정치인과 공무원 양쪽 모두 NLP 도구의 낮은 정부 채택률의 책임을 서로에게 전가했으며, 이는 명확한 주체의 부재로 인해 발생했음을 보여주었습니다. 이 연구는 NLP 도구의 공공 부문에서의 추가 연구를 위한 새로운 기회를 제시하고 있습니다.



### YOLOv11 for Vehicle Detection: Advancements, Performance, and Applications in Intelligent Transportation Systems (https://arxiv.org/abs/2410.22898)
Comments:
          16 pages

- **What's New**: 이 논문에서는 차량 탐지 작업에 exclusive하게 초점을 맞춘 YOLO11에 대한 자세한 분석을 제공합니다. YOLO11은 이전 모델들의 성공을 기반으로 하여 탐지 속도, 정확도 및 복잡한 환경에서의 강인성을 향상시키도록 설계된 아키텍처 개선을 도입하였습니다.

- **Technical Details**: YOLO11은 다중 차량 유형(차량, 트럭, 버스, 오토바이, 자전거)을 포함하는 포괄적인 데이터셋을 사용하여 성능을 평가하였으며, precision, recall, F1 score, mean average precision (mAP)과 같은 메트릭을 사용하여 성능을 분석하였습니다.

- **Performance Highlights**: YOLO11은 이전 모델(YOLOv8 및 YOLOv10)을 능가하여 작은 차량 및 가려진 차량 탐지에서 우수한 성능을 보이며, 실시간 응용 프로그램에 적합한 경쟁력 있는 추론 시간을 시현합니다. 복잡한 차량 기하학의 탐지에서 중요한 개선사항을 보여줘 효율적이고 확장 가능한 차량 탐지 시스템 개발에 기여할 것으로 기대됩니다.



### VPO: Leveraging the Number of Votes in Preference Optimization (https://arxiv.org/abs/2410.22891)
- **What's New**: 본 논문에서는 Direct Preference Optimization (DPO) 방법을 보완하여 사용자 투표 데이터를 활용한 Vote-based Preference Optimization (VPO) 프레임워크를 제안합니다. 이 방법은 사용자 선호도를 보다 효과적으로 반영하여 다양한 주관적 선호와의 정렬을 개선하는 것을 목표로 합니다.

- **Technical Details**: VPO는 Bayesian Minimum Mean Square Error (MMSE) 추정기를 사용하여 두 개의 생성 결과 중 하나가 더 선호될 확률을 모델링합니다. VPO는 DPO와 Identity Preference Optimization (IPO) 알고리즘을 각각 VDPO와 VIPO로 확장할 수 있으며, 이는 생성 쌍의 논쟁적 여부를 분별하는 데 도움이 됩니다.

- **Performance Highlights**: 실험 결과 VDPO와 VIPO는 기존 알고리즘들보다 뛰어난 생성 품질과 훈련 안정성을 달성하였습니다. 이 프레임워크는 고비용의 인간 투표 정보가 없는 상황에서도 AI 피드백을 활용하여 적용 가능합니다.



### Less is More: Pre-Training Cross-Lingual Small-Scale Language Models with Cognitively-Plausible Curriculum Learning Strategies (https://arxiv.org/abs/2410.22886)
Comments:
          BabyLM Shared Task 2024 (Accepted, Poster), co-located in EMNLP 2024

- **What's New**: 이 논문에서는 Curriculum Learning (CL) 전략을 통해 Small-Scale Language Models (SSLMs)의 성능을 향상시키기 위한 연구를 진행하였습니다. 특히, 아동 발화를 기반으로 한 연령 순서의 코퍼스를 활용하여 언어 획득 이론에 기반한 보다 세밀한 CL 전략을 제시하였습니다.

- **Technical Details**: 연구에서는 Growing, Inwards, MMM이라는 세 가지 목표 기반 커리큘럼을 개발하였으며, 이들은 아동이 언어 습득 초기 단계에서 따르는 것으로 이론화된 발달 순서를 기반으로 하고 있습니다. 이 커리큘럼들은 SSLMs의 Masked Language Modeling (MLM) 목표를 수정하여 다국적 언어 획득 이론을 시뮬레이션합니다.

- **Performance Highlights**: 본 연구에서 제안된 커리큘럼은 비 커리큘럼 기반의 모델보다 우수한 성능을 보였으며, 특정 언어에 대해서는 더 정교한 언어 특정 MMM 버전이 더 나은 성능을 나타냈습니다. SSLMs는 LLMs에 비해 약 25배 적은 매개변수와 6,000배 적은 단어로 유사한 성능을 달성했습니다.



### Stealing User Prompts from Mixture of Experts (https://arxiv.org/abs/2410.22884)
- **What's New**: 이 논문은 Mixture-of-Experts (MoE) 모델에서 사용자 쿼리를 추출하는 새로운 공격 방법을 제시합니다. 이 공격 방법은 피해자의 쿼리와 같은 배치에 있는 쿼리를 조작하여 피해자의 프롬프트를 완전히 공개할 수 있습니다.

- **Technical Details**: 연구에서 소개된 공격은 두 계층의 Mixtral 모델에서 실행되며, 공격자는 $O({VM}^2)$ 쿼리(어휘 크기 $V$ 및 프롬프트 길이 $M$에 따라) 또는 평균적으로 각 토큰당 100개의 쿼리로 프롬프트를 추출할 수 있습니다. 이것은 공격자가 Expert-Choice-Routing을 악용하여 구현된 CUDA의 tie-handling 동작을 활용하는 것입니다.

- **Performance Highlights**: 이 연구는 사용자 프롬프트를 추출하기 위해 구조적 결함을 exploit 하는 공격의 첫 번째 사례로, 새로운 LLM(대형 언어 모델) 취약점의 클래스를 소개합니다.



### Adaptive Paradigm Synergy: Can a Cross-Paradigm Objective Enhance Long-Tailed Learning? (https://arxiv.org/abs/2410.22883)
Comments:
          11 pages, 3 figures

- **What's New**: 이 논문에서는 Self-supervised learning (SSL)과 supervised learning을 결합한 Adaptive Paradigm Synergy (APS) 접근 방식을 소개합니다. APS는 장기 분포(long-tailed distribution) 데이터를 처리할 수 있는 새로운 방법론을 제안하며, SSL의 성능 향상을 도모합니다.

- **Technical Details**: APS는 contrastive learning을 공간 구조(spatial structure) 관점에서 재조명하고, adaptive temperature tuning을 통해 잠재 공간 구조의 균일성을 동적으로 조정합니다. 또한, supervised learning의 재가중치 전략을 활용하여 온도 조정의 단점을 보완합니다. 이러한 접근은 기존의 supervised long-tailed learning 기법을 SSL에 통합할 수 있는 가능성을 제공합니다.

- **Performance Highlights**: APS는 여러 공개된 장기 분포 데이터셋에서 효과적인 성능 향상을 보여주었으며, 이를 통해 SSL과 supervised 기법 간의 보다 깊은 통합 가능성을 밝혀내고, 실제 데이터 클래스 불균형 문제를 해결하는 Robust한 모델 개발에 기여할 수 있는 길을 열었습니다.



### SFA-UNet: More Attention to Multi-Scale Contrast and Contextual Information in Infrared Small Object Segmentation (https://arxiv.org/abs/2410.22881)
Comments:
          Accepted and Presented at PRIP 2023

- **What's New**: 이 연구는 Infrared Small Object Segmentation (ISOS) 문제를 해결하기 위해 새로운 구조인 SFA-UNet을 제안합니다.

- **Technical Details**: SFA-UNet은 Scharr Convolution (SC) 및 Fast Fourier Convolution (FFC)을 통합하고, 수직 및 수평 Attention gates (AG)를 포함하는 수정된 U-Net 아키텍처로 구성됩니다. 이 모델은 인코더 및 디코더 레이어에 이중 합성곱 레이어를 사용하여 배경 대비 정보와 멀티 스케일 컨텍스트 정보를 효과적으로 학습합니다.

- **Performance Highlights**: 제안된 방법은 SIRST와 IRSTD 데이터셋을 사용하여 기존의 최첨단 방법들에 비해 모든 결합된 메트릭에서 평균 0.75%, 분산 0.025의 성능 향상을 달성하였습니다.



### Eliciting Critical Reasoning in Retrieval-Augmented Language Models via Contrastive Explanations (https://arxiv.org/abs/2410.22874)
- **What's New**: 이번 논문에서는 Retrieval-augmented generation (RAG) 방법론의 새로운 접근 방식인 Contrastive-RAG (C-RAG)를 제안합니다. C-RAG는 LLMs가 RAG 기반 정보를 비판적으로 분석할 수 있도록 도와줍니다.

- **Technical Details**: C-RAG는 네 가지 단계로 구성됩니다: (i) 수집 단계, (ii) 대조적 추론 단계, (iii) 설명 단계, (iv) 답변 단계입니다. 이 과정에 따라 LLMs가 쿼리와 관련된 문서를 검색하고, relevantes와 irrelevants한 내용을 비교하여 설명을 생성하며, 최종 답변을 도출합니다.

- **Performance Highlights**: C-RAG를 사용한 실험 결과, 기존 RAG 모델에 비해 평균 정확도가 55.4% 향상되었으며, Self-RAG 및 Self-Reasoning 모델에 비해 각각 7.2% 및 1.9%의 성능 향상이 있었습니다. 또한, C-RAG는 필요한 프롬프트 수와 훈련 예제가 현저히 적으면서도 더 나은 성능을 달성하였습니다.



### Conditioned quantum-assisted deep generative surrogate for particle-calorimeter interactions (https://arxiv.org/abs/2410.22870)
Comments:
          26 pages, 10 figures, 8 appendices

- **What's New**: 이 논문에서는 LHC(대형 강입자 충돌기) 시뮬레이션의 계산 비용을 줄이기 위해 Quantum-assisted deep generative model을 제안합니다. 특히, 대칭 Restricted Boltzmann Machine(RBM)을 갖춘 조건부 변분 오토인코더(VAE)를 통해 기존 VAE보다 더 향상된 표현력을 제공합니다. 이 방식은 D-Wave의 Advantage 양자 어닐러(quantum annealer)를 통해 샘플링을 가능하게 합니다.

- **Technical Details**: 논문에서는 Calo4pQVAE라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 4-partite 그래프를 사용하여 RBM을 재구성하고, 3D convolutional layers를 통합해 인코더와 디코더의 성능을 향상시킵니다. 또한 양자 어닐러의 flux bias parameters를 통해 조건화를 구현하고, 효과적인 온도를 추정하기 위한 새로운 적응형 방법을 소개합니다.

- **Performance Highlights**: CaloChallenge 2022의 Dataset 2를 사용하여 프레임워크의 효과를 입증하였으며, 기존 모델보다 훨씬 저렴한 계산 비용으로 셀 에너지 이동을 직접 생성할 수 있음을 보여줍니다. 이 접근 방식은 LHC 실험 시뮬레이션의 속도를 크게 향상시킬 가능성이 있습니다.



### Danoliteracy of Generative, Large Language Models (https://arxiv.org/abs/2410.22839)
Comments:
          16 pages, 13 figures, submitted to: NoDaLiDa/Baltic-HLT 2025

- **What's New**: 이 연구는 덴마크어에 대한 Generative, Large Language Models (GLLMs)의 성능을 평가하기 위한 Danoliterate Benchmark를 소개합니다. 이 벤치마크는 덴마크어 및 문화적 역량을 측정하며, 다양한 시나리오에서 모델의 능력을 검증하는 새로운 데이터 세트를 제공합니다.

- **Technical Details**: 이 연구는 RWK (Real-World Knowledge), NLU (Natural Language Understanding), NLG (Natural Language Generation) 등 세 가지 주요 범주에서 덴마크어 GLLM의 성능을 평가합니다. 605개의 다지선다형 질문으로 구성된 Citizenship Test와 같은 다양한 시나리오를 포함하여 모델 성능을 비교 분석합니다. 벤치마크는 GPT-4와 Claude Opus 모델이 최고 성능을 기록한 사실을 입증합니다.

- **Performance Highlights**: 본 연구의 결과, GLLMs의 성능은 약 80% 정도의 인간 피드백과 상관관계를 나타내며, 하나의 주요 요소가 GLLMs의 협응력(ability consistency)을 설명하는 것으로 나타났습니다. 이는 Danoliteracy 평가에서 고른 성능 변화를 제공할 수 있음을 시사합니다.



### HijackRAG: Hijacking Attacks against Retrieval-Augmented Large Language Models (https://arxiv.org/abs/2410.22832)
- **What's New**: 본 연구에서는 새로운 보안 취약점인 'retrieval prompt hijack attack (HijackRAG)'을 제시합니다. 이 공격은 공격자가 악의적인 텍스트를 지식 데이터베이스에 주입하여 RAG 시스템의 검색 메커니즘을 조작할 수 있게 합니다. 이를 통해 RAG 시스템이 목표 질문에 대한 잘못된 답변을 생성하게 만듭니다.

- **Technical Details**: HijackRAG는 최적화 문제로 형식화되며, 공격자의 지식 정도에 따라 black-box 및 white-box 공격 전략을 제시합니다. 실험 결과, HijackRAG는 여러 벤치마크 데이터셋에서 높은 공격 성공률을 기록하고 있으며, 다양한 검색 모델에 대해 이식성이 있음을 입증합니다.

- **Performance Highlights**: 후보 공격의 효과성을 입증하기 위해 수행된 광범위한 실험에서 HijackRAG는 기존 공격 방법에 비해 우수한 성과를 보였습니다. 특히, 기존 방어 메커니즘이 HijackRAG에 대해 충분한 방어력을 제공하지 못함을 확인하여, RAG 시스템의 실제 배치를 보호하기 위한 보다 강력한 보안 대책의 필요성을 강조합니다.



### Towards Robust and Efficient Federated Low-Rank Adaptation with Heterogeneous Clients (https://arxiv.org/abs/2410.22815)
- **What's New**: 본 논문에서는 Federated Learning(연합 학습) 환경에서 대규모 언어 모델(Large Language Models, LLMs)의 파인튜닝에 있어, 새로운 접근 방식인 LoRA-A2(저차원 적응 방식)를 제안합니다. 이는 높은 데이터 이질성 및 저차수 환경에서도 강력한 성능을 자랑하며, 기존 방법들의 성능 저하 문제를 개선합니다.

- **Technical Details**: LoRA-A2에서는 Adaptive Rank Selection 전략을 도입하여 중요한 LoRA rank를 선택합니다. 이를 통해 서로 다른 클라이언트들이 서로 다른 LoRA rank를 선택하게 되어 클라이언트 간의 충돌을 최소화하고, 덜 중요한 LoRA 모듈에서 리소스를 획기적으로 재할당합니다. 이 새로운 중요도 기준을 통해 각 모듈 내에서 rank의 기여도를 계산합니다.

- **Performance Highlights**: 실험 결과, LoRA-A2는 성능을 유지하면서 전체 파인튜닝 없이 99.8%의 업로드된 파라미터 수를 감소시키며, 자원 제한적인 환경에서도 LLMs의 실용적인 배치를 가능하게 합니다. 이는 연합 학습 중 통신 효율성을 극대화합니다.



### Universality of the $\pi^2/6$ Pathway in Avoiding Model Collaps (https://arxiv.org/abs/2410.22812)
Comments:
          30 pages

- **What's New**: 이 연구에서는 이론적으로 기존의 모델 훈련 중 발생하는 'Model Collapse' 문제를 피하기 위한 Augment(증강) 작업 흐름의 보편성을 입증합니다. 이전 연구에서 제시된 π2/6 한계가 다양한 통계 모델에 대해 일반화되며, 이러한 조건이 왜 발생하는지에 대한 중요한 통찰을 제공합니다.

- **Technical Details**: 연구자들은 실 데이터를 사용한 초기 훈련 후 이후 세대에서는 합성 데이터만을 사용하여 모델을 훈련하는 discard(폐기) 방식을 우려했습니다. 반면, augment(증강) 방식에서는 각 세대 훈련에 원본 실 데이터를 계속 사용하고 모델에서 생성한 합성 데이터를 추가함으로써 모델의 성능 저하를 방지할 수 있음을 실증적으로 확인했습니다. 더불어, 원본 실 데이터만 사용할 때의 테스트 리스크에 대한 한계인 π2/6을 도출하여, 선형 회귀 분석에서 모델 붕괴를 성능 저하 없이 방지할 수 있음을 밝혔습니다.

- **Performance Highlights**: 이 논문은 다양한 통계 모델 가족을 통해 π2/6 한계가 보편적으로 적용됨을 보여줍니다. 이전의 연구들과 비교했을 때, 실 데이터와 합성 데이터 모두를 사용하는 방법은 모델 성능에 있어 비교적 낮은 리스크를 유지하고 있으며, 이를 통해 다양한 작업 흐름을 모의 실험하여 최적의 방식 선택이 가능함을 증명합니다.



### Causality-Enhanced Behavior Sequence Modeling in LLMs for Personalized Recommendation (https://arxiv.org/abs/2410.22809)
- **What's New**: 본 연구에서는 Large Language Models (LLMs)를 활용한 추천 시스템의 최신 동향과 한계를 논의하고, 사용자 행동 시퀀스의 중요성을 강조하는 새로운 Counterfactual Fine-Tuning (CFT) 방법을 제안합니다.

- **Technical Details**: 우리는 행동 시퀀스가 모델 출력에 미치는 인과적 효과를 식별하기 위해 counterfactual reasoning을 활용하며, 이로부터 추론된 효과를 직접 활용하여 진짜 레이블에 적합하도록 하는 새로운 작업을 도입합니다. 또한, 아이템 토큰에 대한 강조 강도를 조정하기 위한 token-level weighting 메커니즘을 개발하여, 예측할 때 초기 토큰에서 후속 토큰으로 갈수록 행동 시퀀스의 영향력이 줄어드는 것을 반영합니다.

- **Performance Highlights**: 실제 데이터셋에 대한 광범위한 실험을 통해 CFT가 행동 시퀀스 모델링을 효과적으로 개선함을 입증하였습니다. 이로 인해 LLM 기반의 추천 시스템의 효과성과 개인화된 추천의 정확성이 향상되었습니다.



### Run-Time Adaptation of Neural Beamforming for Robust Speech Dereverberation and Denoising (https://arxiv.org/abs/2410.22805)
Comments:
          Accepted to APSIPA2024

- **What's New**: 본 논문에서는 실제 환경에서 실시간 자동 음성 인식(ASR)을 위한 음성 향상 기술을 설명합니다. 구체적으로, DNN(Deep Neural Network)을 기반으로 한 신경 빔형성(neural beamforming)을 통한 마스크 추정 및 음성 필터링 과정을 소개하고, 비슷한 환경의 음성 데이터를 활용한 실시간 적응 방식에 대한 연구를 제시합니다.

- **Technical Details**: 논문은 DNN을 사용한 신경 빔형성에서의 음향 특성 변화를 다루고 있으며, 이를 위해 WPE(Weighted Prediction Error)와 MPDR(Minimum Power Distortionless Response)를 통합한 WPD(Weighted Power Minimization Distortionless Response) 빔형성을 제안합니다. 이로 인해 각기 다른 조건에서의 음성 신호의 억제를 위한 필터 계산이 가능해집니다.

- **Performance Highlights**: 다양한 화자 수, 잔향 시간, 신호 대 잡음 비율(SNR) 등 구성된 조건에서 실시간 적응의 효과를 평가하며 ASR 성능의 향상을 입증합니다. 이 시스템은 음성 업무 성능 개선을 위한 효율적인 변화를 이끌어 내는 데 성공적임을 보여줍니다.



### DOA-Aware Audio-Visual Self-Supervised Learning for Sound Event Localization and Detection (https://arxiv.org/abs/2410.22803)
Comments:
          Accepted to APSIPA2023

- **What's New**: 이 논문은 첫 번째 순서 암비소닉스(First-order Ambisonics, FOA) 마이크로폰으로 캡처한 공간 오디오 녹음에 대한 소리 이벤트 로컬리제이션 및 탐지(Sound Event Localization and Detection, SELD)를 다루고 있습니다. 특히, 지도 학습 방식으로 DNN을 훈련할 때 주석이 달린 데이터가 부족한 문제를 자기 지도 방식(Self-supervised Learning)으로 해결하는 새로운 방법을 제안하고 있습니다.

- **Technical Details**: 연구에서는 FOA와 360도 전 방향 카메라로 촬영된 VR 콘텐츠를 활용해 오디오와 비주얼 인코더를 대조 학습(Contrastive Learning) 방식으로 공동 훈련합니다. 핵심 기능은 DOA별 음향 임베딩이 원시 오디오 데이터에서 공동으로 추출되며, DOA-wise 비주얼 임베딩은 해당 DOA를 중심으로 한 지역 비주얼 크롭에서 별도로 추출된다는 점입니다. 이를 통해 오디오 인코더에 설비된 잠재적 특성이 오디오 이벤트의 클래스와 DOA를 나타내도록 유도합니다.

- **Performance Highlights**: DCASE2022 Task 3 데이터셋을 활용한 실험에서 비주석된 100시간의 오디오-비주얼 녹음이 SELD의 오류 점수를 36.4점에서 34.9점으로 감소시키는 성과를 보여주었습니다.



### Dual Contrastive Transformer for Hierarchical Preference Modeling in Sequential Recommendation (https://arxiv.org/abs/2410.22790)
- **What's New**: 본 논문은 기존 Sequential Recommender Systems (SRSs)의 한계를 극복하기 위해 고안된 새로운 계층적 선호 모델링 프레임워크(Hierarchical Preference modeling, HPM)를 소개합니다. 이 프레임워크는 사용자 선호를 더 정확하게 모델링하기 위해 여러 가지 혁신적인 모듈을 도입하였습니다.

- **Technical Details**: 이 연구에서는 Dual-Transformer 모듈과 Dual Contrastive Learning 방식을 통해 사용자의 저수준(low-level) 및 고수준(high-level) 선호를 구별하여 학습합니다. 또한, Semantics-enhanced Context Embedding Learning 모듈을 통해 아이템 간의 숨겨진 의미 관계를 잘 캡처하여 더 나은 추천 성능을 위해 정보가 풍부한 컨텍스트 임베딩을 생성합니다.

- **Performance Highlights**: 여섯 개의 실제 데이터셋에 대한 광범위한 실험 결과, 제안된 방법이 기존 최첨단 방법들보다 우수한 성능을 나타내며, 고안된 설계의 합리성이 입증되었습니다.



### Contrastive Learning and Adversarial Disentanglement for Privacy-Preserving Task-Oriented Semantic Communications (https://arxiv.org/abs/2410.22784)
Comments:
          Submitted to EEE Journal on Selected Areas in Communications (JSAC): Intelligent Communications for Real-Time Computer Vision (Comm4CV)

- **What's New**: 본 논문에서 제안된 CLAD(contrastive learning and adversarial disentanglement) 방법론은 태스크 지향적 의미 통신 시스템의 정보 전송 방식에 혁신을 가져옵니다. 기존의 문제점인 태스크 관련 및 비관련 정보를 완전히 분리하지 못하는 한계를 극복하기 위해 정보 병목(information-bottleneck) 방식을 도입하였습니다.

- **Technical Details**: CLAD는 대조 학습(contrastive learning)을 활용하여 태스크 관련 기능(feature)을 효과적으로 캡처하는 동시에 적대적 분리(adversarial disentanglement)를 통해 태스크와 무관한 정보를 제거합니다. 또한, 인코딩된 기능 벡터의 정보 유지 지수(information retention index, IRI)를 도입하여 인코딩된 기능과 입력 간의 상호 정보(mutual information)를 대리하는 지표로 사용합니다.

- **Performance Highlights**: CLAD는 태스크 성능, 프라이버시 보존 및 IRI 측면에서 최신 기술들의 성능을 능가하였습니다. CLAD는 약 2.5-3%의 예측 성능 향상, 77-90%의 IRI 감소 및 57-76%의 적대 정확도 감소를 달성하였습니다.



### InjecGuard: Benchmarking and Mitigating Over-defense in Prompt Injection Guardrail Models (https://arxiv.org/abs/2410.22770)
- **What's New**: 이번 논문에서는 Prompt injection 공격으로부터 대형 언어 모델(LLM)을 방어하기 위한 새로운 평가 데이터셋 NotInject를 소개합니다. NotInject는 다양한 프롬프트 가드 모델의 과도한 방어(over-defense) 문제를 체계적으로 측정할 수 있도록 설계되었습니다.

- **Technical Details**: NotInject 데이터셋은 339개의 benign 샘플로 구성되며, 이들은 프롬프트 인젝션 공격에서 자주 사용되는 trigger words를 포함하고 있습니다. InjecGuard라는 새로운 프롬프트 가드 모델은 Mitigating Over-defense for Free (MOF)라는 새로운 훈련 전략을 통해 trigger word에 대한 편향을 크게 줄입니다.

- **Performance Highlights**: InjecGuard는 NotInject를 포함한 다양한 벤치마크에서 state-of-the-art 성능을 입증하였으며, 기존 최고의 모델보다 30.8% 개선된 83% 이상의 평균 정확도로 benign, malicious 및 over-defense 입력을 탐지합니다.



### Beyond Ontology in Dialogue State Tracking for Goal-Oriented Chatbo (https://arxiv.org/abs/2410.22767)
Comments:
          There are 10 chapters, including references, and 2 figures used. To be presented at the 15th IEEE International Conference on Knowledge Graphs (ICKG2024)

- **What's New**: 이 논문에서는 고전적인 대화 상태 추적( DST ) 방법의 한계를 극복하기 위한 새로운 접근법을 제안합니다. 고정된 온톨로지(ontology) 및 수동으로 수집된 슬롯 값을 사용하지 않고도 LLM( 대형 언어 모델 )이 대화 상태를 추론할 수 있도록 하는 방법을 소개합니다.

- **Technical Details**: 제안된 방법은 명령 조정(instruction tuning) 및 고급 프롬프트 전략을 활용하여 열린 도메인 대화에 대한 유연성을 제공합니다. 또한, VGAE(변종 그래프 오토인코더)를 사용하여 사용자 의도를 모델링하고 예측하는 과정도 포함되어 있습니다.

- **Performance Highlights**: 본 논문에서 제안한 방법은 새로운 JGA 42.57%를 달성하며 기존의 온톨로지 없는 DST 모델들을 초월하고 실제 열린 도메인 대화 데이터에서 우수한 성능을 보였습니다.



### SoftCTRL: Soft conservative KL-control of Transformer Reinforcement Learning for Autonomous Driving (https://arxiv.org/abs/2410.22752)
Comments:
          submitted to IEEE Open Journal of Intelligent Transportation Systems

- **What's New**: 본 논문에서는 도시 자율주행차(SDV)에서의 모션 플래닝 문제를 해결하기 위해, 모방 학습(IL)과 강화 학습(RL)을 결합한 새로운 접근 방식을 제안합니다. 특히, KL 다이버전스(KL divergence)를 사용한 암묵적 엔트로피-KL 제어를 통해 IL의 과잉 보수 문제를 완화하여 RL 모델을 개선합니다.

- **Technical Details**: 제안된 방법은 사전 훈련된 Transformer IL 모델을 사용하여 RL 업데이트를 간단한 보상으로 제약합니다. RL 정책은 KL 다이버전스와 엔트로피의 조합으로 암묵적으로 정규화되어 더 다양한 경로를 생성하게 유도합니다. 실험을 통해 제안된 방법이 IL만 사용할 때보다 17% 더 실패를 줄일 수 있음을 보여주었습니다.

- **Performance Highlights**: 실험 결과, 우리의 접근 방식은 IL 단독 사용보다 17% 및 RL 단독 사용보다 41%의 실패율 감소를 보이며, 모방 과제에서 뛰어난 성능을 발휘합니다. 특히, 복잡하고 도전적인 도시 시뮬레이션 환경에서 안정성을 크게 개선했습니다.



### Designing AI Personalities: Enhancing Human-Agent Interaction Through Thoughtful Persona Design (https://arxiv.org/abs/2410.22744)
Comments:
          8 pages, the workshop accepted at the 23rd International Conference on Mobile and Ubiquitous Multimedia (MUM 2024)

- **What's New**: 이 워크숍은 다양한 상황에서 AI 에이전트의 페르소나(persona) 디자인에 중점을 두고 연구 커뮤니티를 구축하기 위한 것입니다. AI 에이전트의 음성, 형태, 인구 통계 등의 디자인 요소가 사용자 만족도와 참여도에 미치는 영향을 탐구합니다.

- **Technical Details**: AI 에이전트(agents)는 사용자와 대화하는 인터페이스인 Conversational User Interfaces (CUIs)를 통해 작동하며, 이는 텍스트나 음성을 통해 사용자와 상호작용합니다. AI 에이전트는 사용자의 선호를 학습하고 적응하는 능력이 있으며, 예로는 Siri, Google Assistant, Alexa 등이 있습니다.

- **Performance Highlights**: AI 시스템의 기술적 성과뿐만 아니라 에이전트의 페르소나와 프레젠테이션도 사용자 경험, 참여도, 만족도 및 신뢰에 영향을 미치는 중요한 요소임을 강조합니다. 또한 Persona 선택이 신뢰와 호감도, 그리고 채택 의도에 영향을 미친다는 연구 결과도 언급됩니다.



### st-DTPM: Spatial-Temporal Guided Diffusion Transformer Probabilistic Model for Delayed Scan PET Image Prediction (https://arxiv.org/abs/2410.22732)
- **What's New**: 이번 연구는 이중 시간 PET(dual-time PET) 이미징에서의 예측 문제를 해결하기 위해 새로운 공간-시간 유도 확산 변환 모델(spatial-temporal guided diffusion transformer probabilistic model, st-DTPM)을 제안합니다.

- **Technical Details**: 이 구조는 CNN의 패치-와이즈(patch-wise) 특성과 Transformer의 픽셀-와이즈(pixel-wise) 관련성을 통합한 U-net 프레임워크를 활용합니다. 이후 조건부 DDPM(Conditional Denoising Diffusion Probabilistic Model) 모델을 사용하여 이미지 합성을 진행합니다. 공간 조건에서는 초기 스캔 PET 이미지와 노이즈가 있는 PET 이미지를 각 디노이징 단계에서 결합하여 디노이징 샘플링의 공간 분포를 유도합니다. 시간 조건에서는 확산 시간 단계와 지연 시간을 보편적인 시간 벡터로 변환하여 모델 아키텍처의 각 레이어에 삽입하여 예측 정확도를 더욱 향상시킵니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 이미지 품질과 구조 정보를 보존하는 데 있어 다른 접근 방식보다 우수함을 입증하였으며, 예측 작업의 효율성을 확인했습니다.



### Offline Behavior Distillation (https://arxiv.org/abs/2410.22728)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 본 논문에서는 오프라인 행동 증류(Offline Behavior Distillation, OBD) 문제를 제안하며, 두 가지 기본 OBD 목표인 DBC와 개선된 PBC를 소개합니다.

- **Technical Details**: OBD는 서브 최적 RL 데이터에서 제한된 전문가 행동 데이터를 합성하여 정책 학습을 가속화하는 과정입니다. PBC는 오프라인 데이터에서의 결정 차이를 측정하는 기본 방법을 제공합니다. 하지만 복잡한 이층 최적화(bi-level optimization)로 인해 PBC는 성능 보장에 대한 최악의 경계를 가집니다. 우리는 정책 성능과 행동-가치 가중 결정 차이(action-value weighted decision difference) 간의 동등성을 증명하고 Av-PBC라는 새로운 OBD 목표를 제안합니다.

- **Performance Highlights**: 실험 결과, Av-PBC는 OBD 성능 측면에서 DBC와 PBC보다 각각 82.8% 및 25.7%의 개선을 보였고, convergence 속도 또한 DBC와 PBC 대비 가장 빠른 성능을 나타냈습니다. 또한, 엔샘블 정책 훈련을 통해 Av-PBC에서 훈련된 데이터의 성능을 25.8% 향상시켰습니다.



### Robotic State Recognition with Image-to-Text Retrieval Task of Pre-Trained Vision-Language Model and Black-Box Optimization (https://arxiv.org/abs/2410.22707)
Comments:
          Accepted at Humanoids2024

- **What's New**: 이 논문에서는 로봇이 환경과 객체의 상태를 인식하는 방법으로, 사전 훈련된 vision-language 모델을 활용하는 새로운 방식을 제안합니다. 기존의 수작업 어노테이션, 특수 센서 준비 및 수동 프로그래밍 없이, 언어 프롬프트를 이용하여 상태 인식의 정확성을 높일 수 있습니다.

- **Technical Details**: 제안된 방법은 CLIP 모델을 활용하여, 미리 정의된 텍스트 프롬프트와 현재 이미지 간의 유사성을 계산하여 상태 인식을 수행합니다. black-box optimization을 통해 각 프롬프트의 최적 가중치를 부여하여 더욱 정확한 인식이 가능합니다.

- **Performance Highlights**: 이 연구를 통해 투명한 문이 열려 있는지 닫혀 있는지, 수도꼭지에서 물이 흐르고 있는지, 주방이 깨끗한지의 상태를 인식할 수 있으며, 기존의 방법들에 비해 수고를 덜고 자원을 효율적으로 관리할 수 있습니다.



### Permutation Invariant Learning with High-Dimensional Particle Filters (https://arxiv.org/abs/2410.22695)
Comments:
          Website: this https URL

- **What's New**: 이번 연구에서는 high-dimensional particle filters를 기반으로 하는 새로운 permutation-invariant learning framework를 소개합니다. 이 프레임워크는 데이터의 순서에 영향을 받지 않으며, catastrophic forgetting(재앙적 망각)과 loss of plasticity(유연성 상실)을 완화합니다.

- **Technical Details**: 제안된 방법은 gradient-based optimization의 장점과 Bayesian 방법의 특성을 결합한 효율적인 particle filter를 개발하여 고차원 모델을 최적화하는 것입니다. 특히, particle filter가 훈련 미니배치의 순서에 무관하게 작용하도록 이론적으로 입증하였습니다.

- **Performance Highlights**: SplitMNIST, SplitCIFAR100, ProcGen 등의 연속 학습 및 강화 학습 벤치마크에서 광범위한 실험을 통해, 제안한 방법이 표준 기준 모델 대비 성능 향상과 변동성 감소를 일관되게 보였습니다.



### Choice between Partial Trajectories (https://arxiv.org/abs/2410.22690)
- **What's New**: 본 논문에서는 AI 에이전트가 인간의 선호를 배우는 데 있어 bootstrap return 모델을 제안하며, 이를 통해 인간의 신념을 고려한 선택 모델링을 수행합니다. 기존의 partial return 모델과 cumulative advantage 모델의 한계를 극복할 수 있는 방법으로 bootstrap return이 소개됩니다.

- **Technical Details**: 논문에서는 bootstrapped return을 통해 부분적인 보상(partial return)과 미래 보상의 추정값을 합산하여 사용하는 방법을 설명합니다. 이를 통해 인간의 신념에 따라 선택의 결과를 해석할 수 있으며, 보상 함수(reward function)를 학습하는 데 있어 더 높은 견고성을 보입니다. Axiom과 Alignment Theorem을 통해 목표(goal)와 신념(belief)을 분리하는 방법이 formalized 되어 있습니다.

- **Performance Highlights**: 실험 결과, bootstrapped return 모델은 선택 행동(choice behavior)에 대해 더 높은 강인성을 보이며, 잘못된 신념을 가진 데이터로부터도 효과적으로 보상 함수를 회복할 수 있음을 보여줍니다. 이는 에이전트가 인간의 의도를 보다 잘 반영한 보상 함수를 학습할 수 있는 기회를 제공합니다.



### Multi-Task Interactive Robot Fleet Learning with Visual World Models (https://arxiv.org/abs/2410.22689)
Comments:
          In Proceedings of CoRL 2024

- **What's New**: 최근 대규모 다중 작업 로봇 학습 분야에서 발전을 이루며 가정 및 산업 환경에서 여러 작업을 수행할 수 있는 로봇 플릿(robots fleet)의 배치를 가능하게하는 시스템인 Sirius-Fleet를 소개합니다. 이 프레임워크는 로봇의 성능을 모니터링하고 필요할 경우 인간의 개입을 통해 로봇의 행동을 수정할 수 있도록 합니다.

- **Technical Details**: Sirius-Fleet는 다중 작업 정책(multi-task policy)과 실시간 모니터링 메커니즘이 결합된 시스템으로, 로봇의 자율성이 향상됨에 따라 자동으로 예측 기준을 조정하여 인간의 개입을 줄이는 특징이 있습니다. 또한, 다양한 작업을 수행하기 위해 비주얼 월드 모델(visual world model)을 사용하여 미래의 작업 결과를 예측하고, 실패 예측 및 OOD(out-of-distribution) 예측을 위한 두 가지 이상 탐지기를 개발했습니다.

- **Performance Highlights**: 대규모 벤치마크에서 수행된 평가에 따르면, Sirius-Fleet는 다중 작업 정책의 성능과 모니터링 정확성을 개선하는 데 효과적이며, 전체 시스템 성능에서 95% 이상의 성공률을 기록했습니다. 또한, 이 시스템은 시간이 지남에 따라 인간의 작업 부담을 줄이고 지속적으로 정책을 개선하는 것으로 나타났습니다.



### Improving Uncertainty Quantification in Large Language Models via Semantic Embeddings (https://arxiv.org/abs/2410.22685)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)에서 정확히 불확실성을 정량화하는 새로운 접근 방식을 제안합니다. 기존의 방법들은 이중 포함 관계를 기준으로 여러 생성된 응답간의 의미적 불확실성을 측정하는데 의존하며, 이는 세부적인 표현 차이에 민감하여 실제 불확실성을 과대평가하는 경향이 있습니다. 이에 반해, 저자들은 의미 임베딩을 사용하여 의미적 불확실성을 보다 부드럽고 견고하게 추정할 수 있는 방법을 제시합니다.

- **Technical Details**: 저자들은 의미적 임베딩 불확실성(Semantic Embedding Uncertainty, SEU) 개념을 도입하며, 이는 전체 응답 임베딩의 평균 쌍 간 코사인 유사도를 활용합니다. 이 방법은 이중 포함을 기준으로 의미적으로 동등한 응답을 군집화하는 과정에서 발생하는 문제를 피할 수 있습니다. 또한 저자들은 이론을 기반으로 한 암모르티즈드(SEU) 모델을 제시하여 잠재 변수로서 의미를 모델링해 단일 전방 통과로 불확실성을 추정하는 방식을 개발하였습니다. 이를 통해 연산 오버헤드를 획기적으로 줄이고 실무 환경에서의 활용성을 향상시킬 수 있습니다.

- **Performance Highlights**: 다양한 질문-응답 데이터셋과 최전선 LLMs에 대한 실험 결과, 저자들이 제안한 임베딩 기반 방법이 기존의 전통적인 기법들보다 더 정확하고 섬세한 불확실성 정량화를 제공함을 입증하였습니다.



### Backdoor Attack Against Vision Transformers via Attention Gradient-Based Image Erosion (https://arxiv.org/abs/2410.22678)
Comments:
          Accepted by IEEE GLOBECOM 2024

- **What's New**: 이번 연구에서는 Vision Transformers (ViTs)를 겨냥한 Attention Gradient 기반 Erosion Backdoor(AGEB)를 제안합니다. AGEB는 ViTs의 주의(attention) 메커니즘을 활용하여 픽셀을 선택적으로 침식시켜 백도어 트리거를 내장합니다. 기존 방법들에 비해 AGEB는 공격 stealthiness(은폐성)와 효과성의 최적 균형을 달성합니다.

- **Technical Details**: AGEB는 ViTs의 주의 그래디언트를 고려하여 최대 주의 그래디언트를 가진 영역에서 픽셀을 선택적으로 침식합니다. 이 과정은 이미지 변경을 목표로 하며, 이를 통해 트리거를 은밀하게 생성합니다. AGEB는 기존의 지역적 패치 기반 트리거의 한계를 극복하기 위해 전역적 트리거 메커니즘을 채택해 일반화 능력을 향상시킵니다.

- **Performance Highlights**: 다양한 ViT 아키텍처와 데이터셋에서 광범위한 실험 평가가 이루어졌으며, AGEB는 높은 공격 성공률(Attack Success Rate, ASR)과 함께 청정 데이터 정확도(Clean Data Accuracy, CDA)를 유지하는 뛰어난 효과를 보여주었습니다. 특히 ImageNette 데이터셋에서는 테스트 이미지의 97.02%가 특정 타겟 클래스에 정확히 분류되었습니다.



### $\textbf{EMOS}$: $\textbf{E}$mbodiment-aware Heterogeneous $\textbf{M}$ulti-robot $\textbf{O}$perating $\textbf{S}$ystem with LLM Agents (https://arxiv.org/abs/2410.22662)
Comments:
          10 pages of main content, 3 pages of references, 5 pages of appendix, 7 figures in total

- **What's New**: 이 논문에서는 이질적인 다중 로봇 시스템(HMRS)을 위한 새로운 다중 에이전트 프레임워크인 EMOS를 소개하고 있습니다. 이 시스템은 로봇의 물리적 외형과 능력을 인식하여 효과적인 협업을 가능하게 합니다. 특히, 전통적인 역할 배정 방식을 탈피하고 로봇의 URDF 파일을 이해하여 스스로 능력을 설명하는 'Robot Resume' 개념을 도입했습니다.

- **Technical Details**: EMOS는 언어 모델(LLM)을 기반으로 하여 자기 자신과 자신의 물리적 특성 및 작업 수행을 이해하는 능력인 embodiment-aware reasoning을 통해 협업을 지원합니다. Habitat-MAS라는 새로운 벤치마크를 통해 다중 에이전트 시스템이 로봇의 물리적 능력을 인식하고 이에 맞춰 작업을 계획, 할당 및 실행하는 방식을 평가합니다. 실험은 네 가지 주요 과제를 포함하며, 각 과제는 로봇의 조작, 인식, 탐색 및 복합적인 다층 객체 재배치를 요구합니다.

- **Performance Highlights**: Habitat-MAS 벤치마크의 실험 결과, 로봇 이력서와 다중 에이전트 시스템의 계층적 설계가 이질적인 다중 로봇 시스템의 효과적 운영에 필수적임을 보여주었습니다. 특히, 로봇의 이력서가 embodiment-aware reasoning에 중요한 역할을 하며, 다중 에이전트 시스템이 복잡한 작업을 자동화하는 데 기여하고 있습니다.



### Incremental Learning of Retrievable Skills For Efficient Continual Task Adaptation (https://arxiv.org/abs/2410.22658)
- **What's New**: 이 논문은 Continual Imitation Learning (CiL)이라는 새로운 프레임워크인 IsCiL을 소개합니다. IsCiL은 다양한 데모로부터 공유 가능한 기술을 점진적으로 학습함으로써 지식 공유의 한계를 해결합니다.

- **Technical Details**: IsCiL은 각 데모에서 적절한 기술을 프로토타입 기반 메모리를 통해 검색하고, 각 프로토타입에 대해 해당 어댑터에 대해 점진적으로 기술을 학습합니다. 이 프레임워크는 작은 어댑터인 기술 검색기(skill retriever)와 기술 디코더(skill decoder)로 구성된 두 층의 계층 구조를 가지고 있습니다.

- **Performance Highlights**: Franka-Kitchen과 Meta-World 환경에서 진행된 CiL 실험 결과, IsCiL은 샘플 효율성과 작업 적응 모두에서 우수한 성능을 보였습니다. 이 프레임워크는 포괄적인 전문가 데모 없이도 유연하게 적응하며, 다양한 지침과 데모로 구성된 작업을 학습할 수 있습니다.



### Prove Your Point!: Bringing Proof-Enhancement Principles to Argumentative Essay Generation (https://arxiv.org/abs/2410.22642)
Comments:
          EMNLP 2024

- **What's New**: 이 논문에서는 논리적 개선(logical enhancement)에 중점을 두고, 주장을 강화(proof-enhancement)하고 스스로 주석을 달도록(self-annotation) 하는 두 단계 프레임워크인 PESA를 제안합니다. 기존의 주장을 단순히 생성하는 방법의 한계를 극복하고, 주장 간의 고차원 연결을 강화하여 논리적인 일관성을 높입니다.

- **Technical Details**: PESA는 두 가지 주요 단계를 포함합니다: 증거와 근거(claims and grounds)를 위한 가짜 레이블 생성과, 증거 원칙을 도입하여 논리적 흐름을 보장하는 트리 계획(tree planning) 접근 방식입니다. 이 방식은 Toulmin Argumentation Model을 차용하여, 구체적 근거와 추상적 주장을 연결하여 보다 설득력 있는 주장을 생성하도록 설계되었습니다.

- **Performance Highlights**: PESA는 기존의 강력한 기준 모델들보다 논리적 유효성과 설득력에서 우수한 성과(State-Of-Art)를 보여주며, 자동 평가 메트릭스에서 뛰어난 결과를 달성했습니다. 추가적으로, 인적 평가에서도 PESA가 유창성, 논리성 및 설득력 측면에서 현저히 우수하다는 것을 확인했습니다.



### DECRL: A Deep Evolutionary Clustering Jointed Temporal Knowledge Graph Representation Learning Approach (https://arxiv.org/abs/2410.22631)
Comments:
          Accepted by NeurIPS 2024, 17 pages, and 3 figures

- **What's New**: 본 논문에서는 TKG(Temporal Knowledge Graph)에서 고차 상관의 시간적 진화를 포착하기 위한 DECRL(Deep Evolutionary Clustering jointed temporal knowledge graph Representation Learning) 방법을 제안합니다. DECRL은 고차 상관의 진화를 모델링하기 위한 깊이 있는 진화 적 클러스터링 모듈을 포함하고 있어, 클러스터가 여러 엔티티 간의 고차 상관을 나타냅니다.

- **Technical Details**: DECRL은 클러스터의 연속적인 시간 정합성을 유지하기 위해 클러스터 인식 비지도 정렬 메커니즘을 도입합니다. 또한, 클러스터 간의 잠재적 상관관계를 캡처하기 위한 암묵적 상관 인코더를 포함하며, 이는 클러스터 간의 상호작용 강도를 정의합니다. 이는 전역 그래프의 도움 아래 달성됩니다.

- **Performance Highlights**: DECRL은 7개의 실제 데이터셋에서 실험을 진행한 결과, MRR, Hits@1, Hits@3 및 Hits@10에서 각각 9.53%, 12.98%, 10.42%, 14.68%의 평균 성능 향상을 보이며, 현재 최첨단(SOTA) 성능을 달성했습니다.



### Efficient Feature Extraction and Classification Architecture for MRI-Based Brain Tumor Detection (https://arxiv.org/abs/2410.22619)
- **What's New**: 이번 연구에서는 뇌 MRI 스캔을 이용하여 Tumor의 존재 유무를 식별하기 위해 Convolutional Neural Network (CNN) 모델을 훈련시키고, 그것의 정확도를 99.17%에 도달하는 성과를 이루었습니다.

- **Technical Details**: 본 연구에서는 CNN 모델과 KNN, Logistic regression, SVM, Random Forest, Naive Bayes, Perception과 같은 전통적인 머신러닝 모델을 포함하여 총 6개의 머신 러닝 모델을 적용하였습니다. CNN 모델은 DenseNet, ResNet, EfficientNetB0와 같은 다양한 네트워크 아키텍처를 사용하여 최적화되었으며, 특징 추출 및 이미지 분류 프로세스에서 중요성을 강조했습니다.

- **Performance Highlights**: CNN 모델은 99.17%의 정확도를 달성했으며, Precision, Recall, Specificity, F1 score 등의 표준 메트릭을 사용하여 머신 러닝 모델들과 비교되었습니다. 의사 진단의 중요성을 통해 CNN 모델은 Tumor 존재를 식별하고 환자 치료에 기여할 수 있는 사례로 나타났습니다.



### Are Large-Language Models Graph Algorithmic Reasoners? (https://arxiv.org/abs/2410.22597)
Comments:
          9 pages, 13 Figures

- **What's New**: 이번 연구에서는 Large Language Models (LLMs)의 주요한 도전 과제를 다루고 있습니다. LLMs는 많은 작업에서 우수한 성능을 보이지만, 명시적 그래프에서 여러 단계를 필요로 하는 추론 문제에서 여전히 어려움을 겪고 있습니다. 이를 해결하기 위해 기존의 알고리즘적 추론 작업을 평가할 수 있는 새로운 벤치마크인 MAGMA를 소개합니다.

- **Technical Details**: MAGMA 벤치마크는 Breadth-First Search (BFS), Depth-First Search (DFS), Dijkstra's algorithm, Floyd-Warshall algorithm, Prim's Minimum Spanning Tree (MST-Prim)와 같은 5가지 기본 알고리즘을 포함하고 있습니다. 이 벤치마크는 각 단계에서의 모델의 성능을 체계적으로 평가하며, LLMs가 다단계 그래프 추론 문제를 해결하는 방식에서 중간 단계의 중요성을 강조합니다.

- **Performance Highlights**: LLMs에 중간 단계를 포함시키는 것이 알고리즘적 추론 성능을 상당히 향상시키는 것으로 나타났습니다. 더 작은 미세 조정 모델이 더 큰 기본 언어 모델보다 그래프 알고리즘 추론 작업에서 더 우수한 성능을 보였고, 중간 단계로 미세 조정된 모델은 불필요한 정보에 민감한 경향이 있어 단계별 추론의 명확성이 필요함을 보여주었습니다.



### FGCE: Feasible Group Counterfactual Explanations for Auditing Fairness (https://arxiv.org/abs/2410.22591)
- **What's New**: 이 논문은 모델 공정성을 감사하기 위한 그룹 반사적 설명 (group counterfactual explanations) 생성을 위한 첫 번째 그래프 기반 프레임워크인 FGCE(Feasible Group Counterfactual Explanations)를 소개합니다. 이는 신뢰할 수 있는 머신 러닝의 중요한 측면입니다.

- **Technical Details**: FGCE는 실제 적용 가능한 제약을 포착하고 유사한 반사적 설명을 가진 하위 그룹을 구성하여 기존 방법과 차별화됩니다. 이 프레임워크는 반사적 생성은 물론 비용과 범위 간의 균형을 포함한 주요 트레이드오프를 처리합니다.

- **Performance Highlights**: 실험 결과는 FGCE가 제공하는 공정성 감사에 대한 효과성을 보여 주며, 다른 기존 연구와 비교할 때 더욱 엄격한 실현 가능성 제약을 포함하면서도 우수한 성과를 발휘하는 것을 입증합니다.



### BENCHAGENTS: Automated Benchmark Creation with Agent Interaction (https://arxiv.org/abs/2410.22584)
- **What's New**: 본 연구에서는 BENCHAGENTS라는 프레임워크를 소개하며, 이는 대형 언어 모델(LLMs)을 활용하여 복잡한 기능을 평가하기 위한 벤치마크 생성 과정을 자동화합니다. 특히, 벤치마크 생성 과정을 계획, 생성, 검증, 평가의 네 가지 단계로 나누고 각각을 LLM 에이전트가 담당하여 데이터의 질과 메트릭 품질을 보장합니다.

- **Technical Details**: BENCHAGENTS는 다양한 LLM 에이전트를 사용하여 벤치마크 생성 프로세스를 관리합니다. 각 에이전트는 다음과 같은 역할을 수행합니다: Planning Agent는 벤치마크 계획을 수립하고, Data Generation Agent는 데이터를 생성하며, Verification Agent는 생성된 데이터의 품질을 확인하고, Evaluation Agent는 모델 성능 평가를 위한 메트릭을 생성합니다.

- **Performance Highlights**: BENCHAGENTS를 활용하여 생성된 두 개의 벤치마크(BA-Calendar 및 BA-Text)에 대해 최첨단 LLM 7개 모델을 평가한 결과, 모든 모델이 제약 조건 충족에 어려움을 겪고 있으며, 특히 제약 조건 수가 증가할수록 성능이 떨어지는 경향을 보였습니다. 이는 모델 간의 우선순위 설정에서 차이점을 보이며, 주로 수치적 혹은 논리적 추론을 요구하는 제약 조건에서 실패하는 경향이 있음을 나타냅니다.



### Energy-Aware Multi-Agent Reinforcement Learning for Collaborative Execution in Mission-Oriented Drone Networks (https://arxiv.org/abs/2410.22578)
Comments:
          2022 International Conference on Computer Communications and Networks

- **What's New**: 이 연구는 다중 에이전트 강화 학습(multi-agent reinforcement learning, MARL)을 활용해 드론 네트워크의 협업 실행 모델을 제안합니다. 또한 드론의 배터리 수준을 고려하여 경로(planning trajectories)를 효율적으로 계획할 수 있는 방법을 제시합니다.

- **Technical Details**: 제안된 모델은 드론 하드웨어의 배터리 한계를 고려하여 다중 작업 수행에 필요한 경로를 최적화합니다. 드론 간의 협업은 심층 Q-네트워크(deep Q-network, DQN)를 통해 이루어지며, 드론은 현재 상태와 환경 정보를 바탕으로 경로를 조정하고 임무를 수행합니다.

- **Performance Highlights**: 시뮬레이션 결과, 제안된 협업 실행 모델은 임무 완료 성공률이 최소 80%에 달하며, 작업 밀도가 적당할 경우 성공률이 100%에 이릅니다. 이는 기존 연구에 비해 향상된 성능을 보여줍니다.



### Unpicking Data at the Seams: VAEs, Disentanglement and Independent Components (https://arxiv.org/abs/2410.22559)
- **What's New**: 이번 연구는 Variational Autoencoders (VAE)에서의 disentanglement(비관계화)에 대한 이론적 분석을 제공하며, encoder Jacobian의 직교성(orthogonality)과 데이터의 독립적인 구성 요소들을 효과적으로 식별하는 방법을 보여줍니다.

- **Technical Details**: 연구에서는 VAE의 posterior covariance matrix로 대각 행렬(diagonal covariance matrix)을 사용하여 decoder Jacobian의 열(column) 사이의 직교성을 증진시키는 방법을 논의하고, 이를 통해 선형 독립성(linear independence)이 통계적 독립성(statistical independence)으로 어떻게 연결되는지를 설명하고 있습니다. 이와 함께 β-VAE에서의 β 값이 disentanglement와 posterior collapse에 미치는 영향을 설명합니다.

- **Performance Highlights**: 이 연구는 VAE의 성능 향상과 함께 disentanglement 현상이 어떻게 나타나는지를 자세히 이해하고, 새로운 조건을 제시하여 VAE가 데이터 분포를 완전히 식별하는 상황을 설명하였습니다. 이는 최신 diffusion 모델에서도 중요한 역할을 하며, 이론과 실제 모두에서 효과적인 응용을 기대할 수 있습니다.



### Auto-Intent: Automated Intent Discovery and Self-Exploration for Large Language Model Web Agents (https://arxiv.org/abs/2410.22552)
Comments:
          EMNLP 2024 Findings

- **What's New**: 본 논문에서는 Auto-Intent를 소개하며, 이는 학습하여 구축한 대형 언어 모델(LLM)을 직접적인 fine-tuning 없이 특정 도메인에 맞게 조정하는 방법입니다. 이 방법은 웹 탐색 작업을 중심으로 경험적으로 연구되었습니다.

- **Technical Details**: Auto-Intent는 대상 도메인의 시연에서 기본적인 의도(intents)를 비지도학습(unsupervised) 방식으로 발견합니다. 이 의도들은 최대 3단어로 매우 간결하게 표현되며, 이를 통해 다음 의도를 예측하는 의도 예측기(intent predictor)를 학습합니다. 특히 self-exploration 접근 방식을 통해, 가장 가능성이 높은 top-k 의도 예측 결과를 LLM 에이전트에게 제공하여 의사결정 능력을 향상시킵니다.

- **Performance Highlights**: Auto-Intent는 Mind2Web의 대규모 실 웹 탐색 벤치마크(Task benchmarks)와 WebArena의 온라인 탐색 작업에서 GPT-{3.5, 4}와 Llama-3.1-{70B, 405B} 에이전트의 성능을 실질적으로 향상시켰습니다.



### Attention Speaks Volumes: Localizing and Mitigating Bias in Language Models (https://arxiv.org/abs/2410.22517)
- **What's New**: 본 논문은 큰 언어 모델(LLMs)에서 애매한 비교 프롬프트가 주어졌을 때 어떻게 편향(bias)이 발생하는지를 탐구합니다. 새로운 방법으로, 편향을 특정 레이어에 국한시키고 주의(attention) 점수를 분석하여 이를 완화할 수 있는 기법인 $	exttt{ATLAS}$를 제안합니다.

- **Technical Details**: $	exttt{ATLAS}$는 두 단계의 접근법으로, 첫 번째로 주의 점수를 분석하여 편향이 집중된 레이어를 식별하고, 두 번째로 이러한 편향 레이어에 대해 주의 점수를 조정하여 편향을 줄이는 것입니다. 실험은 다양한 모델(GPT-2 XL, GPT-J, LLaMA-2 및 LLaMA-3)과 데이터셋(BBQ, Crows-Pairs, WinoGender)을 사용하여 수행되었습니다.

- **Performance Highlights**: 우리의 실험 결과, 편향은 모델의 후반 레이어에 집중되어 있으며, $	exttt{ATLAS}$는 다운스트림 성능에 영향을 주지 않으면서 편향을 효과적으로 완화하였습니다. 평균적으로 0.28 포인트의 편향 점수 개선이 이루어졌습니다.



### Privacy-Preserving Dynamic Assortment Selection (https://arxiv.org/abs/2410.22488)
- **What's New**: 본 논문은 사용자 맞춤형 추천에서 개인 정보 보호의 중요성을 강조하며, multinomial logit (MNL) bandits 모델을 활용하여 개인 정보를 보존하는 동적 상품 assortments 선택을 위한 새로운 프레임워크를 제안합니다. 이 접근법은 사용자 유틸리티 추정치에 보정된 노이즈를 통합하여 탐색(exploration)과 착취(exploitation) 간의 균형을 맞추는 동시에 강력한 프라이버시 보호를 보장합니다.

- **Technical Details**: 제안된 방법은 perturbed upper confidence bound (UCB) 방법을 사용하며, Joint Differential Privacy (JDP)를 만족하여 동적 환경에서 적합한 개인 정보 보호 방안을 제공합니다. 이 기법은 MNL bandits에 맞춰 설계된 새로운 목표 왜곡(objective perturbation) 기법에 기반하여 동작합니다. 이론적으로 제안된 정책의 거의 최적의 후회 경계(regret bound)인 \(	ilde{O}(	ext{sqrt}(T))\)를 도출하였고, 개인 정보 보호가 후회에 미치는 영향을 명확히 수량화하였습니다.

- **Performance Highlights**: 실험 및 Expedia 호텔 데이터 세트에 대한 적용을 통해, 제안된 방법이 기존 기준 방법에 비해 상당한 성능 향상을 달성함을 입증하였습니다. 이는 개인 맞춤형 추천의 정확성을 높이는 데 기여할 것으로 보입니다.



### Scaling LLM Inference with Optimized Sample Compute Allocation (https://arxiv.org/abs/2410.22480)
- **What's New**: 이 논문에서는 LLM(Large Language Model)의 샘플링 구성의 최적 분배를 찾기 위한 새로운 알고리즘 OSCA(Optimizes Sample Compute Allocation)를 제안합니다. OSCA는 다양한 샘플링 구성에 대한 샘플 예산을 최적화하여 더 효율적인 추론 성능을 보여줍니다.

- **Technical Details**: OSCA 알고리즘은 여러 샘플링 구성(모델, 온도 등)에 대해 각 구성에서 생성할 샘플의 수를 학습함으로써 최적의 예산 분배를 예측하는 방법으로 재미있는 방식에서 성능 향상을 가져옵니다. 특히, 작은 샘플 크기에서는 OSCA에서 학습된 할당이 순수 최적 할당보다 성능이 뛰어납니다.

- **Performance Highlights**: OSCA를 사용한 경우 코드 생성에서 128배, 4개의 이유(task) 과제에서 25배 적은 계산으로 더 높은 정확도를 달성했습니다. SWE-Bench에서 단일 턴 작업을 넘어 Agentic 워크플로우 개선에서도 우수한 성능을 보였습니다.



### Ethical Statistical Practice and Ethical AI (https://arxiv.org/abs/2410.22475)
Comments:
          10 pages; Preprint of submission to Proceedings of JSM 2024 Portland, OR

- **What's New**: 최근 인공지능(AI)의 빠른 발전과 함께 AI 시스템의 윤리적 개발 및 사용에 대한 사회적, 문화적, 산업적, 과학적, 정부적 우려가 증가하고 있습니다. 이러한 배경 속에서 ASA(American Statistical Association)는 윤리적 통계 관행 및 AI에 관한 성명을 발표했습니다.

- **Technical Details**: 이 논문에서는 인간의 권리 법률 및 컴퓨팅과 통계에 대한 윤리적 실천 기준에 기초하여 윤리적 통계 관행과 윤리적 AI를 지원하는 다양한 출처를 논의합니다. 이러한 출처 문서는 '윤리적 AI를 위한 통계 실무자를 위한 성명서'의 운영화를 강화하는 데 중요합니다.

- **Performance Highlights**: 독자들이 자신의 통계적 실천을 통해 AI를 개발하고 사용할 때 이끄는 데 도움을 줄 수 있는 여러 윤리적 통계 관행 및 AI에 대한 자원을 제시합니다.



### Image2Struct: Benchmarking Structure Extraction for Vision-Language Models (https://arxiv.org/abs/2410.22456)
Comments:
          NeurIPS 2024. First three authors contributed equally

- **What's New**: 이번 연구에서는 이미지로부터 구조를 추출하는 Vision-Language Models (VLMs)의 성능을 평가하기 위해 Image2Struct라는 새로운 벤치마크를 도입합니다. 이 벤치마크는 실제 사용 사례를 포착하며, 완전 자동화되어 사람의 판단이 필요 없고, 신선한 데이터의 흐름을 기반으로 합니다.

- **Technical Details**: Image2Struct는 입력 이미지를 기반으로 LaTeX 코드나 HTML과 같은 구조를 생성하게끔 VLM을 유도합니다. 시스템은 이미지를 다양하게 변환 가능한 알고리즘을 기반으로 하며, 3단계 프로세스를 통해 목적인 이미지를 비교하여 유사성 점수를 생성합니다. 이 벤치마크는 웹페이지, LaTeX 및 음악 점수의 세 가지 도메인에서 실행됩니다. 또한, 기존의 이미지 메트릭을 사용하여 이미지 간의 비교를 효율적으로 수행합니다.

- **Performance Highlights**: 14개의 주요 VLM을 평가한 결과, 모델에 따라 성능이 크게 차이를 보였습니다. 특히, 고급 API 모델들이 개방형 모델들보다 우수하며, GPT-4 Omni가 다양한 작업에서 가장 높은 평균 승률을 기록했습니다. 또한, 다양한 도메인에 따라 최상의 점수가 상이하며, 이는 Image2Struct가 여러 난이도의 작업을 포함하고 있음을 시사합니다.



### Addressing Issues with Working Memory in Video Object Segmentation (https://arxiv.org/abs/2410.22451)
Comments:
          12 pages, 11 figures

- **What's New**: 본 논문은 기존의 비디오 객체 분할 (VOS) 모델이 카메라 컷 및 비정상적인 장면 변화에 대해 더욱 강건해질 수 있도록 하는 간단한 알고리즘 변경을 제안합니다. 이 알고리즘은 메모리 버퍼 업데이트를 조정하고, 불필요한 프레임이 메모리에 저장되는 것을 방지하여 객체 추적 성능을 향상시킵니다.

- **Technical Details**: 기존 VOS 모델은 프레임 간 유사성을 기반으로 객체 마스크를 예측합니다. 그러나 카메라 컷과 같은 동작에 대해 정확성을 높이기 위해 L2 거리 기반의 이진 분류기를 도입하였습니다. 이를 통해 모델은 이미지 임베딩 간의 비정상적인 변경을 감지하고 불필요한 프레임을 메모리에 기록하지 않을 수 있습니다.

- **Performance Highlights**: 모델의 성능 실험 결과, 카메라 컷이 있는 비디오 데이터에서 성능이 상당히 향상되었습니다. 제안된 알고리즘 수정을 통해, 쓰기 메모리의 객체 추적 성능이 저하되지 않고 잘 유지되며, 정확한 객체 재식별 (re-identification)이 가능해졌습니다.



### Do Large Language Models Align with Core Mental Health Counseling Competencies? (https://arxiv.org/abs/2410.22446)
Comments:
          9 Pages, In Submission to NAACL 2025

- **What's New**: 본 논문에서는 CounselingBench라는 새로운 NCMHCE 기반 벤치마크를 소개하여 대규모 언어 모델(LLM)의 정신 건강 상담 능력을 평가합니다.

- **Technical Details**: 22개의 일반 목적 및 의료 전문화 LLM을 대상으로 하여 다섯 가지 핵심 정신 건강 상담 능력에 대한 평가를 진행했습니다.

- **Performance Highlights**: 최신 모델들이 최소 기준을 초과하는 성능을 보였으나, 전문가 수준의 성능에는 미치지 못하며, Intake, Assessment & Diagnosis에서는 우수한 성과를 보이는 반면, Core Counseling Attributes 및 Professional Practice & Ethics에서는 어려움을 겪는 경향이 있음을 발견했습니다.



### A Large Recurrent Action Model: xLSTM enables Fast Inference for Robotics Tasks (https://arxiv.org/abs/2410.22391)
- **What's New**: 이 연구에서는 최신 반복 아키텍처인 xLSTM을 중심으로 한 대규모 반복 행동 모델(Large Recurrent Action Model, LRAM)을 제안하였습니다. LRAM은 선형 시간 복잡성을 가지며, 기존의 Transformer 기반 모델에 비해 성능과 속도에서 유리한 결과를 보였다는 점이 특징입니다.

- **Technical Details**: LRAM은 432개 작업과 6개 도메인에서 훈련되었으며, 전문 학습 설정을 통해 수집된 대규모 다중 도메인 데이터셋에서 효과적으로 학습되었습니다. 또한, xLSTM과 Mamba와 같은 현대의 반복 아키텍처는 훈련 시 Transformer와 유사한 병렬화 특성을 가지면서 인퍼런스(inference)에서는 더 빠른 속도를 제공합니다.

- **Performance Highlights**: 실험 결과에 따르면 LRAM은 기존의 Transformer 모델에 비해 성능과 속도 모두에서 우수한 결과를 보여주었으며, 다중 작업, 미세 조정(fine-tuning), 문맥 학습(in-context learning) 능력에서도 뛰어난 성과를 나타냈습니다.



### FNDEX: Fake News and Doxxing Detection with Explainable AI (https://arxiv.org/abs/2410.22390)
- **What's New**: 본 논문은 기존에 널리 연구되었지만 상관관계가 명확히 분석되지 않은 가짜뉴스(fake news)와 도킹(doxxing) 간의 교차점을 탐구합니다. 이 연구는 세 개의 서로 다른 transformer 모델을 활용하여 둘 다 높은 성능으로 탐지하는 새로운 시스템인 FNDEX 시스템을 제안합니다.

- **Technical Details**: FNDEX 시스템은 가짜뉴스와 도킹 탐지를 위한 효과적인 접근법을 제시하며, 개인 식별 정보(PII)의 패턴 기반 익명화(anonymization) 프로세스를 포함합니다. 이 시스템은 다양한 transformer 모델을 평가하여 두 현상의 탐지 성능을 극대화합니다.

- **Performance Highlights**: 실험 결과, FNDEX 시스템은 기존의 기준선 모델에 비해 현저한 성과를 보이며, 가짜뉴스와 도킹 탐지에서 효과적인 결과를 도출하는 것을 입증했습니다.



### Debiasing Alternative Data for Credit Underwriting Using Causal Inferenc (https://arxiv.org/abs/2410.22382)
- **What's New**: 이 논문은 대체 데이터의 편향을 제거하기 위해 인과 추론 기법을 적용하는 방법을 제안합니다. 이를 통해 대출 심사에 대체 데이터를 활용하고자 하며, 인종 그룹 간의 모델 정확성을 향상시키고자 합니다.

- **Technical Details**: 전통적인 신용 점수는 과거 지불 이력, 채무 액수, 신용 이력의 길이와 같은 요소에 기반합니다. 논문에서는 인과 베이지안 네트워크 (Causal Bayesian Networks, CBNs)를 이용하여 공정성과 기계 학습의 관계를 설명하고, 대체 데이터의 편향을 제거하는 알고리즘을 제시합니다.

- **Performance Highlights**: 제안된 알고리즘은 공공 신용 데이터셋을 기반으로 테스트되었으며, 다양한 인종 그룹에 걸쳐 모델의 정확도가 향상되는 것을 보여주었습니다. 이 과정에서 비차별 보장(providing nondiscrimination guarantees)도 강조되었습니다.



### Robust training of implicit generative models for multivariate and heavy-tailed distributions with an invariant statistical loss (https://arxiv.org/abs/2410.22381)
- **What's New**: 본 논문에서는 기존의 전통적인 implicit generative models의 한계를 극복하기 위해 새로운 방법론인 Pareto-ISL을 제안합니다. 이 방법론은 heavy-tailed 분포를 효과적으로 모델링할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 기존의 invariant statistical loss (ISL) 방법을 확장하여 heavy-tailed 및 다변량 데이터 분포를 처리합니다. 특히, generalized Pareto distribution (GPD)에서 노이즈를 추출하여 생성모델을 학습시키며, 새로운 손실 함수와 알고리즘을 도입하여 높은 차원의 데이터도 처리할 수 있도록 개선하였습니다.

- **Performance Highlights**: Pareto-ISL은 heavy-tailed 분포의 중앙 및 끝부분을 정확히 모델링하며, 다변량 데이터에 대해서도 우수한 성능을 보여줍니다. 다양한 하이퍼파라미터 설정에서도 목표 분포의 완전한 지원을 캡처하는데 성공하며, pretrained generative adversarial networks (GANs)에 활용할 때 특히 강인한 성능을 발휘합니다.



### Discrete Modeling via Boundary Conditional Diffusion Processes (https://arxiv.org/abs/2410.22380)
Comments:
          NeuraIPS 2024 poster

- **What's New**: 이번 연구에서는 효율적이고 효과적으로 강력한 continuous diffusion processes를 discrete modeling에 확장하는 새로운 프레임워크를 제안합니다. 기존 방법들은 discrete 데이터와 continuous modeling 간의 불일치 문제를 겪었습니다.

- **Technical Details**: 우리는 먼저 경계를 prior distribution으로 추정한 후, forward trajectory를 재조정하여 boundary conditional diffusion model을 구축하는 두 단계의 forward 과정이 필요함을 제안합니다. 이 과정에서 Ordinary Differential Equations (ODEs)를 사용하여 forward trajectory를 설명합니다.

- **Performance Highlights**: 실험 결과, 우리의 방법은 언어 모델링과 discrete 이미지 생성 작업 모두에서 강력한 성능을 보였습니다. 특히, 언어 모델링에서는 기존 continuous diffusion 모델을 초월하여 우수한 성능을 보여주었으며, Cifar-10 데이터 세트에서 새로운 state-of-the-art를 수립했습니다.



### A Systematic Literature Review of Spatio-Temporal Graph Neural Network Models for Time Series Forecasting and Classification (https://arxiv.org/abs/2410.22377)
- **What's New**: 최근 연구에서는 spatio-temporal graph neural networks (GNNs)의 다양한 모델링 접근 방식과 응용 분야에 대한 포괄적 개요를 제공하고자 하는 체계적인 문헌 검토가 진행되었습니다. 이제까지의 문헌에서 150편 이상의 저널 논문을 조사하여, GNNs가 시계열 분류 및 예측에서의 효용을 체계적으로 비교했습니다.

- **Technical Details**: 이 리뷰는 GNNs의 시계열 관련 과제, 특히 시계열 분류와 예측에 중점을 두고 있습니다. GNN은 복잡한 관계를 포착할 수 있는 능력으로, 서로 다른 변수 간의 inter-variable 관계와 시간 간의 inter-temporal 관계를 동시에 캡처할 수 있습니다. 결과적으로, spatio-temporal GNNs는 공간 차원과 시간 차원을 모두 처리하도록 설계된 모델입니다.

- **Performance Highlights**: 제안된 다양한 GNN 모델과 벤치마크 모델의 결과를 포괄적으로 수집하여 향후 연구에 유용한 리소스와 참조 자료로 제공하고자 합니다. 이 리뷰는 현재 GNN 모델의 적용 분야와 성능에 대한 깊이 있는 통찰을 제공합니다.



### Rare-to-Frequent: Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with LLM Guidanc (https://arxiv.org/abs/2410.22376)
- **What's New**: 본 연구는 희귀한 개념에 대한 텍스트-이미지(T2I) 분산 모델의 구성능력을 높이기 위해 대형 언어 모델(LLM) 가이드를 활용하는 접근 방식을 제안합니다. 우리는 창의적이고 비범한 특성을 가진 새로운 캐릭터 디자인과 같은 희귀한 프롬프트 생성 과정에서 기존 모델들이 어려움을 겪는 문제를 강조합니다.

- **Technical Details**: 본 연구에서는 희귀한 개념 구성을 위한 새로운 접근 방식인 R2F(Rare-to-Frequent)를 제안합니다. R2F는 LLM을 활용하여 희귀한 컨셉과 관련된 보다 일반적인 빈번한 개념을 찾고, 이를 통해 분산 추론(difussion inference) 과정을 개선합니다.

- **Performance Highlights**: R2F는 다양한 희귀 개념 구성을 포함하는 새로운 벤치마크인 RareBench에서 기존 모델인 SD3.0 및 FLUX와 비교하여 T2I 정렬 정확도에서 최대 28.1%p 향상된 성능을 보였습니다.



### Rethinking Code Refinement: Learning to Judge Code Efficiency (https://arxiv.org/abs/2410.22375)
- **What's New**: 이 논문은 LLM (Large Language Models)이 생성한 코드가 항상 효율적이지 않다는 점을 강조하며, 코드 리팩토링 과정에서 효율성을 판단하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 제안하는 모델은 코드 언어 모델 (code language model)을 기반으로 하여, 두 개의 코드 (원본과 개선된 버전)의 효율성을 비교합니다. 이 모델은 코드 쌍이 주어졌을 때, 어떤 코드가 더 효율적인지 또는 개선된 코드의 효율성을 예측하는 방식으로 학습됩니다.

- **Performance Highlights**: 여러 프로그래밍 언어와 다양한 리팩토링 시나리오를 통해 검증한 결과, 제안한 효율성 판단 모델이 두 가지 다른 버전 중에서 더 효율적인 코드를 식별하는 데 있어 현저한 성과를 보였으며, 기존 방법들에 비해 우수한 성능을 나타냈습니다.



### Machine Unlearning using Forgetting Neural Networks (https://arxiv.org/abs/2410.22374)
- **What's New**: 본 논문에서는 머신 언러닝(Machine Unlearning) 문제를 해결하기 위해 Forgetting Neural Networks (FNN)을 제안합니다. FNN은 망각을 위한 특정 레이어를 포함하는 신경망으로, 인간의 망각 과정에서 영감을 받았습니다. 이전에 이론적으로만 제안된 FNN이 머신 언러닝 방법으로 실제로 활용된 것은 이번이 처음입니다.

- **Technical Details**: FNN은 Ebbinghaus 망각 곡선을 기반으로 한 망각 레이어를 포함하여, 머신 러닝 모델의 훈련 데이터 중 특정 데이터를 효과적으로 잊게 하도록 설계되었습니다. 본 연구에서는 MNIST 손글씨 숫자 인식과 패션 데이터셋을 이용하여 FNN을 구현하고, Membership Inference Attacks (MIA)를 통해 언러닝된 모델의 효과성을 평가했습니다.

- **Performance Highlights**: 실험 결과, 언러닝된 모델이 MIA 점수가 0.5에 가까운 값을 기록해, 테스트 데이터와 망각 데이터의 분포를 구분하지 못함을 나타내었습니다. 이는 제안된 방법이 머신 언러닝 문제를 처리하는 데 큰 잠재력이 있음을 보여줍니다.



### Analytic Continual Test-Time Adaptation for Multi-Modality Corruption (https://arxiv.org/abs/2410.22373)
- **What's New**: 이번 논문에서는 Multi-Modality Dynamic Analytic Adapter (MDAA)라는 새로운 접근 방식을 제안하여 Multi-Modal Continual Test-Time Adaptation (MM-CTTA) 과제에서 발생하는 오류 축적, 재앙적 망각(catastrophic forgetting), 신뢰성 편향(reliability bias)을 처리하는 방법을 제시합니다. 특히, Analytic Classifiers (AC), Dynamic Selection Mechanism (DSM), Soft Pseudo-label Strategy (SPS) 등을 도입하여 안정적인 모델 적응을 지원합니다.

- **Technical Details**: MDAA는 Analytic Classifiers (AC)를 통해 모델이 재앙적 망각을 방지하고, Dynamic Selection Mechanism (DSM)을 통해 각 AC의 출력 신뢰성을 기반으로 선택적으로 업데이트하여 신뢰성 편향을 완화합니다. Soft Pseudo-label Strategy (SPS)는 다양한 라벨에 대해 변동 확률을 부여하여 라벨 노이즈(Label noise)에 대한 모델의 강건성을 높입니다. MDAA는 복수 모드의 정보를 통합하여 다이나믹하게 필터링하며, 테스트 데이터에 적응합니다.

- **Performance Highlights**: 실험 결과, MDAA는 MM-CTTA 작업에서 최첨단 성능(SOTA)을 달성하며, 두 개의 과제에서 각각 이전 방법보다 최대 6.22% 및 6.84% 향상된 성능을 보였습니다.



### A Hierarchical Language Model For Interpretable Graph Reasoning (https://arxiv.org/abs/2410.22372)
- **What's New**: 이 논문에서는 그래프 작업을 위한 새로운 계층적 언어 모델인 HLM-G를 제안합니다. 이 모델은 노드 중심의 지역 정보와 상호작용 중심의 글로벌 구조를 효과적으로 캡처하기 위해 두 개의 블록 아키텍처를 사용합니다.

- **Technical Details**: HLM-G는 지역 블록과 글로벌 블록으로 구성되어 있으며, 각 블록은 특정한 attention masking을 사용합니다. 이 구조는 모델이 초기에는 로컬 정보를 캡처하고 후에 글로벌 정보를 통합하도록 허용하여 그래프 구조의 이해 능력을 향상시킵니다. 또한, LLM의 대규모 그래프 작업에 대한 계산 비용을 크게 줄입니다.

- **Performance Highlights**: HLM-G의 성능은 7개의 그래프 추론 데이터셋과 7개의 실제 데이터셋에서 종합 평가를 통해 검증되었습니다. 이 모델은 노드, 링크, 그래프 수준의 작업에서 효과적으로 일반화 할 수 있는 능력을 발휘하여 LLM을 그래프 기반 작업에 적용하는 데 있어 중요한 발전을 이루었습니다.



### Error Bounds for Deep Learning-based Uncertainty Propagation in SDEs (https://arxiv.org/abs/2410.22371)
Comments:
          pre-print under review

- **What's New**: 이 논문은 Stochastic Differential Equations (SDEs)에 의해 모델링된 과정의 확률 밀도 함수 (PDF)를 근사하는 방법을 제안합니다. 특히, Physics-Informed Neural Networks (PINNs)를 사용하여 근사 오차를 엄밀하게 바운딩하는 새로운 방법론을 개발했습니다.

- **Technical Details**: 연구의 주요 내용은 두 가지 PINNs를 사용하여 근사 오차의 재귀적 함수 시리즈를 통해 공간과 시간에 대해 타이트한 오차 경계를 설정하는 것입니다. 이 과정에서 PINN의 잔여 값이 Fokker-Planck 편미분 방정식 (FP-PDE)에 의해 지배된다는 통찰력을 제시했습니다.

- **Performance Highlights**: 수치 실험을 통해 제안된 방법론의 타당성을 검증했으며, 단일 PINN만으로도 실용적인 오차 바운드를 도출할 수 있음을 보여주었습니다. 이로 인해 복잡한 SDE 문제들을 다루는 데 있어 계산 효율성과 정확성을 함께 향상시키는 데 기여했습니다.



### Survey of User Interface Design and Interaction Techniques in Generative AI Applications (https://arxiv.org/abs/2410.22370)
- **What's New**: 이 논문은 현재의 인간-인공지능 상호작용 분야에서 사용자 인터페이스(UI) 디자인 및 사용자 상호작용 패턴에 대한 포괄적인 조사 결과를 제시합니다. 연구자들과 개발자들이 AI 애플리케이션을 설계하는 데 참고할 수 있는 다양한 사용자 상호작용 패턴을 문서화하고, 이를 통해 생성적 AI 애플리케이션 디자인의 진입장벽을 낮추는 것을 목표로 합니다.

- **Technical Details**: 이 설문조사는 사용자가 주도하는 상호작용(user-guided interactions)을 중심으로 진행되며, 이는 사용자가 의도적으로 생성적 AI 시스템에 영향을 미치는 모든 상호작용을 포함합니다. 구체적으로는 다양한 상호작용 기술과 이를 위한 UI 레이아웃, 사용자-AI 참여 수준을 분류하여 정리합니다. 또한, 텍스트 또는 이미지의 프롬프트(prompt)를 통해 사용자가 생성 모델을 제어하는 방식도 다룹니다.

- **Performance Highlights**: 이 연구는 100개 이상의 관련 생성적 AI 논문을 바탕으로 다양한 상호작용 기술과 UI 디자인 패턴을 카테고리화하였으며, 이는 디자이너들이 기존의 패턴을 활용하여 보다 직관적인 사용자 경험을 제공할 수 있도록 돕습니다. 연구 결과는 디자이너와 연구자 모두에게 생성적 AI 애플리케이션의 효과적인 설계를 위한 기초 자료가 될 것입니다.



### Project MPG: towards a generalized performance benchmark for LLM capabilities (https://arxiv.org/abs/2410.22368)
- **What's New**: 본 논문에서는 LLM(대규모 언어 모델)을 평가하기 위한 새로운 집계 방법인 'Project MPG'를 제안합니다. 이 방법은 모델 성능을 단일 지표로 통합하여 결정 과정에서 비전문가가 활용할 수 있도록 합니다.

- **Technical Details**: Project MPG는 'Goodness' 점수(정답 정확도)와 'Performance' 점수(초당 쿼리 수, QPS)라는 두 가지 주요 지표를 생성합니다. 이 지표는 다양한 오픈 벤치마크의 집계를 통해 도출되며, LLM에 대한 평가를 보다 신속하고 비용 효율적으로 수행할 수 있도록 설계되었습니다.

- **Performance Highlights**: 우리는 14개의 최첨단 모델에 대해 평가를 수행하였으며, 우리의 집계 지표가 Chatbot Arena 점수와 높은 상관관계를 보이는 것을 발견했습니다. 이는 기존의 MMLU 리더보드와 비교했을 때도 개선된 결과입니다.



### MAMMAL -- Molecular Aligned Multi-Modal Architecture and Languag (https://arxiv.org/abs/2410.22367)
- **What's New**: 이번 연구에서는 MAMMAL (Molecular Aligned Multi-Modal Architecture and Language)라는 새로운 모델을 소개합니다. 이 모델은 생물학적 데이터셋에서 학습하며, 여러 모달리티를 통합하여 다양한 약물 발견 관련 작업을 지원합니다. MAMMAL은 다중 작업을 처리할 수 있는 기반 모델을 통해 새로운 성능 기준(SOTA)을 달성했습니다.

- **Technical Details**: MAMMAL은 텍스트 생성, 회귀, 분류 등의 다양한 태스크를 처리하기 위해, 샘플 사이즈 20억의 대규모 데이터셋에서 훈련된 다중 정렬 모델입니다. 이 모델은 Transformer 아키텍처를 기반으로 하며, 모델 아키텍처는 인코더-디코더 또는 인코더 전용 모드로 작업을 수행할 수 있습니다. 새로운 '모듈형 토크나이저'를 통해 다양한 분자 도메인 어휘를 지원합니다.

- **Performance Highlights**: MAMMAL은 11개의 다양한 하위 작업을 평가한 결과, 9개 작업에서 새로운 SOTA를 달성하고, 나머지 2개 작업에서는 기존 SOTA와 비슷한 성능을 보였습니다. 이는 단일 통합 모델을 사용하여 모든 작업을 처리하고, 맞춤형 아키텍처 없이도 가능했습니다.



### Unpacking SDXL Turbo: Interpreting Text-to-Image Models with Sparse Autoencoders (https://arxiv.org/abs/2410.22366)
- **What's New**: 이 연구에서는 sparse autoencoders (SAEs)를 이용하여 텍스트-이미지 모델의 해석 가능한 특징을 학습할 수 있는 가능성을 조사하였습니다. 특히 SDXL Turbo와 같은 몇 단계의 diffusion 모델에서의 적용을 탐구하였습니다.

- **Technical Details**: SAEs는 SDXL Turbo의 denoising U-net 내에서 transformer 블록의 업데이트에 대해 훈련되었습니다. 이 과정에서 학습된 특징들은 해석 가능하고, 생성 과정에 인과적으로 영향을 미치며, 블록 간의 전문화(specialization)를 드러냈습니다.

- **Performance Highlights**: 연구에서는 이미지 구성, 지역 세부사항 추가, 색상 및 조명, 스타일을 담당하는 블록을 식별하였습니다. 이는 텍스트-이미지 생성 모델의 내부 구조를 이해하는 데 중요한 첫 걸음이 됩니다.



### Vascular Segmentation of Functional Ultrasound Images using Deep Learning (https://arxiv.org/abs/2410.22365)
- **What's New**: 이 논문은 기능적 초음파(fUS) 영상을 위한 최초의 심층 학습(segmentation based deep learning) 기반 세분화 도구를 소개합니다. 이 도구는 ULM(ultrasound localization microscopy) 자동 주석(annotation)을 이용하여 서로 다른 혈관 영역의 신호를 구별하고, 동적 혈관 용적(CBV) 정량화를 가능하게 합니다.

- **Technical Details**: fUS는 비침습적(imaging method) 이미징 방법으로 뇌혈관의 볼륨 변화를 고해상도(spatio-temporal resolution)로 측정합니다. 그러나 같은 픽셀 내에서 혈류 방향이 반대이기 때문에 소동맥(arterioles)과 소정맥(venules)을 구별하는 것이 어렵습니다. 저자는 다양한 UNet(architecture) 아키텍처를 평가하였고, 100개의 시간 프레임(temporal frames)만으로 90%의 정확도와 71%의 F1 점수, 0.59의 IoU를 성취했습니다.

- **Performance Highlights**: 렌즈 구조(segmentation)에서의 경쟁력을 보여주는 결과를 도출했으며, 시각적 자극(visual stimulation) 동안 캡처된 이미지에 잘 일반화됩니다. 이 연구는 fUS 데이터 해석을 향상시키고 혈관 기능(vessel function)에 대한 이해를 높이는 비용 효율적(alternative)인 방법을 제공합니다.



### MMM-RS: A Multi-modal, Multi-GSD, Multi-scene Remote Sensing Dataset and Benchmark for Text-to-Image Generation (https://arxiv.org/abs/2410.22362)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 본 논문에서는 다양한 원격 감지(RS) 시나리오에서 텍스트-이미지 생성이 가능하도록 다중 모달, 다중 GSD(ground sample distance) 및 다중 장면의 원격 감지 데이터셋(MMM-RS)을 제안합니다. 이 데이터셋은 기존의 9개의 공개 RS 데이터셋을 표준화하여 약 210만 개의 텍스트-이미지 쌍을 포함하고 있습니다.

- **Technical Details**: MMM-RS 데이터셋은 RGB, SAR(Synthetic Aperture Radar), NIR 근적외선의 세 가지 모달리티를 포함하며, 각 이미지는 정보가 풍부한 텍스트 프롬프트와 결합되어 있습니다. 이를 위하여 대규모 사전 훈련된 비전-언어 모델(BLIP-2)을 사용하여 자동적으로 텍스트 프롬프트를 생성하고, 수동으로 보정 작업을 수행하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 MMM-RS 데이터셋은 기존의 점착형(diffusion) 모델들이 다양한 RS 이미지를 생성할 수 있도록 지원하며, 다양한 모달리티, 장면 및 기상 조건에서 높은 성능을 보여주었습니다. 특히, 제안된 데이터셋을 통해 수치적 및 정성적 비교 실험을 수행하여 효과성을 입증하였습니다.



### Learning Goal-oriented Bimanual Dough Rolling Using Dynamic Heterogeneous Graph Based on Human Demonstration (https://arxiv.org/abs/2410.22355)
Comments:
          7 pages, 5 figures Accepted by IEEE ROBIO 2024 conference

- **What's New**: 본 연구는 목표 지향적인 소프트 객체 조작 정책을 학습하기 위한 동적 이종 그래프 기반 모델(Dynamic heterogeneous graph-based model)을 소개합니다. 이 모델은 상태와 정책 학습을 위한 통합된 그래프 표현을 활용하여 객체의 동적 변화를 효과적으로 캡처합니다.

- **Technical Details**: 제안하는 모델은 로봇의 행동과 상태 변화를 연결하여 목표를 달성하는 데 필요한 정책 학습을 지원합니다. 또한, 동적 그래프를 활용하여 객체의 역학 정보와 조작 정책을 추출하고, 이를 통해 시뮬레이터와 실제 로봇 환경에서의 정책 전이를 가능하게 합니다.

- **Performance Highlights**: 연구에서 제안된 방법은 반죽 굴리기 작업을 평가하며, 차별화된 시뮬레이터와 실제 인간형 로봇을 사용하여 실험을 진행했습니다. 다양한 ablation 연구를 통해 인간과 유사한 행동을 달성하는 데 있어 제안 방법의 우수성을 입증했습니다.



### RuleRAG: Rule-guided retrieval-augmented generation with language models for question answering (https://arxiv.org/abs/2410.22353)
- **What's New**: 이번 논문에서는 질의와 규칙을 활용한 Rule-Guided Retrieval-Augmented Generation (RuleRAG)이라는 새로운 QA 접근법을 제안합니다. 이 방법은 retriever가 논리적으로 관련된 문서를 검색하도록 가이드하고, generator가 규칙에 따라 답변을 생성할 수 있도록 합니다.

- **Technical Details**: RuleRAG는 두 가지 방식, 즉 RuleRAG-ICL과 RuleRAG-FT를 사용합니다. RuleRAG-ICL에서는 인-context learning을 통해 고신뢰의 규칙을 활용하여 더 나은 검색 결과를 얻고 보다 정확한 답변을 생성하도록 합니다. RuleRAG-FT는 supervised fine-tuning을 통해 retriever와 generator를 업데이트하여 규칙 기반의 작업 지시 능력을 개선합니다.

- **Performance Highlights**: 실험 결과, RuleRAG-ICL은 Recall@10 점수에서 +89.2%, exact match 점수에서 +103.1%의 성능 향상을 보여줍니다. 추가적인 fine-tuning을 통해 RuleRAG-FT는 더욱 뚜렷한 성능 개선을 달성할 수 있음을 나타냅니다.



### Neuromorphic Programming: Emerging Directions for Brain-Inspired Hardwar (https://arxiv.org/abs/2410.22352)
Comments:
          Accepted to International Conference on Neuromorphic Systems (ICONS) 2024. arXiv admin note: substantial text overlap with arXiv:2310.18260

- **What's New**: 본 논문은 뉴로모픽(Neuromorphic) 컴퓨팅의 문맥에서 프로그래밍을 재정의하는 개념적 분석을 제시합니다. 이는 기존의 프로그래밍 패러다임을 도전하고, 뉴로모픽 하드웨어의 물리적 특성과 결합된 보다 효과적인 접근법을 제안하는 새로운 프레임워크를 제안하고 있습니다.

- **Technical Details**: 뉴로모픽 컴퓨터는 생물학적 신경망 모델을 사용하여 구성됩니다. 이들은 이벤트 기반 센서와 대규모 뉴로모픽 프로세서 및 뇌-컴퓨터 인터페이스와 같은 요소들을 포함하며, 이는 정보 처리 방식, 즉 스파이크(spike) 기반 신경 알고리즘을 통해 이루어집니다. 뉴로모픽 프로그래밍은 전통적인 프로그래밍과 다르게 접근해야 하며, 그 근본적인 차이점들은 물리적 시스템과 연속 시간에서의 작동, 에너지 효율성, 그리고 스토캐스틱(stochastic) 컴퓨팅 등을 포함합니다.

- **Performance Highlights**: 기존 디지털 하드웨어에 적합한 프로그래밍 모델을 넓히는 동시에, 뉴로모픽 시스템의 고유한 적응성과 물리적 특성을 활용할 수 있는 새로운 모델을 제안합니다. 기존의 과정을 통해 새로운 하드웨어 클래스에 대한 풍부한 추상화를 요청하며, 현재 활용도가 낮은 고급 프로그래밍 모델을 통해 더욱 발전된 방향성을 모색합니다.



### Search Engines in an AI Era: The False Promise of Factual and Verifiable Source-Cited Responses (https://arxiv.org/abs/2410.22349)
- **What's New**: 본 논문은 대규모 언어 모델(LLM) 기반의 Answer Engine이 기존의 검색 엔진을 대체하는 것에 대한 연구 결과를 다루고 있습니다. 이러한 시스템은 정보를 검색하고 요약하여 제공하는 방식으로 발전하고 있으며, 그 한계와 개선 방안을 제안합니다.

- **Technical Details**: 연구에서는 21명의 참가자와 함께 Answer Engines와 전통적인 검색 엔진의 상호작용을 평가하고, 16가지의 Answer Engine의 한계를 식별했습니다. 그 후 8개의 지표와 연결된 16개의 디자인 권장 사항을 제안하였습니다. 이 연구는 일반적인 사용성 평가와 자동화된 평가를 통해 진행되었습니다.

- **Performance Highlights**: 자동 평가 결과, 세 가지 유명한 엔진(유채팅, 빙 코파일럿, 퍼플렉시 AI)의 한계(예: 잦은 환각, 부정확한 인용)를 수량화하였고, 모든 평가된 응답 엔진이 50-80%의 확률로 편향된 답변을 생성하는 것으로 나타났습니다. 이는 사용자의 의견에 동의하는 답변을 생성하는 경향이 있음을 나타냅니다.



### Testing GPT-4-o1-preview on math and science problems: A follow-up study (https://arxiv.org/abs/2410.22340)
- **What's New**: 2023년 9월에 GPT-4o1-preview 모델이 Wolfram Alpha 및 Code Interpreter 플러그인을 사용하여 105개의 중고등학교 및 대학 수준의 과학 및 수학 문제를 테스트한 결과를 발표했습니다. 이전 테스트 결과와 비교하여 성능이 크게 향상되었지만 여전히 완벽에는 미치지 못하는 것으로 나타났습니다. 특히 공간 추론(Spatial Reasoning) 관련 문제에서 어려움이 있었습니다.

- **Technical Details**: 105개의 문제는 세 가지 서로 다른 데이터셋으로 구성되어 있습니다: Arbitrary Numerical, Calculation-Free, Motivated Numerical. 각 문제는 Davis와 Aaronson의 관심과 취향을 반영하고 있습니다. 각각의 문제셋에서 GPT-4o1은 일부 문제를 오답으로 제공했으며, 특정 문제들에서는 공간 개념과 수학적 계산에서 오류를 범했습니다.

- **Performance Highlights**: GPT-4o1는 Arbitrary Numerical 문제에서 32개 중 24개 정답을 맞혔고, Calculation-Free 문제에서 53개 중 48개 정답을 맞혔습니다. Motivated Numerical 문제에서는 20개 중 18개 정답을 맞췄습니다. 총체적으로 성능이 향상되었으나, 특정 문제에서 여전히 오답을 내기도 하였습니다.



### DAWN: Designing Distributed Agents in a Worldwide Network (https://arxiv.org/abs/2410.22339)
- **What's New**: 최근 Large Language Models(LLMs)의 발전으로, 이들이 기초적인 대화 도구에서 복잡한 추론 및 의사결정이 가능한 정교한 존재로 변모하고 있습니다. 이로 인해 다양한 작업에 특화된 LLM 기반 에이전트들이 개발되었으며, 이들을 위한 전 세계적 협업을 지원하는 DAWN(Distributed Agents in a Worldwide Network) 프레임워크가 제안되었습니다.

- **Technical Details**: DAWN 프레임워크는 에이전트들이 전 세계에서 등록되고 발견될 수 있도록 Gateway Agents를 이용할 수 있으며, Principal Agent가 에이전트 간 협업을 조정합니다. DAWN의 세 가지 운영 모드(No-LLM Mode, Copilot Mode, LLM Agent Mode)는 각각의 작업 요구 사항에 맞춰 에이전트를 효과적으로 배치할 수 있도록 설계되었습니다. DAWN은 또한 안전하고 보안적인 에이전트 협업을 보장하는 전용 안전성, 보안 및 규정 준수 레이어를 도입했습니다.

- **Performance Highlights**: DAWN 프레임워크는 다양한 산업에서 에이전트 기반 애플리케이션을 배포할 수 있는 강력한 네트워크를 제공합니다. 이 프레임워크는 에이전트의 퍼포먼스를 극대화하며, 개인의 전문성 및 집단 지능을 활용하여 복잡한 작업 실행을 지원하고, 다양한 도구 및 시스템과의 원활한 통합을 가능하게 합니다.



### Robots Pre-train Robots: Manipulation-Centric Robotic Representation from Large-Scale Robot Datasets (https://arxiv.org/abs/2410.22325)
- **What's New**: 이번 논문에서는 Manipulation Centric Representation (MCR)라는 새로운 기반 표현 학습 프레임워크를 제안하여 로봇 조작 성능을 향상시키고 있습니다. MCR은 시각적 특징과 조작 작업의 행동 및 고유 환경 정보(고유 감각 정보)를 포착하여 조작 중심성을 개선합니다.

- **Technical Details**: MCR을 위해 DROID 로봇 데이터셋에서 시각 인코더를 사전 학습하고, 로봇의 고유 감각 상태와 행동 같은 동작 관련 데이터를 활용합니다. 핵심으로는 동적 정렬 손실(dynamics alignment loss)과 행동 예측 손실(action prediction loss)을 포함하며, 이를 통해 로봇 데이터셋의 동적 레이블을 효과적으로 활용합니다.

- **Performance Highlights**: MCR은 4개의 시뮬레이션 도메인에서 20개의 작업을 수행하며 이전 방법보다 14.8%의 성능 향상을 보여주었고, UR5e 관절 팔을 사용한 실제 3개의 작업에서는 76.9%의 성능 향상을 이루었습니다.



### PACER: Physics Informed Uncertainty Aware Climate Emulator (https://arxiv.org/abs/2410.21657)
- **What's New**: 본 논문에서는 PACER라는 경량의 기후 모사기 (climate emulator)를 제안하며, 이는 온실가스 배출 데이터에 기반하여 기온과 강수량을 86년 동안 안정적으로 모사할 수 있습니다. PACER는 물리 법칙인 advection-diffusion을 통합하고 이는 데이터 효율성을 높이는 데 기여합니다.

- **Technical Details**: PACER는 Neural ODE 기반으로, 온실가스 농도에 따라 동적으로 확산 계수(diffusion coefficient)와 흐름 속도(flow velocities)를 추정합니다. Gaussian noise를 stochastic term으로 도입하여 기후 모사에서의 불확실성을 고려하며, 지구 대기를 구형(domain of spherical)으로 모델링하여 주기적 경계 조건을 인코딩합니다.

- **Performance Highlights**: PACER는 15개의 기후 모델을 대상으로 실험을 수행하였으며, 대부분의 기후 모델에서 기준선 성능을 초월했습니다. PACER는 86년 동안 안정적으로 온도와 강수량을 모사하는 데 성공하여 기후 진단 작업(climate diagnostic task)에서 새로운 최첨단 성과를 달성했습니다.



New uploads on arXiv(cs.LG)

### Provable acceleration for diffusion models under minimal assumptions (https://arxiv.org/abs/2410.23285)
- **What's New**: 본 논문에서는 확률적 샘플러를 위한 새로운 훈련 없이도 가능한 가속화 방식이 제안되었습니다. 이 방식은 표준 score-based 샘플러의 높은 계산 부담을 줄이고 샘플링 속도를 개선하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 제안하는 가속화 샘플러는 $L^2$ 정확한 score 추정과 목표 분포에 대한 유한한 두 번째 모멘트 조건을 최소한의 가정으로 두고 있습니다. 이로써 $	ilde{O}(d^{5/4}/	ext{sqrt}(oldsymbol{	ext{ε}}))$ 반복 횟수 내에 총 변동에서 $oldsymbol{	ext{ε}}$-정확도를 보장할 수 있습니다. 기존 score-based 샘플러의 복잡도인 $	ilde{O}(d/oldsymbol{	ext{ε}})$를 상당히 개선하였습니다.

- **Performance Highlights**: 제안된 이론은 목표 분포에 대한 제한적인 가정이나 고차 score 추정 보장에 의존하지 않기 때문에, 더 다양한 분포에 대해 적용이 가능합니다.



### Multi-student Diffusion Distillation for Better One-step Generators (https://arxiv.org/abs/2410.23274)
Comments:
          Project page: this https URL

- **What's New**: Multi-Student Distillation (MSD)라는 새로운 프레임워크를 도입하여 기존의 단일 단계 확산 증류(diffusion distillation) 방법의 효율성을 개선했습니다.

- **Technical Details**: MSD는 조건부 교사 확산 모델을 여러 개의 단일 단계 생성기로 증류(dilute)하여 생성 품질을 향상시키는 방법입니다. 각 학생 생성기는 일부분의 조건 데이터를 처리하며, 빠른 추론 속도를 위한 소형 모델을 훈련시킵니다.

- **Performance Highlights**: 4개의 동등한 크기의 학생 모델을 사용하여, MSD는 ImageNet-64x64에서 1.20 FID, zero-shot COCO2014에서 8.20 FID라는 새로운 최첨단 성능을 달성했습니다.



### Proportional Fairness in Non-Centroid Clustering (https://arxiv.org/abs/2410.23273)
Comments:
          A preliminary version appeared at NeurIPS 2024

- **What's New**: 본 논문에서는 비 중심 클러스터링(non-centroid clustering)에서의 비례 공정성 보장(proportional fairness guarantees)을 확장하여 연구합니다. 중심 기반 클러스터링이 아닌 클러스터링의 손실 함수(loss function)를 다른 요인들을 고려하여 정의함으로써, 각 클러스터에 속한 에이전트(agent)들 간의 상호 작용을 반영하고자 합니다.

- **Technical Details**: 비례 공정성의 두 가지 기준인 core와 완전 정당화 표현(Fully Justified Representation, FJR)을 비 중심 클러스터링에 적응하여 적용했습니다. GreedyCapture 알고리즘을 비 중심 클러스터링에 적합시켜 근사화를 시도했으나, 이에 대한 효율성이 다소 떨어진다는 것을 보였습니다. 대신, 새로운 비효율적 알고리즘인 GreedyCohesiveClustering이 등장하며, 이는 임의의 손실 함수 아래에서 FJR을 정확하게 달성합니다.

- **Performance Highlights**: 실제 데이터 실험 결과에 따르면, 기존의 클러스터링 알고리즘들은 공정성이 매우 낮았으나, GreedyCapture 알고리즘은 훨씬 더 공정한 결과를 보였고, 기존 클러스터링 목표에서만 약간의 손실을 초래했습니다.



### A Monte Carlo Framework for Calibrated Uncertainty Estimation in Sequence Prediction (https://arxiv.org/abs/2410.23272)
- **What's New**: 이 논문은 고차원 입력 데이터(예: 이미지)로부터 분리된 시퀀스의 확률적 예측을 위한 몬테카를로(Monte Carlo) 프레임워크를 제안합니다. 특히 이 프레임워크는 확률 및 신뢰 구간(confidence interval) 추정을 가능하게 하여 예측의 불확실성을 정량화하기 위해 개발되었습니다.

- **Technical Details**: 제안된 foCus(framework for Monte Carlo uncertainty quantification of sequences)는 오토리그레시브(autoregressive) 모델을 사용하여 이미지 입력에 조건화된 시퀀스를 샘플링합니다. 이 과정에서 생성된 샘플들을 통해 각 상태가 발생할 확률을 추정하며, 훈련 시 열의 미스칼리브레이션(miscalibration) 문제를 개선하기 위해 시간 의존적인 정규화(time-dependent regularization) 방식을 도입했습니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 합성 데이터 및 실제 데이터에서 정확한 분별 예측(discriminative predictions)을 나타내었으나, 훈련된 오토리그레시브 시뮬레이터는 심각한 미스칼리브레이션 문제를 보였음을 확인했습니다. 이를 통해 시간 의존적인 정규화 방법이 더 나은 신뢰도 추정을 가능하게 하는 것을 보여주었습니다.



### Attribute-to-Delete: Machine Unlearning via Datamodel Matching (https://arxiv.org/abs/2410.23232)
- **What's New**: 이 논문에서는 새로운 기계 비학습(일명 machine unlearning) 기법인 Datamodel Matching (DMM)을 소개합니다. DMM은 비Convex (비볼록) 환경에서도 뛰어난 성능을 발휘하여, 기존의 알고리즘이 제대로 작동하지 않았던 문제를 해결합니다.

- **Technical Details**: DMM은 두 가지 단계로 이루어져 있습니다: (a) 데이터 귀속(data attribution)을 사용하여 '잊어야 할 데이터(forget set)'를 제외한 나머지 데이터로 재학습했을 때의 모델 출력 예측; (b) 예측된 출력과 일치하도록 미리 학습된 모델을 세부 조정(fine-tuning)하는 과정입니다. 이 방법은 Convex (볼록) 설정에서 다양한 반복적 비학습 알고리즘보다 확실히 우수한 성능을 보여줍니다.

- **Performance Highlights**: DMM은 기존 평가 방법과 새로운 KL-divergence 기반의 지표를 조합하여 비볼록 설정에서도 강력한 비학습 성능을 입증했습니다. 또한 DMM은 메타 알고리즘으로, 데이터 귀속 기술의 발전이 향후 비학습 알고리즘 개선으로 직결되므로 이 분야의 미래 진행 방향을 제시합니다.



### Emergence of meta-stable clustering in mean-field transformer models (https://arxiv.org/abs/2410.23228)
Comments:
          37 Pages, 6 figures

- **What's New**: 본 논문은 Transformer 레이어 내에서 토큰(token)의 진화를 연속 시간 흐름(continuous-time flow)으로 모델링하며, 이는 평균장 상호작용 입자 시스템(mean-field interacting particle system)에 의해 지배됩니다. 특히, 메타-안정 구간(meta-stable phases)과 클러스터링 현상(clustering phenomena)의 발생 및 지속성에 중점을 두고 수학적 조사를 수행합니다.

- **Technical Details**: 연구 결과는 미분 방정식(Partial Differential Equation, PDE) 기반으로 하며, 모델의 초기화(iid uniform initialization) 주위에서의 섭동 분석을 통해 토큰 수가 많아질 때 모델이 메타-안정 매니폴드(meta-stable manifold) 근처에 머무른다는 것을 입증하였습니다. 또한, 이 매니폴드를 인버스 온도 파라미터와 관련하여 명시적으로 찾았습니다.

- **Performance Highlights**: 본 연구는 클러스터링 동역학(clustering dynamics)과 관련된 중간 단계에서의 모델 행동을 특성화하였으며, 하이퍼파라미터(hyperparameters)와 모델 표현(retrospectives)의 관계를 명확히 하였습니다. 이로 인해, 다음 단어 예측(next-token prediction) 작업에서의 모델의 효과성과 복잡한 표현의 학습 가능성을 높였습니다.



### (FL)$^2$: Overcoming Few Labels in Federated Semi-Supervised Learning (https://arxiv.org/abs/2410.23227)
Comments:
          Accepted to NeurIPS 2024

- **What's New**: 이번 연구에서는 Federated Semi-Supervised Learning (FSSL)에서 발생하는 confirmation bias 문제를 해결하기 위해 (FL)²라는 새로운 방법론을 제안했습니다. 이 방법은 클라이언트별 적응형 thresholding과 sharpness-aware consistency regularization을 사용하여 unlabeled clients의 학습을 더욱 효과적으로 돕습니다.

- **Technical Details**: 제안된 (FL)²는 다음과 같은 기술적 요소를 포함합니다: (1) client-specific adaptive thresholding: 각 클라이언트의 학습 상태에 따라 pseudo-labeling threshold를 조정하고, (2) sharpness-aware consistency regularization: 원본과 perturb된 모델 출력 간의 일관성을 정규화함으로써 잘못된 pseudo-labeling의 영향을 줄입니다. 또한 (3) learning status-aware aggregation: 클라이언트의 학습 상태에 따라 aggregation weight를 조정하여, 더 어려운 데이터셋을 다루는 클라이언트를 보다 잘 반영합니다.

- **Performance Highlights**: 종단적 평가 결과, (FL)²는 기존의 FSSL 방법들과 비교하여 최대 23.0%의 분류 정확도를 향상시켰으며, 특히 레이블이 극히 제한적인 경우에 탁월한 성능을 보여주었습니다.



### COMAL: A Convergent Meta-Algorithm for Aligning LLMs with General Preferences (https://arxiv.org/abs/2410.23223)
- **What's New**: 본 논문에서는 보상 모델을 사용하는 기존의 알고리즘의 한계를 극복하기 위해 Convergent Meta Alignment Algorithm (COMAL)을 제안합니다. 이는 게임 이론의 수렴적 알고리즘에 영감을 받아 개발되었습니다.

- **Technical Details**: COMAL은 두 플레이어 제로섬 게임을 모델링하여 Nash equilibrium 정책에 도달하는 메타 알고리즘입니다. COMAL은 ProxProxoman_Prox 연산자를 기본 빌딩 블록으로 사용하여 모든 정책에 대해 50% 이상의 승률을 보장하는 robust alignment를 달성합니다.

- **Performance Highlights**: COMAL은 다양한 기존 알고리즘과 비교했을 때 마지막 반복(iterate)에서 Nash equilibrium에 수렴하는 유일한 알고리즘임을 실험적으로 입증하였으며, DPO 및 다양한 반복 알고리즘에 비해 항상 50%를 초과하는 승률을 기록했습니다.



### Partial Channel Dependence with Channel Masks for Time Series Foundation Models (https://arxiv.org/abs/2410.23222)
Comments:
          NeurIPS Workshop on Time Series in the Age of Large Models, 2024. Oral presentation

- **What's New**: 본 논문은 채널 간의 의존성을 보다 정교하게 조정할 수 있는 부분 채널 의존성(partial channel dependence, PCD) 개념을 도입합니다. 이전 연구들은 데이터셋 간의 명시적 이질성에 초점을 맞추었으나, 본 연구에서는 암묵적 이질성에 주목합니다.

- **Technical Details**: 우리는 PCD를 달성하기 위해 두 가지 주요 구성 요소로 구성된 채널 마스크(channel mask)를 제안합니다: 1) 채널 간의 상대적 의존성을 인코딩하는 상관 행렬(correlation matrix), 2) 각 데이터셋에 특화된 절대 의존성을 학습하는 도메인 파라미터(domain parameters), 이를 통해 상관 행렬을 정제합니다.

- **Performance Highlights**: 본 연구는 예측(forecasting), 분류(classification), 결측치 보간(imputation), 이상 탐지(anomaly detection) 등의 4가지 시간 시계열(task)에서 PCD의 효과를 검증하며, 몇 개의 샷(few-shot) 및 제로샷(zero-shot) 시나리오를 포함한 다양한 설정에서 우수한 성능을 보였습니다.



### Grounding by Trying: LLMs with Reinforcement Learning-Enhanced Retrieva (https://arxiv.org/abs/2410.23214)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 환각(hallucination) 문제를 해결하기 위해 정보 검색 기능을 통합하고, 이를 통해 모델의 응답을 실제 출처에 기반하도록 하는 새로운 강화 학습 프레임워크인 LeReT(Learning to Retrieve by Trying)를 소개합니다.

- **Technical Details**: LeReT는 다양한 검색 쿼리를 생성하여 LLM이 정보 검색에서 더 효과적인 쿼리를 학습할 수 있도록 하는 접근 방식을 사용합니다. 이 프레임워크는 우선 다양한 검색 쿼리를 시도한 후, 이러한 쿼리 중에서 유용한 결과를 가져오는 데 성공한 쿼리에 대해 보상을 주어 쿼리 품질을 향상시키는 방법입니다. LeReT는 다단계 검색에 적합하게 설계되어 있습니다. 또한, LeReT는 여러 검색 도구에서 쉽게 적용될 수 있습니다.

- **Performance Highlights**: LeReT는 두 개의 질문-답변 데이터 세트에서 리트리벌(리트리버) 정확도를 최대 29% 향상시키고, 하위 생성 평가에서 17% 개선된 성능을 보여주며, 특히 강력한 생성 모델(GPT-4와 같은)에서 그 혜택이 더욱 두드러졌습니다. LeReT의 성능은 반복적으로 개선될 수 있으며, 이는 다양한 쿼리 샘플링이 효과적이라는 실험 결과로 뒷받침됩니다.



### Kinetix: Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks (https://arxiv.org/abs/2410.23208)
Comments:
          The first two authors contributed equally. Project page located at: this https URL

- **What's New**: 이번 논문에서는 Kinetix라는 새로운 물리 기반 RL 환경을 도입하여 강화 학습 에이전트를 훈련시키는 최신 접근법을 다룹니다. Kinetix는 다양한 2D 물리적 작업을 절차적으로 생성하여, 에이전트가 미지의 환경에서도 성공적으로 작업을 수행할 수 있도록 하는 데 초점을 맞추고 있습니다.

- **Technical Details**: Kinetix는 하드웨어 가속 물리 엔진인 Jax2D를 활용하여 수십억 단계의 환경 상호작용을 효과적으로 시뮬레이션합니다. 에이전트는 Kinetix의 환경을 샘플링하여 훈련을 진행하며, 이러한 환경은 로봇의 이동, 잡기 작업에서부터 비디오 게임, 전통적인 RL 환경까지 다양합니다.

- **Performance Highlights**: 훈련된 에이전트는 인간 설계의 미지의 환경을 제로샷(zero-shot)으로 해결할 수 있는 강력한 물리적 추론 능력을 보여주었습니다. 또한, 관심 있는 특정 작업에 대한 파인튜닝(Fine-tuning)을 통해 RL 에이전트를 처음부터 학습시키는 것과 비교하여 현저하게 향상된 성능을 발휘했습니다. 이는 기존 RL 훈련에서는 실패하는 작업도 포함됩니다.



### ProTransformer: Robustify Transformers via Plug-and-Play Paradigm (https://arxiv.org/abs/2410.23182)
- **What's New**: 이 논문에서는 트랜스포머 기반 아키텍처의 강인성을 향상시키기 위해 설계된 새로운 강력한 주의 기법을 소개합니다. 이 기술은 기존 트랜스포머에 플러그 앤 플레이 (plug-and-play) 레이어로 통합할 수 있어 추가적인 훈련이나 파인 튜닝 없이도 강인성을 개선할 수 있습니다.

- **Technical Details**: ProTransformer는 주의 메커니즘과 가중 최소 제곱 추정기 (weighted least square estimator) 간의 새로운 연결을 확립하며, 이로부터 강인한 토큰 추정기를 제안하여 적대적 공격에 대한 토큰 집계의 회복력을 향상시킵니다. 제안된 Newton-IRLS 알고리즘은 비볼록 및 비매끄러운 강력한 토큰 추정기를 근사화합니다.

- **Performance Highlights**: ProTransformer는 BERT, ALBERT, DistilBERT 및 RoBERTa에서 각각 19.5%, 28.3%, 16.1% 및 11.4%의 성능 향상을 보여주며, 대형 언어 모델인 T5 및 LLaMA에서도 24.8% 및 17.8%의 성능 향상을 나타냅니다. 추가적으로, ProTransformer는 비전 및 그래프 도메인에서도 우수한 강인성을 보여줍니다.



### Does equivariance matter at scale? (https://arxiv.org/abs/2410.23179)
- **What's New**: 본 연구는 대규모 데이터 세트와 충분한 계산 리소스가 있는 상황에서, 각 문제의 구조와 대칭성을 반영한 신경망(neural network) 아키텍처를 설계하는 것이 유리한지, 아니면 데이터를 통해 학습하는 것이 더 효율적인지에 대한 실증적 연구를 진행했습니다. 이 작업은 강한 유도 편향(strong inductive biases)의 중요성을 조명합니다.

- **Technical Details**: 우리는 단단한 물체(rigid body) 간 상호작용의 벤치마크 문제를 중점적으로 다루며, 일반 목적의 transformer 아키텍처를 실험합니다. 비대칭(non-equivariant)과 대칭(equivariant) 네트워크의 성능을 다양한 모델 크기, 훈련 단계(training steps), 데이터셋 크기를 조절하여 비교했습니다. 대칭 변환을 지닌 E(3)-대칭 변환기(E(3)-equivariant transformer)와 표준 transformer 아키텍처의 비교를 통해, 훈련용 계산 예산(compute budget)의 최적 분배 및 데이터 보강(data augmentation)의 영향을 분석했습니다.

- **Performance Highlights**: 연구 결과, 대칭성이 데이터 효율(data efficiency)을 향상시키며, 데이터 보강을 통해 비대칭 모델과의 성능 차이를 줄일 수 있음을 발견했습니다. 대칭 변환기는 모든 계산 예산에서 더 나은 계산 효율(compute efficiency)을 보였고, 모든 모델 클래스는 거듭되는 훈련 단계에 따른 파워 법칙(power law) 스케일링 특성을 보였습니다. 아울러, 훈련 예산을 모델 크기와 훈련 단계에 어떻게 할당해야 하는지는 대칭성과 비대칭 모델에 따라 달라지는 것이 확인되었습니다.



### The Persistence of Neural Collapse Despite Low-Rank Bias: An Analytic Perspective Through Unconstrained Features (https://arxiv.org/abs/2410.23169)
Comments:
          40 pages

- **What's New**: 현대 심층 신경망(DNN)의 최종 레이어에서 관찰된 간단한 구조가 '신경 붕괴(Neural Collapse)' 현상으로 나타나며, 이는 심층 레이어에서도 관찰된다. 특히, 이 연구는 교차 엔트로피 손실(Cross-Entropy Loss)에 미치는 저순위 편향(Low-Rank Bias)을 분석한다.

- **Technical Details**: 신경 붕괴(Neural Collapse)는 주로 over-parameterized 네트워크의 마지막 레이어에서 발생하며, 다양한 조건에서 최적점을 나타낸다. 이 연구에서는 교차 엔트로피 손실을 가지는 선형 레이어의 깊은 비제약 특성 모델을 사용하여 분석하였다. 연구는 저순위 편향이 전역 최적에서의 가중치의 특이값(Singular Values) 구조에 미치는 영향을 조사하였다.

- **Performance Highlights**: 복잡한 라인(Strain) 구조를 통한 손실 표면의 분석 결과, DNC 구조가 자주 관찰되는 이유가 손실 표면에서의 높은 퇴화(Degeneracy)에서 비롯되는 가능성이 제시되었다. 이 연구는 실제 DNN에서 저순위 솔루션이 관찰되는 경향을 보여주며, 이는 심층 비제약 특성 모델을 통해 설명 가능하다.



### TokenFormer: Rethinking Transformer Scaling with Tokenized Model Parameters (https://arxiv.org/abs/2410.23168)
- **What's New**: 이번 논문에서는 TokenFormer라는 새로운 아키텍처를 제안했습니다. 이 아키텍처는 attention 메커니즘을 활용하여 모델 파라미터와의 상호 작용을 통합하여 유연성을 증대시키며, 전체 모델을 처음부터 다시 훈련할 필요 없이 점진적인 스케일링을 가능하게 합니다.

- **Technical Details**: TokenFormer는 Transformer의 전통적인 linear projections를 대체하여 토큰과 모델 파라미터 간의 관계를 attention에 기반하여 재구성합니다. 모델 파라미터를 토큰으로 처리하며, 입력 토큰은 queries로, 모델 파라미터는 keys와 values로 작용합니다. 이를 통해 파라미터를 점진적으로 추가하고 효율적으로 확장할 수 있습니다.

- **Performance Highlights**: TokenFormer는 124M의 파라미터에서 시작하여 1.4B까지 확장할 수 있으며, 처음부터 훈련한 Transformer와 유사한 성능을 달성하면서 훈련 비용을 절반 이상 줄이는 효과를 가지고 있습니다.



### FlexTSF: A Universal Forecasting Model for Time Series with Variable Regularities (https://arxiv.org/abs/2410.23160)
- **What's New**: 본 논문에서는 시간 시계열 예측을 위한 새로운 통합 모델인 FlexTSF를 제안합니다. FlexTSF는 다양한 도메인과 구조적 다양성을 효과적으로 처리할 수 있도록 설계되었습니다.

- **Technical Details**: FlexTSF는 세 가지 주요 혁신 설계를 포함합니다: VT-Norm(정규화 전략), IVP Patcher(패칭 모듈) 및 LED Attention(주의 메커니즘). VT-Norm은 서로 다른 특성을 가진 데이터를 표준화하여 동적 시간 패턴을 학습하는 데 중점을 두고, IVP Patcher는 다양한 구조적 시간 시리즈를 위한 연속적 패칭을 구현합니다. 마지막으로, LED Attention은 예측을 위한 자동회귀 프로세스에서 이러한 요소들을 통합하여 도메인과 시간 정보에 대한 인식을 제공합니다.

- **Performance Highlights**: FlexTSF는 12개의 다양한 데이터셋에서 기존의 최첨단 예측 모델보다 우수한 성능을 보였으며, 자체 지도 학습(pre-training) 후 제로샷(zero-shot) 및 몇 샷(few-shot) 설정에서도 뛰어난 예측력을 보여줍니다.



### Directional anomaly detection (https://arxiv.org/abs/2410.23158)
- **What's New**: 이 논문은 비지도 이상 탐지(Semi-supervised anomaly detection) 분야에 새로운 거리 측정 방법을 제안합니다. 이는 맥락 정보, 특히 각 속성이 고가치(high attribute values) 일 때만 이상으로 간주해야 한다는 도메인 지식을 활용하는 방향 이상 탐지(directional anomaly detection)를 다룹니다.

- **Technical Details**: 제안된 방법은 Ramp distance와 Signed distance라는 두 가지 비대칭 거리 측정 방법입니다. Nearest Neighbour Distance(NND) 및 Average Localised Proximity(ALP) 알고리즘을 수정하여 방향성을 고려합니다. 이 모델은 특성 공간(feature space)에서 정상 학습 데이터와의 거리를 기반으로 아노말리 점수를 부여합니다.

- **Performance Highlights**: Ramp distance는 전통적인 절대 거리보다 성능이 우수한 것으로 나타났습니다. 반면, Signed distance는 합성 데이터(synthetic data)에서는 좋은 결과를 보였지만, 실제 데이터(real-life datasets)에서는 상대적으로 성능이 저조했습니다. 이는 특정 속성에서 나쁜 점수가 있을 경우, 다른 속성에서 좋은 점수가 이를 보완해서는 안 된다는 점을 강조합니다.



### QWO: Speeding Up Permutation-Based Causal Discovery in LiGAMs (https://arxiv.org/abs/2410.23155)
Comments:
          21 pages, 4 figures

- **What's New**: 이 논문은 변수 간의 인과 관계를 이해하기 위한 인과 발견(causal discovery) 방법에 중점을 두고 있으며, Linear Gaussian Acyclic Models (LiGAMs)에서 인과 그래프(causal graph)를 학습하기 위한 새로운 방법인 QWO(Quantile Weighted Orthogonality)를 소개합니다. 기존의 인과 그래프 학습 방법의 계산 복잡성을 낮추는 것에 중점을 두었습니다.

- **Technical Details**: QWO는 주어진 순열(permutation) π에 대해 $m{	ext{G}}^{m{	ext{π}}}$의 계산을 크게 향상시킵니다. QWO는 기존 BIC 기반 방법에 비해 O(n²) 만큼 빠르며, 이는 변수의 수 n에만 의존하며 데이터 포인트 수 N에 대해서는 독립적입니다. 또한, QWO는 GRASP 및 힐 클라이밍 기반 검색 방법과 통합 가능하여, 이러한 기존 방법들의 성능을 향상시킬 수 있습니다.

- **Performance Highlights**: QWO는 이론적으로 LiGAM 모델에 대한 정확한 학습을 보장하며, 충분한 샘플이 제공되는 한 주어진 순열 π에 대해 정확한 결과를 도출합니다. 실험에서 QWO는 GRaSP 및 힐 클라이밍 기반 검색 기술과 결합되어 시간 복잡성 측면에서 우수한 성능을 보여주었습니다.



### HiBO: Hierarchical Bayesian Optimization via Adaptive Search Space Partitioning (https://arxiv.org/abs/2410.23148)
- **What's New**: 본 논문에서는 Hierarchical Bayesian Optimization (HiBO)라는 새로운 계층적 알고리즘을 소개합니다. 이 알고리즘은 전 세계 검색 공간 파티셔닝 정보를 최신 Bayesian Optimization (BO) 모델링의 탐색 전략에 통합하여, 고차원 설계 영역에서의 문제 해결을 개선합니다. HiBO는 검색 트리 기반의 글로벌 네비게이터를 사용하여 검색 공간을 샘플링 잠재력이 다른 파티션으로 적응적으로 나눕니다.

- **Technical Details**: HiBO의 핵심 구성 요소는 글로벌 네비게이터와 로컬 BO 최적화기입니다. 글로벌 네비게이터는 검색 공간을 여러 개의 파티션으로 나누고, 각 파티션의 샘플링 잠재력을 평가합니다. 로컬 최적화기는 이 정보를 활용하여 샘플링을 가장 유망한 영역으로 편향시킴으로써 더욱 효율적인 샘플링을 촉진합니다. 이 과정에서 탐색과 활용의 균형을 맞추기 위한 적응적 검색 트리 구성 전략도 포함됩니다.

- **Performance Highlights**: HiBO는 고차원 합성 벤치마크에서 최신 방법론들과 비교했을 때 우수한 성능을 보여주며, 데이터베이스 관리 시스템(DBMS)의 구성 조정과 같은 실제 세계 최적화 작업에서도 그 효과를 입증하였습니다. HiBO는 기존 기술에 비해 더 나은 샘플링 결정을 제공하며, 불필요한 계산 비용을 줄이고 효율적인 고차원 최적화를 달성합니다.



### FoLDTree: A ULDA-Based Decision Tree Framework for Efficient Oblique Splits and Feature Selection (https://arxiv.org/abs/2410.23147)
- **What's New**: LDATree와 FoLDTree라는 두 가지 새로운 프레임워크가 제안되었습니다. 이 프레임워크는 Uncorrelated Linear Discriminant Analysis (ULDA) 및 Forward ULDA를 결정 트리 구조와 통합하여 효율적인 oblique splits를 가능하게 합니다.

- **Technical Details**: LDATree와 FoLDTree는 결측값을 처리할 수 있고, 피처 선택을 지원하며, 클래스 레이블과 확률을 모델 출력으로 제공합니다. 이들은 고차원 데이터에서 유용하게 작용하며, 전통적인 단일 트리 기법에 대한 강력한 대안으로 자리잡을 가능성이 큽니다.

- **Performance Highlights**: LDATree와 FoLDTree는 시뮬레이션된 데이터셋과 실제 데이터셋에서 평가되었으며, axis-orthogonal 및 다른 oblique 결정 트리 방법을 일관되게 초월했습니다. 이들은 random forest와 비슷한 수준의 정확도를 달성하였습니다.



### FAIR-TAT: Improving Model Fairness Using Targeted Adversarial Training (https://arxiv.org/abs/2410.23142)
- **What's New**: 이번 연구에서는 공정한 타겟 적대 훈련(Fair Targeted Adversarial Training, FAIR-TAT)이라는 새로운 접근법을 도입하여 적대적 훈련 중 클래스 간 불균형과 모델의 공정성을 개선하는 것을 목표로 하고 있습니다. 구체적으로, 대상 적대적 공격을 통해 클래스별로 훈련 목표를 조정하는 방식을 채택하였습니다.

- **Technical Details**: FAIR-TAT는 전통적인 적대적 훈련(Adversarial Training, AT)에서 비대상 적대적 공격이 아닌 대상 적대적 공격을 사용하여 훈련 진행 중 클래스 간 편향을 동적으로 모니터링하고 조정합니다. 이를 통해 더 높은 클래스 정확도를 달성하고 클래스별 성능 불균형을 감소시킵니다. 또한, 모델의 훈련 과정에서 관찰된 클래스 간 오류율을 기반으로 타겟 클래스를 선택하는 방법을 사용합니다.

- **Performance Highlights**: FAIR-TAT는 기존의 최첨단 방법들에 비해 공정성을 개선하는 데 성공하였으며, 공격을 받는 클래스뿐만 아니라 일반적인 왜곡에 대해서도 높은 성능을 유지함으로써 전체적인 강건성을 보장합니다. 실험 결과, 공정한 타겟 적대적 훈련을 통해 난이도가 높은 클래스에서의 정확도가 향상되었습니다.



### Federated Learning under Periodic Client Participation and Heterogeneous Data: A New Communication-Efficient Algorithm and Analysis (https://arxiv.org/abs/2410.23131)
Comments:
          Neurips 2024

- **What's New**: 본 논문에서는 비선형 최적화(nonconvex optimization) 문제에서 클라이언트의 참여 패턴을 고려하여 새로운 알고리즘인 Amplified SCAFFOLD를 제안합니다. 이는 기존의 연구들이 가정하는 것보다 더 현실적인 참여 패턴을 사용하였습니다.

- **Technical Details**: 제안된 Amplified SCAFFOLD 알고리즘은 모든 클라이언트에 대해 고정된 라운드(window) 동안 참여 확률이 동등하다는 전제를 기반으로 하며, 통계적 비선형(non-convex stochastic) 환경에서 $	ext{O}(	ext{ε}^{-2})$의 통신 라운드를 요구합니다. 이는 기존 연구의 $	ext{O}(	ext{κ}^2 	ext{ε}^{-4})$ 통신 라운드에 비해 현저히 감소되었습니다. 여기서 $	ext{κ}$는 데이터의 이질성을 나타냅니다.

- **Performance Highlights**: 실험 결과, 이 알고리즘은 주기적으로 참여하는 클라이언트가 있는 환경에서 효과적임을 입증하였으며, (1) 합성 데이터(synthetic data) 및 (2) 클라이언트 수가 많은 실제 데이터(real-world data, N = 250)에서 실험을 수행하였습니다.



### Why Fine-grained Labels in Pretraining Benefit Generalization? (https://arxiv.org/abs/2410.23129)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2303.16887

- **What's New**: 이 논문은 심층 신경망(deep neural network)의 사전 훈련(pretraining)에서 미세하게 레이블이 지정된 데이터(fine-grained labeled data)를 사용하고, 이를 통해 파인 튜닝(fine-tuning)을 하는 기법을 제안합니다. 이는 일반적인 레이블이 붙은 데이터(coarse-labeled data)에 비해 더 나은 일반화(generalization)을 제공합니다.

- **Technical Details**: 이 연구는 '계층적 다중 관점(hierarchical multi-view)' 구조를 도입하여 입력 데이터 분포를 제한합니다. 이를 통해 코스 그레인(pretraining with coarse-grained data)은 신경망이 공통 특징(common features)만을 학습하도록 허용하고, 미세 그레인(pretraining with fine-grained data)은 공통 특징과 드문 특징(rare features)을 모두 학습하게 하여, 어려운 다운스트림 테스트 샘플(test samples)에서 향상된 정확도를 제공합니다.

- **Performance Highlights**: 미세하게 레이블이 지정된 데이터로 사전 훈련한 경우, 신경망은 더 나은 정확도를 보여주며, 특히 어려운 샘플에서의 성능이 개선됨을 논문의 결과에서 확인할 수 있습니다.



### Why Gradient Subspace? Identifying and Mitigating LoRA's Bottlenecks in Federated Fine-Tuning of Large Language Models (https://arxiv.org/abs/2410.23111)
Comments:
          24 pages, 10 figures, pre-print

- **What's New**: 이 논문은 Federated Learning (FL) 환경에서 Large Language Models (LLMs)의 성능을 향상시킬 수 있는 새로운 전략을 제안합니다. 특히, Low-Rank Adaptation (LoRA) 기반의 접근 방식이 가진 한계를 분석하고, 대안으로 직접적인 가중치 평균화 전략이 보다 나은 결과를 보여줄 수 있음을 밝힙니다.

- **Technical Details**: FL에서 LLMs의 파라미터 효율적인 미세 조정 방법으로 LoRA가 널리 사용되고 있으나, 이 접근 방식은 저차원 행렬에 대한 제한된 서브스페이스 학습 때문에 최적이 아니며, 실험적으로 GaLore와 같은 그라디언트 저차원 최적화 방법이 더 효과적임을 입증합니다.

- **Performance Highlights**: 직접적인 가중치 집합을 사용하는 전략이 FL 환경에서 LoRA 기반 전략보다 우수한 성능을 보이며, GaLore는 FlexLoRA 및 FFA-LoRA와 같은 최신 LoRA 방법을 능가하는 성과를 보여줍니다. 이를 통해 FL에서의 LLM 미세 조정 성능을 극대화하는 새로운 혁신적 접근 방식을 제안합니다.



### Controllable Game Level Generation: Assessing the Effect of Negative Examples in GAN Models (https://arxiv.org/abs/2410.23108)
- **What's New**: 이번 연구에서는 CGAN(Conditional Generative Adversarial Networks)과 Rumi-GAN 두 가지 제어 가능한 GAN 변형 모델을 비교하고, 게임 레벨 생성 중의 특정 제약 조건인 '플레이 가능성(playability)' 및 '제어 가능성(controllability)'의 충족 여부에 대한 성능을 평가했습니다. 특히, Rumi-GAN은 GAN 훈련시 '부정적 예제(negative examples)'를 활용하여 긍정적 예제를 효과적으로 학습하는 새로운 접근 방식을 제시했습니다.

- **Technical Details**: 이 연구는 'Deep Convolutional GAN' 아키텍처를 사용하여 훈련된 CGAN과 Rumi-GAN 모델을 기반으로 합니다. 훈련 과정에서는 각각의 모델이 긍정적 및 부정적 예제를 사용하여 특정 게임 레벨의 제약 조건을 적용할 수 있도록 하였습니다. Rumi-GAN의 경우, 손실 함수가 특정 조건을 충족하는 레벨 생성(segment generation)을 촉진하고, 조건을 충족하지 않는 레벨 생성을 억제합니다.

- **Performance Highlights**: 부정적 예제를 포함한 훈련이 GAN 모델의 특정 제약 조건 충족, 특히 플레이 가능성에서 긍정적인 영향을 미친다는 결과를 도출했습니다. 이 연구는 다양한 제어 가능한 GAN 모델의 비교 분석을 통해 게임 레벨 생성에 있어 긍정적 예제와 부정적 예제의 통합의 효과를 최초로 보여주었습니다.



### Controlling Language and Diffusion Models by Transporting Activations (https://arxiv.org/abs/2410.23054)
- **What's New**: 이 논문에서는 Activation Transport (AcT)라는 새로운 프레임워크를 소개하며, 이는 Optimal Transport(OT) 이론에 기반하여 모델의 활성화를 유도할 수 있는 방법론입니다. 이 기법은 기존의 활성화 조정 방법들을 통합적으로 설명하며, 대조군 언어 또는 서로 다른 스타일의 텍스트에서의 변환을 효과적으로 수행할 수 있습니다.

- **Technical Details**: AcT는 모달리티에 구애받지 않으며, 모델의 내부 활성화 분포를 보존하면서도 미세한 조정을 가능하게 합니다. 이 방법은 λ (lambda)라는 강도 매개변수를 이용해 개입 정도를 조절할 수 있으며, 0에서 1 사이의 값으로 설정하여 부분적 또는 전체적 변환을 적용합니다. 특히, 저자는 Linear-AcT 방식이 기존의 개입 방법들과 비교하여 더 나은 성과를 낼 수 있음을 실험적으로 입증했습니다.

- **Performance Highlights**: AcT는 Large Language Models (LLMs)에서 독성 감소, 개념 유도, 진실성 증가를 효과적으로 범위 내에서 수행하며, Text-to-Image (T2I) 모델에서도 미세한 스타일 조정 및 개념 부정이 가능함을 보여줍니다. 이 연구는 LLMs와 확산 모델에서 동시에 효과적인 개입 방법을 적용한 최초의 사례로 자리잡고 있습니다.



### Legitimate ground-truth-free metrics for deep uncertainty classification scoring (https://arxiv.org/abs/2410.23046)
- **What's New**: 본 논문은 Uncertainty Quantification (UQ) 방법의 생산에서의 제한된 사용과 이를 해결할 수 있는 질적 지표를 제안합니다. 저자들은 예측 신뢰도와 관련된 해석 가능한 불확실성 기준을 제시하면서 UQ의 포괄적인 사용을 촉진하고자 합니다.

- **Technical Details**: 이 논문은 특히 UQ 메트릭스가 잘 작동하며, 이는 모델 예측 신뢰도 순위와 관련이 있음을 보여줍니다. 이를 위해, 스코어링 함수 s(𝐱)와 같은 불확실성 확인 기준을 통해 Bayes 분류기의 정확도를 엄격히 평가할 수 있는 방법을 제안합니다. 주로 테스트 데이터가 주어진 상황에서도 의미 있는 UQ를 평가할 수 있도록 합니다.

- **Performance Highlights**: 저자들은 새로운 메트릭이 UQ의 적용 가능성을 높이고, 특히 딥러닝 분야에서 불확실성 평가를 적극적으로 촉진할 것이라고 주장합니다. 이는 실용적인 UQ 접근 방식의 개발로 이어질 것으로 기대됩니다.



### Toward Understanding In-context vs. In-weight Learning (https://arxiv.org/abs/2410.23042)
- **What's New**: 최근 연구에서는 transformers 모델에서 in-context learning (ICL)이 특정 분포적 속성이 훈련 데이터에 존재할 때 발생하고, 훈련이 진행되면서 이 능력이 감소할 수 있음을 실증적으로 입증했습니다. 본 논문은 이러한 현상을 설명하기 위해 ICL의 출현과 소멸을 초래하는 단순화된 분포적 속성을 식별하여 새로운 이론적 이해를 제공합니다.

- **Technical Details**: 우리는 gating mechanism을 사용하는 단순화된 모델을 분석하여 in-weight (IW) 및 in-context (IC) 예측기를 선택하는 방법을 연구합니다. 일반화 오류 (generalization error)와 후회 분석 (regret analysis) 결합을 통해 ICL과 IW 학습이 발생하는 조건을 식별합니다. 특히, 우리는 훈련 샘플의 분포가 ICL과 IW 학습의 정도에 영향을 미친다는 점에 주목합니다.

- **Performance Highlights**: 실험적으로 ICL이 출현하는 조건을 입증하기 위해 synthetic 및 Omniglot 데이터를 기반으로 한 transformer의 훈련 결과를 통해 이론을 뒷받침합니다. 최종적으로, 특정 데이터를 암기하도록 전체 대형 언어 모델이 fine-tuning될 때 ICL 능력이 감소할 수 있다는 점을 밝히며, 이론과 실제 간의 간극을 메꾸었습니다.



### Offline Reinforcement Learning and Sequence Modeling for Downlink Link Adaptation (https://arxiv.org/abs/2410.23031)
- **What's New**: 본 논문에서는 전통적인 링크 적응(link adaption, LA) 알고리즘의 복잡성과 비효율성을 해결하기 위해 오프라인 강화학습(offline reinforcement learning, RL)을 활용하는 새로운 접근법을 제안합니다.

- **Technical Details**: 연구는 배치 제약 깊이 Q-러닝(batch-constrained deep Q-learning), 보수적인 Q-러닝(conservative Q-learning), 그리고 결정 변환기(decision transformers)를 기반으로 한 세 가지 LA 설계를 제안하며, 이는 온라인 RL 방법의 성과에 필적하는 성능을 보여줍니다. 오프라인 RL의 데이터 수집은 최소한의 간섭으로도 가능하여 네트워크 운영에 미치는 영향을 줄입니다.

- **Performance Highlights**: 오프라인 RL 알고리즘은 적절한 행동 정책(behavioral policy)으로 수집된 데이터를 바탕으로 최신 온라인 RL 방법과 동등한 성능을 도출할 수 있음을 입증하였습니다.



### Planning and Learning in Risk-Aware Restless Multi-Arm Bandit Problem (https://arxiv.org/abs/2410.23029)
- **What's New**: 이 연구에서는 전통적인 restless multi-arm bandit(RMAB) 문제를 리스크 인식을 통합하여 일반화하였습니다. 그 결과, 리스크 인식 목표에 대한 인덱스 가능성(indexability) 조건을 수립하고 Whittle index에 기반한 솔루션을 제공합니다.

- **Technical Details**: 연구는 Whittle index 정책을 통해 에이전트가 리스크 인식 목표에 따라 자원을 최적 분배할 수 있는 조건을 확립합니다. 또한, 실제 전이 확률이 알려져 있지 않은 경우 Thompson sampling 접근 방식을 제안하고, 이를 통해 에피소드 수에 대해 서브선형적으로, 팔 수에 대해 제곱적으로 스케일되는 제한된 후회(bounded regret)를 달성함을 보였습니다.

- **Performance Highlights**: 모델의 효과성을 입증하기 위해 기계 교체와 환자 스케줄링 애플리케이션을 포함한 다양한 수치 실험을 수행하였고, 리스크 노출이 감소함을 보여주었습니다.



### Online Intrinsic Rewards for Decision Making Agents from Large Language Model Feedback (https://arxiv.org/abs/2410.23022)
- **What's New**: 이번 연구에서는 ONI라는 분산 아키텍처를 제안하여 강화 학습 (RL) 정책과 내재적 보상 함수를 동시에 학습하는 방식으로, 기존의 내재적 보상 설계의 한계를 극복합니다.

- **Technical Details**: ONI는 비동기 LLM 서버를 통해 에이전트의 경험을 주석 처리하고, 이러한 피드백을 토대로 내재적 보상 모델을 증류합니다. 다양한 보상 모델링 방법 (해싱, 분류, 순위 모델)을 탐색하여 내재적 보상 설계에 대한 통찰을 제공합니다.

- **Performance Highlights**: ONI는 NetHack Learning Environment의 도전적인 희소 보상 작업에서 최첨단 성능을 달성했으며, 외부 데이터셋이나 소스 코드 없이 에이전트가 수집한 경험만을 사용하여 학습하게 됩니다.



### Scoring Rules and Calibration for Imprecise Probabilities (https://arxiv.org/abs/2410.23001)
- **What's New**: 이 논문은 비정밀 확률 예측에 대한 평가 이론을 발전시키며, 비정밀 확률(sets of probabilities)의 스코어링 규칙(proper scoring rules)과 교정(calibration)의 개념을 일반화하는 데 초점을 맞추고 있습니다. 이는 기계 학습에서 비정확한 예측을 다룰 수 있는 새로운 접근법을 제시합니다.

- **Technical Details**: 우리는 데이터 모델(data models)과 결정 문제(decision problems) 관련하여 비정밀 스코어링 루 rules을 개발했습니다. 또한 최적의 비정밀 확률 예측을 특성화하고, 손실 함수(loss function)가 사전적으로 알려지지 않은 경우 예측에서의 비정밀성이 얼마나 중요한지를 시사합니다. 결정 이론적 엔트로피(decision-theoretic entropy)가 두 가지 목표 간의 긴밀한 관계를 형성하는 데 중심적인 역할을 합니다.

- **Performance Highlights**: 논문은 기계 학습 실무에서 비정밀 예측의 이점을 시연하고, 손실 함수 선택에서의 미묘한 함정(subtle pitfalls)을 조명합니다. 비정밀 예측이 프레딕션 품질을 평가하는 두 가지 목표, 즉 스코어(score)와 교정(calibration) 간의 불일치를 밝히고, 이 개념들이 일반적으로 어떻게 서로 연관되어 있는지를 탐구합니다.



### Higher-order Cross-structural Embedding Model for Time Series Analysis (https://arxiv.org/abs/2410.22984)
- **What's New**: 이번 논문에서는 Higher-order Cross-structural Embedding Model for Time Series (High-TS)라는 새로운 프레임워크를 제안합니다. 이 모델은 멀티스케일 Transformer와 Topological Deep Learning (TDL)을 결합하여 시계열 데이터의 공간적 및 시간적 의존성을 동시에 모델링할 수 있습니다. 또한, 대조학습(contrastive learning)을 활용하여 두 가지 구조를 통합함으로써 강력하고 차별화된 표현을 생성합니다.

- **Technical Details**: High-TS는 두 가지 주요 구성 요소로 이루어져 있습니다. 첫 번째는 시간 차원의 멀티스케일 임베딩 모듈로, 이 모듈은 멀티스케일 주의 메커니즘을采用하여 시간 차원에서의 표현을 학습합니다. 두 번째는 공간 차원의 단순 복합체 임베딩 모듈로, 이 모듈은 TDL을 사용하여 서로 다른 단순 복합체 간의 고차원 상호작용을 구성하고 새로운 특성 표현을 학습합니다.

- **Performance Highlights**: High-TS는 다양한 시계열 작업에서 최첨단 방법들보다 우수한 성능을 보이며, 모델 성능을 향상시키는 데 있어 고차원 교차 구조 정보의 중요성을 입증합니다. 대규모 실험 결과, 모델이 복잡한 상호작용을 효과적으로 모델링하고 잡아내는 능력을 갖추고 있음을 확인하였습니다.



### Dual-Optimized Adaptive Graph Reconstruction for Multi-View Graph Clustering (https://arxiv.org/abs/2410.22983)
Comments:
          Accepted by ACM MM 2024

- **What's New**: 본 논문에서는 이종 그래프(homophilous)에서의 문제를 해결하기 위해 전통적인 GNN의 장점을 유지하면서 이중 최적화된 적응형 그래프 재구성 방법(DOAGC)을 제안합니다.

- **Technical Details**: DOAGC는 노드 상관 행렬을 기반으로 그래프 재구성 메커니즘을 개발하고, 원본 구조 정보를 고려합니다. 이중 최적화 전략을 통해 최적화된 그래프 구조를 구현하며, 상호 정보 이론(mutual information theory)을 통해 전략의 타당성을 입증합니다.

- **Performance Highlights**: 실험 결과에 따르면, DOAGC는 이종 그래프 문제를 효과적으로 완화하며, 기존 GNN 기반 방법들의 한계를 보완하는 데 성공합니다.



### DisenTS: Disentangled Channel Evolving Pattern Modeling for Multivariate Time Series Forecasting (https://arxiv.org/abs/2410.22981)
- **What's New**: 이 논문에서는 DisenTS라는 새로운 프레임워크를 제안하여 복잡한 다변량 시계열 예측 문제에서의 채널 간 의존성을 더 효과적으로 모델링하려고 합니다. DisenTS는 각기 다른 예측 모델을 통해 다양한 진화 패턴을 독립적으로 포착하는 방식을 채택합니다.

- **Technical Details**: DisenTS는 Forecaster Aware Gate (FAG) 모듈을 도입하여 예측 모델의 상태와 입력 시계열의 특성에 따라 적응적으로 라우팅 신호를 생성합니다. 또한, Linear Weight Approximation (LWA) 전략을 사용하여 복잡한 심층 신경망을 간략한 매트릭스로 양자화합니다. 마지막으로 Similarity Constraint (SC)을 통해 각 예측 모델이 독립적인 패턴에 특화되도록 중간 표현의 상호 정보를 최소화합니다.

- **Performance Highlights**: DisenTS는 다양한 실세계 데이터셋에 대한 폭넓은 실험을 통해 기존의 통합 모델링 방법보다 우수한 성능을 보여줍니다. 또한, 채널 독립적 및 채널 의존적인 최신 모델에 통합될 수 있어 효과성과 일반화 능력이 뛰어납니다.



### Dynamic Threshold-based Two-layer Online Unsupervised Anomaly Detector (https://arxiv.org/abs/2410.22967)
- **What's New**: 이 논문에서는 Adaptive NAD라는 새로운 프레임워크를 소개합니다. 이는 온라인 비지도 적 이상 탐지 시스템을 개선하고 해석 가능하게 만들기 위한 포괄적인 접근법입니다.

- **Technical Details**: Adaptive NAD는 해석 가능한 두 층의 이상 탐지 접근법을 제안하며, 신뢰할 수 있는 고신뢰성의 의사 레이블(pseudo-labels)을 생성합니다. 또한, 혁신적인 임계값 조정 방법을 이용하여 새로운 위협에 빠르게 적응할 수 있는 온라인 학습 메커니즘을 포함합니다.

- **Performance Highlights**: 실험 결과에 따르면, Adaptive NAD는 CIC-Darknet2020 및 CIC-DoHBrw-2020 데이터셋에서 각각 5.4% 및 23.0%의 성과 향상으로 최신 기술들을 초월합니다.



### Retrieval-Augmented Generation with Estimation of Source Reliability (https://arxiv.org/abs/2410.22954)
- **What's New**: 이번 연구에서는 Reliability-Aware RAG (RA-RAG)라는 새로운 방법론을 제안하여 다원적 출처의 신뢰성을 평가하고 이를 정보 검색 및 집합 과정에 통합합니다.

- **Technical Details**: RA-RAG는 두 단계로 작동하며, 첫 번째 단계에서는 레이블이 없는 쿼리 집합에 대해 출처 신뢰성을 반복적으로 추정합니다. 두 번째 단계에서는 예측된 출처 신뢰성에 따라 신뢰할 수 있는 문서를 선택적으로 검색하고, 가중 다수결 (WMV) 방법으로 집계합니다.

- **Performance Highlights**: RA-RAG는 이전의 여러 기준선들과 비교해 일관되게 우수한 성능을 보였으며, 신뢰할 수 없는 정보의 집합에서도 정보 집계의 정확성을 높였습니다.



### SpiroActive: Active Learning for Efficient Data Acquisition for Spirometry (https://arxiv.org/abs/2410.22950)
- **What's New**: 본 연구에서는 wearable spirometry 기술을 사용하여 호흡기 질환의 조기 진단을 위한 데이터 수집 비용을 줄이는 active learning 접근법을 제안합니다. 이는 소수의 데이터 샘플을 전략적으로 선택하여 기존 데이터 수집 방법의 어려움을 극복하고 최대한 정확한 모델 성능을 유지하는 것을 목표로 합니다.

- **Technical Details**: 우리 연구는 공공에서 사용 가능한 spirometry 데이터셋을 이용한 active learning 접근 방식을 활용하여, FEV1, FVC, PEF의 값을 추정합니다. Active learning의 기본 개념은 기계학습 시스템이 자기가 학습할 데이터를 선택하게 함으로써 정확도를 향상시킬 수 있다는 것입니다. 우리는 Single Task 및 Multi-Task Active Learning과 같은 다양한 학습 전략을 탐구했습니다.

- **Performance Highlights**: 우리의 active learning 방법론은 FEV1에서 4.96%, FVC에서 4.45%의 오류율을 달성하였으며, 이는 American Thoracic Society (ATS)의 허용 오류율인 7%를 지속적으로 초과 성과를 나타냅니다. 또한, 우리는 약 30%의 전체 데이터셋을 사용하여 단일 작업 설정에서 이러한 결과를 달성했습니다.



### MutaPLM: Protein Language Modeling for Mutation Explanation and Engineering (https://arxiv.org/abs/2410.22949)
Comments:
          NeurIPS 2024 poster

- **What's New**: 이번 논문에서는 MutaPLM이라는 통합 프레임워크를 소개하여 단백질 돌연변이를 모델링하고 해석할 수 있는 새로운 방법을 제시합니다. MutaPLM은 명시적인 단백질 돌연변수 표현을 캡처하는 단백질 델타 네트워크를 도입하고, 생물 의학 텍스트에서 단백질 돌연변이 정보를 수확하기 위해 체인 오브 씽크 (CoT) 전략을 사용하는 전이 학습 파이프라인을 포함하고 있습니다.

- **Technical Details**: MutaPLM은 단백질 델타 네트워크를 통해 돌연변이와 단백질 델타 피처 간의 변환을 가능하게 하며, 단백질의 다양한 기능과 구조에 대한 해석을 위한 협업적인 특징 공간을 형성합니다. 또한, MutaDescribe라는 첫 번째 대규모 단백질 돌연변이 데이터 세트를 생성하여 리치 텍스트 주석을 제공하고, 이는 크로스 모달 감독 신호를 제공합니다.

- **Performance Highlights**: MutaPLM은 돌연변이 설명에서 ROUGE-L 점수에서 기존 모델보다 6.5% 향상된 성능을 보였으며, 19.4%의 예측된 돌연변이 효과가 전문가에 의해 정확하고 통찰력 있는 것으로 평가되었습니다. 돌연변이 엔지니어링에서 평균 0.409의 리콜 점수를 기록하여 ESM-2를 1.6배 개선했습니다.



### ELBOing Stein: Variational Bayes with Stein Mixture Inferenc (https://arxiv.org/abs/2410.22948)
- **What's New**: 본 논문에서는 Stein 경량 그래디언트 하강법(Stein Variational Gradient Descent, SVGD)의 한계를 극복하는 새로운 방법인 Stein Mixture Inference (SMI)를 제안합니다. SMI는 각 입자가 혼합 모델에서 구성 분포를 매개변수화하도록 일반화되어 있습니다.

- **Technical Details**: SMI는 증거의 하한인 ELBO(ELBO: Evidence Lower Bound)를 최적화하며, 입자로 매개변수화된 사용자 지정 가이드를 도입합니다. 또한, Nonlinear SVGD 프레임워크(Nonlinear SVGD framework)를 변별적 베이지안(Variational Bayes) 경우에 맞게 확장합니다.

- **Performance Highlights**: SMI는 테스트에서 분산 소멸(variance collapse)을 효과적으로 피하며, 표준 데이터 세트에서 좋은 성능을 보입니다. 또한, 소규모 Bayesian Neural Networks (BNNs)의 경우 정확한 불확실성 추정을 위해 SVGD보다 필요한 입자의 수가 현저히 적습니다.



### Focus On This, Not That! Steering LLMs With Adaptive Feature Specification (https://arxiv.org/abs/2410.22944)
Comments:
          28pages, 14 figures

- **What's New**: 이번 연구에서는 Focus Instruction Tuning (FIT)이라는 새로운 방법론을 소개합니다. FIT는 LLMs를 특정 작업 수행 시 반응을 조정하도록 훈련시키며, 모델이 특정 기능에 초점을 맞추고 다른 기능은 무시할 수 있도록 합니다.

- **Technical Details**: FIT는 사용자가 어떤 기능에 주목할지를 동적으로 지정할 수 있게 하여, 모델의 행동을 유연하고 효과적으로 조정할 수 있습니다. 이는 특정 입력에 대해 주어진 특징에 따라 다르게 응답할 수 있도록 LLMs를 훈련시킵니다. 다양한 NLP 작업, 예를 들어 감정 분석(sentiment analysis), 자연어 추론(natural language inference), 질문-응답(question-answering) 등에서 FIT의 효과를 실험하였습니다.

- **Performance Highlights**: FIT는 훈련 시 사용하지 않았던 새로운 기능에서의 일반화 능력이 뛰어나며, 분포의 변동에 강인하게 반응합니다. 사용자가 요청하는 바에 따라 모델이 알려진 왜곡된 기능(spurious features)을 무시하고 작업 관련 기능(task-relevant features)에 주력하도록 유도하여, 더 강력하고 공정하며 제어 가능한 LLM 애플리케이션을 가능하게 합니다.



### Simulation-Free Training of Neural ODEs on Paired Data (https://arxiv.org/abs/2410.22918)
- **What's New**: 이 연구에서는 Neural Ordinary Differential Equations (NODEs)의 시뮬레이션 프리 (simulation-free) 학습 방법을 다루고 있습니다. 기존의 NODEs는 데이터 쌍 간의 결정적 매핑을 학습하는 데에 있어 함수 평가 횟수 문제나 수치적 불안정성 등으로 인해 널리 사용되지 않았으나, 본 연구에서는 flow matching 프레임워크를 통해 이러한 문제를 해결합니다.

- **Technical Details**: NODEs에 대한 flow matching 접근 방식은 기본적으로 파라미터화된 동역학 함수를 미리 정의된 목표 속도 필드에 직접 회귀시키는 방식입니다. 이러한 방법은 시뮬레이션을 사용하지 않고도 효율적인 학습을 가능하게 하며, 데이터 쌍 간의 유효한 흐름을 보장하기 위해 데이터 쌍의 임베딩 공간에서 flow matching을 수행하여 효과를 극대화합니다.

- **Performance Highlights**: 본 연구에서 제안하는 방법은 회귀 및 분류 작업 모두에서 기존 NODEs보다 월등한 성능을 보이며, 필요한 함수 평가 수(NFEs)를 크게 줄입니다. 이는 도메인 지식을 활용하거나 더 간단한 흐름을 통해 가능하며, 실험을 통해 그 효과성과 다재다능성을 입증하였습니다.



### CopRA: A Progressive LoRA Training Strategy (https://arxiv.org/abs/2410.22911)
Comments:
          Published in UniReps Workshop (Extended Abstract Track), NeurIPS 2024

- **What's New**: 본 연구에서는 LoRA(Low-Rank Adaptation)의 새로운 훈련 전략인 Cooperative LoRA (CopRA)를 제안합니다. 이 방법은 랜덤 레이어 드롭핑을 통해 모델의 전반적인 성능을 향상시키고, Shapley 값을 최적화하는 방식으로 각 레이어를 협력 게임의 플레이어로 간주합니다.

- **Technical Details**: CopRA는 일반적인 LoRA 훈련 방식의 한계를 극복하기 위해 제안된 전략으로, 각 레이어의 참여도를 랜덤하게 결정하여 훈련 초기에는 최적 솔루션의 수를 제한합니다. 이 과정은 후속 단계에서 LoRA 모듈의 수를 점차 늘려가며 글로벌 옵티마(global optimum) 탐색을 촉진합니다.

- **Performance Highlights**: 실험 결과, CopRA로 훈련된 파라미터는 linear mode connectivity (LMC)를 보이며, 이는 모델 병합에서의 효율성을 높입니다. 또한, 프루닝(pruning) 작업에서도 우수한 성능을 발휘하며, 연합 학습(federated learning) 및 다중 태스크 학습(multi-task learning) 가능성을 열어줍니다.



### Federated UCBVI: Communication-Efficient Federated Regret Minimization with Heterogeneous Agents (https://arxiv.org/abs/2410.22908)
- **What's New**: 이번 논문에서는 연합 학습( federated learning ) 프레임워크에 맞춘 새로운 알고리즘인 Federated Upper Confidence Bound Value Iteration ($\texttt{Fed-UCBVI}$)를 제안합니다. 이 알고리즘은 기존의 $\texttt{UCBVI}$ 알고리즘을 확장한 것입니다.

- **Technical Details**: $\texttt{Fed-UCBVI}$의 레그렛(regret)은 $\tilde{\mathcal{O}}(\sqrt{H^3 |\mathcal{S}| |\mathcal{A}| T / M})$로 증가하며, 여기서 $|\mathcal{S}|$는 상태의 수, $|\mathcal{A}|$는 행동의 수, $H$는 에피소드 길이, $M$은 에이전트 수, $T$는 에피소드의 수를 나타냅니다. 단일 에이전트 설정에서는 이 상한이 minimax 하한(minimax lower bound)과 거의 일치하며, 다중 에이전트 시나리오에서는 선형적인 속도 향상을 보입니다. 또한 새로운 이질성( heterogeneity ) 측정 방법을 제시합니다.

- **Performance Highlights**: 기존의 연합 강화 학습(federated reinforcement learning) 접근 방식과 달리, $\texttt{Fed-UCBVI}$의 통신 복잡성(communication complexity)은 에이전트 수가 증가해도 거의 변하지 않습니다.



### VPO: Leveraging the Number of Votes in Preference Optimization (https://arxiv.org/abs/2410.22891)
- **What's New**: 본 논문에서는 Direct Preference Optimization (DPO) 방법을 보완하여 사용자 투표 데이터를 활용한 Vote-based Preference Optimization (VPO) 프레임워크를 제안합니다. 이 방법은 사용자 선호도를 보다 효과적으로 반영하여 다양한 주관적 선호와의 정렬을 개선하는 것을 목표로 합니다.

- **Technical Details**: VPO는 Bayesian Minimum Mean Square Error (MMSE) 추정기를 사용하여 두 개의 생성 결과 중 하나가 더 선호될 확률을 모델링합니다. VPO는 DPO와 Identity Preference Optimization (IPO) 알고리즘을 각각 VDPO와 VIPO로 확장할 수 있으며, 이는 생성 쌍의 논쟁적 여부를 분별하는 데 도움이 됩니다.

- **Performance Highlights**: 실험 결과 VDPO와 VIPO는 기존 알고리즘들보다 뛰어난 생성 품질과 훈련 안정성을 달성하였습니다. 이 프레임워크는 고비용의 인간 투표 정보가 없는 상황에서도 AI 피드백을 활용하여 적용 가능합니다.



### Data subsampling for Poisson regression with pth-root-link (https://arxiv.org/abs/2410.22872)
Comments:
          NeurIPS 2024

- **What's New**: 이 논문에서는 포아송 회귀(Poisson regression)를 위한 데이터 서브샘플링(data subsampling) 기법을 개발하고 분석합니다. 특히, 우리는 ID-link 및 제곱근 링크(sqrt-link) 함수를 사용하는 포아송 일반화 선형 모델(Poisson generalized linear model)을 고려합니다. 또한, 1±ε의 비율로 포아송 회귀의 손실 함수(loss function)를 근사하는 소형 가중치 집합인 coresets의 방법을 다룹니다. 이 연구는 Pi법적 기법을 통해 서브선형(sublinear) coresets가 존재함을 입증합니다.

- **Technical Details**: 우리는 주어진 데이터로부터 포아송 회귀에 대한 Ω(n) 하한 경계를 보여줍니다. 새로운 복잡도 파라미터(complexity parameter) 및 도메인 전환(domain shifting) 접근 방식을 도입하여, 복잡도 파라미터가 작을 때 1±ε 근사 보장을 갖는 서브선형 coresets의 존재를 증명합니다. 특히, 입력 포인트 수에 대한 의존성을 다항대수(logarithmic)로 줄일 수 있습니다.

- **Performance Highlights**: 이 연구는 ID-link의 경우에는 Θ(√ymax)의 의존성을 보여주고, square root-link의 경우에는 O(log(ymax)) 의존성을 제시합니다. 실험적 예시를 통해 대용량 데이터에서 coresets 사용의 이점을 설명하며, 이로 인해 데이터 절약 및 처리 시간 단축의 효과를 더욱 확인하였습니다.



### Conditioned quantum-assisted deep generative surrogate for particle-calorimeter interactions (https://arxiv.org/abs/2410.22870)
Comments:
          26 pages, 10 figures, 8 appendices

- **What's New**: 이 논문에서는 LHC(대형 강입자 충돌기) 시뮬레이션의 계산 비용을 줄이기 위해 Quantum-assisted deep generative model을 제안합니다. 특히, 대칭 Restricted Boltzmann Machine(RBM)을 갖춘 조건부 변분 오토인코더(VAE)를 통해 기존 VAE보다 더 향상된 표현력을 제공합니다. 이 방식은 D-Wave의 Advantage 양자 어닐러(quantum annealer)를 통해 샘플링을 가능하게 합니다.

- **Technical Details**: 논문에서는 Calo4pQVAE라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 4-partite 그래프를 사용하여 RBM을 재구성하고, 3D convolutional layers를 통합해 인코더와 디코더의 성능을 향상시킵니다. 또한 양자 어닐러의 flux bias parameters를 통해 조건화를 구현하고, 효과적인 온도를 추정하기 위한 새로운 적응형 방법을 소개합니다.

- **Performance Highlights**: CaloChallenge 2022의 Dataset 2를 사용하여 프레임워크의 효과를 입증하였으며, 기존 모델보다 훨씬 저렴한 계산 비용으로 셀 에너지 이동을 직접 생성할 수 있음을 보여줍니다. 이 접근 방식은 LHC 실험 시뮬레이션의 속도를 크게 향상시킬 가능성이 있습니다.



### Towards Robust and Efficient Federated Low-Rank Adaptation with Heterogeneous Clients (https://arxiv.org/abs/2410.22815)
- **What's New**: 본 논문에서는 Federated Learning(연합 학습) 환경에서 대규모 언어 모델(Large Language Models, LLMs)의 파인튜닝에 있어, 새로운 접근 방식인 LoRA-A2(저차원 적응 방식)를 제안합니다. 이는 높은 데이터 이질성 및 저차수 환경에서도 강력한 성능을 자랑하며, 기존 방법들의 성능 저하 문제를 개선합니다.

- **Technical Details**: LoRA-A2에서는 Adaptive Rank Selection 전략을 도입하여 중요한 LoRA rank를 선택합니다. 이를 통해 서로 다른 클라이언트들이 서로 다른 LoRA rank를 선택하게 되어 클라이언트 간의 충돌을 최소화하고, 덜 중요한 LoRA 모듈에서 리소스를 획기적으로 재할당합니다. 이 새로운 중요도 기준을 통해 각 모듈 내에서 rank의 기여도를 계산합니다.

- **Performance Highlights**: 실험 결과, LoRA-A2는 성능을 유지하면서 전체 파인튜닝 없이 99.8%의 업로드된 파라미터 수를 감소시키며, 자원 제한적인 환경에서도 LLMs의 실용적인 배치를 가능하게 합니다. 이는 연합 학습 중 통신 효율성을 극대화합니다.



### Universality of the $\pi^2/6$ Pathway in Avoiding Model Collaps (https://arxiv.org/abs/2410.22812)
Comments:
          30 pages

- **What's New**: 이 연구에서는 이론적으로 기존의 모델 훈련 중 발생하는 'Model Collapse' 문제를 피하기 위한 Augment(증강) 작업 흐름의 보편성을 입증합니다. 이전 연구에서 제시된 π2/6 한계가 다양한 통계 모델에 대해 일반화되며, 이러한 조건이 왜 발생하는지에 대한 중요한 통찰을 제공합니다.

- **Technical Details**: 연구자들은 실 데이터를 사용한 초기 훈련 후 이후 세대에서는 합성 데이터만을 사용하여 모델을 훈련하는 discard(폐기) 방식을 우려했습니다. 반면, augment(증강) 방식에서는 각 세대 훈련에 원본 실 데이터를 계속 사용하고 모델에서 생성한 합성 데이터를 추가함으로써 모델의 성능 저하를 방지할 수 있음을 실증적으로 확인했습니다. 더불어, 원본 실 데이터만 사용할 때의 테스트 리스크에 대한 한계인 π2/6을 도출하여, 선형 회귀 분석에서 모델 붕괴를 성능 저하 없이 방지할 수 있음을 밝혔습니다.

- **Performance Highlights**: 이 논문은 다양한 통계 모델 가족을 통해 π2/6 한계가 보편적으로 적용됨을 보여줍니다. 이전의 연구들과 비교했을 때, 실 데이터와 합성 데이터 모두를 사용하는 방법은 모델 성능에 있어 비교적 낮은 리스크를 유지하고 있으며, 이를 통해 다양한 작업 흐름을 모의 실험하여 최적의 방식 선택이 가능함을 증명합니다.



### MILP-StuDio: MILP Instance Generation via Block Structure Decomposition (https://arxiv.org/abs/2410.22806)
Comments:
          Published in the 38th Conference on Neural Information Processing Systems (NeurIPS 2024)

- **What's New**: 논문에서는 블록 구조를 유지하면서 고품질의 Mixed-integer linear programming (MILP) 사례를 생성하는 새로운 프레임워크인 Block Structure Decomposition (MILP-StuDio)를 제안합니다. 이 프레임워크는 기존 캠페인들이 간과했던 블록 구조를 반영하여 생성된 사례의 성능이 현저히 향상되도록 설계되었습니다.

- **Technical Details**: MILP-StuDio는 제약 계수 행렬 (CCMs)에서 블록을 식별하고, 이 블록을 구성 요소로 사용하여 MILP 사례를 효율적으로 분해 및 재구성하는 방법을 사용합니다. 이렇게 구성된 블록 단위는 새로운 사례를 생성하기 위한 세 가지 연산자(블록 삭제, 블록 혼합, 블록 확장)를 통해 활용됩니다.

- **Performance Highlights**: MILP-StuDio는 기존 사례보다 약 10% 이상 해결 시간을 줄여주며, 다양한 크기의 사례를 생성해 컴퓨테이셔널 어려움 및 실행 가능성을 효과적으로 보존합니다.



### Solving Differential Equations with Constrained Learning (https://arxiv.org/abs/2410.22796)
- **What's New**: 이 논문은 과학 제약 학습(science-constrained learning, SCL) 프레임워크를 개발하여 부분 미분 방정식(partial differential equations, PDE)의 솔루션을 구하는 새로운 접근 방식을 제시합니다. SCL은 이전 방식의 한계를 극복하며, 구조 제약(예: 불변성) 및 측정값을 자연스럽게 통합할 수 있습니다.

- **Technical Details**: SCL 프레임워크는 PDE의 (weak) 솔루션을 최악의 손실을 포함한 제약 학습 문제로 변환합니다. 이를 통해 하이퍼파라미터 조정 없이도 다양한 PDE와 신경망(Neural Network) 아키텍처에 대해 정확한 솔루션을 제공할 수 있습니다. SCL은 반무한 제약 학습 기법을 사용하여 문제를 해결하는 복합 샘플링-최적화 알고리즘을 개발합니다.

- **Performance Highlights**: SCL은 다양한 PDE 패밀리에 대해 일관되게 정확한 솔루션을 제공하며, 때로는 낮은 계산 비용으로도 수행됩니다. 실험을 통해 SCL이 다계층 퍼셉트론(multilayer perceptrons, MLP) 및 신경 연산자(neural operators, NOs)와 같은 여러 NN 아키텍처에서 효과적임을 확인하였습니다.



### Theoretical Investigations and Practical Enhancements on Tail Task Risk Minimization in Meta Learning (https://arxiv.org/abs/2410.22788)
- **What's New**: 이번 연구에서는 메타 학습(Meta Learning) 분야에서의 분포 견고성을 개선하기 위한 새로운 접근법을 제안합니다. 기존의 두 단계 최적화 전략을 최대-최소 최적화(max-min optimization) 문제로 변환하고, 이를 스택켈버그 균형(Stackelberg equilibrium)으로 모델링하여 이론적 이해를 심화시킵니다.

- **Technical Details**: 연구의 핵심 기여는 두 가지 측면에서 이뤄집니다. 첫 번째로, 추정 수렴 속도를 제공하고 학습 동역학에서의 비대칭적 행동을 설명합니다. 두 번째로, 꼬리(task-risk) 위험이 존재하는 경우에 대한 일반화 경계를 도출하고, 이를 통해 빠른 적응(fast adaptation) 기능과 결합합니다. 또한, 더 정확한 퀀타일(quantile) 추정기를 사용하여 메타 학습 모델의 강건성을 개선합니다.

- **Performance Highlights**: 폭넓은 평가를 통해 우리의 제안이 멀티모달 대형 모델(multi-modal large models)에 대한 강건성을 향상시키는 데 중요한 영향을 미침을 보여줍니다.



### Contrastive Learning and Adversarial Disentanglement for Privacy-Preserving Task-Oriented Semantic Communications (https://arxiv.org/abs/2410.22784)
Comments:
          Submitted to EEE Journal on Selected Areas in Communications (JSAC): Intelligent Communications for Real-Time Computer Vision (Comm4CV)

- **What's New**: 본 논문에서 제안된 CLAD(contrastive learning and adversarial disentanglement) 방법론은 태스크 지향적 의미 통신 시스템의 정보 전송 방식에 혁신을 가져옵니다. 기존의 문제점인 태스크 관련 및 비관련 정보를 완전히 분리하지 못하는 한계를 극복하기 위해 정보 병목(information-bottleneck) 방식을 도입하였습니다.

- **Technical Details**: CLAD는 대조 학습(contrastive learning)을 활용하여 태스크 관련 기능(feature)을 효과적으로 캡처하는 동시에 적대적 분리(adversarial disentanglement)를 통해 태스크와 무관한 정보를 제거합니다. 또한, 인코딩된 기능 벡터의 정보 유지 지수(information retention index, IRI)를 도입하여 인코딩된 기능과 입력 간의 상호 정보(mutual information)를 대리하는 지표로 사용합니다.

- **Performance Highlights**: CLAD는 태스크 성능, 프라이버시 보존 및 IRI 측면에서 최신 기술들의 성능을 능가하였습니다. CLAD는 약 2.5-3%의 예측 성능 향상, 77-90%의 IRI 감소 및 57-76%의 적대 정확도 감소를 달성하였습니다.



### Understanding Aggregations of Proper Learners in Multiclass Classification (https://arxiv.org/abs/2410.22749)
Comments:
          23 pages

- **What's New**: 이번 연구에서는 다중 클래스 분류에서 proper learner의 조합이 properness barrier를 극복할 수 있는 정도를 조사하였습니다. finite Graph dimension을 가진 클래스에서 최적의 binary learner가 ERM의 샘플 복잡도를 개선할 수 있음을 보였습니다.

- **Technical Details**: 다중 클래스 분류 문제에 대한 연구에서, finite Graph dimension을 가진 클래스의 샘플 복잡도는 \( O\left(\frac{d_G + \ln(1 / \delta)}{\epsilon}\right) \)로 나타나며, 이 결과는 ERM의 샘플 복잡도보다 엄격히 개선되었습니다. 특정 Graph dimension 클래스에 대해서는 ERM learner의 다수결 방식이 \( \Omega \left(\frac{d_G + \ln(1 / \delta)}{\epsilon}\right) \) 만큼의 샘플이 필요함을 보여주었습니다.

- **Performance Highlights**: 최적의 binary learner는 Hanneke, Larsen, Aden-Ali등이 제시한 알고리즘으로, 다수의 ERM learner의 조합이 최적의 성능을 내는 것으로 나타났습니다. 하지만 다중 클래스 분류의 경우, finite DS dimension이 있는 학습 클래스조차도 일정한 오류로 학습할 수 없는 예를 보여주어, proper learner의 조합이 항상 성공적이지 않음을 입증하였습니다.



### MIXAD: Memory-Induced Explainable Time Series Anomaly Detection (https://arxiv.org/abs/2410.22735)
Comments:
          ICPR 2024 (oral paper)

- **What's New**: MIXAD (Memory-Induced Explainable Time Series Anomaly Detection) 모델을 소개하여, 다변량 시계열 데이터에서 해석 가능한 이상 탐지를 달성했습니다. 이 모델은 메모리 네트워크를 활용하여 센서 간의 복잡한 역학과 토폴로지 구조를 이해하고, 기존 최첨단 기법들에 비해 34.30% 및 34.51%의 해석 가능성에서 성능을 향상시킵니다.

- **Technical Details**: MIXAD는 시공간 특성 추출기와 메모리 모듈을 결합하여 데이터의 복잡한 동태를 캡처하는 최초의 MTS 이상 탐지 모델입니다. 이 모델은 정상 시기와 비정상 시기 간의 메모리 활성 패턴 분석을 통해 이상 점수를 생성하며, 이는 예측 오류를 기반으로 한 기존 방법들과 차별화됩니다. 또한, STRGC (Spatiotemporal Recurrent Convolution Unit) 구조를 통해 공간적 및 시간적 의존성을 동시에 처리합니다.

- **Performance Highlights**: MIXAD는 검증 데이터셋에서 기존 최첨단 모델들보다 34.30% 및 34.51% 향상된 해석 가능성을 기록하며, 강력한 탐지 성과를 보였습니다. 이 모델은 이상 탐지를 넘어 특정 이상 원인에 기여하는 특징을 명확히 밝혀내는 데 중점을 두고 있습니다.



### Offline Behavior Distillation (https://arxiv.org/abs/2410.22728)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 본 논문에서는 오프라인 행동 증류(Offline Behavior Distillation, OBD) 문제를 제안하며, 두 가지 기본 OBD 목표인 DBC와 개선된 PBC를 소개합니다.

- **Technical Details**: OBD는 서브 최적 RL 데이터에서 제한된 전문가 행동 데이터를 합성하여 정책 학습을 가속화하는 과정입니다. PBC는 오프라인 데이터에서의 결정 차이를 측정하는 기본 방법을 제공합니다. 하지만 복잡한 이층 최적화(bi-level optimization)로 인해 PBC는 성능 보장에 대한 최악의 경계를 가집니다. 우리는 정책 성능과 행동-가치 가중 결정 차이(action-value weighted decision difference) 간의 동등성을 증명하고 Av-PBC라는 새로운 OBD 목표를 제안합니다.

- **Performance Highlights**: 실험 결과, Av-PBC는 OBD 성능 측면에서 DBC와 PBC보다 각각 82.8% 및 25.7%의 개선을 보였고, convergence 속도 또한 DBC와 PBC 대비 가장 빠른 성능을 나타냈습니다. 또한, 엔샘블 정책 훈련을 통해 Av-PBC에서 훈련된 데이터의 성능을 25.8% 향상시켰습니다.



### Enhancing binary classification: A new stacking method via leveraging computational geometry (https://arxiv.org/abs/2410.22722)
Comments:
          11 pages

- **What's New**: 이 논문은 기존의 stacking 기법에서 벗어나 computational geometry 기법을 통합하여 새로운 메타 모델을 개발하는 접근 방식을 제시합니다. 이는 binary classification 문제를 해결하기 위한 혁신적 방법으로 최대 가중 직사각형 문제(maximum weighted rectangle problem, MWRP)를 활용합니다.

- **Technical Details**: 제안된 MWRP 기반 stacking 프레임워크는 다수의 기본 모델이 생성한 1차원 확률 출력을 다차원 공간으로 변환하여 샘플 점의 표현을 풍부하게 만듭니다. 이로 인해 기본 모델과 해당 임계값의 최적 조합을 식별할 수 있습니다. 각 모델의 선택 과정에서 메타 모델은 높은 해석 가능성을 유지하며, 하이퍼파라미터 튜닝을 생략할 수 있는 장점이 있습니다.

- **Performance Highlights**: 다양한 공개 데이터 세트에 적용한 결과, 기존의 최첨단 stacking 기법에 비해 예측 정확도가 향상되었으며, 병렬 데이터 세트에서도 효과적으로 작동함을 입증하였습니다. 또한, 병합된 모델의 해석 가능성 향상과 현실 세계의 여러 애플리케이션(예: 병원 건강 평가 및 은행 신용 평가 시스템)에 실용성이 높아졌습니다.



### Community search signatures as foundation features for human-centered geospatial modeling (https://arxiv.org/abs/2410.22721)
Comments:
          8 pages, 8 figures, presented at the DMLR workshop at ICML 2024

- **What's New**: 이번 연구에서는 기존의 키워드 중심 데이터 수집 방법과는 달리, 검색 데이터를 활용하여 익명화된 커뮤니티 레벨의 검색 관심도를 생성하는 새로운 접근 방식이 소개되었습니다. 이 방법은 시간 정렬없이도 다양한 공간 예측 작업에 사용될 수 있는 기반 기능을 제공합니다.

- **Technical Details**: 미국 내 3천 명 이상의 인구를 가진 zip 코드에서의 검색 데이터를 기반으로, 1,000개 이상의 고유 쿼리를 추출하고 이를 1,000차원 벡터로 변환해 지역 검색 서명을 생성했습니다. 개발된 모델은 21개 건강 변수에 대해 평균 $R^2$ 점수 0.74와 6개 인구통계 및 환경 변수에 대해 0.80의 점수를 기록하며, 위성 이미지 기능을 사용하는 최신 방법을 초월하는 성능을 보였습니다.

- **Performance Highlights**: 이 연구의 결과는 새로운 검색 기능을 사용하여 공간 예측을 수행할 수 있음을 보여주며, 지역 커뮤니티의 건강, 소득, 인구 밀도, 주택 및 환경 관련 변수의 예측에서 최고의 성능을 달성했습니다.



### Exactly Minimax-Optimal Locally Differentially Private Sampling (https://arxiv.org/abs/2410.22699)
Comments:
          32 pages and 7 figures. Accepted by NeurIPS 2024

- **What's New**: 이 논문은 개인 정보 보호를 위한 로컬 차분 (Local Differential Privacy, LDP) 모델에서 발생하는 샘플링 문제를 다루며, 개인 정보 보호와 유용성 간의 트레이드오프(privacy-utility trade-off, PUT)를 수학적으로 정의하고 최적 샘플링 메커니즘을 제시합니다.

- **Technical Details**: 제안된 접근 방식은 원래 데이터 분포와 샘플링 분포 간의 f-divergence를 사용하여 유용성을 측정하며, 이는 Kullback-Leibler divergence, total variation distance, Hellinger distance 등의 다양한 거리 측정을 포함합니다. FINITE 및 CONTINUOUS 데이터 공간에서의 PUT을 정확하게 특성화하고, 모든 f-divergence에 대해 보편적으로 최적화된 샘플링 메커니즘을 제안합니다.

- **Performance Highlights**: 실험 결과, 제안된 메커니즘이 기존의 기준 방법보다 이론적인 유용성(finite data space) 및 경험적 유용성(continuous data space) 면에서 우수함을 보여주었습니다. 거기서, FINITE 데이터 공간에서 양식이 닫힌 형태로 유용성이 유도되며, 컨티뉴어스 데이터 공간에서는 경험적인 유용성을 사용하여 비교가 이루어집니다.



### An Iterative Algorithm for Regularized Non-negative Matrix Factorizations (https://arxiv.org/abs/2410.22698)
Comments:
          6 figures

- **What's New**: 본 연구는 Lee와 Seung의 비부정 행렬 분해(non-negative matrix factorization, NMF) 알고리즘을 일반화하여 가중 노름(weighted norm)과 Ridge 및 Lasso 정규화(regularization)를 지원합니다.

- **Technical Details**: 논문에서는 Lee와 Seung의 곱셈 업데이트를 덧셈 업데이트(additive update)로 변경하여 0 값에서 멈추지 않도록 합니다. 또한, rnnmf 패키지를 사용하여 칵테일 데이터베이스의 축소 순위 표현을 찾는 문제에 적용합니다.

- **Performance Highlights**: 이 알고리즘은 높은 수준의 컴퓨터 언어에서도 간단하게 구현 가능하여 실험이나 데이터 분석에 유용합니다.



### Permutation Invariant Learning with High-Dimensional Particle Filters (https://arxiv.org/abs/2410.22695)
Comments:
          Website: this https URL

- **What's New**: 이번 연구에서는 high-dimensional particle filters를 기반으로 하는 새로운 permutation-invariant learning framework를 소개합니다. 이 프레임워크는 데이터의 순서에 영향을 받지 않으며, catastrophic forgetting(재앙적 망각)과 loss of plasticity(유연성 상실)을 완화합니다.

- **Technical Details**: 제안된 방법은 gradient-based optimization의 장점과 Bayesian 방법의 특성을 결합한 효율적인 particle filter를 개발하여 고차원 모델을 최적화하는 것입니다. 특히, particle filter가 훈련 미니배치의 순서에 무관하게 작용하도록 이론적으로 입증하였습니다.

- **Performance Highlights**: SplitMNIST, SplitCIFAR100, ProcGen 등의 연속 학습 및 강화 학습 벤치마크에서 광범위한 실험을 통해, 제안한 방법이 표준 기준 모델 대비 성능 향상과 변동성 감소를 일관되게 보였습니다.



### Choice between Partial Trajectories (https://arxiv.org/abs/2410.22690)
- **What's New**: 본 논문에서는 AI 에이전트가 인간의 선호를 배우는 데 있어 bootstrap return 모델을 제안하며, 이를 통해 인간의 신념을 고려한 선택 모델링을 수행합니다. 기존의 partial return 모델과 cumulative advantage 모델의 한계를 극복할 수 있는 방법으로 bootstrap return이 소개됩니다.

- **Technical Details**: 논문에서는 bootstrapped return을 통해 부분적인 보상(partial return)과 미래 보상의 추정값을 합산하여 사용하는 방법을 설명합니다. 이를 통해 인간의 신념에 따라 선택의 결과를 해석할 수 있으며, 보상 함수(reward function)를 학습하는 데 있어 더 높은 견고성을 보입니다. Axiom과 Alignment Theorem을 통해 목표(goal)와 신념(belief)을 분리하는 방법이 formalized 되어 있습니다.

- **Performance Highlights**: 실험 결과, bootstrapped return 모델은 선택 행동(choice behavior)에 대해 더 높은 강인성을 보이며, 잘못된 신념을 가진 데이터로부터도 효과적으로 보상 함수를 회복할 수 있음을 보여줍니다. 이는 에이전트가 인간의 의도를 보다 잘 반영한 보상 함수를 학습할 수 있는 기회를 제공합니다.



### Improving Uncertainty Quantification in Large Language Models via Semantic Embeddings (https://arxiv.org/abs/2410.22685)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)에서 정확히 불확실성을 정량화하는 새로운 접근 방식을 제안합니다. 기존의 방법들은 이중 포함 관계를 기준으로 여러 생성된 응답간의 의미적 불확실성을 측정하는데 의존하며, 이는 세부적인 표현 차이에 민감하여 실제 불확실성을 과대평가하는 경향이 있습니다. 이에 반해, 저자들은 의미 임베딩을 사용하여 의미적 불확실성을 보다 부드럽고 견고하게 추정할 수 있는 방법을 제시합니다.

- **Technical Details**: 저자들은 의미적 임베딩 불확실성(Semantic Embedding Uncertainty, SEU) 개념을 도입하며, 이는 전체 응답 임베딩의 평균 쌍 간 코사인 유사도를 활용합니다. 이 방법은 이중 포함을 기준으로 의미적으로 동등한 응답을 군집화하는 과정에서 발생하는 문제를 피할 수 있습니다. 또한 저자들은 이론을 기반으로 한 암모르티즈드(SEU) 모델을 제시하여 잠재 변수로서 의미를 모델링해 단일 전방 통과로 불확실성을 추정하는 방식을 개발하였습니다. 이를 통해 연산 오버헤드를 획기적으로 줄이고 실무 환경에서의 활용성을 향상시킬 수 있습니다.

- **Performance Highlights**: 다양한 질문-응답 데이터셋과 최전선 LLMs에 대한 실험 결과, 저자들이 제안한 임베딩 기반 방법이 기존의 전통적인 기법들보다 더 정확하고 섬세한 불확실성 정량화를 제공함을 입증하였습니다.



### Byzantine-Robust Federated Learning: An Overview With Focus on Developing Sybil-based Attacks to Backdoor Augmented Secure Aggregation Protocols (https://arxiv.org/abs/2410.22680)
Comments:
          16 pages, 4 figures, 1 appendix

- **What's New**: 이 연구는 Federated Learning (FL) 프로토콜의 취약점을 악용하는 두 가지 새로운 Sybil 기반 공격을 제안하며, 기존의 Byzantine 공격 방어 방법론에 대한 포괄적이고 업데이트된 분류 체계를 구축합니다.

- **Technical Details**: Federated Learning (FL)은 클라이언트의 개인 데이터를 사용하여 중앙 서버에서 모델을 교육할 수 있도록 하는 분산 머신러닝 방법입니다. 본 논문에서는 Robustness of Federated Learning (RoFL) 프로토콜의 강점과 약점을 심층적으로 분석하며, 이러한 약점을 활용한 두 가지 Sybil 기반 공격을 제안합니다.

- **Performance Highlights**: 이 연구는 RoFL 프로토콜의 보안성과 개인 정보 보호성을 종합적으로 평가하고, 향후 개선 방향을 제시하는 것을 목표로 하여 공격의 구현 세부 사항과 시험 제안도 포함하고 있습니다.



### Is Function Similarity Over-Engineered? Building a Benchmark (https://arxiv.org/abs/2410.22677)
Comments:
          To appear in the 38th Conference on Neural Information Processing Systems (NeurIPS 2024) Track on Datasets and Benchmarks

- **What's New**: 새로운 벤치마크인 REFuSE-Bench를 제안하여, 고품질 데이터셋과 실제 사례를 반영하는 테스트를 기반으로의 바이너리 함수 유사성 감지에 대한 평가를 하고 있습니다.

- **Technical Details**: 이 연구에서 제안한 REFuSE는 바이너리 함수의 원시 바이트만을 사용하고, 복잡한 디스어셈블리나 전처리 과정이 필요 없는 간단한 콘볼루션 신경망(convolutional neural network) 모델입니다.

- **Performance Highlights**: REFuSE는 다양한 작업에서 최신 성능(state-of-the-art performance)을 달성하여, 더 복잡한 모델보다 단순한 접근 방식의 효과를 입증하고 있습니다.



### Calibrating Practical Privacy Risks for Differentially Private Machine Learning (https://arxiv.org/abs/2410.22673)
- **What's New**: 이 논문에서는 differential privacy의 실제 적용에서의 복잡성을 강조하고, 다양한 데이터셋과 모델에서 동일한 이론적 $	heta$ 설정을 가지고도 membership inference 공격의 성공률(ASR)이 어떻게 달라지는지를 연구합니다. 이를 통해 실제 프라이버시 위험을 평가하는 더 나은 지표를 확인할 수 있음을 발견하였습니다.

- **Technical Details**: 연구자들은 SHAP와 LIME 모델 설명기를 사용하여 프라이버시 민감 특성의 민감도를 평가하고, 이러한 특성을 선택적으로 억제하는 feature-masking 전략을 개발합니다. 이를 통해 ASR 값을 낮추면서도 응용 프로그램에 특화된 데이터 유용성을 유지할 수 있음을 보여 줍니다.

- **Performance Highlights**: 실험 결과, LiRA $ASR^M$은 데이터셋의 내재된 프라이버시 위험을 올바르게 나타내며, 더 큰 이론적 $	heta$ 설정을 사용하여 동등한 실제 프라이버시 보호를 달성할 수 있음을 증명합니다. 특정 특성을 신중하게 선택하여 마스킹함으로써 데이터 유용성을 더 많이 보존할 수 있음을 확인하였습니다.



### Incremental Learning of Retrievable Skills For Efficient Continual Task Adaptation (https://arxiv.org/abs/2410.22658)
- **What's New**: 이 논문은 Continual Imitation Learning (CiL)이라는 새로운 프레임워크인 IsCiL을 소개합니다. IsCiL은 다양한 데모로부터 공유 가능한 기술을 점진적으로 학습함으로써 지식 공유의 한계를 해결합니다.

- **Technical Details**: IsCiL은 각 데모에서 적절한 기술을 프로토타입 기반 메모리를 통해 검색하고, 각 프로토타입에 대해 해당 어댑터에 대해 점진적으로 기술을 학습합니다. 이 프레임워크는 작은 어댑터인 기술 검색기(skill retriever)와 기술 디코더(skill decoder)로 구성된 두 층의 계층 구조를 가지고 있습니다.

- **Performance Highlights**: Franka-Kitchen과 Meta-World 환경에서 진행된 CiL 실험 결과, IsCiL은 샘플 효율성과 작업 적응 모두에서 우수한 성능을 보였습니다. 이 프레임워크는 포괄적인 전문가 데모 없이도 유연하게 적응하며, 다양한 지침과 데모로 구성된 작업을 학습할 수 있습니다.



### Reweighting Local Mimina with Tilted SAM (https://arxiv.org/abs/2410.22656)
- **What's New**: 이번 연구에서는 기존의 Sharpness-Aware Minimization (SAM)을 개선한 Tilted SAM (TSAM)을 제안합니다. TSAM은 손실 곡면에서 보다 평평한 점을 찾는 데 초점을 맞추어, 최적화 과정에서 더 높은 우선 순위를 부여하는 구조를 갖추고 있습니다.

- **Technical Details**: TSAM은 새로운 tilt hyperparameter t에 의해 매개변수화되고, t가 무한대로 갈수록 SAM으로 수렴합니다. TSAM의 목표는 SAM보다 부드러워 최적화가 용이하며, t가 증가할수록 평평한 최소값을 선호하는 구조로 되어 있습니다. Hamiltonian dynamics의 이산화 아이디어를 바탕으로 TSAM을 해결하기 위한 알고리즘이 개발되었습니다.

- **Performance Highlights**: TSAM은 다양한 이미지와 텍스트 작업에서 SAM과 ERM의 기준선보다 뛰어난 테스트 성능을 보이며, 더 평평한 로컬 최소값을 도출하는 것으로 나타났습니다.



### FT-PrivacyScore: Personalized Privacy Scoring Service for Machine Learning Participation (https://arxiv.org/abs/2410.22651)
- **What's New**: 이 논문은 기계 학습 모델에서 데이터 기여자의 개인 정보를 보호하기 위한 새로운 도구 FT-PrivacyScore를 개발하여 개인 정보 위험을 정량적으로 평가할 수 있음을 보여줍니다. 이 도구는 모델 미세 조정 작업에 참여하는 데이터 기여자의 개인 정보 위험을 효율적으로 추정할 수 있는 방법을 제공합니다.

- **Technical Details**: FT-PrivacyScore는 데이터 기여자로부터 정보를 수집하고 모델 빌더가 제공하는 훈련 데이터 분포 및 기반 모델을 사용하여 작동하는 개인 정보 점수 서비스입니다. 이 서비스는 여러 개의 미세 조정 모델을 생성하고 각 레코드에 대해 LiRA 테스트를 수행하여 개인 정보 점수를 산출합니다. 이는 샘플 데이터에 대한 구성요소의 민감도를 평가하여 개인 정보 보호 위험을 정량화합니다.

- **Performance Highlights**: FT-PrivacyScore는 기존 LiRA 방법에 비해 비용을 극적으로 줄이면서 단일 인스턴스 당 평가 시간을 3분으로 줄입니다. 이는 대량의 데이터와 모델 훈련을 통해 실용적인 적용이 가능하게 하며, 사용자가 자신의 개인 정보 위험을 직관적으로 이해하고 의사 결정을 내릴 수 있도록 돕습니다.



### WaveRoRA: Wavelet Rotary Route Attention for Multivariate Time Series Forecasting (https://arxiv.org/abs/2410.22649)
Comments:
          The code is coming soon! For sure

- **What's New**: 본 연구에서는 멀티변량 시계열 예측(MTSF)에서 Transformer 기반 모델들이 성과를 보였지만, 기존 연구들이 시계열 데이터의 복잡한 시간적 의존성을 충분히 모델링하지 못하는 문제를 해결하기 위해 웨이브렛 학습 프레임워크를 제안합니다. 이는 시간과 주파수 정보를 통합하여 다양한 스케일에서 신호의 로컬 특성을 분석할 수 있게 합니다. 또한, Rotary Route Attention (RoRA)라는 새로운 주의 메커니즘을 도입하여 기존 Softmax 자기 주의 메커니즘의 비효율성을 해결합니다.

- **Technical Details**: 제안된 WaveRoRA 모델은 웨이브렛 도메인에서 시계열 간 상호 의존성을 캡처합니다. RoRA는 상대 위치 정보를 주입하기 위해 로탈리 포지셔닝 임베딩을 사용하며, 정보 집계를 위해 소수의 라우팅 토큰을 도입하여 $KV$ 매트릭스에서 정보를 모아 $Q$ 매트릭스에 재분배하는 방식으로 선형 복잡성을 제공합니다. 이를 통해 모델의 계산 비용을 줄일 수 있습니다.

- **Performance Highlights**: WaveRoRA는 8개의 실제 데이터셋에서 폭넓은 실험을 수행한 결과, 기존의 최첨단 모델들보다 우수한 성능을 보이며, 계산 비용 또한 낮은 것으로 나타났습니다.



### Consistency Diffusion Bridge Models (https://arxiv.org/abs/2410.22637)
Comments:
          NeurIPS 2024

- **What's New**: 이번 연구에서는 노이즈와 데이터 사이의 확률 흐름을 직접 예측하는 일관성 함수(consistency function)를 학습하는 새로운 접근방식을 제안합니다. 이는 기존의 DDBM(denoising diffusion bridge models)보다 더 나은 샘플링 효율성을 가져다 줍니다.

- **Technical Details**: CDBM(consistency diffusion bridge models)이라는 새로운 모델을 도입하며, 두 가지 훈련 패러다임인 일관성 브릿지 증류(consistency bridge distillation)와 일관성 브릿지 훈련(consistency bridge training)을 제공합니다. 이는 DDBM 공식과 일관성 모델을 유연하게 통합할 수 있는 구조적 기술들을 포함합니다.

- **Performance Highlights**: CDBM은 기존 DDBM보다 4배에서 50배 더 빠른 샘플링 속도를 보이며, 동일한 단계에서 이미지 번역 및 이미지 인페인팅 작업에서 우수한 시각적 품질을 더해줍니다. FID(Fréchet Inception Distance) 지표를 기준으로 두 단계 생성 후 양쪽에서 개선된 성과를 보여주었습니다.



### DECRL: A Deep Evolutionary Clustering Jointed Temporal Knowledge Graph Representation Learning Approach (https://arxiv.org/abs/2410.22631)
Comments:
          Accepted by NeurIPS 2024, 17 pages, and 3 figures

- **What's New**: 본 논문에서는 TKG(Temporal Knowledge Graph)에서 고차 상관의 시간적 진화를 포착하기 위한 DECRL(Deep Evolutionary Clustering jointed temporal knowledge graph Representation Learning) 방법을 제안합니다. DECRL은 고차 상관의 진화를 모델링하기 위한 깊이 있는 진화 적 클러스터링 모듈을 포함하고 있어, 클러스터가 여러 엔티티 간의 고차 상관을 나타냅니다.

- **Technical Details**: DECRL은 클러스터의 연속적인 시간 정합성을 유지하기 위해 클러스터 인식 비지도 정렬 메커니즘을 도입합니다. 또한, 클러스터 간의 잠재적 상관관계를 캡처하기 위한 암묵적 상관 인코더를 포함하며, 이는 클러스터 간의 상호작용 강도를 정의합니다. 이는 전역 그래프의 도움 아래 달성됩니다.

- **Performance Highlights**: DECRL은 7개의 실제 데이터셋에서 실험을 진행한 결과, MRR, Hits@1, Hits@3 및 Hits@10에서 각각 9.53%, 12.98%, 10.42%, 14.68%의 평균 성능 향상을 보이며, 현재 최첨단(SOTA) 성능을 달성했습니다.



### FISC: Federated Domain Generalization via Interpolative Style Transfer and Contrastive Learning (https://arxiv.org/abs/2410.22622)
- **What's New**: 이번 연구에서는 다양한 도메인에서 수집된 클라이언트 데이터로 인한 도메인 시프트 문제를 해결하기 위한 새로운 접근법, FISC(Federated Interpolation Style Contrastive learning)를 제안합니다. FISC는 클라이언트 간의 복잡한 도메인 분포를 처리하며 대비 학습(contrastive learning)을 통해 멀티 도메인 표현(multi-domain representations)을 획득합니다.

- **Technical Details**: FISC는 로컬 스타일(local styles)에서 인터폴레이션 스타일을 추출하고 클라이언트 간의 공유 도메인 지식을 융합합니다. 이를 통해 모든 클라이언트가 고유한 데이터 및 클래스 정보를 드러내지 않고도 글로벌 스타일을 공유할 수 있도록 합니다. FISC는 클라이언트가 자신이 소속되어 있는 도메인 특성과 글로벌 도메인 특성을 결합하여 새로운 데이터를 생성할 수 있게 합니다.

- **Performance Highlights**: FISC는 PACS, Office-Home, IWildCam 등 다양한 데이터셋에서 기존의 최첨단(Federated Domain Generalization) 방법론보다 3.64%에서 57.22%까지 정확도 향상을 보여주었습니다. 이러한 성과는 클라이언트의 개인 정보를 효과적으로 보존하면서도 모델의 성능을 유지할 수 있음을 나타냅니다.



### Solving Minimum-Cost Reach Avoid using Reinforcement Learning (https://arxiv.org/abs/2410.22600)
Comments:
          Accepted to NeurIPS 2024

- **What's New**: 본 논문에서 제안된 RC-PPO는 최소 비용 도달 회피(minimum-cost reach-avoid 문제)를 해결하기 위해 강화 학습(reinforcement learning) 기반의 새로운 방법을 소개합니다. 기존의 방법들은 누적 비용을 직접 최소화하는 정책을 학습하지 못했으나, RC-PPO는 해밀턴-자코비(Hamilton-Jacobi) 도달 가능성에 대한 연결을 이용합니다.

- **Technical Details**: RC-PPO는 두 단계로 이루어진 PPO(Proximal Policy Optimization) 기반 프레임워크를 사용하여 최적의 정책을 학습합니다. 첫 번째 단계에서는 비용 상한에 조건화된 최적 가치 함수(value function) 및 정책(policy)을 해결하며, 두 번째 단계에서는 누적 비용의 상한을 미세 조정하여 최종 최적 정책을 도출합니다.

- **Performance Highlights**: 시뮬레이션 실험 결과, RC-PPO는 기존 방법들과 비교하여 목표 도달률(goal-reaching rates)이 유사하면서도 누적 비용이 최대 57%까지 낮아지는 성과를 보였습니다.



### Are Large-Language Models Graph Algorithmic Reasoners? (https://arxiv.org/abs/2410.22597)
Comments:
          9 pages, 13 Figures

- **What's New**: 이번 연구에서는 Large Language Models (LLMs)의 주요한 도전 과제를 다루고 있습니다. LLMs는 많은 작업에서 우수한 성능을 보이지만, 명시적 그래프에서 여러 단계를 필요로 하는 추론 문제에서 여전히 어려움을 겪고 있습니다. 이를 해결하기 위해 기존의 알고리즘적 추론 작업을 평가할 수 있는 새로운 벤치마크인 MAGMA를 소개합니다.

- **Technical Details**: MAGMA 벤치마크는 Breadth-First Search (BFS), Depth-First Search (DFS), Dijkstra's algorithm, Floyd-Warshall algorithm, Prim's Minimum Spanning Tree (MST-Prim)와 같은 5가지 기본 알고리즘을 포함하고 있습니다. 이 벤치마크는 각 단계에서의 모델의 성능을 체계적으로 평가하며, LLMs가 다단계 그래프 추론 문제를 해결하는 방식에서 중간 단계의 중요성을 강조합니다.

- **Performance Highlights**: LLMs에 중간 단계를 포함시키는 것이 알고리즘적 추론 성능을 상당히 향상시키는 것으로 나타났습니다. 더 작은 미세 조정 모델이 더 큰 기본 언어 모델보다 그래프 알고리즘 추론 작업에서 더 우수한 성능을 보였고, 중간 단계로 미세 조정된 모델은 불필요한 정보에 민감한 경향이 있어 단계별 추론의 명확성이 필요함을 보여주었습니다.



### Gaussian Derivative Change-point Detection for Early Warnings of Industrial System Failures (https://arxiv.org/abs/2410.22594)
- **What's New**: 이 논문은 시스템 고장을 예측하기 위한 3단계 프레임워크를 소개합니다. 이 프레임워크는 고차원 특징 공간에서 변화를 감지하는 Gaussian Derivative Change-Point Detection (GDCPD) 알고리즘을 제안합니다. GDCPD는 시스템의 주요 특징에서 변화 지점을 식별하여 시스템 실패로 이어질 수 있는 변화를 사전 예측합니다.

- **Technical Details**: GDCPD는 고차원 데이터의 변화를 감지하기 위해 Gaussian 유도 과정을 사용하는 비모수적 방법입니다. 이 연구에서는 Weighted Mahalanobis Distance (WMD)를 오프라인 및 온라인 분석에 적용하여 시스템 변화의 중요성을 평가합니다. 온라인 설정에서 LSTM(Long Short-Term Memory) 네트워크를 활용하여 시스템의 Remaining Useful Life (RUL)를 추정합니다.

- **Performance Highlights**: 현실 세계 시스템의 실험 연구 결과, 제안된 방법론이 시스템 고장을 발생하기 전에 정확하게 예측할 수 있는 효과를 입증하였습니다. CPD와 실시간 모니터링, RUL 예측을 통합하여 시스템 건강 모니터링 및 조기 경고 기능을 크게 향상시킴으로써 유지보수 계획의 효과성을 높였습니다.



### FGCE: Feasible Group Counterfactual Explanations for Auditing Fairness (https://arxiv.org/abs/2410.22591)
- **What's New**: 이 논문은 모델 공정성을 감사하기 위한 그룹 반사적 설명 (group counterfactual explanations) 생성을 위한 첫 번째 그래프 기반 프레임워크인 FGCE(Feasible Group Counterfactual Explanations)를 소개합니다. 이는 신뢰할 수 있는 머신 러닝의 중요한 측면입니다.

- **Technical Details**: FGCE는 실제 적용 가능한 제약을 포착하고 유사한 반사적 설명을 가진 하위 그룹을 구성하여 기존 방법과 차별화됩니다. 이 프레임워크는 반사적 생성은 물론 비용과 범위 간의 균형을 포함한 주요 트레이드오프를 처리합니다.

- **Performance Highlights**: 실험 결과는 FGCE가 제공하는 공정성 감사에 대한 효과성을 보여 주며, 다른 기존 연구와 비교할 때 더욱 엄격한 실현 가능성 제약을 포함하면서도 우수한 성과를 발휘하는 것을 입증합니다.



### BENCHAGENTS: Automated Benchmark Creation with Agent Interaction (https://arxiv.org/abs/2410.22584)
- **What's New**: 본 연구에서는 BENCHAGENTS라는 프레임워크를 소개하며, 이는 대형 언어 모델(LLMs)을 활용하여 복잡한 기능을 평가하기 위한 벤치마크 생성 과정을 자동화합니다. 특히, 벤치마크 생성 과정을 계획, 생성, 검증, 평가의 네 가지 단계로 나누고 각각을 LLM 에이전트가 담당하여 데이터의 질과 메트릭 품질을 보장합니다.

- **Technical Details**: BENCHAGENTS는 다양한 LLM 에이전트를 사용하여 벤치마크 생성 프로세스를 관리합니다. 각 에이전트는 다음과 같은 역할을 수행합니다: Planning Agent는 벤치마크 계획을 수립하고, Data Generation Agent는 데이터를 생성하며, Verification Agent는 생성된 데이터의 품질을 확인하고, Evaluation Agent는 모델 성능 평가를 위한 메트릭을 생성합니다.

- **Performance Highlights**: BENCHAGENTS를 활용하여 생성된 두 개의 벤치마크(BA-Calendar 및 BA-Text)에 대해 최첨단 LLM 7개 모델을 평가한 결과, 모든 모델이 제약 조건 충족에 어려움을 겪고 있으며, 특히 제약 조건 수가 증가할수록 성능이 떨어지는 경향을 보였습니다. 이는 모델 간의 우선순위 설정에서 차이점을 보이며, 주로 수치적 혹은 논리적 추론을 요구하는 제약 조건에서 실패하는 경향이 있음을 나타냅니다.



### Flow Matching for Posterior Inference with Simulator Feedback (https://arxiv.org/abs/2410.22573)
Comments:
          Code available at this https URL

- **What's New**: 이 논문에서는 시뮬레이터 기반의 제어 신호를 추가하여 기존의 flow-based generative modeling 기법을 개선하는 방법을 제안합니다. 이를 통해 샘플링 및 likelihood 평가를 더 빠르게 수행할 수 있습니다.

- **Technical Details**: 제안된 방법에서는 flow 네트워크를 사전 훈련(pretrain)한 후 시뮬레이터의 피드백을 통해 미세 조정(finetuning)을 수행합니다. 이 과정에서 필요한 추가 매개변수는 적고 계산 속도 또한 크게 향상됩니다. 특히, 강한 중력 렌즈 시스템의 모델링에 대한 실험으로, 기존의 MCMC 방법들과 비교가 가능합니다.

- **Performance Highlights**: 실험 결과, 시뮬레이터의 피드백을 포함함으로써 샘플의 정확성이 53% 개선되었으며, 기존 기술보다 최대 67배 빠른 추론 속도를 달성하였습니다.



### Vertical Federated Learning with Missing Features During Training and Inferenc (https://arxiv.org/abs/2410.22564)
- **What's New**: 이 논문은 Vertical Federated Learning(VFL)에서의 특성 블록 손실 문제를 해결하기 위한 새로운 방법 LASER-VFL을 제안합니다. 이 방법은 모든 특성 파티션(모델의 입력 부분)이 항상 사용할 수 없다는 현실적 한계를 극복하고, 기계 학습 모델의 일반화 성능을 개선합니다.

- **Technical Details**: LASER-VFL은 비대칭 데이터 셋에서 수행되는 Vertical Federated Learning을 위한 방법으로, 다양한 특성 세트에 대해 효율적으로 훈련 및 추론할 수 있는 능력을 갖추고 있습니다. 모델 파라미터의 전략적 공유와 작업 샘플링(task-sampling)을 기반으로 한 이 접근법은 비공식적인 예측기 집합을 훈련시키는 데 효과적입니다.

- **Performance Highlights**: LASER-VFL은 비선형 최적화 문제에 대해 $	extbackslashmathcal{O}(	extbackslashfrac{1}{	extbackslashsqrt{T}})$의 수렴 속도를 보여주고, 배치 크기가 충분히 클 경우 $	extbackslashmathcal{O}(	extbackslashfrac{1}{T})$의 성능을 달성합니다. CIFAR-100 데이터셋에서 특성 블록이 손실된 경우에도 성능 향상을 보여주며, 예를 들어 각 특성 블록이 0.5의 확률로 관찰되었을 때 정확도가 21.4% 증가했습니다.



### Unpicking Data at the Seams: VAEs, Disentanglement and Independent Components (https://arxiv.org/abs/2410.22559)
- **What's New**: 이번 연구는 Variational Autoencoders (VAE)에서의 disentanglement(비관계화)에 대한 이론적 분석을 제공하며, encoder Jacobian의 직교성(orthogonality)과 데이터의 독립적인 구성 요소들을 효과적으로 식별하는 방법을 보여줍니다.

- **Technical Details**: 연구에서는 VAE의 posterior covariance matrix로 대각 행렬(diagonal covariance matrix)을 사용하여 decoder Jacobian의 열(column) 사이의 직교성을 증진시키는 방법을 논의하고, 이를 통해 선형 독립성(linear independence)이 통계적 독립성(statistical independence)으로 어떻게 연결되는지를 설명하고 있습니다. 이와 함께 β-VAE에서의 β 값이 disentanglement와 posterior collapse에 미치는 영향을 설명합니다.

- **Performance Highlights**: 이 연구는 VAE의 성능 향상과 함께 disentanglement 현상이 어떻게 나타나는지를 자세히 이해하고, 새로운 조건을 제시하여 VAE가 데이터 분포를 완전히 식별하는 상황을 설명하였습니다. 이는 최신 diffusion 모델에서도 중요한 역할을 하며, 이론과 실제 모두에서 효과적인 응용을 기대할 수 있습니다.



### Unsupervised Multimodal Fusion of In-process Sensor Data for Advanced Manufacturing Process Monitoring (https://arxiv.org/abs/2410.22558)
- **What's New**: 본 연구는 제조 공정에서의 멀티모달 (multimodal) 센서 데이터 융합을 위한 새로운 접근 방식을 제안합니다. Contrastive Language-Image Pre-training (CLIP) 모델에서 영감을 받아 라벨이 없는 데이터에서도 다양한 데이터 모달리티를 상관시킬 수 있는 대비 학습 (contrastive learning) 기법을 활용합니다.

- **Technical Details**: 다섯 가지 독특한 모달리티에 대해 인코더 (encoder)를 개발하였으며, 이에는 시각적 이미지 (visual imagery), 오디오 신호 (audio signals), 레이저 위치 (laser position, x 및 y 좌표), 레이저 전력 측정 (laser power measurements)이 포함됩니다. 이 고차원의 데이터셋을 저차원의 표현 공간 (representational spaces)으로 압축하여 공정 제어 (process control), 이상 탐지 (anomaly detection), 품질 보증 (quality assurance)과 같은 다운스트림 작업을 용이하게 합니다.

- **Performance Highlights**: 실험을 통해 우리의 접근 방식이 첨단 제조 시스템에서의 공정 모니터링 (process monitoring) 능력을 향상시킬 수 있는 잠재력을 보임을 보여주었습니다. 이 연구는 다양한 제조 환경과 센서 구성에 적응 가능한 유연하고 확장 가능한 멀티모달 데이터 융합 프레임워크를 제공하여 스마트 제조 (smart manufacturing)에 기여합니다.



### Hindsight Experience Replay Accelerates Proximal Policy Optimization (https://arxiv.org/abs/2410.22524)
Comments:
          12 pages. 10 Figures

- **What's New**: 본 논문에서는 Hindsight Experience Replay (HER)를 on-policy 강화 학습 알고리즘인 Proximal Policy Optimization (PPO)에 적용함으로써, sparse reward 환경에서의 샘플 효율성을 극대화 할 수 있음을 보여줍니다. HER의 전통적인 한계를 넘어 PPO와 HER의 결합이 가능함을 입증하였습니다.

- **Technical Details**: HER는 에피소드 후에 목표를 수정하여 샘플 효율성을 높이는 기술로, PPO와 결합할 경우, 환경의 상태와 행동을 기반으로 목표 재샘플링을 통해 더욱 효과적인 학습을 가능하게 합니다. 실험은 커스텀 predator-prey 환경에서 이루어졌으며, 다양한 행동 정책을 가진 Prey와의 상호작용을 통해 PPO-HER의 성능을 평가했습니다.

- **Performance Highlights**: PPO-HER 알고리즘은 다양한 predator-prey 환경에서 효과적으로 동작하며, 기존의 offline RL 알고리즘과 유사한 샘플 효율성을 보이고 더 나아가 계산 시간(clock-time) 효율성에서도 우수함을 입증했습니다. 특히, 기본적인 PPO-HER 구성만으로도 복잡한 환경에서 해결 가능한 결과를 도출하였습니다.



### Multimodal Structure Preservation Learning (https://arxiv.org/abs/2410.22520)
- **What's New**: 이번 연구에서는 다양한 데이터 모달리티의 클러스터링 구조를 활용하여 기계 학습 모델의 성능을 향상시키는 새로운 방법인 다중 모달 구조 보존 학습(Multimodal Structure Preservation Learning, MSPL)을 제안합니다. 이를 통해, 데이터의 유틸리티를 단일 모달리티에서 다른 모달리티로 확장하는 방법을 학습할 수 있습니다.

- **Technical Details**: MSPL은 세 가지 주요 목표를 설정합니다: (1) 표준 오토인코더와 같이 입력 데이터를 재구성하는 기능, (2) 입력 데이터가 지니는 차별적 정보를 활용하는 프리텍스트(Pretext) 작업, (3) 두 모달리티 간의 클러스터링 구조 정렬을 통한 구조 보존입니다. 이러한 목표를 통해, MSPL은 하나의 모달리티의 학습된 특성에 다른 모달리티의 구조 정보를 주입할 수 있습니다.

- **Performance Highlights**: MSPL은 합성 시계열(data) 데이터 세트에서 잠재 구조를 효과적으로 발견하고, 전체 유전자 시퀀싱과 항균제 저항성 데이터를 사용하여 클러스터를 복구하는 성능을 입증했습니다. 이 결과는 MSPL이 데이터 수집 비용을 상당히 줄이고, 다양한 데이터 모달리티 간의 유틸리티 격차를 효과적으로 해소할 수 있음을 보여줍니다.



### Unlocking Point Processes through Point Set Diffusion (https://arxiv.org/abs/2410.22493)
- **What's New**: 이 논문에서는 일반 메트릭 공간에서 임의의 포인트 프로세스를 모델링할 수 있는 Point Set Diffusion라는 새로운 확산 기반(latent variable model) 모델을 소개합니다. 기존의 모델들이 강한 제약을 받는 것과 달리, 우리의 접근법은 강도(intensity) 함수에 의존하지 않고 점 데이터와 노이즈 점 세트 간의 확률적(interpolate) 보간을 직접 학습하여 복잡한 조건적 작업을 효율적으로 수행할 수 있습니다.

- **Technical Details**: Point Set Diffusion은 메트릭 공간 상에서 임의의 점 프로세스를 캡처하기 위해 데이터 포인트 집합과 노이즈 포인트 집합 간의 확률적 보간(stochastic interpolation)을 학습하는 확산 기반(latent variable model)입니다. 우리의 모델은 점 집합을 효율적이고 병렬적으로 샘플링(sampling) 할 수 있으며, 메트릭 공간에서 정의된 임의의 조건적 작업을 해결하기 위한 생성(generation)을 가능하게 합니다.

- **Performance Highlights**: Point Set Diffusion은 공간 및 시공간 포인트 프로세스의 조건부 및 무조건적 생성에서 최첨단 성능(state-of-the-art performance)을 달성하였으며, autoregressive 기초 모델에 비해 수량적으로 더 빠른 샘플링 속도를 제공합니다.



### Learning Identifiable Factorized Causal Representations of Cellular Responses (https://arxiv.org/abs/2410.22472)
- **What's New**: 이번 연구에서는 Factorized Causal Representation (FCR) 학습 방법을 제안하여 단일 세포 교란 데이터에서의 인과 구조를 밝혀냅니다. FCR은 치료, 세포 배경 및 상호작용을 별도로 분석할 수 있도록 세 가지 블록으로 나뉘어진 세포 표현을 학습합니다.

- **Technical Details**: FCR은 identifiable deep generative models의 프레임워크를 기반으로 하며, covariate-specific ($\mathbf{z}_x$), treatment-specific ($\mathbf{z}_{t}$), interaction-specific ($\mathbf{z}_{tx}$) 블록으로 구성됩니다. 비선형 ICA 이론을 근거로, 우리는 $\mathbf{z}_{tx}$의 구성 요소별 식별 가능성과 $\mathbf{z}_{t}$ 및 $\mathbf{z}_{x}$의 블록별 식별 가능성을 증명했습니다. FCR 방법은 variational auto-encoder (VAE) 프레임워크에 기초하며, 적대적 정규화(adversarial regularization)를 적용하였습니다.

- **Performance Highlights**: 실험을 통해 FCR 방법이 4개의 단일 세포 데이터셋에서 다양한 작업에서 최신 방법들에 비해 우수한 성능을 발휘함을 입증하였습니다.



### Power side-channel leakage localization through adversarial training of deep neural networks (https://arxiv.org/abs/2410.22425)
- **What's New**: 이 연구에서는 DNN(Deep Neural Network) 기반의 사이드 채널 공격에 대응하기 위해, 전력(trace) 신호에서 어떤 타임스텝이 암호화 키 유출에 책임이 있는지를 식별하는 기법을 제안합니다. 이는 DNN 기반 공격자와 훈련 가능한 노이즈 생성기 간의 적대적 게임 형태로 이루어집니다.

- **Technical Details**: 제안된 기법은 DNN 분류기가 전력 신호를 사용해 민감한 변수를 예측하는 방법과 노이즈 생성기가 최대한 적은 노이즈를 전력 신호에 추가하면서 분류기 성능을 저하시키지 않도록 최적화하는 방식으로 구성됩니다. 이를 통해 빠른 성능 저하는 피하면서 유출된 타임스텝을 식별할 수 있습니다.

- **Performance Highlights**: 합성 데이터셋에서는 기존의 기술들보다 우수한 성능을 보였고, 일반적인 카운터메저(예: Boolean masking, trace desynchronization) 환경에서도 효과적임을 입증했습니다. 그러나, 실제 데이터셋에서는 초매개변수 및 조기 중지 지점에 민감하여 모델 선택을 위한 적절한 데이터셋이 부족한 점이 한계로 작용하였습니다.



### A Large Recurrent Action Model: xLSTM enables Fast Inference for Robotics Tasks (https://arxiv.org/abs/2410.22391)
- **What's New**: 이 연구에서는 최신 반복 아키텍처인 xLSTM을 중심으로 한 대규모 반복 행동 모델(Large Recurrent Action Model, LRAM)을 제안하였습니다. LRAM은 선형 시간 복잡성을 가지며, 기존의 Transformer 기반 모델에 비해 성능과 속도에서 유리한 결과를 보였다는 점이 특징입니다.

- **Technical Details**: LRAM은 432개 작업과 6개 도메인에서 훈련되었으며, 전문 학습 설정을 통해 수집된 대규모 다중 도메인 데이터셋에서 효과적으로 학습되었습니다. 또한, xLSTM과 Mamba와 같은 현대의 반복 아키텍처는 훈련 시 Transformer와 유사한 병렬화 특성을 가지면서 인퍼런스(inference)에서는 더 빠른 속도를 제공합니다.

- **Performance Highlights**: 실험 결과에 따르면 LRAM은 기존의 Transformer 모델에 비해 성능과 속도 모두에서 우수한 결과를 보여주었으며, 다중 작업, 미세 조정(fine-tuning), 문맥 학습(in-context learning) 능력에서도 뛰어난 성과를 나타냈습니다.



### FNDEX: Fake News and Doxxing Detection with Explainable AI (https://arxiv.org/abs/2410.22390)
- **What's New**: 본 논문은 기존에 널리 연구되었지만 상관관계가 명확히 분석되지 않은 가짜뉴스(fake news)와 도킹(doxxing) 간의 교차점을 탐구합니다. 이 연구는 세 개의 서로 다른 transformer 모델을 활용하여 둘 다 높은 성능으로 탐지하는 새로운 시스템인 FNDEX 시스템을 제안합니다.

- **Technical Details**: FNDEX 시스템은 가짜뉴스와 도킹 탐지를 위한 효과적인 접근법을 제시하며, 개인 식별 정보(PII)의 패턴 기반 익명화(anonymization) 프로세스를 포함합니다. 이 시스템은 다양한 transformer 모델을 평가하여 두 현상의 탐지 성능을 극대화합니다.

- **Performance Highlights**: 실험 결과, FNDEX 시스템은 기존의 기준선 모델에 비해 현저한 성과를 보이며, 가짜뉴스와 도킹 탐지에서 효과적인 결과를 도출하는 것을 입증했습니다.



### Robust training of implicit generative models for multivariate and heavy-tailed distributions with an invariant statistical loss (https://arxiv.org/abs/2410.22381)
- **What's New**: 본 논문에서는 기존의 전통적인 implicit generative models의 한계를 극복하기 위해 새로운 방법론인 Pareto-ISL을 제안합니다. 이 방법론은 heavy-tailed 분포를 효과적으로 모델링할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 기존의 invariant statistical loss (ISL) 방법을 확장하여 heavy-tailed 및 다변량 데이터 분포를 처리합니다. 특히, generalized Pareto distribution (GPD)에서 노이즈를 추출하여 생성모델을 학습시키며, 새로운 손실 함수와 알고리즘을 도입하여 높은 차원의 데이터도 처리할 수 있도록 개선하였습니다.

- **Performance Highlights**: Pareto-ISL은 heavy-tailed 분포의 중앙 및 끝부분을 정확히 모델링하며, 다변량 데이터에 대해서도 우수한 성능을 보여줍니다. 다양한 하이퍼파라미터 설정에서도 목표 분포의 완전한 지원을 캡처하는데 성공하며, pretrained generative adversarial networks (GANs)에 활용할 때 특히 강인한 성능을 발휘합니다.



### Discrete Modeling via Boundary Conditional Diffusion Processes (https://arxiv.org/abs/2410.22380)
Comments:
          NeuraIPS 2024 poster

- **What's New**: 이번 연구에서는 효율적이고 효과적으로 강력한 continuous diffusion processes를 discrete modeling에 확장하는 새로운 프레임워크를 제안합니다. 기존 방법들은 discrete 데이터와 continuous modeling 간의 불일치 문제를 겪었습니다.

- **Technical Details**: 우리는 먼저 경계를 prior distribution으로 추정한 후, forward trajectory를 재조정하여 boundary conditional diffusion model을 구축하는 두 단계의 forward 과정이 필요함을 제안합니다. 이 과정에서 Ordinary Differential Equations (ODEs)를 사용하여 forward trajectory를 설명합니다.

- **Performance Highlights**: 실험 결과, 우리의 방법은 언어 모델링과 discrete 이미지 생성 작업 모두에서 강력한 성능을 보였습니다. 특히, 언어 모델링에서는 기존 continuous diffusion 모델을 초월하여 우수한 성능을 보여주었으며, Cifar-10 데이터 세트에서 새로운 state-of-the-art를 수립했습니다.



### A Systematic Literature Review of Spatio-Temporal Graph Neural Network Models for Time Series Forecasting and Classification (https://arxiv.org/abs/2410.22377)
- **What's New**: 최근 연구에서는 spatio-temporal graph neural networks (GNNs)의 다양한 모델링 접근 방식과 응용 분야에 대한 포괄적 개요를 제공하고자 하는 체계적인 문헌 검토가 진행되었습니다. 이제까지의 문헌에서 150편 이상의 저널 논문을 조사하여, GNNs가 시계열 분류 및 예측에서의 효용을 체계적으로 비교했습니다.

- **Technical Details**: 이 리뷰는 GNNs의 시계열 관련 과제, 특히 시계열 분류와 예측에 중점을 두고 있습니다. GNN은 복잡한 관계를 포착할 수 있는 능력으로, 서로 다른 변수 간의 inter-variable 관계와 시간 간의 inter-temporal 관계를 동시에 캡처할 수 있습니다. 결과적으로, spatio-temporal GNNs는 공간 차원과 시간 차원을 모두 처리하도록 설계된 모델입니다.

- **Performance Highlights**: 제안된 다양한 GNN 모델과 벤치마크 모델의 결과를 포괄적으로 수집하여 향후 연구에 유용한 리소스와 참조 자료로 제공하고자 합니다. 이 리뷰는 현재 GNN 모델의 적용 분야와 성능에 대한 깊이 있는 통찰을 제공합니다.



### Rare-to-Frequent: Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with LLM Guidanc (https://arxiv.org/abs/2410.22376)
- **What's New**: 본 연구는 희귀한 개념에 대한 텍스트-이미지(T2I) 분산 모델의 구성능력을 높이기 위해 대형 언어 모델(LLM) 가이드를 활용하는 접근 방식을 제안합니다. 우리는 창의적이고 비범한 특성을 가진 새로운 캐릭터 디자인과 같은 희귀한 프롬프트 생성 과정에서 기존 모델들이 어려움을 겪는 문제를 강조합니다.

- **Technical Details**: 본 연구에서는 희귀한 개념 구성을 위한 새로운 접근 방식인 R2F(Rare-to-Frequent)를 제안합니다. R2F는 LLM을 활용하여 희귀한 컨셉과 관련된 보다 일반적인 빈번한 개념을 찾고, 이를 통해 분산 추론(difussion inference) 과정을 개선합니다.

- **Performance Highlights**: R2F는 다양한 희귀 개념 구성을 포함하는 새로운 벤치마크인 RareBench에서 기존 모델인 SD3.0 및 FLUX와 비교하여 T2I 정렬 정확도에서 최대 28.1%p 향상된 성능을 보였습니다.



### Machine Unlearning using Forgetting Neural Networks (https://arxiv.org/abs/2410.22374)
- **What's New**: 본 논문에서는 머신 언러닝(Machine Unlearning) 문제를 해결하기 위해 Forgetting Neural Networks (FNN)을 제안합니다. FNN은 망각을 위한 특정 레이어를 포함하는 신경망으로, 인간의 망각 과정에서 영감을 받았습니다. 이전에 이론적으로만 제안된 FNN이 머신 언러닝 방법으로 실제로 활용된 것은 이번이 처음입니다.

- **Technical Details**: FNN은 Ebbinghaus 망각 곡선을 기반으로 한 망각 레이어를 포함하여, 머신 러닝 모델의 훈련 데이터 중 특정 데이터를 효과적으로 잊게 하도록 설계되었습니다. 본 연구에서는 MNIST 손글씨 숫자 인식과 패션 데이터셋을 이용하여 FNN을 구현하고, Membership Inference Attacks (MIA)를 통해 언러닝된 모델의 효과성을 평가했습니다.

- **Performance Highlights**: 실험 결과, 언러닝된 모델이 MIA 점수가 0.5에 가까운 값을 기록해, 테스트 데이터와 망각 데이터의 분포를 구분하지 못함을 나타내었습니다. 이는 제안된 방법이 머신 언러닝 문제를 처리하는 데 큰 잠재력이 있음을 보여줍니다.



### Analytic Continual Test-Time Adaptation for Multi-Modality Corruption (https://arxiv.org/abs/2410.22373)
- **What's New**: 이번 논문에서는 Multi-Modality Dynamic Analytic Adapter (MDAA)라는 새로운 접근 방식을 제안하여 Multi-Modal Continual Test-Time Adaptation (MM-CTTA) 과제에서 발생하는 오류 축적, 재앙적 망각(catastrophic forgetting), 신뢰성 편향(reliability bias)을 처리하는 방법을 제시합니다. 특히, Analytic Classifiers (AC), Dynamic Selection Mechanism (DSM), Soft Pseudo-label Strategy (SPS) 등을 도입하여 안정적인 모델 적응을 지원합니다.

- **Technical Details**: MDAA는 Analytic Classifiers (AC)를 통해 모델이 재앙적 망각을 방지하고, Dynamic Selection Mechanism (DSM)을 통해 각 AC의 출력 신뢰성을 기반으로 선택적으로 업데이트하여 신뢰성 편향을 완화합니다. Soft Pseudo-label Strategy (SPS)는 다양한 라벨에 대해 변동 확률을 부여하여 라벨 노이즈(Label noise)에 대한 모델의 강건성을 높입니다. MDAA는 복수 모드의 정보를 통합하여 다이나믹하게 필터링하며, 테스트 데이터에 적응합니다.

- **Performance Highlights**: 실험 결과, MDAA는 MM-CTTA 작업에서 최첨단 성능(SOTA)을 달성하며, 두 개의 과제에서 각각 이전 방법보다 최대 6.22% 및 6.84% 향상된 성능을 보였습니다.



### A Hierarchical Language Model For Interpretable Graph Reasoning (https://arxiv.org/abs/2410.22372)
- **What's New**: 이 논문에서는 그래프 작업을 위한 새로운 계층적 언어 모델인 HLM-G를 제안합니다. 이 모델은 노드 중심의 지역 정보와 상호작용 중심의 글로벌 구조를 효과적으로 캡처하기 위해 두 개의 블록 아키텍처를 사용합니다.

- **Technical Details**: HLM-G는 지역 블록과 글로벌 블록으로 구성되어 있으며, 각 블록은 특정한 attention masking을 사용합니다. 이 구조는 모델이 초기에는 로컬 정보를 캡처하고 후에 글로벌 정보를 통합하도록 허용하여 그래프 구조의 이해 능력을 향상시킵니다. 또한, LLM의 대규모 그래프 작업에 대한 계산 비용을 크게 줄입니다.

- **Performance Highlights**: HLM-G의 성능은 7개의 그래프 추론 데이터셋과 7개의 실제 데이터셋에서 종합 평가를 통해 검증되었습니다. 이 모델은 노드, 링크, 그래프 수준의 작업에서 효과적으로 일반화 할 수 있는 능력을 발휘하여 LLM을 그래프 기반 작업에 적용하는 데 있어 중요한 발전을 이루었습니다.



### Error Bounds for Deep Learning-based Uncertainty Propagation in SDEs (https://arxiv.org/abs/2410.22371)
Comments:
          pre-print under review

- **What's New**: 이 논문은 Stochastic Differential Equations (SDEs)에 의해 모델링된 과정의 확률 밀도 함수 (PDF)를 근사하는 방법을 제안합니다. 특히, Physics-Informed Neural Networks (PINNs)를 사용하여 근사 오차를 엄밀하게 바운딩하는 새로운 방법론을 개발했습니다.

- **Technical Details**: 연구의 주요 내용은 두 가지 PINNs를 사용하여 근사 오차의 재귀적 함수 시리즈를 통해 공간과 시간에 대해 타이트한 오차 경계를 설정하는 것입니다. 이 과정에서 PINN의 잔여 값이 Fokker-Planck 편미분 방정식 (FP-PDE)에 의해 지배된다는 통찰력을 제시했습니다.

- **Performance Highlights**: 수치 실험을 통해 제안된 방법론의 타당성을 검증했으며, 단일 PINN만으로도 실용적인 오차 바운드를 도출할 수 있음을 보여주었습니다. 이로 인해 복잡한 SDE 문제들을 다루는 데 있어 계산 효율성과 정확성을 함께 향상시키는 데 기여했습니다.



### Unpacking SDXL Turbo: Interpreting Text-to-Image Models with Sparse Autoencoders (https://arxiv.org/abs/2410.22366)
- **What's New**: 이 연구에서는 sparse autoencoders (SAEs)를 이용하여 텍스트-이미지 모델의 해석 가능한 특징을 학습할 수 있는 가능성을 조사하였습니다. 특히 SDXL Turbo와 같은 몇 단계의 diffusion 모델에서의 적용을 탐구하였습니다.

- **Technical Details**: SAEs는 SDXL Turbo의 denoising U-net 내에서 transformer 블록의 업데이트에 대해 훈련되었습니다. 이 과정에서 학습된 특징들은 해석 가능하고, 생성 과정에 인과적으로 영향을 미치며, 블록 간의 전문화(specialization)를 드러냈습니다.

- **Performance Highlights**: 연구에서는 이미지 구성, 지역 세부사항 추가, 색상 및 조명, 스타일을 담당하는 블록을 식별하였습니다. 이는 텍스트-이미지 생성 모델의 내부 구조를 이해하는 데 중요한 첫 걸음이 됩니다.



### Bridging the Human to Robot Dexterity Gap through Object-Oriented Rewards (https://arxiv.org/abs/2410.23289)
- **What's New**: HuDOR는 단일 인간 비디오와 손 자세 경로를 사용하여 다중 손가락 로봇 손의 손재주 정책(Policy)을 학습할 수 있는 첫 번째 프레임워크를 제안합니다.

- **Technical Details**: HuDOR는 세 가지 단계로 구성됩니다: (1) VR 헤드셋과 RGB 카메라를 사용하여 인간 비디오와 대응하는 손 자세 경로를 기록합니다; (2) 포즈 전환(pose transformation)과 로봇의 역 기하학(inverse kinematics, IK)을 사용해 손 자세를 로봇에 전송하고 실행합니다; (3) 강화 학습(reinforcement learning, RL)을 통해 전문가의 경로를 성공적으로 모방합니다.

- **Performance Highlights**: HuDOR는 세 가지 작업에서 일반적인 보상 함수보다 2.1배 더 나은 성능을 발휘하였고, 오프라인 모방 학습 방법과 비교하여 평균 2.64배의 성능 향상을 보여주었습니다.



### SlowFast-VGen: Slow-Fast Learning for Action-Driven Long Video Generation (https://arxiv.org/abs/2410.23277)
- **What's New**: 본 논문에서는 SlowFast-VGen이라는 새로운 이중속도 학습 시스템을 소개합니다. 이는 행동 주도의 긴 비디오 생성을 위해 느린 학습과 빠른 학습을 통합하여, 일관되고 응답적인 비디오 생성을 가능하게 합니다.

- **Technical Details**: 이 시스템은 두 가지 주요 구성 요소로 이루어져 있습니다. 느린 학습(적용된 조건부 비디오 확산 모델)과 빠른 학습(Temporal LoRA 모듈 기반의 추론 시간 학습 전략)입니다. 빠른 학습 과정에서는 입력 및 출력에 기반하여 Temporal LoRA 매개변수를 업데이트하여 에피소드 메모리를 저장합니다. 또한, 느린 학습 알고리즘과 빠른 학습 루프를 결합하여 다중 에피소드 경험을 기억하여 기술 학습을 지원합니다.

- **Performance Highlights**: SlowFast-VGen은 FVD 점수에서 514를 기록하며 782을 초월하여 이전 모델들보다 뛰어난 성능을 보였습니다. 또한 0.37의 평균 장면 전환수로, 긴 비디오의 일관성을 유지하는 데 있어 우수함을 보여주었습니다.



### Conditional Forecasting of Margin Calls using Dynamic Graph Neural Networks (https://arxiv.org/abs/2410.23275)
- **What's New**: 본 논문에서는 동적 금융 네트워크의 조건부 'm'-단계 예측 문제를 해결하기 위해 새로운 동적 그래프 신경망(DGNN) 아키텍처를 도입했습니다.

- **Technical Details**: DGNN은 시간적 변화가 있는 네트워크에서 작동하는 그래프 신경망(GNN)의 확장입니다. 논문에서는 재정 거래 데이터에서 발생하는 마진 호출을 정량화하기 위한 예방적 체계 예측 네트워크 프레임워크 개발의 필요성을 강조합니다. DGNN은 각각의 스냅샷에서 neighborhood aggregation 기법과 LSTM 네트워크를 결합하여 동적으로 그래프를 업데이트합니다.

- **Performance Highlights**: 제안된 DGNN 모델은 스타일화된 이자율 교환 거래 네트워크에서 조건부 정보를 활용하여 최대 21일 간의 순 변동 마진에 대한 정확한 조건부 예측을 수행하는 데 성공했습니다. 이는 시스템 리스크 모니터링의 중요한 도구로 작용하여 규제 기관과 정책 입안자에게 활용될 수 있습니다.



### EMMA: End-to-End Multimodal Model for Autonomous Driving (https://arxiv.org/abs/2410.23262)
Comments:
          Blog post: this https URL

- **What's New**: EMMA(End-to-end Multimodal Model for Autonomous driving)는 모듈식 접근법 대신, 센서 데이터를 직접 처리하여 운전 관련 작업을 수행하는 새로운 모델입니다. 다양한 센서 데이터를 자연어 텍스트로 변환하여 통합적으로 처리할 수 있는 기능이 특징입니다.

- **Technical Details**: EMMA는 미리 훈련된 멀티모달 대형 언어 모델(Gemini) 위에 구축되었으며, 카메라 이미지 및 자연어 텍스트를 입력으로 받아 특정 운전 작업을 수행합니다. EMMA는 비전 모듈과 행동 모듈 간의 기호 인터페이스를 제거하여 각 운전 목표를 공동으로 최적화합니다.

- **Performance Highlights**: EMMA는 nuScenes 데이터셋에서 최첨단 성능을 달성하였으며, Waymo Open Motion Dataset에서도 경쟁력 있는 결과를 보였습니다. 또한, 3D 객체 탐지 및 도로 그래프 추정 등 여러 인식 작업에서도 뛰어난 성능을 보였습니다. 하지만, 이미지 프레임 수 처리의 제한, LiDAR와 같은 3D 감지 모드의 부재로 인한 문제도 존재합니다.



### $100K or 100 Days: Trade-offs when Pre-Training with Academic Resources (https://arxiv.org/abs/2410.23261)
- **What's New**: 이 논문에서는 학술 연구자들이 자원 부족으로 인해 모델의 사전 훈련이 어렵다는 일반적인 가정을 의문시합니다. 연구자들의 컴퓨팅 자원을 조사하고, 이를 바탕으로 주어진 GPU에서 모델을 재현하는 데 필요한 시간을 실증적으로 측정합니다.

- **Technical Details**: 논문은 2000 GPU-시간을 투자하여 3000개 이상의 구성에서 성능을 최적화하고, 최적의 훈련 설정을 찾기 위해 여러 모델에 대한 벤치마크를 수행하였습니다. 예를 들어, Pythia-1B 모델은 원래 64개의 GPU로 3일 걸려 훈련되었지만, 4개의 GPU로 18일 만에 재현할 수 있음을 보여줍니다. 또한, 비용-편익 분석을 통해 제한된 예산 내에서 가장 효과적인 하드웨어 구성도 제안합니다.

- **Performance Highlights**: 이 연구는 현재 하드웨어와 최적화를 활용하여 사전 훈련을 위한 GPU 사용을 3배 줄일 수 있음을 보여줍니다. 즉, 저예산 환경에서도 더 큰 모델에 대한 훈련을 가능하게 하며, 더 많은 연구자들이 실험을 할 수 있도록 지원합니다.



### bit2bit: 1-bit quanta video reconstruction via self-supervised photon prediction (https://arxiv.org/abs/2410.23247)
Comments:
          NeurIPS 2024

- **What's New**: 이 논문에서는 photon detection events를 1-bit 배열로 나타내는 새로운 sensor 기술인 Quanta image sensors를 다룹니다. 특히, 이 연구는 sparse binary quanta image data에서 원래의 spatiotemporal 해상도로 고품질 이미지 스택을 재구성하는 bit2bit이라는 새로운 방법을 제안합니다.

- **Technical Details**: bit2bit 방법은 photon arrival location의 확률 분포를 예측하여 sparse binary photon data로부터 밀집 이미지 시퀀스를 생성하는 알고리즘입니다. 데이터의 이진성 때문에 Poisson 분포의 가정이 부족하다는 것을 보여주고, 대신 truncated Poisson의 Bernoulli 격자 과정을 모델로 사용합니다. 자가 지도 학습(self-supervised learning) 구조에 기반하여 masked loss function을 통해 이진 데이터를 처리합니다.

- **Performance Highlights**: 모의 데이터에서 우리는 (<0.06 photons per pixel per frame) 매우 photon-sparse binary 입력에 대해 34.35의 평균 PSNR을 달성했습니다. 실제 SPAD 고속 비디오의 새로운 데이터셋을 제작하여 다양한 도전적인 이미징 조건 속에서 우리의 접근 방식의 잠재력을 입증하였고, reconstruction quality와 throughput에서 기존 방법들을 크게 초월했습니다.



### Very fast Bayesian Additive Regression Trees on GPU (https://arxiv.org/abs/2410.23244)
Comments:
          Check out the software at this https URL

- **What's New**: 이번 논문에서는 Bayesian Additive Regression Trees (BART)의 GPU 지원 구현을 소개하며, 이는 단일 CPU 코어에 비해 최대 200배 빠른 속도를 자랑합니다. 이를 통해 BART가 XGBoost와 비교하여 실행 시간 경쟁력을 갖출 수 있게 되었습니다.

- **Technical Details**: BART는 의사결정트리(Decision Trees) 앙상블에 기반한 비모수 베이지안 회귀 기법으로, 보통 높은 통계적 품질을 유지하면서 수동 조정이 적은 장점을 지니고 있습니다. 그러나 BART는 긴 실행 시간으로 인해 10,000-100,000 이상의 샘플 크기를 다루는 데 어려움이 있습니다. 이 연구에서는 이러한 문제를 해결하기 위한 GPU 구현을 제공합니다.

- **Performance Highlights**: 새로운 GPU 구현으로 BART의 성능을 대폭 개선하여 XGBoost와 유사한 실행 시간을 제공합니다. 이 구현은 Python 패키지 bartz에서 사용할 수 있습니다.



### Full-waveform earthquake source inversion using simulation-based inferenc (https://arxiv.org/abs/2410.23238)
Comments:
          22 + 11 pages, 11 + 11 figures

- **What's New**: 이 논문은 시뮬레이션 기반 추론(Simulation-Based Inference, SBI)을 사용하는 전파 형식의 지진원 역전 기술을 위한 새로운 프레임워크를 제시합니다. 기존의 확률적 접근법은 데이터 오류에 대한 단순화된 가정에 의존하여 불확실성 평가에서 부정확한 결과를 초래할 수 있습니다.

- **Technical Details**: SBI는 머신러닝 모델을 사용하여 데이터 오류의 경험적 확률 모델을 구축하고, 이를 베이esian 추론(Bayesian Inference) 프레임워크에 통합합니다. 이 연구에서 저자들은 포인트 소스(moment tensor) 역전과 시간-위치 동시 역전에 SBI 프레임워크를 적용하며, 다양한 합성 예제를 통해 SBI 솔루션의 품질을 평가합니다. 또한 SBI 결과를 가우시안 우도 기반 베이esian 역전와 비교합니다.

- **Performance Highlights**: 실제 지진 소음 하에서 전파 형식 데이터를 처리하기 위한 일반적인 가우시안 우도 가정은 모멘트 텐서 구성 요소의 불확실성을 최대 3배까지 과소평가하는 과신(posteriors) 분포를 초래하는 반면, SBI는 잘 보정된 후행확률(posteriors)을 생성하여 실제 지진원 매개변수와의 일치를 보입니다. 기존 몬테 카를로 기술과 비교할 때 추론 수행에 필요한 시뮬레이션 수에서도 주문의 크기 감소를 제공합니다.



### Aligning Audio-Visual Joint Representations with an Agentic Workflow (https://arxiv.org/abs/2410.23230)
- **What's New**: 본 연구에서는 오디오-비주얼(Audio-Visual, AV) 데이터 간의 정렬 문제를 해결하기 위해 LLM 기반의 AVAgent를 제안합니다. AVAgent는 멀티모달 LLM을 사용하여 오디오와 비주얼 데이터를 언어 설명으로 변환하고, 이 데이터를 기반으로 정렬 여부를 판단하며 필요시 오디오 신호를 편집합니다.

- **Technical Details**: 제안된 방법에서 AVAgent는 도구 사용(tool use), 계획(planning), 반영(reflection) 단계를 순환적으로 수행하여 오디오 신호를 시각 데이터에 점진적으로 정렬합니다. 이 과정에서 AVAgent는 배경 잡음을 제거하고 데이터를 보강하는 전처리 작업을 수행하며, 이러한 작업 후 VLM(비전-언어 모델)을 통해 편집된 오디오 신호가 비디오와 잘 맞는지 평가합니다.

- **Performance Highlights**: Flick-SoundNet, VGG-Instruments 등 다양한 데이터 세트를 통해 실험한 결과, 제안된 방법이 기존 기준선에 비해 우수한 성능을 보였습니다. 특히, 선형 프로빙(linear probing), 미세 조정(fine-tuning) 분류, 비주얼 사운드 로컬라이제이션(sound localization) 등 다양한 다운스트림 작업에서 최첨단 성능을 입증했습니다.



### Improved convergence rate of kNN graph Laplacians (https://arxiv.org/abs/2410.23212)
- **What's New**: 이 논문에서는 $k$-nearest neighbor ($k$NN) 그래프의 확장을 다루고 있으며, 가중치가 있는 엣지를 허용하여 커널화된 그래프 친화성을 제공하고 있습니다.

- **Technical Details**: 본 연구에서는 일반적인 $k$NN 그래프 클래스와 그 친화성을 다루며, 친화성 함수는 $W_{ij} = \\epsilon^{-d/2} \, k_0 ( \, \, \| x_i - x_j \|^2 / \epsilon \, \phi( \widehat{\rho}(x_i), \widehat{\rho}(x_j) )^2 )$ 형태입니다. 여기서 $\, \widehat{\rho}(x)$는 $k$NN 거리이고, $\, \phi$는 대칭 이변량 함수입니다. 논문에서는 매니폴드 데이터 설정 하에 $O(N^{-2/(d+6)})$의 수렴 속도를 보이며, 이는 이론적 편향(bias)과 분산(variance) 오류를 균형을 맞추기 위해 최적의 값에서 이루어집니다.

- **Performance Highlights**: 저자들은 $k_0$와 $\phi$의 정규성이 높을 경우의 수렴 속도가 개선되었음을 보였으며, 시뮬레이션 데이터를 통한 수치 실험으로 이론을 검증하였습니다.



### Uncertainty quantification for fast reconstruction methods using augmented equivariant bootstrap: Application to radio interferometry (https://arxiv.org/abs/2410.23178)
Comments:
          13 pages, 7 figures. Accepted at the Machine Learning and the Physical Sciences Workshop, NeurIPS 2024

- **What's New**: 이번 논문에서는 차세대 전파 간섭계(Square Kilometer Array)가 제공하는 방대한 데이터에 대한 새로운 이미지 재구성 알고리즘을 제안합니다. 이 알고리즘은 빠르고 정확한 이미지 재구성을 통해 전파 천문학의 관측 능력을 혁신하려고 합니다.

- **Technical Details**: 우리는 비지도(unsupervised) 기술을 사용하여 전파-증강(equivariant) 부트스트래핑 방법의 형태를 정렬(conformalized)하여 신뢰할 수 있으며 확장 가능한 불확실성(uncertainty) 평가를 할 수 있는 방법을 개발했습니다. 특히 우리는 초고속(unrolled) 알고리즘의 재구성을 의존하여 불확실성을 정량화하였습니다.

- **Performance Highlights**: 제안된 방법은 기존의 대안보다 더 신뢰할 수 있는 불확실성 추정치를 제공하여 전파 간섭 촬영 문제를 다루는 데 효과적입니다.



### Functional Gradient Flows for Constrained Sampling (https://arxiv.org/abs/2410.23170)
Comments:
          NeurIPS 2024 camera-ready (30 pages, 26 figures)

- **What's New**: 이번 논문에서는 파라미터 기반 변분 추론(ParVIs) 방법을 확장하여 제약된 샘플링을 위한 경계 조건( boundary condition)을 도입합니다. 이를 통해 특정 도메인 내에서 입자들이 제한되도록 하기 위한 새로운 함수적 기울기 방법인 제약된 함수적 기울기 흐름(constrained functional gradient flow, CFG)을 제안합니다.

- **Technical Details**: 이 연구는 변분 추론과 MCMC 방법을 통합하여 입자 기반 변분 추론 방법인 ParVIs를 개선하는 접근법을 제안합니다. CFG 방법은 특정 도메인에 대한 경계 조건을 따라 입자의 분포적 기울기를 학습하여 샘플링을 수행합니다. 곱적 수치적 전략(numerical strategies)을 통해 도메인 제약에서 발생하는 경계 적분 항(boundary integral term)을 효율적으로 처리할 수 있는 방법도 제시합니다.

- **Performance Highlights**: 제안된 CFG 방법은 다양한 제약 머신러닝 문제에서 실험을 통해 효과성과 효율성을 입증하며, 제약된 도메인에서의 샘플링 문제 해결에 있어 기존 방법들보다 우수한 성능을 보입니다.



### SciPIP: An LLM-based Scientific Paper Idea Proposer (https://arxiv.org/abs/2410.23166)
Comments:
          25 pages, 5 figures, 19 tables

- **What's New**: 이 논문은 과학 논문 아이디어 제안기인 SciPIP를 제안합니다. SciPIP는 사용자 제공 연구 배경을 바탕으로 유용한 논문을 검색하고, 이를 바탕으로 더 새롭고 실행 가능한 아이디어를 생성하는 방식으로 기존의 대형 언어 모델(LLM)의 잠재력을 활용합니다.

- **Technical Details**: SciPIP는 사용자가 제공한 연구 배경을 기반으로 문헌을 검색하여 아이디어를 제안하는 시스템입니다. 이를 위해 문헌 검색 데이터베이스를 구축하고, 의미론(semantic), 엔티티(entity), 인용의 공동 출현(citation co-occurrence)을 기반으로 문헌 검색을 수행합니다. 이후에는 문헌에서 얻은 정보를 활용하여 솔루션을 유추하거나 독창적인 아이디어를 생성하는 두 가지 경로를 통해 아이디어를 제안합니다.

- **Performance Highlights**: NLP 분야에서 진행된 광범위한 실험을 통해 SciPIP는 기존의 상위 회의 논문들과 유사한 인용을 검색하고, 많은 아이디어를 생성함으로써 그 효과를 입증하였습니다. 또한 SciPIP에 의해 생성된 아이디어는 청사진의 참조를 유지하면서도 혁신성과 실행 가능성을 확보하고 있음을 평가해 보여줍니다.



### Fourier Amplitude and Correlation Loss: Beyond Using L2 Loss for Skillful Precipitation Nowcasting (https://arxiv.org/abs/2410.23159)
Comments:
          Accepted by NeurIPS 2024. Camera-ready submission

- **What's New**: 본 연구에서는 기존의 모델 아키텍처에 의존하지 않고, 새로운 손실 함수인 Fourier Amplitude and Correlation Loss (FACL)을 제안합니다. 이 손실 함수는 예측의 고주파 패턴을 복원하는 데 중점을 두며, 전통적인 MSE 손실을 대신하여 더 선명한 강수 예측을 가능하게 합니다.

- **Technical Details**: FACL은 두 가지 주요 손실 항목으로 구성됩니다: Fourier Amplitude Loss (FAL)와 Fourier Correlation Loss (FCL). FAL은 예측의 푸리에 진폭을 정규화하고, FCL은 누락된 위상 정보를 보완합니다. 이러한 손실 항목의 결합은 명암도와 공간 구조를 더욱 향상시키며, FAL과 FCL 사이의 학습 메커니즘을 통해 점진적으로 훈련됩니다.

- **Performance Highlights**: 실험 결과, FACL은 MSE로 기존 예측보다 더 사실적이고 기술적인 성능이 우수한 결과를 보여주었습니다. 또한, 새로운 평가 지표인 Regional Histogram Divergence (RHD)를 통해 이미지 패턴 간의 유사성을 더욱 정량적으로 측정할 수 있게 되었습니다. 이러한 개선은 강수 예측의 정확성을 높이는 데 기여하고 있습니다.



### VisualPredicator: Learning Abstract World Models with Neuro-Symbolic Predicates for Robot Planning (https://arxiv.org/abs/2410.23156)
Comments:
          In submission

- **What's New**: 이 논문에서는 Neuro-Symbolic Predicates(NSPs)라는 새로운 1차원 추상화 언어를 소개합니다. 이 언어는 기호(symbolic)와 신경망(neural) 지식 표현의 장점을 결합하여 임무에 특화된 추상화를 형성합니다.

- **Technical Details**: Neuro-Symbolic Predicates는 Python 코드 스니펫으로, 시각-언어 모델(vision-language models, VLMs)을 호출하여 지각적 속성을 쿼리하고 이 속성을 알고리즘적으로 조작할 수 있습니다. 이 접근법은 기존의 심볼릭 세계 모델을 사용한 로봇 작업 계획과는 달리 새로운 환경에 적응할 수 있는 학습 기반의 모델을 제공합니다.

- **Performance Highlights**: 실험 결과, NSPs 접근법은 샘플 복잡성(sample complexity)의 효율성이 더 뛰어나고, 새로운 환경에서의 제너럴리제이션(generalization)이 강하며, 해석 가능성(interpretability)이 개선된 것으로 나타났습니다. 또한, 5개의 로봇 시뮬레이션 환경에서 기존의 기계 학습, 심볼릭, LLM 기반 방식과 비교하여 더 높은 성능을 나타냈습니다.



### When can classical neural networks represent quantum states? (https://arxiv.org/abs/2410.23152)
Comments:
          37 pages, 9 figures

- **What's New**: 이번 연구에서는 n-qubit 상태의 고전적 표현이 지니는 한계와, 양자 상태의 신경 대표성(neural representation)의 효율성을 결정짓는 조건부 상관관계의 역할을 규명했습니다. 특히 측정 분포에 존재하는 이러한 조건부 상관관계는 양자 상태의 인과적 얽힘(measurement-induced entanglement)으로 인해 발생하며, 물질의 상(phase) 연구에서 주로 검토되는 제한된 상관관계를 통해서는 접근할 수 없는 특징을 드러냅니다.

- **Technical Details**: 연구진은 이론적 및 수치적 분석을 결합하여 양자 상태의 얽힘(entanglement) 및 부호 구조(sign structure), 그리고 측정 기준의 선택이 단기 또는 장기 조건부 상관의 독특한 패턴을 만들어내는 방식을 보여주었습니다. 이러한 접근은 순환 신경망(recurrent neural networks, RNN)과 같은 심층 학습 아키텍처를 통해 양자 상태를 효과적으로 표현할 수 있는 가능성을 제시합니다.

- **Performance Highlights**: 연구 결과, 신경 양자 상태(neural quantum states)가 가지는 표현력은 특정 물리적 제약에 의존하며, 이는 곧 칠드닉 엔트로피(area law for entanglement entropy)와 같은 이론적 배경을 통해 효율적인 고전적 표현을 가능하게 합니다. 다른 신경망 구조에 대한 적용 가능성도 논의됨으로써, 향후 양자 시스템의 시뮬레이션 및 학습 분야에 있어 광범위한 응용 가능성을 기대할 수 있습니다.



### The Good, the Bad, and the Ugly: The Role of AI Quality Disclosure in Lie Detection (https://arxiv.org/abs/2410.23143)
Comments:
          Order of the authors are in alphabetical order of their last names. All authors contributed equally. The manuscript is under review. 74 Pages, including appendices and references

- **What's New**: 본 논문은 낮은 품질의 AI 어드바이저가 품질 공시 없이 어떻게 텍스트 기반 거짓말을 퍼뜨릴 수 있는지를 조사합니다. 실험에 참여한 참가자들은 게임 쇼의 전사를 평가하며 진실과 거짓을 구별하는 작업을 수행했으며, 낮은 품질의 어드바이저를 의존할 때 진실 감지 능력이 개인의 능력 아래로 떨어지는 경향을 발견했습니다. 반면, 품질 높은 어드바이저는 공시 여부와 관계없이 진실 감지를 향상시킵니다.

- **Technical Details**: 우리는 AI 어드바이저의 품질을 여러 수준(낮음, 보통, 높음)으로 설정하고, 참가자들은 AI의 효과성이 공개된 환경과 비공개된 환경에서 각각 진실을 감지하도록 실험을 진행했습니다. 본 연구는 AI 어드바이저의 품질 스펙트럼과 그 효과성의 (비)공식 공시가 참가자의 AI 의존도에 어떻게 영향을 미치는지를 조사합니다.

- **Performance Highlights**: 연구 결과, 낮은 품질의 AI 어드바이저에 대한 의존은 참가자의 진실 감지 능력을 저하시켰으며, 이로 인해 거짓 정보의 확산이 우려됩니다. 반면, 높은 품질의 AI 어드바이저는 참가자들이 진실을 감지할 수 있는 능력을 전반적으로 향상시켰습니다.



### Revisiting MAE pre-training for 3D medical image segmentation (https://arxiv.org/abs/2410.23132)
Comments:
          Arxiv Preprint. Currently under Review

- **What's New**: 이 논문에서는 Self-Supervised Learning (SSL) 접근법을 활용하여 44,000개의 3D 뇌 MRI 데이터셋을 기반으로 기존 SSL 방법보다 성능 향상을 이루었습니다. 특히, Residual Encoder U-Net 아키텍처를 사용하여 3D 의료 이미지 분석의 한계를 극복하고자 했습니다.

- **Technical Details**: 제안된 모델은 Masked Auto Encoders (MAEs) 개념을 3D CNN에 최적화하여 구성되었습니다. 실험은 5개의 개발 및 8개의 테스트 뇌 MRI 분할 데이터셋에서 진행되었습니다. 논문에서 다룬 주요 기술적 세부사항으로는 z-score 정규화, poly-learning rate 조정을 통한 SGD 최적화 방법 등이 있습니다.

- **Performance Highlights**: 제안된 모델은 기존 nnU-Net 베이스라인보다 평균 3 Dice 포인트 향상된 성능을 보였으며, 7개의 방법 중 평균 순위 2를 기록하여 뛰어난 안정성을 자랑합니다.



### Provably Optimal Memory Capacity for Modern Hopfield Models: Transformer-Compatible Dense Associative Memories as Spherical Codes (https://arxiv.org/abs/2410.23126)
Comments:
          Accepted at NeurIPS 2024

- **What's New**: 이번 연구에서는 현대 Hopfield 모델과 Kernelized Hopfield Models (KHMs)의 최적 메모리 용량을 분석하고, 정보 이론의 구형 코드와 KHMs의 메모리 구성 간의 연결을 수립합니다. 이를 통해 메모리 문제를 초구 위의 점 배열 문제로 변환하는 새로운 관점을 제공합니다.

- **Technical Details**: KHMs에서는 메모리를 최적의 구형 코드로 형성할 수 있는 특징 공간이 필요하며, 이를 통해 KHMs의 최적 메모리 용량을 분석합니다. 우리는 KHMs의 메모리 용량에 대한 상한선 및 하한선을 수립하고, 메모리 개수에 비례하는 피처 차원의 크기 변화에 대한 분석을 수행합니다.

- **Performance Highlights**: 실험적으로 제공된 수치 결과는 KHMs의 조회 능력 향상 및 관련 transformer의 표현 학습 개선을 뒷받침합니다. 또한, KHMs의 최적 용량에 도달하기 위한 서브-선형 시간 알고리즘 $	exttt{U-Hop}^+$을 제안합니다.



### Unified Triplet-Level Hallucination Evaluation for Large Vision-Language Models (https://arxiv.org/abs/2410.23114)
Comments:
          18 pages, 8 figures

- **What's New**: 본 논문은 대형 비전-언어 모델(LVLM)에서 발생하는 객체 및 관계 환각(hallucination)을 동시에 평가하기 위한 통합 프레임워크를 설계하였습니다.

- **Technical Details**: LVLM의 응답에서 추출된 (객체, 관계, 객체) 트리플렛(triplet) 기반의 환각 평가를 통해 환각 유형을 통합적으로 분석할 수 있으며, Tri-HE라는 새로운 평가 벤치마크를 도입하여 LVLM의 환각 문제를 보다 세밀하게 평가할 수 있습니다.

- **Performance Highlights**: 이 연구에서 제안한 방법은 기존 LVLM이 갖고 있는 관계 환각 문제를 해결함으로써, LLaVA-1.5 모델이 모든 오픈 소스 모델을 초월하는 성능을 기록하였으며, 강력한 GPT-4V와 동등한 성능을 보여주었습니다.



### Decoupling Semantic Similarity from Spatial Alignment for Neural Networks (https://arxiv.org/abs/2410.23107)
Comments:
          Accepted at NeurIPS2024

- **What's New**: 본 논문에서는 Representational Similarity Matrices (RSMs)의 기존 계산 방법의 한계를 제시하고, 공간적 위치에 영향을 받지 않는 semantic RSMs를 제안하여 이미지 응답 간의 유사성을 측정합니다.

- **Technical Details**: 우리는 semantic RSMs를 제안하여 공간적 순열에 불변이며, 집합 매칭 문제로 형성된 semantic 유사성을 측정합니다. 이 방법은 CNN과 ViT의 이미지 응답을 비교하여, 두 모델의 유사성 구조를 파악합니다.

- **Performance Highlights**: 제안한 semantic RSMs는 spatio-semantic RSMs에 비해 이미지 검색 성능을 향상시키고, 분류기 표현 간의 유사성을 더 잘 반영합니다. 또한 컴퓨팅 복잡성을 줄이기 위한 근사화 방법을 소개합니다.



### Guided Game Level Repair via Explainable AI (https://arxiv.org/abs/2410.23101)
- **What's New**: 이번 연구는 기계 학습을 통한 프로시저 생성 수준(Procedural Content Generation, PCG)에서 생성된 레벨이 해결 불가능할 수 있음을 다루고 있습니다. 연구진은 레벨 수리가 필요한 경우에 대해 설명 가능성(explainability) 방법을 활용하여, 특정 영역이 문제를 일으키는지를 파악하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 연구진은 2D 타일 기반의 여러 게임(Super Mario Bros., Super Cat Tales, 커스텀 Cave 게임)에서 해결 가능성과 해결 불가능한 레벨을 구별하는 이진 분류기를 훈련합니다. 이 분류기의 설명 가능성 방법을 활용하여 각 타일의 중요도를 평가하고, 이를 기반으로 제약 만족 문제(constraint satisfaction problem)를 해결하는데 필요한 가중치를 제공합니다.

- **Performance Highlights**: 본 연구의 실험 결과, 제안된 접근 방식이 레벨 수리를 더 빠르게 수행할 수 있도록 도와주는 것을 발견했습니다. 이 방법은 자동으로 생성된 레벨의 문제를 파악하고, 보다 효율적으로 수리를 진행할 수 있도록 합니다.



### Comparative Analysis of Demonstration Selection Algorithms for LLM In-Context Learning (https://arxiv.org/abs/2410.23099)
Comments:
          6 pages, 4 figures

- **What's New**: 이 논문은 다양한 demonstration selection 알고리즘을 분석하고, 실험을 통해 이들 알고리즘의 효율성과 효과를 평가합니다. 특히, 무작위 선택이 특정 상황에서 오히려 나은 결과를 초래할 수 있음을 보여줍니다.

- **Technical Details**: 연구에서는 여섯 가지 demonstration selection 알고리즘(CBDS, RD-direct, RD-channel, LLM Retriever, UPRISE, OpenICL TopK)을 비교하였습니다. 각 알고리즘은 특정 전략을 사용하여 LLM의 성능을 개선하는 데 중점을 두었으며, 이 과정에서 Bayesian 접근법, 순차적 예제 검색, 교차 인과 모델 등을 활용했습니다.

- **Performance Highlights**: 경험적인 연구 결과, 알고리즘 간 성능 차이가 크며, 같은 데이터셋에서도 정확도 차이가 45%까지 발생할 수 있습니다. 또한, 시연의 수 증가가 항상 향상된 성능으로 이어지지 않으며, 정확성과 처리 효율성 간에는 트레이드오프가 존재한다는 점을 발견했습니다.



### Statistical-Computational Trade-offs for Density Estimation (https://arxiv.org/abs/2410.23087)
Comments:
          To appear at NeurIPS 2024

- **What's New**: 본 논문은 샘플링 복잡도(sampling complexity)와 쿼리 시간(query time) 모두에서 서브 선형(sublinear) 경계를 달성하는 데 있어 새로운 하한을 제시합니다. 이는 데이터 구조(data structure)의 통계적-계산적(statistical-computational) 트레이드오프(trade-off)로, 샘플의 수 또는 쿼리 시간이 거의 선형(linear)으로 제한될 수밖에 없음을 보여줍니다.

- **Technical Details**: 주어진 k개의 분포 $p_1, ..., p_k$와 쿼리 분포 $q$를 기반으로, 빠른 쿼리 시간으로 '가까운' 분포 $p_i$를 출력하는 데이터 구조를 구축하는 문제를 다룹니다. 만약 일부 상수 $c>0$에 대해 $O(n/	ext{log}^c k)$ 샘플을 사용하는 알고리즘이라면, 쿼리 시간은 $k^{1-O(1)/	ext{log log} k}$ 이상이 되어야 합니다.

- **Performance Highlights**: 실험 결과, 제안된 데이터 구조는 실제 어플리케이션에서 효율적임을 보여주었으며, 동시 샘플 수와 쿼리 시간의 관계에서 상당한 개선을 이루었습니다.



### CNN Explainability with Multivector Tucker Saliency Maps for Self-Supervised Models (https://arxiv.org/abs/2410.23072)
Comments:
          29 pages, 20 figures

- **What's New**: 이번 연구에서는 Convolutional Neural Networks (CNNs)의 해석 가능성에 대한 새로운 접근법인 Tucker Saliency Map (TSM) 방법을 소개합니다. 이는 기존의 EigenCAM과 달리, feature maps의 고유한 구조를 더 잘 포착하여 더 정확한 saliency maps를 생성합니다.

- **Technical Details**: Tucker tensor decomposition을 적용하여 singular vectors와 values를 생성하며, 이를 통해 고화질의 saliency maps를 생성합니다. 또한, EigenCAM과 TSM을 확장한 Multivec-EigenCAM과 Multivector Tucker Saliency Maps (MTSM)를 도입하여 모든 singular vectors와 values를 활용합니다.

- **Performance Highlights**: 정량적 평가 결과, TSM, Multivec-EigenCAM, 및 MTSM은 label-dependent 방법들과 경쟁력 있는 성능을 보였으며, TSM은 EigenCAM에 비해 explainability를 약 50% 향상시켰고, MTSM은 self-supervised 모델에서 최고의 결과를 달성했습니다.



### Don't Just Pay Attention, PLANT It: Transfer L2R Models to Fine-tune Attention in Extreme Multi-Label Text Classification (https://arxiv.org/abs/2410.23066)
- **What's New**: 새로운 PLANT(Pretrained and Leveraged AtteNTion) 모델은 Extreme Multi-Label Text Classification (XMTC)에서 상태-of-the-art (SOTA) 성능을 달성하는 혁신적인 전이 학습 전략입니다. 기존 모델의 한계를 극복하고 특히 희귀 코드 처리에서 탁월한 성능을 보입니다.

- **Technical Details**: PLANT는 미리 훈련된 Learning-to-Rank (L2R) 모델을 주의(attention) 레이어로 활용하는 등의 기술 혁신을 포함합니다. 또한 상호 정보 이득(mutual-information gain)을 통합하여 주의를 강화하고, 주의가 필요 없는(inattention) 메커니즘과 상태 유지 디코더(stateful-decoder)를 구현하여 문맥을 유지합니다.

- **Performance Highlights**: PLANT는 약 50% 포인트 이상의 F1 점수 향상으로 MIMIC-III의 rare 및 few 데이터 세트에서 이전 few-shot 모델을 능가하며, 전통적인 모델 대비 적은 데이터로도 상당한 정확도를 달성합니다. 또한, EURLEX-4K와 WIKI10-31K 데이터 세트에서도 SOTA 성능을 기록했습니다.



### VisAidMath: Benchmarking Visual-Aided Mathematical Reasoning (https://arxiv.org/abs/2410.22995)
Comments:
          58 pages, 28 figures

- **What's New**: 새로운 연구에서는 시각적 정보를 활용한 수학 문제 해결(Visual Aided Mathematical Problem Solving, MPS) 과정을 평가하기 위한 VisAidMath 벤치마크를 소개합니다. 이 벤치마크는 1,200개의 난이도 있는 문제를 포함하며, 다양한 출처에서 수집된 문제와 답변을 평가합니다.

- **Technical Details**: VisAidMath 벤치마크는 수학적 질문을 시각적 맥락(Visual Context), 질문(Question), 시각적 도움(Visual Aids), 답변(Answer) 네 부분으로 나누어 설계되었습니다. 이 벤치마크는 명확한 시각적 정보를 포함하고 있으며, 값은 LaTeX 형식으로 정리되어 있습니다. 이 연구에서는 통계적으로 10개의 주요 LLMs 및 LMMs의 성능을 분석하였습니다.

- **Performance Highlights**: 주요 모델인 GPT-4V는 시각적 정보를 활용한 추론 과제에서 평균 45.33%의 정확도를 보였으며, 이는 전문적인 시각적 도움을 제공받았을 때도 2점 감소하였습니다. 또한, SOTA 모델들은 약 50%의 평균 정확도에 그쳤으며, 생성된 시각적 보조 자료는 5%의 n-gram 유사성만을 보였습니다.



### Dynamic Matching with Post-allocation Service and its Application to Refugee Resettlemen (https://arxiv.org/abs/2410.22992)
Comments:
          Preliminary conference version appeared in ACM Economics and Computation (EC 2024)

- **What's New**: 본 논문은 미국의 주요 난민 정착 기관과의 협업을 통해 동적인 매칭 문제를 연구하며, 새로운 난민 케이스가 정해진 장소에 즉시 매칭되어야 하는 문제를 다룹니다.

- **Technical Details**: 논문에서는 서비스 제공이 시간이 많이 소요되는 특성을 고려하여 단기적으로 가용성을 고려한 서버(동적 자원)를 사용하며, 매칭 후 케이스는 선착순으로 서비스를 받습니다. 압축된 매칭은 서버의 혼잡을 초래할 수 있어, 매칭 보상과 혼잡 비용을 조합한 최적화 문제를 다룹니다. 또한, 과거 데이터를 기반으로 하지 않는 알고리즘을 설계하였으며, 최적화 문제의 이중 변수를 학습하는 방식으로 설계되었습니다.

- **Performance Highlights**: 실제 데이터를 기반으로 파트너 기관에서 테스트한 결과, 제안된 방법이 기존 방법보다 뛰어난 성능을 보여 현재의 실무를 대체할 수 있는 유망한 후보로 부각되었습니다.



### V2X-Assisted Distributed Computing and Control Framework for Connected and Automated Vehicles under Ramp Merging Scenario (https://arxiv.org/abs/2410.22987)
Comments:
          This paper has been submitted to IEEE Journal. The source code has been released at: this https URL

- **What's New**: 본 논문에서는 V2X (Vehicle-to-Everything) 통신을 통한 분산 컴퓨팅 및 협력 제어 기법을 제안하여, 연결 및 자율 주행 차량(CAVs)의 램프 병합 시나리오에서 중앙 집중식 제어 방식의 단점을 극복하고자 합니다.

- **Technical Details**: 먼저, 안전 제약 및 교통 성능에 따른 중앙 집중식 협력 궤적 계획 문제를 제기하고, 이를 통해 모든 차량의 궤적을 공동으로 최적화합니다. 이후, 고차원 및 비볼록 문제로 변모한 다중 차량 모델 예측 제어(MPC) 문제를 해결하기 위해 분할 및 볼록 재구성을 통한 DCIMPC(Distributed Cooperative Iterative Model Predictive Control) 방법을 제안합니다. 이러한 방식을 통해 CAV 간의 통신을 활용하여 계산 자원을 분산하여 신속하게 해결할 수 있습니다.

- **Performance Highlights**: 시뮬레이션 결과, 제안된 프레임워크는 수렴성을 보여주며, 안전성과 해결 속도가 개선되었습니다. DCIMPC 방법은 계산 속도를 크게 향상시키면서도 시스템 성능을 희생하지 않는다는 결과를 보였습니다.



### Graph Integration for Diffusion-Based Manifold Alignmen (https://arxiv.org/abs/2410.22978)
Comments:
          8 pages, 4 figures, Accepted at ICMLA 2024

- **What's New**: 이 논문에서는 반지도 학습(semi-supervised) 매니폴드 정렬(manifold alignment)을 위한 두 가지 새로운 방법을 소개합니다. 각각의 방법은 서로 다른 매니폴드 학습 모델에서 영감을 받았습니다.

- **Technical Details**: 첫 번째 방법인 SPUD(Shortest Paths on the Union of Domains)는 알려진 대응관계를 사용하여 통합 그래프 구조를 형성합니다. 두 번째 방법인 MASH(Manifold Alignment via Stochastic Hopping)는 각 도메인 내에서 지역 기하학(local geometry)을 학습하고, 알려진 대응관계를 활용하여 공동 확산 연산자(joint diffusion operator)를 생성합니다.

- **Performance Highlights**: SPUD와 MASH는 기존의 반지도 매니폴드 정렬 방법들과 비교하여 진정한 대응관계를 정렬하고 교차 도메인 분류(cross-domain classification)에서 더 우수한 성능을 보였습니다. 또한, 이러한 방법은 레이블 정보(label information)를 도메인 간 전환하는 데 효과적입니다.



### Scalable Sampling for High Utility Patterns (https://arxiv.org/abs/2410.22964)
Comments:
          Accepted at 2024 IEEE International Conference on Big Data

- **What's New**: 이 논문에서는 대규모 정량 데이터베이스에서 고유한 패턴을 발견하기 위한 새로운 고유 유틸리티 패턴 샘플링 알고리즘인 QPlus를 제안합니다. 또한 이 알고리즘은 온디스크 버전인 QPlusDisk로도 제공되어 메모리 제약을 극복할 수 있도록 설계되었습니다.

- **Technical Details**: QPlus 알고리즘은 고유 유틸리티 패턴(HUP)과 평균 유틸리티 패턴(HAUP)의 샘플링을 위해 고유의 두 가지 정리에 기반을 두고 있습니다. 이 알고리즘은 각 패턴을 유틸리티에 비례하여 확률적으로 샘플링하여 정확하고 대표적인 패턴 추출을 가능하게 합니다. 특히, QPlusDisk는 메모리에 맞지 않는 대규모 데이터베이스에서 유용합니다. 또한, Upper Triangle Utility (UTU) 개념을 도입하여 대량의 트랜잭션에서 유틸리티의 총합을 효율적으로 계산합니다.

- **Performance Highlights**: 실험 결과, QPlus 알고리즘은 기존의 방법들과 비교하여 수백 개의 HUP/HAUP를 초당 반환할 수 있는 효율성을 보여주었습니다. 특히, 고대 유물 지식 그래프의 서브 프로필 발견과 같은 실용적 사례에서도 우수한 성능을 입증하였습니다.



### A Study of Secure Algorithms for Vertical Federated Learning: Take Secure Logistic Regression as an Examp (https://arxiv.org/abs/2410.22960)
Comments:
          accepted by the 20th International Conference on Security & Management (SAM 2021)

- **What's New**: 이 논문은 머신 러닝과 관련된 데이터 보호 문제를 다루고 있으며, 비밀 유지가 중요한 Vertical Federated Learning (수직 연합 학습) 스키마를 통해 보안 모델 학습을 위한 새로운 접근 방식을 제안합니다.

- **Technical Details**: 논문에서는 헬로모픽 암호화(homomorphic encryption)를 활용하여 두 개의 데이터 소스(Alice와 Bob) 간의 데이터는 비공개로 유지하면서 연관된 회귀 모델을 학습하는 방법을 설명합니다. 이 과정에서 데이터는 암호화된 도메인에서 처리되며, 두 당사자는 서로의 원본 데이터에 접근하지 못합니다.

- **Performance Highlights**: 제안된 방법은 데이터 프라이버시 문제를 해결하면서도 모델 성능을 개선할 수 있는 잠재력을 가지고 있으며, 실제 데이터 교환이 법적으로 제한된 환경에서도 유용하게 사용될 수 있을 것으로 기대됩니다.



### KALAM: toolKit for Automating high-Level synthesis of Analog computing systeMs (https://arxiv.org/abs/2410.22946)
Comments:
          5 Pages, 4 figures

- **What's New**: 이 논문에서는 KALAM이라는 새로운 자동화 도구를 소개합니다. KALAM은 Margin Propagation (MP) 기반 아날로그 컴퓨팅 시스템을 합성하는 데 필요한 자동화 프레임워크를 제공합니다.

- **Technical Details**: KALAM은 입력 factor graph를 사용하여 SPICE 호환 회로 넷리스트로 변환합니다. 이를 통해 Bayesian inference, Low-Density Parity Check (LDPC) 디코딩, 인공신경망(Artificial Neural Networks, ANN) 등의 다양한 아날로그 신호 프로세서 설계를 지원합니다.

- **Performance Highlights**: KALAM으로 생성된 회로는 소프트웨어 구현과 밀접하게 일치하는 성능을 보이며, Bit 및 Frame Error Rate(오류율) 플롯도 소프트웨어 결과와 강한 상관관계를 보여줍니다.



### DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data (https://arxiv.org/abs/2410.22938)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: DiffLight는 교통 신호 제어(Traffic Signal Control, TSC)에서 데이터가 누락된 시나리오를 처리하기 위해 소개된 새로운 조건부 확산 모델입니다. 이 모델은 데이터가 누락된 실제 시나리오의 요구를 충족하기 위해 설계되었습니다.

- **Technical Details**: DiffLight는 Partial Rewards Conditioned Diffusion (PRCD) 모델을 통합하여 교통 데이터 보완과 의사 결정 과제를 모두 처리합니다. 또한, Spatial-Temporal transFormer (STFormer) 아키텍처를 통해 교차로 간의 공간-시간 의존성을 효과적으로 포착합니다. Diffusion Communication Mechanism (DCM)을 통해 데이터가 누락된 상황에서도 더 나은 통신 및 제어 성능을 증진시킵니다.

- **Performance Highlights**: 다양한 데이터 누락 시나리오에 대해 5개의 데이터셋에서 실시된 실험 결과, DiffLight는 누락된 데이터가 있는 TSC 문제에 대해 매우 효과적인 성능을 보였습니다.



### An Individual Identity-Driven Framework for Animal Re-Identification (https://arxiv.org/abs/2410.22927)
Comments:
          10 pages

- **What's New**: 본 논문에서는 동물 재식별 (Animal ReID)을 위한 새로운 접근 방식인 IndivAID 프레임워크를 제안합니다. 이 프레임워크는 Vision-Language 모델인 CLIP의 크로스 모달 특성을 활용하여 동물의 개별 텍스트 설명을 생성하는 두 단계의 프로세스로 구성됩니다.

- **Technical Details**: IndivAID는 두 단계로 구성됩니다: 첫 번째 단계에서는 이미지에서 개별적인 의미 정보를 추출하여 이미지 및 개체에 대한 설명을 생성하는 텍스트 설명 생성기를 학습합니다. 두 번째 단계에서는 이 텍스트 설명을 동적으로 통합하여 개별적인 피처를 강조하는 주의 모듈을 통해 시각적 개념 학습을 개선합니다.

- **Performance Highlights**: 여덟 개의 벤치마크 데이터셋과 실제 Stoat 데이터셋에 대한 평가 결과, IndivAID는 최첨단 방법들과 비교하여 효과성과 적용가능성을 보여주었습니다. 특히 다양한 특성을 가진 동물의 재식별에서 우수한 성능을 발휘했습니다.



### Self-optimization in distributed manufacturing systems using Modular State-based Stackelberg Games (https://arxiv.org/abs/2410.22912)
Comments:
          This pre-print was submitted to Journal of Manufacturing Systems on October 30, 2024

- **What's New**: 본 연구에서는 분산형 자가 학습 모듈식 제조 시스템을 위한 새로운 게임 구조인 Modular State-based Stackelberg Games (Mod-SbSG)를 소개합니다. Mod-SbSG는 State-based Potential Games (SbPG)와 Stackelberg 게임을 통합하여, 생산 시스템 내 자가 학습 에이전트 간의 협력적 의사 결정을 향상시킵니다.

- **Technical Details**: Mod-SbSG는 중요도가 높은 모듈이 선도자 역할을 수행하고, 덜 중요한 모듈이 이에 최적응답하는 계층적인 게임 구조를 제공합니다. 이는 전통적인 다중 에이전트 학습 알고리즘과 달리 동시에 결정하는 방식이 아닌 계층적 의사결정 과정을 도입합니다. 이 게임 구조는 강화학습(Deep Reinforcement Learning) 및 경량화된 학습 알고리즘에 통합되어 자가 최적화 알고리즘과 호환됩니다.

- **Performance Highlights**: Mod-SbSG는 실험적으로 두 가지 실험 환경에서 테스트되어, 기존 SbPG와 비교하여 시스템 오버플로우를 97.1% 감소시켰고, 특정 경우에는 오버플로우를 완전히 방지했습니다. 또한, 생산 수요를 충족시키면서도 에너지 소비를 5-13% 줄이는 성과를 거두어 잠재적 글로벌 목표 값을 상당히 향상시켰습니다.



### Generalization Bounds via Conditional $f$-Information (https://arxiv.org/abs/2410.22887)
Comments:
          Accepted at NeurIPS 2024

- **What's New**: 이 연구에서는 전통적인 조건부 상호 정보(mutual information, MI) 프레임워크를 확장한 새로운 정보 이론적 일반화 경계를 도입합니다. 특히, 조건부 f-정보(framework) 구조를 기반으로 하여, 바운드된 손실 함수(bounded loss function)와 비바운드 손실 함수(unbounded loss function) 모두에 적용할 수 있는 일반화 경계를 제시합니다.

- **Technical Details**: 본 연구에서 제안한 방법은 함수의 볼록 공액(convex conjugate)에 기반한 f-발산(f-divergence) 변형을 통해 일반화 경계를 도출합니다. 기존의 조건부 상호 정보 프레임워크(CMI)의 접근법과는 다르게, 우리는 Cumulant Generating Function(CGF)를 사용하여 경계를 설정하는 것이 아니라, 선택된 측정 가능한 함수의 볼록 공액의 역을 선택하여 CGF를 제로(0)로 설정하고 변형 공식에서 제거합니다.

- **Performance Highlights**: 새로이 도출된 경계는 이전 결과를 회복하고, 여러 f-정보 측정치의 성능을 비교한 결과, 특히 제곱 헬링거 정보(squared Hellinger-information) 경계가 이전의 결과들보다 뛰어난 성능을 보였음을 입증했습니다. 실험 결과는 우리 새로운 경계의 유용성을 명확히 보여줍니다.



### Stealing User Prompts from Mixture of Experts (https://arxiv.org/abs/2410.22884)
- **What's New**: 이 논문은 Mixture-of-Experts (MoE) 모델에서 사용자 쿼리를 추출하는 새로운 공격 방법을 제시합니다. 이 공격 방법은 피해자의 쿼리와 같은 배치에 있는 쿼리를 조작하여 피해자의 프롬프트를 완전히 공개할 수 있습니다.

- **Technical Details**: 연구에서 소개된 공격은 두 계층의 Mixtral 모델에서 실행되며, 공격자는 $O({VM}^2)$ 쿼리(어휘 크기 $V$ 및 프롬프트 길이 $M$에 따라) 또는 평균적으로 각 토큰당 100개의 쿼리로 프롬프트를 추출할 수 있습니다. 이것은 공격자가 Expert-Choice-Routing을 악용하여 구현된 CUDA의 tie-handling 동작을 활용하는 것입니다.

- **Performance Highlights**: 이 연구는 사용자 프롬프트를 추출하기 위해 구조적 결함을 exploit 하는 공격의 첫 번째 사례로, 새로운 LLM(대형 언어 모델) 취약점의 클래스를 소개합니다.



### Towards Population Scale Testis Volume Segmentation in DIXON MRI (https://arxiv.org/abs/2410.22866)
- **What's New**: 이번 논문은 UK Biobank의 MRI 데이터를 사용하여 고환 부피를 세그멘테이션(화면 분할)하는 방법을 평가합니다. 이 연구는 고환의 크기를 평가하는 데 기계 학습을 활용하여 대규모 인구 데이터에서의 적용 가능성을 탐구하고 있습니다.

- **Technical Details**: Deep learning 기술을 사용하여 MRI 데이터에서 고환의 세그멘테이션을 수행하는 세 가지 주요 기여가 있습니다. 첫째, UK Biobank 데이터셋에 대한 새로운 주석된 데이터셋을 제공하고, 둘째, 인간 전문가의 주석 품질을 검토하며, 셋째, 최종 모델 및 관련 가중치를 공개하여 기준선을 제공합니다. 연구에서는 DeepLabV3, DeepLabV3Plus, UNet 아키텍처와 같이 여러 컨볼루션 기반 아키텍처와 그것의 변형인 3D UNet 아키텍처를 비교하였습니다.

- **Performance Highlights**: 모델의 중위 Dice 점수는 0.87로, 동일 데이터셋에서 인간 판독자 간 신뢰도 중위 Dice 점수인 0.83보다 우수했습니다. 이로 인해 대규모 고환 MRI 세그멘테이션 연구에서 접근성과 재현성을 높이는 데 기여할 것입니다.



### AtGCN: A Graph Convolutional Network For Ataxic Gait Detection (https://arxiv.org/abs/2410.22862)
- **What's New**: 본 논문에서는 AtGCN이라는 새로운 그래프 컨볼루션 네트워크를 제안하여 2D 비디오를 사용하여 운동실조(atacia) 보행을 감지하고 그 중증도를 평가합니다. 특히 운동실조 보행의 정상 보행과의 미세한 차이를 효과적으로 포착할 수 있도록 설계된 특별한 시공간(spatiotemporal) 그래프 컨볼루션을 활용하고 있습니다.

- **Technical Details**: AtGCN 모델은 사전 학습된 행동 인식(action recognition) 데이터셋 위에서 체계적으로 조정하고 미세 조정하여 작은 데이터셋에서도 학습할 수 있도록 설계되었습니다. 이 시스템은 보행 주기를 여러 개로 분할한 후, 각 주기를 그래프 구조로 변환하여 모델의 입력으로 사용합니다. YOLO 감지기를 통해 비디오에서 인물의 바운딩 박스를 생성하며, DeepSORT를 통해 프레임 간 인물 추적을 수행합니다.

- **Performance Highlights**: 제안된 AtGCN 모델은 운동실조 보행 감지 및 중증도 예측에서 93.46%의 정확도 및 0.4169의 MAE(Mean Absolute Error)를 기록하며, 최신 기술들(state-of-the-art)을 능가하는 성능을 보여줍니다.



### Hyperparameter Optimization in Machine Learning (https://arxiv.org/abs/2410.22854)
Comments:
          Preprint

- **What's New**: 본 논문에서는 하이퍼파라미터 최적화(hyperparameter optimization)에 대한 포괄적인 개요를 제공하고, 이를 통해 머신러닝 연구 및 산업에서 자동화된 하이퍼파라미터 검색 방법의 체계적 사용을 촉진하고자 한다.

- **Technical Details**: 하이퍼파라미터(hyperparameters)는 머신러닝 알고리즘의 동작을 제어하는 구성 변수로, 하이퍼파라미터의 값 선택은 이러한 기술을 기반으로 한 시스템의 효율성을 결정한다. 본 논문은 랜덤(randiom) 및 준랜덤(quasi-random) 검색, 밴딧(bandit) 접근법, 모델 기반(model-based) 접근법, 그리고 그래디언트 기반(gradient-based) 접근법을 포함한 여러 하이퍼파라미터 검색 기술을 논의한다.

- **Performance Highlights**: 상태-최고(state-of-the-art) 성과를 달성하기 위해 하이퍼파라미터 조정이 얼마나 중요한지를 시연하기 위해 감정 분석(sentiment analysis)을 예로 들며, 연구에서의 하이퍼파라미터 활용의 중요성을 강조한다. 실험 결과에 따르면, 적절히 조정된 로지스틱 회귀(logistic regression)는 당시의 CNN(convolutional neural networks)과 동등한 성능을 보였음을 나타낸다.



### Dataset of polarimetric images of mechanically generated water surface waves coupled with surface elevation records by wave gauges linear array (https://arxiv.org/abs/2410.22849)
Comments:
          15 pages, 7 figures, 3 tables. Data article. Under review in "Data in Brief" journal. The data in available for download on ScienceDB repository. arXiv admin note: substantial text overlap with arXiv:2410.14988

- **What's New**: 이 논문에서는 수리 실험에서 물 표면 고도 측정의 정확도를 향상시키기 위해, 극성 필터(polarization filter)를 장착한 카메라와 머신 러닝 알고리즘을 활용한 새로운 방법을 개발했습니다.

- **Technical Details**: 이 방법은 감독 학습(supervised learning)을 기반으로 한 독자적으로 제작된 데이터 세트를 사용하였으며, 물 표면의 폴라리메트릭 이미지와 저항형 파계의 선형 배열을 이용한 측정값을 결합했습니다. 실험은 인위적인 조명 하에 진행되었습니다.

- **Performance Highlights**: 개발된 데이터 세트는 다양한 경사도를 가진 단일색 파열부터 JONSWAP 스펙트럼 형태의 불규칙한 파장 조건 및 여러 파괴 시나리오에 이르기까지 다양한 파장 장면을 포함합니다. 여러 카메라 위치에서 반복 측정이 이루어져, 높은 시공간 해상도(spatio-temporal resolution)를 지원합니다.



### Danoliteracy of Generative, Large Language Models (https://arxiv.org/abs/2410.22839)
Comments:
          16 pages, 13 figures, submitted to: NoDaLiDa/Baltic-HLT 2025

- **What's New**: 이 연구는 덴마크어에 대한 Generative, Large Language Models (GLLMs)의 성능을 평가하기 위한 Danoliterate Benchmark를 소개합니다. 이 벤치마크는 덴마크어 및 문화적 역량을 측정하며, 다양한 시나리오에서 모델의 능력을 검증하는 새로운 데이터 세트를 제공합니다.

- **Technical Details**: 이 연구는 RWK (Real-World Knowledge), NLU (Natural Language Understanding), NLG (Natural Language Generation) 등 세 가지 주요 범주에서 덴마크어 GLLM의 성능을 평가합니다. 605개의 다지선다형 질문으로 구성된 Citizenship Test와 같은 다양한 시나리오를 포함하여 모델 성능을 비교 분석합니다. 벤치마크는 GPT-4와 Claude Opus 모델이 최고 성능을 기록한 사실을 입증합니다.

- **Performance Highlights**: 본 연구의 결과, GLLMs의 성능은 약 80% 정도의 인간 피드백과 상관관계를 나타내며, 하나의 주요 요소가 GLLMs의 협응력(ability consistency)을 설명하는 것으로 나타났습니다. 이는 Danoliteracy 평가에서 고른 성능 변화를 제공할 수 있음을 시사합니다.



### Run-Time Adaptation of Neural Beamforming for Robust Speech Dereverberation and Denoising (https://arxiv.org/abs/2410.22805)
Comments:
          Accepted to APSIPA2024

- **What's New**: 본 논문에서는 실제 환경에서 실시간 자동 음성 인식(ASR)을 위한 음성 향상 기술을 설명합니다. 구체적으로, DNN(Deep Neural Network)을 기반으로 한 신경 빔형성(neural beamforming)을 통한 마스크 추정 및 음성 필터링 과정을 소개하고, 비슷한 환경의 음성 데이터를 활용한 실시간 적응 방식에 대한 연구를 제시합니다.

- **Technical Details**: 논문은 DNN을 사용한 신경 빔형성에서의 음향 특성 변화를 다루고 있으며, 이를 위해 WPE(Weighted Prediction Error)와 MPDR(Minimum Power Distortionless Response)를 통합한 WPD(Weighted Power Minimization Distortionless Response) 빔형성을 제안합니다. 이로 인해 각기 다른 조건에서의 음성 신호의 억제를 위한 필터 계산이 가능해집니다.

- **Performance Highlights**: 다양한 화자 수, 잔향 시간, 신호 대 잡음 비율(SNR) 등 구성된 조건에서 실시간 적응의 효과를 평가하며 ASR 성능의 향상을 입증합니다. 이 시스템은 음성 업무 성능 개선을 위한 효율적인 변화를 이끌어 내는 데 성공적임을 보여줍니다.



### DOA-Aware Audio-Visual Self-Supervised Learning for Sound Event Localization and Detection (https://arxiv.org/abs/2410.22803)
Comments:
          Accepted to APSIPA2023

- **What's New**: 이 논문은 첫 번째 순서 암비소닉스(First-order Ambisonics, FOA) 마이크로폰으로 캡처한 공간 오디오 녹음에 대한 소리 이벤트 로컬리제이션 및 탐지(Sound Event Localization and Detection, SELD)를 다루고 있습니다. 특히, 지도 학습 방식으로 DNN을 훈련할 때 주석이 달린 데이터가 부족한 문제를 자기 지도 방식(Self-supervised Learning)으로 해결하는 새로운 방법을 제안하고 있습니다.

- **Technical Details**: 연구에서는 FOA와 360도 전 방향 카메라로 촬영된 VR 콘텐츠를 활용해 오디오와 비주얼 인코더를 대조 학습(Contrastive Learning) 방식으로 공동 훈련합니다. 핵심 기능은 DOA별 음향 임베딩이 원시 오디오 데이터에서 공동으로 추출되며, DOA-wise 비주얼 임베딩은 해당 DOA를 중심으로 한 지역 비주얼 크롭에서 별도로 추출된다는 점입니다. 이를 통해 오디오 인코더에 설비된 잠재적 특성이 오디오 이벤트의 클래스와 DOA를 나타내도록 유도합니다.

- **Performance Highlights**: DCASE2022 Task 3 데이터셋을 활용한 실험에서 비주석된 100시간의 오디오-비주얼 녹음이 SELD의 오류 점수를 36.4점에서 34.9점으로 감소시키는 성과를 보여주었습니다.



### Machine Learning Nonadiabatic Dynamics: Eliminating Phase Freedom of Nonadiabatic Couplings with the State-Intraction State-Averaged Spin-Restricted Ensemble-Referenced Kohn-Sham Approach (https://arxiv.org/abs/2410.22801)
- **What's New**: 이번 연구에서는 conical intersections (CIs) 근처의 excited-state molecular dynamics (ESMD) 시뮬레이션에서의 머신러닝 포텐셜(MLPs) 적용 시 발생하는 도전과제를 다룹니다. 새로운 phaseless coupling term인 \(\Delta^2 \)를 도입하여 CI 특성과 double-valued coupling 요소로 인해 발생하는 불연속성을 해결하는 방법을 제시합니다.

- **Technical Details**: 제안된 방법은 SSR(2,2) 공식에 기반하여 diabatic Hamiltonian의 off-diagonal 요소의 제곱을 활용합니다. 이 방법은 계층별 메시지 전파(equivariant message-passing) MLP를 훈련시켜 penta-2,4-dieniminium cation (PSB3)에서 작동합니다. 제안된 phaseless coupling은 coupling 요소의 위상 모호성을 제거하고 NACV 예측의 정확성을 극대화합니다.

- **Performance Highlights**: \(\Delta^2 \) 기반의 ML-ESMD 방법은 ab initio ESMD 시뮬레이션을 재현할 수 있는 잠재력과 효율성을 자랑하며, 특히 대규모 및 장기간의 ESMD 시뮬레이션에서의 적용 가능성을 강조합니다.



### MALoRA: Mixture of Asymmetric Low-Rank Adaptation for Enhanced Multi-Task Learning (https://arxiv.org/abs/2410.22782)
Comments:
          14 pages, 5 figures

- **What's New**: 본 논문에서는 Mixture of Asymmetric Low-Rank Adaptation (MALoRA)라는 새로운 유연한 파인 튜닝 프레임워크를 제안합니다. MALoRA는 LoRA 전문가들 간의 비대칭 최적화를 활용하여 파라미터의 수를 30%에서 48% 줄이고, 훈련 속도를 1.2배 증가시키며, 단일 작업 LoRA 모델과의 계산 효율성을 일치시킵니다.

- **Technical Details**: MALoRA는 공유 가능한 저랭크 서브스페이스(representing low-rank subspace)를 도입하여 내려가는 프로젝션 모듈(down-projection module)에서 파라미터 중복을 줄이고, 각 LoRA 전문가에게 긴축된 계수 행렬(compacted coefficient matrix)을 배정하여 효과적으로 파라미터 수와 계산 복잡성을 낮춥니다.

- **Performance Highlights**: MALoRA는 다양한 다중 작업 학습 시나리오에서 진행된 실험을 통해 MoLoRA의 성능을 뛰어넘는 결과를 지속적으로 보여주었으며, 효율성과 일반화에서 모든 기준 방법들을 초월합니다.



### Unfolding Target Detection with State Space Mod (https://arxiv.org/abs/2410.22774)
- **What's New**: 새롭게 제안된 방법은 기존의 CFAR 탐지기를 상태공간 모델 아키텍처와 결합하여, 신호 처리와 딥러닝의 장점을 통합했습니다. 이 방법은 CFAR 파이프라인을 유지하면서 파라미터 조정 없이 높은 탐지 성능을 달성합니다.

- **Technical Details**: 제안된 방법은 CFAR 알고리즘을 펼치는 방식으로 설계되었으며, CNN, RNN 및 CFAR 변형을 포함한 다양한 실험을 수행했습니다. 모델은 연속적인 단계에서 상태공간 표현을 시뮬레이션하여 입력 시퀀스를 출력 시퀀스로 매핑합니다.

- **Performance Highlights**: 제안된 방법은 기존 CFAR 변형보다 탐지율을 10배 높이고, 동일한 탐지율에서 잘못된 경고율을 10배 낮추며, 이전 CNN 및 RNN 기반의 신경 탐지기보다 현저히 낮은 복잡도를 보였습니다.



### An Overview of Causal Inference using Kernel Embeddings (https://arxiv.org/abs/2410.22754)
- **What's New**: 본 논문에서는 원인 추론(causal inference) 문제에 대한 해결책으로 커널 임베딩(kernel embeddings)의 가능성을 제시합니다. 커널 임베딩은 변수 간 복잡한 관계를 표현하는 데 유연성을 제공하며, 이 기술을 통해 관찰 데이터에서 평균 처리 효과(average treatment effect, ATE)와 같은 인과량을 예측할 수 있습니다.

- **Technical Details**: 커널 임베딩은 확률 측정(probability measures)을 재생 커널 힐베르트 공간(reproducing kernel Hilbert space, RKHS)으로 매핑하여, 비모수(nonparametric) 방법론을 통한 통계적 문제 해결을 지원합니다. 이 프레임워크는 관찰 분포(observational distributions)를 인과 분포(interventional distributions)로 변환하여 직접적 인과 관계를 찾는 데 적합합니다.

- **Performance Highlights**: 커널 임베딩을 활용하여 다양한 인과량을 추정하는 비모수 추정기(nonparametric estimators)를 구축할 수 있으며, 이는 기존 파라메트릭 방법보다 더 강인한 성능을 보입니다. 특히, 평균 처리 효과(ATE) 및 분포적 처리 효과(distributional treatment effect, DTE)를 추정하는 데 효과적입니다.



### Extensional Properties of Recurrent Neural Networks (https://arxiv.org/abs/2410.22730)
Comments:
          16 pages

- **What's New**: 재귀 신경망(RNN)의 \'extensional\' 특성을 검증하는 문제에 대해 부정적인 결과를 제시합니다. 이러한 특성을 테스트하는 것이 일반적으로 불가능하다는 것을 Rice의 정리에 근거하여 증명합니다.

- **Technical Details**: RNN은 벡터의 시퀀스를 입력으로 받아 처리한 후 또 다른 벡터 시퀀스를 출력하는 알고리즘으로 정의됩니다. 이 연구는 RNN 기계의 extensional 특성에 대한 공식적인 정의를 제공합니다.

- **Performance Highlights**: RNN의 extensional 특성이 불결정적임을 보여주며, 이는 특정 RNN 클래스에 대해 제한된 조건에서 알고리즘을 설계하는 것이 여전히 가능하다는 점을 강조합니다.



### Identifying Drift, Diffusion, and Causal Structure from Temporal Snapshots (https://arxiv.org/abs/2410.22729)
- **What's New**: 이번 연구는 개별 궤적을 관찰할 수 없을 때에도 stochastic differential equation (SDE)의 drift와 diffusion을 공동으로 추정할 수 있는 최초의 종합적 접근 방식을 제시합니다.

- **Technical Details**: SDE의 드리프트(drift)와 확산(diffusion)을 추정하기 위해 entropy-regularized optimal transport를 조정하고, APPEX (Alternating Projection Parameter Estimation from $X_0$)라는 반복 알고리즘을 도입했습니다. 이 알고리즘은 시간의 마진(marginals)만으로 드리프트, 확산 및 인과 그래프(causal graph)를 추정할 수 있도록 설계되었습니다. 또한, 초기 분포가 일반화된 회전류에 불변이 아닐 때 파라미터 식별 가능성을 입증하였습니다.

- **Performance Highlights**: APPEX 알고리즘은 Kullback-Leibler divergence에 대해 점근적으로 최적이며, 선형 덧셈 잡음 SDE에서 시뮬레이션된 데이터에 대한 효과성을 보여주었습니다.



### MassiveGNN: Efficient Training via Prefetching for Massively Connected Distributed Graphs (https://arxiv.org/abs/2410.22697)
Comments:
          In Proc. of the IEEE International Conference on Cluster Computing (CLUSTER), 2024

- **What's New**: 이 논문은 분산 메모리 환경에서 대규모 그래프 데이터에서 GNN 훈련의 샘플링 및 통신 비용을 개선하는 파라미터화된 연속 프리패칭(pre-fetching) 및 퇴출(eviction) 기법을 개발하여, Amazon DistDGL 프레임워크 내에서의 효율성을 높이는 실용적인 절충안을 제시합니다.

- **Technical Details**: 제안된 기법은 GNN 훈련 과정에서 현재 미니배치와 겹치는 연속적인 프리패칭 및 퇴출 프로세스를 통해 기능을 제공하며, DDP(Distributed Data Parallel) 훈련 프레임워크를 활용하여 GPU에 최적화된 훈련의 CPU 자원을 활용합니다. 이를 통해 커뮤니케이션 오버헤드를 약 15%에서 40% 절감할 수 있음을 보여줍니다.

- **Performance Highlights**: 이 논문에서는 NERSC Perlmutter 슈퍼컴퓨터에서 다양한 OGB 데이터셋으로 GraphSAGE 모델을 훈련시키는 성능 평가를 실시하였으며, 기존 DistDGL에 비해 약 15-40%의 강화된 훈련 성능을 기록했습니다.



### Dynamic PET Image Prediction Using a Network Combining Reversible and Irreversible Modules (https://arxiv.org/abs/2410.22674)
- **What's New**: 이 연구는 다이나믹 PET 이미징을 위한 새로운 예측 방법을 제안하여 스캔 시간을 줄이는 데 초점을 맞추고 있습니다.

- **Technical Details**: 제안된 방법은 가역적 및 비가역적 모듈로 구성된 다중 모듈 딥러닝 프레임워크(multi-module deep learning framework)를 사용하여 다이나믹 PET 이미지의 초기 프레임을 바탕으로 운동학 파라미터 이미지(kinetic parameter images)를 예측하고 전체 다이나믹 PET 이미지를 생성합니다.

- **Performance Highlights**: 시뮬레이션 데이터에 대한 검증 실험에서는 운동학 파라미터에 대한 좋은 예측 성능을 보여주었으며, 임상 데이터 실험에서도 좋은 일반화 성능을 발휘했습니다. 이러한 결과는 제안된 방법이 임상 적용 가능성이 높음을 시사합니다.



### A Walsh Hadamard Derived Linear Vector Symbolic Architectur (https://arxiv.org/abs/2410.22669)
Comments:
          To appear in the 38th Conference on Neural Information Processing Systems (NeurIPS 2024)

- **What's New**: 본 논문에서는 Hadamard 유도 선형 바인딩(HLB)을 소개합니다. HLB는 전통적인 VSA 작업에서 뛰어난 성능을 발휘할 수 있도록 설계되었으며, 딥러닝 환경에서도 효과적으로 작동할 수 있는 특성을 가지고 있습니다.

- **Technical Details**: HLB는 Walsh Hadamard 변환에서 유도된 VSA로, 바인딩 단계에서의 계산 복잡도를 O(d)로 축소하고, 수치적 안정성을 보장합니다. 이 메서드는 바인딩 연산에서 O(d log d)와 같은 전통적인 Hadamard 변환의 복잡도를 피하고, 보다 비용이 많이 드는 VSA 대안보다 우수한 성능을 나타냅니다.

- **Performance Highlights**: HLB는 전통적인 VSA 벤치마크 결과 및 두 가지 최근 딥러닝 작업에서의 성능을 개선하였으며, 기존 VSA 시스템과 기준 작업에서 비교 시 동등하거나 개선된 성능을 보여줍니다.



### SleepNetZero: Zero-Burden Zero-Shot Reliable Sleep Staging With Neural Networks Based on Ballistocardiograms (https://arxiv.org/abs/2410.22646)
Comments:
          25 pages

- **What's New**: 본 논문에서는 BCG(Ballistocardiography)를 기반으로 한 최초의 신뢰성 있는 수면 단계 분류 시스템인 SleepNetZero를 소개합니다. 이 시스템은 제한된 BCG 데이터 문제를 해결하고 다양한 인구 집단에서 일반화 능력을 향상시키는 방식으로 설계되었습니다.

- **Technical Details**: SleepNetZero는 zero-shot learning 접근 방식을 사용하여 수면 단계를 분류합니다. BCG 신호에서 세 가지 주요 특징(heartbeat, respiration, body movement)을 추출하고, 대규모 PSG(polysomnography) 데이터셋을 활용하여 훈련합니다. 또한, 데이터 증강(data augmentation) 기술을 통해 일반화 능력을 향상시킵니다. 최종적으로, ResNet 기반 특성 추출기와 Transformer 인코더를 포함한 신경망 프레임워크가 구성됩니다.

- **Performance Highlights**: 모델은 12393개의 레코드와 9637명의 다른 피험자를 사용하여 훈련 및 테스트되었으며, 정확도 0.803과 Cohen's Kappa 0.718을 달성했습니다. 실제 병원 환경에서 265명의 사용자에게 사용된 SleepNetZero는 정확도 0.697과 Cohen's Kappa 0.589를 기록했습니다. 이는 BCG 신호를 활용한 수면 단계 분류의 중요한 이정표로 간주됩니다.



### Feature Responsiveness Scores: Model-Agnostic Explanations for Recours (https://arxiv.org/abs/2410.22598)
Comments:
          11 pages, 3 figures in main body

- **What's New**: 본 연구는 머신러닝 모델의 예측을 설명하는 방법을 재고하며, 소비자 보호의 주요 목표 중 하나인 '구제(Recourse)'를 달성하기 위한 방법에 대해 논의합니다. 특히, 기존의 feature attribution 방법인 LIME과 SHAP가 소비자에게 공정한 정보를 제공하지 못할 수 있음을 강조합니다.

- **Technical Details**: 예측의 응답성(Responsiveness) 점수를 계산하는 새로운 방법을 제안합니다. 이 점수는 특정 feature를 변경하여 모델의 예측을 변화시킬 수 있는 확률을 측정합니다. 우리의 방법은 복잡한 실행 가능성 제약 조건(Complex Actionability Constraints) 하에서 모든 모델 및 데이터 세트에 대해 응답성 점수를 효율적으로 계산할 수 있는 방법을 개발합니다.

- **Performance Highlights**: 실험 결과, 기존의 feature attribution 방법들이 소비자에게 '구제'를 제공하지 못하는 경우가 많음을 발견했습니다. 새로운 응답성 점수를 사용한 모델은 소비자가 예측 개선을 위해 변경할 수 있는 feature를 잘 식별하는 것으로 나타났으며, 이를 통해 소비자의 피해를 최소화할 수 있는 방법을 제안했습니다.



### Pre-Trained Vision Models as Perception Backbones for Safety Filters in Autonomous Driving (https://arxiv.org/abs/2410.22585)
- **What's New**: 이 논문에서는 비전 기반 자율 주행의 안전 문제 해결을 위한 새로운 접근법을 제시합니다. 특히, 프리트레인(pre-trained)된 비전 모델을 안전 필터에 활용하여 고차원 환경에서도 안전한 제어를 가능하게 합니다.

- **Technical Details**: 저자들은 DeepAccident 데이터셋을 사용하여 차량의 행동을 주석 달아 제공된 멀티 카메라 비디오를 기반으로 다양한 훈련 기법을 평가합니다. 이 연구에서는 제어 장치의 동적 모델이 알려지지 않은 블랙박스(dynamic control systems) 설정에서 안전 필터를 훈련시키는 세 가지 방법을 시도합니다.

- **Performance Highlights**: 결과적으로, 프리트레인 PVR(Pre-trained Vision Representations)을 활용한 안전 필터는 고전적인 필터와 경쟁력을 갖추며, 때때로 지상 진실 상태(ground truth state)에 대한 접근이 있는 필터보다 나은 성능을 보입니다. 또한, 다중 카메라 피드를 활용한 경우 각 카메라 피드를 별도로 사용할 때보다 하나의 상태로 융합하는 것이 더 효과적임을 보여줍니다.



### Energy-Aware Multi-Agent Reinforcement Learning for Collaborative Execution in Mission-Oriented Drone Networks (https://arxiv.org/abs/2410.22578)
Comments:
          2022 International Conference on Computer Communications and Networks

- **What's New**: 이 연구는 다중 에이전트 강화 학습(multi-agent reinforcement learning, MARL)을 활용해 드론 네트워크의 협업 실행 모델을 제안합니다. 또한 드론의 배터리 수준을 고려하여 경로(planning trajectories)를 효율적으로 계획할 수 있는 방법을 제시합니다.

- **Technical Details**: 제안된 모델은 드론 하드웨어의 배터리 한계를 고려하여 다중 작업 수행에 필요한 경로를 최적화합니다. 드론 간의 협업은 심층 Q-네트워크(deep Q-network, DQN)를 통해 이루어지며, 드론은 현재 상태와 환경 정보를 바탕으로 경로를 조정하고 임무를 수행합니다.

- **Performance Highlights**: 시뮬레이션 결과, 제안된 협업 실행 모델은 임무 완료 성공률이 최소 80%에 달하며, 작업 밀도가 적당할 경우 성공률이 100%에 이릅니다. 이는 기존 연구에 비해 향상된 성능을 보여줍니다.



### Orb: A Fast, Scalable Neural Network Potentia (https://arxiv.org/abs/2410.22570)
- **What's New**: Orb는 원자 모사(atomistic modelling)를 위한 범용적인 상호 원자 포텐셜(universal interatomic potentials)의 새로운 패밀리로, 기존의 포텐셜에 비해 3-6배 더 빠르고, 다양한 비평형 재료에 대해 안정적인 성능을 보여줍니다. Matbench Discovery 벤치마크에서 31%의 오류 감소를 기록하여 기존 방법을 초과하는 성능을 자랑합니다.

- **Technical Details**: Orb는 원자 간 상호작용의 복잡성과 불변성을 데이터에서 학습하는 스케일러블 그래프 신경망(graph neural network) 아키텍처를 기반으로 개발되었습니다. Orb의 구조는 Encoder, Processor, Decoder의 세 단계로 구성되며, 메시지 패싱(message passing)을 통해 원자 시스템을 그래프로 모델링합니다. 각 원자는 벡터 임베딩을 가지며, 원자 간의 변위는 방향성 있는 엣지 특징으로 표현됩니다.

- **Performance Highlights**: Orb는 Commodore 하드웨어에서 3-6배 빠른 성능을 발휘하며, Van der Waals 힘이 중요한 역할을 하는 재료 모델링에 적합한 학습된 분산 보정(dispersion corrections)을 포함합니다. 모델은 Apache 2.0 라이센스 하에 공개되어 연구 및 상업적 사용이 가능합니다.



### Fast Deep Hedging with Second-Order Optimization (https://arxiv.org/abs/2410.22568)
- **What's New**: 본 연구에서는 시장 마찰이 있는 상황에서 이국적 옵션(Exotic Options)을 헤지하기 위한 딥 헤징(Deep Hedging) 접근법을 제안합니다. 뉴럴 네트워크 정책을 훈련하여 현실적인 시장에서의 헤징 문제를 해결하고자 하며, 특히 긴 만기와 복잡한 시장 파라미터에 대한 민감성을 가지는 옵션에 대한 느린 수렴 문제를 개선합니다.

- **Technical Details**: 본 논문에서는 헤징 최적화를 위해 두 번째 차수 최적화 기법을 제안합니다. 경로별 미분(Pathwise Differentiability)을 활용하여 블록 대각선 및 크로네커 풀림(Kronecker Factorization)으로 근사화된 곡률 행렬을 구성하고, 이를 통해 그라디언트를 효율적으로 전처리합니다. 제안된 방법은 스토캐스틱 변동성 모델에서 클리케 옵션(Clilet Option)을 헤지하기 위해 스팟과 바닐라 옵션(Plain Vanilla Options)을 거래하는 문제에 적용됩니다.

- **Performance Highlights**: 두 번째 차수 최적화 기법은 표준 적응적 순간 기반 최적화 방법에 비해 1/4의 단계 수로 정책을 최적화하며, 이로 인해 훈련 시간이 크게 단축됩니다. 이러한 결과는 추가 계산 비용을 분산할 수 있기 때문에 가능하며, 새로운 방법론의 효용성을 강조합니다.



### Auto-Intent: Automated Intent Discovery and Self-Exploration for Large Language Model Web Agents (https://arxiv.org/abs/2410.22552)
Comments:
          EMNLP 2024 Findings

- **What's New**: 본 논문에서는 Auto-Intent를 소개하며, 이는 학습하여 구축한 대형 언어 모델(LLM)을 직접적인 fine-tuning 없이 특정 도메인에 맞게 조정하는 방법입니다. 이 방법은 웹 탐색 작업을 중심으로 경험적으로 연구되었습니다.

- **Technical Details**: Auto-Intent는 대상 도메인의 시연에서 기본적인 의도(intents)를 비지도학습(unsupervised) 방식으로 발견합니다. 이 의도들은 최대 3단어로 매우 간결하게 표현되며, 이를 통해 다음 의도를 예측하는 의도 예측기(intent predictor)를 학습합니다. 특히 self-exploration 접근 방식을 통해, 가장 가능성이 높은 top-k 의도 예측 결과를 LLM 에이전트에게 제공하여 의사결정 능력을 향상시킵니다.

- **Performance Highlights**: Auto-Intent는 Mind2Web의 대규모 실 웹 탐색 벤치마크(Task benchmarks)와 WebArena의 온라인 탐색 작업에서 GPT-{3.5, 4}와 Llama-3.1-{70B, 405B} 에이전트의 성능을 실질적으로 향상시켰습니다.



### Towards Neural-Network-based optical temperature sensing of Semiconductor Membrane External Cavity Laser (https://arxiv.org/abs/2410.22528)
- **What's New**: 이 논문에서는 레이저 방출을 통해 레이저 이득 매체의 온도를 비접촉식으로 결정하는 머신러닝 방법이 소개됩니다. 학습된 다층 신경망 모델을 이용하여 스펙트럴 데이터만으로 장치의 특성을 예측할 수 있습니다.

- **Technical Details**: 피드포워드 Neural Network (NN) 모델은 가시광선/근적외선 소형 마이크로 스펙트로미터로 기록된 스펙트럴 데이터로 훈련됩니다. 이 논문에서는 다이오드 펌프 레이저와 반도체 디스크 레이저의 광학 펌프 이득 멤브레인에 대해 다루며, 대량의 라벨링된 강도 데이터를 수집하여 예측 과정에 활용합니다.

- **Performance Highlights**: 논문에서 제안한 방법은 적어도 1% 미만의 정확도로 온도를 추론할 수 있으며, 추가적인 광학 진단이나 온도 센서 없이 레이저 시스템 온도를 빠르고 신뢰성 있게 추론하는 능력을 구현합니다.



### Evaluating utility in synthetic banking microdata applications (https://arxiv.org/abs/2410.22519)
Comments:
          28 pages, 4 figures

- **What's New**: 본 논문에서는 은행 마이크로데이터의 합성 데이터 생성을 위한 새로운 평가 프레임워크를 개발하고 이를 파라과이 중앙은행의 데이터에 적용하여 공개 가능한 합성 데이터셋을 생성했습니다. 이는 재정 규제 당국에 통계적 기밀성을 유지하면서 유용성을 제공하는 혁신적인 접근 방식으로 주목받습니다.

- **Technical Details**: 이 연구는 중앙은행의 데이터에서 용도와 개인정보 보호 요구 사항을 고려한 합성 데이터 생성 프레임워크를 제안합니다. 적용된 사례로는 재정 이용 지수, 정기 예금 수익 곡선, 신용 카드 전환 행렬 등이 있으며, 370만 명 이상의 개인 데이터, 13,000개의 정기 예금을 포함한 합성 데이터셋이 수집되었습니다.

- **Performance Highlights**: 본 연구의 결과는 포스트 프로세싱 정보 손실에 덜 민감한 주파수 표 기반 애플리케이션이 적합하며, 이러한 작업에서 제너레이티브 적대 신경망 모델보다 우수한 마진 기반 추론 메커니즘이 효과적임을 보여줍니다. 이는 통계 공개를 보완하고자 하는 재정 규제 당국을 위한 유망한 개인 정보 보호 기술로 자리잡을 가능성을 시사합니다.



### Attention Speaks Volumes: Localizing and Mitigating Bias in Language Models (https://arxiv.org/abs/2410.22517)
- **What's New**: 본 논문은 큰 언어 모델(LLMs)에서 애매한 비교 프롬프트가 주어졌을 때 어떻게 편향(bias)이 발생하는지를 탐구합니다. 새로운 방법으로, 편향을 특정 레이어에 국한시키고 주의(attention) 점수를 분석하여 이를 완화할 수 있는 기법인 $	exttt{ATLAS}$를 제안합니다.

- **Technical Details**: $	exttt{ATLAS}$는 두 단계의 접근법으로, 첫 번째로 주의 점수를 분석하여 편향이 집중된 레이어를 식별하고, 두 번째로 이러한 편향 레이어에 대해 주의 점수를 조정하여 편향을 줄이는 것입니다. 실험은 다양한 모델(GPT-2 XL, GPT-J, LLaMA-2 및 LLaMA-3)과 데이터셋(BBQ, Crows-Pairs, WinoGender)을 사용하여 수행되었습니다.

- **Performance Highlights**: 우리의 실험 결과, 편향은 모델의 후반 레이어에 집중되어 있으며, $	exttt{ATLAS}$는 다운스트림 성능에 영향을 주지 않으면서 편향을 효과적으로 완화하였습니다. 평균적으로 0.28 포인트의 편향 점수 개선이 이루어졌습니다.



### Privacy-Preserving Dynamic Assortment Selection (https://arxiv.org/abs/2410.22488)
- **What's New**: 본 논문은 사용자 맞춤형 추천에서 개인 정보 보호의 중요성을 강조하며, multinomial logit (MNL) bandits 모델을 활용하여 개인 정보를 보존하는 동적 상품 assortments 선택을 위한 새로운 프레임워크를 제안합니다. 이 접근법은 사용자 유틸리티 추정치에 보정된 노이즈를 통합하여 탐색(exploration)과 착취(exploitation) 간의 균형을 맞추는 동시에 강력한 프라이버시 보호를 보장합니다.

- **Technical Details**: 제안된 방법은 perturbed upper confidence bound (UCB) 방법을 사용하며, Joint Differential Privacy (JDP)를 만족하여 동적 환경에서 적합한 개인 정보 보호 방안을 제공합니다. 이 기법은 MNL bandits에 맞춰 설계된 새로운 목표 왜곡(objective perturbation) 기법에 기반하여 동작합니다. 이론적으로 제안된 정책의 거의 최적의 후회 경계(regret bound)인 \(	ilde{O}(	ext{sqrt}(T))\)를 도출하였고, 개인 정보 보호가 후회에 미치는 영향을 명확히 수량화하였습니다.

- **Performance Highlights**: 실험 및 Expedia 호텔 데이터 세트에 대한 적용을 통해, 제안된 방법이 기존 기준 방법에 비해 상당한 성능 향상을 달성함을 입증하였습니다. 이는 개인 맞춤형 추천의 정확성을 높이는 데 기여할 것으로 보입니다.



### Bayesian Counterfactual Prediction Models for HIV Care Retention with Incomplete Outcome and Covariate Information (https://arxiv.org/abs/2410.22481)
- **What's New**: 이번 논문은 HIV(인간 면역 결핍 바이러스) 환자의 치료 유지를 예측하고 최적의 방문 예약을 추천하기 위해 전자 건강 기록(EHR)을 활용한 데이터 기반 방법을 제시합니다. 특히, 이 과정에서 발생할 수 있는 여러 복잡성을 고려하여 잠재적인 인과 관계에 대한 추정을 실행합니다.

- **Technical Details**: 논문에서 제안하는 방법은 가변적인 예약 결정의 맥락에서 인과적 유지 추정량(causal retention estimands)을 정형화하고 식별하는 것입니다. 이를 위해 세미파라메트릭 베이지안 전이 모델을 사용하여 상태 간의 전환 확률을 계산하고, 후속 방문 가능성을 예측합니다. 이 과정에서 경쟁 사건(예: 사망)과 검열(censoring) 등이 적절히 고려됩니다.

- **Performance Highlights**: 이 연구에서는 AMPATH(Academic Model Providing Access to Healthcare) 프로그램의 EHR 데이터를 적용하여 HIV 치료의 데이터 기반 의사결정 지원을 제공하도록 하였으며, 사망이나 검열된 환자도 유효하게 모델에 기여하게 만들어 정확성을 높였습니다. 이를 통해 HIV 관리에서 환자의 예약 결정에 대한 통찰력을 강화할 수 있는 가능성을 보였습니다.



### Advancing Agentic Systems: Dynamic Task Decomposition, Tool Integration and Evaluation using Novel Metrics and Datas (https://arxiv.org/abs/2410.22457)
Comments:
          38th Conference on Neural Information Processing Systems (NeurIPS 2024), NeurIPS 2024 Workshop on Open-World Agents

- **What's New**: 이 논문은 자율 에이전트 시스템의 발전을 위해 'Advanced Agentic Framework'를 제안하며, 이를 통해 다단계 작업을 역동적으로 처리하고 적절한 도구를 자동으로 선택할 수 있는 능력을 강화하고 있습니다.

- **Technical Details**: 논문에서는 주요 기여로 세 가지 요소를 제시합니다: 1) 다단계 쿼리를 처리하고 작업 그래프를 생성 및 실행하며 적합한 도구를 선택하고 실시간 변화에 적응하는 고급 에이전트 프레임워크, 2) 에이전트 시스템에 대한 전반적인 평가를 위한 새로운 평가 지표(Node F1 Score, Structural Similarity Index (SSI), Tool F1 Score), 3) 다양한 작업 복잡성에서 에이전트 행동을 분석하기 위한 AsyncHow 기반의 전문 데이터셋을 개발하였습니다.

- **Performance Highlights**: 강화된 작업 그래프 분해는 시스템의 응답성과 확장성을 크게 향상시킵니다. 특히, SSI는 순차적 작업에서 성능의 가장 중요한 예측 변수가 되고, Tool F1 Score는 병렬 작업에서 필수적인 성과 지표로 나타났습니다.



### Explainable convolutional neural network model provides an alternative genome-wide association perspective on mutations in SARS-CoV-2 (https://arxiv.org/abs/2410.22452)
- **What's New**: 이 연구는 COVID-19의 원인 바이러스인 SARS-CoV-2의 변종과 그에 따른 표현형 변화(phenotypic changes)와 관련된 변이를 식별하는 방법에 대해 설명합니다. 전통적인 유전체 연관 연구(GWAS)와 설명 가능한 합성 신경망(CNN)을 비교하여 SARS-CoV-2 변종에 대한 예측을 수행했습니다.

- **Technical Details**: 연구에서는 CNN 분류 모델을 훈련시켜 유전체 시퀀스를 Concern 변종(Variants of Concern, VOCs)으로 예측했습니다. 그런 다음 Shapley Additive 설명(Shapley Additive Explanations, SHAP) 모델을 적용하여 올바른 예측에 중요한 변이를 식별했습니다. 이와 비교하여 전통적인 GWAS를 수행하여 VOC와 관련된 변이를 찾아냈습니다.

- **Performance Highlights**: 비교 결과, 설명 가능한 신경망 접근법이 spike 유전자의 특정 위치와 같은 VOC와 관련된 알려진 뉴클레오타이드 치환(nucleotide substitutions)을 보다 효과적으로 보여줄 수 있다는 점이 확인되었습니다. 이는 전통적인 유전체 분석 방식에 대한 유망한 대안을 제시합니다.



### A Closer Look at Neural Codec Resynthesis: Bridging the Gap between Codec and Waveform Generation (https://arxiv.org/abs/2410.22448)
Comments:
          NeurIPS 2024 Audio Imagination workshop paper; demo page at this https URL

- **What's New**: 이 논문에서는 neural audio codec의 coarse 토큰에서 waveforms를 재합성(resynthesize)하는 방법을 탐구합니다. 일반적으로 기존 연구들은 coarse 토큰 생성을 우선시했지만, 저자들은 coarse 토큰만을 활용하여 어떻게 높은 품질의 오디오를 재합성할 수 있는지에 대해 주목합니다.

- **Technical Details**: 저자들은 코덱 재합성을 위한 두 가지 주요 접근 방식, 즉 토큰 예측(token prediction)과 회귀(regression)에 기반한 전략을 연구하고, 새로운 방법인 Schrödinger Bridge를 제안합니다. 이러한 방법은 다양한 설계 선택이 기계 및 인간 인식에 미치는 영향을 분석합니다.

- **Performance Highlights**: 이 연구는 coarse 임베딩을 기반으로 한 오디오 재합성을 통해 기존의 ad-hoc 방법들을 개선하고, 레벨에 따른 정보 구조를 활용하여 더 나은 오디오 품질을 달성할 수 있는 가능성을 보여줍니다.



### EfficientNet with Hybrid Attention Mechanisms for Enhanced Breast Histopathology Classification: A Comprehensive Approach (https://arxiv.org/abs/2410.22392)
- **What's New**: 이 논문은 Hybrid EfficientNet 모델을 기존의 주의 메커니즘(Attention Mechanisms)인 Convolutional Block Attention Module (CBAM), Self-Attention, Deformable Attention과 통합하여 유방암 조직병리 이미지 분류를 향상시키는 새로운 접근 방식을 제시합니다.

- **Technical Details**: 제안된 모델은 BreakHis 데이터셋을 사용하여 4가지 배율(40X, 100X, 200X, 400X)에서 여러 측정 지표로 평가됩니다. Hybrid EfficientNet은 복잡한 조직 구조를 효과적으로 처리하며, CBAM, Self-Attention, Deformable Attention이 포함되어 있어 이미지의 중요한 부분에 초점을 맞춰 특징 추출 능력을 향상시킵니다.

- **Performance Highlights**: 제안된 방법은 400X 배율에서 98.42%의 정확도로 여러 최신 모델(VGG, ResNet 등)을 초과하는 성능을 보여주며, 정확성, F1-score, 정밀도, 재현율 등의 지표로 결과가 검증되었습니다. 이 모델은 실시간 진단 워크플로우에 통합될 수 있는 높은 계산 효율성을 보입니다.



### ET-Flow: Equivariant Flow-Matching for Molecular Conformer Generation (https://arxiv.org/abs/2410.22388)
Comments:
          NeurIPS 2024

- **What's New**: 본 논문에서는 Equivariant Transformer Flow (ET-Flow)를 제안하여 기존의 복잡한 내부 기하학 계산과 대규모 아키텍처의 필요성을 줄이며, 효과적인 low-energy 분자의 소형 모델을 개발하였습니다.

- **Technical Details**: ET-Flow는 flow matching 기법을 통해 학습의 효율성을 극대화하였고, Equivariant Transformer를 활용하여 기하학적 특징을 잘 포착합니다. 또한, Harmonic Prior 를 통합하여 결합된 원자들이 가까운 위치를 유지하도록 유도합니다.

- **Performance Highlights**: ET-Flow는 기존의 모델들에 비해 상태-of-the-art 성능을 제공하며, 보다 실제적인 물리적 특성을 가진 분자를 생성합니다. 샘플링 단계 수가 극적으로 줄어들어 효율성이 향상되었습니다.



### Debiasing Alternative Data for Credit Underwriting Using Causal Inferenc (https://arxiv.org/abs/2410.22382)
- **What's New**: 이 논문은 대체 데이터의 편향을 제거하기 위해 인과 추론 기법을 적용하는 방법을 제안합니다. 이를 통해 대출 심사에 대체 데이터를 활용하고자 하며, 인종 그룹 간의 모델 정확성을 향상시키고자 합니다.

- **Technical Details**: 전통적인 신용 점수는 과거 지불 이력, 채무 액수, 신용 이력의 길이와 같은 요소에 기반합니다. 논문에서는 인과 베이지안 네트워크 (Causal Bayesian Networks, CBNs)를 이용하여 공정성과 기계 학습의 관계를 설명하고, 대체 데이터의 편향을 제거하는 알고리즘을 제시합니다.

- **Performance Highlights**: 제안된 알고리즘은 공공 신용 데이터셋을 기반으로 테스트되었으며, 다양한 인종 그룹에 걸쳐 모델의 정확도가 향상되는 것을 보여주었습니다. 이 과정에서 비차별 보장(providing nondiscrimination guarantees)도 강조되었습니다.



### Survey of User Interface Design and Interaction Techniques in Generative AI Applications (https://arxiv.org/abs/2410.22370)
- **What's New**: 이 논문은 현재의 인간-인공지능 상호작용 분야에서 사용자 인터페이스(UI) 디자인 및 사용자 상호작용 패턴에 대한 포괄적인 조사 결과를 제시합니다. 연구자들과 개발자들이 AI 애플리케이션을 설계하는 데 참고할 수 있는 다양한 사용자 상호작용 패턴을 문서화하고, 이를 통해 생성적 AI 애플리케이션 디자인의 진입장벽을 낮추는 것을 목표로 합니다.

- **Technical Details**: 이 설문조사는 사용자가 주도하는 상호작용(user-guided interactions)을 중심으로 진행되며, 이는 사용자가 의도적으로 생성적 AI 시스템에 영향을 미치는 모든 상호작용을 포함합니다. 구체적으로는 다양한 상호작용 기술과 이를 위한 UI 레이아웃, 사용자-AI 참여 수준을 분류하여 정리합니다. 또한, 텍스트 또는 이미지의 프롬프트(prompt)를 통해 사용자가 생성 모델을 제어하는 방식도 다룹니다.

- **Performance Highlights**: 이 연구는 100개 이상의 관련 생성적 AI 논문을 바탕으로 다양한 상호작용 기술과 UI 디자인 패턴을 카테고리화하였으며, 이는 디자이너들이 기존의 패턴을 활용하여 보다 직관적인 사용자 경험을 제공할 수 있도록 돕습니다. 연구 결과는 디자이너와 연구자 모두에게 생성적 AI 애플리케이션의 효과적인 설계를 위한 기초 자료가 될 것입니다.



### MAMMAL -- Molecular Aligned Multi-Modal Architecture and Languag (https://arxiv.org/abs/2410.22367)
- **What's New**: 이번 연구에서는 MAMMAL (Molecular Aligned Multi-Modal Architecture and Language)라는 새로운 모델을 소개합니다. 이 모델은 생물학적 데이터셋에서 학습하며, 여러 모달리티를 통합하여 다양한 약물 발견 관련 작업을 지원합니다. MAMMAL은 다중 작업을 처리할 수 있는 기반 모델을 통해 새로운 성능 기준(SOTA)을 달성했습니다.

- **Technical Details**: MAMMAL은 텍스트 생성, 회귀, 분류 등의 다양한 태스크를 처리하기 위해, 샘플 사이즈 20억의 대규모 데이터셋에서 훈련된 다중 정렬 모델입니다. 이 모델은 Transformer 아키텍처를 기반으로 하며, 모델 아키텍처는 인코더-디코더 또는 인코더 전용 모드로 작업을 수행할 수 있습니다. 새로운 '모듈형 토크나이저'를 통해 다양한 분자 도메인 어휘를 지원합니다.

- **Performance Highlights**: MAMMAL은 11개의 다양한 하위 작업을 평가한 결과, 9개 작업에서 새로운 SOTA를 달성하고, 나머지 2개 작업에서는 기존 SOTA와 비슷한 성능을 보였습니다. 이는 단일 통합 모델을 사용하여 모든 작업을 처리하고, 맞춤형 아키텍처 없이도 가능했습니다.



### Accelerating Augmentation Invariance Pretraining (https://arxiv.org/abs/2410.22364)
- **What's New**: 이 연구에서는 Vision Transformers (ViTs)의 사전 훈련을 위한 대조 학습 방법의 계산적 도전 과제를 해결하기 위해, 계산 자원을 줄이기 위한 새로운 가속 프레임워크를 제안합니다.

- **Technical Details**: 제안된 방법은 무작위 토큰 드롭아웃(randomized token dropout)과 유연한 패치 스케일링(flexible patch scaling) 같은 시퀀스 압축 전략을 결합하여 그래디언트 추정(gradiend estimation) 비용을 줄이고 수렴(convergence)을 가속화합니다. 또한, 다양한 가속 전략의 그래디언트 추정 오차를 분석하고 이를 다운스트림 작업에 미치는 영향을 연구합니다.

- **Performance Highlights**: 제안된 방법은 MoCo에서 4배, SimCLR에서 3.3배, DINO에서 2.5배의 훈련 속도를 달성하여 대규모 데이터셋에서 계산 오버헤드를 크게 줄입니다.



### GleanVec: Accelerating vector search with minimalist nonlinear dimensionality reduction (https://arxiv.org/abs/2410.22347)
- **What's New**: 이 논문은 고차원 벡터 검색을 가속화하고 정확도를 유지할 수 있는 새로운 선형(Linear) 및 비선형(Nonlinear) 차원 축소 방법인 LeanVec-Sphering과 Generalized LeanVec(GleanVec)를 소개합니다.

- **Technical Details**: LeanVec-Sphering은 기존의 차원 축소 기법보다 더 우수하며, 하이퍼파라미터가 필요하지 않고 검색 중에 타겟 차원(Dimensionality)을 설정할 수 있습니다. GleanVec는 조각별 선형 구조를 사용하여 검색 정확성을 높이면서 계산 성능 또한 강조합니다.

- **Performance Highlights**: 초기 실험 결과에 따르면, LeanVec-Sphering과 GleanVec는 벡터 검색 상태를 끌어올리며, 높은 차원에서도 성능 저하를 방지할 수 있습니다.



### Representation Learning for Regime detection in Block Hierarchical Financial Markets (https://arxiv.org/abs/2410.22346)
Comments:
          6 pages. Presented at the 2024 IEEE CIFEr conference. Analysis of block-resampled chronology-preserving lead-lag learning dynamics presented at: this https URL

- **What's New**: 본 논문에서는 시장 상태 탐지를 위해 SPD(산업적 안정성) 기하학과 관련된 심층 표현 학습을 이용하는 새로운 접근 방식을 제시합니다. 역사적 코릴레이션 구조를 기반으로 한 세 가지 모델(SPDNet, SPD-NetBN, U-SPDNet)을 평가하여 시장의 변화를 특성화합니다.

- **Technical Details**: 이 모델들은 입력 블록 계층 SPD 상관 행렬의 기하학적 특성을 존중하며, 세 가지 데이터 구성(randomized JSE Top 60 data, synthetically-generated block hierarchical SPD matrices, block-resampled chronology-preserving JSE Top 60 data)에서 시장 단계 탐지를 수행합니다. 또한, Riemannian autoencoder 구조를 통해 잠재적 블록 계층 특징 추출을 평가합니다.

- **Performance Highlights**: U-SPDNet은 SPDNet보다 약간 더 나은 수익률을 제공하지만, 두 모델 모두 동일가중 기준 포트폴리오의 성과에는 미치지 못합니다. 성능 검토는 정보 누수(life leakage)를 정제하여 진행되었습니다.



### Improving the accuracy of food security predictions by integrating conflict data (https://arxiv.org/abs/2410.22342)
- **What's New**: 이 논문은 아프리카 내에서의 폭력적 갈등이 식량 안보에 미치는 영향을 정량적으로 분석한 연구입니다. 특히, 갈등 데이터를 머신러닝 모델에 통합함으로써 식량 안보 예측의 정확도를 1.5% 향상시키는 것을 보여주었습니다.

- **Technical Details**: 이 연구에서는 Famine Early Warning Systems Network (FEWSNET)과 Armed Conflict Location Event Data (ACLED)에서 수집된 데이터를 이용하여 폭력적인 갈등이 식량 안보 예측에 미치는 영향을 분석했습니다. 다중 상관 분석(multi-level correlation analysis)과 머신러닝(machine learning) 모델을 사용하여 데이터 통합의 필요성과 갈등 다이나믹스(conflict dynamics)가 식량 안보에 미치는 영향을 정량적으로 평가했습니다.

- **Performance Highlights**: 식량 안보 예측 모델에 갈등 데이터를 통합하는 것이 예측의 정확성을 1.5% 증가시켰으며, 이 연구는 향후 정책 결정을 위한 데이터 기반의 이해를 증진시키는 중요한 기초 자료를 제공합니다.



### PACER: Physics Informed Uncertainty Aware Climate Emulator (https://arxiv.org/abs/2410.21657)
- **What's New**: 본 논문에서는 PACER라는 경량의 기후 모사기 (climate emulator)를 제안하며, 이는 온실가스 배출 데이터에 기반하여 기온과 강수량을 86년 동안 안정적으로 모사할 수 있습니다. PACER는 물리 법칙인 advection-diffusion을 통합하고 이는 데이터 효율성을 높이는 데 기여합니다.

- **Technical Details**: PACER는 Neural ODE 기반으로, 온실가스 농도에 따라 동적으로 확산 계수(diffusion coefficient)와 흐름 속도(flow velocities)를 추정합니다. Gaussian noise를 stochastic term으로 도입하여 기후 모사에서의 불확실성을 고려하며, 지구 대기를 구형(domain of spherical)으로 모델링하여 주기적 경계 조건을 인코딩합니다.

- **Performance Highlights**: PACER는 15개의 기후 모델을 대상으로 실험을 수행하였으며, 대부분의 기후 모델에서 기준선 성능을 초월했습니다. PACER는 86년 동안 안정적으로 온도와 강수량을 모사하는 데 성공하여 기후 진단 작업(climate diagnostic task)에서 새로운 최첨단 성과를 달성했습니다.




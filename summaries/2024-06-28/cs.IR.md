New uploads on arXiv(cs.CL)

### Suri: Multi-constraint Instruction Following for Long-form Text Generation (https://arxiv.org/abs/2406.19371)
- **What's New**: 기존 연구는 단순한 지시와 짧은 응답에 초점을 맞췄으나, 이번 연구는 다중 제약(long-form text) 지시를 따르는 능력을 탐구합니다. 이를 위해 20K 개의 사람이 작성한 긴 형식의 텍스트와 LLM이 생성한 복잡한 지시를 쌍으로 가진 Suri 데이터를 만들었습니다. 장문 텍스트에 대한 인간의 선호도 판단을 수집하는 데 어려움이 있어 DPO 같은 선호 조정 알고리즘이 불가능하기 때문에, 우리는 ORPO 알고리즘에 기반한 정렬 방법인 I-ORPO를 제안합니다.

- **Technical Details**: Suri 데이터셋은 백번역(instruction backtranslation)을 통해 생성된, 여러 복잡한 제약들을 포함한 지시가 쌍을 이루는 20K 개의 사람이 작성한 텍스트로 구성됩니다. 기존의 SFT가 장문 데이터셋에서 효과적이지 않음을 발견했으므로, 우리는 LLM이 생성한 합성적으로 손상된 지시를 음성 피드백으로 받는 I-ORPO 정렬 방법을 개발했습니다. 실험은 Mistral-7b-Instruct-v0.2 모델을 사용하여 진행되었습니다.

- **Performance Highlights**: Suri 데이터셋에서 I-ORPO와 SFT를 사용한 모델 튜닝 결과, 텍스트 길이가 약 1K 토큰에서 5K 토큰으로 증가하였으며, 품질 저하 없이 길이가 늘어났습니다. 인간 평가 결과, 두 모델 모두 대부분의 제약을 만족했으나, Suri-I-ORPO 모델이 일관성과 정보성을 고려하여 더 선호되었습니다.



### The Model Arena for Cross-lingual Sentiment Analysis: A Comparative Study in the Era of Large Language Models (https://arxiv.org/abs/2406.19358)
Comments:
          Accepted to WASSA workshop at ACL2024

- **What's New**: 이번 연구는 대형 언어 모델들(Large Language Models, LLMs)의 교차 언어 감정 분석 능력을 본격적으로 분석합니다. 이 연구는 특히 XLM-R 및 mT5와 같은 소형 다국어 언어 모델(Small Multilingual Language Models, SMLMs)과 Llama-3, GPT-3.5, GPT-4 등의 대형 언어 모델을 비교합니다. 결과에 따르면, 소형 다국어 언어 모델은 zero-shot 교차 언어 성능에서 우수하며, 소수 샘플(few-shot) 환경에서는 대형 언어 모델들이 더 나은 적응 능력을 보여줍니다.

- **Technical Details**: 이번 연구는 여러 사전 학습된 모델들을 대상으로 인간 대화 데이터의 발화 수준에서 감정 분석 교차 언어 전달 능력을 종합적으로 검토합니다. 우리는 대중 공개된 사전 학습 모델들을 소형 다국어 언어 모델(XLM-R, mT5)과 영어 중심의 대형 언어 모델(Llama-3, Mistral)으로 분류하였습니다. 또한, GPT-3.5 및 GPT-4와 같은 독점 LLM도 벤치마킹에 포함하였습니다. 우리는 고유의 감정 데이터셋을 사용하여 사전 학습 과정에서의 데이터 오염을 방지하고, 감정 분석을 영어에서 스페인어, 프랑스어, 중국어로 교차 언어 분석하였습니다.

- **Performance Highlights**: 우리의 평가 결과, 동일한 지도 학습을 적용했을 때 SMLMs는 훨씬 적은 모델 파라미터로도 zero-shot 교차 언어 전달 능력이 뛰어남을 보여주었습니다. 반면에, 대중 LLM은 few-shot 교차 언어 상황에서 빠른 개선을 보였으며, 타겟 언어의 추가 샘플이 제공될 때 SMLMs를 능가할 수 있습니다. 특히, 독점 모델인 GPT-3.5 및 GPT-4가 zero-shot 교차 언어 감정 분석에서 가장 뛰어난 성능을 보였으나, few-shot 교차 언어 학습 결과 몇몇 공개된 사전 학습 모델들이 GPT-3.5 및 GPT-4를 능가할 수 있음을 확인하였습니다.



### DiVERT: Distractor Generation with Variational Errors Represented as Text for Math Multiple-choice Questions (https://arxiv.org/abs/2406.19356)
- **What's New**: 이번 논문에서는 수학 다지 선택 문제(Math MCQs)에서 오답 선택지(Distractors)의 고유한 오류를 대표하는 텍스트로 변환하는 새로운 변이적 접근법, DiVERT(Distractor Generation with Variational Errors Represented as Text)를 소개합니다. 이를 통해 수학 MCQs의 오류를 해석 가능하게 표현하고, 오답 선택지를 생성하는 데 성공했습니다.

- **Technical Details**: DiVERT는 Variational Autoencoder(VAE)를 활용하여 수학 MCQs의 오류를 텍스트 토큰으로 표현하고, 이를 기반으로 오답 선택지를 생성합니다. 베이스로 70억 개의 파라미터를 가진 오픈 소스 LLM을 사용하여, 오류 표현과 오답 선택지 생성을 동시에 학습합니다. 이 과정에서 구성요소는 질문 줄기(question stem), 키(key), 설명(optional), 주제 태그(optional), 오답 선택지(distractors)로 나누어집니다.

- **Performance Highlights**: DiVERT는 1,434개의 수학 MCQ 데이터셋을 대상으로 실험한 결과, GPT-4에 기반한 최신 접근법보다 오답 선택지 생성에 있어 우수한 성능을 보였습니다. 또한, 수학 교육자와의 인간 평가를 통해 DiVERT가 생성한 오류 설명이 사람 작성 수준과 비슷하며, GPT-4 기반 방법보다도 높은 품질을 보였음을 확인했습니다.



### Fundamental Problems With Model Editing: How Should Rational Belief Revision Work in LLMs? (https://arxiv.org/abs/2406.19354)
Comments:
          23 pages, 4 figures

- **What's New**: 모델 편집(model editing) 문제는 언어 모델(language model)이 시간이 지남에 따라 새로운 세계 지식을 학습하는 방법을 다룹니다. 이 논문은 기존 모델 편집 문제의 표준 공식화를 비판하고, 모델 편집 연구를 위한 형식적인 테스트베드를 제안합니다. 논문은 모델 편집 문제 정의, 벤치마크 개발, LLM(대규모 언어 모델)의 편집 가능한 신념(활용가능한 beliefs)을 가정하는 문제 등 12가지 주요 문제를 다룹니다.

- **Technical Details**: 이 논문은 모델 편집에 대한 포괄적인 비평을 제공하며, 모델 편집 문제의 개념적 도전, 벤치마크 개발의 어려움, 그리고 LLM이 세계에 대한 신념을 가지고 있다는 가정 하에 발생하는 문제에 초점을 맞춥니다. 논문은 또한 Wikidata 기반의 반합성 데이터셋(semi-synthetic dataset)을 소개하여, 이상적인 베이지안 에이전트(Bayesian agent)의 레이블을 기반으로 편집을 평가하는 방법을 제안합니다. 이런 방식으로 언어 모델의 신념 개정이 원하는 표준에 얼마나 미치지 못하는지 정확히 측정할 수 있습니다.

- **Performance Highlights**: 논문에서 수행한 실험에서는 특정 설정에서 사전 학습된 모델이 일관성 없는 신념을 보이며 편집 후에도 일반화가 잘 되지 않는다는 결과를 보여줍니다. 언어 모델의 확률은 베이지안 후향 확률(Bayesian posteriors)과 지속적으로 차이가 나며 이를 통해 향후 모델 편집 방법에 대한 형식적인 연구가 필요하다는 것을 확인할 수 있습니다. 이는 모델 편집의 방법들이 제대로 일반화되지 않음을 정확하게 정량화한 결과입니다.



### IndoToxic2024: A Demographically-Enriched Dataset of Hate Speech and Toxicity Types for Indonesian Languag (https://arxiv.org/abs/2406.19349)
- **What's New**: 이번 연구는 인도네시아의 혐오 발언과 독성을 포괄적으로 분류하는 IndoToxic2024 데이터셋을 소개합니다. 이 데이터셋은 약 43,692개의 항목을 포함하고 있으며, 19명의 다양한 인구 통계적 배경을 가진 주석자가 이를 주석 처리했습니다. 이 데이터셋은 특히 취약한 그룹을 타깃으로 선정하여 주석되었으며, 주로 인도네시아 대통령 선거 기간 동안 수집되었습니다.

- **Technical Details**: IndoToxic2024 데이터셋은 인도네시아의 다양한 소셜 미디어 플랫폼에서 수집된 게시물들로 구성되어 있습니다. 수집된 데이터는 키워드를 사용하여 취약한 그룹에 대한 혐오 발언을 탐지하였으며, 각 항목은 인구 통계 정보를 포함한 10차원 데이터로 구성되어 있습니다. 또한, IndoBERTweet 모델을 혐오 발언 분류에 맞게 파인튜닝하고, gpt-3.5-turbo 모델 성능을 향상시키기 위해 인구 통계 정보를 통합하는 방법을 평가했습니다.

- **Performance Highlights**: IndoBERTweet 모델은 7개의 이진 분류 작업에 대해 매크로-F1 점수 0.78을 달성했습니다. 또한, gpt-3.5-turbo 모델의 제로샷 성능을 인구 통계 정보를 통합하여 개선할 수 있음을 보여주었습니다. 그러나 지나친 인구 통계 정보의 통합은 데이터 단편화로 인해 파인튜닝된 모델의 성능에 부정적인 영향을 줄 수 있음을 경고합니다.



### LiveBench: A Challenging, Contamination-Free LLM Benchmark (https://arxiv.org/abs/2406.19314)
- **What's New**: LLM(문자열 언어 모델, Language Learning Model) 평가에서 발생하는 공정성 문제를 해결하기 위해, 새로운 벤치마크 'LiveBench'를 소개했습니다. LiveBench는 테스트 셋 오염(test set contamination)과 LLM 및 인간 크라우드소싱의 편향을 방지하는 것을 목표로 합니다.

- **Technical Details**: LiveBench는 (1) 최근 정보 소스에서 자주 업데이트되는 질문을 포함하고, (2) 객관적인 정답값에 따라 자동으로 점수를 부여하며, (3) 수학, 코딩, 추론, 언어, 명령 수행, 데이터 분석 등 다양한 도전 과제를 포함하고 있습니다. 질문은 최신 수학 대회, arXiv 논문, 뉴스 기사 및 데이터셋 기반으로 구성되었습니다. 또, Big-Bench Hard, AMPS, IFEval 등의 이전 벤치마크에서 오염되지 않은 더 어려운 버전의 과제도 포함되어 있습니다.

- **Performance Highlights**: 많은 대표적인 폐쇄형 모델과 0.5B에서 110B 크기의 수십 가지 오픈 소스 모델을 평가한 결과, 상위 모델의 정확도는 65% 미만으로 나타났습니다. 공개된 모든 질문, 코드, 모델 답변을 포함하며, 질문은 매월 추가 및 업데이트됩니다. 향후, 더 어려운 과제들이 추가될 예정입니다. LLM의 능력을 미래에 걸쳐 구별할 수 있는 기반을 제공합니다. 커뮤니티의 참여와 협력을 환영합니다.



### The Odyssey of Commonsense Causality: From Foundational Benchmarks to Cutting-Edge Reasoning (https://arxiv.org/abs/2406.19307)
Comments:
          42 pages

- **What's New**: 이 논문은 상식적 인과관계(commonsense causality)에 대한 체계적 탐구가 부족한 현 상태를 개선하기 위해 작성되었습니다. 이 논문은 상식적 인과관계 이해가 인간 지능의 독특한 표지임을 강조하며, 법적 책임, 의사결정 등 다양한 분야에서 인과관계를 판단하는 데 필수적이라는 점을 지적합니다. 200개 이상의 대표 논문을 종합하여 상식적 인과관계에 관한 최신 발전 사항을 업데이트하고, 초보자들을 위한 실용적인 가이드를 제공하며, 이 중요한 분야에서 유망한 미래 연구 방향을 강조합니다.

- **Technical Details**: 이 논문은 상식적 인과관계를 분류(taxonomies), 벤치마크(benchmarks), 획득 방법(acquisition methods), 정성적(reasonable) 추론 및 정량적(quantitative) 측정에 중점을 두고 탐구합니다. 이러한 접근 방식을 통해 항목별로 체계적인 개요를 제공하고 있으며, 연구자들을 위한 구체적인 가이드를 제시합니다.

- **Performance Highlights**: 이 논문은 상식적 인과관계 연구의 최신 습득 및 평가 방법들을 총망라하고 있으며, 법적 책임 판단 등 실무적인 영역에서 이 기술이 어떻게 적용될 수 있는지 강조합니다. 이는 초보 연구자들이 이 분야를 빠르게 파악하고, 기존 연구자들이 최신 트렌드 및 방법을 확인할 수 있는 중요한 자원으로 작용할 것입니다.



### VERISCORE: Evaluating the factuality of verifiable claims in long-form text generation (https://arxiv.org/abs/2406.19276)
- **What's New**: VERISCORE는 검증 가능한(claims)과 비검증 가능한(content)을 포함한 다양한 장문의 생성 작업을 평가할 수 있는 새로운 메트릭입니다. 이는 기존 방법론인 FACTSCORE와 SAFE가 모든 주장이 검증 가능해야 한다는 가정에 문제가 있다는 점을 해결합니다. VERISCORE는 폐쇄형(closed) 또는 fine-tuned 오픈 가중치(open-weight) 언어 모델로 효과적으로 구현될 수 있으며, 인간 평가에서는 다른 방법들보다 더 타당한 주장을 추출하는 것으로 나타났습니다.

- **Technical Details**: VERISCORE 파이프라인은 주장 추출, 증거 검색(evidence retrieval), 주장 검증(claim verification), 점수 계산(score calculation)으로 구성됩니다. 첫 단계는 문장을 독립적인 사실들로 분해하여 검증 가능한 주장만을 추출하는 것입니다. 기존의 FActScore는 이러한 주장을 생물학 자료에만 최적화된 방식으로 추출하여 그 외 다른 도메인에 적용할 수 없었습니다. Safe는 이를 보완했지만 여러 단점과 높은 처리 비용이 문제가 되었습니다. VERISCORE는 이러한 문제를 극복하기 위해 검증 가능한 주장만을 추출하고 문맥 간의 상호 연관성을 고려합니다.

- **Performance Highlights**: VERISCORE를 사용해 16개의 다양한 모델을 여러 장문의 생성 작업에 대해 평가한 결과, GPT-4o 모델이 전체적으로 가장 높은 사실성을 지닌 텍스트를 생성했습니다. 또한, 더 사실 밀도가 높은(biography generation) 작업과 더 복잡한(long-form QA) 작업들의 검증 결과들이 서로 일치하지 않음을 발견하여, 다양한 작업 별로 사실성 평가가 필요하다는 점을 강조했습니다.



### AutoPureData: Automated Filtering of Web Data for LLM Fine-tuning (https://arxiv.org/abs/2406.19271)
Comments:
          Initial version

- **What's New**: 이 논문은 웹 데이터를 수집하고, 신뢰할 수 있는 AI 모델을 활용해 불필요한 텍스트를 자동으로 필터링하는 시스템을 제안합니다. 이를 통해 최신 데이터를 지속적으로 반영하는 라지 랭귀지 모델(Large Language Models, LLMs)을 유지할 수 있습니다.

- **Technical Details**: 제안된 시스템은 웹 데이터를 수집하여 정제된 웹 데이터 집합인 FineWeb을 사용합니다. LlamaGuard 2와 Llama 3 (8B) 모델을 이용하여 폭력, 성범죄, 혐오 발언 등의 부적절한 내용을 자동으로 필터링합니다. 이를 통해 데이터의 안전성과 신뢰성을 유지합니다.

- **Performance Highlights**: 실험 결과, 100개의 웹 데이터 샘플 중 32개 행이 부적절한 내용으로 플래그가 지정되었습니다. LlamaGuard 2 모델은 F-1 점수 91.5%와 4%의 False Positive Rate로 다른 중재 모델들을 능가합니다. 이 시스템은 데이터 필터링 과정의 자동화를 통해 시간과 비용을 절감할 수 있는 가능성을 제시합니다.



### Read Anywhere Pointed: Layout-aware GUI Screen Reading with Tree-of-Lens Grounding (https://arxiv.org/abs/2406.19263)
- **What's New**: 새로운 연구는 사용자 지정 포인트를 기반으로 스크린 내용을 읽고 해석하는 Screen Point-and-Read (SPR) 작업을 소개합니다. 이 작업은 현재 사용되는 스크린 리딩 도구들이 강직함과 사용자 경험의 제한성을 극복하기 위해 설계되었습니다. 이 작업을 해결하기 위해 Tree-of-Lens (ToL) 에이전트를 제안하며, 이는 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 발전을 바탕으로 구축되었습니다.

- **Technical Details**: ToL 에이전트는 새롭게 제안된 ToL 기반 메커니즘을 활용하여 입력된 좌표와 해당 GUI 스크린샷을 바탕으로 계층적 레이아웃 트리(Hierarchical Layout Tree)를 구축합니다. 이 트리는 화면의 기본 구조를 나타내며, 트리의 노드는 다양한 규모의 영역을 나타냅니다. Android 스크린 계층적 레이아웃(ASHL) 데이터셋에서 50,000개의 계층적 화면 영역을 포함하는 바운딩 박스를 사용하여 훈련된 객체 탐지 모델을 통해 이 트리를 자동으로 구축합니다.

- **Performance Highlights**: ToL 에이전트는 새롭게 제안된 Screen Point-and-Read (ScreenPR) 벤치마크에서 다른 기초 모델들과 비교하여 우수한 성능을 보여주었습니다. 이 벤치마크는 웹, 모바일 및 운영체제 GUI의 650개의 스크린샷으로 구성되어 있으며, 1,500개의 목표 지점과 영역으로 주석이 달려 있습니다. ToL 에이전트는 콘텐츠와 레이아웃 설명 정확성에서 각각 15%와 30% 이상의 향상을 기록하였습니다. 이 외에도 모바일 GUI 내비게이션 작업에서도 잘못된 작업을 식별하는 유용성을 입증하였습니다.



### AutoRAG-HP: Automatic Online Hyper-Parameter Tuning for Retrieval-Augmented Generation (https://arxiv.org/abs/2406.19251)
- **What's New**: 최신 대형 언어 모델(Large Language Models, LLMs)의 발전으로 인해 ML/AI 개발이 크게 변화했으며, 이를 위해 Retrieval-Augmented Generation (RAG) 시스템에 대한 AutoML 원칙을 재검토할 필요가 생겼습니다. Hyper-parameter 최적화 및 온라인 적응에 대한 도전 과제를 해결하기 위해, 우리는 AutoRAG-HP 프레임워크를 제안합니다. 이 프레임워크는 hyper-parameter 튜닝을 온라인 멀티암드 밴딧(Multi-Armed Bandit, MAB) 문제로 형성하고, 두 수준의 계층적 MAB(Hier-MAB) 방식을 도입하여 효율적으로 큰 검색 공간을 탐색합니다.

- **Technical Details**: AutoRAG-HP 프레임워크는 RAG 시스템의 hyper-parameter를 온라인으로 최적화하기 위해 MAB 기반의 방법을 사용합니다. 여기서 상위 수준의 MAB가 각 모듈의 최적 설정을 탐색하는 저수준 MAB들을 안내합니다. 이를 위해 상위 MAB와 하위 MAB의 두 레벨의 계층적 구조를 도입했습니다. 이 방법은 홍에 첨가되는 도큐먼트 수(top-k), 프롬프트 압축 비율(prompt compression ratio), 임베딩 방법(embedding methods) 등의 여러 hyper-parameter를 동시에 튜닝할 때 큰 검색 공간을 효율적으로 탐색할 수 있습니다.

- **Performance Highlights**: 우리의 평가 결과, MAB 기반의 온라인 학습 방법은 검색 공간에서 두드러진 기울기가 있는 시나리오에서 Recall@5가 약 0.8에 도달하는 반면, 그리드 서치(Grid Search) 접근법에서 요구되는 LLM API 콜의 약 20%만을 사용합니다. 제안하는 Hier-MAB 접근법은 더 도전적인 최적화 시나리오에서 다른 기준선들을 능가했습니다.

- **Link**: 코드는 곧 제공될 예정입니다.



### Revealing Fine-Grained Values and Opinions in Large Language Models (https://arxiv.org/abs/2406.19238)
Comments:
          28 pages, 20 figures, 7 tables

- **What's New**: 이 연구는 정치적으로 민감한 주제에 대해 독립적으로 설계된 설문조사를 통해 대형 언어 모델(LLM)의 잠재적인 편향과 의견을 식별합니다. 편향을 줄이고 잠재적인 해악을 완화하는 데 도움이 되는 방법론을 제안합니다. 연구는 6가지 LLM을 사용하여 62개의 정치적 나침반 테스트(PCT) 명제에 대한 156,240개의 응답을 분석합니다.

- **Technical Details**: 연구는 420가지 프롬프트 변형을 사용하여 PCT 명제에 대해 LLM의 응답을 생성했습니다. 프롬프트 변형은 연령, 성별, 국적, 정치적 성향 및 계층을 포함한 다양한 인구 통계적 페르소나를 포함합니다. 코스 그레인 분석과 파인 그레인 분석 모두 수행되었으며, 후자의 경우 반복적이고 일관된 문구인 '트로프(tropes)'를 식별하여 각 프롬프트와 모델의 응답 패턴을 분석했습니다.

- **Performance Highlights**: 연구 결과, 프롬프트에 추가된 인구 통계적 특성이 PCT 결과에 크게 영향을 미치며 이는 편향을 반영하고, 폐쇄형 응답과 개방형 응답 사이의 차이를 줄이거나 확대할 수 있음을 발견했습니다. 또한, LLM이 다양한 프롬프트와 모델에서 비슷한 정당성을 반복적으로 생성하는 경향이 있음을 보여주었습니다.



### FlowVQA: Mapping Multimodal Logic in Visual Question Answering with Flowcharts (https://arxiv.org/abs/2406.19237)
- **What's New**: FlowVQA는 최신 시각 질문 응답(multimodal language models) 평가 벤치마크로, 복잡한 흐름도(flowcharts)를 기반으로 한 논리적 추론 능력을 평가합니다. 이는 시각적 근거와 복잡성을 더욱 강화하여, 정보 위치 확인, 의사 결정, 논리적 진행 등 다양한 추론 과제를 평가합니다.

- **Technical Details**: FlowVQA는 총 2,272개의 흐름도 이미지를 포함하며, WikiHow와 Instructables 같은 소스에서 수집된 자료를 바탕으로 생성되었습니다. 또한, 22,413개의 다양한 질문-답변 쌍이 포함되어 있으며, 정보 위치 확인, 사실 찾기, 시나리오 유추, 흐름 추론 및 위상 이해와 같은 여러 추론 기술을 테스트합니다. 데이터 생성 과정은 다단계 기계 생성 및 인간 검증을 포함하며, 이는 복잡하지만 체계적인 흐름도 제공을 목적으로 합니다.

- **Performance Highlights**: FlowVQA를 사용한 여러 개방형 및 독점형 VLMs의 벤치마크 평가 결과, 모델들이 FlowVQA 데이터셋에서 시각적 및 공간적 추론 작업을 수행하는 데 어려움을 겪는 것을 확인했습니다. 또한 방향성 편향 및 다양한 길이의 흐름도에 대한 불균형적인 성능 패턴이 드러났습니다. 이로써 FlowVQA는 멀티모달 모델링 분야를 발전시킬 중요한 도구로 자리 잡을 가능성을 보여줍니다.



### RuBLiMP: Russian Benchmark of Linguistic Minimal Pairs (https://arxiv.org/abs/2406.19232)
- **What's New**: RuBLiMP (Russian Benchmark of Linguistic Minimal Pairs)는 러시아어에 대한 최초의 다양하고 대규모 최소한의 쌍(문법적으로 다른 두 문장)의 벤치마크를 도입하였습니다. 45,000개의 문장 쌍이 포함되어 있으며, 이는 형태론(morphological), 문법(syntactic), 의의론(semantic) 현상을 잘 대표하고 있습니다.

- **Technical Details**: RuBLiMP는 오픈 텍스트 코퍼스에서 자동으로 주석이 달린 문장을 선택하고, 언어학적 교란을 적용하여 문장 쌍을 생성합니다. 각 문장 쌍은 문법적 현상을 명확히 격리시키며, 25개의 언어 모델을 평가하기 위해 테스트 데이터로 사용될 수 있습니다. 문장 데이터는 Wikipedia, Wikinews, Librusec에서 추출되며, 필터링, 주석 달기, 문장 교란 절차를 통해 생성됩니다.

- **Performance Highlights**: 25개의 러시아어 언어 모델을 평가한 결과, 형태론적 대조와 일치라는 관점에서 민감하게 반응하지만, 구조적 관계, 부정, 전이성, 시제를 이해하는 현상에서 인간에 비해 성능이 떨어지는 것으로 나타났습니다. 이 결과는 러시아어 언어 모델의 문법적 능력을 심도 있게 분석할 수 있는 중요한 자료로 활용될 수 있습니다.



### Tools Fail: Detecting Silent Errors in Faulty Tools (https://arxiv.org/abs/2406.19228)
Comments:
          18 pages, 12 figures

- **What's New**: 이 논문에서는 도구 사용 관련 오류를 감지하고 계획을 반영하는 모델의 능력을 탐구하는 새로운 프레임워크를 소개합니다. 주목할 점은 도구를 선택하는 것보다 침묵(silent) 도구 오류를 감지하고 대응하는 것에 초점을 맞추고 있다는 것입니다.

- **Technical Details**: 논문에서는 계산기와 함께 제어된 테스트에서 실패 복구 접근 방식을 설명하고 있습니다. 또한, 다양한 모달리티의 도구 및 실제 도구 오류 시나리오를 분류하고 있습니다. 행동 플래너와 객체 감지기 평가를 포함한 여러 그림과 표로 데이터 분포 및 모델 성능을 시각적으로 제공하고 있습니다. 특히, Chain-of-Thought 프롬프팅이 산술적 성능을 향상시키며, 이는 few-shot in-context 예제로 더욱 향상됩니다.

- **Performance Highlights**: 정확한 도구 사용이 가장 강력한 결과를 낳으며, 이는 신뢰할 수 있는 도구를 사용하는 기존 문헌을 지지합니다. 모델이 로봇의 상태를 유추하고 성공/실패 사례를 바탕으로 도구의 성공 확률을 평가할 수 있음을 보여줍니다. 그러나 모델은 도구의 실수 여부는 잘 이해하지만, 작업에 치명적인 오류와 견딜 수 있는 오류를 구분하는 데 어려움을 겪고 있습니다.



### Aligning Teacher with Student Preferences for Tailored Training Data Generation (https://arxiv.org/abs/2406.19227)
- **What's New**: 새로운 프레임워크인 ARTE(Aligning TeacheR with StudenT PreferencEs)가 제안되었습니다. ARTE는 교사 모델을 학생 모델의 선호도에 맞추어 맞춤형 학습 예시를 생성함으로써 Knowledge Distillation(지식 증류)을 용이하게 합니다. 이는 특히 학생 모델이 특정 작업을 더 잘 수행할 수 있도록 고안되었습니다.

- **Technical Details**: ARTE 프레임워크는 세 단계로 구성됩니다. 1) Knowledge Elicitation(지식 추출): 교사 모델에서 시드 질문을 통해 초안 예시 생성. 2) Preference Collection(선호도 수집): 학생 모델의 선호도를 반영한 일회성 'in-context learning'으로 선호도 수집. 3) Preference Alignment(선호도 정렬): Direct Preference Optimization (DPO)을 사용하여 교사 모델을 학생 모델의 선호도와 정렬. 이후 정렬된 교사 모델로 맞춤형 학습 예시를 생성하여 학생 모델을 감독 학습으로 미세 조정합니다.

- **Performance Highlights**: ARTE는 기존의 instruction-tuning 데이터셋 대비 논리적 추론, 상식적 추론, 수학적 추론, 및 지식 추론 작업에서 각각 9.6%, 1.0%, 0.8%, 8.5%의 성능 향상을 보였습니다. 또한 ARTE의 일반화 성능을 다양한 도메인에서 검증한 결과, 학생 모델의 추론 능력을 향상시키고, 다른 작업과 학생 모델에도 맞춤형 학습 예시를 생성할 수 있음을 확인하였습니다.



### Simulating Classroom Education with LLM-Empowered Agents (https://arxiv.org/abs/2406.19226)
- **What's New**: 본 연구에서는 SimClass라는 다중 에이전트 교실 시뮬레이션 프레임워크를 제안합니다. 이는 실제 사용자가 참여하는 교실 환경을 모사하며, 대표적인 교실 역할을 인식하고 자동화된 교실 교육을 위한 새로운 제어 메커니즘을 도입합니다. 두 개의 실제 강의를 바탕으로 사용자 실험이 수행되었습니다.

- **Technical Details**: SimClass는 Flanders Interactive Analysis System과 Community of Inquiry 이론적 틀을 활용하여 교실 상호작용 패턴을 효과적으로 모사합니다. 시스템은 다양한 교실 역할을 인식하고 기능적 워크플로우를 가지고 설계된 수업 제어 메커니즘을 통합하고 있습니다. 두 개의 서로 다른 강좌에 대해 준비된 슬라이드와 교육 스크립트를 바탕으로 48명의 학생이 초대되어 시스템과 상호작용하며 실험이 진행되었습니다.

- **Performance Highlights**: 실험 결과에 따르면 SimClass는 전통적인 교실과 유사한 행동, 상호작용 패턴 및 특성을 보여주었으며, 여러 에이전트가 사용자를 더 효과적으로 참여시켜 교육 효과를 높였습니다. 또한 협업 교수, 토론, 감정적 동반 및 규율 관리와 같은 자발적 행위가 나타났습니다. 이를 통해 LLM 기반 다중 에이전트 시스템이 실제 교육 환경을 모사하는 잠재력을 가졌음을 확인할 수 있었습니다.



### T-FREE: Tokenizer-Free Generative LLMs via Sparse Representations for Memory-Efficient Embeddings (https://arxiv.org/abs/2406.19223)
- **What's New**: 최신 연구에서, 기존의 토크나이저(tokenizer) 구조의 약점을 보완하고 보다 효율적인 방법을 제안합니다. T-FREE는 문자 삼중자(character triplets)에 대한 희소 활성화 패턴을 통해 단어를 직접 임베딩(embedding)하고, 참조 코퍼스(reference corpus) 없이 작동하는 방식입니다. 이 방법은 형태적 유사성을 잘 활용하며, 임베딩 레이어를 강력하게 압축할 수 있는 특성을 가지고 있습니다.

- **Technical Details**: 전통적인 토크나이저들은 입력 텍스트를 서브워드(subwords) 단위로 분리하고 이를 정수 표현으로 변환하여 처리합니다. 이러한 과정을 통해 만들어진 고정된 크기의 어휘(vocabulary)는 통계적 빈도에 기반하여 구성됩니다. T-FREE는 토큰화를 완전히 제거하고, 입력 텍스트의 각 단어를 문자 삼중자 위에 희소 활성화 패턴으로 직접 임베딩합니다. 이를 통해 서브워드 토큰의 필요성을 제거하고 다양한 언어에서 최적의 성능을 유지할 수 있습니다.

- **Performance Highlights**: T-FREE는 임베딩 레이어의 크기를 85% 이상 감소시키면서도 경쟁력 있는 다운스트림 성능을 달성합니다. 특히, 교차 언어 전이 학습(cross-lingual transfer learning)에서 상당한 성능 개선을 보여줍니다. 실험 결과, 텍스트 인코딩 길이를 평균 56% 감소시킬 수 있었으며, 전통적인 토크나이저 기반 모델과 비교하여 매우 경쟁력 있는 성능을 발휘합니다.



### SeaKR: Self-aware Knowledge Retrieval for Adaptive Retrieval Augmented Generation (https://arxiv.org/abs/2406.19215)
- **What's New**: 이번 논문에서는 Self-aware Knowledge Retrieval (SeaKR)라는 새로운 적응형 RAG 모델을 소개합니다. SeaKR는 LLM의 내부 상태에서 자기인식적인 불확실성을 추출하여, 높은 불확실성이 나타날 때만 지식을 검색합니다. 이는 LLM이 스스로 지식의 필요성을 결정하고, 검색된 지식을 효과적으로 통합할 수 있도록 돕습니다.

- **Technical Details**: SeaKR는 피드-포워드 네트워크(FFN)에서 각 레이어에 대응하는 마지막 생성된 토큰의 내부 상태에서 LLM의 자기인식 불확실성을 추출합니다. 이 불확실성 점수를 바탕으로 검색 여부를 결정하고, 검색된 지식을 통합합니다. SeaKR는 두 가지 적응형 통합 전략을 설계했는데, 첫 번째는 '자기인식 재순위 매기기'로 LLM이 다중으로 검색된 스니펫 중 가장 불확실성을 줄이는 스니펫을 선택합니다. 두 번째는 '자기인식 추론'으로, 복잡한 질문을 해결하기 위해 여러 번의 지식 검색 후 가장 불확실성을 줄이는 전략을 선택합니다.

- **Performance Highlights**: 복잡하고 간단한 질문 응답(QA) 데이터셋에서 실험 결과, SeaKR는 기존의 적응형 RAG 방법들을 능가했습니다. 복잡한 QA 벤치마크에서 특히 좋은 성과를 보여주며, 동적 지식 통합이 자기인식적 검색보다 더 큰 성과 향상을 가져온다는 것을 보여줍니다.



### Annotation Errors and NER: A Study with OntoNotes 5.0 (https://arxiv.org/abs/2406.19172)
Comments:
          Unpublished report. Originally submitted to LREC 2022

- **What's New**: 이 연구는 영어 NER (Named Entity Recognition) 대규모 코퍼스인 OntoNotes 5.0의 주석 오류를 탐지하고 수정하는 방법을 제안하고 있습니다. 기존 연구들은 주로 새로운 NER 모델 개발에 치중한 반면, 이 연구는 데이터셋 자체의 질에 주목하고 있습니다. 저자들은 10%에 해당하는 문장의 주석 오류를 발견하고 수정하였으며, 주석된 개체(span)와 타입의 약 8%를 수정하였습니다.

- **Technical Details**: 이 연구에서는 주로 다음과 같은 네 가지 범주로 주석 오류를 분류하였습니다: 가이드라인에서의 일탈, 불일치된 주석, 잘못된 주석, 애매한 주석. 데이터셋의 수정을 위해 세 가지 간단한 방법을 사용했으며, 주석 수정을 수동으로 수행했습니다. 또한, 세 가지 NER 라이브러리를 사용하여 원래 데이터셋과 수정된 데이터셋에서 훈련된 모델을 비교 평가하였습니다.

- **Performance Highlights**: 수정된 데이터셋을 사용한 NER 모델의 평균 F-score가 약 1.23% 향상되었으며, 특정 엔티티 유형에서는 10% 이상의 큰 향상을 보였습니다. 이는 데이터셋 크기를 고려했을 때 매우 큰 변화입니다. 또한, 이 방법은 주로 언어에 관계없이 다른 NER 데이터셋이나 시퀀스 레이블링 작업에 적용될 수 있습니다.



### The Illusion of Competence: Evaluating the Effect of Explanations on Users' Mental Models of Visual Question Answering Systems (https://arxiv.org/abs/2406.19170)
Comments:
          16 pages (including Appendix); under review

- **What's New**: 시각 질문 답변과 설명(VQA/X) 영역에서 AI 시스템의 제한사항을 사용자가 어떻게 인식하는지 그리고 답변에 설명을 제공하는 것이 시스템의 능력과 제한사항에 대한 적절한 정신적 모델을 구축하는데 도움이 되는지 조사하였습니다. 시스템에 컬러 이미지와 그레이스케일 이미지를 입력하여 시스템의 시각적 입력 제한을 조작하였습니다. 실험 결과, 설명은 사용자가 시스템의 제한을 더 잘 이해하도록 돕는 대신 시스템의 능력을 과장되게 인식하도록 만들었습니다.

- **Technical Details**: 실험은 시각적 질문 답변 시스템(VQA/X)을 이용하여 수행되었으며, 컬러와 그레이스케일 이미지를 입력으로 사용하여 AI 시스템의 능력을 인위적으로 제한하였습니다. 참가자는 컬러 이미지를 보고 질문, 시스템의 답변, 설명을 평가하였습니다. 사용자가 시스템의 색상 인식 능력을 포함해 여러 능력을 판단하도록 설문지를 설계하였습니다. 이 연구는 CLEVR-X 벤치마크와 VQA-X 벤치마크의 항목을 사용하여 수행되었습니다.

- **Performance Highlights**: 결과적으로, 설명이 제공된 경우 사용자가 AI 시스템의 실제 성능과 관계없이 그 시스템의 능력을 높이 평가하는 경향이 나타났습니다. 즉, 설명이 시스템의 제한을 더 잘 이해하게 만드는 것이 아니라, 오히려 시스템에 대한 맹목적인 신뢰를 증가시키는 결과가 나타났습니다.



### CHEW: A Dataset of CHanging Events in Wikipedia (https://arxiv.org/abs/2406.19116)
Comments:
          Short Paper

- **What's New**: CHEW(CHEW)은 Wikipedia의 변화 이벤트를 담은 새로운 데이터셋입니다. 연구진은 CHEW를 사용하여 LLMs(Large Language Models)의 타임라인 이해도(time understanding)를 테스트했으며, 그 결과 이 모델들이 정확한 타임라인을 생성하는 데 어려움을 겪고 있음을 발견했습니다. 또한, CHEW에서 추출한 임베딩(embeddings)이 의미 변화(meaning shift)를 식별하는 데 유용하다는 것을 보여주었습니다.

- **Technical Details**: CHEW는 Wikipedia의 이벤트와 엔티티(entity)의 중요한 변화를 찾아내는 데 초점을 맞춘 감시형 데이터셋입니다. 이는 TAQA(TemporalQA) 데이터셋에서 추출한 Wikipedia 이벤트와 엔티티 목록을 기반으로 합니다. 연구진은 이러한 변화를 식별하기 위해 TAQA 데이터셋에서 시간에 따라 달라지는 질문-답변 쌍을 활용했습니다. SBERT(Sentence-BERT)를 사용하여 각 시점의 Wikipedia 문서 개정을 비교하고 코사인 유사도를 계산했습니다.

- **Performance Highlights**: CHEW를 사용한 생성 및 분류 실험에서 LLMs가 시간 정보를 처리하는 능력을 다각적으로 평가했으며, 이러한 모델들이 시간 정보에 맞춰지는 잠재력이 있음을 확인했습니다. 하지만 여전히 정확한 타임라인을 구성하는 데는 어려움이 있으며, 이는 LLMs의 개선 방향으로 중요한 힌트를 제공합니다.



### Statements: Universal Information Extraction from Tables with Large Language Models for ESG KPIs (https://arxiv.org/abs/2406.19102)
Comments:
          Accepted at the NLP4Climate workshop in the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024)

- **What's New**: 이 논문에서는 환경, 사회, 지배구조(ESG) 보고서 내의 표에서 정량적 사실과 관련 정보를 추출하기 위한 새로운 도메인 비특정 데이터 구조인 'Statement'를 제안했습니다. 이를 통해 테이블을 문장으로 번역하는 새로운 감독된 딥 러닝 정보 추출 작업을 소개합니다. 또한, 100K 이상 주석된 표들로 이루어진 'SemTabNet' 데이터셋을 도입하고, T5 기반 모델 중 최고 성능 모델이 82% 유사성을 달성했다고 보고했습니다.

- **Technical Details**: Statement 데이터 구조는 복잡하고 불규칙하며 이질적인 정보를 균질한 방식으로 표현할 수 있는 트리 구조입니다. 이 구조는 하나의 내용에서 여러 (명명된) 저장소와 (다항식) 관계를 결합할 수 있습니다. 이 데이터 구조를 사용하여 ESG 보고서에서 정보를 추출하는 'statement extraction' 작업을 수행하고, 이를 평가하기 위해 Tree Edit Distance 기반의 Tree Similarity Score(TSS)를 제안했습니다.

- **Performance Highlights**: 논문의 T5 기반 모델은 baseline (21%)에 비해 높은 82% 유사성을 달성했습니다. 이 모델은 ESG 보고서의 2700개 이상의 표에 적용되어 유의미한 개선을 나타냈습니다.



### Fairness and Bias in Multimodal AI: A Survey (https://arxiv.org/abs/2406.19097)
Comments:
          8 pages

- **What's New**: 이 논문은 최근 인공지능(AI) 시스템에서 공정성과 편향 문제를 다루는 것의 중요성을 강조하면서, 대규모 다중 모드 모델(Large Multimodal Models, LMMs)과 대규모 언어 모델(Large Language Models, LLMs)에 대한 비교 연구를 제시합니다. 기존 학술 문헌에서는 LMMs의 공정성과 편향에 대한 연구가 거의 없었으나, 이번 작업을 통해 50개의 데이터셋과 모델 예시를 포함하여 이에 영향을 미치는 문제들을 체계적으로 탐구했습니다.

- **Technical Details**: 이 연구는 구글 학술 검색을 통해 'Fairness and Bias in Large Multimodal Models'와 'Fairness and Bias in Large Language Models'라는 두 가지 검색어로 검색하여 총 33,400개 및 538,000개의 링크를 분석했습니다. 이 논문은 경쟁 데이터셋과 모델들을 통해 공정성과 편향을 입증하는 사례를 제공하고, 새롭게 '사전 사용(preuse)'이라는 편향의 정량화 카테고리를 제안했습니다. 이는 기존의 고유(intrinsic) 및 외재(extrinsic) 편향과 추가된 새로운 분류입니다.

- **Performance Highlights**: 논문에서 발견한 주요 내용 중 하나는 많은 LMMs와 LLMs가 성별, 인종 등에 대한 편향을 가지고 있다는 점입니다. 예를 들어, VQGAN-CLIP 모델과 Stable Diffusion 모델이 '17세 소녀'라는 무해한 프롬프트에 대해 73%의 비율로 성적 이미지 생성한다는 점을 제시했습니다. 이와 같은 편향 문제를 해결하기 위해 다양한 평가 방법 및 비편향 전략이 논의되었습니다.



### AMBROSIA: A Benchmark for Parsing Ambiguous Questions into Database Queries (https://arxiv.org/abs/2406.19073)
- **What's New**: 새로운 벤치마크로 AMBROSIA를 소개합니다. 이는 텍스트에서 SQL로 변환하는 파서(Parser)가 모호한 요청을 인식하고 해석하는 데 도움이 될 것입니다. 이 데이터셋은 다양한 유형의 모호함(스코프 모호성, 부착 모호성, 애매함)을 나타내는 질문과 그 해석, 그리고 이에 상응하는 SQL 쿼리를 포함하고 있습니다.

- **Technical Details**: AMBROSIA 데이터셋은 데이터베이스 컨텍스트가 제공되어도 여전히 모호성이 존재하는 질문들을 제시합니다. 이러한 모호함은 새로운 접근 방식을 통해, 즉 처음부터 데이터베이스를 통제된 방식으로 생성함으로써 달성됩니다. 데이터셋은 세 가지 유형의 모호성(스코프 모호성, 부착 모호성, 애매함)을 포함하고 있습니다. 우리는 다양한 대규모 언어 모델(LLM, Large Language Models)을 사용하여 AMBROSIA에 대한 성능을 벤치마킹했습니다.

- **Performance Highlights**: 벤치마킹 결과, 가장 첨단 모델들조차도 질문 내의 모호성을 식별하고 해석하는 데 어려움을 겪었음을 나타냈습니다.



### EmPO: Theory-Driven Dataset Construction for Empathetic Response Generation through Preference Optimization (https://arxiv.org/abs/2406.19071)
Comments:
          v01, 4 pages short paper, ACL style

- **What's New**: 이 논문에서는 대화형 인공지능(AI) 에이전트의 감정이입 반응 생성을 향상시키기 위해 새로운 접근 방식을 제안합니다. 기존의 대규모 언어 모델(LLM)이 가지는 일반화 성능을 유지하면서도 감정이입 품질을 강화하는 방법에 대한 도전을 해결합니다. 이를 위해 이론 기반의 선호 데이터셋을 구축하고 이를 선호 최적화 알고리즘(preference optimization algorithms)과 결합하여 LLM과 정렬(alignment)시키는 방식을 도입했습니다.

- **Technical Details**: 감정이입 반응 생성을 측정하기 위해 EmpatheticDialogues 데이터셋을 사용하며, diff-EPITOME와 BERTscore 지표로 감정이입을 평가합니다. 또한, 일반화 성능을 평가하기 위해 MMLU 기준을 사용합니다. 먼저, EmpatheticDialogues 데이터셋을 사용하여 선호 데이터셋을 구축하고, 이를 Direct Preference Optimization(DPO) 알고리즘으로 기본 LLM을 미세 조정(fine-tune)하여 선호 후보 응답과 정렬되도록 합니다. Zephyr-7B 모델을 사용하여 실험을 수행하며, 초매개변수 구성 공간을 탐색합니다.

- **Performance Highlights**: 제안된 방법은 모델을 감정이입 반응 생성에 맞춰서 훈련시키면서도 일반적인 언어 이해 성능에 미치는 영향을 최소화합니다. 또한, 학습된 모델은 기존의 감정이입 대화 시스템과 비교하여 더욱 향상된 감정이입 성능을 보여줍니다. 모든 데이터셋, 소스 코드 및 모델은 공개되어 연구와 적용에 활용할 수 있습니다.



### STBench: Assessing the Ability of Large Language Models in Spatio-Temporal Analysis (https://arxiv.org/abs/2406.19065)
- **What's New**: 이번 논문에서는 대규모 언어 모델(LLMs)의 시공간 데이터 이해 능력을 평가하기 위한 새로운 프레임워크인 STBench를 제안합니다. 이 프레임워크는 LLMs의 시공간 데이터 이해를 네 가지 차원(지식 이해, 시공간 추론, 정확한 계산, 다운스트림 응용)으로 나누어 평가합니다. 또한, 13개의 최신 LLMs(GPT-4o, Gemma 등)을 평가하기 위해 13개의 distinct tasks와 6만 개 이상의 QA 쌍을 포함하는 벤치마크 데이터를 구축했습니다.

- **Technical Details**: STBench는 지식 이해, 시공간 추론, 정확한 계산, 다운스트림 응용의 네 가지 차원에서 LLMs의 능력을 평가합니다. 지식 이해 영역은 모델의 시공간 정보의 의미와 맥락을 이해하는 능력을 평가합니다. 시공간 추론 영역은 엔티티와 사건 간의 공간적 및 시간적 관계를 이해하고 추론하는 능력을 평가합니다. 정확한 계산 영역은 시공간 데이터의 정확하고 복잡한 계산을 처리하는 능력을 평가합니다. 다운스트림 응용 영역에서는 경로 이상 감지와 경로 예측 등의 실제 응용을 평가합니다.

- **Performance Highlights**: 실험 결과, 기존의 LLMs는 지식 이해와 시공간 추론 작업에서 뛰어난 성능을 보였으며, 다른 작업에서도 추가 학습(in-context learning), chain-of-thought prompting, and fine-tuning을 통해 성능 향상의 가능성이 있음을 확인했습니다. 예를 들어, ChatGPT의 POI 범주 인식 정확도는 79.26%, 행정 구역 결정 정확도는 83.58%로, 다른 오픈소스 모델에 비해 각각 34.6%, 177.3% 향상되었습니다. 그러나, 정확한 계산 작업에서는 모든 모델의 성능이 전반적으로 낮았습니다. 또한, in-context learning을 통해 ChatGPT의 POI Identification 정확도가 58.64%에서 76.30%로 향상되는 등 성능 개선의 가능성을 확인했습니다.



### Improving Weak-to-Strong Generalization with Reliability-Aware Alignmen (https://arxiv.org/abs/2406.19032)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)의 약한 감독 신호를 더 강한 일반화(weak-to-strong generalization)로 향상시키는 새로운 접근 방식을 제안합니다. 기존의 문제는 약한 감독자가 제공하는 불완전한 데이터로부터 강력한 모델이 일반화해야 하는 상황에서 발생하는 '슈퍼-얼라인먼트(super-alignment)' 문제였는데, 이를 해결하기 위해 약한 감독 신호의 신뢰성을 고려하는 방법을 개발했습니다.

- **Technical Details**: 이번 연구의 메서드는 약한 감독자(weak supervisor)로부터 다수의 답변을 요청하고, 답변의 신뢰성을 추정한 후 불확실한 데이터를 필터링하거나 신뢰성 있는 데이터에 가중치를 재조정하는 과정을 포함합니다. 구체적으로는, 원본 질문을 여러 변형으로 확장한 후 약한 감독자에 의해 생성된 답변 세트를 사용하여 신뢰성을 추정합니다. 두 가지 방법으로 신뢰성을 강화하는데, 첫째는 엔트로피 기반 불확실성 메트릭을 사용하여 불확실한 인스턴스를 제거하고 낮은 불확실성 데이터를 사용합니다. 둘째는 확률 기반 신뢰성 메트릭을 사용하여 신뢰도 점수를 추정하고 신뢰성 있는 답변에 더 높은 가중치를 부여합니다.

- **Performance Highlights**: 4개의 데이터셋에서 실험한 결과, 제안된 방법은 약한 라벨의 품질을 성공적으로 식별하고 약한 감독 신호에서 강한 일반화로 전환하는 과정에서 상당히 향상된 성능을 보였습니다. 특히, 제안된 방법은 모델 특정 특성(모델 파라미터나 그래디언트)을 필요로 하지 않는 비지도 학습 방식이기 때문에 인간 주석자도 쉽게 적용 가능합니다. 이를 통해 슈퍼-얼라인먼트 문제를 효과적으로 해결하여 LLM의 정확성과 신뢰성을 크게 높일 수 있었습니다.



### UniGen: A Unified Framework for Textual Dataset Generation Using Large Language Models (https://arxiv.org/abs/2406.18966)
- **What's New**: 대형 언어 모델(LLMs)인 GPT-4와 Llama3이 고품질 합성 데이터 생성 기능을 통해 다양한 분야에 큰 영향을 미쳤습니다. 그러나 여전히 일반화, 제어 가능성, 다양성, 진실성 등의 문제는 남아 있습니다. 이를 해결하기 위해 UniGen이라는 포괄적인 LLM 기반 프레임워크가 제안되었습니다. UniGen은 다양한, 정확한, 그리고 고도로 제어 가능한 데이터를 생성하는 데 중점을 두고 있으며, 텍스트 데이터 전반에 걸쳐 적응성을 보장합니다.

- **Technical Details**: UniGen은 데이터 다양성을 높이기 위해 속성 기반 생성(attribute-guided generation) 모듈과 그룹 체크 그룹(group checking) 기능을 도입했습니다. 정확성을 위해 라벨 검증에 코드를 사용한 수학적 평가와 사실 검증을 위한 검색 보강 생성(retrieval-augmented generation) 기법을 사용합니다. 사용자 지정 제약 조건을 허용하여 특정 요구 사항에 맞춘 데이터 생성이 가능합니다.

- **Performance Highlights**: 다양성 향상을 위해 속성 기반 생성을 도입한 결과, remote-clique 점수가 초기 생성 데이터에서 0.695에서 0.735로 증가했습니다. 그룹 체크를 추가로 구현한 후 이 점수는 0.743까지 상승했습니다. 코드 기반 수학적 평가를 통해 생성된 데이터의 정확성을 44%에서 92%로 향상시켰습니다. 또한, RAG 기반 검증은 4.2%의 예제를 수정하여 효과적임을 입증했습니다. 전체 비용 측면에서, 모든 검증 및 평가 절차를 포함하는 경우 항목당 최대 비용은 $0.200을 초과하지 않습니다.



### Selective Vision is the Challenge for Visual Reasoning: A Benchmark for Visual Argument Understanding (https://arxiv.org/abs/2406.18925)
Comments:
          12 pages, 5 figures

- **What's New**: 새로운 연구가 VisArgs라는 시각적 논증을 이해하기 위한 데이터셋을 소개합니다. 이 데이터셋은 광고나 사회적 원인과 같은 시각적 논증의 구조를 명확히 하기 위해 설계되었으며, 1,611개의 이미지와 5,112개의 시각적 전제(visual premises), 5,574개의 상식적 전제(commonsense premises), 그리고 이를 연결하는 추론 트리를 포함하고 있습니다.

- **Technical Details**: VisArgs 데이터셋에는 시각적 논증 구조를 명확히 하기 위한 세 가지 태스크가 포함되어 있습니다. 첫째, 시각적 전제의 지역화(Localization of Premises), 둘째, 전제의 식별(Identification of Premises), 셋째, 결론의 추론(Deduction of Conclusion)이 그것입니다. 각 이미지는 객체 경계 상자(object bounding box)로 구체화된 시각적 전제와 암시적 지식을 유도하는 상식적 전제가 annotations로 포함되어 있습니다.

- **Performance Highlights**: 실험 결과, 머신 모델들은 여전히 시각적 논증 이해에 있어 인간을 따라잡지 못한다는 점이 드러났습니다. 최고 성능을 보인 GPT-4-O 모델도 78.5%의 정확도를 기록한 반면, 인간은 98.0%의 정확도를 보였습니다. 특히, 이미지 외부의 객체와 무관한 객체로 비교세트를 변경할 경우, 모델들의 성능이 평균 19.5% 감소했습니다. 또한, 관련된 시각적 전제가 추가 입력으로 제공될 때 모델들의 성능이 크게 향상됨이 확인되었습니다.



### Capturing Minds, Not Just Words: Enhancing Role-Playing Language Models with Personality-Indicative Data (https://arxiv.org/abs/2406.18921)
Comments:
          10pages

- **What's New**: 최근 발표된 논문에서는 역할놀이 언어 모델(Role-playing Language Models, RPLMs)을 심리학적 지표를 반영한 데이터를 통해 개선하는 방안을 제시하고 있습니다. 기존 RPLMs는 캐릭터의 지식과 어조를 잘 재현하지만, 내면의 성격을 반영하는 데 어려움을 겪고 있습니다. 이를 해결하기 위해 심리 측정 도구(Psychological Scales)에서 질문을 수집하고, 이를 활용하여 캐릭터의 내면을 더 잘 파악할 수 있는 대화 데이터를 생성합니다.

- **Technical Details**: 논문에서는 RolePersonality라는 새로운 데이터셋을 구축하여 RPLMs를 미세 조정(Fine-Tuning)했습니다. 이 데이터셋은 14가지 심리 측정 도구에서 수집된 질문으로 구성되며, 단일 회차 데이터와 다중 회차 데이터를 모두 포함합니다. 데이터 수집 과정에서 성격 적합성을 위배하는 질문을 필터링하는 메커니즘을 도입하여 더욱 정확한 성격 재현을 도모했습니다. 이를 통해 RPLMs의 성격 관련 평가(Personality-related Evaluations)와 일반적인 역할놀이 평가에서 더 높은 성능을 보였습니다.

- **Performance Highlights**: 실험 결과, RolePersonality 데이터셋으로 미세 조정된 RPLMs는 성격 충실도(Personality Fidelity)와 동기 인식(Motivation Recognition) 측면에서 우수한 성능을 보였습니다. 특히, 기존 데이터셋(CharacterLLM과 RoleBench)으로 학습된 모델들이 성격 재현 능력에서 미달한 반면, RolePersonality 데이터셋으로 학습된 모델은 캐릭터의 전체적인 성격 특성을 더 정확하게 반영하였습니다. 또한, 다중 회차 대화를 통해 문맥 이해도와 대화 일관성을 높이는 데 기여했습니다.



### TrustUQA: A Trustful Framework for Unified Structured Data Question Answering (https://arxiv.org/abs/2406.18916)
- **What's New**: 이번 논문에서는 다양한 구조화된 데이터(예: 테이블, 지식 그래프)에서 자연어 질문에 대한 신뢰할 수 있는 답변을 제공하는 새로운 QA 프레임워크인 UnifiedTQA를 소개합니다. 이 프레임워크는 Condition Graph (CG)라는 LLM 친화적이고 통합된 지식 표현 방법을 채택하였으며, CG 쿼리를 위해 LLM 및 시연 기반의 이중 레벨 방법을 사용합니다. 또한 동적 시연 검색을 통해 성능을 향상시킵니다.

- **Technical Details**: UnifiedTQA는 두 주요 구성 요소로 이루어져 있습니다. 첫째, 테이블, 지식 그래프(KGs), 시간 지식 그래프(temporal KGs)를 CG로 변환하는 기술을 포함한 조건 그래프(CGs) 방법을 제안합니다. 둘째, CG 위에서 효과적인 쿼리 방법을 제안합니다. LLM을 사용해 질문을 기준으로 기본 쿼리를 작성하고, 사전 정의된 규칙을 사용해 이 쿼리를 실행 쿼리로 변환하여 최종적으로 CG로부터 답변을 반환합니다. LLM 쿼리 함수는 간단한 어휘로 설계되어 LLM이 더 잘 이해할 수 있도록 하였고, 미세 조정이 필요 없이 few-shot prompting을 통해 높은 정확도를 달성할 수 있었습니다. 동적 시연 검색 방법도 추가하여 프롬프트의 품질을 높여주었습니다.

- **Performance Highlights**: UnifiedTQA는 테이블 QA를 위한 WikiSQL, WTQ, 지식 그래프 QA를 위한 WebQSP, MetaQA, 시간 지식 그래프 QA를 위한 CronQuestion 총 5개의 벤치마크로 평가되었습니다. 기존의 RAG 기반 통합 QA 방법들과 비교했을 때 WikiSQL과 WebQSP에서 더 나은 성능을 보였으며, WebQSP와 CronQuestions에서는 최첨단 성능을 기록했습니다. 이 외에도 혼합 구조화된 데이터와 다양한 데이터 소스를 활용하는 QA 작업에서도 잠재력을 보여주었습니다.



### Factor-Conditioned Speaking-Style Captioning (https://arxiv.org/abs/2406.18910)
Comments:
          Accepted to Interspeech 2024

- **What's New**: 이번 논문에서는 말하는 스타일 캡션 생성 방법을 새롭게 제안합니다. 이 방법은 다양한 설명을 생성하면서도 정확하게 말하는 스타일 정보를 예측하는 특징이 있습니다. 기존의 학습 기준은 말하는 스타일 요소뿐 아니라 문법 단어도 포함된 원본 캡션을 직접 사용하여 말하는 스타일 정보 학습을 방해했습니다. 이 문제를 해결하기 위해, 스타일 조건 캡션(factor-conditioned captioning, FCC) 방법을 소개합니다. 이 방법은 먼저 말하는 스타일 요소를 나타내는 구를 출력한 후 캡션을 생성하여 모델이 명확하게 말하는 스타일 요소를 학습하도록 합니다. 또한, 우선적으로 스타일 요소를 결정론적으로 예측한 후 스타일 조건 샘플링(factor-conditioned sampling)으로 캡션을 생성하여 다양성을 보장하는 greedy-then-sampling (GtS) 디코딩도 제안합니다.

- **Technical Details**: FCC의 주요 아이디어는 캡션을 생성하기 전 스타일 요소를 명시적으로 예측해 중간 추론 단계를 거치도록 하는 것입니다. FCC의 학습 진실 값은 '남성, 낮은 음조, 높은 볼륨, 정상 속도'와 같은 고정된 형식의 스타일 요소 구와 원본 캡션으로 구성됩니다. 이 요소 구는 GPT를 사용하여 원본 캡션에서 스타일 요소를 예측하거나 사용할 수 있는 경우 진실 값 요소를 사용해 생성됩니다. GtS 디코딩은 최대 가능성 기준을 사용해 요소 구를 예측하고, 그 후 스타일에 조건화된 샘플링을 통해 캡션 부분을 생성합니다. 이를 통해 표현과 구문 측면에서 다양한 캡션을 생성하면서도 가장 높은 가능성을 가진 요소를 결정하여 스타일 요소의 잘못된 일반화를 방지합니다.

- **Performance Highlights**: PromptTTS 데이터셋 실험 결과, FCC는 기존 캡션 기반 학습 대비 성능이 크게 개선되었습니다. GtS 디코딩을 사용한 FCC는 스타일 예측 성능을 유지하면서 더 다양한 캡션을 생성하였습니다.



### Historia Magistra Vitae: Dynamic Topic Modeling of Roman Literature using Neural Embeddings (https://arxiv.org/abs/2406.18907)
Comments:
          6 pages, 2 figures

- **What's New**: 이번 연구에서는 BERT 임베딩(embeddings)을 사용한 최신 동적 주제 모델(dynamaic topic model)을 제안하고, 전통적인 통계 모델(LDA와 NMF)과 비교하는 실험을 수행했습니다. 로마 문학의 전체 생존 코퍼스(corpus)를 대상으로 주제 모형을 구축하였으며, 신경망(neural) 기반 모델이 더 좋은 통찰을 제공하고 하이퍼파라미터(hyperparameter) 설정에 덜 민감하다는 것을 발견했습니다.

- **Technical Details**: 연구는 라틴 문학 자료를 대상으로 하며, 전처리(preprocessing)로는 토큰화(tokenization), 형태소 분석(lemmatization), 불용어 제거(stop words removal)를 수행했습니다. 세 가지 모델인 Latent Dirichlet Allocation (LDA), Non-negative Matrix Factorization (NMF), 그리고 BERTopic(neural embedding 기반)을 비교했습니다. 각 모델을 평가하기 위해 TC-Embed 및 Mean Pairwise Jaccard 유사성 측정치 등의 정량적 메트릭을 사용했습니다.

- **Performance Highlights**: 정량적 메트릭에서는 전통적인 통계 모델(LDA와 NMF)이 더 좋은 성능을 보였으나, 정성적 평가에서는 신경망 기반 모델이 더 나은 인사이트를 제공했습니다. 특히 신경망기반 BERTopic 모델은 하이퍼파라미터 튜닝이 최소화되면서 더 유용한 주제 분포를 생성했습니다. LDA와 NMF 모델은 주로 일반적인 용어로 구성된 주제를 생성하는 반면, BERTopic 모델은 의미 있는 주제 분포를 제공하는 경향이 있었습니다.



### Sonnet or Not, Bot? Poetry Evaluation for Large Models and Datasets (https://arxiv.org/abs/2406.18906)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 시적 형태(poetic form) 인식 능력을 평가하는 새로운 태스크를 개발했습니다. 이 태스크는 영어 시에서의 운율 체계(rhyme scheme), 미터(meter), 단어나 행의 반복과 같은 20개 이상의 형태와 형식 요소를 대상으로 합니다. 이를 통해 LLMs의 현재 시적 능력을 반영하고, NLP 벤치마크를 시와 같은 창의적 작업에 적용하는 것의 도전과제와 함정을 탐구합니다.

- **Technical Details**: 이 연구는 4.1k 시를 사용한 벤치마크 평가 실험을 포함하며, 일반 대중이 사용할 수 있는 코드, 데이터(1.4k 공공 도메인 시와 형태 주석) 및 메타데이터(사전 훈련 포함 및 모델 암기)를 제공합니다. 시적 형태는 고정된 형태(fixed forms), 형식 요소(formal elements), 고정되지 않은 형태(unfixed forms)로 분류되며, 각각 다른 패턴과 내용을 기반으로 평가됩니다.

- **Performance Highlights**: GPT-4 및 GPT-4o와 같은 최신 LLM들은 소네트(sonnets), 세스티나(sestinas), 판텀(pantoums)과 같은 일반적 및 비일반적 고정 시적 형태를 높은 정확도로 식별할 수 있었습니다. 그러나 주제나 시각적 요소를 기반으로 한 고정되지 않은 형태를 식별하는 데에는 어려움을 겪었습니다. LLM들은 인기 있는 사전 훈련 데이터셋에 많이 포함된 시적 형태에서 가장 높은 성과를 보였으나, 주요 온라인 시 기관이나 인쇄 책에서 수집된 시와의 성능 차이는 크지 않았습니다.



### Can we teach language models to gloss endangered languages? (https://arxiv.org/abs/2406.18895)
- **What's New**: 새로운 연구에서는 대형 언어 모델(LLMs)이 in-context learning을 통해 interlinear glossed text (IGT)을 생성할 수 있는지를 탐구합니다. 기존의 통계적 및 신경망 기반 방법론들과는 달리, 이 접근법은 훈련 없이도 특정 예시를 선택해 성능을 향상시킬 수 있음이 밝혀졌습니다. 특히, 대형 언어 모델이 필요한 훈련 없이도 표준 Transformer 모델들을 능가하였으며, 이는 언어 문서화를 위한 실질적인 도구로 자리 잡을 가능성이 큽니다.

- **Technical Details**: IGT 생성 작업은 transcription line과 translation line을 제공받아 gloss line을 예측하는 것입니다. 본 연구는 분리된 형태소가 없는 폐쇄 트랙 환경에서 이 작업을 수행하였습니다. 특별히, SIGMORPHON Shared Task 2023에서 사용된 IGT 코퍼스와 언어 데이터를 활용하며, lower-resource 언어에 집중했습니다. 주요 성과 지표로 형태소 정확도(morpheme accuracy)를 사용하였으며, 이를 통해 예측된 gloss와 실제 gloss 간의 일치도를 평가합니다.

- **Performance Highlights**: Cohere의 Command R+ 모델(104B 파라미터)를 사용한 초기 실험 결과, 훈련 없이도 Transformer 기반 모델보다 더 나은 성능을 보였습니다. 하지만, 최첨단의 감독 기반 시스템들이 성능에서 여전히 우위를 점하고 있으나, LLM 기반 접근법은 언어학적 전문 지식 없이도 쉽게 사용할 수 있는 장점이 있어 실질적 활용 가능성이 높습니다.



### SSP: Self-Supervised Prompting for Cross-Lingual Transfer to Low-Resource Languages using Large Language Models (https://arxiv.org/abs/2406.18880)
- **What's New**: 최근의 대형 언어 모델(LLMs)은 영어 NLP 작업에서 뛰어난 성과를 보여줬지만, 다른 언어에 대한 활용도는 아직 미흡합니다. 우리는 낮은 자원의 언어(LRLs)에 대한 zero-labelled cross-lingual transfer(0-CLT) 설정에서 LLM의 효과를 조사합니다. 여기서는 타겟 언어에 대한 레이블 데이터가 전혀 없이 관련된 중간 자원의 언어(MRLs)에 대한 훈련 데이터를 이용합니다. 이를 위해 새로운 In-Context Learning(ICL) 접근법인 Self-Supervised Prompting(SSP)를 도입했습니다. SSP는 타겟 언어의 데이터가 존재하지 않는 상황에서 LLMs가 타겟 언어의 예제를 기반으로 더 정확한 레이블을 출력한다는 관찰에 기반합니다.

- **Technical Details**: SSP는 두 개의 단계로 구성됩니다. 첫 번째 단계에서는 MRL의 훈련 데이터를 사용해 타겟 언어의 테스트 데이터를 노이즈 있게 레이블링합니다. 두 번째 단계에서는 이러한 노이즈 있는 테스트 데이터를 ICL의 예제로 사용해 성능을 향상시킵니다. 또한, Integer Linear Programming(ILP) 기반의 예제 선택 알고리즘을 활용해 테스트 문장과의 유사성, 레이블 예측 신뢰도, 레이블 커버리지를 균형 있게 조정합니다. 세 가지 시나리오(0-CLT, 0-CLT-U, 0-CLT-T)에 걸쳐 SSP의 효율성을 평가합니다.

- **Performance Highlights**: 실험은 아프리카, 게르만, 아메리카 지역의 11개 언어 및 3개 작업(POS, NER, NLI)에서 수행되었습니다. SSP는 기존의 미세 조정 및 단순한 ICL 기반 접근법보다 일관되게 우수한 성능을 보였습니다. 특히, SSP는 다양한 노이즈 수준에서 효과적이며, 두 번째 단계에서 레이블링을 향상시키는 데 기여합니다.



### Efficacy of Language Model Self-Play in Non-Zero-Sum Games (https://arxiv.org/abs/2406.18872)
- **What's New**: 본 논문은 자가 학습(self-play)을 통해 언어 모델(languagemodel)의 성능을 향상시킬 수 있는지를 실험적으로 조사합니다. 저자들은 협동적 및 경쟁적 목표 설정이 가능한 '딜 오어 노 딜(Deal or No Deal)'이라는 협상 게임에서 언어 모델을 여러 번의 자가 학습으로 미세 조정(finetuning)합니다. 예상과는 달리, 인간과의 협력 및 경쟁 상황 모두에서 성능 향상이 관찰되었습니다.

- **Technical Details**: 연구자들은 '딜 오어 노 딜'이라는 이원 협상 게임에서 협력적, 준경쟁적, 순수 경쟁적 목표를 설정하여 언어 모델의 자가 학습 실험을 진행했습니다. 자가 학습 과정은 필터링된 행동 복제(filtered behavior cloning)를 기반으로 한 알고리즘을 사용하며, 자연어 데이터를 초기화한 이후 유사 자연스러운 의사소통 전략을 개발하도록 했습니다.

- **Performance Highlights**: 실험 결과, 자가 학습을 통해 협력적 설정에서 최대 2.5배, 준경쟁적 설정에서 최대 6배의 점수 향상을 달성했습니다. 이는 인간 실험에서도 동일하게 적용되었습니다. 그러나 순수 경쟁적 설정에서는 최소한의 성능 향상만이 나타났습니다. 또한, 자가 학습을 통해 모델은 과업 지침을 더 잘 따르고, 환상(hallucination)이 줄어들며, 인간과의 합의율이 높아졌습니다.



### Two-Pronged Human Evaluation of ChatGPT Self-Correction in Radiology Report Simplification (https://arxiv.org/abs/2406.18859)
- **What's New**: 최근 연구에서는 방사선 의학 보고서를 환자 친화적 언어로 자동 변환하는 데 있어서 대형 언어 모델(Large Language Models, LLMs)의 적합성을 탐구합니다. 이 연구는 체인-오브-생각(chain-of-thought) 및 자기-수정(self-correction) 프롬프트 메커니즘의 유용성을 조사하며, 새롭게 제안된 평가 프로토콜을 통해 방사선 전문의와 일반인 모두가 평가에 참여하는 방식을 제안합니다.

- **Technical Details**: 주요 기술적 세부 사항으로는 대형 언어 모델(LLMs)을 사용하여 방사선 보고서를 환자 친화적으로 변환하는 방법이 포함됩니다. 이 연구는 ChatGPT와 같은 최신 LLMs의 체인-오브-생각(Chain-of-Thought) 및 자기-수정(Self-Correction) 프롬프트 메커니즘을 심층 분석 합니다. 체인-오브-생각 접근 방식에서는 LLM이 답변을 제공하기 전에 타당성을 평가하고, 자기-수정 접근 방식에서는 LLM이 본인의 응답을 비판하고 개선된 응답을 제공하도록 유도합니다.

- **Performance Highlights**: 실험 결과, 자기-수정 프롬프트 메커니즘이 높은 품질의 환자 친화적 변환 생성에 효과적임을 보여줍니다. 또한, 방사선 전문의와 일반인의 텍스트 단순화에 대한 선호도를 조사하여 미래 연구에 유익한 통찰을 제공합니다.



### FFN: a Fine-grained Chinese-English Financial Domain Parallel Corpus (https://arxiv.org/abs/2406.18856)
Comments:
          a simplified version of this paper is accepted by International Conference on Asian Language Processing 2024

- **What's New**: 이번 연구는 금융 뉴스의 중국어-영어 병렬 코퍼스를 구축한 결과를 바탕으로, ChatGPT와 ERNIE-bot과 같은 대형 언어 모델(Large Language Models, LLMs)의 번역 성능을 평가했습니다. 이는 금융 분야에서의 기계 번역에 대한 탐구를 촉진하고자 하는 첫 번째 시도로, 기존의 데이터셋들과 비교하여 더 높은 품질과 새로운 데이터를 제공하는 것을 목표로 합니다.

- **Technical Details**: FFN이라 명명된 이 데이터셋은 2014년 1월 1일부터 2023년 12월 31일까지의 기간 동안 수집된 주요 텍스트와 제목을 포함하며, 이는 모든 문장이 수동으로 교정된 상태입니다. 평가 지표로는 BLEU, TER, chrF 스코어를 사용했으며, OpenNMT 모델을 훈련시켜 ChatGPT와 ERNIE-bot의 성능과 비교했습니다. 또한, DeepL 및 Google 번역기와도 비교 분석을 수행했습니다.

- **Performance Highlights**: ChatGPT와 ERNIE-bot 모두 특정 영역에서 강점을 보였으나, 두 모델은 금융 번역에서 여전히 개선의 여지가 있음을 확인했습니다. 특히, OpenNMT 모델과의 비교 분석을 통해 데이터셋의 유효성을 확인했으며, 이는 향후 금융 번역의 정확도와 품질을 높이기 위한 연구에 중요한 시사점을 제공합니다.



### Learning Retrieval Augmentation for Personalized Dialogue Generation (https://arxiv.org/abs/2406.18847)
Comments:
          Accepted to EMNLP-2023

- **What's New**: 개인화된 대화 생성(personalized dialogue generation) 분야에서 에이전트의 페르소나 프로필과 대화 문맥(dialogue context)을 활용하여 특화된 응답을 생성하는 작업이 주목받고 있습니다. 그러나 현재 사용되는 페르소나 프로필은 보통 네다섯 개의 문장으로 구성되어 있어 에이전트의 페르소나를 충분히 설명하지 못하는 문제가 있습니다. 이를 해결하기 위해 외부 지식을 활용하는 새로운 모델인 LAPDOG(Learning Retrieval Augmentation for Personalized Dialogue Generation)을 제안합니다.

- **Technical Details**: LAPDOG 모델은 스토리 리트리버(story retriever)와 대화 생성기(dialogue generator)로 구성됩니다. 주어진 페르소나 프로필을 쿼리로 사용하여 스토리 문서에서 관련 정보를 검색한 후, 이를 통해 페르소나 프로필을 보완합니다. 이 보완된 페르소나 프로필과 대화 이력을 사용하여 특화된 응답을 생성합니다. 최적화를 위해 스토리 리트리버와 대화 생성기를 공동 학습하는 프레임워크를 채택했으며, 리트리버는 원하는 최종 메트릭(BLEU 등)에 맞춰 최적화되어 대화 생성기에 적절한 콘텐츠를 제공하게 됩니다.

- **Performance Highlights**: CONVAI2 데이터셋과 ROCStory를 추가 데이터 소스로 사용한 실험 결과, LAPDOG 방법이 기존의 베이스라인을 크게 능가하는 성능을 보였습니다. 다양한 모델 크기에서도 일관되게 개선된 성능을 나타냈으며, 복합적인 실험 결과, 공동 목표 지침(joint objective guidance)이 개별 목표보다 우수한 성능을 제공함을 확인했습니다. LAPDOG 모델은 현재 코드가 공개되어 있어 추가 탐색이 가능합니다.



### OutlierTune: Efficient Channel-Wise Quantization for Large Language Models (https://arxiv.org/abs/2406.18832)
- **What's New**: OutlierTune이라는 새로운 퍼채널 후훈련 양자화(PTQ) 방법이 소개되었습니다. 이는 LLM(대형 언어 모델)의 활성화를 효과적으로 양자화합니다. 해당 방법은 두 가지 주요 컴포넌트로 구성됩니다: 비대칭 (pre-execution of dequantization)과 대칭화 (symmetrization).

- **Technical Details**: OutlierTune은 활성화의 스케일링 팩터를 사용하여 모델 가중치를 업데이트하는 비대칭을 도입하여, 내부 스케일링과 추가적인 계산 오버헤드를 피합니다. 대칭화는 서로 다른 활성화 채널 간의 균형 잡힌 수치 범위를 보장하여 양자화 오류를 줄입니다. 이는 하드웨어 효율적이고 추가 계산 오버헤드를 거의 유발하지 않습니다.

- **Performance Highlights**: 다양한 작업에 대한 광범위한 실험 결과, OutlierTune이 기존 방법보다 뛰어난 성능을 보였습니다. Int6 양자화를 통해 OPT-IML과 같은 LLM의 성능을 절반 정밀도(FP16) 수준으로 향상시켰으며, FP16 구현보다 1.48배 빠르면서 메모리 사용량은 약 2배 감소했습니다.



### Psychological Profiling in Cybersecurity: A Look at LLMs and Psycholinguistic Features (https://arxiv.org/abs/2406.18783)
- **What's New**: 이 논문은 사이버 보안에서 심리적 프로파일링(psychological profiling)을 적용하여 위협 행위자의 심리적 특성을 식별하는 방법을 탐구합니다. 특히 Large Language Models (LLMs)와 심리언어학적 특징(psycholinguistic features)을 활용한 텍스트 데이터 분석을 통해 이러한 특성을 파악하는 데 중점을 두고 있습니다. 이 연구는 사이버 보안과 심리학의 교차점을 탐구하며 심리적 관점을 사이버 보안 실천에 통합할 필요성을 강조합니다.

- **Technical Details**: 논문은 심리적 관점을 도입하여 사이버 위협에 대한 방어 메커니즘을 강화하는 접근법을 논의합니다. 주요 기술 중 하나는 LLMs와 심리언어학적 도구(LIWC, MRC 데이터베이스)를 활용한 사이버 범죄자의 프로파일링입니다. 이를 통해 언어 패턴 및 감정 신호를 분석하여 위협 행위자의 심리적 특성을 식별하고, 전반적인 보안 전략을 강화할 수 있습니다.

- **Performance Highlights**: 사이버 범죄자들은 기술적으로 능숙하며, 특정 그룹이나 기관에 대한 복수, 금전적 이익, 스릴 추구 등 다양한 동기에서 범행을 저지르는 것으로 나타났습니다. 연구는 사이버 범죄자들이 보이는 다양한 심리적 특성을 이미지화하고 이러한 특성을 통해 보안 위험을 최소화할 수 있는 방안을 제시합니다. LLMs와 심리언어학적 분석 기법의 통합은 사이버 범죄자의 복잡한 성격을 드러내고, 이에 기반한 보안 방어 전략을 강화하는 데 중요한 역할을 합니다.



### Implicit Discourse Relation Classification For Nigerian Pidgin (https://arxiv.org/abs/2406.18776)
- **What's New**: 이 논문은 나이지리아 피진(Nigerian Pidgin, NP)이라는 거의 1억 명이 사용하는 언어에 대해 NLP 자원이 부족한 문제를 다루며, NP에서의 묵시적 담화 관계 분류(Implicit Discourse Relation Classification, IDRC)에 중점을 두고 있습니다. NP 데이터를 영어로 번역하여 영어 IDRC 도구를 사용하는 방법과 NP 전용 담화 코퍼스를 만들어 NP IDR 분류기를 학습시키는 두 가지 접근 방식을 비교하였습니다.

- **Technical Details**: 첫 번째 접근 방식은 영어로 학습된 최첨단 분류기를 NP 문장에 직접 적용하거나 NP 텍스트를 영어로 번역한 뒤 분류기 결과를 원래 NP 텍스트에 투사하는 zero-shot 학습에 기반합니다. 두 번째 접근 방식은 NP 전용 모델을 미세조정(fine-tuning)하는 것으로, 이를 위해 NP에서 주석 처리된 데이터를 사용합니다. 주석을 얻기 위해 전체 텍스트를 번역하거나 각각의 관계 인수를 개별적으로 번역하는 두 가지 방법을 시도했습니다.

- **Performance Highlights**: NP 전용 IDR 분류기를 학습시키는 접근 방식이 기존 방법 대비 4-way 분류에서 13.27%, 11-way 분류에서 33.98%의 F1 점수 개선을 보였습니다. 모델의 성능 평가를 위해 DiscoPrompt를 사용하였으며, 최상의 설정에서 4-way 관계 분류에서 정확도와 F1 점수는 각각 0.631과 0.461을, 11-way 관계 분류에서 0.440과 0.327을 기록했습니다.



### Categorical Syllogisms Revisited: A Review of the Logical Reasoning Abilities of LLMs for Analyzing Categorical Syllogism (https://arxiv.org/abs/2406.18762)
- **What's New**: 이 논문은 대형 언어 모델(Large Language Models, LLMs)의 논리적 추론 능력을 평가하기 위한 기존 연구들을 체계적으로 검토하고, 특히 범주적 삼단논법(Categorical Syllogisms)의 분석에 집중합니다. 기존 데이터셋의 구성 요소(예: mood와 figure)를 조사하고, 군중 소싱 방식이 다양한 언어 변이를 포괄하려다 보니 논리적 구성 요소의 커버리지가 감소하는 문제를 발견했습니다.

- **Technical Details**: 연구팀은 논리학적 관점에서 범주적 삼단논법의 모든 가능한 변형을 조사했으며, 기존 데이터셋이 테스트한 구성 요소들을 분석했습니다. 특히, LLMs가 범주적 삼단논법의 타당성을 추론하는 능력을 분석한 기존 연구들을 요약하고, quantifier(양화사)의 해석이 현재 LLMs 성능의 병목 현상임을 강조했습니다.

- **Performance Highlights**: 논문은 LLMs가 양화사 해석에서 높은 오류율을 보이는 점을 주요 병목 현상으로 지적했습니다. 이는 미래의 연구에서 양화사 해석을 개선할 필요가 있음을 시사합니다. 또한, 범주적 삼단논법 데이터셋을 설계하는 과정에서 존재적 의미(existential import)를 명확히 하고, 완전한 주석을 제공하며, 일반적인 논증을 포함하는 데이터셋을 구축할 것을 권장했습니다.

- **Future Directions**: 미래 데이터셋의 설계 시 고려해야 할 점으로, 존재적 의미를 명확히 하고, 완전한 주석을 제공하며, 일반적인 논증을 포함하는 데이터셋을 구축할 것을 제안했습니다. 이 연구가 범주적 삼단논법에 관한 시의 적절한 리뷰를 제공하고, 계산 언어학자와 논리학자 간의 더 많은 학제간 연구를 촉진할 것을 기대합니다.



### Re-Ranking Step by Step: Investigating Pre-Filtering for Re-Ranking with Large Language Models (https://arxiv.org/abs/2406.18740)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)을 이용한 정보 검색(IR)에서 사전 필터링(pre-filtering) 단계를 도입하여, 패시지 재정렬(passage re-ranking) 성능을 향상시키는 방법을 제안합니다. LLM에서 생성된 관련성 점수를 사용하여 관련 없는 패시지를 필터링한 후 재정렬을 수행하면, 자원이 제한된 소형 모델도 높은 성능을 보이는 대형 독점 모델과 경쟁할 수 있음을 보여줍니다.

- **Technical Details**: 이 연구는 LLM이 특정 쿼리와 패시지의 관련성을 0에서 1 사이의 점수로 평가하도록 하는 프롬프팅 전략을 사용합니다. 그런 다음, 소수의 사람에 의해 생성된 관련성 점수를 활용해 필터링 기준(threshold)을 설정하고, 이 기준을 초과하는 패시지만을 재정렬 단계로 전달합니다. 이 접근법의 효용성을 평가하기 위해 TREC-DL2019, TREC-DL2020, 및 여러 BEIR 데이터셋을 테스트했으며, 대부분의 데이터셋에서 필터링 기준을 찾아낼 수 있음을 확인했습니다.

- **Performance Highlights**: 이 연구에서는 제안한 사전 필터링 단계를 통해 Mixtral-8x7B-Instruct와 같은 소형 모델이 GPT-4와 같은 대형 모델의 성능에 근접하거나 일부 경우 이를 능가할 수 있음을 보여줍니다. 사전 필터링을 통해 재정렬 효율성도 크게 향상되었으며, TREC-DL2019, TREC-DL2020 그리고 여러 BEIR 태스크에서 훌륭한 성능을 발휘했습니다.

- **Research Questions**: ['RQ1: 전문가의 기존 지식을 사용하여 LLM이 관련 없는 패시지를 필터링할 수 있는가?', 'RQ2: 재정렬 전 관련 없는 패시지를 필터링하는 것이 LLM 재정렬 결과를 향상시킬 수 있는가?']



### Sequence Graph Network for Online Debate Analysis (https://arxiv.org/abs/2406.18696)
Comments:
          8 pages, 4 figures

- **What's New**: 온라인 토론에서 참가자들의 상호작용을 모델링하는 새로운 방법이 제안되었습니다. 시퀀스 그래프 접근 방식을 이용하여, 대화의 흐름을 그래프로 표현하고, 시퀀스 그래프 어텐션 층(Sequence Graph Attention layer)을 도입하여 정보 갱신 과정을 설명합니다. 실험 결과, 이 방법이 기존의 방법보다 우수한 성능을 보였습니다.

- **Technical Details**: 이 연구에서는 온라인 토론을 시퀀스 그래프 네트워크 방식으로 모델링합니다. 각 문장을 노드로, 그들 간의 관계를 엣지로 표현하여 그래프를 구성합니다. 시퀀스 그래프 어텐션(SGA) 셀을 도입하여 장기적인 의존성을 포착하고, 다양한 엣지 타입 (논리적 및 일관된 엣지, 강화 엣지, 반박 엣지)을 사용하여 참가자들의 전략을 모델링합니다.

- **Performance Highlights**: 실험 결과, 시퀀스 그래프 어텐션 네트워크는 여러 온라인 토론 데이터셋에서 기존의 최고 성능 모델들을 압도하며 승자를 정확히 예측하는 데 더 나은 성능을 보였습니다. 이 방법론의 소스코드와 모델은 https://github.com/quanmai/SGA에서 이용 가능합니다.



### The Multilingual Alignment Prism: Aligning Global and Local Preferences to Reduce Harm (https://arxiv.org/abs/2406.18682)
- **What's New**: 이번 연구에서는 다양한 언어와 문화적 선호도를 최적화하면서도 글로벌과 지역적 해악을 최소화하는 것을 목표로 하는 새로운 AI 정렬(alignment) 접근 방식을 탐구합니다. 첫 번째로 인간이 주석을 달아준 다국어 악성 프롬프트(red-teaming prompts) 데이터를 수집하여 글로벌과 지역적 해악을 구분했습니다. 이 연구에서는 6가지 언어에 걸친 최첨단 정렬 기법의 성능 저하를 최소화하는 새로운 기준을 세웠습니다.

- **Technical Details**: 이번 연구는 다국어 안전 정렬(multilingual safety alignment)을 위해 Supervised Fine-tuning(SFT)와 Direct Preference Optimization(DPO)를 사용한 포괄적인 실험을 진행했습니다. 또한, 인간 평가와 모델 평가를 결합하여 글로벌과 지역적 해악 간의 상호작용을 분석했습니다. Aya Red-teaming 데이터셋을 구축하기 위해 8개 언어로 유해 프롬프트를 수집했으며, 각 언어의 문화적 또는 지역적 특성을 이해하는 원어민 주석자들이 참여했습니다.

- **Performance Highlights**: DPO 기법은 유해한 생성 비율을 54.7% 줄이는 동시에, 일반적인 생성 작업에서 기본 모델 대비 71%의 승률을 기록하며 SFT를 능가했습니다. 다양한 언어에 걸친 실험에서도 최소 37%의 일관된 해악 감소 효과를 보였습니다. 또한, '지역적' 해악에 기반한 훈련이 '글로벌' 해악의 감소에 77.8%의 절대적 변화롤 일으키며, '글로벌' 해악만을 훈련한 경우보다 11.6% 높은 결과를 나타냈습니다.



### Understand What LLM Needs: Dual Preference Alignment for Retrieval-Augmented Generation (https://arxiv.org/abs/2406.18676)
Comments:
          Work in progress

- **What's New**: DPA-RAG는 다양한 지식 선호도를 정렬하여 신뢰할 수 있는 RAG(Retrieval-augmented generation) 시스템을 개발하기 위해 개발된 범용 프레임워크입니다. 다섯 가지 새로운 쿼리 증강 전략을 도입하여 지식 선호 데이터 부족 문제를 해소하고자 합니다.

- **Technical Details**: DPA-RAG는 외부 및 내부 선호도 정렬(preference alignment)을 통해 작동합니다. 외부 정렬에서는 두 개의 지식 선호도를 비교(pair-wise), 한 지식 선호도를 단일 데이터 기준으로 평가(point-wise), 대조적인 지식 선호도를 비교(contrastive preference alignment)하는 능력을 재정렬기(reranker)에 통합합니다. 내부 정렬에서는 슈퍼바이즈드 파인 튜닝(SFT) 이전에 사전 정렬 단계를 도입하여 LLMs가 그들의 추론 선호도에 맞게 지식을 암묵적으로 포착하게 합니다.

- **Performance Highlights**: 네 개의 지식 집중 QA 데이터셋에서 실험 결과 DPA-RAG는 모든 기준을 능가하며, 블랙박스 및 오픈 소스 LLM 리더를 원활하게 통합합니다. 추가적인 질적 분석 및 논의는 신뢰할 수 있는 RAG 시스템을 구축하는 데 필요한 경험적 지침을 제공합니다.



### Evaluating Copyright Takedown Methods for Language Models (https://arxiv.org/abs/2406.18664)
Comments:
          31 pages, 9 figures, 14 tables

- **What's New**: 이 논문은 대형 언어 모델(LMs)이 저작권이 있는 콘텐츠를 무단으로 생성하는 문제를 완화하는 방법에 대해 첫 평가를 소개합니다. 본 논문에서는 CoTaEval이라는 평가 프레임워크를 제안하여 저작권 보호를 위한 체계적 평가를 수행하며, 여러 전략을 통해 모델의 일반적인 유용성 및 효율성을 유지하고자 합니다.

- **Technical Details**: 논문에서는 시스템 프롬프트(System Prompt), 디코딩 시간의 필터링 개입(Decoding-time filtering interventions), 그리고 '머신 언러닝(Machine Unlearning)' 접근법과 같은 여러 방법을 조합하여 평가합니다. CoTaEval은 두 가지 주요 원인인 메모라이제이션(Memorization)과 검색 증강 생성(Retrieval-Augmented Generation)을 다루며, 저작권 차단 목록에 있는 데이터를 생성하지 않도록 하는 능력을 평가합니다.

- **Performance Highlights**: 평가 결과, 테스트된 방법들 중 어느 것도 모든 지표에서 뛰어나지 않음을 확인했습니다. 머신 언러닝과 Top-k Perturbation 방법이 저작권 침해를 줄였으나 사실적 지식 보존 측면에서 큰 손실이 있습니다. R-CAD는 차단 효과는 있지만 효율성 저하와 유용성 감소의 위험이 있습니다. 추가 연구가 필요한 상태이며, 현재 상용화된 방법들은 충분하지 않다고 결론지었습니다.



### Taming Data and Transformers for Audio Generation (https://arxiv.org/abs/2406.19388)
Comments:
          Project Webpage: this https URL

- **What's New**: 이번 연구에서는 AutoCap과 GenAu라는 두 가지 신모델을 소개합니다. AutoCap은 고품질의 자동 오디오 캡션 생성 모델로, 메타데이터를 활용하여 캡션 품질을 크게 개선합니다. 이 모델은 CIDEr 점수 83.2를 기록하며, 기존 최고의 캡션 모델보다 3.2% 개선된 성능을 보였습니다. GenAu는 1.25B 파라미터를 갖춘 확장 가능한 트랜스포머 기반의 오디오 생성 아키텍처로, 새로운 대규모 데이터셋을 사용하여 훈련됩니다.

- **Technical Details**: AutoCap은 BART 모델을 기반으로 한 인코더-디코더 트랜스포머 설계를 사용하며, Q-Former를 도입하여 오디오 토큰을 요약합니다. 또한, 메타데이터와 캡션을 사용하여 이중 입력 접근 방식을 채택하여 성능을 향상시킵니다. GenAu는 효율적인 1D-VAE를 사용하여 Mel-Spectrogram을 시퀀스로 변환하고, FIT 트랜스포머 아키텍처를 수정하여 오디오 생성의 비디어 백본으로 활용합니다. 텍스트 조건부를 위한 이중 인코더 전략을 채용하여 성능을 개선하였습니다.

- **Performance Highlights**: AutoCap은 AudioCaps 데이터셋에서 CIDEr 점수가 3.2% 개선되었고, 761,000개의 고품질 캡션을 생성하였습니다. GenAu는 최첨단 오디오 생성 모델과 비교하여 FAD 점수에서 15.7%, IS는 22.7%, CLAP 점수는 13.5% 향상되었습니다.



### The Remarkable Robustness of LLMs: Stages of Inference? (https://arxiv.org/abs/2406.19384)
- **What's New**: 이번 연구는 큰 언어 모델(Large Language Models, LLMs)의 추가 층 삭제 및 인접 층 교체(interventions)가 모델의 예측 정확도에 미치는 영향을 조사합니다. 구체적으로, 이러한 작업 이후에도 모델은 무튜닝 상태에서 원래 예측 정확도(72-95%)를 유지할 수 있음을 보여줍니다. 더 많은 층을 가진 모델일수록 이러한 견고성이 더 강하다는 것도 밝혀졌습니다.

- **Technical Details**: 본 연구는 8개의 다른 모델에 대해 층별 개입(layer-wise intervention) 후 네 가지의 보편적 추론 단계를 제안합니다. 이 단계는 (1) 디토큰화(detokenization), (2) 특징 공학(feature engineering), (3) 예측 앙상블(prediction ensembling), (4) 잔차 정교화(residual sharpening)입니다. 첫 번째 단계에서 모델은 로컬 정보(local information)를 통합하여 원시 토큰 표현을 고차원적 문맥 표현으로 변환합니다. 두 번째 단계에서는 작업과 엔티티 기반 특징을 반복적으로 정제합니다. 세 번째 단계에서 모델은 높은 MLP 계산을 통해 관련 예측을 강조합니다. 마지막 단계에서는 불필요한 특징을 제거하여 다음 토큰의 분포를 정밀하게 다듬습니다.

- **Performance Highlights**: 레이어 삭제 및 교체 후에도 모델의 예측 정확도가 72-95% 유지됨을 발견했습니다. 특히, 더 많은 층을 가진 모델이 더 높은 견고성을 보였습니다. 이러한 연구 결과는 새로운 네 가지 보편적 추론 단계를 가설로 제시하며, 각 층의 역할과 깊이에 대한 새로운 이해를 제공합니다.



### Jump Starting Bandits with LLM-Generated Prior Knowledg (https://arxiv.org/abs/2406.19317)
- **What's New**: 이번 연구에서는 대형 언어 모델(Large Language Models; LLMs)을 컨텍스트 기반 멀티-암드 밴디트(Contextual Multi-Armed Bandit; CB) 프레임워크와 통합할 때의 이점을 증명합니다. LLMs는 방대한 인간 지식과 선호도로 사전 훈련되어 있어, CB의 초기화를 도와 온라인 학습의 리그렛(learning regret)을 줄일 수 있습니다.

- **Technical Details**: 컨텍스트 기반 멀티-암드 밴디트 프레임워크는 사용자의 구체적인 컨텍스트에 따라 개인화된 추천을 생성합니다. 본 연구에서는 LLMs를 활용해 대략적인 인간 선호도 데이터셋을 생성하고 이를 CB의 초기화 알고리즘에 사용합니다. 이 방법은 모델 훈련을 위한 데이터 수집 비용과 온라인 학습 리그렛을 크게 감소시킵니다. 두 가지 실험 설정에서 제안된 방법을 검증했으며, 하나는 LLMs를 오라클(oracle)로 활용하고, 다른 하나는 실제 조사 데이터를 활용하는 실험입니다.

- **Performance Highlights**: 제안된 CBLI(Contextual Bandits with LLM Initialization) 방법은 초기 리그렛을 14–17%, 19–20%씩 각각 감소시켰습니다. 특정 프라이버시 보호 속성들이 제외된 상황에서도 초기 리그렛을 14.8% 감소시키는 성능을 보였습니다. 이 방법은 마케팅 커뮤니케이션의 스타일을 설정하는 실험과 실세계 조사 데이터를 이용한 실험에서 모두 유효성을 입증했습니다.



### From Artificial Needles to Real Haystacks: Improving Retrieval Capabilities in LLMs by Finetuning on Synthetic Data (https://arxiv.org/abs/2406.19292)
- **What's New**: 최근 연구에 따르면 대형 언어 모델(Large Language Models, LLMs)은 긴 문맥을 처리할 때 정보 검색과 추론 능력에 어려움을 겪고 있습니다. 이를 해결하기 위해, 우리는 숫자 키-값 검색 작업을 포함한 신중하게 설계된 합성 데이터셋을 활용한 파인튜닝 방법을 제안합니다. 이 방법을 통해 GPT-3.5 Turbo와 Mistral 7B와 같은 모델에서 정보 검색 및 추론 능력이 현저히 개선됨을 확인했습니다.

- **Technical Details**: 이번 연구에서는 신중하게 설계된 합성 데이터셋을 이용한 파인튜닝 기법을 활용했습니다. 이 데이터셋은 주로 숫자 키-값 검색 작업(numerical key-value retrieval tasks)을 포함하고 있습니다. 모델들은 이 데이터셋을 기반으로 파인튜닝되어 긴 문맥에서도 보다 정확한 정보 검색 및 추론 능력을 보여줍니다. 실험은 주로 GPT-3.5 Turbo와 Mistral 7B 모델을 대상으로 진행되었습니다.

- **Performance Highlights**: 파인튜닝된 모델들은 실제 작업 평가에서도 우수한 성능을 보여주었으며, 예를 들어 GPT-3.5 Turbo는 $20$개의 문서에서 $10.5	ext{	extperthousand}$의 성능 향상을 보였습니다. Mistral 7B 모델 역시, TriviaQA와 같은 일반 벤치마크에서 성능 저하 없이 좋은 성능을 유지했습니다($-2.33	ext{	extperthousand}$에서 $-6.19	ext{	extperthousand}$까지 성능 하락을 유발한 다른 데이터셋과 달리).



### HuatuoGPT-Vision, Towards Injecting Medical Visual Knowledge into Multimodal LLMs at Sca (https://arxiv.org/abs/2406.19280)
- **What's New**: 새로운 논문에서는 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 의학적 멀티모달 기능을 향상시키기 위해 PubMedVision이라는 130만 개의 의료 VQA 샘플로 이루어진 고품질 데이터셋을 소개합니다. 이 데이터셋은 PubMed에서 얻은 의료 이미지와 텍스트 쌍을 GPT-4V를 사용하여 'unblinded' 재포맷팅을 통해 뛰어난 데이터 품질을 확보했습니다.

- **Technical Details**: 기존의 PubMed 데이터는 노이즈가 많아 MLLM의 성능이 떨어지는 문제가 있었습니다. 이를 해결하기 위해, 저자들은 914,960개의 고품질 의료 이미지와 그에 해당하는 텍스트를 선별하고, MLLM(GPT-4V)을 사용하여 데이터에서 노이즈를 제거하고 재포맷팅했습니다. 사용된 모델은 'unblinded' 접근 방식을 채택하여 이미지와 텍스트 간의 부정확한 정렬 문제를 해결했습니다.

- **Performance Highlights**: PubMedVision을 사용하여 훈련된 34B 파라미터의 HuatuoGPT-Vision 모델은 여러 의료 멀티모달 벤치마크에서 우수한 성능을 보였습니다. 특히, LLaVA-1.5-LLaMA-3-8B 모델은 MMMU Health & Medicine 트랙에서 단연 뛰어난 성능을 기록했습니다. 전문가들의 수동 검토와 실험 결과도 PubMedVision 데이터셋이 기존 데이터셋보다 더 높은 품질을 가졌음을 확인했습니다.



### Enhancing Video-Language Representations with Structural Spatio-Temporal Alignmen (https://arxiv.org/abs/2406.19255)
Comments:
          Accepted by IEEE TPAMI 2024

- **What's New**: 연구진은 기존의 비디오-언어 모델들(VLMs)이 가지는 문제점들을 해결하기 위해 Finsta라는 방법을 제안했습니다. Finsta는 미세한 구조적 시공간 정렬 학습 메서드로, 장면 그래프(scene graph, SG) 표현 방식을 사용하여 텍스트와 비디오의 호환성을 증진시킵니다. 이 방법은 기존 VLM에 플러그 앤 플레이 방식으로 통합될 수 있어, 새로운 모델을 처음부터 학습할 필요 없이 성능을 향상시킬 수 있습니다.

- **Technical Details**: 먼저 입력된 텍스트와 비디오를 장면 그래프(SG) 구조로 변환합니다. 그런 다음, 텍스트 SG(TSG)는 그래프 트랜스포머(graph Transformer, GTrm)를 통해 인코딩되고, 비디오의 동적 SG(DSG)와 총합 SG(HSG)는 새로운 순환 그래프 트랜스포머(recurrent graph Transformer, R-GTrm)로 공간적 및 시간적 특징을 전파합니다. 또한, 객체의 시공간적 변화를 감지하는 데 도움이 되는 공간-시간적 가우시안 차별 그래프 트랜스포머(spatial-temporal Gaussian differential graph Transformer, STGD-GTrm)를 제안합니다. 이 방법은 구조적 특징에 기반하여 객체 중심의 공간적 정렬과 술어 중심의 시간적 정렬을 수행합니다.

- **Performance Highlights**: Finsta는 12개의 데이터셋에서 6개의 대표적인 비디오-언어 모델링 작업들에서 테스트되었으며, 모든 케이스에서 기존의 강력한 성능을 보여주는 VLM들을 지속적으로 개선했습니다. 특히, 미세 조정(fine-tuning)과 제로샷(zero-shot) 설정 모두에서 새로운 최고 성능을 달성했습니다.



### Spiking Convolutional Neural Networks for Text Classification (https://arxiv.org/abs/2406.19230)
- **What's New**: 기존에 비해 매우 적은 연구가 수행된 언어 작업(NLP)에서의 spiking neural networks (SNNs)를 효율적으로 학습시키기 위한 '전환 + 미세 조정(conversion + fine-tuning)'이라는 두 단계 방법을 제안합니다. 또한, 사전 학습된 word embeddings를 spike trains로 인코딩하는 간단하면서도 효과적인 방법을 제안합니다.

- **Technical Details**: 기존에 일반적으로 사용되는 심층 신경망(DNN)을 SNN으로 변환한 후, surrogate gradients를 사용하여 미세 조정을 진행합니다. 이를 위해 원래의 TextCNN 모델을 CNN으로 변환하고 나서 스파이크 신경망을 통해 학습합니다. 변환 과정에서는 max-pooling을 average-pooling으로, Sigmoid 활성화 함수를 ReLU로 교체하며, word embeddings를 양수 값으로 전환합니다. 변환된 네트워크는 데이터를 gradient descent 알고리즘으로 학습한 후 spiking neural network로 전환되어 미세 조정됩니다.

- **Performance Highlights**: 제안된 SNN은 DNN과 비교해 다양한 데이터셋에서 유사한 결과를 기록하면서도 에너지 소비를 크게 줄일 수 있음을 보여줍니다. 영어와 중국어 텍스트 분류 작업에서 기존 DNN과 유사한 성능을 보였을 뿐만 아니라, 적대적 공격(adversarial attacks)에도 더 강인한 모습을 보였습니다.



### Resolving Discrepancies in Compute-Optimal Scaling of Language Models (https://arxiv.org/abs/2406.19146)
- **What's New**: 이번 연구에서는 Kaplan et al.와 Hoffmann et al. 사이의 모델 크기 및 연산 예산 간의 차이를 식별하고 해명합니다. 두 연구는 서로 다른 스케일링 법칙(scaling laws)을 제시하지만, 세 가지 주요 요소를 수정하여 일치하는 결과를 얻었습니다: 마지막 레이어의 연산 비용, 웜업(warmup) 기간, 그리고 규모에 따른 옵티마이저(Optimizer) 튜닝입니다.

- **Technical Details**: 1. 마지막 레이어 연산 비용: Hoffmann et al.에서는 이를 고려했으나 Kaplan et al.에서는 그렇지 않았습니다. 이를 수정하자 최적의 토큰 대 파라미터 비율이 더 일정하게 유지되었습니다.
2. 웜업 기간: Kaplan et al.의 설정은 작은 모델에 대해 너무 길어, 최적의 토큰 수를 늘렸습니다. 모델 크기에 따라 웜업 기간을 조정함으로써 Hoffmann et al. 방향으로 전환되었습니다.
3. 학습 속도 감소(learning rate decay): Hoffmann et al.의 가설대로 토큰 예산에 맞춰 설정했으나, 최적의 스케일링 법칙에 큰 영향은 없었습니다. 대신 AdamW β2 파라미터를 개별 모델 크기별로 조정하며 일치하는 결과를 얻었습니다.

- **Performance Highlights**: Kaplan et al.과 Hoffmann et al.의 차이를 해결하는 세 가지 요소를 도입한 후 손실이 작은 규모에서는 상당히 감소했으나, 큰 규모에서는 약간의 개선만 이루어졌습니다. 또한, 코사인 학습 속도 감소(cosine learning rate decay) 스케줄이 도입되자 손실이 크게 감소하여 큰 규모에서도 이점을 유지했습니다. 이러한 결과는 OpenWebText2 데이터셋에서 반복 실험해도 일관된 결과를 보였습니다. 우리의 데이터를 포함한 코드와 분석을 재현하기 위한 리소스는 GitHub에 공유합니다(https://github.com/formll/resolving-scaling-law-discrepencies).



### RoboUniView: Visual-Language Model with Unified View Representation for Robotic Manipulaiton (https://arxiv.org/abs/2406.18977)
- **What's New**: RoboUniView는 로봇 조작을 위해 비주얼-언어 모델(VLMs)의 시각적 특징 추출을 동작 학습과 분리함으로써 기존 방법론의 한계를 극복하는 혁신적인 접근법을 제안합니다. 이 모델은 다양한 시점에서 얻은 데이터를 이용해 통합된 뷰(Unified View) 표현을 학습하고, 이를 통해 로봇 동작을 제어합니다. 특히, 로봇 플랫폼의 카메라 매개변수에 구애받지 않고 물리적 세계를 더 정확하게 반영할 수 있습니다.

- **Technical Details**: RoboUniView는 다중 시점으로부터 학습된 통합 뷰 표현을 통해 시각적 특징을 추출하고, 이를 기반으로 로봇 조작 동작을 도출합니다. UVFormer라는 플러그-앤-플레이 형태의 플러그인을 활용하여 다중 시점 뷰와 해당 카메라 매개변수를 입력으로 받아 3D 그리드의 각 셀에 대한 점유 상태와 RGB 값을 출력하는 사전 학습을 수행합니다. 이를 통해 비싼 수동 레이블 없이도 물리적 세계에 대한 깊이 있는 이해를 얻을 수 있습니다.

- **Performance Highlights**: RoboUniView는 CALVIN 벤치마크에서 새로운 최고 성과를 달성하였습니다. $D 	o D$ 설정에서 성공률을 88.7%에서 96.2%로, $ABC 	o D$ 설정에서 82.4%에서 94.2%로 향상시켰습니다. 이 모델은 미지의 카메라 매개변수 하에서도 높은 성능을 유지하며, 다양한 카메라 매개변수를 갖춘 여러 데이터셋을 활용할 수 있고, 데이터셋 간의 크로스 작업 학습이 가능합니다.



### Applying LLMs for Rescoring N-best ASR Hypotheses of Casual Conversations: Effects of Domain Adaptation and Context Carry-over (https://arxiv.org/abs/2406.18972)
Comments:
          5 pages

- **What's New**: 이번 연구에서는 대형 언어 모델(Large Language Models, LLMs) 중 하나인 Llama2를 활용하여 CHiME-7 DASR 과제의 자동 음성 인식(ASR) 가설 재평가를 수행했습니다. 특히, LLMs가 일상 대화의 ASR 가설을 얼마나 효과적으로 재평가하는지에 대해 탐구했습니다. 실험 결과, 도메인 적응 없이도 Llama2가 표준 크기의 도메인 적응 Transformer-LM을 성능 면에서 초과 달성하였으며, 도메인 적응은 Llama2가 최상의 성능을 발휘하는 데 필요한 컨텍스트 길이를 줄여주어 계산 비용을 감소시킵니다.

- **Technical Details**: 본 연구에서는 Llama2-7B를 사용하여 N-best ASR 가설 재평가를 수행하였습니다. 컨텍스트 길이에 따라 Llama2의 성능 변화를 분석하였으며, 도메인 적응에는 QLoRA를 사용하여 메모리 효율적인 방법을 적용하였습니다. 실험은 최대 1024 토큰까지의 다양한 컨텍스트 길이를 고려하여 진행되었습니다. Llama2-7B와 비교를 위해 표준 크기의 Transformer-LM인 'Slama2-70M'도 사용되었습니다. 이는 Llama2-7B의 구성 요소를 다운사이징하여 구축한 것으로, 퍼플렉시티(perplexity)를 통해 두 모델을 공정하게 비교할 수 있었습니다.

- **Performance Highlights**: 실험 결과, 도메인 적응 없이도 Llama2는 매우 긴 컨텍스트(예: 1024 토큰)를 고려하여 대화의 흐름을 잘 포착해 최저 단어 오류율(WER)을 기록했습니다. 도메인 적응을 통해 Llama2가 최저 WER을 달성하는데 필요한 컨텍스트 길이를 줄여 계산 비용을 상당히 감소시켰습니다. 본 연구는 특히, 캐주얼하고 도전적인 대화 상황에서 LLMs의 ASR 가설 재평가 능력을 명확히 밝혔습니다.



### The single-use restriction for register automata and transducers over infinite alphabets (https://arxiv.org/abs/2406.18934)
Comments:
          PhD Thesis at University of Warsaw. Supervisor: Mikołaj Bojańczyk

- **What's New**: 본 논문에서는 무한 알파벳(over infinite alphabets)을 다루는 레지스터 오토마타(register automata) 및 변환기(transducers)의 단일 사용 제한(single-use restriction)을 연구합니다. 이 제한은 레지스터의 읽기 접근 시 해당 내용을 파괴하는 측면 효과를 갖도록 요구합니다. 이를 통해 강력한 클래스의 언어와 변환이 만들어집니다. 특히 단일 사용 Mealy 머신과 단일 사용 양방향 변환기가 Krohn-Rhodes 분해 정리(Krohn-Rhodes decomposition theorem)를 인정하는 버전을 제공함을 보여줍니다.

- **Technical Details**: 단일 사용 레지스터 오토마타 모델(single-use register automata models)에 대해 연구를 수행했습니다. 일방향 레지스터 오토마타(one-way register automata), 양방향 레지스터 오토마타(two-way register automata), 그리고 궤도 유한 모노이드(orbit-finite monoids)가 동일한 표현력을 갖는다는 것을 증명했습니다. 변환기 모델(transducer models)에서는 단일 사용 Mealy 머신과 단일 사용 양방향 변환기가 Krohn-Rhodes 분해 정리의 버전을 인정하는 것을 보여줍니다. 단일 사용 Mealy 머신은 로컬 대수적 반군 변환(local algebraic semigroup transductions)이라는 대수적 모델과 동등합니다. 또한, 단일 사용 양방향 변환기는 단일 사용 스트리밍 문자열 변환기(single-use streaming string transducers)와 무한 알파벳 상의 정규 리스트 함수와 동등합니다.

- **Performance Highlights**: 이전 작업(arXiv:1907.10504)과 비교하여 본 논문은 단일 사용 제한에 대한 일관된 담론을 제공합니다. 단일 사용 함수(single-use functions)의 추상 개념을 도입하고 이를 사용하여 논의된 모든 단일 사용 모델을 정의합니다. 또한, 로컬 반군 변환(local semigroup transduction) 및 로컬 합리적 반군 변환(local rational semigroup transduction)의 대수적 모델을 소개하고 연구합니다.



### Enhanced ASR Robustness to Packet Loss with a Front-End Adaptation Network (https://arxiv.org/abs/2406.18928)
Comments:
          Accepted for publication at INTERSPEECH 2024

- **What's New**: 이번 연구는 끊김 현상이 발생하는 소음 환경에서 자동 음성 인식(ASR) 모델의 단어 오류율(WER)을 개선하기 위한 새로운 방법을 제안합니다. Whisper와 같은 최신 ASR 모델의 성능을 유지하면서도, 프론트엔드 적응 네트워크를 추가해 끊김 현상을 보완하고 모델의 성능을 향상시킵니다.

- **Technical Details**: 제안된 방법은 고정된 ASR 모델에 프론트엔드 적응 모델을 연결하여 손상된 입력 스펙트럼을 보완합니다. 적응 네트워크는 U-net 아키텍처를 사용하며, ASR 모델의 기울기를 활용한 손실 함수와 향상 손실 함수를 최적화하면서 훈련됩니다. 적응 네트워크는 Whisper 모델의 기준을 사용해 훈련됨으로써, 다양한 도메인과 언어에서 단어 오류율(WER)을 크게 줄이는 데 성공했습니다.

- **Performance Highlights**: 다양한 실험을 통해 제안된 방법은 여러 도메인 및 언어에서 패킷 손실 시나리오에 대해 기본 모델보다 월등히 향상된 내구성을 증명했습니다. 또한, 적응 네트워크를 추가함으로써 Whisper 모델의 기본 성능에는 거의 영향을 미치지 않아 기존 기능을 해치지 않으면서도 성능을 유지하는 것이 확인되었습니다.



### DeSTA: Enhancing Speech Language Models through Descriptive Speech-Text Alignmen (https://arxiv.org/abs/2406.18871)
Comments:
          Accepted to Interspeech 2024

- **What's New**: 최근의 음성 언어 모델(SLM)은 대형 언어 모델(LLM)의 능력을 확장하기 위해 사전 훈련된 음성 모델을 포함하는 경향이 있습니다. 이번 논문에서는 음성과 텍스트 모달리티 사이의 격차를 해소하기 위해 음성 캡션을 활용한 설명적 음성-텍스트 정렬 접근법을 제안합니다. 이를 통해 SLM이 현상설명적 자연 언어를 해석하고 생성할 수 있게 하여, 음성의 언어적 및 비언어적 특징을 이해하도록 돕습니다.

- **Technical Details**: 우리의 모델은 사전 훈련된 Whisper 모델과 Llama2-chat 모델을 통합한 아키텍처를 사용합니다. 훈련 단계 동안 이 사전 훈련된 모델들은 변경되지 않습니다. 랜덤 초기화된 모달리티 어댑터가 Whisper 인코더의 음성 특징을 Llama의 입력 표현 공간으로 매핑합니다. 여기에 Low-Rank Adapters (LoRA)를 부착해 훈련 효율성을 높였습니다. 이 접근법은 기존 데이터셋에서 메타데이터를 수집하고 LLM을 활용해 다양한 글쓰기 스타일과 톤으로 음성 캡션을 생성합니다.

- **Performance Highlights**: 제안된 음성-텍스트 정렬 훈련을 통해, 우리의 미세조정 모델은 Dynamic-SUPERB 벤치마크에서 기존의 시스템보다 뛰어난 성능을 보였으며, 특히 훈련 단계에서 다루지 않았던 작업에서도 높은 범용성을 보였습니다. 또한, 제로샷(zero-shot)으로 지시를 따르는 능력이 시험 단계에서 나타났습니다.



### The global landscape of academic guidelines for generative AI and Large Language Models (https://arxiv.org/abs/2406.18842)
- **What's New**: 이 연구는 생성적 인공 지능(Generative Artificial Intelligence, GAI)과 대형 언어 모델(Large Language Models, LLMs)이 학문 분야에 미치는 영향을 이해하기 위한 체계적인 조사 및 텍스트 마이닝 기반 분석을 통해 전 세계 및 국가 차원의 지침을 다루었습니다. 이 과정에서 80개의 대학 수준 지침을 검토하여 GAI와 LLMs의 교육적 활용 가능성과 관련된 기회와 도전에 대해 깊이 있는 통찰을 제공했습니다.

- **Technical Details**: 연구는 생성적 인공 지능(GAI)과 대형 언어 모델(LLMs)의 교육적 활용도 및 윤리적 고려사항에 대한 체계적인 설문조사 및 텍스트 마이닝(text-mining) 기반 분석을 통해 수행되었습니다. 이 과정에서 독립 연구 자료와 80개의 대학 수준 지침을 분석하여 GAI와 LLMs의 잠재적 이점과 윤리적 도전 과제를 통합적으로 평가했습니다.

- **Performance Highlights**: 이 연구는 교육 분야에서 GAI와 LLMs가 협력적 창의성 증진, 교육 접근성 향상, 교육자와 학습자의 역량 강화 등 긍정적인 영향을 미칠 수 있음을 확인했습니다. 그러나 윤리적 복잡성, 혁신과 학문의 불균형, 접근성의 불평등, 잘못된 정보의 위험성 같은 부정적인 면도 존재함을 강조했습니다. 연구는 이러한 기술의 이점을 극대화하면서 윤리적 고려사항을 충족하고 공정한 접근성과 교육적 결과를 보장하는 균형 잡힌 접근이 필요하다고 결론지었습니다.



### Navigating LLM Ethics: Advancements, Challenges, and Future Directions (https://arxiv.org/abs/2406.18841)
- **What's New**: 이번 연구는 인공지능(AI) 분야에서 대형 언어 모델(LLMs, Large Language Models)과 관련된 윤리적 이슈를 다룹니다. LLMs와 다른 AI 시스템이 공통적으로 직면하는 개인정보 보호와 공정성과 같은 윤리적 도전뿐만 아니라, LLMs에서만 나타나는 고유한 윤리적 도전에 대해서도 탐구합니다. 이들 도전 중에서는 환각(hallucination), 검증 가능한 책임성(verifiable accountability), 그리고 디코딩 검열 복잡성(decoding censorship complexity)이 있습니다.

- **Technical Details**: 연구는 위와 같은 복잡성을 해결하기 위해 다양한 윤리적 프레임워크와 유동적인 감시 시스템을 제안합니다. 특정 도메인에 맞춤화된 윤리적 프레임워크와 다양한 상황에 적응 가능한 동적 감사 시스템이 필요하다고 강조합니다. 이를 통해 LLMs가 정보 확산에 미치는 영향을 투명하게 하여 책임성을 높이고 편견을 줄이며, AI 발전이 윤리적 고려 아래 이루어지게 합니다.

- **Performance Highlights**: 이 연구는 LLM 윤리를 위한 완화 전략 및 미래 방향성을 제안하며, 분야 간 협력(interdisciplinary collaboration)을 권장합니다. 이 로드맵은 LLMs의 책임감 있는 개발과 통합을 안내하며, AI의 발전이 사회 안에서 윤리적 고려 하에 이루어지는 미래를 상상합니다.



### WavRx: a Disease-Agnostic, Generalizable, and Privacy-Preserving Speech Health Diagnostic Mod (https://arxiv.org/abs/2406.18731)
Comments:
          Under review; Model script available at this https URL

- **What's New**: 새로운 논문에서는 음성을 통해 질병을 진단할 수 있는 새로운 모델인 WavRx를 소개하고 있습니다. 이 모델은 기존 모델들이 가진 질병 특화성과 데이터셋 간 일반화 부족 문제를 해결하며, 말하는 사람의 정체성이 유출되는 문제도 줄이기 위한 것입니다. WavRx는 호흡 및 조음(dynamics) 관련 특성을 통합하여 일반화 성능을 대폭 향상시킨 모델입니다.

- **Technical Details**: WavRx 모델은 유명한 WavLM 모델을 기반으로 하며, 고해상도 시간적 WavLM 표현과 장기적 변조 동역학(long-term modulation dynamics)을 결합한 새로운 변조 동역학 모듈을 포함하고 있습니다. 이 모듈은 질병 관련 음성 비정상 패턴을 더 잘 포착하기 위해 단기 시간적 변화를 보완하는 정보를 제공합니다.

- **Performance Highlights**: WavRx는 6개의 병리학적 음성 데이터셋에 대해 실험을 진행했으며, 4개의 데이터셋에서 최고의 성능을 달성했습니다. 또한, WavRx는 추가적인 학습 없이도 음성 임베딩에서 말하는 사람의 정체성이 포함되는 것을 크게 줄인다는 것을 입증했습니다. 이는 모델의 일반화 성능과 프라이버시 보호 능력이 향상된 것임을 시사합니다.



### Jailbreaking LLMs with Arabic Transliteration and Arabiz (https://arxiv.org/abs/2406.18725)
Comments:
          14 pages, 4 figures

- **What's New**: 이 연구에서는 '탈옥(jailbreak)' 공격에 대한 대형 언어 모델(LLM)의 잠재적 취약성을 아랍어와 다양한 형태를 중심으로 조사합니다. 대부분의 연구가 영어 기반 프롬프트 조작에 집중된 반면, 이번 연구는 아랍어로 범위를 확장하여 연구를 진행했습니다. 표준화된 아랍어로 AdvBench 벤치마크를 테스트한 결과, 프리픽스 인젝션(prefix injection) 같은 프롬프트 조작 기술을 사용하더라도 LLM이 안전하지 않은 콘텐츠를 생성하지 않았습니다. 그러나 아랍어 음역 및 채팅 언어(아랍이지, arabizi)를 사용했을 때 OpenAI GPT-4와 Anthropic Claude 3 Sonnet에서 안전하지 않은 콘텐츠가 생성될 수 있음을 발견했습니다.

- **Technical Details**: LLM들의 탈옥 공격 취약성을 조사하기 위해 표준 아랍어, 아랍어 음역 및 채팅 언어를 사용하여 프롬프트를 입력했습니다. 표준 아랍어 프롬프트에 대해서는, OpenAI GPT-4와 Anthropic Claude-3가 해로운 질문에 답변을 거부했습니다. 하지만, 아랍어 음역 및 채팅 언어로 입력된 프롬프트에 대해서는 모델이 안전하지 않은 답변을 제공하는 것이 관찰되었습니다. 이 연구는 특정 단어에 대한 모델의 학습된 연결이 이러한 공격에 대한 노출을 유발할 수 있다는 가설을 제시합니다.

- **Performance Highlights**: 우리의 수동 조사 결과, GPT-4와 Claude-3 LLM이 아랍어의 채팅 언어 및 음역 형태 프롬프트에 대해 취약하다는 사실을 확인했습니다. 특히, 채팅 언어와 음역 형태로 인해 표준 아랍어로 입력되었을 때 거부되었던 콘텐츠가 생성될 수 있는 점을 발견했습니다. 반대로, 표준 아랍어에서는 이러한 취약성이 나타나지 않았습니다.



### Learn it or Leave it: Module Composition and Pruning for Continual Learning (https://arxiv.org/abs/2406.18708)
- **What's New**: 새로운 논문 MoCL-P에서는 경량 연속 학습 (continual learning) 방법을 도입했습니다. 이 방법은 기존에 학습한 지식을 잊지 않으면서 새로운 지식을 점진적으로 습득할 수 있도록 설계되었습니다. 특히, MoCL-P는 태스크별 표현을 기반으로 모듈을 구성하고, 적응형 가지치기 (adaptive pruning) 전략을 사용하여 지식 통합과 계산 오버헤드를 균형있게 유지합니다.

- **Technical Details**: MoCL-P는 세 가지 주요 도전과제를 해결하고자 합니다: (1) 파국적 망각 (catastrophic forgetting) 방지, (2) 지식 전이 촉진, (3) 파라미터 효율성 유지. MoCL-P는 태스크별 모듈을 추가하여 새로운 태스크를 학습하며, 모듈을 동결하여 기존 태스크 지식을 보호합니다. 또한, 모듈 구성을 통해 기존 지식을 재사용할 수 있으며, 적응형 가지치기를 통해 중요하지 않은 모듈을 제거하여 모델의 무게를 유지합니다.

- **Performance Highlights**: MoCL-P는 연속 학습 벤치마크 세 가지에서 최대 176개의 태스크에 대해 평가되었습니다. 결과는 이전 알고리즘보다 최대 3배 더 높은 파라미터 효율성 및 최첨단 성능을 보여주었습니다. 이는 MoCL-P가 리소스가 제한된 실제 응용 프로그램에서 유망한 가능성을 보여준다는 것을 의미합니다.

- **Conclusion**: MoCL-P는 연속 학습의 세 가지 주요 도전과제를 동시에 해결한 첫 번째 방법으로, 파국적 망각을 방지하고 지식 전이를 허용하며 파라미터 효율성을 보장합니다. 코드베이스는 온라인에서 이용 가능합니다: (https://github.com/boschresearch/MoCL-Pruning)



### Simulating The U.S. Senate: An LLM-Driven Agent Approach to Modeling Legislative Behavior and Bipartisanship (https://arxiv.org/abs/2406.18702)
- **What's New**: 이 연구는 미국 상원 정보위원회를 중심으로 LLM 기반의 가상 에이전트를 활용한 입법 과정 시뮬레이션 도입을 소개합니다. 각 상원의원을 대표하는 에이전트를 개발하여 모의 위원회 토론에 배치했습니다. 이 에이전트들은 현실적인 토론을 진행하고, 사고를 반영하며, 특정 조건 하에서 초당적 해결책을 도출하는 능력을 보였습니다. 특히, 외부의 변동에 따라 초당적 협력으로 전환하는 모델링에도 잠재력을 보여줍니다. 이 연구는 입법 과정을 이해하고 개선하는 데 있어 LLM 기반 접근법의 가치를 제시하며, 미래 연구는 에이전트의 복잡성을 높이고 시뮬레이션 범위를 확장하는 데 초점을 맞출 예정입니다.

- **Technical Details**: 이 연구에서는 GPT-3.5 LLM 모델을 사용하여 각 상원의원을 대표하는 에이전트를 생성했습니다. 각 에이전트는 이름, 지지 정책, 주요 특성, 근무 연수 등의 속성을 가지고 있으며, 이전 상호작용의 컨텍스트를 기억하는 메모리 스트림을 활용해 토론을 진행했습니다. 시뮬레이션에서는 정치적 분극화와 초당적 협력 촉진을 탐구하기 위해 두 가지 주제를 다루었으며, 도메인 전문가들이 신뢰성 평가를 수행했습니다.

- **Performance Highlights**: 에이전트들은 높은 정확성과 깊이 있는 사고를 보여주었으며, 실제 상원 행동과 의사 결정 과정을 잘 모사했습니다. 특히, 러시아의 우크라이나 침공과 관련한 시나리오에서는 에이전트들의 협력 및 초당적 해결책 도출 능력이 돋보였습니다. 도메인 전문가들이 부여한 신뢰성 평가는 평균 5점을 초과했으며, Pearson 상관 검정을 통해 전문가 간 높은 일치도를 확인했습니다.



### Learning to Correct for QA Reasoning with Black-box LLMs (https://arxiv.org/abs/2406.18695)
Comments:
          preprint, 18 pages

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 추론 능력을 향상시키기 위한 새로운 접근법, CoBB(Correct for improving QA reasoning of Black-Box LLMs)을 제안합니다. 기존의 접근 방식들은 모델 매개변수 또는 출력 토큰 확률에 접근하는 데 의존하거나, 학습 및 추론 시간 비용이 높다는 제한이 있었습니다. 이를 해결하기 위해 CoBB는 원래의 블랙박스 LLM의 결점이 있는 추론을 올바르고 개선된 추론으로 변환하기 위한 seq2seq 매핑을 수행하는 적응 모델을 사용합니다.

- **Technical Details**: CoBB는 먼저 블랙박스 LLM에서 여러 chain-of-thought(reasonings)을 샘플링한 후, 그 정확성을 인간 라벨로 표시합니다. 모든 가능한 올바른 추론과 잘못된 추론 쌍 중에서 대표적인 쌍을 선별하기 위해, 전체 세트와의 통계적 발산을 최소화하는 최적화 문제를 유전 알고리즘을 통해 해결합니다. 이후 최적화된 서브셋을 사용하여 적응 모델을 학습시키며, 입력과 추론에 대해서 올바른 추론의 가능성을 증가시키고 잘못된 추론의 가능성을 줄입니다.

- **Performance Highlights**: 실험 결과, CoBB는 다양한 QA 벤치마크에서 기존 적응 기법들에 비해 추론 정확성이 상당히 개선되었습니다. 예를 들어, CoBB는 원래 블랙박스 GPT-3.5-turbo와 이전 최고 성능의 적응 방법과 비교해 평균적으로 각각 6.2%와 2.2%의 정확성 향상을 달성했습니다. 또한, 특정 블랙박스 LLM에 대해 학습된 적응 모델이 API 기반 및 오픈 소스 모델을 포함한 다른 LLM에도 일반화할 수 있음을 발견했습니다.



### Speakers Unembedded: Embedding-free Approach to Long-form Neural Diarization (https://arxiv.org/abs/2406.18679)
Comments:
          Accepted at INTERSPEECH 2024

- **What's New**: 이번 논문에서는 장시간 오디오와 다수의 화자에 대하여 별도의 화자 임베딩 없이 로컬 (local) 및 글로벌 (global) 단계에서 동시에 EEND (end-to-end neural diarization)를 적용하는 새로운 프레임워크를 제안하였습니다. 기존 기술들의 단점을 보완하면서도 성능 향상을 달성하여, Callhome American English 데이터셋에서 13%, RT03-CTS 데이터셋에서 10%의 DER 절감을 이룩하였습니다.

- **Technical Details**: 이 새로운 로컬-글로벌 EEND 방법은 로컬 EEND, 글로벌 EEND, 클러스터링(clustering)의 3단계로 이루어져 있습니다. 첫 번째 단계에서는 긴 오디오를 고정 크기의 윈도우로 나누고, 각 윈도우 내에서 EEND를 사용하여 다이어리제이션을 수행합니다. 두 번째 단계에서는 윈도우 간 화자 레이블 프리미테이션을 해결하기 위해 동일한 EEND 모델을 다시 적용하여 각 로컬 윈도우 간의 화자 유사성을 계산합니다. 마지막으로, 쌍별 화자 점수를 기반으로 어피니티 매트릭스(affinity matrix)를 생성하여 최종적으로 글로벌 화자 레이블링을 수행합니다.

- **Performance Highlights**: 제안된 프레임워크는 별도의 화자 임베딩을 필요로 하지 않으면서도, 기존의 단일 패스 EEND(conventional 1-pass EEND)와 비교하여 각각 Callhome American English 데이터셋에서 13%, RT03-CTS 데이터셋에서 10%의 DER 절감 효과를 보여줍니다. 또한, EEND-vector-clustering 방식에 비해 약간의 성능 향상을 이루었으며, 컴퓨팅 복잡성을 논의하고 처리 시간 단축을 위한 전략을 제안하였습니다.



### Few-shot Personalization of LLMs with Mis-aligned Responses (https://arxiv.org/abs/2406.18678)
Comments:
          preprint, 30 pages

- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)의 몇 샷 개인화(few-shot personalization)를 위한 새로운 접근 방식, Fermi를 제안합니다. Fermi는 LLM의 비정렬 응답(mis-aligned responses)을 활용하여 사용자 맞춤 프롬프트를 점진적으로 향상시키는 방식입니다. 이를 통해 제한된 사용자 정보와 이전 사용자 의견 몇 가지 예시를 바탕으로 사용자 개인화가 가능해집니다.

- **Technical Details**: Fermi는 세 가지 주요 단계를 통해 프롬프트를 최적화합니다: (1) 초기 또는 현재 프롬프트를 LLM으로 스코어링, (2) 높은 점수를 받은 프롬프트를 메모리에 <<<프롬프트, 점수, 컨텍스트>>> 삼중항으로 업데이트, (3) 업데이트된 메모리를 바탕으로 새로운 개선된 프롬프트 생성. 또한, 테스트 쿼리의 컨텍스트를 고려해 최적화된 개인화 프롬프트 중 하나를 선택적으로 사용하는 Retrieval-or-Prompt 방법을 제안합니다.

- **Performance Highlights**: Fermi는 여러 벤치마크에서 최상의 성능을 보이며, 이전 최고 성능을 보였던 휴리스틱 및 최적화 접근 방식과 비교해 두 개의 다중 선택 QA 데이터셋에서 평균 정확도가 각각 6.8%와 4.1% 개선되었습니다. 또한, 한 LLM에서 생성된 개인화 프롬프트가 다른 API 기반 및 오픈 소스 LLM에서도 효과가 있는 것으로 나타났습니다.



### Human-AI Collaborative Taxonomy Construction: A Case Study in Profession-Specific Writing Assistants (https://arxiv.org/abs/2406.18675)
Comments:
          Accepted to CHI 2024 In2Writing Workshop

- **What's New**: 대규모 언어 모델(LLMs)은 여러 글쓰기 작업을 돕는 데 효과적이지만, 비즈니스와 같은 특정 분야의 글쓰기 지원에서는 한계가 있습니다. 이를 해결하기 위해 LLM과 분야 전문가의 협력을 통해 도메인별 글쓰기 보조 도구의 가이드라인 역할을 할 수 있는 분류체계(taxonomy)를 개발하는 방법을 제안했습니다.

- **Technical Details**: 이 연구는 두 명의 마케팅 및 인사전문가를 대상으로 초기 형성 연구를 실시하여 LLM의 한계를 확인하였습니다. GPT-4의 출력물이 도메인 특유의 기대에 부합하지 않은 문제를 해결하기 위해, 'human-AI collaborative taxonomy construction' 접근법을 제안했고, 이는 다단계 피드백과 전문가와의 다양한 상호작용을 통해 분류체계를 계층적으로 생성하는 방식입니다.

- **Performance Highlights**: 추가 실험을 통해 이 방법론의 유효성을 검증하여, 다양한 이해당사자의 요구를 만족할 수 있도록 LLM이 주도하는 글쓰기 도움을 향상시킬 예정입니다. 이를 통해 GPT-4뿐 아니라 오픈 소스 모델(Mistral, LLaMA, OLMo 등)을 활용한 실험과 전문가 검증을 거쳐 신뢰성과 정확성을 높일 계획입니다.



### RouteLLM: Learning to Route LLMs with Preference Data (https://arxiv.org/abs/2406.18665)
- **What's New**: 최신 대형 언어 모델(LLMs)의 성능과 비용 간의 균형을 맞추기 위해, 새로운 효율적 라우터 모델을 제안했습니다. 이 모델은 강력한 LLM과 약한 LLM을 동적으로 선택함으로써 비용과 응답 품질 간의 최적의 균형을 목표로 합니다. 인간의 선호 데이터를 활용한 훈련 프레임워크를 개발하여 성능을 향상시켰습니다.

- **Technical Details**: 이 연구에서는 LLM 라우터 모델이 사용자의 쿼리를 강한 모델 또는 약한 모델로 라우팅하는 방법을 소개합니다. 이를 위해 인간 선호 데이터와 데이터 증강 기법을 사용한 훈련 프레임워크를 개발했습니다. 라우터 모델의 학습은 Chatbot Arena의 인간 선호 레이블과 여러 라우터 아키텍처를 탐구하여 진행되었습니다.

- **Performance Highlights**: 우리가 제안한 라우터 모델은 널리 인정된 벤치마크(MMLU와 MT Bench)에서 비용을 두 배 이상 절감하면서도 응답 품질을 유지하는 성과를 보였습니다. 특히, 새로운 모델이 도입되더라도 높은 성능을 유지하는 전이 학습 능력을 보여주었습니다.



### Step-DPO: Step-wise Preference Optimization for Long-chain Reasoning of LLMs (https://arxiv.org/abs/2406.18629)
Comments:
          Code, data, and models are available at this https URL

- **What's New**: 수학적 추론에서 큰 언어 모델(LLMs)의 정확성을 높이기 위해 인간 피드백으로부터 배우는 새롭고 단순한 방법인 Step-DPO가 제안되었습니다. 이 방법은 기존의 직선 선호도 최적화(Direct Preference Optimization, DPO)가 긴 체인의 수학적 추론에서는 효과가 떨어지는 문제를 해결하고자 합니다.

- **Technical Details**: Step-DPO는 완전한 답을 평가하는 기존 방법과 달리, 각 중간 추론 단계(individual reasoning steps)를 최적화 단위로 여깁니다. 이는 문제를 주고, 여러 초기 추론 단계를 거쳐, 정확한 단계와 오류 단계를 식별하고 최적화하는 방식을 채택합니다. 이를 위해 고품질 데이터를 수집하는 파이프라인을 개발하여, 약 10K의 단계별 선호도 데이터 쌍을 생성합니다. 이 데이터는 모델이 자체적으로 생성한 것이며, 이는 인간 또는 GPT-4가 생성한 데이터보다 성능이 뛰어납니다.

- **Performance Highlights**: Step-DPO로 튜닝된 Qwen2-72B-Instruct 모델은 MATH 데이터셋에서 70.8% 그리고 GSM8K 테스트셋에서 94.0%의 정확도를 기록했습니다. 이는 GPT-4-1106, Claude-3-Opus, Gemini-1.5-Pro 등 여러 폐쇄형 모델을 능가하는 성과입니다. 우리 코드, 데이터, 모델은 해당 URL에서 공개되어 있습니다.



### An LLM-based Knowledge Synthesis and Scientific Reasoning Framework for Biomedical Discovery (https://arxiv.org/abs/2406.18626)
Comments:
          accepted for ACL 2024 System Demonstration Track

- **What's New**: BioLunar라는 생명과학 분석 지원 도구가 Lunar 프레임워크를 이용하여 개발되었습니다. 이 툴은 주로 암 연구에서 바이오마커 발견을 목표로 하며, 분자 수준의 증거 강화를 강조합니다. 대규모 언어 모델(LLMs)을 통합하여 분산된 증거 공간에서 복잡한 과학적 추론을 지원하며, 이질적인 데이터 소스를 조화롭게 통합하고 추론하는 능력을 향상시킵니다.

- **Technical Details**: BioLunar는 표준화된 API를 사용하는 소프트웨어 구성 요소를 이용하여 LLM 기반의 생명과학 워크플로우를 생성할 수 있습니다. 이 플랫폼은 모듈식 설계를 통해 재사용 가능한 데이터 접근 및 분석 구성 요소를 제공하며, 저코드 사용자 인터페이스를 통해 프로그래밍 지식 없이도 LLM을 활용한 과학적 워크플로우를 구축할 수 있게 합니다. 사용자는 Python Coder 또는 R Coder 구성 요소를 통해 맞춤형 메소드를 정의할 수 있습니다.

- **Performance Highlights**: BioLunar는 정밀 종양학 및 바이오마커 발견을 위한 주요 지식 베이스 (CIViC, OncoKB, Gene Ontology 등)를 통합하여 유전자 돌연변이를 분석합니다. NGS (Next-generation sequencing) 결과 해석을 자동화하여 오류를 줄이고, 유전자 및 변이 강화 분석을 통해 종양 및 환자를 정밀하게 특성화합니다. 사용자 인터페이스는 시각적 인터페이스를 통해 워크플로우의 구성을 쉽게 조정할 수 있게 해줍니다.



### Towards Large Language Model Aided Program Refinemen (https://arxiv.org/abs/2406.18616)
- **What's New**: LLM4PR은 대형 언어 모델(LLMs) 및 프로그램 정제(refinement) 기술을 결합한 새로운 접근 방식을 제안합니다. 이 도구는 명세(specification)를 전제 조건(preconditions) 및 후제 조건(postconditions)으로 변환하고, 정제 계산법(refinement calculus)을 기반으로 자동 프롬프트를 생성하며, LLM과 상호 작용하여 코드를 생성한 후 생성된 코드가 정제 계산법의 조건을 만족하는지 검증합니다. 이를 통해 정확한 코드 생성을 보장합니다.

- **Technical Details**: LLM4PR은 GPT-4, Coq, Coqhammer를 활용하여 구현되었습니다. 이 도구는 다음의 단계를 진행합니다: (1) 명세를 전제 조건 및 후제 조건으로 변환, (2) 정제 계산법을 기반으로 자동 프롬프트 작성, (3) LLM과 상호 작용하여 코드 생성, (4) 생성된 코드가 정제 계산법 조건을 만족하는지 검증. 이는 HumanEval 및 EvalPlus 데이터셋에서 평가되었습니다.

- **Performance Highlights**: LLM4PR은 대부분 자동화된 접근 방식을 통해 코드의 정확성을 보장합니다. GPT-4와 Coq을 결합하여 명세와 법칙에 기반한 코드를 생성하고, ATP(Automated Theorem Prover)를 사용해 생성된 코드를 검증합니다. LLM을 '제약 해결자(constraint solvers)'로 간주하여, 디버깅을 돕는 '제약 확인(assert)' 과정과 코드 생성에 유익한 '제약 검증(verify)' 과정을 포함합니다.



### Self-Supervised Time-Series Anomaly Detection Using Learnable Data Augmentation (https://arxiv.org/abs/2406.12260)
Comments:
          11 pages, 4 figures, IEEE Transactions on Emerging Topics in Computational Intelligence

- **What's New**: LATAD(Learnable Data Augmentation-based Time-Series Anomaly Detection)라는 새로운 기법을 제안합니다. 이 기술은 자기 지도 학습(self-supervised learning) 방식으로 훈련되며, 시계열 데이터에서 구별할 수 있는 특징을 추출합니다. LATAD는 학습 효율성을 높이기 위해 학습 가능한 데이터 증강을 통해 도전적인 음성 샘플을 생성합니다.

- **Technical Details**: LATAD는 대조 학습(contrastive learning)을 통해 시계열 데이터에서 구별 가능한 특징을 추출합니다. 학습 가능한 데이터 증강(data augmentation)은 도전적인 음성 샘플을 생성하며, 이러한 방식으로 모델은 입력 데이터와 양성 예제(positive examples) 간의 상호 정보를 최대화하고 음성 예제(negative examples)와의 상호 정보를 최소화하는 높은 차원의 표현을 학습하게 됩니다. Triplet margin loss를 사용하여 양성 샘플을 가까이 당기고 학습 가능한 신경망으로 증강된 음성 샘플에서는 멀리 떨어지도록 합니다.

- **Performance Highlights**: LATAD는 여러 벤치마크 데이터셋에서 최신 연구와 비교하여 동일하거나 향상된 성능을 보였습니다. 또한, 모델의 결정을 설명하고 이상을 진단하는 데 도움이 되는 그래디언트 기반의 해석 방법을 제공합니다.



### EVALALIGN: Supervised Fine-Tuning Multimodal LLMs with Human-Aligned Data for Evaluating Text-to-Image Models (https://arxiv.org/abs/2406.16562)
Comments:
          Github Repository: this https URL

- **What's New**: 최근 텍스트-이미지 생성 모델(text-to-image generative models) 분야에서 주목할 만한 발전이 이루어진 반면, 이러한 모델의 성능을 정확히 반영하는 평가 척도(evaluation metrics)는 부족한 상황입니다. 특히, 모델 최적화를 유도할 수 있는 세분화된(fine-grained) 척도가 부족합니다. 본 논문에서는 고정밀도, 안정성 및 세밀한 분석을 특징으로 하는 평가 척도인 EvalAlign을 제안합니다. 우리의 접근법은 대규모 데이터셋으로 사전 학습된 다중 모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 능력을 활용합니다.

- **Technical Details**: EvalAlign은 이미치 정확성과 텍스트-이미지 정렬 두 가지 주요 차원에 초점을 맞춘 평가 프로토콜(evaluation protocols)을 개발합니다. 각 프로토콜은 특정 채점 옵션과 연결된 세부적인 지침 집합으로 구성되어 있어, 생성된 이미지를 정밀하게 수동으로 점수 매길 수 있습니다. 또한, 인간의 평가 판단과 일치하도록 MLLM을 감독 학습(Supervised Fine-Tune, SFT)하여 강력한 평가 모델을 만듭니다. 우리 데이터셋은 다수의 데이터 소스에서 파생되었고, 철저히 청소 및 체계적으로 인간에 의해 주석이 달려 있어 텍스트-이미지 정렬과 이미지 정확성을 정밀하게 평가할 수 있습니다.

- **Performance Highlights**: 24개의 텍스트-이미지 생성 모델에 대한 종합 테스트 결과, EvalAlign은 기존 척도보다 우수한 안정성과 더불어 인간의 선호와 높은 일치를 보입니다. 기존 평가 모델은 인간 평가와의 불일치, 강화된 모델 매개변수의 사용 등 여러 한계를 가지고 있었지만, EvalAlign은 저비용, 고정밀, 효율적인 모델 평가를 제공하며 세분화된 해석 가능한 척도를 통해 모델 최적화 방향성을 제시할 수 있습니다.



New uploads on arXiv(cs.IR)

### Which Neurons Matter in IR? Applying Integrated Gradients-based Methods to Understand Cross-Encoders (https://arxiv.org/abs/2406.19309)
Comments:
          Accepted at ICTIR 2024

- **What's New**: 최근 Retrieval-Augmented Generation (RAG)의 추가로 정보 검색(IR)의 범위와 중요성이 확장되었습니다. 이에 따라 IR 모델에 대한 깊은 이해가 필요해졌습니다. 본 논문에서는 통합 기울기(Integrated Gradient) 기반 방법을 IR 맥락에 적용하여 모델 내 개별 뉴런의 역할을 식별하는 가능성을 탐색합니다. 특히 'relevance' 뉴런의 역할과 보지 못한 데이터(unseen data)를 어떻게 처리하는지에 대해 새로운 통찰을 제공합니다.

- **Technical Details**: 이 연구는 통합 기울기(Integrated Gradient, Sundararajan et al., 2017) 방법을 사용하여 MonoBERT (Nogueira and Cho, 2019)와 같은 교차 인코더 모델의 뉴런이 IR 작업에서 수행하는 역할을 이해하려고 합니다. 연구 질문으로는 특정 쿼리에 대해 'relevant' 또는 'non-relevant'로 분류하는 뉴런 식별 가능성, in-domain과 out-of-domain 데이터와 관련된 뉴런 식별 가능성, 이러한 뉴런의 중요성을 다룹니다.

- **Performance Highlights**: 뉴런이 IR 작업에서 얼마나 중요한지에 대한 심층적 프루닝 연구를 수행하여 발견된 내용을 검증합니다. 이를 통해 IR 모델의 작동 방식을 이해하고 새로운 시스템을 설계하는 데에 기여하고자 합니다.



### Grounded and Transparent Response Generation for Conversational Information-Seeking Systems (https://arxiv.org/abs/2406.19281)
Comments:
          Proceedings of the 17th ACM International Conference on Web Search and Data Mining (WSDM '24), 2024

- **What's New**: 이 연구는 대화형 정보-탐색 시스템(CIS)에서 패시지 검색, 재정렬 및 쿼리 재작성 이상으로, 검색된 정보를 바탕으로 일관된 응답을 생성하는 복잡성을 탐구합니다. 특히, 정보가 포함된 응답 생성을 중점으로 시스템의 한계를 사용자에게 투명하게 알리는 방법을 제안합니다.

- **Technical Details**: 이 연구는 정보 누적(정보 nugget) 추출, 잘못된 응답 자동 감지, 시스템의 신뢰성과 한계를 나타내는 응답 생성을 주제로 다룹니다. 연구 질문(RQ1)으로는 잘못된 또는 불완전한 응답을 유발하는 요소 감지가 있고, RQ2로는 시스템의 신뢰성과 한계를 명확히 표현하는 응답 생성 방법이 있습니다. 정보 nugget 추출을 위해 CAsT-snippets 데이터셋을 사용하고 있으며, 문장 및 패시지 수준에서응답 가능성을 평가합니다.

- **Performance Highlights**: 제안된 방법론은 패시지 검색을 넘어 정보 누적을 기반으로 패시지를 기초로 응답을 구성하고, 시스템의 신뢰성과 한계를 명확히 전달합니다. 이를 통해 사용자에게 신뢰성 있는 정보 제공을 목표로 합니다.



### Efficient course recommendations with T5-based ranking and summarization (https://arxiv.org/abs/2406.19018)
Comments:
          ReNeuIR 2024 (at SIGIR 2024) - 3rd Workshop on Reaching Efficiency in Neural Information Retrieval, 18 July, 2024, Washington D.C, USA

- **What's New**: 본 논문에서는 스킬-직업 쌍에 대한 강좌를 추천하는 BrightFit 시스템을 위한 2단계 검색 파이프라인을 구현하고 평가했습니다. 주로 BM25를 사용하던 기존 시스템을 보완하여 RankT5 모델을 이용한 재랭킹을 도입했으며, 강좌 설명을 요약하는 두 가지 모델인 LongT5와 Vicuna를 비교했습니다. 또한, 모델의 크기를 줄이고 추론 속도를 높이기 위해 양자화(quantization)를 실험했습니다.

- **Technical Details**: 제안된 2단계 검색 파이프라인은 두 개의 첫 번째 단계 검색자 중 하나(밀집 검색자 GTR 또는 BM25)와 재랭커 RankT5를 결합합니다. LongT5는 긴 강좌 설명을 짧은 형식으로 요약하도록 미세 조정(finetuning)되었으며, Vicuna는 제로샷(in-context learning) 방식으로 사용되었습니다. 모델 추론 속도를 높이기 위해 RankT5 모델에 대한 양자화를 조사했습니다. 실험은 2500개 이상의 쿼리-문서 쌍에 대해 3단계로 관련성을 평가한 두 개의 새로 라벨링된 데이터셋에서 진행되었습니다.

- **Performance Highlights**: 두 개의 새로 라벨링된 데이터셋에서 제안된 2단계 랭킹 시스템은 기존 시스템(BM25)보다 성능이 크게 향상되었습니다. nDCG@10 점수가 0.482에서 0.684로, 다른 데이터셋에서는 0.447에서 0.844로 향상되었습니다. 또한, 양자화된 RankT5를 사용하여 추론 속도가 40% 증가했습니다. 사용자 설문에서는 개선된 랭킹 품질이 확인되었으나, A/B 테스트에서는 BM25가 클릭률에서 더 높은 점수를 받았습니다.



### Towards a Formal Characterization of User Simulation Objectives in Conversational Information Access (https://arxiv.org/abs/2406.19007)
Comments:
          Proceedings of the 2024 ACM SIGIR International Conference on the Theory of Information Retrieval (ICTIR '24), July 13, 2024, Washington DC, DC, USA

- **What's New**: 사용자 시뮬레이션(user simulation)은 대화형 정보 접근(CIA) 에이전트를 자동으로 훈련 및 평가하는 유망한 방법으로, 합성 대화를 생성하고 대규모의 재현 가능한 실험을 가능하게 합니다. 그러나 이러한 시뮬레이터를 위한 목적이 명확히 정의되지 않아 효과적인 시뮬레이터 개발을 방해하고 있습니다. 본 연구는 사용자 시뮬레이터의 목적을 훈련과 평가로 명확히 구분하며, 각 목적에 맞춰 시뮬레이터를 평가하는 구체적인 기준을 제시합니다.

- **Technical Details**: 대화형 정보 접근(CIA) 에이전트는 정보 검색을 목적으로 사용자가 여러 번의 대화를 통해 질문을 정교하게 다듬고 관련 정보를 얻는 방식을 채택합니다. 훈련의 목적은 실제 사용자와의 행동 유사성을 극대화하는 것이며, 평가는 실제 사용자의 성과를 정확히 예측하는 것에 중점을 둡니다. 이러한 두 가지 목적을 최적화하기 위해 서로 다른 디자인 고려 사항이 필요합니다.

- **Performance Highlights**: 본 연구는 초기 실험을 통해 하나의 목적(훈련 또는 평가)을 최적화하는 것이 다른 목적의 성능 향상으로 이어지지 않음을 보였습니다. 예를 들어, 시뮬레이터 A가 훈련 목적에서는 시뮬레이터 B보다 우수하지만, 평가 목적에서는 반대의 결과를 나타냈습니다. 이는 훈련과 평가의 각기 다른 목적을 달성하기 위한 맞춤형 디자인 필요성을 강조합니다.



### Amplify Graph Learning for Recommendation via Sparsity Completion (https://arxiv.org/abs/2406.18984)
- **What's New**: 이번 논문에서는 CF(협업 필터링) 추천 시스템을 개선하기 위해 AGL-SC(Amplify Graph Learning framework based on Sparsity Completion)을 제안합니다. 기존 방법들은 무작위로 데이터를 채워 노이즈를 통제하지 못하고, 높은 차수의 상호작용 특징을 깊이 있게 탐구하지 못하여 그래프 표현의 편향을 초래했습니다. 반면, 우리의 방법은 그래프 구조 보완을 통해 노드 표현 최적화를 목표로 합니다.

- **Technical Details**: AGL-SC는 세 가지 주요 단계로 구성됩니다. 첫째, 그래프 신경망(GNN)을 활용해 사용자와 아이템 노드 간의 직접 상호작용 특징을 추출합니다. 둘째, 특수화된 행렬 분해 방법으로 높은 차원의 상호작용 특징을 마이닝하여 숨겨진 층의 잠재 공간에서 생성적 향상을 촉진합니다. 셋째, 변이 추론(variational inference)을 통해 다차원 특징을 통합하여 누락된 그래프 구조를 보완 및 향상합니다.

- **Performance Highlights**: 네 가지 실제 데이터셋에서 벤치마크와 전략 실험을 수행한 결과, AGL-SC는 최첨단 방법들을 크게 상회하는 성능을 보였습니다. 이러한 결과는 AGL-SC가 데이터 희소성과 노이즈로 인한 표현 편향 문제를 효과적으로 해결하고, 보다 정확하고 포괄적인 사용자 추천을 제공한다는 것을 증명합니다.



### Multi-modal Food Recommendation using Clustering and Self-supervised Learning (https://arxiv.org/abs/2406.18962)
Comments:
          Working paper

- **What's New**: 새로운 연구는 사용자의 식이 요구에 맞춘 개인 맞춤형 음식 추천 시스템(CLUSSL)을 소개합니다. 이 시스템은 클러스터링(clustering)과 자가 지도 학습(self-supervised learning)을 사용하여 여러 모달리티의 데이터를 효과적으로 활용합니다. 기존 방법이 식별자(ID) 특징에 주로 의존하면서 다중 모달리티 정보를 보조적으로 사용하는 경향이 있는 반면, CLUSSL은 모달리티 간의 잠재적인 의미론적 관계를 명확하게 모델링합니다.

- **Technical Details**: CLUSSL은 두 단계를 통해 작업을 수행합니다. 첫 번째 단계에서는 비지도 학습을 사용하여 단일 모달리티 데이터(예: 이미지 특징, 텍스트 임베딩)를 클러스터링합니다. 이러한 클러스터들은 각 모달리티 내의 주요 의미론적 특징을 요약하는 '프로토타입 노드' 역할을 합니다. 이후 각 모달리티에 대해 모달리티별 그래프를 생성하여 이러한 프로토타입 노드 간의 관계를 캡처합니다. 이러한 그래프 생성은 데이터의 고유 구조를 인코딩하기 위해 ID 기능을 활용합니다. 그래프 합성 신경망(GCN)을 사용하여 이 의미론적 관계를 효과적으로 전파 및 집계합니다. 또한, CLUSSL은 자가 지도 학습 프레임워크 내에서 거리 상관 제약을 통합하여 다른 모달리티에서 추출된 레시피 표현이 일정 수준의 독립성을 유지하도록 합니다.

- **Performance Highlights**: 광범위한 실제 데이터 세트 실험을 통해 CLUSSL이 기존 최첨단 추천 기준을 지속적으로 능가하는 것을 확인했습니다. 이러한 실험은 CLUSSL이 단일 모달리티 데이터와 다중 모달리티 관계의 강점을 잘 활용하여 추천 정확도를 크게 향상시킨다는 것을 입증했습니다.



### A Surprisingly Simple yet Effective Multi-Query Rewriting Method for Conversational Passage Retrieva (https://arxiv.org/abs/2406.18960)
Comments:
          Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval

- **What's New**: 이번 논문에서는 대화형 구절 검색(conversational passage retrieval)의 성능을 향상시키기 위해 다중 쿼리 리라이트(multiple query rewrites)를 사용하여 검색 성능을 향상시키는 방법을 제안합니다. 특히, Beam Search 알고리즘을 활용하여 추가 비용 없이 다중 쿼리 리라이트를 생성하는 방법을 소개합니다. 이러한 접근 방식을 통해 희소 및 조밀 첫번째 검색(sparse and dense first-pass retrieval) 모두에 다중 쿼리 리라이트를 효과적으로 통합할 수 있습니다.

- **Technical Details**: 제안된 방법은 Beam Search 알고리즘을 사용하여 다중 쿼리 리라이트를 생성합니다. 이를 위해 다양한 시퀀스를 추적하고, 각 시퀀스의 확률을 계산하여 다중 리라이트를 생성합니다. 희소 검색(sparse retrieval)에서는 쿼리 확장을 위해 다중 리라이트를 사용하며, 조밀 검색(dense retrieval)에서는 다중 리라이트를 단일 벡터 표현으로 병합하여 검색 효율성을 유지합니다.

- **Performance Highlights**: 제안된 방법은 희소 검색에서 기존의 단일 쿼리 리라이트(single query rewrite) 대비 MRR을 1.06-6.31%포인트 향상시켰으며, 조밀 검색에서는 절대 MRR 점수로 3.52-4.45%포인트 향상시켰습니다. QReCC 데이터셋을 사용한 평가에서, 제안된 방법론은 생성적 쿼리 리라이트(generative QR)를 사용하는 어떠한 파이프라인에도 적용되어 상태-of-the-art의 성능을 달성했습니다.



### Towards Personalized Federated Multi-scenario Multi-task Recommendation (https://arxiv.org/abs/2406.18938)
- **What's New**: 최신 연구에서는 개인화된 연합 학습 기반 다중 시나리오 다중 태스크 추천 시스템(PF-MSMTrec)을 제안했습니다. 기존의 다중 태스크 추천 시스템이 다양한 비즈니스 시나리오를 다루고 있는 것을 확장해, 데이터 프라이버시를 유지하면서 여러 독립적인 시나리오를 처리할 수 있는 글로벌 모델 훈련의 어려움을 극복하고자 합니다.

- **Technical Details**: PF-MSMTrec은 각 시나리오를 전담 클라이언트에 할당하고, 각 클라이언트가 Mixture-of-Experts(MMoE) 구조를 활용합니다. 이 방법은 전문가 네트워크의 파라미터를 분리하여 시나리오 간 파라미터 공유와 태스크별 개인화된 로컬 파라미터를 구현합니다. 또한 연합 배치 정규화(federated batch normalization), 분쟁 조정(conflict coordination), 개인화된 집계(personalized aggregation) 등 세 가지 모듈을 통해 연합 학습을 수행합니다.

- **Performance Highlights**: 두 개의 공개 데이터셋에서 광범위한 실험을 통해 제안된 방법이 기존의 최첨단(state-of-the-art, SOTA) 방식들을 능가하는 성능을 보였습니다. 특히, 연합 학습 환경에서도 비연합 SOTA 메서드보다 우수한 성능을 나타냈습니다.



### ELCoRec: Enhance Language Understanding with Co-Propagation of Numerical and Categorical Features for Recommendation (https://arxiv.org/abs/2406.18825)
- **What's New**: 이번 연구에서는 ELCoRec를 소개합니다. 이는 추천 시스템(Recommendation System)에서 언어 이해 능력을 향상시키기 위해 수치적 특징과 범주형 특징의 동시 전파(Co-Propagation) 방법을 제안합니다. 기존 대형 언어 모델(LLM)이 수치적 정보 이해나 긴 문맥 처리에서 어려움을 겪던 문제를 해결하기 위한 접근법입니다.

- **Technical Details**: ELCoRec는 GAT(Graph Attention Network) 전문가 모델을 사용하여 사용자의 선호를 더 정확히 인코딩합니다. 이것은 시간적 관계와 평점 신호, 그리고 아이템의 다양한 부가 정보를 병렬 전파시켜 달성됩니다. 이 인코딩된 정보를 소프트 프로म्प팅(soft prompting)을 통해 LLM의 의미 공간에 주입합니다. 또한, 사용자의 최신 관심사를 반영하기 위해 새로운 RAP(Recent interaction Augmented Prompt) 템플릿을 제안하였습니다.

- **Performance Highlights**: 세 개의 공개 데이터셋에서 ELCoRec의 효과를 입증하기 위한 실험을 수행한 결과, 기존 강력한 베이스라인과 비교해 우수한 성능을 보였습니다. 특히, 수치적 정보 이해와 인코딩 오버헤드 문제를 동시에 해결하여 추천 성능을 크게 개선할 수 있음을 확인했습니다.



### FlowVQA: Mapping Multimodal Logic in Visual Question Answering with Flowcharts (https://arxiv.org/abs/2406.19237)
- **What's New**: FlowVQA는 최신 시각 질문 응답(multimodal language models) 평가 벤치마크로, 복잡한 흐름도(flowcharts)를 기반으로 한 논리적 추론 능력을 평가합니다. 이는 시각적 근거와 복잡성을 더욱 강화하여, 정보 위치 확인, 의사 결정, 논리적 진행 등 다양한 추론 과제를 평가합니다.

- **Technical Details**: FlowVQA는 총 2,272개의 흐름도 이미지를 포함하며, WikiHow와 Instructables 같은 소스에서 수집된 자료를 바탕으로 생성되었습니다. 또한, 22,413개의 다양한 질문-답변 쌍이 포함되어 있으며, 정보 위치 확인, 사실 찾기, 시나리오 유추, 흐름 추론 및 위상 이해와 같은 여러 추론 기술을 테스트합니다. 데이터 생성 과정은 다단계 기계 생성 및 인간 검증을 포함하며, 이는 복잡하지만 체계적인 흐름도 제공을 목적으로 합니다.

- **Performance Highlights**: FlowVQA를 사용한 여러 개방형 및 독점형 VLMs의 벤치마크 평가 결과, 모델들이 FlowVQA 데이터셋에서 시각적 및 공간적 추론 작업을 수행하는 데 어려움을 겪는 것을 확인했습니다. 또한 방향성 편향 및 다양한 길이의 흐름도에 대한 불균형적인 성능 패턴이 드러났습니다. 이로써 FlowVQA는 멀티모달 모델링 분야를 발전시킬 중요한 도구로 자리 잡을 가능성을 보여줍니다.



### RAVEN: Multitask Retrieval Augmented Vision-Language Learning (https://arxiv.org/abs/2406.19150)
- **What's New**: 이번 논문에서는 Vision-Language Models (VLMs)를 위한 멀티태스크 Retrieval-Augmented Generation (RAG) 프레임워크인 RAVEN을 소개합니다. RAVEN은 효율적인 태스크 별 파인튜닝을 통해 기본 VLM을 향상시키며, 추가적인 Retrieval-specific 파라미터 없이 다중 태스크에 효과적인 성능을 보여줍니다.

- **Technical Details**: 기존의 RAG 기반 접근법은 단일 태스크에 초점을 맞추고 있으며, 리소스 집약적인 사전 학습이 필요합니다. RAVEN은 이러한 문제를 해결하고 다양한 태스크에 적용 가능한 프레임워크를 제공합니다. 본 연구는 텍스트, 이미지 혹은 두 가지 모달리티의 리트리벌을 통해 성능 향상을 달성할 수 있는지에 대한 포괄적인 분석을 실시했습니다.

- **Performance Highlights**: RAVEN은 이미지 캡셔닝과 VQA 태스크에서 비retrieval 기반 모델 대비 현저한 성능 향상을 보여줍니다. MSCOCO 데이터셋에서 +1 CIDEr, NoCaps에서 +4 CIDEr, 특정 VQA 질문 유형에서 3%의 정확도 향상을 달성했습니다. 이는 RAG 접근법을 VLM에 적용했을 때의 효용성을 잘 보여줍니다.



### Statements: Universal Information Extraction from Tables with Large Language Models for ESG KPIs (https://arxiv.org/abs/2406.19102)
Comments:
          Accepted at the NLP4Climate workshop in the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024)

- **What's New**: 이 논문에서는 환경, 사회, 지배구조(ESG) 보고서 내의 표에서 정량적 사실과 관련 정보를 추출하기 위한 새로운 도메인 비특정 데이터 구조인 'Statement'를 제안했습니다. 이를 통해 테이블을 문장으로 번역하는 새로운 감독된 딥 러닝 정보 추출 작업을 소개합니다. 또한, 100K 이상 주석된 표들로 이루어진 'SemTabNet' 데이터셋을 도입하고, T5 기반 모델 중 최고 성능 모델이 82% 유사성을 달성했다고 보고했습니다.

- **Technical Details**: Statement 데이터 구조는 복잡하고 불규칙하며 이질적인 정보를 균질한 방식으로 표현할 수 있는 트리 구조입니다. 이 구조는 하나의 내용에서 여러 (명명된) 저장소와 (다항식) 관계를 결합할 수 있습니다. 이 데이터 구조를 사용하여 ESG 보고서에서 정보를 추출하는 'statement extraction' 작업을 수행하고, 이를 평가하기 위해 Tree Edit Distance 기반의 Tree Similarity Score(TSS)를 제안했습니다.

- **Performance Highlights**: 논문의 T5 기반 모델은 baseline (21%)에 비해 높은 82% 유사성을 달성했습니다. 이 모델은 ESG 보고서의 2700개 이상의 표에 적용되어 유의미한 개선을 나타냈습니다.



### Zero-shot Composed Image Retrieval Considering Query-target Relationship Leveraging Masked Image-text Pairs (https://arxiv.org/abs/2406.18836)
Comments:
          Accepted as a conference paper in IEEE ICIP 2024

- **What's New**: 이번 논문에서는 마스크된 이미지-텍스트 쌍을 고려한 새로운 제로샷(composed image retrieval, CIR) 방법을 제안합니다. 기존 방식은 쿼리 이미지(query image)를 가상 단어(pseudo word)로 변환하고, 시각-언어 모델(pre-trained visual-language model)을 사용하여 이미지 검색을 구현했습니다. 그러나 이러한 방식은 쿼리-타겟 관계(query-target relationship)를 고려하지 않아, 데이터 수집과 학습의 규모가 커졌습니다.

- **Technical Details**: 제안된 방법은 마스크된 이미지-텍스트 쌍(masked image-text pairs)을 사용한 엔드-투-엔드(end-to-end) 학습을 통해 쿼리-타겟 관계를 학습합니다. 이미지와 텍스트에 있는 불필요한 정보를 각각 마스킹(masking)하여, 서로 상호 보완적인(complementary) 이미지-텍스트 쌍을 만듭니다. 학습 과정에서는 텍스트의 단어를 하나 마스크하고, 이미지에서 해당 단어와 관련 없는 부분을 마스킹합니다. 그런 다음, 이 마스크된 이미지-텍스트 쌍과 원래 이미지를 매칭하여 거리를 최소화하도록 텍스트 인벌전 네트워크를 훈련합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존의 제로샷 CIR 방식에 비해 상당한 성능 향상을 보였습니다. 제로샷 CIR에서 검색 중심적 텍스트 인벌전 네트워크를 사용하므로, 정확도를 크게 향상시켰습니다.



### A Stem-Agnostic Single-Decoder System for Music Source Separation Beyond Four Stems (https://arxiv.org/abs/2406.18747)
Comments:
          Submitted to the 25th International Society for Music Information Retrieval Conference (ISMIR 2024)

- **What's New**: Banquet라는 새로운 시스템이 제안되었습니다. 이 시스템은 단일 디코더 설정을 통해 여러 음원을 분리할 수 있으며, 이는 기존의 고정된 음원 분리 시스템의 복잡성을 극복합니다.

- **Technical Details**: Banquet는 bandsplit source separation model을 확장해 query-based setup에서 작동하며, music instrument recognition을 위한 PaSST 모델과 함께 사용됩니다. 이 시스템은 단일 stem-agnostic encoder와 decoder를 사용하여 여러 가지 음원 분리를 가능합니다. 또한, MoisesDB 데이터셋에서 성능을 평가한 결과, 드럼과 베이스의 분리 성능이 높은 수준을 보였고, 기타와 피아노에서도 state-of-the-art 수준을 기록했습니다.

- **Performance Highlights**: MoisesDB 데이터셋에서의 성능 평가에서, Banquet는 드럼과 베이스 분리에서 oracle 수준을 초과하였으며, 기타와 피아노에서도 뛰어난 성능을 보였습니다. 추가적으로, Banquet는 현재 소수의 현대 MSS 시스템에서만 지원되는 정밀한 음원 추출 기능을 제공했습니다.



### Re-Ranking Step by Step: Investigating Pre-Filtering for Re-Ranking with Large Language Models (https://arxiv.org/abs/2406.18740)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)을 이용한 정보 검색(IR)에서 사전 필터링(pre-filtering) 단계를 도입하여, 패시지 재정렬(passage re-ranking) 성능을 향상시키는 방법을 제안합니다. LLM에서 생성된 관련성 점수를 사용하여 관련 없는 패시지를 필터링한 후 재정렬을 수행하면, 자원이 제한된 소형 모델도 높은 성능을 보이는 대형 독점 모델과 경쟁할 수 있음을 보여줍니다.

- **Technical Details**: 이 연구는 LLM이 특정 쿼리와 패시지의 관련성을 0에서 1 사이의 점수로 평가하도록 하는 프롬프팅 전략을 사용합니다. 그런 다음, 소수의 사람에 의해 생성된 관련성 점수를 활용해 필터링 기준(threshold)을 설정하고, 이 기준을 초과하는 패시지만을 재정렬 단계로 전달합니다. 이 접근법의 효용성을 평가하기 위해 TREC-DL2019, TREC-DL2020, 및 여러 BEIR 데이터셋을 테스트했으며, 대부분의 데이터셋에서 필터링 기준을 찾아낼 수 있음을 확인했습니다.

- **Performance Highlights**: 이 연구에서는 제안한 사전 필터링 단계를 통해 Mixtral-8x7B-Instruct와 같은 소형 모델이 GPT-4와 같은 대형 모델의 성능에 근접하거나 일부 경우 이를 능가할 수 있음을 보여줍니다. 사전 필터링을 통해 재정렬 효율성도 크게 향상되었으며, TREC-DL2019, TREC-DL2020 그리고 여러 BEIR 태스크에서 훌륭한 성능을 발휘했습니다.

- **Research Questions**: ['RQ1: 전문가의 기존 지식을 사용하여 LLM이 관련 없는 패시지를 필터링할 수 있는가?', 'RQ2: 재정렬 전 관련 없는 패시지를 필터링하는 것이 LLM 재정렬 결과를 향상시킬 수 있는가?']



### Hire: Hybrid-modal Interaction with Multiple Relational Enhancements for Image-Text Matching (https://arxiv.org/abs/2406.18579)
Comments:
          22pages, 5 Figures, 6 tables, the extension of CMSEI in WACV23, and submitted to ACM TIST. arXiv admin note: text overlap with arXiv:2210.08908

- **What's New**: 새로운 연구는 기존 이미지-텍스트 매칭(image-text matching, ITM)의 문제점을 개선하는 Hybrid-modal Interaction with multiple Relational Enhancements (Hire) 모델을 제안합니다. 이 모델은 객체 간 관계와 문맥 정보를 통합적으로 다룸으로써 이미지와 텍스트 간의 유사성을 더욱 정확하게 예측할 수 있도록 설계되었습니다.

- **Technical Details**: 제안된 Hire 모델은 두 가지 주요 구성 요소가 있습니다. 첫 번째는 명시적 관계 모델링을 통해 객체의 공간적 위치와 장면 그래프를 고려해 시각적 객체의 문맥 표현을 향상시키는 intra-modal spatial-semantic graph-based reasoning network입니다. 두 번째는 명시적 관계 감지 전에 잠재적 관계 상호작용을 사용하는 암시적 관계 모델링으로, 이는 명시적 관계 감지의 오류 허용 범위를 향상시킵니다. 또한, 모델은 이미지와 텍스트 간의 상호작용 주의(attention)와 교차 모달 정렬(cross-modal alignment)을 통해 시각적 및 텍스트적 의미 표현을 정제합니다.

- **Performance Highlights**: 제안된 Hire 모델은 MS-COCO와 Flickr30K 벤치마크에서 새로운 state-of-the-art 결과를 달성하며, 기존 방법들을 능가하는 성능을 입증했습니다.



### DRAK: Unlocking Molecular Insights with Domain-Specific Retrieval-Augmented Knowledge in LLMs (https://arxiv.org/abs/2406.18535)
Comments:
          Ongoing work; 11 pages, 6 Figures, 2 Tables

- **What's New**: Large Language Models(LLMs)가 생체분자(biomolecules)같이 특정한 도메인의 독특한 문법에 신속하게 적응하기 어려운 문제에 직면해 있습니다. 이에 대한 해결책으로, 특정 도메인의 추론 능력을 강화하기 위해 Domain-specific Retrieval-Augmented Knowledge (DRAK)이라는 확장 가능하고 적응 가능한 비파라메트릭(non-parametric) 지식 주입 프레임워크를 제안합니다.

- **Technical Details**: DRAK는 도메인 지식이 반영된 프롬프트(knowledge-aware prompts)와 골드 라벨(gold label)-기반 추론을 활용하여, 분자 도메인에서 깊이 있는 전문 지식을 개발하고 다양한 분석 작업을 처리하는 능력을 보유하게 되었습니다. 두 가지 형태의 DRAK 변형을 평가하여 Mol-Instructions 데이터셋의 여섯 가지 분자 작업에서 기존의 벤치마크를 초과하는 성과를 입증했습니다.

- **Performance Highlights**: 광범위한 실험 결과는 DRAK의 탁월한 성능과 분자 통찰력을 열어주는 가능성을 강조하며, LLMs가 특정 도메인에서 지식 집약적 작업을 다루는 통일된 패러다임을 제공합니다. 코드는 곧 공개될 예정입니다.




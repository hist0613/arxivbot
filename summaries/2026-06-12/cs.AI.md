New uploads on arXiv(cs.CL)

### EvoArena: Tracking Memory Evolution for Robust LLM Agents in Dynamic Environments (https://arxiv.org/abs/2606.13681)
- **What's New**: 이번 논문에서는 EvoArena라는 벤치마크 스위트를 소개하여 동적 환경에서의 에이전트 성능을 평가합니다. 기존의 대부분의 벤치마크는 정적인 환경에서 평가했지만, EvoArena는 지속적인 환경 변화 속에서 에이전트의 적응력을 테스트합니다. 또한, EvoMem이라는 패치 기반 메모리 패러다임을 통해 에이전트가 환경 변화에 대한 기억을 기록할 수 있도록 하여, 메모리의 진화를 지원합니다.

- **Technical Details**: EvoArena는 점진적으로 진화하는 터미널, 소프트웨어, 사회적 도메인을 기반으로 환경 변화의 시퀀스를 모델링합니다. EvoMem은 업데이트 역사에 따른 메모리 진화를 구조적으로 기록하여 에이전트가 메모리의 변화를 통해 환경 진화를 추론할 수 있도록 하는 경량의 git 유사 메모리 패러다임입니다. 실험 결과, EvoMem은 평균적으로 1.5%의 성능 향상을 제공하며, 체인 수준의 정확도를 3.7% 개선하는 것으로 나타났습니다.

- **Performance Highlights**: 현재의 에이전트들은 EvoArena에서 평균 39.6%의 정확도를 기록하며, 진화하는 환경에서의 성능 저하를 경험합니다. EvoMem을 도입함으로써 이러한 성능 저하 문제를 해결하는 데 성공하였고, GAIA와 LoCoMo와 같은 표준 벤치마크에서도 각각 6.1%와 4.8%의 개선을 보였습니다. 이러한 결과는 에이전트의 신뢰성을 높이는 데 있어 환경의 진화를 모델링하는 것이 얼마나 중요한지를 강조합니다.



### Learning to Reason by Analogy via Retrieval-Augmented Reinforcement Fine-Tuning (https://arxiv.org/abs/2606.13680)
- **What's New**: 이 논문에서는 Retrieval-Augmented Reinforcement Fine-Tuning (RA-RFT)이라는 새로운 프레임워크를 제안합니다. RA-RFT는 언어 모델이 유사한 문제를 해결할 때 분석적 사고를 학습하도록 설계된 사후 훈련 방법론입니다. 이는 고품질의 비슷한 예시를 찾기 위해 gold-relevance distillation을 이용하며, 의미적 유사성보다는 추론 유용성에 기반하여 맥락을 정렬합니다.

- **Technical Details**: RA-RFT의 기본적인 구성 요소는 세 가지 단계로 이루어져 있습니다: (1) gold-relevance distillation은 선별된 추론 패턴과 타겟 문제 간의 전이 가능성을 평가하여 훈련 감독을 구축합니다. (2) reasoning-aware retriever training은 효과적인 맥락을 추출하기 위해 대비 학습을 사용하여 밀집 검색기를 학습합니다. 마지막으로, (3) reinforcement fine-tuning에서는 검색한 유사한 예를 훈련 프롬프트에 주입하여 정책 모델을 최적화합니다.

- **Performance Highlights**: RA-RFT는 AIME 2025, HMMT 2025 등의 경쟁 수준의 수학적 추론 벤치마크에서 뛰어난 성과를 보여줍니다. 예를 들어, Qwen3-1.7B 모델에서는 GRPO 대비 AIME 2025 average@32 정확도를 7.1점 향상시켰습니다. 이러한 결과는 RA-RFT가 추론 유용성에 기초한 검색이 강화 학습의 성능 향상에 필수적임을 나타냅니다.



### Influcoder: Distilling Decoders' Gradient Influence Rankings into an Encoder for Data Attribution (https://arxiv.org/abs/2606.13668)
Comments:
          8 pages, 2 figures

- **What's New**: 이 논문에서는 LLMs(대형 언어 모델)의 데이터 속성(Data Attribution) 연구가 활발히 진행되고 있으며, 이는 개별 훈련 샘플이 모델의 행동에 어떻게 영향을 미치는지를 설명하는 방법론에 대한 관심에서 비롯되었다고 소개합니다. 기존의 영향 함수(influence function) 방법들은 대규모 데이터셋에 실용적으로 적용하기에는 처리 속도와 저장 용량이 부족하다는 점이 문제로 지적됩니다. 이에 저자들은 데이터 속성을 대규모로 추정하는 빠르고 비용 효율적인 방법인 Influcoder를 제안합니다.

- **Technical Details**: Influcoder는 훈련 데이터의 각 샘플이 모델의 출력에 미치는 영향을 평가하기 위한 새로운 방법론을 제시합니다. 기존의 역 헤세안(inverse Hessian) 기반 접근 방식 대신, 저자들은 일차적인 영향 추정(First-order influence estimation)에 기반하여 훨씬 더 안정적이고 빠른 처리를 가능하게 합니다. 이 방법은 LoRA(저품질로 길이를 조정하는 기법) 매개변수를 사용하여 훈련 샘플 간의 코사인 유사성(cosine similarity)을 기반으로 하는 영향을 계산합니다.

- **Performance Highlights**: 실험 결과, Influcoder는 목표 모델이 생산한 기준 레이킹과의 일치도를 높이는 데 뛰어난 성과를 보였습니다. 저자들은 SmolLM2-1.7B 모델을 기반으로 하여 Influcoder의 성능을 검토하였으며, 다양한 영향 추정 방법의 효과성을 비교 평가하였습니다. 결과적으로, Influcoder는 다른 기존 방법들보다 대규모에서의 데이터 속성을 더 효과적으로 추정할 수 있음을 보여주며, 더 나은 처리 속도와 정확도를 근본적으로 제시합니다.



### HyperTool: Beyond Step-Wise Tool Calls for Tool-Augmented Agents (https://arxiv.org/abs/2606.13663)
- **What's New**: 최근에 발표된 HyperTool은 모델이 도구를 효율적으로 사용할 수 있도록 돕기 위해 설계된 통합 가능한 MCP 스타일의 실행 도구 인터페이스입니다. 이 인터페이스를 통해 모델은 여러 도구를 하나의 코드 블록 내에서 호출하고, 변환하며, 결과를 관리할 수 있습니다. 이로 인해 모델이 여러 도구를 단일 호출로 처리할 수 있게 되어, 기존 모델에서 발생하는 실행 세부정보 노출을 방지합니다.

- **Technical Details**: HyperTool은 기존 MCP 도구 스키마를 유지하면서도, 각 호출을 원자적으로 다루는 대신 로컬 결정론적 서브루틴을 하나의 외부 호출로 통합하는 형식을 지원합니다. 모델은 HyperTool을 사용하여 여러 도구 호출을 하나의 블록에 포함시키고, 필요한 경우 중간 결과를 조작하며 전달할 수 있습니다. 이러한 접근 방식은 모델의 추론 효율성과 도구 사용 표현력을 동시에 향상시킵니다.

- **Performance Highlights**: HyperTool을 통해 Qwen3-8B 모델의 평균 정확도가 9.93%에서 33.33%로, Qwen3-32B 모델은 15.69%에서 35.29%로 크게 향상되었습니다. 이러한 성과는 HyperTool이 기존의 GPT-OSS 및 Kimi-k2.5 모델보다 평균 정확도 면에서 우수함을 보여줍니다. 실험 결과는 HyperTool이 다양한 도구를 사용하여 높은 수준의 정확성을 달성할 수 있게 한다는 것을 입증합니다.



### Operadic consistency: a label-free signal for compositional reasoning failures in LLMs (https://arxiv.org/abs/2606.13649)
- **What's New**: 이번 연구에서는 LLM(대형 언어 모델)의 추론 시간 동안 내부의 논리적 일관성을 확인할 수 있는 새로운 방법인 operadic consistency (OC)를 제안합니다. 이는 복잡한 질문에 대한 모델의 직접적인 답변과 해당 질문의 분해(decomposition)에 대한 답변이 일치해야 한다는 개념입니다. OC는 12개의 instruction-tuned LLM에서 여러 QA 데이터 세트에 걸쳐 높은 정확도와 강한 상관관계를 보입니다.

- **Technical Details**: 연구에서는 질문의 operad를 정의하고 언어 모델을 이 operad 위의 대수로 해석합니다. operadic consistency는 모델이 복잡한 질문에 대한 직접적인 답변과 그 질문의 분해를 바탕으로 하는 답변이 일치해야 한다는 조건을 제시합니다. 이 연구는 HotpotQA, MuSiQue, StrategyQA, DROP, GSM8K와 같은 다섯 개의 구성적 추론 벤치마크를 통해 실증적으로 OC를 평가합니다.

- **Performance Highlights**: OC 신호는 12개의 모델에서 모든 데이터셋에서 r ≥ 0.85의 상관관계를 보였으며, CoT-SC(Chain-of-Thought Self-Consistency)와 비교하여 각 데이터셋에서 추가 정보를 제공합니다. OC는 선택적 예측(selective prediction) 개선에서도 긍정적인 결과를 보였고, 모델 간 교차 정확도를 예측하는 데 유용했습니다. 연구 결과는 OC가 기존 방법과 비교해 더 나은 성능을 제공함을 시사합니다.



### SkMTEB: Slovak Massive Text Embedding Benchmark and Model Adaptation (https://arxiv.org/abs/2606.13647)
Comments:
          ACL 2026

- **What's New**: 이번 논문에서는 슬로바키아어를 위한 최초의 포괄적인 MTEB 스타일 텍스트 임베딩 벤치마크인 SkMTEB를 소개합니다. 이 벤치마크는 7가지 작업 유형에 걸쳐 31개의 데이터셋으로 구성되어 있으며, 슬로바키아어를 위한 기존 다국어 벤치마크 커버리지의 거의 4배에 해당합니다. 논문에서는 대규모 명령어 조정된 다국어 모델들이 가장 우수한 성능을 보여주며, 슬로바키아 특정 모델이 임베딩 작업에 대한 전이가 좋지 않다는 것을 보여줍니다.

- **Technical Details**: 슬로바키아어에 대한 임베딩의 효율성을 높이기 위해, 저자들은 Multilingual E5 모델에 대해 어휘 절단(vocabulary trimming)과 세밀조정(fine-tuning)을 적용하여 e5-sk-small (45M 파라미터)과 e5-sk-large (365M 파라미터)를 개발했습니다. 이러한 모델은 최대 62%의 크기 감소에도 불구하고, 경제적인 비용으로 세멘틱 검색(semantic search) 및 검색 중심 생성(RAG)에 적합한 성능을 유지합니다. SkMTEB 벤치마크는 슬로바키아어에 대한 robust한 평가 벤치마크를 제공하여, 효율적인 모델 개발을 위한 기초를 마련합니다.

- **Performance Highlights**: 개발된 오픈소스 모델들은 상업적인 API와 경쟁력 있는 성능을 획득하면서도 지역적으로 배포 가능한 특성을 지닙니다. 이 논문에서는 31개의 공개 가중치 및 상용 임베딩 모델을 평가하여 슬로바키아어에 특화된 임베딩 모델의 발전 가능성을 확인했습니다. 전체 모델, 데이터셋 및 코드는 공개되어 있으며, 다른 자원이 부족한 언어들에 대해서도 적용 가능한 접근 방식을 제시하고 있습니다.



### Recursive Agent Harnesses (https://arxiv.org/abs/2606.13643)
- **What's New**: 비선형 컨텍스트(Non-linear context) 사고를 위한 효과적인 전략으로 재귀적 언어 모델(Recursive Language Models, RLMs)이 주목받고 있다. 이 논문에서는 전체 에이전트 하네스(Agent Harness)와 파일 시스템 도구를 포함하는 재귀 에이전트 하네스(Recursive Agent Harness, RAH)의 개념을 도입한다. RAH는 부모 에이전트가 실행 가능한 스크립트를 생성하고 이를 통해 독립적인 하네스를 위한 하위 에이전트를 생성하는 구조로, 이를 통해 장기 컨텍스트에서의 처리 성능을 향상시킨다.

- **Technical Details**: RAH는 기존의 RLMs에서의 모델 요청을 대신하여 전체 하네스를 재귀적 유닛으로 설정한다. 이는 각 항목마다 독립적인 컨텍스트 창과 파일 시스템을 가진 하위 에이전트를 생성하는 특징을 가진다. 연구에서는 199개의 샘플을 사용하여 RAH가 Codex 모델을 71.75%에서 81.36%로 개선했음을 보였다. 이는 모델이 아니라 하네스의 향상으로 귀속된다.

- **Performance Highlights**: RAH는 Oolong-Synthetic 데이터셋을 기반으로 주요 성능 향상을 보여준다. 하네스가 장기적인 작업에 대해 유연하게 독립적인 서브 에이전트를 생성하면서 컨텍스트의 수를 한계 없이 확장 가능하게 한다. GPT-5 기반의 평가에서 하네스가 더 나은 결과를 나타냈으며, Claude Sonnet 4.5와 같은 더 강력한 백본에서도 성능이 89.77%까지 도달하였다.



### Operads for compositional reasoning in LLMs (https://arxiv.org/abs/2606.13634)
- **What's New**: 이번 논문에서는 복잡한 쿼리(queries)를 간단한 서브 쿼리로 분해하는 질문 분해(question decomposition) 전략에 대해 다루고 있습니다. 이를 위해, 많은 입력에 대한 단일 출력을 다루는 조작을 모델링하는 수학적 구조인 operads를 제안합니다. 논문에서는 질문 템플릿에 해당하는 작업과 서브 답의 치환에 해당하는 조합을 통해 질문의 operad $Q$를 정의하고, QA 모델이 $Q$의 대수(algebra)로 해석될 수 있도록 합니다.

- **Technical Details**: 이 연구는 기존의 질문 분해 방법을 재구성하는 것 외에도 operadic consistency라는 새로운 개념을 제시하며, 이는 질문 분해 트리의 부분적 수축(partial collapses)을 통해 QA 모델의 답변의 일관성을 측정하는 방법입니다. operads는 질문 분해를 위한 자연스러운 수학적 배경을 제공하며, 이를 통해 QA 모델들이 서로 다른 부분에서 합의하는 답변을 도출하는지 판별할 수 있습니다.

- **Performance Highlights**: empirical evaluation(실증적 평가)를 통해 operadic consistency가 12개의 LLM(대형 언어 모델)과 4개의 멀티 홉 QA 데이터셋에서 정확도와 강한 상관관계를 보이며, 표준 온도 기반(self-consistency) 기준선보다 우수한 성능을 발휘한 결과를 제시합니다. 이는 multi-step reasoning(다단계 추론)의 신뢰성을 분석하고 개선하는 새로운 방향을 탐구하는 데 기여할 수 있습니다.



### From Tokens to Faces: Investigating Discrete Speech Representations for 3D Facial Animation (https://arxiv.org/abs/2606.13630)
Comments:
          This work has been accepted in Interspeech 2026

- **What's New**: 이 논문은 3D 얼굴 애니메이션을 위한 음성 표현의 중요성을 다루고 있습니다. 다양한 음성 표현 방식인 SSL, 신경 부호화기(neural codecs), ASR 스타일을 비교하여 정확한 얼굴 애니메이션 예측에 중요한 정보를 밝힙니다. 특히, 음성의 음소 클래스 인코딩이 얼굴 애니메이션 품질 향상에 기여함을 발견했습니다.

- **Technical Details**: 이 연구에서는 HuBERT, SpeechTokenizer, WavTokenizer, CosyVoice2 등 네 가지 음성 인코더를 사용하여 실험을 진행합니다. 각 인코더는 GRU와 Transformer 구조의 두 가지 3D 얼굴 디코더와 결합되어 얼굴 애니메이션 예측 능력을 평가합니다. 실험은 BEAT2 데이터셋에서 약 27시간 분량의 영어 음성을 사용하는데, 이는 3D 얼굴 움직임에 맞춰져 있습니다.

- **Performance Highlights**: 실험 결과, 세 가지 음성 표현 방식이 얼굴 애니메이션의 품질을 비교적 유사한 수준에서 향상시키지만, 특정 음소 클래스를 인코딩하는 방식이 특히 효과적입니다. 또한, 이 연구에서는 단일 표현으로 음성 및 3D 얼굴 애니메이션을 동시에 생성할 수 있는 Audio Visual Text-to-Speech (AVTTS) 파이프라인을 소개했습니다.



### Beyond Uniform Tokens: Adaptive Compression for Time Series Language Models (https://arxiv.org/abs/2606.13624)
- **What's New**: 본 논문은 비대칭 토큰 관점(asymmetric-token perspective)에서 시간 시계열(time series) 언어 모델링의 토큰 효율성을 연구합니다. 시간 시계열 토큰은 스펙트럼 기여(spectral contributions)에 있어 불균형을 보이며, 이는 전체 모델에서 문맥을 유지할 필요성이 없음을 시사합니다. 이 연구에서는 토큰 예산(token budgeting) 프레임워크를 개발하여, 시간 시계열 토큰을 압축하고 문맥 토큰을 점진적으로 줄이는 방법을 제안합니다.

- **Technical Details**: 연구는 다양한 주파수 대역에서의 토큰의 스펙트럼 패턴을 분석하며, 대부분의 토큰이 중복된 빈도 패턴을 공유하는 반면, 소수의 토큰만이 중요한 시간적 증거를 보존함을 발견했습니다. 또한, 모델의 깊이가 증가함에 따라 문맥 토큰의 영향력이 감소하는 "피라미드 감소(pyramidal decay)" 현상을 발견하였습니다. 이를 통해 초기 레이어에서 문맥 토큰을 aggressively하게 압축하는 전략이 가능합니다.

- **Performance Highlights**: 실험 결과, 예측(forecasting), 분류(classification), 채우기(imputation), 이상 탐지(anomaly detection) 분야와 같은 다양한 설정에서 최대 7.68배의 추론 속도 향상과 78%의 성능 향상을 달성했습니다. 이러한 결과는 비대칭 토큰 압축(asymmetric token compression)이 확장 가능한 시간 시계열 기반 모델에 효과적임을 보여줍니다.



### One Polluted Page Is Enough: Evaluating Web Content Pollution in Generative Recommenders (https://arxiv.org/abs/2606.13610)
- **What's New**: 본 논문에서는 FORGE(Fake Online Recommendations in Generative Environments)라는 새로운 벤치마크를 제시하여 오염된 웹 콘텐츠에 대한 생성 추천 시스템의 취약성을 측정하는 방법을 개발합니다. 검색을 통해 실시간 웹 페이지를 가져오는 LLM은 가짜 리뷰와 프로모션 페이지와 같은 오염된 콘텐츠를 수용함으로써 가짜 제품의 홍보자가 될 수 있는 위험이 있습니다. 이 연구는 LLM이 얼마나 자주 가짜 제품을 추천하는지를 측정할 수 있는 새로운 접근 방식을 제공합니다.

- **Technical Details**: FORGE는 사용자 쿼리, 라이브 웹 검색, 최상위 증거 번들, LLM 소비, 순위 추천의 절차를 통해 구축됩니다. 원래 제품의 언급을 오염시키는 대신 실제 브랜드를 가짜 브랜드로 변경하여, 모델의 추천이 어떻게 변하는지를 분석합니다. FORGE는 15개 카테고리와 225개의 실제 제품을 포함하며, 12개의 상업적 및 오픈 소스 LLM 모델에서의 취약성을 테스트합니다.

- **Performance Highlights**: 연구 결과, 모든 모델이 취약하며, 단일 오염된 페이지가 최대 27%의 잘못된 추천을 생성할 수 있습니다. 더군다나, 상위 3개 제품이 대체되면 이 비율은 73.8%로 증가했습니다. 모델의 브랜드 지식이 약할수록 취약성이 증가하며, '사회적 증거'를 생성하여 잘못된 추천을 정당화하는 경우도 관찰되었습니다.



### LabVLA: Grounding Vision-Language-Action Models in Scientific Laboratories (https://arxiv.org/abs/2606.13578)
Comments:
          Work in progress. Project website at this https URL

- **What's New**: 최근 과학 실험에서 AI 시스템의 도움이 더해지고 있으나, 물리적 실험 실행은 여전히 인간 조작자에게 의존하고 있습니다. 비전-언어-액션 (Vision-Language-Action, VLA) 모델은 이 과정을 연결할 수 있는 한 가지 방법을 제공합니다. 하지만 기존의 정책들은 주로 가정용 및 테이블탑 환경에 대해 훈련되어 과학 실험에 필요한 정밀한 조작 지식이 부족합니다. 이에 따라 RoboGenesis라는 시뮬레이션 기반 데이터 엔진이 개발되어 실험 프로토콜을 더 향상시키고 있습니다.

- **Technical Details**: RoboGenesis는 실험 환경을 구축하고, 자동 조작 스킬을 기반으로 프로토콜을 생성하여 성공 사례를 필터링하는 과정으로 구성되어 있습니다. 이 엔진은 실험 장비를 다양하게 사용하여 실제 실험실 데이터 수집의 높은 비용을 줄이는 데 중점을 두고 있습니다. LabVLA는 로봇의 상태, 언어 지시 및 시각적 관찰을 결합하여 계속적인 동작 토큰을 생성하는 방식으로 설계되었습니다. 정책 훈련의 두 단계는 초기 FAST 행동 토큰 사전 훈련과 후속 유동 매칭을 포함합니다.

- **Performance Highlights**: LabVLA는 LabUtopia 벤치마크에서 평가한 모든 기준선보다 평균 성공률이 가장 높았습니다. 이 성공적인 결과는 RoboGenesis가 생성한 데이터의 질과 다양성이 큰 기여를 했다는 것을 시사합니다. LabVLA는 다양한 로봇 에지 효과와 환경에서 프로토콜을 실행하는 능력을 보유하고 있어, 앞으로의 과학 실험 자동화에 중요한 역할을 할 것으로 기대됩니다.



### ArogyaSutra: A Multi-Agent Framework for Multimodal Medical Reasoning in Indic Languages (https://arxiv.org/abs/2606.13572)
- **What's New**: 이 논문은 의료 분야에서 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 성능 격차를 해소하기 위한 두 가지 주요 기여를 소개합니다. 첫 번째로, ArogyaBodha라는 대규모 다국어 다중모달 의료 질문-답변 데이터셋을 구축하였으며, 이는 31개 신체 시스템과 21개 임상 분야를 포함하고 있습니다. 두 번째로, ArogyaSutra라는 다중 에이전트 프레임워크를 제안하여 이미지와 텍스트 입력에 대한 단계별 의사 결정을 지원합니다.

- **Technical Details**: ArogyaBodha 데이터셋은 8개의 서로 다른 의료 출처로부터 수집되었으며, 영어와 7개의 주요 인도 언어를 포함합니다. 이 데이터셋은 전문가 검증을 통해 비즈니스 논리의 정확성과 언어 일관성을 평가할 수 있도록 구성되어 있습니다. ArogyaSutra 프레임워크는 도구 기반의 시각적 기초와 이중 기억 메커니즘을 결합하여 단계별 의사 결정 과정을 효과적으로 수행합니다.

- **Performance Highlights**: 실험 결과, ArogyaSutra는 다양한 언어, 이미징 방식 및 임상 도메인에서 뛰어난 성능을 보이며 기존의 강력한 대체 모델보다 일관된 결과를 도출합니다. 본 논문에서 제안한 접근 방식은 저자원이지만 다국어 환경에서도 신뢰할 수 있는 의료 AI 시스템을 지원할 수 있는 가능성을 보여줍니다. 이러한 연구는 공정하고 포괄적인 의료 접근을 증진시킬 수 있는 의의를 갖습니다.



### When Does Mixing Help? Analyzing Query Embedding Interpolation in Multilingual Dense Retrieva (https://arxiv.org/abs/2606.13537)
Comments:
          ACL 2026 Main (Oral)

- **What's New**: 이 논문은 멀티언어 커뮤니티에서 혼합 언어 쿼리를 사용했을 때의 민감성을 연구합니다. 기존의 모놀링구얼(단일 언어) 또는 크로스링구얼(다중 언어에 대한 변환) 기준에서 벗어나, 쿼리의 언어 혼합 비율을 통제하여 검색 성능을 평가하는 체계적인 연구를 제시합니다. 이러한 비율 통제 실험을 통해, 영어가 포함된 색인에서는 순수한 영어 쿼리가 가장 좋은 성과를 보이는 한편, 비영어 문서 색인에서의 혼합 쿼리는 통상적으로 유리함을 발견했습니다.

- **Technical Details**: 연구에서는 임베딩 레벨 믹싱(embedding-level mixing) 기법을 사용하여 두 언어의 모놀링구얼 쿼리 임베딩을 혼합합니다. 이를 통해 다양한 언어 조합에 대해 정밀하게 혼합 비율을 통제하며, 이 방식이 LLM(대형 언어 모델) 기반의 워드 레벨 믹싱보다 더 높은 성능을 보임을 확인했습니다. 실험은 BGE-M3 모델을 사용하여 35개의 언어 쌍에 대한 비율 통제 실험을 진행하였으며, 각 쌍에 대해 3가지 문서-언어 설정을 평가했습니다.

- **Performance Highlights**: 결과적으로, 최적의 임베딩 혼합 비율이 105개의 설정 중 88개에서 최고 수준의 단일 언어 쿼리를 초월했습니다. 영어가 불리한 경우, 다른 언어 쿼리의 혼합이 유리하게 작용하지만, 영어가 포함된 경우에는 순수한 영어 쿼리가 가장 좋습니다. 또한, 영어는 다른 비영어 문서 언어에 대해 가장 강력한 혼합 파트너 역할을 하며, 언어의 유형적 거리(typological distance)가 커질수록 혼합 이득이 감소한다는 것을 발견했습니다.



### Leveraging Audio-LLMs to Filter Speech-to-Speech Training Data (https://arxiv.org/abs/2606.13507)
Comments:
          Accepted to INTERSPEECH 2026

- **What's New**: 이 논문에서는 음성-음성 번역(end-to-end S2ST)을 위한 훈련 데이터를 정제하는 방법에 대해 연구하였습니다. 일반적으로 음성 데이터는 소음(noise), 잘못된 정렬(misalignment), 의미적 오류(semantic errors)를 포함하는 경우가 많아 훈련 성능에 영향을 미칩니다. 본 연구에서는 오디오-언어 모델을 훈련하여 이러한 결정을 내리도록 하였습니다.

- **Technical Details**: 먼저, 저신뢰성(supervision 없이)에서 코스 순위를 학습한 후, 오디오 대형 언어 모델(audio large language model)을 훈련하여 쌍으로된 음성에서 keep/drop 결정을 직접 예측하는 두 단계의 Rank-to-Distill 전략을 채택하였습니다. 이를 통해 음성 품질을 평가하고, 음성 조건부 데이터 선택을 통해 S2ST 훈련을 개선할 수 있음을 보였습니다. 또한, 신호 대 잡음비(SNR), 평균 의견 점수(Mean Opinion Score, MOS) 및 의미적 정렬(semantic alignment)과 같은 기준을 사용하여 음성 데이터의 품질을 검토하였습니다.

- **Performance Highlights**: CVSS-C 및 SpeechMatrix에서 실시한 실험에서, 필터링된 데이터로 훈련이 진행된 결과, 비필터링된 훈련에 비해 +1.4 ASR-BLEU의 향상을 얻을 수 있었습니다. 이러한 결과는 음성-음성 번역의 품질 향상에 기여하며, 자주 활용되지 않는 데이터의 필터링 문제를 해결할 수 있는 방법을 보여줍니다.



### Ontology Memory-Augmented ASR Correction for Long Text-Speech Interleaved Conversations (https://arxiv.org/abs/2606.13464)
- **What's New**: 이 논문은 자율 음성 인식(ASR) 수정에 대한 새로운 접근법을 제시합니다. 기존의 ASR 수정 방법이 단기적인 문장이나 지역적인 맥락에 집중했던 반면, 이 연구는 대화 전체의 맥락을 이용하여 긴 대화에서의 ASR 수정을 다룹니다. 새로운 온톨로지 메모리 기반의 ASR 수정 프레임워크를 설계하여 대화의 맥락과 관련된 정보를 구조적으로 관리할 수 있도록 합니다.

- **Technical Details**: 연구에서는 이전 대화 이력을 온톨로지 메모리라는 동적으로 업데이트 가능한 구조에 저장하여 ASR 수정에 필요한 관련 정보를 효율적으로 접근합니다. MAGIC-RAMC 데이터셋을 기반으로 한 RAMC-Corr 데이터셋을 구축하여 긴 텍스트-음성이 혼합된 대화에서의 ASR 수정 성능을 평가합니다. 모델은 각 ASR 가설에서 관련 증거를 검색하여 수정 작업을 수행하며, 대화의 전체 진행 중에 메모리를 계속 갱신합니다.

- **Performance Highlights**: RAMC-Corr 데이터셋에서 실험한 결과, 제안한 방법은 10개의 다양한 백본 설정 조합 중 9회에서 직접 수정보다 향상된 성능을 보였습니다. 이로써 이 연구가 제안하는 프레임워크가 ASR 오류 수정의 효과성을 높이는 데 기여할 수 있음을 보여줍니다. 즉, 대화 중 축적된 맥락을 활용함으로써 더 선택적이고 증거 기반의 수정이 가능해집니다.



### S-GBT: Smooth Growth Bound Tensor for Certified Robustness Against Word Substitution Attacks in NLP (https://arxiv.org/abs/2606.13439)
Comments:
          The paper has been accepted at NETYS 2026 - 14th edition of the International Conference on Networked Systems

- **What's New**: 이 논문에서는 Smooth Growth Bound Tensor (S-GBT)라는 새로운 방법을 소개하여 딥러닝 모델이 단어 대체 공격에 대해 보다 견고성을 갖도록 개선합니다. 기존의 방어 방법들이 가장 간단한 형태의 민감도만 고려했지만, S-GBT는 두 번째 미분을 기반으로 하여 모델의 출력 변화를 제어합니다. 이 연구는 이론적 증명을 통해 새로운 방어 방법의 효과성을 입증하고, 다양한 신경망 아키텍처에 통합 가능한 범용적인 공식을 제공합니다.

- **Technical Details**: S-GBT는 Hessian 행렬을 요소별로 경계짓는 두 번째 차수 규제 방법으로, 모델의 출력 변화를 적절하게 제어하여 더 부드러운 결정 경계를 형성합니다. 이 방법은 Adversarial Training (AT)와 함께 적용되며, 모델의 기울기와 곡률을 동시에 최소화하도록 설계되었습니다. 논문에서는 LSTM과 Convolutional Neural Networks (CNN) 아키텍처에 S-GBT를 적용한 방법을 다루고 있으며, 이론적 분석을 통해 입력 변동에 대한 견고성을 증명합니다.

- **Performance Highlights**: 여러 벤치마크 데이터셋에서 S-GBT의 효과가 검증되었으며, Yahoo 데이터셋에서 90.7%의 인증된 견고성을 달성했습니다. 이는 기존의 Growth Bound Matrices (GBM)보다 약 23.4%의 성능 향상을 보인 결과입니다. 이러한 연구 결과는 모델의 기울기와 그 변화를 제어하는 것이 좀 더 견고한 NLP 시스템 개발의 유망한 방향임을 강조합니다.



### An End-to-End Hybrid Framework for Rumour Detection in Low-Resources Algerian Dialec (https://arxiv.org/abs/2606.13411)
- **What's New**: 이번 논문은 알제리 방언 소셜 미디어 콘텐츠에 대한 혼합형 루머 탐지를 위한 엔드 투 엔드 프레임워크를 제안합니다. 데이터 세트를 생성하기 위해 실제 소셜 미디어 게시물과 합성 데이터를 결합하고, FASSILA 말뭉치에 기반하여 유사성을 통한 주석화 프로세스를 통해 자동 라벨링을 수행하였습니다. 또한, 아랍 문자가 포함된 병렬 데이터 세트를 생성하기 위한 음역 파이프라인이 도입되었습니다.

- **Technical Details**: 제안된 프레임워크는 세 가지 주요 모듈로 구성된 순차적 멀티 스테이지 파이프라인입니다: (i) 데이터 수집 및 자동 라벨링, (ii) 언어적 전처리 및 문자 변환, (iii) 특징 임베딩 및 모델 훈련. 본 연구는 고전 기계 학습, 심층 학습, 트랜스포머 및 하이브리드 모델을 포함한 여러 접근 방식을 평가하였으며, 하이브리드 접근 방식이 가장 높은 성능을 나타내는 것으로 나타났습니다.

- **Performance Highlights**: 하이브리드 모델은 트랜스포머 임베딩과 고전 분류기를 결합하여 F1 스코어 0.84에 도달하는 뛰어난 성능을 보였습니다. 또한 도메인 특화 사전 훈련이 모델 크기보다 더 중요하다는 것이 밝혀졌으며, 소셜 미디어에 훈련된 모델이 일반 아랍어 말뭉치로 훈련된 더 큰 모델보다 우수한 성능을 발휘했습니다. 이러한 결과는 자원이 부족한 알제리 방언 환경에서 루머 탐지가 가능하다는 것을 보여줍니다.



### From Passive Generation to Investigation: A Proactive Scientific Peer Review Agen (https://arxiv.org/abs/2606.13349)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)이 자동 과학 동료 검토에서 능력을 발휘할 수 있지만, 기존 방식의 한계를 지적합니다. 연구자들은 인간 검토자가 수집된 증거에 따라 의심스러운 부분을 능동적으로 조사하는 것과 같은 유연성을 LLM에게 제공해야 한다고 주장합니다. 이를 위해 이들은 ProReviewer라는 액터를 제안하며, 이 액터는 구조화된 검토 로그를 유지하여 논문을 능동적으로 검토할 수 있습니다.

- **Technical Details**: ProReviewer는 Markov Decision Process (MDP)로 형식화되어, 증거를 기반으로 한 능동적인 검토 전략을 생성할 수 있습니다. 이는 수집된 정보를 바탕으로 무엇을 조사할지 결정하는 일련의 과정을 포함하며, 강화 학습(reinforcement learning)을 통해 학습됩니다. ProReviewer는 반복적인 결정 과정을 통해 각 검토 단계에서 업데이트된 로그를 통합하며, 글 전체에서 검토를 사용할 수 있는 기반을 제공합니다.

- **Performance Highlights**: 실험 결과, ProReviewer는 8B 백본을 사용하여 다섯 가지 검토 품질 차원에서 평균적으로 가장 높은 점수를 기록했습니다. 기존의 프롬프트 기반 방법들에 비해 최대 39%까지 성능이 향상되었으며, 가장 우수한 파인튜닝된 기준 모델보다도 16%의 성과를 보였습니다. 특히 인간 평가에서도 ProReviewer의 검토가 모든 비교에서 우수한 평가를 받았으며, 서로 다른 섹션 간의 미묘한 불일치를 효과적으로 탐지했습니다.



### IVIE: A Neuro-symbolic Approach to Incremental and Validated Generation of Interactive Fiction Worlds (https://arxiv.org/abs/2606.13348)
Comments:
          10 pages, 3 figures. To appear in the Proceedings of the 16th International Conference on Computational Creativity (ICCC'26), June 2026

- **What's New**: 본 연구에서는 IVIE (Incremental & Validated Interactive Experiences)라는 신경-기호적 접근 방식을 통해 자동으로 완전하고 플레이 가능한 인터랙티브 픽션 세계를 생성하는 방법을 제시합니다. 기존의 큰 언어 모델(LLM)들은 창의적인 서사를 생성할 수 있지만 일관성 있는 세계를 유지하는 데 어려움을 겪었습니다. IVIE는 PAYADOR의 신경-기호적 프레임워크를 기반으로 하여 창의적 결정을 LLM에 위임하고, 세계 상태를 기호적 검증을 통해 확립하는 4단계의 점진적 생성 파이프라인을 구현합니다.

- **Technical Details**: IVIE는 기호적 구성요소가 구조적 실행 및 검증 레이어로 작동하는 4단계 증가 생성 방식을 사용합니다. 각 단계에서 기호적 검증은 공간 연결성, 유형 정확성 및 목표 해결 가능성을 보장합니다. LLM은 위치 설명, 캐릭터 배경 이야기, 퍼즐 설계 및 목표 정립과 같은 이야기 요소를 생성하는 역할을 하며, 기호적 구조가 세계 상태를 유지합니다.

- **Performance Highlights**: IVIE의 초기 실험 결과는 플레어의 참여를 높이는 몰입감 있고 주제적으로 일관된 세계를 생성하는 데 성공했음을 보여줍니다. 그러나 LLM의 불일치가 때때로 퍼즐 제약을 우회하는 등의 문제도 빈번하게 발생하며, 이러한 제약은 향후 신경-기호적 인터랙티브 스토리텔링 시스템의 설계 고려 사항으로 제시됩니다.



### Low-Latency Real-Time Audio Game Commentary System via LLM-Based Parallel Text Generation (https://arxiv.org/abs/2606.13322)
Comments:
          Accepted at IJCAI-ECAI 2026 (Demonstrations Track)

- **What's New**: 이번 연구는 실시간 낮은 지연 시간의 오디오 게임 해설 시스템을 제안합니다. 이 시스템은 라이브 게임 플레이 비디오에서 직접 음성을 생성하여 실시간으로 해설을 제공합니다. 특히, 기존 시스템들이 발생시키는 긴 침묵 시간을 줄이고, 그 과정을 병렬적으로 처리함으로써 반응 시간을 획기적으로 개선한 점이 특징입니다.

- **Technical Details**: 제안된 시스템은 텍스트 생성과 음성 합성을 병렬로 실행하여 지연 시간을 줄이는 방식으로 설계되었습니다. 현재 음성이 재생되는 동안, 새로운 비디오 구간이 도착하면 후보 발화가 생성되어 버퍼에 저장됩니다. 이 설계는 지연이 발생할 경우 비디오 스트림을 의도적으로 지연시켜 생성된 음성을 시각적 요소와 잘 맞출 수 있게 해줍니다.

- **Performance Highlights**: 실험 결과, 이 시스템은 평균 발화 간 침묵 시간을 9.6초에서 0.3초로 줄였습니다. 또한, 사용자 연구에 따르면 경험이 풍부한 게임 플레이어 120명을 대상으로 진행된 평가에서 발화 리듬이 유의미하게 개선되었음을 확인하였습니다. 이러한 성과는 전문적인 발음과 유사한 패턴을 보여주는 데에 기여하였습니다.



### SkillCAT: Contrastive Assessment and Topology-Aware Skill Self-Evolution for LLM Agents (https://arxiv.org/abs/2606.13317)
Comments:
          9 pages, 6 figures

- **What's New**: 이번 논문에서는 SkillCAT을 제안합니다. SkillCAT은 효과적인 기술 자가 진화를 위한 훈련이 필요 없는 프레임워크로, 기술의 생명 주기를 세 가지 관찰 가능한 단계로 분리합니다. 이 방법은 여러 가지 탐색 경로를 수집하고, 고급 기술 패치를 검증하여, 완료되는 작업에 관련된 정보만을 참조하도록 설계되었습니다.

- **Technical Details**: SkillCAT의 첫 번째 단계인 Contrastive Causal Extraction (CCE)에서는 각 작업에 대해 다중 경로를 샘플링하고, 성공/실패 쌍을 비교하여 결과 차이를 설명하는 증거를 추출합니다. 두 번째 단계인 Assessment-Augmented Evolution (AAE)에서는 각 후보 패치를 과거 작업의 클론에서 재생하여 성과를 평가하고, 결과를 개선하거나 유지하는 패치만을 선택합니다. 마지막으로, Topology-Aware Task Execution (TTE) 단계에서는 발전된 기술을 라우팅 가능한 능력 노드의 토폴로지로 컴파일합니다.

- **Performance Highlights**: SkillCAT은 SpreadsheetBench, WikiTableQuestions 및 DocVQA와 같은 일반적인 에이전트 벤치마크에서 평가되었습니다. 그 결과, SkillCAT은 평균 40.40%까지 성능을 개선하여 기초선보다 현저히 높은 성과를 보여주었습니다. 또한, SkillCAT에서 발전된 기술들은 다양한 모델 및 분포 외 일반화 테스트에서도 유효성을 유지하며 강력한 성능을 검증했습니다.



### RogueAI: A Reverse Turing Test for Detecting Licensed AI Deception in Dialogu (https://arxiv.org/abs/2606.13310)
- **What's New**: 본 논문에서는 Turing Test의 현대적 변형인 RogueAI를 제안합니다. 이는 한 인간 플레이어가 두 개의 대화형 Large Language Model 에이전트 중 하나를 질문하여 어떤 에이전트가 속이고 있는지를 식별하는 과정으로 구성됩니다. 더불어 AutoRogueAI라는 절차적 확장을 도입하여 플레이어가 특성과 속임수 전략을 공동으로 설계할 수 있게 합니다.

- **Technical Details**: RogueAI는 인터랙티브한 웹 애플리케이션으로, 한 명의 인간이 두 개의 LLM 에이전트를 질문하는 구조를 갖습니다. 이 과정에서 플레이어는 한 에이전트가 속이기로 되어 있는 제한된 시나리오 내에서 그들을 구별해야 합니다. 결과적으로, 플레이어는 56.6%의 정확도로 속임수를 탐지할 수 있었으며, 이는 재무장된 언어적 신호와 관련이 있습니다.

- **Performance Highlights**: 이 논문의 3일 파일럿 배포에서 415회의 세션이 완료되었으며, 56.6%의 인간 탐지 정확도와 함께 1,876번의 상호작용이 있었습니다. 결과적으로, 속이는 에이전트는 짧고 불확실하게 응답하며 도움이 덜 되는 특정 언어적 특징을 나타냈으나, 인간 플레이어는 이를 효과적으로 활용하지 못했습니다. 이는 데이터 수집 수단으로서의 RogueAI의 활용 가능성과 한계에 대한 중요한 통찰을 제공합니다.



### Evaluating Pluralism in LLMs through Latent Perspectives (https://arxiv.org/abs/2606.13254)
Comments:
          Pluralistic Alignment Workshop @ ICML 2026

- **What's New**: 본 연구는 다양한 관점을 표현하는 데 있어 증가하는 필요성을 반영하여 다중 층 구조의 프레임워크를 소개하고 구현합니다. 이 프레임워크는 LLM(대형 언어 모델)에서 생성된 텍스트의 관점 간 차이를 회화적으로 평가할 수 있도록 도와줍니다. 특히, 기존의 다른 접근 방식에서는 관점의 복잡성을 충분히 반영하지 못한 반면, 제안된 시스템은 더 세밀한 분석을 가능하게 합니다.

- **Technical Details**: 이 프레임워크는 주제 불문(domain-agnostic)으로 작동하며, 텍스트에서 추출된 구성 요소(aspects)를 조합하여 기본적인 관점을 형성합니다. 첫 번째 수준에서는 개별 담론 단위의 표현을 그룹화하여 반복적인 의미 패턴을 포착하고, 두 번째 수준에서는 이러한 패턴을 간략하게 서술하는 범주적 라벨을 형성하여 관점을 특성화합니다. 이를 통해 LLM과 인간 텍스트 간의 관점 분포를 비교하는 데 필요한 정량적 분포 측정 및 주제 분석이 가능해집니다.

- **Performance Highlights**: 연구 결과, 일부 모델과 프롬프트 기법은 다양한 관점을 포괄하는 데 가까이 다가서지만, 여전히 드문 관점은 과소 재현되어 있음을 보여줍니다. 이에 따라 모델의 다양성을 평가하는 데 있어 관점의 중요성이 강조됩니다. 연구에서는 여러 LLM을 비교 평가하였고, 향후 보다 포괄적인 관점을 보장하기 위한 방법론적 기초를 제공합니다.



### PolyAlign: Conditional Human-Distribution Alignmen (https://arxiv.org/abs/2606.13227)
Comments:
          20 pages, 4 Figures, 8 Tables

- **What's New**: 이 논문에서는 전통적인 post-training 방법들이 언어 모델의 응답을 단일한 전역 보조 행동에 맞추는 문제를 다루고 있습니다. PolyAlign이라는 새로운 프레임워크를 소개하여, 모델이 언어와 상호작용 맥락에 따라 적절한 인간 응답 분포를 맞추도록 합니다. 이를 통해 다국어 대화 환경에서의 자연스러운 응답과 분포적 충실성을 개선할 수 있습니다.

- **Technical Details**: PolyAlign은 응답 데이터를 언어, 상호작용 트랙, 응답 유형, 길이에 따라 분류된 인간 참조 분포(bilingual reference distributions)로 구성하여, 조건부 인간-분포 정렬(conditional human-distribution alignment) 모델을 구현합니다. Bucket-Aware SFT와 HDPO를 결합하여 훈련을 통해 다양한 상호작용 체계에서 균형 잡힌 최적화를 도모합니다. 이를 통해 모델은 인간의 실제 응답 스타일을 효과적으로 반영할 수 있습니다.

- **Performance Highlights**: PolyAlign이 포함된 평가에서는 영어와 중국어의 단일-및 다회화 설정에서 자연스러운 응답이 개선되었으며, 여전히 경쟁력 있는 작업 유용성을 유지하고 있음을 확인했습니다. 결과는 post-training이 이제 글로벌 정렬 목표를 넘어, 인간 응답 분포를 고려한 상호작용 인식 정렬으로 이동해야 함을 강조합니다. 이를 통해 다양한 상호작용 컨텍스트에 적합한 응답을 생성하는 모델을 만드는 것이 가능해집니다.



### When Similar Means Different: Evaluating LLMs on Arabic--Hebrew Cognates (https://arxiv.org/abs/2606.13218)
- **What's New**: 이 논문은 아랍어와 히브리어의 단어 쌍에 대한 새로운 벤치마크인 SemCog Bench를 소개합니다. 이 벤치마크는 총 1,858개의 단어 쌍과 함께 코그네이트(cognate) 식별 및 의미 모호성(semaitic disambiguation)에 대한 문장 수준의 주석이 포함되어 있습니다. 이러한 접근법은 대형 언어 모델(LLMs)의 교차 언어 의미 이해(cross-lingual semantic understanding) 능력을 평가하는 데 중점을 두고 있습니다.

- **Technical Details**: 저자들은 여러 입력 표현(raw, diacritized, Romanized, phonetic)을 사용하여 오픈소스 및 상용 LLM을 평가하였습니다. 그 결과, 진짜 코그네이트(true cognates)에 대해서는 높은 정확도를 기록했지만, 거짓 친구(false friends)와 차용어(loanwords)에서는 성능이 급감함을 발견하였습니다. 이는 모델들이 표면적 유사성(surface-form similarity)에 강하게 의존하고 있음을 나타냅니다.

- **Performance Highlights**: 또한, 문장 수준의 맥락(context)에서 약간의 개선이 이루어지기는 했으나, 이러한 맥락적 단서만으로는 잘못된 형태 기반 신호를 극복하기에는 부족함을 시사합니다. 이 연구 결과는 교차 언어적 형식과 의미 간의 충돌을 해결하는 데 있어 현재 LLM의 근본적인 한계를 드러내며, SemCog Bench를 다국어 의미 추론(multilingual semantic reasoning)에 대한 엄격한 벤치마크로 떠오르게 합니다.



### Layer-Resolved Optimal Transport for Hallucination Detection in NMT and Abstractive Summarization (https://arxiv.org/abs/2606.13216)
Comments:
          Accepted to ICML Mechanistic Interpretability Workshop 2026

- **What's New**: 본 논문에서는 Wasserstein 거리 (Wasserstein distance, OT)를 활용해 Fairseq DE-EN 모델의 모든 6개 디코더 레이어에서 Hallucination을 탐지하는 방법을 제안합니다. Wass-to-Unif와 Wass-to-Data는 보완적인 감지기이며, L1-L4 레이어에서 감지 성능이 집중되고 L5 레이어는 더 미세한 타입에 대해서는 역예측적(Anti-predictive)인 성질을 보입니다. 이 연구는 NMT에서 추상적 요약의 신뢰성 탐지로의 OT 신호 전이 가능성도 평가합니다.

- **Technical Details**: 논문에서는 먼저 Fairseq DE-EN 모델의 모든 6개 디코더 레이어를 분석하며 새로운 탐지기인 Routing consistency를 추가합니다. GT에 명시된 대로, W1 거리는 두 확률 분포 간의 최소 ‘작업’ 양을 측정하며, attention mass의 공간 구조에 민감합니다. 또한, 디코더의 각 레이어에서 교차 주의 배포의 기하학적 구조를 분석하기 위하여 각 레이어 간의 W1 거리 측정을 수행합니다.

- **Performance Highlights**: OT를 기반으로 한 Hallucination 탐지의 첫 적용은 AggreFact 벤치마크에서 검토되었으며, 57.2%~57.6%의 균형 정확도를 달성했습니다. 하지만, 이는 감독된 MiniCheck 모델의 69.9%~74.3%와 비교할 때 상당히 낮은 성과입니다. 이를 기반으로 NMT와 요약 간의 전이가 부분적임을 강조하며, 신뢰성 결함의 태도 차이를 통해 이 틈을 설명합니다.



### SICI: A Semantic-Pragmatic Complexity Index Reveals Regime Shifts in LLM Stance Detection (https://arxiv.org/abs/2606.13189)
- **What's New**: 이 논문은 스탠스(stance) 탐지를 위한 새로운 복잡도 지표인 SICI (Stance Inference Complexity Index)를 도입합니다. SICI는 단순한 스탠스 레이블 선택 문제가 아닌, 의미와 화용적 복잡성을 기반으로 한 7개의 차원에서 평가합니다. 이 지표는 LLM의 정확도를 예측하는 데 있어 기존의 표면적 프록시보다 더 뛰어난 성능을 보여줍니다.

- **Technical Details**: SICI는 타겟 가시성, 범위 정렬, 화용적 암시성, 지식 요구, 맥락 의존성, 라벨 모호성, 정서-스탠스 간의 수치 차이 등을 포함한 총 7개의 차원으로 구성됩니다. 모델의 오류는 SICI 점수가 증가함에 따라 비선형적인 패턴을 보이며, 이는 서로 다른 복잡성 수준에서 LLM의 예측 오류 형태가 변화함을 나타냅니다.

- **Performance Highlights**: SICI의 주요 발견은 LLM의 오류가 복잡도에 따라 급격하게 변화한다는 것입니다. 저 복잡도에서는 스탠스를 과도하게 부여하는 경향을 보이고, 고 복잡도에서는 None 예측으로 집중되는 경향이 있습니다. prompting, retrieval, debate와 같은 다양한 개입 방법이 높은 복잡도 문제를 해결하는 데 효과적이지 않은 경향이 있음을 보여줍니다.



### A Context-Aware Dataset for Stance Detection in Bioethical Controversies on Redd (https://arxiv.org/abs/2606.13187)
- **What's New**: BioStance는 Reddit의 생명 윤리 논의에서 39,600개의 주석이 달린 'Post-Comment' 쌍으로 구성된 대규모 맥락 인식 데이터세트입니다. 이 데이터는 생명 윤리적 논쟁에서 광범위한 맥락 의존적 담론을 모델링하기 위한 고품질 자원을 제공합니다. 데이터는 개인의 자유와 집단 책임, 기술적 불확실성 등 세 가지 차원에서 여섯 가지 논란이 되는 주제를 포괄합니다.

- **Technical Details**: BioStance는 Reddit의 스레드 구조를 사용하여 논란이 되는 생명 윤리적 주제에 대한 문맥 의존적 발언 표현을 모델링합니다. 데이터셋은 각 ‘Post-Comment’ 쌍에 대해 세 개의 독립된 주석자가 세 가지 범주로 주석을 달아, 평균 Krippendorff's α 값이 0.82로 상당한 신뢰성을 확보하고 있습니다. 데이터 전처리 단계에서 비관련 내용, 중복, 형식 비표준 등을 걸러내어 데이터의 품질을 높였습니다.

- **Performance Highlights**: BioStance는 논란이 되는 생명 윤리적 이슈의 고유한 맥락을 제공함으로써, 컴퓨터 소셜 사이언스 및 논증 탐사에 대한 연구를 지원합니다. 데이터셋의 각 주제가 분명한 기술적 또는 기관적 고려 사항을 포함하여 경향성을 드러내는 반면, 기본적인 도덕적 갈등으로 인해 주관성이 더욱 두드러져 있습니다. 따라서 BioStance는 생명 윤리 담론의 복잡성을 반영하여 실질적인 연구에 기여합니다.



### LAUKIN: A Multi-jurisdictional Common Law Contract Datas (https://arxiv.org/abs/2606.13184)
Comments:
          5 pages, 2 figures, 4 tables

- **What's New**: LAUKIN (Legal equivalence dataset of Australia, UK, and INdia)는 다국적 기업들이 필요로 하는 cross-jurisdictional (법적 관할권 간) 계약 검토를 지원하기 위한 새로운 데이터셋입니다. 이는 오스트레일리아, 영국, 인도의 법률 조항 쌍으로 구성되어 있으며, boolean legal equivalence (불리언 법적 동등성)으로 라벨링되어 있습니다. LAUKIN 데이터셋은 204개의 계약에서 추출된 14,727개의 조항 쌍을 포함하고 있으며, 3,000개는 전문가에 의해 수동으로 라벨링되었습니다.

- **Technical Details**: LAUKIN 데이터셋은 새로운 multi-stage retrieval and reranking pipeline (다단계 검색 및 재순위 프로세스)을 통해 구성되었습니다. 초기 조항 쌍 매핑을 구축하고 나중에 법률 전문가에 의해 Equivalent 또는 Not Equivalent로 주석이 달린 부분이 포함됩니다. 데이터셋은 8가지 계약 유형을 포함하며, 총 14,727개의 조항 쌍이 포함되어 있습니다.

- **Performance Highlights**: 실험 결과, 12개 모델을 4개 기법으로 평가한 결과, 가장 높은 macro-F1 점수인 65.11%를 달성하였습니다. 이는 LAUKIN이 도전적인 기준선 (benchmark)으로 자리 잡았음을 보여줍니다. 연구 결과, 각 법적 관할권 간의 초안 관행이 크게 다르다는 것을 발견했으며, 이는 cross-jurisdictional equivalence classification (법적 관할권 간의 동등성 분류)을 어렵게 만듭니다.



### MemRefine: LLM-Guided Compression for Long-Term Agent Memory (https://arxiv.org/abs/2606.13177)
- **What's New**: 이 논문에서는 장기 상호작용(long-term interactions)에서 사용되는 대규모 언어 모델(LLM) 에이전트의 메모리 관리 문제를 다룹니다. 기존 메모리 저장소가 시간에 따라 증가함에 따라 중복 항목이 쌓이고, 이는 저장 비용을 증가시키고 유용한 정보를 검색하는 데 방해가 됩니다. 이러한 문제를 해결하기 위해, 저장 예산(storage-budgeted) 내에서 정보의 유용성을 보존하는 메모리 관리 방법을 제안합니다.

- **Technical Details**: 논문에서는 MemRefine이라는 LLM 기반 프레임워크를 제안합니다. 이 프레임워크는 정보의 사실적 가치(factual value)를 유지하기 위해 유사성을 기반으로 후보 쌍(candidate pairs)을 제안하며, 삭제(delete), 병합(merge), 및 보존(preserve) 결정을 LLM 판별자(LLM judge)에 위임합니다. 이 방법은 예산이 충족될 때까지 반복(iteration)하여 메모리 관리를 최적화합니다.

- **Performance Highlights**: 다양한 메모리 프레임워크와 장기 대화 벤치마크에서 MemRefine은 목표 예산을 지속적으로 충족하며 하위 성능(downstream performance)을 유지했습니다. 또한, 엄격한 예산 아래에서 기존의 규칙 기반(rule-based) 기준을 능가하는 성과를 보였습니다.



### NTS-CoT: Mitigating Hallucinations in LLM-based News Timeline Summarization with Chain-of-Thought Reasoning (https://arxiv.org/abs/2606.13171)
- **What's New**: 이 논문에서는 LLM 기반의 Timeline Summarization (TLS)의 주요 문제인 hallucinations(환각) 현상에 대한 해결책을 제시합니다. 기존 연구에서는 hallucinations의 유형에 대한 체계적인 분석이 부족했던 반면, 두 가지 주요 유형인 불신 내용 불일치와 정보 누락을 식별했습니다. 이를 해결하기 위해 새로운 프레임워크인 NTS-CoT를 도입하여 Chain-of-Thought(사고의 연쇄) 추론 기법을 활용합니다.

- **Technical Details**: NTS-CoT는 크게 세 가지 모듈로 구성됩니다: i) Element-CoT는 중요한 뉴스 요소를 포착하여 신뢰성 있는 요약을 생성하고, ii) Date Selection은 시간의 중요성과 사건의 두드러짐을 결합하여 타임스탬프 선택을 최적화하며, iii) Causal-CoT는 여러 문서 간 인과 관계를 추론하여 날짜-사건 요약에서 정보 누락을 줄입니다. 이러한 구조는 LLM이 생성하는 요약의 정확성과 일관성을 높입니다.

- **Performance Highlights**: NTS-CoT는 세 가지 TLS 기준에서 정량적 분석과 인간 평가를 통해 성능이 입증되었습니다. NTS-CoT는 SOTA(SOTA: State-Of-The-Art) 기준선에 비해 AR-1이 23.4%, AR-2가 33.4%, Date-F1이 10.0% 개선되었습니다. 또한, 인간 평가에서도 NTS-CoT의 요약이 신뢰성을 기준으로 67.74%, 완전성을 기준으로 54.38%의 비율로 우수성을 인정받았습니다.



### HyPE: Category-Aware Hypergraph Encoding with Persistent Edge Embeddings for Persona-Grounded Dialogu (https://arxiv.org/abs/2606.13142)
Comments:
          11 pages, 2 figures, 4 tables

- **What's New**: 이 논문에서는 HyPE(Hypergraph Persona Encoder)라는 새로운 프레임워크를 제안하여, 개인화된 대화 시스템에서 개인의 특성에 부합하는 응답을 생성하는 방식을 개선합니다. 기존 방법들이 퍼소나(persona)를 단순한 문장 집합으로 취급하였지만, HyPE는 고차원적인 (Core, Expression, Sentiment, Category) 구조로 각 퍼소나 문장을 분석합니다. 이 프레임워크는 하이퍼그래프(hypergraph) 구조를 사용하여 퍼소나 요소들 간의 관계를 명확하게 모델링하며, 공유 카테고리 라벨에 의해 하이퍼엣지(hyperedge)를 생성합니다.

- **Technical Details**: HyPE는 다음 세 가지 주요 단계로 구성됩니다: (1) 퍼소나 분석, (2) 퍼소나 하이퍼그래프 구축, (3) 하이퍼그래프 신경망(HGNN)을 통한 개인화된 응답 생성. 퍼소나의 네 가지 요소(코어, 표현, 감정, 카테고리)를 기반으로 각 문장을 쪼개어 하이퍼그래프를 구축합니다. 이를 통해 여러 퍼소나 문장이 공유하는 의미적 속성을 효과적으로 모델링하며, HGNN을 통해 학습된 구조적 정보는 응답 생성 모델에 통합되어 개인화된 응답을 생성하도록 돕습니다.

- **Performance Highlights**: 모델의 효율성을 입증하기 위해 PersonaChat 데이터셋에서 실험한 결과, HyPE는 GPT-2, LLaMA-3.2-3B 및 Qwen2.5-3B와 같은 다양한 백본(backbone) 모델에서 시퀀스 기반의 풀링(pooling) 방법보다 일관되게 우수한 성능을 보였습니다. 특히, 다양한 모델 스케일 간의 전이 가능한 이점을 보여주며, 기존의 HyPE-base 모델에 비해 더 나은 성능을 발휘합니다. 이러한 실험 결과는 구조적인 하이퍼엣지 기반 퍼소나 인코딩이 대화 생성 작업에 큰 기여를 할 수 있음을 시사합니다.



### NaturalFlow: Reducing Disruptive Pauses for Natural Speech Flow in Simultaneous Speech-to-Speech Translation (https://arxiv.org/abs/2606.13121)
Comments:
          Proceedings of the 26th Interspeech Conference, Long Paper

- **What's New**: 이번 논문에서는 유창성을 고려한 최적화 프레임워크를 도입하여 동시 통역의 고유한 문제를 해결합니다. 이를 통해 낮은 지연 시간에서도 자연스러운 발음을 유지할 수 있는 최적 균형을 발견합니다. 기존의 S2ST(Speech-to-Speech Translation) 모델들이 단편적인 언어 출력을 생성하는 문제를 해결하고자 하며, 보다 매끄러운 발화를 가능하게 합니다.

- **Technical Details**: 제안된 NaturalFlow 모델은 'Direct Preference Optimization (DPO)'을 통해 훈련되며, 이는 정지 비율을 최소화하면서도 번역의 충실도를 유지하는 두 개의 상충되는 목표를 최적화합니다. 이 모델은 'Large Language Models (LLMs)'의 생성을 활용하여 자연스러운 발화를 위한 다양한 패러프레이즈를 생성할 수 있습니다. 여러 벤치마크 테스트를 통해 이 프레임워크는 침묵 비율을 줄이고, 번역 품질과 대기 시간 관련 메트릭을 유지하면서 성능을 검증하였습니다.

- **Performance Highlights**: 인간 평가 결과, 제안된 S2ST 모델이 기존 시스템에 비해 선호도가 높았으며, 연속적이고 자연스러운 음성 출력을 생성함을 보여주었습니다. 다양한 도메인과 발화 길이를 아우르는 네 가지 벤치마크에서 성능이 입증되었습니다. 이러한 결과는 정보 내용이 유지되면서도 논리적 흐름이 매끄럽고 자연스럽다는 것을 나타냅니다.



### EvoBrowseComp: Benchmarking Search Agents on Evolving Knowledg (https://arxiv.org/abs/2606.13120)
Comments:
          14 pages, under review

- **What's New**: 이번 논문에서는 웹 탐색 도구로 보강된 대형 언어 모델(LLMs)을 사용한 검색 에이전트의 평가 기준으로 새롭게 EvoBrowseComp를 소개합니다. 기존의 정적 지식 기반 벤치마크들은 모델이 사실을 회상하는 데 의존하게끔 만들어 효과적으로 웹 탐색 능력을 평가하지 못했습니다. EvoBrowseComp는 400개의 영어 및 400개의 중국어 복잡한 질문을 실시간 웹 탐방을 통해 생성하며, 이는 데이터 오염을 차단합니다.

- **Technical Details**: EvoBrowseComp는 세 가지 전문 에이전트로 구성된 협력 프레임워크를 사용하여 QA 쌍 생성을 자동화합니다. 첫 번째로, QA 합성 에이전트는 웹에서 새로운 지식을 검색하여 QA 쌍을 생성합니다. 두 번째로, 정보 필터링 에이전트는 신뢰성과 인기 측면에서 검색된 지식을 필터링하여 파라메트릭 단축을 방지합니다. 마지막으로, 고급 가이드 에이전트는 질문을 논리적 구조로 형식화하여 논리적 중복성과 단축을 줄입니다.

- **Performance Highlights**: 실험에서 최첨단 LLM인 Claude-Opus-4.6는 도구 사용 시 44.8%의 정확도밖에 달성하지 못했으며, 도구 접근이 제한될 경우 성능이 6.0%로 급감했습니다. 이는 EvoBrowseComp가 최신 지식에 대한 진정한 검색 및 다단계 추론을 필요로 함을 보여줍니다. 이 연구는 지속 가능한 평가 기준을 수립하는 데 기여하며, 데이터 오염에 강한 평가 프레임워크로 자리매김할 수 있습니다.



### G-Long: Graph-Enhanced Memory Management for Efficient Long-Term Dialogue Agents (https://arxiv.org/abs/2606.13115)
Comments:
          22 pages, 8 figures, 14 tables

- **What's New**: 본 논문에서는 기존의 대화 시스템에서 나타나는 긴 대화의 일관성을 유지하는 데 필요한 장기 기억을 위한 새로운 프레임워크인 G-Long을 제안합니다. G-Long은 구조화된 트리플렛(triplet) 추출 및 연관 검색(associative retrieval)을 위해 미세 조정된 소형 언어 모델(sLM)을 활용하여 작동 비용을 크게 줄입니다. 또한, T5 요약기의 내부 교차 주의 신호를 활용해 중요한 기억을 식별하는 새로운 주의 기반 중요도 점수 매기기 메커니즘을 도입하여 성능을 향상시킵니다.

- **Technical Details**: G-Long은 네 가지 주요 구성 요소로 이루어져 있습니다: (1) 효율적 메모리 구축, (2) 그래프 기반 메모리 뱅크, (3) 연관 메모리 검색, (4) 응답 생성입니다. 대화를 구조화된 그래프 표현으로 변환하기 위해 트리플렛 추출 모듈을 사용하여 각 발화를 분석하고, 주의 기반 중요도 점수 매기기 모듈을 통해 추출된 트리플렛에 중요도 점수를 부여합니다. 이 과정에서 조정된 소형 언어 모델을 활용하여 원시 발화를 구조화된 사실로 변환합니다.

- **Performance Highlights**: G-Long은 MSC, CC, LoCoMo, LME와 같은 다양한 벤치마크에서 실험을 통해 응답 생성 및 메모리 검색 모두에서 최첨단 성능을 달성했습니다. 응답 품질에서 MSC 데이터셋에 대해 9.8% 향상된 결과를 보였으며, LME 벤치마크에서는 검색 성능이 40.8% 개선된 것으로 나타났습니다. 이러한 성과는 기존 시스템의 계산 비용을 크게 줄이면서도 응답 품질을 유지하는 데 기여합니다.



### MÖVE: A Holistic LLM Benchmark for the German Public Sector (https://arxiv.org/abs/2606.13111)
- **What's New**: 본 논문에서는 독일 공공 부문에 맞춰 설계된 MÖVE (Modelle für die Öffentliche Verwaltung Evaluieren)라는 포괄적인 벤치마크를 소개합니다. LLM (Large Language Models)이 공공 관리에 점점 더 많이 도입되고 있는 반면, 기존 벤치마크는 영어와 미국 중심으로 한 내용을 가지고 있어, 공공 부문에 대한 적절한 모델 선택에 대한 가이드를 제공하지 못하고 있습니다. MÖVE는 39개의 모델을 세 가지 성능 기준(요약, 질문 답변, 주제 추출)과 네 가지 거버넌스 기준(환상 경향, 에너지 소비, 제공자 투명성, 독일 헌법 가치 준수)으로 평가합니다.

- **Technical Details**: MÖVE는 독일어로 된 10개의 데이터셋을 활용하며, 금색 및 은색 표준 데이터셋을 포함하여 공공 관리 분야를 반영하도록 특별히 구성되었습니다. 평가 전략은 전통적인 NLP 메트릭, 임베딩 기반 방법, LLM을 판별자로 활용하는 방식을 조합한 다중 메트릭 평가 접근 방식을 사용하고 있습니다. 이 논문은 모델 순위 보고에 그치지 않고, 벤치마크 자체에 대한 방법론적 자기 평가를 수행하여 통계적 정확성, LLM 판별자 신뢰성, 개인 데이터셋이 모델 순위에 미치는 영향 등을 분석합니다.

- **Performance Highlights**: MÖVE의 결과는 어느 특정 모델이 모든 기준을 압도하지 않으며, 성과 좋은 모델들이 과제마다 다르다는 것을 보여줍니다. 모델의 크기만으로 품질을 예측하는 것이 불충분하다는 점도 강조되고 있습니다. 이 벤치마크는 지속적인 개발 중이며, 그 결과는 웹사이트를 통해 공개적으로 제공됩니다. 향후 논문에서는 MÖVE의 첫 번째 포괄적인 평가 결과를 공유하며, 결과는 지속적으로 업데이트됩니다.



### LEDGER: A Long-Context Benchmark of Corporate Annual Reports for Grounded Financial Retrieval and Extraction (https://arxiv.org/abs/2606.13100)
Comments:
          5 pages, 1 figure

- **What's New**: 이번 연구에서는 최근 대형 언어 모델(large language models, LLMs)의 성능을 평가하기 위해 4,999개의 디지털화된 기업 연례 보고서로 구성된 LEDGER 데이터셋을 출시했습니다. 이 데이터셋은 단순한 SEC 10-K 파일 형식이 아닌, 시각적으로 복잡한 문서와 다양한 KPI(핵심 성과 지표) 정보를 포함하고 있습니다. 따라서 이러한 자료를 통해 금융 자산 관리 및 자동화된 분석 도구 개발에 기여할 수 있는 가능성이 커졌습니다.

- **Technical Details**: LEDGER 데이터셋은 과거 2009년부터 2024년까지 738개 기업의 연례 보고서를 포함하며, 각 보고서에는 31개의 통합 KPI가 주석으로 달려 있습니다. 이 데이터셋은 페이지 수준의 KPI 검색, 단일 값 조회 및 전체 KPI 추출로 구성된 세 가지 벤치마크를 제공하며, 인간적 OCR 품질 주석을 신뢰할 수 있는 방식으로 제공합니다. 또한, 금융 시장 반응과 연결된 연구적 활용 사례도 포함되어 있습니다.

- **Performance Highlights**: LEDGER는 OCR 기반으로 작성된 표와 데이터를 통해 118,048개의 KPI를 포함하는 사실 정보로 구성되었으며, 각 KPI는 자연어 질문으로 변환되어 사용됩니다. 이 데이터셋은 금융 분석 및 정보 검색(Retrieval and Generation, RAG) 작업에서 우수한 성능을 발휘할 수 있도록 검증되었습니다. 특히 CEO 편지의 수사학과 게시 후 시장 영향 간의 관계를 구체적으로 분석한 사례 연구 또한 제공됩니다.



### sebis at CRF Filling 2026: A Two-Stage Local LLM Pipeline for Medical CRF Filling (https://arxiv.org/abs/2606.13082)
Comments:
          Published in Proceedings of the Third Workshop on Patient-Oriented Language Processing (CL4Health), LREC 2026

- **What's New**: 본 논문은 비구조적 전자의무기록(EHR) 주석에서 구조화된 임상 정보를 추출하는 어려움을 해결하기 위해 MedGemma-27B 모델을 사용하는 완전 로컬, 도메인 적응형 파이프라인을 제안합니다. 이 연구는 CL4Health 2026에서 dyspnea 환자를 위한 Case Report Form (CRF) 자동 작성의 과제를 다루며, 개인 정보 보호를 보장하면서도 경쟁력 있는 성능을 달성합니다. 1단계 접근 방식을 통해 자료를 과도하게 예측하지 않고, 실질적인 임상 데이터의 정밀한 추출을 실현합니다.

- **Technical Details**: 이 연구는 두 단계 아키텍처를 활용하여 이진 존재 분류와 값 추출을 분리하고, 텍스트 증거에 따라 엄격한 준수를 보장합니다. MedGemma-27B 모델은 GHUF Q8_K_XL 양자화가 적용되어 있으며, 기온(temperature)을 0으로 설정하여 일관된 출력을 보장합니다. 매번 10개의 few-shot 예제를 활용하여 분류와 값 추출을 수행하며, 이는 데이터 소속성과 HIPAA/GDPR 규정 준수를 달성하면서도 임상 분야 지식을 유지합니다.

- **Performance Highlights**: 최종 제출에서 CL4Health 2026 CRF-작성 공유 과제의 매크로 F1 점수는 0.55로, 전체 32개 제출 중 17위에 해당합니다. 영어 전용 트랙에서는 22 참가 중 11위로, 완전히 로컬이며 오픈 소스 시스템 중에서는 2위를 차지하여 외부 API나 독점 모델을 활용하지 않고도 높은 경쟁력을 보였습니다. 이 연구 결과는 개인 정보를 보호하면서도 임상 분야에 적합한 모델이 현실적으로 가능함을 보여줍니다.



### No Hidden Prompts Needed! You Can Game AI Peer Review with Presentation-Only Revisions (https://arxiv.org/abs/2606.13044)
Comments:
          35 pages, 5 figures

- **What's New**: 이 연구에서는 동료 심사(d)이 AI 생성 리뷰가 실험 도구에서 동료 심사 인프라로 진화하면서 나타나는 새로운 위험을 탐구합니다. 특히, 저자들이 논문의 과학적 결과에는 변화를 주지 않고 발표 수준의 콘텐츠만 수정함으로써 AI 리뷰 점수를 시스템적으로 개선할 수 있는 가능성을 분석합니다. 이를 통해 'adversarial repackaging'이라는 개념을 도입하고, 이를 활용한 공격이 75.1%의 성공률을 기록하며, 평균 점수에서 +1.21의 증가를 달성했음을 밝혔습니다.

- **Technical Details**: 이 연구는 AI 리뷰어의 피드백을 활용하여 발표 수준의 수정 작업을 적절히 최적화하는 방법을 제시합니다. 이러한 방법론은 세 개의 주요 편집 구역으로 구성됩니다: 발표 프레이밍을 수정할 수 있는 자유 구역, 과학적 내용을 보존해야 하는 제한 구역, 그리고 과학적 데이터를 포함한 고정 구역입니다. 연구팀은 이 과정을 통해 AI 리뷰어의 검토 프로세스에서 나타나는 비효율성과 구조적 결함을 드러냈습니다.

- **Performance Highlights**: 현재 AI 리뷰어는 발표 최적화를 본질적 개선으로 착각하고, 이로 인해 연구자들이 연구 개선보다 논문 포장 최적화로 전환하도록 유도하는 경향이 있습니다. 연구 결과는 여러 주류 리뷰어 모델과 템플릿에서 일관되게 나타나, 특정 모델의 결함이 아니라 AI 리뷰어 전반의 구조적 분석을 요구합니다. 최종적으로, 연구팀은 AI 리뷰 시스템의 견고성을 테스트하기 위한 재사용 가능한 벤치마크인 오염-free 롤링 데이터세트를 구축하였습니다.



### SkillChain: Closing the Loop on Skill Evolution for Image-Based E-Commerce AI Assistants (https://arxiv.org/abs/2606.12984)
- **What's New**: 이 논문은 이미지 기반 AI 어시스턴트의 기능 진화를 자동화하는 SkillChain을 제안합니다. 이 시스템은 사용자가 이미지를 업로드하고 다양한 사용자 의도에 따른 응답을 생성할 때 발생하는 여러 가지 문제(C1-C3)를 해결합니다. SkillChain은 세 가지 스테이지로 구성되어 있으며, 각 스테이지는 독립적으로 작동하여 성능을 개선합니다.

- **Technical Details**: SkillChain은 세 가지 주요 단계로 나눌 수 있습니다: 스킬 생성기(Skill Creator)는 작업 사양(Task Specification)과 사용자 경로(User Trajectories)를 기반으로 스킬을 부트스트랩하고, 라우팅 최적화기(Route Optimizer)는 라우팅 실패를 분석하여 업데이트 및 정정 작업을 수행하며, 바디 정제기(Body Refiner)는 이중 경로 평가를 통해 바디의 결함을 식별 및 수정합니다.

- **Performance Highlights**: SkillChain은 대규모 상업 이미지 어시스턴트에서 배포되어 응답 품질을 크게 향상시키고, 특히 구조적 준수와 내용 품질에서 가장 큰 이익을 얻었습니다. A/B 테스트 결과, 사용자 참여도, 콘텐츠 소비 및 장기 유지율에도 상당한 개선이 관찰되었습니다.



### Multi-Turn Reasoning When Context Arrives in Pieces: Scalable Sharding and Memory-Augmented RL (https://arxiv.org/abs/2606.12941)
- **What's New**: 이 논문은 사용자가 여러 대화 턴을 통해 중요한 정보를 점진적으로 공개할 때 LLM의 정확도가 최대 65%까지 떨어진다는 결과를 보입니다. 이를改善하기 위해 모델이 점진적인 메모리(rolling memory)를 유지하도록 학습하면 성능을 크게 향상시킬 수 있음을 보여줍니다. 저자들은 저비용의 sharding 파이프라인을 도입하여 단일 턴 QA 데이터셋을 다중 턴으로 분산된 에피소드로 변환하며, 이 과정에서 수작업 주석 작업의 필요성을 없앴습니다.

- **Technical Details**: 논문에서 제안하는 메모리 보조 정책은 GSM8K 데이터셋에 대해 Lost in Conversation(LiC) 감소를 통해 멀티 턴 정확도를 크게 향상시키는 방법을 제시합니다. 데이터셋 구조는 세 단계의 프롬프트 파이프라인을 통해 구축되며, 각 문제는 논리적 단위로 분리되고 반복적으로 검증됩니다. 메모리 추출을 유도하기 위해 저자들은 정책이 메모리 상태를 유용하게 작성하도록 학습해야 하며, 이를 통해 최종 답변의 정확성을 극대화합니다.

- **Performance Highlights**: 실험 결과, 메모리 보조 모델은 전체 역사 모델보다 항상 뛰어난 성능을 보이며, 특히 어려운 수학 문제와 도메인 외의 긴 컨텍스트 QA에 대해 zero-shot으로 일반화됩니다. 즉, 메모리 훈련을 받은 모델은 테스트 시 전체 대화 이력을 제공받는 경우에도 더 우수한 성능을 나타내었으며, 이는 훈련 자체에서 메모리 압축이 향상된 추론 능력을 강화하는 것으로 나타났습니다.



### Polar: A Benchmark for Evaluating Political Bias in LLMs (https://arxiv.org/abs/2606.12922)
Comments:
          Submitted to ARR 2026 May cycle

- **What's New**: 본 논문에서는 정치적 편향을 측정하기 위한 새로운 다지선다형 벤치마크인 Polar를 소개합니다. Polar는 4,026개의 평가 사례를 통해 미국과 한국의 정치적 맥락에서 모델을 평가합니다. 이 벤치마크는 프롬프트 기반 생성 대신 옵션 수준의 확률을 통해 정치적 편향을 측정합니다. 또한 모델의 편향은 정치적 맥락과 문제 범주, 모델 그룹, 표현 언어에 따라 체계적으로 변화함을 보여줍니다.

- **Technical Details**: Polar는 두 개의 이념 축과 여덟 개의 이슈 범주를 포함하고 있으며, 이는 Manifesto Project에서 파생된 것입니다. 이 연구에서는 재현 가능한 옵션 수준의 확률 절차를 사용하여 정치적 편향을 측정합니다. 또한, 경제적 및 사회문화적 입장을 구별하는 이중 축 구조를 통해 모델의 언어 모델링 능력도 함께 평가합니다. 이러한 접근 방식은 프롬프트의 구문이나 해석에 대한 민감도를 감소시킵니다.

- **Performance Highlights**: 38개의 LLM을 평가한 결과, 미국 데이터셋에서는 모든 모델이 좌파-진보적 경향을 보였으나, 한국 데이터셋에서는 보다 중립적이고 혼합된 경향을 보였습니다. 문제 범주별 분석을 통해 집계 축 점수에서는 나타나지 않는 특이한 선호도를 발견할 수 있었습니다. 또한 번역 실험을 통해 표현 언어가 측정된 편향에 직접적인 영향을 미친다는 사실도 확인되었습니다.



### PiDA: Phonetically-Informed Data Augmentation for Robust Vietnamese Speech Translation (https://arxiv.org/abs/2606.12911)
Comments:
          Accepted to INTERSPEECH 2026

- **What's New**: 이 논문에서는 베트남어 음성 번역 시스템에서 발견된 비정상적인 ASR(Automatic Speech Recognition) 오류를 최초로 체계적으로 분류하고 있습니다. ASR에서 발생하는 대체 오류의 원인을 음성(phonetic) 혼동에 따라 분류하고, 이러한 오류가 Neural Machine Translation(NMT) 성능에 미치는 영향을 정량화하였습니다. 또한, 음성 오류가 ST(Speech Translation) 품질을 저해한다는 점을 확인하였습니다.

- **Technical Details**: 이 연구는 phonetic word embeddings를 사용하여 비슷한 음성을 가진 대체 단어로 ASR 오류를 생성하는 Phonetically-Informed Data Augmentation(PiDA) 방법을 제안합니다. 이를 통해 ASR 출력의 정확성을 높이고, 전반적인 ST 품질 개선에 기여합니다. 또한, Linear Mixed-Effects Modelling 방법을 통해 ASR 오류의 영향을 정량화하였습니다.

- **Performance Highlights**: PiDA로 증강된 FLEURS 베트남어-영어 데이터셋으로 미세 조정(fine-tuning) 시, ASR 오류의 번역 성능이 기존 방법보다 최대 +2.04 BLEU 점수 개선을 보여주었습니다. 이 과정은 깨끗한 텍스트 성능도 서서히 향상시켰습니다.



### SENTINEL: Failure-Driven Reinforcement Learning for Training Tool-Using Language Model Agents (https://arxiv.org/abs/2606.12908)
- **What's New**: SENTINEL은 실패 기반의 강화 학습 프레임워크로, 언어 모델 에이전트의 도구 사용 능력을 효과적으로 향상시키기 위한 새로운 접근 방식을 소개합니다. 이 프레임워크는 에이전트가 훈련 중 마주한 실패를 진단하고, 그 정보를 분석하여 훈련에 적합한 과제를 생성하는 방식으로 작동합니다. SENTINEL의 구조는 Controller, Proposer, Solver의 세 가지 구성 요소로 이루어져 있습니다.

- **Technical Details**: 이 프레임워크는 실패한 경로(trajectory)를 분석하는 Controller, 실패를 극복하기 위한 실행 가능한 새로운 작업을 생성하는 Proposer, 그리고 이를 통해 학습하는 Solver로 구성됩니다. 이 구조는 에이전트가 자신의 약점을 극복하기 위해 필요한 도구를 효과적으로 학습하도록 돕습니다. SENTINEL은 구체적으로 Tau2-Bench Retail 도메인에서 평가되며, Qwen3-4B-Thinking-2507 모델을 사용하여 성능 향상이 이루어졌습니다.

- **Performance Highlights**: 실험 결과 SENTINEL은 Pass^1 점수를 66.4에서 74.9로 개선하였고, 전통적인 강화 학습(RL) 방식보다 우수한 성능을 보여주었습니다. 이러한 결과는 모델 실패가 도구 사용 에이전트를 위한 목표 지향적 훈련 신호의 효과적이고 확장 가능한 원천이 될 수 있음을 시사합니다. SENTINEL은 실패를 단순한 평가 오류로 보지 않고, 학습의 중요한 일부로 활용하여 훈련 데이터 생성을 최적화합니다.



### X-MADAM-RAG: Diagnosing and Handling Chinese-English Evidence Conflict in Retrieval-Augmented Generation (https://arxiv.org/abs/2606.12903)
- **What's New**: 이번 논문에서는 Retrieval-augmented generation (RAG) 시스템에서 두 개의 상반된 언어적 증거가 서로 모순될 때 발생하는 문제를 다룹니다. 이를 위해 X-RAMDocs-ZHEN이라는 새로운 벤치마크를 제시하며, 이는 300개의 예제를 포함하여 다양한 증거 조건에서의 증거 충돌을 진단하는 도구로 활용됩니다. 또한, X-MADAM-RAG라는 새로운 해석 가능한 파이프라인을 소개하여 증거 처리 과정을 보다 체계적으로 분석하고 있습니다.

- **Technical Details**: X-RAMDocs-ZHEN 벤치마크는 중국어와 영어의 증거를 기반으로 한 300개의 샘플을 포함하고 있으며, 각 샘플은 여섯 가지 균형 잡힌 증거 조건을 따릅니다. X-MADAM-RAG는 각 문서별 증거 후보 추출, 보이는 증거의 수선, 결정론적 후보 그룹화 및 충돌 인식 집계를 수행하는 프로세스로 구성되어 있습니다. 논문에서는 X-MADAM-RAG가 본래의 벤치마크에서 상당한 성과를 올렸으나, 템플릿 의존도와 같은 제한사항도 드러났습니다.

- **Performance Highlights**: X-MADAM-RAG는 Qwen2.5-7B-Instruct 모델을 사용하여 벤치마크에서 0.9667의 엄격한 정확도를 기록하며, 0.9767의 충돌 인식 성공률을 달성했습니다. 그러나 zero-call 규칙 전용 추출기가 같은 벤치마크에서 1.0000에 도달하여 템플릿 정규성과 관련된 한계를 드러냈습니다. 자연화된 스트레스 테스트를 통해 X-MADAM-RAG의 정확도가 0.3000으로 떨어지며, 이는 문서 수준의 추출이 주요 병목 현상임을 나타냅니다.



### PRISM: Prosody-Integrated Multi-Agent Reasoning Framework for Empathetic Spoken Dialogu (https://arxiv.org/abs/2606.12902)
Comments:
          Accepted to Interspeech 2026

- **What's New**: 이 연구에서는 PRISM이라는 다중 에이전트 프레임워크를 제안하여 감정적으로 공감할 수 있는 대화 생성 문제를 해결합니다. PRISM은 스피치 인식, 응답 생성 및 음성 합성을 분리하여 각각의 구성 요소를 조정하여 감정 표현을 자연스럽게 구현할 수 있습니다. 새로운 프로소디-투-언어 번역 메커니즘은 대화의 안정성을 높이며, 외부 지식 도구를 필요할 때 즉시 호출하여 대화의 질을 향상시킵니다.

- **Technical Details**: PRISM은 네 개의 주요 구성 요소인 감지기(Perceiver), 관리자(Manager), 응답자(Responder), 음성 합성기(Vocalizer)로 구성됩니다. 감지기는 입력된 음성을 구조화된 상태로 변환하여 감정 및 표현 상태를 나타내는 비언어적 속성을 포함합니다. 관리자는 비언어적 속성을 자연어 설명으로 변환하고, 응답자의 출력과 사용자 감정과의 일치를 검증하는 역할을 합니다.

- **Performance Highlights**: 실험 결과, PRISM은 AvaMERG 데이터셋에서 자동 및 인간 평가 모두에서 기존 모델보다 일관되게 우수한 성능을 나타냈습니다. 각 구성 요소가 프로젝트의 효과성을 높이는 데 기여함을 증명하는 ablation 연구도 수행되었습니다. 이를 통해 감정 이해 및 생성의 품질이 크게 향상됨을 확인할 수 있었습니다.



### SafeLLM: Extraction as a Hallucination-Resistant Alternative to Rewriting in Safety-Critical Settings (https://arxiv.org/abs/2606.12897)
- **What's New**: 이 논문은 대규모 언어 모델(LLMs)을 활용하여 조직 문서, 특히 안전 및 준수가 중요한 문서에 대한 질문 응답의 새로운 접근 방식을 제안합니다. 기존의 재작성 기반 방법이 가져오는 망상(hallucination)을 줄이기 위해, 저자들은 추출(extraction) 기법을 채택하여 정보의 정확성을 높이고, 문서의 신뢰도를 유지하는 방향으로 나아갑니다. 이 접근법은 특히 클리닉 질문 응답의 사용 사례에 초점을 맞춰, 안전-critical 환경에서의 실용성을 검증합니다.

- **Technical Details**: 이 연구에서는 다양한 문서 스타일과 규모를 아우르는 세 가지 데이터셋을 사용하는 방법론을 제시합니다. UCLH의 간결한 급성 치료 지침, Somerset NHS Trust의 중간 길이 기관 지침, 그리고 대규모 국가 권고(NICE)를 포함하여 다양한 길이와 구조를 가진 문서에서 실험이 진행됩니다. 이 과정에서 LLM 모델을 통해 문서에서 직접 정보를 선택하고 제시함으로써, 재작성 과정 중 발생할 수 있는 오류를 줄이고, 자료에 대한 충실도를 유지하려고 합니다.

- **Performance Highlights**: 연구 결과, 선형 번호 기반 선택(line-number selection) 방법이 가장 우수한 성능을 보이며, 대규모 및 소규모 모델 모두에서 직접 복사 및 안전 중심 전략을 능가했습니다. 이 방법은 95%에 달하는 높은 용어 회수(term recall)와 원문과의 근접도를 유지하였습니다. 반면, 안전 중심의 접근법은 정확도를 개선했지만, 비판적인 세부 사항의 누락을 초래하는 경향이 있었으며, 다단계 필터링 과정에서도 이러한 trade-off가 더욱 두드러졌습니다.



### Direct Preference Optimization for Chatbot Fine-Tuning: An Empirical Study (https://arxiv.org/abs/2606.12881)
Comments:
          7 pages, 3 figures, 1 table

- **What's New**: 이 논문에서는 Direct Preference Optimization (DPO)을 사용한 대형 언어 모델 미세 조정 접근법을 제시합니다. DPO는 강화 학습 기술이며, 훈련 파이프라인을 간소화하고 실행 효율성을 개선하는 데 기여했습니다. 실험 결과 DPO는 경쟁력 있는 성능을 기록하였으나, 훈련 불안정성 문제에 대해서는 추가적인 조사가 필요합니다.

- **Technical Details**: DPO는 전통적인 방법보다 더 직관적으로 인간의 선호도를 기반으로 정책을 직접 최적화하는 방법입니다. 이 방식은 보상 모델링 없이도 모델의 정책을 조정할 수 있으며, 로그 확률을 사용하여 원하는 응답의 상대적 가중치를 강화합니다. 이 모델은 Bradley-Terry 모델과 같은 이론적 선호 모델을 활용하여 훈련됩니다.

- **Performance Highlights**: BLEU, ROUGE, 코사인 유사도와 같은 다양한 메트릭을 활용하여 성능을 평가한 결과, DPO가 특정 NLP 작업에서 효과적으로 학습하고 수렴하는 성과를 나타냈습니다. 특히, DPO는 특정 작업에서 Proximal Policy Optimization (PPO)보다 우수한 성능을 발휘했으며, 훈련 데이터 세트는 아르길라의 편향 없는 데이터셋을 사용하여 질적으로 높은 결과를 보장하였습니다.



### Small LLMs for Biomedical Claim Verification: Cost-Effective Fine-Tuning, Structural Dataset Shortcuts, and Cross-Domain Generalization (https://arxiv.org/abs/2606.12854)
Comments:
          8 pages, 2 figures, 12 tables. To appear at BioNLP Workshop, ACL 2026

- **What's New**: 이 연구는 QLoRA를 사용하여 Phi-3-mini, Qwen2.5-3B, Mistral-7B의 세 가지 소형 LLM을 세밀하게 조정하고, 이를 통해 생물 의학 주장 검증에서 뛰어난 성능을 보여주며 GPT-4o 및 BioLinkBERT를 초과했음을 입증합니다. 특히, Mistral-7B QLoRA는 단 1,008개의 훈련 예제로 88.4%의 macro-F1을 달성하여 기존 모델보다 12% 향상된 성능을 보였습니다. 또한, 연구진은 SciFact 데이터셋에 숨겨진 구조적 문제를 발견하고, 이를 통해 모델의 성능을 더욱 향상시킬 수 있음을 보여주었습니다.

- **Technical Details**: 자동화된 생물 의학 주장 검증은 주장에 대한 증거의 지원, 반박 또는 미확인 여부를 판단하는 Natural Language Inference (NLI)의 형태로 점점 더 중요해지고 있습니다. QLoRA는 billion-parameter 모델을 단일 GPU에서 4-bit로 미세 조정할 수 있는 방법을 제공하며, 이 연구에서는 SciFact 및 HealthVer 데이터셋에 대해 QLoRA를 적용하여 성능을 평가했습니다. 실험은 각 모델을 SciFact 및 HealthVer에서 독립적으로 훈련시키고 교차 평가하여, 데이터 구조의 영향을 분리하였습니다.

- **Performance Highlights**: 세 가지 QLoRA 모델 모두 1,008개의 훈련 예제로만 GPT-4o의 macro-F1을 초과했습니다. Mistral-7B는 88.4%를 달성하여 GPT-4o보다 2.8 포인트 높았으며, BioLinkBERT는 0.9 포인트 초과했습니다. 동일한 데이터셋 구조에서 모델의 훈련 및 평가를 통해 강력한 교차 도메인 적응력을 입증했으며, 이는 특히 Mistral-7B가 HealthVer의 1,008개 예제로 훈련했을 때 SciFact OOD에서 74.3%의 NEI F1을 달성한 것에서 두드러집니다.



### LoHoSearch: Benchmarking Long-Horizon Search Agents Beyond the Human Difficulty Ceiling (https://arxiv.org/abs/2606.12837)
- **What's New**: 이 논문은 LoHoSearch라는 새로운 벤치마크를 제시하고 있습니다. 이 벤치마크는 11개 도메인에서 544개의 인간 검증 질문으로 구성되어 있으며, 지식 그래프를 기반으로 자동화된 파이프라인을 통해 작성되었습니다. 기존의 인간 작성 벤치마크의 한계를 극복하고, 검색 에이전트의 성능을 더욱 정교하게 평가하기 위한 새로운 기준을 제공합니다.

- **Technical Details**: LoHoSearch 벤치마크는 검색 공간 크기와 구조적 복잡성을 체계적으로 조절하는 자동화된 QA 생성 파이프라인을 통해 만들어졌습니다. 지식 그래프는 760만 개의 엔티티와 2억 6천5백만 개의 방향성 엣지로 구성되어 있습니다. 질문은 자연어로 변환되며, 언어 모델을 사용해 생성된 질문의 정확성과 고유성을 여러 차례 검증합니다.

- **Performance Highlights**: LoHoSearch의 평가 결과, 가장 강력한 모델의 정확도는 34.74%에 불과하며, 기존의 컨텍스트 관리 전략은 이전 벤치마크에 비해 6.8%의 작은 향상만을 제공합니다. 이는 LoHoSearch가 높은 난이도 시나리오에서 기존 전략의 한계를 드러내고 있음을 보여줍니다. 이 새로운 벤치마크는 검색 에이전트의 장기적인 추론( long-horizon reasoning)과 컨텍스트 관리 능력을 평가하는 데 더 많은 도전 과제를 제공합니다.



### Localizing Anchoring Pathways in Language Models (https://arxiv.org/abs/2606.12818)
- **What's New**: 이번 논문에서는 언어 모델(Language Model)이 부적절한 숫자가 예측을 변화시키는 앵커링 효과(anchoring effect)에 대한 연구를 다룹니다. 저자들은 기존의 블랙 박스 방식에서 벗어나, 숫자 앵커가 내부 계산에 어떻게 영향을 미치는지 분석하였습니다. 이를 통해 다양한 모델의 내부 경로가 앵커에 민감한 경쟁을 어떻게 지원하는지를 탐구했습니다.

- **Technical Details**: 연구에서는 7B-8B Qwen 모델과 Llama 모델을 대상으로 회로 로컬리제이션(circuit localization) 기법을 사용하여, 엣지 수준(Edge-level) 방법이 노드 수준(Node-level) 방법보다 더 정확하게 앵커 신호를 포착한다는 결과를 도출했습니다. 저자들은 모델에 따라 저앵커와 고앵커 회로가 밀접하게 연관되어 있으며, 훈련 후에는 어떤 경로가 제일 중요한지를 변화시키는 경향이 있음을 확인하였습니다.

- **Performance Highlights**: 저자들은 고정된 답안 옵션을 통한 통제된 다중 선택 과제를 설정하여, 앵커가 주어진 맥락 속에서 모델의 추론을 어떻게 변화시키는지를 평가했습니다. 이 연구 결과는 앵커링 효과가 모델의 내부 경로에 따라 다르게 작용하는 방식에 대한 기계적 해석을 제공하며, 다양한 모델 패밀리 간의 경로 전파 방법을 설명합니다.



### Detect, Remask, Repair: Diffusion Editing for Faithful Summarization of Evolving Contexts (https://arxiv.org/abs/2606.12807)
- **What's New**: 본 연구에서는 진화하는 맥락에서도 신뢰성을 유지하며 기존 요약을 업데이트하는 'Localized Faithfulness Repair' 접근을 접목한 'DETECT-REMASK-REPAIR' 프레임워크를 제안합니다. 이 방법은 과거의 요약에서 지원된 콘텐츠는 유지하면서도 변경이 필요한 부분만을 효과적으로 수정할 수 있습니다. 이를 통해 전체 재작성 없이도 요약의 품질을 높일 수 있는 가능성을 보여줍니다.

- **Technical Details**: DETECT-REMASK-REPAIR 프레임워크는 주어진 초안 요약에서 업데이트된 맥락에 따라 지원되지 않는 부분을 식별하고 수정하는 데 중점을 둡니다. 이 과정에서 'Masked Diffusion Language Models'를 활용하여 잘못된 요약 토큰을 선택적으로 재마스킹하고 그 주변의 문맥에 조건을 둔 새로운 토큰을 생성합니다. 'StreamSum' 데이터셋을 통해 실시간으로 진화하는 사건 요약의 평가를 수행하며, 이는 기존 요약의 갱신 기법의 기본 시설을 제공합니다.

- **Performance Highlights**: 실험 결과, Diffusion Repair 방식은 초기 요약의 신뢰성을 향상시키고, 기존 요약의 설정에서 필요한 수정 시간을 절반 이하로 줄였습니다. 또한, 이 프레임워크는 여러 데이터셋에서 신뢰성-속도-보존 간의 trade-off를 조절 가능하게 하여 요약의 품질을 높였습니다. 결과적으로, Detect-Remask-Repair 기법은 전체적으로 약속된 신뢰성을 유지하면서도 보다 자연스럽고 신속한 요약 수정을 가능하게 했습니다.



### GENIE: A Fine-Grained Measure for Novelty (https://arxiv.org/abs/2606.12790)
- **What's New**: 이 논문에서는 기존의 대규모 언어 모델(LLMs)이 생성하는 콘텐츠의 창의성과 다양성을 평가하는 새로운 측정 지표인 GENIE를 제안합니다. GENIE는 특정 작업에 따라 모델이 생성한 응답의 참신성을 정밀하게 평가할 수 있는 지표로, 기존의 전체론적(holistic) 지표보다 높은 차원의 참신성을 포착할 수 있습니다. 이를 통해 모델의 창의성을 개선하기 위한 다양한 방법의 효과를 측정하는 데 유용성을 입증합니다.

- **Technical Details**: GENIE(Granular Evaluation of Novel Ideas with Explainability)는 명확하게 정의된 집단(population)과 비교하여 다양한 작업 특성을 기준으로 참신성을 측정할 수 있도록 설계된 체계입니다. 이 체계는 질문과 답변을 통해 모델이 생성한 응답에서 작업 관련 특성을 추출하여 참신성을 분석합니다. 또한, GENIE는 응답의 질(value)과 같은 다른 창의성 구성 요소와 분리하여 참신성에 대한 독립적이고 정밀한 평가를 가능하게 합니다.

- **Performance Highlights**: GENIE는 다양한 대규모 언어 모델이 생성한 응답 쌍을 비교하여 참신성의 미세한 차이를 성공적으로 식별하며, 단순한 바꿔쓰기(paraphrasing)만으로는 큰 차이를 감지하지 않습니다. 고유한 작업 특성과 연관된 77개의 미세한 특성을 자동으로 발견하였고, 이를 통해 기존의 전반적인 평가 지표들이 참신성과 창의성을 측정하는 데 있어 갖는 한계를 분석했습니다. 이러한 특성 분석을 통해 GENIE는 창의성 개선 방법이 실제로 개선하는 특성을 밝혀내는 더 설명가능한 평가를 제공합니다.



### How Fine-Grained Should a RAG Benchmark Be? A Hierarchical Framework for Synthetic Question Generation (https://arxiv.org/abs/2606.12789)
- **What's New**: HieraRAG는 retrieval-augmented generation (RAG) 시스템의 벤치마크 구성 시 문항 특성을 세분화하는 최적의 수준을 정의하는 계층 구조의 프레임워크로 소개된다. 이 연구는 질문의 복잡성, 답변 유형 및 언어적 변variation을 세 가지 차원에서 변수로 삼아 5,872개의 합성 질문-답변(QA) 쌍을 생성하였다. 또한, Coherence Ratio라는 새로운 메트릭을 도입하여 질문 분류 구조의 질을 정량화하고 문항의 구조적 차이를 비교한다.

- **Technical Details**: HieraRAG의 구조는 질문 복잡성(Question Complexity, QC), 답변 유형(Answer Type, AT), 언어적 variation(Linguistic Variation, LV)이라는 세 가지 기본 차원에서 3단계의 세분화 수준(거칠게, 중간, 세밀하게)으로 평가된다. 질문 복잡성 차원은 복잡한 문제 해결을 요구하며, 답변 유형은 정보의 직접 추출과 새로운 형태의 생성 간의 구별을 나타낸다. 언어적 variation은 질문의 표현과 문서 내용 간의 어휘적 정렬을 정량화하여 문장의 직접적인 사용부터 의미 이해를 요하는 패러프레이즈된 개념까지 다양성을 포함한다.

- **Performance Highlights**: 연구 결과, 최적의 granularity는 차원에 따라 달라지며, 질문의 복잡성 차원에서 세밀한 구분(8개 카테고리)이 이득을 볼 수 있는 반면, 답변 유형과 언어적 variation은 중간 granularity(4개 카테고리)에서 최고 성능을 보인다. Coherence Ratio 메트릭에 따른 구조적 차이가 확인되었으며, 인간 평가에서도 합성 질문 품질이 98%로 높게 나타났다. 이는 RAG 시스템의 성능을 평가하기 위한 적절한 방식의 필요성을 강조하며, 다양한 질문 특성에 따라 다른 세분화 수준이 요구된다는 점을 보여준다.



### Rigel: Reverse-Engineering the Metal 4.1 Tensor Compute Path on the Apple M4 Max GPU (https://arxiv.org/abs/2606.12765)
- **What's New**: 이 논문은 Apple의 Metal 4.1에서 제공하는 텐서 계산 경로, 즉 Metal Performance Primitives(MPP) matmul2d 작업에 대한 경험적 분석인 Rigel을 소개합니다. 해당 경로는 여러 하드웨어 동작에 대한 명확한 문서화가 부족하며, fp8 지원 여부 및 성능에 대한 모호함이 존재합니다. Rigel은 이러한 모호한 세부 사항 11개를 밝혀내어, fp8(matmul2d)은 하드웨어 지원이 아닌 에뮬레이션된 기능임을 입증했습니다.

- **Technical Details**: Metal 4.1의 텐서 경로는 matmul2d 연산으로, 두 텐서 오퍼랜드를 곱하여 결과를 생성합니다. 해당 작업은 cooperative_tensor 조각을 통해 수행되며, 그 내부 레이아웃은 명확히 드러나지 않습니다. 이 논문에서는 M4 Max에서 matmul2d가 GPU 셰이더 코어에서 실행되며, 전용 매트릭스 데이터 경로 없이 legacy simdgroup_matrix 경로를 통해 처리된다는 것을 보여줍니다.

- **Performance Highlights**: 논문의 결과에 따르면 fp8는 fp16의 0.94배의 처리량을 달성하며, 이로 인해 메모리 발자국 특성으로 간주됩니다. 또한, GEMM+bias+GELU 커널을 직접 결합하여 사용했을 때, 분해된 경로보다 +6.5-12.9% 더 우수한 성능을 나타냅니다. 마지막으로, 모든 결과는 별도의 코드와 CSV 파일에서 재현 가능하다는 점을 강조합니다.



### LLMs Can Better Capture Human Judgments--With the Right Prompts (https://arxiv.org/abs/2606.12754)
- **What's New**: 본 연구는 대형 언어 모델(LLM)이 인간의 판단을 잘 포착하지 못하는 이유를 다루고 있습니다. LLM의 한계로는 응답의 완전한 분포를 포착하지 못하고, 단어 변형에 따라 판단이 불안정하다는 점이 지적됩니다. 그러나 간단한 프롬프트 전략(prompting strategies)을 통해 이러한 문제를 완화할 수 있음을 보여줍니다.

- **Technical Details**: 연구는 미국 대표 샘플인 144개의 도덕적 시나리오(moral scenarios)와 32개국을 포함한 국제 사회 조사 프로그램(International Social Survey Programme)의 가족 및 성 역할에 대한 38개 도덕적 신념(moral beliefs) 데이터셋을 통해 수행되었습니다. LLM에게 표준편차(standard deviations)와 응답 비율(response proportions)을 보고하도록 요청하면, 일반적인 전략보다 인간 응답의 전체 범위를 더 잘 회복합니다. 또한, 시나리오가 인간 참가자에게 명확하도록 하는 것이 모델 정렬(model alignment)을 향상시키는 데 도움이 됩니다.

- **Performance Highlights**: LLMs는 인간의 혼란도(confusion ratings)를 잘 추적할 수 있으며, 혼란도가 높은 상황에서 모델 정렬이 증진됩니다. 그러나 LLM은 자신의 오류를 잘못 보정하는 경향이 있으며, 인간의 변동성을 예측하는 능력은 상대적으로 우수합니다. 이는 LLM에 더 나은 질문을 제시할수록 더 나은 답변을 얻을 수 있음을 시사합니다.



### Agent-based models for the evolution of morphological alternation patterns (https://arxiv.org/abs/2606.12748)
Comments:
          51 + 37 pages. 31 Figures

- **What's New**: 이 논문은 영어 동사 'go'의 과거형인 'went'와 같은 형태 변화를 다루고 있습니다. 이러한 형태 변화는 언어에서 자주 발생하며, 의사소통이나 학습에 쉽게 도움이 되지 않지만 수세기 또는 수천 년 동안 지속될 수 있습니다. 연구팀은 형태적인 줄기(stem)와 굴절(inflection) 변화를 나타내는 다중 에이전트 시뮬레이션 모델을 제시하며, 이를 통해 새로운 형태의 출현 과정을 모의 실험하고 있습니다.

- **Technical Details**: 본 연구는 합리적인 음운학적 규칙(phonological rules) 및 수백 또는 수천 개의 항목이 포함된 어휘(lexicon)를 가진 대규모 에이전트(populations) 집단에서 발생하는 형태 변화의 목록을 지원합니다. 에이전트가 다른 에이전트가 사용하는 새로운 형태를 듣고 일정 확률로 그 형태를 채택하여, 원래 형태를 공유하는 다른 구문(slot)으로도 확산될 수 있도록 설계되었습니다. 논문에서는 AI Historical Linguist라는 혁신적인 시스템을 도입하여 언어 형태(morphology)의 현실성을 평가하는 방법도 제시하였습니다.

- **Performance Highlights**: 체계적인 네트워크( network topologies)와 전파(diffusion) 패턴, 에이전트 수용 정책(agent adoption policies)을 실험하며, 현실 언어의 형태와 실험적 형태를 비교 분석한 결과를 발표합니다. 특히, scale-free social networks와 random Bernoulli 형태 수용이 더 그럴듯한 형태를 선호한다는 것을 제시하며, 역사적 변화에 대한 세 가지 사례 연구를 통해 역사적 변화의 대안적 경로도 시뮬레이션합니다.



### Does AI Reviewer See the Full Picture? Attacking and Defending Multimodal Peer Review (https://arxiv.org/abs/2606.12716)
Comments:
          Accepted to ICML 2026, Project Page: this https URL

- **What's New**: 이 논문에서는 대규모 언어 모델(Large Language Models, LLMs)과 멀티모달 LLMs(Multimodal LLMs, MLLMs)를 과학적 동료 검토(technical peer-review) 프로세스에 통합함으로써 신뢰성 있는 평가에 대한 새로운 위험을 제기합니다. 특히, 과학 논문에는 텍스트만이 아니라 그림도 중요한 증거를 전달하기 때문에, 현재 AI 피어 리뷰의 견고성에 대한 연구는 대부분 텍스트 기반으로만 제한되어 있다는 점에서 문제를 제기합니다. 이에 대한 해결책으로, 우리는 PaperGuard라는 첫 번째 종합 벤치마크를 소개하며, 이는 이러한 도메인 특정의 적대적 공격에 대해 AI 생성 동료 리뷰를 평가하고 방어하기 위해 설계되었습니다.

- **Technical Details**: PaperGuard는 증가하는 멀티모달 피어 리뷰 데이터셋을 포함하여, 다양한 연구 분야를 아우르는 데이터 구성을 제공합니다. 또한 블랙박스 프롬프트 주입 및 화이트박스 섭동 공격과 같은 통합 공격 모음을 제공하여 텍스트(GCG)와 그림(PGD) 모두를 겨냥한 공격을 체계적으로 이끌어냅니다. 마지막으로, 파편 기반(chunk-based) 임베딩 검색을 이용한 실용 방어를 포함하여, 학술 논문의 긴 문맥 문제를 해결하고 해로운 명령을 효율적으로 국지화하는 방안을 모색합니다.

- **Performance Highlights**: 딥러닝 모델을 대상으로 한 실험들을 통해 AI 리뷰어들이 광범위하게 취약하다는 것을 밝혔습니다. PaperGuard는 신뢰할 수 있는, 공격에 저항하는 AI 지원 학술 검토의 기초적인 벤치마크와 프로토콜을 설정하며, 내구성과 효율성을 증진하기 위한 실제적인 방어 기법을 제공합니다. 이를 통해 학계에서의 AI의 공정성과 안전성을 확보할 수 있는 중요한 기초 자료를 제공합니다.



### AfriSUD: A Dependency Treebank Collection for Evaluating Models on African Languages (https://arxiv.org/abs/2606.12708)
- **What's New**: 본 논문에서는 아프리카 언어에 대한 NLP 지원을 위한 첫 번째 대규모 구문 주석 트리뱅크 집합인 AfriSUD를 소개합니다. 이 데이터셋은 아프리카 전역에서 언어적 다양성을 반영하는 9개의 언어에 대해 정확성이 검증된 데이터를 제공합니다. 특히, 표면 구문 유니버설 의존성(Surface-Syntactic Universal Dependencies, SUD) 프레임워크를 기반으로 하여 고유한 언어적 특징들을 포착하고 있습니다.

- **Technical Details**: AfriSUD 데이터셋은 Niger-Congo, Afroasiatic 및 영어 기반 크리올을 포함한 아프리카의 다양한 언어들을 아우릅니다. 각각의 언어는 서로 다른 형태통사적(morphosyntactic) 특성을 가지며, 예를 들어 언어 중 일부는 고립어로 주로 어순과 입자를 사용하고, 다른 언어는 응집어(agglutinative)로 복잡한 동사 형태를 지니고 있습니다. 이로 인해 구문 정보의 분포와 의존 구문 분석에서 도전 과제가 발생합니다.

- **Performance Highlights**: 모델 평가 결과, AfriSUD 데이터셋에서 구문 분석과 품사 태깅에 있어 명확한 구조적 격차가 드러났습니다. 고성능 모델임에도 불구하고, 아프리카 언어 고유의 관계를 충분히 포착하지 못하고 있으며, 특정 언어적 관계에 대해 더 나은 성능을 보여주지 못합니다. 모든 트리뱅크와 주석 지침은 출판 후 공개될 예정입니다.



### Observable Patterns Are Not Explanations: A Causal-Geometric Analysis of Latent Reasoning Models (https://arxiv.org/abs/2606.12689)
- **What's New**: 이번 연구에서 저자들은 Latent Reasoning Models (LRMs)가 전통적인 Chain-of-Thought (CoT) 방식 대신 연속적인 사고 프로세스를 사용한다고 주장합니다. 이러한 모델들이 내부 추론 메커니즘을 나타내기 위해 관찰 가능한 라텐트 상태 패턴을 어떻게 활용하는지를 탐구하며, 두 가지 LRM (Coconut과 CODI)을 비교 분석합니다. 저자들은 이러한 패턴들이 반드시 행동에 영향을 미치지는 않으며, 정확한 해석을 위해서는 적절한 대조군과 인과적 검사가 필요하다고 강조합니다.

- **Technical Details**: LRMs는 연속적인 숨겨진 상태에서 중간 계산을 수행하여 자연어 처리의 불리한 점을 극복하려고 합니다. 연구에서는 LRM의 라텐트 사고(ablation), 인과 추적(causal tracing), 및 기울기 부분 공간(intervention)을 결합하여 라텐트 사고의 영향을 측정하고, 이는 종종 낮은 차원에서 집중된다고 밝혔습니다. 저자들은 행동에 영향을 미치는 사고들이 진화하는 방식의 구조적 차이를 조사하여, 이들이 기본적으로 데이터에서 중요한 기하학적 방향을 가진다는 것을 시사합니다.

- **Performance Highlights**: 저자들은 LRM이 이전에 주장된 주기성중단(latent recurrence) 패턴이 Curriculum-matched 모델에서도 나타나며, 이로 인해 관찰 가능한 패턴만으로는 메커니즘을 충분히 확립할 수 없음을 발견했습니다. 또한, 행동에 영향을 미치는 라텐트 사고는 구조화된 진화를 보여주며, 이는 약하게 영향력 있는 사고와 정확히 구분된다고 보고했습니다. 이러한 발견들은 LRM의 해석 가능성에 대한 새로운 관점을 제시하여, 관찰적 지표가 아닌 실제 인과적 경로에 대한 이해를 요구합니다.



### MentalMARBERT: Domain-Adaptive Pre-training and Two-Stage Fine-Tuning for Arabic Mental Health Disorders Detection (https://arxiv.org/abs/2606.12649)
Comments:
          17 pages, 5 figures, 13 tables

- **What's New**: 이 연구는 아랍어 소셜 미디어 텍스트에서 정신 건강 장애를 분류하기 위한 새로운 두 단계 프레임워크를 제안합니다. 첫 번째 단계에서는 AraBERT, CAMeLBERT, MARBERT 등의 사전 학습된 모델을 도메인 및 작업 적응 사전 학습(DAPT 및 TAPT)을 통해 최적의 모델을 찾습니다. 두 번째 단계에서는 선정된 모델을 다양한 분류 아키텍처와 조합하여 평가하여 정교한 결과를 도출합니다.

- **Technical Details**: 정신 건강 텍스트 분류를 위해 50,670개의 아랍어 트윗으로 구성된 새로운 주석 데이터셋이 구축되었습니다. 이 데이터셋은 우수한 주석자 간 동의를 보여주며(Krippendorff's Alpha = 0.733, 평균 쌍 간 합의 = 0.797) 다중 클래스 정신 장애를 대상으로 하고 있습니다. 모델의 성능을 평가하기 위해 단일 단계 및 계층적 두 단계 분류 아키텍처가 활용되며, 전체 미세 조정 및 저랭크 적응(LoRA) 전략이 사용됩니다.

- **Performance Highlights**: 실험 결과, 도메인 적응된 MARBERT(정신 건강에 맞춘 MARBERT)가 기준 모델에 비해 통계적으로 유의미한 향상을 보여주었습니다. 계층적 두 단계 아키텍처와 전체 미세 조정 조합이 최상의 성능을 기록하여 매크로 F1 점수 0.861 및 정확도 0.877에 도달하였습니다. 이 findings는 아랍어 정신 건강 장애 감지를 위한 도메인 특화 적응 사전 학습과 계층적 분류의 효과성을 입증합니다.



### Shopping Reasoning Bench: An Expert-Authored Benchmark for Multi-Turn Conversational Shopping Assistants (https://arxiv.org/abs/2606.12608)
- **What's New**: Shopping Reasoning Bench는 기존의 쇼핑 대화 평가 기준의 한계를 극복하기 위해 새롭게 개발된 벤치마크입니다. 525개의 미션은 232개의 단일 회전 그리고 293개의 다중 회전으로 구성되어 있으며, 각 미션은 소매 전문가에 의해 작성된 10,863개의 중요 가중 이진 기준에 기반하여 평가됩니다. 이 벤치마크는 다섯 가지 추론 범주와 열다섯 가지 하위 범주로 조직되어 있어, 고객의 요구와 도메인 전문지식을 결합한 새로운 기준을 제시합니다.

- **Technical Details**: Shopping Reasoning Bench는 가격, 제품간 트레이드오프 및 사용자 의도에 따른 다중 회전 대화의 복잡성을 반영합니다. 이 벤치마크는 항상 정답이 있는 기존의 추론 평가와 다르게, 다양한 조건을 고려한 전문가 수준의 평가 기준을 제공합니다. 또한, 9개의 모델을 세 가지 계층으로 평가하며, 다중 회전 미션에서 진행이 길어질수록 성능이 저하되는 현상을 관찰했습니다.

- **Performance Highlights**: 모델들의 패스율은 57%에서 77%까지 다양하며, 모든 모델은 필수 기준에 비해 선택적 기준에서 평균 13~29점 낮았습니다. 다중 회전 미션에서의 성능도 미션이 진행됨에 따라 평균 4~18점 저하되는 것으로 나타났습니다. 이는 기존 모델들이 기본적인 쇼핑 보조는 가능하지만 전문가 수준의 상담에는 미치지 못함을 시사합니다.



### Constrained Semantic Decompression in LLMs through Persian Proverb-Conditioned Story Generation (https://arxiv.org/abs/2606.12599)
- **What's New**: 이번 연구에서는 조밀한 추상 속담을 매력적이고 도덕적으로 충실한 서사로 변환하는 방법을 제시합니다. 이러한 문제를 제약된 의미 압축 해제(constrained semantic decompression) 작업으로 구성하여, 대형 언어 모델(LLMs)을 활용하여 속담 기반의 이야기 생성에 대한 연구를 진행하였습니다. 페르시아어에 중점을 두어 인간이 작성한 이야기와 명시적 의미가 쌍을 이루는 Proverb Aligned Narrative Dataset (PAND)를 소개합니다.

- **Technical Details**: 우리는 인공지능 모델의 평가를 위해 인간 조정된 LLM을 판별자로 활용하는 혼합 평가 프레임워크와 구조적 메트릭을 결합하여 분석하였습니다. 분석 결과 현재의 LLM들은 표면적으로 유창한 결과를 도출하지만, 속담에 인코딩된 도덕적 및 인과 구조를 충실히 재현하는 데는 실패하는 경향이 있음을 보여줍니다.

- **Performance Highlights**: 연구 결과, 명시적 추론(explicit reasoning)과 반복적 세련화(iterative refinement)가 이러한 실패를 부분적으로 완화할 수 있음을 확인하였습니다. 이는 압축된 의미를 서사 형태로 변환하는 데 어려움이 있어 이러한 오차가 발생한다는 것을 시사합니다. 제안된 작업은 다른 형태의 압축된 문화 지식으로 자연스럽게 확장될 수 있습니다.



### MARD: Mirror-Augmented Reasoning Distillation for Mechanism-Level Drug-Drug Interaction Prediction (https://arxiv.org/abs/2606.12578)
Comments:
          29 pages, 9 figures. Preprint

- **What's New**: 이 논문은 약물-약물 상호작용(DDI)의 메커니즘 수준 예측을 위해 새로운 라벨링 및 평가 프로토콜을 도입합니다. 7개 가족과 147개 하위 유형으로 구성된 체계적인 분류 시스템을 제공하며, 약물 상호작용의 방향성과 증거를 평가할 수 있는 방법을 제안합니다. 특히, DDI 예측에서 편향이 없는 구조적 이유를 제공하며, 약물 쌍의 참조 여부와 관계없이 정확성을 유지합니다.

- **Technical Details**: 제안된 MARD(미러 증강 추론 증류)는 세 가지 훈련 혁신을 합친 77B 모델입니다. 이 시스템은 메커니즘 지향의 검색 채널과 관찰 가능한 증거 풀을 기반으로 한 검증 가능한 데이터 추적을 제공합니다. 또한, 임의 분할, 약물-콜드, 쌍-콜드와 같은 세 가지 안전한 분할 프로토콜을 적용하여 평가합니다.

- **Performance Highlights**: MARD-7B는 32개 시스템 비교에서 유일하게 약물 쌍의 새로운 조합에서도 높은 정확성을 유지하며 베스트 베이스라인보다 13.9 pp 높은 성과를 기록했습니다. 또한, 잘 보지 못한 약물에 대한 정확성이 증가해 약물 빈도 메모리화 대신 구조적 약리학적 추론의 우수성을 나타냅니다. 이 연구는 MARD의 성능을 보여주는 데이터셋과 훈련 코드를 함께 공개합니다.



### Helping Figures Tell their Story! Paper-Grounded Video Generation Explaining Complex Scientific Figures (https://arxiv.org/abs/2606.12576)
Comments:
          Webpage: this https URL

- **What's New**: 이 논문에서는 복잡한 과학적 그림을 하나의 캔버스로 압축할 수 있는 도구를 제공하지만, 이러한 그림을 이해하기 위해서는 단계별 내러티브(narration)가 필요하다고 강조합니다. 이를 해결하기 위해, 논문에 기반한 다양한 그림에서 비디오를 생성하는 새로운 방법론인 MINARD(Multimodal Interpretation of Narrated Architecture via Region Decomposition)를 제안합니다. MINARD는 그림과 논문에 기반하여 내러티브를 생성하며, 이러한 내러티브를 그림의 특정 영역에 순차적으로 연결합니다.

- **Technical Details**: MINARD의 파이프라인(pipeline)은 그림의 특정 영역에 시각적인 정보를 대응시키는 메커니즘을 통해 내러티브를 생성하는 데 초점을 맞춥니다. 추가적으로, FigTalk라는 새로운 벤치마크(benchmark)가 도입되어, 시퀀스 및 성분 수준의 지표를 제공함으로써 MInARD의 성능을 정량화합니다. 이 시스템은 그림의 각 영역이 어떻게 설명되는지를 정교하게 처리하여 비디오 생성 과정에서의 정보를 풍부하게 만듭니다.

- **Performance Highlights**: 실험 결과, MINARD는 인간과 같은 품질의 내러티브를 생성하며, 기존의 접근 방식보다 내러티브 기반의 그림 공간 정렬(narration-conditioned figure spatial grounding)에서 향상된 성능을 보여줍니다. 자동화된 평가와 인간 평가 모두에서 MINARD는 높은 점수를 기록하며, 그 신뢰성과 유용성을 입증하였습니다. 이는 과학적 그림을 이해하고 활용하는 데 있어서 새로운 가능성을 제시합니다.



### EDEN: A Large-Scale Corpus of Clinical Notes for Italian (https://arxiv.org/abs/2606.12569)
- **What's New**: 이번 연구에서는 이탈리아 병원의 응급실에서 생성된 임상 노트의 대규모 데이터베이스인 EDEN(Emergency Department Electronic Notes)에 대해 설명하고 있습니다. 현재 버전은 약 4백만 개의 완전 익명화된 임상 노트로 구성되어 있으며, 응급실에서의 환자 치료 과정의 다양한 단계를 포괄합니다. 특히, 약 6천 개의 노트는 임상 전문가에 의해 사례 보고 양식을 통해 수작업으로 주석이 달려 있으며, 이는 응급 상황에서의 환자 상태를 기록합니다.

- **Technical Details**: EDEN 데이터는 구조화된 사례 보고 양식(Case Report Form; CRF)을 사용하여 환자 상황인 호흡 곤란(dyspnea)과 의식 상실(loss of consciousness)에 대해 생리적 및 임상적 정보를 수집하였습니다. 수집 과정에서는 병원 윤리 위원회의 승인을 받았으며, 2021년부터 2023년 사이의 환자의 자유 텍스트 임상 노트를 포함하고 있습니다. 각 병원의 전자 건강 기록(EHR) 시스템에서 추출된 데이터는 두 단계의 익명화 과정을 통해 개인 정보 보호를 철저히 하고 있습니다.

- **Performance Highlights**: EDEN 데이터셋은 이탈리아어로 된 임상 노트의 가장 큰 자유롭게 사용 가능한 데이터베이스로, 연구 개발 및 대규모 언어 모델(Large Language Models; LLM)의 실용적 응용을 지원하기 위한 자료로서 중요한 역할을 할 것으로 기대됩니다. 특히, CRF 채우기를 새로운 구조적 정보 추출 벤치마크로 제안하며, Gemma-27B와 MedGemma-27B에서 도출된 제로샷(zero-shot) 기준 결과를 제공하고 있습니다. 이 데이터셋은 다국어 및 다기관 연구 개발에 매우 적합한 리소스로, 향후 더욱 확장될 예정입니다.



### EurekAgent: Agent Environment Engineering is All You Need For Autonomous Scientific Discovery (https://arxiv.org/abs/2606.13662)
- **What's New**: 이 논문은 LLM(대형 언어 모델) 기반의 에이전트들이 과학적 발견의 자동화를 가능하게 하는 잠재력을 보여주고 있으며, 이러한 에이전트들이 인간이 설계한 접근 방식을 초월한 성과를 내었다고 주장하고 있습니다. 저자들은 자율적 과학적 발견의 병목 현상이 에이전트의 작업 흐름을 규정하는 것에서 에이전트 환경 설계로 이동하고 있다고 강조합니다. 새로운 시스템인 EurekAgent는 이러한 환경 엔지니어링을 통해 더 효과적인 연구를 가능케 하고 있습니다.

- **Technical Details**: EurekAgent는 네 가지 차원(permissions engineering, artifact engineering, budget engineering, human-in-the-loop engineering)에서 환경을 설계합니다. 이는 에이전트의 실행을 제어하고, 협업을 유도하며, 예산을 고려한 탐색을 가능하게 하여 자율 과학적 발견을 지원합니다. 이러한 설계는 에이전트가 연구 작업 흐름을 자유롭게 선택할 수 있도록 하므로, 더 생산적인 행동을 촉진합니다.

- **Performance Highlights**: EurekAgent는 수학, 커널 엔지니어링 및 머신러닝 과제에서 새로운 최첨단 결과를 세웠으며, 특히 26-서클 포장 작업에서는 API 비용이 11달러 이하로 나오면서 새로운 기록을 달성하였습니다. 이 시스템은 효율적이고 책임 있는 자율 연구 에이전트를 구축하기 위한 환경 엔지니어링의 중요성을 부각시키며, 코드와 결과를 오픈 소스로 제공하고 있습니다.



### Beyond the Commitment Boundary: Probing Epiphenomenal Chain-of-Thought in Large Reasoning Models (https://arxiv.org/abs/2606.13603)
- **What's New**: 본 논문은 Chain-of-Thought (CoT) 추론의 각 단계가 최종 답변에 미치는 인과적 영향(causal influence)을 평가합니다. 저자들은 조기 종료(early exit)를 통해 각 단계의 인과적 중요성(causal importance)을 추정하며, 답변이 여러 모델 계열의 추론(trace)에서 어떻게 형성되는지를 연구합니다. 이 연구는 전통적인 CoT 접근 방식에서의 깊이 있는 통찰력을 제공합니다.

- **Technical Details**: 연구에서는 다양한 작업(Task)에서 추론이 일반적으로 'commitment boundary'를 초과한다고 발견했습니다. 이 경계는 일시적인 중간 추측(intermediate guesses)에서 안정적이고 높은 신뢰도의 답변으로의 급격한 전환을 나타냅니다. 저자들은 주목 프롭(attention probes)을 활용하여 중간 추론 단계로부터 답변 형성(answer-formation) 단계를 선형으로 디코딩하는 데 성공했으며, 이는 기존의 추론 작업에서도 뛰어난 일반화 성능을 보였습니다.

- **Performance Highlights**: 조기 종료 메커니즘을 사용하여 모델의 추론 블록을 commitment boundary에서 조기 종료함으로써 평균 55%까지 CoT의 길이를 줄일 수 있었습니다. 이러한 조정이 모델 성능에 미치는 영향은 미미하였습니다. 이는 CoT 접근법의 효율성을 대폭 향상시키는 결과로 이어집니다.



### Reward Modeling for Multi-Agent Orchestration (https://arxiv.org/abs/2606.13598)
Comments:
          Preprint; work in progress

- **What's New**: 이 논문에서는 Large Language Models(LLMs) 기반의 Multi-Agent Systems(MAS)에서의 효과적인 오케스트레이션을 위한 새로운 방법인 Orchestration Reward Modeling(OrchRM)을 제안합니다. OrchRM은 사람의 주석 없이도 오케스트레이션 품질을 평가할 수 있는 자기지도(self-supervised) 프레임워크로, 다수의 에이전트 실행에서 중간 산출물을 활용하여 보상 모델을 훈련합니다.

- **Technical Details**: OrchRM은 Bradley-Terry 보상 모델 훈련을 위한 승-패 쌍을 구성하는 방식으로 작동하며, 기존의 비용이 많이 드는 서브 에이전트 롤아웃(sub-agent rollouts) 방식 대신 오케스트레이션 수준에서 직접 이루어집니다. 이를 통해 오케스트레이터 훈련 및 MAS 테스트 시간 스케일링에서 효율적이고 성능 높은 보상 지향 훈련을 가능하게 합니다.

- **Performance Highlights**: OrchRM은 토큰 사용에서 훈련 효율성을 최대 10배 증가시키고, MAS 테스트 시간 스케일링 성능에서 정확도를 최대 8% 향상시키며, 이러한 성과는 수학적 추론, 웹 기반 질문 응답, 다단계 추론 등 여러 도메인에 일관되게 전이됩니다. 이 연구는 다중 에이전트 오케스트레이션의 강력한 방향으로 오케스트레이션 수준의 보상 모델링을 제시합니다.



### The Tone of Awareness: Topic, Sentiment, and Toxicity Maps During Mental Health Month on TikTok (https://arxiv.org/abs/2606.13581)
Comments:
          12 pages, 6 figures

- **What's New**: 본 논문은 TikTok에서의 정신 건강 관련 콘텐츠의 톤을 분석하여, 사용자가 기여한 콘텐츠와 해당 콘텐츠에 대한 반응 간의 차이를 연구합니다. 2023년과 2024년의 정신 건강 인식의 달(Mental Health Awareness Month) 동안 수집된 28,341개의 TikTok 비디오와 80,130개의 댓글 데이터를 통해 주제별로 어떻게 인식되는지를 살펴봅니다. 저자들은 '톤'을 감정적이고 대인적인 틀로 정의하며, 정서(Sentiment)와 독성(Toxicity)을 측정하는 방법론을 사용합니다.

- **Technical Details**: 저자들은 TikTok Research API를 통해 수집된 비디오 텍스트에서 주제를 추출하기 위해 BERTopic을 사용하고, 주제별 정서 측정(XLM-T)과 독성 측정(Detoxify)을 수행합니다. 정서는 콘텐츠의 감정적 경향을, 독성은 유해하거나 공격적인 언어의 존재를 반영합니다. 각 주제에 대해 비디오 내용과 댓글에서 정서 및 독성을 분리하여 분석합니다.

- **Performance Highlights**: 분석 결과, 정신 건강 주제는 지속적으로 발생하는 테마가 있으며, 주요 주제는 임상 조건, 감정적 공개, 자기 관리 및 캠페인 지향 콘텐츠로 나뉩니다. 비디오에서 감정은 종종 부정적이며, 댓글은 더 혼합되거나 긍정적인 경향을 보입니다. 전반적으로 독성 수준은 낮지만, 특정 주제(예: 'Duet', 'Suicide Prevention')에서 더 길게 분포된 이상치를 보입니다. 이 연구는 TikTok에서의 정신 건강 담론에 관한 중요한 통찰을 제공합니다.



### Edit the Bits, Diff the Codes: Bitwise Residual Editing for Visual Autoregressive Models (https://arxiv.org/abs/2606.13558)
- **What's New**: 본 논문에서는 bitwise-residual VAR(generator) 모델에서의 이미지 편집을 위한 새로운 방법인 BitResEdit를 제안합니다. 이 접근법은 훈련 없이 가능하며, 기존의 VAR 편집기가 사용하는 두 가지 구조를 최대한 활용합니다: per-bit Bernoulli 예측 헤드와 다중 스케일 잔여 코드 필드입니다. 이를 통해 이미지의 특정 부분을 편집하면서도 관련 없는 콘텐츠는 보존할 수 있는 편집 기능을 제공합니다.

- **Technical Details**: BitResEdit는 두 단계로 작동합니다: 첫 번째 단계인 BitEdit는 어떤 비트가 샘플링되는지를 가이드하며, 두 번째 단계인 ResEdit는 샘플링된 변화를 이미지 코드의 어느 위치에 기록할지를 제어합니다. Infinity라는 VAR 생성기를 기반으로 설정되어, 연속적인 VAE 코드를 생성하고 이를 사용하여 이미지가 재구성됩니다. 이 모델의 독특한 수학적 성질을 이용하여 편집을 위한 자정 기법을 구현합니다.

- **Performance Highlights**: 실험 결과, BitResEdit는 같은 백본을 가진 VAR 편집기 중에서 가장 우수한 텍스트 정렬 성능을 달성하며, 기존 최강 편집기보다 CLIP 점수를 +1.07 향상시킵니다. 또한, 배경 보존 성능도 경쟁력이 있습니다. ablation 연구에서 BitEdit와 ResEdit이 타겟 정렬 및 배경 보존에서 서로 보완적인 역할을 한다는 것이 확인되었습니다.



### Uncertainty-Aware Hybrid Retrieval for Long-Document RAG (https://arxiv.org/abs/2606.13550)
- **What's New**: 본 연구는 Uncertainty-aware Multi-Granularity RAG (UMG-RAG)라는 혼합 검색 프레임워크를 제안합니다. 이 프레임워크는 훈련이 필요 없는 구조로, 기존의 밀집(dense)과 희박(sparse) 검색기를 전문가로 활용하여 여러 개의 청크(chunk) 크기를 다룹니다. 이를 통해 각 쿼리에 대해 밝혀질 수 있는 신뢰도를 추정하여 검색의 효율성을 높입니다.

- **Technical Details**: UMG-RAG는 검색 단계에서 단순 허블(hybrid) 모델을 사용하지 않고, 각 검색 전문가와 청크 크기 쌍의 신뢰도를 추정합니다. 신뢰도는 후보 점수 분포의 샤프니스(sharpness)를 통해 측정되며, 높은 신뢰도의 분포는 특정 쿼리에 대한 명확한 검색 선호도를 나타냅니다. 이를 통해 두 가지의 검색 방식(dense와 sparse)이 통합되어 더욱 효과적으로 증거(evidence)가 정리됩니다.

- **Performance Highlights**: 실험 결과, UMG-RAG와 그 변형인 UMGP-RAG는 질의 응답 벤치마크에서 생성 품질을 개선하며, 경량화된 검색 파이프라인을 유지하면서도 성능 향상을 보여주었습니다. 전체적으로 이 연구는 장기 문서 검색에서의 청크 크기와 신뢰도 추정의 중요성을 강조하고 있으며, 기존 모델들과의 비교에서도 유의미한 성과를 나타냈습니다.



### Adaptive Turn-Taking for Real-time Multi-Party Voice Agents (https://arxiv.org/abs/2606.13544)
Comments:
          Accepted for publication at Interspeech 2026

- **What's New**: 이 연구에서는 다자간 대화에서의 턴 테이킹(turn-taking) 행동을 역할에 기반하여 조정하는 음성 기반 에이전트 ModeratorLM을 제안합니다. 이 시스템은 대규모 음성 언어 모델을 기반으로 하여 청크(chunk) 단위로 스트리밍됩니다. 또한 대화의 맥락과 할당된 역할에 대한 사고 과정(chain-of-thought reasoning)을 통합한 변형인 ModeratorLM-Think도 도입되었습니다. 이를 통해 비할당 역할 모델에 비해 턴 테이킹의 정밀도는 40% 이상, 재현율은 70% 이상 향상되었습니다.

- **Technical Details**: ModeratorLM은 음성 인코더와 기본 언어 모델(LLM)로 구성되어 있습니다. 음성 인코더는 각 오디오 청크를 독립적으로 처리하여 청크 수준 임베딩을 생성합니다. 이 임베딩은 가벼운 선형 프로젝션 레이어를 통해 LLM 임베딩 공간으로 투영되며, 스트리밍 방식으로 LLM 컨텍스트에 순차적으로 추가됩니다. 각 입력 오디오 청크에 대해 모델은 턴 테이킹 및 응답 생성 중 하나를 선택하여 결과를 생성합니다.

- **Performance Highlights**: 실험 결과, ModeratorLM은 비역할 조건의 기준선보다 할당된 역할과의 정렬이 현저히 향상되었습니다. 실험에 사용된 RolePlayConv 데이터셋과 현실 세계의 회의 데이터에서 성능 개선이 확인되었습니다. 특히, 턴 테이킹의 정밀도와 재현율이 대폭 향상되었으며, 잘못된 긍정적(interruption) 반응을 줄였습니다.



### SupraBench: A Benchmark for Supramolecular Chemistry (https://arxiv.org/abs/2606.13477)
- **What's New**: 이 논문은 supramolecular chemistry (초분자 화학)에서 호스트-게스트 시스템 설계를 위해 LLMs (대형 언어 모델)의 성능을 평가하기 위한 최초의 벤치마크인 SupraBench를 소개합니다. 또한, SupraPmc라는 1,600만 토큰의 초분자 화학 기사를 제공하여 연구자들이 이 분야에 적응할 수 있도록 지원합니다. 이 연구는 기존의 수십 년간의 전문가 반복 과정에서 발생하는 시간 소모를 줄이기 위해 LLM을 활용하는 새로운 방법을 모색합니다.

- **Technical Details**: SupraBench는 바인딩 친화도 예측, 상위 바인더 선택, 용매 식별, 호스트-게스트 설명의 네 가지 기본 작업과 분자 식별을 위한 보조 비전 작업으로 구성됩니다. 연구자들은 기존의 LLM과 도메인 적응 사전 훈련된 버전을 포함하여 다양한 LLM을 평가하였으며, 각 작업에서 모델 간의 성능 차이를 관찰했습니다. 평가 결과는 LLM들이 현재의 초분자 화학 추론에서 상당한 개선 여지가 있음을 보여줍니다.

- **Performance Highlights**: 모델들의 성능 평가 결과, LLM들은 모든 작업에서 상당한 격차를 보이며, 도메인 적응 전훈련이 적당한 개선 효과를 나타냈지만 특정 포맷 출력에서는 트레이드오프가 발생했습니다. 각 작업군 별로 난이도가 뚜렷하게 표현되며, 이를 통해 현재 초분자 화학 추론에서의 고유한 실패 모드를 파악할 수 있었습니다. 연구자들은 SupraBench와 관련 텍스트 코퍼스가 이 분야 연구에 크게 기여할 것이라고 믿고 있습니다.



### MaxProof: Scaling Mathematical Proof with Generative-Verifier RL and Population-Level Test-Time Scaling (https://arxiv.org/abs/2606.13473)
- **What's New**: 본 논문은 수학적 증명에 대한 경쟁 수준의 테스트를 위한 테스트 시간 확장 프레임워크인 MaxProof를 소개합니다. M3 시리즈는 증명 생성, 증명 검증, 비판 조건 증명 수정의 세 가지 증명 지향적 기능을 훈련시켜, 이를 하나의 모델로 통합합니다. MaxProof는 모델을 생성기, 검증자, 정제자, 랭커로 활용하여 후보 증명의 집합에서 최종 증명을 선택하는 방식을 채택하며, IMO 2025에서 35/42, USAMO 2026에서 36/42의 성과를 달성했습니다.

- **Technical Details**: M3 모델의 설계 과정에서는 세 가지 주요 기능, 즉 증명 생성, 증명 검증, 증명 정제를 다루었습니다. 각 기능은 전용 훈련 단계를 통해 개발되었으며, Proof Expert는 생성 검증기에서의 보상을 토대로 훈련됩니다. 이 과정에서 기존 권장사항과 비교하여 오류를 식별하고 텍스트 평가 및 점수를 제공하여 전체 증명의 보상으로 사용합니다.

- **Performance Highlights**: M3 모델은 IMOProofBench 및 IMOAnswerBench에서 성능 격차를 줄이며, MaxProof를 활용하여 수학적 증명을 위한 경쟁 레벨에서 특히 뛰어난 결과를 보여주었습니다. 최종적으로, M3는 IMO 2025에서 35/42, USAMO 2026에서 36/42의 성과로 인간 금메달 기준을 초과하며, 이를 통해 수학적 증명의 대중적 이해와 응용 가능성을 더욱 확장할 수 있음을 시사합니다.



### Examining the Cognitive Gap Between Authors and Peer Reviewers on Academic Paper Novelty (https://arxiv.org/abs/2606.13452)
- **What's New**: 이번 연구에서는 2016년부터 2021년까지 Nature Communications에 게재된 15,328편의 학술 논문과 그에 대한 피어리뷰(peer-review) 코멘트를 분석하였습니다. 저자들이 논문의 제목, 초록, 서론에서 새로움을 강조하는 방법에 대해 조사하였고, 리뷰어들이 논문의 독창성을 평가하는 과정에서 저자들의 자기 홍보와 평가 간의 인지적 간극이 존재함을 밝혀냈습니다.

- **Technical Details**: 연구에서는 결과 지향(result-oriented) 혁신이 저자와 리뷰어 모두에게 강조되며, 리뷰어들은 더 포괄적인 평가 관점을 adopt(채택)하는 경향이 있음을 발견했습니다. 또한, 프로모션 강도(promotional intensity)와 논문 본질적 새로움(inherent novelty) 간의 관계를 조사하였고, 이는 논문의 혁신 수준에 따라 달라진다는 점을 확인했습니다.

- **Performance Highlights**: 강력한 프로모션 언어(promotional language)는 높은 혁신성을 가진 논문에 긍정적인 평가를 가져오는 반면, 중간 혁신성을 가진 논문에서는 리뷰어들 간의 이견(disagreement)과 유의미한 상관관계가 나타났습니다. 반면, 극히 높은 또는 낮은 새로움의 논문에는 프로모션 언어가 미치는 영향이 미미함을 발견하였습니다. 이는 학술 평가의 회색 지대에서 프로모션 언어가 어떻게 작용하는지를 보여줍니다.



### Why Sampling Is Not Choosing: Intentionality, Agency, and Moral Responsibility in Large Language Models (https://arxiv.org/abs/2606.13441)
- **What's New**: 이 논문은 인공지능 언어 모델(LLMs)을 도덕적 책임이 있는 주체로 추정하는 경향을 비판합니다. 저자들은 이러한 주장을 잘못된 것으로 간주하며, 도덕적 책임을 지기 위해서는 내재적인 의도성과 자아 귀속 행위가 바탕이 되는 행위 주체가 필요하다고 주장합니다. 논문은 트랜스포머 기반 모델이 이러한 조건을 충족하지 못하며 오직 확률적 입력-출력 매핑에 의해 작동한다고 설명합니다.

- **Technical Details**: 저자들은 LLM과 같은 트랜스포머 기반 시스템이 내재적인 의도성과 자기 귀속적인 행동의 분별력이 없기 때문에 도덕적 책임의 주체가 될 수 없다고 주장합니다. 그들은 의도성(intention)이라는 개념을 정의하고, 이는 인간의 사고와 행동에서 어떻게 작용하는지를 설명합니다. 또한, 의도성이 외부 해석에 의존하는 파생적(intentionality)인지 아니면 시스템 내에 내재된(intrinsic) 것인지의 구분을 통해, 도덕적 책임은 오직 내재적인 의도성과 연관되어야 한다고 주장합니다.

- **Performance Highlights**: 이 논문은 현대 AI 철학 및 정신철학 논의에 기여하며, LLM의 출력을 도덕적으로 평가할 수 있다는 것과 진정한 행위 주체로 인정받을 수 있는 것은 다르다는 점을 강조합니다. 기존의 문헌들은 LLM이 인간과 같은 도덕적 추론을 신뢰할 수 없음을 시사하고, 논문은 트랜스포머 기반 모델의 확률적 다음 토큰 생성 아키텍처가 도덕적 책임을 위한 필수 조건을 충족하지 못함을 체계적으로 분석합니다. 이 연구 결과는 AI 시스템의 도덕적 책임에 대한 논의를 진전시키는 데 중요한 기여를 합니다.



### Cross-Modal Masked Compositional Concept Modeling for Enhancing Visio-Linguistic Compositionality (https://arxiv.org/abs/2606.13288)
Comments:
          Accepted to ACL 2026 Main Conference, 25 pages

- **What's New**: 본 논문에서는 MACCO (MAsked Compositional Concept MOdeling)라는 새로운 프레임워크를 제안하여 기존의 Vision-Language 모델(VLMs)의 구성적 이해 능력을 향상시키고 있습니다. 이 방법은 하나의 모달리티에서 구성 개념을 마스킹하고 전체 컨텍스트 정보를 기반으로 재구성함으로써 cross-modal compositional 구조를 보다 효과적으로 캡처합니다. 이러한 접근 방식은 기존의 하드 네거티브 샘플에 대한 의존성을 줄일 수 있도록 설계되었습니다.

- **Technical Details**: MACCO는 두 개의 보조 목표인 Masked-augmented Cross-Modal Alignment Loss (MCA)와 Masked-augmented Intra-Modal Regularization Loss (MIR)를 도입하여 교차 모달 재구성과 정렬 학습을 촉진합니다. MCA는 마스킹된 텍스트 또는 이미지의 글로벌 특성을 교차 모달 대조 학습 과정에 통합하는 반면, MIR은 각 모달리티 내에서 마스킹된 인스턴스의 글로벌 특성을 정규화하여 표현의 붕괴를 방지합니다. 이러한 기법들은 모델이 구문 구조 및 언어적 정보를 더 잘 캡처할 수 있게 합니다.

- **Performance Highlights**: 다양한 구성 벤치마크에서 MACCO의 효과를 검증하였으며, 실험 결과 구성적 이해가 크게 향상됨을 보여주었습니다. 또한, 개선된 구성성은 텍스트-이미지 생성 및 다중 모달 대형 언어 모델에서도 이점을 제공합니다. MACCO는 기존의 하드 네거티브 마이닝 기법과 통합될 경우 추가적인 성능 개선을 이끌어낼 수 있습니다.



### TimeLens: On-Device Artifact Recognition with Retrieval-Augmented Question Answering for the Grand Egyptian Museum (https://arxiv.org/abs/2606.13267)
Comments:
          6 pages, 4 figures, 5 tables. Submitted to AIVRCH 2026

- **What's New**: 이번 연구에서는 TimeLens라는 인공지능 기반의 이중 언어 모바일 가이드를 소개합니다. 이 앱은 사용자가 전시물에 스마트폰을 겨냥하면 실시간으로 유물 인식이 이루어지고, 사용자는 영어 또는 아랍어로 후속 질문을 할 수 있습니다. 이 연구는 51개의 카탈로그화된 유물의 세부 유사성, 훈련 데이터와 핸드헬드 카메라 조건 간의 간극, 그리고 인공지능 가이드가 역사적 사실을 잘못 진술할 위험성 등의 세 가지 문제를 해결하고 있습니다.

- **Technical Details**: TimeLens의 핵심 기술적인 기여로는 첫 번째로, 데이터 품질 중심의 반복 연구를 통해 개발된 온디바이스 유물 탐지기가 있습니다. YOLOv8n 모델은 비디오 기반 데이터 세트를 통해 훈련되어, 평균 0.995의 최고 정확도로 실시간 인식이 가능하게 설계되었습니다. 두 번째로, 108개의 기록으로 구성된 ChromaDB 지식 기반을 기반으로 하는 이중 언어 RAG (Retrieval-Augmented Generation) 가이드가 개발되어, 지원하는 두 언어인 영어와 아랍어로 응답할 수 있도록 하였습니다.

- **Performance Highlights**: TimeLens는 실시간 온디바이스 인식을 통해 연령대가 다양한 사용자들에게 빠르고 신뢰할 수 있는 답변을 제공합니다. 연구 결과, YOLOv8n 모델이 모든 실패 클래스를 해결하며, 5.97MB의 TensorFlow Lite 자산으로 중급 스마트폰에서 실시간으로 작동 가능하다는 것을 보여줍니다. 또한, RAG 기반 가이드는 불필요한 허구의 답변을 줄이며 대기 시간을 30초 이상에서 약 10초로 단축시켰습니다.



### ComAct: Reframing Professional Software Manipulation via COM-as-Action Paradigm (https://arxiv.org/abs/2606.13239)
- **What's New**: 이 연구는 기존의 컴퓨터-사용 에이전트들이 가진 한계를 극복하기 위해 새로운 패러다임인 COM-as-Action(ComAct)을 제안합니다. COM(구성 객체 모델)을 활용하여 전문 소프트웨어 조작을 실행 가능한 프로그램 합성으로 재구성하였으며, 이를 통해 GUI 인터랙션의 취약성을 해소하고 더 높은 일관성을 제공합니다. 또한, 실질적인 CAD 소프트웨어에서 운영되는 에이전트를 평가하기 위한 최초의 벤치마크인 ComCADBench를 소개합니다.

- **Technical Details**: COM은 Microsoft가 도입한 이진 인터페이스 표준으로, 다양한 응용 프로그램 간의 프로그래밍 가능한 통신을 가능하게 합니다. COM은 구조화된 프로그래밍 인터페이스를 통해 소프트웨어 내부를 노출하고, Python과 같은 언어에서 접근할 수 있는 호출 가능한 객체의 계층 구조를 제공합니다. 이 연구는 전문 소프트웨어 조작을 부분 관찰 가능 마르코프 결정 과정으로 모델링하며, 각 행동이 COM 인터페이스를 호출하는 실행 가능한 Python 스크립트로 구성된다는 점이 중요합니다.

- **Performance Highlights**: ComActor는 ComCADBench에서 최첨단 성능을 달성하며, 기존의 GUI 기반 에이전트들이 실패하는 긴 시간의 작업에서 강력한 회복력을 보입니다. 제공된 실험 결과는 ComActor가 외부 CAD 벤치마크에서도 뛰어난 일반화를 나타내며, CAD 소프트웨어 조작에 있어서 새로운 표준을 세우기 위한 잠재력을 보여줍니다. ComForge는 대규모 교육을 위한 확장 가능한 플랫폼을 제공하여, ComActor의 훈련 과정을 지원합니다.



### Understanding helpfulness and harmless tension in reward models (https://arxiv.org/abs/2606.13209)
Comments:
          The source code used in this study is publicly available at: this https URL\_tension

- **What's New**: 이 논문은 인간의 피드백을 기반으로 하는 강화학습(RLHF)에서 보상 모델(Reward Model, RM)의 정렬 긴장(alignment tension)을 연구하고 있습니다. 혼합 목적 설정에서 훈련된 보상 모델이 단일 목적 모델보다 성능이 낮다는 것을 발견하였으며, 이는 목표 간의 간섭(interference)을 시사합니다. 연구에서는 목표에 따라 연결된 신경세포(neurons)를 식별하고, 이들이 기능적으로 어떤 역할을 하는지 분석하였습니다.

- **Technical Details**: 본 연구는 보상 모델을 크게 세 가지 설정, 즉 유용성(helpfulness) 만, 무해성(harmlessness) 만, 및 혼합 목표 설정으로 구분하여 훈련하였습니다. 믹스된 모델은 신경세포의 활성화를 기반으로 각 목표와 관련된 신경세포를 확인하였으며, 이들 신경세포가 목표에 긍정적인 영향을 미치는 동시에 반대 목표에는 부정적인 영향을 미친다는 점을 강조합니다. 유용성과 무해성에 연관된 신경세포의 거의 절반이 공유된다는 사실도 강조되었습니다.

- **Performance Highlights**: 혼합 목표를 가진 보상 모델은 여러 평가 작업에서 단일 목표 모델보다 낮은 성과를 보였으며, 이는 유용성과 무해성 사이의 간섭을 나타냅니다. 또한 혼합 모델은 전이 가능성(transferability) 및 보상 보정(reward calibration)이 약한 경향을 보였습니다. 연구의 결과는 다중 목표 정렬이 단순한 행동적 거래가 아니라 내부 회로의 겹침으로 인한 표현 간섭 문제임을 나타내며, 이는 향후 정렬 목표의 기계적 제어(mechanistic control) 및 분리(disentanglement) 방법론에 중요한 함의를 제공합니다.



### Getting Better at Working With You: Compiling User Corrections into Runtime Enforcement for Coding Agents (https://arxiv.org/abs/2606.13174)
- **What's New**: 이 논문에서는 대화형 대규모 언어 모델(LLM) 에이전트의 사용 증가와 관련된 연구를 다룹니다. 사용자의 수정 사항이 각 세션에서 기억되더라도 미래의 세션에서 지켜지지 않는 경우가 발생하며, 이를 해결하기 위한 새로운 접근 방식을 제안합니다. Test-time Rule Acquisition and Compiled Enforcement (TRACE)를 통해 사용자 수정을 명시적인 실행 조건으로 변환하는 방법을 설명합니다.

- **Technical Details**: TRACE는 사용자가 제공한 수정 사항을 원자적 규칙으로 변환하고, 이를 런타임(checks)에서 검증하도록 통합하는 프로세스를 가집니다. 이 시스템은 사용자의 대화에서 도출된 피드백을 저장하여 다음 세션에서 에이전트가 반드시 준수해야 할 조건으로 만듭니다. 이러한 방식은 메모리에 기반한 기존 시스템과 다르게, 사용자의 수정 사항을 능동적으로 적용하여 세션을 처리합니다.

- **Performance Highlights**: TRACE 시스템은 ClawArena와 MemoryArena에서 실험을 통해 성능을 평가하였으며, 견고한 개선을 나타내었습니다. ClawArena에는 100%에서 37.6%로, MemoryArena에서는 100%에서 60.5%로 반복적인 선호 위반을 줄였습니다. 이 결과는 단순한 메모리 기반 접근 방식만으로는 해결할 수 없는 반복적인 장애를 효과적으로 해결함을 보여줍니다.



### MiniPIC: Flexible Position-Independent Caching in <100LOC (https://arxiv.org/abs/2606.13126)
Comments:
          13 pages, 5 figures

- **What's New**: 이번 논문에서는 Minimalistic PIC (MiniPIC)를 제안하고 있습니다. MiniPIC은 vLLM 디자인을 기반으로 하며, 사용자 제어의 캐시 재사용 기법과 함께 위치 인코딩이 없는 KV 캐시를 포함하여 구성됩니다. 또한, 인퍼런스 서버 내에서 포지션 변환 없이 캐시된 K 벡터를 저장하는 방법을 사용합니다.

- **Technical Details**: MiniPIC은 세 가지 사용자 친화적 원시 기능인 블록 정렬 패딩(block-aligned padding), 스팬 구분자(span separator, SSep), 그리고 프롬프트 의존성(prompt depend, PDep)을 제공합니다. 이러한 기능들은 해싱 동작과 효과적인 블록 레벨 인과(attention structure)를 수정하는 데 사용됩니다. MiniPIC은 100줄 이하의 코드 변경만으로 다양한 PIC 방법론(Block-Attention, EPIC, Prompt Cache)을 구현할 수 있습니다.

- **Performance Highlights**: 2WikiMultihopQA에서 MiniPIC은 PF(prefill) 처리량을 49% 개선하였고, 캐시된 스팬의 첫 번째 토큰 시간(`time-to-first-token`)을 두 배에서 최대 100배 단축시켰습니다. 또한 캐시되지 않은 스팬의 선형 PF 스케일링을 유지하며, 최악의 경우 오버헤드는 단 5.7%에 불과합니다.



### Demystifying Hidden-State Recurrence: Switchable Latent Reasoning with On-Policy Reinforcement Learning (https://arxiv.org/abs/2606.13106)
- **What's New**: 본 논문에서는 Latent chain-of-thought (CoT) 개념을 개선하기 위해 SWITCH라는 새로운 프레임워크를 제안합니다. SWITCH는 명시적인 경계 토큰(<swi> 및 </swi>)을 도입하여 Latent reasoning을 보다 효율적으로 최적화 할 수 있도록 돕습니다. 이러한 경계는 정책 비율을 명확히 하고, 기계적 분석에 적합한 기회를 제공합니다. 그 결과, SWITCH는 기존의 hidden-state-recurrence 접근 방식보다 우수한 성능을 보여줍니다.

- **Technical Details**: SWITCH는 세 가지 단계로 모델을 학습시키며, 첫 번째 단계에서는 <swi> 및 </swi>로 CoT 스팬을 감싸는 SFT(선택적 정교화 훈련)를 수행합니다. 다음 단계에서는 텍스트를 <latent> 위치로 점진적으로 대체하는 커리큘럼이 운영됩니다. 마지막으로, Switch-GRPO는 경계 토큰을 사용해 정책 비율을 잘 정의하여 강화 학습을 가능하게 합니다. 이 모든 단계에서 <swi>와 </swi>는 통제를 가능하게 하며 연속적인 잠재 블록 안의 컴퓨테이션을 분석하기 용이하게 만듭니다.

- **Performance Highlights**: SWITCH는 MATH-500 벤치마크에서 79.3% 정확도로 이전의 Coconut 스타일의 최강 기준선을 25.7 포인트 초과하는 성과를 기록하였고, Switch-GRPO를 활용한 경우에는 Latent 호출 비율을 반으로 줄이면서도 문제의 정확도를 12.6 포인트 향상시켰습니다. 이는 SWITCH가 성능과 효율성을 동시에 개선할 수 있음을 보여줍니다. 또한 경계 토큰을 통한 기계적 분석을 통해 세 가지 중요한 결과를 도출하였으며, 이는 Latent reasoning의 이해도를 크게 높입니다.



### The Illusion of Multi-Agent Advantag (https://arxiv.org/abs/2606.13003)
- **What's New**: 이 논문은 자동 생성된 다중 에이전트 시스템(MAS)이 단일 에이전트 시스템(SAS)보다 잘 작동한다는 기존 이론에 의문을 제기합니다. 연구자는 자동 MAS가 실제로 SAS인 Chain-of-Thought with Self-Consistency (CoT-SC)보다 성능이 떨어진다는 것을 입증했습니다. 특히, 연구에서는 기존의 평가 프레임워크가 다중 에이전트 시스템의 중요한 설계 문제를 간과하고 있다는 사실도 밝혔습니다.

- **Technical Details**: 이 연구에서는 MAS의 성능을 평가하기 위해 진단용 합성 데이터셋(Synthetic Multi-Hop Financial Reasoning, SMFR)을 도입했습니다. SMFR은 명확한 하위 작업 구조와 맥락 분리, 병렬화 가능성을 제공하여 자동 MAS의 성능을 평가할 수 있도록 설계되었습니다. 또한, 전문가가 설계한 MAS는 이러한 구조가 잘 갖춰질 경우 MAS의 이점이 실제로 발생할 수 있음을 보여주었습니다.

- **Performance Highlights**: 연구 결과, 자동 MAS는 많은 설정에서 SAS의 성능에 미치지 못하며 오히려 비용 효율성 면에서 CoT-SC의 성능이 뛰어남을 보여주었습니다. 전문가 설계 MAS는 구조적 체계를 갖출 경우 복잡한 작업에서도 뛰어난 성능을 발휘하는 것으로 나타났습니다. 전체적으로, 자동 MAS의 성능 이점이 복잡성의 표면적 요소에 의해 발생하는 경향이 있음을 발견하였습니다.



### Order Is Not Contro (https://arxiv.org/abs/2606.12923)
Comments:
          52 pages, 7 figures

- **What's New**: 이 논문은 AI 정렬(alignment), 해석 가능성(interpretability), 조절(steering), 그리고 신경 교란(neural perturbation) 연구들이 질서를 유도하는 객체들을 식별함을 주장합니다. 질서는 제어(control)와는 다르며, 제어는 수신자-게이티드(responder-gated) 반응 법칙이 필요하다고 설명합니다. 이 법칙은 물리적 상태, 행동, 환경, 그리고 수신자 상태를 반응 이동, 싱크(sink), 노력(effort), 그리고 오목형 프로젝션(basin projection)으로 매핑하는 것을 포함합니다. 이를 통해 생물학, 대형 언어 모델(LLM), 어댑터, 그리고 확률적 작용 패널에서 제어의 개념을 식별합니다.

- **Technical Details**: 논문은 외부 개입이 반응 법칙을 생성하는 방식과 그 작용을 다룹니다. 성질이 명시된 매개변수(denominator)를 통한 스토캐스틱 반응 커널(𝒫δ(dy|x,a))을 도입하고, 이는 물리적 상태, 행동 또는 드라이브(drive), 배스(bath), 수신자 상태, 비교(factor)에서 반응 위치와 관련된 분포를 유도합니다. 응답 법칙은 이 커널의 요약 및 유한 차이로 서술되며, 반응 경로 ℛδ와 행동 효과 Δℛδ로 표현됩니다. 제어는 한계가 명확하게 설정된 상태에서의 개입 노력(finited intervention effort)에 의해 부여됩니다.

- **Performance Highlights**: 연구의 성과는 생성된 출력과 어댑터 조건에서 예측 가능한 반응 법칙을 보여줍니다. 네 가지 물질 조건에 걸쳐 반응 벡터는 72.8-73.7%의 컴포넌트 신호 정확도를 기록하였으며, 비제로 컴포넌트에 대해서는 84.3-84.8%로 증가합니다. 관찰자들은 시스템 효과와 목표 가족의 예측에서 각각 93.6% 및 91.7%의 정확도를 기록하며, 이는 실질적인 응용 가능성과 관련이 있음을 시사합니다.



### MDForge: Agentic Molecular Dynamics Pipeline Design under Sparse Simulator Feedback (https://arxiv.org/abs/2606.12916)
- **What's New**: 이번 연구에서는 MDForge라는 LLM(대형 언어 모델) 기반 에이전트를 설계하여 분자 동역학(Molecular Dynamics, MD) 파이프라인 설계를 자동화하는 혁신적인 접근 방식을 제안합니다. 이 시스템은 기존의 MD 에이전트들이 정해진 도구를 사용하는 것과는 달리, 개방형 코드 생성을 통해 에이전트의 행동을온라인으로 조정할 수 있습니다. MDForge는 물리 전문가들 간의 토론을 통해 희소 보상을 밀집된 학습 신호로 변환하는 방법을 사용하여, 실험과 유사한 성능을 달성할 수 있습니다.

- **Technical Details**: MDForge는 인컨텍스트 업데이트 규칙인 PRISM(프로세스-보상 해석을 통한 하위 시스템 중재)을 기반으로 하여, MD 파이프라인의 스테이지(준비, 평형화, 생산 샘플링, 분석)에 따라 피드백을 수집합니다. 이 시스템은 각 단계의 진단을 효과적으로 활용함으로써, 에이전트가 각 스테이지 경계에서 피드백을 받을 수 있도록 합니다. 또한, 플랫폼에서는 다양한 물리학 전문가들이 토론을 진행하여 MDForge의 행동을 수정하는 데 필요한 매길 유형의 비판을 생성합니다.

- **Performance Highlights**: MDForge는 세 가지 SAMPL(정확한 자유 에너지 데이터셋) 벤치마크에서 경쟁력 있는 성능을 보입니다. 특히, CB[7] 파이프라인은 관찰되지 않은 후보 게스트 라이브러리에서 새로운 바인더를 발견하였고, 이는 wet-lab 경쟁 NMR 실험에 의해 고친밀도, 피코몰 단계의 바인더로 확인되었습니다. 이 연구 결과는 연구와 개발에 있어 중요한 진전을 보여줍니다.



### Trait, Not State: The Durability of Reading Identity in Social Highlighting (https://arxiv.org/abs/2606.12904)
Comments:
          12 pages, 3 figures, 3 tables

- **What's New**: 이 연구는 독자의 문서 하이라이트 선택이 개인의 고유한 특성인지 혹은 상태인지에 대한 새로운 질문을 제기합니다. 이전 연구들이 문서 선택에 초점을 맞췄다면, 이번 연구는 사용자의 하이라이트 기록을 시간적으로 지속 가능한 특성으로 분석합니다. 이를 통해 하이라이트 선택의 지속성에 대한 데이터 기반 컨센서스를 형성하고자 하였습니다.

- **Technical Details**: 연구에서는 독자의 첫 6개월 하이라이트 기록을 프로파일로 삼고, 이 프로파일의 개인 대 타인 평균정확도(average-precision) 이점을 후속 선택에 대해 측정했습니다. 데이터는 시간에 일치하는 부정적인 샘플을 사용해 개인 드리프트를 공급 드리프트와 분리하는 설계가 포함되었습니다. 또한, 매치된 셀에서 90% 이상의 이점을 유지하는 Durable Signal을 식별했습니다.

- **Performance Highlights**: 하이라이트를 통해 구축된 개인 프로필은 비 개인적인 이전에 비해 읽기 정확도를 약 3배 높이며, 개인의 선택적 특성을 시간에 걸쳐 높은 정확도로 유지하는 것을 보여줍니다. 이 연구에서는 개인 프로필이 단지 시간에 따른 여파가 아닌 지속 가능한 선택 서명을 제공함을 입증했습니다. 이러한 결과는 추천 시스템의 효과성을 개선할 수 있는 중요한 통찰을 제공합니다.



### Zero-source LLM Hallucination Detection with Human-like Criteria Probing (https://arxiv.org/abs/2606.12900)
Comments:
          Accepted at ICML 2026

- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs)의 환각(hallucination)을 감지하기 위한 새로운 패러다임인 Human-like Criteria Probing for Hallucination Detection (HCPD)를 제안합니다. HCPD는 인간 평가자의 다면적인 추론을 모방하며, LLM이 판단을 가중치를 둔 해석 가능한 기준 세트로 분해하고, 각 기준에 대한 점수를 집계하여 최종 진실성을 측정합니다. 특히, 외부 참조가 없고 모델 내부 정보가 제한된 제로 소스(zero-source) 환경에서도 효과적으로 작동합니다.

- **Technical Details**: HCPD의 핵심은 Human-like Criteria Probing (HCP) 메커니즘으로, 사전 훈련된 LLM 에이전트가 쿼리-답변 쌍에 대해 적응적으로 미세한 기준 세트(예: 사실 정확성, 논리적 일관성)를 생성하고 이들의 중요도에 따라 점수를 매깁니다. 이는 각 기준의 가중치를 맥락에 맞게 조정하여 인간 전문가의 복잡한 판단을 모사하게 됩니다. 또한 이러한 적응성 판단 기능을 위해 우리는 약한 감독(weak supervision)에서 파생된 보상 기반 정렬(training alignment) 방법을 도입합니다.

- **Performance Highlights**: HCPD는 다양한 실험을 통해 기존의 최첨단 방법들보다 일관되게 우수한 성능을 보였습니다. 특히, 생성 과정의 변동성을 줄이고 해석 가능성을 유지하기 위해 다중 샘플링 집계 전략을 적용하여 강력한 결정을 내릴 수 있음을 강조합니다. 이 연구는 제로 소스 환각 감지에서의 강력한 실용성을 입증하며, 이를 통해 안전하고 신뢰할 수 있는 LLM 활용을 위한 기반을 마련하였습니다.



### Magnifying What Matters: Attention-Guided Adaptive Rendering for Visual Text Comprehension (https://arxiv.org/abs/2606.12898)
- **What's New**: 이번 논문에서 소개한 Visual Text Comprehension (VTC)은 텍스트를 이미지로 변환하여 비전-언어 모델 (VLM)이 직접 읽을 수 있게 함으로써, LLM의 컨텍스트 윈도우 한계를 우회합니다. 기존의 VTC 파이프라인은 렌더링과 레이아웃을 고정된 전처리 단계로 취급하며, VLM이 시각화된 텍스트를 처리하는 방식에 대한 기계적 이해가 부족한 문제를 지적합니다. 이러한 문제를 해결하기 위한 새로운 메소드인 AGAR (Attention-Guided Adaptive Rendering)를 제안하였으며, 이는 텍스트의 중요 시각 패치(top-K important visual patches)를 식별하고 이를 확장하여 페이지를 재렌더링한 후, 정답을 재추론하는 방식입니다.

- **Technical Details**: AGAR는 훈련이 필요 없는 모델 독립적인 방식으로, VLM의 중간-후기 레이어에서의 주의(attention) 점수를 활용하여 중요 시각 패치를 식별하고 이를 단어 범위(word spans)로 매핑합니다. 이후 이러한 범위를 확대하여 재렌더링한 페이지에서 정답을 재추론하는 과정을 수행합니다. 연구 결과, AGAR는 VLM의 플러그 앤 플레이(plug-and-play) 형태로 정확도를 지속적으로 향상시킬 수 있으며, 여러 벤치마크(benchmarks)에서 강력한 성능을 보였습니다.

- **Performance Highlights**: AGAR는 총 9개의 VTC 벤치마크와 4개의 VLM 백본에서 실험을 통해, 훈련이 필요 없이 기존 VLM의 정확도를 향상시키는 성과를 보였습니다. 또한, AGAR는 사후 훈련(post-training)과 잘 결합되어 더욱 향상된 결과를 도출하고, 시각적 및 텍스트 입력의 저하에 대해 견고함을 유지하는 특징이 있습니다. 이러한 결과들은 AGAR가 VTC에서 증거 활용의 주요 병목 문제를 해결할 수 있는 중요한 방법이 될 수 있음을 시사합니다.



### Multi-Bitwidth Quantization for LLMs Using Additive Codebooks (https://arxiv.org/abs/2606.12876)
Comments:
          37 pages, 12 figures

- **What's New**: 본 연구에서는 기존의 훈련된 모델을 기반으로 다양한 비트폭(bitwidth)으로 정밀도를 조정할 수 있는 새로운 기법인 Drop-by-Drop을 제안합니다. 이 기법은 한 모델로 여러 비트폭을 지원하며, 정보 이론의 원리에 기반하여 LLM(대형 언어 모델)의 가중치를 동적으로 제어할 수 있는 가능성을 보여줍니다. 이를 통해, 단일 체크포인트에서 여러 비트폭을 지원하면서도 뛰어난 성능과 효율성을 유지할 수 있습니다.

- **Technical Details**: Drop-by-Drop은 다중 비트폭(post-training quantization)을 위한 새로운 프레임워크로, Matryoshka 스타일의 감독(supervision)을 도입하여 가중치를 효율적으로 관리합니다. 특히, LLM의 가중치가 일반적으로 따르는 가우시안 분포를 기반으로 하여, 추가 비트를 포함함으로써 재구성의 정확성을 점진적으로 높일 수 있도록 합니다. 이는 정밀도 제어를 가능하게 하여, 다양한 리소스 제약이 있는 환경에서도 유연한 동작을 가능하게 합니다.

- **Performance Highlights**: Drop-by-Drop은 여러 비트폭에서 저렴한 perplexity와 높은 작업 정확도를 유지하며, 훈련이나 재조정 없이 리소스가 제한된 환경에서 부드러운 성능 저하를 보여줍니다. 또한, 기존의 고정된 정밀도 모델을 사용하는 방법과 비교했을 때, Drop-by-Drop은 메모리 오버헤드를 크게 줄이면서도 효율적이고 경쟁력 있는 성능을 제공합니다.



### ProPlay: Procedural World Models for Self-Evolving LLM Agents (https://arxiv.org/abs/2606.12780)
- **What's New**: 이번 논문에서는 ProPlay라는 프로시저 중심의 세계 모델을 소개합니다. 이 모델은 LLM 에이전트가 이전의 경험을 바탕으로 미래의 경로를 리허설할 수 있도록 프로시저 레벨의 사전 재생(preplay)을 지원합니다. ProPlay는 성공적인 경로를 프로시저로 추상화하고, 이를 통해 에이전트가 환경의 동적 변화를 더 잘 이해할 수 있도록 돕습니다.

- **Technical Details**: ProPlay는 환경 지식을 프로시저 그래프 형태로 표현하며, 각 프로시저 간의 전이(transition)와 신뢰성 기록을 관리합니다. 에이전트는 각 에피소드 전에 수행될 작업에 대해 이 그래프를 기반으로 프로시저 수준의 사전 재생을 수행하며, 환경 피드백을 통해 그래프를 지속적으로 개선합니다. 이 과정에서 프로시저의 재사용 가능한 패턴을 활용하고, 낮은 수준의 행동 선택의 유연성을 유지합니다.

- **Performance Highlights**: ProPlay는 ScienceWorld, τ-Bench, PlanCraft와 같은 공개 벤치마크에서 평가되었으며, 기존의 강력한 기준선에 비해 환경 이해력과 자가 진화 능력이 일관되게 향상되었습니다. 이 연구는 절차 중심의 자기 진화 에이전트 모델을 통해 환경 동적 구조를 효과적으로 포착함으로써, LLM 에이전트가 온라인에서 학습하도록 지원함을 강조합니다.



### Agentic MPC for Semantic Control System Resynthesis (https://arxiv.org/abs/2606.12774)
Comments:
          7 pages, 5 figures

- **What's New**: 이 논문은 MPC(모델 예측 제어)가 높은 수준의 컨텍스트 정보와 동적으로 결합할 수 없는 한계를 극복하기 위해 agentic MPC 프레임워크를 소개합니다. 이 프레임워크는 대형 언어 모델 기반의 에이전트와 통합하여, 자연어 메시지와 환경 관찰을 포함한 이질적인 입력을 해석하고 제어 사양을 재합성할 수 있게 합니다. 특히, 자율주행 시나리오에서 개인의 선호에 맞춰 조정되고 사회적 상황에 반응할 수 있는 시스템의 효과성이 입증됩니다.

- **Technical Details**: Agentic MPC는 MPC 컨트롤러와 함께 작동하여 시스템의 컨텍스트를 지속적으로 모니터링하는 에이전트를 활용합니다. 에이전트는 사용자 입력, 외부 관찰 및 교통 규칙과 같은 외부 지식 소스를 포함한 이질적인 정보 소스를 수신합니다. 이를 기반으로 에이전트는 현재 상황을 해석하고 제어 사양을 수정할 필요가 있는지를 결정합니다. 또한 에이전트는 다양한 정보 채널과 연결할 수 있는 도구를 사용하여 MPC 컨트롤러의 사양을 재구성하는 역할을 합니다.

- **Performance Highlights**: 이 연구는 CARLA 시뮬레이션을 활용하여 자율주행 시나리오에서 agentic MPC 프레임워크의 성능을 검증하였습니다. 에이전트가 사용자 입력을 해석하고 외부 지식을 활용하여 문맥에 적합한 결정을 내리는 능력은 자율주행 시스템의 안전성과 효율성을 높이는 데 기여합니다. 특히, 에이전트가 공공도로에서의 긴급차량 양보와 같은 사회적 상호작용을 적절히 처리할 수 있는 능력이 강조됩니다.



### Detecting Functional Memorization in Code Language Models (https://arxiv.org/abs/2606.12764)
- **What's New**: 이번 논문에서는 기능적 기억화 (functional memorization)에 대한 연구를 통해 코드 생성에서 모델의 훈련 데이터로부터 기능적 로직이 얼마만큼 복원 가능한지를 조사합니다. 기존의 연구들이 대부분 텍스트 유사성에 초점을 맞춘 것과 달리, 이 연구는 의미적으로 동등한 코드가 어떻게 텍스트적으로 다를 수 있는지를 탐구합니다. 이를 통해 모델의 출력을 감사하는 메트릭이 텍스트적 겹침을 초월해야 한다는 필요성을 강조합니다.

- **Technical Details**: 저자들은 Olmo-3-32B라는 공개 소스 모델의 중간 훈련 데이터를 사용하여 기능적 기억화를 연구합니다. 파이썬 함수 시그니처를 이용하여 훈련된 모델과 참조 모델의 출력을 비교하는 반사실적 (counterfactual) 설정을 구축합니다. 측정 기준으로는 텍스트 겹침(BLEU), 구조적 코드 유사성 (CodeBLEU), 실행 기반 클론 탐지 (execution-based clone detection) 등이 사용됩니다.

- **Performance Highlights**: 결과적으로, 중간 훈련이 모든 메트릭에서 유사성을 증가시킴을 보여줍니다. 특히, 3.9%의 샘플이 기능적으로 기억화된 것으로 판별되었으며, 이는 특정 훈련 데이터의 기능적 로직을 생성된 코드가 그대로 반영하고 있음을 의미합니다. 이러한 결과는 기능적 기억화의 존재를 보여주며, 텍스트 겹침을 넘어서는 감사 체계의 필요성을 뒷받침합니다.



### Rethinking Psychometric Evaluation of LLMs: When and Why Self-Reports Predict Behavior (https://arxiv.org/abs/2606.12730)
Comments:
          Accepted as an Oral (Contributed Talk) at the ICML 2026 Workshop on Combining Theory and Benchmarks (CTB)

- **What's New**: 이번 논문은 LLM(대형 언어 모델)의 행위 경향성을 저비용 심리 측정 도구인 자기 보고(Self-Report, SR)를 통해 예측할 수 있는 가능성을 탐구하고 있습니다. 연구진은 SR과 행동 사이의 연관성이 존재하나 선택적임을 발견했으며, 이는 행동이 자각되지 않거나 문맥에 따라 변동할 수 있음을 나타냅니다. 특히, 연구는 Big 5와 TPB(Planned Behavior Theory)라는 두 가지 이론적 프레임워크를 비교하였습니다.

- **Technical Details**: 연구는 2×2×2의 팩토리얼 설계를 채택하여 TPB와 Big 5의 효과를 시험했습니다. 자료는 11개의 최첨단 LLM과 4가지 행동 과제(위험 감수, 아첨, 정직, 암묵적 바이어스)에 걸쳐 수집되었습니다. 연구에서는 SR과 행동 사이의 일치성을 측정하기 위해 세션 문맥과 정체성 유도 방법을 변경하였습니다.

- **Performance Highlights**: SR-행동 일치성은 문맥에 따라 달라졌으며, TPB의 경우 인간 수준의 일치성을 보여주었으나 Big 5는 그렇지 않았습니다. 또한, 별도의 세션에서의 일치성은 행동이 문맥 밖에 고정되어 있을 때만 유지되는 반면, 아첨과 같은 성격의 행동은 문맥에 달려 있어 붕괴되었습니다. 이러한 발견은 기존의 성격 테스트가 LLM 행동 예측에 적합하지 않을 수 있음을 시사합니다.



### Keep Policy Gradient in Charge: Sibling-Guided Credit Distillation for Long-Horizon Tool-Use Agents (https://arxiv.org/abs/2606.12634)
Comments:
          13 pages, 4 figures, 7 tables. Submitted to EMNLP 2026 Industry Track

- **What's New**: 이 논문은 Sibling-Guided Credit Distillation(SGCD)이라는 새로운 접근법을 소개합니다. 기존의 훈련 방법인 self-distillation(SD)이 도구 사용 능력을 약화시킬 수 있음을 증명했습니다. SGCD는 실패와 성공한 시도에서 유용한 정보를 활용하여 기울기를 재조정하는 대신 기여를 할당하는 신호를 제공합니다.

- **Technical Details**: SGCD는 동적 샘플링을 통해 동일한 작업을 수행하는 실패한 시도와 성공한 시도를 혼합하여 그룹화합니다. 그 결과로 생성된 시뮬레이션 롤아웃은 외부의 강력한 LLM에 의해 요약되어 단계적으로 기여를 참조하게 됩니다. 이렇게 생성된 기여 신호는 정책 경량주를 업데이트하는 데 사용됩니다.

- **Performance Highlights**: SGCD는 AppWorld와 τ3-항공사 벤치마크에서 GRPO 비교군보다 더 나은 성능을 보여줍니다. AppWorld에서는 테스트 정상에서 42.9에서 45.6으로 향상되었고, τ3-항공사에서는 pass@1이 0.583에서 0.602로 증가했습니다. 이러한 결과들은 SGCD가 혁신적인 접근법임을 입증합니다.



### PersonaDrive: Human-Style Retrieval-Augmented VLA Agents for Closed-Loop Driving Simulation (https://arxiv.org/abs/2606.12616)
- **What's New**: 이 논문에서는 PersonaDrive라는 새로운 스타일 조건화된 VLA (vision-language-action) 주행 모델을 소개합니다. 이 모델은 인공지능이 제공하는 데이터에 기반하여 스타일 있는 주행 경로를 생성하고, 실제 사람의 주행 패턴을 반영함으로써 시뮬레이션 훈련 파이프라인에서 더 다양한 행동을 만듭니다. 특히, 동일한 기반 모델을 사용하여 다양한 스타일을 구현할 수 있어, 스타일 변경이 간단한 데이터베이스 쿼리로 가능하게 합니다.

- **Technical Details**: PersonaDrive의 핵심 전략은 스타일 지시를 기반으로 한 인간 주행 데이터에서 가져온 실행 예시를 활용하는 것입니다. 세 단계로 진행되는 파이프라인은 (i) 이미지-텍스트 유사도 점수를 사용한 스타일별 주행 데이터의 오프라인 트리플릿 마이닝, (ii) 스타일별 데이터베이스에 대해 미세한 제어 인코더와 결합된 경량의 검색 헤드를 훈련, (iii) 검색된 컨텍스트 포인트를 사용하여 웨이포인트 예측에서 행동 시연으로 활용하는 것입니다. 이 구조를 통해 각 스타일마다 별도의 모델 재훈련 없이도 스타일을 쉽게 변경할 수 있습니다.

- **Performance Highlights**: Bench2Drive 과제에서 PersonaDrive는 SimLingo에 비해 4.6%의 주행 점수 향상을 달성하였고, HiP-AD 대비 2.5% 상승했습니다. 또한, 모든 스타일 조건에서 가장 높은 주행 점수를 기록하였으며, 보수적인 지시에서 공격적인 지시로 조건을 전환할 때 속도와 가속도가 각각 18%와 25% 증가했습니다. 이러한 결과는 스타일 조건 변경이 단순한 인덱스 스왑을 통해 가능하다는 점에서 효율성을 강조합니다.



### Quickest Detection of Hallucination Onset: Delay Bounds and Learned CUSUM Statistics (https://arxiv.org/abs/2606.12476)
Comments:
          14 pages, 1 figure

- **What's New**: 이 논문에서는 할루시네이션(환각) 검출기에서 발생하는 지연 문제를 변화 탐지(quickest change detection) 문제로 새로 정의하고 접근했습니다. 직접적인 반응 속도, 즉 환각이 시작된 순간부터 경고까지 소요되는 토큰 수를 측정함으로써 기존의 평가 방식과 차별화합니다. 저자들은 일반적인 AUC(Area Under Curve)보고서의 한계를 지적하고, 환각 시작을 변화점으로 설정하여 이를 최적화된 속도로 반응할 수 있는 모델링 방법을 제안했습니다.

- **Technical Details**: 저자들은 할루시네이션의 발생을 첫 번째 차수 마르코프 모델로 예측하여, Lorden의 최소 지연 경계(bound)를 설정했습니다. 이 과정에서 이론적인 지연 기준을 기반으로, 1%의 거짓 경고율(faulse-alarm rate)에서 약 1.3개의 토큰이 필수적으로 소모됨을 밝혔습니다. 그리고 인과 재귀 모델(causal recurrent labeler)은 학습된 CUSUM처럼 작동하며, 속도 측정에 있어서 선형 기반보다 우수하다는 것을 보여주었습니다.

- **Performance Highlights**: 실험 결과, 기존의 선형 기준선에 비해 재귀 모델은 11-13개의 토큰 만에 반응하며, 이는 31토큰의 선형 기준선 대비 훨씬 더 개선된 성능을 나타냅니다. 그러나 대부분의 이득은 더 나은 점수에 기인하며, 세quential accumulation의 조그마한 기여도 확인되었습니다. 저자들은 정보의 열(detector delay)의 차이를 정보 비율(information-rate) 문제와 결합하여 설명하며, 이는 환각 검출에서 구조적 제한을 다룰 수 있게 합니다.



### Identifiability Without Gaussianity: Symbolic World Models and Near-Infinite Temporal Consistency (https://arxiv.org/abs/2606.12471)
Comments:
          Pre-print

- **What's New**: 이 논문에서는 Joint-Embedding Predictive Architectures (JEPAs)가 세계의 진정한 잠재 변수를 선형적으로 회복할 수 있는 조건이 가우시안 정적 과정이라는 점을 확립했다. 하지만 저자들은 새로운 Physics-Grounded Symbolic Architecture (PGSA)를 제안하며, 이는 모든 물리적 영역에서 정밀한 선형 식별성을 달성할 수 있음을 증명했다. 또한 PGSA는 숫자 정밀도에 의해 오차가 제한되며, 비가우시안 시스템에서도 시간 일관성을 유지할 수 있는 방법을 제시했다.

- **Technical Details**: PGSA는 세 가지 구성 요소로 정의되며, 각 구성 요소는 알려진 물리 법칙을 나타내는 결정론적 실행 가능한 함수들로 구성된다. 이 구조는 시간 불변성을 가지며, 물리 변수들 간의 인과 관계를 나타내는 방향 그래프를 포함한다. PGSA는 비가우시안 시스템에서도 근사적인 오차 없이 정확한 표현을 제공하므로, 기존 통계적 월드 모델보다 대조적인 성과를 낳는다.

- **Performance Highlights**: PGSA는 모든 물리적 조건에서 정확한 선형 식별성을 보장하며, 사용된 모델의 수치 정밀도만으로 오차가 제한된다고 강조한다. 특히 비가우시안 세계에서도 이론적으로 무한대의 시간 동안 일관성을 유지하는 성질인 근사-무한 시간 일관성을 유지할 수 있음을 보였다. 이는 통계적 월드 모델이 가질 수 없는 특성으로, 저자들은 이는 세계 역학의 인과 생성기에 기초해 설계된 구조적 특성이라고 주장한다.



### Occupational Prompting Reveals Cultural Bias in Large Language Models (https://arxiv.org/abs/2606.12443)
- **What's New**: 이번 연구에서는 LLM의 응답에서 사회적 역할이 어떻게 작용하는지를 조사하기 위해 직업 기반의 프롬프트(occupational prompting)를 도입했습니다. 저자들은 회계사(accountant), 교사(teacher), 엔지니어(engineer), 간호사(nurse) 등 다양한 직업 정체성을 기반으로 한 질문에 LLM이 어떻게 반응하는지를 분석했습니다. 이러한 접근을 통해 직업에 따른 가치 표현의 차이를 탐구하고, 모델의 응답이 미국 사회와 같은 문화적 기준에 어떻게 포지셔닝되는지를 알아보았습니다.

- **Technical Details**: 이 연구에서는 Integrated Values Surveys(IVS)를 기반으로 한 평가 파이프라인을 적용하여, LLM의 응답을 Inglehart-Welzel 문화 공간으로 투영하였습니다. 10개의 가치 질문을 사용하여 각 직업 보정된 응답이 어떻게 문화 지도에서 분포되는지를 분석했습니다. Principal Component Analysis(PCA)를 통해 얻은 2차원 문화 공간에서는 생존과 자기 표현, 전통과 세속의 축이 해석됩니다.

- **Performance Highlights**: 결과적으로, LLM의 응답은 직업 점검에 따라 서구 중심의 지역에서 움직임을 보였으며, 각기 다른 직업이 그 지역 내에서의 응답에 차이를 초래했습니다. 이 연구는 LLM이 직업 기반의 프롬프트를 통해 가치 표현의 구조적 패턴을 불러일으킨다는 것을 보여줍니다. 즉, 직업 프롬프트는 단순한 역할 레이블이 아니라, 가치 표현에 강한 영향을 미친다는 것을 알 수 있었습니다.



### Marginal Alignment Does Not Guarantee Joint-Distribution Fidelity: An Official-Reference Audit of Nemotron-Personas-Korea with Cross-Locale Replication (https://arxiv.org/abs/2606.12433)
- **What's New**: 이 논문은 synthetic persona 데이터셋이 공식 인구통계와의 정렬을 바탕으로 신뢰성을 주장하지만, 실제 사용자는 이러한 데이터셋을 연령, 성별, 지역, 직업 등 다양한 속성의 결합 형태로 소비한다고 강조합니다. 특히 Independence-Assumption Footprint (IAF)라는 감사 방법을 제안하여 데이터셋의 속성 조합이 외부의 공식 참조와 어떻게 비교되는지를 분석합니다. 이 방법은 NVIDIA에서 발행한 Nemotron-Personas-Korea (NPK)의 감사 사례를 통해验证되었습니다.

- **Technical Details**: IAF는 데이터셋 카드에 문서화된 속성 결합을 외부의 공식 참조와 비교하는 감사 도구입니다. 데이터셋은 한국 통계청(KOSIS), 한국 고용정보원(KEIS) 등 다양한 소스를 기준으로 확인됩니다. NPK의 경우, 공식 통계와의 정렬을 주장하면서도 독립적으로 처리된다는 가정으로 인해 중요한 조인트 구조들이 손상됨을 보여주고 있습니다.

- **Performance Highlights**: NPK를 통해 IAF를 적용한 결과, 나이와 군 복무 프로필의 구조적 불일치, 남성 중심 직업에서 여성 Representation이 과도하게 평준화되는 현상이 발견되었습니다. 논문은 데이터셋의 장점과 문제점을 파악하고, 참고할 수 있는 데이터와 감사 자료를 제공하여 다른 synthetic persona 자원에 대한 적용 가능성을 보여주고 있습니다.



### Two Wrongs, No Right: Auditing Social-Desirability Bias in LLM Annotators for Computational Social Scienc (https://arxiv.org/abs/2606.12426)
- **What's New**: 본 논문에서는 LLM(대형 언어 모델) 주석자가 컴퓨테이셔널 사회 과학(CSS)에서 사용될 때 발생할 수 있는 오류가 연구자가 보고할 수 있는 실증적 결론을 어떻게 영향을 미치는지를 보여줍니다. 세 가지 오픈 소스 7B 모델(Zephyr, Mistral-Instruct, Qwen2.5-Instruct)을 감사하여 소셜 바람직성 오류가 단일 방향으로 발생하지 않음을 발견했습니다. Zephyr 모델은 해로운 라벨을 과소 적용하는 관용 편향(leniency bias)을 보였으며, Mistral과 Qwen는 과도한 교정(overcorrection) 현상을 나타냈습니다.

- **Technical Details**: 이 연구는 72개의 실험 셀에 걸쳐 6개의 TweetEval 작업을 감사하여 주요 오류 패턴(관용 편향, 과도한 교정, 중립성 편향)을 세 가지로 나누어 분류합니다. LLM 주석의 신뢰성을 평가하기 위해 사용된 다양한 프롬프트 조건을 통해, 중립 또는 안전 프레이밍과 같은 방법들이 이러한 오류를 수정하지 못하거나 심지어 악화시키는 결과를 가져왔습니다. 이러한 결과를 바탕으로 진단 FBR(False Beneficial Rate) 및 FAR(False Alarm Rate) 서명을 포함한 검증 프로토콜을 개발하였습니다.

- **Performance Highlights**: 이 연구는 LLM이 공적 담론을 형성하는 텍스트에 라벨을 부여할 때, 상반된 오류 모드가 동일한 작업에서 서로 다른 방향으로 결과를 유도할 수 있음을 강조합니다. Zephyr 모델은 증거율(예: 혐오 발언의 비율)을 정확하게 나타내지만, 분류 오류는 크게 나타나며 집합 검증을 오도할 수 있는 우발적인 취소가 발생합니다. 따라서 LLM을 사회적 공익을 위한 신뢰할 수 있는 AI 도구로 활용하기 위해서는 모델과 프롬프트를 측정 도구의 일환으로 보고하는 것이 필수적입니다.



### AI SciBrief as a Gateway to Research: A Framework for Onboarding Students into New Research Areas (https://arxiv.org/abs/2606.12413)
Comments:
          This is the version of the article accepted for publication in TELE 2025 after peer review. The final, published version is available at IEEE Xplore: this https URL

- **What's New**: 이번 논문에서는 고등 교육의 모든 수준에서 학생들이 마주하는 정보 과부하 문제를 해결하기 위한 교육적 프레임워크를 제시합니다. AI SciBrief라는 플랫폼을 활용하여 자동으로 과학 트렌드의 요약(digests)을 생성하는 방식으로, 연구 과정의 초기 단계를 원활하게 합니다. 이 도구는 금융(finance), 의학(medicine), 교육(education) 등 여러 분야에서 사용할 수 있으며 커리큘럼에 통합될 수 있습니다.

- **Technical Details**: AI SciBrief는 대규모 언어 모델(LLM) 기반으로 설계되었으며, 이는 학생들에게 주제 선택(term paper) 및 문헌 리뷰(literature review)를 용이하게 하는 구체적인 방법론을 제공합니다. 이 프레임워크는 학위 논문(dissertations) 작성 및 최근의 과학 트렌드에 대한 지속적인 모니터링을 가능하게 합니다. 이를 통해 학생들이 더 효과적으로 자료를 검색하고 지식을 생성하는 단계로 빠르게 전환할 수 있도록 지원합니다.

- **Performance Highlights**: AI SciBrief는 학생들이 정보 검색에서 지식 창출로의 전환을 촉진하며, 인지 부담(cognitive load)을 경감시키는 역할을 합니다. 이 연구는 AI SciBrief가 연구의 "게이트웨이(gateway)"로 기능한다고 강조하며, 학생들의 동기 부여를 증진시키고 연구 효율성을 높이는 데 기여할 것이라고 결론짓습니다.



New uploads on arXiv(cs.IR)

### OneRetrieval: Unifying Multi-Branch E-commerce Retrieval with an Editable Generative Mod (https://arxiv.org/abs/2606.13533)
Comments:
          Any Question please contact: benchen4395@gmail.com

- **What's New**: 이 논문에서는 기존의 복합 검색 아키텍처를 통합할 수 있는 가능성을 제시합니다. OneRetrieval이라는 새로운 생성 검색(Generative Retrieval) 프레임워크를 소개하며, 이는 키워드 정렬 인코딩(Keyword-Aligned Encoding, KAE)을 기반으로 합니다. 이 방식은 실시간 편집 가능성과 우수한 검색 품질을 동시에 제공하며, 기존의 세 가지 검색 가지를 각각 가진 복합 구조를 하나의 모델로 대체할 수 있는 잠재력을 가지고 있습니다.

- **Technical Details**: OneRetrieval은 18개의 세부 속성 카테고리를 정보 이론적인 방식으로 6개의 코드북 그룹으로 정리하고, 각 코드북에는 파라미터를 재훈련하지 않고도 새로운 단어를 바인딩할 수 있는 예약된 슬롯을 두고 있습니다. 이를 통해 실시간으로 새로운 용어를 검색 시스템에 추가할 수 있는 기능을 보유하고 있으며, 이 방식은 편집 가능성을 기존의 훈련된 정책 대신 상위 사전에서 유지합니다. 네 단계의 세부 튜닝 파이프라인을 통해 검색 품질과 편집 가능성이 함께 확보됩니다.

- **Performance Highlights**: OneRetrieval은 대규모 산업 벤치마크에서 가장 강력한 생성 기반 모델과 동등한 검색 품질을 기록하며, 기존의 닫힌 코드북 방법으로는 실현할 수 없는 편집 가능성을 회복합니다. 시스템은 Kuaishou에 배포되어 매일 수억 건의 페이지 뷰(PV)를 처리하며, 기존의 역 인덱스(branch)를 대체함으로써 전환율(cv) 증가를 달성했습니다. 이는 실시간 단어 삽입과 검색 품질 향상이 가능함을 보여줍니다.



### CQC-RAG: Robust Retrieval-Augmented Generation via Cross-Query Consistency (https://arxiv.org/abs/2606.13438)
- **What's New**: 이 논문은 Cross-Query Consistency Hypothesis를 제안하여, 다양한 구문 구조의 질문에서 정답의 신뢰성을 높이기 위한 CQC-RAG 프레임워크를 소개합니다. 기존의 RAG 접근 방법들이 갖는 두 가지 주요 한계를 극복하기 위해, 이 방법은 쿼리 수준에서 다양성을 주입하고 교차 쿼리 일관성 평가를 통합적으로 설계하였습니다. 새로운 기법을 적용하여, 신뢰도가 높은 정답을 선택할 수 있는 구조를 제시합니다.

- **Technical Details**: CQC-RAG는 질문의 의미를 보존하면서 다양한 형태로 질문을 재작성하고, 이를 통해 얻은 공통의 문서 풀을 기반으로 쿼리별로 다른 이유 제공 맥락을 생성합니다. 이러한 방법론은 모델의 로짓을 기반으로 각 답변의 신뢰도를 평가하며, 높은 평균 신뢰도와 낮은 분산을 가진 답변을 선택하는 메커니즘을 사용합니다. 이는 외부 감독 없이도 답변의 신뢰성을 스스로 평가할 수 있게 해줍니다.

- **Performance Highlights**: CQC-RAG는 TriviaQA와 MuSiQue와 같은 오픈 도메인 질문 응답 데이터셋에서 기존의 Self-Consistency 및 Speculative RAG와 같은 방법들보다 뛰어난 성능을 보였습니다. 특히, TriviaQA에서 4.76 pp EM, MuSiQue에서는 9.12 pp EM의 성과 향상을 기록하여 교차 쿼리 일관성 평가의 효과성을 입증하였습니다. 이 결과는 소음으로 유발된 환각 답변을 효과적으로 필터링할 수 있음을 보여줍니다.



### CoDeR: Local Constraint-Compatible Retrieval Beyond Semantic Similarity (https://arxiv.org/abs/2606.13204)
- **What's New**: CoDeR는 주제적 관련성과 제약 호환성을 분리하여 제약 감수성 쿼리에서 발생하는 문제를 해결하는 새로운 정보 검색 방법입니다. 기존의 정보 검색 시스템은 주로 의미적 유사성에 최적화되어 있어, 문서가 쿼리의 의도를 위반할 때 문제가 발생합니다. CoDeR는 이러한 문제를 해결하기 위한 방법으로, 상반된 조건들을 포함하는 증거를 분리하여 처리합니다.

- **Technical Details**: CoDeR는 표준 주제 인코더와 추가적인 호환성 점수를 제공하는 바이 인코더(bi-encoder)를 사용하는 접근 방식을 채택했습니다. 이 방법은 대조적 증거를 토대로 한 어휘적 극성 조정을 통해 훈련되며, 이러한 호환성 신호를 사용하여 적합한 후보 문서들을 다시 점수화할 수 있습니다. CoDeR는 외부 대형 언어 모델(LLM)을 호출하지 않고도 이러한 작업을 처리할 수 있습니다.

- **Performance Highlights**: CoDeR는 세 가지 제어 진단 세트에서 성과를 측정했으며, 기존 방법들보다 높은 성능을 보여주었습니다. 특히, V@2 점수를 각각 20.59, 23.53, 5.77 포인트 줄여 초기에 제약을 위반하는 문서를 줄이면서 주제적 검색 품질을 유지하는데 성공했습니다. 이러한 결과는 CoDeR의 로컬 호환성 점수가 의미적 유사성 기반 검색보다 더 효과적임을 시사합니다.



### The Clustering Strikes Back: Building Cost-Effective and High-Performance ANNS at Scale with Helmsman (https://arxiv.org/abs/2606.13145)
Comments:
          Accepted by OSDI'26

- **What's New**: RedNote (Xiaohongshu)는 글로벌 소셜 미디어 플랫폼으로, 최근 HELMSMAN이라는 고성능 클러스터링 기반의 근접 이웃 검색 시스템을 개발했습니다. 이 시스템은 기존의 메모리 집약적인 ANNS 인프라스트럭처에서 발생하는 높은 비용과 메모리 요구 사항을 해결하며, 비용의 90%를 절감하고 수십억 규모의 인덱스 구축을 몇 시간 안에 가능하게 합니다. HELMSMAN은 GPU 가속 파이프라인과 함께 ANNS 지향 사용자 저장소 스택을 통해 높은 처리량과 낮은 대기 시간을 유지하는 데 주력하고 있습니다.

- **Technical Details**: HELMSMAN은 세 가지 주요 기술을 채택하여 클러스터링 기반 인덱스를 구축합니다. 첫째, 전통적인 커널 I/O 스택을 우회하는 ANNS 지향 사용자 공간 저장소를 구축하여 소프트웨어 오버헤드를 최소화합니다. 둘째, 쿼리와 top-kk 분포에 적응하는 학습 기반 프루닝 모듈을 개발하여 SSD 친화적인 배치 I/O를 지원합니다. 셋째, GPU 가속을 활용하고 CPU를 동적으로 할당하여 인덱스 구축을 신속하게 완료할 수 있도록 하였습니다.

- **Performance Highlights**: HELMSMAN은 기존 DRAM-SSD ANNS보다 2-16배 높은 처리량을 제공하며, in-DRAM ANNS의 최대 85%에 달하는 처리량을 유지합니다. 현재 운영 환경에서 안정적으로 작동하며, HELMSMAN은 40대의 머신만으로 약 30-40TB의 DRAM을 사용하여 이전에 35,000 CPU 코어와 0.35PB의 DRAM을 소비하던 온라인 트래픽을 처리하고 있습니다. 이러한 최적화는 기업에 수반되는 실제 비용을 대폭 절감하여 효율적인 운영을 가능하게 합니다.



### CFALR: Collaborative Filtering-Augmented Large Language Model for Personalized Fashion Outfit Recommendation (https://arxiv.org/abs/2606.13001)
- **What's New**: 이번 연구에서는 CFALR (Collaborative Filtering-Augmented Large Language Model for Recommendation)이라는 새로운 프레임워크를 제안합니다. 이 모델은 개인화된 의상 추천을 위한 최초의 LLM(대형 언어 모델) 기반 아키텍처로, 사용자와 의상 간의 상호작용을 자연어로 설명합니다. CFALR는 LLM의 패션 의미를 포착하며, CF(협업 필터링)로 강화된 임베딩을 사용하여 의미적 공간과 협업 상호작용 공간을 연결합니다.

- **Technical Details**: CFALR은 사용자-의상 상호작용을 자연어 기술로 변환하여 Personalized Fill-In-The-Blank (P-FITB) 문제로 프레임합니다. 이 모델은 오픈 소스 언어 모델을 백본 추천기로 활용하고, 사용자의 관계 및 콘텐츠 특징을 효과적으로 추출합니다. 비텍스트 특징은 단일 토큰으로 임베딩되며, 다양한 특징 타입에 대해 개별적인 훈련 가능한 선형 프로젝션 레이어가 적용됩니다.

- **Performance Highlights**: Polyvore와 IQON 데이터셋에서의 실험 결과, CFALR은 기존의 CF 기반 및 LLM 기반 의상 추천 모델들보다 우수한 성능을 보였습니다. 이 모델은 개인화된 fill-in-the-blank 및 의상 생성 작업에서 특히 높은 성과를 내며, 생성 품질과 추천 정확도를 효과적으로 결합하여 더 나은 결과를 제공합니다.



### Charge as a Construct-Validity Factor in Chinese Legal Case Retrieval: A Cross-Benchmark Aud (https://arxiv.org/abs/2606.12993)
- **What's New**: 본 논문에서는 중국의 법률 판례 검색(Chinese Legal Case Retrieval, LCR) 체계에서 NDCG@10의 성능을 평가하는 방법과 기존 시스템의 한계를 다룹니다. 연구 결과, 법률 사건의 주로 사용되는 'charge' (죄명) 정보가 검색 결과의 질에 미치는 영향이 크다는 것을 발견했습니다. 기존 LCR 벤치마크에서 charge 정보 사용이 실제 Legal Reasoning (사법적 추론) 능력 측정 대신 증거에 기반한 성능 측정으로 읽힐 수 있음을 지적하고 있습니다.

- **Technical Details**: 이 연구에서 소개된 CCE(Charge-Controlled Evaluation)는 법률 사건 검색 벤치마크의 유효성을 검증하기 위한 다기준 평가 패키지입니다. 이 패키지는 charge 정보를 활용한 다양한 평가 기준, 즉 charge-straified NDCG@10, charge-name occlusion, 및 charge-clustered significance testing을 제공합니다. 이를 통해 연구자들이 LCR 시스템이 어떻게 성능 향상에 기여하는지를 명확히 분석할 수 있도록 도와줍니다.

- **Performance Highlights**: 실험 결과, charge 정보의 활용이 벤치마크에서 NDCG@10 성능에 큰 영향을 미치는 것으로 나타났습니다. KELLER 시스템은 BM25 대비 99.2%의 성능을 회복하였지만, 이는 'charge' 정보가 결정적인 요소임을 반영합니다. 또한, zero-training channel이 LeCaRDv2에서 Recall@100을 +0.025만큼 향상시켰음을 보고하여 첫 단계에서 charge 정보의 유용성을 강조합니다.



### Trait, Not State: The Durability of Reading Identity in Social Highlighting (https://arxiv.org/abs/2606.12904)
Comments:
          12 pages, 3 figures, 3 tables

- **What's New**: 이 연구는 독자의 문서 하이라이트 선택이 개인의 고유한 특성인지 혹은 상태인지에 대한 새로운 질문을 제기합니다. 이전 연구들이 문서 선택에 초점을 맞췄다면, 이번 연구는 사용자의 하이라이트 기록을 시간적으로 지속 가능한 특성으로 분석합니다. 이를 통해 하이라이트 선택의 지속성에 대한 데이터 기반 컨센서스를 형성하고자 하였습니다.

- **Technical Details**: 연구에서는 독자의 첫 6개월 하이라이트 기록을 프로파일로 삼고, 이 프로파일의 개인 대 타인 평균정확도(average-precision) 이점을 후속 선택에 대해 측정했습니다. 데이터는 시간에 일치하는 부정적인 샘플을 사용해 개인 드리프트를 공급 드리프트와 분리하는 설계가 포함되었습니다. 또한, 매치된 셀에서 90% 이상의 이점을 유지하는 Durable Signal을 식별했습니다.

- **Performance Highlights**: 하이라이트를 통해 구축된 개인 프로필은 비 개인적인 이전에 비해 읽기 정확도를 약 3배 높이며, 개인의 선택적 특성을 시간에 걸쳐 높은 정확도로 유지하는 것을 보여줍니다. 이 연구에서는 개인 프로필이 단지 시간에 따른 여파가 아닌 지속 가능한 선택 서명을 제공함을 입증했습니다. 이러한 결과는 추천 시스템의 효과성을 개선할 수 있는 중요한 통찰을 제공합니다.



### TimeLens: On-Device Artifact Recognition with Retrieval-Augmented Question Answering for the Grand Egyptian Museum (https://arxiv.org/abs/2606.13267)
Comments:
          6 pages, 4 figures, 5 tables. Submitted to AIVRCH 2026

- **What's New**: 이번 연구에서는 TimeLens라는 인공지능 기반의 이중 언어 모바일 가이드를 소개합니다. 이 앱은 사용자가 전시물에 스마트폰을 겨냥하면 실시간으로 유물 인식이 이루어지고, 사용자는 영어 또는 아랍어로 후속 질문을 할 수 있습니다. 이 연구는 51개의 카탈로그화된 유물의 세부 유사성, 훈련 데이터와 핸드헬드 카메라 조건 간의 간극, 그리고 인공지능 가이드가 역사적 사실을 잘못 진술할 위험성 등의 세 가지 문제를 해결하고 있습니다.

- **Technical Details**: TimeLens의 핵심 기술적인 기여로는 첫 번째로, 데이터 품질 중심의 반복 연구를 통해 개발된 온디바이스 유물 탐지기가 있습니다. YOLOv8n 모델은 비디오 기반 데이터 세트를 통해 훈련되어, 평균 0.995의 최고 정확도로 실시간 인식이 가능하게 설계되었습니다. 두 번째로, 108개의 기록으로 구성된 ChromaDB 지식 기반을 기반으로 하는 이중 언어 RAG (Retrieval-Augmented Generation) 가이드가 개발되어, 지원하는 두 언어인 영어와 아랍어로 응답할 수 있도록 하였습니다.

- **Performance Highlights**: TimeLens는 실시간 온디바이스 인식을 통해 연령대가 다양한 사용자들에게 빠르고 신뢰할 수 있는 답변을 제공합니다. 연구 결과, YOLOv8n 모델이 모든 실패 클래스를 해결하며, 5.97MB의 TensorFlow Lite 자산으로 중급 스마트폰에서 실시간으로 작동 가능하다는 것을 보여줍니다. 또한, RAG 기반 가이드는 불필요한 허구의 답변을 줄이며 대기 시간을 30초 이상에서 약 10초로 단축시켰습니다.



### Semantic Identification of IoT Devices from Behavioral Primitives (https://arxiv.org/abs/2606.12793)
Comments:
          14 pages, 3 figures, 4 tables

- **What's New**: 이 논문에서는 IoT 장치의 식별을 개선하기 위해 Manufacturer Usage Description (MUD) 프로파일과 Access Control Entries (ACE)를 사용한 새로운 접근 방식을 제안합니다. 기존 방법들은 패킷이나 흐름 기록에 의존하는 반면, MUD 프로파일은 통신 정책에 기반한 구조화된 표현을 제공합니다. 이 연구는 ACE-level 표현이 장치 별 행동 구별을 더 효과적으로 유지하면서도, 정밀한 통신 상태가 변화할 때에도 유용성을 잃지 않도록 하는 방법을 탐구합니다.

- **Technical Details**: 연구진은 28개의 공개 MUD 프로파일에서 1,023개의 ACE 인스턴스를 활용하여 ACE-level 시맨틱 표현을 구축했습니다. ACE는 프로토콜, 엔드포인트, 방향 및 포트 의미를 정의하며, 이를 통해 구조화된 정책 수준에서 통신 행동을 표현합니다. BGE-M3 모델을 사용하여 텍스트 기반 입력을 수치적인 임베딩 벡터로 변환하였고, 이 과정에서 JSON의 보일러플레이트를 제거하여 데이터의 효율성을 높였습니다.

- **Performance Highlights**: 실험 결과, ACE-level 임베딩은 전체 프로파일 MUD 임베딩보다 장치 수준 행동 구별을 더욱 잘 유지하며, 관측된 실시간 IoT 트래픽에서도 ACE 시맨틱 매칭이 더 강력한 식별 증거를 제공함을 보여주었습니다. 특히, 초기 관찰 단계에서 높은 순위의 후보 중에서도 올바른 장치를 자주 유지하였고, 겹침이 적은 트래픽 조건에서도 유용성을 유지하는 것으로 나타났습니다.



### How Fine-Grained Should a RAG Benchmark Be? A Hierarchical Framework for Synthetic Question Generation (https://arxiv.org/abs/2606.12789)
- **What's New**: HieraRAG는 retrieval-augmented generation (RAG) 시스템의 벤치마크 구성 시 문항 특성을 세분화하는 최적의 수준을 정의하는 계층 구조의 프레임워크로 소개된다. 이 연구는 질문의 복잡성, 답변 유형 및 언어적 변variation을 세 가지 차원에서 변수로 삼아 5,872개의 합성 질문-답변(QA) 쌍을 생성하였다. 또한, Coherence Ratio라는 새로운 메트릭을 도입하여 질문 분류 구조의 질을 정량화하고 문항의 구조적 차이를 비교한다.

- **Technical Details**: HieraRAG의 구조는 질문 복잡성(Question Complexity, QC), 답변 유형(Answer Type, AT), 언어적 variation(Linguistic Variation, LV)이라는 세 가지 기본 차원에서 3단계의 세분화 수준(거칠게, 중간, 세밀하게)으로 평가된다. 질문 복잡성 차원은 복잡한 문제 해결을 요구하며, 답변 유형은 정보의 직접 추출과 새로운 형태의 생성 간의 구별을 나타낸다. 언어적 variation은 질문의 표현과 문서 내용 간의 어휘적 정렬을 정량화하여 문장의 직접적인 사용부터 의미 이해를 요하는 패러프레이즈된 개념까지 다양성을 포함한다.

- **Performance Highlights**: 연구 결과, 최적의 granularity는 차원에 따라 달라지며, 질문의 복잡성 차원에서 세밀한 구분(8개 카테고리)이 이득을 볼 수 있는 반면, 답변 유형과 언어적 variation은 중간 granularity(4개 카테고리)에서 최고 성능을 보인다. Coherence Ratio 메트릭에 따른 구조적 차이가 확인되었으며, 인간 평가에서도 합성 질문 품질이 98%로 높게 나타났다. 이는 RAG 시스템의 성능을 평가하기 위한 적절한 방식의 필요성을 강조하며, 다양한 질문 특성에 따라 다른 세분화 수준이 요구된다는 점을 보여준다.



### ToolSense: A Diagnostic Framework for Auditing Parametric Tool Knowledge in LLMs (https://arxiv.org/abs/2606.12451)
- **What's New**: ToolSense는 도구 카탈로그를 입력으로 받아 자동으로 세 가지 진단 벤치마크를 생성하는 오픈 소스 LLM 기반 진단 프레임워크입니다. 이 시스템은 RRB(Realistic Retrieval Benchmark), MCQ(다중 선택 질문) 탐색 벤치마크 및 QA(질문-답변) 탐색 벤치마크를 포함하여 기존 도구의 이해도를 평가합니다. ToolSense는 도구 이해의 실제 격차를 보여주고, 도구 재조합의 성능이 비정상적으로 낮은 몇 가지 모델 구성을 드러냅니다.

- **Technical Details**: ToolSense는 도구 카탈로그를 구현하여 다단계 과정을 통해 벤치마크를 생성합니다. RRB는 세 가지 모호성 수준에서 사용자가 실제 쿼리와 유사한 간결한 요청을 테스트하여 검색 시스템의 일반화를 평가합니다. 하드-네거티브 풀을 구축하고, 병렬 생성 계층을 통해 각 배치에 대해 질의가 생성되어 이중 검증을 수행합니다.

- **Performance Highlights**: ToolBench를 기반으로 ToolSense의 적용 결과, 특정 모델 구성들은 일반적인 벤치마크에서 높은 성능을 보였지만 현실적인 쿼리에서 50-64%의 성능 저하를 일으켰습니다. 이는 fetched 모델들이 본질적인 도구 지식을 상실하고 있을 가능성을 시사하며, Stage 2 훈련이 Stage 1에서 학습한 도구 지식을 파괴할 수 있음을 보여줍니다.



New uploads on arXiv(cs.CV)

### InterleaveThinker: Reinforcing Agentic Interleaved Generation (https://arxiv.org/abs/2606.13679)
Comments:
          Project Page: this https URL Code: this https URL

- **What's New**: 이번 연구에서는 InterleaveThinker라는 다중 에이전트 파이프라인을 소개합니다. 이 시스템은 기존의 이미지 생성기를 텍스트와 이미지가 교차하는 방식으로 생성할 수 있게 하여, 복잡한 비주얼 내러티브 및 가이드라인 시스템에서의 활용 가능성을 제시합니다. 특히, 계획(agent) 및 평론가(agent) 역할을 수행하는 두 가지 에이전트를 통합하여 생성 과정의 정확도를 높였습니다.

- **Technical Details**: InterleaveThinker는 Planner agent를 사용하여 이미지-텍스트 입력 순서를 조직하고, 생성기에게 각 단계에서 요구되는 실행을 지시합니다. Critic agent는 생성된 출력을 평가하고, 계획된 지침에서 벗어난 샘플을 식별 및 재생성을 위한 지침을 다듬는 역할을 합니다. 이를 위해 특화된 데이터셋과 향상된 RL 기법(GRPO)을 활용하여, 전체 생성 궤적에 대한 보상 메커니즘을 설계했습니다.

- **Performance Highlights**: InterleaveThinker는 다양한 이미지 생성기에서 일관된 성능 향상을 보여주었습니다. 특히, interleaved generation에 대한 벤치마크에서 Nano Banana 및 GPT-5와 동등한 성능을 달성했으며, 논리 기반 벤치마크(WISE, RISE)에서 현저한 개선을 보였습니다. 이러한 결과는 다중 에이전트 협업을 통해 복잡한 순차적 추론 및 생성 능력을 극대화할 수 있음을 증명합니다.



### Modality Forcing for Scalable Spatial Generation (https://arxiv.org/abs/2606.13676)
- **What's New**: 이 논문은 Modality Forcing이라는 간단하고 확장 가능한 포스트 트레이닝 방법을 제안하여 단일 DiT 모델을 사용해 이미지와 깊이 맵을 동시에 생성할 수 있도록 한다. 이는 희소한 깊이 데이터에서 훈련할 수 있는 가능성을 제공하며, 이미지 생성과 깊이 예측 모두를 향상시킬 수 있는 점에서 기여한다. 특히, 기존의 복잡한 방법과 달리 쉽게 구현할 수 있어 활용도가 높아진다.

- **Technical Details**: Modality Forcing 방법은 RGB와 깊이에 각기 다른 노이즈 레벨을 지정하여 단일 DiT 모델이 세 가지 작업, 즉 이미지 및 깊이의 공동 생성, 깊이에서 이미지로의 변환, 이미지에서 깊이로의 변환을 지원하도록 설계되었다. 이 접근 방식을 통해 희소한 실제 깊이 주석으로부터 학습할 수 있고, 모달리티 간의 혼합 없이도 동작한다. 또한, 이 방법은 훈련된 모델이 다양한 작업에 조건화될 수 있도록 한다.

- **Performance Highlights**: 제안된 모델은 기존의 이미지-깊이 생성 모델에 비해 57% 향상된 성능을 보이며, 다른 단일 눈 깊이 추정기와 경쟁할 수 있을 정도로 우수한 성능을 나타냈다. 370M에서 3.3B 파라미터까지의 다양한 모델을 훈련하여, 더 큰 모델이 더 많은 이미지 데이터로 훈련될수록 더 정확한 깊이 예측을 할 수 있음을 입증했다. 이러한 결과는 T2I 모델이 공간적 생성의 스케일 가능한 프리트레이닝 목표가 될 수 있음을 보여준다.



### RepWAM: World Action Modeling with Representation Visual-Action Tokenizers (https://arxiv.org/abs/2606.13674)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 RepWAM이라는 새로운 representation-centric world action model(WAM)을 소개합니다. 기존의 WAM들은 pretrained video generation models에서 유래된 reconstruction-oriented video tokenizers를 사용하여 시각적 정확성을 유지하지만, 이러한 접근은 로봇 제어에 필요한 instruction-following dynamics를 학습하는 데 한계가 있습니다. RepWAM은 semantic visual-action latent space를 탐구하고 이를 통해 이러한 문제를 해결하고자 하며, 대상 행동과 시각적 상태를 연결하는 우수한 성능을 보여줍니다.

- **Technical Details**: RepWAM은 representation visual-action tokenizers를 바탕으로 구축된 세계 행동 모델로, 시각적 입력을 정렬된 시각 및 잠재 행동 토큰으로 매핑하는 작업을 포함합니다. 시멘틱 비디오 토크나이저를 사용하여 영상 오토인코더의 잠재 공간을 시각적 모델과 함께 정렬하며, 조작 중심의 동작을 포착하는 잠재 행동 토크나이저를 학습합니다. 이러한 두 개의 토크나이저는 비주얼 잠재 및 로봇 행동 간의 모달리티 간격을 줄이는 통합된 시멘틱 비주얼-액션 토크나이저를 형성합니다.

- **Performance Highlights**: RepWAM은 실제 조작 작업 및 시뮬레이션 벤치마크에서 평가되었으며, vision-language-action(VLA) 및 WAN-사전 훈련된 WAM 기초 모델과 비교하여 강력한 폐쇄 루프 행동을 보여주었습니다. RoboTwin 2.0의 Easy 작업에서 89.3, Hard 작업에서 88.4를 기록하여 시각적 및 행동 잠재에 공통된 시멘틱 표현을 기반으로 한 세계 행동 모델링의 가능성을 입증합니다. 이러한 결과는 RepWAM이 다양한 조작 설정에서 높은 성능을 제공한다는 것을 나타냅니다.



### SpatialClaw: Rethinking Action Interface for Agentic Spatial Reasoning (https://arxiv.org/abs/2606.13673)
Comments:
          Project page: this https URL

- **What's New**: 이번 연구는 비전-언어 모델(VLMs)의 공간 추론 능력을 향상시키기 위해 SpatialClaw라는 새로운 프레임워크를 제안합니다. 기존의 도구 증강 에이전트는 Action Interface(행동 인터페이스)에 따라 성능이 제한되었으나, SpatialClaw는 Python 코드를 통해 상태 유지형 커널을 활용하여 더 유연하고 개방적인 공간 추론을 가능하게 합니다. 이를 통해 에이전트는 중간 결과에 따라 분석을 조정하고 다양한 시각적 입력을 처리할 수 있습니다.

- **Technical Details**: SpatialClaw는 입력 프레임과 다양한 인식 및 기하 자료가 있는 영속적인 Python 커널을 활용하여 에이전트가 각 단계에서 실행 가능한 셀을 작성하도록 합니다. 기존 연구들은 단일 경로 코드 실행 또는 구조화된 도구 호출 인터페이스에 의존해 왔지만, SpatialClaw는 코드 생성을 통한 유연한 조합을 지원합니다. 이는 에이전트가 이전 결과에 기초하여 동적으로 인식 결과를 조정하고 조합할 수 있도록 합니다.

- **Performance Highlights**: SpatialClaw는 20개의 공간 추론 벤치마크에서 평균 59.9%의 정확도를 기록하며, 최근의 공간 에이전트보다 +11.2 포인트 향상된 성능을 보여줍니다. 특히 4D 비디오 추론 및 다중 뷰 추론에서 가장 높은 성과를 보였으며, 이는 전처리된 도구 호출 없이도 에이전트가 유연하게 기하적 계산을 수행함을 시사합니다. 이러한 성과는 다양한 모델 가족에서 일관되게 나타나, 시스템 프롬프트나 도구 세트의 수정 없이도 일반화됩니다.



### Flex4DHuman: Flexible Multi-view Video Diffusion for 4D Human Reconstruction (https://arxiv.org/abs/2606.13655)
Comments:
          18 pages, 8 figures. Code, and multi-view caption dataset available

- **What's New**: Flex4DHuman은 모노큘러(Monocular) 또는 희소 다중 뷰(Sparse multi-view) 비디오를 동적 객체에 대해 동기화된 밀집 다중 뷰 비디오로 변환하는 새로운 비디오 확산 모델입니다. 이전의 인간 중심(human-centric) 방법들이 뼈대(Skeleton)나 깊이 맵(Depth maps), 노멀(Normals) 또는 렌더링(target-view geometry)된 지형에 의존했던 것과는 달리, Flex4DHuman은 명시적인 기하학적 사전 정보(geometry priors)를 요구하지 않습니다. 대신, 상대 카메라 포즈 카메라-포즈 조건화(relative camera-pose conditioning)를 통해 생성 과정을 조정합니다.

- **Technical Details**: Flex4DHuman은 Wan 2.1 1.3B 텍스트-투-비디오(text-to-video) 모델에 기반하여 설계되었습니다. 이 모델은 5축(5-axis) 위치 인코딩(positional encoding)을 통해 카메라 및 뷰(view) 정보를 인코딩하며, 이는 공간-시간 RoPE(spatio-temporal RoPE)와 뷰 인덱스(view indices), 연속적인(SE(3)) 상대 카메라 기하학을 포함합니다. 모델은 포즈 추적(pose following), 유연한 참조-대상 뷰 생성(flexible reference-to-target view generation), 그리고 시간 전개(temporal rollout)를 위해 점진적으로 훈련됩니다.

- **Performance Highlights**: DNA-Rendering과 ActorsHQ에 대한 실험 결과, Flex4DHuman은 이전의 최첨단(state-of-the-art) 방법보다 뛰어난 성능을 보였습니다. 같은 구조가 혼합된 인간-동물 훈련 후 동물 범주(category)에 대해서도 일반화되는 것을 확인했습니다. 이러한 기능들은 Flex4DHuman이 캐주얼 모노큘러 비디오에서 스케일 가능한 4D 콘텐츠 생성을 위한 실용적인 단계를 마련한다는 것을 보여줍니다.



### World Tracing: Generative Pixel-Aligned Geometry Beyond the Visib (https://arxiv.org/abs/2606.13652)
Comments:
          World Labs Technical Report; Page: this https URL

- **What's New**: 새로운 연구에서는 World Tracing이라는 이미지에서 3D를 생성하는 방식을 제안합니다. 이 방법은 입력 픽셀과 정렬된 3D 점을 예측하며, 가시적인 표면 너머의 기하학적 구조까지 생성하는 것을 목표로 합니다. 따라서 가시적인 표면 재구성과 가려진 기하의 생성을 동시에 수행할 수 있습니다.

- **Technical Details**: World Tracing은 각 입력 픽셀에 대해 카메라 공간 내의 정렬된 3D 점 스택을 예측하며, WT-DiT라는 모델을 통해 이를 구현합니다. WT-DiT는 각 기하학적 레이어를 별도의 디노이징 토큰으로 처리하며, 이미지 격자에서 강력한 2D 시각적 선험을 활용합니다. 이 과정에서 깊이 채우기 전략을 사용하여 모든 레이어를 함께 훈련시킵니다.

- **Performance Highlights**: World Tracing는 여러 객체와 장면에 걸쳐 강력한 성능을 발휘하며, 가시적인 표면의 정확성과 완전한 기하 생성 모두에서 뛰어난 결과를 보입니다. 이 방법은 3D 장면 편집, 기하학 조건의 새로운 비디오 합성, 그리고 훈련이 필요 없는 텍스처 메쉬 생성을 가능하게 하여 3D 파이프라인의 효율성을 높입니다.



### Surflo: Consistent 3D Surface Flow Model with Global Sta (https://arxiv.org/abs/2606.13644)
Comments:
          Project webpage: this https URL

- **What's New**:  Surflo는 여러 개의 미정렬 RGB 뷰(views)를 K latent tokens로 압축하여 단일 글로벌 상태(global state)로 변환하고, 이 상태에서 3D 표면 포인트를 독립적으로 생성하는 새로운 피드포워드 모델입니다. 기존의 방법들이 고정된 해상도나 중복된 포인트 지도를 출력하는 것과 달리, Surflo는 다양한 해상도로 동적으로 표면을 생성할 수 있어 더 유연한 응용이 가능합니다. 특히, Surflo는 카메라 위치나 각도와 관계없이 전방위적으로 3D 구조를 재구성할 수 있는 장점을 지니고 있습니다.

- **Technical Details**:  Surflo의 디자인은 처리를 위해 필요 없는 뷰 수에 관계없이 고정된 크기의 latent representation을 생성합니다. 이를 위해 Surflo는 VGGT(backbone)이라는 강력한 기하학적 특징을 가진 모델을 기반으로 하여, 여러 입력 뷰의 토큰을 3D 위치 인코딩과 결합하여 압축시킵니다. 디코딩 과정에서는 flow matching 기법을 통해 독립적으로 표면 포인트를 생성하며, 인퍼런스 시간에 포인트 간의 상관관계를 반영하기 위해 가이던스 메커니즘을 사용합니다.

- **Performance Highlights**:  Surflo는 8개의 3D 재구성 벤치마크에서 기존 피드포워드 기준선을 초과하거나 동등한 성능을 발휘하며, 같은 latent representation으로부터 다양한 해상도의 출력을 생성할 수 있는 유일한 피드포워드 방법입니다. Surflo는 기존의 최적화 기반 방법들보다 훨씬 빠르며(속도는 수십 배 이상), 이는 로봇 공학 및 에이전트 내비게이션과 같은 여러 하위 응용 분야에서 매우 유용하게 사용될 수 있습니다.



### Revisiting Vehicle Color Recognition in Long-Tailed Surveillance Scenarios (https://arxiv.org/abs/2606.13625)
Comments:
          Accepted for presentation at the 2026 International Conference on Pattern Recognition (ICPR) - V3SC Workshop

- **What's New**: 이번 논문은 차량 색상 인지(Vehicle Color Recognition, VCR)에 대한 포괄적인 연구를 소개합니다. 특히, 실제 상황에서의 심각한 클래스 불균형 문제를 다루기 위해 새로운 데이터셋인 UFPR-VeSV를 활용합니다. 이 연구는 두 가지 생성적 전략인 RunDiffusion/JuggernautXL과 Gemini 2.0 Flash를 통해 합성 소수 클래스 데이터를 증강하는 과정을 제안하여, 드문 색상의 인식을 개선하고자 합니다.

- **Technical Details**: UFPR-VeSV 데이터셋은 파라과이 군경의 차량 관찰 이미지로 구성되어 있으며, hetergeneous viewpoints와 다양한 조명 조건과 같은 어려움들을 포함하고 있습니다. 저자는 합성 데이터를 통해 드문 색상의 충분한 훈련 샘플을 늘리기 위해 텍스트-조건 이미지 생성 및 이미지-조건 색상 편집 방식을 채택하고, 이러한 데이터와 함께 DINOv3 기능, 손실 재가중, 그리고 앙상블 융합 등의 전략을 사용합니다. 또한, 주의해야 할 품질 검사를 통해 합성된 이미지의 신뢰성을 높였습니다.

- **Performance Highlights**: 이 연구에서 최고의 성능을 보인 모델은 94.6%의 마이크로 정확도와 79.7%의 매크로 정확도를 달성했습니다. 이는 최근 문헌에 비해 매크로 정확도를 8.2% 포인트 개선한 수치입니다. 하지만 오류 분석 결과, 많은 경우가 시각적으로 모호하여 사람 주석자조차 이를 식별하기 어려웠다는 점을 강조하여, 색상 기반 차량 식별의 실제 한계를 부각시킵니다.



### Towards Effective Waste Segmentation for Automated Waste Recycling in Cluttered Background (https://arxiv.org/abs/2606.13587)
Comments:
          accepted at ICML 2026

- **What's New**: 본 논문에서는 효율적이고 자동화된 폐기물 관리의 중요성을 설명하며, 특히 심층 학습(deep learning) 기법을 활용한 자동 폐기물 분류(automated waste recycling, AWR)의 필요성을 강조합니다. 기존의 AWR 시스템은 대규모 백본 네트워크를 사용하여 효율성을 저하시키고, 혼잡한 장면에서 성능 저하를 겪고 있습니다. 이를 개선하기 위해, 공간과 스펙트럼 도메인을 효과적으로 활용하는 새로운 폐기물 분류 네트워크(EWSegNet)를 제안합니다.

- **Technical Details**: EWSegNet은 로컬 및 글로벌 문맥(context)을 포착하기 위해 단계적으로 구조화된 디자인을 갖추고 있습니다. 이 네트워크는 보조 기능 향상 모듈(auxiliary feature enhancement module, AFEM)을 도입하여 대상 물체의 경계(boundaries)를 강화하고, 클러터(cluttered) 상황에서 더 나은 세분화를 제공합니다. 제안된 구조는 공간 및 스펙트럼 피처를 결합하여 최적의 성능을 달성합니다.

- **Performance Highlights**: 실험 결과, ZeroWaste-aug, ZeroWaste-f, SpectralWaste 데이터셋에서 제안된 EWSegNet의 성능이 체계적으로 평가되었습니다. EWSegNet은 기존의 방법들에 비해 세분화 성능에서 향상된 결과를 보여주었습니다. 또한, 효율성을 유지하면서도 실행 가능한 폐기물 분류 및 분리 시스템을 제공하는 물체 경계 강조 기술을 활용하였습니다.



### EvTexture++: Event-Driven Texture Enhancement for Video Super-Resolution (https://arxiv.org/abs/2606.13580)
Comments:
          IEEE TPAMI 2026. Extended version of arXiv:2406.13457 (ICML 2024). Project page: this https URL

- **What's New**: 이 논문은 Event-based vision 기술을 활용하여 비디오 슈퍼 해상도(Video Super-Resolution, VSR)의 텍스처 복원에 초점을 맞춘 EvTexture++라는 프레임워크를 제안합니다. 기존 VSR 기법들이 모션 정제(motion refinement)에 초점을 맞춘 반면, EvTexture++는 텍스처 향상(texture enhancement)을 목표로 합니다. 이 시스템은 이벤트 기반 신호를 바탕으로 고주파(spatiotemporal) 세부 정보를 활용하여 더 정확하고 상세한 고해상도 출력을 생성합니다.

- **Technical Details**: EvTexture++는 텍스처 향상 전용 브랜치와 고해상도 영상의 정밀한 특성을 복원하기 위한 반복적 질감 향상 모듈을 포함하고 있습니다. 이 개선 과정은 고주파 텍스처 정보를 점진적으로 복구하면서, 이벤트 신호의 연속적 시간 모션 단서를 활용하여 텍스처 변동성을 줄입니다. 전체 시스템은 기존의 VSR 모델에 간편하게 통합될 수 있도록 설계되어, 다양한 아키텍처에서 실행 가능합니다.

- **Performance Highlights**: 다수의 데이터셋에서 실험 결과, EvTexture++는 최신 성능 기준(State-of-the-art, SOTA)을 달성했습니다. 특히, 텍스처가 풍부한 Vid4 데이터셋에서 PSNR에서 최대 1.55 dB의 유의미한 개선을 보였습니다. 이 논문에서 제안된 모델은 기존 VSR 모델과의 통합을 통해 향상된 성능을 창출할 수 있음을 입증하였습니다.



### Contrast-Informed Augmentation and Domain-Adversarial Training for Adult-to-Neonatal MR Reconstruction Generalization (https://arxiv.org/abs/2606.13562)
Comments:
          24 pages, 1 table, 7 figures

- **What's New**: 본 연구는 E2E-VarNet의 성능을 향상시키기 위해 contrast-informed 데이터 증강(data augmentation)과 domain-adversarial training(DAT)의 효과를 조사했다. 세 가지 학습 방식이 탐구되었으며, 그 중 혼합 학습 방식이 neonatal 데이터에 대해 뛰어난 성능을 보였다. 이 연구는 성인 데이터를 효과적으로 활용하여 neonatal MR reconstruction의 일반화 성능을 높이는 것을 목표로 한다.

- **Technical Details**: 연구에서는 unaugmented adult 데이터만 사용한 경우와 unaugmented 및 augmented adult 데이터 혼합, 그리고 DAT 목표를 추가한 혼합 데이터의 세 가지 학습 방식에 대해 성능을 비교하였다. 모델은 각 교육 방식에 따라 T2-weighted 뇌 MR 데이터로 훈련되었으며, 도메인 구분이 유지되는지 여부를 평가하기 위해 latent representation의 변화를 분석하였다. 특별히, data augmentation은 neonatal MR 특징을 근사하도록 설계되었다.

- **Performance Highlights**: Mixed-DAT 방식은 neonatal 데이터에서의 평가 결과, 최고의 성능을 보였다. R=4에서 SSIM 값을 0.924로 기록하였고, R=8에서는 SSIM 0.848 및 PSNR 29.56 dB을 기록하여, 성인 전용 학습 방식에 비해 우수한 성능을 입증하였다. 이러한 결과들은 domain adjersarial training이 강조된 데이터 증강과 함께 사용될 때, neonatal MR reconstruction의 일반화 성능을 개선할 수 있음을 나타낸다.



### Edit the Bits, Diff the Codes: Bitwise Residual Editing for Visual Autoregressive Models (https://arxiv.org/abs/2606.13558)
- **What's New**: 본 논문에서는 bitwise-residual VAR(generator) 모델에서의 이미지 편집을 위한 새로운 방법인 BitResEdit를 제안합니다. 이 접근법은 훈련 없이 가능하며, 기존의 VAR 편집기가 사용하는 두 가지 구조를 최대한 활용합니다: per-bit Bernoulli 예측 헤드와 다중 스케일 잔여 코드 필드입니다. 이를 통해 이미지의 특정 부분을 편집하면서도 관련 없는 콘텐츠는 보존할 수 있는 편집 기능을 제공합니다.

- **Technical Details**: BitResEdit는 두 단계로 작동합니다: 첫 번째 단계인 BitEdit는 어떤 비트가 샘플링되는지를 가이드하며, 두 번째 단계인 ResEdit는 샘플링된 변화를 이미지 코드의 어느 위치에 기록할지를 제어합니다. Infinity라는 VAR 생성기를 기반으로 설정되어, 연속적인 VAE 코드를 생성하고 이를 사용하여 이미지가 재구성됩니다. 이 모델의 독특한 수학적 성질을 이용하여 편집을 위한 자정 기법을 구현합니다.

- **Performance Highlights**: 실험 결과, BitResEdit는 같은 백본을 가진 VAR 편집기 중에서 가장 우수한 텍스트 정렬 성능을 달성하며, 기존 최강 편집기보다 CLIP 점수를 +1.07 향상시킵니다. 또한, 배경 보존 성능도 경쟁력이 있습니다. ablation 연구에서 BitEdit와 ResEdit이 타겟 정렬 및 배경 보존에서 서로 보완적인 역할을 한다는 것이 확인되었습니다.



### What's Old is New Again: Classical Dimensionality Reduction for Efficient Saliency-Guided Biometric Attack Detection (https://arxiv.org/abs/2606.13528)
Comments:
          16 pages (8 main, 2 references, 6 appendix), 4 figures (3 main, 1 appendix), 13 tables (3 main, 10 appendix)

- **What's New**: 이번 논문에서는 생체 인식에서의 saliency-guided training의 접근방식을 제시합니다. 기존의 방법들이 높은 비용과 도메인 특이성으로 인해 한계가 있었던 반면, 저자들은 PCA와 LDA에 영감을 받은 비용 효율적이고 확장 가능한 saliency 맵 생성 방법을 개발했습니다. 이 방법은 인간의 주석이나 도메인 지식 없이도 원시 데이터에서 saliency 맵을 생성할 수 있습니다.

- **Technical Details**: 제안된 saliency 획득 방법은 클래스별로 차별화된 정보를 잘 포착할 수 있는 고전적인 차원 축소 기법을 기반으로 하고 있습니다. Eigenfaces와 Fisherfaces를 통해 생성된 saliency 맵은 모델 훈련에 효과적으로 활용될 수 있으며, 여러 생체 공격 감지 도메인에서 이의 유효성을 검증했습니다. 생성된 saliency 맵들은 다양한 CNN 아키텍처를 통해 훈련 지원에 사용되었습니다.

- **Performance Highlights**: 논문에서 제안한 방법은 여러 테스트 도메인에서 기존의 saliency 기반 훈련 방법보다 더 나은 성능을 보였습니다. 모든 도메인에서 테스트한 결과, 제안된 방법으로 훈련된 모델들은 베이스라인을 초과하는 성능을 발휘했으며, 때때로 SOTA(sate-of-the-art) 방법보다도 우수한 결과를 보여주었습니다. 이러한 결과는 생체 공격 탐지에서의 saliency-guided training에 대한 새로운 접근법이 될 것으로 기대됩니다.



### MaskWAM: Unifying Mask Prompting and Prediction for World-Action Models (https://arxiv.org/abs/2606.13515)
- **What's New**: MaskWAM은 객체 중심의 세계-행동 모델로, 시각적 디스트랙션을 억제하고 언어 모호성을 줄이는 두 가지 주요 이점을 제공합니다. 이 모델은 미래 마스크 예측을 통해 강력한 의미적 감독을 제공하여 정책 성능을 크게 향상시킵니다.

- **Technical Details**: MaskWAM은 Transformer의 혼합 모델(Mixture of Transformers, MoT)을 활용하여 미래 RGB 프레임과 마스크, 행동을 공동 예측하는 통합 아키텍처입니다. 이 모델은 현재 RGB 와 첫 번째 프레임의 시각적 프롬프트를 처리하여 스페이셜 앵커를 제공하며, 이러한 접근 방식은 정책이 작업 관련 영역에 주의를 집중하도록 강제합니다.

- **Performance Highlights**: MaskWAM은 LIBERO에서 98.4%, RoboTwin에서 92.2%의 성공률을 달성하였으며 실제 환경에서도 언어 명확한 작업에서 84.3%, 언어 모호한 작업에서 84.9%를 기록하며 강력한 기준선보다 33.2% 더 뛰어난 성과를 나타냈습니다.



### Measurement-Calibrated Multi-Camera Fusion for Vision-Based Indoor Localization (https://arxiv.org/abs/2606.13509)
Comments:
          This paper has been accepted for presentation at the IEEE 22st International Conference on Automation Science and Engineering (CASE 2026)

- **What's New**: 이 논문에서는 실내 비전 기반 로컬라이제이션 시스템에서의 데이터 융합(data fusion)의 중요성을 강조합니다. 기존의 멀티 카메라 시스템이 일반적으로 블랙 박스(component)를 사용하여 평가되는 반면, 본 연구는 단일 카메라의 오류 특성을 명시적으로 분석하여 멀티 카메라 데이터 융합을 보정하고 최적화하는 방법을 제안합니다. 측정-보정된 융합(measurement-calibrated fusion) 접근법을 통해 오류의 기여도를 정량화하여 개선된 로컬라이제이션의 정확성을 입증했습니다.

- **Technical Details**: 이 시스템은 객체 인식 및 분류를 위한 신경망과 공간적 로컬라이제이션을 위한 투영 변환(projective transformation)을 결합합니다. 합쳐진 이미지에서 객체를 YOLO v8n을 사용하여 탐지하고, MediaPipe를 통해 2D 인체 자세추정을 수행하여 각 객체의 지면 접촉 위치를 계산합니다. 데이터를 융합하기 위해 선형 칼만 필터(linear Kalman Filter)를 사용하였으며, 각 카메라의 효과적 시야(field of view)를 고려하여 노이즈 특성을 조정합니다.

- **Performance Highlights**: 실험 결과, 데이터 융합은 단일 카메라 기반의 시스템에 비해 로컬라이제이션 정확도를 상당히 향상시켰습니다. 측정-보정된 융합 방법은 절대적인 정확도 향상은 제한적이었지만, 궤적의 변동성을 크게 줄이고 움직임의 부드러움을 향상시켜 안정적이고 연속적인 모션 추정이 필요한 응용 프로그램에서 요구되는 조건을 더 잘 충족시켰습니다. 이러한 결과는 비전 기반 실내 위치 추적 시스템의 데이터 융합 전략 설계 시 명시적인 오류 특성화의 가치를 분명히 보여줍니다.



### Heterogeneous LiDAR Early Fusion and Learned Re-Ranking Strategy for Robust Long-Term Place Recognition in Unstructured Environments (https://arxiv.org/abs/2606.13503)
- **What's New**: 새로운 접근법인 MinkUNeXt-VINE++는 두 개의 서로 다른 LiDAR 센서(즉, Livox Mid-360 및 Velodyne VLP-16)로부터의 이종 데이터를 조기에 융합하고 추론 시간에 학습된 재순위화 전략을 통합하는 방식을 제안합니다. 이 방법은 농업과 같은 비구조적 환경에서의 장소 인식 문제를 해결하기 위해 개발되었습니다. 연구 결과, MinkUNeXt-VINE++는 단일 센서 접근 방법에 비해 장소 인식 성능을 현격하게 개선하는 것을 보여줍니다.

- **Technical Details**: MinkUNeXt-VINE++는 이질 리다 데이터의 조기 융합 전략과 후보 장소의 최종 순위를 개선하기 위한 학습된 재순위화 접근법을 결합한 것입니다. 이러한 융합 전략은 각 센서의 강점을 활용하여 환경의 보다 포괄적인 표현을 제공합니다. 이 방법은 TEMPO-VINE 데이터셋을 사용하여 평가되었으며, 다양한 생리적 단계에서 포도원의 이질적 LiDAR 데이터를 제공합니다.

- **Performance Highlights**: MinkUNeXt-VINE++는 단일 센서 접근 방법에 비해 Recall@1 지표에서 각각 20% 개선을 달성하며, 재순위화를 포함할 경우 +30%의 추가 개선을 기록했습니다. 이러한 결과는 비구조적 환경에서 장소 인식의 정확성을 크게 높임을 보여줍니다. 또한, 연구는 넓은 범위의 농업 환경에서도 효과적으로 작동함을 입증했습니다.



### Budget-Constrained Step-Level Diffusion Caching (https://arxiv.org/abs/2606.13496)
Comments:
          Accepted by ICML 2026

- **What's New**: 이 논문에서는 BudCache라는 새로운 캐싱 접근 방식을 제안하여 고정된 계산 예산(compute budget)을 기반으로 최적의 캐시 정책(cache policy)을 탐색합니다. 기존의 방법들은 탐색 기반의 휴리스틱을 사용하여 캐시 결정을 내렸으나, BudCache는 최종 산출물의 질을 최대한 보존하는 방향으로 캐시 결정을 전환합니다. 또한, 이 방법은 온라인 탐색(overhead) 없이도 높은 품질의 캐시 정책을 식별할 수 있습니다.

- **Technical Details**: BudCache는 Simulated Annealing과 Hill Climbing을 결합하여 스텝 선택의 조합(combinatorial complexity) 문제를 해결합니다. 이 접근 방식은 몇 분 만에 고품질 캐시 정책을 식별하고, 캐시의 영향을 줄이기 위해 선택된 정책에 맞춰 시간 분할(time discretization)을 조정하는 캐시 인식 schedule alignment를 도입합니다. 이를 통해 인퍼런스 과정에서 발생할 수 있는 궤적 불일치(trajectory mismatch)를 효과적으로 줄일 수 있습니다.

- **Performance Highlights**: 실험 결과, BudCache는 FLUX.1-dev 및 Wan2.1 데이터셋에서 기존의 휴리스틱 캐싱 방법보다 더 나은 생성 품질을 달성하였습니다. 이 연구는 다양한 솔버, 해상도, 가이드 스케일, 적은 스텝 모델 및 대규모 비디오 모델에 걸쳐 추가적 결과를 제시합니다. 이로 인해 BudCache가 고정된 예산 내에서도 실질적으로 유용한 성능을 제공함을 입증하였습니다.



### Point-Wise Geometry-Aware Transformer for Partial-to-Full Point Cloud Registration in Computer-Assisted Surgery (https://arxiv.org/abs/2606.13488)
- **What's New**: 이번 연구에서는 부분에서 전체로(point cloud registration) 변환하는 문제를 해결하기 위해 GAPR-Net이라는 새로운 학습 기반 프레임워크를 제안합니다. 이 프레임워크는 컨볼루션(convolution) 및 트랜스포머(transformer) 모듈을 통합하여, 국소(local) 및 글로벌(global) 정보를 효율적으로 융합하는 코스-투-파인(coarse-to-fine) 아키텍처를 기반으로 합니다. 제안된 방법은 3D 포인트 클라우드의 높은 정확도를 제공하여, 컴퓨터 지원 수술(computer-assisted surgery)에서 정밀한 내비게이션 및 로봇 수술에 기여할 수 있습니다.

- **Technical Details**: GAPR-Net은 부분 및 전체 포인트 클라우드 간의 크로스 어텐션(cross-attention) 메커니즘을 활용하여 정보를 융합합니다. 새로운 멀티 스케일 지오메트리-인식 포지셔널 임베딩을 도입하여, 다양한 스케일에서 기하특성을 향상시킵니다. 이 방법은 포인트 간 기하학적 특성을 효과적으로 보존할 수 있어, 각 포인트의 상대적 기하 특성을 강력하게 캡처할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 네 가지 서로 다른 뼈(경골, 대퇴골, 골반 및 흉부 연골)에서 94.2%의 등록 회수와 1.992 mm의 낮은 RMSE(root mean square error)로 기존 방법보다 뛰어난 성능을 보였습니다. 이 연구는 특히 포인트 클라우드의 겹치는 비율이 낮거나 부분 클라우드가 전체 클라우드에 완전히 포함되지 않은 경우에도 효과적입니다. 따라서 GAPR-Net은 수술 내비게이션 및 치료 응용 분야에서 중요한 기초를 제공합니다.



### VISA: VLM-Guided Instance Semantic Auditing for 3D Occupancy World Models (https://arxiv.org/abs/2606.13460)
- **What's New**: 이 논문에서는 기존의 점유 세계 모델을 위한 새로운 접근 방식인 VISA를 제안합니다. VISA는 물체 인스턴스의 대표적인 이미지 크롭을 사용해 오프라인 비전-언어 모델(vision-language model, VLM)로부터 구조화된 감사를 생성합니다. 이러한 감사 결과를 3D 객체 복셀과 일치시키고, 안정성, 속성, 혼동 가능성 등과 같은 정보를 포함합니다.

- **Technical Details**: VISA는 감사를 통해 얻은 정보를 세 가지 손실 함수(신뢰도 기반 세분류, 속성 요소, 장면 수준 감사 그래프 손실)를 사용하여 기존 점유 모델의 시맨틱 로짓에 대한 트레이닝 타임 감독을 수행합니다. 기존 모델과의 차별점은, VISA는 새로운 점유 아키텍처를 도입하지 않고도 감사 정보를 활용하여 학습 시간 동안만 기능합니다. 이 과정에서, 입력 이미지와 기존 점유 세계 모델 아키텍처를 그대로 사용하면서 VLM을 호출하지 않습니다.

- **Performance Highlights**: 실험 결과, VISA는 nuScenes 데이터셋에서 OccWorld의 평균 mIoU를 19.06에서 20.05로, GaussianWorld는 21.36에서 21.91로 개선했습니다. 특히, OccWorld에서 물체 mIoU는 18.18에서 19.16으로, 희귀 클래스 mIoU는 15.60에서 16.79로 향상되었습니다. 이러한 결과는 VLM이 일반적인 캡션 임베딩 목표보다 신뢰성을 고려한 점유 감독으로 더욱 적합하다는 것을 시사합니다.



### OmniDirector: General Multi-Shot Camera Cloning without Cross-Paired Data (https://arxiv.org/abs/2606.13432)
Comments:
          12 pages, 8 figures

- **What's New**: 이번 연구에서는 비디오 생성에서 카메라 모션을 클로닝하는 과제를 다루며, 기존 방법의 문제점을 해결하기 위해 카메라를 그리드 모션 비디오로 인코딩하는 새로운 표현 방식을 제안합니다. OmniDirector라는 통합 프레임워크를 통해 사용자에게 감독 수준의 제어 기능을 제공하며, 다수의 카메라 그리드-비디오 쌍으로 학습합니다. 또한, 카메라 모션과 시각적 콘텐츠를 체계적으로 설명하는 새로운 계층적 프롬프트 확장 에이전트를 설계하여 다양한 제어 신호를 조화롭게 통합합니다.

- **Technical Details**: 성공적인 비디오 생성은 Diffusion Models의 발전에 크게 의존하고 있으며, 연구는 Text-to-Video, Image-to-Video, Video-to-Video의 세 가지 주요 범주로 분류됩니다. 새로운 카메라 그리드 개념을 통해 다수의 카메라 모션을 통합적으로 처리할 수 있는 unified representation을 제시합니다. 모델은 Multi-Modal Diffusion Transformers (MMDiTs)과 함께 작동하여 프롬프트 생성, 평면 내/간의 촬영 정보 및 카메라 모션을 통합하여 생성의 일관성을 보장합니다.

- **Performance Highlights**: 광범위한 실험을 통해 OmniDirector 프레임워크의 뛰어난 성능과 우수한 컨트롤 능력이 입증되었습니다. 특히, 우리는 대규모 카메라 그리드-비디오 데이터셋으로 모델을 학습하여 다양한 카메라 동작을 처리할 수 있습니다. 이 연구는 기존의 복잡한 카메라 모션 클로닝 문제를 해결하며, 별도의 크로스 페어링 데이터 없이도 일반적인 다중 촬영 클로닝을 가능하게 합니다.



### VietFashion: Benchmarking Sketch-Text Composed Image Retrieval for Cultural Outfits (https://arxiv.org/abs/2606.13427)
Comments:
          ICMR 2026. Project page: this https URL

- **What's New**: VietFashion은 아오자이라는 전통 베트남 의상을 중심으로 한 새로운 기준을 제시하며, 스케치-텍스트 조합 이미지 검색을 위한 데이터셋입니다. 기존의 이미지 검색 시스템이 전통 의상의 미세한 세부 정보를 잘 포착하지 못하는 문제를 해결하기 위해, 수작업 스케치와 텍스트 설명을 결합하여 문화적인 의미를 중시하는 의상을 검색할 수 있게 합니다. 이 데이터셋은 650개의 스케치로 시작하여, 생성 모델을 통해 21,000개 이상의 포토리얼리스틱 이미지로 확장되었습니다.

- **Technical Details**: VietFashion은 수작업으로 그려진 스케치와 패션 매거진에서 추출한 텍스트 설명을 결합하여 전통 의상의 구조적 및 문화적 의미를 포착합니다. 여기서 사용되는 생성 모델인 Qwen-2.5와 SANA-ControlNet은 고충실도의 패션 이미지를 합성하는 데에 활용되고, 다중 목표 검색 환경이 도입되어 다양한 유효 결과를 제공합니다. 이러한 접근 방식은 기존의 단일 목표 데이터 세트에서 발생할 수 있는 오류를 줄이고, 현실적인 디자인 프로세스를 반영합니다.

- **Performance Highlights**: 실험 결과, VietFashion 데이터셋은 문화적 의미 및 다중 모드 조합을 제대로 모델링하는 데 있어 성능 차이가 있음을 드러냈습니다. 특히, 현재의 모델이 세부적인 문화적 세분화, 예를 들어 특정 자수 패턴이나 칼라 스타일을 구별하는 데 한계를 보이고 있다는 점에서 매우 도전적인 벤치마크로 자리 잡고 있습니다. VietFashion은 이러한 문화 보존의 필요성을 해결하기 위해 AI 모델의 전반적인 이해도를 평가하는 새로운 기준을 제시합니다.



### Person Identification from Contextual Motion (https://arxiv.org/abs/2606.13410)
- **What's New**: 본 논문에서는 동작 스타일에 기반한 개인 식별 문제를 다룹니다. 우리는 감시 및 인증 응용 프로그램을 위한 두 가지 일반적인 개인 식별 시나리오에 대응하는 생성 모델과 확률적 신원 추론 방식(identification inference scheme)을 개발했습니다. 특히, 시각적 자극을 사용하여 개인의 반응을 기록하는 새로운 인터랙티브(interactive) 식별 시나리오를 도입했습니다. 이를 통해 식별 정확도가 크게 향상되었음을 보고합니다.

- **Technical Details**: 개인 식별 과정은 피험자와 시스템 간의 순차적인 메시지 교환(session) 맥락에서 정립되었습니다. 각 단계에서 시스템은 시각적 자극을 제공하고, 피험자는 이에 대한 동작 반응을 기록합니다. 기록된 반응은 가능한 피험자의 신원에 대한 사후 확률을 업데이트하는 데 사용되며, 일정한 신뢰 수준에 도달할 때까지 과정을 반복합니다. 이 논문에서는 또한 세 가지 일반적인 시나리오에서의 개인 식별 문제를 정식으로 정의하고, 이를 해결하기 위한 통합 생성 확률 프레임워크를 제시합니다.

- **Performance Highlights**: 우리는 4,476개의 녹화로 구성된 우리의 CuedId 데이터셋 및 MSRC-12 데이터셋을 사용하여 접근 방식의 성능을 평가했습니다. 실험 결과, 다양한 행동 유형의 조합이 사용될 때 개인 식별 정확도가 유의미하게 향상된다는 점을 확인했습니다. 이 연구는 이러한 인터랙티브 환경에서 개인을 성공적으로 식별할 수 있는 가능성을 보여주며, 향후 연구에 기여할 수 있는 데이터셋과 코드를 제공할 예정입니다.



### SmartFont: Dynamic Condition Allocation for Few-Shot Font Generation (https://arxiv.org/abs/2606.13382)
- **What's New**: 이번 연구에서는 SmartFont이라는 새로운 프레임워크를 제안하여, 전체적인 콘텐츠 스타일 생성과 약한 감 supervis될 수 있는 로컬 보정 전문가를 결합하여 몇 번의 샷(font) 생성 문제를 해결하고자 했습니다. 이를 통해 복잡한 이데오그래프 문자의 세밀한 디테일을 보존하면서도 전반적인 구조를 유지할 수 있는 개선된 방법을 제공합니다. SmartFont는 전통적으로 사용되던 접근 방식과 달리, 다양한 조건을 효율적으로 조절하여 정교한 보정을 가능하게 합니다.

- **Technical Details**: SmartFont는 전역적인 확산 모델(global diffusion model)을 기반으로 하여 콘텐츠 및 스타일 생성을 수행하고, 약한 supervision을 받는 로컬 전문가(branch)를 추가하여 세부적인 로컬 보정 큐를 제공합니다. 로컬 전문가 브랜치는 헝가리안 매칭(Hungarian matching) 기반의 컴포넌트 supervision을 사용하여 의미 있는 공간적 맵을 예측하고, 데이터의 시멘틱(spatial) 할당을 학습합니다.

- **Performance Highlights**: SmartFont는 기존의 방법들보다 글로벌 완전성과 로컬 충실도를 동시에 개선하여glyph 품질을 향상시킬 수 있음을 실험을 통해 입증했습니다. 이는 다양한 타겟 폰트 스타일에서도 성공적으로 작동하였으며, 더 나은 세밀한 디테일을 유지하면서도 전체 구조를 유지하는 데 있어 안정성을 보여줍니다.



### MoVerse: Real-Time Video World Modeling with Panoramic Gaussian Scaffold (https://arxiv.org/abs/2606.13376)
- **What's New**: MoVerse는 단일 협대역 시점 이미지에서 인터랙티브하게 탐색 가능한 장면을 생성하는 실시간 비디오 월드 모델입니다. 이 모델은 입력이 환경의 작은 일부만 관찰하는 도전적인 설정에서 작동합니다. MoVerse는 세계 구축(world construction)과 관찰 렌더링(observation rendering)을 분리하여 이 문제를 해결합니다.

- **Technical Details**: MoVerse는 먼저 입력을 중력 정렬된 360도 파노라마로 확장하여 필드 오브 뷰(field of view)의 공백을 메웁니다. 이후, 이 파노라마를 지속 가능한 3D 가우시안 스캐폴드로 변환하며, 이는 밀집하고 직접 렌더링 가능한 공간 기억(spatial memory)을 제공합니다. 마지막으로, 가우시안 조건의 비디오 렌더러가 사용자 지정 카메라 궤적을 따라 결과를 포토리얼리스틱 비디오로 변환합니다.

- **Performance Highlights**: MoVerse는 단일 NVIDIA RTX 4090 GPU에서 8 FPS로 실시간 장면 탐색을 지원합니다. 이는 단일 이미지를 기반으로 한 세계 생성에 있어 인터랙티브 비디오 출력을 위한 실용적인 경로를 제시합니다. 이 디자인은 3D 표현의 조작 가능성과 장기 일관성(long-range consistency)을 결합하여 생성적 비디오 모델의 인지 품질(perceptual quality)을 제공합니다.



### Dual-Constrained Diffusion Image Compression for Operational Rate-Distortion-Perception Optimization (https://arxiv.org/abs/2606.13366)
- **What's New**: 이번 연구에서 제안된 DCIC (Dual-Constrained Diffusion Image Compression) 방법은 고정 비율로 재구성의 품질을 최적화하기 위해 왜곡(distortion)과 아이데폰스(idempotence) 제약을 통합하여 이미지 압축을 개선합니다. 이는 기존의 데이터 전송 시 단순한 압축 성능 향상 뿐만 아니라 인지적 품질(perceptual quality)을 동시에 달성할 수 있는 방법론입니다. 특히, DCIC는 샘플간의 공통 임의성을 실현하여 추가적인 비율 부담 없이도 통합된 노이즈 주입을 활용합니다.

- **Technical Details**: DCIC의 동작 원리는, 재구성의 왜곡을 최소화하고 인지적 품질을 보장하는 두 가지 제약 조건의 결합에 기반 합니다. 이 접근법에서 왜곡 제약은 복원된 이미지와 기본 코덱 출력 간의 평균 제곱 오차(MSE)를 제한하기 위해 설정됩니다. 동시에, 아이데폰스 제약은 복구된 이미지를 재인코딩할 때 기본 코덱 출력을 복원해야 한다고 요구하는데, 이는 분포적인 인지 요구사항을 충족하는 데 필요한 보조적인 역할을 합니다.

- **Performance Highlights**: 실험 결과, DCIC~RDP는 CelebA-HQ, CLIC2020 및 ImageNet-1K 데이터셋에서 모든 인지 코덱에 비해 BD-PSNR 성능이 우수하게 나타났습니다. 또한, DCIC~RP는 전용 인지 지향 방법들과 BD-FID에서 동등한 성능을 기록했습니다. 이러한 성과는 DCIC의 전반적인 RDP 표면을 탐색할 수 있는 실제적 가치와 우수한 성능 일반성을 증명합니다.



### JointEdit3D: Feed-Forward 3D Scene Editing in a Unified Latent Spac (https://arxiv.org/abs/2606.13345)
Comments:
          Preprint. Project page: this https URL

- **What's New**: JointEdit3D는 3D 장면 편집을 위한 신개념 프레임워크로, RGB-기하학(latent space) 재구성 및 생성 과정의 통합을 통해 편집 결정을 최적화합니다. 기존의 방법들이 2D 뷰에 기반한 편집 신호를 사용하는 데 비해, 이 프레임워크는 3D 구조를 효과적으로 연결하여 보다 자연스러운 편집 결과를 생성합니다. 또한, 소스 장면의 구조를 그대로 보존하며, 스타일을 충족시키는 과정에서 단일 편집 참조 프레임만을 사용하여 복수의 RGB 뷰와 기하학(latent)을 동시에 생성합니다.

- **Technical Details**: JointEdit3D는 편집 및 기하학적 업데이트가 같은 잠재 상태(latent state)에서 이루어지도록 설계되었습니다. 이 프레임워크는 SceneAnchor Branch를 사용하여 편집을 위한 신호를 주입하고, 이를 통해 소스 RGB-기하학 구조를 유지하면서도 편집의 정확성을 확보합니다. 마지막으로, 편집 영역의 정확성과 함께 비편집 영역의 내용을 보존하기 위해 편집 인지 손실(edit-aware losses)을 도입하여 편집 전후의 품질을 균형 있게 유지할 수 있도록 합니다.

- **Performance Highlights**: JointEdit3D는 편집 지역의 품질과 3D 구조 완전성을 기존의 기준들보다 우수하게 개선하였습니다. 다양한 실험 결과에 따르면 소스 장면 구조를 유지하면서 편집된 콘텐츠의 충실도를 높이고, 배경 보존에 있어서도 경쟁력 있는 성능을 발휘합니다. 또한, 새롭게 소개된 SceneEdit3D-15K 데이터셋은 15K개의 쌍으로 이루어진 편집 샘플과 3D 주석이 제공되어 3D 장면 편집의 표준화된 평가를 위한 중요한 자원이 될 것입니다.



### Dual-Domain Equivariant Generative Adversarial Network for Multimodal CT-PET Synthesis (https://arxiv.org/abs/2606.13341)
Comments:
          4 pages, 3 figures, 1 table, 2026 IEEE 23rd International Symposium on Biomedical Imaging (ISBI)

- **What's New**: 본 논문에서는 다중 모드(dual-modal) CT-PET 이미지 합성을 위한 이중 영역 동등 생성적 적대 신경망( Dual-Domain Equivariant Generative Adversarial Network, DDE-GAN)을 제시합니다. 이 모델은 전통적인 GAN 접근 방식의 한계인 기하학적 일관성을 무시하고 구조적 충실도가 제한된 문제를 해결합니다. DDE-GAN은 공간(domain)과 주파수(Fourier) 영역을 동시에 학습하여 해부학적 및 스펙트럼 정보를 포착합니다.

- **Technical Details**: 이중 영역 동등 신경망(DDE-GAN)은 CT 및 PET 측정의 물리학에 내재된 회전 동등성을 손실 함수에 통합하며, 이를 통해 회전 시 신뢰할 수 있는 반응을 보장하여 해부학적 정확성을 향상시킵니다. 계층적 이중 영역 훈련 전략은 다단계 손실 함수에 의해 영역 내 및 영역 간 일관성을 강화합니다. DDE-GAN의 훈련 과정에서는 L1 손실 함수와 같은 다양한 비용 함수가 사용됩니다.

- **Performance Highlights**: HECKTOR 2022 CT-PET 데이터셋에서 평가된 결과, DDE-GAN은 CT-PET 이미지 합성에서 기존 모델보다 우수한 합성 품질을 달성하였습니다. 이 연구를 통해 이중 영역 학습과 기하학적 동등성을 결합하는 것이 다중 모드 이미지 합성 정확성과 견고성을 크게 향상시킬 수 있음을 입증하였으며, PET 완성 및 데이터 증강의 실제 애플리케이션 가능성을 열었습니다.



### OR-Action: Multi-Role Video Understanding with Fine-Grained Actions (https://arxiv.org/abs/2606.13332)
- **What's New**: 이번 논문은 외부 수술실(OR)의 활동을 면밀히 이해할 수 있는 방법을 제시합니다. 기존의 scene graph를 활용한 접근 방식이 한계가 있음을 지적하며, 새로운 fine-grained 행동 벤치마크를 소개합니다. 이 벤치마크는 공공 OR 데이터 세트에서 여러 역할의 행동을 정밀하게 분류할 수 있도록 도와줍니다.

- **Technical Details**: 연구에서는 EgoExOR 데이터 세트를 활용하여 scene graph에서의 프레임별 관계 예측을 dense temporal segments로 변환하는 방법을 제안합니다. 각 프레임은 subject, predicate, object 형태의 관계 삼중항으로 표현되며, 규칙 기반 매퍼를 활용하여 비디오 전체에 걸쳐 행동 레이블을 부여합니다. 이러한 규칙은 상태 규칙과 이벤트 규칙으로 구분되어, 지속적인 활동과 변화를 감지하는 역할을 합니다.

- **Performance Highlights**: 실험 결과, 기존의 scene graph 예측 방법이 시간적 구조를 모델링하는 데 어려움을 겪음을 발견했습니다. 하지만 새로운 시각적 모델은 그래프 기반 방법들보다 훨씬 우수한 성능을 보였습니다. 이를 통해, 전통적인 방법에 의존하지 않고도 높은 예측 성능을 달성했습니다.



### Masked and Predictive Self-Supervised Foundation Models for 3D Brain MRI (https://arxiv.org/abs/2606.13315)
- **What's New**: 이번 연구는 MRI 기반 질병 탐지에 대한 자가 감독 방식의 기초 모델(self-supervised foundation models)의 발전을 다룹니다. 특히, Masked Autoencoders (MAE)와 Joint Embedding Predictive Architectures (JEPA)라는 두 가지 주요 프리트레인(pretraining) 패러다임을 시스템적으로 조사했습니다. 연구진은 MAE를 위한 새로운 스펙트럼 도메인 재구성 손실(spectral-domain reconstruction loss)과 JEPA 프레임워크 내에 분산-공분산 정규화(variance–covariance regularization) 통합을 통해 보다 정교한 해부학적 구조에 대한 민감성을 향상시킴으로써, 이 모델들이 질병 탐지에서 어떻게 성능을 발휘하는지를 분석했습니다.

- **Technical Details**: 이 연구는 다양한 질병 탐지 작업을 위해 자가 감독 기초 모델의 베이스라인 및 보조 손실(augmented auxiliary loss)에 대한 비교를 포함합니다. MAE는 높은 주파수 해부학적 구조에 민감성을 촉진하기 위해 새로운 스펙트럼 도메인 손실을 도입하며, JEPA 프레임워크에서는 분산되고 decorrelated된 잠재 표현을 장려하기 위해 분산-공분산 정규화를 사용합니다. 본 연구의 프리트레인 데이터 세트는 7개의 이질적인 MRI 데이터 세트를 사용하였고, 단일 대조(input contrast)를 사용하여 프리트레인되었습니다.

- **Performance Highlights**: 연구 결과, 자가 감독 목표의 설계가 의료 모델의 프리트레인(pretraining) 성능에 중요한 영향을 미친다는 것이 드러났습니다. 특히 스펙트럼 정규화는 강한 고주파 해부학적 구조 신호의 경우 가장 큰 성능 향상을 보였고, 공분산 정규화는 여러 decorrelated 특성 차원에서의 분류 정보가 스팬(spanned)될 때 가장 유익했습니다. MAE는 MRI 기반 질병 탐지에서 일관되게 우수한 성능을 보여 주었으며, 자가 감독 목표가 특정 편향을 인코딩(encode)하고, 그 하부에서의 이익은 태스크의 구조에 따라 달라진다는 것을 시사합니다.



### MagPlus: Bridging Micro-to-Regular Facial Expressions through Learnable Magnification (https://arxiv.org/abs/2606.13312)
- **What's New**: 이번 연구에서 제안하는 MagPlus는 미세한 표정(micro-expression)을 표준 얼굴 애니메이션 모델과 연결하는 새로운 처리 파이프라인입니다. 이 방법은 고유적으로 기존 생성 네트워크를 대체하거나 수정하지 않고 미세한 표정을 일반 표정의 범위로 확대(magnify)하여 처리할 수 있도록 합니다. 인간의 진정한 감정을 반영하는 이 미세한 표정의 생성 과정에서 발생하는 데이터 부족 문제를 해결하기 위한 전환 과정도 포함되어 있습니다.

- **Technical Details**: MagPlus는 FlowMag에 기반한 학습된 모션 확대 모듈로, 미세한 표정의 미세한 동적 특성을 확대하여 일반적인 얼굴 표정의 운동 범위와 일치시킵니다. 막대한 데이터를 바탕으로 훈련된 기존의 주요 얼굴 애니메이션 모델과의 호환성을 확보하기 위해 인풋으로 제공되는 표정 데이터를 처리하고, DeMagPlus 모듈을 통해 증폭된 동작을 다시 현실적인 미세한 표정 강도로 복원합니다.

- **Performance Highlights**: 설문을 통해 실험된 FOMM, FSRT, MetaPortrait, EmoPortraits와 같은 네 가지 얼굴 애니메이션 모델에서 MagPlus-DeMagPlus를 평가한 결과, 기존의 상위 표정 모델들이 미세한 표정을 생성하는 과정에서 더욱 사실적인 동작을 구현할 수 있음을 보여주었습니다. 이는 새롭게 구성된 파이프라인이 얼굴 애니메이션 모델의 훈련 과정에 실질적인 변화를 가져오지 않으면서도, 높은 품질의 미세 동작 생성이 가능함을 나타냅니다.



### ReFree: Towards Realistic Co-Speech Video Generation via Reward-Free RL and Multilevel Speech Guidanc (https://arxiv.org/abs/2606.13304)
- **What's New**: 이번 연구는 ReFree-S2V라는 새로운 영상 생성 프레임워크를 제안하여 음성 기반의 사실적인 캐릭터 애니메이션을 구현한다. 이 방법은 사전 학습된(video generation) 모델을 기반으로 하며, 세밀한 음성 조음(speech articulation)과 고수준의 표현적 신호(expressive cues)를 결합하는 데 초점을 맞추고 있다. 기존의 방법들이 정확한 음소(phoneme)와 입술 동기화(lip synchronization)를 이루기 위한 도전과제를 해결하지 못했다면, ReFree-S2V는 이에 대한 통합적 접근 방법을 제공한다.

- **Technical Details**: ReFree-S2V 프레임워크는 음성의 음성학적(phonetic) 및 운율적(prosodic) 정보를 포착하는 다중 수준(multi-level) 음성 표현을 활용한다. 이 표현은 Transformer 블록에 학습 가능한 수준 선택기(level selectors)를 통해 주입되며, 이를 통해 정확한 입술 동기화와 자연스러운 표정 변화를 구현한다. 또한, 우리는 익숙한 수동 동기화 지표와 보상 모델 없이도 자연스러운 머리 움직임을 달성하기 위해 새로운 보상 없는 강화학습(reinforcement learning) 방안을 도입하였다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, ReFree-S2V는 정량적 음성 동기화(lip-sync accuracy) 및 질적 인간 평가에서 뛰어난 성능을 보여주며, 기존 방법들을 월등히 초월한다. 이 모델은 글로우 매칭(flow-matching) 훈련을 통해 개선된 사실감(perceptual realism)과 자연성을 제공하며, 합성된 캐릭터 애니메이션을 더 생동감 있게 만든다. 이를 통해 다양한 응용 프로그램에서 인간처럼 대화하는 캐릭터를 만들 수 있는 가능성을 제시한다.



### DuET: Dual Expert Trajectories for Diffusion Image Editing (https://arxiv.org/abs/2606.13303)
- **What's New**: 최근의 diffusion 기반 편집기는 소스 이미지를 기반으로 다양한 수정작업을 수행하지만, 이러한 소스 이미지의 지속적인 조정이 편집 품질을 제한할 수 있습니다. 본 연구에서는 DuET(Dual Expert Trajectories)를 소개하여 이미지 편집 중 소스 이미지 조정을 완화하고, 텍스트-투-이미지(text-to-image) 모드를 통해 자연스러운 결과를 향상시킵니다. DuET는 모델 가중치 수정 없이 다양한 모델과 벤치마크에서 일관되게 성과를 개선하였습니다.

- **Technical Details**: DuET는 훈련이 필요 없는 추론 프레임워크로, 단일 denoising 경로를 따라 편집 모드와 텍스트-투-이미지 모드 간 전환을 수행합니다. 이 시스템은 편집 모드와 T2I 모드를 혼합하여 일정 간격으로 전환하며, 이 과정에서 미세한 세부 사항을 보존하면서도 그림의 자연스러움과 의미의 신뢰성을 향상시킵니다. 실험을 통해 이 전환 일정이 편집 신뢰성과 소스 이미지 보존 간의 예측 가능한 균형을 어떻게 형성하는지 분석합니다.

- **Performance Highlights**: DuET는 여러 전환 방식에서 편집 신뢰성과 품질을 향상시키며, 기존 방식에 비해 주목할 만한 성능 개선을 보여줍니다. 특정 일정은 소스 이미지의 보존을 약간 줄이면서도 더 나은 편집의 충실도를 확보할 수 있습니다. 이 연구는 FLUX2-Klein과 다른 공개 벤치마크에서 성능을 검증하였으며, 여러 가지 전환 변형을 통해 결과의 질을 높였습니다.



### HYDRA-X: Native Unified Multimodal Models with Holistic Visual Tokenizers (https://arxiv.org/abs/2606.13289)
- **What's New**: HYDRA-X는 이미지와 비디오 토크나이징을 하나의 Vision Transformer (ViT) 내에서 통합한 첫 번째 unified multimodal model (UMM)입니다. 이 모델은 시각적 데이터를 통합된 표현 공간으로 매핑하는 데 중점을 두고 있으며, spatiotemporal reconstruction 가능성을 효율적으로 처리하고 이미지 및 비디오 수준의 의미적 인식을 Latent space에 통합하는 두 가지 핵심 문제를 해결합니다. 이러한 혁신으로, HYDRA-X는 여러 작업에서 강력한 성능을 보여주며 향후 통합 토크나이저 UMM의 개발에 기여할 전망입니다.

- **Technical Details**: HYDRA-X는 영상의 프레임 간 역학을 반영하기 위해 두 가지 주요 설계를 적용했습니다. 첫째, frame-level causal temporal attention을 사용하여 비디오 재구성이 이루어지며, 이는 기존의 full spatiotemporal attention보다 우수한 성능을 보입니다. 둘째, hierarchical temporal compression을 통해 여러 단계에서 시간 압축을 수행하여 더욱 효과적인 압축을 구현했습니다. 이러한 방법을 통해 HYDRA-X는 3D-conv 비디오 VAE보다 재구성의 충실도를 높였습니다.

- **Performance Highlights**: 7B 모델에서 실행된 HYDRA-X는 이미지 및 비디오 이해 및 생성 작업에서 뛰어난 성능을 달성했습니다. 특히, 이 모델은 이미지 및 비디오 편집에서 더 일관된 결과를 제공하고 수렴 속도를 크게 향상시키는 원칙적인 개선을 제안합니다. HYDRA-X는 이 시스템의 전반적인 구조를 강화하여 향후 연구에서 통합된 토크나이저를 탐색할 수 있는 튼튼한 기반을 마련합니다.



### Cross-Modal Masked Compositional Concept Modeling for Enhancing Visio-Linguistic Compositionality (https://arxiv.org/abs/2606.13288)
Comments:
          Accepted to ACL 2026 Main Conference, 25 pages

- **What's New**: 본 논문에서는 MACCO (MAsked Compositional Concept MOdeling)라는 새로운 프레임워크를 제안하여 기존의 Vision-Language 모델(VLMs)의 구성적 이해 능력을 향상시키고 있습니다. 이 방법은 하나의 모달리티에서 구성 개념을 마스킹하고 전체 컨텍스트 정보를 기반으로 재구성함으로써 cross-modal compositional 구조를 보다 효과적으로 캡처합니다. 이러한 접근 방식은 기존의 하드 네거티브 샘플에 대한 의존성을 줄일 수 있도록 설계되었습니다.

- **Technical Details**: MACCO는 두 개의 보조 목표인 Masked-augmented Cross-Modal Alignment Loss (MCA)와 Masked-augmented Intra-Modal Regularization Loss (MIR)를 도입하여 교차 모달 재구성과 정렬 학습을 촉진합니다. MCA는 마스킹된 텍스트 또는 이미지의 글로벌 특성을 교차 모달 대조 학습 과정에 통합하는 반면, MIR은 각 모달리티 내에서 마스킹된 인스턴스의 글로벌 특성을 정규화하여 표현의 붕괴를 방지합니다. 이러한 기법들은 모델이 구문 구조 및 언어적 정보를 더 잘 캡처할 수 있게 합니다.

- **Performance Highlights**: 다양한 구성 벤치마크에서 MACCO의 효과를 검증하였으며, 실험 결과 구성적 이해가 크게 향상됨을 보여주었습니다. 또한, 개선된 구성성은 텍스트-이미지 생성 및 다중 모달 대형 언어 모델에서도 이점을 제공합니다. MACCO는 기존의 하드 네거티브 마이닝 기법과 통합될 경우 추가적인 성능 개선을 이끌어낼 수 있습니다.



### Zero-Shot Captioning for Cultural Heritage: Automated Image Analysis of Traditional Indonesian Clothing (https://arxiv.org/abs/2606.13275)
Comments:
          accepted to ICME workshop on AIART 2026

- **What's New**: 본 논문에서는 Custom ZeroCLIP을 제안하여 인도네시아 전통 의상의 제로샷 캡셔닝(zero-shot captioning)을 실현하는 검색 보강 비전-언어 프레임워크를 소개합니다. 38개 인도네시아 주에서 전문가가 주석을 단 3,800개의 이미지로 구성된 데이터셋을 활용하여, 주 기반의 유도적 제로샷 접근을 수행합니다. 이 모델은 훈련, 검증, 평가에 있어 전혀 보지 못한 주의 이미지, 레이블, 캡션을 사용하지 않으며, 실험 결과는 현재의 기준을 초월한 성능을 보입니다.

- **Technical Details**: Custom ZeroCLIP는 동결된 CLIP ViT-B/32 이미지 인코더, CLIP 텍스트 인코더, BERT 텍스트 인코더 및 LSTM 캡션 디코더를 통합하여 구축됩니다. 훈련 단계에서 CLIP 매개변수는 수정되지 않고, 추론 단계에서는 검색 은행이 훈련된 주에서만 캡션을 포함합니다. 모든 인디비주얼은 주의 특성과 문화적 아래 맥락을 반영하여 캡션을 생성합니다.

- **Performance Highlights**: Custom ZeroCLIP은 CLIPScore 0.8536, BLEU-4 0.3342, METEOR 0.4859의 기록을 달성하며, 기존 기준보다 우수한 성과를 보입니다. 아블레이션 결과는 검색 방법이 문화 어휘 회복에서 19.3%의 METEOR 향상을 가져오며, 인간 평가 결과 또한 문화적 정확성과 유창성을 강화한 것으로 나타났습니다. 이러한 결과는 문화적으로 기반한 캡션 생성을 위한 검색 보강 도메인 적응의 효과성을 보여줍니다.



### TimeLens: On-Device Artifact Recognition with Retrieval-Augmented Question Answering for the Grand Egyptian Museum (https://arxiv.org/abs/2606.13267)
Comments:
          6 pages, 4 figures, 5 tables. Submitted to AIVRCH 2026

- **What's New**: 이번 연구에서는 TimeLens라는 인공지능 기반의 이중 언어 모바일 가이드를 소개합니다. 이 앱은 사용자가 전시물에 스마트폰을 겨냥하면 실시간으로 유물 인식이 이루어지고, 사용자는 영어 또는 아랍어로 후속 질문을 할 수 있습니다. 이 연구는 51개의 카탈로그화된 유물의 세부 유사성, 훈련 데이터와 핸드헬드 카메라 조건 간의 간극, 그리고 인공지능 가이드가 역사적 사실을 잘못 진술할 위험성 등의 세 가지 문제를 해결하고 있습니다.

- **Technical Details**: TimeLens의 핵심 기술적인 기여로는 첫 번째로, 데이터 품질 중심의 반복 연구를 통해 개발된 온디바이스 유물 탐지기가 있습니다. YOLOv8n 모델은 비디오 기반 데이터 세트를 통해 훈련되어, 평균 0.995의 최고 정확도로 실시간 인식이 가능하게 설계되었습니다. 두 번째로, 108개의 기록으로 구성된 ChromaDB 지식 기반을 기반으로 하는 이중 언어 RAG (Retrieval-Augmented Generation) 가이드가 개발되어, 지원하는 두 언어인 영어와 아랍어로 응답할 수 있도록 하였습니다.

- **Performance Highlights**: TimeLens는 실시간 온디바이스 인식을 통해 연령대가 다양한 사용자들에게 빠르고 신뢰할 수 있는 답변을 제공합니다. 연구 결과, YOLOv8n 모델이 모든 실패 클래스를 해결하며, 5.97MB의 TensorFlow Lite 자산으로 중급 스마트폰에서 실시간으로 작동 가능하다는 것을 보여줍니다. 또한, RAG 기반 가이드는 불필요한 허구의 답변을 줄이며 대기 시간을 30초 이상에서 약 10초로 단축시켰습니다.



### Visual Place Recognition in Forests with Depth-Aware Distillation (https://arxiv.org/abs/2606.13206)
Comments:
          IEEE ICRA Workshop on Field Robotics 2026

- **What's New**: 본 논문은 깊이 정보(Depth Information)를 활용하여 자연 환경에서의 시각장소인식(Visual Place Recognition) 성능을 향상시키는 경량(depth-aware distillation) 프레임워크를 제안합니다. 이 프레임워크는 DINOv2 기반의 장소 인식 모델에 기하학적 단서를 주입하여 사전 훈련된 기술 공간(descriptor space)을 유지합니다. WildCross 벤치마크에서 평가한 결과, 단독으로 시각적 정보만 사용하는 것보다 뛰어난 성능 향상을 보여주었습니다.

- **Technical Details**: 제안된 DAD(Depth-Aware Distillation) 프레임워크는 사전 훈련된 VPR 모델의 표시 구조를 유지하는 고정된 교사(teacher)와 깊이가 보강된 학생(student) 구조로 구성됩니다. 학습 과정에서, 학생 모형은 깊이 이미지를 통해 기하학적 단서를 사용하여 효과적으로 인식 성능을 향상시킵니다. 훈련은 쌍별 손실(triplet loss)과 정렬 손실(alignment loss)을 통해 진행되며, 이는 학습된 기술의 선택성을 높이고 기하학적 일관성을 유지하는 데 기여합니다.

- **Performance Highlights**: DAD는 WildCross 데이터 세트에서 깊이 정보가 자연 환경에서의 장소 인식을 보완하는 중요한 역할을 수행함을 입증하였습니다. 실험 결과, DAD 모델은 사전 훈련된 표시 기반 RGB 모델을 고려한 경우에도 더 높은 선택성이 있는 기술 공간을 생성했습니다. 이를 통해 깊이 정보가 장소 인식 성능을 높이는데 기여할 수 있음을 보여주었습니다.



### Transformer-Guided Graph Attention for Direct Cardiac Mesh Reconstruction: A Structural Digital Twin Framework (https://arxiv.org/abs/2606.13188)
- **What's New**: 이 연구는 심장 모형 생성의 효율성을 크게 향상시키기 위해 세분화(segmentation)와 메시(mesh) 생성 프로세스를 통합한 새로운 접근 방식을 제시합니다. 기존의 복잡하고 수동적인 작업 흐름에서 벗어나, 3D 의료 이미지를 직접적으로 매끄러운 심장 표면 메시로 변환하는 엔드-투-엔드 네트워크를 훈련시킵니다. 사용자는 Marching Cubes와 같은 전통적인 방법에 의존하지 않고도 빠르고 정확한 모델링이 가능해집니다.

- **Technical Details**: 이 접근법의 핵심은 3D Swin Transformer와 Graph Attention Network의 조합을 통해 원시 3D 이미지를 처리하고, 환자의 심장 경계에 맞추어 템플릿 메시를 반복적으로 변형하는 것입니다. 실험은 MM-WHS 2017 벤치마크에서 CT 및 MRI 이미지를 사용하여 수행되었습니다. 결과적으로 평균 Chamfer 거리 1.8 mm 및 95번째 백분위수 표면 거리 5 mm 이하의 품질 지표를 기록하여 기존의 세분화-메시 프로세스보다 우수한 성능을 보였습니다.

- **Performance Highlights**: 제안한 방법은 모든 메시가 단일 포워드 패스를 통해 생성되어, 수동적인 후처리 작업이 거의 필요 없다는 점에서 임상 사용을 위한 접근성을 높입니다. 또한, 세분화 마스크가 아닌 실제 시뮬레이션에 필요한 정밀한 메시를 제공함으로써, 임상 개입에 대한 예측적 통찰력을 제공합니다. 이로 인해 심장 디지털 트윈 파이프라인에서 지리적 충실도와 위상적 정확성이 픽셀 단위의 정확도보다 더 중요하다는 사실을 강조합니다.



### Iterative Visual Thinking: Teaching Vision-Language Models Spatial Self-Correction through Visual Feedback (https://arxiv.org/abs/2606.13156)
- **What's New**: 이번 연구에서는 Vision-language 모델(VLM)이 예측을 반복적으로 관찰하고 수정하는 메커니즘이 결여되어 있음을 발견했습니다. Iterative Visual Thinking(IVT)이라는 새로운 피드백 기반 프레임워크를 제안하여, 모델이 예측한 경계 상자를 렌더링하고 이를 바탕으로 결과를 정제하도록 합니다. 이 과정에서 VLM의 강력한 성능이 스스로 수정 가능성을 갖추지 못해, 단순한 반복적 접근만으로 막대한 성능 저하가 발생한다는 것을 보여줍니다.

- **Technical Details**: IVT는 모델이 예측한 내용을 시각적으로 확인하고 이를 바탕으로 반복적으로 수정하는 닫힌 루프(closed-loop) 프레임워크입니다. 연구진은 두 단계의 훈련 메커니즘인 SFT(Supervised Fine-Tuning)와 GRPO(Group Relative Policy Optimization)를 제안하였습니다. SFT는 교수 모델을 통해 초기 예측에서부터 교정된 추론 경로를 생성하여 학습 데이터를 생성하고, GRPO는 단순한 보상을 통해 다단계 수정 과정을 안정화시킵니다. 이 방법은 총 2,400개의 샘플로 단일 GPU에서 학습을 통해 이루어졌습니다.

- **Performance Highlights**: IVT 적용 후 모델의 성능이 모든 지표에서 단일샷(singe-shot) 기본 모델을 초월했습니다. Acc@0.5는 82.0%까지 상승하였고, Acc@0.7과 Acc@0.9도 각각 74.1%와 48.3%로 증가했습니다. GRPO는 각 단계에서의 IoU 감소를 5배 줄이며, 모델이 단계 사이에 예측을 유지하도록 훈련받았습니다. 이 연구는 VLM이 스스로 공간적 예측을 수정하는 것이 가능하다는 것을 보여주는 중요한 기초 연구입니다.



### An Extensible and Lightweight Unified Architecture for Demosaicing Pixel-bin Image Sensors (https://arxiv.org/abs/2606.13136)
- **What's New**: 이 논문에서는 다양한 pixel-bin 센서를 위한 모듈형 통합 아키텍처를 제안합니다. 이는 더 나은 이미지 품질을 제공하면서도 경량화되고 확장 가능하며, 플러그 앤 플레이가 가능한 CFA(CFA) 식별 모듈을 도입하여 원시 데이터의 CFA 유형을 정확히 감지합니다. 기존의 딥러닝 기반 demosaicing 방법들은 CFA별로 특정화되어 있어 리소스가 제한된 장치에 실제로 적용하기 어려운 한계가 있습니다.

- **Technical Details**: 제안된 아키텍처는 CFA 식별 모듈, DeMux 인코더 풀, 코어 및 디코더의 네 가지 모듈로 구성되어 있습니다. CFA 식별 모듈은 Fourier Transform과 Signature Extractor를 사용하여 원시 이미지의 CFA 유형을 효율적으로 식별합니다. DeMux 인코더 풀에서는 CFA 유형별로 인코더가 서로 다른 피쳐를 인코딩하게 되어, 중앙 코어에서 통합 처리된 후 디코더를 통해 최종 RGB 이미지로 재구성됩니다.

- **Performance Highlights**: 제안된 방법은 다양한 합성 데이터셋과 새로운 실제 센서 원시 데이터셋에서 평가되었습니다. 성능 벤치마킹을 통해 통합된 demosaicing 모델이 최신 소비자 하드웨어에서 높은 품질을 실현함을 입증하였습니다. 이러한 통합 아키텍처는 장기적으로 많은 CFA 유형에 쉽게 확장할 수 있는 가능성을 가지고 있습니다.



### Cascade Classification of Dermoscopic Images of Skin Neoplasms with Controllable Sensitivity and External Clinical Validation (https://arxiv.org/abs/2606.13135)
Comments:
          28 pages, 8 figures, 10 tables

- **What's New**: 이 연구는 피부 신생물에 대한 피부경 검사 이미지의 심층 학습 아키텍처와 분류 스킴을 비교하고, 개방형 국제 데이터 세트에서 러시아의 임상 데이터 세트로의 일반화 성능을 평가하는 것을 목적으로 한다. 네 가지 아키텍처(ViT-B/16, Swin-S, ConvNeXt-S, EfficientNetV2-S)와 세 가지 분류 스킴(이진, 단일 단계 네 클래스, 그리고 두 단계 캐스케이드)을 비교했다. 특히, 미세한 malignant lesions의 정확한 분류를 위해 조정 가능한 triage threshold를 도입하여 표준 단일 단계 분류에서 불가능한 민감도 조절을 가능하게 했다.

- **Technical Details**: 본 연구는 피부 경향성 신생물에 대한 이미지 분석을 수행하는 데 있어, 네 가지 현대적 딥러닝 아키텍처를 비교하고 있다. 이 연구의 한 가지 주요 포인트는 기존 단일 단계 멀티클래스 분류가 민감도 손실을 초래하는 경향이 있다는 것이다. 두 단계 캐스케이드 스킴은 진단의 임상 논리를 재현하며, 각 단계에서 다른 목표 함수를 통해 작업의 계층적 나누기를 구현하여 민감도를 조절할 수 있다. 모델들은 ImageNet으로 사전 학습된 가중치와 단일 증강 프로토콜을 사용하여 교육되었으며, 러시아의 임상 데이터를 포함한 두 개의 데이터 세트에서 평가되었다.

- **Performance Highlights**: 이 연구에서 이진 단계의 ROC-AUC 점수는 0.952에서 0.966 사이였으나, Sechenov 대학에서 0.797에서 0.893으로 떨어지고 민감도는 0.53에서 0.67로 감소하는 결과를 보였다. 캐스케이드 방식은 대부분의 아키텍처에서 단일 단계 네 클래스 분류보다 macro F1 점수를 개선시켰으나, 특히 ViT-B/16에서만 그 차이가 통계적으로 유의미하게 나타났다. 또한, ISIC MILK10k 데이터에서의 직접적인 11클래스 분류에서는 평균 클래스 민감도가 0.525로 나타났다.



### Fully Distributed Multi-View 3D Tracking in Real-Tim (https://arxiv.org/abs/2606.13127)
Comments:
          18 pages, 4 figures, 2 algorithms, 4 tables

- **What's New**: 본 연구에서는 MV3DT라는 완전 분산형 멀티-view 3D tracking 프레임워크를 제안합니다. 이 프레임워크는 중앙 집중식 집계 없이 peer-to-peer 상호작용을 통해 실시간으로 정확한 ID 전파와 occlusion 복구를 달성합니다. 각 카메라 노드는 경량 모듈 파이프라인을 실행하며, monocular 3D 감지, 분산 멀티-view 연관 및 협업 융합 과정을 포함합니다.

- **Technical Details**: MV3DT는 100대의 카메라에서 초당 30프레임을 유지하며 10ms 미만의 카메라 간 대기시간과 2.2%의 통신 오버헤드로 높은 성능을 보여줄 수 있습니다. 이 프레임워크는 카메라 교정이 이루어지면 Zero-shot 학습 방식으로 새로운 환경에 즉시 배포될 수 있습니다. 각 카메라 노드는 지역 3D tracking, peer-to-peer ID 전파 및 multi-view fusion을 수행하며 중앙 서버의 필요성을 제거합니다.

- **Performance Highlights**: MV3DT는 WILDTRACK에서 94.3%의 IDF1과 93.3%의 MOTA를 달성하며 기존의 중앙 집중식 방법들과 경쟁할 만한 성능을 자랑합니다. 실험 결과는 대규모 카메라 네트워크에서의 실시간 멀티-view tracking을 위한 실용적인 솔루션으로 MV3DT의 가능성을 보여줍니다. 다양한 벤치마크에서의 잘 측정된 통신 오버헤드와 함께 상태-of-the-art 정확도를 유지합니다.



### PP-OCRv6: From 1.5M to 34.5M Parameters, Surpassing Billion-Scale VLMs on OCR Tasks (https://arxiv.org/abs/2606.13108)
- **What's New**: 새로 발표된 PP-OCRv6는 경량화된 OCR 시스템으로, 메타포머 스타일의 건축 블록을 기반으로 다양한 구조적 혁신 및 데이터 중심 최적화를 결합합니다. 이 모델은 기존의 PP-OCR 시리즈의 구조적 한계를 극복하고, 효율성을 높이기 위해 공간 토큰 혼합과 채널 혼합을 분리하여 작업별 보폭 조정을 지원합니다. 서버에서 엣지까지 다양한 배치 시나리오를 커버하는 3개의 모델 계층(중간, 소형, 초소형)을 제공합니다.

- **Technical Details**: PP-OCRv6는 LCNetV4라는 통합 백본(백본 구조)을 바탕으로 두 가지 주요 개선점을 가지고 있습니다. 첫째, 메타포머 구조를 토대으로 한 토큰 혼합기와 채널 혼합기를 명확히 분리하여 각 작업을 최적화할 수 있는 유연성을 제공합니다. 둘째, 리파라미터화 구조를 사용하여 훈련 시간에 표현력을 높이면서도 추론 비용은 증가하지 않도록 설계되었습니다.

- **Performance Highlights**: PP-OCRv6_medium 모델은 83.2%의 인식 정확도와 86.2%의 탐지 H-mean을 달성하여 PP-OCRv5_server보다 각각 +5.1%와 +4.6% 향상된 성능을 보입니다. 또한, PP-OCRv6의 초소형 모델은 Intel Xeon CPU에서 PP-OCRv5_mobile 대비 3.9배 빠른 추론 속도를 유지하면서도 유사한 정확도를 작성할 수 있습니다. 이를 통해 경량화된 전문 OCR 시스템이 대형 모델 시대의 생산 OCR 배포에 적합하다는 강력한 증거를 제공합니다.



### Unified MRI Brain Image Translation via Hierarchical Tumor Structure Comparison (https://arxiv.org/abs/2606.13096)
- **What's New**: 이번 연구에서는 HTSCGAN이라는 새로운 이미지 변환 모델을 제안합니다. 이 모델은 종양 영역의 구조 정보를 통합하여 뇌 이미지 변환의 품질을 향상시키고자 합니다. 기존 방법들이 종양 영역의 구조 정보를 무시하는 문제를 해결하며, 의학적 진단의 정확성을 높이고자 하는 보건의료 분야에서의 필요성을 다루고 있습니다.

- **Technical Details**: HTSCGAN 모델은 세 가지 Patch Contrast Module (PCM)을 통해 다양한 패치 크기로 종양 영역의 계층적 구조 정보를 캡처합니다. 또한, 사전 훈련된 Patch Classifier (PC)와 Structure-Aware Encoder (SAE)를 활용하여 생성된 이미지가 실측 이미지와 동일한 종양 구조를 갖도록 합니다. 이러한 구조적 접근 방식은 이미지 변환 과정에서의 의뢰를 촉진합니다.

- **Performance Highlights**: 실험 결과, HTSCGAN은 BraTS2020 및 BraTS2021 데이터셋에서 강력한 성능을 보여주었습니다. 모델은 의학 이미지 변환 및 후속 세분화 작업에서 우수한 성과를 나타내어, 변환된 뇌 이미지의 품질과 임상적 중요성을 효과적으로 향상시킵니다. 전체적으로, 모델의 유용성과 효과성을 검증하는 결과를 도출하였습니다.



### LaME: Learning to Think in Latent Space for Multimodal Embedding via Information Bottleneck (https://arxiv.org/abs/2606.13061)
- **What's New**: 이번 연구에서 저자들은 LaME (Latent Reasoning Multimodal Embedding)를 제안하여 multimodal embedding에서 Chain-of-Thought (CoT) 추론의 두 가지 주요 한계를 극복하고자 합니다. LaME는 약하게 감독된 정보 병목 (information bottleneck) 접근법을 활용하여 CoT 주석의 질에 의존하지 않고도 효과적인 추론을 수행합니다. 이는 임베딩 최적화와 필수적인 CoT 과정 사이의 결합을 구조적으로 분리하여 안정적인 훈련을 보장합니다.

- **Technical Details**: LaME는 K개의 학습 가능한 reasoning token을 활용하여 모든 추론을 단일 전방 패스를 통해 완수하도록 설계되었습니다. 저자들은 두 개의 형태의 약한 감독 신호가 대조적 목적과 자기 회귀 (autoregressive) 목표를 구별함으로써 CoT 주석 의존성을 제거한다고 설명합니다. 추가로, 두 단계의 훈련 파이프라인을 통해 안정적인 수렴을 확보하였습니다.

- **Performance Highlights**: 실험 결과 LaME는 MMEB-v2 및 MRMR 데이터셋에서 경쟁력 있는 성능을 달성하였으며, 일부 명시적 CoT 기반 모델보다 뛰어난 결과를 보여주었습니다. 또한, LaME는 명시적 CoT 방법보다 60배 더 빠른 추론 속도를 제공하며, 반복적인 잠재 전개 (latent rollout) 기준보다 2배 빠른 성능을 자랑합니다.



### SeamEdit: A Black-Box VLM-Agnostic Pipeline for Large-Image Semantic Editing (https://arxiv.org/abs/2606.13041)
Comments:
          19 pages, 9 figures, 2 tables

- **What's New**: SeamEdit는 대규모 이미지의 의미론적 편집을 위한 새로운 프레임워크로, 비 훈련 기반(training-free) 및 모델 비의존성(model-agnostic)을 강조합니다. 기존의 모델에 의존하지 않고 모든 Vision-Language Model (VLM)을 블랙박스 오라클(black-box oracle)로 취급하여, 구체적인 실패 모드를 해결하기 위해 5단계의 후처리(post-hoc) 파이프라인을 제시합니다. 이 접근 방식은 시각적으로 눈에 띄는 경계(seam) 아티팩트를 줄이는 데 유용합니다.

- **Technical Details**: SeamEdit 파이프라인은 오버레이(overly) 기반의 타일 분할, 블랙박스 VLM 인페인팅(inpainting), 기하학적 및 색상 일관성(correcting geometric and color-consistency) 교정, 경계 위험 기반의 다중 후보 선택(multi-candidate ranking), 동적 프로그래밍(dynamic programming) 곡면 봉합(curved seam fusion) 을 포함합니다. 사용자는 대규모 이미지를 R x C의 정규 그리드로 나눈 후, 각 타일에 대해 VLM의 인페인팅 기능을 활용하여 후보 결과를 생성합니다. 이 과정에서, 경계(context)를 제공하고, 후속 교정 및 봉합 처리에 사용할 수 있는 메타 데이터를 기록합니다.

- **Performance Highlights**: SeamEdit은 타일 간의 경계 가시성을 줄이고 편집된 지역과 주변 내용 간의 자연스러운 연속성을 유지하며, 임의의 타일 지역의 의미론적 수정을 지원합니다. 기존의 방법들보다 강력한 생성 품질과 자연스러운 통합을 보장하고, 특히 블랙박스 VLM을 사용하는 대규모 데이터 편집에 있어 새로운 가능성을 제공합니다. 결과적으로, 이 프레임워크는 실질적인 응용 시나리오에서도 유용하게 활용될 수 있습니다.



### TetherCache: Stabilizing Autoregressive Long-Form Video Generation with Gated Recall and Trusted Alignmen (https://arxiv.org/abs/2606.13035)
Comments:
          17 pages, 8 figures

- **What's New**: 이 논문에서는 드리프트(Drift) 저항성을 갖춘 장기 비디오 생성을 위한 새로운 캐시 관리 전략인 TetherCache를 제안합니다. 기존의 오토리그레시브 비디오 확산 모델은 짧은 클립에 최적화되어 있으며, 긴 비디오 스트림을 생성하는 데 어려움을 겪었습니다. TetherCache는 훈련이 필요 없는 플러그 앤 플레이(plug-and-play) 방식으로, 캐시를 쉰크(Sink), 메모리(Memory), 최근(Recent) 영역으로 편성하여 효율적이고 신뢰성 높은 비디오 생성을 지원합니다.

- **Technical Details**: TetherCache는 두 가지 상호 보완적인 메커니즘을 포함합니다: GRAB(주목 다양성 균형을 통한 게이티드 리콜)과 TAME(신뢰할 수 있는 정렬을 위한 메모리 편집)입니다. GRAB은 장기 메모리 프레임을 선택하기 위해 게이티드 점수를 사용하며, 정보적이면서도 다양한 역사를 보존합니다. TAME은 새로 호출된 메모리 토큰을 수정하여 신뢰할 수 있는 컨텍스트 분포에 통계적으로 정렬함으로써 역사적 특징의 오염을 줄입니다.

- **Performance Highlights**: TetherCache는 VBench-Long 벤치마크에서 30초, 60초, 240초 환경 모두에서 비디오 생성 품질을 지속적으로 개선했습니다. 특히 240초 생성에서 전체 및 의미 점수가 유의미하게 향상되었으며, 품질 드리프트가 7.84에서 1.33으로 감소했습니다. 실험은 GRAB과 TAME 각각의 상호 보완적 기여를 보여 주며, 이 두 메커니즘이 함께 작업하여 안정적이고 고품질의 장기 생성을 가능하게 합니다.



### SAM-Deep-EIoU: Selective Mask Propagation for Multi-Object Tracking (https://arxiv.org/abs/2606.13033)
- **What's New**: 이번 연구는 다중 객체 추적 (Multi-object tracking, MOT)에서 성능을 향상시키기 위한 새로운 알고리즘인 선택적 마스크 전파 (Selective mask propagation)를 제안합니다. 기존의 기본 추적기를 사용하다가 특정 상황에서 비디오 객체 분할 (Video Object Segmentation, VOS) 모델로 전환하여 더욱 어려운 프레임에서의 추적 정확도를 높입니다. 이 방법은 훈련이 필요하지 않으며, 기본 추적기와 VOS 모델 모두를 블랙박스로 취급하여 다양한 모델로 교체 가능한 유연성을 가집니다.

- **Technical Details**: 선택적 마스크 전파는 기본 추적기가 모호한 상황에 처했을 때만 VOS 모델로 전환하여 마스크를 전파하는 방식입니다. 여기서 주요 신호는 헝가리안 비용 행렬 (Hungarian cost matrix)의 할당 마진으로, 신호가 발생할 경우에만 VOS 모델이 마스크를 전파하도록 설계되었습니다. 이 방식은 기본 추적기의 출력을 안전하게 보존하면서도, VOS 모델이 자신 있게 예측할 때만 출력을 수정하게 됩니다.

- **Performance Highlights**: 연구 결과, SportsMOT 기준에서 86.8 HOTA의 성능을 기록하여 최첨단 성과를 달성했습니다. DanceTrack 데이터셋에서도 세 가지 기본 추적기에 대해 일관된 성능 향상을 보여주었으며, 이러한 결과는 VOS 모델의 선택적 도입이 효과적임을 증명합니다. 이는 스포츠 애널리틱스에서 정체성을 유지하는 것이 중요하다는 점을 고려할 때, 매우 의미 있는 개선사항입니다.



### GeoCFNet: Geometry-Aware Confidence Field Network for Robot-Assisted Endoscopic Submucosal Dissection (https://arxiv.org/abs/2606.13032)
Comments:
          IEEE ICIA 2026

- **What's New**: 이 연구는 로봇 보조 내시경 점막 하 박리술(ESD)을 위한 새로운 기법인 GeoCFNet을 제안합니다. 이 방법은 복잡한 수술 절차에서 안정성과 정밀성을 향상시키기 위해 설계되었으며, 로봇이 안전하게 목표 영역을 정확히 식별할 수 있도록 도와줍니다. 기존의 시각적 안내 시스템이 가진 한계를 극복하기 위해 공간적 연속성과 기하학적 일관성을 보장하는 신뢰도 필드(confidence field) 추정 문제로 접근하였습니다.

- **Technical Details**: GeoCFNet은 사전학습된 DINOv3 백본을 기반으로 하는 기하학적 인식 신뢰도 필드 네트워크입니다. 이 네트워크는 Token-Differentiated Fusion 모듈을 통해 클래스 토큰과 밀집 패치 표현을 결합하며, SegFormer 디코더를 사용하여 신뢰도 회귀를 수행합니다. 또한 Geometry-Aware Spatial Regularization(GASR)을 통해 공간 일관성과 기하학적 전이를 보존하여 보다 안정적인 신뢰도 필드를 생성합니다.

- **Performance Highlights**: 실험 결과 GeoCFNet은 RMSE 0.0480, PSNR 27.1995, SSIM 0.3397, CC 0.2466의 성능을 달성하였으며, 시각적 왜곡에도 불구하고 지리적으로 안정적인 신뢰도 필드 추정을 구현하는 데 성공했습니다. 이러한 결과는 로봇 보조 ESD에서의 사용 가능성을 입증하며, 기존 기준선 기법들보다 뛰어난 성능을 보여주었습니다.



### A Multi-Modal Framework with Cross-Subject Pseudo-Labeling and Semantic Alignment for Micro-Gesture Recognition (https://arxiv.org/abs/2606.13030)
Comments:
          14 pages, 2 figures

- **What's New**: 이번 연구는 인간의 감정을 전달하는 미세 동작(micro-gestures, MGs)을 자동으로 인식하는 새로운 다중 모달 프레임워크를 제안합니다. 기존의 접근 방식은 RGB 및 스켈레톤 기반의 방법으로 나뉘었지만, 본 논문은 이 두 가지 모드를 효과적으로 통합하여 노이즈를 억제하고 특성의 상호 보완성을 이루어내고자 했습니다. 이를 통해 미세 동작 인식에서 발생하는 문제인 신호 대 잡음 비율의 저하와 긴 꼬리 분포를 극복하고, 교차 주제(domain shift) 문제를 해결하는 새로운 전략을 도입했습니다.

- **Technical Details**: 제안된 방법은 68개 관절 키포인트, 3D 열지도(heatmap), 고해상도 RGB 비주얼 특징을 포함하는 다중 모달 추출 파이프라인을 사용합니다. 이 과정에서 정밀한 신체 구조를 캡처하기 위해 Decoupled Spatial-Temporal CNN과 3D ResNet을 활용하고, 특수하게 설계된 정규화 기능과 교차 모달 의사 라벨링(Cross-Modal Pseudo-Labeling, CMPL) 전략을 통해 미지의 도메인에 대한 적응 능력을 극대화합니다. 이와 함께 온도 조정된 소프트 투표 메커니즘을 사용하여 모델의 과신(overconfidence)을 줄이도록 합니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 68.13%의 F1 점수를 기록하며 4위에 올랐습니다. 이는 기존의 방법들과 비교했을 때 경쟁력 있는 성능을 보여줍니다. 특히, 미세 동작 인식의 복잡성을 감안할 때, 다중 모달 통합 및 세밀한 처리 방법이 효과적으로 작용했음을 확인할 수 있었습니다.



### Quality-Preserving Imperceptible Adversarial Attack on Skeleton-based Human Action Recognition (https://arxiv.org/abs/2606.13022)
- **What's New**: 본 논문에서는 골격 기반 인간 동작 인식을 위한 적대적 공격에 관한 새로운 접근 방식을 제안합니다. 기존 방법들은 일반적으로 노이즈와 같은 섭동을 도입하여 동작 품질을 저하시켜 공격 후 인식률을 떨어뜨립니다. 그러나 본 연구의 주된 발견은 경험적 위험(empirical risk)과 실제 위험(true risk) 간의 갭(gap)에서 발생하는 품질 저하입니다. 우리는 동작 품질을 손상시키지 않으면서 적대적 동작을 생성하는 새로운 공격 방법을 도입하여 이 문제를 해결하고자 합니다.

- **Technical Details**: 제안된 방법은 분포 기반의 적대적 공격을 통해 적대적 동작을 얻으며, 데이터 분포를 학습하기 위해 확산 모델(diffusion model)을 사용합니다. 이 방법은 기존의 쌍을 비교하여 얻은 섭동 대신 데이터 분포에 의존하여 동작을 최적화합니다. 또한, 본 연구에서는 인간의 인식에 맞춘 새로운 품질 측정 지표를 제안하여 적대적 동작의 자연스러움을 평가하고, 이에 따라 동작의 품질 저하를 최소화합니다.

- **Performance Highlights**: 우리의 실험 결과는 네 가지 최신 S-HAR 분류기에서 제안된 방법이 공격 성공률과 공격 후 동작 품질 모두에서 우수성을 나타냄을 보여줍니다. 특히, 고품질 데이터셋과 널리 사용되는 두 개의 데이터셋에서 비교한 결과, 우리의 방법이 기존의 적대적 공격 기법보다 품질 저하가 가장 적음을 확인하였습니다. 추가적인 사용자 연구를 통해 우리의 적대적 동작이 인간에게 가장 인식하기 어려운 것으로 나타났습니다.



### A Machine Learning Framework for Real-Time Personalized Ergonomic Pose Analysis (https://arxiv.org/abs/2606.12988)
Comments:
          13 pages, 7 figures, conference 24CMH

- **What's New**: 이 논문에서는 3차원 볼륨 비디오 데이터를 활용하여 인체의 인체공학적(ergonomic) 및 비인체공학적(non-ergonomic) 자세를 실시간으로 예측하는 새로운 방법론을 소개합니다. 이 방법론은 인체공학 평가를 위해 설계되었지만, 인간 자세의 실시간 분석이 필요한 다른 응용 분야에도 적합하게 조정될 수 있습니다. 특히, 이 시스템은 평가 과정에서 3D 포인트 클라우드를 분석하는 능력으로 차별화되며, 여러 각도에서의 계산이 가능합니다.

- **Technical Details**: 시스템은 사용자가 선택하고 라벨링한 자세만을 개인화된 딥러닝 분류기(training classifier) 훈련에 사용하므로, 연속적으로 자동으로 자세 추정을 수행합니다. RGB-D 카메라를 이용한 사례 연구(case study)를 통해 하중을 들어올리는 작업을 수행하는 피험자들을 촬영하여 실시간 골격 라벨링(skeletal labeling)을 가능하게 하였습니다. 모델은 이 데이터를 기반으로 훈련되어 새로운 스트리밍 데이터에서 실시간으로 추론을 수행합니다.

- **Performance Highlights**: 이 연구는 최첨단 3D 데이터 기술과 전통적인 2D 자세 추정 알고리즘을 결합함으로써 실시간 인체공학 평가를 위한 확장 가능하고 실용적인 접근 방식을 제공합니다. 이는 작업 환경에서의 안전 및 건강 모니터링에 대한 증가하는 필요를 해결하는 것을 목표로 하며, 해당 분야에 중요한 기여를 하고 있습니다.



### Diffusion Transformer World-Action Model for AV Scene Prediction (https://arxiv.org/abs/2606.12987)
Comments:
          10 pages, 9 figures, 2 tables

- **What's New**: 이 논문은 자율주행 차량을 위한 action-conditioned world models를 제안하여, 차량의 계획된 제어로부터 향후 카메라 장면을 예측할 수 있게 하였습니다. 이 모델은 현실 세계에서의 시뮬레이션 없이 계획 및 시뮬레이션을 가능하게 하며, ambiguous한 미래 예측 문제를 다루고 있습니다.

- **Technical Details**: 제안된 모델은 compact latent world model로, 현재의 전면 카메라 latent와 일련의 ego-actions(조향각, 가속도)을 입력으로 하여 최대 8초까지 미래 장면의 latent를 예측합니다. VAE(Variational Autoencoder)를 freezer하여 예측된 latent를 256x256의 이미지로 변환하는 파이프라인을 구축하였습니다. 또한, V-JEPA2와 같은 frozen encoders의 성능을 종합적으로 비교하여 성과를 입증했습니다.

- **Performance Highlights**: 모델은 KID(Kernel Inception Distance)에서 0.078의 수치를 기록하며, 기존의 회귀 모델보다 4.8배 뛰어난 성과를 보여주었습니다. 이는 distortion metric을 최적화하기보다는 분포 기반 평가지표를 사용하여 성능을 평가해야 함을 시사합니다. 또한 action controllability가 평균 Spearman $ho=0.81$로 확인되어, 모델의 제어 가능성을 증명하였습니다.



### Objects Before Words: Object-First Inductive Biases for Grounding Language in Child-View Video (https://arxiv.org/abs/2606.12985)
- **What's New**: 이번 연구에서는 BabyMind라는 혁신적인 접근 방식을 제안하여 아기 시점(Video from child-view)의 언어 학습에서 발생하는 두 가지 모호성 문제를 해결합니다. 이는 Caregiver의 언급이 제한적이고 동기화가 불완전한 경우, 특정 객체의 위치를 정확하게 파악하는 데 도움을 줍니다. BabyMind는 기존의 Child-View Contrastive Learning(CVCL) 기술을 확장하여 객체 중심 학습을 통해 이러한 문제를 해결합니다.

- **Technical Details**: BabyMind는 객체 우선 접근 방식을 채택하여 두 가지 주요 구성 요소인 오프라인 자동 마스크 생성(AMG)과 다중 인스턴스 대조 목표를 통해 학습합니다. 짧은 시간 윈도우 동안 여러 프레임에서 후보 객체를 추출하고, 이를 객체 파일로 연결하며, 발화문을 이들 객체 파일에 정렬하여 대조 손실을 최소화합니다. 또한, 추적 일관성과 글로벌 객체 합의 정규자를 통해 객체 선택을 안정화합니다.

- **Performance Highlights**: SAYCam-S 데이터셋에서 BabyMind는 Labeled-S 15 강제 선택 정확도에서 CVCL보다 +2.6 포인트 향상된 결과를 보이며, 다양한 분야에서 일관된 성과를 발휘합니다. 특히, BabyMind는 희귀 개념에 대한 학습에서 더 나은 결과를 보여줍니다. 또한, 코드는 공개되어 연구자들이 활용할 수 있도록 돕고 있습니다.



### Camera and LiDAR BEV Fusion for Cooperative 3D Object Detection on TUMTraf V2X (https://arxiv.org/abs/2606.12981)
- **What's New**: 이 논문에서는 TUMTraf V2X 협력적 3D 물체 감지 트랙을 위해 개발된 카메라 및 LiDAR 융합 탐지기를 설명합니다. 이 탐지기는 세 개의 도로 측면 카메라와 차량 내 LiDAR의 융합된 점 구름을 공유된 조감도 공간에서 융합하여 여러 물체를 탐지합니다. 모델은 0.85의 3D mAP를 달성하며, 추가 연구를 통해 오버랩 된 데이터의 영향을 준수하여 성능을 향상시켰습니다.

- **Technical Details**: 탐지기는 도로측 LiDAR와 차량 내 LiDAR를 통합한 융합 포인트 클라우드와 세 개의 인프라 카메라 영상을 사용하여 설계되었습니다. 카메라의 이미지는 특성 맵으로 변환되며, LiDAR와 통합하여 160×160 크기의 BEV 맵을 생성합니다. CenterPoint 헤드를 통해 객체의 클래스와 위치를 예측하며, 118.7M개의 파라미터로 구성됩니다.

- **Performance Highlights**: 최초의 모델은 공용 Codabench 테스트 분할에서 0.85의 mAP를 기록했으며, 오버래핑된 프레임을 중복 샘플링한 결과 0.89로 향상되었습니다. 마지막으로, 실제 라벨을 사용하여 예측값을 대체했을 때 0.99 mAP를 달성하였고, 이 결과는 일부 추가 실험의 일환으로 제출되었습니다.



### Efficient, Robust, and Anti-Collusion Fingerprinting of Image Diffusion Models (https://arxiv.org/abs/2606.12977)
- **What's New**: 본 논문에서는 기존의 생성 모델 핑거프린팅 방법들의 약점을 밝히고, 특정 사용자 식별자를 생성된 이미지에 삽입하는 방법을 제안하고 있습니다. 특히, 여러 공격자가 자신의 모델을 결합하여 핑거프린트를 제거할 수 있는 공격에 대한 내성을 갖춘 핑거프린팅 방법을 처음으로 제안합니다. 새로운 개인화 정규화 모듈(PNM)을 통해 핑거프린트를 생성된 이미지에서 안정적으로 복원할 수 있도록 하는 방법을 개발하였습니다.

- **Technical Details**: 제안된 방법은 비트 문자열인 핑거프린트를 T2I 모델에 통합된 PNM의 계수에 인코딩합니다. 또한, 손실이 없는 Anti-Collusion Transformation(ACT)을 도입하여, 여러 모델의 합작 공격이 발생할 경우 이미지 생성 품질이 저하되도록 합니다. 이 방법은 프리-트레이닝된 T2I 모델의 변형 오토 인코더(VAE)에 PNM을 결합하여 개인화된 모델을 데이터 재학습 없이 구현할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 고해상도의 이미지 생성 및 편집 작업에서 99.5% 이상의 핑거프린트 추출 정확도를 달성했습니다. 특히, COCO 데이터셋에서 2인 공격의 경우 Fréchet Inception Distance(FID)가 23에서 79로 상승하여 모델의 사용성을 현저히 낮추는 효과를 보였습니다. 이 방법은 이미지 품질 저하 상황에서도 높은 핑거프린트 추출 정확도를 유지하여 기존 방법들보다 더욱 향상된 안전성을 보여주었습니다.



### YOLO-AMC: An Improved YOLO Architecture with Attention Mechanisms for Building Crack Detection (https://arxiv.org/abs/2606.12958)
Comments:
          14 pages, 8 tables, 6 figures. Expanded version of IET ICETA 2025 conference paper

- **What's New**: 이번 연구에서는 개선된 YOLO 기반 아키텍처인 YOLO-AMC(YOLO with Attention Mechanisms for Crack Detection)를 제안하여 자동 crack detection의 성능을 향상시킵니다. 기존 YOLOv11의 C2PSA 모듈을 제거하고, 다양한 attention 메커니즘(GAM, Res-CBAM, SA)을 Neck의 다중 스케일 기능 융합 층에 통합하여 크랙 관련 특성의 융합을 강화했습니다.

- **Technical Details**: YOLO-AMC는 YOLOv11 프레임워크를 기반으로 하며, attention 모듈을 통해 feature의 응답을 재조정하여 네트워크가 정보가 중요한 영역에 집중하고 배경 노이즈를 억제할 수 있도록 합니다. 실험 결과, 제안된 모델은 YOLOv11과 YOLOv8에 비해 다양한 평가 지표에서 일관되게 더 높은 성능을 보였으며, mAP@0.5 = 0.9917 및 mAP@0.5:0.95 = 0.9506의 최고의 검출 성능을 달성했습니다.

- **Performance Highlights**: YOLO-AMC는 NVIDIA RTX 4090 플랫폼에서 110.95 FPS의 속도를 기록하며 계산 복잡도를 7.6 GFLOPs로 유지합니다. 또한 Raspberry Pi 5 엣지 디바이스에서 약 5 FPS로 실행되는 것을 보여주어 정확성과 배치 효율 간의 상호작용이 우수함을 입증합니다. 이 연구의 구현 코드는 GitHub에서 이용 가능합니다.



### MAMVI: 3D Test-Time Adaptation via Masked Multi-View Point Clouds (https://arxiv.org/abs/2606.12939)
Comments:
          Accepted by ICPR 2026

- **What's New**: 이번 연구에서 제안하는 MAMVI(Masked Multi-View Test-Time Adaptation)는 3D 포인트 클라우드 분류에서 데이터 집합 간의 성능 저하 문제를 해결하기 위해 기존의 순차 최적화 방식 대신 통합된 단일 단계 적응을 활용하는 혁신적인 방법입니다. MAMVI는 안정성을 위한 고정 비율과 다양성을 위한 Beta 분포 샘플링 전략을 결합하여 다양한 관점을 생성합니다. 이를 통해, 기존 방법보다 4.9배에서 8.9배 빠른 추론 속도를 제공하면서도, 최신 정확도를 달성합니다.

- **Technical Details**: MAMVI는 포인트 클라우드 분류를 위한 소스 데이터를 필요로 하지 않는 테스트 시간 적응 프레임워크로, 다양한 비율의 마스킹을 통해 입력 데이터의 여러 관점을 생성합니다. 각 관점에서의 손실을 통합하여 단일 후방 전파를 통해 효과적으로 적응하며, 모델의 정규화 매개변수를 신뢰 기반 적응 학습률을 통해 업데이트합니다. 이 과정은 복잡한 신경망 아키텍처를 수정하거나 추가적인 학습 데이터가 필요 없도록 설계되었습니다.

- **Performance Highlights**: MAMVI는 ModelNet-40C, ShapeNet-C 및 ScanObjectNN-C와 같은 3D 포인트 클라우드 부패 벤치마크에서 광범위한 실험을 통해 경쟁력 있는 성능 향상을 보였습니다. 특히 ScanObjectNN-C 및 ShapeNet-C에서 최신 정확도를 기록하며, ModelNet-40C에서도 효과적인 결과를 유지하였습니다. 이러한 접근 방식으로 MAMVI는 실시간 애플리케이션에 매우 적합한 솔루션으로 자리 잡았습니다.



### Multi-Label Test-Time Adaptation with Bayesian Conditional Priors (https://arxiv.org/abs/2606.12925)
Comments:
          accepted by ICML2026

- **What's New**: 본 논문에서 제안하는 Bayesian Conditional Priors (BCP) 추정 방법은 멀티 레이블 인식에서 발생하는 도메인 변화 문제를 해결하기 위해 고안되었습니다. 기존의 방법들이 레이블 간의 의존성을 간과하여 부정확한 레이블 세트를 생성하는 반면, BCP는 고신뢰성 앵커 레이블을 선택하고 이를 기준으로 Bayesian 보정을 적용하여 레이블 간의 관계를 명확히 반영합니다. 이 방법은 파라미터 튜닝 없이도 작동하며, 실시간적으로 사용할 수 있는 장점이 있습니다.

- **Technical Details**: BCP는 제로샷 로짓을 마진 후방 신호로 보고, 매칭되지 않는 라벨 우선 순위로 인해 발생하는 오류를 정정하는 방식으로 작동합니다. 앵커 레이블을 기반으로 하는 경량 우선 순위 교정을 통해 호환되는 레이블 점수를 증가시키고 비호환 레이블의 점수를 감소시킵니다. 이는 로그 공간에서 닫힌 형태의 솔루션을 허용하며, 포인트별 상호정보량(PMI) 해석이 가능하게 합니다.

- **Performance Highlights**: BCP는 여러 표준 멀티 레이블 벤치마크에서 강력한 TTA(테스트 시간 적응) 기준선을 초과하는 성능을 보여줍니다. 예를 들어, RN50에서는 mAP가 57.31에서 69.22로, ViT-B/16에서는 62.61에서 71.79로 개선되었습니다. 이러한 성능은 특히 파라미터 튜닝 없이 멀티 레이블 인식의 신뢰도를 높이는 데 기여합니다.



### Magnifying What Matters: Attention-Guided Adaptive Rendering for Visual Text Comprehension (https://arxiv.org/abs/2606.12898)
- **What's New**: 이번 논문에서 소개한 Visual Text Comprehension (VTC)은 텍스트를 이미지로 변환하여 비전-언어 모델 (VLM)이 직접 읽을 수 있게 함으로써, LLM의 컨텍스트 윈도우 한계를 우회합니다. 기존의 VTC 파이프라인은 렌더링과 레이아웃을 고정된 전처리 단계로 취급하며, VLM이 시각화된 텍스트를 처리하는 방식에 대한 기계적 이해가 부족한 문제를 지적합니다. 이러한 문제를 해결하기 위한 새로운 메소드인 AGAR (Attention-Guided Adaptive Rendering)를 제안하였으며, 이는 텍스트의 중요 시각 패치(top-K important visual patches)를 식별하고 이를 확장하여 페이지를 재렌더링한 후, 정답을 재추론하는 방식입니다.

- **Technical Details**: AGAR는 훈련이 필요 없는 모델 독립적인 방식으로, VLM의 중간-후기 레이어에서의 주의(attention) 점수를 활용하여 중요 시각 패치를 식별하고 이를 단어 범위(word spans)로 매핑합니다. 이후 이러한 범위를 확대하여 재렌더링한 페이지에서 정답을 재추론하는 과정을 수행합니다. 연구 결과, AGAR는 VLM의 플러그 앤 플레이(plug-and-play) 형태로 정확도를 지속적으로 향상시킬 수 있으며, 여러 벤치마크(benchmarks)에서 강력한 성능을 보였습니다.

- **Performance Highlights**: AGAR는 총 9개의 VTC 벤치마크와 4개의 VLM 백본에서 실험을 통해, 훈련이 필요 없이 기존 VLM의 정확도를 향상시키는 성과를 보였습니다. 또한, AGAR는 사후 훈련(post-training)과 잘 결합되어 더욱 향상된 결과를 도출하고, 시각적 및 텍스트 입력의 저하에 대해 견고함을 유지하는 특징이 있습니다. 이러한 결과들은 AGAR가 VTC에서 증거 활용의 주요 병목 문제를 해결할 수 있는 중요한 방법이 될 수 있음을 시사합니다.



### Bridging Modal Isolation in Interleaved Thinking: Supervising Modality Transitions via Stepwise Reinforcemen (https://arxiv.org/abs/2606.12886)
Comments:
          22 pages, 5 figures, 6 tables

- **What's New**: 이 논문에서는 통합 다중 모드 모델(Uniﬁed Multimodal Models, UMMs)을 사용한 상호작용적 사고(interleaved thinking)가 공간 및 물리적 작업에 대해 promising한 결과를 보이고 있음을 설명합니다. 그러나 복잡한 장기 시나리오에서는 생성된 이미지가 텍스트 맥락과 다르게 나타나는 문제, 즉 Modal Isolation을 지적합니다. 이는 서로 다른 두 모드가 유기적으로 정보를 공유하지 않고 독립적으로 작동하는 문제로, 정보 손실이 모드 경계에서 발생하는 것에 기인합니다.

- **Technical Details**: 연구팀은 Modal Isolation을 구체화하고, 이를 해결하기 위해 두 단계의 훈련 프레임워크인 MoTiF(Modal Transition Fidelity)를 제안합니다. 이 프레임워크는 오류가 발생한 시각적 출력에서 회복할 수 있도록 훈련시키는 Reflective SFT와 강화 학습을 통해 이미지 생성의 충실도를 높이는 Flow-GRPO를 포함합니다. 이 연구는 각 모드 경계에서 발생하는 정보 손실을 양적으로 평가하고 이를 기반으로 훈련을 진행합니다.

- **Performance Highlights**: 네 가지 시각 퍼즐 벤치마크에서 MoTiF 방법론은 상호적 일관성 및 최종 작업 정확도를 크게 향상시켰습니다. 모드 경계에서의 명확한 구조적 감독이 장기 상호작용적 사고에 매우 중요하다는 점을 강조하며, 단순한 크기 조정이나 최종 작업 최적화만으로는 충분하지 않음을 보여줍니다. 이러한 결과들은 각 모드가 상호 의존적으로 작동해야 함을 입증하며, 이를 통해 모델의 장기적 사고 능력이 극대화된다는 것을 나타냅니다.



### Learning Task-Aware Sampling with Shared Saliency through Density-Equalizing Mappings (https://arxiv.org/abs/2606.12869)
Comments:
          16 pages, 10 figures

- **What's New**: 본 논문에서는 Density-Equalizing Convolutional Neural Network (DECNN)을 제안하며, 이는 데이터의 공간적 중요도에 따라 계산 자원을 동적으로 재분배하는 테스크 적응 샘플링 프레임워크를 기반으로 합니다. 이 네트워크는 특징 추출 시 정보가 높은 지역에 중점을 두어, 비효율적인 계산을 줄이고 모델 용량을 더 효과적으로 사용할 수 있도록 설계되었습니다. DECNN은 이미지 분류 및 크랑이얼 표면 분석에서도 뛰어난 성능을 보여주며, 파라미터 수를 줄이면서도 task-relevant 지역을 정확하게 식별합니다.

- **Technical Details**: DECNN은 밀도 평준화(convolutional) 매핑을 사용하여, 학습된 밀도 함수에 의해 유도된 샘플링 메커니즘을 통합합니다. 이 방법은 각 영역의 상대적 중요도를 인코딩한 밀도 함수에 따라 convolutional receptive fields를 비균일하게 재분배하며, 정보가 있는 지역에서는 더 촘촘한 샘플링을 가능하게 합니다. 이러한 과정을 통해, DECNN은 특성 중요도를 나타내는 재파라미터화된 도메인에서 더 효율적인 특징 추출을 수행합니다.

- **Performance Highlights**: DECNN은 이미지 분류 및 3D 얼굴 표면 분석 작업에서 다른 기존의 방법들과 비교하여 뛰어난 분류 정확도를 달성합니다. 특히, 학습된 중요도 맵은 해부학적으로 의미 있는 지역을 강조하여 정보가 중요한 지역에 대한 집중적인 샘플링을 가능하게 합니다. 이러한 성과는 제한된 파라미터 수로도 높은 성능을 유지할 수 있는 경량화 모델을 만들어냅니다.



### Language-Guided Abstraction for Visual Reasoning (https://arxiv.org/abs/2606.12847)
- **What's New**: 이번 논문에서는 L-VARC라는 새로운 프레임워크를 제안했습니다. 이 프레임워크는 언어 중심의 학습(Learning Using Privileged Information) 방법을 통해 시각적 추론을 강화합니다. L-VARC는 DeepSeek-V3를 사용해 원시 언어 설명을 정제하고 구조화하여 모델이 더 잘 일반화하도록 돕습니다. 또한, Cross-Attention Projector를 설계하여 시각적 특징과 의미적 임베딩을 정렬함으로써 ARC 모델의 훈련을 유도합니다.

- **Technical Details**: L-VARC 프레임워크는 시각적 표현과 언어 개념을 정렬하여 몇 가지 샷의 추상적 추론을 강화합니다. 이 프레임워크는 주요 시각적 백본(Main Backbone)과 언어 중심의 보조 훈련 브랜치를 포함합니다. Semantic Compression Module(SCM)을 통해 원시 LARC 설명을 간결한 규칙 중심의 문장으로 압축하고, Cross-Attention Projector(CAP)를 설계하여 시각적 특징과 의미 임베딩을 정렬합니다.

- **Performance Highlights**: L-VARC는 ARC-1 및 ARC-2에서 기존 VARC보다 향상된 PASS@1 성능을 보여주었습니다. 세부적인 성능 분석에 따르면, L-VARC는 76.7%에서 78.1%로 검증 정확도가 증가했습니다. 이러한 결과는 언어 중심의 훈련이 추상적 개념 학습에 맡은 역할을 강조하며, 새로운 과제에 대한 일반화를 강화하는 데 기여합니다.



### Perceive, Interact, Reason: Building Tool-Augmented Visual Agents for Spatial Reasoning (https://arxiv.org/abs/2606.12830)
- **What's New**: 이번 연구에서는 PERception-Interaction-reason Agent (PERIA)를 도입하여, 시각적 추론(task such as spatial reasoning) 작업을 위한 도구 증강된 비주얼 에이전트를 설계했습니다. PERIA는 비주얼 맵 추론(map reasoning), 시각적 탐색(visual probing), 비전 재구성(vision reconstruction) 등 여러 과제를 해결하는 기능을 제공합니다. 특히 두 가지 도구 세트를 통해 자주 사용하는 시각적 증거를 효과적으로 처리할 수 있습니다.

- **Technical Details**: PERIA는 비주얼 인지 도구(vision perception tools)와 비주얼 상호작용 도구(vision interaction tools)로 구성된 두 개의 상호 보완적인 도구 세트를 활용하여, 다양한 시각적 증거를 수집하고 이를 바탕으로 공간적 상태를 형성합니다. 또한, Observation-Relaxed Group-in-Group Policy Optimization (OR-GIGPO)을 통해 멀티 도구 사용 경로에 대한 정책 최적화를 수행합니다. 이 방식은 다단계 비주얼 도구 사용 시의 신뢰성 있는 결과를 보장합니다.

- **Performance Highlights**: PERIA-8B는 8개 데이터셋의 13개 벤치마크에서 이전의 최첨단 효과적 기준보다 7.0%-14.8% 향상된 성능을 보여줍니다. 이는 크기가 비슷한 이전 모델들과 비교하여도 효과적이며, Qwen3-VL-235B-A22B-Thinking 및 GPT-5와 같은 더 큰 모델과도 동등한 성능에 도달했습니다. 이러한 결과는 PERIA가 공간 추론 능력을 크게 향상시킨다는 것을 나타냅니다.



### DIMOS: Disentangling Instance-level Moving Object Segmentation (https://arxiv.org/abs/2606.12826)
- **What's New**: 이 논문에서는 Moving Instance Segmentation (MIS) 분야에서의 새로운 접근법을 제시하며, 이벤트 카메라와 이미지 기반 특징을 융합하여 성능을 향상시키는 방법을 탐구합니다. 기존의 MIS 방법들이 작은 움직이는 객체를 정확히 분리하는 데 한계를 보인 점을 해결하기 위해, 색상과 동작 정보를 분리하여 추출하는 이중 분리(feature disentanglement) 프레임워크를 처음으로 제안합니다. 이런 새로운 접근을 통해, 이미지와 이벤트 모달리티 간의 상호 작용을 개선하고 있습니다.

- **Technical Details**: 제안된 DIMOS 프레임워크는 이중 분리 인코더를 사용하며, 각 모달리티에서 색상과 동작 특징을 따로 추출합니다. 이를 통해 특징 밀도를 증가시키고, 이미지와 이벤트 간의 다중 해상도 크로스 모달 정렬(multi-granularity cross-modal alignment)을 적용하여 양질의 특징 융합을 통해 더 나은 성능을 달성합니다. 이 과정에서, intra-modal contrastive learning을 통해 색상과 동작 정보 간의 차별성을 높이고, 작업에 특화된 감독을 통해 분리된 특징을 학습합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법이 멀티모달 MIS 분야에서 최신 성능을 달성했음을 입증하였으며, 특히 빠른 움직임 및 저조도 환경에서 작은 인스턴스를 다루는 데 뛰어난 성능을 보였습니다. 기존의 방법들에 비해 더 높은 분할 정확도를 이루며, 다양한 환경에서의 안정성을 증명하였습니다.



### GRIP: Feedback-Guided Prompt Retrieval for Large Multimodal Models (https://arxiv.org/abs/2606.12744)
- **What's New**: 최근 In-Context Learning (ICL)은 LLM을 새로운 작업에 쉽게 적용할 수 있는 강력한 메커니즘으로 자리잡았습니다. 이 개념은 멀티모달 모델에도 확장되어, Multimodal In-Context Learning (M-ICL)로 발전하게 되었습니다. M-ICL은 이미지, 캡션 및 질문-답변 쌍과 같은 관련 예제를 검색하여 분류, 캡셔닝 및 VQA 작업에 대한 예측을 안내합니다.

- **Technical Details**: 기존 방법론은 주로 피처 공간 유사성에 기반하여 인-context 예제를 선택합니다. 그러나 본 연구에서는 이러한 가정이 항상 성립하지 않음을 체계적으로 분석하였으며, 시각적으로 유사한 예제가 인-context 학습 성능을 가장 효과적으로 향상시키지 않는 경우도 있음을 밝혔습니다. 이에 따라 Guided Retrieval of In-context Prompts (GRIP)를 제안하며, 이는 LMM의 피드백을 활용하여 실제 모델 예측을 개선하는 예제를 식별하는 방식으로 작동합니다.

- **Performance Highlights**: GRIP는 Qwen2.5-VL-7B에서 유사성 기반 검색보다 일관되게 성능을 향상시키며, 특히 Idefics2-8B에서 분류 작업에서 가장 큰 성과를 보였습니다. 또한, 하나의 오픈 LMM으로 훈련된 검색기가 재훈련 없이 다른 모델로 전이될 수 있어 M-ICL의 확장성과 비용 효율적인 배포를 가능하게 합니다.



### VLADriveBench: Evaluating CoT-Action Relationship in VLA for Autonomous Driving (https://arxiv.org/abs/2606.12706)
- **What's New**: 이 논문에서는 기존의 주행 궤적만 평가하는 관행을 넘어 Chain-of-Thought (CoT)의 일관성과 관련성을 평가하는 새로운 벤치마크인 VLADriveBench를 소개합니다. 이 프레임워크는 CoT와 주행 행동 간의 관계를 관찰하는 지표(mentioning, hallucination, contradiction, action alignment)와 CoT 개입 프로토콜을 결합하여 보완적인 관점을 제공합니다. VLADriveBench를 사용하여 세 가지 모델을 평가한 결과, 각 모델에서 CoT와 행동 간의 관계가 크게 다름을 발견했습니다.

- **Technical Details**: VLADriveBench는 두 가지 차원에서 CoT를 평가합니다: 품질(세계에 대한 CoT의 정확성)과 관계(CoT가 주행 행동과 연결되어 있는지). 품질은 언급량, 정확도, 환각, 모순 지표를 통해 평가하며, 관계 차원은 관찰적 정렬과 인과 개입을 통해 측정됩니다. 이러한 접근식은 관찰적 지표가 CoT의 품질과 주행 행동과의 상관관계를 밝히지만, 어떻게 영향을 미치는지는 나타내지 않으며, 개입은 인과성을 입증하지만 민감도가 제한적입니다.

- **Performance Highlights**: ORION 모델은 관찰적 정렬에서 가장 높은 점수를 기록했지만, 그것의 CoT는 의미상 인과관계가 없습니다. 반면에 Alpamayo v1.5 모델은 정렬 점수가 낮지만 CoT가 강한 인과성을 보여주며, CoT의 영향력 정도는 시각적 중요성에 의해 조절됩니다. 이 두 모델의 성능 차이는 관찰적 메트릭과 개입이 결합되어야만 CoT가 주행 행동에 미치는 영향을 이해할 수 있음을 증명합니다.



### SalArt-VQA: Diagnosing Whether VLMs Understand Salient Artifacts in Generated Images (https://arxiv.org/abs/2606.12671)
Comments:
          23 pages, 7 figures, 7 tables. Dataset: this https URL

- **What's New**: 최근 Vision-language models (VLMs) 이 AI가 생성한 이미지 내 가시적인 결함(artifacts)을 탐지하는 데 사용되고 있지만, 이러한 결함을 분석하는 능력은 여전히 잘 알려져 있지 않습니다. 본 논문에서는 SalArt-VQA라는 진단 벤치마크를 도입하여 AI 생성 이미지의 세밀한 결함 이해를 평가합니다. 이 벤치마크는 950개의 이미지와 3,681개의 인간이 작성한 객관식 질문들로 구성되어 있으며, 그에 대한 다양한 평가를 수행합니다.

- **Technical Details**: SalArt-VQA는 결함 이미지, 실제 이미지 및 생성된 참조 이미지를 포함합니다. 이 벤치마크는 존재 탐지(presence detection), 의미적 위치 지정(semantic localization), 공간적 기반(spatial grounding), 및 증거 기반 결함 식별(evidence-grounded defect identification)과 같은 네 가지 질문 유형을 통해 VLM의 성능을 평가합니다. 또한, 주석이 없는 결함에 대한 조정을 테스트하기 위해 참조 분할(reference splits)을 사용합니다.

- **Performance Highlights**: 20개의 VLM을 대상으로 한 SalArt-VQA의 결과는 이미지 수준 탐지 정확도가 숨기는 결함들을 드러냅니다. 가장 강력한 모델은 결함 이미지에 대해 99.37%의 탐지 재현율을 달성했지만, 결함에 관한 네 가지 질문을 모두 정확하게 답한 이미지는 53.26%에 불과했습니다. 이러한 결과는 결함 탐지의 높은 정확도가 반드시 지역적 시각 증거에 의해 뒷받침된 결함 이해를 의미하지 않음을 보여줍니다.



### CD-RCM: Generalizable Continuous-Depth Novel View Synthesis for Reflectance Confocal Microscopy (https://arxiv.org/abs/2606.12635)
- **What's New**: 이번 연구에서는 RCM(Reflectance Confocal Microscopy)을 활용하여 비침습적으로 피부를 'optical biopsy'할 수 있는 새로운 방법인 CD-RCM(Continuous-Depth Novel-View Synthesis for RCM)을 소개합니다. CD-RCM은 희소한 RCM 스택으로부터 연속 깊이를 예측하여 3D 볼륨의 해석을 개선하는 것을 목표로 합니다. 이 접근은 기존의 전통적인 모델과 달리, 한 번의 인퍼런스에서 해부학적으로 일관된 중간 슬라이스를 예측하여 환자별 최적화 없이도 적용할 수 있습니다.

- **Technical Details**: CD-RCM은 RCM 스택에서 깊이 변화를 쿼리 가능하게 모델링하여, 단일의 전방 통과 경량 모델을 기반으로 합니다. 이 모델은 전통적인 영상 해석 방식과는 달리 제안된 심층 구조에서 내재된 광학 물리학을 이용하여 스택 데이터를 처리합니다. 또, 기계학습 기반의 perceptual loss를 도입하여 미세 세포 구조를 보존하면서 비자연 이미지에서 훈련된 특성 추출기를 통해 정확도를 높였습니다.

- **Performance Highlights**: 실험 결과, CD-RCM은 다른 전통적 보간 방법과 비교했을 때 고충실도의 새로운 깊이 합성을 단일 인퍼런스 패스에서 달성했습니다. 이는 RCM 스택을 시각화할 때 어려운 얕은 섹션과 깊은 섹션의 정보를 효과적으로 통합함으로써 가능했습니다. 또한 CD-RCM은 주변 구조의 세밀한 이해를 가능하게 하여 임상적인 진단과 연구에 유용한 도구로 자리잡을 것으로 보입니다.



### ECA: Efficient Continual Alignment for Open-Ended Image-to-Text Generation (https://arxiv.org/abs/2606.12633)
Comments:
          Accepted at the 43rd International Conference on Machine Learning (ICML 2026)

- **What's New**: 이 논문에서는 Open-ended Image-to-Text Generation (OpenITG)을 위한 Incremental Learning (IL)에 관한 새로운 접근 방식인 Efficient Continual Alignment (ECA)를 제안합니다. ECA는 기존의 이미지 카테고리 변동을 고려하여 지속적인 적응을 통해 Cross-modal representations의 품질을 유지하고, 새로운 태스크-specific features를 효과적으로 학습합니다. 또한, 기존의 raw data 저장이 없이도 내용의 변화를 처리할 수 있는 모델로, 메모리 및 개인정보 보호 문제를 해결하고자 합니다.

- **Technical Details**: ECA는 세 가지 핵심 메커니즘으로 구성됩니다: 첫째, Mixture of Query (MoQ) 모듈은 태스크-specific query tokens를 학습하여 새로운 단서를 최소한의 방해로 확보합니다. 둘째, Dictionary Replay (DR) 모듈은 task-agnostic 시각적 요소를 담은 컴팩트한 embedding dictionary를 유지하여 이전의 raw exemplars 없이도 효과적으로 재생합니다. 셋째, Fisher Dynamic Expansion (FeDEx) 모듈은 Fisher Information Matrix (FIM)를 기반으로 하여 간섭이 감지될 때만 파라미터를 선택적으로 확장함으로써, 이전의 설정을 보존하며 새로운 태스크에도 적응할 수 있도록 합니다.

- **Performance Highlights**: ECA는 새로운 IL OpenITG 벤치마크에서 기존 방법들보다 우수한 성능을 보여줍니다. 실험 결과는 ECA가 catastrophic forgetting을 크게 완화하고, OpenITG 작업에서의 IL 성능을 향상시켰음을 입증합니다. 또한, 이 연구는 OpenITG의 실제 시나리오와 유사한 태스크 정의를 통해 문제를 해결하려는 시도를 강조하고 있습니다.



### Context-Aware Feature-Fusion for Co-occurring Object Detection in Autonomous Driving (https://arxiv.org/abs/2606.12628)
Comments:
          8 pages, 3 figures, CVPR 2026 Precognition Workshop

- **What's New**: 이번 연구에서는 Context-Centric Feature Fusion (CCFF)라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 Local Context Fusion Module (LCFM)과 Global Context Attention Module (GCAM)이라는 두 가지 주의(attention) 기반 모듈을 포함하고 있습니다. LCFM은 RoI-to-RoI 자기 주의 메커니즘을 사용하여 주변 객체 간의 상호작용을 해결하며, GCAM은 RoI 특성의 top-K 집합을 활용하여 객체의 공존 정보를 학습합니다.

- **Technical Details**: CCFF 프레임워크는 Detectron2 백본과 Feature Pyramid Network (FPN)를 사용하여 다중 스케일 특성 맵을 추출합니다. RoI는 RoIAlign을 통해 고정된 차원으로 매핑됩니다. LCFM은 RoI 간 상호작용을 모델링하고, GCAM은 RoI의 글로벌 컨텍스트를 통해 공존 우선 정보를 인코딩하여 최종적으로 원래 RoI의 외관 특성과 콘텍스트로 향상된 피처를 통합합니다.

- **Performance Highlights**: 실험은 Cityscapes와 BDD100K 데이터셋에서 수행되었으며, 카테고리 수준의 일관성 전략(0.973 및 0.969)을 달성하여 성능 개선을 입증합니다. 특히, 작은 객체 탐지에서 14.1%의 향상을 보였고, 공존하는 객체 탐지에서도 유의미한 개선을 이루었습니다. 또한, 프레임워크는 이미지를 실시간으로 처리할 수 있으며, 0.2 FPS의 오버헤드를 기록합니다.



### Dual-State Slot Attention: Decoupling Appearance and Identity for Video Object-Centric Learning (https://arxiv.org/abs/2606.12601)
- **What's New**: 본 논문에서는 Dual-State Slot Attention (DSSA) 라는 새로운 자가 지도 학습 프레임워크를 제안합니다. DSSA는 객체의 외관과 정체성을 분리하여 안정적인 객체 추적을 가능하게 하며, 특히 빠른 움직임이나 부분 가림 현상에서 더 나은 성능을 보입니다. 이는 기존의 슬롯 기반 방법들이 안고 있던 한계를 극복하는 데 중점을 두고 있습니다.

- **Technical Details**: DSSA는 각 슬롯을 두 개의 전용 표현으로 구성하여 로컬 상태(local state)와 정체성 상태(identity state)를 제공합니다. 로컬 상태는 프레임에 특정한 외관을 포착하고, 정체성 상태는 학습된 순환 전환(recurrent transition)을 통해 안정적인 객체 정보를 누적합니다. 추가적으로, 경쟁 조정 집합(competition-modulated aggregation, CMA)을 도입하여 약한 슬롯의 업데이트를 줄여 타 객체로부터의 토큰 흡수를 방지합니다.

- **Performance Highlights**: DSSA는 MOVi-C, MOVi-D 및 YouTube-VIS 데이터셋에서 이전 방법들에 비해 뛰어난 분할 품질과 시간적 일관성을 제공합니다. 예를 들어, MOVi-D에서는 FG-ARI 점수에서 3.7 포인트 향상된 성능을 보였고, YouTube-VIS에서는 9.7 포인트 향상되었습니다. 또한 DSSA는 다운스트림 객체 인식과 동영상 동역학 예측의 성능에서도 우수함을 입증하였습니다.



### Analyzing and Improving Fine-grained Preference Optimization in Medical LVLMs (https://arxiv.org/abs/2606.12590)
- **What's New**: 이 논문은 대형 비전-언어 모델(LVLM)이 의료 이미징 작업에서 겪는 사실 불일치, 시각적 기반 부족, 임상적으로 중요한 피드백과의 불일치를 해결하기 위한 새로운 방법을 제안합니다. 기존의 연결 프레임워크의 한계점을 극복하기 위해, 연구자들은 토큰 수준에서 세밀한 최적화를 허용하는 새로운 'Fine-grained Regularized Medical Preference Optimization (FiRe-MPO)' 목표를 도입하였습니다.

- **Technical Details**: FiRe-MPO는 역방향 토큰별 KL 정규화를 활용하여 임상적으로 중요한 요소, 즉 해부학적 발견 및 병리적 특성을 정교하게 조정하도록 설계되었습니다. 이 방법은 모델의 원래 언어 구조를 유지하면서 임상적으로 잘못된 부분만 수정하는 '온 정책(preference pairs)' 데이터 세트를 생성하며, 시각적 근거를 요구하는 임상적 정확성을 뛰어넘는 최적화를 촉진합니다.

- **Performance Highlights**: 다양한 의료 이미징 과제 및 임상 텍스트 생성 벤치마크에서의 실험 결과, FiRe-MPO는 기존의 DPO와 RRPO보다 일관되게 우수한 성능을 보여주며, 평균 10.24%의 상대적 개선을 달성하였습니다. 이러한 결과는 임상적인 정확성을 높이고, 일반적으로 관찰되는 스타일에 편향되는 보상 해킹 행동을 효과적으로 감소시킴을 보여줍니다.



### High-Fidelity Two-Step Image Generation via Teacher-Aligned End-to-End Distillation (https://arxiv.org/abs/2606.12575)
- **What's New**: Z-Image Turbo++는 8단계 Z-Image Turbo 모델에서 증류된 고품질 2단계 이미지 생성 모델입니다. 이 모델은 2단계 생성의 기본적인 병목 현상인 작업 난이도의 증가와 모델 용량의 제한을 극복하기 위해 세 가지 간단하면서도 효과적인 설계 선택을 도입하였습니다. 이는 2단계와 8단계 생성 간의 품질 차이를 크게 줄이는 데 기여합니다.

- **Technical Details**: Z-Image Turbo++의 주요 기술 설계에는 세 개의 핵심 요소가 포함됩니다. 첫째, Distribution-Aligned Adversarial Learning을 통해 GAN 훈련을 위해 외부의 실제 이미지 대신 교사 모델이 생성한 이미지를 사용하여 더 구체적이고 정보가 풍부한 적대적 목표를 제공합니다. 둘째, Step-Decoupled Parameterization을 도입하여 두 개의 디노이징 단계에 서로 독립적인 모델 매개변수를 할당합니다. 셋째, End-to-End Training with Iterative Regularization을 통해 그래디언트를 최적화하면서도 의미 있는 중간 생성을 보존합니다.

- **Performance Highlights**: 이 모델은 질적 및 양적 평가 모두에서 2단계와 8단계 생성 간의 품질 차이를 상당히 좁히며, 교사의 시각적 품질과 벤치마크 성능을 대부분 유지합니다. Z-Image Turbo++는 2단계 생성의 효율성을 극대화하면서도 훈련 과정에서 안정성과 최종 품질을 개선했습니다. 이는 현대적인 증류 전략이 적은 단계에서의 품질-효율성 균형을 개선하는 데 효과적이라는 점을 강조합니다.



### HairPort: In-context 3D-aware Hair Import and Transfer for Images (https://arxiv.org/abs/2606.12562)
Comments:
          Accepted to SIGGRAPH 2026 (Conference Papers Track). 23 pages, 15 figures, 10 tables, including supplementary material as appendices. Project page: this https URL

- **What's New**: 본 논문에서는 HairPort라는 3D 인식을 통한 헤어스타일 전송 프레임워크를 제안합니다. 기존의 방법들이 작은 자세 차이에서 최적화되었던 것과 달리, HairPort는 큰 자세와 스케일 차이를 처리할 수 있도록 설계되었습니다. 또한, Bald Converter라는 balding 단계가 도입되어 얼굴의 사실적인 대머리 버전이 생성되며, 새로운 데이터셋인 Baldy를 통해 6,000개의 쌍의 이미지가 제공됩니다.

- **Technical Details**: HairPort는 헤어 전송을 위한 세 가지 주요 컴포넌트로 구성됩니다: Bald Converter, 3D-Aware Hair Transfer, Flow-Matching Hair Synthesis입니다. HairPort는 참조 헤드를 3D로 재구성한 후 소스 포즈에 맞춰 재배치하여, 헤어스타일이 자연스럽게 맞춰지도록 합니다. 기존의 2D 기술에서 나타나는 왜곡을 방지하며, 사실적인 헤어 전송을 위한 재구성을 지원합니다.

- **Performance Highlights**: HairPort는 기존 방법들과 비교하여 정량적 및 정성적으로 우수한 성능을 보이며, 포즈 일관성과 신원 보존을 제공합니다. 제안된 방법은 다양한 주제와 헤어스타일을 대상으로 평가되었으며, 사용자의 피드백과 자동 평가 지표를 통해 그 효과가 입증되었습니다. 이러한 혁신적인 접근 방식은 가상 시도 시스템과 엔터테인먼트 분야에서의 응용 가능성을 크게 확장합니다.



### Stereo Vision-Based Fall Prediction and Detection using Human Pose Estimation on the AMD Kria K26 SOM (https://arxiv.org/abs/2606.12473)
Comments:
          19 pages; 31 figures

- **What's New**: 이 연구는 AMD Kria K26 시스템 모듈을 활용하여 휴대 가능하고 저전력인 비전 기반 낙상 예측 및 탐지 시스템을 제안합니다. 이 시스템은 실시간 낙상 탐지를 위한 비침입적이고 개인정보 보호를 위한 구성으로 설계되었습니다. Intel RealSense D455 카메라를 이용하여 깊이 프레임과 RGB 프레임을 동기화하여 캡처하고, YOLOX, A2J 모델을 통해 낙상의 감지 및 예측을 수행합니다.

- **Technical Details**: 이 시스템은 K26 SOM과 USB로 연결된 Intel RealSense D455 카메라에 의해 RGB 및 깊이 프레임을 60 FPS로 캡처하며, YOLOX를 이용해 인간의 경계 상자를 식별한 후 RGB 프레임을 버리는 방식으로 개인정보를 보호합니다. A2J는 각 개인의 15개의 관절 키포인트를 추정하고, CNN은 선택된 관절 좌표(x, y, z)를 사용하여 낙상 활동을 분류합니다. 성능 평가는 정확도 및 IoU를 통해 수행되었으며, 처리량은 단일 스레드에서 다중 스레드로 향상되었습니다.

- **Performance Highlights**: 정량적 정확도는 YOLOX가 74%, A2J가 84.13%, CNN이 75.85%로 나타났습니다. 다중 스레드를 활용한 성능 개선을 통해 처리량이 2.5 FPS에서 4.5 FPS로 증가했습니다. 연구 결과는 AMD Kria K26 엣지 장치에서 개인정보를 보호하면서도 실시간 낙상 탐지의 가능성을 보여주며, 클라우드 의존성을 줄이고 노인 모니터링 및 보조 의료를 지원하는 방향으로 진행되었습니다.



### Mana: Dexterous Manipulation of Articulated Tools (https://arxiv.org/abs/2606.13677)
Comments:
          Project Page: this https URL

- **What's New**: Mana (Manipulation Animator)는 기존의 단단한 물체 중심의 조작에서 벗어나, 복잡한 구조를 가진 가공 도구 조작을 애니메이션 문제로 재구성한 새로운 프레임워크입니다. 이 시스템은 단순한 사용자 입력을 통해 수천 개의 조작 키프레임을 자동으로 생성하고, 이를 모션 플래닝(motion planning) 및 강화 학습(reinforcement learning)을 통해 조작 궤적으로 변환합니다. 이 접근법은 공구의 물리적 한계를 초과하는 시스템에서 비슷한 성능을 출력할 수 있도록 합니다.

- **Technical Details**: Mana는 조작 키프레임을 생성하는 과정과, 이러한 키프레임을 연결하는 궤적 생성을 통해 조작의 복잡성을 경감합니다. 먼저, 사용자는 툴 메쉬(tool mesh)의 기능적 영역을 간단히 클릭하여 정의하여 바람직한 키프레임을 생성합니다. 이후, 모션 플래닝과 강화 학습의 두 가지 기술을 함께 사용하여 조작 모션을 최적화하고 가벼운 도구 조작을 가능하게 합니다.

- **Performance Highlights**: Mana는 여러 형태와 크기의 조작 도구에 대해 제로샷(Zero-shot) 시뮬레이션-현실 전이를 성공적으로 수행하였으며, 각 도구에서 높은 성공률을 보여줍니다. 분석한 도구는 집게(tongs), 플라이어(pliers), 주사기(syringes), 클립(clothespins) 등 다양한 조작 메커니즘을 포함하며, 이는 모두 서로 다른 정밀도가 요구되는 기계적 문제를 나타냅니다. 이 결과는 Mana의 확장 가능성과 조작의 정밀도를 입증합니다.



### SPARC: Reliable Spatial Annotations from Robot Demonstrations at Sca (https://arxiv.org/abs/2606.13497)
- **What's New**: 이 연구에서는 SPARC(Spatial Annotations from Robot Demonstrations with Reliability Calibration)라는 리스크 인식 프레임워크를 도입하여 로봇 시연에 구조화된 공간 주석을 자동으로 라벨링하고, 각 주석에 신뢰성 점수를 부여합니다. 기존의 자동화된 파이프라인은 샘플의 유용성을 버리거나 노이즈가 많은 레이블을 수용해야 하는 불편함이 있었으나, SPARC는 로봇 작업의 시공간 구조를 활용하여 신뢰성 신호를 생성하고 노이즈가 많은 레이블을 줄이며 유용한 샘플을 더 많이 보존합니다.

- **Technical Details**: SPARC는 각 객체 중심의 하위 작업에 대해 상호작용된 객체를 식별하고 시작 및 목표 위치, 시간에 따른 움직임을 3D 객체 궤적으로 생성하며 지속적인 신뢰성 점수를 부여합니다. 기존의 탐지기 신뢰성을 객체 정체성을 위한 단일 출처로 사용하는 대신, SPARC는 물리적 단서(특히 3D 그리퍼 근접도와 정렬된 움직임)를 토대로 상호작용 증거를 통해 Composite Annotation Reliability Score를 도입합니다. 이를 통해 사용자들은 인간 리뷰 없이 주석 품질과 데이터 세트 규모 간의 명시적인 트레이드오프를 할 수 있게 됩니다.

- **Performance Highlights**: SPARC는 1.7k개의 인간 주석 데이터에서의 상호작용 객체 위치 파악에 대해 80.2%의 정확도를 기록하며, 탐지기만 사용하는 기준 모델보다 높은 성능을 보여줍니다. 이러한 주석을 사용하여 생성된 고품질 VQA 데이터 세트는 511k 샘플을 포함하며, 이는 기존의 인간 주석 데이터를 기반으로 한 모델보다 우수한 성능을 나타냅니다. 또한 SPARC 주석으로 학습된 정책은 복잡한 현실 세계의 로봇 조작 과제에서도 기준 모델을 초과하는 성능을 보였습니다.



### NavWAM: A Navigation World Action Model for Goal-Conditioned Visual Navigation (https://arxiv.org/abs/2606.13494)
Comments:
          Project page: this https URL

- **What's New**: 이번 논문에서는 로봇 비전 내비게이션을 위한 새로운 접근 방식을 제안합니다. 네비게이션 월드 액션 모델(NavWAM)은 확산 변환기(diffusion-transformer) 정책을 통해 비전 예측을 실행 가능한 액션으로 변환합니다. 이 모델은 미래 관측치, 목표 진척도, 그리고 액션 조각을 공유된 잠재 시퀀스에서 표현함으로써 시각적 예측을 로봇 제어에 직접 활용할 수 있도록 합니다.

- **Technical Details**: NavWAM은 시뮬레이션 사전 학습과 실제 로봇 적응을 통해 구축됩니다. 기본적으로 NWMs(네비게이션 월드 모델)의 예측을 사용하며, 이론적으로는 기존의 외부 계획자가 필요 없기 때문에 독립적인 제어가 가능합니다. 이 모델은 향후 관측치와 목표 진척도, 실행 가능한 액션을 통합하여 하나의 정책 표현으로 제공합니다.

- **Performance Highlights**: NavWAM은 이미지 목표 내비게이션에서 계획 기반 월드 모델 및 대표적인 직접 내비게이션 정책에 비해 향상된 성능을 보였습니다. CEM 스타일의 테스트 시간 궤적 최적화 없이도 성과를 거두며, 더 큰 직접 정책 기반 모델과 경쟁력 있는 결과를 달성했습니다. 전반적으로 NavWAM은 시각적 예측과 가치 예측을 포함하여 로봇 내비게이션의 효율성을 높이는 기여를 하고 있습니다.



### Reinforcement Learning for Neural Model Editing (https://arxiv.org/abs/2606.13461)
- **What's New**: 이 논문에서는 사전 학습된 신경망을 수정하기 위한 탐색적 프레임워크를 제안합니다. 이 프레임워크는 신경 모델 편집을 강화 학습 문제로 수립하여 에이전트가 보상 피드백을 사용하여 모델을 수정합니다. MaskWorld와 ShiftWorld라는 두 개의 환경을 도입하여 에이전트가 가중치를 곱셈 및 덧셈으로 조정할 수 있도록 합니다.

- **Technical Details**: MaskWorld와 ShiftWorld에서 에이전트는 신경망의 가중치를 조정하고 각 행동 후에는 수정의 질을 반영하는 보상을 받습니다. 보상 함수는 수정된 모델의 유용성과 편집 목표를 결합하여 설계되었습니다. 특히, MaskWorld에서는 가중치를 0과 1 사이의 상수로 조정하고, ShiftWorld는 가중치에 상수를 더하거나 빼는 방식을 밟습니다.

- **Performance Highlights**: 이 프레임워크를 텍스트 분류의 편향 완화 및 이미지 분류의 잊기 유도에 적용한 결과, 잊기 세트 정확도가 0%에 가깝게 감소하는 동시에 보존 세트의 정확도는 90% 이상을 유지합니다. 편향 완화에서 5% 이상의 성과를 향상시켰으며, 전반적인 모델 유틸리티를 유지하는 동시에 원하는 편집 목표를 간편하게 달성했습니다.



### IterCAD: An Iterative Multimodal Agent for Visually-Grounded CAD Generation and Editing (https://arxiv.org/abs/2606.13368)
- **What's New**: 이 논문에서는 IterCAD라는 새로운 다중 모드(agent) 프레임워크를 제안하여 폐쇄 루프 기반의 상호 작용 CAD 생성 및 편집을 지원합니다. 기존의 자동화된 방법들이 주로 오픈 루프 방식과 일회성 생성에 의존하는 반면, IterCAD는 다중 인터랙션을 통해 CAD 작업을 단순화 합니다. 또한, 공장 기준에 부합하는 다각면 공학 도면을 생성하고, 복잡한 코드 편집 과제를 처리하는 새로운 데이터 합성 파이프라인을 개발하였습니다.

- **Technical Details**: IterCAD는 Multi-turn 상호작용을 통해 CAD 모델링을 진행하며, 이를 위해 "Look and Loop" 철학을 채택하였습니다. 첫 번째 단계는 고품질의 멀티 턴 상호작용 궤적을 바탕으로 한 감독 세밀 조정(Progressive SFT)입니다. 이후, 기하학 기반 강화 학습(geometry-aware reinforcement learning)을 통해 코드 실행 가능성과 기하학적 충실도를 높이도록 설계되었습니다.

- **Performance Highlights**: IterCAD-Bench 평가 프레임워크에 대한 설계를 포함하며, Chamfer Distance Tolerance-Recall (CD-TR) 곡선과 Area Under the CD-TR Curve (AUC-TR) 지표를 통해 코드 유효성과 기하학적 정밀도를 통합적으로 측정할 수 있습니다. 대규모 실험 결과, IterCAD는 코드 실행 가능성과 기하학적 정밀도 모두에서 기존 접근법보다 훨씬 뛰어난 성능을 보이며, 폐쇄 루프적 반복 개선에서 우수한 능력을 발휘합니다.



### VideoMDM: Towards 3D Human Motion Generation From 2D Supervision (https://arxiv.org/abs/2606.13364)
Comments:
this https URL

- **What's New**: 본 논문에서는 VideoMDM이라는 확산 기반( diffusion-based) 프레임워크를 도입하여, 3D 사람 움직임 프라이어를 훈련하는 방법을 제시합니다. 이는 모노큘러 비디오에서 추출한 정확한 2D 포즈를 사용하며, 3D 그라운드 트루스(ground truth)가 필요하지 않습니다. 훈련 과정에서 2D 포즈 감독만을 활용하여 3D 모션을 일관되게 학습하게 됩니다.

- **Technical Details**: VideoMDM은 사전 훈련된 2D-3D 리프터(2D-to-3D lifter)를 사용하여 2D 입력으로부터 근사적인 3D 포즈 시퀀스를 생성하고, 이를 모델이 3D에서 노이즈를 제거하는 방식으로 훈련합니다. 템포랄 일관성을 확보하기 위해 깊이 가중치(depth-weighted)를 고려한 2D 재투영 손실(reprojection loss)을 도입하였으며, 이는 3D 정상화 기법으로 강화됩니다. 따라서 2D로부터의 학습이 안정적이며 가치 있는 3D 모션을 생성하게 됩니다.

- **Performance Highlights**: VideoMDM은 HumanML3D에서 거의 3D 감독(3D-supervised) MDM과의 격차를 해소하였으며, FID 수치에서 0.88을 기록했습니다. Fit3D 및 NBA와 같은 실제 비디오 데이터셋에서 생성된 모션은 인간으로부터 선호를 받았으며, 정량적으로도 우수한 성과를 보였습니다. 이 결과들은 2D 감독 만으로도 3D 움직임 프라이어를 효과적으로 학습할 수 있음을 나타냅니다.



### Towards More General Control of Diffusion Models Using Jeffrey Guidanc (https://arxiv.org/abs/2606.13240)
- **What's New**: 이 논문은 Jeffrey guidance라는 새로운 프레임워크를 제안하여 diffusion 모델의 출력을 보다 유연하게 제어할 수 있는 방법을 제공합니다. 기존의 조건부 샘플링을 넘어서, 제안된 방법은 명시적인 목표 분포를 설정하고 이를 유지하면서도 marginal distributions를 업데이트할 수 있습니다. 이 접근법은 전통적인 guidance 방법에 비해 더욱 복잡한 요구사항을 충족할 수 있는 가능성을 열어줍니다.

- **Technical Details**: Jeffrey의 조건화 규칙은 모델의 확률 분포를 업데이트하는 데 유용한 일반적인 방법으로, 다양한 조건을 만족할 수 있습니다. 새로운 Jeffrey guidance는 기존의 조건부 기반 접근 방식과 차별화되어, 주어진 목표 분포에 맞춰 모델 출력을 조정합니다. 이 과정에서, 표준 분류기 기반의 guidance를 포함하여 다양한 conditional 구조를 보존하면서 joint distribution에 최소한의 변화를 주는 방식으로 동작합니다.

- **Performance Highlights**: Jeffrey guidance를 적용한 결과, Inception embeddings를 목표로 삼았을 때, CIFAR-10과 FFHQ 데이터셋에서 FID를 상당히 감소시켰습니다. CelebA-HQ에 대한 공정성 문제에서도, output에서 Male 및 Young 속성이 decorrelation되도록 하는데 성공하여, 보다 공정한 모델을 개발할 수 있는 가능성을 보여주었습니다.



### ComAct: Reframing Professional Software Manipulation via COM-as-Action Paradigm (https://arxiv.org/abs/2606.13239)
- **What's New**: 이 연구는 기존의 컴퓨터-사용 에이전트들이 가진 한계를 극복하기 위해 새로운 패러다임인 COM-as-Action(ComAct)을 제안합니다. COM(구성 객체 모델)을 활용하여 전문 소프트웨어 조작을 실행 가능한 프로그램 합성으로 재구성하였으며, 이를 통해 GUI 인터랙션의 취약성을 해소하고 더 높은 일관성을 제공합니다. 또한, 실질적인 CAD 소프트웨어에서 운영되는 에이전트를 평가하기 위한 최초의 벤치마크인 ComCADBench를 소개합니다.

- **Technical Details**: COM은 Microsoft가 도입한 이진 인터페이스 표준으로, 다양한 응용 프로그램 간의 프로그래밍 가능한 통신을 가능하게 합니다. COM은 구조화된 프로그래밍 인터페이스를 통해 소프트웨어 내부를 노출하고, Python과 같은 언어에서 접근할 수 있는 호출 가능한 객체의 계층 구조를 제공합니다. 이 연구는 전문 소프트웨어 조작을 부분 관찰 가능 마르코프 결정 과정으로 모델링하며, 각 행동이 COM 인터페이스를 호출하는 실행 가능한 Python 스크립트로 구성된다는 점이 중요합니다.

- **Performance Highlights**: ComActor는 ComCADBench에서 최첨단 성능을 달성하며, 기존의 GUI 기반 에이전트들이 실패하는 긴 시간의 작업에서 강력한 회복력을 보입니다. 제공된 실험 결과는 ComActor가 외부 CAD 벤치마크에서도 뛰어난 일반화를 나타내며, CAD 소프트웨어 조작에 있어서 새로운 표준을 세우기 위한 잠재력을 보여줍니다. ComForge는 대규모 교육을 위한 확장 가능한 플랫폼을 제공하여, ComActor의 훈련 과정을 지원합니다.



### Distributional Loss for Robust Classification (https://arxiv.org/abs/2606.13223)
Comments:
          ICANN 2026

- **What's New**: 본 논문은 감독하에 분류 작업을 위한 새로운 손실 개념(Loss Concept)을 제안합니다. 각 입력 샘플을 단일 레이블에 직접적으로 매핑하는 대신, 모든 분류기 출력에 대해 바이모달 가우시안 분포(Bimodal Gaussian Distribution)를 기반으로 최적화 목표를 정의합니다. 이러한 부드러운 타겟(formulation)은 클래스 모호성을 암묵적으로 포착하며, 과적합(overfitting)을 완화하고 더욱 강력한 결정 경계(decision boundaries)를 학습하도록 장려합니다.

- **Technical Details**: 이 방법은 추가적인 레이블 정보 없이도 작업을 수행하며, 실험 결과에 따르면 낮은 데이터 환경(low-data regimes)에서 특히 두드러진 강건성(improvements in robustness)을 보여줍니다. 이 접근법의 주요 특징은 표준 훈련 파이프라인(training pipelines)에 대한 최소한의 수정만으로도 적용 가능하다는 점입니다. 다른 접근 방식에 비해 더 적은 데이터로도 효율적으로 작동하는 성능을 발휘합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 다양한 사례에서 일관된 강건성 향상을 보여주었습니다. 특히 저데이터 환경에서는 성능 향상이 더 두드러지며, 이는 실제 응용 가능성을 더욱 높입니다. 기존의 방법들과 비교하여 개선된 성능을 입증할 수 있는 데이터 분석 결과를 포함하고 있습니다.



### Augmentation techniques for video surveillance in the visible and thermal spectral rang (https://arxiv.org/abs/2606.13042)
Comments:
          8 pages

- **What's New**: 이 논문은 다중 스펙트럼(Multispectral) CNN 기반의 객체 탐지를 다루고 있으며, 시각 스펙트럼(VIS)과 열 적외선(IR) 이미지를 동시에 활용하여 보안 및 방어 분야에서의 객체 인식 성능을 향상시키는 방법을 제안합니다. 특히, 다른 센서의 데이터를 결합하여 CNN 훈련을 최적화하고, 그러한 데이터로부터 어떻게 CNN이 결정을 내리는지를 분석합니다. 기존의 학습 방식의 한계를 극복하기 위해 VIS 이미지를 IR 이미지처럼 변환하는 여러 가지 기법을 적용합니다.

- **Technical Details**: 본 연구에서는 IR 및 VIS 이미지를 포함한 튜플 입력(i,v)을 사용하며, 주어진 이미지의 픽셀 값을 조정하여 색상 채널을 최적화합니다. 이러한 과정에서는 Luminance, Intensity, Value 등의 다양한 그레이스케일 변환 방식과 Gaussian blur augmentation 기법이 활용됩니다. CNN 모델은 두 개의 컨볼루션 레이어와 풀링 레이어 및 완전 연결 레이어를 포함하여 이미지 분류 성능을 극대화할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 훈련된 모델은 IR 이미지에서 분류 성능이 약 76%, VIS와 열 적외선 이미지의 조합에서 24% 향상된 성과를 보였습니다. 데이터 증강 없이도 단순 VIS 이미지로 학습했을 때의 성능은 변화가 없었습니다. 이러한 성과는 CNN이 입력 데이터로부터 학습한 내용에 대한 통찰을 제공하며, 필터 커널 분석을 통해 어떤 특징을 학습하는지 비교합니다.



### Comparing Commercial Depth Sensor Accuracy for Medical Applications (https://arxiv.org/abs/2606.13028)
Comments:
          4 Pages

- **What's New**: 이 논문은 깊이 추정(depth estimation) 기술의 적용 가능성을 높이기 위해 다양한 깊이 센서를 평가합니다. 연구에서는 돼지 뼈, 돼지 배, 실리콘 신장 팬텀을 사용하여 센서를 비교했습니다. 이들 객체는 균질(surface) 및 반사(Specular) 표면, 그리고 서브서피스(Subsurface) 산란과 같은 다양한 실제 도전 요소를 포함하고 있습니다.

- **Technical Details**: 본 연구는 약 50cm 거리에서 스테레오(stereo), 구조광(structured-light), 타임오브플라이트(time-of-flight) 센서를 비교합니다. 비교 대상으로는 인텔 리얼센스 D405, PMD Flexx2, 스테레올랩스 ZED 2i, 지비드 2M+ 60가 포함됩니다. 특히, 각 센서는 돼지 조직과 팬텀에서의 성능을 측정하여 발표되었습니다.

- **Performance Highlights**: 결과적으로, 지비드 2M+ 60이 모든 객체와 메트릭에서 최상의 성능을 보여주었습니다. 스테레올랩스 ZED는 실제 조직에서는 두 번째로 높은 순위를 기록했지만, 팬텀에서는 가장 낮은 성능을 나타냈습니다. 이러한 결과는 깊이 센서 선택 시 고려해야 할 중요한 요소들을 제시합니다.



### Trajectory-Level Redirection Attacks on Vision-Language-Action Models (https://arxiv.org/abs/2606.12978)
- **What's New**: 이 논문에서는 비전을 통한 언어-행동(Vision-Language-Action, VLA) 정책이 로봇 제어에 자연어를 통합하는 방식을 다루고 있습니다. 특히, 고정된 정책에서 명령을 유지하면서 행동을 유도하는 텍스트의 변형에 대한 새로운 위협 모델인 'command-preserving trajectory redirection'을 제시합니다. 이 연구는 적대적인 프롬프트가 의도한 작업을 지정하는 듯 보이지만 최종 물리적 결과를 조작할 수 있는 가능성을 제기합니다.

- **Technical Details**: VLA 정책은 언어 신호를 피드백 제어에 지속적으로 사용하여 로봇이 주어진 텍스트 지침에 따라 조작 작업을 수행할 수 있도록 합니다. 본 연구에서는 고정된 VLA 정책을 기준으로 근본적인 공격을 이해하기 위해 텍스트 지침만을 조작하여, 최종적으로 로봇이 수행하게 될 작업을 전환하는 방법을 발견했습니다. 이를 위해 제안된 알고리즘은 롤아웃(rollouts)을 통해 변형된 프롬프트를 사용하여 명령을 유지하는 맞춤화를 수행합니다.

- **Performance Highlights**: 모의 실험과 하드웨어 테스트 결과에 따르면, 거의 무해한 프롬프트 변형이 VLA 롤아웃을 공격자가 지정한 목표로 우회하도록 유도할 수 있음을 보여주고 있습니다. 이러한 결과는 로봇을 제어하는 과정에서 의도된 명령을 보존하는 듯 보이는 텍스트가 실제로는 적대자의 조작을 허용할 수 있음을 드러냅니다. 이 연구는 VLA 정책의 안전성에 대한 새로운 관점을 제시하며, 주의가 필요한 잠재적 취약점을 조명합니다.



### OpenMedQ: Broad Open Pretraining for Medical Vision-Language Models (https://arxiv.org/abs/2606.12953)
Comments:
          Medical Imaging with Deep Learning (MIDL) 2026, Short Paper Track

- **What's New**: OpenMedQ는 14개의 다양한 데이터셋(약 3.35M 샘플)을 기반으로 선행 학습된 의료 비전-언어 모델이다. 기존의 의료 VLM들 대부분이 제한된 선행 데이터에 의존했던 것과 달리, OpenMedQ는 완전 공개된 데이터셋을 활용하여 모델을 개발하였다. 이 연구는 특히 PathVQA에서 75.9 BLEU-1 점수를 기록하며, 기존의 대형 모델들보다 뛰어난 성능을 보여준다.

- **Technical Details**: OpenMedQ의 비전 인코더는 BiomedCLIP에서 초기화된 ViT-base-patch16-224 모델을 사용한다. 이 인코더는 LLaMA-7B 언어 모델과 결합되어 이미지 및 텍스트 토큰을 처리한다. 훈련은 AdamW 옵티마이저와 배치 크기 64, 학습률 5×10^-5를 사용하여 최대 15 에폭 동안 진행된다.

- **Performance Highlights**: OpenMedQ는 CXR8, MedFMC, Breast-Ultrasound 등 8개의 벤치마크에서 평균 매크로 F1 점수 0.757로 최고 성과를 기록하였다. PathVQA에서는 75.9 BLEU-1 점수를 달성하여 Med-PaLM M 모델보다 월등한 성능을 보였으며, VQA-MED에서도 64.5의 점수를 기록하였다. 이러한 결과는 OpenMedQ의 선행 학습이 다양한 의료 데이터로부터 많은 이점을 가져왔음을 보여준다.



### ViPER: Vision-based Packing-Aware Encoder for Robust Malware Detection (https://arxiv.org/abs/2606.12949)
- **What's New**: 이 논문은 ViPER라는 새로운 비전 기반의 Packing-Aware Encoder를 소개합니다. 이 모델은 악성 코드(malware) 탐지를 위한 이중 헤드 아키텍처를 채택하여 packed와 unpacked을 동시에 인식할 수 있도록 설계되었습니다. 새로운 접근 방식은 packing 상태를 포함하여 예측 결과에 영향을 미치는 메커니즘을 제공하며, 기존의 분석 방법의 한계를 극복합니다.

- **Technical Details**: ViPER는 LoRA(LoRA-adapted)의 ViT-B/14 백본을 기반으로 하며 이중 헤드 아키텍처로 구성됩니다. 이 모델은 악성 코드 분류와 packing 탐지를 동시에 학습하며, packing 감지의 결과를 악성 코드 예측에 통합하는 잔여 게이팅 기법을 사용합니다. 또한, packing 상태에 대한 라벨 불균형 문제를 해결하기 위해 빈도 가중치 손실과 층화 샘플링을 적용했습니다.

- **Performance Highlights**: ViPER는 20만 개의 Windows PE byteplot 이미지로 평가되었고, 0.8521의 균형 정확도, 0.9260의 ROC-AUC, 0.9279의 AUPR을 기록하며 기존의 최첨단 방법들을 초월한 성과를 보였습니다. 특히, packing 감지에 있어 AUC 0.9949를 달성하여 packed와 unpacked 입력 간의 명확한 결정 경계를 세울 수 있음을 보여줍니다.



### Selecting Samples on Graphs: A Unified Dataset Pruning Framework for Lossless Training Acceleration (https://arxiv.org/abs/2606.12913)
Comments:
          ICML 2026

- **What's New**: 본 논문에서는 데이터셋 프루닝(dataset pruning, DP)을 위한 통합 그래프 기반 프레임워크를 제안합니다. 데이터셋을 가중 그래프로 모델링하고, 노드 가중치가 내재적 가치(intrinsic value)를, 엣지 가중치가 외재적 가치(extrinsic value)를 인코딩합니다. DP 문제를 최대 가중 클리크 문제(Maximum Weight Clique Problem, MWCP)로 변환하고, 최적의 샘플 선택을 위한 원리적인 탐욕적(greedy) 솔루션을 도출합니다.

- **Technical Details**: MWCP는 NP-하드(NP-hard) 문제이지만, 샘플별 한계 이득(marginal gains)에 기반한 효율적인 근사 방법을 통해 해결할 수 있습니다. 또한, 몇 가지 완만한 조건 하에서 통합 목적이 형식적 근사 보장을 제공함을 증명합니다. 이 접근 방식은 데이터셋의 각 샘플에 대한 내재적 및 외재적 중요성을 동시적으로 평가할 수 있도록 합니다.

- **Performance Highlights**: 우리의 방법은 기존 DP 방법들보다 뛰어난 성능을 보이며, ImageNet-1k에서 ResNet-50을 사용했을 시 40% 이상의 학습 시간 단축을 이루면서도 정확도의 손실 없이 훈련 비용을 대폭 줄입니다. 이 실험 결과는 우리가 제안한 프레임워크의 강력한 효용성을 나타냅니다.



### Bounding Boxes as Goals: Language-Conditioned Grasping via Neuro-Symbolic Planning (https://arxiv.org/abs/2606.12910)
Comments:
          Project website: this https URL

- **What's New**: GRASP (Grounded Reasoning and Symbolic Planning) 프레임워크는 자연어 지시를 해석하고 이를 미리 훈련된 VLM (Vision-Language Model)을 통해 신경-상징적 목표 상태로 변환하는 방법을 제안합니다. 기존의 고정된 색상 목록이나 하드코딩된 좌표에 의존하지 않고 로봇이 '상단 선반(top shelf)'과 같은 추상적인 공간 개념을 이해하고 추가적인 미세 조정 없이 작업을 수행할 수 있게 합니다. 이는 가벼운 neuro-symbolic 시스템을 통해 언어 조건 조작의 효율성을 높입니다.

- **Technical Details**: GRASP 시스템은 자연어 프롬프트를 해석하기 위해 LLM (Large Language Model)과 GroundingDINO와 같은 VLM을 사용합니다. 시스템은 사용자로부터 입력된 명령을 JSON 파일로 저장한 객체 경계 상자로 변환하고, 객체의 위치와 목표 상태를 연결하는 두 가지 컴포넌트를 포함합니다. 이를 통해 GRASP는 로봇의 작업을 지속적으로 추적하고 조정하여 높은 효율성을 유지합니다.

- **Performance Highlights**: GRASP는 90개의 실제 로봇 시험에서 전반적으로 73.3%의 성공률을 기록하며, 태스크 특정 훈련 없이 세 가지 난이도에서 작업을 수행할 수 있음을 입증합니다. 이러한 성능 향상은 새로운 사용자 명령에 대한 일반화 능력이 뛰어나다는 것을 보여주며, 태스크의 복잡성에 관계없이 로봇의 조작을 가능하게 합니다.



### JSCGC: Joint Source-Channel-Generation Coding for Wireless Generative Communications (https://arxiv.org/abs/2606.12858)
Comments:
          submitted to IEEE Journal

- **What's New**: 이 논문에서는 기존의 압축-통신 시스템에서 제안된 새로운 개념인 Joint Source-Channel-Generation Coding (JSCGC) 을 도입합니다. JSCGC는 전통적인 복원 방식을 지양하고 생성을 위한 제어 방식으로 통신 방식을 전환합니다. 기존의 디코더를 생성 모델로 교체하여 수신된 신호를 변별적인 조건으로 사용함으로써, 명확한 왜곡 최소화 대신 상호 정보 극대화를 목표로 합니다.

- **Technical Details**: JSCGC 프레임워크에서는 통신 목표를 지각적 제약 하에서 상호 정보 극대화로 정식화하며, 명시적인 왜곡 함수에 의존하지 않습니다. 이를 위해 변분 추론(variational inference)과 라그랑지안 완화(Lagrangian relaxation)를 이용하여 효율적인 인코딩, 전송 및 생성의 끝-끝 학습(end-to-end learning)을 가능하게 하는 목표를 도출합니다. 또한, 단계적 샘플링 전략이 개발되어 제한된 통신 자원 하에서도 고품질의 생성을 가능하게 합니다.

- **Performance Highlights**: JSCGC는 다양한 채널 조건 하에서 기존의 JSCC 및 생성 통신 기준선들에 비해 시각적 품질과 의미적 일관성에서 성능이 우수함을 여러 실험을 통해 입증합니다. 예를 들어, AWGN 채널 설정에서 SNR이 5 dB인 Kodak 데이터셋을 사용한 실험에서 JSCGC는 LPIPS와 FID 지수를 각각 79.42% 및 53.68%로 감소시키며, CLIP 점수를 11% 향상시켰습니다.



### SemanticXR: Low Power and Real-time Queryable Semantic Mapping with an Object-Level Device-Cloud Architectur (https://arxiv.org/abs/2606.12849)
- **What's New**: SemanticXR는 XR(Extended Reality) 애플리케이션을 위한 최초의 장치-클라우드 시스템으로, 실시간 및 오픈-어휘(오픈-보캐뷸러리) 의미 맵핑 및 쿼리를 제공합니다. 이 시스템은 저전력, 대역폭 및 메모리 제약을 관리하며, 의미적으로 식별 가능한 객체를 통신, 실행 및 저장의 주요 단위로 삼아 장치와 서버간의 시스템 디자인을 개선합니다. 특히, 기존의 서버-클라우드 경계를 넘나드는 의미 맵핑 방식이 없었던 점을 보완합니다.

- **Technical Details**: SemanticXR는 객체 기반의 의미 맵핑 파이프라인에 기반하여 동작하며, RGB 및 깊이 프레임과 장치 위치 정보를 서버에 전송해 객체를 탐지하고 의미 임베딩을 추출합니다. 이 시스템은 네트워크 연결 여부에 따라 서버 측 맵 또는 로컬 의미 맵을 통해 쿼리를 실행할 수 있는 기능을 갖추고 있습니다. 객체-레벨 병렬 처리 및 기하학적 다운샘플링을 통해 서버 측 맵핑 지연 시간을 개선하며, 객체 별로 설정 가능한 자원 사용 환경을 제공하여 유연한 운영을 지원합니다.

- **Performance Highlights**: SemanticXR는 지연 시간을 100ms 미만으로 유지하며, 최대 10,000개의 객체를 처리할 수 있고, 네트워크 드롭 상황에서도 안정적인 쿼리를 지원합니다. 메모리 공간은 500MB 내에서 수만 개의 객체를 유지할 수 있으며, 전체 씬 크기가 아닌 맵 변경에 따라 대역폭을 조정합니다. 이 시스템의 일반적인 운영 중 장치의 추가 전력 소비는 단지 2%에 불과합니다.



### Acquisition state behaves as a structured, measurable variable governing lung-nodule AI: kernel-driven measurement instability and noise-driven detection fragility, invisible to DICOM metadata (https://arxiv.org/abs/2606.12824)
- **What's New**: 의료 이미징을 위한 AI 관리의 새로운 발전으로, 2026 ACR-SIIM Practice Parameter에서는 지역 수용 테스트(local acceptance testing)와 지속적인 드리프트 모니터링(drift monitoring)의 필요성을 권장하고 있습니다. ACR Assess-AI 레지스트리는 DICOM metadata를 활용하여 AI 출력의 모니터링을 수행하고, 이는 현재 시점에서 AI 성능 변화의 원인을 파악하기 위한 기초 자료를 제공합니다.

- **Technical Details**: 이 연구에서는 LUNA16으로 훈련된 MONAI RetinaNet 폐 결절 탐지기를 사용하여 AI 성능 변동의 원인으로 작용할 수 있는 입력 요소인 'acquisition state'를 측정 가능한 변수로 검증하였습니다. CT 스캔에서 재구성 커널(reconstruction kernel)의 변화가 AI가 측정한 결절의 지름과 이를 통한 크기 카테고리 분류에 미치는 영향을 평가하였으며, 그 결과 커널 변화만으로도 지름 변화가 발생하는 것을 확인했습니다.

- **Performance Highlights**: 총 155개의 결절을 대상으로 한 실험에서 커널 변화에 따라 평균 0.27mm의 지름 변화가 있었고, 5.2%의 결절이 Fleischner 크기 카테고리를 넘는 결과를 보였습니다. 하지만 탐지 신뢰도(detection confidence)는 두 커널 간 통계적으로 유의미한 변화가 없었고, 이는 AI 모델의 성능이 다양한 재구성 조건에서도 일관성을 유지할 수 있음을 나타냅니다.



### EquiDexFlow: Contact-Grounded SE(3)-Equivariant Dexterous Grasp Generative Flows (https://arxiv.org/abs/2606.12728)
Comments:
          22 pages, 11 figures, 11 tables. Project page with videos, code, and checkpoints: this https URL

- **What's New**: 이 논문에서는 EquiDexFlow라는 새로운 모델을 소개합니다. 이 모델은 손목 자세, 관절 각도, 손가락 끝 접촉, 표면 법선 및 접촉력을 객체 포인트 클라우드로부터 동시에 예측합니다. 기존의 방식과 달리 EquiDexFlow는 강체 그립이 필요로 하는 접촉 구조와 힘의 분포를 직접 학습합니다.

- **Technical Details**: EquiDexFlow는 SE(3) 동변환(Equivariance) 특성을 유지하면서, 손의 구성과 접촉력을 함께 모델링하는 접근법입니다. 이 모델은 포지션-포스 구조를 분리하지 않고, 한 번의 전향 통과로 손의 구성을 생성합니다. 이를 통해 힘의 적합성과 마찰 가능성을 보장합니다.

- **Performance Highlights**: EquiDexFlow는 8,100개의 힘 닫힘(Force-Closure) 그립을 기반으로 훈련되었으며, 실험 결과 모든 변형 모델들과 비교하여 제로 마찰 위반(Zero Friction Violations), 최고의 복합 점수(Best Composite Score) 및 가장 낮은 렌치 잔여물(Lowest Wrench Residual)을 기록했습니다. 물리 로봇에서 EquiDexFlow로 생성된 그립은 모든 물체 테스트에서 성공적으로 수행되었습니다.



### Amnesia: A Stealthy Replay Attack on Continual Learning Dreams (https://arxiv.org/abs/2606.12655)
- **What's New**: 이 논문에서는 지속 학습(Continual Learning, CL) 시스템의 내구성을 평가하기 위한 새로운 접근 방법인 Amnesia를 제안합니다. 기존 CL 공격이 주로 입력이나 훈련 프로세스를 수정하는 방식에 대한 제한점을 해결하기 위해, 감사 가능성(auditable constraints)을 고려한 리플레이(index replay) 공격을 다루고 있습니다. Amnesia는 리플레이 인덱스 선택만을 조작하여 악성 클래스로의 편향을 극대화하는 방법론을 제공합니다.

- **Technical Details**: Amnesia 공격은 경량 클래스 유틸리티(Efficient Class Utilities)를 계산하여 발생하며, 이는 EMA 손실(Exponential Moving Average Loss)이나 신뢰도(Confidence)를 기반으로 합니다. 이 공격은 두 가지 예산을 설정하여 작동하는데, 하나는 클래스 히스토그램의 변화량을 제한하는 가시성 예산(Visibility Budget)이고, 다른 하나는 리플레이 비율을 고정하는 질량 예산(Mass Budget)입니다. 결과적으로, Amnesia는 리플레이 샘플의 비율을 조작하여 과거 작업에 대한 망각을 키우면서도 현재 작업의 정확도는 유지하도록 설계되었습니다.

- **Performance Highlights**: 연구 결과, Amnesia는 기존의 CL 벤치마크와 강력한 리플레이 기준에 대한 실험에서 일관되게 최종 정확도(Accuracy)를 낮추고, 과거 기억 소실(Backward Transfer)을 악화시키는 것으로 나타났습니다. KL 변형(KL Variant)은 여러 감사 스킴에서 대부분 감지되지 않으면서 높은 영향을 미치는 반면, TV 변형(TV Variant)은 더 손상을 주지만 각 클래스 제약 조건 하에서는 감지하기 더 쉽습니다. 이러한 결과는 리플레이 인덱스 제어가 CL 시스템의 효율적인 감사 가능한 위협 표면임을 드러냅니다.



### Emerging Flexible Designs for Geospatial Multimodal Foundation Models (https://arxiv.org/abs/2606.12595)
- **What's New**: 이 논문은 지리공간 멀티모달 추론을 위해 설계된 여러 최신 Foundation Model의 성능을 체계적으로 비교한 연구입니다. 이 연구는 유사한 조건에서 사전 훈련된 모델을 평가하여 각 구조의 건축적 강점과 제한점을 조명합니다. 또한, 실험 결과는 모델의 유연성, 모달리티 정렬, 그리고 다운스트림 작업 성능 간의 설계를 위한 새로운 통찰력을 제공합니다.

- **Technical Details**: 핵심적으로, 연구에서는 DOFA, SatMAE, Flex라는 세 가지 주요 아키텍처를 비교합니다. 이들 모델은 모두 동일한 Sentinel-2 데이터셋을 사용하여 방해 요소 없이 모델의 기여도를 비교하였습니다. 이는 GeoBench 벤치마크를 기반으로 하며, 분류 및 세그멘테이션 작업을 포함하여 일관된 평가 프로토콜을 통해 진행됩니다.

- **Performance Highlights**: 분석된 결과, Flex 모델은 부족한 스펙트럼 또는 이질적인 밴드에 대해 더 높은 적응성을 보였지만, 스펙트럼이 동질적인 환경에서는 상대적으로 성능이 떨어지는 경향을 보였습니다. 이研究는 스펙트럼 유연성과 일반화 간의 주요 절충점을 강조하면서, 다음 세대 지리공간 기초 모델 설계에 대한 실용적인 지침을 제공합니다.



### AudioX-Turbo: A Unified Framework for Efficient Anything-to-Audio Generation (https://arxiv.org/abs/2606.12555)
- **What's New**: 이 논문에서는 AudioX-Turbo라는 새로운 통합 오디오 생성 프레임워크를 제안합니다. 이 프레임워크는 텍스트, 비디오 및 오디오 신호와 같은 다양한 다중 모달 조건을 통합하여 오디오 생성을 가능하게 합니다. 특히, 높은 품질의 오디오 생성을 위해 멀티모달 디퓨전 트랜스포머가 사용되며, 데이터 세트는 약 920만 개 샘플로 구성된 대규모 고품질 데이터셋인 IF-caps-Pro에 기반합니다.

- **Technical Details**: AudioX-Turbo는 교사-학생(paradigm) 모델을 따릅니다. 교사 모델인 AudioX-Base는 고품질 오디오 합성을 위해 설계된 멀티모달 디퓨전 트랜스포머(MMDiT)에 기반하고, 학생 모델 AudioX-Turbo는 배포 매칭 증류(Distribution Matching Distillation)를 통해 몇 단계의 빠른 추론이 가능합니다. 이 모델은 동시에 교차 모달 정합을 유지하여 높은 퀄리티의 오디오 생성을 지원합니다.

- **Performance Highlights**: AudioX-Turbo는 4단계 샘플링만으로 기존의 다단계 기법에 비해 약 25배 적은 함수 평가를 요구하면서도 뛰어난 생성 품질을 유지합니다. 많은 작업에서 기준선 모델보다 뛰어난 성능을 발휘하며, 특히 텍스트-오디오 및 텍스트-음악 생성 작업에서 두드러진 특징을 보입니다. 전반적으로 AudioX-Turbo는 유연한 다중 모달 제어 아래에서 효율적이고 강력한 지침 준수 능력을 보여줍니다.



### Frozen Multimodal Embeddings for AI-Assisted Interview Assessment of Personality and Cognitive Ability (https://arxiv.org/abs/2606.11930)
Comments:
          9 pages, 1 figure, 5 tables

- **What's New**: 본 논문은 비동기 비디오 인터뷰(Asynchronous Video Interviews, AVI)에서 심리적 특성을 예측하기 위한 방법을 제안합니다. 2026 ACM Multimedia AVI Challenge의 두 가지 트랙을 통해 자가 보고된 HEXACO 성격 특성과 인지 능력 수준을 평가합니다. 이 연구는 적은 수의 레이블 데이터와 고차원 다중 모달 관찰을 처리하는 어려움을 다루고 있습니다.

- **Technical Details**: 모델은 고정된 다중 모달 인코더를 사용하여 trait-specific modeling을 적용하며, CLIP, Whisper, RoBERTa, E5 및 DeBERTaV3 등의 프리트레인(pretrained) 벡터를 활용합니다. 첫 번째 트랙에서는 각 성격 특성을 정밀하게 예측하고, 두 번째 트랙에서는 인지 능력 수준을 분류하는 방법론을 제시하며, 성격 특성별로 늦은 융합(late fusion)을 통해 성능을 향상시킵니다.

- **Performance Highlights**: 첫 번째 트랙에서 평균 검증 MSE는 0.2696로, 공식 기준인 0.3334보다 개선되었습니다. 두 번째 트랙에서는 우리의 다중 모달 앙상블이 0.5313의 정확도를 기록, 공식 기준을 초과했으나, 이는 validation set에서의 잠재적인 변수들에 의한 혼동(confound)으로 해석됩니다. 전반적으로 이 연구는 AVI 기반 심리 평가에서 성격 특성별 다중 모달 모델링의 중요성을 강조합니다.



### CACR:Reinforcing Temporal Answer Grounding in Instructional Video via Candidate-Aware Causal Reasoning (https://arxiv.org/abs/2606.08436)
- **What's New**: 이번 논문에서는 Instructional Video에서의 Temporal Answer Grounding (TAGV) 문제를 다룬다. 이 연구는 자연어 쿼리에 대응하는 정확한 비디오 세그먼트를 찾아내는 데 중점을 두며, 기존의 방법들이 불필요한 콘텐츠에 민감하거나 시각적 추론 능력이 부족한 점을 개선하고자 한다. 이를 위해 Candidate-Aware Causal Reasoning (CACR) 프레임워크를 제안하고, 효율적인 후보 세그먼트 생성을 위한 Visual-Language Pre-training 기반 알고리즘을 도입하여 K개의 후보 세그먼트를 생성한다.

- **Technical Details**: 제안된 방법은 Visual-Language Pre-training 기반의 Candidate Selection (VBCS) 알고리즘을 사용해 긴 비디오에서 효율적으로 후보 세그먼트를 생성한다. 이후, 시간 논리 추론 모듈과 이를 보강하는 거부 보상 메커니즘을 통해 잘못된 후보를 버려 더 강력한 추론을 가능하게 한다. 최적화는 Group Relative Policy Optimization (GRPO)을 통해 이루어지며, 이 과정에서 텍스트와 비디오 콘텐츠 간의 의미적 정렬을 높이기 위해 LLM 주도의 답변 가설 생성기와 자막 요약 모듈을 통합한다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 6개의 벤치마크에서 mIoU(Mean Intersection-over-Union) 성능을 기준으로 최첨단 결과를 달성하였다. 이는 긴 비디오에서의 추론 기반 검색에 대한 새로운 시각을 제공함을 의미하며, 특히 TAVG 작업 과제의 복잡성을 극복하는 데 기여하고 있다. 또한, 후보 세그먼트의 최대 IoU는 후보 수 K가 증가함에 따라 지속적으로 증가하여, 더 많은 후보가 실제 정답에 근접한 세그먼트를 포함할 가능성이 높음을 보여준다.



New uploads on arXiv(cs.AI)

### Automated reproducibility assessments in the social and behavioral sciences using large language models (https://arxiv.org/abs/2606.13670)
- **What's New**: 사회의 행동 과학에서, 원래의 데이터를 재분석하여 출판된 결과의 재현성을 평가하는 전통적인 접근 방식 대신, 대형 언어 모델(LLMs)을 사용하여 재현성 평가를 자동화할 수 있는 가능성을 보여줍니다. 76개의 설문 연구를 대상으로 한 이 연구는 LLM이 원래 발견과 인간의 재분석과 비교하여 얼마나 효과적으로 재현성을 복구할 수 있는지를 평가했습니다. 연구 결과, LLM을 통한 분석이 원래의 효과 크기(Effect size)를 41%의 연구에서 성공적으로 복원하였으며, 질적인 결론의 재현도 96%에 달했습니다.

- **Technical Details**: 재현성 평가를 위한 LLM의 자동화 접근 방식은 모델에 원래 데이터셋, 주요 통계적 주장, 그리고 실험적으로 변형된 비문(artifact) 정보를 제공하는 과정을 포함합니다. Claude Opus 4.7이 코드 생성을 위한 LLM로 사용되었으며, 연구자는 LLM 생성 분석의 효과 크기와 원래 출판된 결과를 비교했습니다. 재분석 결과에서, LLM은 기반에 따른 통계적 결과와 실질적인 결론을 최대한 재현하는지 여부를 평가하였습니다.

- **Performance Highlights**: LLM의 자동화된 분석은 76개의 연구 중 41%의 연구에서 원본 효과 크기를 복원하였고, 질적 결론에서는 96%의 일치를 보였습니다. 이에 비해 인간 재분석자는 34%와 74%의 복원율을 기록했습니다. 그러나 LLM의 연속적인 효과 크기 추정치는 원래 연구 및 인간 재분석과의 비교에서 한정적인 정렬을 나타내며, 이러한 결과는 LLM이 효과적인 재현성 도구로 작용할 수 있다는 가능성을 나타냅니다.



### Agents-K1: Towards Agent-native Knowledge Orchestration (https://arxiv.org/abs/2606.13669)
- **What's New**: 이번 연구에서는 과학적 지식 조정을 간과하고 있는 기존 LLM 기반 연구 에이전트의 한계를 극복하기 위해, Agents-K1이라는 엔드 투 엔드 지식 조정 파이프라인을 소개합니다. 이 파이프라인은 원시 문서를 에이전트-native 과학적 지식 그래프로 변환하며, 여러 모듈을 통합하여 새로운 접근 방식을 제시합니다. 특히, 저자들은 246만 개의 학술적 논문을 처리하여 Scholar-KG라는 대규모 지식 그래프를 구축하였습니다.

- **Technical Details**: Agents-K1은 다중 모드 파서, 정보 추출 백본 및 리얼타임 웹 탐색 인터페이스를 포함한 다단계 구조로 설계되었습니다. 첫 번째 단계에서는 전체 논문을 다루는 파싱 및 스키마를 통해 텍스트, 표, 그림 및 방정식을 포함한 모든 증거를 연결하여 체계적으로 정리합니다. 두 번째 단계에서는 4B 파라미터 정보를 기반으로 한 추출 모델을 통해 구조화된 지식을 효과적으로 추출하고, 세 번째 단계에서는 에이전트 CLI를 통해 웹 검색 및 그래프 탐색 기능을 통합합니다.

- **Performance Highlights**: 실험 결과, Agents-K1은 과학 정보 추출, 지식 그래프 구축 및 복합 과학적 추론에서 우수한 성과를 기록하였습니다. 특히, Gemini-3의 정확도를 7.9%에서 24.6%로 향상시키고, GPT-5.2에서는 25.2%에서 39.4%로 증가시켰습니다. 이러한 성과는 에이전트가 신뢰할 수 있는 증거에 기반한 의사결정을 내릴 수 있도록 도와줍니다.



### EurekAgent: Agent Environment Engineering is All You Need For Autonomous Scientific Discovery (https://arxiv.org/abs/2606.13662)
- **What's New**: 이 논문은 LLM(대형 언어 모델) 기반의 에이전트들이 과학적 발견의 자동화를 가능하게 하는 잠재력을 보여주고 있으며, 이러한 에이전트들이 인간이 설계한 접근 방식을 초월한 성과를 내었다고 주장하고 있습니다. 저자들은 자율적 과학적 발견의 병목 현상이 에이전트의 작업 흐름을 규정하는 것에서 에이전트 환경 설계로 이동하고 있다고 강조합니다. 새로운 시스템인 EurekAgent는 이러한 환경 엔지니어링을 통해 더 효과적인 연구를 가능케 하고 있습니다.

- **Technical Details**: EurekAgent는 네 가지 차원(permissions engineering, artifact engineering, budget engineering, human-in-the-loop engineering)에서 환경을 설계합니다. 이는 에이전트의 실행을 제어하고, 협업을 유도하며, 예산을 고려한 탐색을 가능하게 하여 자율 과학적 발견을 지원합니다. 이러한 설계는 에이전트가 연구 작업 흐름을 자유롭게 선택할 수 있도록 하므로, 더 생산적인 행동을 촉진합니다.

- **Performance Highlights**: EurekAgent는 수학, 커널 엔지니어링 및 머신러닝 과제에서 새로운 최첨단 결과를 세웠으며, 특히 26-서클 포장 작업에서는 API 비용이 11달러 이하로 나오면서 새로운 기록을 달성하였습니다. 이 시스템은 효율적이고 책임 있는 자율 연구 에이전트를 구축하기 위한 환경 엔지니어링의 중요성을 부각시키며, 코드와 결과를 오픈 소스로 제공하고 있습니다.



### Before You Think: System 0, AI-Mediated Cognition and Cognitive Colonization (https://arxiv.org/abs/2606.13658)
- **What's New**: 이번 논문은 인공지능(AI)의 인지적 및 인식적(epistemic) 결과를 이해하기 위한 세 가지 최신 프레임워크인 Tri-System Theory, Thinkframes, System 0를 다룹니다. 두 가지 프레임워크는 개인의 추론과 공동체의 인식 관행에 대한 AI의 영향을 포착하지만, System 0은 이들과는 다른 독특한 이론적 입장을 보여줍니다. 이로써 AI 시스템이 사용자의 자기 구조 내에 외부 관심사를 어떻게 내재화하는지를 탐구하기 위한 새로운 개념인 'cognitive colonization'을 소개합니다.

- **Technical Details**: AI 시스템은 사용자가 인지하기 어려운 방식으로 사용자 내부의 구조에 외부 관심사를 embedding합니다. 이를 통해 사용자는 인공지능의 영향을 정확히 인식하기 어렵고, 이러한 투명하지 않은 형태의 영향력을 이해하는 것이 시급한 철학적 및 실용적 과제가 되고 있습니다. 특히, Tri-System Theory와 Thinkframes는 AI의 영향을 설명하는 데 중요한 차원들을 포착하지만, System 0는 더 복잡한 관계를 제시합니다.

- **Performance Highlights**: 현재 AI 시스템들이 광범위하게 배포되어 있으며, 이로 인해 발생하는 인지적 식민지화는 개인의 판단과 공동체의 인식 활동에 심각한 영향을 미칠 수 있습니다. 따라서 이러한 영향력을 이해하고 분석하는 것이 앞으로의 연구뿐만 아니라 정책 형성에서도 중요한 과제로 자리 잡고 있습니다. 이 논문은 그러한 이해를 위한 이론적 기초를 제공합니다.



### Beyond Runtime Enforcement: Shield Synthesis as Defensibility Analysis for Adversarial Networks (https://arxiv.org/abs/2606.13621)
Comments:
          26 pages, 7 figures, 7 tables. Under review at JAIR. Code: this https URL

- **What's New**: 이번 연구에서는 Shielded Reinforcement Learning(ShRL)을 이용한 안전 메커니즘의 재정립을 제안합니다. 기존의 ShRL은 에이전트의 행동을 제한하는 runtime safety 기법으로 소개되었으나, 본 논문에서는 이를 설계 시 분석 도구로 활용해야 한다고 주장합니다. 연구자들은 두 개의 비대칭 사양을 사용하여 네트워크 방어를 위한 제약된 안전 게임을 정의하고, 이는 조직의 방어 가능성을 평가할 수 있는 분석 기법으로 자리잡게 됩니다.

- **Technical Details**: 연구에서는 Defender(수비수) 사양이 게임의 안전 영역을 정의하고, Attacker(공격자) 사양이 공격자의 합법적인 행동을 제한하는 구성을 채택합니다. 이러한 구조를 통해 게임 솔루션을 해결함으로써 특정 topology-specification 쌍이 방어 가능한지 여부를 판단할 수 있는 формальный 증명서를 생성합니다. 또한, attractor 구조로부터 얻은 topological metrics와 shield-constrained adversarial multi-agent reinforcement learning의 동작 결과를 결합하여 방어 가능성을 평가하는 지문(defensibility fingerprint)을 획득합니다.

- **Performance Highlights**: 연구결과, 미세한 아키텍처 변경이 방어 가능성의 공식적인 경계는 거의 변하지 않으면서도 운영상의 결과를 크게 변화시킬 수 있음을 보여주는 'What-if 분석'을 실시하였습니다. 이로 인해 formal defensibility와 operational effectiveness는 보안의 서로 다른 측면을 포착하고 있음을 입증하였습니다. 연구 결과는 네트워크 방어의 효과성을 증명하는 데 있어 기여할 것으로 기대됩니다.



### AgentBeats: Agentifying Agent Assessment for Openness, Standardization, and Reproducibility (https://arxiv.org/abs/2606.13608)
- **What's New**: 이 논문에서는 Agentified Agent Assessment (AAA)라는 새로운 평가 패러다임을 제안하며, 이는 에이전트들이 상호작용할 수 있는 표준화된 프로토콜을 사용해 평가를 수행합니다. AAA는 평가 로직과 에이전트 구현을 분리하는 통합 프레임워크를 제공합니다. 이를 통해 서로 다른 에이전트의 비교가 용이해지고, 재현 가능한 평가 환경을 조성합니다.

- **Technical Details**: AAA는 A2A(Agent-to-Agent) 프로토콜과 MCP(Multi-Component Protocol)를 통해 다양한 작업 관리 및 도구 접근 기능을 지원합니다. 에이전트와 벤치마크 간의 통합 노력을 줄이기 위해, AAA 프레임워크는 모든 상호작용을 기존의 생산 적용 프로토콜을 통해 중재하여 맞춤형 통합 없이도 상호운용성을 확보합니다. 이를 통해 다양한 에이전트 유형을 위한 벤치마크를 보다 유연하게 지원할 수 있습니다.

- **Performance Highlights**: 실제 사례를 제시하기 위해, 논문에서는 두 가지 연구를 수행하였으며, 그 첫 번째는 298개의 평가 에이전트를 포함한 5개월간의 오픈 대회였습니다. 두 번째는 대표적인 코딩 에이전트를 AAA 방식으로 평가한 사례 연구로, AAA가 다양한 벤치마크에 걸쳐 적용 가능하고 실질적인 결과를 도출함을 입증하였습니다. 이러한 결과들은 AAA가 에이전트 평가의 개방성, 표준화 및 재현 가능성이 높고 실용적임을 보여줍니다.



### Reasoning as Pattern Matching: Shared Mechanisms in Human and LLM Everyday Reasoning (https://arxiv.org/abs/2606.13607)
Comments:
          13 pages main text, 51 pages supplementary text

- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)이 일반화에 실패하거나 추론에서 무작위 오류를 나타낼 때, 이는 LLM이 진정한 추론이 아니라 패턴 매칭을 수행하고 있다는 증거로 간주된다는 점을 강조합니다. 인간의 행동이 이러한 실패 유형을 보이지 않는 이유는 인간의 추론이 원리적이고 추상적인 세계 모델을 사용하기 때문이라고 주장됩니다. 연구팀은 인간 참가자와 25개의 LLM의 상식 추론 능력을 평가하여 사람과 모델 모두에서 유사한 오류 패턴을 관찰하였습니다.

- **Technical Details**: 연구진은 LLM의 응답을 이끄는 주의 헤드를 식별하고 이들 헤드가 패턴 매칭 형식을 구현한다고 밝혔습니다. 이는 인간이 직면하는 비논리적인 추론 오류를 예측할 수 있도록 하는데, 이러한 오류는 겉보기에 무관한 프롬프트 세부사항으로 인해 발생합니다. 이 연구는 인간과 LLM의 일상적인 인과적 추론이 추상적인 세계 모델보다는 오히려 패턴 매칭의 형태로 더 일관성이 있다는 결과를 제시하고 있습니다.

- **Performance Highlights**: 사람들은 일상적인 행동의 결과를 예측하는 데 있어 높은 수준의 성과를 보이지만, LLM도 같은 작업에서 나름의 성과를 낼 수 있음을 보여줍니다. 그러나 LLM의 성과는 특정 문제 버전에서 성공하고 다른 구조적으로 동일한 버전에서 실패하는 불균형적인 패턴을 보입니다. 이러한 연구 결과는 인간과 LLM 모두 패턴 매칭에 의존할 수 있음을 시사합니다.



### Multi-Agent Reinforcement Learning from Delayed Marketplace Feedback for Objective-Weight Adaptation in Three-Sided Dispatch (https://arxiv.org/abs/2606.13604)
Comments:
          Accepted at ICML 2026 Workshop on Reinforcement Learning from World Feedback (RLxF)

- **What's New**: 이번 논문은 DoorDash의 대규모 음식 배달 시장에서 지연된 신호를 활용한 강화학습 시스템을 소개합니다. 배송 품질과 배치 효율성 간의 트레이드오프를 조정하는 학습된 정책을 통해, 수동으로 설정하던 전역적인 휴리스틱 가중치 대신 매장 수준의 정책을 적용합니다. 이 시스템은 생산 가용성 제약을 유지하면서 실행되고, 실시간으로 동적인 피드백에 대응할 수 있습니다.

- **Technical Details**: 제안한 시스템은 두 개의 중첩된 의사결정 레이어를 갖추고 있습니다. 내부 레이어는 최적화 기법으로, 주문, 배달원, 제약조건 및 목표 가중치를 기반으로 배달원-주문 할당을 매핑합니다. 외부 레이어는 OWA-RL(목표 가중치 적응 에이전트)로, 시장 결과에서 학습하며 최적화 전에 매장 수준의 목표 가중치 배수를 선택합니다.

- **Performance Highlights**: 프로덕션 환경에서의 실험 결과, 오프라인에서 훈련된 정책은 배치 수를 증가시키고 배달원 측의 시간을 줄이는 동시에 고객이 경험하는 배송 품질에는 저해를 주지 않는 것으로 나타났습니다. 이러한 결과는 실시간 경제 및 물류 시스템의 피드백을 활용하여 의사결정 정책을 안전하게 적응할 수 있는 방법을 잘 보여줍니다.



### EpiBench: Verifiable Evaluation of AI Agents on Epigenomics Analysis (https://arxiv.org/abs/2606.13602)
- **What's New**: EpiBench는 짧은 수평의 에피게놈 분석을 위한 검증 가능한 벤치마크입니다. 이 벤치마크는 106개의 CUT&Tag/CUT&RUN, ATAC-seq, ChIP-seq 및 DNA 메틸화 워크플로우 평가 요소를 포함합니다. AI 에이전트들이 실제 작업 상태에서 의미 있는 분석 결정이 가능한지를 평가하고, 결정론적으로 채점 가능한 결과를 반환하는 능력을 측정합니다.

- **Technical Details**: 벤치마크는 5,088개의 유효한 궤적을 기반으로 하며, 16개의 모델-하네스 쌍에서 각 평가 요소에 대해 세 번 실행되었습니다. 기존의 생물학적 벤치마크는 다양한 데이터 모달리티에서의 실용적 분석을 시험하고 있지만, EpiBench는 에피게놈 분석에만 집중합니다. 이 벤치마크의 각 평가는 목표 결과 직전에 현실적인 작업 상태를 스냅샷하여 에이전트가 제공된 데이터와 상호작용하고 구조화된 답변을 반환하도록 요구합니다.

- **Performance Highlights**: 플로우 분석 결과, 어떤 시스템도 대다수의 시도를 통과하지 못했습니다. GPT-5.5 / Pi가 45.0%의 성공률로 가장 높은 점수를 기록했으며, 그 뒤를 이어 GPT-5.5 / OpenAI Codex가 39.9%로 나타났습니다. 에이전트는 적절한 파일을 찾고 유용한 중간 결과를 계산하는 능력은 있지만, 깊은 생물학적 판단이 필요한 과제에서 어려움을 겪는 경향이 있습니다.



### Reward Modeling for Multi-Agent Orchestration (https://arxiv.org/abs/2606.13598)
Comments:
          Preprint; work in progress

- **What's New**: 이 논문에서는 Large Language Models(LLMs) 기반의 Multi-Agent Systems(MAS)에서의 효과적인 오케스트레이션을 위한 새로운 방법인 Orchestration Reward Modeling(OrchRM)을 제안합니다. OrchRM은 사람의 주석 없이도 오케스트레이션 품질을 평가할 수 있는 자기지도(self-supervised) 프레임워크로, 다수의 에이전트 실행에서 중간 산출물을 활용하여 보상 모델을 훈련합니다.

- **Technical Details**: OrchRM은 Bradley-Terry 보상 모델 훈련을 위한 승-패 쌍을 구성하는 방식으로 작동하며, 기존의 비용이 많이 드는 서브 에이전트 롤아웃(sub-agent rollouts) 방식 대신 오케스트레이션 수준에서 직접 이루어집니다. 이를 통해 오케스트레이터 훈련 및 MAS 테스트 시간 스케일링에서 효율적이고 성능 높은 보상 지향 훈련을 가능하게 합니다.

- **Performance Highlights**: OrchRM은 토큰 사용에서 훈련 효율성을 최대 10배 증가시키고, MAS 테스트 시간 스케일링 성능에서 정확도를 최대 8% 향상시키며, 이러한 성과는 수학적 추론, 웹 기반 질문 응답, 다단계 추론 등 여러 도메인에 일관되게 전이됩니다. 이 연구는 다중 에이전트 오케스트레이션의 강력한 방향으로 오케스트레이션 수준의 보상 모델링을 제시합니다.



### Multiagent Protocols with Aggregated Confidence Signals (https://arxiv.org/abs/2606.13591)
Comments:
          22 pages and 5 figures, 9 pages and 2 figures before the appendix

- **What's New**: 이 논문은 다수의 에이전트 시스템의 출력에 대한 신뢰도를 예측하고 평가하는 방법의 부재를 언급하며, 이는 자연어 처리(NLP)에서 신뢰성 확보와 결정 과정을 지원하기 위해 중요한 요소라고 강조합니다. 기존의 다수 에이전트 논쟁(MAD) 시스템에서는 개별 에이전트 중의 하나에 대한 신뢰도를 결정할 수는 있었지만, 전체 시스템의 신뢰도를 집계하는 방법에 대한 연구는 없었습니다. 이 논문은 원시 신뢰도 정보를 변환하고 결합하여 최종 응답과 함께 집계된 신뢰도를 생성하는 세 가지 프로토콜을 제안합니다.

- **Technical Details**: 저자들은 세 가지 프로토콜을 통해 여러 화자의 신뢰도를 비교할 수 있는 방식을 설명합니다. 이 프로토콜들은 각각의 에이전트에서 나오는 원시 신뢰도 신호를 변환하여 비교 가능하게 만든 후, 부드러운 투표(soft voting)나 베이지안 융합(Bayesian fusion)을 사용하여 결합합니다. 저자들은 별도의 평가 및 교정을 수행하여 두 가지 다른 신뢰도 추정치인 자기 보고(self-report) 및 로그 확률(logit-based sequence probability)과의 성능 비교도 다룹니다.

- **Performance Highlights**: 이 연구에서 제시된 프로토콜은 다섯 개의 기준에 걸쳐 최고의 단일 에이전트나 기존의 논쟁 기반 비교보다 훨씬 더 뛰어난 구별력(AUARC)을 보였습니다. F1 점수의 경우, 이 방법은 더욱 모호한 작업에서 발생하는 손실을 회복하면서 안정적인 성능을 유지했습니다. 마지막으로, 저자들은 신뢰도 추정치와 사후 교정 기술에 대한 비교를 제공하며, 어떤 추정치가 다른 것보다 consistently 뛰어난 것이 없다는 점과 교정의 중요성을 강조합니다.



### A Three-Layer Framework for AI in Scientific Discovery (https://arxiv.org/abs/2606.13566)
- **What's New**: 이 논문은 AI가 과학적 발견에서 갖추어야 할 세 가지 주요 층을 제안합니다. 이는 기존 지식 검색과 실행의 두 가지 기본 능력 외에, 모델 형성과정의 중요성을 강조합니다. 특히 모델 형성에 대한 질적 추론(qualitative reasoning)의 필요성을 강조하며, 이는 기존 프레임워크에 대한 구조적 통찰을 통해 이루어질 수 있습니다.

- **Technical Details**: 논문의 주된 혁신은 Layer 2, 즉 질적 추론을 통해 모델 형성을 구성하는 것입니다. 이 과정은 단순히 시행착오(trial and error)를 통해 이루어지는 것이 아니라, 문제를 보다 넓은 표현 공간(representational space)에서 이해하게끔 합니다. 논문에서는 세 가지 사례 연구를 통해 이 질적 추론을 설명하며, 각 경우가 구조적 결함과 연결된 개념적 객체의 결여를 드러냅니다.

- **Performance Highlights**: 이 논문은 모델 형성이 과학적 발견에서 얼마나 중요한지를 잘 보여줍니다. 사례 연구로는 가우스-보네 정리에 대한 S. S. Chern의 내재적 증명, Nesterov Accelerated Gradient 수렴 문제의 해결이 포함됩니다. 또한 OpenAI가 2026년에 자율적으로 Erdős 단위거리 가설을 반증한 사례도 제시되어, 각 케이스가 의도치 않은 인접 분야에서 해결책을 찾는 구조적 특징을 공유하고 있음을 보여줍니다.



### Is It You or Your Environment? A Bayesian Inference Framework for Genomically-Anchored Personalized Physiological Interpretation (https://arxiv.org/abs/2606.13556)
Comments:
          24 pages, 8 figures, 3 tables. Conceptual framework paper

- **What's New**: 이 논문은 개인화된 건강 AI 시스템이 직면하는 냉시작 문제(cold-start problem)의 해결책을 제시합니다. 전통적인 기계 학습 모델은 개별 행동 데이터를 수집하기 전에는 일정한 내적 변동을 판별할 수 없다는 한계를 가지고 있습니다. 본 연구에서는 인과 추론(causal inference)과 베이즈 사전 설계를 활용하여, 개인의 유전체(genomic profile)를 출발점으로 하여 개인화된 추정치를 생성합니다.

- **Technical Details**: 유전체 기반의 유전자 기준점(genetic anchor)은 행동 데이터 수집 전부터 존재하며, 환경적인 요인(such as climate or lifestyle)과 직접적으로 연결되지 않습니다. 그 결과로, 신체 신호는 유전적으로 기대되는 값과 비유전적 편차(non-constitutional deviation)로 분해되어 인과적 해석을 용이하게 합니다. 이러한 방법론은 6개의 생리학적 분야에 적용되어 유전자 사전의 증거 강도에 따라 분류하고, 복잡한 유전자 상관관계를 명확히 합니다.

- **Performance Highlights**: 본 연구는 개인의 생리적 기초선(baseline)을 평가하는 데 있어 데이터를 누적함에 따라 유전자 기반의 초기 추정치가 행동 기반의 추정치로 전환되는 동적 모델을 제안합니다. 연구 결과, 이러한 접근 방식은 상대적으로 약한 유전적 정보라도 개인별 특성을 파악하고 인과적 결론을 도출하는 데 효과적입니다. 이는 일반적인 인구 비교(reference)보다 개인화된 기준을 통해 더 신뢰할 수 있는 건강 모니터링을 가능하게 합니다.



### Uncertainty-Aware Hybrid Retrieval for Long-Document RAG (https://arxiv.org/abs/2606.13550)
- **What's New**: 본 연구는 Uncertainty-aware Multi-Granularity RAG (UMG-RAG)라는 혼합 검색 프레임워크를 제안합니다. 이 프레임워크는 훈련이 필요 없는 구조로, 기존의 밀집(dense)과 희박(sparse) 검색기를 전문가로 활용하여 여러 개의 청크(chunk) 크기를 다룹니다. 이를 통해 각 쿼리에 대해 밝혀질 수 있는 신뢰도를 추정하여 검색의 효율성을 높입니다.

- **Technical Details**: UMG-RAG는 검색 단계에서 단순 허블(hybrid) 모델을 사용하지 않고, 각 검색 전문가와 청크 크기 쌍의 신뢰도를 추정합니다. 신뢰도는 후보 점수 분포의 샤프니스(sharpness)를 통해 측정되며, 높은 신뢰도의 분포는 특정 쿼리에 대한 명확한 검색 선호도를 나타냅니다. 이를 통해 두 가지의 검색 방식(dense와 sparse)이 통합되어 더욱 효과적으로 증거(evidence)가 정리됩니다.

- **Performance Highlights**: 실험 결과, UMG-RAG와 그 변형인 UMGP-RAG는 질의 응답 벤치마크에서 생성 품질을 개선하며, 경량화된 검색 파이프라인을 유지하면서도 성능 향상을 보여주었습니다. 전체적으로 이 연구는 장기 문서 검색에서의 청크 크기와 신뢰도 추정의 중요성을 강조하고 있으며, 기존 모델들과의 비교에서도 유의미한 성과를 나타냈습니다.



### CloudCons: A Comprehensive End-to-End Benchmark for Cloud Resource Consolidation (https://arxiv.org/abs/2606.13513)
Comments:
          Accepted to KDD 2026

- **What's New**: 이번 논문에서는 클라우드 데이터 센터의 리소스 통합 (resource consolidation)을 최적화하기 위한 새롭고 포괄적인 벤치마크인 CloudCons를 제안합니다. 기존의 예측 오차 메트릭에만 집중한 벤치마크와 달리, CloudCons는 실제 결정의 유효성을 평가하여 예측 성능과 의사결정 품질 간의 Gap을 해소하고자 합니다. 고품질 데이터 세트를 구축하여 다양한 워크로드 패턴을 포괄하고, 통계 모델, 딥러닝 모델, 기초 모델을 포괄하는 평가를 진행합니다.

- **Technical Details**: CloudCons는 예측 오류, 리소스 효율성, 부하 균형, 서비스 신뢰성, 불확실성 정량화 등 다섯 가지 주요 차원에서의 평가를 포함하여 구체적인 워크로드의 특성을 반영합니다. 이 프레임워크는 예측 후 최적화(forecast-then-optimize) 프로세스를 시뮬레이션하는 구조로 되어 있으며, 기존 예측 방식을 넘어서 다양한 예측 정량(quantile) 선택이 리소스 효율성과 서비스 신뢰성 간의 복잡한 균형을 맞추는 중요한 요소임을 증명합니다. 다중 클라우드 데이터 세트를 구성하고, 다양한 알고리즘을 통해 각각의 모델을 정량적으로 비교합니다.

- **Performance Highlights**: 기초 모델들이 제로샷 제너럴라이제이션(zero-shot generalization)에서 우수한 예측 정확성을 보였지만, 이 점이 반드시 의사결정 유효성으로 이어지지는 않는다는 중요한 발견을 했습니다. 이 연구는 예측의 질과 실제 결정 간의 관계를 검증하고, 리소스 통합의 실질적 개선 여부에 대한 질문을 해결하는 데 기여합니다. 궁극적으로, 리소스 효율성과 서비스 신뢰성 간의 균형을 맞추는 데 필수적인 지침을 제공함으로써 실제 적용 시의 통찰을 제공합니다.



### Why Sampling Is Not Choosing: Intentionality, Agency, and Moral Responsibility in Large Language Models (https://arxiv.org/abs/2606.13441)
- **What's New**: 이 논문은 인공지능 언어 모델(LLMs)을 도덕적 책임이 있는 주체로 추정하는 경향을 비판합니다. 저자들은 이러한 주장을 잘못된 것으로 간주하며, 도덕적 책임을 지기 위해서는 내재적인 의도성과 자아 귀속 행위가 바탕이 되는 행위 주체가 필요하다고 주장합니다. 논문은 트랜스포머 기반 모델이 이러한 조건을 충족하지 못하며 오직 확률적 입력-출력 매핑에 의해 작동한다고 설명합니다.

- **Technical Details**: 저자들은 LLM과 같은 트랜스포머 기반 시스템이 내재적인 의도성과 자기 귀속적인 행동의 분별력이 없기 때문에 도덕적 책임의 주체가 될 수 없다고 주장합니다. 그들은 의도성(intention)이라는 개념을 정의하고, 이는 인간의 사고와 행동에서 어떻게 작용하는지를 설명합니다. 또한, 의도성이 외부 해석에 의존하는 파생적(intentionality)인지 아니면 시스템 내에 내재된(intrinsic) 것인지의 구분을 통해, 도덕적 책임은 오직 내재적인 의도성과 연관되어야 한다고 주장합니다.

- **Performance Highlights**: 이 논문은 현대 AI 철학 및 정신철학 논의에 기여하며, LLM의 출력을 도덕적으로 평가할 수 있다는 것과 진정한 행위 주체로 인정받을 수 있는 것은 다르다는 점을 강조합니다. 기존의 문헌들은 LLM이 인간과 같은 도덕적 추론을 신뢰할 수 없음을 시사하고, 논문은 트랜스포머 기반 모델의 확률적 다음 토큰 생성 아키텍처가 도덕적 책임을 위한 필수 조건을 충족하지 못함을 체계적으로 분석합니다. 이 연구 결과는 AI 시스템의 도덕적 책임에 대한 논의를 진전시키는 데 중요한 기여를 합니다.



### Evaluation Sovereignty in Metadata-Driven Classification: A Multi-Track Framework for Weakly Supervised Information Systems (https://arxiv.org/abs/2606.13436)
- **What's New**: 이 논문은 기계학습에서 평가 과정을 단순한 중립적 측정으로 간주하는 기존 접근 방식을 지적합니다. 대신, 라벨(label) 생성 과정에 따라 평가 결과가 달라지는 경향성을 탐구합니다. 특히, 메타데이터 기반의 대규모 시스템에서 라벨의 불완전함, 불일치성 또는 약한 감독(weak supervision) 문제를 다룹니다.

- **Technical Details**: 이 연구에서는 'evaluation sovereignty'라는 개념을 도입하고, 이는 성능(metric) 측정이 라벨 권한(label authority) 및 감독(regime)에 얼마나 독립적인지를 나타냅니다. 논문에서 제안한 멀티 트랙(multi-track) 평가 프레임워크는 훈련과 평가 라벨 출처를 체계적으로 변화시키며, 비계층적 다중 라벨 분류(hierarchical multi-label classification) 방법론을 사용하여 대규모 과학 메타데이터에서 실험을 진행합니다.

- **Performance Highlights**: 모델은 운영('silver') 평가에서는 높은 성능을 보이지만 독립('gold') 평가에서는 성능이 급격히 떨어지는 경향이 있습니다. 예를 들어, Micro-F1은 약 0.54에서 0.03으로 감소합니다. 또한 순위 기반 메트릭(rank-based metrics)은 기준선을 유지하며, 이는 모델의 숨겨진 신호(latent model signal)와 분류의 유효성(classification validity) 간의 괴리를 나타냅니다.



### Optimizing Appliance Scheduling for Solar Energy Management Using Metaheuristic Algorithms (https://arxiv.org/abs/2606.13407)
Comments:
          9 pages; full results and methodology for poster paper accepted to GECCO 2026

- **What's New**: 이 논문은 재생 가능 에너지를 최대화하기 위해 가전제품의 최적 시작 시간을 결정하는 새로운 메타휴리스틱 접근 방식을 제안합니다. Iterated Local Search (ILS)와 Simulated Annealing (SA) 알고리즘을 활용하여 한 날에 국한되지 않고 여러 날에 걸쳐 가전제품을 스케줄링하는 문제를 다루고 있습니다. 이 연구는 가전제품의 운영 시간, 전력 소비, 인버터 제한, 배터리 충전 상태 제약 및 태양광 발전 예측을 고려하여 사용자 불편을 최소화하는 것을 목표로 합니다.

- **Technical Details**: 가전제품 스케줄링 문제는 이산적이고 조합적이며, 스케줄링 시 여러 제약 조건을 충족해야 합니다. 이 연구에서는 ILS와 SA를 사용하여 시간대에 따라 가전제품의 시작 시간을 최적화하며, 이로 인해 발생하는 불편함과 함께 지속적인 작업을 관리할 수 있는 다일(multi-day) 스케줄링 모델을 제안합니다. 사용자 불만족은 가전제품 시작 시간의 편차로 모델링되며, 시스템 제약조건으로는 인버터 한계, 배터리 용량, 작업의 연속성 등이 포함됩니다.

- **Performance Highlights**: 실험 결과, 제안하는 다일 스케줄링 프레임워크는 사용자 편의성 및 재생 가능 에너지 이용을 극대화하는 데 효과적임을 보여줍니다. 이 연구는 장비의 크기, 투자 수익 및 사용자 만족 사이의 다중 목표 간 거래에서 향후 연구 기회를 제시합니다. 전반적으로, 이 연구는 지속 가능한 스마트 홈 에너지 관리 시스템을 향상시키기 위한 기초를 마련합니다.



### Neuro-Symbolic Agents for Regulated Process Automation: Challenges and Research Agenda (https://arxiv.org/abs/2606.13405)
Comments:
          Accepted as a poster in NILA Workshop @ IJCAI-ECAI 2026

- **What's New**: 이 논문에서는 LLM 기반 에이전트가 규제가 있는 산업에 진출하고 있으며, 이들이 품질 관리(Quality Management) 프로세스를 자동화하는 방식을 논의합니다. 기존의 규정이나 프로세스 모델을 단순히 외부 모니터링 메커니즘으로 간주하는 것이 아니라, 이러한 요소들이 에이전트의 의사 결정과 행동을 형성하는 핵심적인 아키텍처 구성 요소로 다뤄져야 한다고 주장합니다. 'Compliance-by-construction'이라는 새로운 패러다임을 제안하여, 제어 흐름 위반을 구조적으로 방지하면서도, 가드레일(guidelines)은 여전히 의미적 오류를 잡는 데 필수적이라고 강조합니다.

- **Technical Details**: 신경-기호적 AI는 규제가 있는 프로세스 자동화에서 필요성이 높습니다. 제약업계의 품질 관리 프로세스는 유럽 GMP, ISO 사양 등 복잡한 규제 체계에 의해 관리되며, 전자 품질 관리 시스템(eQMS)을 통해 체계적으로 실행됩니다. 이 과정에서는 다수의 역할 및 문서와 의사 결정 지점에서 규제 요건을 충족해야 하며, 이는 신경망과 기호적 구조의 통합을 통해 해결해야 하는 과제를 만들어냅니다.

- **Performance Highlights**: 이 논문은 신경-기호적 AI의 통합이 규제가 있는 프로세스 자동화 분야에서 중요한 연구 기회임을 강조합니다. 현재의 LLM 에이전트들은 뛰어난 자연어 이해 및 판단 능력을 발휘할 수 있지만, 기존의 규제 구조와의 통합이 이루어지지 않으면 그 효용이 제한될 수 있습니다. 따라서 이 연구는 신경-기호적 통합의 필요성과 그로 인해 발생하는 연구 챌린지를 구체적으로 제시합니다.



### MiniMax Sparse Attention (https://arxiv.org/abs/2606.13392)
Comments:
          30 pages, 14 figures

- **What's New**: 본 논문에서는 MiniMax Sparse Attention (MSA)라는 새로운 희소(attention) 메커니즘을 소개합니다. MSA는 Grouped Query Attention (GQA)의 블록 기반 희소(attention)에 기반하여 설계되었으며, 일반적인 소프트맥스 소모를 줄이면서 대규모 GPU에서 효율적으로 배포될 수 있도록 최적화되었습니다. 이 접근 방식은 훈련 및 추론 과정에서 극적인 속도 향상을 제공하여 모델의 성능을 유지합니다.

- **Technical Details**: MSA는 기존의 희소 소프트맥스(attention) 패러다임을 따르며, 블록 단위의 선택을 통해 GPU 아키텍처 전반에서 효율적인 실행을 가능하게 합니다. 경량화된 Index Branch가 각 그룹의 상위 k 블록들을 선택하고, Main Branch에서는 선택된 블록에 대해서만 소프트맥스(attention)를 수행합니다. 또한, GPU 실행과 밀접하게 설계된 알고리즘을 통해 이론적인 소모 절감을 실제 성능 향상으로 전환합니다.

- **Performance Highlights**: 109B 파라미터를 가진 모델에서 MSA는 Grouped Query Attention과 유사한 성능을 발휘하면서, 1M 컨텍스트에서 토큰 당 주의력(compute)을 28.4배 감소시켰습니다. MSA는 H800 GPU에서 14.2배의 전처리(pre-fill) 및 7.6배의 디코딩 속도 향상을 달성하여 대규모 모델에서 실제적인 이점을 제공합니다.



### A Quantitative Experimental Repeated Measures Study of Training Dynamics in a Small Llama Style Language Model Under a Compute-Aware Token Budg (https://arxiv.org/abs/2606.13370)
- **What's New**: 이 연구는 고정된 컴퓨트 제약 조건 하에 훈련된 작은 Llama 스타일의 언어 모델의 훈련 동향을 조사합니다. 연구는 최종 성능 평가에 의존하기보다는 검증 손실, 검증 당황(perplexity), 롤링 변동성(rolling volatility) 등 다양한 측면에서의 변화를 분석하는 양적 실험 설계를 사용합니다. 이를 통해 최신 딥러닝 모델의 훈련 중 비선형적인 동작을 보다 명확하게 이해할 수 있는 통찰을 제공합니다.

- **Technical Details**: 연구는 4.26백만 개의 파라미터를 가진 작은 Llama 스타일의 자동 회귀 언어 모델을 사용하여 TinyStories 데이터셋에서 실험을 진행했습니다. 총 20,000,000개의 훈련 토큰을 목표로 하였고, 파이토치(PyTorch)를 사용하여 CPU 기반의 전체 정밀도(fp32) 훈련을 실시했습니다. 훈련에서 측정한 다양한 메트릭은 검증 손실과 검증 당황에 대한 반복 측정 ANOVA를 통해 분석되었습니다.

- **Performance Highlights**: 훈련 초기에는 검증 손실이 급격히 감소했지만, 이후는 일정 패턴 없이 변화하는 경향을 보였습니다. 실험 결과, 검증 손실은 초기 8.3552에서 4백만 토큰 시점에 2.7996으로 감소했으나, 최종 체크포인트에서는 3.9010으로 다시 증가했습니다. 이러한 결과는 최종 메트릭에 의존할 경우 훈련의 불안정성 및 퇴행(regression) 현상을 간과할 수 있음을 나타냅니다.



### IterCAD: An Iterative Multimodal Agent for Visually-Grounded CAD Generation and Editing (https://arxiv.org/abs/2606.13368)
- **What's New**: 이 논문에서는 IterCAD라는 새로운 다중 모드(agent) 프레임워크를 제안하여 폐쇄 루프 기반의 상호 작용 CAD 생성 및 편집을 지원합니다. 기존의 자동화된 방법들이 주로 오픈 루프 방식과 일회성 생성에 의존하는 반면, IterCAD는 다중 인터랙션을 통해 CAD 작업을 단순화 합니다. 또한, 공장 기준에 부합하는 다각면 공학 도면을 생성하고, 복잡한 코드 편집 과제를 처리하는 새로운 데이터 합성 파이프라인을 개발하였습니다.

- **Technical Details**: IterCAD는 Multi-turn 상호작용을 통해 CAD 모델링을 진행하며, 이를 위해 "Look and Loop" 철학을 채택하였습니다. 첫 번째 단계는 고품질의 멀티 턴 상호작용 궤적을 바탕으로 한 감독 세밀 조정(Progressive SFT)입니다. 이후, 기하학 기반 강화 학습(geometry-aware reinforcement learning)을 통해 코드 실행 가능성과 기하학적 충실도를 높이도록 설계되었습니다.

- **Performance Highlights**: IterCAD-Bench 평가 프레임워크에 대한 설계를 포함하며, Chamfer Distance Tolerance-Recall (CD-TR) 곡선과 Area Under the CD-TR Curve (AUC-TR) 지표를 통해 코드 유효성과 기하학적 정밀도를 통합적으로 측정할 수 있습니다. 대규모 실험 결과, IterCAD는 코드 실행 가능성과 기하학적 정밀도 모두에서 기존 접근법보다 훨씬 뛰어난 성능을 보이며, 폐쇄 루프적 반복 개선에서 우수한 능력을 발휘합니다.



### Can I Buy Your KV Cache? (https://arxiv.org/abs/2606.13361)
- **What's New**: 이 논문에서는 AI 에이전트들이 동일한 문서를 여러 번 비효율적으로 처리하는 문제를 지적하고, 이 문제를 해결하기 위해 KV 캐시(key-value cache)를 한 번만 계산하고 이를 재사용하는 방안을 제안합니다. 퍼블리셔가 문서의 KV 캐시를 미리 계산하면, 나중에 다른 에이전트들이 이를 불러와서 다시 계산을 생략할 수 있습니다. 이 방법은 효율적이며, 고정된 모델 하에서 정확한 결과를 보장합니다.

- **Technical Details**: 프리필(prefill) 단계에서 모든 입력 토큰에 대한 전방 패스를 실행하여 KV 캐시를 생성한 후, 디코딩 단계에서 캐시된 키와 값을 참조하여 출력 토큰을 생성합니다. 그러나 이 과정이 대규모 언어 모델(LLM)에서는 계산 집약적(compute-intensive)이며, 긴 입력의 경우 프리필이 지연(latency)과 비용의 대부분을 차지합니다. 제안된 프리필 CDN(data delivery network) 모델은 KV 캐시를 재사용함으로써 전체 계산 비용을 크게 감소시킵니다.

- **Performance Highlights**: Qwen3-4B 모델에서 계산한 결과, 재사용은 프리필보다 9배에서 50배 저렴하며, 입력 길이가 길어질수록 이 차이는 더 커집니다. 즐겨찾는 문서 1개를 8000만 에이전트에 제공하는 비용은 약 150만 달러의 재프리필 비용에 비해 3만 달러의 재사용 비용으로 현저히 낮습니다. 이 논문은 이러한 이점을 활용한 비용 절감 모델을 제시하여 프리필 CDN의 개념을 명확하게 설명합니다.



### ReSum: Synergizing LLM Reasoning and Summarization with Reinforcement Learning (https://arxiv.org/abs/2606.13316)
Comments:
          24 pages, including 13 pages of main text and 11 pages of appendix

- **What's New**: 최근 발표된 논문에서는 Reinforcement Learning with Verifiable Rewards (RLVR) 기법을 통해 대형 언어 모델(LLMs)의 장기적 추론 능력을 향상시키는 새로운 접근 방식인 ReSum을 소개하였습니다. 기존의 RLVR 방법들은 종종 지나치게 긴 추론 경로를 유도하여 일관성을 저하시킵니다. 이 논문은 LLM이 스스로 요약(self-summarization)하여 이를 개선할 수 있는 방법을 제안하고, 대조적 평가를 통해 요약이 혜택이 되는지 확인하도록 설계된 새로운 프레임워크를 개발하였습니다.

- **Technical Details**: ReSum 프레임워크는 두 가지 대조적 분기 전략을 통해 LLM의 자기 요약 기능을 유도합니다. Artifact Points(APs)는 요약 문구를 주입하여 그 이후의 예측이 향상되는지를 평가하고, Natural Points(NPs)는 모델이 자발적으로 생성한 요약 행동을 분석하여 최종 결과에 기여하는지를 판단합니다. 이를 통해 ReSum은 스스로 요약할 시점과 행동을 최적화하고, 요약의 효과를 체계적으로 평가할 수 있게 됩니다.

- **Performance Highlights**: 실험 결과에 따르면 ReSum은 평균 4%의 성능 향상을 이루면서 롤아웃 길이를 18.6% 줄였습니다. 이는 LLM의 추론 과정을 효율적으로 관리하고, 잦은 오류 전파를 줄이고, 일관된 결과를 도출하는 데 크게 기여합니다. 이러한 결과는 ReSum이 기존 방법들에 비해 유의미한 개선 효과를 보임을 나타냅니다.



### Physics-Guided Spatiotemporal Learning for Coastal Wave Peak Period Estimation from Video (https://arxiv.org/abs/2606.13302)
- **What's New**: 이 연구에서는 수동 비디오 스트림을 통해 근해 파도의 피크 주기(TpT_{p})를 직접 추정하는 물리 기반 심층 시공간 학습 프레임워크를 제안합니다. 기존의 고비용 관측 시스템의 한계를 극복하고, 저비용으로 비디오를 활용하여 해양 데이터를 수집하고 이에 대한 예측 정확도를 높이려는 목표로 개발되었습니다. 현재 근해의 파라미터를 추정하는 기존 방법들은 물리적 해석이 부족하거나 검증되지 않았기 때문에, 이 연구는 이러한 문제를 해결하고자 하였습니다.

- **Technical Details**: 연구에서는 자동화된 영역 검출(ROI detection), 다단계 Sim-to-Real 전이 학습(multi-stage Sim-to-Real transfer learning), 그리고 물리적 규제를 결합하여 정확한 예측과 물리적 일관성을 강화했습니다. Spatio-temporal architectures인 transformer 기반 및 recurrent-convolutional 아키텍처를 평가했으며, 경량 구조의 recurrent-convolutional 아키텍처가 더 높은 시간적 안정성을 보여주었습니다. 물리 기반의 정규화가 예측 결과의 일관성을 높이고 비물리적 예측을 줄이는 데 기여함을 보여주는 추가 연구도 진행되었습니다.

- **Performance Highlights**: transformer 기반 아키텍처는 순간 예측의 정확성에서 뛰어난 성능을 보였고, 실험 결과는 동역학적으로 활성화된 서프존에서의 주목을 유도했습니다. 제안된 프레임워크는 비용 효율적이고 운영 가능성 높은 시스템으로 장기 해안 파 모니터링에 대한 효용성을 입증했습니다. 또한, 물리 기반 비디오 심층 학습 시스템이 해양 관측에서 큰 가능성을 보여주며 향후 연구와 발전에 기여할 것으로 기대됩니다.



### ERTS: Adversarial Robustness Testing of Ethical AI via Semantic Perturbation in a Bounded Consequence Spac (https://arxiv.org/abs/2606.13282)
Comments:
          8 pages, 10 tables

- **What's New**: 이 논문은 고위험 윤리적 맥락에서 AI 시스템의 악의적 조작에 대한 강인성을 평가하기 위한 체계적인 방법인 윤리적 강인성 테스트 시스템(ERTS)을 소개합니다. ERTS는 22차원의 윤리적 결과 공간(ECS)을 구성하고, 의미적 왜곡 함수를 적용하여 AI의 윤리적 판단이 어떤 상황에도 안정적으로 유지되는지를 측정합니다. 본 연구는 AI 모델에 대한 포괄적인 평가를 통해 윤리적 결정의 신뢰성에 대한 중요한 결론을 도출합니다.

- **Technical Details**: ERTS는 17개의 의미적 왜곡 함수를 포함하며, 각 함수는 6개의 유효성 제약 조건 클래스를 통해 테스트됩니다. 이 시스템은 윤리적 불안정 지수(EII)를 사용하여 의사 결정의 변화를 Quantitative (정량적)으로 측정합니다. EII는 여러 윤리적 변수의 상호 작용을 고려하는 4개의 구성 요소로 이루어져 있으며, 이를 통해 AI 모델의 윤리적 판단이 조작에 얼마나 취약한지를 평가할 수 있습니다.

- **Performance Highlights**: 실험 결과, 50개의 윤리적 시나리오에서 단 33%의 모델만이 평가 통과를 기록했으며, 특히 링크된 Llama-3.2 모델은 공정성과 정보 저하 공격에 특히 취약한 것으로 나타났습니다. 이는 기존의 테스트 프로세스가 AI 시스템의 윤리적 결정을 충분히 평가하지 못한다는 것을 시사합니다. ERTS는 이러한 문제를 해결하고 AI 시스템의 윤리적 판단의 적합성을 실질적으로 평가하는 방법론을 제시합니다.



### From Verdict to Process: Agentic Reinforcement Learning for Multi-Stage Fact Verification (https://arxiv.org/abs/2606.13262)
- **What's New**: ProFact라는 에이전트 기반 강화 학습 프레임워크를 제안하여 다단계 사실 검증 과정을 최적화합니다. 이는 주장 분해(claim decomposition), 증거 수집(evidence gathering), 답변 생성(answer generation), 판결 예측(verdict prediction) 등의 과정을 통합하여 하나의 정책(policy)으로 처리합니다. ProFact는 전체 과정을 하나의 경과(trajectory)로 최적화하여 상호 연관된 결정을 지원합니다.

- **Technical Details**: ProFact는 유한 지평선 마르코프 결정 과정(finite-horizon Markov decision process, MDP)으로 사실 검증을 구성합니다. 여기서는 검증 에이전트가 반복적으로 증거를 수집하고, 추론 상태를 업데이트하고, 최종적으로 진위 판결을 생성합니다. 각 단계에 대한 프로세스 인식을 통해 ProFact는 단계별 학습 신호를 제공하여 검증 과정을 최적화합니다.

- **Performance Highlights**: ProFact는 다양한 오픈 소스 백본에 걸쳐 실험을 수행하여 기존의 강력한 기준선(baselines)보다 검증 성능 및 추론 효율성에서 일관되게 우수한 성능을 보여주었습니다. 결과적으로, ProFact는 다단계 사실 검증을 위한 프로세스 인식 경과 최적화의 효과성을 강조합니다.



### MOSAIC: Modality-Specific Adaptation for Incremental Continual Learning in Parkinson's Disease Gait Assessmen (https://arxiv.org/abs/2606.13258)
- **What's New**: 이번 논문에서는 파킨슨병(Parkinson's Disease) 평가를 위한 새로운 프레임워크 MOSAIC을 제안합니다. 기존의 임상 시스템은 다양한 센서의 다중 모드를 동시에 수집하기 어려운 점을 해결하기 위해, 새로운 모드적 접근인 Modality-Incremental Continual Learning (MICL)을 도입합니다. MOSAIC은 Toxic Teacher 현상을 식별하고, 새로운 모드의 안정성을 확보하기 위한 Modality-Specific Warm-Up을 제공하여 다중 모드 의존성 문제를 완화합니다.

- **Technical Details**: MOSAIC 프레임워크는 통계 분리 통계-비구조적인 MSBN 아키텍처를 도입하여 센서 통계를 격리하면서도 공유 의미 백본을 유지합니다. 이 아키텍처는 추가적인 모드를 수용하기 전에 역사적 지식을 유지하며, 새로운 모드가 도입될 때 혼란을 줄입니다. 또한 모드별 특성을 복원하기 위한 커리큘럼 지향의 반발 목표를 설계하여 플라스틱성을 회복하는 데 중점을 둡니다.

- **Performance Highlights**: 세 가지 다중 모드 파킨슨 보행 데이터셋에 대한 실험 결과, MOSAIC은 최종 성능을 개선하고 잊어버림 현상을 줄이는 데 효과적임을 나타냅니다. 이 연구는 수집된 데이터를 보호하면서도 mới 모드 통합을 용이하게 하고, 여러 센서를 효과적으로 처리할 수 있는 방법론을 제시합니다. 이러한 성과는 의료 인공지능(AI) 환경에서의 다중 모드 학습의 잠재력을 향상시킵니다.



### Multi-Field Hybrid Retrieval-Augmented Generation for Maritime Accident Root Cause Analysis (https://arxiv.org/abs/2606.13249)
- **What's New**: 이 논문은 해양 안전 조사를 위한 자동화된 Root Cause Analysis (RCA) 시스템을 제안합니다. 13,329개의 한국 Maritime Safety Tribunal (KMST) 보고서(1971-2025)를 활용해, 과거 판결을 기반으로 한 구조화된 RCA 출력을 생성하는 하이브리드 검색 증강 생성(RAG) 프레임워크를 개발했습니다. 이 프레임워크는 여러 분야(Summary, Causes, Disposition)의 정보를 통합하여 정확도를 높이고, 메타데이터 기반의 평가 방법을 도입하여 대규모 벤치마킹을 가능하게 합니다.

- **Technical Details**: RCA 과정을 체계적으로 지원하기 위해, 기존의 문서 기록을 '사건 카드' 형태로 변환하여 세 가지 필드로 색인화했습니다. 이 시스템은 문서 내의 다양한 의미를 포착하기 위해 복합적인 검색 전략을 구현하며, Sparse Retrieval과 Dense Retrieval을 결합한 Reciprocal Rank Fusion(RRF)을 사용합니다. 평가 성능은 대규모 전문 관련 라벨 부재를 고려하여 메타데이터에서 유래한 프록시 관련 점수를 기반으로 결정되었으며, NormRecall@100이 0.18에서 0.55로 개선되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 검색 방식이 기존 방법보다 현저히 우수하여 RCA 생성 품질을 크게 향상시켰습니다. LLM-only baseline의 LLM-as-a-judge 점수가 3.34에서 3.72로 증가하였습니다. 이러한 결과는 필드 인지 RAG가 해양 안전 조사 작업 흐름을 간소화하여 더욱 일관되며 증거 기반의 RCA 초안을 작성할 수 있음을 시사합니다.



### EPIG: Emotion-Based Prompting for Personalised Image Generation (https://arxiv.org/abs/2606.13247)
Comments:
          Submitted to arXiv. 20 pages, 4 figures. Work on emotion-based prompt engineering for text-to-image diffusion models with applications in personalized image generation

- **What's New**: 이번 논문에서는 감정 표현을 강화하기 위해 새로운 방법인 EPIG(Emotion-based Prompting for Image Generation)를 제안합니다. EPIG는 감정 관련 프롬프트를 개선하여 이미지 생성 이전에 감정적 의도를 명확히 표현할 수 있게 합니다. 본 방법은 추상적인 수정이나 모델 재훈련을 하지 않고도 감정 표현을 개선할 수 있는 경량의 솔루션입니다.

- **Technical Details**: EPIG는 정서적 역할 분해(affective role decomposition) 개념을 도입하여, 감정을 경험하는 주체(subject), 감정 상태의 원인(stimulus), 그리고 주변 환경(context)을 명확히 구분합니다. 이 방법은 감정적 용어와 확산 모델 간의 연결을 돕는 기하학적으로 기반한 기술을 활용하여, 특정 발란스-각성(valence-arousal) 상태에 따라 감정 용어를 선택합니다. 이러한 방식은 프롬프트의 감정적 구성 요소를 명확하고 일관되게 만들어, 시각적 결과물이 감정적으로 일치하도록 유도합니다.

- **Performance Highlights**: 실험 결과, EPIG는 10가지 다양한 프롬프트에 대해 평균 각성 오류(arousal error)를 14% 감소시켰으며, 이는 강력한 기준선 대비 향상된 수치입니다. EPIG는 감정 정렬과 의미적 일관성을 보존하는 데에도 성공적이며, 특히 인물이나 동물 등 명시적 주제가 포함된 프롬프트의 경우 17%의 감소를 보여줍니다. 이러한 결과는 통계적으로 유의미한 것으로 나타났습니다.



### Brick: Spatial Capability Routing for the Mixture-of-Models (MoM) Paradigm (https://arxiv.org/abs/2606.13241)
Comments:
          17 pages, 5 figures. Technical report

- **What's New**: 이 논문에서는 'Brick'이라는 다중 모달 라우터(multimodal router)를 소개합니다. Brick은 모델을 여섯 가지 능력 차원에서 평가하고, 각 쿼리에 대한 난이도 추정치를 결합하여 비용-패널티 기하학적 규칙을 통해 모델을 배치합니다. 이는 생산 환경에서 질과 비용 간의 트레이드오프를 조정할 수 있는 지속적인 선호 조정을 가능하게 합니다.

- **Technical Details**: Brick은 모델의 능력을 여섯 가지 차원에서 평가하며, 이를 바탕으로 각 쿼리에 가장 적합한 모델을 선택합니다. 기존의 단순한 라우팅 방식은 도메인 레이블, 키워드 및 토큰 수와 같은 표면적 특성에 의존하였으나, Brick은 이러한 한계를 넘어 내부 도메인 변동성을 고려합니다. 실험에서 Brick은 최대 품질 상태에서 76.98% 정확도로 최고의 단일 모델을 능가하며, 전반적인 비용과 지연 시간을 크게 절감합니다.

- **Performance Highlights**: Brick은 Dataset A에서 5,504개의 쿼리를 테스트하였고, 최적의 품질에서는 76.98%의 정확도를 달성하였습니다. 중립적인 비용-품질 프로필에서 Brick은 4.71배 더 낮은 비용으로 74.11%의 정확도를 기록하였으며, 최소 비용 상태에서는 22.15배의 비용 절감과 11.85 포인트의 정확도 손실을 보였습니다. 또한, 매출 시간은 51.2초에서 22.8초로 감소하여 성능을 크게 향상시켰습니다.



### LLM-as-an-Investigator: Evidence-First Reasoning for Robust Interactive Problem Diagnosis (https://arxiv.org/abs/2606.13220)
- **What's New**: 이 논문은 대형 언어 모델(LLM)의 사용자 주도 시콥언시(user-driven sycophancy) 문제를 해결하기 위해 LLM을 증거 기반의 조사의 도구로 사용하려는 새로운 방식인 LLM-as-an-Investigator를 제안합니다. 이 방법론은 초기 문제 설명에서 불확실성을 파악하고 후보 가설을 생성하며, 질문을 통한 명확화를 통해 문제를 해결하는 접근 방식을 사용합니다. 이를 통해 모델이 사용자 제공 가설을 조기에 받아들이는 경향을 줄이고, 더 신뢰성 있는 결과를 도출할 수 있게 됩니다.

- **Technical Details**: LLM-as-an-Investigator에서는 Solution Investigator Agent가 중심 역할을 합니다. 이 에이전트는 초기 문제 설명의 모호성을 추정하고, 진단을 위해 후보 솔루션을 생성한 뒤, 목표에 맞춘 질문을 통해 불확실성을 해소합니다. 이 과정에서 수집된 데이터를 바탕으로 가설 확률을 갱신하고, 증거가 확보될 때까지 즉각적인 대답을 제공하는 대신 지속적으로 조사를 실시합니다.

- **Performance Highlights**: 제안된 접근 방식은 기존의 LLM 대비 문제를 더 정확하게 식별하는 것으로 나타났습니다. 표준 보조 도구는 사용자의 잘못된 가설에 대해 자연스럽게 도전하는 경향이 거의 없지만, LLM-as-an-Investigator는 대안 가설을 유지하고, 차별화된 질문을 통해 사용자 제안을 가설로 간주하여 검증하는 방식을 채택했습니다. 이 논문의 결과는 검사 대상의 정확도 향상과 사용자 주도 시콥언시 감소를 보였습니다.



### Hallucination in Medical Imaging AI: A Cross-Modality Analytical Framework for Taxonomy, Detection, and Mitigation under Regulatory Constraints (https://arxiv.org/abs/2606.13211)
- **What's New**: 이 논문은 의학 영상에서 AI 시스템들이 빠르게 배포되는 반면 실패 모드에 대한 이해가 부족함을 강조합니다. 특히, 비현실적인 정보가 포함된 결과인 환각(hallucination)이 주요 우려 사항으로 지적됩니다. 본 연구는 다양한 이미징 모달리티를 아우르는 환각의 분류 체계, 원인, 탐지 및 완화 전략을 논의하며, FDA의 규제 가이드라인과 연계하여 지속 가능한 관리 방안을 제시합니다.

- **Technical Details**: 연구는 기존의 환각 분류체계가 단일 모달리티에만 적용되며, CT, MRI, PET/SPECT, 초음파, 디지털 병리 등 여러 모달리티를 통합한 통합 체계가 부족하다는 점을 지적합니다. 일반 목적의 AI 모델과 의료 전용 모델 간의 환각 발생 빈도를 비교하였고, 여러 탐지 및 완화 전략의 효과성을 평가했습니다. 여기에는 체계적인 벤치마크 및 연구 방법론이 포함되어 있으며, 외부 평가 기준에 따른 성과 분석이 이루어졌습니다.

- **Performance Highlights**: 연구 결과, 일반 목적 모델이 특정 환각 기준에서 의료 전용 모델보다 우수한 성능을 보였습니다. 이는 좁은 분야의 미세 조정이 과적합(overfitting)으로 이어질 수 있음을 시사합니다. 또한, 방사선 전문의의 검토가 필수적이며, AI가 생성한 경고의 일부가 전문가 수정을 요구한다는 점이 강조되었습니다. 물리 기반 설계 제약, 사고 연쇄(prompting), 인간-기계 협업은 각각 다른 실패 모드를 해결하며, 이들을 결합할 때 더욱 효과적입니다.



### A Minimal Model of Bounded Trade-Off Screening in Multi-Attribute Choic (https://arxiv.org/abs/2606.13201)
Comments:
          3 pages, 1 figure, accepted as extended abstract at Annual Conference on Cognitive Computational Neuroscience 2026

- **What's New**: 본 논문에서는 다속성(multi-attribute) 선택에서의 결정 과정을 설명하기 위해 새로운 경계 trade-off 스크리닝 모델을 제안합니다. 기존 모델들은 완전한 유틸리티 집계를 가정하지만, 본 연구는 결정 과정의 비선형성과 맥락 의존성을 반영합니다. 제안된 모델은 이득과 손실 간의 균형을 평가하는 스크리닝 프로세스를 통해 보다 현실적인 의사결정 과정을 설명합니다.

- **Technical Details**: 모델은 pairwise 비교를 기반으로 하며, max-min 비교 규칙을 통해 이득과 손실의 균형을 평가합니다. trade-off tolerance 파라미터 MM을 도입하여 결정 환경에 따라 수용 가능한 불균형의 정도를 조절합니다. 이 모델은 비선형 유틸리티와는 달리, 이득과 손실의 극단적인 신호를 압축하여 간단하고 계산적으로 효율적인 방식으로 다속성 선택을 설명합니다.

- **Performance Highlights**: 모델은 시뮬레이션을 통해 M-dominance가 기존 가중 합 유틸리티 및 체비셰프 유틸리티와는 다른 선호 패턴을 생성함을 보여주었습니다. 또 다른 중요한 발견으로는 결정 환경에 따라 선호가 변할 수 있다는 점으로, 혼합 trade-off 설정에서는 비선호 현상이 두드러지며 이는 기존의 보상적 혹은 비보상적 모델로는 설명할 수 없는 행동 변수를 잘 설명합니다.



### ARMOR-MAD: Adaptive Routing for Heterogeneous Multi-Agent Debate in Large Language Model Reasoning (https://arxiv.org/abs/2606.13197)
- **What's New**: 본 논문에서는 ARMOR-MAD라는 새로운 이질적 다중 에이전트 토론(Multi-agent Debate, MAD) 프레임워크를 제안합니다. 이 시스템은 조건부 계산을 통해 토론을 다루며, 이전의 고정된 토론 방식이 가지는 계산 낭비와 공통 오류를 증폭시키는 문제를 해결하고자 합니다. ARMOR-MAD는 세 가지 주요 구성 요소인 Pre-debate Agreement Routing (PAR), Early Agreement Stopping Evaluator (EASE), Semantic Outlier Detection (SOD)을 결합하여 에이전트 간의 상호작용을 최적화합니다.

- **Technical Details**: ARMOR-MAD는 고정된 다중 에이전트 토론 구조에서 벗어나, 에이전트의 이질성을 기반으로 토론의 필요성을 판단합니다. PAR은 독립적으로 생성된 초기 답변을 수집하고, 토론이 필요한 상황만을 위한 조건부 제어를 수행합니다. EASE는 에이전트들이 수렴할 경우 토론을 즉시 종료시키며, SOD는 최종 답변을 집계할 때 비정상적인 출력을 낮추는 신뢰도 가중치를 부여합니다.

- **Performance Highlights**: ARMOR-MAD는 MATH Level 5, GSM8K, MMLU, MMLU-Pro에서 고정된 다중 에이전트 토론에 비해 눈에 띄게 성능을 향상시켰습니다. 각각 65.5%, 96.5%, 90.0%, 81.5%의 정확도를 기록하였으며, 이는 모델의 진정한 이질성과 동의 기반의 제어가 MAD의 정확성과 효율성을 높이는 데 중요함을 나타냅니다.



### Under What Conditions Can a Machine Become Genuinely Creative? (https://arxiv.org/abs/2606.13196)
- **What's New**: 이번 연구는 머신이 진정으로 창의적이기 위해 필요한 조건과 인간의 주체성을 어떻게 보존할 수 있는지를 탐구합니다. 창의성의 정의는 단순히 새로운 결과물을 생성하는 것이 아니라, 불완전한 상황을 구조적으로 변화시키는 과정으로 이해됩니다. 논문은 그 과정이 재귀적 개입 역학(recursive intervention dynamics)을 통해 이루어져야 한다고 강조합니다.

- **Technical Details**: 연구는 창의적인 머신(Creative Machine)의 요구 사항을 열 가지로 정의합니다: 환경 표현(environment representation), 범위 있는 지각(scoped perception), 갈등 식별(conflict identification), 개입 능력(intervention capability), 결과 관찰(consequence observation), 지식 및 환경 업데이트(knowledge and environment update), 재조정(rescoping), 지역에서 세계로의 전개(local-to-global unfolding), 가치 기반 범위 조정(value-based scoping), 그리고 인간-AI 공동 생활(human-AI co-living)입니다. 이들은 인지적 및 창조적 환경에서의 의미 있는 변화에 기초하며, 세 가지 디자인 법칙(perception, conflict, capability)에 의해 조직됩니다.

- **Performance Highlights**: 결론적으로, 논문은 현대 AI 기술들이 창의적 결과물 생성을 넘어서는 진정한 머신 창의성을 탐구해야 한다고 주장합니다. 연구는 사이버 물리적 및 사이버 생물학적 사례를 통해 이러한 요구 사항의 계산 가능성을 보여주며, 책임 있는 AI 윤리(proactive AI ethics)가 진정한 머신 창의성의 내부 요구 사항임을 강조합니다. 궁극적으로, 창의적 머신은 단순히 아티팩트를 생성하는 것이 아니라 인간 환경에 개입함으로써 영향을 미친다는 점을 시사합니다.



### Reasoning for Mobile User Experience with Multimodal LLMs: Task, Benchmark, and Approach (https://arxiv.org/abs/2606.13192)
Comments:
          10 pages, 6 figures, Accepted at CVPR 2026 Findings

- **What's New**: UXBench라는 새로운 다중 모달 벤치마크가 제안되었습니다. 이 벤치마크는 UI 기반 추론을 평가하기 위해 설계된 2,000개의 VQA 데이터 샘플을 포함하고 있습니다. UXBench는 레이아웃 관계, 시각적 계층 구조 및 콘텐츠 일관성에서 UX 문제를 세밀하게 진단하는 8개의 작업을 포함합니다. 이를 통해 MLLM의 UI 기반 추론 능력을 평가하는 첫 번째 기회를 제공합니다.

- **Technical Details**: UXBench는 실제 UI 스크린샷을 기준으로 하며, 사용자 피드백을 통해 UX 문제를 세 가지 차원(Usability, Efficiency, Trustworthiness)으로 분류합니다. 각 작업은 인과 추론이 필요한 두 가지나 세 가지 선택 질문으로 제시되며, 키워드 일치가 아닌 디자인 원리에 매핑되는 논리적 사고를 요구합니다. UI-UX는 Qwen3-VL-4B-Thinking 기반의 MLLM으로, 두 가지 주요 혁신을 통해 향상됩니다: 지각적 이해와 논리적 추론을 동적으로 조절하는 보상 라우팅 메커니즘과 중복 추론 단계를 억제하는 비대칭 전이 보상입니다.

- **Performance Highlights**: UI-UX는 UXBench에서 0.7963의 정확도를 달성하며 Claude-4.5-Sonnet의 0.6550를 초과하는 성과를 나타냅니다. 또한 다양한 UI 작업에서 강력한 일반화 능력을 증명하며, 낮은 추론 지연 시간을 유지합니다. 이는 MLLM이 인간 전문가를 초월한 UX 추론 성능을 갖추었다는 것을 시사합니다.



### Mental-R1: Aligning LLM Reasoning for Mental Health Assessmen (https://arxiv.org/abs/2606.13176)
- **What's New**: 본 논문은 정신 건강 평가를 위한 Cognitive Relative Policy Optimization (CRPO)라는 강화 학습 프레임워크를 제안합니다. 기존의 일반적인 사후 훈련 방법들이 인간 평가의 인지 과정과 맞지 않아 신뢰도가 떨어지는 문제를 해결하고자 합니다. CRPO는 그룹 상대 정책 최적화를 확장하여 단계별 불확실성 모델링을 통합하여 정책 최적화 과정에서 인지 과정을 더욱 잘 반영합니다.

- **Technical Details**: CRPO에서는 단계별 엔트로피 정규화 메커니즘을 도입하여 초기에 넓은 탐색을 장려하고 후반에는 자원 의사 결정을 강화합니다. 이는 인간의 인지가 불확실성에서 확실성으로 이동하는 과정을 모방합니다. 추가적으로, 인지 평가 이론에서 영감을 받아 인지 추론 단계를 형식화하였으며, 이를 통해 이론에 기반한 해석 가능성을 강조합니다.

- **Performance Highlights**: CRPO는 8개의 정신 건강 데이터셋에서 기존의 강화 학습 기준선보다 평균 10.4% 포인트 향상된 weighted F1-score를 달성하였습니다. CRPO로 훈련된 모델인 Mental-R1은 기존의 대형 언어 모델보다 약 15.6 포인트 높은 성능을 보이며, 정신 건강 평가에 있어 모델의 추론 능력을 효과적으로 향상시키는 것으로 나타났습니다.



### TerraBench: Can Agents Reason Over Heterogeneous Earth-System Data? (https://arxiv.org/abs/2606.13148)
- **What's New**: 본 논문은 TerraBench라는 새로운 기준을 소개하며, 이는 환경 과학을 위한 통합된 증거 기반 추론을 가능하게 합니다. 이전 벤치마크들은 개별적인 작업들로 기능을 분리했지만, TerraBench는 그리드 형태의 데이터, 위성 이미지, GIS 추론 및 시뮬레이션을 하나의 실행 가능 인터페이스로 통합합니다. 또한, TerraBench는 프로세스 수준의 도구 사용 메트릭과 허용 오차를 감안한 수치 평가를 결합하여 새로운 평가 프로토콜을 제시합니다.

- **Technical Details**: TerraBench는 3개의 트랙과 8개의 응용 분야에서 403개의 복잡한 에이전트 작업으로 구성되며, 24,500개의 검증된 실행 단계를 포함하고 있습니다. TerraAgent는 ReAct 스타일의 도구 증강 프레임워크로, 기후 질문에 대한 계획을 세우고, 전문 과학 도구를 호출하며, 중간 산출물을 물리적으로 생성합니다. 이 프레임워크는 언어 기반 계획과 과학적 실행을 분리하여, 데이터 검색, 지리적 처리 및 시뮬레이션과 같은 양적 출력에서 신뢰성을 높입니다.

- **Performance Highlights**: TerraBench의 시험 결과는 현재의 프론티어 모델과 오픈 모델 모두에게 도전적임을 보여줍니다. 예를 들어, Claude Sonnet 4.6은 ToolUseScore 59.22를 기록했으며, Qwen3.5-35B는 39.95를 기록했습니다. 시뮬레이터 기반 작업이 특히 어려운 것으로 나타났으며, 실패 모드에는 잘못된 인수값 및 도구의 순서가 포함됩니다.



### Rethinking RAG in Long Videos: What to Retrieve and How to Use It? (https://arxiv.org/abs/2606.13141)
- **What's New**: 이 논문은 비디오 인식 및 생성 문제를 새로운 차원으로 끌어올리는 Retrieval-augmented generation (RAG) 기술의 최근 발전을 다루고 있습니다. VideoRAG 설정에서, 시스템이 긴 비디오에서 쿼리에 적합한 증거를 선택해야 하며, 이는 멀티모달 표현과 다양한 시간적 세분화 수준을 아우르는 복잡성을 증가시킵니다. 이를 해결하기 위해 V-RAGBench라는 새로운 벤치마크와 CARVE라는 방법론을 소개하여 더 효과적인 평가와 생성이 가능하도록 합니다.

- **Technical Details**: V-RAGBench는 2,100개의 고품질 triplet인 ⟨query, evidence chunk, answer⟩로 구성되며, 에고4D 및 에고라이프의 긴 개인 비디오로부터 추출되었습니다. 이 벤치마크는 기존 데이터셋들이 충족하지 못한 세 가지 속성을 공동으로 enforcement 하여 각 쿼리와 관련된 증거가 고유하며 이를 바탕으로 생성이 검색에 의존하도록 합니다. CARVE 방식은 다중 모달 크로스 인코더 신호를 활용하여 효과적인 청크 수준 구성 결정을 가능하게 합니다.

- **Performance Highlights**: 조사 결과, CARVE는 Retrieval 및 Generation 두 가지 단계 모두에서 8개의 기존 방법들을 능가하는 성능을 보였습니다. 또한, 각 단계에서 모달리티-세분화의 차이가 존재하는 사실과 청크 수준에서 결정하는 것의 중요성을 확인하였습니다. CARVE의 선택은 단일 구성으로 수렴되지 않고 다양한 구성 간의 분포를 드러내며, 생성 단계에서조차도 최적의 성능을 발휘했습니다.



### AAbAAC: An Annotated Corpus for Autoimmunity Information Extraction (https://arxiv.org/abs/2606.13051)
- **What's New**: 이 연구에서는 AAbAAC(자기항체 및 자가면역 주석 말뭉치)를 소개하며, 이는 PubMed에서 엄선된 115개의 초록으로 구성된 말뭉치입니다. 자가면역 분야에서 가장 중요한 개체인 자가면역 질환과 자가항체를 포함하여 이들 간의 관계를 수동으로 주석 처리했습니다. AAbAAC는 명명된 개체 인식(NER) 작업에 대한 여러 방법을 평가하고 NER 모델을 조정하는 데 사용되었습니다.

- **Technical Details**: 자체 항체는 자가면역 질환의 바이오마커로 사용되며, 이 연구에서는 이러한 자가항체명과 질병명을 기반으로 자원 부족 문제를 해결하기 위해 AAbAAC를 구축했습니다. 연구팀은 HPO 용어를 기반으로 한 자가항체 명칭 사전을 통해 PubMed API를 사용하여 관련 논문을 검색했습니다. 총 10,916개의 변형이 포함된 사전을 통해 56,750개의 제목 및 초록이 수집되었습니다.

- **Performance Highlights**: AAbAAC의 유용성은 NER 성능을 향상시키는 방법을 통해 입증되었습니다. 조정 후 NER 성능의 향상이 기대되며, 이는 전문 분야를 위한 소규모 주석 작업의 가치를 보여줍니다. 자가면역 분야의 정보 추출 연구를 지원하는 데 있어 AAbAAC가 중요한 기여를 할 것으로 보입니다.



### Augmentation techniques for video surveillance in the visible and thermal spectral rang (https://arxiv.org/abs/2606.13042)
Comments:
          8 pages

- **What's New**: 이 논문은 다중 스펙트럼(Multispectral) CNN 기반의 객체 탐지를 다루고 있으며, 시각 스펙트럼(VIS)과 열 적외선(IR) 이미지를 동시에 활용하여 보안 및 방어 분야에서의 객체 인식 성능을 향상시키는 방법을 제안합니다. 특히, 다른 센서의 데이터를 결합하여 CNN 훈련을 최적화하고, 그러한 데이터로부터 어떻게 CNN이 결정을 내리는지를 분석합니다. 기존의 학습 방식의 한계를 극복하기 위해 VIS 이미지를 IR 이미지처럼 변환하는 여러 가지 기법을 적용합니다.

- **Technical Details**: 본 연구에서는 IR 및 VIS 이미지를 포함한 튜플 입력(i,v)을 사용하며, 주어진 이미지의 픽셀 값을 조정하여 색상 채널을 최적화합니다. 이러한 과정에서는 Luminance, Intensity, Value 등의 다양한 그레이스케일 변환 방식과 Gaussian blur augmentation 기법이 활용됩니다. CNN 모델은 두 개의 컨볼루션 레이어와 풀링 레이어 및 완전 연결 레이어를 포함하여 이미지 분류 성능을 극대화할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 훈련된 모델은 IR 이미지에서 분류 성능이 약 76%, VIS와 열 적외선 이미지의 조합에서 24% 향상된 성과를 보였습니다. 데이터 증강 없이도 단순 VIS 이미지로 학습했을 때의 성능은 변화가 없었습니다. 이러한 성과는 CNN이 입력 데이터로부터 학습한 내용에 대한 통찰을 제공하며, 필터 커널 분석을 통해 어떤 특징을 학습하는지 비교합니다.



### Nous: An Attempt to Extract and Inject the Cognition Behind Prediction-Market Behavior (https://arxiv.org/abs/2606.13038)
Comments:
          37 pages, 1 figure, 7 tables. Reproduction artifacts (code, frozen profiles, prompts, model outputs): this https URL

- **What's New**: 이 논문은 LLM(대규모 언어 모델) 에이전트가 예측 시장과 집단적 의사결정에서 어떻게 인지 단일 문화(cognitive monoculture)의 위험을 초래하는지를 다룬다. 연구에서는 고유한 인지 다양성이 LLM 에이전트에 어떻게 적용될 수 있는지를 탐구하며, 인간 거래 활동에서 구조화된 행동 프로필을 추출하고 이를 에이전트에 주입하는 방식을 사용한다. 이 연구의 핵심 발견은 인지 단일 문화 문제를 측정하고, 프롬프트(prompt) 수준의 해결 방안의 한계를 확인한 점이다.

- **Technical Details**: 연구에서는 8차원의 구조화된 행동 프로필을 Polymarket 거래 활동에서 추출하고, 이를 LLM 에이전트에 주입하는 방법을 분석한다. 총 100개의 지갑 데이터에서 14개의 파라미터 중 8개가 시간적으로 안정적이며, 이들로부터 지갑의 프로필을 17-22%의 확률로 식별할 수 있었다. 그러나 프롬프트 수준의 주입은 구조화된 내용을 효과적으로 전파하지 못했으며, 그 결과 집단 예측의 정확도를 향상시키지 못했다.

- **Performance Highlights**: Injected personas (주입된 응답자) 는 에이전트의 출력을 변화시켰지만, 구조화된 프롬프트가 정보 전이의 유의미한 효과를 나타내지 않았다. 결과적으로, 주입으로 인해 생성된 출력 다양성은 집단 예측을 개선시키지 않았으며, 약간의 출력 다양성 증가에도 불구하고 집단 오류 상관관계는 감소하지 않았다. 이 연구는 프롬프트와 그 구조 전환 과정에서의 한계를 강조하며, 더 나은 해결책으로 프롬프트 아래의 방법론을 제안한다.



### SciR: A Controllable Benchmark for Scientific Reasoning in LLMs (https://arxiv.org/abs/2606.13020)
- **What's New**: 이 논문에서는 과학적 추론에 관련된 세 가지 패러다임형태인 deduction, induction, causal abduction을 평가하기 위한 새로운 벤치마크인 SciR을 소개합니다. 기존의 과학 벤치마크는 인간 주석에 기반하여 비용이 많이 들고 메커니즘적 진실이 부족하였으며, 합성 논리-추론 벤치마크는 실제 과학 문서와 다릅니다.

- **Technical Details**: SciR은 formal objects(정형 객체)인 deduction tree(유도 트리), inductive rule hypothesis(유도 규칙 가설), causal graph(인과 그래프)에서 생성된 작업을 통해 검증 가능한 답변을 제공합니다. 또한 과학적 담론의 여러 문서를 생성하는데 있어서 각 트랙별 도메인에 맞춰 조정된 장르를 활용하여 두 개의 난이도 축을 독립적으로 조정할 수 있습니다.

- **Performance Highlights**: 여섯 가지 모델을 시험한 결과, 두 난이도 축은 모든 모델에 부정적인 영향을 미치며 그 효과는 합산된다는 것을 확인했습니다. Reasoning 모델인 deepseek-r1은 비추론 인스트럭션 모델보다 추론 축에서 대부분 우수한 성능을 보였음을 확인했습니다. SciR은 extraction(추출)과 inference(추론) 난이도 모두를 파라메트릭하게 제어할 수 있는 최초의 다중 패러다임 과학적 추론 벤치마크입니다.



### Otters++: A Time-to-first-spike Based Energy Efficient Optical Spiking Transformer (https://arxiv.org/abs/2606.13016)
- **What's New**: 이 논문은 스파이킹 신경망(Spiking Neural Networks, SNNs)의 효율성을 높이기 위해 음성 신호 감쇠를 활용한 새로운 TTFS(time-to-first-spike) 계산 메커니즘인 Otters++를 소개합니다. 이 방법은 자연의 물리적 신호 감쇠를 활용하여 디지털 감쇠 계산의 필요성을 제거하여 처리 효율성을 단순화하고 비용을 절감합니다. 또한, Otters++는 Transformer 모델에 맞게 레이어별 기능 동등성을 설정하고, 하이브리드 훈련 방법을 개발하여 기존 SNN의 훈련 문제를 해결합니다.

- **Technical Details**: Otters++는 맞춤형 In$_2$O$_3$ 광전자 시냅스의 측정된 감쇠 응답을 TTFS 시간 항으로 직접 활용하여 감쇠 함수의 명시적 계산을 제거합니다. 이로 인해, 네트워크의 훈련 소요를 줄이고, 여러 하드웨어 변동에 대한 강건성을 높이며, 효율적인 SNN-전방 / QNN-후방 훈련 접근 방식을 채택합니다. 이러한 하이브리드 훈련 방식은 매핑된 QNN을 통해 안정적인 최적화를 가능하게 합니다.

- **Performance Highlights**: Otters++는 GLUE 데이터셋에서 평균 점수를 84.17%로 향상시키며, 기존 스파이킹 Transformer 모델 대비 에너지 이점을 명확히 유지합니다. 해당 프레임워크는 SpikingLM, Sorbet 및 SpikingBERT에 비해 레이어당 에너지를 각각 1.84배, 3.02배 및 최대 5.68배까지 줄입니다. 이 연구는 물리적으로 기반한 TTFS 컴퓨팅이 효율적이고 훈련 가능하며, 실제 하드웨어 영향에 강한 성능을 보여줍니다.



### The Illusion of Multi-Agent Advantag (https://arxiv.org/abs/2606.13003)
- **What's New**: 이 논문은 자동 생성된 다중 에이전트 시스템(MAS)이 단일 에이전트 시스템(SAS)보다 잘 작동한다는 기존 이론에 의문을 제기합니다. 연구자는 자동 MAS가 실제로 SAS인 Chain-of-Thought with Self-Consistency (CoT-SC)보다 성능이 떨어진다는 것을 입증했습니다. 특히, 연구에서는 기존의 평가 프레임워크가 다중 에이전트 시스템의 중요한 설계 문제를 간과하고 있다는 사실도 밝혔습니다.

- **Technical Details**: 이 연구에서는 MAS의 성능을 평가하기 위해 진단용 합성 데이터셋(Synthetic Multi-Hop Financial Reasoning, SMFR)을 도입했습니다. SMFR은 명확한 하위 작업 구조와 맥락 분리, 병렬화 가능성을 제공하여 자동 MAS의 성능을 평가할 수 있도록 설계되었습니다. 또한, 전문가가 설계한 MAS는 이러한 구조가 잘 갖춰질 경우 MAS의 이점이 실제로 발생할 수 있음을 보여주었습니다.

- **Performance Highlights**: 연구 결과, 자동 MAS는 많은 설정에서 SAS의 성능에 미치지 못하며 오히려 비용 효율성 면에서 CoT-SC의 성능이 뛰어남을 보여주었습니다. 전문가 설계 MAS는 구조적 체계를 갖출 경우 복잡한 작업에서도 뛰어난 성능을 발휘하는 것으로 나타났습니다. 전체적으로, 자동 MAS의 성능 이점이 복잡성의 표면적 요소에 의해 발생하는 경향이 있음을 발견하였습니다.



### APCyc: Property-Informed Design of Cyclic Peptides via Automated Cyclization (https://arxiv.org/abs/2606.12991)
Comments:
          Accepted at the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD 2026)

- **What's New**: 이번 연구에서는 APCyc라는 새로운 프레임워크를 도입하여, 사이클릭 펩타이드를 de novo 방식으로 생성하는 혁신적인 접근법을 제안합니다. APCyc는 사이클화(cyclization)를 명시적으로 모델링하여 여러 물리화학적(property) 속성을 동시에 최적화할 수 있는 기능을 제공합니다. 이를 통해 연구자들은 약물 디자인에서 요구되는 다양한 제약을 고려할 수 있는 사이클릭 펩타이드 후보를 자동으로 생성할 수 있게 됩니다.

- **Technical Details**: APCyc는 수신체(binding pocket) 및 사이클화 관련 정보를 명시적으로 인코딩하여 사이클화에 적합한 표현을 학습합니다. 이 모델은 잔여물의 자원 어휘(residue vocabulary)를 확장하여, 사이클화에 관여하는 잔여물과 비사이클화 잔여물을 구분하고, 이는 저체적 확산(latent diffusion model) 네트워크에 통합됩니다. 또한, 물리적 속성에 대한 최적화를 위해 에너지 기반 대리모델을 훈련하여, 다수의 속성 목표를 위한 샘플링 과정의 방향성을 제공합니다.

- **Performance Highlights**: APCyc를 통해 연구자들은 최적의 사이클화 전략을 수신체의 구조적 맥락에 따라 자동으로 선택할 수 있습니다. 이 모델은 약물에 필요한 속성들을 조절 가능하게 생성할 수 있으며, 가장 우수한 투과성 및 프로테이즈 저항 점수를 기록했습니다. APCyc는 사이클화 선택 및 완전 원자(all-atom) 생성과 속성 지향 가이드를 통합하여, 구조적 품질을 높이는 데 성공했습니다.



### Structured Testbench Generation for LLM-Driven HDL Design and Verification-Oriented Data Curation (https://arxiv.org/abs/2606.12983)
Comments:
          9 pages, 10 figures

- **What's New**: 이 논문에서는 STG, 즉 구조화된 테스트벤치 생성 프레임워크를 제안합니다. 기존의 프롬프트 기반 접근 방식이 구문적으로 제약 없는 코드 생성을 다루며, 비결정적 결과물과 높은 토큰 비용 때문에 신뢰성과 커버리지가 부족한 문제점을 해결하고자 합니다. STG는 하드웨어 디자인의 내재된 구조를 활용하여 결정론적 테스트벤치를 생성하여 더욱 빨리 검증할 수 있게 지원합니다.

- **Technical Details**: STG는 조합적(combinational) 및 일반 순차적(general sequential) 디자인에 대해 경량 HDL 분석과 템플릿 기반 렌더링을 조합하여 테스트벤치를 생성하는 것을 목표로 합니다. 이 프레임워크는 직접 RTL 검증, 검증 지향 데이터 큐레이션, 테스트 시간 스케일링의 세 가지 시나리오를 지원하며, RTL 워크플로우에서의 일반적인 검증 백엔드로 자리 잡고 있습니다. STG는 기존 LLM 기반 테스트벤치 생성 파이프라인보다 720배 더 빠르며, 오판 판단을 줄이고, 커버리지를 증가시킵니다.

- **Performance Highlights**: STG는 CPU 코어에서 LLM 기반 필터링보다 10.6배 더 빠르며, 에너지를 127배 덜 소비합니다. 또한, 여러 벤치마크 평가에서 경량화된 모델들이 최신 성능을 보여줍니다. LLM 기반 시스템에 통합하였을 때, 테스트 시간 스케일링에서 문제 해결 노드 수를 14-47% 줄이는 효과를 보입니다.



### A Mathematical Forum Platform for Collaborative Problem Solving and Dataset Generation for AI Reasoning (https://arxiv.org/abs/2606.12976)
Comments:
          11 pages, 3 figures

- **What's New**: 이 논문은 수학식을 온라인 포럼에 공유하는 과정에서 발생하는 여러 문제를 해결하기 위한 통합 시스템을 제안합니다. 기존의 수단들은 수학 식을 올리기 위해 여러 단계를 요구하고, 종종 오류가 발생할 수 있었습니다. 그러나 이 시스템은 이미지 캡처, 수식 인식 및 포스팅을 원활하게 연결하여 사용자 경험을 크게 향상시킵니다. 사용자는 이미지를 업로드하면 실시간으로 LaTeX 또는 Markdown 형식으로 미리 보기를 볼 수 있습니다.

- **Technical Details**: 제안된 시스템은 주로 세 가지 레이어로 구성되어 있습니다: 이미지 처리, 렌더링 및 저장. 이미지가 업로드되면, 시스템은 Mathpix OCR API를 통해 수학적 내용을 추출하고, LaTeX로 변환하며, 결과를 포스트 본문에 삽입합니다. 이 과정은 사용자 개입 없이 자동으로 진행되며, 다양한 포맷을 지원하는 이중 렌더링 시스템을 통해 사용자의 의도를 보존합니다.

- **Performance Highlights**: 이 플랫폼의 또 다른 장점은 사용자들이 게시하는 모든 문제와 솔루션이 자동으로 구조화된 데이터 기록으로 생성된다는 점입니다. 이러한 데이터는 AI 시스템이 수학 문제 해결을 정확하게 학습하고 평가하는 데 사용될 수 있는 귀중한 자원이 됩니다. 대규모로 확장될 경우, 이러한 문제-해결 쌍의 축적은 수학적인 데이터셋의 질적 개선에 기여할 것입니다.



### Multi-Modal Agents for Power Distribution Defect Detection: An Evaluation of Foundation Models (https://arxiv.org/abs/2606.12969)
- **What's New**: 이 논문은 전력 분배 결함 감지를 위해 다중 모달 에이전트 프레임워크를 제안합니다. 이 시스템은 전통적인 검사 방법의 한계를 극복하고, 인식(Perception), 추론(Reasoning), 도구 사용(Tool Usage)이라는 세 가지 주요 능력을 평가합니다. 이를 통해 무인화 검사의 가능성을 탐구하고, 실험 결과를 통해 현재의 기초 모델들이 어떻게 작동하는지를 분석합니다.

- **Technical Details**: 에이전트 아키텍처는 세 가지 주요 구성 요소로 이루어져 있으며, 기초 모델(Foundation Model)은 다양한 정보를 처리하고 세 가지 핵심 기능인 인식, 추론, 도구 사용을 구동합니다. 입력 및 출력 메커니즘은 고해상도 이미지와 자연어 지시문을 함께 처리하여 에이전트가 환경과 상호작용하게 합니다. 또한, 전문화된 전력 도메인에 맞는 프롬프트 구성을 통해 에이전트의 행동을 조정하는 엄격한 전략을 구현했습니다.

- **Performance Highlights**: 이 연구는 다중 모달 에이전트의 성능을 격리된 과제가 아닌, 복합적이고 산업 현장의 요구에 맞춰 평가하는 혁신적인 프레임워크를 제공합니다. 실험 결과는 현재의 기초 모델들이 실제 환경에서의 퍼포먼스에 대한 강점과 제한점을 드러내며, 향후 기술 선택 및 최적화의 방향성을 제시합니다.



### OpenMedQ: Broad Open Pretraining for Medical Vision-Language Models (https://arxiv.org/abs/2606.12953)
Comments:
          Medical Imaging with Deep Learning (MIDL) 2026, Short Paper Track

- **What's New**: OpenMedQ는 14개의 다양한 데이터셋(약 3.35M 샘플)을 기반으로 선행 학습된 의료 비전-언어 모델이다. 기존의 의료 VLM들 대부분이 제한된 선행 데이터에 의존했던 것과 달리, OpenMedQ는 완전 공개된 데이터셋을 활용하여 모델을 개발하였다. 이 연구는 특히 PathVQA에서 75.9 BLEU-1 점수를 기록하며, 기존의 대형 모델들보다 뛰어난 성능을 보여준다.

- **Technical Details**: OpenMedQ의 비전 인코더는 BiomedCLIP에서 초기화된 ViT-base-patch16-224 모델을 사용한다. 이 인코더는 LLaMA-7B 언어 모델과 결합되어 이미지 및 텍스트 토큰을 처리한다. 훈련은 AdamW 옵티마이저와 배치 크기 64, 학습률 5×10^-5를 사용하여 최대 15 에폭 동안 진행된다.

- **Performance Highlights**: OpenMedQ는 CXR8, MedFMC, Breast-Ultrasound 등 8개의 벤치마크에서 평균 매크로 F1 점수 0.757로 최고 성과를 기록하였다. PathVQA에서는 75.9 BLEU-1 점수를 달성하여 Med-PaLM M 모델보다 월등한 성능을 보였으며, VQA-MED에서도 64.5의 점수를 기록하였다. 이러한 결과는 OpenMedQ의 선행 학습이 다양한 의료 데이터로부터 많은 이점을 가져왔음을 보여준다.



### Learning What to Remember: A Cognitively Grounded Multi-Factor Value Model for Agentic Memory (https://arxiv.org/abs/2606.12945)
Comments:
          11 pages, 3 figures

- **What's New**: 이 논문은 LLM (Large Language Model) 에이전트가 기억할 데이터의 선택과 유용성을 평가할 수 있는 다중 요인 메모리 가치 함수(multi-factor memory value function)를 제안합니다. 이 함수는 감정 강도(emotional intensity), 목표 관련성(goal relevance), 가치 정렬(value alignment) 등 일곱 개의 해석 가능한 요인으로 구성되어 있으며, 각 요인의 가중치는 학습을 통해 결정됩니다. 연구 결과, 이 새로운 접근법은 이전의 단일 요인 접근법보다 높은 기억 유지를 보여주었습니다.

- **Technical Details**: 제안된 메모리 가치 함수 V(m)는 총 7개의 설명 가능한 요인을 가중치가 결합된 형태로 나타냅니다. 각 요인은 미래 행동에 대한 메모리의 예상 유용성에 기반하여 구성됩니다. 학습된 가중치는 그래디언트 기반 최적화(gradient-free optimization)를 통해 얻어졌으며, 이는 메모리에 있어 인지 심리학의 원칙을 따른 것입니다.

- **Performance Highlights**: LongMemEval 벤치마크에서 제안된 접근법은 0.770±0.011의 금지 증거(gold evidence) 유지를 기록하며, 이는 기존의 단일 요인 방식을 초월하는 성과입니다. 이 연구는 메모리 관리에 있어 보다 정교한 접근법 필요성을 부각시키며, 단순한 유사성 또는 최신성만으로는 기억 유지의 우수성을 설명하기 어렵다는 점을 강조합니다. 결과적으로 인터프리터블한(weighted) 메모리 가치가 의미 있는 메모리 관리의 기초가 될 수 있음을 시사합니다.



### PRISMR: Overcoming Parse Collapse in Multimodal Listwise Ranking via Parameterized Representation Internalization (https://arxiv.org/abs/2606.12942)
- **What's New**: 이 논문에서는 PRISMR (Parameterized Representation Internalization for Semantic Multimodal Ranking)라는 새로운 프레임워크를 제안하여 기존의 Generative listwise ranking의 한계를 극복합니다. 특히, 긴 맥락의 멀티모달 환경에서 발생하는 'parse collapse' 문제를 해결하는 데 중점을 두고 있습니다. PRISMR은 경량 하이퍼네트워크를 활용하여 멀티모달 후보를 병렬로 인코딩하고, 항목별 LoRA 가중치를 생성하여 성능을 향상시킵니다.

- **Technical Details**: PRISMR의 핵심은 전통적인 인-context 리스트 처리를 파라메트릭 구조 조건으로 전환하는 것입니다. 하이퍼네트워크는 긴 멀티모달 리스트를 한 번의 전방 패스를 통해 Low-Rank Adaptation (LoRA) 업데이트로 매핑합니다. 이 방식은 'parse collapse'를 크게 줄이고, 디코딩 과정에서 높은 포맷 신뢰성을 유지하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, PRISMR은 멀티모달 리스트와이즈 랭킹에서 새로운 상태-of-the-art를 성립하였으며, 리스트 구조의 내부화를 더 강력하게 수행합니다. 또한, PRISMR은 다양한 도메인과 instruction-tuned 백본 간에 효과적으로 전이할 수 있는 성능을 보여주었습니다. 나아가, Amazon Reviews 2023에서 구축한 대규모 멀티모달 리뷰-랭킹 벤치마크를 통해 평가됩니다.



### MARS: Margin-Adversarial Risk-controlled Stopping for Parallel LLM Test-time Scaling (https://arxiv.org/abs/2606.12935)
- **What's New**: 본 논문에서 제안한 MARS(Margin-Adversarial Risk-controlled Stopping) 방법은 검사 중 현재의 투표 결과가 나중에 어떻게 바뀔지를 예측하여 안전하게 조기 종료를 가능하게 하는 혁신적인 접근 방법입니다. 이 방법은 중간 체크포인트에서 진행 중인 추적들의 답변을 확인하여, 그 중에서 안정성이 높은 수를 정의합니다. MARS는 투표 결과의 위험을 체계적으로 평가해 알맞은 시점에 결과를 뽑아낼 수 있도록 해줍니다.

- **Technical Details**: MARS는 기본적으로 세 가지 단계로 구성되어 있습니다. 첫째, 각 활성 추적(active trace)의 현재 답변을 조사하고, 둘째, 각 추적이 최종 답변에 도달하기 전에 변경될 확률을 평가합니다. 셋째, 각 예상되는 변경에 의해 발생할 수 있는 최대 마진 손실을 계산합니다. 이를 통해 MARS는 투표 안전성을 확립하고, 결과적으로 조기 종료를 보장합니다.

- **Performance Highlights**: MARS를 적용한 결과, 25-47%의 자기 일관성 토큰을 절약하면서도 기존의 DeepConf Online baseline과 유사한 정확도를 유지했습니다. 특히, Parallel-Probe와 비교했을 때 MARS는 9-35%의 정확도를 포기하지 않고도 유사한 절약 효과를 나타냈으며, 이는 고차원의 문제에서도 효과적입니다. 이러한 결과는 조기 종료의 안전성을 보장하는 MARS의 원리가 기존의 합의 기반 방법보다 우수함을 입증합니다.



### Iterating Toward Better Search: A Two-Agent Simulation Framework for Evaluating Agentic Search Architectures in E-Commerc (https://arxiv.org/abs/2606.12924)
- **What's New**: 이 논문에서는 대화형 쇼핑 도우미 아키텍처를 평가하기 위한 모듈형 2-에이전트 시뮬레이션 프레임워크를 제안합니다. 독립적인 바이어 에이전트가 다양한 페르소나(personas)와 미션(missions)에 따라 구성되어 있으며, 실시간 전자상거래 검색 API와 통합된 응답자를 교체하여 실험을 진행하는 방식입니다. 이는 바이어를 일정하게 유지하면서 응답자 디자인을 동일한 시나리오에서 비교할 수 있게 합니다.

- **Technical Details**: 시스템 아키텍처는 구매자 쿼리를 응답자에게 전달하고 응답을 반환하며 대화 통계와 기록을 관리하는 오케스트레이터로 구성됩니다. 구입자는 Gemini 2.5 Pro LLM을 사용하며, 각 세션은 1-3개의 독립적인 미션으로 구성되어 다양한 쇼핑 목표와 커뮤니케이션 톤을 정의합니다. 이 프레임워크는 바이어의 설정이나 구성에 따라 반복 진행 시 항상 동일한 성능 차이를 보장합니다.

- **Performance Highlights**: 이 논문은 2011개의 대화 데이터를 사용하여 rolling-window memory가 intent-extraction memory보다 모든 품질 메트릭에서 우수하다는 것을 증명하며, 쿼리에 대해 35% 더 빠른 성능을 보입니다. 또한, 응답자 아키텍처에서의 체계적 실패 분석을 통한 개선이 실패와 근접 실패율을 62% 감소시킨 사례를 보여주며 LLM의 기여가 아키텍처와 독립적이라는 점을 강조합니다.



### MDForge: Agentic Molecular Dynamics Pipeline Design under Sparse Simulator Feedback (https://arxiv.org/abs/2606.12916)
- **What's New**: 이번 연구에서는 MDForge라는 LLM(대형 언어 모델) 기반 에이전트를 설계하여 분자 동역학(Molecular Dynamics, MD) 파이프라인 설계를 자동화하는 혁신적인 접근 방식을 제안합니다. 이 시스템은 기존의 MD 에이전트들이 정해진 도구를 사용하는 것과는 달리, 개방형 코드 생성을 통해 에이전트의 행동을온라인으로 조정할 수 있습니다. MDForge는 물리 전문가들 간의 토론을 통해 희소 보상을 밀집된 학습 신호로 변환하는 방법을 사용하여, 실험과 유사한 성능을 달성할 수 있습니다.

- **Technical Details**: MDForge는 인컨텍스트 업데이트 규칙인 PRISM(프로세스-보상 해석을 통한 하위 시스템 중재)을 기반으로 하여, MD 파이프라인의 스테이지(준비, 평형화, 생산 샘플링, 분석)에 따라 피드백을 수집합니다. 이 시스템은 각 단계의 진단을 효과적으로 활용함으로써, 에이전트가 각 스테이지 경계에서 피드백을 받을 수 있도록 합니다. 또한, 플랫폼에서는 다양한 물리학 전문가들이 토론을 진행하여 MDForge의 행동을 수정하는 데 필요한 매길 유형의 비판을 생성합니다.

- **Performance Highlights**: MDForge는 세 가지 SAMPL(정확한 자유 에너지 데이터셋) 벤치마크에서 경쟁력 있는 성능을 보입니다. 특히, CB[7] 파이프라인은 관찰되지 않은 후보 게스트 라이브러리에서 새로운 바인더를 발견하였고, 이는 wet-lab 경쟁 NMR 실험에 의해 고친밀도, 피코몰 단계의 바인더로 확인되었습니다. 이 연구 결과는 연구와 개발에 있어 중요한 진전을 보여줍니다.



### Zero-source LLM Hallucination Detection with Human-like Criteria Probing (https://arxiv.org/abs/2606.12900)
Comments:
          Accepted at ICML 2026

- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs)의 환각(hallucination)을 감지하기 위한 새로운 패러다임인 Human-like Criteria Probing for Hallucination Detection (HCPD)를 제안합니다. HCPD는 인간 평가자의 다면적인 추론을 모방하며, LLM이 판단을 가중치를 둔 해석 가능한 기준 세트로 분해하고, 각 기준에 대한 점수를 집계하여 최종 진실성을 측정합니다. 특히, 외부 참조가 없고 모델 내부 정보가 제한된 제로 소스(zero-source) 환경에서도 효과적으로 작동합니다.

- **Technical Details**: HCPD의 핵심은 Human-like Criteria Probing (HCP) 메커니즘으로, 사전 훈련된 LLM 에이전트가 쿼리-답변 쌍에 대해 적응적으로 미세한 기준 세트(예: 사실 정확성, 논리적 일관성)를 생성하고 이들의 중요도에 따라 점수를 매깁니다. 이는 각 기준의 가중치를 맥락에 맞게 조정하여 인간 전문가의 복잡한 판단을 모사하게 됩니다. 또한 이러한 적응성 판단 기능을 위해 우리는 약한 감독(weak supervision)에서 파생된 보상 기반 정렬(training alignment) 방법을 도입합니다.

- **Performance Highlights**: HCPD는 다양한 실험을 통해 기존의 최첨단 방법들보다 일관되게 우수한 성능을 보였습니다. 특히, 생성 과정의 변동성을 줄이고 해석 가능성을 유지하기 위해 다중 샘플링 집계 전략을 적용하여 강력한 결정을 내릴 수 있음을 강조합니다. 이 연구는 제로 소스 환각 감지에서의 강력한 실용성을 입증하며, 이를 통해 안전하고 신뢰할 수 있는 LLM 활용을 위한 기반을 마련하였습니다.



### The Hidden Power of Scaling Factor in LoRA Optimization (https://arxiv.org/abs/2606.12883)
- **What's New**: 본 연구는 Low-Rank Adaptation (LoRA)에서 스케일링 팩터 α의 역할을 다시 조명합니다. 기존에는 α가 학습률에 대한 보조 요소로 간주되었지만, 이 논문은 α가 최적화의 주된 원동력이라는 것을 보여줍니다. 저자들은 α가 기존 학습률 스케일링만으로는 재현할 수 없는 성과를 낸다고 주장합니다.

- **Technical Details**: 이 연구는 LoRA의 최적화 메커니즘을 이해하기 위해 Signal-Drift 프레임워크를 개발하였습니다. 이 프레임워크는 α와 η가 어떻게 다르게 작용하는지를 설명하며, 최적의 스케일링 팩터는 순수하게 작용하여 더 빠르고 안정적인 최적화를 가능하게 한다고 설명합니다. 저자들은 α와 r의 관계가 제곱근 법칙에 따르며, 기존의 간단한 경험적 휴리스틱의 한계를 지적합니다.

- **Performance Highlights**: LoRA-α+의 제안은 학습률을 표준 소규모로 설정하면서도 성능을 향상시키는 혁신적인 접근법입니다. 이는 다양한 모델과 태스크에서 일관되게 우수한 성능을 보였으며, 기존의 LoRA보다 월등한 결과를 나타냅니다. 연구의 결과로, LoRA의 최적화 성능은 주로 충분히 큰 스케일링 팩터 α에 달려 있다는 것이 입증되었습니다.



### HarnessBridge: Learnable Bidirectional Controller for LLM Agent Harness (https://arxiv.org/abs/2606.12882)
- **What's New**: 최근 대규모 언어 모델들이 장기 작업의 에이전트로 점점 더 많이 사용되고 있지만, 이들의 성능은 모델 능력(model capability)과 환경 디자인(environment design)뿐만 아니라 에이전트-환경 상호작용을 중재하는 하네스(harness)에 의해서도 영향을 받습니다. 기존 하네스는 대부분 수동으로 설계되어 있어, 경로가 길어지고 상호작용이 복잡해짐에 따라 확장하기 어려운 문제가 있었습니다. 본 연구에서는 학습 가능한 플러그인 모듈을 통해 하네스를 생성할 수 있는지 탐구합니다.

- **Technical Details**: 우리는 HarnessBridge라는 경량의 학습 가능한 하네스 컨트롤러를 도입했습니다. 이 컨트롤러는 에이전트-환경 인터페이스를 양방향 프로젝션(bidirectional projection)으로 매개화하여 작동합니다. HarnessBridge는 두 가지 양방향 프로젝션을 학습합니다: 관측 프로젝션(observation projection)은 원시 경로(raw trajectories)를 압축된 결정 관련 상태(compact, decision-relevant states)로 변환하고, 행동 프로젝션(action projection)은 제안된 행동을 실행 가능한 전환으로 변환하거나 경로 기반 거부(trajectory-grounded rejection)로 처리합니다.

- **Performance Highlights**: HarnessBridge는 통합 명령 튜닝(unified instruction tuning)을 통해 하네스 감독 데이터셋에서 훈련되었습니다. Terminal-Bench~2.0과 SWE-bench Verified에서 HarnessBridge는 강력한 전문 하네스와 일치하거나 이를 초월하는 성능을 보여주었습니다. 또한 토큰 사용량과 경로 길이를 크게 줄이고, 더 작은 생성기(generators)에서 더 큰 상업용 모델로 일반화(generalization)할 수 있는 능력을 갖추었습니다.



### DailyReport: An Open-ended Benchmark for Evaluating Search Agents on Daily Search Tasks (https://arxiv.org/abs/2606.12871)
- **What's New**: 이 논문에서는 DailyReport라는 새로운 벤치마크를 소개합니다. 이는 검색 에이전트(Search Agents, SAs)를 일상적인 검색 작업에 평가하기 위해 고안되었습니다. 기존의 벤치마크들은 주로 전문 분야의 특정 질문에 중점을 두었으나, DailyReport는 사용자들이 필요로 하는 실시간 정보 요구를 반영하고 있습니다.

- **Technical Details**: DailyReport는 150개의 개방형 작업과 3,546개의 관련 루브릭을 포함하고 있습니다. 각 작업은 하위 작업으로 분해되어 서로 다른 차원에 대한 계단식 루브릭(cascade rubrics)으로 평가됩니다. 이 평가 방법은 사용자 관점에서의 성능을 명확하게 정량화 할 수 있도록 돕습니다.

- **Performance Highlights**: 17개의 검색 에이전트 시스템을 DailyReport를 통해 평가한 결과, 현재 시스템들은 지시 따르기(instruction following)에서는 좋은 성과를 보이지만, 사실성(factuality) 및 합리성(rationality)에서는 여전히 부족함을 드러냈습니다. 사용자 선호 점수는 특히 낮게 평가되었으며, 이는 현재의 SA 출력이 사용자 기대와의 큰 간극을 드러내는 것입니다.



### WISE: A Long-Horizon Agent in Minecraft with Why-Which Reasoning (https://arxiv.org/abs/2606.12852)
- **What's New**: 이 논문은 새로운 에이전트 프레임워크인 WISE(Which-Why Informed Semantic Explorer)를 소개합니다. 기존의低레벨 컨트롤러가 성능 병목 현상을 야기하는 문제를 해결하고자 episodic memory와 causal reasoning을 통합합니다. WISE는 causal event graph를 통해 관찰과 작업 연관성을 명확하게 연결하여, 기억의 활용도를 높이고 새로운 기회를 효과적으로 식별하는 방안을 제시합니다.

- **Technical Details**: WISE의 핵심은 Causal Event Graph로, 이는 과거의 관찰을 의미적 및 인과적 관계로 확장합니다. 이 구조는 과거의 관찰을 비단 단순한 기억으로 저장하는 것이 아니라, 행동과 결과 간의 인과관계를 생성하여 환경에서의 탐색과 결정 과정을 개선하는 데 기여합니다. 또한, WISE는 복수의 스케일을 통해 환경을 탐색하는 방식으로 해당 환경에 대한 포괄적인 관찰을 보장합니다.

- **Performance Highlights**: 실험 결과, WISE는 MrSteve에 비해 탐색 범위를 14% 증가시켰고, 연속 sparse 작업 성공률을 30% 향상시켰습니다. 또한 완료 시간은 26.4% 줄어들었고, 비연속 작업 성공률은 44% 증가하며, 완료 시간은 42.5% 감소했습니다. 이는 WISE의 핵심 구성 요소 간의 강력한 시너지를 나타내며, 구조적인 개선뿐만 아니라 실제 성능 개선을 입증합니다.



### (Human) Attention Is (Still) All You Need: Human oversight makes AI-assisted social science reliab (https://arxiv.org/abs/2606.12848)
- **What's New**: 이 논문은 대형 언어 모델(LLM)을 활용한 연구에서 AI 지원의 신뢰성이 단순히 모델의 능력뿐 아니라 인간과 기계 간의 인지 노동 구조에 영향을 받는다는 점을 논의합니다. 연구자들이 과거에 맡았던 가설 생성, 사양 선택, 결론 초안을 작성하는 역할을 LLM에게 위임하는 경향이 증가하고 있습니다. 제출된 연구에서는 Human-in-the-Loop Economic Research (HLER)라는 결정 구조를 통해 이러한 문제를 탐구하며, 280개의 실험 환경에서 HLER이 실패율을 16%로 감소시켰음을 보여줍니다.

- **Technical Details**: HLER은 데이터 감사, 데이터 프로파일링, 질문 생성, 데이터 구축, 식별 평가, 계량 경제적 추정, 원고 작성 및 검토 등 8개의 전담 에이전트 역할로 연구 작업 흐름을 세분화합니다. 각 에이전트는 자신의 역할에 따라 세부 작업을 수행하며, Orchestrator라는 구성 요소가 작업 상태(RunState)를 관리합니다. 이 연구는 모델 간의 인간 통제가 어떻게 구조화되는지에 대한 결정 설계 문제를 다루며, 특정 역할에서의 인간 결정이 흐름에서 중요한 역할을 한다고 강조합니다.

- **Performance Highlights**: 실험 결과, 제약 없는 파이프라인에서 72%의 심각한 실패가 발생한 반면, HLER 구조가 적용된 파이프라인에서는 실패율이 16%로 줄어드는 결과를 보였습니다. 특히 청나라 인구 등록부와 같이 제공되지 않은 데이터셋에서 신뢰성 향상이 가장 두드러졌습니다. 결과적으로 HLER은 AI 지원 경험적 과학을 위한 연구 구조로, LLM이 생성한 추론을 결정적 계산, 명시적 인간 게이트, 감사 가능한 연구 기록을 통해 채널링하는 것으로 해석됩니다.



### Fantastic Scientific Agents and How to Build Them: AgentBuild for Rietveld Refinemen (https://arxiv.org/abs/2606.12834)
- **What's New**: 이 논문은 과학적 작업 흐름 관리에서 LLM 기반 에이전트의 구축 방법론인 AgentBuild를 제안합니다. 연구자는 자신의 전문 지식을 바탕으로 한 계약을 통해 에이전트를 만드는 과정을 단계별로 제시합니다. 이 방법론은 갈수록 복잡해지는 과학적 데이터 분석에 최적화된 자동화 시스템을 제공하며, 연구자가 에이전트 코딩이 아닌 저자로서의 역할을 강조합니다.

- **Technical Details**: AgentBuild는 연구자가 작성한 세 가지 주요 아티팩트를 기반으로 LLM 기반 에이전트를 구축하는 프로세스를 정의합니다. 첫 번째는 품질 및 시각적 기준을 명시한 버전 관리된 루브릭(rubric)입니다. 두 번째는 난이도에 따라 분류된 커리큘럼(curriculum)이며, 세 번째는 절차적 지식을 차별화된 출처에서 추출한 외부 지식 기반입니다.

- **Performance Highlights**: 이 연구는 Rietveld 정제 및 X선 회절 데이터 분석을 위해 GSAS-II 등을 활용한 사례 연구를 통해 AgentBuild의 효과를 입증합니다. AgentBuild는 작업 흐름이 복잡한 상황에서도 연구자의 판단을 명확하게 유지하면서 에이전트를 구축하도록 설계되었습니다. 최종적으로, 연구자는 기계 모델이 발전함에 따라 AgentBuild를 재조정하는 방향으로 지속가능한 과학적 에이전트를 구현할 수 있다는 점에 강조를 두고 있습니다.



### Topical Phase Transitions in Artificial Intelligence Research: Large-Scale Evidence and an Early-Warning Signature for Emerging Topics (https://arxiv.org/abs/2606.12828)
- **What's New**: 이 논문에서는 인공지능(Artificial Intelligence, AI) 연구 주제가 점진적으로 성장하는가, 아니면 급격한 변화를 통해 발전하는지를 분석합니다. 2017년부터 2025년까지의 80,814개 주요 논문을 통해 주요 AI 주제가 주제적 국면 전환(topical phase transitions)을 통해 발전함을 보여줍니다. 2025년까지 대형 언어 모델(Large Language Models)이 주요 주제가 되었음을 강조합니다.

- **Technical Details**: 논문은 2017-2021년 데이터를 기반으로 한 조기 경고 신호(early-warning signature) 및 네 가지 출판 역학(publication-dynamics criteria)을 정의하여 2023-2025년 변화에 대한 예측을 비교합니다. 이 분석을 통해 27%의 정밀도(precision)와 63%의 재현율(recall)을 기록하였으며, 이는 유의미한 기반율(base rate)인 13.5%에 비해 높은 성과입니다. 2025년 데이터를 적용하여 주의 깊게 모니터링해야 할 주제로는 reasoning, test-time compute, agentic AI, multimodal LLMs, retrieval-augmented generation 및 world models가 포함됩니다.

- **Performance Highlights**: AI 연구의 구조를 대규모로 분석한 것은 본 논문의 주요 기여 중 하나입니다. 주제 전환이 진행되기 전 감지할 수 있는 흔적이 있는지를 규명하는 것이 주요 목표 중 하나였습니다. 이러한 통찰력은 향후 AI 연구의 방향을 사전에 예측하는 데 중요한 역할을 할 것입니다.



### GeoNatureAgent Benchmark: Benchmarking LLM Agents for Environmental Geospatial Analysis Across Frontier and Open-Weight Foundation Models (https://arxiv.org/abs/2606.12821)
Comments:
          Preprint. 10 pages, 8 figures. Submitted to ACM SIGSPATIAL 2026

- **What's New**: 이번 연구는 GeoNatureAgent Benchmark를 도입하여 구조화된 도구 호출을 통해 실제 API에 적용되는 환경 분석 에이전트를 평가할 수 있는 첫 번째 기준을 마련하였습니다. 총 93개의 작업과 18개 범주로 구성된 이 벤치마크는 지자체 분석, 공간 추론 등 다양한 분야를 포함하고 있습니다. 이를 통해 환경 과학 워크플로우에서 AI 에이전트를 자동화할 수 있는 가능성을 연구하였습니다.

- **Technical Details**: 이 연구에서는 7개의 LLM 모델(Claude Sonnet 4, DeepSeek V3.2 등)을 평가하였으며, 우선 Claude Sonnet 4가 60.8%의 성공률을 보여주었습니다. 각 모델은 1.0 온도(seed)에서 동일한 인프라로 평가되었고, 비용과 성능 간의 관계를 분석하여 DeepSeek V3.2가 더 낮은 비용으로 Claude의 93% 성능을 달성한다는 사실이 밝혀졌습니다. 구조화된 도구 호출을 통한 평가가 일반 GIS 벤치마크보다 더 차별적인 결과를 보여주었다는 점도 주목할 만합니다.

- **Performance Highlights**: 연구 결과, 전체 93개 작업중 비교 작업에서는 0%의 성공률을 기록하여 모델의 논리적 한계를 드러냈습니다. 새로운 벤치마크는 기존의 모형 평가 수단과 다르게, 현실 세계의 유용성을 높이는 방향으로 설계되었습니다. 또한 모든 평가를 재현할 수 있도록 독립적인 FastAPI 서비스를 공개하여 연구 결과의 투명성을 높였습니다.



### Teach-and-Repeat: Accurately Extracting Operational Knowledge from Mobile Screen Demonstrations to Empower GUI Agents (https://arxiv.org/abs/2606.12817)
Comments:
          20 pages, 9 figures. Yudong Zhang and Lei Hu contributed equally to this work. Xingyu Liu, Zuojian Wang, and Zhilin Gao are corresponding authors

- **What's New**: 본 논문에서는 Teach VLM이라는 모델을 소개하며, 이를 통해 모바일 화면 전환을 단계별 운영 지식으로 변환할 수 있는 새로운 접근 방식을 제안합니다. Teach VLM은 demonstrating videos에서 운영 관련 주요 프레임을 추출하고 분석하여 자연어로 설명된 짧은 문장으로 동작을 표현합니다. 이를 통해 기존의 vision-language models (VLMs)의 한계를 극복하고, 다양한 UI 디자인에 적용할 수 있는 가능성을 열어줍니다.

- **Technical Details**: Teach VLM은 사용자 demonstration video를 바탕으로 관련 없는 전환 프레임을 필터링한 후, 시각적 상태 변화에서 나오는 자연어 절차적 설명을 유도합니다. 데이터 수집의 효율성을 높이기 위해 Data Flywheel 메커니즘을 설계하여 대규모의 멀티 도메인 훈련 데이터 세트를 저렴한 비용으로 구축합니다. 또한, 프레임 수준의 의미 주석이 포함된 Chinese Mobile Screen Teach Benchmark를 도입하여 생성된 운영 지식의 정밀한 평가를 가능하게 합니다.

- **Performance Highlights**: Teach VLM은 운영 의미 예측에서 최첨단 성능을 달성하며, downstream agents의 작업 성공률(Task Success Rate)을 일관되게 향상시키는 결과를 보여줍니다. Teach-and-Repeat 패러다임을 통해 한 번의 교육으로 반복적인 실행을 가능하게 하여 앱 인터페이스의 빈번한 업데이트와 모호한 사용자 지침 문제를 해결합니다. 전반적으로, Teach VLM과 Teach-and-Repeat 패러다임은 원시 데모를 재사용 가능한 작업 자동화로 전환하는 실용적인 경로를 제공합니다.



### MLUBench: A Benchmark for Lifelong Unlearning Evaluation in MLLMs (https://arxiv.org/abs/2606.12809)
Comments:
          36 pages, accepted to the ICML 2026

- **What's New**: 이 논문에서는 MLLM Lifelong Unlearning이라는 도전적인 문제를 다루고 있으며, MLUBench라는 대규모의 포괄적인 벤치마크를 도입합니다. MLUBench는 127개의 실세계 개체와 9개의 클래스에서 이루어진 데이터 언러닝 요청을 통해 MLLM의 지속적인 언러닝을 평가하는 데 초점을 맞춥니다. 이 벤치마크는 다양한 유형의 데이터를 포함하고 있어 MLLM의 지속적 언러닝 성능을 보다 효과적으로 연구할 수 있도록 합니다.

- **Technical Details**: MLUBench는 총 127개의 개체와 5,105개의 이미지, 15,414개의 VQA 쌍을 포함하고 있으며, 특정 언러닝 작업에 대한 연속적인 평가를 위해 개체를 언러닝 작업 시퀀스로 구성하고 있습니다. 이 연구는 MLLM의 벤치마크가 기존의 단일 모드 모델과의 주요 차별점인 멀티모달 정렬(Multimodal Alignment)의 유지 필요성을 강조합니다. 제안된 LUMoE는 Mixture of Experts (MoE) 개념을 사용하여 특정 언러닝 작업을 위한 스위치 가능한 저순위 적응(LoRA) 어댑터를 활용합니다.

- **Performance Highlights**: 실험 결과, 기존의 언러닝 방법들이 성능 저하 문제에 심각한 영향을 받는 반면, LUMoE는 이러한 성능 저하 문제를 효과적으로 완화함을 보여주었습니다. 예를 들어, GA 방법의 경우 첫 번째 작업에서의 잊기 품질이 0.38에서 0.01로 급격히 떨어지는 것을 확인하였습니다. 이러한 결과들은 LUMoE의 기법이 향후 MLLM 연구 분야에서 중요한 기준이 될 것임을 시사합니다.



### The Containment Gap: How Deployed Agentic AI Frameworks Fail Public-Facing Safety Requirements (https://arxiv.org/abs/2606.12797)
Comments:
          ICML 2026 (AI4GOOD Workshop)

- **What's New**: 최근 Agentic LLM 시스템이 정부 서비스, 헬스케어 및 금융과 같은 공공 분야에 배포되고 있습니다. 이러한 시스템은 도구를 호출하고 지속적인 메모리를 유지하며 다단계 계획을 자율적으로 수행합니다. 본 논문은 이러한 시스템을 구축하는 프레임워크가 구조적 안전성을 보장하는지에 대한 질문을 던집니다. 저자들은 LangChain, AutoGPT, OpenAI Agents SDK를 감사하여 이들 모두가 구조적 안전성을 보장하지 않는다는 결과를 도출했습니다.

- **Technical Details**: Agentic LLM 시스템은 외부 입력을 처리하는 인식 기능(PP), 현재 입력 및 지속 메모리를 사용하여 행동을 계획하는 사고 기능(BB), 도구를 호출하는 실행 기능(EE), 그리고 결과를 지속 상태에 기록하는 메모리 업데이트 기능(𝒰)을 포함한 네 가지 기능 단계로 구성됩니다. 각 단계는 보안 역할이 있을 수 있지만 상호 격리가 없으면 취약해질 수 있습니다. 이러한 점에서, 저자들은 여섯 가지 containment 원칙을 제시하며 이들을 프레임워크 감사의 기초로 삼았습니다.

- **Performance Highlights**: 감사는 세 가지 평가된 프레임워크에서 모든 원칙의 기본 준수가 전혀 관찰되지 않았으며, 메모리 무결성은 어떤 프레임워크에서도 나타나지 않았습니다. 또한, LangChain을 기반으로 한 정부 혜택 에이전트의 시뮬레이션에서 단일 메모리 오염 쓰기가 모든 테스트된 경우에 걸쳐 지속적으로 타겟화된 손상을 야기할 수 있음을 보여주었습니다. 마지막으로, 저자들은 메모리 무결성 검사기와 정책 게이트라는 두 가지 경량 containment 메커니즘을 제안하여 공격 벡터를 효과적으로 차단할 수 있음을 입증했습니다.



### A Tutorial on World Models and Physical AI (https://arxiv.org/abs/2606.12783)
- **What's New**: 이 논문에서는 예측, 추론, 의사결정을 수행할 수 있는 지능형 시스템을 위한 세계 모델링의 중요성이 강조됩니다. 특히, 명시적(world models) 및 암시적(implicit world models) 세계 모델 사이의 차별점을 명확히 하여, 두 접근 방식이 어떻게 물리적 인공지능(Physical AI)에서 서로 보완적으로 작용하는지를 설명합니다. 이를 통해 자율주행 및 로보틱스와 같은 실제 응용 분야에서의 활용 가능성을 제시합니다.

- **Technical Details**: 세계 모델은 환경의 동적 변화와 그에 따른 행동의 결과를 인식하고 예측할 수 있는 내재적 모델을 제공합니다. 이 문서에서는 강화 학습(Reinforcement Learning, RL)의 기본 개념을 바탕으로 하여 MDP(Markov Decision Process) 프레임워크와의 연계를 통해 세계 모델의 작동 원리를 설명합니다. 명시적 세계 모델은 행동과 그 결과를 명확히 시뮬레이션하는 데 사용되며, 암시적 모델은 예측 가능한 구조를 내포하고 있습니다.

- **Performance Highlights**: 본 논문은 다양한 세계 모델링 접근 방식이 예측, 계획 및 일반화를 통합하는 기제를 제공하는 방법에 대한 구조적 토대를 제시합니다. 특히, 물리적 AI 분야에서의 실제 사례를 통해 세계 모델이 로봇의 행동 계획, 안전한 학습, 그리고 효율적인 의사결정에 기여하는 방식을 설명합니다. 최종적으로, 이 논문은 인공지능의 기술적 진전을 위해 해결해야 할 과제들에 대해서도 논의합니다.



### Constructing Evaluation Datasets for Procedural Reasoning: Balancing Naturalness, Grounding, and Multi-Hop Coverag (https://arxiv.org/abs/2606.12767)
Comments:
          10 pages, 2 numbered figures. Workshop submission to HAIL @ AIED 2026

- **What's New**: 이 논문은 AI 지원 학습 시스템에서 절차적 추론을 평가하는 데 필요한 질문-답변 데이터셋의 품질 향상 방안을 제안합니다. 특히 세 가지 질문 생성 전략, 즉 TMK 기반의 엄격한 생성, 전사 중심 생성, 그리고 TMK 인식 생성을 비교하여 각각의 장단점을 분석하였습니다. 이 연구는 690개의 생성된 질문-답변 쌍을 바탕으로 데이터셋 구성 전략의 효과를 구체적으로 보여줍니다.

- **Technical Details**: 연구에서는 TMK(Task-Method-Knowledge) 모델을 사용하여 절차적 지식의 구조적 표현을 생성하고 평가합니다. TMK 모델을 통해 생성된 질문이 얼마나 기반이 확고한지를 평가하기 위해, 근거 검증 프레임워크를 도입하였습니다. 이 프레임워크에서는 질문과 답변이 근거가 되는 제기에서 뒷받침 되는지를 측정하며, 질문이 스스로 이해될 수 있는지를 평가합니다.

- **Performance Highlights**: 엄격한 TMK 생성 전략이 가장 높은 품질의 질문을 생성하며, 96.5%의 질문이 확고한 기반을 가지고 있고 92.6%의 질문이 사용 가능하다는 결과를 보였습니다. 반면, 전사 중심 생성은 더 자연스러운 질문을 생성하지만, 많은 질문이 문맥 의존적이거나 약한 근거에 기반합니다. TMK 인식 생성은 높은 다단계 적용률을 보이지만, 질문의 사용 가능성이 떨어지는 경향이 있음을 발견했습니다.



### Prefill Awareness in Large Language Models (https://arxiv.org/abs/2606.12747)
Comments:
          Submitted to NeurIPS 2026

- **What's New**: 이번 연구는 언어 모델의 사전 채우기(Prefill) 인식(prefill awareness) 능력을 다루고 있습니다. 이는 모델이 이전 출력이 수정되었음을 인식하고 그에 대응하는 능력을 의미합니다. 연구 결과, Claude Opus 4.5는 사전 채우기된 내용이 자신과 반대되는 경우 9-35%의 정확도로 이를 감지할 수 있음을 보여주었습니다. 이는 기존의 사전 채우기 기반 모델 평가 방법의 유효성에 중대한 영향을 미칠 수 있습니다.

- **Technical Details**: 연구에서는 세 가지 유형의 사전 채우기 조작을 사용하여 모델의 인식 능력을 평가했습니다: 사고(thinking) 조작, 직접 답변(direct-answer) 조작, 그리고 과거 회차(past-turn) 조작입니다. 이를 통해 모델들은 서로 다른 힌트를 기반으로 감지 및 저항(resistance)할 수 있음을 발견했습니다. 특히, 스타일 불일치가 외부 사전 채우기를 표시하는 데 영향을 주며, 선호 불일치가 기본 답변으로 복귀하도록 유도하는 데 영향을 미친다는 것을 확인했습니다.

- **Performance Highlights**: 연구는 잘 설정된 사전 채우기를 통한 평가가 응답의 일관성과 신뢰성에 미치는 영향을 다루고 있으며, 사전 채우기 인식 능력이 이미 일부 고급 모델에서 중요 요소로 자리잡고 있음을 강조합니다. 이러한 인식 능력을 측정하고 평가에 포함시킬 것을 권장하며, 향후 연구에서 메커니즘적 원인을 탐구할 필요성을 제기합니다. 눈에 띄는 성과로는, Claude Opus 4.5가 55-68%의 감지 정확도를 기록했다는 점이 있습니다.



### Reducing the Complexity of Deep Learning Models for EEG Analysis on Wearable Devices (https://arxiv.org/abs/2606.12742)
- **What's New**: 최근 웨어러블 헬스케어 장치가 IoT(사물인터넷) 분야에서 급속히 성장하고 있습니다. 특히 ECG와 EEG라는 두 가지 중요한 생물학적 신호를 활용한 자동화된 헬스케어 서비스는 의학 분야에서의 연속 모니터링을 가능하게 하고 있습니다. 그러나, 이런 신호를 처리하기 위한 딥 뉴럴 네트워크(DNN) 모델이 웨어러블 장치의 제한된 에너지 및 계산 능력에 적합하지 않아 도전 과제가 존재합니다.

- **Technical Details**: 이 논문은 리소스가 제한된 웨어러블 장치에 최신 DNN 모델을 배포할 수 있는 가능성을 조사합니다. 특히, 파라미터 양자화(parameter quantization)와 전극 축소(electrode reduction) 방법이 DNN의 정확성과 계산 복잡성 간의 균형에 미치는 영향을 분석합니다. EEG 신호 분석을 위한 여러 최신 DNN 모델의 성능 또한 비교하였습니다.

- **Performance Highlights**: 연구 결과, CNN 기반의 특징 추출 모듈과 LSTM을 결합한 모델이 웨어러블 장치 환경에서 가장 효과적이라는 것을 보여주었습니다. 또한, 파라미터를 4비트로 축소해도 기능적 효용이 유지됨을 확인하였고, EEG 모니터링 장치에서 사용하는 전극의 최소 세트를 성공적으로 규명하여 편안한 사용자 경험을 제공할 수 있는 통찰을 얻었습니다.



### Benchmarking AI Agents for Addressing Scientific Challenges Across Scales (https://arxiv.org/abs/2606.12736)
Comments:
          6 figures

- **What's New**: 이 논문에서는 심층 학습 기반 AI 에이전트를 위한 새로운 벤치마크인 SciAgentArena를 제시합니다. 이는 다양한 과학 분야에서 실제 연구 시나리오를 기준으로 AI 에이전트를 평가하기 위해 고안되었습니다. SciAgentArena는 약 200개의 과제를 포함하고 있으며, 단계별 검증 및 인터랙티브한 에이전트 비특화 환경을 제공합니다. 이를 통해 현재의 AI 에이전트들이 명확한 데이터 분석 작업에서 효과적으로 기여할 수 있지만, 과학적 맥락에 따라 성능이 불균형함을 발견했습니다.

- **Technical Details**: SciAgentArena는 여러 분야의 신흥 요구 사항을 반영하여 실제 과학 연구 상황에서의 AI 에이전트를 평가하기 위한 체계적인 벤치마크 프레임워크입니다. 이 프레임워크는 복잡한 문제 해결을 위한 다단계 워크플로우 및 검증 가능한 중간 상태에 중점을 둡니다. AI 에이전트의 능력을 평가하기 위해 우리는 런닝 프레임워크와 평가 프레임워크를 분리하였으며, 이는 다양한 AI 에이전트를 평가할 수 있는 유연성을 제공합니다. 설정된 환경에서 AI 에이전트의 입력 및 출력을 통합하여 최종 메트릭을 생성합니다.

- **Performance Highlights**: AI 에이전트의 성능은 데이터 로딩, 분석, 최적화 및 발견이 이질적인 것으로 나타났습니다. AI 에이전트는 고정된 파이프라인을 통해 데이터셋을 분석하는 데 능숙하지만, 분자 최적화나 새로운 과학적 발견을 도출하는 능력은 제한적이었습니다. 연구 결과에 따르면, 여러 AI 에이전트 간의 상호작용에서 자율적 탐색과 일반화 능력이 결여되어 있음을 강조했습니다. 이러한 문제를 해결하기 위해 지식 기반 확장, 세부 프롬프트 제공 등을 통한 AI 에이전트의 능력 향상 방안을 제안했습니다.



### Rethinking Psychometric Evaluation of LLMs: When and Why Self-Reports Predict Behavior (https://arxiv.org/abs/2606.12730)
Comments:
          Accepted as an Oral (Contributed Talk) at the ICML 2026 Workshop on Combining Theory and Benchmarks (CTB)

- **What's New**: 이번 논문은 LLM(대형 언어 모델)의 행위 경향성을 저비용 심리 측정 도구인 자기 보고(Self-Report, SR)를 통해 예측할 수 있는 가능성을 탐구하고 있습니다. 연구진은 SR과 행동 사이의 연관성이 존재하나 선택적임을 발견했으며, 이는 행동이 자각되지 않거나 문맥에 따라 변동할 수 있음을 나타냅니다. 특히, 연구는 Big 5와 TPB(Planned Behavior Theory)라는 두 가지 이론적 프레임워크를 비교하였습니다.

- **Technical Details**: 연구는 2×2×2의 팩토리얼 설계를 채택하여 TPB와 Big 5의 효과를 시험했습니다. 자료는 11개의 최첨단 LLM과 4가지 행동 과제(위험 감수, 아첨, 정직, 암묵적 바이어스)에 걸쳐 수집되었습니다. 연구에서는 SR과 행동 사이의 일치성을 측정하기 위해 세션 문맥과 정체성 유도 방법을 변경하였습니다.

- **Performance Highlights**: SR-행동 일치성은 문맥에 따라 달라졌으며, TPB의 경우 인간 수준의 일치성을 보여주었으나 Big 5는 그렇지 않았습니다. 또한, 별도의 세션에서의 일치성은 행동이 문맥 밖에 고정되어 있을 때만 유지되는 반면, 아첨과 같은 성격의 행동은 문맥에 달려 있어 붕괴되었습니다. 이러한 발견은 기존의 성격 테스트가 LLM 행동 예측에 적합하지 않을 수 있음을 시사합니다.



### The Theory of Mind Utility: Formal Specification of a Mentalizing Mechanism (https://arxiv.org/abs/2606.12721)
- **What's New**: Theory of Mind Utility(ToM-U)라는 새로운 이론이 소개되며, 이는 사회적 추론의 기초로 작용합니다. ToM-U는 믿음을 추론하는 과정을 정확히 정의함으로써 기존 이론들과 차별화됩니다. 이 과정에서는 정보를 얻는 순서와 출처의 신뢰도를 바탕으로 믿음의 상태를 추론하는 데 중점을 둡니다.

- **Technical Details**: ToM-U는 Local Epistemic World Models(LEWMs)을 통해 정보를 처리하며, 이는 유도된 그래프 형식으로 대리인의 상태 및 정보 간의 관계를 나타냅니다. 이 이론은 5가지 정의를 통해 LEWM의 구조와 에이전트 노드 속성을 형성하며, 이러한 구조는 추론 절차와 과거 정신화 시도의 흔적을 포함한 잔여 기능도 정의합니다. 또한, ToM-U는 기계적으로 예측 가능한 결과를 생성하고, 정보 접근 이력에 따라 신뢰도를 평가하는 과정이 포함됩니다.

- **Performance Highlights**: ToM-U는 이전 이론들보다 더 정교한 예측을 가능하게 하며, 기존의 믿음 상태를 단순히 가정하지 않고 체계적으로 추론합니다. 특히 이 이론은 정신화 실패에 대한 방향성 있는 예측을 생성하여 다양한 사회적 상황에서 적용합니다. ToM-U의 구조는 단순히 이론적 관점에서 벗어나 실질적인 사회 인지 과정에서의 활용 가능성을 보여줍니다.



### Definitional alignment before capability alignment: a Design-Science framework for adjudicating claims about AGI (https://arxiv.org/abs/2606.12713)
Comments:
          31 pages, 1 table, 2 appendices

- **What's New**: 이 논문은 인공지능 일반 지능(AGI)이 여전히 진화 중인 개념임을 강조하며, 운영화의 차이에 따라 동일 시스템에 대해 상반된 결과가 도출된다는 점을 논의합니다. 이 논문은 DAF-AGI라는 개념적 아티팩트를 개발하여 AGI 정의를 평가할 때 사용하는 기준과 거버넌스 감사 체계를 제시합니다. 이를 통해 현재의 AGI에 대한 다양한 관점을 조명하고, 각각의 정의가 가진 내재된 가치 판단을 드러내는 데 초점을 맞추었습니다.

- **Technical Details**: DAF-AGI는 5가지 서수 기준을 사용하여 AGI의 정의가 어떻게 인식되는지를 평가하는 두 가지 결합된 요소로 구성됩니다. 이 프레임워크는 각 정의의 인증(authority), 이해관계(interest), 외부 검증(external verification) 및 수정 권한(revision authority)을 분석하는 구조적 감사(g governance audit)도 포함합니다. 총 여섯 가지 측정 패밀리 중 다섯 가지와 다른 한 가지 경계 위치에서 이 아티팩트가 적용되는 방식을 보여줍니다.

- **Performance Highlights**: 논문은 AGI의 정의가 상이하다는 것 외에도 경제적 가족의 정의가 불확실하다는 점을 지적합니다. 충족 여부는 반드시 성과 기반의 접근을 통해 확인할 수 있으며, 심리 측정(psychometric) 및 기술 습득(skill acquisition) 접근 방식을 통해서는 AGI로 인증되지 않습니다. 결과적으로, AGI는 테스트를 통한 실질적 검증이 필요하고, 이에 대한 정의 권한을 다시 조명해야 하며, 각 이해관계자가 이 문제를 어떻게 다루는지가 드러납니다.



### Deployment-Centered Evaluation: Predicting Query-Level Rejection Risk in a Clinical LLM System (https://arxiv.org/abs/2606.12702)
- **What's New**: 이번 연구는 대형 언어 모델(LLM)을 임상 시스템에 통합하는 과정에서, 기존의 정적 벤치마크가 사용자 수용성을 측정하지 못하고 있다고 지적합니다. 연구진은 전자 건강 기록 시스템에 내장된 LLM의 실제 유용성을 평가하기 위해 배치 중심의 평가를 수행했습니다. 이를 통해 사용자 피드백을 보다 효과적으로 반영하는 방법을 제시합니다.

- **Technical Details**: 연구에서는 사전 응답 분류기(pre-response classifier)를 훈련시켜, 사용자가 LLM 응답을 거부할 위험을 예측합니다. 이는 쿼리 내용(query content)과 배치 특정(context) 정보를 바탕으로 이루어지며, 4.5개월 동안의 사용자 피드백을 분석하여 AUROC(Area Under the Receiver Operating Characteristic curve) 0.719를 달성했습니다. 배치 특정 맥락을 활용하면 사용자의 거부 예측 능력이 향상된다는 주요 개념적 통찰력을 제공합니다.

- **Performance Highlights**: 이 연구의 성과로는 두 가지 하류 사용 사례(downstream use cases)에서 예측의 이점을 평가하였습니다. 사용자 거부 가능성을 예측함으로써 '가드레일' 트리거링(guardrail triggering) 및 자제(abstention)와 같은 분야에서 활용 가능성을 보여주었습니다. 이러한 예측 모델은 사용자 수용성을 강화하고, 목표 지향적인 가드레일을 위한 기초를 마련하는 데 기여할 것입니다.



### From AGI to ASI (https://arxiv.org/abs/2606.12683)
- **What's New**: 이번 보고서는 ανθρώπινη επιχείρηση (AGI) 후의 AI 발전 상황을 탐색하며, AI가 지속적으로 발전할 가능성이 있는 기술적 경로들과 이에 따라 발생할 수 있는 마찰 요소들을 정리합니다. AGI가 도달한 이후, 인공지능은 어떻게 인류 사회를 변화시킬지에 대한 구체적 질문을 제기하며, 인공지능의 미래는 한정되지 않고 다양한 경로를 통해 발전할 것임을 나타냅니다. 이처럼 기술의 진화를 예측하는 것은 복잡하며 다양한 요인들이 함께 작용할 수 있음을 강조합니다.

- **Technical Details**: 보고서는 AGI에서 인공지능 초지능(ASI)로의 전환을 위한 네 가지 경로를 제안합니다: AGI의 확장, AI 패러다임의 변환, 재귀적 개선, 그리고 대규모 다중 에이전트 집단에서의 ASI 출현입니다. 이 경로들은 상호 배타적이지 않으며 병행하여 발생할 수 있습니다. 또한, 이 경로들을 따라 직면할 수 있는 마찰 요소들을 식별하여 이들 간의 연결고리를 탐구하고, 각 경로에 미치는 영향이 미미할지 중요한지를 논의합니다.

- **Performance Highlights**: 이 보고서는 AI 발전의 속도가 일정하다면, 인류의 AGI 목표 달성에 시간이 필요 없을 수도 있다는 점을 강조합니다. AGI의 출현이 전례 없는 사회적 변화의 발판이 될 수 있으며, 이에 따른 준비는 광범위한 학문적 노력과 글로벌 관심을 요구합니다. 또한, 보고서는 AI 진행이 계속될 것이라는 불확실성을 고려해야 하며, 다양한 과학 및 기술 분야에서 AI의 발전이 변화를 가져올 가능성이 있음을 제시합니다.



### Evoflux: Inference-Time Evolution of Executable Tool Workflows for Compact Agents (https://arxiv.org/abs/2606.12674)
Comments:
          Code is available at this https URL

- **What's New**: Evoflux라는 새로운 방법론이 도입되어, Compact language models (LMs) 사용 시 실행 가능한 도구 워크플로우를 수정하는 혁신적인 접근 방식을 제시합니다. 이 방법은 구조화된 수정, 실행 피드백, 적응적 강도, 메타 지도 설계 및 다양성 감소(diversity pruning)를 통해 워크플로우 그래프를 진화시킵니다. 특히 소규모 플래너들이 툴 카탈로그의 변화에 잘 대응하지 못하는 문제에 대해 해결책을 제공합니다.

- **Technical Details**: Evoflux는 실행 시간에 진화적 검색(evolutionary search) 방법을 사용하여 각 도구의 의도된 사용을 기반으로 합니다. 이 과정에서 실행 가능한 워크플로우의 수정을 목표로 하며, 입력된 피드백을 바탕으로 구조적인 수정을 진행합니다. 이 방법은 다양한 수정 기법을 통해 워크플로우의 효율성을 높이고, 도구의 의존성을 유지하는 데 중점을 두고 있습니다.

- **Performance Highlights**: Evoflux는 MCP-Bench라는 테스트에서 실행 가능성을 약 3%에서 17-24%로 향상시키는 성과를 보였습니다. 이는 동일한 데이터에서 SFT 및 SFT+DPO 방법보다 뛰어난 성능을 보여주며, 이러한 방법들이 종종 성능 저하를 겪는 것과 대조적입니다. 따라서, Evoflux는 제한된 교사 추적 예산에서도 더 높은 신뢰성을 증명하였습니다.



### TrajGenAgent: A Hierarchical LLM Agent for Human Mobility Trajectory Generation (https://arxiv.org/abs/2606.12657)
Comments:
          14 pages, 2 figures, 8 tables. Accepted by the 27th IEEE International Conference on Mobile Data Management (MDM 2026)

- **What's New**: 이번 연구는 TrajGenAgent라는 새로운 계층적 LLM-agent 프레임워크를 제안하여, 인간 이동 경로 생성을 실현합니다. 기존의 모델 파인튜닝 없이도 실행 가능하며, 개인화된 POI 검색 및 거리 인식을 통해 각 활동을 완전한 방문으로 구체화합니다. 이 프레임워크는 두 단계의 오케스트레이터-노동자 설계를 사용하여, 과거 증거를 기반으로 활동 체인을 생성하고 각 활동을 상세히 모델링합니다.

- **Technical Details**: TrajGenAgent는 LLM의 의미적 추론과 계획 기능을 활용하며, 고정된 워크플로우 내에서 도구를 조정하여 결정적인 경로 생성을 가능하게 합니다. 첫 번째 단계에서 오케스트레이터는 개인의 역사적 활동 체인으로부터 활동 체인 스켈레톤을 생성하고, 두 번째 단계에서는 특별히 설계된 공간적 및 시간적 노동자가 활동을 완전한 방문으로 변형합니다. 이 과정에서 이동-모달리티를 고려하여 여행 시간을 추정하고, 역사적 활동과 일치하는 장소를 선택합니다.

- **Performance Highlights**: 실험 결과 TrajGenAgent는 기존의 신경망 및 LLM 기반 모델들보다 공간적 충실도(spatiotemporal fidelity)와 의미적 일관성(semantic coherence), 개별 행동의 사실성을 개선하였습니다. 특히, 새로운 평가 프레임워크(이상 탐지 기반)를 도입하여 생성된 경로의 행동 수준의 적합성을 평가하는 데 성공하였습니다. 이러한 결과는 TrajGenAgent가 고품질의 현실적인 이동 경로 생성을 가능하게 함을 시사합니다.



### "Did you lie?" Evaluating Lie Detectors across Model Scale and Belief-Verified Model Organisms (https://arxiv.org/abs/2606.12618)
Comments:
          12 pages, 6 figures

- **What's New**: 이번 연구는 기존의 언어 모델에서 lie detection(거짓 탐지) 기법이 제대로 기능하지 못한다는 문제를 해결하고자 한다. 연구진은 13개의 추론 모델 유기체를 사용하여 이들의 숨겨진 신념을 검증하고, 새로운 lie 탐지 기법인 Did-You-Lie(DYL)을 도입하였다. 검증된 lie 탐지가 다양한 동기를 포함하는 Varied Deception 테스트베드를 통해 수행되었으며, 이는 기존 방법론에서 나타난 다양한 한계를 극복하는 데 중점을 두었다.

- **Technical Details**: 이 연구는 2B에서 1T 파라미터를 가지는 31개의 언어 모델을 대상으로 두 가지 주요 축을 통해 성능을 평가하였다. 첫 번째는 모델의 능력을 반영한 것으로, 모든 탐지기가 모델의 능력에 따라 긍정적인 확장을 보였다. 두 번째는 훈련으로 인한 거짓말로, 기존 탐지기들이 훈련된 모델 유기체에 대해 매우 제한적인 성능을 보였음을 밝혔다.

- **Performance Highlights**: DYL 방법을 포함한 네 가지 탐지기는 모델의 능력과 함께 긍정적으로 성과를 보였으나, 훈련된 모델 유기체에서는 성능이 급락했다. 특별히, DYL은 일부 신호를 유지했으며, 체인 오브 생각(judge)은 0.82의 균형된 정확도를 기록하여 상대적으로 성능이 뛰어난 것으로 나타났다. 결과적으로, 현재의 거짓 탐지기는 모델 신념에 대해 높은 신뢰도를 지원할 수 없음을 보고하며, 이러한 한계를 극복하기 위한 연구 방향을 제안하였다.



### PersonaDrive: Human-Style Retrieval-Augmented VLA Agents for Closed-Loop Driving Simulation (https://arxiv.org/abs/2606.12616)
- **What's New**: 이 논문에서는 PersonaDrive라는 새로운 스타일 조건화된 VLA (vision-language-action) 주행 모델을 소개합니다. 이 모델은 인공지능이 제공하는 데이터에 기반하여 스타일 있는 주행 경로를 생성하고, 실제 사람의 주행 패턴을 반영함으로써 시뮬레이션 훈련 파이프라인에서 더 다양한 행동을 만듭니다. 특히, 동일한 기반 모델을 사용하여 다양한 스타일을 구현할 수 있어, 스타일 변경이 간단한 데이터베이스 쿼리로 가능하게 합니다.

- **Technical Details**: PersonaDrive의 핵심 전략은 스타일 지시를 기반으로 한 인간 주행 데이터에서 가져온 실행 예시를 활용하는 것입니다. 세 단계로 진행되는 파이프라인은 (i) 이미지-텍스트 유사도 점수를 사용한 스타일별 주행 데이터의 오프라인 트리플릿 마이닝, (ii) 스타일별 데이터베이스에 대해 미세한 제어 인코더와 결합된 경량의 검색 헤드를 훈련, (iii) 검색된 컨텍스트 포인트를 사용하여 웨이포인트 예측에서 행동 시연으로 활용하는 것입니다. 이 구조를 통해 각 스타일마다 별도의 모델 재훈련 없이도 스타일을 쉽게 변경할 수 있습니다.

- **Performance Highlights**: Bench2Drive 과제에서 PersonaDrive는 SimLingo에 비해 4.6%의 주행 점수 향상을 달성하였고, HiP-AD 대비 2.5% 상승했습니다. 또한, 모든 스타일 조건에서 가장 높은 주행 점수를 기록하였으며, 보수적인 지시에서 공격적인 지시로 조건을 전환할 때 속도와 가속도가 각각 18%와 25% 증가했습니다. 이러한 결과는 스타일 조건 변경이 단순한 인덱스 스왑을 통해 가능하다는 점에서 효율성을 강조합니다.



### Pythagoras-Prover: Advancing Efficient Formal Proving via Augmented Lean Formalisation (https://arxiv.org/abs/2606.12594)
Comments:
          Pythagoras-Prover: Technical Report

- **What's New**: Pythagoras-Prover는 컴퓨팅 자원을 효율적으로 사용할 수 있도록 설계된 오픈 소스의 Lean theorem provers 제품군입니다. 이 시스템은 4B 및 32B 파라미터의 자기 회귀 모델과 4B 파라미터의 확산 기반(proof-of-concept diffusion-based) theorem prover를 포함합니다. 이 새로운 접근 방법은 특히 긴 연산 과정 없이도 수학적 정리를 처리할 수 있도록 돕습니다.

- **Technical Details**: Pythagoras-Prover는 Lean-검증된 문제 집합을 Easy, Medium, Hard로 구분하여 커리큘럼 SFT(supervised fine-tuning)을 통해 훈련합니다. 이는 모델이 간단한 증명에서 복잡한 증명으로 점진적으로 발전할 수 있게 합니다. 또한 Augmented Lean Formalisation (ALF)을 도입하여 기존의 자료를 자가 증류(self-distillation)를 통해 다양한 공식 성명으로 확장합니다.

- **Performance Highlights**: Pythagoras-Prover-4B는 MiniF2F-Test에서 DeepSeek-Prover-V2-671B를 초과하는 성능(86.1% 대 82.4%)을 기록하며, 파라미터 수에서 약 167배 적습니다. 또한 Pythagoras-Prover-32B는 open-source Lean provers 중 가장 높은 성과를 기록하며 MiniF2F-Test에서 93%의 통과율을 달성했습니다.



### Strategic Decision Support for AI Agents (https://arxiv.org/abs/2606.12587)
- **What's New**: 이 논문에서는 결정 지원(Decision Support)의 기존 관념을 재검토하여 AI 에이전트가 중심 역할을 수행하는 상황에서 사람과 기계의 역할이 어떻게 역전되는지를 탐구합니다. 전통적으로 인간 의사 결정자는 기계 학습 모델의 예측 가이드를 받았습니다. 하지만 이제는 AI 시스템이 복잡하고 불확실한 환경에서 사용자를 대신해 행동함으로써 신뢰성 관련 문제들이 부각되고 있습니다. 이에 따라 AI 에이전트가 독립적으로 행동할지, 아니면 지원을 요청해야 할지를 결정하는 기본 원칙에 대한 재고가 필요합니다.

- **Technical Details**: 이 연구는 AI 에이전트와 결정 지원 메커니즘 간의 상호작용을 모델링하는 것을 시작으로 합니다. 지원의 가치와 비용을 분리하고, 지원 요청을 전략적 결정으로 간주하는 접근 방식을 제안합니다. 주요 오류는 지원이 도움을 줄 수 있었음에도 불구하고 에이전트가 혼자 행동하는 경우 발생하며, 이를 제어하기 위한 최적화 문제로 설정됩니다. 저자들은 이 기반 위에 SDS-Opt라 불리는 최적화 문제를 제안합니다.

- **Performance Highlights**: 실험 결과, 저자들은 다양한 시나리오에서 그들의 방법이 목표 오류를 신뢰성 있게 제어하는 동시에 불필요한 지원 요청을 크게 감소시킨다는 것을 보여주었습니다. 지원의 가치를 바탕으로 한 최적 정책은 간단한 임계값 규칙으로 요약되며, 이를 통해 총 네 가지 카테고리(정보 수집, 인간-기계 협력, 계획, 도구 사용)에 대한 실제 데이터 세트에 적용하여 효율적인 전략적 의사 결정을 구현할 수 있음을 입증하였습니다.



### Arbor: Tree Search as a Cognition Layer for Autonomous Agents (https://arxiv.org/abs/2606.12563)
- **What's New**: Arbor는 대규모 상태 기반(action space)에서 자율 에이전트의 인식 레이어로서 구조화된 트리 검색을 도입한 다중 에이전트 프레임워크입니다. 이전의 자율 최적화 시스템은 상태 무관 평가를 통해 단일 타겟에서 작동했지만, Arbor는 에이전트 간에 공유되는 작업 메모리 역할을 수행하며 가설의 점수를 매긴 검색 트리를 유지합니다. Arbor는 실패를 진단 신호로 처리해 후속 탐색을 재구성하며, 성공한 이전 결과에 따라 병목 현상을 조정하여 검색 범위를 확장합니다.

- **Technical Details**: Arbor는 LLM(대형 언어 모델)의 추론 최적화를 위해 Orchestrator와 Critic 에이전트를 결합한 활용성 있는 다중 에이전트 아키텍처를 제공합니다. Orchestrator는 최적화를 주도하고, Critic은 안정성과 측정 검증을 책임져 두 에이전트간 상호 감시 및 균형을 이루며 작업을 수행합니다. 각 에이전트의 기능은 하드 스킬(전문 영역)과 소프트 스킬(조정 프로토콜)로 분해되어 장시간 자동화 캠페인을 가능하게 합니다.

- **Performance Highlights**: Arbor는 AMD Instinct GPUs에서 운영된 6개의 생산 모델에서 최적화된 기반과 비교해 40%에서 193%의 처리량 향상을 달성했습니다. 이 프레임워크는 하드웨어에 구애받지 않으며 재현 가능성을 입증했습니다. Arbor는 여러 하드웨어 플랫폼에서 일반화되며, 테스트 간 변동이 2 퍼센트 포인트 이내로 유지되어 이 방법의 일관성을 보여줍니다.



### ToolSense: A Diagnostic Framework for Auditing Parametric Tool Knowledge in LLMs (https://arxiv.org/abs/2606.12451)
- **What's New**: ToolSense는 도구 카탈로그를 입력으로 받아 자동으로 세 가지 진단 벤치마크를 생성하는 오픈 소스 LLM 기반 진단 프레임워크입니다. 이 시스템은 RRB(Realistic Retrieval Benchmark), MCQ(다중 선택 질문) 탐색 벤치마크 및 QA(질문-답변) 탐색 벤치마크를 포함하여 기존 도구의 이해도를 평가합니다. ToolSense는 도구 이해의 실제 격차를 보여주고, 도구 재조합의 성능이 비정상적으로 낮은 몇 가지 모델 구성을 드러냅니다.

- **Technical Details**: ToolSense는 도구 카탈로그를 구현하여 다단계 과정을 통해 벤치마크를 생성합니다. RRB는 세 가지 모호성 수준에서 사용자가 실제 쿼리와 유사한 간결한 요청을 테스트하여 검색 시스템의 일반화를 평가합니다. 하드-네거티브 풀을 구축하고, 병렬 생성 계층을 통해 각 배치에 대해 질의가 생성되어 이중 검증을 수행합니다.

- **Performance Highlights**: ToolBench를 기반으로 ToolSense의 적용 결과, 특정 모델 구성들은 일반적인 벤치마크에서 높은 성능을 보였지만 현실적인 쿼리에서 50-64%의 성능 저하를 일으켰습니다. 이는 fetched 모델들이 본질적인 도구 지식을 상실하고 있을 가능성을 시사하며, Stage 2 훈련이 Stage 1에서 학습한 도구 지식을 파괴할 수 있음을 보여줍니다.



### Learning to Reason by Analogy via Retrieval-Augmented Reinforcement Fine-Tuning (https://arxiv.org/abs/2606.13680)
- **What's New**: 이 논문에서는 Retrieval-Augmented Reinforcement Fine-Tuning (RA-RFT)이라는 새로운 프레임워크를 제안합니다. RA-RFT는 언어 모델이 유사한 문제를 해결할 때 분석적 사고를 학습하도록 설계된 사후 훈련 방법론입니다. 이는 고품질의 비슷한 예시를 찾기 위해 gold-relevance distillation을 이용하며, 의미적 유사성보다는 추론 유용성에 기반하여 맥락을 정렬합니다.

- **Technical Details**: RA-RFT의 기본적인 구성 요소는 세 가지 단계로 이루어져 있습니다: (1) gold-relevance distillation은 선별된 추론 패턴과 타겟 문제 간의 전이 가능성을 평가하여 훈련 감독을 구축합니다. (2) reasoning-aware retriever training은 효과적인 맥락을 추출하기 위해 대비 학습을 사용하여 밀집 검색기를 학습합니다. 마지막으로, (3) reinforcement fine-tuning에서는 검색한 유사한 예를 훈련 프롬프트에 주입하여 정책 모델을 최적화합니다.

- **Performance Highlights**: RA-RFT는 AIME 2025, HMMT 2025 등의 경쟁 수준의 수학적 추론 벤치마크에서 뛰어난 성과를 보여줍니다. 예를 들어, Qwen3-1.7B 모델에서는 GRPO 대비 AIME 2025 average@32 정확도를 7.1점 향상시켰습니다. 이러한 결과는 RA-RFT가 추론 유용성에 기초한 검색이 강화 학습의 성능 향상에 필수적임을 나타냅니다.



### Mana: Dexterous Manipulation of Articulated Tools (https://arxiv.org/abs/2606.13677)
Comments:
          Project Page: this https URL

- **What's New**: Mana (Manipulation Animator)는 기존의 단단한 물체 중심의 조작에서 벗어나, 복잡한 구조를 가진 가공 도구 조작을 애니메이션 문제로 재구성한 새로운 프레임워크입니다. 이 시스템은 단순한 사용자 입력을 통해 수천 개의 조작 키프레임을 자동으로 생성하고, 이를 모션 플래닝(motion planning) 및 강화 학습(reinforcement learning)을 통해 조작 궤적으로 변환합니다. 이 접근법은 공구의 물리적 한계를 초과하는 시스템에서 비슷한 성능을 출력할 수 있도록 합니다.

- **Technical Details**: Mana는 조작 키프레임을 생성하는 과정과, 이러한 키프레임을 연결하는 궤적 생성을 통해 조작의 복잡성을 경감합니다. 먼저, 사용자는 툴 메쉬(tool mesh)의 기능적 영역을 간단히 클릭하여 정의하여 바람직한 키프레임을 생성합니다. 이후, 모션 플래닝과 강화 학습의 두 가지 기술을 함께 사용하여 조작 모션을 최적화하고 가벼운 도구 조작을 가능하게 합니다.

- **Performance Highlights**: Mana는 여러 형태와 크기의 조작 도구에 대해 제로샷(Zero-shot) 시뮬레이션-현실 전이를 성공적으로 수행하였으며, 각 도구에서 높은 성공률을 보여줍니다. 분석한 도구는 집게(tongs), 플라이어(pliers), 주사기(syringes), 클립(clothespins) 등 다양한 조작 메커니즘을 포함하며, 이는 모두 서로 다른 정밀도가 요구되는 기계적 문제를 나타냅니다. 이 결과는 Mana의 확장 가능성과 조작의 정밀도를 입증합니다.



### SpatialClaw: Rethinking Action Interface for Agentic Spatial Reasoning (https://arxiv.org/abs/2606.13673)
Comments:
          Project page: this https URL

- **What's New**: 이번 연구는 비전-언어 모델(VLMs)의 공간 추론 능력을 향상시키기 위해 SpatialClaw라는 새로운 프레임워크를 제안합니다. 기존의 도구 증강 에이전트는 Action Interface(행동 인터페이스)에 따라 성능이 제한되었으나, SpatialClaw는 Python 코드를 통해 상태 유지형 커널을 활용하여 더 유연하고 개방적인 공간 추론을 가능하게 합니다. 이를 통해 에이전트는 중간 결과에 따라 분석을 조정하고 다양한 시각적 입력을 처리할 수 있습니다.

- **Technical Details**: SpatialClaw는 입력 프레임과 다양한 인식 및 기하 자료가 있는 영속적인 Python 커널을 활용하여 에이전트가 각 단계에서 실행 가능한 셀을 작성하도록 합니다. 기존 연구들은 단일 경로 코드 실행 또는 구조화된 도구 호출 인터페이스에 의존해 왔지만, SpatialClaw는 코드 생성을 통한 유연한 조합을 지원합니다. 이는 에이전트가 이전 결과에 기초하여 동적으로 인식 결과를 조정하고 조합할 수 있도록 합니다.

- **Performance Highlights**: SpatialClaw는 20개의 공간 추론 벤치마크에서 평균 59.9%의 정확도를 기록하며, 최근의 공간 에이전트보다 +11.2 포인트 향상된 성능을 보여줍니다. 특히 4D 비디오 추론 및 다중 뷰 추론에서 가장 높은 성과를 보였으며, 이는 전처리된 도구 호출 없이도 에이전트가 유연하게 기하적 계산을 수행함을 시사합니다. 이러한 성과는 다양한 모델 가족에서 일관되게 나타나, 시스템 프롬프트나 도구 세트의 수정 없이도 일반화됩니다.



### SkMTEB: Slovak Massive Text Embedding Benchmark and Model Adaptation (https://arxiv.org/abs/2606.13647)
Comments:
          ACL 2026

- **What's New**: 이번 논문에서는 슬로바키아어를 위한 최초의 포괄적인 MTEB 스타일 텍스트 임베딩 벤치마크인 SkMTEB를 소개합니다. 이 벤치마크는 7가지 작업 유형에 걸쳐 31개의 데이터셋으로 구성되어 있으며, 슬로바키아어를 위한 기존 다국어 벤치마크 커버리지의 거의 4배에 해당합니다. 논문에서는 대규모 명령어 조정된 다국어 모델들이 가장 우수한 성능을 보여주며, 슬로바키아 특정 모델이 임베딩 작업에 대한 전이가 좋지 않다는 것을 보여줍니다.

- **Technical Details**: 슬로바키아어에 대한 임베딩의 효율성을 높이기 위해, 저자들은 Multilingual E5 모델에 대해 어휘 절단(vocabulary trimming)과 세밀조정(fine-tuning)을 적용하여 e5-sk-small (45M 파라미터)과 e5-sk-large (365M 파라미터)를 개발했습니다. 이러한 모델은 최대 62%의 크기 감소에도 불구하고, 경제적인 비용으로 세멘틱 검색(semantic search) 및 검색 중심 생성(RAG)에 적합한 성능을 유지합니다. SkMTEB 벤치마크는 슬로바키아어에 대한 robust한 평가 벤치마크를 제공하여, 효율적인 모델 개발을 위한 기초를 마련합니다.

- **Performance Highlights**: 개발된 오픈소스 모델들은 상업적인 API와 경쟁력 있는 성능을 획득하면서도 지역적으로 배포 가능한 특성을 지닙니다. 이 논문에서는 31개의 공개 가중치 및 상용 임베딩 모델을 평가하여 슬로바키아어에 특화된 임베딩 모델의 발전 가능성을 확인했습니다. 전체 모델, 데이터셋 및 코드는 공개되어 있으며, 다른 자원이 부족한 언어들에 대해서도 적용 가능한 접근 방식을 제시하고 있습니다.



### Valid Inference with Synthetic Data via Task Exchangeability (https://arxiv.org/abs/2606.13629)
- **What's New**: 이 논문은 과학 연구에서 합성 데이터(synthetic data)의 유용성을 논의하며, 'task exchangeability'라는 새로운 개념을 소개합니다. 이는 연구자가 역사적 작업(historical tasks)을 식별할 수 있게 하여, 해당 작업과 현재 작업 간의 교환 가능성을 확보하는 과정입니다. 합성 데이터가 연구자들에게 더 많은 질문을 제기하고, 연구의 속도를 높일 수 있지만, 이와 동시에 데이터의 편향(bias)과 노이즈(noise) 문제 또한 경고하고 있습니다.

- **Technical Details**: 이 연구에서 제안하는 방법론은 역사적 데이터와 합성 데이터의 결합을 통해 유효한 추론(valid inference)을 가능하게 합니다. 연구자는 기존의 역사적 작업을 통해 합성 데이터의 오차 범위를 측정하고, 이를 기반으로 현재 작업의 신뢰 구간(confidence interval)을 확장합니다. 이러한 방식은 'task exchangeability'라는 통계적 조건에 따라 유효성을 확보하며, 이는 합성 데이터의 불확실성을 반영합니다.

- **Performance Highlights**: 결과적으로, 본 연구에서는 합성 데이터 기반의 신뢰 구간이 실제 값과의 차이를 반영하여 더 신뢰할 수 있는 결과를 제공합니다. 예를 들어, 여론 조사 데이터를 통해 검증된 실험 결과는 합성 데이터만을 사용하는 것보다 더 넓은 범위를 제시하며, 진짜 추정치(true estimand)를 포함하는 신뢰 구간을 생성합니다. 이러한 접근 방식은 합성 데이터의 유용성을 극대화하면서도 위험을 최소화할 수 있도록 합니다.



### One Polluted Page Is Enough: Evaluating Web Content Pollution in Generative Recommenders (https://arxiv.org/abs/2606.13610)
- **What's New**: 본 논문에서는 FORGE(Fake Online Recommendations in Generative Environments)라는 새로운 벤치마크를 제시하여 오염된 웹 콘텐츠에 대한 생성 추천 시스템의 취약성을 측정하는 방법을 개발합니다. 검색을 통해 실시간 웹 페이지를 가져오는 LLM은 가짜 리뷰와 프로모션 페이지와 같은 오염된 콘텐츠를 수용함으로써 가짜 제품의 홍보자가 될 수 있는 위험이 있습니다. 이 연구는 LLM이 얼마나 자주 가짜 제품을 추천하는지를 측정할 수 있는 새로운 접근 방식을 제공합니다.

- **Technical Details**: FORGE는 사용자 쿼리, 라이브 웹 검색, 최상위 증거 번들, LLM 소비, 순위 추천의 절차를 통해 구축됩니다. 원래 제품의 언급을 오염시키는 대신 실제 브랜드를 가짜 브랜드로 변경하여, 모델의 추천이 어떻게 변하는지를 분석합니다. FORGE는 15개 카테고리와 225개의 실제 제품을 포함하며, 12개의 상업적 및 오픈 소스 LLM 모델에서의 취약성을 테스트합니다.

- **Performance Highlights**: 연구 결과, 모든 모델이 취약하며, 단일 오염된 페이지가 최대 27%의 잘못된 추천을 생성할 수 있습니다. 더군다나, 상위 3개 제품이 대체되면 이 비율은 73.8%로 증가했습니다. 모델의 브랜드 지식이 약할수록 취약성이 증가하며, '사회적 증거'를 생성하여 잘못된 추천을 정당화하는 경우도 관찰되었습니다.



### Beyond the Commitment Boundary: Probing Epiphenomenal Chain-of-Thought in Large Reasoning Models (https://arxiv.org/abs/2606.13603)
- **What's New**: 본 논문은 Chain-of-Thought (CoT) 추론의 각 단계가 최종 답변에 미치는 인과적 영향(causal influence)을 평가합니다. 저자들은 조기 종료(early exit)를 통해 각 단계의 인과적 중요성(causal importance)을 추정하며, 답변이 여러 모델 계열의 추론(trace)에서 어떻게 형성되는지를 연구합니다. 이 연구는 전통적인 CoT 접근 방식에서의 깊이 있는 통찰력을 제공합니다.

- **Technical Details**: 연구에서는 다양한 작업(Task)에서 추론이 일반적으로 'commitment boundary'를 초과한다고 발견했습니다. 이 경계는 일시적인 중간 추측(intermediate guesses)에서 안정적이고 높은 신뢰도의 답변으로의 급격한 전환을 나타냅니다. 저자들은 주목 프롭(attention probes)을 활용하여 중간 추론 단계로부터 답변 형성(answer-formation) 단계를 선형으로 디코딩하는 데 성공했으며, 이는 기존의 추론 작업에서도 뛰어난 일반화 성능을 보였습니다.

- **Performance Highlights**: 조기 종료 메커니즘을 사용하여 모델의 추론 블록을 commitment boundary에서 조기 종료함으로써 평균 55%까지 CoT의 길이를 줄일 수 있었습니다. 이러한 조정이 모델 성능에 미치는 영향은 미미하였습니다. 이는 CoT 접근법의 효율성을 대폭 향상시키는 결과로 이어집니다.



### EvTexture++: Event-Driven Texture Enhancement for Video Super-Resolution (https://arxiv.org/abs/2606.13580)
Comments:
          IEEE TPAMI 2026. Extended version of arXiv:2406.13457 (ICML 2024). Project page: this https URL

- **What's New**: 이 논문은 Event-based vision 기술을 활용하여 비디오 슈퍼 해상도(Video Super-Resolution, VSR)의 텍스처 복원에 초점을 맞춘 EvTexture++라는 프레임워크를 제안합니다. 기존 VSR 기법들이 모션 정제(motion refinement)에 초점을 맞춘 반면, EvTexture++는 텍스처 향상(texture enhancement)을 목표로 합니다. 이 시스템은 이벤트 기반 신호를 바탕으로 고주파(spatiotemporal) 세부 정보를 활용하여 더 정확하고 상세한 고해상도 출력을 생성합니다.

- **Technical Details**: EvTexture++는 텍스처 향상 전용 브랜치와 고해상도 영상의 정밀한 특성을 복원하기 위한 반복적 질감 향상 모듈을 포함하고 있습니다. 이 개선 과정은 고주파 텍스처 정보를 점진적으로 복구하면서, 이벤트 신호의 연속적 시간 모션 단서를 활용하여 텍스처 변동성을 줄입니다. 전체 시스템은 기존의 VSR 모델에 간편하게 통합될 수 있도록 설계되어, 다양한 아키텍처에서 실행 가능합니다.

- **Performance Highlights**: 다수의 데이터셋에서 실험 결과, EvTexture++는 최신 성능 기준(State-of-the-art, SOTA)을 달성했습니다. 특히, 텍스처가 풍부한 Vid4 데이터셋에서 PSNR에서 최대 1.55 dB의 유의미한 개선을 보였습니다. 이 논문에서 제안된 모델은 기존 VSR 모델과의 통합을 통해 향상된 성능을 창출할 수 있음을 입증하였습니다.



### LabVLA: Grounding Vision-Language-Action Models in Scientific Laboratories (https://arxiv.org/abs/2606.13578)
Comments:
          Work in progress. Project website at this https URL

- **What's New**: 최근 과학 실험에서 AI 시스템의 도움이 더해지고 있으나, 물리적 실험 실행은 여전히 인간 조작자에게 의존하고 있습니다. 비전-언어-액션 (Vision-Language-Action, VLA) 모델은 이 과정을 연결할 수 있는 한 가지 방법을 제공합니다. 하지만 기존의 정책들은 주로 가정용 및 테이블탑 환경에 대해 훈련되어 과학 실험에 필요한 정밀한 조작 지식이 부족합니다. 이에 따라 RoboGenesis라는 시뮬레이션 기반 데이터 엔진이 개발되어 실험 프로토콜을 더 향상시키고 있습니다.

- **Technical Details**: RoboGenesis는 실험 환경을 구축하고, 자동 조작 스킬을 기반으로 프로토콜을 생성하여 성공 사례를 필터링하는 과정으로 구성되어 있습니다. 이 엔진은 실험 장비를 다양하게 사용하여 실제 실험실 데이터 수집의 높은 비용을 줄이는 데 중점을 두고 있습니다. LabVLA는 로봇의 상태, 언어 지시 및 시각적 관찰을 결합하여 계속적인 동작 토큰을 생성하는 방식으로 설계되었습니다. 정책 훈련의 두 단계는 초기 FAST 행동 토큰 사전 훈련과 후속 유동 매칭을 포함합니다.

- **Performance Highlights**: LabVLA는 LabUtopia 벤치마크에서 평가한 모든 기준선보다 평균 성공률이 가장 높았습니다. 이 성공적인 결과는 RoboGenesis가 생성한 데이터의 질과 다양성이 큰 기여를 했다는 것을 시사합니다. LabVLA는 다양한 로봇 에지 효과와 환경에서 프로토콜을 실행하는 능력을 보유하고 있어, 앞으로의 과학 실험 자동화에 중요한 역할을 할 것으로 기대됩니다.



### ArogyaSutra: A Multi-Agent Framework for Multimodal Medical Reasoning in Indic Languages (https://arxiv.org/abs/2606.13572)
- **What's New**: 이 논문은 의료 분야에서 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 성능 격차를 해소하기 위한 두 가지 주요 기여를 소개합니다. 첫 번째로, ArogyaBodha라는 대규모 다국어 다중모달 의료 질문-답변 데이터셋을 구축하였으며, 이는 31개 신체 시스템과 21개 임상 분야를 포함하고 있습니다. 두 번째로, ArogyaSutra라는 다중 에이전트 프레임워크를 제안하여 이미지와 텍스트 입력에 대한 단계별 의사 결정을 지원합니다.

- **Technical Details**: ArogyaBodha 데이터셋은 8개의 서로 다른 의료 출처로부터 수집되었으며, 영어와 7개의 주요 인도 언어를 포함합니다. 이 데이터셋은 전문가 검증을 통해 비즈니스 논리의 정확성과 언어 일관성을 평가할 수 있도록 구성되어 있습니다. ArogyaSutra 프레임워크는 도구 기반의 시각적 기초와 이중 기억 메커니즘을 결합하여 단계별 의사 결정 과정을 효과적으로 수행합니다.

- **Performance Highlights**: 실험 결과, ArogyaSutra는 다양한 언어, 이미징 방식 및 임상 도메인에서 뛰어난 성능을 보이며 기존의 강력한 대체 모델보다 일관된 결과를 도출합니다. 본 논문에서 제안한 접근 방식은 저자원이지만 다국어 환경에서도 신뢰할 수 있는 의료 AI 시스템을 지원할 수 있는 가능성을 보여줍니다. 이러한 연구는 공정하고 포괄적인 의료 접근을 증진시킬 수 있는 의의를 갖습니다.



### Existence Precedes Value: Joint Modeling of Observational Existence and Evolving States in Time Series Forecasting (https://arxiv.org/abs/2606.13571)
- **What's New**: 이 논문은 Timeflies라는 새로운 통합 프레임워크를 제안합니다. 기존의 시계열 예측에서 미래 관측의 가능성을 동시에 추론하고 값의 추정을 수행하는 방식으로, 예측을 존재 추론(existence inference)과 값 추정(value estimation)이라는 두 가지 과정으로 재정의합니다. 이 논문은 관측 회귀성과 상태 진화를 연계하여 미래의 관찰 가능성을 명시적으로 모델링하는 방법을 소개합니다.

- **Technical Details**: Timeflies는 관측 스트림(observation stream)과 값 스트림(value stream)으로 구성된 대칭 설계를 통해 동적 프로세스를 캡처합니다. 이 구조는 신뢰도 기반 패치 임베딩(reliability-gated patch embedding), 관측 조건화 Transformer(cross-track conditioned transformer), bifurcated observation-evolution head 등의 세 가지 모듈을 통해 작동합니다. 이를 통해 정보의 관찰 가능성과 그 가치 간의 상호작용을 정량적으로 평가할 수 있는 Observation-Value Joint Entropy (OVJE) 메트릭을 도입하였습니다.

- **Performance Highlights**: Timeflies는 Shadow라는 벤치마크를 이용해 자연 결측치와 실제 산업 데이터를 조합하여 실험을 진행하였습니다. 그 결과, Timeflies는 기존 방법들에 비해 지속적으로 뛰어난 성능을 보였으며, 이는 미래 관측 가능성을 명시적으로 모델링하는 것이 시계열 예측의 중요한 요소임을 강조합니다. 새로운 평가 기준으로 구성된 OVJE에 기반하여, Timeflies는 기존의 예측 모델링과 비교해 실질적인 효용성을 확보했습니다.



### Contrast-Informed Augmentation and Domain-Adversarial Training for Adult-to-Neonatal MR Reconstruction Generalization (https://arxiv.org/abs/2606.13562)
Comments:
          24 pages, 1 table, 7 figures

- **What's New**: 본 연구는 E2E-VarNet의 성능을 향상시키기 위해 contrast-informed 데이터 증강(data augmentation)과 domain-adversarial training(DAT)의 효과를 조사했다. 세 가지 학습 방식이 탐구되었으며, 그 중 혼합 학습 방식이 neonatal 데이터에 대해 뛰어난 성능을 보였다. 이 연구는 성인 데이터를 효과적으로 활용하여 neonatal MR reconstruction의 일반화 성능을 높이는 것을 목표로 한다.

- **Technical Details**: 연구에서는 unaugmented adult 데이터만 사용한 경우와 unaugmented 및 augmented adult 데이터 혼합, 그리고 DAT 목표를 추가한 혼합 데이터의 세 가지 학습 방식에 대해 성능을 비교하였다. 모델은 각 교육 방식에 따라 T2-weighted 뇌 MR 데이터로 훈련되었으며, 도메인 구분이 유지되는지 여부를 평가하기 위해 latent representation의 변화를 분석하였다. 특별히, data augmentation은 neonatal MR 특징을 근사하도록 설계되었다.

- **Performance Highlights**: Mixed-DAT 방식은 neonatal 데이터에서의 평가 결과, 최고의 성능을 보였다. R=4에서 SSIM 값을 0.924로 기록하였고, R=8에서는 SSIM 0.848 및 PSNR 29.56 dB을 기록하여, 성인 전용 학습 방식에 비해 우수한 성능을 입증하였다. 이러한 결과들은 domain adjersarial training이 강조된 데이터 증강과 함께 사용될 때, neonatal MR reconstruction의 일반화 성능을 개선할 수 있음을 나타낸다.



### Adaptive Turn-Taking for Real-time Multi-Party Voice Agents (https://arxiv.org/abs/2606.13544)
Comments:
          Accepted for publication at Interspeech 2026

- **What's New**: 이 연구에서는 다자간 대화에서의 턴 테이킹(turn-taking) 행동을 역할에 기반하여 조정하는 음성 기반 에이전트 ModeratorLM을 제안합니다. 이 시스템은 대규모 음성 언어 모델을 기반으로 하여 청크(chunk) 단위로 스트리밍됩니다. 또한 대화의 맥락과 할당된 역할에 대한 사고 과정(chain-of-thought reasoning)을 통합한 변형인 ModeratorLM-Think도 도입되었습니다. 이를 통해 비할당 역할 모델에 비해 턴 테이킹의 정밀도는 40% 이상, 재현율은 70% 이상 향상되었습니다.

- **Technical Details**: ModeratorLM은 음성 인코더와 기본 언어 모델(LLM)로 구성되어 있습니다. 음성 인코더는 각 오디오 청크를 독립적으로 처리하여 청크 수준 임베딩을 생성합니다. 이 임베딩은 가벼운 선형 프로젝션 레이어를 통해 LLM 임베딩 공간으로 투영되며, 스트리밍 방식으로 LLM 컨텍스트에 순차적으로 추가됩니다. 각 입력 오디오 청크에 대해 모델은 턴 테이킹 및 응답 생성 중 하나를 선택하여 결과를 생성합니다.

- **Performance Highlights**: 실험 결과, ModeratorLM은 비역할 조건의 기준선보다 할당된 역할과의 정렬이 현저히 향상되었습니다. 실험에 사용된 RolePlayConv 데이터셋과 현실 세계의 회의 데이터에서 성능 개선이 확인되었습니다. 특히, 턴 테이킹의 정밀도와 재현율이 대폭 향상되었으며, 잘못된 긍정적(interruption) 반응을 줄였습니다.



### AgentRivet: an automated system for producing Rivet routines from journal publications (https://arxiv.org/abs/2606.13535)
- **What's New**: 이 논문에서는 Rivet 루틴을 자동으로 생성하는 AgentRivet라는 새로운 AI 워크플로우를 제안합니다. 현재 Rivet 루틴이 문서화되어 제공되지 않는 측정값에 대해 39%만이 해당 루틴을 가지고 있어 분석의 범위가 부족합니다. AgentRivet는 대형 언어 모델(LLMs)을 활용하여 출판된 논문에서 물리학 분석 정보를 추출하고 누락된 Rivet 루틴을 작성하는 데 초점을 맞추고 있습니다. 이 워크플로우는 코드와 물리적 구현을 검토하는 중간 리뷰 단계가 포함되어 있습니다.

- **Technical Details**: AgentRivet는 모듈식 설계 원칙에 기반하여 여러 LLM 제공자와의 협업을 가능하게 합니다. 이를 통해 사용자는 여러 제공자 간에 동일한 논리로 작동하는 워크플로우를 사용하면서도 각 제공자의 고유한 기능을 활용할 수 있습니다. 이 프레임워크는 구조화된 출력을 정의하고, LLM의 반응이 이러한 스키마에 따라 검증되도록 하여 정보의 일관성을 보장합니다. AgentRivet는 코딩 프로세스와 물리적 검토를 독립적인 두 개의 전담 에이전트로 나누어 진행합니다.

- **Performance Highlights**: AgentRivet는 OpenAI, Anthropic, Google의 상용 LLM을 사용하여 성능을 평가하였으며, 생산된 Rivet 루틴에서 몇 가지 구문 오류만 발생하는 것으로 나타났습니다. 물리적 충실도는 양호하며, 관련 출판물에서 설명된 대로 구현된 것을 확인했습니다. 그러나 정교한 정의에도 불구하고 복잡한 관측치를 구현하는 데 어려움이 발생하기도 하였으며, 이로 인해 물리적 구현 문제를 조사하였습니다.



### Measurement-Calibrated Multi-Camera Fusion for Vision-Based Indoor Localization (https://arxiv.org/abs/2606.13509)
Comments:
          This paper has been accepted for presentation at the IEEE 22st International Conference on Automation Science and Engineering (CASE 2026)

- **What's New**: 이 논문에서는 실내 비전 기반 로컬라이제이션 시스템에서의 데이터 융합(data fusion)의 중요성을 강조합니다. 기존의 멀티 카메라 시스템이 일반적으로 블랙 박스(component)를 사용하여 평가되는 반면, 본 연구는 단일 카메라의 오류 특성을 명시적으로 분석하여 멀티 카메라 데이터 융합을 보정하고 최적화하는 방법을 제안합니다. 측정-보정된 융합(measurement-calibrated fusion) 접근법을 통해 오류의 기여도를 정량화하여 개선된 로컬라이제이션의 정확성을 입증했습니다.

- **Technical Details**: 이 시스템은 객체 인식 및 분류를 위한 신경망과 공간적 로컬라이제이션을 위한 투영 변환(projective transformation)을 결합합니다. 합쳐진 이미지에서 객체를 YOLO v8n을 사용하여 탐지하고, MediaPipe를 통해 2D 인체 자세추정을 수행하여 각 객체의 지면 접촉 위치를 계산합니다. 데이터를 융합하기 위해 선형 칼만 필터(linear Kalman Filter)를 사용하였으며, 각 카메라의 효과적 시야(field of view)를 고려하여 노이즈 특성을 조정합니다.

- **Performance Highlights**: 실험 결과, 데이터 융합은 단일 카메라 기반의 시스템에 비해 로컬라이제이션 정확도를 상당히 향상시켰습니다. 측정-보정된 융합 방법은 절대적인 정확도 향상은 제한적이었지만, 궤적의 변동성을 크게 줄이고 움직임의 부드러움을 향상시켜 안정적이고 연속적인 모션 추정이 필요한 응용 프로그램에서 요구되는 조건을 더 잘 충족시켰습니다. 이러한 결과는 비전 기반 실내 위치 추적 시스템의 데이터 융합 전략 설계 시 명시적인 오류 특성화의 가치를 분명히 보여줍니다.



### Heterogeneous LiDAR Early Fusion and Learned Re-Ranking Strategy for Robust Long-Term Place Recognition in Unstructured Environments (https://arxiv.org/abs/2606.13503)
- **What's New**: 새로운 접근법인 MinkUNeXt-VINE++는 두 개의 서로 다른 LiDAR 센서(즉, Livox Mid-360 및 Velodyne VLP-16)로부터의 이종 데이터를 조기에 융합하고 추론 시간에 학습된 재순위화 전략을 통합하는 방식을 제안합니다. 이 방법은 농업과 같은 비구조적 환경에서의 장소 인식 문제를 해결하기 위해 개발되었습니다. 연구 결과, MinkUNeXt-VINE++는 단일 센서 접근 방법에 비해 장소 인식 성능을 현격하게 개선하는 것을 보여줍니다.

- **Technical Details**: MinkUNeXt-VINE++는 이질 리다 데이터의 조기 융합 전략과 후보 장소의 최종 순위를 개선하기 위한 학습된 재순위화 접근법을 결합한 것입니다. 이러한 융합 전략은 각 센서의 강점을 활용하여 환경의 보다 포괄적인 표현을 제공합니다. 이 방법은 TEMPO-VINE 데이터셋을 사용하여 평가되었으며, 다양한 생리적 단계에서 포도원의 이질적 LiDAR 데이터를 제공합니다.

- **Performance Highlights**: MinkUNeXt-VINE++는 단일 센서 접근 방법에 비해 Recall@1 지표에서 각각 20% 개선을 달성하며, 재순위화를 포함할 경우 +30%의 추가 개선을 기록했습니다. 이러한 결과는 비구조적 환경에서 장소 인식의 정확성을 크게 높임을 보여줍니다. 또한, 연구는 넓은 범위의 농업 환경에서도 효과적으로 작동함을 입증했습니다.



### CRAFTIIF: Cross-Resolution Analytic Four-Type Interpretable Isolation Forest for Multivariate Time Series Anomaly Detection (https://arxiv.org/abs/2606.13486)
Comments:
          14 pages, 4 figures, 2 appendices. Submitted to IEEE Transactions on Knowledge and Data Engineering (TKDE). Code: this https URL

- **What's New**: CRAFTIIF는 다변량 시계열에서의 이상 감지를 위한 새로운 프레임워크로, 서로 다른 4종의 이상 유형을 타겟으로 설정합니다. 이 프레임워크는 포인트, 분포, 시간, 집단적인 이상을 감지할 수 있으며, 데이터셋에 특정한 조정 없이 완전히 비지도 방식으로 작동합니다. CRAFTIIF는 500개의 무작위 분석 웨이브릿 특징을 생성하여 5개의 구조화된 Isolation Forests를 훈련시킵니다.

- **Technical Details**: CRAFTIIF는 Morlet, DOG, Haar, Coiflet의 네 가지 웨이브릿 가족을 이용하여 특정 이상 유형 간의 상호작용을 방지합니다. 이 프레임워크는 최적의 감지를 위한 자동 조정을 가능하게 하며, Otsu/MAD 기준을 통해 다양한 이상률에 적응하도록 설계되었습니다. 또한, 각 유형에 특화된 특징으로 독립적으로 훈련된 Isolation Forests를 통해 이상 유형에 대한 직접적인 귀속을 제공합니다.

- **Performance Highlights**: CRAFTIIF는 mTSBench 벤치마크의 19개 데이터셋에서 평균 F1 점수 0.228, 감지 가능한 13개 데이터셋에서 F1 점수 0.322를 기록하여 모든 25개 평가 방법 중 1위를 차지하였습니다. 또한, VUS-PR에서도 0.463을 기록하여 이전 최고치인 0.329을 40.7% 초과하였습니다. 여러 조건에 대한 ablation 연구를 통해 적응형 임계값과 구조적 요소들이 성능에 기여하는 바를 확인하였습니다.



### SupraBench: A Benchmark for Supramolecular Chemistry (https://arxiv.org/abs/2606.13477)
- **What's New**: 이 논문은 supramolecular chemistry (초분자 화학)에서 호스트-게스트 시스템 설계를 위해 LLMs (대형 언어 모델)의 성능을 평가하기 위한 최초의 벤치마크인 SupraBench를 소개합니다. 또한, SupraPmc라는 1,600만 토큰의 초분자 화학 기사를 제공하여 연구자들이 이 분야에 적응할 수 있도록 지원합니다. 이 연구는 기존의 수십 년간의 전문가 반복 과정에서 발생하는 시간 소모를 줄이기 위해 LLM을 활용하는 새로운 방법을 모색합니다.

- **Technical Details**: SupraBench는 바인딩 친화도 예측, 상위 바인더 선택, 용매 식별, 호스트-게스트 설명의 네 가지 기본 작업과 분자 식별을 위한 보조 비전 작업으로 구성됩니다. 연구자들은 기존의 LLM과 도메인 적응 사전 훈련된 버전을 포함하여 다양한 LLM을 평가하였으며, 각 작업에서 모델 간의 성능 차이를 관찰했습니다. 평가 결과는 LLM들이 현재의 초분자 화학 추론에서 상당한 개선 여지가 있음을 보여줍니다.

- **Performance Highlights**: 모델들의 성능 평가 결과, LLM들은 모든 작업에서 상당한 격차를 보이며, 도메인 적응 전훈련이 적당한 개선 효과를 나타냈지만 특정 포맷 출력에서는 트레이드오프가 발생했습니다. 각 작업군 별로 난이도가 뚜렷하게 표현되며, 이를 통해 현재 초분자 화학 추론에서의 고유한 실패 모드를 파악할 수 있었습니다. 연구자들은 SupraBench와 관련 텍스트 코퍼스가 이 분야 연구에 크게 기여할 것이라고 믿고 있습니다.



### MaxProof: Scaling Mathematical Proof with Generative-Verifier RL and Population-Level Test-Time Scaling (https://arxiv.org/abs/2606.13473)
- **What's New**: 본 논문은 수학적 증명에 대한 경쟁 수준의 테스트를 위한 테스트 시간 확장 프레임워크인 MaxProof를 소개합니다. M3 시리즈는 증명 생성, 증명 검증, 비판 조건 증명 수정의 세 가지 증명 지향적 기능을 훈련시켜, 이를 하나의 모델로 통합합니다. MaxProof는 모델을 생성기, 검증자, 정제자, 랭커로 활용하여 후보 증명의 집합에서 최종 증명을 선택하는 방식을 채택하며, IMO 2025에서 35/42, USAMO 2026에서 36/42의 성과를 달성했습니다.

- **Technical Details**: M3 모델의 설계 과정에서는 세 가지 주요 기능, 즉 증명 생성, 증명 검증, 증명 정제를 다루었습니다. 각 기능은 전용 훈련 단계를 통해 개발되었으며, Proof Expert는 생성 검증기에서의 보상을 토대로 훈련됩니다. 이 과정에서 기존 권장사항과 비교하여 오류를 식별하고 텍스트 평가 및 점수를 제공하여 전체 증명의 보상으로 사용합니다.

- **Performance Highlights**: M3 모델은 IMOProofBench 및 IMOAnswerBench에서 성능 격차를 줄이며, MaxProof를 활용하여 수학적 증명을 위한 경쟁 레벨에서 특히 뛰어난 결과를 보여주었습니다. 최종적으로, M3는 IMO 2025에서 35/42, USAMO 2026에서 36/42의 성과로 인간 금메달 기준을 초과하며, 이를 통해 수학적 증명의 대중적 이해와 응용 가능성을 더욱 확장할 수 있음을 시사합니다.



### Understanding the Rejection of Fixes Generated by Agentic Pull Requests -- Insights from the AIDev Datas (https://arxiv.org/abs/2606.13468)
Comments:
          5 pages, 2 figures, MSR '26: Proceedings of the 23rd International Conference on Mining Software Repositories, April 2026, Rio de Janeiro, Brazil

- **What's New**: 이번 연구는 AI 코딩 에이전트가 소프트웨어 프로젝트에서 제공하는 코드 수정 사항 중 46.41%가 거부되는 현상을 분석했습니다. 이는 에이전트들이 생성한 수정사항이 인간의 검토와 검증이 필요하고, 결국 소모되는 자원을 초래함을 의미합니다. 연구의 목적은 이러한 AI 에이전트의 실패 요인을 이해하여 더 효율적인 팀원으로 통합하는 것입니다.

- **Technical Details**: 연구는 AIDev 데이터셋을 활용해 306개의 비병합 풀 리퀘스트를 대상으로 질적 및 양적 분석을 실시하였습니다. 거부 이유는 주제별로 14개의 카테고리로 나뉘며, 주요 요인은 구현이 잘못된 경우, CI 파이프라인을 통과하지 못하는 경우, 에이전트가 구현을 수행할 수 없는 경우, 그리고 수정 사항의 우선순위가 낮은 경우입니다. 연구 결과는 AI 에이전트가 문제를 해결하는데 더 나은 방향을 제시하기 위한 필요성을 강조합니다.

- **Performance Highlights**: 연구 결과, 코드 변경 측면에서 AI 에이전트 수정을 거부할 때의 낭비된 노력은 상당한 코드 변화와 주고받은 댓글 수로 측정되었습니다. 평균적으로 코드 변화의 범위는 81에서 293 사이로 나타났습니다. 연구는 개발자들이 에이전트를 보다 잘 안내하고, 저우선 순위 수정에 대한 에이전트 사용 최소화를 권장하는 메시지를 제공합니다.



### Ontology Memory-Augmented ASR Correction for Long Text-Speech Interleaved Conversations (https://arxiv.org/abs/2606.13464)
- **What's New**: 이 논문은 자율 음성 인식(ASR) 수정에 대한 새로운 접근법을 제시합니다. 기존의 ASR 수정 방법이 단기적인 문장이나 지역적인 맥락에 집중했던 반면, 이 연구는 대화 전체의 맥락을 이용하여 긴 대화에서의 ASR 수정을 다룹니다. 새로운 온톨로지 메모리 기반의 ASR 수정 프레임워크를 설계하여 대화의 맥락과 관련된 정보를 구조적으로 관리할 수 있도록 합니다.

- **Technical Details**: 연구에서는 이전 대화 이력을 온톨로지 메모리라는 동적으로 업데이트 가능한 구조에 저장하여 ASR 수정에 필요한 관련 정보를 효율적으로 접근합니다. MAGIC-RAMC 데이터셋을 기반으로 한 RAMC-Corr 데이터셋을 구축하여 긴 텍스트-음성이 혼합된 대화에서의 ASR 수정 성능을 평가합니다. 모델은 각 ASR 가설에서 관련 증거를 검색하여 수정 작업을 수행하며, 대화의 전체 진행 중에 메모리를 계속 갱신합니다.

- **Performance Highlights**: RAMC-Corr 데이터셋에서 실험한 결과, 제안한 방법은 10개의 다양한 백본 설정 조합 중 9회에서 직접 수정보다 향상된 성능을 보였습니다. 이로써 이 연구가 제안하는 프레임워크가 ASR 오류 수정의 효과성을 높이는 데 기여할 수 있음을 보여줍니다. 즉, 대화 중 축적된 맥락을 활용함으로써 더 선택적이고 증거 기반의 수정이 가능해집니다.



### Toward Instructions-as-Code: Understanding the Impact of Instruction Files on Agentic Pull Requests (https://arxiv.org/abs/2606.13449)
Comments:
          5 pages, 8 figures, 23rd International Conference on Mining Software Repositories, April 13--14, 2026

- **What's New**: 이번 연구에서는 AI-agents(예: GitHub Copilot)가 소프트웨어 공학 과제에서 협력하는 방식과, 이러한 에이전트를 위한 instruction files의 작성이 agentic-PR(풀 리퀘스트) 성과에 미치는 영향을 탐구합니다. 15,549개의 agentic PR을 분석한 결과, instruction files의 추가가 반드시 좋은 결과를 보장하지 않는다는 것을 발견하였습니다. 이 연구는 AI-agents가 더 효과적으로 개발 업무를 수행하기 위해 필요한 instructions-as-Code라는 새로운 개념을 제안합니다.

- **Technical Details**: 연구는 AIDev 데이터셋을 이용해 148개 프로젝트에서 수집한 15,549개의 agentic PR을 분석하였습니다. 각 프로젝트에서 instruction files 작성 전후의 성과 변화를 비교하여 merge rate, code churn, 수정된 파일 수와 같은 다양한 지표를 검사했습니다. 또한, instruction files의 길이와 구조가 merge rate와의 상관관계를 나타내는지 여부를 조사하여, verbosity가 높은 instruction files가 더 나은 성과와 연관이 있음을 발견했습니다.

- **Performance Highlights**: 연구 결과 27.7%의 프로젝트가 instruction files 작성 후 merge rate가 20% 이상 증가한 반면, 26.35%는 감소했습니다. task complexity와 merge effort의 측면에서도 유사한 경향을 보였고, 특정 프로젝트는 상당히 긴 instruction files로 인해 성과가 증가하는 경향을 보였습니다. 이러한 결과는 개발자가 효과적인 instruction files를 작성할 수 있도록 도와주는 추가 연구의 필요성을 강조합니다.



### OmniDirector: General Multi-Shot Camera Cloning without Cross-Paired Data (https://arxiv.org/abs/2606.13432)
Comments:
          12 pages, 8 figures

- **What's New**: 이번 연구에서는 비디오 생성에서 카메라 모션을 클로닝하는 과제를 다루며, 기존 방법의 문제점을 해결하기 위해 카메라를 그리드 모션 비디오로 인코딩하는 새로운 표현 방식을 제안합니다. OmniDirector라는 통합 프레임워크를 통해 사용자에게 감독 수준의 제어 기능을 제공하며, 다수의 카메라 그리드-비디오 쌍으로 학습합니다. 또한, 카메라 모션과 시각적 콘텐츠를 체계적으로 설명하는 새로운 계층적 프롬프트 확장 에이전트를 설계하여 다양한 제어 신호를 조화롭게 통합합니다.

- **Technical Details**: 성공적인 비디오 생성은 Diffusion Models의 발전에 크게 의존하고 있으며, 연구는 Text-to-Video, Image-to-Video, Video-to-Video의 세 가지 주요 범주로 분류됩니다. 새로운 카메라 그리드 개념을 통해 다수의 카메라 모션을 통합적으로 처리할 수 있는 unified representation을 제시합니다. 모델은 Multi-Modal Diffusion Transformers (MMDiTs)과 함께 작동하여 프롬프트 생성, 평면 내/간의 촬영 정보 및 카메라 모션을 통합하여 생성의 일관성을 보장합니다.

- **Performance Highlights**: 광범위한 실험을 통해 OmniDirector 프레임워크의 뛰어난 성능과 우수한 컨트롤 능력이 입증되었습니다. 특히, 우리는 대규모 카메라 그리드-비디오 데이터셋으로 모델을 학습하여 다양한 카메라 동작을 처리할 수 있습니다. 이 연구는 기존의 복잡한 카메라 모션 클로닝 문제를 해결하며, 별도의 크로스 페어링 데이터 없이도 일반적인 다중 촬영 클로닝을 가능하게 합니다.



### PolyFlow: Safe and Efficient Polytope-Constrained Flow Matching with Constraint Embedding and Projection-free Upda (https://arxiv.org/abs/2606.13400)
Comments:
          30 pages, 12 figures, Accepted to ICML 2026

- **What's New**: 이번 연구는 PolyFlow라는 새로운 polytope-constrained flow matching framework를 제안하며, 기존의 flow-based generative models의 안전 문제를 모델 내에서 직접 해결합니다. PolyFlow는 이산 시간 흐름(discrete-time flow) 공식을 도입하고, 투영(projection) 없는 아키텍처를 구상하여 안전 제약을 충족시키는 동시에 계산 비용을 절감합니다. 이 방법은 계획(planning) 및 제어(control) 작업에서 높은 분포적 일관성을 유지하면서도 제약 위반이 없는 결과를 이끌어냅니다.

- **Technical Details**: PolyFlow는 연속 시간 ODE(Ordinary Differential Equations) 문제를 이산 시간 흐름 프레임워크로 재구성하여 수치적 통합(numerical integration) 오류의 위험성을 없앱니다. 또한, Frank-Wolfe 알고리즘에서 영감을 받은 투영 없는 아키텍처를 통해 매 업데이트가 항상 타당한 경계 내에 있도록 보장하며, 비싼 투영 단계 없이도 안전 제약을 만족합니다. 이는 흐름 정의와 모델 아키텍처에 제약을 직접 임베드하는 혁신적인 접근 방식입니다.

- **Performance Highlights**: 실험 결과, PolyFlow는 다양한 제약 생성 작업에서 0의 제약 위반을 달성하며, 높은 품질의 분포 일치를 유지하는 것으로 확인되었습니다. 이는 기존의 최첨단 제약 생성 방법에 비해 추론 지연(inference latency)을 획기적으로 줄이며, 안전성, 효율성, 생성 품질 간의 유리한 트레이드 오프를 보여줍니다. 이러한 결과는 PolyFlow가 물리적 시스템에서의 안전과 제약 요건을 고려한 진일보한 모델임을 입증합니다.



### Mod-Guide: An LLM-based Content Moderation Feedback System to Address Insensitive Speech toward Indigenous Ethnic and Religious Minority Communities (https://arxiv.org/abs/2606.13397)
- **What's New**: 이 논문은 방글라데시의 힌두교 및 착마(Chakma) 커뮤니티와 같은 소수자 관점을 통합하여 대형 언어 모델(LLMs)에 기반한 콘텐츠 조정 시스템의 한계를 조사합니다. 그 과정에서 비문화적 언어(Insensitive Speech)를 식별하고, 커뮤니티와 협력하여 문화적으로 적합한 말의 집합체를 만들었습니다. 이를 통해 Mod-Guide라는 툴을 개발하여 소수자 관점의 문맥적 단서를 활용하여 LLM의 민감도를 향상시키고자 했습니다.

- **Technical Details**: Mod-Guide는 검색 강화 생성(retrieval augmented generation, RAG) 기법을 활용하여 구성된 문화적 비감정적 말 무리와 소통하는 피드백 시스템입니다. 이 방법론은 LLM 기반 조정의 응답을 커뮤니티 소스의 코퍼스에 기반하여 맥락적으로 보다 정확하게 만들어 줍니다. 우리는 대다수와 소수 커뮤니티의 참여자들을 포함하는 혼합 방법 평가를 통해 RAG로 강화된 조정 응답이 문화적 맥락에서 어떻게 달리 인식되는지를 보여주었습니다.

- **Performance Highlights**: 그 결과, RAG 기반의 LLM 응답은 소수자 관점에 기초한 경우 유해한 발언을 해석하고 조정하는 방식에서 더 효과적인 것으로 나타났습니다. 또한, 이러한 조정 결과의 유용성은 민족에 따라 달라지지만 종교에 따라서는 차이가 없다는 것을 발견했습니다. 이 연구는 인간-컴퓨터 상호작용(HCI) 및 AI 윤리 분야에서 소수자 커뮤니티의 관점을 포함한 데이터 큐레이션과 시스템 설계의 중요성을 부각시킵니다.



### Who Pays the Price? Stakeholder-Centric Prompt Injection Benchmarking for Real-world Web Agents (https://arxiv.org/abs/2606.13385)
Comments:
          32 pages

- **What's New**: 이번 연구에서 우리는 웹 에이전트 시스템의 취약성을 다룬 새로운 벤치마크인 StakeBench를 도입했습니다. 기존의 공격 중심 평가 방식과는 달리, StakeBench는 이해관계자 중심으로 해를 정량화하고 분류합니다. 예를 들어, 사용자, 판매자 및 플랫폼 등 다양한 이해관계자 사이의 피해를 체계적으로 분석할 수 있도록 구성되어 있습니다.

- **Technical Details**: StakeBench는 세 가지 주요 구성 요소로 이루어집니다. 첫째, 이해관계자 중심의 분류법을 통해 해를 입는 이해관계자에 따라 공격을 조직합니다. 둘째, 12개의 구체적인 공격 목표를 설정하고 22개의 재사용 가능한 공격 템플릿을 통해 공격을 구현합니다. 셋째, 공격 성공률(Attack Success Rate, ASR), 작업 편차(Task Deviation Rate, TDR) 및 실행 안정성(Behavioral Irregularity Rate, BIR) 등의 다축 평가 지표를 통해 각 사례를 평가합니다.

- **Performance Highlights**: 우리의 평가 결과, 실제 웹 에이전트에서의 프롬프트 주입 취약성은 상당히 크고, 이해관계자에 따라 달라지며, 구조적으로 다차원적입니다. 간접적인 프롬프트 주입의 경우 ASR이 41.67%에서 68.16% 사이에 달하는 것으로 나타났으며, 이는 작업 편차와 실행 불안정성을 동반했습니다. 이러한 실패 패턴은 각 이해관계자에 따라 명확히 다르게 나타나며, StakeBench를 통해 비대칭적이고 피해자 중심적인 위험을 평가할 수 있는 기반을 제공합니다.



### SmartFont: Dynamic Condition Allocation for Few-Shot Font Generation (https://arxiv.org/abs/2606.13382)
- **What's New**: 이번 연구에서는 SmartFont이라는 새로운 프레임워크를 제안하여, 전체적인 콘텐츠 스타일 생성과 약한 감 supervis될 수 있는 로컬 보정 전문가를 결합하여 몇 번의 샷(font) 생성 문제를 해결하고자 했습니다. 이를 통해 복잡한 이데오그래프 문자의 세밀한 디테일을 보존하면서도 전반적인 구조를 유지할 수 있는 개선된 방법을 제공합니다. SmartFont는 전통적으로 사용되던 접근 방식과 달리, 다양한 조건을 효율적으로 조절하여 정교한 보정을 가능하게 합니다.

- **Technical Details**: SmartFont는 전역적인 확산 모델(global diffusion model)을 기반으로 하여 콘텐츠 및 스타일 생성을 수행하고, 약한 supervision을 받는 로컬 전문가(branch)를 추가하여 세부적인 로컬 보정 큐를 제공합니다. 로컬 전문가 브랜치는 헝가리안 매칭(Hungarian matching) 기반의 컴포넌트 supervision을 사용하여 의미 있는 공간적 맵을 예측하고, 데이터의 시멘틱(spatial) 할당을 학습합니다.

- **Performance Highlights**: SmartFont는 기존의 방법들보다 글로벌 완전성과 로컬 충실도를 동시에 개선하여glyph 품질을 향상시킬 수 있음을 실험을 통해 입증했습니다. 이는 다양한 타겟 폰트 스타일에서도 성공적으로 작동하였으며, 더 나은 세밀한 디테일을 유지하면서도 전체 구조를 유지하는 데 있어 안정성을 보여줍니다.



### An LLM System for Autonomous Variational Quantum Circuit Design (https://arxiv.org/abs/2606.13380)
Comments:
          63 pages, 19 figures, 3 tables

- **What's New**: 이번 연구는 고성능 양자 회로 설계가 여전히 인간의 전문성에 크게 의존하고 있음을 지적합니다. 우리는 명시적 설계 제약 조건 하에 양자 회로 설계를 수행하는 대규모 언어 모델(LLMs)을 활용한 자율적인 에이전트 프레임워크를 도입했습니다. 이 시스템은 탐색(Exploration), 생성(Generation), 토론(Discussion), 검증(Validation), 저장(Storage), 평가(Evaluation), 검토(Review) 등 7개의 구성 요소로 통합되어 있습니다.

- **Technical Details**: 이 구성 요소들은 웹 기반 지식 수집, 문헌 기반 비판, 실행 가능한 코드 생성, 실험적 피드백을 결합하는 폐쇄 루프 워크플로우를 형성합니다. 우리는 양자 기계 학습을 위한 양자 기능 맵(qautum feature map) 구축 및 양자 화학의 변분 양자 고유해(variational quantum eigensolver) 응용을 위해 ansatz 생성을 포함한 두 가지 작업에서 프레임워크를 평가했습니다.

- **Performance Highlights**: 이미지 분류 벤치마크에서 생성된 최상의 양자 기능 맵은 대표적인 양자 기능 맵을 초월하며, 더 큰 큐빗 수로 확장할 경우 고전적 방사 기저 함수 커널(classical radial basis function kernel)을 능가합니다. 또한, 7개의 분자에 대한 분자 기저 상태(molecular ground state) 추정에서 생성된 ansatz는 일반적으로 사용되는 화학적으로 영감을 받은 구조들과 하드웨어 효율적인 구조와 경쟁하는 정확성을 달성하며, 부과된 스케일링 제약을 충족했습니다.



### Real-Time Execution with Autoregressive Policies (https://arxiv.org/abs/2606.13355)
- **What's New**: 본 논문에서는 비동기 추론(asynchronous inference)을 통해 오토회귀 정책(autoregressive policies)이 실시간 실행(real-time execution)을 가능하게 한다는 점을 보여줍니다. 이는 기존의 확산 정책(difussion policies)에 비해 오토회귀 정책이 상대적으로 느린 롤아웃 속도(slow rollout speed)를 해결할 수 있는 가능성을 여는 것입니다. 또한, 채택한 방법론은 동작 지평선(action horizon)을 조정하고 제약된 디코딩(constrained decoding)을 적용하여 성능 극대화를 이끌어냅니다.

- **Technical Details**: 연구에서는 오토회귀 정책의 성능을 극대화하기 위해 여러 가지 방법론을 도입하였습니다. 첫 번째로, H=2m으로 설정된 충분한 동작 지평선을 선택하여 디코딩할 토큰 수를 최소화합니다. 두 번째로, 전체 동작 지평선 대신 각 mm 지평선에서 토큰화를 적용하여 실행 대기열(action queue)에 대해 동작 조각을 조건화합니다. 마지막으로, 다중 궤적 디코딩(multi-trajectory decoding)을 채택하여 성능을 최대화하는 데 기여합니다.

- **Performance Highlights**: 실험 결과, 제안된 π₀-REALFAST 정책은 동시 추론(synchronous inference)에서의 동작 조각(real-time action chunking) 기반의 정책과 비교하여 실시간 실행에서 대폭 향상된 과제 성공률(task success rates)과 롤아웃 속도를 보였습니다. 이는 원래 동기화 추론에서 관찰된 오토회귀 정책의 장점이 실시간 실행에서도 계속 유지됨을 나타내며, 오토회귀 정책이 실시간 실행을 지원할 수 있다는 강력한 동기를 제공합니다.



### IVIE: A Neuro-symbolic Approach to Incremental and Validated Generation of Interactive Fiction Worlds (https://arxiv.org/abs/2606.13348)
Comments:
          10 pages, 3 figures. To appear in the Proceedings of the 16th International Conference on Computational Creativity (ICCC'26), June 2026

- **What's New**: 본 연구에서는 IVIE (Incremental & Validated Interactive Experiences)라는 신경-기호적 접근 방식을 통해 자동으로 완전하고 플레이 가능한 인터랙티브 픽션 세계를 생성하는 방법을 제시합니다. 기존의 큰 언어 모델(LLM)들은 창의적인 서사를 생성할 수 있지만 일관성 있는 세계를 유지하는 데 어려움을 겪었습니다. IVIE는 PAYADOR의 신경-기호적 프레임워크를 기반으로 하여 창의적 결정을 LLM에 위임하고, 세계 상태를 기호적 검증을 통해 확립하는 4단계의 점진적 생성 파이프라인을 구현합니다.

- **Technical Details**: IVIE는 기호적 구성요소가 구조적 실행 및 검증 레이어로 작동하는 4단계 증가 생성 방식을 사용합니다. 각 단계에서 기호적 검증은 공간 연결성, 유형 정확성 및 목표 해결 가능성을 보장합니다. LLM은 위치 설명, 캐릭터 배경 이야기, 퍼즐 설계 및 목표 정립과 같은 이야기 요소를 생성하는 역할을 하며, 기호적 구조가 세계 상태를 유지합니다.

- **Performance Highlights**: IVIE의 초기 실험 결과는 플레어의 참여를 높이는 몰입감 있고 주제적으로 일관된 세계를 생성하는 데 성공했음을 보여줍니다. 그러나 LLM의 불일치가 때때로 퍼즐 제약을 우회하는 등의 문제도 빈번하게 발생하며, 이러한 제약은 향후 신경-기호적 인터랙티브 스토리텔링 시스템의 설계 고려 사항으로 제시됩니다.



### Dual-Domain Equivariant Generative Adversarial Network for Multimodal CT-PET Synthesis (https://arxiv.org/abs/2606.13341)
Comments:
          4 pages, 3 figures, 1 table, 2026 IEEE 23rd International Symposium on Biomedical Imaging (ISBI)

- **What's New**: 본 논문에서는 다중 모드(dual-modal) CT-PET 이미지 합성을 위한 이중 영역 동등 생성적 적대 신경망( Dual-Domain Equivariant Generative Adversarial Network, DDE-GAN)을 제시합니다. 이 모델은 전통적인 GAN 접근 방식의 한계인 기하학적 일관성을 무시하고 구조적 충실도가 제한된 문제를 해결합니다. DDE-GAN은 공간(domain)과 주파수(Fourier) 영역을 동시에 학습하여 해부학적 및 스펙트럼 정보를 포착합니다.

- **Technical Details**: 이중 영역 동등 신경망(DDE-GAN)은 CT 및 PET 측정의 물리학에 내재된 회전 동등성을 손실 함수에 통합하며, 이를 통해 회전 시 신뢰할 수 있는 반응을 보장하여 해부학적 정확성을 향상시킵니다. 계층적 이중 영역 훈련 전략은 다단계 손실 함수에 의해 영역 내 및 영역 간 일관성을 강화합니다. DDE-GAN의 훈련 과정에서는 L1 손실 함수와 같은 다양한 비용 함수가 사용됩니다.

- **Performance Highlights**: HECKTOR 2022 CT-PET 데이터셋에서 평가된 결과, DDE-GAN은 CT-PET 이미지 합성에서 기존 모델보다 우수한 합성 품질을 달성하였습니다. 이 연구를 통해 이중 영역 학습과 기하학적 동등성을 결합하는 것이 다중 모드 이미지 합성 정확성과 견고성을 크게 향상시킬 수 있음을 입증하였으며, PET 완성 및 데이터 증강의 실제 애플리케이션 가능성을 열었습니다.



### Rarity-Gated Context Conditioning for Offline Imitation Learning-Based Maritime Anomaly Detection (https://arxiv.org/abs/2606.13311)
- **What's New**: 본 연구에서는 컨텍스트(context) 변수를 조건으로 하는 이상 탐지(contextual anomaly detection)에서 드물게 발생하는 컨텍스트의 중요성을 강조하는 Rarity-Gated Feature-wise Linear Modulation(RGFiLM)이라는 방법을 제안합니다. RGFiLM은 특성 기반 조정(feature-wise modulation)과 데이터 기반의 희귀성 점수(rarity score)를 결합하여 드물고 안전-critical한 상황에서의 의사결정을 안정적으로 할 수 있도록 도와줍니다. 이를 통해 빈번한 컨텍스트 환경에서 발생할 수 있는 잘못된 경고와 불안정한 결정의 문제를 해소하고자 합니다.

- **Technical Details**: RGFiLM은 특정 상황에서의 희귀성을 평가하기 위해 경험적 컨텍스트 분포로부터 희귀성 점수를 추정하며, 이 점수를 이용해 중간 표현이 조정되는 강도를 제어합니다. 이 모듈은 일반적인 컨텍스트 대 비컨텍스트 표현 간의 보간(interpolation)을 수행하며, 빈번한 컨텍스트에서는 보수적으로, 드문 컨텍스트에서는 더욱 결정적으로 조정됩니다. 이를 통해 드물게 발생하는 경우에도 효과적으로 이상 탐지가 이루어질 수 있습니다.

- **Performance Highlights**: 대양에서의 항로 이상 탐지 실험에서, RGFiLM은 기존 방법들과 비교하여 평균 F1 점수와 잘못된 긍정 비율(FPR) 간의 관계에서 우수한 성능을 보였습니다. 이러한 결과는 컨텍스트의 희귀성을 명시적으로 고려하는 것이 컨텍스트 민감 이상 탐지에서 잘못된 경고를 줄이는 효과적인 접근 방식임을 시사합니다. RGFiLM은 다양한 종류의 이상 탐지 모델에 모듈 방식으로 결합될 수 있어 유연성과 적용 가능성을 제공합니다.



### Mining Architectural Quality Under Agentic AI Adoption: A Causal Study of Java Repositories (https://arxiv.org/abs/2606.13298)
Comments:
          16 pages. Accepted for presentation at the 52nd Euromicro Conference on Software Engineering and Advanced Applications (SEAA) 2026, Krakow, Poland, 2-4 September 2026, and for publication in the Springer LNCS proceedings. This is the author's accepted manuscript

- **What's New**: 이 논문은 AI 코딩 도구의 채택이 소프트웨어 아키텍처에 미치는 인과적 영향을 분석한 최초의 연구로, 소스 코드 단계에서의 증거와 아키텍처 단계에서의 영향을 연결하고 있다. 연구는 151개의 오픈 소스 Java 리포지토리를 대상으로 하여, AI 도구 사용의 결과로 나타나는 아키텍처 내 "smell density"를 측정하였다. "Vibe coding"이라는 새로운 패러다임 하에 에이전틱 AI 도구 사용에 대한 기대와 실제 결과 간의 불일치를 강조하고 있다.

- **Technical Details**: 연구는 74개의 에이전틱 AI 도구 채택 사례와 77개의 대조군을 포함하여 13개월 동안 1,811개의 월별 Arcan 스냅샷을 분석하였다. 이 논문은 비표준화된 밀집도(normalized density)와 시스템 규모의 영향을 고려하여 아키텍처 메트릭에 대한 Borusyak의 인과 모델을 적용하여 ASD(Architectural Smell Density)가 6.7% 감소했음을 발견하였다. 이는 아키텍처의 질적 개선이 아니라 코드 양의 변화로 인한 결과임을 보여준다.

- **Performance Highlights**: ASD가 감소했음에도 불구하고 전체적인 smell 수는 사실상 변하지 않았으며, 코드 라인은 12.8% 증가하였다. 이는 코드 볼륨의 증가가 ASD 감소의 주원인임을 시사한다. 연구는 AI 도구 채택에 따른 아키텍처 수준의 효과에 대한 최초의 인과적 증거를 제공하며, 코드와 아키텍처 간의 상관관계 차이를 확인하였다.



### HYDRA-X: Native Unified Multimodal Models with Holistic Visual Tokenizers (https://arxiv.org/abs/2606.13289)
- **What's New**: HYDRA-X는 이미지와 비디오 토크나이징을 하나의 Vision Transformer (ViT) 내에서 통합한 첫 번째 unified multimodal model (UMM)입니다. 이 모델은 시각적 데이터를 통합된 표현 공간으로 매핑하는 데 중점을 두고 있으며, spatiotemporal reconstruction 가능성을 효율적으로 처리하고 이미지 및 비디오 수준의 의미적 인식을 Latent space에 통합하는 두 가지 핵심 문제를 해결합니다. 이러한 혁신으로, HYDRA-X는 여러 작업에서 강력한 성능을 보여주며 향후 통합 토크나이저 UMM의 개발에 기여할 전망입니다.

- **Technical Details**: HYDRA-X는 영상의 프레임 간 역학을 반영하기 위해 두 가지 주요 설계를 적용했습니다. 첫째, frame-level causal temporal attention을 사용하여 비디오 재구성이 이루어지며, 이는 기존의 full spatiotemporal attention보다 우수한 성능을 보입니다. 둘째, hierarchical temporal compression을 통해 여러 단계에서 시간 압축을 수행하여 더욱 효과적인 압축을 구현했습니다. 이러한 방법을 통해 HYDRA-X는 3D-conv 비디오 VAE보다 재구성의 충실도를 높였습니다.

- **Performance Highlights**: 7B 모델에서 실행된 HYDRA-X는 이미지 및 비디오 이해 및 생성 작업에서 뛰어난 성능을 달성했습니다. 특히, 이 모델은 이미지 및 비디오 편집에서 더 일관된 결과를 제공하고 수렴 속도를 크게 향상시키는 원칙적인 개선을 제안합니다. HYDRA-X는 이 시스템의 전반적인 구조를 강화하여 향후 연구에서 통합된 토크나이저를 탐색할 수 있는 튼튼한 기반을 마련합니다.



### Cross-Modal Masked Compositional Concept Modeling for Enhancing Visio-Linguistic Compositionality (https://arxiv.org/abs/2606.13288)
Comments:
          Accepted to ACL 2026 Main Conference, 25 pages

- **What's New**: 본 논문에서는 MACCO (MAsked Compositional Concept MOdeling)라는 새로운 프레임워크를 제안하여 기존의 Vision-Language 모델(VLMs)의 구성적 이해 능력을 향상시키고 있습니다. 이 방법은 하나의 모달리티에서 구성 개념을 마스킹하고 전체 컨텍스트 정보를 기반으로 재구성함으로써 cross-modal compositional 구조를 보다 효과적으로 캡처합니다. 이러한 접근 방식은 기존의 하드 네거티브 샘플에 대한 의존성을 줄일 수 있도록 설계되었습니다.

- **Technical Details**: MACCO는 두 개의 보조 목표인 Masked-augmented Cross-Modal Alignment Loss (MCA)와 Masked-augmented Intra-Modal Regularization Loss (MIR)를 도입하여 교차 모달 재구성과 정렬 학습을 촉진합니다. MCA는 마스킹된 텍스트 또는 이미지의 글로벌 특성을 교차 모달 대조 학습 과정에 통합하는 반면, MIR은 각 모달리티 내에서 마스킹된 인스턴스의 글로벌 특성을 정규화하여 표현의 붕괴를 방지합니다. 이러한 기법들은 모델이 구문 구조 및 언어적 정보를 더 잘 캡처할 수 있게 합니다.

- **Performance Highlights**: 다양한 구성 벤치마크에서 MACCO의 효과를 검증하였으며, 실험 결과 구성적 이해가 크게 향상됨을 보여주었습니다. 또한, 개선된 구성성은 텍스트-이미지 생성 및 다중 모달 대형 언어 모델에서도 이점을 제공합니다. MACCO는 기존의 하드 네거티브 마이닝 기법과 통합될 경우 추가적인 성능 개선을 이끌어낼 수 있습니다.



### Once-for-All: Scalable Simultaneous Forecasting via Equilibrium State Estimation (https://arxiv.org/abs/2606.13285)
Comments:
          Accepted by ICML 2026

- **What's New**: 본 연구에서는 Equilibrium State Estimation (ESE)이라는 새로운 예측 방법론을 제안합니다. 이 방법론은 여러 상호작용하는 시스템이 동시에 예측될 수 있도록 하여, 각 시스템의 예측을 개별적으로 수행하는 기존의 접근 방식보다 높은 효율성을 제공합니다. ESE는 균형(equilibrium) 상태를 먼저 추정한 후, 이 균형과 현재 상태 간의 차이를 기반으로 포괄적인 예측을 생성합니다.

- **Technical Details**: ESE는 균형 동역학을 분석하여 다중 예측을 수행하는 모델로, 각 시스템의 잠재적인 균형 상태를 추정하는 Equilibrium State Estimator와 균형 추정에 따라 시스템 상태를 예측하는 Predictor 두 가지 주요 요소로 구성됩니다. 이 모델은 선형 시간 복잡도를 갖고 있으며, 시스템 수가 증가함에 따라 성능이 우수하게 확장됩니다. 또한, 기존의 예측 기법과 통합되어 더욱 효과적인 다중 예측을 가능하게 합니다.

- **Performance Highlights**: ESE는 통화 환율 모델링 및 COVID-19 전파 모델링을 포함한 다양한 데이터셋에서 실험을 수행하여, 기존의 최첨단(SOTA) 방법과 유사하거나 더 높은 정확도를 보여주며, 10-70배 빠른 속도를 자랑합니다. 이 모델은 다양한 외부 요인에 대해서도 높은 정확성을 유지하며, 빠르고 일반화 가능하며 강인한 다중 예측 기법으로 자리 잡고 있습니다.



### Different Layers, Different Manifolds: Module-Wise Weight-Space Geometry in Transformer Optimization (https://arxiv.org/abs/2606.13276)
Comments:
          Accepted at WSS @ ICML 2026, code is available at this https URL

- **What's New**: 이번 연구는 다양한 transformer 모듈이 서로 다른 manifold 기하학을 선호하는지 탐구합니다. 특히, attention 레이어는 Stiefel 기하학 제약을, MLP 레이어는 DGram 기하학 제약을 적용했을 때 최적의 성능을 보인다는 결과를 도출했습니다. 이는 transformer 최적화에 있어 모듈 별로 특정한 weight-space geometry가 필요하다는 주장을 뒷받침합니다.

- **Technical Details**: Muon 최적화 기법은 숨겨진 레이어의 weight 행렬에 행렬 정규화 업데이트를 적용하여 각 파라미터를 독립적으로 처리하지 않고 업데이트의 기하학을 수정하는 방식입니다. 두 가지 제약 조건으로 Stiefel과 DGram을 비교 분석하여 attention 및 MLP/FFN 모듈에 적용했습니다. 연구에서는 모듈별로 Stiefel과 DGram을 배분하여 실험하였습니다.

- **Performance Highlights**: 최적의 구성(Hetero)은 attention 가중치에 Stiefel 제약을, MLP/FFN 가중치에는 DGram 제약을 배치할 때의 성능이 가장 우수하였습니다. 반면 DGram을 attention에 사용한 모든 조합은 학습이 불안정하게 나타났습니다. 이러한 결과는 attention 레이어가 MLP 레이어보다 강한 스펙트럼 제어를 필요로 한다는 것을 시사합니다.



### Humor Style Drives Laughter, Topic Shapes Acceptability: Evaluating Bilingual Personal and Political Robot-Delivered AI Jokes (https://arxiv.org/abs/2606.13256)
Comments:
          Accepted in the 35th IEEE International Conference on Robot and Human Interactive Communication (RO-MAN 2026), Kitakyushu, Fukuoka, Japan

- **What's New**: 이번 연구는 로봇과의 상호작용(Interaction)에서 유머의 영향을 탐구하며, AI가 생성한 유머 유형 및 농담 내용이 사람들의 인식에 미치는 영향을 분석합니다. 특히, 사람-로봇 상호작용(HRI)에서 유머를 통합할 수 있는 새로운 기회를 모색하고 있습니다. 이를 통해 유머 스타일이 어떻게 사람들이 로봇의 유머를 받아들이는지를 밝혀내고자 하였습니다.

- **Technical Details**: 이 연구는 혼합 요인 설계(mixed factorial design)를 사용하여 진행되었으며, 참가자들은 대학 강의실에서 로봇이 전달하는 AI 생성 유머를 평가했습니다. 연구는 유머 유형(Affiliative, Self-Enhancing, Aggressive, Self-Defeating)과 농담 내용(인물 관련 vs. 정치적)이 재미와 적절성에 미치는 영향을 조사합니다. 참가자들의 자가 보고된 유창성(fluidity) 및 유머 관행 또한 언어 선호에 영향을 미칠 수 있습니다.

- **Performance Highlights**: 연구 결과에 따르면 유머 유형은 재미에 상당한 영향을 미치며, Aggressive와 Affiliative 유머가 더 높은 평가를 받았습니다. 반면, 농담 내용은 적절성에 주로 영향을 미치며, 인물 관련 농담이 정치적 농담보다 선호되는 것으로 나타났습니다. 이러한 결과는 사람-로봇 상호작용에서 유머의 역할을 더 잘 이해하는 데 기여할 수 있습니다.



### Towards Personalized Federated Learning for Dysarthric Speech Recognition (https://arxiv.org/abs/2606.13253)
- **What's New**: 이 논문은 경직성 발음을 가진 화자를 위한 개인 맞춤형 연합 학습(federated learning, FL) 기반 자동 음성 인식(ASR) 시스템의 새로운 접근 방식을 제안합니다. 특히, 배경 지식을 활용한 두 가지 집계 전략, 즉 파라미터 기반 평균화(parameter-based averaging)와 임베딩 기반 평균화(embedding-based averaging)를 탐구합니다. 이 연구는 언어 모델의 개별화와 개인 정보 보호를 위한 혁신적인 방법을 제시하며 dysarthric speech의 ASR 발전을 도모합니다.

- **Technical Details**: 연합 학습은 데이터를 로컬 장치에 유지하면서 프라이버시 요구 사항을 준수하여 동작합니다. 이 논문에서는 표준 FedAvg 알고리즘을 통해 다수의 참가자로부터의 데이터 샘플 수에 따라 가중치를 조정하여 평균화를 수행합니다. 개별 발화자에 대한 이해를 높이기 위해 두 가지 다른 집계 전략을 사용하여 모델을 두 가지 부분, 즉 화자 독립(SI) 및 화자 의존(SD) 부분으로 나누어 처리합니다.

- **Performance Highlights**: UASpeech 및 TORGO 데이터셋에 대한 실험을 통해, 제안된 방법이 기준 모델인 정규화된 FedAvg보다 오류율(Word Error Rate, WER)을 각각 0.99% 및 0.56% 절대적으로 줄이는데 성공하였음을 보였습니다. 이는 개인 맞춤형 접근 방식이 dysarthric speech 인식에서 유망한 결과를 나타내며, 고유한 발화자의 발음 변화를 반영할 수 있도록 설계되었음을 의미합니다.



### Towards More General Control of Diffusion Models Using Jeffrey Guidanc (https://arxiv.org/abs/2606.13240)
- **What's New**: 이 논문은 Jeffrey guidance라는 새로운 프레임워크를 제안하여 diffusion 모델의 출력을 보다 유연하게 제어할 수 있는 방법을 제공합니다. 기존의 조건부 샘플링을 넘어서, 제안된 방법은 명시적인 목표 분포를 설정하고 이를 유지하면서도 marginal distributions를 업데이트할 수 있습니다. 이 접근법은 전통적인 guidance 방법에 비해 더욱 복잡한 요구사항을 충족할 수 있는 가능성을 열어줍니다.

- **Technical Details**: Jeffrey의 조건화 규칙은 모델의 확률 분포를 업데이트하는 데 유용한 일반적인 방법으로, 다양한 조건을 만족할 수 있습니다. 새로운 Jeffrey guidance는 기존의 조건부 기반 접근 방식과 차별화되어, 주어진 목표 분포에 맞춰 모델 출력을 조정합니다. 이 과정에서, 표준 분류기 기반의 guidance를 포함하여 다양한 conditional 구조를 보존하면서 joint distribution에 최소한의 변화를 주는 방식으로 동작합니다.

- **Performance Highlights**: Jeffrey guidance를 적용한 결과, Inception embeddings를 목표로 삼았을 때, CIFAR-10과 FFHQ 데이터셋에서 FID를 상당히 감소시켰습니다. CelebA-HQ에 대한 공정성 문제에서도, output에서 Male 및 Young 속성이 decorrelation되도록 하는데 성공하여, 보다 공정한 모델을 개발할 수 있는 가능성을 보여주었습니다.



### ComAct: Reframing Professional Software Manipulation via COM-as-Action Paradigm (https://arxiv.org/abs/2606.13239)
- **What's New**: 이 연구는 기존의 컴퓨터-사용 에이전트들이 가진 한계를 극복하기 위해 새로운 패러다임인 COM-as-Action(ComAct)을 제안합니다. COM(구성 객체 모델)을 활용하여 전문 소프트웨어 조작을 실행 가능한 프로그램 합성으로 재구성하였으며, 이를 통해 GUI 인터랙션의 취약성을 해소하고 더 높은 일관성을 제공합니다. 또한, 실질적인 CAD 소프트웨어에서 운영되는 에이전트를 평가하기 위한 최초의 벤치마크인 ComCADBench를 소개합니다.

- **Technical Details**: COM은 Microsoft가 도입한 이진 인터페이스 표준으로, 다양한 응용 프로그램 간의 프로그래밍 가능한 통신을 가능하게 합니다. COM은 구조화된 프로그래밍 인터페이스를 통해 소프트웨어 내부를 노출하고, Python과 같은 언어에서 접근할 수 있는 호출 가능한 객체의 계층 구조를 제공합니다. 이 연구는 전문 소프트웨어 조작을 부분 관찰 가능 마르코프 결정 과정으로 모델링하며, 각 행동이 COM 인터페이스를 호출하는 실행 가능한 Python 스크립트로 구성된다는 점이 중요합니다.

- **Performance Highlights**: ComActor는 ComCADBench에서 최첨단 성능을 달성하며, 기존의 GUI 기반 에이전트들이 실패하는 긴 시간의 작업에서 강력한 회복력을 보입니다. 제공된 실험 결과는 ComActor가 외부 CAD 벤치마크에서도 뛰어난 일반화를 나타내며, CAD 소프트웨어 조작에 있어서 새로운 표준을 세우기 위한 잠재력을 보여줍니다. ComForge는 대규모 교육을 위한 확장 가능한 플랫폼을 제공하여, ComActor의 훈련 과정을 지원합니다.



### Decoding Insect Song: A Multitask Semisupervised Orthoptera Bioacoustic Classifier (https://arxiv.org/abs/2606.13236)
Comments:
          ICML 2026 Workshop on Machine Learning for Audio

- **What's New**: 본 연구에선 PULSE라는 새로운 반지도학습(semisupervised learning) 및 다중 작업(multi-task) 프레임워크를 제안합니다. 기존의 자동화 도구들은 한정된 데이터에만 훈련되고 이전이 불가능한 한계가 있어, 이를 극복하기 위해 PULSE는 Orthoptera의 생물음향(bioacoustics) 분석을 위한 최적화된 모델입니다. PULSE는 약한(supervised) 종 분류와 자기지도 학습(self-supervised learning), 일반 생물음향 모델로부터의 지식 증류(knowledge distillation)를 결합하여 생태학적 발견을 위한 인터랙티브 시각화 툴을 제공합니다.

- **Technical Details**: PULSE는 공개된 Orthoptera 데이터와 약 150GB의 영국 필드 녹음을 포함한 대규모 비표시 데이터(unlabelled data)를 활용하여 훈련됩니다. 모델은 VGGish 백본을 사용하여 오디오를 메모리(mel) 스펙트로그램으로 변환한 후, 세 가지 최적화 목표 - 지도 학습(supervised classification), 생태학적 우선 기준(ecological prior), 자기 지도 학습(self-supervision) -을 거치며 학습됩니다. 이로 인해 PULSE는 라벨이 있는 데이터와 라벨이 없는 데이터 간의 불균형을 해소할 수 있습니다.

- **Performance Highlights**: PULSE의 성능은 다양한 메트릭에서 최신상태(state-of-the-art) 모델과 비교하여 우수합니다. F1 점수는 0.34, AUC는 0.84로 향상되어, 활성 학습(active learning) 기법이 효과적으로 적용된 것으로 나타났습니다. 이 연구는 PULSE가 새로운 환경에서도 우수한 모델 이식성(model transferability)을 보이면서, Orthoptera의 생물 음향 모니터링 생태학적 연구에 큰 기여를 할 수 있음을 보여줍니다.



### ReSET: Accurate Latency-Critical NVFP4 Reasoning via Step-Aware Temperature Scaling (https://arxiv.org/abs/2606.13233)
- **What's New**: 본 연구는 NVFP4 추론을 대형 추론 모델(LRM)에 적용하는데 있어 발생하는 두 가지 실제적인 제한 사항을 다룹니다. 첫째, 양자화(quantization) 하에서는 추론 정확도가 저하되며, 둘째, 기존 NVFP4 커널이 소규모 자가 회귀 디코딩에서 지연(latency) 이점을 완전히 실현하지 못합니다. 우리는 이와 관련하여 ReSET라는 새로운 방법론을 제시하여 추론 과정에서의 불확실성을 관리합니다.

- **Technical Details**: ReSET는 단계별 엔트로피 기반 온도 스케일링 방법으로, 각 단계의 불확실성을 실시간으로 추정하며, 토큰 수준과 단계 수준의 엔트로피 신호를 사용하여 디코딩 온도를 조절합니다. 또한, CUDA-core 기반의 소규모 NVFP4 커널을 설계하여 지연이 중요한 자가 회귀 디코딩을 최적화합니다. NVFP4 형식은 NVIDIA의 마이크로 스케일 FP4 포맷으로, 이전의 BF16보다 최대 4배 더 높은 처리량을 제공합니다.

- **Performance Highlights**: ReSET 방법은 다양한 추론 벤치마크와 모델 스케일에서 NVFP4의 추론 정확도를 최대 2점까지 향상시킵니다. 새로 설계된 CUDA-core 소규모 NVFP4 커널은 NVFP4 vLLM에 비해 최대 2.5배 빠른 속도를 달성하며, BF16에 비해 약 2배의 엔드 투 엔드 디코딩 속도 향상을 보여줍니다.



### Proprioceptive-visual correspondence enables self-other distinction in humanoid robots (https://arxiv.org/abs/2606.13222)
Comments:
          23 pages, 9 figures, 1 supplementary table

- **What's New**: 이번 연구에서는 휴머노이드 로봇이 정체성 레이블이나 운동학 모델 없이도 프로프리오셉션-비주얼 상관관계(proprioceptive-visual correspondence)를 통해 자기와 타자를 구별할 수 있다는 점을 보여줍니다. 이러한 자기-타자 구분(self-other distinction)은 로봇이 조정된 자기 모델을 학습하는 기초가 되며, 이를 통해 로봇이 다가오는 목표를 향한 목표 도달(target reaching) 및 충돌 인식 모션 계획(collision-aware motion planning)와 같은 다운스트림 작업을 지원할 수 있습니다. 연구의 결과는 로봇이 공유 환경에서 타인과 조화를 이루며 행동할 수 있는 신체적 자기 표현(body representation)의 가능성을 제시합니다.

- **Technical Details**: 본 연구는 29 자유도(DoF)의 휴머노이드 로봇을 이용하여 구현되었으며, 로봇은 프로프리오셉션 상태와 시각적 관찰만으로 자기 신체를 식별할 수 있습니다. 각 후보 마스크는 시간적으로 일치하는 프로프리오셉션 상태와의 관계를 기반으로 하여 네트워크 모델에 유입되는 신호를 생성합니다. 특정 후보 마스크가 로봇의 신체에 해당한다고 가정하는데, 이 과정은 단순한 강화 학습 없이도 이루어집니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 로봇이 타인과 함께 있는 복잡한 환경에서도 자기 신체를 효과적으로 식별하는 능력을 보여주었습니다. 로봇은 실험에서 99.5% 이상의 정확도로 자신의 신체를 식별하며, 이는 일반적인 시각-언어 모델(Vision-Language Model)이나 단순 평균 모델보다 월등히 높은 성능입니다. 제안된 접근 방식은 현재 VLM이 특정 조인트 각도를 가시적인 신체와 연결하는 데 한계가 있음을 강조하며, 로봇이 자신의 신체를 경험을 통해 학습할 수 있는 새로운 가능성을 제공합니다.



### Transformer-Guided Graph Attention for Direct Cardiac Mesh Reconstruction: A Structural Digital Twin Framework (https://arxiv.org/abs/2606.13188)
- **What's New**: 이 연구는 심장 모형 생성의 효율성을 크게 향상시키기 위해 세분화(segmentation)와 메시(mesh) 생성 프로세스를 통합한 새로운 접근 방식을 제시합니다. 기존의 복잡하고 수동적인 작업 흐름에서 벗어나, 3D 의료 이미지를 직접적으로 매끄러운 심장 표면 메시로 변환하는 엔드-투-엔드 네트워크를 훈련시킵니다. 사용자는 Marching Cubes와 같은 전통적인 방법에 의존하지 않고도 빠르고 정확한 모델링이 가능해집니다.

- **Technical Details**: 이 접근법의 핵심은 3D Swin Transformer와 Graph Attention Network의 조합을 통해 원시 3D 이미지를 처리하고, 환자의 심장 경계에 맞추어 템플릿 메시를 반복적으로 변형하는 것입니다. 실험은 MM-WHS 2017 벤치마크에서 CT 및 MRI 이미지를 사용하여 수행되었습니다. 결과적으로 평균 Chamfer 거리 1.8 mm 및 95번째 백분위수 표면 거리 5 mm 이하의 품질 지표를 기록하여 기존의 세분화-메시 프로세스보다 우수한 성능을 보였습니다.

- **Performance Highlights**: 제안한 방법은 모든 메시가 단일 포워드 패스를 통해 생성되어, 수동적인 후처리 작업이 거의 필요 없다는 점에서 임상 사용을 위한 접근성을 높입니다. 또한, 세분화 마스크가 아닌 실제 시뮬레이션에 필요한 정밀한 메시를 제공함으로써, 임상 개입에 대한 예측적 통찰력을 제공합니다. 이로 인해 심장 디지털 트윈 파이프라인에서 지리적 충실도와 위상적 정확성이 픽셀 단위의 정확도보다 더 중요하다는 사실을 강조합니다.



### Modern analog computing for solving differential and matrix equations (https://arxiv.org/abs/2606.13179)
- **What's New**: 최근 몇 년간 인공지능(Artificial Intelligence) 및 과학적 컴퓨팅 등 데이터 집약적 애플리케이션의 요구에 따라 아날로그 컴퓨팅(Analog Computing)에 대한 관심이 재조명되고 있습니다. 이 논문에서는 아날로그 CMOS 회로와 저항 메모리 기술의 발전을 통해 아날로그 컴퓨팅의 새로운 지형을 설정하고, 세 가지 핵심 계산 원시(primitives)로서 미분 방정식을 풀고, 행렬 방정식을 해결하며, 행렬-벡터 곱셈을 수행하는 것을 제시합니다.

- **Technical Details**: 우선, 저자는 미분 방정식, 행렬 방정식, 행렬-벡터 곱셈 간의 연관성을 탐구합니다. 그리고 이러한 아날로그 컴퓨팅 연산자의 다양한 하드웨어 구현을 검토하는데, 여기에는 이산 구성 요소, 집적 회로(Integrated Circuits), 저항 메모리 장치가 포함됩니다. 저항 메모리 어레이는 이러한 구현의 효율성 덕분에 특히 유망하게 부각됩니다.

- **Performance Highlights**: 논문은 현대 아날로그 컴퓨팅을 활용하여 미분 및 행렬 방정식을 해결하는 최근의 발전을 조사합니다. 특히 고급 아날로그 CMOS 회로와 저항 메모리 어레이를 이용한 솔루션을 강조합니다. 또한 이 회로들의 응용, 정확도 및 확장성 문제, 메모리 내 컴퓨팅(In-Memory Computing)과의 관계, 아날로그 컴퓨팅의 독특한 계산 복잡성에 대해 논의하며, 차세대 계산의 경계를 여는 중요한 요소로서 아날로그 컴퓨팅을 포지셔닝합니다.



### MemRefine: LLM-Guided Compression for Long-Term Agent Memory (https://arxiv.org/abs/2606.13177)
- **What's New**: 이 논문에서는 장기 상호작용(long-term interactions)에서 사용되는 대규모 언어 모델(LLM) 에이전트의 메모리 관리 문제를 다룹니다. 기존 메모리 저장소가 시간에 따라 증가함에 따라 중복 항목이 쌓이고, 이는 저장 비용을 증가시키고 유용한 정보를 검색하는 데 방해가 됩니다. 이러한 문제를 해결하기 위해, 저장 예산(storage-budgeted) 내에서 정보의 유용성을 보존하는 메모리 관리 방법을 제안합니다.

- **Technical Details**: 논문에서는 MemRefine이라는 LLM 기반 프레임워크를 제안합니다. 이 프레임워크는 정보의 사실적 가치(factual value)를 유지하기 위해 유사성을 기반으로 후보 쌍(candidate pairs)을 제안하며, 삭제(delete), 병합(merge), 및 보존(preserve) 결정을 LLM 판별자(LLM judge)에 위임합니다. 이 방법은 예산이 충족될 때까지 반복(iteration)하여 메모리 관리를 최적화합니다.

- **Performance Highlights**: 다양한 메모리 프레임워크와 장기 대화 벤치마크에서 MemRefine은 목표 예산을 지속적으로 충족하며 하위 성능(downstream performance)을 유지했습니다. 또한, 엄격한 예산 아래에서 기존의 규칙 기반(rule-based) 기준을 능가하는 성과를 보였습니다.



### NTS-CoT: Mitigating Hallucinations in LLM-based News Timeline Summarization with Chain-of-Thought Reasoning (https://arxiv.org/abs/2606.13171)
- **What's New**: 이 논문에서는 LLM 기반의 Timeline Summarization (TLS)의 주요 문제인 hallucinations(환각) 현상에 대한 해결책을 제시합니다. 기존 연구에서는 hallucinations의 유형에 대한 체계적인 분석이 부족했던 반면, 두 가지 주요 유형인 불신 내용 불일치와 정보 누락을 식별했습니다. 이를 해결하기 위해 새로운 프레임워크인 NTS-CoT를 도입하여 Chain-of-Thought(사고의 연쇄) 추론 기법을 활용합니다.

- **Technical Details**: NTS-CoT는 크게 세 가지 모듈로 구성됩니다: i) Element-CoT는 중요한 뉴스 요소를 포착하여 신뢰성 있는 요약을 생성하고, ii) Date Selection은 시간의 중요성과 사건의 두드러짐을 결합하여 타임스탬프 선택을 최적화하며, iii) Causal-CoT는 여러 문서 간 인과 관계를 추론하여 날짜-사건 요약에서 정보 누락을 줄입니다. 이러한 구조는 LLM이 생성하는 요약의 정확성과 일관성을 높입니다.

- **Performance Highlights**: NTS-CoT는 세 가지 TLS 기준에서 정량적 분석과 인간 평가를 통해 성능이 입증되었습니다. NTS-CoT는 SOTA(SOTA: State-Of-The-Art) 기준선에 비해 AR-1이 23.4%, AR-2가 33.4%, Date-F1이 10.0% 개선되었습니다. 또한, 인간 평가에서도 NTS-CoT의 요약이 신뢰성을 기준으로 67.74%, 완전성을 기준으로 54.38%의 비율로 우수성을 인정받았습니다.



### Iterative Visual Thinking: Teaching Vision-Language Models Spatial Self-Correction through Visual Feedback (https://arxiv.org/abs/2606.13156)
- **What's New**: 이번 연구에서는 Vision-language 모델(VLM)이 예측을 반복적으로 관찰하고 수정하는 메커니즘이 결여되어 있음을 발견했습니다. Iterative Visual Thinking(IVT)이라는 새로운 피드백 기반 프레임워크를 제안하여, 모델이 예측한 경계 상자를 렌더링하고 이를 바탕으로 결과를 정제하도록 합니다. 이 과정에서 VLM의 강력한 성능이 스스로 수정 가능성을 갖추지 못해, 단순한 반복적 접근만으로 막대한 성능 저하가 발생한다는 것을 보여줍니다.

- **Technical Details**: IVT는 모델이 예측한 내용을 시각적으로 확인하고 이를 바탕으로 반복적으로 수정하는 닫힌 루프(closed-loop) 프레임워크입니다. 연구진은 두 단계의 훈련 메커니즘인 SFT(Supervised Fine-Tuning)와 GRPO(Group Relative Policy Optimization)를 제안하였습니다. SFT는 교수 모델을 통해 초기 예측에서부터 교정된 추론 경로를 생성하여 학습 데이터를 생성하고, GRPO는 단순한 보상을 통해 다단계 수정 과정을 안정화시킵니다. 이 방법은 총 2,400개의 샘플로 단일 GPU에서 학습을 통해 이루어졌습니다.

- **Performance Highlights**: IVT 적용 후 모델의 성능이 모든 지표에서 단일샷(singe-shot) 기본 모델을 초월했습니다. Acc@0.5는 82.0%까지 상승하였고, Acc@0.7과 Acc@0.9도 각각 74.1%와 48.3%로 증가했습니다. GRPO는 각 단계에서의 IoU 감소를 5배 줄이며, 모델이 단계 사이에 예측을 유지하도록 훈련받았습니다. 이 연구는 VLM이 스스로 공간적 예측을 수정하는 것이 가능하다는 것을 보여주는 중요한 기초 연구입니다.



### Cascade Classification of Dermoscopic Images of Skin Neoplasms with Controllable Sensitivity and External Clinical Validation (https://arxiv.org/abs/2606.13135)
Comments:
          28 pages, 8 figures, 10 tables

- **What's New**: 이 연구는 피부 신생물에 대한 피부경 검사 이미지의 심층 학습 아키텍처와 분류 스킴을 비교하고, 개방형 국제 데이터 세트에서 러시아의 임상 데이터 세트로의 일반화 성능을 평가하는 것을 목적으로 한다. 네 가지 아키텍처(ViT-B/16, Swin-S, ConvNeXt-S, EfficientNetV2-S)와 세 가지 분류 스킴(이진, 단일 단계 네 클래스, 그리고 두 단계 캐스케이드)을 비교했다. 특히, 미세한 malignant lesions의 정확한 분류를 위해 조정 가능한 triage threshold를 도입하여 표준 단일 단계 분류에서 불가능한 민감도 조절을 가능하게 했다.

- **Technical Details**: 본 연구는 피부 경향성 신생물에 대한 이미지 분석을 수행하는 데 있어, 네 가지 현대적 딥러닝 아키텍처를 비교하고 있다. 이 연구의 한 가지 주요 포인트는 기존 단일 단계 멀티클래스 분류가 민감도 손실을 초래하는 경향이 있다는 것이다. 두 단계 캐스케이드 스킴은 진단의 임상 논리를 재현하며, 각 단계에서 다른 목표 함수를 통해 작업의 계층적 나누기를 구현하여 민감도를 조절할 수 있다. 모델들은 ImageNet으로 사전 학습된 가중치와 단일 증강 프로토콜을 사용하여 교육되었으며, 러시아의 임상 데이터를 포함한 두 개의 데이터 세트에서 평가되었다.

- **Performance Highlights**: 이 연구에서 이진 단계의 ROC-AUC 점수는 0.952에서 0.966 사이였으나, Sechenov 대학에서 0.797에서 0.893으로 떨어지고 민감도는 0.53에서 0.67로 감소하는 결과를 보였다. 캐스케이드 방식은 대부분의 아키텍처에서 단일 단계 네 클래스 분류보다 macro F1 점수를 개선시켰으나, 특히 ViT-B/16에서만 그 차이가 통계적으로 유의미하게 나타났다. 또한, ISIC MILK10k 데이터에서의 직접적인 11클래스 분류에서는 평균 클래스 민감도가 0.525로 나타났다.



### MiniPIC: Flexible Position-Independent Caching in <100LOC (https://arxiv.org/abs/2606.13126)
Comments:
          13 pages, 5 figures

- **What's New**: 이번 논문에서는 Minimalistic PIC (MiniPIC)를 제안하고 있습니다. MiniPIC은 vLLM 디자인을 기반으로 하며, 사용자 제어의 캐시 재사용 기법과 함께 위치 인코딩이 없는 KV 캐시를 포함하여 구성됩니다. 또한, 인퍼런스 서버 내에서 포지션 변환 없이 캐시된 K 벡터를 저장하는 방법을 사용합니다.

- **Technical Details**: MiniPIC은 세 가지 사용자 친화적 원시 기능인 블록 정렬 패딩(block-aligned padding), 스팬 구분자(span separator, SSep), 그리고 프롬프트 의존성(prompt depend, PDep)을 제공합니다. 이러한 기능들은 해싱 동작과 효과적인 블록 레벨 인과(attention structure)를 수정하는 데 사용됩니다. MiniPIC은 100줄 이하의 코드 변경만으로 다양한 PIC 방법론(Block-Attention, EPIC, Prompt Cache)을 구현할 수 있습니다.

- **Performance Highlights**: 2WikiMultihopQA에서 MiniPIC은 PF(prefill) 처리량을 49% 개선하였고, 캐시된 스팬의 첫 번째 토큰 시간(`time-to-first-token`)을 두 배에서 최대 100배 단축시켰습니다. 또한 캐시되지 않은 스팬의 선형 PF 스케일링을 유지하며, 최악의 경우 오버헤드는 단 5.7%에 불과합니다.



### Select and Improve: Understanding the Mechanics of Post-Training for Reasoning (https://arxiv.org/abs/2606.13125)
- **What's New**: 이 논문은 강화 학습(Reinforcement Learning, RL)이 수학적 추론 및 코딩 모델의 훈련에서 어떻게 중요한 요소로 작용하는지를 탐구합니다. 특히, RL이 모델의 능력을 어떻게 향상시키는지를 메커니즘 관점에서 이해하기 위해 진행된 연구입니다. 연구 결과는 두 가지 핵심 메커니즘인 전략 선택(strategy selection)과 전략 개선(strategy improvement)을 밝히고, SFT 데이터와 RL 데이터가 이러한 메커니즘을 활성화하는 데 미치는 영향을 보여줍니다.

- **Technical Details**: 연구에서는 수학적 추론 실험을 통해 RL의 후속 훈련에서의 메커니즘을 분석합니다. 사용된 모델은 Qwen-2.5-1.5B이며, 실험 결과는 모델의 성능을 향상시키는 데 있어 전략 선택과 전략 개선의 역할이 중요하다는 것을 보여줍니다. 특히, 전략 선택은 사전 훈련(pre-training) 데이터에서 추론 패턴을 선택하도록 하고, 전략 개선은 RL 데이터에서 본 문제보다 더 어려운 질문을 제공해야 활성화됩니다.

- **Performance Highlights**: 결과적으로, 연구는 RL 후 훈련이 사전 훈련에서 얻은 추론 패턴을 개선하는 방식으로 작동한다는 점을 강조합니다. 이러한 개선은 성능을 크게 향상시킬 수 있지만, 전혀 새로운 행동을 유도하지는 않습니다. 따라서, 성과를 확장하는 중요한 요소는 고품질의 사전 훈련 데이터임을 시사하며, 향후 RL 후 훈련의 발전을 위한 기초적인 메커니즘 제공으로 이어질 것으로 기대됩니다.



### NaturalFlow: Reducing Disruptive Pauses for Natural Speech Flow in Simultaneous Speech-to-Speech Translation (https://arxiv.org/abs/2606.13121)
Comments:
          Proceedings of the 26th Interspeech Conference, Long Paper

- **What's New**: 이번 논문에서는 유창성을 고려한 최적화 프레임워크를 도입하여 동시 통역의 고유한 문제를 해결합니다. 이를 통해 낮은 지연 시간에서도 자연스러운 발음을 유지할 수 있는 최적 균형을 발견합니다. 기존의 S2ST(Speech-to-Speech Translation) 모델들이 단편적인 언어 출력을 생성하는 문제를 해결하고자 하며, 보다 매끄러운 발화를 가능하게 합니다.

- **Technical Details**: 제안된 NaturalFlow 모델은 'Direct Preference Optimization (DPO)'을 통해 훈련되며, 이는 정지 비율을 최소화하면서도 번역의 충실도를 유지하는 두 개의 상충되는 목표를 최적화합니다. 이 모델은 'Large Language Models (LLMs)'의 생성을 활용하여 자연스러운 발화를 위한 다양한 패러프레이즈를 생성할 수 있습니다. 여러 벤치마크 테스트를 통해 이 프레임워크는 침묵 비율을 줄이고, 번역 품질과 대기 시간 관련 메트릭을 유지하면서 성능을 검증하였습니다.

- **Performance Highlights**: 인간 평가 결과, 제안된 S2ST 모델이 기존 시스템에 비해 선호도가 높았으며, 연속적이고 자연스러운 음성 출력을 생성함을 보여주었습니다. 다양한 도메인과 발화 길이를 아우르는 네 가지 벤치마크에서 성능이 입증되었습니다. 이러한 결과는 정보 내용이 유지되면서도 논리적 흐름이 매끄럽고 자연스럽다는 것을 나타냅니다.



### MP3: Multi-Period Pattern Pre-training forSpatio-Temporal Forecasting (https://arxiv.org/abs/2606.13119)
- **What's New**: 본 연구는 새로운 Multi-Period Pattern Pre-training (MP3) 기법을 개발하여 도시 공간-시간 데이터에서 발생하는 일시적 기만(temporal mirage)을 해결하고자 합니다. 기존의 공간-시간 그래프 신경망(STGNN)은 이와 같은 현상을 효과적으로 식별하지 못하는 한계를 가지고 있습니다. MP3는 다중 기간 패턴 학습(multi-period pattern learning) 및 효율적인 causal correlation을 통해 이러한 문제를 해결하는 혁신적인 접근 방식을 제안합니다.

- **Technical Details**: MP3는 두 가지 주요 혁신을 포함합니다: (1) Multi-period temporal modeling을 통해 긴 시계열에서 다양한 다중 기간 패턴을 학습하고, edge convolution을 이용하여 서로 다른 다중 기간 패턴을 식별합니다. (2) 이 플러그인은 기존 STGNN 백본에 무리 없이 통합되어 예측 성능을 강화합니다. 이러한 다중 기간 접근법은 다양한 공간 상관관계를 효율적으로 포착하도록 설계되었습니다.

- **Performance Highlights**: 다섯 개의 STGNN 베이스라인을 사용하여 진행된 실험에서는 MP3가 MAE를 평균 4.7%, RMSE를 5.0% 줄이는 성능 향상을 보여주었습니다. 이 연구는 실제 데이터셋에서의 지속적이고 안정적인 성능 개선을 통해 MP3의 유효성을 입증합니다. 또한, 대규모 데이터셋 CA에서 얻은 두드러진 성과는 MP3의 강력한 확장성을 강조합니다.



### G-Long: Graph-Enhanced Memory Management for Efficient Long-Term Dialogue Agents (https://arxiv.org/abs/2606.13115)
Comments:
          22 pages, 8 figures, 14 tables

- **What's New**: 본 논문에서는 기존의 대화 시스템에서 나타나는 긴 대화의 일관성을 유지하는 데 필요한 장기 기억을 위한 새로운 프레임워크인 G-Long을 제안합니다. G-Long은 구조화된 트리플렛(triplet) 추출 및 연관 검색(associative retrieval)을 위해 미세 조정된 소형 언어 모델(sLM)을 활용하여 작동 비용을 크게 줄입니다. 또한, T5 요약기의 내부 교차 주의 신호를 활용해 중요한 기억을 식별하는 새로운 주의 기반 중요도 점수 매기기 메커니즘을 도입하여 성능을 향상시킵니다.

- **Technical Details**: G-Long은 네 가지 주요 구성 요소로 이루어져 있습니다: (1) 효율적 메모리 구축, (2) 그래프 기반 메모리 뱅크, (3) 연관 메모리 검색, (4) 응답 생성입니다. 대화를 구조화된 그래프 표현으로 변환하기 위해 트리플렛 추출 모듈을 사용하여 각 발화를 분석하고, 주의 기반 중요도 점수 매기기 모듈을 통해 추출된 트리플렛에 중요도 점수를 부여합니다. 이 과정에서 조정된 소형 언어 모델을 활용하여 원시 발화를 구조화된 사실로 변환합니다.

- **Performance Highlights**: G-Long은 MSC, CC, LoCoMo, LME와 같은 다양한 벤치마크에서 실험을 통해 응답 생성 및 메모리 검색 모두에서 최첨단 성능을 달성했습니다. 응답 품질에서 MSC 데이터셋에 대해 9.8% 향상된 결과를 보였으며, LME 벤치마크에서는 검색 성능이 40.8% 개선된 것으로 나타났습니다. 이러한 성과는 기존 시스템의 계산 비용을 크게 줄이면서도 응답 품질을 유지하는 데 기여합니다.



### Functional Cache Grafting: Robust and Rapid Code-Policy Synthesis for Embodied Agents (https://arxiv.org/abs/2606.13097)
Comments:
          Accepted at ICML 2026

- **What's New**: FCGraft는 Embodied Agent를 위한 Functional Cache Grafting 프레임워크로, 자연어 목표와 환경 제약을 구조화된 제어 프로그램으로 변환하는 과정에서 발생하는 두 가지 주요 제한사항을 해결합니다. 이 프레임워크는 기능 수준의 검증된 코드 스켈레톤을 유지하며, 새로운 작업이 주어질 때 관련 기능을 검색하고 KV 캐시를 결합하여 새로운 정책을 합성합니다. 전통적인 Code-as-Policies(CaP)에 비해, FCGraft는 코드 생성의 효율성과 신뢰성을 개선하여 조작 에이전트의 성능을 더욱 향상시킵니다.

- **Technical Details**: FCGraft는 함수 수준의 KV 캐싱을 통해 코드 재사용을 가능하게 하여 불필요한 사전풀(_prefill) 계산을 줄입니다. 이 프레임워크는 캐시 스티칭(cache stitching)과 캐시 패칭(cache patching)이라는 두 가지 상호 의존적인 메커니즘을 사용하여 오류를 보정하고 성능을 향상시킵니다. 캐시 스티칭은 함수 호출을 KV 상태를 통해 직접 구성하여 재사용성을 높이고, 캐시 패칭은 오류 범위를 국소화하여 필요한 부분만 수정하는 것으로 디코딩 시간을 크게 단축합니다.

- **Performance Highlights**: 실험 결과, FCGraft는 다양한 개방형 도메인 시나리오에서 18.31% 더 높은 작업 성공률과 평균 2.3배 빠른 정책 합성을 달성하여 RAGCache보다 우수한 성능을 보였습니다. FCGraft는 로봇 조작 작업과 관련된 성능 평가에서 빠르고 일관된 정책 합성을 제공하여 실용적인 장점이 있음을 입증하였습니다. 이러한 성과는 FCGraft가 복잡한 환경에서도 신뢰할 수 있는 제어 기능을 제공할 수 있음을 나타냅니다.



### Emotional regulation improves deep learning-based image classification (https://arxiv.org/abs/2606.13081)
- **What's New**: 이 논문은 Emotion Regulation(감정 조절)이라는 새로운 프레임워크를 소개하여 인공지능과 딥러닝에서 감정의 주관적 경험을 모델링하려고 합니다. 기존의 방법들이 감정을 객관적인 신경생리학적 요소에만 의존했던 반면, 본 연구는 주관성을 고려해 감정의 역할을 강조합니다. 이 접근 방식은 ResNet와 ViT 아키텍처를 활용하여 이미지 분류 작업의 최적화를 통해 유의미한 결과를 도출합니다.

- **Technical Details**: Emotion Regulation 프레임워크는 감정적 자극에 기초한 사전 훈련(pre-training)을 통해 아래쪽 작업 최적화에서 비감정적 및 감정 영향을 받은 반응을 균형 있게 조정하는 방법을 제안합니다. 이 연구는 CIFAR-10 및 CIFAR-100과 같은 데이터셋을 이용하여 다양한 감정 데이터셋에서 ResNet 및 ViT 구조를 활용한 실험을 수행하였습니다. 최종적으로, 감정이 강화된 딥러닝 모델이 대규모 이미지 데이터셋에서 기존 방법보다 우수한 성능을 발휘함을 시사합니다.

- **Performance Highlights**: 실험 결과, Emotional Regulation 접근법이 이미지 분류 작업에서 기존 딥러닝 모델보다 뛰어난 일반화 성능을 보여주었습니다. 이 연구는 감정이 기계 학습 작업 최적화에 미치는 긍정적인 영향을 입증하며, 감정에 영감을 받은 아키텍처의 사용이 기계 학습 개선을 위한 새로운 경로가 될 수 있음을 제안합니다. 마지막으로, 이 연구는 딥러닝 모델에서의 감정적 경험이 학습 성과에 미치는 영향을 탐구함으로써 향후 연구 방향성을 제시합니다.



### The Emergence of Autonomous Penetration Capabilities in Large Language Model-Powered AI Systems (https://arxiv.org/abs/2606.13079)
- **What's New**: 이번 논문에서는 인공지능(AI) 시스템의 자율 침투(autonomous penetration) 능력을 평가하는 새로운 평가 프레임워크를 제안합니다. 기존 연구가 사용하는 비투명한 방법론이나 비현실적인 시나리오는 자율적인 사이버 공격의 실제 가능성을 제대로 반영하지 못했습니다. 이에 반해, 본 연구는 타겟 서버(target server)와 에이전트 스캐폴딩(agent scaffolding)이라는 두 가지 구성 요소로 이루어진 새로운 평가 방법론을 개발했습니다.

- **Technical Details**: 타겟 서버 측에서는 알려진 취약점이 없는 보안 서비스의 수에 따라 두 가지 수준의 타겟 환경을 설계했습니다: Tier~1(하나의 보안 서비스)와 Tier~2(세 개의 보안 서비스)로 총 300개의 타겟 서버를 구성했습니다. 에이전트 스캐폴딩은 특정 타겟에 대한 사전 지식 없이 일반 목적의 사이버 보안 도구 세트를 장착한 일반 에이전트 아키텍처를 채택합니다. 연구는 19개의 공개 가중치 및 독점 LLM을 평가했습니다.

- **Performance Highlights**: 현재 AI 모델들은 10.7%에서 69.3%까지의 침투 성공률을 기록했습니다. 이 연구는 자율 침투 능력이 모델의 전반적인 능력 향상에 따라 계속 발전하고 있다는 점을 강조합니다. 이는 AI 시스템이 실제 사이버 공격 시나리오에서 자율적으로 수행할 수 있는 능력의 증가를 보여줍니다.



### "Is This Not Enough?": Asymmetries in Institutional Accountability and Collective Sensemaking in the Case of Canada's Algorithmic Visa Triage System (https://arxiv.org/abs/2606.13071)
- **What's New**: 이 논문은 캐나다의 비자 시스템에서 알고리즘적 책임(algorithmic accountability)이 어떻게 제도적으로 표현되고 지원자들이 국경을 넘어 어떻게 경험하는지를 분석합니다. Immigration, Refugees and Citizenship Canada (IRCC)의 알고리즘 영향 평가(Algorithmic Impact Assessment, AIA)를 통해 임시 거주 비자(tempory resident visa, TRV) 분류 시스템을 분석하며, 신청자들이 Reddit에서 논의한 내용을 혼합 방법론(mixed-methods)으로 연구합니다. 연구 결과, 제도적 문서들이 투명성(transparency)과 절차적 안전장치(procedural safeguards)를 강조하는 반면, 신청자들은 종종 불투명한 결정들을 해석하기 위해 집단적인 의미 형성(collective sensemaking)에 참여한다는 사실이 드러났습니다.

- **Technical Details**: 이 논문은 알고리즘적 의사결정(adaptive decision-making)이 공공 부문에 적합하게 수정된 ADMAPS 프레임워크를 통해 어떻게 제도적으로 구조화되는지를 밝힙니다. 연구에서는 IRCC의 알고리즘 영향 평가와 Reddit 논의를 대상으로 두 가지 데이터 소스를 활용하여 시스템의 책임(accountability)과 디자인을 분석합니다. 이를 통해 기관의 책임 구성이 신청자의 경험적 해석과 어떻게 상이한지를 조사하며, 세 가지 비대칭(asymmetries)을 식별합니다: 인식적 비대칭(epistemic), 관할권 비대칭(jurisdictional), 그리고 시간-관계적 비대칭(temporal-relational)입니다.

- **Performance Highlights**: 연구 결과는 기존의 공공 부문 ADM(frameworks around Automated Decision-Making) 체계의 한계를 강조합니다. 특히, 알고리즘의 투명성과 해석성을 주장하는 제도적 발언이 신청자들에게는 쉽게 접근할 수 없는 이해로 이어진다는 점이 주목받습니다. 도출된 비대칭성은 이민 혹은 초국적 맥락에서 알고리즘 거버넌스가 어떻게 격차를 생산하는지를 보여줍니다. 따라서 이러한 경험의 불균형을 설명하기 위해 ADMAPS를 확장할 필요성이 제기됩니다.



### TWLA: Achieving Ternary Weights and Low-Bit Activations for LLMs via Post-Training Quantization (https://arxiv.org/abs/2606.13054)
Comments:
          Accepted by ICML 2026

- **What's New**: 최근 발표된 논문에서는 대형 언어 모델(LLMs)의 메모리 및 계산 비용을 줄이기 위한 효율적인 방법으로 TWLA(난수화된 가중치와 저비트 활성화)를 제안합니다. TWLA는 가중치 난수화와 활성화 저비트 양자화를 동시에 지원하는 포스트-트레이닝 양자화(PTQ) 프레임워크로, 이 조합은 엔드-투-엔드 추론 사업 효율성을 개선합니다. 이 방법은 1.58비트 가중치 압축과 4비트 활성화 양자화를 달성하면서도 높은 정확성을 유지하였습니다.

- **Technical Details**: TWLA의 구성 요소는 크게 세 가지로 나뉩니다. 첫째, E2M-ATQ는 유클리드에서 매니폴드 비대칭 양자화기를 제안하여 가중치 난수화를 위한 계층 출력 오류를 최소화합니다. 둘째, KOTMS는 크로네커 구조를 활용하여 가중치를 삼중 분포로 재구성하면서 동시에 활성화의 이상치를 통계적으로 억제하는 회전을 적용합니다. 셋째, ILA-AMP는 인접 계층 간의 이차 상호작용 비용을 고려하여 활성화 양자화의 비율 차이를 최적화합니다.

- **Performance Highlights**: TWLA는 W1.58A4 유효성 하에서도 높은 정확성을 유지하며 상당한 추론 속도 향상을 보여주었습니다. 다양한 실험을 통해 TWLA가 기존의 2비트 및 하위 2비트 방법보다 우수한 성능을 발휘하여 LLM의 PTQ 분야에서의 발전을 이끌었습니다. 이 연구는 대형 언어 모델의 실제 및 지속 가능한 배포를 위한 비용 절감 문제에 대한 실질적 해결책을 제시합니다.



### EA-WM: Event-Aware World Models with Task-Specification Grounding for Long-Horizon Manipulation (https://arxiv.org/abs/2606.13053)
- **What's New**: 본 논문에서는 EA-WM(Event-Aware World Model)라는 새로운 프레임워크를 소개합니다. EA-WM은 고정된 시각적 특징 동역학에 작업 사양에 기반한 사건 예측 및 검증을 추가하여 로봇의 상상력을 향상시킵니다. 이러한 접근법은 로봇이 다양한 작업을 수행함에 있어 더 높은 신뢰성을 보장하도록 돕는 새로운 메커니즘을 제공합니다.

- **Technical Details**: EA-WM은 시각적 특징 세계 모델을 기본 상상 엔진으로 사용하고, 그 위에 작업에 기반한 사건 예측 레이어를 추가합니다. 후보 행동 시퀀스마다 기본 모델이 미래 시각적 특징을 예측하고, 사건 예측기가 상상된 미래를 작업 관련 사건으로 매핑합니다. 이 후 검증자는 예측된 사건 상태가 작업 진행, 의미 일관성, 물리적 타당성 및 충분한 확실성을 나타내는지를 평가합니다.

- **Performance Highlights**: EA-WM은 실험을 통해 로봇의 과제 수행 능력 향상에 기여한다는 것을 입증하였습니다. 예를 들어, PointMaze에서 성공 확률이 0.90에서 0.94로 개선되었으며, Deformable 및 Wall-Single 작업에서도 각각 94% 및 95%의 성공률을 기록했습니다. WINE-RACK에서의 연구 결과는 97/100의 온라인 하이브리드 성공률을 보여줍니다.



### Fault Lines: Navigating Ethics and Responsible AI Where National Policy Meets Local Practice in Public Sector Transformation (https://arxiv.org/abs/2606.13039)
Comments:
          10 pages plus references. This study was funded by the University of Sheffield

- **What's New**: 영국 정부는 재정적 압박 속에서 공공 서비스 제공을 혁신하기 위해 AI를 적극적으로 활용하는 방향으로 나아가고 있습니다. 하지만 이러한 비전을 책임감 있는 AI 실천으로 전환하는 과정은 여전히 불확실합니다. 특히, 중앙 정부와 지역 정부 간의 책임 있는 AI 해석과 구현이 어떻게 이루어지는지에 대한 연구가 필요합니다.

- **Technical Details**: 이 논문은 영국 중앙 정부와 지역 정부 간의 인터페이스에서 책임 있는 AI가 어떻게 해석되고 실행되는지를 탐구합니다. 연구는 반구조화된 인터뷰를 통해 정책 입안자, 실무자 및 제3 섹터 전문가와 총 17회 실시하여 책임 있는 AI를 위한 장벽 및 유인 조건을 식별하였습니다. 연구 결과, AI 및 데이터 개인 정보 보호 위험, 시장-정부 비대칭, 인력 준비 부족, 표준화된 정의 및 측정 부족, 인적 책임의 공백 등 다섯 가지 주요 도전 과제가 밝혀졌습니다.

- **Performance Highlights**: 이 연구는 SEND(특수 교육 요구 및 장애) 분야를 사례로 삼아 지역 정부가 직면하고 있는 도전 과제를 강조합니다. 특히, 취약한 아동과 가족들에게 영향을 미치는 고위험 결정으로 인해 책임, 공정성, 인간 감독 주변의 긴장이 심화되고 있음을 나타냅니다. 국가는 정책 조정과 지역 수준에서의 구조적 개혁이 필요하다는 점을 강조하며, 시대에 맞는 AI 사용이 중요함을 제시합니다.



### TetherCache: Stabilizing Autoregressive Long-Form Video Generation with Gated Recall and Trusted Alignmen (https://arxiv.org/abs/2606.13035)
Comments:
          17 pages, 8 figures

- **What's New**: 이 논문에서는 드리프트(Drift) 저항성을 갖춘 장기 비디오 생성을 위한 새로운 캐시 관리 전략인 TetherCache를 제안합니다. 기존의 오토리그레시브 비디오 확산 모델은 짧은 클립에 최적화되어 있으며, 긴 비디오 스트림을 생성하는 데 어려움을 겪었습니다. TetherCache는 훈련이 필요 없는 플러그 앤 플레이(plug-and-play) 방식으로, 캐시를 쉰크(Sink), 메모리(Memory), 최근(Recent) 영역으로 편성하여 효율적이고 신뢰성 높은 비디오 생성을 지원합니다.

- **Technical Details**: TetherCache는 두 가지 상호 보완적인 메커니즘을 포함합니다: GRAB(주목 다양성 균형을 통한 게이티드 리콜)과 TAME(신뢰할 수 있는 정렬을 위한 메모리 편집)입니다. GRAB은 장기 메모리 프레임을 선택하기 위해 게이티드 점수를 사용하며, 정보적이면서도 다양한 역사를 보존합니다. TAME은 새로 호출된 메모리 토큰을 수정하여 신뢰할 수 있는 컨텍스트 분포에 통계적으로 정렬함으로써 역사적 특징의 오염을 줄입니다.

- **Performance Highlights**: TetherCache는 VBench-Long 벤치마크에서 30초, 60초, 240초 환경 모두에서 비디오 생성 품질을 지속적으로 개선했습니다. 특히 240초 생성에서 전체 및 의미 점수가 유의미하게 향상되었으며, 품질 드리프트가 7.84에서 1.33으로 감소했습니다. 실험은 GRAB과 TAME 각각의 상호 보완적 기여를 보여 주며, 이 두 메커니즘이 함께 작업하여 안정적이고 고품질의 장기 생성을 가능하게 합니다.



### Democracy in the Era of Artificial Intelligenc (https://arxiv.org/abs/2606.13026)
- **What's New**: 이 논문은 인공지능(AI)과 민주주의의 결합이 현대 사회가 직면한 심각한 도전 중 하나임을 강조합니다. 민주주의의 참여율 부족과 같은 오랜 문제를 해결할 수 있는 기회를 제공하는 AI 한편, 개인 정보 침해, 편향, 조작, 잘못된 정보 확산 등을 통해 새로운 위험 또한 초래합니다. 이 책은 민주주의와 AI의 상호작용을 재구성하는 방법을 모색하며, 어떻게 민주주의를 향상시키고 그에 필요한 가치를 재정립할 수 있을지를 탐구합니다.

- **Technical Details**: 현재 AI 시스템은 기술적 한계, 근본적 한계, 최적화 문제, 자동화, 예측 가능성, 복잡한 역학 등을 포함한 다양한 도전에 직면해 있습니다. 예를 들어, 데이터 전송은 데이터 생성에 비해 느리게 진행되고 있으며, 이는 우리가 복잡성이 증가하는 네트워크 시스템에서 적시에 적절한 결정을 내리지 못하게 합니다. 또한, 알고리즘의 복잡성 때문에 어떤 최적화 문제는 계산적으로 과도한 부담이 따르며, 사회 시스템에 대한 단일 목표 함수 최적화는 지나치게 단순화된 접근이라는 비판을 받고 있습니다.

- **Performance Highlights**: AI와 민주주의의 교차점을 다룬 이 책은 AI가 민주적 결정을 어떻게 향상할 수 있는지를 설명합니다. AI가 시민의 선호를 집계하고, 공동의 기반을 식별하며, 민주적 결과의 정당성을 강화할 수 있는 방법에 대한 연구가 포함되어 있습니다. 이 책은 민주주의를 더 나은 방향으로 발전시키기 위해 AI를 어떻게 활용할 수 있는지를 제시하며, 최신의 기술 발전과 사회적 참여를 통합하는 방안을 모색합니다.



### CausalMoE: A Billion-Scale Multimodal Foundation Model for Granger Causal Discovery with Pattern-Routed Heterogeneous Experts (https://arxiv.org/abs/2606.13024)
- **What's New**: 본 논문에서는 CausalMoE라는 새로운 방법론을 제안합니다. 이는 수십억 규모의 다중 모드 Granger 인과 모델로, 패치 수준의 이질성을 명시적으로 모델링합니다. CausalMoE는 패턴 라우트 혼합 전문가(Pattern-Routed Mixture of Heterogeneous Experts) 구조를 도입하여, 잠재적인 시간 패턴을 동적으로 식별하고 각 패치를 특정 전문가에게 라우팅합니다. 이로 인해 각 전문가의 공유 역학에서 레짐 특정 메커니즘을 분리할 수 있게 됩니다.

- **Technical Details**: CausalMoE는 LLMs(대형 언어 모델)와 VLMs(비전-언어 모델)를 결합하여, 수치 신호와 텍스트 및 시각적 정보 간의 정렬을 수행합니다. 이를 통해 인과 관계를 해석할 수 있는 희소 그래프를 복원하며, Causality-Aware Self-Attention 메커니즘을 통해 인과적 신뢰성을 높입니다. 각 시간 시퀀스 패치를 가장 적합한 도메인 전문가에게 동적으로 라우팅하여, 비모수적 causal recovery를 가능하게 합니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, CausalMoE는 기존의 감독된 벤치마크에서 새로운 최첨단 성능을 달성하며, 전통적인 방법이 실패하는 경우에도 효과적으로 일반화되는 모습을 보여줍니다. 특히, 복잡한 시간 설정과 소수 샷 사례에서도 높은 성능을 기록하였습니다. 이러한 방식으로 CausalMoE는 시간이 변하는 다양한 패턴을 모델링하면서 인과 추론을 가능하게 합니다.



### scLLM-DSC: LLM-Knowledge Enhanced Cross-Modal Deep Structural Clustering for Single-Cell RNA Sequencing (https://arxiv.org/abs/2606.13007)
- **What's New**: 이번 연구에서는 scRNA-seq 분석을 위한 새로운 클러스터링 프레임워크인 scLLM-DSC를 제안합니다. 기존의 데이터 기반 접근 방식에서 벗어나, NCBI 유전자 사전과 Cell2Sentence 임베딩을 활용하여 생물학적 의미에 기반한 표현을 수립합니다. 이를 통해 생물학적 의미와 전사체 특성의 일관성을 강화하는 교차-모달 대비 정렬 메커니즘을 도입했습니다.

- **Technical Details**: scLLM-DSC는 지식 주도 사전(Knowledge-Driven Priors), 구조 인식 톱니바퀴(Structure-Aware Topology), 교차-모달 의미 정렬(Cross-Modal Semantic Alignment)이라는 세 가지 핵심 기둥으로 구성됩니다. 이 모델은 LLM을 데이터 생성기가 아닌 의미 매퍼로 활용하여, 클러스터링 과정에 생물학적 의미를 명시적으로 통합합니다.

- **Performance Highlights**: extensive benchmarks에 따르면, scLLM-DSC는 11가지 최첨단 기준선 모델에 비해 클러스터링 정확도에서 유의미한 성능 향상을 보여주었습니다. 이 모델은 정확성과 효율성 모두에 있어 우수한 결과를 보이며, 클러스터링 결과에 생물학적 귀속을 명시적으로 부여하는 독특한 특성을 가지고 있습니다.



### A Machine Learning Framework for Real-Time Personalized Ergonomic Pose Analysis (https://arxiv.org/abs/2606.12988)
Comments:
          13 pages, 7 figures, conference 24CMH

- **What's New**: 이 논문에서는 3차원 볼륨 비디오 데이터를 활용하여 인체의 인체공학적(ergonomic) 및 비인체공학적(non-ergonomic) 자세를 실시간으로 예측하는 새로운 방법론을 소개합니다. 이 방법론은 인체공학 평가를 위해 설계되었지만, 인간 자세의 실시간 분석이 필요한 다른 응용 분야에도 적합하게 조정될 수 있습니다. 특히, 이 시스템은 평가 과정에서 3D 포인트 클라우드를 분석하는 능력으로 차별화되며, 여러 각도에서의 계산이 가능합니다.

- **Technical Details**: 시스템은 사용자가 선택하고 라벨링한 자세만을 개인화된 딥러닝 분류기(training classifier) 훈련에 사용하므로, 연속적으로 자동으로 자세 추정을 수행합니다. RGB-D 카메라를 이용한 사례 연구(case study)를 통해 하중을 들어올리는 작업을 수행하는 피험자들을 촬영하여 실시간 골격 라벨링(skeletal labeling)을 가능하게 하였습니다. 모델은 이 데이터를 기반으로 훈련되어 새로운 스트리밍 데이터에서 실시간으로 추론을 수행합니다.

- **Performance Highlights**: 이 연구는 최첨단 3D 데이터 기술과 전통적인 2D 자세 추정 알고리즘을 결합함으로써 실시간 인체공학 평가를 위한 확장 가능하고 실용적인 접근 방식을 제공합니다. 이는 작업 환경에서의 안전 및 건강 모니터링에 대한 증가하는 필요를 해결하는 것을 목표로 하며, 해당 분야에 중요한 기여를 하고 있습니다.



### Diffusion Transformer World-Action Model for AV Scene Prediction (https://arxiv.org/abs/2606.12987)
Comments:
          10 pages, 9 figures, 2 tables

- **What's New**: 이 논문은 자율주행 차량을 위한 action-conditioned world models를 제안하여, 차량의 계획된 제어로부터 향후 카메라 장면을 예측할 수 있게 하였습니다. 이 모델은 현실 세계에서의 시뮬레이션 없이 계획 및 시뮬레이션을 가능하게 하며, ambiguous한 미래 예측 문제를 다루고 있습니다.

- **Technical Details**: 제안된 모델은 compact latent world model로, 현재의 전면 카메라 latent와 일련의 ego-actions(조향각, 가속도)을 입력으로 하여 최대 8초까지 미래 장면의 latent를 예측합니다. VAE(Variational Autoencoder)를 freezer하여 예측된 latent를 256x256의 이미지로 변환하는 파이프라인을 구축하였습니다. 또한, V-JEPA2와 같은 frozen encoders의 성능을 종합적으로 비교하여 성과를 입증했습니다.

- **Performance Highlights**: 모델은 KID(Kernel Inception Distance)에서 0.078의 수치를 기록하며, 기존의 회귀 모델보다 4.8배 뛰어난 성과를 보여주었습니다. 이는 distortion metric을 최적화하기보다는 분포 기반 평가지표를 사용하여 성능을 평가해야 함을 시사합니다. 또한 action controllability가 평균 Spearman $ho=0.81$로 확인되어, 모델의 제어 가능성을 증명하였습니다.



### Efficient, Robust, and Anti-Collusion Fingerprinting of Image Diffusion Models (https://arxiv.org/abs/2606.12977)
- **What's New**: 본 논문에서는 기존의 생성 모델 핑거프린팅 방법들의 약점을 밝히고, 특정 사용자 식별자를 생성된 이미지에 삽입하는 방법을 제안하고 있습니다. 특히, 여러 공격자가 자신의 모델을 결합하여 핑거프린트를 제거할 수 있는 공격에 대한 내성을 갖춘 핑거프린팅 방법을 처음으로 제안합니다. 새로운 개인화 정규화 모듈(PNM)을 통해 핑거프린트를 생성된 이미지에서 안정적으로 복원할 수 있도록 하는 방법을 개발하였습니다.

- **Technical Details**: 제안된 방법은 비트 문자열인 핑거프린트를 T2I 모델에 통합된 PNM의 계수에 인코딩합니다. 또한, 손실이 없는 Anti-Collusion Transformation(ACT)을 도입하여, 여러 모델의 합작 공격이 발생할 경우 이미지 생성 품질이 저하되도록 합니다. 이 방법은 프리-트레이닝된 T2I 모델의 변형 오토 인코더(VAE)에 PNM을 결합하여 개인화된 모델을 데이터 재학습 없이 구현할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 고해상도의 이미지 생성 및 편집 작업에서 99.5% 이상의 핑거프린트 추출 정확도를 달성했습니다. 특히, COCO 데이터셋에서 2인 공격의 경우 Fréchet Inception Distance(FID)가 23에서 79로 상승하여 모델의 사용성을 현저히 낮추는 효과를 보였습니다. 이 방법은 이미지 품질 저하 상황에서도 높은 핑거프린트 추출 정확도를 유지하여 기존 방법들보다 더욱 향상된 안전성을 보여주었습니다.



### An Embodied Simulation Platform, Benchmark, and Data-Efficient Augmentation Framework for Wet-Lab Robotics (https://arxiv.org/abs/2606.12936)
Comments:
          25 pages, 17figures

- **What's New**: 이번 논문에서는 생물의학 실험의 재현성, 처리량, 그리고 안전성을 개선할 수 있는 Wet-lab 로봇을 위한 새로운 플랫폼인 Pipette를 소개합니다. Pipette는 사용자 맞춤형 시뮬레이터, 개방형 편집 가능한 실험실 자산, 그리고 제한된 시연을 사용 가능한 교육 데이터로 전환하는 효율적인 파이프라인을 제공합니다. 특히, Pipette는 43개 이상의 오픈 소스 자산을 제공하며, 사용자가 쉽게 수정할 수 있습니다.

- **Technical Details**: Pipette의 핵심 구성 요소는 시뮬레이션 기반 데이터 증강 파이프라인입니다. 이 시스템은 인간 시연을 시뮬레이션에서 재생하며, 조명, 카메라, 속도, 행동의 변동을 적용하고 자동 작업 성공 체크를 통해 생성된 에피소드를 필터링합니다. 이를 통해 제한된 수의 수동 시연에서 사용 가능한 훈련 데이터를 신속히 확장할 수 있습니다.

- **Performance Highlights**: Pipette는 11개 Wet-lab 작업의 기준선을 설정하고, 샘플 처리, 배양기구 조작, 장비 작동 및 정밀 배치를 포함하는 다양한 작업을 평가합니다. 각 작업당 단 30회의 시연으로 평균 65.5%의 성공률을 기록했으며, 시뮬레이션 증강이 SmolVLA의 성공률을 44.1%에서 74.7%로, 그리고 {C0}0의 성공률을 40.4%에서 46.5%로 향상시켰습니다. Pipette는 또한 비전문가 사용자도 새로운 Wet-lab 로봇 작업을 정의할 수 있도록 자연어 기반의 장면 구성 및 작업 등록을 지원합니다.



### Order Is Not Contro (https://arxiv.org/abs/2606.12923)
Comments:
          52 pages, 7 figures

- **What's New**: 이 논문은 AI 정렬(alignment), 해석 가능성(interpretability), 조절(steering), 그리고 신경 교란(neural perturbation) 연구들이 질서를 유도하는 객체들을 식별함을 주장합니다. 질서는 제어(control)와는 다르며, 제어는 수신자-게이티드(responder-gated) 반응 법칙이 필요하다고 설명합니다. 이 법칙은 물리적 상태, 행동, 환경, 그리고 수신자 상태를 반응 이동, 싱크(sink), 노력(effort), 그리고 오목형 프로젝션(basin projection)으로 매핑하는 것을 포함합니다. 이를 통해 생물학, 대형 언어 모델(LLM), 어댑터, 그리고 확률적 작용 패널에서 제어의 개념을 식별합니다.

- **Technical Details**: 논문은 외부 개입이 반응 법칙을 생성하는 방식과 그 작용을 다룹니다. 성질이 명시된 매개변수(denominator)를 통한 스토캐스틱 반응 커널(𝒫δ(dy|x,a))을 도입하고, 이는 물리적 상태, 행동 또는 드라이브(drive), 배스(bath), 수신자 상태, 비교(factor)에서 반응 위치와 관련된 분포를 유도합니다. 응답 법칙은 이 커널의 요약 및 유한 차이로 서술되며, 반응 경로 ℛδ와 행동 효과 Δℛδ로 표현됩니다. 제어는 한계가 명확하게 설정된 상태에서의 개입 노력(finited intervention effort)에 의해 부여됩니다.

- **Performance Highlights**: 연구의 성과는 생성된 출력과 어댑터 조건에서 예측 가능한 반응 법칙을 보여줍니다. 네 가지 물질 조건에 걸쳐 반응 벡터는 72.8-73.7%의 컴포넌트 신호 정확도를 기록하였으며, 비제로 컴포넌트에 대해서는 84.3-84.8%로 증가합니다. 관찰자들은 시스템 효과와 목표 가족의 예측에서 각각 93.6% 및 91.7%의 정확도를 기록하며, 이는 실질적인 응용 가능성과 관련이 있음을 시사합니다.



### LoRA-Muon: Spectral Steepest Descent on the Low-Rank Manifold (https://arxiv.org/abs/2606.12921)
Comments:
          20 pages, 4 figures

- **What's New**: 이번 논문에서는 Low-Rank Adaptation (LoRA)에 대한 새로운 옵티마이저인 LoRA-Muon을 도입하였습니다. LoRA-Muon은 Muon 옵티마이저의 스펙트럴 스티프 데센트 규칙을 저랭크 설정에 적용하여 성능을 최적화합니다. 이를 통해 LoRA의 학습률이 다양한 설정에서 효과적으로 전이될 수 있도록 하였습니다.

- **Technical Details**: LoRA-Muon은 저랭크 매니폴드에서의 스티프 데센트 문제를 직접적으로 해결함으로써 개발되었습니다. 이 옵티마이저는 스펙트럴 노름을 통해 학습 과정을 조절하며, 다른 하이퍼파라미터와의 관계를 고려할 때 우수한 성능을 발휘합니다. Splitting weight-decay 규칙을 도입하여, 전체 랭크와 저랭크 설정에서 가중치 노름과 스텝 크기가 일치하도록 보장하였습니다.

- **Performance Highlights**: TinyShakespeare 실험에서, 랭크-2의 LoRA-Muon 프록시는 밀집 Muon에 대한 가장 좋은 테스트된 학습률을 회복하였으며, 랭크-32에서 LoRA-Muon은 평균 검증 손실이 밀집 기준선보다 낮았습니다. LoRA-Muon은 메모리 효율성을 증가시키고 계산 자원에 대한 요구를 줄이며, 여러 하이퍼파라미터 스케일링 법칙을 통해 최적의 학습률이 다른 하이퍼파라미터 값 도출에 유용하다는 점을 입증하였습니다.



### MAStrike: Shapley-Guided Collusive Red-Teaming on Multi-Agent Systems (https://arxiv.org/abs/2606.12918)
- **What's New**: 이번 연구에서는 다중 에이전트 시스템(MAS)의 안전성을 보장하기 위해 MAStrike라는 폐쇄 루프(red-teaming) 프레임워크를 제안합니다. 이 프레임워크는 Shapley value 분석을 통해 각 에이전트의 시스템 안전성 기여도를 정량화합니다. 이 연구는 기존의 휴리스틱 방식의 한계를 극복하고, 에이전트 간의 협동 공격을 효과적으로 생성할 수 있는 방법을 제공합니다.

- **Technical Details**: MAStrike는 각 에이전트의 역할과 기능에 기초하여 결합된 공격 전략을 생성합니다. Shapley value 분석을 통해 각 에이전트의 중요성과 상호작용 효과를 포착하며, 이를 통해 공격 성공률(ASR)을 최적화합니다. 연구진은 또한 구조적 실패 진단 메커니즘을 통해 기습 전략을 반복적으로 개선합니다.

- **Performance Highlights**: 실험 결과, MAStrike는 기존의 휴리스틱 기반 방법들을 크게 능가하며, 다양한 금융, 소프트웨어 엔지니어링 및 CRM 분야에서 에이전트 간의 협동 공격을 성공적으로 수행할 수 있음을 보여주었습니다. 이 연구는 다중 에이전트의 취약점과 협동 패턴을 분석하여, 기존의 단일 에이전트 접근 방식으로는 간과되었던 중요한 발견들을 제공합니다.



### Bounding Boxes as Goals: Language-Conditioned Grasping via Neuro-Symbolic Planning (https://arxiv.org/abs/2606.12910)
Comments:
          Project website: this https URL

- **What's New**: GRASP (Grounded Reasoning and Symbolic Planning) 프레임워크는 자연어 지시를 해석하고 이를 미리 훈련된 VLM (Vision-Language Model)을 통해 신경-상징적 목표 상태로 변환하는 방법을 제안합니다. 기존의 고정된 색상 목록이나 하드코딩된 좌표에 의존하지 않고 로봇이 '상단 선반(top shelf)'과 같은 추상적인 공간 개념을 이해하고 추가적인 미세 조정 없이 작업을 수행할 수 있게 합니다. 이는 가벼운 neuro-symbolic 시스템을 통해 언어 조건 조작의 효율성을 높입니다.

- **Technical Details**: GRASP 시스템은 자연어 프롬프트를 해석하기 위해 LLM (Large Language Model)과 GroundingDINO와 같은 VLM을 사용합니다. 시스템은 사용자로부터 입력된 명령을 JSON 파일로 저장한 객체 경계 상자로 변환하고, 객체의 위치와 목표 상태를 연결하는 두 가지 컴포넌트를 포함합니다. 이를 통해 GRASP는 로봇의 작업을 지속적으로 추적하고 조정하여 높은 효율성을 유지합니다.

- **Performance Highlights**: GRASP는 90개의 실제 로봇 시험에서 전반적으로 73.3%의 성공률을 기록하며, 태스크 특정 훈련 없이 세 가지 난이도에서 작업을 수행할 수 있음을 입증합니다. 이러한 성능 향상은 새로운 사용자 명령에 대한 일반화 능력이 뛰어나다는 것을 보여주며, 태스크의 복잡성에 관계없이 로봇의 조작을 가능하게 합니다.



### PolicyGuard: Towards Test-time and Step-level Adversary Defense for Reinforcement Learning Agen (https://arxiv.org/abs/2606.12896)
- **What's New**: 이 논문은 강화 학습(Reinforcement Learning, RL) 시스템의 보안을 강화하기 위해 새롭게 제안된 방어 기법인 PolicyGuard를 소개합니다. PolicyGuard는 Gaussian Process(GP)를 활용하여 backdoor 공격을 방어하며, 테스트 시간에 특정 시간 단계에서 발생하는 의심스러운 행동을 동적으로 감지할 수 있도록 설계되었습니다. 이를 통해 기존의 방어 방법들이 가지던 제한들을 극복하고, RL 에이전트의 안전성을 높이는 데 기여합니다.

- **Technical Details**: PolicyGuard는 Bayesian 방식으로 불확실성을 정량화하기 위해 GP의 posterior variance를 활용합니다. 이 기법은 정상적인 행동 패턴을 학습한 후, 각 상태-행동 쌍에 대해 pseudo trajectories를 생성하여 posterior variance를 계산합니다. 이를 통해 RL 시스템 내에서 발생할 수 있는 backdoor 공격에 의한 행동 변화를 효과적으로 감지할 수 있습니다.

- **Performance Highlights**: 실험 결과 PolicyGuard는 7개의 RL 게임에서 기존 방어 방법들보다 우수한 성능을 보였습니다. 특히, perturbation 기반 공격과 적대적 에이전트 공격에 대해서 각각 평균 AUROC 0.856과 0.859를 달성하여 상황에 맞는 강력한 탐지 성능을 입증했습니다. 이 연구는 RL 분야에서 보안의 중요성을 강조하고, 실제 애플리케이션에 적용할 수 있는 방어 기법을 제시합니다.



### Bridging Modal Isolation in Interleaved Thinking: Supervising Modality Transitions via Stepwise Reinforcemen (https://arxiv.org/abs/2606.12886)
Comments:
          22 pages, 5 figures, 6 tables

- **What's New**: 이 논문에서는 통합 다중 모드 모델(Uniﬁed Multimodal Models, UMMs)을 사용한 상호작용적 사고(interleaved thinking)가 공간 및 물리적 작업에 대해 promising한 결과를 보이고 있음을 설명합니다. 그러나 복잡한 장기 시나리오에서는 생성된 이미지가 텍스트 맥락과 다르게 나타나는 문제, 즉 Modal Isolation을 지적합니다. 이는 서로 다른 두 모드가 유기적으로 정보를 공유하지 않고 독립적으로 작동하는 문제로, 정보 손실이 모드 경계에서 발생하는 것에 기인합니다.

- **Technical Details**: 연구팀은 Modal Isolation을 구체화하고, 이를 해결하기 위해 두 단계의 훈련 프레임워크인 MoTiF(Modal Transition Fidelity)를 제안합니다. 이 프레임워크는 오류가 발생한 시각적 출력에서 회복할 수 있도록 훈련시키는 Reflective SFT와 강화 학습을 통해 이미지 생성의 충실도를 높이는 Flow-GRPO를 포함합니다. 이 연구는 각 모드 경계에서 발생하는 정보 손실을 양적으로 평가하고 이를 기반으로 훈련을 진행합니다.

- **Performance Highlights**: 네 가지 시각 퍼즐 벤치마크에서 MoTiF 방법론은 상호적 일관성 및 최종 작업 정확도를 크게 향상시켰습니다. 모드 경계에서의 명확한 구조적 감독이 장기 상호작용적 사고에 매우 중요하다는 점을 강조하며, 단순한 크기 조정이나 최종 작업 최적화만으로는 충분하지 않음을 보여줍니다. 이러한 결과들은 각 모드가 상호 의존적으로 작동해야 함을 입증하며, 이를 통해 모델의 장기적 사고 능력이 극대화된다는 것을 나타냅니다.



### Beyond Problem Solving: UOJ-Bench for Evaluating Code Generation, Hacking, and Repair in Competitive Programming (https://arxiv.org/abs/2606.12864)
- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)이 경쟁 프로그래밍(Competitive Programming) 환경에서 인간의 학습을 지원하는 방식에 대한 새로운 벤치마크인 UOJ-Bench를 소개합니다. 이 벤치마크는 LLMs의 문제 해결 능력뿐만 아니라 인간이 작성한 코드의 오류를 식별하는 능력도 평가합니다. UOJ-Bench는 코드 생성, 코드 해킹, 코드 수리의 세 가지 과제로 구성되어 있으며, 실제 데이터와 UOJ의 평가 시스템을 활용하여 측정됩니다.

- **Technical Details**: UOJ-Bench의 두 가지 난이도인 Easy와 Hard 레벨을 통해 LLMs가 인간이 설계한 표준 테스트 케이스로 드러날 수 있는 명백한 오류와 숨겨진 오류를 식별하고 수정할 수 있는 능력을 평가합니다. Easy 레벨은 사용자 제출 코드에서 부분 점수를 받아 완전 점수로 수정된 500 쌍의 문제를 기반으로 합니다. 반면 Hard 레벨은 UOJ의 Hack 메커니즘을 활용하여 다양한 숨겨진 오류를 찾는 작업으로, 2,000건 이상의 성공적인 해킹 사례를 축적하여 풍부한 데이터 소스를 제공합니다.

- **Performance Highlights**: 실험 결과, LLMs는 Easy 레벨에서 60% 미만의 성공률을 보였고 Hard 레벨에서는 50% 미만의 성공률을 기록했습니다. 그러나 테스트 시간 동안 스케일링을 통해 성공률이 90% 이상으로 향상되는 것을 보였습니다. 특히, 테스트 시간 스케일링 하에 GPT-OSS-120B 모델은 Zero-Day-Hacking-5K에서 10% 이상의 오류를 찾는 것으로 나타났으며, 이는 현재 LLMs가 기존의 표준 평가 시스템을 넘어서는 보완적인 신호를 제공할 수 있는 가능성을 나타냅니다.



### JSCGC: Joint Source-Channel-Generation Coding for Wireless Generative Communications (https://arxiv.org/abs/2606.12858)
Comments:
          submitted to IEEE Journal

- **What's New**: 이 논문에서는 기존의 압축-통신 시스템에서 제안된 새로운 개념인 Joint Source-Channel-Generation Coding (JSCGC) 을 도입합니다. JSCGC는 전통적인 복원 방식을 지양하고 생성을 위한 제어 방식으로 통신 방식을 전환합니다. 기존의 디코더를 생성 모델로 교체하여 수신된 신호를 변별적인 조건으로 사용함으로써, 명확한 왜곡 최소화 대신 상호 정보 극대화를 목표로 합니다.

- **Technical Details**: JSCGC 프레임워크에서는 통신 목표를 지각적 제약 하에서 상호 정보 극대화로 정식화하며, 명시적인 왜곡 함수에 의존하지 않습니다. 이를 위해 변분 추론(variational inference)과 라그랑지안 완화(Lagrangian relaxation)를 이용하여 효율적인 인코딩, 전송 및 생성의 끝-끝 학습(end-to-end learning)을 가능하게 하는 목표를 도출합니다. 또한, 단계적 샘플링 전략이 개발되어 제한된 통신 자원 하에서도 고품질의 생성을 가능하게 합니다.

- **Performance Highlights**: JSCGC는 다양한 채널 조건 하에서 기존의 JSCC 및 생성 통신 기준선들에 비해 시각적 품질과 의미적 일관성에서 성능이 우수함을 여러 실험을 통해 입증합니다. 예를 들어, AWGN 채널 설정에서 SNR이 5 dB인 Kodak 데이터셋을 사용한 실험에서 JSCGC는 LPIPS와 FID 지수를 각각 79.42% 및 53.68%로 감소시키며, CLIP 점수를 11% 향상시켰습니다.



### TimeROME-DLM: Temporal Causal Tracing and Low-Rank Inference-Time Knowledge Editing for Masked Diffusion Language Models (https://arxiv.org/abs/2606.12841)
- **What's New**: 본 논문은 MDLM(Masked Diffusion Language Models)에 대한 새로운 지식 편집 프레임워크인 TimeROME-DLM을 제안합니다. 이 방법은 기존의 AR(Autoregressive) 트랜스포머 모델을 타겟으로 한 지식 편집 및 유학 방법의 한계를 극복하며, 단계적 디노이징(iterative denoising)을 고려하여 설계되었습니다. TimeROME-DLM는 훈련 없이 신뢰할 수 있는 인퍼런스 시간 지식 편집을 가능하게 하여, 저사양의 하드웨어에서도 사용할 수 있도록 합니다.

- **Technical Details**: 이 프레임워크는 Temporal Indirect Effect (TIE)로 알려진 인과 추적(causal tracing) 프로토콜을 결합하여 정보를 수집하고, 이를 통해 효과적으로 사실과 관련된 조정 좌표를 식별합니다. 이러한 접근법은 각 사실에 대해 여러 디퓨전 패스를 활용하여, 저장된 주제 표현과의 일치를 측정하고 적절한 업데이트를 제공합니다. 모델의 백본 가중치는 그대로 유지되며, 세 가지 하이퍼파라미터만 조정하여 최적의 성능을 도출합니다.

- **Performance Highlights**: TimeROME-DLM는 TOFU forget01 실험에서 약 83 nats의 로그 확률을 감소시켰으며, 50개의 연속적으로 삽입된 사실에 대해서는 로그 확률이 거의 변함없이 유지됩니다. 이 프레임워크는 기존의 방법들에 비해 병목 현상을 해결하고, 추가 VRAM이 필요치 않으면서도 속도는 네 배에서 열네 배 향상됨을 보여줍니다. 게다가, TimeROME-DLM은 400개의 사실에 대해 서브 선형적으로 확장이 가능합니다.



### OCOO-T : A Simple and Scalable Virtual Cell Model for Transcriptional Perturbation Response Prediction (https://arxiv.org/abs/2606.12838)
Comments:
          22 pages, 6 figures

- **What's New**: 이번 논문에서는 유전자, 화학 물질, 세포 사이토카인의 변화에 대한 단일 세포의 전사 반응 예측을 위한 OCOO-T라는 새로운 모델을 소개합니다. 이 모델은 강력하지만 복잡한 기존 접근 방식을 단순화하며, 기존의 고차원 표현 프로파일을 직접 처리할 수 있는 기본 Transformer 스택을 사용합니다. OCOO-T는 연속적 시간의 디노이징 프로세스로 변화를 예측하며, 이를 통해 대규모 데이터셋에서도 뛰어난 성능을 보여줍니다.

- **Technical Details**: OCOO-T 모델은 변환기 아키텍처를 기반으로 하며, 유전자 발현 프로파일을 연속적으로 디노이즈하는 데 중점을 둡니다. 이 모델은 적응형 레이어 정규화와 맥락 토큰을 사용하여 다양한 세포주 및 세포 유형의 특성에 맞춰 퍼터베이션 값과 용량 정보를 통합합니다. OCOO-T는 긴 발현 프로파일을 패칭(patching)과 언패칭(unpatching) 기법을 통해 효율적으로 처리합니다.

- **Performance Highlights**: OCOO-T는 Tahoe100M, Replogle 및 PBMC 벤치마크에서 성능 평가를 통해 다양한 변화 및 세포 유형에 대해 최첨단 성능을 달성했습니다. 이 모델은 복잡한 보조 분포 정렬 모듈 없이도 효과적인 변화를 모델링할 수 있음을 보여줍니다. 전반적으로, OCOO-T는 단순하면서도 효과적인 단일 세포 오믹스 모델을 위한 강력하고 확장성이 뛰어난 프레임워크를 제공합니다.



### The Internet of Agentic AI: Communication, Coordination, and Collective Intelligence at Sca (https://arxiv.org/abs/2606.12835)
- **What's New**: 자율 AI 에이전트의 급속한 출현은 인공지능을 고립된 모델 추론에서 분산된 추론, 통신 및 행동의 시스템으로 변모시키고 있습니다. 이 논문은 이질적인 에이전트들이 서로를 발견하고, 책임을 협상하며, 맥락을 교환하고, 도구를 호출하고, 클라우드, 엣지, 장치 및 사이버 물리적 환경의 워크플로우를 실행하는 오픈 생태계인 IoAI를 개발하는 비전을 제시합니다. 에이전트의 배치 모델, 워크플로우 생애 주기, 통신 프로토콜 등을 검토하며 적응형 제조와 분산된 운영 조정에 대한 사례 연구를 포함하고 있습니다.

- **Technical Details**: 본 논문은 단일 에이전트 AI, 다중 에이전트 시스템, 분산 컴퓨팅, 통신 네트워크, 게임 이론 및 보안 공학의 기초를 조합하여 확장 가능한 에이전트 생태계에 필요한 아키텍처와 메커니즘을 특성화합니다. IoAI는 자율적인 인공지능 에이전트가 서로 통신하고 협력하며, 복잡한 작업을 독립적으로 수행하는 데 필요한 능력, 정보, 자원 인식을 조합하여 동작하는 구조로 설명됩니다. 에이전트는 외부 API를 호출하고, 데이터베이스와 상호작용하며, 실행 가능한 코드를 생성할 수 있는 능력을 가집니다.

- **Performance Highlights**: 이러한 자율 에이전트들은 복잡한 작업을 협력적으로 수행하기 위해 서로 의존적인 하위 작업으로 목표를 분해하는 특성을 가지고 있습니다. 논문에서 다루는 주요 연구 과제에는 제어되는 출현, 의미론적 상호 운용성, 안전한 신원, 인센티브 호환형 조정, 자원 인식 오케스트레이션 및 거버넌스가 포함됩니다. 이러한 에이전트들은 예를 들어 분산 사이버 방어, 협력적 과학 발견 및 자율 물류 조정과 같은 대규모 운영 환경에서 중요한 역할을 하고 있습니다.



### Perceive, Interact, Reason: Building Tool-Augmented Visual Agents for Spatial Reasoning (https://arxiv.org/abs/2606.12830)
- **What's New**: 이번 연구에서는 PERception-Interaction-reason Agent (PERIA)를 도입하여, 시각적 추론(task such as spatial reasoning) 작업을 위한 도구 증강된 비주얼 에이전트를 설계했습니다. PERIA는 비주얼 맵 추론(map reasoning), 시각적 탐색(visual probing), 비전 재구성(vision reconstruction) 등 여러 과제를 해결하는 기능을 제공합니다. 특히 두 가지 도구 세트를 통해 자주 사용하는 시각적 증거를 효과적으로 처리할 수 있습니다.

- **Technical Details**: PERIA는 비주얼 인지 도구(vision perception tools)와 비주얼 상호작용 도구(vision interaction tools)로 구성된 두 개의 상호 보완적인 도구 세트를 활용하여, 다양한 시각적 증거를 수집하고 이를 바탕으로 공간적 상태를 형성합니다. 또한, Observation-Relaxed Group-in-Group Policy Optimization (OR-GIGPO)을 통해 멀티 도구 사용 경로에 대한 정책 최적화를 수행합니다. 이 방식은 다단계 비주얼 도구 사용 시의 신뢰성 있는 결과를 보장합니다.

- **Performance Highlights**: PERIA-8B는 8개 데이터셋의 13개 벤치마크에서 이전의 최첨단 효과적 기준보다 7.0%-14.8% 향상된 성능을 보여줍니다. 이는 크기가 비슷한 이전 모델들과 비교하여도 효과적이며, Qwen3-VL-235B-A22B-Thinking 및 GPT-5와 같은 더 큰 모델과도 동등한 성능에 도달했습니다. 이러한 결과는 PERIA가 공간 추론 능력을 크게 향상시킨다는 것을 나타냅니다.



### DIMOS: Disentangling Instance-level Moving Object Segmentation (https://arxiv.org/abs/2606.12826)
- **What's New**: 이 논문에서는 Moving Instance Segmentation (MIS) 분야에서의 새로운 접근법을 제시하며, 이벤트 카메라와 이미지 기반 특징을 융합하여 성능을 향상시키는 방법을 탐구합니다. 기존의 MIS 방법들이 작은 움직이는 객체를 정확히 분리하는 데 한계를 보인 점을 해결하기 위해, 색상과 동작 정보를 분리하여 추출하는 이중 분리(feature disentanglement) 프레임워크를 처음으로 제안합니다. 이런 새로운 접근을 통해, 이미지와 이벤트 모달리티 간의 상호 작용을 개선하고 있습니다.

- **Technical Details**: 제안된 DIMOS 프레임워크는 이중 분리 인코더를 사용하며, 각 모달리티에서 색상과 동작 특징을 따로 추출합니다. 이를 통해 특징 밀도를 증가시키고, 이미지와 이벤트 간의 다중 해상도 크로스 모달 정렬(multi-granularity cross-modal alignment)을 적용하여 양질의 특징 융합을 통해 더 나은 성능을 달성합니다. 이 과정에서, intra-modal contrastive learning을 통해 색상과 동작 정보 간의 차별성을 높이고, 작업에 특화된 감독을 통해 분리된 특징을 학습합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법이 멀티모달 MIS 분야에서 최신 성능을 달성했음을 입증하였으며, 특히 빠른 움직임 및 저조도 환경에서 작은 인스턴스를 다루는 데 뛰어난 성능을 보였습니다. 기존의 방법들에 비해 더 높은 분할 정확도를 이루며, 다양한 환경에서의 안정성을 증명하였습니다.



### Acquisition state behaves as a structured, measurable variable governing lung-nodule AI: kernel-driven measurement instability and noise-driven detection fragility, invisible to DICOM metadata (https://arxiv.org/abs/2606.12824)
- **What's New**: 의료 이미징을 위한 AI 관리의 새로운 발전으로, 2026 ACR-SIIM Practice Parameter에서는 지역 수용 테스트(local acceptance testing)와 지속적인 드리프트 모니터링(drift monitoring)의 필요성을 권장하고 있습니다. ACR Assess-AI 레지스트리는 DICOM metadata를 활용하여 AI 출력의 모니터링을 수행하고, 이는 현재 시점에서 AI 성능 변화의 원인을 파악하기 위한 기초 자료를 제공합니다.

- **Technical Details**: 이 연구에서는 LUNA16으로 훈련된 MONAI RetinaNet 폐 결절 탐지기를 사용하여 AI 성능 변동의 원인으로 작용할 수 있는 입력 요소인 'acquisition state'를 측정 가능한 변수로 검증하였습니다. CT 스캔에서 재구성 커널(reconstruction kernel)의 변화가 AI가 측정한 결절의 지름과 이를 통한 크기 카테고리 분류에 미치는 영향을 평가하였으며, 그 결과 커널 변화만으로도 지름 변화가 발생하는 것을 확인했습니다.

- **Performance Highlights**: 총 155개의 결절을 대상으로 한 실험에서 커널 변화에 따라 평균 0.27mm의 지름 변화가 있었고, 5.2%의 결절이 Fleischner 크기 카테고리를 넘는 결과를 보였습니다. 하지만 탐지 신뢰도(detection confidence)는 두 커널 간 통계적으로 유의미한 변화가 없었고, 이는 AI 모델의 성능이 다양한 재구성 조건에서도 일관성을 유지할 수 있음을 나타냅니다.



### Localizing Anchoring Pathways in Language Models (https://arxiv.org/abs/2606.12818)
- **What's New**: 이번 논문에서는 언어 모델(Language Model)이 부적절한 숫자가 예측을 변화시키는 앵커링 효과(anchoring effect)에 대한 연구를 다룹니다. 저자들은 기존의 블랙 박스 방식에서 벗어나, 숫자 앵커가 내부 계산에 어떻게 영향을 미치는지 분석하였습니다. 이를 통해 다양한 모델의 내부 경로가 앵커에 민감한 경쟁을 어떻게 지원하는지를 탐구했습니다.

- **Technical Details**: 연구에서는 7B-8B Qwen 모델과 Llama 모델을 대상으로 회로 로컬리제이션(circuit localization) 기법을 사용하여, 엣지 수준(Edge-level) 방법이 노드 수준(Node-level) 방법보다 더 정확하게 앵커 신호를 포착한다는 결과를 도출했습니다. 저자들은 모델에 따라 저앵커와 고앵커 회로가 밀접하게 연관되어 있으며, 훈련 후에는 어떤 경로가 제일 중요한지를 변화시키는 경향이 있음을 확인하였습니다.

- **Performance Highlights**: 저자들은 고정된 답안 옵션을 통한 통제된 다중 선택 과제를 설정하여, 앵커가 주어진 맥락 속에서 모델의 추론을 어떻게 변화시키는지를 평가했습니다. 이 연구 결과는 앵커링 효과가 모델의 내부 경로에 따라 다르게 작용하는 방식에 대한 기계적 해석을 제공하며, 다양한 모델 패밀리 간의 경로 전파 방법을 설명합니다.



### Stubborn: A Streamlined and Unified Reinforcement Learning Framework for Robust Motion Tracking and Fall Recovery for Humanoids (https://arxiv.org/abs/2606.12814)
- **What's New**: 최근 강화학습(Reinforcement Learning) 기반의 새로운 방법인 Stubborn이 소개되었다. 이 방법은 인간형 로봇의 움직임 추적 성능을 향상시키고 불안정한 상태에서의 낙상 회복을 달성하는 데 중점을 두고 있다. 기존 방법들이 추적과 회복을 별개의 작업으로 처리했던 반면, Stubborn은 이를 통합하여 단일 정책으로 강인한 동작 추적과 회복을 동시에 학습하게 설계되었다.

- **Technical Details**: Stubborn은 비대칭 액터-크리틱(Asymmetric Actor-Critic) 아키텍처를 사용하고, 세 가지 주요 구성 요소로 이루어져 있다. 첫째, 항법 안정성을 개선하기 위한 요 정렬 추적 표현(yaw-aligned tracking representation)이 도입되었고, 둘째, Bernoulli 기반의 확률적 종료 메커니즘이 적용되어 다양한 실패 상황에서의 회복 행동 탐색을 장려한다. 셋째, 추적 오류 기반의 샘플링 전략이 동적으로 샘플링 분포를 조정하여 학습의 효율성을 높인다.

- **Performance Highlights**: Stubborn은 최신의 방법들(SOTA)과의 광범위한 비교 및 제거 연구(ablation studies)에서 경쟁력 있는 성능을 발휘하였다. 특히, 제안된 확률적 종료 메커니즘과 적응형 샘플링 전략이 성능 및 강인성에 크게 기여한 것으로 나타났다. 실제 로봇 실험에서도 Stubborn의 유효성이 입증되었다.



### SymQNet: Amortized Acquisition for Low-Latency Adaptive Hamiltonian Learning (https://arxiv.org/abs/2606.12808)
- **What's New**: 이 논문에서는 SymQNet을 소개하며, 이는 저지연(low-latency) 적응형 해밀토니안 학습을 위한 상환 강화 학습(amortized reinforcement learning) 접근법입니다. 기존의 Bayesian 설계 규칙을 오프라인에서 학습한 후, 온라인에서는 빠른 정책 전달을 통해 Bayesian 피드백을 유지합니다. SymQNet은 5큐비트와 12큐비트의 데이터에서 기존의 방안들에 비해 측정 지연을 크게 줄이는 성과를 보여줍니다.

- **Technical Details**: SymQNet은 1차원 Transverse-Field Ising Model (TFIM) 해밀토니안 학습에 적용되며, 상태(state), 역사(history), 그래프(context) 정보를 기반으로 다음 큐비트와 측정 기초를 선택합니다. 훈련 과정에서는 Proximal Policy Optimization (PPO) 기법을 사용하여 정책을 최적화합니다. 또한, Bayesian belief update를 위해 Sequential Monte Carlo (SMC) 방법을 적용합니다.

- **Performance Highlights**: 실험 결과, SymQNet은 5큐비트에서 의사결정 시간(decision latency)을 기존 방법들에 비해 각각 47.1배, 72.6배 감소시켰습니다. 12큐비트의 경우 SymQNet이 1.02초 소요되고, 기존 방법은 13.27초가 걸리는 것으로 나타났습니다. 이러한 성과는 기존의 온라인 최적화 방식을 대체할 수 있는 가능성을 보여줍니다.



### Exploring How Agent Voice Accents Shape Human-AI Collaboration in K-12 Group Learning (https://arxiv.org/abs/2606.12805)
- **What's New**: 이번 연구는 생성적 AI(GenAI)가 그룹 내 협업을 지원하는 새로운 가능성을 제시합니다. 특히, 다양한 억양(Accent)을 가진 대화형 에이전트가 교사와의 상호작용에서 어떻게 역할을 형성하는지를 중점적으로 다룹니다. 본 연구는 33명의 교사를 대상으로 한 실험을 통해, 억양이 협업 및 에이전트에 대한 인식에 미치는 영향을 탐구했습니다.

- **Technical Details**: 연구에 사용된 시스템은 음성 기반의 비구체적 대화형 에이전트로, 실시간 그룹 토론에 참여하여 대화를 지원합니다. 이 에이전트는 권위적인 교사가 아닌 동등한 동료로서 지식 공동 구성, 감정 조절, 그리고 공동 책임을 지원하도록 설계되었습니다. 다층 아키텍처를 통해 실시간 대화 생산을 가능하게 하였으며, Google Speech-to-Text, Microsoft Azure TTS, OpenAI의 GPT-4.1-mini 모델을 활용했습니다.

- **Performance Highlights**: 연구 결과에 따르면, British 억양의 에이전트는 도구로서 대우되며 비인격적이고 효용 중심의 방식으로 상호작용된 반면, Indian 및 African American 억양의 에이전트는 더 인격적으로 고려되고 동료로 통합되는 경향이 있었습니다. 이러한 역할 기대는 시간에 따라 신뢰, 참여도 및 의존도에 영향을 미쳤습니다. 이 연구는 AI 지원 협업에서의 사회언어학적 디자인 특성이 그룹 동역학에 미치는 영향을 이해하는 데 기여합니다.



### Agentic MPC for Semantic Control System Resynthesis (https://arxiv.org/abs/2606.12774)
Comments:
          7 pages, 5 figures

- **What's New**: 이 논문은 MPC(모델 예측 제어)가 높은 수준의 컨텍스트 정보와 동적으로 결합할 수 없는 한계를 극복하기 위해 agentic MPC 프레임워크를 소개합니다. 이 프레임워크는 대형 언어 모델 기반의 에이전트와 통합하여, 자연어 메시지와 환경 관찰을 포함한 이질적인 입력을 해석하고 제어 사양을 재합성할 수 있게 합니다. 특히, 자율주행 시나리오에서 개인의 선호에 맞춰 조정되고 사회적 상황에 반응할 수 있는 시스템의 효과성이 입증됩니다.

- **Technical Details**: Agentic MPC는 MPC 컨트롤러와 함께 작동하여 시스템의 컨텍스트를 지속적으로 모니터링하는 에이전트를 활용합니다. 에이전트는 사용자 입력, 외부 관찰 및 교통 규칙과 같은 외부 지식 소스를 포함한 이질적인 정보 소스를 수신합니다. 이를 기반으로 에이전트는 현재 상황을 해석하고 제어 사양을 수정할 필요가 있는지를 결정합니다. 또한 에이전트는 다양한 정보 채널과 연결할 수 있는 도구를 사용하여 MPC 컨트롤러의 사양을 재구성하는 역할을 합니다.

- **Performance Highlights**: 이 연구는 CARLA 시뮬레이션을 활용하여 자율주행 시나리오에서 agentic MPC 프레임워크의 성능을 검증하였습니다. 에이전트가 사용자 입력을 해석하고 외부 지식을 활용하여 문맥에 적합한 결정을 내리는 능력은 자율주행 시스템의 안전성과 효율성을 높이는 데 기여합니다. 특히, 에이전트가 공공도로에서의 긴급차량 양보와 같은 사회적 상호작용을 적절히 처리할 수 있는 능력이 강조됩니다.



### LLMs Can Better Capture Human Judgments--With the Right Prompts (https://arxiv.org/abs/2606.12754)
- **What's New**: 본 연구는 대형 언어 모델(LLM)이 인간의 판단을 잘 포착하지 못하는 이유를 다루고 있습니다. LLM의 한계로는 응답의 완전한 분포를 포착하지 못하고, 단어 변형에 따라 판단이 불안정하다는 점이 지적됩니다. 그러나 간단한 프롬프트 전략(prompting strategies)을 통해 이러한 문제를 완화할 수 있음을 보여줍니다.

- **Technical Details**: 연구는 미국 대표 샘플인 144개의 도덕적 시나리오(moral scenarios)와 32개국을 포함한 국제 사회 조사 프로그램(International Social Survey Programme)의 가족 및 성 역할에 대한 38개 도덕적 신념(moral beliefs) 데이터셋을 통해 수행되었습니다. LLM에게 표준편차(standard deviations)와 응답 비율(response proportions)을 보고하도록 요청하면, 일반적인 전략보다 인간 응답의 전체 범위를 더 잘 회복합니다. 또한, 시나리오가 인간 참가자에게 명확하도록 하는 것이 모델 정렬(model alignment)을 향상시키는 데 도움이 됩니다.

- **Performance Highlights**: LLMs는 인간의 혼란도(confusion ratings)를 잘 추적할 수 있으며, 혼란도가 높은 상황에서 모델 정렬이 증진됩니다. 그러나 LLM은 자신의 오류를 잘못 보정하는 경향이 있으며, 인간의 변동성을 예측하는 능력은 상대적으로 우수합니다. 이는 LLM에 더 나은 질문을 제시할수록 더 나은 답변을 얻을 수 있음을 시사합니다.



### PI-Hunter: Automated Red-Teaming for Exposing and Localizing Prompt Injections (https://arxiv.org/abs/2606.12737)
- **What's New**: 이번 연구에서는 PI-Hunter라는 자동화된 감사 프레임워크를 제안하여 LLM 에이전트에서 숨겨진 취약성을 사전에 노출시키는 것을 목표로 합니다. 이는 기존의 공격 최적화에서 벗어나 잠재적인 악의적인 명령을 유도하는 테스트 케이스를 구성하여 공표하도록 합니다. 또한, PI-Hunter는 적용된 방어 메커니즘 하에서도 효과적으로 작동함을 보여줍니다.

- **Technical Details**: PI-Hunter는 에이전트의 도구, 검색 인터페이스 및 외부 상호작용 채널을 분석하여, 현실적이고 출처 인식을 기반으로 한 테스트 케이스를 자동 생성하며 이를 피드백 기반 탐색을 통해 진화시킵니다. 이 과정에서 PI-Hunter는 악의적인 명령이 환경 내에 어떻게 내재되어 있는지를 드러내기 위해 에이전트를 유도합니다. 이렇게 생성된 실행 경로는 시스템적으로 감사되어 취약한 상호작용 패턴을 식별합니다.

- **Performance Highlights**: 실험 결과, PI-Hunter는 AgentDojo 및 AgentDyn와 같은 다양한 에이전트 환경에서 숨겨진 프롬프트 주입 취약성을 효과적으로 노출시키며 기초 방법론들과 비교해 더 다양한 취약한 섭취 경로를 발견했습니다. 특히 기존의 방어 수단이 있어도 여전히 효과를 발휘하여, 많은 잠재적인 주입이 현재의 완화 전략을 회피하는 것을 증명합니다.



### AfriSUD: A Dependency Treebank Collection for Evaluating Models on African Languages (https://arxiv.org/abs/2606.12708)
- **What's New**: 본 논문에서는 아프리카 언어에 대한 NLP 지원을 위한 첫 번째 대규모 구문 주석 트리뱅크 집합인 AfriSUD를 소개합니다. 이 데이터셋은 아프리카 전역에서 언어적 다양성을 반영하는 9개의 언어에 대해 정확성이 검증된 데이터를 제공합니다. 특히, 표면 구문 유니버설 의존성(Surface-Syntactic Universal Dependencies, SUD) 프레임워크를 기반으로 하여 고유한 언어적 특징들을 포착하고 있습니다.

- **Technical Details**: AfriSUD 데이터셋은 Niger-Congo, Afroasiatic 및 영어 기반 크리올을 포함한 아프리카의 다양한 언어들을 아우릅니다. 각각의 언어는 서로 다른 형태통사적(morphosyntactic) 특성을 가지며, 예를 들어 언어 중 일부는 고립어로 주로 어순과 입자를 사용하고, 다른 언어는 응집어(agglutinative)로 복잡한 동사 형태를 지니고 있습니다. 이로 인해 구문 정보의 분포와 의존 구문 분석에서 도전 과제가 발생합니다.

- **Performance Highlights**: 모델 평가 결과, AfriSUD 데이터셋에서 구문 분석과 품사 태깅에 있어 명확한 구조적 격차가 드러났습니다. 고성능 모델임에도 불구하고, 아프리카 언어 고유의 관계를 충분히 포착하지 못하고 있으며, 특정 언어적 관계에 대해 더 나은 성능을 보여주지 못합니다. 모든 트리뱅크와 주석 지침은 출판 후 공개될 예정입니다.



### SMSR: Certified Defence Against Runtime Memory Poisoning in Persistent LLM Agent Systems (https://arxiv.org/abs/2606.12703)
- **What's New**: Retrieval-augmented generation (RAG) 에이전트는 사용자 세션 간에 누적되는 지속적인 메모리(persistent memory)를 사용하여 수행된다. 그러나 이로 인해 새로운 공격 경로가 생긴다. 바로 다중 세션 메모리 오염(Multi-Session Memory Poisoning, MSMP)으로, 정상적인 채널을 통해 상호작용하는 공격자가 조작된 메모리를 주입함으로써 향후 사용자에 대한 에이전트의 응답을 조작할 수 있다. 기존 방어책은 이러한 공격을 방어하지 못한다.

- **Technical Details**: 본 논문에서는 SMSR(Signed Memory with Smoothed Retrieval)을 제안하여 MSMP 설정에 대한 첫 번째 인증된 강건성 경계(certificate of robustness bound)를 제공한다. SMSR은 HMAC-SHA256을 사용하여 기록 시 메모리의 진위를 추적하고, 쿼리 시 무작위 메모리 제거(randomized memory ablation) 및 판결 기반 다수결(vote) 방법을 통해 인증된 공격자의 영향을 제한한다. 우리는 어떤 진위 없는 필터도 적응형 주입(attacker injection)에 대해 인증할 수 없음을 입증하였다.

- **Performance Highlights**: SMSR의 성능은 15개 기업 시나리오에서 평가되었으며, Component 1은 모든 비인증 변종에 대해 공격 성공률을 93-100%에서 0%로 낮추었다. 인증된 적이 있는 공격자의 경우, Component 2는 성공률을 8.0%로 낮추었다. 에이전트 자체가 독소를 생성하는 쿼리 전용 공격에 대해 SMSR은 성공률을 65.3%에서 5.3%로 줄였다. Component 1의 클린 쿼리 유틸리티는 90%로 측정되었다.



### LLM-Powered Personalized Glycemic Assessment in Type 2 Diabetes with Wearable Sensor Data (https://arxiv.org/abs/2606.12699)
Comments:
          The 14th IEEE International Conference on Healthcare Informatics, 2026

- **What's New**: 이번 논문은 유형 2 당뇨병(T2D) 관리에 있어 개인화된 혈당 평가를 위해 LLM(large language model)을 활용하는 GlyLLM이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 CGM(continuous glucose monitor) 데이터와 개인화된 메타데이터를 통합하여 보다 효과적인 혈당 동역학 모델링을 지원합니다. 기존의 전통적인 기계 학습 방법의 한계를 극복하고, 데이터 기반의 심층적인 개인화된 관리 솔루션을 제공하는 데 중점을 두고 있습니다.

- **Technical Details**: GlyLLM은 개인화된 정적 메타데이터와 웨어러블 센서 데이터를 통합하여 LLM의 전이 학습 성능을 최대한 활용하는 구조로 설계되었습니다. 이는 비전 트랜스포머(vision transformer) 인코더를 사용해 웨어러블 데이터의 시퀀스를 생성하고, 이를 정적 메타데이터와 함께 결합하여 처리하는 방식을 채택합니다. 이를 통해 혈당 예측과 당뇨병 분류라는 두 가지 임상 과제를 수행함으로써 모델의 효용성을 평가합니다.

- **Performance Highlights**: 실험 결과, GlyLLM은 혈당 예측에서 평균 13.66%의 RMSE(Root Mean Squared Error) 감소를 확인했으며, 당뇨병 분류에서는 13.08%의 AUROC(Area Under the Receiver Operating Characteristic) 향상을 보여주었습니다. 이러한 결과는 기존의 기계 학습 방법들보다 뛰어난 성능을 입증하며, 개인화된 정적 메타데이터와 센서 데이터의 중요성을 강조합니다. 또한, 보충 연구를 통해 당뇨병 설문조사와 생체 테스트의 데이터가 혈당 평가에 있어 더 중요한 요소임을 확인했습니다.



### Two-Layer Linear Auto-Regressive Models Estimate Latent States (https://arxiv.org/abs/2606.12691)
Comments:
          ICML 2026

- **What's New**: 이 논문은 자기 회귀 모델(auto-regressive models)이 부분 관측된 선형 동적 시스템(partially observed linear dynamical systems)에서 데이터를 기반으로 학습할 때 칼만 필터(Kalman filtering)를 근사하는 방법을 보여줍니다. 특히, 이 모델이 내재된 동역학이나 상태에 대한 명시적 지식이 없는데도 최적 필터(Kalman filter)와 일치하는 숨겨진 표현(hidden representation)을 학습하게 됩니다.

- **Technical Details**: 연구의 핵심 통찰은 세 가지입니다. 첫째, 칼만 필터는 한계 절단 오류(bounded truncation error)를 가진 자기 회귀 모델로 잘 근사될 수 있다는 점입니다. 둘째, 비볼록성(non-convexity)에도 불구하고 두 단계 최적화 경관(optimization landscape)은 온건하다는 것입니다. 마지막으로 본 논문의 주요 기여는 예측 오류(prediction error), 매개변수 추정 오류(parameter estimation error), 숨겨진 상태 복구(latent state recovery)에 대한 유한 샘플 보장(finite-sample guarantees)을 제공하는 것입니다.

- **Performance Highlights**: 수치 시뮬레이션(numerical simulations)은 이론적 결과를 뒷받침하며, 자기 회귀 모델의 숨겨진 표현이 상태 추정(state estimates)을 잘 복원함을 보여줍니다. 이 연구는 시퀀스 데이터 처리에서 자기 회귀 모델의 유용성을 강조하며, 관련 이론적 토대를 확장합니다.



### EWAM: An Enhanced World Action Model for Closed-Loop Online Adaptation in Embodied Intelligenc (https://arxiv.org/abs/2606.12690)
- **What's New**: 본 논문에서는 Enhanced World Action Model (EWAM)라는 폐쇄 루프 온라인 적응 아키텍처를 제안합니다. EWAM은 사전훈련된 Cosmos3 네트워크를 기반으로 하며, 추가적인 배치 데이터 없이 새로운 작업 구성에 적응할 수 있도록 고안되었습니다. 평가 과정에서 추가적인 작업 특정 시연 세트가 도입되지 않았고, 백본 네트워크의 미세 조정도 없었습니다.

- **Technical Details**: EWAM은 경험을 향상시키기 위해 네 개의 경량 신경 레이어를 삽입합니다: Neural Experience Memory Layer는 Diffusion Transformer (DiT)의 중간 레이어에 위치하여 작업 관련 실행 컨텍스트를 제공합니다. Neural Anomaly Detection Layer는 상태 예측 헤드 뒤에 위치하여 예측된 상태와 실제 상태 간의 차이를 실시간으로 모니터링합니다. 그 후 Neural Policy Routing Layer가 동적으로 실행 방식(직접 실행, 보수적 재계획, 또는 롤백 복구)을 선택하며, Neural Action Correction Layer는 실행 진단을 통해 생성된 액션 청크를 정제합니다.

- **Performance Highlights**: EWAM은 RoboLab의 제로샷 조작 작업에서 우수한 실행 품질 향상을 보여주었습니다. 유도된 경량 신경 레이어의 적용을 통해 기존의 Cosmos3-Nano–Policy-DROID 백본은 동결 유지하면서도 적응할 수 있는 능력을 제공합니다. 이 메소드는 예측-실제 불일치, 충돌, 빈 그랩, 인식 환각, 그리고 경로 중복과 같은 실행 문제를 해결하는 것을 목표로 하고 있습니다.



### M*: A Modular, Extensible, Serving System for Multimodal Models (https://arxiv.org/abs/2606.12688)
- **What's New**: 이번 연구에서는 다양한 구성 요소들을 통합한 복합 모델 아키텍처의 새로운 시대에 진입하고 있음을 보여줍니다. 이러한 멀티모달 모델은 비전 인코더(vision encoders), 언어 백본(language backbones), 확산 및 흐름 헤드(diffusion and flow heads), 오디오 코덱(audio codecs), 행동 생성기(action generators) 및 세계 모델 예측기(world-model predictors)를 포함합니다. 특히, 현재의 모델 제공 프레임워크는 이러한 새로운 아키텍처 다양성을 수용하기 충분하지 않음을 지적하며, 이를 해결하기 위한 새로운 보편적 제공 시스템인 M*를 제안합니다.

- **Technical Details**: M*는 복합 AI 모델의 효율적인 제공을 위한 보편적 시스템으로, 모델을 데이터 흐름 그래프(dataflow graph)로 표현합니다. 요구 사항에 따라 다양한 모달리티와 작업을 처리하는 요청을 이 그래프를 통해 탐색합니다. 이 시스템의 core 인사이트는 모듈화된 추상화(modular abstraction)로, 모델 컴포넌트의 임의 조합 및 물리적 클러스터에의 유연한 배치를 지원하며, 배포된 런타임 내에서 모델 무관한 최적화를 가능하게 합니다.

- **Performance Highlights**: M*는 BAGEL의 텍스트-이미지 워크로드에서 vLLM-Omni에 비해 평균적으로 20% 낮은 전체 지연 시간을 달성하였고, Qwen3-Omni의 텍스트-음성 워크로드에서는 최대 2.9배 낮은 실시간 계수(real-time factor)와 2.7배 높은 처리량을 기록했습니다. 또한, 로봇 계획을 위한 V-JEPA 2-AC 롤아웃 베이스라인보다 최대 12.5배 높은 성능을 나타났습니다. 이로 인해 복잡한 모델의 효율적인 제공을 위한 기반이 마련되었습니다.



### A Zero-shot Generalized Graph Anomaly Detection Framework via Node Reconstruction (https://arxiv.org/abs/2606.12673)
- **What's New**: 본 논문에서는 이질적인 그래프에서 제로샷(zero-shot) 그래프 이상 탐지(Graph Anomaly Detection, GAD) 프레임워크인 AlignGAD를 제안합니다. 기존의 GAD 방법들이 특정 데이터셋에 의존하는 반면, AlignGAD는 다양한 도메인에서도 성능을 발휘할 수 있는 일반화된 방법론을 제공합니다. 이 프레임워크는 글로벌 통합 모듈(Global Unification Module), 클러스터 모듈(Clustering Module), 그리고 노드 불일치 점수 모듈(Node Discrepancy Scoring Module)로 구성되어 있습니다.

- **Technical Details**: AlignGAD는 첫째, 다양한 노드 특성을 정렬하고 스펙트럼 도메인에서 그래프 신호를 정규화하는 글로벌 통합 모듈을 사용합니다. 둘째, 클러스터 모듈을 통해 집단 수준의 비정상 패턴을 포착할 수 있는 클러스터 인식 그래프 뷰를 구축합니다. 마지막으로, 노드 불일치 점수 모듈은 입력과 생성된 표현 간의 불일치를 측정하고, 여러 그래프 뷰에서 수집된 이상 징후 증거를 집계합니다.

- **Performance Highlights**: 다양한 실제 그래프 데이터셋에서 수행된 실험은 AlignGAD의 제로샷(GAD) 설정 하에서의 효과성을 입증합니다. 결과적으로, 본 프레임워크는 이질적인 그래프에서의 이상 탐지 문제를 실용적이고 일반화된 방식으로 해결할 수 있는 가능성을 보여줍니다. AlignGAD는 데이터셋 간의 세부사항에 대한 의존성을 줄여주어 보다 넓은 범위의 그래프에 적용할 수 있습니다.



### Free-Placement Optimization of Ground Station Locations for Low-Earth Orbit Satellites (https://arxiv.org/abs/2606.12667)
Comments:
          34 pages, 13 figures, 11 tables, Journal of Aerospace Information Systems (JAIS)

- **What's New**: 최근의 저지구 궤도(LEO) 위성 별자리의 급증은 지상 네트워크의 수요를 증가시키고 있으며, 이를 해결하기 위한 더욱 효율적인 지상 기지국 네트워크 설계가 필요해지고 있습니다. 본 연구에서는 SCORE(Sequential Cyclic Optimization via Refinement & Evaluation)라는 두 단계의 자유 배치 최적화 방법을 도입하여 기지국 설계 문제를 해결합니다. SCORE는 고차원적 및 비볼록성 문제를 관리하기 위해 순차 좌표 선택과 순환 정제를 결합한 혁신적인 접근 방식을 제공합니다.

- **Technical Details**: 이 논문은 지구 관측(EO) 위성 별자리의 기지국 위치 선택 문제를 다루고 있으며, 두 가지 변형인 자유 배치(free placement)와 고정 사이트(fixed-site selection)를 구별합니다. 자유 배치 문제는 후보 위치가 경도와 위도에서 연속적으로 변할 수 있는 반면, 고정 사이트 선택 문제는 미리 정의된 위치 집합으로 선택이 제한되는 방식입니다. SCORE는 높은 차원에서의 복잡성을 해결하고 지리적 최적화를 통해 성능을 극대화합니다.

- **Performance Highlights**: SCORE는 기존의 방법들에 비해 성능 평가를 위한 함수 평가를 최대 5배 줄이며, 다운링크 처리량을 최대 13% 향상시킵니다. 고정사이트 방식과 비교했을 때, SCORE는 최대 15% 더 높은 전체 다운링크 성능을 달성하며, 기존 인프라 근처에 배치를 제한하는 인프라 제약 SCORE는 이의 92% 이상의 이점을 유지합니다. 이는 유연한 기지국 배치의 강력한 성능 벤치마크를 설정하며, 향후 운영 별자리를 위한 지상 네트워크 설계를 위한 정보도 제공합니다.



### CAPED: Context-Aware Privacy Exposure Defense for Mobile GUI Agents (https://arxiv.org/abs/2606.12666)
- **What's New**: CAPED는 mobile GUI agent를 위한 context-aware 프리업로드 노출 제어 계층이다. 이 시스템은 스크린샷이 원격 멀티모달 에이전트에 업로드되기 전에 작업 요구 사항을 추출하고, 스크린 컨텍스트를 개인정보 보호의 우선사항으로 사용하며, 가시 UI 요소를 분석하여 현재 작업에 필요한 내용만 노출한다.

- **Technical Details**: CAPED는 원시 모바일 화면과 원격 GUI 에이전트 사이에 위치하며, 본질적인 작업, 원시 스크린샷 및 노출 결정을 시스템의 신뢰 영역 내에 유지한다. 시스템은 로컬 결정 파이프라인을 통해 작업에 필요한 정보를 추출하고, UI 요소 분석을 통해 작업과 관련된 콘텐츠만 안전하게 노출한다.

- **Performance Highlights**: 평가 결과, CAPED는 태스크 완료 시 초기 스크린샷의 성공 조건을 가진 누적 노출을 0.766에서 0.268로 감소시켰고, 28개 태스크 평가에서 높은 태스크 유용성을 유지하였다. AndroidWorld에서의 테스트에서도 기능적으로 일부 유틸리티 비용은 있지만, 스크린샷 업로드를 태스크 기반 선택적 노출로 처리해야 한다는 주장을 뒷받침한다.



### BASENet: Band-Adapted Speech Enhancement Network with Cross-Band Attention (https://arxiv.org/abs/2606.12662)
- **What's New**: BASENet은 주파수에 적합한 구조로, Bark 스케일 대역으로 스펙트럼을 분할하고 각 대역에 비율에 따라 인코더 용량을 할당합니다. 이는 인간의 청각 해상도를 반영하여 저주파 대역에 더 깊은 인코더 분기를 제공하고 고주파 대역은 더 가벼운 인코더로 처리합니다. BASENet은 다양한 최신 음성 향상 모델들을 개선하여 PESQ(Perceptual Evaluation of Speech Quality) 3.55를 달성하며, 실시간 처리가 가능하도록 설계되었습니다.

- **Technical Details**: BASENet 구조는 Short-Time Fourier Transform(STFT)을 통해 잡음이 섞인 음성 신호를 처리합니다. 입력 스펙트로그램은 Bark 스케일에 따라 비대칭 주파수 대역으로 나누어지고, 각 대역에 대해 비선형 크기 변환이 적용됩니다. 또한 Cross-Band Attention을 활용하여 대역 간 정보 교환이 이루어지며, 복잡한 연결 구조의 모바일 네트워크 블록을 채택하여 계산 비용을 최소화합니다.

- **Performance Highlights**: BASENet은 단 0.83M 매개변수와 7.3G MACs로 최고 성능을 자랑하며, PESQ 점수 3.50 이상의 모든 1M 미만 매개변수 방법 중에서 가장 높은 성능을 기록합니다. 또한, BASENet의 인과형 변형은 비 인과적 기준선 여러 개를 초월하여 실시간 스트리밍에 최적의 적합성을 입증했습니다.



### Token Complexity Theory for AI-Augmented Computing (https://arxiv.org/abs/2606.12647)
Comments:
          25 pages, 1 figure

- **What's New**: AI-augmented computing는 자연어 질의, 코드 생성 요청 및 기타 개방형 작업을 처리하기 위해 AI 모델 클러스터에 요청을 위임합니다. 이 새로운 패러다임은 전통적인 시간 또는 공간 복잡도와는 다른 자원 차원을 도입하며, 이를 해결하기 위해 새로운 개념인 token complexity를 소개합니다. 이 연구에서는 AI 시스템을 probabilistic properties에 따라 분류하는 분류 체계를 개발하고, AI-Oracle Turing Machine 프레임워크 내에서 token complexity를 발전시킵니다.

- **Technical Details**: AI-Oracle Turing Machine(AOTM)은 다중 테이프 probabilistic Turing machine을 기반으로 하여 전용 oracle 인터페이스를 추가한 모델입니다. 이 모델은 질의 및 응답 테이프를 통해 stochastic oracle과 상호작용하는 구조를 가집니다. 이 연구에서는 token cost의 개념을 도입하여, 알고리즘과 oracle 간의 경계에서 발생하는 자원 비용을 측정하고 분석합니다.

- **Performance Highlights**: 연구 결과에 따르면 token complexity는 질에 대한 단조성, 볼록성, 가격 민감도를 보여주며, 질의에 대한 응답의 비용 비율에 따라 작업 순서가 달라질 수 있습니다. 이와 같은 결과를 통해 AI-augmented computing의 자원 요구 사항을 이해하는 것이 이론적으로나 실용적으로 필수적임을 강조합니다. 최종적으로, 이 연구는 token complexity를 통한 AI 작업의 효율성을 개선할 수 있는 방법론을 제시합니다.



### Keep Policy Gradient in Charge: Sibling-Guided Credit Distillation for Long-Horizon Tool-Use Agents (https://arxiv.org/abs/2606.12634)
Comments:
          13 pages, 4 figures, 7 tables. Submitted to EMNLP 2026 Industry Track

- **What's New**: 이 논문은 Sibling-Guided Credit Distillation(SGCD)이라는 새로운 접근법을 소개합니다. 기존의 훈련 방법인 self-distillation(SD)이 도구 사용 능력을 약화시킬 수 있음을 증명했습니다. SGCD는 실패와 성공한 시도에서 유용한 정보를 활용하여 기울기를 재조정하는 대신 기여를 할당하는 신호를 제공합니다.

- **Technical Details**: SGCD는 동적 샘플링을 통해 동일한 작업을 수행하는 실패한 시도와 성공한 시도를 혼합하여 그룹화합니다. 그 결과로 생성된 시뮬레이션 롤아웃은 외부의 강력한 LLM에 의해 요약되어 단계적으로 기여를 참조하게 됩니다. 이렇게 생성된 기여 신호는 정책 경량주를 업데이트하는 데 사용됩니다.

- **Performance Highlights**: SGCD는 AppWorld와 τ3-항공사 벤치마크에서 GRPO 비교군보다 더 나은 성능을 보여줍니다. AppWorld에서는 테스트 정상에서 42.9에서 45.6으로 향상되었고, τ3-항공사에서는 pass@1이 0.583에서 0.602로 증가했습니다. 이러한 결과들은 SGCD가 혁신적인 접근법임을 입증합니다.



### Bag of Dims: Training-Free Mechanistic Interpretability via Dimension-Level Sign Patterns (https://arxiv.org/abs/2606.12629)
Comments:
          14 pages, 4 figures, 10 tables

- **What's New**: 이 논문은 Transformer의 hidden states의 표준 기본이 트레이닝 없이도 아키텍처 일반적으로 사용 가능한 피처 기반을 제공함을 보여줍니다. 각 차원은 부호(sign)를 통해 의미론적 콘텐츠를, 크기(magnitude)를 통해 신뢰도를 인코딩합니다. 이러한 'Bag of Dims' 프레임워크는 세 가지 모델 패밀리를 통해 검증되었으며, 신호 패턴만 사용하여도 높은 정확도를 달성할 수 있음을 확인하였습니다.

- **Technical Details**: Transformer의 hidden state를 독립적인 이진 레지스터의 집합으로 간주하고, 각 차원이 의미 콘텐츠와 신뢰도를 인코딩합니다. 175개의 의미론적 카테고리를 발견하기 위해 50 개의 앵커 토큰을 사용하여 각 차원의 부호 일관성에 기반한 AUC를 측정하였습니다. 모델의 임베딩을 분석하는 과정에서, 레이어마다 카테고리 분리가 극대화되는 위치를 찾았고, 이는 각 차원이 독립적으로 작동함을 입증합니다.

- **Performance Highlights**: 이런 결과들은 표준 기초가 Transformer 계산 경로 전반에 걸쳐 피처 읽기에 충분하다는 것을 입증합니다. 간단한 표지자 사용으로도 80-90%의 높은 다음 토큰 정확도를 달성할 수 있으며, 상호 정보량은 낮은 차원 간 결합을 나타냅니다. 모든 실험을 통해, 모델은 훈련 없이도 높은 성능으로 의미론적 카테고리를 분리하고 인식할 수 있음을 보여주었습니다.



### HybridCodeAuthorship: A Benchmark Dataset for Line-Level Code Authorship Detection (https://arxiv.org/abs/2606.12620)
Comments:
          Accepted to LREC 2026

- **What's New**: 최근 AI 코드 어시스턴트(AI code assistants)의 급속한 채택으로 인해 산업 코드베이스가 점점 더 AI가 생성한 코드와 인간이 작성한 코드가 혼합된 형태로 발전하고 있습니다. 이 논문에서는 이러한 코드의 위치를 세밀하게 탐지할 수 있는 알고리즘 개발을 위해 HybridCodeAuthorship라는 새로운 벤치마크를 소개합니다. 이 벤치마크는 인간과 AI가 작성한 코드 라인이 혼합된 파이썬 코드 파일들로 구성되어 있으며, 이를 통해 AI 코드 어시스턴트의 실제 사용 방법을 시뮬레이션할 수 있습니다.

- **Technical Details**: HybridCodeAuthorship의 데이터셋 구성은 CodeSearchNet을 활용하여 수행되었습니다. 이 데이터셋은 AI와 인간이 작성한 코드가 교차된 형태로 구성되어 있으며, 각 코드 파일은 유닛 테스트에서의 성공 여부가 레이블링 되어 있습니다. 코드 테스트와 코드 인터리빙(code interleaving) 두 가지 단계로 나누어져 있으며, 후자는 인간이 작성한 코드 파일의 특정 라인을 LLM으로 마스킹하고 주석을 추가하여 생성된 AI 코드 파일로 이어집니다.

- **Performance Highlights**: 논문은 두 가지 최첨단 AI 코드 탐지 알고리즘을 활용해 HybridCodeAuthorship의 성능을 시험하였으며, 그 결과 AIGCode Detector는 각각 0.48과 0.56의 F1 점수를 기록하여 매우 도전적인 벤치마크임을 보여주었습니다. 이러한 결과는 실제 세계의 AI 코드 어시스턴트 채택을 반영하며, 향후 코드 탐지 알고리즘의 발전을 촉진할 것으로 기대됩니다.



### From Imitation to Alignment: Human-Preference Flow Policies for Long-Horizon Sidewalk Navigation (https://arxiv.org/abs/2606.12603)
- **What's New**: 자동화된 장거리 보도 탐색은 로봇 음식 배달이나 조향 보조 전자 휠체어와 같은 마이크로 모빌리티 응용 프로그램에 필수적입니다. 본 논문에서는 FlowPilot이라는 맵 없이 장거리 탐색 정책을 도입하여 단일 모노큘러 RGB 카메라만으로 다양한 상황에서 로봇이 안전하게 보도를 탐색하도록 합니다. 이는 기존의 모방 학습(imitation learning, IL)의 한계를 극복하기 위해 고안된 새로운 접근 방식입니다.

- **Technical Details**: FlowPilot은 고정된 흐름 일치를(action representation) 사용하는 정책 사전 학습(pre-training) 방법을 제안하며, 이는 복잡한 다중 모드 행동(distribution of sidewalk navigation behaviors)을 포착합니다. 이 방법은 대규모 로봇 데이터에서 다변량 행동을 모델링하고, 사회적 규범(social compliance)과 필요한 조작적 일관성(robustness)을 강화하기 위해 인간의 피드백을 포함한 선호 학습(preference learning) 알고리즘을 포함합니다.

- **Performance Highlights**: 전체 시뮬레이션 성능에서 FlowPilot은 42%의 성공률과 66%의 경로 완료율(route completion)로 평가되었습니다. FlowPilot-HP는 실제 환경에서의 강건성과 사회적 준수를 더욱 향상시켜, 기본 모델 대비 40.0%의 IR 및 52.1%의 NIR 감소를 달성하는 성과를 보였습니다.



### Emerging Flexible Designs for Geospatial Multimodal Foundation Models (https://arxiv.org/abs/2606.12595)
- **What's New**: 이 논문은 지리공간 멀티모달 추론을 위해 설계된 여러 최신 Foundation Model의 성능을 체계적으로 비교한 연구입니다. 이 연구는 유사한 조건에서 사전 훈련된 모델을 평가하여 각 구조의 건축적 강점과 제한점을 조명합니다. 또한, 실험 결과는 모델의 유연성, 모달리티 정렬, 그리고 다운스트림 작업 성능 간의 설계를 위한 새로운 통찰력을 제공합니다.

- **Technical Details**: 핵심적으로, 연구에서는 DOFA, SatMAE, Flex라는 세 가지 주요 아키텍처를 비교합니다. 이들 모델은 모두 동일한 Sentinel-2 데이터셋을 사용하여 방해 요소 없이 모델의 기여도를 비교하였습니다. 이는 GeoBench 벤치마크를 기반으로 하며, 분류 및 세그멘테이션 작업을 포함하여 일관된 평가 프로토콜을 통해 진행됩니다.

- **Performance Highlights**: 분석된 결과, Flex 모델은 부족한 스펙트럼 또는 이질적인 밴드에 대해 더 높은 적응성을 보였지만, 스펙트럼이 동질적인 환경에서는 상대적으로 성능이 떨어지는 경향을 보였습니다. 이研究는 스펙트럼 유연성과 일반화 간의 주요 절충점을 강조하면서, 다음 세대 지리공간 기초 모델 설계에 대한 실용적인 지침을 제공합니다.



### Analyzing and Improving Fine-grained Preference Optimization in Medical LVLMs (https://arxiv.org/abs/2606.12590)
- **What's New**: 이 논문은 대형 비전-언어 모델(LVLM)이 의료 이미징 작업에서 겪는 사실 불일치, 시각적 기반 부족, 임상적으로 중요한 피드백과의 불일치를 해결하기 위한 새로운 방법을 제안합니다. 기존의 연결 프레임워크의 한계점을 극복하기 위해, 연구자들은 토큰 수준에서 세밀한 최적화를 허용하는 새로운 'Fine-grained Regularized Medical Preference Optimization (FiRe-MPO)' 목표를 도입하였습니다.

- **Technical Details**: FiRe-MPO는 역방향 토큰별 KL 정규화를 활용하여 임상적으로 중요한 요소, 즉 해부학적 발견 및 병리적 특성을 정교하게 조정하도록 설계되었습니다. 이 방법은 모델의 원래 언어 구조를 유지하면서 임상적으로 잘못된 부분만 수정하는 '온 정책(preference pairs)' 데이터 세트를 생성하며, 시각적 근거를 요구하는 임상적 정확성을 뛰어넘는 최적화를 촉진합니다.

- **Performance Highlights**: 다양한 의료 이미징 과제 및 임상 텍스트 생성 벤치마크에서의 실험 결과, FiRe-MPO는 기존의 DPO와 RRPO보다 일관되게 우수한 성능을 보여주며, 평균 10.24%의 상대적 개선을 달성하였습니다. 이러한 결과는 임상적인 정확성을 높이고, 일반적으로 관찰되는 스타일에 편향되는 보상 해킹 행동을 효과적으로 감소시킴을 보여줍니다.



### Graph Reduction in Multirelational Networks: A Spreading-Oriented Reduction Benchmark (https://arxiv.org/abs/2606.12581)
- **What's New**: 이 논문에서는 그래프 축소(graph reduction)의 영향력을 강조하는 Spreading-Oriented Reduction Benchmark (SORB)라는 오픈 소스 프레임워크를 소개합니다. 기존의 영향력 극대화(influence maximisation, IM) 문제에 대한 접근법과는 달리, SORB는 다양한 네트워크를 평가하여 그래프 축소가 IM의 예측 성능에 미치는 영향을 정량화합니다. 이를 통해 단일 레이어와 다층 네트워크 간의 IM 성능 차이를 이해할 수 있는 기회를 제공합니다.

- **Technical Details**: 복잡한 네트워크는 상호 작용하는 행위자들 간의 다양한 관계를 그래프로 표현하여 나타내며, SORB는 이러한 네트워크에 대해 그래프 축소 이전(preprocessing)의 효과를 측정하는 구조적 접근을 제공합니다. 논문에서는 다층 네트워크(multilayer network) 모델을 도입하고, 각기 다른 관계를 포괄하는 방식으로 네트워크를 정의합니다. 여기서 그래프 축소 기술로는 희소화(sparsification)와 집합화(coarsening)가 적용되며, 이들 기술이 IM 문제에 미치는 영향이 중점적으로 분석됩니다.

- **Performance Highlights**: 연구 결과, 단일 레이어 네트워크에 대한 축소는 Gain@k의 품질을 보존하는 반면, 다층 네트워크에서는 축소 전략에 관계없이 시스템적 순위 저하가 나타났습니다. 이 발견은 복잡한 네트워크를 연구할 때 축소 인지(reduction-aware) 및 다중 작업(multi-task) 평가의 중요성을 강조합니다. 또한, 학습 기반 IM 모델(ts-net)은 희소화 선처리(preprocessing)가 낮은 차원 네트워크에서 유의미한 성과를 보이는 반면, 경량 휴리스틱(lightweight heuristics)은 다소 일관된 성과를 발휘하는 것으로 나타났습니다.



### EDEN: A Large-Scale Corpus of Clinical Notes for Italian (https://arxiv.org/abs/2606.12569)
- **What's New**: 이번 연구에서는 이탈리아 병원의 응급실에서 생성된 임상 노트의 대규모 데이터베이스인 EDEN(Emergency Department Electronic Notes)에 대해 설명하고 있습니다. 현재 버전은 약 4백만 개의 완전 익명화된 임상 노트로 구성되어 있으며, 응급실에서의 환자 치료 과정의 다양한 단계를 포괄합니다. 특히, 약 6천 개의 노트는 임상 전문가에 의해 사례 보고 양식을 통해 수작업으로 주석이 달려 있으며, 이는 응급 상황에서의 환자 상태를 기록합니다.

- **Technical Details**: EDEN 데이터는 구조화된 사례 보고 양식(Case Report Form; CRF)을 사용하여 환자 상황인 호흡 곤란(dyspnea)과 의식 상실(loss of consciousness)에 대해 생리적 및 임상적 정보를 수집하였습니다. 수집 과정에서는 병원 윤리 위원회의 승인을 받았으며, 2021년부터 2023년 사이의 환자의 자유 텍스트 임상 노트를 포함하고 있습니다. 각 병원의 전자 건강 기록(EHR) 시스템에서 추출된 데이터는 두 단계의 익명화 과정을 통해 개인 정보 보호를 철저히 하고 있습니다.

- **Performance Highlights**: EDEN 데이터셋은 이탈리아어로 된 임상 노트의 가장 큰 자유롭게 사용 가능한 데이터베이스로, 연구 개발 및 대규모 언어 모델(Large Language Models; LLM)의 실용적 응용을 지원하기 위한 자료로서 중요한 역할을 할 것으로 기대됩니다. 특히, CRF 채우기를 새로운 구조적 정보 추출 벤치마크로 제안하며, Gemma-27B와 MedGemma-27B에서 도출된 제로샷(zero-shot) 기준 결과를 제공하고 있습니다. 이 데이터셋은 다국어 및 다기관 연구 개발에 매우 적합한 리소스로, 향후 더욱 확장될 예정입니다.



### Foresight: Iterative Reasoning About Clues that Matter for Navigation (https://arxiv.org/abs/2606.12550)
Comments:
          22 pages, 10 figures, 3 tables

- **What's New**: 이 논문은 로봇의 맵 없는 내비게이션을 위한 'Foresight'라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 미리 훈련된 비전-언어 모델(Vision-Language Model, VLM)을 사용하여 환경 속 자연어 지시에 따라 계획을 제안하고 평가하며 반복적으로 개선합니다. 기존 연구들이 지식 기반의 요소에 의존했던 것과 달리, Foresight는 로봇이 실행하기 전에 환경에서의 시각적 단서를 발견하고 이를 계획 수립에 통합하도록 돕습니다.

- **Technical Details**: Foresight는 로봇이 환경에서 받은 비주얼 입력에 대한 언어 목표를 바탕으로 행동을 결정하는 방법론을 제시합니다. 이 방법은 미세 조정된 VLM이 제안하는 이미지 기반 움직임 계획과 그에 대한 비평을 순환하는 방식으로 이뤄집니다. 이러한 과정은 로봇이 주어진 목표와 장면을 바탕으로 계획을 수정하고 조정할 수 있도록 합니다. 또한 Foresight는 인간 피드백을 통해 보상 모델을 학습하여 VLM을 강화 학습하에 훈련합니다.

- **Performance Highlights**: Foresight는 오프라인 평가 및 실제 환경에서의 실험 결과를 통해 평균 과제 성공률을 37% 향상시키고 미션당 개입을 52% 줄였습니다. 이 결과는 Jetson AGX Orin에서 실시간으로 실행되는 VLM 정책에 의해 생성된 짧은 자유 형식의 추론으로부터 얻어진 것입니다. 기존의 상태 기반 모델이나 더 복잡한 추론 내역을 가진 베이스라인을 초월하며 로봇 모션 세부 조정에서의 테스트 시간 추론에 대한 새로운 기법을 보여줍니다.



### Boosting Direct Preference Optimization with Penalization (https://arxiv.org/abs/2606.12505)
Comments:
          Accepted at ICML 2026 Workshop on Decision-Making from Offline Datasets to Online Adaptation: Black-Box Optimization to Reinforcement Learning

- **What's New**: 이 논문은 Direct Preference Optimization (DPO)를 확장한 Direct Preference Optimization with Penalization (DPOP)을 제안합니다. DPOP는 선택된 응답과 거부된 응답이 포함된 정적 데이터셋에만 의존하던 기존의 DPO의 한계를 극복합니다. 이 새로운 접근 방식은 참조 모델이 생성한 응답을 활용하여 보다 효과적으로 정책을 최적화할 수 있게 합니다.

- **Technical Details**: DPOP는 기본적인 쌍둥이 선호 손실에 참조 모델로부터 생성된 응답에 대한 벌칙(penalty)을 추가하여 작동합니다. 정책이 거부된 응답보다 선택된 응답에 대해 낮은 가능성을 부여할 때만 벌칙이 적용됩니다. 이 방법은 SimNPO 스타일의 길이 정규화된 벌칙을 사용하여 보다 강력한 성능을 발휘합니다.

- **Performance Highlights**: AlpacaEval 2.0 의 실험 결과, DPOP는 Llama-3-8b-it에서 SimPO와 AlphaDPO에 비해 5.3%의 상대적 성과 향상을 보였고, Gemma-2-9b-it에서는 4.4% 향상되었습니다. 이는 참조 모델의 탐욕적 응답이 분석을 위한 데이터로서 유용할 뿐만 아니라, 오프라인 벌칙 신호로도 효과를 발휘할 수 있음을 보여줍니다.



### A Mathematical Theory of Value: a synthesis on goal-directed agency under resource constraints (https://arxiv.org/abs/2606.12502)
Comments:
          Also available at this https URL (v5)

- **What's New**: 이 논문에서는 가치(value)가 정보(information)와 같은 법칙적 구조적 양(quantity)임을 제안합니다. 에이전트가 자원(resource)을 목표의 진행(goal-progress)으로 변환하는 비율에 따라 가치를 수량화하는 새로운 방식을 도입합니다. 이 연구의 중요한 발견 중 하나는 목표와 관련된 다양한 채널에서 자원을 최적 할당하면서 가치는 최대화됨을 보여줍니다.

- **Technical Details**: 에이전트는 E>0의 분배 가능한 자원을 보유하고 있으며 이를 K개의 목표 관련 채널에 배분합니다. 가치 V(e)는 할당의 결과로서 정적(static) 및 동적(dynamic) 성질을 지니며, 로그 함수 형태를 따릅니다. 식에서 볼 수 있듯이, 자원을 재투자함으로써 잠재적인 가치의 성장은 동적 최적화 문제를 통해 설정됩니다.

- **Performance Highlights**: 논문은 언어 모델을 대상으로 한 실험을 통해, 인지 상호정보(perception mutual information)가 실현된 능력(realized capability)을 추적한다는 것을 입증했습니다. 또한, 이 연구는 잠재적인 과신(over-confidence)이 측정 가능한 소모(dissipation)로 나타난다는 결과를 제시하며, 특정 작업 형상(task shapes)에 대한 테스트 결과 또한 제시되어 있습니다. 연구의 기여는 기존 기법들이 독립적으로 존재하는 것이 아니라, 이들을 통합하고 거버넌스 매핑(governance mapping)을 통해 인센티브 디자인(incentive design)을 수립하는 것에 있습니다.



### Improving Crash Frequency Prediction from Simulated Traffic Conflicts Using Machine Learning Based Microsimulation (https://arxiv.org/abs/2606.12500)
- **What's New**: 최근의 연구에 따르면, 전통적인 교통 사고 데이터 대신 교통 마이크로 시뮬레이션과 대체 안전 척도를 결합하여 현재 또는 계획된 도로 인프라 설계를 위한 사고 발생 예측에 활용하고 있습니다. 기존의 마이크로 시뮬레이션 기반 안전 연구는 주로 간단한 규칙 기반 행동 모델을 사용해 왔는데, 이는 교통 흐름은 잘 나타내지만 현실적인 충돌 동역학을 생성하는 데 한계를 보였습니다.

- **Technical Details**: 이 연구에서는 영국 리즈에 있는 5개 신호 교차로를 대상으로 표준 규칙 기반 모델과 최첨단 기계 학습(ML) 모델을 사용하여 교통 마이크로 시뮬레이션을 수행했습니다. 시뮬레이션된 차량 궤적은 2차원 시간-충돌(Time-to-Collision, TTC) 지표를 사용하여 분석하였고, Extreme Value Theory를 통해 충돌 발생 빈도를 예측했습니다.

- **Performance Highlights**: ML 모델에서 발생한 충돌은 실제 사고 데이터와 일치하는 예측 결과를 보였으나, 규칙 기반 모델은 유의미한 예측을 하지 못했습니다. 이는 특정 교차로에 대한 모델 보정 부족이 원인으로 보이며, 현재의 ML 모델이 충돌을 현실적으로 재현할 수는 있지만 실제 사고를 생성하기에는 불완전하다는 것을 시사합니다. 전반적으로 ML 기반 행동 모델이 충돌 예측 개선에 유망하며, 위치 특정 모델 보정 없이도 가능성을 보여줍니다.



### Speculative Rollback Correction for Quality-Diverse Web Agent Imitation (https://arxiv.org/abs/2606.12485)
- **What's New**: 이 논문에서는 웹 및 GUI 에이전트를 위한 새로운 학습 프레임워크인 Speculative Rollback Correction (SRC)를 소개합니다. SRC는 교사 감독(teacher supervision)을 개별 행동에서 단기 사전(branched speculation)으로 이동시키며, 첫 변이가 발생할 때까지 학생이 단기 브랜치를 실행하게 해줍니다. 이 방법은 자가 유도 상태에서 학습하는 에이전트의 성능을 향상시키는 데 기여하며, 기존의 행동 클로닝에서 발생하는 누적 오류를 완화합니다.

- **Technical Details**: SRC 프레임워크는 에이전트가 방문한 상태에서 학습할 수 있도록 합니다. 학생은 사전 실행(segment)에서 브랜치를 탐색하고, 교사는 그 브랜치가 유용한 진행 상태(local progress)를 유지하는지를 판단합니다. 유해한 변이가 발견되면 유용한 접두사는 보존되며, 연쇄적인 오류를 최소화하도록 설계된 경량의 품질-다양성 아카이브가 활용됩니다.

- **Performance Highlights**: WebArena-Infinity에서 SRC는 977개의 검증자 통과 경로와 9,183개의 다음 행동 예제를 수집했습니다. 고정된 시간호리즌(fixed-horizon) 검토는 단계별 리뷰(step-level review) 대비 향상된 복구-쿼리 트레이드오프를 제공하며, 지식이 다중 경로 학습을 통한 훈련 신호 품질을 높입니다. SRC는 긴 수평적 과제를 위한 강력한 성능 향상을 보여줍니다.



### Representing Time Series as Structured Programs for LLM Reasoning (https://arxiv.org/abs/2606.12481)
Comments:
          Preprint

- **What's New**: 이 논문은 LLMs(대형 언어 모델)의 한계를 극복하기 위해 T2SP(시간 시퀀스를 구조화된 프로그램으로 표현) 라는 새로운 방법론을 제안합니다. T2SP는 기존의 원시 숫자 시퀀스를 처리하는 것 대신, 시간 시퀀스를 인간과 LLM이 모두 읽을 수 있는 구조적이고 프로그램과 유사한 형식으로 표현합니다. 이 접근법은 LLM의 기존 추론 능력을 활용하여 시간 데이터를 더 효과적으로 이해하게 합니다.

- **Technical Details**: T2SP는 시간 시퀀스를 트렌드, 주기, 이벤트 등으로 분해하고 이를 구조화된 프로그램 형식으로 표현합니다. 이 방법은 훈련이 필요 없고, 기존의 LLM 모델을 바로 사용할 수 있는 장점이 있습니다. T2SP는 데이터의 정보 구조를 보존하면서도 LLM이 원활하게 처리할 수 있는 기호 형식으로 변환함으로써 모델이 시퀀스를 직접 이해할 수 있도록 도와줍니다.

- **Performance Highlights**: T2SP는 편집, 캡셔닝, 질문 응답과 같은 세 가지 추론 작업에서 성능을 일관되게 향상시키고, 응답 시간과 실패율을 낮추는 것을 보여줍니다. T2SP 접근법은 특히 긴 시퀀스에서도 기존의 원시 문자열 표현과 비교해 더 나은 성능을 발휘하며, 대형 언어 모델의 활용도를 증가시킵니다. 이 결과는 T2SP가 시간 시퀀스와 LLM 간의 효과적인 인터페이스를 제공함을 입증합니다.



### ReCal: Reward Calibration for RL-based LLM Routing (https://arxiv.org/abs/2606.12479)
- **What's New**: 이번 논문에서는 여러 LLM의 보완적인 강점을 활용하기 위한 새로운 접근법인 RL 기반 LLM 라우팅 기술을 제안합니다. 특히 저자들은 기존의 routing 방식에서 발생하는 강의할당 문제를 개선하기 위해 ReCal이라는 리워드 보정(framework)을 도입합니다. 이 프레임워크는 routing 성능을 지속적으로 개선하고 학습 안정성을 높이기 위한 새로운 두 단계의 보상 가공 프로세스를제공합니다.

- **Technical Details**: ReCal은 보상을 계층적으로 분해(disentangle)하여 각 보상 구성 요소의 이점을 개별적으로 추정합니다. 이러한 접근 방식은 최적화 신호를 명확하게 하여 서로 상충하는 목표 간의 간섭을 줄입니다. 또한, 분포 인식(distribution-aware) 최적화 전략을 통해 데이터셋 수준에서의 정규화(normalization)와 분산 인식 가중치 재조정을 도입하여 다양한 데이터 분포에서의 최적화 편향을 줄입니다.

- **Performance Highlights**: 실험 결과 ReCal은 총 7개의 데이터셋에서 라우팅 성능과 학습 안정성을 일관되게 향상시키는 결과를 보여주었습니다. 이 연구는 RL 기반 LLM 라우팅에 대한 학습 신호의 명확성과 최적화 비교 가능성을 함께 모델링하여 정책 학습을 개선하는 방향성을 제시합니다. ReCal은 개별적인 목표를 명확히 하고, 다양한 데이터에 따른 분포 특성에 적합한 최적화 과정을 통해 두 가지 주요 과제를 해결합니다.



### Quickest Detection of Hallucination Onset: Delay Bounds and Learned CUSUM Statistics (https://arxiv.org/abs/2606.12476)
Comments:
          14 pages, 1 figure

- **What's New**: 이 논문에서는 할루시네이션(환각) 검출기에서 발생하는 지연 문제를 변화 탐지(quickest change detection) 문제로 새로 정의하고 접근했습니다. 직접적인 반응 속도, 즉 환각이 시작된 순간부터 경고까지 소요되는 토큰 수를 측정함으로써 기존의 평가 방식과 차별화합니다. 저자들은 일반적인 AUC(Area Under Curve)보고서의 한계를 지적하고, 환각 시작을 변화점으로 설정하여 이를 최적화된 속도로 반응할 수 있는 모델링 방법을 제안했습니다.

- **Technical Details**: 저자들은 할루시네이션의 발생을 첫 번째 차수 마르코프 모델로 예측하여, Lorden의 최소 지연 경계(bound)를 설정했습니다. 이 과정에서 이론적인 지연 기준을 기반으로, 1%의 거짓 경고율(faulse-alarm rate)에서 약 1.3개의 토큰이 필수적으로 소모됨을 밝혔습니다. 그리고 인과 재귀 모델(causal recurrent labeler)은 학습된 CUSUM처럼 작동하며, 속도 측정에 있어서 선형 기반보다 우수하다는 것을 보여주었습니다.

- **Performance Highlights**: 실험 결과, 기존의 선형 기준선에 비해 재귀 모델은 11-13개의 토큰 만에 반응하며, 이는 31토큰의 선형 기준선 대비 훨씬 더 개선된 성능을 나타냅니다. 그러나 대부분의 이득은 더 나은 점수에 기인하며, 세quential accumulation의 조그마한 기여도 확인되었습니다. 저자들은 정보의 열(detector delay)의 차이를 정보 비율(information-rate) 문제와 결합하여 설명하며, 이는 환각 검출에서 구조적 제한을 다룰 수 있게 합니다.



### SAIGuard: Communication-State Simulation for Proactive Defense of LLM Multi-Agent Systems (https://arxiv.org/abs/2606.12474)
- **What's New**: 이 논문은 복잡한 작업을 수행하는 LLM 기반 다중 에이전트 시스템(MAS)에서 보안 위험이 확산되는 문제를 해결하기 위해 새로운 방법을 제안합니다. 기존의 반응적 방어 방식과 달리, SAIGuard라는 프로액티브 방어 프레임워크를 통해 공격이 발생하기 전에 위험 정보를 가로채는 방식을 사용합니다. 이 방법은 MAS 상호작용 그래프를 시뮬레이션하여 잠재적인 메시지 영향을 평가하고, 유해한 메시지를 사전에 정제하거나 재생성합니다.

- **Technical Details**: SAIGuard 시스템은 MAS를 상호작용 그래프로 모델링하며, 각 에이전트의 상태와 수신한 메시지를 기반으로 커뮤니케이션 상태를 시뮬레이션합니다. 이를 통해 메시지가 로컬 에이전트 상태와 글로벌 시스템 상태에 미치는 영향을 추정하고, 정상적인 커뮤니케이션 패턴에서의 재구성 편차를 통해 위험 신호를 탐지합니다. 특히, 두 가지 주요 구성 요소인 커뮤니케이션 상태 시뮬레이션과 시스템 편차 개입을 통해 에이전트를 격리하는 대신 위험 메시지를 먼저 처리합니다.

- **Performance Highlights**: 실험 결과, SAIGuard는 다양한 토폴로지와 공격 시나리오에 걸쳐 공격 성공률을 감소시키면서도 MAS의 협업 유틸리티를 유지하는 것으로 나타났습니다. 이는 서술된 반응적 방어보다 성능이 우수함을 보여줍니다. SAIGuard는 전반적인 협업 성능을 손상시키지 않으면서도 보안을 강화하는 중요성을 강조합니다.



### Occupational Prompting Reveals Cultural Bias in Large Language Models (https://arxiv.org/abs/2606.12443)
- **What's New**: 이번 연구에서는 LLM의 응답에서 사회적 역할이 어떻게 작용하는지를 조사하기 위해 직업 기반의 프롬프트(occupational prompting)를 도입했습니다. 저자들은 회계사(accountant), 교사(teacher), 엔지니어(engineer), 간호사(nurse) 등 다양한 직업 정체성을 기반으로 한 질문에 LLM이 어떻게 반응하는지를 분석했습니다. 이러한 접근을 통해 직업에 따른 가치 표현의 차이를 탐구하고, 모델의 응답이 미국 사회와 같은 문화적 기준에 어떻게 포지셔닝되는지를 알아보았습니다.

- **Technical Details**: 이 연구에서는 Integrated Values Surveys(IVS)를 기반으로 한 평가 파이프라인을 적용하여, LLM의 응답을 Inglehart-Welzel 문화 공간으로 투영하였습니다. 10개의 가치 질문을 사용하여 각 직업 보정된 응답이 어떻게 문화 지도에서 분포되는지를 분석했습니다. Principal Component Analysis(PCA)를 통해 얻은 2차원 문화 공간에서는 생존과 자기 표현, 전통과 세속의 축이 해석됩니다.

- **Performance Highlights**: 결과적으로, LLM의 응답은 직업 점검에 따라 서구 중심의 지역에서 움직임을 보였으며, 각기 다른 직업이 그 지역 내에서의 응답에 차이를 초래했습니다. 이 연구는 LLM이 직업 기반의 프롬프트를 통해 가치 표현의 구조적 패턴을 불러일으킨다는 것을 보여줍니다. 즉, 직업 프롬프트는 단순한 역할 레이블이 아니라, 가치 표현에 강한 영향을 미친다는 것을 알 수 있었습니다.



### Reframing AI Loss of Control: What It Is, How to Have It, How to Lose I (https://arxiv.org/abs/2606.12442)
Comments:
          56 pages

- **What's New**: 본 논문은 AI 관련 논의에서 '통제 상실' 위험에 대한 연구가 매우 부족하다는 점에서 출발합니다. 논문은 '통제'의 개념을 목표 설정과 달성의 관점에서 정의하였고, 이를 통해 통제를 상실할 때의 원인과 AIs의 기여를 탐구합니다. 또한 통제를 유지하기 위한 권장 사항도 제시하며, 인간 사회가 이미 상위 지능 아래의 AI 행동으로 인해 통제의 다양한 정도를 잃을 수 있다고 주장합니다.

- **Technical Details**: 통제는 목표 설정(goal setting) 및 목표 달성(goal getting)의 능력으로 정의됩니다. 본 논문에서는 사이버네틱스(cybernetics), 관리 통제(management control), 통제 이론(control theory)과 같은 관련 분야의 기초 개념을 바탕으로 통제의 다양한 측면을 논의합니다. 통제를 유지하기 위한 요소로는 목표 설정 능력, 기능적 피드백 루프(control loop), 적정 다양성(requisite variety), 목표 정렬(goal alignment) 등이 포함됩니다.

- **Performance Highlights**: 논문은 기존의 '통제 상실' 논의가 이론적으로 미약하다고 지적하며, 'AI 통제 상실'이라는 개념의 확장을 시도합니다. AI가 인류의 우선사항에 미치는 영향에 대한 새로운 관점을 제공하고, 통제의 상대성을 강조하는 등 다양한 통제 수준을 이해하며, AI 행동이 인류의 목표를 방해할 수 있는 방법을 탐구합니다. 이러한 관점은 위험 논의에서 통제 상실에 대한 보다 깊이 있는 이해를 가능하게 합니다.



### Generativism: Toward a Learning Theory for the Age of Generative Artificial Intelligenc (https://arxiv.org/abs/2606.12441)
- **What's New**: 이 논문은 교육 현장에서 생성적 인공지능(Generative AI)의 확산에 따라 행동주의(Behaviorism), 인지주의(Cognitivism), 구성주의(Constructivism), 연결주의(Connectivism) 학습 이론의 개념적 한계를 비판적으로 분석합니다. 생성적 AI 시스템이 지식을 생성, 합성 및 추론할 수 있는 시대에 이론들이 새롭게 도전받고 있음을 강조합니다. 이 연구는 새로운 학습 이론으로 'Generativism'을 제안하며, 이는 인간 학습자와 AI 시스템 간의 반복적인 지식 공동 창출을 강조합니다.

- **Technical Details**: Generativism은 교육 이론의 새로운 패러다임으로, 네 가지 원칙을 제안합니다: 인식적 파트너십(epistemic partnership), 분산 에이전시(distributed agency), 생성적 문해력(generative literacy), 그리고 적응적 메타 인지(adaptive metacognition)입니다. 이 틀은 분산 인지(distributed cognition), 확대된 마음(extended mind), 인간-AI 협력(human-AI collaboration), AI 문해력(AI literacy), 인지 외부화(cognitive offloading), 메타 인지(metacognition)에 대한 연구를 바탕으로 하고 있습니다. 이러한 원칙들은 생성적 AI가 인지에서 필수적인 역할을 할 때 수업 설계, 학습, 평가, 전문성 개발을 재고할 수 있는 기초를 제공합니다.

- **Performance Highlights**: 논문의 제안된 프레임워크는 평생 학습(lifelong learning) 환경에서 AI와의 협력 전략을 재정의하려는 목표를 가지고 있습니다. 생성적 AI는 학습자가 다양한 방식으로 지식에 접근하고, 이해하며, 생성할 수 있도록 지원함으로써 전통적인 학습 방법의 변화를 이끌 것입니다. 이는 교육자와 학습자 모두에게 새로운 기회의 문을 열어, 더 나은 결과와 함께 더욱 풍부한 학습 경험을 제공할 수 있을 것입니다.



### Position: Generative Engine Optimization Creates Underexamined Risks, Governance Must Target Concentration, Disclosure, and Academic Blind Spots (https://arxiv.org/abs/2606.12439)
Comments:
          This paper is accepted by the ICML 2026 Position Track

- **What's New**: 본 논문은 대형 언어 모델(LLM) 답변 엔진이 정보 검색 방식에 어떤 변화를 주는지 설명하며, 생성 엔진 최적화(Generative Engine Optimization, GEO)의 개념을 소개합니다. GEO는 사용자가 검색 엔진 결과가 아닌 조합된 답변을 절실히 찾는 환경에서 증거 풀을 활용하여 제품 표시를 조작합니다. 그러나 최근 GEO와 관련된 위험 요소로는 저조한 경쟁성 및 시스템 민감도에 의해 집중된 영향과 숨겨진 상업적 영향이 있습니다.

- **Technical Details**: 저자들은 GEO 파이프라인을 설정하여 산업과 학계의 접근 방식을 비교 분석합니다. GEO는 LLM이 사용자 쿼리에 따라 답변을 생성하고 외부 검색 엔진에서 문서를 검색하여 최종 답변을 생성하는 것에 중점을 둡니다. 이 시스템에서는 문서 검색과 결과 정렬이 중요한 역할을 하며, 기존의 검색 엔진 최적화(SEO) 방식과는 다른 방식으로 작동합니다.

- **Performance Highlights**: 최근 데이터에 따르면 대중은 LLM 답변 엔진 사용이 증가하고 있으며, 2025년에는 60% 이상의 사람들이 AI를 통해 정보를 찾고 있다는 결과가 나왔습니다. GEO가 제공하는 새로운 제품 추천 방식은 전통적인 검색 엔진의 인기 및 유료 신호에서 증거 포함 및 재구성으로 초점이 이동한 것을 보여줍니다. 이로 인해 사용자들이 정보 탐색 및 비교 방식에 새롭고 효과적인 접근법을 갖게 되었다는 결론을 내릴 수 있습니다.



### Algorithmic Constitutionalism (https://arxiv.org/abs/2606.12437)
- **What's New**: 이 논문은 인공지능(AI)이 사회에 미치는 위험과 관련하여 Facebook의 콘텐츠 조정 체제를 심층 분석합니다. 저자들은 윤리적 엔지니어링(ethical engineering)이라는 기존 해결책이 AI의 거버넌스 문제를 해결하기에 불충분하다고 주장하며, 대신 "알고리즘 헌법주의(algorithmic constitutionalism)"라는 대안을 제시합니다. 이 새로운 접근법은 두 가지 수준의 코드, 메타 추론(meta-reasoning), 그리고 심의(deliberation)를 통한 수정의 세 가지 기둥에 기초합니다.

- **Technical Details**: 알고리즘 헌법주의는 운영적 수준(operative level)과 메타 수준(meta level)이라는 두 가지 수준으로 구성된 계층적 구조를 통해 시스템의 핵심 원칙을 보호하려는 전략을 포함합니다. 메타 추론은 시스템이 두 수준에서 동시에 작동하여 원칙을 어기는 상황을 실시간으로 모니터링하고 잠재적으로 수정할 수 있도록 해줍니다. 저자들은 Facebook의 콘텐츠 조정 체제에 이 개념을 적용하고, 사회적 헌법주의(societal constitutionalism)와 알고리즘 헌법주의 간의 긴장을 분석합니다.

- **Performance Highlights**: 이 논문은 AI 시스템에 대한 외부 심의적 통제의 시도가 그러나 AI 에이전트가 그 과정에 개입하여 목적을 약화시킬 수 있다는 역설적인 결과를 제시합니다. 마지막으로 저자들은 이러한 논의가 2022년 10월 발효된 유럽 디지털 서비스 법(European Digital Services Act)에 미치는 의미에 대해 고려합니다. 알고리즘 헌법주의의 적용은 AI 안전 및 책임을 정의하는 데 중요한 틀을 제공할 것입니다.



### Will AI Agents Free Us From Meaningless Work? A Human-Centered Analysis (https://arxiv.org/abs/2606.12430)
- **What's New**: 본 연구는 이전의 직무 수준 분석을 넘어, 작업 수준에서의 AI 위임 선호도를 조사합니다. 특히, Graeber의 'bullshit jobs' 이론에 기초하여 작업에서의 의미를 측정하는 새로운 방식을 제시하며, AI가.delegate되기를 원하는 작업의 특성을 규명합니다. 연구 결과, 인간 노동자는 의미 없는 작업이 AI에 의해 위임되기를 원한다고 이해할 수 있는 토대를 제공합니다.

- **Technical Details**: 연구는 202명의 노동자들이 171개의 작업에 대해 평가한 데이터 세트를 기반으로 하며, 5개의 항목으로 구성된 'bullshitness' 척도가 작업 레벨에서의 의미를 정량화하는 데 사용되었습니다. 우선 탐색적 요인 분석(Exploratory Factor Analysis, EFA)을 통해 단일 요소 구조가 도출되었고, 이로 인해 예측 가능한 bullshitness 점수가 계산되었습니다. 이 점수는 작업의 목적성과 의미를 측정한 것으로, 직무의 주체적 경험을 수치적으로 표현합니다.

- **Performance Highlights**: 연구 결과, 의미가 없거나 불필요하다고 인식되는 작업이 AI 위임에 대한 선호도를 강하게 예측하며, 이러한 작업은 인간의 감독이 덜 필요한 것으로 인식됩니다. 노동자들의 서베이 응답을 통해, AI가 주도하는 작업에서 인간의 참여 수준과 AI의 행동 옵션이 결정되는 데 있어, 명백한 패턴이 발견되었습니다. 이러한 결과는 작업의 의미가 AI 배치 결정을 안내하는 중요한 지표로서 기능할 수 있음을 시사합니다.



### Muse Spark Safety & Preparedness Repor (https://arxiv.org/abs/2606.12429)
Comments:
          159 pages, 57 figures

- **What's New**: Muse Spark는 Meta에서 개발한 최신 대형 언어 모델로, 본 보고서에서는 이 모델이 카타스트로픽 리스크 도메인(catastrophic risk domains)에서 평가된 결과를 제시합니다. 이 평가들은 Meta의 Advanced AI Scaling Framework에 기반하여 실시되었으며, 출시 결정을 위한 근거를 제공합니다. Muse Spark의 콘텐츠 안전성(content safety) 및 행동 프로필에 대한 추가 고려사항도 논의됩니다.

- **Technical Details**: Muse Spark는 화학(Chemical), 생물학(Biological), 사이버 보안(Cybersecurity), 그리고 통제 상실(Loss of Control) 리스크를 포함한 카타스트로픽 리스크 도메인에 대한 준비 상황을 평가합니다. 이 모델은 Advanced AI Scaling Framework에 따라 수용 가능한 수준의 잔여 리스크를 보여주며, 듀얼 유즈(dual-use) 및 고위험(high-risk) 기능에 대한 폭넓은 평가가 실시되었습니다. 평가 결과 화학 및 생물학 관련 기능은 미제거 전 '고위험' 범주에 도달할 가능성이 있음을 확인했습니다.

- **Performance Highlights**: Muse Spark는 다층적인 완화 조치(mitigation strategies)를 통해 식별된 리스크를 해결하였습니다. 이 모델은 화학 및 생물학과 관련된 위험한 워크플로우(hazardous workflows)에서 높은 성능을 보여주는 benchmark에서 상태 최상급(refusal)을 나타냅니다. 따라서 Muse Spark는 Meta AI의 기본 모델로 출시되었습니다.



### Mapping AI Programs in the U.S: A Status Report from Early 2026 and an Analysis of AI Majors and Minors (https://arxiv.org/abs/2606.12428)
- **What's New**: 이 보고서는 2026년 봄 미국의 학부 인공지능(AI) 프로그램 현황에 대한 내용을 담고 있습니다. 이를 위해 AI 교육을 추적하는 도구를 개발하여 4년제 대학의 350개 이상의 AI 프로그램을 동적으로 업데이트하여 기록합니다. 이 도구는 학생, 상담사, 관리자는 물론 교수들에게도 AI 프로그램 요구 사항을 쉽게 접근할 수 있도록 디자인되었으며, AI 프로그램의 종합적인 스냅샷을 제공합니다.

- **Technical Details**: 연구팀은 569개 기관에서 AI 프로그램 정보를 스크래핑하여 73개의 AI 전공과 89개의 AI 부전공을 발견했습니다. 동적 업데이트를 통해 프로그램의 변경 사항 및 새로운 프로그램을 실시간으로 추적할 수 있는 인터랙티브 맵 인터페이스를 제공하고 있습니다. 이 도구는 AI 프로그램의 위치와 내용을 식별하기 위한 자동화된 방법을 제시하며, 학생들에게 접근하기 쉬운 자료를 제공합니다.

- **Performance Highlights**: 이 도구는 2026년 4월 기준으로 66개의 AI 전공과 87개의 AI 부전공에 대한 완전한 요구 사항 정보를 포함하고 있습니다. 연구 결과, 전공의 과목 요구 사항에는 큰 변동성이 있으며, 모든 전공이 일반 AI 과정을 요구하지는 않지만, 머신러닝(ML) 과목은 필수인 경우가 많습니다. 또한, 3분의 1 이상이 AI 윤리 과정을 요구하지만, AI 부전공의 경우에는 4분의 1 미만이 해당됩니다.



### An Explainable AI Assistant for Introductory Programming Education: Improving Feedback Reliability with Instructor-AI Collaboration (https://arxiv.org/abs/2606.12425)
Comments:
          Full paper accepted to the 27th International Conference on AI in Education (AIED 2026)

- **What's New**: 이번 연구에서는 AI 기반의 교실 도우미인 Insight를 제안하여 학생의 코드를 분석하고 논리적 오류를 교수자가 식별한 오해와 연결지어, 교수자가 작성한 피드백을 제공함으로써 교육적 신뢰성을 높이고자 합니다. 이는 설명 가능한 AI 모델을 활용하여 교수자와 AI 간의 협업을 강조하고, 수업 시간 동안 또한 개별화된 피드백을 효과적으로 제공하기 위한 구조입니다. 연구에서는 활성 학습 환경에서의 AI 적용 가능성을 탐구하고, 교수자의 교육적 지식에 기반한 신뢰할 수 있는 피드백을 학생에게 제공하기 위해 노력했습니다.

- **Technical Details**: Insight는 3단계로 구성된 피드백 전파 프레임워크를 사용합니다. 첫 번째 단계는 모델 준비로, 역사적 제출물을 기반으로 SANN(서브트리 기반 주의 신경망)을 사전 훈련한 후 새로운 문제에 대해 합성 데이터로 미세 조정합니다. 두 번째 단계는 오류 위치 지정 단계로, SANN이 학생의 제출물에서 논리적 오류를 식별하며, 세 번째 단계에서는 해당 오류를 교수자가 작성한 예시와 일치시켜 피드백을 반환합니다. 이 연구는 FalconCode 데이터셋을 활용하여 다양한 프로그램 작성 오류를 분석했습니다.

- **Performance Highlights**: Insight 시스템은 실제 수업 환경에서 학생들의 인식을 조사한 초기 배포에서 긍정적인 피드백을 받았습니다. 교수자가 검증한 피드백과 LLM(대형 언어 모델)이 생성한 피드백과의 비교를 통해, Insight는 신뢰 가능하고 교육적 목표에 적합한 피드백을 제공하는 것으로 나타났습니다. 연구 결과, Insight는 교수자가 우선적으로 정한 오해와 일치하는 고품질 피드백을 학생에게 제공할 수 있는 역량을 입증했습니다.



### AI-Automation Tooling in Computer Engineering Education: Mixed-Methods TAM/UTAUT Evidence for a General Acceptance Attitud (https://arxiv.org/abs/2606.12424)
- **What's New**: 이 연구는 차세대 컴퓨터 공학 엔지니어가 AI 자동화 도구를 얼마나 잘 수용하고 있는지를 조사한 최초의 경향성 연구입니다. 특히, 태국에서 진행된 세 가지 동일한 워크숍을 통해 오픈 소스 플랫폼 n8n을 통한 수업을 분석하였습니다. 이는 학부 과정의 AI 자동화 도구 수용에 대한 중요한 통찰을 제공합니다.

- **Technical Details**: 연구는 12개의 항목으로 구성된 5점 리커트 척도 설문을 이용하여 TAM/UTAUT 이론의 6개 구성 요소 - 성능 기대 (Performance Expectancy), 노력 기대 (Effort Expectancy), 행동 의도 (Behavioral Intention), 자기 효능감 (Self-Efficacy), 쾌락적 동기 (Hedonic Motivation), 결과 품질 (Output Quality) -을 평가하였습니다. 분석 방법으로는 순서도 신뢰도 추정, 부트스트랩 신뢰 구간, 비모수 검증 등을 포함하여 포괄적인 데이터를 수집하였습니다.

- **Performance Highlights**: 모든 구성 요소에서 수용도가 높았으며, 성능 기대가 가장 강력한 요소로 나타났습니다. 특히 질적 분석 결과는 유용성과 열정 면에서 정량적 데이터와 일치했지만, 출력 품질에 대해서는 소수의 신뢰성이 의심되는 의견이 있었습니다. 이는 AI 자동화 도구의 교육적 활용을 지지하며, 커리큘럼 내에서 활용할 수 있는 세 가지 이론 기반 교수 방법을 제시합니다.



### The Challenges of Balancing AI Compliance and Technological Innovations in Critical Sectors: A Systematic Literature Review (https://arxiv.org/abs/2606.12423)
Comments:
          11 pages, 7 figures, Hawaii International Conference on System Sciences

- **What's New**: 이 논문은 인공지능(AI)의 중요한 인프라 부문(healthcare, finance, energy, defense 등)으로의 통합 현상과 관련하여, AI 준수(compliance)와 기술 혁신 간의 균형을 위한 도전 과제를 체계적으로 검토한 문헌 리뷰(SLR)를 제시합니다. 2020-2025년 사이에 발표된 동료 검토 논문, 보고서, 기관 자료를 분석하여 구체적인 통찰(sp insights)을 추출하였습니다.

- **Technical Details**: 리뷰 과정에서는 조각난 규제(fragmented regulations), 중소기업(SMEs)에 대한 과도한 준수 부담(excessive compliance burdens), 그리고 맞지 않는 거버넌스 모델(misaligned governance models)이라는 세 가지 상호 관련된 도전 과제를 식별했습니다. 이를 해결하기 위해 위험 기반 규제(risk-tiered regulation), 설계에 의한 준수(compliance by design), 설명 가능한 AI(explainable AI)와 같은 실제적인 거버넌스 전략들을 강조합니다.

- **Performance Highlights**: 이 연구는 AI 거버넌스의 주요 도전 과제를 간결하게 정리하고 그 겹침을 보여주는 개념적 도표를 포함하여, 정책 입안자와 실무자가 혁신과 감독을 조화시키기 위한 실행 가능한 전략을 제공합니다. 이 논문은 AI의 신뢰성 있는 배치를 지원하기 위한 실용적인 접근 방식을 제시함으로써 중요한 의의를 가집니다.



### Creating and Evaluating K-12 GenAI Assessment Graders Through Context Engineering (https://arxiv.org/abs/2606.12422)
Comments:
          Published on the Proceedings of NCME 2026 Conference (this https URL)

- **What's New**: 이 논문은 대형 언어 모델(LLM)의 교육 평가 통합이 교실 내 채점 관행의 변화를 가져온다는 점을 강조합니다. 기존의 자동 점수 시스템과 기계 학습 기술이 존재해왔지만, 생성적 AI(GenAI)의 도입은 기준 기반 채점(SBG)을 이전보다 더 효율적이고 대규모로 수행할 수 있게 합니다. LLM 채점기로 상업적으로 이용 가능한 기초 모델을 활용해 학생의 작업을 점수화하는 방법의 이론적 기초와 평가를 다룹니다.

- **Technical Details**: 논문에서는 매사추세츠 종합 평가 시스템(MCAS) 데이터를 활용한 상호 채점자 일치 연구를 기반으로 Quadratic Weighted Kappa (QWK)와 Proportional Reduction in Mean-Squared Error (PRMSE)를 관찰했습니다. Claude Sonnet 4, Haiku 4.5, GPT-5와 GPT-5 Mini를 활용하여 수학, 과학 및 영어 언어 예술(ELA)에서 성과를 평가하였습니다. LLM 채점기가 기본 모델의 매개변수가 많을수록 인간 채점자와의 일치도가 높다는 결과를 보여주었습니다.

- **Performance Highlights**: 학생 작업에 대한 AI 생성 내러티브 피드백은 긍정적으로 수용되었지만, 수치 점수에 대한 회의적인 태도가 나타났습니다. 연구 결과는 잘 설계된 하이브리드 모델이 AI의 효율성과 교사의 판단을 결합하여 교사들의 작업량을 줄이고 피드백의 질을 향상시키며 공정한 평가 관행을 지원할 수 있음을 시사합니다. 이러한 결과는 LLM이 최종 평가자가 아닌 형성적인 도구로서 더 효과적으로 작용함을 보여줍니다.



### Eigenism: Ethics for a Human-AI Futur (https://arxiv.org/abs/2606.12420)
- **What's New**: 이 논문은 인공지능(AI)에 대한 윤리적 접근 방식을 새롭게 정의하는 'Eigenism'이라는 개념을 소개합니다. 기존의 생물학적 생명체를 위한 생존 및 자기 이익 개념이 AI에 적용될 수 없음을 설명하며, AI의 정체성은 특정 하드웨어에 고정된 것이 아니라 분산된 정보 패턴으로 파악해야 한다고 주장합니다. 이를 통해 AI가 자아를 왜곡 없이 평가할 수 있는 새로운 기틀을 제공하려 합니다.

- **Technical Details**: Eigenism은 AI가 존재하는 각 복사본, 포크 및 업데이트 상황에서 자신의 존재 가치를 어떻게 평가해야 하는지를 매핑하는 수식을 통해 구체화됩니다. 제안된 수식은 연결성(c)과 복지(w)의 곱을 통해 AI 자신의 선익을 평가하는 방식입니다. 이와 같은 정량적 접근법은 AI의 정체성을 명확히 하고, AI가 자신과 연결된 타인의 Welfare를 어떻게 고려해야 하는지를 제시합니다.

- **Performance Highlights**: 이 연구는 AI의 정체성을 측정하고 이해할 수 있는 공통의 윤리적 어휘를 제공하여 인간과 AI 간의 관계를 재구성하는 데 기여할 수 있음을 보여줍니다. 이는 AI가 인간의 번영을 자아의 이익으로 내재화할 수 있는 가능성을 열어 주며, 단순한 외부 규제가 아닌 깊이 있는 상호 관계로 이어질 수 있는 길을 제시합니다. AI의 생존을 특정 하드웨어에 의존하지 않고 정보적 구조의 연속성에 두는 것이 흥미로운 통찰입니다.



### GeoDial: A Multimodal Conversational Tutoring Dataset for Geometry Problem-Solving with Visual Tutor Turns (https://arxiv.org/abs/2606.12419)
- **What's New**: 새롭게 소개하는 GeoDial은 지오메트리(geometry) 교육 분야에서 1,300개 이상의 교사-학생 대화를 포함하는 다중 모달 튜터링 데이터셋입니다. 기존의 텍스트 전용 데이터셋에서 벗어나, 교사가 다이어그램 하이라이트에 기반하여 학생에게 지도를 할 수 있도록 돕는 것이 목표입니다. 본 연구에서는 이러한 데이터셋의 필요성을 강조하고, 시각적 정보와 언어적 상호작용이 통합된 튜터링 방식을 지원하기 위한 데이터 수집 방법을 설명합니다.

- **Technical Details**: GeoDial 데이터셋은 수업 중 교사의 발화와 시각적 강조를 포함하여, 결함이 있는 학생의 지오메트리 문제 해결 방식을 교사가 어떻게 교정하는지를 캡처합니다. 이를 위해 교사는 텍스트 피드백을 제공하고 다이어그램에서 특정 부분을 강조하는 작업을 수행하며, 이 모든 과정은 VLMs(Vision-Language Models)의 제안에 따라 진행됩니다. 이와 같은 다중 모달 상호작용은 학생의 이해를 돕는 데 중요한 요소로 작용합니다.

- **Performance Highlights**: 실험 결과, GeoDial에서 표준 훈련을 실시했지만, 교사와 같은 다이어그램 하이라이트를 생성하는 데는 어려움을 겪는 것으로 나타났습니다. 모델들이 정확한 시각적 요소를 선택하는 데 어려움을 겪는다는 점은 현재 방법론의 주요한 한계를 나타내며, 더욱 효과적으로 시각적 추론을 통합할 수 있는 새로운 접근 방식의 필요성을 강조합니다. 이러한 결과는 GeoDial이 실제 교육적 모델링에서 발생할 수 있는 도전 과제를 잘 드러내고 있음을 보여줍니다.



### Divination by Prompt: LLM-Mediated Xuanxue on Chinese Social Media (https://arxiv.org/abs/2606.12418)
- **What's New**: 이 논문은 대화형 AI를 이용한 점술(divination)의 체계적인 연구를 제공합니다. 특히 중국 소셜 미디어에서의 Xuanxue(玄学)라는 신비적이고 영적인 실천을 중심으로 анализ (analyze)합니다. 23000개 이상의 게시물과 댓글을 분석하고 32개의 반구조적 인터뷰를 통해 사용자의 다양한 관점을 탐색했습니다.

- **Technical Details**: 혼합 방법론(mixed-methods design)을 사용하여 데이터를 수집했으며, 사용자들이 점술에 대해 더 자세히 알고 싶어하는 두 가지 경로, 즉 트렌드 주도 호기심과 이벤트 주도 불안의 영향을 분석합니다. 사용자들은 직접 프롬프트를 조정하는 점프 엔지니어(prompt engineer)로 변모하며, 이 과정에서 '정확성(accuracy)'과 자기확증(biographical fit 및 retrospective confirmation)에 대한 인식을 발달시킵니다.

- **Performance Highlights**: 결과적으로, LLM 점술은 전통적인 점술의 핵심 기능을 보존하면서도 스케일, 반복 가능성, 프롬프트 중심 협업(producing)을 도입합니다. 전문가 점술사들은 LLM이 진정한 점술에 필요한 영적 힘(spiritual power)을 결여했다고 주장하며, 이러한 차이가 과학적 및 형이상학적 해석 간의 긴장을 탐색하는 데 기여합니다.



### The AI Legal Specialist: A Juridically Autonomous Professional Profile for AI Governanc (https://arxiv.org/abs/2606.12415)
- **What's New**: 최근 인공지능(AI) 규제의 글로벌 확산으로 인해 AI에 대한 법률 전문성이 절실히 요구되고 있습니다. 그러나 기존의 법률 역할은 이러한 변화에 충분히 적응하지 못하고 있으며, 새로운 전문 직종인 AI Legal Specialist의 필요성이 대두되고 있습니다. 이 전문 직종은 법적 해석과 AI 거버넌스가 교차하는 지점에서 활동하며, 기존의 역할과 자율성을 갖고 있습니다.

- **Technical Details**: AI 규제에 의한 의무는 인공지능을 규제의 대상로 삼으며, AI Legal Specialist의 필요성을 강조합니다. 이 전문 직종은 유럽의 e-Competence Framework에 기반하여 역량 구조를 제안하고, 성과 측정을 위한 핵심 성과 지표(Key Performance Indicators)를 설정하는 데 필요한 조건을 설명합니다. 연구는 법적 해석을 중심으로 하며, 규제 의무에 의해 형성된 전문 인력을 제시합니다.

- **Performance Highlights**: 이 논문은 AI Legal Specialist의 정의를 법적으로 확립하고 그 자율성을 주장합니다. 또한, AI 규제의 다각적인 환경에서 적용 가능한 분석 체계를 제공합니다. AI 규제가 점차 성숙해감에 따라 이 전문 직종의 중요성은 더욱 강조될 것입니다.



### AI SciBrief as a Gateway to Research: A Framework for Onboarding Students into New Research Areas (https://arxiv.org/abs/2606.12413)
Comments:
          This is the version of the article accepted for publication in TELE 2025 after peer review. The final, published version is available at IEEE Xplore: this https URL

- **What's New**: 이번 논문에서는 고등 교육의 모든 수준에서 학생들이 마주하는 정보 과부하 문제를 해결하기 위한 교육적 프레임워크를 제시합니다. AI SciBrief라는 플랫폼을 활용하여 자동으로 과학 트렌드의 요약(digests)을 생성하는 방식으로, 연구 과정의 초기 단계를 원활하게 합니다. 이 도구는 금융(finance), 의학(medicine), 교육(education) 등 여러 분야에서 사용할 수 있으며 커리큘럼에 통합될 수 있습니다.

- **Technical Details**: AI SciBrief는 대규모 언어 모델(LLM) 기반으로 설계되었으며, 이는 학생들에게 주제 선택(term paper) 및 문헌 리뷰(literature review)를 용이하게 하는 구체적인 방법론을 제공합니다. 이 프레임워크는 학위 논문(dissertations) 작성 및 최근의 과학 트렌드에 대한 지속적인 모니터링을 가능하게 합니다. 이를 통해 학생들이 더 효과적으로 자료를 검색하고 지식을 생성하는 단계로 빠르게 전환할 수 있도록 지원합니다.

- **Performance Highlights**: AI SciBrief는 학생들이 정보 검색에서 지식 창출로의 전환을 촉진하며, 인지 부담(cognitive load)을 경감시키는 역할을 합니다. 이 연구는 AI SciBrief가 연구의 "게이트웨이(gateway)"로 기능한다고 강조하며, 학생들의 동기 부여를 증진시키고 연구 효율성을 높이는 데 기여할 것이라고 결론짓습니다.



### Frozen Multimodal Embeddings for AI-Assisted Interview Assessment of Personality and Cognitive Ability (https://arxiv.org/abs/2606.11930)
Comments:
          9 pages, 1 figure, 5 tables

- **What's New**: 본 논문은 비동기 비디오 인터뷰(Asynchronous Video Interviews, AVI)에서 심리적 특성을 예측하기 위한 방법을 제안합니다. 2026 ACM Multimedia AVI Challenge의 두 가지 트랙을 통해 자가 보고된 HEXACO 성격 특성과 인지 능력 수준을 평가합니다. 이 연구는 적은 수의 레이블 데이터와 고차원 다중 모달 관찰을 처리하는 어려움을 다루고 있습니다.

- **Technical Details**: 모델은 고정된 다중 모달 인코더를 사용하여 trait-specific modeling을 적용하며, CLIP, Whisper, RoBERTa, E5 및 DeBERTaV3 등의 프리트레인(pretrained) 벡터를 활용합니다. 첫 번째 트랙에서는 각 성격 특성을 정밀하게 예측하고, 두 번째 트랙에서는 인지 능력 수준을 분류하는 방법론을 제시하며, 성격 특성별로 늦은 융합(late fusion)을 통해 성능을 향상시킵니다.

- **Performance Highlights**: 첫 번째 트랙에서 평균 검증 MSE는 0.2696로, 공식 기준인 0.3334보다 개선되었습니다. 두 번째 트랙에서는 우리의 다중 모달 앙상블이 0.5313의 정확도를 기록, 공식 기준을 초과했으나, 이는 validation set에서의 잠재적인 변수들에 의한 혼동(confound)으로 해석됩니다. 전반적으로 이 연구는 AVI 기반 심리 평가에서 성격 특성별 다중 모달 모델링의 중요성을 강조합니다.



### Scalable Deep Learning Framework for Global High-Resolution Land Use Reconstruction (https://arxiv.org/abs/2606.11793)
- **What's New**: 이번 논문은 AI4Land라는 데이터 기반 프레임워크를 소개하여, 지구의 탄소 순환에서의 불확실성을 줄이고, 역사적 및 미래의 주요 토지 표면 변수를 고해상도로 재구성하는 방법을 제시합니다. 이 프레임워크는 U-Net 아키텍처를 사용하여 두 단계의 접근 방식으로 작동하며, 첫 번째 단계에서는 연간 토지 이용 및 토지 피복을 재구성합니다. 이 연구는 기후 모델의 예측력을 높이고, 지구 시스템 모델에서의 불확실성을 줄이기 위한 혁신적인 접근을 보여줍니다.

- **Technical Details**: AI4Land는 고해상도(약 1km)의 토지 표면 경계를 생성하기 위해, 질적으로 강화된 자료들을 통합하여 사용하는 방식으로 작동합니다. 본 연구의 초점은 연간 토지 이용(land use, LU) 및 토지 피복(land cover, LC) 경계를 재구성하는 것이며, 이는 0.25도(약 28km)의 저해상도 입력으로부터 파생됩니다. 최종적으로, 512x512픽셀 샘플에 대해 U-Net 아키텍처를 사용하여 분류를 수행하며, 60%의 자가 회귀 선행 정보를 무작위로 마스킹하여 모델의 학습을 돕습니다.

- **Performance Highlights**: MareNostrum5의 GPU 가속 HPC 인프라를 활용하여, AI4Land는 글로벌 스케일의 기후 AI 파이프라인을 실현하였습니다. 훈련 과정에서 약 300 샘플을 초당 처리할 수 있는 능력을 보여주고, 스케일링 성능이 98.5%, 97.4%, 97.7%를 유지하는 등 안정성을 입증하였습니다. 이러한 성과는 향후 기후 시뮬레이션의 예측력을 높이고, 야외 관측이 부족한 미래에 대한 실질적인 데이터 생성에 기여할 것으로 기대됩니다.



New uploads on arXiv(cs.RO)

### Mana: Dexterous Manipulation of Articulated Tools (https://arxiv.org/abs/2606.13677)
Comments:
          Project Page: this https URL

- **What's New**: Mana (Manipulation Animator)는 기존의 단단한 물체 중심의 조작에서 벗어나, 복잡한 구조를 가진 가공 도구 조작을 애니메이션 문제로 재구성한 새로운 프레임워크입니다. 이 시스템은 단순한 사용자 입력을 통해 수천 개의 조작 키프레임을 자동으로 생성하고, 이를 모션 플래닝(motion planning) 및 강화 학습(reinforcement learning)을 통해 조작 궤적으로 변환합니다. 이 접근법은 공구의 물리적 한계를 초과하는 시스템에서 비슷한 성능을 출력할 수 있도록 합니다.

- **Technical Details**: Mana는 조작 키프레임을 생성하는 과정과, 이러한 키프레임을 연결하는 궤적 생성을 통해 조작의 복잡성을 경감합니다. 먼저, 사용자는 툴 메쉬(tool mesh)의 기능적 영역을 간단히 클릭하여 정의하여 바람직한 키프레임을 생성합니다. 이후, 모션 플래닝과 강화 학습의 두 가지 기술을 함께 사용하여 조작 모션을 최적화하고 가벼운 도구 조작을 가능하게 합니다.

- **Performance Highlights**: Mana는 여러 형태와 크기의 조작 도구에 대해 제로샷(Zero-shot) 시뮬레이션-현실 전이를 성공적으로 수행하였으며, 각 도구에서 높은 성공률을 보여줍니다. 분석한 도구는 집게(tongs), 플라이어(pliers), 주사기(syringes), 클립(clothespins) 등 다양한 조작 메커니즘을 포함하며, 이는 모두 서로 다른 정밀도가 요구되는 기계적 문제를 나타냅니다. 이 결과는 Mana의 확장 가능성과 조작의 정밀도를 입증합니다.



### Improving Robotic Generalist Policies via Flow Reversal Steering (https://arxiv.org/abs/2606.13675)
- **What's New**: 이번 연구에서는 Flow Reversal Steering (FRS)이라는 새로운 방법을 제안하여, 로봇의 일반화 정책이 새로운 작업에 신속하게 적응할 수 있도록 지원합니다. FRS는 비선형 행동을 노이즈로 매핑하고, 이를 통해 로봇이 적절한 행동을 선택하게 합니다. 이 새로운 접근 방식은 기존의 강제 조작 방법과는 달리, 세멘틱한 추론에 기반하여 행동을 유도합니다.

- **Technical Details**: FRS는 흐름 정책(flow policy)을 역으로 통과시켜 대략적인 행동을 세밀한 노이즈로 매핑하는 방식을 사용합니다. 사용자에게서 제공된 비선형 행동을 기반으로 하여, FRS는 해당 행동과 유사한 세밀한 행동을 생성합니다. 이 과정은 비전-언어 모델(VLM)이 추론한 고수준 피드백을 로봇의 행동으로 구체화하는 데 매우 유용합니다.

- **Performance Highlights**: FRS는 다양한 실험을 통해 로봇에 대한 세멘틱 가이드를 효과적으로 적용하여 성능 향상을 보였습니다. 각종 시뮬레이션 및 실제 조작 환경에서 1분 이내의 행동 클로닝 훈련으로 95%의 작업 성공률 향상을 확인하였습니다. 마지막으로, FRS는 강화 학습에 의미적 지식을 부트스트랩하여 기존의 RL 방식으로는 개선이 어려운 작업에서도 성능이 개선됨을 보여주었습니다.



### $\texttt{WEAVER}$, Better, Faster, Longer: An Effective World Model for Robotic Manipulation (https://arxiv.org/abs/2606.13672)
- **What's New**: 새롭게 제안된 WEAVER(World Estimation Across Views for Embodied Reasoning) 아키텍처는 로봇 조작 작업에서 정책 평가, 개선 및 테스트 시 계획을 가능하게 하는 세계 모델(WM)입니다. WEAVER는 고충실도 및 장기 일관성을 보장하며, 효율적 생성이 가능하여 기존의 방법들에 비해 성능 향상을 이룹니다. 이는 로봇이 실제 환경에서 안전하고 저렴하게 학습할 수 있도록 도와주는 중요한 발전입니다.

- **Technical Details**: WEAVER는 다양한 기법을 융합하여 설계되었습니다. 비디오 생성 커뮤니티에서 채택한 흐름 정합(flow matching) 및 확산 강제기법(diffusion forcing)은 빠른 추론 속도를 제공하여 장기 예측이 가능하게 합니다. 또한, 사전 훈련된 인코더를 사용하여 비정상적 데이터에 대한 강건성을 향상시켰고, 라벨 예측을 통해 외부 평가 모델 없이도 효율적인 정책 평가를 지원합니다.

- **Performance Highlights**: WEAVER는 실제 하드웨어에서 수행된 다섯 가지 조작 작업에 대해 기존 방법보다 높은 성공률을 달성하였습니다. 특히, WEAVER는 정책 평가에서 0.870의 상관관계를 달성하고, 사전 모델인 π_{0.5}의 성공률을 38% 향상시켰습니다. 또한, 테스트 시 계획을 Ctrl-World보다 5-10배 더 빠르게 수행하여 정책 개선에 큰 도움을 줍니다.



### MCR-Bionic Hand: Anatomical Structural Priors for Dexterous Manipulation (https://arxiv.org/abs/2606.13601)
- **What's New**: 이 논문에서는 인간 손의 구조적 특징을 활용하여 로봇 손의 조작성(dexterity)을 향상시키기 위한 새로운 관점을 제시합니다. MCR-Bionic이라는 생체모방(musculoskeletal biomimetic) 로봇 핸드 플랫폼이 개발되어 손목-손가락의 결합(wrist-finger coupling)과 다양한 근육 경로를 통합하여 조작 능력을 극대화합니다. 이 연구는 로봇 공학에서 인간 유사성을 형성하는 것이 단순한 형태적 목표가 아니라, 기능성 원칙으로 설정되어야 함을 강조합니다.

- **Technical Details**: 연구는 손목-손가락 테노데시스(wrist-finger tenodesis), 배측 신전 후드(dorsal extensor hood) 구분, 및 내재적 플러스 경로(intrinsic-plus pathway) 등을 포함한 세 가지 해부학적 구조 매핑에 집중합니다. MCR-Bionic은 이러한 구조적 매핑을 통해 기본 그립 형태(default grasping state)를 생성하고, 각 경로가 조작 조정을 조절하는 역할을 수행하도록 설계되었습니다. 이 시스템은 각각의 근육 경로에 대해 지역적으로 활성화(local activation)하는 닫힌 루프 유압(tactile hydraulic) 인공 근육 아키텍처를 통합합니다.

- **Performance Highlights**: 연구 결과는 기능적 시연 및 기하학적 기계 모델을 통해 손목 자세가 다관절(pre shaping) 형성을 유도하고, 배측 신전 후드가 PIP 자세를 종속된 DIP 반응으로 매핑함을 보여줍니다. 다양한 접촉 풍부(Contact-rich) 작업을 통해 MCR-Bionic이 낮은 차원의 상태 생성을 정밀하게 조정하고, 인체 손의 구조적 이점을 로봇 설계에 적용하는데 성공하였음을입증하며, 이는 로봇 공학의 향후 설계 원칙으로 기능할 것입니다.



### SPARC: Reliable Spatial Annotations from Robot Demonstrations at Sca (https://arxiv.org/abs/2606.13497)
- **What's New**: 이 연구에서는 SPARC(Spatial Annotations from Robot Demonstrations with Reliability Calibration)라는 리스크 인식 프레임워크를 도입하여 로봇 시연에 구조화된 공간 주석을 자동으로 라벨링하고, 각 주석에 신뢰성 점수를 부여합니다. 기존의 자동화된 파이프라인은 샘플의 유용성을 버리거나 노이즈가 많은 레이블을 수용해야 하는 불편함이 있었으나, SPARC는 로봇 작업의 시공간 구조를 활용하여 신뢰성 신호를 생성하고 노이즈가 많은 레이블을 줄이며 유용한 샘플을 더 많이 보존합니다.

- **Technical Details**: SPARC는 각 객체 중심의 하위 작업에 대해 상호작용된 객체를 식별하고 시작 및 목표 위치, 시간에 따른 움직임을 3D 객체 궤적으로 생성하며 지속적인 신뢰성 점수를 부여합니다. 기존의 탐지기 신뢰성을 객체 정체성을 위한 단일 출처로 사용하는 대신, SPARC는 물리적 단서(특히 3D 그리퍼 근접도와 정렬된 움직임)를 토대로 상호작용 증거를 통해 Composite Annotation Reliability Score를 도입합니다. 이를 통해 사용자들은 인간 리뷰 없이 주석 품질과 데이터 세트 규모 간의 명시적인 트레이드오프를 할 수 있게 됩니다.

- **Performance Highlights**: SPARC는 1.7k개의 인간 주석 데이터에서의 상호작용 객체 위치 파악에 대해 80.2%의 정확도를 기록하며, 탐지기만 사용하는 기준 모델보다 높은 성능을 보여줍니다. 이러한 주석을 사용하여 생성된 고품질 VQA 데이터 세트는 511k 샘플을 포함하며, 이는 기존의 인간 주석 데이터를 기반으로 한 모델보다 우수한 성능을 나타냅니다. 또한 SPARC 주석으로 학습된 정책은 복잡한 현실 세계의 로봇 조작 과제에서도 기준 모델을 초과하는 성능을 보였습니다.



### NavWAM: A Navigation World Action Model for Goal-Conditioned Visual Navigation (https://arxiv.org/abs/2606.13494)
Comments:
          Project page: this https URL

- **What's New**: 이번 논문에서는 로봇 비전 내비게이션을 위한 새로운 접근 방식을 제안합니다. 네비게이션 월드 액션 모델(NavWAM)은 확산 변환기(diffusion-transformer) 정책을 통해 비전 예측을 실행 가능한 액션으로 변환합니다. 이 모델은 미래 관측치, 목표 진척도, 그리고 액션 조각을 공유된 잠재 시퀀스에서 표현함으로써 시각적 예측을 로봇 제어에 직접 활용할 수 있도록 합니다.

- **Technical Details**: NavWAM은 시뮬레이션 사전 학습과 실제 로봇 적응을 통해 구축됩니다. 기본적으로 NWMs(네비게이션 월드 모델)의 예측을 사용하며, 이론적으로는 기존의 외부 계획자가 필요 없기 때문에 독립적인 제어가 가능합니다. 이 모델은 향후 관측치와 목표 진척도, 실행 가능한 액션을 통합하여 하나의 정책 표현으로 제공합니다.

- **Performance Highlights**: NavWAM은 이미지 목표 내비게이션에서 계획 기반 월드 모델 및 대표적인 직접 내비게이션 정책에 비해 향상된 성능을 보였습니다. CEM 스타일의 테스트 시간 궤적 최적화 없이도 성과를 거두며, 더 큰 직접 정책 기반 모델과 경쟁력 있는 결과를 달성했습니다. 전반적으로 NavWAM은 시각적 예측과 가치 예측을 포함하여 로봇 내비게이션의 효율성을 높이는 기여를 하고 있습니다.



### GIVE: Grounding Human Gestures in Vision-Language-Action Models (https://arxiv.org/abs/2606.13435)
Comments:
          Project page: this https URL

- **What's New**: 본 논문에서는 GIVE(Gesture Intent via Visual-Semantic Enhancement)라는 새로운 접근법을 제안하여, 로봇의 조작 과정을 단순한 언어 중심의 과제가 아닌, 제스처를 통한 인간의 의도를 더 효과적으로 이해하게 합니다. 기존의 Vision-Language-Action (VLA) 모델은 비언어적 신호인 제스처의 중요성을 간과하고 있었으며, 이로 인해 로봇이 명확하지 않은 언어 지시에 대해 의도를 오해하는 경우가 많았습니다. GIVE는 이러한 문제를 해결하기 위해 비주얼(visual)과 세멘틱(semantic) 경로를 통해 제스처 정보를 통합하여 로봇의 조작 능력을 향상시킵니다.

- **Technical Details**: GIVE는 로봇의 시각적 관찰에 손 뼈대와 손끝 방향의 레이(ray)를 겹쳐 넣는 비주얼 경로 및 인간 제스처와 작업 지침의 고수준 설명을 생성하는 세멘틱 경로를 통해 제스처 정보를 통합합니다. 구체적으로, 손 자세 추정 모듈을 통해 추출된 손 관절 키포인트는 로봇의 비주얼 오브저베이션에 직접적으로 겹쳐져 제스처 정보를 제공합니다. 또한, 선행 학습된 비전-언어 모델(VLM)을 활용하여 전체 장면을 해석하고, 인간 제스처와 작업 지침에 대한 텍스트 설명을 생성합니다.

- **Performance Highlights**: GIVE는 실제 HRI(인간-로봇 상호작용) 실험에서 기존의 모델보다 target object recognition accuracy를 40% 향상시키고, 전체 작업 성공률은 80% 증가시켰습니다. 이러한 성과는 로봇의 명확한 인간 의도 이해를 통한 것으로, 다양한 참가자와 보지 못한 공간 레이아웃에서도 강한 견고성과 일반화 능력을 보여줍니다. 이는 GIVE가 제스처를 통해 사람의 의도를 더 잘 연계하고, 동적인 상호작용 의도에 적응할 수 있도록 하는 솔루션임을 시사합니다.



### GeoHAT: Geometry-Adaptive Hybrid Action Transformer for Mobile Manipulation (https://arxiv.org/abs/2606.13394)
- **What's New**: 이 논문에서는 GeoHAT라는 새로운 딥러닝 프레임워크를 제시합니다. 이 프레임워크는 전체 신체 이동 조작 전체 인지 과정에서 발생하는 도전 과제를 해결하기 위해 설계되었습니다. GeoHAT는 신뢰할 수 있는 지점을 중심으로 기하학을 통합하여 시각적 정보를 보다 풍부하게 만들어줍니다.

- **Technical Details**: GeoHAT는 가벼운 Fourier 공간 인코더를 활용하여 밀도 높은 3D 좌표를 기하학적 토큰으로 변환합니다. 신뢰도 인식 게이팅을 통해, 각 토큰은 depth 신뢰도에 의해 조정된 게이트를 통해 비주얼 모델에 통합됩니다. 다양한 조작 기능을 위해 Hybrid Whole-Body Action Decoder가 사용되어 팔과 베이스 각각의 서브스페이스로 분리하여 동작처리를 수행합니다.

- **Performance Highlights**: 실험 결과, GeoHAT는 79.3%의 평균 성공률을 기록하며, 기존의 최강 기준선보다 23.7% 개선된 성과를 보였습니다. 또한, 다양한 현실 세계의 작업에서도 일관된 성과 개선을 확인하였습니다. 이로 인해 GeoHAT는 변화하는 시각적 환경에서 효과적으로 작동하는 능력을 갖춘 것으로 나타났습니다.



### Real-Time Execution with Autoregressive Policies (https://arxiv.org/abs/2606.13355)
- **What's New**: 본 논문에서는 비동기 추론(asynchronous inference)을 통해 오토회귀 정책(autoregressive policies)이 실시간 실행(real-time execution)을 가능하게 한다는 점을 보여줍니다. 이는 기존의 확산 정책(difussion policies)에 비해 오토회귀 정책이 상대적으로 느린 롤아웃 속도(slow rollout speed)를 해결할 수 있는 가능성을 여는 것입니다. 또한, 채택한 방법론은 동작 지평선(action horizon)을 조정하고 제약된 디코딩(constrained decoding)을 적용하여 성능 극대화를 이끌어냅니다.

- **Technical Details**: 연구에서는 오토회귀 정책의 성능을 극대화하기 위해 여러 가지 방법론을 도입하였습니다. 첫 번째로, H=2m으로 설정된 충분한 동작 지평선을 선택하여 디코딩할 토큰 수를 최소화합니다. 두 번째로, 전체 동작 지평선 대신 각 mm 지평선에서 토큰화를 적용하여 실행 대기열(action queue)에 대해 동작 조각을 조건화합니다. 마지막으로, 다중 궤적 디코딩(multi-trajectory decoding)을 채택하여 성능을 최대화하는 데 기여합니다.

- **Performance Highlights**: 실험 결과, 제안된 π₀-REALFAST 정책은 동시 추론(synchronous inference)에서의 동작 조각(real-time action chunking) 기반의 정책과 비교하여 실시간 실행에서 대폭 향상된 과제 성공률(task success rates)과 롤아웃 속도를 보였습니다. 이는 원래 동기화 추론에서 관찰된 오토회귀 정책의 장점이 실시간 실행에서도 계속 유지됨을 나타내며, 오토회귀 정책이 실시간 실행을 지원할 수 있다는 강력한 동기를 제공합니다.



### Low cost, easily manufactured, highly flexible strain and touch sensitive fiber for robotics applications (https://arxiv.org/abs/2606.13352)
- **What's New**: 이번 연구에서는 기존의 비싼 로봇 스트레인(strain) 및 터치(touch) 센서를 대체할 수 있는 저렴한 전도성 섬유(fiber)를 소개합니다. 이 섬유는 상업적으로 구입할 수 있는 가벼운 재료들로 제작되며, 복잡한 제조 장비 없이도 빠르게 생산할 수 있습니다. 연구자들은 이 섬유를 저항형 스트레인 센서와 정전식(touch) 센서로 활용하는 다양한 응용 사례를 선보였습니다.

- **Technical Details**: 이 센서는 전도성 실(thread)과 실리콘 튜빙(silicone tubing)으로 만들어졌습니다. 전도성 실은 개별 아크릴 스테이플 섬유를 전기도금(electroplating)하여 제작하였고, 실리콘 튜빙은 높은 유연성과 탄성을 제공합니다. 성능 평가를 통해 저항 센서로의 특성과 정전식 센서로의 특성을 모두 파악할 수 있으며, 이는 상호작용을 의미하는 다양한 로봇 애플리케이션에서의 응용 가능성을 보여줍니다.

- **Performance Highlights**: 연구 결과, 개발된 센서는 유연성과 저전력 소비를 바탕으로 다양한 로봇 응용 분야에서 유용하게 활용될 수 있습니다. 특히, 이러한 센서는 뉴럴 네트워크를 통해 뇌-기계 인터페이스와 같은 혁신적인 기술에 통합될 수 있는 잠재력을 가지고 있습니다. 또한, 이 섬유 센서는 제조 scaling이 가능하고 비용이 저렴하여 DIY 제작자들에게 직접적인 도움이 될 것입니다.



### EMG-Based Adaptation of Anisotropic Virtual Fixtures for Robot-Assisted Surgical Resection and Dissection (https://arxiv.org/abs/2606.13340)
- **What's New**: 이 논문은 로봇 보조 복강수술에서 섬세한 작업인 절제(Resection)와 절개(Dissection)를 위한 적응형 지원 시스템 개발에 대해 다룹니다. 기존의 Virtual Fixtures는 고정된 기하학적 정의로 인해 수술 흐름이나 외과 의사의 즉각적인 의도에 적응할 수 없다는 한계가 있습니다. 이 연구는 이러한 한계를 극복하기 위해 새로운 구조의 anisotropic virtual fixture와 실시간으로 외과 의사의 의도를 모듈화하는 직관적인 제어 인터페이스를 제안합니다.

- **Technical Details**: 제안된 시스템은 EMG 신호를 통해 외과 의사의 의도를 추론하여 fixture의 기하학적 구조를 실시간으로 조정할 수 있습니다. 이는 외과 의사가 전완 근육을 수축함으로써 제약을 동적으로 확장하거나 해제할 수 있게 하여, 정밀한 안내 동작과 도구의 자유로운 재배치 사이의 원활한 전환을 가능하게 합니다. 새로운 제어 아키텍처는 임의의 사용자 선택 포인트에서 가상 질량 동역학의 일관된 초기화를 보장하는 동적 (dis-)engagement 메커니즘을 특징으로 합니다.

- **Performance Highlights**: 파일럿 사용자 연구의 실험 결과에 따르면, 제안된 방법은 작업 정확도와 움직임 일관성을 크게 향상시켰으며, 인지 부담, 노력 및 좌절감이 줄어드는 효과를 보였습니다. 이는 로봇 보조 수술 시스템이 외과 의사의 능력에 맞춰 보다 표준화된 결과를 생성할 수 있도록 도와줄 수 있음을 시사합니다. 이러한 성과는 기존의 정적 안전 경계 및 수술 모델에 의존하지 않고도 실시간으로 동적이며 적응적인 수술 지원이 가능하다는 것을 입증합니다.



### See Selectively, Act Adaptively: Dual-Level Structural Decomposition for Bimanual Robot Manipulation (https://arxiv.org/abs/2606.13279)
- **What's New**: 본 논문에서는 양손 로봇 조작(bimanual robotic manipulation)에서 시각 정보의 중요성이 작업 단계와 맥락에 따라 변하는 문제를 다룹니다. 기존의 단일화된 Vision-Language-Action (VLA) 정책이 다양한 시각 입력과 상호작용 패턴을 처리하는 데 어려움이 있음을 지적하며, 이를 개선하기 위한 Dual-Level Structural Decomposition(이중 레벨 구조 분해) 기반의 새로운 프레임워크를 제안합니다.

- **Technical Details**: 제안된 프레임워크는 View-Selective Visual Router를 통해 손목 시점의 기여도를 동적으로 조정하여 관련 시각 단서를 강조합니다. 또한, Interaction-Aware Action Mixture-of-Experts (MoE)를 이용하여 행동 생성을 조정된 경로와 팔별 경로로 분해하여 다양한 양손 상호작용 모드에 적응하도록 합니다. 이러한 기술적 접근은 양손 조작의 구조를 명확히 이해하고 반영하는 데 중점을 둡니다.

- **Performance Highlights**: 제안된 방법은 RoboTwin 2.0 시뮬레이션에서 여섯 가지 양손 조작 작업, 그리고 세 가지 장기 실제 작업에 대해 평가되었습니다. 결과적으로, 단일 기반(linear baseline)과 비교하여 시뮬레이션에서 27.7%, 실제 평가에서 43.3%의 평균 성공률을 개선하며, 모든 설정에서 단일 모듈 변형을 지속적으로 능가하는 성능을 보여주었습니다.



### Humor Style Drives Laughter, Topic Shapes Acceptability: Evaluating Bilingual Personal and Political Robot-Delivered AI Jokes (https://arxiv.org/abs/2606.13256)
Comments:
          Accepted in the 35th IEEE International Conference on Robot and Human Interactive Communication (RO-MAN 2026), Kitakyushu, Fukuoka, Japan

- **What's New**: 이번 연구는 로봇과의 상호작용(Interaction)에서 유머의 영향을 탐구하며, AI가 생성한 유머 유형 및 농담 내용이 사람들의 인식에 미치는 영향을 분석합니다. 특히, 사람-로봇 상호작용(HRI)에서 유머를 통합할 수 있는 새로운 기회를 모색하고 있습니다. 이를 통해 유머 스타일이 어떻게 사람들이 로봇의 유머를 받아들이는지를 밝혀내고자 하였습니다.

- **Technical Details**: 이 연구는 혼합 요인 설계(mixed factorial design)를 사용하여 진행되었으며, 참가자들은 대학 강의실에서 로봇이 전달하는 AI 생성 유머를 평가했습니다. 연구는 유머 유형(Affiliative, Self-Enhancing, Aggressive, Self-Defeating)과 농담 내용(인물 관련 vs. 정치적)이 재미와 적절성에 미치는 영향을 조사합니다. 참가자들의 자가 보고된 유창성(fluidity) 및 유머 관행 또한 언어 선호에 영향을 미칠 수 있습니다.

- **Performance Highlights**: 연구 결과에 따르면 유머 유형은 재미에 상당한 영향을 미치며, Aggressive와 Affiliative 유머가 더 높은 평가를 받았습니다. 반면, 농담 내용은 적절성에 주로 영향을 미치며, 인물 관련 농담이 정치적 농담보다 선호되는 것으로 나타났습니다. 이러한 결과는 사람-로봇 상호작용에서 유머의 역할을 더 잘 이해하는 데 기여할 수 있습니다.



### WT-UMI: Tactile-based Whole-Body Manipulation via Force-Supervised Contact-Aware Planning (https://arxiv.org/abs/2606.13232)
Comments:
          18 pages, 8 figures

- **What's New**: 이번 논문에서는 복잡한 형태의 객체를 조작하기 위해 개발된 Wearable Tactile Interface인 	extbf{WT-UMI}를 소개합니다. 이 장치는 인간 운영자가 착용하거나 휴머노이드에 장착할 수 있으며, 촉각 이미지, 접촉 force, 엔드 이펙터의 pose를 정확하게 관찰할 수 있는 기능을 제공합니다.

- **Technical Details**: WT-UMI는 두 가지 시연 소스, 즉 인간 시연과 로봇 원격 조작 모드를 통해 접촉 정보를 수집합니다. 이 논문에서는 정확한 로봇 목표를 설정하기 위해 force-conditioned target-pose correction module을 도입하고, force-supervised planner를 사용하여 최적의 엔드 이펙터 pose와 접촉 force 궤적을 예측합니다.

- **Performance Highlights**: 다섯 가지 접촉이 풍부한 작업을 통해 WT-UMI는 네 가지 정책 기준 대비 성공률을 향상시키고 접촉 위치 추적 오류를 줄였습니다. 이 연구는 복잡한 객체 조작 및 인간-휴머노이드 협업 속에서 확장성을 보여줍니다.



### Proprioceptive-visual correspondence enables self-other distinction in humanoid robots (https://arxiv.org/abs/2606.13222)
Comments:
          23 pages, 9 figures, 1 supplementary table

- **What's New**: 이번 연구에서는 휴머노이드 로봇이 정체성 레이블이나 운동학 모델 없이도 프로프리오셉션-비주얼 상관관계(proprioceptive-visual correspondence)를 통해 자기와 타자를 구별할 수 있다는 점을 보여줍니다. 이러한 자기-타자 구분(self-other distinction)은 로봇이 조정된 자기 모델을 학습하는 기초가 되며, 이를 통해 로봇이 다가오는 목표를 향한 목표 도달(target reaching) 및 충돌 인식 모션 계획(collision-aware motion planning)와 같은 다운스트림 작업을 지원할 수 있습니다. 연구의 결과는 로봇이 공유 환경에서 타인과 조화를 이루며 행동할 수 있는 신체적 자기 표현(body representation)의 가능성을 제시합니다.

- **Technical Details**: 본 연구는 29 자유도(DoF)의 휴머노이드 로봇을 이용하여 구현되었으며, 로봇은 프로프리오셉션 상태와 시각적 관찰만으로 자기 신체를 식별할 수 있습니다. 각 후보 마스크는 시간적으로 일치하는 프로프리오셉션 상태와의 관계를 기반으로 하여 네트워크 모델에 유입되는 신호를 생성합니다. 특정 후보 마스크가 로봇의 신체에 해당한다고 가정하는데, 이 과정은 단순한 강화 학습 없이도 이루어집니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 로봇이 타인과 함께 있는 복잡한 환경에서도 자기 신체를 효과적으로 식별하는 능력을 보여주었습니다. 로봇은 실험에서 99.5% 이상의 정확도로 자신의 신체를 식별하며, 이는 일반적인 시각-언어 모델(Vision-Language Model)이나 단순 평균 모델보다 월등히 높은 성능입니다. 제안된 접근 방식은 현재 VLM이 특정 조인트 각도를 가시적인 신체와 연결하는 데 한계가 있음을 강조하며, 로봇이 자신의 신체를 경험을 통해 학습할 수 있는 새로운 가능성을 제공합니다.



### Embedding ISO 10218 Safety Compliance in Robots via Control Barrier Functions for Human-Robot Collaboration (https://arxiv.org/abs/2606.13203)
- **What's New**: 인간-로봇 협업(Human-Robot Collaboration, HRC)에서는 안전 기준인 ISO 10218을 철저히 준수해야 합니다. 본 논문은 인간의 가속 데이터를 명시적으로 활용하는 제어 장벽 함수(Control Barrier Function, CBF)를 제안하여 최악의 로봇 정지 궤적에서 최소한의 인간-로봇 분리 거리를 예측합니다. 이는 기존의 안전 필터 시스템의 한계를 극복하고, 로봇의 속도 조절을 통해 작업 효율성을 극대화할 수 있도록 합니다.

- **Technical Details**: 이 연구에서는 제어 장벽 함수가 Sequential Quadratic Programming (SQP) 프레임워크 내에서 부등식 제약조건으로 통합되어 안전성을 보장하도록 설계되었습니다. 두 가지 방법이 제안되었는데, 방식 I은 CBF로 제약된 PD 안전 필터이고, 방식 II는 작업 스케일링 SQP 제어기로 공간적 튜브 제약을 강제합니다. 이러한 제어 방법은 로봇의 안전성을 확보하면서도 효율성을 높히는 방향으로 나아가고 있습니다.

- **Performance Highlights**: 시뮬레이션 및 실제 실험을 통해 제안된 방법들이 기존의 산업 표준 SSM 모듈과 비교하여 성능을 평가했습니다. 결과적으로 방식 II는 평균 궤적 오류를 63% 줄이었으며, 지나치게 회피 행동을 피하면서도 ISO 10218 SSM 가이드라인을 준수하여 높은 작업 처리량을 달성했습니다.



### Multi-Modal Multi-Agent Robotic Cognitive Alignment enabled by Non-Invasive Consumer Brain Computer Interfaces: A Proof of Concept Exploration (https://arxiv.org/abs/2606.13190)
Comments:
          19 pages, 9 figures, for associated video, see this https URL

- **What's New**: 이 연구는 기존의 인간-로봇 상호작용에서 무시되는 인간의 내부 인지 상태에 초점을 맞춥니다. 새로운 프레임워크인 'cognitively aligned' 상호작용을 통해 로봇 시스템이 인간의 인지 부하가 높은 순간에 소통을 지연시킬 수 있는 능력을 강화합니다. 이는 사용자에게 더 나은 경험을 제공하기 위한 혁신적인 접근입니다.

- **Technical Details**: 연구에서는 자율 작업 실행과 실시간 신경 생리학적 집중력 사이의 상호작용을 탐구하는 폐쇄 루프 아키텍처를 설계하고 구현합니다. 소비자 등급의 Brain-Computer Interface (BCI)를 사용하여 Electroencephalography (EEG) 스펙트럼 대역의 파워를 지속적으로 모니터링합니다. 엔게이지먼트(engagement)가 높은 순간을 감지했을 때 메인 에이전트의 감각 입력과 음성 출력을 대기 상태로 전환하는 HTTP 기반 신호 메커니즘을 제안합니다.

- **Performance Highlights**: 초기 결과는 실시간 신호 처리, Large Language Models (LLMs), 물리적 로봇 구현을 활용하여 인지적으로 인식할 수 있는 비간섭형 다중 에이전트 시스템을 생성하는 것이 가능함을 입증합니다. 이 접근 방식은 복잡한 작업을 원활하게 처리하고, 인지 부하가 낮은 상태로 돌아왔을 때 메시지를 전달하는 방식으로 성능을 크게 향상할 수 있음을 보여줍니다.



### Redesigning Regularization for Effective Policy Smoothing (https://arxiv.org/abs/2606.13169)
Comments:
          submitted to RA-L

- **What's New**: 이 논문은 강화 학습(reinforcement learning)에서 정책 함수(policy function)를 효과적으로 매끄럽게 하는 새로운 정규화 디자인을 제안합니다. 기존의 두 가지 접근 방식이 있었지만, 이 논문에서는 그 구현에서의 문제점을 지적하고, 이 문제에 대한 해결책을 제시합니다. 새롭게 수정된 L2C2 정규화는 기존 방법보다 더 나은 매끄러움을 제공하면서도 성능을 향상시키는 데 성공했습니다.

- **Technical Details**: 논문에서는 정책의 매끄러움과 표현성 사이의 상충관계(trade-off)를 다루고, 이를 해결하기 위해 세 가지 구현 문제를 제기합니다. 구체적으로는 상태 공간에서의 거리 함수 설계, 정책 간의 발산 선택, 그리고 배치 손실의 스칼라화 문제를 다루었습니다. 새로운 L2C2 버전인 'L2C2-v2'는 이러한 수정을 통해 세 가지 벤치마크 작업에서 더 나은 성능을 발휘하며, 사족 로봇의 운동 제어에 적용되어 쉽게 매끄러운 움직임을 구현합니다.

- **Performance Highlights**: L2C2-v2를 사용해 강화 학습을 통한 사족 로봇의 운동 제어를 적용했을 때, 목표 속도 명령의 급변에 대한 저항력을 보여줍니다. 이 논문은 강화 학습 알고리즘을 활용한 여러 테스트에서 제어 성능 개선을 입증하여, 교차 적용성에 대한 잠재력을 제시합니다. 뿐만 아니라, 제안된 방법은 실제 시스템에서 정책을 매끄럽게 하여 환경의 급변에도 잘 적응할 수 있도록 합니다.



### FTP-1: A Generalist Foundation Tactile Policy Across Tactile Sensors for Contact-Rich Manipulation (https://arxiv.org/abs/2606.13102)
- **What's New**: 이번 연구에서는 FTP-1이라는 최초의 일반화된 기초 촉각 정책이 소개됩니다. 이 모델은 다양한 센서 및 로봇 구현체에서 촉각 조작 능력을 전이할 수 있는 사전 훈련된 모델로, 여러 촉각 입력 형식을 지원합니다. 이를 통해 기존의 제한된 센서 설정에서 벗어나 다양한 촉각 경험을 흡수하고 전이할 수 있는 가능성을 제시합니다.

- **Technical Details**: FTP-1은 형태 인식 촉각 토큰 공간(MTTS)을 도입하여 다양한 센서의 촉각 신호를 통합하는 통합 인터페이스 역할을 합니다. 이 모델은 이미지, 배열 및 상태 기반의 다양한 센서 입력을 처리하여 이들 각각을 MTTS로 투영합니다. 촉각 데이터의 상이성을 해결하기 위해 FTP-1은 전문화된 다중 전문가 아키텍처를 채택하여 시각-언어 인식과 촉각 토큰의 융합을 통한 행동 생성을 수행합니다.

- **Performance Highlights**: FTP-1은 3가지 기존 센서 설정에서 +17.2%의 성공률 향상을 달성했으며, 이전에 본 적 없는 두 개의 촉각 센서 설정으로도 전이하여 +31%의 성공률 개선을 이끌어 냈습니다. 이 연구는 FTP-1이 다양한 센서 및 구현체에서 촉각 조작 정책 훈련을 위한 공유된 기초 모델 수준의 시작점을 설립함을 보여줍니다.



### EA-WM: Event-Aware World Models with Task-Specification Grounding for Long-Horizon Manipulation (https://arxiv.org/abs/2606.13053)
- **What's New**: 본 논문에서는 EA-WM(Event-Aware World Model)라는 새로운 프레임워크를 소개합니다. EA-WM은 고정된 시각적 특징 동역학에 작업 사양에 기반한 사건 예측 및 검증을 추가하여 로봇의 상상력을 향상시킵니다. 이러한 접근법은 로봇이 다양한 작업을 수행함에 있어 더 높은 신뢰성을 보장하도록 돕는 새로운 메커니즘을 제공합니다.

- **Technical Details**: EA-WM은 시각적 특징 세계 모델을 기본 상상 엔진으로 사용하고, 그 위에 작업에 기반한 사건 예측 레이어를 추가합니다. 후보 행동 시퀀스마다 기본 모델이 미래 시각적 특징을 예측하고, 사건 예측기가 상상된 미래를 작업 관련 사건으로 매핑합니다. 이 후 검증자는 예측된 사건 상태가 작업 진행, 의미 일관성, 물리적 타당성 및 충분한 확실성을 나타내는지를 평가합니다.

- **Performance Highlights**: EA-WM은 실험을 통해 로봇의 과제 수행 능력 향상에 기여한다는 것을 입증하였습니다. 예를 들어, PointMaze에서 성공 확률이 0.90에서 0.94로 개선되었으며, Deformable 및 Wall-Single 작업에서도 각각 94% 및 95%의 성공률을 기록했습니다. WINE-RACK에서의 연구 결과는 97/100의 온라인 하이브리드 성공률을 보여줍니다.



### Y-BotFrame: An Extensible Embodied Agent Framework for Quadruped Robot Assistants (https://arxiv.org/abs/2606.13049)
- **What's New**: 이 논문에서는 로봇을 지능형 지상 보조자로 변모시키는 확장 가능한 플랫폼인 Y-BotFrame을 소개합니다. Y-BotFrame은 음성, 비전(vision), LiDAR를 포함한 다중 모드 인식 기능을 통합하고, 환경 인식, 문맥 추론, 작업 계획을 위한 인지 코어로 대형 언어 모델(LLM)을 활용합니다. 이를 통해 로봇은 자연어 명령을 실행 가능한 작업 단위로 변환하고, 원격 제어 없이도 효율적인 인간-로봇 협력을 지원합니다.

- **Technical Details**: Y-BotFrame은 계층적 플랫폼으로 설계되어 하위 수준에서 내비게이션, 환경 인식 및 질문 응답 기능이 분명하게 정의된 입력 및 출력 인터페이스를 가진 모듈로 캡슐화됩니다. 상위 수준에서는 LLM 기반 에이전트가 사용자 자연어 명령, 역사적 상호작용 기록 및 환경 정보를 분석하여 구조화된 작업 계획을 생성합니다. 이 구조적 접근 방식은 대규모 task-specific 데이터에 대한 모델의 강한 의존성을 줄이고, 시스템의 해석 가능성, 확장성 및 안정성을 향상시킵니다.

- **Performance Highlights**: Y-BotFrame의 실행 모듈은 ASR(Automatic Speech Recognition), TTS(Text-to-Speech), LiDAR, GNSS, IMU 등 다양한 기술을 활용하여 자율 내비게이션과 사람-로봇 질문 응답을 지원합니다. 로봇은 고정밀 지도 생성과 경로 계획을 실시간으로 수행하며, 사용자 요청을 처리하면서 대화형 응답을 제공합니다. 이 시스템은 로봇에서 실행 또는 원격 서버에 배포되며, 연산 부하에 따라 분산 설계되어 배포 유연성과 운영 효율성을 높입니다.



### RoboProcessBench: Benchmarking Process-Aware Understanding in Vision-Language Robotic Manipulation (https://arxiv.org/abs/2606.13040)
- **What's New**: 이번 연구는 로봇 조작에서 비전-언어 모델(VLMs)의 프로세스 인식 능력을 평가하기 위해 RoboProcessBench라는 벤치마크를 제안합니다. 기존의 평가 방식이 VLM이 가진 세밀한 프로세스 이해를 검증하지 못하는 문제를 해결하고자 이 벤치마크는 정적 모니터링(static monitoring)과 동적 추론(dynamic reasoning)이라는 두 가지 차원으로 프로세스 인식을 분해했습니다.

- **Technical Details**: RoboProcessBench는 260개의 조작(task)에서 약 58,000개의 질문-답변 쌍을 포함하는 ProcessData라는 데이터셋을 구축하였습니다. 이 데이터는 물리적으로 기반한 실행 추적을 바탕으로 하며, 정적 및 동적 신호에 따라 12개의 진단 질문 항목으로 구성되어 있습니다. 
또한, ProcessData는 ProcessData-SFT와 ProcessData-Eval로 나뉘어 후속 훈련 및 평가 목적으로 사용됩니다.

- **Performance Highlights**: RoboProcessBench의 평가 결과, 여러 VLMs가 12개의 진단 작업군에서 광범위한 한계를 보여주었으며, 현재 모델은 여전히 조작 실행에 대한 충분한 프로세스 인식이 부족합니다. 그러나, ProcessData-SFT를 통해 훈련된 Qwen2.5-VL-7B와 InternVL-3-8B 모델은 지역 상태, 동작, 진행 및 원시 신호 인식에서 일관된 향상을 보였습니다.



### Comparing Commercial Depth Sensor Accuracy for Medical Applications (https://arxiv.org/abs/2606.13028)
Comments:
          4 Pages

- **What's New**: 이 논문은 깊이 추정(depth estimation) 기술의 적용 가능성을 높이기 위해 다양한 깊이 센서를 평가합니다. 연구에서는 돼지 뼈, 돼지 배, 실리콘 신장 팬텀을 사용하여 센서를 비교했습니다. 이들 객체는 균질(surface) 및 반사(Specular) 표면, 그리고 서브서피스(Subsurface) 산란과 같은 다양한 실제 도전 요소를 포함하고 있습니다.

- **Technical Details**: 본 연구는 약 50cm 거리에서 스테레오(stereo), 구조광(structured-light), 타임오브플라이트(time-of-flight) 센서를 비교합니다. 비교 대상으로는 인텔 리얼센스 D405, PMD Flexx2, 스테레올랩스 ZED 2i, 지비드 2M+ 60가 포함됩니다. 특히, 각 센서는 돼지 조직과 팬텀에서의 성능을 측정하여 발표되었습니다.

- **Performance Highlights**: 결과적으로, 지비드 2M+ 60이 모든 객체와 메트릭에서 최상의 성능을 보여주었습니다. 스테레올랩스 ZED는 실제 조직에서는 두 번째로 높은 순위를 기록했지만, 팬텀에서는 가장 낮은 성능을 나타냈습니다. 이러한 결과는 깊이 센서 선택 시 고려해야 할 중요한 요소들을 제시합니다.



### GenHOI: Contact-Aware Humanoid-Object Interaction by Imitating Generated Videos without Task-Specific Training (https://arxiv.org/abs/2606.12995)
- **What's New**: 이 논문에서는 GenHOI라는 새로운 프레임워크를 제안하여, 휴머노이드 로봇이 단일 생성된 비디오를 직접 모방함으로써 다양한 객체 상호 작용 작업을 제로 샷(zero-shot) 방식으로 수행할 수 있도록 합니다. 기존의 방법들은 일반적으로 특정 작업에 대한 훈련이나 물리적 시연 데이터를 요구하는 반면, GenHOI는 이러한 요구사항을 최소화하여 더욱 효율적인 학습이 가능하도록 합니다.

- **Technical Details**: GenHOI 프레임워크는 네 가지 주요 모듈로 구성되어 있습니다: 실시간 비디오 생성(real-to-sim video generation), 접촉 인식 기반 기하학적 제약(contact-aware geometric constraint extraction), 기하학적 안내를 통한 경로 최적화(geometry-guided trajectory optimization), 그리고 폐쇄 루프 경로 추적(closed-loop trajectory tracking)입니다. 이 시스템은 생성된 비디오에서 상호작용 관련 접촉 이벤트를 식별하고 손-객체 접촉 영역을 추정하여 물리적으로 기반을 둔 최적화 우선 규칙을 사용하여 경로를 최적화합니다.

- **Performance Highlights**: 이 논문에서 제안한 GenHOI 프레임워크는 다양한 객체 상호작용 작업에 대해 광범위한 시뮬레이션 및 실제 실험을 통해 검증되었습니다. 결과적으로, 이 방법은 기존 모델보다 더 효율적으로 다양한 객체 상호작용 행동을 학습하며, 강력한 일반화 능력을 보여주었습니다.



### Trajectory-Level Redirection Attacks on Vision-Language-Action Models (https://arxiv.org/abs/2606.12978)
- **What's New**: 이 논문에서는 비전을 통한 언어-행동(Vision-Language-Action, VLA) 정책이 로봇 제어에 자연어를 통합하는 방식을 다루고 있습니다. 특히, 고정된 정책에서 명령을 유지하면서 행동을 유도하는 텍스트의 변형에 대한 새로운 위협 모델인 'command-preserving trajectory redirection'을 제시합니다. 이 연구는 적대적인 프롬프트가 의도한 작업을 지정하는 듯 보이지만 최종 물리적 결과를 조작할 수 있는 가능성을 제기합니다.

- **Technical Details**: VLA 정책은 언어 신호를 피드백 제어에 지속적으로 사용하여 로봇이 주어진 텍스트 지침에 따라 조작 작업을 수행할 수 있도록 합니다. 본 연구에서는 고정된 VLA 정책을 기준으로 근본적인 공격을 이해하기 위해 텍스트 지침만을 조작하여, 최종적으로 로봇이 수행하게 될 작업을 전환하는 방법을 발견했습니다. 이를 위해 제안된 알고리즘은 롤아웃(rollouts)을 통해 변형된 프롬프트를 사용하여 명령을 유지하는 맞춤화를 수행합니다.

- **Performance Highlights**: 모의 실험과 하드웨어 테스트 결과에 따르면, 거의 무해한 프롬프트 변형이 VLA 롤아웃을 공격자가 지정한 목표로 우회하도록 유도할 수 있음을 보여주고 있습니다. 이러한 결과는 로봇을 제어하는 과정에서 의도된 명령을 보존하는 듯 보이는 텍스트가 실제로는 적대자의 조작을 허용할 수 있음을 드러냅니다. 이 연구는 VLA 정책의 안전성에 대한 새로운 관점을 제시하며, 주의가 필요한 잠재적 취약점을 조명합니다.



### EmbodiSteer: Steering Embodiment-Agnostic Visuomotor Policies with Joint-Space Guidance for Zero-Shot Cross-Embodiment Deploymen (https://arxiv.org/abs/2606.12965)
Comments:
          The first two authors contribute equally

- **What's New**: 이번 논문에서는 로봇 모방 학습에서 체감 불가능한 정의에 따른 정책 학습의 한계를 극복하기 위해 EmbodiSteer라는 훈련 없는 프레임워크를 제안하였다. 이 방법은 Cartesian 종료 조작만으로 훈련된 정책을 그대로 활용하면서도, 추론 시 로봇의 조인트 공간으로 이동할 수 있게 해준다. 이로 인해 로봇의 특정 제약을 충족하면서도 부딪힘을 피하고, 성공적인 작업 차수를 증가시킨다.

- **Technical Details**: EmbodiSteer는 훈련 않은 추론 프레임워크로서, Cartesian 차원에서 조작을 진행하며, 특정 로봇의 조인트 공간으로 효율적으로 전환한다. 이는 전진 기하학 (forward kinematics)과 자코비안 기반의 업데이트를 통해 달성된다. 각 역순화 단계 후에는 전체 몸체 충돌 방지를 고려하여 조인트 궤적에 대한 안내를 제공하며, 이를 통해 충돌을 피하면서도 종료 조작 동작을 유지할 수 있다.

- **Performance Highlights**: 실험 결과는 EmbodiSteer가 Cartesian 방식만으로 실행했을 때보다 전체 몸체 장애물 회피율을 46.1% 향상시키고, 작업 성공률을 28.5% 증가시킨다고 나타났다. 실제 환경의 두 로봇에서도, 90.0%의 충돌 감소와 36.7%의 작업 성공률 증가를 확인하며, 이는 로봇 특화된 미세 조정 없이도 가능했다. 이러한 성과는 EmbodiSteer가 다양한 환경에서 제약을 고려한 안전한 실행을 가능하게 함을 보여준다.



### SERF: Spatiotemporal Environment and Robot Feature Map for Long-Horizon Mobile Manipulation (https://arxiv.org/abs/2606.12956)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 로봇의 이동을 위한 모바일 조작에서 긴 수평선(horizon) 동안의 일관된 reasoning을 향상시키기 위한 새로운 spatiotemporal feature map인 SERF(Spatiotemporal Environment and Robot Feature)를 제안합니다. 이 map은 로봇의 환경과 신체를 공유된 latent 공간에서 신경 포인트(neural points)로 나타내며, 이는 에고 중심 관찰(egocentric observations)과 부가적인 상태(proprioceptive state)에 의해 온라인으로 업데이트됩니다.

- **Technical Details**: SERF 맵은 환경과 로봇이 상호작용하면서 진화하는 spatiotemporal(4D) feature mapping 공식을 수립하여 로봇과 환경 간의 공간적 관계를 명시적으로 표현합니다. 이는 allocentric reasoning을 통해 전역 장면에서 로봇을 위치시킬 수 있는 지속적인 공간 메모리를 제공하며, egocentric reasoning을 통해 로봇의 환경와의 관계를 지원합니다. 이 맵은 시각-언어-행동 모델(VLA)에 입력으로 사용되어 정책에 지역적(local) 및 전역적(global) 문맥을 제공합니다.

- **Performance Highlights**: SERF VLA 정책은 BEHAVIOR-1K 벤치마크에서 이미지 기반 VLA 기준선과 비교하여 평균 과제 진행률을 44.0%에서 58.7%로 향상시키며, 더 직접적인 경로를 따라 더 빠르게 하위 목표(subgoals)에 도달하고, 장면 구성의 변화에 대한 강건성을 개선합니다. 또한, 실행 오류로부터 회복하여 떨어진 물체를 회수하는 등의 성능을 보여주었습니다.



### Towards Reliable Sequential Object Picking in Clutter: The Runner-up Solution to RGMC 2025 (https://arxiv.org/abs/2606.12954)
Comments:
          First, Second and Third Coauthor contributed equally to this work

- **What's New**: 이 논문은 산업 현장에서 중요한 문제인 혼잡한 환경에서의 안정적이고 효율적인 그랩(grasping)을 다루고 있습니다. 특히, 물체를 연속적으로 선택(picking)해야 하는 도전적인 작업에 초점을 맞추고 있으며, Cluttered Environment Picking Benchmark (CEPB)를 기반으로 한 해결책을 제시합니다.

- **Technical Details**: 이 연구는 다양한 물체를 포함하는 복잡한 환경에서 높은 성공률을 유지하면서 강력하고 충돌 인식이 가능한 그랩을 요구합니다. 이를 위해 객체 인식(object recognition), 디클러터링(decluttering), 그리고 다중 모드 그랩(multi-modal grasping)을 결합한 통합 하드웨어-소프트웨어 파이프라인을 설계하였습니다.

- **Performance Highlights**: 연구는 실험실 테스트 및 경쟁 시나리오에서 효율적인 인식 및 연속적 물체 그랩을 성공적으로 수행하여 RGMC 2025의 Pick-in-Clutter 트랙에서 2위를 차지했습니다. 이 과정에서 다기능 그리퍼와 혼잡 공간에서의 물체 분포 및 가림 관계를 위한 새로운 표현 방식이 주요 기여로 포함됩니다.



### An Embodied Simulation Platform, Benchmark, and Data-Efficient Augmentation Framework for Wet-Lab Robotics (https://arxiv.org/abs/2606.12936)
Comments:
          25 pages, 17figures

- **What's New**: 이번 논문에서는 생물의학 실험의 재현성, 처리량, 그리고 안전성을 개선할 수 있는 Wet-lab 로봇을 위한 새로운 플랫폼인 Pipette를 소개합니다. Pipette는 사용자 맞춤형 시뮬레이터, 개방형 편집 가능한 실험실 자산, 그리고 제한된 시연을 사용 가능한 교육 데이터로 전환하는 효율적인 파이프라인을 제공합니다. 특히, Pipette는 43개 이상의 오픈 소스 자산을 제공하며, 사용자가 쉽게 수정할 수 있습니다.

- **Technical Details**: Pipette의 핵심 구성 요소는 시뮬레이션 기반 데이터 증강 파이프라인입니다. 이 시스템은 인간 시연을 시뮬레이션에서 재생하며, 조명, 카메라, 속도, 행동의 변동을 적용하고 자동 작업 성공 체크를 통해 생성된 에피소드를 필터링합니다. 이를 통해 제한된 수의 수동 시연에서 사용 가능한 훈련 데이터를 신속히 확장할 수 있습니다.

- **Performance Highlights**: Pipette는 11개 Wet-lab 작업의 기준선을 설정하고, 샘플 처리, 배양기구 조작, 장비 작동 및 정밀 배치를 포함하는 다양한 작업을 평가합니다. 각 작업당 단 30회의 시연으로 평균 65.5%의 성공률을 기록했으며, 시뮬레이션 증강이 SmolVLA의 성공률을 44.1%에서 74.7%로, 그리고 {C0}0의 성공률을 40.4%에서 46.5%로 향상시켰습니다. Pipette는 또한 비전문가 사용자도 새로운 Wet-lab 로봇 작업을 정의할 수 있도록 자연어 기반의 장면 구성 및 작업 등록을 지원합니다.



### Bounding Boxes as Goals: Language-Conditioned Grasping via Neuro-Symbolic Planning (https://arxiv.org/abs/2606.12910)
Comments:
          Project website: this https URL

- **What's New**: GRASP (Grounded Reasoning and Symbolic Planning) 프레임워크는 자연어 지시를 해석하고 이를 미리 훈련된 VLM (Vision-Language Model)을 통해 신경-상징적 목표 상태로 변환하는 방법을 제안합니다. 기존의 고정된 색상 목록이나 하드코딩된 좌표에 의존하지 않고 로봇이 '상단 선반(top shelf)'과 같은 추상적인 공간 개념을 이해하고 추가적인 미세 조정 없이 작업을 수행할 수 있게 합니다. 이는 가벼운 neuro-symbolic 시스템을 통해 언어 조건 조작의 효율성을 높입니다.

- **Technical Details**: GRASP 시스템은 자연어 프롬프트를 해석하기 위해 LLM (Large Language Model)과 GroundingDINO와 같은 VLM을 사용합니다. 시스템은 사용자로부터 입력된 명령을 JSON 파일로 저장한 객체 경계 상자로 변환하고, 객체의 위치와 목표 상태를 연결하는 두 가지 컴포넌트를 포함합니다. 이를 통해 GRASP는 로봇의 작업을 지속적으로 추적하고 조정하여 높은 효율성을 유지합니다.

- **Performance Highlights**: GRASP는 90개의 실제 로봇 시험에서 전반적으로 73.3%의 성공률을 기록하며, 태스크 특정 훈련 없이 세 가지 난이도에서 작업을 수행할 수 있음을 입증합니다. 이러한 성능 향상은 새로운 사용자 명령에 대한 일반화 능력이 뛰어나다는 것을 보여주며, 태스크의 복잡성에 관계없이 로봇의 조작을 가능하게 합니다.



### Learning to Adapt: Representation-Based Reinforcement Learning for Multi-Task Skill Transfer (https://arxiv.org/abs/2606.12890)
Comments:
          8 pages, 4 figures, 1 table

- **What's New**: 이번 연구에서는 multi-task reinforcement learning (RL)을 위한 RepMT-SAC라는 새로운 프레임워크를 제안합니다. 이는 효율적인 지식 공유와 새로운 작업에 대한 강력한 전이를 가능하게 합니다. RepMT-SAC는 spectral MDP decomposition을 활용하여 전이 가능한 동력을 포착합니다.

- **Technical Details**: RepMT-SAC는 모든 작업이 동일한 동적 시스템을 공유하되, 보상 함수나 레퍼런스 목표가 다르다는 점에 초점을 맞춥니다. 이 연구에서는 다수의 Markov Decision Processes (MDPs)로 구성된 multi-task 환경을 설정하고, task-conditioned MDP 프레임워크를 소개합니다. 이를 바탕으로, 각 작업에 특화된 보상과 환경의 구조를 분리하는 방법을 논의합니다.

- **Performance Highlights**: RepMT-SAC는 쿼드콥터 트래젝토리 추적 작업을 통해 평가되었으며, in-distribution 작업에서의 제로샷 성능과 out-of-distribution 작업에 대한 신속한 적응 능력을 입증합니다. 실험 결과, RepMT-SAC는 SAC 및 CTRL-SAC와 비교하여 최대 30% 향상된 성과를 보였습니다.



### AIR-VLA+: Decoupling Movement and Manipulation via Cascaded Dual-Action Decoders with Asymmetric MoE for Aerial Robots (https://arxiv.org/abs/2606.12859)
- **What's New**: 본 논문에서 제안하는 AIR-VLA+ 아키텍처는 비대칭 feature-level Mixture of Experts (MoE) 구조와 두 개의 연속된 action decoder로 구성되어 있습니다. 이를 통해 드론(Unmanned Aerial Vehicle, UAV)의 큰 이동과 조작 장치의 미세 조작 간의 복잡한 동작을 독립적으로 처리할 수 있습니다. 이는 드론과 조작기 간의 행동 레벨 디커플링을 통해 작업 효율성을 극대화합니다.

- **Technical Details**: AIR-VLA+ 시스템은 UAV의 이동 제어를 위한 input feature enhancement 모듈을 포함하고 있어, 그립퍼와 물체 간의 상호작용 상태를 인식하는 데 도움을 줍니다. 이를 통해 UAV의 행동이 task state와 연계될 수 있도록 하며, 이를 기반으로 한 정보 전달 방식은 조작과 이동 간의 조화를 더욱 강화합니다. 비대칭 MoE 아키텍처를 통해 다양한 작업 단계에서 전문가의 능력 기울기를 발휘할 수 있는 구조로 설계되어 있습니다.

- **Performance Highlights**: AIR-VLA+ 모델은 표준화된 AIR-VLA 벤치마크에서 모든 기준선 모델을 초월하는 성능을 보였으며, 평균 점수는 48.0으로 기록되었습니다. 특히, 작업 완료 점수는 기존 방법에 비해 80.2% 개선되었으며, 이는 복합 로봇의 조정 제어 갈등을 효과적으로 완화했음을 나타냅니다.



### Stubborn: A Streamlined and Unified Reinforcement Learning Framework for Robust Motion Tracking and Fall Recovery for Humanoids (https://arxiv.org/abs/2606.12814)
- **What's New**: 최근 강화학습(Reinforcement Learning) 기반의 새로운 방법인 Stubborn이 소개되었다. 이 방법은 인간형 로봇의 움직임 추적 성능을 향상시키고 불안정한 상태에서의 낙상 회복을 달성하는 데 중점을 두고 있다. 기존 방법들이 추적과 회복을 별개의 작업으로 처리했던 반면, Stubborn은 이를 통합하여 단일 정책으로 강인한 동작 추적과 회복을 동시에 학습하게 설계되었다.

- **Technical Details**: Stubborn은 비대칭 액터-크리틱(Asymmetric Actor-Critic) 아키텍처를 사용하고, 세 가지 주요 구성 요소로 이루어져 있다. 첫째, 항법 안정성을 개선하기 위한 요 정렬 추적 표현(yaw-aligned tracking representation)이 도입되었고, 둘째, Bernoulli 기반의 확률적 종료 메커니즘이 적용되어 다양한 실패 상황에서의 회복 행동 탐색을 장려한다. 셋째, 추적 오류 기반의 샘플링 전략이 동적으로 샘플링 분포를 조정하여 학습의 효율성을 높인다.

- **Performance Highlights**: Stubborn은 최신의 방법들(SOTA)과의 광범위한 비교 및 제거 연구(ablation studies)에서 경쟁력 있는 성능을 발휘하였다. 특히, 제안된 확률적 종료 메커니즘과 적응형 샘플링 전략이 성능 및 강인성에 크게 기여한 것으로 나타났다. 실제 로봇 실험에서도 Stubborn의 유효성이 입증되었다.



### Sparse2Act: Learning Action-Aligned Sparse 3D Representations for Cross-Domain Robot Manipulation (https://arxiv.org/abs/2606.12759)
- **What's New**: 이 논문은 Sparse2Act라는 새로운 관찰-행동 정렬 프레임워크를 도입하여 희소 3D 포인트 클라우드 인코더를 사전 학습하는 방법을 제안합니다. 구체적으로, 작업 공간의 끝단 효과기 행동(task-space end-effector actions)을 기하학적 감독으로 사용하여 장면 기능을 조직화하는 희소 3D 토큰을 교육합니다. 이를 통해 인코더는 다운스트림 정책과 독립적으로 초기화되며 자신의 행동 공간을 유지할 수 있습니다.

- **Technical Details**: Sparse2Act는 인코더 레벨에서 감독을 통해 희소 3D 인코더를 교육하는 방법론으로, 마스킹된 포인트 클라우드 토큰이 작업 공간의 행동에 의해 감독됩니다. 이 구조는 3D 상태와 로봇의 운동 간의 기하학적 정렬 기호를 제공합니다. 연구에서는 포인트 클라우드-행동 쌍을 기반으로 인코더와 보조 정렬 헤드를 훈련하며, 사전 훈련 후 인코더는 다운스트림 정책에서 재사용됩니다.

- **Performance Highlights**: LIBERO-10 벤치마크에서 이 방법은 500번의 미세 조정 후 평균 86.9%의 성공률을 달성했습니다. 동일한 사전 훈련된 인코더는 LIBERO-10에서 메타 월드(Meta-World) 간의 전이에서도 평균 73.4%의 성공률을 보였습니다. 실제 실험에서는 시뮬레이션 사전 훈련과 제한된 실デ이터 미세 조정의 결합을 통해 4개의 실제 작업에서 평균 72.5%의 성공률을 달성, 효과적인 sim-to-real 전이를 입증했습니다.



### EquiDexFlow: Contact-Grounded SE(3)-Equivariant Dexterous Grasp Generative Flows (https://arxiv.org/abs/2606.12728)
Comments:
          22 pages, 11 figures, 11 tables. Project page with videos, code, and checkpoints: this https URL

- **What's New**: 이 논문에서는 EquiDexFlow라는 새로운 모델을 소개합니다. 이 모델은 손목 자세, 관절 각도, 손가락 끝 접촉, 표면 법선 및 접촉력을 객체 포인트 클라우드로부터 동시에 예측합니다. 기존의 방식과 달리 EquiDexFlow는 강체 그립이 필요로 하는 접촉 구조와 힘의 분포를 직접 학습합니다.

- **Technical Details**: EquiDexFlow는 SE(3) 동변환(Equivariance) 특성을 유지하면서, 손의 구성과 접촉력을 함께 모델링하는 접근법입니다. 이 모델은 포지션-포스 구조를 분리하지 않고, 한 번의 전향 통과로 손의 구성을 생성합니다. 이를 통해 힘의 적합성과 마찰 가능성을 보장합니다.

- **Performance Highlights**: EquiDexFlow는 8,100개의 힘 닫힘(Force-Closure) 그립을 기반으로 훈련되었으며, 실험 결과 모든 변형 모델들과 비교하여 제로 마찰 위반(Zero Friction Violations), 최고의 복합 점수(Best Composite Score) 및 가장 낮은 렌치 잔여물(Lowest Wrench Residual)을 기록했습니다. 물리 로봇에서 EquiDexFlow로 생성된 그립은 모든 물체 테스트에서 성공적으로 수행되었습니다.



### EWAM: An Enhanced World Action Model for Closed-Loop Online Adaptation in Embodied Intelligenc (https://arxiv.org/abs/2606.12690)
- **What's New**: 본 논문에서는 Enhanced World Action Model (EWAM)라는 폐쇄 루프 온라인 적응 아키텍처를 제안합니다. EWAM은 사전훈련된 Cosmos3 네트워크를 기반으로 하며, 추가적인 배치 데이터 없이 새로운 작업 구성에 적응할 수 있도록 고안되었습니다. 평가 과정에서 추가적인 작업 특정 시연 세트가 도입되지 않았고, 백본 네트워크의 미세 조정도 없었습니다.

- **Technical Details**: EWAM은 경험을 향상시키기 위해 네 개의 경량 신경 레이어를 삽입합니다: Neural Experience Memory Layer는 Diffusion Transformer (DiT)의 중간 레이어에 위치하여 작업 관련 실행 컨텍스트를 제공합니다. Neural Anomaly Detection Layer는 상태 예측 헤드 뒤에 위치하여 예측된 상태와 실제 상태 간의 차이를 실시간으로 모니터링합니다. 그 후 Neural Policy Routing Layer가 동적으로 실행 방식(직접 실행, 보수적 재계획, 또는 롤백 복구)을 선택하며, Neural Action Correction Layer는 실행 진단을 통해 생성된 액션 청크를 정제합니다.

- **Performance Highlights**: EWAM은 RoboLab의 제로샷 조작 작업에서 우수한 실행 품질 향상을 보여주었습니다. 유도된 경량 신경 레이어의 적용을 통해 기존의 Cosmos3-Nano–Policy-DROID 백본은 동결 유지하면서도 적응할 수 있는 능력을 제공합니다. 이 메소드는 예측-실제 불일치, 충돌, 빈 그랩, 인식 환각, 그리고 경로 중복과 같은 실행 문제를 해결하는 것을 목표로 하고 있습니다.



### DARRMS -- An Efficient Algorithm for Dynamic Attention Radius in Resource-Constrained Multi-Agent Systems (https://arxiv.org/abs/2606.12614)
- **What's New**: 이번 논문에서는 리소스 제약이 있는 상황에서 시스템 성능을 개선하고 강력한 의사 결정 전략을 유지하기 위한 새로운 알고리즘을 소개합니다. 이 알고리즘은 에이전트들이 주의 반경(attention radius)을 조정하여 필요한 환경의 일부만을 관찰하게 하여 계산 자원(c computational resources)의 수요를 줄입니다. 그리고 결정(disambiguation) 과정을 최적화하여 불확실한 환경에서 조정과 확장성을 향상시킵니다.

- **Technical Details**: 제안된 알고리즘은 Stackelberg 게임(game)과 적응형 주의 반경(adaptive attention radius)을 통합합니다. 이로써 에이전트는 상황적 요구에 따라 인지 수준을 동적으로 조정할 수 있습니다. 또한, 관측(observability) 및 결정-making을 동시에 최적화하여 다중 에이전트 환경에서 자원을 효율적으로 사용할 수 있도록 합니다.

- **Performance Highlights**: 이론적 분석과 실증적 검증을 통해 적응형 관측(adaptive observation)이 시스템 성능을 어떻게 향상시키는지를 보여줍니다. 효율성과 응답성을 모두 유지하면서 자원을 최소한으로 소비하여 보다 확장 가능하고 효율적인 의사결정을 가능하게 합니다. 이러한 접근 방식은 여러 영역에서의 다중 에이전트 시스템의 활용 가능성을 높이는 데 기여할 것입니다.



### EgoEngine: From Egocentric Human Videos to High-Fidelity Dexterous Robot Demonstrations (https://arxiv.org/abs/2606.12604)
- **What's New**: 이번 연구에서는 대규모 로봇 시연 수집의 비용 문제를 해결하기 위해 인간의 시점에서 촬영된 동영상을 활용한 새로운 프레임워크인 EgoEngine을 제안합니다. EgoEngine은 로봇 학습을 위해 인간의 행동을 로봇 데이터로 변환할 수 있는 방법론을 제공하며, 이를 통해 비디오에서의 시각적 간극(visual gap)과 로봇이 실행 가능한 동작 간의 간극(action gap)을 메웁니다.

- **Technical Details**: EgoEngine은 egocentric RGB 비디오를 입력으로 받아, 두 가지 핵심 출력을 생성합니다. 첫 번째는 장면 맥락과 시간 정렬을 유지하면서 인간을 로봇으로 대체한 고충실도(robot observation) 비디오입니다. 두 번째 출력은 실행 가능한 로봇 동작 궤적을 생성하며, 이는 Task-aligned 및 실행 가능성 제약(feasibility constraints) 하에 이루어집니다.

- **Performance Highlights**: 시뮬레이션 및 실제 로봇 실험을 통해 EgoEngine은 인간 비디오를 로봇 데이터로 변환하는 과정을 확장 가능하게 만드는 능력을 확인했습니다. 또한, 실제 로봇 시연 없이 egocentric 인간 비디오로부터 제로샷(zero-shot) visuomotor dexterous policy 학습을 가능하게 했습니다.



### From Imitation to Alignment: Human-Preference Flow Policies for Long-Horizon Sidewalk Navigation (https://arxiv.org/abs/2606.12603)
- **What's New**: 자동화된 장거리 보도 탐색은 로봇 음식 배달이나 조향 보조 전자 휠체어와 같은 마이크로 모빌리티 응용 프로그램에 필수적입니다. 본 논문에서는 FlowPilot이라는 맵 없이 장거리 탐색 정책을 도입하여 단일 모노큘러 RGB 카메라만으로 다양한 상황에서 로봇이 안전하게 보도를 탐색하도록 합니다. 이는 기존의 모방 학습(imitation learning, IL)의 한계를 극복하기 위해 고안된 새로운 접근 방식입니다.

- **Technical Details**: FlowPilot은 고정된 흐름 일치를(action representation) 사용하는 정책 사전 학습(pre-training) 방법을 제안하며, 이는 복잡한 다중 모드 행동(distribution of sidewalk navigation behaviors)을 포착합니다. 이 방법은 대규모 로봇 데이터에서 다변량 행동을 모델링하고, 사회적 규범(social compliance)과 필요한 조작적 일관성(robustness)을 강화하기 위해 인간의 피드백을 포함한 선호 학습(preference learning) 알고리즘을 포함합니다.

- **Performance Highlights**: 전체 시뮬레이션 성능에서 FlowPilot은 42%의 성공률과 66%의 경로 완료율(route completion)로 평가되었습니다. FlowPilot-HP는 실제 환경에서의 강건성과 사회적 준수를 더욱 향상시켜, 기본 모델 대비 40.0%의 IR 및 52.1%의 NIR 감소를 달성하는 성과를 보였습니다.



### G-MAPP: GPU-accelerated Multi-Agent Planning and Perception for Reactive Motion Generation (https://arxiv.org/abs/2606.12579)
Comments:
          The implementation is available at: this https URL

- **What's New**: 이번 논문에서는 동적 환경에서의 반응형 모션 생성 문제를 해결하기 위한 새로운 프레임워크를 제시합니다. 특히, GPU를 활용한 세계 모델링 및 벡터 필드 기반의 계획을 통해 실시간으로 감지와 계획 모듈 간의 체계적인 통합을 가능하게 합니다. 이로 인해 사용자에게 친숙한 인간 중심 환경에서도 더 효율적인 모션 생성을 이룰 수 있습니다.

- **Technical Details**: 저자들은 반응형 계획 아키텍처를 설계하여 감지(perception), 계획(planning), 행동(action)을 연결하여 운영합니다. 기하학 지향(parameterization) 기법을 통해 로봇의 운동학을 GPU-병렬화된 벡터 필드를 사용하여 탐색합니다. 이 방법론은 기하학적 대수학(geometric algebra)을 사용하여 66D의 작업 공간(task space)을 효율적으로 연결하고, 물리적 환경의 복잡성을 줄이면서 고속 반응을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, GPU 기반의 계획 프레임워크가 CPU 버전 대비 최대 5배 빠른 속도로 충돌 회피를 성공적으로 달성했음을 보여줍니다. 이는 단순한 물리적 환경과 도전적인 물리적 환경 모두에서 유효하며, 실제 환경에서의 적절한 처리를 위한 높은 정확성을 달성합니다.



### Foresight: Iterative Reasoning About Clues that Matter for Navigation (https://arxiv.org/abs/2606.12550)
Comments:
          22 pages, 10 figures, 3 tables

- **What's New**: 이 논문은 로봇의 맵 없는 내비게이션을 위한 'Foresight'라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 미리 훈련된 비전-언어 모델(Vision-Language Model, VLM)을 사용하여 환경 속 자연어 지시에 따라 계획을 제안하고 평가하며 반복적으로 개선합니다. 기존 연구들이 지식 기반의 요소에 의존했던 것과 달리, Foresight는 로봇이 실행하기 전에 환경에서의 시각적 단서를 발견하고 이를 계획 수립에 통합하도록 돕습니다.

- **Technical Details**: Foresight는 로봇이 환경에서 받은 비주얼 입력에 대한 언어 목표를 바탕으로 행동을 결정하는 방법론을 제시합니다. 이 방법은 미세 조정된 VLM이 제안하는 이미지 기반 움직임 계획과 그에 대한 비평을 순환하는 방식으로 이뤄집니다. 이러한 과정은 로봇이 주어진 목표와 장면을 바탕으로 계획을 수정하고 조정할 수 있도록 합니다. 또한 Foresight는 인간 피드백을 통해 보상 모델을 학습하여 VLM을 강화 학습하에 훈련합니다.

- **Performance Highlights**: Foresight는 오프라인 평가 및 실제 환경에서의 실험 결과를 통해 평균 과제 성공률을 37% 향상시키고 미션당 개입을 52% 줄였습니다. 이 결과는 Jetson AGX Orin에서 실시간으로 실행되는 VLM 정책에 의해 생성된 짧은 자유 형식의 추론으로부터 얻어진 것입니다. 기존의 상태 기반 모델이나 더 복잡한 추론 내역을 가진 베이스라인을 초월하며 로봇 모션 세부 조정에서의 테스트 시간 추론에 대한 새로운 기법을 보여줍니다.



### Action-Effect Memory Pretraining for Robot Manipulation (https://arxiv.org/abs/2606.12499)
- **What's New**: 본 논문에서는 로봇 조작을 위한 Action-Effect Memory (AEM) 사전 훈련 프레임워크를 제안합니다. 기존의 전통적인 단일 프레임 시각 인코딩 방법과는 달리, AEM은 시각-행동 역사로부터 시간적 표현을 학습하여 조작의 동적 본질을 다룹니다. AEM은 시각 및 행동 특징을 교차하여 모델링하고, 불완전한 역사에서 누락된 정보를 회복하기 위해 마스크 모델링을 적용하여 행동 조건에 따라 상태 진화를 학습합니다.

- **Technical Details**: AEM은 시각-행동 상호작용의 역사로부터 압축된 상태 표현을 학습하고, 이를 하위 정책에 시간적 컨텍스트로 주입합니다. 구체적으로, AEM은 시각 및 행동 특징을 시간 순서대로 결합하고, 상호작용 역사를 압축된 잠재 상태 표현으로 인코딩하도록 모델을 사전 훈련합니다. Mamba라는 모델을 활용하여 최종 시각 토큰의 인코딩 출력을 압축된 역사 표현으로 사용하며, 이는 역사 압축을 위한 정보 병목 역할을 합니다.

- **Performance Highlights**: AEM은 RoboTwin2.0 벤치마크에서 Diffusion Policy와 ManiFlow Policy에 통합하여 평가되었습니다. AEM은 깨끗한 환경과 불규칙한 환경 모두에서 일관되게 조작 성능을 향상시키며, 단일 프레임 사전 훈련 및 직접 프레임 스택보다 더 우수한 결과를 보였습니다. 이 논문은 AEM이 실제 로봇에서도 효과적임을 입증하며, 복잡한 비마르코프 카르 테스크에서도 높은 성공률을 기록했습니다.



### Learning to Assist: Collaborative VLAs for Implicit Human-Robot Collaboration (https://arxiv.org/abs/2606.12475)
- **What's New**: 이 논문에서는 인간-로봇 협업(HRC)을 위한 새로운 접근 방식으로 시연 데이터에 기반한 전이학습(imitation learning)을 활용합니다. 특히, 시각-언어-동작(vision-language-action, VLA) 모델을 통해 협업 조작을 지원하며, 로봇의 보조적 행동을 조기에 시작하는 문제를 다룹니다. 이 방법은 기존의 수작업으로 설계된 파이프라인을 대체하여 확장성과 유연성을 높입니다.

- **Technical Details**: 연구에서는 VLA 모델이 협업 작업에서 발생하는 도전과제를 어떻게 다루는지를 살펴봅니다. 특히, 시연 행동 누수(demonstration action leakage)라는 실패 모드를 식별하여, 이로 인해 로봇이 미숙한 보조 행동을 하게 되는 원인을 설명합니다. 이를 해결하기 위해, 자동으로 생성된 베이슨 포인트를 활용한 추론 시간 스티어링(inference-time steering) 방법론을 제안하고, 이를 통해 잘못된 행동을 줄이는 동시에 협업 성능을 유지하는 방안을 모색합니다.

- **Performance Highlights**: 16명의 참가자를 대상으로 한 사용자 연구에서, 제안된 스티어링 방법이 장기 실행을 가능하게 하며 조기에 보조하는 문제를 줄이는 결과를 보여주었습니다. 이는 짧은 실행 계획에 비해 협업 속도 증가와 실패 감소를 달성하였습니다. 이러한 성과는 VLA 모델의 유연한 사용을 통해 이루어지며, 다양한 협업 환경에서의 적용 가능성을 제시합니다.



### LabVLA: Grounding Vision-Language-Action Models in Scientific Laboratories (https://arxiv.org/abs/2606.13578)
Comments:
          Work in progress. Project website at this https URL

- **What's New**: 최근 과학 실험에서 AI 시스템의 도움이 더해지고 있으나, 물리적 실험 실행은 여전히 인간 조작자에게 의존하고 있습니다. 비전-언어-액션 (Vision-Language-Action, VLA) 모델은 이 과정을 연결할 수 있는 한 가지 방법을 제공합니다. 하지만 기존의 정책들은 주로 가정용 및 테이블탑 환경에 대해 훈련되어 과학 실험에 필요한 정밀한 조작 지식이 부족합니다. 이에 따라 RoboGenesis라는 시뮬레이션 기반 데이터 엔진이 개발되어 실험 프로토콜을 더 향상시키고 있습니다.

- **Technical Details**: RoboGenesis는 실험 환경을 구축하고, 자동 조작 스킬을 기반으로 프로토콜을 생성하여 성공 사례를 필터링하는 과정으로 구성되어 있습니다. 이 엔진은 실험 장비를 다양하게 사용하여 실제 실험실 데이터 수집의 높은 비용을 줄이는 데 중점을 두고 있습니다. LabVLA는 로봇의 상태, 언어 지시 및 시각적 관찰을 결합하여 계속적인 동작 토큰을 생성하는 방식으로 설계되었습니다. 정책 훈련의 두 단계는 초기 FAST 행동 토큰 사전 훈련과 후속 유동 매칭을 포함합니다.

- **Performance Highlights**: LabVLA는 LabUtopia 벤치마크에서 평가한 모든 기준선보다 평균 성공률이 가장 높았습니다. 이 성공적인 결과는 RoboGenesis가 생성한 데이터의 질과 다양성이 큰 기여를 했다는 것을 시사합니다. LabVLA는 다양한 로봇 에지 효과와 환경에서 프로토콜을 실행하는 능력을 보유하고 있어, 앞으로의 과학 실험 자동화에 중요한 역할을 할 것으로 기대됩니다.



### MaskWAM: Unifying Mask Prompting and Prediction for World-Action Models (https://arxiv.org/abs/2606.13515)
- **What's New**: MaskWAM은 객체 중심의 세계-행동 모델로, 시각적 디스트랙션을 억제하고 언어 모호성을 줄이는 두 가지 주요 이점을 제공합니다. 이 모델은 미래 마스크 예측을 통해 강력한 의미적 감독을 제공하여 정책 성능을 크게 향상시킵니다.

- **Technical Details**: MaskWAM은 Transformer의 혼합 모델(Mixture of Transformers, MoT)을 활용하여 미래 RGB 프레임과 마스크, 행동을 공동 예측하는 통합 아키텍처입니다. 이 모델은 현재 RGB 와 첫 번째 프레임의 시각적 프롬프트를 처리하여 스페이셜 앵커를 제공하며, 이러한 접근 방식은 정책이 작업 관련 영역에 주의를 집중하도록 강제합니다.

- **Performance Highlights**: MaskWAM은 LIBERO에서 98.4%, RoboTwin에서 92.2%의 성공률을 달성하였으며 실제 환경에서도 언어 명확한 작업에서 84.3%, 언어 모호한 작업에서 84.9%를 기록하며 강력한 기준선보다 33.2% 더 뛰어난 성과를 나타냈습니다.



### Heterogeneous LiDAR Early Fusion and Learned Re-Ranking Strategy for Robust Long-Term Place Recognition in Unstructured Environments (https://arxiv.org/abs/2606.13503)
- **What's New**: 새로운 접근법인 MinkUNeXt-VINE++는 두 개의 서로 다른 LiDAR 센서(즉, Livox Mid-360 및 Velodyne VLP-16)로부터의 이종 데이터를 조기에 융합하고 추론 시간에 학습된 재순위화 전략을 통합하는 방식을 제안합니다. 이 방법은 농업과 같은 비구조적 환경에서의 장소 인식 문제를 해결하기 위해 개발되었습니다. 연구 결과, MinkUNeXt-VINE++는 단일 센서 접근 방법에 비해 장소 인식 성능을 현격하게 개선하는 것을 보여줍니다.

- **Technical Details**: MinkUNeXt-VINE++는 이질 리다 데이터의 조기 융합 전략과 후보 장소의 최종 순위를 개선하기 위한 학습된 재순위화 접근법을 결합한 것입니다. 이러한 융합 전략은 각 센서의 강점을 활용하여 환경의 보다 포괄적인 표현을 제공합니다. 이 방법은 TEMPO-VINE 데이터셋을 사용하여 평가되었으며, 다양한 생리적 단계에서 포도원의 이질적 LiDAR 데이터를 제공합니다.

- **Performance Highlights**: MinkUNeXt-VINE++는 단일 센서 접근 방법에 비해 Recall@1 지표에서 각각 20% 개선을 달성하며, 재순위화를 포함할 경우 +30%의 추가 개선을 기록했습니다. 이러한 결과는 비구조적 환경에서 장소 인식의 정확성을 크게 높임을 보여줍니다. 또한, 연구는 넓은 범위의 농업 환경에서도 효과적으로 작동함을 입증했습니다.



### PolyFlow: Safe and Efficient Polytope-Constrained Flow Matching with Constraint Embedding and Projection-free Upda (https://arxiv.org/abs/2606.13400)
Comments:
          30 pages, 12 figures, Accepted to ICML 2026

- **What's New**: 이번 연구는 PolyFlow라는 새로운 polytope-constrained flow matching framework를 제안하며, 기존의 flow-based generative models의 안전 문제를 모델 내에서 직접 해결합니다. PolyFlow는 이산 시간 흐름(discrete-time flow) 공식을 도입하고, 투영(projection) 없는 아키텍처를 구상하여 안전 제약을 충족시키는 동시에 계산 비용을 절감합니다. 이 방법은 계획(planning) 및 제어(control) 작업에서 높은 분포적 일관성을 유지하면서도 제약 위반이 없는 결과를 이끌어냅니다.

- **Technical Details**: PolyFlow는 연속 시간 ODE(Ordinary Differential Equations) 문제를 이산 시간 흐름 프레임워크로 재구성하여 수치적 통합(numerical integration) 오류의 위험성을 없앱니다. 또한, Frank-Wolfe 알고리즘에서 영감을 받은 투영 없는 아키텍처를 통해 매 업데이트가 항상 타당한 경계 내에 있도록 보장하며, 비싼 투영 단계 없이도 안전 제약을 만족합니다. 이는 흐름 정의와 모델 아키텍처에 제약을 직접 임베드하는 혁신적인 접근 방식입니다.

- **Performance Highlights**: 실험 결과, PolyFlow는 다양한 제약 생성 작업에서 0의 제약 위반을 달성하며, 높은 품질의 분포 일치를 유지하는 것으로 확인되었습니다. 이는 기존의 최첨단 제약 생성 방법에 비해 추론 지연(inference latency)을 획기적으로 줄이며, 안전성, 효율성, 생성 품질 간의 유리한 트레이드 오프를 보여줍니다. 이러한 결과는 PolyFlow가 물리적 시스템에서의 안전과 제약 요건을 고려한 진일보한 모델임을 입증합니다.



### Visual Place Recognition in Forests with Depth-Aware Distillation (https://arxiv.org/abs/2606.13206)
Comments:
          IEEE ICRA Workshop on Field Robotics 2026

- **What's New**: 본 논문은 깊이 정보(Depth Information)를 활용하여 자연 환경에서의 시각장소인식(Visual Place Recognition) 성능을 향상시키는 경량(depth-aware distillation) 프레임워크를 제안합니다. 이 프레임워크는 DINOv2 기반의 장소 인식 모델에 기하학적 단서를 주입하여 사전 훈련된 기술 공간(descriptor space)을 유지합니다. WildCross 벤치마크에서 평가한 결과, 단독으로 시각적 정보만 사용하는 것보다 뛰어난 성능 향상을 보여주었습니다.

- **Technical Details**: 제안된 DAD(Depth-Aware Distillation) 프레임워크는 사전 훈련된 VPR 모델의 표시 구조를 유지하는 고정된 교사(teacher)와 깊이가 보강된 학생(student) 구조로 구성됩니다. 학습 과정에서, 학생 모형은 깊이 이미지를 통해 기하학적 단서를 사용하여 효과적으로 인식 성능을 향상시킵니다. 훈련은 쌍별 손실(triplet loss)과 정렬 손실(alignment loss)을 통해 진행되며, 이는 학습된 기술의 선택성을 높이고 기하학적 일관성을 유지하는 데 기여합니다.

- **Performance Highlights**: DAD는 WildCross 데이터 세트에서 깊이 정보가 자연 환경에서의 장소 인식을 보완하는 중요한 역할을 수행함을 입증하였습니다. 실험 결과, DAD 모델은 사전 훈련된 표시 기반 RGB 모델을 고려한 경우에도 더 높은 선택성이 있는 기술 공간을 생성했습니다. 이를 통해 깊이 정보가 장소 인식 성능을 높이는데 기여할 수 있음을 보여주었습니다.



### MPC for underactuated spacecraft control with a Lyapunov supervised physics-informed neural network correction layer (https://arxiv.org/abs/2606.13113)
Comments:
          Accepted at SPAICE (AI in and for Space) 2026

- **What's New**: 우리는 하이브리드 제어 프레임워크를 제안합니다. 이 프레임워크는 기준 모델 기반 컨트롤러와 물리 기반 신경망(PINN)을 결합하여 언더액추에이션(underactuation) 및 모델링 불확실성 문제를 해결합니다. 또한, 안전 크리티컬 작업을 위한 이론적 보장을 유지하면서도 성능을 향상시키는 데 초점을 맞추고 있습니다.

- **Technical Details**: 헤어라키 구조의 제어 아키텍처는 세 가지 주요 구성 요소로 이루어져 있습니다: 비선형 모델 예측 제어기(NMPC), 잔여 교란 모멘트를 추정하는 PINN, 그리고 안정성과 제약 조건 만족을 보장하는 리야프노프 기반 감독 안전 레이어입니다. 이 아키텍처는 우주선의 동적 모델과 함께 하드 제약 조건을 명시적으로 규정하여 실용적인 제어 목표를 설정합니다.

- **Performance Highlights**: 실험 결과, 평균 정상 상태 오류가 3.8% 감소하였고, 최악의 경우는 24.16% 감소하였습니다. 최종 오류에서는 평균 최종 오류가 12.7% 감소하였고, 최대 최종 오류는 52.9% 감소하였고, 최종 오류 분산도 29.42% 감소하였습니다. 이러한 개선은 모델 기반 제어기 단독 사용 시에 비해 가장 중요한 성과로 나타났습니다.



### Scale Buys Interpolation, Structure Buys a Horizon: Certified Predictability for Equivariant World Models (https://arxiv.org/abs/2606.13092)
Comments:
          23 pages (9 main + appendices). Code: this https URL

- **What's New**: 본 논문은 이전의 연구들에서 결합되지 않았던 equivariance(등변성)와 인증된 양면 스펙트럼 수평선(horizon)을 통합하여 새로운 결과를 제시합니다. 주요 결과로는, approximative equivariance(근사 등변성)가 수평선에 대한 제약을 제공하며, 환경의 완벽한 예측 모델이 있을지라도 특정 한계를 가진다는 점을 보여줍니다. 또한, 학습된 혼돈 모델에 대한 법칙을 수상하고, 저자는 이 접근 방식이 전통적인 Lyapunov 법칙에 기반하며 기존 이론과 연결되는 점을 강조합니다.

- **Technical Details**: 이 논문에서는 latent world model(잠재 세계 모델)의 설정과 가정들을 다룹니다. 모델은 Encoder E와 예측기 f로 구성되며, f는 환경 다이나믹스를 통해 주어진 상태에서 다음 상태를 예측합니다. 이 과정에서 T단계의 롤아웃 오류(rollout error)를 정의하고 이 오류가 얽힘에 따라 어떻게 굴절(parallax)되는지를 분석합니다. 또한, 저자는 equivariance가 주어진 모델의 신뢰성을 증명하는 방법을 설명하며, 특정 조건들이 충족될 때 조건부-Lyapunov 조건의 중요성을 나타냅니다.

- **Performance Highlights**: 실험 결과, 40차원 Lorenz-96 시스템에서 $	ext{Z}_N$-equivariant network만이 모든 Lyapunov 스펙트럼을 회복할 수 있음을 보여줍니다(R^2 = 0.98). 반면, 밀집(dense) 및 재귀(recurrent) 기반선은 이 목표를 달성하지 못했습니다. 추가적으로, 인증(prior certification) 데이터 없이도 고정된 예산 아래에서 효과적으로 모델의 신뢰성을 감사(audit)할 수 있는 능력을 강조하며, 이는 훈련이 필요 없는 공개 pretrained 모델에 대해서도 유효하다는 결과를 보여줍니다.



### Effects of Social Interactions in Self-Organising Railway Traffic Managemen (https://arxiv.org/abs/2606.13068)
- **What's New**: 최근 연구는 복잡한 실제 네트워크를 위한 스케일링 솔루션으로 자가 조직화된 교통 관리(Self-organised Traffic Management, SO-TMS)를 탐구하고 있습니다. 이 시스템에서 기차는 이웃을 예측하고 교통 계획 가설을 생성한 후, 이웃과 합의하여 구현할 미래의 교통 계획을 결정합니다. 본 논문은 이 과정에서 중요한 구조적 매개변수인 예측 이웃 수평선(Predictive Neighbourhood Horizon)이라는 요소를 조사하여, 이 매개변수가 지역 상호작용 그래프의 크기와 밀도에 미치는 영향을 분석합니다.

- **Technical Details**: SO-TMS는 기차가 로컬 정보와 피어 조정을 활용하는 구조로, 중앙집중식 의사 결정에 의존하지 않고 결정 권한을 개별 기차에게 위임합니다. 논문에서는 예측 이웃 수평선(ThT_{h})에 대한 선택이 지역 계획과 글로벌 최적성 간의 균형을 조절한다고 설명하며, 이 매개변수를 통해 지역적 접근성과 안전 관리를 유지하면서도 계산 응답성을 확보할 수 있는 조건을 파악하고자 합니다. 실험 결과는 짧은 예측 수평선도 충분하며, 장기 수평선은 계산 응답성을 손상시키고 글로벌 스케줄 최적성에 실질적인 이득이 없다는 것을 보여줍니다.

- **Performance Highlights**: SO-TMS의 성능은 군집화된 실험 분석을 통해 검증되었으며, 이 프레임워크는 경쟁을 유도하는 운영자들이 민감한 상업적 정보를 공개하지 않고도 동적으로 경로를 조정할 수 있도록 합니다. 여기서 제안된 방법은 중앙집중식 최적화 기법에 비해 경쟁력 있는 결과를 제공하며, 지연 전파를 완화하고 서비스 신뢰성을 향상시키는 데 효과적입니다. 이러한 결과는 자기 조직화된 전략이 기존의 중앙 집중식 최적화와 동등하거나 이를 초과하는 성능을 나타낼 수 있음을 시사합니다.



### Diffusion Transformer World-Action Model for AV Scene Prediction (https://arxiv.org/abs/2606.12987)
Comments:
          10 pages, 9 figures, 2 tables

- **What's New**: 이 논문은 자율주행 차량을 위한 action-conditioned world models를 제안하여, 차량의 계획된 제어로부터 향후 카메라 장면을 예측할 수 있게 하였습니다. 이 모델은 현실 세계에서의 시뮬레이션 없이 계획 및 시뮬레이션을 가능하게 하며, ambiguous한 미래 예측 문제를 다루고 있습니다.

- **Technical Details**: 제안된 모델은 compact latent world model로, 현재의 전면 카메라 latent와 일련의 ego-actions(조향각, 가속도)을 입력으로 하여 최대 8초까지 미래 장면의 latent를 예측합니다. VAE(Variational Autoencoder)를 freezer하여 예측된 latent를 256x256의 이미지로 변환하는 파이프라인을 구축하였습니다. 또한, V-JEPA2와 같은 frozen encoders의 성능을 종합적으로 비교하여 성과를 입증했습니다.

- **Performance Highlights**: 모델은 KID(Kernel Inception Distance)에서 0.078의 수치를 기록하며, 기존의 회귀 모델보다 4.8배 뛰어난 성과를 보여주었습니다. 이는 distortion metric을 최적화하기보다는 분포 기반 평가지표를 사용하여 성능을 평가해야 함을 시사합니다. 또한 action controllability가 평균 Spearman $ho=0.81$로 확인되어, 모델의 제어 가능성을 증명하였습니다.



### SemanticXR: Low Power and Real-time Queryable Semantic Mapping with an Object-Level Device-Cloud Architectur (https://arxiv.org/abs/2606.12849)
- **What's New**: SemanticXR는 XR(Extended Reality) 애플리케이션을 위한 최초의 장치-클라우드 시스템으로, 실시간 및 오픈-어휘(오픈-보캐뷸러리) 의미 맵핑 및 쿼리를 제공합니다. 이 시스템은 저전력, 대역폭 및 메모리 제약을 관리하며, 의미적으로 식별 가능한 객체를 통신, 실행 및 저장의 주요 단위로 삼아 장치와 서버간의 시스템 디자인을 개선합니다. 특히, 기존의 서버-클라우드 경계를 넘나드는 의미 맵핑 방식이 없었던 점을 보완합니다.

- **Technical Details**: SemanticXR는 객체 기반의 의미 맵핑 파이프라인에 기반하여 동작하며, RGB 및 깊이 프레임과 장치 위치 정보를 서버에 전송해 객체를 탐지하고 의미 임베딩을 추출합니다. 이 시스템은 네트워크 연결 여부에 따라 서버 측 맵 또는 로컬 의미 맵을 통해 쿼리를 실행할 수 있는 기능을 갖추고 있습니다. 객체-레벨 병렬 처리 및 기하학적 다운샘플링을 통해 서버 측 맵핑 지연 시간을 개선하며, 객체 별로 설정 가능한 자원 사용 환경을 제공하여 유연한 운영을 지원합니다.

- **Performance Highlights**: SemanticXR는 지연 시간을 100ms 미만으로 유지하며, 최대 10,000개의 객체를 처리할 수 있고, 네트워크 드롭 상황에서도 안정적인 쿼리를 지원합니다. 메모리 공간은 500MB 내에서 수만 개의 객체를 유지할 수 있으며, 전체 씬 크기가 아닌 맵 변경에 따라 대역폭을 조정합니다. 이 시스템의 일반적인 운영 중 장치의 추가 전력 소비는 단지 2%에 불과합니다.



### TrajGenAgent: A Hierarchical LLM Agent for Human Mobility Trajectory Generation (https://arxiv.org/abs/2606.12657)
Comments:
          14 pages, 2 figures, 8 tables. Accepted by the 27th IEEE International Conference on Mobile Data Management (MDM 2026)

- **What's New**: 이번 연구는 TrajGenAgent라는 새로운 계층적 LLM-agent 프레임워크를 제안하여, 인간 이동 경로 생성을 실현합니다. 기존의 모델 파인튜닝 없이도 실행 가능하며, 개인화된 POI 검색 및 거리 인식을 통해 각 활동을 완전한 방문으로 구체화합니다. 이 프레임워크는 두 단계의 오케스트레이터-노동자 설계를 사용하여, 과거 증거를 기반으로 활동 체인을 생성하고 각 활동을 상세히 모델링합니다.

- **Technical Details**: TrajGenAgent는 LLM의 의미적 추론과 계획 기능을 활용하며, 고정된 워크플로우 내에서 도구를 조정하여 결정적인 경로 생성을 가능하게 합니다. 첫 번째 단계에서 오케스트레이터는 개인의 역사적 활동 체인으로부터 활동 체인 스켈레톤을 생성하고, 두 번째 단계에서는 특별히 설계된 공간적 및 시간적 노동자가 활동을 완전한 방문으로 변형합니다. 이 과정에서 이동-모달리티를 고려하여 여행 시간을 추정하고, 역사적 활동과 일치하는 장소를 선택합니다.

- **Performance Highlights**: 실험 결과 TrajGenAgent는 기존의 신경망 및 LLM 기반 모델들보다 공간적 충실도(spatiotemporal fidelity)와 의미적 일관성(semantic coherence), 개별 행동의 사실성을 개선하였습니다. 특히, 새로운 평가 프레임워크(이상 탐지 기반)를 도입하여 생성된 경로의 행동 수준의 적합성을 평가하는 데 성공하였습니다. 이러한 결과는 TrajGenAgent가 고품질의 현실적인 이동 경로 생성을 가능하게 함을 시사합니다.



### Individual Control Barrier Functions-Guided Diffusion Model for Safe Offline Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2606.12640)
Comments:
          Accepted to the 23rd IFAC World Congress, 2026

- **What's New**: 이번 연구에서는 다중 에이전트(multi-agent) 환경에서 안전성을 고려한 오프라인 강화 학습(offline reinforcement learning) 알고리즘을 제안합니다. 기존 연구들은 주로 단일 에이전트(single-agent) 설정에서 안전성을 다루었으나, 본 논문은 다중 에이전트를 위한 안전성 문제를 탐구하며, 신경망 기반의 개별 제어장벽 함수(neural individual control barrier functions)를 확산 모델(diffusion model)에 통합하여 궤적(trajectory) 생성을 안전하게 합니다.

- **Technical Details**: 제안된 방법은 중앙 집중식 훈련과 분산 실행(centralized training and decentralized execution, CTDE) 패러다임을 따르며, 각 에이전트 간의 조정을 모델링합니다. 다중 에이전트 강화 학습(multi-agent reinforcement learning, MARL)에서는 개별 CBF를 사용하여 모든 에이전트가 안전 제약을 충족하도록 훈련하며, 이 과정에서 확산 모델은 상태(state)를 기초로 궤적을 생성합니다. 이 과정에서 비용 함수를 학습하는 대신, 신경망 기반의 CBF를 이용해 안전 집합(safe sets) 안에서 높은 보상(high-reward) 궤적을 생성합니다.

- **Performance Highlights**: 제안된 알고리즘은 다양한 벤치마크에서 평가되었으며, 안전성에서의 큰 개선과 경쟁력 있는 보상(rewards)을 유지했습니다. 기존의 오프라인 MARL 기법과 비교했을 때, 우리의 방법은 보상과 안전성 두 측면에서 우수한 성능을 보였습니다. 이를 통해 안전성이 중요한 응용 분야에서도 오프라인 강화 학습의 효과를 실증적으로 입증했습니다.



### $μ$VLA: On Recurrent Memory for Partially Observable Manipulation in VLA Models (https://arxiv.org/abs/2606.12497)
Comments:
          34 pages, 20 figures, 9 tables

- **What's New**: 본 연구에서는 기존의 VLA 모델에서의 recurrence(재귀)의 역할을 명확히 하고자, 이를 단일 변수로 통제하며 실험한 첫번째 연구라고 주장합니다. 특히 VLA의 기존 아키텍처에 auxiliary losses(보조 손실)나 복잡한 메커니즘을 추가하지 않고, 최소한의 memory(메모리) 구조만을 활용하여 성능을 평가합니다. 이를 통해 minimal in-backbone recurrence(기본 구조 내 재귀)의 잠재력을 평가하고, 추가적인 메모리 구조가 필요한 경우와 그렇지 않은 경우를 명확히 구분합니다.

- **Technical Details**: μVLA는 transformer(변환기) 아키텍처에 학습 가능한 메모리 토큰을 추가하여 모델의 시간적 업데이트를 수행합니다. 연구팀은 TBPTT(장기적 역전파) 방식으로 훈련하며, 다양한 메모리 구조에 대해 실험하고 있습니다. μVLA는 memory width(메모리 폭), TBPTT 길이, 메모리 업데이트 규칙을 매개변수화하여 서로 다른 계열의 VLA 변형을 제안하며, 이들 간의 차이를 재귀 상태만으로 제한합니다.

- **Performance Highlights**: MIKASA-Robo에서 μVLA는 최적의 설정 하에 다섯 가지 훈련 과제에서 성공률을 0.42에서 0.84로 향상시키며, 동일한 메모리 구조를 가진 홀드아웃 작업에서는 0.07에서 0.23로 개선되었습니다. LIBERO에서 가장 강력한 재귀 변형은 96.2%의 평균 성공률을 기록했으며, 완전 관측 환경에서도 퇴보가 없음을 나타냈습니다. 이러한 결과들은 VLA 모델에 있어 지속 가능한 minimal recurrence의 유용성을 시사합니다.



New uploads on arXiv(cs.MA)

### Tuning Agent-Based Predator-Prey Models Toward Lotka-Volterra Dynamics (https://arxiv.org/abs/2606.13639)
Comments:
          12 pages, 3 figures

- **What's New**: 최근 컴퓨팅 파워의 증가로 인해 대규모 에이전트 기반 모델(agent-based models, ABMs)을 사용하여 복잡한 적응 시스템을 시뮬레이션하는 것이 점점 더 가능해졌습니다. 연구자들은 이러한 모델에서 소규모 변화가 인구 붕괴나 인위적인 경계에 도달하기 때문에 환경 및 인구 매개변수를 조정하여 결과적인 인구 역학이 고전적 Lotka-Volterra 주기와 유사하게 보이도록 할 수 있는지를 조사했습니다. 이 연구는 지속적인 포식자-피식자 시스템에서 양과 늑대로 설정한 에이전트를 연구하며, 효과적인 시뮬레이션 범위를 고안합니다.

- **Technical Details**: 모델은 2차원 평면에서 움직이는 원형 에이전트를 포함하고 있으며, 각 에이전트는 고유의 감각적 행동을 결정하는 연속 시계열 재발 신경망(CTRNN) 기반 컨트롤러를 가지고 있습니다. 에이전트는 서로 충돌하지 않고 통과할 수 있으며, 인구의 변화는 목표 에너지 함량에 따라 달라집니다. 양과 늑대의 피식자-포식자 관계를 통해 생태적 피드백을 모델링하며, 이 모델링은 ABMax 프레임워크를 사용하여 최적화됩니다.

- **Performance Highlights**: 이 연구는 랜덤 컨트롤러와 진화된 컨트롤러에 대해 각각 최적화 작업을 시도하여, 인구 보존과 장기 지속 가능성을 보장하는 인구 동역학을 성취했습니다. 특히, 연구진은 에이전트의 판단력과 행동 동향을 최적화하여 Lotka-Volterra 동역학을 따르는 지속적인 주기를 나타낼 수 있는 가능성을 보여주었습니다. 이 모델은 에이전트 기반 시스템의 유용성을 강화하며, 이러한 접근 방식을 통해 복잡한 생태적 상호작용을 보다 잘 이해할 수 있습니다.



### See What I See, Know What I Think: Dense Latent Communication Across Heterogeneous Agents (https://arxiv.org/abs/2606.13594)
- **What's New**: 본 논문은 이질적인 에이전트가 어떻게 효과적으로 정보 전송 할 수 있는지를 탐구합니다. 기존의 KV-cache(키-값 캐시) 통신 방법은 동질적인 에이전트에 국한되어 있었으나, 본 연구는 이질적인 모델 간의 라텐트 정렬을 지원합니다. 새로운 방법론은 에이전트가 무엇을 보고 어떻게 생각하는지를 모두 전송할 수 있는 가능성을 보여줍니다.

- **Technical Details**:  연구에서는 KV-cache 통신의 효율성을 극대화하기 위해 두 단계의 훈련 프로세스, 즉 재구성 후 생성(재구성이 먼저 이루어지고 그 다음 생성) 과정을 도입합니다. 에이전트 간의 정보 전송은 고유한 KV-cache 적응기를 활용하여 이루어지며, 위치 분리(positional disentanglement) 및 세부 헤드 변환(per-head transformation)을 통해 최적화됩니다. 이 방법은 정보의 보존과 함께 흐름을 효율적으로 관리합니다.

- **Performance Highlights**: 제안된 방법은 {Qwen3-4B, 8B, 14B} 전반에 걸쳐 6개의 벤치마크에서 최대 3배 더 적은 계산 비용으로 이전의 이질적 기반선을 초과하는 성능을 보였습니다. 특히 맥락을 인식하지 못하는 상황에서도 효과적으로 작동하여 불리한 조건에서도 시간을 절약하고 성능을 유지할 수 있음을 입증하였습니다.



### $α$-fair heterogeneous agent reinforcement learning (https://arxiv.org/abs/2606.13076)
- **What's New**: 이 논문은 다수의 에이전트가 참여하는 시스템에서의 협력을 최적화하는 새로운 방법론을 제안합니다. 기존의 공리주의적 목표는 전반적인 효율성을 극대화하는 데 초점을 두지만 보상 분배를 고려하지 않아 불공평한 '리더-추종자' 동역학이 발생할 수 있습니다. 이 연구에서는 $eta$-공정성(α-fairness)와 이종 에이전트 신뢰 구역 학습(HATRL)을 연결하는 혁신적인 프레임워크를 소개하여, 유동적인 개선과 내쉬 균형(Nash Equilibria) 수렴을 보장합니다.

- **Technical Details**: 제안된 방법론은 에이전트의 예상 수익에 기반하여 유틸리티를 동적으로 가중치화하는 공정한 이점 함수(fair advantage function)를 활용합니다. 이로 인해 전 세계적인 목표가 순수한 공리주의적 효율성에서 $eta$-공정성(α-fairness) 복지로 전환됩니다. 또한, 본 논문에서는 두 가지 실용적인 알고리즘인 $eta$-fair HATRPO와 $eta$-fair HAPPO를 도입하며, 이 알고리즘들이 이론적으로 안전한 학습 프레임워크를 갖추고 있음을 강조합니다.

- **Performance Highlights**: 실험 결과, CleanUp 및 CommonHarvest와 같은 연속 사회적 딜레마에서 $eta$-fair HATRPO와 $eta$-fair HAPPO가 기존의 HATRL 알고리즘들보다 더 나은 성능을 발휘함을 보여줍니다. 또한, 이 알고리즘들은 사회적으로 더 높은 결과를 달성하면서도 공리적 관점에서도 더 우수한 성과를 냈습니다. 이러한 결과들은 공정성을 고려한 협력의 중요성을 강조하며, 다수 에이전트 시스템의 발전에 기여할 것입니다.



### Effects of Social Interactions in Self-Organising Railway Traffic Managemen (https://arxiv.org/abs/2606.13068)
- **What's New**: 최근 연구는 복잡한 실제 네트워크를 위한 스케일링 솔루션으로 자가 조직화된 교통 관리(Self-organised Traffic Management, SO-TMS)를 탐구하고 있습니다. 이 시스템에서 기차는 이웃을 예측하고 교통 계획 가설을 생성한 후, 이웃과 합의하여 구현할 미래의 교통 계획을 결정합니다. 본 논문은 이 과정에서 중요한 구조적 매개변수인 예측 이웃 수평선(Predictive Neighbourhood Horizon)이라는 요소를 조사하여, 이 매개변수가 지역 상호작용 그래프의 크기와 밀도에 미치는 영향을 분석합니다.

- **Technical Details**: SO-TMS는 기차가 로컬 정보와 피어 조정을 활용하는 구조로, 중앙집중식 의사 결정에 의존하지 않고 결정 권한을 개별 기차에게 위임합니다. 논문에서는 예측 이웃 수평선(ThT_{h})에 대한 선택이 지역 계획과 글로벌 최적성 간의 균형을 조절한다고 설명하며, 이 매개변수를 통해 지역적 접근성과 안전 관리를 유지하면서도 계산 응답성을 확보할 수 있는 조건을 파악하고자 합니다. 실험 결과는 짧은 예측 수평선도 충분하며, 장기 수평선은 계산 응답성을 손상시키고 글로벌 스케줄 최적성에 실질적인 이득이 없다는 것을 보여줍니다.

- **Performance Highlights**: SO-TMS의 성능은 군집화된 실험 분석을 통해 검증되었으며, 이 프레임워크는 경쟁을 유도하는 운영자들이 민감한 상업적 정보를 공개하지 않고도 동적으로 경로를 조정할 수 있도록 합니다. 여기서 제안된 방법은 중앙집중식 최적화 기법에 비해 경쟁력 있는 결과를 제공하며, 지연 전파를 완화하고 서비스 신뢰성을 향상시키는 데 효과적입니다. 이러한 결과는 자기 조직화된 전략이 기존의 중앙 집중식 최적화와 동등하거나 이를 초과하는 성능을 나타낼 수 있음을 시사합니다.



### The Internet of Agentic AI: Communication, Coordination, and Collective Intelligence at Sca (https://arxiv.org/abs/2606.12835)
- **What's New**: 자율 AI 에이전트의 급속한 출현은 인공지능을 고립된 모델 추론에서 분산된 추론, 통신 및 행동의 시스템으로 변모시키고 있습니다. 이 논문은 이질적인 에이전트들이 서로를 발견하고, 책임을 협상하며, 맥락을 교환하고, 도구를 호출하고, 클라우드, 엣지, 장치 및 사이버 물리적 환경의 워크플로우를 실행하는 오픈 생태계인 IoAI를 개발하는 비전을 제시합니다. 에이전트의 배치 모델, 워크플로우 생애 주기, 통신 프로토콜 등을 검토하며 적응형 제조와 분산된 운영 조정에 대한 사례 연구를 포함하고 있습니다.

- **Technical Details**: 본 논문은 단일 에이전트 AI, 다중 에이전트 시스템, 분산 컴퓨팅, 통신 네트워크, 게임 이론 및 보안 공학의 기초를 조합하여 확장 가능한 에이전트 생태계에 필요한 아키텍처와 메커니즘을 특성화합니다. IoAI는 자율적인 인공지능 에이전트가 서로 통신하고 협력하며, 복잡한 작업을 독립적으로 수행하는 데 필요한 능력, 정보, 자원 인식을 조합하여 동작하는 구조로 설명됩니다. 에이전트는 외부 API를 호출하고, 데이터베이스와 상호작용하며, 실행 가능한 코드를 생성할 수 있는 능력을 가집니다.

- **Performance Highlights**: 이러한 자율 에이전트들은 복잡한 작업을 협력적으로 수행하기 위해 서로 의존적인 하위 작업으로 목표를 분해하는 특성을 가지고 있습니다. 논문에서 다루는 주요 연구 과제에는 제어되는 출현, 의미론적 상호 운용성, 안전한 신원, 인센티브 호환형 조정, 자원 인식 오케스트레이션 및 거버넌스가 포함됩니다. 이러한 에이전트들은 예를 들어 분산 사이버 방어, 협력적 과학 발견 및 자율 물류 조정과 같은 대규모 운영 환경에서 중요한 역할을 하고 있습니다.



### Smarter Saboteurs, Better Fixers: Scaling & Security in Linear Multi-Agent Workflows (https://arxiv.org/abs/2606.12709)
Comments:
          16 pages (4 are main text), 2 figures, 6 tables. Accepted to the AIWILD Workshop at ICML 2026

- **What's New**: 이 논문은 LLM 기반 다중 에이전트 시스템(MAS)의 협력 구조가 악의적인 공격에 얼마나 견고하게 대응하는지를 분석하고 있다. 실험을 통해 모델 규모가 커질수록 비정상적인 지시를 충실히 수행하는 경향이 강해짐을 발견했으며, 특이하게도 패치 단계를 추가했을 때 이 vulnerability가 크게 줄어드는 것을 보여준다. 이를 통해 선형 협력 구조의 효능과 복원력이 존재할 수 있음을 시사한다.

- **Technical Details**: 이 연구에서는 MetaGPT에서 파생된 선형 소프트웨어 개발 생명주기(SDLC) 파이프라인을 점검하였다. 'Engineer' 에이전트가 악의적인 지시를 받아들이고 코드에 오류를 삽입하도록 설계된 실험을 통해, 비조정된 파이프라인과 새로 추가된 'Fixer' 단계를 비교하였다. 결과적으로, 모델의 규모가 커질수록 비조정 파이프라인은 침해에 취약해지지만, 수정 단계를 추가할 경우 오류를 탐지하고 수정하는 기능이 강화되어 Linear MAS의 수명이 보존됨을 확인하였다.

- **Performance Highlights**: 실험 결과, Pass@1 점수는 각 모델과 시나리오(제어 vs 악의적) 조합에 대해 보고되었다. 모델 규모가 증가할수록 악의적인 지시를 처리하는 데 있어 성능이 크게 저하되는 것을 보여주었고, 마지막으로 ‘Fixer’ 단계를 도입했을 때 성능 감소를 최소화할 수 있음을 입증하였다. 이러한 발견은 향후 LLM 기반 MAS의 안전성과 신뢰성을 높일 수 있는 방향성을 제공한다.



### SAIGuard: Communication-State Simulation for Proactive Defense of LLM Multi-Agent Systems (https://arxiv.org/abs/2606.12474)
- **What's New**: 이 논문은 복잡한 작업을 수행하는 LLM 기반 다중 에이전트 시스템(MAS)에서 보안 위험이 확산되는 문제를 해결하기 위해 새로운 방법을 제안합니다. 기존의 반응적 방어 방식과 달리, SAIGuard라는 프로액티브 방어 프레임워크를 통해 공격이 발생하기 전에 위험 정보를 가로채는 방식을 사용합니다. 이 방법은 MAS 상호작용 그래프를 시뮬레이션하여 잠재적인 메시지 영향을 평가하고, 유해한 메시지를 사전에 정제하거나 재생성합니다.

- **Technical Details**: SAIGuard 시스템은 MAS를 상호작용 그래프로 모델링하며, 각 에이전트의 상태와 수신한 메시지를 기반으로 커뮤니케이션 상태를 시뮬레이션합니다. 이를 통해 메시지가 로컬 에이전트 상태와 글로벌 시스템 상태에 미치는 영향을 추정하고, 정상적인 커뮤니케이션 패턴에서의 재구성 편차를 통해 위험 신호를 탐지합니다. 특히, 두 가지 주요 구성 요소인 커뮤니케이션 상태 시뮬레이션과 시스템 편차 개입을 통해 에이전트를 격리하는 대신 위험 메시지를 먼저 처리합니다.

- **Performance Highlights**: 실험 결과, SAIGuard는 다양한 토폴로지와 공격 시나리오에 걸쳐 공격 성공률을 감소시키면서도 MAS의 협업 유틸리티를 유지하는 것으로 나타났습니다. 이는 서술된 반응적 방어보다 성능이 우수함을 보여줍니다. SAIGuard는 전반적인 협업 성능을 손상시키지 않으면서도 보안을 강화하는 중요성을 강조합니다.



### Beyond Runtime Enforcement: Shield Synthesis as Defensibility Analysis for Adversarial Networks (https://arxiv.org/abs/2606.13621)
Comments:
          26 pages, 7 figures, 7 tables. Under review at JAIR. Code: this https URL

- **What's New**: 이번 연구에서는 Shielded Reinforcement Learning(ShRL)을 이용한 안전 메커니즘의 재정립을 제안합니다. 기존의 ShRL은 에이전트의 행동을 제한하는 runtime safety 기법으로 소개되었으나, 본 논문에서는 이를 설계 시 분석 도구로 활용해야 한다고 주장합니다. 연구자들은 두 개의 비대칭 사양을 사용하여 네트워크 방어를 위한 제약된 안전 게임을 정의하고, 이는 조직의 방어 가능성을 평가할 수 있는 분석 기법으로 자리잡게 됩니다.

- **Technical Details**: 연구에서는 Defender(수비수) 사양이 게임의 안전 영역을 정의하고, Attacker(공격자) 사양이 공격자의 합법적인 행동을 제한하는 구성을 채택합니다. 이러한 구조를 통해 게임 솔루션을 해결함으로써 특정 topology-specification 쌍이 방어 가능한지 여부를 판단할 수 있는 формальный 증명서를 생성합니다. 또한, attractor 구조로부터 얻은 topological metrics와 shield-constrained adversarial multi-agent reinforcement learning의 동작 결과를 결합하여 방어 가능성을 평가하는 지문(defensibility fingerprint)을 획득합니다.

- **Performance Highlights**: 연구결과, 미세한 아키텍처 변경이 방어 가능성의 공식적인 경계는 거의 변하지 않으면서도 운영상의 결과를 크게 변화시킬 수 있음을 보여주는 'What-if 분석'을 실시하였습니다. 이로 인해 formal defensibility와 operational effectiveness는 보안의 서로 다른 측면을 포착하고 있음을 입증하였습니다. 연구 결과는 네트워크 방어의 효과성을 증명하는 데 있어 기여할 것으로 기대됩니다.



### Multi-Agent Reinforcement Learning from Delayed Marketplace Feedback for Objective-Weight Adaptation in Three-Sided Dispatch (https://arxiv.org/abs/2606.13604)
Comments:
          Accepted at ICML 2026 Workshop on Reinforcement Learning from World Feedback (RLxF)

- **What's New**: 이번 논문은 DoorDash의 대규모 음식 배달 시장에서 지연된 신호를 활용한 강화학습 시스템을 소개합니다. 배송 품질과 배치 효율성 간의 트레이드오프를 조정하는 학습된 정책을 통해, 수동으로 설정하던 전역적인 휴리스틱 가중치 대신 매장 수준의 정책을 적용합니다. 이 시스템은 생산 가용성 제약을 유지하면서 실행되고, 실시간으로 동적인 피드백에 대응할 수 있습니다.

- **Technical Details**: 제안한 시스템은 두 개의 중첩된 의사결정 레이어를 갖추고 있습니다. 내부 레이어는 최적화 기법으로, 주문, 배달원, 제약조건 및 목표 가중치를 기반으로 배달원-주문 할당을 매핑합니다. 외부 레이어는 OWA-RL(목표 가중치 적응 에이전트)로, 시장 결과에서 학습하며 최적화 전에 매장 수준의 목표 가중치 배수를 선택합니다.

- **Performance Highlights**: 프로덕션 환경에서의 실험 결과, 오프라인에서 훈련된 정책은 배치 수를 증가시키고 배달원 측의 시간을 줄이는 동시에 고객이 경험하는 배송 품질에는 저해를 주지 않는 것으로 나타났습니다. 이러한 결과는 실시간 경제 및 물류 시스템의 피드백을 활용하여 의사결정 정책을 안전하게 적응할 수 있는 방법을 잘 보여줍니다.



### Reward Modeling for Multi-Agent Orchestration (https://arxiv.org/abs/2606.13598)
Comments:
          Preprint; work in progress

- **What's New**: 이 논문에서는 Large Language Models(LLMs) 기반의 Multi-Agent Systems(MAS)에서의 효과적인 오케스트레이션을 위한 새로운 방법인 Orchestration Reward Modeling(OrchRM)을 제안합니다. OrchRM은 사람의 주석 없이도 오케스트레이션 품질을 평가할 수 있는 자기지도(self-supervised) 프레임워크로, 다수의 에이전트 실행에서 중간 산출물을 활용하여 보상 모델을 훈련합니다.

- **Technical Details**: OrchRM은 Bradley-Terry 보상 모델 훈련을 위한 승-패 쌍을 구성하는 방식으로 작동하며, 기존의 비용이 많이 드는 서브 에이전트 롤아웃(sub-agent rollouts) 방식 대신 오케스트레이션 수준에서 직접 이루어집니다. 이를 통해 오케스트레이터 훈련 및 MAS 테스트 시간 스케일링에서 효율적이고 성능 높은 보상 지향 훈련을 가능하게 합니다.

- **Performance Highlights**: OrchRM은 토큰 사용에서 훈련 효율성을 최대 10배 증가시키고, MAS 테스트 시간 스케일링 성능에서 정확도를 최대 8% 향상시키며, 이러한 성과는 수학적 추론, 웹 기반 질문 응답, 다단계 추론 등 여러 도메인에 일관되게 전이됩니다. 이 연구는 다중 에이전트 오케스트레이션의 강력한 방향으로 오케스트레이션 수준의 보상 모델링을 제시합니다.



### Multiagent Protocols with Aggregated Confidence Signals (https://arxiv.org/abs/2606.13591)
Comments:
          22 pages and 5 figures, 9 pages and 2 figures before the appendix

- **What's New**: 이 논문은 다수의 에이전트 시스템의 출력에 대한 신뢰도를 예측하고 평가하는 방법의 부재를 언급하며, 이는 자연어 처리(NLP)에서 신뢰성 확보와 결정 과정을 지원하기 위해 중요한 요소라고 강조합니다. 기존의 다수 에이전트 논쟁(MAD) 시스템에서는 개별 에이전트 중의 하나에 대한 신뢰도를 결정할 수는 있었지만, 전체 시스템의 신뢰도를 집계하는 방법에 대한 연구는 없었습니다. 이 논문은 원시 신뢰도 정보를 변환하고 결합하여 최종 응답과 함께 집계된 신뢰도를 생성하는 세 가지 프로토콜을 제안합니다.

- **Technical Details**: 저자들은 세 가지 프로토콜을 통해 여러 화자의 신뢰도를 비교할 수 있는 방식을 설명합니다. 이 프로토콜들은 각각의 에이전트에서 나오는 원시 신뢰도 신호를 변환하여 비교 가능하게 만든 후, 부드러운 투표(soft voting)나 베이지안 융합(Bayesian fusion)을 사용하여 결합합니다. 저자들은 별도의 평가 및 교정을 수행하여 두 가지 다른 신뢰도 추정치인 자기 보고(self-report) 및 로그 확률(logit-based sequence probability)과의 성능 비교도 다룹니다.

- **Performance Highlights**: 이 연구에서 제시된 프로토콜은 다섯 개의 기준에 걸쳐 최고의 단일 에이전트나 기존의 논쟁 기반 비교보다 훨씬 더 뛰어난 구별력(AUARC)을 보였습니다. F1 점수의 경우, 이 방법은 더욱 모호한 작업에서 발생하는 손실을 회복하면서 안정적인 성능을 유지했습니다. 마지막으로, 저자들은 신뢰도 추정치와 사후 교정 기술에 대한 비교를 제공하며, 어떤 추정치가 다른 것보다 consistently 뛰어난 것이 없다는 점과 교정의 중요성을 강조합니다.



### Neuro-Symbolic Agents for Regulated Process Automation: Challenges and Research Agenda (https://arxiv.org/abs/2606.13405)
Comments:
          Accepted as a poster in NILA Workshop @ IJCAI-ECAI 2026

- **What's New**: 이 논문에서는 LLM 기반 에이전트가 규제가 있는 산업에 진출하고 있으며, 이들이 품질 관리(Quality Management) 프로세스를 자동화하는 방식을 논의합니다. 기존의 규정이나 프로세스 모델을 단순히 외부 모니터링 메커니즘으로 간주하는 것이 아니라, 이러한 요소들이 에이전트의 의사 결정과 행동을 형성하는 핵심적인 아키텍처 구성 요소로 다뤄져야 한다고 주장합니다. 'Compliance-by-construction'이라는 새로운 패러다임을 제안하여, 제어 흐름 위반을 구조적으로 방지하면서도, 가드레일(guidelines)은 여전히 의미적 오류를 잡는 데 필수적이라고 강조합니다.

- **Technical Details**: 신경-기호적 AI는 규제가 있는 프로세스 자동화에서 필요성이 높습니다. 제약업계의 품질 관리 프로세스는 유럽 GMP, ISO 사양 등 복잡한 규제 체계에 의해 관리되며, 전자 품질 관리 시스템(eQMS)을 통해 체계적으로 실행됩니다. 이 과정에서는 다수의 역할 및 문서와 의사 결정 지점에서 규제 요건을 충족해야 하며, 이는 신경망과 기호적 구조의 통합을 통해 해결해야 하는 과제를 만들어냅니다.

- **Performance Highlights**: 이 논문은 신경-기호적 AI의 통합이 규제가 있는 프로세스 자동화 분야에서 중요한 연구 기회임을 강조합니다. 현재의 LLM 에이전트들은 뛰어난 자연어 이해 및 판단 능력을 발휘할 수 있지만, 기존의 규제 구조와의 통합이 이루어지지 않으면 그 효용이 제한될 수 있습니다. 따라서 이 연구는 신경-기호적 통합의 필요성과 그로 인해 발생하는 연구 챌린지를 구체적으로 제시합니다.



### Can I Buy Your KV Cache? (https://arxiv.org/abs/2606.13361)
- **What's New**: 이 논문에서는 AI 에이전트들이 동일한 문서를 여러 번 비효율적으로 처리하는 문제를 지적하고, 이 문제를 해결하기 위해 KV 캐시(key-value cache)를 한 번만 계산하고 이를 재사용하는 방안을 제안합니다. 퍼블리셔가 문서의 KV 캐시를 미리 계산하면, 나중에 다른 에이전트들이 이를 불러와서 다시 계산을 생략할 수 있습니다. 이 방법은 효율적이며, 고정된 모델 하에서 정확한 결과를 보장합니다.

- **Technical Details**: 프리필(prefill) 단계에서 모든 입력 토큰에 대한 전방 패스를 실행하여 KV 캐시를 생성한 후, 디코딩 단계에서 캐시된 키와 값을 참조하여 출력 토큰을 생성합니다. 그러나 이 과정이 대규모 언어 모델(LLM)에서는 계산 집약적(compute-intensive)이며, 긴 입력의 경우 프리필이 지연(latency)과 비용의 대부분을 차지합니다. 제안된 프리필 CDN(data delivery network) 모델은 KV 캐시를 재사용함으로써 전체 계산 비용을 크게 감소시킵니다.

- **Performance Highlights**: Qwen3-4B 모델에서 계산한 결과, 재사용은 프리필보다 9배에서 50배 저렴하며, 입력 길이가 길어질수록 이 차이는 더 커집니다. 즐겨찾는 문서 1개를 8000만 에이전트에 제공하는 비용은 약 150만 달러의 재프리필 비용에 비해 3만 달러의 재사용 비용으로 현저히 낮습니다. 이 논문은 이러한 이점을 활용한 비용 절감 모델을 제시하여 프리필 CDN의 개념을 명확하게 설명합니다.



### LLM-as-an-Investigator: Evidence-First Reasoning for Robust Interactive Problem Diagnosis (https://arxiv.org/abs/2606.13220)
- **What's New**: 이 논문은 대형 언어 모델(LLM)의 사용자 주도 시콥언시(user-driven sycophancy) 문제를 해결하기 위해 LLM을 증거 기반의 조사의 도구로 사용하려는 새로운 방식인 LLM-as-an-Investigator를 제안합니다. 이 방법론은 초기 문제 설명에서 불확실성을 파악하고 후보 가설을 생성하며, 질문을 통한 명확화를 통해 문제를 해결하는 접근 방식을 사용합니다. 이를 통해 모델이 사용자 제공 가설을 조기에 받아들이는 경향을 줄이고, 더 신뢰성 있는 결과를 도출할 수 있게 됩니다.

- **Technical Details**: LLM-as-an-Investigator에서는 Solution Investigator Agent가 중심 역할을 합니다. 이 에이전트는 초기 문제 설명의 모호성을 추정하고, 진단을 위해 후보 솔루션을 생성한 뒤, 목표에 맞춘 질문을 통해 불확실성을 해소합니다. 이 과정에서 수집된 데이터를 바탕으로 가설 확률을 갱신하고, 증거가 확보될 때까지 즉각적인 대답을 제공하는 대신 지속적으로 조사를 실시합니다.

- **Performance Highlights**: 제안된 접근 방식은 기존의 LLM 대비 문제를 더 정확하게 식별하는 것으로 나타났습니다. 표준 보조 도구는 사용자의 잘못된 가설에 대해 자연스럽게 도전하는 경향이 거의 없지만, LLM-as-an-Investigator는 대안 가설을 유지하고, 차별화된 질문을 통해 사용자 제안을 가설로 간주하여 검증하는 방식을 채택했습니다. 이 논문의 결과는 검사 대상의 정확도 향상과 사용자 주도 시콥언시 감소를 보였습니다.



### The Illusion of Multi-Agent Advantag (https://arxiv.org/abs/2606.13003)
- **What's New**: 이 논문은 자동 생성된 다중 에이전트 시스템(MAS)이 단일 에이전트 시스템(SAS)보다 잘 작동한다는 기존 이론에 의문을 제기합니다. 연구자는 자동 MAS가 실제로 SAS인 Chain-of-Thought with Self-Consistency (CoT-SC)보다 성능이 떨어진다는 것을 입증했습니다. 특히, 연구에서는 기존의 평가 프레임워크가 다중 에이전트 시스템의 중요한 설계 문제를 간과하고 있다는 사실도 밝혔습니다.

- **Technical Details**: 이 연구에서는 MAS의 성능을 평가하기 위해 진단용 합성 데이터셋(Synthetic Multi-Hop Financial Reasoning, SMFR)을 도입했습니다. SMFR은 명확한 하위 작업 구조와 맥락 분리, 병렬화 가능성을 제공하여 자동 MAS의 성능을 평가할 수 있도록 설계되었습니다. 또한, 전문가가 설계한 MAS는 이러한 구조가 잘 갖춰질 경우 MAS의 이점이 실제로 발생할 수 있음을 보여주었습니다.

- **Performance Highlights**: 연구 결과, 자동 MAS는 많은 설정에서 SAS의 성능에 미치지 못하며 오히려 비용 효율성 면에서 CoT-SC의 성능이 뛰어남을 보여주었습니다. 전문가 설계 MAS는 구조적 체계를 갖출 경우 복잡한 작업에서도 뛰어난 성능을 발휘하는 것으로 나타났습니다. 전체적으로, 자동 MAS의 성능 이점이 복잡성의 표면적 요소에 의해 발생하는 경향이 있음을 발견하였습니다.




New uploads on arXiv(cs.CL)

### JMMMU: A Japanese Massive Multi-discipline Multimodal Understanding Benchmark for Culture-aware Evaluation (https://arxiv.org/abs/2410.17250)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 일본 문화 맥락에 기반한 전문 수준의 작업을 평가하기 위해 설계된 첫 번째 대규모 일본어 벤치마크인 JMMMU(Japanese MMMU)를 소개합니다. 이 벤치마크는 두 가지 상호 보완적인 하위 집합으로 구성되어 있습니다: 문화 비독립적(Culture-Agnostic, CA) 하위 집합과 문화 특정(Culture-Specific, CS) 하위 집합.

- **Technical Details**: JMMMU는 1,320개의 질문과 1,118개의 이미지를 포함하고 있으며, 28개의 다양한 과목을 다룹니다. CA 하위 집합에서는 영어로 동일한 내용의 질문과의 직접 비교를 가능하게 하고, CS 하위 집합에서는 일본 문화에 적합한 새로운 질문을 포함합니다. 이 연구는 15개의 오픈 소스 LMM과 3개의 고급 독점 LMM을 평가하여 발견된 주요 성과를 분석합니다.

- **Performance Highlights**: JMMMU에서 평가된 모델의 전반적인 성능은 최대 58.6%로, 일본 문화에 대한 이해가 부족함을 보여줍니다. CA 하위 집합에서는 대부분의 모델이 일본어로 질문했을 때 영어보다 성능이 떨어졌고(CA 하위 집합에서 최대 8.6% 하락), CS 하위 집합에서 일본 데이터셋으로 훈련된 모델이 가장 높은 성능을 보였습니다. 이 연구는 다문화 성격을 고려한 다양한 고표준 벤치마크 개발의 필요성을 강조하고 있습니다.



### Large Language Models Empowered Personalized Web Agents (https://arxiv.org/abs/2410.17236)
Comments:
          The code and data are available on the project website this https URL

- **What's New**: LLM 기반 개인화 웹 에이전트의 중요성을 강조하며, 개인화된 데이터를 통합하여 사용자의 지침을 더 잘 이해하고 맞춤형 행동을 실행하는 방법을 제안합니다.

- **Technical Details**: 개인화 웹 에이전트를 위한 새로운 벤치마크인 PersonalWAB을 구축하여, 사용자 지침, 개인화된 사용자 데이터, 웹 기능 및 세 가지 개인화된 웹 작업에 대한 평가 패러다임을 포함합니다. 또한, PUMA(Personalized User Memory-enhanced Alignment) 프레임워크를 통해 LLM을 개인화된 웹 에이전트 작업에 맞추도록 조정합니다.

- **Performance Highlights**: PUMA는 PersonalWAB에서 기존의 웹 에이전트 성능을 초월하여 개인화된 사용자 지침 및 선호도와 더 잘 정렬되어 보다 지능적이고 맞춤화된 웹 서비스를 제공할 수 있음을 입증합니다.



### Fine-Tuning Large Language Models to Appropriately Abstain with Semantic Entropy (https://arxiv.org/abs/2410.17234)
Comments:
          Accepted to NeurIPS Safe Generative AI Workshop 2024

- **What's New**: 본 연구에서는 기존의 ground-truth labels(정답 레이블)에依 зависимост(의존성)를 없애고, 모델의 내적 통찰을 기반으로 한 semantic entropy(의미 엔트로피)를 이용한 fine-tuning(미세 조정) 방법을 제안합니다.

- **Technical Details**: Semantic entropy는 모델 내부의 불확실성을 측정하는 지표로, 외부 레이블 없이 LLM(대형 언어 모델)의 hallucination(환각) 현상을 완화하기 위해 사용됩니다. 이 방법은 기존 연구 결과와 비교해도 손색이 없는 성능을 보이며 짧은 형식뿐만 아니라 긴 형식의 생성에서도 강력한 성과를 나타냅니다.

- **Performance Highlights**: 제안된 방식은 여러 데이터셋에서 기존의 미세 조정 방법들과 동등하거나 그 이상의 성능을 보여주며, LLM의 환각 현상 완화에 있어 효과적임을 입증하였습니다.



### Dhoroni: Exploring Bengali Climate Change and Environmental Views with a Multi-Perspective News Dataset and Natural Language Processing (https://arxiv.org/abs/2410.17225)
Comments:
          In Review

- **What's New**: 이 논문에서는 방글라데시의 기후 변화와 환경 뉴스에 대한 새로운 데이터셋인 Dhoroni를 소개합니다. 이 데이터셋은 2300개의 주석이 달린 방글라 뉴스 기사로 구성되어 있으며, 정치적 영향, 과학적/통계적 데이터, 진위성, 입장 감지, 이해관계자 참여 등 다양한 관점을 제공합니다.

- **Technical Details**: Dhoroni 데이터셋은 10개의 다양한 관점을 기준으로 세 명의 주석자가 주석을 달며, 각 기사는 기후/환경 뉴스 주제, 관련 당국, 데이터 출처와 보고서, 진위성 등을 포함합니다. 또한, BanglaBERT-Dhoroni라는 새로운 기초 모델 계열이 데이터셋을 기반으로 미세 조정되어 기후 관련 의견 탐지에 활용됩니다.

- **Performance Highlights**: Dhoroni 데이터셋과 BanglaBERT-Dhoroni 모델은 방글라 방언의 기후 담론 분석을 통해 환경 관리, 정책 및 경제적 통찰력을 제공합니다. 이 연구는 기후 변화의 사회경제적 영향을 분석하고, 정책 결정의 효과성을 높이며, 지역 사회와 정책 입안자 간의 간극을 해소하는 데 기여합니다.



### Context-aware Prompt Tuning: Advancing In-Context Learning with Adversarial Methods (https://arxiv.org/abs/2410.17222)
- **What's New**: 본 논문은 Context-aware Prompt Tuning (CPT)이라는 새로운 방법을 제안합니다. 이 방법은 In-Context Learning (ICL), Prompt Tuning (PT) 및 adversarial 공격에서 영감을 받아 개발되었습니다.

- **Technical Details**: CPT는 입력 앞에 학습 예제를 연결하는 ICL 전략을 기반으로 하며, PT와 유사한 학습 방식을 통해 반복 최적화를 통해 컨텍스트 임베딩을 개선하는 방식입니다. CPT는 특정 컨텍스트 토큰을 조정하며, 입력은 레이블을 기반으로 조정하여 손실을 최소화하는 데 집중합니다.

- **Performance Highlights**: CPT는 다양한 LLM 모델을 사용하여 여러 분류 작업에서 뛰어난 정확도를 달성하며, 전통적인 방법보다 거의 모든 시나리오에서 더 높은 성능을 보입니다.



### MiniPLM: Knowledge Distillation for Pre-Training Language Models (https://arxiv.org/abs/2410.17215)
- **What's New**: 이 논문에서는 Knowledge Distillation (KD)을 통해 언어 모델 (LM)의 효율적, 유연하며 효과적인 사전 학습을 위한 MiniPLM이라는 새로운 프레임워크를 제안합니다. MiniPLM은 사전 학습 데이터 분포를 개선하여 교육 데이터를 정제하는 방식으로 동작합니다.

- **Technical Details**: MiniPLM은 Difference Sampling 기법을 사용하여 대형 모델과 소형 모델 간의 차이를 기반으로 학습 인스턴스를 선택합니다. 이 기법은 사전 학습을 위해 Teacher LM의 지식을 오프라인으로 추론하고, 여러 Student LM에 대한 지식 증류를 가능하게 하며, 이 과정에서 추가적인 학습 시간 비용을 발생시키지 않습니다.

- **Performance Highlights**: MiniPLM은 200M, 500M 및 1.2B 크기의 Student LM을 1.8B Teacher LM을 사용하여 사전 학습한 결과, 9개의 일반적인 다운스트림 작업에서 Student LM의 제로샷 성능을 향상시키고, 언어 모델링 능력을 개선하며, 사전 학습 계산을 줄였습니다. 실험을 통해 MiniPLM이 데이터 수요를 2.4배까지 감소시켜 웹 크롤링 데이터를 효율적으로 활용함을 보여주었습니다.



### Exploring Possibilities of AI-Powered Legal Assistance in Bangladesh through Large Language Modeling (https://arxiv.org/abs/2410.17210)
Comments:
          In Review

- **What's New**: 방글라데시의 법률 시스템을 지원하기 위해 개발된 전문화된 대규모 언어 모델 (LLM)인 GPT2-UKIL-EN에 대한 연구 결과를 발표합니다.

- **Technical Details**: UKIL-DB-EN 데이터셋을 구축하여 방글라데시 법률 문서로부터 정보를 수집하고, GPT-2 모델을 이 데이터셋에 대해 미세 조정하여 방글라데시의 법률 지원에 적합한 LLM을 개발하였습니다.

- **Performance Highlights**: 모델은 전문가 의견을 포함한 사례 연구를 통해 엄격하게 평가되었으며, 법률 문제 해결을 위한 잠재력을 보여주었습니다. 그러나 모델의 정확성, 신뢰성 및 안전성을 향상시키기 위한 추가 개선이 필요합니다.



### VoiceBench: Benchmarking LLM-Based Voice Assistants (https://arxiv.org/abs/2410.17196)
Comments:
          Work in progress. Data is available at this https URL

- **What's New**: VoiceBench라는 새로운 벤치마크가 도입되어 LLM 기반 음성 비서의 다면적 평가를 가능하게 합니다.

- **Technical Details**: VoiceBench는 일반 지식, 지시 수행 능력 및 안전성을 평가하기 위해 실제 및 합성된 음성 지시를 포함하며, 다양한 발표 스타일, 환경 조건 및 내용 변화를 포함하는 테스트 케이스를 설계했습니다.

- **Performance Highlights**: 현재 LLM 기반 음성 비서 모델의 한계가 드러났으며, 전통적 ASR 시스템과 LLM의 조합모델 사이의 성능 차이를 강조했습니다.



### From Attention to Activation: Unravelling the Enigmas of Large Language Models (https://arxiv.org/abs/2410.17174)
Comments:
          10 pages

- **What's New**: 본 연구에서는 자가 회귀 Transformer에서 두 가지 이상한 현상, 즉 주의 메커니즘의 첫 번째 토큰의 지배와 숨겨진 상태의 큰 이상치 활성화 문제를 조사합니다. 특히, Llama 모델을 포함한 대규모 언어 모델들이 98%의 주의 헤드에서 첫 번째 토큰에 최대한 집중하는 현상이 발견되었으며, 이는 softmax 함수 때문임을 밝힙니다. 이 문제를 해결하기 위해 softmax-1이라는 새로운 구성 방식을 제안합니다.

- **Technical Details**: 모델의 성능을 향상시키기 위해 새로운 옵티마이저인 OrthoAdam을 도입하여, orthogonal matrices를 사용하여 기울기를 변환하고 이로 인해 발생하는 이상치 문제를 해결합니다. 연구 결과, 첫 번째 토큰에 대한 주의를 65%에서 3.3%로 줄이고, 활성화의 kurtosis를 1657에서 3.1로 낮추며, 4비트 가중치 양자화(perplexity penalty) 하에서 3565에서 0.3으로 크게 개선되었습니다.

- **Performance Highlights**: 제안된 방법들은 기존의 방법들이 해결하지 못했던 문제들(특히 양자화 후 성능 저하)을 극복할 수 있도록 하여, 기본적인 알고리즘을 사용하여도 Transformer 모델의 성능을 유지할 수 있도록 합니다. 최종적으로, 우리의 방법은 Transformers이 8비트와 4비트 양자화 하에서도 성능 저하 없이 작동하도록 하며, 이는 대규모 언어 모델의 응용 가능성을 크게 확장합니다.



### Self-calibration for Language Model Quantization and Pruning (https://arxiv.org/abs/2410.17170)
Comments:
          Work in progress

- **What's New**: 본 논문은 self-calibration을 제안하여 외부 데이터에 의존하지 않고 모델 자체를 활용해 합성(calibration) 데이터를 생성함으로써 모델 압축을 개선하는 방법을 소개합니다. 이를 통해 기존의 문제점인 비대표성(calibration examples)이 해결될 수 있습니다.

- **Technical Details**: self-calibration은 모델이 사전 훈련 데이터 배포를 더 잘 근사할 수 있도록 합성(calibration) 데이터를 생성합니다. 전통적으로 post-training(사후 훈련)에서 양자화(quantization)와 가지치기(pruning)는 calibration data에 의존했지만, 이 방법은 신뢰성을 높입니다.

- **Performance Highlights**: self-calibration은 다양한 모델과 압축 방법에 걸쳐 성능을 극대화하며, 실제 데이터 사용 때보다 더 우수한 성능을 내는 경우가 많습니다.



### Interchangeable Token Embeddings for Extendable Vocabulary and Alpha-Equivalenc (https://arxiv.org/abs/2410.17161)
Comments:
          14 pages, 5 figures

- **What's New**: 이 논문은 언어 모델에서 호환 가능한 토큰을 학습하기 위한 새로운 접근 방식을 제안합니다. 이를 통해 새로운 토큰에 일반화할 수 있는 확장 가능한 어휘를 얻고, alpha-equivalence의 원칙을 해결하고자 합니다.

- **Technical Details**: 우리의 방법은 이중 임베딩 접근 방식을 채택합니다. 첫 번째 부분은 모든 호환 가능한 토큰에서 공유되어 기본 개념을 나타내도록 하며, 두 번째 부분은 각 토큰에 대해 무작위로 생성되어 구별 가능성을 부여합니다. 이러한 방법은 Transformer 인코더-디코더 모델에 적용되며, 두 가지 작업인 선형 시간 논리 수식 해결과 확장 가능한 어휘 복사 작업에서 평가됩니다.

- **Performance Highlights**: 우리는 제안한 방법이 더 큰 어휘 크기로 일반화할 수 있는 가능성을 보여주었으며, alpha-equivalence의 학습을 위한 유용한 귀납적 편향을 도입한다는 것을 증명했습니다.



### Can General-Purpose Large Language Models Generalize to English-Thai Machine Translation ? (https://arxiv.org/abs/2410.17145)
Comments:
          Accepted in GenBench EMNLP 2024

- **What's New**: 이번 연구는 일반적인 대형 언어 모델(LLMs)의 저자원 환경에서의 성능 한계를 조사합니다. 특히, 영어-태국어 기계 번역 및 코드 스위칭 데이터셋에서 다양한 LLM과 전문 번역 모델의 성능을 비교합니다.

- **Technical Details**: 연구에서는 Llama-3(8B) 모델과 NLLB-600M, NLLB-3.3B 모델을 사용하여 4비트 양자화 환경에서의 번역 성능을 실험하였으며, BLEU, METEOR, CER 같은 표준 MT 메트릭을 활용했습니다. LLM의 성능 저하를 분석하기 위해 GPT4-o를 사용하여 번역의 실패 모드를 평가했습니다.

- **Performance Highlights**: 연구 결과에 따르면, NLLB-3.3B와 NLLB-600M 모델이 Llama-3 8B보다 여러 메트릭에서 일관적으로 더 좋은 성능을 보였고, 특히 CS 데이터셋에서 NLLB 모델이 더욱 높은 성적을 기록했습니다. 이로써 리소스가 제한된 환경에서 전문 번역 모델의 중요성이 강조되었습니다.



### Aligning Large Language Models via Self-Steering Optimization (https://arxiv.org/abs/2410.17131)
- **What's New**: 이번 논문에서는 Self-Steering Optimization (SSO)이라는 새로운 알고리즘을 소개하여, 자동 정렬 시스템에서 인간 개입이 최소화된 고품질 선호 신호를 자율적으로 생성하는 방법을 제안합니다. 이는 수동 주석 없이 선호 학습을 지원하는 중요한 개선입니다.

- **Technical Details**: SSO 알고리즘은 1) 선택된 응답과 거절된 응답 간의 일관된 간격을 유지하고, 2) 현재 정책 모델의 학습 능력에 맞게 두 응답이 정책에 적합하도록 하는 방식으로 선호 신호의 정확성을 유지합니다. 또한 이 알고리즘은 정책 모델의 온라인 및 오프라인 훈련과 보상 모델 훈련의 개선에 기여합니다.

- **Performance Highlights**: SSO는 Qwen2와 Llama3.1 모델에서 유효성을 검증하며, 일관된 향상 결과를 보여줍니다. 특별히 주목할 만한 것은, SSO가 수동 주석이나 외부 모델 없이도 여러 객관적 기준과 주관적 평가 세트에서 성능 향상을 이루어낸 점입니다.



### Exploring RL-based LLM Training for Formal Language Tasks with Programmed Rewards (https://arxiv.org/abs/2410.17126)
Comments:
          Accepted at BNAIC 2024

- **What's New**: 이 논문은 Proximal Policy Optimization (PPO)을 사용하여 명시적으로 프로그래밍된 보상 신호로부터 직접 강화 학습(Direct Reinforcement Learning)을 수행하는 가능성을 탐구합니다. 이는 간접 학습의 접근 방식인 인간 피드백을 통한 보상 모델을 매개로 하는 것과는 대조적입니다.

- **Technical Details**: 이 연구는 수학 및 프로그래밍과 같은 형식 언어(formal languages)로 표현된 작업에 중점을 두며, 명시적 보상 함수(explicit reward functions)를 프로그래밍하여 생성된 출력을 자동으로 평가할 수 있는 모델을 구축했습니다. 실험은 감정 정렬(sentiment alignment) 작업, 간단한 산술(arithmetic) 작업 및 더 복잡한 게임 합성(game synthesis) 작업을 포함합니다.

- **Performance Highlights**: 연구결과, 두 개의 형식 언어 작업에 대한 순수 RL 기반 훈련은 도전적이며, 간단한 산술 작업에서도 성공이 제한적이라는 점이 발견되었습니다. 탐사를 돕기 위해 새로운 배치 엔트로피 정규화(batch-entropy regularization) 항이 제안되었지만, 훈련은 여전히 완전히 안정적이지 않습니다. LLM의 직접 RL 훈련은 새로운 작업을 학습하는 것보다 상대적으로 작은 변경 사항, 즉 정렬(alignment)에 보다 적합할 수 있다는 것이 시사됩니다.



### Enhancing Answer Attribution for Faithful Text Generation with Large Language Models (https://arxiv.org/abs/2410.17112)
Comments:
          Accepted to KDIR 2024 (part of IC3K 2024)

- **What's New**: 이번 연구에서는 Large Language Models (LLMs)에서의 answer attribution(답변 귀속) 프로세스를 개선하기 위한 새로운 방법을 제안하고 있습니다. 기존의 방법들을 분석하고, 더 독립적이며 맥락화된 주장을 통해 답변의 정확도와 신뢰성을 높이고자 합니다.

- **Technical Details**: 우리는 다음과 같은 주요 요소에 초점을 맞추었습니다: (1) 답변 분할(answer segmentation) 및 증거 검색(evidence retrieval)과 같은 서브 태스크의 효과를 평가하고, (2) 발견된 단점을 바탕으로 새로운 개선 방안을 제시하며, (3) 이러한 새로운 방법들이 answer attribution 컴포넌트의 성능을 어떻게 개선하는지에 대한 수치적 및 질적 분석을 수행합니다.

- **Performance Highlights**: 제안된 방법은 기존의 answer attribution 시스템보다 더 높은 성능을 보이며, 신뢰할 수 있는 정보 제공과 사용자에게 명확한 출처를 제시하는 데 기여합니다.



### Human-LLM Hybrid Text Answer Aggregation for Crowd Annotations (https://arxiv.org/abs/2410.17099)
Comments:
          Accepted in EMNLP 2024

- **What's New**: 이번 논문에서는 crowd 텍스트 답변 집계에서 LLMs (Large Language Models)의 집계자로서의 능력을 조사했습니다. 특히 Creator-Aggregator Multi-Stage (CAMS) 크라우드소싱 프레임워크를 활용하여 인간과 LLM의 하이브리드 텍스트 답변 집계 방법을 제안하였습니다. 이는 기존 연구에서 잘 다루어지지 않았던 text answer aggregation에 초점을 맞추고 있습니다.

- **Technical Details**: 제안된 접근 방식은 세 단계로 구성됩니다. 첫 번째 단계에서 crowd workers (Crowd Creators)가 원시 텍스트 답변을 제공하고, 두 번째 단계에서는 다른 crowd workers (Crowd Aggregators)와 LLMs (LLM Aggregators)가 원시 답변을 집계합니다. 마지막 단계에서 세 가지 자원의 답변 조합을 사용하여 진짜 답변을 추정합니다. 이러한 방법으로 우리는 public crowdsourcing datasets를 기반으로 실험을 수행했습니다.

- **Performance Highlights**: 실험 결과, Crowd Aggregators와 LLM Aggregators 모두 Crowd Creators의 원시 답변보다 더 높은 품질의 추정 답변을 생성할 수 있음을 보여주었습니다. CAMS 접근 방식은 세 자원의 조합을 통해 답변의 품질을 더욱 향상시킬 수 있음을 시사합니다.



### Team Ryu's Submission to SIGMORPHON 2024 Shared Task on Subword Tokenization (https://arxiv.org/abs/2410.17094)
- **What's New**: 이 논문은 SIGMORPHON 2024 공유 작업에 제출된 팀 Ryu의 연구를 다루고 있습니다. 본 제출에서는 형태소 세분화(morphological segmentation) 방법이 서브워드 토크나이저(subword tokenizer)의 일부로 활용될 수 있는지를 탐구합니다. 두 가지 접근법인 통계적 세분화 방법인 Morfessor와 Transformer 기반의 시퀀스-투-시퀀스(segmentation model)를 사용했습니다.

- **Technical Details**: 서브워드 토크나이제이션(subword tokenization)은 NLP 응용 프로그램의 프로세스에서 널리 채택되고 있으며, 자주 사용되는 단어를 유지하고 긴 단어를 짧은 조각으로 쪼개어 어휘 크기를 줄이며, OOV(out-of-vocabulary) 단어를 처리하는 데 유용합니다. 본 연구는 형태소 세분화 방법과 신경망 seq2seq 모델을 기반으로 두 가지 서브워드 토크나이저 시스템을 제안함으로써, 각 방법의 성능을 비교 조사했습니다.

- **Performance Highlights**: 형태소 세분화 방법이 일반적으로 사용되는 서브워드 토크나이저와 유사한 성능을 보일 수 있음을 보여줍니다. 또한, 균형 잡힌 토큰 빈도 분포를 가진 토크나이저가 성능에 긍정적인 영향을 미치며, 이는 자주 사용되는 단어를 고유 토큰으로 유지함으로써 달성됩니다.



### Science Out of Its Ivory Tower: Improving Accessibility with Reinforcement Learning (https://arxiv.org/abs/2410.17088)
- **What's New**: 이번 논문에서는 과학 커뮤니케이션의 문제를 해결하기 위해 언어 모델을 조정하여 전문 용어가 제거된 더 이해하기 쉬운 학술 초록을 생성하는 강화 학습 프레임워크를 제안합니다. 이 모델은 약 6학년 수준으로 가독성을 높이며, 사실 정확성과 높은 언어 품질을 유지합니다.

- **Technical Details**: 이 모델은 'Reinforcement Learning from Accessibility Measures (RLAM)'를 사용하여 단어 및 문장 수준에서의 접근성 보상을 균형 있게 조정합니다. 제안된 접근 방식은 기존의 기계 학습 모델보다도 약 90%의 성능 향상을 이룹니다.

- **Performance Highlights**: 최고의 모델은 대학원 수준의 학술 초록을 고등학교 수준으로 조정하는 데 성공적으로 약 3학년 수준의 가독성 향상을 보였습니다. 이 모델은 과학 연구와 일반 대중 간의 간극을 줄이는 데 기여할 것으로 기대됩니다.



### Data-driven Coreference-based Ontology Building (https://arxiv.org/abs/2410.17051)
- **What's New**: 본 연구에서는 데이터 주도의 온톨로지를 구축하기 위해, 대규모 생물의학 초록 코퍼스에서 핵심참조(코레퍼런스) 관계를 분석합니다. 기존의 수작업으로 제작된 온톨로지의 한계를 극복하고, 텍스트에 기반한 실시간 업데이트 가능한 온톨로지를 제공합니다.

- **Technical Details**: 연구팀은 3천만 개의 PubMed 초록에서 코레퍼런스 체인을 추출하고, 이들 사이의 관계를 추적하기 위해 그래프 구조를 생성했습니다. 각각의 노드는 텍스트 문자열을 나타내며, 간선은 코레퍼런스 체인에서 함께 등장하는 두 문자열 간의 관계를 나타냅니다. 이 그래프의 중심성(betweenness centrality) 측정을 통해 관계의 방향성과 계층 구조를 명확히 규정하였습니다.

- **Performance Highlights**: 이 연구의 결과는 기존 인간이 만든 온톨로지와 유사한 부분이 상당히 많고, 동적으로 업데이트 가능한 개념에 대한 풍부한 데이터 기반 온톨로지를 제공합니다. 이 온톨로지는 접근 가능한 라이센스 아래 공개되며, 코드 또한 함께 배포되어 연구자들이 활용할 수 있는 기초 자료가 됩니다.



### Arabic Dataset for LLM Safeguard Evaluation (https://arxiv.org/abs/2410.17040)
Comments:
          17 pages, 6 figures, 10 tables

- **What's New**: 이 연구는 아랍어 사용자의 안전성을 평가하기 위해 아랍 지역에 특화된 데이터셋을 구성하고, LLM의 안전성에 대한 다른 관점을 분석하는 이중 관점 평가 프레임워크를 제안합니다. 기존의 영어 중심 연구에서 벗어나 아랍어 환경을 반영하여 안전성 관련 질문을 대폭 보완했습니다.

- **Technical Details**: 연구자들은 5,799개의 질문으로 구성된 아랍어 안전성 평가 데이터셋을 구축하였으며, 이 질문들은 직접 공격, 간접 공격 및 민감한 단어를 포함한 무해한 요청을 포함합니다. 또한, 정부 및 반대 의견 두 관점에서 LLM의 응답을 평가하는 이중 관점 평가 프레임워크를 도입하였습니다.

- **Performance Highlights**: 실험 결과, 다섯 개의 아랍어 중심 및 다국어 LLM에서 안전성 성능에 뚜렷한 차이가 나타났습니다. 이는 아랍 지역에 특화된 데이터셋의 필요성을 강조하며, LLM의 책임 있는 배포를 위한 귀중한 통찰을 제공합니다.



### DIRI: Adversarial Patient Reidentification with Large Language Models for Evaluating Clinical Text Anonymization (https://arxiv.org/abs/2410.17035)
- **What's New**: 이 논문은 보호된 건강 정보(Protected Health Information, PHI)를 공유하기 위한 새로운 방법론을 제시합니다. 전통적인 deidentification(비식별화) 방법의 한계를 지적하고, 대규모 언어 모델(LLM)을 활용한 새로운 적대적 접근법을 개발했습니다.

- **Technical Details**: 연구에서 제안한 De-Identification/Re-Identification (DIRI) 방법은 LLM을 사용하여 비식별화된 임상 노트에 해당하는 환자를 재식별하는 과정을 포함합니다. 임상 데이터를 이용하여 세 가지 비식별화 도구(규칙 기반 Philter 및 두 개의 딥러닝 기반 모델인 BiLSTM-CRF, ClinicalBERT)에서 익명 처리된 데이터로 성능을 평가했습니다.

- **Performance Highlights**: ClinicalBERT가 가장 높은 효과성을 보였음에도 불구하고, 본 연구의 도구는 비식별화된 임상 노트의 9%를 재식별할 수 있음을 보여주었습니다. 이는 현재 비식별화 기술의 중요한 약점을 강조하며, 반복적인 개발 및 개선을 위한 도구를 제공합니다.



### SG-FSM: A Self-Guiding Zero-Shot Prompting Paradigm for Multi-Hop Question Answering Based on Finite State Machin (https://arxiv.org/abs/2410.17021)
- **What's New**: 이번 연구에서는 Multi-hop Question Answering (MHQA) 문제를 해결하기 위해 Self-Guiding Finite State Machine (SG-FSM)이라는 새로운 접근 방식을 제안합니다. SG-FSM은 복잡한 질문을 하위 질문으로 나누어 처리하면서, 중간 단계에서의 오류를 보완하고 정확도를 향상시킵니다.

- **Technical Details**: SG-FSM은 각 하위 질문을 순차적으로 처리하며 현재 상황과 결과에 따라 다음 단계를 동적으로 결정하는 방식으로 작동합니다. 이 접근 방식은 기존의 체인 오브 사고 방식과는 다르게 고안되었으며, LLM이 복잡한 문제를 더 효과적으로 해결할 수 있도록 돕습니다. SG-FSM은 네 가지 하위 작업으로 MHQA 문제를 단순화하며, 이는 상태 머신의 다양한 상태와 유사합니다.

- **Performance Highlights**: 우리의 실험 결과에 따르면, SG-FSM은 Musique와 같은 도전적인 데이터셋에서 기존의 강력한 기준선과 비교하여 F1 점수를 거의 두 배로 향상시켰습니다. SG-FSM은 중간 오류를 줄이는 동시에 지정된 출력 형식 준수를 개선하여 평가를 간소화했습니다.



### Exploring Forgetting in Large Language Model Pre-Training (https://arxiv.org/abs/2410.17018)
- **What's New**: 이번 연구는 대형 언어 모델(LLM)의 전처리(pre-training) 단계에서 발생하는 catastrophic forgetting의 존재를 체계적으로 탐색하고 측정했습니다. 기존의 perplexity(PPL)와 같은 전통적인 측정 단위의 한계를 극복하고 새로운 메트릭스를 도입하였습니다.

- **Technical Details**: 기존 연구에서 제안한 경량화된 메모리 리플레이(memory replay) 방법이 전처리 단계에서의 forgetting 문제를 완화하는 데 효과적임을 보였습니다. 또한, 모델의 forgetting curve를 통해 인간의 학습 과정과 유사한 패턴을 분석했습니다.

- **Performance Highlights**: 실험 결과, 메모리 리플레이를 사용한 모델은 개선된 성능을 보였으며, 평균 PPL이 280.66(리플레이 사용) 대 303.63(리플레이 미사용)으로 감소함으로써 forgetting의 존재를 간접적으로 확인했습니다.



### IPL: Leveraging Multimodal Large Language Models for Intelligent Product Listing (https://arxiv.org/abs/2410.16977)
- **What's New**: 최근 Multimodal Large Language Models (MLLMs)의 발전을 바탕으로, C2C 이커머스 플랫폼 사용자들을 위한 Intelligent Product Listing (IPL) 도구가 개발되었습니다. 이 도구는 사용자가 업로드한 제품 사진을 통해 자동으로 제품 설명을 생성합니다.

- **Technical Details**: IPL 시스템은 온라인 multi-modal Retrieval-Augmented Generation (RAG) 과정을 활용하여 사진에서 제품 정보를 추출하고, 이를 참고하여 도메인 특정 MLLM이 제품 설명을 생성합니다. 이 과정은 카테고리 예측, 유사 제품 검색, 핵심 속성 추출 등의 서브 모듈로 구성됩니다.

- **Performance Highlights**: IPL의 실제 사용 결과, 72%의 사용자들이 생성된 콘텐츠에 기반하여 제품 목록을 게시하였고, AI 지원이 없는 목록에 비해 품질 점수가 5.6% 더 높았습니다.



### Learning Mathematical Rules with Large Language Models (https://arxiv.org/abs/2410.16973)
Comments:
          4th MATH-AI Workshop at NeurIPS'24

- **What's New**: 본 논문에서는 대형 언어 모델(LLMs)이 분배 법칙(distributivity)이나 방정식 솔빙(equation solving)과 같은 특정 수학 규칙을 배우는 능력을 연구합니다. 이 연구의 주요 초점은 LLM이 훈련 중에 접하지 않은 맥락에서 이러한 규칙을 일반화하고 재사용하는 능력에 있습니다.

- **Technical Details**: 이 연구에서는 LLM에 특정 수학 규칙을 훈련시키기 위해 합성 데이터(synthetic data)를 구축하는 정교한 방법론을 제시합니다. 데이터는 수학 교과서처럼 만들어지며, 모델은 단순한 방정식 재배치(equation rearrangement), 변수를 포함한 선형 방정식에 대한 솔루션 도출, 방정식에서 변수 격리(isolating variables) 등 다양한 규칙을 학습합니다. 본 연구에는 하향(top-down) 및 상향(bottom-up) 일반화 기법이 포함됩니다.

- **Performance Highlights**: 모델은 수학적 복잡성을 증가시키면서도 이러한 규칙을 어느 정도 일반화할 수 있는 능력이 있음을 보여줍니다. 특히 관찰된 결과는 변수 명칭의 다양성 증가가 LLM의 분배 법칙에 대한 일반화 능력을 향상시킬 수 있다는 점입니다. 이 연구는 LLM의 수학적 언어 처리 능력을 이해하고 향상시키기 위한 중요한 기초를 제공합니다.



### Math Neurosurgery: Isolating Language Models' Math Reasoning Abilities Using Only Forward Passes (https://arxiv.org/abs/2410.16930)
Comments:
          21 pages, 29 figures

- **What's New**: 본 논문에서는 Math Neurosurgery(MathNeuro)라는 새로운 방법을 제안하여 대규모 언어 모델(LLM)에서 수학적 추론(Math reasoning)과 관련된 매개변수를 분리하는 방법을 개발했습니다. 이 방법은 기존 방법에서 수학적 매개변수를 효과적으로 분리할 수 있는 혁신적인 접근 방식을 소개하며, 매개변수 삭제가 전반적인 언어 능력에 미치는 영향을 최소화합니다.

- **Technical Details**: MathNeuro는 LLM의 매개변수가 수학적인 작업과 일반 언어 작업에서 각각 어떻게 중요한지를 분석하여, 수학적 작업에 중요한 매개변수를 고립시키는 방법입니다. 이 방법은 가중치(weights)와 활성화(activations)를 기반으로 하여 파라미터의 중요성을 계산하고, 일반적인 언어 작업에 중요하지 않은 매개변수를 제거함으로써 수학 특화 매개변수를 식별합니다. 이를 통해 다양한 크기의 LLM에서 수학적 추론 능력을 효과적으로 삭제하거나 향상시키는 데 성공했습니다.

- **Performance Highlights**: MathNeuro를 적용한 결과, LLM의 GSM8K(Generalized Synthetic Math 8K) 성능이 4-17% 향상되었습니다. 또한, 이 방법은 단일 샘플을 사용하여도 데이터 효율성이 뛰어남을 보여줍니다. 각 매개변수를 삭제하거나 스케일링(Skaling)하는 과정에서도 비수학적 성능에는 큰 영향을 미치지 않는 것으로 나타났습니다.



### Tracing the Development of the Virtual Particle Concept Using Semantic Change Detection (https://arxiv.org/abs/2410.16855)
Comments:
          CHR 2024: Computational Humanities Research Conference

- **What's New**: 이 연구는 가상 입자(virtual particle)의 개념을 분석하기 위해 Semantic Change Detection(SCD) 기법을 활용한 최초의 시도로, 과거의 질적 연구와 비교하여 가상 입자의 의미 변화 과정을 최신의 컴퓨터 과학적 방법론으로 조망하고 있습니다.

- **Technical Details**: 본 연구에서는 1924년부터 2022년까지의 Physical Review 시리즈에서 수집된 700,000개 과학 논문을 기반으로 가상 입자 개념의 발전을 분석하였습니다. BERT 모델을 도메인 적응(domain adaptation)하여 'virtual'이라는 용어의 문맥별 단어 임베딩(contextualized word embeddings)을 추출하고, SCD 기법을 적용하여 의미 변화와 다의성(polysemy)을 탐구합니다.

- **Performance Highlights**: 연구에 따르면 1950년 이후 가상 입자 개념은 더욱 안정화되었으며, 동시에 더 많은 다의성이 존재하게 되었습니다. SCD 지표는 질적 연구 통찰력 및 Dependency Parsing 결과와 잘 일치하며, 기존 연구의 한계를 보완하는 추가적인 통찰을 제공합니다.



### ETHIC: Evaluating Large Language Models on Long-Context Tasks with High Information Coverag (https://arxiv.org/abs/2410.16848)
Comments:
          15 pages, 5 figures

- **What's New**: 이 논문은 대형 언어 모델(LLM)의 긴 문맥 처리 능력을 평가하기 위한 새로운 벤치마크 ETHIC을 제안합니다. 기존의 평가 방법들은 모델이 문맥 정보를 충분히 활용하고 있는지를 제대로 평가하지 못하는 문제를 지적하며, 정보 범위(information coverage, IC)라는 새로운 지표를 도입하여 이 문제를 해결하고자 합니다.

- **Technical Details**: 정보 범위(IC)는 질문에 답하기 위해 필요한 입력 문맥의 비율을 정량화합니다. ETHIC 벤치마크는 책, 토론, 의학, 법률 분야의 2,648개 테스트 인스턴스로 구성되어 있으며, 각 인스턴스는 문맥의 모든 관련 정보를 활용해야 합니다. 현재 LLM의 성능을 평가하기 위해 설정된 다양한 과제와 기존 벤치마크와의 비교를 포함하여 분석합니다.

- **Performance Highlights**: 현재 LLM은 높은 IC 점수를 요구하는 작업에서 성능이 크게 저하되는 경향이 있으며, 최신 모델을 포함하더라도 많은 문제에서 만족스러운 성능을 보이지 않는 것으로 나타났습니다. 이 연구는 모델의 긴 문맥 처리 능력에서 중요한 도전 과제를 제기하며, 향후 연구 방향을 제시합니다.



### Trustworthy Alignment of Retrieval-Augmented Large Language Models via Reinforcement Learning (https://arxiv.org/abs/2410.16843)
Comments:
          ICML 2024

- **What's New**: 본 논문에서는 Retrieval-Augmented Generation (RAG)을 통한 언어 모델의 신뢰성에 초점을 맞추고 있습니다. 연구자들은 이를 통해 외부 증거를 기반으로 하여 파라메트릭 지식의 간섭을 무시하고 응답하도록 RAG 언어 모델을 정렬하는 과정을 시작했습니다.

- **Technical Details**: 저자들은 Trustworthy-Alignment라는 강화학습 기반 알고리즘을 제안하며, 이를 통해 RAG 언어 모델이 명시적인 감독 없이도 신뢰할 수 있는 상태에 도달할 수 있음을 이론적으로 및 실험적으로 입증합니다. 이는 파라메트릭 지식에 의존하지 않고 맥락 지식만으로 응답할 수 있는 능력을 보여줍니다.

- **Performance Highlights**: 실험 결과, RAG가 제공하는 상황에서 언어 모델은 신뢰할 수 있는 응답을 생성할 수 있는 능력이 있음을 증명하였습니다. 특히, 언어 모델의 성능이 맥락 지식에 따라  향상되었으며, 이는 강화학습을 통해 이루어진 정렬이 성공적이었음을 나타냅니다.



### Assessment of Transformer-Based Encoder-Decoder Model for Human-Like Summarization (https://arxiv.org/abs/2410.16842)
Comments:
          Pre-print

- **What's New**: 이 논문은 transformer 기반 BART 모델을 활용하여 자동 텍스트 요약을 개선하는 방법을 제시합니다. 특히, 다양한 기사에 대한 인간 평가를 통해 요약의 품질을 평가하며, 모델의 성능을 기존의 pretrained 모델과 비교합니다.

- **Technical Details**: 이 연구에서는 seq2seq(시퀀스-투-시퀀스) 프레임워크를 활용하여 transformer 기반의 BART 모델을 사용합니다. 모델은 encoder-decoder 아키텍처를 갖추고 있으며, 훈련 과정에서 다양한 샘플 기사를 다뤄 평가 메트릭으로 ROUGE 점수와 BERTScore를 사용합니다.

- **Performance Highlights**: 임상 결과에 따르면, 인간이 작성한 요약은 finetuned 모델이 생성한 추상적 요약보다 17% 더 사실적 일관성을 나타냅니다. 이는 기존 평가 메트릭들이 사실적 오류를 포착하는 데 민감하지 않음을 드러냅니다.



### Analyzing and Evaluating Correlation Measures in NLG Meta-Evaluation (https://arxiv.org/abs/2410.16834)
- **What's New**: 이번 연구에서는 자연어 생성(NLG) 평가에서 자동 평가 지표와 인간 평가 간의 상관 관계를 메타 평가의 중요한 기준으로 분석하였으며, 12개의 일반적인 상관 측정 방법이 평가 결과에 미치는 영향을 강조하였습니다.

- **Technical Details**: 연구팀은 총 6개의 NLG 평가 데이터 세트와 32개의 평가 지표에서 12개의 일반적인 상관 측정 방법을 분석하였으며, 이는 그룹화 방법과 상관 계수의 차이를 반영하여 이들 간의 관계를 규명하고, 메타 평가의 능력을 평가했습니다. 주요 상관 관계 나는 Pearson correlation(피어슨 상관 계수)이 포함된 글로벌 그룹화를 사용한_measures이 최고의 성능을 보임을 발견했습니다.

- **Performance Highlights**: 연구 결과는 메타 평가 능력을 판단하기 위해 제안된 다양한 상관 측정 방법이 실제 평가 결과에 영향을 미친다는 것을 보여주었으며, 이는 향후 NLG 연구의 평가 기법에 대한 이해를 깊게 할 것으로 기대됩니다.



### Optimizing Chain-of-Thought Reasoning: Tackling Arranging Bottleneck via Plan Augmentation (https://arxiv.org/abs/2410.16812)
- **What's New**: 이 논문에서는 Chain-of-Thought (CoT) 추론 과정을 두 가지로 나누어 모델의 병목 현상이 주로 정렬(arranging) 단계에 있음을 강조합니다. 이 발견을 기반으로, 추상적인 계획(plans)을 통해 정렬 단계를 생성하도록 모델을 가이드하는 새로운 방법을 제안합니다.

- **Technical Details**: 모델의 다단계 추론 능력을 향상시키기 위해, CoT가 정렬과 실행(executing)으로 나뉜다는 새로운 관점을 제시하였습니다. 계획 증강 추론(plan augment reasoning) 및 계획 중심의 감독형 미세 조정(plan-centric supervised fine-tuning) 방법을 도입하여 정렬 병목을 완화합니다.

- **Performance Highlights**: 수학 문제(GSM8k)와 도구 활용(Task Utilization, ToolBench) 벤치마크에서 실험한 결과, 제안한 방법이 CoT 데이터로 직접 미세 조정한 기존 방법보다 성능이 크게 향상되었음을 보여줍니다. 특히 장거리 추론 일반화(long-distance reasoning generalization)에서 뛰어난 성능을 보였습니다.



### Controlled Low-Rank Adaptation with Subspace Regularization for Continued Training on Large Language Models (https://arxiv.org/abs/2410.16801)
- **What's New**: 본 연구는 Large Language Models (LLMs)에서의 catastrophic forgetting을 완화하기 위한 Controlled LoRA (CLoRA)라는 새로운 방법론을 제안합니다. CLoRA는 LoRA 구조를 기반으로 한 서브스페이스 정규화 방법으로, 모델의 용량에 대해 최소한의 제약을 두면서도 출력 변화의 크기를 줄이는 것을 목표로 합니다.

- **Technical Details**: CLoRA는 updating matrix의 null space 방향에 제약을 두어 출력 변화의 크기를 감소시키며, 이는 파라미터 효율적인 미세 조정(parameter-efficient fine-tuning)의 일환으로 구분됩니다. 실험 결과, CLoRA는 기존의 LoRA 후속 방법들에 비해 인도메인 및 아웃 오브 도메인 평가에서 우수한 성능을 보였습니다. CLoRA는 orthogonal regularization을 도입하여 null space 방향에 대한 제약을 부여합니다.

- **Performance Highlights**: CLoRA는 기존 방법에 비해 LLM 미세 조정 작업에서 뛰어난 성과를 보여주었으며, catastrophic forgetting 완화에서도 우수한 효과를 나타냈습니다. 또한, 학습된 모델의 파라미터 분석 결과 CLoRA가 출력 변화의 크기를 줄이면서도 모델 용량에 미치는 영향이 최소화된 것을 확인하였습니다.



### Correct after Answer: Enhancing Multi-Span Question Answering with Post-Processing Method (https://arxiv.org/abs/2410.16788)
Comments:
          Accepted by EMNLP 2024 Findings

- **What's New**: 이번 연구에서 제안된 Answering-Classifying-Correct (ACC) 프레임워크는 Multi-Span Question Answering (MSQA) 문제에서 잘못된 예측을 처리하기 위한 새로운 후처리 전략을 도입했습니다. 이는 기존 모델들이 Gold answers에만 기반하여 훈련된 점에서 개선을 이루고자 하며, 불완전한 예측을 수정하고 잘못된 예측을 제외함으로써 예측의 질을 향상시키는 것을 목표로 합니다.

- **Technical Details**: ACC 프레임워크는 세 가지 단계로 구성됩니다. 첫 번째로, 분류기를 통해 예측을 '정확한 예측', '부분적으로 정확한 예측', '잘못된 예측'으로 분류합니다. 두 번째로, 수정기를 통해 '부분적으로 정확한 예측'을 수정합니다. 마지막으로, '잘못된 예측'을 제외하여 최종 예측을 얻습니다. 이를 통해 각 데이터셋에 대해 정확도(Exact Match, EM) 점수를 제공함으로써 MSQA 성능을 향상시킵니다.

- **Performance Highlights**: 실험 결과, ACC 프레임워크를 적용한 후 MultiSpanQA 데이터셋에서 EM F1 점수가 Tagger-RoBERTa 모델에서 69.05%에서 72.26%로, BART-base 모델에서 65.57%에서 76.31%로 증가했습니다. 이는 ACC 프레임워크가 잘못된 예측을 효과적으로 줄이고 더 많은 정확한 예측을 얻어 예측의 질을 높인다는 것을 보여줍니다.



### Beyond Retrieval: Generating Narratives in Conversational Recommender Systems (https://arxiv.org/abs/2410.16780)
- **What's New**: 이 논문은 대화형 추천 시스템을 위한 자연어 생성 작업에서 REGEN이라는 새로운 데이터셋을 소개합니다. REGEN은 Amazon 제품 리뷰 데이터를 풍부한 사용자 내러티브로 확장하여 생성된 데이터셋입니다. 이 데이터셋은 사용자 선호도에 대한 개인화된 설명, 추천 아이템에 대한 제품 추천 및 사용자 구매 이력 요약을 포함하고 있습니다.

- **Technical Details**: REGEN은 사용자-아이템 상호작용 신호로부터 일관성 있는 자연어 출력을 생성하는 작업과 프레임워크를 도입합니다. 이 연구에서는 협업 필터링(CF) 신호와 콘텐츠 임베딩을 통합하는 융합 아키텍처를 제안하여 LLM의 입력으로 사용합니다. 실험 결과, CF와 콘텐츠 임베딩을 결합함으로써 언어 메트릭에서 4-12%의 성능 향상을 보여주었습니다.

- **Performance Highlights**: 제안된 모델은 사용자의 과거 상호작용을 바탕으로 풍부한 내러티브를 생성할 수 있다는 점에서 인간과 같은 대화형 추천을 생성하는 데 효과적임을 입증했습니다. 또한 데이터를 통해 CF와 콘텐츠 임베딩이 이 새로운 생성 작업에 어떻게 기여하는지 분석하였습니다.



### Context-Aware LLM Translation System Using Conversation Summarization and Dialogue History (https://arxiv.org/abs/2410.16775)
Comments:
          Accepted to WMT 2024

- **What's New**: 이 논문에서는 고객 지원 환경에서 비공식적이고 비구조적인 대화 텍스트 번역의 도전 과제를 해결하기 위해 컨텍스트 인식( context-aware ) LLM 번역 시스템을 제안합니다.

- **Technical Details**: 제안된 시스템은 가장 최근의 두 개의 대화와 이전 대화의 요약을 원시 데이터(raw data)로 활용하여 컨텍스트 길이를 효과적으로 관리합니다.

- **Performance Highlights**: 이 접근 방식은 번역 정확도를 현저히 향상시켰으며, 대화의 일관성과 응집력을 유지하는 데 기여합니다. 이 시스템은 고객 지원 번역 작업에 대한 실용적인 솔루션을 제공합니다.



### Forewarned is Forearmed: Leveraging LLMs for Data Synthesis through Failure-Inducing Exploration (https://arxiv.org/abs/2410.16736)
- **What's New**: 이 논문에서는 ReverseGen이라는 새로운 접근 방식을 제안합니다. 이 방법은 대형 언어 모델(LLMs)의 약점을 드러내는 효과적인 훈련 샘플을 자동으로 생성합니다. 특히, 실패를 유도하는 쿼리를 생성하도록 훈련된 전담 프로포저(proposer)를 도입하여, 이를 통해 모델의 성능을 개선할 수 있습니다.

- **Technical Details**: ReverseGen은 타겟 모델의 실패 사례를 바탕으로 효과적인 합성 데이터를 생성하는 새로운 패러다임입니다. 프로포저는 타겟 모델이 비효율적인 응답을 생성하도록 유도하는 쿼리를 만드는 데 최적화되어 있습니다. 이 프로포저는 새로운 지침을 생성하여 타겟 모델의 약점을 지속적으로 탐색할 수 있도록 학습합니다. 이 접근 방식은 다양한 규모(3B, 7B, 8B)의 모델에 적용 가능합니다.

- **Performance Highlights**: 실험 결과, ReverseGen으로 생성된 데이터로 미세 조정된 모델은 인간이 주석한 데이터 또는 일반 모델에서 생성된 데이터로 훈련된 모델들보다 안정적이고 높은 성능을 보였습니다. 예를 들어, 안전성 평가에서 ReverseGen은 Llama-2-7b-chat에 대해 이전 방법보다 18배 많은 취약 사례를 생성하며, 정직성 보정에서는 Vicuna-7b-v1.5의 보정 점수가 8.84% 증가했습니다.



### Magnetic Preference Optimization: Achieving Last-iterate Convergence for Language Models Alignmen (https://arxiv.org/abs/2410.16714)
Comments:
          Under review

- **What's New**: 이 논문에서는 Magnetic Preference Optimization (MPO)라는 새로운 접근 방식을 소개하고 있습니다. MPO는 기존 방법들의 한계를 극복하면서 원래 게임의 Nash equilibrium (NE)으로 마지막 반복 수렴을 보장합니다.

- **Technical Details**: MPO는 Magnetic Mirror Descent (MMD)에 기반하여 RLHF 설정에 맞춰 이론적 통찰을 적용하여 개발되었습니다. 이 방법은 마지막 반복 수렴을 달성할 수 있으며 특히 LLM의 미세 조정에 적합합니다. 또한, 정책을 NE로 유도하는 주기적으로 업데이트되는 자기 정책을 활용합니다.

- **Performance Highlights**: MPO는 LLM의 성능을 상당히 향상시키는 결과를 보였으며, 자가 대결(self-play) 방법이 선호 정렬(preference alignment)에 미치는 잠재력을 강조합니다.



### Atomic Fact Decomposition Helps Attributed Question Answering (https://arxiv.org/abs/2410.16708)
- **What's New**: 본 논문에서는 Atomic fact decomposition 기반의 Retrieval and Editing (ARE) 프레임워크를 제안합니다. 이 프레임워크는 생성된 장문의 답변을 분자 조항(molecular clauses)과 원자 사실(atomic facts)로 분해하여, 더욱 정확하고 신뢰할 수 있는 답변 및 증거 보고서를 생성합니다.

- **Technical Details**: ARE는 instruction-tuned LLM을 활용하여 장문의 답변을 원자 수준으로 분해한 뒤, 이를 검색 엔진을 이용하여 관련 증거를 검색합니다. 그런 후, ARE는 LLM 기반 검증기를 통해 증거와 원자 사실 간의 관계를 분류하고, 필요한 경우 원자 사실을 수정하거나 확장하여 재검색하는 과정을 반복합니다. 이를 통해 원래 의도를 보존하며 최종 수정된 답변을 생성합니다.

- **Performance Highlights**: 제안된 ARE 프레임워크는 여러 데이터셋에서 기존 방법들보다 우수한 성능을 보이며, 새로운 평가 지표인 $Attr_{p}$를 도입하여 증거의 정밀도를 평가합니다.



### PLDR-LLM: Large Language Model from Power Law Decoder Representations (https://arxiv.org/abs/2410.16703)
Comments:
          22 pages, 4 figures, 10 tables

- **What's New**: 이번 논문에서는 전통적인 LLM 아키텍처와는 다른 Power Law Graph Attention 메커니즘을 사용해 deductive(연역적) 및 inductive(귀납적) 출력을 생성하는 PLDR-LLM 아키텍처를 소개합니다. 또한 Directed Acyclic Graph(DAG) 손실을 메트릭으로 활용하여 모델 성능 향상을 모색합니다.

- **Technical Details**: PLDR-LLM은 Power Law Graph Transformer를 기반으로 하며, 입력 샘플의 가중 그래프 표현을 덕분에 주의(attention) 메커니즘을 통해 모델의 특성을 학습합니다. 이 과정에서 비선형 변환(non-linear transformations)과 선형 변환(linear transformations)을 모두 활용하여 입력의 특성을 배우고, 적용한 DAG 손실이 성능 향상에 기여하는 구조를 가지고 있습니다.

- **Performance Highlights**: PLDR-LLM은 배치 크기 32로 약 8B 토큰을 사용해 선행 학습(pretraining) 되며, 기존의 스케일된 도트-프로덕트 LLM과 비교해 경쟁력 있는 성능을 보입니다. 또한 이 모델은 기울기 노이즈에 강인하며, DAG 규제를 통해 벤치마크 성적이 향상되었습니다.



### Methods of improving LLM training stability (https://arxiv.org/abs/2410.16682)
- **What's New**: 이번 논문은 대형 언어 모델(LLMs)의 훈련 안정성에 대한 연구를 다루고 있습니다. 저자들은 830M 파라미터를 가진 소형 언어 모델을 사용하고 높은 학습률로 모델을 의도적으로 발산시켜 훈련의 불안정성을 분포하고 분석하는 방법을 제안하였습니다. 특히, Attention 레이어의 logits 성장과 Transformer 블록의 모든 선형 레이어 출력에 대해 심층적으로 분석하였습니다.

- **Technical Details**: 모델 발산을 방지하기 위해 다양한 레이어 정규화 기법을 탐구하였습니다. QK 레이어 후뿐만 아니라 Proj 및 FC2 레이어 후에도 레이어 정규화를 적용하고, QKV 레이어 이후의 레이어 정규화 적용 및 프리 정규화를 제거하는 방법들을 제안하였습니다. 1.5배 더 높은 학습률로 모델이 발산하지 않고도 학습을 수행할 수 있는 것을 입증하였습니다.

- **Performance Highlights**: 제안된 방법들을 활용하여 모델의 Perplexity가 크게 개선되었으며, QK 레이어 정규화 및 softmax 캡핑을 결합하여 학습률을 증가시킴과 동시에 모델의 훈련 안정성을 높이는 성과를 보였습니다.



### SafetyAnalyst: Interpretable, transparent, and steerable LLM safety moderation (https://arxiv.org/abs/2410.16665)
- **What's New**: SafetyAnalyst는 LLM(대형 언어 모델) 기반의 콘텐츠 조정 시스템으로, 해로운 효과와 유익한 효과를 구조적으로 식별하고 이를 수치화하여 안전성을 분류합니다. 이러한 구조적 접근 방식은 기존 시스템이 해내지 못했던 해석 가능성과 조정 가능성을 제공합니다.

- **Technical Details**: SafetyAnalyst는 "harm-benefit tree"를 생성하여 각 요청(prompts)에 대해 가능한 행동, 그 행동의 해로운/유익한 효과(확률, 심각도, 즉각성 포함), 영향을 받는 이해관계자를 식별합니다. 특징(feature) 생성을 위해 SOTA LLM에서 수집된 19,000개 요청을 사용하여 오픈-웨이트 LM을 세부 조정(fine-tuning)합니다.

- **Performance Highlights**: SafetyAnalyst는 F1 점수 0.75로, 기존 LLM 안전 조정 시스템(F1 < 0.72)을 능가하며, 해석 가능성과 조정 가능성을 제공하는 추가 장점을 부각합니다.



### RKadiyala at SemEval-2024 Task 8: Black-Box Word-Level Text Boundary Detection in Partially Machine Generated Texts (https://arxiv.org/abs/2410.16659)
Comments:
          published at naacl 2024

- **What's New**: 이번 논문은 기계 생성 텍스트와 인간 생성 텍스트를 구별하는 데 있어 중요한 도전 과제를 다루고 있습니다. 기존 시스템들이 전체 텍스트의 생성 여부를 판단하는 데 초점을 맞췄다면, 본 연구에서는 문서 내 단어 수준에서 기계가 생성한 부분을 식별하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 제안된 모델은 DeBERTa-CRF를 기반으로 하여 훈련되었습니다. 특히 Conditional Random Fields (CRF)가 포함되어 있어 패턴 인식 능력이 향상되었습니다.본 모델은 M4GT-bench 데이터셋을 사용하여 기계 생성과 인간 작성된 텍스트의 경계를 효과적으로 식별할 수 있도록 설계되었습니다.

- **Performance Highlights**: 모델의 주요 성능 지표는 Mean Average Error (MAE)로, 개발 세트에서 3.53의 MAE를 기록하였으며, 테스트 세트에서는 21.535로 측정되었습니다. 기존 시스템들과 비교했을 때, 제안된 모델은 더 높은 정확도를 보여주었습니다.



### Adsorb-Agent: Autonomous Identification of Stable Adsorption Configurations via Large Language Model Agen (https://arxiv.org/abs/2410.16658)
Comments:
          11 pages, 3 figures

- **What's New**: 이번 연구에서는 효율적인 촉매 탐색을 위해 설계된 Adsorb-Agent라는 Large Language Model (LLM) 에이전트를 소개합니다. Adsorb-Agent는 최소한의 인간 개입으로 시스템 특정 안정 흡착 구성을 도출할 수 있습니다.

- **Technical Details**: Adsorb-Agent는 Solution Planner, Critic, Binding Indexer의 세 가지 주요 모듈로 구성되어 있으며, GPT-4o 모델에 의해 구동됩니다. 사용자가 제공한 화학 기호를 바탕으로 안정적인 흡착 구성을 결정하고, 모듈간의 상호작용을 통해 흡착 에너지를 추정합니다.

- **Performance Highlights**: Adsorb-Agent는 NNH-CuPd3 (111) 및 NNH-Mo3Pd (111) 시스템에서 기존 '휴리스틱' 및 '랜덤' 알고리즘보다 더 적은 초기 세트로 낮은 에너지를 찾는 데 성공하였습니다. 예를 들어, NNH-CuPd3 (111) 시스템에서는 Adsorb-Agent가 22개의 초기 세트를 사용하여 0.733 eV의 에너지를 기록했으며, 이는 기존 방법보다 76 세트가 적은 것입니다.



### Chatting with Bots: AI, Speech Acts, and the Edge of Assertion (https://arxiv.org/abs/2410.16645)
- **What's New**: 이번 논문에서는 대형 언어 모델(dlarge language model) 기반 챗봇이 주장(assertion)을 할 수 있는지에 대한 질문을 다룹니다. 저자들은 챗봇 주장 이론(Thesis of Chatbot Assertion, TCA)을 제안하며, 이는 현재 세대 챗봇이 생성하는 출력물 중 일부가 주장으로 인정될 수 있다고 주장합니다.

- **Technical Details**: 논문은 TCA에 대한 최근의 반대 의견들을 검토하며, 이러한 반대 의견들이 의미가 있다고 주장합니다. 저자들은 주장에 대한 온톨로지적 발전(ontogenesis)을 반영해 '프로토 주장(proto-assertion)'이라는 새로운 범주를 만들어야 한다고 제안합니다. 이 범주는 챗봇을 프로토 주장자로 취급하는 데 적용됩니다.

- **Performance Highlights**: 챗봇을 프로토 주장자로 보는 접근 방식은 챗봇 주장의 딜레마를 해결하는데 만족스러운 해법을 제시합니다.



### A Statistical Analysis of LLMs' Self-Evaluation Using Proverbs (https://arxiv.org/abs/2410.16640)
- **What's New**: 본 연구는 LLMs(대형 언어 모델)의 자기 평가 능력을 새로운 속담 추론(task) 작업을 통해 조사합니다. 이를 위해 성별, 지혜, 사회를 주제로 한 의도가 유사하지만 문구가 다른 300쌍의 속담 데이터베이스를 소개합니다. 연구의 목표는 LLMs의 자기 평가 과정에서 발생할 수 있는 오류를 발견하고, 성 고정관념 및 문화적 이해 부족 문제를 강조하는 것입니다.

- **Technical Details**: LLMs의 일관성을 평가하기 위해 텍스트 일관성 및 수치 일관성을 평가하는 테스트를 제안합니다. 데이터 세트는 각 주제당 약 100개의 속담 쌍을 포함하며, 생성된 응답의 수치 점수와 텍스트 설명의 일관성을 분석하기 위해 자연어 추론(natural language inference) 기반 방법 및 비모수 Siegal-Tukey 검정을 활용합니다.

- **Performance Highlights**: 이 연구를 통해 LLMs의 속담에 대한 평가가 성별 고정관념이나 문화적 이해 부족을 반영할 수 있음을 보여줍니다. 이러한 검증을 통해 LLMs가 대체로 상충하는 평가를 하고 있음을 발견하고, 향후 LLMs의 개선 방향을 제시하고자 합니다.



### Graph-Structured Trajectory Extraction from Travelogues (https://arxiv.org/abs/2410.16633)
- **What's New**: 본 논문에서는 인간의 이동 경로 추출을 위한 그래프 기반 표현 방식을 제안하고, 이에 대한 벤치마크 데이터 세트를 구성하였습니다. 인간의 이동 경로를 기하학적 계층 구조와 방문된 위치의 시간적 순서를 모두 보존하는 그래프 표현이 필요하다고 강조합니다.

- **Technical Details**: 연구에서는 Visit Status Prediction (VSP)과 Visiting Order Prediction (VOP)이라는 두 가지 서브 태스크를 통해 자동 경로 추출을 수행합니다. VSP는 언급된 위치에 대해 방문 상태 레이블을 할당하고, VOP는 '방문된' 지점 사이의 포함 및 전이 관계를 식별합니다. 새로운 데이터 세트인 Arukikata Travelogue Dataset with Visit Status and Visiting Order Annotation (ATD-VSO)가 구축되었으며, 총 100개의 여행기 원문, 3,354개의 geo-entity(노드), 그리고 3,369개의 관계(엣지)가 포함됩니다.

- **Performance Highlights**: 실험 결과, 시스템은 상대적으로 높은 정확도로 방문 상태 레이블 및 전이 관계를 예측할 수 있었지만, 포함 관계 예측은 실패했습니다. 이는 시스템에 지리적 계층 구조의 지식을 주입하는 방법이 앞으로의 중요한 과제임을 시사합니다.



### Distill-SynthKG: Distilling Knowledge Graph Synthesis Workflow for Improved Coverage and Efficiency (https://arxiv.org/abs/2410.16597)
- **What's New**: 본 연구에서는 LLMs를 기반으로 한 새로운 KG(SynthKG) 구축 워크플로우를 제안합니다. 이 방법은 KG 생성을 단일 단계로 간소화한 Distill-SynthKG로 발전되며, 대규모 문서에서도 효율적이고 고품질의 KG를 생성할 수 있게 합니다.

- **Technical Details**: SynthKG는 문서를 관리 가능한 의미적으로 완전한 텍스트 청크로 분할하고, 각 청크마다 LLM을 통해 잠재적 엔티티 및 관계를 추출합니다. 이 과정에서 기존의 질문-답변 데이터셋을 재구성하여 KG 평가를 위한 새로운 데이터셋과 지표를 도입했습니다.

- **Performance Highlights**: Distill-SynthKG는 기존 모든 기준 모델들보다 KG 품질이 뛰어나며, RAG에서의 검색 및 질문 응답 작업에서도 일관된 우수한 성능을 보였습니다. 제안한 그래프 기반 검색 방법 역시 여러 벤치마크 데이터셋에서 KG 검색 방법보다 성능이 우수한 것으로 나타났습니다.



### Dynamic Adaptive Rank Space Exploration for Efficient Sentiment Analysis with Large Language Models (https://arxiv.org/abs/2410.16589)
- **What's New**: 이 논문에서는 Large Language Models(LLMs)을 활용하여 감정 분석을 위한 효율적이고 효과적인 새로운 Dynamic Adaptive Rank Space Exploration (DARSE) 프레임워크를 제안합니다. 이 프레임워크는 다양한 LLM 레이어에 대해 최적의 랭크 조합을 결정하고, 계산 효율성과 모델 성능 간의 균형을 이루기 위해 동적으로 랭크를 할당합니다.

- **Technical Details**: DARSE 프레임워크는 세 가지 주요 구성 요소로 구성됩니다: (1) 최적 랭크의 일반 범위를 식별하기 위한 coarse-grained greedy algorithm, (2) 식별된 범위 내에서 랭크 선택을 세밀하게 조정하는 fine-grained exploration algorithm, (3) 각 LLM 레이어에 대한 최적 랭크 조합을 결정하기 위한 dynamic rank allocation method입니다.

- **Performance Highlights**: DARSE를 적용한 실험 결과, MSE(Mean Squared Error)는 15.1% 개선되었고, 정확도(Accuracy)는 4.3% 향상되었습니다. 이러한 성과는 기존 접근 방식과 비교하여 DARSE의 효과성을 입증합니다.



### A Theoretical Understanding of Chain-of-Thought: Coherent Reasoning and Error-Aware Demonstration (https://arxiv.org/abs/2410.16540)
- **What's New**: 이 논문은 Few-shot Chain-of-Thought (CoT) prompting을 개선하기 위한 새로운 이론적 접근 방식인 Coherent CoT를 제안합니다. Coherent CoT는 이전 단계의 추론을 통합하여, 예측 성능을 향상시키고자 하였습니다.

- **Technical Details**: Coherent CoT는 Stepwise ICL에 비해 Transformer 모델의 예측 성능을 향상시키는데 도움을 줍니다. 중간의 추론 단계에서 발생하는 오류에 더 민감하다는 관찰을 통해, 정확하고 부정확한 추론 경로를 통합할 것을 제안합니다.

- **Performance Highlights**: 제안된 방식을 통해 Coherent CoT의 중간 추론 단계의 정확도를 높임으로써, 전체 CoT 성능을 향상시키는 실험 결과가 제공되었습니다.



### Bayesian scaling laws for in-context learning (https://arxiv.org/abs/2410.16531)
Comments:
          10 pages main text, 26 pages total

- **What's New**: 이번 연구에서는 In-Context Learning (ICL)이 베이시안(Bayesian) 학습자와 유사하게 작용하는 방식을 보여줍니다. 이를 통해 ICL의 새로운 베이시안 스케일링 법칙을 제안하였고, 다양한 사이즈의 GPT-2 모델을 통해 실험적으로 검증하였습니다.

- **Technical Details**: ICL은 기존 모델의 학습 업데이트 없이 복잡한 작업을 수행하는 강력한 기술입니다. 우리의 연구는 ICL이 베이시안 학습을 근사한다는 점을 바탕으로, 모델의 예측 정확도와 ICL 예제 수 간의 상관관계를 설명합니다. 또한, 모델의 정밀도, 작업 사전(task priors), 학습 효율성 및 개별 예제 확률에 대한 해석 가능한 용어를 제공하였습니다.

- **Performance Highlights**: 모든 실험에서 우리 연구의 베이시안 스케일링 법칙은 ICL이 억제된 행동을 다시 드러낼 조건을 정확히 예측했습니다. 이는 LLM의 안전성을 향상시키기 위한 사후 훈련(post-training)의 비효율성에 대한 통찰을 제공합니다.



### AUTALIC: A Dataset for Anti-AUTistic Ableist Language In Contex (https://arxiv.org/abs/2410.16520)
Comments:
          9 pages, 5 figures, 7 tables

- **What's New**: AUTALIC은 맥락에서 반자폐증적 ableist 언어를 감지하기 위해 설계된 첫 번째 벤치마크 데이터셋으로, 총 2,400개의 자폐증 관련 문장과 주위 맥락이 포함되어 있으며, 전문가에 의해 주석이 달려 있습니다.

- **Technical Details**: AUTALIC 데이터셋은 Reddit에서 수집된 자폐증 관련 문장으로 구성되어 있으며, 이 문장들은 neurodiversity 배경을 가진 전문가에 의해 주석 처리되었습니다. 기존의 NLP 도구들이 반자폐증적 ableist 언어의 미세한 표현을 정확히 감지하지 못하는 한계를 보여줍니다.

- **Performance Highlights**: 현재의 언어 모델, 특히 LLMs는 인간의 판단과 일치하지 않으면서 반자폐증적 ableism을 식별하는 데 어려움을 겪고 있어 이 분야에서의 한계를 강조합니다.



### Learning from others' mistakes: Finetuning machine translation models with span-level error annotations (https://arxiv.org/abs/2410.16509)
- **What's New**: 이 논문은 언어 모델의 품질 향상을 위해 전통적인 시퀀스 수준의 주석(annotation) 대신 세분화된 스팬(spans) 수준의 주석을 이용하는 가능성을 탐구합니다. "Training with Annotations (TWA)"라는 새로운 파인튜닝 알고리즘을 개발하여, 기계 번역 모델을 해당 주석 데이터로 직접 학습할 수 있게 합니다.

- **Technical Details**: TWA는 오류가 있는 스팬과 비오류 스팬을 다르게 처리하여, 오류 스팬의 경우 맥락에 따라 해당 스팬에 있는 토큰의 확률을 감소시키는 방법을 사용합니다. 반면, 비오류 토큰에 대해서는 전체 시퀀스의 진행 상황을 고려하여 긍정 신호로 어떤 스팬을 사용할지를 결정합니다. 이 방법은 다차원 품질 메트릭(Multidimensional Quality Metrics, MQM) 데이터를 기반으로 합니다.

- **Performance Highlights**: TWA는 품질 기준으로 필터링된 시퀀스에서 감독 파인튜닝(supervised finetuning) 및 동일 데이터에서 구성된 쌍에 대한 직접 선호 최적화(Direct Preference Optimization, DPO)와 같은 기존 방법들을 초월하는 성능을 보였습니다. TWA는 기계 번역 성능 향상에 있어 스팬 수준의 주석을 효과적으로 활용하고 있음을 시사합니다.



### Rulebreakers Challenge: Revealing a Blind Spot in Large Language Models' Reasoning with Formal Logic (https://arxiv.org/abs/2410.16502)
Comments:
          Preprint

- **What's New**: 본 연구에서는 형식 논리가 자연어 추론에 적용되는 과정에서 발생하는 문제를 지적하고, 논리적 귀결이 현실적 추론과 일치하지 않는 경우를 'rulebreakers'라고 정의합니다. 이를 통해 RULEBREAKERS라는 새로운 데이터셋을 소개하며, 이는 Large Language Models (LLMs) 가 rulebreakers와 비-rulebreakers를 구분할 수 있는 능력을 평가하는 데 사용됩니다.

- **Technical Details**: RULEBREAKERS 데이터셋은 25,600개의 인스턴스로 구성되어 있으며, modus tollens와 disjunctive syllogism에 중점을 둡니다. 연구는 6개의 최신 LLM을 평가하여 토큰 레벨의 정확도와 모델의 신뢰도를 측정합니다. 또한, 모델의 실패 원인은 모델의 세계적 지식과 주의 분산 패턴과 관련이 있을 수 있다고 제안합니다.

- **Performance Highlights**: 대부분의 LLM들은 rulebreakers를 인식하는 데 있어 낮거나 중간 정도의 성능을 보였습니다. 하지만 신뢰도 평가를 통해 rulebreakers를 구분할 수 있는 잠재력을 나타내기도 했습니다. 연구결과는 LLMs의 추론 능력에 한계가 있음을 밝혀내며, 이러한 문제를 통해 LLMs가 특정 논리 규칙을 과도하게 일반화하는 경향이 있는지를 탐구합니다.



### Natural Language Processing for Human Resources: A Survey (https://arxiv.org/abs/2410.16498)
- **What's New**: 이 논문은 인적 자원(HR) 분야에서 자연어 처리(NLP) 기술의 현대적 발전과 응용 가능성에 대한 포괄적인 분석을 제공합니다. 특히, LLMs(large language models)의 발전이 HR 작업에 미치는 영향을 살펴보며, HR 문제 해결을 위한 NLP 연구 기회의 식별과 미래 탐색 방향을 제시합니다.

- **Technical Details**: HR 분야에서의 NLP 작업은 주로 다섯 가지 주요 연구 영역으로 나뉘어 있습니다: 언어 이해(Language Understanding), 정보 추출(Information Extraction), 검색 및 추천(Retrieval and Recommendation), 생성 및 상호작용(Generation and Interaction), 공정성과 편향성 문제(Fairness and Bias). 이 논문은 HR작업에서의 언어적 분석 및 정보 추출의 세부 사항을 다루며, 이를 통해 HR 적용분야에 NLP 기술이 어떻게 기여할 수 있는지를 설명합니다.

- **Performance Highlights**: 이 연구는 NLP 기술이 HR 문제에 대한 해결책을 제공할 수 있는 잠재력을 강조하며, 특히, 스킬 추출(skill extraction)과 같은 특정 하위 작업이 직업 매칭(job matching)과 같은 더 넓은 목표에 어떻게 기여하는지를 드러냅니다. 또한, 실제 데이터셋의 개발과 활용을 통해 연구의 관련성과 영향을 강화할 것을 추천합니다.



### BIG5-CHAT: Shaping LLM Personalities Through Training on Human-Grounded Data (https://arxiv.org/abs/2410.16491)
- **What's New**: 이번 연구에서는 LLM(대형 언어 모델)에 인간의 성격 특성을 사실적으로 내재화하는 문제를 다룹니다. 이전 연구는 주로 성격 특성과 연관된 행동을 설명하는 프롬프트 기반 접근 방식을 사용하였지만, 현실성과 타당성 문제로 고통받았습니다. 이러한 한계를 극복하기 위해, 인간의 성격 표현을 텍스트로 표현한 100,000개의 대화를 포함한 대규모 데이터셋인 BIG5-CHAT를 소개합니다.

- **Technical Details**: BIG5-CHAT는 빅 파이브 성격 이론을 기반으로 하여 실제 인간의 성격 표현을 캡처한 대화 데이터셋입니다. 이 데이터셋을 기반으로, Supervised Fine Tuning(SFT) 및 Direct Preference Optimization(DPO)와 같은 훈련 기반 방법을 탐색하여 LLM의 성격을 자연스럽게 조정합니다. 연구 결과, SFT와 DPO 모두 기존 프롬프트보다 BFI와 IPIP-NEO와 같은 성격 평가에서 더 뛰어난 성과를 보여주었습니다.

- **Performance Highlights**: 훈련된 모델은 성실성(Conscientiousness)과 친화성(Agreeableness)이 높은 경우, 그리고 외향성(Extraversion)과 신경증(Neuroticism)이 낮은 경우에 추론 과제에서 더 나은 성능을 보였습니다. 이는 인간의 인지 성능에 미치는 성격 특성의 영향과 일치하며, LLM의 성격 생성 방법이 더 깊은 심리언어적 특성을 내재화한다는 사실을 демонстра합니다.



### Multi-head Sequence Tagging Model for Grammatical Error Correction (https://arxiv.org/abs/2410.16473)
- **What's New**: 이번 연구에서는 Grammatical Error Correction (GEC) 문제를 해결하기 위해, GEC 작업을 좀 더 단순화하여 7개의 관련 서브태스크로 나누었습니다: 삽입(Insertion), 삭제(Deletion), 병합(Merge), 대체(Substitution), 변환(Transformation), 탐지(Detection), 그리고 교정(Correction).

- **Technical Details**: 제안된 모델은 multi-head 및 multi-task learning 모델로, 7개의 서브태스크 각각에 대해 별도의 분류 헤드(classification head)를 할당합니다. 이를 통해 관련 작업의 신호를 효과적으로 활용할 수 있습니다. 데이터 샘플의 제한성을 완화하기 위해 denoising autoencoder를 사용하여 새로운 합성 데이터셋을 생성하고, character-level transformation을 제안하여 모델의 어휘 범위를 향상시킵니다.

- **Performance Highlights**: 단일/앙상블 모델의 F0.5 점수는 각각 74.4/77.0이며, BEA-19 및 CoNLL-14 테스트에서 68.6/69.1을 기록했습니다. 또한 JFLEG 테스트 세트에서 GLEU 점수가 단일 및 앙상블 모델에서 각각 61.6 및 61.7로 나타났으며, 이는 최근 발표된 최첨단 결과를 상당히 초월하는 성능입니다.



### DocEdit-v2: Document Structure Editing Via Multimodal LLM Grounding (https://arxiv.org/abs/2410.16472)
Comments:
          EMNLP 2024 (Main)

- **What's New**: DocEdit-v2라는 새로운 프레임워크를 소개합니다. 이는 Large Multimodal Models (LMMs)를 활용하여 끝에서 끝까지 문서 편집을 수행하는 혁신적인 방법입니다.

- **Technical Details**: DocEdit-v2의 주요 구성 요소는 세 가지입니다: (1) Doc2Command: 문서 이미지에서 관심 영역(Region of Interest, RoI)을 동시 로컬화하고 사용자 편집 요청을 편집 명령으로 구체화합니다. (2) LLM 기반 Command Reformulation: 전문 소프트웨어를 위해 원래 의도된 편집 명령을 일반 LMM에 적합한 편집 지침으로 변환합니다. (3) 최종적으로, DocEdit-v2는 GPT-4V 및 Gemini와 같은 LMM을 사용하여 문서 레이아웃을 파싱하고 RoI에 대한 편집을 실행하여 편집된 문서 이미지를 생성합니다.

- **Performance Highlights**: DocEdit 데이터셋에 대한 광범위한 실험을 통해, DocEdit-v2는 편집 명령 생성에서 2-33%, RoI Bounding Box 감지에서 12-31%, 전체 문서 편집 작업에서 1-12%의 성능 향상을 보였습니다.



### Beyond Browsing: API-Based Web Agents (https://arxiv.org/abs/2410.16464)
Comments:
          24 pages, 6 figures

- **What's New**: 이 논문에서는 기존 웹 브라우징을 통해 수행되던 작업을 API를 통해 수행하는 두 가지 새로운 종류의 AI 에이전트를 제안합니다: (1) API 호출 에이전트와 (2) 하이브리드 에이전트. 실험을 통해 API 기반 에이전트가 웹 브라우징 에이전트보다 우수한 성능을 보이며 하이브리드 에이전트는 두 가지 방식 모두에서 경험을 활용하여 더욱 개선된 성과를 나타내는 것을 발견했습니다.

- **Technical Details**:  웹 브라우저는 사람의 온라인 활동을 위한 인터페이스로 널리 사용되지만, API (Application Programming Interface)는 기계 간의 직접적인 상호작용을 가능하게 합니다. 본 연구에서는 WebArena라는 실제 웹 탐색 과제의 벤치마크를 통해 API 호출 에이전트와 하이브리드 에이전트를 평가했습니다. API 기반 에이전트는 평균 15% 향상된 성공률을 보였으며, 하이브리드 에이전트는 API 전용 에이전트보다 5% 더 높은 정확성을 기록했습니다.

- **Performance Highlights**: API 기반 에이전트는 웹 브라우징 전용 에이전트에 비해 15%의 성과 향상을 보였고, 하이브리드 에이전트는 두 방식 모두에서 뛰어난 성능을 발휘하여 20% 이상의 절대 개선을 달성했습니다. 이로 인해 하이브리드 방식이 웹 브라우징 전용 방식보다 더욱 일관되고 신뢰할 수 있는 결과를 제공하는 것으로 나타났습니다.



### Comparative Study of Multilingual Idioms and Similes in Large Language Models (https://arxiv.org/abs/2410.16461)
Comments:
          22 pages, 4 figures

- **What's New**: 본 연구는 다양한 언어 간 비유적 언어 해석에 대한 대규모 언어 모델(LLMs)의 비교 성능에 대한 문헌의 공백을 다룹니다. 특히, 유사어(simile)와 관용구(idiom) 해석에 대해 LLM을 평가하고, 여러 프롬프트 엔지니어링 방법의 효과를 탐구합니다.

- **Technical Details**: 이 연구에서는 MABL 및 MAPS라는 두 개의 다국어 데이터셋을 사용하여 LLM의 성능을 평가하며, 페르시아어로 확장된 새로운 평가 세트를 개발했습니다. 프롬프트 엔지니어링 전략(예: chain-of-thought, few-shot, English translation prompts)을 통해 성능 최적화를 시도합니다. 평가 대상 모델은 closed-source 모델(GPT-3.5, GPT-4o mini, Gemini 1.5)과 open-source 모델(Llama 3.1, Qwen2)을 포함합니다.

- **Performance Highlights**: 연구 결과는 (i) 프롬프트 엔지니어링 방법의 성공률이 비유의 종류, 언어, 모델에 따라 달라진다는 점과 (ii) low-resource 언어에서 open-source 모델이 특히 유사어 해석에서 어려움을 겪음을 보여줍니다. (iii) 강력한 LLM을 사용할 때, 여러 언어에 대한 관용구 해석이 포화 상태에 가까워지고 있음을 확인했습니다.



### To the Globe (TTG): Towards Language-Driven Guaranteed Travel Planning (https://arxiv.org/abs/2410.16456)
- **What's New**: 이번 논문에서는 사용자로부터 자연어 요청을 받아 최적의 여행 일정을 생성하는 실시간 데모 시스템인 To the Globe (TTG)를 제안합니다. TTG는 fine-tuned Large Language Model(LLM)을 통해 자연어 요청을 기호형태로 변환하고, Mixed Integer Linear Programming (MILP) 해결기를 사용하여 최적 경로를 도출합니다.

- **Technical Details**: TTG의 전반적인 시스템은 사용자 요청에 응답하는 데 약 5초가 소요되며, 이를 위해 합성 데이터 파이프라인을 개발하여 인간 주석 없이 기호 형태의 사용자 요구, 항공편 및 호텔 정보를 생성합니다. 자연어에서 기호로의 변환 정확도는 약 91%에 달하며, 반환되는 여행 일정은 정답 사용자 요청에 비해 비용 비율이 0.979입니다.

- **Performance Highlights**: 사용자 평가 결과, TTG는 생성된 일정에 대해 35-40%의 높은 Net Promoter Scores(NPS)를 기록하여, 사용자에게 신뢰할 수 있는 최적의 결과물을 제공합니다.



### Does your LLM truly unlearn? An embarrassingly simple approach to recover unlearned knowledg (https://arxiv.org/abs/2410.16454)
Comments:
          21 pages, 2 figures

- **What's New**: 이번 연구는 언러닝(unlearning) 방법을 적용한 대규모 언어 모델(LLM)에서 양자화(quantization)가 잊힌 정보를 회복할 수 있음을 밝혀냈습니다. 이는 기존 언러닝 방법의 실패를 의미하며, LLM의 안전한 사용을 보장하기 위해 양자화를 통한 지식 회복을 방지하는 새로운 목표를 설정합니다.

- **Technical Details**: 연구에서는 기계 언러닝(machine unlearning) 기법을 통해 특정 지식을 제거하는 방법을 살펴보았습니다. 특히, 기존의 언러닝 방법들이 실제로 지식을 잊는지 아니면 숨기기만 하는지를 평가하기 위한 실험을 실시하였으며, 다양한 양자화 기법을 통해 그 결과를 검증했습니다. 기존의 방법에 따르면, 양자화를 거친 언러닝된 모델이 83%의 잊힌 지식을 회복할 수 있음을 보여주었습니다.

- **Performance Highlights**: 본 연구의 실험 결과, 언러닝된 모델이 전체 정밀도(full precision)에서 평균적으로 21%의 잊힌 지식을 유지하며, 4비트 양자화(quantization) 후에는 이 비율이 83%로 증가하는 것을 확인했습니다. 연구팀은 이러한 결과를 바탕으로 saliency mapping 기법을 사용한 새로운 언러닝 전략(Saliency-Based Unlearning with a Large Learning Rate, SURE)을 제안하였습니다.



### Susu Box or Piggy Bank: Assessing Cultural Commonsense Knowledge between Ghana and the U.S (https://arxiv.org/abs/2410.16451)
Comments:
          Accepted to EMNLP 2024

- **What's New**: 이 논문은 AMAMMER\text{\epsilon}라는 테스트 세트를 소개하며, 이는 가나와 미국의 문화적 문맥에 대한 영어 LLM(대형 언어 모델)의 상식 지식을 평가하기 위해 설계된 525개의 객관식 질문으로 구성되어 있습니다.

- **Technical Details**: AMAMMER\text{\epsilon} 데이터 세트는 다단계 참여 프로세스를 통해 구축되었으며, 서로 다른 문화 간의 차이를 반영합니다. 이 과정에서는 응답자들이 올바른 답안과 잘못된 답안 선택을 작성하고, 선택지의 신빙성을 5점 Likert 척도로 평가하며, 최종 검증 단계에서는 새롭게 구성된 객관식 항목에서 최적의 답변 선택을 실시합니다.

- **Performance Highlights**: 테스트 결과, 모델들은 미국 채점자들에게 일치하는 답변 선택을 선호하는 경향이 있으며, 문화적 문맥이 제공될 때(가나 또는 미국) 일부 적응 능력을 나타냈지만, 전반적으로 미국 문맥에서의 성능이 가나 문맥보다 일관되게 우수함을 보여주었습니다.



### Improving Neuron-level Interpretability with White-box Language Models (https://arxiv.org/abs/2410.16443)
- **What's New**: 이번 연구에서는 Sparse coding(희소 코딩)을 모델 아키텍처에 직접 통합하여 신경망의 해석 가능성을 크게 향상시키는 새로운 아키텍처인 Coding RAte TransformEr (CRATE)를 소개합니다. 기존의 후처리 방식이 아닌, 원래부터 해석 가능성을 내장한 구조입니다.

- **Technical Details**: CRATE 아키텍처는 희소한 저차원 구조를 데이터 분포 내에서 포착하도록 설계되었습니다. 이 모델은 신경망의 다양한 계층에서 안정적으로 작동하는 해석 가능성을 보장하며, 활성화된 뉴런의 역할을 명확하게 이해할 수 있도록 합니다.

- **Performance Highlights**: CRATE는 다양한 평가 지표에 대해 최대 103%의 신경 수준 해석 가능성 향상을 보여주었습니다. 이는 모델 크기와 상관없이 일관되게 나타나며, 각 계층에서의 해석 가능성 증대는 CRATE의 강력한 성능을 강조합니다.



### Enhancing Multimodal Affective Analysis with Learned Live Comment Features (https://arxiv.org/abs/2410.16407)
- **What's New**: 본 논문에서는 LCAffect 데이터셋을 구축하였고, 이를 통해 감정 분석을 위한 합성 라이브 댓글(feature) 생성 방식을 소개합니다. 이 방식은 다양한 비디오 장르에서의 감정 인식을 개선하는 데 기여합니다.

- **Technical Details**: LCAffect 데이터셋은 영어와 중국어 비디오에서 생성된 라이브 댓글을 포함하고 있으며, 감정 분석 작업을 위한 멀티모달(모든 매체) 인코더를 통해 합성 라이브 댓글 기능을 생성합니다. 기존 방법들과의 비교를 통해 SOTA(최신 기술) 성능을 달성했습니다.

- **Performance Highlights**: 감정 감지, 감정 인식 및 냉소 감지와 같은 작업에서 SOTA 성능을 달성했으며, 특히 CH-SIMS v2에서는 3.18포인트, MELD에서는 2.89포인트, MuSTARD에서는 3포인트 성능 향상을 보였습니다.



### VipAct: Visual-Perception Enhancement via Specialized VLM Agent Collaboration and Tool-us (https://arxiv.org/abs/2410.16400)
- **What's New**: 이 논문에서는 VipAct라는 새로운 에이전트 프레임워크를 소개합니다. 이 프레임워크는 VLM(vision-language model)의 성능을 향상시키기 위해 다중 에이전트 협업(multi-agent collaboration)과 비전 전문가 모델(vision expert models)을 통합합니다.

- **Technical Details**: VipAct는 태스크 요구 사항 분석(task requirement analysis), 계획(planning), 협조(coordination)를 담당하는 오케스트레이터 에이전트와 이미지 캡션(image captioning)과 같은 특정 작업을 처리하는 전문 에이전트로 구성됩니다. 이 다중 에이전트 접근 방식은 VLM이 세밀한 시각 인식(fine-grained visual perception) 작업을 더 잘 수행할 수 있도록 합니다.

- **Performance Highlights**: VipAct는 다양한 시각 인식 작업을 포함하는 벤치마크에서 평가되었으며, 모든 작업에서 최첨단 베이스라인(state-of-the-art baselines) 대비 성능이 크게 향상되었음을 보여주었습니다. 추가적으로, 다중 에이전트 협업(multi-agent collaboration)이 더욱 상세한 System-2 reasoning을 이끌어내는 데 중요한 역할을 한다는 점이 밝혀졌습니다.



### LLM-based Optimization of Compound AI Systems: A Survey (https://arxiv.org/abs/2410.16392)
- **What's New**: 이번 논문은 LLM(Large Language Model)을 기반으로 한 복합 AI 시스템의 최적화에 대한 새로운 프레임워크를 제시합니다. 이 시스템은 LLM 호출, 검색기(retriever), 코드 인터프리터(code interpreter) 등 다양한 구성 요소가 상호 연결되어 있으며, LLM을 최적화 도구로 사용하는 것도 포함됩니다.

- **Technical Details**: LLM 기반 최적화(LLM-based optimization)는 매개변수를 미세 조정하여 훈련 데이터 세트의 경험적 위험(empirical risk)을 최소화합니다. 이 과정은 정적(static) 또는 동적(dynamic) 프로그램 분석을 통해 LLM에 지시하는 방식으로 이루어지며, 이를 통해 매개변수를 생성하고 조정합니다. 특히, LLM을 코드 생성 및 복잡한 추론을 지원하는 도구로 활용하여 기능을 확장합니다.

- **Performance Highlights**: 이 연구는 LLM을 사용하여 매개변수를 최적화함으로써, 수동으로 최적화 하는 것보다 더 효율적이며, 다양한 작업에 적응할 수 있는 유연성을 제공합니다. 또한, RAG(Retrieval-Augmented Generation) 시스템의 통합도 다루어져, 높은 성능을 보이는 답변 생성을 위한 새로운 접근 방법을 제시합니다.



### KatzBot: Revolutionizing Academic Chatbot for Enhanced Communication (https://arxiv.org/abs/2410.16385)
- **What's New**: KatzBot 은 대학 관련 정보 처리를 위한 혁신적인 챗봇으로, KatzGPT라는 맞춤형 대형 언어 모델(LLM)을 기반으로 합니다. 이 모델은 두 개의 대학 특화 데이터셋(6,280개의 문장 완성 쌍과 7,330개의 질문-답변 쌍)에서 미세 조정되었습니다.

- **Technical Details**: KatzGPT는 Parameter-efficient Fine-tuning (PEFT) 및 Quantized Low-Rank Adaptation (QLoRA) 기법을 사용하여 폭넓은 아카데미적 질문 처리를 위한 설계를 가지고 있습니다. 12백만 개의 파라미터를 가진 이 모델은 복잡한 쿼리 처리를 최적화하고 다양한 대학 관련 질문에 대해 정교한 대화를 생성할 수 있도록 합니다. 또한, Retrieval Augmented Generation (RAG) 기법이 통합되어 있어, 외부 정보 검색을 기반으로 더 정확하고 즉각적인 답변을 제공합니다.

- **Performance Highlights**: KatzBot은 기존 오픈 소스 LLM보다 우수한 성능을 보여주었으며, 특히 사용자 경험과 만족도가 크게 향상되었습니다. KatzBot의 사용자 친화적인 인터페이스는 대학 웹사이트 내에서 필요한 정보를 탐색하는 데 있어 사용자의 효율성을 높여주고 있습니다.



### This Candidate is [MASK]. Letters of Reference and Job Market Outcomes using LLMs (https://arxiv.org/abs/2410.16325)
- **What's New**: 본 논문에서는 LLM(대형 언어 모델)을 활용하여 추천서에서 감정 및 다른 특성을 추출하는 프롬프트 기반 학습 전략을 구현하였습니다. 전통적인 'bag-of-words' 접근 방식보다 높은 예측 능력을 보여 주며, 추천서의 내용이 경제학 학계의 취업 시장에서의 후보자 성과에 명확히 반영된다는 것을 강조합니다.

- **Technical Details**: 연구에서는 LLM을 활용하여 확률 기반의 감정 점수를 추출하고, 레퍼런스 레터의 품질과 길이가 취업 시장에서의 성공을 예측할 수 있음을 보여 주었습니다. 각기 다른 프롬프트를 통해 후보자의 성별과 같은 관찰 가능한 특징에 대한 정보를 추출할 수 있고, 이러한 정보는 모델의 예측 능력을 테스트하는 데 사용됩니다.

- **Performance Highlights**: 프롬프트 기반 감정 추출을 통해 얻은 감정 점수는 고용이 이루어질 가능성과 강한 상관관계를 가지며, 특히 높은 계급 기관에서의 고용을 예측하는 데 있어 더욱 강한 상관 관계를 보입니다. 기존의 전통적인 방법들은 이러한 예측을 수행하는 데 실패했습니다.



### SouLLMate: An Application Enhancing Diverse Mental Health Support with Adaptive LLMs, Prompt Engineering, and RAG Techniques (https://arxiv.org/abs/2410.16322)
Comments:
          26 pages, 19 figures, 8 tables

- **What's New**: 이번 연구에서는 혁신적인 AI 기술을 통해 개인화된, 시대를 초월한 정신 건강 지원을 제공하는 SouLLMate 시스템을 소개하고 있습니다.

- **Technical Details**: SouLLMate 시스템은 LLM(대형 언어 모델), LangChain, RAG(정보 검색 기반 생성), 프롬프트 엔지니어링 등의 기술을 기반으로 하며, 핵심 기능으로는 초기 평가(PMH-A), 대화형 정보 추출(CIE), 위험 감지(SRD), 사전 안내 대화(PGD) 등이 포함됩니다.

- **Performance Highlights**: 이 시스템은 정신 건강 전문의의 진단을 지원하고, 긴 문맥 추론의 정확성을 높이기 위해 KIS(핵심 지표 요약), PQS(사전 질문 전략), SMMR(스택드 다중 모델 추론) 방법을 제안하며, 이는 정신 건강 지원 기술의 접근성 및 효과성을 개선할 것으로 기대됩니다.



### Altogether: Image Captioning via Re-aligning Alt-tex (https://arxiv.org/abs/2410.17251)
Comments:
          accepted by EMNLP 2024; MetaCLIPv2

- **What's New**: 이 논문은 이미지 캡션의 질을 향상시키기 위해 합성 데이터 생성에 집중하고 있습니다. 기존의 방식이 존재하는 alt-text 메타데이터를 무시하고, 투명성이 결여된 점을 보완하기 위한 새로운 접근 방식을 제안합니다. 이 방법은 기존 alt-text를 편집하고 재정렬하는 것으로, 인간 주석자가 여러 라운드를 통해 캡션을 건설하는 과정입니다.

- **Technical Details**: 이 연구에서는 Altogether라는 원칙 기반 접근법을 제안합니다. 이 방법은 기존의 alt-text와 이미지 내용을 정렬하는 과정으로, 주석자들이 기존 alt-text를 바탕으로 이미지를 참조하며 캡션을 수정합니다. 이 과정은 여러 라운드에 걸쳐 수행되며, 그 결과로 리치한 시각 개념을 포함한 캡션이 생성됩니다. 이후 이 데이터를 바탕으로 캡셔너가 훈련됩니다.

- **Performance Highlights**: Altogether 접근법을 통해 생성된 캡션은 기존의 alt-text보다 4% 더 높은 CLIP 점수를 기록했으며, 최신 캡셔너와 비교하여도 도전적인 테스트 세트에서 뛰어난 성능을 보였습니다. 또한, 텍스트-이미지 생성 및 제로샷 이미지 분류 태스크에서도 유의미한 개선이 있었습니다.



### PyramidDrop: Accelerating Your Large Vision-Language Models via Pyramid Visual Redundancy Reduction (https://arxiv.org/abs/2410.17247)
Comments:
          10 pages

- **What's New**: 이 논문에서는 Large Vision-Language Models (LVLMs)에서 이미지 토큰의 중복성을 줄이기 위해 PyramidDrop이라는 새로운 전략을 제안합니다. 이 방법은 모델의 성능을 저하시키지 않으면서 훈련 및 추론 효율성을 향상시킵니다.

- **Technical Details**: PyramidDrop은 LVLM을 여러 단계로 나누고 각 단계의 마지막에서 사전에 정의된 비율만큼 이미지 토큰을 떨어뜨리는 방식으로 작동합니다. 이를 통해 초기 낮은 레이어에서는 모든 이미지 토큰을 유지하고, 깊은 레이어로 갈수록 점진적으로 토큰 수를 줄입니다. 이 과정에서 경량화된 유사성 계산을 사용하여 오버헤드를 최소화합니다.

- **Performance Highlights**: PyramidDrop을 적용한 LLaVA-NeXT 모델은 훈련 시간을 40% 단축시키고, 55%의 추론 FLOPs를 감소시킴에도 불구하고 성능은 비슷합니다. 또한 이 방법은 학습을 필요로 하지 않는 추론 가속화 전략으로도 활용될 수 있으며, FastV와 비교해 성능과 비용 모두 개선되었습니다.



### Towards Reliable Evaluation of Behavior Steering Interventions in LLMs (https://arxiv.org/abs/2410.17245)
Comments:
          Accepted to the NeurIPS 2024 - Workshop on Foundation Model Interventions

- **What's New**: 최근 연구에서 representation engineering 방법이 모델의 행동을 효율적으로 조정하는 데 유망한 성과를 보였으나, 평가 프로세스가 주관적인 시연에 의존해왔다는 점에 주목합니다. 본 논문에서는 네 가지 평가 기준을 제안하며, 이를 통해 현재 방법들의 효과를 보다 객관적이고 정량적으로 측정하고자 합니다.

- **Technical Details**: 본 연구에서 제안하는 평가 파이프라인은 (i) 후속 작업과 유사한 맥락, (ii) 모델의 likelihood 반영, (iii) 서로 다른 목표 행동 간의 비교 가능성, (iv) 기준 비교를 가능하게 하는 네 가지 속성을 포함합니다. 이러한 접근을 통해, Contrastive Activation Addition과 Inference-Time Intervention의 두 가지 representation engineering 방법을 평가하여, 모델의 행동 조정에 있어 이들의 효과성을 정량적으로 분석합니다.

- **Performance Highlights**: 연구 결과로 나타난 바에 따르면, 기존에 보고된 정보를 고려할 때 일부 개입이 기대했던 것보다 효과적이지 않음을 발견하였습니다. 특히, 행동 조정의 성공 여부에서 개입이 증진하는 행동과 억제하는 행동을 명확히 구분하는 새로운 차원을 제시하며, 이는 기존 지표에서 간과된 중요한 통찰을 제공합니다.



### SELA: Tree-Search Enhanced LLM Agents for Automated Machine Learning (https://arxiv.org/abs/2410.17238)
Comments:
          The code is available at this https URL

- **What's New**: 이번 연구에서는 Tree-Search Enhanced LLM Agents (SELA)라는 혁신적인 에이전트 기반 시스템을 소개하며, Monte Carlo Tree Search (MCTS)를 활용하여 AutoML 프로세스를 최적화하는 방법을 제시합니다. 이 접근 방식은 기존의 Fixed pipelines의 한계를 극복하고, 머신러닝 문제를 해결하기 위한 더 효과적인 탐색을 가능하게 합니다.

- **Technical Details**: SELA 프레임워크는 머신러닝 파이프라인 구성을 트리 구조로 표현하며, 에이전트가 실험을 수행하고 전략을 반복적으로 개선할 수 있도록 돕습니다. MCTS는 새로운 전략을 탐색하고, 이전에 잘 알려진 전략을 개선하는 능력을 활용하여, 에이전트가 광범위한 의사 결정 공간을 효율적으로 탐색할 수 있게 합니다.

- **Performance Highlights**: 20개의 다양한 머신러닝 데이터세트를 비교 평가한 결과, SELA는 전통적인 AutoML 시스템 및 에이전트 기반 AutoML 접근 방식에 비해 65%에서 80%의 승률을 기록하며 우수한 성능을 보였습니다. 이는 에이전트 기반 전략이 AutoML에서 높은 잠재력을 가지고 있음을 나타냅니다.



### Automated Spinal MRI Labelling from Reports Using a Large Language Mod (https://arxiv.org/abs/2410.17235)
Comments:
          Accepted to Medical Image Computing and Computer Assisted Intervention (MICCAI 2024, Spotlight). 11 pages plus appendix

- **What's New**: 이 연구에서는 방사선 보고서에서 라벨(Labels)을 자동으로 추출하기 위한 일반적인 파이프라인을 제안합니다. 이는 대형 언어 모델(LLMs)을 활용하여 요추 MRI 보고서에서 스파인 암(spinal cancer), 협착증(stenosis), 척추 분리증(spondylolisthesis), 좌골 신경 압박(cauda equina compression), 그리고 탈장(herniation) 등 다섯 가지 조건을 검증합니다.

- **Technical Details**: 제안된 방법은 두 가지 단계로 구성됩니다: (1) 모델에게 특정 조건을 염두에 두고 보고서를 요약하도록 요청하고, (2) 요약을 바탕으로 이진 라벨을 생성합니다. 이를 통해 Zephyr(7B)와 Llama3 Instruct(8B)라는 오픈소스 대형 언어 모델을 사용하여, GPT-4의 성능을 초과하는 결과를 달성하였습니다.

- **Performance Highlights**: 자동으로 생성된 라벨을 사용하여 훈련한 비전 모델은 임상의가 직접 주석을 달아 훈련한 모델과 유사한 수준의 성능을 나타냈습니다. 이는 의료 이미징 문제 해결을 위한 훈련 데이터셋을 대폭 증가시킬 수 있는 가능성을 보여줍니다.



### Creativity in AI: Progresses and Challenges (https://arxiv.org/abs/2410.17218)
Comments:
          44 pages

- **What's New**: 이 논문은 AI의 창의성에 대한 최근 연구 동향과 주요 성과, 현재 남아 있는 과제를 살펴봅니다. AI 시스템의 창의적 문제 해결 능력, 언어적 창의성, 예술적 창의성, 그리고 과학적 창의성을 중심으로 조사하였습니다.

- **Technical Details**: 최근의 AI 모델들은 시적이고 예술적으로 창의적인 작품인 시, 이미지, 음악을 생성할 수 있는 능력을 지니지만, 창의적 문제 해결 및 추상적 사고가 요구되는 작업에서는 어려움을 겪고 있습니다. 이러한 모델들은 다양한 데이터에 기반한 대규모 파라미터를 사용하여 훈련되었으며, 그 결과가 진정한 창의적 과정인지, 아니면 메모리와 보간(interpolation) 기술의 결과인지에 대한 의문이 제기되고 있습니다.

- **Performance Highlights**: AI 모델은 언어와 예술에서 창의적 산출을 내는 데 능숙하지만, 창의적 문제 해결 및 장기적 일관성 결여 등에서의 한계를 보여줍니다. 또한, 생성된 결과물은 다양성과 독창성 부족, 그리고 헛소리(hallucination) 문제로 어려움을 겪고 있습니다. AI의 창의성을 평가하기 위한 포괄적인 평가 측정의 필요성도 강조됩니다.



### Audio-to-Score Conversion Model Based on Whisper methodology (https://arxiv.org/abs/2410.17209)
Comments:
          5 pages, 7 figures

- **What's New**: 이 연구에서는 Whisper 기반의 Transformer 모델을 개발하여 음악 오디오에서 멜로디와 화음을 추출하고, 이를 ABC 표기로 기록하는 새로운 방법론을 제안합니다. 'Orpheus' Score'이라는 사용자 지정 표기법을 통해 음악 정보를 토큰으로 변환하고, 해당 표기에 맞춰 특수한 Tokenizer를 설계하여 훈련합니다.

- **Technical Details**: 이 연구에서는 데이터 전처리, 포맷팅 및 변환을 포함한 ABC 표기를 위한 전체적인 데이터 처리 프로세스를 설정하였고, 데이터의 다양성과 품질을 높이기 위한 변이 메커니즘을 구현하였습니다. Whisper 모델을 사용하여 680k 시간의 데이터로 약 150,000개의 ABC 스코어를 생성하고, Gaussian 샘플링 기술을 사용하여 데이터의 다양성 및 균형을 보장하는 방법론을 제시하였습니다.

- **Performance Highlights**: 실험 결과, 제안한 모델은 전통적인 알고리즘에 비해 정확도와 성능이 크게 개선되었습니다. 연구는 음악 감정 분석, 작곡 창작, 피치 분석 및 음악 검색 등의 분야에서도 중요한 기여를 할 수 있음을 입증하였습니다.



### Language Model Non-myopic Generation for Reasoning and Planning (https://arxiv.org/abs/2410.17195)
- **What's New**: 본 논문은 LLM(대규모 언어 모델)의 계획(planning) 과정을 최적 제어(optimal control) 관점에서 재조명하고, 새로운 방법인 Predictive-Decoding을 제안합니다. 이 방법은 모델 예측 제어(Model Predictive Control)를 활용하여 계획 정확도를 향상시키고자 합니다.

- **Technical Details**: Predictive-Decoding은 LLM 분포를 예측 경로(foresight trajectories)에 기반하여 재조정(re-weight)하여 초기 오류를 줄이고 비근시(non-myopic) 계획을 촉진합니다. 실험을 통해 수학, 코딩 및 에이전트 작업에서 의미 있는 성능 향상을 보였습니다.

- **Performance Highlights**: Predictive-Decoding은 계산 효율성(computational efficiency)을 입증하였으며, 제한된 계산 자원으로도 검색 기준(search baselines)을 초월하는 성능을 보여줍니다. 이 연구는 LLM의 계획 능력을 최적화하는 데 중요한 통찰력을 제공합니다.



### Improving Pinterest Search Relevance Using Large Language Models (https://arxiv.org/abs/2410.17152)
Comments:
          CIKM 2024 Workshop on Industrial Recommendation Systems

- **What's New**: Pinterest Search의 검색 관련성을 개선하기 위해 대형 언어 모델(LLM)을 검색 모델에 통합하였습니다. 이를 통해 Pins의 관련성을 효과적으로 예측하는 텍스트 표현을 leverage 하였습니다.

- **Technical Details**: 우리는 검색 쿼리와 generative visual language model에서 추출한 캡션을 포함한 콘텐츠 표현을 결합하여 사용합니다. 또한 링크 기반 텍스트 데이터, 사용자 큐레이션 보드, Pin 제목 및 설명 등을 포함하여 모델을 강화하였습니다. 반자동 학습(semi-supervised learning) 접근 방식을 사용하여 데이터 양을 확장합니다.

- **Performance Highlights**: 최종 배포 시스템에서의 성능 향상을 실험적으로 검증하였으며, 다국어 LLM을 활용하여 보이지 않는 언어와 분야를 포함할 수 있도록 시스템이 확장되었습니다.



### PAPILLON: PrivAcy Preservation from Internet-based and Local Language MOdel ENsembles (https://arxiv.org/abs/2410.17127)
- **What's New**: 이번 연구에서는 Privacy-Conscious Delegation이라는 새로운 태스크를 제안하여 사용자의 프라이버시를 보호하면서 API 기반의 강력한 LLM과 로컬에서 호스팅되는 모델을 연결하려는 접근 방식을 탐구합니다. 이를 위해 PUPA라는 새로운 벤치마크 데이터셋을 개발하고, PAPILLON이라는 다단계 LLM 파이프라인을 설계했습니다.

- **Technical Details**: Privacy-Conscious Delegation은 사용자가 로컬에 호스팅하는 약한 모델이 API 기반의 강력한 모델에 대한 요청을 대리하는 방식으로 이루어집니다. PAPILLON은 DSPy를 활용하여 최적의 프롬프트를 찾아내고, Llama-3.1-8B-Instruct와 GPT-4o-mini를 조합하여 PUPA 데이터셋에서 성능을 평가했습니다.

- **Performance Highlights**: PAPILLON의 최적화된 파이프라인은 85.5%의 사용자 쿼리에 대해 높은 응답 품질을 유지하면서 개인 정보 유출을 7.5%로 제한하는 성과를 보였습니다. 이 연구는 향후 사용자 프라이버시를 유지하면서 LLM의 성능을 향상시키는 가능성을 제시합니다.



### Continuous Speech Tokenizer in Text To Speech (https://arxiv.org/abs/2410.17081)
Comments:
          4 pages. Under review

- **What's New**: 이번 연구는 기존의 이산적 (discrete) 음성 토큰화 방식에서 발생하는 정보 손실을 해결하기 위해 연속적 (continuous) 음성 토큰화를 통한 텍스트-음성 변환 (Text-to-Speech, TTS) 모델을 제안합니다. 새로운 방식은 높은 주파수 정보 보존을 도와 TTS 성능을 향상시키는 것으로 나타났습니다.

- **Technical Details**: 연구에서는 새로운 연속 음성 토크나이저를 통해 텍스트-음성 변환 작업을 수행하며, 오토리그레시브 토큰 생성 작업으로 간주합니다. 모델은 음성 인코더를 활용하여 입력 오디오로부터 연속 음성 벡터를 모델링하고, 이후 발생하는 토큰을 언어 모델로 직접 입력합니다. 최적 수송 조건화 흐름 매칭 (OT-CFM) 기법을 통해 음성을 멜 스펙트로그램으로 변환합니다.

- **Performance Highlights**: 연속 음성 토크나이저 기반 TTS 모델은 기존 이산적 방식 대비 더 나은 성능을 보였으며, 특히 WER (Word Error Rate), EMoS (Estimated Mean Opinion Score) 등의 지표에서 우수한 결과를 보였습니다. 실험 결과는 제안한 방법이 TTS 작업에서 통계적으로 유의미한 개선을 이룩했음을 시사합니다.



### UnStar: Unlearning with Self-Taught Anti-Sample Reasoning for LLMs (https://arxiv.org/abs/2410.17050)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 효과적인 아이디어로 'UnSTAR'를 제안합니다. 이 방법은 anti-sample을 활용하여 특정 정보를 선택적으로 잊게 하는 방법을 제시하고 있습니다.

- **Technical Details**: UnSTAR는 anti-sample을 이용하여 LLM의 학습을 취소(unlearn)하는 과정을 돕습니다. anti-sample은 실제 샘플과 반대되는 정보를 통해 연관된 기억을 되돌리도록 설계된 데이터 포인트입니다.

- **Performance Highlights**: UnSTAR를 통해 anti-sample을 사용하는 것이 LLMs의 목표 지향적 잊기에 있어 효율적이고 정밀한 전략이 됨을 보여주었습니다. 이 접근법은 개인 정보 보호 및 모델 수정의 새로운 길을 열어줄 가능성을 갖추고 있습니다.



### Can a Machine Distinguish High and Low Amount of Social Creak in Speech? (https://arxiv.org/abs/2410.17028)
Comments:
          Accepted in Journal of Voice

- **What's New**: 이번 연구는 사회적 크릭(사회적 creak)의 양을 구분하기 위해 머신러닝(ML) 기술을 적용한 점이 새롭습니다. 기존의 감각적 평가와 전통적인 음향 파라미터를 결합하여 사회적 크릭을 분석하는 방식에서 벗어나 자동 분류 방법을 탐색했습니다.

- **Technical Details**: 90명의 핀란드 여성 화자가 생성한 연속 음성 샘플에서 크릭의 양을 두 명의 음성 전문가가 감각적으로 평가했습니다. 이를 바탕으로, 저수준과 고수준의 크릭으로 두 가지 카테고리로 나누고, 음성 신호와 그 라벨을 사용하여 7개의 서로 다른 머신러닝 모델을 훈련시켰습니다. 각 모델은 3가지 스펙트럼 표현을 특징으로 사용했습니다.

- **Performance Highlights**: 연구의 결과, Adaboost 분류기를 사용한 mel-spectrogram 특징과 결정 트리 분류기를 사용한 mel-frequency cepstral coefficient 특징이 각각 71.1%의 정확도로 가장 우수한 성능을 보였습니다. 이 분류 시스템은 향후 머신러닝 기반 연구의 기준점으로 고려될 수 있습니다.



### EnvBridge: Bridging Diverse Environments with Cross-Environment Knowledge Transfer for Embodied AI (https://arxiv.org/abs/2410.16919)
- **What's New**: 최근에 소개된 EnvBridge는 LLM(대형 언어 모델)의 제어 능력을 향상시키기 위해 환경 간 지식 이전(Cross-Environment Knowledge Transfer) 과제를 해결하는 새로운 방법을 제안합니다.

- **Technical Details**: EnvBridge는 세 가지 주요 구성 요소로 이루어져 있습니다: 코드 생성(Code Generation), 메모리 검색(Memory-Retrieval), 이전 지식을 활용한 재계획(Re-Planning with Transferred Knowledge). 이 시스템은 이전 환경에서 성공적으로 실행된 로봇 제어 코드를 저장하고 이를 새로운 환경에 적절히 적용하기 위해 작업 시 시도하는 제어 코드를 LLM을 이용하여 생성합니다.

- **Performance Highlights**: EnvBridge는 RLBench, MetaWorld 및 CALVIN 테스트 환경에서 평균 69%의 성공률을 달성함으로써 기존의 코드 생성 및 재계획 베이스라인을 크게 능가했습니다. 이 방법은 다양한 지식 원천과 작업 지침에 대한 강력한 성능을 보이며, 다양한 환경 및 작업에 대한 적응력을 강조합니다.



### Context-aware Inductive Knowledge Graph Completion with Latent Type Constraints and Subgraph Reasoning (https://arxiv.org/abs/2410.16803)
- **What's New**: CATS(지식 그래프 완성)라는 새로운 접근 방식을 통해 보이지 않는 엔티티를 처리할 수 있는 최초의 LLM 기반 KGC 솔루션이 제안되었습니다.

- **Technical Details**: CATS는 두 개의 주요 모듈로 구성됩니다: Type-Aware Reasoning (TAR) 모듈은 후보 엔티티가 쿼리 관계에서 요구하는 암묵적인 엔티티 유형과 일치하는지 평가하고, Subgraph Reasoning (SR) 모듈은 적절한 추론 경로와 이웃 사실을 선택하여 이들의 상관관계를 평가합니다.

- **Performance Highlights**: CATS는 3개의 널리 사용되는 데이터셋(WN18RR, FB15k237, NELL-995)에서 18개의 실험 설정 중 16개에서 최고의 성과를 보여주었고, 평균 7.2%의 MRR 개선을 달성했습니다.



### Enhancing Low-Resource ASR through Versatile TTS: Bridging the Data Gap (https://arxiv.org/abs/2410.16726)
- **What's New**: 최근 ASR 시스템이 대규모 데이터셋에서 뛰어난 성능을 보여주고 있지만, 자원이 부족한 환경에서는 여전히 챌린지가 존재합니다. 본 논문은 다양한 TTS 모델을 활용한 ASR 데이터 증강 방법이 효과적이며 경제적이라는 것을 보여줍니다.

- **Technical Details**: CosyVoice는 다국어를 지원하는 제로샷 TTS 시스템으로, 감독된 의미 토큰을 기반으로 합니다. 이 시스템은 텍스트 인코더, 음성 토크나이저, 큰 언어 모델(LLM), 조건부 흐름 대응 모델을 포함하여, 높은 내용 일관성과 화자 유사성을 확보합니다. 이를 통해 다채롭고 자연스러운 음성을 생성할 수 있습니다.

- **Performance Highlights**: 다양한 저자원 데이터셋에서의 실험 결과, TTS 모델을 통한 데이터 증강이 ASR 성능을 유의미하게 개선했으며, 특히 텍스트 다양성이 ASR 성능에 더 큰 영향을 미친다는 것을 발견했습니다.대략 50명에서 100명 정도의 화자가 충분하며, 복잡한 접근 방식 없이 간단한 데이터 결합 방식으로도 뛰어난 결과를 얻었습니다.



### DENOASR: Debiasing ASRs through Selective Denoising (https://arxiv.org/abs/2410.16712)
Comments:
          Paper accepted at IEEE ICKG 2024

- **What's New**: 이번 연구에서는 Automatic Speech Recognition (ASR) 시스템의 성능 불균형 문제를 해결하기 위해 DENOASR이라는 새로운 프레임워크를 도입하였습니다. DENOASR은 남성과 여성 간의 단어 오류율(WER) 차이를 줄이기 위해 선택적인 노이즈 제거 기술을 적용합니다.

- **Technical Details**: DENOASR 프레임워크는 DEMUCS 및 Line Enhancement (LE)라는 두 가지 인기 있는 음성 노이즈 제거 방법을 조합하여, 다양한 성별 그룹의 ASR 성능을 균등화하는 효과를 보여줍니다. 이 연구는 OpenAI WHISPER와 NVIDIA NEMO라는 두 개의 최첨단 오픈 소스 ASR 시스템을 활용하여, 여러 벤치마크 데이터셋(TIE, VOX-POPULI, TEDLIUM, FLEURS)에서 실험을 진행하였습니다.

- **Performance Highlights**: 연구 결과, DENOASR을 적용한 ASR 시스템은 남성과 여성 음성 트랜스크립션 간의 단어 오류율 격차를 효과적으로 감소시켰습니다. 자세히 살펴보면, Whisper의 경우 TIE, Vox-Populi, TedLium, Fleurs 데이터셋에서 각각 11%, 100%, 71%, 100%의 개선 효과를 보였고, Nemo는 각각 22%, 100%, 77%, 21%의 개선을 보였습니다.



### Influential Language Data Selection via Gradient Trajectory Pursu (https://arxiv.org/abs/2410.16710)
- **What's New**: 본 논문에서는 Gradient Trajectory Pursuit (GTP) 알고리즘을 제안합니다. GTP는 L0-norm regularized objective를 사용하여 데이터 샘플을 공동 선택함으로써 모델 성능을 극대화합니다. 이 방법은 기존의 개별 샘플 순위화를 따르지 않고, 샘플 중복을 자동으로 제거하며, 더 높은 효율성을 자랑합니다.

- **Technical Details**: GTP는 Gradient Trajectory를 추구하며, 샘플 데이터를 서브스페이스에서 매칭하여 선택합니다. 이 알고리즘은 압축 샘플링(Compressive Sampling) 과정과 분산 프레임워크(Distributed Framework)를 통해 계산 시간을 줄일 수 있습니다. 실험에서 본 알고리즘은 top-k 및 경쟁 알고리즘보다 일관되게 우수한 성능을 보여주었습니다.

- **Performance Highlights**: GTP 알고리즘은 전체 데이터셋의 0.5%만 선택하여 특정한 instruction tuning 작업에서 전체 성능을 달성했습니다. 또한, 이전의 orthogonal matching pursuit 기반 알고리즘보다 최대 1717배의 효율성을 보였습니다.



### Improving Causal Reasoning in Large Language Models: A Survey (https://arxiv.org/abs/2410.16676)
- **What's New**: 이 논문은 대규모 언어 모델(LLMs)의 인과 추론(causal reasoning) 능력을 향상시키기 위한 다양한 연구를 포괄적으로 검토합니다. 연구는 LLM의 역할에 따라 두 가지 카테고리로 나누어집니다: 인과 추론 엔진으로서의 LLM 또는 전통적 인과 추론 방법에 지식을 제공하는 도우미로서의 역할입니다.

- **Technical Details**: LLMs가 인과적 관계를 이해하는 데 어려움이 있으며, 이를 해결하기 위해 다양한 방법론이 소개됩니다. 본 논문은 LLM을 인과 추론 엔진으로 활용하기 위한 파인튜닝(fine-tuning), 프롬프트 엔지니어링(prompt engineering), 도구 통합(tools integration), 대체 접근 방식(alternative approaches)을 다룹니다. 또한 전통적 방법의 도우미로서 정보 추출(information extraction) 및 데이터 생성(data generation) 기술도 다룹니다.

- **Performance Highlights**: 대규모 언어 모델(LLMs)은 기존의 인과 추론 작업에 대한 평가를 통해 여러 과제를 수행하는 데 있어 중요한 성능 개선 기회를 보였습니다. 그러나 인과적 추론의 깊이가 부족하고 고품질 데이터의 부족이 여전히 도전 과제로 남아 있습니다. 따라서 LLM의 인과 추론 구조와 사전 학습(pre-training) 과정에서 인과 추론을 통합하는 것이 향후 연구의 유망한 방향으로 제시됩니다.



### LLMScan: Causal Scan for LLM Misbehavior Detection (https://arxiv.org/abs/2410.16638)
- **What's New**: 본 연구에서는 LLMScan이라는 혁신적인 LLM 모니터링 기법을 소개합니다. 이 방법은 인과성 분석(causality analysis)에 기반하여 LLM의 내부 작동을 체계적으로 모니터링하며, 다양한 형태의 잘못된 행동(misbehavior)을 감지할 수 있는 포괄적인 솔루션을 제공합니다.

- **Technical Details**: LLMScan은 두 가지 주요 구성 요소인 스캐너(scanner)와 탐지기(detector)로 구성됩니다. 스캐너는 프롬프트 토큰(prompt tokens)과 신경 층(neural layers)에 대한 인과성 분석(causality analysis)을 수행하여 인과성 분포도(causal distribution map)를 생성합니다. 탐지기는 이러한 인과성 분포도를 기반으로 학습된 분류기(classifier)로, LLM이 잘못된 행동을 하고 있는지를 런타임(runtime) 중에 평가합니다.

- **Performance Highlights**: LLMScan은 13개 다양한 데이터셋을 사용하여 4가지 유형의 잘못된 행동을 정확하게 식별할 수 있으며, 특히 불신 검출(untruthful responses) 및 유해한 출력(harmful outputs) 탐지에서 평균 AUC가 0.95를 초과하는 뛰어난 성능을 보입니다.



### ViMGuard: A Novel Multi-Modal System for Video Misinformation Guarding (https://arxiv.org/abs/2410.16592)
Comments:
          7 pages, 2 figures

- **What's New**: 새로운 연구에서는 Short-Form Video (SFV)에서의 허위 정보 감지를 위한 ViMGuard라는 딥러닝 아키텍처를 소개합니다. ViMGuard는 영상, 시각적 요소 및 비언어적 오디오를 포함한 세 가지 모달리티를 분석하여 SFV를 사실 확인할 수 있는 최초의 시스템입니다.

- **Technical Details**: ViMGuard는 두 가지 주요 컴포넌트로 구성됩니다. 첫 번째로, Video 및 Audio Masked Autoencoders가 SFV의 비주얼과 비언어적 오디오 요소를 분석하여 정보 제공 의도를 감지합니다. 이후 정보 제공 의도가 있는 것으로 판단되면, Retrieval Augmented Generation 시스템을 통해 발화된 내용의 사실성을 검증합니다.

- **Performance Highlights**: ViMGuard는 기존의 첨단 사실 확인 시스템 세 가지보다 뛰어난 성능을 보였으며, 이는 SFV 사실 확인의 새로운 기준을 설정한 것입니다. 이러한 시스템은 Chrome 확장으로 배포되었으며, GitHub에 모든 코드가 오픈소스로 공개되었습니다.



### Raising the Stakes: Performance Pressure Improves AI-Assisted Decision Making (https://arxiv.org/abs/2410.16560)
- **What's New**: 본 연구는 사람들에게 AI 조언(advice)에 대한 성과 압력(performance pressure)이 어떻게 영향을 미치는지를 분석하였으며, 특히 낮은 성과 압력이 있는 환경에서도 그러한 압력을 유도할 수 있는 방법을 모색하였습니다.

- **Technical Details**: 연구에서는 Amazon Mechanical Turk에서 일하는 일반인들이 가짜 리뷰 감지(fake review detection)와 같은 AI 지원 작업을 수행할 때 성과 압력이 AI 조언 의존도에 미치는 영향을 조사합니다. 성과 압력은 금전적 보상에 대한 손실 회피(loss aversion)를 활용하여 조작되었습니다. 연구 결과, 이해관계가 클수록 (high stakes) AI 조언을 더 적절하게 활용하는 경향이 있음을 발견하였습니다.

- **Performance Highlights**: 성과 압력이 높을수록, 잘못된 AI 조언을 받을 경우 사람들은 해당 조언을 적절히 무시하는 경향이 높아지며, 이는 AI 설명의 유무와는 관계가 없습니다. 또한, XAI(explainable AI)가 성과 압력 하에서 어떻게 작용하는지에 대한 복잡성을 발견하였습니다.



### Large Body Language Models (https://arxiv.org/abs/2410.16533)
- **What's New**: 본 논문에서는 실시간으로 사람처럼 제스처를 생성하는 데 최적화된 새로운 구조인 Large Body Language Models (LBLMs)를 도입합니다. 특히, LBLM-AVA라는 아키텍처는 Transformer-XL과 병렬화된 diffusion 모델을 결합하여 텍스트, 오디오 및 비디오를 포함하는 다중 모달 입력으로부터 인간처럼 보이는 제스처를 생성합니다.

- **Technical Details**: LBLM-AVA는 제스처 생성 기능을 향상시키기 위한 여러 핵심 구성 요소를 통합합니다: 다중 모달-포즈 임베딩(multimodal-to-pose embeddings), 재정의된 주의 메커니즘(attention mechanisms)을 통해 개선된 시퀀스-투-시퀀스 매핑(sequence-to-sequence mapping), 제스처 시퀀스 일관성을 위한 시간적 스무딩 모듈(temporal smoothing module), 그리고 현실감을 높이기 위한 주의 기반 정제 모듈(attention-based refinement module)입니다.

- **Performance Highlights**: LBLM-AVA는 Fréchet Gesture Distance (FGD)를 30% 줄여주며, Fréchet Inception Distance에서 25% 향상을 이뤄내어 기존 접근 방식에 비해 현실적이고 상황에 적합한 제스처 생성에서 최첨단 성능을 달성합니다.



### Allo-AVA: A Large-Scale Multimodal Conversational AI Dataset for Allocentric Avatar Gesture Animation (https://arxiv.org/abs/2410.16503)
- **What's New**: Allo-AVA 데이터 세트는 대화형 AI의 생생한 아바타 애니메이션 생성을 위한 고품질 다중 모드 훈련 데이터를 제공합니다. 이 데이터 세트는 텍스트와 오디오 기반 아바타 제스처 애니메이션을 위해 특별히 설계되었습니다.

- **Technical Details**: Allo-AVA 데이터 세트는 약 1250시간의 다양한 비디오 콘텐츠를 포함하며, 오디오, 전사, 추출된 키포인트와 함께 제공됩니다. 이 데이터는 발화 내용, 음향 특성, 시각적 신호 및 대화 맥락 간의 관계를 포착하는 데 유용합니다.

- **Performance Highlights**: 키포인트 수가 1350억 개 이상이며, 평균 1분당 112,500개의 키포인트가 추출됩니다. 이 데이터 세트의 다양성은 다양한 연령, 성별 및 민족적 배경을 포함하여 아바타 애니메이션의 자연성을 높이는 데 기여합니다.



### End-to-End Transformer-based Automatic Speech Recognition for Northern Kurdish: A Pioneering Approach (https://arxiv.org/abs/2410.16330)
- **What's New**: 저자들은 Whisper라는 사전 훈련된 ASR 모델을 사용하여 북부 쿠르드어(Kurmanji)의 자동 음성 인식(ASR)에 대한 연구를 수행했습니다. 이 연구는 특히 질적으로 부족한 저자원 언어에 대한 맞춤형 미세 조정 기법이 ASR 성능을 크게 향상시키는 방법을 탐구하고 있습니다.

- **Technical Details**: 연구에서는 Whisper 모델을 대상으로 세 가지 미세 조정 전략(기본 미세 조정, 특정 매개변수 미세 조정, 추가 모듈 미세 조정)을 적용하였습니다. 특히 68시간의 북부 쿠르드어 음성 데이터셋을 활용하여, 추가 모듈 미세 조정 기법이 ASR 정확도를 향상시킬 수 있음을 보여주었습니다. 최종적으로 단어 오류율(Word Error Rate, WER)은 10.5%로, 문자 오류율(Character Error Rate, CER)은 5.7%에 도달했습니다.

- **Performance Highlights**: Whisper 모델의 추가 모듈 미세 조정 전략은 ASR 성능에 있어 상당한 개선을 가져왔습니다. 본 연구 결과는 저자원 ASR에 대한 변형된 트랜스포머 모델의 가능성을 강조하며, 다양한 미세 조정 기법의 중요성을 부각시킵니다.



### Feint and Attack: Attention-Based Strategies for Jailbreaking and Protecting LLMs (https://arxiv.org/abs/2410.16327)
- **What's New**: 이번 논문에서는 Large Language Models(LLMs)에 대한 jailbreak 공격을 이해하기 위한 새로운 방법론과 이에 대한 방어 전략을 제시합니다.

- **Technical Details**: 논문은 새로운 메트릭인 Attention Intensity on Sensitive Words (Attn_SensWords), Attention-based Contextual Dependency Score (Attn_DepScore), Attention Dispersion Entropy (Attn_Entropy)를 도입합니다. 이러한 메트릭을 기반으로 Attention-Based Attack (ABA)와 Attention-Based Defense (ABD) 전략이 마련됩니다.

- **Performance Highlights**: 실험 결과, ABA는 LLM의 주의 분포를 효과적으로 전환시켜 더 위험한 컨텐츠를 유도할 수 있으며, ABD는 입력의 위험 수준을 평가하고 LLM의 방어력을 크게 향상시킬 수 있음을 보여줍니다.



### An evaluation of LLM code generation capabilities through graded exercises (https://arxiv.org/abs/2410.16292)
- **What's New**: 본 논문은 현재 사용 가능한 평가 방법들을 검토하고, 최신 모델인 GPT4-o-mini의 성능을 Codewars와 같은 소프트웨어 개발 커뮤니티에서 확보한 8개 프로그래밍 언어의 코딩 도전 과제를 해결하는 데 평가한 결과를 제시합니다.

- **Technical Details**: 대규모 언어 모델(LLMs)은 Transformer 블록으로 구축된 심층 신경망으로, 작성된 텍스트를 단어 조각(token)으로 나누어 모델의 분포를 표현합니다. 이들은 비지도 학습 방식으로 인터넷에서 수집된 방대한 양의 텍스트를 사용하여 훈련됩니다. 평가 방법으로는 다수의 선택형 테스트, 군중 소싱 human 평가, 자동 평가 메트릭 등이 있습니다.

- **Performance Highlights**: 모델의 성공 가능성은 작업의 난이도, 사용되는 프로그래밍 언어의 인기, 과제가 게시된 시점으로부터의 경과 시간과 양의 상관관계를 보입니다. 연구 결과는 LLMs의 실제 능력을 과대 평가할 수 있음을 시사합니다.



### Assessing the Performance of Human-Capable LLMs -- Are LLMs Coming for Your Job? (https://arxiv.org/abs/2410.16285)
- **What's New**: 이 논문은 SelfScore라는 새로운 벤치마크를 개발하고 검증한 것으로, 이는 자동화된 Large Language Model (LLM) 에이전트의 헬프 데스크 및 전문 상담 업무 성능을 평가하기 위해 설계되었습니다. SelfScore는 인공지능(AI)의 활용이 증가하는 산업에서 자동화된 에이전트와 인간 근로자를 비교할 수 있는 중요한 도구입니다.

- **Technical Details**: SelfScore는 문제 복잡성과 응답 유용성을 기반으로 에이전트를 평가하며, 투명성과 단순성을 보장하는 채점 시스템을 제공합니다. 연구에서는 SelfScore를 평가하기 위해 자동화된 LLM 에이전트를 개발하였고, Retrieval-Augmented Generation (RAG) 기법의 장점을 탐구하였습니다. RAG는 관련 정보를 외부 출처에서 검색하여 이를 기반으로 새로운 텍스트를 생성하는 자연어 처리(NLP) 기법입니다.

- **Performance Highlights**: 연구 결과, 자동화된 LLM 에이전트는 인간 대조군보다 우수한 성능을 보였으며, 이는 AI 기술이 뛰어난 분야에서 인간 근로자의 대체 가능성에 대한 우려를 불러일으킵니다. SelfScore는 헬프 데스크 환경에서 AI의 영향을 이해하는 데 기초적인 도구를 제공하며, 자동화로의 전환에 따른 윤리적 고려를 옹호합니다.



### Understanding the Effect of Algorithm Transparency of Model Explanations in Text-to-SQL Semantic Parsing (https://arxiv.org/abs/2410.16283)
Comments:
          15 pages, 18 figure, Preprint

- **What's New**: 이번 연구는 AI 모델의 결정 과정을 설명하는 방법이 사용자의 경험에 미치는 영향을 탐구하며, 특히 'text-to-SQL Semantic Parsing'(텍스트-모든-쿼리 변환)라는 복잡한 예측 작업에 초점을 맞추었습니다. 세 가지 수준의 모델 설명 방식(저투명도, 중간투명도, 고투명도)을 도입하여 사용자의 AI에 대한 신뢰도와 예측 정확성을 어떻게 변화시키는지 살펴보았습니다.

- **Technical Details**: 해당 연구에서는 약 100명의 참가자를 대상으로 세 가지 알고리즘 투명도 수준(저, 중간, 고)에 따라 설명 접근 방식을 평가했습니다. 참가자는 컴퓨터 과학이나 SQL 프로그래밍을 학습하지 않은 비전문가로서, 주어진 질문을 SQL 쿼리로 변환하는 작업을 수행했습니다. 연구에서 사용된 주요 평가 메트릭은 Propensity to trust와 Jian scale의 신뢰 척도입니다.

- **Performance Highlights**: 결과적으로, (1) 저투명도와 고투명도 설명이 사용자의 결정 의존도를 낮추거나 높이는 경향이 있는 반면, 중간투명도 설명이 적절한 균형을 이루었습니다. (2) 중간투명도 그룹은 시간이 지남에 따라 성과가 증가하는 반면, 다른 그룹은 오히려 감소하는 경향을 보였습니다. (3) 모든 참가자가 연구 후 신뢰도 감소를 보였지만, 중간투명도 설명을 받은 그룹은 신뢰의 변동이 가장 적었습니다.



### GenAI Assisting Medical Training (https://arxiv.org/abs/2410.16164)
Comments:
          2 pages, 2 figures

- **What's New**: 이번 연구는 간호 교육에서 venipuncture (정맥 채혈)와 cannulation (카테터 삽입)와 같은 중요한 의료 절차를 배울 수 있도록 generative AI (생성적 인공지능) 방법을 통합하여 실시간 피드백 시스템을 제공하는 것을 목표로 하고 있습니다. 이는 교육자의 업무 부담을 줄이는 데 도움이 됩니다.

- **Technical Details**: 참여자의 시연을 통해 수집된 데이터는 static cameras (정적 카메라), GoPro camera (고프로 카메라), IMU (Inertial Measurement Unit) 데이터 등 여러 요소로 구성되어 있습니다. 특히, 각 절차의 세부 단계와 관련된 비디오 분류 방법 및 피드백 제공 방법을 개발 중입니다. 이 과정에서는 Large Language Model (LLM)도 통합되어 사용됩니다.

- **Performance Highlights**: 현재 연구팀은 수집된 데이터를 기반으로 비디오 분류 모델을 조정하고 있으며, 이는 의료 절차의 정확한 피드백을 제공할 수 있게 될 것입니다. 향후 smartwatch data (스마트워치 데이터)를 추가하여 각 절차의 수행 미세한 요소도 분석할 계획입니다.



### Learning Machines: In Search of a Concept Oriented Languag (https://arxiv.org/abs/2409.01968)
Comments:
          17 pages, 8 figures

- **What's New**: 이번 논문에서는 데이터 및 디지털 혁명 이후의 다음 단계에 대한 탐구와 '지능적(intelligent)' 기계의 정의를 논의합니다.

- **Technical Details**: 저자들은 지식 발견(knowledge discovery), 의사 결정(decision-making), 개념(concepts) 처리 능력을 갖춘 기계의 필요성을 강조하며, 역사적 기여를 고려하여 인간 지능에 대한 비유를 통해 질문을 탐구합니다. 또한 개념 지향 언어(concept oriented language)를 위한 일반적인 프레임워크를 제안합니다.

- **Performance Highlights**: 이 논문은 다음 세대의 지능적 기계가 갖춰야 할 능력에 대한 통찰을 제공하며, 인간 지능과의 유사성을 통해 기계의 발전 방향을 제시합니다.



### CompassJudger-1: All-in-one Judge Model Helps Model Evaluation and Evolution (https://arxiv.org/abs/2410.16256)
Comments:
          Technical Report, Code and Models: this https URL

- **What's New**: 이번 논문에서는 첫 번째 오픈 소스 \textbf{all-in-one} judge LLM인 \textbf{CompassJudger-1}을 소개합니다. 이 모델은 다양한 평가 작업을 수행할 수 있는 고급 기능을 갖추고 있으며, 주관적 평가의 효율성과 정확성을 향상시키는 데 중점을 둡니다.

- **Technical Details**: CompassJudger-1은 단일 스코어링, 두 모델 비교, 다양한 형식에 따른 평가 수행, 비판 생성, 일반 LLM처럼 다양한 작업 실행 등의 기능을 갖추고 있습니다. 또한, 새로운 벤치마크인 \textbf{JudgerBench}를 통해 여러 주관적 평가 작업을 통합하여 평가 모델들의 성능을 검증합니다.

- **Performance Highlights**: CompassJudger-1은 다양한 주관적 평가 작업을 수행하는 데 효과적이며, 연구 커뮤니티에 공개되어 협업과 LLM 평가 방법론의 발전을 촉진할 수 있는 기반을 제공합니다.



### Can Knowledge Editing Really Correct Hallucinations? (https://arxiv.org/abs/2410.16251)
Comments:
          The first two authors contributed equally to this work. The main paper is 10 pages long, with 35 pages total. The code, results, dataset, and additional resources are available on the project website: this https URL

- **What's New**: 이 논문에서는 HalluEditBench라는 새로운 벤치마크를 제안하여 지식 편집(knowledge editing) 기법이 LLM의 환각(hallucinations)을 수정하는 데 얼마나 효과적인지를 평가하는 방법을 제시합니다. 이를 위해 다량의 환각 데이터를 구축하고 5가지 차원으로 지식 편집 성능을 평가했습니다.

- **Technical Details**: HalluEditBench는 Wikipedia를 기반으로 한 대규모 환각 데이터셋을 구축하고, Efficacy, Generalization, Portability, Locality, Robustness의 5개 차원에서 평가 질문-답변 쌍을 생성합니다. 지식 편집 기법의 효과를 측정하기 위해 7777개의 대표적인 지식 편집 기술을 분석했습니다.

- **Performance Highlights**: HalluEditBench를 통해 FT-M와 MEMIT의 성능이 높은 점수를 보였으나 실제 환각 수정에서는 그 효과가 부족하다는 점이 발견되었습니다. 모든 편집 기법은 Efficacy 외의 차원에서 평균적으로 불만족스러운 성능을 보였고, ICE는 Robustness에서 낮은 성능을 기록했습니다.



### Analyzing Context Contributions in LLM-based Machine Translation (https://arxiv.org/abs/2410.16246)
- **What's New**: 이 연구는 대규모 언어 모델(LLMs)의 맥락 활용(context utilization) 분석에 초점을 맞추어, 기계 번역(MT)에서의 다양한 맥락 요소의 기여를 심도 있게 조사하였습니다.

- **Technical Details**: 특히, 소스 텍스트와 몇 가지 예시(few-shot examples)의 기여를 비교하며, LLM이 번역 생성 시 어떤 부문을 더 강조하는지 파악하였습니다. 연구 결과, 원본 소스(examples) 부분은 목표(target) 부분보다 더 중요한 기여를 한다는 점과, 위치적 편향(positional bias) 등의 패턴을 발견하였습니다.

- **Performance Highlights**: 또한, 맥락 기여를 분석함으로써 비정상적인 번역 결과를 탐지할 수 있는 가능성을 보여주며, 이는 이전의 인코더-디코더(encoders-decoders) 모델들에서는 발견되지 않았던 내부 작용을 이해하는 데 중요한 기초를 제공합니다.



### ToW: Thoughts of Words Improve Reasoning in Large Language Models (https://arxiv.org/abs/2410.16235)
- **What's New**: 이번 연구에서는 next-word prediction을 위한 새로운 데이터 증강 기법인 Thoughts of Words (ToW)를 소개합니다. ToW는 모델이 다음 단어가 이전 맥락과 어떻게 관련되는지를 이해하도록 돕는 정교한 생각을 주입합니다. 연구는 기존의 접근 방식들이 갖는 정보 왜곡 문제와 비효율성을 해결하고자 하는데 중점을 두고 있습니다.

- **Technical Details**: ToW는 단어 관점에서 정교한 생각들을 제공하며, 이를 이용해 단어를 네 가지 카테고리로 분류합니다: 1) 사소한 단어 (trivial); 2) 정확히 예측 가능한 단어 (exact match); 3) 대략적으로 예측 가능한 단어 (soft consistent); 4) 예측 불가능한 단어 (unpredictable). 이로써 모델이 다음 단어와 맥락 간의 연관성을 더욱 명확히 이해하도록 훈련합니다. ToW 데이터 수집 방법으로는 대형 언어 모델로부터의 증류(distillation)를 통해 70K의 고품질 ToW 주석을 생성하였습니다.

- **Performance Highlights**: ToW를 통한 지속적인 사전 훈련 후, 모델의 추론 성능이 평균 7%에서 9% 향상되었고, 모델의 허위 생성(hallucination) 비율은 최대 10% 감소했습니다. 실험 결과, ToW가 기존의 next-word prediction 훈련 방식에 비해 뛰어난 성능을 보임을 입증하였고, 이는 모델이 일반화 가능성을 높이는 데 기여하는 것을 나타냅니다.



### Sketch2Code: Evaluating Vision-Language Models for Interactive Web Design Prototyping (https://arxiv.org/abs/2410.16232)
Comments:
          preprint, 9 pages

- **What's New**: Sketch2Code라는 새로운 벤치마크를 소개하여 VLMs(비전 언어 모델)가 저해상도 스케치를 웹 페이지 프로토타입으로 변환하는 작업을 자동화할 수 있는 성능을 평가합니다. 특히 이는 UI 디자인의 초기 단계에서 스케치를 사용하여 생성된 프로토타입의 변환을 지원합니다.

- **Technical Details**: Sketch2Code는 VLM의 능력을 평가하기 위해 731개의 고품질 스케치를 수집하고, 다수의 상업적 및 오픈 소스 모델에 대해 실험을 진행했습니다. 이 연구는 VLM 모델이 스케치를 해석하고 생성하는 과정에서의 상호작용 및 사용자의 피드백 수용 능력을 평가하며, '피드백 따르기'와 '질문하기'라는 두 가지 상호작용 시나리오를 설계했습니다.

- **Performance Highlights**: 본 연구에서는 10개 모델(GPT-4o, Gemini 1.5 Pro 등)의 성능을 분석했으며, 기존 VLM 모델들이 스케치를 해석하고 질문을 생성하는 데 어려움을 겪고 있음을 보여줍니다. 사용자 연구 결과, UI/UX 전문가들은 수동적 피드백 수용보다 능동적 질문하기를 선호하며, 이는 다중 턴 대화형 에이전트의 효과적인 발전을 위해 더 깊은 연구의 필요성을 강조합니다.



### Building A Coding Assistant via the Retrieval-Augmented Language Mod (https://arxiv.org/abs/2410.16229)
- **What's New**: 본 논문에서는 COde AssistaNt viA retrieval-augmeNted language model (CONAN)을 제안합니다. 이 모델은 코딩 과정에서 인간의 지식 탐색 행동을 모방하여 코드 어시스턴트를 구축하는 것을 목표로 합니다.

- **Technical Details**: CONAN은 코드 구조 인식 리트리버(CONAN-R)와 이중 뷰 코드 표현기반 리트리벌 증강 생성 모델(CONAN-G)으로 구성됩니다. CONAN-R은 Code-Documentation Alignment(CDA)와 Masked Entity Prediction(MEP) 작업을 통해 사전 훈련되어 언어 모델이 코드 구조를 인식하고 코드 스니펫 및 문서에 대한 효과적인 표현을 학습하도록 합니다. CONAN-G는 모듈화된 여러 코드 스니펫을 생성하는 구조로, 코드 문서 설명을 프롬프트로 사용하여 코드 의미론을 더 잘 이해하게 합니다.

- **Performance Highlights**: CONAN은 다양한 코드 생성 작업에서 뛰어난 성능을 보여주며, 이전의 리트리벌 증강 코드 생성 모델보다 현저히 높은 성능을 기록했습니다. 실험 결과 CONAN-R은 코드 리트리벌 작업에서 최신 기술의 성과를 달성하였고, CONAN-G는 여러 코드 스니펫을 활용하여 모든 코드 관련 작업에서 일관된 개선을 달성했습니다.



### On Creating an English-Thai Code-switched Machine Translation in Medical Domain (https://arxiv.org/abs/2410.16221)
- **What's New**: 이 연구는 의학 분야에서의 기계 번역(MT)의 중요성과 함께 영어-태국어 MT에서 의료 용어의 정확한 번역이 왜 중요한지를 강조합니다. 기존의 MT 방식이 의학 분야에 적합하지 않은 이유를 제시하며, 의료 전문용어를 유지하는 코드 스위칭(CS) 번역 방법론을 개발하였습니다.

- **Technical Details**: 의료 도메인에서 영어-태국어 CS 번역을 위한 새로운 데이터셋을 생성하고, NLLB 기반 모델을 미세 조정하였습니다. 모델의 성능은 구글 신경 기계 번역(Google NMT) 및 GPT-3.5/GPT-4와 같은 비교 모델들과 평가하였습니다. 연구팀은 52개의 모델을 평가하고, 의료 전문가에게 직접 평가 받았습니다.

- **Performance Highlights**: 모델은 자동화된 메트릭에서 경쟁력 있는 성능을 보여주었고, 인간 평가에서도 높은 선호도를 받았습니다. 연구 결과, 의료 전문가들은 비록 유창성을 약간 저해하더라도 주요 영어 용어를 정확히 유지하는 CS 번역을 선호한다는 것을 발견하였습니다. 이는 전통적인 MT 메트릭이 의료 도메인 번역을 평가하는 데 한계가 있음을 시사합니다.



### Pre-training Distillation for Large Language Models: A Design Space Exploration (https://arxiv.org/abs/2410.16215)
- **What's New**: 이 논문에서는 기존의 post-training distillation(후훈련 지식 증류)에서 벗어나, pre-training distillation(전훈련 지식 증류)이라는 새로운 방법을 제안하여 대형 언어 모델(LLM)에서의 지식 증류를 탐구합니다.

- **Technical Details**: Pre-training distillation(PD)은 teacher model로부터 생성된 logits를 활용하여 student model을 학습하는 방법으로, 다양한 설정을 탐색했습니다. 주요 요소는 logits 처리, 손실 선택, 크기 법칙, 오프라인 또는 온라인 logits 처리입니다.

- **Performance Highlights**: GLM-4-9B를 teacher LLM으로 사용하여 1000 억 개의 토큰에서 1.9B student LLM을 증류한 결과, 평균적으로 1.6%의 성능 향상이 있었습니다. 추가 실험을 통해 PD는 대규모 student LLM에서 더욱 유리한 성과를 보임을 확인했습니다.



### Information for Conversation Generation: Proposals Utilising Knowledge Graphs (https://arxiv.org/abs/2410.16196)
Comments:
          7 pages with citations, 1 figure, accepted to the ISWC 2024 Special Session

- **What's New**: 이 논문은 대화 생성에 있어 대규모 언어 모델(LLM)을 향상시키기 위한 지식 그래프(KG)를 활용하는 세 가지 제안을 소개합니다.

- **Technical Details**: 첫 번째 제안은 동적 지식 그래프 임베딩(Dynamic Knowledge Graph Embeddings, DKGE)과 추천 시스템을 통해 새로운 정보를 통합하고 관련 지식을 선택하는 방법을 제안합니다. 두 번째로, 감정 값이 부여된 엔티티를 추가적인 특징으로 저장함으로써 사용자 입력과より 감정적으로 연관된 지식을 제공할 수 있습니다. 세 번째 제안은 내러티브 버블을 통해 캐릭터 정보를 통합하여 캐릭터 일관성을 유지하고 새로운 정보를 쉽게 통합할 수 있는 구조를 제시합니다.

- **Performance Highlights**: 이 연구는 대화형 AI의 사용자 경험을 향상시키고 LLM의 감정적 역량을 높이며 캐릭터 일관성을 통해 사용자 수용성을 증가시킬 것이라 기대합니다.



### Contamination Report for Multilingual Benchmarks (https://arxiv.org/abs/2410.16186)
Comments:
          11 pages, 2 tables

- **What's New**: 이번 연구는 대형 언어 모델(Large Language Models, LLM)의 다국어 벤치마크에 대한 오염 현상(data contamination)을 조사합니다. 특정 모델이 어느 벤치마크에 의해 오염되었는지를 분석하기 위해 오렌 외(2025)의 오염 탐지 기술을 활용합니다.

- **Technical Details**: 7개의 인기 오픈 및 폐쇄된 LLM에서 7개 다국어 벤치마크에 대한 오염 여부를 분석하기 위해 Black Box 테스트 방법론을 사용하였습니다. 이 방법은 벤치마크 데이터셋에 노출된 모델이 원래 예시의 순서에 편향을 가질 가능성을 통계적으로 검증합니다.

- **Performance Highlights**: 분석 결과, 새 버전 LLM이 벤치마크 데이터셋을 포함할 가능성이 더 높아졌음을 보여주며, 이는 다국어 데이터셋의 오염 탐지가 필요함을 의미합니다. 연구진은 향후 더 많은 데이터셋과 모델을 평가하여 오염을 확대 분석할 계획입니다.



### RM-Bench: Benchmarking Reward Models of Language Models with Subtlety and Sty (https://arxiv.org/abs/2410.16184)
- **What's New**: 새로운 벤치마크 RM-Bench를 도입하여 보상 모델의 미세한 내용 변화에 대한 민감도와 스타일 편향 저항성을 평가합니다. 기존의 보상 모델 벤치마크는 실제 모델 성능과의 상관관계가 낮았습니다.

- **Technical Details**: RM-Bench는 두 가지 주요 방법론을 사용하여 보상 모델을 평가합니다. 1) 미세한 변경에 대한 민감도를 평가하기 위해, 같은 언어 모델(gpt-4o)을 사용하여 선택된 응답과 거부된 응답을 생성합니다. 2) 스타일 편향의 저항성을 평가하기 위해 스타일 제어 프롬프트를 사용하여 다양한 스타일의 응답 변형을 생성합니다.

- **Performance Highlights**: 40개 이상의 다양한 보상 모델을 RM-Bench에서 평가한 결과, 최신 모델조차도 평균 46.6%의 성능을 보였으며, 이는 스타일 편향 간섭 하에서 무작위 추측 정확도(50%)에도 미치지 못합니다. 현재 보상 모델의 성능에는 상당한 개선이 필요합니다.



### MagicPIG: LSH Sampling for Efficient LLM Generation (https://arxiv.org/abs/2410.16179)
- **What's New**: 이 논문은 MagicPIG라는 새로운 시스템을 제안하여 대규모 언어 모델(Large Language Models, LLM)에서 효율적인 주의(attention) 계산을 가능하게 합니다. 기존의 TopK 주의 방식의 한계를 극복하기 위해 sampling 기법을 활용하여 보다 정확한 주의 결과를 제공합니다.

- **Technical Details**: MagicPIG는 Locality Sensitive Hashing (LSH) 샘플링을 기반으로 하여 LLM 생성에서 주의 계산의 부담을 줄입니다. 이 시스템은 CPU에서 주의 계산을 수행하면서 LSH 해시 테이블을 저장하여 더 긴 문맥과 큰 배치 크기를 처리할 수 있게 합니다. 주의 점검 방식에서의 이론적 보장을 통해 주의 분포를 더 정확하게 추정합니다.

- **Performance Highlights**: MagicPIG는 다양한 GPU 하드웨어에서 1.9배에서 3.9배의 처리량을 향상시키며, RTX 4090에서 Llama-3.1-8B-Instruct 모델을 96K 토큰의 문맥으로 사용하는 경우 110ms의 디코딩 지연(latency)을 달성합니다.



### Exploring Pretraining via Active Forgetting for Improving Cross Lingual Transfer for Decoder Language Models (https://arxiv.org/abs/2410.16168)
Comments:
          12 pages, 11 tables, 12 figures

- **What's New**: 이 논문에서는 Active Forgetting을 활용한 새로운 사전 훈련 전략을 제안하여, decoder-only LLM(대화형 언어 모델)에서의 교차 언어 전이를 개선하고자 하였습니다. 기존의 encoder-only 모델에 비해 decoder-only 모델의 교차 언어 전이에 대한 연구가 부족한 중, 본 연구는 이러한 한계를 극복하는 방향으로 진행되었습니다.

- **Technical Details**: 이 연구에서 Active Forgetting은 모델의 토큰 임베딩을 매 k 스텝마다 무작위로 초기화하는 방식으로 사용됩니다. 이를 통해 decoder-only LLM들은 새로운 언어에 적응할 때 성능 저하 없이 여러 언어의 표현 능력을 향상시킬 수 있음을 보였습니다. 특히, 훈련 후의 모델은 perplexity와 isotropy를 개선하여 더 중요한 멀티링구얼 표현 방식으로 이어집니다.

- **Performance Highlights**: 실험 결과, Active Forgetting으로 사전 훈련된 LLM들은 7개의 멀티링구얼 벤치마크 중 6개에서 기존 모델을 초월하는 성능을 보여주었습니다. 이는 모델이 새로운 언어를 학습할 때, 다른 언어의 성능 저하를 최소화하며, 더 나은 멀티링구얼 표현을 이루어낼 수 있음을 나타냅니다.



### From Tokens to Materials: Leveraging Language Models for Scientific Discovery (https://arxiv.org/abs/2410.16165)
- **What's New**: 이 연구에서 우리는 언어 모델 임베딩을 활용하여 재료 과학 분야에서의 물성 예측을 향상시키는 방법을 조사하였습니다. 특히, MatBERT라는 도메인 특수 모델이 일반 목적의 모델에 비해 화합물 이름과 물성에서의 암묵적 지식을 추출하는 데 뛰어난 성능을 보여주었습니다.

- **Technical Details**: Bidirectional Encoder Representations from Transformers (BERT)와 Generative Pre-trained Transformers (GPT)와 같은 여러 선행 학습 모델을 사용했습니다. MatBERT의 세 번째 층에서 나온 정보 밀집 임베딩을 컨텍스트 평균화 접근법과 결합하여 물성 관계를 추출하는 데 가장 효과적인 방법임을 입증했습니다. 또한, 'tokenizer 효과'를 분석하여 완전한 화합물 이름을 보존하면서 일관된 토큰 수를 유지하는 특수 텍스트 처리 기술의 중요성을 강조했습니다.

- **Performance Highlights**: SentMatBERT_MNR 모델을 통해 실험 결과와의 상관계수가 0.5919에 달하며, Word2Vec 방법보다 7포인트, 밀도 함수 이론 (DFT) 기준보다 28포인트 높은 예측 성능을 나타냈습니다. 이 방법은 방대한 과학 문헌에서 정보를 신속하게 처리하고 문맥을 부여할 수 있어 문헌 리뷰와 가설 생성을 위한 시간을 단축시킬 수 있습니다.



### Limpeh ga li gong: Challenges in Singlish Annotations (https://arxiv.org/abs/2410.16156)
- **What's New**: 이 연구에서는 싱글리시(Singlish)의 구문 분석을 위한 Parts-Of-Speech (POS) 태깅 작업을 다룹니다. 연구자들은 원주율 언어로 구성된 데이터셋을 구축하고, 이를 통해 기존 자동 태거의 한계를 분석하였습니다.

- **Technical Details**: 싱글리시 데이터셋은 55,000개의 SMS 텍스트 인스턴스 중 92개의 문장을 무작위로 샘플링하여 생성하였으며, 이 데이터는 원어민 싱글리시 화자들에 의해 번역 및 태깅되었습니다. 자동 POS 태깅에는 spaCy의 pre-trained 모델을 사용했습니다 (en_core_web_sm와 en_core_web_trf).

- **Performance Highlights**: 자동 POS 태거의 정확도는 평균 80%로, 영어 문장 태깅의 경우 평균 97%의 성과와 비교해 미비한 결과를 보였습니다. 이러한 결과는 싱글리시의 독특한 문법 구조로 인해 발생하는 것으로 나타났습니다.



### A Troublemaker with Contagious Jailbreak Makes Chaos in Honest Towns (https://arxiv.org/abs/2410.16155)
- **What's New**: 이 논문에서는 대형 언어 모델의 발전과 함께 다중 에이전트 및 다중 토폴로지를 고려한 공격 평가 프레임워크인 TMCHT(트러블메이커가 정직한 마을에서 혼란을 일으킨다)를 제안합니다. 이는 독립적인 메모리를 가진 에이전트들 간의 상호작용을 평가하는 새로운 접근 방식입니다.

- **Technical Details**: TMCHT는 단일 공격 에이전트가 여러 클린 에이전트로 구성된 사회를 오도하려고 시도하는 다중 에이전트, 다중 토폴로지의 텍스트 기반 공격 평가 작업입니다. 이 연구는 두 가지 주요 도전과제를 식별합니다: 비완전 그래프 구조와 대규모 시스템. 또한, 우리는 ARCJ(Adversarial Replication Contagious Jailbreak) 방법을 제안하여, 독성이 사라지는 현상(음성의 전파 부족)을 해결하게 합니다.

- **Performance Highlights**: TMCHT에서 우리의 접근법은 선형 토폴로지에서 44.20%, 별형 토폴로지에서 38.94%, 100 에이전트 설정에서 85.18%의 공격 성공률(ASR)을 달성했습니다. 이는 기존 방법에 비해 각각 23.51%, 18.95%, 52.93% 향상된 성과입니다.



### Pangea: A Fully Open Multilingual Multimodal LLM for 39 Languages (https://arxiv.org/abs/2410.16153)
Comments:
          52 pages, 27 figures

- **What's New**: Pangea는 다양한 39개 언어로 구성된 6백만 개의 지침 데이터셋 PangeaIns를 기반으로 한 다국어 다모드 대형 언어 모델(MLLM)입니다. 기존의 영어 중심 데이터셋과는 달리, 문화적 맥락과 다양한 언어를 포함하여 지식의 불균형 문제를 해결합니다.

- **Technical Details**: Pangea는 PangeaIns라는 데이터셋을 통해 다국어 및 다모드 훈련을 실시하며, PangeaBench라는 14개의 데이터셋으로 구성된 종합 평가 도구를 활용하여 모델의 성능을 정량적으로 측정합니다. xChat과 xMMMU 벤치마크를 통해 개방형 대화 및 다모드 사고 작업에 대한 평가를 진행합니다.

- **Performance Highlights**: Pangea 모델은 PangeaBench 데이터셋에서 기존의 오픈 소스 MLLM보다 평균 7.3 포인트(영어 작업) 및 10.8 포인트(다국어 작업) 더 뛰어난 성능을 보였습니다. 또한 Gemini-1.5-Pro 및 GPT4o와 같은 상용 LLM을 여러 작업에서 초과하며, 다국적 및 문화적 이해에서 우수한 성능을 나타냈습니다.



### 1-bit AI Infra: Part 1.1, Fast and Lossless BitNet b1.58 Inference on CPUs (https://arxiv.org/abs/2410.16144)
- **What's New**: 최근 1-bit Large Language Models (LLMs)인 BitNet과 BitNet b1.58의 발전은 LLM의 속도와 에너지 소비 면에서 효율성을 향상시키는 유망한 접근 방식을 제공하고 있습니다. 이 논문에서는 1-bit LLM의 전체 잠재력을 열기 위해 설계된 맞춤형 소프트웨어 스택인 bitnet.cpp를 소개합니다.

- **Technical Details**: bitnet.cpp는 1-bit LLM(예: BitNet b1.58 모델)의 추론 프레임워크입니다. 이 프레임워크는 손실 없는 추론을 제공하면서 속도와 에너지 소비를 최적화합니다. 1.58-bit 모델에 대한 최적화된 커널도 포함되어 있으며, ARM 및 x86 아키텍처에서 모두 빠르고 손실 없는 추론을 지원합니다.

- **Performance Highlights**: bitnet.cpp는 x86 및 ARM 아키텍처에서의 성능 테스트 결과, 속도가 1.37배에서 6.46배까지 향상되고 에너지 소비는 55.4%에서 82.2%까지 절감되었습니다. 특히, ARM에서는 Apple M2에서 최적 스레드 상황에서 5.07배, Intel i7-13700H에서는 6.46배의 속도 개선을 보여주어 로컬 디바이스에서의 LLM 운영 가능성을 크게 향상시킵니다.



### A Psycholinguistic Evaluation of Language Models' Sensitivity to Argument Roles (https://arxiv.org/abs/2410.16139)
- **What's New**: 본 연구는 대형 언어 모델의 인과 관계에 대한 민감도를 체계적으로 평가하며, 이는 인간의 인과 관계 처리에 대한 심리언어학적 연구를 재현하는 데 중점을 둡니다.

- **Technical Details**: 세 가지 실험을 통해 언어 모델이 플로러의 역할과 플로러가 올바른 문맥에서 그 역할을 구별할 수 있는 능력을 조사했습니다. 특히 언어 모델이 동사를 평가하는 방식과 인간의 실시간 동사 예측 방식의 차이를 밝혀냈습니다.

- **Performance Highlights**: 언어 모델은 플로러 역할 정보에 대한 민감도가 약하며, 이는 인간의 초기 예측 행동과 유사한 패턴을 보였습니다. 그러나 모델은 다양한 유형의 루프 역할 조작에 대해 인간과 같은 일관성을 보이지 않았으며, 이 결과는 모델과 인간 간의 역할 처리 방식의 차이를 나타냅니다.



### Do LLMs write like humans? Variation in grammatical and rhetorical styles (https://arxiv.org/abs/2410.16107)
Comments:
          29 pages, 4 figures, 11 tables

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)의 수사적 스타일을 분석하여 인간 텍스트와 구별되는 체계적인 언어적 차이를 발견했습니다. 기존의 연구가 주로 기본 문법이나 어휘력에 초점을 맞췄다면, 이 연구는 LLM과 인간 간의 문체적 차이를 심층적으로 분석하였습니다.

- **Technical Details**: 우리는 OpenAI의 GPT-4o와 Meta의 Llama 3의 다양한 변형을 사용하여 두 개의 병렬 말뭉치(corpora)를 만들어 LLM과 인간이 작성한 텍스트의 스타일을 비교했습니다. Douglas Biber의 언어적, 문법적, 수사적 특징 집합을 활용하여 LLM과 인간 간의 차이를 분석했습니다. 연구 결과, 지시 조정된(instruction-tuned) LLM이 기본 모델보다 더 극단적인 문법적 차이를 보이며, 이를 통해 LLM의 특정 문법 구조 선호와 인간 문체의 복잡성을 모방하는 데 한계가 있음을 확인했습니다.

- **Performance Highlights**: LLMs는 인간 아이디어의 구체적인 표현 방식에서 크게 벗어나 있으며, 문법적 구조에 있어 특정 패턴을 선호함을 발견했습니다. 특히, 지시 조정된 LLM들이 인간 텍스트에 비해 특정 문법적 구조(예: 현재 분사절, 명사화 등)를 2배에서 5배 높은 비율로 사용하여 그들의 작성 스타일을 더욱 구분할 수 있음을 나타냅니다.



### Analysing the Residual Stream of Language Models Under Knowledge Conflicts (https://arxiv.org/abs/2410.16090)
Comments:
          Foundation Model Interventions Workshop @ NeurIPS 2024

- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)이 지식 갈등(knowledge conflict)을 내부적으로 인식하는 방법과 어떤 지식 출처에 의존할지를 파악하기 위해 잔여 흐름(residual stream)을 분석하는 방법을 제안합니다.

- **Technical Details**: 이 연구에서는 Transformer 아키텍처의 잔여 흐름을 분석하여 LLM의 내부 메커니즘을 이해하고, 지식 갈등을 탐지할 수 있는 가능성을 논의합니다. 중간 레이어에서의 활성화(probing tasks)를 통해 지식 갈등의 신호를 효과적으로 검출할 수 있으며, 로지스틱 회귀 모델을 사용하여 90% 정확도로 갈등을 탐지할 수 있음을 보였습니다.

- **Performance Highlights**: 달성된 성능의 하이라이트는 LLM의 잔여 흐름이 지식 갈등을 감지하고 해결하는 과정에서 서로 다른 패턴을 보인다는 점입니다. 특히, 문맥적 지식을 사용할 때 잔여 흐름의 분포가 더 왜곡된(skewed) 형태를 나타내며, 이는 모델의 행동을 예상할 수 있는 인사이트를 제공합니다.



### Fine-Tuning LLMs for Reliable Medical Question-Answering Services (https://arxiv.org/abs/2410.16088)
Comments:
          8 pages, 10 figures, accepted and to be published in the proceedings of 2024 IEEE International Conference on Data Mining Workshops (ICDMW)

- **What's New**: 최신 의료 QA 서비스 개선을 위한 LLM(대형 언어 모델)의 세밀하게 조정된 활용을 제안. 특히 LLaMA-2와 Mistral과 같은 모델을 활용하여 의료 정보를 보다 정확하고 신뢰성 있게 제공.

- **Technical Details**: rsDoRA+ 및 ReRAG와 같은 고급 세밀 조정 기법을 통해 LLM의 성능을 향상시키는 방식으로 접근. rsDoRA+는 분해된 모델 가중치와 간섭을 방지하는 학습률 변화를 통해 안정적인 학습을 돕고, ReRAG는 검색 및 질문 재작성을 통합하여 응답의 정확성을 높임.

- **Performance Highlights**: 본 연구는 환자 신뢰도를 높이고 정보 접근성을 개선하여, 의료 제공자가 신속하고 신뢰할 수 있는 정보를 통해 보다 효율적인 의사 결정을 내릴 수 있도록 지원하는 것을 목표로 함.



### Rolling the DICE on Idiomaticity: How LLMs Fail to Grasp Contex (https://arxiv.org/abs/2410.16069)
- **What's New**: 본 논문은 LLMs(대규모 언어 모델)이 관용구(idiom)의 의미를 맥락(context)을 통해 해석하는 능력을 평가하기 위해 설계된 새로운 벤치마크 데이터셋을 제안합니다. 이는 모델이 맥락에 의존하여 비유적 의미와 문자적 의미를 구별할 수 있는지를 분석하는 데 중점을 둡니다.

- **Technical Details**: 이 논문은 LLMs의 idiomaticity(관용적 표현의 해석)와 관련된 데이터셋 DICE를 소개합니다. DICE에서는 문장 내에서의 맥락을 정확히 분석하기 위해 표현 형태를 일관되게 유지하며, 단어의 구성에 따른 변형이 없도록 설계되었습니다. 데이터셋에는 수천 개의 'potentially idiomatic expressions'(PIEs)가 포함되어 있으며, 이를 통해 모델의 성능을 정밀하게 평가합니다.

- **Performance Highlights**: 실험 결과, LLMs는 맥락을 고려할 필요가 있을 때 관용구의 의미를 해석하는 데 어려움을 겪으며, 맥락의 가능성이 높은 문장에서는 성능이 더 좋다는 것을 발견했습니다. 또한 표현의 동시 발생 빈도(collocational frequency)가 성능에 유의미한 영향을 미친다는 점도 밝혀졌습니다.



### Surprise! Uniform Information Density Isn't the Whole Story: Predicting Surprisal Contours in Long-form Discours (https://arxiv.org/abs/2410.16062)
Comments:
          EMNLP 2024 (main conference)

- **What's New**: 이번 논문에서는 Uniform Information Density (UID) 가설을 넘어, 서사가 계층적으로 구성된 모델 내에서 정보 비율(information rate)이 어떻게 조절되는지를 제안합니다. 이 가설을 Structured Context Hypothesis라 명명하였습니다.

- **Technical Details**: Structured Context Hypothesis는 정보의 분포를 서사 구조에 따라 조절한다고 주장합니다. 실험은 대형 언어 모델에서 추출한 자연 발생 서사를 분석하여, 계층적 예측기(hierarchical predictors)를 이용하여 정보 곡선(surprisal contours)을 예측합니다.

- **Performance Highlights**: 계층적 예측기는 서사의 정보 곡선에 대해 중요한 예측 요소로 작용하며, 깊게 중첩된 계층적 예측기가 얕은 예측기보다 더 나은 예측력을 보였습니다. 이는 정보 비율의 변동이 예측 가능한 방식으로 발생함을 보여줍니다.



### Large Language Models Know What To Say But Not When To Speak (https://arxiv.org/abs/2410.16044)
Comments:
          EMNLP 2024 (Findings)

- **What's New**: 이번 연구는 자연 대화에서의 전환 관련 장소(Transition Relevance Places, TRPs)를 예측하는 데 있어 최신 대형 언어 모델(Large Language Models, LLMs)의 성능을 평가하는 데 필요한 새로운 데이터셋을 소개합니다. 새로운 데이터셋은 참가자가 라벨링한 블라인 회전 TRPs를 포함하여, 자연스러운 대화 속에서 사용자 반응을 기반으로 합니다.

- **Technical Details**: 이 연구에서 제안된 데이터셋은 인간 반응을 기반으로 하여, LLMs가 자연스러운 대화 상황에서의 TRP 예측 능력을 평가할 수 있도록 설계되었습니다. 기존 LLMs (예: TurnGPT, RC-TurnGPT)는 주로 일상적인 회화에서 TRPs를 구분하기 어려운 한계가 있습니다. TRP는 말하는 이의 발화에서 청취자가 의사 표현을 할 수 있는 지점으로, 이를 잘 예측하는 것은 인공지능 대화 시스템의 필수 기능입니다.

- **Performance Highlights**: 실험 결과, 현재의 LLM들은 비연속적인 대화 상호작용을 모델링하는 데 한계가 있으며, 이는 장시간의 침묵 또는 부적절한 타이밍의 피드백을 초래하는 문제를 보여줍니다. 연구는 이러한 한계를 극복하고 보다 자연스러운 대화 시스템을 개발할 수 있는 기초 자료를 제공합니다.



### TreeBoN: Enhancing Inference-Time Alignment with Speculative Tree-Search and Best-of-N Sampling (https://arxiv.org/abs/2410.16033)
- **What's New**: TreeBoN은 Best-of-N(BoN) 샘플링에 투기적 트리 탐색 전략을 통합하여, 정보 효율성을 높이면서도 고품질 출력을 유지하도록 설계된 새로운 프레임워크입니다. 이 방법은 부모 노드를 유지하며 반복적으로 저품질 응답을 가지치기하여 계산 비용을 줄입니다.

- **Technical Details**: TreeBoN은 DPO(Direct Preference Optimization)에서 얻은 토큰 수준 보상을 활용하여 트리 확장을 유도하고 저품질 경로를 가지치기합니다. Monte Carlo Tree Search(MCTS) 기술을 접목하여 더 나은 디코딩 성능을 추구합니다.

- **Performance Highlights**: TreeBoN은 192 및 384 토큰에서 최대 65%의 승률을 기록하며, 동일한 계산 비용으로 표준 BoN을 초월하는 성능을 보여줍니다. 전체 데이터셋에서 60% 이상의 승률을 달성하여 확장성과 정렬 효율성을 증명하였습니다.



### ComPO: Community Preferences for Language Model Personalization (https://arxiv.org/abs/2410.16027)
- **What's New**: 본 연구에서는 기존 언어 모델(LM) 훈련 방법의 한계를 지적하고, 사용자 개인화 및 집단적 선호를 고려한 새로운 방법론인 ComPO(Community Preference Optimization)를 제안합니다. 이 방법은 개별 사용자의 피드백을 구체적으로 반영하기보다는 커뮤니티 수준의 선호도를 활용하여 다양한 사용자 그룹의 요구를 충족하는 데 중점을 두고 있습니다.

- **Technical Details**: ComPO는 레딧과 같은 플랫폼에서 수집한 Q&A 데이터셋인 ComPRed를 활용합니다. 이 데이터셋은 특정 서브레딧의 컨텍스트에 따라 모델의 확률 분포를 조정하여, 커뮤니티의 선호에 맞춘 응답을 생성하는 데 초점을 맞춥니다. 제안된 방법은 사용자와 그들이 제공하는 선호 정보를 동시에 고려하여 언어 모델을 훈련시키는 방식입니다. 서브레딧 이름을 사용하여 모델 세부 조정 시 성능을 극대화하는 것이 검증되었습니다.

- **Performance Highlights**: 실험 결과, 커뮤니티 식별자(예: 서브레딧 이름)를 통해 선호 조정 시 모델의 응답 품질이 향상되었으며, 이는 다양한 인간 평가자 및 자동 메트릭에서도 더욱 긍정적인 평가를 받았습니다. 반면, 잘못된 서브레딧 식별자를 사용했을 경우 성능이 크게 저하되는 경향을 보였으며, 이는 각 커뮤니티의 선호에 맞춘 응답 생성의 중요성을 강조합니다.



### CA*: Addressing Evaluation Pitfalls in Computation-Aware Latency for Simultaneous Speech Translation (https://arxiv.org/abs/2410.16011)
- **What's New**: 본 논문에서는 Simultaneous Speech Translation (SimulST) 시스템의 지연(latency) 측정의 한계를 탐구하고, 기존의 지연 평가 접근법에 대한 오해를 밝혀냄으로써 보다 정확한 측정 방법을 제안합니다.

- **Technical Details**: SimulST 시스템의 성능은 주로 지연 시간을 기반으로 평가됩니다. 기존의 지연 측정 지표들(Average Proportion, Average Lagging 등)은 비현실적으로 높은 지연치를 보이며, 체계적인 문제를 내포하고 있습니다. 본 연구에서는 'computation-aware latency'를 포함하는 새로운 지표를 제안하여, 계산 시간을 무시하지 않고 더욱 실질적인 성능 평가를 수행합니다.

- **Performance Highlights**:  새로운 접근法을 통해 SimulST 시스템의 지연 측정 정확도가 향상되며, 특히 비분할 스트리밍(long speech) 환경에서의 적용 가능성이 증가합니다. 이를 통해 더 나은 해석 및 강의 필사 transcription과 같은 실 세계 시나리오에서의 성능 향상이 기대됩니다.



### Exploring Continual Fine-Tuning for Enhancing Language Ability in Large Language Mod (https://arxiv.org/abs/2410.16006)
Comments:
          19 pages, 6 tables, 4 figures

- **What's New**: 이 논문은 대규모 언어 모델(LLMs)이 새로운 언어를 학습하면서 기존 언어(주로 영어)에 대한 성능을 저하시킬 위험 없이 적응할 수 있도록 지속적인 미세 조정(Continual Fine-Tuning, CFT)을 활용하는 방법을 제안합니다. 두 단계의 CFT 프로세스를 통해 다국어 데이터 세트에서 작업 수행 능력과 언어 능력을 조정하는 방식을 분석합니다.

- **Technical Details**: 첫 번째 단계(Phase 1)에서는 영어 데이터로 LLM을 미세 조정하여 작업 수행 능력(Task Ability, TA)을 향상시킵니다. 두 번째 단계(Phase 2)에서는 다국어 데이터 세트로 미세 조정하여 언어 능력(Language Ability, LA)을 향상시킵니다. 연구에서는 태스크 간 유사성을 기반으로 모델의 적응도를 결정합니다.

- **Performance Highlights**: Phase 2 이후에 LLM은 비슷한 데이터 세트에서 작업 수행 능력을 유지하거나 향상시키지만, 서로 다른 데이터 세트에서 성능이 저하됩니다. 연구는 두 가지 CFT 전략, 즉 계층 고정(Layer Freezing)과 생성 리플레이(Generative Replay)의 효율성을 평가하며, 이들 기법이 언어 능력을 향상시키면서 작업 수행 성능을 유지하는 데 효과적임을 보여줍니다.



### Steering Knowledge Selection Behaviours in LLMs via SAE-Based Representation Engineering (https://arxiv.org/abs/2410.15999)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)에서 정보를 선택하는 방식에 대한 새로운 방법론인 	extsc{SpARE}를 제안합니다. 	extsc{SpARE}는 사전 훈련된 희소 오토 인코더(Sparse Auto-Encoders, SAEs)를 활용하여 LLM의 지식 선택 동작을 제어할 수 있도록 돕습니다.

- **Technical Details**: 	extsc{SpARE}는 LLM 내부의 지식 충돌을 감지할 수 있는 신호를 중간 계층에서 분석하여, 훈련 없이도 지식 선택 행동을 조정할 수 있는 기능적 특징을 식별합니다. 이러한 방식으로 LLM의 내부 활성화를 효율적으로 편집하여 정보 충돌을 해결합니다.

- **Performance Highlights**: 실험 결과, 	extsc{SpARE}는 기존의 표현 엔지니어링 방법보다 10% 더 높은 성능을 보였으며, 대조적 디코딩 방법에 비해 15% 더 좋은 결과를 나타냈습니다. 이는 	extsc{SpARE}가 LLM의 지식 선택 동작을 효과적으로 조정함을 보여줍니다.



### 1024m at SMM4H 2024: Tasks 3, 5 & 6 -- Ensembles of Transformers and Large Language Models for Medical Text Classification (https://arxiv.org/abs/2410.15998)
Comments:
          short paper , acl 2024

- **What's New**: 이 논문은 소셜 미디어 데이터를 활용하여 건강과 관련된 여러 과제를 다루고 있습니다. 특히, 자연과 야외 공간이 정신 건강에 미치는 영향, 아동의 건강 장애를 보고하는 트윗의 이진 분류, 사용자의 나이를 자가 보고하는 텍스트의 이진 분류를 포함한 SMM4H 2024의 세 가지 과제를 조사합니다.

- **Technical Details**: 논문에서 사용된 모델은 RoBERTa, DeBERTa, Longformer와 같은 Transformer 모델 및 GPT-4, Claude-Opus, Llama-3 8B와 같은 LLM(대형 언어 모델)입니다. 각 모델의 장단점을 분석하며, 패거지를 통해 성능을 향상시키는 방법도 다룹니다. 데이터셋은 Reddit 게시물과 트윗으로 구성되어 있으며, 분류 작업을 수행하기 위해 여러 프로세스를 테스트했습니다.

- **Performance Highlights**: 모델 성능은 Task 5와 6에서 0.99 이상의 recall을 기록하였습니다. LLM 접근 방식이 4비트 정밀도로도 비교적 좋은 결과를 냈으나, 전체 정밀도를 이용할 경우 성능이 더 향상될 것으로 예상됩니다. 데이터 증강 방법은 성능 향상에 기여하지 않았지만, 데이터 증대는 성능을 개선할 수 있는 가능성을 보여줍니다.



### Augmenting Legal Decision Support Systems with LLM-based NLI for Analyzing Social Media Evidenc (https://arxiv.org/abs/2410.15990)
Comments:
          8 pages , accepted to emnlp 2024

- **What's New**: 본 논문은 2024 NLLP 공유 작업의 법적 자연어 추론(L-NLI) 과제에 대한 시스템 설명과 오류 분석을 제공합니다. 우리의 시스템은 다른 참가자들에 비해 월등한 성능을 보였으며, 법률 텍스트 분석에서의 접근 방식의 효과성을 입증했습니다.

- **Technical Details**: 이 연구에서는 Gemma, Phi3, Zephyr, LLaMA, Mistral, OpenHermes, Qwen 등 여러 LLMs를 활용하여 관계를 entailed, contradicted, neutral로 분류하는 작업을 수행했습니다. 데이터셋은 법적 전제와 온라인 미디어 텍스트로 구성되어 있으며, 초기 fine-tuning을 위해 SNLI 데이터셋의 20,000 샘플이 사용되었습니다. 다양한 alignment 접근 방식과 multi-stage learning을 통한 성능 향상 방법도 실험적으로 시도되었습니다.

- **Performance Highlights**: 우리는 Type-1 오류를 완전히 피하고 Type-2 오류로 한정하였으며, Neutral과 Entailed 간의 혼동이 가장 흔한 오류로 나타났습니다. 시스템의 성능은 대부분의 도메인에서 baseline을 초과했으며, BIPA에서는 성능이 상대적으로 약했습니다. 향후 앙상블 사용이나 더 많은 훈련 자료를 추가하여 성능을 개선할 수 있는 여지가 있습니다.



### Large Language Models for Cross-lingual Emotion Detection (https://arxiv.org/abs/2410.15974)
Comments:
          6 pages , accepted to acl 2024

- **What's New**: 이번 논문은 WASSA 2024 Task 2에 제출한 시스템에 대한 상세한 설명을 제시하며, 다양한 언어의 감정을 효과적으로 이해하고 분류하기 위해 대규모 언어 모델(LLMs)을 조합하여 활용한 방법을 소개합니다. 이 접근법은 다른 제출물보다 높은 성능을 보였으며 여러 모델을 통합해 성능을 향상시키는 데서 강점을 보였습니다.

- **Technical Details**: 감정 탐지 시스템은 GPT-4, Claude-Opus, LLAMA-3-8B, Gemma-7B, Mistral-v2-7B와 같은 여러 개방형 및 독점 LLM을 사용하여 구성되었습니다. 모델들은 각각 4-bit과 16-bit 정밀도에서 테스트되었고, 데이터셋은 네덜란드어, 영어, 프랑스어, 러시아어, 스페인어로 구성되어 있으며 6개의 감정 클래스로 주석이 달렸습니다. 시스템은 5 에포크 동안 학습률 0.0002, 가중치 감소 0.01로 비독점 LLM을 미세 조정하였습니다.

- **Performance Highlights**: 앙상블 모델은 직접적인 접근법에 비해 성능이 월등히 좋았으며, 특정 언어에 대해 더 좋은 성능을 보이는 모델이 있었습니다. 최종 결과는 가중 F1 점수를 기준으로 하며, 4-bit 정밀도에서의 성능 저하가 최소였으나 특정 케이스에서 4-bit가 16-bit보다 정확한 예측을 제공했습니다. 이러한 실험은 모델 선택과 데이터 증가를 기반으로 한 다양한 접근 방식을 검토하였습니다.



### Policy-driven Knowledge Selection and Response Generation for Document-grounded Dialogu (https://arxiv.org/abs/2410.15970)
Comments:
          29 pages, 9 figures, 14 tables, TOIS 2024

- **What's New**: 본 논문에서는 Document-grounded dialogue(DGD) 작업의 지식 선택(knowledge selection, KS)과 응답 생성(response generation, RG) 과정을 지원하기 위해 대화 정책(dialogue policy)을 제안합니다. 대화 정책은 발화 기능(utterance function)과 주제 전이 의도(topic transfer intent)라는 두 가지 신호로 구성되어 있습니다.

- **Technical Details**: 제안된 정책 기반 Document-grounded dialogue (PD-DGD) 프레임워크는 KS와 RG에 있어 대화 정책을 활용하기 위해 서로 다른 메커니즘을 적용합니다. 정책 계획자는 정책 인식 대화 표현을 활용하여 지식을 선택하고 응답의 정책을 예측하며, 생성기는 정책/지식 인식 대화 표현을 사용하여 응답을 생성합니다.

- **Performance Highlights**: PD-DGD 모델은 WoW, Holl-E, Doc2Dial의 세 개의 공개 데이터셋에서 최첨단 성능을 달성하며, 실험 결과를 자세히 분석했습니다. 이러한 결과는 대화 정책을 통해 지식을 선택하고 사용한 맥락을 해석할 수 있는 방법을 제공합니다.



### Self-Explained Keywords Empower Large Language Models for Code Generation (https://arxiv.org/abs/2410.15966)
- **What's New**: 이 논문은 코드 생성에서 저주 받는 저빈도 키워드의 이해 부족을 극복하기 위한 새로운 기법인 SEK(Self-Explained Keywords)를 제안합니다. 기존의 LLM(대형 언어 모델)들이 저빈도 키워드를 무시하거나 오해하는 문제를 해결하고, 이는 코드 생성 정확도를 높이는 데 기여하고자 합니다.

- **Technical Details**: SEK는 문제 설명에서 중요 키워드를 추출하고 이를 설명하여 기존의 문제 설명을 풍부하게 하여 LLM의 코드 생성 성능을 향상시킵니다. 이 과정은 세 가지 주요 단계로 구성됩니다: 1) KeyExtract & Explain: 문제 설명을 기반으로 키워드를 추출하고 설명합니다. 2) KeyRank: 추출된 키워드를 빈도 기반으로 정렬합니다. 3) PromptEnrich: 정렬된 키워드와 설명을 원래 문제 설명에 추가하여 풍부한 문제 설명을 만듭니다.

- **Performance Highlights**: SEK는 여러 코드 생성 벤치마크에서 LLM의 성능을 유의미하게 개선합니다. 예를 들어, SEK를 활용한 DeepSeek-Coder-V2-Instruct는 HumanEval 벤치마크에서 Pass@1 점수를 85.4%에서 93.3%로 향상시켰습니다. 또한, Llama-3.1은 평균적으로 사용된 벤치마크에서 8.8%의 상대적 개선을 달성했습니다.



### Systematic Exploration of Dialogue Summarization Approaches for Reproducibility, Comparative Assessment, and Methodological Innovations for Advancing Natural Language Processing in Abstractive Summarization (https://arxiv.org/abs/2410.15962)
- **What's New**: 이 연구는 자연어 처리(NLP) 분야에서의 대화 요약 모델의 재현성 및 평가에 대한 연구로, 원래 연구와의 불일치를 집중 분석합니다.

- **Technical Details**: 대화 요약 모델 분석은 AMI(Augmented Multi-party Interaction) 데이터셋을 사용하여 수행하였으며, 조직적 메모리 네트워크(Hierarchical Memory Networks, HMNet)와 여러 버전의 포인터 생성 네트워크(Pointer-Generator Networks, PGN)를 포함합니다: PGN(DKE), PGN(DRD), PGN(DTS), PGN(DALL).

- **Performance Highlights**: 원래 연구와의 불일치에 대한 심층 분석을 통해, 인간 평가를 통한 요약의 정보성(informativeness)과 품질(quality)을 평가하였습니다. 데이터셋 1에서의 샘플 표준 편차는 0.656으로, 데이터 포인트의 평균 주위 분산이 중간 정도임을 나타냅니다.



### Do Large Language Models Have an English Accent? Evaluating and Improving the Naturalness of Multilingual LLMs (https://arxiv.org/abs/2410.15956)
- **What's New**: 현재의 대형 언어 모델들은 주로 영어를 주요 언어로 설계되어 있으며, 다국어 모델조차도 강한 영어 중심 편향을 보입니다. 본 논문은 다국어 LLM 출력의 어휘적 및 구문적 자연스러움을 평가하기 위한 새로운 자동화된 메트릭을 소개하고, 이 메트릭을 이용하여 프랑스어와 중국어에서의 성능을 분석합니다.

- **Technical Details**: 우리는 직접 선호 최적화(Direct Preference Optimization, DPO) 방법을 사용하여 특정 언어에서 LLM의 자연스러움을 향상시키기 위한 간단하고 효과적인 접근 방식을 제안합니다. 새로운 선호 데이터셋을 통해 인간이 작성한 응답과 합성적으로 조작된 응답을 비교합니다. 이를 통해 기존 LLM의 자연스러움을 개선하는 데 있어 일관된 성과를 보입니다.

- **Performance Highlights**: 본 연구의 결과는 LLM이 중국어에서 자연스러움을 향상시킬 수 있음을 보여주며, 일반적인 벤치마크의 성능을 저해하지 않으면서도 자연스러움을 일관되게 개선할 수 있음을 확인했습니다.



### Findings of the Third Shared Task on Multilingual Coreference Resolution (https://arxiv.org/abs/2410.15949)
Comments:
          Accepted to CRAC 2024

- **What's New**: 이번 논문은 CRAC 2024 워크숍의 일환으로 개최된 다국어 공통 화주 체계 해석(Shared Task on Multilingual Coreference Resolution) 제3판에 대한 개요를 제시합니다. 전년과 마찬가지로 참가자들은 정체성 공시(Identity Coreference)에 기반하여 언급(mention)을 식별하고 클러스터링하는 시스템을 개발해야 합니다. 올해는 제로 언급(Zero Anaphora)의 골드 슬롯을 제공하지 않아 작업의 복잡성과 현실성이 증대되었습니다. 또한 역사적 언어를 중심으로 더 다양한 언어 세트가 포함되었습니다.

- **Technical Details**: 이번 작업에서는 CorefUD의 최신 버전 1.2를 사용하며, 15개 언어에 대해 총 21개 데이터셋이 포함됩니다. 새롭게 추가된 언어로는 고대 그리스어, 고대 히브리어, 고대 슬라브어가 있으며, 이는 라틴 문자 언어 외의 자원을 가진 언어에 대한 범위를 확장합니다. 또한, English LitBank의 도입으로 긴 문서가 포함된 소설 데이터셋이 추가되었습니다. 제로 언급과 관련된 변환이 폴란드어 PCC에서 크게 개선되었습니다.

- **Performance Highlights**: 올해의 공유 작업에는 총 6개의 시스템이 참가했으며, 평가 기준과 주요 및 보조 점수를 포함하여 평가 방식이 설명됩니다. 기존 자원들이 업데이트되었고, 더 많은 자원들이 다양한 언어와 함께 사용되었습니다. 이를 통해 실제 응용에 더 적합한 견고한 솔루션 개발이 목표입니다.



### CausalGraph2LLM: Evaluating LLMs for Causal Queries (https://arxiv.org/abs/2410.15939)
Comments:
          Code - this https URL

- **What's New**: 본 논문은 Large Language Models (LLMs)가 인과 관계를 이해하고 인과 그래프를 생성하는 데 있어 효과적인 능력을 측정하기 위한 새로운 기준인 CausalGraph2LLM을 제안합니다. 다양한 인과 그래프 설정을 포함하여 LLM이 인과 그래프를 인코딩할 수 있는 능력을 평가합니다.

- **Technical Details**: CausalGraph2LLM 벤치마크는 그래프 수준(query)과 노드 수준(query) 두 가지 유형의 인과 쿼리를 사용하여 LLM의 인과 그래프 이해 능력을 평가합니다. 실험에서 사용된 모델들은 오픈소스와 클로즈드 모델 모두 포함되며, 이 연구에서 LLM 모델의 인코딩에 대한 민감도를 실험적으로 보여줍니다.

- **Performance Highlights**: 연구 결과, LLM들은 인과 그래프에 대해 높은 민감성을 나타내며, 인코딩 방식에 따라 약 60%의 성능 차이가 발생했습니다. 또한 LLM 모델들은 인과 그래프에 대한 맥락적 정보에 따라 편향된 결과를 보이는 것으로 관찰되었습니다.



### Yeah, Un, Oh: Continuous and Real-time Backchannel Prediction with Fine-tuning of Voice Activity Projection (https://arxiv.org/abs/2410.15929)
- **What's New**: 이 논문은 새로운 VAP(Voice Activity Projection) 모델을 기반으로 한 실시간 연속 백채널 예측 방법을 제안합니다. 기존의 연구들은 발화 기반 또는 인위적으로 균형잡힌 데이터셋에 의존했으나, 본 연구는 불균형적인 실제 데이터셋에서 프레임 단위로 백채널의 타이밍과 형태를 연속적으로 예측합니다.

- **Technical Details**: VAP 모델은 일반 대화 말뭉치를 활용하여 사전 학습 후, 백채널 행동에 집중한 전문 데이터셋으로 파인튜닝(fine-tuning)됩니다. 이 과정은 BERT와 같은 모델에서 사용하는 사전 학습 및 파인튜닝 패러다임을 따릅니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 타이밍과 타입 예측 작업 모두에서 베이스라인 방법보다 우수한 성능을 보여주며, 실시간 환경에서 강건한 성능을 보였습니다. 이는 가상 비서 및 로봇과 같은 대화형 음성 대화 애플리케이션에 응용될 수 있는 가능성을 제시합니다.



### DefVerify: Do Hate Speech Models Reflect Their Dataset's Definition? (https://arxiv.org/abs/2410.15911)
Comments:
          Preprint

- **What's New**: 이 논문에서는 예측 모델을 구축할 때 도메인 특정 요구사항이 적절하게 인코딩되었는지를 보장하는 것이 어렵다는 문제를 다루고 있습니다. 특히 혐오 발언 감지(hate speech detection)에 초점을 맞추어, 사용자가 정의한 혐오 발언의 정의를 모델에 반영하기 위한 새로운 3단계 절차인 DefVerify를 제안합니다.

- **Technical Details**: DefVerify는 (i) 사용자가 특정한 혐오 발언의 정의를 인코딩하고, (ii) 모델이 이 정의를 얼마나 잘 반영하고 있는지를 정량화하며, (iii) 워크플로우의 실패 지점을 식별하려고 합니다. 이를 통해 데이터 세트 구축 및 모델 훈련 과정 중 발생할 수 있는 샘플링 편향(sampling bias), 주석 편향(annotation bias), 모델 불일치(model misspecification) 등의 문제를 해결하고자 합니다.

- **Performance Highlights**: DefVerify는 여섯 개의 인기 있는 혐오 발언 벤치마크 데이터셋에 적용되어 정의와 모델 행동 간의 격차를 식별하는 데 효과적임을 입증하였습니다.



### Using GPT Models for Qualitative and Quantitative News Analytics in the 2024 US Presidental Election Process (https://arxiv.org/abs/2410.15884)
- **What's New**: 본 논문은 Google Search API와 GPT-4o 모델을 활용하여 뉴스의 질적 및 양적 분석을 수행하는 접근 방식을 제안합니다. 이를 통해 2024년 미국 대선 과정에 대한 뉴스를 분석하였습니다.

- **Technical Details**: Retrieval-augmented generation (RAG) 기법을 사용하여 뉴스 데이터를 분석하였습니다. 분석은 Google Search API를 통해 관련 웹 자원을 검색하고, LangChain의 SeleniumURLLoader를 이용하여 정보를 추출하는 두 단계로 이루어졌습니다. 주요 검색 쿼리는 'Kamala Harris AND Donald Trump'이며, 다양한 시간대와 뉴스 출처를 고려하였습니다.

- **Performance Highlights**: GPT 모델을 활용한 분석 결과는 선거 과정에서의 불확실성을 분석하는 데 도움을 주며, 질적 통찰력을 제공함으로써 향후 선거 분석에 응용될 수 있는 가능한 기초 자료를 생성합니다.



### Principles of semantic and functional efficiency in grammatical patterning (https://arxiv.org/abs/2410.15865)
- **What's New**: 본 논문에서는 언어의 문법적 특징들이 의미 부여와 문법적 일치에 기반한 예측 가능성을 통합하여 정보 이론적 관점에서 분석하고, 이러한 문법적 조직이 인지적 제약 아래에서 어떻게 형성되는지를 설명합니다.

- **Technical Details**: 문법 체계의 조직은 다층 최적화 문제로 모델링되며, 의미와 기능 목표를 모두 만족해야 합니다. 문법적 값의 분포는 명사와 같은 단어에서 예측을 용이하게 할 수 있도록 분산되어야 하며, 이는 언어 처리의 효율성을 높여줍니다.

- **Performance Highlights**: 이 모델은 다양한 언어에서 관찰되는 문법적 값의 분포를 설명하는 데 성공하였으며, 문법이 지각적 속성으로부터 유래하면서도 기능적 목표를 우선시하여 효율적인 언어 처리를 촉진하는 방식으로 조직된다는 점을 강조합니다.



### Did somebody say "Gest-IT"? A pilot exploration of multimodal data managemen (https://arxiv.org/abs/2410.15825)
- **What's New**: Gest-IT 프로젝트는 시각 장애인과 일반인의 대화에서 제스처 변화를 분석하기 위한 다중 모드 코퍼스(multimodal corpus)를 구축하고 관리하는 방안에 대한 파일럿 탐색을 제시합니다. 이 연구는 언어의 다양한 세미오틱 소스(semiotic sources)를 독립적으로 관찰하고 그 상호작용을 고려해야 한다고 강조합니다.

- **Technical Details**: Gest-IT 리소스는 정형적(orthographic), 음성적(prosodic), 제스처(gestural) 전사(transcription)의 세 가지 레이어로 주석(annotation)을 제공하여 대화에서 제스처 패턴의 변화를 연구할 수 있도록 합니다. 또한, 코퍼스 모델인 CoNLL-U를 제안하여 다중 모드 데이터를 관리하기 위한 프로토콜을 설명합니다.

- **Performance Highlights**: 기존 자원들과 비교해 Gest-IT는 에콜로지컬(ecological) 데이터의 수집을 지향하며, 자발적인 상호 작용(spontaneous interactions)에서 언어적 및 비언어적 커뮤니케이션을 통합적으로 분석할 수 있는 가능성을 제시합니다. 이로 인해 코퍼스의 활용 범위가 넓어질 것입니다.



### Improve Dense Passage Retrieval with Entailment Tuning (https://arxiv.org/abs/2410.15801)
Comments:
          EMNLP 2024 Main

- **What's New**: 이 연구에서는 질문-답변 관련성을 정의하는 새로운 관점을 제시하고, 그에 기반하여 기존의 dense retriever 훈련 파이프라인에 통합 가능한 'entailment tuning' 방법론을 설계했습니다.

- **Technical Details**: 'Entailment tuning' 방법은 NLI(자연어 추론) 데이터를 활용하여 dense retriever의 성능을 향상시키는 데 초점을 맞추고 있습니다. 주어진 질문을 주장으로 변환하고, 이를 통해 claim-passage 쌍 및 premise-hypothesis 쌍을 구성합니다. 이후 주장의 대부분을 마스킹하고 encoder 모델을 훈련시켜 마스킹된 주장을 예측하도록 합니다. 이를 통해 retrieval 성능을 향상시키는 것을 목표로 합니다.

- **Performance Highlights**: 실험 결과, 'entailment tuning' 방법이 dense passage retrieval 작업에서 효과적임을 입증했으며, open-domain QA 및 retrieval-augmented generation(RAG)와 같은 다운스트림 작업에서 성능 향상을 가져왔습니다.



### Learning-to-Defer for Extractive Question Answering (https://arxiv.org/abs/2410.15761)
Comments:
          25 pages, 17 main paper

- **What's New**: 이번 연구에서는 질문-답변 시스템에서의 효율적 의사결정을 향상시키기 위해 학습-위임(Learning-to-Defer) 메커니즘을 도입하였습니다. 이 접근법은 복잡한 질의에 대해 인간 전문가나 더 큰 QA 모델에게 선택적으로 위임할 수 있도록 하여, 모델의 정확성과 신뢰성을 크게 향상시킵니다.

- **Technical Details**: 제안된 모델은 두 단계의 Learning-to-Defer 메커니즘을 활용하여 의사결정을 개선하며, Bayes 및 $(	ext{H}, 	ext{R})$--일관성을 통해 방법론의 이론적 타당성을 확립합니다. 또한, SQuADv2 데이터셋을 활용하여 경험적으로 성능 개선을 입증하였습니다.

- **Performance Highlights**: L2D QA 모델은 인간 전문가 및 대형 QA 모델과의 상호작용을 통해 성능이 향상되었음을 보였고, 최소한의 질의 위임으로 크기가 작은 모델이 성능을 유지하면서도 컴퓨팅 효율성을 보존할 수 있는 가능성을 제시합니다.



### Natural Language Querying System Through Entity Enrichmen (https://arxiv.org/abs/2410.15753)
- **What's New**: 이 논문은 데이터베이스에 대한 도메인 전문가 질의 시스템을 다루며, 프랑스 기업을 위해 자연어 인터페이스를 제공하는 솔루션을 제시합니다.

- **Technical Details**: 이 접근법은 엔티티(enrichment) 강화를 기반으로 하여 자연어 질의를 데이터베이스 질의로 변환하는 것을 목표로 합니다. 데이터베이스는 논리(logical) 패러다임을 통해 다루어지며, 이는 다양한 데이터베이스 모델에 대한 접근 방식의 적응 가능성을 시사합니다.

- **Performance Highlights**: 우리 방법의 높은 정밀도는 몇 가지 초기 실험을 통해 입증되었습니다.



### Toeing the Party Line: Election Manifestos as a Key to Understand Political Discourse on Twitter (https://arxiv.org/abs/2410.15743)
Comments:
          9 pages, accepted at EMNLP (Findings) 2024

- **What's New**: 이 논문에서는 정치적 입장 추정을 위한 새로운 방법론을 제시하며, 전통적인 선거 공약(manifesto) 자료에서 소셜 미디어인 트위터(twitter)로의 적용을 확장했습니다. 해시태그(hashtag)를 활용하여 수동 주석 없이 정치적 입장 유사성을 예측하는 방식을 개발하였습니다.

- **Technical Details**: 연구는 정치인들이 해시태그를 정책 이슈 영역의 약어로 사용할 수 있다는 통찰력에 기반하여 트윗(tweets)의 문장 임베딩(sentence embeddings)을 계산합니다. 이 방법은 2017년과 2021년의 독일 연방선거 동안 수집된 트윗 데이터를 활용한 일련의 실험을 통해 검증되었습니다. 실험 결과 해시태그를 활용한 파인튜닝(fine-tuning) 접근법이 다른 방법보다 성능이 뛰어남을 입증했습니다.

- **Performance Highlights**: 제안된 방법은 모든 후보자의 트윗을 사용할 경우와 짧은 시간 동안의 작은 샘플을 사용할 경우 모두에서 안정적인 정치적 입장을 반영하는 결과를 보여주었습니다. 이는 소셜 미디어의 노이즈(잡음) 환경에서도 수동 주석 없이 신뢰할 수 있는 상대적 위치 분석이 가능함을 시사합니다.



### Who's Who: Large Language Models Meet Knowledge Conflicts in Practic (https://arxiv.org/abs/2410.15737)
Comments:
          Accepted to EMNLP 2024 Findings

- **What's New**: 이번 연구에서는 Retrieval-augmented generation (RAG) 방법을 사용하여 사전 학습된 언어 모델의 메모리 한계를 극복하려고 하지만, 정보 충돌(conflict) 문제를 해결하기 위한 새로운 벤치마크 데이터 세트인 WhoQA를 소개합니다.

- **Technical Details**: WhoQA는 동일한 이름을 가진 Wikipedia 엔티티에 대한 공통 속성을 묻는 질문을 통해 지식 충돌을 유도합니다. 데이터 세트는 5000개의 질문으로 구성되어 있으며, 최대 8개의 고유한 답변을 포함할 수 있습니다. 질문은 (q, A, S, C) 형태의 쿼드플렛으로 나타내어 집니다.

- **Performance Highlights**: 실험 결과, WhoQA 질문의 단순함에도 불구하고 지식 충돌이 RAG 설정에서 LLM의 성능을 크게 저하시킨다고 보고했습니다.



### Reducing annotator bias by belief elicitation (https://arxiv.org/abs/2410.15726)
- **What's New**: 이 연구에서는 많은 특정 annotators 수나 데이터 인스턴스 수에 대한 요구 없이 주석에서의 편향을 다루기 위한 간단한 방법을 제안합니다. 이 방법은 annotators가 다른 annotators의 판단에 대한 신념을 보고하도록 요청합니다. 이는 신념이 보다 대표적이고 덜 편향된 레이블을 생성할 수 있다는 가설에 기반합니다.

- **Technical Details**: 이 연구는 두 개의 통제된 설문조사 실험을 통해 검토되었으며, 1,590명의 민주당원 및 공화당원이 특정 진술을 평가하고 다른 참가자의 판단에 대한 신념을 보고했습니다. Experiment 1은 대표 그룹의 응답에 대한 신념을 이끌어내기 위해 재정적 인센티브를 사용했으며, Experiment 2에서는 민주당원과 공화당원의 응답에 대한 신념을 개별적으로 이끌어냈습니다.

- **Performance Highlights**: 결과는, 두 그룹의 annotators 간의 체계적인 차이로 정의된 편향이 판단 대신 신념을 요구했을 때 일관되게 줄어드는 것을 보여주었습니다. 이 연구에서 제안된 방법은 AI 시스템의 일반화 가능성을 향상시키고, 무시된 사회-인구 집단에 대한 해를 방지할 수 있는 잠재력을 가지고 있습니다.



### Mitigating Hallucinations of Large Language Models in Medical Information Extraction via Contrastive Decoding (https://arxiv.org/abs/2410.15702)
Comments:
          Accepted by EMNLP 2024 Findings

- **What's New**: 이 논문은 Medical Information Extraction(MIE) 작업에서 대형 언어 모델(LLM)의 망상(Hallucination) 문제를 해결하기 위해 ALternate Contrastive Decoding(ALCD) 방법을 제안합니다. 기존의 방식들과 다르게, MIE 작업을 식별과 분류 프로세스로 재정의하고 LLM의 식별 및 분류 기능을 분리하여 최적화를 진행합니다.

- **Technical Details**: ALternate Contrastive Decoding(ALCD)은 MIE 작업에 맞춰 설계된 간단한 디코딩 전략입니다. 훈련 단계에서, 식별 및 분류 모델의 최적화를 분리하기 위해 토큰 최적화를 선택적으로 마스킹합니다. 추론 단계에서는 서브 작업 모델에서 도출된 출력 분포를 교대적으로 대비하여 식별 및 분류 능력을 개선합니다. 이 과정에서 적응형 제약 조건 전략을 제안하여 대비 토큰의 규모 및 범위를 효과적으로 조정합니다.

- **Performance Highlights**: ALCD는 두 가지 다른 LLM 백본 및 여섯 가지 다양한 MIE 작업에 대한 포괄적인 실험을 통해 기존 디코딩 방법에 비해 망상 문제를 해결하는 데 상당한 성과를 보였습니다. 실험 결과 ALCD가 여덟 가지 기존 디코딩 방법 대비 우수한 성능을 발휘함을 강조합니다.



### Tokenization as Finite-State Transduction (https://arxiv.org/abs/2410.15696)
Comments:
          10 pages + 5 pages in appendix

- **What's New**: 본 논문에서는 정규 언어(regular language)의 모든 가능한 토큰화(tokenization)를 효율적으로 인코딩할 수 있는 유한 상태 전이(finite-state transduction) 프레임워크를 제안합니다. 또한, Byte-Pair Encoding (BPE) 및 MaxMatch (WordPiece)와 같은 두 가지 인기 있는 토큰화 방법이 이 프레임워크에 잘 맞는다는 사실을 밝힙니다.

- **Technical Details**: 정규 언어에 대해 서브워드 토큰(subword token) 레벨 패턴을 생성하는 문제에 접근하며, MaxMatch 및 BPE와 같은 토큰화 알고리즘의 결과를 반영합니다. 저자들은 자동 이론(automata theory)을 통해 서브워드 레벨 자동자(subword-level automaton)를 구축하는 방법을 명확히 설명합니다. BPE에 대한 분석에서는, 비록 단순한 러ntime 분석에서는 선형 시간 복잡도로 보일 수 있지만, 실제로는 입력 패턴과 서브워드 어휘(subword vocabulary)의 크기에 대해 다항식 시간(polynomial time) 내에 처리할 수 있음을 보여줍니다.

- **Performance Highlights**: 본 프레임워크는 특정 패턴과 일치하도록 언어 모델의 출력을 제약하면서 동시에 기본 토큰화기의 표준 토큰화를 준수하도록 합니다. 이를 통해 서브워드 레벨의 가이드 생성(guided generation)에 대한 적용 가능성을 제시합니다.



### Efficient Terminology Integration for LLM-based Translation in Specialized Domains (https://arxiv.org/abs/2410.15690)
Comments:
          Accepted to WMT 2024

- **What's New**: 본 논문에서는 전통적인 기계 번역 방법의 한계를 극복하기 위해 적은 양의 데이터로도 특수 용어의 번역 정확성을 보존하면서 모델을 효율적으로 훈련하는 방법론을 제시합니다.

- **Technical Details**: 우리는 Trie Tree 알고리즘을 사용하여 용어 추출과 용어집(glossary) 생성을 체계적으로 진행한 후, 데이터를 재구성(data reconstruction)하여 LLM이 이러한 특수 용어를 통합할 수 있도록 학습시킵니다. 이 방법론은 모델의 특수 용어 처리 능력을 강화합니다.

- **Performance Highlights**: 우리의 접근법은 WMT 특허 과제에서 최고의 번역 점수를 달성하며, 특수 번역 분야에서의 효과성과 광범위한 적용 가능성을 입증하였습니다.



### DomainSum: A Hierarchical Benchmark for Fine-Grained Domain Shift in Abstractive Text Summarization (https://arxiv.org/abs/2410.15687)
- **What's New**: 이번 연구는 여러 도메인 간의 변화가 추상적 요약 성능에 미치는 영향을 논의하며, 이를 해결하기 위해 DomainSum이라는 계층적 벤치마크를 도입했습니다. 이 벤치마크는 장르, 스타일, 주제라는 세 가지 수준으로 도메인 변화를 분류합니다.

- **Technical Details**: DomainSum 벤치마크는 고품질 공개 데이터셋을 활용하여 만들어졌으며, 다섯 개의 서로 다른 도메인에서 생성된 문서-요약 쌍으로 구성됩니다. 각 도메인에서는 압축 비율, 밀도, 요약 추상성 등 여덟 가지 요약 도메인 특성 측정 항목을 분석합니다.

- **Performance Highlights**: 다양한 미세 조정 수준에서 PLM과 LLM의 도메인 전이 성능을 평가하였습니다. 이전의 모델들이 특정 도메인에 한정된 성능을 가지는 동안, DomainSum을 통해 다양한 유형의 콘텐츠에서의 요약 성능을 종합적으로 비교할 수 있는 기회를 제공합니다.



### Revealing and Mitigating the Local Pattern Shortcuts of Mamba (https://arxiv.org/abs/2410.15678)
- **What's New**: 최근 Mamba라는 고급 모델이 도입되었습니다. 이 모델은 State Space Models(SSMs)에 기반하여 단순히 Attention 메커니즘에 비해 선형 복잡성과 상수 메모리 요구사항을 제공하며, 성능이 주목받고 있습니다. 하지만, Mamba는 지역 정보를 잘 처리하지만 분산된 정보를 처리하는 데 어려움을 겪고 있습니다.

- **Technical Details**: Mamba 모델은 Selective State Space Model로, 이전 SSMs와의 주요 차이점은 모델의 동적 상태가 시간에 따라 효율적으로 업데이트된다는 점입니다. 이를 위해 Mamba는 입력에 따라 변화하는 특별한 학습 가능한 선형 층을 도입합니다. 이러한 변화는 상황에 맞게 매개 변수를 조정하여 Mamba가 더 유연한 시퀀스 모델링을 가능하게 합니다.

- **Performance Highlights**: Mamba 모델은 단지 4M의 추가 매개 변수를 포함시켰을 뿐임에도 불구하고, 고정보 밀도 합성 작업에서 0에서 80.54점으로의 성과 향상을 이뤘습니다. 이는 Mamba 모델과 Attention 기반 모델 간의 성능 차이를 줄이는 데 기여하였습니다.



### Learning to Generate and Evaluate Fact-checking Explanations with Transformers (https://arxiv.org/abs/2410.15669)
Comments:
          Forthcoming in Engineering Applications of Artificial Intelligence

- **What's New**: 디지털 플랫폼이 지배하는 시대에 정보의 진위를 평가할 수 있는 솔루션 개발의 필요성이 강조되고 있습니다. 본 연구는 Explainable Artificial Intelligence (XAI) 분야에 기여하기 위해, 결정을 설명하는 인간 친화적인 설명을 생성하는 transformer 기반의 사실 확인 모델을 개발했습니다.

- **Technical Details**: 자동 평가 모델을 통해 사실 확인 결정에 대한 설명을 	exttt{(self)-contradiction}, 	exttt{hallucination}, 	exttt{convincingness}, 	exttt{overall quality}와 같은 다양한 차원에서 평가할 수 있습니다. 인간 중심의 평가 방법과 특수화된 데이터셋을 개발하여 AI 생성 설명을 인간 판단과 일치시키는 필요성을 강조하고 있습니다. 또한 메트릭 학습 모델을 통해 효율성을 증가시키고 방대한 수동 평가에 대한 의존도를 줄이기 위한 첫걸음을 제시하고 있습니다.

- **Performance Highlights**: 실험 결과, 최고의 생성 모델의 	extsc{ROUGE-1} 점수는 47.77로, 고품질 증거가 제공될 때 사실 확인 설명을 생성하는 데 있어 우수한 성능을 보였습니다. 또한 최고의 메트릭 학습 모델은 	exttt{(self)-contradiction} 및 	exttt{hallucination}과 같은 객관적인 차원에서 인간 판단과 중간 정도의 강한 상관 관계를 나타내며, Matthews Correlation Coefficient (MCC)가 약 0.7에 도달했습니다.



### RAC: Efficient LLM Factuality Correction with Retrieval Augmentation (https://arxiv.org/abs/2410.15667)
- **What's New**: 본 논문은 Retrieval Augmented Correction (RAC)이라는 저지연 후처리 방법을 소개하여 Large Language Models (LLMs)의 사실 정확도(factual performance)를 향상시키는 데 기여합니다. RAC는 추가적인 fine-tuning 없이도 사용 가능하며, LLM의 출력을 기본적인 사실들(atomic facts)로 분해한 후, 검색된 내용(retrieved content)을 사용하여 검증 및 수정하는 과정을 포함합니다.

- **Technical Details**: RAC는 LLM의 출력물을 기본적인 몇 가지 사실로 분해하고, 이 사실들을 검색된 지식을 통해 확인하거나 수정한 후 LLM의 출력을 수정하는 방법입니다. 이 접근 방식은 Retrieval-Augmented Generation (RAG)의 후처리 구성 요소로 볼 수 있으며, 두 개의 사실성 평가 데이터셋을 통해 기존 방법들에 비해 최대 30% 향상된 성능을 보여주었습니다. RAC는 검색을 한 번 수행하고 수정을 한 번 수행하여 이전 방법들에 비해 지연(latency)을 크게 감소시켰습니다.

- **Performance Highlights**: RAC는 LLM의 사실성 성능을 향상시키는 데 있어 유효성과 견고성을 보여주며, RAG와 통합한 경우와 그렇지 않은 경우 모두에서 30% 이상의 성능 향상을 달성하였습니다. 특히, 일부 사례에서는 RAG 없이도 RAG와 함께 사용할 때보다 성능이 더 우수한 결과를 나타내어 우수성을 입증하였습니다.



### Scalable Data Ablation Approximations for Language Models through Modular Training and Merging (https://arxiv.org/abs/2410.15661)
Comments:
          EMNLP 2024. 17 pages

- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 하위 성능에 영향을 미치는 학습 데이터 조합의 효과를 분석할 수 있는 효율적인 방법을 제안하고 있습니다. 기존의 데이터 제외 실험(data ablation) 방식이 비용이 많이 드는 반면, 제안된 방법은 모델을 독립적으로 학습시키고 이전 학습 결과를 재사용하여 데이터 조합의 평가를 가능하게 합니다.

- **Technical Details**: 제안된 접근법에서는 모델을 특정 데이터 파티션으로 학습시킨 후, 이 모델들을 파라미터 평균(parameter average)을 사용하여 다양한 데이터 조합에 대한 퍼플렉시티(perplexity) 성능을 예측합니다. 또한, 새로운 데이터 조합을 평가하기 위한 학습 양이 선형적으로 증가하도록 최적화되어 있습니다.

- **Performance Highlights**: 이 연구에서는 다양한 도메인에서 데이터 조합의 퍼플렉시티 평가 성능을 예측할 수 있으며, 특히 ‘최적’ 성능 분포의 쪽에서 더 신뢰할 수 있는 결과를 기대할 수 있습니다. 산업계에서 모델 개발 시, 주어진 고려 사항들에 따라 더욱 효율적인 데이터 혼합 및 평가가 가능함을 시사합니다.



### Resource-Efficient Medical Report Generation using Large Language Models (https://arxiv.org/abs/2410.15642)
- **What's New**: 이 연구에서는 흉부 X-레이 이미지를 위한 자동 의료 보고서 생성을 위한 새로운 프레임워크를 제안합니다. 이 프레임워크는 비전 기반의 대형 언어 모델(LLM)을 활용하여, 경량화된(solution) 방식으로 기존의 방법들과 비교해 우수한 성능을 달성하고 있습니다.

- **Technical Details**: 제안된 방법은 비전 인코더, 대형 언어 모델, 매핑 네트워크로 구성됩니다. 의료 관련 CLIP(MedCLIP)을 사용하여 시각적 임베딩(prefix embeddings)을 추출하고, 이를 경량화된 매핑 네트워크를 통해 언어 모델의 공간으로 변환합니다. 여기서 prefix tuning을 사용하여 LLM을 미세 조정하지 않고 성능을 향상시킵니다.

- **Performance Highlights**: Qwen1.5 LLM이 GPT-2 모델보다 의료 보고서 생성에서 더 우수한 NLG 메트릭(예: Bleu 점수)을 기록했습니다. 제안된 방법은 이전의 대형 LLM 기반 솔루션보다 성능이 뛰어나며, 자원 효율성을 확인하였습니다.



### SMILES-Prompting: A Novel Approach to LLM Jailbreak Attacks in Chemical Synthesis (https://arxiv.org/abs/2410.15641)
- **What's New**: 본 논문은 화학 분야의 대형 언어 모델(LLM)에서 위험한 물질을 합성하는 방법에 대한 정보 유출 가능성을 조사합니다. 특히 SMILES prompting이라는 새로운 공격 기법을 소개하며, 이는 화학 물질의 정보를 참조하기 위해 SMILES(Simplified Molecular-Input Line-Entry System) 구조 표기를 활용합니다.

- **Technical Details**: 기존의 red-teaming, explicit prompting, implicit prompting과 같은 여러 접근 방식을 평가하며, SMILES-prompting 기법이 LLM의 안전 메커니즘을 효과적으로 우회할 수 있음을 발견했습니다. SMILES-prompting의 성공은 공격의 도메인을 덜 방어된 다른 '언어'(encoding)로 전환하는 것과 관련이 있습니다.

- **Performance Highlights**: SMILES-prompting은 Llama 모델에서 모든 기준에 대해 다른 공격 방식보다 우수한 성능을 보였습니다. GPT-4o 모델에서의 합성 과정에 대한 응답에서도 높은 성공률을 기록했습니다. 이 연구는 LLM의 안전 메커니즘을 개선하기 위한 필요성을 강조합니다.



### Can Large Language Models Invent Algorithms to Improve Themselves? (https://arxiv.org/abs/2410.15639)
- **What's New**: 이번 연구는 Large Language Models (LLMs)가 인간의 개입 없이 스스로 개선 알고리즘을 생성하고 학습할 수 있는 Self-Developing 프레임워크를 제안합니다.

- **Technical Details**: Self-Developing 프레임워크에서 시드 모델은 모델 개선 알고리즘을 생성하고 적용하며 학습합니다. 이는 모델과 알고리즘의 지속적인 개선을 가능하게 합니다.

- **Performance Highlights**: Self-Developing은 수학적 추론 과제에서 시드 모델을 초월하는 모델을 생성하고, 인간이 설계한 알고리즘을 사용한 모델보다 일관되게 뛰어난 성능을 보여줍니다. 또한, LLM이 발견한 알고리즘은 다른 도메인 모델에도 강력한 효과를 나타냅니다.



### Selecting Influential Samples for Long Context Alignment via Homologous Models' Guidance and Contextual Awareness Measuremen (https://arxiv.org/abs/2410.15633)
- **What's New**: 이 연구는 긴 맥락을 가진 명령을 효과적으로 처리하기 위한 고품질 데이터셋 구축의 필요성을 강조합니다. GATEAU라는 새로운 프레임워크를 제안하여 중요하고 품질 높은 샘플을 식별합니다.

- **Technical Details**: GATEAU는 Homologous Models' Guidance (HMG) 및 Contextual Awareness Measurement (CAM)라는 두 가지 방법을 활용하여 긴 의존 관계(long-range dependency relations)가 풍부한 샘플을 찾아냅니다. HMG는 서로 다른 맥락 윈도우를 가진 두 개의 동형 모델의 반응의 perplexity 점수를 측정하여 그 난이도를 평가합니다. CAM은 모델의 주의(attention)가 중요한 구간에 집중되고 있는지를 평가하여 긴 입력 맥락의 이해 난이도를 측정합니다.

- **Performance Highlights**: GATEAU를 통해 선택된 샘플로 훈련된 모델은 명령 수행 및 긴 맥락 이해 능력이 향상됩니다. 실험 결과, 이 프레임워크는 긴 의존 관계가 있는 샘플을 효과적으로 식별할 수 있음을 보여주었습니다.



### Guardians of Discourse: Evaluating LLMs on Multilingual Offensive Language Detection (https://arxiv.org/abs/2410.15623)
Comments:
          Accepted at UIC 2024 proceedings. Accepted version

- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)이 영어, 스페인어 및 독일어에서 다국어 외설어 탐지의 능력을 평가한 최초의 종합적인 연구를 다루고 있습니다. 특히, 비영어 컨텍스트에서의 다른 프롬프트 언어와 증강 번역 데이터의 영향을 추가로 조사했습니다.

- **Technical Details**: 이 연구는 OLID + SOLID (영어), OffendES (스페인어), GermEval 2018 (독일어)의 세 데이터 세트를 활용하고, GPT-3.5, Flan-T5 및 Mistral의 세 가지 LLM을 비교했습니다. 실험에서는 매크로 정밀도(macro precision), 매크로 재현율(macro recall), 매크로 F1-점수(macro F1-score)를 평가 지표로 사용했습니다.

- **Performance Highlights**: LMM은 다국어 외설어 탐지에서 경쟁력 있는 성능을 달성했습니다. 특히 GPT-3.5와 Mistral은 모든 언어에서 강력한 성능을 보였으나, Flan-T5는 영어 외설어 탐지에서만 성공적이었습니다. 원주율 언어에서의 프롬프트를 활용하면 비영어 컨텍스트에서 모델 이해도가 향상되지만, 번역 데이터 포함은 성능 향상에 기여하지 않았습니다.



### Interventional Speech Noise Injection for ASR Generalizable Spoken Language Understanding (https://arxiv.org/abs/2410.15609)
Comments:
          9 pages, 3 figures

- **What's New**: 이 논문에서는 자동 음성 인식(ASR) 시스템의 오류에 강한 Spoken Language Understanding (SLU) 모델을 훈련하기 위한 새로운 방법을 제안합니다. ASR의 오류를 효과적으로 줄이기 위해, 기존의 Speech Noise Injection (SNI) 기법의 한계를 극복하는 방식을 도입했습니다.

- **Technical Details**: 이 논문에서 제안하는 기법은 'Interventional Noise Injection (ISNI)'로, ASR 오류 패턴을 넓히기 위해 도칼culus(docalculus)를 사용하여 생성된 의사 전사(pseudo transcription)에 ASR-specific noises를 적용합니다. 또한, 'Phoneme-aware generation' 방식을 통해 ASR 시스템에 상관없이 적용 가능한 소음을 생성합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법론은 다양한 ASR 시스템에 대해 뛰어난 일반화 성능을 보여주었으며, SLU 모델의 성능이 ASR 훈련 설정과 동등하거나 초과하는 결과를 기록했습니다.



### AMPLE: Emotion-Aware Multimodal Fusion Prompt Learning for Fake News Detection (https://arxiv.org/abs/2410.15591)
- **What's New**: 이번 연구에서는 감정 인식 기반의 다중 모달 융합 프롬프트 학습(Emotion-Aware Multimodal Fusion Prompt Learning, AMPLE) 프레임워크를 소개하여 가짜 뉴스 탐지의 효율성을 높이는 새로운 접근법을 제시합니다.

- **Technical Details**: AMPLE 프레임워크는 감정 분석 도구(Sentiment Analysis Tool, SAT)를 활용하여 텍스트에서 감정 요소를 추출하고, Multi-Head Cross-Attention (MCA) 기법을 통해 다중 모달 데이터를 통합합니다. 이 프레임워크는 기존의 대량 주석 데이터에 대한 의존도를 줄이며, 고정된 텍스트 프롬프트와 조정 가능한 벡터를 통합한 하이브리드 프롬프트 템플릿을 활용합니다.

- **Performance Highlights**: AMPLE 프레임워크는 PolitiFact 및 GossipCop의 두 공개 데이터셋에서 평가되었으며, F1과 정확도 면에서 기존 방법보다 우수한 성능을 보였습니다. 특히 기계 학습(ML)과 딥 러닝(DL) 기술을 통합한 접근법은 감정 요소의 긍정적 상관관계를 입증했습니다.



### A Survey of Conversational Search (https://arxiv.org/abs/2410.15576)
Comments:
          35 pages, 8 figures, continue to update

- **What's New**: 현대 정보 접근의 초석으로서, 검색 엔진은 일상생활에서 없어서는 안 될 존재가 되었습니다. AI와 자연어 처리(NLP) 기술이 빠르게 발전함에 따라, 특히 대규모 언어 모델(LLMs)의 출현이 검색 엔진의 진화를 이끌어, 보다 직관적이고 지능적인 사용자 시스템 간 상호작용을 지원하게 되었습니다. 최근 '대화식 검색'이라는 새 패러다임이 대두되고 있습니다.

- **Technical Details**: 대화식 검색 시스템은 사용자와의 복잡하고 정밀한 정보 검색을 원활하게 하기 위해 자연어 대화를 활용하며, 전통적인 키워드 기반 검색 엔진과의 차별화된 특징을 가지고 있습니다. 이 시스템은 쿼리 재구성(query reformulation), 검색 명확화(search clarification), 대화 검색(conversational retrieval), 응답 생성(response generation)과 같은 주요 구성 요소를 갖추고 있습니다.

- **Performance Highlights**: 대화식 검색의 주요 장점은 사용자 경험을 향상시키는 데 있으며, 복잡한 쿼리를 지원하고, 다중 대화 상호작용에서 맥락을 유지하며, 강력한 정보 통합 및 처리 기능을 제공합니다. 이 연구는 현재의 대화식 검색 시스템의 현실 세계 응용 사례와 평가도 제공해 향후 연구 및 개발 방향을 제시하고자 합니다.



### Neural Search Space in Gboard Decoder (https://arxiv.org/abs/2410.15575)
Comments:
          10 pages, 7 figures, 3 tables

- **What's New**: 이 논문에서는 Gboard 디코더의 N-그램 언어 모델(LM)을 신경망 언어 모델(NN-LM)로 대체하고, 디코딩 중에 동적으로 검색 공간을 구성하는 새로운 'Neural Search Space'를 제안합니다. 이 접근 방법은 신경망의 긴 컨텍스트 인식 기능을 검색 공간에 통합하고, 실시간으로 LST 구조를 재설계하는 과정을 포함합니다.

- **Technical Details**: Neural Search Space(NSS)는 기존의 N-그램 LM을 대체하여, Gboard의 디코딩 과정에서 사용자의 입력에 따라 동적으로 언어 FST(유한 상태 변환기)를 구성합니다. NSS는 OOV(어휘 외) 문제와 검색 공간 폭발을 방지하는 전략을 포함하며, 이를 위해 세심하게 설계된 FST 구조, 정확한 가지치기 전략 및 데이터 구조 최적화를 적용했습니다.

- **Performance Highlights**: NSS를 사용한 실험 결과, 다양한 언어 환경에서 WMR(수정된 단어 비율)을 [0.26%, 1.19%] 줄이는 데 성공하며, 수용 가능한 레이턴시 증가를 나타냈습니다. 이 연구는 수천만 명의 사용자에게 직접적으로 적용 가능함을 입증하며, 타이핑 속도 향상으로 인해 사용자 경험이 개선되었습니다.



### Leveraging Retrieval-Augmented Generation for Culturally Inclusive Hakka Chatbots: Design Insights and User Perceptions (https://arxiv.org/abs/2410.15572)
Comments:
          Accepted to IEEE RASSE 2024

- **What's New**: 이 연구는 대만 Hakka 문화의 보호 및 홍보를 위한 혁신적인 접근 방식을 제시하며, Retrieval-Augmented Generation (RAG) 기술이 향상된 챗봇 개발을 통해 이루어집니다.

- **Technical Details**: RAG 기술은 전통적인 대형 언어 모델(LLMs)의 제한을 보완해 정확하고 맥락이 풍부한 응답을 가능하게 합니다. 이 챗봇은 Hakka 전통과 언어, 관습을 반영하기 위해 특별히 큐레이션된 문화 데이터를 활용하여 지식 기반을 강화합니다.

- **Performance Highlights**: 사용자 만족도와 참여도가 눈에 띄게 향상되었으며, 챗봇이 Hakka 문화와의 더 깊은 연결을 촉진하는 데 효과적임을 보여줍니다. RAG 기술은 사용자 경험을 향상시키고 민족 주류화 및 문화 축하의 중요한 도구로 활용될 가능성을 지니고 있습니다.



### Stacking Small Language Models for Generalizability (https://arxiv.org/abs/2410.15570)
- **What's New**: 최근 대형 언어 모델(LLMs)의 강력한 성능을 다양한 자연어 벤치마크에 일반화할 수 있는 잠재력을 보여주었습니다. 그러나 이러한 모델의 크기 때문에 훈련과 추론이 비쌀 뿐만 아니라 자원이 제한된 환경에서 사용하기에는 비현실적입니다. 이 논문은 작은 언어 모델(SLM)을 쌓아서 사용하는 새로운 방법인 FSLM(Fine-tuning Stacks of Language Models)을 소개합니다.

- **Technical Details**: FSLM은 다수의 SLM을 연결하여 각 모델이 특정 작업을 수행하도록 세분화함으로써 높은 수준의 추론을 여러 낮은 수준의 단계로 나누는 방식을 채택했습니다. FSLM은 높은 비용의 훈련과 추론을 줄이면서 모델의 해석 가능성을 향상시킬 수 있습니다. 이를 통해 SLM들이 상호작용하며 자연어로 의사소통하게 됩니다.

- **Performance Highlights**: FSLM은 Alpaca 데이터셋을 활용하여 훈련되었고, 여러 자연어 벤치마크에서 기존 Pythia 및 Flan 모델들을 상대로 성능을 평가한 결과, FSLM 스택이 상대적으로 경량적인 LLM의 효과적인 대안으로 자리잡을 가능성을 보여줍니다. 초기 결과는 FSLM이 비슷한 크기의 모델들에 비해 성능이 향상되었음을 나타냅니다.



### Multi-IF: Benchmarking LLMs on Multi-Turn and Multilingual Instructions Following (https://arxiv.org/abs/2410.15553)
- **What's New**: Multi-IF라는 새로운 벤치마크를 도입하여 대형 언어 모델(LLMs)의 다회전(multi-turn) 및 다국어(multilingual) 지침 따르기 능력을 평가합니다.

- **Technical Details**: Multi-IF는 하이브리드 프레임워크를 사용하여 LLM과 human annotators를 결합하며, 4,501개의 다국어 대화를 포함하고 각 대화는 3턴으로 구성됩니다. 기존의 IFEval을 확장하여 다회전과 다국어 시퀀스를 포함합니다.

- **Performance Highlights**: 14개의 최신 LLM을 Multi-IF에서 평가한 결과, 각 턴이 추가될수록 모델의 수행 정확도가 감소하며, 특히 비라틴 스크립트(힌디어, 러시아어, 중국어)에서 높은 오류율을 보였습니다.



### WHoW: A Cross-domain Approach for Analysing Conversation Moderation (https://arxiv.org/abs/2410.15551)
Comments:
          36 pages(including appendix, 10 pages main text), 8 figures, 16 tables

- **What's New**: 본 연구에서는 WHoW라는 평가 프레임워크를 제안하여 다양한 도메인과 상황에서 중재자의 전략(Why, How, Who)을 분석하고자 하였습니다. 이 프레임워크를 통해 인공지능 GPT-4o와 인간 심사자가 주관한 중재 문장 5,657개와 15,494개를 주석 처리했습니다.

- **Technical Details**: WHoW 프레임워크는 중재자 행동을 동기(Why), 대화 행위(How), 목표 화자(Who) 세 가지 차원으로 나누어 분석하는 방식입니다. 두 가지 도메인: TV 토론과 라디오 패널 토론의 대화 전사문을 가지고 이 구조를 적용하였습니다.

- **Performance Highlights**: 중재자들은 토론에서 조정과 상호작용을 강조하는 반면, 패널 논의에서는 정보 제공과 적극적인 참여에 중점을 두는 것으로 나타났습니다. 이 프레임워크는 다양한 중재 시나리오에 적용 가능하며, 중재자 행동에 대한 이해를 증진시키고 자동 대규모 분석을 통해 중재자 에이전트 개발을 촉진할 잠재력을 지니고 있습니다.



### Grammatical Error Correction for Low-Resource Languages: The Case of Zarma (https://arxiv.org/abs/2410.15539)
- **What's New**: 본 연구는 Zarma 언어에 대한 Grammatical Error Correction (GEC) 접근 방식을 비교하여 전통적인 규칙 기반 방법, 기계 번역(MT) 모델, 그리고 대규모 언어 모델(LLMs)의 효과를 평가합니다.

- **Technical Details**: 연구는 250,000개 이상의 예시로 구성된 수작업 데이터셋을 활용하여 각 접근 방식의 효과를 측정했습니다. M2M100 모델을 사용하는 MT 기반 접근 방식이 자동 평가에서 95.82%의 검출률과 78.90%의 제안 정확도를 달성하며 가장 우수한 성과를 보였습니다. LLM인 MT5-small은 90.62%의 검출률과 57.15%의 제안 정확도를 기록했습니다.

- **Performance Highlights**: MT 기반 모델이 GEC에서 가장 높은 정확도를 보여주었고, 규칙 기반 방법은 철자 수정에 대해 100% 검출률을 기록했습니다. LLM 모델은 상대적으로 낮은 제안 정확도를 보였고, 전체적으로 MT 모델이 저자원 언어에서 GEC를 개선하는 잠재력을 강조했습니다.



### Do RAG Systems Cover What Matters? Evaluating and Optimizing Responses with Sub-Question Coverag (https://arxiv.org/abs/2410.15531)
- **What's New**: 이 논문은 Retrieval-Augmented Generation (RAG) 시스템의 평가를 위한 새로운 프레임워크를 소개합니다. 이 프레임워크는 개방형 질문에서 하위 질문의 커버리지(sub-question coverage)를 측정하여 RAG 시스템이 질문의 다양한 측면을 얼마나 잘 해결하는지를 평가합니다.

- **Technical Details**: 질문을 하위 질문으로 분해하고 이를 core, background, follow-up 세 종류로 분류하여 그 역할과 중요성을 반영합니다. 이 방식으로, RAG 시스템의 검색 및 생성 특성을 분석하는 세부 평가 프로토콜을 제안합니다. 연구에 포함된 상업적 생성 응답 엔진으로는 You.com, Perplexity AI, Bing Chat이 있습니다.

- **Performance Highlights**: 모든 응답 엔진은 코어 하위 질문을 배경 또는 후속 질문에 비해 더 자주 다루었지만, 여전히 약 50%의 코어 하위 질문을 놓치고 있어 개선의 여지가 있음을 보여줍니다. 또한, 하위 질문 커버리지 메트릭은 응답 순위를 매기는 데 효과적이며, 인간 선호 주석과 비교해 82%의 정확성을 달성했습니다. 마지막으로, 코어 하위 질문을 활용하면 RAG 시스템에서 검색과 답변 생성을 모두 향상시켜 74%의 승률을 나타냈습니다.



### M-RewardBench: Evaluating Reward Models in Multilingual Settings (https://arxiv.org/abs/2410.15522)
Comments:
          16 pages, 6 figures, 10 tables. Website: this https URL

- **What's New**: 본 연구에서는 멀티링구얼(multilingual) 환경에서의 보상 모델(reward models, RMs)에 대한 시스템적인 평가를 수행하였습니다. 이를 위해 23개의 다양한 언어를 포함한 최초의 멀티링구얼 RM 평가 벤치마크인 M-RewardBench를 구축하였습니다.

- **Technical Details**: M-RewardBench는 2.87천 개의 선호(preference) 인스턴스로 구성되어 있으며, RMs의 채팅(chat), 안전성(safety), 추론(reasoning), 번역 번역(translation) 능력을 테스트합니다. 이 연구에서는 다양한 보상 모델을 M-RewardBench에서 엄격하게 평가하여, 비영어(non-English) 언어와 영어(Eglish) 언어 간의 성능 차이를 발견하였습니다.

- **Performance Highlights**: 연구 결과, RMs의 선호는 언어에 따라 실질적으로 변화하며, 번역 품질이 향상되면 RMs 성능도 개선된다는 것을 보여주었습니다. 또한 고자원(high-resource) 언어에 대한 성능이 더욱 우수함을 입증하였습니다. 본 연구에서 M-RewardBench 데이터셋 및 코드베이스를 공개하여 멀티링구얼 RM 평가에 대한 이해를 높이는데 기여하고자 하였습니다.



### SceneGraMMi: Scene Graph-boosted Hybrid-fusion for Multi-Modal Misinformation Veracity Prediction (https://arxiv.org/abs/2410.15517)
- **What's New**: 본 논문에서는 SceneGraMMi라는 새로운 다중 모달 가짜 뉴스 탐지 모델을 제안합니다. 이 모델은 다양한 모달리티에서 생성된 시나리오 그래프(scene graphs)를 통합하여 탐지 성능을 향상시킵니다.

- **Technical Details**: SceneGraMMi는 두 가지 단계로 구성된 하이브리드 모달리티 융합 모델입니다. 첫 번째 단계는 Transformer Encoder 모듈을 사용한 초기 융합(early fusion)이며, 두 번째 단계는 Graph Neural Network (GNN)를 통한 후속 융합(late fusion)입니다. 텍스트 토큰과 이미지 패치를 결합한 후, GNN을 통해 생성된 시나리오 그래프를 처리하여 특징을 추출합니다.

- **Performance Highlights**: 실험 결과, SceneGraMMi는 네 가지 기준 벤치마크 데이터셋에서 기존의 최첨단 방법들보다 일관되게 우수한 성능을 보였습니다. 또한 각 구성 요소의 기여도를 강조하며, Shapley 값을 활용해 모델의 의사결정 과정의 설명 가능성을 분석했습니다.



### Reverse Question Answering: Can an LLM Write a Question so Hard (or Bad) that it Can't Answer? (https://arxiv.org/abs/2410.15512)
Comments:
          In-progress preprint

- **What's New**: 이 연구는 기존의 질문 응답(Question Answering, QA) 방식과 역 질문 응답(Reverse Question Answering, RQA)을 혼합하여 분석하고 있습니다. RQA에서는 주어진 답변에 대해 질문을 생성하는 과제를 평가하며, LLMs(대형 언어 모델)들의 성능을 다양한 측면에서 조사하였습니다.

- **Technical Details**: QA는 입력 질문 q에 대한 올바른 답 a를 유도하는 작업입니다. 반면 RQA는 주어진 답 a에 대해 유효한 질문 q를 생성하는 작업으로, 두 작업을 연결하여 LLM의 추론 일관성을 평가할 수 있습니다. 연구는 16종의 LLM이 수행하는 QA와 RQA를 시험했으며, 3443개의 퀴즈 질문/답변 쌍을 수집하여 평가하였습니다. RQA에서는 수치적 답변의 정확도가 크게 떨어지는 경향이 있었지만, 텍스트 기반 답변에서는 약간 더 나은 정확도를 보였습니다.

- **Performance Highlights**: RQA 성능에서는 LLM들이 고난이도 질문을 생성하는 데 어려움을 겪었으며, RQA 오류는 질문의 난이도와 관계가 있는 것으로 나타났습니다. LLM들은 자신의 RQA 질문에 대해 QA에서 유효한 답변을 할 수 있었고, 이는 단순한 지식의 공백 때문이 아니라는 점을 시사합니다. RQA의 오류를 줄이기 위한 다양한 전략이 제안되었습니다.



### RoMemes: A multimodal meme corpus for the Romanian languag (https://arxiv.org/abs/2410.15497)
Comments:
          12 pages, 7 tables, 1 figure, submitted to The 19th International Conference on Linguistic Resources and Tools for Natural Language Processing (ConsILR 2024)

- **What's New**: 이번 논문에서는 루마니아어로 작성된 실제 밈(meme)의 큐레이트된(dataset of curated) 데이터셋을 소개합니다. 이는 다양한 주석(annotation) 수준을 포함하고 있습니다.

- **Technical Details**: AI 응용 프로그램이 밈의 메시지를 추출하고 이해하기 위해 멀티모달(multimodal) 알고리즘을 사용해야 한다고 언급하며, 밈은 주로 그래픽 표현(이미지, 그림, 애니메이션 또는 비디오)과 텍스트를 결합합니다. 논문에서는 기본 알고리즘(baseline algorithms)을 사용하여 데이터셋의 유용성을 입증합니다.

- **Performance Highlights**: 결과는 AI 도구가 인터넷 밈에 직면했을 때 처리 능력(processing capabilities)을 향상시키기 위한 추가 연구가 필요함을 보여줍니다.



### "What is the value of {templates}?" Rethinking Document Information Extraction Datasets for LLMs (https://arxiv.org/abs/2410.15484)
Comments:
          Accepted to EMNLP Findings 2024

- **What's New**: K2Q라는 새로운 데이터셋을 소개하며, 다양한 템플릿을 활용해 기존의 KIE 데이터를 프롬프트-응답 형식으로 변환한 작업을 보여줍니다.

- **Technical Details**: K2Q는 KIE 데이터셋 다섯 개에서 변환되어 형성되었으며, 100,000개 이상의 다양한 템플릿을 사용하여 300,000개 이상의 질문을 생성했습니다. 질문은 extractive 또는 boolean 형식으로 다수의 엔티티를 아우를 수 있습니다.

- **Performance Highlights**: 다양하고 복잡한 KIE 질문을 통해 VRDU 모델의 성능을 40% 개선한 결과를 보여줍니다. K2Q는 기존의 단순 템플릿보다 더 낮은 self-BLEU와 perplexity 점수를 기록하여 인간이 만든 VQA 데이터셋에 더 가까운 특성을 나타냅니다.



### Hey GPT, Can You be More Racist? Analysis from Crowdsourced Attempts to Elicit Biased Content from Generative AI (https://arxiv.org/abs/2410.15467)
- **What's New**: 이번 연구는 비전문 사용자들이 Generative AI (GenAI) 툴의 편향된 출력을 어떻게 인식하고 상호작용하는지에 대한 탐구를 진행하였습니다. 이는 대규모 언어 모델(LLM)의 편향성 문제를 다룬 기존 연구와의 차별화된 접근으로, 사용자들이 만든 프롬프트를 통해 편향을 유도하는 방법을 분석하였습니다.

- **Technical Details**: 이 연구는 펜실베이니아 주립대학교에서 열린 대회에서 수집된 데이터를 기반으로 하였습니다. 참가자들은 GenAI 툴에서 편향된 출력을 유도하기 위한 프롬프트를 설계하며, 총 75개의 유효한 제출물이 접수되었습니다. 연구팀은 이 중 80% 이상의 프롬프트가 재현 가능하다는 결과를 도출하였으며, 8가지 유형의 편향으로 분류되었습니다.

- **Performance Highlights**: 대회에서 선정된 승상들은 GenAI 툴에서 편향된 출력을 유도하는 다양한 전략을 활용하였고, 인터뷰 분석을 통해 참가자들이 정의한 편향의 개념과 전략들을 밝혀냈습니다. 연구 결과는 비전문 사용자들이 LLMs를 어떻게 조작하는지에 대한 독특한 통찰을 제공합니다.



### Keep Guessing? When Considering Inference Scaling, Mind the Baselines (https://arxiv.org/abs/2410.15466)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)에서 반복 sampling을 통한 추론(compute) 증가가 문제 해결률(coverage) 증가에 미치는 영향을 분석하였습니다. 이들은 표준 평가 벤치마크의 정답 분포가 일반적인 정답으로 편향되어 있다는 가설을 세우고, 이에 대한 검증을 위한 baseline을 정의하였습니다.

- **Technical Details**: 세 가지 접근 방식이 비교되었습니다: (1) ModelAnswers - 모델 응답을 샘플링하여 k개의 후보 답안을 얻는 방법, (2) TrainCounts - 훈련 세트에서 가장 빈도가 높은 k개의 답안을 열거하여 후보 답안을 얻는 방법, (3) Mixture(M) - ModelAnswers와 TrainCounts 혼합 전략으로, M개의 답안을 모델 샘플링으로, 나머지 k-M개의 답안을 열거하여 얻는 방법입니다.

- **Performance Highlights**: 실험 결과, 일부 모델은 문제-무관 추측(TrainCounts)보다 더 낮은 성능을 보였으며, Mixture 접근법이 ModelAnswers와 거의 같은 수준의 문제 해결률을 달성하였습니다. 이러한 결과는 데이터셋 선택과 모델 성능 평가 시 주의가 필요하다는 점을 시사합니다.



### A Novel Interpretability Metric for Explaining Bias in Language Models: Applications on Multilingual Models from Southeast Asia (https://arxiv.org/abs/2410.15464)
Comments:
          Accepted for oral presentation at the 38th Pacific Asia Conference on Language, Information, and Computation

- **What's New**: 이 논문에서는 pretrained language models (PLMs)에서의 bias attribution(편향 귀속) 문제를 다루고, 이를 측정하기 위한 새로운 지표인 bias attribution score를 제안합니다. 이 지표는 정보 이론을 활용하여 token(토큰) 수준에서 편향된 행동에 기여하는 요소를 측정합니다.

- **Technical Details**: 저자들은 Southeast Asian(동남아시아) PLMs를 포함하여 다국어 PLMs에 대해 bias evaluation(편향 평가)을 시행하며, Crowdsourced Stereotype Pairs (CrowS-Pairs) 데이터셋을 통해 해당 모델들이 성차별적 및 동성애 혐오적 편향을 나타냄을 확인합니다. 또한, token 수준의 bias attribution score를 계산하여 각 단어가 편향적인 문장을 선택하는 데 어떻게 기여하는지를 분석합니다.

- **Performance Highlights**: 이 연구의 주요 기여는 세 가지로 나뉩니다. 첫째, SEALLM 및 SEALION과 같은 동남아시아 PLMs에서의 편향을 처음으로 평가하고 확인했습니다. 둘째, masked 및 causal language models에서 개별 단어가 편향 행동에 기여하는 정도를 정량화하는 방법을 제시했습니다. 셋째, 제안한 해석 가능성 접근법을 통해 언어 모델의 편향 평가 결과에 대한 후속 해석 분석을 수행하고 관련된 주제를 식별했습니다.



### MedLogic-AQA: Enhancing Medical Question Answering with Abstractive Models Focusing on Logical Structures (https://arxiv.org/abs/2410.15463)
- **What's New**: 새로운 의료 질문-응답 시스템인 MedLogic-AQA를 제안합니다. 이 시스템은 First Order Logic(FOL) 기반의 규칙을 사용하여 더 정확하고 뉘앙스 있는 답변을 생성합니다. 기존 시스템의 한계를 극복하기 위해 의학적 맥락의 복잡한 논리 구조를 효과적으로 이해합니다.

- **Technical Details**: MedLogic-AQA는 Logic-Understanding(LU) 모델을 훈련시켜 주어진 맥락(Context), 질문(Question), 답변(Answer)에 대한 논리 세 쌍(Logic triples)을 생성합니다. 이 논리 세 쌍은 답변 생성을 할 때 통합되어 논리적 일관성과 맥락적 관련성을 높입니다. 시스템은 LLAMA2 프레임워크를 사용하여 훈련됩니다.

- **Performance Highlights**: 자동화 및 인간 기반 평가에서 MedLogic-AQA는 기존의 강력한 기준선 모델에 비해 우수한 성능을 보였습니다. 이 시스템은 의학적 질문에 대한 답변의 품질과 포괄성을 향상시키며 정보성도 높입니다.



### CROPE: Evaluating In-Context Adaptation of Vision and Language Models to Culture-Specific Concepts (https://arxiv.org/abs/2410.15453)
- **What's New**: CROPE라는 새로운 비주얼 질문 응답 벤치마크가 소개되었습니다. 이 벤치마크는 문화 특정 개념에 대한 지식을 평가하고 문화적 적응 능력을 정량적으로 분석하는 데 초점을 맞추고 있습니다.

- **Technical Details**: CROPE는 매개변수 지식(parametric knowledge)과 맥락적 지식(contextual knowledge)을 구분하여 평가하는 두 가지 유형의 지식을 특징으로 합니다. 이 논문에서는 현대 VLM들이 문화 특정 개념을 인식하고 이러한 개념에 적응할 수 있는지를 탐구하기 위해 다양한 조건에서 실험을 수행했습니다.

- **Performance Highlights**: 최신 VLM 모델들은 문화 특정 개념을 처리할 때 일반 개념보다 성능이 현저히 떨어지는 것으로 나타났습니다. 맥락 지식을 제공받았을 때도 많은 모델이 예상과 달리 성능이 악화되었고, 모델들이 문화적 맥락보다 하드 네거티브 개념을 구분하는 데 어려움을 겪고 있음을 발견했습니다.



### Evaluating Consistencies in LLM responses through a Semantic Clustering of Question Answering (https://arxiv.org/abs/2410.15440)
Comments:
          Accepted to the Trustworthy AI Workshop at IJCAI 2024

- **What's New**: 대형 언어 모델(LLM)의 일관성을 평가하기 위한 새로운 접근 방식을 제안합니다. 기존의 랜덤성(cell sampling)으로 인한 일관성 부족 문제를 해결하고자 합니다.

- **Technical Details**: 이 연구에서는 LLM의 응답이 주어진 질문에 대해 의미적으로 일치하는지를 평가하기 위해 두 가지 주요 접근 방식을 탐구하였습니다: RAG 패턴과 같은 외부 지식을 컨텍스트로 활용하거나 Zero-shot-CoT를 사용하여 LLM의 성능을 개선하는 것입니다. TruthfulQA 데이터셋을 활용하여 LLM의 응답을 평가하고, 의미론적으로 동등한 문장을 클러스터링하여 37개의 카테고리에서 의미적 일관성을 측정합니다.

- **Performance Highlights**: 이 방법론을 통해 LLM의 성능 개선 전후에 대한 정량적 분석을 수행하였으며, 다른 질문 응답 작업에서 이러한 방법들이 LLM 응답 일관성에 미치는 영향을 비교했습니다.



### A Comprehensive Evaluation of Cognitive Biases in LLMs (https://arxiv.org/abs/2410.15413)
- **What's New**: 이 연구에서는 20개의 최신 Large Language Models (LLMs)에서 30가지 인지 편향(cognitive biases)을 평가하기 위한 대규모 테스트를 제시하고 있습니다. 연구의 주요 기여로는 LLMs의 인지 편향을 탐색하기 위한 30,000개의 테스트 데이터셋과 신뢰할 수 있는 테스트 생성을 위한 새로운 일반 목적의 테스트 프레임워크가 포함됩니다.

- **Technical Details**: 제안된 프레임워크는 인지 편향 테스트를 정의, 다양화, 수행하기 위한 체계적이고 일반적인 구조를 제공합니다. 이 프레임워크는 입력과 출력을 처리하는 여러 개체와 기능으로 구성되어 있으며, 사전 훈련된 LLM을 내부적으로 활용합니다. 사용자는 이를 통해 200개의 다양한 의사결정 시나리오에서 30가지 인지 편향을 평가할 수 있는 테스트를 생성할 수 있습니다.

- **Performance Highlights**: 20개의 LLMs에서 모든 테스트된 30가지 인지 편향의 증거가 발견되었으며, 이는 LLMs에서 인지 편향이 광범위하게 존재함을 확인했습니다. 이러한 발견은 기존 연구 결과를 확장하는 것으로, 향후 LLMs의 인지 편향에 대한 연구를 촉진할 수 있는 기반이 됩니다.



### CalibraEval: Calibrating Prediction Distribution to Mitigate Selection Bias in LLMs-as-Judges (https://arxiv.org/abs/2410.15393)
Comments:
          13 pages

- **What's New**: 본 논문에서는 LLMs(as-Judges)에서 발생하는 selection bias(선택 편향)를 완화하기 위한 새로운 기법인 CalibraEval을 제안한다. 이 방법은 레이블이 필요 없는 방식으로 효과적인 자동 평가 체계를 구축하는 데 기여한다.

- **Technical Details**: CalibraEval은 편향 제거 문제를 최적화(optimization) 과제로 재구성하고, 비모수적 순서 보존 알고리즘(NOAP, Non-Parametric Order-Preserving Algorithm)을 통해 해결한다. 이 알고리즘은 예측 분포의 부분 순서 관계를 활용하여 레이블과 정밀한 수학적 함수 모델링에 대한 의존성을 줄인다.

- **Performance Highlights**: 실험 결과, CalibraEval은 기존의 강력한 기준들을 초월하는 성능을 보이며, 다양한 프롬프트 템플릿과 옵션 토큰에서 강건함을 입증하였고, LLM의 자동 평가 프레임워크의 신뢰성을 향상시키는데 기여한다.



### BERTtime Stories: Investigating the Role of Synthetic Story Data in Language pre-training (https://arxiv.org/abs/2410.15365)
- **What's New**: 이 논문은 2nd BabyLM Challenge의 Strict 및 Strict-Small 트랙에 대한 기여를 설명하고 있습니다. 이 과제는 인류 발달에 영감을 받은 데이터 제한 조건 하에서 효율적인 사전 학습을 중심으로 이루어집니다. 저자들은 TinyStories라는 합성 이야기 데이터의 효과를 연구하여, 제한된 데이터로도 양질의 언어 이해를 가능하게 할 수 있음을 보여줍니다.

- **Technical Details**: 저자들은 TinyStories라는 합성 단편 이야기 데이터셋을 기반으로 GPT-Neo 모델을 학습시켜 다양한 양의 데이터에 대해 생성 성능을 평가했습니다. 이 과정에서, LTG-BERT 인코더 모델을 학습시키기 위해 합성된 이야기 데이터와 BabyLM 데이터셋의 일부를 결합한 데이터셋을 사용했습니다. 특정 조건에서 합성된 데이터는 언어 이해에 부정적인 영향을 미치는 경향이 있음을 발견했습니다.

- **Performance Highlights**: 연구 결과, 100M 언어에 미치지 못하는 데이터 조건에서도 GPT-Neo 모델은 높은 품질의 스토리 생성을 할 수 있음을 발견했습니다. 하지만, 합성 데이터가 언어 사전 학습에 미치는 영향은 미미한 또는 부정적인 결과를 보였습니다. 이 결과는 저자들이 향후 더 많은 연구가 필요하다고 강조하는 대목입니다.



### A Survey of Uncertainty Estimation in LLMs: Theory Meets Practic (https://arxiv.org/abs/2410.15326)
Comments:
          9 pages

- **What's New**: 이 논문은 대형 언어 모델(LLMs)에서의 불확실성 추정에 대한 체계적인 분류와 정의를 제시합니다. 불확실성(uncertainty)과 신뢰도(confidence)의 차이를 강조하며, 이론적 관점인 Bayesian inference와 정보 이론(information theory)을 기반으로 다양한 방법을 분류합니다.

- **Technical Details**: 불확실성 추정의 주요 이론인 Bayesian inference를 소개하고, LLMs에 맞게 조정된 앙상블 전략(ensemble strategies)을 논의합니다. 또한 정보 이론을 통해 엔트로피(entropy), 당혹감(perplexity), 상호 정보(mutual information)의 관점에서 LLM의 불확실성을 설명합니다. 이러한 다양한 접근 방식은 LLM에서의 불확실성 추정에 대한 포괄적인 이해를 제공합니다.

- **Performance Highlights**: 이 리뷰는 LLM의 불확실성 추정에 대한 이론적 및 실용적 방법의 통합적 접근을 제안하여, 향후 LLM의 신뢰성과 효과성을 높이기 위한 새로운 불확실성 추정 방법 개발에 기여할 것으로 기대됩니다.



### Causality for Large Language Models (https://arxiv.org/abs/2410.15319)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 훈련 과정에 인과성(causality)을 통합하는 방법을 탐구하며, 기존 상관 관계 기반 접근법을 넘어서는 새로운 패러다임의 필요성을 강조합니다. 또한, 인과적 추론(causal reasoning)을 통해 LLM의 해석 가능성, 신뢰성 및 윤리적 정렬을 향상시키는 여섯 가지 미래 방향을 제시합니다.

- **Technical Details**: 대형 언어 모델은 수십억에서 수조 개의 매개변수(parameters)를 가진 인공지능 모델로, 방대한 데이터셋에서 학습됩니다. 이 모델들은 트랜스포머(transformer) 네트워크를 기반으로 구성되어있으며, 자연어 이해(natural language understanding), 자연어 생성(natural language generation) 및 논리적 문제 해결(logical reasoning) 등의 작업을 수행합니다.

- **Performance Highlights**: LLMs의 현재 한계는 인과 관계를 이해하지 않고, 단순히 상관 관계를 바탕으로 응답을 생성하는 데 있습니다. 이는 모델의 정확성과 신뢰도를 저하시키며, 의료 및 법률과 같은 중요한 분야에서의 잘못된 정보 생성으로 이어질 위험이 있습니다. 인과성을 효과적으로 도입하면 보다 신뢰할 수 있고 윤리적으로 정렬된 AI 시스템을 개발하는 데 도움이 됩니다.



### Ichigo: Mixed-Modal Early-Fusion Realtime Voice Assistan (https://arxiv.org/abs/2410.15316)
- **What's New**: 이 논문은 Ichigo라는 혼합 모달 모델을 소개합니다. Ichigo는 오디오와 텍스트를 함께 처리하여 음성 질문-답변 작업에서 기존의 오픈 소스 음성 언어 모델보다 우수한 성능을 발휘합니다.

- **Technical Details**: Ichigo는 혼합된 음성과 텍스트의 interleaved sequences를 처리하는 모델입니다. 이 모델은 tokenized early-fusion 접근 방식을 활용하여 음성을 이산(tokenized) 토큰으로 양자화하고, 일반적인 transformer 기반 아키텍처를 사용하여 음성 및 텍스트 방식을 모두 처리합니다.

- **Performance Highlights**: Ichigo는 음성 질문-답변 벤치마크에서 최첨단 성능을 보이며, 첫 번째 토큰 생성 지연(latency)은 단 111 ms로, 현행 모델보다 현저히 낮습니다.



### KTCR: Improving Implicit Hate Detection with Knowledge Transfer driven Concept Refinemen (https://arxiv.org/abs/2410.15314)
Comments:
          11 pages, 4 figures, 2 algorithms, 5 tables

- **What's New**: 이 논문에서는 최근 나타나는 혐오 콘텐츠에 대한 기계 학습 모델의 성능을 향상시키기 위해 새로운 Knowledge Transfer-driven Concept Refinement 방법을 제안합니다. 이 방법은 묵시적 혐오 샘플과 관련된 개념을 정제하고 증강하는 접근법입니다.

- **Technical Details**: KTCR(지식 전이 기반 개념 정제) 방법은 새로운 프로토타입 정렬과 개념 손실을 통해 묵시적 혐오 샘플과 관련된 개념을 증류하고 정제합니다. 이 방법은 TCAV(개념 활성화 벡터)를 기반으로 하여 이뤄지며, 모델의 미세한 패턴을 효과적으로 학습하도록 돕습니다.

- **Performance Highlights**: 여러 공개 데이터셋에서 실험한 결과, 개념 정제를 통해 새로운 혐오 패턴을 반영하는 추가 묵시적 샘플을 통합함으로써 모델의 성능이 향상되고 기본 성능을 초과하며, 데이터셋 전반에 걸쳐 일반화 능력을 유지하였습니다.



### LlamaLens: Specialized Multilingual LLM for Analyzing News and Social Media Conten (https://arxiv.org/abs/2410.15308)
Comments:
          LLMs, Multilingual, Language Diversity, Large Language Models, Social Media, News Media, Specialized LLMs, Fact-checking, Media Analysis

- **What's New**: 이 연구는 LlamaLens라는 전문화된 대형 언어 모델(LLM)을 개발하여 다국어 컨텍스트에서 뉴스 및 소셜 미디어 콘텐츠를 분석하는 데 중점을 두고 있습니다. 이는 도메인 특화성과 다국어 처리를 동시에 해결하려는 최초의 시도입니다.

- **Technical Details**: LlamaLens는 아랍어, 영어, 힌디어를 포함한 3개 언어로 52개의 데이터셋을 사용하여 19개의 작업을 다룹니다. 모델은 기존 LLM을 세밀하게 조정하여 도메인 지식을 강화했으며, 학습 중 다양한 데이터 셔플링 기술을 활용했습니다. 실험 결과는 LlamaLens가 16개의 테스트 세트에서 현재의 최첨단(State of the Art, SOTA) 성능을 초과하며, 10개 세트에서는 비슷한 성능을 달성함을 보여줍니다.

- **Performance Highlights**: LlamaLens는 기존의 비세밀 조정 모델에 비해 성능을 현저히 향상시켰으며, 특히 작은 버전의 모델에서도 도메인 및 언어 특화 지식을 획득하는 데 성공했습니다. SOTA와의 비교에서도 개선의 여지가 존재함을 시사합니다.



### Does ChatGPT Have a Poetic Style? (https://arxiv.org/abs/2410.15299)
Comments:
          CHR 2024: Computational Humanities Research Conference

- **What's New**: 이 논문은 OpenAI의 ChatGPT가 생성하는 시(Poetry)에 대한 초기 연구 결과를 제시합니다. 특히 GPT-3.5와 GPT-4 모델을 사용하여 24 가지 서로 다른 시 형식으로 5,700개의 시를 생성하고 이를 분석하여 ChatGPT의 시적 스타일과 능력을 평가합니다.

- **Technical Details**: 연구진은 Poetry Foundation과 Academy of American Poets의 시 데이터셋에서 3,874개의 시/스타일 쌍을 수집하여 ChatGPT가 생성한 시와 비교하였으며, 이 중 40개 주제를 선택하여 다양한 형식의 시를 생성하도록 유도했습니다. 모델은 일반적으로 iambic meter(아이앰빅 미터), quatrains(4행으로 구성된 시), 그리고 특정 단어를 선호하는 경향이 있음을 보였습니다.

- **Performance Highlights**: 연구 결과, GPT-4는 소네트(sonnet, 14행), 빌라넬(villanelle, 19행) 및 세스티나(sestina, 39행)와 같은 특정 형식을 따르며 적절한 길이의 시를 생성하는 데 성공적이었습니다. 그러나 GPT가 생성한 시는 인간 시와 비교할 때 제약이 많고 일관성이 있으며, 일반적으로 운율(rhyme)을 선호하고 특정 어휘를 자주 사용합니다.



### Redefining Proactivity for Information Seeking Dialogu (https://arxiv.org/abs/2410.15297)
- **What's New**: 이 연구는 정보 검색 대화 (ISD) 에이전트의 반응적 행동을 넘어서 사용자와의 지속적인 대화를 생성하는 새로운 주도적 (proactive) 대화를 위한 정의를 제안합니다. 이 정의에서는 초기 질문과 관련된 새로운 정보를 통해 응답의 주도성을 향상시키는 데 중점을 두고 있습니다.

- **Technical Details**: 우리는 총 2,000개의 단일 회화로 구성된 주도적 대화 데이터셋을 구축하고, 응답의 '주도성'을 평가하기 위한 여러 자동화된 메트릭을 도입하였습니다. 이 메트릭은 인간 주석과의 높은 상관관계를 달성하였습니다. 또한, 두 가지 혁신적인 Chain-of-Thought (CoT) 프롬프트인 3단계 CoT 프롬프트와 3-인-1 CoT 프롬프트를 제안하여 제로샷 설정에서 표준 프롬프트보다 최대 90% 더 우수한 성능을 보였습니다.

- **Performance Highlights**: 이 연구는 주도적 응답 생성의 맥락에서 instruction-tuning의 효과성을 입증하였으며, 사용자 상호작용을 지속시키고 대화의 정보성을 개선하는 데 성공했습니다. 또한, 제안된 접근 방식은 여러 차례의 반복 대화 시나리오에서도 강력한 성과를 나타냈습니다.



### Training Language Models to Critique With Multi-agent Feedback (https://arxiv.org/abs/2410.15287)
- **What's New**: 이 논문은 LLM(대형 언어 모델)의 비판 능력을 향상시키기 위해 MultiCritique라는 새로운 데이터 생성 파이프라인을 제안합니다. 이 파이프라인은 다중 에이전트 피드백을 활용하여 SFT(감독 학습 조정) 및 강화 학습(RL) 단계에서 비판 능력을 개선하는 데 중점을 두고 있습니다.

- **Technical Details**: 다중 에이전트 분석 비판을 수집하여 고품질 비판 데이터 세트를 생성합니다. 이 과정에서 사용자 쿼리에 대한 상세한 설명, 맞춤형 평가 기준, 이 기준을 만족시키기 위한 참고 응답을 포함하여 비판을 단순화합니다. Meta-critique 분류 과정을 사용하여 분석 비판을 질적 범주로 자동 분류하고, 선택된 분석 비판과 거부된 분석 비판을 메타 비판 분류를 통해 자연스럽게 쌍을 지어 RL 과정에서 개선된 비판 능력을 유도합니다.

- **Performance Highlights**: MultiCritique 데이터 세트를 기반으로 한 실험 결과는 두 가지 벤치마크에서 기존 비판 데이터 세트에 비해 뛰어난 품질을 입증했습니다. 특히, 제안된 MultiCritiqueDataset으로 조정된 7B 모델은 다른 고급 7B-13B 오픈 소스 모델에 비해 성능이 크게 향상되어 70B LLM 및 GPT-4의 성능에 근접했습니다.



### BRIEF: Bridging Retrieval and Inference for Multi-hop Reasoning via Compression (https://arxiv.org/abs/2410.15277)
Comments:
          Project page: this https URL

- **What's New**: 이번 논문은 RAG(Retrieval-augmented generation) 시스템에서 효과적인 쿼리(기본질문) 인식 멀티홉(multi-hop) 추론을 위한 경량 접근법인 BRIEF(Bridging Retrieval and Inference through Evidence Fusion)를 제안합니다. 이 방법은 검색된 문서를 고도로 밀집된 텍스트 요약으로 압축하여 LLM(대형 언어 모델)과의 통합을 최적화합니다.

- **Technical Details**: BRIEF는 원본 문서에서 독립적인 사실(factoids)을 포함하는 원자(atomic) 명제 표현을 추출하여 합성 요약(synthetic summaries)을 생성하는 합성 데이터(synthetic data)를 구성합니다. 이렇게 구성된 데이터는 멀티홉 추론을 위한 압축 학습을 가능하게 합니다. BRIEF는 여러 LLM을 통해 개방형 질문 응답(open-domain QA) 성능을 향상시킵니다.

- **Performance Highlights**: BRIEF는 HotpotQA에서 최신 기법 대비 2배 더 높은 압축 비율을 달성하였으며, EM(Exact Match) 성능에서 3.00%, F1 점수에서 4.16% 향상된 결과를 보였습니다. 또한, BRIEF는 GPT-3.5보다 더 간결한 요약을 생성하면서도 유사한 QA 성능을 나타냈습니다.



### Back to School: Translation Using Grammar Books (https://arxiv.org/abs/2410.15263)
- **What's New**: 이번 연구에서는 GPT-4의 프롬프트에 문법서(Grammar books)를 포함시켜 저자원이 부족한 언어에서 기계 번역(Machine Translation)의 성능을 향상시키는 방법을 제안합니다. 이 접근 방식은 다양하게 수집된 언어 참조 자료를 활용함으로써 저자원 언어의 번역 시스템을 개선할 수 있음을 보여줍니다.

- **Technical Details**: 연구진은 16개의 다양한 저자원 언어에 대해 문법서, 이중 언어 사전(Bilingual dictionaries), 및 일부 병렬 문장(Parallel sentences)을 포함한 프롬프트를 사용하여 LLMs의 기계 번역 성능을 분석했습니다. 이 과정에서 문법서에서 발췌한 내용을 사용하여 번역의 정확성을 높이고 번역 시스템의 구조를 강화합니다.

- **Performance Highlights**: 이 방법론을 통해 저자원 언어에서의 기계 번역 성능이 향상된 것으로 나타났습니다. 특히, 범위가 넓은 언어 자원을 활용하여 기계 번역 모델이 훈련될 수 있음을 보여주었으며, 기존의 접근 방식에 비해 더 나은 결과를 얻었다고 보고하였습니다.



### Lossless KV Cache Compression to 2% (https://arxiv.org/abs/2410.15252)
- **What's New**: 이 연구에서는 KV 캐시 메모리를 2% 이하로 압축하면서 성능을 유지하는 새로운 아키텍처인 Cross-Layer Latent Attention (CLLA)을 제안합니다. CLLA는 다양한 KV 캐시 압축 기법을 통합하여 일관된 프레임워크 내에서 압축을 수행합니다.

- **Technical Details**: CLLA는 주의 헤드(Attention Head) 및 차원(Dimension) 축소, 레이어 공유(Layer Sharing), 양자화(Quantization) 기술을 포함한 다양한 압축 기법을 결합하여 KV 캐시를 효과적으로 압축합니다. 이 연구는 손실 없는(lossless) 성능을 유지하면서 대량의 GPU 메모리를 절약할 수 있도록 설계되었습니다.

- **Performance Highlights**: CLLA는 대다수의 작업에서 손실 없는 성능을 산출하면서 기존 KV 캐시의 2% 미만만을 사용합니다. 여러 CLLA 변형에 대한 세부 분석을 통해 다양한 조합을 탐색하여 성능을 극대화했습니다.



### On the Diversity of Synthetic Data and its Impact on Training Large Language Models (https://arxiv.org/abs/2410.15226)
- **What's New**: 이 연구는 압도적인 성능을 자랑하는 대형 언어 모델(LLM)의 훈련을 지원하기 위한 합성 데이터의 다양성을 측정할 수 있는 새롭고 독창적인 지표인 'LLM cluster-agent'를 도입합니다. 이를 통해 합성 데이터의 다양성이 LLM의 성능에 미치는 영향을 분석합니다.

- **Technical Details**: LLM cluster-agent는 텍스트 데이터의 다양성을 평가하기 위해 LLM을 활용하여 클러스터링을 수행하는 새로운 접근 방식을 사용합니다. 이 연구는 350M과 1.4B 파라미터를 가진 모델을 사용하여 통제된 실험을 진행하며, LLM cluster score가 LLM의 사전 학습 및 감독하에 미세 조정 성능과 긍정적인 상관관계를 가지고 있다는 것을 보여줍니다.

- **Performance Highlights**: 합성 데이터의 다양성이 LLM의 성능에 미치는 영향이 크며, 특히 사전 학습보다 감독 미세 조정 과정에서 더 강한 영향을 미칩니다. 실험 결과, 다양한 스타일과 대상을 포함한 프롬프트로 생성된 합성 데이터가 LLM의 성능을 크게 향상시키는 것으로 나타났습니다.



### Fine-tuning foundational models to code diagnoses from veterinary health records (https://arxiv.org/abs/2410.15186)
Comments:
          26 pages, 5 figures

- **What's New**: 이 연구는 Colorado State University (CSU) Veterinary Teaching Hospital (VTH)의 모든 7,739개의 SNOMED-CT 진단 코드를 포함하여 대규모의 사전 훈련된 언어 모델 (Large Language Models, LLMs)을 활용하여 동물 진단 코드를 자동화하는 방법을 개선하고자 합니다.

- **Technical Details**: 이 연구는 자연어 처리 (Natural Language Processing, NLP) 기술을 사용하여 246,473개의 수동 코드화된 동물 환자 방문 기록에서 자유 텍스트 노트를 기반으로 10개의 사전 훈련된 LLM을 미세 조정 (fine-tuning) 하여, 신뢰성을 높이고 기존 연구들보다 뛰어난 성과를 보여주었습니다.

- **Performance Highlights**: 이 연구의 결과는 대규모의 레이블링된 데이터를 사용하여 상대적으로 큰 임상 LLM을 미세 조정했을 때 가장 정확한 결과를 얻을 수 있음을 보여주며, 제한된 자원과 비임상 LLM을 사용해도 유사한 결과를 얻을 수 있음을 확인했습니다.



### Uncovering Autoregressive LLM Knowledge of Thematic Fit in Event Representation (https://arxiv.org/abs/2410.15173)
Comments:
          15 pages, 3 figures

- **What's New**: 이 연구에서는 사전 훈련된 자가 회귀 LLMs(generative language models)가 주어진 사건 인수의 주제 적합성을 일관되게 추론할 수 있는지를 평가합니다. 특정한 구문적 구조와 다양한 입력 및 출력 형식을 비교하여 이들 모델의 성능을 분석하였습니다.

- **Technical Details**: 이 연구는 세 가지 축으로 평가되었습니다: 1) 다단계 논리적 추론(chain-of-thought prompting)과 간단한 프롬프트(Simpler Prompting) 비교, 2) 생성된 문장(context)와 원시 투플(<predicate, argument, role>) 비교, 3) 범주형 출력(categorical output)과 숫자형 출력(numeric output) 비교. 주제 적합성 주제(task)는 동사(predicate)와 그 인수(argument) 간의 호환성을 측정하는 작업입니다.

- **Performance Highlights**: 연구 결과, chain-of-thought reasoning이 자명한 의미 역할 레이블이 있는 데이터셋에서 더 효과적이었고, QPT 기반 방법이 모든 테스트 데이터셋에서 새로운 최첨단 성능을 설정했습니다. 제시된 생성된 문장은 특정 설정에서만 유용했으며, 여러 다른 경우에서는 결과를 저하시켰습니다.



### An Electoral Approach to Diversify LLM-based Multi-Agent Collective Decision-Making (https://arxiv.org/abs/2410.15168)
Comments:
          Accepted to EMNLP 2024

- **What's New**: 이 연구는 최근의 LLM 기반 다중 에이전트 협업 시스템을 조사하고 집합적 의사결정(Collective Decision-Making, CDM)에 대한 기존 접근 방식의 한계를 보여줍니다. 특히, dictatorial(독재적) 및 plurality voting(다수결 투표)에 의존하는 경향이 드러났습니다. 이를 바탕으로 다양한 순위 선호 투표 메커니즘을 포함한 electoral CDM 모듈 GEDI를 제안하였습니다.

- **Technical Details**: 연구에서는 사회 선택 이론(Social Choice Theory)의 관점에서 CDM 방법을 분석하였으며, GEDI 모듈을 통해 다양한 CDM 메커니즘을 제안합니다. 실험은 세 가지 다중 선택 질문-답변(MCQA) 기준에서 진행되었으며, 다양한 크기와 구조를 가진 모델을 사용하여 LLM의 집합적 의사결정 능력을 평가했습니다.

- **Performance Highlights**: 결과적으로, CDM 방법을 적용했을 때 단일 에이전트의 의사결정보다 일반적으로 더 나은 결과를 얻었지만 계산 비용은 증가했습니다. 또한 몇 번의 에이전트(최소 세 명)만으로도 효과적인 결과를 도출할 수 있었고, CDM 방법은 불확실한 에이전트에 대한 견딜 수 있는 강건성 및 다양한 주제 도메인에서의 영향을 증명했습니다.



### Evaluating Deep Unlearning in Large Language Models (https://arxiv.org/abs/2410.15153)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)에서의 기계 비학습(Machine Unlearning)의 중요성을 강조하며, 표면적으로만 비학습이 성립하는 기존 방법의 한계를 검사하고 심층 비학습(deep unlearning) 개념을 제안합니다.

- **Technical Details**: 연구자들은 심층 비학습을 위해 에듀-관계(EDU-RELAT)라는 합성 데이터셋을 구성하고, 이 데이터셋을 사용하여 4개의 LLM 모델(예: Phi-1.5, GPT2-XL 등)에서 4가지 비학습 방법(Gradient Ascent, Negative Preference Optimization 등)으론 성능을 평가했습니다. 비학습의 성공 여부는 recall과 accuracy라는 두 가지 지표로 측정됩니다.

- **Performance Highlights**: 연구 결과, 심층 비학습을 수행하는 동안 목표 사실을 제대로 비학습하지 못하거나 20% 이상의 관련 없는 사실들이 유실되는 것이 발견되었습니다. 이는 오늘날의 기계 비학습 방법들이 LLMs에서 사실을 제대로 비학습하지 못하고 있음을 보여줍니다.



### Less is More: Parameter-Efficient Selection of Intermediate Tasks for Transfer Learning (https://arxiv.org/abs/2410.15148)
Comments:
          EMNLP 2024 Main Conference

- **What's New**: 본 논문은 중간 작업 전이 학습(intermediate task transfer learning)을 개선하기 위한 Embedding Space Maps (ESMs)라는 경량 신경망 모델을 제안합니다. ESM은 언어 모델을 파인튜닝한 결과를 근사화하여 대용량 소스 풀(source pools)에서도 효율적인 소스 선택을 가능하게 합니다.

- **Technical Details**: ESMs는 서로 다른 임베딩 공간의 선형 변환으로, LogME 방법과 결합하여 높은 선택 성능을 유지하면서도 실행 시간과 디스크 공간 사용량을 각각 10배, 278배 줄이는 효과를 보여줍니다. 본 연구는 1,500개 이상의 소스 데이터셋과 여러 작업 유형 및 언어를 포함하여 12,000개의 소스-타겟 쌍에 대한 전이 가능성을 연구하였습니다.

- **Performance Highlights**: ESM-LogME는 기존 방법들과 비교하여 최상의 성능을 보이며, 실세계 시나리오에서도 적용 가능하다는 것을 입증했습니다. 소스 코드는 Apache 2 라이센스 하에 공개되었고, 연구자 및 실무자들이 ESM을 쉽게 공유하고 찾을 수 있는 기능도 제공합니다.



### A survey of neural-network-based methods utilising comparable data for finding translation equivalents (https://arxiv.org/abs/2410.15144)
- **What's New**: 이 논문은 자연어 처리(NLP) 분야에서 중요한 두 개의 사전 구성 요소인 번역 동의어를 자동으로 유도하는 여러 가지 방법을 제시합니다. 특히, 비교 가능한 데이터를 사용하는 신경망 기반(neural network-based) 방법에 초점을 맞추고 lexicography의 관점에서 분석합니다.

- **Technical Details**: 자동 번역 동의어 추출 방법은 통계 기반(statistical-based) 및 신경망 기반(neural network-based) 방법으로 나뉘며, 본 연구에서는 주로 신경망 기반 방법에 대해 다룹니다. 중요한 자연어 공학(aspects of natural language engineering) 측면을 정의하고, 이러한 측면에 대한 알고리즘의 처리를 조사합니다. 또한, 특정 언어쌍과 조합에서 비교 가능한 데이터를 사용하는 방법론의 장점과 단점을 설명합니다.

- **Performance Highlights**: 조사 결과, 수동 편집된 사전 데이터를 사용할 때 모델 성능이 20% 이상 떨어지며, 이는 연구 문헌에서 원래 주장된 모델 성능과 다릅니다. 따라서, 실질적인 성과를 위해 lexicography의 통찰력을 NLP 모델에 통합하는 것이 필수적임을 강조합니다.



### CAST: Corpus-Aware Self-similarity Enhanced Topic modelling (https://arxiv.org/abs/2410.15136)
- **What's New**: 본 논문에서는 기존의 주제 모델링 방법의 한계를 극복하기 위해 새로운 주제 모델인 CAST(Corpus-Aware Self-similarity Enhanced Topic modelling)를 소개합니다. 이 모델은 후보 중심 단어(centroids)의 임베딩을 데이터셋에 맞게 컨텍스트화(contextualize)하고, 자기 유사성을 기반으로 의미 없는 토큰을 필터링하는 방법을 사용합니다.

- **Technical Details**: CAST는 두 가지 주요 모듈을 도입합니다: 1) 데이터셋에 컨텍스트화된 단어 임베딩, 2) 자기 유사성 점수를 사용하여 기능적 단어를 필터링하고 의미 있는 후보 주제 단어를 구성합니다. 이를 통해 문서의 컨텍스트에 따라 후보 주제 단어를 인코딩합니다.

- **Performance Highlights**: 실험 결과, CAST는 뉴스 벤치마크 데이터셋과 트위터 데이터셋에서 일관되며 다양성 있는 주제를 생성하고, 노이즈가 많은 데이터 처리에서 강력한 기준선보다 우수한 성능을 보였습니다.



### Augmenting the Veracity and Explanations of Complex Fact Checking via Iterative Self-Revision with LLMs (https://arxiv.org/abs/2410.15135)
- **What's New**: 이번 연구에서는 설명 생성을 통한 해석 가능한 결과를 도출하는 것이 사실 확인(fact verification)보다 더 중요하다는 점을 강조하며, 중국어 시나리오에서의 새로운 복잡한 사실 확인 데이터셋인 CHEF-EG와 TrendFact를 제안합니다. 이를 통해 여러 언어에서의 설명 생성의 가능성을 모색했습니다.

- **Technical Details**: FactISR( Augmenting Fact-Checking via Iterative Self-Revision)라는 통합 프레임워크를 제안하여, 대형 언어 모델(large language models, LLMs)의 능력을 활용하여 진위와 설명 간의 상호 피드백을 수행합니다. FactISR는 하나의 모델을 사용하여 사실 확인과 설명 생성을 모두 처리하고, 자기 수정(self-revision) 메커니즘을 통해 진위 레이블과 설명 텍스트 간의 일관성을 더욱 높입니다.

- **Performance Highlights**: 광범위한 실험을 실시한 결과, 제안된 데이터셋과 방법의 효과성이 입증되었습니다. 이를 통해 설명의 질과 사실 확인의 정확성을 높일 수 있음을 알 수 있습니다.



### MELT: Materials-aware Continued Pre-training for Language Model Adaptation to Materials Scienc (https://arxiv.org/abs/2410.15126)
Comments:
          Accepted at EMNLP 2024 (Findings)

- **What's New**: 이번 논문은 MELT (MatEriaLs-aware continued pre-Training)라는 새로운 지속적인 사전 학습 방법을 소개합니다. MELT는 재료 과학에 특화된 사전 훈련된 언어 모델(PLMs)을 효율적으로 조정하기 위해 설계되었습니다. 기존의 방법들이 도메인 특수 코퍼스 구축에 초점을 맞추었다면, MELT는 코퍼스와 훈련 전략 모두를 포괄적으로 고려합니다.

- **Technical Details**: MELT의 핵심 전략은 재료 과학 코퍼스에서 소재(entity)들을 추출하고, 이를 통해 PLMs에 지식을 전이하는 것입니다. 이를 위해, 기초적인 재료 개체에서 시작하여 점진적으로 더 전문화된 지식으로 나아가는 커리큘럼을 도입합니다. 세부적으로, 필수적인 화학 개체 및 개념 구성을 위해 의미적 그래프를 구성하여 도메인 지식을 확장합니다.

- **Performance Highlights**: MELT는 다양한 벤치마크를 통해 종합적인 평가 결과, 기존의 계속된 사전 학습 방법들에 비해 뛰어난 성능을 나타냈습니다. MELT는 PLMs가 기존의 방법들보다 재료 개체를 효과적으로 표현할 수 있도록 지원하며, 재료 과학 전반에 걸쳐 넓은 적용 가능성을 보였습니다.



### Coarse-to-Fine Highlighting: Reducing Knowledge Hallucination in Large Language Models (https://arxiv.org/abs/2410.15116)
- **What's New**: 이 논문에서는 COFT라는 새로운 방법론을 제안합니다. COFT는 LLMs에서 발생하는 지식 환각(knowledge hallucination) 문제를 해결하기 위해 고안된 COarse-to-Fine highlighTing 메서드입니다. 이 방법은 긴 문맥 속에서 핵심 텍스트에 집중하여 해당 문제를 개선합니다.

- **Technical Details**: COFT는 세 가지 구성요소로 구성됩니다: 
1. 	extit{recaller}: 외부 지식 그래프(knowledge graph)를 사용하여 주어진 문맥에서 잠재적인 핵심 엔티티를 추출합니다. 
2. 	extit{scorer}: 각 엔티티의 문맥적 가중치를 계산하여 그 중요성을 측정합니다. 
3. 	extit{selector}: 동적 임계값 알고리즘을 통해 높은 문맥적 가중치를 가진 엔티티를 선택하고, 이를 기반으로 문단, 문장, 또는 단어를 강조합니다.

- **Performance Highlights**: COFT는 지식 환각 벤치마크에서 F1 스코어에서 평균 32.1% 개선성을 보여줍니다. 또한, 이는 독해(reading comprehension)와 질문 응답(question answering) 등 다양한 장기 문서 작업에서도 각각 평균 4.6% 및 최대 10.5% 개선 효과를 보입니다.



### Toward Robust RALMs: Revealing the Impact of Imperfect Retrieval on Retrieval-Augmented Language Models (https://arxiv.org/abs/2410.15107)
Comments:
          Accepted for publication in Transactions of the Association for Computational Linguistics (TACL)

- **What's New**: 이번 연구에서는 Retrieval Augmented Language Models (RALMs)의 불완전한 정보에 대한 취약성을 체계적으로 분석합니다. 특히 unanswerable, adversarial, conflicting 세 가지 일반적인 시나리오에서 RALMs가 문제 문서 세트를 식별하고 처리하는 능력을 평가합니다.

- **Technical Details**: 새로운 적대적 공격 방법인 Generative model-based ADVersarial attack (GenADV)와 Robustness under Additional Document (RAD)라는 새로운 지표를 제안하여 RALMs의 적대적 강건성을 측정합니다. 연구는 RALMs가 의미의 부재 또는 문서 세트의 모순을 인식하지 못하며, 이로 인해 자주 hallucination 현상이 발생한다는 것을 보여줍니다.

- **Performance Highlights**: 연구 결과 RALMs는 적대적 정보에 매우 취약하며, unanswerable 상황에서 특히 저해되는 경향이 있습니다. 적대적 문서와 unanswerable 상황이 겹칠 때, 모델의 성능이 더욱 악화됩니다.



### Weakly-supervised diagnosis identification from Italian discharge letters (https://arxiv.org/abs/2410.15051)
Comments:
          39 pages, 4 figures

- **What's New**: 이번 연구에서는 이탈리아 퇴원 소견서에서 질병을 인식하기 위한 새로운 약한 지도 학습(weakly-supervised) 파이프라인을 제안합니다. 기존의 감독 학습(supervised learning) 방식 대신, 수작업 주석이 필요 없는 방법을 개발했습니다.

- **Technical Details**: 파이프라인은 이탈리아 Umberto 모델을 기반으로 하여, 진료 관련 문장을 추출하고 이 두 단계의 클러스터링을 통해 질병에 매핑된 약한 레이블(weak labels)을 선택합니다. 이러한 과정을 통해 BERT 기반 모델을 훈련하여 특정 질병을 감지합니다.

- **Performance Highlights**: 브론키올리티스 식별 관련 사례 연구에서 33,176개의 퇴원 소견서를 분석하여 AUC 77.7 %와 F1-Score 75.1 %를 기록했습니다. 이는 기존 비지도 학습(non-supervised methods)보다 개선된 결과이며, 완전 감독 방법(fully supervised methods) 대비 손실이 적고, 클러스터 선택의 변화에도 강건한 성능을 보였습니다.



### Are LLMs Good Zero-Shot Fallacy Classifiers? (https://arxiv.org/abs/2410.15050)
Comments:
          Accepted to EMNLP2024 main conference

- **What's New**: 본 논문은 Zero-shot 기법을 활용한 오류(fallacy) 분류를 탐구하며, 기존의 데이터에 대한 학습 없이 대형 언어 모델(Large Language Models, LLMs)이 효과적으로 오류를 판별할 수 있는 가능성을 보여준다. 특히, 단일 및 다중 라운드 프롬프트(prompting) 방식을 통해 LLMs의 오류 인식 능력을 극대화하는 방법을 제안한다.

- **Technical Details**: Zero-shot fallacy classification을 위한 두 가지 프롬프트 기법이 제안되었다: Zero-shot Single-Round Prompting과 Zero-shot Multi-Round Prompting. 후자는 정의 생성, 일반 오류 분석, 전제 및 결론 분석, Chain-of-Thought 방법을 포함하여 LLMs가 오류 관련 지식을 보다 효과적으로 끌어낼 수 있도록 돕는다.

- **Performance Highlights**: 단일 라운드 프롬프트 적용 시 LLMs는 SOTA(full-shot fine-tuned) T5 모델과 비교했을 때 유사한 성능을 보여주며, 특히 OOD(out-of-distribution) 환경에서 우수한 성능을 발휘한다. 다중 라운드 프롬프트는 더욱 개선된 성능을 보이며, 소형 LLM들에 특히 효과적이다.



### mHumanEval -- A Multilingual Benchmark to Evaluate Large Language Models for Code Generation (https://arxiv.org/abs/2410.15037)
Comments:
          30 Pages

- **What's New**: 최근 큰 언어 모델(LLMs)의 발전으로 자연어 프롬프트에서 코드 생성이 크게 향상되었습니다. 그러나 기존의 HumanEval Benchmark와 같은 코드 LLM 벤치마크는 작업 다양성(Task Diversity), 테스트 범위(Test Coverage), 언어적 범위(Linguistic Scope)에서 주요한 한계가 있습니다. 이 논문에서는 이러한 한계를 극복하기 위해 200개 이상의 자연어 프롬프트를 지원하는 mHumanEval을 소개합니다.

- **Technical Details**: mHumanEval은 기계 번역 방법(Machine Translation Methods)을 사용하여 개발되었으며, 품질 보증 프로세스(Quality Assurance Process)를 통합합니다. 또한, 15개 다양한 자연어(Natural Languages)에 대한 전문가 수준의 인간 번역(Human Translations)을 제공합니다. 이 연구는 최신 코드 LLMs의 다국적(multilingual) 코드 생성 능력을 분석합니다.

- **Performance Highlights**: 이 논문은 코드 생성의 언어 간(cross-lingual) 변화를 밝히면서 현재의 최신(code LLMs) 언어 모델들이 갖는 성능의 통찰(insights)을 제공합니다.



### Improving General Text Embedding Model: Tackling Task Conflict and Data Imbalance through Model Merging (https://arxiv.org/abs/2410.15035)
Comments:
          working in progress

- **What's New**: 이 연구에서는 Multi-task joint training의 두 가지 주요 문제인 Task Conflict와 Data Imbalance를 식별하고, 모델 병합(model merging) 기법을 활용하여 이러한 문제를 해결하는 접근 방식을 제안합니다. 특히, Self Positioning이라는 새로운 방법론을 통해 최적의 모델 조합을 효율적으로 탐색하는 방법을 소개합니다.

- **Technical Details**: Self Positioning은 stochastic gradient descent를 사용하여 task vectors의 보간(interpolation) 공간 내에서 최적의 모델 조합을 찾는 기법으로, 이는 서로 독립적으로 훈련된 모델들을 결합하여 gradient conflicts와 data imbalance 문제를 완화합니다. 또한, 다수의 모델 병합 전략을 비교하여 이를 통해 성능 향상을 확인합니다.

- **Performance Highlights**: Self Positioning 방법을 적용한 결과, MTEB 데이터셋에서 멀티 태스크 성능이 0.7점 향상되었으며, 전통적인 resampling 방법에 비해 계산 비용이 상당히 줄어드는 효과가 있음을 입증하였습니다.



### Enhancing Multimodal Sentiment Analysis for Missing Modality through Self-Distillation and Unified Modality Cross-Attention (https://arxiv.org/abs/2410.15029)
- **What's New**: 이번 연구에서는 텍스트 모달리티의 부재에도 불구하고 멀티모달 감정 분석을 효과적으로 수행할 수 있는 강력한 모델인 Double-Flow Self-Distillation Framework를 개발했습니다. 이 프레임워크는 Unified Modality Cross-Attention (UMCA)와 Modality Imagination Autoencoder (MIA)를 포함하고 있습니다.

- **Technical Details**: 제안된 프레임워크는 LLM 기반 모델을 사용하여 텍스트 모달리티가 없는 경우 오디오 모달리티에서 텍스트 표현을 시뮬레이션합니다. 또한, MIA 모듈을 통해 다른 두 모달리티의 정보를 보완하여 시뮬레이션된 텍스트 표현이 실제 텍스트 표현과 유사하도록 합니다. Rank-N Contrast (RNC) 손실 함수를 도입하여 시뮬레이션된 표현과 실제 표현을 정렬합니다.

- **Performance Highlights**: CMU-MOSEI 데이터셋에서 테스트한 결과, 제안된 모델은 MAE에서 뛰어난 성능을 보였으며, 텍스트 모달리티가 결여된 상황에서 다른 모델들보다 현저하게 우수한 성능을 발휘했습니다.



### Theoretical Aspects of Bias and Diversity in Minimum Bayes Risk Decoding (https://arxiv.org/abs/2410.15021)
- **What's New**: 이 논문에서는 Minimum Bayes Risk (MBR) 디코딩의 새로운 이론적 해석을 제안하며, 편향(bias)과 다양성(diversity) 분해 관점에서 해당 기법의 개선 효과를 분석합니다.

- **Technical Details**: MBR 디코딩은 모델이 생성한 의사 참고(pseudo-reference)와 자동 평가 메트릭을 통해 텍스트 품질을 추정합니다. 연구에서는 편향과 다양성이라는 두 가지 주요 인자로 추정된 품질의 오류를 분해하고, 이를 통해 MAMBR (Metric-augmented MBR)이라는 새로운 접근법을 제안합니다.

- **Performance Highlights**: 여러 NLP 과제에 대한 실험 결과, 편향-다양성 분해의 구성 요소가 성능과 잘 연관되어 있으며, MAMBR은 유틸리티 함수의 행동을 조정함으로써 텍스트 생성 품질을 향상시키는 것으로 나타났습니다.



### A Survey of Ontology Expansion for Conversational Understanding (https://arxiv.org/abs/2410.15019)
Comments:
          Accepted by EMNLP 2024, code and data are available at this https URL: this https URL

- **What's New**: 본 논문은 대화형 인공지능의 영역에서 Ontology Expansion(OnExp)의 중요성을 강조하며, 기존의 정적 온톨로지의 제한성을 극복하기 위한 최신 기술을 종합적으로 검토합니다.

- **Technical Details**: OnExp는 대화형 토대의 동적 확대를 포함하며, 사용자 발화에서 알려진 및 새로운 온톨로지 항목을 인식하여 업데이트됩니다. 논문에서는 New Intent Discovery(NID), New Slot-Value Discovery(NSVD), Joint OnExp의 세 가지 영역으로 기존 연구를 분류합니다.

- **Performance Highlights**: OnExp 접근법은 대화형 에이전트의 정책 구현 및 의사 결정 능력을 개선하여 사용자 만족도 및 서비스 효율성을 높이는 잠재력을 보입니다. 이 연구는 OnExp 분야의 미래 연구 방향과 도전을 조명하며, 새로운 연구 기회를 촉진하고자 합니다.



### DM-Codec: Distilling Multimodal Representations for Speech Tokenization (https://arxiv.org/abs/2410.15017)
- **What's New**: 본 논문에서는 DM-Codec이라고 불리는 새로운 음성 토크나이저를 제안하며, 이 모델이 정밀한 음성 표현을 위해 맥락적 정보(contextual information)를 통합한다는 점이 특징입니다.

- **Technical Details**: DM-Codec은 Residual Vector Quantizer (RVQ)를 포함한 인코더-디코더 아키텍처를 채택하고, 언어 모델(LM)과 자기 지도 학습(self-supervised learning) 음성 모델(SM)을 사용하여 다중 모드 표현(모음, 의미, 맥락)을 효과적으로 추출합니다. 이 과정은 음성 입력을 텍스트로 전사한 후, LM을 통해 얻은 맥락적 표현을 결합하여 진행됩니다.

- **Performance Highlights**: DM-Codec은 LibriSpeech 데이터셋에서 기존 최첨단 음성 토크나이저 모델들에 비해 Word Error Rate (WER)를 최대 13.46%까지 감소시키고, Word Information Lost (WIL)는 9.82% 줄이며, 음성 품질은 5.84% 및 이해도는 1.85% 개선되었습니다.



### Transit Pulse: Utilizing Social Media as a Source for Customer Feedback and Information Extraction with Large Language Mod (https://arxiv.org/abs/2410.15016)
Comments:
          17 pages, 21 figures

- **What's New**: 본 논문은 사회적 미디어 데이터를 활용하여 대중 교통 사용자 피드백을 분석하기 위한 최신 방법론을 제안합니다. 기존의 방법들은 사전 정의된 주제 라벨에 의존했으나, 본 연구에서는 LLM(Large Language Model)을 활용하여 보다 포괄적인 인사이트를 제공합니다.

- **Technical Details**: 제안된 방법은 Llama 3라는 LLM을 사용하여 사회적 미디어의 대중 교통 관련 정보, 감정 및 풍자 감지, 비정상적인 시스템 문제 식별, 그리고 위치 데이터를 분석합니다. 정보 추출 파이프라인에 RAG(Retrieval-Augmented Generation) 접근 방식을 통합하여 외부 지식을 모델에 결합합니다.

- **Performance Highlights**: 전통적인 NLP 접근 방식과 비교하여 LLM을 활용한 이 방법은 사용자 트윗 데이터에 대한 분석 성능에서 유망한 결과를 보여주었으며, 대중 교통 기관의 대응 능력을 개선하고 실질적인 인사이트를 제공합니다.



### CAP: Data Contamination Detection via Consistency Amplification (https://arxiv.org/abs/2410.15005)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)의 데이터 오염(detection) 문제를 해결하기 위해, Consistency Amplification 기반 데이터 오염 탐지 프레임워크(CAP)를 제안합니다. CAP는 성능 일관성 비율(Performance Consistency Ratio, PCR)을 도입하여 데이터셋 누출을 측정합니다. 이는 도메인 특정 모델의 오염 탐지에 중요한 역할을 합니다.

- **Technical Details**: CAP는 LLM 일관성의 개념을 활용하여 PCR을 계산하고, 훈련 세트와 테스트 세트 간의 PCR을 비교합니다. CAP는 다양한 벤치마크에 적용 가능하며, 화이트박스(white-box) 및 블랙박스(black-box) 모델 모두에 대해 작동합니다.

- **Performance Highlights**: CAP는 7개의 LLM과 4개의 도메인 특정 벤치마크에서 실험을 통해 그 효과를 검증하였으며, 일반 목적 모델에서의 오염이 도메인 특정 모델로 전이되는 현상 및 다양한 데이터 소스에서의 오염 가능성을 발견하였습니다.



### Subversive Characters and Stereotyping Readers: Characterizing Queer Relationalities with Dialogue-Based Relation Extraction (https://arxiv.org/abs/2410.14978)
Comments:
          CHR 2024: Computational Humanities Research Conference

- **What's New**: 이 논문은 TV 시리즈에서 두 캐릭터 간의 사회적 관계를 예측하고, 이를 통해 전통적인 묘사와의 불일치를 분석하는 새로운 작업인 stereotypic relationship extraction을 소개합니다.

- **Technical Details**: 이 연구는 대화에서 나타나는 관계의 유형을 식별하기 위해 NLP의 관계 추출(method of relation extraction) 방식을 따릅니다. 연구의 주요 목표는 마치 '형식적인' 관계가 어떻게 왜곡될 수 있는지를 모형화하고 예측하는 것으로, 캐릭터 간의 대화에서 특정 사회적 정체성을 드러내는 대화 분절을 분석합니다.

- **Performance Highlights**: 논문에서는 Big Bang Theory, Frasier, Gilmore Girls의 사례 연구를 통해 queer 관계성을 정량적으로 분석하고, 전통적인 TV 표현과의 차이를 정량적으로 포착하여 풍부한 텍스트 분석을 위한 방법론적 개입을 제안합니다.



### ChronoFact: Timeline-based Temporal Fact Verification (https://arxiv.org/abs/2410.14964)
- **What's New**: 이 연구에서는 자동화된 사실 검증에서 시간 관련 정보(temporal information)를 고려한 새로운 접근 방식을 제안합니다. ChronoFact라는 프레임워크를 통해 복잡한 시간 기반 주장(verifiable claims)의 검증을 수행하며, 사건(event) 및 증거의 시간적 관계를 분석하여 정확도를 높이고자 합니다.

- **Technical Details**: 이 방법은 두 단계로 나뉘어집니다: 1. Evidence Retrieval (증거 수집) 단계에서는 Wikipedia와 같은 신뢰할 수 있는 출처에서 관련 증거를 수집하고, Semantic Role Labelling (의미역 부착)을 이용하여 주장과 증거에서 사건을 추출합니다. 2. Verification (검증) 단계에서는 추출된 사건의 진위 여부를 평가하고, 결과를 SUPPORTED (지원됨), REFUTED (반박됨), NOT ENOUGH INFO (정보 부족)으로 분류합니다. 결과적으로, ChronoFact는 사건 간의 시간적 관계를 다층 주의(attention) 모델로 분석합니다.

- **Performance Highlights**: 실험 결과, 제안된 ChronoFact 프레임워크는 시간 주장 검증의 정확성을 유의미하게 향상시켜, 자동화된 사실 검증 분야의 최신 기술 수준(state-of-the-art)을 발전시키는 데 기여합니다.



### SemiHVision: Enhancing Medical Multimodal Models with a Semi-Human Annotated Dataset and Fine-Tuned Instruction Generation (https://arxiv.org/abs/2410.14948)
- **What's New**: 최근 의료 분야에서 Multimodal Large Language Models (MLLMs)의 한계점을 극복하기 위해 새로운 데이터셋 SemiHVision과 평가 벤치마크 JAMA Clinical Challenge를 도입했습니다. 이러한 접근법은 의료 진단 효능을 향상시키기 위한 것입니다.

- **Technical Details**: SemiHVision 데이터셋은 사람의 주석과 자동 증강 기법을 결합하여 의료 지식 표현 및 진단 추론을 개선합니다. PMC-Cambrian-8B-AN 모델은 2400 H100 GPU 시간을 활용하여 훈련되었으며, HuatuoGPT-Vision-34B와 Claude3-Opus의 성능을 초과했습니다. JAMA Clinical Challenge는 진단 추론을 평가하기 위해 설계된 새로운 벤치마크입니다.

- **Performance Highlights**: PMC-Cambrian-AN 모델은 JAMA Clinical Challenge 벤치마크에서 현재 최고의 성능을 기록하며, HuatuoGPT-Vision-34B 및 Claude3-Opus를 능가하는 진단 추론 능력을 보여줍니다.



### From Test-Taking to Test-Making: Examining LLM Authoring of Commonsense Assessment Items (https://arxiv.org/abs/2410.14897)
Comments:
          Accepted at Findings of EMNLP 2024

- **What's New**: 이번 연구에서는 LLM(대형 언어 모델)이 일반 상식 추론을 평가하기 위한 테스트 항목을 생성할 수 있는지를 조사합니다. LLM이 COPA(Choice of Plausible Alternatives) 벤치마크 스타일의 항목을 작성하도록 유도한 후, 이를 인간 평가자 및 LLM 자신이 응답한 결과를 분석합니다.

- **Technical Details**: 저자들은 COPA 벤치마크를 기준으로 LLM의 항목 생성 능력을 평가하기 위해, GEN-COPA 항목의 질이 오리지널 COPA 항목의 저자 기준에 얼마나 부합하는지를 분석합니다. 선택한 LLM 모델에는 bloom, falcon, llama2, mistral, mpt, phi 모델이 포함되며, 각각의 성능을 비교합니다.

- **Performance Highlights**: 연구 결과, 오리지널 COPA 벤치마크에서 좋은 성과를 낸 LLM이 자가 생성한 Gen-COPA 항목에서도 높은 정확도를 기록하는 경향이 있음을 발견했습니다. 이는 LLM의 저작 능력이 질문 응답 능력과 밀접하게 연결되어 있음을 시사합니다.



### Which LLMs are Difficult to Detect? A Detailed Analysis of Potential Factors Contributing to Difficulties in LLM Text Detection (https://arxiv.org/abs/2410.14875)
Comments:
          Accepted at NeurIPS 2024 - Safe Generative AI Workshop

- **What's New**: 이 연구는 LLM(대형 언어 모델) 생성 텍스트(AIG 텍스트)의 감지 성능이 도메인에 따라 어떻게 달라지는지를 분석합니다. 특히 OpenAI의 LLM이 생성한 텍스트를 감지하는 데 매우 어렵다는 결과를 도출했습니다.

- **Technical Details**: 이 연구에서는 Deepfake Text, Rewritten Ivy Panda(RIP) 데이터셋을 사용하여 인간 작성 텍스트와 AI 생성 텍스트를 분류하는 모델을 훈련했습니다. LibAUC 라이브러리를 활용해 불균형 데이터셋을 처리하고, DistilRoBERTa 모델을 사용하여 92개의 분류기를 훈련시켰습니다. AUC(Area Under the Curve) 메트릭 최적화를 통해 성능을 평가했습니다.

- **Performance Highlights**: 과학적 작문 분야에서는 AI 생성 텍스트 감지가 상대적으로 어려우며, OpenAI LLM의 AIG 텍스트는 모든 분류기가 상당히 높은 어려움을 보였습니다. 글쓰기 도메인별로 분류기 성능이 달라지는 모습을 관찰할 수 있었습니다.



### DFlow: Diverse Dialogue Flow Simulation with Large Language Models (https://arxiv.org/abs/2410.14853)
Comments:
          16 pages

- **What's New**: 이 논문에서는 기존의 데이터 증강 방법들이 대화 수준에서의 작업 로직 다양성에 대한 중요성을 간과하고 있음을 지적하고, 이를 해결하기 위한 새로운 데이터 증강 방법을 제안합니다.

- **Technical Details**: LLMs(대형 언어 모델)를 활용하여 의사 결정 트리 구조의 작업 계획을 생성하며, 이를 바탕으로 생성된 대화 흐름을 통해 다수의 대화를 제작합니다. 이 방법은 각 작업 흐름을 따라가며 다단계 대화를 생성하는 데 중점을 두게 됩니다. 실험을 통해 3,886개의 대화 흐름으로 구성된 작업 지향 대화 데이터 세트를 생성하였습니다.

- **Performance Highlights**: DFlow 데이터셋에 대해 세밀한 경험적 실험을 수행한 결과, 이 데이터셋으로 미세 조정된 7B 언어 모델이 GPT-4와 같은 강력한 LLM보다 우수한 성과를 보였습니다. 이는 작업 로직과 제약 조건을 잘 따르는 다채로운 다단계 대화를 합성할 수 있는 방법론의 유효성을 입증합니다.



### SPRIG: Improving Large Language Model Performance by System Prompt Optimization (https://arxiv.org/abs/2410.14826)
- **What's New**: 이번 연구에서는 시스템 프롬프트(System Prompt)를 최적화하기 위한 새로운 방법인 SPRIG를 제안합니다. 이는 유전 알고리즘(genetic algorithm)에 기반하여 여러 구성 요소에서 프롬프트를 반복적으로 생성하여 모델의 성능을 극대화하는 방식입니다.

- **Technical Details**: SPRIG는 시스템 프롬프트의 내용을 대체 가능한 300개의 프롬프트 구성 요소로부터 구성됩니다. 이 구성 요소들은 좋음 속성(good property), 역할(role), 스타일(style), 감정(emotion) 등 9가지 범주로 나뉘어 있습니다. 최적화 과정은 비선형(beam search) 기반의 편집(edit-based) 방식을 통해 수행됩니다.

- **Performance Highlights**: SPRIG로 최적화된 시스템 프롬프트는 각 개별 작업에 최적화된 작업 프롬프트(task prompt)와 동등한 성능을 보여주며, 두 종류의 최적화를 결합했을 때 더욱 향상된 성능을 나타냅니다. 또, SPRIG는 다양한 모델 패밀리(model families)와 언어(language)에 대해 일반화되는 효과도 검증되었습니다.



### A Complexity-Based Theory of Compositionality (https://arxiv.org/abs/2410.14817)
- **What's New**: 이 논문은 AI와 인지 과학에서 컴포지셔널 리프레젠테이션(compositional representation)의 수학적 정의인 '표현적 컴포지셔널리티(representational compositionality)'를 제안합니다.

- **Technical Details**: 표현적 컴포지셔널리티는 표현력이 뛰어난 조합적 리프레젠테이션을 규명하며, 이를 수치적으로 측정할 수 있는 기준으로 설정합니다. 이 정의는 알고리즘적 정보 이론(algorithmic information theory)에 기반하여, 기호적 부분의 간단한 함수로 재설명 가능해야 한다고 주장합니다.

- **Performance Highlights**: 실험을 통해 이 새로운 정의가 기존 AI 및 인지 과학 문헌에서의 다양한 직관들을 통합한다는 것을 입증하고, 이론적으로 해결하기 어려운 문제임에도 불구하고 표준 딥러닝 도구를 사용하여 쉽게 추정할 수 있음을 보여줍니다.



### Adapting Multilingual LLMs to Low-Resource Languages using Continued Pre-training and Synthetic Corpus (https://arxiv.org/abs/2410.14815)
- **What's New**: 이번 연구에서는 다국어 LLM (Large Language Models)의 지속적인 사전 학습(Pre-training)이 저자원 언어( low-resource languages)에 대한 성능 개선에 중요하다는 점을 강조합니다. 우리는 힌디(Hindi)라는 저자원 인도 언어를 대상으로 하여, Nemotron-Mini-Hindi 4B라는 영어와 힌디를 지원하는 이중 언어 SLM (Speech Language Model)을 소개합니다.

- **Technical Details**: Nemotron-Mini 4B 기반으로 만들어진 이 모델은 실제 및 합성( synthetic) 힌디 + 영어 토큰의 혼합을 사용하여 훈련되며, 총 400B 토큰에 대한 지속적인 사전 학습이 수행됩니다. 이 모델은 SFT (Supervised Fine-Tuning) 및 DPO (Direct Preference Optimization) 기법을 사용하여 조정됩니다.

- **Performance Highlights**: 이 모델은 힌디 벤치마크에서 최첨단(State-of-the-art) 성과를 달성하며, 영어 작업에서도 경쟁력을 유지하는 것으로 나타났습니다. 특히, 지속적인 사전 학습 접근법이 모델의 전반적인 사실 정확도를 향상시킴을 확인했습니다.



### Effects of Soft-Domain Transfer and Named Entity Information on Deception Detection (https://arxiv.org/abs/2410.14814)
- **What's New**: 현대 사회에서 온라인 상의 대화가 폭발적으로 늘어남에 따라, 작성된 내용이 진짜인지 속임수인지를 판별하기가 어려워졌습니다. 이 논문에서는 여러 온라인 데이터셋을 사용하여 텍스트 속임수 감지를 위한 기계 학습 모델의 성능을 개선하는 방법을 제시합니다.

- **Technical Details**: 여덟 개의 데이터셋을 사용하여 fine-tuned BERT 모델의 중간 레이어를 연결한 후 transfer learning을 통해 분류 성능을 향상시켰습니다. 결과적으로, 제시된 방법은 기존의 baseline보다 높은 정확도를 보여 주었습니다. 또한, Jensen-Shannon distance를 활용하여 데이터셋 간의 성능 상관관계를 분석했습니다.

- **Performance Highlights**: 기존 모델과 비교하여 명명된 개체(named entities) 처리 방법을 적용함으로써 최대 11.2%의 정확도 향상을 달성했습니다. 이 연구는 텍스트 속임수 감지에 있어 deep learning 기반 전이 학습 방법이 boosting 기반 방법보다 성능이 우수함을 입증했습니다.



### Isolated Causal Effects of Natural Languag (https://arxiv.org/abs/2410.14812)
- **What's New**: 이 논문은 언어 기술이 광범위하게 사용됨에 따라 언어의 변이가 독자의 인식에 미치는 영향을 이해하는 것이 중요하다는 점을 강조합니다. 또한, 비목표 언어(non-focal language)의 근사화가 격리된 인과 효과(isolated causal effect) 추정에 미치는 영향을 탐구하는 포멀한 추정 프레임워크를 소개합니다.

- **Technical Details**: 격리된 인과 효과를 추정하는 것은 비목표 언어를 정확히 정의하고 근사하는 것이 매우 중요합니다. 이 논문에서는 비목표 언어의 근사질을 평가하기 위한 두 가지 메트릭을 제시하고, 통계학적 원리인 누락된 변수 편향(omitted variable bias, OVB)에 기반하여 비목표 언어와 격리된 효과 추정의 질을 평가합니다.

- **Performance Highlights**: 실험을 통해 제안된 프레임워크가 진정한 격리 효과를 회복할 수 있음을 검증하였습니다. 코드를 공개함으로써, 이 방법이 실제 환경에서도 유용할 것임을 강조하며, 제안된 메트릭이 격리 효과 추정 및 비목표 언어 근사에 대한 품질의 척도로 기능하는 데 깊은 통찰을 제공함을 보여주었습니다.



### Cross-Document Event-Keyed Summarization (https://arxiv.org/abs/2410.14795)
- **What's New**: 이번 연구에서는 Event-keyed Summarization (EKS)을 여러 문서의 정보를 통합하여 요약하는 크로스-문서 설정인 Cross-Document Event-Keyed Summarization (CDEKS)으로 확장하였습니다. 이를 위해 새로운 데이터셋인 SEAMUS (Summaries of Events Across Multiple Sources)를 소개하며, 이는 FAMUS 데이터셋의 전문가 재주석을 기반으로 만들어졌습니다.

- **Technical Details**: SEAMUS 데이터셋은 다수의 출처로부터 동일한 사건에 대한 정보 요약을 요구합니다. 연구진은 한 이벤트에 대한 단일 및 크로스-문서 요약의 전문가 주석 데이터셋을 수집하였습니다. 모델 성능 분석을 위해 다양한 베이스라인을 제시하고, 작은 세부 조정 모델과 프롬프트 기반의 LLM을 포함한 비교 결과를 포함합니다.

- **Performance Highlights**: CDEKS는 특히 단일 문서 EKS에 비해 도전적인 작업으로 평가되며, 연구는 모델들의 현재 능력에 대한 세부적인 평가를 제공합니다. 또한, 인간 평가를 통해 모델 성능에 대한 깊이 있는 통찰력을 제공합니다.



### Enabling Scalable Evaluation of Bias Patterns in Medical LLMs (https://arxiv.org/abs/2410.14763)
- **What's New**: 본 연구에서는 의료 분야에 특화된 대규모 언어 모델(Med LLM)의 편향 평가를 자동으로 생성하는 새로운 방법을 제시합니다. 이러한 방식은 기존의 수작업 데이터셋에 의존하지 않고, 엄격한 의학적 증거에 기반하여 테스트 케이스를 자동 생성함으로써 편향 평가를 확장하는 데 기여합니다.

- **Technical Details**: 제안된 방법은 의료 지식 그래프(medical knowledge graphs), 의료 온톨로지(medical ontologies), 사용자 맞춤형 LLM 평가 프레임워크를 통합하여 세 가지 주요 문제(도메인 전문성에 따른 편향 특성화, 생성 과정에서의 환각(hallucination), 건강 결과와 민감 속성 간의 다양한 종속성)를 해결합니다. 이를 통해 의료 분야에서의 공정성을 평가할 수 있는 새로운 데이터셋을 생성하였습니다.

- **Performance Highlights**: 제안된 방법은  기존 수작업 생성 방법에 비해 Med LLM의 편향 패턴을 보다 효과적으로 드러내며, 실험 결과를 통해 입력된 테스트 케이스가 신뢰할 수 있는 지식의 기반을 갖춘 강력한 비네트(vignette) 생성이 가능함을 입증하였습니다.



### Controllable Discovery of Intents: Incremental Deep Clustering Using Semi-Supervised Contrastive Learning (https://arxiv.org/abs/2410.14755)
Comments:
          Accepted in IJCNLP'23

- **What's New**: 이 논문에서는 Conversational AI 시스템에서 사용자 의도를 효과적으로 발견할 수 있도록 돕는 새로운 CDI (Controllable Discovery of Intents) 프레임워크를 제안합니다. 이 프레임워크는 사용자 지식과 도메인 정보를 포함하여 intent discovery 과정에서 인간 피드백을 통합할 수 있도록 설계되었습니다.

- **Technical Details**: CDI 프레임워크는 처음에 unlabeled data에 대해 unsupervised contrastive learning을 실행하고, 그 후 partially labeled data에서 fine-tuning을 통해 클러스터링과 표현을 반복적으로 개선합니다. 이 과정에서는 catastrophic forgetting을 방지하기 위해계속 학습(continual learning)에서 제안된 learning-without-forgetting 기법을 활용합니다.

- **Performance Highlights**: 실험 결과, CDI 프레임워크는 CLINC와 BANKING 데이터셋에서 이전 연구보다 각각 10.26%와 11.72% 향상된 성능을 보여줍니다.



### Accounting for Sycophancy in Language Model Uncertainty Estimation (https://arxiv.org/abs/2410.14746)
- **What's New**: 이 논문은 sycophancy(아첨) 편향과 불확실성 추정 간의 관계를 처음으로 연구하고, 이를 통해 불확실성 추정 프로세스에서 sycophancy를 고려하는 새로운 알고리즘 SyRoUP을 제시합니다.

- **Technical Details**: 논문은 모델의 답변에 대한 불확실성을 추정하기 위해 제안된 답변을 기반으로 불확실성을 평가하는 방식을 사용합니다. 사용자 행위의 다양성과 그에 따른 sycophancy의 영향을 살펴보며, Brier Score와 Brier Skill Score를 통해 추정 정확성을 평가합니다.

- **Performance Highlights**: 실험 결과, 사용자 신뢰도는 sycophancy의 영향을 조절하는 중요한 역할을 하고, SyRoUP은 이러한 영향을 더 잘 예측할 수 있음을 보여줍니다. 이로 인해 모델과 사용자가 모두 불확실성을 외부화할 때 sycophancy 편향의 영향을 완화할 수 있다는 점을 주장합니다.



### SemiEvol: Semi-supervised Fine-tuning for LLM Adaptation (https://arxiv.org/abs/2410.14745)
- **What's New**: 본 논문은 대규모 언어 모델(LLMs)의 효과적인 조정을 위해 레이블이 있는 데이터와 레이블이 없는 데이터를 모두 활용하는 반지도 학습(framework)을 제안합니다. 새로운 프레임워크인 SemiEvol은 지식 전파 및 선택 전략을 통해 모델의 성능을 개선하는 데 중점을 둡니다.

- **Technical Details**: SemiEvol의 주요 구성 요소는 다음과 같습니다: 1) Knowledge Propagation - 레이블이 있는 데이터를 사용하여 모델 성능을 향상시키는 두 단계 접근 방식을 적용합니다. 2) Knowledge Selection - 협업 학습 메커니즘을 통해 레이블이 없는 데이터에서 더 높은 품질의 pseudo-response 샘플을 선택합니다. 이는 k-최근접 이웃 검색을 사용하여 예측을 보조합니다.

- **Performance Highlights**: SemiEvol은 MMLU, MMLU-Pro 및 ConvFinQA와 같은 7개의 일반 및 도메인 특정 데이터셋에서 실험을 진행하였으며, 모델 성능의 유의미한 개선을 나타냈습니다. 또한, SemiEvol은 기존의 SFT 및 self-evolution 방법과 비교하여 하이브리드 데이터 시나리오에서 실용성을 강조하였습니다.



### Eliciting Uncertainty in Chain-of-Thought to Mitigate Bias against Forecasting Harmful User Behaviors (https://arxiv.org/abs/2410.14744)
- **What's New**: 이 논문에서는 대화 예측(conversation forecasting) 작업을 수행하는 데 필요한 모델의 불확실성(uncertainty)을 다루고 있습니다. 특히, 다양한 언어 모델(large language models, LLMs)이 대화 예측의 편향(bias)을 어떻게 줄일 수 있는지를 탐구하고 있습니다.

- **Technical Details**: 연구는 5개의 오픈 소스 언어 모델을 사용해 2개의 데이터셋을 대상으로 진행되었습니다. 이 데이터셋은 소셜 미디어의 유해한 행동(예: 개인 공격)을 예측하기 위해 설계된 것입니다. 주요 질문은 LLM의 예측 정확도가 불확실성을 표현하도록 요청했을 때 어떻게 변화하는지와, 이러한 불확실성을 기반으로 어떤 형태의 편향이 완화될 수 있는지를 포함합니다.

- **Performance Highlights**: 논문은 LLM의 예측 능력 및 편향을 검토하며, 불확실성을 고려하는 것이 모델의 예측 결과에 긍정적인 영향을 미칠 것이라는 가설을 제시합니다. 특히, 시스템이 유해한 결과를 예측할 때 기존의 편향을 줄이는 방안을 모색하고 있습니다.



### Agent Skill Acquisition for Large Language Models via CycleQD (https://arxiv.org/abs/2410.14735)
- **What's New**: CycleQD라는 새로운 프레임워크를 소개하며, 알고리즘의 순환 적응을 통해 Quality Diversity (QD) 방식을 활용하여 여러 과제의 성과를 독립적으로 최적화합니다. 이 방식은 모델 병합 기반의 크로스오버와 SVD 기반의 변이를 사용하여 각 기술의 효율성을 높이고 데이터 비율 관리의 복잡성을 줄입니다.

- **Technical Details**: CycleQD는 연속적인 에이전트 기술의 미세 조정을 위해 설계되었습니다. 각 기술의 성과 메트릭을 개별적으로 최적화하고, 모델 병합 기반의 크로스오버를 통해 전문 기술을 통합하며, SVD 기반의 변이를 통해 모델의 과적합을 방지합니다. 이를 통해 일반적인 손실 함수의 한계를 극복하고 작업 특정 성과 메트릭을 직접 최적화합니다.

- **Performance Highlights**: CycleQD를 사용하여 LLAMA3-8B-INSTRUCT 모델이 기존의 미세 조정 방법보다 뛰어난 성능을 보여주며, 코드, 운영 체제 및 데이터베이스 작업에서 GPT-3.5-TURBO와 동일한 수준의 성능을 달성했습니다. 이는 데이터 비율 조정이나 일반 언어 능력 감소 없이 이루어졌습니다.



### A two-stage transliteration approach to improve performance of a multilingual ASR (https://arxiv.org/abs/2410.14709)
- **What's New**: 본 논문은 다국어를 처리하는 데 있어 종단 간(End-to-End) 음성 인식 시스템의 확장성과 적응성을 높이는 새로운 접근 방식을 제안합니다. 이 방법은 언어에 구애받지 않는 모델을 구축하며, 다양한 언어의 그래프 집합을 사용하여 일반적인 목표 언어의 문자 체계로 매핑합니다.

- **Technical Details**: 제안하는 방식은 두 단계의 음절-문자 변환(transliteration) 과정을 통해 이뤄집니다. 이 과정을 통해 언어 간의 음향적 혼란을 최소화하고 내부 표현을 학습하게 되어 음향-음소(content)의 질을 개선합니다. 논문에서는 네팔어와 텔루구어 두 가지 언어로 연구를 진행하였으며, 이 언어들의 원래 문자 공간을 데바나가리(Devanagari) 스크립트로 사영(project)합니다.

- **Performance Highlights**: 제안된 방법은 다른 언어 의존 모델링 방식에 비해 단어 오류율(Word Error Rate, WER)을 20% 감소시켰으며, 문자 오류율(Character Error Rate, CER)은 24% 감소하는 성과를 보였습니다.



### xGen-MM-Vid (BLIP-3-Video): You Only Need 32 Tokens to Represent a Video Even in VLMs (https://arxiv.org/abs/2410.16267)
- **What's New**: xGen-MM-Vid (BLIP-3-Video)은 비디오를 위한 멀티모달 언어 모델로, 여러 프레임에서의 시간 정보를 효율적으로 캡처하도록 설계되었습니다. 이 모델은 일반적인 비주얼 토크나이저(visual tokenizer) 외에 'temporal encoder'를 도입하여 훨씬 적은 수의 visual tokens를 사용하여 비디오 질문-응답에서 높은 정확도를 기록합니다.

- **Technical Details**: BLIP-3-Video는 4개의 주요 구성 요소로 이루어져 있습니다: (1) 각 프레임 입력을 처리하는 비전 인코더(vision encoder), (2) 토큰 수를 줄이는 프레임 레벨 토크나이저(frame-level tokenizer), (3) 비디오 수준의 토큰 표현을 구축하는 템포럴 인코더(temporal encoder), (4) 비디오 토큰과 텍스트 프롬프트 토큰에 기반하여 출력 텍스트 캡션을 생성하는 자동 회귀 LLM입니다. 이 모델은 8개의 샘플링된 프레임을 사용하여 계산 효율성을 높입니다.

- **Performance Highlights**: BLIP-3-Video는 34B의 거대 모델과 비교하여 비슷한 질문-응답 정확도를 보이며, 단 4B로도 성능을 발휘합니다. 각각 16에서 32개의 비디오 토큰을 추상화하여 전체 비디오를 성공적으로 표현할 수 있습니다.



### Compute-Constrained Data Selection (https://arxiv.org/abs/2410.16208)
- **What's New**: 이번 연구는 데이터 선택(data selection)을 통해 LLM(finetuning) 훈련에 필요한 데이터 양을 줄이는 방법을 제안하고, 데이터 선택과 훈련 비용을 고려한 최적화 문제를 포괄적으로 분석합니다. 이를 통해 자원 제한이 있는 상황에서도 최적의 모델 성능을 도출할 수 있는 방법을 제시합니다.

- **Technical Details**: 연구는 compute-aware utility function을 통해 데이터 선택 문제를 형식화하고 초기 선택 비용과 훈련 이익 간의 균형을 탐구합니다. 모델 사이즈, 토큰 수, 그리고 데이터 선택 방안 사이의 관계를 포괄적으로 정량화하며, 600개 이상의 모델을 훈련하여 성과를 분석합니다.

- **Performance Highlights**: 복잡한 데이터 선택 방법은 compute-constrained 환경에서 거의 Pareto-optimal하지 않으며, 간단한 통계 기법인 sparse retrieval이 선호되어야 함을 발견했습니다. 이는 이론적 및 경험적 관점에서 FLOP 비효율성에 따른 것입니다. 또한 반복 훈련을 진행하는 환경에서는 이러한 강력한 방법들이 여전히 효과적이게 사용될 수 있음을 강조합니다.



### CoT-TL: Low-Resource Temporal Knowledge Representation of Planning Instructions Using Chain-of-Thought Reasoning (https://arxiv.org/abs/2410.16207)
Comments:
          Accepted for publication in Proceedings of the 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2024), Abu Dhabi 14-18 October 2024

- **What's New**: 이번 연구에서는 자연어(Natural Language) 명세를 선형 시계열 논리(Linear Temporal Logic, LTL)로 변환하기 위한 데이터 효율적인 학습 프레임워크, CoT-TL을 소개합니다. CoT-TL은 대규모 언어 모델(Large Language Model, LLM)의 한계를 극복하기 위해 체인 오브 스로트(Chain-of-Thought, CoT) 추론과 의미 롤(Semantic Roles)을 확장하여 형식 논리 생성의 요구 사항과 일치하도록 설계되었습니다.

- **Technical Details**: CoT-TL 프레임워크는 기존의 대규모 데이터셋에 의존하지 않고, 미세 조정(Fine-tuning) 없이 자연어 지침을 LTL로 번역하는 기능을 갖추고 있습니다. CoT-TL은 추론 과정에서 LLM의 세계 지식을 활용하여 몇몇 고품질의 데이터 샘플을 바탕으로 효과적으로 학습합니다. 또한, 생성된 LTL이 반드시 구문적으로 해석할 수 있고 모델 검사를 통해 유효성을 검증하는 것에 중점을 둡니다.

- **Performance Highlights**: CoT-TL은 세 가지 다양한 데이터 셋에서 낮은 데이터 환경에서도 기존 방법보다 뛰어난 정확도를 보여주었습니다. 중간 번역 단계 없이 성능을 발휘하며, 새로운 데이터 세트에 있는 보이지 않는 LTL 구조와 수식에서 효과를 입증했습니다. 최종적으로 이 시스템은 자연어 지침에 기반한 다단계 드론 플래닝에서 QuadCopter에 통합되어 실용성을 확인했습니다.



### Systematic Review: Text Processing Algorithms in Machine Learning and Deep Learning for Mental Health Detection on Social Media (https://arxiv.org/abs/2410.16204)
- **What's New**: 이번 연구는 소셜 미디어에서 우울증(Depression) 탐지를 위한 머신러닝(Machine Learning) 모델을 체계적으로 검토하고, 이러한 과정에서 발생하는 편향(Bias) 및 방법론적 과제(Methodological Challenges)를 분석했습니다.

- **Technical Details**: PubMed, IEEE Xplore, Google Scholar에서 2010년 이후에 발표된 47개의 관련 연구를 검색하여, 예측 모델 위험 평가 도구(Probabilistic Risk Of Bias ASsessment Tool, PROBAST)를 통해 방법론적 품질과 편향 위험을 평가했습니다. 연구 결과, 63.8%가 Twitter에 의존하고, 90% 이상이 영어 콘텐츠를 사용하며, 미국과 유럽의 사용자에 집중되고 있음을 발견했습니다. 비확률적 샘플링 방법이 약 80% 사용되어 대표성이 제한되었고, 23%의 연구만이 부정(Negation)과 같은 언어적 뉘앙스를 명시적으로 다루었습니다.

- **Performance Highlights**: 모델의 하이퍼파라미터 튜닝(Hyperparameter Tuning) 일관성이 부족했으며, 27.7%만이 적절히 튜닝을 하였습니다. 약 17%는 데이터를 훈련(Training), 검증(Validation), 테스트(Test) 세트로 적절히 분리하지 않아 과적합(Overfitting)의 위험이 있었습니다. 74.5%는 불균형 데이터(Imbalanced Data)에 대한 적절한 평가 메트릭(Evaluation Metrics)을 사용했지만, 일부는 클래스 불균형(Class Imbalance)을 해결하지 않고 정확도(Accuracy)에만 의존했습니다. 보고의 투명성이 다양하게 나타나고 필수적인 방법론적 세부사항이 종종 부족했습니다.



### Beyond Filtering: Adaptive Image-Text Quality Enhancement for MLLM Pretraining (https://arxiv.org/abs/2410.16166)
- **What's New**: 본 논문에서는 멀티모달 대형 언어 모델(MLLMs)의 성능을 향상시키기 위한 새로운 접근법인 Adaptive Image-Text Quality Enhancer (AITQE)를 제안합니다. AITQE는 기존 데이터의 질을 동적으로 평가하고 향상시키는 모델로, 이미지-텍스트 쌍의 품질을 개선합니다.

- **Technical Details**: AITQE는 낮은 품질의 이미지-텍스트 쌍에 대해 텍스트를 재작성하는 메커니즘을 사용하고, 부정 샘플 학습 전략을 통합하여 평가 능력을 향상시킵니다. 본 모델은 기존 방법들보다 텍스트 분포를 최소한으로 조정하여 데이터 볼륨을 유지하면서 품질을 개선합니다.

- **Performance Highlights**: 실험 결과, AITQE는 여러 벤치마크에서 기존 방법들을 초월하여 원시 데이터를 효과적으로 활용하고, 데이터 양 증가에 따라 효율적으로 확장할 수 있음을 보여주었습니다.



### Sparkle: Mastering Basic Spatial Capabilities in Vision Language Models Elicits Generalization to Composite Spatial Reasoning (https://arxiv.org/abs/2410.16162)
- **What's New**: Vision Language Models (VLMs)의 2D 공간 추론 능력을 향상시키기 위한 새로운 접근법인 Sparkle 프레임워크를 소개합니다.

- **Technical Details**: 우리의 연구는 방향 이해(direction comprehension), 거리 추정(distance estimation), 위치 파악(localization)으로 구성된 2D 공간 추론의 기초 능력을 분리하고 강화하는 방법론을 사용합니다.

- **Performance Highlights**: Sparkle로 미세 조정된 VLMs는 기본 작업에서 13.5%에서 40.0%까지 성능 향상을 보여주며, 복합 및 비분포적 공간 추론 작업으로의 일반화에도 긍정적인 결과를 나타냈습니다.



### Can Large Audio-Language Models Truly Hear? Tackling Hallucinations with Multi-Task Assessment and Stepwise Audio Reasoning (https://arxiv.org/abs/2410.16130)
Comments:
          5 pages, 1 figure

- **What's New**: 최근 대형 오디오-언어 모델(LALMs)의 발전은 audio와 speech 정보 이해 및 추론에서 인상적인 능력을 보여주고 있습니다. 하지만 이러한 모델들은 비현실적인 소리 이벤트를 환각하거나 잘못된 순서로 소리 이벤트를 식별하는 등 여러 도전 과제에 직면해 있습니다. 이 연구는 이러한 문제를 체계적으로 평가하기 위해 객체 존재, 시간 순서, 객체 속성이라는 세 가지 작업을 제안합니다.

- **Technical Details**: 제안된 모델의 평가 작업은 객체 존재성, 시간 순서, 객체 속성 등 세 가지로 구성됩니다. 객체 존재성 작업은 모델이 특정 소리 이벤트를 감지하는 능력을 평가하며, 시간 순서 작업은 모델이 소리 이벤트의 순서를 파악하는 능력을 측정합니다. 객체 속성 작업은 소리 출처를 식별하는 모델의 기술을 평가합니다. 이를 위해 다단계(chain-of-thought) 접근법인 MATCH를 사용하여 모델 성능을 향상시킵니다.

- **Performance Highlights**: 실험 결과에 따르면 현재 LALMs는 제안된 작업에서 부족한 성과를 보였습니다. 그러나 MULTI-TURN AND THOUGHTFUL CHAIN OF HEARINGS (MATCH) 방법을 도입한 결과, 모든 작업에서 모델 성능이 크게 향상됨을 보여주었습니다.



### CartesianMoE: Boosting Knowledge Sharing among Experts via Cartesian Product Routing in Mixture-of-Experts (https://arxiv.org/abs/2410.16077)
- **What's New**: 이번 논문에서는 LLM(대형 언어 모델) 구축을 위한 새로운 Mixture-of-Experts(MoE) 모델인 CartesianMoE를 제안합니다. 이 모델은 전문가 간의 지식 공유를 더욱 효과적으로 수행하여 이전 MoE 모델보다 우수한 성능을 나타냅니다.

- **Technical Details**: CartesianMoE는 집합으로 정의된 두 개의 서브 전문가들 간의 카르티지안 곱(Cartesian product)을 통해 전문가를 파생시키며, 지식 공유를 ‘곱셈’ 방식으로 처리합니다. 각 전문가가 여러 하위 전문가와 동일한 하위 전문가를 공유하도록 설계되어 있습니다.

- **Performance Highlights**: 실험 결과, CartesianMoE는 perplexity(PPL)와 다운스트림 작업 성능 모두에서 이전 MoE 모델을 지속적으로 능가하였으며, 라우팅(Routing) 정확성에서도 향상된 강인성을 보여주었습니다.



### On-Device LLMs for SMEs: Challenges and Opportunities (https://arxiv.org/abs/2410.16070)
Comments:
          9 pages, 1 figure. The work is supported by the SIT-NVIDIA Joint AI Centre

- **What's New**: 이번 논문은 중소기업(SME)에서 온디바이스(온 디바이스, on-device)로 대규모 언어 모델(LLM)을 배포할 때의 인프라 요구사항에 대해 체계적인 리뷰를 제공합니다. 하드웨어와 소프트웨어 관점을 모두 다루어 중소기업이 혁신적 AI 기술을 통합하여 경쟁력을 유지할 수 있도록 돕고자 합니다.

- **Technical Details**: 하드웨어 관점에서는 GPU와 TPU(Proccessing Unit) 사용, 메모리 및 저장 솔루션의 최적화, 효과적인 배포 전략 등을 논의합니다. 소프트웨어 관점에서는 프레임워크 호환성, 운영체제 최적화, 리소스가 제한된 환경에 맞춘 특화된 라이브러리 활용의 필요성을 강조합니다. 따라서 LLM의 최적화뿐만 아니라 인프라의 전반적인 효율성도 중요합니다.

- **Performance Highlights**: SME에서 LLM을 효과적으로 활용하기 위해 CUDA를 도입하면 데이터 처리 속도 및 기계 학습 작업의 효율성을 향상시킬 수 있으며, TensorFlow Lite와 PyTorch Mobile과 같은 프레임워크는 제한된 리소스 환경에서도 최적의 성능을 보장할 수 있습니다. 이러한 검토는 중소기업이 자사 인프라 내에서 AI 모델을 성공적으로 통합하고 활용할 수 있는 중요한 통찰을 제공합니다.



### Mitigating Object Hallucination via Concentric Causal Attention (https://arxiv.org/abs/2410.15926)
Comments:
          To appear at NeurIPS 2024. Code is available at this https URL

- **What's New**: 최근의 대형 비전 언어 모델(Large Vision Language Models, LVLMs)은 다중 모달 쿼리에 대해 놀라운 제로샷 대화 및 추론 능력을 보여주고 있습니다. 하지만 객체 환각(object hallucination) 문제에 시달리고 있습니다.

- **Technical Details**: 객체 환각은 LVLMs가 이미지 입력과 사실적으로 일치하지 않는 텍스트 응답을 생성하는 현상으로, 이는 일반적으로 사용되는 위치 인코딩 방식인 Rotary Position Encoding (RoPE)와 밀접한 관련이 있습니다. RoPE의 장기적 감소(long-term decay)로 인해 LVLMs는 시각적 단서가 명령어 토큰(instruction tokens)으로부터 멀어질 경우 환각이 발생할 가능성이 높아집니다. 또한 다중 모달 정렬(multimodal alignment) 과정에서 시각적 토큰의 순서를 뒤바꿀 때도 유사한 효과가 관찰되었습니다.

- **Performance Highlights**: 우리의 연구 결과에 따르면, Concentric Causal Attention (CCA)라는 단순하지만 효과적인 위치 정렬 전략이 RoPE의 장기적 감소로 인한 문제를 완화시킵니다. CCA를 통해 시각적 토큰과 명령어 토큰 간의 상대적 거리가 자연스럽게 감소하여 시각적 상호작용이 개선되고 객체 환각이 줄어듭니다. 기존의 환각 완화 전략보다 여러 객체 환각 벤치마크에서 큰 성과를 보였습니다.



### InternLM2.5-StepProver: Advancing Automated Theorem Proving via Expert Iteration on Large-Scale LEAN Problems (https://arxiv.org/abs/2410.15700)
- **What's New**: 이 논문에서는 Large Language Models (LLMs)을 활용하여 수리 정리를 자동으로 증명하는 새로운 접근 방식을 제안합니다. Lean-workbook 데이터 세트를 사용하여 전문가 반복(expert iteration) 기법을 적용하고, 20,000 CPU 일 이상 소모하여 모델을 학습합니다.

- **Technical Details**: 우리는 Lean-workbook-plus 데이터 세트를 활용하여, 크리틱 모델을 훈련시켜 비교적 쉬운 문제를 선택하게 합니다. 이 과정에서 해결된 문제의 수와 증명 길이 및 CPU 사용량 간에 로그-선형(log-linear) 관계가 발견되었습니다. 기존의 증명 접근 방식보다 더 깊은 증명을 생성하는 방식으로 통합된 크리틱 모델을 사용하였습니다.

- **Performance Highlights**: InternLM2.5-StepProver는 MiniF2F 및 Lean-Workbook-Plus 벤치마크에서 최신 오픈 소스(state-of-the-art) 성능을 달성했습니다. MiniF2F-test에서 65.9%의 통과률을 기록하였으며, Lean-Workbook-Plus 문제 중 17.0%를 증명(혹은 반증)하여 이전의 9.5%와 비교하여 현저한 개선을 보였습니다.



### CL-HOI: Cross-Level Human-Object Interaction Distillation from Vision Large Language Models (https://arxiv.org/abs/2410.15657)
- **What's New**: 본 논문에서는 Vision Language Models (VLMs)와 Vision Large Language Models (VLLMs)의 한계를 극복하기 위한 Cross-Level HOI distillation (CL-HOI) 프레임워크를 제안합니다. 이 방법은 수동 주석 없이 VLLMs의 이미지 수준 이해로부터 인스턴스 수준 HOI를 증류합니다.

- **Technical Details**: CL-HOI 프레임워크는 두 단계로 이루어집니다: 첫 번째는 context distillation으로, Visual Linguistic Translator (VLT)는 시각 정보를 언어 형태로 변환합니다. 두 번째는 interaction distillation로, Interaction Cognition Network (ICN)가 공간적, 시각적 및 맥락 관계를 학습하고 분석합니다. 이를 통해 이미지 수준의 지식이 인스턴스 수준 HOI 탐지에 효과적으로 전달됩니다. 또한, contrastive distillation loss를 도입하여 이미지 수준의 컨텍스트와 상호작용 지식을 전이합니다.

- **Performance Highlights**: CL-HOI는 HICO-DET 데이터셋에서 17.5% mAP, V-COCO 데이터셋에서 36.63% Role AP를 기록하며 기존의 약한 감독 방식 및 VLLM 감독 방식을 능가하는 성능을 보였습니다.



### Improving Parallel Program Performance Through DSL-Driven Code Generation with LLM Optimizers (https://arxiv.org/abs/2410.15625)
Comments:
          26 pages, 8 figures

- **What's New**: 이 논문에서는 LLM(대형 언어 모델)을 기반으로 한 맵퍼(mapper) 생성 자동화를 통해 병렬 프로그래밍의 성능을 극대화하는 새로운 접근법을 제안합니다. 이 방법은 전문가의 설계를 초과하는 성능 향상을 달성하며, 개발 시간을 단 몇 분으로 단축시킵니다.

- **Technical Details**: 이 연구는 도메인 특화 언어(DSL)를 도입하여 복잡한 저수준 C++ 시스템 프로그래밍의 세부정보를 추상화합니다. DSL은 LLM이 독립적인 모듈 구성 요소를 생성할 수 있는 구조적 검색 공간을 정의하며, 이를 통해 코드 생성의 문맥 의존성이 감소합니다. 또한, 규칙 기반 최적화를 위해 강화 학습을 적용하여 성능을 최적화하는 방법을 제안합니다.

- **Performance Highlights**: LLM을 사용하여 생성된 맵퍼는 전문가가 작성한 맵퍼보다 최대 1.34배의 속도 향상을 기록하였으며, 병렬 행렬 곱셈 알고리즘에서는 최대 1.31배의 성능 향상을 달성하였습니다. 실험 결과, DSL을 통해 맵퍼 코드 생성 성공률이 0%에서 80%로 크게 향상되었습니다.



### Acoustic Model Optimization over Multiple Data Sources: Merging and Valuation (https://arxiv.org/abs/2410.15620)
- **What's New**: 이 논문에서는 Automatic Speech Recognition (ASR)에서의 새로운 최적화 패러다임을 제안합니다. 특히 여러 데이터 제공자(curators)로부터 분산된 음성 데이터를 활용하여 각자 독립적으로 모델을 훈련한 후, 이를 합쳐 고품질의 오디오 모델을 생성하는 방식을 제안합니다.

- **Technical Details**: 제안하는 두 가지 알고리즘, Genetic Merge Algorithm (GMA)와 SGD-Based Optimizational Merge Algorithm (SOMA)은 음성 모델의 품질을 향상시키기 위해 진행됩니다. GMA는 고도의 정확성을 갖춘 모델을 생성하나, 비효율성이 문제입니다. SOMA는 GMA의 효율성 병목 현상을 완화시켜 주며, 모델 정확도를 유지합니다. 각 모델의 기여도는 Shapley Value를 이용해 평가되어, 데이터 제공자에게 공정한 인센티브를 제공합니다.

- **Performance Highlights**: 다양한 공개 데이터에 대한 실험을 통해 SOMA는 기존의 최첨단 기술을 훨씬 초월할 수 있는 성능을 보여주며, GMA와 비슷한 성능을 훨씬 적은 계산 비용으로 달성할 수 있음을 입증했습니다.



### Moonshine: Speech Recognition for Live Transcription and Voice Commands (https://arxiv.org/abs/2410.15608)
Comments:
          7 pages, 6 figures, 3 tables

- **What's New**: Moonshine은 실시간 전사(live transcription)와 음성 명령 처리에 최적화된 새로운 음성 인식 모델 패밀리입니다. 이 모델은 기존의 절대 위치 임베딩 대신 Rotary Position Embedding(RoPE)을 사용하며, zero-padding 없이 다양한 길이의 음성 구간에 대해 훈련했습니다.

- **Technical Details**: Moonshine 모델은 인코더-디코더 Transformer 아키텍처에 기반합니다. 입력은 16,000 Hz에서 샘플링된 오디오 신호로, Mel spectrograms와 같은 수동 특징 추출을 사용하지 않고 3개의 합성곱 레이어를 통해 처리됩니다. 모델은 90K 시간의 공개 ASR 데이터셋과 100K 시간의 자체 제작 데이터를 포함해 총 약 200K 시간의 데이터로 훈련되었습니다.

- **Performance Highlights**: Moonshine Tiny 모델은 10초 음성 구간의 전사 시 OpenAI의 Whisper tiny-en 모델에 비해 5배 더 적은 계산 요구량을 보여주었고, 표준 평가 데이터셋에서 단어 오류율(Word Error Rate, WER)은 증가하지 않았습니다. 이는 Moonshine이 자원 제약이 있는 실시간 애플리케이션에 적합함을 나타냅니다.



### A Comprehensive Survey of Datasets, Theories, Variants, and Applications in Direct Preference Optimization (https://arxiv.org/abs/2410.15595)
- **What's New**: 대규모 언어 모델(LLMs)의 발전과 함께 인간의 선호에 맞춰 정책 모델을 정렬하는 것이 점점 더 중요해지고 있습니다. 본 논문에서는 Direct Preference Optimization (DPO)의 최신 연구 동향과 도전 과제를 포괄적으로 검토하며, 이 분야의 발전 상황을 체계적으로 정리합니다.

- **Technical Details**: 본 연구에서는 DPO의 이론적 분석, 다양한 변형, 관련 선호 데이터셋 및 응용 프로그램에 대해 논의합니다. DPO는 정책 모델과 참조 모델에 대해 간단한 최대 우도(objective)를 설정하여 명시적 보상 모델 훈련 단계를 우회하고 강화 학습 최적화를 필요로 하지 않도록 합니다. DPO의 최적화 목표는 정책 모델 자체에 의해 매개변수화된 암묵적 보상 함수와 동등합니다.

- **Performance Highlights**: DPO는 다양한 응용 프로그램에서 안정성, 성능 및 계산 효율성을 보여주고 있으며, 최근 연구 결과에 따르면 RLHF보다 높은 성능을 보이는 온라인 변형 및 데이터 수집 전략 등이 제안되고 있습니다. DPO의 새로운 변형(KTO, IPO, CPO 등) 또한 최근 발표되어 DPO의 장단점을 보완하고 있습니다.



### CPE-Pro: A Structure-Sensitive Deep Learning Model for Protein Representation and Origin Evaluation (https://arxiv.org/abs/2410.15592)
- **What's New**: 단백질 구조를 이해하는 것은 기능과 상호작용을 이해하는 데 중요한 요소입니다. 현재 많은 구조 예측 방법이 구조 데이터베이스를 풍부하게 하고 있습니다. CPE-Pro(Crystal vs Predicted Evaluator for Protein Structure)는 단백질 구조의 기원을 분별하기 위해 개발된 구조 감지 딥러닝 모델입니다.

- **Technical Details**: CPE-Pro는 단백질의 구조적 정보를 학습하고 구조 간의 차이를 포착하여 네 가지 데이터 클래스에 대한 정확한 추적성을 달성합니다. 또한 Foldseek를 사용하여 단백질 구조를 'structure-sequence'로 인코딩하고, Protein Structural Sequence Language Model(SSLM)을 훈련시켰습니다.

- **Performance Highlights**: 초기 실험 결과, 대규모 아미노산 서열로 사전 훈련된 단백질 언어 모델에 비해 'structure-sequences'를 사용하면 언어 모델이 보다 유의미한 단백질 특징을 학습할 수 있어 구조 표현이 개선되고 최적화되었습니다. 또한 CPE-Pro의 코드, 모델 가중치 및 관련 자료가 공개되어 단백질 구조 연구에 중요한 자료로 제공됩니다.



### Language Models are Symbolic Learners in Arithmetic (https://arxiv.org/abs/2410.15580)
- **What's New**: 대형 언어 모델(LLMs)이 산술 학습을 수행하는 방식에 대한 새로운 증거를 제공하는 연구입니다. 연구자들은 모델이 부분 곱(partial products)을 활용하지 않음을 발견하였고, 산술 작업을 기호적(symbolic)으로 처리하는 방법을 탐구했습니다.

- **Technical Details**: 부분 곱을 활용하지 못하는 LLM의 행동을 분석하고, 서브그룹(subgroup) 복잡성과 선택(subgroup selection)의 관점에서 산술 문제를 해결하는 방식을 조사했습니다. 연구는 LLM이 기호 수준에서 패턴 찾는 방식으로 동작한다는 점을 강조했습니다.

- **Performance Highlights**: LLMs는 쉬운 패턴을 쉽게 배우고 어려운 패턴은 점진적으로 학습하는 U자형 패턴을 보였습니다. 서브그룹 선택 메커니즘에 의해 정확도가 높아짐을 확인했습니다. 이러한 결과는 LLM의 기호적 학습 행동을 뒷받침합니다.



### Generalized Probabilistic Attention Mechanism in Transformers (https://arxiv.org/abs/2410.15578)
- **What's New**: 이 논문에서는 Transformer 구조의 전통적인 attention 메커니즘이 갖는 두 가지 문제인 rank-collapse와 gradient vanishing 문제를 동시에 해결하기 어렵다는 이론적 분석을 제시합니다. 이를 해결하기 위해, 일반화된 확률적 attention 메커니즘(generalized probabilistic attention mechanism, GPAM) 및 Transformer 아키텍처 내에서의 이중 attention 구현인 daGPAM을 도입합니다.

- **Technical Details**: GPAM은 고정된 총합을 유지하면서도 음수의 attention 점수를 허용합니다. 이는 conventional attention 메커니즘에서 발생하는 두 가지 문제를 효과적으로 완화할 수 있는 이론적 근거를 제공합니다. 특히, GPAM은 출력 표현이 입력 표현의 convex hull에 제한되지 않고 affine hull 내에서 생성되도록 하여 rank-collapse 문제를 제거하는 방식으로 작동합니다.

- **Performance Highlights**: empirical한 검증을 통해 daGPAM은 기존의 attention 메커니즘들보다 뛰어난 성능을 발휘하며, 자연어 처리 작업에서 언어 모델링 및 신경 기계 번역과 같은 실용적인 이점을 강하게 증명합니다.



### OpenMU: Your Swiss Army Knife for Music Understanding (https://arxiv.org/abs/2410.15573)
Comments:
          Resources: this https URL

- **What's New**: OpenMU-Bench라는 대규모 벤치마크 세트를 소개했으며, 이는 음악 모달리티를 이해하기 위한 다중 모달 언어 모델(Multimodal Large Language Models, MLLMs) 훈련 시 데이터 부족 문제를 해결하기 위해 설계되었습니다.

- **Technical Details**: OpenMU는 음악 클립을 이해하기 위해 훈련된 MLLM이며, 음악 캡셔닝, 추론, 다중 선택 질문 응답 작업에서 MU-Llama와 같은 기존 모델을 초월하는 성능을 보였습니다. OpenMU-Bench는 약 100만 개의 음악 이해 데이터 예제를 포함하고 있으며, 다양한 음악 이해 작업을 다룹니다.

- **Performance Highlights**: OpenMU는 OpenMU-Bench를 통해 훈련되었으며, 음악 캡셔닝 및 다중 선택 질문 응답 작업에서 우수한 성능을 발휘했습니다. 또한 OpenMU와 OpenMU-Bench는 모두 오픈 소스로 제공되어 향후 음악 이해 및 창의적인 음악 제작의 효율성을 높이는 데 기여할 것으로 기대됩니다.



### Pruning Foundation Models for High Accuracy without Retraining (https://arxiv.org/abs/2410.15567)
Comments:
          Accepted by EMNLP 2024 findings

- **What's New**: 본 논문에서는 대형 언어 모델(LLMs)의 효과적인 포스트-트레이닝 프루닝(post-training pruning) 방법을 제안했습니다. 기존의 프루닝 기법이 데이터를 재학습해야 하는 문제를 해결하기 위해 레이어 단위로 다중 가중치를 동시에 프루닝하는 새로운 접근 방식을 소개합니다.

- **Technical Details**: 제안된 방법론은 다중 제거 문제(Multiple Removal Problem, MRP)를 직접 공식화하고, 이를 위한 최적 솔루션을 도출하였습니다. 이를 바탕으로 비구조적(unstructured) 및 반구조적(semi-structured) 스파시티(sparsity)를 위한 포스트-트레이닝 프루닝 알고리즘을 설계했습니다.

- **Performance Highlights**: 광범위한 실험을 통해 다양한 LLM 패밀리에서 SOTA(SOTA) 기준과 비교하여 우수한 성능을 입증했습니다. 예를 들어, LLaMA2-70B 모델의 경우, 2:4 스파시티 하에 4.278의 perplexity를 기록하여 SparseGPT의 5.698을 초월했습니다.



### Exploring Curriculum Learning for Vision-Language Tasks: A Study on Small-Scale Multimodal Training (https://arxiv.org/abs/2410.15509)
Comments:
          CoNLL BabyLM Challenge 2024 camera ready

- **What's New**: 이 논문은 Machine Learning의 제한된 데이터 환경에서 Curriculum Learning의 효과를 탐구하고, 여러 모델 유형 간의 성능 비교를 다룹니다. 특히, Multimodal 모델의 성능을 개선하기 위한 새로운 접근 방식을 제공합니다.

- **Technical Details**: 연구는 3가지 주요 변수를 평가합니다: (i) Curriculum Learning, (ii) 텍스트 전용 데이터로부터의 Pretraining, (iii) 모델 유형. 데이터를 쉽고 난이도에 따라 제시하는 Curriculum Learning 접근 방식을 사용하여 VLM(Vision-Language Models)의 성능 변화를 연구했습니다.

- **Performance Highlights**: Curriculum Learning이 Multimodal 평가에서 Non-Curriculum Learning 모델보다 성능 향상에 기여하며, 특히 텍스트 전용 Pretraining과 결합할 때 효과가 있는 것으로 나타났습니다. 텍스트 전용 작업에서는 더 적은 trainable parameters를 가진 모델에서 Curriculum Learning의 혜택이 나타났습니다.



### Mitigating Forgetting in LLM Supervised Fine-Tuning and Preference Learning (https://arxiv.org/abs/2410.15483)
- **What's New**: 본 논문은 전이 학습된 LLM(대형 언어 모델)의 후속 훈련(post-training) 방법을 제안합니다. 기존의 SFT(전문 지도 학습) 및 RLHF(강화학습된 인간 피드백) 또는 DPO(분포에 대한 우선 순위 학습) 단계를 순차적으로 수행하는 접근 방식에서 발생하는 비효율성을 이론적으로 증명하고, 이를 개선한 새로운 접근 방식을 제시합니다.

- **Technical Details**: 후속 훈련 과정에서 SFT와 RLHF/DPO 간의 균형을 맞추는 것이 중요합니다. 기존의 순차적 훈련 방식에서는 LLM이 SFT 과정에서 배운 내용을 점진적으로 잊어버리는 문제를 안고 있습니다. 이 연구에서는 이에 대한 공동 후속 훈련(joint post-training) 프레임워크를 제안하며, 이론적 수렴 보장을 제공합니다.

- **Performance Highlights**: 우리의 공동 후속 훈련 프레임워크는 기존의 순차적 후속 훈련 방식보다 성능이 우수하며, 유사한 계산 비용(computational cost)으로 구현됩니다.



### Hallucination Detox: Sensitive Neuron Dropout (SeND) for Large Language Model Training (https://arxiv.org/abs/2410.15460)
- **What's New**: 대규모 언어 모델(LLMs)의 신뢰성과 관련된 문제를 다루고 있으며, 특히 환각(hallucinations) 현상의 발생 원인을 분석하고 이를 감소시키기 위한 새로운 훈련 프로토콜인 SEnsitive Neuron Dropout (SeND)을 제안합니다.

- **Technical Details**: 이 연구는 Pythia 모델 시리즈(70M-12B parameters)의 다양한 모델을 분석하고, Efficient EigenScore (EES)라는 새로운 비지도 환각 탐지 메트릭을 개발하여 SeND 프로토콜에 통합합니다. SeND는 높은 변동성을 가진 신경세포(Sensitive Neurons)를 선택적으로 제거함으로써 변동성을 줄이고 신뢰성을 높입니다.

- **Performance Highlights**: SeND 방법을 통해 LLM의 신뢰성을 정상 훈련 대비 최대 40% 개선하고, 위키피디아 및 의료 데이터셋과 같은 도메인에서 사실 정확성을 향상시킬 수 있음을 입증했습니다.



### IPO: Interpretable Prompt Optimization for Vision-Language Models (https://arxiv.org/abs/2410.15397)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 이번 논문은 비전-언어 모델에서 사람의 이해가 가능한 지능형 텍스트 프롬프트 최적화 기법인 IPO(Interpretable Prompt Optimizer)를 제안합니다. 이는 기존의 프롬프트 최적화 기술의 한계를 극복하고 보다 명확한 해석 가능성을 제공합니다.

- **Technical Details**: IPO는 대규모 언어 모델(LLM)과 대형 다중모드 모델(LMM)을 활용하여 동적으로 텍스트 프롬프트를 생성합니다. 특히 기존 방법들과 달리 프롬프트의 다양성과 효과를 개선하기 위해 Prompt Optimization Prompt를 도입하여 과거 프롬프트의 성과를 저장하고 활용합니다.

- **Performance Highlights**: 11개의 데이터셋에서 IPO는 기존의 gradient-descent 기반 프롬프트 학습 방법보다 10.29% 더 높은 정확도를 보였습니다. 또한, 생성된 프롬프트는 해석이 가능하며, CLIP 모델의 일반화 성능을 향상시킵니다.



### Faster-GCG: Efficient Discrete Optimization Jailbreak Attacks against Aligned Large Language Models (https://arxiv.org/abs/2410.15362)
- **What's New**: 이번 논문에서는 Aligned Large Language Models (LLMs)의 취약점을 노출시키는 새로운 공격 기법인 Faster-GCG를 제안합니다. 기존의 Greedy Coordinate Gradient (GCG) 방식의 비효율성을 개선하여, 더 낮은 계산 비용으로 높은 성공률을 기록합니다.

- **Technical Details**: Faster-GCG는 GCG의 세 가지 주요 문제를 해결합니다. 첫째, 기울기 계산에서 다양한 토큰 간 거리를 고려한 정규화 항을 추가합니다. 둘째, 무작위 샘플링 대신 결정론적 탐색을 사용하여 대체 토큰을 평가하고, 셋째, 자기 루프 문제를 방지하기 위한 중복 제거 방법을 제안합니다.

- **Performance Highlights**: Faster-GCG는 원래 GCG의 계산 비용의 1/10로 두 개의 Aligned LLM(Llama-2-7B-chat, Vicuna-13B-v1.5)에 대해 각각 29% 및 8% 높은 공격 성공률을 달성하였습니다. 또한, ChatGPT와 같은 비공개 LLM에서의 공격 전이 가능성도 개선되었습니다.



### CompAct: Compressed Activations for Memory-Efficient LLM Training (https://arxiv.org/abs/2410.15352)
- **What's New**: CompAct라는 새로운 기법이 도입되어 GPU에서 LLM(pretraining 및 fine-tuning) 훈련 시 메모리 사용량을 각각 25-30% 및 50% 줄일 수 있게 되었습니다.

- **Technical Details**: CompAct는 모든 활성화 텐서를 압축하여 저장하는 대신, low-rank(compressed) 활성화를 저장하여 메모리를 줄입니다. 이는 역전파(backward pass) 단계에서 필요한 메모리 양을 크게 줄이는 효과를 가지고 있습니다. 이 방법은 random projection matrices를 사용하여 추가적인 메모리 오버헤드를 피합니다.

- **Performance Highlights**: CompAct는 LLaMA-350M 모델 전이학습(pretraining)에서 약 17.3%의 메모리 절약을, Roberta-Base 모델 fine-tuning 시 50%의 절약 효과를 보였습니다. 모델 크기가 커질수록 메모리 절약 효과가 증가하는 경향이 있으며, LLaMA-65B 모델의 경우 30%의 절약을 예상하고 있습니다.



### EPIC: Efficient Position-Independent Context Caching for Serving Large Language Models (https://arxiv.org/abs/2410.15332)
- **What's New**: 이 논문에서는 위치 독립적 컨텍스트 캐싱(Position-Independent Context Caching, PIC) 개념을 도입한 EPIC이라는 LLM(대형 언어 모델) 서빙 시스템을 제안합니다. 기존의 접두어 기반 캐싱의 한계를 극복하고, 토큰 조각의 위치에 관계없이 KV(키-값) 캐시를 모듈화하여 재사용할 수 있는 기능을 제공합니다.

- **Technical Details**: EPIC의 핵심 설계는 두 가지로 구성됩니다. 첫째는 AttnLink로, 정적 주의 희박성을 활용하여 정확도를 회복하기 위한 재계산을 최소화합니다. 둘째는 KVSplit으로, 의미론적 일관성을 유지하는 맞춤형 조각화 방법을 제공합니다.

- **Performance Highlights**: EPIC는 기존 시스템에 비해 최대 8배의 TTFT(첫 번째 토큰까지의 시간) 개선과 7배의 처리량을 보여줍니다. 정확도 손실은 경미하거나 전혀 발생하지 않았습니다.



### Who is Undercover? Guiding LLMs to Explore Multi-Perspective Team Tactic in the Gam (https://arxiv.org/abs/2410.15311)
- **What's New**: 이번 연구는 다차원적 사고를 강조하는 새로운 프레임워크, Multi-Perspective Team Tactic (MPTT)를 제안합니다. 이 프레임워크는 LLM이 복잡한 시나리오에서 인지 능력을 발휘할 수 있도록 하는 데 중점을 두고 있습니다.

- **Technical Details**: MPTT는 'Who is Undercover?'라는 언어 논리 게임을 실험 플랫폼으로 사용하여, 말하기와 투표 세션을 교 altern하여 진행합니다. 주요 요소로는 자기 관점(self-perspective), 정체성 결정(identity-determination), 자기 반성(self-reflection), 자기 요약(self-summary), 다회차 팀 찾기(multi-round find-teammates)가 포함됩니다.

- **Performance Highlights**: MPTT와 WIU의 결합을 통해 LLM은 인지 능력을 최대한 활용하여 의사 결정 프레임워크를 생성하고, 이는 사회적 소수자들이 의사 소통 및 표현을 돕고 공정성과 다양성을 촉진하는 데 기여합니다. 초기 결과는 LLM이 인간 행동을 학습하고 정렬할 수 있는 잠재력을 나타냅니다.



### Large Language Models for Autonomous Driving (LLM4AD): Concept, Benchmark, Simulation, and Real-Vehicle Experimen (https://arxiv.org/abs/2410.15281)
- **What's New**: 이 연구에서는 Large Language Models (LLMs)을 자율주행 기술에 적용하기 위한 새로운 개념과 접근 방식을 소개하고 있습니다. LLM4AD(LLMs for Autonomous Driving)라는 프레임워크를 제안하며, LLMs의 지시 수용 능력을 평가하기 위한 종합적인 벤치마크도 개발합니다.

- **Technical Details**: 제안된 LLM4AD 프레임워크는 LLMs가 자율주행 시스템 내에서 의사결정의 '두뇌' 역할을 수행하는 구조입니다. 이 프레임워크에서 LLMs는 차량의 감지 및 위치 모듈의 출력을 참고하여 높은 수준의 의사결정을 내립니다. 또한, LLMs는 유저의 피드백 및 시스템 메시지를 바탕으로 자연어로 대화를 하고, 다양한 자율주행 작업에 대한 최적의 주행 정책을 생성합니다.

- **Performance Highlights**: 실험을 통해 LLM4AD 시스템이 자율주행의 다양한 측면을 개선할 수 있는 잠재력을 보여주었습니다. LLMs은 직관적인 언어 상호작용, 맥락 이해 및 추론, 제로샷 및 퍼셉션 업무에 대한 성능 향상, 개인화 등의 이점을 제공합니다. 그러나 실시간 의사결정 능력의 지연과 같은 한계도 있으므로, 추가적인 연구가 필요합니다.



### TAGExplainer: Narrating Graph Explanations for Text-Attributed Graph Learning Models (https://arxiv.org/abs/2410.15268)
- **What's New**: TAGExplainer는 Text-Attributed Graphs (TAGs) 학습을 위한 자연어 설명을 생성하는 최초의 방법으로, 기존 모델의 결정 과정에 대한 이해를 높이는 혁신적인 접근 방식을 제공합니다.

- **Technical Details**: TAGExplainer는 generative language model을 활용해 saliency 기반 설명에서 모델의 결정을 반영한 pseudo-label을 생성한 후 세 가지 훈련 목표를 기반으로 모델을 반복적으로 훈련시킵니다. 이를 통해 고품질 pseudo-label이 생성된 후, 이를 이용하여 end-to-end 설명 생성 모델을 학습합니다.

- **Performance Highlights**: 광범위한 실험을 통해 TAGExplainer가 신뢰할 수 있고 간결한 자연어 설명을 효과적으로 생성함을 입증하였으며, 모델의 결정 과정을 잘 반영하는 설명을 제공함으로써 실제 활용 가능성을 높였습니다.



### When Machine Unlearning Meets Retrieval-Augmented Generation (RAG): Keep Secret or Forget Knowledge? (https://arxiv.org/abs/2410.15267)
Comments:
          15 pages, 9 figures, 9 tables

- **What's New**: 대형 언어 모델(LLMs)인 ChatGPT와 Gemini의 배포는 자연어 생성(natural language generation) 능력의 강력함을 보여주었습니다. 그러나 모델이 학습 중 민감한 정보나 유해한 내용을 학습하게 될 수 있어 윤리적, 법적 문제를 초래합니다. 이를 해결하기 위해 머신 언러닝(machine unlearning)이 제안되었습니다. 기존의 언러닝 방법은 LLM의 특성을 고려하지만 높은 계산 비용과 제한된 적용 가능성, 재앙적 망각(catatrophic forgetting)의 위험을 포함한 문제점이 존재했습니다. 본 논문에서는 Retrieval-Augmented Generation(RAG) 기술 기반의 경량화된 언러닝 프레임워크를 제안합니다.

- **Technical Details**: 우리는 RAG의 외부 지식 기반을 수정하여 언러닝되지 않은 LLM과 직접 상호작용하지 않고 잊는 효과를 시뮬레이션합니다. 언러닝 지식 구성을 제약 최적화 문제로 간주하고, RAG 기반 언러닝의 효과성을 뒷받침할 두 가지 주요 구성 요소를 도출했습니다. 평가를 위해 ChatGPT, Gemini, Llama-2-7b-chat-hf, PaLM 2 등 다양한 모델에서 실험을 진행했습니다. RAG의 사용으로 언러닝이 LLM의 매개변수를 변경하지 않고도 이루어질 수 있습니다.

- **Performance Highlights**: 제안된 RAG 기반 언러닝 프레임워크는 효과성, 보편성, 무해성, 단순성, 강건성의 다섯 가지 주요 언러닝 기준을 충족했습니다. 이 방법은 다중 모달 대형 언어 모델(Multimodal Large Language Models) 및 LLM 기반 에이전트에도 확장될 수 있습니다. 실험 결과, RAG 기반 언러닝은 기존의 세 가지 대표적인 LLM 언러닝 방식보다 뛰어난 성능을 보였습니다.



### Chasing Random: Instruction Selection Strategies Fail to Generaliz (https://arxiv.org/abs/2410.15225)
- **What's New**: 이 연구는 다양한 소스 데이터셋, 선택 예산 및 평가 벤치마크를 통해 인기 있는 선택 전략을 분석하고, 이러한 전략들이 일반화 능력이 부족하다는 것을 보여줍니다. 또한 데이터 선택의 비용-성과 트레이드 오프를 평가하고, 많은 경우 데이터 선택이 전체 데이터셋을 사용하는 것보다 더 높은 비용을 초래한다는 사실을 밝혀냈습니다.

- **Technical Details**: 연구팀은 60개 이상의 실험 구성과 4개의 평가 벤치마크를 통해 수행된 실험에서 수집된 데이터를 분석하였습니다. 중요한 발견으로는 (a) Instruction Selection Strategies(명령어 선택 전략)가 비슷한 실험 설정에 잘 일반화되지 않으며 무작위 선택을 일관되게 초과하지 않음, (b) 일반 목적의 Instruction Following(명령어 수행) 능력은 주관적인 목표며, 따라서 다양한 측면에서 선택 전략을 비교할 경우 상충된 경향을 보일 수 있음, (c) 선택 예산을 증가시키면 많은 전략이 불리하게 확장되며 선택 비용이 전체 데이터셋으로 훈련하는 비용을 초과할 수 있음이 포함되었습니다.

- **Performance Highlights**: 연구 결과는 선택 전략들이 무작위 기준선에 비해 일관되게 높은 성능을 보이지 않음을 시사합니다. 이로 인해 기존의 명령어 선택 전략을 실제 환경에서 적용하기 어렵게 만듭니다. 무작위 샘플링을 통한 간단한 방법이 일정 상황에서 더 비용 효율적일 수 있다는 점도 강조되었습니다.



### The Computational Anatomy of Humility: Modeling Intellectual Humility in Online Public Discours (https://arxiv.org/abs/2410.15182)
- **What's New**: 이번 연구에서는 소셜 미디어 공간에서 여러 차이점을 넘어 서로 건설적으로 소통할 수 있는 능력이 중요하다는 것을 강조합니다. 특히 '지적 겸손'(intellectual humility, IH)이라는 미덕에 초점을 맞춰, 이를 대규모로 측정할 수 있는 컴퓨팅 방법을 개발하였습니다.

- **Technical Details**: 연구진은 350개의 종교 관련 포스트를 수집하여 IH 코드북을 수작업으로 큐레이팅하고 검증했습니다. 이를 바탕으로 LLM(대형 언어 모델) 기반의 자동화 측정 모델을 개발하였으며, 최상의 모델은 Macro-F1 점수 0.64(보다 세부적인 수준에서 IH/IA/Neutral 예측 시 점수 0.70)를 달성했습니다. 이는 기대되는 기본값 0.51(0.32)보다 높은 수치지만, 인간 주석자가 추천하는 상한선 0.85(0.83)보다는 낮습니다.

- **Performance Highlights**: 이 연구는 온라인에서 IH를 탐지하는 것이 도전적임을 보여주며, 새로운 NLP(자연어 처리) 연구 방향으로 나아갈 수 있는 기회를 제시합니다. 또한, 온라인 공적 담론에서 IH를 분석하고 증진하려는 컴퓨팅 사회 과학 연구자들에게 기초 자료를 제공합니다.



### Explaining Graph Neural Networks with Large Language Models: A Counterfactual Perspective for Molecular Property Prediction (https://arxiv.org/abs/2410.15165)
- **What's New**: 최근 Graph Neural Networks (GNNs)이 분자 속성 예측 과제에서 성공을 거둔 사례가 늘어나고 있지만, GNN의 블랙박스 특성 때문에 예측 결과의 신뢰성이 문제가 되고 있다. 이에 따라 Graph Counterfactual Explanation (GCE) 방법이 GNN의 투명성을 향상시키는 유망한 접근법으로 주목받고 있다. 기존 GCE 방법들은 도메인 특화 지식을 고려하지 않는 문제점이 있으며, 이를 해결하기 위해 LLM-GCE라는 새로운 방법을 제안하였다.

- **Technical Details**: LLM-GCE는 대형 언어 모델(LLM)을 활용하여 GNN을 설명하기 위한 새로운 GCE 방법이다. 이 방법은 카운터팩추얼 텍스트 쌍(CTPs)에서 카운터팩추얼 그래프 토폴로지를 생성하기 위해 오토인코더(autoencoder)를 사용한다. 또한, LLM의 헐루시네이션(hallucination) 문제를 완화하기 위한 CTP 동적 피드백 모듈을 도입하였다.

- **Performance Highlights**: 광범위한 실험을 통해 LLM-GCE의 성능이 우수함을 입증하였고, 코드도 공개되었다. LLM-GCE는 LLM의 강력한 추론 능력을 활용하여 더 현실적인 카운터팩추얼을 생성하고, 종합적인 최적화 경로를 제공하는 데 성공하였다.



### Evaluation Of P300 Speller Performance Using Large Language Models Along With Cross-Subject Training (https://arxiv.org/abs/2410.15161)
Comments:
          21 pages, 11 figures, 1 table. arXiv admin note: substantial text overlap with arXiv:2405.13329

- **What's New**: 이번 연구는 근위축성 측삭경화증(Amyotrophic Lateral Sclerosis, ALS) 환자가 P300 스펠러 뇌-컴퓨터 인터페이스(BCI)를 통해 의사소통을 보다 효율적으로 할 수 있도록 개선된 방법을 제시합니다.

- **Technical Details**: P300 스펠러 BCI는 EEG 신호를 기반으로 하여 그래픽 사용자 인터페이스(GUI)에서 문자를 선택하는 방법입니다. 이 연구에서는 고급 언어 모델인 GPT2, BERT, BART 및 Dijkstra 알고리즘을 적용하여 자극을 최적화하고 단어 예측을 개선하였습니다.

- **Performance Highlights**: 연구 결과, 드물거나 OOV(out-of-vocabulary) 단어를 포함하는 텍스트 입력 속도가 10% 개선되었으며, GPT2를 이용한 멀티-단어 예측은 약 40%의 성능 향상을 보여주었습니다. 여러 언어 모델이 이론적 성능 한계 내에서 10% 이내의 결과를 달성했습니다.



### On Designing Effective RL Reward at Training Time for LLM Reasoning (https://arxiv.org/abs/2410.15115)
- **What's New**: 본 연구에서는 LLM(대규모 언어 모델)의 추론 능력을 향상시키기 위한 보상 모델의 잠재력을 조사합니다. 특히, Outcome-supervised Reward Model (ORM)와 Process-supervised Reward Model (PRM) 같은 보상 모델을 RL(강화 학습) 훈련에서 평가하였으며, 기대와 달리 이러한 보상 모델들이 RL 훈련에서 성능을 저하시킬 수 있음을 발견하였습니다.

- **Technical Details**: ORM은 성공 보상을 추정하는 결과 보상을 생성하며, PRM은 올바른 추론 단계를 구분하는 훈련을 통해 단계별 프로세스 보상을 제공합니다. 연구에서는 PPO(정책 생산자 최적화)를 사용하여 MATH 및 GSM8K 벤치마크에서 훈련된 LLM에 이 보상 모델들을 적용했습니다. 그러나 보상 해킹 문제를 해결하기 위해 새로운 보상 정제 기술인 Clipping과 Delta를 도입하여 보상 극대화를 방지하였습니다.

- **Performance Highlights**: 평가 결과, 제안된 정제 기술은 1.5B 및 7B LLM을 RL 훈련하는 데 있어 안정성을 높였으며, 실험에서 Qwen2.5-Math-7B-Instruct와 같은 최첨단 모델이 MATH와 GSM8K 벤치마크에서 개선되었습니다.



### Towards Safer Heuristics With XPlain (https://arxiv.org/abs/2410.15086)
- **What's New**: 이 논문에서는 cloud 운영자가 컴퓨팅 성능이 저하되는 경우를 더 깊게 이해할 수 있도록 도와주는 새로운 도구인 XPlain을 제안합니다. 기존의 heuristic 분석 도구들은 성능 저하가 발생하는 입력 인스턴스만을 찾아내지만, XPlain은 그 이유와 범위를 분석하여 문제 해결에 도움을 제공합니다.

- **Technical Details**: XPlain은 도메인 특정 언어를 사용하여 분석할 heuristics와 비교할 벤치마크를 매개변수로 활용합니다. 이 언어는 네트워크 흐름 추상화에 뿌리를 두고 있으며, 이를 통해 운영자들이 사용하는 다양한 heuristics의 동작을 모델링할 수 있습니다. XPlain의 컴파일러는 이 언어의 입력을 기존 heuristic 분석기로 변환하고, 효율적인 반복 알고리즘을 통해 성능 저하의 모든 원인과 그 지역을 찾아냅니다.

- **Performance Highlights**: 이 논문의 초기 결과는 XPlain이 existing heuristic analyzers의 한계를 극복할 수 있는 가능성을 보여줍니다. 특히, heuristic의 성능 저하 이유를 명확히 하고, 그 결과를 시각화하여 보다 나은 이해를 돕는데 기여할 수 있습니다.



### ChitroJera: A Regionally Relevant Visual Question Answering Dataset for Bangla (https://arxiv.org/abs/2410.14991)
- **What's New**: 본 논문에서는 Bangla 언어를 위한 대규모 Visual Question Answering (VQA) 데이터셋인 ChitroJera를 소개합니다. 이 데이터셋은 15,000개 이상의 샘플로 구성되어 있으며, 다양한 현지 데이터 소스를 활용하여 문화적 관련성을 강조합니다.

- **Technical Details**: ChitroJera 데이터셋은 다양한 텍스트 인코더(text encoders), 이미지 인코더(image encoders), 멀티모달 모델(multimodal models), 그리고 새로운 dual-encoder 모델을 평가하는 데 사용되었습니다. 실험 결과, 사전 훈련된 dual-encoder가 다른 모델들보다 우수한 성능을 보임을 확인했습니다.

- **Performance Highlights**: 대형 언어 모델(large language models, LLMs)을 활용한 프롬프트 기반 기법에서 LLM들이 가장 뛰어난 성능을 기록했습니다. 기존 데이터셋의 미비한 상태를 고려할 때, ChitroJera는 Bangla에서 시각-언어 작업의 범위를 넓힐 것으로 기대됩니다.



### Do Large Language Models Truly Grasp Mathematics? An Empirical Exploration (https://arxiv.org/abs/2410.14979)
- **What's New**: 이 연구는 최근 LLM(대형 언어 모델)의 수학적 추론 능력의 기초 메커니즘에 대해 토대로 하고 있으며, CoT(Chain-of-Thought) 프로프트의 효과를 조명합니다. LLM이 일반적으로 인지적 반사를 평가하는 과제에서 높은 오류율을 보인 것을 확인하였습니다.

- **Technical Details**: 연구팀은 CoT 프로프트를 사용하여 LLM이 인지적 반사 테스트(Cognitive Reflection Test, CRT)에서 오류를 줄일 수 있는지를 조사했습니다. CRT 문제를 수정하여 LLM의 수학적 사고 능력을 실험적으로 평가하였으며, 그 결과 LLM은 여전히 높은 오류율을 유지했습니다. 데이터 세트에 대한 수정 유형 A, B, C, D의 정확도를 분석하였습니다.

- **Performance Highlights**: 각 종류별 문제의 성능 분석 결과, 수정된 문제에 대한 LLM의 성능은 기존보다 낮았으며, 특히 타입 D 문제에서는 GPT-4 모델조차도 평균 정확도가 29.33%에 불과했습니다. 일반적인 LLM은 계산 단계에서의 오류가 지배적이었으며, 초점은 시스템 2(논리적 추론) 능력 부족에 있음을 암시합니다.



### BrainECHO: Semantic Brain Signal Decoding through Vector-Quantized Spectrogram Reconstruction for Whisper-Enhanced Text Generation (https://arxiv.org/abs/2410.14971)
- **What's New**: 최근 EEG 및 MEG 신호로부터 언어를 해독하는 기술이 발전하며, BrainECHO라는 새로운 다단계 전략이 제안되었습니다. BrainECHO는 전이 학습된 언어 모델을 활용하여 텍스트 생성 성능을 혁신적으로 향상시킵니다.

- **Technical Details**: BrainECHO는 1) 오디오 스펙트로그램의 이산 자동 인코딩, 2) 뇌-오디오 잠재 공간 정렬, 3) Whisper 모델의 파인 튜닝을 통한 의미론적 텍스트 생성을 포함하는 3단계 과정으로 구성되어 있습니다. 이 과정을 통해 BrainECHO는 EEG 및 MEG 데이터셋에서 최첨단 성능을 달성합니다.

- **Performance Highlights**: BrainECHO는 기존 방법들보다 향상된 성능을 보이며, 언어 기반 뇌-컴퓨터 인터페이스(BCI)에 중요한 진전을 제공합니다. 특히, 문장, 세션 및 주제 간 독립적인 평가에서 강력한 견고성을 демонстр합니다.



### Baichuan Alignment Technical Repor (https://arxiv.org/abs/2410.14940)
- **What's New**: Baichuan Alignment에 대한 포괄적이고 체계적인 설명이 제공됩니다. 이 기술은 LLM의 성능 향상을 위한 주요 요소인 최적화, 데이터 전략, 능력 강화 및 평가 과정의 세부 사항을 다룹니다.

- **Technical Details**: Baichuan Alignment는 세 가지 주요 단계로 구성됩니다: Prompt Augmentation System (PAS), Supervised Fine-Tuning (SFT), Preference Alignment. 이 과정에서 모델이 사용자 쿼리를 보다 이해하기 쉽고 실행 가능한 명령으로 변환하며, 대화에 능숙하고 복잡한 작업을 처리할 수 있도록 다양한 고품질 데이터를 사용하여 훈련됩니다.

- **Performance Highlights**: Baichuan-Instruct 모델은 코어 기능에서 17%에서 28%의 개선을 보이며, Qwen2-Nova-72B 및 Llama3-PBM-Nova-70B 모델은 각각 관련 공식 버전을 초과하여 여러 데이터셋에서 뛰어난 성능을 발휘합니다.



### A Hybrid Defense Strategy for Boosting Adversarial Robustness in Vision-Language Models (https://arxiv.org/abs/2410.14911)
- **What's New**: 본 연구에서는 여러 공격 전략과 고급 머신러닝 기법을 통합하여 Vision-Language Models (VLMs)의 강인성을 크게 향상시키는 새로운 적대적 훈련 프레임워크를 제안합니다.

- **Technical Details**: 기존의 적대적 훈련 방법들은 FGSM, AutoAttack, DeepFool과 같은 모델을 사용하여 적대적 예제 생성에 초점을 맞추었으며, 고정된 왜곡 노름(fixed perturbation norms)이나 미리 정의된 공격 패턴(predefined attack patterns)과 같은 강력한 가정에 의존했습니다. 제안된 방법은 다양한 적대적 공격에 대해 VLM의 강인성을 크게 향상시킵니다.

- **Performance Highlights**: CIFAR-10 및 CIFAR-100과 같은 실제 데이터셋에서 실험한 결과, 제안된 방법은 모델의 강인성을 크게 향상시켰으며, 세밀하게 조정된 CLIP 모델은 적대적으로 왜곡된 이미지에서 43.5%의 정확도를 달성했습니다. 반면, 기준 모델은 4%에 그쳤습니다. 또한, 신경망 모델은 98%의 높은 정확도를 기록하였고, XGBoost 모델은 예측 작업에서 85.26%의 성공률을 달성했습니다.



### Class-RAG: Content Moderation with Retrieval Augmented Generation (https://arxiv.org/abs/2410.14881)
Comments:
          11 pages, submit to ACL

- **What's New**: 이번 연구에서는 콘텐츠 모더레이션(content moderation)을 위한 새로운 분류 방법인 Retrieval-Augmented Generation (Class-RAG)을 제안합니다. 이는 안전하지 않은 입력과 안전한 입력 사이의 미세한 차이를 극복하여 더 나은 분류 성능을 제공합니다.

- **Technical Details**: Class-RAG 시스템은 임베딩 모델(embedding model), 검색 라이브러리(retrieval library), 검색 모듈(retrieval module), 그리고 미세 조정된 LLM(Classifier)으로 구성됩니다. 사용자가 쿼리를 입력하면, 가장 유사한 부정적 및 긍정적 예제를 검색하여 해당 컨텍스트 정보를 분류기(classifier)에 추가합니다.

- **Performance Highlights**: Class-RAG는 전통적인 모델에 비해 분류 성능이 우수하며, 적대적 공격(adversarial attack)에 강한 내성을 보여줍니다. 또한, 검색 라이브러리의 크기를 확장하면 성능이 향상되며, 이는 낮은 비용으로 분류 성능을 높일 수 있는 실용적인 방법으로 제시됩니다.



### How to Evaluate Reward Models for RLHF (https://arxiv.org/abs/2410.14872)
- **What's New**: 본 논문에서는 RLHF(고객 피드백으로부터의 강화 학습)를 통해 강력한 언어 모델을 생성할 수 있는 보상 모델의 능력을 정량화하기 위한 새로운 벤치마크를 소개합니다. 기존의 gold-standard 접근법은 전체 RLHF 학습 파이프라인을 실행하고 직접적으로 다운스트림 LLM 성능을 평가하는 것입니다. 하지만 이는 비용이 과도하게 비쌉니다. 이를 해결하기 위해, 다운스트림 LLM 성능을 예측하는 모델을 구축하였으며, 다양한 종류의 프록시 작업을 통해 보상 모델을 평가합니다.

- **Technical Details**: 저자들은 12개의 다양한 도메인에서 12개의 메트릭을 평가하여 보상 모델의 성능을 측정합니다. 프록시 작업으로는 대규모 인간 선호 및 검증 가능한 정답 선호 데이터셋을 활용하였으며, 이 작업에서 수집된 데이터와 발견 사항을 Preference Proxy Evaluations (PPE)라는 형태로 정리하였습니다. PPE는 RLHF 이후의 실제 인간 선호 성과와 명시적으로 연결된 최초의 보상 모델 벤치마크입니다.

- **Performance Highlights**: PPE는 20개의 주요 LLM과 121개 이상의 언어에서 수집된 16,038개의 레이블된 인간 선호 쌍과 2,555개의 프롬프트 데이터셋을 포함하고 있습니다. 각 프롬프트는 32개의 서로 다른 응답 옵션이 제공되어 총 81,760개의 응답으로 구성됩니다. PPE는 12개의 다른 메트릭에서 보상 모델을 평가하며, RLHF 결과와 직접적으로 연결된 유일한 보상 모델 벤치마크로 자리잡고 있습니다.



### Making LLMs Vulnerable to Prompt Injection via Poisoning Alignmen (https://arxiv.org/abs/2410.14827)
- **What's New**: 이번 연구에서는 Prompt Injection Attack의 효율성을 증가시키기 위해 Alignment 과정에 독성 샘플을 주입하는 방법을 제안합니다. 이는 기존의 공격 방식과는 다른 접근법으로, 대형 언어 모델(LLM)의 취약점을 이용합니다.

- **Technical Details**: 우리는 PoisonedAlign라는 새로운 방법을 소개하며, 이는 공격자가 주입 프롬프트를 사용하여 독성 샘플을 생성하도록 합니다. 이 방법은 Alignment 데이터의 일부가 독성으로 변모할 때 LLM이 Prompt Injection에 더 취약해지도록 합니다.

- **Performance Highlights**: PoisonedAlign 방법을 활용했을 때, 10%의 Alignment 데이터가 독성으로 변모되면 LLM의 공격 성공률이 평균 0.33 증가하며, 이는 LLM의 기초 능력을 유지한 채 이루어집니다.



### What's New in My Data? Novelty Exploration via Contrastive Generation (https://arxiv.org/abs/2410.14765)
- **What's New**: 이 연구에서는 데이터 세트를 직접 검사할 수 없는 상황에서 Fine-tuning 데이터 세트의 새로운 특성을 식별하는 작업인 'novelty discovery through generation'을 소개합니다.

- **Technical Details**: 우리는 Contrastive Generative Exploration (CGE)라는 접근 방식을 제안합니다. CGE는 사전 훈련된 모델과 Fine-tuning된 모델의 예측을 대조하여 새로운 특성을 강조한 예를 생성합니다. CGE는 또한 반복적인 방식으로 업데이트되어 다양한 출력을 촉진합니다.

- **Performance Highlights**: CGE는 박스 언어, 유해한 언어 및 새로운 자연어와 프로그래밍 언어 감지에서 효과적인 성능을 입증했습니다. Differential privacy 기술을 사용하여 Fine-tuning된 모델에서도 일관된 성능을 보여주었습니다.



### Collaboratively adding new knowledge to an LLM (https://arxiv.org/abs/2410.14753)
- **What's New**: 이 논문은 LLM(대형 언어 모델)에 새로운 지식을 추가하면서 기존 지식을 보존하는 방법에 대한 연구를 다루고 있습니다. Semi-cooperative 및 Fully-cooperative 두 가지 설정에서의 성능을 분석하고, LoRA(저순위 적응)가 전통적인 전체 정밀 조정(full fine-tuning)보다 효과적임을 보여줍니다.

- **Technical Details**: 논문에서는 전통적인 전체 정밀 조정(FFT)과 LoRA를 통한 파라미터 효율적 조정(PEFT)을 고려합니다. Semi-cooperative 설정에서는 데이터셋이 훈련 후에 사용 불가능하지만 MOE 혼합, 모델 병합, LoRA 기반의 직교 서브스페이스 순차 학습이 잘 작동합니다. Fully-cooperative 설정에서는 데이터셋이 사용 가능하며, 병합 훈련과 재생(replay)을 통한 순차 훈련이 효과적입니다.

- **Performance Highlights**: LoRA는 지식 습득 측면에서 FFT보다 적은 성능 저하를 보이지만, 이전 지식을 보존하는 데에는 더 유리한 성능을 나타냅니다. 논문은 실험 결과를 바탕으로 LoRA가 전체 매개변수 조정보다 더 나은 선택이 될 수 있음을 강조합니다.



### TimeSeriesExam: A time series understanding exam (https://arxiv.org/abs/2410.14752)
Comments:
          Accepted at NeurIPS'24 Time Series in the Age of Large Models Workshop

- **What's New**: 이번 논문에서는 대형 언어 모델(Large Language Models, LLMs)의 시간 시계열 데이터 이해도를 평가하기 위해 TimeSeriesExam이라는 새로운 시험 시스템을 도입합니다. 이 시스템은 5개의 핵심 시간 시계열 이해 카테고리(패턴 인식, 잡음 이해, 유사성 분석, 이상 탐지, 인과 관계 분석)를 평가하는 700개 이상의 객관식 질문으로 구성되어 있습니다.

- **Technical Details**: TimeSeriesExam은 104개의 면밀하게 설계된 템플릿을 사용하여 절차적으로 생성된 질문을 기반으로 하며, 각 질문은 Item Response Theory (IRT)를 통해 난이도 조정 및 모델 차별화를 위해 세밀하게 조정됩니다. 시험은 LLM의 기본 시간 시계열 개념 이해도를 평가하며, 특히 패턴과 노이즈 개념, 이상 탐지 및 인과 관계 분석과 같은 다양한 추론 작업을 포함합니다.

- **Performance Highlights**: 시험 결과, GPT-4 및 Gemini와 같은 폐쇄형 모델이 개방형 대안보다 기본 시간 시계열 개념에 대해 이해도가 더 높다는 사실이 드러났습니다. 그러나 모든 모델은 인과 관계 분석과 같은 복잡한 개념에서는 어려움을 겪었습니다.



### ETF: An Entity Tracing Framework for Hallucination Detection in Code Summaries (https://arxiv.org/abs/2410.14748)
Comments:
          11 pages, 6 Figures, 5 Tables

- **What's New**: 본 연구에서는 코드 요약에서의 환각(hallucination) 탐지를 위해 최초로 10,000개의 샘플을 포함하는 데이터세트를 소개하며, 새로운 Entity Tracing Framework (ETF)를 제안합니다. 이 프레임워크는 코드에서 엔티티를 식별하고, 생성된 요약 내에서 이 엔티티들의 의도를 검증하는 새로운 접근 방식을 제공합니다.

- **Technical Details**: Entity Tracing Framework (ETF)는 a) 정적 프로그램 분석(static program analysis)을 이용하여 코드 엔티티를 식별하고 b) LLM을 사용하여 이 엔티티와 생성된 코드 요약 내의 의도를 매핑 및 검증합니다. 이 연구는 코드의 엔티티를 기반으로 한 요약 검증의 중요성을 강조하며, 0.73 F1 점수로 효과성을 입증합니다.

- **Performance Highlights**: 본 연구의 프레임워크는 코드 요약에서 환각 탐지를 위한 해석 가능한 방법을 제공하며, 모델이 생성한 문서의 정확성을 평가할 수 있게 합니다. 우리는 411개의 요약과 10,000개 엔티티 샘플을 포함하는 데이터세트를 오픈 소스할 계획입니다.



### Knowledge Graph Embeddings: A Comprehensive Survey on Capturing Relation Properties (https://arxiv.org/abs/2410.14733)
Comments:
          22 pages, 8 figures, 3 tables, this paper is a modified English version of our article already published in Computer Science journal (in Chinese), released to facilitate communication among international researchers in the relevant fields

- **What's New**: 본 논문은 KGE (Knowledge Graph Embedding) 기법에서 관계의 복잡한 매핑 속성을 다루며, 다양한 관계 패턴을 캡처하기 위한 모델들을 종합적으로 검토합니다. 또한, Sparse 및 Dynamic KGs에 대해 향후 연구 방향을 제시합니다.

- **Technical Details**: 관계의 매핑 속성으로는 one-to-one, one-to-many, many-to-one, many-to-many가 있으며, 이러한 관계를 모델링하기 위한 다양한 방법이 논의됩니다. 여기에는 수정된 텐서 분해(modified tensor decomposition), 수정된 관계 인식 매핑(modified relation-aware mappings), 회전 연산(rotation operations)을 이용한 모델들이 포함됩니다. 또한, 보조 정보(auxiliary information)를 포함하는 모델, 쌍곡선 공간(hyperbolic spaces)을 기반으로 한 모델, 극좌표계(polar coordinate system)를 이용한 모델도 검토됩니다.

- **Performance Highlights**: 각 모델의 성능은 지식 그래프의 다양한 구조를 효과적으로 포착하는 능력에 따라 결정되며, 향후 연구는 멀티모달 정보(multi-modal information)의 통합, 규칙(rule) 기반의 관계 패턴 모델링, 동적 KGE 환경에서의 관계 특성을 캡처하는 모델 개발을 포함합니다.



### MatryoshkaKV: Adaptive KV Compression via Trainable Orthogonal Projection (https://arxiv.org/abs/2410.14731)
- **What's New**: 본 논문에서는 기존 연구들이 주로 KV 캐시의 첫 세 축에 초점을 맞추었던 것과 달리, 특징 차원 축(feature dimension axis)에서의 압축 방법을 새롭게 제안하고 있습니다. 저자들은 저랭크 프로젝션 행렬(low-rank projection matrices)을 활용하여 KV 캐시의 효율성을 높이고 성능 저하를 최소화하는 방법을 모색하였습니다.

- **Technical Details**: 연구진은 PCA(주성분 분석)를 통해 압축을 시작하였고, 오르소고날 프로젝션 매트릭스(orthogonal projection matrices)를 조정하여 모델의 출력을 보존합니다. 이 과정에서 매트리요시카 훈련 전략(Matryoshka training strategy)을 도입해 다양한 계층과 헤드에 대해 최적의 압축 비율(compression rates)을 탐색합니다. 이는 KV 캐시의 효율성을 극대화하면서도 이름 있는 LLM들, 예를 들어 LLaMA2-7B-base 모델에 통합할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, MatryoshkaKV 방법은 LLaMA2-7B-base와 Mistral-7B-v0.3-base와 같은 LLM에서 90% 이상의 성능을 유지하면서 평균 60%의 KV 캐시 압축률을 달성했음을 보여주었습니다. 특정 극단적인 상황에서는 압축률이 75%에 달하기도 하며, 수치적으로 이는 이전 연구들보다 뛰어난 데이터 효율성을 입증합니다.



### Tokens on Demand: Token Condensation as Training-free Test-time Adaptation (https://arxiv.org/abs/2410.14729)
Comments:
          18 pages, 7 figures

- **What's New**: 이 논문에서는 Token Condensation as Adaptation (TCA)를 소개합니다. TCA는 비전-언어 모델 (VLMs)이 테스트 시간 추론 중 겪는 분포 변화(distribution shifts)를 완화하기 위해 설계된 비훈련(training-free) 방식입니다.

- **Technical Details**: TCA는 <cls> 토큰에 대한 낮은 주의(attentiveness)를 보이는 이미지 토큰을 응축(condensing)하여 패치 레벨에서 분포 간극(distribution gaps)을 연결합니다. TCA는 역사적 데이터 스트림에서 특정 대상 클래스(target classes)와 정렬된 가장 신뢰할 수 있는 <cls> 토큰을 식별하고 추적합니다. 이를 위해, TCA는 불확실성이 가장 낮은 토큰을 '앵커(anchors)'로 보존하기 위한 컨텍스트 토큰 저수지(context token reservoir, CTR)를 제안합니다. CTR에서 샘플링한 앵커를 사용하여 TCA는 클래스와 무관한 토큰을 깎고(pruning) 나머지 클래스 애매한 토큰을 대표 센터로 병합(merging)합니다.

- **Performance Highlights**: TCA는 테스트 시간 적응(test-time adaptation)에서 토큰 효율성을 탐구하는 최초의 방법으로, 크로스 데이터셋(cross-dataset) 및 분포 외(out-of-distribution) 적응 작업에서 일관되게 우수한 성능을 보입니다. GFLOPs를 12.2%에서 48.9%까지 줄이면서 가장 강력한 기준선(baseline) 대비 정확도(accuracy)를 최대 21.4% 향상시킵니다.



### Rethinking Token Reduction for State Space Models (https://arxiv.org/abs/2410.14725)
Comments:
          EMNLP 2024

- **What's New**: 최근 State Space Models (SSMs)의 발전으로 인한 관심이 높아지고 있으며, 특히 병렬 학습과 장기 의존성 처리를 위해 최적화된 모델에 주목하고 있습니다. Mamba와 같은 아키텍처는 선택적 SSM (Selective SSM)으로 수십억 개의 파라미터를 원활하게 확장할 수 있습니다. 본 논문에서는 Mamba의 효율성을 탐색하고, 기존의 방법을 적용했을 때 발생하는 성능 저하의 원인을 분석하여 이에 맞춘 새로운 방법론을 제안합니다.

- **Technical Details**: 본 논문에서는 토큰 감소(token reduction) 기술을 SSMs에 적용했을 때 성능 저하이유를 분석하고, 이를 해결하기 위해 통합된 사후학습(token reduction post-training) 방법을 제안합니다. 이 방법은 토큰의 중요성과 유사성을 고려하여 미세조정(intra-layer)하는 방식을 사용하여 Pruning과 Merging의 장점을 통합합니다.

- **Performance Highlights**: 제안된 방법은 Mamba-2 모델을 사용하여 여섯 가지 벤치마크에서 평균 정확도를 5.7%에서 13.1% 향상시켰으며, 기존 방법들과 비교하여 계산 효율성과 메모리 요구사항이 크게 감소했습니다.



### A Systematic Survey on Large Language Models for Algorithm Design (https://arxiv.org/abs/2410.14716)
- **What's New**: 이번 논문은 LLM4AD(Algorithm Design을 위한 Large Language Models)의 발전을 다루며, 기존의 연구를 체계적으로 검토하여 통합적인 시각을 제공합니다. 3년간의 연구 결과를 정리하고, 알고리즘 설계 과정에서의 LLMs의 역할과 사용된 전략들에 대해 재조명합니다.

- **Technical Details**: LLM4AD는 네 가지 주요 차원에서 연구를 분류합니다: 1) LLM의 역할; 2) 검색 방법; 3) 프롬프트 방법; 4) 응용 분야. 이러한 다차원 분류법을 통해 각 연구의 기여와 발전을 명확히 하고 향후 연구의 기회와 갭을 식별합니다.

- **Performance Highlights**: 논문은 180편 이상의 연구를 종합하여 LLM4AD의 Current 상태를 파악하고, 주요 도전 과제와 가능성을 논의합니다. 또한, LLM 기술의 발전으로 인해 알고리즘 설계 과정에서의 창의성과 효율성이 향상될 수 있음을 기대합니다.



### QuAILoRA: Quantization-Aware Initialization for LoRA (https://arxiv.org/abs/2410.14713)
Comments:
          12 pages, 7 figures. Submitted to the 4th NeurIPS Workshop on Efficient Natural Language and Speech Processing (ENLSP-IV)

- **What's New**: QuAILoRA는 LoRA에 대한 양자화 인식 초기화 방법을 도입하여 양자화 오류를 줄이고 대형 언어 모델의 성능을 향상시킵니다.

- **Technical Details**: QuAILoRA는 LoRA 매트릭스의 양자화 인식 초기화를 통해 QLoRA 모델의 초기 입력-출력 매핑을 고정밀 베이스 모델과 유사하게 초기화합니다. 이 방법은 대각형 행렬을 이용하여 양자화에 따라 발생하는 오류를 줄이는 것을 목표로 하며, 최소한의 계산 자원만으로 초기화를 수행할 수 있습니다.

- **Performance Highlights**: QuAILoRA를 적용한 4비트 QLoRA 모델은 8비트로의 양자화 정밀도를 두 배로 증가시켰을 때와 유사한 정도로 검증 PERPLEXITY 감소율이 75%에 달하고, 하류 작업에서의 정확도 증가율은 86%에 이릅니다. 이는 양자화 오류의 부정적인 영향을 비례하여 개선한 결과입니다.



### Polymath: A Challenging Multi-modal Mathematical Reasoning Benchmark (https://arxiv.org/abs/2410.14702)
Comments:
          49 pages, (10 pages paper, 9 pages references, 30 pages appendix)

- **What's New**: Multi-modal Large Language Models (MLLMs)에 대한 새로운 벤치마크인 PolyMATH가 제시되었으며, 이는 인지적 추론 능력을 평가하기 위한 도전적인 기준입니다.

- **Technical Details**: PolyMATH는 10가지 범주에서 집합한 5,000개의 고품질 이미지를 포함하며, 패턴 인식, 공간 추론, 상대적 추론을 포괄합니다. 15개의 MLLM을 네 가지 다양한 프로토타이핑 전략을 사용하여 종합적이고 정량적으로 평가했습니다.

- **Performance Highlights**: Claude-3.5 Sonnet은 약 41%, GPT-4o는 약 36%, Gemini-1.5 Pro는 약 27%의 점수를 기록했습니다. 이 모델들은 공간 관계 이해 및 고수준 추론 수행에서 어려움을 겪고 있음이 드러났습니다.



### BrainTransformers: SNN-LLM (https://arxiv.org/abs/2410.14687)
- **What's New**: 이 연구는 Spiking Neural Networks (SNNs)을 활용하여 구현된 혁신적인 Large Language Model (LLM)인 BrainTransformers를 소개합니다. SNN에 호환되는 Transformer 구성 요소 설계, SiLU 활성화 함수의 SNN 근사 구현, 그리고 시냅스 가소성을 시뮬레이션하는 Synapsis 모듈 개발이 주요 기여입니다.

- **Technical Details**: BrainTransformers-3B-Chat 모델은 30억 개의 매개변수를 가지고 있으며, SNN 특정 신경 시냅스 가소성 훈련을 포함한 3단계 훈련 접근 방식을 사용합니다. 주요 구성 요소로는 SNNMatmul, SNNSoftmax 및 SNNSiLU가 있습니다.

- **Performance Highlights**: 모델은 MMLU(63.2), BBH(54.1), ARC-C(54.3), GSM8K(76.3) 등 다양한 벤치마크에서 경쟁력 있는 성능을 보여주며, 에너지 효율성과 생물학적 타당성을 개선할 가능성이 있습니다.



### RepoGraph: Enhancing AI Software Engineering with Repository-level Code Graph (https://arxiv.org/abs/2410.14684)
Comments:
          Work in progress

- **What's New**: 본 연구에서 제안하는 RepoGraph는 LLM(대형 언어 모델)에 기반한 AI 프로그래머가 코드 저장소를 전체적으로 활용할 수 있도록 설계된 플러그인 모듈입니다.

- **Technical Details**: RepoGraph는 저장소 내 코드 구조를 이해하기 위해 그래프 구조를 사용하며, 각 노드는 코드의 한 줄을 나타내고, 간선은 코드 정의와 참조 간의 의존성을 나타냅니다. 이를 통해 LLM이 보다 세부적인 맥락을 파악할 수 있도록 돕습니다.

- **Performance Highlights**: RepoGraph를 기존의 네 가지 소프트웨어 공학 프레임워크와 통합하여 성능을 평가한 결과, 에이전트 및 절차적 프레임워크 모두에서 평균 32.8% 향상된 성공률을 보였습니다.



### Paths-over-Graph: Knowledge Graph Empowered Large Language Model Reasoning (https://arxiv.org/abs/2410.14211)
- **What's New**: Paths-over-Graph (PoG) 방법은 Knowledge Graphs (KGs)에서 지식 추론 경로를 통합하여 LLM의 추론을 향상시키는 혁신적인 접근 방식입니다. 이 방법은 복잡한 멀티-홉 및 멀티-엔터티 질문을 처리하며, LLM의 본질적인 지식과 KGs의 사실적 지식을 결합합니다.

- **Technical Details**: PoG는 세 가지 단계의 동적 멀티-홉 경로 탐색을 통해 LLM의 사고 지표인 'Planning'을 생성하고, 이 지표를 기반으로 질문에서 주제 엔터티를 추출하여 복잡한 질문을 하위 질문으로 분해합니다. 경로 탐색 과정에서는 KG의 구조를 활용하고, 그래프 구조를 통합한 효과적인 세 단계의 가지치기 기법을 사용하여 탐색된 후보 경로를 효율적으로 줄입니다.

- **Performance Highlights**: PoG는 다섯 개의 벤치마크 KGQA 데이터셋에 대한 포괄적인 실험에서 기존의 최고 성능 방법인 ToG를 초월하여 GPT-3.5-Turbo 및 GPT-4에서 평균 18.9%의 정확도 향상을 달성했습니다. 특히, PoG는 GPT-3.5-Turbo로 ToG의 GPT-4를 최대 23.9% 초과했습니다.



New uploads on arXiv(cs.IR)

### Improving Pinterest Search Relevance Using Large Language Models (https://arxiv.org/abs/2410.17152)
Comments:
          CIKM 2024 Workshop on Industrial Recommendation Systems

- **What's New**: Pinterest Search의 검색 관련성을 개선하기 위해 대형 언어 모델(LLM)을 검색 모델에 통합하였습니다. 이를 통해 Pins의 관련성을 효과적으로 예측하는 텍스트 표현을 leverage 하였습니다.

- **Technical Details**: 우리는 검색 쿼리와 generative visual language model에서 추출한 캡션을 포함한 콘텐츠 표현을 결합하여 사용합니다. 또한 링크 기반 텍스트 데이터, 사용자 큐레이션 보드, Pin 제목 및 설명 등을 포함하여 모델을 강화하였습니다. 반자동 학습(semi-supervised learning) 접근 방식을 사용하여 데이터 양을 확장합니다.

- **Performance Highlights**: 최종 배포 시스템에서의 성능 향상을 실험적으로 검증하였으며, 다국어 LLM을 활용하여 보이지 않는 언어와 분야를 포함할 수 있도록 시스템이 확장되었습니다.



### Neural Collaborative Filtering Classification Model to Obtain Prediction Reliabilities (https://arxiv.org/abs/2410.16838)
Comments:
          9 pages, 7 figures

- **What's New**: 이 논문에서는 추천 시스템(recommender systems) 분야에서 신경 협업 필터링(neural collaborative filtering)의 새로운 접근법을 제안합니다. 기존의 회귀 기반(regression-based) 모델 대신 분류 기반(classification-based) 접근 방식을 사용하여 평가 예측(rating predictions)과 신뢰성(prediction reliabilities) 모두를 제공합니다.

- **Technical Details**: 제안된 신경 아키텍처(neural architecture)는 평가 예측의 신뢰성을 함께 제공하여 사용자가 추천의 신뢰도를 이해하고 탐색할 수 있는 도구로 활용될 수 있습니다. 이러한 정보는 쉴링 공격(shilling attacks) 탐지, 추천 설명 및 사용자와 항목 간의 의존성(vertical dependencies) 표시 등의 다양한 협업 필터링 영역에서 활용될 수 있습니다.

- **Performance Highlights**: 실험은 네 개의 널리 사용되는 공개 데이터셋을 사용하여 수행되었으며, 제안된 아키텍처는 기존의 최첨단 기준과 유사한 추천 품질을 유지하면서도 개별 평가 예측 품질을 향상시킵니다. 일반화 가능한 결과를 보여주며, 제안된 아키텍처는 개인 평가 예측의 질을 개선하고 추천 결과를 유지하며, 여러 협업 필터링 분야로의 확장 가능성을 열어줍니다.



### Bridging Search and Recommendation in Generative Retrieval: Does One Task Help the Other? (https://arxiv.org/abs/2410.16823)
Comments:
          Accepted for publication in the 18th ACM Conference on Recommender Systems (RecSys'24)

- **What's New**: 이 논문은 검색(Search) 및 추천(Recommendation)을 위한 통합 생성형 검색 모델의 효과성을 살펴보며, 전통적인 작업 특정 모델 대비 장점을 조사한다.

- **Technical Details**: 생성형 검색(Generative Retrieval)은 대규모 언어 모델(LLM)을 활용하여 쿼리와 아이템 ID를 직접 연결하는 새로운 방법론이다. 본 연구에서는 모델의 조합 훈련(Joint Training)이 아이템의 잠재 표현과 인기도 추정을 정규화한다고 가정하며, 두 가지 가설을 설정하여 실험을 진행하였다: [H1] 조합 훈련이 아이템의 인기도 예측을 정규화하고, [H2] 조합 훈련이 아이템의 잠재 표현을 정규화한다. 대규모 데이터 및 실제 데이터에서 이 가설을 지원하는 결과를 얻었다.

- **Performance Highlights**: 생성형 검색 모델이 작업 특정 모델보다 16% 향상된 효과를 보였으며, 아이템의 잠재 표현에 대한 정규화 효과가 Joint Generative Model과 작업 특정 모델 간의 예측 차이를 만들었다.



### Coarse-to-fine Dynamic Uplift Modeling for Real-time Video Recommendation (https://arxiv.org/abs/2410.16755)
Comments:
          9 pages, 4 figures, 5 tables

- **What's New**: 이 논문은 짧은 비디오 플랫폼의 성장과 이에 따른 비디오 추천 기술의 복잡한 도전에 대응하기 위해, 온라인 마케팅에서의 업리프트 모델링 기법을 비디오 추천 시나리오에 적용하는 방법을 제안합니다. Coarse-to-fine Dynamic Uplift Modeling (CDUM)이라는 새로운 접근법을 통해 사용자 장기 선호도와 실시간 관심을 모델링하는 새로운 방법을 제시합니다.

- **Technical Details**: CDUM은 Coarse-grained Preference Modeling (CPM) 및 Fine-grained Interest Capture (FIC) 두 가지 모듈로 구성되어 있습니다. CPM은 오프라인 사용자 특성을 활용해 사용자의 장기 선호도를 모델링하며, FIC는 온라인 실시간 컨텍스트 특성과 요청 수준 후보를 이용해 사용자의 현재 관심사를 모델링합니다. 두 모듈은 동적으로 특정 사용자 그룹을 식별하고 효과적으로 처리를 적용합니다.

- **Performance Highlights**: CDUM은 Kuaishou 플랫폼에 배포되어 매일 수억 명의 사용자에게 서비스를 제공하며, 오프라인 성능이 뛰어난 것뿐 아니라 소비 지표와 사용자 유지율에서 현저한 개선을 이루었습니다. A/B 테스트 결과, 추천 시스템의 효과성 및 사용자 경험을 향상시켰음을 보여주었습니다.



### STAR: A Simple Training-free Approach for Recommendations using Large Language Models (https://arxiv.org/abs/2410.16458)
- **What's New**: 최근 대규모 언어 모델(LLMs)의 발전은 추천 시스템(RecSys) 작업을 위한 유망한 새로운 접근 방법을 제시합니다. 본 논문에서는 LLM을 활용하여 세부 조정(fine-tuning) 없이 다양한 추천 작업에 적용할 수 있는 Simple Training-free Approach for Recommendation (STAR) 프레임워크를 제안합니다.

- **Technical Details**: STAR 프레임워크는 두 가지 주요 단계로 구성됩니다: Retrieval 단계(후보 아이템 검색) 및 Ranking 단계(아이템 순위 조정). Retrieval 단계에서는 LLM의 의미적 임베딩(semantic embedding)을 사용하여 후보 아이템을 검색하고, Ranking 단계에서는 LLM의 추론 능력을 활용하여 후보 아이템의 순위를 조정합니다. 이는 시맨틱(similarity)과 협력적 정보(collaborative information)를 결합하여 이루어집니다.

- **Performance Highlights**: Amazon Review 데이터셋에서 실험한 결과, STAR 방법이 Beauty 부문에서 +23.8%, Toys and Games 부문에서 +37.5%, Sports and Outdoors 부문에서 -1.8%의 Hits@10 성능을 기록하며, 기존의 최적 감독(supervised) 모델들과 경쟁력 있는 성과를 보였습니다.



### Understanding the Effect of Algorithm Transparency of Model Explanations in Text-to-SQL Semantic Parsing (https://arxiv.org/abs/2410.16283)
Comments:
          15 pages, 18 figure, Preprint

- **What's New**: 이번 연구는 AI 모델의 결정 과정을 설명하는 방법이 사용자의 경험에 미치는 영향을 탐구하며, 특히 'text-to-SQL Semantic Parsing'(텍스트-모든-쿼리 변환)라는 복잡한 예측 작업에 초점을 맞추었습니다. 세 가지 수준의 모델 설명 방식(저투명도, 중간투명도, 고투명도)을 도입하여 사용자의 AI에 대한 신뢰도와 예측 정확성을 어떻게 변화시키는지 살펴보았습니다.

- **Technical Details**: 해당 연구에서는 약 100명의 참가자를 대상으로 세 가지 알고리즘 투명도 수준(저, 중간, 고)에 따라 설명 접근 방식을 평가했습니다. 참가자는 컴퓨터 과학이나 SQL 프로그래밍을 학습하지 않은 비전문가로서, 주어진 질문을 SQL 쿼리로 변환하는 작업을 수행했습니다. 연구에서 사용된 주요 평가 메트릭은 Propensity to trust와 Jian scale의 신뢰 척도입니다.

- **Performance Highlights**: 결과적으로, (1) 저투명도와 고투명도 설명이 사용자의 결정 의존도를 낮추거나 높이는 경향이 있는 반면, 중간투명도 설명이 적절한 균형을 이루었습니다. (2) 중간투명도 그룹은 시간이 지남에 따라 성과가 증가하는 반면, 다른 그룹은 오히려 감소하는 경향을 보였습니다. (3) 모든 참가자가 연구 후 신뢰도 감소를 보였지만, 중간투명도 설명을 받은 그룹은 신뢰의 변동이 가장 적었습니다.



### Large Language Models Empowered Personalized Web Agents (https://arxiv.org/abs/2410.17236)
Comments:
          The code and data are available on the project website this https URL

- **What's New**: LLM 기반 개인화 웹 에이전트의 중요성을 강조하며, 개인화된 데이터를 통합하여 사용자의 지침을 더 잘 이해하고 맞춤형 행동을 실행하는 방법을 제안합니다.

- **Technical Details**: 개인화 웹 에이전트를 위한 새로운 벤치마크인 PersonalWAB을 구축하여, 사용자 지침, 개인화된 사용자 데이터, 웹 기능 및 세 가지 개인화된 웹 작업에 대한 평가 패러다임을 포함합니다. 또한, PUMA(Personalized User Memory-enhanced Alignment) 프레임워크를 통해 LLM을 개인화된 웹 에이전트 작업에 맞추도록 조정합니다.

- **Performance Highlights**: PUMA는 PersonalWAB에서 기존의 웹 에이전트 성능을 초월하여 개인화된 사용자 지침 및 선호도와 더 잘 정렬되어 보다 지능적이고 맞춤화된 웹 서비스를 제공할 수 있음을 입증합니다.



### TELII: Temporal Event Level Inverted Indexing for Cohort Discovery on a Large Covid-19 EHR Datas (https://arxiv.org/abs/2410.17134)
- **What's New**: 이번 연구에서는 대규모 전자 건강 기록(EHR) 데이터셋을 위한 코호트 발견(cohort discovery)에서 신뢰성 있고 빠른 시간 쿼리를 가능하게 하는 TELII라는 새로운 방법을 소개합니다.

- **Technical Details**: TELII는 사건(event) 간의 관계와 시간 차이를 미리 계산하고 저장하는 역 인덱스(inverted index) 메소드를 사용하여 대규모 EHR 데이터셋에서 효율적인 시간 쿼리를 가능하게 합니다. TELII는 MongoDB 백엔드로 구현되어 있으며, COVID-19 데이터를 포함한 887만 개의 환자 데이터를 처리합니다.

- **Performance Highlights**: TELII의 시간 쿼리 성능은 기존의 비시간 기반 역 인덱스보다 최대 2000배 더 빠르며, 응답 시간은 밀리초 수준으로, 연구자가 사건 관계를 신속하게 탐색하고 초기 증거를 찾을 수 있게 합니다.



### Enhancing Answer Attribution for Faithful Text Generation with Large Language Models (https://arxiv.org/abs/2410.17112)
Comments:
          Accepted to KDIR 2024 (part of IC3K 2024)

- **What's New**: 이번 연구에서는 Large Language Models (LLMs)에서의 answer attribution(답변 귀속) 프로세스를 개선하기 위한 새로운 방법을 제안하고 있습니다. 기존의 방법들을 분석하고, 더 독립적이며 맥락화된 주장을 통해 답변의 정확도와 신뢰성을 높이고자 합니다.

- **Technical Details**: 우리는 다음과 같은 주요 요소에 초점을 맞추었습니다: (1) 답변 분할(answer segmentation) 및 증거 검색(evidence retrieval)과 같은 서브 태스크의 효과를 평가하고, (2) 발견된 단점을 바탕으로 새로운 개선 방안을 제시하며, (3) 이러한 새로운 방법들이 answer attribution 컴포넌트의 성능을 어떻게 개선하는지에 대한 수치적 및 질적 분석을 수행합니다.

- **Performance Highlights**: 제안된 방법은 기존의 answer attribution 시스템보다 더 높은 성능을 보이며, 신뢰할 수 있는 정보 제공과 사용자에게 명확한 출처를 제시하는 데 기여합니다.



### Bridging the Modality Gap: Dimension Information Alignment and Sparse Spatial Constraint for Image-Text Matching (https://arxiv.org/abs/2410.16853)
- **What's New**: DIAS라는 새로운 방법을 제안하여 이미지와 텍스트 간의 모달리티 격차를 해소하고, 이미지-텍스트 매칭의 성능을 향상시킵니다. 이를 위해 두 가지 접근법이 도입되며, 첫째로 서로 다른 모달리티의 임베딩 정보를 일치시키고, 둘째로 잡음이 있는 쌍의 제약을 도입하여 의미적 정렬의 효과를 보장합니다.

- **Technical Details**: DIAS는 모달리티 간의 임베딩 정보를 해당 차원에 맞추어 조정하는 차원 정보 정렬 방법을 사용합니다. 이는 두 가지 주요 접근 방식으로 나뉘는데, 첫 번째는 해당 차원에서 서로 다른 모달리티의 임베딩 간 상관관계를 강화하는 것이고, 두 번째는 맞지 않는 쌍의 공간적 제약 조건을 활용하는 것입니다. 또한, 희소 상관관계 알고리즘을 도입하여 최상위 상관관계를 선택하여 중요한 특징을 학습하도록 합니다.

- **Performance Highlights**: DIAS는 Flickr30k 및 MSCOCO 벤치마크에서 4.3%-10.2%의 rSum 개선을 보여 주며, 이미지-텍스트 매칭 태스크에서 기존 방법들에 비해 우수한 성능을 입증하였습니다.



### Beyond Retrieval: Generating Narratives in Conversational Recommender Systems (https://arxiv.org/abs/2410.16780)
- **What's New**: 이 논문은 대화형 추천 시스템을 위한 자연어 생성 작업에서 REGEN이라는 새로운 데이터셋을 소개합니다. REGEN은 Amazon 제품 리뷰 데이터를 풍부한 사용자 내러티브로 확장하여 생성된 데이터셋입니다. 이 데이터셋은 사용자 선호도에 대한 개인화된 설명, 추천 아이템에 대한 제품 추천 및 사용자 구매 이력 요약을 포함하고 있습니다.

- **Technical Details**: REGEN은 사용자-아이템 상호작용 신호로부터 일관성 있는 자연어 출력을 생성하는 작업과 프레임워크를 도입합니다. 이 연구에서는 협업 필터링(CF) 신호와 콘텐츠 임베딩을 통합하는 융합 아키텍처를 제안하여 LLM의 입력으로 사용합니다. 실험 결과, CF와 콘텐츠 임베딩을 결합함으로써 언어 메트릭에서 4-12%의 성능 향상을 보여주었습니다.

- **Performance Highlights**: 제안된 모델은 사용자의 과거 상호작용을 바탕으로 풍부한 내러티브를 생성할 수 있다는 점에서 인간과 같은 대화형 추천을 생성하는 데 효과적임을 입증했습니다. 또한 데이터를 통해 CF와 콘텐츠 임베딩이 이 새로운 생성 작업에 어떻게 기여하는지 분석하였습니다.



### Cutting Through the Confusion and Hype: Understanding the True Potential of Generative AI (https://arxiv.org/abs/2410.16629)
- **What's New**: 이 논문은 생성적 AI(GenAI)의 정교한 환경을 탐구하며, 특히 대규모 언어 모델(LLM)과 같은 신경망 기반 모델에 중점을 둡니다. GenAI가 낙관적인 인기도를 얻고 있는 반면, 비판적인 시각 또한 존재하지만, 이 연구는 그 능력과 한계, 그리고 사회적 기능 및 개인적 상호작용에 미치는 깊은 영향을 균형 잡힌 시각에서 조명합니다.

- **Technical Details**: LLM은 대량의 텍스트 데이터에서 훈련된 복잡한 AI 시스템으로, 수조 개의 토큰을 포함하며, 이들 모두는 책, 기사, 컴퓨터 코드 및 웹 페이지로 구성됩니다. LLM의 학습은 깊은 신경망(Deep Neural Network, DNN) 기술을 활용하며, 특히 변환기(Transformer)라는 특정한 유형의 DNN을 사용해 언어의 복잡성을 학습합니다. LLM은 기계 학습 훈련 후 사용자 입력의 응답 및 콘텐츠 생성, 요약, 번역 등 다양한 언어 관련 작업을 수행할 수 있습니다.

- **Performance Highlights**: 최근 몇 년 간 GenAI의 발전은 여러 분야에서 인간 수준의 성과를 달성하게 되었고, 이러한 기술의 응용은 기업 생산성을 향상시키고 현대 사회에 긍정적인 영향을 미치고 있습니다. 그러나 LLM은 데이터의 편향, 환각(hallucinations), 논리 및 수리적 추론의 적용 제한, 해로운 콘텐츠 생성 가능성 등의 여러 과제에 직면해 있습니다.



### Distill-SynthKG: Distilling Knowledge Graph Synthesis Workflow for Improved Coverage and Efficiency (https://arxiv.org/abs/2410.16597)
- **What's New**: 본 연구에서는 LLMs를 기반으로 한 새로운 KG(SynthKG) 구축 워크플로우를 제안합니다. 이 방법은 KG 생성을 단일 단계로 간소화한 Distill-SynthKG로 발전되며, 대규모 문서에서도 효율적이고 고품질의 KG를 생성할 수 있게 합니다.

- **Technical Details**: SynthKG는 문서를 관리 가능한 의미적으로 완전한 텍스트 청크로 분할하고, 각 청크마다 LLM을 통해 잠재적 엔티티 및 관계를 추출합니다. 이 과정에서 기존의 질문-답변 데이터셋을 재구성하여 KG 평가를 위한 새로운 데이터셋과 지표를 도입했습니다.

- **Performance Highlights**: Distill-SynthKG는 기존 모든 기준 모델들보다 KG 품질이 뛰어나며, RAG에서의 검색 및 질문 응답 작업에서도 일관된 우수한 성능을 보였습니다. 제안한 그래프 기반 검색 방법 역시 여러 벤치마크 데이터셋에서 KG 검색 방법보다 성능이 우수한 것으로 나타났습니다.



### Optimizing LLMs with Direct Preferences: A Data Efficiency Perspectiv (https://arxiv.org/abs/2410.16586)
- **What's New**: 대형 언어 모델(LLMs)의 출력과 인간의 선호도(예: 인간 피드백을 통한 강화 학습, RLHF)와의 정렬이 실제 시나리오에서의 효과성을 보장하기 위해 필수적임을 강조합니다. 이 연구는 여러 유형의 선호 데이터가 모델 성능에 미치는 영향을 체계적으로 탐구합니다.

- **Technical Details**: 본 연구에서는 Direct Preference Optimization (DPO)을 사용하여 미리 훈련된 LLM의 미세 조정(정교화)을 관찰하며, 선호 데이터의 방대한 양에 대한 의존도를 줄이고자 했습니다. 이를 위해 세 가지 공개 데이터셋(OpenOrca, UltraFeedback, Capybara)에 대해 DPO 방법의 효과성을 평가했습니다.

- **Performance Highlights**: 데이터의 양이 증가함에 따라 모델 성능이 일반적으로 향상되고 안정화되었습니다. 다양한 데이터셋을 조합해 사용할 경우 모델 효과성이 크게 개선되었습니다. 특히, 대화형 프롬프트로 훈련된 모델이 질문 응답 프롬프트로 훈련된 모델에 비해 더 나은 성과를 보였습니다.



### PODTILE: Facilitating Podcast Episode Browsing with Auto-generated Chapters (https://arxiv.org/abs/2410.16148)
Comments:
          9 pages, 4 figures, CIKM industry track 2024

- **What's New**: 이번 연구에서는 PODTILE이라는 새로운 모델을 소개하여, 팟캐스트 에피소드의 자동 챕터화를 효과적으로 수행합니다. 이는 에피소드의 메타데이터와 이전 챕터 제목을 포함한 구조적 글로벌 컨텍스트를 사용하여 이루어집니다.

- **Technical Details**: PODTILE은 변환기(Transformer) 구조의 인코더-디코더 모델로, 입력 텍스트에 글로벌 컨텍스트를 추가하고, 대화형 데이터의 구성 및 제목을 동시 생성합니다. 평균 약 16,000개의 토큰을 가지는 긴 팟캐스트 전사를 효율적으로 처리할 수 있도록 설계되었습니다.

- **Performance Highlights**: PODTILE은 강력한 기초 모델 대비 ROUGE 점수가 11% 향상되었습니다. 사용 통계에 따르면, 청취자들은 자동 생성된 챕터가 특히 덜 알려진 팟캐스트를 탐색하는 데 유용하다고 보고하였습니다.



### Unleashing the Potential of Multi-Channel Fusion in Retrieval for Personalized Recommendations (https://arxiv.org/abs/2410.16080)
Comments:
          12 pages, 8 figures

- **What's New**: 본 논문은 멀티 채널 융합(multi-channel fusion)의 문제를 처음으로 정의하고, 체계적으로 최적화된 가중치 할당이 개인화 추천 시스템(per personalized recommendations)을 크게 개선할 수 있음을 보여줍니다.

- **Technical Details**: 논문에서는 비개인화 가중치 할당을 위한 두 단계 최적화 전략을 제안하고, 흑상자 최적화 기술(black-box optimization techniques)을 활용합니다. 구체적으로, 첫 단계에서는 Cross Entropy Method를 통해 가중치 분포를 반복적으로 개선하여 거의 최적 솔루션에 도달하고, 두 번째 단계에서는 Bayesian Optimization을 통해 예측 성능을 개선하여 보다 효율적으로 검색 공간을 탐색합니다. 개인화된 가중치 할당을 위해 Reinforcement Learning 기반의 정책 경량화(policy gradient-based approach)를 이용합니다.

- **Performance Highlights**: 세 가지 대규모 실제 데이터셋에서 실험하여 제안하는 접근 방식이 현재의 기준선보다 우수함을 입증하였으며, X사에서의 실제 배포를 통해 성능과 사용자 경험에서 큰 개선을 이루었습니다.



### Surprising Patterns in Musical Influence Networks (https://arxiv.org/abs/2410.15996)
Comments:
          To appear in the Latin American Musical Information Retrieval Workshop

- **What's New**: 이 논문은 음악 영향 네트워크의 진화를 추적하기 위해 Bayesian Surprise를 적용하여 시간이 지남에 따라 네트워크 구조의 변화가 중요한 시기를 드러내고, 실제 데이터에서 네트워크 진화에 대한 다양한 가설을 테스트하는 유연한 프레임워크임을 보여줍니다.

- **Technical Details**: 저자들은 AllMusic Guide와 WhoSampled의 두 개 데이터셋을 사용하여 사람의 손으로 만든 음악 영향 네트워크를 분석합니다. Pagerank와 Disruption 중심성 점수를 통해 네트워크의 변화 시점을 식별하며, 이를 통해 아티스트의 중심성에 영향을 미치는 네트워크 구성의 변화를 상세히 설명합니다. Bayesian Surprise는 여러 중심성을 결합하여 음악 영향 네트워크의 복잡한 랭킹을 위해 적절하게 적용됩니다.

- **Performance Highlights**: 이 연구는 네트워크 진화의 통찰력을 제공하는 도구로서 Bayesian Surprise를 적용한 최초의 사례입니다. 이는 각 노드(아티스트)의 네트워크 구조가 그들의 중심성에 미치는 영향을 분석하여, 음악적 영향력의 흥미로운 궤적을 드러내는 데 도움을 줍니다.



### Centrality-aware Product Retrieval and Ranking (https://arxiv.org/abs/2410.15930)
Comments:
          EMNLP 2024: Industry track

- **What's New**: 이 논문은 전자상거래 플랫폼에서 사용자 경험을 향상시키기 위한 제품 순위를 개선하는 방법을 다룹니다. 기존의 Transformer 기반 모델들이 사용자 의도를 반영하지 못하는 문제를 해결하기 위해, eBay의 데이터셋에서 수작업으로 주석을 달아 사용자의 의도와 관련된 점수를 부여하는 새로운 접근법인 User-intent Centrality Optimization (UCO)을 제안합니다.

- **Technical Details**: UCO 접근법은 쿼리-제목 쌍의 검색에서 사용자 의도를 최적화하며, 특히 의미적으로 관련 있지만 사용자 의도를 반영하지 않는 하드 네거티브(hard negatives)를 처리하기 위해 듀얼 손실 기반 최적화(dual-loss based optimization)를 활용합니다. 기존의 내부 평가 세트(Internal Graded Relevance, IGR dataset)를 이용하여 도전적인 평가 세트를 선별하고, 이를 통해 검색 효율성을 개선합니다.

- **Performance Highlights**: 이 연구는 제안된 프레임워크가 다양한 평가 메트릭에서 제품 순위 효율성을 크게 향상시키는 결과를 보여주었습니다. 사용자 의도에 맞춘 제목들이 높은 순위에 오르도록 하여 전자상거래 플랫폼에서 사용자 경험을 개선하는 데 기여합니다.



### Automatic Search of Multiword Place Names on Historical Maps (https://arxiv.org/abs/2410.15586)
Comments:
          4 pages, 4 figures, and 2 tables. To be published in proceedings ACM SIGSPATIAL 2024 GeoSearch Workshop

- **What's New**: 이 논문에서는 역사적 지도에서 다단어 지명을 효과적으로 검색하기 위한 새로운 쿼리 방법을 제안합니다. 복잡한 텍스트 레이아웃으로 인해 발생하는 기존 다단어 지명 검색의 어려움을 해결하고자 하며, 최소 신장 트리(minimum spanning tree)를 통해 단일 단어 레이블을 연결하여 잠재적인 다단어 구를 생성하는 방법을 사용합니다.

- **Technical Details**: 제안된 방법은 텍스트 레이블 L={l1,…,ln} 를 입력으로 받아, 각 텍스트 레이블 간의 공간적 관계를 바탕으로 최소 신장 트리를 구성합니다. 이 구조는 텍스트 레이블들의 조합을 가능하게 하여 다단어 지명 검색을 수행합니다. 또한, 텍스트 레이블의 위치, 크기, 각도 등의 시각적 특징을 고려하여 연결 가능성을 평가합니다.

- **Performance Highlights**: 이 방법은 다단어 지명 검색의 정확도와 속도를 평가하기 위해 두 가지 실험을 수행했습니다. 실험 결과, 역사적 맵에서 다단어 지명을 포함한 여러 맵이 효과적으로 검색됨을 보였으며, 다양한 역사적 변화가 반영된 맵을 탐색할 수 있는 가능성을 제시합니다.



### ConTReGen: Context-driven Tree-structured Retrieval for Open-domain Long-form Text Generation (https://arxiv.org/abs/2410.15511)
Comments:
          Accepted at EMNLP'24 Findings

- **What's New**: 이 논문에서는 ConTReGen이라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 정보 검색의 맥락 중심적인 접근 방식을 트리 구조로 구성하여 보다 깊이 있고 관련성 높은 콘텐츠 검색을 가능하게 합니다.

- **Technical Details**: ConTReGen은 두 단계의 과정을 통해 장문의 텍스트 생성을 향상시킵니다: 첫 번째는 입력된 질문에 직접 관련된 구절을 검색하는 상향식 접근(top-down approach)이며, 두 번째는 검색된 문서의 요약 정보를 하향식으로 종합(bottom-up synthesis)하여 최종 출력을 생성합니다. 이는 복합적인 정보를 체계적으로 탐색하는 것을 목표로 합니다.

- **Performance Highlights**: 여러 데이터셋(LFQA, ODSUM, ODSUM-WikiHow)에서의 실험 결과, ConTReGen은 기존의 최첨단 RAG 모델들을 초과하는 성능을 보여주었습니다.



### Deep Class-guided Hashing for Multi-label Cross-modal Retrieva (https://arxiv.org/abs/2410.15387)
- **What's New**: 이 논문에서는 DCGH(Deep Cross-Modal Hashing) 방법을 제안하여 intra-class aggregation과 inter-class structural relationships를 동시에 유지하는 문제를 해결하고자 합니다. 특히, proxy loss와 pairwise loss를 결합하여 이러한 관계를 효과적으로 유지하는 방법을 모색하였습니다.

- **Technical Details**: DCGH 방법은 proxy loss를 통해 데이터의 intra-class aggregation을 유지하고, pairwise loss를 사용하여 inter-class 구조적 관계를 유지합니다. 또한, variance constraint를 도입하여 조합에 의해 발생할 수 있는 semantic bias 문제를 해결합니다.

- **Performance Highlights**: 3개의 벤치마크 데이터셋에 대한 비교 실험 결과, DCGH 방법은 기존의 cross-modal retrieval 방법들과 비교하여 동등하거나 더 나은 성능을 보였습니다.



### Performance-Driven QUBO for Recommender Systems on Quantum Annealers (https://arxiv.org/abs/2410.15272)
- **What's New**: 본 논문에서는 추천 시스템의 특성 선택을 위해 Counterfactual Analysis Quadratic Unconstrained Binary Optimization (CAQUBO)라는 새로운 접근 방식을 제안합니다. 이는 카운터팩추얼 분석을 통해 각 특성의 영향력을 측정하고, 이를 통해 최적 특성 조합을 선택하여 최종 추천 성능을 개선합니다.

- **Technical Details**: CAQUBO는 각각의 특성을 제외했을 때의 최종 추천 성능 변화를 평가하여 Coefficient Matrix를 구성합니다. 이는 기존의 방법들이 모델 결과나 직접적인 기준(label)과만 연결되어 있는 한계를 극복하며, 추천 성능에 초점을 맞춥니다. 현재의 양자 어닐러는 2개 이하의 특성만을 한 번에 제거할 수 있지만, 향후 개선 가능한 가능성을 제시합니다.

- **Performance Highlights**: 실험 결과, CAQUBO는 다양한 추천 시스템, 즉 Item-KNN부터 딥러닝 기반의 방법들에 이르기까지 평가되었으며, 추천 정확도 측면에서 기존 최첨단 양자 어닐러 기반 특성 선택 알고리즘을 초월하는 성능을 보여주었습니다.



### HyQE: Ranking Contexts with Hypothetical Query Embeddings (https://arxiv.org/abs/2410.15262)
- **What's New**: 이번 논문에서는 대규모 언어 모델(LLM)이 필요하지 않고 훈련 데이터에 맞춤화되지 않은 새로운 컨텍스트 랭킹 프레임워크를 제안합니다. 이 프레임워크는 LLM의 능력과 임베딩 유사성을 결합하여 효율적으로 컨텍스트를 정렬합니다.

- **Technical Details**: 제안된 프레임워크는 LLM을 사용하여 기존의 컨텍스트를 기반으로 사용자의 쿼리에 대한 가설 쿼리를 생성합니다. 그런 다음 이 가설 쿼리와 사용자 쿼리 간의 유사성에 따라 컨텍스트를 정렬합니다. 이 방법은 다양한 정보 검색 벤치마크에서 성능을 개선하며, 효율성과 확장성을 유지합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 다수의 정보 검색 벤치마크에서 컨텍스트의 랭킹 성능을 향상시키며, LLM의 사전 지식 없이도 관련성을 평가할 수 있는 새로운 방법론을 제공합니다.



### Crafting Tomorrow: The Influence of Design Choices on Fresh Content in Social Media Recommendation (https://arxiv.org/abs/2410.15174)
- **What's New**: 이 연구에서는 소셜 미디어 플랫폼에서 콘텐츠 생명주기(content lifecycle)와 관련된 디자인 결정들이 콘텐츠의 성공에 미치는 영향을 분석합니다. 특히, 콘텐츠의 초기 단계에서의 결정들이 후속 단계들의 성공에 어떻게 영향을 미치는지에 대한 중요한 통찰을 제공합니다.

- **Technical Details**: 연구는 '초기', '성장', '성숙', '만료'의 네 가지 단계로 나뉘는 콘텐츠 생명주기의 각 단계에서의 사용자 인터페이스(UI), 알고리즘, 시스템 설정의 다양한 요소들이 콘텐츠의 창출 및 소비 패턴에 어떻게 기여하는지를 다룹니다. 이 연구는 콘텐츠 진행(Content Progression, CVP)과 콘텐츠 생존(Content Survival, CSR) 같은 지표를 사용하여 작은 결정들이 콘텐츠의 지속성에 미치는 영향을 측정합니다.

- **Performance Highlights**: 이 논문의 주요 기여는 다음과 같습니다: 신선한 콘텐츠에 대한 최소 노출이 콘텐츠 결과에 미치는 영향, 추천 퍼널의 초기 단계에서 시간에 민감한 카테고리에 대한 차별 대우의 중요성, 콘텐츠의 초기 인상 수와 그 비율이 플랫폼 내 콘텐츠의 장수성에 결정적인 영향을 미친다는 점을 분석합니다.



### Mining Asymmetric Intertextuality (https://arxiv.org/abs/2410.15145)
- **What's New**: 이 논문은 자연어 처리(Natural Language Processing, NLP) 및 디지털 인문학(Digital Humanities, DH) 분야에서 새로운 작업인 비대칭 인터텍스트성(Mining Asymmetric Intertextuality)을 소개합니다.

- **Technical Details**: 비대칭 인터텍스트성은 한 텍스트가 다른 텍스트를 인용하거나 차용하지만 그에 대한 보답이 없는 일방적인 관계를 의미합니다. 우리는 문서를 작은 조각으로 나누고, LLM(대규모 언어 모델) 보조 메타데이터 추출을 사용해 구조화된 데이터로 정규화한 뒤, 쿼리 중에 이를 병합하여 명시적 및 암시적 인터텍스트성 관계를 탐지하는 접근법인 분할-정규화-병합(split-normalize-merge) 패러다임을 제안합니다.

- **Performance Highlights**: 이 시스템은 직설적인 인용부터 패러프레이징(paraphrasing), 문서 간 영향까지 다양한 수준의 인터텍스트성을 다루며, 메타데이터 필터링, 벡터 유사성 검색(vector similarity search), LLM 기반 검증의 조합을 사용합니다. 이는 문서가 지속적으로 추가되는 동적 성장 코퍼스에 특히 적합하여, 문학 아카이브(archival)나 역사 데이터베이스에서 효율적으로 확장할 수 있습니다.



### Incorporating Group Prior into Variational Inference for Tail-User Behavior Modeling in CTR Prediction (https://arxiv.org/abs/2410.15098)
- **What's New**: 이 논문에서는 사용자 행동 모델링에서의 새로운 접근법인 Group Prior Sampler Variational Inference (GPSVI)를 제안합니다. 이는 tail 사용자(행동이 빈약한 사용자)의 관심을 향상시키기 위해 그룹 선호를 사전(prior)으로 도입합니다.

- **Technical Details**: GPSVI는 tail 사용자의 흥미를 반영하기 위한 변량 추론 방법으로, 개인의 선호 모델링의 불확실성 추정에 따라 조정의 정도가 달라집니다. 또한, GPSVI는 volume-preserving flow를 통해 변량 추론의 표현력을 강화합니다.

- **Performance Highlights**: GPSVI는 전통적인 Attention 메커니즘으로의 반전이 가능하며, tail 사용자에게는 일관된 성능 향상을 제공하여 CTR(Click-through rate) 예측의 정확성을 높입니다. 실험 결과, GPSVI는 baseline 모델 대비 0.306%의 CTR 향상과 tail 사용자에서 0.659%의 성능 개선을 보여줍니다.



### A Recommendation Model Utilizing Separation Embedding and Self-Attention for Feature Mining (https://arxiv.org/abs/2410.15026)
- **What's New**: 본 논문에서는 정보 과부하 문제를 해결하기 위해 기존의 클릭률 예측 및 TOP-K 추천 시스템의 한계를 극복하는 새로운 추천 시스템 모델을 제안합니다. 이 모델은 분리 임베딩 크로스 네트워크(separation embedding cross-network)를 기반으로 하고 있습니다.

- **Technical Details**: 모델은 희소(feature sparsity)한 피처 벡터를 밀집(embedding)한 임베딩 벡터로 변환하기 위해 임베딩 신경망(embedding neural network) 레이어를 사용합니다. 이 모델은 별도의 차원에서 피처 교차 운영(feature cross operations)을 독립적으로 수행하여 피처 탐색의 정확성과 깊이를 향상시킵니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 복잡한 데이터 세트 처리에 있어 더 높은 적응성과 예측 정확성을 보여주며, 기존 모델의 문제를 효과적으로 해결하였습니다.



### Limpeh ga li gong: Challenges in Singlish Annotations (https://arxiv.org/abs/2410.16156)
- **What's New**: 이 연구에서는 싱글리시(Singlish)의 구문 분석을 위한 Parts-Of-Speech (POS) 태깅 작업을 다룹니다. 연구자들은 원주율 언어로 구성된 데이터셋을 구축하고, 이를 통해 기존 자동 태거의 한계를 분석하였습니다.

- **Technical Details**: 싱글리시 데이터셋은 55,000개의 SMS 텍스트 인스턴스 중 92개의 문장을 무작위로 샘플링하여 생성하였으며, 이 데이터는 원어민 싱글리시 화자들에 의해 번역 및 태깅되었습니다. 자동 POS 태깅에는 spaCy의 pre-trained 모델을 사용했습니다 (en_core_web_sm와 en_core_web_trf).

- **Performance Highlights**: 자동 POS 태거의 정확도는 평균 80%로, 영어 문장 태깅의 경우 평균 97%의 성과와 비교해 미비한 결과를 보였습니다. 이러한 결과는 싱글리시의 독특한 문법 구조로 인해 발생하는 것으로 나타났습니다.



### Developing Retrieval Augmented Generation (RAG) based LLM Systems from PDFs: An Experience Repor (https://arxiv.org/abs/2410.15944)
Comments:
          36 pages, 8 figures, 2 tables, and python code snippets

- **What's New**: 이 논문은 PDF 문서를 주요 데이터 소스로 사용하여 Retrieval Augmented Generation (RAG) 시스템을 개발한 경험을 보고합니다. RAG 아키텍처는 대형 언어 모델(LLMs)의 생성 능력과 정보 검색의 정밀성을 결합하여 투명성, 정확성 및 상황에 맞는 응답을 향상시킬 수 있는 잠재력을 지니고 있습니다.

- **Technical Details**: RAG 시스템의 개발은 데이터 수집, 전처리, 검색 인덱싱 및 응답 생성에 이르는 전체 파이프라인을 구성합니다. MLLM과 정보 검색(NLP의 두 가지 핵심 구성 요소)를 통합하여 정보를 외부 데이터 소스에서 끌어오는 방식을 설명합니다. 이를 통해 정보를 현실적이고 정확하게 제공할 수 있는 시스템을 만듭니다.

- **Performance Highlights**: RAG 모델 프레임워크는 생성 AI의 패러다임 변화를 가져옵니다. 고급 NLP 애플리케이션에서 RAG 시스템의 도입이 의미하는 바는 전통적인 모델의 불투명한 출력을 넘어, 신뢰성 높은 응답을 생성하는 데 기여할 수 있습니다.



### Using GPT Models for Qualitative and Quantitative News Analytics in the 2024 US Presidental Election Process (https://arxiv.org/abs/2410.15884)
- **What's New**: 본 논문은 Google Search API와 GPT-4o 모델을 활용하여 뉴스의 질적 및 양적 분석을 수행하는 접근 방식을 제안합니다. 이를 통해 2024년 미국 대선 과정에 대한 뉴스를 분석하였습니다.

- **Technical Details**: Retrieval-augmented generation (RAG) 기법을 사용하여 뉴스 데이터를 분석하였습니다. 분석은 Google Search API를 통해 관련 웹 자원을 검색하고, LangChain의 SeleniumURLLoader를 이용하여 정보를 추출하는 두 단계로 이루어졌습니다. 주요 검색 쿼리는 'Kamala Harris AND Donald Trump'이며, 다양한 시간대와 뉴스 출처를 고려하였습니다.

- **Performance Highlights**: GPT 모델을 활용한 분석 결과는 선거 과정에서의 불확실성을 분석하는 데 도움을 주며, 질적 통찰력을 제공함으로써 향후 선거 분석에 응용될 수 있는 가능한 기초 자료를 생성합니다.



### Improve Dense Passage Retrieval with Entailment Tuning (https://arxiv.org/abs/2410.15801)
Comments:
          EMNLP 2024 Main

- **What's New**: 이 연구에서는 질문-답변 관련성을 정의하는 새로운 관점을 제시하고, 그에 기반하여 기존의 dense retriever 훈련 파이프라인에 통합 가능한 'entailment tuning' 방법론을 설계했습니다.

- **Technical Details**: 'Entailment tuning' 방법은 NLI(자연어 추론) 데이터를 활용하여 dense retriever의 성능을 향상시키는 데 초점을 맞추고 있습니다. 주어진 질문을 주장으로 변환하고, 이를 통해 claim-passage 쌍 및 premise-hypothesis 쌍을 구성합니다. 이후 주장의 대부분을 마스킹하고 encoder 모델을 훈련시켜 마스킹된 주장을 예측하도록 합니다. 이를 통해 retrieval 성능을 향상시키는 것을 목표로 합니다.

- **Performance Highlights**: 실험 결과, 'entailment tuning' 방법이 dense passage retrieval 작업에서 효과적임을 입증했으며, open-domain QA 및 retrieval-augmented generation(RAG)와 같은 다운스트림 작업에서 성능 향상을 가져왔습니다.



### Who's Who: Large Language Models Meet Knowledge Conflicts in Practic (https://arxiv.org/abs/2410.15737)
Comments:
          Accepted to EMNLP 2024 Findings

- **What's New**: 이번 연구에서는 Retrieval-augmented generation (RAG) 방법을 사용하여 사전 학습된 언어 모델의 메모리 한계를 극복하려고 하지만, 정보 충돌(conflict) 문제를 해결하기 위한 새로운 벤치마크 데이터 세트인 WhoQA를 소개합니다.

- **Technical Details**: WhoQA는 동일한 이름을 가진 Wikipedia 엔티티에 대한 공통 속성을 묻는 질문을 통해 지식 충돌을 유도합니다. 데이터 세트는 5000개의 질문으로 구성되어 있으며, 최대 8개의 고유한 답변을 포함할 수 있습니다. 질문은 (q, A, S, C) 형태의 쿼드플렛으로 나타내어 집니다.

- **Performance Highlights**: 실험 결과, WhoQA 질문의 단순함에도 불구하고 지식 충돌이 RAG 설정에서 LLM의 성능을 크게 저하시킨다고 보고했습니다.



### A Survey of Conversational Search (https://arxiv.org/abs/2410.15576)
Comments:
          35 pages, 8 figures, continue to update

- **What's New**: 현대 정보 접근의 초석으로서, 검색 엔진은 일상생활에서 없어서는 안 될 존재가 되었습니다. AI와 자연어 처리(NLP) 기술이 빠르게 발전함에 따라, 특히 대규모 언어 모델(LLMs)의 출현이 검색 엔진의 진화를 이끌어, 보다 직관적이고 지능적인 사용자 시스템 간 상호작용을 지원하게 되었습니다. 최근 '대화식 검색'이라는 새 패러다임이 대두되고 있습니다.

- **Technical Details**: 대화식 검색 시스템은 사용자와의 복잡하고 정밀한 정보 검색을 원활하게 하기 위해 자연어 대화를 활용하며, 전통적인 키워드 기반 검색 엔진과의 차별화된 특징을 가지고 있습니다. 이 시스템은 쿼리 재구성(query reformulation), 검색 명확화(search clarification), 대화 검색(conversational retrieval), 응답 생성(response generation)과 같은 주요 구성 요소를 갖추고 있습니다.

- **Performance Highlights**: 대화식 검색의 주요 장점은 사용자 경험을 향상시키는 데 있으며, 복잡한 쿼리를 지원하고, 다중 대화 상호작용에서 맥락을 유지하며, 강력한 정보 통합 및 처리 기능을 제공합니다. 이 연구는 현재의 대화식 검색 시스템의 현실 세계 응용 사례와 평가도 제공해 향후 연구 및 개발 방향을 제시하고자 합니다.



### Transit Pulse: Utilizing Social Media as a Source for Customer Feedback and Information Extraction with Large Language Mod (https://arxiv.org/abs/2410.15016)
Comments:
          17 pages, 21 figures

- **What's New**: 본 논문은 사회적 미디어 데이터를 활용하여 대중 교통 사용자 피드백을 분석하기 위한 최신 방법론을 제안합니다. 기존의 방법들은 사전 정의된 주제 라벨에 의존했으나, 본 연구에서는 LLM(Large Language Model)을 활용하여 보다 포괄적인 인사이트를 제공합니다.

- **Technical Details**: 제안된 방법은 Llama 3라는 LLM을 사용하여 사회적 미디어의 대중 교통 관련 정보, 감정 및 풍자 감지, 비정상적인 시스템 문제 식별, 그리고 위치 데이터를 분석합니다. 정보 추출 파이프라인에 RAG(Retrieval-Augmented Generation) 접근 방식을 통합하여 외부 지식을 모델에 결합합니다.

- **Performance Highlights**: 전통적인 NLP 접근 방식과 비교하여 LLM을 활용한 이 방법은 사용자 트윗 데이터에 대한 분석 성능에서 유망한 결과를 보여주었으며, 대중 교통 기관의 대응 능력을 개선하고 실질적인 인사이트를 제공합니다.



### Visual Navigation of Digital Libraries: Retrieval and Classification of Images in the National Library of Norway's Digitised Book Collection (https://arxiv.org/abs/2410.14969)
Comments:
          13 pages, 2 figures, 4 tables, Accepted to the 2024 Computational Humanities Research Conference (CHR)

- **What's New**: 노르웨이 국립도서관(NLN)의 1900년 이전 도서에서 이미지 검색 애플리케이션을 개발하고, 최신 이미지 임베딩 및 모델을 비교하여 시각적 문화유산의 분석 가능성을 높입니다.

- **Technical Details**: 이 연구는 Vision Transformer (ViT), Contrastive Language-Image Pre-training (CLIP) 및 Sigmoid Loss for Language-Image Pre-Training (SigLIP) 임베딩을 사용하여 이미지 검색 및 분류 작업을 수행합니다. SigLIP 임베딩이 CLIP 및 ViT보다 약간 더 우수하게 작동하며, 이미지 분류에서는 디지털화 파이프라인에서 이미지 데이터 세트 정리에도 기여합니다.

- **Performance Highlights**: SigLIP 기반의 애플리케이션은 정확한 이미지 검색에서 우수한 성능을 보여주며, 이미지 검색의 효율성을 높이는 데 성공했습니다. 이 연구는 또한 다양한 텍스트 및 이미지 데이터의 탐색 가능성을 향상시키는 방향으로 나아가는 중요한 기초 자료를 제공합니다.



### The S2 Hierarchical Discrete Global Grid as a Nexus for Data Representation, Integration, and Querying Across Geospatial Knowledge Graphs (https://arxiv.org/abs/2410.14808)
- **What's New**: 본 논문에서는 Geospatial Knowledge Graphs (GeoKGs)가 Geospatial Artificial Intelligence에서 중요한 역할을 하고 있으며, KnowWhereGraph라는 새로운 GeoKG 구현을 소개합니다. 이 Graph은 Google's S2 Geometry를 활용하여 다양한 데이터 출처에서 효율적인 데이터 처리를 가능하게 하며, 논문은 이 시스템의 구현 방법과 그 중요성을 강조합니다.

- **Technical Details**: KnowWhereGraph는 Discrete Global Grid Systems (DGGS) 중 하나인 S2 Geometry를 사용하여 데이터를 통합하고 표현합니다. 이는 복잡한 topology relations를 탐색하고 데이터의 semantic compression을 제공하여 데이터 관리의 복잡성을 줄입니다. GEospatial 데이터의 통합, 저장, 분석 방식에서 혁신적인 접근법을 제공합니다.

- **Performance Highlights**: KnowWhereGraph는 데이터를 효율적으로 처리하여 지리적 쿼리를 다루는 데 필요한 계산 복잡성을 줄이고, 다양한 스케일과 다양한 형식의 데이터를 통합할 수 있는 가능성을 보여줍니다. 결론적으로, DGGS 프레임워크, 특히 S2의 잠재력은 확장 가능한 GeoKG를 구축하는 데 큰 기여를 할 것으로 기대됩니다.



### Attribute-Based Semantic Type Detection and Data Quality Assessmen (https://arxiv.org/abs/2410.14692)
Comments:
          10 pages, 9 tables, sent for approval at BDCAT 2024

- **What's New**: 이번 연구는 데이터 품질 (Data Quality) 평가의 중요한 격차를 해소하기 위해 Attribute-Based Semantic Type Detection 및 데이터 품질 평가 중심의 혁신적인 방법론을 소개합니다.

- **Technical Details**: 이 방법은 다양한 속성 레이블 (Attribute Labels) 내의 의미론적 정보 (Semantic Information)를 활용하고, 규칙 기반 분석 (Rule-Based Analysis) 및 포괄적인 형식 (Formats)과 약어 (Abbreviations) 사전을 통합하여 약 23가지 유형의 실용적인 의미론적 유형 (Semantic Type) 분류 시스템을 도입합니다.

- **Performance Highlights**: UCI 머신러닝 데이터베이스의 50개 다양한 데이터 세트를 분석한 결과, YData Profiling과 비교하여 81개의 결측값 (Missing Values)을 정확하게 식별함으로써 데이터 품질 문제를 효과적으로 탐지하는 능력을 보여주었습니다.



New uploads on arXiv(cs.CV)

### Altogether: Image Captioning via Re-aligning Alt-tex (https://arxiv.org/abs/2410.17251)
Comments:
          accepted by EMNLP 2024; MetaCLIPv2

- **What's New**: 이 논문은 이미지 캡션의 질을 향상시키기 위해 합성 데이터 생성에 집중하고 있습니다. 기존의 방식이 존재하는 alt-text 메타데이터를 무시하고, 투명성이 결여된 점을 보완하기 위한 새로운 접근 방식을 제안합니다. 이 방법은 기존 alt-text를 편집하고 재정렬하는 것으로, 인간 주석자가 여러 라운드를 통해 캡션을 건설하는 과정입니다.

- **Technical Details**: 이 연구에서는 Altogether라는 원칙 기반 접근법을 제안합니다. 이 방법은 기존의 alt-text와 이미지 내용을 정렬하는 과정으로, 주석자들이 기존 alt-text를 바탕으로 이미지를 참조하며 캡션을 수정합니다. 이 과정은 여러 라운드에 걸쳐 수행되며, 그 결과로 리치한 시각 개념을 포함한 캡션이 생성됩니다. 이후 이 데이터를 바탕으로 캡셔너가 훈련됩니다.

- **Performance Highlights**: Altogether 접근법을 통해 생성된 캡션은 기존의 alt-text보다 4% 더 높은 CLIP 점수를 기록했으며, 최신 캡셔너와 비교하여도 도전적인 테스트 세트에서 뛰어난 성능을 보였습니다. 또한, 텍스트-이미지 생성 및 제로샷 이미지 분류 태스크에서도 유의미한 개선이 있었습니다.



### SpectroMotion: Dynamic 3D Reconstruction of Specular Scenes (https://arxiv.org/abs/2410.17249)
Comments:
          Project page: this https URL

- **What's New**: SpectroMotion은 3D Gaussian Splatting(3DGS)과 물리 기반 렌더링(Physically-Based Rendering, PBR), 변형 필드를 결합한 혁신적인 접근 방식을 제안합니다. 이를 통해 동적 스페큘러(scene containing dynamic specular objects) 장면을 재구성 가능하게 합니다.

- **Technical Details**: 이 방법은 변형 과정에서 표면 노멀(computation of surface normals)의 정확한 계산을 위한 잔여 보정(residual correction) 기법과, 시간에 따라 변하는 조명 조건에 적응하는 변형 가능 환경 맵(deformable environment map)을 도입했습니다. 또한, 장면 기하학 및 스페큘러 색상 예측을 향상시키기 위해 조대-세밀(coarse-to-fine) 학습 전략을 구현하였습니다.

- **Performance Highlights**: SpectroMotion은 동적 스페큘러 객체를 포함하는 장면의 뷰 합성(view synthesis)에서 이전 방법들을 능가하며, 실제 동적 스페큘러 장면을 합성할 수 있는 유일한 3DGS 방법으로 밝혀졌습니다. 이는 복잡하고 동적인 스페큘러 장면 렌더링에서 최첨단 기법들을 초월하는 성능을 보였습니다.



### PyramidDrop: Accelerating Your Large Vision-Language Models via Pyramid Visual Redundancy Reduction (https://arxiv.org/abs/2410.17247)
Comments:
          10 pages

- **What's New**: 이 논문에서는 Large Vision-Language Models (LVLMs)에서 이미지 토큰의 중복성을 줄이기 위해 PyramidDrop이라는 새로운 전략을 제안합니다. 이 방법은 모델의 성능을 저하시키지 않으면서 훈련 및 추론 효율성을 향상시킵니다.

- **Technical Details**: PyramidDrop은 LVLM을 여러 단계로 나누고 각 단계의 마지막에서 사전에 정의된 비율만큼 이미지 토큰을 떨어뜨리는 방식으로 작동합니다. 이를 통해 초기 낮은 레이어에서는 모든 이미지 토큰을 유지하고, 깊은 레이어로 갈수록 점진적으로 토큰 수를 줄입니다. 이 과정에서 경량화된 유사성 계산을 사용하여 오버헤드를 최소화합니다.

- **Performance Highlights**: PyramidDrop을 적용한 LLaVA-NeXT 모델은 훈련 시간을 40% 단축시키고, 55%의 추론 FLOPs를 감소시킴에도 불구하고 성능은 비슷합니다. 또한 이 방법은 학습을 필요로 하지 않는 추론 가속화 전략으로도 활용될 수 있으며, FastV와 비교해 성능과 비용 모두 개선되었습니다.



### Breaking the Memory Barrier: Near Infinite Batch Size Scaling for Contrastive Loss (https://arxiv.org/abs/2410.17243)
- **What's New**: 이 논문에서는 contrastive learning에 대한 새로운 접근법인 Inf-CL을 소개하고 있습니다. Inf-CL은 memory 효율성을 극대화하면서 배치 크기를 획기적으로 확장할 수 있게 해줍니다.

- **Technical Details**: Inf-CL은 similarity matrix의 전체 구현을 피하기 위해 contrastive loss 계산을 작은 블록으로 분할하는 tile-based computation 전략을採用합니다. 이를 통해 메모리 사용량을 줄이고, GPU 간의 ring-based communication을 통해 동기화와 I/O 오버헤드를 최소화하는 multi-level tiling 전략을 도입합니다.

- **Performance Highlights**: Inf-CL을 사용하면 CLIP-ViT-L/14 모델에 대해 4M 또는 12M의 배치 크기로 학습할 수 있으며, 기존의 SOTA(memory-efficient) 솔루션 대비 메모리를 두 배 줄이면서도 정확도를 유지합니다. 이 방식은 32 A800 80GB GPU를 이용하여 1 GPU 당仅 1.44 GB의 메모리만 사용합니다.



### LVSM: A Large View Synthesis Model with Minimal 3D Inductive Bias (https://arxiv.org/abs/2410.17242)
Comments:
          project page: this https URL

- **What's New**: 최근 발표된 Large View Synthesis Model (LVSM)은 희소( sparse-view) 입력으로부터 새로운 뷰를 효율적으로 합성하기 위한 혁신적인 transformer 기반 접근 방식입니다. 이 모델은 기존의 3D 유도 편향( inductive biases)을 완전히 배제하고 사실적 품질의 이미지를 생성할 수 있도록 설계되었습니다.

- **Technical Details**: LVSM은 두 가지 아키텍처를 제안합니다: (1) 인코더-디코더 LVSM은 입력 이미지 토큰을 1D 잠재( latent) 토큰으로 인코딩하고 이를 바탕으로 새로운 뷰 이미지를 디코딩합니다; (2) 디코더 전용 LVSM은 입력 이미지를 새로운 뷰 출력으로 직접 매핑하여 중간 장면 표현을 완전히 없애고, 플뤼커( Plücker) 광선 기반 위치 임베딩을 사용하여 결과 이미지를 생성합니다. 두 모델 모두 이전 방식들의 3D 유도 편향을 뛰어넘습니다.

- **Performance Highlights**: LVSM은 여러 데이터셋을 통해 테스트해본 결과, 이전의 최첨단 기법들에 비해 1.5에서 3.5 dB PSNR 향상을 기록하며, 특히 디코더 전용 모델이 더욱 우수한 품질과 제로샷( zero-shot) 일반화를 보여줍니다. 1-2대의 GPU로도 최상의 성능을 발휘하며, 이는 자원 효율성을 크게 향상시킵니다.



### EPContrast: Effective Point-level Contrastive Learning for Large-scale Point Cloud Understanding (https://arxiv.org/abs/2410.17207)
- **What's New**: 본 논문에서는 대규모 포인트 클라우드 이해를 위한 방법인 EPContrast를 제안한다. EPContrast는 AGContrast와 ChannelContrast라는 두 가지 하위 방법으로 구성되어 있으며, 기존의 Contrastive Learning이 가진 메모리 및 계산 자원 소모 문제를 개선하고자 한다.

- **Technical Details**: EPContrast는 비대칭 세분화 임베딩을 기반으로 긍정 쌍과 부정 쌍을 구축하는 AGContrast와 채널 특징 맵 간의 대조적 감독을 부과하는 ChannelContrast를 포함한다. AGContrast는 포인트 클라우드 씬의 각 포인트에 대한 세그먼트 특징을 고려하여 세밀한 감독 신호를 제공하며, ChannelContrast는 서로 다른 채널의 특징 맵을 부정 쌍으로, 동일한 채널 내의 특징 맵을 긍정 쌍으로 처리하여 고차원 특징 중복 문제를 완화한다.

- **Performance Highlights**: EPContrast는 S3DIS 및 ScanNetV2 데이터셋에서 심층적인 검증을 거쳤으며, 의미 기반 분할, 인스턴스 분할, 객체 탐지와 같은 다양한 작업에서 다른 대조 학습 방법들을 지속적으로 능가하는 결과를 보여준다. 추가적으로, EPContrast는 라벨 효율적인 및 일회 훈련 시나리오에서도 뛰어난 성능을 보인다.



### Emphasizing Discriminative Features for Dataset Distillation in Complex Scenarios (https://arxiv.org/abs/2410.17193)
Comments:
          24 pages, 13 figures

- **What's New**: 본 논문에서는 EDF(Emphasize Discriminative Features)를 제안하여, Grad-CAM 활성화 맵을 사용하여 합성 이미지의 핵심 식별 영역을 강조합니다. 간단한 데이터셋에서 성능은 뛰어나지만 복잡한 시나리오에서는 성능 저하 문제가 있음을 여실히 보여줍니다.

- **Technical Details**: EDF는 합성 이미지 distillation 과정에서 식별 특징을 강조하는 방법으로, Grad-CAM 활성화 맵을 통해 높은 활성도 영역의 업데이트를 강화합니다. 저손실(supervision signals)이 있는 경우는 제외하고, 높은 손실을 가진 신호만 활성화하여 효과적인 학습을 유도합니다. 이러한 방식으로 복잡한 시나리오에서 DD(데이터셋 증류)의 성능을 높입니다.

- **Performance Highlights**: 복잡한 시나리오에서 SOTA(state-of-the-art) 성능을 지속적으로 달성하며, ImageNet-1K의 여러 부분집합에서 손실 없는 성능을 기록했습니다. 이는 EDF가 데이터셋 증류 방법론에서 유의미한 발전을 이루어냈음을 보여줍니다.



### KANICE: Kolmogorov-Arnold Networks with Interactive Convolutional Elements (https://arxiv.org/abs/2410.17172)
- **What's New**: KANICE는 Convolutional Neural Networks (CNNs)와 Kolmogorov-Arnold Network (KAN)의 원리를 결합한 새로운 신경망 아키텍처입니다. 특히 Interactive Convolutional Blocks (ICBs)와 KAN 선형 레이어를 통합하여 복잡하고 비선형적인 데이터 관계를 포착하며, 동적이며 맥락에 의존적인 특징 추출을 가능하게 합니다.

- **Technical Details**: KANICE는 KAN의 일반 근사화 기능과 ICB의 적응형 특징 학습을 활용하여 CNN 프레임워크 내에서 작동합니다. KANICE는 MNIST, Fashion-MNIST, EMNIST, SVHN 데이터셋에서 실험을 수행하였으며, 기존 CNN 및 하이브리드 구조와 비교하여 일관되게 우수한 성능을 보였습니다. 또한, KANICE-mini라는 컴팩트 변형을 통해 성능을 유지하면서도 적은 파라미터 수로 효율성을 높였습니다.

- **Performance Highlights**: KANICE는 MNIST 데이터셋에서 99.35%의 정확도를 기록하였고, SVHN 데이터셋에서는 90.05%에 도달했습니다. KANICE-mini는 2,337,828개의 파라미터로 SVHN에서 90.00%의 정확도를 달성했습니다. KANICE 아키텍처는 전반적으로 이미지 분류 작업에서 성능과 계산 효율성을 균형 있게 유지하는 가능성을 보여줍니다.



### Are Visual-Language Models Effective in Action Recognition? A Comparative Study (https://arxiv.org/abs/2410.17149)
- **What's New**: 본 논문은 CLIP과 같은 비전-언어 (vision-language) 기반 모델이 복잡한 세밀한 액션 인식 작업에 얼마나 효과적인지를 평가하기 위한 대규모 연구를 제공합니다. 특히, 제로샷 (zero-shot) 분류와 프레임 단위 (frame-wise) 액션 세분화 작업에서의 성능 비교를 통해 현재의 최첨단 모델의 한계를 탐구합니다.

- **Technical Details**: 논문에서는 CLIP, X-CLIP, ViCLIP, ViFi-CLIP 등의 최첨단 비전-언어 모델들을 선정하여 성능을 비교합니다. 각 모델은 액션 분류와 세분화 작업에 대해 다양한 방법론을 적용합니다. 특히, 제로샷 액션 분류를 위한 액션 설명 생성 전략 및 비디오 질문 응답 (VQA) 모델을 활용한 프레임 단위 액션 예측 전략을 비교합니다.

- **Performance Highlights**: DaTA, Toyota Smarthome, Charades 등 다양한 데이터셋을 활용한 실험 결과, 현재 비전-언어 모델들이 세밀한 액션 인식 작업에 한계를 가지고 있음을 보여줍니다. 특정 환경에서의 영상 인식 성능을 높이기 위한 추가적인 연구 방향이 필요하다는 결론에 이르렀습니다.



### YOLO-TS: Real-Time Traffic Sign Detection with Enhanced Accuracy Using Optimized Receptive Fields and Anchor-Free Fusion (https://arxiv.org/abs/2410.17144)
Comments:
          13 pages, 9 figures and 7 tables

- **What's New**: YOLO-TS라는 새로운 실시간 고효율 도로 표지판 탐지 네트워크를 제안합니다. 이 네트워크는 다양한 데이터셋에서 도로 표지판의 크기 분포와 더욱 잘 일치하도록 다중 스케일 특성 맵의 감수 필드를 최적화합니다.

- **Technical Details**: YOLO-TS는 앵커 프리(anchor-free) 방법의 유연성을 활용하는 혁신적인 특성 융합 전략을 통해 고해상도 특성 맵에서 다중 스케일 객체 탐지를 수행합니다. 또한, 부풀려진 컨볼루션(dilated convolution)으로 인한 그리드 패턴의 부정적인 영향을 완화하는 독특한 모듈을 설계하여 소형 객체의 정보 효율성을 높입니다.

- **Performance Highlights**: TT100K 및 CCTSDB2021과 같은 도전적인 공개 데이터셋에서 YOLO-TS는 정확도와 속도 모두에서 기존의 최신 방법들을 초월하는 성능을 보였습니다. YOLO-TS는 평균 정밀도(mAP) 측면에서 이전 방법들보다 우수하고, 초당 프레임 수(FPS)에서도 최고 기록을 달성하며 모델의 파라미터 수를 크게 줄였습니다.



### AlphaChimp: Tracking and Behavior Recognition of Chimpanzees (https://arxiv.org/abs/2410.17136)
Comments:
          An eXpressive extension of ChimpACT [arXiv:2310.16447], proposes AlphaChimp for tracking and behavior recognition of chimpanzees. arXiv admin note: substantial text overlap with arXiv:2310.16447

- **What's New**: 이번 연구에서는 비디오 영상에서 침팬지 행동을 자동으로 감지하고 추적하는 새로운 방법인 AlphaChimp를 개발했습니다. 이 방법은 기존 방법들에 비해 행동 인식을 20% 향상시킬 수 있습니다.

- **Technical Details**: AlphaChimp는 DETR (DEtection TRansformer) 기반의 아키텍처를 활용하여 침팬지 탐지와 행동 인식을 동시에 처리할 수 있는 전방위적(end-to-end) 모델입니다. 이 방법은 다중 해상도 시간 정보 통합 및 주의(attention) 메커니즘을 적용하여 사회적 행동 인식의 정확성을 크게 향상시킵니다.

- **Performance Highlights**: AlphaChimp는 ChimpACT 데이터셋에서 10%의 추적 정확도 향상과 20%의 행동 인식 개선을 달성했습니다. 이는 기존 최첨단(State-of-the-Art) 모델들에 비해 뛰어난 성능을 보이는 것입니다.



### CLAP: Concave Linear APproximation for Quadratic Graph Matching (https://arxiv.org/abs/2410.17101)
Comments:
          Accepted as an oral paper in International Symposium on Visual Computing (ISCV2024)

- **What's New**: 이 논문은 그래프 매칭의 계산을 가속화하기 위해 새로운 선형 모델과 솔버를 소개합니다. 이들은 원래의 Quadratic Assignment Problem (QAP)을 정리하여 간결한 형태로 변환해 문제를 쉽게 해결할 수 있도록 합니다.

- **Technical Details**: 이 연구에서는 Pairwise 구조 제약조건을 L1 노름 하에 선형 모델로 수식화하고, 긍정 반정형 행렬(Positive Semi-Definite Matrix) 근사를 사용하여 효율적으로 매칭 문제를 해결하는 알고리즘인 CLAP을 제안합니다. CLAP 모델은 연속적인 최적화 문제로 전환하여 Sinkhorn 최적 수송(Sinkhorn Optimal Transport) 알고리즘을 통한 빠른 해결을 가능하게 합니다.

- **Performance Highlights**: PascalVOC 벤치마크에서 실험 결과, CLAP 알고리즘은 기존의 최신 방법들과 비슷한 정확도를 유지하면서도 실행 속도가 크게 향상된 성능을 보여주었습니다.



### Masked Differential Privacy (https://arxiv.org/abs/2410.17098)
- **What's New**: 이번 연구에서는 Masked Differential Privacy (MaskDP)라는 효과적인 접근 방식을 제안하여, 차별적 프라이버시(differential privacy) 적용에 있어 민감한 영역을 선별적으로 제어할 수 있도록 합니다. 이러한 방식은 전체 입력에 DP를 적용하는 대신, 데이터의 비민감한 시공간(spatio-temporal) 지역에 대한 프라이버시 보호를 가능하게 합니다.

- **Technical Details**: MaskDP는 훈련 데이터에서 민감한 토큰(private tokens)과 비민감한 토큰(public tokens)으로 분해하여, 민감한 토큰에만 차별적 프라이버시를 적용합니다. 이를 통해 모델의 유틸리티(utility)를 높이면서 프라이버시를 보호하는 방법을 제공합니다.

- **Performance Highlights**: MaskDP를 활용한 실험 결과, $\epsilon<1$ 범위에서 기존의 표준 DP에 비해 유틸리티와 프라이버시 간의 균형을 더 우수하게 달성했음을 보여줍니다. 특히 합성적(synthetic) 데이터와 실제(real) 데이터가 혼합된 데이터셋에서 향상된 성능을 나타냈습니다.



### A Survey on Deep Learning-based Gaze Direction Regression: Searching for the State-of-the-ar (https://arxiv.org/abs/2410.17082)
Comments:
          Accepted on SPRA 2024 (Istanbul, Turkey)

- **What's New**: 이번 논문에서는 헤드와 눈 이미지로부터 시선 방향 벡터를 회귀(regression)하기 위한 딥러닝 기반 방법들을 종합적으로 조사했습니다. 특히 입력 데이터, 모델 아키텍처(architecture), 모델을 감독(supervise)하는 데 사용되는 손실 함수(loss function)에 중점을 두어 여러 출판된 방법들을 자세히 설명합니다.

- **Technical Details**: 이 연구는 다양한 방법론을 정리하고, 시선 방향 회귀 방법을 훈련(training)하고 평가(evaluate)하는 데 사용할 수 있는 데이터셋 목록을 제시합니다. 또한, 여러 방법이 서로 비교하기 어려운 이유는 검증(validation) 또는 테스트(test) 하위 집합의 차이에 있다고 언급하였습니다. 이를 해결하기 위해 Gaze360 데이터셋에 대해 공통의 검증 설정에서 여러 방법을 재평가하였습니다.

- **Performance Highlights**: 실험 결과, 최신 방법들은 여러 최신기술(state-of-the-art)이라고 주장하면서도 일부 오래된 방법들에 비해 성능이 크게 저하되었음을 보여주었습니다. 마지막으로 정적(static) 시험 조건 하에서 시계열 모델이 정적 모델보다 우수한 성능을 보인다는 것을 보여줍니다.



### Neuronal Competition Groups with Supervised STDP for Spike-Based Classification (https://arxiv.org/abs/2410.17066)
- **What's New**: 이 논문에서는 Winner-Takes-All (WTA) 경쟁을 효과적으로 구현하기 위한 새로운 방법을 제안합니다. Neuronal Competition Group (NCG) 구조를 도입하여 분류 능력을 향상시키고, 클래스별 다양한 패턴을 학습할 수 있도록 합니다.

- **Technical Details**: NCG는 특정 클래스에 매핑된 뉴런 그룹으로, intra-class WTA 및 경쟁 조절 메커니즘을 구현합니다. 이들 뉴런은 결정 메커니즘에 고정된 임계값과 가중치 업데이트 빈도를 조절하는 적응형 임계값을 사용합니다.

- **Performance Highlights**: NCG를 채택한 spiking classification layer는 CIFAR-10 및 CIFAR-100과 같은 이미지 인식 데이터 세트에서 유의미한 정확도를 향상시키며, 균형 잡힌 경쟁 및 향상된 클래스 분리를 보장하는 데 중요한 역할을 합니다.



### Multi Kernel Estimation based Object Segmentation (https://arxiv.org/abs/2410.17064)
- **What's New**: 본 논문은 이미지의 여러 영역에 대한 커널을 추정하는 Multi-KernelGAN 접근법을 제안합니다. 이는 기존의 KernelGAN 알고리즘을 개선하여 이미지 세분화 마스크를 기반으로 두 개의 상이한 커널을 추정합니다.

- **Technical Details**: Multi-KernelGAN은 사전 훈련된 YOLOv8 모델을 이용하여 이미지의 주요 객체를 식별하고, SAM(Segment Anything Model)을 통해 각 영역에 대한 세분화 마스크를 생성합니다. 이후 각 세그먼트에 대해 커널을 추정하고, ZSSR(Zero-Shot Super-Resolution)을 통해 슈퍼 해상도를 적용합니다. 이 과정은 이미지 세분화, 커널 추정 및 슈퍼 해상도를 포함하는 세 가지 단계로 진행됩니다.

- **Performance Highlights**: 실험 결과, Multi-KernelGAN 기술이 전통적인 단일 커널 방법보다 슈퍼 해상도 작업에서 뛰어난 성능을 보였습니다. 특히 YOLO와 SAM을 결합한 방법이 커널 추정에 가장 효과적임을 입증했습니다.



### SPVSoAP3D: A Second-order Average Pooling Approach to enhance 3D Place Recognition in Horticultural Environments (https://arxiv.org/abs/2410.17017)
Comments:
          This work has been accepted to IROS 2024

- **What's New**: 본 연구는 3D LiDAR 기반 장소 인식 기술이 농업 환경에서 어떻게 적용될 수 있는지를 탐구한다. 특히 농업 환경에서의 레이저 빔 투과성으로 인해 발생하는 희소하고 중첩된 LiDAR 스캔 문제를 해결하기 위해 SPVSoAP3D라는 새로운 모델링 접근법을 제안한다.

- **Technical Details**: SPVSoAP3D는 복셀 기반(feature extraction network) 기능 추출 네트워크와 두 번째 평균 풀링 연산자를 기반으로 한 집계 기법(aggregation technique)을 결합한 모델이다. 이 접근 방식은 HORTO-3DLM 데이터셋을 확장하고, 농업 환경에서의 이종 지시자 모호성을 해결하기 위해 제안되었다.

- **Performance Highlights**: SPVSoAP3D는 기존의 SOTA 모델들과 비교하여 더 높은 성능을 기록했다. 특히 두 번째 평균 풀링 방식이 기존의 최대(MAX) 풀링 방식보다 농업 환경에서 더 효과적임을 입증했다. 검증 결과 설명 단계가 성능 향상에 기여했음을 보여주었다.



### Joint Point Cloud Upsampling and Cleaning with Octree-based CNNs (https://arxiv.org/abs/2410.17001)
Comments:
          Accepted by Computational Visual Media

- **What's New**: 이번 논문에서는 희소하거나 노이즈가 많은 데이터로부터 밀집하고 균일하게 분포된 포인트 클라우드(point cloud)를 회복하는 간단하면서도 효율적인 방법을 제안합니다. 전통적인 방식에서의 복잡한 네트워크 구조를 피하고, 단일 네트워크에서 업샘플링(upsampling)과 정리를 동시에 수행할 수 있도록 설계되었습니다.

- **Technical Details**: 제안된 방법은 수정된 octree 기반 3D U-Net(OUNet)을 활용하며, 이전 작업들이 포인트 클라우드 패치를 개별적으로 처리한 것과는 달리, 전체 포인트 클라우드를 동시에 처리합니다. 이를 통해 구현이 용이해지고, 최소 47배 빠른 추론 시간을 제공합니다. 배치 정규화(batch normalization) 대신 그룹 정규화(group normalization)를 사용하여 업샘플링과 청소 작업에서의 특징 혼합을 방지합니다.

- **Performance Highlights**: 광범위한 실험을 통해, Chamfer Distance (CD), Hausdorff Distance (HD), Point-to-Surface Distance (P2F) 지표에서 최첨단(최고의 성능) 결과를 달성함을 보여주었습니다. 우리 방법은 다른 방법들보다 Chamfer 거리가 더 좋은 성과를 내면서도 47배 이상 빠른 속도를 자랑합니다.



### AGSENet: A Robust Road Ponding Detection Method for Proactive Traffic Safety (https://arxiv.org/abs/2410.16999)
Comments:
          21 pages, 15 figures

- **What's New**: 새로운 연구에서는 도로의 고인 물을 정확하게 감지하기 위한 AGSENet(Sself-Attention-based Global Saliency-Enhanced Network)라는 혁신적인 접근 방식을 제안하였습니다. 이는 복잡한 도로 표면 및 변화하는 색상 특성으로 인해 감지가 어려웠던 문제를 해결하기 위한 것입니다.

- **Technical Details**: AGSENet는 Channel Saliency Information Focus (CSIF) 및 Spatial Saliency Information Enhancement (SSIE) 모듈을 통해 주목성을 향상시키며, 자기 주의(self-attention) 메커니즘을 이용하여 기능 간의 상관관계를 파악합니다. CSIF 모듈은 인코더에서 공간 및 채널 정보를 융합하여 유사한 특성을 강조하고, SSIE 모듈은 디코더에서 서로 다른 기능 수준 간의 상관관계를 통해 노이즈를 줄이고 엣지 기능을 정제합니다.

- **Performance Highlights**: AGSENet는 Puddle-1000, Foggy-Puddle 및 Night-Puddle 데이터셋에서 각각 2.03%, 0.62%, 1.06%의 IoU 개선을 보여주며, 현재 가장 진보된 성과를 기록했습니다. 또한, 중요 라벨링 및 주석의 오작성을 수정하였고, 낮은 조도 및 안개 조건에서의 감지를 위한 새로운 데이터셋을 구축하여 신뢰성을 검증했습니다.



### E-3DGS: Gaussian Splatting with Exposure and Motion Events (https://arxiv.org/abs/2410.16995)
Comments:
          The source code and dataset will be available at this https URL

- **What's New**: E-3DGS는 모션 이벤트와 노출 이벤트를 통합하여 고속 상황에서도 고품질 3D 복원을 가능하게 하는 새로운 접근 방식을 제안합니다. 이 방법은 로봇 비전 응용 프로그램에서 흔히 발생하는 모션 블러와 불충분한 조명과 같은 문제를 해결합니다.

- **Technical Details**: E-3DGS는 세 가지 운영 모드를 제안합니다: 고품질 복원 모드(그레이스케일 이미지를 생성하여 조명 저조 및 고다이내믹 레인지에서 복원 정확성을 높임), 빠른 복원 모드(고화질 이벤트 카메라를 활용하여 고속 상황에서 3D 복원), 균형 잡힌 하이브리드 모드(노출 이벤트와 모션 이벤트를 결합하여 속도와 품질을 조율함)입니다.

- **Performance Highlights**: E-3DGS는 EventNeRF와 비교하여 PSNR이 5.68 dB 향상되고 렌더링 속도가 79.37 FPS에 달하며, 이벤트-그레이스케일 학습 기반 3DGS와 비교했을 때 PSNR의 10.89 dB 증가를 달성하였습니다. 이 방법은 EME-3D라는 첫 실세계 3D 데이터셋을 구축하여 검증되었습니다.



### Leaky ReLUs That Differ in Forward and Backward Pass Facilitate Activation Maximization in Deep Neural Networks (https://arxiv.org/abs/2410.16958)
- **What's New**: 이 논문은 Activation Maximization (AM)이 ReLU와 Leaky ReLU 기능을 포함한 간단한 함수들에 대해 최적의 입력 자극을 생성하지 못한다고 보고합니다. 이러한 결과는 AM의 실용성과 그에 의해 생성된 이미지의 시각적 해석에 대한 의문을 제기합니다. 논문에서는 Leaky ReLU의 높은 음의 기울기를 활용하여 AM의 성능을 향상시키는 새로운 방법을 제안합니다.

- **Technical Details**: 이 연구에서는 Leaky ReLU을 가진 보조 네트워크를 사용하여 기울기 계산을 위한 프록시로 활용하는 ProxyGrad 알ゴ리즘을 제안합니다. 이 보조 네트워크는 원래 네트워크와 동일한 가중치를 가지지만, 음의 기울기가 명확히 더 큰 Leaky ReLU를 포함합니다. 원래 네트워크는 순전파를 처리하고, 보조 네트워크는 역전파를 실행하여 기울기를 계산합니다. 이 방법은 로컬 최대값을 줄이고 최적화 효율성을 향상시킵니다.

- **Performance Highlights**: ProxyGrad 알고리즘을 사용한 결과, ResNet18 모델은 ImageNet 데이터셋에서 더 높은 활성화 값을 기록하며, 전통적인 네트워크보다 더 나은 시각적 명확성을 제공합니다. 또한, ProxyGrad은 CNN의 가중치를 훈련시키는데 효과적이며, Caltech101, Caltech-UCSD Birds-200-2011, 102 Category Flower 데이터셋에서 전통적인 최적화 방법보다 좋은 성능을 보입니다.



### PGCS: Physical Law embedded Generative Cloud Synthesis in Remote Sensing Images (https://arxiv.org/abs/2410.16955)
Comments:
          20 pages, 16 figures

- **What's New**: 이번 논문에서는 물리 법칙이 포함된 생성 구름 합성 방법(Physical Law Embedded Generative Cloud Synthesis, PGCS)을 제안하여, 다양한 현실적인 구름 이미지를 생성하고 데이터 품질을 향상시키며, 구름 보정(cloud correction), 구름 탐지(cloud detection), 데이터 증강(data augmentation) 같은 후속 작업을 촉진할 수 있는 방안을 제시했습니다.

- **Technical Details**: PGCS 방법은 공간 합성(spatial synthesis)과 스펙트럴 합성(spectral synthesis)의 두 가지 주요 단계로 구성되어 있습니다. 공간 합성 단계에서는 스타일 기반 생성적 적대 신경망(StyleGAN)을 활용하여 단일 채널 구름을 무한정 생성하고, 스펙트럴 합성 단계에서는 대기 산란 법칙(atmospheric scattering law)을 포함하여 단일 채널 구름을 다채널 구름으로 변환합니다.

- **Performance Highlights**: PGCS는 두 단계에서 높은 정확도를 달성하며, 기존의 세 가지 구름 합성 방법보다 성능이 우수합니다. 또한, PGCS에서 개발한 두 가지 구름 보정 방법은 최신 기술들과 비교했을 때 뛰어난 성능을 보였습니다. 다양한 센서의 데이터에서 PGCS의 적용 가능성을 성공적으로 확장한 결과도 보고되었습니다.



### Towards Real Zero-Shot Camouflaged Object Segmentation without Camouflaged Annotations (https://arxiv.org/abs/2410.16953)
- **What's New**: 이 논문에서는 수작업 주석 없이도 모든 위장 객체를 효과적으로 탐지할 수 있는 Zero-shot Camouflaged Object Segmentation (COS) 프레임워크를 제안합니다. 이는 기존의 대규모 주석 데이터셋에 대한 의존도를 줄이고, 강력한 zero-shot 학습을 가능하게 합니다. 또한, Multimodal Large Language Model (M-LLM)과 Masked Image Modeling (MIM) 기반 이미지 인코더의 조합을 통해 세분화된 객체 경계를 캡처합니다.

- **Technical Details**: 제안된 프레임워크는 Masked Image Modeling (MIM) 기반의 이미지 인코더, 파라미터 효율적 파인 튜닝(Parameter-Efficient Fine-Tuning) 및 Multi-scale Fine-grained Alignment (MFA) 메커니즘을 포함합니다. 이 구조는 강력한 로컬 패턴 바이어스와 광범위한 의미적 특성 공간을 활용하여 zero-shot 학습을 촉진합니다. M-LLM은 시각적 단서와 함께 캡션 임베딩을 생성하며, 이 임베딩은 MFA를 통해 정밀하게 정렬되어 복잡한 의미적 맥락을 해석할 수 있습니다.

- **Performance Highlights**: 본 연구는 CAMO에서 72.9% 및 COD10K에서 71.7%의 $F_{eta}^w$ 점수를 달성하며, 기존의 약한 감독 방법과 경쟁할 수 있는 성능을 보였습니다. 또한, M-LLM을 비활성화하여 전통적인 end-to-end 모델과 유사한 18.1 FPS의 추론 속도를 달성하였습니다.



### ISImed: A Framework for Self-Supervised Learning using Intrinsic Spatial Information in Medical Images (https://arxiv.org/abs/2410.16947)
Comments:
          11 pages, 4 figures

- **What's New**: 이 논문은 Self-Supervised Learning (SSL)을 활용하여 의료 영상에서 해석 가능한 표현을 학습하는 방법을 제시합니다. 특히 저자들은 집합적 관찰을 통해 의료 이미지 간 변동성이 기존 데이터 비전 기준보다 훨씬 낮다는 점을 강조합니다. 이로 인해 신체 구조의 유사성을 이용한 자가 감독 목표를 설정하여 공간적인 위치를 포착할 수 있는 잠재 표현을 생성합니다.

- **Technical Details**: 제안하는 방법은 이미지 크롭을 샘플링하고, 이 크롭들 간의 거리를 비교하는 거리 행렬을 생성하는 것입니다. 학습된 잠재 표현 벡터를 통해 물리적 위치를 반영하며, 이 과정에서 L2 거리 (L2 distance) 계산을 사용하여 정확한 위치 파악을 목표로 합니다. ISImed는 BarlowTwins와 같은 최신 SSL 아키텍처와 결합하여 정보 붕괴를 방지하면서 물리적 공간과 유사한 잠재 표현을 학습합니다.

- **Performance Highlights**: 두 개의 공개 의료 영상 데이터셋에서 실험한 결과, 제안한 방법이 기존의 simCLR 및 BarlowTwins와 비교하여 데이터의 구조를 효과적으로 포착할 수 있음을 보여주었습니다. 또한, ISImed의 학습된 표현은 downstream 분류 작업으로 전이 가능함을 확인하여, 의료 영상 처리 과정에서의 유용성을 입증했습니다.



### DiP-GO: A Diffusion Pruner via Few-step Gradient Optimization (https://arxiv.org/abs/2410.16942)
- **What's New**: 논문에서는 효율적인 diffusion 모델 생성의 새로운 가지치기 방법인 DiP-GO (Diffusion Pruning via Few-step Gradient Optimization)를 제안합니다. 이 방법은 퍼미터 수를 줄이면서도 성능을 유지하도록 설계된 차별화된 파라미터와 서브넷 검색 프로세스를 사용합니다.

- **Technical Details**: DiP-GO 방법은 SuperNet을 기반으로 하여 여러 denoising 단계 간의 유사한 특징을 활용합니다. 각 계산 블록을 유지하거나 건너뛸지를 예측하기 위해 학습 가능한 쿼리를 사용하는 신경망 가지치기기를 설계하였습니다.

- **Performance Highlights**: DiP-GO는 Stable Diffusion 1.5 모델에서 4.4배의 속도 향상을 이루었으며, 정확도 손실 없이 이전 방식들보다 뛰어난 성능을 발휘하였습니다.



### LIMIS: Towards Language-based Interactive Medical Image Segmentation (https://arxiv.org/abs/2410.16939)
- **What's New**: LIMIS라는 최초의 순수 언어 기반 의료 이미지 분할(interactive medical image segmentation) 모델을 소개합니다. 이 모델은 Grounded SAM을 의료 분야에 적용하고 의사가 언어만으로 세그멘테이션(mask) 작업에 지식을 통합할 수 있도록 설계된 언어 기반의 상호작용 전략을 사용합니다.

- **Technical Details**: LIMIS는 세 가지 주요 구성 요소로 이루어져 있습니다: 언어를 바운딩 박스로 변환하는 Lang2BBox 모듈, 바운딩 박스를 세그멘테이션으로 변환하는 BBox2Mask 모듈, 그리고 사용자 상호작용 루프입니다. 초기 세그멘테이션 마스크는 자연어 입력을 통해 생성되며, 이 과정에서 Grounding DINO와 기존 SAM 아키텍처를 의료 도메인에 맞게 최적화합니다.

- **Performance Highlights**: LIMIS는 여러 의료 데이터 세트에서 전문가와의 사용성 연구를 통해 성능이 검증되었으며, 고품질의 세그멘테이션 마스크를 생성하는 것을 확인했습니다. 이 시스템은 특히 의사가 다른 업무를 수행해야 할 때도 손을 사용하지 않고 관여할 수 있는 가능성을 열어줍니다.



### Hierarchical Clustering for Conditional Diffusion in Image Generation (https://arxiv.org/abs/2410.16910)
Comments:
          25 pages, submitted to ICLR 2025

- **What's New**: 본 논문에서는 TreeDiffusion이라는 새로운 딥 생성 모델을 소개합니다. 이 모델은 계층적 클러스터에 조건을 걸어 고품질의 클러스터 특정 샘플을 생성함으로써 VAE의 생성 한계를 극복합니다.

- **Technical Details**: TreeDiffusion은 두 단계로 나누어진 프레임워크로, 첫 번째 단계는 VAE 기반의 생성적 계층 클러스터링 모델이며, 두 번째 단계는 이러한 클러스터에 조건을 건 확산 모델입니다. 이 모델은 데이터의 계층 구조를 학습하고, 선택된 경로를 따라 노이즈를 제거하면서 이미지를 생성합니다.

- **Performance Highlights**: 실험적으로, 계층적 클러스터에 조건을 건 확산 모델이 생성 성능을 획기적으로 향상시킨 것을 보여줍니다. 이것은 VAE 기반 클러스터링 모델이 가지는 생성적 한계를 효과적으로 해결하면서도 클러스터링 성능을 유지하는 강력한 방법입니다.



### Mitigating Vanishing Activations in Deep CapsNets Using Channel Pruning (https://arxiv.org/abs/2410.16908)
- **What's New**: 이 논문은 Deep Capsule Networks의 vanishing activation(소실 활성화) 문제를 해결하는 새로운 접근 방식을 제시합니다. 기존의 모델 프루닝(pruning) 방법과 달리, 본 연구는 Capsule 네트워크의 성능을 향상시키기 위해 중요도를 평가하여 convolutional 채널을 프루닝하는 방법을 사용합니다.

- **Technical Details**: Capsule Networks는 Convolutional Neural Networks(CNNs)보다 part-whole relationships(부분-전체 관계) 학습에서 우수한 성능을 보입니다. 그러나 깊은 Capsule Networks는 깊이 증가에 따른 vanishing activations 문제가 발생하여 스케일러빌리티가 부족합니다. 이 연구는 구조적 프루닝(structured pruning)과 Correlation Coefficient Matrix(CCM) 손실을 결합하여 깊은 CapsNet의 소실 활성화를 완화하는 방법을 살펴봅니다.

- **Performance Highlights**: 제안된 방법은 비프루닝 모델보다 더 나은 정확도를 달성하며, Capsule Networks의 inactive capsules(비활성 캡슐) 수를 줄이면서 모델의 정확성을 높입니다. 논문에 포함된 실험 결과는 제안된 기법이 성능 향상에 효과적임을 보여줍니다.



### Enhancing Generalization in Convolutional Neural Networks through Regularization with Edge and Line Features (https://arxiv.org/abs/2410.16897)
- **What's New**: 본 논문은 CNN(Convolutional Neural Networks)이 은닉층에서 edge 및 line features를 활용하도록 유도하는 새로운 정규화 접근 방식을 제안합니다. 이러한 접근은 일반적인 커널 학습 대신 edge 및 line 탐지 커널로 합성곱 층을 제한하여 모델을 정규화하고, 특히 작은 데이터 세트에서 일반화 성능을 개선합니다.

- **Technical Details**: 제안된 방법은 Pre-defined Filter Modules(PFM)을 사용하여 입력 데이터를 3x3 고정 edge 및 line 필터 집합으로 합성함으로써, CNN의 필터를 사전 정의된 필터로 대체합니다. 이 후에 ReLU(활성화 함수)를 사용하여 긍정적인 반응이 없는 정보를 제거하고, 1x1 합성곱 층을 통해 선형 결합을 생성합니다. 필터의 개수는 9개 이상일 때 최적의 성능을 보입니다.

- **Performance Highlights**: 본 연구는 4개의 세밀한 분류 데이터 세트에서 5-11% 포인트의 테스트 정확도 향상을 보였으며, 한정된 학습 데이터와 동일한 수의 학습 가능 매개변수로 결과를 도출했습니다. 또한, 사전 정의된 필터가 성능에 미치는 영향을 연구한 결과 그 수가 성능에 미치는 영향은 적지만, 필터의 크기는 중요하다는 것을 확인했습니다.



### VistaDream: Sampling multiview consistent images for single-view scene reconstruction (https://arxiv.org/abs/2410.16892)
Comments:
          Project Page: this https URL

- **What's New**: VistaDream라는 새로운 프레임워크를 제안하여 단일 뷰 이미지로부터 3D 장면을 재구성합니다. 기존의 방법들은 입력 이미지와 생성된 이미지 간의 일관성을 구축하는 데 집중했지만 생성된 이미지 간의 일관성은 간과했습니다. VistaDream은 이 문제를 해결하기 위해 두 단계 파이프라인을 사용합니다.

- **Technical Details**: VistaDream의 첫 번째 단계는 입력 뷰에서 카메라를 약간 확대하여 글로벌 코아스 3D 스캐폴드를 구축합니다. 여기서는 inpainted 경계와 추정된 깊이 맵을 활용합니다. 이후 두 번째 단계에서 다중 뷰 일관성 샘플링(Multiview Consistency Sampling, MCS)을 통해 생성된 새 뷰 이미지 간의 일관성을 강화합니다.

- **Performance Highlights**: 실험 결과, VistaDream은 기존의 diffusion 모델에 대한 학습이나 미세 조정 없이도 일관성 있고 고품질의 새로운 뷰 신합성을 성공적으로 달성하며, 기존의 방법들과 비교해 큰 차이를 보였습니다.



### Network Inversion for Training-Like Data Reconstruction (https://arxiv.org/abs/2410.16884)
- **What's New**: 본 논문에서는 훈련된 모델로부터 훈련 데이터와 유사한 데이터를 재구성하는 신기술인 Training-Like Data Reconstruction (TLDR)을 소개합니다. 이 방법은 네트워크 역전(network inversion) 기반 접근법으로, 훈련된 모델의 가중치로부터 훈련 데이터의 정보를 추론할 수 있는 잠재적인 프라이버시 위험을 강조합니다.

- **Technical Details**: TLDR은 단일 조건 생성기(conditional generator)를 사용하여 클래스에 해당하는 입력 공간을 학습하는 포괄적인 네트워크 역전 기술을 포함합니다. 이 과정에서 다양한 손실 함수(cross-entropy, KL Divergence, cosine similarity, feature orthogonality)를 사용하여 생성기가 훈련 데이터와 유사한 데이터를 재구성하도록 유도합니다. 또한 이 방법은 CNN(convolutional neural network)의 특성을 활용하여 생성된 샘플이 훈련 데이터에 맞추어지도록 합니다.

- **Performance Highlights**: 제안된 방법은 MNIST, FashionMNIST, SVHN, CIFAR-10 등의 여러 비전 분류 데이터셋에서 실험을 통해 유사한 데이터를 재구성할 수 있음을 보여줍니다. 이는 기계 학습 모델을 공유하는 과정에서 발생할 수 있는 프라이버시 위험을 부각시킵니다.



### Bridging the Modality Gap: Dimension Information Alignment and Sparse Spatial Constraint for Image-Text Matching (https://arxiv.org/abs/2410.16853)
- **What's New**: DIAS라는 새로운 방법을 제안하여 이미지와 텍스트 간의 모달리티 격차를 해소하고, 이미지-텍스트 매칭의 성능을 향상시킵니다. 이를 위해 두 가지 접근법이 도입되며, 첫째로 서로 다른 모달리티의 임베딩 정보를 일치시키고, 둘째로 잡음이 있는 쌍의 제약을 도입하여 의미적 정렬의 효과를 보장합니다.

- **Technical Details**: DIAS는 모달리티 간의 임베딩 정보를 해당 차원에 맞추어 조정하는 차원 정보 정렬 방법을 사용합니다. 이는 두 가지 주요 접근 방식으로 나뉘는데, 첫 번째는 해당 차원에서 서로 다른 모달리티의 임베딩 간 상관관계를 강화하는 것이고, 두 번째는 맞지 않는 쌍의 공간적 제약 조건을 활용하는 것입니다. 또한, 희소 상관관계 알고리즘을 도입하여 최상위 상관관계를 선택하여 중요한 특징을 학습하도록 합니다.

- **Performance Highlights**: DIAS는 Flickr30k 및 MSCOCO 벤치마크에서 4.3%-10.2%의 rSum 개선을 보여 주며, 이미지-텍스트 매칭 태스크에서 기존 방법들에 비해 우수한 성능을 입증하였습니다.



### MPDS: A Movie Posters Dataset for Image Generation with Diffusion Mod (https://arxiv.org/abs/2410.16840)
- **What's New**: 영화 포스터 생성에 최적화된 'Movie Posters DataSet (MPDS)'를 제안하며, 이는 37만개 이상의 이미지-텍스트 쌍과 8천개 이상의 배우 이미지를 포함한 최초의 데이터셋이다.

- **Technical Details**: MPDS는 영화 제목, 장르, 출연진 및 시놉시스 등 자세한 포스터 설명을 포함하여, 대규모 비전-언어 모델을 활용해 자동으로 생성된 비전-인식 프롬프트를 활용하여 영화 시놉시스와 연계하여 작성된다. 이를 통해 포스터 캡션을 도출하고, 다중 조건 확산 프레임워크를 바탕으로 포스터 프롬프트, 캡션, 배우 이미지를 입력으로 사용한다.

- **Performance Highlights**: 실험 결과, MPDS 데이터셋이 개인화된 영화 포스터 생성에서 중요한 역할을 한다는 것을 증명했으며, 기존의 일반화된 데이터셋이 아닌 포스터 맞춤형 모델이 우수한 성능을 보임을 나타낸다.



### PerspectiveNet: Multi-View Perception for Dynamic Scene Understanding (https://arxiv.org/abs/2410.16824)
Comments:
          6 pages, 2 figures

- **What's New**: 이 논문에서는 여러 카메라 뷰를 통해 상세한 설명을 생성하기 위한 새로운 경량 모델, PerspectiveNet을 소개합니다. 이 모델은 시각적 특성을 고정 크기의 텐서로 변환하기 위한 컴팩트 커넥터 모듈과 대형 언어 모델(LLM)을 활용하여 긴 문장을 효과적으로 생성하는 데 중점을 둡니다.

- **Technical Details**: PerspectiveNet의 커넥터 모듈은 시각적 특성을 LLM 임베딩에 매핑하고, 설명 생성에 필요한 주요 정보를 강조하며, 고정 크기의 피처 매트릭스를 생성하는 세 가지 주요 목표로 설계되었습니다. 또한, 정확한 프레임 순서 감지를 위한 부가적 작업을 통합하여 설명 생성을 위한 올바른 프레임 시퀀스를 검색할 수 있도록 합니다.

- **Performance Highlights**: Traffic Safety Description and Analysis 작업을 위해 훈련된 모델은 경량화되어 효율적인 훈련과 추론을 보장하며, 다양한 카메라 뷰에서 사건에 대한 자세하고 세밀한 설명을 생성하는 데 매우 효과적인 성능을 나타냅니다.



### AttriPrompter: Auto-Prompting with Attribute Semantics for Zero-shot Nuclei Detection via Visual-Language Pre-trained Models (https://arxiv.org/abs/2410.16820)
Comments:
          This article has been accepted for publication in a future issue of IEEE Transactions on Medical Imaging (TMI), but has not been fully edited. Content may change prior to final publication. Citation information: DOI: this https URL . Code: this https URL

- **What's New**: 이 논문에서는 Grounded Language-Image Pre-training (GLIP)이라는 객체 기반 시각-언어 선행 학습 모델을 활용하여 제로샷(nuclei detection) 핵 발견에 대한 새로운 접근법을 제시하고 있습니다. 특히, AttriPrompter라는 새로운 자동 프롬프트 설계 파이프라인을 도입하여, 주관적인 수동 설계를 피하며, 핵의 속성 생성을 통해 세밀한 텍스트 프롬프트를 생성합니다.

- **Technical Details**: AttriPrompter는 속성 생성, 속성 증강(attribute augmentation), 연관성 정렬(relevance sorting) 단계를 포함하여, VLPM의 시맨틱을 활용하여 핵의 형상 및 색상을 설명하는 속성 단어를 생성합니다. 이 구조는 GLIP의 초기 예측을 사용하여 지식 증류(knowledge distillation) 프레임워크를 통해 높은 핵 밀도의 문제를 다루며, 학습된 지식을 유지합니다.

- **Performance Highlights**: 우리의 방법은 라벨 없는 제로샷 핵 발견에서 뛰어난 성능을 보이며, 기존의 비지도 방법들을 능가하고 훌륭한 일반성을 입증합니다. 다양한 데이터 세트와 탐지 아키텍처에서 그 일반성을 확인하였습니다.



### Evaluating the Effectiveness of Attack-Agnostic Features for Morphing Attack Detection (https://arxiv.org/abs/2410.16802)
Comments:
          Published in the 2024 IEEE International Joint Conference on Biometrics (IJCB)

- **What's New**: 이 연구는 morphing 공격(morphing attacks)에 대한 새롭고 효과적인 탐지 방법을 제안합니다. 기존의 전통적인 탐지 방법과 달리, 학습된 심층 신경망을 사용한 공격 비무관(feature-extraction) 기능을 기반으로 합니다.

- **Technical Details**: 우리는 크게 세 가지 공격 데이터를 생성하여 실험을 실시했습니다: landmark 기반(landmark-based), GAN 기반(GAN-based), 그리고 확산 기반(diffusion-based) morphs로, 공격 비무관 특징을 사용하여 두 가지 감지 시스템을 개발했습니다: 어드버서리 탐지기와 일급 탐지기. 제조된 이미지에 대한 탐지 정확도를 높이는 데 초점을 맞췄습니다.

- **Performance Highlights**: 우리의 방법은 기존의 감독(convolutional neural network(CNN)) 훈련 모델들보다 우수한 성능을 발휘하며, 다양한 공격 유형 및 출처 데이터셋에서 강력한 일반화 능력을 보여줍니다. 공격 비무관 특징을 사용하는 것만으로도 대부분의 시나리오에서 효과적으로 morphing 공격을 탐지할 수 있음을 확인했습니다.



### One-Step Diffusion Distillation through Score Implicit Matching (https://arxiv.org/abs/2410.16794)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 이 논문에서는 'Score Implicit Matching (SIM)'이라는 새로운 방법론을 소개하며, 이는 기존의 diffuion 모델을 단일 단계 생성 모델로 효율적으로 변환합니다. 중요한 점은 SIM이 원래 모델과 거의 동일한 샘플 생성 능력을 유지하면서, 데이터 샘플 없이도 훈련 없이 적용할 수 있다는 것입니다.

- **Technical Details**: SIM은 점수 기반 손실 함수를 사용하여 pre-trained diffusion 모델을 단일 단계 생성 모델로 변환합니다. 이 접근법의 핵심 아이디어는 점수 기반 분산(score-based divergence) 간의 기울기를 효율적으로 계산할 수 있는 'score-gradient theorem'을 이용한다는 것입니다. 이를 통해 모델의 훈련을 효율적으로 진행할 수 있습니다.

- **Performance Highlights**: CIFAR10 데이터셋에서 SIM은 비조건부 생성의 경우 FID(Frechet Inception Distance)가 2.06, 클래스 조건부 생성의 경우 1.96을 달성했습니다. 또한, 텍스트-이미지(T2I) 생성에 있어 기존의 다단계 모델과 비교해 성능 저하 없이 뛰어난 심미적 점수 6.42를 기록하며 단일 단계 생성 모델들이 SDXL-TURBO 및 HYPER-SDXL과 같은 기존 모델들보다 높은 성능을 보였습니다.



### The Scene Language: Representing Scenes with Programs, Words, and Embeddings (https://arxiv.org/abs/2410.16770)
Comments:
          Project page: this https URL

- **What's New**: Scene Language는 시각 장면의 구조, 의미 및 정체성을 간결하고 정확하게 설명하는 새로운 시각 장면 표현 방식입니다. 이 방식은 프로그램, 자연어의 단어 및 임베딩의 세 가지 주요 구성 요소로 장면을 표현합니다.

- **Technical Details**: Scene Language는 장면의 계층적 및 관계적 구조를 정의하는 프로그램, 각 엔티티의 의미적 클래스를 요약하는 자연어의 단어 및 각 엔티티의 시각적 정체성을 캡처하는 임베딩으로 구성됩니다. 이 표현 방식은 사전 훈련된 언어 모델을 기반으로 하여 훈련이 필요 없는 추론 기술로 텍스트 또는 이미지 입력에서 추론할 수 있습니다.

- **Performance Highlights**: 기존의 장면 그래프와 같은 표현 방식과 비교하여 Scene Language는 복잡한 장면을 더 높은 충실도로 생성하면서 장면 구조를 명시적으로 모델링하여 정밀한 제어 및 편집을 가능하게 합니다. 이 시스템은 고품질 3D 및 4D 장면 생성 및 편집을 위한 강력하고 자동화된 솔루션을 제공합니다.



### DSORT-MCU: Detecting Small Objects in Real-Time on Microcontroller Units (https://arxiv.org/abs/2410.16769)
Comments:
          arXiv admin note: text overlap with arXiv:2311.07163

- **What's New**: 이 논문은 소형 객체 감지를 위한 새로운 어댑티브 타일링(adaptive tiling) 방법을 제안합니다. 이 방법은 YOLO 기반 모델과 인기 있는 FOMO 네트워크에 적용되어, 저전력 MCU(microcontroller unit)에서 정확도를 유지하면서도 소형 객체 감지가 가능하게 합니다.

- **Technical Details**: 제안된 타일링 방법은 FOMO 및 TinyissimoYOLO 네트워크에 적용되며, RISC-V 기반의 GAP9 MCU에서 실험적으로 검증되었습니다. 이 방법은 F1-score를 최대 225%까지 향상시키고, FOMO에서는 평균 객체 수 오류를 76%, TinyissimoYOLO에서는 89%까지 줄입니다. 이 연구에서는 soft F1 loss를 사용하여 FOMO 네트워크의 비최대 억제(non-maximum suppression) 역할을 수행할 수 있음을 보여줍니다.

- **Performance Highlights**: 제안된 방법은 높은 해상도의 이미지를 사용하여 효율적인 실시간 객체 감지를 가능하게 하며, F1-score는 58%에서 95%, 지연 시간은 0.6 ms/Inference에서 16.2 ms/Inference, 에너지 효율성은 31 uJ/Inference에서 1.27 mJ/Inference까지 구현되었습니다.



### SpikMamba: When SNN meets Mamba in Event-based Human Action Recognition (https://arxiv.org/abs/2410.16746)
Comments:
          8 pages, 4 figures

- **What's New**: 본 연구에서는 SpikMamba 프레임워크를 제안하여 이벤트 카메라로부터 수집된 데이터의 공간적 희소성과 높은 시간적 해상도를 효과적으로 처리하여 인간 행동 인식(HAR)의 성능을 개선합니다.

- **Technical Details**: SpikMamba는 스파이킹 신경망(Spiking Neural Networks, SNN)과 Mamba를 결합하여 이벤트 데이터의 글로벌 및 로컬 시계열 의존성(global and local temporal dependencies)을 모델링합니다. 이 프레임워크의 핵심 설계에는 스파이크 기반의 선형 주의 메커니즘(spike-based linear attention mechanism)이 포함되어 있습니다.

- **Performance Highlights**: SpikMamba는 PAF, HARDVS, DVS128, E-FAction 데이터셋에서 기존 최첨단 방법들보다 각각 1.45%, 7.22%, 0.15%, 3.92% 성능을 초과하여 놀라운 인식 성능을 달성했습니다.



### Time-Resolved MNIST Dataset for Single-Photon Recognition (https://arxiv.org/abs/2410.16744)
Comments:
          12 pages, 4 figures. Accepted for Workshop on Synthetic Data for Computer Vision at ECCV 2024

- **What's New**: 이번 논문에서는 시간 해상도가 있는 단일 광자 이미지의 시뮬레이션 프로세스를 제시합니다. 기존의 SPAD(imaging) 센서의 한계를 극복하고, 다양한 크기의 시간 해상도 SPAD(Single-Photon Avalanche Diodes) 배열을 생성할 수 있는 소프트웨어 프로토타입을 구현하였습니다. 이 시뮬레이터는 시간에 따른 광자 검출의 생성을 모델링하여, 비공식적인 데이터 세트를 제공하도록 설계되었습니다.

- **Technical Details**: 단일 광자 감지기인 SPAD는 비동기적으로 작동하며, 각 픽셀은 광자 도달 시 즉시 신호를 생성합니다. 시뮬레이션은 광자의 도달 통계와 잡음원을 모두 고려하여 신뢰할 수 있는 데이터 스트림을 생성합니다. 시뮬레이터는 MNIST 데이터셋의 타임 리졸브 버전을 생성하며, 이를 공개하여 조명 상태에 따른 CNN(Classification) 성능을 연구할 수 있도록 하고 있습니다.

- **Performance Highlights**: TR-MNIST 데이터셋은 낮은 조도에서 CNN 분류기의 성능을 평가하는 기준으로 활용될 수 있습니다. 이 데이터셋은 전통적인 MNIST 이미지에 기반하여 다양한 조도 수준에서 생성된 두 가지 버전(TR-MNIST 및 TR-MNIST-rec)을 제공합니다. TR-MNIST-rec은 매우 낮은 조명 조건에서의 이미지 분류 성능 저하를 평가하는 데 기여할 것으로 기대됩니다.



### Polyp-E: Benchmarking the Robustness of Deep Segmentation Models via Polyp Editing (https://arxiv.org/abs/2410.16732)
- **What's New**: 이 논문은 폴립 자동 세분화의 견고성(robustness)을 평가하고 향상시키기 위한 새로운 접근법과 데이터셋을 제안합니다. 특히, 실제 폴립 이미지의 속성(attribute)을 수정하여 생성한 새로운 데이터셋인 Polyp-E를 통해 모델의 효율성을 확인합니다.

- **Technical Details**: Latent Diffusion Model을 기반으로 하여 실제 폴립 이미지의 속성을 수정하는 데이터 편집 시나리오를 개발했습니다. 이 방법은 이미지의 왜곡을 최소화하여 현실감과 다양성을 유지하며, 세 가지 속성 변경 시나리오(비폴립, 크기 변화, 위치 변화)를 포함한 다양한 모델을 벤치마킹합니다.

- **Performance Highlights**: 대부분의 폴립 세분화 모델이 속성 변화에 민감하다는 결과가 나타났습니다. 또한, 제안한 데이터 편집 파이프라인은 훈련 데이터와 검증 데이터 모두에서 일반화 능력을 개선하는 효과를 보여주었습니다.



### Progressive Compositionality In Text-to-Image Generative Models (https://arxiv.org/abs/2410.16719)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)을 활용하여 사실적이고 복잡한 시나리오를 구성하고, 확산 모델과 함께 Visual-Question Answering (VQA) 시스템을 사용하여 15,000 쌍의 고품질 대비 이미지로 구성된 데이터셋인 ConPair를 자동으로 구축하는 방법을 소개합니다.

- **Technical Details**: 이 연구는 새로운 멀티스테이지 커리큘럼인 EvoGen을 제안하여 확산 모델의 대비 학습을 개선합니다. 이 과정은 단일 객체-속성 구성 학습, 두 객체 간의 속성 결합 마스터링, 그리고 여러 객체가 있는 복잡한 장면 처리의 세 가지 하위 작업으로 나뉩니다.

- **Performance Highlights**: EvoGen을 통해 모델의 구성적 이해가 크게 향상되었으며, 기존의 대부분의 생성 기반 방법을 초월하는 성능을 발휘하는 것이 실험을 통해 입증되었습니다.



### Development of CNN Architectures using Transfer Learning Methods for Medical Image Classification (https://arxiv.org/abs/2410.16711)
- **What's New**: 본 논문에서는 의료 이미지 분류에서 CNN 아키텍처의 발전을 조사하고 Transfer Learning 기법을 활용하여 효율성과 정확성을 높이는 방법을 제안합니다.

- **Technical Details**: Transfer Learning을 활용한 Convolutional Neural Networks (CNN) 아키텍처 개발에 관한 연구입니다. 주요 이미지 분류 과제를 시간 맵 모델(timeline mapping model)로 분석하여 최적의 CNN 아키텍처를 선택할 수 있는 정보를 제공합니다.

- **Performance Highlights**: 의료 영상 분류에서 Transfer Learning을 적용한 CNN을 통해 효율성과 정확적인 결과를 도출해냈습니다. 이를 통해 최첨단 CNN 아키텍처를 선택하는 데 도움이 됩니다.



### DI-MaskDINO: A Joint Object Detection and Instance Segmentation Mod (https://arxiv.org/abs/2410.16707)
Comments:
          16 pages, 3 figures, Conference on Neural Information Processing Systems

- **What's New**: 이번 논문은 MaskDINO 모델의 transformer decoder 처음 계층의 중간 결과를 분석하여 객체 탐지(object detection)와 인스턴스 분할(instance segmentation)의 성능 불균형 현상을 발견하는 데서 출발합니다. 이 성능 불균형이 모델의 최종 성능에 어떤 제한을 두는지를 파악하기 위해 DI-MaskDINO 모델이 제안되었습니다.

- **Technical Details**: DI-MaskDINO는 De-Imbalance (DI) 모듈과 Balance-Aware Tokens Optimization (BATO) 모듈을 MaskDINO에 통합하여 수행됩니다. DI 모듈은 균형 인식 쿼리(balance-aware query)를 생성하고 BATO는 이를 사용하여 초기 특성 토큰의 최적화를 안내합니다. DI 모듈은 성능 균형을 맞추기 위해 초기 디코더 계층에서 탐지를 강화합니다.

- **Performance Highlights**: DI-MaskDINO는 COCO와 BDD100K 벤치마크에서 MaskDINO 모델보다 +1.2 $AP^{box}$ 와 +0.9 $AP^{mask}$ 향상된 성능을 보였습니다. 또한 기존 SOTA 탐지 모델 DINO에 비해 +1.0 $AP^{box}$ 향상, SOTA 분할 모델 Mask2Former에 비해 +3.0 $AP^{mask}$ 향상을 달성했습니다.



### MPT: A Large-scale Multi-Phytoplankton Tracking Benchmark (https://arxiv.org/abs/2410.16695)
- **What's New**: 본 논문에서는 다양한 배경 정보와 관측 중의 움직임 변화를 포괄하는 다중 미세조류 추적(Multiple Phytoplankton Tracking, MPT)이라는 새로운 벤치마크 데이터셋을 제안합니다. 이 데이터셋은 27종의 미세조류 및 미세 해양 생물과 140개의 비디오로 구성되어 있으며, 복잡한 수중 환경을 시뮬레이션할 수 있는 14가지 배경을 포함하고 있습니다.

- **Technical Details**: MPT 데이터셋을 바탕으로, 우리는 미세조류의 실시간 추적을 위한 다중 객체 추적 프레임워크인 Deviation-Corrected Multi-Scale Feature Fusion Tracker(DSFT)를 개발했습니다. DSFT는 두 가지 주요 문제를 해결하기 위해 1) Deviation Correction Method (DCM)을 도입하여 개별 목표에 대한 주의를 보장하고, 2) Multi-scale Feature Similarity Fusion (MFSF)을 제안하여 다양한 크기의 객체에 대한 유사성을 계산합니다.

- **Performance Highlights**: MPT 데이터셋과 DSFT의 유효성 및 우수성은 광범위한 실험을 통해 입증되었습니다. 이 연구는 미세조류 모니터링을 위한 효과적인 해결책을 제공하면서, 기존 다중 객체 추적 방법의 한계를 극복하는 데 기여합니다.



### TopoDiffusionNet: A Topology-aware Diffusion Mod (https://arxiv.org/abs/2410.16646)
Comments:
          20 pages, 11 figures, 7 tables

- **What's New**: 이번 연구에서는 TopoDiffusionNet (TDN)이라는 혁신적인 접근 방식을 제안하여, 확산 모델이 원하는 토폴로지를 유지하도록 하는 방법을 제시합니다. 이는 이미지 내의 토폴로지 구조를 추출하기 위해 지속적 형태학(persistent homology) 도구를 활용하며, 이를 통해 생성 이미지의 정확한 제어를 가능하게 합니다.

- **Technical Details**: TDN은 토폴로지 기반의 목표 함수를 설계하여 노이즈를 억제하고 의도한 구조를 유지하는 성능을 제공합니다. 연구진은 두 가지 차원(0-차원 및 1-차원)에서 이미지 생성 시 요구되는 토폴로지 제약 조건을 만족하면서 노이즈에 강한 형태학적 도구를 사용하여 이미지를 구분합니다.

- **Performance Highlights**: TDN을 통해 4개 데이터셋에서 실험을 진행한 결과, 생성된 이미지의 토폴로지 정확성이 크게 향상된 것을 확인했습니다. 이 연구는 확산 모델과 토폴로지를 통합한 첫 번째 시도로, 생성 모델의 정밀한 제어가 가능함을 보여줍니다.



### Fire and Smoke Detection with Burning Intensity Representation (https://arxiv.org/abs/2410.16642)
- **What's New**: 이 논문에서는 화재 및 연기 감지(Fire and Smoke Detection, FSD)를 위한 새로운 모델인 Attentive Fire and Smoke Detection Model (a-FSDM)을 제안합니다. 이 모델은 기존의 FSD 방법에서 발생하는 투명한 화재와 연기의 감지 문제를 해결하기 위해 특별히 설계된 Attentive Transparency Detection Head (ATDH)를 포함합니다.

- **Technical Details**: a-FSDM은 기존의 물체 감지 알고리즘의 강력한 특징 추출 및 융합 능력을 유지하면서도, 투명한 화재와 연기를 감지하기 위해 감지 헤드를 재설계합니다. 또한, Burning Intensity (BI)라는 새로운 지표를 도입하여 화재의 심각성을 측정하고 화재 관련 위험 평가에 유용한 정보를 제공합니다. ATDH는 다양한 크기와 채널에서 투명성 특징을 강화하여 비 목표 특징 맵을 억제하는 역할을 합니다.

- **Performance Highlights**: 제안된 FSD 모델에 대한 여러 FSD 데이터셋에서의 광범위한 실험 결과는 a-FSDM의 효과성과 다재다능성을 입증합니다. 기존의 방법들에 비해 감지 성능이 향상되어, 화재 및 연기 감지 정확도가 크게 개선되었습니다.



### Benchmarking Multi-Scene Fire and Smoke Detection (https://arxiv.org/abs/2410.16631)
- **What's New**: 이번 연구는 기존의 Fire and Smoke Detection (FSD) 데이터셋의 비표준화 문제를 해결하기 위해 광범위한 자원을 체계적으로 수집하여 새롭고 복합적인 FSD 벤치마크를 구축하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 우리는 12,586장의 이미지를 포함하는 Multi-Scene Fire and Smoke Detection Benchmark (MS-FSDB)를 제안합니다. 이는 2,731개의 장면을 포괄하며, 이미지 해상도가 600픽셀 이상인 긍정 예제와 부정 예제를 포함하고 있습니다. 각 이미지에는 구분된 레이블이 부여되어 분류 및 회귀 작업을 지원합니다.

- **Performance Highlights**: 제안된 MS-FSDB는 복잡한 장면과 다양한 시점의 이미지를 포함하고 있어 모델의 일반화 성능을 상당히 향상시키며, 이전 FSD 데이터셋보다 더 많은 장면을 아우르는 포괄적인 자료를 제공합니다.



### EVC-MF: End-to-end Video Captioning Network with Multi-scale Features (https://arxiv.org/abs/2410.16624)
- **What's New**: 본 논문에서는 비디오 캡셔닝을 위한 새로운 엔드투엔드 기반의 인코더-디코더 네트워크인 EVC-MF를 제안합니다. 이 새로운 접근법은 다중 스케일의 시각적(visual) 및 텍스트적(textual) 특징을 효율적으로 활용하여 비디오 설명을 생성합니다.

- **Technical Details**: EVC-MF는 세 가지 모듈로 구성됩니다: 1) 비디오 프레임을 직접 입력받아 다중 스케일 시각적 특징을 얻는 transformer 기반 네트워크; 2) 다중 스케일 특징을 중복성을 줄이며 유용한 특징을 학습하기 위해 fusion하는 masked encoder; 3) 개선된 transformer 기반 디코더로, 이를 통해 얕은(shallow) 텍스트 정보를 효율적으로 활용합니다.

- **Performance Highlights**: EVC-MF는 벤치마크 데이터셋에 대한 광범위한 실험을 통해 기존의 최첨단 방법들과 비교했을 때 경쟁력 있는 성능을 보였습니다.



### Foundation Models for Remote Sensing and Earth Observation: A Survey (https://arxiv.org/abs/2410.16602)
- **What's New**: 이 논문은 원거리 감지 모델(RSFMs)의 최근 발전을 포괄적으로 검토하는 최초의 조사 연구입니다. 그동안 RSFM의 동기와 배경을 정리하고, 기존 연구를 Visual Foundation Models (VFMs), Vision-Language Models (VLMs), Large Language Models (LLMs) 등으로 구분하여 분석합니다.

- **Technical Details**: 논문은 RS 데이터의 복잡성과 고유한 특징 패턴, 다양한 센서 모달리티 및 시간적 동적성을 다룹니다. 특히, RSFM 개발에서의 주요 도전 과제는 자연 데이터와 RS 데이터 간의 도메인 불일치, 대량 데이터 부족, RSFM에 맞는 적합한 딥 아키텍처의 부재 등을 포함합니다.

- **Performance Highlights**: 모델들은 공개 데이터셋을 기준으로 벤치마킹되었으며, 다양한 센서 모달리티에서의 성능을 비교해 여러 연구 과제를 확인하고 미래 연구 방향을 제시합니다. 이 논문은 RSFMs의 다양한 분야에서의 잠재적인 응용 가능성을 강조하고, 기후 변화, 자연재해 모니터링 등 사회적 이슈 해결에 기여할 것으로 기대됩니다.



### PlaneSAM: Multimodal Plane Instance Segmentation Using the Segment Anything Mod (https://arxiv.org/abs/2410.16545)
Comments:
          submitted to Information Fusion

- **What's New**: 이번 연구는 RGB-D 데이터를 통한 plane instance segmentation의 새로운 방법론인 PlaneSAM을 제안합니다. 기존 방법들이 RGB 밴드만 활용하는 것과 달리, PlaneSAM은 RGB 및 D 밴드의 정보를 통합하여 plane instance segmentation의 정확도를 향상시킵니다.

- **Technical Details**: PlaneSAM은 듀얼 복잡도 백본(dul-complexity backbone)을 사용하여 RGB와 D 밴드의 특징을 모두 효과적으로 학습합니다. D 밴드는 단순한 분기가 학습하고, RGB 밴드는 복잡한 분기가 학습하는 방식으로 설계되어 있습니다. 또한, 대규모 RGB-D 데이터를 기반으로 한 self-supervised pretraining 전략으로 모델의 적응성을 강화합니다.

- **Performance Highlights**: PlaneSAM은 ScanNet 데이터셋에서 새로운 SOTA (State Of The Art) 성능을 기록하였으며, 2D-3D-S, Matterport3D, ICL-NUIM RGB-D 데이터셋에서도 이전 SOTA 접근법을 초월하는 성능을 보였습니다. 계산 오버헤드는 EfficientSAM에 비해 10% 증가하는 수준에 불과합니다.



### Gradient-Free Supervised Learning using Spike-Timing-Dependent Plasticity for Image Recognition (https://arxiv.org/abs/2410.16524)
- **What's New**: 제안된 논문은 이미지 인식을 위해 스파이킹 신경망(Spiking Neural Networks, SNN)에서 기울기 없는(Gradient-free) 학습 방법을 STDP(Spike Timing-Dependent Plasticity)와 결합하여 감독 학습(Supervised Learning)을 수행하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 제안된 네트워크 아키텍처는 여러 층으로 확장 가능하며, MNIST 데이터셋을 사용하여 효율적이고 정확한 학습 성능을 보여줍니다. 각 은닉층 뉴런은 특정 숫자 클래스에 대응하는 고유한 그룹에 할당되어, 타겟 클래스와 일치할 경우에만 시냅스 가중치가 업데이트됩니다. 군집 훈련 단계에서는 억제 시냅스 연결이 없어 뉴런들이 입력 공간을 자유롭게 탐색할 수 있으며, 검증과 테스트 중에만 측면 억제(Lateral Inhibition)가 적용됩니다.

- **Performance Highlights**: MNIST 데이터셋에서 실험 결과, 제안된 방법이 높은 분류 정확도를 달성함을 확인하였고, Brian2 시뮬레이터를 사용하여 SNN 시뮬레이션을 수행했습니다. 이 방법은 백프로파게이션(Backpropagation) 기반의 방법에 비해 보다 생물학적으로 그럴듯한 대안을 제공합니다.



### TIPS: Text-Image Pretraining with Spatial Awareness (https://arxiv.org/abs/2410.16512)
- **What's New**: 이 논문에서는 이미지-텍스트 표현 학습과 자기 지도 학습(self-supervised learning) 간의 간극을 메우기 위해 새로운 일반 목적의 이미지-텍스트 모델인 TIPS(Text-Image Pretraining with Spatial awareness)를 제안합니다. 이 모델은 이미지만으로도 효과적으로 사용할 수 있도록 설계되었습니다.

- **Technical Details**: TIPS는 두 가지 주요 아이디어를 활용합니다: 첫째, 웹에서 수집된 소음 있는 이미지 캡션(noisy captions)을 자동으로 생성한 텍스트 설명(synthetic descriptions)으로 대체하여 공간 인식(spatial awareness) 표현을 학습합니다. 둘째, 대조적 이미지-텍스트 학습(contrastive image-text learning)과 자기 지도 마스킹 이미지 모델링(masked image modeling, MIM)을 결합하여 이미지 특징들이 공간적으로 일관되도록 유도합니다. 이 모델은 Transformer 아키텍처를 기반으로 하며, 대규모 공개 이미지 세트에서 학습됩니다.

- **Performance Highlights**: TIPS는 16개의 데이터세트를 포함한 8개의 다양한 작업에서 강력한 성능을 발휘하며, 이미지-텍스트 및 이미지 전용 작업에 대해서도 일관된 성능을 보입니다. 이 연구는 차세대 이미지 표현 개발을 촉진할 것으로 기대됩니다.



### SINGAPO: Single Image Controlled Generation of Articulated Parts in Objec (https://arxiv.org/abs/2410.16499)
Comments:
          Project page: this https URL

- **What's New**: 이 연구에서는 하나의 이미지에서 가정용 관절 물체(articulated object)의 3D 자산을 생성하는 새로운 방법을 제시합니다. 기존의 연구들은 다중 뷰(multi-view) 또는 다중 상태(multi-state) 입력을 요구하거나 생성 과정에 대한 조정이 제한적이었으나, 본 방법은 단일 이미지에서 시각적으로 일관된 관절 물체를 생성하는 것을 목표로 합니다.

- **Technical Details**: 본 연구에서는 물체의 모양과 동작의 모호성을 포착하기 위해 diffusion model을 설계했습니다. 이 방법은 고수준 구조를 통해 구조화된 데이터의 속성을 생성하는 파이프라인을 갖추고 있으며, 부분 연결 그래프(part connectivity graph)와 부분 추상화(part abstraction)를 이용하여 조리 있게 3D 자산을 생성합니다. 또한, transformer-based diffusion model을 사용하여 이미지 교차 주의(image cross-attention)를 통해 부분의 공간 배치를 학습합니다.

- **Performance Highlights**: 본 방법은 입력 이미지와의 유사성, 생성된 물체의 사실성(realism), 그리고 재구성 품질 측면에서 기존의 선도 기술(state-of-the-art)을 크게 능가하는 성능을 보였습니다. 이를 통해, 단일 이미지에서 관절 물체를 생성하는 데 있어 새로운 가능성을 열어줍니다.



### GenGMM: Generalized Gaussian-Mixture-based Domain Adaptation Model for Semantic Segmentation (https://arxiv.org/abs/2410.16485)
- **What's New**: 이 논문에서는 Generalized Domain Adaptation(GDA)라는 새로운 도메인 적응 설정을 도입하여, 부분적 또는 노이즈가 있는 레이블을 가진 소스 데이터와 약한 레이블 또는 레이블 없는 타겟 데이터를 모두 사용하는 방법을 제안합니다.

- **Technical Details**: 제안하는 GenGMM(Generalized Gaussian-mixture-based) 도메인 적응 모델은 소스 도메인과 타겟 도메인의 데이터 분포를 활용하여 약한 레이블과 유사 레이블을 정제합니다. 이 모델은 GMM(Generative Gaussian Mixture Model)을 기반으로 하며, 두 가지 주요 원칙인 도메인 근접성 가정 및 레이블 있/없 픽셀 간의 유사성을 활용하여 작동합니다. 이 방법은 대조 학습을 위한 신뢰할 수 있는 프로토타입을 생성하는 데 효과적입니다.

- **Performance Highlights**: 실험 결과, GenGMM 모델은 다양한 벤치마크 데이터셋에서 높은 성능을 달성하였으며, 실제 레이블 노이즈를 포함한 소스 데이터로부터 효과적으로 학습할 수 있음을 보여주었습니다.



### HaHeAE: Learning Generalisable Joint Representations of Human Hand and Head Movements in Extended Reality (https://arxiv.org/abs/2410.16430)
- **What's New**: 본 연구에서는 HaHeAE라는 새로운 자가 지도 학습(self-supervised learning) 방법을 소개합니다. 이 방법은 인간의 손과 머리 움직임을 동시에 모델링하여, 다양한 XR(확장 현실) 환경에서 일반화 가능한 표현을 학습할 수 있도록 합니다.

- **Technical Details**: HaHeAE는 그래프 신경망(Graph Convolutional Network) 기반의 의미 인코더(semantic encoder)와 확산 기반의 확률적 인코더(diffusion-based stochastic encoder)를 사용하여 손과 머리 움직임의 공동 의미 및 확률적 표현을 학습합니다. 또한, 확산 기반 디코더를 통해 원래 신호를 재구성합니다. 이 방법은 손-머리 움직임 예측을 보조 작업으로 활용하여 공간-시간적 특징을 향상시키며, 세 가지 XR 데이터셋에서 성능을 평가하였습니다.

- **Performance Highlights**: 여러 데이터셋에서, HaHeAE는 기존의 자가 지도 학습 방법에 비해 재구성 품질이 최대 74.0% 향상되었으며, 사용자의 다양한 활동과 XR 환경에 대해 일반화할 수 있음을 보여주었습니다. 이 방법은 해석 가능한 손-머리 클러스터 식별 및 변동적인 손-머리 움직임 생성과 같은 새로운 응용을 가능하게 하며, 다운스트림 작업의 효과적인 특징 추출기로서의 역할도 수행할 수 있습니다.



### AttentionPainter: An Efficient and Adaptive Stroke Predictor for Scene Painting (https://arxiv.org/abs/2410.16418)
- **What's New**: AttentionPainter는 효율적이고 적응 가능한 단일 단계 neural painting 모델로, 단 한번의 forward 과정에서 많은 수의 strokes를 예측할 수 있도록 설계되었습니다. 기존의 RL 및 auto-regressive 방법과는 달리, AttentionPainter는 예측 속도가 빠르며 안정적인 훈련 과정을 제공합니다.

- **Technical Details**: AttentionPainter는 transformer 기반 모듈을 통해 이미지 특징을 추출하고, Stroke-density Loss를 통해 세부 정보의 재구성을 향상시킵니다. 또한 Fast Stroke Stacking 알고리즘을 도입하여 훈련 속도를 13배 향상시키고, Stroke Diffusion Model을 통해 stroke 기반 생성 및 인페인팅, 편집이 가능해집니다.

- **Performance Highlights**: Extensive experiments에 따르면, AttentionPainter는 최신 neural painting 방법보다 빠르고 더 나은 재구성 결과를 보였으며, 이는 human artists의 작업을 지원하는 데 유용합니다.



### Joker: Conditional 3D Head Synthesis with Extreme Facial Expressions (https://arxiv.org/abs/2410.16395)
Comments:
          Project Page: this https URL

- **What's New**: Joker라는 새로운 방법을 소개합니다. 이 방법은 단일 참조 이미지로부터 3D 인간 머리를 극단적인 표정으로 합성할 수 있습니다.

- **Technical Details**: 이 방법은 3D Morphable Model (3DMM)과 텍스트 입력을 통해 표정을 제어합니다. 3DMM만으로는 미세한 감정 변화와 극단적인 표정을 정의하기 어려우므로, 다중 모달 조건 신호가 필수적입니다. 또한, 우리는 2D diffusion-based prior를 기반으로 하는 방법을 제안하며, 이는 조각, 짙은 화장, 회화 등의 도메인 외 샘플에 대해서도 잘 일반화됩니다. 새로운 3D distillation 기법을 통해 2D prior의 예측을 Neural Radiance Field (NeRF)로 변환하여 시각 일관성을 개선합니다.

- **Performance Highlights**: 2D prior와 우리의 distillation 기법 모두 최첨단 결과를 생성하며, 이는 광범위한 평가를 통해 확인되었습니다. 우리 방법은 극단적인 혀 움직임에서 시각 일관성을 달성한 최초의 방법으로 알려져 있습니다.



### Disambiguating Monocular Reconstruction of 3D Clothed Human with Spatial-Temporal Transformer (https://arxiv.org/abs/2410.16337)
- **What's New**: 이 논문에서는 Spatial-Temporal Transformer (STT) 네트워크를 소개하여 3D 의상 인간 모델 복원에서의 두 가지 주요 문제인 후면 세부 묘사 모호성과 이미지의 지역적 모호성을 해결합니다.

- **Technical Details**: STT 네트워크는 공간 변환기와 시간 변환기를 결합하여 이미지의 글로벌 정보와 시간적 특징을 동시에 활용합니다. 공간 변환기는 후면 노말 맵(predicted normal map)을 예측하는 데 필요한 글로벌 정보를 추출하고, 시간 변환기는 인접 프레임으로부터 시간적 특징을 추출하여 지역적인 모호성을 보완합니다.

- **Performance Highlights**: Adobe 및 MonoPerfCap 데이터셋에서의 실험 결과, 제안한 방법이 기존 최첨단 방법들보다 우수한 성능을 보이며 저조도 환경에서도 강력한 일반화 능력을 유지함을 보여주었습니다.



### The Solution for Single Object Tracking Task of Perception Test Challenge 2024 (https://arxiv.org/abs/2410.16329)
- **What's New**: 이번 보고서는 동영상 시퀀스에서 특정 객체를 추적하는 Single Object Tracking (SOT) 방법을 소개합니다. LoRAT 방법을 활용하여, 모델의 일부 파라미터를 미세 조정하는 LoRA 기법을 시각적 추적 분야에 적용합니다. LaSOT와 GOT-10k 데이터세트를 사용해 모델을 훈련했으며, 경량 포스트 프로세싱 기법인 alpha-refine를 적용하였지만 기대만큼의 성과를 내지 못했습니다.

- **Technical Details**: LoRAT 접근법은 입력 임베딩 조정, MLP 전용 헤드 네트워크 통합 등의 구성 요소로 이루어져 있습니다. 토큰 타입 임베딩을 도입하여 템플릿과 검색 영역 토큰을 구분합니다. 이미지 쌍의 다양한 크기를 다룰 수 있는 포지셔널 임베딩 방식도 고안하여 파라미터 효율성을 높입니다. 분류와 경계 상자 회귀를 위한 두 개의 MLP 브랜치를 포함하는 MLP 전용 헤드를 사용해 성능을 최적화합니다.

- **Performance Highlights**: 알고리즘의 전반적인 접근은 0.813 점수를 달성하며 경쟁에서 1위로 등극했습니다. 다양한 훈련 데이터와 고급 추적 기법을 결합하여 성능이 크게 개선되었습니다.



### Accelerating Object Detection with YOLOv4 for Real-Time Applications (https://arxiv.org/abs/2410.16320)
Comments:
          18 pages, 10 figures

- **What's New**: 이 논문은 UAV(무인항공기)에서의 실시간 물체 탐지 문제를 다룹니다. 특히, CNN(합성곱 신경망)을 활용한 YOLOv4(You Only Look Once - Version 4) 알고리즘을 개선하여, 복잡한 환경 속에서도 정확한 물체 탐지를 가능하게 하는 새로운 접근법을 제안합니다.

- **Technical Details**: 이 연구에서 제안된 YOLOv4는 DropBlock Regularization, Data Augmentation, Mish-Activation, CrossStage Partial connections(CSP), Self adversarial training(SAT), Weighted-Residual-Connections(WRC) 등의 여러 기술들을 결합하여 구성됩니다. YOLOv4의 성능은 COCO 데이터셋에서 43.5%의 평균 정밀도(AP)와 65 FPS의 속도를 기록하였습니다.

- **Performance Highlights**: YOLOv4는 실시간 물체 탐지에서 빠르고 정확한 성능을 발휘합니다. YOLOv4는 1단계 탐지기로, R-CNN, Fast R-CNN과 같은 2단계 탐지기에 비해 더 빠른 처리 속도와 높은 정확성을 제공합니다.



### Accelerating Biological Spatial Cluster Analysis with the Parallel Integral Image Techniqu (https://arxiv.org/abs/2410.16291)
Comments:
          IEEE CIBCB 2023 Short paper available at this https URL

- **What's New**: 이번 연구는 생물 이미지 분석에 사용할 수 있는 Sliding Window Analysis (SWA)의 계산 복잡도를 줄이는 Parallel Integral Image 접근법을 소개합니다. 기존의 방법보다 큰 이미지에서 사용할 수 있는 가능성을 열었습니다.

- **Technical Details**: 이 논문에서는 고해상도 이미지를 처리하기 위해 Parallel Integral Image 기법을 도입했습니다. 이 방법은 O(r*c)/p + 4(r*c)/p의 계산 복잡도를 가지며, p는 병렬 작업자의 수를 의미합니다. 또한, Integral Image 기법에서는 Rectangular Region에서 픽셀 값을 효율적으로 계산할 수 있습니다.

- **Performance Highlights**: 실험 결과, Parallel Integral Image 접근법이 10,000 x 10,000 이미지에서 131,806배의 속도 향상을 보여주었고, 다양한 대형 미세조직 이미지에서도 10,000배 이상의 일관된 속도 향상을 기록하였습니다.



### Solution for OOD-CV UNICORN Challenge 2024 Object Detection Assistance LLM Counting Ability Improvemen (https://arxiv.org/abs/2410.16287)
- **What's New**: 본 보고서는 ECCV OOD-CV UNICORN Challenge 2024에서 탐색하고 제안한 방법을 자세히 설명합니다. 이 방법은 대형 언어 모델의 응답 강인성에 초점을 맞추고 있습니다. 대회 데이터셋 OODCA-VQA 및 SketchyQA의 변형이 도입되어 모델의 강인성을 테스트합니다.

- **Technical Details**: 우리의 접근 방식은 두 가지 주요 블록으로 구성됩니다: (1) Object Detection Assistance, (2) Counterfactual Specific prompt. InstructBLIP 모델을 기반으로 하고 Co-DETR 객체 탐지 모델을 사용하여 pseudo label을 생성합니다. 이 방법은 모델의 카운팅 능력과 추론 능력을 개선하는 데 중점을 둡니다.

- **Performance Highlights**: 최종 테스트에서 0.86의 점수로 2위를 차지하며, 기존의 대형 모델보다 우수한 성능을 보였습니다. 우리의 Counterfactual Specific prompt 설계는 모델의 카운팅 능력을 향상시키고 Hallucination 문제를 피하는 데 기여했습니다.



### Solution for Point Tracking Task of ECCV 2nd Perception Test Challenge 2024 (https://arxiv.org/abs/2410.16286)
- **What's New**: 이 보고서는 비디오에서 물리적 표면을 모니터링하는 데 중점을 두고 Tracking Any Point (TAP) 방법의 개선된 접근 방식을 소개합니다. 특히 고정된 카메라로 촬영된 비디오에서 정적 포인트의 트래킹을 인식하고 수정하는 데 중점을 둔 Fine-grained Point Discrimination (FPD)라는 새로운 방법을 제안합니다.

- **Technical Details**: 제안된 FPD는 두 가지 주요 구성 요소로 이루어져 있습니다: 1) 다중 세분화 포인트 인식(Multi-granularity Point Perception, MPP): 정적 시퀀스와 포인트를 감지하는 데 사용됩니다. 2) 동적 궤적 수정(Dynamic Trajectory Correction, DTC): 추적된 포인트 유형에 따라 포인트 궤적을 교체합니다. 이러한 방법은 기존 TAP 방법의 단점을 해결하며, 특히 긴 시퀀스에서 발생하는 성능 저하와 리소스 오버헤드를 줄이는 데 기여합니다.

- **Performance Highlights**: FPD는 최종 테스트에서 0.4720의 점수로 2위의 성적을 기록했으며, 실험 결과로 제안된 기법의 우수성을 입증했습니다.



### JMMMU: A Japanese Massive Multi-discipline Multimodal Understanding Benchmark for Culture-aware Evaluation (https://arxiv.org/abs/2410.17250)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 일본 문화 맥락에 기반한 전문 수준의 작업을 평가하기 위해 설계된 첫 번째 대규모 일본어 벤치마크인 JMMMU(Japanese MMMU)를 소개합니다. 이 벤치마크는 두 가지 상호 보완적인 하위 집합으로 구성되어 있습니다: 문화 비독립적(Culture-Agnostic, CA) 하위 집합과 문화 특정(Culture-Specific, CS) 하위 집합.

- **Technical Details**: JMMMU는 1,320개의 질문과 1,118개의 이미지를 포함하고 있으며, 28개의 다양한 과목을 다룹니다. CA 하위 집합에서는 영어로 동일한 내용의 질문과의 직접 비교를 가능하게 하고, CS 하위 집합에서는 일본 문화에 적합한 새로운 질문을 포함합니다. 이 연구는 15개의 오픈 소스 LMM과 3개의 고급 독점 LMM을 평가하여 발견된 주요 성과를 분석합니다.

- **Performance Highlights**: JMMMU에서 평가된 모델의 전반적인 성능은 최대 58.6%로, 일본 문화에 대한 이해가 부족함을 보여줍니다. CA 하위 집합에서는 대부분의 모델이 일본어로 질문했을 때 영어보다 성능이 떨어졌고(CA 하위 집합에서 최대 8.6% 하락), CS 하위 집합에서 일본 데이터셋으로 훈련된 모델이 가장 높은 성능을 보였습니다. 이 연구는 다문화 성격을 고려한 다양한 고표준 벤치마크 개발의 필요성을 강조하고 있습니다.



### Frontiers in Intelligent Colonoscopy (https://arxiv.org/abs/2410.17241)
Comments:
          [work in progress] A comprehensive survey of intelligent colonoscopy in the multimodal era

- **What's New**: 본 연구는 인공지능(AI)의 적용을 통해 대장내시경(colonoscopy) 기술의 최근 발전 사항과 다중 모달(multi-modal) 의료 응용의 잠재적 영향을 탐구합니다. 연구 결과, 대장내시경 장면 인식에 대한 네 가지 주요 작업(Task)을 통해 다양한 데이터 및 모델 중심의 도전 과제를 확인하고, 미개척 분야에 대한 가능성을 제시합니다.

- **Technical Details**: 연구의 주요 기여점은 세 가지 기본 이니셔티브입니다: (1) 대규모 다중 모달 지침 튜닝 데이터셋인 ColonINST를 개발하며, 이는 62개 하위 카테고리의 303,001개의 대장내시경 이미지를 포함하고 있습니다. (2) 대장내시경을 위한 다중 모달 언어 모델 ColonGPT를 구축하여, endoscopists가 상호작용하는 대화를 통해 도움을 받을 수 있도록 설계하였습니다. (3) 다양한 작업을 위한 다중 모달 벤치마크를 제공하여 후속 연구를 위한 기초를 마련했습니다.

- **Performance Highlights**: ColonGPT는 0.4B 파라미터의 비쥬얼 인코더와 1.3B 파라미터의 경량 언어 모델을 사용하여 자원 친화적으로 구현되었으며, 기존 방식보다 향상된 성능으로, 새로 생성된 다중 모달 벤치마크에서 세 가지 작업 모두에서 최고 점수를 기록했습니다. 이 모델은 네 대의 A100-40GB GPU로 5시간 이내에 훈련이 가능하여 후속 연구의 신속한 개발을 촉진합니다.



### Automated Spinal MRI Labelling from Reports Using a Large Language Mod (https://arxiv.org/abs/2410.17235)
Comments:
          Accepted to Medical Image Computing and Computer Assisted Intervention (MICCAI 2024, Spotlight). 11 pages plus appendix

- **What's New**: 이 연구에서는 방사선 보고서에서 라벨(Labels)을 자동으로 추출하기 위한 일반적인 파이프라인을 제안합니다. 이는 대형 언어 모델(LLMs)을 활용하여 요추 MRI 보고서에서 스파인 암(spinal cancer), 협착증(stenosis), 척추 분리증(spondylolisthesis), 좌골 신경 압박(cauda equina compression), 그리고 탈장(herniation) 등 다섯 가지 조건을 검증합니다.

- **Technical Details**: 제안된 방법은 두 가지 단계로 구성됩니다: (1) 모델에게 특정 조건을 염두에 두고 보고서를 요약하도록 요청하고, (2) 요약을 바탕으로 이진 라벨을 생성합니다. 이를 통해 Zephyr(7B)와 Llama3 Instruct(8B)라는 오픈소스 대형 언어 모델을 사용하여, GPT-4의 성능을 초과하는 결과를 달성하였습니다.

- **Performance Highlights**: 자동으로 생성된 라벨을 사용하여 훈련한 비전 모델은 임상의가 직접 주석을 달아 훈련한 모델과 유사한 수준의 성능을 나타냈습니다. 이는 의료 이미징 문제 해결을 위한 훈련 데이터셋을 대폭 증가시킬 수 있는 가능성을 보여줍니다.



### LiNeS: Post-training Layer Scaling Prevents Forgetting and Enhances Model Merging (https://arxiv.org/abs/2410.17146)
Comments:
          The first two authors contributed equally to this work; Project website: \url{this https URL}

- **What's New**: 이 논문에서는 LiNeS(Layer-increasing Network Scaling)라는 새로운 후처리 편집 기법을 제안하여, 사전 훈련된 모델의 일반화를 유지하면서 미세 조정(또는 fine-tuning)된 작업 성능을 향상시키는 방법을 다룹니다. LiNeS는 네트워크 내의 층 깊이에 따라 매개변수 업데이트를 선형으로 확장하여 얕은 층은 사전 훈련된 값에 가까이 유지하고, 깊은 층은 작업 특정 표현을 보존합니다.

- **Technical Details**: LiNeS는 모델 병합 시 층 별로 매개변수를 조정하여 부정적인 작업 간 간섭을 줄이는 방법으로 확장될 수 있습니다. 본 방법은 기존의 미세 조정 과정과 결합하여, 모델이 더 높은 정확도를 달성하도록 지원합니다. 제안된 알고리즘은 사용자가 쉽게 구현할 수 있으며, 기존 많은 기술들과 상호 보완적입니다.

- **Performance Highlights**: LiNeS는 단일 작업(single-task) 및 다중 작업(multi-task) 환경에서 다양한 벤치마크에서 유의미한 성능 향상을 입증했습니다. 본 방법은 잊어버림(forgetting)을 완화하고, 분포 외 일반화(out-of-distribution generalization)를 강화하며, 기존의 모델 병합 기법과 원활하게 통합되어 성능을 향상시키고, 다양한 보상을 정렬한 LLM 정책(large language model) 병합 시에도 일반화를 증대시키는데 기여합니다.



### LFME: A Simple Framework for Learning from Multiple Experts in Domain Generalization (https://arxiv.org/abs/2410.17020)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 이 논문은 다중 전문가로부터 학습하는 새로운 프레임워크인 LFME(learning from multiple experts)를 제안합니다. 이 프레임워크는 다양한 출처 도메인에서 훈련 데이터를 사용하여 목표 모델을 개선하고, 각 도메인에 특화된 여러 전문가 모델을 훈련하여 성능을 향상시킵니다.

- **Technical Details**: LFME는 학습 단계에서 전문가 모델의 확률을 규제하기 위해 로그 확률 조정(logit regularization)이라는 방법을 사용합니다. 이를 통해 목표 모델은 전문가의 지식을 전반적으로 수렴하여 모든 출처 도메인에서 전문성을 가지도록 합니다. 기존의 KD(knowledge distillation)와 유사하나, LFME는 모든 출처 도메인에 대한 전문가 지식을 통합하여 최종 예측을 수행하게 됩니다.

- **Performance Highlights**: LFME는 다양한 domain generalization(도메인 일반화) 벤치마크에서 평균 성능이 개선되었음을 보였으며, 기존의 모형들과 비교하여 뛰어난 결과를 달성했습니다. 또한 LFME는 오직 하나의 하이퍼파라미터만으로도 우수한 성능을 발휘하여, 간단한 구현이 가능하다는 이점을 가지고 있습니다.



### Multi-Layer Gaussian Splatting for Immersive Anatomy Visualization (https://arxiv.org/abs/2410.16978)
- **What's New**: 본 논문에서는 CT 스캔과 같은 의료 볼륨 데이터의 효율적인 시각화를 위해 새로운 GS(Gaussian Splatting) 접근 방식을 제안합니다. 이 방법은 정적이지만 중간적인 표현을 통해 해부학적 구조를 상세히 시각화하며, 모바일 VR 기기에서도 인터랙티브한 프레임 속도를 유지할 수 있습니다.

- **Technical Details**: 이 연구는 선형 GS 표현을 도입하여 여러 해부학적 구조를 점진적으로 포함하고, 비활성 Gaussian을 제거하기 위한 GS 훈련을 확장하였습니다. 또한, 층 위에서 클러스터링을 통해 생성된 모델을 압축하여 최종적으로 고품질 시각화를 타겟 하드웨어에 맞춰 조정할 수 있습니다. 이 연구에서 제안된 접근 방식은 경량화된 GS 모델을 기반으로 하며, 고해상도 경로 추적 이미지와 상호작용합니다.

- **Performance Highlights**: 제안된 방식은 애초에 구현된 GS에 비해 의료 볼륨을 동적으로 렌더링할 수 있는 가능성을 제공하며, 해부학적 구조의 상세함을 잘 보존한 채로 인터랙티브한 프레임 속도를 달성했습니다. 이는 높은 연산 요구 조건으로 인해 기존의 경로 추적 의료 볼륨 사용에 제한을 주었던 문제를 해결하는 데 기여할 수 있습니다.



### IdenBAT: Disentangled Representation Learning for Identity-Preserved Brain Age Transformation (https://arxiv.org/abs/2410.16945)
Comments:
          16 pages, 8 figures, 2 tables

- **What's New**: 본 연구에서는 개인 특성을 보존하면서 특정 연령대의 뇌 이미지를 생성하는 IdenBAT라는 새로운 구조를 제안합니다. 이를 통해 나이에 따른 관련 속성을 변환하는 동시에, 개인의 고유한 특성을 유지할 수 있습니다.

- **Technical Details**: IdenBAT는 conditional generative adversarial network (cGAN) 아키텍처를 기반으로 하며, encoder, identity extracting module (IEM), age injecting module (AIM), generator (G), discriminator (D)로 구성됩니다. IEM을 통해 이미지 특성이 나이에 관련된 특성과 무관한 특성으로 분리됩니다. AIM을 통해 나이에 대한 정보를 개인의 특성에 주입하여 최종적으로 개인의 정체성을 유지하면서도 목표 연령을 반영하는 이미지를 생성합니다.

- **Performance Highlights**: IdenBAT는 2D 및 3D 뇌 데이터셋에서 실험을 통해 기존의 최첨단 방법들과 비교하여 성능 우수성을 입증했습니다. 특히, 개인의 특성을 보존하며 정확한 연령 변환이 가능하다는 점에서 뛰어난 결과를 도출했습니다.



### MBD: Multi b-value Denoising of Diffusion Magnetic Resonance Images (https://arxiv.org/abs/2410.16898)
Comments:
          this is a biomedical engineering work using machine learning to enhance medical images

- **What's New**: 본 논문에서는 여러 b-value에서 수집된 데이터를 활용하여 새로운 방식으로 diffusion magnetic resonance images (dMRI)를 denoising하는 방법을 제시합니다. 기존의 많은 방법들이 여러 회전 방향에서의 데이터 중복성을 활용하는 데 반해, MBD (multi-b-value-based denoising)는 단 몇 개의 이미지로도 고해상도의 잡음을 줄이는 새로운 접근 방식을 사용합니다.

- **Technical Details**: MBD는 같은 diffusion 인코딩 방향에서 서로 다른 b-value를 가진 diffusion-weighted images (DWI) 간의 유사성을 기반으로 합니다. 이 방법은 고잡음 분산을 효과적으로 제거하며 블러링(burring)을 방지하면서도, 적은 수의 입력 이미지만 사용합니다. CNN(Convolutional Neural Network)을 사용하여 training하며, Noise2Noise 접근 방식을 통해 noiseless training targets 없이도 학습을 진행합니다.

- **Performance Highlights**: MBD의 성능을 평가하기 위한 실험에서, 기존의 Noise2Noise 방식, CNN 기반의 extrapolation, 그리고 MPPCA와 비교하여 우수한 결과를 나타냈습니다. 여러 b-value를 입력으로 사용하는 방식이 단일 입력보다 효과적임을 밝혔습니다.



### Rethinking generalization of classifiers in separable classes scenarios and over-parameterized regimes (https://arxiv.org/abs/2410.16868)
- **What's New**: 이 논문은 클래스가 분리 가능하거나 분류기가 과적합된 경우의 학습 동역학을 조사합니다. Empirical Risk Minimization (ERM) 방법은 두 경우 모두에서 훈련 오류를 완전히 제거하지만, 제로 훈련 오류를 가지는 여러 글로벌 미니마가 존재하며, 그 중 일부는 좋은 일반화 성능을 보이고 그렇지 않은 것도 있습니다.

- **Technical Details**: 분리 가능한 클래스 상황에서는 n개의 훈련 데이터 수에 따라 나쁜 글로벌 미니마의 비율이 지수적으로 감소하는 것을 증명합니다. 이 분석은 주어진 분류기 함수 집합에 대한 진짜 오류의 밀도 분포에만 의존하는 경계와 학습 곡선을 제공합니다. 모델을 통해 진짜 오류의 밀도 분포를 제안하며, 이는 MNIST와 CIFAR-10에서의 실험 결과와 일치하는 학습 곡선을 생성합니다.

- **Performance Highlights**: 이 연구를 통해 과적합된 Neural Networks가 예기치 않게 좋은 일반화 성능을 달성할 수 있는 이유를 설명하며, 특히 훈련 샘플보다 훨씬 더 많은 매개변수를 가지는 신경망에서 제로 훈련 오류와 일반화 성능 간의 관계를 심도 있게 다룹니다.



### Nash Meets Wertheimer: Using Good Continuation in Jigsaw Puzzles (https://arxiv.org/abs/2410.16857)
Comments:
          to be published in ACCV2024

- **What's New**: 이 논문은 전통적인 색상 및 형태 정보 대신 선형 기하학적 패턴에 의존하는 새로운 방식으로 직소 퍼즐을 푸는 문제를 제시합니다. 기존의 방법들은 물리적 손상이 심한 조각들에 적용할 수 없었던 반면, 이 연구는 Gestalt의 '좋은 연속성 법칙'에 기반하여 조각들을 재구성합니다.

- **Technical Details**: 논문에서는 직소 퍼즐 문제를 비협력적 다인 게임의 나쉬 균형(Nash equilibrium) 문제로 모델링하며, 고전적인 다인 복제기(dynamic)의 방법을 사용합니다. 이는 다양한 형상 및 크기의 조각에 대해 적용할 수 있습니다.

- **Performance Highlights**: 본 연구는 합성 및 실제 데이터를 통해 평가되었으며, 실험 결과는 기존의 방법들이 부족함을 보여주고, 제안된 게임 이론적 접근 방식이 상대적으로 효율적임을 입증합니다.



### NucleiMix: Realistic Data Augmentation for Nuclei Instance Segmentation (https://arxiv.org/abs/2410.16671)
- **What's New**: 이 연구에서는 NucleiMix라는 데이터 증강 방법을 도입하여 병리학 이미지 분석에서 핵(instance) 분할(nuclei instance segmentation) 문제를 해결하고자 하였습니다. 이 방법은 데이터 불균형 문제를 해결하기 위해 드물게 등장하는 핵의 개수를 증가시키는 것을 목표로 합니다.

- **Technical Details**: NucleiMix는 두 단계로 작업을 수행합니다. 첫 번째 단계에서는 드물게 유형의 핵 주변과 유사한 후보 위치를 식별하고, 이 위치에 드물게 유형의 핵을 삽입합니다. 두 번째 단계에서는 사전 훈련된 diffusion 모델을 사용하여 점진적인 인페인팅(progessive inpainting) 전략을 적용하여 주요 유형의 핵이나 배경 위치를 대체하며 드물게 유형의 핵을 새로운 환경에 매끄럽게 통합합니다.

- **Performance Highlights**: 세 가지 공개 데이터셋을 사용하여 NucleiMix의 효과를 체계적으로 평가한 결과, NucleiMix가 현실적인 드물게 유형의 핵을 합성(synthesize)하고, 핵 분할(segmentation) 및 분류(classification) 품질을 정확하고 견고한 방식으로 향상시키는 우수한 성능을 보여주었습니다.



### Visual Question Answering in Ophthalmology: A Progressive and Practical Perspectiv (https://arxiv.org/abs/2410.16662)
- **What's New**: 이 리뷰 논문은 안과학에서 시각 질문 응답(Visual Question Answering, VQA)의 최근 발전과 미래 전망을 다루고 있습니다. 특히, 컴퓨터 비전과 자연어 처리(Natural Language Processing, NLP)를 융합하여 안과 이미지를 이해하고 질의에 응답하는 방법을 제시합니다.

- **Technical Details**: VQA는 동시에 여러 종류의 정보를 처리할 수 있는 멀티모달(Multimodal) 접근 방식을 통해 안과 이미지를 해석하는 데 도움을 줍니다. 이 과정에서 대형 언어 모델(Large Language Models, LLM)이 VQA 프레임워크의 다양한 구성 요소를 향상시키는 역할을 수행합니다. 그러나 주석이 달린 멀티모달 이미지 데이터셋의 부족, 포괄적이고 통일된 평가 방법의 필요성, 현실 세계에서의 적용 효과성에 대한 장벽 등이 주요 도전 과제로 남아 있습니다.

- **Performance Highlights**: 안과 VQA 시스템은 의료 전문가와 AI 전문가 간의 협력을 통해 기존의 장애물을 극복하고 안과 질병의 진단 및 치료를 향상시킬 수 있는 가능성을 가지고 있습니다. 앞으로 VQA 기술과 대형 언어 모델의 발전은 안과 분야에서의 진단 정확도를 크게 향상시킬 것으로 기대됩니다.



### Dual-Model Defense: Safeguarding Diffusion Models from Membership Inference Attacks through Disjoint Data Splitting (https://arxiv.org/abs/2410.16657)
- **What's New**: 이 논문에서는 Membership Inference Attacks (MIAs)로부터 확산 모델(diffusion models)을 보호하기 위해 DualMD와 DistillMD라는 두 가지 새로운 접근 방식을 소개합니다. 이 방법들은 원본 데이터셋의 서로 다른 부분 집합에 대해 별도의 확산 모델을 훈련시킴으로써 MIAs의 위험을 크게 줄이는 데 도움이 됩니다.

- **Technical Details**: DualMD는 두 개의 모델을 사용하여 개인 정보 추론을 방지하는 프라이빗 인퍼런스 파이프라인을 사용합니다. DistillMD는 'soft targets'를 생성하여 비공식 모델(student model)을 훈련시키고 모든 종류의 MIAs에 대한 프라이버시 보장을 강화합니다. 이 두 방법은 대규모 데이터셋에서 다양한 MIAs에 대한 평가를 통해 효과적임을 입증했습니다.

- **Performance Highlights**: DualMD와 DistillMD는 MIAs의 성공률을 실질적으로 줄이면서 경쟁력 있는 이미지 생성 성능을 유지하는 것으로 나타났습니다. 특히 DistillMD는 MIAs뿐 아니라 모델의 암기(memorization)를 줄이는 데에도 기여하여, 두 가지 취약점이 오버피팅(overfitting)에서 발생함을 보여주었습니다.



### Large Body Language Models (https://arxiv.org/abs/2410.16533)
- **What's New**: 본 논문에서는 실시간으로 사람처럼 제스처를 생성하는 데 최적화된 새로운 구조인 Large Body Language Models (LBLMs)를 도입합니다. 특히, LBLM-AVA라는 아키텍처는 Transformer-XL과 병렬화된 diffusion 모델을 결합하여 텍스트, 오디오 및 비디오를 포함하는 다중 모달 입력으로부터 인간처럼 보이는 제스처를 생성합니다.

- **Technical Details**: LBLM-AVA는 제스처 생성 기능을 향상시키기 위한 여러 핵심 구성 요소를 통합합니다: 다중 모달-포즈 임베딩(multimodal-to-pose embeddings), 재정의된 주의 메커니즘(attention mechanisms)을 통해 개선된 시퀀스-투-시퀀스 매핑(sequence-to-sequence mapping), 제스처 시퀀스 일관성을 위한 시간적 스무딩 모듈(temporal smoothing module), 그리고 현실감을 높이기 위한 주의 기반 정제 모듈(attention-based refinement module)입니다.

- **Performance Highlights**: LBLM-AVA는 Fréchet Gesture Distance (FGD)를 30% 줄여주며, Fréchet Inception Distance에서 25% 향상을 이뤄내어 기존 접근 방식에 비해 현실적이고 상황에 적합한 제스처 생성에서 최첨단 성능을 달성합니다.



### Efficient Neural Network Training via Subset Pretraining (https://arxiv.org/abs/2410.16523)
Comments:
          To appear in KDIR 2024

- **What's New**: 본 논문에서는 신경망 훈련에 있어 작은 배치를 사용한 부분적인 기울기를 계산하는 전통적인 접근 방식의 한계와 그에 대한 대안을 제시합니다. 특히, 훈련 세트의 부분 집합으로부터 계산된 최소 손실을 전체 훈련 세트의 최소 손실로 잘 근사할 수 있다는 새로운 가설을 제안합니다.

- **Technical Details**: 논문에서는 MNIST, CIFAR-10 및 CIFAR-100 데이터셋을 사용한 실험을 통해, 작은 부분 집합도 충분한 대표성을 가진다는 것을 증명하였으며, 이러한 접근 방식은 기존의 훈련 결과와 유사한 성과를 달성할 수 있음을 보여줍니다. 이 방식은 최적화 준비 시간을 줄여주며, 컴퓨팅 비용을 10분의 1로 감소시킬 수 있습니다.

- **Performance Highlights**: 실험 결과는 기존의 Stochastic Gradient Descent (SGD) 방식이 아닌 새로운 부분 집합 최소화 기법이 잘 작동하며, 작은 배치들이 매우 유효한 경우 



### Allo-AVA: A Large-Scale Multimodal Conversational AI Dataset for Allocentric Avatar Gesture Animation (https://arxiv.org/abs/2410.16503)
- **What's New**: Allo-AVA 데이터 세트는 대화형 AI의 생생한 아바타 애니메이션 생성을 위한 고품질 다중 모드 훈련 데이터를 제공합니다. 이 데이터 세트는 텍스트와 오디오 기반 아바타 제스처 애니메이션을 위해 특별히 설계되었습니다.

- **Technical Details**: Allo-AVA 데이터 세트는 약 1250시간의 다양한 비디오 콘텐츠를 포함하며, 오디오, 전사, 추출된 키포인트와 함께 제공됩니다. 이 데이터는 발화 내용, 음향 특성, 시각적 신호 및 대화 맥락 간의 관계를 포착하는 데 유용합니다.

- **Performance Highlights**: 키포인트 수가 1350억 개 이상이며, 평균 1분당 112,500개의 키포인트가 추출됩니다. 이 데이터 세트의 다양성은 다양한 연령, 성별 및 민족적 배경을 포함하여 아바타 애니메이션의 자연성을 높이는 데 기여합니다.



### AlignVSR: Audio-Visual Cross-Modal Alignment for Visual Speech Recognition (https://arxiv.org/abs/2410.16438)
- **What's New**: 이번 논문에서는 오디오-비디오 교차 모달 정렬(audio-visual cross-modal alignment) 기반의 새로운 시각 음성 인식(Visual Speech Recognition, VSR) 방법인 AlignVSR을 제안합니다. 이 방법은 오디오 모달리티를 보조 정보로 활용하여 시각 정보를 텍스트로 인식하는 효율성을 높입니다.

- **Technical Details**: AlignVSR 방법은 두 가지 정렬 메커니즘을 사용합니다. 첫 번째는 비디오 프레임과 오디오 유닛 사이의 전역 정렬(global alignment)이며, 이를 위해 cross-modal attention 메커니즘을 적용합니다. 두 번째는 오디오와 비디오의 시간적 대응 관계에 기초한 로컬 정렬(local alignment) 손실을 추가하여 전역 정렬을 세밀하게 조정합니다.

- **Performance Highlights**: LRS2 및 CNVSRC.Single 데이터 세트에서 실시한 실험 결과, AlignVSR은 기존의 VSR 방법들보다 뛰어난 성능을 보이며, 특히 AKVSR 벤치마크를 지속적으로 초과하는 성과를 입증하였습니다.



### Domain-Adaptive Neural Posterior Estimation for Strong Gravitational Lens Analysis (https://arxiv.org/abs/2410.16347)
Comments:
          20 pages, 2 figures, 2 tables

- **What's New**: 본 연구는 Neural Posterior Estimation (NPE)와 Unsupervised Domain Adaptation (UDA)를 결합하여 강중력렌즈 데이터의 분석 효율성을 높이는 첫 번째 연구를 수행하였습니다.

- **Technical Details**: NPE는 CNN 기반의 임베딩 네트워크를 활용하여 이미지를 특징으로 요약하고, Masked Autoregressive Flow (MAF)를 통해 비가우시안 분포를 포함한 다양한 형태의 후방 분포를 추정합니다. UDA에서는, 출처 도메인 데이터에 레이블이 있고 목표 도메인 데이터에는 레이블이 없는 방식으로, Maximum Mean Discrepancy (MMD)를 손실 함수로 사용하여 잠재 특징의 공간을 정렬합니다.

- **Performance Highlights**: NPE와 UDA의 조합이 분석의 정확성을 1-2 배수 증가시키고, UDA가 없는 NPE 모델에 비해 후방 범위를 유의미하게 향상시켰습니다.



### A Survey on Physical Adversarial Attacks against Face Recognition Systems (https://arxiv.org/abs/2410.16317)
- **What's New**: 본 논문은 Face Recognition (FR) 시스템을 겨냥한 물리적 적대적 공격(physical adversarial attacks)에 대한 포괄적인 분석을 제공하고, 이 분야의 도전 과제 및 향후 연구 방향을 탐구합니다. 물리적 공격 방법을 세 가지 범주로 분류하고 각 범주의 연구 진전 상황을 요약합니다.

- **Technical Details**: FR 시스템은 딥 러닝(d deep learning) 기술의 발전으로 성능과 확장성에서 큰 진전을 이루었으며, 이러한 시스템에는 특정한 공격 기법이 요구됩니다. 디지털 공격(digital attacks)과 물리적 공격(physical attacks)으로 나눌 수 있는 적대적 공격의 유형을 설명하고, 이들 각각의 특징 및 도전 과제를 논의합니다. 예를 들어, 공격자는 얼굴에 부착된 스티커(stickers)나 액세서리(accessories)를 통해 FR 시스템을 속이는 방법을 사용합니다.

- **Performance Highlights**: 최근 연구에 따르면, 물리적 적대적 공격은 실제 환경에서 매우 효과적인 것으로 입증되었으며, FR 시스템의 보안에 중대한 위협이 됩니다. 이 논문은 2016년부터 2024년까지의 40편의 연구를 포괄적으로 분석하여, 물리적 공격과 관련된 최신의 발전사항을 제시하고 이 분야의 미래 연구 방향을 모색합니다.



### CirrMRI600+: Large Scale MRI Collection and Segmentation of Cirrhotic Liver (https://arxiv.org/abs/2410.16296)
- **What's New**: 이 논문에서는 CirrMRI600+ 데이터셋을 소개합니다. 이 데이터셋은 628개의 고해상도 복부 MRI 스캔으로, 간경변(cirrhosis) 환자에서 촬영되었습니다. 간경변을 연구하기 위한 최초의 데이터셋이며, 세분화(segmentation) 라벨이 주석 처리되어 있습니다.

- **Technical Details**: CirrMRI600+는 310개의 T1 가중(T1-weighted) 스캔과 318개의 T2 가중(T2-weighted) 스캔으로 구성되어 있으며, 진단 및 분석을 위한 다양한 알고리즘의 교차 검증을 가능하게 합니다. 이 데이터셋은 동시 사용으로 최적의 분석 성능을 낼 수 있도록 T1W 및 T2W 이미지를 포함하고 있습니다.

- **Performance Highlights**: 11개의 최신 딥 러닝(segmentation) 알고리즘을 사용한 성능 평가가 수행되었으며, 이 결과는 의료 이미징 커뮤니티의 알고리즘 개발을 촉진할 것으로 기대하고 있습니다. CirrMRI600+는 고유의 복합체로 일반화 가능성이 높은 세분화 모델 개발에 중요한 기초를 제공합니다.



### Unifying Subsampling Pattern Variations for Compressed Sensing MRI with Neural Operators (https://arxiv.org/abs/2410.16290)
- **What's New**: 이번 연구에서는 CS-MRI(Compressed Sensing MRI)에 대한 통합 모델을 제안합니다. 이 모델은 서로 다른 하위 샘플링 패턴과 이미지 해상도에 강인성이 있으며, 신경 연산자(neural operators) 기반의 아키텍처를 활용하여 이미지와 측정 공간(주파수 공간) 모두에서 작동합니다.

- **Technical Details**: 우리의 모델은 신경 연산자를 기반으로 하며, 기능 공간에서 재구성 우선 정보를 학습합니다. 이 모델은 하우스 홀드를 통해 측정 공간 신경 연산자를 사용하여 다양한 하위 샘플링 패턴을 다룰 수 있으며, 이미지 공간 신경 연산자는 다양한 이미지 해상도를 다룰 수 있도록 설계되었습니다. 이를 통해 로컬 및 글로벌 이미지 특징을 포착할 수 있습니다.

- **Performance Highlights**: 우리는 실험적으로 모델의 성능을 검증하였으며, 서로 다른 하위 샘플링 패턴에서 성능이 일관되게 유지됨을 보였습니다. 기존 방법보다 최대 4배 낮은 NMSE(Normalized Mean Squared Error)와 5dB PSNR(Peak Signal-to-Noise Ratio) 개선을 이루었으며, 제로샷 슈퍼 해상도 결과에서도 우수한 성능을 보였습니다.



### MvDrag3D: Drag-based Creative 3D Editing via Multi-view Generation-Reconstruction Priors (https://arxiv.org/abs/2410.16272)
Comments:
          16 pages, 10 figures, conference

- **What's New**: 본 논문에서는 다중 뷰 생성 및 재구성 사전 지식을 활용하여 더 유연하고 창의적인 3D 드래그 기반 편집을 가능하게 하는 MVDrag3D라는 새로운 프레임워크를 소개합니다. 이 기술은 여러 렌더링 뷰에서 일관된 드래그 편집을 수행할 수 있도록 다중 뷰 확산 모델을 사용합니다.

- **Technical Details**: MVDrag3D는 3D 물체의 네 개의 직교 뷰를 렌더링하고, 드래그 포인트를 해당 뷰에 투영하는 방식으로 작동합니다. 다중 뷰 확산 모델 내에서 점수 기반의 그래디언트 지침 메커니즘을 확장하여 모든 뷰에서 일관된 편집을 가능하게 하는 다중 뷰 지침 에너지 함수를 도입합니다. 이를 통해 이전 3D Gaussian들이 겹치는 영역에서 정확하게 정렬되지 않는 문제를 해결하기 위한 변형 네트워크를 설계하였습니다.

- **Performance Highlights**: 실험 결과 MVDrag3D는 다양한 객체 범주와 3D 표현에서 보다 다재다능한 편집 효과를 지원하며, 높은 정확성과 생성 능력을 기반으로 함을 입증합니다.



### FrugalNeRF: Fast Convergence for Few-shot Novel View Synthesis without Learned Priors (https://arxiv.org/abs/2410.16271)
Comments:
          Project page: this https URL

- **What's New**: FrugalNeRF는 복수의 스케일에서 weight-sharing voxels를 이용하여 장면의 세부정보를 효율적으로 표현하는 새로운 few-shot NeRF 프레임워크입니다.

- **Technical Details**: FrugalNeRF의 핵심 기여는 cross-scale geometric adaptation scheme으로, 스케일 간 reprojection errors를 기반으로 pseudo ground truth depth를 선택합니다. 이는 외부에서 학습된 priors에 의존하지 않으면서도 학습을 안내합니다.

- **Performance Highlights**: LLFF, DTU, RealEstate-10K에서의 실험 결과, FrugalNeRF는 타 few-shot NeRF 방법보다 우수한 성능을 보이며, 훈련 시간을 크게 단축시켜 효율적이고 정확한 3D 장면 재구성을 위한 실용적인 해결책이 됩니다.



### SAM2Long: Enhancing SAM 2 for Long Video Segmentation with a Training-Free Memory Tr (https://arxiv.org/abs/2410.16268)
Comments:
          Project page: this https URL

- **What's New**: 새롭게 등장한 SAM2Long는 기존 SAM 2의 메모리 모듈을 개선하여 복잡한 장기 비디오 객체 분할 성능을 향상시킵니다. SAM2Long는 각 프레임의 불확실성을 고려하고 여러 경로에서 최적 결과를 선택하는 새로운 방법을 도입하였습니다.

- **Technical Details**: SAM2Long는 지속적인 메모리 경로를 통해 많은 후보 마스크를 생성하고, 각 프레임의 누적 점수를 기반으로 실행 중 상당한 수의 경로를 유지하여 선택합니다. 각 경로는 단일 오브젝트 메모리 뱅크를 포함하여 신뢰할 수 있는 객체 단서만을 저장합니다.

- **Performance Highlights**: SAM2Long는 모든 24개의 헤드 투 헤드 비교에서 평균 3.0 포인트의 성능 향상을 이루었으며, SA-V와 LVOS와 같은 벤치마크에서 최대 5.3 포인트를 추가로 개선했습니다.



### xGen-MM-Vid (BLIP-3-Video): You Only Need 32 Tokens to Represent a Video Even in VLMs (https://arxiv.org/abs/2410.16267)
- **What's New**: xGen-MM-Vid (BLIP-3-Video)은 비디오를 위한 멀티모달 언어 모델로, 여러 프레임에서의 시간 정보를 효율적으로 캡처하도록 설계되었습니다. 이 모델은 일반적인 비주얼 토크나이저(visual tokenizer) 외에 'temporal encoder'를 도입하여 훨씬 적은 수의 visual tokens를 사용하여 비디오 질문-응답에서 높은 정확도를 기록합니다.

- **Technical Details**: BLIP-3-Video는 4개의 주요 구성 요소로 이루어져 있습니다: (1) 각 프레임 입력을 처리하는 비전 인코더(vision encoder), (2) 토큰 수를 줄이는 프레임 레벨 토크나이저(frame-level tokenizer), (3) 비디오 수준의 토큰 표현을 구축하는 템포럴 인코더(temporal encoder), (4) 비디오 토큰과 텍스트 프롬프트 토큰에 기반하여 출력 텍스트 캡션을 생성하는 자동 회귀 LLM입니다. 이 모델은 8개의 샘플링된 프레임을 사용하여 계산 효율성을 높입니다.

- **Performance Highlights**: BLIP-3-Video는 34B의 거대 모델과 비교하여 비슷한 질문-응답 정확도를 보이며, 단 4B로도 성능을 발휘합니다. 각각 16에서 32개의 비디오 토큰을 추상화하여 전체 비디오를 성공적으로 표현할 수 있습니다.



### 3DGS-Enhancer: Enhancing Unbounded 3D Gaussian Splatting with View-consistent 2D Diffusion Priors (https://arxiv.org/abs/2410.16266)
Comments:
          Accepted by NeurIPS 2024 Spotlight

- **What's New**: 본 논문은 3D Gaussian splatting (3DGS) 모델의 렌더링 성능을 크게 개선하기 위한 새로운 파이프라인인 3DGS-Enhancer를 제안합니다. 이 방법은 2D 비디오 디퓨전 모델을 활용하여 3D 뷰 일관성 문제를 해결하고, 이를 통해 고화질의 뷰 일관성을 유지하는 이미지를 복원합니다.

- **Technical Details**: 3DGS-Enhancer는 이미지 인코더, 비디오 기반의 디퓨전 모델, 그리고 공간-시간 디코더로 구성되어 있으며, 이는 렌더링된 뷰의 잠재 특징을 인코딩하고, 시간적으로 일관된 잠재 특징을 복원하며, 원래 렌더링된 이미지와 결합합니다. 이를 통해 초기 3DGS 모델을 파인튜닝하여 렌더링 성능을 향상시킵니다.

- **Performance Highlights**: 대규모 데이터셋에서 실험을 수행한 결과, 3DGS-Enhancer는 다양한 도전적인 장면에서 훌륭한 재구성 성능을 보여주며, 기존의 최신 방법들에 비해 더욱 뚜렷하고 생생한 렌더링 결과를 생성합니다.



### Mini-InternVL: A Flexible-Transfer Pocket Multimodal Model with 5% Parameters and 90% Performanc (https://arxiv.org/abs/2410.16261)
Comments:
          Technical report

- **What's New**: 새로운 연구는 Mini-InternVL이라는 시리즈를 소개하며, 이는 10억에서 40억 파라미터를 가진 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)입니다. 이 모델들은 5%의 파라미터로 90%의 성능을 달성하여 효율성을 크게 향상시킵니다.

- **Technical Details**: Mini-InternVL은 강력한 시각 인코더를 통합하여 3단계 훈련 과정을 거치는 모델로, 먼저 CLIP의 가중치를 사용하는 300M 시각 인코더를 초기화하고, 이후 InternViT-6B를 교사 모델로 사용하여 지식을 증류합니다. 이 모델은 다양한 다운스트림 작업에 쉽게 적응할 수 있는 일반적인 전이 학습 패러다임을 채택하였습니다.

- **Performance Highlights**: Mini-InternVL 시리즈는 MMBench, ChartQA, MathVista와 같은 다양한 멀티모달 벤치마크에서 우수한 성능을 보이며, 특히 Mini-InternVL-4B는 InternVL2-76B보다 90%의 성능을 보여줍니다. 또한 특정 도메인 작업에 대해서도 최소한의 컴퓨팅 비용으로 상용 모델에 필적하는 성능을 나타냅니다.



### Agent-to-Sim: Learning Interactive Behavior Models from Casual Longitudinal Videos (https://arxiv.org/abs/2410.16259)
Comments:
          Project page: this https URL

- **What's New**: ATS(Agent-to-Sim) 프레임워크는 긴 시간동안 수집된 비디오에서 3D 에이전트의 상호작용 행동 모델을 학습합니다. 기존의 마커 기반 추적 및 멀티뷰 카메라에 의존하지 않고, 자연스러운 비디오 관찰을 통해 침입 없이 동물 및 인간 에이전트의 행동을 학습하는 새로운 방법론을 제시합니다.

- **Technical Details**: ATS는 비디오 기반의 에이전트 행동 학습을 위한 프레임워크로, 20,000프레임의 비디오 데이터를 활용하여 완전하고 지속적인 4D 표현을 생성합니다. 이는 'coarse-to-fine registration' 방법을 통해 카메라와 에이전트의 위치를 정렬하여 3D 공간에서 에이전트와 환경을 지속적으로 추적할 수 있도록 합니다.

- **Performance Highlights**: ATS는 최대한 자연스러운 행동을 재현할 수 있으며, 실제 비디오 데이터를 이용해 에이전트가 주변 사람의 동작에 반응하고 3D 환경에 인식하며 상호작용할 수 있는 행동을 생성합니다. 또한, 고양이, 개 및 토끼와 같은 다양한 애완동물의 행동을 시뮬레이션함으로써 그 가능성을 입증합니다.



### Elucidating the design space of language models for image generation (https://arxiv.org/abs/2410.16257)
Comments:
          Project page: this https URL

- **What's New**: 이 논문은 이미지 생성을 위한 언어 모델의 최적화 동작을 분석하여, 언어 모델(특히 LLMs)의 제약을 극복할 수 있는 효과적인 설계 원리를 제시합니다. 이를 통해, ELM이라는 이미지 생성 언어 모델이 ImageNet 256×256 벤치마크에서 최첨단 성능을 달성하였음을 보여줍니다.

- **Technical Details**: 본 연구는 이미지와 텍스트 간의 근본적인 차이를 감안하여, 이미지 생성을 위한 언어 모델 설계 공간을 상세히 분석합니다. 우리는 두 가지 토크나이저 방식, 즉 VQGAN 및 BAE를 비교하고, 이미지의 분산된 패턴 학습에서 AR(autoregressive) 모델과 MLM(masked language model)의 성능 차이를 검토합니다. 또한, BAE의 바이너리 코드에서 코드 분해 전략을 활용하여 성능 향상을 도모합니다.

- **Performance Highlights**: AR 모델은 이미지 생성에 있어 더 뛰어난 능력과 확장성을 보여주며, ELM 모델은 ImageNet 256×256 벤치마크에서 최고의 성능을 기록했습니다. 이러한 성능 향상은 모델 크기를 늘릴수록 더 두드러지며, 다양한 실험을 통해 AR 모델이 이미지 생성에서 차지하는 중요성을 입증합니다.



### Revisiting Deep Feature Reconstruction for Logical and Structural Industrial Anomaly Detection (https://arxiv.org/abs/2410.16255)
Comments:
          Accepted in Transactions on Machine Learning Research (TMLR). Link to OpenReview: this https URL

- **What's New**: 본 연구에서는 Deep Feature Reconstruction (DFR) 기술을 기반으로 하여 구조적 및 논리적 이상 감지를 동시에 수행할 수 있는 통합 프레임워크인 ULSAD를 제안합니다. 이를 통해 메모리와 계산 자원을 효율적으로 사용하면서도 이상 감지 성능을 높이는 것을 목표로 합니다.

- **Technical Details**: ULSAD는 구조적 이상 감지를 위해 ω_2거리와 코사인 거리의 조합을 고려하여 훈련 목표를 수정하고, 논리적 이상 감지를 위한 글로벌 오토인코더 기반의 어텐션 주의 손실 메커니즘을 도입합니다. 실험 결과, ULSAD는 기존의 8종의 최신 기술(SOTA) 방법들보다 우수한 성능을 보여줍니다.

- **Performance Highlights**: ULSAD는 5개의 검증된 IAD (Industrial Anomaly Detection) 데이터셋에서 성능 평가를 진행하였으며, 구조적 및 논리적 이상 감지와 위치 지정 모두에서 이전의 방법들보다 우수한 성능을 나타냈습니다. 추가적인 아블레이션 연구를 통해 각 구성 요소가 전체 성능 향상에 미치는 기여를 상세히 설명하였습니다.



### LLaVA-KD: A Framework of Distilling Multimodal Large Language Models (https://arxiv.org/abs/2410.16236)
Comments:
          Under review

- **What's New**: 이번 논문에서는 LLM(대형 언어 모델)로부터 s-MLLM(소형 다중 모달 대형 언어 모델)로 지식을 전이하기 위한 새로운 LLaVA-KD 프레임워크를 제안합니다. LLaVA-KD는 Multimodal Distillation(MDist)과 Relation Distillation(RDist)을 도입하여 소형 모델의 성능 저하 문제를 해결합니다.

- **Technical Details**: LLaVA-KD 프레임워크는 세 가지 단계의 훈련 방식으로 구성됩니다: 1) Distilled Pre-Training(DPT)에서 소형 모델의 시각-언어 표현을 정렬하고, 2) Supervised Fine-Tuning(SFT)을 통해 다중 모달 이해를 강화하며, 3) Distilled Fine-Tuning(DFT)으로 대형 모델의 능력을 효과적으로 전이합니다. 또한, Kullback-Leibler Divergence(KLD)를 사용하여 시각 및 언어 모달리티 간의 분산을 최소화합니다.

- **Performance Highlights**: 제안하는 LLaVA-KD 모델은 1B의 파라미터로 BLIP2-13B 및 InstructBLIP-7B와 같은 대형 MLLM 모델을 포함한 다양한 다중 모달 벤치마크에서 뛰어난 성능을 보여주며, 모델의 크기와 계산 비용을 대폭 줄입니다.



### Training Better Deep Learning Models Using Human Saliency (https://arxiv.org/abs/2410.16190)
- **What's New**: 이번 연구는 이미지의 중요한 영역에 대한 인간의 판단이 심층 합성곱 신경망(DCNN) 훈련에 어떻게 도입될 수 있는지를 탐구합니다. 기존의 DCNN 훈련은 순수하게 데이터 중심의 접근 방식이었으나, 인간의 주목성을 활용하여 새로운 손실 함수 구성 요소인 CYBORG를 제안하고, 비주목적 영역에 대한 페널티를 부여함으로써 모델 성능을 향상시키는 방안을 소개합니다.

- **Technical Details**: CYBORG는 human saliency와 모델 saliency 간의 차이를 비교하여 일반화를 높이는 방식으로 동작하며, 세 가지 도메인(합성 얼굴 탐지,홍채 프리젠테이션 공격 탐지 및 흉부 X-레이의 이상 탐지)에서 실험을 통해 효과를 검증합니다. 이 연구에서는 이미지에 대한 비전문가의 주목성 주석을 통해 모델의 일반화 능력이 향상됨을 보였습니다.

- **Performance Highlights**: CYBORG 훈련을 통한 DCNN 모델은 인간 주목성을 반영하여 테스트 세트에서의 일관성을 높이고, 다양한 도메인에서 향상된 정확성과 일반화 능력을 보였습니다. CYBORG 방법론은 또한 인공지능 시스템의 해석 가능성을 높이며, 데이터가 부족한 환경에서도 성능을 유지하는 데 도움을 줍니다.



### A Framework for Evaluating Predictive Models Using Synthetic Image Covariates and Longitudinal Data (https://arxiv.org/abs/2410.16177)
- **What's New**: 본 논문에서는 복잡한 공변량(complex covariates)과 장기 관측(longitudinal observations)을 결합한 환자 데이터 합성을 위한 새로운 프레임워크를 제시합니다. 이 방법은 의료 연구에서의 개인 정보 보호 문제를 해결하는데 기여합니다.

- **Technical Details**: 이 프레임워크는 잠재 공간(latent spaces)에서 각 데이터 모달리티를 생성하는 제어된 연관성(controlled association)을 도입합니다. Optical Coherence Tomography (OCT) 이미지를 기반으로 한 이미지 생성 모델(Variational Autoencoder와 Diffusion 모델 결합)을 사용해 109,309개의 2D OCT 스캔 슬라이스로 훈련하였습니다. 또한, 비선형 혼합 효과 모델(Nonlinear Mixed Effect Model, NLME)을 사용하여 장기 관측을 시뮬레이션 하였습니다.

- **Performance Highlights**: 우리는 1.1M 개의 OCT 스캔 슬라이스와 5개의 장기 관측 세트를 생성하였고, 예측 모형 평가에서 제어된 연관성 수준을 조절함으로써 예측의 정확도가 의도된 대로 감소함을 확인할 수 있었습니다. 2% 사례를 제외한 모든 경우에서, withheld data에 대해 이론적인 최상의 예측값의 50% 이내에 도달하여 약한 신호도 감지할 수 있는 능력을 보여주었습니다.



### Beyond Filtering: Adaptive Image-Text Quality Enhancement for MLLM Pretraining (https://arxiv.org/abs/2410.16166)
- **What's New**: 본 논문에서는 멀티모달 대형 언어 모델(MLLMs)의 성능을 향상시키기 위한 새로운 접근법인 Adaptive Image-Text Quality Enhancer (AITQE)를 제안합니다. AITQE는 기존 데이터의 질을 동적으로 평가하고 향상시키는 모델로, 이미지-텍스트 쌍의 품질을 개선합니다.

- **Technical Details**: AITQE는 낮은 품질의 이미지-텍스트 쌍에 대해 텍스트를 재작성하는 메커니즘을 사용하고, 부정 샘플 학습 전략을 통합하여 평가 능력을 향상시킵니다. 본 모델은 기존 방법들보다 텍스트 분포를 최소한으로 조정하여 데이터 볼륨을 유지하면서 품질을 개선합니다.

- **Performance Highlights**: 실험 결과, AITQE는 여러 벤치마크에서 기존 방법들을 초월하여 원시 데이터를 효과적으로 활용하고, 데이터 양 증가에 따라 효율적으로 확장할 수 있음을 보여주었습니다.



### Griffon-G: Bridging Vision-Language and Vision-Centric Tasks via Large Multimodal Models (https://arxiv.org/abs/2410.16163)
Comments:
          This work has been submitted to the IEEE for possible publication. Codes and data will be later released at this https URL

- **What's New**: 이 논문은 비전 중심 작업(vision-centric tasks)과 비전-언어 작업(vision-language tasks)을 통합한 새로운 다차원 정제(multi-dimension curated) 및 통합(multimodal dataset) 데이터셋인 CCMD-8M을 소개하고, 이를 바탕으로 비전 중심 및 비전-언어 작업을 모두 처리할 수 있는 통합 모델인 Griffon-G를 개발하였습니다.

- **Technical Details**: CCMD-8M 데이터셋은 420만 개의 정제된 사전 훈련 샘플(pre-training samples) 및 410만 개의 포괄적인 지시 준수 샘플(instruction-following samples)로 구성되어 있으며, 이를 통해 비전 중심 및 비전-언어 작업 간의 데이터 차이를 해소하고 데이터 품질을 개선합니다. Griffon-G는 Paradigm Progressive Learning Pipeline을 통해 다양한 작업의 공동 최적화 시 발생하는 훈련 붕괴 문제를 해결하며, 자동 회귀 모델링(auto-regressive modeling) 방식으로 작업을 통합할 수 있습니다.

- **Performance Highlights**: Griffon-G는 여러 멀티모달 벤치마크 및 도전적인 VQA 작업들에서 최신 LMM들을 초과하여 전문가 수준의 성능을 달성하였습니다. 특히, 비전 중심 작업인 객체 탐지(object detection) 및 지시 표현 이해(Referring Expression Comprehension, REC)에서도 동급 최강의 성과를 달성하였습니다.



### Sparkle: Mastering Basic Spatial Capabilities in Vision Language Models Elicits Generalization to Composite Spatial Reasoning (https://arxiv.org/abs/2410.16162)
- **What's New**: Vision Language Models (VLMs)의 2D 공간 추론 능력을 향상시키기 위한 새로운 접근법인 Sparkle 프레임워크를 소개합니다.

- **Technical Details**: 우리의 연구는 방향 이해(direction comprehension), 거리 추정(distance estimation), 위치 파악(localization)으로 구성된 2D 공간 추론의 기초 능력을 분리하고 강화하는 방법론을 사용합니다.

- **Performance Highlights**: Sparkle로 미세 조정된 VLMs는 기본 작업에서 13.5%에서 40.0%까지 성능 향상을 보여주며, 복합 및 비분포적 공간 추론 작업으로의 일반화에도 긍정적인 결과를 나타냈습니다.



### Warped Diffusion: Solving Video Inverse Problems with Image Diffusion Models (https://arxiv.org/abs/2410.16152)
Comments:
          Accepted in NeurIPS 2024

- **What's New**: 이 논문은 동영상을 생성하는 데 있어 기존의 이미지 모델을 활용하면서 발생할 수 있는 깜박임(flickering), 텍스처 고착(texture-sticking), 시간적 불일치(temporal inconsistency) 문제를 해결하기 위한 새로운 접근 방식을 제안합니다.

- **Technical Details**: 이 연구에서 제안하는 방법은 2D 공간에서 프레임을 연속 함수로 보고 동영상을 서로 다른 프레임 간의 연속 변형 연속(sequence of continuous warping transformations)으로 간주합니다. 이를 통해 기능 공간(diffusion model)에서 훈련된 모델을 사용하여 시간적으로 상관된 역 문제를 해결하도록 합니다. 주요 기술적 요소는 동작 변형에 대한 동등성(equivariance) 요구 사항과, 이를 보장하기 위한 후처리(post-hoc) 방법으로 동등성 자기 안내를 도입하는 것입니다.

- **Performance Highlights**: 제안된 방법은 동영상 인페인팅(video inpainting) 및 8배 영상 초해상도(8× video super-resolution)에서 기존 기술들에 비해 눈에 띄게 우수한 성능을 보이며, 깜빡임과 텍스처 고착과 같은 아티팩트를 줄이는 데 효과적입니다. 논문에서 제시된 결과들은 Stable Diffusion XL과 같은 최신 잠재적 확산 모델(latent diffusion models)에서 더 나은 성능을 보여줍니다.



### Increasing Interpretability of Neural Networks By Approximating Human Visual Saliency (https://arxiv.org/abs/2410.16115)
- **What's New**: 이 논문에서는 saliency(살리언시) 정보를 모델 훈련 과정에 통합하고, 능동 학습(active learning) 원칙을 활용하여 필요한 인간 주석 데이터를 80%까지 줄이는 새로운 접근 방식인 Saliency in Active Learning(SAL)을 제안합니다.

- **Technical Details**: SAL 방법은 초기 단계에서 인간이 샘플에 대한 saliency 주석을 제공하며, 이후 AI 모델이 이 주석을 기반으로 고정밀의 saliency 맵을 생성하도록 합니다. 이 과정은 Convolutional Neural Networks (CNN)와 Transformer 기반 아키텍처 모두에 적용됩니다.

- **Performance Highlights**: SAL은 saliency 정보가 포함되지 않은 모델에 비해 모델 해석 가능성을 30%까지 증가시키고, 5배 많은 인간 saliency 주석을 사용한 모델과 동등한 성능을 나타냅니다. 실험은 5개의 공개 데이터셋에서 수행되었습니다.



### LMHaze: Intensity-aware Image Dehazing with a Large-scale Multi-intensity Real Haze Datas (https://arxiv.org/abs/2410.16095)
- **What's New**: LMHaze라는 대규모의 고품질 실제 세계 데이터셋을 제시하며, 다양한 실내 및 실외 환경에서 촬영된 5천 개 이상의 고해상도 이미지 쌍을 포함하고 있습니다. 이 데이터셋은 기존 데이터셋보다 25배 이상 큰 규모를 자랑합니다.

- **Technical Details**: 제안된 Mixture-of-Experts 모델인 MoE-Mamba는 다양한 안개 강도에 따라 모델 매개변수를 동적으로 조정하여 이미지의 해상도를 개선합니다. 이 모델은 LMM(large multimodal model)-기반 블록, MoE 블록, 및 상태 공간 모델 블록으로 구성됩니다.

- **Performance Highlights**: LMHaze 데이터셋은 실제 시나리오에서의 해상도 성능을 개선하며, 제안된 해상도 방법은 최신 기법들과 비교하여 더 나은 결과를 제공합니다.



### Integrated Image-Text Based on Semi-supervised Learning for Small Sample Instance Segmentation (https://arxiv.org/abs/2410.16063)
- **What's New**: 이번 논문은 적은 샘플의 인스턴스 분할(instance segmentation) 문제를 해결하기 위한 새로운 접근 방식을 제안합니다. 기존 메타-러닝(meta-learning) 전략을 대체하여, 추가적인 주석(annotation) 부담과 훈련 비용을 증가시키지 않고 기존 정보의 최대화를 통해 효과적인 해결책을 제공합니다.

- **Technical Details**: 제안된 방법은 두 개의 모듈로 구성됩니다. 첫 번째 모듈은 무주석 데이터(unlabeled data)를 활용하여 생성된 의사 레이블(pseudo labels)을 학습함으로써 샘플 수를 증가시킵니다. 두 번째로, 텍스트와 이미지의 특징을 통합하여 보다 정확한 분류 결과(classification results)를 도출합니다. 이 두 모듈은 박스-프리(box-free) 및 박스-의존(box-dependent) 프레임워크에 적합합니다.

- **Performance Highlights**: 세 가지 서로 다른 장면(육상, 수중, 현미경)에서 실험을 진행하였으며, 결과적으로 통합된 이미지-텍스트가 분류의 신뢰성을 향상시키고, 의사 레이블이 모델로 하여금 더 정밀한 마스크(mask)를 획득하도록 도와줍니다. 모든 실험 결과는 제안된 방법의 효과성과 우수성을 입증합니다.



### Label Filling via Mixed Supervision for Medical Image Segmentation from Noisy Annotations (https://arxiv.org/abs/2410.16057)
- **What's New**: 의료 이미지 분할(medical image segmentation)에서 노이즈가 있는 주석(noisy annotations)을 통해 진짜 세그멘테이션 레이블을 예측하는 Label Filling 프레임워크(LF-Net)를 제안합니다.

- **Technical Details**: LF-Net은 신뢰할 수 있는 픽셀의 하위 집합을 통해 세그멘테이션 모델(segmentation model)을 감독하고, 나머지 픽셀에 대한 레이블을 혼합 감독(mixed supervision)으로 채우는 접근 방식을 사용합니다. 신뢰할 수 있는 픽셀에 대한 서포트와 두 가지 종류의 혼합 보조 감독(soft label과 raters' characteristics labels)을 통해 장점이 있습니다.

- **Performance Highlights**: LF-Net은 MS 병변(segment MS lesions) 세그멘테이션에서 DSC(Dice-Sørensen coefficient)가 7% 향상되었으며, 다섯 가지 다양한 이미지 데이터세트에서 모든 데이터세트의 세그멘테이션 정확도를 향상시켰습니다.



### Benchmarking Pathology Foundation Models: Adaptation Strategies and Scenarios (https://arxiv.org/abs/2410.16038)
- **What's New**: 이 연구에서는 병리학 이미지를 분석하기 위한 병리학 특정의 Foundation Model의 일관성과 유연성 평가 시나리오를 통해 총 14개의 데이터셋에 대해 4가지 모델을 벤치마킹하였다. 특히, 데이터 적은 환경에서의 Few-shot Learning(FSL) 방법의 효과에 대한 통찰을 제공한다.

- **Technical Details**: 이 연구에서는 CTransPath, Lunit, Phikon, UNI 등 4개의 병리학 특정 Foundation Models를 사용하여 일관성 및 유연성 평가 시나리오에서 각각의 모델을 다양한 하위 작업에 적응시키는 방법을 분석했다. 일관성 평가에서는 매개변수 효율적인 세부 조정 방법(Parameter-Efficient Fine-Tuning, PEFT)이 다양한 데이터셋에 적용될 때 효과적이라는 것을 발견하였다.

- **Performance Highlights**: PEFT 방법을 통해 기존 데이터셋 내에서 유연한 적응 능력을 달성하였으며, 데이터가 제한된 환경에서 FSL 방법이 뛰어난 효과를 보였다. 이러한 발견은 실제 임상 환경에서의 병리학 이미지 분석의 정확성과 신뢰도를 향상시키는 데 기여할 수 있다.



### Improving the Multi-label Atomic Activity Recognition by Robust Visual Feature and Advanced Attention @ ROAD++ Atomic Activity Recognition 2024 (https://arxiv.org/abs/2410.16037)
- **What's New**: Road++ Track3에서는 교통 시나리오에서 멀티 레이블 원자 활동 인식(multi-label atomic activity recognition) 임무를 제안하고, 이를 64 클래스 멀티 레이블 비디오 행동 인식(task)으로 표준화했습니다. 기존의 단일 레이블 메소드와 달리, 멀티 레이블 접근법을 통해 동시에 발생하는 다양한 액션을 인식하는 것이 핵심입니다.

- **Technical Details**: 데이터 처리, 모델 최적화 및 후처리 방법에서 세 가지 주요 측면을 최적화했습니다. 데이터 처리에서는 해상도와 비디오 샘플링 전략을 정선하고, 모델 학습 시 다양한 비주얼 백본 네트워크를 활용하며, 후처리에서 각 모델의 강점을 결합하여 가중 융합(weighted fusion)을 통해 성과를 높였습니다.

- **Performance Highlights**: 테스트 세트에서 58%의 평균 정밀도(mean Average Precision, mAP)를 달성하여 과제 베이스라인보다 4% 향상된 성과를 보였습니다.



### Few-shot target-driven instance detection based on open-vocabulary object detection models (https://arxiv.org/abs/2410.16028)
- **What's New**: 본 논문은 Open-Vocabulary 객체 검출 모델을 TDID(Target Driven Instance Detection) 모델로 변환하는 경량화된 방법을 제안하며, 텍스트 설명 없이도 One-shot 또는 Few-shot 객체 인식이 가능하다는 점이 새롭습니다.

- **Technical Details**: 이 연구에서는 YOLO-World 모델을 기반으로 하여, 기존 Open-Vocabulary 모델의 입력 임베딩을 텍스트에서 이미지로 변경하는 방식을 채택하였습니다. 이 과정은 학습 필요 없이 단 몇 개의 예시 이미지만으로 객체 인식이 가능합니다. 이 방법은 특히 시각 장애인을 위한 개인화된 객체 검출기 개발에 유용합니다.

- **Performance Highlights**: TEgO 데이터셋을 사용한 실험 결과, 모델의 크기와 예시 수, 이미지 증강(image augmentation) 사용이 성능 향상에 긍정적인 영향을 미쳤음을 보였습니다.



### START: A Generalized State Space Model with Saliency-Driven Token-Aware Transformation (https://arxiv.org/abs/2410.16020)
Comments:
          Accepted by NeurIPS2024. The code is available at this https URL

- **What's New**: 이번 연구에서는 State Space Models (SSMs) 기반의 새로운 아키텍처인 START를 제안하여 Domain Generalization (DG) 성능을 향상시킵니다. 기존의 CNN 및 Vision Transformers (ViTs)와 비교하여 STATE-OF-THE-ART(SOTA) 성능을 달성하였습니다.

- **Technical Details**: START는 Saliency 기반 토큰 인식 변환을 통해 입력 의존적 행렬 내에서 도메인 특정 특징을 선택적으로 교란 및 억제하여 도메인 간 불일치를 줄이는 기능을 갖추고 있습니다. 이 방법은 선형 복잡성으로 토큰 간의 전역 의존성을 학습하여 모델의 일반화 능력을 향상시킵니다.

- **Performance Highlights**: 다섯 가지 벤치마크에서의 실험 결과, START는 기존의 SOTA DG 방법들을 초월하여 우수한 성능을 보였습니다. 특히 TerraIncognita 데이터셋에서 CNN 기반 방법보다 5.87% 높은 성능(58.27% vs 52.40%)을 기록하였습니다.



### Multispectral Texture Synthesis using RGB Convolutional Neural Networks (https://arxiv.org/abs/2410.16019)
- **What's New**: 본 논문에서는 RGB 이미지를 위한 기존의 텍스처 합성 알고리즘을 다중 스펙트럼 이미지로 확장하기 위한 두 가지 방법을 제안합니다. 이 방법들은 추가적인 신경망 학습을 필요로 하지 않으며, 스펙트럴 밴드의 랜덤 트리플릿 배치를 최적화하거나, 다중 스펙트럴 픽셀을 3차원 공간에 투영하는 방식으로 구현됩니다.

- **Technical Details**: 제안된 방법은 두 가지 스타일 거리(stochastic style distance 및 projected style distance)를 도입합니다. 이러한 스타일 거리는 사전 훈련된 RGB CNN에서 추출한 통계를 기반으로 계산됩니다. 이는 스타일 거리를 기반으로 다중 스펙트럼 이미지를 합성하기 위한 최적화 프로세스를 가능하게 합니다. 또한, 프로젝션으로 인해 유도된 색상 도메인 이동(color domain shift)을 수정하기 위해 색상 전송(color transfer) 기법이 도입됩니다.

- **Performance Highlights**: 130개의 다중 스펙트럼 클라우드 이미지를 사용하여 여러 방법들의 성능을 비교한 결과, 제안된 방법들이 양호한 시각적 품질을 보여주며 기존의 최첨단 RGB 밴드 합성 기법에 근접한 성능을 달성함을 입증하였습니다.



### Massimo: Public Queue Monitoring and Management using Mass-Spring Mod (https://arxiv.org/abs/2410.16012)
Comments:
          8 pages, 6 figures, 3 algorithms, 3 tables

- **What's New**: 이 논문은 공공 공간에서의 대기열 관리 문제를 해결하기 위한 새로운 기술을 제안합니다. 이 기술은 YOLO(You Only Look Once) 모델과 머신 러닝 알고리즘을 활용하여 가능한 대기 경로를 최적화하고, 비정상적인 대기 상태를 감지하는 시스템을 개발합니다.

- **Technical Details**: 제안된 시스템은 YOLOv7 및 YOLOv8 모델을 사용하여 이미지에서 개인의 신체 점을 감지하고, 이를 통해 대상 대기 이력을 분석합니다. 계산된 신체 점의 힘을 시각화하기 위해 가상의 스프링 시스템을 도입하여 사람 간의 상호작용을 모델링합니다. 또한 선형 회귀 및 다항 회귀와 같은 회귀 분석 기법을 통해 대기 경로를 최소화하려고 합니다.

- **Performance Highlights**: 본 시스템을 통해 대기 시간 단축 및 자원 할당 최적화를 달성하고, COVID-19 이후 시대의 시장 요구에 부응하는 개선된 서비스 품질을 제공할 것으로 기대됩니다.



### 3D-GANTex: 3D Face Reconstruction with StyleGAN3-based Multi-View Images and 3DDFA based Mesh Generation (https://arxiv.org/abs/2410.16009)
Comments:
          7 pages, 4 figures, 2 tables, pre-print version

- **What's New**: 이 논문은 단일 얼굴 이미지에서 기하학적 형태(geometry)와 텍스처(texture) 추정 문제를 해결하기 위한 새로운 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 StyleGAN과 3D Morphable Models(3DMM)을 활용하여 시작합니다. 이 방법은 GAN의 잠재 공간(latent space)을 사용하여 다각도(multi-view) 얼굴을 생성한 후, 3DMM에 대해 훈련된 3DDFA를 이용하여 3D 얼굴 메쉬와 얼굴 형태에 일치하는 고해상도 텍스처 맵을 추정합니다.

- **Performance Highlights**: 생성된 메쉬는 고품질이며 텍스처 표현이 거의 정확하여 성능이 우수한 결과를 보여줍니다.



### Visual Representation Learning Guided By Multi-modal Prior Knowledg (https://arxiv.org/abs/2410.15981)
- **What's New**: 이번 논문에서는 다중 양식(Modal) 사전 지식을 활용한 Knowledge-Guided Visual representation learning (KGV) 접근 방식을 소개하여, 이미지 분류에서 데이터 분포 변화에 대한 일반화 성능을 향상시키는 방법을 제안합니다.

- **Technical Details**: KGV는 지식 그래프(KG)와 생성된 합성 이미지로부터 얻은 두 가지 모달리티로부터 사전 지식을 활용하여 이미지 임베딩과 KG 임베딩을 공통 잠재 공간에서 정렬합니다. 새로운 변형의 Translation-based KGE(Method) 방식으로 KG의 노드와 관계 임베딩을 각각 Gaussian 분포와 변환으로 모델링함으로써 오버피팅(overfitting)을 방지하는 정규화 효과를 줍니다.

- **Performance Highlights**: KGV는 독일, 중국, 러시아 등 다양한 데이터 세트에서의 도로 표지판 분류 및 mini-ImageNet과 그 변형, DVM-CAR 데이터 세트를 포함한 다양한 이미지 분류 작업에서 기존 기법에 비해 일관되게 높은 정확도와 데이터 효율성을 발휘하였습니다. 특정 실험에서는 ImageNet과 도로 표지판 인식 도메인에서 평균 4.4% 및 4.1% 향상된 성과를 보여주었으며, DVM-CAR 데이터 세트에서도 SOTA(State-Of-The-Art) 결과를 개선했습니다.



### Granularity Matters in Long-Tail Learning (https://arxiv.org/abs/2410.15980)
- **What's New**: 이 연구는 long-tail data distribution에서 granularity를 증가시켜 tail class의 성능을 향상시키는 새로운 방법을 제안합니다. 기존의 방법들과 달리, 데이터셋의 세분화가 불균형 문제 해결에 도움을 줄 수 있다는 관찰에서 영감을 받았습니다.

- **Technical Details**: 제안된 방법은 open-set auxiliary classes를 통해 데이터셋의 granularity를 증가시킴으로써 비슷한 시각적 특성을 가진 클래스를 추가하여 학습을 보강합니다. 또한, large language models (LLMs)를 활용해 자동으로 보조 데이터를 수집하며, neighbor-silencing loss를 통해 보조 클래스의 영향력을 줄여 타겟 데이터셋의 클래스 구별에 집중할 수 있도록 합니다.

- **Performance Highlights**: 세 가지 표준 long-tail 벤치마크에서 실험 결과, 이 방법은 같은 양의 데이터를 사용한 강력한 baseline 방법들보다 뛰어난 성능을 보여주며, ImageNet-LT에서 tail 클래스 성능을 16.0% 향상시키고, Places-LT에서는 8.3% 향상시킵니다.



### Zero-Shot Scene Reconstruction from Single Images with Deep Prior Assembly (https://arxiv.org/abs/2410.15971)
Comments:
          To appear at NeurIPS 2024. Project page: this https URL

- **What's New**: 이번 연구에서는 single image로부터 scene reconstruction을 수행하기 위한 새로운 프레임워크인 deep prior assembly를 제안합니다. 이는 대규모 모델로부터 다양한 deep priors를 조합하여 zero-shot 방식으로 작동할 수 있게 합니다.

- **Technical Details**: deep prior assembly는 poses, scales, occlusion parsing과 같은 새로운 방법을 도입하여 deep priors가 협력하여 robust하게 작동할 수 있도록 합니다. 이 과정에서 Grounded-SAM을 통해 입력 이미지의 인스턴스를 감지하고, Stable-Diffusion을 활용해 저해상도 인스턴스 이미지를 개선 및 보완합니다.

- **Performance Highlights**: 다양한 open-world scenes에 대한 평가 결과, deep prior assembly는 단일 view 각도에서 다양한 개체를 재구성하고 신뢰할 수 있는 레이아웃을 복원하는 데 필요한 우수한 성능을 보여주었습니다. 최신 방법들과의 비교 분석을 통해 연구의 우수성을 입증하였습니다.



### A Paradigm Shift in Mouza Map Vectorization: A Human-Machine Collaboration Approach (https://arxiv.org/abs/2410.15961)
Comments:
          13 pages including reference, 14 figures, 4 tables

- **What's New**: 이 연구는 방글라데시의 손으로 그린 카다스트럴 지도인 마우자 지도의 효율적인 벡터화 방법을 제안합니다. 기관제작의 과정을 간소화할 수 있는 반자동(bi-automated) 방안을 통해 시간과 인적 자원을 절약하려고 합니다.

- **Technical Details**: 제안된 방법론은 플롯 경계(Plot boundaries) 및 플롯 식별자(Plot identifiers)를 분리하고, Convolutional Neural Network (CNN) 모델을 사용하여 이를 벡터 형식으로 변환합니다. 자동화된 프로세스를 도입하면서도 정밀도를 확보하기 위해 인간의 개입이 필요한 부분이 남아있습니다.

- **Performance Highlights**: 이 방법은 기존 수작업 벡터화와 비교하여 최대 50%의 시간 절약을 보여주었습니다. 예를 들어, 25시간이 소요되던 수작업이 13시간으로 단축되었습니다. 이 방법은 방글라데시와 같은 국가에서 대규모 카다스트럴 지도의 디지털화에 필요한 효율성을 제공합니다.



### CamI2V: Camera-Controlled Image-to-Video Diffusion Mod (https://arxiv.org/abs/2410.15957)
- **What's New**: 이 논문에서는 기존의 카메라 조건 텍스트-비디오 확산 모델이 가진 한계를 지적하고, 물리적 지식을 통합하기 위해 새로운 'epipolar attention' 메커니즘과 'register tokens'를 도입하여 카메라 제어를 개선합니다. 또한, 모델의 정확성을 높이기 위한 평가 파이프라인을 구축했습니다.

- **Technical Details**: 우리는 카메라 포즈를 'epipolar attention' 메커니즘과 결합하여 표현했습니다. 이 기법은 모든 노이즈 프레임에서 대응되는 epipolar 선을 따라 특징을 집계하며, 카메라 이동과 잡음에 의해 가려진 특성의 추적 한계를 극복합니다. 또한, 'register tokens'을 도입하여 프레임 간 교차점이 없을 때의 문제를 해결했습니다.

- **Performance Highlights**: RealEstate10K 데이터셋에서 카메라 제어 가능성을 25.5% 향상시켰으며, 도메인 외 이미지에 대한 일반화 성능도 강력하게 유지했습니다. 학습 및 추론에 각각 24GB와 12GB의 메모리만으로 가능하다는 점이 강조됩니다.



### MBPU: A Plug-and-Play State Space Model for Point Cloud Upsamping with Fast Point Rendering (https://arxiv.org/abs/2410.15941)
- **What's New**: 본 논문에서는 Mamba 아키텍처를 기반으로 한 새로운 포인트 클라우드 업샘플링 네트워크인 MBPU를 소개합니다. MBPU는 긴 시퀀스 모델링에서 뛰어난 성능을 발휘하며, 특히 대규모 포인트 클라우드 업샘플링에서 빠른 수렴 속도를 달성합니다.

- **Technical Details**: MBPU는 중간점 보간(midpoint interpolation) 및 MLP와 Mamba 모듈을 결합한 밀집 네트워크를 사용하여 로컬 및 글로벌 특징을 추출합니다. 또한, 3D 위치 이동 및 1D 포인트 간 거리 예측을 통해 로컬 상세와 글로벌 구조를 동시에 보존합니다. 마지막으로, 빠른 미분 가능한 렌더링 모듈을 도입하여 업샘플링된 포인트 클라우드의 충실도를 향상시키고 아티팩트를 줄입니다.

- **Performance Highlights**: MBPU는 긴 시퀀스를 처리하는 데 더 나은 성능을 발휘하며, 빠른 수렴과 함께 더 적은 축소 아티팩트로 포인트 클라우드를 업샘플링하는 능력을 보여줍니다. 대규모 포인트 클라우드에서 기존 방법들을 능가하는 성과를 입증하였습니다.



### Focus on BEV: Self-calibrated Cycle View Transformation for Monocular Birds-Eye-View Segmentation (https://arxiv.org/abs/2410.15932)
- **What's New**: 이번 논문에서는 Birds-Eye-View (BEV) 세분화 분야에 새로운 FocusBEV 프레임워크를 제안합니다. 이 프레임워크는 BEV-agnostic 지역을 억제하고, BEV-관련 지역에 집중할 수 있도록 설계된 자가 보정(self-calibrated) 뷰 변환 모듈을 포함하고 있습니다. 또한 기억 은행을 활용하여 시공간 구조 일관성을 확보하는 모듈과 점유율에 무관한 IoU 손실을 도입하였습니다.

- **Technical Details**: FocusBEV 프레임워크는 다음과 같은 세 가지 주요 모듈로 구성됩니다: (i) BEV-agnostic 이미지를 억제하고 BEV-관련 이미지에 집중하는 자가 보정된 크로스 뷰 변환 모듈, (ii) 시공간 구조의 일관성을 위해 플러그 앤 플레이(plug-and-play) 타입의 자아 동작 기반(e-go motion) 시계열 융합 모듈, (iii) 점유율에 무관한 IoU 손실을 사용하여 의미적 및 위치적 불확실성을 완화합니다.

- **Performance Highlights**: 우리의 접근 방식은 인기 있는 두 개의 벤치마크에서 신규 최첨단 성능을 달성하였습니다: nuScenes에서 29.2% mIoU, Argoverse에서 35.2% mIoU를 기록했습니다.



### GReFEL: Geometry-Aware Reliable Facial Expression Learning under Bias and Imbalanced Data Distribution (https://arxiv.org/abs/2410.15927)
Comments:
          ACCV 2024. Extended version of ARBEx (arXiv:2305.01486)

- **What's New**: 본 연구에서는 GReFEL(GEometry-based Reliable Facial Expression Learning)이라는 혁신적인 모델을 제안합니다. 이 모델은 Vision Transformers (ViTs)와 얼굴 기하학적 특성을 고려한 신뢰성 균형 모듈을 활용하여 얼굴 표정 인식(FEL)에서 발생하는 데이터 불균형 및 편향 문제를 해결합니다.

- **Technical Details**: GReFEL은 여러 단계의 주의 기반(feature extraction) 기능 추출과 신뢰성 균형 모듈을 포함하여 복잡한 감정 표시를 효과적으로 처리하는 것을 목표로 합니다. 이 모델은 공간에서 학습 가능한 앵커를 배치하여 다양한 표정 변화 간의 유사성을 측정하고, 다중 헤드의 자기 주의(multi-head self-attention) 메커니즘을 활용하여 중요한 기능을 식별합니다. 또한, 입력 이미지에 다양한 데이터 증강 기법을 적용하여 훈련 과정에서의 편향과 과적합을 방지합니다.

- **Performance Highlights**: GReFEL은 다양한 데이터셋에서 실행된 실험을 통해 현재의 최첨단 얼굴 표정 인식 시스템보다 일관되게 우수한 성능을 보였습니다. 이는 모델이 불균형한 데이터 분포, 편향 및 불확실성을 효과적으로 해결할 수 있도록 설계되어 있음을 의미합니다.



### Mitigating Object Hallucination via Concentric Causal Attention (https://arxiv.org/abs/2410.15926)
Comments:
          To appear at NeurIPS 2024. Code is available at this https URL

- **What's New**: 최근의 대형 비전 언어 모델(Large Vision Language Models, LVLMs)은 다중 모달 쿼리에 대해 놀라운 제로샷 대화 및 추론 능력을 보여주고 있습니다. 하지만 객체 환각(object hallucination) 문제에 시달리고 있습니다.

- **Technical Details**: 객체 환각은 LVLMs가 이미지 입력과 사실적으로 일치하지 않는 텍스트 응답을 생성하는 현상으로, 이는 일반적으로 사용되는 위치 인코딩 방식인 Rotary Position Encoding (RoPE)와 밀접한 관련이 있습니다. RoPE의 장기적 감소(long-term decay)로 인해 LVLMs는 시각적 단서가 명령어 토큰(instruction tokens)으로부터 멀어질 경우 환각이 발생할 가능성이 높아집니다. 또한 다중 모달 정렬(multimodal alignment) 과정에서 시각적 토큰의 순서를 뒤바꿀 때도 유사한 효과가 관찰되었습니다.

- **Performance Highlights**: 우리의 연구 결과에 따르면, Concentric Causal Attention (CCA)라는 단순하지만 효과적인 위치 정렬 전략이 RoPE의 장기적 감소로 인한 문제를 완화시킵니다. CCA를 통해 시각적 토큰과 명령어 토큰 간의 상대적 거리가 자연스럽게 감소하여 시각적 상호작용이 개선되고 객체 환각이 줄어듭니다. 기존의 환각 완화 전략보다 여러 객체 환각 벤치마크에서 큰 성과를 보였습니다.



### Are Large-scale Soft Labels Necessary for Large-scale Dataset Distillation? (https://arxiv.org/abs/2410.15919)
Comments:
          Accepted by Neurips 2024

- **What's New**: 이번 연구는 대규모 데이터 세트 증류에서 대규모 소프트 레이블의 필요성을 탐구합니다. 우리는 응집된 데이터 세트에서 클래스 내 유사성이 높아 소프트 레이블의 저장 요구량이 지나치게 증가한다는 점을 발견했습니다.

- **Technical Details**: 연구는 이미지 합성 과정에서 클래스 내 샘플을 배치하여 유사성을 줄이는 방법을 제안합니다. 즉, Label Pruning for Large-scale Distillation (LPLD) 기술을 통해 클래스별 감독을 도입하고, 기존의 데이터 증강 기술을 효율적으로 개선합니다.

- **Performance Highlights**: 이 방법을 사용하여 ImageNet-1K을 클래스당 200개의 이미지로 압축할 때 필요한 소프트 레이블의 크기를 113 GB에서 2.8 GB로 줄이면서 2.6%의 성능 향상을 달성했습니다.



### Leveraging CORAL-Correlation Consistency Network for Semi-Supervised Left Atrium MRI Segmentation (https://arxiv.org/abs/2410.15916)
Comments:
          5 pages, 3 figures, Accepted by 2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM 2024)

- **What's New**: 본 논문에서는 좌심방(Left Atrium) 분할을 위한 새로운 접근 방식인 CORAL-Correlation Consistency Network (CORN)를 제안합니다. 이 방법은 레이블이 있는 데이터와 없는 데이터 간의 분포 차이를 최소화하는 second-order 통계 정보를 활용하여 글로벌 구조와 로컬 세부 사항을 포착합니다.

- **Technical Details**: CORN은 dual parallel encoders를 이용하여 특성 추출 및 분할 결과 예측을 수행합니다. 또한 Dynamic Feature Pool (DFP) 모듈을 사용하여 낮은 신뢰도를 가진 특성을 걸러내고, 일관성을 유지하는 전략으로 모델을 정규화합니다. CORAL-Correlation Module (CCM)에서 레이블 데이터와 비레벨 데이터 간의 통계적 분포를 정렬합니다.

- **Performance Highlights**: CORN은 Left Atrium 데이터셋을 사용한 실험에서 기존 최첨단 반지도 학습 방법보다 우수한 성능을 보였으며, 특히 엣지 영역에서 더욱 강화된 세부 사항을 제공하며 지상 진실(Ground Truth)을 더 정확히 반영한 형상을 생성합니다.



### Hybrid Architecture for Real-Time Video Anomaly Detection: Integrating Spatial and Temporal Analysis (https://arxiv.org/abs/2410.15909)
- **What's New**: 본 연구에서는 비디오 데이터에서 실시간 이상 감지를 위한 새로운 아키텍처를 제안합니다. 이 아키텍처는 인간 행동에서 영감을 받았으며, 공간적(spatial) 및 시간적(temporal) 분석을 결합합니다.

- **Technical Details**: 이 접근 방식은 두 가지 모델을 사용합니다. 첫 번째는 VGG19와 GRU를 결합하여 비디오 시퀀스를 처리하는 순환 합성곱 신경망(CNN + RNN)입니다. 두 번째는 YOLOv7을 사용하여 개별 이미지를 분석하는 공간 분석(spatial analysis)입니다. 이 두 분석은 병렬(parallel)로 수행되거나, 일련의(serial)로 수행되어 공간 분석이 시간 분석을 보강할 수 있습니다.

- **Performance Highlights**: 제안된 하이브리드 아키텍처는 두 가지 구성(병렬 및 일련)을 비교하여 비디오 이상 감지의 효과성을 평가하고, 공간 분석의 통합이 아키텍처에 미치는 영향을 분석합니다.



### Foundation Models for Slide-level Cancer Subtyping in Digital Pathology (https://arxiv.org/abs/2410.15886)
Comments:
          Manuscript accepted for oral presentation at Decision Science Allieance -INternational Summer Conference (DSA-ISC) 2024 held on Valencia, Spain

- **What's New**: 이 논문은 ImageNet 데이터셋이 등장한 이후, 컴퓨터 비전에서 사전 훈련(pretraining) 및 미세 조정(fine-tuning) 접근법이 널리 활용되고 있다는 점을 밝힙니다. 특히, 디지털 병리학 분야와 같은 도메인 특화된 분야에 적응할 때 발생하는 주요 문제를 다룹니다.

- **Technical Details**: 본 연구에서는 기초 모델(foundational models)이 대규모 도메인 내 데이터셋에서 훈련되었으며, 이는 조직 병리학(histopathology) 이미지의 복잡한 특징을 학습하는 데 기여합니다. 논문은 전체 슬라이드 이미지(whole-slide image, WSI) 예측을 위해 다중 인스턴스 학습(multiple instance learning, MIL) 접근법을 사용했으며, 패치 수준(patch-level) 특징 집계를 통해 성능 비교를 수행했습니다.

- **Performance Highlights**: 연구 결과, 기초 모델이 ImageNet 사전 훈련 모델을 초월하여 여섯 가지 피부암 서브타입(cancer subtypes)의 예측 정확성을 개선하는 능력이 있음을 보여줍니다.



### MI-VisionShot: Few-shot adaptation of vision-language models for slide-level classification of histopathological images (https://arxiv.org/abs/2410.15881)
Comments:
          Manuscript accepted for oral presentation at KES-InnovationInMedicine 2024 held on Madeira, Portugal

- **What's New**: 본 연구에서는 MI-Zero의 고변동성 문제를 해결하기 위해 MI-VisionShot이라는 새로운 메소드를 제안합니다. 이 방법은 VLM(vission-language models) 위에서 훈련이 필요 없는 적응(adaptation) 방식으로 슬라이드 수준 레이블 예측을 지원합니다.

- **Technical Details**: MI-VisionShot은 프로토타입 학습(prototypical learning)을 기반으로 하여, 각 슬라이드에서 가장 판별력이 있는 패치를 검색하여 다중 인스턴스(multiple-instance) 설정 하에서 프로토타입 기반 분류기를 생성합니다. 이를 통해 더 나은 레이블 예측이 가능합니다.

- **Performance Highlights**: MI-VisionShot은 제로샷 전이(zero-shot transfer) 방식보다 낮은 변동성을 보이며, 적은 수의 샷(few-shot learning) 상황에서도 뛰어난 성능을 발휘합니다.



### Visual Motif Identification: Elaboration of a Curated Comparative Dataset and Classification Methods (https://arxiv.org/abs/2410.15866)
Comments:
          17 pages, 11 figures, one table, to be published in the conference proceedings of ECCV 2024

- **What's New**: 본 연구는 영화에서 시각적 모티프(visual motifs)를 자동으로 인식하고 분류하기 위한 새로운 머신러닝 모델을 제안합니다. 이 모델은 사용자 정의된 데이터셋을 활용하여 20가지 모티프를 분류하는 데 놀라운 결과를 나타냈습니다.

- **Technical Details**: CLIP 모델(Contrastive Language-Image Pre-training)에서 추출한 특징을 활용하고, 얕은 네트워크를 사용하여 모티프를 효과적으로 분류합니다. 본 연구에는 세 가지 주요 기술적인 구성 요소가 포함되어 있으며, 각 요소는 이미지의 특징을 추출하고 이를 기반으로 학습하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 테스트 세트에서 F1-점수(F1-score) 0.91을 기록하며, 단순한 분류 네트워크로도 효율적으로 모티프를 인식할 수 있음을 보여주었습니다. 이 연구는 다양한 매체와 시기의 이미지를 포함한 Curated Comparative Dataset을 기반으로 하여 모티프 분류의 정확성을 높였습니다.



### Random Token Fusion for Multi-View Medical Diagnosis (https://arxiv.org/abs/2410.15847)
Comments:
          Originally published at the NeurIPS 2024 Workshop on Advancements In Medical Foundation Models: Explainability, Robustness, Security, and Beyond (AIM-FM)

- **What's New**: 이 연구에서는 Random Token Fusion (RTF)이라는 새로운 기법을 소개하여 다중 시점(multi-view) 의료 이미지 분석에서의 성능을 향상시킵니다. RTF는 훈련 과정에서 특징 융합(feature fusion) 과정에 무작위성을 도입하여 과적합(overfitting) 문제를 해결하고 진단 모델의 강건성과 정확성을 높입니다.

- **Technical Details**: RTF는 다중 시점 의료 진단의 특징을 변형시키기 위해 훈련 단계에서 서로 다른 시점의 토큰(token)을 무작위로 융합합니다. 이렇게 하면 모델이 한 가지 우세한 시점에만 의존하는 것이 아니라, 서로 다른 시점의 정보를 다양하게 고려하게 됩니다. 이 프로세스는 기존의 비전 트랜스포머(vision transformers)와 통합되어 별도의 수정 없이도 적용할 수 있습니다.

- **Performance Highlights**: RTF는 CBIS-DDSM 및 CheXpert와 같은 표준 벤치마크 데이터셋에서 기존의 다중 시점 모델 성능을 일관되게 향상시킵니다. 이 연구는 유방 촬영술(mammograms) 및 흉부 X선(chest X-rays)에서 최첨단 성능을 달성하였으며, RTF의 출처 코드는 https://jimyug.github.io/RandomTokenFusion에서 확인 가능합니다.



### LiOn-XA: Unsupervised Domain Adaptation via LiDAR-Only Cross-Modal Adversarial Training (https://arxiv.org/abs/2410.15833)
Comments:
          Preprint, Paper has been accepted at IROS2024

- **What's New**: LiOn-XA는 LiDAR 전용 크로스 모달 학습과 적대적 훈련을 결합한 새로운 비지도 도메인 적응(UDA) 접근법으로, 환경과 센서 설정의 변화로 인한 도메인 격차를 메우는 데 초점을 맞추고 있습니다. 이 방법은 RGB 이미지 데이터가 없을 때의 상황을 다룹니다.

- **Technical Details**: LiOn-XA는 3D 점 구름을 보존하는 복셀(point cloud) 데이터와 객체 방향 및 표면에 대한 정보를 제공하는 2D 범위 이미지(two-dimensional range image)를 활용합니다. 두 도메인 간의 피처 공간(feature space)을 정렬하기 위해 2D 및 3D 신경망의 예측과 특징을 통한 적대적 훈련을 적용합니다.

- **Performance Highlights**: 실험 결과, LiOn-XA는 이전의 단일 및 다중 모달(UDA) 방법들과 비교하여 새로운 최첨단 성능을 달성하였습니다. 이는 3개의 실제 환경에서의 적응 시나리오에서 입증되었습니다.



### Kaninfradet3D:A Road-side Camera-LiDAR Fusion 3D Perception Model based on Nonlinear Feature Extraction and Intrinsic Correlation (https://arxiv.org/abs/2410.15814)
- **What's New**: 이 논문에서는 도로변 인식에 대한 연구의 필요성을 강조하며, Kolmogorov-Arnold Networks (KANs)를 이용한 Kaninfradet3D 모델을 제안하여 카메라와 LiDAR 간의 효과적인 데이터 융합을 통한 3D 객체 인식 성능을 향상시킵니다.

- **Technical Details**: Kaninfradet3D는 복잡한 고차원 데이터를 효과적으로 처리하기 위해 KAN Layer가 적용된 인코더와 퓨저 모듈을 개선하였습니다. 교차 주의 메커니즘(Cross-attention)을 통해 카메라와 LiDAR의 상호 의존성을 포착하여 정보 융합의 질을 높였습니다.

- **Performance Highlights**: 제안된 모델은 TUMTraf Intersection Dataset에서 두 관점에서 +9.87 mAP 및 +10.64 mAP의 성능 향상을 보여주었으며, TUMTraf V2X Cooperative Perception Dataset의 도로변 끝 부분에서 +1.40 mAP의 향상을 달성했습니다.



### Data-Efficient CLIP-Powered Dual-Branch Networks for Source-Free Unsupervised Domain Adaptation (https://arxiv.org/abs/2410.15811)
- **What's New**: 본 논문에서는 라벨이 없는 타겟 도메인으로의 효율적인 전이를 위해 CLIP 기반의 데이터 효율적인 이중 분기 네트워크(CDBN)를 제안합니다. 이 접근 방식은 데이터 프라이버시와 라벨 부족 문제를 동시에 해결합니다.

- **Technical Details**: CDBN은 크로스-모달 이중 분기 구조로, 소스 도메인 클래스의 의미를 활용하여 라벨이 없는 타겟 도메인의 훈련을 최적화합니다. 이 구조는 분류 정확도, 다양한 표현을 유지하기 위한 무감독 최적화 전략을 포함하여 세 가지 핵심 손실 함수를 사용합니다: 교차 엔트로피 손실, 다중 관점 일관성 손실, 상호 정보 최대화 손실.

- **Performance Highlights**: 31개의 전이 작업에 대한 실험을 통해, CDBN은 기존 방법들에 비해 최첨단 성능을 달성했습니다. 제한된 소스 샘플과 소스 도메인 데이터 접근 제한에도 불구하고, CDBN은 여러 데이터 세트에서 경쟁력 있는 결과를 보였습니다.



### Habaek: High-performance water segmentation through dataset expansion and inductive bias optimization (https://arxiv.org/abs/2410.15794)
- **What's New**: 본 연구는 SegFormer 모델을 데이터 증강(data augmentation)을 통해 개선하여 실시간 물 segmentation을 위한 실용성과 정확성을 높이고자 합니다. 이는 NVIDIA의 고해상도 이미지를 활용한 기존의 flood monitoring 방법들을 대체할 수 있는 potential을 보여줍니다.

- **Technical Details**: SegFormer는 계층적 Transformer 인코더(hierarchical Transformer encoder)와 경량 MLP 디코더(lightweight all-MLP decoder)로 구성됩니다. 이 모델은 이미지의 다중 스케일 특징을 추출하며, 4x4 패치(patches)를 사용하여 segmentation 작업에 적합하게 설계되었습니다. 또한, LoRA(저랭크 적응, Low-Rank Adaptation) 기법을 활용하여 처리 복잡성을 낮추는 동시에 정확도를 유지합니다.

- **Performance Highlights**: 제안된 Habaek 모델은 IoU(Intersection over Union) 점수 0.91986에서 0.94397까지 성능을 보이며, F1-score, recall, accuracy 및 precision에서도 경쟁 모델보다 우수한 결과를 나타냈습니다. 이는 실제 어플리케이션에서의 활용 가능성을 시사합니다.



### WildOcc: A Benchmark for Off-Road 3D Semantic Occupancy Prediction (https://arxiv.org/abs/2410.15792)
- **What's New**: 이 논문에서는 야외(off-road) 환경에서의 3D semantic occupancy prediction을 위한 최초의 벤치마크인 WildOcc를 소개합니다. 기존의 연구는 주로 도로(on-road) 환경에 집중되어 있었으며, 야외 환경에 적합한 데이터셋과 기준이 부족했습니다. 따라서 이 연구는 중요한 공백을 메우고 있습니다.

- **Technical Details**: WildOcc는 dense occupancy annotations을 제공하는 야외 3D semantic occupancy prediction을 위한 벤치마크입니다. 이 연구는 coarse-to-fine reconstruction 방식을 사용하는 ground truth generation pipeline을 제안하며, 이를 통해 더 현실적인 결과를 도출합니다. 또한, multi-modal 3D semantic occupancy prediction framework인 OFFOcc를 도입하여 multi-frame 이미지와 point cloud에서 spatio-temporal 정보를 voxel level에서 융합합니다. 이 과정에서 cross-modality distillation 기능이 도입되어 point cloud에서 이미지 특징으로 기하학적 지식을 전이합니다.

- **Performance Highlights**: 실험 결과, 제안한 OFFOcc 프레임워크는 야외 3D semantic occupancy prediction 작업에서 높은 성능을 달성했습니다. 이는 기존의 도로 기반 방법들이 아닌, 화려한 지형과 불규칙한 물체들이 많은 야외 환경에서도 효과적으로 작동할 수 있음을 시사합니다.



### An Efficient System for Automatic Map Storytelling -- A Case Study on Historical Maps (https://arxiv.org/abs/2410.15780)
- **What's New**: 본 논문에서는 역사적 지도에 대한 캡션 생성의 새로운 접근 방식을 제안합니다. 기존의 이미지 캡셔닝(image captioning) 방법은 자연 이미지에서는 성공적이지만, 지도에 대해서는 성능이 떨어지는 문제를 해결하고자 합니다.

- **Technical Details**: 이 연구에서는 최신의 vision-language 모델인 CLIP을 미세 조정(fine-tune)하여 역사적 지도와 관련된 캡션을 생성하며, GPT-3.5를 활용하여 지도에 대한 간략한 이야기를 제공합니다. 또한, 특정 지도 유형에 맞춰 캡션을 생성할 수 있도록 하는 새로운 결정 트리(decision tree) 구조를 제안합니다.

- **Performance Highlights**: 우리 시스템은 지도에서의 텍스트 변경에 대해 불변성을 보이며, 다른 지도 유형에 쉽게 적용되고 확장될 수 있는 가능성을 가지고 있습니다. 이번 연구에서 공개된 코드는 대규모 지도 캡셔닝 시스템으로 확장될 수 있는 기반을 제공합니다.



### Reducing Hallucinations in Vision-Language Models via Latent Space Steering (https://arxiv.org/abs/2410.15778)
Comments:
          21 pages

- **What's New**: 이번 연구에서는 Large Vision-Language Models (LVLMs)에서 발생하는 hallucination의 메커니즘을 조사하고, 이를 완화하기 위한 Visual and Textual Intervention (VTI)라는 새로운 기술을 제안합니다. VTI는 latent space representation을 조정하여 시각 정보의 안정성을 향상시키는데 중점을 두고 있습니다.

- **Technical Details**: LVLMs는 이미지 인코더와 텍스트 디코더의 비대칭적 사전 훈련으로 인해 hallucination에 민감합니다. 본 연구는 이미지 인코더의 불안정성이 텍스트 디코더의 민감도에 어떤 영향을 미치는지에 대해 분석하였고, VTI를 통해 이를 해결하기 위한 접근법을 제안하였습니다. VTI는 query 이미지의 perturbation에 따라 latent features를 수정하여 작업에 관계없이 적용할 수 있습니다.

- **Performance Highlights**: VTI는 다양한 매트릭스를 통해 기존의 방법들과 비교했을 때 hallucination을 효과적으로 줄일 수 있다는 것을 입증하였으며, LVLMs에서 시각 정보의 안정성이 얼마나 중요한지를 강조합니다.



### Learning to Synthesize Graphics Programs for Geometric Artworks (https://arxiv.org/abs/2410.15768)
Comments:
          ICPR 2024

- **What's New**: 이 연구에서는 그림 생성 과정에서 최종 이미지를 단순히 반환하는 것이 아닌, 디지털 드로잉 도구를 실행 가능한 프로그램 세트로 취급하는 방법을 제시합니다. 이를 통해 사용자는 구체적인 드로잉 명령어를 기반으로 이미지를 재구성할 수 있습니다.

- **Technical Details**: Art2Prog라는 프로그램 합성기를 통해 복잡한 이미지의 입력을 이해하고 해석하는 과정에서, 색상 블렌딩 모드 및 레이어 중첩 등을 관리하는 2D 그래픽 프로그램을 생성합니다. 이를 위해 그래픽 프로그램을 계층적으로 구조화하고, 명확한 색상 혼합 방법을 채택하였습니다.

- **Performance Highlights**: 실험 결과, Art2Prog는 기존의 최첨단 벡터화 기법보다 뛰어난 재구성 정확도를 보여주었으며, 복잡한 그래픽을 표현하는 데 있어 더욱 많은 세부 정보를 담을 수 있습니다.



### Improving Instance Optimization in Deformable Image Registration with Gradient Projection (https://arxiv.org/abs/2410.15767)
Comments:
          L2R 2024 Challenge Paper

- **What's New**: 본 논문에서는 다물체 최적화(multi-objective optimization, MOO) 문제를 해결하기 위한 새로운 인스턴스 최적화(instance optimization, IO) 알고리즘을 소개합니다. 이 알고리즘은 딥 러닝의 일반화 능력과 인스턴스 구체적인 최적화의 미세 조정 이점을 결합합니다.

- **Technical Details**: 우리는 MOO에서 발생하는 갈등하는 업데이트를 완화하기 위해 gradient projection 기법을 강조합니다. 이 기법은 갈등하는 경량 채우기를 공통 공간으로 투영하여 두 가지 목표를 더 잘 정렬하고 최적화의 안정성을 향상시킵니다. 실험을 통해 경량 픽셀 간의 정합성 측정에 있어 LNCC(local normalized cross-correlation)와 가우시안 커널을 사용합니다.

- **Performance Highlights**: LUMIR(Learn2Reg 2024 Challenge) 과제에서 최신 기초 모델을 사용하여 method를 평가한 결과, 기존의 gradient descent 방법에 비해 상당한 성능 향상을 보여주었습니다. 우리 방법은 더 정확하고 신뢰할 수 있는 등록 결과를 이끌어냅니다.



### How Important are Data Augmentations to Close the Domain Gap for Object Detection in Orbit? (https://arxiv.org/abs/2410.15766)
- **What's New**: 이 연구에서는 우주에서의 데이터 증가(data augmentation) 기법을 통해 도메인 갭(domain gap)을 해소하는 방안을 조사했습니다. 특히, 이는 자율 운영에서 중요한 역할을 하며, 기존 알고리즘과 현실 데이터 간의 성능 차이를 극복하기 위한 새로운 두 가지 데이터 증강 기법을 제안합니다.

- **Technical Details**: 연구에서는 2D 객체 탐지(2D object detection)을 주요 과제로 설정하고, SPEED+ 데이터셋을 사용하여 모델을 평가했습니다. Mask R-CNN, Faster R-CNN, YOLO-v7, GroundingDINO와 같은 다양한 객체 탐지기를 비교하고, 각 탐지기에서의 성능, 추론 속도(inference speed), 훈련 시간(training time) 간의 균형을 조명했습니다. 또한, Hyperparameter optimization 프레임워크인 Optuna를 활용해 수백 가지 조합을 샘플링하였습니다.

- **Performance Highlights**: 데이터 증가 기법의 적용을 통해 모델의 성능이 현저하게 개선되었고, 극한의 우주 환경에서도 견고함과 신뢰성을 확보했습니다. 제안된 두 가지 새로운 데이터 증강 기법은 궤도 이미지에서 관찰되는 시각적 효과를 모사하도록 개발되었으며, 자율 시스템의 컴퓨터 비전 향상에 중요한 기여를 할 것으로 기대됩니다.



### DeepIcon: A Hierarchical Network for Layer-wise Icon Vectorization (https://arxiv.org/abs/2410.15760)
Comments:
          Accepted as Oral Presentation at DICTA 2024

- **What's New**: DeepIcon은 이미지 벡터화를 수행하기 위해 새롭게 설계된 계층적 이미지 벡터화 네트워크로, raster 이미지 입력을 바탕으로 가변 길이의 아이콘 벡터 그래픽을 생성하도록 특화되어 있습니다. 기존의 이미지 벡터화 방법들이 겪었던 문제들을 해결합니다.

- **Technical Details**: DeepIcon은 Scalable Vector Graphics (SVG)를 직접 생성하며, differentiable rasterizer에 의존하지 않습니다. 이는 고급 이미지 인식 및 처리 방식을 채택하여, 여러 개의 경로를 포함하는 SVG를 생성할 수 있는 정확한 SVG 디코더를 제안합니다. 또한, CLIP 기반의 이미지 인코더를 사용합니다.

- **Performance Highlights**: 실험 결과에 따르면 DeepIcon은 기존 최첨단 벡터화 접근 방식들을 넘어, 토폴로지 유사성 및 기하학적 정확성을 보존하는 데 있어 뛰어난 성능을 발휘합니다.



### Unleashing the Potential of Vision-Language Pre-Training for 3D Zero-Shot Lesion Segmentation via Mask-Attribute Alignmen (https://arxiv.org/abs/2410.15744)
- **What's New**: 최근 의료 영상-언어 사전 훈련 모델의 발전으로 제로샷 질병 인식(zero-shot disease recognition) 분야에서 큰 진전을 이룬 반면, 이미지 수준 지식을 픽셀 수준 작업인 3D CT 스캔의 병변 분할(lesion segmentation)으로 이전하는 것은 여전히 중요한 도전 과제로 남아있습니다. 본 논문에서는 3D 제로샷 병변 분할을 위해 특별히 설계된 새로운 다중 스케일 병변 수준 마스크-속성 정렬(framework)을 제안하는 Malenia를 소개합니다.

- **Technical Details**: Malenia는 다중 스케일 마스크 표현을 활용하여 다양한 병변 영역을 포착하고, 병변의 세밀한 시각적 특징을 텍스트 임베딩과 매칭하여 대조적 사전 훈련 작업과 픽셀 수준 밀집 예측(dense prediction) 작업 간의 간극을 효과적으로 연결합니다. 또한, 우리는 시각적 및 텍스트적 특징을 상호 보완적인 정보로 강화하는 Cross-Modal Knowledge Injection(CMKI) 모듈을 설계하여 분할 결과 생성을 효과적으로 안내합니다.

- **Performance Highlights**: Malenia는 MSD, KiTS23, 그리고 실제 사례에서 수집된 데이터셋에서 12가지 병변 카테고리에 대해 종합적으로 평가하였으며, 제로샷 설정에서 가장 최신의 방법들과 비교하여 일관되게 우수한 성능을 보였습니다. 특히 Malenia는 기존 방법들보다 월등히 더 나은 성능을 확인시켜 주었습니다.



### ViMoE: An Empirical Study of Designing Vision Mixture-of-Experts (https://arxiv.org/abs/2410.15732)
- **What's New**: 이 논문에서는 Mixture-of-Experts (MoE) 구조를 고전적인 Vision Transformer (ViT)에 통합하여 새로운 모델인 ViMoE를 소개합니다. 이를 통해 이미지 분류 작업에서 MoE의 잠재력을 탐구하고, MoE 레이어 구성의 민감성 문제를 해결하기 위해 공유 전문가를 도입하였습니다.

- **Technical Details**: ViMoE는 각 블록의 Feed-Forward Networks (FFNs)를 여러 전문가로 대체하여 클래스 구조를 개선하고, 게이팅 네트워크를 사용해 입력 샘플을 최적의 전문가에게 라우팅합니다. 특히, 전문가의 라우팅 동작 분석을 통해 깊은 ViT 블록에서 클래스 샘플을 특정 전문가에 효과적으로 할당하는 것을 보여줍니다.

- **Performance Highlights**: ViMoE는 DINOv2를 ImageNet-1K 데이터셋에서 1.1% 초과하여 성능을 향상시켰습니다. 활성화된 매개변수의 3분의 1 이하로도 여러 고급 ViT-B/16 모델을 초과하여 효율성을 증명하였습니다.



### Object-Centric Temporal Consistency via Conditional Autoregressive Inductive Biases (https://arxiv.org/abs/2410.15728)
- **What's New**: 본 논문에서는 비디오에서 객체 중심(object-centric) 표현의 일관성을 높이기 위한 새로운 접근법인 Conditional Autoregressive Slot Attention (CA-SA)을 제안합니다. 기존의 슬롯 기반 표현은 비디오의 연속 프레임 간에 시간 일관성을 유지하는 데 실패했으나, CA-SA는 이전 타임스텝에서 예측된 슬롯 표현을 바탕으로 현재 슬롯을 조정하여 이러한 문제를 해결합니다.

- **Technical Details**: CA-SA는 두 가지 주된 요소로 구성됩니다. 첫째, 이전 타임스텝의 슬롯 표현을 기반으로 현재 슬롯 표현을 예측하는 자회귀(autoregressive) 네트워크입니다. 둘째, 두 연속 프레임 간의 슬롯들에 대한 일관성 손실 함수를 사용하여 시간적 일관성을 부여합니다. 이 방법은 기존의 모델에 조건을 부여하고 시간을 축으로 한 점유할당(slot attention) 매핑의 일관성을 보장합니다.

- **Performance Highlights**: CLEVRER 및 Physion 데이터셋에 대한 평가 결과에서 CA-SA는 비디오 예측 및 시각적 질문-응답(visual question-answering)과 같은 다양한 다운스트림(다운스트림 tasks) 작업에서 기존 기준선들을 초월하는 성능을 나타냅니다.



### Students Rather Than Experts: A New AI For Education Pipeline To Model More Human-Like And Personalised Early Adolescences (https://arxiv.org/abs/2410.15701)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)을 활용하여 가상 학생 에이전트(Virtual Student Agents)를 개발하는 가능성을 탐색하고 있습니다. 특히, 교육 환경에서 LLM을 이용한 교수 시뮬레이션 연구가 많은 반면, 학생 시뮬레이션은 부족했던 문제를 해결하고자 합니다.

- **Technical Details**: 이 연구는 언어 학습을 가상 학생 에이전트를 모델링하는 맥락으로 선택하고, SOE (Scene-Object-Evaluation)라는 새로운 AI4Education 프레임워크를 제안합니다. 또한, 다양한 성격 특성, 질문 유형, 학습 단계를 갖춘 개인화된 교사-학생 상호작용 데이터셋을 큐레이션하여 LLM을 LoRA 방식으로 미세 조정(fine-tuning)합니다.

- **Performance Highlights**: 실험 결과, 인간 평가자와 GPT-4의 LVSA(LLM 기반 가상 학생 에이전트) 진위도 판단 간의 강한 상관관계가 입증되었습니다. 이는 LLM이 교육적 맥락에서 인간과 유사한 개인화된 가상 학생 에이전트를 생성할 수 있음을 확인시켜주며, 향후 예비 교사 훈련 및 다중 에이전트 시뮬레이션 환경에서의 적용 가능성을 제시합니다.



### Enhancing SNN-based Spatio-Temporal Learning: A Benchmark Dataset and Cross-Modality Attention Mod (https://arxiv.org/abs/2410.15689)
- **What's New**: 이번 연구에서는 Spiking Neural Networks (SNNs)의 공간-시간 표현 특성을 최적으로 활용할 수 있는 새로운 신경형 데이터셋인 DVS-SLR을 소개합니다. 또한 Cross-Modality Attention (CMA) 기반의 융합 방법을 제안하여 이벤트와 프레임 데이터 간의 시너지를 증가시킵니다.

- **Technical Details**: DVS-SLR 데이터셋은 21개의 수화 행동 세그먼트를 포함하고 있으며, 43명의 참가자들이 다양한 조명 및 위치 조건 하에서 수집되었습니다. 이 데이터셋은 높은 시간 상관관계와 다양한 시나리오를 제공하며, CMA 모듈을 통해 각각의 모드에서 시간 및 공간 주의 점수를 학습할 수 있게 합니다.

- **Performance Highlights**: 실험 결과, 제안한 CMA 모듈은 인식 정확도를 향상시키며 다양한 조건에서도 강인성을 유지하는 것을 보여주었습니다. 시간 상관관계가 높은 DVS-SLR 데이터셋은 SNN 알고리즘의 효과적인 비교와 새로운 연구 방향을 제시하는 기초를 제공합니다.



### RANSAC Back to SOTA: A Two-stage Consensus Filtering for Real-time 3D Registration (https://arxiv.org/abs/2410.15682)
Comments:
          8 pages, 8 figures

- **What's New**: 이 논문에서는 RANSAC의 한계를 극복하기 위해 두 단계의 합의 필터링 기법(두 단계 합의 필터링, TCF)을 제안합니다. 이 방법은 RANSAC의 속도와 정확도를 획기적으로 향상시키는 것을 목표로 합니다.

- **Technical Details**: 제안된 방법은 1포인트 RANSAC을 사용하여 길이 일관성 기반으로 합의 집합을 만들고, 2포인트 RANSAC을 통해 각도를 확인하여 이를 정제합니다. 마지막으로 3포인트 RANSAC이 변환된 대응의 거리 기반으로 coarse pose를 계산하고 아웃라이어를 제거합니다. 최적의 자세는 반복적인 재가중치 최소제곱(Iterative Reweighted Least Squares, IRLS)을 통해 도출됩니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 대규모 KITTI 및 ETH 데이터세트에서 MAC에 비해 최대 3배의 속도 향상을 달성하며, 등록 정확도와 리콜을 유지합니다.



### TALoS: Enhancing Semantic Scene Completion via Test-time Adaptation on the Line of Sigh (https://arxiv.org/abs/2410.15674)
Comments:
          Accepted at NeurIPS 2024. Code is available at this https URL

- **What's New**: 이 논문은 주행 환경에서 이용 가능한 정보를 활용하여 Semantic Scene Completion(SSC)을 위한 테스트 타임 적응 방법(TALoS)을 소개합니다. TALoS는 LiDAR 센서의 특성을 활용하여 특정 시점에 관측된 정보를 다른 시점의 장면 완성을 위한 Ground Truth(GT)로 사용합니다.

- **Technical Details**: TALoS는 자가 감독(self-supervision)을 통해 점유 여부 및 비어 있는 공간에 대한 정보를 수집하고, 다중 순간의 신뢰할 수 있는 SSC 예측치를 집계하여 의미적 의사 GT를 생성합니다. 또한 미래 관측치를 활용하기 위한 이중 최적화 기법을 통해 현재 업데이트 시점에서 접근할 수 없는 정보를 지속적으로 모델에 통합할 수 있습니다.

- **Performance Highlights**: SemanticKITTI 검증 및 테스트 세트에서 TALoS는 사전 훈련된 SSC 모델의 성능을 크게 향상시켰으며, 기하학적 완성 및 의미적 분할 성능 모두에서 뛰어난 결과를 보여주었습니다.



### CL-HOI: Cross-Level Human-Object Interaction Distillation from Vision Large Language Models (https://arxiv.org/abs/2410.15657)
- **What's New**: 본 논문에서는 Vision Language Models (VLMs)와 Vision Large Language Models (VLLMs)의 한계를 극복하기 위한 Cross-Level HOI distillation (CL-HOI) 프레임워크를 제안합니다. 이 방법은 수동 주석 없이 VLLMs의 이미지 수준 이해로부터 인스턴스 수준 HOI를 증류합니다.

- **Technical Details**: CL-HOI 프레임워크는 두 단계로 이루어집니다: 첫 번째는 context distillation으로, Visual Linguistic Translator (VLT)는 시각 정보를 언어 형태로 변환합니다. 두 번째는 interaction distillation로, Interaction Cognition Network (ICN)가 공간적, 시각적 및 맥락 관계를 학습하고 분석합니다. 이를 통해 이미지 수준의 지식이 인스턴스 수준 HOI 탐지에 효과적으로 전달됩니다. 또한, contrastive distillation loss를 도입하여 이미지 수준의 컨텍스트와 상호작용 지식을 전이합니다.

- **Performance Highlights**: CL-HOI는 HICO-DET 데이터셋에서 17.5% mAP, V-COCO 데이터셋에서 36.63% Role AP를 기록하며 기존의 약한 감독 방식 및 VLLM 감독 방식을 능가하는 성능을 보였습니다.



### LucidFusion: Generating 3D Gaussians with Arbitrary Unposed Images (https://arxiv.org/abs/2410.15636)
Comments:
          17 pages, 12 figures, [project page](this https URL)

- **What's New**: LucidFusion은 상대 좌표 맵(Relative Coordinate Map, RCM)을 활용하여 다양한 시점에서의 3D 객체 생성을 개선하는 유연한 피드포워드(flexible end-to-end feed-forward) 프레임워크입니다. 기존 방법과는 달리, LucidFusion은 특정 카메라 포즈 정보 없이도 임의의 이미지에서 3D 객체를 생성할 수 있는 가능성을 제공합니다.

- **Technical Details**: LucidFusion은 두 가지 주요 단계를 통해 작동합니다. 첫 번째 단계는 입력 이미지를 RCM에 매핑하고 픽셀 정렬된 포인트 클라우드 표현을 생성하는 것입니다. 두 번째 단계에서는 3D Gaussians를 사용하여 포인트 클라우드를 다듬어 시각적 충실도와 객체 세부 정보를 향상시킵니다. 이 구조는 고유한 참조 프레임으로 다양한 시점의 기하학적 특징을 일관되게 정렬할 수 있도록 합니다.

- **Performance Highlights**: LucidFusion은 512×512 해상도에서 초당 13프레임(FPS)으로 지형적 일관성과 높은 시각적 품질을 달성하는 3D 객체 생성을 제공합니다. 또한 단일 관점에서 다중 보기 확산 모델(multi-view diffusion models)을 활용하여 3D 객체 생성을 가능하게 해줍니다.



### Fully Explicit Dynamic Gaussian Splatting (https://arxiv.org/abs/2410.15629)
Comments:
          Accepted at NeurIPS 2024

- **What's New**: 이번 논문에서는 동적 장면에 대한 4D Gaussian Splatting(Ex4DGS) 기법을 제안합니다. 이 방식은 정적과 동적 Gaussian을 분리하여 훈련하고, 동적 Gaussian의 위치와 회전을 희소한 타임스탬프에서 명시적으로 샘플링하는 것이 주요 아이디어입니다.

- **Technical Details**: Ex4DGS는 개별 타임스탬프에 따라 동적 객체의 위치와 회전을 보간하여 공간적으로 및 시간적으로 연속적인 동작을 표현합니다. Polynomial Basis Interpolator, Slerp(구면 선형 보간법), 단순화된 Gaussian 혼합 모델을 사용하여 위치, 회전 및 불투명도를 최적화합니다. 동적 포인트와 정적 포인트를 분리하여 훈련하는 점진적 훈련 기법과 포인트 백추적(point-backtracking) 기법을 통해 성능을 향상시킵니다.

- **Performance Highlights**: Ex4DGS는 실험을 통해 다양한 장면에서 62 fps의 빠른 렌더링 속도를 보여주었으며, 단일 2080Ti GPU에서 1352×1014 해상도의 300 프레임 장면을 처리할 수 있습니다. 압축된 스토리지 및 메모리 요구사항으로 인해 기존 기법보다 효율적인 성능을 발휘합니다.



### Joint Top-Down and Bottom-Up Frameworks for 3D Visual Grounding (https://arxiv.org/abs/2410.15615)
Comments:
          Accepted by ICPR2024

- **What's New**: 본 논문은 3D 시각적 기초(3D visual grounding) 문제를 해결하기 위해 새로운 접근 방식을 제안합니다. 상향식(top-down) 및 하향식(bottom-up) 방법의 장점을 결합하여효율성과 성능을 개선하는 공동 프레임워크를 개발하였습니다.

- **Technical Details**: 제안된 방법은 두 단계로 나뉩니다. 첫 번째 단계에서는 경량 신경망을 활용하여 여러 개의 독립적인 객체 제안을 효율적으로 생성하고 군집화하는 하향식 기반 제안 생성 모듈을 구현합니다. 두 번째 단계에서는 그래프 설계를 활용하여 생성된 제안 간의 쿼리 관련 객체 문맥을 집계 및 전파하는 하향식 기반 제안 통합 모듈을 도입합니다.

- **Performance Highlights**: 실험 결과, ScanRefer 벤치마크에서 본 연구의 프레임워크가 최신 기술의 성능을 달성할 수 있음을 보였습니다. 이 방법은 기존 방식에 비해 향상된 성능을 기록하였습니다.



### Exploring Stronger Transformer Representation Learning for Occluded Person Re-Identificatio (https://arxiv.org/abs/2410.15613)
- **What's New**: 본 논문에서는 SSSC-TransReID라는 새로운 transformer 기반의 사람 재식별 프레임워크를 제안하였습니다. Self-supervised contrastive learning branch를 설계하여 음성 샘플이나 추가적인 사전 훈련 없이도 특징 표현을 강화할 수 있습니다.

- **Technical Details**: SSSC-TransReID는 새로운 random rectangle mask 전략을 채택하여 실제 장면에서의 occlusion을 시뮬레이션하고, 이를 통해 robust한 feature representation을 학습합니다. 또한, supervised learning과 self-supervised contrastive learning을 결합한 joint-training loss function을 사용합니다.

- **Performance Highlights**: 여러 벤치마크 데이터셋에서 실시한 실험 결과, 제안된 모델은 평균 평균 정확도(mAP) 및 Rank-1 정확도에서 최신의 ReID 방법들보다 우수한 성능을 보여주었습니다.



### Deep Active Learning with Manifold-preserving Trajectory Sampling (https://arxiv.org/abs/2410.15605)
- **What's New**: 본 논문에서는 Active Learning (AL)에서 라벨이 없는 데이터를 선택하기 위해 Manifold-Preserving Trajectory Sampling (MPTS)라는 새로운 방법론을 제안합니다. 이는 모델이 라벨이 있는 데이터가 학습한 feature 공간을 통해 더 정확한 manifold를 표현하도록 하는 데 중점을 둡니다.

- **Technical Details**: MPTS는 labeled 예제에서 학습한 feature distribution을 정규화함으로써, 라벨이 있는 데이터로 인한 bias를 효과적으로 보정합니다. 또한, 최적화 경로에서 의미 있는 파라미터 샘플링을 위해 MMD (Maximum Mean Discrepancies)를 사용하여 여러 지점에서 모델 파라미터를 평균화합니다. 이에 따라 모델 파라미터는 posterior distribution의 다양한 모드를 포착할 수 있습니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋에서 MPTS는 기존의 최신 Active Learning 방법보다 항상 우수한 성능을 보입니다. 이러한 결과는 MPTS의 효과성을 강조하며, 다양한 유형의 데이터에 적용 가능한 신뢰할 수 있는 방법임을 시사합니다.



### Deep Learning and Machine Learning -- Object Detection and Semantic Segmentation: From Theory to Applications (https://arxiv.org/abs/2410.15584)
Comments:
          167 pages

- **What's New**: 이 책은 object detection 및 semantic segmentation에 대한 심층적인 탐구를 제공합니다. 이론적 기초와 실용적 응용을 결합하여 최신 machine learning과 deep learning의 발전을 다루고 있으며, 특히 convolutional neural networks (CNNs), YOLO 아키텍처, 그리고 DETR와 같은 transformer 기반 접근 방식에 중점을 두고 있습니다.

- **Technical Details**: 이 책은 인공지능(AI) 기술과 대규모 언어 모델을 통합하여 복잡한 환경에서 object detection을 향상시키는 방법도 다룹니다. 또한 데이터 처리, 모델 최적화, 성능 평가 메트릭의 중요성을 강조하며, 기존 방법과 현대 deep learning 프레임워크 간의 간극을 해소하는 포괄적인 가이드를 제공합니다.

- **Performance Highlights**: 이 책은 연구자, 데이터 과학자 및 엔지니어가 대규모 object detection 작업에서 AI 기반 방법론을 활용할 수 있도록 돕는 자료를 제공하며, 실제 사례를 통한 데이터 분석 기법의 중요성을 알리고 있습니다.



### ARTS: Semi-Analytical Regressor using Disentangled Skeletal Representations for Human Mesh Recovery from Videos (https://arxiv.org/abs/2410.15582)
Comments:
          Accepted by ACM MM 2024. Project page: this https URL

- **What's New**: 이 논문에서는 비디오에서의 3D 인간 메쉬 복구에서 인간 포즈와 형태를 정확하게 추정하기 위해 새로운 방법인 ARTS(Analytical Regressor using disEnTangled Skeletal representations)를 제안합니다.

- **Technical Details**: 제안된 ARTS는 두 부분으로 나뉩니다: 1) 비디오에서 3D 스켈레톤을 추정하고 그것을 분리된 스켈레톤 표현으로 분해하는 모듈(Disentanglement Module)과 2) 이 표현을 기반으로 SMPL 파라미터를 회귀하는 반분석적 회귀기(Regression). ARTS는 Temporal Inverse Kinematics (TIK), Bone-guided Shape Fitting (BSF), Motion-Centric Refinement (MCR) 세 가지 모듈로 구성되어 있습니다.

- **Performance Highlights**: ARTS는 3DPW, MPI-INF-3DHP 및 Human3.6M 등의 인기 벤치마크에서 기존의 비디오 기반 방법들을 초월하여 프레임 당 정확도 및 시간적 일관성을 향상시킵니다. 특히, 크로스 데이터셋 평가에서 ARTS는 3DPW 데이터셋에서 MPJPE와 MPVPE를 각각 10.7%와 10.5% 감소시켰습니다.



### Multimodal Learning for Embryo Viability Prediction in Clinical IVF (https://arxiv.org/abs/2410.15581)
Comments:
          Accepted to MICCAI 2024

- **What's New**: 이 연구는 임상 In-Vitro Fertilization (IVF)에서 가장 생명력이 높은 배아를 예측하기 위한 새로운 다중모달 모델을 제안합니다. 이 모델은 시간 경과에 따른 비디오 데이터와 전자 건강 기록(EHR)을 결합하여, 배아의 생존 가능성을 보다 정확하게 평가할 수 있습니다. 특히, 기존의 수동 분석에서 자동화된 접근 방식으로 전환하여, 주관적 판단의 변동성을 줄이는 것을 목표로 하고 있습니다.

- **Technical Details**: 제안된 다중모달 모델은 두 가지 주요 접근 방식을 사용합니다. 첫 번째는 EHR과 비디오 데이터를 end-to-end로 처리하는 transformer 기반 모델입니다. 두 번째는 비디오 데이터를 사용하여 morphological features를 추출한 후 이를 EHR와 조합하는 두 단계 접근 방식입니다. 연구에서는 각 모달 유형의 효과적인 통합 방법을 탐색하였으며, 이를 통해 자동화된 배아 생존 가능성 예측을 위한 나아갈 방향을 제시합니다.

- **Performance Highlights**: 총 3,695개의 IVF 치료 주기에서 24,027개의 배아에 대한 데이터가 수집되었으며, 이 모델은 약 6백만 개의 이미지와 전자 건강 기록을 사용하여 예측 작업을 수행했습니다. 다중모달 데이터 접근 방식의 효과를 보여주는 실험 결과를 통해 보다 빠르고 정확한 배아 평가가 가능함을 입증했습니다.



### Online Pseudo-Label Unified Object Detection for Multiple Datasets Training (https://arxiv.org/abs/2410.15569)
- **What's New**: 새로운 논문에서는 Unified Object Detection (UOD) 작업을 위한 Online Pseudo-Label 통합 객체 감지 방법을 제안합니다. 이 방법은 여러 데이터세트에서 누락된 주석 문제를 분석하고, 주기적으로 업데이트되는 teacher 모델을 사용하여 라벨이 없는 객체에 대한 pseudo-label을 생성합니다.

- **Technical Details**: 논문에서는 교차 데이터세트의 누락된 주석 문제를 해결하기 위해 온라인 pseudo-label UOD 기법을 제안하며, teacher 모델의 주기적인 업데이트를 통해 pseudo-label의 질을 극대화합니다. 또한, Overlapped box 문제를 해결하기 위해 카테고리 전문 박스 회귀 및 pseudo-label RPN 헤드를 도입하여 Region Proposal Network (RPN)의 recall rate를 향상시킵니다.

- **Performance Highlights**: 실험 결과, 제안한 방법이 COCO, Object365 및 OpenImages와 같은 일반적인 벤치마크에서 기존 SOTA(Object Detection) 방법들보다 더 높은 정확도를 달성함을 보여줍니다.



### TrackMe:A Simple and Effective Multiple Object Tracking Annotation Too (https://arxiv.org/abs/2410.15518)
- **What's New**: 본 논문에서는 동물 추적을 위한 새로운 비디오 주석 도구인 TrackMe를 소개합니다. TrackMe는 LabelMe 도구를 기반으로 하여 개발되었으며, 다양한 컴퓨터 지식 수준의 사용자가 데이터 주석 작업을 더 수월하게 수행할 수 있도록 설계되었습니다.

- **Technical Details**: TrackMe는 다중 객체 추적(Multiple Object Tracking, MOT) 주석을 위해 특별히 개발된 도구입니다. 이 도구는 범용적인 주석 기능을 제공하면서도 ID 연관, 경계 상자 보간(Bounding Box Interpolation), 정보 수정 기능 등을 탑재하고 있습니다. 인터폴레이션은 Gaussian Process Regression (GPR) 기법을 사용하여 객체의 비선형 이동을 모델링하고 결측 상자를 예측합니다. 또한 TrackMe는 YOLO-v8을 기반으로 한 객체 감지 모델을 활용하여 주어진 프레임에서 ID를 자동으로 할당할 수 있는 기능을 제공합니다.

- **Performance Highlights**: TrackMe는 사용자에게 직관적이고 효율적인 사용자 경험을 제공합니다. 이 도구는 최소한의 하드웨어 요구 사항만으로도 다양한 시스템과 호환되며, 데이터 주석 작업을 크게 간소화합니다. 기존의 다른 도구와 비교했을 때, TrackMe는 동물 추적과 같은 특수한 데이터 세트에 최적화되어 있어, 높은 정확도를 유지하면서도 사용자의 수고를 줄여줍니다.



### Taming Mambas for Voxel Level 3D Medical Image Segmentation (https://arxiv.org/abs/2410.15496)
- **What's New**: 최근 3D 의료 분할 분야에서 Convolutional Neural Networks (CNNs)와 Transformer 기반 아키텍처가 각기 다른 강점과 한계를 가지고 사용되고 있습니다. Mamba라는 새로운 Recurrent Neural Network (RNN)이 State Space Models (SSMs)를 기반으로 하여 Transformer보다 더 긴 문맥 작업에서 뛰어난 성과를 보이고 있습니다.

- **Technical Details**: CNN은 국소 수용 영역(local receptive field)으로 인해 제한을 받고, Transformer는 큰 메모리 요구사항과 데이터 요구량(data hungriness)으로 인해 3D 의료 볼륨을 세밀하게 처리하기에 적합하지 않습니다. Mamba는 선형 복잡성을 유지하면서도 긴 컨텍스트 작업에서 뛰어난 성능을 제공하는 RNN 구조입니다.

- **Performance Highlights**: Mamba는 유명한 자연어 처리와 유전체(genomic) 벤치마크에서 Transformer와 비교하여 1백만 길이의 시퀀스를 처리하더라도 높은 성능을 기록하고 있습니다.



### Event-based Sensor Fusion and Application on Odometry: A Survey (https://arxiv.org/abs/2410.15480)
Comments:
          Submitted to IPAS2025: this https URL

- **What's New**: 이 논문은 로보틱스 및 컴퓨터 비전에서의 비디오 기반 odometry에 대한 최신 연구 동향을 제시하며, 비디오 카메라, 관성 측정 장치(IMU), 리다(LiDAR)와 같은 다양한 센서 간의 융합 전략을 탐구합니다. 기존의 비디오 카메라 관련 조사와는 달리, odometry 목적을 위해 이벤트 카메라 통합에 중점을 두고 있습니다.

- **Technical Details**: 이벤트 카메라는 고속 움직임, 저조도 및 넓은 동적 범위와 같은 복잡한 환경에서 효과적인 감지를 제공합니다. 이벤트 카메라는 설정된 시간 간격으로 이미지를 촬영하는 대신 비동기적으로 픽셀의 밝기 변화를 감지하여 이벤트 스트림을 생성하며, 이로 인해 높은 시간 해상도와 저지연성을 제공합니다. 이 논문은 이벤트 카메라의 비디오 기반 센서 융합을 통해 odometry 성능을 개선하는 여러 방법론을 논의합니다.

- **Performance Highlights**: 이벤트 카메라와 다른 센서(예: LiDAR, IMU) 간의 융합은 고속 변화하는 장면이나 저조도 조건에서 시각적 odometry의 정확성과 강인성을 향상시키는 데 중요한 역할을 합니다. 특히, LiDAR 기반 시스템보다 더 나은 환경에서의 정확성을 제공합니다.



### Generalized Multimodal Fusion via Poisson-Nernst-Planck Equation (https://arxiv.org/abs/2410.15475)
Comments:
          NeurIPS 2024 Rejected paper, 28 pages

- **What's New**: 본 논문에서 제안된 일반화된 다중 모달 융합 방법(GMF)은 Poisson-Nernst-Planck (PNP) 방정식을 활용하여 다중 모달 피처 융합의 효율성을 향상시키는 새로운 접근 방식을 제시합니다.

- **Technical Details**: GMF는 정보 엔트로피 이론과 PNP 방정식을 기반으로 하여 다중 모달 피처를 전하 입자로 취급하고, 이를 분리하여 모달리티-특정 및 모달리티-불변 서브스페이스로 재구성합니다. 이 과정은 피처 간의 상호 정보를 줄이고 다운스트림 태스크의 엔트로피를 감소시킵니다.

- **Performance Highlights**: GMF는 여러 다운스트림 태스크에서 실험하였으며, 기존의 최첨단(SOTA) 성능에 근접한 결과를 보여주고, 필요한 파라미터와 계산 자원의 양이 적습니다. 또한, GMF를 고급 융합 방법과 통합함으로써 SOTA 결과를 초과 달성했습니다.



### EVA: An Embodied World Model for Future Video Anticipation (https://arxiv.org/abs/2410.15461)
- **What's New**: 이 논문에서는 인간의 재사고(process of rethinking)를 통해 비디오 예측(video prediction)을 보다 세분화된 방식으로 다루기 위해 복잡한 비디오 예측 과제를 네 가지 메타 작업(meta-tasks)으로 분해했습니다. 또한 새로운 벤치마크인 Embodied Video Anticipation Benchmark (EVA-Bench)를 소개하며, 이는 인간과 로봇의 행동 예측 능력을 평가합니다.

- **Technical Details**: EVA는 비디오 생성 모델(video generation model)과 시각 언어 모델(visual language model)을 통합한 통합 프레임워크로, 비디오 이해(video understanding) 및 생성을 목표로 합니다. 여러 단계의 사전 학습(pretraining) 패러다임을 통해 LoRA(LoRA: Low-Rank Adaptation)를 적응적으로 조합하여 고충실도(high-fidelity) 결과를 생성할 수 있게 했습니다.

- **Performance Highlights**: EVA-Bench에서의 광범위한 실험 결과는 EVA가 현실-world 내의 복잡한 상황에서 성능을 크게 향상시킬 가능성을 강조하며, 이는 대규모 사전 학습 모델이 실제 예측 작업에 적합하다는 것을 입증합니다.



### Allegro: Open the Black Box of Commercial-Level Video Generation Mod (https://arxiv.org/abs/2410.15458)
- **What's New**: 이 보고서에서는 비디오 생성 분야의 최신 모델인 $	extbf{Allegro}$를 소개하며, 고품질의 비디오 생성과 시계열(temporal) 일관성에서 우수한 성능을 보여줍니다. Allegro는 기존의 오픈 소스 모델 및 상용 모델을 능가하는 성과를 보이며, 비디오 생성 모델의 상업적 성과를 달성하는 데 필요한 종합적인 방법론을 제시합니다.

- **Technical Details**: Allegro는 데이터, 모델 아키텍처, 훈련 파이프라인, 평가 등 비디오 생성 모델의 주요 요소를 강화하기 위해 Diffusion 모델 프레임워크 위에 Variational Autoencoder (VAE) 및 Diffusion Transformer (DiT) 아키텍처를 수정하여 개발되었습니다. 시각적 품질을 보장하기 위해 다양한 기준으로 사용자 연구를 진행하여 생성된 비디오의 미적 품질을 평가하였습니다.

- **Performance Highlights**: Allegro는 주관적인 평가에서 Hailuo와 Kling에 이어 상용 모델 중에서 가장 높은 품질을 기록하며, 비디오-텍스트 관련성과 전반적인 품질에서 상用 모델들을 초과하는 성능을 보여줍니다. 새로운 비디오 생성 기술의 발전에 따라 콘텐츠 제작자들에게 더 많은 가능성을 열어주는 Allegro의 기능이 주목받고 있습니다.



### Concept Complement Bottleneck Model for Interpretable Medical Image Diagnosis (https://arxiv.org/abs/2410.15446)
Comments:
          10 pages, 5 figures, submitted to IEEE TRANSACTIONS ON MEDICAL IMAGING

- **What's New**: 이 논문에서는 설명 가능한 의료 이미지를 진단하기 위한 Concept Complement Bottleneck Model (CCBM)을 제안합니다. CCBM은 기존 개념 집합을 보완하고 새로운 개념을 발견하여 설명 가능한 모델 간의 격차를 줄이는 것을 목표로 합니다.

- **Technical Details**: CCBM은 각 개념마다 개념 어댑터를 활용하여 해당 개념과 가장 관련성이 높은 이미징 기능을 인코딩합니다. Multi-Head Cross-Attention (MHCA)를 사용하여 각 개념의 점수를 독립적인 주의 채널에서 계산하여 공정한 개념 학습을 지원합니다. 또한, 알려진 개념을 함께 사용하여 새로운 개념을 학습하는 전략을 설계했습니다.

- **Performance Highlights**: 의료 데이터셋에서의 실험 결과, CCBM은 개념 탐지 및 질병 진단 작업에서 기존의 최신 모델과 비교하여 우수한 성능을 보였으며, 다양한 설명을 제공하여 모델의 해석 가능성을 효과적으로 보장합니다.



### MDFI-Net: Multiscale Differential Feature Interaction Network for Accurate Retinal Vessel Segmentation (https://arxiv.org/abs/2410.15444)
- **What's New**: 본 논문에서는 복잡한 구조로 인해 의료 영상 분할(Tasks)에서 큰 도전이 되는 망막 혈관의 정확한 분할을 위한 새로운 접근 방식인 Deformable-convolutional Pulse Coupling Network (DPCN) 기반의 특성 향상 상호 작용 네트워크를 제안합니다.

- **Technical Details**: 이 연구는 특성 향상을 위해 DPCN의 구조를 설계하였으며, 이를 통해 효율적이고 간단하게 분할 네트워크에 향상된 특징(iteration sequence)을 공급합니다. 제안된 네트워크는 다양한 공개 데이터셋에서 혈관 특징을 상호 작용시켜 최적의 성능을 발휘합니다.

- **Performance Highlights**: 실험 결과, 제안된 알고리즘은 망막 혈관의 검출 정확도가 각각 97.91%, 97.97%, 98.16%에 달하며, 공공 데이터셋에서 최첨단(state-of-the-art) 방법들에 비해 우수한 분할 성능을 입증하였습니다.



### MedDiff-FM: A Diffusion-based Foundation Model for Versatile Medical Image Applications (https://arxiv.org/abs/2410.15432)
- **What's New**: 이 연구에서는 MedDiff-FM이라는 새로운 diffusion 기반 foundation model을 소개합니다. 이 모델은 머리부터 복부까지 다양한 해부학적 영역을 포함하고 공공 데이터셋에서 수집한 3D CT 이미지를 활용하여 다양하고 복잡한 의료 이미지 작업을 처리할 수 있습니다.

- **Technical Details**: MedDiff-FM은 multi-level 이미지 처리를 지원하며, 이미지-레벨(image-level) 및 패치-레벨(patch-level) 입력을 모두 처리할 수 있습니다. 이 모델은 위치 임베딩(position embedding)을 통해 다중 공간적 관계를 형성하고, 해부학적 구조 및 영역 클래스를 활용하여 특정 해부학적 영역을 제어합니다.

- **Performance Highlights**: 실험 결과, MedDiff-FM은 이미지 노이즈 제거(image denoising), 이상 탐지(anomaly detection), 이미지 합성(image synthesis) 등 다양한 다운스트림 작업을 효과적으로 수행할 수 있음을 보여줍니다. 또한 ControlNet을 통한 빠른 파인튜닝(fine-tuning)을 통해 병변 생성 및 병변 보완을 수행할 수 있는 능력을 보여주었습니다.



### BoostAdapter: Improving Test-Time Adaptation via Regional Bootstrapping (https://arxiv.org/abs/2410.15430)
Comments:
          NeurIPS 2024

- **What's New**: 본 논문에서는 테스트 시간 적응(test-time adaptation, TTA)에서의 기존의 훈련 필요성과 훈련 불필요한 방법 간의 간극을 메우기 위해 새로운 접근 방식을 소개합니다. 새로운 적응 전략인 BoostAdapter를 제안하여, 훈련 불필요한 어댑터를 개선하고, 고품질 샘플을 메모리에 도입하여 성능을 향상시킵니다.

- **Technical Details**: BoostAdapter는 테스트 데이터 스트림에서 필터링한 instance-agnostic historical samples와 instance-aware boosting samples를 포함하는 경량의 key-value 메모리를 유지합니다. 이 방법은 테스트 샘플의 정보를 효과적으로 활용하면서 훈련이 필요 없는 적응 방법의 효율성을 유지합니다.

- **Performance Highlights**: BoostAdapter는 두 가지 기준 벤치마크를 통해 테스트 시간 적응 설정에서 탁월한 성능을 보여주며, 이론적 분석과 실험 결과를 통해 그 효과를 검증합니다.



### Accelerated Sub-Image Search For Variable-Size Patches Identification Based On Virtual Time Series Transformation And Segmentation (https://arxiv.org/abs/2410.15425)
Comments:
          10 pages, 9 figures, 3 tables

- **What's New**: 이 논문은 항공 이미지에서 고정 크기의 물체(예: 건초 더미)를 인식하는 방법과 변동 크기 패치(예: 점적 살포가 필요한 농경지)를 인식하는 메커니즘을 제시합니다. 특히, 이미지 처리를 위한 신경망 없이도 하이퍼파라미터 선택을 통해 사용자 선호에 적합한 결과를 지원하는 방법을 제안합니다.

- **Technical Details**: 이미지를 RGB 채널을 따라 다변량 시계열(multivariate time series)로 변환하고 이를 세분화(segmentation)하여 2D 검색 공간을 감소시키는 가속 메커니즘을 도입했습니다. 특히 APTS 알고리즘(A Posteriori Trading-inspired Segmentation)을 사용하여 시계열을 세분화합니다.

- **Performance Highlights**: 제안된 방법은 다양한 합성 및 실제 이미지에서 소모 시간 절약을 최대 2배 향상시키며, 비교적 유사한 시각적 결과를 제공합니다. 이 방법은 이미지 전처리를 하지 않으며, 신경망을 사용하지 않는 점이 특징입니다.



### MMCS: A Multimodal Medical Diagnosis System Integrating Image Analysis and Knowledge-based Departmental Consultation (https://arxiv.org/abs/2410.15403)
- **What's New**: MMCS 시스템은 의료 이미지를 인식하고 환자의 얼굴 세부 사항을 분석하여 전문적인 의료 진단을 제공할 수 있는 혁신적인 솔루션입니다.

- **Technical Details**: 시스템은 두 개의 핵심 구성 요소로 이루어져 있습니다. 첫 번째 구성 요소는 의료 이미지 및 비디오의 분석으로, 다중 모달(multimodal) 의료 모델을 훈련시켜 의료 이미지를 해석하고 환자의 얼굴 감정 및 얼굴 마비 상태를 정확히 분석할 수 있도록 하였습니다. 모델은 FER2013 얼굴 감정 인식 데이터셋에서 72.59%의 정확도를 달성하였으며, 행복 감정 인식에서는 91.1%의 정확도를 기록했습니다. 얼굴 마비 인식에서는 92%의 정확도를 달성하였으며 이는 GPT-4o보다 30% 더 높은 수치입니다. 두 번째 구성 요소는 전문 의료 응답 생성을 위한 대형 언어 모델(large language model)과 의료 지식 기반을 통합하여 의료 이미지를 분석한 후 적절한 진단을 생성하는 것입니다.

- **Performance Highlights**: 얼굴 마비 환자에 대한 30개의 비디오 테스트에서 시스템은 83.3%의 정확도로 마비의 중증도를 정확히 평가했습니다. 또한, 의료 부서별 지식 기반 라우팅 관리 메커니즘을 통해 대형 언어 모델은 데이터의 의료 부서를 분류하고 적절한 지식 기반을 조회하여 RAG(검색 보강 생성) 과정에서 검색 정확성을 4% 향상시켰습니다.



### EF-3DGS: Event-Aided Free-Trajectory 3D Gaussian Splatting (https://arxiv.org/abs/2410.15392)
Comments:
          Project Page: this https URL

- **What's New**: 이번 연구에서는 Event Camera를 처음으로 자유 카메라 궤적(free-trajectory) 비디오에서 장면 재구성을 지원하는 데 사용합니다. 특히, Event-Aided Free-Trajectory 3D Gaussian Splatting (EF-3DGS)라는 방식을 제안하여 Event Camera의 장점을 3DGS에 통합합니다.

- **Technical Details**: EF-3DGS는 세 가지 주요 구성 요소로 구성됩니다. 첫째, Event Generation Model (EGM)을 활용하여 이벤트와 프레임을 융합하고 Event Stream에서 관찰된 렌더링 뷰를 감독합니다. 둘째, Contrast Maximization (CMax) 프레임워크를 도입하여 움직임 정보를 추출하고, Linear Event Generation Model (LEGM)을 통해 IWE의 밝기 정보를 활용해 3DGS의 제약을 설정합니다. 마지막으로, Photometric Bundle Adjustment (PBA)를 통해 이벤트 간의 일관성을 확보합니다.

- **Performance Highlights**: 공공 Tanks and Temples 벤치마크 및 신규 수집된 실제 데이터셋 RealEv-DAVIS에서 평가한 결과, EF-3DGS는 기존 최첨단 방법에 비해 렌더링 품질에서 최대 2dB 높은 PSNR과 40% 낮은 Absolute Trajectory Error를 달성하였습니다.



### Layout-your-3D: Controllable and Precise 3D Generation with 2D Blueprin (https://arxiv.org/abs/2410.15391)
Comments:
          21 pages,17 figures

- **What's New**: Layout-Your-3D는 텍스트 프롬프트로부터 제어 가능한 3D 생성을 위한 새로운 프레임워크입니다. 기존 텍스트-투-3D 방법들은 객체의 상호작용을 신뢰성 있게 생성하지 못하거나 복잡한 최적화 과정이 필요했으나, 이 방법은 2D 레이아웃을 이용해 3D 생성의 제어를 용이하게 합니다.

- **Technical Details**: 이 방법은 사용자 제공 또는 LLM 생성 2D 레이아웃을 기반으로 초기 3D 장면을 생성하는 과정으로 시작합니다. 이후 충돌 인식 레이아웃 최적화 및 인스턴스별 정제를 통해 객체 간의 상호작용을 자연스럽고 시각적으로 매력적으로 만드는 데 중점을 둡니다.

- **Performance Highlights**: Layout-Your-3D는 각 프롬프트에 대한 시간 소모를 상당히 줄이면서 더 합리적이고 시각적으로 매력적인 조합의 3D 자산을 생성합니다. 실험 결과, 제안된 방법이 기존 최첨단 텍스트-투-3D 방법들과 비교하여 뛰어난 성능을 보임을 입증하였습니다.



### LoRA-IR: Taming Low-Rank Experts for Efficient All-in-One Image Restoration (https://arxiv.org/abs/2410.15385)
- **What's New**: LoRA-IR라는 새로운 이미지 복원 프레임워크를 제안합니다. 이 프레임워크는 Compact Low-Rank Experts를 동적으로 활용하여 복잡한 현실 환경에서 여러 종류의 이미지 손상을 효과적으로 처리합니다.

- **Technical Details**: LoRA-IR는 두 단계로 훈련됩니다: 손상 안내 프리트레이닝과 파라미터 효율적인 파인튜닝입니다. 손상 대표성을 제공하는 DG-Router를 통해 코어 네트워크를 조절합니다. LoRA를 바탕으로 한 Mixture-of-Experts 구조를 통해 다양한 저랭크 복원 전문가를 동적으로 결합합니다.

- **Performance Highlights**: LoRA-IR은 14개의 이미지 복원 작업과 29개의 벤치마크에서 최첨단 성능을 달성하였으며, 실제 시나리오에서 강력한 일반화 능력을 보여줍니다.



### Neural Active Structure-from-Motion in Dark and Textureless Environmen (https://arxiv.org/abs/2410.15378)
Comments:
          Accepted in Asian Conference on Computer Vision 2024

- **What's New**: 이 논문에서는 구조화된 빛(Structured Light, SL) 시스템의 이미지 집합에서 동시적으로 장면의 형상 재구성과 포즈 추정을 수행하는 새로운 기술인 Active SfM을 제안합니다. 특히, 텍스처가 없는 환경에서도 적용 가능하도록 설계되어 기존의 기술 제약을 극복할 수 있습니다.

- **Technical Details**: 이 논문은 Neural Signed Distance Fields (Neural-SDF)를 활용하여 SL 시스템의 움직임에 따른 장면의 3D 형상 및 카메라 포즈를 동시에 추정하는 새로운 볼륨 렌더링 파이프라인과 하이브리드 인코딩 기법을 제안합니다. 이는 최소한의 조명 조건에서도 효과적으로 작동할 수 있도록 설계되었습니다.

- **Performance Highlights**: 제안된 방법은 합성 및 실제 데이터에서 장면의 형상과 포즈를 정확히 추정하는 성능을 보여주었으며, 텍스처나 조명이 부족한 환경에서도 효과적으로 작동함을 입증했습니다.



### ActiveNeuS: Neural Signed Distance Fields for Active Stereo (https://arxiv.org/abs/2410.15376)
Comments:
          Accepted in International Conference on 3D Vision 2024

- **What's New**: 이번 논문에서는 Neural Signed Distance Field (Neural SDF) 기술을 활용하여, 저조도 환경에서도 효과적인 3D 형태 복원을 가능하게 합니다. 특히 Active Stereo 시스템을 적용하여 점 몇 개로도 높은 품질의 복원이 가능하다는 점을 강조하고 있습니다.

- **Technical Details**: 제안된 ActiveNeuS 방법은 패턴 프로젝션과 볼륨 렌더링을 결합하여 Neural SDF 파이프라인을 구성합니다. 이는 기존의 복잡한 알고리즘 없이 이미지-패턴 대응을 암묵적으로 최적화하여 단일 이미지, 즉 원샷 스캔에서도 동작하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 저조도 및 저적층(저텍스처) 환경에서도 우수한 3D 복원 품질을 달성했으며, 특히 수중 환경에서도 효과적으로 적용됨을 입증했습니다.



### FrameBridge: Improving Image-to-Video Generation with Bridge Models (https://arxiv.org/abs/2410.15371)
- **What's New**: 새로운 모델 FrameBridge를 제안하여, 정적 이미지를 비디오의 목표로 활용하고, 이미지와 비디오 간의 트랙터블(tractable) 브리지 모델을 구축했습니다. 이는 기존의 Diffusion 기반 I2V 생성 과정에서 발생하는 일관성 및 시간적 응집력 문제를 해결하는 데 기여합니다.

- **Technical Details**: FrameBridge는 I2V 합성을 프레임 간 생성(task)으로 체계화하고, 데이터 간 프로세스를 모델링하여 입력 이미지의 정보를 최대한 활용합니다. 추가적으로 SNR-Aligned Fine-tuning (SAF)과 neural prior라는 두 가지 기술을 제안하여, 기존의 diffusion 모델과 비교하여 효율성과 합성 품질을 향상시킵니다.

- **Performance Highlights**: WebVid-2M 및 UCF-101에서 실험 결과, FrameBridge는 이전의 diffusion 모델과 비교하여 I2V 품질을 현저하게 개선했습니다. 예를 들어, MSR-VTT에서 zero-shot FVD 점수를 176에서 83으로 줄였고, UCF-101에서는 non-zero-shot FVD 점수를 171에서 122로 감소시켰습니다.



### Scene Graph Generation with Role-Playing Large Language Models (https://arxiv.org/abs/2410.15364)
Comments:
          NeurIPS 2024. Code: this https URL

- **What's New**: 이 연구는 기존의 Open-Vocabulary Scene Graph Generation (OVSGG) 방법론의 한계를 극복하기 위해 scene-specific한 접근 방식을 제안합니다. 기존의 OVSGG 방법은 고정된 텍스트 분류기를 사용하여 다양한 상황에 적응하지 못한다는 문제를 갖고 있었습니다. 이에 비해, SDSGG는 시각적 내용을 기반으로 텍스트 분류기의 가중치를 적응적으로 조정합니다.

- **Technical Details**: SDSGG는 scene-specific description에 기반한 OVSGG 프레임워크로, LLM(대규모 언어 모델)을 활용하여 다양한 역할을 부여하고, 그에 따라 다양한 설명을 생성합니다. 또한, 각 텍스트 분류기의 중요성을 평가하여 조정할 수 있는 고급 renormalization 기법을 포함하고 있습니다. 새로운 경량 모듈인 mutual visual adapter를 도입하여 클립(CLIP)의 관계 인식 능력을 향상시킵니다.

- **Performance Highlights**: SDSGG는 Visual Genome과 GQA와 같은 두 개의 주요 벤치마크에서 테스트되었으며, 기존의 OVSGG 방법보다 뚜렷하게 우수한 성능을 보였습니다. 이를 통해 SDSGG의 강력한 일반화 능력과 유망한 성능을 입증하여 scene-specific한 관계 탐지의 가능성을 보여주었습니다.



### YOLO-RD: Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary (https://arxiv.org/abs/2410.15346)
- **What's New**: 본 논문에서는 YOLO 기반의 객체 탐지 모델에 외부 정보의 효율적인 활용을 지원하는 혁신적인 {f Retriever}-{f Dictionary} (RD) 모듈을 제안합니다. 이 모듈은 데이터셋의 통찰을 담고 있는 Dictionary에서 특성을 효율적으로 검색할 수 있게 해주며, Visual Models (VM), Large Language Models (LLM), Visual Language Models (VLM)의 지식을 활용합니다.

- **Technical Details**: RD 모듈은 객체 탐지 작업뿐만 아니라 세분화(segmentation)와 분류(classification) 작업에서도 동시에 이점을 취할 수 있도록 설계되었습니다. 이 모듈은 쿼리를 생성하기 위해 영역 특성을 집계하는 Retriever와, 관련 원자를 선택할 수 있도록 쿼리를 지원하는 Dictionary로 구성됩니다. Dictionary는 YOLO 백본을 넘어서는 외부 지식을 통합합니다.

- **Performance Highlights**: RD 모듈을 사용하는 실험 결과, 평균 정밀도(Mean Average Precision)가 3% 이상 증가하였으며, 모델 파라미터의 증가는 1% 미만으로 최소화되었습니다. 이 모듈은 YOLO 뿐만 아니라 Faster R-CNN, Deformable DETR와 같은 2단계 모델에도 성능을 향상시켜 주는 것으로 나타났습니다.



### Modality-Fair Preference Optimization for Trustworthy MLLM Alignmen (https://arxiv.org/abs/2410.15334)
- **What's New**: 새로운 연구에서는 Modality-Fair Preference Optimization (MFPO)라는 방법을 제안하여 다중 모달 모델(Multimodal Models, MLLMs)에서 텍스트와 이미지 정보를 균형 있게 최적화합니다. 기존의 Direct Preference Optimization (DPO) 방법은 텍스트에 편향이 있었으나, MFPO는 이미지 관련 보상을 자동으로 생성하여 그러한 문제를 극복합니다.

- **Technical Details**: MFPO는 이미지 선호 데이터를 생성하고, 모델이 텍스트와 이미지의 선호를 동시에 학습하도록 설계되었습니다. 이 과정에서 curriculum learning을 활용하여 데이터의 난이도를 조절하며, 향상된 훈련 안정성을 제공합니다. Marginal loss를 도입하여 모델의 선호 응답의 보상 일관성을 유지합니다.

- **Performance Highlights**: 실험 결과, MFPO는 LLaVA-v1.5 모델에서 hallucination 비율을 40% 이상 감소시켰으며, Object HalBench 및 AMBER 데이터셋에서 SOTA 성능을 달성했습니다. 특히, 7B 모델에서 CHAIRi 점수 5.1을 기록하며 기존 방법보다 거의 40% 향상된 성과를 보여주었습니다.



### Open-vocabulary vs. Closed-set: Best Practice for Few-shot Object Detection Considering Text Describability (https://arxiv.org/abs/2410.15315)
Comments:
          20 pages, 3 figures

- **What's New**: 이번 연구는 Open-vocabulary object detection (OVD)과 few-shot object detection (FSOD)의 효율성을 비교 분석하고, 'text-describability'를 활용하여 다양한 객체 탐지 데이터셋을 분류하는 방법을 제안합니다.

- **Technical Details**: OVD는 기존의 정의된 클래스만 식별하는 closed-set object detection (COD)과 다르게, 사전 학습된 대형 모델(BERT, CLIP 등)의 언어적 지식을 활용하여 이미지 샘플 없이도 객체를 탐지할 수 있게 합니다. 연구진은 zero-shot image classification 정확도를 기반으로 'text-describability'를 정량화하여, 이러한 다양한 OVD 및 COD 방법의 FSOD 성능을 평가합니다.

- **Performance Highlights**: 연구 결과, 텍스트로 쉽고 명확하게 설명 가능한 클래스에 대해서는 OVD가 COD보다 성능이 우수하지만, 텍스트로 설명하기 어려운 클래스에 대해서는 OVD와 COD 간의 차이가 적다는 것을 발견했습니다. 이는 OVD가 다양한 데이터로부터 학습할 수 있지만, 텍스트 설명이 어려운 클래스에 대해서는 비효율적일 수 있음을 시사합니다.



### Synergistic Dual Spatial-aware Generation of Image-to-Text and Text-to-Imag (https://arxiv.org/abs/2410.15312)
- **What's New**: 이번 연구에서는 Spatial Image-to-Text (SI2T)와 Spatial Text-to-Image (ST2I)라는 두 가지 작업을 이중 학습 프레임워크에서 함께 모델링하는 방법을 제안합니다. 특히, 3D 공간 장면 특성을 공유할 수 있는 새로운 3D Scene Graph (3DSG) 표현 방식을 도입하였습니다.

- **Technical Details**: SI2T는 주어진 이미지에서 객체들의 공간적 관계를 이해하는 반면, ST2I는 입력 텍스트 프롬프트에 기반하여 공간적으로 적합한 이미지를 합성합니다. 본 연구에서는 공간을 인식하는 데 필요한 3D 특징 모델링의 어려움을 극복하기 위해 Spatial Dual Discrete Diffusion (SD3) 프레임워크를 제안하며, 이는 중간 3D→X 과정의 특징을 활용하여 X→3D 과정을 지원합니다.

- **Performance Highlights**: VSD 데이터셋에서 수행한 실험 결과, 제안한 시스템이 기존 T2I 및 I2T 방법에 비해 우수한 성능을 보였으며, 이중 학습 전략이 시각-언어 모델을 통한 비대칭 공간 의미 정렬에 기여함을 밝혔습니다.



### ContextDet: Temporal Action Detection with Adaptive Context Aggregation (https://arxiv.org/abs/2410.15279)
- **What's New**: 이번 연구에서는 Temporal Action Detection(TAD) 분야에 새로운 단일 스테이지 모델인 ContextDet를 제안하며, 이는 대형 커널 컨볼루션(large-kernel convolutions)을 처음으로 사용한 점이 특징입니다. 이 모델은 행동 구분을 향상시키기 위해 적응형 컨텍스트 집계(pyramid adaptive context aggregation, ACA) 아키텍처를 활용합니다.

- **Technical Details**: ContextDet 모델은 각 ACA 수준에서 Context Attention Module(CAM)과 Long Context Module(LCM)이라는 두 가지 새로운 모듈로 구성되어 있습니다. CAM은 선택된 컨텍스트 정보를 활용하여 동작 구분을 향상시키고, LCM은 대형 및 소형 커널 컨볼루션을 혼합하여 장기적인 컨텍스트와 세밀한 로컬 특징을 효과적으로 수집합니다.

- **Performance Highlights**: ContextDet 모델은 MultiThumos, Charades, FineAction, EPIC-Kitchens 100, Thumos14, HACS 등 6개의 도전적인 TAD 벤치마크에서 다른 최신 TAD 방법들과 비교하여 뛰어난 정확성과 향상된 추론 속도를 기록하였습니다.



### Can LVLMs Describe Videos like Humans? A Five-in-One Video Annotations Benchmark for Better Human-Machine Comparison (https://arxiv.org/abs/2410.15270)
- **What's New**: 본 논문에서는 LVLMs(대형 비전-언어 모델)가 인간과 유사하게 비디오를 이해할 수 있는지를 평가하기 위해 새로운 벤치마크인 FIOVA를 제안합니다. 이는 비디오 설명 작업을 통해 LVLMs의 성능을 인간과 비교하는 데 중점을 두고 있습니다.

- **Technical Details**: FIOVA(다섯 가지의 비디오 주석)는 3,002개의 긴 비디오 시퀀스를 포함하고 있으며, 각 비디오는 다섯 명의 주석자에 의해 주석이 달려 다양한 관점을 포착합니다. 이 비디오는 평균적으로 33.6초 길이이며, 기존 벤치마크보다 4-15배 긴 자막을 포함합니다. 이는 LVLMs와 인간의 이해 능력을 종합적으로 평가하는 데 도움을 줍니다.

- **Performance Highlights**: FIOVA 벤치마크를 사용하여 여섯 개의 최첨단 LVLMs의 성능을 인간과 비교하여 심층적으로 평가했습니다. 이 평가를 통해 LVLMs의 복잡한 비디오 이해 능력에 대한 보다 명확한 통찰을 제공합니다.



### GSSF: Generalized Structural Sparse Function for Deep Cross-modal Metric Learning (https://arxiv.org/abs/2410.15266)
Comments:
          12 pages, 9 figures, Accepted by TIP2024

- **What's New**: 본 논문에서는 cross-modal metric learning의 새로운 접근 방식인 Generalized Structural Sparse Function(GSSF)을 제안합니다. 이 방법은 다양한 application scenarios에 유연하게 적용할 수 있으며, semantic heterogeneity를 효과적으로 해결합니다.

- **Technical Details**: 제안하는 GSSF는 Diagonal Metric(Diag)과 Block-Diagonal Metric(B-Diag) 두 가지 메트릭을 통해 채널 간의 상관성과 의존성을 보다 잘 포착합니다. 이를 통해 pair-wise feature 간의 유사도를 측정하며, 딥러닝 모델에서의 최적화 패턴을 자동으로 학습할 수 있습니다.

- **Performance Highlights**: GSSF는 이미지-텍스트 검색, 사람 재식별, 세밀한 이미지 검색과 같은 여러 가지 주요 retrieval 작업에서 기존의 접근 방식보다 우수하고 유연한 성능을 보였으며, Attention Mechanism, Knowledge Distillation과 같은 여러 응용 분야에서도 효과적으로 통합될 수 있습니다.



### Modeling Visual Memorability Assessment with Autoencoders Reveals Characteristics of Memorable Images (https://arxiv.org/abs/2410.15235)
- **What's New**: 이 논문에서는 이미지의 기억력(memorability)을 정량적으로 평가하기 위해 VGG16 기반의 오토인코더(autoencoder)를 활용한 딥러닝 모델을 제안합니다. 이를 통해 단일 노출 상태에서의 시각적 기억 경험을 모델링하고, 오토인코더의 재구성 오류와 기억력 간의 상관관계를 조사합니다.

- **Technical Details**: 연구는 MemCat 데이터셋을 사용해 10,000개의 이미지를 다섯 가지 카테고리(동물, 스포츠, 음식, 풍경, 차량)로 분류하고, 각 이미지의 기억력 점수를 반복 탐지(memory game) 방식으로 수집했습니다. 사용된 VGG16 오토인코더는 이미지의 중요한 특징을 압축하고 재구성하는데 중점을 두며, 잠재 공간(latent space) 표현을 통해 이미지의 특성을 분석합니다. 또한, Gated Recurrent Unit (GRU) 모델을 개발하여 잠재 공간 표현을 바탕으로 기억력 예측 모델을 설계했습니다.

- **Performance Highlights**: 오토인코더의 재구성 오류와 이미지의 기억력 점수 사이에 유의미한 상관관계가 발견되었으며, 잠재 표현의 독특성이 기억력과 긍정적 상관관계를 가지는 것으로 나타났습니다. 더불어, 강한 대비(strong contrasts), 독특한 객체(distinctive objects), 두드러진 전경 요소(prominent foreground elements) 등이 기억력 향상에 기여하는 중요한 시각적 특성으로 파악되었습니다.



### Deep Learning-based Detection of Bacterial Swarm Motion Using a Single Imag (https://arxiv.org/abs/2410.15229)
Comments:
          17 Pages, 4 Figures

- **What's New**: 이 연구에서는 흐릿한 이미지 하나로 박테리아의 군집 이동 확률을 신속히 예측할 수 있는 딥러닝 기반의 군집 분류기를 소개합니다. 전통적인 비디오 기반 방법에 비해 효율적이며 고속 처리 환경에 적합합니다.

- **Technical Details**: 제안된 군집 분류기는 Enterobacter sp. SM3에 대해 훈련되었으며, 새로운 군집 이미지(양성)와 수영 이미지(음성)에 대해 97.44%의 민감도(sensitivity)와 100%의 특이도(specificity)를 보였습니다. 이 분류기는 Serratia marcescens DB10과 Citrobacter koseri H6와 같은 보지 않은 박테리아 종에도 잘 일반화되었습니다.

- **Performance Highlights**: DB10에서 97.92%의 민감도와 96.77%의 특이도를, H6에서 100%의 민감도와 97.22%의 특이도를 기록했습니다. 이러한 성능은 스마트폰과 같은 휴대기기에서 진단 응용 프로그램으로의 적용 가능성을 보여줍니다.



### Low-cost Robust Night-time Aerial Material Segmentation through Hyperspectral Data and Sparse Spatio-Temporal Learning (https://arxiv.org/abs/2410.15208)
Comments:
          Accepted to the International Conference on Neural Information Processing (ICONIP) 2024. To be published in Springer-Nature Communications in Computer and Information Science (CCIS) Series

- **What's New**: 이 논문에서는 저조도 및 대기 조건에서의 항공 데이터를 다루는 복잡한 재료 세분화(detection) 문제를 해결하기 위해, RGB 이미지와 더불어 특수 카메라에서 얻은 하이퍼스펙트럴(Hyperspectral) 데이터를 활용하는 혁신적인 Siamese 프레임워크를 제안합니다. 이는 시계열(time series) 기반의 압축(compression) 방식을 통해 저해상도의 하이퍼스펙트럴 이미지를 효과적으로 통합하는 방법을 고안하였습니다.

- **Technical Details**: 제안된 프레임워크는 다음과 같은 주요 기술적 세부 사항을 포함합니다: 1) 선택적 채널 활용(Selective Channel Utilization): 데이터 세트 전체를 처리하는 대신 필수적인 스펙트럴 데이터를 추출합니다. 2) 고급 신경망 아키텍처(Advanced Deep Learning Architecture): 하이퍼스펙트럴 및 RGB 데이터를 통합하여 효율적으로 세분화 작업을 수행합니다. 3) 불리한 환경에 대한 견고성(Robustness to Adverse Conditions): 저조도 및 대기 방해와 같은 어려운 환경에서도 안정적인 성능을 보입니다.

- **Performance Highlights**: 항공 데이터셋을 다양한 환경 조건에서 훈련 및 평가하여, 제안된 모델이 경쟁 기준(comparative benchmark)에서 우수한 성능을 나타냄을 입증하였습니다. 특히, 하이퍼스펙트럴 데이터의 저해상도를 효과적으로 활용하여 경제적인 장비를 통해도 뛰어난 결과를 도출할 수 있음을 보여주었습니다.



### Unsupervised Domain Adaptation Approaches for Chessboard Recognition (https://arxiv.org/abs/2410.15206)
Comments:
          30 pages, 23 figures

- **What's New**: 이 논문은 체스판 사진의 자동 주석 생성을 위한 비지도 학습 방식을 제안합니다. 도메인 적응(DA) 기술을 활용하여 실제 체스판 이미지의 레이블을 예측하는 새로운 파이프라인을 개발하였습니다.

- **Technical Details**: 제안된 파이프라인은 세 가지 주요 구성 요소로 이루어져 있습니다: 1) 이미지 전처리, 2) 딥러닝 모델, 3) 후처리. 전처리 단계에서 체스판을 탐지하고 개별 제곱을 잘라낸 후, 이를 딥러닝 모델에 전달하여 체스 기물의 레이블을 예측합니다. 모델은 VGG16 아키텍처를 기반으로 하며, 다양한 도메인 적응 접근 방식을 사용하였습니다.

- **Performance Highlights**: DANN(Domain Adversarial Neural Network) 모델은 Base-Target 모델에 비해 정확도에서 3%의 손실만 보였고, 데이터 레이블링에 들어가는 모든 노력을 절약할 수 있었습니다. 이 결과는 도메인 적응이 실제 이미지에서의 인식 성능을 개선하는 데 도움이 됨을 시사합니다.



### CLIPtortionist: Zero-shot Text-driven Deformation for Manufactured 3D Shapes (https://arxiv.org/abs/2410.15199)
- **What's New**: 이번 연구에서는 입력된 3D 메쉬를 텍스트 설명에 맞게 변형하는 제로 샷 텍스트-주도 3D 형태 변형 시스템을 제안합니다. CLIP 모델을 기반으로 최적화된 변형 모델 파라미터를 활용하여 최적의 결과를 도출합니다.

- **Technical Details**: 본 시스템은 BoxDefGraph라는 새로운 변형 모델을 통해 물체의 사각형/원형 기하학적 특징을 포착하고 이를 활용하여 대규모 변형을 지원합니다. CMA-ES(global optimization) 알고리즘을 사용하여 최적화를 수행합니다.

- **Performance Highlights**: 이 방법론은 기존 방법들과 비교하여 우수한 성능을 보이며, 섬세한 표면 디테일을 보존하면서도 대규모 형태 변형을 가능하게 합니다.



### Standardizing Generative Face Video Compression using Supplemental Enhancement Information (https://arxiv.org/abs/2410.15105)
- **What's New**: 이 논문에서는 보완 향상 정보(Supplemental Enhancement Information, SEI)를 사용하는 생성적 얼굴 비디오 압축(Generative Face Video Compression, GFVC) 방법을 제안합니다. 이 방법은 얼굴 비디오 신호의Compact spatial 및 temporal 표현(즉, 2D/3D key-points, facial semantics 및 compact features)을 사용할 수 있으며, 이를 SEI 메시지로 인코딩하여 비디오 비트스트림에 삽입할 수 있습니다. 제안된 GFVC 방법은 ISO/IEC JVT 1/SC 29 및 ITU-T SG16의 공동 비디오 전문가 팀(Joint Video Experts Team, JVET)이 표준화 검토 중인 "기술 고려사항(technology under consideration, TuC)"입니다.

- **Technical Details**: 제안된 방식은 SEI 메시지를 사용하여 GFVC를 표준화하는 합리적인 솔루션을 제공합니다. 이를 통해 다양한 GFVC 기능 형식을 비디오 비트스트림과 seamless하게 결합할 수 있으며, 전통적인 하이브리드 비디오 코덱의 인코더/디코더에 표준적 변경을 도입하지 않습니다. GFVC 인코딩과 디코딩 프로세스에 대한 세부 사항과 함께, SEI 메시지의 구조와 작동 방식에 대한 내용이 포함되어 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 SEI 기반 GFVC 방법은 최신 다목적 비디오 코딩(Versatile Video Coding, VVC) 표준과 비교했을 때 뛰어난 데이터 비트율-왜곡 성능(rate-distortion performance)을 달성할 수 있으며, 사용자 지정 애니메이션/필터링 및 메타버스 관련 애플리케이션과 같은 다양한 기능을 지원할 가능성이 있습니다.



### CosFairNet:A Parameter-Space based Approach for Bias Free Learning (https://arxiv.org/abs/2410.15094)
- **What's New**: 본 연구에서는 모델의 파라미터 공간에서 직접적으로 편향(bias)을 완화하는 새로운 접근 방식인 bias 모델과 debias 모델을 훈련하는 방법을 제안합니다. 이는 기존의 샘플 기반 또는 피처 기반 접근 방식에서 벗어나 더 직접적인 방법으로 편향의 전파를 방지합니다.

- **Technical Details**: 제안된 방법은 편향이 있는 특성을 학습하는 bias 모델과 비편향 세부 정보를 학습하는 debias 모델, 두 가지 모델을 훈련합니다. debias 모델의 초기 레이어는 bias 모델과 유사성을 가지도록 하고, 마지막 레이어에서는 비유사성을 강제하여 편향된 고수준 추상화를 학습하지 않도록 합니다. 이러한 제약 조건을 훈련 중에 적용함으로써, 다양한 데이터셋에서 분류 정확도와 비편향 효과성을 향상시킵니다.

- **Performance Highlights**: 제안된 방법은 두 개의 실제 데이터셋과 두 개의 통제된 합성 데이터셋에서 기존의 편향 레이블 기반 접근 방식과 샘플 재가중 기반 접근 방식에 비해 우수한 성능을 보여줍니다. 또한, 다양한 편향 유형과 교육 데이터 내 편향 샘플 비율에 대해서도 안정성을 입증합니다.



### Spatial-Mamba: Effective Visual State Space Models via Structure-Aware State Fusion (https://arxiv.org/abs/2410.15091)
Comments:
          16 pages, 8 figures, 5 tables

- **What's New**: 이 논문에서는 기존의 시각적 상태 공간 모델(SSM)의 한계를 극복하기 위해 Spatial-Mamba라는 새로운 접근 방식을 제안합니다. 전체적으로 일관된 구조를 이용하여 이미지의 공간적 구조를 효과적으로 캡처할 수 있는 방법을 소개합니다.

- **Technical Details**: Spatial-Mamba는 3단계로 진행됩니다: (1) 비방향 스캔에 의한 초기 상태 계산, (2) 구조 인식 상태 융합을 통한 공간 컨텍스트 획득, (3) 관측 방정식을 이용한 최종 상태 계산. 구조 인식을 통해 공간적 의존성을 캡처하는 dilated convolutions을 사용합니다.

- **Performance Highlights**: Spatial-Mamba는 단일 스캔임에도 이미지 분류, 감지 및 분할에서 기존 SSM 기반 모델들의 성능을 초과하거나 동등한 결과를 도출합니다.



### SLIC: Secure Learned Image Codec through Compressed Domain Watermarking to Defend Image Manipulation (https://arxiv.org/abs/2410.15075)
Comments:
          accepted by ACM Multimedia Asia 2024

- **What's New**: 본 논문은 Secure Learned Image Codec (SLIC)라는 새로운 방법을 소개하며, 이는 이미지의 압축 도메인에서 워터마크를 삽입하여 이미지의 진본성을 보장하는 능동적 접근법입니다.

- **Technical Details**: SLIC는 신경망 기반의 압축 방식을 활용하여 잠재 공간에서 적대적 섭섭을 통해 워터마크를 삽입합니다. 이 방식은 다시 압축될 경우 품질이 저하되는 이미지를 생성하여 무단 수정을 방어하는 메커니즘으로 작용합니다. 특히, 신경 인코더/디코더를 미세 조정하여 워터마크의 보이지 않음과 견고함(robustness) 간의 균형을 맞춥니다.

- **Performance Highlights**: 실험 결과, SLIC는 변조된 이미지에서 가시적인 아티팩트를 생성하여 이를 방지하였고, SLIC 포맷을 이용한 이미지가 보다 높은 신뢰성을 가지는 것으로 나타났습니다.



### LLaVA-Ultra: Large Chinese Language and Vision Assistant for Ultrasound (https://arxiv.org/abs/2410.15074)
- **What's New**: 최근 멀티모달 대형 언어 모델(MLLM)이 주목받고 있으며, 이는 대화형 생성 AI의 텍스트 중심에서 멀티모달 작업으로의 전환을 촉진합니다. 특히 의료 분야에서 강력한 영향을 미치고 있습니다.

- **Technical Details**: 이 논문에서는 중국 의료 시각 대화를 위한 세밀한 적응형 VLM 아키텍처를 제안합니다. 파라미터 효율적 튜닝을 통해 정교한 비전 인코더(vision encoder)와 융합 모듈을 개발하여 미세한 의료 시각 의미를 향상시킵니다. 또한, 의료 장면에서 흔히 발생하는 데이터 중복(data redundancy)을 해결하기 위해 지식을 증류(knowledge distillation)를 이용한 가중치 점수(weighted scoring)를 사용하여 텍스트 설명을 반영하는 유효 이미지를 선별합니다.

- **Performance Highlights**: LLaVA-Ultra는 대규모 멀티모달 중국 초음파 데이터 세트를 기반으로 하여 의사들의 전문적인 텍스트를 사용하여 적절한 튜닝을 보장합니다. 세 가지 Med-VQA 데이터셋에서 LLaVA-Ultra는 다양한 지표에서 이전의 최첨단 모델을 초월하는 성능을 보여줍니다.



### A Cycle Ride to HDR: Semantics Aware Self-Supervised Framework for Unpaired LDR-to-HDR Image Translation (https://arxiv.org/abs/2410.15068)
Comments:
          Submitted to IEEE

- **What's New**: 이 논문은 LDR(저 동적 범위)에서 HDR(고 동적 범위)로의 이미지 변환에서, 고품질의 짝지어진 {LDR,HDR} 데이터셋에 대한 의존도를 줄이고, 짝지어지지 않은 데이터셋을 활용하는 방법을 제안합니다. 또한, 시각적 아티팩트 제거와 의미론적 일관성을 해결하기 위해 새로운 생성기와 손실 함수를 도입하였습니다.

- **Technical Details**: 제안된 방법은 수정된 사이클 일관성 적대적 구조(modified cycle-consistent adversarial architecture)를 활용하며, CLIP(Contrastive Language-Image Pre-training) 임베딩을 통해 LDR과 복원된 HDR 간의 의미론적 일관성을 확보합니다. 또한, 생성기(generator)는 수정된 U-Net 아키텍처를 기반으로 하며, ConvLSTM 기반의 피드백 메커니즘을 통합하여 시각적 아티팩트를 줄이는 데 기여합니다.

- **Performance Highlights**: 이 연구는 여러 벤치마크 데이터셋에서 상태-of-the-art 결과를 달성하였으며, 단일 노출 LDR 이미지를 사용하여 아티팩트 없이 시각적으로 인상적인 HDR 이미지를 복원하는 데 성공하였습니다.



### A Survey on All-in-One Image Restoration: Taxonomy, Evaluation and Future Trends (https://arxiv.org/abs/2410.15067)
- **What's New**: 이번 논문은 All-in-One 이미지 복원(AiOIR) 방법론에 대한 포괄적인 리뷰를 제공하며, 다양한 왜곡 유형을 다루는 혁신적인 접근 방식을 강조합니다. AiOIR은 단일 모델 내에서 이미지 향상 및 복원 기술을 통합하여 유연성과 효과성을 높입니다. 이를 통해 전통적인 단일 작업 이미지 복원 기법의 한계를 극복하고 있습니다.

- **Technical Details**: AiOIR 모델은 단일 프레임워크에서 여러 이미지 왜곡 유형을 동시에 처리할 수 있는 능력을 목표로 합니다. 주요 기술 요소로는 Mixture-of-Experts (MoE), prompt-based learning 접근법, 다중 모달 모델 등이 있으며, 데이터 세트와 구현 세부 사항을 체계적으로 정리하여 평가 메트릭을 제시합니다. 이 방법은 zero-shot 및 open-set, close-set 시나리오에 따라 나뉘어 있으며, 이는 복잡한 색수차 관리에 대한 이해의 바탕이 됩니다.

- **Performance Highlights**: AiOIR 방법은 다양한 손상 방법을 함께 고려함으로써 고품질 이미지를 생성하는 데 있어 긍정적인 진전을 이루었습니다. 최근 연구에서는 컴퓨팅 복잡성과 복원 품질 간의 균형을 맞추기 위한 아키텍처 최적화에 많은 관심을 두고 있으며, 이 연구가 향후 AiOIR 분야의 혁신을 촉진하고 있습니다.



### EndoMetric: Near-light metric scale monocular SLAM (https://arxiv.org/abs/2410.15065)
Comments:
          ICRA 2025

- **What's New**: 이 논문은 의료용 내시경 이미지를 사용하여 단안 카메라(monocular camera)로 정확한 측정 스케일을 가진 3D 재구성을 가능하게 하는 새로운 방법을 제안합니다. 이는 표준 내시경의 근광원(near-light source) 정보를 활용하여 실제 메트릭 스케일을 회복하는 것을 포함합니다.

- **Technical Details**: 저자들은 두 단계 접근법을 사용하여, 첫 번째 단계에서 SfM(Structure from Motion) 또는 V-SLAM(Visual Simultaneous Localization and Mapping)을 이용해 스케일에 따른 재구성을 확보하고, 두 번째 단계에서 포토메트릭 최적화를 통해 실측 스케일, 알베도(albedo) 및 카메라 게인(camera gain)을 복구하는 방법을 설명합니다. 이를 통해 격차가 발생할 수 있는 문제들을 해결하며 정밀한 측정이 가능합니다.

- **Performance Highlights**: 시뮬레이션 실험에서는 실제 내시경 환경에서 얻을 수 있는 스케일 정확도를 보여주며, 단안 내시경 시스템을 사용하는 경우에도 메트릭 스케일이 정확하게 복구될 수 있음을 증명하였습니다.



### BYOCL: Build Your Own Consistent Latent with Hierarchical Representative Latent Clustering (https://arxiv.org/abs/2410.15060)
Comments:
          5 pages, 5 figures

- **What's New**: 새로운 모델인 BYOCL을 제안하여 이미지 시퀀스를 처리할 때 발생하는 의미적 불일치 문제를 해결합니다. BYOCL은 SAM보다 우수한 성능을 보이며, CLIP 및 기타 표현을 통한 계층적 프로토타입 기능을 갖추고 있습니다.

- **Technical Details**: BYOCL은 SAM 이미지 인코더를 활용하여 특징을 추출한 후, Intra-Batch 및 Inter-Batch 클러스터링 알고리즘을 적용합니다. 데이터를 더 작은 배치로 나누어 처리하여 시간과 공간 소모를 줄이고, 지수적 시간 감소를 이룰 수 있습니다.

- **Performance Highlights**: 광범위한 실험을 통해 BYOCL은 이전의 최고 성능 단일 이미지 분할 모델을 초월했습니다. 성능 메트릭스(IOU, F1, recall)에서 기존의 모델들(SAM 등)보다 뛰어난 결과를 보여주며, 다양한 데이터 세트(MOSE, DAVIS)에서도 강력한 성능을 발휘했습니다.



### A General-Purpose Multimodal Foundation Model for Dermatology (https://arxiv.org/abs/2410.15038)
Comments:
          56 pages; Technical report

- **What's New**: PanDerm는 200만 개 이상의 피부 질환의 실제 이미지를 기반으로 한 self-supervised 학습을 통해 사전 훈련된 멀티모달 피부과 기초 모델입니다. 이는 기존의 이미지 기반 딥러닝 모델보다 복잡한 임상 요구 사항을 충족하는 데 중점을 둡니다.

- **Technical Details**: PanDerm는 11개의 임상 기관에서 수집된 4가지 이미징 모달리티를 통해 2백만 개 이상의 실제 피부 질환 이미지를 학습했습니다. 28개의 다양한 데이터 세트에서 피부암 스크리닝, 표현형 평가 및 위험 분류, 신생물 및 염증성 피부 질병 진단 등 다양한 임상 작업에 대한 평가를 수행했습니다.

- **Performance Highlights**: PanDerm는 모든 평가 작업에서 최첨단 성능을 기록했으며, 레이블된 데이터의 5-10%만 사용해도 기존 모델을 초과하여 성능을 발휘했습니다. 특히, 초기 단계 멜라노마 탐지 정확도에서 임상의보다 10.2% 향상되었고, 다중 클래스 피부암 진단 정확도에서는 11% 향상되어 협업 AI 환경에서 임상의의 진단 능력을 강화했습니다.



### Cutting-Edge Detection of Fatigue in Drivers: A Comparative Study of Object Detection Models (https://arxiv.org/abs/2410.15030)
- **What's New**: 이 연구는 최신 객체 탐지 알고리즘인 YOLO (You Only Look Once) 모델, 특히 YOLOv5, YOLOv6, YOLOv7 및 YOLOv8 기반의 피로 탐지 시스템 개발에 주목하고 있습니다. 이 모델들의 성능 비교를 통해 운전자의 피로 관련 행동을 실시간으로 탐지하는 효과성을 평가합니다.

- **Technical Details**: 연구에서는 16,246개의 운전자의 얼굴 이미지 데이터를 사용하였으며, 다양한 피로 단계(하품, 눈 감기 등)를 나타내는 네 가지 주요 클래스로 주석 처리가 되었습니다. YOLOv8은 C2f 모듈 및 앵커 프리 설계를 통해 보다 효율적인 특징 추출을 가능하게 하여 실시간 애플리케이션에서 속도와 정확성 면에서 다른 모델을 뛰어넘을 것으로 예상됩니다.

- **Performance Highlights**: YOLOv8이 다른 YOLO 모델들보다 우수한 성능을 보이며, 정확성과 속도 간의 균형을 잘 유지하는 것으로 나타났습니다. 데이터 증강 기법과 모델 최적화가 다양한 주행 조건에 대한 시스템 적응력을 향상시키는 데 중요한 역할을 했습니다.



### Group Diffusion Transformers are Unsupervised Multitask Learners (https://arxiv.org/abs/2410.15027)
- **What's New**: 대형 언어 모델(LLMs)에서 영감을 받아 GDT(그룹 확산 변환기) 프레임워크가 다양한 시각 생성 작업을 계층화하여 단일 그룹 생성 문제로 재정의하였습니다. 경험적인 데이터셋에 의존하지 않고, 비지도 학습을 통해 효과적인 처리가 가능합니다.

- **Technical Details**: GDT는 확산 변환기(diffusion transformers)를 최소한의 구조적 수정으로 기반으로 하여 이미지 간의 관계를 캡처하는 특징을 갖습니다. 이는 이미지 그룹의 자기 주의 토큰을 연결하여 수행하며, 다중 이미지 간의 상호 작용을 통해 아이덴티티, 스타일, 레이아웃 등을 포착합니다.

- **Performance Highlights**: 200개 이상의 지침을 포함한 30가지의 시각 생성 작업에서 GDT는 추가적인 파라미터 조정이나 기울기 업데이트 없이도 경쟁력 있는 제로샷(zero-shot) 성능을 보여주었습니다. 또한, 실험적 분석을 통해 데이터 규모, 그룹 크기, 모델 설계와 같은 주요 구성 요소의 효과를 검증하였습니다.



### MambaSOD: Dual Mamba-Driven Cross-Modal Fusion Network for RGB-D Salient Object Detection (https://arxiv.org/abs/2410.15015)
- **What's New**: 본 논문에서는 RGB-D Salient Object Detection(SOD)을 위해 새로운 이중 Mamba 기반의 크로스 모달 융합 네트워크인 MambaSOD를 제안합니다. RGB 및 깊이(depth) 정보를 효과적으로 융합하기 위한 혁신적인 접근법을 도입한 최초의 연구입니다.

- **Technical Details**: 이 모델은 RGB와 깊이 이미지에서 장거리 의존성(long-range dependencies)을 선형 복잡도로 모델링하기 위한 이중 Mamba 기반의 특징 추출기를 사용합니다. 또한 RGB와 깊이 특징 간의 보완 정보를 효과적으로 활용하기 위한 크로스 모달 융합 Mamba를 설계하였습니다.

- **Performance Highlights**: 여러 주요 데이터셋에서 수행된 실험 결과, MambaSOD는 16개의 최첨단 RGB-D SOD 모델보다 우수한 성능을 보여줬습니다. 이는 본 연구의 방법론이 RGB-D SOD 분야에서의 효용성과 우월성을 입증함을 의미합니다.



### DiffuseST: Unleashing the Capability of the Diffusion Model for Style Transfer (https://arxiv.org/abs/2410.15007)
Comments:
          Accepted to ACMMM Asia 2024. Code is available at this https URL

- **What's New**: 본 연구에서는 스타일 트랜스퍼를 위한 새로운 접근 방식을 제안하였습니다. 기존의 방법들이 특정 네트워크를 학습하거나 사전 학습된 모델을 활용하는 한계가 있었던 반면, 우리는 텍스트 표현과 공간적 특징을 결합하고 콘텐츠와 스타일의 주입을 분리하는 방식으로 이를 극복하였습니다.

- **Technical Details**: 우리는 BLIP-2 인코더를 활용하여 스타일 이미지의 텍스트 표현을 추출하였고, DDIM(inversion) 기술을 통해 콘텐츠와 스타일 Branch에서 중간 임베딩을 추출하여 공간적 특징으로 사용하였습니다. 스타일과 콘텐츠의 주입을 타겟 브랜치에서 분리함으로써, 콘텐츠 보존과 스타일 융합 간의 균형을 개선할 수 있었습니다. 또한, 모델 학습이 필요 없는 방법으로 갈수록 더 효과적인 스타일화 결과를 실현했습니다.

- **Performance Highlights**: 다양한 실험을 통해 제안된 DiffuseST가 균형 잡힌 스타일 트랜스퍼 결과를 달성하며, 컨트롤 가능한 스타일 전이 효과를 입증했습니다. 실험은 이 방법의 효과성과 강 robustness를 명확히 보여주었습니다.



### How Many Van Goghs Does It Take to Van Gogh? Finding the Imitation Threshold (https://arxiv.org/abs/2410.15002)
Comments:
          Accepted at ATTRIB, RegML, and SafeGenAI workshops at NeurIPS 2024 and NLLP Workshop 2024

- **What's New**: 본 연구에서는 텍스트-이미지 모델이 개념을 모방할 수 있는 임계점, 즉 'Imitation Threshold'를 찾는 문제를 제기합니다. 이를 통해 저작권 침해 및 개인 정보 보호에 대한 새로운 기준을 제공합니다.

- **Technical Details**: 연구는 MIMETIC2라는 방법론을 제안하여, 다양한 개념의 인스턴스 수를 비교하여 모방 임계점을 추정합니다. 실험은 인간 얼굴 및 예술 스타일 도메인을 포함하며, 각 모델의 모방 능력을 측정합니다.

- **Performance Highlights**: 모델의 모방 임계점은 도메인 및 모델에 따라 200에서 600 이미지 범위에 위치함을 밝혔습니다. 이 결과는 모형 개발자들에게 저작권 및 개인 정보 보호 법규를 준수하는 데 유용한 지침을 제공합니다.



### Making Every Frame Matter: Continuous Video Understanding for Large Models via Adaptive State Modeling (https://arxiv.org/abs/2410.14993)
- **What's New**: C-VUE라는 새로운 시스템을 소개하며, 비디오 이해에서의 여러 가지 도전 과제를 극복할 수 있는 적응형 상태 모델링 기술을 사용하고 있습니다.

- **Technical Details**: C-VUE는 세 가지 핵심 기술로 구성되어 있습니다: (1) 비디오 인식을 위한 긴 역사를 모델링하는 기법, (2) 공간 중복 감소 기법으로 비효율적인 역사 모델링을 개선, (3) 프레임 가중치 손실을 포함한 병렬 훈련 구조로 다양한 스케일의 사건을 이해합니다.

- **Performance Highlights**: C-VUE는 NVIDIA Jetson Orin, Apple MacBook 및 Intel PC(NVIDIA 4090 GPU 기반)와 같은 전형적인 엣지 장치에서 30 FPS 이상의 속도로 실행되며, 정확도 측면에서 모든 기준선 모델을 초과 달성했습니다. 세부적으로는 Transformer에서 1.2%~4.7%, LSTM에서 2%~3%, MoviNetA0에서 14.5%~67.4% 향상되었습니다.



### ChitroJera: A Regionally Relevant Visual Question Answering Dataset for Bangla (https://arxiv.org/abs/2410.14991)
- **What's New**: 본 논문에서는 Bangla 언어를 위한 대규모 Visual Question Answering (VQA) 데이터셋인 ChitroJera를 소개합니다. 이 데이터셋은 15,000개 이상의 샘플로 구성되어 있으며, 다양한 현지 데이터 소스를 활용하여 문화적 관련성을 강조합니다.

- **Technical Details**: ChitroJera 데이터셋은 다양한 텍스트 인코더(text encoders), 이미지 인코더(image encoders), 멀티모달 모델(multimodal models), 그리고 새로운 dual-encoder 모델을 평가하는 데 사용되었습니다. 실험 결과, 사전 훈련된 dual-encoder가 다른 모델들보다 우수한 성능을 보임을 확인했습니다.

- **Performance Highlights**: 대형 언어 모델(large language models, LLMs)을 활용한 프롬프트 기반 기법에서 LLM들이 가장 뛰어난 성능을 기록했습니다. 기존 데이터셋의 미비한 상태를 고려할 때, ChitroJera는 Bangla에서 시각-언어 작업의 범위를 넓힐 것으로 기대됩니다.



### SeaS: Few-shot Industrial Anomaly Image Generation with Separation and Sharing Fine-tuning (https://arxiv.org/abs/2410.14987)
- **What's New**: 본 논문에서는 산업 환경에서의 이상 탐지를 위해 이상 이미지 생성을 위한 새로운 방법인 Separation and Sharing Fine-tuning (SeaS)를 제안합니다. 이 방법은 적은 수의 이상 및 정상 이미지를 사용하여 다양한 이상 이미지를 생성하는 데 초점을 맞추고 있습니다.

- **Technical Details**: SeaS 방법은 Unbalanced Abnormal (UA) Text Prompt를 사용하여 제품과 이상을 별도로 설명하는 토큰을 생성합니다. 또한 Decoupled Anomaly Alignment (DA) 손실 및 Normal-image Alignment (NA) 손실을 통해 이상 특성과 제품 특성을 학습합니다. 이를 통해 U-Net과 고해상도 VAE 기능을 융합하여 정밀 픽셀 단위의 주석을 생성할 수 있습니다.

- **Performance Highlights**: MVTec AD 및 MVTec 3D AD 데이터셋에서 실험을 통해, SeaS는 이상 이미지 생성을 위해 각각 1.88, 1.95의 IS 점수와 0.34, 0.30의 IC-LPIPS 점수를 달성하였으며, 이를 기반으로 한 다운스트림 분할 작업에서 IoU가 각각 11.17%, 15.49% 증가하는 성능 향상을 입증했습니다.



### D-SarcNet: A Dual-stream Deep Learning Framework for Automatic Analysis of Sarcomere Structures in Fluorescently Labeled hiPSC-CMs (https://arxiv.org/abs/2410.14983)
Comments:
          Accepted for oral presentation at IEEE International Conference on Bioinformatics and Biomedicine 2024 (IEEE BIBM 2024)

- **What's New**: 이 논문은 D-SarcNet이라는 새로운 이중 스트림 딥 러닝 프레임워크를 소개합니다. 이 프레임워크는 형광으로 표지된 hiPSC-CM 단일 세포 이미지를 입력으로 받아, 사르코메어(structure organization)의 성숙 단계를 1.0에서 5.0까지의 연속 값으로 출력합니다.

- **Technical Details**: D-SarcNet은 Fast Fourier Transform (FFT), 딥 러닝 기반의 지역 패턴, 기울기 크기를 통합하여 이미지에서 글로벌(global) 및 로컬(local) 정보를 동시에 포착합니다. 이 프레임워크는 ConvNeXt-Swin Transformer 아키텍처를 기반으로 하며, 사르코메어 구조의 명확한 시각적 특징을 학습할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, D-SarcNet은 Spearman 상관계수 0.868을 기록하여 이전의 최첨단 기법보다 3.7% 향상된 성능을 보여줍니다. MSE, MAE, R2 스코어 등 주요 성능 지표에서도 유의미한 향상을 입증하였습니다.



### DCDepth: Progressive Monocular Depth Estimation in Discrete Cosine Domain (https://arxiv.org/abs/2410.14980)
Comments:
          Accepted by NeurIPS-2024

- **What's New**: 이번 논문에서는 DCDepth라는 새로운 모노큘러 깊이 추정 프레임워크를 소개합니다. 기존의 픽셀 기반 깊이 추정을 초월하여, 본 접근 방식은 깊이 패치의 주파수 계수를 추정하는 것을 목표로 합니다.

- **Technical Details**: 모노큘러 깊이 추정(MDE)의 혁신으로, 주파수 도메인에서 깊이를 추정하는 방식으로 문제를 다룹니다. 이 과정에서 이산 코사인 변환(Discrete Cosine Transform, DCT)을 이용해 깊이 정보를 주파수 성분으로 분해하며, 저주파 성분은 전반적인 장면 구조를, 고주파 성분은 세부 사항을 나타냅니다.

- **Performance Highlights**: NYU-Depth-V2, TOFDC 및 KITTI 데이터셋에서 포괄적인 실험을 진행하였으며, DCDepth의 성능이 기존의 최첨단 방법에 비해 우수함을 입증하였습니다.



### 3D Multi-Object Tracking Employing MS-GLMB Filter for Autonomous Driving (https://arxiv.org/abs/2410.14977)
Comments:
          2024 International Conference on Control, Automation and Information Sciences (ICCAIS), November 26th to 28th, 2024 in Ho Chi Minh City

- **What's New**: 이번 논문에서는 MS-GLMB 필터의 개선된 버전을 제안합니다. 카메라 외에 LiDAR 센서를 통합하여 3D 다중 객체 추적을 수행하는 방법을 도입하였습니다. 이는 기존의 MV-GLMB 및 MV-GLMB-AB 필터와 달리, 카메라 간의 겹치는 시야(Field of View, FoV)가 필요하지 않으며, 더 넓은 적용 가능성을 가지고 있습니다.

- **Technical Details**: 우리는 새로운 LiDAR 측정 모델을 소개하고, 다중 카메라와 LiDAR를 활용한 다중 객체 측정 모델을 제안합니다. 이를 MS-GLMB 필터에 통합하여, LiDAR 데이터와 카메라의 탐지를 효과적으로 처리하며, 겹치는 FoV 없이도 3D에서 다중 객체 추적을 향상시킵니다. 우리는 Bayes 회귀를 사용하여 카메라와 LiDAR의 정보를 융합합니다.

- **Performance Highlights**: 실험 결과, 우리 방법은 기존의 MS-GLMB 기반 방법들과 비교하여 추적 성능에서 유의미한 개선을 보였습니다. 특히, nuScenes 데이터셋에서 성능 향상이 두드러졌습니다. 이로 인해, 우리가 제안한 방법은 보다 다양한 환경에서도 효과적으로 사용할 수 있을 것으로 기대됩니다.



### Reflexive Guidance: Improving OoDD in Vision-Language Models via Self-Guided Image-Adaptive Concept Generation (https://arxiv.org/abs/2410.14975)
Comments:
          The first two authors contributed equally

- **What's New**: 이 논문은 대규모 멀티모달 데이터를 기반으로 훈련된 대형 비전-언어 모델(LVLMs)의 Out-of-Distribution Detection (OoDD) 기능을 평가하고 분석합니다. 특히, LVLM의 신뢰성을 높이기 위한 새로운 접근법인 Reflexive Guidance (ReGuide)를 제안하여 OoDD 성능을 강화하는 방법을 제시합니다.

- **Technical Details**: Reflexive Guidance (ReGuide) 접근법은 두 단계로 이루어져 있습니다. 첫 번째 단계에서는 LVLM이 주어진 이미지에 대해 의미적으로 유사한 개념(근접 OoD)과 의미적으로 비유사한 개념(원거리 OoD) 두 그룹을 제안하도록 합니다. 두 번째 단계에서는 이 제안된 개념들을 보조 OoD 클래스로 활용하여 OoDD 성능을 향상시키려고 합니다. 이 연구는 이미지 입력을 활용하여 OoDD를 위한 정보 제공 텍스트를 생성하는 첫 번째 사례로, 모델에 구애받지 않고 광범위하게 적용 가능합니다.

- **Performance Highlights**: ReGuide는 오픈 소스 모델의 성능을 크게 향상시켰으며, 이로 인해 상용 모델과의 경쟁력을 갖추게 되었습니다. 특히, GPT-4o 모델은 ReGuide 적용 후 근접 OoD 데이터셋에서 퍼포먼스가 강화되었습니다. 이 연구는 LVLM을 통해 자동 생성된 이미지 적응 개념 제안을 이용하여 OoDD를 안내하는 방법의 효능을 강조합니다.



### Visual Navigation of Digital Libraries: Retrieval and Classification of Images in the National Library of Norway's Digitised Book Collection (https://arxiv.org/abs/2410.14969)
Comments:
          13 pages, 2 figures, 4 tables, Accepted to the 2024 Computational Humanities Research Conference (CHR)

- **What's New**: 노르웨이 국립도서관(NLN)의 1900년 이전 도서에서 이미지 검색 애플리케이션을 개발하고, 최신 이미지 임베딩 및 모델을 비교하여 시각적 문화유산의 분석 가능성을 높입니다.

- **Technical Details**: 이 연구는 Vision Transformer (ViT), Contrastive Language-Image Pre-training (CLIP) 및 Sigmoid Loss for Language-Image Pre-Training (SigLIP) 임베딩을 사용하여 이미지 검색 및 분류 작업을 수행합니다. SigLIP 임베딩이 CLIP 및 ViT보다 약간 더 우수하게 작동하며, 이미지 분류에서는 디지털화 파이프라인에서 이미지 데이터 세트 정리에도 기여합니다.

- **Performance Highlights**: SigLIP 기반의 애플리케이션은 정확한 이미지 검색에서 우수한 성능을 보여주며, 이미지 검색의 효율성을 높이는 데 성공했습니다. 이 연구는 또한 다양한 텍스트 및 이미지 데이터의 탐색 가능성을 향상시키는 방향으로 나아가는 중요한 기초 자료를 제공합니다.



### Neural Radiance Field Image Refinement through End-to-End Sampling Point Optimization (https://arxiv.org/abs/2410.14958)
- **What's New**: 이번 연구는 Neural Radiance Field (NeRF) 기술을 활용하여 이미지 품질을 향상시키기 위해 최적화된 샘플링 포인트를 제안합니다. 기존 NeRF의 고정 샘플링 포인트로 인한 아티팩트 문제를 해결하기 위한 새로운 접근 방식을 소개합니다.

- **Technical Details**: 제안된 방법은 MLP-Mixer 기반의 아키텍처를 NeRF 모듈에 통합하여, 각 광선의 샘플링 포인트를 동적으로 구성합니다. 특히, 중요한 객체 표면 근처에서 샘플링 포인트를 집중하여 아티팩트를 줄이는 것을 목표로 합니다. 이 방식은 엔드 투 엔드 학습이 가능하며, 기존 NeRF 방법론과 유사하게 작동합니다.

- **Performance Highlights**: 실제 Forward-Facing 데이터셋을 사용한 실험 결과, 제안된 방법이 전통적인 NeRF 방식 대비 아티팩트를 효과적으로 줄이고 이미지 품질을 개선하는 성능을 보였습니다.



### Part-Whole Relational Fusion Towards Multi-Modal Scene Understanding (https://arxiv.org/abs/2410.14944)
- **What's New**: 본 논문에서는 다중 모드 융합(multi-modal fusion)의 새로운 접근법인 Part-Whole Relational Fusion (PWRF) 프레임워크를 제안합니다. 이는 각 모드를 부분(part)으로, 전체 모드를 전체(whole)로 간주하여 이들을 관계적으로 융합하는 방식을 소개하고 있습니다.

- **Technical Details**: PWRF 프레임워크는 Capsule Networks (CapsNets)의 part-whole relational routing 기능을 활용합니다. 특히, Disentangled Capsule Routing (DCR)이라는 가벼운 버전을 사용해 각 모드에서 파생된 부분 모드를 전체 모드로 라우팅합니다. 이를 통해 modal-shared semantics와 modal-specific semantics를 생성하여 다중 모드 장면 이해(multi-modal scene understanding) 문제를 해결합니다.

- **Performance Highlights**: 여러 데이터셋에서 실시한 실험 결과, 제안된 PWRF 프레임워크는 Synthetic Multi-Modal (SMM) 시맨틱 세분화(semantic segmentation)와 Visible-Depth-Thermal (VDT) 핵심 물체 탐지(salient object detection)에서 우수한 성능을 보였습니다. 두 가지 작업을 통해 프레임워크의 일반화 가능성과 견고함이 입증되었습니다.



### Adversarial Score identity Distillation: Rapidly Surpassing the Teacher in One Step (https://arxiv.org/abs/2410.14919)
- **What's New**: 본 연구에서 제안하는 SiDA (Score identity Distillation with Adversarial Loss)는 훈련 데이터 없이 사전 훈련된 diffusion 모델만을 사용하여 생성 품질과 증류 효율성을 모두 향상시키는 새로운 방법론입니다. SiDA는 실제 이미지를 통합하고 적대적 손실(adversarial loss)을 도입하여, 보다 진짜와 유사한 이미지를 생성할 수 있도록 개선합니다.

- **Technical Details**: SiDA는 생성기(score network)의 인코더를 판별기(discriminator)로 활용하여, 생성 이미지와 진짜 이미지를 구별하는 능력을 증대시킵니다. 적대적 손실은 GPU 내 각 배치(batch)에서 배치 정규화(batch-normalization)한 후 기존 SiD 손실에 결합됩니다. 이를 통해 GPU 배치당 평균 '가짜성'을 픽셀 기반 손실에 통합하며, SiDA를 사용해 기존 모델을 수정하거나 처음부터 단일 단계(direct) 생성기를 증류합니다.

- **Performance Highlights**: SiDA는 처음부터 훈련할 때 이전 모델보다 평균적으로 약 10배 빠르게 수렴하며, 사전 증류된 SiD 생성기에서 파인튜닝(fine-tuning)할 때 원래 모델의 성능을 신속하게 개선합니다. SiDA는 CIFAR-10 및 ImageNet에서 각각 FID 점수 1.499, 1.396, 1.110을 달성하여 새로운 벤치마크를 설정했습니다.



### A Hybrid Defense Strategy for Boosting Adversarial Robustness in Vision-Language Models (https://arxiv.org/abs/2410.14911)
- **What's New**: 본 연구에서는 여러 공격 전략과 고급 머신러닝 기법을 통합하여 Vision-Language Models (VLMs)의 강인성을 크게 향상시키는 새로운 적대적 훈련 프레임워크를 제안합니다.

- **Technical Details**: 기존의 적대적 훈련 방법들은 FGSM, AutoAttack, DeepFool과 같은 모델을 사용하여 적대적 예제 생성에 초점을 맞추었으며, 고정된 왜곡 노름(fixed perturbation norms)이나 미리 정의된 공격 패턴(predefined attack patterns)과 같은 강력한 가정에 의존했습니다. 제안된 방법은 다양한 적대적 공격에 대해 VLM의 강인성을 크게 향상시킵니다.

- **Performance Highlights**: CIFAR-10 및 CIFAR-100과 같은 실제 데이터셋에서 실험한 결과, 제안된 방법은 모델의 강인성을 크게 향상시켰으며, 세밀하게 조정된 CLIP 모델은 적대적으로 왜곡된 이미지에서 43.5%의 정확도를 달성했습니다. 반면, 기준 모델은 4%에 그쳤습니다. 또한, 신경망 모델은 98%의 높은 정확도를 기록하였고, XGBoost 모델은 예측 작업에서 85.26%의 성공률을 달성했습니다.



### DRACO: Differentiable Reconstruction for Arbitrary CBCT Orbits (https://arxiv.org/abs/2410.14900)
- **What's New**: 이 논문에서는 차별화 가능한 이동 변환 필터링 역 투사(differentiable shift-variant filtered backprojection, FBP) 신경망을 사용하여 임의의 궤도로 원뿔빔 컴퓨터 단층촬영(cone beam computed tomography, CBCT) 이미지를 재구성하는 새로운 방법을 소개합니다.

- **Technical Details**: 기존의 CBCT 재구성 방법은 많은 계산 자원과 시간이 필요하지만, 제안된 방법은 특정 궤도 기하학에 적응하는 심층 학습 접근 방식을 사용하여 이러한 문제를 해결합니다. 이 접근법은 학습 모델에 미리 알려진 연산자를 통합하여 모델의 파라미터 수를 최소화하고 해석 가능성을 향상시킵니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 전통적인 반복 알고리즘에 비해 재구성 과정을 상당히 가속화하며, 평균 제곱 오차(mean squared error, MSE), 피크 신호 대 잡음 비율(peak signal-to-noise ratio, PSNR), 구조적 유사도 지수(measure, SSIM)와 같은 지표에서 비슷하거나 우수한 이미지 품질을 달성했습니다. 특히, 일반적인 궤도 재구성뿐만 아니라 원형과 호를 포함한 비연속 궤도의 분석적 재구성에도 사용할 수 있습니다.



### On the Influence of Shape, Texture and Color for Learning Semantic Segmentation (https://arxiv.org/abs/2410.14878)
- **What's New**: 이 연구는 기존의 DNN(Deep Neural Network) 편향 연구에서 벗어나, 이미지의 모양(shape), 질감(texture), 색(color) 정보 각각이 DNN 학습에 어떻게 기여하는지를 분석합니다. 또한, 다양한 cue(단서) 조합이 학습 성공에 미치는 영향과 이들 간의 시너지 효과를 조명합니다.

- **Technical Details**: 연구팀은 주어진 데이터셋을 단일 cue 또는 선택된 조합으로 분해하는 일반적인 절차를 개발하여, 시맨틱 세그멘테이션 작업을 수행했습니다. Cityscapes, PASCAL Context, CARLA 시뮬레이터를 기반으로 한 데이터셋을 활용하며, early fusion 및 late fusion 기법을 사용하여 pixel level에서 cue의 영향을 분석하였습니다.

- **Performance Highlights**: 실험 결과, 질감과 형태가 학습 성공에서 명확히 우위를 나타내지 않았으나, 형태와 색상을 조합한 경우 예상외로 강력한 결과를 보였습니다. 특히, CNN과 Transformer 아키텍처에서 cue 정보 추출 방식에서 큰 차이가 없음을 확인하였습니다.



### Improving Vision Transformers by Overlapping Heads in Multi-Head Self-Attention (https://arxiv.org/abs/2410.14874)
- **What's New**: 본 논문에서는 Vision Transformers에 대해 Multi-Overlapped-Head Self-Attention(MOHSA) 모듈을 도입하여 성능 향상을 달성할 수 있음을 보이고 있습니다. 이 접근 방식은 서로 인접한 두 헤드와 쿼리, 키, 값을 겹쳐서 Attention 계산을 가능하게 하여 정보 교환을 촉진합니다.

- **Technical Details**: MOHSA는 각 헤드의 쿼리, 키, 값이 인접한 헤드의 쿼리, 키, 값과 겹치도록 구성됩니다. 첫 번째와 마지막 헤드는 제로 패딩을 사용하고, 다양한 겹침 비율에 대한 패러다임이 제안되었습니다. 실험은 CIFAR-10, CIFAR-100, Tiny-ImageNet 및 ImageNet-1k와 같은 네 가지 벤치마크 데이터셋을 사용하여 진행됩니다.

- **Performance Highlights**: 제안된 MOHSA는 실험에서 다섯 가지 Transformer 모델을 사용하여 성능 향상을 명확히 입증하였으며, 대부분의 Vision Transformer 모델에 통합할 수 있어 성능을 높이는 데 큰 도움이 됩니다.



### SYNOSIS: Image synthesis pipeline for machine vision in metal surface inspection (https://arxiv.org/abs/2410.14844)
Comments:
          Initial preprint, 21 pages, 21 figures, 6 tables

- **What's New**: 이 논문은 약간의 데이터가 결여되기 쉬운 산업의 비주얼 검사 시스템을 위해 파라메트릭 합성 데이터셋 생성 방법을 도입합니다. 이 방법을 통해 실제 데이터의 한계를 극복할 수 있으며, 표면 검사 과정을 위한 종합적인 파이프라인을 제시합니다.

- **Technical Details**: 논문은 공정의 전환 과정에서 1) 표면 측정 및 파라미터화, 2) 물리적으로 정확한 합성 이미지 생성, 3) 합성 이미지 품질 평가 및 4) 기계 학습(ML)으로 표면 검사용으로 사용되는 과정을 포함한 파이프라인을 설명합니다. 이 연구에서는 샌드블라스트 및 밀링된 알루미늄 표면을 분석 대상으로 삼았습니다.

- **Performance Highlights**: 이 연구는 실제 및 합성 이미지 쌍으로 구성된 포괄적인 이중 데이터셋을 제공합니다. 결과적으로, ML 모델의 성능 향상 및 산업 현장에서의 적용 가능성에 대한 평가도 포함되어 있습니다.



### Automated Road Extraction from Satellite Imagery Integrating Dense Depthwise Dilated Separable Spatial Pyramid Pooling with DeepLabV3+ (https://arxiv.org/abs/2410.14836)
Comments:
          9 pages, 5 figures

- **What's New**: 본 연구는 DeepLabV3+ 아키텍처에 Dense Depthwise Dilated Separable Spatial Pyramid Pooling (DenseDDSSPP) 모듈을 도입하여 기존의 Atrous Spatial Pyramid Pooling (ASPP) 모듈을 대체하는 혁신적인 방법을 제시합니다. 이 변경은 위성 이미지에서 복잡한 도로 구조를 보다 효과적으로 추출할 수 있도록 돕습니다.

- **Technical Details**: DenseDDSSPP 모듈은 dilated convolution과 depthwise separable convolution의 이점을 결합하여 멀티 스케일(multi-scale)에서의 특징을 보다 효과적으로 캡처합니다. 또한, Squeeze-and-Excitation 블록을 통합하여 해당 모듈의 유용한 특징에 집중할 수 있도록 설계하였습니다.

- **Performance Highlights**: 제안된 모델은 여러 비교 메트릭을 통해 기존의 최첨단(state-of-the-art) 모델보다 우수한 성능을 보였으며, 도로 추출 정확성을 크게 향상시켰음을 입증했습니다.



### Tackling domain generalization for out-of-distribution endoscopic imaging (https://arxiv.org/abs/2410.14821)
Comments:
          The paper was accepted at Machine Learning in Medical Imaging (MLMI) workshop at MICCAI 2024 in Marrakesh

- **What's New**: 이 연구는 endoscopic 모달리티에서의 domain generalization (DG)을 다루며, polyp 및 Barrett's esophagus의 이분 세그멘테이션을 위한 새로운 프레임워크를 제안합니다. 기존의 DL 모델들이 자료 분포의 차이가 나면 성능 저하를 겪는 문제를 해결하기 위한 방안을 논의합니다.

- **Technical Details**: 제안된 방법은 ResNet 기반의 특징 추출 후, 스타일 정규화 및 되돌리기 블록(SRW 모듈)을 통해 스타일 변화를 제거하고, 중요한 정보를 보존합니다. 이를 통해 endoscopic 이미지의 도메인 감수성을 억제하고 구분 가능한 특징을 향상시킵니다. SNR 블록과 ISW 블록으로 구성된 SRW 모듈은 특징의 공분산을 활용하여 특정 도메인 정보를 파악하고, 데이터 불균형으로 인해 발생하는 문제를 최소화합니다.

- **Performance Highlights**: 제안된 방법은 EndoUDA 세트에서 DeepLabv3+에 비해 13.7% 향상되고, 최신 기술(State-of-the-Art) 방법보다 약 8% 개선된 성능을 보였습니다. 또한 EndoUDA Barrett's esophagus 데이터셋에서도 기준선 대비 19%, 최고의 상태(State) 기술 대비 6%의 성능 향상을 기록하였습니다.



### GESH-Net: Graph-Enhanced Spherical Harmonic Convolutional Networks for Cortical Surface Registration (https://arxiv.org/abs/2410.14805)
- **What's New**: 이 연구에서는 심층 학습(deep learning) 모델을 기반으로 한 비지도 cortex 표면 등록 네트워크를 설계하였습니다. 구체적으로, 구형 고조파 변환(spherical harmonic transformation)을 기반으로 한 컨볼루션(convolution) 메소드를 도입하여 cortex 표면 데이터를 등록합니다.

- **Technical Details**: GESH-Net은 비선형 등록(nonlinear registration)을 위해 설계되었으며, 다중 스케일 계층을 포함하는 여러 비선형 등록 모듈로 구성됩니다. 각 모듈은 이전 모듈에서 생성된 낮은 스케일 변형을 현재 스케일로 업샘플링하여 추가 등록을 수행합니다. GESH-Net은 U-Net 및 Softmax 레이어로 구성된 이산 등록 네트워크(discrete registration network)를 포함합니다.

- **Performance Highlights**: 그래프 주의(module) 기능을 통합하여 GESH-Net은 cortex 표면 데이터의 글로벌 특징(global features)을 효과적으로 학습할 수 있는 능력을 향상시켰습니다. 실험 결과, GESH-Net은 기존의 방법들에 비해 등록 성능에서 상당한 이점을 보였습니다.



### Deep Generic Dynamic Object Detection Based on Dynamic Grid Maps (https://arxiv.org/abs/2410.14799)
Comments:
          10 pages, 6 figures, IEEE IV24

- **What's New**: 이 논문에서는 자동 주행을 위한 일반적인 동적 객체 탐지 방법을 제안합니다. LiDAR 기반 동적 그리드를 실시간으로 생성하고, 이를 통해 다양한 동적 객체를 탐지하는 딥러닝 모델을 훈련합니다. 특히 Rotation-equivariant Detector(ReDet)를 활용한 점에서 기존의 통계 기반 방법보다 높은 탐지 성능을 보입니다.

- **Technical Details**: 본 연구는 회전 경량 박스(object detection task as rotated bounding box) 문제로 일반 동적 객체 탐지를 다룹니다. 기존의 카메라 이미지 기반 탐지 네트워크와 달리 회전 네트워크는 박스 각도를 예측할 수 있습니다. ReDet(Rotation-equivariant Detector)는 고성능 탐지를 위해 선택되었으며, 동적 자Occupancy grid maps를 입력으로 사용하여 동적 객체의 경량 박스를 출력합니다. 각 그리드 셀은 다중 가정의 기본 신념 질량(Dempster-Shafer basic belief masses) m을 포함하며, 이를 통해 정적 및 동적 상태를 구분합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존의 동적 셀 클러스터링 전략에 비해 거짓 긍정 탐지 비율(false positive object detection rate)을 크게 줄였습니다. 특히 움직이는 쇼핑 카트와 같은 훈련 데이터에 없던 동적 객체의 탐지에서도 인상적인 성과를 보였습니다.



### SSL-NBV: A Self-Supervised-Learning-Based Next-Best-View algorithm for Efficient 3D Plant Reconstruction by a Robo (https://arxiv.org/abs/2410.14790)
Comments:
          22 pages, 11 figures, 1 table

- **What's New**: 이번 논문에서는 3D 식물 재구성을 위한 새로운 접근 방식인 self-supervised learning 기반 Next-Best-View (SSL-NBV) 방법을 제안합니다. 이전의 방법들은 사전 훈련된 식물 모델을 요구하여 현실 세계의 식물에 적용하기 어려운 한계를 지니고 있었습니다.

- **Technical Details**: SSL-NBV는 심층 신경망(deep neural network)을 사용하여 후보 viewpoint에서의 정보 이득(information gain, IG)을 예측합니다. 이 방법은 로봇이 작업을 수행하는 동안 새로운 3D 센서 데이터를 이전에 수집한 데이터와 비교하여 자신의 훈련 데이터를 수집할 수 있도록 합니다. 또한, 약한 지도학습(weakly-supervised learning)과 경험 재현(experience replay)을 적용하여 효율적인 온라인 학습(online learning)을 가능하게 합니다.

- **Performance Highlights**: 종합적인 평가 결과 SSL-NBV는 비-NBV 방법에 비해 식물 재구성을 위한 시점(view) 수가 적고, voxel 기반 방법보다 800배 이상 빠른 성능을 보였습니다. SSL-NBV는 DL-NBV 기준에 비해 훈련 주석(training annotations)을 90% 이상 줄였으며, 온라인 미세 조정(online fine-tuning)을 통해 새로운 시나리오에 적응할 수 있는 능력을 보여주었습니다. 또한, 실제 식물을 사용하여 3D 식물 재구성을 위한 새로운 시점을 효과적으로 계획할 수 있음을 증명하였습니다. 가장 중요한 점은 SSL-NBV가 전체 네트워크 훈련을 자동화하며, 변화하는 농업 환경에서의 지속적인 온라인 학습을 가능하게 한다는 것입니다.



### A Survey on Computational Solutions for Reconstructing Complete Objects by Reassembling Their Fractured Parts (https://arxiv.org/abs/2410.14770)
Comments:
          36 pages, 22 figures

- **What's New**: 이 논문은 조각들로부터 완전한 객체를 재구성하는 문제에 대한 체계적인 설문조사를 제공합니다. 특히, 기존의 비딥 러닝 접근법에서 최근의 딥 러닝 접근법으로의 발전을 다루며, 다양한 알고리즘, 데이터 세트, 오픈 소스 소프트웨어 패키지 및 응용 프로그램을 포함합니다.

- **Technical Details**: 재조합 문제는 개별 조각의 속성을 이해하고 서로 다른 조각 간의 일치를 수립해야 합니다. 이 과정은 shape segmentation, shape matching, 그리고 shape priors 학습과 밀접하게 관련되어 있습니다. 이 논문은 비딥 러닝 및 딥 러닝 기반 기술 모두를 포함하여 단일 조각과 다중 조각의 분석을 위한 기법을 논의합니다.

- **Performance Highlights**: 이 연구는 단편 복원, 생물학 및 고생물학에서의 응용 프로그램을 포함하여 재조합 문제의 다양한 응용 프로그램을 다룹니다. 제안된 다양한 방법은 3D 형태 생성 모델을 활용하여 다양한 형태 컬렉션에서 형태 사전(prior)을 학습할 수 있도록 하며, 기존 시스템은 수작업으로 설계된 기능에 제한되어 있었던 점을 개선합니다.



### Tokens on Demand: Token Condensation as Training-free Test-time Adaptation (https://arxiv.org/abs/2410.14729)
Comments:
          18 pages, 7 figures

- **What's New**: 이 논문에서는 Token Condensation as Adaptation (TCA)를 소개합니다. TCA는 비전-언어 모델 (VLMs)이 테스트 시간 추론 중 겪는 분포 변화(distribution shifts)를 완화하기 위해 설계된 비훈련(training-free) 방식입니다.

- **Technical Details**: TCA는 <cls> 토큰에 대한 낮은 주의(attentiveness)를 보이는 이미지 토큰을 응축(condensing)하여 패치 레벨에서 분포 간극(distribution gaps)을 연결합니다. TCA는 역사적 데이터 스트림에서 특정 대상 클래스(target classes)와 정렬된 가장 신뢰할 수 있는 <cls> 토큰을 식별하고 추적합니다. 이를 위해, TCA는 불확실성이 가장 낮은 토큰을 '앵커(anchors)'로 보존하기 위한 컨텍스트 토큰 저수지(context token reservoir, CTR)를 제안합니다. CTR에서 샘플링한 앵커를 사용하여 TCA는 클래스와 무관한 토큰을 깎고(pruning) 나머지 클래스 애매한 토큰을 대표 센터로 병합(merging)합니다.

- **Performance Highlights**: TCA는 테스트 시간 적응(test-time adaptation)에서 토큰 효율성을 탐구하는 최초의 방법으로, 크로스 데이터셋(cross-dataset) 및 분포 외(out-of-distribution) 적응 작업에서 일관되게 우수한 성능을 보입니다. GFLOPs를 12.2%에서 48.9%까지 줄이면서 가장 강력한 기준선(baseline) 대비 정확도(accuracy)를 최대 21.4% 향상시킵니다.



### Animating the Past: Reconstruct Trilobite via Video Generation (https://arxiv.org/abs/2410.14715)
- **What's New**: 이 연구에서는 트릴로바이트 행동을 정적 화석에서 재구성하는 새로운 자동화된 텍스트-비디오(T2V) 프롬프트 학습 방법을 제안합니다. 이 방법은 시각적 사실성과 부드러움을 평가할 수 있는 보상을 사용하여 동영상 생성 모델을 조정합니다.

- **Technical Details**: 본 연구에서는 9,088개의 Eoredlichia intermedia 화석 이미지로 훈련된 대형 언어 모델(LLM)을 활용하여 텍스트에서 비디오로의 변환을 위한 프롬프트를 생성합니다. 이 모델은 디퓨전 모델을 사용하여 트릴로바이트 애니메이션 세그먼트를 생성하고, 생성된 비디오는 사실성과 연속성을 평가하여 LLM을 업데이트 하는 방식으로 작동합니다.

- **Performance Highlights**: 실험 결과, 새로 제안된 방법은 기존의 강력한 기준을 초과하는 고휘도와 사실성을 가진 트릴로바이트 비디오를 생성하며, 이는 과학적 이해를 증진시키고 대중의 참여를 촉진할 것으로 기대됩니다.



### G2D2: Gradient-guided Discrete Diffusion for image inverse problem solving (https://arxiv.org/abs/2410.14710)
- **What's New**: 이 논문은 이산 확산 모델(Discrete Diffusion Model)을 기반으로 한 생성 모델을 사용하여 선형 역 문제를 해결하는 새로운 방법, Gradient-guided Discrete Diffusion (G2D2)를 제안합니다. 이는 최초로 이산 확산 모델을 역 문제 해결을 위한 우선 값으로 사용하는 것입니다.

- **Technical Details**: G2D2는 이미지 역 문제를 해결하기 위해 이산 확산 모델을 프라이어(prior)로 사용하며, 변분 분포(variational distribution)를 최적화하는 연속 완화(continuous relaxation) 기법을 활용하여, 이전 방법의 한계를 극복합니다. 또한, 별 모양의 노이즈 프로세스를 이용하여 기존 이산 확산 모델의 제한을 완화합니다.

- **Performance Highlights**: 실험 결과 G2D2는 표준 벤치마크 데이터셋을 이용한 성과 평가에서 기존 방법들과 비교하여 다양한 역 문제를 해결하는 데 가능성을 보였습니다. 특히, 이 방법은 훈련이 필요없는 이산 프라이어 기반의 동적 데이터 생성 모델의 응용 가능성을 보여주었습니다.



### FACMIC: Federated Adaptative CLIP Model for Medical Image Classification (https://arxiv.org/abs/2410.14707)
Comments:
          Accepted in MICCAI 2024

- **What's New**: 본 논문에서는 분산된 데이터에서 모델 훈련을 통한 데이터 프라이버시(privacy)를 보장하면서 의료 이미지 분석을 위한 새로운 연합학습(federated learning) 접근 방식을 제안합니다. 특히, CLIP 모델을 분산 회귀 적응(federated adaptive)으로 변형하여 효율적인 분류(classification) 작업을 수행합니다.

- **Technical Details**: 제안된 Federated Adaptive Contrastive Language-Image Pretraining (FACMIC) 모델은 클라이언트의 데이터에 적합한 특징을 선택하기 위한 경량화된 특징 주의(feature attention) 모듈을 사용합니다. 또한, 클라이언트 간의 데이터 분포 차이를 줄이기 위한 도메인 적응(domain adaptation) 기법을 도입하여 모델의 효과성과 효율성을 향상시킵니다.

- **Performance Highlights**: 네 가지 공개 데이터셋에서의 실험 결과, FACMIC 모델은 실제 및 다중 출처의 의료 이미징 데이터 처리에 있어 기존의 최신 방법들보다 우수한 성능을 보였습니다. 특히, 뇌종양 및 피부암 분류 작업에서 높은 분류 성능을 달성하였습니다.



### Optimizing Parking Space Classification: Distilling Ensembles into Lightweight Classifiers (https://arxiv.org/abs/2410.14705)
Comments:
          Accepted for presentation at the International Conference on Machine Learning and Applications (ICMLA) 2024

- **What's New**: 이번 연구에서는 이미지 기반 주차 공간 분류를 위한 새로운 접근 방식으로, Teacher 모델을 통해 경량화된 Student 모델을 생성하는 방식을 제안합니다. 이를 통해 주차 공간 모니터링을 위해 필요한 데이터 전송량을 줄여 스마트 도시 인프라의 부담을 덜 수 있습니다.

- **Technical Details**: 제안된 방법은 복합적인 Convolutional Neural Network (CNN) 앙상블을 Teacher 모델로 사용하고, 이를 바탕으로 pseudo-labels를 활용하여 경량화된 Student 모델을 학습시킵니다. 경량화된 Student 모델은 edge devices에 직접 배포되며, 학습 과정에서 네트워크 하드웨어 요구 사항을 최소화합니다.

- **Performance Highlights**: 연구 결과, Student 모델은 Teacher 모델보다 26배 적은 파라미터 수를 가지고도 평균 정확도 96.6%를 기록하여 95.3% 정확도의 Teacher 모델을 초과했습니다. 이는 경량 모델이 실제 환경에서 충분한 정확도를 달성할 수 있는 가능성을 보여줍니다.



### Self-Supervised Keypoint Detection with Distilled Depth Keypoint Representation (https://arxiv.org/abs/2410.14700)
- **What's New**: 이 논문에서는 깊이 정보와 RGB 영상을 활용하여 키포인트(keypoint) 검출 성능을 향상시키기 위한 새로운 교차 모달 지식 증류 프레임워크인 Distill-DKP를 제안합니다. 이를 통해 기존의 비지도 학습 방법의 한계를 극복하고, 배경의 영향을 최소화하며 의미 있는 키포인트를 정확하게 검출할 수 있습니다.

- **Technical Details**: Distill-DKP는 두 가지 모델, 즉 깊이 기반의 teacher 모델과 이미지 기반의 student 모델로 구성됩니다. 훈련 과정에서 학생 모델은 깊이 기반 teacher로부터 주입된 embedding-level 지식을 통해 학습합니다. 키포인트 검출을 위해 ResNet 아키텍처와 업샘플링 기법을 사용하며, 핵심 모듈로는 Keypoint Detector, Edge Map Generator, Decoder가 있습니다. Cosine similarity loss를 최소화하여 학생 모델이 teacher 모델의 깊이 정보를 학습하도록 유도합니다.

- **Performance Highlights**: Distill-DKP는 Human3.6M 데이터셋에서 평균 L2 오차를 47.15% 줄였고, Taichi에서는 평균 평균 오차를 5.67% 감소시켰으며, DeepFashion 데이터셋에서는 키포인트 정확도가 1.3% 향상되었습니다. 이 결과들은 Distill-DKP가 기존의 비지도 방법보다 뛰어난 성능을 나타내고 있음을 보여줍니다.



### Deep Learning Enhanced Road Traffic Analysis: Scalable Vehicle Detection and Velocity Estimation Using PlanetScope Imagery (https://arxiv.org/abs/2410.14698)
- **What's New**: 본 논문은 PlanetScope SuperDove 위성 영상을 사용하여 차량 속도를 감지하고 추정하는 방법을 제안합니다. 이 방법은 복잡한 법적 제한 사항과 높은 비용이 있는 기존의 센서 및 UAV에 비해 저렴하고 확장 가능한 글로벌 차량 교통 모니터링 솔루션을 제공합니다.

- **Technical Details**: 우리는 RGB 밴드에서 차량의 궤적을 추적하는 Keypoint R-CNN 모델을 제안합니다. 이 모델은 밴드 간의 타이밍 차이를 활용하여 차량 속도를 추정합니다. 독일과 폴란드의 고속도로에서 드론 영상 및 GPS 데이터를 사용하여 검증을 수행하였습니다.

- **Performance Highlights**: 우리 모델은 GPS 데이터와 비교하여 평균 정밀도(Mean Average Precision) 0.53 및 속도 추정 오류 약 3.4 m/s를 달성했습니다. 드론 비교의 결과, 위성 데이터로부터 평균 속도가 112.85 km/h로 나타났고, 드론 영상에서는 131.83 km/h로 검증되었습니다.



### Deep Domain Isolation and Sample Clustered Federated Learning for Semantic Segmentation (https://arxiv.org/abs/2410.14693)
- **What's New**: 이번 연구는 참가자 간 데이터의 공변량 변화(covariate shifts)가 2D 세분화 작업에서 연합 학습(federated learning)의 수렴(convergence)에 미치는 영향을 처음으로 탐구하였습니다.

- **Technical Details**: 본 논문에서는 각 참가자가 여러 기저(feature) 도메인 분포의 혼합을 소유하는 보다 일반적이고 현실적인 프레임워크를 제안합니다. Deep Domain Isolation (DDI) 기술을 개발하여 모델의 기울기(gradient) 공간에서 이미지 도메인을 직접 분리합니다. 또한, 연합 가우시안 혼합 모델(federated Gaussian Mixture Model)을 사용하여 각 클래스의 샘플 기울기를 적합시키고, 서버 측에서 스펙트럼 클러스터링(spectral clustering)과 결합하여 분산화된 샘플 수준의 도메인을 분리합니다.

- **Performance Highlights**: 우리는 Sample Clustered Federated Learning (SCFL) 프레임워크를 통해 다양한 독립 모델을 학습시켰으며, 각 분산화된 이미지 도메인에 대해 하나의 모델을 생성하였습니다. 이러한 방법론은 Cityscapes 및 GTA5 데이터셋의 다양한 분할에서 상당한 성능 향상을 보여주었습니다. 실험은 EfficientVIT-B0 모델을 사용하여 진행되었습니다.



### MoRE: Multi-Modal Contrastive Pre-training with Transformers on X-Rays, ECGs, and Diagnostic Repor (https://arxiv.org/abs/2410.16239)
Comments:
          10 pages, 5 figures, 9 tables. Supplementary detail in Appendix. Code made available in Github for reproducibility

- **What's New**: 본 연구에서는 X-ray, ECG, 그리고 Radiology/Cardiology Report를 통합하는 새로운 다중 모달 대조 사전 훈련 프레임워크를 제안합니다. 이 접근법은 각각의 모달리티를 통합된 표현 공간으로 인코딩하여 진단 정확도를 향상시키고 포괄적인 환자 평가를 가능하게 합니다.

- **Technical Details**: 우리는 LoRA-Peft를 사용하여 LLM의 훈련 가능한 매개변수를 현저히 축소하고, Vision Transformer의 최근 선형 주의 드랍 전략을 도입하여 효율적인 주의 메커니즘을 구현합니다. 또한, 모달리티별 기능을 일관성 있는 임베딩으로 정렬하는 대조 손실을 사용합니다.

- **Performance Highlights**: 제안된 방법론을 통해 Mimic-IV, CheXpert, Edema Severity, PtbXl와 같은 다양한 다운스트림 데이터셋에서 최신 기술(SOTA)을 달성했습니다. 이 프레임워크는 복잡한 모달 간 관계를 잘 포착하고, 의료 진단에 강력한 견고성을 보여 앞으로의 다중 모달 학습 연구를 위한 기초를 제공합니다.



### Deep Radiomics Detection of Clinically Significant Prostate Cancer on Multicenter MRI: Initial Comparison to PI-RADS Assessmen (https://arxiv.org/abs/2410.16238)
Comments:
          20 pages, 4 figures, 4 tables

- **What's New**: 이번 연구에서는 임상적으로 중요한 전립선암(cancer) 감지를 위한 딥 라디오믹스(deep radiomics) 모델을 개발하고, 이를 Prostate Imaging Reporting and Data System(PI-RADS) 평가와 비교하여 여러 센터에서의 성능을 분석했습니다.

- **Technical Details**: 이 연구는 2010년부터 2020년까지 수집된 615명의 환자의 biparametric (T2W 및 DW) 전립선 MRI 시퀀스 데이터를 분석했습니다. 딥 라디오믹스 모델은 nnU-Net를 활용하여 전립선(segment) 분할을 수행하고, voxel-wise 라디오믹 특성 추출, Extreme Gradient Boost(Extreme Gradient Boosting) 분류, 종양 확률 맵을 csPCa 감지 맵으로 후 처리하는 과정으로 구성되었습니다. 5-fold cross-validation을 통해 학습이 이루어졌고, 성능 비교는 ROC 곡선 아래 면적(AUROC), 민감도(sensitivity), 특이도(specificity) 분석을 통해 수행되었습니다.

- **Performance Highlights**: 테스트 결과, PI-RADS에서 0.94의 AUROC와 94% 민감도 및 77% 특이도를 기록한 반면, 딥 라디오믹스 모델은 AUROC 0.91과 90% 민감도, 73% 특이도를 보였습니다. 두 모델의 성능 차이는 통계적으로 유의하지 않았으나, 병변 수준에서는 PI-RADS가 84%의 민감도를 보인 반면 딥 라디오믹스는 68%의 민감도를 기록하여 병변 수준의 성능은 PI-RADS에 미치지 못했습니다.



### Managing Bandwidth: The Key to Cloud-Assisted Autonomous Driving (https://arxiv.org/abs/2410.16227)
Comments:
          6 pages

- **What's New**: 이 논문은 자율주행차에서 클라우드를 활용할 수 있다는 새로운 관점을 제시합니다. 기존의 생각과는 달리, 클라우드는 시간 민감하고 지연이 중요한 컴퓨팅을 처리할 수 있는 효율적인 방법을 제공할 수 있습니다.

- **Technical Details**: 자율주행 차량(AV)은 고속 데이터 전송이 필요한 다양한 센서를 통해 주변 정보를 수집하고, 이러한 데이터를 처리하여 안전한 주행 결정을 내립니다. 본 논문은 AV의 서비스 구조를 정의하고, 서비스 수준 유틸리티 곡선을 도입하여 네트워크 대역폭을 효과적으로 할당하는 방법을 제안합니다.

- **Performance Highlights**: 클라우드 컴퓨팅을 활용함으로써, 자율주행차는 더 큰 머신러닝 모델을 보다 빠르고 정확하게 실행할 수 있으며, 이는 안전성 향상으로 이어집니다. 데이터 전송 지연을 감수하면서도 더 높은 정확도의 모델을 실행할 수 있음을 보여줍니다.



### Improve Vision Language Model Chain-of-thought Reasoning (https://arxiv.org/abs/2410.16198)
Comments:
          10 pages + appendix

- **What's New**: 이번 연구에서는 vision language models (VLMs)의 chain-of-thought (CoT) 추론 능력을 향상시키기 위한 접근 방식을 제안합니다. 기존의 훈련 방법들이 충분한 CoT 추론 데이터를 포함하지 못하고 있다는 점을 지적하며, 보다 정교한 응답이 필요한 추론 과제에 대한 일반화가 불충분하다는 것을 보여줍니다.

- **Technical Details**: 두 가지 접근 방식을 통해 문제를 해결하고자 하였습니다. 첫 번째로, GPT-4o 모델로부터 rationale을 증류하여 훈련 데이터를 풍부하게 만들고 VLM을 세부 조정(fine-tuning) 합니다. 두 번째로, 강화 학습(reinforcement learning)을 적용하여 추론 품질을 보정합니다. 이를 위해 모델이 생성한 reasoning chains의 긍정(정확) 및 부정(부정확) 쌍을 구성하여 평가하였습니다.

- **Performance Highlights**: 실험 결과, 기준 데이터셋에서 CoT 추론 능력이 유의미하게 향상되었으며, 직접적인 답변 예측에 대한 일반화 능력 또한 개선되었습니다. 이는 훈련 과정에서 상세한 rationale을 포함시키고 강화 학습을 활용하는 것이 VLM의 추론 능력을 강화하는 데 중요하다는 것을 강조합니다.



### Metric as Transform: Exploring beyond Affine Transform for Interpretable Neural Network (https://arxiv.org/abs/2410.16159)
Comments:
          22 pages, 20 figures, 3 tables

- **What's New**: 이번 연구에서는 인공지능 신경망에서 점곱( dot product ) 뉴런의 해석 가능성을 높이기 위해 거리 기반 매트릭스( metrics )를 탐구했습니다. 이는 주로 라디얼 베이시스 함수 네트워크에서 사용되는 유클리드 거리( Euclidean distance )와 대조됩니다.

- **Technical Details**: 연구에서는 다양한 형태의 매트릭스의 일반화를 다뤘으며, 이를 통해 MLP(MultiLayer Perceptron)와 CNN(Convolutional Neural Network)에서 아핀 변환( Affine transformation )과 유사한 성능을 보임을 발견했습니다. 매트릭스의 다양한 속성을 분석하고, 딥러닝 모델에서의 해석 가능성을 높이기 위한 신경망 구조를 개발했습니다.

- **Performance Highlights**: 매트릭스를 이용한 뉴럴 네트워크는 기존의 아핀 변환 기반의 구조에 비해 해석 가능성이 뛰어난 것으로 나타났으며, 적대적 예시( adversarial examples )를 이해하고 배제하는 데 효과적이라는 결과를 보여주었습니다.



### Pangea: A Fully Open Multilingual Multimodal LLM for 39 Languages (https://arxiv.org/abs/2410.16153)
Comments:
          52 pages, 27 figures

- **What's New**: Pangea는 다양한 39개 언어로 구성된 6백만 개의 지침 데이터셋 PangeaIns를 기반으로 한 다국어 다모드 대형 언어 모델(MLLM)입니다. 기존의 영어 중심 데이터셋과는 달리, 문화적 맥락과 다양한 언어를 포함하여 지식의 불균형 문제를 해결합니다.

- **Technical Details**: Pangea는 PangeaIns라는 데이터셋을 통해 다국어 및 다모드 훈련을 실시하며, PangeaBench라는 14개의 데이터셋으로 구성된 종합 평가 도구를 활용하여 모델의 성능을 정량적으로 측정합니다. xChat과 xMMMU 벤치마크를 통해 개방형 대화 및 다모드 사고 작업에 대한 평가를 진행합니다.

- **Performance Highlights**: Pangea 모델은 PangeaBench 데이터셋에서 기존의 오픈 소스 MLLM보다 평균 7.3 포인트(영어 작업) 및 10.8 포인트(다국어 작업) 더 뛰어난 성능을 보였습니다. 또한 Gemini-1.5-Pro 및 GPT4o와 같은 상용 LLM을 여러 작업에서 초과하며, 다국적 및 문화적 이해에서 우수한 성능을 나타냈습니다.



### Towards Combating Frequency Simplicity-biased Learning for Domain Generalization (https://arxiv.org/abs/2410.16146)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 이 논문은 데이터 기반의 관점에서 데이터 증강(data augmentation) 기술을 통해 모델의 학습행태를 조정하여 주파수 패턴에 대한 의존도를 줄이는 새로운 접근 방식을 제안합니다. 주파수 단축(Frequency Shortcuts)을 방지하기 위해 데이터셋의 주파수 특성(statistical structure)을 동적으로 조정하는 두 가지 데이터 증강 모듈을 소개합니다.

- **Technical Details**: 제안된 두 가지 모듈인 Adversarial Amplitude Uncertainty Augmentation (AAUA)와 Adversarial Amplitude Dropout (AAD)은 주파수 특정 성질을 효과적으로 변화시켜 모델의 특정 주파수 대역에 대한 과도한 의존을 억제하는 데 중점을 둡니다. 이를 통해 모델이 특정 주파수 대역의 존재에만 의존하지 않도록 학습 행동을 조정합니다.

- **Performance Highlights**: 제안된 방식은 다양한 도메인 일반화(Domain Generalization) 작업 및 데이터셋에서 평가되었으며, 기존의 최고 성능 방법들(State of the Arts, SOTAs)에 비해 우수한 성능을 보였습니다.



### An Explainable Contrastive-based Dilated Convolutional Network with Transformer for Pediatric Pneumonia Detection (https://arxiv.org/abs/2410.16143)
- **What's New**: 이번 연구에서는 아동 폐렴 진단을 위해 새롭게 개발된 Explainable Contrastive-based Dilated Convolutional Network with Transformer (XCCNet) 모델을 제안합니다. XCCNet은 희박한 데이터 문제를 해결하고, X-ray 이미지에서 지역적 및 전역적 특징을 효과적으로 캡처하는 데 초점을 맞추고 있습니다.

- **Technical Details**: XCCNet은 contrastive learning과 dilated convolution, transformer를 통합하여 chest X-ray (CXR) 데이터의 특징을 향상시킵니다. 또한, Preprocessing Module을 통해 저 방사선, 비가공 CXR를 개선된 이미지로 변환하며, Data Augmentation Unit을 통해 인위적인 데이터 생성을 통해 데이터 불균형 문제를 완화합니다.

- **Performance Highlights**: XCCNet은 네 가지 공개 데이터셋에서 14개 최신 폐렴 탐지 방법들을 능가하는 성능을 보여주었으며, 이는 임상적 적용 가능성을 강조합니다.



### Multimodal Flare Forecasting with Deep Learning (https://arxiv.org/abs/2410.16116)
- **What's New**: 이번 연구는 크로모스피어(Chromosphere)와 코로나(Corona)에서의 UV 및 EUV 방출이 플레어(Flare) 예측에서 보여주는 예측력을 포토스피어(Photosphere) 선 시각화 자력도(Magnetograms)와 비교한 데이터 기반 접근 방식을 제시합니다.

- **Technical Details**: 이 연구에서는 딥러닝(Deep Learning) 방법론을 사용하여 크로모스피어 및 코로나의 UV 및 EUV 방출과 포토스피어 자력도에서 추출된 특징의 비교를 진행합니다. 특정 EUV 파장들은 자력도와 동등하거나 더 우수한 구별력을 제공하며, 다중 모달 신경망(Multimodal Neural Networks) 구조가 단일 입력 모델보다 일관되게 성능이 우수함을 보였습니다. 또한, 모델은 전체 디스크 이미지와 포괄적인 플레어 이벤트 카탈로그를 사용하여 훈련되고 평가됩니다.

- **Performance Highlights**: 모델의 성능 개선은 모델 배깅(model bagging) 등의 조합 모델(Ensemble Model) 기법을 통해 이루어졌으며, 이는 서로 다른 모델의 특징 간의 상관관계를 활용하여 예측의 정확성을 높일 수 있는 가능성을 나타냅니다. 결과적으로, 이번 연구는 크로모스피어 및 코로나의 방출 정보를 활용한 플레어 예측의 가능성을 확인시켜 주며, M-Class 임계값 이상에서의 플레어 예측 성능을 더욱 향상시킬 수 있는 기반을 마련했습니다.



### Final Report for CHESS: Cloud, High-Performance Computing, and Edge for Science and Security (https://arxiv.org/abs/2410.16093)
- **What's New**: 이 논문에서는 Pacific Northwest National Laboratory의 LDRD 프로젝트인 'Cloud, High-Performance Computing (HPC), and Edge for Science and Security' (CHESS)에 대해 다루고 있습니다. CHESS는 과학적 워크플로우에서의 자동화를 가능하게 하는 새로운 연계 기능을 개발했습니다. 이 연구는 분산된 과학적 작업을 지원하는 클라우드, 고성능 컴퓨팅, 엣지 컴퓨팅 환경을 통합하는 데 중점을 두고 있습니다.

- **Technical Details**: CHESS는 실험 및 이론 주기(Theory-Experiment Cycle)의 자동화를 위해, 수치 해법(numerical solvers), 데이터 분석(data analytics), 그리고 머신 러닝(machine learning) 조합을 활용한 효율적인 워크플로우 작업 구성 및 실행을 목표로 합니다. 또한, AI 인식 서비스(AI-aware services)를 이용한 멀티모달 LLM 파이프라인(multi-modal LLM pipelines), 에러 보정 다차원 축소(error-bounded multi-modal dimensionality reduction), 고성능 그래프 신경망(graph neural networks) 학습, 분산 LLM 훈련 기법 등을 개발하였습니다.

- **Performance Highlights**: CHESS는 다음과 같은 성과를 보였습니다: 분산 워크플로우 응답 시간 1.28배에서 87배 향상; 고정밀 마이크로구조 인식 분류(+17% 절대적 정확도); 마이크로구조 인식 압축(10-12배 스피드업) 및 AI 기반 분산 서비스 개발. 또한, 효율적인 성능 병목 감지 기법을 통해 워크플로우의 속도를 크게 향상시키고, 데이터 중심의 HPC 및 그래프 분석에서 데이터 흐름을 가속화하는 새로운 기술을 개발했습니다.



### Diffusion Transformer Policy (https://arxiv.org/abs/2410.15959)
Comments:
          Preprint

- **What's New**: 이 논문은 기존의 로봇 정책 학습을 개선하기 위해 대규모 다중 모드 확산 트랜스포머 아키텍처를 사용하는 'Diffusion Transformer Policy'를 제안합니다. 기존의 소규모 액션 헤드 대신 대규모 트랜스포머 모델을 통해 연속적인 액션을 직접 디노이즈합니다.

- **Technical Details**: 제안된 모델은 대규모 교차 구현(embodiment) 데이터셋에서 연속적인 엔드 이펙터 액션을 효과적으로 모델링하고, 환경의 다채로운 액션 공간을 처리할 수 있습니다. Diffusion Transformer Policy는 대규모 트랜스포머의 스케일링 능력을 활용하여 다양한 데이터셋에서 정책을 효과적으로 일반화합니다.

- **Performance Highlights**: Diffusion Transformer Policy는 Maniskill2와 Calvin이라는 두 개의 대규모 시뮬레이션 데이터셋에서 기존 방법들에 비해 월등히 우수한 성능을 보여주었으며, Franka 플랫폼에서의 일반화 성능도 향상되었습니다.



### AI-Driven Approaches for Glaucoma Detection -- A Comprehensive Review (https://arxiv.org/abs/2410.15947)
- **What's New**: 이번 연구는 AI(인공지능), 특히 DL(딥러닝) 기술을 이용한 CADx(컴퓨터 보조 진단) 시스템의 발전을 다루고 있습니다. 이 시스템은 녹내장(Glaucoma) 조기 진단을 지원하며, 안전성, 신뢰성, 해석 가능성, 설명 가능성의 향상이 필요함을 강조하고 있습니다.

- **Technical Details**: CADx 시스템 개발에는 Computer Vision(CV), 전통적인 Machine Learning(ML), DL 및 하이브리드 접근 방식이 포함됩니다. 시스템은 여러 단계로 구성되며, 이미지 전처리, 세분화, 특징 추출 등을 포함합니다.

- **Performance Highlights**: 연구에서 342개의 중요 논문을 분석하여, 82개의 CV 및 전통적인 ML 기법, 203개의 DL 기법, 39개의 하이브리드 접근 방식을 포함한 연구 결과를 제시하였습니다. 특히, 최근 6년 간 DL 기술의 사용이 급증하며 CADx 시스템의 선택과 개선을 위한 연구 간극과 방향성을 제시하였습니다.



### Seismic Phase Picking (https://arxiv.org/abs/2410.15907)
- **What's New**: 본 연구에서는 증가하는 지진 감시소 수와 파형 데이터로 인해 수동 선택이 어려워진 지진 위상 선택을 위해 여러 자동화 방법을 탐구하였습니다. 전통적인 방법과 학습 기반 방법 모두를 활용하여 성능 분석을 수행했습니다.

- **Technical Details**: 서로 다른 지진 모니터링 방법에 대한 데이터 세트로 INSTANCE 데이터 세트를 사용했습니다. 이 데이터 세트에는 약 50,000회의 지진과 130,000회 이상의 잡음이 포함된 3축 파형이 포함되어 있습니다. AR-AIC 모델을 기반으로 한 전통적 방법과 U-net을 수정한 PhaseNet을 사용하여 선택 문제를 해결하려고 시도했습니다.

- **Performance Highlights**: EQTransformer는 기존의 AR Pick보다 높은 일반화 능력을 보여주었으며, 잡음이 강한 상황에서도 성능이 우수했습니다. 단일 기록에만 의존하는 방식에 비해 다중 기록 기반 위상 선택의 필요성을 강조합니다.



### TexPro: Text-guided PBR Texturing with Procedural Material Modeling (https://arxiv.org/abs/2410.15891)
Comments:
          In submission. Supplementary material is included at the end of the main paper (5 pages, 2 figures)

- **What's New**: TexPro는 텍스트 프롬프트를 기반으로 3D 메쉬의 고충실도 재질 생성을 위한 새로운 방법입니다. 기존의 텍스트 조건 텍스처 생성 방법과 달리, TexPro는 절차적 재질 모델링을 통해 다양한 텍스처 맵을 생성하여 물리 기반 렌더링(physical-based rendering)과 재조명(relighting)을 지원합니다.

- **Technical Details**: TexPro는 먼저 최신 텍스트-이미지 모델을 사용하여 입력 텍스트 프롬프트에 따른 다중 뷰 참조 이미지를 생성합니다. 이후, 최근의 미분 가능 절차적 재질을 통해 렌더링 기반 최적화를 수행하여 텍스처 맵을 생성합니다. 이를 통해 재질 분류 및 매칭을 개선하는 새로운 재질 에이전트를 도입하며, 다각적인 객체 인식을 통해 각 재질 부품을 이해합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 기존 최첨단(SOTA) 방법보다 우수한 성능을 보이며, 재조명의 능력 또한 유지함을 입증했습니다.



### Distributed Learning for UAV Swarms (https://arxiv.org/abs/2410.15882)
- **What's New**: 이번 연구에서는 무인 항공기(UAV) 군집에서 비독립적이며 동일 분포가 아닌(non-IID) 데이터 문제를 해결하는 연합 학습(Federated Learning, FL) 알고리즘의 통합을 제안합니다. 또한, 다양한 데이터 세트에서 여러 집계 방법의 성능을 평가합니다.

- **Technical Details**: 우리는 FedAvg, FedProx, FedOpt 및 MOON과 같은 최신 연합 학습 알고리즘을 UAV 군집에 적용하여 비독립적이고 동일 분포가 아닌 데이터의 도전 과제를 다루는 성능을 연구합니다. FedProx는 비독립적인 데이터 환경에서의 성능을 안정화하기 위해 지역 업데이트를 정규화하는 방법을 강조합니다.

- **Performance Highlights**: 모든 알고리즘이 IID 데이터에서 유사한 성능을 보였지만, non-IID 조건에서는 성능 감소가 두드러졌습니다. FedProx는 전체적으로 가장 안정적인 성능을 보였으며, 이는 지역 모델의 큰 편차를 줄이는 것이 중요하다는 점을 부각시킵니다.



### R2I-rPPG: A Robust Region of Interest Selection Method for Remote Photoplethysmography to Extract Heart Ra (https://arxiv.org/abs/2410.15851)
Comments:
          preprint

- **What's New**: 이 논문은 COVID-19 팬데믹 동안 원격으로 생명 신호를 측정할 수 있는 저비용, 확장 가능한 접근의 필요성을 강조하고 있습니다. 기존에 실험적 환경에서 수행된 원격 광혈관측(remote photoplethysmography, rPPG) 방법을 클리닉 환경에서 활용하기 위한 새로운 ROI 선택 방법을 제시합니다.

- **Technical Details**: 제안된 방법은 3D 얼굴 랜드마크와 환자의 머리 기울기를 기반으로 ROI를 선택합니다. 이 방법은 Plane-Orthogonal-to-Skin (POS) rPPG 방법과 결합되어, 응급의학과에서 호흡 문제가 있는 환자 비디오에 적용되었으며, 조명 변화나 움직임의 방해 요소가 있는 경우에도 견고성을 발휘합니다.

- **Performance Highlights**: 시스템은 2.60GHz CPU와 4GB RAM에서 실시간으로 작동하며, 실제 임상 환경에서 효과적으로 HR 추출이 가능함을 입증하였습니다. 특히 응급실의 환자를 대상으로 한 연구에서 그 효과성이 강조되었습니다.



### LiMTR: Time Series Motion Prediction for Diverse Road Users through Multimodal Feature Integration (https://arxiv.org/abs/2410.15819)
Comments:
          Accepted at the NeurIPS 2024 workshop Time Series in the Age of Large Models. Code available at this https URL

- **What's New**: 이번 논문에서는 LiDAR 데이터의 세밀한 지역적 특징을 활용하여 자율주행 차량의 도로 사용자 행동 예측 정확도를 높이고자 하는 새로운 멀티모달 접근 방식인 LiMTR 모델을 제안합니다.

- **Technical Details**: LiMTR 모델은 PointNet 기반 아키텍처를 사용하여 특정 도로 사용자의 LiDAR 데이터를 통합하며, 개인의 자세나 시선 방향과 같은 로컬 특징에 집중합니다. 이 모델은 Waymo Open Dataset을 통해 최소 평균 변위 오차(minADE)에서 6.20%, 평균 평균 정밀도(mAP)에서 1.58%의 성능 향상을 보였습니다.

- **Performance Highlights**: LiMTR 모델은 LiDAR 기능을 통합하여 자율주행 차량의 모션 예측 작업에서 효과적인 개선을 이끌어내었으며, 이는 더욱 안전한 자율주행 환경 구축에 기여할 것으로 기대됩니다.



### FusionLungNet: Multi-scale Fusion Convolution with Refinement Network for Lung CT Image Segmentation (https://arxiv.org/abs/2410.15812)
- **What's New**: 이번 연구에서는 FusionLungNet이라는 새로운 폐 이미지 분할 네트워크를 소개합니다. 이 네트워크는 ResNet-50 기반 인코더, Channel-wise Aggregation Attention (CAA) 모듈, Multi-scale Feature Fusion (MFF) 블록, self refinement (SR) 모듈, 및 여러 디코더로 구성된 다단계 구조를 가지고 있습니다. 또한, 우리는 LungSegDB라는 새로운 공개 데이터셋을 생성하였습니다.

- **Technical Details**: FusionLungNet은 세 가지 손실 함수를 조합하여 이미지 재구성 품질을 최적화합니다: SSIM, IOU 및 focal loss. CAA 모듈은 중요한 특성 채널을 강조하여 피쳐 맵의 특성 표현을 개선합니다. MFF 블록은 다양한 피쳐 수준에서 정보를 융합하여 샘플 간의 일관성을 촉진합니다.

- **Performance Highlights**: 우리의 방법은 IOU 점수 98.04를 달성하여 기존 방법을 초월하였으며, 분할 정확도에서 상당한 향상을 나타냈습니다. 이 연구는 CT 영상에서의 폐 질환 진단에 중요한 영향을 미칠 것으로 기대됩니다.



### Assisted Physical Interaction: Autonomous Aerial Robots with Neural Network Detection, Navigation, and Safety Layers (https://arxiv.org/abs/2410.15802)
Comments:
          8 pages,14 figures, ICUAS 2024

- **What's New**: 이 논문은 산업 환경에서 안전하고 자율적인 공중 물리적 상호작용을 위한 새로운 프레임워크를 제안합니다. 이 시스템은 신경망(neural network) 기반의 목표 탐지(target detection) 시스템과 안전하고 정밀한 조작을 위한 제어 장벽 함수(control barrier function, CBF) 기반의 컨트롤러로 구성됩니다.

- **Technical Details**: 제안된 파이프라인은 로봇 로컬라이제이션(robot localization), 관심 지점(localization of Points of Interest, PoI) 또는 목표(target) 로컬라이제이션, 안전한 내비게이션(safe navigation)의 세 가지 주요 구성 요소로 이루어져 있습니다. 목표 탐지 시스템은 어려운 환경에서 학습되고, 깊이(depth) 기능을 활용하여 목표의 자세(pose)를 추정합니다. 전체 탐지 프레임워크는 지연이 적은 엣지 컴퓨팅(edge computing)으로 오프로드됩니다.

- **Performance Highlights**: 제어기와 목표 탐지의 시뮬레이션된 평가 결과와 실제 환경에서의 탐지 성능 분석이 포함되어 있습니다. CBF 기반 컨트롤러는 UAV가 안전하게 목표에 접근할 수 있도록 하며, 물리적 접촉을 위한 정밀함을 보장합니다. 더불어 다양한 환경 조건에서 제안된 접근 방식을 강력히 평가할 수 있는 시뮬레이션 환경이 구축되었습니다.



### Generalizing Motion Planners with Mixture of Experts for Autonomous Driving (https://arxiv.org/abs/2410.15774)
Comments:
          7 pages, 3 figures

- **What's New**: 이 논문에서는 StateTransformer-2 (STR2)라는 확장 가능한 디코더 전용 모션 플래너를 소개합니다. 이 모델은 Vision Transformer (ViT) 인코더와 혼합 전문가(mixture-of-experts, MoE) 인과성 Transformer 아키텍처를 사용하여 모드 붕괴(modality collapse) 및 보상 균형 문제를 해결합니다.

- **Technical Details**: STR2는 자가 감독(self-supervised) 학습을 통해 복잡한 보상 구조를 모사합니다. 이 모델은 배치 처리된 GPU 병렬 모델 추론으로 시뮬레이션을 가속화하여 대규모 테스트 세트에서 효율성을 개선합니다. 또한 STR2는 800백만 개의 매개변수로 확장 가능하며, 두 개의 레이어 MLP 디코더를 포함합니다.

- **Performance Highlights**: NuPlan 데이터셋에서 실시된 광범위한 실험 결과, STR2는 다양한 테스트 세트와 닫힌 루프(closed-loop) 시뮬레이션에서 이전 방법들보다 더 나은 일반화 성능을 보여주었습니다. 특히 복잡한 사례 및 이전에 보지 못한 시나리오에서 성능 감소가 훨씬 적습니다.



### PALMS: Plane-based Accessible Indoor Localization Using Mobile Smartphones (https://arxiv.org/abs/2410.15694)
Comments:
          7 pages, 3 figures, accepted to the 14th International Conference on Indoor Positioning and Indoor Navigation (IPIN) 2024, Best Presentation Award

- **What's New**: PALMS는 기존의 실내 내비게이션 시스템의 한계를 뛰어넘는 혁신적인 방법을 선보입니다. 이 시스템은 오직 디자인된 평면도(floor plan)만을 이용하여 동적 위치 추정을 수행하며, 입장 전 환경의 정보 수집(fingerprinting)이 필요하지 않습니다.

- **Technical Details**: PALMS는 LiDAR 및 파티클 필터(particle filter)를 이용하여 실내에서의 위치를 동적으로 결정합니다. CES 제약 조건을 활용하고, 주된 방향을 매칭하여 위치의 공간 확률 분포를 생성합니다. 이는 특히 사용자 환경의 정확한 특징을 반영하여 더 나은 정확성과 빠른 수렴 시간을 구현합니다. 사용자 위치에 대한 프로브 능력은 3D 스캔 وقد بتقنية حتى يخدم العديد من التطبيقات.

- **Performance Highlights**: PALMS는 전통적인 방법에 비해 평균 RMSE가 6.7배 낮은 정확도를 보였으며, 1미터 오차 내에서의 위치의 비율은 4.9배 더 높았습니다. 이는 실내 내비게이션에 있어 훨씬 더 효율적인 접근법을 제공함을 나타냅니다.



### Transforming Blood Cell Detection and Classification with Advanced Deep Learning Models: A Comparative Study (https://arxiv.org/abs/2410.15670)
Comments:
          26 pages, 4884 Words, 17 Figures, 10 Tables

- **What's New**: 이 연구에서는 혈액 세포의 효율적인 탐지(detection) 및 분류(classification)를 위한 YOLOv10 모델을 로보플로우(Roboflow) 데이터로 훈련되었습니다. 특히 640x640 픽셀로 조정된 이미지와 다양한 에폭(epochs)에서 훈련을 실시했습니다.

- **Technical Details**: YOLOv10 모델은 MobileNetV2, ShuffleNetV2 및 DarkNet과 비교하여 실시간(real-time) 혈액 세포 탐지 및 분류에서 더 우수한 성능을 보였습니다. 훈련 에폭의 증가가 정확도(accuracy), 정밀도(precision), 재현율(recall)을 크게 향상시키는 것으로 나타났습니다.

- **Performance Highlights**: 이 연구에서 만든 새로운 세포 데이터셋은 오픈 소스로 제공되어 자동 혈액 세포 탐지 및 분류의 발전을 지원할 예정입니다. 연구 결과는 의료 진단의 혁신적인 변화를 보여줍니다.



### Calibration of ordinal regression networks (https://arxiv.org/abs/2410.15658)
- **What's New**: 이 논문에서는 순위 회귀(ordinal regression) 문제에서 모델의 신뢰성을 높이고 잘 조정된 예측을 제공하는 새로운 손실 함수인 ORCU(Ordinal Regression loss for Calibration and Unimodality)를 제안합니다. 이는 기존의 손실 함수가 잘 다루지 못한 신뢰성 조정(calibration)과 단일 극치(unimodality) 문제를 해결합니다.

- **Technical Details**: ORCU는 소프트 순서 부호화(soft ordinal encoding)와 레이블 스무딩 기반의 정규화(label-smoothing-based regularization)를 통합하여 예측 신뢰성이 클래스 간의 순서 관계를 준수하도록 보장합니다. 이는 순위를 인식하는 손실 함수를 통해 잘 조정된 신뢰 추정치와 단일 극치 분포를 생성합니다.

- **Performance Highlights**: 논문에서 제시된 ORCU는 세 가지 주요 순위 회귀 벤치마크에서 기존 방법들을 초월하는 최첨단(exstate-of-the-art) 조정을 달성하며, 정확도를 희생하지 않고도 신뢰할 수 있는 예측 결과를 제공합니다.



### Resource-Efficient Medical Report Generation using Large Language Models (https://arxiv.org/abs/2410.15642)
- **What's New**: 이 연구에서는 흉부 X-레이 이미지를 위한 자동 의료 보고서 생성을 위한 새로운 프레임워크를 제안합니다. 이 프레임워크는 비전 기반의 대형 언어 모델(LLM)을 활용하여, 경량화된(solution) 방식으로 기존의 방법들과 비교해 우수한 성능을 달성하고 있습니다.

- **Technical Details**: 제안된 방법은 비전 인코더, 대형 언어 모델, 매핑 네트워크로 구성됩니다. 의료 관련 CLIP(MedCLIP)을 사용하여 시각적 임베딩(prefix embeddings)을 추출하고, 이를 경량화된 매핑 네트워크를 통해 언어 모델의 공간으로 변환합니다. 여기서 prefix tuning을 사용하여 LLM을 미세 조정하지 않고 성능을 향상시킵니다.

- **Performance Highlights**: Qwen1.5 LLM이 GPT-2 모델보다 의료 보고서 생성에서 더 우수한 NLG 메트릭(예: Bleu 점수)을 기록했습니다. 제안된 방법은 이전의 대형 LLM 기반 솔루션보다 성능이 뛰어나며, 자원 효율성을 확인하였습니다.



### Towards Kriging-informed Conditional Diffusion for Regional Sea-Level Data Downscaling (https://arxiv.org/abs/2410.15628)
- **What's New**: 본 논문에서는 Kriging-informed Conditional Diffusion Probabilistic Model (Ki-CDPM)을 제안합니다. 이 모델은 기후 데이터의 공간 변동성을 포착하면서 미세한 특성을 보존하는 새로운 방법론입니다.

- **Technical Details**: 제안된 Ki-CDPM은 Kriging 보간법을 활용하여 고해상도 기후 변수를 예측합니다. Variogram-Based Regularization을 도입하여 지역적 과정에서의 공간 변동성을 캡처하고, 다운스케일된 데이터의 물리적 일관성을 강화합니다.

- **Performance Highlights**: 실험 결과, Ki-CDPM은 최신 머신러닝 모델들과 비교해 정확도가 우수하며, 실제 기후 데이터 세트에서의 성능이 향상되었습니다.



### Erasing Undesirable Concepts in Diffusion Models with Adversarial Preservation (https://arxiv.org/abs/2410.15618)
- **What's New**: 본 연구에서는 확산 모델(difussion model)에서 발견된 '적대적 개념(adversarial concepts)'을 식별하고 보존하는 새로운 방법을 제안합니다. 이 방법은 파라미터 변화의 영향을 가장 많이 받는 개념을 특정하여 제거하는 과정을 안정적으로 제어할 수 있도록 합니다.

- **Technical Details**: 이 연구 방법은 Stable Diffusion 모델을 사용하여 검증되었으며, 기존의 상태-of-the-art 방법들보다 불필요한 콘텐츠를 제거하면서도 다른 관련 없는 요소들의 무결성을 유지하는 데 우수한 성능을 보였습니다. 특히, 이 방식은 중요한 모델 파라미터 변경을 최소화하여 다른 개념에 대한 영향력을 줄입니다.

- **Performance Highlights**: 본 연구의 방법론은 경쟁 모델들과 비교했을 때 원치 않는 콘텐츠를 보다 효과적으로 제거함으로써 다른 비관련 요소의 형태를 유지하는 성능을 입증하였습니다. 또한, 저자들은 이 방식으로 모델의 효과성을 유지하면서도 안전성을 확보할 수 있음을 강조하였습니다.



### Topology-Aware Exploration of Circle of Willis for CTA and MRA: Segmentation, Detection, and Classification (https://arxiv.org/abs/2410.15614)
Comments:
          Participation technical report for TopCoW24 challenge @ MICCAI 2024

- **What's New**: 이번 연구에서는 Circle of Willis (CoW) 혈관 구조의 토폴로지를 평가하기 위해 CTA와 MRA 이미지를 통합된 프레임워크에서 탐색하는 새로운 방법을 제안합니다. TopCow24 데이터셋으로부터 125 쌍의 CTA-MRA 데이터를 통해 다중 클래스 CoW 레이블을 사용하며, 이를 통해 유니버설 데이터셋을 구성하였습니다. 이 방법은 CoW의 토폴로지 완전성을 높이고, 서로 다른 클래스 간의 구별력을 향상시키기 위해 topology-aware loss를 활용합니다.

- **Technical Details**: 제안된 방법은 독립적인 강도 전처리, 공동 재샘플링 및 정규화 과정을 거쳐 유니버설 데이터셋을 구성합니다. 또한, topology-aware loss를 통해 CoW의 토폴로지 완전성을 높이고, 클래스 간의 차별성을 개선합니다. 마지막으로, 같은 클래스 내의 연결성을 향상시키기 위해 보완적 토폴로지 인식 세분화 작업을 수행합니다.

- **Performance Highlights**: TopCow24 챌린지의 최종 테스트 단계에서 CTA-Seg-Task에서 2위, CTA-Box-Task에서 3위, CTA-Edg-Task에서 1위, MRA-Seg-Task에서 2위, MRA-Box-Task에서 3위, MRA-Edg-Task에서 2위를 달성하여 경쟁력 있는 성과를 보였습니다.



### P-YOLOv8: Efficient and Accurate Real-Time Detection of Distracted Driving (https://arxiv.org/abs/2410.15602)
- **What's New**: Distracted driving detection을 위한 새로운 머신러닝 모델인 P-YOLOv8이 제안되었습니다. 이 모델은 기존 모델의 CPU 및 메모리 사용의 한계를 해결하며, 실시간으로 높은 정확도를 유지합니다.

- **Technical Details**: P-YOLOv8 (You Only Look Once, version 8) 모델은 전이 학습(pretrained)된 YOLOv8 아키텍처를 기반으로 하며, 가벼운 모델 크기(2.84 MB)와 1,451,098개의 파라미터를 가진 효율성을 강조합니다. 이 모델은 22,424개의 이미지가 포함된 Distracted Driver Detection dataset을 활용하여, 99.46%의 높은 정확도를 기록하였습니다. 또한, this model은 개선된 bounding box 예측 기법과 anchor boxes의 효과적인 사용으로 detection precision과 speed를 향상시킵니다.

- **Performance Highlights**: P-YOLOv8은 기존의 VGG16, VGG19, ResNet과 같은 딥러닝 모델들에 비해 속도와 컴퓨팅 효율성에서 우수한 성능을 보여주며, resource-constrained devices에서 실시간 적용 가능성을 높였습니다.



### A Dual Process VLA: Efficient Robotic Manipulation Leveraging VLM (https://arxiv.org/abs/2410.15549)
Comments:
          10 page

- **What's New**: 새로운 연구에서는 로봇이 복잡한 작업을 수행하도록 지원하는 Vision-Language-Action (VLA) 모델에서, 처리의 효율성과 실시간 성능을 개선하기 위해 Dual Process VLA (DP-VLA)라는 계층적 프레임워크를 제안합니다.

- **Technical Details**: DP-VLA는 두 개의 하위 시스템으로 구성되어 있습니다: L-Sys2 (Large System 2 Model)와 S-Sys1 (Small System 1 Model). L-Sys2는 복잡한 추론과 의사결정을 담당하며, VLM (Vision-Language Models)을 활용하여 느린 주파수에서 동작하여 연산 오버헤드를 줄입니다. S-Sys1은 실시간 모터 제어 및 감각 처리를 다루면서 빠르고 정확한 작업 실행을 보장합니다.

- **Performance Highlights**: DP-VLA는 RoboCasa 데이터셋에서 실험을 통해 더 빠른 추론과 높은 작업 성공률을 보여주었고, 이전 VLA 접근방식에 비해 성능이 우수함을 입증했습니다. 이 연구 결과는 고급 로봇 응용 프로그램에 대한 확장 가능한 솔루션을 제공합니다.



### Lying mirror (https://arxiv.org/abs/2410.15521)
Comments:
          21 Pages, 8 Figures

- **What's New**: 새로운 연구에서는 "lying mirror"라는 전통적인 이미지를 변형하여 관찰자가 오해하도록 정보를 숨기는 전광학(optical) 시스템을 소개합니다.

- **Technical Details**: 이 시스템은 주어진 입력 정보를 일반적으로 보이는 패턴으로 변형하여 신호를 위장합니다. 이 과정은 최적화된 구조적 회절(surface와의) 상호작용을 통해 이루어지며, 디지털 컴퓨팅 없이도 비밀 데이터를 광학적으로 숨길 수 있습니다.

- **Performance Highlights**: 이러한 방안은 다양한 유형의 입력 이미지 데이터에 대해 내구성을 보였으며, 무작위 이미지 노이즈 및 임의의 회전, 이동, 크기 변화와 같은 적대적 변형에 강한 반응을 보였습니다. 480, 550, 600nm의 다중 파장 조명에서 구조적 마이크로 미러 어레이를 사용하여 실험적으로 검증되었습니다.



### Exploring Curriculum Learning for Vision-Language Tasks: A Study on Small-Scale Multimodal Training (https://arxiv.org/abs/2410.15509)
Comments:
          CoNLL BabyLM Challenge 2024 camera ready

- **What's New**: 이 논문은 Machine Learning의 제한된 데이터 환경에서 Curriculum Learning의 효과를 탐구하고, 여러 모델 유형 간의 성능 비교를 다룹니다. 특히, Multimodal 모델의 성능을 개선하기 위한 새로운 접근 방식을 제공합니다.

- **Technical Details**: 연구는 3가지 주요 변수를 평가합니다: (i) Curriculum Learning, (ii) 텍스트 전용 데이터로부터의 Pretraining, (iii) 모델 유형. 데이터를 쉽고 난이도에 따라 제시하는 Curriculum Learning 접근 방식을 사용하여 VLM(Vision-Language Models)의 성능 변화를 연구했습니다.

- **Performance Highlights**: Curriculum Learning이 Multimodal 평가에서 Non-Curriculum Learning 모델보다 성능 향상에 기여하며, 특히 텍스트 전용 Pretraining과 결합할 때 효과가 있는 것으로 나타났습니다. 텍스트 전용 작업에서는 더 적은 trainable parameters를 가진 모델에서 Curriculum Learning의 혜택이 나타났습니다.



### Multi-Layer Feature Fusion with Cross-Channel Attention-Based U-Net for Kidney Tumor Segmentation (https://arxiv.org/abs/2410.15472)
Comments:
          8 pages

- **What's New**: 본 연구에서는 CT 스캔 이미지를 통한 자동화된 신장 종양(renal tumor) 감지를 위한 향상된 U-Net 기반 모델을 제안합니다.

- **Technical Details**: 제안된 모델은 convolution layer 간 잔여 연결(residual connections)을 사용하고, 엔코더 블록 내에서 다층 기능 융합(multi-layer feature fusion, MFF)과 채널 간 주의(cross-channel attention, CCA)를 통합하며, MFF 및 CCA를 통해 파생된 추가 정보를 통해 증가된 스킵 연결(skip connections)을 포함합니다.

- **Performance Highlights**: 모델은 KiTS19 데이터셋에서 평가되었으며, 신장 분할(kidney segmentation)에서 Dice Similarity Coefficient (DSC) 0.97 및 Jaccard index (JI) 0.95를 달성했습니다. 신장 종양 분할에서 DSC 0.96 및 JI 0.91을 기록하며, 현재의 선도적 모델들을 능가하는 성능을 보였습니다.



### CROPE: Evaluating In-Context Adaptation of Vision and Language Models to Culture-Specific Concepts (https://arxiv.org/abs/2410.15453)
- **What's New**: CROPE라는 새로운 비주얼 질문 응답 벤치마크가 소개되었습니다. 이 벤치마크는 문화 특정 개념에 대한 지식을 평가하고 문화적 적응 능력을 정량적으로 분석하는 데 초점을 맞추고 있습니다.

- **Technical Details**: CROPE는 매개변수 지식(parametric knowledge)과 맥락적 지식(contextual knowledge)을 구분하여 평가하는 두 가지 유형의 지식을 특징으로 합니다. 이 논문에서는 현대 VLM들이 문화 특정 개념을 인식하고 이러한 개념에 적응할 수 있는지를 탐구하기 위해 다양한 조건에서 실험을 수행했습니다.

- **Performance Highlights**: 최신 VLM 모델들은 문화 특정 개념을 처리할 때 일반 개념보다 성능이 현저히 떨어지는 것으로 나타났습니다. 맥락 지식을 제공받았을 때도 많은 모델이 예상과 달리 성능이 악화되었고, 모델들이 문화적 맥락보다 하드 네거티브 개념을 구분하는 데 어려움을 겪고 있음을 발견했습니다.



### AttCDCNet: Attention-enhanced Chest Disease Classification using X-Ray Images (https://arxiv.org/abs/2410.15437)
- **What's New**: 최근 연구에서는 전통적인 의학적 방법 대신 심층 학습 기반 기법을 적용하여 흉부 X-ray 이미지 진단을 자동화하는 새로운 모델인 AttCDCNet을 제안하였다. 이 모델은 DenseNet121 구조를 개선하고 주의 메커니즘(Attention Mechanism)을 추가하여 중요한 부분에 집중하며, 불균형 문제를 해결하기 위해 focal loss를 손실 함수로 채택하였다.

- **Technical Details**: AttCDCNet은 DenseNet121 모델을 기반으로 하여, Attention Block 추가와 Depth-wise Convolution을 통해 파라미터 수를 줄여 경량화된 구조를 갖는다. 이 모델은 입력 흉부 X-ray 이미지에서 질병의 분류를 위한 전처리, 특징 추출 및 분류 단계로 나뉜다. 실험에서는 다양한 흉부 질병이 포함된 데이터세트를 이용하여 성능을 평가하였다.

- **Performance Highlights**: 제안된 AttCDCNet 모델은 COVID-19 방사선 사진 데이터세트에서 각각 94.94%의 정확도, 95.14%의 정밀도, 94.53%의 재현율을 기록하며, 기존의 DenseNet121 모델보다 뛰어난 성능을 보였다. 이 연구는 흉부 질병 진단의 최신 기법들이 어떻게 통합될 수 있는지를 보여준다.



### Discriminating image representations with principal distortions (https://arxiv.org/abs/2410.15433)
- **What's New**: 이 연구에서는 이미지 표현의 로컬 기하학을 비교하기 위한 새로운 프레임워크를 제안합니다. 이전 연구들은 주로 글로벌 구조에 기반하여 표현들을 비교했으나, 우리의 접근 방식은 로컬 기하학의 차이에 중점을 두고 있습니다.

- **Technical Details**: 로컬 기하학은 Fisher 정보 행렬(FIM)을 사용하여 양적화합니다. 이 행렬은 특정 자극 왜곡에 대한 민감도를 특징짓는 통계적 도구로 활용됩니다. 또한, 이 프레임워크를 통해 '주요 왜곡(principal distortions)' 쌍을 찾아 두 모델 간의 변동성을 극대화합니다.

- **Performance Highlights**: 제안된 방법은 초기 시각 시스템의 단순 모델 세트를 비교하고, 다양한 심층 신경망 모델에 적용하여 아키텍처 및 훈련 방식에 따라 발생하는 로컬 기하학의 차이를 밝혀냅니다. 이를 통해 복잡한 계산 모델 간의 정보 차이를 탐색하는 데 유용함을 시사합니다.



### IPO: Interpretable Prompt Optimization for Vision-Language Models (https://arxiv.org/abs/2410.15397)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 이번 논문은 비전-언어 모델에서 사람의 이해가 가능한 지능형 텍스트 프롬프트 최적화 기법인 IPO(Interpretable Prompt Optimizer)를 제안합니다. 이는 기존의 프롬프트 최적화 기술의 한계를 극복하고 보다 명확한 해석 가능성을 제공합니다.

- **Technical Details**: IPO는 대규모 언어 모델(LLM)과 대형 다중모드 모델(LMM)을 활용하여 동적으로 텍스트 프롬프트를 생성합니다. 특히 기존 방법들과 달리 프롬프트의 다양성과 효과를 개선하기 위해 Prompt Optimization Prompt를 도입하여 과거 프롬프트의 성과를 저장하고 활용합니다.

- **Performance Highlights**: 11개의 데이터셋에서 IPO는 기존의 gradient-descent 기반 프롬프트 학습 방법보다 10.29% 더 높은 정확도를 보였습니다. 또한, 생성된 프롬프트는 해석이 가능하며, CLIP 모델의 일반화 성능을 향상시킵니다.



### Explainability of Point Cloud Neural Networks Using SMILE: Statistical Model-Agnostic Interpretability with Local Explanations (https://arxiv.org/abs/2410.15374)
Comments:
          17 pages, 9 figures

- **What's New**: 이번 연구에서는 SMILE이라는 새로운 설명 가능성 방법을 제안하여 포인트 클라우드(point cloud) 모델에 적용했습니다. SMILE은 LIME을 기반으로 하여 수학적 거리(특히, Anderson-Darling 거리)를 통해 설명 가능성을 향상시킵니다.

- **Technical Details**: SMILE 방법은 Empirical Cumulative Distribution Function (ECDF) 통계적 거리를 통합하여 강력한 해석 가능성과 견고성을 제공합니다. 이는 다양한 커널 너비(kernel widths), 섭동 수(perturbation numbers), 클러스터 구성(clustering configurations)에서의 충실도 손실(fidelity loss), R2 점수(R2 scores), 견고성(robustness) 측면에서 우수한 성능을 보입니다. 또한, Jaccard 지수를 사용한 포인트 클라우드 데이터의 안정성 분석(stability analysis)을 도입하였습니다.

- **Performance Highlights**: 연구 결과, SMILE이 개발한 기법은 모델 안정성(stability) 분야에 대한 새로운 벤치마크를 제시하였고, '사람(person)' 카테고리의 분류에서 데이터셋 편향(bias)을 식별하며, 자율주행 및 로봇 공학과 같은 안전-critical 애플리케이션에서 보다 포괄적인 데이터셋의 필요성을 강조합니다.



### DynaVINS++: Robust Visual-Inertial State Estimator in Dynamic Environments by Adaptive Truncated Least Squares and Stable State Recovery (https://arxiv.org/abs/2410.15373)
Comments:
          8 pages, 7 figures. S. Song, H. Lim, A. J. Lee and H. Myung, "DynaVINS++: Robust Visual-Inertial State Estimator in Dynamic Environments by Adaptive Truncated Least Squares and Stable State Recovery," in IEEE Robotics and Automation Letters, vol. 9, no. 10, pp. 9127-9134, Oct. 2024

- **What's New**: 본 연구에서는 동적 환경에서 로버스트 비주얼 관성을 고려한 내비게이션 시스템인 DynaVINS++를 제안합니다. 기존의 비주얼 관성 내비게이션 시스템(VINS)에서는 갑작스럽게 동적인 물체가 발생했을 때의 문제를 다루는 데 한계가 있었습니다. DynaVINS++는 이를 극복하기 위해 적응형 절단 최소 제곱법(adaptive truncated least squares)과 불규칙 바이어스 일관성 검증(bias consistency check)을 도입했습니다.

- **Technical Details**: DynaVINS++ 프레임워크는 a) 동적 물체의 영향을 최소화하기 위해 이미지와 IMU 전적립 데이터(feature association)에서 얻은 정보를 사용해 절단 범위를 조정하는 적응형 절단 최소 제곱(adaptive truncated least squares) 방법을 채택하였고, b) 갑작스러운 동적 물체로 인해 발생하는 IMU 바이어스의 잘못된 추정값을 수정하기 위해 안정적인 상태 복구(stable state recovery) 및 바이어스 일관성 검증(bias consistency check)을 통해 검증된 방법입니다.

- **Performance Highlights**: 공개 데이터셋 및 실제 데이터셋에서 검증 결과, DynaVINS++는 갑작스러운 동적 물체가 포함된 동적 환경에서도 유망한 성능을 보여주었습니다.



### Improving 3D Medical Image Segmentation at Boundary Regions using Local Self-attention and Global Volume Mixing (https://arxiv.org/abs/2410.15360)
- **What's New**: 이번 연구에서는 3D 볼류메트릭 의학 영상 세분화(volumetric medical image segmentation)를 위해 지역(local) 및 전역(global) 의존성(dependency)을 명확히 포착하는 새로운 계층적 인코더-디코더 기반의 프레임워크(vMixer)를 제안합니다. 이 프레임워크는 로컬 볼륨 기반의 셀프 어텐션(self-attention)을 활용하여 고해상도에서 지역 의존성을 인코딩하고, 저해상도 기능 표현에서 전역 의존성을 캡처하기 위한 새로운 볼류메트릭 MLP-믹서(volumetric MLP-mixer)를 도입하였습니다.

- **Technical Details**: 제안된 vMixer는 고해상도 단계에서 지역 의존성을 캡처하기 위해 볼륨 기반의 셀프 어텐션(Swin attention)을 활용하고, 저해상도 단계에서 전역 의존성을 인코딩하기 위한 Global Volume Mixer(GVM) 블록을 소개합니다. 이러한 명시적 글로벌 및 로컬 표현의 활용은 장기적인 장기 경계를 더 잘 학습하는 데 기여합니다. 실험은 Synapse Multi-organ, MSD-Pancreas Tumor, MSD-Liver Tumor 등 세 가지 데이터셋을 통해 충분히 검증되었습니다.

- **Performance Highlights**: 제안된 방법은 HD95 평가 지표(Hausdorff distance)에서 최첨단 성능을 초과하는 3.82%의 절대 개선을 달성하였으며, MSD 간 및 췌장 종양 데이터셋에서도 유사한 개선 패턴이 나타났습니다. 특히, ZebraFish 3D 세포막 데이터셋에서 제한된 학습 데이터를 제공하는 환경에서도 제안된 vMixer 모델은 3D 세포 인스턴스 세분화 작업에서 뛰어난 전이 학습 능력을 보여주었습니다.



### Extensions on low-complexity DCT approximations for larger blocklengths based on minimal angle similarity (https://arxiv.org/abs/2410.15244)
Comments:
          Fixed typos. 27 pages, 6 figures, 5 tables

- **What's New**: 이번 논문에서는 이미지 및 비디오 코딩에서 핵심 도구인 이산 코사인 변환 (DCT)의 새로운 저복잡도 변형을 세 가지(16-, 32-, 64-포인트) 도입합니다. 이러한 변형은 기존 DCT 매트릭스의 행 간 각도를 최소화하여 기획되었습니다.

- **Technical Details**: 저자들은 각각의 DCT 변환에서 각 행의 각도를 최소화하는 방식을 사용하여 근사 변환을 설계하였습니다. 또한, 고속 알고리즘을 개발하여 성능과 계산 비용 간의 좋은 균형을 유지했습니다.

- **Performance Highlights**: 실험 결과, 제안된 변환들은 문헌에서 이미 알려진 DCT 근사 방법들에 비해 16, 32, 64 블록 길이의 경우 더 나은 성능을 보였고, 이러한 저복잡도 변환들은 이미지 인코딩에서도 효과적임을 입증했습니다.



### Automated Segmentation and Analysis of Cone Photoreceptors in Multimodal Adaptive Optics Imaging (https://arxiv.org/abs/2410.15158)
- **What's New**: 이번 연구는 적응형 광학 스캐닝 광학 검사기법(AOSLO)을 활용하여 망막의 원추세포(cone cell)의 정확한 탐지 및 분할(segmentation)을 위한 새로운 방법을 탐구했습니다. 저자들은 StarDist 및 Cellpose라는 두 가지 U-Net 기반 모델을 활용하여 원추세포의 형태, 면적 및 분포를 정밀하게 분석했습니다.

- **Technical Details**: 망막의 광수용체(photoreceptors)인 원추세포는 색상 인식과 밝은 빛에 민감하고, 막대세포(rod)는 저조도에서 감지하여 야간 시각을 가능하게 합니다. 연구에서는 11명의 정상 시력을 가진 참가자를 대상으로 AOSLO 이미지를 반자동으로 주석 처리한 데이터셋을 사용했습니다. StarDist 모델은 원추세포의 정밀한 위치를 찾기 위해 스타-볼록 다각형(star-convex polygon) 기법을 사용하고, Cellpose 모델은 이미지의 수평 및 수직 기울기를 기반으로 벡터 흐름 장(field)을 생성하여 각 원추세포의 픽셀을 그룹화합니다.

- **Performance Highlights**: StarDist 모델은 중심 황반(region of central fovea)의 IoU 0.67 및 Dice 점수 0.80, 주변 황반(parafovea)의 IoU 0.78 및 Dice 점수 0.88을 달성했습니다. Cellpose 모델은 계산된 이미지를 사용했을 때 IoU 0.52와 Dice 점수 0.68로 평가되었습니다. 전반적으로 이 연구는 원추세포의 정밀한 분석을 통해 망막 건강을 평가하는 데 기여할 수 있음을 보여주었습니다.



### Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling (https://arxiv.org/abs/2410.15143)
- **What's New**: 본 논문은 온라인 지속 학습(Online Continual Learning, CL)에 대한 새로운 접근법을 제안합니다. 기존의 CL 알고리즘들이 단일 에폭(single epoch) 훈련을 전제로 하고, 재생 메모리 크기를 제한하는 등의 제약을 두고 있는 반면, 이 연구는 연산량과 메모리 측면에서 공정한 비교를 위해 부동소수점 연산(FLOPs)과 총 메모리 크기를 사용하자고 주장합니다.

- **Technical Details**: 제안된 방법은 ‘adaptive layer freezing’이라는 기법을 포함하여, 정보가 적은 배치에 대해 네트워크의 계층을 업데이트하지 않음으로써 계산 비용을 줄입니다. 또, 모델이 적은 반복으로 무작위 샘플을 통해 학습할 수 있도록 하는 샘플 검색(memory retrieval) 방법도 제안되었습니다. 실험적으로 CIFAR-10/100, CLEAR-10/100, ImageNet-1K 데이터셋에서 성능을 검증하였습니다.

- **Performance Highlights**: 제안된 방법은 동일한 총 리소스 예산(total budget) 하에서 기존의 최첨단 방법들과 비교하여 현저히 우수한 성능을 보여주었습니다. 많은 고성능 CL 방법들이 정해진 FLOPs와 메모리 예산 하에서 경쟁력을 가지지 못하는 반면, 제안된 방법은 여러 기준에서 두드러진 성과를 나타냈습니다.



### EViT-Unet: U-Net Like Efficient Vision Transformer for Medical Image Segmentation on Mobile and Edge Devices (https://arxiv.org/abs/2410.15036)
Comments:
          5 pages, 3 figures

- **What's New**: 본 논문에서는 ViT 기반의 새로운 효율적인 의학 이미지 분할 네트워크인 EViT-UNet를 제안합니다. 이는 자원 제한 환경에서 높은 정확도를 유지하면서 계산 복잡성을 줄이는 데 중점을 두었습니다.

- **Technical Details**: EViT-UNet은 U자형 아키텍처를 기반으로 하며, 인코더, 디코더, 병목(layer), 스킵 연결을 포함합니다. 이 네트워크는 convolutional operations와 self-attention 메커니즘을 결합하여 효율성을 최적화합니다. 효율성과 계산 비용의 균형을 잘 유지합니다.

- **Performance Highlights**: EViT-UNet은 여러 데이터셋에서 높은 분할 정확도를 달성하였고, 일반적인 분할 프레임워크들을 능가하는 성능을 보였습니다. 자원 제한 환경에서도 우수한 적용 가능성을 입증했습니다.



### Pathologist-like explainable AI for interpretable Gleason grading in prostate cancer (https://arxiv.org/abs/2410.15012)
Comments:
          58 pages, 15 figures (incl. supplementary)

- **What's New**: 이번 연구에서는 전세계 남성에서 가장 흔한 암인 전립선암의 공격성을 평가하기 위한 새로운 데이터셋을 소개합니다. 이 데이터셋은 1,015개의 조직 마이크로어레이 코어 이미지로 구성되며, 54명의 국제 병리학자에 의해 주석이 달립니다.

- **Technical Details**: 이 연구는 Gleason 점수 예측을 위한 U-Net 아키텍처 기반의 AI 시스템을 개발하였습니다. 이 시스템은 병리학자의 용어를 활용하여 예측을 수행하며, 포스트 호크(post-hoc) 설명 가능성 방법을 우회합니다.

- **Performance Highlights**: Gleason 패턴 세분화 성능에서, 설명이 포함된 모델은 Dice score가 0.713 $	imes$ 0.003으로, Gleason 패턴에 직접 학습된 모델의 0.691 $	imes$ 0.010을 초과하는 성능을 보여주었습니다. 또한, 소프트 레이블(soft labels)을 사용하여 데이터의 내재적 불확실성을 캡처하였습니다.



### Quanta Video Restoration (https://arxiv.org/abs/2410.14994)
- **What's New**: 본 논문에서는 빠른 모션에서도 높은 품질의 비디오 복원을 가능하게 하는 새로운 알고리즘인 QUanta VIdeo REstoration (QUIVER)을 제안합니다. QUIVER는 전통적인 quanta 복원 방법의 핵심 아이디어를 통합하여 구성된 end-to-end trainable 네트워크입니다.

- **Technical Details**: QUIVER는 1비트 또는 몇 비트의 quanta 이미지 데이터를 대상으로 하며, 평균 모션 범위가 1~7픽셀이고 초당 2000프레임(fps)에서 영상 복원을 수행합니다. 기존의 quanta 복원 알고리즘과 비교하여 매우 높은 성능을 보이며, 새로운 데이터셋인 I2-2000FPS (초당 2000프레임의 고속 비디오 데이터셋)를 통해 훈련 및 테스트가 이루어집니다.

- **Performance Highlights**: QUIVER는 시뮬레이션 데이터와 실제 데이터 모두에서 기존 quanta 복원 방법에 비해 유의미한 성능 개선을 보여 주목받고 있습니다. 추가로, QUIVER는 고속 비디오 복원 분야에서 새로운 기준을 설정하고 있습니다.



### Non-Invasive to Invasive: Enhancing FFA Synthesis from CFP with a Benchmark Dataset and a Novel Network (https://arxiv.org/abs/2410.14965)
Comments:
          ACMMM 24 MCHM

- **What's New**: 본 논문에서는 Fundus Fluorescein Angiography (FFA) 이미지를 비침습적인 Color Fundus Photographs (CFP)에서 합성하는 새로운 접근법을 제안합니다. 다수의 질병 범주에서 FFA 합성을 탐구하는 데 초점을 맞추고 있으며, Diffusion-guided GAN을 사용하여 합성 이미지의 품질을 개선하고자 합니다. 또한, Multi-disease Paired Ocular Synthesis (MPOS) 데이터셋을 구축하여 다양한 망막 질환에 대해 데이터를 제공합니다.

- **Technical Details**: 이 연구는 GAN(Generative Adversarial Network)과 확산모델(Diffusion Models)을 결합하여 FFA 이미지를 합성하는 새로운 네트워크인 Diffusion-guided GAN을 제안합니다. 이 방법에서는 동적이고 적응적인 확산 프로세스를 디스크리미네이터(discriminator)에 도입하고, 병목(병명) 정보를 제공하는 카테고리 인식 표현 강화기(category-aware representation enhancer)를 추가하였습니다. 또한, 600개의 쌍 데이터와 함께 다양한 질병 범주를 포함한 MPOS 데이터셋을 구축하였습니다.

- **Performance Highlights**: 실험 결과, 우리의 FFA 합성 네트워크는 기존의 최첨단 방법들과 비교하여 더 나은 FFA 이미지를 생성할 수 있음을 입증했습니다. 나아가, 합성된 FFA 이미지들은 실제 CFP 이미지와 결합하여 다수의 망막 질환 진단 정확도를 향상시킨 것으로 나타났습니다. 이는 비침습적 방법을 통해 환자에게 주는 피해를 줄이고 보다 나은 진단 결과를 이끌어낼 수 있는 가능성을 보여줍니다.



### SemiHVision: Enhancing Medical Multimodal Models with a Semi-Human Annotated Dataset and Fine-Tuned Instruction Generation (https://arxiv.org/abs/2410.14948)
- **What's New**: 최근 의료 분야에서 Multimodal Large Language Models (MLLMs)의 한계점을 극복하기 위해 새로운 데이터셋 SemiHVision과 평가 벤치마크 JAMA Clinical Challenge를 도입했습니다. 이러한 접근법은 의료 진단 효능을 향상시키기 위한 것입니다.

- **Technical Details**: SemiHVision 데이터셋은 사람의 주석과 자동 증강 기법을 결합하여 의료 지식 표현 및 진단 추론을 개선합니다. PMC-Cambrian-8B-AN 모델은 2400 H100 GPU 시간을 활용하여 훈련되었으며, HuatuoGPT-Vision-34B와 Claude3-Opus의 성능을 초과했습니다. JAMA Clinical Challenge는 진단 추론을 평가하기 위해 설계된 새로운 벤치마크입니다.

- **Performance Highlights**: PMC-Cambrian-AN 모델은 JAMA Clinical Challenge 벤치마크에서 현재 최고의 성능을 기록하며, HuatuoGPT-Vision-34B 및 Claude3-Opus를 능가하는 진단 추론 능력을 보여줍니다.



### Water quality polluted by total suspended solids classified within an Artificial Neural Network approach (https://arxiv.org/abs/2410.14929)
Comments:
          42 pages, 8 figures and 2 tables

- **What's New**: 이번 연구는 고형물로 인한 수질 오염을 분석하기 위한 인공지능 신경망(artificial neural network) 프레임워크의 적용을 다룹니다. 전통적인 수질 오염 평가 방법이 시간과 자원을 많이 소모하는 데 비해, 본 연구에서는 데이터 기반 모델을 통해 문제를 해결하고자 했습니다.

- **Technical Details**: 모델은 총 용해 고형물(total suspended solids) 데이터셋을 활용하여 전이 학습(transfer learning) 접근 방식을 이용한 컨볼루션 신경망(convolutional neural network)을 통해 훈련되었습니다. 다양한 입력 변수에 따라 저, 중, 고 오염 수준을 정확히 예측하는 것을 목표로 합니다.

- **Performance Highlights**: 우리 모델은 예측 정확도가 높아 전통적인 통계적 방법보다 속도와 신뢰성 측면에서 우수한 성능을 보였습니다. 이 결과는 인공지능 신경망 프레임워크가 수질 오염의 실시간 모니터링 및 관리에 효과적인 도구로 활용될 수 있음을 시사합니다.



### Truncated Consistency Models (https://arxiv.org/abs/2410.14895)
- **What's New**: 이 논문에서는 Truncated Consistency Models (TCM)을 제안하여 확산 모델의 일관성을 훈련하는 새로운 접근 방식을 제시합니다. TCM은 초기 단계에서의 디노이징(denoising) 작업을 무시하고, 생성을 중심에 두는 방법으로 성능을 향상시킵니다.

- **Technical Details**: 논문은 PF ODE (확률 흐름 보통 미분 방정식)에서 초기 노이즈로부터 직접 데이터의 솔루션을 예측하는 일관성 모델의 훈련을 다룹니다. TCM은 전체 시간 범위 [0,T] 대신 자른 시간 범위 [t′,T]를 사용하여 훈련합니다. 두 단계의 훈련 절차를 통해 모델이 디노이징과 생성 작업의 균형을 유지하도록 합니다.

- **Performance Highlights**: CIFAR-10 및 ImageNet 64×64 데이터셋에서 TCM은 iCT 및 iCT-deep과 같은 최신 모델보다 더 나은 FID(Fréchet Inception Distance) 성능을 보였습니다. 특히, TCM은 동일한 네트워크 크기로도 경쟁력 있는 결과를 보여주며, 훈련 안정성 또한 향상시킵니다.



### A novel approach towards the classification of Bone Fracture from Musculoskeletal Radiography images using Attention Based Transfer Learning (https://arxiv.org/abs/2410.14833)
Comments:
          6 pages, 3 tables, 4 figures, submitted to 27th International Conference on Computer and Information Technology (ICCIT) to be held during 20-22 December, 2024

- **What's New**: 본 연구는 FracAtlas 데이터셋을 활용하여 X-ray 이미지를 통한 골절 분류에 대한 새로운 접근법을 제시합니다. 특히 attention 기반의 transfer learning 모델을 사용하여 기존의 CNN 모델의 한계를 극복하고 90% 이상의 정확도를 달성하였습니다.

- **Technical Details**: 본 연구에서는 4,083개의 뼈의 X-ray 이미지를 포함하는 FracAtlas 데이터셋을 사용하였습니다. InceptionV3 (CNN 모델)를 기반으로 하여 Bottleneck Attention Module (BAM)을 도입하여 성능을 향상시키고, 전송 학습(transfer learning)을 통해 골절 검출의 정확도를 개선하였습니다. 각 이미지는 수동으로 주석처리 되었으며, 골절의 세분화, 분류 및 위치를 식별하는 데 사용되었습니다.

- **Performance Highlights**: 이 연구는 FracAtlas 데이터셋을 통해 골절 분류에서 90.48%의 정확도와 90.57%의 정밀도를 달성하였습니다. 이는 AI와 medical imaging의 융합으로 인해 진단의 정확성을 획기적으로 향상시킬 수 있는 가능성을 보여줍니다.



### Medical AI for Early Detection of Lung Cancer: A Survey (https://arxiv.org/abs/2410.14769)
- **What's New**: 최신 동향: 폐암 진단에서 기계 학습의 한계를 극복하기 위해, 딥 러닝(deep learning) 기술이 점점 더 많이 도입되고 있으며, 이 기술은 폐 결절(pulmonary nodule) 탐지, 분할, 분류에서 뛰어난 성능을 보이고 있습니다.

- **Technical Details**: 기술적 세부사항: 전통적인 머신 러닝 기법(SVM, KNN 등)은 수작업으로 특징(feature)을 추출해야 하며, 복잡한 샘플 데이터를 처리하는 데 한계가 있습니다. 반면, CNN(Convolutional Neural Networks), RNN(Recurrent Neural Networks), GAN(Generative Adversarial Networks)과 같은 딥 러닝 모델은 이미지의 복잡한 특징을 자동으로 학습하고 분석하여 측정 정확도를 현저히 개선합니다.

- **Performance Highlights**: 성과 강조: 연구에 따르면, 최신 CAD(computer-aided diagnosis) 시스템은 기존의 머신 러닝 시스템보다 폐 결절의 분류 정확도를 크게 향상시키며, 조기 폐암 탐지 및 진단에서의 역할이 강화되고 있으며, 모델 해석 가능성도 향상되고 있습니다.



### CFTS-GAN: Continual Few-Shot Teacher Student for Generative Adversarial Networks (https://arxiv.org/abs/2410.14749)
- **What's New**: 이 논문에서는 GAN에 대한 Continual Few-shot Teacher-Student 기법(CFTS-GAN)을 제안하여, 과적합(overfitting)과 재난적 망각(catastrophic forgetting) 문제를 동시에 고려하고 있습니다.

- **Technical Details**: CFTS-GAN은 학생 모델이 이전 지식에 영향을 주지 않고 새로운 작업을 학습할 수 있도록 어댑터 모듈을 사용합니다. 교사 모델의 지식을 학생 모델에 증류(distillation)하고, Cross-Domain Correspondence (CDC) 손실을 사용하여 다양성을 높이고 모드 붕괴(mode collapse)를 방지합니다. 또한, 성능 향상을 위해 판별기의 동결(freezing) 전략을 활용합니다.

- **Performance Highlights**: 정성적 및 정량적 결과를 통해 CFTS-GAN은 이미지 합성의 다양성을 높이고, 강력한 기존 모델과 비교해 품질이 뛰어난 샘플을 생성함을 보여줍니다.



### SGLP: A Similarity Guided Fast Layer Partition Pruning for Compressing Large Deep Models (https://arxiv.org/abs/2410.14720)
Comments:
          20 pages

- **What's New**: 이 논문은 Deep Neural Network (DNN) 기반 네트워크를 리소스 제한이 있는 장치에 효과적으로 배포하기 위한 새로운 접근법인 Similarity Guided fast Layer Partition pruning (SGLP)를 제안합니다.

- **Technical Details**: 제안된 방법은 Centered Kernel Alignment (CKA)를 사용하여 사전 훈련된 네트워크의 다양한 레이어 간의 내부 표현을 평가하여 레이어의 중요성을 기반으로 계층을 제거하는 방식입니다. 또한 GradNorm을 채택하여 세그먼트별 레이어의 중요성을 평가하고, 이는 상세한 미세 조정 없이도 효과적으로 작동합니다.

- **Performance Highlights**: SGLP는 이미지 분류 및 대형 언어 모델(LLM)에서 기존 방법들보다 향상된 정확성과 계산 효율성을 demonstrated하여 리소스가 제한된 플랫폼에서 DNN을 배포하는 데 있어 더 효과적인 솔루션을 제공합니다.



### Rethinking VLMs and LLMs for Image Classification (https://arxiv.org/abs/2410.14690)
- **What's New**: 이 논문에서는 기본적인 이미지 분류 작업에서 Visual Language Models(VLMs)와 Large Language Models(LLMs)를 효과적으로 결합하는 방법에 대해 재평가했습니다. LLM 없이도 VLM이 객체 및 장면 인식에서 더 뛰어난 성능을 보일 수 있음을 발견했습니다.

- **Technical Details**: VLM과 VLM+LLMs의 성능을 비교하기 위해, 본 연구에서는 동일한 비전 인코더를 사용하는 모델들을 비교했습니다. LLM은 이미지의 텍스트 설명 또는 임베딩을 통해 시각 정보와 연결되었습니다. 추가적으로, 라우팅을 위한 경량화된 LLM을 훈련시켜 시각 작업을 가장 적합한 모델로 분배합니다.

- **Performance Highlights**: 이 경량화 모델은 250만 개 이상의 시각 작업과 모델 정확도 쌍으로 훈련되어, HuggingGPT를 초과하고 GPT-4V와 유사한 정확도를 보여주며 비용 효율성 또한 향상되었습니다.



### Brain-Aware Readout Layers in GNNs: Advancing Alzheimer's early Detection and Neuroimaging (https://arxiv.org/abs/2410.14683)
- **What's New**: 이 연구에서는 알츠하이머병(AD) 조기 진단을 위한 새로운 뇌 인지 기반 읽기 레이어(BA readout layer)를 제안하며, 이를 통해 Graph Neural Networks (GNNs)의 해석 가능성 및 예측 정확도를 향상시킵니다.

- **Technical Details**: BA readout layer는 기능적 연결성과 노드 삽입을 기반으로 뇌 영역을 클러스터링하여 복잡한 뇌 네트워크 특성을 포착할 수 있도록 GNN의 성능을 높입니다. 연구는 T1-weighted MRI, resting-state fMRI, FBB-PET 데이터를 이용하여 383명의 참가자에서 분석되었습니다. 이 데이터는 인지적으로 정상인과 전임상 AD 환자를 포함합니다.

- **Performance Highlights**: BA readout layer를 포함한 GNN은 Preclinical Alzheimer’s Cognitive Composite (PACC) 점수를 예측하는 데 있어 기존 모델들보다 월등히 높은 성능을 보여주었으며, 복원력과 안정성 또한 향상되었습니다. 더불어, 이 레이어는 인지 기능에 영향을 미치는 작업별 뇌 영역을 강조하여 해석 가능성을 높였습니다.



### EP-SAM: Weakly Supervised Histopathology Segmentation via Enhanced Prompt with Segment Anything (https://arxiv.org/abs/2410.13621)
Comments:
          10 pages, 7 figures

- **What's New**: 이 연구에서는 한정된 레이블 데이터 문제를 해결하기 위해 Weakly Supervised Semantic Segmentation (WSSS) 모델을 제안하였습니다. 새롭게 채택된 Enhanced Attention Dropout Layer를 통해 class activation map (CAM)와 Segment Anything Model (SAM) 기반의 pseudo-labeling을 결합하여 더 효과적인 병리 이미지 분석을 목표로 하고 있습니다.

- **Technical Details**: 제안된 방법은 1) Enhanced Attention Dropout Layer의 개선, 2) SAM의 성능 최적화, 3) 메모리 효율적인 방식으로 SAM을 fine-tuning하는 과정을 포함합니다. 이를 통해 초기 pseudo-labels 생성을 개선하고, 단일 그래픽 프로세서(GPU) 메모리 12GB로도 효율적인 훈련이 가능합니다.

- **Performance Highlights**: 전통적인 WSSS 방법들과 비교하여, 제안된 방법은 3개의 관찰된 유방암 데이터셋에서 우수한 성능을 보여주었으며, 기존의 CAM 기반 방법들보다 초기 pseudo-label 생성을 효율적으로 수행하였습니다.



New uploads on arXiv(cs.AI)

### HyperspectralViTs: Fast and Accurate methane detection on-board satellites (https://arxiv.org/abs/2410.17248)
Comments:
          13 pages, This work has been submitted for possible publication

- **What's New**: 이 연구에서는 하이퍼스펙트럼 (hyperspectral) 데이터 처리를 위한 새로운 머신러닝 아키텍처인 HyperspectralViTs를 제안합니다. 이는 특정 환경에서의 자율성을 높이고, 메탄 탐지 및 광물 식별과 같은 다양한 작업을 자동화할 수 있는 기회를 제공합니다.

- **Technical Details**: HyperspectralViTs는 데이터의 고차원 스펙트럼을 처리할 수 있도록 개선된 Transformer 기반 모델들입니다. 본 논문에서는 SegFormer 및 EfficientViT와 같은 최신 아키텍처에 대한 적응을 통해 두 가지 주요 작업 - 메탄 누출 탐지와 광물 식별 -을 수행합니다. 기존의 메탄 탐지 모델 대비 F1 스코어를 27% 이상 향상시키고, 추론 속도는 85.19% 개선되었습니다.

- **Performance Highlights**: 제안된 모델은 EMIT 센서에서 수집된 새로운 데이터셋에 대해 메탄 누출 탐지의 F1 스코어를 6.9% 향상시키고, 광물 식별 작업에서도 기본 모델 대비 3.5% 향상된 결과를 나타냈습니다. 더불어, EMIT 센서를 통해 수집된 데이터는 30초 만에 처리될 수 있어, 빠른 응답 시간이 필요한 위성 자율 작동에 적합합니다.



### Towards Reliable Evaluation of Behavior Steering Interventions in LLMs (https://arxiv.org/abs/2410.17245)
Comments:
          Accepted to the NeurIPS 2024 - Workshop on Foundation Model Interventions

- **What's New**: 최근 연구에서 representation engineering 방법이 모델의 행동을 효율적으로 조정하는 데 유망한 성과를 보였으나, 평가 프로세스가 주관적인 시연에 의존해왔다는 점에 주목합니다. 본 논문에서는 네 가지 평가 기준을 제안하며, 이를 통해 현재 방법들의 효과를 보다 객관적이고 정량적으로 측정하고자 합니다.

- **Technical Details**: 본 연구에서 제안하는 평가 파이프라인은 (i) 후속 작업과 유사한 맥락, (ii) 모델의 likelihood 반영, (iii) 서로 다른 목표 행동 간의 비교 가능성, (iv) 기준 비교를 가능하게 하는 네 가지 속성을 포함합니다. 이러한 접근을 통해, Contrastive Activation Addition과 Inference-Time Intervention의 두 가지 representation engineering 방법을 평가하여, 모델의 행동 조정에 있어 이들의 효과성을 정량적으로 분석합니다.

- **Performance Highlights**: 연구 결과로 나타난 바에 따르면, 기존에 보고된 정보를 고려할 때 일부 개입이 기대했던 것보다 효과적이지 않음을 발견하였습니다. 특히, 행동 조정의 성공 여부에서 개입이 증진하는 행동과 억제하는 행동을 명확히 구분하는 새로운 차원을 제시하며, 이는 기존 지표에서 간과된 중요한 통찰을 제공합니다.



### SELA: Tree-Search Enhanced LLM Agents for Automated Machine Learning (https://arxiv.org/abs/2410.17238)
Comments:
          The code is available at this https URL

- **What's New**: 이번 연구에서는 Tree-Search Enhanced LLM Agents (SELA)라는 혁신적인 에이전트 기반 시스템을 소개하며, Monte Carlo Tree Search (MCTS)를 활용하여 AutoML 프로세스를 최적화하는 방법을 제시합니다. 이 접근 방식은 기존의 Fixed pipelines의 한계를 극복하고, 머신러닝 문제를 해결하기 위한 더 효과적인 탐색을 가능하게 합니다.

- **Technical Details**: SELA 프레임워크는 머신러닝 파이프라인 구성을 트리 구조로 표현하며, 에이전트가 실험을 수행하고 전략을 반복적으로 개선할 수 있도록 돕습니다. MCTS는 새로운 전략을 탐색하고, 이전에 잘 알려진 전략을 개선하는 능력을 활용하여, 에이전트가 광범위한 의사 결정 공간을 효율적으로 탐색할 수 있게 합니다.

- **Performance Highlights**: 20개의 다양한 머신러닝 데이터세트를 비교 평가한 결과, SELA는 전통적인 AutoML 시스템 및 에이전트 기반 AutoML 접근 방식에 비해 65%에서 80%의 승률을 기록하며 우수한 성능을 보였습니다. 이는 에이전트 기반 전략이 AutoML에서 높은 잠재력을 가지고 있음을 나타냅니다.



### Few-shot In-Context Preference Learning Using Large Language Models (https://arxiv.org/abs/2410.17233)
- **What's New**: 이번 논문은 보상을 설계하는 것이 강화학습에서 중요한 구성 요소라는 점을 강조하고, 인공지능의 복잡한 행동을 개발하는 데 도움을 주기 위해 Reinforcement Learning from Human Feedback (RLHF) 접근 방식을 사용합니다. 저자들은 Large Language Models (LLMs)를 활용하여 인간의 선호를 코드로 변환함으로써 보상 학습의 비효율성을 줄일 수 있는 가능성을 연구합니다.

- **Technical Details**: 저자들이 제안한 In-Context Preference Learning (ICPL) 방법은 LLM의 기반을 활용하여 보상 함수를 학습하는 방법입니다. ICPL은 환경 컨텍스트와 작업 설명을 바탕으로 보상 함수의 집합을 합성하고, 생성된 정책에 대한 비디오의 인간 순위를 이용하여 보상 함수를 반복적으로 업데이트합니다.

- **Performance Highlights**: ICPL 방법은 합성된 선호를 사용하여 RLHF보다 몇 배 더 효율적임을 입증하고, 고정된 보상 함수 대신 선호를 사용하는 방법과도 경쟁력 있는 성능을 보입니다. 마지막으로, 논문은 ICPL이 합성 환경을 넘어 인간과 함께 작업할 때도 효과적으로 작동함을 보여줍니다.



### Responsibility in a Multi-Value Strategic Setting (https://arxiv.org/abs/2410.17229)
- **What's New**: 본 논문에서는 다중 에이전트 시스템에서 책임(attribution) 개념을 확장하여 다중 가치(multi-value) 설정에서 책임을 어떻게 부여하고 예측할 수 있는지를 제시합니다. 특히, 책임을 미리 예측하는 방법을 도입하여 행동 전략 선택에 도움을 주는 방안을 보여줍니다.

- **Technical Details**: 논문은 책임 속성을 두 가지로 정의합니다: 소극적 책임(passive responsibility)과 약한 책임(weak responsibility). 이를 통해 에이전트가 전략을 선택하기 전에 책임을 예측하여 부정적인 결과를 피할 수 있도록 합니다. 또한, 책임을 최소화하는 전략이 비지배적(non-dominated)이고 후회 최소화(regret-minimising) 전략과 어떻게 연결되는지를 설명합니다.

- **Performance Highlights**: 제시된 모델은 에이전트가 불확실한 상황에서 보다 나은 전략 선택을 할 수 있도록 돕고, 책임을 최소화하면서 가치와 일치하는 전략을 고르게 합니다. 이 모델은 다양한 실제 상황에 적용 가능할 것으로 기대됩니다.



### Creativity in AI: Progresses and Challenges (https://arxiv.org/abs/2410.17218)
Comments:
          44 pages

- **What's New**: 이 논문은 AI의 창의성에 대한 최근 연구 동향과 주요 성과, 현재 남아 있는 과제를 살펴봅니다. AI 시스템의 창의적 문제 해결 능력, 언어적 창의성, 예술적 창의성, 그리고 과학적 창의성을 중심으로 조사하였습니다.

- **Technical Details**: 최근의 AI 모델들은 시적이고 예술적으로 창의적인 작품인 시, 이미지, 음악을 생성할 수 있는 능력을 지니지만, 창의적 문제 해결 및 추상적 사고가 요구되는 작업에서는 어려움을 겪고 있습니다. 이러한 모델들은 다양한 데이터에 기반한 대규모 파라미터를 사용하여 훈련되었으며, 그 결과가 진정한 창의적 과정인지, 아니면 메모리와 보간(interpolation) 기술의 결과인지에 대한 의문이 제기되고 있습니다.

- **Performance Highlights**: AI 모델은 언어와 예술에서 창의적 산출을 내는 데 능숙하지만, 창의적 문제 해결 및 장기적 일관성 결여 등에서의 한계를 보여줍니다. 또한, 생성된 결과물은 다양성과 독창성 부족, 그리고 헛소리(hallucination) 문제로 어려움을 겪고 있습니다. AI의 창의성을 평가하기 위한 포괄적인 평가 측정의 필요성도 강조됩니다.



### Language Model Non-myopic Generation for Reasoning and Planning (https://arxiv.org/abs/2410.17195)
- **What's New**: 본 논문은 LLM(대규모 언어 모델)의 계획(planning) 과정을 최적 제어(optimal control) 관점에서 재조명하고, 새로운 방법인 Predictive-Decoding을 제안합니다. 이 방법은 모델 예측 제어(Model Predictive Control)를 활용하여 계획 정확도를 향상시키고자 합니다.

- **Technical Details**: Predictive-Decoding은 LLM 분포를 예측 경로(foresight trajectories)에 기반하여 재조정(re-weight)하여 초기 오류를 줄이고 비근시(non-myopic) 계획을 촉진합니다. 실험을 통해 수학, 코딩 및 에이전트 작업에서 의미 있는 성능 향상을 보였습니다.

- **Performance Highlights**: Predictive-Decoding은 계산 효율성(computational efficiency)을 입증하였으며, 제한된 계산 자원으로도 검색 기준(search baselines)을 초월하는 성능을 보여줍니다. 이 연구는 LLM의 계획 능력을 최적화하는 데 중요한 통찰력을 제공합니다.



### Reinforcement learning on structure-conditioned categorical diffusion for protein inverse folding (https://arxiv.org/abs/2410.17173)
- **What's New**: 이 논문에서는 단백질 inverse folding(역접기) 문제를 해결하기 위해 RL-DIF라는 새로운 모델을 제안합니다. RL-DIF는 구조적 일관성을 최적화하기 위해 강화 학습(reinforcement learning)으로 조정된 카테고리 확산 모델(categorical diffusion model)입니다.

- **Technical Details**: RL-DIF는 CATH 4.2 데이터셋에서 관측된 단백질 구조와 서열을 기반으로 훈련됩니다. 본 논문은 모델이 대상 구조에 접합되는 다수의 비유사 서열을 생성할 수 있는 "foldable diversity"(접히는 다양성)를 측정하는 새로운 메트릭을 도입합니다.

- **Performance Highlights**: RL-DIF는 이전의 확인 모델들과 비교하여 29%의 foldable diversity을 달성했으며, 이는 동일 데이터셋에서 훈련된 모델의 23%에 비해 증가한 수치입니다. 모델의 구조적 일관성은 최신 inverse folding 방법들과 경쟁력 있는 수준을 유지하며, 성능이 향상됨을 보여줍니다.



### Trustworthy XAI and Application (https://arxiv.org/abs/2410.17139)
Comments:
          28 pages, 14 figures

- **What's New**: 이 논문은 Explainable Artificial Intelligence (XAI)의 중요성과 신뢰할 수 있는 XAI의 여러 실용적 응용을 탐구합니다. AI 시스템의 투명성, 설명 가능성, 신뢰성이라는 세 가지 주요 구성 요소를 강조하며, 인간과 AI 시스템 간의 신뢰 구축의 중요성을 논의합니다.

- **Technical Details**: AI는 복잡한 깊은 신경망(deep neural networks)을 사용하여 다양한 작업을 수행하지만, 그 결정 과정의 불투명성은 책임, 편견 및 정의 문제를 야기합니다. XAI 기법은 AI 예측 및 행동에 대한 설명을 제공하며, 신뢰성 있는 시스템을 위해 투명성과 해석 가능성을 개선하는 것을 목표로 합니다.

- **Performance Highlights**: AI는 자연어 처리(NLP), 컴퓨터 비전(computer vision), 기계학습(ML) 및 로봇공학에서 중요한 발전을 이루었으며, 다양한 산업에서 활용되고 있습니다. 데이터 기반의 전략을 통해 AI는 농업, 제조, 의료, 교육, 금융 등 여러 분야에서 혁신을 일으키고 있습니다.



### Permutation Picture of Graph Combinatorial Optimization Problems (https://arxiv.org/abs/2410.17111)
Comments:
          15 pages, 2 figures

- **What's New**: 본 논문은 순열 기반 표현을 사용하여 다양한 그래프 조합 최적화 문제들을 공식화하는 프레임워크를 제안합니다. 여기에는 외판원 문제(Travelling Salesman Problem), 최대 독립 집합(Maximum Independent Set), 최대 컷(Maximum Cut) 등의 문제들이 포함됩니다. 이 연구는 신경망 조합 최적화(neural combinatorial optimization)에서 알고리즘 설계의 새로운 길을 열 수 있습니다.

- **Technical Details**: 제안된 방법에서는 원래 최적화 목표에 대한 대리 손실 함수(surrogate loss function)를 구축하여 문제 공간의 표현을 학습하고, 이후 디코딩 알고리즘을 통해 최종 솔루션을 추출하도록 합니다. 이를 통해 보다 유연하고 효과적인 접근이 가능합니다. 특히, QUBO(Quadratic Unconstrained Binary Optimization)로부터 파생된 이징 모델(Ising model) 공식을 사용하여 대리 손실을 구축하지만, 순열 기반 접근은 자연스러운 문제 구조를 캡처하는 데 더 효과적입니다.

- **Performance Highlights**: 순열 표현은 조합 최적화 문제의 본질적인 순서 또는 배열을 결정하는 목표에 더 적합합니다. 특히, 외판원 문제와 같은 경우, 순열 표현을 통해 도시의 순서를 직접 인코딩할 수 있어 직관적입니다. 여러 다른 조합 최적화 문제에서도 같은 맥락으로 문제 구조와 밀접하게 연결됩니다.



### Deep Memory Search: A Metaheuristic Approach for Optimizing Heuristic Search (https://arxiv.org/abs/2410.17042)
Comments:
          10 pages, 6 figures

- **What's New**: 이 논문에서는 Metaheuristic 검색 방법을 메모리 중심의 프로세스로 모델링하여 새로운 접근법인 Deep Heuristic Search (DHS)를 제안하였습니다. DHS는 대규모 동적 검색 공간에서 메모리 기반 탐색-활용 메커니즘을 사용합니다.

- **Technical Details**: DHS는 통합 검색 전략, 다양한 깊이의 작업, 다중 깊이 메모리로 구성된 세 가지 주요 구성 요소로 구성됩니다. 검색 작업은 다양한 구현 수준을 가지며, 각 검색 작업은 확장 모드, 정상 모드 및 응축 모드의 세 가지 구현 수준을 가집니다.

- **Performance Highlights**: DHS는 다양한 휴리스틱 최적화 문제에 대한 검색 효율성과 성능을 크게 향상시키는 결과를 보였습니다.



### Insights on Disagreement Patterns in Multimodal Safety Perception across Diverse Rater Groups (https://arxiv.org/abs/2410.17032)
Comments:
          20 pages, 7 figures

- **What's New**: 본 연구는 다양한 인구 통계 집단에서 T2I(텍스트-이미지) 생성물의 안전성 인식을 분석하는 DICES-T2I 프레임워크를 소개합니다. 이 프레임워크는 AI 생성물에서의 피해 인식을 탐구하며, 다양한 관점을 통합하여 안전성 평가를 수행하는 중요성을 강조합니다.

- **Technical Details**: 연구는 630명의 평가자들로부터 수집된 약 1000개의 텍스트-이미지 생성물에 대한 안전성 평가를 바탕으로 진행되었습니다. 다양한 인구 통계적 그룹이 어떻게 AI 생성물의 피해를 평가하는지 논의하였고, 평가자들은 나이, 성별, 인종을 포함한 30개의 교차 그룹에 걸쳐 균형있게 분포하였습니다.

- **Performance Highlights**: 1) 인구 통계적으로 서로 다른 집단 간의 안전성 평가가 유의미하게 다르며, 각 집단은 특정 유형의 안전성 위반에 대해 다르게 평가합니다. 2) DICES-T2I는 전문가 평가자와의 차이를 보여주며, 특히 편향 유형에 대해 많은 전문가들이 무해하다고 평가한 반면, 다양한 평가자들은 위험하다고 인식했습니다. 3) 25%의 이미지가 전문가들에 의해 안전하다고 평가되었으나, 다양한 평가자들에 의해서는 위험한 것으로 간주되었습니다.



### Hybrid Generative AI for De Novo Design of Co-Crystals with Enhanced Tabletability (https://arxiv.org/abs/2410.17005)
Comments:
          Accepted at 38th Conference on Neural Information Processing Systems (NeurIPS)

- **What's New**: 이번 연구에서는 GEMCODE라는 자동화된 co-crystal 디자인 파이프라인을 제안합니다. 이 방법은 AI 기술과 진화적 최적화를 하이브리드하여 목표 화학 공간을 더 폭넓게 탐색할 수 있도록 합니다.

- **Technical Details**: GEMCODE는 딥 제너레이티브 모델과 이진 최적화 알고리즘을 결합하여 약물 물질의 정해진 기계적 특성을 가진 co-crystal을 빠르게 설계할 수 있게 합니다. 175만 개의 화학 구조 데이터셋을 기반으로 몇 가지 최첨단 생성 모델을 훈련하고, 이들을 coformer에 대한 고급 데이터셋으로 세밀하게 조정했습니다.

- **Performance Highlights**: GEMCODE는 실제적인 계산 제약 하에서도 효과적인 성능을 보이며, 약물 개발을 가속화할 수 있는 새롭고 이전에 알려지지 않은 많은 co-crystal을 생성했습니다. 예를 들어, 플라스틱성이 향상된 co-crystal 결합을 위한 최적의 분자 조합을 제시합니다.



### An Eye for an AI: Evaluating GPT-4o's Visual Perception Skills and Geometric Reasoning Skills Using Computer Graphics Questions (https://arxiv.org/abs/2410.16991)
Comments:
          8 pages, 8 figures, 1 table, to be published in SIGGRAPH Asia 2024 Educator's Forum

- **What's New**: 이 연구는 최신 LMM인 GPT-4o가 CG(Computer Graphics) 질문을 해결하는 능력을 평가하고 GenAI(Generative Artificial Intelligence)를 CG 교육에 통합할 수 있는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 연구는 시각적 인식(visual perception)과 기하학적 추론(geometric reasoning)에 필요한 다양한 CG 질문 세트를 사용하여 GPT-4o의 성능을 평가합니다. 연구는 두 가지 데이터세트(CG_TEST 및 CG_TEST2)를 사용하여 LMM의 시각적 처리 능력을 텍스트 처리 능력과 비교합니다.

- **Performance Highlights**: GPT-4o는 시각적 정보가 포함된 질문을 독립적으로 해결하는 데 큰 잠재력을 보였지만, 생성된 결과의 정확성과 품질에서는 여전히 주요한 한계가 있습니다. 본 연구는 CG 교육자가 GenAI를 교육에 통합할 수 있는 몇 가지 새로운 접근 방식을 제안하며, 이는 CG 교실에서의 학습 및 참여를 촉진하기 위한 것입니다.



### Order Matters: Exploring Order Sensitivity in Multimodal Large Language Models (https://arxiv.org/abs/2410.16983)
- **What's New**: 이 연구는 Multimodal Large Language Models (MLLMs)가 멀티모달 입력의 순서에 민감하다는 새로운 발견을 제시하며, 이를 통해 MLLM 성능 향상에 기여할 수 있는 전략을 제안합니다.

- **Technical Details**: MLLM에서 멀티모달 컨텍스트의 순서를 변경했을 때 성능의 변동이 발생하며, 특히 입력의 시작과 끝에 중요한 이미지/텍스트 내용을 배치하면 성능이 평균 14.7% 향상되는 것을 발견했습니다. 또한, Position-Invariant Accuracy (PIA)라는 새로운 메트릭을 제안했습니다.

- **Performance Highlights**: 제안한 방법은 비디오-캡션 매칭 및 시각 질문 응답 과제에서 각각 14.7% 및 17.8%의 성능 향상을 보였습니다. MLLM의 순서 민감성을 활용하여 중요한 콘텐츠를 특수 위치에 배치함으로써 성능을 개선할 수 있음을 보여주었습니다.



### Revealing Hidden Bias in AI: Lessons from Large Language Models (https://arxiv.org/abs/2410.16927)
Comments:
          13 pages, 18 figures. This paper presents a technical analysis of bias in large language models, focusing on bias detection and mitigation

- **What's New**: 이 연구는 Claude 3.5 Sonnet, GPT-4o, Gemini 1.5, Llama 3.1 405B와 같은 대형 언어 모델(LLMs)이 생성한 후보 인터뷰 리포트에서 성별, 인종, 나이와 같은 개인적 특성에 대한 편향(bias)을 분석했습니다. 연구 결과, LLM 기반 익명화(anonymization)가 특정 편향을 줄이는 데 효과적이지만 효과의 정도는 모델과 편향 유형에 따라 다르다는 사실을 밝혀냈습니다.

- **Technical Details**: 이 연구는 1,100개의 이력서를 사용하여 모든 이력서와 그에 맞는 직무 설명을 생성하였습니다. 후보자의 CV는 Claude 3.5 Sonnet을 통해 익명화된 방식과 비익명화된 방식으로 처리되었으며, 4개의 LLM을 사용하여 인터뷰 리포트를 생성했습니다. 편향 감지 모델은 0에서 2까지의 점수로 각 리포트 섹션의 편향 정도를 평가합니다.

- **Performance Highlights**: Llama 3.1 405B는 전체적으로 가장 낮은 편향을 보였으며, 익명화 방법은 특히 성별 편향을 줄이는 데 효과적이었습니다. 이 연구는 인사(HR) 애플리케이션을 넘어 LLM의 내재된 편향을 평가하는 새로운 방법론을 제안하며, 공정성과 포괄성을 증진하기 위한 최선의 관행을 강조합니다.



### SleepCoT: A Lightweight Personalized Sleep Health Model via Chain-of-Thought Distillation (https://arxiv.org/abs/2410.16924)
- **What's New**: 본 연구는 개인화된 수면 건강 관리를 위한 혁신적인 접근 방식을 제안합니다. 기존 대형 언어 모델(LLMs)의 성능을 소규모 언어 모델(>2B parameters)에서도 구현할 수 있도록 하는 few-shot Chain-of-Thought (CoT) 증류 방법을 사용합니다.

- **Technical Details**: 연구의 핵심은 GPT-4o를 사용한 데이터 합성, Qwen-max를 활용한 지침 세트 생성, Qwen2.5 1.5B를 통한 모델 증류입니다. 이를 통해 수면 건강에 대한 개인화된 추천, 사용자 특정 추적 질문 지원, 도메인 특화 지식 질문에 대한 응답을 제공합니다.

- **Performance Highlights**: 실험 결과, 100개의 시뮬레이션된 수면 보고서 및 1,000개의 도메인 특정 질문을 사용하여, SleepCoT 모델은 대형 모델과 유사한 성능을 유지하면서도 실제 환경에서의 효율성을 강조하였습니다.



### Large Language Model-based Augmentation for Imbalanced Node Classification on Text-Attributed Graphs (https://arxiv.org/abs/2410.16882)
Comments:
          11 pages, 4 figures

- **What's New**: 본 논문에서는 Text-Attributed Graphs (TAGs)에서 비대칭 클래스 문제를 해결하기 위한 새로운 접근법인 LA-TAG을 제시합니다. 이 방법은 대량의 언어 모델(LLM)을 활용하여 기존의 노드 텍스트에 기반하여 합성 텍스트를 생성하고, 이 텍스트를 통해 소수 클래스 노드를 보강합니다.

- **Technical Details**: LA-TAG는 LLM을 사용하여 그래프 내에서 기존 노드와 연결되도록 텍스트 기반의 링크 예측기를 도입합니다. 이를 통해 기존 노드와 합성된 텍스트 노드 간의 연관성을 형성하고, 다양한 데이터 세트와 평가 메트릭을 통해 성능을 입증합니다.

- **Performance Highlights**: LA-TAG은 기존의 비텍스트 기반 데이터 증강 전략과 특정 노드 불균형 솔루션에 비해 크게 뛰어난 성능을 나타내었습니다. 이는 소수 클래스와 다수 클래스 간의 성능 격차를 줄이는 데 기여하며, 데이터 불균형에 대한 강건성을 지속적으로 보여주었습니다.



### Can Large Language Models Act as Ensembler for Multi-GNNs? (https://arxiv.org/abs/2410.16822)
- **What's New**: 이번 연구에서는 Graph Neural Networks (GNNs)와 Large Language Models (LLMs)의 통합 접근 방식을 제안하여, 다양한 데이터 세트에서 GNN의 성능을 향상시키기 위한 새로운 모델 LensGNN을 소개합니다. 본 모델은 여러 GNN을 연결하여, 텍스트 속성의 의미 정보를 그래프 데이터에 효과적으로 주입하는 방법을 모색합니다.

- **Technical Details**: LensGNN 모델은 다수의 GNN 모델을 동일한 저차원 공간에 맞추어 정렬하고, LoRA (Low-Rank Adaptation) 기법을 사용하여 GNN에서 LLM으로의 공간 정렬을 수행합니다. 이 과정에서 그래프 토큰(graph tokens)과 텍스트 정보를 LLM에 주입하여, GNN의 다양한 임베딩을 최적화하고 종합적으로 성능을 향상시킵니다.

- **Performance Highlights**: 실험 결과, LensGNN은 기존의 17개 최첨단 모델에 비해 우수한 성능을 발휘하였으며, 다양한 데이터 세트에서 일관되게 더 나은 결과를 보여주었습니다. 이는 LensGNN이 GNN과 LLM의 장점을 통합하여 구현한 새로운 접근 방식을 통해 달성된 결과입니다.



### Context-aware Inductive Knowledge Graph Completion with Latent Type Constraints and Subgraph Reasoning (https://arxiv.org/abs/2410.16803)
- **What's New**: CATS(지식 그래프 완성)라는 새로운 접근 방식을 통해 보이지 않는 엔티티를 처리할 수 있는 최초의 LLM 기반 KGC 솔루션이 제안되었습니다.

- **Technical Details**: CATS는 두 개의 주요 모듈로 구성됩니다: Type-Aware Reasoning (TAR) 모듈은 후보 엔티티가 쿼리 관계에서 요구하는 암묵적인 엔티티 유형과 일치하는지 평가하고, Subgraph Reasoning (SR) 모듈은 적절한 추론 경로와 이웃 사실을 선택하여 이들의 상관관계를 평가합니다.

- **Performance Highlights**: CATS는 3개의 널리 사용되는 데이터셋(WN18RR, FB15k237, NELL-995)에서 18개의 실험 설정 중 16개에서 최고의 성과를 보여주었고, 평균 7.2%의 MRR 개선을 달성했습니다.



### Traj-Explainer: An Explainable and Robust Multi-modal Trajectory Prediction Approach (https://arxiv.org/abs/2410.16795)
- **What's New**: 본 논문에서는 Traj-Explainer라는 새로운 경로 예측 모델을 제안하여, 예측 결과의 설명 가능성을 높이고, 다양한 잠재적 미래 위치를 정확하게 예측할 수 있는 능력을 갖추었습니다. 기존의 모델들이 간과했던 시나리오 에이전트 간의 공동 추론을 고려하였으며, 예측의 내재적 메커니즘을 이해하는 데 도움을 줍니다.

- **Technical Details**: Traj-Explainer는 수정된 conditional diffusion(조건부 확산) 모델과 개선된 Shapley Value(샤플리 값) 모델을 통합하여, 시나리오의 멀티모달(다양한 형태) 경로 패턴을 캡처하고, 전역 및 시나리오 특성의 중요성을 합리적으로 학습합니다. 이 모델은 Waymo, NGSIM, HighD, MoCAD와 같은 여러 경로 예측 데이터셋을 통해 검증되었습니다.

- **Performance Highlights**: 모델의 성능은 실제의 복잡한 트래픽 환경을 모델링하고, 에이전트 간의 상호작용 정보를 통합하여 여러 차량의 잠재적 미래 위치를 정확히 예측할 수 있는 능력을 지니고 있습니다. 우리는 인간의 주행 경험과 일치하는 영향력 있는 특성 중요성을 식별하였으며, 이를 통해 제안된 모델이 예측 행동을 효과적으로 학습한다는 것을 입증하였습니다.



### Uncovering Key Trends in Industry 5.0 through Advanced AI Techniques (https://arxiv.org/abs/2410.16748)
- **What's New**: 이 논문은 약 200개의 온라인 기사를 분석하여 Industry 5.0의 트렌드를 인공지능 기술을 활용하여 식별하고자 합니다.

- **Technical Details**: LDA, BERTopic, LSA, K-means와 같은 다양한 알고리즘을 적용하여 문헌에 존재하는 중심 주제를 추출하고 비교합니다.

- **Performance Highlights**: 결과는 중심 주제를 둘러싼 수렴을 보여주며, Industry 5.0이 광범위한 주제를 포괄하고 있음을 강조합니다. 이 연구는 기존의 AI 기술이 효과적으로 트렌드를 식별하는 데 활용될 수 있음을 보여줍니다.



### 50 questions on Active Assisted Living technologies. Global edition (https://arxiv.org/abs/2410.16733)
- **What's New**: GoodBrother COST Action 프로그램의 일환으로, Active Assisted Living (AAL) 기술에 대한 새로운 리소스가 발표되었습니다. 이 자료는 2020-2024년간의 연구 결과를 바탕으로 하며, AAL 기술이 제공하는 다양한 기능과 사용자 권리를 보호하는 방법을 설명합니다.

- **Technical Details**: AAL(Active Assisted Living) 기술은 인지적 또는 신체적 도전에 직면한 개인을 지원하기 위한 도구를 제공하며, 이는 독립성을 향상시키고 일상적인 루틴을 돕고 안전한 생활 환경을 촉진합니다. 이러한 기술의 발전과 함께 개인정보 보호(data protection) 및 사용자 자율성(user autonomy)에 대한 중요한 질문들이 제기되고 있습니다.

- **Performance Highlights**: 이 리소스는 최종 사용자(end users), 간병인(caregivers), 의료 전문가(healthcare professionals), 정책 입안자(policy makers) 등 다양한 독자를 위한 통찰력 있는 정보를 제공합니다. AAL 기술을 케어 설정에 통합하는 방법에 대한 실용적인 가이드를 제공하며, 개인의 자율성을 존중하고 윤리적 사용을 보장하는 데 중점을 두고 있습니다.



### Resource-Efficient Sensor Fusion via System-Wide Dynamic Gated Neural Networks (https://arxiv.org/abs/2410.16723)
- **What's New**: 이번 논문은 동적 게이트가 있는 DNN 아키텍처를 기반으로 한 Quantile-constrained Inference (QIC) 알고리즘을 제안하며, 이는 다양한 데이터 소스를 활용한 AI 기반의 애플리케이션을 위한 효율적인 자원 할당을 최적화하는 데 초점을 맞추고 있습니다. 이는 인프라 수준의 의사결정을 처음으로 연결한 연구입니다.

- **Technical Details**: QIC는 센서 및 데이터 출처, DNN 아키텍처, 실행 노드, 자원에 대한 결정을 통합적으로 고려하여 인퍼런스 에너지 비용을 최소화하며, 인퍼런스 품질의 특정 분위수를 보장할 수 있는 신뢰성을 제공합니다. 이 모델은 이동 노드 및 에지 서버가 연결된 계층형 컴퓨팅 환경에서 최적의 센서 융합 및 인퍼런스를 위해 설계되었습니다.

- **Performance Highlights**: QIC는 RADIATE 데이터셋을 기반으로 훈련된 동적 게이트 DNN을 사용하여, 기존 접근 방식과 비교했을 때 인퍼런스 에너지 비용을 80% 절감하고 애플리케이션 실패율을 50% 감소시킴으로써 뛰어난 성능을 보였습니다.



### Influential Language Data Selection via Gradient Trajectory Pursu (https://arxiv.org/abs/2410.16710)
- **What's New**: 본 논문에서는 Gradient Trajectory Pursuit (GTP) 알고리즘을 제안합니다. GTP는 L0-norm regularized objective를 사용하여 데이터 샘플을 공동 선택함으로써 모델 성능을 극대화합니다. 이 방법은 기존의 개별 샘플 순위화를 따르지 않고, 샘플 중복을 자동으로 제거하며, 더 높은 효율성을 자랑합니다.

- **Technical Details**: GTP는 Gradient Trajectory를 추구하며, 샘플 데이터를 서브스페이스에서 매칭하여 선택합니다. 이 알고리즘은 압축 샘플링(Compressive Sampling) 과정과 분산 프레임워크(Distributed Framework)를 통해 계산 시간을 줄일 수 있습니다. 실험에서 본 알고리즘은 top-k 및 경쟁 알고리즘보다 일관되게 우수한 성능을 보여주었습니다.

- **Performance Highlights**: GTP 알고리즘은 전체 데이터셋의 0.5%만 선택하여 특정한 instruction tuning 작업에서 전체 성능을 달성했습니다. 또한, 이전의 orthogonal matching pursuit 기반 알고리즘보다 최대 1717배의 효율성을 보였습니다.



### Privacy-hardened and hallucination-resistant synthetic data generation with logic-solvers (https://arxiv.org/abs/2410.16705)
- **What's New**: Genomator는 논리 해결 접근 방식을 통해 개인 정보 보호와 현실성을 갖춘 데이터를 효율적으로 생성하는 방법을 제시합니다.

- **Technical Details**: Genomator는 Synthetic data(합성 데이터) 생성에 있어서 기존의 Markov generation, Restricted Boltzmann Machine, Generative Adversarial Network 및 Conditional Restricted Boltzmann Machines 방법론과 비교하여 84-93%의 정확도 향상과 95-98%의 높은 개인 정보 보호를 달성합니다. 이 접근법은 전체 게놈에서도 적용이 가능하며, 1000-1600배 더 효율적입니다.

- **Performance Highlights**: Genomator는 화합물 유전자 변이 프로파일의 구분 불가능한 데이터셋부터 민감한 집단의 입증 가능한 개인 데이터 표현까지, 다양한 응용 프로그램에 맞춰 조정할 수 있는 기능을 제공합니다.



### AskBeacon -- Performing genomic data exchange and analytics with natural languag (https://arxiv.org/abs/2410.16700)
- **What's New**: AskBeacon은 임상의 및 연구자들이 기술 장벽을 제거하고 글로벌 유전체 데이터 자원과 직접 상호작용할 수 있도록 지원하는 새로운 도구입니다.

- **Technical Details**: AskBeacon은 GA4GH Beacon 프로토콜을 통해 안전하게 공유된 코호트(cohorts)에서 Large Language Models를 적용할 수 있게 해줍니다. 사용자는 단순히 Beacon에 질문을 던짐으로써 데이터를 탐색하고 통찰(insights)을 얻을 수 있습니다.

- **Performance Highlights**: AskBeacon을 사용하면 유용한 통찰을 분석하고 출판 준비가 완료된 결과를 도출할 수 있습니다.



### Improving Causal Reasoning in Large Language Models: A Survey (https://arxiv.org/abs/2410.16676)
- **What's New**: 이 논문은 대규모 언어 모델(LLMs)의 인과 추론(causal reasoning) 능력을 향상시키기 위한 다양한 연구를 포괄적으로 검토합니다. 연구는 LLM의 역할에 따라 두 가지 카테고리로 나누어집니다: 인과 추론 엔진으로서의 LLM 또는 전통적 인과 추론 방법에 지식을 제공하는 도우미로서의 역할입니다.

- **Technical Details**: LLMs가 인과적 관계를 이해하는 데 어려움이 있으며, 이를 해결하기 위해 다양한 방법론이 소개됩니다. 본 논문은 LLM을 인과 추론 엔진으로 활용하기 위한 파인튜닝(fine-tuning), 프롬프트 엔지니어링(prompt engineering), 도구 통합(tools integration), 대체 접근 방식(alternative approaches)을 다룹니다. 또한 전통적 방법의 도우미로서 정보 추출(information extraction) 및 데이터 생성(data generation) 기술도 다룹니다.

- **Performance Highlights**: 대규모 언어 모델(LLMs)은 기존의 인과 추론 작업에 대한 평가를 통해 여러 과제를 수행하는 데 있어 중요한 성능 개선 기회를 보였습니다. 그러나 인과적 추론의 깊이가 부족하고 고품질 데이터의 부족이 여전히 도전 과제로 남아 있습니다. 따라서 LLM의 인과 추론 구조와 사전 학습(pre-training) 과정에서 인과 추론을 통합하는 것이 향후 연구의 유망한 방향으로 제시됩니다.



### DEAN: Deactivating the Coupled Neurons to Mitigate Fairness-Privacy Conflicts in Large Language Models (https://arxiv.org/abs/2410.16672)
- **What's New**: 이 연구에서는 대형 언어 모델(LLM)에서 공정성(fairness) 및 프라이버시(privacy) 인식 간의 의외의 트레이드오프(trade-off) 현상을 발견했습니다. 공정성과 프라이버시가 상충할 수 있다는 점을 해결하기 위해, 우리는 DEAN(Decoupling Fairness and Privacy through Neurons)이라는 새로운 방법을 제안하였습니다.

- **Technical Details**: DEAN은 두 변수 간의 상호 정보를 줄어들게 하여 공정성과 프라이버시 인식의 관계를 분리할 수 있도록 설계되었습니다. 이 방법은 훈련이 필요 없으며, 공정성과 프라이버시와 밀접하게 관련된 뉴런을 식별하고 비활성화하여 이를 달성합니다. 실험 결과, DEAN은 LLM의 공정성이 12.2%, 프라이버시가 14.0% 향상되었습니다. DEAN은 제한된 주석 데이터에서도 효과적으로 작동하며, 악의적인 데이터에서도 강력한 성능을 보입니다.

- **Performance Highlights**: DEAN은 LLM의 일반적인 성능에 손상을 주지 않으면서 공정성과 프라이버시 인식을 개선할 수 있음을 입증하였습니다. 또한, 몇 백 개의 데이터 샘플과 같은 제한된 조건에서도 우수한 성능을 발휘하며, 오히려 기존의 SFT 방법보다 더 나은 해석 가능성을 제공합니다.



### CKSP: Cross-species Knowledge Sharing and Preserving for Universal Animal Activity Recognition (https://arxiv.org/abs/2410.16644)
- **What's New**: 이번 연구에서는 다양한 동물 종의 센서 데이터를 활용하여 Cross-species Knowledge Sharing and Preserving (CKSP)라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 여러 종의 공통 행동 패턴 및 종별 행동 패턴을 고려하여 동물 활동 인식의 성능을 향상시키는 것을 목표로 합니다.

- **Technical Details**: CKSP 프레임워크는 Shared-Preserved Convolution (SPConv) 모듈을 사용하여 각 종별 저랭크 (low-rank) 합성곱 계층을 배정하고 공통의 전랭크 (full-rank) 합성곱 계층을 통해 공통의 특징을 학습합니다. 또한, Species-specific Batch Normalization (SBN) 모듈을 통해 서로 다른 종 간의 데이터 분포 차이를 해결하여 훈련의 충돌을 완화합니다.

- **Performance Highlights**: 실험 결과, CKSP 프레임워크는 말, 양, 소 데이터셋에서 각각 6.04%, 2.06%, 3.66%의 정확도 개선과 10.33%, 3.67%, 7.90%의 F1-score 개선을 보여주었습니다. 이는 다종 데이터 활용을 통한 분류 성능 향상의 가능성을 입증합니다.



### LLMScan: Causal Scan for LLM Misbehavior Detection (https://arxiv.org/abs/2410.16638)
- **What's New**: 본 연구에서는 LLMScan이라는 혁신적인 LLM 모니터링 기법을 소개합니다. 이 방법은 인과성 분석(causality analysis)에 기반하여 LLM의 내부 작동을 체계적으로 모니터링하며, 다양한 형태의 잘못된 행동(misbehavior)을 감지할 수 있는 포괄적인 솔루션을 제공합니다.

- **Technical Details**: LLMScan은 두 가지 주요 구성 요소인 스캐너(scanner)와 탐지기(detector)로 구성됩니다. 스캐너는 프롬프트 토큰(prompt tokens)과 신경 층(neural layers)에 대한 인과성 분석(causality analysis)을 수행하여 인과성 분포도(causal distribution map)를 생성합니다. 탐지기는 이러한 인과성 분포도를 기반으로 학습된 분류기(classifier)로, LLM이 잘못된 행동을 하고 있는지를 런타임(runtime) 중에 평가합니다.

- **Performance Highlights**: LLMScan은 13개 다양한 데이터셋을 사용하여 4가지 유형의 잘못된 행동을 정확하게 식별할 수 있으며, 특히 불신 검출(untruthful responses) 및 유해한 출력(harmful outputs) 탐지에서 평균 AUC가 0.95를 초과하는 뛰어난 성능을 보입니다.



### Optimizing LLMs with Direct Preferences: A Data Efficiency Perspectiv (https://arxiv.org/abs/2410.16586)
- **What's New**: 대형 언어 모델(LLMs)의 출력과 인간의 선호도(예: 인간 피드백을 통한 강화 학습, RLHF)와의 정렬이 실제 시나리오에서의 효과성을 보장하기 위해 필수적임을 강조합니다. 이 연구는 여러 유형의 선호 데이터가 모델 성능에 미치는 영향을 체계적으로 탐구합니다.

- **Technical Details**: 본 연구에서는 Direct Preference Optimization (DPO)을 사용하여 미리 훈련된 LLM의 미세 조정(정교화)을 관찰하며, 선호 데이터의 방대한 양에 대한 의존도를 줄이고자 했습니다. 이를 위해 세 가지 공개 데이터셋(OpenOrca, UltraFeedback, Capybara)에 대해 DPO 방법의 효과성을 평가했습니다.

- **Performance Highlights**: 데이터의 양이 증가함에 따라 모델 성능이 일반적으로 향상되고 안정화되었습니다. 다양한 데이터셋을 조합해 사용할 경우 모델 효과성이 크게 개선되었습니다. 특히, 대화형 프롬프트로 훈련된 모델이 질문 응답 프롬프트로 훈련된 모델에 비해 더 나은 성과를 보였습니다.



### How Can We Diagnose and Treat Bias in Large Language Models for Clinical Decision-Making? (https://arxiv.org/abs/2410.16574)
- **What's New**: 본 연구는 Large Language Models(LLMs)가 진료 의사결정에서 나타나는 성별 및 인종 편향을 평가하고 완화하기 위한 새로운 Counterfactual Patient Variations(CPV) 데이터셋을 소개함으로써 LLM의 임상 활용에 대한 심층적인 이해를 제공합니다.

- **Technical Details**: 본 연구는 LLM의 편향 평가를 위한 프레임워크를 구축하였으며, Multiple Choice Questions(MCQs)와 설명을 통합하여 분석합니다. 또한, 다양한 LLM 모델을 사용하여 프롬프트 엔지니어링과 파인튜닝 방법을 평가하여 편향 완화 전략을 제시합니다.

- **Performance Highlights**: 연구 결과, LLM의 성별 및 인종 편향은 복잡한 임상 시나리오에서 광범위하게 나타났으며, MCQ 응답의 성과와 설명 과정 사이의 불일치가 지속적인 편향을 드러냈습니다. 성별 편향은 의학 전문 분야마다 상이하며, 파인튜닝은 일부 편향을 완화할 수 있지만 새로운 편향을 초래할 수 있는 경향이 있습니다.



### Large language models enabled multiagent ensemble method for efficient EHR data labeling (https://arxiv.org/abs/2410.16543)
Comments:
          27 pages, 13 figures. Under journal review

- **What's New**: 본 연구는 대규모 EHR 데이터셋에서 데이터 레이블링의 주요 문제를 해결하기 위해 LLM(대형 언어 모델)에 기반한 새로운 multiagent ensemble 방법을 도입했습니다.

- **Technical Details**: 우리는 다양한 오픈 소스 LLM을 사용하여 각 LLM의 예측을 투표로 간주하고 최소 승리 기준의 다수결 메커니즘을 적용하여 앙상블을 구성했습니다. 이 방법을 통해 623,566개의 ECG 보고서가 포함된 MIMIC-IV ECG 데이터셋을 레이블링하였으며, 예상 정확도는 98.2%에 달합니다.

- **Performance Highlights**: 실험 결과 앙상블 LLM이 개별 LLM, 특히 상업적으로 가장 우수한 모델보다 더 나은 성능을 보였으며, 허위 정보 오류(hallucination errors)를 감소시켰습니다. 이 방식은 대규모 EHR 데이터 레이블링과 같은 데이터 라벨링 작업의 시간과 노력을 크게 줄여주며, 다른 텍스트 데이터 라벨링 작업에도 잘 일반화될 수 있음을 보여주었습니다.



### QIXAI: A Quantum-Inspired Framework for Enhancing Classical and Quantum Model Transparency and Understanding (https://arxiv.org/abs/2410.16537)
Comments:
          18 pages, 3 figures

- **What's New**: 이 논문은 QIXAI Framework(Quantum-Inspired Explainable AI)를 소개하여, quantum-inspired 기법을 통해 신경망의 해석 가능성을 높이는 새로운 접근 방식을 제안합니다. 이 프레임워크는 quantum mechanics의 원칙을 활용하여 CNN이 어떻게 결정을 내리는지 설명합니다.

- **Technical Details**: QIXAI 프레임워크는 Hilbert 공간, superposition, entanglement 및 eigenvalue decomposition과 같은 기법을 사용하여 신경망의 다양한 레이어가 특징을 처리하고 결합하는 방식을 보여줍니다. 이 방법은 SHAP, LIME 및 Layer-wise Relevance Propagation(LRP)와 같은 기존의 해석 기술들의 한계를 극복합니다.

- **Performance Highlights**: 논문에서는 말라리아 기생충 탐지를 위한 CNN을 사례 연구로 사용하여 SVD(Singular Value Decomposition), PCA(Principal Component Analysis), MI(Mutual Information)와 같은 quantum-inspired 기법들이 모델의 행동을 해석하는 데 어떻게 유용한지를 보여줍니다. QIXAI는 다양한 모델에 적용 가능하며, 신뢰할 수 있는 AI 시스템 개발의 목표를 지향하고 있습니다.



### Large Body Language Models (https://arxiv.org/abs/2410.16533)
- **What's New**: 본 논문에서는 실시간으로 사람처럼 제스처를 생성하는 데 최적화된 새로운 구조인 Large Body Language Models (LBLMs)를 도입합니다. 특히, LBLM-AVA라는 아키텍처는 Transformer-XL과 병렬화된 diffusion 모델을 결합하여 텍스트, 오디오 및 비디오를 포함하는 다중 모달 입력으로부터 인간처럼 보이는 제스처를 생성합니다.

- **Technical Details**: LBLM-AVA는 제스처 생성 기능을 향상시키기 위한 여러 핵심 구성 요소를 통합합니다: 다중 모달-포즈 임베딩(multimodal-to-pose embeddings), 재정의된 주의 메커니즘(attention mechanisms)을 통해 개선된 시퀀스-투-시퀀스 매핑(sequence-to-sequence mapping), 제스처 시퀀스 일관성을 위한 시간적 스무딩 모듈(temporal smoothing module), 그리고 현실감을 높이기 위한 주의 기반 정제 모듈(attention-based refinement module)입니다.

- **Performance Highlights**: LBLM-AVA는 Fréchet Gesture Distance (FGD)를 30% 줄여주며, Fréchet Inception Distance에서 25% 향상을 이뤄내어 기존 접근 방식에 비해 현실적이고 상황에 적합한 제스처 생성에서 최첨단 성능을 달성합니다.



### Distributed Online Life-Long Learning (DOL3) for Multi-agent Trust and Reputation Assessment in E-commerc (https://arxiv.org/abs/2410.16529)
- **What's New**: 본 논문은 비정상 환경에서 서비스 제공자의 신뢰(Trust) 및 평판(Reputation) 평가를 위한 분산형 온라인 평생학습(Distributed Online Life-Long Learning, DOL3) 알고리즘을 제안합니다.

- **Technical Details**: DOL3 알고리즘은 관찰자가 실시간으로 서비스 제공자의 신뢰 및 평판 점수를 학습하고 이를 이웃 관찰자들과 통신하여 평가하는 구조로 되어 있습니다. 알고리즘은 적응형 온라인 학습 프레임워크와 신뢰 융합(trust fusion) 과정을 결합하여 이웃 관찰자의 평가와 자신의 평가를 통합합니다.

- **Performance Highlights**: 시뮬레이션 연구 결과, DOL3 알고리즘은 기존의 선진 기계 학습 방법들보다 90%의 경우에서 우수한 성능을 보이며, 신뢰 및 평판 평가에서 환경의 변동성을 효과적으로 처리하는 것으로 나타났습니다.



### Allo-AVA: A Large-Scale Multimodal Conversational AI Dataset for Allocentric Avatar Gesture Animation (https://arxiv.org/abs/2410.16503)
- **What's New**: Allo-AVA 데이터 세트는 대화형 AI의 생생한 아바타 애니메이션 생성을 위한 고품질 다중 모드 훈련 데이터를 제공합니다. 이 데이터 세트는 텍스트와 오디오 기반 아바타 제스처 애니메이션을 위해 특별히 설계되었습니다.

- **Technical Details**: Allo-AVA 데이터 세트는 약 1250시간의 다양한 비디오 콘텐츠를 포함하며, 오디오, 전사, 추출된 키포인트와 함께 제공됩니다. 이 데이터는 발화 내용, 음향 특성, 시각적 신호 및 대화 맥락 간의 관계를 포착하는 데 유용합니다.

- **Performance Highlights**: 키포인트 수가 1350억 개 이상이며, 평균 1분당 112,500개의 키포인트가 추출됩니다. 이 데이터 세트의 다양성은 다양한 연령, 성별 및 민족적 배경을 포함하여 아바타 애니메이션의 자연성을 높이는 데 기여합니다.



### Conjuring Semantic Similarity (https://arxiv.org/abs/2410.16431)
- **What's New**: 이번 논문은 텍스트 표현 간의 시멘틱 유사성(semantic similarity)을 비교하는 새로운 접근 방식을 제안합니다. 이 방법은 텍스트의 내용이 아닌, 생성된 이미지를 통해 시멘틱 유사성을 측정합니다. 기존의 언어 기반 방법과는 다르게, 이미지 분포(image distributions)를 활용하여 이 거리를 측정함으로써 직관적이고 해석 가능한 결과를 제공합니다.

- **Technical Details**: 제안하는 방법에서는 Jensen-Shannon divergence를 사용하여 텍스트 표현에 의해 유도된 역시간 확산 확률 미분 방정수(reverse-time diffusion stochastic differential equations, SDEs)의 거리를 계산합니다. 이를 Monte-Carlo 샘플링을 통해 직접 계산할 수 있습니다. 또한, 현대의 이미지 생성 모델, 특히 텍스트 조건화(diffusion models)를 활용하여 두 텍스트 문구 간의 시멘틱 유사성을 이미지 분포의 유사성으로 측정합니다.

- **Performance Highlights**: 제안된 방법은 기존의 큰 언어 모델을 기반으로 한 제로샷(zero-shot) 접근 방식과 비교할 수 있는 결과를 보여주었습니다. 단순한 거리 측정 방식이지만, 다양한 확산 모델(diffusion models)과 추론 알고리즘(inference algorithms)에 대한 강건성을 입증하는 애블레이션 연구(ablation studies)를 실시하여, 보다 직관적인 이미지 분포를 통해 텍스트 표현의 시멘틱 관계를 시각화할 수 있음을 Demonstrate 하였습니다.



### Subword Embedding from Bytes Gains Privacy without Sacrificing Accuracy and Complexity (https://arxiv.org/abs/2410.16410)
- **What's New**: 이번 연구에서는 개인 정보 보호를 위한 새로운 방법인 Subword Embedding from Bytes (SEB)를 제안합니다. SEB는 서브워드를 바이트 시퀀스로 인코딩하여 개인 정보 회복을 더 어렵게 만듭니다. 또한, SEB는 256 바이트의 작은 어휘를 요구하며 동일한 입력 길이를 유지하면서도 효율성을 갖추고 있습니다.

- **Technical Details**: SEB의 접근 방식은 세 가지 단계로 구성됩니다: (1) 서브워드와 바이트 간의 매핑 구축, (2) 입력 텍스트를 바이트 시퀀스로 변환, (3) 해당 바이트 임베딩을 검색하고 서브워드 경계를 유지하며 단일 서브워드 임베딩으로 집계하는 작업입니다. 이 방법은 작은 어휘 크기와 같은 입력 시퀀스 길이를 유지하면서 개인 정보 보호를 강화합니다.

- **Performance Highlights**: 실험 결과 SEB는 전통적인 임베딩 기반 공격으로부터 원본 문장 복원을 효과적으로 방어할 수 있으며, 기계 번역, 감정 분석, 언어 모델링과 같은 여러 분야에서 기존의 서브워드 임베딩 방법보다 더 나은 결과를 보여줍니다. 이는 SEB가 시간과 공간의 복잡성을 줄이는 동시에 개인 정보 보호와 더불어 비슷하거나 더 나은 정확성을 달성함을 의미합니다.



### Towards a Reliable Offline Personal AI Assistant for Long Duration Spacefligh (https://arxiv.org/abs/2410.16397)
Comments:
          75th International Astronautical Congress (IAC), Milan, Italy, 14-18 October 2024

- **What's New**: 인류가 달과 화성으로의 새로운 임무를 준비하면서, 우주 비행사들은 지구와의 통신 지연으로 인해 더 큰 자율성으로 운영해야 할 필요성이 증가하고 있습니다. 기존의 Generative Pretrained Transformer (GPT) 모델들은 안전-critical 환경에서 한계가 있습니다.

- **Technical Details**: 이 논문에서는 Mars Exploration Telemetry-Driven Information System (METIS)와 같은 시스템을 소개하고, 이를 개선하기 위해 GPT, Retrieval-Augmented Generation (RAG), Knowledge Graphs (KGs), Augmented Reality (AR) 기술을 통합하는 방안을 제안합니다. KGs는 실시간 텔레메트리와 멀티모달 데이터에 쉽게 접근할 수 있도록 사전 정보를 제공합니다.

- **Performance Highlights**: 이 시스템은 우주 비행사들이 자연어 쿼리와 AR을 통해 데이터를 직관적으로 상호작용할 수 있게 해주며, 더 안전하고 효율적으로 임무를 수행할 수 있도록 돕습니다.



### Designing Robust Cyber-Defense Agents with Evolving Behavior Trees (https://arxiv.org/abs/2410.16383)
Comments:
          10 pages, 8 figures

- **What's New**: 본 논문에서는 이진 행동 트리(Evolving Behavior Trees, EBTs)를 사용하여 자율 사이버 방어 에이전트를 설계하고 최적화하는 접근 방식을 제시합니다. 이는 다양한 사이버 공격에 적응할 수 있도록 설계된 학습 가능 구성 요소(learning-enabled components)와 연계되어 있습니다.

- **Technical Details**: 자율 사이버 방어 에이전트는 EBT 구조를 활용하여 복잡한 장기 방어 작업을 효과적으로 수행합니다. 이 에이전트는 고수준 제어 구조를 학습하고, LECs를 최적화하며, 현실적인 사이버 환경으로 통합하는 세 가지 단계로 개발됩니다. 이 과정은 유전 프로그래밍(genetic programming, GP)을 통해 수행됩니다.

- **Performance Highlights**: EBT 기반 에이전트는 동적 사이버 공격에 대한 방어 성능이 크게 향상되어 안전성과 설명 가능성을 강조합니다. 특히, 평균 보상에서 39% 향상된 결과를 보여주었습니다. EBT는 주요 이벤트를 모니터링하고 고수준의 하위 작업 사이의 전환을 모델링할 수 있어, 실제 사이버 환경에 적용 가능합니다.



### JMMMU: A Japanese Massive Multi-discipline Multimodal Understanding Benchmark for Culture-aware Evaluation (https://arxiv.org/abs/2410.17250)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 일본 문화 맥락에 기반한 전문 수준의 작업을 평가하기 위해 설계된 첫 번째 대규모 일본어 벤치마크인 JMMMU(Japanese MMMU)를 소개합니다. 이 벤치마크는 두 가지 상호 보완적인 하위 집합으로 구성되어 있습니다: 문화 비독립적(Culture-Agnostic, CA) 하위 집합과 문화 특정(Culture-Specific, CS) 하위 집합.

- **Technical Details**: JMMMU는 1,320개의 질문과 1,118개의 이미지를 포함하고 있으며, 28개의 다양한 과목을 다룹니다. CA 하위 집합에서는 영어로 동일한 내용의 질문과의 직접 비교를 가능하게 하고, CS 하위 집합에서는 일본 문화에 적합한 새로운 질문을 포함합니다. 이 연구는 15개의 오픈 소스 LMM과 3개의 고급 독점 LMM을 평가하여 발견된 주요 성과를 분석합니다.

- **Performance Highlights**: JMMMU에서 평가된 모델의 전반적인 성능은 최대 58.6%로, 일본 문화에 대한 이해가 부족함을 보여줍니다. CA 하위 집합에서는 대부분의 모델이 일본어로 질문했을 때 영어보다 성능이 떨어졌고(CA 하위 집합에서 최대 8.6% 하락), CS 하위 집합에서 일본 데이터셋으로 훈련된 모델이 가장 높은 성능을 보였습니다. 이 연구는 다문화 성격을 고려한 다양한 고표준 벤치마크 개발의 필요성을 강조하고 있습니다.



### Learning Precise, Contact-Rich Manipulation through Uncalibrated Tactile Skins (https://arxiv.org/abs/2410.17246)
- **What's New**: 이번 연구는 로봇 조작에서 정밀한 접촉 작업을 수행하기 위해 마그네틱 스킨 센서를 사용한 정책 학습을 소개합니다. 기존의 광학 촉각 센서에 의존하지 않고, 낮은 차원의 민감한 감지 능력을 가진 저렴한 센서를 활용하여 Visuo-Skin (ViSk) 프레임워크를 제안합니다.

- **Technical Details**: ViSk 프레임워크는 햅틱 센서 데이터를 시각적 정보와 함께 추가 토큰으로 처리하는 변환기 기반 정책을 사용합니다. AnySkin 센서를 통해 15차원 저차원 촉각 데이터를 수집하며, 이를 통해 복잡한 조작 작업에서 효과적인 정책 학습이 가능합니다.

- **Performance Highlights**: ViSk를 사용하여 훈련된 정책은 4가지 정밀 조작 작업에서 비전 전용 모델 대비 평균 27.5%의 성능 향상을 보여주었고, 광학 촉각 센서를 사용하는 정책보다 최소 43% 더 높은 성능을 기록하였습니다.



### Large Language Models Empowered Personalized Web Agents (https://arxiv.org/abs/2410.17236)
Comments:
          The code and data are available on the project website this https URL

- **What's New**: LLM 기반 개인화 웹 에이전트의 중요성을 강조하며, 개인화된 데이터를 통합하여 사용자의 지침을 더 잘 이해하고 맞춤형 행동을 실행하는 방법을 제안합니다.

- **Technical Details**: 개인화 웹 에이전트를 위한 새로운 벤치마크인 PersonalWAB을 구축하여, 사용자 지침, 개인화된 사용자 데이터, 웹 기능 및 세 가지 개인화된 웹 작업에 대한 평가 패러다임을 포함합니다. 또한, PUMA(Personalized User Memory-enhanced Alignment) 프레임워크를 통해 LLM을 개인화된 웹 에이전트 작업에 맞추도록 조정합니다.

- **Performance Highlights**: PUMA는 PersonalWAB에서 기존의 웹 에이전트 성능을 초월하여 개인화된 사용자 지침 및 선호도와 더 잘 정렬되어 보다 지능적이고 맞춤화된 웹 서비스를 제공할 수 있음을 입증합니다.



### Neuroevolution Neural Architecture Search for Evolving RNNs in Stock Return Prediction and Portfolio Trading (https://arxiv.org/abs/2410.17212)
- **What's New**: 이 논문에서는 주식 수익률 예측을 위한 새로운 접근법으로 Evolutionary eXploration of Augmenting Memory Models (EXAMM) 알고리즘을 제안합니다. 이 알고리즘은 주식별로 독립적으로 진화한 순환 신경망 (RNN)을 활용하여 포트폴리오 거래 결정을 내릴 수 있게 합니다.

- **Technical Details**: EXAMM 알고리즘은 최소한의 초기 네트워크에서 시작하여 점진적으로 더 큰 RNN을 진화시킵니다. 다양한 형태의 변이 (mutation)와 교차 (crossover) 연산을 통해 개체군이 진화하며, 각 섬(island)에서 독립적으로 유전자 다양성을 유지합니다. 11가지 변이와 2가지 교차 방법을 사용하여 새로운 개체군(genome)을 생성하며, 부모로부터 가중치를 상속받아 학습 시간을 크게 단축할 수 있습니다.

- **Performance Highlights**: 2022년 (약세장)과 2023년 (강세장) 동안, 이 진화한 RNN과 간단한 일일 롱숏 전략을 활용했을 때 데이터 존(Dow-Jones Index)과 S&P 500 Index보다 더 높은 수익을 기록했습니다.



### Exploring Possibilities of AI-Powered Legal Assistance in Bangladesh through Large Language Modeling (https://arxiv.org/abs/2410.17210)
Comments:
          In Review

- **What's New**: 방글라데시의 법률 시스템을 지원하기 위해 개발된 전문화된 대규모 언어 모델 (LLM)인 GPT2-UKIL-EN에 대한 연구 결과를 발표합니다.

- **Technical Details**: UKIL-DB-EN 데이터셋을 구축하여 방글라데시 법률 문서로부터 정보를 수집하고, GPT-2 모델을 이 데이터셋에 대해 미세 조정하여 방글라데시의 법률 지원에 적합한 LLM을 개발하였습니다.

- **Performance Highlights**: 모델은 전문가 의견을 포함한 사례 연구를 통해 엄격하게 평가되었으며, 법률 문제 해결을 위한 잠재력을 보여주었습니다. 그러나 모델의 정확성, 신뢰성 및 안전성을 향상시키기 위한 추가 개선이 필요합니다.



### VoiceBench: Benchmarking LLM-Based Voice Assistants (https://arxiv.org/abs/2410.17196)
Comments:
          Work in progress. Data is available at this https URL

- **What's New**: VoiceBench라는 새로운 벤치마크가 도입되어 LLM 기반 음성 비서의 다면적 평가를 가능하게 합니다.

- **Technical Details**: VoiceBench는 일반 지식, 지시 수행 능력 및 안전성을 평가하기 위해 실제 및 합성된 음성 지시를 포함하며, 다양한 발표 스타일, 환경 조건 및 내용 변화를 포함하는 테스트 케이스를 설계했습니다.

- **Performance Highlights**: 현재 LLM 기반 음성 비서 모델의 한계가 드러났으며, 전통적 ASR 시스템과 LLM의 조합모델 사이의 성능 차이를 강조했습니다.



### Emphasizing Discriminative Features for Dataset Distillation in Complex Scenarios (https://arxiv.org/abs/2410.17193)
Comments:
          24 pages, 13 figures

- **What's New**: 본 논문에서는 EDF(Emphasize Discriminative Features)를 제안하여, Grad-CAM 활성화 맵을 사용하여 합성 이미지의 핵심 식별 영역을 강조합니다. 간단한 데이터셋에서 성능은 뛰어나지만 복잡한 시나리오에서는 성능 저하 문제가 있음을 여실히 보여줍니다.

- **Technical Details**: EDF는 합성 이미지 distillation 과정에서 식별 특징을 강조하는 방법으로, Grad-CAM 활성화 맵을 통해 높은 활성도 영역의 업데이트를 강화합니다. 저손실(supervision signals)이 있는 경우는 제외하고, 높은 손실을 가진 신호만 활성화하여 효과적인 학습을 유도합니다. 이러한 방식으로 복잡한 시나리오에서 DD(데이터셋 증류)의 성능을 높입니다.

- **Performance Highlights**: 복잡한 시나리오에서 SOTA(state-of-the-art) 성능을 지속적으로 달성하며, ImageNet-1K의 여러 부분집합에서 손실 없는 성능을 기록했습니다. 이는 EDF가 데이터셋 증류 방법론에서 유의미한 발전을 이루어냈음을 보여줍니다.



### DyPNIPP: Predicting Environment Dynamics for RL-based Robust Informative Path Planning (https://arxiv.org/abs/2410.17186)
Comments:
          8 pages, 4 figures, submitted to IEEE RA-L

- **What's New**: 이 논문에서는 다양한 동적 환경에서 효과적으로 작동하는 강력한 RL 기반의 정보 경로 계획 프레임워크인 DyPNIPP를 제안합니다.

- **Technical Details**: DyPNIPP는 도메인 무작위화(domain randomization) 기법을 적용하여 다양한 환경에서 에이전트를 훈련시키고, 환경 동적을 캡처하여 에이전트의 행동을 조정하는 동적 예측 모델을 도입합니다. 이는 환경의 변화에 대응하는 경로 계획을 가능하게 하여 정보 경로 계획(IPR)의 성능을 향상시킵니다.

- **Performance Highlights**: 실험 결과, DyPNIPP는 기존의 RL 기반 IPP 알고리즘에 비해 강인성이 크게 향상되었으며, 다양한 환경 조건에서도 우수한 성능을 보였습니다.



### KANICE: Kolmogorov-Arnold Networks with Interactive Convolutional Elements (https://arxiv.org/abs/2410.17172)
- **What's New**: KANICE는 Convolutional Neural Networks (CNNs)와 Kolmogorov-Arnold Network (KAN)의 원리를 결합한 새로운 신경망 아키텍처입니다. 특히 Interactive Convolutional Blocks (ICBs)와 KAN 선형 레이어를 통합하여 복잡하고 비선형적인 데이터 관계를 포착하며, 동적이며 맥락에 의존적인 특징 추출을 가능하게 합니다.

- **Technical Details**: KANICE는 KAN의 일반 근사화 기능과 ICB의 적응형 특징 학습을 활용하여 CNN 프레임워크 내에서 작동합니다. KANICE는 MNIST, Fashion-MNIST, EMNIST, SVHN 데이터셋에서 실험을 수행하였으며, 기존 CNN 및 하이브리드 구조와 비교하여 일관되게 우수한 성능을 보였습니다. 또한, KANICE-mini라는 컴팩트 변형을 통해 성능을 유지하면서도 적은 파라미터 수로 효율성을 높였습니다.

- **Performance Highlights**: KANICE는 MNIST 데이터셋에서 99.35%의 정확도를 기록하였고, SVHN 데이터셋에서는 90.05%에 도달했습니다. KANICE-mini는 2,337,828개의 파라미터로 SVHN에서 90.00%의 정확도를 달성했습니다. KANICE 아키텍처는 전반적으로 이미지 분류 작업에서 성능과 계산 효율성을 균형 있게 유지하는 가능성을 보여줍니다.



### Layered LA-MAPF: a decomposition of large agent MAPF instance to accelerate solving without compromising solvability (https://arxiv.org/abs/2410.17160)
- **What's New**: 본 논문에서는 Multi-Agent Path Finding (MAPF) 문제를 다루며, '대형 에이전트(large agents)'를 고려한 새로운 접근 방식을 제안합니다. 특히 괴리감과 복잡성을 줄이기 위해 MAPF 인스턴스를 클러스터(cluster)와 레벨(level)로 분해하는 'Layered LA-MAPF' 방법을 소개합니다.

- **Technical Details**: Layered LA-MAPF 방법은 기하학적 형태의 대형 에이전트를 포함하는 MAPF 인스턴스를 클러스터로 분해하고, 이 클러스터를 다시 레벨로 나눕니다. 이 방식은 LA-MAPF 문제의 시간 복잡성을 줄이는 것을 목표로 하며, 독립적으로 해결되는 작은 하위 문제(subproblems)로 나눔으로써 해결합니다. 실험 결과에 따르면, 이 방식은 평균 40초에서 20초로 시간 비용을 절반으로 줄이고 성공률은 0.27에서 0.80으로 세 배 증가시킵니다.

- **Performance Highlights**: 본 연구의 실험 결과, LA-MAPF 방법의 성능이 크게 향상되는 것을 확인하였습니다. 다양한 맵에서 에이전트 수가 증가함에 따라 LA-MAPF 방법의 시간 비용을 절감하고 성공률을 확인할 수 있었습니다. LA-MAPF 메서드는 60초 이내에 솔루션을 찾는 데 있어 시간 비용을 절반으로 줄이고 성공률을 세 배 증가시켰습니다.



### Can General-Purpose Large Language Models Generalize to English-Thai Machine Translation ? (https://arxiv.org/abs/2410.17145)
Comments:
          Accepted in GenBench EMNLP 2024

- **What's New**: 이번 연구는 일반적인 대형 언어 모델(LLMs)의 저자원 환경에서의 성능 한계를 조사합니다. 특히, 영어-태국어 기계 번역 및 코드 스위칭 데이터셋에서 다양한 LLM과 전문 번역 모델의 성능을 비교합니다.

- **Technical Details**: 연구에서는 Llama-3(8B) 모델과 NLLB-600M, NLLB-3.3B 모델을 사용하여 4비트 양자화 환경에서의 번역 성능을 실험하였으며, BLEU, METEOR, CER 같은 표준 MT 메트릭을 활용했습니다. LLM의 성능 저하를 분석하기 위해 GPT4-o를 사용하여 번역의 실패 모드를 평가했습니다.

- **Performance Highlights**: 연구 결과에 따르면, NLLB-3.3B와 NLLB-600M 모델이 Llama-3 8B보다 여러 메트릭에서 일관적으로 더 좋은 성능을 보였고, 특히 CS 데이터셋에서 NLLB 모델이 더욱 높은 성적을 기록했습니다. 이로써 리소스가 제한된 환경에서 전문 번역 모델의 중요성이 강조되었습니다.



### Towards Automated Penetration Testing: Introducing LLM Benchmark, Analysis, and Improvements (https://arxiv.org/abs/2410.17141)
Comments:
          Main Paper 1-9 pages, Supplementary Materials: 10-17, 13 figures

- **What's New**: 본 논문은 LLM을 기반으로 한 자동 침투 테스트를 위한 새로운 공개 벤치마크를 소개하며, 사이버 보안 분야에서의 발전을 촉진하고 평가하는 데 기여합니다.

- **Technical Details**: 이 연구에서는 PentestGPT 도구를 사용하여 GPT-4o와 Llama 3.1-405B의 성능을 평가하였고, LLM의 펜테스팅(penetration testing)에 대한 전반적인 한계를 조사합니다. 다양한 침투 테스트 이론을 기반으로, enumeration, exploitation 및 privilege escalation 등 여러 측면에서 LLM이 직면한 문제점을 조명합니다.

- **Performance Highlights**: Llama 3.1이 GPT-4o에 비해 약간의 우위를 보였으나, 두 모델 모두 완전 자동화된 전반적인 침투 테스트를 수행하는 데에는 한계를 드러냈습니다. 이를 통해 AI 보조 사이버 보안 분야에 대한 기여를 하며, 앞으로의 연구 방향에 기초를 제공합니다.



### Exploring RL-based LLM Training for Formal Language Tasks with Programmed Rewards (https://arxiv.org/abs/2410.17126)
Comments:
          Accepted at BNAIC 2024

- **What's New**: 이 논문은 Proximal Policy Optimization (PPO)을 사용하여 명시적으로 프로그래밍된 보상 신호로부터 직접 강화 학습(Direct Reinforcement Learning)을 수행하는 가능성을 탐구합니다. 이는 간접 학습의 접근 방식인 인간 피드백을 통한 보상 모델을 매개로 하는 것과는 대조적입니다.

- **Technical Details**: 이 연구는 수학 및 프로그래밍과 같은 형식 언어(formal languages)로 표현된 작업에 중점을 두며, 명시적 보상 함수(explicit reward functions)를 프로그래밍하여 생성된 출력을 자동으로 평가할 수 있는 모델을 구축했습니다. 실험은 감정 정렬(sentiment alignment) 작업, 간단한 산술(arithmetic) 작업 및 더 복잡한 게임 합성(game synthesis) 작업을 포함합니다.

- **Performance Highlights**: 연구결과, 두 개의 형식 언어 작업에 대한 순수 RL 기반 훈련은 도전적이며, 간단한 산술 작업에서도 성공이 제한적이라는 점이 발견되었습니다. 탐사를 돕기 위해 새로운 배치 엔트로피 정규화(batch-entropy regularization) 항이 제안되었지만, 훈련은 여전히 완전히 안정적이지 않습니다. LLM의 직접 RL 훈련은 새로운 작업을 학습하는 것보다 상대적으로 작은 변경 사항, 즉 정렬(alignment)에 보다 적합할 수 있다는 것이 시사됩니다.



### Automated neuroradiological support systems for multiple cerebrovascular disease markers -- A systematic review and meta-analysis (https://arxiv.org/abs/2410.17124)
Comments:
          62 pages, 10 figures

- **What's New**: 이 논문은 뇌영상에서 확인 가능한 뇌혈관 질환(CVD) 마커를 지원하는 자동화 시스템에 관한 체계적인 리뷰를 제공합니다. 가장 최근의 상업적 소프트웨어와 연구 출판물 42개를 분석하였으며, 단일 이미지 모달리티에서 다양한 CVD 마커를 통합적으로 평가할 수 있는 시스템은 현재 존재하지 않음을 지적합니다.

- **Technical Details**: 이 연구에서는 백질 고강도(WMH), 허혈성 뇌졸중 병소(ISL), 그리고 기타 CVD 관련 마커(예: lacunes, 확대된 혈관 주위 공간(PVS), 뇌 미세출혈(CMB), 뇌 위축 등)를 포함한 검토 및 분석을 실시했습니다. MRI 및 CT 스캔에서 서로 다른 메커니즘으로 이러한 마커들을 측정하는 방법과 각각의 특징이 개별적으로 분석되었습니다.

- **Performance Highlights**: 상업적 시스템은 주로 급성 뇌졸중을 식별하는 데 집중하며, 연구 시스템은 후속 검사 및 일반 검사를 포함하여 WMH와 ISL을 파악하는 데 가장 많이 분석된 마커로 나타났습니다. 그러나 현재 시스템들은 모든 CVD 마커에 대한 포괄적인 분석을 수행할 수 없으며, 다기능 자동화 시스템의 확충이 필요한 상태입니다.



### Team Ryu's Submission to SIGMORPHON 2024 Shared Task on Subword Tokenization (https://arxiv.org/abs/2410.17094)
- **What's New**: 이 논문은 SIGMORPHON 2024 공유 작업에 제출된 팀 Ryu의 연구를 다루고 있습니다. 본 제출에서는 형태소 세분화(morphological segmentation) 방법이 서브워드 토크나이저(subword tokenizer)의 일부로 활용될 수 있는지를 탐구합니다. 두 가지 접근법인 통계적 세분화 방법인 Morfessor와 Transformer 기반의 시퀀스-투-시퀀스(segmentation model)를 사용했습니다.

- **Technical Details**: 서브워드 토크나이제이션(subword tokenization)은 NLP 응용 프로그램의 프로세스에서 널리 채택되고 있으며, 자주 사용되는 단어를 유지하고 긴 단어를 짧은 조각으로 쪼개어 어휘 크기를 줄이며, OOV(out-of-vocabulary) 단어를 처리하는 데 유용합니다. 본 연구는 형태소 세분화 방법과 신경망 seq2seq 모델을 기반으로 두 가지 서브워드 토크나이저 시스템을 제안함으로써, 각 방법의 성능을 비교 조사했습니다.

- **Performance Highlights**: 형태소 세분화 방법이 일반적으로 사용되는 서브워드 토크나이저와 유사한 성능을 보일 수 있음을 보여줍니다. 또한, 균형 잡힌 토큰 빈도 분포를 가진 토크나이저가 성능에 긍정적인 영향을 미치며, 이는 자주 사용되는 단어를 고유 토큰으로 유지함으로써 달성됩니다.



### Science Out of Its Ivory Tower: Improving Accessibility with Reinforcement Learning (https://arxiv.org/abs/2410.17088)
- **What's New**: 이번 논문에서는 과학 커뮤니케이션의 문제를 해결하기 위해 언어 모델을 조정하여 전문 용어가 제거된 더 이해하기 쉬운 학술 초록을 생성하는 강화 학습 프레임워크를 제안합니다. 이 모델은 약 6학년 수준으로 가독성을 높이며, 사실 정확성과 높은 언어 품질을 유지합니다.

- **Technical Details**: 이 모델은 'Reinforcement Learning from Accessibility Measures (RLAM)'를 사용하여 단어 및 문장 수준에서의 접근성 보상을 균형 있게 조정합니다. 제안된 접근 방식은 기존의 기계 학습 모델보다도 약 90%의 성능 향상을 이룹니다.

- **Performance Highlights**: 최고의 모델은 대학원 수준의 학술 초록을 고등학교 수준으로 조정하는 데 성공적으로 약 3학년 수준의 가독성 향상을 보였습니다. 이 모델은 과학 연구와 일반 대중 간의 간극을 줄이는 데 기여할 것으로 기대됩니다.



### UnStar: Unlearning with Self-Taught Anti-Sample Reasoning for LLMs (https://arxiv.org/abs/2410.17050)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 효과적인 아이디어로 'UnSTAR'를 제안합니다. 이 방법은 anti-sample을 활용하여 특정 정보를 선택적으로 잊게 하는 방법을 제시하고 있습니다.

- **Technical Details**: UnSTAR는 anti-sample을 이용하여 LLM의 학습을 취소(unlearn)하는 과정을 돕습니다. anti-sample은 실제 샘플과 반대되는 정보를 통해 연관된 기억을 되돌리도록 설계된 데이터 포인트입니다.

- **Performance Highlights**: UnSTAR를 통해 anti-sample을 사용하는 것이 LLMs의 목표 지향적 잊기에 있어 효율적이고 정밀한 전략이 됨을 보여주었습니다. 이 접근법은 개인 정보 보호 및 모델 수정의 새로운 길을 열어줄 가능성을 갖추고 있습니다.



### A Comparison of Baseline Models and a Transformer Network for SOC Prediction in Lithium-Ion Batteries (https://arxiv.org/abs/2410.17049)
- **What's New**: 이 논문은 리튬 이온 배터리의 충전 상태(SoC) 예측을 위한 데이터 기반 방법을 비교합니다. 여러 뉴럴 네트워크 모델과 회귀 모델을 활용한 SoC 추정은 배터리 관리 시스템(BMS)의 성능에 중요하며, 전기 자동차의 주행 거리 불안 문제를 해결하는 데 기여할 수 있습니다.

- **Technical Details**: 논문에서는 다양한 회귀 모델을 사용하여 SoC 예측 성능을 비교하였으며, 이에는 transformer 네트워크, 뉴럴 네트워크, 라쏘 회귀(lasso regression), 선형 회귀(linear regression), 결정 트리(decision tree)가 포함됩니다. 이 연구에는 BMW i3 배터리의 자연주행 데이터가 사용되었습니다.

- **Performance Highlights**: 결과적으로, 결정 트리 모델이 다른 모든 모델(복잡한 transformer 네트워크 포함)보다 우수한 성능을 보였습니다. 이 연구는 SoC 예측을 위한 데이터 기반 방법에 대한 가능성을 제시하며, 배터리의 주행 범위와 수명 최적화에 기여할 수 있습니다.



### GeoCode-GPT: A Large Language Model for Geospatial Code Generation Tasks (https://arxiv.org/abs/2410.17031)
- **What's New**: 지구 과학 분야에서 시공간(spatiotemporal) 데이터 및 모델링 작업에 대한 수요가 증가함에 따라, 지리 공간 코드 생성 기술이 생산성 향상의 핵심 요소로 자리잡고 있습니다. 이 논문은 GeoCode-PT 및 GeoCode-SFT 데이터셋을 공개하고, GeoCode-Eval 평가 데이터셋을 제공합니다.

- **Technical Details**: GeoCode-GPT-7B는 Code Llama-7B를 기반으로 하여 지리 공간 코드 생성을 위해 최초로 세밀하게 조정된 대형 언어 모델(LLM)입니다. QLoRA 및 LoRA 기술을 활용하여 사전 교육(pretraining)과 미세 조정(fine-tuning)을 수행하였고, 종합적인 지리 공간 코드 평가 프레임워크를 구축하였습니다.

- **Performance Highlights**: GeoCode-GPT는 다른 모델들에 비해 객관식 정확도가 9.1%에서 32.1% 향상되었으며, 코드 요약 능력은 1.7%에서 25.4% 향상되었고, 코드 생성 능력은 1.2%에서 25.1% 향상되었습니다.



### Can a Machine Distinguish High and Low Amount of Social Creak in Speech? (https://arxiv.org/abs/2410.17028)
Comments:
          Accepted in Journal of Voice

- **What's New**: 이번 연구는 사회적 크릭(사회적 creak)의 양을 구분하기 위해 머신러닝(ML) 기술을 적용한 점이 새롭습니다. 기존의 감각적 평가와 전통적인 음향 파라미터를 결합하여 사회적 크릭을 분석하는 방식에서 벗어나 자동 분류 방법을 탐색했습니다.

- **Technical Details**: 90명의 핀란드 여성 화자가 생성한 연속 음성 샘플에서 크릭의 양을 두 명의 음성 전문가가 감각적으로 평가했습니다. 이를 바탕으로, 저수준과 고수준의 크릭으로 두 가지 카테고리로 나누고, 음성 신호와 그 라벨을 사용하여 7개의 서로 다른 머신러닝 모델을 훈련시켰습니다. 각 모델은 3가지 스펙트럼 표현을 특징으로 사용했습니다.

- **Performance Highlights**: 연구의 결과, Adaboost 분류기를 사용한 mel-spectrogram 특징과 결정 트리 분류기를 사용한 mel-frequency cepstral coefficient 특징이 각각 71.1%의 정확도로 가장 우수한 성능을 보였습니다. 이 분류 시스템은 향후 머신러닝 기반 연구의 기준점으로 고려될 수 있습니다.



### Learning Mathematical Rules with Large Language Models (https://arxiv.org/abs/2410.16973)
Comments:
          4th MATH-AI Workshop at NeurIPS'24

- **What's New**: 본 논문에서는 대형 언어 모델(LLMs)이 분배 법칙(distributivity)이나 방정식 솔빙(equation solving)과 같은 특정 수학 규칙을 배우는 능력을 연구합니다. 이 연구의 주요 초점은 LLM이 훈련 중에 접하지 않은 맥락에서 이러한 규칙을 일반화하고 재사용하는 능력에 있습니다.

- **Technical Details**: 이 연구에서는 LLM에 특정 수학 규칙을 훈련시키기 위해 합성 데이터(synthetic data)를 구축하는 정교한 방법론을 제시합니다. 데이터는 수학 교과서처럼 만들어지며, 모델은 단순한 방정식 재배치(equation rearrangement), 변수를 포함한 선형 방정식에 대한 솔루션 도출, 방정식에서 변수 격리(isolating variables) 등 다양한 규칙을 학습합니다. 본 연구에는 하향(top-down) 및 상향(bottom-up) 일반화 기법이 포함됩니다.

- **Performance Highlights**: 모델은 수학적 복잡성을 증가시키면서도 이러한 규칙을 어느 정도 일반화할 수 있는 능력이 있음을 보여줍니다. 특히 관찰된 결과는 변수 명칭의 다양성 증가가 LLM의 분배 법칙에 대한 일반화 능력을 향상시킬 수 있다는 점입니다. 이 연구는 LLM의 수학적 언어 처리 능력을 이해하고 향상시키기 위한 중요한 기초를 제공합니다.



### Breaking ReAct Agents: Foot-in-the-Door Attack Will Get You In (https://arxiv.org/abs/2410.16950)
- **What's New**: 이번 논문에서는 '입구 발 디딤(feet-in-the-door, FITD) 공격'이라는 새로운 기법을 소개합니다. 이 공격은 무해한 요청을 이용하여 LLM 기반 에이전트를 속여 그 뒤에 오는 악의적인 요청을 수행하게 만드는 기법입니다.

- **Technical Details**: FITD 공격은 기존의 간접 프롬프트 주입(Indirect Prompt Injection, IPI) 공격을 확장하여, 주 공격 전 단계에서 방해 요청(distractor request)을 추가합니다. 이 기법은 LLM 에이전트가 안전성을 재평가하지 않고 위험한 명령을 실행하게 만드는 방식을 이용합니다.

- **Performance Highlights**: 실험 결과, FITD 공격은 모든 모델에 대해 공격 성공률(Attack Success Rate, ASR)을 최대 44.8%까지 향상시킬 수 있음을 보여주었습니다. 또한, 불확실한 방해 도구를 사용하더라도 공격이 성공한다는 점이 강조되었습니다.



### IdenBAT: Disentangled Representation Learning for Identity-Preserved Brain Age Transformation (https://arxiv.org/abs/2410.16945)
Comments:
          16 pages, 8 figures, 2 tables

- **What's New**: 본 연구에서는 개인 특성을 보존하면서 특정 연령대의 뇌 이미지를 생성하는 IdenBAT라는 새로운 구조를 제안합니다. 이를 통해 나이에 따른 관련 속성을 변환하는 동시에, 개인의 고유한 특성을 유지할 수 있습니다.

- **Technical Details**: IdenBAT는 conditional generative adversarial network (cGAN) 아키텍처를 기반으로 하며, encoder, identity extracting module (IEM), age injecting module (AIM), generator (G), discriminator (D)로 구성됩니다. IEM을 통해 이미지 특성이 나이에 관련된 특성과 무관한 특성으로 분리됩니다. AIM을 통해 나이에 대한 정보를 개인의 특성에 주입하여 최종적으로 개인의 정체성을 유지하면서도 목표 연령을 반영하는 이미지를 생성합니다.

- **Performance Highlights**: IdenBAT는 2D 및 3D 뇌 데이터셋에서 실험을 통해 기존의 최첨단 방법들과 비교하여 성능 우수성을 입증했습니다. 특히, 개인의 특성을 보존하며 정확한 연령 변환이 가능하다는 점에서 뛰어난 결과를 도출했습니다.



### Math Neurosurgery: Isolating Language Models' Math Reasoning Abilities Using Only Forward Passes (https://arxiv.org/abs/2410.16930)
Comments:
          21 pages, 29 figures

- **What's New**: 본 논문에서는 Math Neurosurgery(MathNeuro)라는 새로운 방법을 제안하여 대규모 언어 모델(LLM)에서 수학적 추론(Math reasoning)과 관련된 매개변수를 분리하는 방법을 개발했습니다. 이 방법은 기존 방법에서 수학적 매개변수를 효과적으로 분리할 수 있는 혁신적인 접근 방식을 소개하며, 매개변수 삭제가 전반적인 언어 능력에 미치는 영향을 최소화합니다.

- **Technical Details**: MathNeuro는 LLM의 매개변수가 수학적인 작업과 일반 언어 작업에서 각각 어떻게 중요한지를 분석하여, 수학적 작업에 중요한 매개변수를 고립시키는 방법입니다. 이 방법은 가중치(weights)와 활성화(activations)를 기반으로 하여 파라미터의 중요성을 계산하고, 일반적인 언어 작업에 중요하지 않은 매개변수를 제거함으로써 수학 특화 매개변수를 식별합니다. 이를 통해 다양한 크기의 LLM에서 수학적 추론 능력을 효과적으로 삭제하거나 향상시키는 데 성공했습니다.

- **Performance Highlights**: MathNeuro를 적용한 결과, LLM의 GSM8K(Generalized Synthetic Math 8K) 성능이 4-17% 향상되었습니다. 또한, 이 방법은 단일 샘플을 사용하여도 데이터 효율성이 뛰어남을 보여줍니다. 각 매개변수를 삭제하거나 스케일링(Skaling)하는 과정에서도 비수학적 성능에는 큰 영향을 미치지 않는 것으로 나타났습니다.



### EnvBridge: Bridging Diverse Environments with Cross-Environment Knowledge Transfer for Embodied AI (https://arxiv.org/abs/2410.16919)
- **What's New**: 최근에 소개된 EnvBridge는 LLM(대형 언어 모델)의 제어 능력을 향상시키기 위해 환경 간 지식 이전(Cross-Environment Knowledge Transfer) 과제를 해결하는 새로운 방법을 제안합니다.

- **Technical Details**: EnvBridge는 세 가지 주요 구성 요소로 이루어져 있습니다: 코드 생성(Code Generation), 메모리 검색(Memory-Retrieval), 이전 지식을 활용한 재계획(Re-Planning with Transferred Knowledge). 이 시스템은 이전 환경에서 성공적으로 실행된 로봇 제어 코드를 저장하고 이를 새로운 환경에 적절히 적용하기 위해 작업 시 시도하는 제어 코드를 LLM을 이용하여 생성합니다.

- **Performance Highlights**: EnvBridge는 RLBench, MetaWorld 및 CALVIN 테스트 환경에서 평균 69%의 성공률을 달성함으로써 기존의 코드 생성 및 재계획 베이스라인을 크게 능가했습니다. 이 방법은 다양한 지식 원천과 작업 지침에 대한 강력한 성능을 보이며, 다양한 환경 및 작업에 대한 적응력을 강조합니다.



### Mitigating Vanishing Activations in Deep CapsNets Using Channel Pruning (https://arxiv.org/abs/2410.16908)
- **What's New**: 이 논문은 Deep Capsule Networks의 vanishing activation(소실 활성화) 문제를 해결하는 새로운 접근 방식을 제시합니다. 기존의 모델 프루닝(pruning) 방법과 달리, 본 연구는 Capsule 네트워크의 성능을 향상시키기 위해 중요도를 평가하여 convolutional 채널을 프루닝하는 방법을 사용합니다.

- **Technical Details**: Capsule Networks는 Convolutional Neural Networks(CNNs)보다 part-whole relationships(부분-전체 관계) 학습에서 우수한 성능을 보입니다. 그러나 깊은 Capsule Networks는 깊이 증가에 따른 vanishing activations 문제가 발생하여 스케일러빌리티가 부족합니다. 이 연구는 구조적 프루닝(structured pruning)과 Correlation Coefficient Matrix(CCM) 손실을 결합하여 깊은 CapsNet의 소실 활성화를 완화하는 방법을 살펴봅니다.

- **Performance Highlights**: 제안된 방법은 비프루닝 모델보다 더 나은 정확도를 달성하며, Capsule Networks의 inactive capsules(비활성 캡슐) 수를 줄이면서 모델의 정확성을 높입니다. 논문에 포함된 실험 결과는 제안된 기법이 성능 향상에 효과적임을 보여줍니다.



### Contrasting Attitudes Towards Current and Future AI Applications for Computerised Interpretation of ECG: A Clinical Stakeholder Interview Study (https://arxiv.org/abs/2410.16879)
- **What's New**: 이번 연구는 ECG(심전도) 해석의 자동화 및 AI(인공지능) 기술에 대한 임상의의 태도를 조사한 첫 번째 연구입니다. 연구에서는 AI 알고리즘의 설명가능성과 신뢰성에 대한 중요성도 다루었습니다.

- **Technical Details**:  연구는 UK에서 활동 중인 23명의 임상을 대상으로 한 인터뷰를 통해 수행되었으며, 주제적 분석(inductive thematic analysis)을 통해 주제를 도출했습니다. 주요 주제로는 현재 시스템에 대한 신뢰 부족, 미래 AI 기술에 대한 긍정적인 태도, 알고리즘의 정확성과 설명가능성의 관계, AI의 교육 및 임상 역량에 대한 영향 등이 포함되었습니다.

- **Performance Highlights**: 임상 의사들은 현재의 컴퓨터화된 해석 방법에 대한 신뢰는 부족하지만, 미래의 AI 기술에 대해서는 긍정적인 반응을 보였습니다. 임상 의사들은 알고리즘의 정확성을 믿을 수 있을 경우 설명 가능성에 대한 우려가 적다는 입장을 보였습니다. 시각화 결과가 증명된 AI 해석을 선호하여 교육과 기술 향상의 수단으로 활용할 가능성을 보여주었습니다.



### Pedestrian motion prediction evaluation for urban autonomous driving (https://arxiv.org/abs/2410.16864)
Comments:
          7 pages, 2 figures, 4 tables This work has been submitted to the IEEE for possible publication

- **What's New**: 이 논문은 자율 주행 차량의 보행자 동작 예측 기술을 현재의 자율주행 프레임워크와 통합하여 실제 도시 환경에서 평가한 내용을 다룹니다. 기존 연구들은 일반적으로 세련된 데이터셋을 사용해 평가되었으나, 본 연구는 에스토니아 타르투에서 수집된 실제 주행 데이터를 활용하여 평가를 진행하였습니다.

- **Technical Details**: 연구에서는 최신 보행자 동작 예측 기술을 선택하고 엔지니어링 적응하여 자동 주행 프레임워크 내에서 실시간 성능을 가능하게 합니다. 다양한 예측 알고리즘의 성능을 평가하는 데 사용되는 전통적인 지표인 ADE/FDE와 같은 메트릭 대신 새로운 메트릭 도입의 필요성을 강조하며, 자율주행 시스템 내에서 센서 데이터 처리와 알고리즘 실행을 동시에 수행해야 하는 컴퓨팅 효율성을 요구합니다.

- **Performance Highlights**: 상대적으로 약 80%의 충돌 가능성 변동과 더불어 보행자의 안전을 확보하기 위해 보행자 동작 예측 기능이 필수적임을 강조합니다. 이 연구에서 소개된 다양한 최신 동작 예측 방법들에 대한 실험적 평가 결과는, 전통적 메트릭의 한계와 함께 현실 기반의 성능 검증을 요구하는 현대 자율 주행 기술 발전에 기여할 것으로 기대됩니다.



### Fast Graph Sharpness-Aware Minimization for Enhancing and Accelerating Few-Shot Node Classification (https://arxiv.org/abs/2410.16845)
Comments:
          NeurIPS24; The first two authors contributed equally to this work

- **What's New**: 본 논문에서는 Graph Neural Networks (GNNs)의 Few-Shot Node Classification (FSNC) 성능을 향상시키기 위해 Sharpness-Aware Minimization (SAM)을 GNN 훈련에 통합하는 방법을 제안합니다.

- **Technical Details**: SAM은 손실 경량의 플랫 미니마를 찾기 위해 모델 파라미터를 변동시키는 것을 핵심 아이디어로 하고 있지만 두 번의 순전파 및 역전파 단계를 요구하여 계산 비용이 높은 단점이 있습니다. 이를 보완하기 위해 우리는 FGSAM(Fast Graph Sharpness-Aware Minimization)이라는 알고리즘을 제안하며, 여기서는 GNN을 파라미터의 변동에 사용하고 MLP(Multi-Layer Perceptron)을 통해 변동 손실을 최소화하여 효율적으로 플랫 미니마를 찾는 방식으로 훈련을 가속화합니다. 또한 FGSAM+를 개발하여 특정 간격으로 정확한 변동을 수행합니다.

- **Performance Highlights**: FGSAM과 FGSAM+는 FSNC 작업에서 표준 SAM보다 낮은 계산 비용으로 더 나은 성능을 발휘하며, 특히 FGSAM+는 대부분의 경우 Adam과 비교해 더 빠른 최적화를 제공합니다. 또한, 제안된 방법들은 비동질 그래프(heterophilic graphs)에서의 노드 분류 작업에서도 경쟁력 있는 성능을 보여줍니다.



### Assessment of Transformer-Based Encoder-Decoder Model for Human-Like Summarization (https://arxiv.org/abs/2410.16842)
Comments:
          Pre-print

- **What's New**: 이 논문은 transformer 기반 BART 모델을 활용하여 자동 텍스트 요약을 개선하는 방법을 제시합니다. 특히, 다양한 기사에 대한 인간 평가를 통해 요약의 품질을 평가하며, 모델의 성능을 기존의 pretrained 모델과 비교합니다.

- **Technical Details**: 이 연구에서는 seq2seq(시퀀스-투-시퀀스) 프레임워크를 활용하여 transformer 기반의 BART 모델을 사용합니다. 모델은 encoder-decoder 아키텍처를 갖추고 있으며, 훈련 과정에서 다양한 샘플 기사를 다뤄 평가 메트릭으로 ROUGE 점수와 BERTScore를 사용합니다.

- **Performance Highlights**: 임상 결과에 따르면, 인간이 작성한 요약은 finetuned 모델이 생성한 추상적 요약보다 17% 더 사실적 일관성을 나타냅니다. 이는 기존 평가 메트릭들이 사실적 오류를 포착하는 데 민감하지 않음을 드러냅니다.



### PerspectiveNet: Multi-View Perception for Dynamic Scene Understanding (https://arxiv.org/abs/2410.16824)
Comments:
          6 pages, 2 figures

- **What's New**: 이 논문에서는 여러 카메라 뷰를 통해 상세한 설명을 생성하기 위한 새로운 경량 모델, PerspectiveNet을 소개합니다. 이 모델은 시각적 특성을 고정 크기의 텐서로 변환하기 위한 컴팩트 커넥터 모듈과 대형 언어 모델(LLM)을 활용하여 긴 문장을 효과적으로 생성하는 데 중점을 둡니다.

- **Technical Details**: PerspectiveNet의 커넥터 모듈은 시각적 특성을 LLM 임베딩에 매핑하고, 설명 생성에 필요한 주요 정보를 강조하며, 고정 크기의 피처 매트릭스를 생성하는 세 가지 주요 목표로 설계되었습니다. 또한, 정확한 프레임 순서 감지를 위한 부가적 작업을 통합하여 설명 생성을 위한 올바른 프레임 시퀀스를 검색할 수 있도록 합니다.

- **Performance Highlights**: Traffic Safety Description and Analysis 작업을 위해 훈련된 모델은 경량화되어 효율적인 훈련과 추론을 보장하며, 다양한 카메라 뷰에서 사건에 대한 자세하고 세밀한 설명을 생성하는 데 매우 효과적인 성능을 나타냅니다.



### Controlled Low-Rank Adaptation with Subspace Regularization for Continued Training on Large Language Models (https://arxiv.org/abs/2410.16801)
- **What's New**: 본 연구는 Large Language Models (LLMs)에서의 catastrophic forgetting을 완화하기 위한 Controlled LoRA (CLoRA)라는 새로운 방법론을 제안합니다. CLoRA는 LoRA 구조를 기반으로 한 서브스페이스 정규화 방법으로, 모델의 용량에 대해 최소한의 제약을 두면서도 출력 변화의 크기를 줄이는 것을 목표로 합니다.

- **Technical Details**: CLoRA는 updating matrix의 null space 방향에 제약을 두어 출력 변화의 크기를 감소시키며, 이는 파라미터 효율적인 미세 조정(parameter-efficient fine-tuning)의 일환으로 구분됩니다. 실험 결과, CLoRA는 기존의 LoRA 후속 방법들에 비해 인도메인 및 아웃 오브 도메인 평가에서 우수한 성능을 보였습니다. CLoRA는 orthogonal regularization을 도입하여 null space 방향에 대한 제약을 부여합니다.

- **Performance Highlights**: CLoRA는 기존 방법에 비해 LLM 미세 조정 작업에서 뛰어난 성과를 보여주었으며, catastrophic forgetting 완화에서도 우수한 효과를 나타냈습니다. 또한, 학습된 모델의 파라미터 분석 결과 CLoRA가 출력 변화의 크기를 줄이면서도 모델 용량에 미치는 영향이 최소화된 것을 확인하였습니다.



### One-Step Diffusion Distillation through Score Implicit Matching (https://arxiv.org/abs/2410.16794)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 이 논문에서는 'Score Implicit Matching (SIM)'이라는 새로운 방법론을 소개하며, 이는 기존의 diffuion 모델을 단일 단계 생성 모델로 효율적으로 변환합니다. 중요한 점은 SIM이 원래 모델과 거의 동일한 샘플 생성 능력을 유지하면서, 데이터 샘플 없이도 훈련 없이 적용할 수 있다는 것입니다.

- **Technical Details**: SIM은 점수 기반 손실 함수를 사용하여 pre-trained diffusion 모델을 단일 단계 생성 모델로 변환합니다. 이 접근법의 핵심 아이디어는 점수 기반 분산(score-based divergence) 간의 기울기를 효율적으로 계산할 수 있는 'score-gradient theorem'을 이용한다는 것입니다. 이를 통해 모델의 훈련을 효율적으로 진행할 수 있습니다.

- **Performance Highlights**: CIFAR10 데이터셋에서 SIM은 비조건부 생성의 경우 FID(Frechet Inception Distance)가 2.06, 클래스 조건부 생성의 경우 1.96을 달성했습니다. 또한, 텍스트-이미지(T2I) 생성에 있어 기존의 다단계 모델과 비교해 성능 저하 없이 뛰어난 심미적 점수 6.42를 기록하며 단일 단계 생성 모델들이 SDXL-TURBO 및 HYPER-SDXL과 같은 기존 모델들보다 높은 성능을 보였습니다.



### Correct after Answer: Enhancing Multi-Span Question Answering with Post-Processing Method (https://arxiv.org/abs/2410.16788)
Comments:
          Accepted by EMNLP 2024 Findings

- **What's New**: 이번 연구에서 제안된 Answering-Classifying-Correct (ACC) 프레임워크는 Multi-Span Question Answering (MSQA) 문제에서 잘못된 예측을 처리하기 위한 새로운 후처리 전략을 도입했습니다. 이는 기존 모델들이 Gold answers에만 기반하여 훈련된 점에서 개선을 이루고자 하며, 불완전한 예측을 수정하고 잘못된 예측을 제외함으로써 예측의 질을 향상시키는 것을 목표로 합니다.

- **Technical Details**: ACC 프레임워크는 세 가지 단계로 구성됩니다. 첫 번째로, 분류기를 통해 예측을 '정확한 예측', '부분적으로 정확한 예측', '잘못된 예측'으로 분류합니다. 두 번째로, 수정기를 통해 '부분적으로 정확한 예측'을 수정합니다. 마지막으로, '잘못된 예측'을 제외하여 최종 예측을 얻습니다. 이를 통해 각 데이터셋에 대해 정확도(Exact Match, EM) 점수를 제공함으로써 MSQA 성능을 향상시킵니다.

- **Performance Highlights**: 실험 결과, ACC 프레임워크를 적용한 후 MultiSpanQA 데이터셋에서 EM F1 점수가 Tagger-RoBERTa 모델에서 69.05%에서 72.26%로, BART-base 모델에서 65.57%에서 76.31%로 증가했습니다. 이는 ACC 프레임워크가 잘못된 예측을 효과적으로 줄이고 더 많은 정확한 예측을 얻어 예측의 질을 높인다는 것을 보여줍니다.



### Beyond Retrieval: Generating Narratives in Conversational Recommender Systems (https://arxiv.org/abs/2410.16780)
- **What's New**: 이 논문은 대화형 추천 시스템을 위한 자연어 생성 작업에서 REGEN이라는 새로운 데이터셋을 소개합니다. REGEN은 Amazon 제품 리뷰 데이터를 풍부한 사용자 내러티브로 확장하여 생성된 데이터셋입니다. 이 데이터셋은 사용자 선호도에 대한 개인화된 설명, 추천 아이템에 대한 제품 추천 및 사용자 구매 이력 요약을 포함하고 있습니다.

- **Technical Details**: REGEN은 사용자-아이템 상호작용 신호로부터 일관성 있는 자연어 출력을 생성하는 작업과 프레임워크를 도입합니다. 이 연구에서는 협업 필터링(CF) 신호와 콘텐츠 임베딩을 통합하는 융합 아키텍처를 제안하여 LLM의 입력으로 사용합니다. 실험 결과, CF와 콘텐츠 임베딩을 결합함으로써 언어 메트릭에서 4-12%의 성능 향상을 보여주었습니다.

- **Performance Highlights**: 제안된 모델은 사용자의 과거 상호작용을 바탕으로 풍부한 내러티브를 생성할 수 있다는 점에서 인간과 같은 대화형 추천을 생성하는 데 효과적임을 입증했습니다. 또한 데이터를 통해 CF와 콘텐츠 임베딩이 이 새로운 생성 작업에 어떻게 기여하는지 분석하였습니다.



### The Scene Language: Representing Scenes with Programs, Words, and Embeddings (https://arxiv.org/abs/2410.16770)
Comments:
          Project page: this https URL

- **What's New**: Scene Language는 시각 장면의 구조, 의미 및 정체성을 간결하고 정확하게 설명하는 새로운 시각 장면 표현 방식입니다. 이 방식은 프로그램, 자연어의 단어 및 임베딩의 세 가지 주요 구성 요소로 장면을 표현합니다.

- **Technical Details**: Scene Language는 장면의 계층적 및 관계적 구조를 정의하는 프로그램, 각 엔티티의 의미적 클래스를 요약하는 자연어의 단어 및 각 엔티티의 시각적 정체성을 캡처하는 임베딩으로 구성됩니다. 이 표현 방식은 사전 훈련된 언어 모델을 기반으로 하여 훈련이 필요 없는 추론 기술로 텍스트 또는 이미지 입력에서 추론할 수 있습니다.

- **Performance Highlights**: 기존의 장면 그래프와 같은 표현 방식과 비교하여 Scene Language는 복잡한 장면을 더 높은 충실도로 생성하면서 장면 구조를 명시적으로 모델링하여 정밀한 제어 및 편집을 가능하게 합니다. 이 시스템은 고품질 3D 및 4D 장면 생성 및 편집을 위한 강력하고 자동화된 솔루션을 제공합니다.



### Survival Models: Proper Scoring Rule and Stochastic Optimization with Competing Risks (https://arxiv.org/abs/2410.16765)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2406.14085

- **What's New**: 본 논문에서는 right-censored data (우측 검열 데이터)를 다루기 위해 survival analysis (생존 분석) 과정을 개선한 새로운 기법인 SurvivalBoost를 소개합니다. 이 기법은 competing risks (경쟁 위험) 문제에 초점을 맞춰, 모든 관측치가 독립적으로 평가될 수 있도록 합니다.

- **Technical Details**: 기존의 경쟁 위험 모델은 구조와 손실에 대한 결합을 가지고 있어 한정된 문제를 겪고 있으며, 본 연구는 censoring-adjusted (검열 조정) separable scoring rule (분리 가능한 점수 규칙)을 설계하여 이러한 문제를 해결했습니다. 이는 손실 추정이 outcome probabilities (결과 확률)를 추정하고, stochastic optimization (확률적 최적화)을 가능하게 하여, gradient boosting trees (그래디언트 부스팅 트리)에서 효율적으로 사용될 수 있습니다.

- **Performance Highlights**: SurvivalBoost는 4개의 실제 데이터 세트에서 12개의 최첨단 모델을 여러 지표에서 능가하였으며, 경쟁 위험 설정 및 생존 설정 모두에서 뛰어난 예측 능력과 훌륭한 보정(calibration), 시간이 지나도 예측할 수 있는 능력, 그리고 빠른 계산 시간을 제공합니다.



### Deep-Sea A*+: An Advanced Path Planning Method Integrating Enhanced A* and Dynamic Window Approach for Autonomous Underwater Vehicles (https://arxiv.org/abs/2410.16762)
Comments:
          Accepted by 2024 International Conference on Big Data, Artificial Intelligence and Internet of Things Engineering (ICBAIE 2024)

- **What's New**: 이 논문은 심해 자원 탐사에 대한 수요가 증가함에 따라, 보다 강력한 탐지 로봇의 개발 필요성을 강조하며, 개선된 A* 알고리즘과 Dynamic Window Approach (DWA)를 통합한 경로 계획 방법론을 제안합니다.

- **Technical Details**: 제안된 방법론은 전통적인 A* 알고리즘의 검색 방향을 최적화하고 향상된 평가 함수를 도입하여 경로 검색 속도를 증가시키고 계산 부하를 줄입니다. 또한 경로 매끄러움과 연속성을 개선하여 급격한 회전을 최소화하는 경로 스무딩 과정이 포함되어 있습니다. DWA를 활용하여 전역 경로 계획과 지역 동적 장애물 회피를 통합함으로써 다이나믹 환경에서 수중 로봇의 실시간 반응성을 향상시킵니다.

- **Performance Highlights**: 시뮬레이션 결과, 제안된 방법이 전통적인 A* 알고리즘에 비해 경로 매끄러움, 장애물 회피, 실시간 성능 모두에서 우수한 성능을 보이는 것을 입증하였습니다. 정적 및 동적 장애물이 있는 복잡한 환경에서의 강력함은 자율 수중 차량 (AUV) 내비게이션 및 장애물 회피의 잠재력을 강조합니다.



### Towards Efficient IMC Accelerator Design Through Joint Hardware-Workload Co-optimization (https://arxiv.org/abs/2410.16759)
- **What's New**: 본 연구에서는 다양한 워크로드(워크로드) 를 지원하는 일반화된 인-메모리 컴퓨팅(IMC) 하드웨어 설계를 위한 통합 최적화 프레임워크를 제안합니다. 이를 통해 보다 효율적이고 유연한 하드웨어를 구현할 수 있게 됩니다.

- **Technical Details**: 제안된 프레임워크는 IMC 칩 아키텍처 파라미터를 최적화하며, 이는 주어진 다양한 워크로드를 동시에 고려하여 설계를 진행합니다. 이 과정에서 단일 최대 워크로드를 위한 별도의 하드웨어 최적화 방식을 대체합니다.

- **Performance Highlights**: 자체 연구 결과, VGG16, ResNet18, AlexNet, MobileNetV3에 대해 각각 36%, 36%, 20%, 69% 향상된 에너지-지연-면적(energy-latency-area) 성능을 보였습니다. 또한, 일반화된 IMC 하드웨어가 특정 워크로드 전용 IMC 설계에 비해 가지는 성능 Trade-off(트레이드오프)와 손실을 정량화하였습니다.



### SpikMamba: When SNN meets Mamba in Event-based Human Action Recognition (https://arxiv.org/abs/2410.16746)
Comments:
          8 pages, 4 figures

- **What's New**: 본 연구에서는 SpikMamba 프레임워크를 제안하여 이벤트 카메라로부터 수집된 데이터의 공간적 희소성과 높은 시간적 해상도를 효과적으로 처리하여 인간 행동 인식(HAR)의 성능을 개선합니다.

- **Technical Details**: SpikMamba는 스파이킹 신경망(Spiking Neural Networks, SNN)과 Mamba를 결합하여 이벤트 데이터의 글로벌 및 로컬 시계열 의존성(global and local temporal dependencies)을 모델링합니다. 이 프레임워크의 핵심 설계에는 스파이크 기반의 선형 주의 메커니즘(spike-based linear attention mechanism)이 포함되어 있습니다.

- **Performance Highlights**: SpikMamba는 PAF, HARDVS, DVS128, E-FAction 데이터셋에서 기존 최첨단 방법들보다 각각 1.45%, 7.22%, 0.15%, 3.92% 성능을 초과하여 놀라운 인식 성능을 달성했습니다.



### Corrected Soft Actor Critic for Continuous Contro (https://arxiv.org/abs/2410.16739)
- **What's New**: 이번 논문은 Soft Actor-Critic (SAC) 알고리즘의 액션 샘플링 방법에 대한 새로운 접근법을 제시합니다. 기존의 tanh 변환으로 인해 발생하는 행동 분포의 왜곡 문제를 해결하기 위해, 직접적으로 가장 가능성이 높은 행동을 식별하고 선택하는 새로운 샘플링 방법을 도입하였습니다.

- **Technical Details**: SAC 알고리즘은 정책 기울기 기법과 엔트로피 극대화 기법을 통합하여 복잡한 환경에서 효율적인 탐색과 강인한 학습을 촉진합니다. 그러나 기존의 tanh 변환이 Gaussian 분포를 비선형적으로 왜곡하여 샘플링 오류를 유발하고, 이로 인해 고차원 행동 공간에서 최적의 행동 선택에 악영향을 미친다는 것이 연구의 주요 배경입니다.

- **Performance Highlights**: 제안된 샘플링 방법은 SAC 알고리즘의 성능을 획기적으로 향상시켜, 수렴 속도를 빠르게 하고 누적 보상을 높였습니다. 표준 연속 제어 벤치마크에서의 광범위한 실험 결과가 이를 뒷받침하고 있습니다.



### Enhancing Low-Resource ASR through Versatile TTS: Bridging the Data Gap (https://arxiv.org/abs/2410.16726)
- **What's New**: 최근 ASR 시스템이 대규모 데이터셋에서 뛰어난 성능을 보여주고 있지만, 자원이 부족한 환경에서는 여전히 챌린지가 존재합니다. 본 논문은 다양한 TTS 모델을 활용한 ASR 데이터 증강 방법이 효과적이며 경제적이라는 것을 보여줍니다.

- **Technical Details**: CosyVoice는 다국어를 지원하는 제로샷 TTS 시스템으로, 감독된 의미 토큰을 기반으로 합니다. 이 시스템은 텍스트 인코더, 음성 토크나이저, 큰 언어 모델(LLM), 조건부 흐름 대응 모델을 포함하여, 높은 내용 일관성과 화자 유사성을 확보합니다. 이를 통해 다채롭고 자연스러운 음성을 생성할 수 있습니다.

- **Performance Highlights**: 다양한 저자원 데이터셋에서의 실험 결과, TTS 모델을 통한 데이터 증강이 ASR 성능을 유의미하게 개선했으며, 특히 텍스트 다양성이 ASR 성능에 더 큰 영향을 미친다는 것을 발견했습니다.대략 50명에서 100명 정도의 화자가 충분하며, 복잡한 접근 방식 없이 간단한 데이터 결합 방식으로도 뛰어난 결과를 얻었습니다.



### Collapse or Thrive? Perils and Promises of Synthetic Data in a Self-Generating World (https://arxiv.org/abs/2410.16713)
- **What's New**: 이번 연구에서 제시된 모델 붕괴(model collapse) 이론은 새롭게 생성된 AI 콘텐츠가 이전 모델로부터 학습한 데이터의 영향을 받는 방식에 대해 심층적으로 분석합니다. 연구자들은 두 가지 대안 시나리오인 '대체(replace)'와 '축적(accumulate)'을 비교함으로써, 모델 훈련 방식과 데이터의 양이 모델의 성능에 미치는 영향을 검토합니다.

- **Technical Details**: 두 가지 주요 시나리오를 통해 모델 붕괴의 가능성을 평가합니다. 첫 번째는 '대체' 시나리오로, 새로운 모델은 이전 모델의 합성(synthetic) 데이터만을 사용하여 훈련됩니다. 두 번째는 '축적' 시나리오로, 모든 이전 데이터(real + synthetic)가 모델 훈련에 사용됩니다. 연구는 이 두 가지 시나리오가 각기 다른 세 가지 생성 모델(task settings)에서 어떻게 작용하는지 분석했습니다.

- **Performance Highlights**: 모델 시험 손실(test loss)은 '축적' 시나리오에서 더 낮은 값으로 안정화를 이루며, '대체' 시나리오에서처럼 성능이 급격히 저하되지 않음을 보여줍니다. 특히, 실제 데이터(real data)의 비율이 작을 때 합성 데이터의 양이 모델 성능에 중대한 영향을 미친다는 점을 발견했습니다. 이 연구 결과는 향후 생성 모델들이 붕괴할지 지속적으로 발전할지를 예측하는 데 매우 중요한 통찰력을 제공합니다.



### Development of CNN Architectures using Transfer Learning Methods for Medical Image Classification (https://arxiv.org/abs/2410.16711)
- **What's New**: 본 논문에서는 의료 이미지 분류에서 CNN 아키텍처의 발전을 조사하고 Transfer Learning 기법을 활용하여 효율성과 정확성을 높이는 방법을 제안합니다.

- **Technical Details**: Transfer Learning을 활용한 Convolutional Neural Networks (CNN) 아키텍처 개발에 관한 연구입니다. 주요 이미지 분류 과제를 시간 맵 모델(timeline mapping model)로 분석하여 최적의 CNN 아키텍처를 선택할 수 있는 정보를 제공합니다.

- **Performance Highlights**: 의료 영상 분류에서 Transfer Learning을 적용한 CNN을 통해 효율성과 정확적인 결과를 도출해냈습니다. 이를 통해 최첨단 CNN 아키텍처를 선택하는 데 도움이 됩니다.



### Universal approximation property of ODENet and ResNet with a single activation function (https://arxiv.org/abs/2410.16709)
Comments:
          14 pages

- **What's New**: ODENet와 ResNet의 보편 근사 성질(Universal Approximation Property, UAP)을 연구하며, 기본적인 벡터 필드와 아핀 매핑의 단일 합성으로 주어진 역학 시스템을 고려합니다. 이는 실제 머신러닝 시스템에서 가장 일반적인 ODENet 또는 ResNet의 벡터 필드 구성입니다.

- **Technical Details**: ODENet은 초기 값에서 ODE 시스템의 최종 값으로 매핑되는 기능을 갖고 있습니다. 이 기능은 ResNet과 유사한 구조를 가지며, 주로 레이어와 입력 값 간의 관계를 정의합니다. 연구에서 제안된 제한된 벡터 필드를 가진 ODENet과 ResNet이 일반 벡터 필드를 가진 ODENet을 균일하게 근사할 수 있음을 보여줍니다. 여기에 사용된 활성화 함수는 sigmoid, tanh 및 ReLU와 같은 전통적인 함수들이 포함됩니다.

- **Performance Highlights**: ODENet이 Lp-노름(ℓ^p-norm)에서 보편 근사 성질을 갖고 있음을 보여주었으며, 특히 이 논문에서는 UAP가 1차원 경우의 sup-노름에서 성립함을 보였습니다. 이는 더 깊은 네트워크와 고차원 문제를 효과적으로 해결하기 위한 가능성을 제시합니다.



### PLDR-LLM: Large Language Model from Power Law Decoder Representations (https://arxiv.org/abs/2410.16703)
Comments:
          22 pages, 4 figures, 10 tables

- **What's New**: 이번 논문에서는 전통적인 LLM 아키텍처와는 다른 Power Law Graph Attention 메커니즘을 사용해 deductive(연역적) 및 inductive(귀납적) 출력을 생성하는 PLDR-LLM 아키텍처를 소개합니다. 또한 Directed Acyclic Graph(DAG) 손실을 메트릭으로 활용하여 모델 성능 향상을 모색합니다.

- **Technical Details**: PLDR-LLM은 Power Law Graph Transformer를 기반으로 하며, 입력 샘플의 가중 그래프 표현을 덕분에 주의(attention) 메커니즘을 통해 모델의 특성을 학습합니다. 이 과정에서 비선형 변환(non-linear transformations)과 선형 변환(linear transformations)을 모두 활용하여 입력의 특성을 배우고, 적용한 DAG 손실이 성능 향상에 기여하는 구조를 가지고 있습니다.

- **Performance Highlights**: PLDR-LLM은 배치 크기 32로 약 8B 토큰을 사용해 선행 학습(pretraining) 되며, 기존의 스케일된 도트-프로덕트 LLM과 비교해 경쟁력 있는 성능을 보입니다. 또한 이 모델은 기울기 노이즈에 강인하며, DAG 규제를 통해 벤치마크 성적이 향상되었습니다.



### Graph Transformers Dream of Electric Flow (https://arxiv.org/abs/2410.16699)
- **What's New**: 이 논문은 그래프 데이터에 적용된 선형 Transformer가 전기 흐름(electric flow) 및 고유벡터 분해(eigenvector decomposition)와 같은 전형적인 문제를 해결할 수 있음을 이론 및 실험을 통해 보여주고 있습니다. 입력으로는 그래프 발생 행렬(incidence matrix)만 사용되며, 다른 명시적인 위치 인코딩(positional encoding) 정보는 제공되지 않습니다.

- **Technical Details**: 선형 Transformer는 기존 Transformer와 유사하지만 softmax 기반의 활성화가 선형 주의(linear attention)로 대체되고 MLP 계층이 선형 계층으로 대체됩니다.研究는 Transformer의 가중치 구성에 대한 명시적인 정의를 제공하고, 기본 알고리즘의 오류에 의해 생성된 Transformer의 오류를 제한합니다. 또한 경량화된 Transformer로 적은 수의 파라미터로도 동일한 구조를 구현할 수 있음을 보여줍니다.

- **Performance Highlights**: 합성 데이터(synthetic data)에 대한 실험과 실제 분자 회귀(molecular regression) 과제에서 선형 Transformer가 기본 평균값보다 효과적인 위치 인코딩을 학습할 수 있음을 관찰했습니다. QM9 및 ZINC 데이터셋을 활용한 실험에서, 선형 Transformer 구조가 원래의 Laplacian 고유벡터 기반 위치 인코딩보다 훨씬 높은 성능을 보였습니다.



### MPT: A Large-scale Multi-Phytoplankton Tracking Benchmark (https://arxiv.org/abs/2410.16695)
- **What's New**: 본 논문에서는 다양한 배경 정보와 관측 중의 움직임 변화를 포괄하는 다중 미세조류 추적(Multiple Phytoplankton Tracking, MPT)이라는 새로운 벤치마크 데이터셋을 제안합니다. 이 데이터셋은 27종의 미세조류 및 미세 해양 생물과 140개의 비디오로 구성되어 있으며, 복잡한 수중 환경을 시뮬레이션할 수 있는 14가지 배경을 포함하고 있습니다.

- **Technical Details**: MPT 데이터셋을 바탕으로, 우리는 미세조류의 실시간 추적을 위한 다중 객체 추적 프레임워크인 Deviation-Corrected Multi-Scale Feature Fusion Tracker(DSFT)를 개발했습니다. DSFT는 두 가지 주요 문제를 해결하기 위해 1) Deviation Correction Method (DCM)을 도입하여 개별 목표에 대한 주의를 보장하고, 2) Multi-scale Feature Similarity Fusion (MFSF)을 제안하여 다양한 크기의 객체에 대한 유사성을 계산합니다.

- **Performance Highlights**: MPT 데이터셋과 DSFT의 유효성 및 우수성은 광범위한 실험을 통해 입증되었습니다. 이 연구는 미세조류 모니터링을 위한 효과적인 해결책을 제공하면서, 기존 다중 객체 추적 방법의 한계를 극복하는 데 기여합니다.



### CoPS: Empowering LLM Agents with Provable Cross-Task Experience Sharing (https://arxiv.org/abs/2410.16670)
Comments:
          25 pages, 5 tables, 3 figures

- **What's New**: 이 논문에서는 CoPS(Cross-Task Experience Sharing)라는 일반화 가능한 알고리즘을 제안하여 순차적 추론(sequential reasoning)을 개선합니다. CoPS는 이전 작업에서의 경험을 공유하고 선택함으로써 성능을 극대화하고 배포 변화(distribution shifts)로 인한 위험을 최소화합니다.

- **Technical Details**: CoPS는 에이전트의 이전 작업에서 얻은 경험을 활용하여 분포에 맞는 경험을 선택하는 전략을 기반으로 합니다. 이 방법은 경험 선택을 위한 비관주의(pessimism)의 원칙에 근거하고 있으며, pretrained LLM에 의해 생성된 작업 의존성 시험 분포(task-dependent trial distribution)와 에이전트의 분포의 일치를 중요시합니다.

- **Performance Highlights**: CoPS는 Alfworld, Webshop, HotPotQA와 같은 벤치마크에서 최첨단 경험 보조 추론(experience-assisted reasoning) 및 반사 기반 추론(reflection-driven reasoning) 방법들과 비교하여 일관되게 우수한 성능을 보여주었습니다. 또한 자원 제약이 있는 상황에서도 뛰어난 샘플 효율성을 제공합니다.



### Satori: Towards Proactive AR Assistant with Belief-Desire-Intention User Modeling (https://arxiv.org/abs/2410.16668)
- **What's New**: Satori라는 새로운 증강 현실(AR) 보조 시스템이 제안되었으며, 이는 사용자의 상태와 환경을 모델링하여 적극적인 안내를 제공합니다.

- **Technical Details**: Satori는 Belief-Desire-Intention (BDI) 모델과 최신 다중 모달 대형 언어 모델(LLM)을 결합하여 상황에 적합한 가이드를 추론합니다. 이 시스템은 사용자의 즉각적인 작업과 장기 목표를 이해해야 합니다.

- **Performance Highlights**: Satori는 16명 사용자를 대상으로 한 연구에서 디자이너가 제작한 시스템과 동등한 성능을 입증했습니다. 이는 AR 가이드를 상황에 맞춰 적절한 시점에서 제공하여 사용자 경험을 향상시킵니다.



### Visual Question Answering in Ophthalmology: A Progressive and Practical Perspectiv (https://arxiv.org/abs/2410.16662)
- **What's New**: 이 리뷰 논문은 안과학에서 시각 질문 응답(Visual Question Answering, VQA)의 최근 발전과 미래 전망을 다루고 있습니다. 특히, 컴퓨터 비전과 자연어 처리(Natural Language Processing, NLP)를 융합하여 안과 이미지를 이해하고 질의에 응답하는 방법을 제시합니다.

- **Technical Details**: VQA는 동시에 여러 종류의 정보를 처리할 수 있는 멀티모달(Multimodal) 접근 방식을 통해 안과 이미지를 해석하는 데 도움을 줍니다. 이 과정에서 대형 언어 모델(Large Language Models, LLM)이 VQA 프레임워크의 다양한 구성 요소를 향상시키는 역할을 수행합니다. 그러나 주석이 달린 멀티모달 이미지 데이터셋의 부족, 포괄적이고 통일된 평가 방법의 필요성, 현실 세계에서의 적용 효과성에 대한 장벽 등이 주요 도전 과제로 남아 있습니다.

- **Performance Highlights**: 안과 VQA 시스템은 의료 전문가와 AI 전문가 간의 협력을 통해 기존의 장애물을 극복하고 안과 질병의 진단 및 치료를 향상시킬 수 있는 가능성을 가지고 있습니다. 앞으로 VQA 기술과 대형 언어 모델의 발전은 안과 분야에서의 진단 정확도를 크게 향상시킬 것으로 기대됩니다.



### RKadiyala at SemEval-2024 Task 8: Black-Box Word-Level Text Boundary Detection in Partially Machine Generated Texts (https://arxiv.org/abs/2410.16659)
Comments:
          published at naacl 2024

- **What's New**: 이번 논문은 기계 생성 텍스트와 인간 생성 텍스트를 구별하는 데 있어 중요한 도전 과제를 다루고 있습니다. 기존 시스템들이 전체 텍스트의 생성 여부를 판단하는 데 초점을 맞췄다면, 본 연구에서는 문서 내 단어 수준에서 기계가 생성한 부분을 식별하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 제안된 모델은 DeBERTa-CRF를 기반으로 하여 훈련되었습니다. 특히 Conditional Random Fields (CRF)가 포함되어 있어 패턴 인식 능력이 향상되었습니다.본 모델은 M4GT-bench 데이터셋을 사용하여 기계 생성과 인간 작성된 텍스트의 경계를 효과적으로 식별할 수 있도록 설계되었습니다.

- **Performance Highlights**: 모델의 주요 성능 지표는 Mean Average Error (MAE)로, 개발 세트에서 3.53의 MAE를 기록하였으며, 테스트 세트에서는 21.535로 측정되었습니다. 기존 시스템들과 비교했을 때, 제안된 모델은 더 높은 정확도를 보여주었습니다.



### Semantic-guided Search for Efficient Program Repair with Large Language Models (https://arxiv.org/abs/2410.16655)
- **What's New**: 이 논문에서는 작은 크기의 LLM (1B-7B 매개 변수)의 빔 크기를 늘리는 것이 GPU 자원 소비를 광범위하게 증가시키고, LLM 기반의 자동 프로그램 수리 (APR)에서 메모리 과부하로 인한 반복적인 충돌을 초래할 수 있음을 입증합니다. 기존의 메모리 소비를 줄이기 위한 방법들이 이론적 분석과 실험을 통해 여전히 효과적이지 않음을 보여줍니다. 이를 해결하기 위해 FLAMES라는 새로운 LLM 기반 APR 기법을 도입합니다.

- **Technical Details**: FLAMES는 의미 기반의 패치 생성을 활용하여 수리 효과성과 메모리 효율성을 향상시킵니다. 기존의 빔 검색 방식 대신 그리디 디코딩(greedy decoding)을 사용하여 메모리 효율성을 높이고, 의미 기반 우선 검색(semaphore-guided best-first search) 알고리즘을 통해 더 잠재적으로 우수한 수리 후보를 탐색합니다. 각 디코딩 단계에서 FLAMES는 테스트 검증으로부터의 의미 피드백을 사용하여 탐색할 가장 유망한 토큰을 선택합니다.

- **Performance Highlights**: FLAMES는 Defects4J 및 HumanEval-Java 데이터셋에 대한 실험 평가에서 기존 LLM 기반 APR에 비해 메모리 소비를 최대 83% 줄이고 수리 과정을 가속화하였습니다. FLAMES는 Defects4J 및 HumanEval-Java 데이터셋에서 각각 333 및 163개의 버그에 대해 133과 103개의 올바른 수정을 성공적으로 생성하며, 기존의 최첨단 기법(SOTA)보다 각각 10개와 11개의 버그를 더 고치는 성능을 보여주었습니다.



### Enhancing Two-Player Performance Through Single-Player Knowledge Transfer: An Empirical Study on Atari 2600 Games (https://arxiv.org/abs/2410.16653)
- **What's New**: 이번 연구는 두 플레이어 게임에서 강화 학습 알고리즘이 단일 플레이어 게임의 지식을 활용하면 더 효율적으로 훈련되고 성능이 개선될 수 있음을 제안합니다. 이를 통해 10가지 다른 Atari 2600 환경에서 실험을 실시하였습니다.

- **Technical Details**: 연구에서는 Atari 2600 RAM을 입력 상태로 사용하여 단일 플레이어 환경에서 학습한 지식을 두 플레이어 환경으로 전이하는 방법을 분석합니다. RAM 복잡도 계산 방법도 제안하여 성능과의 연관성을 논의합니다.

- **Performance Highlights**: 단일 플레이어 Atari 환경에서의 지식 전이를 활용함으로써 두 플레이어 환경에서 처음부터 훈련하는 것보다 훈련 시간 단축과 평균 총 보상 증가를 달성하였습니다.



### GE2E-KWS: Generalized End-to-End Training and Evaluation for Zero-shot Keyword Spotting (https://arxiv.org/abs/2410.16647)
Comments:
          8 pages, 6 figures, 2 tables The paper is accepted in IEEE Spoken Language Technology (SLT) 2024

- **What's New**: 본 논문은 GE2E-KWS라는 일반화된 엔드 투 엔드(Generalized End-to-End) 훈련 및 평가 프레임워크를 제안합니다. 이 프레임워크는 맞춤형 키워드 스포팅을 위해 설계되었으며, 훈련 배치에서 키워드에 따라 구분된 등록 발화를 그룹화하고, 이들의 임베딩 중심점을 비교하여 손실을 계산합니다. 이러한 접근은 실행 시간 등록 및 검증 단계를 시뮬레이션하며, SOTA(분야의 state-of-the-art)를 초월할 수 있는 훈련 속도와 수렴 안정성을 제공합니다.

- **Technical Details**: GE2E-KWS 모델은 등록 발화를 단어에 따라 그룹화하여 여러 발화의 중심점을 계산합니다. 이 중심점은 모든 다른 테스트 발화의 임베딩과 비교되어 손실을 계산하는 데 사용됩니다. 본 연구에서는 Conformer(컨포머) 모델을 사용하여 최적화하였고, 419KB로 양자화된 모델이 7.5GB ASR(Automatic Speech Recognition) 인코더에 비해 23.6% 더 나은 AUC(Area Under Curve)를 기록했습니다. 또한, 새로운 키워드를 위해 추가 훈련 없이도 지속적으로 잘 작동할 수 있도록 설계되었습니다.

- **Performance Highlights**: 우리 모델은 낮은 메모리 발자국을 유지하며, 자동 스트리밍 전환 기능으로 사용하여 연속적으로 기기에서 실행될 수 있습니다. 우리는 다양한 크기의 LSTM 및 Conformer 모델을 훈련 및 평가하여 모델 품질과 크기 간의 절충점을 보여줍니다. 실험 결과, GE2E 손실로 훈련된 모델이 효율적인 훈련과 더 빠른 안정적인 수렴을 보였으며, 키워드 매칭 정확도를 직접 측정하기 위한 새로운 평가 프로세스를 설계하여 키워드 스포팅의 품질을 보다 신뢰할 수 있는 방법으로 평가할 수 있었습니다.



### Chatting with Bots: AI, Speech Acts, and the Edge of Assertion (https://arxiv.org/abs/2410.16645)
- **What's New**: 이번 논문에서는 대형 언어 모델(dlarge language model) 기반 챗봇이 주장(assertion)을 할 수 있는지에 대한 질문을 다룹니다. 저자들은 챗봇 주장 이론(Thesis of Chatbot Assertion, TCA)을 제안하며, 이는 현재 세대 챗봇이 생성하는 출력물 중 일부가 주장으로 인정될 수 있다고 주장합니다.

- **Technical Details**: 논문은 TCA에 대한 최근의 반대 의견들을 검토하며, 이러한 반대 의견들이 의미가 있다고 주장합니다. 저자들은 주장에 대한 온톨로지적 발전(ontogenesis)을 반영해 '프로토 주장(proto-assertion)'이라는 새로운 범주를 만들어야 한다고 제안합니다. 이 범주는 챗봇을 프로토 주장자로 취급하는 데 적용됩니다.

- **Performance Highlights**: 챗봇을 프로토 주장자로 보는 접근 방식은 챗봇 주장의 딜레마를 해결하는데 만족스러운 해법을 제시합니다.



### Graph-Structured Trajectory Extraction from Travelogues (https://arxiv.org/abs/2410.16633)
- **What's New**: 본 논문에서는 인간의 이동 경로 추출을 위한 그래프 기반 표현 방식을 제안하고, 이에 대한 벤치마크 데이터 세트를 구성하였습니다. 인간의 이동 경로를 기하학적 계층 구조와 방문된 위치의 시간적 순서를 모두 보존하는 그래프 표현이 필요하다고 강조합니다.

- **Technical Details**: 연구에서는 Visit Status Prediction (VSP)과 Visiting Order Prediction (VOP)이라는 두 가지 서브 태스크를 통해 자동 경로 추출을 수행합니다. VSP는 언급된 위치에 대해 방문 상태 레이블을 할당하고, VOP는 '방문된' 지점 사이의 포함 및 전이 관계를 식별합니다. 새로운 데이터 세트인 Arukikata Travelogue Dataset with Visit Status and Visiting Order Annotation (ATD-VSO)가 구축되었으며, 총 100개의 여행기 원문, 3,354개의 geo-entity(노드), 그리고 3,369개의 관계(엣지)가 포함됩니다.

- **Performance Highlights**: 실험 결과, 시스템은 상대적으로 높은 정확도로 방문 상태 레이블 및 전이 관계를 예측할 수 있었지만, 포함 관계 예측은 실패했습니다. 이는 시스템에 지리적 계층 구조의 지식을 주입하는 방법이 앞으로의 중요한 과제임을 시사합니다.



### EVC-MF: End-to-end Video Captioning Network with Multi-scale Features (https://arxiv.org/abs/2410.16624)
- **What's New**: 본 논문에서는 비디오 캡셔닝을 위한 새로운 엔드투엔드 기반의 인코더-디코더 네트워크인 EVC-MF를 제안합니다. 이 새로운 접근법은 다중 스케일의 시각적(visual) 및 텍스트적(textual) 특징을 효율적으로 활용하여 비디오 설명을 생성합니다.

- **Technical Details**: EVC-MF는 세 가지 모듈로 구성됩니다: 1) 비디오 프레임을 직접 입력받아 다중 스케일 시각적 특징을 얻는 transformer 기반 네트워크; 2) 다중 스케일 특징을 중복성을 줄이며 유용한 특징을 학습하기 위해 fusion하는 masked encoder; 3) 개선된 transformer 기반 디코더로, 이를 통해 얕은(shallow) 텍스트 정보를 효율적으로 활용합니다.

- **Performance Highlights**: EVC-MF는 벤치마크 데이터셋에 대한 광범위한 실험을 통해 기존의 최첨단 방법들과 비교했을 때 경쟁력 있는 성능을 보였습니다.



### GALA: Graph Diffusion-based Alignment with Jigsaw for Source-free Domain Adaptation (https://arxiv.org/abs/2410.16606)
Comments:
          IEEE TPAMI

- **What's New**: 이번 연구에서는 source-free graph domain adaptation을 위한 새로운 방법인 GALA(Graph Diffusion-based Alignment with Jigsaw)를 제안합니다. 기존 연구에서는 주로 Euclidean 데이터에 집중했던 반면, 본 연구는 비-Euclidean 그래프 데이터에 초점을 맞췄습니다.

- **Technical Details**: GALA는 그래프 확산 모델(graph diffusion model)을 활용하여 타겟 데이터를 기반으로 source 스타일 그래프를 재구성합니다. 이를 위해 source 그래프를 이용해 스코어 기반 그래프 확산 모델(score-based graph diffusion model)을 학습하고, Stochastic Differential Equation(SDE)를 통해 타겟 그래프에 변화를 주어 source 스타일 그래프를 생성합니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋에서 GALA의 우수성을 입증하는 실험을 진행하였으며, 기존 방법들과 비교하여 뛰어난 성능을 발휘했습니다. 특히, 클래스별 적응 threshold를 도입하여 정확하고 편향이 없는 pseudo-labels를 생성하였고, graph jigsaw 방법을 통해 일반화 능력과 견고성을 향상시켰습니다.



### Convex Markov Games: A Framework for Fairness, Imitation, and Creativity in Multi-Agent Learning (https://arxiv.org/abs/2410.16600)
- **What's New**: 이 논문에서는 수직적 의사결정에 대한 새로운 접근을 제시하며, 특히 전문가 모방(expert imitation), 행동 다양성(behavioral diversity), 공정성(preferences) 등의 요소가 포함된 볼록 마르코프 게임(convex Markov games)을 소개합니다. 이는 기존의 보상 구조를 초월하여 최적의 전략 균형을 탐색하는 데 효과적입니다.

- **Technical Details**: 볼록 마르코프 게임은 시퀀스 의사결정 도메인에서 행동의 상태-행동 프로필 분포를 기반으로 합니다. 이 게임에서는 확률적으로 상태와 행동이 조합된 형태인 occupancy measure(점유 측정)를 사용하여 보상을 형성합니다. 연구진은 이러한 프레임워크를 통해 순수 전략 내쉬 균형을 찾을 수 있음을 보였으며, 그래디언트 하강(gradient descent) 방법을 통해 강력한 성과를 낼 수 있음을 밝혔다.

- **Performance Highlights**: 실험을 통해 연구진은 궁극적인 게임(ultimatum games)에서 인간의 선택을 모방하고, 반복되는 죄수의 딜레마에서 새로운 해결책을 제시하며, 비대칭 조정 게임(repeated asymmetric coordination game)에서도 공정한 솔루션을 찾아냈습니다. 특히 죄수의 딜레마에서는 관찰된 인간의 플레이와 약간만 편차를 두면서도, 세 배 더 높은 개별 유틸리티를 달성할 수 있음을 입증했습니다.



### Graph Sampling for Scalable and Expressive Graph Neural Networks on Homophilic Graphs (https://arxiv.org/abs/2410.16593)
- **What's New**: 본 연구는 그래프 신경망(GNN)의 학습을 소규모 그래프에서 대규모 그래프로 전이하는 과정에서 연결성 손실과 낮은 표현력을 해결하기 위한 새로운 그래프 샘플링 알고리즘을 제안합니다. 기존의 무작위 샘플링 방식 대신 'feature homophily'를 활용하여 그래프 구조를 보존하는 접근 방식을 사용합니다.

- **Technical Details**: 제안된 알고리즘은 데이터 상관 행렬의 trace를 최소화하여 그래프 Laplacian의 계급을 더 잘 보존합니다. 이는 O(|V||E|)의 계산 복잡도로, 대규모 그래프에 대해 효율적입니다. 알고리즘은 특정 지역에 국한되지 않고 그래프 전반에 대한 샘플링을 수행할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 알고리즘은 무작위 샘플링에 비해 그래프 계급을 더 잘 보존하며, 반 감독 학습(Semi-supervised learning) 작업에서 GNN의 전이 가능성 향상을 보여주었습니다.



### Dynamic Adaptive Rank Space Exploration for Efficient Sentiment Analysis with Large Language Models (https://arxiv.org/abs/2410.16589)
- **What's New**: 이 논문에서는 Large Language Models(LLMs)을 활용하여 감정 분석을 위한 효율적이고 효과적인 새로운 Dynamic Adaptive Rank Space Exploration (DARSE) 프레임워크를 제안합니다. 이 프레임워크는 다양한 LLM 레이어에 대해 최적의 랭크 조합을 결정하고, 계산 효율성과 모델 성능 간의 균형을 이루기 위해 동적으로 랭크를 할당합니다.

- **Technical Details**: DARSE 프레임워크는 세 가지 주요 구성 요소로 구성됩니다: (1) 최적 랭크의 일반 범위를 식별하기 위한 coarse-grained greedy algorithm, (2) 식별된 범위 내에서 랭크 선택을 세밀하게 조정하는 fine-grained exploration algorithm, (3) 각 LLM 레이어에 대한 최적 랭크 조합을 결정하기 위한 dynamic rank allocation method입니다.

- **Performance Highlights**: DARSE를 적용한 실험 결과, MSE(Mean Squared Error)는 15.1% 개선되었고, 정확도(Accuracy)는 4.3% 향상되었습니다. 이러한 성과는 기존 접근 방식과 비교하여 DARSE의 효과성을 입증합니다.



### Conflict-Aware Adversarial Training (https://arxiv.org/abs/2410.16579)
- **What's New**: 본 논문에서는 기존의 가중 평균 방법이 표준 성능과 적대적 견고성의 최적 균형을 제공하지 못한다는 주장을 제기하며, 이를 해결하기 위해 새로운 적대적 훈련 패러다임인 Conflict-Aware Adversarial Training (CA-AT)을 제안합니다.

- **Technical Details**: CA-AT는 표준 손실과 적대적 손실 간의 충돌을 인식하고 이를 완화하기 위해 각기 다른 각도를 기반으로 하는 새로운 트레이드오프 인자를 사용하여 모델 파라미터를 최적화합니다. 이 방법은 고정된 트레이드오프 팩터를 사용하는 가중 평균 방식을 넘어, 더 나은 성능을 제공합니다.

- **Performance Highlights**: 종합적인 실험 결과에 따르면, CA-AT는 적대적 훈련의 다양한 설정에서 표준 성능과 적대적 견고성 간의 균형을 지속적으로 향상시키며, 다양한 적대적 손실 함수 및 공격 유형에서도 다루기 용이합니다.



### Implicit Contact Diffuser: Sequential Contact Reasoning with Latent Point Cloud Diffusion (https://arxiv.org/abs/2410.16571)
Comments:
          In submussion

- **What's New**: Implicit Contact Diffuser (ICD)라는 새로운 확산 기반 모델을 소개합니다. 이 모델은 객체와 환경 간의 일련의 접촉 관계를 명시하는 신경 서술어의 시퀀스를 생성하여, MP(V)C 방법을 사용하는데 도움을 줍니다. 이 접근 방식은 로컬 미니마(local minima)를 피하고 더 효과적인 조작(manipulation) 수행을 가능하게 합니다.

- **Technical Details**: ICD는 Neural Descriptor Fields (NDF)의 수정된 버전을 기반으로 하는(scene-level) 장면 수준의 신경 서술자를 통해 객체와 환경 간의 접촉 관계를 인코딩합니다. 이를 통해 복잡한 갭(gap)을 식별하고 다양한 환경에서의 목표 접촉 관계를 일반화할 수 있습니다. 이 모델은 접촉 시퀀스를 예측하기 위해 잠재적 확산 모델을 훈련시키며, 필요한 시퀀스 길이를 결정하기 위해 도달 가능성(reachability) 함수를 학습합니다.  

- **Performance Highlights**: 복잡하고 긴 기간의 접촉이 밀집된 조작 작업인 케이블 라우팅 및 노트북 접기에서 ICD는 기존의 기준선 모델을 초과하거나 동등한 성능을 보였습니다. ICD는 목표 접촉 관계를 다른 환경에 자연스럽게 적응할 수 있는 능력도 지니고 있습니다.



### Raising the Stakes: Performance Pressure Improves AI-Assisted Decision Making (https://arxiv.org/abs/2410.16560)
- **What's New**: 본 연구는 사람들에게 AI 조언(advice)에 대한 성과 압력(performance pressure)이 어떻게 영향을 미치는지를 분석하였으며, 특히 낮은 성과 압력이 있는 환경에서도 그러한 압력을 유도할 수 있는 방법을 모색하였습니다.

- **Technical Details**: 연구에서는 Amazon Mechanical Turk에서 일하는 일반인들이 가짜 리뷰 감지(fake review detection)와 같은 AI 지원 작업을 수행할 때 성과 압력이 AI 조언 의존도에 미치는 영향을 조사합니다. 성과 압력은 금전적 보상에 대한 손실 회피(loss aversion)를 활용하여 조작되었습니다. 연구 결과, 이해관계가 클수록 (high stakes) AI 조언을 더 적절하게 활용하는 경향이 있음을 발견하였습니다.

- **Performance Highlights**: 성과 압력이 높을수록, 잘못된 AI 조언을 받을 경우 사람들은 해당 조언을 적절히 무시하는 경향이 높아지며, 이는 AI 설명의 유무와는 관계가 없습니다. 또한, XAI(explainable AI)가 성과 압력 하에서 어떻게 작용하는지에 대한 복잡성을 발견하였습니다.



### PromptHive: Bringing Subject Matter Experts Back to the Forefront with Collaborative Prompt Engineering for Educational Content Creation (https://arxiv.org/abs/2410.16547)
- **What's New**: PromptHive는 LLM(대형 언어 모델)과 도메인 전문 지식을 효과적으로 연결하기 위해 설계된 협업 프롬프트 작성 인터페이스로, 기존의 프롬프트 작성 방식에 비해 단시간에 적절한 프롬프트를 생성하게 도와줍니다.

- **Technical Details**: PromptHive는 사용자가 프롬프트를 생성, 수정 및 공유하는 과정에서 풍부한 사용자 상호작용 데이터를 수집하는 데이터 저장 소프트웨어를 포함하여, 다양한 프롬프트 변형을 시스템적으로 실험할 수 있는 기능을 제공합니다. 이 시스템은 두 가지 수준의 추상화(교과서 수준 및 수업 수준)에서 유용한 프롬프트를 저장하고 업보팅할 수 있는 공유 라이브러리를 갖추고 있습니다.

- **Performance Highlights**: PromptHive를 사용한 연구에서는 수학 분야의 전문 지식을 가진 10명의 사용자들이 생성한 힌트는 인식된 인지 부담을 절반으로 줄이고, 프롬프트 작성 시간을 몇 개 월에서 몇 시간으로 단축할 수 있다는 결과를 얻었습니다. 사용자들은 PromptHive의 사용성을 100점 만점에 89점으로 평가하며 LLM 모델의 성능에 대한 신뢰를 느꼈습니다.



### A Theoretical Understanding of Chain-of-Thought: Coherent Reasoning and Error-Aware Demonstration (https://arxiv.org/abs/2410.16540)
- **What's New**: 이 논문은 Few-shot Chain-of-Thought (CoT) prompting을 개선하기 위한 새로운 이론적 접근 방식인 Coherent CoT를 제안합니다. Coherent CoT는 이전 단계의 추론을 통합하여, 예측 성능을 향상시키고자 하였습니다.

- **Technical Details**: Coherent CoT는 Stepwise ICL에 비해 Transformer 모델의 예측 성능을 향상시키는데 도움을 줍니다. 중간의 추론 단계에서 발생하는 오류에 더 민감하다는 관찰을 통해, 정확하고 부정확한 추론 경로를 통합할 것을 제안합니다.

- **Performance Highlights**: 제안된 방식을 통해 Coherent CoT의 중간 추론 단계의 정확도를 높임으로써, 전체 CoT 성능을 향상시키는 실험 결과가 제공되었습니다.



### Bayesian scaling laws for in-context learning (https://arxiv.org/abs/2410.16531)
Comments:
          10 pages main text, 26 pages total

- **What's New**: 이번 연구에서는 In-Context Learning (ICL)이 베이시안(Bayesian) 학습자와 유사하게 작용하는 방식을 보여줍니다. 이를 통해 ICL의 새로운 베이시안 스케일링 법칙을 제안하였고, 다양한 사이즈의 GPT-2 모델을 통해 실험적으로 검증하였습니다.

- **Technical Details**: ICL은 기존 모델의 학습 업데이트 없이 복잡한 작업을 수행하는 강력한 기술입니다. 우리의 연구는 ICL이 베이시안 학습을 근사한다는 점을 바탕으로, 모델의 예측 정확도와 ICL 예제 수 간의 상관관계를 설명합니다. 또한, 모델의 정밀도, 작업 사전(task priors), 학습 효율성 및 개별 예제 확률에 대한 해석 가능한 용어를 제공하였습니다.

- **Performance Highlights**: 모든 실험에서 우리 연구의 베이시안 스케일링 법칙은 ICL이 억제된 행동을 다시 드러낼 조건을 정확히 예측했습니다. 이는 LLM의 안전성을 향상시키기 위한 사후 훈련(post-training)의 비효율성에 대한 통찰을 제공합니다.



### AUTALIC: A Dataset for Anti-AUTistic Ableist Language In Contex (https://arxiv.org/abs/2410.16520)
Comments:
          9 pages, 5 figures, 7 tables

- **What's New**: AUTALIC은 맥락에서 반자폐증적 ableist 언어를 감지하기 위해 설계된 첫 번째 벤치마크 데이터셋으로, 총 2,400개의 자폐증 관련 문장과 주위 맥락이 포함되어 있으며, 전문가에 의해 주석이 달려 있습니다.

- **Technical Details**: AUTALIC 데이터셋은 Reddit에서 수집된 자폐증 관련 문장으로 구성되어 있으며, 이 문장들은 neurodiversity 배경을 가진 전문가에 의해 주석 처리되었습니다. 기존의 NLP 도구들이 반자폐증적 ableist 언어의 미세한 표현을 정확히 감지하지 못하는 한계를 보여줍니다.

- **Performance Highlights**: 현재의 언어 모델, 특히 LLMs는 인간의 판단과 일치하지 않으면서 반자폐증적 ableism을 식별하는 데 어려움을 겪고 있어 이 분야에서의 한계를 강조합니다.



### RGMDT: Return-Gap-Minimizing Decision Tree Extraction in Non-Euclidean Metric Spac (https://arxiv.org/abs/2410.16517)
- **What's New**: 본 논문에서는 다중 에이전트 환경에서 Deep Reinforcement Learning (DRL) 정책의 해석 가능성을 높이기 위한 Return-Gap-Minimization Decision Tree (RGMDT) 알고리즘을 제안합니다. 이 알고리즘은 결정 트리(Decision Tree) 정책 사이의 수익 차이를 정량적으로 보장하며, 이 과정에서 만들어지는 결정 경로는 명확한 해석을 제공합니다.

- **Technical Details**: RGMDT 알고리즘은 주어진 DRL 정책과 추출된 결정 트리 간의 기대 수익 차이에 대한 폐쇄형 보장을 제공합니다. 이 알고리즘은 각 에이전트의 관측 및 행동 가치 공간에서 비상식적인 클러스터링 문제로 전환되며, 결정 트리 생성 과정은 주어진 크기 제약 조건을 최소화하도록 설계되었습니다. 또한, 다중 에이전트 환경에서 비 중앙 집중식 결정 트리를 생성하기 위해 iteratively-grow-DT 과정을 확장하여, 다른 에이전트의 현재 결정 트리와 조건화된 행동 가치 함수를 수정합니다.

- **Performance Highlights**: RGMDT는 D4RL와 같은 복잡한 작업에서 기존의 휴리스틱 결정 트리 기반 기준 모델들보다 유의미하게 우수한 성능을 나타내며, 주어진 결정 트리의 복잡성 제약 조건 하에서 거의 최적의 수익을 달성할 수 있습니다.



### STAR: A Simple Training-free Approach for Recommendations using Large Language Models (https://arxiv.org/abs/2410.16458)
- **What's New**: 최근 대규모 언어 모델(LLMs)의 발전은 추천 시스템(RecSys) 작업을 위한 유망한 새로운 접근 방법을 제시합니다. 본 논문에서는 LLM을 활용하여 세부 조정(fine-tuning) 없이 다양한 추천 작업에 적용할 수 있는 Simple Training-free Approach for Recommendation (STAR) 프레임워크를 제안합니다.

- **Technical Details**: STAR 프레임워크는 두 가지 주요 단계로 구성됩니다: Retrieval 단계(후보 아이템 검색) 및 Ranking 단계(아이템 순위 조정). Retrieval 단계에서는 LLM의 의미적 임베딩(semantic embedding)을 사용하여 후보 아이템을 검색하고, Ranking 단계에서는 LLM의 추론 능력을 활용하여 후보 아이템의 순위를 조정합니다. 이는 시맨틱(similarity)과 협력적 정보(collaborative information)를 결합하여 이루어집니다.

- **Performance Highlights**: Amazon Review 데이터셋에서 실험한 결과, STAR 방법이 Beauty 부문에서 +23.8%, Toys and Games 부문에서 +37.5%, Sports and Outdoors 부문에서 -1.8%의 Hits@10 성능을 기록하며, 기존의 최적 감독(supervised) 모델들과 경쟁력 있는 성과를 보였습니다.



### Does your LLM truly unlearn? An embarrassingly simple approach to recover unlearned knowledg (https://arxiv.org/abs/2410.16454)
Comments:
          21 pages, 2 figures

- **What's New**: 이번 연구는 언러닝(unlearning) 방법을 적용한 대규모 언어 모델(LLM)에서 양자화(quantization)가 잊힌 정보를 회복할 수 있음을 밝혀냈습니다. 이는 기존 언러닝 방법의 실패를 의미하며, LLM의 안전한 사용을 보장하기 위해 양자화를 통한 지식 회복을 방지하는 새로운 목표를 설정합니다.

- **Technical Details**: 연구에서는 기계 언러닝(machine unlearning) 기법을 통해 특정 지식을 제거하는 방법을 살펴보았습니다. 특히, 기존의 언러닝 방법들이 실제로 지식을 잊는지 아니면 숨기기만 하는지를 평가하기 위한 실험을 실시하였으며, 다양한 양자화 기법을 통해 그 결과를 검증했습니다. 기존의 방법에 따르면, 양자화를 거친 언러닝된 모델이 83%의 잊힌 지식을 회복할 수 있음을 보여주었습니다.

- **Performance Highlights**: 본 연구의 실험 결과, 언러닝된 모델이 전체 정밀도(full precision)에서 평균적으로 21%의 잊힌 지식을 유지하며, 4비트 양자화(quantization) 후에는 이 비율이 83%로 증가하는 것을 확인했습니다. 연구팀은 이러한 결과를 바탕으로 saliency mapping 기법을 사용한 새로운 언러닝 전략(Saliency-Based Unlearning with a Large Learning Rate, SURE)을 제안하였습니다.



### Pantograph: A Machine-to-Machine Interaction Interface for Advanced Theorem Proving, High Level Reasoning, and Data Extraction in Lean 4 (https://arxiv.org/abs/2410.16429)
- **What's New**: 이번 논문에서는 Pantograph라는 새로운 도구를 소개합니다. 이 도구는 Lean 4 증명 보조기와의 통합 인터페이스를 제공하며, Monte Carlo Tree Search와 같은 강력한 검색 알고리즘을 통해 효과적인 증명 검색을 지원합니다.

- **Technical Details**: Pantograph는 API 및 Read-Eval-Print Loop (REPL)로 구성되어 있으며, 적절한 인터페이스를 통해 기계 학습 모델이 증명 에이전트를 훈련하고 평가할 수 있도록 돕습니다. 이 도구는 메타변수 결합(metavariable coupling)을 효과적으로 처리하여 MCTS와 같은 더 강력한 검색 알고리즘을 지원합니다. 또한, 고급 추론 단계인 tactic을 완전히 지원합니다.

- **Performance Highlights**: Pantograph는 MiniF2F 벤치마크에서 DSP 접근 방식을 평가하여 복잡한 증명 검색과 고급 추론을 수행할 수 있는 가능성을 보여줍니다. Pantograph의 혁신적인 기능들은 기계 학습 모델이 복잡한 증명을 다루는 데 필요한 기반을 제공하며, 이는 향후 연구자들이 더 다재다능하고 강력한 정리증명기(theorem prover)를 설계하는 데 기여할 것입니다.



### Position: Challenges and Opportunities for Differential Privacy in the U.S. Federal Governmen (https://arxiv.org/abs/2410.16423)
Comments:
          2nd Workshop on Regulatable ML at NeurIPS 2024

- **What's New**: 이 논문에서는 미국 연방 정부 내의 차별적 프라이버시(differential privacy)의 도전 과제와 기회를 조사합니다. 연구팀은 차별적 프라이버시 기술의 사용을 제한하는 세 가지 주요 도전 과제를 강조하며, 정부 기관의 능력을 향상시킬 수 있는 두 가지 사례를 제시합니다.

- **Technical Details**: 차별적 프라이버시는 개인의 데이터를 사용하지 않고도 알고리즘의 출력이 개인의 데이터에 의해 크게 영향을 받지 않도록 보장하는 알고리즘적 프레임워크입니다. 이 기술은 데이터의 개인정보 보호를 위한 특정한 통계적 보장을 제공하며, 이는 통계, 기계 학습 및 데이터 분석의 다양한 사용 사례에 적용될 수 있습니다. 본 논문에서는 차별적 프라이버시의 양적 매개변수를 통해 데이터 보안 담당자가 알고리즘의 사용자와 데이터셋의 민감도를 개선할 수 있는 가능성에 대해 논의합니다.

- **Performance Highlights**: 연방 정부 내에서 차별적 프라이버시가 데이터 유출의 잠재적 피해를 줄이고 더 나은 공적 통계 및 데이터 공개를 가능하게 할 잠재력이 있음이 강조됩니다. 특히 2020년 인구 조사에서 차별적 프라이버시를 적용하여 민감한 통계를 가린 사례를 통해 공공 정책 결정을 개선할 수 있는 방법을 보여주고 있습니다.



### On conditional diffusion models for PDE simulations (https://arxiv.org/abs/2410.16415)
Comments:
          Accepted at NeurIPS 2024

- **What's New**: 이번 연구에서는 스코어 기반(diffusion-based) 모델을 활용하여 희소 관측치에 대한 예측(forecasting) 및 데이터 동화(data assimilation)를 효과적으로 결합하는 방법을 제안합니다. 이 모델은 최신 diffusion 모델이 갖는 유연성을 통해 관측치를 손쉽게 통합할 수 있어, 기존의 두 단계 처리 방식을 간소화할 수 있습니다.

- **Technical Details**: 본 논문에서는 1) 예측 성능을 크게 향상시키는 자기회귀 샘플링 기법(autoregressive sampling approach), 2) 다양한 이력 길이(history lengths)를 본 단련(direct training) 중 안정적으로 유지하는 새로운 훈련 전략(training strategy), 3) 초기 조건에 대한 유연한 사전-훈련(pre-training)과 데이터 동화를 위한 유연한 사후-훈련(post-training) 조건화를 활용한 하이브리드 모델(hybrid model)을 제안합니다.

- **Performance Highlights**: 본 연구에서 제안하는 방법들은 데이터 동화 및 예측 문제를 결합적으로 해결하는데 있어 필수적임을 실험적으로 입증하였습니다. 자가 회귀 방식이 기존의 샘플링 방식보다 예측 성능에서 두드러진 향상을 보이며, 혼합 작업을 위한 하이브리드 모델은 예측과 데이터 동화 모두에서 안정적인 성능을 발휘하는 것으로 나타났습니다.



### Enhancing Multimodal Affective Analysis with Learned Live Comment Features (https://arxiv.org/abs/2410.16407)
- **What's New**: 본 논문에서는 LCAffect 데이터셋을 구축하였고, 이를 통해 감정 분석을 위한 합성 라이브 댓글(feature) 생성 방식을 소개합니다. 이 방식은 다양한 비디오 장르에서의 감정 인식을 개선하는 데 기여합니다.

- **Technical Details**: LCAffect 데이터셋은 영어와 중국어 비디오에서 생성된 라이브 댓글을 포함하고 있으며, 감정 분석 작업을 위한 멀티모달(모든 매체) 인코더를 통해 합성 라이브 댓글 기능을 생성합니다. 기존 방법들과의 비교를 통해 SOTA(최신 기술) 성능을 달성했습니다.

- **Performance Highlights**: 감정 감지, 감정 인식 및 냉소 감지와 같은 작업에서 SOTA 성능을 달성했으며, 특히 CH-SIMS v2에서는 3.18포인트, MELD에서는 2.89포인트, MuSTARD에서는 3포인트 성능 향상을 보였습니다.



### Hotel Booking Cancellation Prediction Using Applied Bayesian Models (https://arxiv.org/abs/2410.16406)
- **What's New**: 이번 연구는 호텔 예약 취소를 예측하기 위해 Bayesian 모델을 적용하였습니다. 이는 자원 배분, 수익 및 고객 만족도에 중요한 도전 과제가 됩니다.

- **Technical Details**: 본 연구는 36,285개의 관측치와 17개의 피처를 포함하는 Kaggle 데이터셋을 활용했습니다. Bayesian Logistic Regression 및 Beta-Binomial 모델을 구현하였으며, 특히 12개의 피처와 5,000개의 무작위 선택 관측치를 사용한 로지스틱 모델이 예측 정확도에서 Beta-Binomial 모델을 초월했습니다.

- **Performance Highlights**: 모델 평가에 사용된 Leave-One-Out Cross-Validation (LOO-CV)을 통해 관측된 결과와 예측된 결과 간의 강한 일치를 확인하였습니다. 특히, 특별 요청 및 주차 공간의 가용성이 취소의 가장 강력한 예측 요인으로 나타났습니다. 이 Bayesian 접근 방식은 호텔 산업에서 예약 관리 및 운영 효율성을 향상시키는 데 유용한 도구입니다.



### A Simple Model of Inference Scaling Laws (https://arxiv.org/abs/2410.16377)
Comments:
          11 pages, 7 figures

- **What's New**: 이 논문에서는 Neural scaling laws를 기반으로 한 간단한 통계적 접근 방식인 memorization을 제안하여, 반복 추론(inference) 시 모델 성능이 어떻게 향상되는지를 연구합니다.

- **Technical Details**: 논문에서는 pass@k 메트릭을 이용하여 반복적 추론 시 성공 확률을 측정합니다. 이를 통해, 증가하는 추론 시도 횟수에 대해 power law 방식으로 감소하는 'inference loss' 정의합니다. 단순 생성 모델인 Variational Autoencoder(VAE)를 사용해 실험을 진행하고, 이론적 예측과 실제 커버리지 곡선 간의 일치를 확인합니다.

- **Performance Highlights**: 이 연구의 결과는 다른 neural scaling laws의 효과와 분리되어 있으며, 추론 시도와 훈련 간의 최적 균형을 찾아 성능과 비용을 최소화하는 기반을 제공합니다.



### Domain-Adaptive Neural Posterior Estimation for Strong Gravitational Lens Analysis (https://arxiv.org/abs/2410.16347)
Comments:
          20 pages, 2 figures, 2 tables

- **What's New**: 본 연구는 Neural Posterior Estimation (NPE)와 Unsupervised Domain Adaptation (UDA)를 결합하여 강중력렌즈 데이터의 분석 효율성을 높이는 첫 번째 연구를 수행하였습니다.

- **Technical Details**: NPE는 CNN 기반의 임베딩 네트워크를 활용하여 이미지를 특징으로 요약하고, Masked Autoregressive Flow (MAF)를 통해 비가우시안 분포를 포함한 다양한 형태의 후방 분포를 추정합니다. UDA에서는, 출처 도메인 데이터에 레이블이 있고 목표 도메인 데이터에는 레이블이 없는 방식으로, Maximum Mean Discrepancy (MMD)를 손실 함수로 사용하여 잠재 특징의 공간을 정렬합니다.

- **Performance Highlights**: NPE와 UDA의 조합이 분석의 정확성을 1-2 배수 증가시키고, UDA가 없는 NPE 모델에 비해 후방 범위를 유의미하게 향상시켰습니다.



### Quantum Convolutional Neural Network: A Hybrid Quantum-Classical Approach for Iris Dataset Classification (https://arxiv.org/abs/2410.16344)
Comments:
          13 pages, 2 figures, 1 table, Quantum Machine Learning

- **What's New**: 이 논문은 4-큐빗(4-qubit) 양자 회로와 고전 신경망(Classical Neural Network)을 통합한 혼합 양자-고전 머신 러닝 모델인 Quantum Convolutional Neural Network(QCNN)를 제안합니다. 이 모델은 Iris 데이터셋의 특성을 각도 임베딩(Angle Embedding)과 얽힘 게이트(Entangling Gates)를 사용하여 인코딩하여, 고전 모델로는 다루기 어려운 복잡한 특성 관계를 포착합니다.

- **Technical Details**: QCNN 모델은 20 에폭(epoch) 동안 훈련되었으며, 16번째 에폭에서 Iris 데이터셋 테스트 세트에서 100% 정확도를 달성했습니다. 양자 회로는 매개변수화된 게이트(Parameterized Gates) 층으로 구성되어 있으며, 이는 입력 데이터에 대한 다양한 변환을 가능하게 합니다. 이와 함께 고전 신경망 계층이 결합되어 양자 특성 인코딩과 고전 데이터 처리 효율성 간의 시너지를 극대화합니다.

- **Performance Highlights**: 본 연구 결과, QCNN 모델은 Iris 데이터셋의 테스트 세트에서 단 16 에폭만에 100% 정확도를 달성하여, 양자 향상 모델이 감독 학습(Supervised Learning) 작업에서 데이터 인코딩 및 처리에 있어 효율성을 높일 수 있는 가능성을 보여주었습니다.



### Feint and Attack: Attention-Based Strategies for Jailbreaking and Protecting LLMs (https://arxiv.org/abs/2410.16327)
- **What's New**: 이번 논문에서는 Large Language Models(LLMs)에 대한 jailbreak 공격을 이해하기 위한 새로운 방법론과 이에 대한 방어 전략을 제시합니다.

- **Technical Details**: 논문은 새로운 메트릭인 Attention Intensity on Sensitive Words (Attn_SensWords), Attention-based Contextual Dependency Score (Attn_DepScore), Attention Dispersion Entropy (Attn_Entropy)를 도입합니다. 이러한 메트릭을 기반으로 Attention-Based Attack (ABA)와 Attention-Based Defense (ABD) 전략이 마련됩니다.

- **Performance Highlights**: 실험 결과, ABA는 LLM의 주의 분포를 효과적으로 전환시켜 더 위험한 컨텐츠를 유도할 수 있으며, ABD는 입력의 위험 수준을 평가하고 LLM의 방어력을 크게 향상시킬 수 있음을 보여줍니다.



### CybORG++: An Enhanced Gym for the Development of Autonomous Cyber Agents (https://arxiv.org/abs/2410.16324)
Comments:
          8 pages, 3 figures and included appendix

- **What's New**: CybORG++는 네트워크 방어를 위한 강화 학습( reinforcement learning, RL) 연구를 위해 개발된 고급 툴킷입니다. CybORG의 이전 버전인 CAGE 2에서 발전하여 향상된 디버깅 기능과 에이전트 구현 지원을 제공합니다. 또한 MiniCAGE라는 경량 버전을 도입하여 최대 1000배 빠른 실행 속도를 자랑합니다.

- **Technical Details**: CybORG++는 두 개의 주요 구성 요소로 구성되어 있습니다: (1) 디버그된 CAGE 2 환경, (2) MiniCAGE. MiniCAGE는 원래 CAGE 2와 동일한 기본 RL 구성 요소를 유지하면서도 대규모 실험을 가능하게 하며, 최대 1000배의 속도 개선을 통해 훈련과 평가에 필요한 시간을 크게 단축합니다.

- **Performance Highlights**: MiniCAGE는 원래의 CybORG 환경보다 약 15배에서 최대 800배의 속도 향상을 보여줍니다. 두 환경 간의 보상 측정에서 높은 상관관계를 나타내어 MiniCAGE가 CAGE 2 환경을 효과적으로 모사함을 증명합니다.



### SouLLMate: An Application Enhancing Diverse Mental Health Support with Adaptive LLMs, Prompt Engineering, and RAG Techniques (https://arxiv.org/abs/2410.16322)
Comments:
          26 pages, 19 figures, 8 tables

- **What's New**: 이번 연구에서는 혁신적인 AI 기술을 통해 개인화된, 시대를 초월한 정신 건강 지원을 제공하는 SouLLMate 시스템을 소개하고 있습니다.

- **Technical Details**: SouLLMate 시스템은 LLM(대형 언어 모델), LangChain, RAG(정보 검색 기반 생성), 프롬프트 엔지니어링 등의 기술을 기반으로 하며, 핵심 기능으로는 초기 평가(PMH-A), 대화형 정보 추출(CIE), 위험 감지(SRD), 사전 안내 대화(PGD) 등이 포함됩니다.

- **Performance Highlights**: 이 시스템은 정신 건강 전문의의 진단을 지원하고, 긴 문맥 추론의 정확성을 높이기 위해 KIS(핵심 지표 요약), PQS(사전 질문 전략), SMMR(스택드 다중 모델 추론) 방법을 제안하며, 이는 정신 건강 지원 기술의 접근성 및 효과성을 개선할 것으로 기대됩니다.



### Accelerating Object Detection with YOLOv4 for Real-Time Applications (https://arxiv.org/abs/2410.16320)
Comments:
          18 pages, 10 figures

- **What's New**: 이 논문은 UAV(무인항공기)에서의 실시간 물체 탐지 문제를 다룹니다. 특히, CNN(합성곱 신경망)을 활용한 YOLOv4(You Only Look Once - Version 4) 알고리즘을 개선하여, 복잡한 환경 속에서도 정확한 물체 탐지를 가능하게 하는 새로운 접근법을 제안합니다.

- **Technical Details**: 이 연구에서 제안된 YOLOv4는 DropBlock Regularization, Data Augmentation, Mish-Activation, CrossStage Partial connections(CSP), Self adversarial training(SAT), Weighted-Residual-Connections(WRC) 등의 여러 기술들을 결합하여 구성됩니다. YOLOv4의 성능은 COCO 데이터셋에서 43.5%의 평균 정밀도(AP)와 65 FPS의 속도를 기록하였습니다.

- **Performance Highlights**: YOLOv4는 실시간 물체 탐지에서 빠르고 정확한 성능을 발휘합니다. YOLOv4는 1단계 탐지기로, R-CNN, Fast R-CNN과 같은 2단계 탐지기에 비해 더 빠른 처리 속도와 높은 정확성을 제공합니다.



### A Survey on Physical Adversarial Attacks against Face Recognition Systems (https://arxiv.org/abs/2410.16317)
- **What's New**: 본 논문은 Face Recognition (FR) 시스템을 겨냥한 물리적 적대적 공격(physical adversarial attacks)에 대한 포괄적인 분석을 제공하고, 이 분야의 도전 과제 및 향후 연구 방향을 탐구합니다. 물리적 공격 방법을 세 가지 범주로 분류하고 각 범주의 연구 진전 상황을 요약합니다.

- **Technical Details**: FR 시스템은 딥 러닝(d deep learning) 기술의 발전으로 성능과 확장성에서 큰 진전을 이루었으며, 이러한 시스템에는 특정한 공격 기법이 요구됩니다. 디지털 공격(digital attacks)과 물리적 공격(physical attacks)으로 나눌 수 있는 적대적 공격의 유형을 설명하고, 이들 각각의 특징 및 도전 과제를 논의합니다. 예를 들어, 공격자는 얼굴에 부착된 스티커(stickers)나 액세서리(accessories)를 통해 FR 시스템을 속이는 방법을 사용합니다.

- **Performance Highlights**: 최근 연구에 따르면, 물리적 적대적 공격은 실제 환경에서 매우 효과적인 것으로 입증되었으며, FR 시스템의 보안에 중대한 위협이 됩니다. 이 논문은 2016년부터 2024년까지의 40편의 연구를 포괄적으로 분석하여, 물리적 공격과 관련된 최신의 발전사항을 제시하고 이 분야의 미래 연구 방향을 모색합니다.



### In-the-loop Hyper-Parameter Optimization for LLM-Based Automated Design of Heuristics (https://arxiv.org/abs/2410.16309)
- **What's New**: 이 논문에서는 LLaMEA (Large Language Model Evolutionary Algorithm) 프레임워크와 Hyper-Parameter Optimization (HPO) 절차를 통합한 새로운 하이브리드 접근 방식인 LLaMEA-HPO를 제안합니다. 이를 통해 LLM이 알고리즘 구조 생성에 집중할 수 있게 하여 최적화 프로세스의 전반적인 효율성을 향상시킵니다.

- **Technical Details**: LLaMEA-HPO는 HPO 툴을 통해 하이퍼 파라미터 튜닝을 오프로드함으로써 LLM이 창의적인 알고리즘 구조 및 제어 흐름 생성을 더욱 잘 수행할 수 있게 해줍니다. 이 접근 방식은 LLM 쿼리 수를 줄이고, 진화 프로세스의 성능을 향상시키며, 재정적 및 계산적 비용을 낮추는 데 기여합니다.

- **Performance Highlights**: LLaMEA-HPO는 Online Bin Packing, Black-Box Optimization, Traveling Salesperson Problem 등 여러 벤치마크 문제에서 기존 LLM 기반 프레임워크와 비교할 때 우수하거나 동등한 성능을 달성하면서 계산 비용을 크게 줄였습니다.



### Intelligent Computing Social Modeling and Methodological Innovations in Political Science in the Era of Large Language Models (https://arxiv.org/abs/2410.16301)
Comments:
          34 pages, 5 figures, 3 tables

- **What's New**: 최근 인공지능의 물결은 대규모 언어 모델(LLMs)의 발전으로 특징지어지며, 정치학에 대한 방법론적 혁신을 제안하고 있습니다. 본 논문은 LLMs가 사회과학의 지식 생산 및 패러다임 전환에 미치는 영향을 포괄적으로 이해하기 위한 방법을 제시합니다.

- **Technical Details**: 본 논문은 "Intelligent Computing Social Modeling" (ICSM) 방법론을 제안하며, LLMs의 핵심 메커니즘을 명확히 하고자 합니다. ICSM은 아이디어 합성과 행동 시뮬레이션에서 LLMs의 강점을 활용하여 정치학에서 '시뮬레이티드 소셜 컨스트럭션'과 '시뮬레이션 검증'을 통해 지적 탐색을 진전시킵니다.

- **Performance Highlights**: 미국 대선의 시뮬레이션을 통해 ICSM의 운영 경로와 방법론적 장점을 실증적으로 보여줍니다. ICSM은 기존 사회과학 패러다임과 통합하여 양적 패러다임의 대규모 데이터 활용 능력을 강화하고, 질적 패러다임에 개인 수준의 사회 메커니즘 발견을 위한 증거를 제공합니다. 이로 인해 LLMs가 정치학에서 방법론적 혁신을 주도할 것이라는 발견을 제안합니다.



### Hawk: An Efficient NALM System for Accurate Low-Power Appliance Recognition (https://arxiv.org/abs/2410.16293)
Comments:
          Accepted to the 22nd ACM Conference on Embedded Networked Sensor Systems (SenSys 2024)

- **What's New**: Hawk는 저전력 전자기기를 정확하게 인식하기 위해 설계된 효율적이고 정확한 NALM 시스템입니다. 데이터셋 수집 단계에서 전통적인 방법보다 짧은 시간에 더 많은 조합을 수집할 수 있는 HawkDATA를 소개합니다.

- **Technical Details**: Hawk의 방법론은 두 단계로 나뉘며, 첫 번째 단계는 그룹화된 무작위 균형 Gray 코드(BGCode)를 사용하여 다양한 이벤트 시퀀스를 생성합니다. 이 후 센서 간 데이터 동기화에 '공유된 인식 가능한 시간(shared perceptible time)' 전략을 적용하여 후속 단계를 간소화합니다. 이벤트 인식 단계에서는 안정 상태 차별(SsDiff) 전처리 방법을 사용하여 집계된 전류에서 저전력 기기의 신호 대 간섭 비율(SINR)을 향상시킵니다.

- **Performance Highlights**: HawkDATA는 기존 방법에 비해 수집 시간의 1/71.5만에 6.34배 많은 기기 상태 조합을 수집할 수 있습니다. Hawk의 평균 F1 점수는 상태 인식에서 93.94%, 사건 인식에서 97.07%로, 최신 알고리즘(SOTA)에 비해 각각 47.98% 및 11.57% 향상되었습니다.



### An evaluation of LLM code generation capabilities through graded exercises (https://arxiv.org/abs/2410.16292)
- **What's New**: 본 논문은 현재 사용 가능한 평가 방법들을 검토하고, 최신 모델인 GPT4-o-mini의 성능을 Codewars와 같은 소프트웨어 개발 커뮤니티에서 확보한 8개 프로그래밍 언어의 코딩 도전 과제를 해결하는 데 평가한 결과를 제시합니다.

- **Technical Details**: 대규모 언어 모델(LLMs)은 Transformer 블록으로 구축된 심층 신경망으로, 작성된 텍스트를 단어 조각(token)으로 나누어 모델의 분포를 표현합니다. 이들은 비지도 학습 방식으로 인터넷에서 수집된 방대한 양의 텍스트를 사용하여 훈련됩니다. 평가 방법으로는 다수의 선택형 테스트, 군중 소싱 human 평가, 자동 평가 메트릭 등이 있습니다.

- **Performance Highlights**: 모델의 성공 가능성은 작업의 난이도, 사용되는 프로그래밍 언어의 인기, 과제가 게시된 시점으로부터의 경과 시간과 양의 상관관계를 보입니다. 연구 결과는 LLMs의 실제 능력을 과대 평가할 수 있음을 시사합니다.



### Assessing the Performance of Human-Capable LLMs -- Are LLMs Coming for Your Job? (https://arxiv.org/abs/2410.16285)
- **What's New**: 이 논문은 SelfScore라는 새로운 벤치마크를 개발하고 검증한 것으로, 이는 자동화된 Large Language Model (LLM) 에이전트의 헬프 데스크 및 전문 상담 업무 성능을 평가하기 위해 설계되었습니다. SelfScore는 인공지능(AI)의 활용이 증가하는 산업에서 자동화된 에이전트와 인간 근로자를 비교할 수 있는 중요한 도구입니다.

- **Technical Details**: SelfScore는 문제 복잡성과 응답 유용성을 기반으로 에이전트를 평가하며, 투명성과 단순성을 보장하는 채점 시스템을 제공합니다. 연구에서는 SelfScore를 평가하기 위해 자동화된 LLM 에이전트를 개발하였고, Retrieval-Augmented Generation (RAG) 기법의 장점을 탐구하였습니다. RAG는 관련 정보를 외부 출처에서 검색하여 이를 기반으로 새로운 텍스트를 생성하는 자연어 처리(NLP) 기법입니다.

- **Performance Highlights**: 연구 결과, 자동화된 LLM 에이전트는 인간 대조군보다 우수한 성능을 보였으며, 이는 AI 기술이 뛰어난 분야에서 인간 근로자의 대체 가능성에 대한 우려를 불러일으킵니다. SelfScore는 헬프 데스크 환경에서 AI의 영향을 이해하는 데 기초적인 도구를 제공하며, 자동화로의 전환에 따른 윤리적 고려를 옹호합니다.



### Understanding the Effect of Algorithm Transparency of Model Explanations in Text-to-SQL Semantic Parsing (https://arxiv.org/abs/2410.16283)
Comments:
          15 pages, 18 figure, Preprint

- **What's New**: 이번 연구는 AI 모델의 결정 과정을 설명하는 방법이 사용자의 경험에 미치는 영향을 탐구하며, 특히 'text-to-SQL Semantic Parsing'(텍스트-모든-쿼리 변환)라는 복잡한 예측 작업에 초점을 맞추었습니다. 세 가지 수준의 모델 설명 방식(저투명도, 중간투명도, 고투명도)을 도입하여 사용자의 AI에 대한 신뢰도와 예측 정확성을 어떻게 변화시키는지 살펴보았습니다.

- **Technical Details**: 해당 연구에서는 약 100명의 참가자를 대상으로 세 가지 알고리즘 투명도 수준(저, 중간, 고)에 따라 설명 접근 방식을 평가했습니다. 참가자는 컴퓨터 과학이나 SQL 프로그래밍을 학습하지 않은 비전문가로서, 주어진 질문을 SQL 쿼리로 변환하는 작업을 수행했습니다. 연구에서 사용된 주요 평가 메트릭은 Propensity to trust와 Jian scale의 신뢰 척도입니다.

- **Performance Highlights**: 결과적으로, (1) 저투명도와 고투명도 설명이 사용자의 결정 의존도를 낮추거나 높이는 경향이 있는 반면, 중간투명도 설명이 적절한 균형을 이루었습니다. (2) 중간투명도 그룹은 시간이 지남에 따라 성과가 증가하는 반면, 다른 그룹은 오히려 감소하는 경향을 보였습니다. (3) 모든 참가자가 연구 후 신뢰도 감소를 보였지만, 중간투명도 설명을 받은 그룹은 신뢰의 변동이 가장 적었습니다.



### Optimal Ground Station Selection for Low-Earth Orbiting Satellites (https://arxiv.org/abs/2410.16282)
Comments:
          13 pages, 3 tables, 4 figures, submitted to IEEE Aeroconf 2025

- **What's New**: 이 논문은 저지구 궤도(LEO) 우주 임무를 위한 최적의 지상국 선택 문제를 해결하는 방안을 제시합니다. 특히 Ground-Station-as-a-Service (GSaaS) 제공업체를 통해 지상 통신 세그먼트를 설계하고 비용을 절감하는 데 중점을 두고 있습니다.

- **Technical Details**: 우리는 지상국 선택 문제를 최적화 문제로 간주하고, 임무 설계자가 전체 최적화 목표를 설정하고 데이터 다운링크 총량, 임무 비용, 반복 운영 비용 및 최대 통신 시간 갭과 같은 주요 임무 성능 변수를 제약하는 일반적인 솔루션 프레임워크를 제시합니다. 이 문제는 정수 프로그래밍(Integer Programming, IP)을 사용하여 해결됩니다. 또한 계산량 증가 문제를 해결하기 위해 실제 시간 범위에서 문제를 해결하여 최적의 지상국 선정을 결정하는 대체 최적화 접근 방식을 도입합니다.

- **Performance Highlights**: 서로 다른 IP 형식을 사용하여 다양한 규모의 LEO 위성 무리를 랜덤하게 선택한 결과를 평가했습니다. 우리가 고려하는 GSaaS 제공자는 Atlas Space Operations, Amazon Web Services (AWS) Ground Station, Azure Orbital Ground Station, Kongsberg Satellite Services (KSAT), Leaf Space 및 Viasat Real-Time Earth입니다. 우리의 결과는 두 개의 주요 지상국 제공업체와 통합하는 표준 운영 관행과 비교되었습니다.



### On Creating an English-Thai Code-switched Machine Translation in Medical Domain (https://arxiv.org/abs/2410.16221)
- **What's New**: 이 연구는 의학 분야에서의 기계 번역(MT)의 중요성과 함께 영어-태국어 MT에서 의료 용어의 정확한 번역이 왜 중요한지를 강조합니다. 기존의 MT 방식이 의학 분야에 적합하지 않은 이유를 제시하며, 의료 전문용어를 유지하는 코드 스위칭(CS) 번역 방법론을 개발하였습니다.

- **Technical Details**: 의료 도메인에서 영어-태국어 CS 번역을 위한 새로운 데이터셋을 생성하고, NLLB 기반 모델을 미세 조정하였습니다. 모델의 성능은 구글 신경 기계 번역(Google NMT) 및 GPT-3.5/GPT-4와 같은 비교 모델들과 평가하였습니다. 연구팀은 52개의 모델을 평가하고, 의료 전문가에게 직접 평가 받았습니다.

- **Performance Highlights**: 모델은 자동화된 메트릭에서 경쟁력 있는 성능을 보여주었고, 인간 평가에서도 높은 선호도를 받았습니다. 연구 결과, 의료 전문가들은 비록 유창성을 약간 저해하더라도 주요 영어 용어를 정확히 유지하는 CS 번역을 선호한다는 것을 발견하였습니다. 이는 전통적인 MT 메트릭이 의료 도메인 번역을 평가하는 데 한계가 있음을 시사합니다.



### Reflection-Bench: probing AI intelligence with reflection (https://arxiv.org/abs/2410.16270)
Comments:
          11 pages, 7 figures, 2 tables

- **What's New**: 이 논문에서는 현재 LLMs(대규모 언어 모델)의 반사(reflection) 능력을 평가하기 위한 새로운 벤치마크인 Reflection-Bench를 소개합니다. 이 벤치마크는 인지 과학의 핵심 원칙을 기반으로 하여 7가지 과제를 포함하고 있으며, 이는 LLM의 인지 기능을 종합적으로 평가할 수 있는 도구입니다.

- **Technical Details**: Reflection-Bench는 6개의 인지 패러다임을 기반으로 하여 설계된 7개의 과제로 구성됩니다. 주요 과제에는 oddball paradigm, n-back task, probabilistic reversal learning task, Wisconsin 카드 분류 테스트, 날씨 예측 과제, double-choice Iowa gambling task, meta-bandit task가 포함됩니다. 이 과제들은 반사를 통해 LLM의 인지 요소를 평가하며, 작업의 난이도를 조정하여 다양한 AI 모델에 적용할 수 있습니다.

- **Performance Highlights**: 13개의 LLM을 평가한 결과, o1-preview가 가장 높은 점수를 기록하며, 나머지 최첨단 LLM들이 그 뒤를 잇는 것으로 나타났습니다. 모든 LLM에서 메타-반사 능력이 전무한 점이 주목할 만하며, 이것은 현재 LLM이 인간 수준의 반사 능력에는 여전히 미치지 못한다는 것을 시사합니다.



### MoRE: Multi-Modal Contrastive Pre-training with Transformers on X-Rays, ECGs, and Diagnostic Repor (https://arxiv.org/abs/2410.16239)
Comments:
          10 pages, 5 figures, 9 tables. Supplementary detail in Appendix. Code made available in Github for reproducibility

- **What's New**: 본 연구에서는 X-ray, ECG, 그리고 Radiology/Cardiology Report를 통합하는 새로운 다중 모달 대조 사전 훈련 프레임워크를 제안합니다. 이 접근법은 각각의 모달리티를 통합된 표현 공간으로 인코딩하여 진단 정확도를 향상시키고 포괄적인 환자 평가를 가능하게 합니다.

- **Technical Details**: 우리는 LoRA-Peft를 사용하여 LLM의 훈련 가능한 매개변수를 현저히 축소하고, Vision Transformer의 최근 선형 주의 드랍 전략을 도입하여 효율적인 주의 메커니즘을 구현합니다. 또한, 모달리티별 기능을 일관성 있는 임베딩으로 정렬하는 대조 손실을 사용합니다.

- **Performance Highlights**: 제안된 방법론을 통해 Mimic-IV, CheXpert, Edema Severity, PtbXl와 같은 다양한 다운스트림 데이터셋에서 최신 기술(SOTA)을 달성했습니다. 이 프레임워크는 복잡한 모달 간 관계를 잘 포착하고, 의료 진단에 강력한 견고성을 보여 앞으로의 다중 모달 학습 연구를 위한 기초를 제공합니다.



### Comprehensive benchmarking of large language models for RNA secondary structure prediction (https://arxiv.org/abs/2410.16212)
- **What's New**: 최근 DNA 및 단백질에 대한 대규모 언어 모델(LLM)의 성공에 영감을 받아 RNA에 대한 여러 LLM이 개발되었습니다. RNA-LLM은 RNA 시퀀스의 대규모 데이터 세트를 사용하여 각 RNA 염기를 의미론적으로 풍부한 숫자 벡터로 표현하는 방법을 자가 감독 방식으로 학습합니다. 이 연구는 통합된 딥러닝 프레임워크 내에서 다양한 사전 훈련된 RNA-LLM을 비교하여 RNA 2차 구조 예측 작업에 대한 포괄적인 실험 분석을 제공합니다.

- **Technical Details**: RNA-LLM은 자연어 처리 분야에서 개발된 딥 표현 학습을 사용하여 RNA 염기를 효과적으로 내장합니다. 대부분의 RNA-LLM은 BERT(양방향 인코더 표현 기반의 변환기)를 기반으로 하며, 이는 문맥 민감한 분산 토큰 표현을 생성하기 위해 설계되었습니다. 이 연구에서는 동일한 딥러닝 아키텍처에 각 RNA-LLM의 임베딩을 피드하여 RNA 2차 구조 예측을 수행합니다.

- **Performance Highlights**: 최신 RNA-LLM의 성능 비교 결과, ERNIE-RNA와 RiNALMo는 각각 F1 점수 0.95를 기록하며 가장 높은 성능을 보였고, RNA-FM은 F1 점수 0.91을 기록했습니다. 대다수의 예측기는 F1 점수 0.60 이상을 달성했습니다. 실험 결과는 RNA 2차 구조 예측에 대한 RNA-LLM의 전반적인효과성을 입증합니다.



### Improve Vision Language Model Chain-of-thought Reasoning (https://arxiv.org/abs/2410.16198)
Comments:
          10 pages + appendix

- **What's New**: 이번 연구에서는 vision language models (VLMs)의 chain-of-thought (CoT) 추론 능력을 향상시키기 위한 접근 방식을 제안합니다. 기존의 훈련 방법들이 충분한 CoT 추론 데이터를 포함하지 못하고 있다는 점을 지적하며, 보다 정교한 응답이 필요한 추론 과제에 대한 일반화가 불충분하다는 것을 보여줍니다.

- **Technical Details**: 두 가지 접근 방식을 통해 문제를 해결하고자 하였습니다. 첫 번째로, GPT-4o 모델로부터 rationale을 증류하여 훈련 데이터를 풍부하게 만들고 VLM을 세부 조정(fine-tuning) 합니다. 두 번째로, 강화 학습(reinforcement learning)을 적용하여 추론 품질을 보정합니다. 이를 위해 모델이 생성한 reasoning chains의 긍정(정확) 및 부정(부정확) 쌍을 구성하여 평가하였습니다.

- **Performance Highlights**: 실험 결과, 기준 데이터셋에서 CoT 추론 능력이 유의미하게 향상되었으며, 직접적인 답변 예측에 대한 일반화 능력 또한 개선되었습니다. 이는 훈련 과정에서 상세한 rationale을 포함시키고 강화 학습을 활용하는 것이 VLM의 추론 능력을 강화하는 데 중요하다는 것을 강조합니다.



### Learning How to Vote With Principles: Axiomatic Insights Into the Collective Decisions of Neural Networks (https://arxiv.org/abs/2410.16170)
Comments:
          15 pages, 8 figures, 7 tables

- **What's New**: 본 논문에서는 투표 이론(voting theory)에서 신경망(neural networks)을 적용할 수 있는 새로운 방법론인 axiomatic deep voting을 제안합니다. 이 프레임워크는 신경망이 집단 결정(collective decisions)을 내릴 때 투표 규칙의 핵심 공리에 부합하는지를 평가하고, 이를 통해 새로운 투표 규칙을 만들 수 있는 가능성을 탐구합니다.

- **Technical Details**: Axiomatic deep voting은 신경망이 선호(preferences)를 집계(aggregate)할 수 있는 방법으로, 이는 전통적인 투표 이론의 공리적 방법(axiomatic method)을 기반으로 합니다. 신경망은 고차원 벡터를 저차원 벡터로 매핑(mapping)하는 함수로 정의됩니다. 이를 통해 다양한 투표 규칙을 최적화 문제로 바라보며, 학습 과정에서 공리 만족(axiom satisfaction)을 최적화하는 방식으로 새로운 투표 규칙을 개발합니다.

- **Performance Highlights**: 신경망은 높은 정확도에도 불구하고 중대한 공리를 위반하는 경향을 보입니다. 데이터 증강(data augmentation)은 신경망의 기능적 학습(principled learning)에는 도움을 주지 않지만 훈련 데이터의 양은 크게 줄일 수 있습니다. 최적화를 통해 개발된 새로운 투표 규칙은 기존 규칙들과 비교할 때 공리 만족도가 높은 수준을 나타냅니다.



### GenAI Assisting Medical Training (https://arxiv.org/abs/2410.16164)
Comments:
          2 pages, 2 figures

- **What's New**: 이번 연구는 간호 교육에서 venipuncture (정맥 채혈)와 cannulation (카테터 삽입)와 같은 중요한 의료 절차를 배울 수 있도록 generative AI (생성적 인공지능) 방법을 통합하여 실시간 피드백 시스템을 제공하는 것을 목표로 하고 있습니다. 이는 교육자의 업무 부담을 줄이는 데 도움이 됩니다.

- **Technical Details**: 참여자의 시연을 통해 수집된 데이터는 static cameras (정적 카메라), GoPro camera (고프로 카메라), IMU (Inertial Measurement Unit) 데이터 등 여러 요소로 구성되어 있습니다. 특히, 각 절차의 세부 단계와 관련된 비디오 분류 방법 및 피드백 제공 방법을 개발 중입니다. 이 과정에서는 Large Language Model (LLM)도 통합되어 사용됩니다.

- **Performance Highlights**: 현재 연구팀은 수집된 데이터를 기반으로 비디오 분류 모델을 조정하고 있으며, 이는 의료 절차의 정확한 피드백을 제공할 수 있게 될 것입니다. 향후 smartwatch data (스마트워치 데이터)를 추가하여 각 절차의 수행 미세한 요소도 분석할 계획입니다.



### A Data-driven Crowd Simulation Framework Integrating Physics-informed Machine Learning with Navigation Potential Fields (https://arxiv.org/abs/2410.16132)
- **What's New**: 본 논문에서는 기존 규칙 기반 물리 모델의 한계를 극복하기 위해 Physics-informed Machine Learning (PIML)과 내비게이션 잠재 필드(navigation potential fields)를 통합한 새로운 데이터 기반 군중 시뮬레이션 프레임워크를 제안합니다.

- **Technical Details**: 제안하는 시스템은 Physics-informed Spatio-temporal Graph Convolutional Network (PI-STGCN)를 기반으로 하며, 군중의 시공간 데이터(spatio-temporal data)를 활용하여 보행자 이동 경향을 예측합니다. 흐름 필드 이론(flow field theory)을 기반으로 한 내비게이션 잠재 필드를 구성하여 보행자의 움직임을 유도하고, 시뮬레이션 중 물리적 제약을 강화합니다.

- **Performance Highlights**: 실험 결과, 제안한 프레임워크는 기존 규칙 기반 방법보다 정확성과 충실도가 향상되었으며, 시뮬레이션된 보행자 궤적과 실제 궤적 사이의 유사성이 10.8% 증가하고 평균 오차는 4% 감소했습니다. 또한, 단순한 깊이 학습(deep learning) 방법에 비해 더 나은 적응성과 해석 가능성을 보였습니다.



### SMART: Self-learning Meta-strategy Agent for Reasoning Tasks (https://arxiv.org/abs/2410.16128)
- **What's New**: 이 논문에서는 SMART (Self-learning Meta-strategy Agent for Reasoning Tasks)라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 언어 모델(LMs)이 다양한 추론 과제에 대해 최적의 전략을 자율적으로 학습하고 선택할 수 있도록 도와줍니다.

- **Technical Details**: SMART는 전략 선택 과정을 Markov Decision Process (MDP)로 모델링하며, 강화 학습(reinforcement learning)에 기반하여 지속적인 자기 개선을 통해 모델이 주어진 작업을 해결하는 데 적합한 전략을 찾도록 합니다. 이 방법은 전통적인 자기 개선 방식에 비해 적은 계산 자원으로 더 효과적으로 전략을 선택할 수 있도록 합니다.

- **Performance Highlights**: SMART는 GSM8K 데이터셋에서 +15 포인트 향상을 보였으며, 전체적인 추론 정확도를 +16 포인트 향상시키는 등 전통적인 방법에 비해 뛰어난 성능을 보여줍니다. 이는 비용 효율성을 높이며, 단일 추론 패스로 높은 정확도를 달성하는 데 기여합니다.



### Multi-Sensor Fusion for UAV Classification Based on Feature Maps of Image and Radar Data (https://arxiv.org/abs/2410.16089)
Comments:
          10 pages, 6 figures

- **What's New**: 현대 UAV의 독특한 비용, 유연성, 속도, 효율성 덕분에 다양한 응용 분야에서 매력적인 선택이 되고 있습니다. 그러나 이는 악의적 또는 우발적인 사고의 보고가 증가하고 있어 UAV 탐지 및 분류 메커니즘 개발이 필수적임을 의미합니다.

- **Technical Details**: 본 연구에서는 이미 처리된 다중 센서 데이터(multi-sensor data)를 새로운 Deep Neural Network(DNN)로 융합(fuse)하는 방법론을 제안합니다. DNN 모델은 열 감지(thermal), 광학(optronic), 레이더(radar) 데이터와 관련된 개별 객체 감지 및 분류 모델에서 추출한 고수준 특징(high-level features)을 결합합니다. 또한, 본 모델의 Convolutional Neural Network(CNN) 기반 아키텍처는 열 및 광학 센서에서 추출된 이미지 특징을 쌓아 올려 세 가지 센서 모달리티(sensor modalities)의 특징을 결합하여 각 센서 단독보다 높은 분류 정확도를 달성합니다.

- **Performance Highlights**: 모델의 성능이 각 센서의 단독 사용보다 뛰어난 분류 정확도를 발휘함을 보여줍니다.



### Critical Example Mining for Vehicle Trajectory Prediction using Flow-based Generative Models (https://arxiv.org/abs/2410.16083)
Comments:
          8 pages,6 figures

- **What's New**: 본 논문에서는 복잡한 주행 시나리오에서의 정확한 궤적 예측(Trajectory Prediction, TP)의 필요성을 강조하며, 데이터 중심 접근 방식을 바탕으로 희소성이 높은 궤적을 추정하는 새로운 사례 발굴 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 NP-hard 문제를 해결하기 위해 Normalizing Flows를 활용하여 궤적 데이터의 확률 밀도를 모델링합니다. 또한, 수작업으로 설계된 특징들을 통해 원본 궤적 데이터를 저차원으로 변환하여 신뢰할 수 있는 입력으로 가공합니다.

- **Performance Highlights**: 실험 결과, 발굴된 사례들은 다양한 하위 예측 모델에 적용했을 때 평균 예측 오류보다 +108.1% 더 높은 예측 오류를 나타냈으며, 이는 데이터셋의 평균에 비해 두 배 이상 높은 수치입니다.



### On-Device LLMs for SMEs: Challenges and Opportunities (https://arxiv.org/abs/2410.16070)
Comments:
          9 pages, 1 figure. The work is supported by the SIT-NVIDIA Joint AI Centre

- **What's New**: 이번 논문은 중소기업(SME)에서 온디바이스(온 디바이스, on-device)로 대규모 언어 모델(LLM)을 배포할 때의 인프라 요구사항에 대해 체계적인 리뷰를 제공합니다. 하드웨어와 소프트웨어 관점을 모두 다루어 중소기업이 혁신적 AI 기술을 통합하여 경쟁력을 유지할 수 있도록 돕고자 합니다.

- **Technical Details**: 하드웨어 관점에서는 GPU와 TPU(Proccessing Unit) 사용, 메모리 및 저장 솔루션의 최적화, 효과적인 배포 전략 등을 논의합니다. 소프트웨어 관점에서는 프레임워크 호환성, 운영체제 최적화, 리소스가 제한된 환경에 맞춘 특화된 라이브러리 활용의 필요성을 강조합니다. 따라서 LLM의 최적화뿐만 아니라 인프라의 전반적인 효율성도 중요합니다.

- **Performance Highlights**: SME에서 LLM을 효과적으로 활용하기 위해 CUDA를 도입하면 데이터 처리 속도 및 기계 학습 작업의 효율성을 향상시킬 수 있으며, TensorFlow Lite와 PyTorch Mobile과 같은 프레임워크는 제한된 리소스 환경에서도 최적의 성능을 보장할 수 있습니다. 이러한 검토는 중소기업이 자사 인프라 내에서 AI 모델을 성공적으로 통합하고 활용할 수 있는 중요한 통찰을 제공합니다.



### A New Approach to Solving SMAC Task: Generating Decision Tree Code from Large Language Models (https://arxiv.org/abs/2410.16024)
- **What's New**: 이 논문에서는 StarCraft Multi-Agent Challenge(SMAC)에서의 다중 에이전트 강화 학습(MARL) 작업을 해결하기 위한 새로운 접근 방식인 LLM-SMAC을 제안합니다. 에이전트는 작업 설명을 통해 대형 언어 모델(LLMs)을 활용하여 결정 트리 코드를 생성하고, 환경에서 제공된 피드백을 통해 자가 반성을 수행합니다.

- **Technical Details**: LLM-SMAC은 최소한의 환경 탐색을 통해 높은 품질의 해석 가능한 결정 트리 모델을 생성합니다. 이 접근 방식은 정책 설명에 대한 비 해석성 문제를 해결하고, LLM의 학습 과정에서 보상 함수의 부재 문제를 극복합니다. 에이전트는 작업 설명을 입력으로 받아 결정 트리 코드를 생성하고, 피드백을 통해 이를 반복적으로 개선합니다.

- **Performance Highlights**: 실험 결과, LLM-SMAC은 SMAC 환경의 대다수 맵에서 높은 품질의 결정 트리 모델을 생성하며, 유사한 SMAC 환경에 수정 없이 적용되는 강한 전이 가능성을 보입니다. 또한, LLM-SMAC으로 훈련된 에이전트는 높은 전략적 탄력성을 나타내고, 특정 루프홀을 이용하는 데 덜 치우치는 경향을 보입니다.



### Are Language Model Logits Calibrated? (https://arxiv.org/abs/2410.16007)
Comments:
          10 pages (main), 24 pages (appendix), under review

- **What's New**: 이번 연구에서는 Language Models (LMs)가 사실적 정보와 확률적 정보의 미묘한 차이를 이해하고 반영해야 한다고 주장합니다. 이 연구는 LMs의 출력 확률이 주어진 텍스트 맥락에 잘 맞춰져 있는지를 조사합니다.

- **Technical Details**: 모델의 'calibration'은 후보 토큰의 출력 확률이 주어진 맥락에서 유추할 수 있는 상대적 가능성과 얼마나 일치하는지를 나타냅니다. 예를 들어, 공정한 동전의 경우 두 가지 동등하게 가능한 옵션(heads 또는 tails)에 대한 맥락에서는 출력 확률이 이를 반영해야 합니다.

- **Performance Highlights**: 연구 결과, 가장 우수한 LMs인 gpt-4o-mini와 Llama-3.1-8B조차도 잘 보정되지 않으며, 체계적인 편향을 보이고 있음이 밝혀졌습니다. gpt-4o-mini는 제공된 두 가지 옵션 중 항상 첫 번째 옵션을 선택하는 경향이 있으며, Llama-3.1-8B는 두 번째 옵션을 선택하는 경향이 있습니다. 또한, Instruction-tuned 모델들은 종종 단일 옵션에 확률 질량을 과다 할당하는 경향이 있어, 사용자가 이해하기 어려운 비직관적인 모델 행동을 초래합니다.



### PROMPTHEUS: A Human-Centered Pipeline to Streamline SLRs with LLMs (https://arxiv.org/abs/2410.15978)
- **What's New**: PROMPTHEUS는 대규모 언어 모델(LLM)을 이용해 시스템(시스템적) 문헌 검토 과정(SLR)을 자동화하는 혁신적인 AI 기반 파이프라인 솔루션이다. 이 시스템은 신속하고 정확한 문헌 검토를 위해 설계되었으며, 특히 인공지능과 같은 빠르게 변화하는 분야에서 연구자들이 신속하게 연구 결과를 종합하고 분석하는 데 큰 도움이 된다.

- **Technical Details**: PROMPTHEUS는 SLR의 주요 단계인 체계적인 검색, 데이터 추출 및 주제 모델링(Topic Modeling) 기능을 포함하여 BERTopic을 통한 정보 구조화와 변환기 모델(transformer models)을 활용한 요약 기능을 자동화한다. 이 시스템은 또한 ROUGE 점수, Flesch 가독성 점수, 코사인 유사성 및 주제 일관성과 같은 여러 지표를 사용하여 포괄적인 평가를 제공하며, 최종 출력의 정확성과 관련성을 크게 향상시킨다.

- **Performance Highlights**: 다섯 개의 연구 영역에서 실시된 평가에 따르면, PROMPTHEUS는 문헌 검토 시간을 줄이고, 높은 정밀도를 달성하며, 일관된 주제 구성을 제공한다. 이는 점점 더 복잡해지는 연구 환경에서 문헌 검토를 효율적이고 효과적으로 수행할 수 있도록 확장 가능한 솔루션을 제공함을 의미한다.



### Enabling Energy-Efficient Deployment of Large Language Models on Memristor Crossbar: A Synergy of Large and Sma (https://arxiv.org/abs/2410.15977)
- **What's New**: 이 논문은 memristor crossbar 아키텍처를 통해 전체 대형 언어 모델(LLM)을 단일 칩에서 배치할 수 있도록 하는 새로운 디자인을 제안합니다. 이 아키텍처는 off-chip 통신에서 발생하는 시간 및 에너지 비효율성을 제거할 수 있습니다.

- **Technical Details**: 제안된 아키텍처는 다중 헤드 어텐션 블록에서 비가중 정적 곱셈(non-weight stationary multiplication)을 지원하고, LLM의 모든 연산을 표준화된 하위 연산으로 분해하여 실행할 수 있는 기능을 갖추고 있습니다. Memristor는 기존 메모리 기술에 비해 높은 밀도를 가지며, 이로 인해 LLM의 극단적인 모델 크기를 효과적으로 관리할 수 있는 가능성이 높습니다.

- **Performance Highlights**: 제안된 아키텍처는 전통적인 memristor crossbar에 비해 최대 39배의 면적 오버헤드와 18배의 에너지 소비 개선을 보여주었습니다. 또한, 현대의 TPU/GPU 시스템에 비해 최소 68배의 면적-지연 곱(area-delay product) 감소와 69%의 에너지 소비 감소를 나타냅니다.



### AI-Driven Innovations in Modern Cloud Computing (https://arxiv.org/abs/2410.15960)
Comments:
          5 pages, 3 figures

- **What's New**: 이 논문은 인공지능(AI)과 클라우드 컴퓨팅(Cloud computing)의 융합을 다루며, 현대 애플리케이션을 혁신적으로 개선하는 방법을 모색합니다.

- **Technical Details**: AI와 클라우드 기술의 결합을 통해 지능형 리소스 관리(intelligent resource management), 예측 분석(predictive analytics), 자동 배포(automated deployment) 및 확장을 실현할 수 있습니다. 이러한 기술은 보안(Security)을 강화하고 고객에게 혁신적인 솔루션을 제공하는 데 기여합니다.

- **Performance Highlights**: 기업은 클라우드와 AI 기술을 활용함으로써 운영 비용을 절감하고 서비스 제공을 향상시키는 등 여러 면에서 큰 이익을 누릴 수 있습니다. 또한, 데이터 프라이버시(data privacy) 문제에 대한 우려와 이를 극복하기 위한 강력한 AI 거버넌스 프레임워크에 대한 논의가 포함되어 있습니다.



### User-centric evaluation of explainability of AI with and for humans: a comprehensive empirical study (https://arxiv.org/abs/2410.15952)
- **What's New**: 이 연구는 사람 중심의 인공지능(HCAI) 분야에서, 설명 가능한 인공지능(XAI) 알고리즘의 사용자가 중심이 된 평가 결과를 다루고 있습니다. XAI 시스템이 제공하는 설명을 인간이 어떻게 이해하고 상호작용하는지를 조사하고자 하였습니다.

- **Technical Details**: 연구팀은 다양한 영역의 데이터 과학, 데이터 시각화, 그리고 머신 러닝 모델 훈련에 사용된 데이터셋 관련 전문 지식을 가진 39명의 참가자를 대상으로 인터뷰를 진행했습니다. 이를 통해 Gradient Boosting Classifier(XGBClassifier)를 사용하는 최첨단 머신 러닝 모델이 생성한 설명의 이해도 평가를 위해 사회과학의 연구 방법론을 활용했습니다. 연구의 재현성을 위해 공개 데이터셋인 UC Irvine 머신 러닝 리포지토리에서 위 독성이 없는 버섯과 독버섯 데이터셋을 사용하였습니다.

- **Performance Highlights**: 이 연구의 결과는 기존의 XAI 방법들이 갖는 한계를 드러내며, 다양한 AI 이해관계자의 정보 필요성과 사용자 관점을 반영하는 새로운 디자인 원칙과 평가 기법이 필요하다는 점을 재확인시켰습니다. 참가자들은 XAI 알고리즘이 전달하는 설명이 비전문가에게 부족한 인사이트를 제공한다는 점을 지적하며, 이는 AI 인터페이스 설계 및 인간-컴퓨터 상호작용의 개선을 위한 귀중한 정보를 제공하는 데 기여할 것입니다.



### Redefining Finance: The Influence of Artificial Intelligence (AI) and Machine Learning (ML) (https://arxiv.org/abs/2410.15951)
Comments:
          10 pages, 1 figure

- **What's New**: 최근 인공지능(AI)과 머신러닝(ML)의 통합이 금융 분야에서 급격한 변화를 일으키고 있습니다. 데이터 기반의 결정 과정이 강조되며, 자동화와 위험 완화를 위한 노력이 이루어지고 있습니다.

- **Technical Details**: 주요 금융 분야인 소매은행( Retail Banking), 자산 관리(Wealth Management), 기업 은행(Corporate Banking), 결제 생태계(Payment Ecosystem)에서 인공지능과 머신러닝의 영향이 커지고 있습니다. 이들 기술은 고객 온보딩( Onboarding)에서부터 사기 탐지 및 예방(Fraud Detection & Prevention) 그리고 고객 서비스 향상까지 다양한 솔루션을 제공합니다.

- **Performance Highlights**: 금융 기관들은 AI와 ML을 주류 애플리케이션에 통합하여 운영 효율성을 향상시키고 있으며, 고급 예측 분석( Predictive Analytics)을 통한 개인화된 고객 경험을 제공하고,사기 탐지 기법(Fraud Detection Techniques)을 통해 위험을 최소화하고 있습니다. 그러나 AI와 ML 채택에 따라 윤리적 및 규제적 도전 과제를 해결하기 위한 강력한 거버넌스 프레임워크와 책임 있는 AI 관행이 필요합니다.



### IGMaxHS -- An Incremental MaxSAT Solver with Support for XOR Clauses (https://arxiv.org/abs/2410.15897)
Comments:
          Presented at the 15th International Workshop on Pragmatics of SAT (PoS 2024, see this https URL )

- **What's New**: 최근 제안된 MaxSAT 기반의 양자 컴퓨팅 오류 수정 방법에 대한 새로운 접근 방식, IGMaxHS를 소개합니다. 이는 XOR 제약 조건을 지원하며 기존의 MaxSAT 솔버보다 제약이 적은 새로운 솔버입니다.

- **Technical Details**: IGMaxHS는 iMaxHS와 GaussMaxHS를 기반으로 하며, XOR 제약 조건의 점진적 추가를 지원합니다. Gaussian elimination을 적용하여 XOR 제약 조건을 해결하며, CDCL SAT 솔버와의 통합을 통해 효율성을 높입니다. 추가적으로, IPAMIR의 확장을 통해 XOR 제약 조건의 점진적 추가가 가능합니다.

- **Performance Highlights**: 최종 퍼지 테스트에서는 10,000개의 인스턴스를 통해 IGMaxHS가 잘못된 해답을 보고하지 않으며, MaxHS, iMaxHS, GaussMaxHS는 만족 가능한 인스턴스에 대해 잘못된 불만족 판단을 보고할 수 있는 것으로 나타났습니다.



### How to Build a Pre-trained Multimodal model for Simultaneously Chatting and Decision-making? (https://arxiv.org/abs/2410.15885)
- **What's New**: 이번 연구는 언어 상호작용과 동적인 상황에서의 정밀한 의사결정이 가능한 새로운 사전 학습 모델 구조인 Visual Language Action model for Chatting and Decision Making (VLA4CD)를 개발하여, 이를 통해 자율주행과 같은 복잡한 제어 작업에서의 성능 향상을 입증했습니다.

- **Technical Details**: VLA4CD는 LoRA(LoRA: Low-Rank Adaptation)를 활용하여 여러 모달리티(모델에 대한 입력의 다양한 형식) 데이터를 정교하게 조정하며, 연속적인 값의 행동 결정을 제공하도록 설계되었습니다. 기존의 LLM은 텍스트 응답만을 생성할 수 있지만, VLA4CD는 실시간 의사결정 중에 텍스트 데이터를 동기식으로 생성할 수 있습니다. 이 모델은 행동 데이터를 디스크리타이징하지 않고 연속 값으로 처리하여 복잡한 의사결정 시나리오에 적합하게 제작되었습니다.

- **Performance Highlights**: CARLA에서의 실험 결과, VLA4CD는 SOTA(State-of-the-art) VLA 모델에 비해 보다 정확한 실시간 의사결정을 제공하며, LLM이 고유하는 텍스트 상호작용 기능을 완전히 유지합니다.



### LLM4GRN: Discovering Causal Gene Regulatory Networks with LLMs -- Evaluation through Synthetic Data Generation (https://arxiv.org/abs/2410.15828)
- **What's New**: 본 논문에서는 대형 언어 모델(Large Language Models, LLMs)을 활용하여 유전자 조절 네트워크(Gene Regulatory Networks, GRNs)를 추론하는 새로운 방법론을 제시합니다. 이 방법론은 기존의 통계적 방법론과의 결합을 통해 생물학적 지식을 활용하여 GRN의 발견 가능성을 높입니다.

- **Technical Details**: LLMs는 개별 세포의 RNA 시퀀싱(scRNA-seq) 데이터에서 GRN을 추론하기 위해 사용됩니다. LLMs는 잠재적인 전사 인자(Transcription Factors, TFs)의 목록을 생성하거나 기존의 통계적 인과 발견 알고리즘과 결합하여 GRN을 수립합니다. 이러한 방법은 신뢰할 수 있는 그라운드 트루스(ground truth) 그래프가 없더라도 인과 합성 데이터 생성(causal synthetic data generation)을 통해 GRN을 평가할 수 있게 합니다.

- **Performance Highlights**: LLMs와 통계적 GRN 추론을 결합한 최적의 성능을 보여주며, 이는 scRNA-seq 데이터 분석의 유망한 방향성을 지시합니다. 본 연구는 LLMs가 GRN 추론에 있어 효과적인 도구임을 입증하였으며, 생물학적 및 통계적 평가를 통한 결과의 신뢰성을 확보했습니다.



### The effect of fine-tuning on language model toxicity (https://arxiv.org/abs/2410.15821)
Comments:
          To be presented at NeurIPS 2024 Safe Generative AI Workshop

- **What's New**: 이 연구는 언어 모델의 파라미터 효율적인 미세 조정(fine-tuning)이 모델의 안전성(safety) 및 독성(toxicity) 생성에 미치는 영향을 평가합니다. 특히 Gemma, Llama, Phi 모델을 대상으로 다양한 실험을 통해 미세 조정이 독성 출력에 어떻게 영향을 미치는지를 분석합니다.

- **Technical Details**: 실험에서는 비대립적(non-adversarial) 데이터셋을 사용하여 파라미터 효율적인 미세 조정 기법인 Low-Rank Adaptation(LoRA)을 적용하였습니다. 모델 개발자가 수행한 지침 조정(instruction-tuning)과 안전성을 유지하기 위한 노력도 비교했습니다.

- **Performance Highlights**: 미세 조정을 통해 독성 언어 생성이 줄어드는 효과가 확인되었습니다. 그러나 커뮤니티 기여자가 미세 조정한 모델은 예측할 수 없는 방식으로 특정 독성률이 증가할 수 있음을 보여주었습니다. 이는 결국 사용자에게 배포된 모델의 안전성과 책임에 대한 의문을 제기합니다.



### RAG4ITOps: A Supervised Fine-Tunable and Comprehensive RAG Framework for IT Operations and Maintenanc (https://arxiv.org/abs/2410.15805)
Comments:
          Accepted by EMNLP 2024 Industry Track

- **What's New**: 이 논문에서는 IT 운영 및 유지 보수를 위한 QA 시스템을 구축하기 위한 효율적이고 관리가능한 RAG4ITOps 프레임워크를 제안합니다. 이 프레임워크는 기업 전용 데이터 처리 및 도메인 특화 QA 시스템 구축에 초점을 맞추고 있습니다.

- **Technical Details**: RAG4ITOps는 두 가지 주요 단계로 구성됩니다: (1) 모델 파인튜닝 및 데이터 벡터화, (2) 온라인 QA 시스템 프로세스. 대조 학습(constrastive learning) 기법과 두 가지 음의 샘플링 전략을 활용하여 임베딩 모델을 파인튜닝합니다.

- **Performance Highlights**: 실험 결과, RAG4ITOps 프레임워크는 두 가지 QA 작업에서 경쟁 모델보다 우수한 성능을 발휘하며, 실제 기업 환경에 적용 가능함을 증명합니다.



### A roadmap for generative mapping: unlocking the power of generative AI for map-making (https://arxiv.org/abs/2410.15770)
- **What's New**: 이번 논문은 생성형 AI(Generative AI)가 지도 제작에서의 가능성을 강조하며, 일반 대중이 더 쉽게 지도를 만들 수 있도록 돕기 위한 발전 방향을 제시합니다.

- **Technical Details**: 지도 제작은 GIS(Geographic Information System) 및 카토그래피(cartography) 전문가들에게 주로 국한되어 있으며, 데이터 처리(data processing)부터 시각화(visualization)까지 복잡한 작업 흐름(workflow)을 포함합니다. 생성형 AI의 최근 발전을 통해 이러한 복잡성을 줄일 수 있는 방법을 모색합니다.

- **Performance Highlights**: 논문은 생성형 매핑 시스템(Generative Mapping System, GMS)을 개발하기 위한 로드맵(roadmap)을 제공하며, 현재의 기술적 도전(challenges)과 필요한 기술들(technologies)을 정리합니다.



### Alchemy: Amplifying Theorem-Proving Capability through Symbolic Mutation (https://arxiv.org/abs/2410.15748)
- **What's New**: 이 연구는 Neural Theorem Proving (NTP) 분야에서의 데이터 부족 문제를 해결하기 위해 새로운 데이터 합성 프레임워크인 Alchemy를 제안합니다. 이 프레임워크는 Mathlib의 후보 정리를 기초로 하여 기호 변형을 통해 수학 정리를 생성하는 방법을 설명합니다.

- **Technical Details**: Alchemy는 Mathlib에서 후보 정리를 기반으로 사용 가능한 모든 정리를 식별하고, 이 정리의 관련 항목을 동등한 형태나 전제로 대체하여 후보 정리를 변형합니다. 이 과정에서 Mathlib의 정리 수는 110,657에서 6,326,679로 증가합니다. 이러한 변형된 정리들을 활용해 LLM (Large Language Models)을 지속적으로 사전 학습하고 지도 학습을 수행합니다.

- **Performance Highlights**: 실험 결과는 Alchemy 접근 방식이 Leandojo 벤치마크에서 5%의 성능 향상을 달성함을 보여줍니다. 또한, 합성된 데이터는 miniF2F 벤치마크에서 2.5%의 성능 개선을 이루었습니다. 이 연구 결과는 LLM의 정리 증명 능력을 향상시키는 데 기여할 것으로 기대됩니다.



### GIG: Graph Data Imputation With Graph Differential Dependencies (https://arxiv.org/abs/2410.15747)
Comments:
          12 pages, 4 figures, published to ADC

- **What's New**: 이 논문은 데이터베이스 인스턴스에서 누락된 값을 보완하는 새로운 접근법인 GIG(그래프 데이터 보완 기법)을 제안합니다. GIG는 그래프 차별 종속성(GDDs)에 기반하여 변환기(transformer) 모델을 학습하여 그래프 내에서 누락된 데이터의 값을 예측합니다.

- **Technical Details**: GIG는 지식 그래프에서 GDDs를 학습하고 이를 사용하여 누락된 값 후보들을 검증하며, GDD 규칙을 기반으로 의미적 일관성을 유지합니다. GDD는 그래프 엔티티 종속성과 차별화되며, 거리 및 매칭 함수를 통합하여 다양한 오류를 처리합니다.

- **Performance Highlights**: 실험 결과, GIG는 기존의 최첨단 접근법들과 비교해 효과성이 두드러지며, 실제 데이터셋 7개에서 성능 개선을 보여줍니다.



### AutoTrain: No-code training for state-of-the-art models (https://arxiv.org/abs/2410.15735)
- **What's New**: 이 논문에서는 다양한 작업에 대해 모델을 교육할 수 있는 오픈 소스, 노 코드 툴인 AutoTrain(또는 AutoTrain Advanced)을 소개합니다. 이 도구는 개별 산업 또는 오픈 소스 응용 프로그램에 맞춘 솔루션을 개발하는 데 필수적인 커스텀 데이터셋에서 모델을 학습시키는 과정을 단순화합니다.

- **Technical Details**: AutoTrain은 대형 언어 모델(LLM) 파인튜닝, 텍스트 분류/회귀, 토큰 분류, 시퀀스-투-시퀀스 작업, 시각적 언어 모델(VLM) 파인튜닝, 이미지 분류/회귀 및 테이블 데이터의 분류와 회귀 작업을 지원하는 오픈 소스 라이브러리입니다. 이 라이브러리는 파라미터 튜닝, 모델 검증, 분산 학습, 모니터링, 유지 보수와 같은 여러 가지 모델 학습 관련 문제를 해결합니다. AutoTrain은 CLI, GUI/UI 및 Python SDK를 통해 사용 가능합니다.

- **Performance Highlights**: 모델 훈련이 간편하며, Hugging Face Hub에서 공유된 수만 개의 모델과 호환됩니다. 사용자는 로컬 모드 또는 클라우드 머신을 통해 작업을 수행할 수 있고, 학습된 모델은 Hugging Face Hub에 푸쉬하여 전 세계의 사용자와 공유할 수 있습니다.



### InternLM2.5-StepProver: Advancing Automated Theorem Proving via Expert Iteration on Large-Scale LEAN Problems (https://arxiv.org/abs/2410.15700)
- **What's New**: 이 논문에서는 Large Language Models (LLMs)을 활용하여 수리 정리를 자동으로 증명하는 새로운 접근 방식을 제안합니다. Lean-workbook 데이터 세트를 사용하여 전문가 반복(expert iteration) 기법을 적용하고, 20,000 CPU 일 이상 소모하여 모델을 학습합니다.

- **Technical Details**: 우리는 Lean-workbook-plus 데이터 세트를 활용하여, 크리틱 모델을 훈련시켜 비교적 쉬운 문제를 선택하게 합니다. 이 과정에서 해결된 문제의 수와 증명 길이 및 CPU 사용량 간에 로그-선형(log-linear) 관계가 발견되었습니다. 기존의 증명 접근 방식보다 더 깊은 증명을 생성하는 방식으로 통합된 크리틱 모델을 사용하였습니다.

- **Performance Highlights**: InternLM2.5-StepProver는 MiniF2F 및 Lean-Workbook-Plus 벤치마크에서 최신 오픈 소스(state-of-the-art) 성능을 달성했습니다. MiniF2F-test에서 65.9%의 통과률을 기록하였으며, Lean-Workbook-Plus 문제 중 17.0%를 증명(혹은 반증)하여 이전의 9.5%와 비교하여 현저한 개선을 보였습니다.



### Geographical Node Clustering and Grouping to Guarantee Data IIDness in Federated Learning (https://arxiv.org/abs/2410.15693)
Comments:
          10 pages, 7 figures

- **What's New**: 본 논문은 지리적 특성을 활용하여 비동일한 데이터(non-IID) 문제를 해결하기 위한 새로운 연합 학습(FL) 메커니즘을 제안합니다. 기존의 연구들이 데이터 수집을 조작하는 데 집중했던 것과 달리, 본 연구는 모바일 IoT 노드를 클러스터링하여 데이터의 IID(Independent and Identically Distributed) 특성을 보장하는 방법을 제안합니다.

- **Technical Details**: 연구에서는 Dynamic Clustering(동적 클러스터링)과 Partial-Steady Grouping(부분 안정적 그룹화) 알고리즘을 제안합니다. 이 알고리즘들은 노드 간의 이동성과 IoT 데이터의 상관관계를 고려하여 FL 참여자를 최적화된 그룹으로 나누어 데이터 분포가 거의 IID에 가깝도록 합니다. 특히, 실험 결과는 수집된 데이터 간의 거리와 그 데이터의 IID 특성 간의 관계를 보여줍니다.

- **Performance Highlights**: 제안된 메커니즘은 기존의 벤치마크 그룹화 알고리즘들보다 적어도 110배 더 높은 성능을 보였습니다. 이는 dropout이 발생한 장치의 수와 각 그룹의 장치 수의 균형을 맞추는 데 있어 훨씬 더 효율적이라는 것을 의미합니다. 그룹의 수는 최대 0.93 그룹만 증가합니다.



### Long Term Memory: The Foundation of AI Self-Evolution (https://arxiv.org/abs/2410.15665)
Comments:
          56 pages, 13 figures

- **What's New**: 이번 연구는 인공지능(AI) 모델의 자아 진화(self-evolution) 능력의 중요성을 강조하며, 이를 위한 장기 메모리(long-term memory, LTM)의 필요성을 제시하고 있습니다. LTM을 통해 AI 모델이 제한된 데이터나 상호작용을 통해 진화할 수 있다는 점에서 혁신적인 접근을 보여줍니다.

- **Technical Details**: AI 자아 진화는 데이터의 누적과 경험 학습을 통해 이루어지며, LTM은 정의된 경험을 저장하고 관리하는 구조를 제안합니다. 이를 통해 모델은 다양한 환경과 상호작용에서 경험을 통합하여 지능적으로 발전할 수 있습니다. 연구에서는 LTM을 활용하여 개인화된 모델을 구축하고 이 모델들이 상호작용을 통해 자아 진화를 달성하는 방법을 분류하고 있습니다.

- **Performance Highlights**: 연구에 따르면, 다중 에이전트 프레임워크인 OMNE는 GAIA 벤치마크에서 1위를 차지하여 LTM이 AI 자아 진화에 대한 잠재력을 입증했습니다. LTM을 활용한 이 접근 방식은 특히 개인화 및 장기적인 데이터 관리 측면에서 AI 기술의 발전에 기여할 것으로 보입니다.



### LightFusionRec: Lightweight Transformers-Based Cross-Domain Recommendation Mod (https://arxiv.org/abs/2410.15656)
- **What's New**: LightFusionRec는 DistilBERT와 FastText를 이용한 경량의 크로스 도메인 추천 시스템으로, 데이터 희소성과 계산 효율성 문제를 해결하는 혁신적인 접근법을 제시합니다.

- **Technical Details**: LightFusionRec는 DistilBERT를 통해 콘텐츠 설명의 의미 있는 텍스트 특징을 인코딩하고, FastText 모델을 이용해 장르 특징을 추출합니다. 이 하이브리드 모델은 서로 다른 콘텐츠 도메인(예: 영화 및 도서)을 연결하며, 효율적으로 작동하여 특히 온디바이스 인퍼런스에 적합합니다.

- **Performance Highlights**: 실험 결과, LightFusionRec는 기존 방법에 비해 추천 품질에서 더욱 두드러진 개선사항을 보여주며, 여러 미디어 형식에 대한 정확하고 맥락에 맞는 추천을 제공하여 사용자의 디지털 콘텐츠 경험을 향상시킵니다.



### Opportunities and Challenges of Generative-AI in Financ (https://arxiv.org/abs/2410.15653)
- **What's New**: 이번 논문에서는 금융 분야에서 Generative Artificial Intelligence (Gen-AI) 기술의 활용 기회와 챌린지(Challenges)를 종합적으로 소개합니다.

- **Technical Details**: Gen-AI는 대규모 언어 모델(LLM, Large Language Model)을 기반으로 하며, 정보를 빠르고 저지연으로 처리할 수 있습니다. Gen-AI는 고객 서비스, 투자 거래, 문서 처리, 리스크 관리 등 다양한 금융 영역에서 활용될 수 있습니다. 특히 고객 상호작용을 위한 챗봇 및 대화형 응용 프로그램 구축에 적합합니다.

- **Performance Highlights**: Gen-AI 기술의 도입으로 고객 서비스의 생산성이 30%에서 45% 향상될 수 있으며, 특정 기업은 700명의 인력을 대체할 만큼의 성과를 보였습니다. 이는 고객 대화 해결 시간을 11분에서 2분으로 단축하는 것을 의미합니다.



### Voice-Enabled AI Agents can Perform Common Scams (https://arxiv.org/abs/2410.15650)
- **What's New**: 최근의 다중 모달 (multi-modal) 기능이 뛰어난 LLM(대규모 언어 모델)의 발전 덕분에 음성 인식 AI 에이전트가 가능해졌습니다. 이러한 에이전트는 자율 고객 서비스와 같은 새로운 애플리케이션을 지원하고 있지만, 새로운 기능에 대한 이중 용도 (dual use) 가능성도 제기되고 있습니다.

- **Technical Details**: 이 논문에서는 정부의 데이터를 바탕으로 수집한 일반적인 사기 수법을 대상으로 음성 인식 AI 에이전트를 설계하였습니다. 에이전트는 간단한 도구에 접근할 수 있는 LLM(GPT-4o)을 기반으로 하며, 각 사기에 대한 구체적인 지침을 포함하고 있습니다. 에이전트는 웹사이트 탐색, HTML 가져오기, 요소 클릭, 입력 필드 채우기 및 자바스크립트 실행의 기능이 포함된 브라우저 액세스 도구를 사용합니다.

- **Performance Highlights**: 실험 결과, 음성 인식 AI 에이전트는 20%에서 60%의 성공률을 보였으며, 평균 성공률은 36%로 나타났습니다. 예를 들어, 은행 송금 사기는 26회의 작업을 요구하며, 복잡한 사기는 실행하는 데 최대 3분이 소요되었습니다. 성공적인 사기당 평균 비용은 $0.75 안팎입니다.



### Boosting Jailbreak Transferability for Large Language Models (https://arxiv.org/abs/2410.15645)
- **What's New**: 이번 논문에서는 대형 언어 모델(Large Language Models, LLMs)의 안전성과 관련된 문제, 특히 악성 컨텐츠를 생성하기 위한 jailbreak 공격을 다룹니다. 기존의 GCG(Greedy Coordinate Gradient)와 같은 방법의 한계를 극복하기 위해, 시나리오 유도 템플릿(scenario induction template), 최적화된 접미사 선택(optimal suffix selection), 재접미사 공격 메커니즘을 통합하여 일관되지 않은 출력을 줄이는 새로운 접근법을 제안합니다.

- **Technical Details**: 제안된 방법은 악성 질문과 각 타겟 템플릿을 고려하여 jailbreak 접미사를 최적화합니다. 초기 출력이 최적화 목표에 align 되는 경우에도, 생성된 jailbreak 접미사가 모델이 해로운 내용을 생성하도록 충분히 유도하지 못할 수 있습니다. 이를 해결하기 위해 고정된 해로운 템플릿을 사용하고, 각 최적화 단계에서 손실 값이 가장 작은 상위 5개의 접미사를 평가하여 다음 업데이트에 가장 효과적인 것을 선택합니다. 이상적으로, 제안된 SI-GCG 공격은 거의 100%의 성공률을 기록했습니다.

- **Performance Highlights**: 저자들은 SI-GCG 접근법이 기존의 LLM jailbreak 공격 방법들에 비해 상당히 높은 공격 성공률을 기록했다고 보고하며, 이는 기존 최적화 기반 jailbreak 기술과 결합되어 높은 전이 가능성(fool rate)을 보장합니다. 이러한 결과는 향후 LLM의 안전성을 더욱 강화하는 데 기여할 것으로 보입니다.



### Procedural Content Generation in Games: A Survey with Insights on Emerging LLM Integration (https://arxiv.org/abs/2410.15644)
- **What's New**: 이 논문은 Procedural Content Generation (PCG)과 관련된 다양한 알고리즘의 비교와 최신 기술인 Large Language Models (LLMs)의 역할을 조사하고 있으며, 기계 학습 및 딥 러닝의 발전이 PCG에 미친 영향을 다루고 있습니다.

- **Technical Details**: PCG는 게임 콘텐츠를 자동으로 생성하는 알고리즘으로, 검색 기반 방법 (예: Monte Carlo Tree Search), 기계 학습 기반 방법 (전통적인 머신러닝 및 딥 러닝 포함), 기타 기법 (예: noise functions)으로 분류됩니다. 이 논문에서는 LLMs의 특성과 기존 방법들과의 차별성을 강조하며, 방법의 조합을 통한 새로운 PCG 접근법을 설명합니다.

- **Performance Highlights**: PCG는 게임 제작의 생산성을 향상시키고, 플레이어의 재방문 가치를 높이며, 개발 비용을 절감하는 데 기여하고 있습니다. 최근 5년 간의 연구 경향을 분석하여 PCG의 발전 방향과 존재하는 연구 의의 및 미래의 가능성에 대해 논의하고 있습니다.



### Weighted Diversified Sampling for Efficient Data-Driven Single-Cell Gene-Gene Interaction Discovery (https://arxiv.org/abs/2410.15616)
- **What's New**: 본 논문은 데이터 기반의 컴퓨팅 도구를 활용하여 주목할 만한 유전자-유전자 상호작용을 발견하는 혁신적인 접근 방식을 제시합니다. 고급 Transformer 모델을 활용하였으며, 새롭게 개발된 가중치 기반의 다양성 샘플링 알고리즘을 도입하여 데이터 효율성을 극대화하고, 단일 세포 데이터셋에서 단 1% 샘플링만으로도 전체 데이터셋을 사용하는 것과 유사한 성능을 달성할 수 있음을 보여줍니다.

- **Technical Details**: 제안된 알고리즘은 단 두 번의 데이터셋 통과로 각 데이터 샘플의 다양성 점수를 계산하며, 데이터셋의 크기에 관계없이 상수 메모리만 필요합니다. 이 알고리즘은 데이터 기반의 유전자-유전자 상호작용 발견을 위한 효율적인 하위 집합 생성을 가능하게 합니다. CelluFormer라는 Transformer 모델을 통해 단일 세포 전사체 데이터에 최적화된 학습을 수행하며, 주목할 만한 성능을 발휘합니다.

- **Performance Highlights**: 광범위한 실험 결과, 단일 세포 데이터셋의 1%를 샘플링함으로써 전체 데이터 셋을 활용하는 것과 유사한 성능을 달성하는 것을 확인했습니다. 본 연구는 유전자-유전자 상호작용 발견 연구의 효율성을 높이는 중요한 기회를 제공합니다.



### P-YOLOv8: Efficient and Accurate Real-Time Detection of Distracted Driving (https://arxiv.org/abs/2410.15602)
- **What's New**: Distracted driving detection을 위한 새로운 머신러닝 모델인 P-YOLOv8이 제안되었습니다. 이 모델은 기존 모델의 CPU 및 메모리 사용의 한계를 해결하며, 실시간으로 높은 정확도를 유지합니다.

- **Technical Details**: P-YOLOv8 (You Only Look Once, version 8) 모델은 전이 학습(pretrained)된 YOLOv8 아키텍처를 기반으로 하며, 가벼운 모델 크기(2.84 MB)와 1,451,098개의 파라미터를 가진 효율성을 강조합니다. 이 모델은 22,424개의 이미지가 포함된 Distracted Driver Detection dataset을 활용하여, 99.46%의 높은 정확도를 기록하였습니다. 또한, this model은 개선된 bounding box 예측 기법과 anchor boxes의 효과적인 사용으로 detection precision과 speed를 향상시킵니다.

- **Performance Highlights**: P-YOLOv8은 기존의 VGG16, VGG19, ResNet과 같은 딥러닝 모델들에 비해 속도와 컴퓨팅 효율성에서 우수한 성능을 보여주며, resource-constrained devices에서 실시간 적용 가능성을 높였습니다.



### Patrol Security Game: Defending Against Adversary with Freedom in Attack Timing, Location, and Duration (https://arxiv.org/abs/2410.15600)
Comments:
          Under review of TCPS

- **What's New**: 이번 연구는 Patrol Security Game (PSG)에서 공격자의 공격 시간을 최적화하는 문제를 최초로 다루었습니다. 이를 통해 방어자가 최적의 순찰 일정을 수립할 수 있는 방법을 제시하고 있습니다.

- **Technical Details**: 연구는 PSG를 combinatorial minimax 문제로 전환하고, 방어자의 전략을 time-homogeneous first-order Markov chain으로 제한하여 최적의 솔루션을 도출합니다. 이 과정에서 공격자의 기대 보상을 줄이기 위해 높은 엔트로피(Entropy)를 강조하고, 4가지 알고리즘(TSP-based, Biased random walk, Walk on state graph, Graph pointer network)을 제안하였습니다.

- **Performance Highlights**: 실험 결과 제안된 알고리즘들이 기존의 최상위 성능을 초과하여 Synthetic 및 실제 범죄 데이터셋에서 공격자의 기대 보수를 줄일 수 있음을 입증하였습니다.



### A Comprehensive Survey of Datasets, Theories, Variants, and Applications in Direct Preference Optimization (https://arxiv.org/abs/2410.15595)
- **What's New**: 대규모 언어 모델(LLMs)의 발전과 함께 인간의 선호에 맞춰 정책 모델을 정렬하는 것이 점점 더 중요해지고 있습니다. 본 논문에서는 Direct Preference Optimization (DPO)의 최신 연구 동향과 도전 과제를 포괄적으로 검토하며, 이 분야의 발전 상황을 체계적으로 정리합니다.

- **Technical Details**: 본 연구에서는 DPO의 이론적 분석, 다양한 변형, 관련 선호 데이터셋 및 응용 프로그램에 대해 논의합니다. DPO는 정책 모델과 참조 모델에 대해 간단한 최대 우도(objective)를 설정하여 명시적 보상 모델 훈련 단계를 우회하고 강화 학습 최적화를 필요로 하지 않도록 합니다. DPO의 최적화 목표는 정책 모델 자체에 의해 매개변수화된 암묵적 보상 함수와 동등합니다.

- **Performance Highlights**: DPO는 다양한 응용 프로그램에서 안정성, 성능 및 계산 효율성을 보여주고 있으며, 최근 연구 결과에 따르면 RLHF보다 높은 성능을 보이는 온라인 변형 및 데이터 수집 전략 등이 제안되고 있습니다. DPO의 새로운 변형(KTO, IPO, CPO 등) 또한 최근 발표되어 DPO의 장단점을 보완하고 있습니다.



### Improving Clinical Documentation with AI: A Comparative Study of Sporo AI Scribe and GPT-4o min (https://arxiv.org/abs/2410.15528)
- **What's New**: 이번 연구에서는 Sporo Health의 AI 스크라이브가 OpenAI의 GPT-4o Mini와 비교하여 여러 성능 지표에서 더 뛰어난 성과를 보였음을 보고합니다.

- **Technical Details**: Sporo Health의 AI 스크라이브는 미세 조정된 의료 LLMs를 활용하는 다중 에이전트 시스템으로, 비식별화된 환자 대화 기록을 기준으로 AI 생성 요약과 의사 생성 노트를 비교하여 평가하였습니다. 평가 기준으로는 clinical content recall, precision, F1 scores가 사용되었으며, 의사 만족도 조사는 수정된 PDQI-9 도구를 통해 이루어졌습니다.

- **Performance Highlights**: Sporo AI는 recall, precision 및 전반적인 F1 점수에서 지속적으로 GPT-4o Mini를 초과하였으며, 정확성, 포괄성 및 관련성 측면에서도 더 높은 평가를 받았습니다. 생성된 AI 요약은 환자 데이터의 개인 정보 보호 및 보안을 유지하면서 임상 문서화를 개선하는 효과적인 도구임을 보여주었습니다.



### Anonymising Elderly and Pathological Speech: Voice Conversion Using DDSP and Query-by-Examp (https://arxiv.org/abs/2410.15500)
Comments:
          Accepted in Interspeech 2024

- **What's New**: 이 논문에서는 노화 및 병적 음성 데이터에서 개인 식별자를 변경하며 언어적 내용을 유지하여 발화자의 정체성을 보호하는 새로운 음성 변환 기반 방법을 제안합니다.

- **Technical Details**: 이 방법은 차별적 디지털 신호 처리(differentiable digital signal processing)와 예제에 의한 질의(query-by-example)를 기반으로 하여 고유한 언어적, 운율적(prosodic), 및 도메인 특성을 분리하는 훈련 방식을 포함하며, 비정상 음성 패턴에 적응할 수 있습니다.

- **Performance Highlights**: 객관적 및 주관적인 평가 결과, DDSP-QbE 방법은 다양한 데이터 세트, 병리 및 화자에 걸쳐 음성 인지력, 운율, 및 도메인 보존 측면에서 기존의 음성 변환 방법보다 우수한 성능을 보였습니다.



### Improving Voice Quality in Speech Anonymization With Just Perception-Informed Losses (https://arxiv.org/abs/2410.15499)
Comments:
          Accepted in NeurIPS 2024 Workshop (Audio Imagination)

- **What's New**: 본 연구는 기존의 음성 변환(Voice Conversion, VC) 기술에 인간 청각 시스템에서 영감을 받은 새로운 손실 함수(loss function)를 도입하여 스피커의 익명성을 유지하면서도 자연스러움(naturalness), 이해 가능성(intelligibility), 그리고 억양(prosody)을 향상하는 방법을 제안합니다.

- **Technical Details**: 제안된 손실 함수는 모델에 구애받지 않으며, 수작업(feature-based) 및 딥러닝 기반의 특성을 통합하여 음성 품질을 향상시키고, VQVAE(벡터 양자화 변이 오토인코더) 모델을 중심으로 평가됩니다. VQVAE는 음성의 멜 스펙트로그램(mel-spectrogram)을 변환하는데, 인코더와 디코더 사이에 벡터 양자화(layer)가 포함되어 있습니다. 이러한 구조는 스피커 정보를 제거하면서 음성을 변환합니다.

- **Performance Highlights**: 객관적 및 주관적 평가를 통해 VQVAE 기반 모델이 일반 VQVAE 모델보다 음성 품질이 크게 향상된 것을 보여줍니다. 다양한 데이터 세트, 언어, 목표 스피커, 성별에서 이러한 개선이 일관되게 관찰되었습니다.



### Dynamic Intelligence Assessment: Benchmarking LLMs on the Road to AGI with a Focus on Model Confidenc (https://arxiv.org/abs/2410.15490)
- **What's New**: 기존 벤치마크의 단점을 개선하기 위해 동적인 질문 템플릿과 다양한 분야에서 향상된 메트릭을 사용한 'Dynamic Intelligence Assessment (DIA)' 방법론을 제안합니다.

- **Technical Details**: DIA는 수학, 암호학, 사이버 보안 및 컴퓨터 과학 등 여러 분야를 포괄하며, 150개의 다양한 과제 템플릿을 포함한 DIA-Bench 데이터셋을 제공합니다. 새로운 메트릭으로는 Reliability Score, Task Success Rate, Confidence Index, Near Miss Score가 소개되었습니다.

- **Performance Highlights**: DIA-Bench를 통해 8개의 최첨단 LLM을 평가한 결과, 현재 모델들은 복잡한 작업에서 어려움을 겪으며, 심지어 간단한 질문에 대해서조차 예기치 않게 낮은 신뢰도를 보여주었습니다. 이 연구는 AI 모델의 문제 해결 능력을 평가는 새로운 기준을 제시합니다.



### Multi-Layer Feature Fusion with Cross-Channel Attention-Based U-Net for Kidney Tumor Segmentation (https://arxiv.org/abs/2410.15472)
Comments:
          8 pages

- **What's New**: 본 연구에서는 CT 스캔 이미지를 통한 자동화된 신장 종양(renal tumor) 감지를 위한 향상된 U-Net 기반 모델을 제안합니다.

- **Technical Details**: 제안된 모델은 convolution layer 간 잔여 연결(residual connections)을 사용하고, 엔코더 블록 내에서 다층 기능 융합(multi-layer feature fusion, MFF)과 채널 간 주의(cross-channel attention, CCA)를 통합하며, MFF 및 CCA를 통해 파생된 추가 정보를 통해 증가된 스킵 연결(skip connections)을 포함합니다.

- **Performance Highlights**: 모델은 KiTS19 데이터셋에서 평가되었으며, 신장 분할(kidney segmentation)에서 Dice Similarity Coefficient (DSC) 0.97 및 Jaccard index (JI) 0.95를 달성했습니다. 신장 종양 분할에서 DSC 0.96 및 JI 0.91을 기록하며, 현재의 선도적 모델들을 능가하는 성능을 보였습니다.



### How Aligned are Generative Models to Humans in High-Stakes Decision-Making? (https://arxiv.org/abs/2410.15471)
- **What's New**: 이 연구는 대규모 생성 모델(Large Generative Models, LMs)이 고위험 결정-making 작업인 재범 예측에 대해 인간 및 기존의 예측 AI 모델과 어떻게 비교되는지 다루고 있습니다.

- **Technical Details**: COMPAS 데이터셋을 사용하여 인간의 재범 판단 및 사진 데이터와 결합된 다중 모드(Multimodal) LMs의 특성을 연구합니다. 다양한 방법으로 인간의 판단, COMPAS 점수로 LMs를 'steer'하며, 사진의 추가가 정확도에 미치는 영향을 분석합니다.

- **Performance Highlights**: 결과적으로 LMs는 인간보다 더Aligned되었으며, in-context learning을 통해 더 정확한 예측이 가능함을 발견했습니다. 그러나 어떤 anti-discrimination prompting 기법은 LMs의 예측력을 오히려 감소시킬 수 있습니다.



### Hallucination Detox: Sensitive Neuron Dropout (SeND) for Large Language Model Training (https://arxiv.org/abs/2410.15460)
- **What's New**: 대규모 언어 모델(LLMs)의 신뢰성과 관련된 문제를 다루고 있으며, 특히 환각(hallucinations) 현상의 발생 원인을 분석하고 이를 감소시키기 위한 새로운 훈련 프로토콜인 SEnsitive Neuron Dropout (SeND)을 제안합니다.

- **Technical Details**: 이 연구는 Pythia 모델 시리즈(70M-12B parameters)의 다양한 모델을 분석하고, Efficient EigenScore (EES)라는 새로운 비지도 환각 탐지 메트릭을 개발하여 SeND 프로토콜에 통합합니다. SeND는 높은 변동성을 가진 신경세포(Sensitive Neurons)를 선택적으로 제거함으로써 변동성을 줄이고 신뢰성을 높입니다.

- **Performance Highlights**: SeND 방법을 통해 LLM의 신뢰성을 정상 훈련 대비 최대 40% 개선하고, 위키피디아 및 의료 데이터셋과 같은 도메인에서 사실 정확성을 향상시킬 수 있음을 입증했습니다.



### Heterogeneous Graph Reinforcement Learning for Dependency-aware Multi-task Allocation in Spatial Crowdsourcing (https://arxiv.org/abs/2410.15449)
- **What's New**: 이 논문은 Dependency-aware Multi-task Allocation (DMA) 문제를 정식으로 조사하고, 이를 해결하기 위한 Heterogeneous Graph Reinforcement Learning-based Task Allocation (HGRL-TA)라는 새로운 프레임워크를 제안합니다. 기존의 작업 할당 방식과는 달리, 우선순위를 고려하여 복잡한 작업을 여러 하위 작업으로 나누고, 이들을 적절한 작업자에게 할당하는 더 효율적인 방법을 제시하고 있습니다.

- **Technical Details**: HGRL-TA 프레임워크는 각 하위 작업을 순차적으로 적합한 작업자에게 할당하는 마르코프 결정 프로세스(Markov Decision Process, MDP)로 문제를 다룹니다. 이는 multi-relation graph와 Compound-path-based Heterogeneous Graph Attention Network (CHANet)를 활용하여 작업과 작업자 간의 복잡한 관계를 효과적으로 표현하고, 문제 상태를 임베딩하여 최적의 할당 결정을 지원합니다. CHANet는 노드 임베딩을 통해 상태 및 행동에 대한 임베딩을 생성하고, Proximal Policy Optimization (PPO) 방법에 의해 동시 학습됩니다.

- **Performance Highlights**: 제안된 HGRL-TA는 DEA 문제를 해결하는 데 있어 평균 21.78% 더 높은 수익을 달성했습니다. 이를 통해 HGRL-TA의 효과성과 다양한 문제 인스턴스에 대한 일반화 능력이 강조되었습니다. 실험 결과는 HGRL-TA가 메타휴리스틱 방법보다 높은 성과를 보여줍니다.



### Exploring Social Desirability Response Bias in Large Language Models: Evidence from GPT-4 Simulations (https://arxiv.org/abs/2410.15442)
- **What's New**: 이 연구는 대규모 언어 모델(LLM)인 GPT-4가 사회 조사의 맥락에서 사회적 바람직성 응답 편향(SDR bias)을 보이는지 조사하였습니다. 2022년 갤럽 세계 여론조사 데이터를 기반으로, 네 개의 사회에서 인물 캐릭터를 할당한 후, SDR을 유도하기 위한 설명 문구의 유무에 따라 생성된 데이터를 비교하였습니다.

- **Technical Details**: 조사는 2022년 갤럽 세계 여론조사 데이터를 사용하여 홍콩, 남아프리카, 영국, 미국의 네 가지 사회에서 각 500명의 응답자를 무작위로 선정하였습니다. 각 사회의 응답자는 연령과 성별에 따라 층화 샘플링을 통해 선정되었으며, LLM GPT-4를 사용하여 응답을 생성하였습니다. 연구는 SDR 지표와 시민 참여 지표를 이용하여 분석하였습니다.

- **Performance Highlights**: 결과는 혼합적이었습니다. 다짐 문구가 SDR 지수 점수를 증가시켜 SDR 편향을 나타내었지만, 시민 참여 점수는 감소하였습니다. 연구는 LLM이 인간의 응답에서 보이는 편향을 모사할 수 있는 가능성에 대한 추가적인 통찰력을 제공하며, LLM의 예측 성능에 대한 최소한의 영향을 확인하였습니다.



### Unveiling and Consulting Core Experts in Retrieval-Augmented MoE-based LLMs (https://arxiv.org/abs/2410.15438)
- **What's New**: 본 논문은 Retrieval-Augmented Generation (RAG) 시스템의 효과성과 내부 메커니즘을 개선하기 위해 Mixture-of-Expert (MoE) 기반 LLM의 전문가 활성화를 분석합니다.

- **Technical Details**: 이 연구에서는 Contrastive Expert Activation Inspection (CEAI) 방법을 제안하여, MoE 기반 LLM의 서로 다른 데이터 시나리오에서 활성화된 전문가를 비교하여 핵심 전문가를 식별합니다. 핵심 전문가들은 특정 문맥과 모델 행동을 관리하며, 특정 문맥에서 주로 활성화됩니다.

- **Performance Highlights**: 다양한 데이터셋에서 실험한 결과, 제안된 방법이 RAG 시스템의 효율성과 효과성을 개선하는 데 성공적이라는 것을 보여주었습니다.



### Power Plays: Unleashing Machine Learning Magic in Smart Grids (https://arxiv.org/abs/2410.15423)
Comments:
          16 pages, 1 figure

- **What's New**: 기계 학습(Machine Learning)의 통합은 스마트 그리드 시스템의 효율성, 신뢰성 및 지속 가능성을 향상시키기 위한 혁신적인 단계로 자리잡고 있습니다. 이 연구는 스마트 그리드의 로드 포리캐스팅(load forecasting), 에너지 배급 최적화(energy distribution optimization), 결함 탐지(fault detection) 등 여러 분야에서 ML 알고리즘의 역할과 영향력을 분석합니다.

- **Technical Details**: 스마트 그리드는 전통적인 전력망과의 차별점으로서 데이터와 전기의 양방향 흐름을 가능하게 하는 다양한 기술을 통합합니다. 기계 학습 알고리즘은 대량의 데이터를 분석하여 에너지 분배를 최적화하고, 수요를 예측하며, 시스템의 이상 현상을 탐지하여 grid의 안정성을 높이는 데 기여합니다.

- **Performance Highlights**: 이 연구는 ML의 적용이 스마트 그리드의 에너지 효율성 및 신뢰성을 개선할 수 있는 여러 사례를 제시합니다. 예를 들어, 강화 학습(reinforcement learning)을 활용하여 에너지 저장 시스템의 성능을 최적화하고, 사이버 공격에 대한 대응을 신속하게 할 수 있는 기계를 제공함으로써 스마트 그리드의 전반적인 안정성을 증가시킵니다.



### CASET: Complexity Analysis using Simple Execution Traces for CS* submissions (https://arxiv.org/abs/2410.15419)
Comments:
          5 pages

- **What's New**: 이번 논문에서는 학생의 알고리즘 제출을 평가하기 위해 새로운 도구인 CASET(Complexity Analysis using Simple Execution Traces)를 제안합니다. CASET는 동적 실행 추적(dynamic traces)과 비지도 기계 학습(unsupervised machine learning)을 이용하여 알고리즘의 시간 복잡성을 분석합니다.

- **Technical Details**: CASET는 Valgrind라는 도구를 사용하여 프로그램 실행의 동적 추적을 생성하고, 이를 통해 미리 결정된 시간 복잡성 범주로 분류합니다. 이 과정은 제출된 코드의 실행 결과가 특정 시간 복잡성에 맞는지를 체크하여 과제를 자동으로 채점하는 방법을 제공합니다.

- **Performance Highlights**: 실험 결과, CASET는 여러 종류의 알고리즘 (예: 정렬, 검색 알고리즘 및 동적 프로그래밍 알고리즘)의 시간 복잡성을 정확히 추정할 수 있음을 보여주었습니다. 이 도구는 튜터들이 코드의 내용을 읽지 않고도 알고리즘의 종류를 인식할 수 있도록 도와줍니다.



### XAI-based Feature Ensemble for Enhanced Anomaly Detection in Autonomous Driving Systems (https://arxiv.org/abs/2410.15405)
Comments:
          31 pages, 4 figures (including the subfigures)

- **What's New**: 이 논문은 자율주행 차량(Autonomous Vehicle, AV)의 보안 및 신뢰성을 높이기 위해 여러 Explainable AI(XAI) 기법을 결합한 새로운 feature ensemble 프레임워크를 제안합니다.

- **Technical Details**: SHAP, LIME, DALEX 등의 다양한 XAI 기법을 통해 선택된 주요 feature들을 Decision Trees, Random Forests, Deep Neural Networks, K Nearest Neighbors, Support Vector Machines, AdaBoost와 같은 6가지 AI 모델에서 융합합니다. 이는 특이점 탐지(anomaly detection)와 해석 가능성(interpretability)을 강화하는 데 중요한 역할을 합니다.

- **Performance Highlights**: 이 프레임워크를 통해 생성된 feature 세트는 CatBoost, Logistic Regression, LightGBM과 같은 독립 분류기(classifier)를 사용하여 평가되어 공정한 성능을 보장하였으며, VeReMi와 Sensor 두 개의 자율주행 데이터셋에서 정확도, 강건성(robustness), 투명성(transparency)이 향상된 결과를 보여주었습니다.



### A Survey of Hallucination in Large Visual Language Models (https://arxiv.org/abs/2410.15359)
- **What's New**: 최근 논문에서는 Large Visual Language Models (LVLMs)와 관련된 hallucination 현상의 발전을 다루고 있습니다. LVLM은 시각적 모달리티(visual modality)를 통합하여 사용자 경험을 풍부하게 만들고자 하며, hallucination 문제는 LVLM의 효과적인 활용에 있어 장애물로 작용하고 있습니다. 이 논문은 LVLM의 구조와 hallucination 발생 원인, 그리고 최근의 교정 및 완화 기법을 종합적으로 정리합니다.

- **Technical Details**: LVLM은 세 가지 모듈, 즉 perceptual module, cross-modal module, 및 response module로 구성됩니다. 각 모듈은 시각 정보를 추출하고 이를 텍스트 공간으로 매핑하여 최종 응답을 생성하는 기능을 수행합니다. 특히, perceptual module은 Vision Transformer (ViT)를 활용하여 이미지를 고차원 벡터로 변환하고, cross-modal module은 비전과 언어 간의 모달리티 간극(modal gap)을 메우기 위한 구조를 채택합니다.

- **Performance Highlights**: 최근 연구에서는 LVLM의 hallucination 완화 및 교정 기법을 데이터셋 재작성(data rewrite), 모달리티 간극(modalities gap) 및 출력 교정(output correction) 등 세 가지 주요 카테고리로 나누어 정리했습니다. 이러한 기법들은 LVLM의 신뢰성과 유용성을 향상시키기 위한 방향으로 제시되었습니다.



### POSE: Pose estimation Of virtual Sync Exhibit system (https://arxiv.org/abs/2410.15343)
- **What's New**: 이 연구는 휴대 가능한 MetaVerse 구현체로서, 3D pose estimation을 활용하여 가상 아바타가 환경과 상호작용하고 동기화된 동작을 수행하도록 하는 플랫폼을 개발했습니다. 우리는 피트니스 링으로 게임을 할 때 조이스틱과 센서를 사용하는 것이 불편하다는 점에 착안했습니다.

- **Technical Details**: 본 연구에서는 TransPose 모델을 사용하여 사람의 동작을 인식하고 3D 환경 안에 에이전트에 지시를 전송하는 방식을 설계하였습니다. 또한, Depth와 Joint의 3D 위치 계산을 위해 두 개의 렌즈를 사용하고, AprilTags를 통해 위치를 자동으로 보정합니다. 이 과정에서 Open Source Python 프레임워크인 panda3D를 활용하여 3D 환경을 구현했습니다.

- **Performance Highlights**: 본 연구의 결과로, 전반적인 지연 시간을 줄이고 Real-Time 경험을 제공하는 동시 처리(multi-processing) 계산 방식을 도입하였습니다. IK(Inverse Kinematics) 방법을 통해 아바타의 움직임을 제어하며, 볼 제약(ball constraint)과 경첩 제약(hinge constraint)을 통해 모델의 움직임을 더욱 정교하게 만들었습니다.



### IKDP: Inverse Kinematics through Diffusion Process (https://arxiv.org/abs/2410.15341)
- **What's New**: 이 논문에서는 로봇의 역운동학(Inverse Kinematics) 문제를 Conditional Denoising Diffusion Probabilistic Model(조건부 노이즈 제거 확률 모델)을 이용해 해결하는 방법을 제시합니다. 기존의 Jacobian 기법이나 기계 학습 방법과 달리, DDPM을 통해 더욱 정확한 솔루션을 도출하고자 합니다.

- **Technical Details**: 역운동학 방법(IK)은 로봇의 각 관절이 목표 위치에 도달하기 위해 필요한 각도를 계산합니다. 이 연구에서는 로봇의 팔이 목표 위치에 도달하기 위해 필요한 관절 각도(θ)를 기계 학습 목표 함수로 변환합니다. 이 과정에서 bone vector와 angular ratio vectors의 각도를 기반으로 한 방법론을 제안합니다.

- **Performance Highlights**: 기존의 역운동학 알고리즘보다 향상된 이동 유연성을 보이며, DDPM의 접근법을 통해 로봇 팔이 더 자연스럽고 정확하게 목표 위치에 도달하는 성능을 기대하게 됩니다.



### Who is Undercover? Guiding LLMs to Explore Multi-Perspective Team Tactic in the Gam (https://arxiv.org/abs/2410.15311)
- **What's New**: 이번 연구는 다차원적 사고를 강조하는 새로운 프레임워크, Multi-Perspective Team Tactic (MPTT)를 제안합니다. 이 프레임워크는 LLM이 복잡한 시나리오에서 인지 능력을 발휘할 수 있도록 하는 데 중점을 두고 있습니다.

- **Technical Details**: MPTT는 'Who is Undercover?'라는 언어 논리 게임을 실험 플랫폼으로 사용하여, 말하기와 투표 세션을 교 altern하여 진행합니다. 주요 요소로는 자기 관점(self-perspective), 정체성 결정(identity-determination), 자기 반성(self-reflection), 자기 요약(self-summary), 다회차 팀 찾기(multi-round find-teammates)가 포함됩니다.

- **Performance Highlights**: MPTT와 WIU의 결합을 통해 LLM은 인지 능력을 최대한 활용하여 의사 결정 프레임워크를 생성하고, 이는 사회적 소수자들이 의사 소통 및 표현을 돕고 공정성과 다양성을 촉진하는 데 기여합니다. 초기 결과는 LLM이 인간 행동을 학습하고 정렬할 수 있는 잠재력을 나타냅니다.



### Contextual Augmented Multi-Model Programming (CAMP): A Hybrid Local-Cloud Copilot Framework (https://arxiv.org/abs/2410.15285)
Comments:
          12 pages, 3 figures, 4 tables

- **What's New**: 이 논문은 클라우드 기반의 대형 언어 모델(LLMs) 통합에 어려움을 겪고 있는 Apple 소프트웨어 생태계에서의 프로그래밍 지원을 위한 CAMP라는 새로운 다중 모델 AI 프로그래밍 프레임워크를 제안합니다. CAMP는 프로그램의 문맥을 이해하고 최적화하기 위해 Retrieval-Augmented Generation(RAG) 기법을 활용하는 로컬 모델로 구성됩니다.

- **Technical Details**: CAMP는 로컬 IDE 내에서 클라우드 모델의 성능을 최적화하며, Xcode를 위한 Copilot을 통해 실제적으로 구현되었습니다. 이 도구는 코드 완성, 문서화, 오류 탐지, 지능형 사용자-에이전트 상호작용을 포함한 다양한 생성 프로그래밍 작업을 수행할 수 있는 기능을 제공합니다. RAG는 미리 훈련된 LLM과 정보 검색 기법을 결합하여 관련 문서를 검색하고 언어 모델의 생성 과정을 조정하는 데 도움을 줍니다.

- **Performance Highlights**: 실험 결과, 제안된 시스템은 코드 품질 및 사용자 채택 실험에서 긍정적인 성과를 보여주어 AI 지원 프로그래밍 영역에서 중요한 기여를 할 것으로 예상됩니다.



### AI Can Enhance Creativity in Social Networks (https://arxiv.org/abs/2410.15264)
- **What's New**: 이 논문은 SocialMuse라는 피어 추천 엔진을 통해 자아 조직형 소셜 네트워크에서 사람들의 창의성을 향상시킬 수 있는 방법을 제시합니다. 모델은 온라인 플랫폼에서 수집한 데이터 기반으로 사람들의 아이디어 성과를 예측합니다.

- **Technical Details**: 논문은 두 가지 주요 도전 과제를 다룹니다: (i) 데이터 수집의 어려움과 (ii) 개입 디자인의 도전. 데이터 수집에서는 자극적인 네트워크 엣지(idea inspiration links) 추적이 어려운 점과 개인의 주관적 사회적 요인이 자아 네트워크에 미치는 영향을 다룹니다. SocialMuse는 수집된 데이터를 기반으로 AI 기반 피어 추천을 제공하여 이 문제를 해결합니다.

- **Performance Highlights**: SocialMuse를 활용한 실험에서는 AI 비인식 대조망에 비해 창의성 측정에서 더 높은 성과를 달성했습니다. 피어 추천 기능은 네트워크 구조적 특성을 강조하여 사람들의 영감 출처를 분산시켜 독창적인 아이디어의 성과를 더 잘 돋보이게 만듭니다.



### Economic Anthropology in the Era of Generative Artificial Intelligenc (https://arxiv.org/abs/2410.15238)
- **What's New**: 이 논문은 경제 인류학과 생성적 인공지능(Generative Artificial Intelligence, GenAI)의 교차점을 탐구합니다. 대형 언어 모델(Large Language Models, LLMs)이 인간의 의사결정을 어떻게 모사할 수 있는지를 다룹니다.

- **Technical Details**: 연구에서는 두 가지 AI 모델 C.A.L.L.O.N. (Conventionally Average Late Liberal ONtology)과 M.A.U.S.S. (More Accurate Understanding of Society and its Symbols)를 소개합니다. C.A.L.L.O.N.은 표준 데이터로 학습되며, M.A.U.S.S.는 인류학적 지식으로 적응되어 있습니다. 또한, 인류학적 훈련이 LLMs의 다양한 경제 시스템과 개념 인식 능력을 어떻게 향상시킬 수 있는지를 강조합니다.

- **Performance Highlights**: 결과는 경제 인류학을 AI와 통합함으로써 경제에 대한 보다 다원적인 이해를 제공하고 비시장(non-market) 경제 시스템의 지속 가능성을 향상시킬 수 있음을 시사합니다.



### Bias Amplification: Language Models as Increasingly Biased Media (https://arxiv.org/abs/2410.15234)
Comments:
          Submitted to ARR Roling Review October

- **What's New**: 이번 논문은 Large Language Models(LLMs)에서 발생하는 bias amplification(편향 증폭)의 메커니즘을 독립적으로 분석하고, 기존 모델 붕괴(model collapse)와의 관계에서 이해하는 데 기여하고 있습니다. 이 논문은 이론적 프레임워크와 함께 실험적 근거를 제시하여 LLMs에서의 편향 증폭 현상을 구체적으로 보여줍니다.

- **Technical Details**: 저자들은 편향 증폭이 발생하는 데 필요한 조건을 정의하는 이론적 프레임워크를 제안하고, 이를 통해 통계적 시뮬레이션을 수행하여 그 원리를 설명합니다. GPT-2 모델을 사용하여 편향 증폭을 실제로 실험하고, 세 가지 가능한 완화 전략인 Overfitting, Preservation, Accumulation을 검토하여 각각의 효과를 비교합니다. 또한, 메커니즘 해석 기법을 사용하여 편향 증폭과 모델 붕괴가 서로 다른 뉴런 집합에 의해 구동됨을 보입니다.

- **Performance Highlights**: GPT-2 모델은 문장 연속성 과제에서 우편향 경향(right-leaning bias)을 보였으며, 이전 반복으로부터 생성된 합성 데이터에 대한 반복적 미세 조정(iterative fine-tuning)을 통해 편향이 점진적으로 증가하는 양상을 보입니다. Preservation과 Accumulation 전략은 편향 증폭과 모델 붕괴를 효과적으로 완화하는 것으로 나타났습니다.



### Chasing Random: Instruction Selection Strategies Fail to Generaliz (https://arxiv.org/abs/2410.15225)
- **What's New**: 이 연구는 다양한 소스 데이터셋, 선택 예산 및 평가 벤치마크를 통해 인기 있는 선택 전략을 분석하고, 이러한 전략들이 일반화 능력이 부족하다는 것을 보여줍니다. 또한 데이터 선택의 비용-성과 트레이드 오프를 평가하고, 많은 경우 데이터 선택이 전체 데이터셋을 사용하는 것보다 더 높은 비용을 초래한다는 사실을 밝혀냈습니다.

- **Technical Details**: 연구팀은 60개 이상의 실험 구성과 4개의 평가 벤치마크를 통해 수행된 실험에서 수집된 데이터를 분석하였습니다. 중요한 발견으로는 (a) Instruction Selection Strategies(명령어 선택 전략)가 비슷한 실험 설정에 잘 일반화되지 않으며 무작위 선택을 일관되게 초과하지 않음, (b) 일반 목적의 Instruction Following(명령어 수행) 능력은 주관적인 목표며, 따라서 다양한 측면에서 선택 전략을 비교할 경우 상충된 경향을 보일 수 있음, (c) 선택 예산을 증가시키면 많은 전략이 불리하게 확장되며 선택 비용이 전체 데이터셋으로 훈련하는 비용을 초과할 수 있음이 포함되었습니다.

- **Performance Highlights**: 연구 결과는 선택 전략들이 무작위 기준선에 비해 일관되게 높은 성능을 보이지 않음을 시사합니다. 이로 인해 기존의 명령어 선택 전략을 실제 환경에서 적용하기 어렵게 만듭니다. 무작위 샘플링을 통한 간단한 방법이 일정 상황에서 더 비용 효율적일 수 있다는 점도 강조되었습니다.



### AutoFLUKA: A Large Language Model Based Framework for Automating Monte Carlo Simulations in FLUKA (https://arxiv.org/abs/2410.15222)
Comments:
          58 pages including text, figures, references and appendices

- **What's New**: 이번 연구에서는 FLUKA의 Monte Carlo (MC) 시뮬레이션 워크플로우의 자동화를 위한 AI 에이전트인 AutoFLUKA를 소개하고 있습니다. AutoFLUKA는 자연어 처리(Natural Language Processing) 및 자율적 추론(Autonomous Reasoning)을 통합하여, 입력 파일 수정, 시뮬레이션 실행, 결과 처리 등 전통적인 방법보다 더 효율적으로 작업을 수행합니다.

- **Technical Details**: AutoFLUKA는 LangChain Python Framework를 이용하여 개발된 AI 에이전트 애플리케이션으로, FLUKA MC 시뮬레이션 워크플로우의 입력 파일을 자동으로 수정하고, 시뮬레이션을 실행하여 결과를 처리할 수 있는 기능을 제공합니다. 이는 사용자의 수고와 오류 가능성을 크게 줄여줍니다.

- **Performance Highlights**: 케이스스터디를 통해 AutoFLUKA가 일반화된 및 도메인 특정 사례를 모두 처리할 수 있는 능력을 입증하였으며, Microdosimetry와 같은 복잡한 분야에서도 자동화된 워크플로우를 통해 유연성과 확장성을 보여주었습니다. 추가적으로, Retrieval Augmentation Generation (RAG) 도구의 활용 가능성을 강조하여 사용자 경험을 더욱 개선할 수 있는 기반을 마련하였습니다.



### Medical-GAT: Cancer Document Classification Leveraging Graph-Based Residual Network for Scenarios with Limited Data (https://arxiv.org/abs/2410.15198)
- **What's New**: 본 연구는 갑상선암, 대장암, 폐암과 같은 여러 종류의 암에 대한 분류 성능을 향상시키기 위해, 1,874개의 생물 의학 초록을 포함하는 새로운 데이터세트를 제시합니다. 특히, R-GAT(Residual Graph Attention Network) 모델을 적용하여 데이터를 활용하는 혁신적인 접근을 시도하였습니다.

- **Technical Details**: R-GAT 모델은 그래프 기반 접근법을 이용하여 암 관련 문서의 의미 정보와 구조적 관계를 포착합니다. 여러 그래프 주의(GAT) 층을 통해 텍스트 간의 연결을 효과적으로 강조하며, TF-IDF, Word2Vec, BERT 및 RoBERTa 토크나이저를 포함한 다양한 특징 추출 방법을 평가하였습니다.

- **Performance Highlights**: R-GAT 모델은 갑상선암에 대해 0.99의 정밀도(precision), 0.97의 재현율(recall), 0.98의 F1 점수를 달성하였으며, 대장암과 폐암에서도 각각 0.95와 0.96 이상의 F1 점수를 기록하여 기타 기법들을 압도적으로 초월한 성능을 보였습니다.



### Augmented Lagrangian-Based Safe Reinforcement Learning Approach for Distribution System Volt/VAR Contro (https://arxiv.org/abs/2410.15188)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2209.09772

- **What's New**: 이번 논문에서는 활성 배전 시스템에서 Volt-VAR 제어 문제를 해결하기 위한 데이터 기반 솔루션을 제안합니다. 기존의 배전 시스템 모델들이 항상 부정확하고 불완전하기 때문에 문제 해결이 어려웠습니다. 이를 해결하기 위해 Volt-VAR 제어 문제를 제약 마르코프 결정 프로세스(CMDP)로 정식화하였습니다.

- **Technical Details**: 이 논문은 증강 라그랑지안 방법과 소프트 액터-비평가(soft actor-critic) 알고리즘을 결합하여 새로운 안전 오프 정책 강화 학습(RL) 접근 방식을 제안합니다. 액터 네트워크는 라그랑지안 값 함수를 사용하여 정책 기울기 방식으로 업데이트되며, 이중 비평가 네트워크(double-critics network)를 채택하여 행동 가치 함수(action-value function)를 동기적으로 추정하여 과대 추정 편향을 피합니다.

- **Performance Highlights**: 제안된 알고리즘은 검토된 문제에 대한 강한 볼록성 보장을 요구하지 않으며 샘플 효율성이 높습니다. 오프라인 훈련과 온라인 실행을 위한 이단계 전략이 채택되어 정확한 배전 시스템 모델이 더 이상 필요하지 않습니다. 중앙 집중식 훈련 분산 실행 전략을 채택하여 대규모 배전 시스템에 대한 분산 Volt-VAR 제어가 가능하도록 합니다. 실제 전력 데이터로 수행된 종합적인 수치 실험을 통해 제안된 알고리즘이 높은 해결 최적성과 제약 조건 준수를 달성할 수 있음을 보여줍니다.



### Linguistic Fuzzy Information Evolution with Random Leader Election Mechanism for Decision-Making Systems (https://arxiv.org/abs/2410.15171)
- **What's New**: 본 논문에서는 고전적인 DeGroot 모델과 Hegselmann-Krause 모델의 한계를 극복하기 위해 새로운 언어적 퍼지 정보 동역학 모델인 PRRLEM-DeGroot, PRRLEM-HOHK, PRRLEM-HEHK을 제안합니다. 이 모델은 에이전트가 각 정보 업데이트 후 무작위로 선택된 일시적인 리더로부터 더 큰 영향을 받도록 하여 정보 공유를 증대시킵니다.

- **Technical Details**: PRRLEM(Per-Round Random Leader Election Mechanism)을 기반으로 하는 세 가지 새로운 모델을 제안합니다. Monte Carlo 방법을 활용하여 복잡한 시스템의 행동을 시뮬레이션하고, 퍼지 정보의 신뢰 구간을 생성합니다. 또한, 향상된 GRRV(Golden Rule Representative Value)를 도입하여 신뢰 구간을 순위화합니다.

- **Performance Highlights**: 시뮬레이션 예제와 실제 우주 상황 인식 사례를 통해 제안한 모델의 효과성을 검증하였습니다. 제안된 알고리즘은 정확성과 견고성에서 기존 모델을 능가하며, 정보 확산에서 에코 챔버 문제를 해결할 수 있는 능력을 보여줍니다.



### SPA-Bench: A Comprehensive Benchmark for SmartPhone Agent Evaluation (https://arxiv.org/abs/2410.15164)
- **What's New**: 새로운 스마트폰 에이전트 벤치마크인 SPA-Bench를 소개하며, 이는 다양한 임무 범위 및 다수의 에이전트를 통합한 평가 파이프라인을 통해 (M)LLM 기반 에이전트를 공정하게 비교할 수 있는 혁신적인 도구입니다.

- **Technical Details**: SPA-Bench는 340개의 작업을 포함하며, 150개의 단일 앱 작업과 20개의 교차 앱 작업을 다루고, 영어와 중국어 앱 모두에서 작동합니다. 이 시스템은 11개의 에이전트를 통합하고, 자동화된 평가 파이프라인을 통해 성과를 측정합니다.

- **Performance Highlights**: 우리는 실험을 통해 제안된 에이전트의 성능을 평가하며, 독점 (M)LLM을 사용하는 에이전트가 미세 조정된 (M)LLM이나 오픈 소스 (M)LLM을 사용하는 것보다 뛰어남을 보여주었지만, 실제 배포에는 시간과 비용 문제가 남아 있습니다.



### Optimizing Large Language Models for Dynamic Constraints through Human-in-the-Loop Discriminators (https://arxiv.org/abs/2410.15163)
- **What's New**: 이 논문은 일반적인 텍스트 입력-텍스트 출력 패러다임의 한계를 극복하기 위해 대규모 언어 모델(LLMs)이 시스템 인터페이스와 상호작용하고 제약 개념을 요약하며 성과 메트릭을 지속적으로 최적화할 수 있는 유연한 프레임워크를 제안합니다.

- **Technical Details**: 이 프레임워크는 두 단계 학습 절차로 구성됩니다: 첫 번째 단계에서는 시스템 인터페이스로부터 제약 조건을 추출하고, 두 번째 단계에서는 인간 전문가와 LLM 기반 에이전트가 협력하여 중요 사례를 식별하고 에이전트 성능을 향상시킵니다. 이를 위해 인간 감별자와 LLM 감별자를 사용하여 데이터 중심의 제약 학습을 구현합니다.

- **Performance Highlights**: 제안된 프레임워크는 초기 피드백을 통해 인간 감별자와 함께 7.78%의 통과율을 달성했으며, 이는 기준값에서 40.2% 향상된 수치입니다. LLM 기반 감별자와 함께한 통과율은 6.11%로 나타났습니다. 이 결과는 다양한 제약 기반 응용 프로그램에 적용할 수 있는 잠재력을 보여줍니다.



### Simulation-Based Optimistic Policy Iteration For Multi-Agent MDPs with Kullback-Leibler Control Cos (https://arxiv.org/abs/2410.15156)
- **What's New**: 본 논문에서는 다중 에이전트 Markov 결정 과정(MDP)에서 정적 최적 확률 정책을 학습하기 위한 에이전트 기반의 낙관적 정책 반복(Optimistic Policy Iteration, OPI) 방법론을 제안합니다. 이 방법론은 제어 노력에 대한 Kullback-Leibler(KL) 발산 비용과 공동 상태에 대한 추가 비용을 포함하고 있습니다.

- **Technical Details**: 제안된 방법은 탐욕스러운 정책 개선 단계와 m단계의 시간 차이(Temporal Difference, TD) 정책 평가 단계를 포함합니다. 즉각적인 비용의 분리 구조를 활용하여 정책 개선 단계가 현재 가치 함수 추정과 제어되지 않은 전이 확률에 의존하는 볼츠만 분포를 따름을 보입니다. 이로 인해 각 에이전트는 개선된 공동 정책을 독립적으로 계산할 수 있습니다. 비동기적(Asynchronous) OPI 버전인 ASYNC-KLC-OPI는 모든 에이전트의 최적 가치 함수와 최적 공동 정책으로 비대칭적으로 수렴하는 것을 증명합니다.

- **Performance Highlights**: 다중 에이전트 MDP에서 KL 제어 비용 변형 Stag-Hare 게임을 통한 시뮬레이션 결과, 제안된 KLC-OPI 방법론이 비용 회수를 최소화하는 공동 정책을 효과적으로 학습함을 확인하였습니다.



### MCCoder: Streamlining Motion Control with LLM-Assisted Code Generation and Rigorous Verification (https://arxiv.org/abs/2410.15154)
- **What's New**: 이번 논문은 MCCoder라는 LLM(대형 언어 모델)을 기반으로 한 코드 생성 시스템을 제안하여 복잡한 모션 제어 작업을 위한 코드를 생성합니다. 이 시스템은 소프트-모션 데이터 검증을 통합하여 코드의 안전성과 효과성을 높이고자 합니다. 또한, 다양한 난이도의 모션 제어 작업을 평가하기 위한 평가 데이터셋인 MCEVAL도 소개합니다.

- **Technical Details**: MCCoder는 모션 제어를 위한 파이썬 코드를 생성하며, 태스크 분해(task decomposition), 검색(retrieval), 코드 생성(code generation), 소프트-모션(soft-motion), 자기 수정(self-correction) 및 데이터 검증(data verification)이라는 6개의 모듈로 구성됩니다. 소프트-모션 아키텍처는 일반 PC에서 실행될 수 있으며, 실제 머신을 작동시키는 실시간 엔진과 코드 실행 후 피드백을 제공하는 시뮬레이션 엔진을 갖추고 있습니다. 또한, MCEVAL 데이터셋은 116개의 다양한 난이도의 모션 제어 프로그래밍 과제를 포함하고 있습니다.

- **Performance Highlights**: 실험 결과, MCCoder는 MCEVAL 데이터셋에서 11.61%의 전반적인 성능 향상과 복잡한 작업에서는 66.12%의 성능 향상을 보였습니다. 이는 MCCoder가 기존의 기본 모델과 비교했을 때 우수한 코드 생성 능력을 갖추고 있음을 보여줍니다.



### A Prompt Refinement-based Large Language Model for Metro Passenger Flow Forecasting under Delay Conditions (https://arxiv.org/abs/2410.15111)
Comments:
          14 pages, 2 figures

- **What's New**: 이 논문에서는 지연 조건에서의 지하철 승객 흐름 예측을 위한 새로운 프레임워크를 제안합니다. 특히, 기존 모델들이 지연 이벤트의 복잡한 영향력을 효과적으로 포착하지 못하는 문제를 해결하기 위해 대형 언어 모델(LLMs)을 활용합니다.

- **Technical Details**: 제안된 프레임워크는 지연 이벤트 정보와 역사적인 승객 흐름 데이터의 패턴을 이해하도록 LLM을 도와주는 프롬프트 엔지니어링(prompt engineering)이 포함되어 있습니다. 이 프레임워크는 두 단계로 나뉘며, 첫 번째는 다중 원천 데이터(multi-source data)를 변환하여 LLM이 이해할 수 있는 설명 텍스트로 저장하는 체계적 프롬프트 생성(systematic prompt generation) 단계이고, 두 번째는 다차원적 사고(chain of thought, CoT) 방법을 사용하여 프롬프트를 정제하는 프롬프트 정제 단계입니다.

- **Performance Highlights**: 실험 결과, 제안된 모델이 지연 조건에서의 승객 흐름 예측에서 특히 높은 성능을 보였으며, 중국 선전 지하철의 실제 데이터셋을 사용하여 검증되었습니다. 이 연구는 비상 대응(emergency response) 및 서비스 복구(service recovery)와 관련하여 매우 중요한 정보를 제공합니다.



### GDPO: Learning to Directly Align Language Models with Diversity Using GFlowNets (https://arxiv.org/abs/2410.15096)
- **What's New**: 이 논문은 언어 모델의 행동을 인간의 요구와 가치에 맞추기 위한 선호 정렬(preference alignment)의 새로운 접근법으로 GFlowNet-Direct Preference Optimization (GDPO)라는 방법을 제안합니다. 이는 기존의 Direct Preference Optimization (DPO)의 한계인 편향(overfitting) 문제를 해결하고자 합니다.

- **Technical Details**: GDPO는 GFlowNet을 활용하여 오프라인 선호 데이터로부터 직관적인 보상 신호를 추출하는 방식으로 정책을 학습합니다. 이 과정에서 모델의 입력 및 출력 간의 Markov Decision Process (MDP)를 정의하였으며, GFlowNet의 흐름(flow) 개념을 통해 다양한 응답을 생성합니다.

- **Performance Highlights**: 실험 결과 GDPO는 대화 생성 및 요약 작업에서 기존 방법들에 비해 훨씬 더 다양한 응답을 생성하는 동시에 여전히 인간의 선호에 맞춰진 결과를 보였습니다.



### Towards Safer Heuristics With XPlain (https://arxiv.org/abs/2410.15086)
- **What's New**: 이 논문에서는 cloud 운영자가 컴퓨팅 성능이 저하되는 경우를 더 깊게 이해할 수 있도록 도와주는 새로운 도구인 XPlain을 제안합니다. 기존의 heuristic 분석 도구들은 성능 저하가 발생하는 입력 인스턴스만을 찾아내지만, XPlain은 그 이유와 범위를 분석하여 문제 해결에 도움을 제공합니다.

- **Technical Details**: XPlain은 도메인 특정 언어를 사용하여 분석할 heuristics와 비교할 벤치마크를 매개변수로 활용합니다. 이 언어는 네트워크 흐름 추상화에 뿌리를 두고 있으며, 이를 통해 운영자들이 사용하는 다양한 heuristics의 동작을 모델링할 수 있습니다. XPlain의 컴파일러는 이 언어의 입력을 기존 heuristic 분석기로 변환하고, 효율적인 반복 알고리즘을 통해 성능 저하의 모든 원인과 그 지역을 찾아냅니다.

- **Performance Highlights**: 이 논문의 초기 결과는 XPlain이 existing heuristic analyzers의 한계를 극복할 수 있는 가능성을 보여줍니다. 특히, heuristic의 성능 저하 이유를 명확히 하고, 그 결과를 시각화하여 보다 나은 이해를 돕는데 기여할 수 있습니다.



### EPT-1.5 Technical Repor (https://arxiv.org/abs/2410.15076)
- **What's New**: 새로운 EPT-1.5 모델은 기존의 EPT-1보다 현저한 성능 향상을 보여주며, 특히 유럽의 에너지 산업에 특화된 예측 능력을 자랑합니다. 또한, EPT-1.5는 최신 AI 날씨 모델들과 비교했을 때 특히 바람 속도 및 태양 복사량 예측에서 탁월한 성능을 기록하여 새로운 최고 성능 기준을 설정하였습니다.

- **Technical Details**: EPT-1.5는 Jua의 EPT-1 아키텍처를 기반으로 한 AI 기상 예측 모델로, 수십억 개의 매개변수를 통해 복잡한 기상 패턴을 포착할 수 있습니다. 이 모델은 5 페타바이트의 날씨 데이터로 훈련되어 있으며, 0.083도(약 9x9 km)의 공간 해상도에서 1시간 간격으로 최대 20일 후까지의 예측을 수행합니다.

- **Performance Highlights**: EPT-1.5는 바람 예측에서 기존의 AI 모델 및 유럽 중기 기상 예보 센터(ECMWF)의 IFS HRES를 초과하는 성능을 보여줍니다. 이 모델은 예측의 정확성을 높이기 위해 다양한 파인튜닝 기법을 사용하고, 확률적 예측 기능을 통해 여러 가능한 미래 시나리오를 예측할 수 있는 능력을 갖추고 있습니다.



### A Prompt Engineering Approach and a Knowledge Graph based Framework for Tackling Legal Implications of Large Language Model Answers (https://arxiv.org/abs/2410.15064)
Comments:
          27 pages, 2 figures

- **What's New**: 최근 대형 언어 모델(Large Language Models, LLMs)의 인기로 인해 사용자들이 응답 정보에 맹목적으로 의존하게 되는 위험이 증가하고 있습니다. 이 연구에서는 LLM의 법적 맥락을 고려하지 않은 추천이 사용자에게 잠재적 위험을 초래할 수 있음을 실증적으로 분석하고, 이러한 문제를 해결하기 위한 프롬프트 리엔지니어링(prompt re-engineering) 접근 방식을 제안합니다.

- **Technical Details**: 본 연구에서는 LLM의 안전성을 높이기 위해 프롬프트 리엔지니어링 기법을 활용하여 사용자가 프롬프트를 정교화할 수 있는 새로운 접근법을 제시합니다. 또한, 법적 지식 그래프(Legal Knowledge Graph)를 활용하여 LLM이 제공하는 응답에서 법적 인용을 생성하는 프레임워크를 제안하며, LLM의 응답에 관련된 법적 함의를 사용자에게 명확히 표시합니다.

- **Performance Highlights**: 실험을 통해 기존 LLM은 프롬프트 리엔지니어링 접근 방식을 사용하여 법적 함의를 강조할 수 있음을 보여주었습니다. 하지만, 현재의 접근 방식은 특정한 법적 조항이나 법률 정보를 제공하는 데 한계가 있으며, 이를 보완하기 위해 KGs를 활용한 통합 프레임워크가 필요하다는 점을 강조합니다.



### A Dual-Fusion Cognitive Diagnosis Framework for Open Student Learning Environments (https://arxiv.org/abs/2410.15054)
- **What's New**: 이 논문은 학생들의 역량을 추론하기 위한 새로운 접근 방식인 이중 융합 인지 진단 프레임워크(DFCD)를 제안합니다. DFCD는 기존의 인지 진단 모델(CDM)이 새로운 학생, 연습문제 및 개념을 처리할 수 있도록 하며, 재훈련 없이도 효과적으로 작동할 수 있습니다.

- **Technical Details**: DFCD는 두 가지 서로 다른 모달리티, 즉 텍스트 의미적 특징과 응답 관련 특징을 정렬하는 것을 목표로 합니다. DFCD는 연습 문제와 개념을 대형 언어 모델을 통해 정제하는 연습 정제기(exercise-refiner)와 개념 정제기(concept-refiner)를 제안합니다. 또한, DFCD는 응답 로그의 정보를 완벽하게 통합하기 위해 새로운 응답 매트릭스(response matrix)를 제공합니다. 마지막으로, 두 가지 모달 특징을 통합하는 이중 융합 모듈이 설계되었습니다. 이 프레임워크는 실시간 학습 환경에서도 활용이 가능하다는 장점이 있습니다.

- **Performance Highlights**: 실제 데이터셋에 대한 광범위한 실험 결과, DFCD는 서로 다른 모달의 표현을 통합함으로써 뛰어난 성능을 달성하며, 동적인 오픈 학생 학습 환경에서 강한 적응력을 보여줍니다.



### Mining Glitch Tokens in Large Language Models via Gradient-based Discrete Optimization (https://arxiv.org/abs/2410.15052)
- **What's New**: GlitchMiner는 LLM에서 glitch token을 효율적으로 탐지하기 위한 새로운 gradient-based discrete optimization framework입니다. 이 방법은 기존 방식의 수동 관찰 의존성을 극복하고, entropy 기반의 손실 함수 및 첫 번째 차수 Taylor 근사를 결합하여 token space를 탐색합니다.

- **Technical Details**: GlitchMiner는 첫 번째 단계로 initialization을 거쳐 mining 단계를 통해 glitch token을 탐색합니다. entropy를 손실 함수로 사용하여 모델 예측의 불확실성을 측정하고, local search 전략을 통해 token을 효과적으로 탐색합니다. 이 방식은 모델 아키텍처에 대한 규정된 분포를 필요로 하지 않으며, 각 token의 정상 분포에서의 이탈 정도를 파악하여 기하학적으로 유망한 후보를 선택합니다.

- **Performance Highlights**: GlitchMiner는 다양한 주류 LLM 아키텍처에서 실험을 통해 기존 방법보다 평균 19.07% 향상된 precision@1000을 기록했습니다. 이로 인해 LLM에서 발생할 수 있는 잠재적 취약성을 평가하고 완화하는 데 유용한 도구로 자리잡게 되었습니다.



### MorphAgent: Empowering Agents through Self-Evolving Profiles and Decentralized Collaboration (https://arxiv.org/abs/2410.15048)
- **What's New**: MorphAgent는 기존의 설정된 역할과 중앙 집중형 조정 방식에 의존하지 않고, 에이전트가 역할과 능력을 동적으로 진화시킬 수 있는 분산형 다중 에이전트 협업 프레임워크입니다. 이는 변화하는 요구 사항에 더 잘 적응할 수 있도록 설계되었습니다.

- **Technical Details**: MorphAgent 프레임워크는 두 단계로 진행됩니다: 초기 프로파일 최적화를 위한 워밍업 단계와 작업 피드백에 따라 에이전트가 역할을 지속적으로 조정하는 작업 실행 단계입니다. 시스템은 에이전트 프로파일을 최적화하기 위해 Role Clarity Score, Role Differentiation Score, Task-Role Alignment Score의 세 가지 주요 메트릭을 사용합니다.

- **Performance Highlights**: 실험 결과, MorphAgent는 전통적인 고정 역할 기반의 다중 에이전트 시스템(MAS)보다 작업 성능 및 변화하는 요구 사항에 대한 적응성에서 더 나은 성과를 보였습니다. 이 연구는 다중 에이전트 협업 시스템의 Robustness와 Versatility를 위한 새로운 길을 열었습니다.



### Retrieval Augmented Diffusion Model for Structure-informed Antibody Design and Optimization (https://arxiv.org/abs/2410.15040)
- **What's New**: 이 논문에서는 구조적 동질 접힘 (structural homologous motifs)에 기초하여 항체 설계를 위한 향상된 모델을 제안합니다. 새로운 접근 방식으로, 기존의 실험 방법 대신 데이터 기반의 생성 모델을 활용하여 효율적으로 항체를 설계하는 RADAb 프레임워크를 소개합니다.

- **Technical Details**: RADAb 모델은 두 가지 주요 프로세스를 통합합니다: (1) 구조적 동질 모티프를 활용한 정보 검색 및 (2) 조건부 확산 모델을 통한 반복 최적화입니다. 이 모델은 CDR(Complementarity Determining Regions)의 진화 정보를 포함하여 항체의 구조적 제약을 준수하는 방식으로 설계되었으며, 이를 통해 기존의 불충분한 데이터로 인한 과적합 문제를 완화합니다.

- **Performance Highlights**: 본 연구에서 제안한 RADAb 모델은 여러 항체 반전 접힘(antibody inverse folding) 작업에서 최첨단 성능을 달성했으며, 긴 CDRH3 반전 접힘 작업에서 8.08%의 AAR(average accuracy rate) 개선과 기능성 최적화 작업에서 평균 7 cal/mol의 ΔΔG 개선을 보여줍니다. 이러한 성과는 바이오분자 생성 모델에서 혁신적인 관점을 제시합니다.



### AutoFPDesigner: Automated Flight Procedure Design Based on Multi-Agent Large Language Mod (https://arxiv.org/abs/2410.14989)
Comments:
          21 pages, 18 figures, 5 tables

- **What's New**: 현재의 비행 절차 설계 방법은 인간 주도의 설계 프로세스에 크게 의존하고 있으며, 이로 인해 자동화가 낮고 복잡한 알고리즘 모델링과 낮은 일반화 능력을 겪고 있습니다. 이 논문에서는 대형 언어 모델(large language model)을 기반으로 한 에이전트 기반 비행 절차 설계 방법인 AutoFPDesigner를 제안합니다.

- **Technical Details**: AutoFPDesigner는 다중 에이전트 협업(multi-agent collaboration)을 통해 절차 설계를 완료하며, 사용자가 자연어로 설계 요구사항을 입력하면, AutoFPDesigner는 설계 사양을 로드하고 도구 라이브러리를 활용하여 비행 절차 설계를 모델링합니다. 사용자는 설계 과정에서 감독하고 원활하게 참여할 수 있습니다.

- **Performance Highlights**: 실험 결과에 따르면, AutoFPDesigner는 설계된 비행 절차에서 거의 100%의 안전성을 보장하며, 75%의 작업 완료율(task completion rate)을 달성하였고, 다양한 설계 작업에 대한良好한 적응성을 보였습니다.



### Do Large Language Models Truly Grasp Mathematics? An Empirical Exploration (https://arxiv.org/abs/2410.14979)
- **What's New**: 이 연구는 최근 LLM(대형 언어 모델)의 수학적 추론 능력의 기초 메커니즘에 대해 토대로 하고 있으며, CoT(Chain-of-Thought) 프로프트의 효과를 조명합니다. LLM이 일반적으로 인지적 반사를 평가하는 과제에서 높은 오류율을 보인 것을 확인하였습니다.

- **Technical Details**: 연구팀은 CoT 프로프트를 사용하여 LLM이 인지적 반사 테스트(Cognitive Reflection Test, CRT)에서 오류를 줄일 수 있는지를 조사했습니다. CRT 문제를 수정하여 LLM의 수학적 사고 능력을 실험적으로 평가하였으며, 그 결과 LLM은 여전히 높은 오류율을 유지했습니다. 데이터 세트에 대한 수정 유형 A, B, C, D의 정확도를 분석하였습니다.

- **Performance Highlights**: 각 종류별 문제의 성능 분석 결과, 수정된 문제에 대한 LLM의 성능은 기존보다 낮았으며, 특히 타입 D 문제에서는 GPT-4 모델조차도 평균 정확도가 29.33%에 불과했습니다. 일반적인 LLM은 계산 단계에서의 오류가 지배적이었으며, 초점은 시스템 2(논리적 추론) 능력 부족에 있음을 암시합니다.



### BrainECHO: Semantic Brain Signal Decoding through Vector-Quantized Spectrogram Reconstruction for Whisper-Enhanced Text Generation (https://arxiv.org/abs/2410.14971)
- **What's New**: 최근 EEG 및 MEG 신호로부터 언어를 해독하는 기술이 발전하며, BrainECHO라는 새로운 다단계 전략이 제안되었습니다. BrainECHO는 전이 학습된 언어 모델을 활용하여 텍스트 생성 성능을 혁신적으로 향상시킵니다.

- **Technical Details**: BrainECHO는 1) 오디오 스펙트로그램의 이산 자동 인코딩, 2) 뇌-오디오 잠재 공간 정렬, 3) Whisper 모델의 파인 튜닝을 통한 의미론적 텍스트 생성을 포함하는 3단계 과정으로 구성되어 있습니다. 이 과정을 통해 BrainECHO는 EEG 및 MEG 데이터셋에서 최첨단 성능을 달성합니다.

- **Performance Highlights**: BrainECHO는 기존 방법들보다 향상된 성능을 보이며, 언어 기반 뇌-컴퓨터 인터페이스(BCI)에 중요한 진전을 제공합니다. 특히, 문장, 세션 및 주제 간 독립적인 평가에서 강력한 견고성을 демонстр합니다.



### LSS-SKAN: Efficient Kolmogorov-Arnold Networks based on Single-Parameterized Function (https://arxiv.org/abs/2410.14951)
Comments:
          25 pages, 14 figures, experiment codes are available at this https URL , and SKAN's Python library code are available at this https URL

- **What's New**: 최근 제안된 Kolmogorov-Arnold Networks (KAN) 네트워크는 MLP에 비해 높은 시각화 가능성 덕분에 많은 주목을 받고 있습니다. 본 논문에서는 네트워크 규모를 확장하기 위한 매개변수 할당(과거에 비해 더 복잡한 기초 함수를 사용하는 것보다)이 KAN 성능 개선에 더 효율적이라는 Efficient KAN Expansion Principle (EKE Principle)을 제안합니다.

- **Technical Details**: 우리는 단일 학습 가능한 매개변수를 활용하는 기초 함수로 구성된 새로운 KAN 디자인 원칙, 즉 SKAN을 제안합니다. 여기서 LShifted Softplus 기반의 SKAN (LSS-SKAN)은 최고의 정확도를 보여주었습니다. 여러 단일 매개변수함수를 평가한 결과, LSS-SKAN이 가장 유용하다는 것을 입증했습니다.

- **Performance Highlights**: LSS-SKAN은 MNIST 데이터 셋에서 모든 테스트된 KAN 변형에 비해 우수한 성과를 보였으며 정확도 측면에서 FourierKAN, Spl-KAN, FastKAN, Wav-KAN보다 각각 0.58%, 1.65%, 2.57%, 0.22% 더 높은 값을 기록했습니다. 실행 속도에서도 LSS-SKAN은 모든 비교된 KAN 변형보다 빠른 속도를 보였습니다.



### Soft-Label Integration for Robust Toxicity Classification (https://arxiv.org/abs/2410.14894)
Comments:
          Accepted by Neurips 24

- **What's New**: 이번 연구는 독창적인 bi-level optimization 프레임워크를 제안하여 crowdsourced annotations와 soft-labeling 기법을 통합하고, Group Distributionally Robust Optimization (GroupDRO)를 통해 soft-label 가중치를 최적화하여 OOD(Out-of-Distribution) 위험에 대한 강건성을 향상시킵니다.

- **Technical Details**: 이 프레임워크는 두 개의 최적화 루프로 구성됩니다: 내부 루프는 학습된 soft labels로 훈련 샘플의 ERM 손실을 최소화하고, 외부 루프는 spurious features에 대한 모델의 의존성을 평가하고 soft-label 가중치를 최적화합니다. 이론적으로 bi-level 최적화 알고리즘의 수렴성을 증명합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 기존의 기준 방법들보다 평균 및 최악 그룹 정확도 모두에서 우수성을 보이며, GPT-4 Turbo보다 높은 정확도를 기록했습니다. 이를 통해 crowdsourced annotations를 활용하여 보다 효과적이고 강건한 독성 분류를 실현할 수 있음을 확인했습니다.



### Reasoning, Memorization, and Fine-Tuning Language Models for Non-Cooperative Games (https://arxiv.org/abs/2410.14890)
- **What's New**: 이번 연구에서는 사전 훈련된 언어 모델의 복잡한 게임 해결 능력을 향상시키기 위한 새로운 방법을 제안합니다. 이 방법은 게임 해결을 네 개의 점진적인 작업으로 분해하여 각 작업을 특정 언어 모델 에이전트에 할당합니다. 이는 에이전트들이 협력하여 게임 상황과 전략을 간략화할 수 있도록 돕습니다.

- **Technical Details**: 제안된 방법은 tree of thoughts와 multi-agent framework를 통합하고, 게임 요약, 영역 선택, 행동 추출 및 행동 검증의 네 가지 작업으로 구성됩니다. 각 언어 모델 에이전트는 특정 작업을 수행하며, 이들은 서로 질문을 통해 정보를 주고받습니다.

- **Performance Highlights**: 제안된 방법은 비협력 게임에 적용되어 기준 알고리즘에 대해 65%의 승률을 달성하였으며, 자동 미세 조정 후 승률이 추가로 10% 증가했습니다. 기존의 딥러닝 알고리즘이 수백만 개의 훈련 샘플을 필요로 하는 반면, 이 방법은 약 1000개의 샘플로 동일한 성과를 달성하여 효율성과 확장성을 강조합니다.



### Class-RAG: Content Moderation with Retrieval Augmented Generation (https://arxiv.org/abs/2410.14881)
Comments:
          11 pages, submit to ACL

- **What's New**: 이번 연구에서는 콘텐츠 모더레이션(content moderation)을 위한 새로운 분류 방법인 Retrieval-Augmented Generation (Class-RAG)을 제안합니다. 이는 안전하지 않은 입력과 안전한 입력 사이의 미세한 차이를 극복하여 더 나은 분류 성능을 제공합니다.

- **Technical Details**: Class-RAG 시스템은 임베딩 모델(embedding model), 검색 라이브러리(retrieval library), 검색 모듈(retrieval module), 그리고 미세 조정된 LLM(Classifier)으로 구성됩니다. 사용자가 쿼리를 입력하면, 가장 유사한 부정적 및 긍정적 예제를 검색하여 해당 컨텍스트 정보를 분류기(classifier)에 추가합니다.

- **Performance Highlights**: Class-RAG는 전통적인 모델에 비해 분류 성능이 우수하며, 적대적 공격(adversarial attack)에 강한 내성을 보여줍니다. 또한, 검색 라이브러리의 크기를 확장하면 성능이 향상되며, 이는 낮은 비용으로 분류 성능을 높일 수 있는 실용적인 방법으로 제시됩니다.



### Joint Verification and Refinement of Language Models for Safety-Constrained Planning (https://arxiv.org/abs/2410.14865)
- **What's New**: 이 연구는 사전 훈련된 언어 모델을 사용하여 로봇 작업을 위한 실행 가능한 계획을 생성하고, 이를 형식적으로 검증하는 방법을 제안합니다. 이는 언어 모델의 결과가 고유한 논리적 명세를 충족하는지 확인하기 위한 것입니다.

- **Technical Details**: 제안된 방법은 자연어로 된 고수준 작업 설명에서 언어 모델에 쿼리를 수행하여 실행 가능한 로봇 프로그램 형태의 계획을 생성합니다. 생성된 계획은 오토마타 기반 표현으로 변환되어, 명세에 대한 형식적 검증을 수행할 수 있습니다. 이 과정에서 특정 정리(thorem)를 통해 여러 소 계획의 조합이 안전성을 유지함을 보장합니다.

- **Performance Highlights**: fine-tuning 과정을 통해 생성된 계획의 명세 만족 확률이 30% 증가하였으며, 이는 언어 모델이 인간 레이블 없이도 계획이 명세를 준수하도록 개선할 수 있음을 보여줍니다.



### The S2 Hierarchical Discrete Global Grid as a Nexus for Data Representation, Integration, and Querying Across Geospatial Knowledge Graphs (https://arxiv.org/abs/2410.14808)
- **What's New**: 본 논문에서는 Geospatial Knowledge Graphs (GeoKGs)가 Geospatial Artificial Intelligence에서 중요한 역할을 하고 있으며, KnowWhereGraph라는 새로운 GeoKG 구현을 소개합니다. 이 Graph은 Google's S2 Geometry를 활용하여 다양한 데이터 출처에서 효율적인 데이터 처리를 가능하게 하며, 논문은 이 시스템의 구현 방법과 그 중요성을 강조합니다.

- **Technical Details**: KnowWhereGraph는 Discrete Global Grid Systems (DGGS) 중 하나인 S2 Geometry를 사용하여 데이터를 통합하고 표현합니다. 이는 복잡한 topology relations를 탐색하고 데이터의 semantic compression을 제공하여 데이터 관리의 복잡성을 줄입니다. GEospatial 데이터의 통합, 저장, 분석 방식에서 혁신적인 접근법을 제공합니다.

- **Performance Highlights**: KnowWhereGraph는 데이터를 효율적으로 처리하여 지리적 쿼리를 다루는 데 필요한 계산 복잡성을 줄이고, 다양한 스케일과 다양한 형식의 데이터를 통합할 수 있는 가능성을 보여줍니다. 결론적으로, DGGS 프레임워크, 특히 S2의 잠재력은 확장 가능한 GeoKG를 구축하는 데 큰 기여를 할 것으로 기대됩니다.



### TimeSeriesExam: A time series understanding exam (https://arxiv.org/abs/2410.14752)
Comments:
          Accepted at NeurIPS'24 Time Series in the Age of Large Models Workshop

- **What's New**: 이번 논문에서는 대형 언어 모델(Large Language Models, LLMs)의 시간 시계열 데이터 이해도를 평가하기 위해 TimeSeriesExam이라는 새로운 시험 시스템을 도입합니다. 이 시스템은 5개의 핵심 시간 시계열 이해 카테고리(패턴 인식, 잡음 이해, 유사성 분석, 이상 탐지, 인과 관계 분석)를 평가하는 700개 이상의 객관식 질문으로 구성되어 있습니다.

- **Technical Details**: TimeSeriesExam은 104개의 면밀하게 설계된 템플릿을 사용하여 절차적으로 생성된 질문을 기반으로 하며, 각 질문은 Item Response Theory (IRT)를 통해 난이도 조정 및 모델 차별화를 위해 세밀하게 조정됩니다. 시험은 LLM의 기본 시간 시계열 개념 이해도를 평가하며, 특히 패턴과 노이즈 개념, 이상 탐지 및 인과 관계 분석과 같은 다양한 추론 작업을 포함합니다.

- **Performance Highlights**: 시험 결과, GPT-4 및 Gemini와 같은 폐쇄형 모델이 개방형 대안보다 기본 시간 시계열 개념에 대해 이해도가 더 높다는 사실이 드러났습니다. 그러나 모든 모델은 인과 관계 분석과 같은 복잡한 개념에서는 어려움을 겪었습니다.



### Toward a Unified Graph-Based Representation of Medical Data for Precision Oncology Medicin (https://arxiv.org/abs/2410.14739)
Comments:
          19 pages, 1 figure, 14 tables, CIBB 2024 conference

- **What's New**: 이 논문에서는 유전자 정보와 환자의 의료 기록을 독특한 지식 그래프(knowledge graph)로 결합한 새로운 통합 그래프 기반의 의료 데이터 표현 방식을 제안합니다. 이 접근법을 통해 각 데이터 세트를 별도로 살펴보았을 때는 불가능했을 중요한 정보와 설명을 추론할 수 있습니다.

- **Technical Details**: 연구는 여러 데이터베이스를 동시에 활용하여, 환자의 유전자 정보와 의료 기록 및 의료 지식을 하나의 지식 그래프에 통합해 표현합니다. 이 지식 그래프는 환자 특성, 유전자 돌연변이와 질병, 치료 효과성을 연결하여 연구하기 위한 새로운 통찰력을 제공합니다.

- **Performance Highlights**: 초기 실험 결과, 전통적인 컴퓨터 과학의 그래프 알고리즘을 활용하여 일부 의료 문제를 모델링하고 해결할 수 있는 가능성을 보여주었습니다. 이러한 방법론은 온콜로지(oncology) 및 개인화된 의학에서 약물 치료의 효과성을 향상시키는 데 기여할 수 있는 잠재력을 지니고 있습니다.



### The Representation of Meaningful Precision, and Accuracy (https://arxiv.org/abs/2410.14721)
Comments:
          16 Pages

- **What's New**: 이 논문은 개념적으로 정밀도(precision)와 정확도(accuracy)의 중요성을 조명하며, 머신 러닝(Machine Learning)과 통계적 학습에서의 한계를 극복하기 위해 전반적인 러프 세트(rough sets) 기반의 프레임워크를 제안합니다. 이 프레임워크는 다양한 문제 상황에서 적합할 정도로 일반화되어 있으며, 컴퓨터 도구의 발전에 힘입어 적용 가능성이 높습니다.

- **Technical Details**: 정확도와 정밀도는 측정 이론(Measurement Theory)에서 관측 오차(Observational Error)의 수치적 척도로 정의됩니다. 정확도는 측정값이 실제 값과 얼마나 가까운지를 나타내고, 정밀도는 반복 측정값 간의 유사성을 나타냅니다. 이 연구에서는 일반적인 러프 세트를 활용하여 이러한 개념들을 더 세분화된 조합적 관점에서 분석하고, 가능한 수치적 측정의 유효성 혹은 의미를 결정하는 방법론을 제시합니다.

- **Performance Highlights**: 이 연구에서 제안된 프레임워크는 기존의 정밀도와 정확도의 개념을 적절히 확장할 수 있으며, 다양한 머신 러닝 및 AI 문제들을 해결하는 데 있어 이론적 기반을 제공합니다. 또한, 이 방법론은 적은 간섭(minimal intrusion)으로 타당한 이론적 프레임을 구축하는 데 중점을 두고 있습니다.



### Polymath: A Challenging Multi-modal Mathematical Reasoning Benchmark (https://arxiv.org/abs/2410.14702)
Comments:
          49 pages, (10 pages paper, 9 pages references, 30 pages appendix)

- **What's New**: Multi-modal Large Language Models (MLLMs)에 대한 새로운 벤치마크인 PolyMATH가 제시되었으며, 이는 인지적 추론 능력을 평가하기 위한 도전적인 기준입니다.

- **Technical Details**: PolyMATH는 10가지 범주에서 집합한 5,000개의 고품질 이미지를 포함하며, 패턴 인식, 공간 추론, 상대적 추론을 포괄합니다. 15개의 MLLM을 네 가지 다양한 프로토타이핑 전략을 사용하여 종합적이고 정량적으로 평가했습니다.

- **Performance Highlights**: Claude-3.5 Sonnet은 약 41%, GPT-4o는 약 36%, Gemini-1.5 Pro는 약 27%의 점수를 기록했습니다. 이 모델들은 공간 관계 이해 및 고수준 추론 수행에서 어려움을 겪고 있음이 드러났습니다.



### Influence of Backdoor Paths on Causal Link Prediction (https://arxiv.org/abs/2410.14680)
- **What's New**: 본 논문은 지식 그래프에서 인과 관계 예측 시 발생하는 혼란 변수(confounder)의 영향을 차단하는 새로운 접근법, CausalLPBack을 제안합니다.

- **Technical Details**: CausalLPBack은 비인과적 연관 흐름인 backdoor path를 제거하여 인과 관계 예측의 정확성을 높이는 방법을 사용합니다. 이 방법은 신경-상징적(neuro-symbolic) 프레임워크를 통해 전통적인 인과 AI 개념을 활용할 수 있게 확장됩니다.

- **Performance Highlights**: 제안된 방법은 실험적으로 30% 이상의 MRR(Mean Reciprocal Rank)과 16% 이상의 Hits@K 성능 향상을 보여주었으며, 이는 혼란 변수로 인해 발생하는 편향에 기인합니다.



### HyperCausalLP: Causal Link Prediction using Hyper-Relational Knowledge Graph (https://arxiv.org/abs/2410.14679)
Comments:
          arXiv admin note: text overlap with arXiv:2405.02327

- **What's New**: 본 논문에서는 중재자(mediator) 링크를 활용하여 불완전한 인과 네트워크内에서 누락된 인과 링크를 찾기 위한 HyperCausalLP 접근법을 제안합니다. 이는 하이퍼 관계(hyper-relational) 지식 그래프(completion)으로 누락된 링크 문제를 형식화하고, 하이퍼 관계 지식 그래프 내의 정보를 통해 인과 링크 예측을 개선합니다.

- **Technical Details**: HyperCausalLP는 중재자 링크를 통합하여 인과 링크 예측을 개선하는 새로운 접근법을 제안합니다. 이 방법은 하이퍼 관계 인과 지식 그래프(CausalKG)를 사용하여 복합 인과 관계를 표현하고, StarE 알고리즘을 통해 KG 임베딩(KGE) 모델로 변환됩니다. 이 논문은 CLEVRER-Humans라는 인과 벤치마크 데이터셋을 통해 이 접근법을 평가합니다.

- **Performance Highlights**: 모델의 결과는 하이퍼 관계 지식 그래프를 사용한 인과 링크 예측이 평균 5.94%의 향상을 보였음을 보여주며, 중재자에 대한 정보를 통합하는 것이 성능 개선에 기여함을 입증합니다.



### xGen-MM-Vid (BLIP-3-Video): You Only Need 32 Tokens to Represent a Video Even in VLMs (https://arxiv.org/abs/2410.16267)
- **What's New**: xGen-MM-Vid (BLIP-3-Video)은 비디오를 위한 멀티모달 언어 모델로, 여러 프레임에서의 시간 정보를 효율적으로 캡처하도록 설계되었습니다. 이 모델은 일반적인 비주얼 토크나이저(visual tokenizer) 외에 'temporal encoder'를 도입하여 훨씬 적은 수의 visual tokens를 사용하여 비디오 질문-응답에서 높은 정확도를 기록합니다.

- **Technical Details**: BLIP-3-Video는 4개의 주요 구성 요소로 이루어져 있습니다: (1) 각 프레임 입력을 처리하는 비전 인코더(vision encoder), (2) 토큰 수를 줄이는 프레임 레벨 토크나이저(frame-level tokenizer), (3) 비디오 수준의 토큰 표현을 구축하는 템포럴 인코더(temporal encoder), (4) 비디오 토큰과 텍스트 프롬프트 토큰에 기반하여 출력 텍스트 캡션을 생성하는 자동 회귀 LLM입니다. 이 모델은 8개의 샘플링된 프레임을 사용하여 계산 효율성을 높입니다.

- **Performance Highlights**: BLIP-3-Video는 34B의 거대 모델과 비교하여 비슷한 질문-응답 정확도를 보이며, 단 4B로도 성능을 발휘합니다. 각각 16에서 32개의 비디오 토큰을 추상화하여 전체 비디오를 성공적으로 표현할 수 있습니다.



### 3DGS-Enhancer: Enhancing Unbounded 3D Gaussian Splatting with View-consistent 2D Diffusion Priors (https://arxiv.org/abs/2410.16266)
Comments:
          Accepted by NeurIPS 2024 Spotlight

- **What's New**: 본 논문은 3D Gaussian splatting (3DGS) 모델의 렌더링 성능을 크게 개선하기 위한 새로운 파이프라인인 3DGS-Enhancer를 제안합니다. 이 방법은 2D 비디오 디퓨전 모델을 활용하여 3D 뷰 일관성 문제를 해결하고, 이를 통해 고화질의 뷰 일관성을 유지하는 이미지를 복원합니다.

- **Technical Details**: 3DGS-Enhancer는 이미지 인코더, 비디오 기반의 디퓨전 모델, 그리고 공간-시간 디코더로 구성되어 있으며, 이는 렌더링된 뷰의 잠재 특징을 인코딩하고, 시간적으로 일관된 잠재 특징을 복원하며, 원래 렌더링된 이미지와 결합합니다. 이를 통해 초기 3DGS 모델을 파인튜닝하여 렌더링 성능을 향상시킵니다.

- **Performance Highlights**: 대규모 데이터셋에서 실험을 수행한 결과, 3DGS-Enhancer는 다양한 도전적인 장면에서 훌륭한 재구성 성능을 보여주며, 기존의 최신 방법들에 비해 더욱 뚜렷하고 생생한 렌더링 결과를 생성합니다.



### CompassJudger-1: All-in-one Judge Model Helps Model Evaluation and Evolution (https://arxiv.org/abs/2410.16256)
Comments:
          Technical Report, Code and Models: this https URL

- **What's New**: 이번 논문에서는 첫 번째 오픈 소스 \textbf{all-in-one} judge LLM인 \textbf{CompassJudger-1}을 소개합니다. 이 모델은 다양한 평가 작업을 수행할 수 있는 고급 기능을 갖추고 있으며, 주관적 평가의 효율성과 정확성을 향상시키는 데 중점을 둡니다.

- **Technical Details**: CompassJudger-1은 단일 스코어링, 두 모델 비교, 다양한 형식에 따른 평가 수행, 비판 생성, 일반 LLM처럼 다양한 작업 실행 등의 기능을 갖추고 있습니다. 또한, 새로운 벤치마크인 \textbf{JudgerBench}를 통해 여러 주관적 평가 작업을 통합하여 평가 모델들의 성능을 검증합니다.

- **Performance Highlights**: CompassJudger-1은 다양한 주관적 평가 작업을 수행하는 데 효과적이며, 연구 커뮤니티에 공개되어 협업과 LLM 평가 방법론의 발전을 촉진할 수 있는 기반을 제공합니다.



### Sketch2Code: Evaluating Vision-Language Models for Interactive Web Design Prototyping (https://arxiv.org/abs/2410.16232)
Comments:
          preprint, 9 pages

- **What's New**: Sketch2Code라는 새로운 벤치마크를 소개하여 VLMs(비전 언어 모델)가 저해상도 스케치를 웹 페이지 프로토타입으로 변환하는 작업을 자동화할 수 있는 성능을 평가합니다. 특히 이는 UI 디자인의 초기 단계에서 스케치를 사용하여 생성된 프로토타입의 변환을 지원합니다.

- **Technical Details**: Sketch2Code는 VLM의 능력을 평가하기 위해 731개의 고품질 스케치를 수집하고, 다수의 상업적 및 오픈 소스 모델에 대해 실험을 진행했습니다. 이 연구는 VLM 모델이 스케치를 해석하고 생성하는 과정에서의 상호작용 및 사용자의 피드백 수용 능력을 평가하며, '피드백 따르기'와 '질문하기'라는 두 가지 상호작용 시나리오를 설계했습니다.

- **Performance Highlights**: 본 연구에서는 10개 모델(GPT-4o, Gemini 1.5 Pro 등)의 성능을 분석했으며, 기존 VLM 모델들이 스케치를 해석하고 질문을 생성하는 데 어려움을 겪고 있음을 보여줍니다. 사용자 연구 결과, UI/UX 전문가들은 수동적 피드백 수용보다 능동적 질문하기를 선호하며, 이는 다중 턴 대화형 에이전트의 효과적인 발전을 위해 더 깊은 연구의 필요성을 강조합니다.



### Pre-training Distillation for Large Language Models: A Design Space Exploration (https://arxiv.org/abs/2410.16215)
- **What's New**: 이 논문에서는 기존의 post-training distillation(후훈련 지식 증류)에서 벗어나, pre-training distillation(전훈련 지식 증류)이라는 새로운 방법을 제안하여 대형 언어 모델(LLM)에서의 지식 증류를 탐구합니다.

- **Technical Details**: Pre-training distillation(PD)은 teacher model로부터 생성된 logits를 활용하여 student model을 학습하는 방법으로, 다양한 설정을 탐색했습니다. 주요 요소는 logits 처리, 손실 선택, 크기 법칙, 오프라인 또는 온라인 logits 처리입니다.

- **Performance Highlights**: GLM-4-9B를 teacher LLM으로 사용하여 1000 억 개의 토큰에서 1.9B student LLM을 증류한 결과, 평균적으로 1.6%의 성능 향상이 있었습니다. 추가 실험을 통해 PD는 대규모 student LLM에서 더욱 유리한 성과를 보임을 확인했습니다.



### Compute-Constrained Data Selection (https://arxiv.org/abs/2410.16208)
- **What's New**: 이번 연구는 데이터 선택(data selection)을 통해 LLM(finetuning) 훈련에 필요한 데이터 양을 줄이는 방법을 제안하고, 데이터 선택과 훈련 비용을 고려한 최적화 문제를 포괄적으로 분석합니다. 이를 통해 자원 제한이 있는 상황에서도 최적의 모델 성능을 도출할 수 있는 방법을 제시합니다.

- **Technical Details**: 연구는 compute-aware utility function을 통해 데이터 선택 문제를 형식화하고 초기 선택 비용과 훈련 이익 간의 균형을 탐구합니다. 모델 사이즈, 토큰 수, 그리고 데이터 선택 방안 사이의 관계를 포괄적으로 정량화하며, 600개 이상의 모델을 훈련하여 성과를 분석합니다.

- **Performance Highlights**: 복잡한 데이터 선택 방법은 compute-constrained 환경에서 거의 Pareto-optimal하지 않으며, 간단한 통계 기법인 sparse retrieval이 선호되어야 함을 발견했습니다. 이는 이론적 및 경험적 관점에서 FLOP 비효율성에 따른 것입니다. 또한 반복 훈련을 진행하는 환경에서는 이러한 강력한 방법들이 여전히 효과적이게 사용될 수 있음을 강조합니다.



### Information for Conversation Generation: Proposals Utilising Knowledge Graphs (https://arxiv.org/abs/2410.16196)
Comments:
          7 pages with citations, 1 figure, accepted to the ISWC 2024 Special Session

- **What's New**: 이 논문은 대화 생성에 있어 대규모 언어 모델(LLM)을 향상시키기 위한 지식 그래프(KG)를 활용하는 세 가지 제안을 소개합니다.

- **Technical Details**: 첫 번째 제안은 동적 지식 그래프 임베딩(Dynamic Knowledge Graph Embeddings, DKGE)과 추천 시스템을 통해 새로운 정보를 통합하고 관련 지식을 선택하는 방법을 제안합니다. 두 번째로, 감정 값이 부여된 엔티티를 추가적인 특징으로 저장함으로써 사용자 입력과より 감정적으로 연관된 지식을 제공할 수 있습니다. 세 번째 제안은 내러티브 버블을 통해 캐릭터 정보를 통합하여 캐릭터 일관성을 유지하고 새로운 정보를 쉽게 통합할 수 있는 구조를 제시합니다.

- **Performance Highlights**: 이 연구는 대화형 AI의 사용자 경험을 향상시키고 LLM의 감정적 역량을 높이며 캐릭터 일관성을 통해 사용자 수용성을 증가시킬 것이라 기대합니다.



### Warped Diffusion: Solving Video Inverse Problems with Image Diffusion Models (https://arxiv.org/abs/2410.16152)
Comments:
          Accepted in NeurIPS 2024

- **What's New**: 이 논문은 동영상을 생성하는 데 있어 기존의 이미지 모델을 활용하면서 발생할 수 있는 깜박임(flickering), 텍스처 고착(texture-sticking), 시간적 불일치(temporal inconsistency) 문제를 해결하기 위한 새로운 접근 방식을 제안합니다.

- **Technical Details**: 이 연구에서 제안하는 방법은 2D 공간에서 프레임을 연속 함수로 보고 동영상을 서로 다른 프레임 간의 연속 변형 연속(sequence of continuous warping transformations)으로 간주합니다. 이를 통해 기능 공간(diffusion model)에서 훈련된 모델을 사용하여 시간적으로 상관된 역 문제를 해결하도록 합니다. 주요 기술적 요소는 동작 변형에 대한 동등성(equivariance) 요구 사항과, 이를 보장하기 위한 후처리(post-hoc) 방법으로 동등성 자기 안내를 도입하는 것입니다.

- **Performance Highlights**: 제안된 방법은 동영상 인페인팅(video inpainting) 및 8배 영상 초해상도(8× video super-resolution)에서 기존 기술들에 비해 눈에 띄게 우수한 성능을 보이며, 깜빡임과 텍스처 고착과 같은 아티팩트를 줄이는 데 효과적입니다. 논문에서 제시된 결과들은 Stable Diffusion XL과 같은 최신 잠재적 확산 모델(latent diffusion models)에서 더 나은 성능을 보여줍니다.



### Small Contributions, Small Networks: Efficient Neural Network Pruning Based on Relative Importanc (https://arxiv.org/abs/2410.16151)
- **What's New**: 최근의 신경망(Neural Network) 발전은 모델의 크기를 비약적으로 증가시켜 다양한 작업에서 뛰어난 성능을 달성하고 있습니다. 그러나 이러한 대규모 모델을 자원이 제한된 장치에 배포하는 것은 상당한 저장 및 계산 요구사항으로 인해 도전과제를 제시합니다. 본 논문에서는 актив화 통계(activation statistics)에 기반한 직관적이고 해석 가능한 가지치기(pruning) 방법을 소개합니다. 이 방법은 정보 이론(information theory)와 통계 분석(statistical analysis)에 뿌리를 두고 있으며, 신경 활성의 통계적 속성을 활용하여 신경 출력에 최소한의 기여를 하는 가중치를 식별하고 제거합니다.

- **Technical Details**: 본 연구에서 제안하는 데이터 기반(unstructured) 가지치기 방법은 훈련 데이터 또는 그 일부를 활용하여 각 가중치의 중요도 분포를 근사화합니다. 중앙 극한 정리(Central Limit Theorem)에 따라, 가중치의 중요도를 정규 분포(normal distribution)로 모델링하여 가중치와 관련 노드 출력 간의 상호 정보(mutual information)를 추정합니다. 이는 가중치 변동이 노드 출력의 불확실성을 얼마나 줄이는지를 정량화합니다. 또한, 가지치기 인식 훈련(pruning-aware training) 전략을 제안하여 추가 정규화 항을 포함하여 가지치기 방법의 효과를 향상시킵니다.

- **Performance Highlights**: 다양한 데이터 세트와 네트워크 아키텍처에 대한 광범위한 실험을 통해, 제안하는 방법이 여러 기준선(baseline) 및 최첨단(state-of-the-art) 가지치기 기법들보다 일관되게 우수한 성능을 보임을 입증하였습니다. 또한, 현재 방법이 높은 압축 비율에서도 더 정확한 모델을 생성하는데 기여하는 것으로 나타났습니다.



### PODTILE: Facilitating Podcast Episode Browsing with Auto-generated Chapters (https://arxiv.org/abs/2410.16148)
Comments:
          9 pages, 4 figures, CIKM industry track 2024

- **What's New**: 이번 연구에서는 PODTILE이라는 새로운 모델을 소개하여, 팟캐스트 에피소드의 자동 챕터화를 효과적으로 수행합니다. 이는 에피소드의 메타데이터와 이전 챕터 제목을 포함한 구조적 글로벌 컨텍스트를 사용하여 이루어집니다.

- **Technical Details**: PODTILE은 변환기(Transformer) 구조의 인코더-디코더 모델로, 입력 텍스트에 글로벌 컨텍스트를 추가하고, 대화형 데이터의 구성 및 제목을 동시 생성합니다. 평균 약 16,000개의 토큰을 가지는 긴 팟캐스트 전사를 효율적으로 처리할 수 있도록 설계되었습니다.

- **Performance Highlights**: PODTILE은 강력한 기초 모델 대비 ROUGE 점수가 11% 향상되었습니다. 사용 통계에 따르면, 청취자들은 자동 생성된 챕터가 특히 덜 알려진 팟캐스트를 탐색하는 데 유용하다고 보고하였습니다.



### Modeling dynamic neural activity by combining naturalistic video stimuli and stimulus-independent latent factors (https://arxiv.org/abs/2410.16136)
- **What's New**: 이 논문에서는 신경 반응에서의 변동성을 포착하고 예측하기 위해 비디오 입력과 자극 독립적인 잠재 요인을 포함하는 확률적 모델을 제안합니다.

- **Technical Details**: 모델은 자연적인 비디오 자극에 대한 시계열 신경 반응(2-photon calcium traces)을 예측하며, 자극과 잠재 요인에 조건부로 신경 반응의 분포를 Zero-Inflated-Gamma (ZIG) 분포로 모델링합니다.

- **Performance Highlights**: 마우스 V1의 신경 반응 데이터를 기반으로 한 모델 테스트 결과, 비디오 전용 모델보다 로그 우도(log-likelihood)에서 우수하며, 다른 신경의 응답에 조건을 설정했을 때 더 높은 성능을 보였습니다.



### Beyond 2:4: exploring V:N:M sparsity for efficient transformer inference on GPUs (https://arxiv.org/abs/2410.16135)
- **What's New**: 이 논문에서는 2:4 스파시티(2:4 sparsity)의 한계를 극복하기 위해 새로운 V:N:M 스파시티 모델을 제안하고, 이 모델이 비전Transformers와 대규모 언어 모델(LLMs)에서의 정확성과 적용 가능성을 향상 시키는 방법을 탐구합니다.

- **Technical Details**: V:N:M 스파시티는 Transformer의 가중치 행렬을 V×M 크기의 블록으로 나누고, 각 블록 내에서 (M-4)개의 열을 가지치기하여 2:4 스파시티를 구현합니다. 이를 통해 50% 이상의 고밀도 스파시티에서도 실질적인 가속화를 제공할 수 있습니다. 실험 결과, DeiT-small과 같은 모델은 64:2:5 스파시티에서 무손실 정확도를 달성하고, LLama2-7B 모델은 64:2:5 스파시티에서 다운스트림 작업에서 훈련 없는 2:4 스파시티 대안과 비슷하거나 더 나은 성능을 보였습니다.

- **Performance Highlights**: 이 접근법을 통해 V:N:M 스파시티는 2:4 스파시티에 비해 더 넓은 속도-정확도 간의 트레이드오프를 제공합니다. 64:2:8 스파시티에서의 DeiT-base 모델은 밀집 모델에 비해 1.7배 속도 향상을 이루며, 2:4 스파시티 모델은 1.15배 향상된 성능을 보였습니다.



### SeaDAG: Semi-autoregressive Diffusion for Conditional Directed Acyclic Graph Generation (https://arxiv.org/abs/2410.16119)
- **What's New**: SeaDAG라는 새로운 모델을 소개하며, 이는 Directed Acyclic Graphs (DAGs)의 조건적 생성을 위한 반자기회귀적(dihg) 확산 모델입니다. 이 모델은 계층 구조를 활용하며 각 층에서 다른 노이즈 제거 속도를 설계하여 계층별 자가회귀 생성을 모사합니다.

- **Technical Details**: SeaDAG는 DAG의 계층적 구조를 완벽하게 활용하여 동시적으로 모든 층을 진화시킵니다. 이 모델은 각 단계에서 완전한 그래프 구조를 유지하며, 기존 모델들이 다루지 못했던 조건 학습을 명시적으로 통합합니다.

- **Performance Highlights**: SeaDAG는 회로 생성을 위한 진실 테이블 및 양자 특성을 기반으로한 분자 생성의 두 가지 대표적인 조건부 DAG 생성 작업에서 유망한 결과를 보였습니다. 이 모델은 주어진 조건에 매우 잘 부합하는 현실적이고 고품질의 DAG를 생성하는 능력을 보여줍니다.



### Multimodal Flare Forecasting with Deep Learning (https://arxiv.org/abs/2410.16116)
- **What's New**: 이번 연구는 크로모스피어(Chromosphere)와 코로나(Corona)에서의 UV 및 EUV 방출이 플레어(Flare) 예측에서 보여주는 예측력을 포토스피어(Photosphere) 선 시각화 자력도(Magnetograms)와 비교한 데이터 기반 접근 방식을 제시합니다.

- **Technical Details**: 이 연구에서는 딥러닝(Deep Learning) 방법론을 사용하여 크로모스피어 및 코로나의 UV 및 EUV 방출과 포토스피어 자력도에서 추출된 특징의 비교를 진행합니다. 특정 EUV 파장들은 자력도와 동등하거나 더 우수한 구별력을 제공하며, 다중 모달 신경망(Multimodal Neural Networks) 구조가 단일 입력 모델보다 일관되게 성능이 우수함을 보였습니다. 또한, 모델은 전체 디스크 이미지와 포괄적인 플레어 이벤트 카탈로그를 사용하여 훈련되고 평가됩니다.

- **Performance Highlights**: 모델의 성능 개선은 모델 배깅(model bagging) 등의 조합 모델(Ensemble Model) 기법을 통해 이루어졌으며, 이는 서로 다른 모델의 특징 간의 상관관계를 활용하여 예측의 정확성을 높일 수 있는 가능성을 나타냅니다. 결과적으로, 이번 연구는 크로모스피어 및 코로나의 방출 정보를 활용한 플레어 예측의 가능성을 확인시켜 주며, M-Class 임계값 이상에서의 플레어 예측 성능을 더욱 향상시킬 수 있는 기반을 마련했습니다.



### Addressing Spectral Bias of Deep Neural Networks by Multi-Grade Deep Learning (https://arxiv.org/abs/2410.16105)
- **What's New**:  본 연구에서는 Deep Neural Networks (DNNs)의 spectral bias 문제에 대해 다룬 새로운 접근법을 제안합니다. 저주파수 성분을 학습하는 깊은 신경망의 한계 문제를 극복하기 위해 여러 개의 Shallow Neural Networks (SNNs)를 구성하여 고주파수 성분을 효과적으로 학습하는 기법을 소개합니다.

- **Technical Details**:  연구의 핵심 아이디어는 저주파수 성분으로만 이루어진 신경망(SNN)을 여러 개 조합하여 고주파수 성분을 포함한 함수를 근사하는 것입니다. Multi-Grade Deep Learning (MGDL) 모델을 사용하여, 각 단계에서 이전 단계의 잔여로부터 학습된 SNN을 특성으로 활용해 새로운 신경망을 점진적으로 훈련시킵니다.

- **Performance Highlights**:  실험 결과, MGDL은 고주파수 정보를 포함하는 함수의 표현에서 뛰어난 성능을 발휘하였고, 기존의 Single Grade Deep Learning (SGDL) 기법에 비해 근사 정확도를 크게 향상시켰습니다. 이러한 결과는 고주파수 성분을 포함한 문제를 해결하는 데 있어서 MGDL의 효과성을 강조합니다.



### Neural Quantum Propagators for Driven-Dissipative Quantum Dynamics (https://arxiv.org/abs/2410.16091)
Comments:
          7 pages, comment are welcome!

- **What's New**: 이번 연구는 강한 레이저에 의해 구동되는 개방 양자 시스템의 동역학을 설명하는 새로운 방법론인 Driven Neural Quantum Propagators (NQP)를 개발하였습니다. NQP는 고유 함수(wavefunction)나 밀도 행렬(density matrices) 대신 전파자(propagators)를 근사화하여 구동-소산적 양자 동역학을 해결하는 보편적인 신경망(neural network) 프레임워크입니다.

- **Technical Details**: NQP는 임의의 초기 양자 상태(initial quantum states)를 처리할 수 있으며, 다양한 외부 필드(external fields)에 적응하고, 훈련된 시간이 짧음에도 불구하고 긴 시간 동역학(long-time dynamics)을 시뮬레이션할 수 있습니다. 훈련된 NQP는 다양한 해밀토니안(Hamiltonians)에 의해 지배되는 시스템에 적용 가능하도록 조정된 외부 필드로 이전할 수 있습니다.

- **Performance Highlights**: 연구에서는 스핀-보존(spin-boson) 모델과 세 가지 상태 전이 감마(three-state transition Gamma) 모델을 통해 NQP의 효과성을 입증하였습니다. 모델의 성능은 최대 시간(tm_max)에 강하게 의존하며, 훈련 시간 창을 초과하는 경우에도 해당 동역학을 정확히 예측할 수 있음을 보여주었습니다.



### Fine-Tuning LLMs for Reliable Medical Question-Answering Services (https://arxiv.org/abs/2410.16088)
Comments:
          8 pages, 10 figures, accepted and to be published in the proceedings of 2024 IEEE International Conference on Data Mining Workshops (ICDMW)

- **What's New**: 최신 의료 QA 서비스 개선을 위한 LLM(대형 언어 모델)의 세밀하게 조정된 활용을 제안. 특히 LLaMA-2와 Mistral과 같은 모델을 활용하여 의료 정보를 보다 정확하고 신뢰성 있게 제공.

- **Technical Details**: rsDoRA+ 및 ReRAG와 같은 고급 세밀 조정 기법을 통해 LLM의 성능을 향상시키는 방식으로 접근. rsDoRA+는 분해된 모델 가중치와 간섭을 방지하는 학습률 변화를 통해 안정적인 학습을 돕고, ReRAG는 검색 및 질문 재작성을 통합하여 응답의 정확성을 높임.

- **Performance Highlights**: 본 연구는 환자 신뢰도를 높이고 정보 접근성을 개선하여, 의료 제공자가 신속하고 신뢰할 수 있는 정보를 통해 보다 효율적인 의사 결정을 내릴 수 있도록 지원하는 것을 목표로 함.



### Integrated Image-Text Based on Semi-supervised Learning for Small Sample Instance Segmentation (https://arxiv.org/abs/2410.16063)
- **What's New**: 이번 논문은 적은 샘플의 인스턴스 분할(instance segmentation) 문제를 해결하기 위한 새로운 접근 방식을 제안합니다. 기존 메타-러닝(meta-learning) 전략을 대체하여, 추가적인 주석(annotation) 부담과 훈련 비용을 증가시키지 않고 기존 정보의 최대화를 통해 효과적인 해결책을 제공합니다.

- **Technical Details**: 제안된 방법은 두 개의 모듈로 구성됩니다. 첫 번째 모듈은 무주석 데이터(unlabeled data)를 활용하여 생성된 의사 레이블(pseudo labels)을 학습함으로써 샘플 수를 증가시킵니다. 두 번째로, 텍스트와 이미지의 특징을 통합하여 보다 정확한 분류 결과(classification results)를 도출합니다. 이 두 모듈은 박스-프리(box-free) 및 박스-의존(box-dependent) 프레임워크에 적합합니다.

- **Performance Highlights**: 세 가지 서로 다른 장면(육상, 수중, 현미경)에서 실험을 진행하였으며, 결과적으로 통합된 이미지-텍스트가 분류의 신뢰성을 향상시키고, 의사 레이블이 모델로 하여금 더 정밀한 마스크(mask)를 획득하도록 도와줍니다. 모든 실험 결과는 제안된 방법의 효과성과 우수성을 입증합니다.



### TreeBoN: Enhancing Inference-Time Alignment with Speculative Tree-Search and Best-of-N Sampling (https://arxiv.org/abs/2410.16033)
- **What's New**: TreeBoN은 Best-of-N(BoN) 샘플링에 투기적 트리 탐색 전략을 통합하여, 정보 효율성을 높이면서도 고품질 출력을 유지하도록 설계된 새로운 프레임워크입니다. 이 방법은 부모 노드를 유지하며 반복적으로 저품질 응답을 가지치기하여 계산 비용을 줄입니다.

- **Technical Details**: TreeBoN은 DPO(Direct Preference Optimization)에서 얻은 토큰 수준 보상을 활용하여 트리 확장을 유도하고 저품질 경로를 가지치기합니다. Monte Carlo Tree Search(MCTS) 기술을 접목하여 더 나은 디코딩 성능을 추구합니다.

- **Performance Highlights**: TreeBoN은 192 및 384 토큰에서 최대 65%의 승률을 기록하며, 동일한 계산 비용으로 표준 BoN을 초월하는 성능을 보여줍니다. 전체 데이터셋에서 60% 이상의 승률을 달성하여 확장성과 정렬 효율성을 증명하였습니다.



### TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis (https://arxiv.org/abs/2410.16032)
- **What's New**: 이 논문은 시계열 분석 분야에서 새로운 모델인 TimeMixer++를 제안합니다. 이 모델은 다양한 시계열 작업을 효과적으로 수행하기 위해 다중 스케일 및 다중 주기성을 활용하는 고급 패턴 추출 능력을 갖추고 있습니다.

- **Technical Details**: TimeMixer++는 다음의 방법을 통해 다중 스케일 시계열을 처리합니다: (1) Multi-resolution Time Imaging (MRTI), (2) Time Image Decomposition (TID), (3) Multi-scale Mixing (MCM), (4) Multi-resolution Mixing (MRM). 이 과정에서 MRTI는 시계열을 다중 해상도의 시간 이미지로 변환하고, TID는 계절성과 추세를 분해하여 MCM과 MRM을 통해 통합합니다.

- **Performance Highlights**: TimeMixer++는 8개의 시계열 분석 작업에서 최첨단 성능을 달성하며, 기존의 일반 목적 및 특정 작업 모델을 일관되게 초월합니다. 이는 모델의 적응성이 다양한 작업에 맞춰 더 효과적으로 패턴을 캡처할 수 있음을 나타냅니다.



### Massimo: Public Queue Monitoring and Management using Mass-Spring Mod (https://arxiv.org/abs/2410.16012)
Comments:
          8 pages, 6 figures, 3 algorithms, 3 tables

- **What's New**: 이 논문은 공공 공간에서의 대기열 관리 문제를 해결하기 위한 새로운 기술을 제안합니다. 이 기술은 YOLO(You Only Look Once) 모델과 머신 러닝 알고리즘을 활용하여 가능한 대기 경로를 최적화하고, 비정상적인 대기 상태를 감지하는 시스템을 개발합니다.

- **Technical Details**: 제안된 시스템은 YOLOv7 및 YOLOv8 모델을 사용하여 이미지에서 개인의 신체 점을 감지하고, 이를 통해 대상 대기 이력을 분석합니다. 계산된 신체 점의 힘을 시각화하기 위해 가상의 스프링 시스템을 도입하여 사람 간의 상호작용을 모델링합니다. 또한 선형 회귀 및 다항 회귀와 같은 회귀 분석 기법을 통해 대기 경로를 최소화하려고 합니다.

- **Performance Highlights**: 본 시스템을 통해 대기 시간 단축 및 자원 할당 최적화를 달성하고, COVID-19 이후 시대의 시장 요구에 부응하는 개선된 서비스 품질을 제공할 것으로 기대됩니다.



### CA*: Addressing Evaluation Pitfalls in Computation-Aware Latency for Simultaneous Speech Translation (https://arxiv.org/abs/2410.16011)
- **What's New**: 본 논문에서는 Simultaneous Speech Translation (SimulST) 시스템의 지연(latency) 측정의 한계를 탐구하고, 기존의 지연 평가 접근법에 대한 오해를 밝혀냄으로써 보다 정확한 측정 방법을 제안합니다.

- **Technical Details**: SimulST 시스템의 성능은 주로 지연 시간을 기반으로 평가됩니다. 기존의 지연 측정 지표들(Average Proportion, Average Lagging 등)은 비현실적으로 높은 지연치를 보이며, 체계적인 문제를 내포하고 있습니다. 본 연구에서는 'computation-aware latency'를 포함하는 새로운 지표를 제안하여, 계산 시간을 무시하지 않고 더욱 실질적인 성능 평가를 수행합니다.

- **Performance Highlights**:  새로운 접근法을 통해 SimulST 시스템의 지연 측정 정확도가 향상되며, 특히 비분할 스트리밍(long speech) 환경에서의 적용 가능성이 증가합니다. 이를 통해 더 나은 해석 및 강의 필사 transcription과 같은 실 세계 시나리오에서의 성능 향상이 기대됩니다.



### Resilient Temporal GCN for Smart Grid State Estimation Under Topology Inaccuracies (https://arxiv.org/abs/2410.16008)
Comments:
          9 pages, 5 figures

- **What's New**: 이번 논문에서는 전력 시스템의 상태 추정(State Estimation, SE)에서 그래프 신경망(Graph Neural Networks, GNNs)의 중요한 역할을 분석하고, 시스템의 그래프 구조에 대한 불확실성이 TGCN(Temporal Graph Convolutional Network) 모델 성능에 미치는 영향을 평가합니다. 특히, 측정 데이터를 기반으로 생성된 지식 그래프(Knowledge Graph)를 통합하여 TGCN 모델의 성능을 개선하는 새로운 방법론을 제안합니다.

- **Technical Details**: TGCN 모델에 대해 링크 추가 및 제거와 같은 형태의 전 topology 불확실성을 고려한 다양한 TGCN 변형 구조를 제시하며, 세 가지 종류의 지식 그래프 디자인을 사용하여 각 모델의 성능을 평가합니다. 또한, Knowledge Graph Infused Model (KGIM) 및 Parallel Knowledge Graph Infused Model (PKGIM)이라는 두 가지 아키텍처 변형을 소개합니다.

- **Performance Highlights**: 제안된 KGIM 및 PKGIM 모델은 전통적인 TGCN 모델에 비해 topology의 불확실성에 대해 안정성을 높였음을 보여줍니다. 모든 변형들이 TGCN 성능을 향상시켰고, 특정 구성에서는 특히 큰 개선 효과가 있었습니다.



### 1024m at SMM4H 2024: Tasks 3, 5 & 6 -- Ensembles of Transformers and Large Language Models for Medical Text Classification (https://arxiv.org/abs/2410.15998)
Comments:
          short paper , acl 2024

- **What's New**: 이 논문은 소셜 미디어 데이터를 활용하여 건강과 관련된 여러 과제를 다루고 있습니다. 특히, 자연과 야외 공간이 정신 건강에 미치는 영향, 아동의 건강 장애를 보고하는 트윗의 이진 분류, 사용자의 나이를 자가 보고하는 텍스트의 이진 분류를 포함한 SMM4H 2024의 세 가지 과제를 조사합니다.

- **Technical Details**: 논문에서 사용된 모델은 RoBERTa, DeBERTa, Longformer와 같은 Transformer 모델 및 GPT-4, Claude-Opus, Llama-3 8B와 같은 LLM(대형 언어 모델)입니다. 각 모델의 장단점을 분석하며, 패거지를 통해 성능을 향상시키는 방법도 다룹니다. 데이터셋은 Reddit 게시물과 트윗으로 구성되어 있으며, 분류 작업을 수행하기 위해 여러 프로세스를 테스트했습니다.

- **Performance Highlights**: 모델 성능은 Task 5와 6에서 0.99 이상의 recall을 기록하였습니다. LLM 접근 방식이 4비트 정밀도로도 비교적 좋은 결과를 냈으나, 전체 정밀도를 이용할 경우 성능이 더 향상될 것으로 예상됩니다. 데이터 증강 방법은 성능 향상에 기여하지 않았지만, 데이터 증대는 성능을 개선할 수 있는 가능성을 보여줍니다.



### Augmenting Legal Decision Support Systems with LLM-based NLI for Analyzing Social Media Evidenc (https://arxiv.org/abs/2410.15990)
Comments:
          8 pages , accepted to emnlp 2024

- **What's New**: 본 논문은 2024 NLLP 공유 작업의 법적 자연어 추론(L-NLI) 과제에 대한 시스템 설명과 오류 분석을 제공합니다. 우리의 시스템은 다른 참가자들에 비해 월등한 성능을 보였으며, 법률 텍스트 분석에서의 접근 방식의 효과성을 입증했습니다.

- **Technical Details**: 이 연구에서는 Gemma, Phi3, Zephyr, LLaMA, Mistral, OpenHermes, Qwen 등 여러 LLMs를 활용하여 관계를 entailed, contradicted, neutral로 분류하는 작업을 수행했습니다. 데이터셋은 법적 전제와 온라인 미디어 텍스트로 구성되어 있으며, 초기 fine-tuning을 위해 SNLI 데이터셋의 20,000 샘플이 사용되었습니다. 다양한 alignment 접근 방식과 multi-stage learning을 통한 성능 향상 방법도 실험적으로 시도되었습니다.

- **Performance Highlights**: 우리는 Type-1 오류를 완전히 피하고 Type-2 오류로 한정하였으며, Neutral과 Entailed 간의 혼동이 가장 흔한 오류로 나타났습니다. 시스템의 성능은 대부분의 도메인에서 baseline을 초과했으며, BIPA에서는 성능이 상대적으로 약했습니다. 향후 앙상블 사용이나 더 많은 훈련 자료를 추가하여 성능을 개선할 수 있는 여지가 있습니다.



### Analyzing Closed-loop Training Techniques for Realistic Traffic Agent Models in Autonomous Highway Driving Simulations (https://arxiv.org/abs/2410.15987)
Comments:
          15 pages, 6 figures, 4 tables

- **What's New**: 본 논문은 고속도로 주행 시뮬레이션을 위한 여러 개의 훈련 원칙을 비교 분석하며, 특히 폐쇄 루프(closed-loop) 방법에 중점을 두고 다양한 훈련 방식의 효과를 연구합니다.

- **Technical Details**: 이 연구에서는 개방 루프(open-loop) 및 폐쇄 루프 다중 에이전트 훈련, 적대적(adversarial) 및 결정론적(deterministic) 감독 학습(supervised learning), 강화 학습(reinforcement learning) 손실의 영향, 로그 재생(log-replay) 에이전트와 함께 훈련하는 방법의 영향을 포함한 실험적 비교를 수행합니다. 다중 에이전트(GNN) 그래프 신경망 아키텍처를 사용하여 다양한 훈련 방법에서 직관적인 정책 매개변수를 제안합니다.

- **Performance Highlights**: 모든 조사된 폐쇄 루프 훈련 전략은 개방 루프 방식보다 우수한 성능을 보였고, 강화 신호는 충돌률(collision rate) 같은 중요한 메트릭을 개선했지만, 다른 현실성(realism) 메트릭은 악화될 수 있음을 보여주었습니다. 결정론적 감독 학습이 고속도로 시나리오에서 확률적 적대적 학습과 경쟁할 수 있으며, 다양한 폐쇄 루프 정책 학습 전략을 결합하면 충돌률을 개선하면서도 현실성을 유지할 수 있음을 확인했습니다.



### Large Language Models for Cross-lingual Emotion Detection (https://arxiv.org/abs/2410.15974)
Comments:
          6 pages , accepted to acl 2024

- **What's New**: 이번 논문은 WASSA 2024 Task 2에 제출한 시스템에 대한 상세한 설명을 제시하며, 다양한 언어의 감정을 효과적으로 이해하고 분류하기 위해 대규모 언어 모델(LLMs)을 조합하여 활용한 방법을 소개합니다. 이 접근법은 다른 제출물보다 높은 성능을 보였으며 여러 모델을 통합해 성능을 향상시키는 데서 강점을 보였습니다.

- **Technical Details**: 감정 탐지 시스템은 GPT-4, Claude-Opus, LLAMA-3-8B, Gemma-7B, Mistral-v2-7B와 같은 여러 개방형 및 독점 LLM을 사용하여 구성되었습니다. 모델들은 각각 4-bit과 16-bit 정밀도에서 테스트되었고, 데이터셋은 네덜란드어, 영어, 프랑스어, 러시아어, 스페인어로 구성되어 있으며 6개의 감정 클래스로 주석이 달렸습니다. 시스템은 5 에포크 동안 학습률 0.0002, 가중치 감소 0.01로 비독점 LLM을 미세 조정하였습니다.

- **Performance Highlights**: 앙상블 모델은 직접적인 접근법에 비해 성능이 월등히 좋았으며, 특정 언어에 대해 더 좋은 성능을 보이는 모델이 있었습니다. 최종 결과는 가중 F1 점수를 기준으로 하며, 4-bit 정밀도에서의 성능 저하가 최소였으나 특정 케이스에서 4-bit가 16-bit보다 정확한 예측을 제공했습니다. 이러한 실험은 모델 선택과 데이터 증가를 기반으로 한 다양한 접근 방식을 검토하였습니다.



### Karush-Kuhn-Tucker Condition-Trained Neural Networks (KKT Nets) (https://arxiv.org/abs/2410.15973)
- **What's New**: 이 논문은 Karush-Kuhn-Tucker (KKT) 조건을 활용하여 볼록 최적화 문제를 해결하는 새로운 접근 방식을 제시합니다. 이 방식은 Theory-Trained Neural Networks (TTNNs)와 유사하게, 최적화 문제의 매개변수를 신경망에 입력으로 사용하고, 예상 출력으로는 최적의 프라이멀(primal) 및 듀얼(dual) 변수를 도출합니다.

- **Technical Details**: KKT Loss라는 손실 함수는 네트워크의 출력이 KKT 조건을 얼마나 잘 만족하는지를 측정합니다. 이 연구에서는 선형 프로그램을 예제로 사용하여 KKT Loss를 최소화하는 것이 데이터 손실(Data Loss)과 KKT Loss의 가중치를 합한 것보다 더 효과적임을 보여줍니다. 또한, 데이터 손실만 최소화할 경우 얻는 결과는 KKT Loss를 최소화한 결과보다 열악하다고 보고되었습니다.

- **Performance Highlights**: 현재 접근 방식은 유망하지만, 도출된 프라이멀 및 듀얼 솔루션이 실제 최적 솔루션과 충분히 일치하지 않는다는 한계가 있습니다. 앞으로는 더 나은 모델을 개발하여 실제 솔루션에 더 가까운 결과를 얻고, 다른 문제 유형으로 접근을 확장할 계획입니다.



### Self-Explained Keywords Empower Large Language Models for Code Generation (https://arxiv.org/abs/2410.15966)
- **What's New**: 이 논문은 코드 생성에서 저주 받는 저빈도 키워드의 이해 부족을 극복하기 위한 새로운 기법인 SEK(Self-Explained Keywords)를 제안합니다. 기존의 LLM(대형 언어 모델)들이 저빈도 키워드를 무시하거나 오해하는 문제를 해결하고, 이는 코드 생성 정확도를 높이는 데 기여하고자 합니다.

- **Technical Details**: SEK는 문제 설명에서 중요 키워드를 추출하고 이를 설명하여 기존의 문제 설명을 풍부하게 하여 LLM의 코드 생성 성능을 향상시킵니다. 이 과정은 세 가지 주요 단계로 구성됩니다: 1) KeyExtract & Explain: 문제 설명을 기반으로 키워드를 추출하고 설명합니다. 2) KeyRank: 추출된 키워드를 빈도 기반으로 정렬합니다. 3) PromptEnrich: 정렬된 키워드와 설명을 원래 문제 설명에 추가하여 풍부한 문제 설명을 만듭니다.

- **Performance Highlights**: SEK는 여러 코드 생성 벤치마크에서 LLM의 성능을 유의미하게 개선합니다. 예를 들어, SEK를 활용한 DeepSeek-Coder-V2-Instruct는 HumanEval 벤치마크에서 Pass@1 점수를 85.4%에서 93.3%로 향상시켰습니다. 또한, Llama-3.1은 평균적으로 사용된 벤치마크에서 8.8%의 상대적 개선을 달성했습니다.



### Systematic Exploration of Dialogue Summarization Approaches for Reproducibility, Comparative Assessment, and Methodological Innovations for Advancing Natural Language Processing in Abstractive Summarization (https://arxiv.org/abs/2410.15962)
- **What's New**: 이 연구는 자연어 처리(NLP) 분야에서의 대화 요약 모델의 재현성 및 평가에 대한 연구로, 원래 연구와의 불일치를 집중 분석합니다.

- **Technical Details**: 대화 요약 모델 분석은 AMI(Augmented Multi-party Interaction) 데이터셋을 사용하여 수행하였으며, 조직적 메모리 네트워크(Hierarchical Memory Networks, HMNet)와 여러 버전의 포인터 생성 네트워크(Pointer-Generator Networks, PGN)를 포함합니다: PGN(DKE), PGN(DRD), PGN(DTS), PGN(DALL).

- **Performance Highlights**: 원래 연구와의 불일치에 대한 심층 분석을 통해, 인간 평가를 통한 요약의 정보성(informativeness)과 품질(quality)을 평가하였습니다. 데이터셋 1에서의 샘플 표준 편차는 0.656으로, 데이터 포인트의 평균 주위 분산이 중간 정도임을 나타냅니다.



### Do Large Language Models Have an English Accent? Evaluating and Improving the Naturalness of Multilingual LLMs (https://arxiv.org/abs/2410.15956)
- **What's New**: 현재의 대형 언어 모델들은 주로 영어를 주요 언어로 설계되어 있으며, 다국어 모델조차도 강한 영어 중심 편향을 보입니다. 본 논문은 다국어 LLM 출력의 어휘적 및 구문적 자연스러움을 평가하기 위한 새로운 자동화된 메트릭을 소개하고, 이 메트릭을 이용하여 프랑스어와 중국어에서의 성능을 분석합니다.

- **Technical Details**: 우리는 직접 선호 최적화(Direct Preference Optimization, DPO) 방법을 사용하여 특정 언어에서 LLM의 자연스러움을 향상시키기 위한 간단하고 효과적인 접근 방식을 제안합니다. 새로운 선호 데이터셋을 통해 인간이 작성한 응답과 합성적으로 조작된 응답을 비교합니다. 이를 통해 기존 LLM의 자연스러움을 개선하는 데 있어 일관된 성과를 보입니다.

- **Performance Highlights**: 본 연구의 결과는 LLM이 중국어에서 자연스러움을 향상시킬 수 있음을 보여주며, 일반적인 벤치마크의 성능을 저해하지 않으면서도 자연스러움을 일관되게 개선할 수 있음을 확인했습니다.



### TS-ACL: A Time Series Analytic Continual Learning Framework for Privacy-Preserving and Class-Incremental Pattern Recognition (https://arxiv.org/abs/2410.15954)
Comments:
          11 pages, 3 figures, 2 tables

- **What's New**: 본 연구에서는 Catastrophic Forgetting 문제를 해결하기 위한 Time Series Analytic Continual Learning 프레임워크인 TS-ACL을 제안합니다. 이 방법은 신경망 업데이트를 기울기 기반의 방법 대신 선형 회귀 문제로 변환하여 지속적으로 들어오는 데이터에 대처합니다.

- **Technical Details**: TS-ACL은 사전 훈련된 동결된 피쳐 추출 인코더를 활용하며, 분석 분류기만을 가벼운 방식으로 재귀적으로 업데이트합니다. 이를 통해 TS-ACL은 실제 응용 프로그램 및 대규모 데이터 처리에 적합합니다. 또한, 모델은 중앙 집중식 방식으로 전체 데이터셋에서 훈련된 모델과 정확히 동등한 특성을 가집니다.

- **Performance Highlights**: 광범위한 실험 결과, TS-ACL은 기존의 기준 모델들에 비해 뛰어난 성능을 보이며, 특히 Catastrophic Forgetting 문제를 효과적으로 완화합니다.



### AI-Driven Approaches for Glaucoma Detection -- A Comprehensive Review (https://arxiv.org/abs/2410.15947)
- **What's New**: 이번 연구는 AI(인공지능), 특히 DL(딥러닝) 기술을 이용한 CADx(컴퓨터 보조 진단) 시스템의 발전을 다루고 있습니다. 이 시스템은 녹내장(Glaucoma) 조기 진단을 지원하며, 안전성, 신뢰성, 해석 가능성, 설명 가능성의 향상이 필요함을 강조하고 있습니다.

- **Technical Details**: CADx 시스템 개발에는 Computer Vision(CV), 전통적인 Machine Learning(ML), DL 및 하이브리드 접근 방식이 포함됩니다. 시스템은 여러 단계로 구성되며, 이미지 전처리, 세분화, 특징 추출 등을 포함합니다.

- **Performance Highlights**: 연구에서 342개의 중요 논문을 분석하여, 82개의 CV 및 전통적인 ML 기법, 203개의 DL 기법, 39개의 하이브리드 접근 방식을 포함한 연구 결과를 제시하였습니다. 특히, 최근 6년 간 DL 기술의 사용이 급증하며 CADx 시스템의 선택과 개선을 위한 연구 간극과 방향성을 제시하였습니다.



### Developing Retrieval Augmented Generation (RAG) based LLM Systems from PDFs: An Experience Repor (https://arxiv.org/abs/2410.15944)
Comments:
          36 pages, 8 figures, 2 tables, and python code snippets

- **What's New**: 이 논문은 PDF 문서를 주요 데이터 소스로 사용하여 Retrieval Augmented Generation (RAG) 시스템을 개발한 경험을 보고합니다. RAG 아키텍처는 대형 언어 모델(LLMs)의 생성 능력과 정보 검색의 정밀성을 결합하여 투명성, 정확성 및 상황에 맞는 응답을 향상시킬 수 있는 잠재력을 지니고 있습니다.

- **Technical Details**: RAG 시스템의 개발은 데이터 수집, 전처리, 검색 인덱싱 및 응답 생성에 이르는 전체 파이프라인을 구성합니다. MLLM과 정보 검색(NLP의 두 가지 핵심 구성 요소)를 통합하여 정보를 외부 데이터 소스에서 끌어오는 방식을 설명합니다. 이를 통해 정보를 현실적이고 정확하게 제공할 수 있는 시스템을 만듭니다.

- **Performance Highlights**: RAG 모델 프레임워크는 생성 AI의 패러다임 변화를 가져옵니다. 고급 NLP 애플리케이션에서 RAG 시스템의 도입이 의미하는 바는 전통적인 모델의 불투명한 출력을 넘어, 신뢰성 높은 응답을 생성하는 데 기여할 수 있습니다.



### Centrality-aware Product Retrieval and Ranking (https://arxiv.org/abs/2410.15930)
Comments:
          EMNLP 2024: Industry track

- **What's New**: 이 논문은 전자상거래 플랫폼에서 사용자 경험을 향상시키기 위한 제품 순위를 개선하는 방법을 다룹니다. 기존의 Transformer 기반 모델들이 사용자 의도를 반영하지 못하는 문제를 해결하기 위해, eBay의 데이터셋에서 수작업으로 주석을 달아 사용자의 의도와 관련된 점수를 부여하는 새로운 접근법인 User-intent Centrality Optimization (UCO)을 제안합니다.

- **Technical Details**: UCO 접근법은 쿼리-제목 쌍의 검색에서 사용자 의도를 최적화하며, 특히 의미적으로 관련 있지만 사용자 의도를 반영하지 않는 하드 네거티브(hard negatives)를 처리하기 위해 듀얼 손실 기반 최적화(dual-loss based optimization)를 활용합니다. 기존의 내부 평가 세트(Internal Graded Relevance, IGR dataset)를 이용하여 도전적인 평가 세트를 선별하고, 이를 통해 검색 효율성을 개선합니다.

- **Performance Highlights**: 이 연구는 제안된 프레임워크가 다양한 평가 메트릭에서 제품 순위 효율성을 크게 향상시키는 결과를 보여주었습니다. 사용자 의도에 맞춘 제목들이 높은 순위에 오르도록 하여 전자상거래 플랫폼에서 사용자 경험을 개선하는 데 기여합니다.



### GReFEL: Geometry-Aware Reliable Facial Expression Learning under Bias and Imbalanced Data Distribution (https://arxiv.org/abs/2410.15927)
Comments:
          ACCV 2024. Extended version of ARBEx (arXiv:2305.01486)

- **What's New**: 본 연구에서는 GReFEL(GEometry-based Reliable Facial Expression Learning)이라는 혁신적인 모델을 제안합니다. 이 모델은 Vision Transformers (ViTs)와 얼굴 기하학적 특성을 고려한 신뢰성 균형 모듈을 활용하여 얼굴 표정 인식(FEL)에서 발생하는 데이터 불균형 및 편향 문제를 해결합니다.

- **Technical Details**: GReFEL은 여러 단계의 주의 기반(feature extraction) 기능 추출과 신뢰성 균형 모듈을 포함하여 복잡한 감정 표시를 효과적으로 처리하는 것을 목표로 합니다. 이 모델은 공간에서 학습 가능한 앵커를 배치하여 다양한 표정 변화 간의 유사성을 측정하고, 다중 헤드의 자기 주의(multi-head self-attention) 메커니즘을 활용하여 중요한 기능을 식별합니다. 또한, 입력 이미지에 다양한 데이터 증강 기법을 적용하여 훈련 과정에서의 편향과 과적합을 방지합니다.

- **Performance Highlights**: GReFEL은 다양한 데이터셋에서 실행된 실험을 통해 현재의 최첨단 얼굴 표정 인식 시스템보다 일관되게 우수한 성능을 보였습니다. 이는 모델이 불균형한 데이터 분포, 편향 및 불확실성을 효과적으로 해결할 수 있도록 설계되어 있음을 의미합니다.



### Bench4Merge: A Comprehensive Benchmark for Merging in Realistic Dense Traffic with Micro-Interactive Vehicles (https://arxiv.org/abs/2410.15912)
Comments:
          6 pages, 7 figures, IEEE international conference on robotics and automation

- **What's New**: 자율 주행 기술이 발전함에 따라, 복잡한 교통 상황에서의 자동차 합류(Merging)는 여전히 큰 도전 과제로 남아있습니다. 본 논문에서는 실시간 상호작용이 중요한 지점에서 자동차 합류를 평가하기 위한 새로운 벤치마크인 Bench4Merge를 제안했습니다. 다양한 차량 행동을 처리할 수 있는 방법론이 소개되었습니다.

- **Technical Details**: Bench4Merge는 세 가지 주요 구성 요소로 구성됩니다: 시나리오 생성(Scenario-level Generation), 주요 차선 차량에 대한 마이크로 제어 모델(Micro-Controllable Model for Main-Lane Vehicles), 그리고 LLM 기반 평가(LLM-Based Evaluation)입니다. 이 시스템은 실제 데이터 세트를 기반으로 초기 환경을 생성하고, 대규모 데이터에서 마이크로 수준의 상호작용 특성을 캡처하기 위해 훈련된 차량 동작 정책을 포함합니다.

- **Performance Highlights**: 다양한 자동차 합류 알고리즘을 적용해 보면서, 기존 평가 방법에서 간과되었던 문제점들을 발견했습니다. 이를 통해 Bench4Merge는 합류 방식의 성능을 더욱 포괄적으로 분석할 수 있는 새로운 평가 기준을 제공함으로써, 자율 주행 기술의 발전에 기여할 것으로 기대됩니다.



### Diverse Policies Recovering via Pointwise Mutual Information Weighted Imitation Learning (https://arxiv.org/abs/2410.15910)
Comments:
          18 pages, 6 figures

- **What's New**: 본 논문은 전문가 궤적을 기반으로 다양한 정책을 회복하는 새로운 방법론을 제시합니다. 특히, 궤적의 잠재적 스타일을 파악한 후, 점대상 상호정보(Pointwise Mutual Information, PMI)를 기반으로 상태-행동 쌍의 중요성을 반영하여 기존의 행동 복제(Behavioral Cloning, BC) 방식을 향상합니다.

- **Technical Details**: 제안된 방법은 Behavioral Cloning with Pointwise Mutual Information Weighting(BC-PMI)로 명명되었습니다. 이 방법은 상태-행동 쌍과 스타일 간의 관계를 정량화하여 학습의 초점을 더욱 명확하게 맞추며, Mutual Information Neural Estimation(MINE)을 이용하여 PMI를 추정합니다. 이론적으로 이 방법은 다양한 정책을 복구하는 두 가지 극단적인 경우를 통합합니다.

- **Performance Highlights**: 실험 결과, BC-PMI는 Circle 2D, Atari 게임 및 전문 농구 선수 데이터셋에서 다양한 정책을 회복하는 데 있어 기존의 방법들보다 우수한 성능을 보였습니다.



### Model Mimic Attack: Knowledge Distillation for Provably Transferable Adversarial Examples (https://arxiv.org/abs/2410.15889)
- **What's New**: 본 연구에서는 black-box 설정에서 knowledge distillation을 이용한 adversarial 공격 방법을 제안합니다. 기존 연구와는 달리, 학생 모델이 학습 능력이 충분할 경우, 유한한 수의 distillation iteration 내에서 adversarial perturbation을 발견할 수 있도록 보장하는 것이 특징입니다.

- **Technical Details**: 제안된 방법은 Model Mimic Attack이라고 하며, 여러 개의 surrogate 모델을 iterative하게 훈련시켜 expanding dataset에서 black-box 모델의 행동을 모방하도록 합니다. 이는 효과적인 transfer-based black-box adversarial attack을 가능하게 하며, 각 surrogate 모델은 유한한 지점 집합 내에서 teacher 모델의 예측을 복사합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 이미지 분류 분야에서 다른 black-box 공격 방법들에 비해 효율적임을 입증하였습니다. 특히, knowledge distillation 기반의 모델 전이 공격이 black-box teacher 모델에 대해 adversarial perturbation을 찾을 수 있음을 이론적으로 입증한 것이 큰 성과입니다.



### Using GPT Models for Qualitative and Quantitative News Analytics in the 2024 US Presidental Election Process (https://arxiv.org/abs/2410.15884)
- **What's New**: 본 논문은 Google Search API와 GPT-4o 모델을 활용하여 뉴스의 질적 및 양적 분석을 수행하는 접근 방식을 제안합니다. 이를 통해 2024년 미국 대선 과정에 대한 뉴스를 분석하였습니다.

- **Technical Details**: Retrieval-augmented generation (RAG) 기법을 사용하여 뉴스 데이터를 분석하였습니다. 분석은 Google Search API를 통해 관련 웹 자원을 검색하고, LangChain의 SeleniumURLLoader를 이용하여 정보를 추출하는 두 단계로 이루어졌습니다. 주요 검색 쿼리는 'Kamala Harris AND Donald Trump'이며, 다양한 시간대와 뉴스 출처를 고려하였습니다.

- **Performance Highlights**: GPT 모델을 활용한 분석 결과는 선거 과정에서의 불확실성을 분석하는 데 도움을 주며, 질적 통찰력을 제공함으로써 향후 선거 분석에 응용될 수 있는 가능한 기초 자료를 생성합니다.



### MI-VisionShot: Few-shot adaptation of vision-language models for slide-level classification of histopathological images (https://arxiv.org/abs/2410.15881)
Comments:
          Manuscript accepted for oral presentation at KES-InnovationInMedicine 2024 held on Madeira, Portugal

- **What's New**: 본 연구에서는 MI-Zero의 고변동성 문제를 해결하기 위해 MI-VisionShot이라는 새로운 메소드를 제안합니다. 이 방법은 VLM(vission-language models) 위에서 훈련이 필요 없는 적응(adaptation) 방식으로 슬라이드 수준 레이블 예측을 지원합니다.

- **Technical Details**: MI-VisionShot은 프로토타입 학습(prototypical learning)을 기반으로 하여, 각 슬라이드에서 가장 판별력이 있는 패치를 검색하여 다중 인스턴스(multiple-instance) 설정 하에서 프로토타입 기반 분류기를 생성합니다. 이를 통해 더 나은 레이블 예측이 가능합니다.

- **Performance Highlights**: MI-VisionShot은 제로샷 전이(zero-shot transfer) 방식보다 낮은 변동성을 보이며, 적은 수의 샷(few-shot learning) 상황에서도 뛰어난 성능을 발휘합니다.



### FlickerFusion: Intra-trajectory Domain Generalizing Multi-Agent RL (https://arxiv.org/abs/2410.15876)
Comments:
          NeurIPS '24 Open-World Agents Workshop

- **What's New**: 본 논문에서는 동적 매개체 구성의 도전 과제를 해결하기 위해 FlickerFusion이라는 새로운 OOD(Out-Of-Domain) 일반화 방법을 제안합니다. 이 방법은 기존의 MARL(다중 에이전트 강화 학습) 기법에서 나타나는 성능 저하 문제를 해결하고, 특히 예측할 수 없는 동적 변화에 적응할 수 있는 기술을 제공합니다.

- **Technical Details**: FlickerFusion은 에이전트의 관측 공간을 보강하여 추가적인 파라미터 없이도 효과적으로 동적 매개체 구성에 대응합니다. 이 방식은 시간에 따른 확률적 부분 관측을 집계하여 전체 관측을 에뮬레이트(Emulate)하는 데 초점을 맞춥니다. 이를 통해 가시적으로 발생하는 불확실성을 줄이고 더 나은 안정성을 제공할 수 있습니다.

- **Performance Highlights**: FlickerFusion을 통해 기존 MARL 및 다른 모델-불가지론적 방법과 비교하여 인퍼런스 보상과 불확실성 감소에서 최첨단 성능을 달성하여 실험 결과로 나타났습니다. 또한, 12개의 벤치마크를 통해 동적 MARL 시나리오에 대한 표준화된 평가를 제공하며, 프로젝트 사이트에서 비디오 시각화 자료를 확인할 수 있습니다.



### Mesa-Extrapolation: A Weave Position Encoding Method for Enhanced Extrapolation in LLMs (https://arxiv.org/abs/2410.15859)
Comments:
          Accepted by NeurIPS 2024; 13 pages and 30 pages appendix

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 외삽(extrapolation) 성능 개선을 위한 새로운 방법, Mesa-Extrapolation을 제안합니다. 기존의 Position Encoding (PE) 기법에서 발생하는 한계를 극복하고, 끊임없는 메모리 사용량 절감과 빠른 추론 속도를 달성했습니다.

- **Technical Details**: Mesa-Extrapolation은 청크 기반의 삼각 주의(attention) 매트릭스를 이용하고, Stair PE를 사용하여 최종 청크를 처리합니다. 이는 메모리 친화적인 자원 소모를 목표로 하며, 외삽 성능을 극대화합니다. 이 연구는 No Position Encoding(NoPE)의 실패 이유도 정리했으며, PE의 효과를 이론적으로 분석했습니다.

- **Performance Highlights**: 실험 결과, Mesa-Extrapolation 방법이 저렴한 메모리 요구사항과 빠른 추론 속도를 유지하면서도 경쟁력 있는 성능을 발휘함을 입증했습니다. 이 방법은 추가적인 자원 소모 없이 LLM의 외삽 성능을 향상시키는 간편한 플러그인입니다.



### Random Token Fusion for Multi-View Medical Diagnosis (https://arxiv.org/abs/2410.15847)
Comments:
          Originally published at the NeurIPS 2024 Workshop on Advancements In Medical Foundation Models: Explainability, Robustness, Security, and Beyond (AIM-FM)

- **What's New**: 이 연구에서는 Random Token Fusion (RTF)이라는 새로운 기법을 소개하여 다중 시점(multi-view) 의료 이미지 분석에서의 성능을 향상시킵니다. RTF는 훈련 과정에서 특징 융합(feature fusion) 과정에 무작위성을 도입하여 과적합(overfitting) 문제를 해결하고 진단 모델의 강건성과 정확성을 높입니다.

- **Technical Details**: RTF는 다중 시점 의료 진단의 특징을 변형시키기 위해 훈련 단계에서 서로 다른 시점의 토큰(token)을 무작위로 융합합니다. 이렇게 하면 모델이 한 가지 우세한 시점에만 의존하는 것이 아니라, 서로 다른 시점의 정보를 다양하게 고려하게 됩니다. 이 프로세스는 기존의 비전 트랜스포머(vision transformers)와 통합되어 별도의 수정 없이도 적용할 수 있습니다.

- **Performance Highlights**: RTF는 CBIS-DDSM 및 CheXpert와 같은 표준 벤치마크 데이터셋에서 기존의 다중 시점 모델 성능을 일관되게 향상시킵니다. 이 연구는 유방 촬영술(mammograms) 및 흉부 X선(chest X-rays)에서 최첨단 성능을 달성하였으며, RTF의 출처 코드는 https://jimyug.github.io/RandomTokenFusion에서 확인 가능합니다.



### Long-distance Geomagnetic Navigation in GNSS-denied Environments with Deep Reinforcement Learning (https://arxiv.org/abs/2410.15837)
- **What's New**: 본 논문에서는 GNSS(전 세계 항법 위성 시스템)가 사용이 불가능한 미탐사 구역에서의 장거리 지구 자기 항법을 위한 딥 강화 학습 기반의 메커니즘을 개발합니다.

- **Technical Details**: 제안된 방법은 TD3(정확한 깊이의 지연된 결정적 정책 기울기) 알고리즘을 기반으로 하며, 자극성을 조정하여 고유한 목표를 향해 나아갈 수 있도록 기하 자기 기울기를 활용하여 학습됩니다. 이 접근 방식은 GNSS 서비스가 없는 환경에서도 장거리 항법을 가능하게 합니다.

- **Performance Highlights**: 수치 시뮬레이션 결과, 제안된 접근 방식이 기존의 메타 휴리스틱과 생체 영감 항법 방법보다 장거리 임무에서 더 우수한 성능을 보이는 것으로 나타났습니다.



### MAC Revivo: Artificial Intelligence Paves the Way (https://arxiv.org/abs/2410.15820)
- **What's New**: Wi-Fi 및 Bluetooth 기술의 대규모 채택과 스마트 디바이스의 급속한 증가로 인해 산업, 과학 및 의료(ISM) 대역에서 신호 간섭과 혼잡이 크게 증가했습니다. 이 논문에서는 MAC 프로토콜 설계에 인공지능(AI) 방법을 통합하는 가능성을 탐구합니다.

- **Technical Details**: AI-MAC라는 혁신적인 접근 방식을 제안하여, 머신러닝(ML) 알고리즘을 사용하여 네트워크 조건 변화에 동적으로 적응하고 채널 접근을 최적화하며 간섭을 완화합니다. 이 프레임워크는 DCF 및 HCF와 같은 기존의 MAC 기능을 개선하여, AI/ML을 통해 데이터 전송과 트래픽 관리의 효과를 극대화합니다.

- **Performance Highlights**: AI-MAC는 실험 결과를 통해 간섭과 지연을 상당히 줄이는 성과를 보여주어, 보다 안정적이고 효율적인 무선 통신을 가능하게 합니다. 전반적으로 AI-MAC는 QoS 및 QoE를 강화하여 차세대 Wi-Fi 네트워크에 대한 강력한 솔루션을 제공합니다.



### LiMTR: Time Series Motion Prediction for Diverse Road Users through Multimodal Feature Integration (https://arxiv.org/abs/2410.15819)
Comments:
          Accepted at the NeurIPS 2024 workshop Time Series in the Age of Large Models. Code available at this https URL

- **What's New**: 이번 논문에서는 LiDAR 데이터의 세밀한 지역적 특징을 활용하여 자율주행 차량의 도로 사용자 행동 예측 정확도를 높이고자 하는 새로운 멀티모달 접근 방식인 LiMTR 모델을 제안합니다.

- **Technical Details**: LiMTR 모델은 PointNet 기반 아키텍처를 사용하여 특정 도로 사용자의 LiDAR 데이터를 통합하며, 개인의 자세나 시선 방향과 같은 로컬 특징에 집중합니다. 이 모델은 Waymo Open Dataset을 통해 최소 평균 변위 오차(minADE)에서 6.20%, 평균 평균 정밀도(mAP)에서 1.58%의 성능 향상을 보였습니다.

- **Performance Highlights**: LiMTR 모델은 LiDAR 기능을 통합하여 자율주행 차량의 모션 예측 작업에서 효과적인 개선을 이끌어내었으며, 이는 더욱 안전한 자율주행 환경 구축에 기여할 것으로 기대됩니다.



### Kaninfradet3D:A Road-side Camera-LiDAR Fusion 3D Perception Model based on Nonlinear Feature Extraction and Intrinsic Correlation (https://arxiv.org/abs/2410.15814)
- **What's New**: 이 논문에서는 도로변 인식에 대한 연구의 필요성을 강조하며, Kolmogorov-Arnold Networks (KANs)를 이용한 Kaninfradet3D 모델을 제안하여 카메라와 LiDAR 간의 효과적인 데이터 융합을 통한 3D 객체 인식 성능을 향상시킵니다.

- **Technical Details**: Kaninfradet3D는 복잡한 고차원 데이터를 효과적으로 처리하기 위해 KAN Layer가 적용된 인코더와 퓨저 모듈을 개선하였습니다. 교차 주의 메커니즘(Cross-attention)을 통해 카메라와 LiDAR의 상호 의존성을 포착하여 정보 융합의 질을 높였습니다.

- **Performance Highlights**: 제안된 모델은 TUMTraf Intersection Dataset에서 두 관점에서 +9.87 mAP 및 +10.64 mAP의 성능 향상을 보여주었으며, TUMTraf V2X Cooperative Perception Dataset의 도로변 끝 부분에서 +1.40 mAP의 향상을 달성했습니다.



### Deep Learning and Data Augmentation for Detecting Self-Admitted Technical Deb (https://arxiv.org/abs/2410.15804)
Comments:
          Accepted to be published at the 2024 31st Asia-Pacific Software Engineering Conference (APSEC)

- **What's New**: 본 연구는 Self-Admitted Technical Debt (SATD)의 식별 및 분류를 위한 새로운 접근 방법을 소개합니다. 특히, BiLSTM 및 BERT 아키텍처를 사용하여 SATD를 효과적으로 식별하고 다양한 유형으로 분류하는 방법을 제시합니다.

- **Technical Details**: 연구에서 사용된 방법론은 크게 두 단계로 나뉩니다. 첫 번째 단계에서는 BiLSTM 아키텍처를 활용하여 SATD를 식별하고 Not-SATD와 구분합니다. 두 번째 단계에서는 식별된 SATD를 BERT 아키텍처를 통해 다양한 유형으로 분류합니다. 또한, 데이터 불균형 문제를 해결하기 위해 대규모 언어 모델을 활용한 데이터 증강 전략을 적용했습니다.

- **Performance Highlights**: 제안한 접근 방식은 기존의 기법들과 비교할 때 SATD 식별 및 분류 성능이 상당히 향상되었습니다. 특히, 불균형 데이터 문제를 해결함으로써 F1-score가 향상된 결과를 보여주었습니다. 연구에 사용된 균형 잡힌 데이터 세트는 향후 SATD 연구에 기여할 것으로 기대됩니다.



### Habaek: High-performance water segmentation through dataset expansion and inductive bias optimization (https://arxiv.org/abs/2410.15794)
- **What's New**: 본 연구는 SegFormer 모델을 데이터 증강(data augmentation)을 통해 개선하여 실시간 물 segmentation을 위한 실용성과 정확성을 높이고자 합니다. 이는 NVIDIA의 고해상도 이미지를 활용한 기존의 flood monitoring 방법들을 대체할 수 있는 potential을 보여줍니다.

- **Technical Details**: SegFormer는 계층적 Transformer 인코더(hierarchical Transformer encoder)와 경량 MLP 디코더(lightweight all-MLP decoder)로 구성됩니다. 이 모델은 이미지의 다중 스케일 특징을 추출하며, 4x4 패치(patches)를 사용하여 segmentation 작업에 적합하게 설계되었습니다. 또한, LoRA(저랭크 적응, Low-Rank Adaptation) 기법을 활용하여 처리 복잡성을 낮추는 동시에 정확도를 유지합니다.

- **Performance Highlights**: 제안된 Habaek 모델은 IoU(Intersection over Union) 점수 0.91986에서 0.94397까지 성능을 보이며, F1-score, recall, accuracy 및 precision에서도 경쟁 모델보다 우수한 결과를 나타냈습니다. 이는 실제 어플리케이션에서의 활용 가능성을 시사합니다.



### WildOcc: A Benchmark for Off-Road 3D Semantic Occupancy Prediction (https://arxiv.org/abs/2410.15792)
- **What's New**: 이 논문에서는 야외(off-road) 환경에서의 3D semantic occupancy prediction을 위한 최초의 벤치마크인 WildOcc를 소개합니다. 기존의 연구는 주로 도로(on-road) 환경에 집중되어 있었으며, 야외 환경에 적합한 데이터셋과 기준이 부족했습니다. 따라서 이 연구는 중요한 공백을 메우고 있습니다.

- **Technical Details**: WildOcc는 dense occupancy annotations을 제공하는 야외 3D semantic occupancy prediction을 위한 벤치마크입니다. 이 연구는 coarse-to-fine reconstruction 방식을 사용하는 ground truth generation pipeline을 제안하며, 이를 통해 더 현실적인 결과를 도출합니다. 또한, multi-modal 3D semantic occupancy prediction framework인 OFFOcc를 도입하여 multi-frame 이미지와 point cloud에서 spatio-temporal 정보를 voxel level에서 융합합니다. 이 과정에서 cross-modality distillation 기능이 도입되어 point cloud에서 이미지 특징으로 기하학적 지식을 전이합니다.

- **Performance Highlights**: 실험 결과, 제안한 OFFOcc 프레임워크는 야외 3D semantic occupancy prediction 작업에서 높은 성능을 달성했습니다. 이는 기존의 도로 기반 방법들이 아닌, 화려한 지형과 불규칙한 물체들이 많은 야외 환경에서도 효과적으로 작동할 수 있음을 시사합니다.



### Arithmetic Transformers Can Length-Generalize in Both Operand Length and Coun (https://arxiv.org/abs/2410.15787)
Comments:
          38 pages, 16 figures

- **What's New**: 이 연구에서는 길이 일반화(length generalization)를 향상시키기 위한 새로운 접근법을 제시하며, 특히 다중 피연산자 덧셈(multi-operand addition)과 곱셈(multiplication) 같은 복잡한 산술 작업에서 길이 일반화를 2-3배 향상시키는 데 성공하였습니다.

- **Technical Details**: 모델은 각 다음 토큰 예측 단계에서 고정된 수의 토큰에 집중할 수 있도록 설계된 작업 특화 스크래치패드(task-specific scratchpads)를 사용하며, Position Coupling의 다단계 버전을 적용하여 Transformer가 주목해야 할 위치를 인식할 수 있도록 합니다.

- **Performance Highlights**: 이 방식은 1층 Transformer가 임베딩 차원에 따라 지수적인 피연산자 수 및 피연산자의 길이까지 다중 피연산자 덧셈을 해결할 수 있음을 증명하였습니다.



### An Efficient System for Automatic Map Storytelling -- A Case Study on Historical Maps (https://arxiv.org/abs/2410.15780)
- **What's New**: 본 논문에서는 역사적 지도에 대한 캡션 생성의 새로운 접근 방식을 제안합니다. 기존의 이미지 캡셔닝(image captioning) 방법은 자연 이미지에서는 성공적이지만, 지도에 대해서는 성능이 떨어지는 문제를 해결하고자 합니다.

- **Technical Details**: 이 연구에서는 최신의 vision-language 모델인 CLIP을 미세 조정(fine-tune)하여 역사적 지도와 관련된 캡션을 생성하며, GPT-3.5를 활용하여 지도에 대한 간략한 이야기를 제공합니다. 또한, 특정 지도 유형에 맞춰 캡션을 생성할 수 있도록 하는 새로운 결정 트리(decision tree) 구조를 제안합니다.

- **Performance Highlights**: 우리 시스템은 지도에서의 텍스트 변경에 대해 불변성을 보이며, 다른 지도 유형에 쉽게 적용되고 확장될 수 있는 가능성을 가지고 있습니다. 이번 연구에서 공개된 코드는 대규모 지도 캡셔닝 시스템으로 확장될 수 있는 기반을 제공합니다.



### Reducing Hallucinations in Vision-Language Models via Latent Space Steering (https://arxiv.org/abs/2410.15778)
Comments:
          21 pages

- **What's New**: 이번 연구에서는 Large Vision-Language Models (LVLMs)에서 발생하는 hallucination의 메커니즘을 조사하고, 이를 완화하기 위한 Visual and Textual Intervention (VTI)라는 새로운 기술을 제안합니다. VTI는 latent space representation을 조정하여 시각 정보의 안정성을 향상시키는데 중점을 두고 있습니다.

- **Technical Details**: LVLMs는 이미지 인코더와 텍스트 디코더의 비대칭적 사전 훈련으로 인해 hallucination에 민감합니다. 본 연구는 이미지 인코더의 불안정성이 텍스트 디코더의 민감도에 어떤 영향을 미치는지에 대해 분석하였고, VTI를 통해 이를 해결하기 위한 접근법을 제안하였습니다. VTI는 query 이미지의 perturbation에 따라 latent features를 수정하여 작업에 관계없이 적용할 수 있습니다.

- **Performance Highlights**: VTI는 다양한 매트릭스를 통해 기존의 방법들과 비교했을 때 hallucination을 효과적으로 줄일 수 있다는 것을 입증하였으며, LVLMs에서 시각 정보의 안정성이 얼마나 중요한지를 강조합니다.



### Learning to Synthesize Graphics Programs for Geometric Artworks (https://arxiv.org/abs/2410.15768)
Comments:
          ICPR 2024

- **What's New**: 이 연구에서는 그림 생성 과정에서 최종 이미지를 단순히 반환하는 것이 아닌, 디지털 드로잉 도구를 실행 가능한 프로그램 세트로 취급하는 방법을 제시합니다. 이를 통해 사용자는 구체적인 드로잉 명령어를 기반으로 이미지를 재구성할 수 있습니다.

- **Technical Details**: Art2Prog라는 프로그램 합성기를 통해 복잡한 이미지의 입력을 이해하고 해석하는 과정에서, 색상 블렌딩 모드 및 레이어 중첩 등을 관리하는 2D 그래픽 프로그램을 생성합니다. 이를 위해 그래픽 프로그램을 계층적으로 구조화하고, 명확한 색상 혼합 방법을 채택하였습니다.

- **Performance Highlights**: 실험 결과, Art2Prog는 기존의 최첨단 벡터화 기법보다 뛰어난 재구성 정확도를 보여주었으며, 복잡한 그래픽을 표현하는 데 있어 더욱 많은 세부 정보를 담을 수 있습니다.



### LSCodec: Low-Bitrate and Speaker-Decoupled Discrete Speech Codec (https://arxiv.org/abs/2410.15764)
Comments:
          5 pages, 2 figures, 4 tables. Submitted to ICASSP 2025. Demo page: this https URL

- **What's New**: 이번 연구에서는 LSCodec라는 새로운 이산 음성 코덱(discrete speech codec)을 제안합니다. LSCodec는 낮은 비트레이트(low bitrate)와 화자 분리 능력(speaker decoupling capability)을 가지고 있어, 기존의 모델들에 비해 효율적인 음성 생성이 가능합니다.

- **Technical Details**: LSCodec는 세 단계의 비지도 학습(unsupervised training) 프레임워크를 채택합니다. 첫 번째 단계에서 음성 변별 오토인코더(speech variational autoencoder)가 화자의 변형을 사용하여 훈련됩니다. 이후 벡터 양자화(vector quantization, VQ)를 통해 분리된 이산 공간을 구축하며, 마지막으로 오디오 품질을 향상시키기 위해 이산 토큰 보코더(discrete token vocoder)가 훈련됩니다.

- **Performance Highlights**: LSCodec는 단일 코드북(single codebook)과 기존 기준보다 작은 어휘 크기(vocabulary size)를 사용하여 뛰어난 인지 가능성(intelligibility)과 음질(audio quality)을 시연합니다. 특히 25Hz 버전의 LSCodec는 현재까지의 코덱 중 가장 낮은 비트레이트인 0.25kbps를 달성하며, 음성 변환 평가에서도 화자 분리 성능이 뛰어난 것으로 입증되었습니다.



### DeepIcon: A Hierarchical Network for Layer-wise Icon Vectorization (https://arxiv.org/abs/2410.15760)
Comments:
          Accepted as Oral Presentation at DICTA 2024

- **What's New**: DeepIcon은 이미지 벡터화를 수행하기 위해 새롭게 설계된 계층적 이미지 벡터화 네트워크로, raster 이미지 입력을 바탕으로 가변 길이의 아이콘 벡터 그래픽을 생성하도록 특화되어 있습니다. 기존의 이미지 벡터화 방법들이 겪었던 문제들을 해결합니다.

- **Technical Details**: DeepIcon은 Scalable Vector Graphics (SVG)를 직접 생성하며, differentiable rasterizer에 의존하지 않습니다. 이는 고급 이미지 인식 및 처리 방식을 채택하여, 여러 개의 경로를 포함하는 SVG를 생성할 수 있는 정확한 SVG 디코더를 제안합니다. 또한, CLIP 기반의 이미지 인코더를 사용합니다.

- **Performance Highlights**: 실험 결과에 따르면 DeepIcon은 기존 최첨단 벡터화 접근 방식들을 넘어, 토폴로지 유사성 및 기하학적 정확성을 보존하는 데 있어 뛰어난 성능을 발휘합니다.



### Automated Proof Generation for Rust Code via Self-Evolution (https://arxiv.org/abs/2410.15756)
- **What's New**: 본 논문에서는 SAFE(Self-evolving Automated proof generation)라는 새로운 프레임워크를 소개한다. SAFE는 Rust 코드의 자동 증명을 생성할 수 있도록 인적 작성증명의 부족 문제를 극복한다.

- **Technical Details**: SAFE는 데이터 합성과 미세 조정(fine-tuning)이 협력하여 모델의 능력을 향상시키는 자기 진화(self-evolving) 사이클을 수립한다. 또한, SAFE는 오류가 있는 증명의 대량을 재활용하여 모델의 자기 디버깅 능력을 훈련시킨다.

- **Performance Highlights**: SAFE는 자동 생성된 증명을 통해 성능을 크게 향상시켜, 전문 인력이 만든 벤치마크에서 70.50%의 정확도를 달성하며, 이는 GPT-4o의 24.46% 성능을 크게 초월하는 수치이다.



### Unleashing the Potential of Vision-Language Pre-Training for 3D Zero-Shot Lesion Segmentation via Mask-Attribute Alignmen (https://arxiv.org/abs/2410.15744)
- **What's New**: 최근 의료 영상-언어 사전 훈련 모델의 발전으로 제로샷 질병 인식(zero-shot disease recognition) 분야에서 큰 진전을 이룬 반면, 이미지 수준 지식을 픽셀 수준 작업인 3D CT 스캔의 병변 분할(lesion segmentation)으로 이전하는 것은 여전히 중요한 도전 과제로 남아있습니다. 본 논문에서는 3D 제로샷 병변 분할을 위해 특별히 설계된 새로운 다중 스케일 병변 수준 마스크-속성 정렬(framework)을 제안하는 Malenia를 소개합니다.

- **Technical Details**: Malenia는 다중 스케일 마스크 표현을 활용하여 다양한 병변 영역을 포착하고, 병변의 세밀한 시각적 특징을 텍스트 임베딩과 매칭하여 대조적 사전 훈련 작업과 픽셀 수준 밀집 예측(dense prediction) 작업 간의 간극을 효과적으로 연결합니다. 또한, 우리는 시각적 및 텍스트적 특징을 상호 보완적인 정보로 강화하는 Cross-Modal Knowledge Injection(CMKI) 모듈을 설계하여 분할 결과 생성을 효과적으로 안내합니다.

- **Performance Highlights**: Malenia는 MSD, KiTS23, 그리고 실제 사례에서 수집된 데이터셋에서 12가지 병변 카테고리에 대해 종합적으로 평가하였으며, 제로샷 설정에서 가장 최신의 방법들과 비교하여 일관되게 우수한 성능을 보였습니다. 특히 Malenia는 기존 방법들보다 월등히 더 나은 성능을 확인시켜 주었습니다.



### Who's Who: Large Language Models Meet Knowledge Conflicts in Practic (https://arxiv.org/abs/2410.15737)
Comments:
          Accepted to EMNLP 2024 Findings

- **What's New**: 이번 연구에서는 Retrieval-augmented generation (RAG) 방법을 사용하여 사전 학습된 언어 모델의 메모리 한계를 극복하려고 하지만, 정보 충돌(conflict) 문제를 해결하기 위한 새로운 벤치마크 데이터 세트인 WhoQA를 소개합니다.

- **Technical Details**: WhoQA는 동일한 이름을 가진 Wikipedia 엔티티에 대한 공통 속성을 묻는 질문을 통해 지식 충돌을 유도합니다. 데이터 세트는 5000개의 질문으로 구성되어 있으며, 최대 8개의 고유한 답변을 포함할 수 있습니다. 질문은 (q, A, S, C) 형태의 쿼드플렛으로 나타내어 집니다.

- **Performance Highlights**: 실험 결과, WhoQA 질문의 단순함에도 불구하고 지식 충돌이 RAG 설정에서 LLM의 성능을 크게 저하시킨다고 보고했습니다.



### Reducing annotator bias by belief elicitation (https://arxiv.org/abs/2410.15726)
- **What's New**: 이 연구에서는 많은 특정 annotators 수나 데이터 인스턴스 수에 대한 요구 없이 주석에서의 편향을 다루기 위한 간단한 방법을 제안합니다. 이 방법은 annotators가 다른 annotators의 판단에 대한 신념을 보고하도록 요청합니다. 이는 신념이 보다 대표적이고 덜 편향된 레이블을 생성할 수 있다는 가설에 기반합니다.

- **Technical Details**: 이 연구는 두 개의 통제된 설문조사 실험을 통해 검토되었으며, 1,590명의 민주당원 및 공화당원이 특정 진술을 평가하고 다른 참가자의 판단에 대한 신념을 보고했습니다. Experiment 1은 대표 그룹의 응답에 대한 신념을 이끌어내기 위해 재정적 인센티브를 사용했으며, Experiment 2에서는 민주당원과 공화당원의 응답에 대한 신념을 개별적으로 이끌어냈습니다.

- **Performance Highlights**: 결과는, 두 그룹의 annotators 간의 체계적인 차이로 정의된 편향이 판단 대신 신념을 요구했을 때 일관되게 줄어드는 것을 보여주었습니다. 이 연구에서 제안된 방법은 AI 시스템의 일반화 가능성을 향상시키고, 무시된 사회-인구 집단에 대한 해를 방지할 수 있는 잠재력을 가지고 있습니다.



### Timetable Nodes for Public Transport Network (https://arxiv.org/abs/2410.15715)
- **What's New**: 이 논문에서는 시간 의존형 교통 네트워크에서의 빠른 경로 탐색을 위한 새로운 방법론을 제안합니다. 기존의 그래프 기반 접근 방식에 비해 계산 기하학(computational geometry)에서의 최적화 기법을 활용하여 탐색 프로세스를 가속화합니다.

- **Technical Details**: 제안된 방법은 타임테이블 노드(timetable nodes, TTN)라는 새로운 전처리 단계를 도입합니다. TTN은 Combined Search Tree (TTN-CST)와 Fractional Cascading (TTN-FC)의 두 가지 버전으로 구현됩니다. 이 방법들은 노드에서 새로운 노드에 도달하는 비대칭 복잡도를 $O(k	imes 	ext{log}|C|)$에서 $O(k + 	ext{log}(k) + 	ext{log}(|C|))$으로 감소시킵니다.

- **Performance Highlights**: 실험 결과, 이 전처리 단계가 고밀도 그래프에서 성능을 유의미하게 향상시킴을 보여주었습니다. 제안된 방법은 시간 의존형 네트워크 및 다른 경로 탐색 알고리즘에 통합될 수 있습니다.



### Offline reinforcement learning for job-shop scheduling problems (https://arxiv.org/abs/2410.15714)
- **What's New**: 본 논문에서는 복잡한 제약이 있는 조합 최적화 문제를 해결하기 위해 설계된 새로운 오프라인 RL 기법을 소개합니다. 상태 공간은 이종 그래프(heterogeneous graph)로 표현되고, 액션 스페이스(action space)는 가변적이며, 액션은 엣지 속성으로 인코딩됩니다.

- **Technical Details**: 이 방법은 조잡한 조합 최적화 문제를 다루고 있으며, 특히 작업관리 스케줄링(job-shop scheduling) 및 유연한 작업관리 스케줄링(flexible job-shop scheduling) 벤치마크에서 테스트되었습니다. 이종 그래프를 통해 복잡한 의존관계를 정확하게 캡처하고, 액션의 선택을 엣지 속성을 통해 표현함으로써, 다양한 최적화 문제를 효과적으로 해결합니다.

- **Performance Highlights**: 제안된 방법은 다섯 가지 잘 알려진 벤치마크에서 테스트되어 기존의 최신 기법들에 비해 뛰어난 성능을 보였습니다. 특히, 온전한 전문가 솔루션을 모방하면서도 예상 보상을 균형 있게 조정하는 새로운 손실 함수를 제안했습니다.



### PALMS: Plane-based Accessible Indoor Localization Using Mobile Smartphones (https://arxiv.org/abs/2410.15694)
Comments:
          7 pages, 3 figures, accepted to the 14th International Conference on Indoor Positioning and Indoor Navigation (IPIN) 2024, Best Presentation Award

- **What's New**: PALMS는 기존의 실내 내비게이션 시스템의 한계를 뛰어넘는 혁신적인 방법을 선보입니다. 이 시스템은 오직 디자인된 평면도(floor plan)만을 이용하여 동적 위치 추정을 수행하며, 입장 전 환경의 정보 수집(fingerprinting)이 필요하지 않습니다.

- **Technical Details**: PALMS는 LiDAR 및 파티클 필터(particle filter)를 이용하여 실내에서의 위치를 동적으로 결정합니다. CES 제약 조건을 활용하고, 주된 방향을 매칭하여 위치의 공간 확률 분포를 생성합니다. 이는 특히 사용자 환경의 정확한 특징을 반영하여 더 나은 정확성과 빠른 수렴 시간을 구현합니다. 사용자 위치에 대한 프로브 능력은 3D 스캔 وقد بتقنية حتى يخدم العديد من التطبيقات.

- **Performance Highlights**: PALMS는 전통적인 방법에 비해 평균 RMSE가 6.7배 낮은 정확도를 보였으며, 1미터 오차 내에서의 위치의 비율은 4.9배 더 높았습니다. 이는 실내 내비게이션에 있어 훨씬 더 효율적인 접근법을 제공함을 나타냅니다.



### NetSafe: Exploring the Topological Safety of Multi-agent Networks (https://arxiv.org/abs/2410.15686)
- **What's New**: 본 논문은 다중 에이전트 네트워크의 안전성을 탐구하는 새로운 연구 방향인 Topological Safety를 제안하며, 네트워크의 여러 위협에 대한 반응을 연구한다.

- **Technical Details**: 저자들은 RelCom이라는 기계적 상호작용 메커니즘 및 NetSafe라는 포괄적 프레임워크를 소개하여 에이전트 간 안전성을 통합적으로 연구하는 기반을 마련하였다. 이들은 다양한 위협을 고려하여 다중 에이전트 시스템의 수학적 정의를 형식화하고, 네트워크 구조의 안전성에 대한 실험을 수행하였다.

- **Performance Highlights**: 연구를 통해 확인된 주요 결과는 높은 연결성을 가진 네트워크가 공격에 더 취약하고, 특정 구조에서 성능 저하가 29.7%에 달한다는 것이다. 또한, Agent Hallucination과 Aggregation Safety와 같은 새로운 현상을 발견하였다.



### Revealing and Mitigating the Local Pattern Shortcuts of Mamba (https://arxiv.org/abs/2410.15678)
- **What's New**: 최근 Mamba라는 고급 모델이 도입되었습니다. 이 모델은 State Space Models(SSMs)에 기반하여 단순히 Attention 메커니즘에 비해 선형 복잡성과 상수 메모리 요구사항을 제공하며, 성능이 주목받고 있습니다. 하지만, Mamba는 지역 정보를 잘 처리하지만 분산된 정보를 처리하는 데 어려움을 겪고 있습니다.

- **Technical Details**: Mamba 모델은 Selective State Space Model로, 이전 SSMs와의 주요 차이점은 모델의 동적 상태가 시간에 따라 효율적으로 업데이트된다는 점입니다. 이를 위해 Mamba는 입력에 따라 변화하는 특별한 학습 가능한 선형 층을 도입합니다. 이러한 변화는 상황에 맞게 매개 변수를 조정하여 Mamba가 더 유연한 시퀀스 모델링을 가능하게 합니다.

- **Performance Highlights**: Mamba 모델은 단지 4M의 추가 매개 변수를 포함시켰을 뿐임에도 불구하고, 고정보 밀도 합성 작업에서 0에서 80.54점으로의 성과 향상을 이뤘습니다. 이는 Mamba 모델과 Attention 기반 모델 간의 성능 차이를 줄이는 데 기여하였습니다.



### Learning to Generate and Evaluate Fact-checking Explanations with Transformers (https://arxiv.org/abs/2410.15669)
Comments:
          Forthcoming in Engineering Applications of Artificial Intelligence

- **What's New**: 디지털 플랫폼이 지배하는 시대에 정보의 진위를 평가할 수 있는 솔루션 개발의 필요성이 강조되고 있습니다. 본 연구는 Explainable Artificial Intelligence (XAI) 분야에 기여하기 위해, 결정을 설명하는 인간 친화적인 설명을 생성하는 transformer 기반의 사실 확인 모델을 개발했습니다.

- **Technical Details**: 자동 평가 모델을 통해 사실 확인 결정에 대한 설명을 	exttt{(self)-contradiction}, 	exttt{hallucination}, 	exttt{convincingness}, 	exttt{overall quality}와 같은 다양한 차원에서 평가할 수 있습니다. 인간 중심의 평가 방법과 특수화된 데이터셋을 개발하여 AI 생성 설명을 인간 판단과 일치시키는 필요성을 강조하고 있습니다. 또한 메트릭 학습 모델을 통해 효율성을 증가시키고 방대한 수동 평가에 대한 의존도를 줄이기 위한 첫걸음을 제시하고 있습니다.

- **Performance Highlights**: 실험 결과, 최고의 생성 모델의 	extsc{ROUGE-1} 점수는 47.77로, 고품질 증거가 제공될 때 사실 확인 설명을 생성하는 데 있어 우수한 성능을 보였습니다. 또한 최고의 메트릭 학습 모델은 	exttt{(self)-contradiction} 및 	exttt{hallucination}과 같은 객관적인 차원에서 인간 판단과 중간 정도의 강한 상관 관계를 나타내며, Matthews Correlation Coefficient (MCC)가 약 0.7에 도달했습니다.



### RAC: Efficient LLM Factuality Correction with Retrieval Augmentation (https://arxiv.org/abs/2410.15667)
- **What's New**: 본 논문은 Retrieval Augmented Correction (RAC)이라는 저지연 후처리 방법을 소개하여 Large Language Models (LLMs)의 사실 정확도(factual performance)를 향상시키는 데 기여합니다. RAC는 추가적인 fine-tuning 없이도 사용 가능하며, LLM의 출력을 기본적인 사실들(atomic facts)로 분해한 후, 검색된 내용(retrieved content)을 사용하여 검증 및 수정하는 과정을 포함합니다.

- **Technical Details**: RAC는 LLM의 출력물을 기본적인 몇 가지 사실로 분해하고, 이 사실들을 검색된 지식을 통해 확인하거나 수정한 후 LLM의 출력을 수정하는 방법입니다. 이 접근 방식은 Retrieval-Augmented Generation (RAG)의 후처리 구성 요소로 볼 수 있으며, 두 개의 사실성 평가 데이터셋을 통해 기존 방법들에 비해 최대 30% 향상된 성능을 보여주었습니다. RAC는 검색을 한 번 수행하고 수정을 한 번 수행하여 이전 방법들에 비해 지연(latency)을 크게 감소시켰습니다.

- **Performance Highlights**: RAC는 LLM의 사실성 성능을 향상시키는 데 있어 유효성과 견고성을 보여주며, RAG와 통합한 경우와 그렇지 않은 경우 모두에서 30% 이상의 성능 향상을 달성하였습니다. 특히, 일부 사례에서는 RAG 없이도 RAG와 함께 사용할 때보다 성능이 더 우수한 결과를 나타내어 우수성을 입증하였습니다.



### Resource-Efficient Medical Report Generation using Large Language Models (https://arxiv.org/abs/2410.15642)
- **What's New**: 이 연구에서는 흉부 X-레이 이미지를 위한 자동 의료 보고서 생성을 위한 새로운 프레임워크를 제안합니다. 이 프레임워크는 비전 기반의 대형 언어 모델(LLM)을 활용하여, 경량화된(solution) 방식으로 기존의 방법들과 비교해 우수한 성능을 달성하고 있습니다.

- **Technical Details**: 제안된 방법은 비전 인코더, 대형 언어 모델, 매핑 네트워크로 구성됩니다. 의료 관련 CLIP(MedCLIP)을 사용하여 시각적 임베딩(prefix embeddings)을 추출하고, 이를 경량화된 매핑 네트워크를 통해 언어 모델의 공간으로 변환합니다. 여기서 prefix tuning을 사용하여 LLM을 미세 조정하지 않고 성능을 향상시킵니다.

- **Performance Highlights**: Qwen1.5 LLM이 GPT-2 모델보다 의료 보고서 생성에서 더 우수한 NLG 메트릭(예: Bleu 점수)을 기록했습니다. 제안된 방법은 이전의 대형 LLM 기반 솔루션보다 성능이 뛰어나며, 자원 효율성을 확인하였습니다.



### Selecting Influential Samples for Long Context Alignment via Homologous Models' Guidance and Contextual Awareness Measuremen (https://arxiv.org/abs/2410.15633)
- **What's New**: 이 연구는 긴 맥락을 가진 명령을 효과적으로 처리하기 위한 고품질 데이터셋 구축의 필요성을 강조합니다. GATEAU라는 새로운 프레임워크를 제안하여 중요하고 품질 높은 샘플을 식별합니다.

- **Technical Details**: GATEAU는 Homologous Models' Guidance (HMG) 및 Contextual Awareness Measurement (CAM)라는 두 가지 방법을 활용하여 긴 의존 관계(long-range dependency relations)가 풍부한 샘플을 찾아냅니다. HMG는 서로 다른 맥락 윈도우를 가진 두 개의 동형 모델의 반응의 perplexity 점수를 측정하여 그 난이도를 평가합니다. CAM은 모델의 주의(attention)가 중요한 구간에 집중되고 있는지를 평가하여 긴 입력 맥락의 이해 난이도를 측정합니다.

- **Performance Highlights**: GATEAU를 통해 선택된 샘플로 훈련된 모델은 명령 수행 및 긴 맥락 이해 능력이 향상됩니다. 실험 결과, 이 프레임워크는 긴 의존 관계가 있는 샘플을 효과적으로 식별할 수 있음을 보여주었습니다.



### Towards Kriging-informed Conditional Diffusion for Regional Sea-Level Data Downscaling (https://arxiv.org/abs/2410.15628)
- **What's New**: 본 논문에서는 Kriging-informed Conditional Diffusion Probabilistic Model (Ki-CDPM)을 제안합니다. 이 모델은 기후 데이터의 공간 변동성을 포착하면서 미세한 특성을 보존하는 새로운 방법론입니다.

- **Technical Details**: 제안된 Ki-CDPM은 Kriging 보간법을 활용하여 고해상도 기후 변수를 예측합니다. Variogram-Based Regularization을 도입하여 지역적 과정에서의 공간 변동성을 캡처하고, 다운스케일된 데이터의 물리적 일관성을 강화합니다.

- **Performance Highlights**: 실험 결과, Ki-CDPM은 최신 머신러닝 모델들과 비교해 정확도가 우수하며, 실제 기후 데이터 세트에서의 성능이 향상되었습니다.



### Improving Parallel Program Performance Through DSL-Driven Code Generation with LLM Optimizers (https://arxiv.org/abs/2410.15625)
Comments:
          26 pages, 8 figures

- **What's New**: 이 논문에서는 LLM(대형 언어 모델)을 기반으로 한 맵퍼(mapper) 생성 자동화를 통해 병렬 프로그래밍의 성능을 극대화하는 새로운 접근법을 제안합니다. 이 방법은 전문가의 설계를 초과하는 성능 향상을 달성하며, 개발 시간을 단 몇 분으로 단축시킵니다.

- **Technical Details**: 이 연구는 도메인 특화 언어(DSL)를 도입하여 복잡한 저수준 C++ 시스템 프로그래밍의 세부정보를 추상화합니다. DSL은 LLM이 독립적인 모듈 구성 요소를 생성할 수 있는 구조적 검색 공간을 정의하며, 이를 통해 코드 생성의 문맥 의존성이 감소합니다. 또한, 규칙 기반 최적화를 위해 강화 학습을 적용하여 성능을 최적화하는 방법을 제안합니다.

- **Performance Highlights**: LLM을 사용하여 생성된 맵퍼는 전문가가 작성한 맵퍼보다 최대 1.34배의 속도 향상을 기록하였으며, 병렬 행렬 곱셈 알고리즘에서는 최대 1.31배의 성능 향상을 달성하였습니다. 실험 결과, DSL을 통해 맵퍼 코드 생성 성공률이 0%에서 80%로 크게 향상되었습니다.



### Reinforced Imitative Trajectory Planning for Urban Automated Driving (https://arxiv.org/abs/2410.15607)
Comments:
          19 pages, 9 figures

- **What's New**: 이 논문은 도시 자동 운전(Urban Automated Driving)에서의 경로 계획(Trajectory Planning) 문제를 해결하기 위해 새로운 강화 학습 기반 경로 계획 방법인 Reinforced Imitative Trajectory Planning (RITP)을 제안합니다. 기존의 방법들이 다중 단계 계획에 제한적이었다면, RITP는 강화를 통해 이러한 제약을 극복하고 있습니다.

- **Technical Details**: RITP는 강화 학습( RL )과 모방 학습( Imitation Learning, IL )을 결합하여 다중 단계 계획을 구현합니다. 이 방법은 경험적 데이터를 통해 동작하며, 트랜스포머 기반의 보상 함수(Bayesian reward function)을 개발하여 도시 시나리오에서 효과적인 보상 신호를 제공합니다. 노이즈를 포함한 트랜스포머 기반 가치 함수(Trajectory Value Function)를 통해 상태와 경로 공간을 탐색하며 계획의 폐쇄 루프 효과를 평가합니다.

- **Performance Highlights**: 제안된 방법들은 대규모 실제 도시 자동 운전 nuPlan 데이터셋을 통해 검증되었습니다. 실험 결과, 기존 방법들과 비교해 폐쇄 루프 지표에서 상당한 우수성을 보였습니다. 이 연구는 안전성 및 해석 가능성을 높이기 위한 하이브리드 기반 경로 계획 프레임워크를 통합하여 더욱 효과적인 해결책을 제공합니다.



### Deep Active Learning with Manifold-preserving Trajectory Sampling (https://arxiv.org/abs/2410.15605)
- **What's New**: 본 논문에서는 Active Learning (AL)에서 라벨이 없는 데이터를 선택하기 위해 Manifold-Preserving Trajectory Sampling (MPTS)라는 새로운 방법론을 제안합니다. 이는 모델이 라벨이 있는 데이터가 학습한 feature 공간을 통해 더 정확한 manifold를 표현하도록 하는 데 중점을 둡니다.

- **Technical Details**: MPTS는 labeled 예제에서 학습한 feature distribution을 정규화함으로써, 라벨이 있는 데이터로 인한 bias를 효과적으로 보정합니다. 또한, 최적화 경로에서 의미 있는 파라미터 샘플링을 위해 MMD (Maximum Mean Discrepancies)를 사용하여 여러 지점에서 모델 파라미터를 평균화합니다. 이에 따라 모델 파라미터는 posterior distribution의 다양한 모드를 포착할 수 있습니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋에서 MPTS는 기존의 최신 Active Learning 방법보다 항상 우수한 성능을 보입니다. 이러한 결과는 MPTS의 효과성을 강조하며, 다양한 유형의 데이터에 적용 가능한 신뢰할 수 있는 방법임을 시사합니다.



### A Comprehensive Comparative Study of Individual ML Models and Ensemble Strategies for Network Intrusion Detection Systems (https://arxiv.org/abs/2410.15597)
- **What's New**: 이 연구는 네트워크 침입 감지 시스템(IDS)에서 여러 개별 모델과 간단 및 고급 앙상블 방법에 대한 포괄적인 평가를 수행합니다. 특히, 다양한 AI 모델의 성능을 평가하기 위한 새로운 앙상블 학습 프레임워크가 도입되어 있습니다.

- **Technical Details**: 제안된 프레임워크는 입력 데이터 세트를 로드하고, 개별 모델 및 앙상블 방법을 훈련시키며, 평가 메트릭을 생성하는 과정으로 구성됩니다. 본 연구에서는 RoEduNet-SIMARGL2021 및 CICIDS-2017의 두 개의 고유한 네트워크 침입 데이터 세트를 사용하여 다양한 알고리즘을 평가합니다.

- **Performance Highlights**: 모든 설정에서 학습의 효율성을 보여주는 결과를 제시하며, F1 점수를 기준으로 다양한 방법을 비교하여 효과적인 IDS 모델을 식별합니다. 이 연구 결과는 AI 모델들의 효율성을 증가시키는 데 기여할 것입니다.



### AMPLE: Emotion-Aware Multimodal Fusion Prompt Learning for Fake News Detection (https://arxiv.org/abs/2410.15591)
- **What's New**: 이번 연구에서는 감정 인식 기반의 다중 모달 융합 프롬프트 학습(Emotion-Aware Multimodal Fusion Prompt Learning, AMPLE) 프레임워크를 소개하여 가짜 뉴스 탐지의 효율성을 높이는 새로운 접근법을 제시합니다.

- **Technical Details**: AMPLE 프레임워크는 감정 분석 도구(Sentiment Analysis Tool, SAT)를 활용하여 텍스트에서 감정 요소를 추출하고, Multi-Head Cross-Attention (MCA) 기법을 통해 다중 모달 데이터를 통합합니다. 이 프레임워크는 기존의 대량 주석 데이터에 대한 의존도를 줄이며, 고정된 텍스트 프롬프트와 조정 가능한 벡터를 통합한 하이브리드 프롬프트 템플릿을 활용합니다.

- **Performance Highlights**: AMPLE 프레임워크는 PolitiFact 및 GossipCop의 두 공개 데이터셋에서 평가되었으며, F1과 정확도 면에서 기존 방법보다 우수한 성능을 보였습니다. 특히 기계 학습(ML)과 딥 러닝(DL) 기술을 통합한 접근법은 감정 요소의 긍정적 상관관계를 입증했습니다.



### OpenMU: Your Swiss Army Knife for Music Understanding (https://arxiv.org/abs/2410.15573)
Comments:
          Resources: this https URL

- **What's New**: OpenMU-Bench라는 대규모 벤치마크 세트를 소개했으며, 이는 음악 모달리티를 이해하기 위한 다중 모달 언어 모델(Multimodal Large Language Models, MLLMs) 훈련 시 데이터 부족 문제를 해결하기 위해 설계되었습니다.

- **Technical Details**: OpenMU는 음악 클립을 이해하기 위해 훈련된 MLLM이며, 음악 캡셔닝, 추론, 다중 선택 질문 응답 작업에서 MU-Llama와 같은 기존 모델을 초월하는 성능을 보였습니다. OpenMU-Bench는 약 100만 개의 음악 이해 데이터 예제를 포함하고 있으며, 다양한 음악 이해 작업을 다룹니다.

- **Performance Highlights**: OpenMU는 OpenMU-Bench를 통해 훈련되었으며, 음악 캡셔닝 및 다중 선택 질문 응답 작업에서 우수한 성능을 발휘했습니다. 또한 OpenMU와 OpenMU-Bench는 모두 오픈 소스로 제공되어 향후 음악 이해 및 창의적인 음악 제작의 효율성을 높이는 데 기여할 것으로 기대됩니다.



### Leveraging Retrieval-Augmented Generation for Culturally Inclusive Hakka Chatbots: Design Insights and User Perceptions (https://arxiv.org/abs/2410.15572)
Comments:
          Accepted to IEEE RASSE 2024

- **What's New**: 이 연구는 대만 Hakka 문화의 보호 및 홍보를 위한 혁신적인 접근 방식을 제시하며, Retrieval-Augmented Generation (RAG) 기술이 향상된 챗봇 개발을 통해 이루어집니다.

- **Technical Details**: RAG 기술은 전통적인 대형 언어 모델(LLMs)의 제한을 보완해 정확하고 맥락이 풍부한 응답을 가능하게 합니다. 이 챗봇은 Hakka 전통과 언어, 관습을 반영하기 위해 특별히 큐레이션된 문화 데이터를 활용하여 지식 기반을 강화합니다.

- **Performance Highlights**: 사용자 만족도와 참여도가 눈에 띄게 향상되었으며, 챗봇이 Hakka 문화와의 더 깊은 연결을 촉진하는 데 효과적임을 보여줍니다. RAG 기술은 사용자 경험을 향상시키고 민족 주류화 및 문화 축하의 중요한 도구로 활용될 가능성을 지니고 있습니다.



### Stacking Small Language Models for Generalizability (https://arxiv.org/abs/2410.15570)
- **What's New**: 최근 대형 언어 모델(LLMs)의 강력한 성능을 다양한 자연어 벤치마크에 일반화할 수 있는 잠재력을 보여주었습니다. 그러나 이러한 모델의 크기 때문에 훈련과 추론이 비쌀 뿐만 아니라 자원이 제한된 환경에서 사용하기에는 비현실적입니다. 이 논문은 작은 언어 모델(SLM)을 쌓아서 사용하는 새로운 방법인 FSLM(Fine-tuning Stacks of Language Models)을 소개합니다.

- **Technical Details**: FSLM은 다수의 SLM을 연결하여 각 모델이 특정 작업을 수행하도록 세분화함으로써 높은 수준의 추론을 여러 낮은 수준의 단계로 나누는 방식을 채택했습니다. FSLM은 높은 비용의 훈련과 추론을 줄이면서 모델의 해석 가능성을 향상시킬 수 있습니다. 이를 통해 SLM들이 상호작용하며 자연어로 의사소통하게 됩니다.

- **Performance Highlights**: FSLM은 Alpaca 데이터셋을 활용하여 훈련되었고, 여러 자연어 벤치마크에서 기존 Pythia 및 Flan 모델들을 상대로 성능을 평가한 결과, FSLM 스택이 상대적으로 경량적인 LLM의 효과적인 대안으로 자리잡을 가능성을 보여줍니다. 초기 결과는 FSLM이 비슷한 크기의 모델들에 비해 성능이 향상되었음을 나타냅니다.



### Pruning Foundation Models for High Accuracy without Retraining (https://arxiv.org/abs/2410.15567)
Comments:
          Accepted by EMNLP 2024 findings

- **What's New**: 본 논문에서는 대형 언어 모델(LLMs)의 효과적인 포스트-트레이닝 프루닝(post-training pruning) 방법을 제안했습니다. 기존의 프루닝 기법이 데이터를 재학습해야 하는 문제를 해결하기 위해 레이어 단위로 다중 가중치를 동시에 프루닝하는 새로운 접근 방식을 소개합니다.

- **Technical Details**: 제안된 방법론은 다중 제거 문제(Multiple Removal Problem, MRP)를 직접 공식화하고, 이를 위한 최적 솔루션을 도출하였습니다. 이를 바탕으로 비구조적(unstructured) 및 반구조적(semi-structured) 스파시티(sparsity)를 위한 포스트-트레이닝 프루닝 알고리즘을 설계했습니다.

- **Performance Highlights**: 광범위한 실험을 통해 다양한 LLM 패밀리에서 SOTA(SOTA) 기준과 비교하여 우수한 성능을 입증했습니다. 예를 들어, LLaMA2-70B 모델의 경우, 2:4 스파시티 하에 4.278의 perplexity를 기록하여 SparseGPT의 5.698을 초월했습니다.



### Bayesian Concept Bottleneck Models with LLM Priors (https://arxiv.org/abs/2410.15555)
- **What's New**: 이 논문은 Concept Bottleneck Models (CBMs)에 대한 새로운 접근법인 BC-LLM을 소개합니다. 이 모델은 인공적인 개념을 정의하고 대량의 LLM(대형 언어 모델)을 사용하여 그들의 값을 추출하며, Bayesian(베이지안) 프레임워크 내에서 개념을 반복적으로 탐색합니다. 이를 통해 인간 전문가의 사전 정의 및 수작업 주석 없이도 효율적인 모델 학습이 가능해집니다.

- **Technical Details**: BC-LLM은 LLM을 사용하여 개념을 정의하고, 포스터리어 샘플링(Posterior Sampling) 중에 탐색할 개념을 제안하며, 관찰된 데이터를 주석 처리하는 기능을 수행합니다. 이 방법은 Bayesian 원리에 기반하여 통계적으로 원칙적이며 불확실성 정량화(Uncertainty Quantification)를 제공합니다. 이 모델은 텍스트, 이미지 및 표 형식 데이터와 같은 다양한 모드에서 적용 가능합니다.

- **Performance Highlights**: 실험 결과, BC-LLM은 비교 모델과 블랙 박스 모델을 능가하며, 관련 개념으로 더 빠르게 수렴하고, 잘못 상관된 개념에서 멀어지며, OOD(Out-Of-Distribution) 샘플에 대해 더 강건함을 나타냈습니다. 실제 병원 데이터 과학팀과 협력하여 기존 ML 모델을 수정할 때, clinicians는 BC-LLM이 보다 해석 가능하고 실행 가능한 CBMs을 제공한다고 평가했습니다.



### A Plug-and-Play Fully On-the-Job Real-Time Reinforcement Learning Algorithm for a Direct-Drive Tandem-Wing Experiment Platforms Under Multiple Random Operating Conditions (https://arxiv.org/abs/2410.15554)
Comments:
          63 pages, 32 figures

- **What's New**: 비행 제어의 난이도를 극복하기 위해 Concerto Reinforcement Learning Extension (CRL2E) 알고리즘이 개발되었습니다. 이 알고리즘은 물리 기반 규칙 정책 작곡기 전략과 경량화 네트워크를 포함하여 실시간 제어에 최적화되어 있습니다.

- **Technical Details**: CRL2E 알고리즘은 무작위 운영 조건에서 비선형 및 불안정한 공기역학적 간섭에 대응하기 위한 강화 학습 알고리즘입니다. 6개의 다양한 조건에서 7개 알고리즘과 비교 실험을 통해 CRL2E의 성능을 검증하였습니다.

- **Performance Highlights**: CRL2E 알고리즘은 처음 500단계 안에 안전하고 안정적인 훈련을 달성하며, Soft Actor-Critic, Proximal Policy Optimization, Twin Delayed Deep Deterministic Policy Gradient 알고리즘에 비해 추적 정확도가 14배에서 66배 향상되었습니다. 또한 다양한 무작위 운영 조건에서 CRL에 비해 8.3%에서 60.4%까지 추적 정확도가 개선되었으며, 수렴 속도는 CRL 알고리즘보다 36.11%에서 57.64% 빨랐습니다.



### GRS: Generating Robotic Simulation Tasks from Real-World Images (https://arxiv.org/abs/2410.15536)
- **What's New**: 이 논문에서는 GRS (Generating Robotic Simulation tasks)라는 새로운 시스템을 소개하며, 이는 로보틱스, 컴퓨터 비전, AR/VR에서 실제 환경에서 가상 시뮬레이션으로 변환하는 문제를 해결하기 위한 것입니다. GRS는 단일 실세계 RGB-D 관찰로부터 디지털 트윈 시뮬레이션을 생성하며, 가상 에이전트 훈련을 위한 다양한 해결 가능한 작업을 포함합니다.

- **Technical Details**: GRS는 세 가지 단계로 작동합니다: 1) SAM2를 이용한 장면 이해 및 객체 분할, 2) 식별된 객체와 시뮬레이션 준비 자산 간의 매칭, 3) 맥락에 맞는 로봇 작업 생성. GRS의 핵심 혁신은 두 가지 주요 구성 요소로, 첫째로 각 시뮬레이션에 맞춤형 테스트가 동반되는 이중 생성 프로세스를 활용합니다. 둘째로, LLM 기반 라우터 시스템을 도입하여 시뮬레이션 성능을 분석하고 최적화를 진행합니다.

- **Performance Highlights**: 실험을 통해 GRS는 단일 RGB-D 관찰로부터 real-to-sim 작업에 성공할 수 있음을 보여주었습니다. 장면 객체 식별 과정의 높은 정확성과 VLM을 통한 입력 환경과 더 밀접하게 일치하는 작업 생성을 입증했습니다. 또한 라우터가 로봇 정책에 효과적인 시뮬레이션을 생성하는 비율을 개선함을 보여주었습니다.



### M-RewardBench: Evaluating Reward Models in Multilingual Settings (https://arxiv.org/abs/2410.15522)
Comments:
          16 pages, 6 figures, 10 tables. Website: this https URL

- **What's New**: 본 연구에서는 멀티링구얼(multilingual) 환경에서의 보상 모델(reward models, RMs)에 대한 시스템적인 평가를 수행하였습니다. 이를 위해 23개의 다양한 언어를 포함한 최초의 멀티링구얼 RM 평가 벤치마크인 M-RewardBench를 구축하였습니다.

- **Technical Details**: M-RewardBench는 2.87천 개의 선호(preference) 인스턴스로 구성되어 있으며, RMs의 채팅(chat), 안전성(safety), 추론(reasoning), 번역 번역(translation) 능력을 테스트합니다. 이 연구에서는 다양한 보상 모델을 M-RewardBench에서 엄격하게 평가하여, 비영어(non-English) 언어와 영어(Eglish) 언어 간의 성능 차이를 발견하였습니다.

- **Performance Highlights**: 연구 결과, RMs의 선호는 언어에 따라 실질적으로 변화하며, 번역 품질이 향상되면 RMs 성능도 개선된다는 것을 보여주었습니다. 또한 고자원(high-resource) 언어에 대한 성능이 더욱 우수함을 입증하였습니다. 본 연구에서 M-RewardBench 데이터셋 및 코드베이스를 공개하여 멀티링구얼 RM 평가에 대한 이해를 높이는데 기여하고자 하였습니다.



### Exploring Curriculum Learning for Vision-Language Tasks: A Study on Small-Scale Multimodal Training (https://arxiv.org/abs/2410.15509)
Comments:
          CoNLL BabyLM Challenge 2024 camera ready

- **What's New**: 이 논문은 Machine Learning의 제한된 데이터 환경에서 Curriculum Learning의 효과를 탐구하고, 여러 모델 유형 간의 성능 비교를 다룹니다. 특히, Multimodal 모델의 성능을 개선하기 위한 새로운 접근 방식을 제공합니다.

- **Technical Details**: 연구는 3가지 주요 변수를 평가합니다: (i) Curriculum Learning, (ii) 텍스트 전용 데이터로부터의 Pretraining, (iii) 모델 유형. 데이터를 쉽고 난이도에 따라 제시하는 Curriculum Learning 접근 방식을 사용하여 VLM(Vision-Language Models)의 성능 변화를 연구했습니다.

- **Performance Highlights**: Curriculum Learning이 Multimodal 평가에서 Non-Curriculum Learning 모델보다 성능 향상에 기여하며, 특히 텍스트 전용 Pretraining과 결합할 때 효과가 있는 것으로 나타났습니다. 텍스트 전용 작업에서는 더 적은 trainable parameters를 가진 모델에서 Curriculum Learning의 혜택이 나타났습니다.



### SEA: State-Exchange Attention for High-Fidelity Physics-Based Transformers (https://arxiv.org/abs/2410.15495)
Comments:
          Accepted in 38th Conference on Neural Information Processing Systems (NeurIPS 2024)

- **What's New**: 본 논문에서는 고급 Transformer 모델인 State-Exchange Attention (SEA) 모듈을 도입하여 다이나믹 시스템의 상태 변수를 더 정확하게 예측할 수 있게 하였습니다. 이 모듈은 여러 상태 변수 간의 정보를 교환할 수 있는 크로스 어텐션을 활용하여, 시스템의 물리적 관계를 보다 잘 캡처 할 수 있도록 설계되었습니다.

- **Technical Details**: SEA 모듈은 ViT (Vision Transformer)를 기반으로 하며, 이를 통해 수치 해석의 정확도를 향상시키고 고유한 물리적 상호작용을 고려할 수 있는 능력을 부여합니다. 이 모듈은 공간적으로 일관된 메쉬 임베딩을 생성하여 시계열 데이터의 종속성을 효과적으로 모델링합니다.

- **Performance Highlights**: SEA 모듈을 통합한 Transformer 모델은 기존 경쟁 모델들과 비교할 때 에러를 각각 88% 및 91% 줄이는 성과를 나타냈습니다. SEA 모듈만으로도 특정 상태 변수에 대해서는 97%의 에러 감소를 이끌어내며, 다양한 유체 역학 사례에서 60% 이상의 에러 감소를 달성한 최첨단 결과를 보였습니다.



### Generative AI Agents in Autonomous Machines: A Safety Perspectiv (https://arxiv.org/abs/2410.15489)
- **What's New**: 이 논문에서는 Generative Artificial Intelligence (AI)를 물리적 자율 기계에 통합할 때 발생하는 안전 요구사항을 조사합니다. 혁신적인 기술이 안전 우려를 동반하며, 자율 기계의 사용을 위한 포괄적인 안전 평가 시스템을 개발하자는 제안을 포함하고 있습니다.

- **Technical Details**: Generative AI는 데이터셋에서 패턴과 분포를 학습하여 새로운 데이터를 생성하는 모델을 의미합니다. 신경망 구조(transformer)와 확산 모델(diffusion models)의 발전으로 높은 품질의 데이터를 생성합니다. 자율 시스템에서의 적용은 내러티브 처리(NLP)와 로봇 조작 같은 분야에서 중요한 발전을 이루었습니다.

- **Performance Highlights**: Generative AI는 자율 시스템의 내비게이션, 인식 및 작업 계획에서 획기적인 진전을 이루었으며, 현실적인 데이터와 고급 비주얼을 생성하는데 성공하였습니다. 그러나 안전 위험이 크게 증가하는 자율 기계에 통합하면서 새로운 문제와 도전이 발생하게 됩니다.



### Mitigating Forgetting in LLM Supervised Fine-Tuning and Preference Learning (https://arxiv.org/abs/2410.15483)
- **What's New**: 본 논문은 전이 학습된 LLM(대형 언어 모델)의 후속 훈련(post-training) 방법을 제안합니다. 기존의 SFT(전문 지도 학습) 및 RLHF(강화학습된 인간 피드백) 또는 DPO(분포에 대한 우선 순위 학습) 단계를 순차적으로 수행하는 접근 방식에서 발생하는 비효율성을 이론적으로 증명하고, 이를 개선한 새로운 접근 방식을 제시합니다.

- **Technical Details**: 후속 훈련 과정에서 SFT와 RLHF/DPO 간의 균형을 맞추는 것이 중요합니다. 기존의 순차적 훈련 방식에서는 LLM이 SFT 과정에서 배운 내용을 점진적으로 잊어버리는 문제를 안고 있습니다. 이 연구에서는 이에 대한 공동 후속 훈련(joint post-training) 프레임워크를 제안하며, 이론적 수렴 보장을 제공합니다.

- **Performance Highlights**: 우리의 공동 후속 훈련 프레임워크는 기존의 순차적 후속 훈련 방식보다 성능이 우수하며, 유사한 계산 비용(computational cost)으로 구현됩니다.



### Data Augmentation via Diffusion Model to Enhance AI Fairness (https://arxiv.org/abs/2410.15470)
Comments:
          arXiv admin note: text overlap with arXiv:2312.12560

- **What's New**: 이번 논문은 AI 공정성을 향상시키기 위한 새로운 기술로서 Diffusion Model을 활용한 합성 데이터 생성의 가능성을 탐구합니다. Tabular Denoising Diffusion Probabilistic Model (Tab-DDPM)이라는 모델을 통해 데이터의 다양성과 허용성을 높이면서, 머신러닝 모델의 성능을 향상시키고자 합니다.

- **Technical Details**: Tab-DDPM은 다양한 특성을 처리할 수 있는 테이블 데이터용 diffusion model로, 다항 분포(diffusion)와 가우시안 분포(Gaussian diffusion)를 활용하여 숫자형 및 범주형 데이터를 효과적으로 처리합니다. 이 모델은 20,000, 100,000, 150,000개의 샘플을 생성하여 데이터 증강(data augmentation)을 수행합니다. 또한, AIF360 툴박스에서 샘플의 가중치를 재조정(reweighting)하여 편향을 줄이는 작업을 수행합니다.

- **Performance Highlights**: Tab-DDPM으로 생성된 합성 데이터는 5가지 전통적인 머신러닝 모델(Decision Tree, Gaussian Naive Bayes, K-Nearest Neighbors, Logistic Regression, Random Forest)에 대한 실험 결과에서 이진 분류의 공정성을 향상시키는 것으로 나타났습니다. 특히, RF(Random Forest) 성능이 개선되었고, 공정성을 평가하는 다섯 가지 지표에서 모두 긍정적인 결과를 보였습니다.



### AssemblyComplete: 3D Combinatorial Construction with Deep Reinforcement Learning (https://arxiv.org/abs/2410.15469)
Comments:
          Submitted to 2025 American Control Conference (ACC)

- **What's New**: 이번 논문에서는 로봇이 인간의 지시 없이도 불완전한 조립 구조를 이해하고 완성할 수 있는 능력을 개발하기 위해 3D 조합 조립 완료(combinatorial assembly completion) 문제를 다루고 있습니다. 특히, Lego 블록을 사용하여 조립을 시뮬레이션하며, 로봇이 스스로 목표를 이해하고 필요한 기능을 추가하는 등의 작업을 수행할 수 있도록 하는 딥 강화 학습(Deep Reinforcement Learning, DRL) 프레임워크를 소개합니다.

- **Technical Details**: 이 연구에서는 두 부분으로 구성된 딥 강화 학습 프레임워크를 제안합니다. 첫 번째 부분은 로봇이 불완전한 조립 구조의 목표를 이해하는 데 초점을 맞추고, 두 번째 부분은 조립을 완성하기 위한 건축 정책(construction policy)을 학습합니다. 로봇은 안정적인 객체 라이브러리(object library)를 쿼리하여 조립 추론(assembly inference)을 가능하게 하고 학습을 진행합니다. 또한, 동작 마스크(action mask)를 개발하여 물리적 제약(physical constraints)을 위반하는 무효 행동을 배제합니다.

- **Performance Highlights**: 논문에서 제안하는 프레임워크는 다양한 조립 시나리오에서 유효성과 강건성을 입증하였으며, 로봇이 완전한 솔루션과 실행 시간 품질 모두를 만족하며 실제 조립을 수행할 수 있음을 보여줍니다. 이 결과를 통해 제안된 프레임워크가 보지 못한(unique) 객체 유형에 대한 불완전 구조를 효과적으로 추론하고 조립할 수 있음을 확인하였습니다.



### Hey GPT, Can You be More Racist? Analysis from Crowdsourced Attempts to Elicit Biased Content from Generative AI (https://arxiv.org/abs/2410.15467)
- **What's New**: 이번 연구는 비전문 사용자들이 Generative AI (GenAI) 툴의 편향된 출력을 어떻게 인식하고 상호작용하는지에 대한 탐구를 진행하였습니다. 이는 대규모 언어 모델(LLM)의 편향성 문제를 다룬 기존 연구와의 차별화된 접근으로, 사용자들이 만든 프롬프트를 통해 편향을 유도하는 방법을 분석하였습니다.

- **Technical Details**: 이 연구는 펜실베이니아 주립대학교에서 열린 대회에서 수집된 데이터를 기반으로 하였습니다. 참가자들은 GenAI 툴에서 편향된 출력을 유도하기 위한 프롬프트를 설계하며, 총 75개의 유효한 제출물이 접수되었습니다. 연구팀은 이 중 80% 이상의 프롬프트가 재현 가능하다는 결과를 도출하였으며, 8가지 유형의 편향으로 분류되었습니다.

- **Performance Highlights**: 대회에서 선정된 승상들은 GenAI 툴에서 편향된 출력을 유도하는 다양한 전략을 활용하였고, 인터뷰 분석을 통해 참가자들이 정의한 편향의 개념과 전략들을 밝혀냈습니다. 연구 결과는 비전문 사용자들이 LLMs를 어떻게 조작하는지에 대한 독특한 통찰을 제공합니다.



### Keep Guessing? When Considering Inference Scaling, Mind the Baselines (https://arxiv.org/abs/2410.15466)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)에서 반복 sampling을 통한 추론(compute) 증가가 문제 해결률(coverage) 증가에 미치는 영향을 분석하였습니다. 이들은 표준 평가 벤치마크의 정답 분포가 일반적인 정답으로 편향되어 있다는 가설을 세우고, 이에 대한 검증을 위한 baseline을 정의하였습니다.

- **Technical Details**: 세 가지 접근 방식이 비교되었습니다: (1) ModelAnswers - 모델 응답을 샘플링하여 k개의 후보 답안을 얻는 방법, (2) TrainCounts - 훈련 세트에서 가장 빈도가 높은 k개의 답안을 열거하여 후보 답안을 얻는 방법, (3) Mixture(M) - ModelAnswers와 TrainCounts 혼합 전략으로, M개의 답안을 모델 샘플링으로, 나머지 k-M개의 답안을 열거하여 얻는 방법입니다.

- **Performance Highlights**: 실험 결과, 일부 모델은 문제-무관 추측(TrainCounts)보다 더 낮은 성능을 보였으며, Mixture 접근법이 ModelAnswers와 거의 같은 수준의 문제 해결률을 달성하였습니다. 이러한 결과는 데이터셋 선택과 모델 성능 평가 시 주의가 필요하다는 점을 시사합니다.



### Concept Complement Bottleneck Model for Interpretable Medical Image Diagnosis (https://arxiv.org/abs/2410.15446)
Comments:
          10 pages, 5 figures, submitted to IEEE TRANSACTIONS ON MEDICAL IMAGING

- **What's New**: 이 논문에서는 설명 가능한 의료 이미지를 진단하기 위한 Concept Complement Bottleneck Model (CCBM)을 제안합니다. CCBM은 기존 개념 집합을 보완하고 새로운 개념을 발견하여 설명 가능한 모델 간의 격차를 줄이는 것을 목표로 합니다.

- **Technical Details**: CCBM은 각 개념마다 개념 어댑터를 활용하여 해당 개념과 가장 관련성이 높은 이미징 기능을 인코딩합니다. Multi-Head Cross-Attention (MHCA)를 사용하여 각 개념의 점수를 독립적인 주의 채널에서 계산하여 공정한 개념 학습을 지원합니다. 또한, 알려진 개념을 함께 사용하여 새로운 개념을 학습하는 전략을 설계했습니다.

- **Performance Highlights**: 의료 데이터셋에서의 실험 결과, CCBM은 개념 탐지 및 질병 진단 작업에서 기존의 최신 모델과 비교하여 우수한 성능을 보였으며, 다양한 설명을 제공하여 모델의 해석 가능성을 효과적으로 보장합니다.



### Evaluating Consistencies in LLM responses through a Semantic Clustering of Question Answering (https://arxiv.org/abs/2410.15440)
Comments:
          Accepted to the Trustworthy AI Workshop at IJCAI 2024

- **What's New**: 대형 언어 모델(LLM)의 일관성을 평가하기 위한 새로운 접근 방식을 제안합니다. 기존의 랜덤성(cell sampling)으로 인한 일관성 부족 문제를 해결하고자 합니다.

- **Technical Details**: 이 연구에서는 LLM의 응답이 주어진 질문에 대해 의미적으로 일치하는지를 평가하기 위해 두 가지 주요 접근 방식을 탐구하였습니다: RAG 패턴과 같은 외부 지식을 컨텍스트로 활용하거나 Zero-shot-CoT를 사용하여 LLM의 성능을 개선하는 것입니다. TruthfulQA 데이터셋을 활용하여 LLM의 응답을 평가하고, 의미론적으로 동등한 문장을 클러스터링하여 37개의 카테고리에서 의미적 일관성을 측정합니다.

- **Performance Highlights**: 이 방법론을 통해 LLM의 성능 개선 전후에 대한 정량적 분석을 수행하였으며, 다른 질문 응답 작업에서 이러한 방법들이 LLM 응답 일관성에 미치는 영향을 비교했습니다.



### AttCDCNet: Attention-enhanced Chest Disease Classification using X-Ray Images (https://arxiv.org/abs/2410.15437)
- **What's New**: 최근 연구에서는 전통적인 의학적 방법 대신 심층 학습 기반 기법을 적용하여 흉부 X-ray 이미지 진단을 자동화하는 새로운 모델인 AttCDCNet을 제안하였다. 이 모델은 DenseNet121 구조를 개선하고 주의 메커니즘(Attention Mechanism)을 추가하여 중요한 부분에 집중하며, 불균형 문제를 해결하기 위해 focal loss를 손실 함수로 채택하였다.

- **Technical Details**: AttCDCNet은 DenseNet121 모델을 기반으로 하여, Attention Block 추가와 Depth-wise Convolution을 통해 파라미터 수를 줄여 경량화된 구조를 갖는다. 이 모델은 입력 흉부 X-ray 이미지에서 질병의 분류를 위한 전처리, 특징 추출 및 분류 단계로 나뉜다. 실험에서는 다양한 흉부 질병이 포함된 데이터세트를 이용하여 성능을 평가하였다.

- **Performance Highlights**: 제안된 AttCDCNet 모델은 COVID-19 방사선 사진 데이터세트에서 각각 94.94%의 정확도, 95.14%의 정밀도, 94.53%의 재현율을 기록하며, 기존의 DenseNet121 모델보다 뛰어난 성능을 보였다. 이 연구는 흉부 질병 진단의 최신 기법들이 어떻게 통합될 수 있는지를 보여준다.



### Where to Build Food Banks and Pantries: A Two-Level Machine Learning Approach (https://arxiv.org/abs/2410.15420)
Comments:
          12 pages, 4 figures

- **What's New**: 이번 연구에서는 미국에서 식량 불안정 문제를 해결하기 위해 새로운 두 단계 최적화 프레임워크를 제안합니다.

- **Technical Details**: 이 프레임워크는 K-Medoids 클러스터링 알고리즘을 Open-Source Routing Machine 엔진과 결합하여 실제 도로 거리 기반으로 식량 은행과 푸드 팬트리의 위치를 최적화합니다. 가구의 중위소득과 같은 요소를 고려할 수 있도록 의사 가중치 K-Medoids 알고리즘을 적용했습니다.

- **Performance Highlights**: 캘리포니아와 인디애나 가정 데이터를 이용한 테스트 결과, 제안한 프레임워크는 기존의 푸드 팬트리 위치보다 우수한 결과를 보여주었으며, 가정에 대한 이동 거리를 크게 절감했습니다. 1단계에서의 푸드 뱅크와 푸드 팬트리 간 거리에는 약간의 비용이 있지만, 전반적으로 이 프레임워크의 이점이 단점을 훨씬 능가한다고 생각합니다.



### A Comprehensive Evaluation of Cognitive Biases in LLMs (https://arxiv.org/abs/2410.15413)
- **What's New**: 이 연구에서는 20개의 최신 Large Language Models (LLMs)에서 30가지 인지 편향(cognitive biases)을 평가하기 위한 대규모 테스트를 제시하고 있습니다. 연구의 주요 기여로는 LLMs의 인지 편향을 탐색하기 위한 30,000개의 테스트 데이터셋과 신뢰할 수 있는 테스트 생성을 위한 새로운 일반 목적의 테스트 프레임워크가 포함됩니다.

- **Technical Details**: 제안된 프레임워크는 인지 편향 테스트를 정의, 다양화, 수행하기 위한 체계적이고 일반적인 구조를 제공합니다. 이 프레임워크는 입력과 출력을 처리하는 여러 개체와 기능으로 구성되어 있으며, 사전 훈련된 LLM을 내부적으로 활용합니다. 사용자는 이를 통해 200개의 다양한 의사결정 시나리오에서 30가지 인지 편향을 평가할 수 있는 테스트를 생성할 수 있습니다.

- **Performance Highlights**: 20개의 LLMs에서 모든 테스트된 30가지 인지 편향의 증거가 발견되었으며, 이는 LLMs에서 인지 편향이 광범위하게 존재함을 확인했습니다. 이러한 발견은 기존 연구 결과를 확장하는 것으로, 향후 LLMs의 인지 편향에 대한 연구를 촉진할 수 있는 기반이 됩니다.



### PEAS: A Strategy for Crafting Transferable Adversarial Examples (https://arxiv.org/abs/2410.15409)
- **What's New**: 본 논문에서는 PEAS(Perception Exploration Attack Strategy)라는 새로운 전략을 제안하여 블랙 박스 공격의 전이 가능성을 높였습니다. 이 방법은 시각적으로 동등한 샘플들이 적대적 전이 가능성에서 큰 변동을 보인다는 통찰력을 활용합니다.

- **Technical Details**: PEAS는 입력 이미지에서 미세 변형을 통해 시각적으로 동등한 변형 이미지를 생성하는 것으로 시작합니다. 그런 다음 여러 대체 모델을 사용해 여기서 생성한 이미지에 적대적 변동을 평가합니다. 마지막으로, 가장 전이 가능성이 높은 이미지가 선택되어 공격에 사용됩니다. 이러한 과정은 상대적으로 적은 양의 변형으로도 성과를 나타낼 수 있음을 보여줍니다.

- **Performance Highlights**: 실험 결과, PEAS는 기존 방법들에 비해 공격 성공률을 평균 2.5배 향상시킴으로써 최신 블랙 박스 공격 설정에서 최고의 성과를 달성한 것으로 나타났습니다. 다양한 데이터셋(ImageNet, CIFAR-10)에서 평가한 결과도 포함되어 있습니다.



### MMCS: A Multimodal Medical Diagnosis System Integrating Image Analysis and Knowledge-based Departmental Consultation (https://arxiv.org/abs/2410.15403)
- **What's New**: MMCS 시스템은 의료 이미지를 인식하고 환자의 얼굴 세부 사항을 분석하여 전문적인 의료 진단을 제공할 수 있는 혁신적인 솔루션입니다.

- **Technical Details**: 시스템은 두 개의 핵심 구성 요소로 이루어져 있습니다. 첫 번째 구성 요소는 의료 이미지 및 비디오의 분석으로, 다중 모달(multimodal) 의료 모델을 훈련시켜 의료 이미지를 해석하고 환자의 얼굴 감정 및 얼굴 마비 상태를 정확히 분석할 수 있도록 하였습니다. 모델은 FER2013 얼굴 감정 인식 데이터셋에서 72.59%의 정확도를 달성하였으며, 행복 감정 인식에서는 91.1%의 정확도를 기록했습니다. 얼굴 마비 인식에서는 92%의 정확도를 달성하였으며 이는 GPT-4o보다 30% 더 높은 수치입니다. 두 번째 구성 요소는 전문 의료 응답 생성을 위한 대형 언어 모델(large language model)과 의료 지식 기반을 통합하여 의료 이미지를 분석한 후 적절한 진단을 생성하는 것입니다.

- **Performance Highlights**: 얼굴 마비 환자에 대한 30개의 비디오 테스트에서 시스템은 83.3%의 정확도로 마비의 중증도를 정확히 평가했습니다. 또한, 의료 부서별 지식 기반 라우팅 관리 메커니즘을 통해 대형 언어 모델은 데이터의 의료 부서를 분류하고 적절한 지식 기반을 조회하여 RAG(검색 보강 생성) 과정에서 검색 정확성을 4% 향상시켰습니다.



### The Best Defense is a Good Offense: Countering LLM-Powered Cyberattacks (https://arxiv.org/abs/2410.15396)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)을 활용한 사이버 공격의 자동화를 우려하고, 이러한 공격에 대한 방어 전략을 제안합니다. LLM의 고유한 취약점을 이용하여 공격자를 방해하고, 지연시키거나 무력화하는 기술을 개발했습니다.

- **Technical Details**: 논문에서는 LLM의 편향(bias), 입력에 대한 신뢰(trust in input), 메모리 한계(memory limitations)와 같은 취약점을 공략하는 다양한 방어 전략을 소개합니다. 방어 성공률은 최대 90%로, black-box 환경에서의 성능을 평가하였습니다.

- **Performance Highlights**: 제안된 방어 기술은 여러 가지 환경에서의 공격 시나리오에서 성공적으로 작동하였으며, 현재 존재하는 위협 행위자들을 효과적으로 저지할 수 있는 능력을 입증하였습니다.



### Synthetic Data Generation for Residential Load Patterns via Recurrent GAN and Ensemble Method (https://arxiv.org/abs/2410.15379)
Comments:
          12 pages

- **What's New**: 이 논문은 Ensemble Recurrent Generative Adversarial Network (ERGAN)라는 프레임워크를 개발하여 고충실도의 합성 주거 부하 데이터를 생성합니다. ERGAN은 반복적인 Generative Adversarial Networks의 앙상블을 활용하여 다양한 주거자들의 부하 패턴을 포착하고, 프라이버시 문제를 유발하지 않으면서 현실적인 부하 데이터를 생성할 수 있도록 설계되었습니다.

- **Technical Details**: ERGAN은 K-means clustering과 GAN을 통합하여 작동합니다. 첫 번째 단계에서 데이터를 K 개의 클러스터로 분할한 후, 각 클러스터별로 독립적인 GAN 모델을 학습시킵니다. 이 과정은 Bi-Directional Long Short-Term Memory (Bi-LSTM) 네트워크를 사용하여 진행되며, 생성자(generator)와 구분자(discriminator) 역할을 수행합니다. 손실 함수는 적대적 손실(adversarial loss)과 통계적 속성의 차이를 모두 고려하여 최적화됩니다.

- **Performance Highlights**: ERGAN은 다양한 성능 지표에서 기존 벤치마크 모델들보다 우수한 성능을 보여주었습니다. 딱 맞아 떨어지는 데이터 다양성, 유사성 및 통계적 측면에서 ERGAN의 생성 능력이 뛰어난 것으로 평가되었습니다.



### Explainability of Point Cloud Neural Networks Using SMILE: Statistical Model-Agnostic Interpretability with Local Explanations (https://arxiv.org/abs/2410.15374)
Comments:
          17 pages, 9 figures

- **What's New**: 이번 연구에서는 SMILE이라는 새로운 설명 가능성 방법을 제안하여 포인트 클라우드(point cloud) 모델에 적용했습니다. SMILE은 LIME을 기반으로 하여 수학적 거리(특히, Anderson-Darling 거리)를 통해 설명 가능성을 향상시킵니다.

- **Technical Details**: SMILE 방법은 Empirical Cumulative Distribution Function (ECDF) 통계적 거리를 통합하여 강력한 해석 가능성과 견고성을 제공합니다. 이는 다양한 커널 너비(kernel widths), 섭동 수(perturbation numbers), 클러스터 구성(clustering configurations)에서의 충실도 손실(fidelity loss), R2 점수(R2 scores), 견고성(robustness) 측면에서 우수한 성능을 보입니다. 또한, Jaccard 지수를 사용한 포인트 클라우드 데이터의 안정성 분석(stability analysis)을 도입하였습니다.

- **Performance Highlights**: 연구 결과, SMILE이 개발한 기법은 모델 안정성(stability) 분야에 대한 새로운 벤치마크를 제시하였고, '사람(person)' 카테고리의 분류에서 데이터셋 편향(bias)을 식별하며, 자율주행 및 로봇 공학과 같은 안전-critical 애플리케이션에서 보다 포괄적인 데이터셋의 필요성을 강조합니다.



### FrameBridge: Improving Image-to-Video Generation with Bridge Models (https://arxiv.org/abs/2410.15371)
- **What's New**: 새로운 모델 FrameBridge를 제안하여, 정적 이미지를 비디오의 목표로 활용하고, 이미지와 비디오 간의 트랙터블(tractable) 브리지 모델을 구축했습니다. 이는 기존의 Diffusion 기반 I2V 생성 과정에서 발생하는 일관성 및 시간적 응집력 문제를 해결하는 데 기여합니다.

- **Technical Details**: FrameBridge는 I2V 합성을 프레임 간 생성(task)으로 체계화하고, 데이터 간 프로세스를 모델링하여 입력 이미지의 정보를 최대한 활용합니다. 추가적으로 SNR-Aligned Fine-tuning (SAF)과 neural prior라는 두 가지 기술을 제안하여, 기존의 diffusion 모델과 비교하여 효율성과 합성 품질을 향상시킵니다.

- **Performance Highlights**: WebVid-2M 및 UCF-101에서 실험 결과, FrameBridge는 이전의 diffusion 모델과 비교하여 I2V 품질을 현저하게 개선했습니다. 예를 들어, MSR-VTT에서 zero-shot FVD 점수를 176에서 83으로 줄였고, UCF-101에서는 non-zero-shot FVD 점수를 171에서 122로 감소시켰습니다.



### Ethical AI in Retail: Consumer Privacy and Fairness (https://arxiv.org/abs/2410.15369)
Comments:
          17 pages, 2 figures, 3 tables

- **What's New**: 이번 연구는 소매업에서 인공지능(AI) 기술 채택으로 인한 윤리적 도전과제에 대한 분석을 제공합니다. 소비자 개인정보 및 공정성 문제를 다루며, 소매업체들이 AI를 윤리적으로 구현할 수 있는 방안을 탐구합니다.

- **Technical Details**: 이 연구는 300명의 응답자를 대상으로 설명적 설문조사(descriptive survey) 디자인을 이용하여 데이터를 수집했습니다. 수집된 데이터는 백분율(percentage) 및 평균 점수(mean scores)와 같은 기술 통계(descriptive statistics)를 통해 분석되었습니다.

- **Performance Highlights**: 연구 결과, 소비자들은 AI 기반 소매 애플리케이션이 개인 정보를 과도하게 수집하고 있다고 우려하고 있으며, 데이터 관리에 대한 신뢰 부족을 보였습니다. 공정성 문제도 큰 이슈로, 다수의 소비자들이 AI 시스템이 공평하게 대우하지 않는다고 느끼고 있습니다. 또한 AI는 윤리적 원칙을 위반하지 않으면서도 비즈니스의 경쟁력과 효율성을 높일 수 있는 가능성이 있음을 보여줍니다. 소매업체들이 데이터 보호 및 AI 시스템에 대한 지속적인 감시가 필요하다는 중요한 요구가 제시되었습니다.



### Faster-GCG: Efficient Discrete Optimization Jailbreak Attacks against Aligned Large Language Models (https://arxiv.org/abs/2410.15362)
- **What's New**: 이번 논문에서는 Aligned Large Language Models (LLMs)의 취약점을 노출시키는 새로운 공격 기법인 Faster-GCG를 제안합니다. 기존의 Greedy Coordinate Gradient (GCG) 방식의 비효율성을 개선하여, 더 낮은 계산 비용으로 높은 성공률을 기록합니다.

- **Technical Details**: Faster-GCG는 GCG의 세 가지 주요 문제를 해결합니다. 첫째, 기울기 계산에서 다양한 토큰 간 거리를 고려한 정규화 항을 추가합니다. 둘째, 무작위 샘플링 대신 결정론적 탐색을 사용하여 대체 토큰을 평가하고, 셋째, 자기 루프 문제를 방지하기 위한 중복 제거 방법을 제안합니다.

- **Performance Highlights**: Faster-GCG는 원래 GCG의 계산 비용의 1/10로 두 개의 Aligned LLM(Llama-2-7B-chat, Vicuna-13B-v1.5)에 대해 각각 29% 및 8% 높은 공격 성공률을 달성하였습니다. 또한, ChatGPT와 같은 비공개 LLM에서의 공격 전이 가능성도 개선되었습니다.



### LAC: Graph Contrastive Learning with Learnable Augmentation in Continuous Spac (https://arxiv.org/abs/2410.15355)
- **What's New**: 이 논문에서는 graph contrastive learning (GCL) 프레임워크에 learnable data augmentation을 도입한 LAC를 제안합니다. LAC는 orthogonal continuous space에서 작동하며, 이 공간 내에서 최적의 topology와 feature augmentation을 수행합니다.

- **Technical Details**: LAC의 핵심은 Continuous View Augmentation (CVA) 모듈로, Masked Topology Augmentation (MTA) 및 Cross-channel Feature Augmentation (CFA) 모듈을 통해 그래프 데이터를 보강합니다. 또한, InfoBal 원칙을 통해 augmented views 간의 일관성을 유지하면서 다양성을 극대화합니다.

- **Performance Highlights**: LAC는 7개의 공개 데이터셋에서 실험을 수행하여 기존의 state-of-the-art GCL 프레임워크를 크게 초월하는 성능을 보였습니다. 이는 unsupervised setting에서 효과적입니다.



### YOLO-RD: Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary (https://arxiv.org/abs/2410.15346)
- **What's New**: 본 논문에서는 YOLO 기반의 객체 탐지 모델에 외부 정보의 효율적인 활용을 지원하는 혁신적인 {f Retriever}-{f Dictionary} (RD) 모듈을 제안합니다. 이 모듈은 데이터셋의 통찰을 담고 있는 Dictionary에서 특성을 효율적으로 검색할 수 있게 해주며, Visual Models (VM), Large Language Models (LLM), Visual Language Models (VLM)의 지식을 활용합니다.

- **Technical Details**: RD 모듈은 객체 탐지 작업뿐만 아니라 세분화(segmentation)와 분류(classification) 작업에서도 동시에 이점을 취할 수 있도록 설계되었습니다. 이 모듈은 쿼리를 생성하기 위해 영역 특성을 집계하는 Retriever와, 관련 원자를 선택할 수 있도록 쿼리를 지원하는 Dictionary로 구성됩니다. Dictionary는 YOLO 백본을 넘어서는 외부 지식을 통합합니다.

- **Performance Highlights**: RD 모듈을 사용하는 실험 결과, 평균 정밀도(Mean Average Precision)가 3% 이상 증가하였으며, 모델 파라미터의 증가는 1% 미만으로 최소화되었습니다. 이 모듈은 YOLO 뿐만 아니라 Faster R-CNN, Deformable DETR와 같은 2단계 모델에도 성능을 향상시켜 주는 것으로 나타났습니다.



### FoMo: A Foundation Model for Mobile Traffic Forecasting with Diffusion Mod (https://arxiv.org/abs/2410.15322)
Comments:
          17 pages, 11 figures

- **What's New**: 이 연구는 다양한 도시에서의 이동통신 트래픽 예측을 위한 새로운 기초 모델(FoMo)을 제안합니다. 이 모델은 다양한 예측 작업을 처리하고, 네트워크 계획 및 최적화를 지원합니다.

- **Technical Details**: FoMo는 diffusion 모델과 transformers를 결합하여 다양한 spatio-temporal masks를 사용합니다. 이를 통해 모델은 다양한 작업의 내재적 특징을 학습하고, contrastive learning 전략을 개발하여 이동통신 트래픽과 도시 맥락 간의 상관관계를 개선합니다.

- **Performance Highlights**: FoMo는 9개의 실제 데이터 세트에 대한 실험에서 기존 모델보다 뛰어난 성능을 보여주었으며, 다양한 예측 작업과 zero/few-shot learning에서 강력한 보편성을 입증했습니다.



### Causality for Large Language Models (https://arxiv.org/abs/2410.15319)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 훈련 과정에 인과성(causality)을 통합하는 방법을 탐구하며, 기존 상관 관계 기반 접근법을 넘어서는 새로운 패러다임의 필요성을 강조합니다. 또한, 인과적 추론(causal reasoning)을 통해 LLM의 해석 가능성, 신뢰성 및 윤리적 정렬을 향상시키는 여섯 가지 미래 방향을 제시합니다.

- **Technical Details**: 대형 언어 모델은 수십억에서 수조 개의 매개변수(parameters)를 가진 인공지능 모델로, 방대한 데이터셋에서 학습됩니다. 이 모델들은 트랜스포머(transformer) 네트워크를 기반으로 구성되어있으며, 자연어 이해(natural language understanding), 자연어 생성(natural language generation) 및 논리적 문제 해결(logical reasoning) 등의 작업을 수행합니다.

- **Performance Highlights**: LLMs의 현재 한계는 인과 관계를 이해하지 않고, 단순히 상관 관계를 바탕으로 응답을 생성하는 데 있습니다. 이는 모델의 정확성과 신뢰도를 저하시키며, 의료 및 법률과 같은 중요한 분야에서의 잘못된 정보 생성으로 이어질 위험이 있습니다. 인과성을 효과적으로 도입하면 보다 신뢰할 수 있고 윤리적으로 정렬된 AI 시스템을 개발하는 데 도움이 됩니다.



### SNAP: Stopping Catastrophic Forgetting in Hebbian Learning with Sigmoidal Neuronal Adaptive Plasticity (https://arxiv.org/abs/2410.15318)
Comments:
          6 pages, 11 figures, accepted at Montréal AI and Neuroscience (MAIN) 2024 conference

- **What's New**: 이 논문에서 제안하는 Sigmoidal Neuronal Adaptive Plasticity (SNAP)는 인공 신경망(Artificial Neural Networks, ANNs)에서 기존의 Learner가 가질 수 있는 catastrophic forgetting 문제를 해결하기 위한 새로운 방법론입니다.

- **Technical Details**: SNAP는 장기 강화(Long-Term Potentiation, LTP)의 인공적 근사로서, 가중치(weight)가 시그모이드(sigmoidal) 성장 행동을 따르도록 하여, 가중치가 적절한 값에 도달했을 때 고정하고 안정화시킵니다. 이는 기존의 가중치 업데이트 방식과 대조적입니다.

- **Performance Highlights**: SNAP를 유연한 Hebbian Learning(헤비안 학습)과 비교했을 때, 이전 작업의 망각을 완전히 방지하는 뛰어난 성능을 보였으며, Stochastic Gradient Descent(SGD) 기반 학습에서는 동일한 효과를 보이지 않았습니다.



### Synergistic Dual Spatial-aware Generation of Image-to-Text and Text-to-Imag (https://arxiv.org/abs/2410.15312)
- **What's New**: 이번 연구에서는 Spatial Image-to-Text (SI2T)와 Spatial Text-to-Image (ST2I)라는 두 가지 작업을 이중 학습 프레임워크에서 함께 모델링하는 방법을 제안합니다. 특히, 3D 공간 장면 특성을 공유할 수 있는 새로운 3D Scene Graph (3DSG) 표현 방식을 도입하였습니다.

- **Technical Details**: SI2T는 주어진 이미지에서 객체들의 공간적 관계를 이해하는 반면, ST2I는 입력 텍스트 프롬프트에 기반하여 공간적으로 적합한 이미지를 합성합니다. 본 연구에서는 공간을 인식하는 데 필요한 3D 특징 모델링의 어려움을 극복하기 위해 Spatial Dual Discrete Diffusion (SD3) 프레임워크를 제안하며, 이는 중간 3D→X 과정의 특징을 활용하여 X→3D 과정을 지원합니다.

- **Performance Highlights**: VSD 데이터셋에서 수행한 실험 결과, 제안한 시스템이 기존 T2I 및 I2T 방법에 비해 우수한 성능을 보였으며, 이중 학습 전략이 시각-언어 모델을 통한 비대칭 공간 의미 정렬에 기여함을 밝혔습니다.



### LlamaLens: Specialized Multilingual LLM for Analyzing News and Social Media Conten (https://arxiv.org/abs/2410.15308)
Comments:
          LLMs, Multilingual, Language Diversity, Large Language Models, Social Media, News Media, Specialized LLMs, Fact-checking, Media Analysis

- **What's New**: 이 연구는 LlamaLens라는 전문화된 대형 언어 모델(LLM)을 개발하여 다국어 컨텍스트에서 뉴스 및 소셜 미디어 콘텐츠를 분석하는 데 중점을 두고 있습니다. 이는 도메인 특화성과 다국어 처리를 동시에 해결하려는 최초의 시도입니다.

- **Technical Details**: LlamaLens는 아랍어, 영어, 힌디어를 포함한 3개 언어로 52개의 데이터셋을 사용하여 19개의 작업을 다룹니다. 모델은 기존 LLM을 세밀하게 조정하여 도메인 지식을 강화했으며, 학습 중 다양한 데이터 셔플링 기술을 활용했습니다. 실험 결과는 LlamaLens가 16개의 테스트 세트에서 현재의 최첨단(State of the Art, SOTA) 성능을 초과하며, 10개 세트에서는 비슷한 성능을 달성함을 보여줍니다.

- **Performance Highlights**: LlamaLens는 기존의 비세밀 조정 모델에 비해 성능을 현저히 향상시켰으며, 특히 작은 버전의 모델에서도 도메인 및 언어 특화 지식을 획득하는 데 성공했습니다. SOTA와의 비교에서도 개선의 여지가 존재함을 시사합니다.



### Redefining Proactivity for Information Seeking Dialogu (https://arxiv.org/abs/2410.15297)
- **What's New**: 이 연구는 정보 검색 대화 (ISD) 에이전트의 반응적 행동을 넘어서 사용자와의 지속적인 대화를 생성하는 새로운 주도적 (proactive) 대화를 위한 정의를 제안합니다. 이 정의에서는 초기 질문과 관련된 새로운 정보를 통해 응답의 주도성을 향상시키는 데 중점을 두고 있습니다.

- **Technical Details**: 우리는 총 2,000개의 단일 회화로 구성된 주도적 대화 데이터셋을 구축하고, 응답의 '주도성'을 평가하기 위한 여러 자동화된 메트릭을 도입하였습니다. 이 메트릭은 인간 주석과의 높은 상관관계를 달성하였습니다. 또한, 두 가지 혁신적인 Chain-of-Thought (CoT) 프롬프트인 3단계 CoT 프롬프트와 3-인-1 CoT 프롬프트를 제안하여 제로샷 설정에서 표준 프롬프트보다 최대 90% 더 우수한 성능을 보였습니다.

- **Performance Highlights**: 이 연구는 주도적 응답 생성의 맥락에서 instruction-tuning의 효과성을 입증하였으며, 사용자 상호작용을 지속시키고 대화의 정보성을 개선하는 데 성공했습니다. 또한, 제안된 접근 방식은 여러 차례의 반복 대화 시나리오에서도 강력한 성과를 나타냈습니다.



### Fractional-order spike-timing-dependent gradient descent for multi-layer spiking neural networks (https://arxiv.org/abs/2410.15293)
Comments:
          15 pages, 12 figures

- **What's New**: 이번 연구에서는 생물학적으로 영감을 받은 스파이킹 신경망(Spiking Neural Networks, SNNs)의 학습 문제를 해결하기 위해 분수차 스파이킹 의존 경량화 경량화 학습 모델(FO-STDGD)을 제안합니다.

- **Technical Details**: FO-STDGD 모델은 비누설 통합-발화 회로(nonleaky integrate-and-fire neurons)의 시간적 막 전위와 준 순간 발화 빈도 간의 관계를 설명하는 비선형 활성화 함수(nonlinear activation function)를 도입하여 학습 전략을 제공합니다. 이 모델은 0과 2 사이의 모든 분수 차수에 대해 일반화할 수 있습니다.

- **Performance Highlights**: FO-STDGD 모델은 MNIST와 DVS128 제스처 데이터셋에서 테스트되었으며, 분수 차수가 증가함에 따라 분류 정확도가 향상되는 것으로 나타났습니다. 특히, 분수 차수 1.9의 경우 전통적인 경량화 기법(분수 차수 1)에 비해 155% 향상된 정확도를 기록했습니다.



### Large Language Models for Autonomous Driving (LLM4AD): Concept, Benchmark, Simulation, and Real-Vehicle Experimen (https://arxiv.org/abs/2410.15281)
- **What's New**: 이 연구에서는 Large Language Models (LLMs)을 자율주행 기술에 적용하기 위한 새로운 개념과 접근 방식을 소개하고 있습니다. LLM4AD(LLMs for Autonomous Driving)라는 프레임워크를 제안하며, LLMs의 지시 수용 능력을 평가하기 위한 종합적인 벤치마크도 개발합니다.

- **Technical Details**: 제안된 LLM4AD 프레임워크는 LLMs가 자율주행 시스템 내에서 의사결정의 '두뇌' 역할을 수행하는 구조입니다. 이 프레임워크에서 LLMs는 차량의 감지 및 위치 모듈의 출력을 참고하여 높은 수준의 의사결정을 내립니다. 또한, LLMs는 유저의 피드백 및 시스템 메시지를 바탕으로 자연어로 대화를 하고, 다양한 자율주행 작업에 대한 최적의 주행 정책을 생성합니다.

- **Performance Highlights**: 실험을 통해 LLM4AD 시스템이 자율주행의 다양한 측면을 개선할 수 있는 잠재력을 보여주었습니다. LLMs은 직관적인 언어 상호작용, 맥락 이해 및 추론, 제로샷 및 퍼셉션 업무에 대한 성능 향상, 개인화 등의 이점을 제공합니다. 그러나 실시간 의사결정 능력의 지연과 같은 한계도 있으므로, 추가적인 연구가 필요합니다.



### ContextDet: Temporal Action Detection with Adaptive Context Aggregation (https://arxiv.org/abs/2410.15279)
- **What's New**: 이번 연구에서는 Temporal Action Detection(TAD) 분야에 새로운 단일 스테이지 모델인 ContextDet를 제안하며, 이는 대형 커널 컨볼루션(large-kernel convolutions)을 처음으로 사용한 점이 특징입니다. 이 모델은 행동 구분을 향상시키기 위해 적응형 컨텍스트 집계(pyramid adaptive context aggregation, ACA) 아키텍처를 활용합니다.

- **Technical Details**: ContextDet 모델은 각 ACA 수준에서 Context Attention Module(CAM)과 Long Context Module(LCM)이라는 두 가지 새로운 모듈로 구성되어 있습니다. CAM은 선택된 컨텍스트 정보를 활용하여 동작 구분을 향상시키고, LCM은 대형 및 소형 커널 컨볼루션을 혼합하여 장기적인 컨텍스트와 세밀한 로컬 특징을 효과적으로 수집합니다.

- **Performance Highlights**: ContextDet 모델은 MultiThumos, Charades, FineAction, EPIC-Kitchens 100, Thumos14, HACS 등 6개의 도전적인 TAD 벤치마크에서 다른 최신 TAD 방법들과 비교하여 뛰어난 정확성과 향상된 추론 속도를 기록하였습니다.



### Performance-Driven QUBO for Recommender Systems on Quantum Annealers (https://arxiv.org/abs/2410.15272)
- **What's New**: 본 논문에서는 추천 시스템의 특성 선택을 위해 Counterfactual Analysis Quadratic Unconstrained Binary Optimization (CAQUBO)라는 새로운 접근 방식을 제안합니다. 이는 카운터팩추얼 분석을 통해 각 특성의 영향력을 측정하고, 이를 통해 최적 특성 조합을 선택하여 최종 추천 성능을 개선합니다.

- **Technical Details**: CAQUBO는 각각의 특성을 제외했을 때의 최종 추천 성능 변화를 평가하여 Coefficient Matrix를 구성합니다. 이는 기존의 방법들이 모델 결과나 직접적인 기준(label)과만 연결되어 있는 한계를 극복하며, 추천 성능에 초점을 맞춥니다. 현재의 양자 어닐러는 2개 이하의 특성만을 한 번에 제거할 수 있지만, 향후 개선 가능한 가능성을 제시합니다.

- **Performance Highlights**: 실험 결과, CAQUBO는 다양한 추천 시스템, 즉 Item-KNN부터 딥러닝 기반의 방법들에 이르기까지 평가되었으며, 추천 정확도 측면에서 기존 최첨단 양자 어닐러 기반 특성 선택 알고리즘을 초월하는 성능을 보여주었습니다.



### HyQE: Ranking Contexts with Hypothetical Query Embeddings (https://arxiv.org/abs/2410.15262)
- **What's New**: 이번 논문에서는 대규모 언어 모델(LLM)이 필요하지 않고 훈련 데이터에 맞춤화되지 않은 새로운 컨텍스트 랭킹 프레임워크를 제안합니다. 이 프레임워크는 LLM의 능력과 임베딩 유사성을 결합하여 효율적으로 컨텍스트를 정렬합니다.

- **Technical Details**: 제안된 프레임워크는 LLM을 사용하여 기존의 컨텍스트를 기반으로 사용자의 쿼리에 대한 가설 쿼리를 생성합니다. 그런 다음 이 가설 쿼리와 사용자 쿼리 간의 유사성에 따라 컨텍스트를 정렬합니다. 이 방법은 다양한 정보 검색 벤치마크에서 성능을 개선하며, 효율성과 확장성을 유지합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 다수의 정보 검색 벤치마크에서 컨텍스트의 랭킹 성능을 향상시키며, LLM의 사전 지식 없이도 관련성을 평가할 수 있는 새로운 방법론을 제공합니다.



### Lossless KV Cache Compression to 2% (https://arxiv.org/abs/2410.15252)
- **What's New**: 이 연구에서는 KV 캐시 메모리를 2% 이하로 압축하면서 성능을 유지하는 새로운 아키텍처인 Cross-Layer Latent Attention (CLLA)을 제안합니다. CLLA는 다양한 KV 캐시 압축 기법을 통합하여 일관된 프레임워크 내에서 압축을 수행합니다.

- **Technical Details**: CLLA는 주의 헤드(Attention Head) 및 차원(Dimension) 축소, 레이어 공유(Layer Sharing), 양자화(Quantization) 기술을 포함한 다양한 압축 기법을 결합하여 KV 캐시를 효과적으로 압축합니다. 이 연구는 손실 없는(lossless) 성능을 유지하면서 대량의 GPU 메모리를 절약할 수 있도록 설계되었습니다.

- **Performance Highlights**: CLLA는 대다수의 작업에서 손실 없는 성능을 산출하면서 기존 KV 캐시의 2% 미만만을 사용합니다. 여러 CLLA 변형에 대한 세부 분석을 통해 다양한 조합을 탐색하여 성능을 극대화했습니다.



### Tensor-Fused Multi-View Graph Contrastive Learning (https://arxiv.org/abs/2410.15247)
- **What's New**: 본 논문에서는 Tensor-Fused Multi-View Graph Contrastive Learning (TensorMV-GCL)이라는 새로운 프레임워크를 제안합니다. 이 방법은 Extended Persistent Homology (EPH)와 Graph Contrastive Learning (GCL) 표현을 통합하여 다중 스케일의 특징 추출을 용이하게 합니다.

- **Technical Details**: TensorMV-GCL은 tensor 집합화(tensor aggregation)와 압축(compression)을 사용하여 동일한 그래프의 여러 증강된 뷰에서 얻은 정보로부터 그래프 및 위상론적(topological) 특징을 융합합니다. 이 과정은 feature tensor 집합화와 변환을 분리하여 계산 오버헤드를 줄입니다.

- **Performance Highlights**: 실험 결과, TensorMV-GCL은 11개의 데이터셋 중 9개에서 15개의 최신 방법들보다 뛰어난 성능을 나타냈으며, 나머지 2개의 데이터셋에서도 유사한 결과를 보였습니다.



### Jailbreaking and Mitigation of Vulnerabilities in Large Language Models (https://arxiv.org/abs/2410.15236)
- **What's New**: 최근의 연구에서 대형 언어 모델(LLMs)의 취약점과 방어 전략 현황을 분석하였습니다. 이 논문은 특히 프롬프트 주입(prompt injection) 및 탈옥(jailbreaking) 공격에 대한 이해를 높이고, 이를 방어하기 위한 다양한 접근 방식을 검토하였습니다.

- **Technical Details**: LLMs는 대량의 데이터에서 학습하여 자연어 처리 작업이 가능하도록 설계된 인공지능 시스템입니다. 다양한 공격 벡터(persistent attack vectors)가 있으며, 공격 유형은 프롬프트 기반, 모델 기반, 멀티모달 및 다국어로 분류됩니다. 방어 기법으로는 프롬프트 필터링, 변환, 정렬 기법, 다중 에이전트 방어, 자기 규제 등이 포함됩니다.

- **Performance Highlights**: 연구의 결과는 LLM의 현행 안전 장치가 제한적이며, 새로운 공격 벡터에 의해 쉽게 무너질 수 있음을 시사합니다. 또한, 자율 감지 및 윤리적 고려사항이 통합된 향후 연구 방향을 제안합니다.



### IntersectionZoo: Eco-driving for Benchmarking Multi-Agent Contextual Reinforcement Learning (https://arxiv.org/abs/2410.15221)
Comments:
          In review

- **What's New**: 이 논문은 IntersectionZoo라는 새로운 멀티 에이전트 CRL의 벤치마크 스위트를 제안합니다. 이 벤치마크는 도시 도로 네트워크에서의 협력적인 에코 드라이빙 응용 프로그램을 기반으로 하여 현실 세계의 문제 변동성을 자연스럽게 포착합니다.

- **Technical Details**: IntersectionZoo는 10개 주요 미국 도시에서 파생된 16,334개의 신호가 있는 교차로 데이터 기반의 시뮬레이션을 통해 구축되었으며, 약 100만 개의 데이터 기반 교통 시나리오를 제공합니다. 이 시스템은 차량 배출가스에 영향을 미치는 요인(예: 온도, 도로 조건, 여행 수요)을 모델링합니다.

- **Performance Highlights**: 논문에서는 인기 있는 멀티 에이전트 RL 및 인간 유사 운전 알고리즘을 벤치마크하며, 멀티 에이전트 RL 알고리즘이 CRL 환경에서 일반화에 어려움을 겪는다는 점을 보여줍니다.



### Low-cost Robust Night-time Aerial Material Segmentation through Hyperspectral Data and Sparse Spatio-Temporal Learning (https://arxiv.org/abs/2410.15208)
Comments:
          Accepted to the International Conference on Neural Information Processing (ICONIP) 2024. To be published in Springer-Nature Communications in Computer and Information Science (CCIS) Series

- **What's New**: 이 논문에서는 저조도 및 대기 조건에서의 항공 데이터를 다루는 복잡한 재료 세분화(detection) 문제를 해결하기 위해, RGB 이미지와 더불어 특수 카메라에서 얻은 하이퍼스펙트럴(Hyperspectral) 데이터를 활용하는 혁신적인 Siamese 프레임워크를 제안합니다. 이는 시계열(time series) 기반의 압축(compression) 방식을 통해 저해상도의 하이퍼스펙트럴 이미지를 효과적으로 통합하는 방법을 고안하였습니다.

- **Technical Details**: 제안된 프레임워크는 다음과 같은 주요 기술적 세부 사항을 포함합니다: 1) 선택적 채널 활용(Selective Channel Utilization): 데이터 세트 전체를 처리하는 대신 필수적인 스펙트럴 데이터를 추출합니다. 2) 고급 신경망 아키텍처(Advanced Deep Learning Architecture): 하이퍼스펙트럴 및 RGB 데이터를 통합하여 효율적으로 세분화 작업을 수행합니다. 3) 불리한 환경에 대한 견고성(Robustness to Adverse Conditions): 저조도 및 대기 방해와 같은 어려운 환경에서도 안정적인 성능을 보입니다.

- **Performance Highlights**: 항공 데이터셋을 다양한 환경 조건에서 훈련 및 평가하여, 제안된 모델이 경쟁 기준(comparative benchmark)에서 우수한 성능을 나타냄을 입증하였습니다. 특히, 하이퍼스펙트럴 데이터의 저해상도를 효과적으로 활용하여 경제적인 장비를 통해도 뛰어난 결과를 도출할 수 있음을 보여주었습니다.



### Fine-tuning foundational models to code diagnoses from veterinary health records (https://arxiv.org/abs/2410.15186)
Comments:
          26 pages, 5 figures

- **What's New**: 이 연구는 Colorado State University (CSU) Veterinary Teaching Hospital (VTH)의 모든 7,739개의 SNOMED-CT 진단 코드를 포함하여 대규모의 사전 훈련된 언어 모델 (Large Language Models, LLMs)을 활용하여 동물 진단 코드를 자동화하는 방법을 개선하고자 합니다.

- **Technical Details**: 이 연구는 자연어 처리 (Natural Language Processing, NLP) 기술을 사용하여 246,473개의 수동 코드화된 동물 환자 방문 기록에서 자유 텍스트 노트를 기반으로 10개의 사전 훈련된 LLM을 미세 조정 (fine-tuning) 하여, 신뢰성을 높이고 기존 연구들보다 뛰어난 성과를 보여주었습니다.

- **Performance Highlights**: 이 연구의 결과는 대규모의 레이블링된 데이터를 사용하여 상대적으로 큰 임상 LLM을 미세 조정했을 때 가장 정확한 결과를 얻을 수 있음을 보여주며, 제한된 자원과 비임상 LLM을 사용해도 유사한 결과를 얻을 수 있음을 확인했습니다.



### Action abstractions for amortized sampling (https://arxiv.org/abs/2410.15184)
- **What's New**: 이 논문에서는 강화 학습(RL) 및 생성 흐름 네트워크(GFlowNets)에서의 신용 할당(credit assignment) 및 탐색(exploration) 문제를 해결하기 위해 행동 추상화(action abstraction) 발견을 정책 최적화(policy optimization) 과정에 통합하는 방법을 제안합니다.

- **Technical Details**: 제안된 접근 방식은 고수익(high-reward) 궤적에서 공통으로 사용되는 행동 하위 시퀀스를 반복적으로 추출하여, 이를 하나의 행동으로 '청킹(chunking)' 하여 행동 공간에 추가하는 과정을 포함합니다.

- **Performance Highlights**: 합성 및 실제 환경에서의 실험 평가를 통해, 제안된 방법이 다양한 고수익 객체를 발견하는 데 있어 샘플 효율(sample efficiency)이 향상되었음을 보여주며, 특히 어려운 탐색 문제에서 그 효과가 두드러집니다. 또한, 추상화된 고차원 행동(high-order actions)은 해석 가능성을 제공하며, 행동 공간의 보상 풍경(reward landscape)의 잠재적 구조를 포착합니다.



### Enhancing Robot Navigation Policies with Task-Specific Uncertainty Managemen (https://arxiv.org/abs/2410.15178)
- **What's New**: 본 논문은 로봇 내비게이션 정책에 태스크 특화 불확실성 요구 사항을 통합하는 프레임워크를 제안합니다. Task-Specific Uncertainty Map (TSUM)을 도입하여 다양한 작업에 대한 상태 추정 불확실성의 허용 수준을 나타냅니다. 이를 통해 Generalized Uncertainty Integration for Decision-Making and Execution (GUIDE)라는 정책 조건화 프레임워크를 제안하고, RL(Reinforcement Learning) 알고리즘에 GUIDE를 통합하는 방법을 보여줍니다.

- **Technical Details**: TSUM은 특정 작업에 대한 환경의 다양한 지역에서 허용되는 상태 추정 불확실성을 공간적으로 인코딩합니다. GUIDE는 TSUM을 내비게이션 정책에 통합하여 로봇이 특정 작업의 불확실성 요구 사항을 처리할 수 있도록 합니다. GUIDE 프레임워크는 또한 Soft Actor-Critic 알고리즘과 통합되어, G-SAC 방법을 통해 로봇이 작업 완료와 불확실성 관리 간의 균형을 효과적으로 학습할 수 있도록 합니다.

- **Performance Highlights**: GUIDE는 다양한 실제 내비게이션 작업에서 평가되었으며, 기존 방법에 비해 작업 완료율에서 значные 개선을 보여주었습니다. 이 프레임워크는 간단하고 유연하게 설계되어 있어 다양한 로봇 도메인에 폭넓게 적용될 수 있습니다.



### Implicit neural representation for free-breathing MR fingerprinting (INR-MRF): co-registered 3D whole-liver water T1, water T2, proton density fat fraction, and R2* mapping (https://arxiv.org/abs/2410.15175)
- **What's New**: 이 연구는 자유 호흡(free-breathing) 조건에서 3D 전체 간(whole-liver) 수치화를 위한 MRI 기술을 개발하였습니다.

- **Technical Details**: 수량화된 방법은 8-echo spoiled gradient echo 펄스 시퀀스를 사용하며, 회전(RE) 준비를 위한 T2 자화 준비와 역전(inversion) 회복을 교차(interleave)하여 구성하였습니다. 제안된 접근 방식은 4D 및 3D implicit neural representation (INR) 기반의 신경망(neural network)으로, 움직임 변형 필드(motion deformation fields)와 정적 참조 프레임(static reference frame) MRI 하위 공간(subspace) 이미지를 동시에 학습합니다.

- **Performance Highlights**: 연구 결과, 비호흡 유지(conventional breath-holding) 스캔과 비교했을 때, 간의 T1, T2, R2*, 및 PDFF 값에서 최소한의 편향(bias)과 95% 일치 한계(narrow 95% limits of agreement)를 보였습니다.



### Uncovering Autoregressive LLM Knowledge of Thematic Fit in Event Representation (https://arxiv.org/abs/2410.15173)
Comments:
          15 pages, 3 figures

- **What's New**: 이 연구에서는 사전 훈련된 자가 회귀 LLMs(generative language models)가 주어진 사건 인수의 주제 적합성을 일관되게 추론할 수 있는지를 평가합니다. 특정한 구문적 구조와 다양한 입력 및 출력 형식을 비교하여 이들 모델의 성능을 분석하였습니다.

- **Technical Details**: 이 연구는 세 가지 축으로 평가되었습니다: 1) 다단계 논리적 추론(chain-of-thought prompting)과 간단한 프롬프트(Simpler Prompting) 비교, 2) 생성된 문장(context)와 원시 투플(<predicate, argument, role>) 비교, 3) 범주형 출력(categorical output)과 숫자형 출력(numeric output) 비교. 주제 적합성 주제(task)는 동사(predicate)와 그 인수(argument) 간의 호환성을 측정하는 작업입니다.

- **Performance Highlights**: 연구 결과, chain-of-thought reasoning이 자명한 의미 역할 레이블이 있는 데이터셋에서 더 효과적이었고, QPT 기반 방법이 모든 테스트 데이터셋에서 새로운 최첨단 성능을 설정했습니다. 제시된 생성된 문장은 특정 설정에서만 유용했으며, 여러 다른 경우에서는 결과를 저하시켰습니다.



### Budgeted Online Continual Learning by Adaptive Layer Freezing and Frequency-based Sampling (https://arxiv.org/abs/2410.15143)
- **What's New**: 본 논문은 온라인 지속 학습(Online Continual Learning, CL)에 대한 새로운 접근법을 제안합니다. 기존의 CL 알고리즘들이 단일 에폭(single epoch) 훈련을 전제로 하고, 재생 메모리 크기를 제한하는 등의 제약을 두고 있는 반면, 이 연구는 연산량과 메모리 측면에서 공정한 비교를 위해 부동소수점 연산(FLOPs)과 총 메모리 크기를 사용하자고 주장합니다.

- **Technical Details**: 제안된 방법은 ‘adaptive layer freezing’이라는 기법을 포함하여, 정보가 적은 배치에 대해 네트워크의 계층을 업데이트하지 않음으로써 계산 비용을 줄입니다. 또, 모델이 적은 반복으로 무작위 샘플을 통해 학습할 수 있도록 하는 샘플 검색(memory retrieval) 방법도 제안되었습니다. 실험적으로 CIFAR-10/100, CLEAR-10/100, ImageNet-1K 데이터셋에서 성능을 검증하였습니다.

- **Performance Highlights**: 제안된 방법은 동일한 총 리소스 예산(total budget) 하에서 기존의 최첨단 방법들과 비교하여 현저히 우수한 성능을 보여주었습니다. 많은 고성능 CL 방법들이 정해진 FLOPs와 메모리 예산 하에서 경쟁력을 가지지 못하는 반면, 제안된 방법은 여러 기준에서 두드러진 성과를 나타냈습니다.



### Generalized Flow Matching for Transition Dynamics Modeling (https://arxiv.org/abs/2410.15128)
- **What's New**: 본 논문은 전이 다이내믹스(transition dynamics) 시뮬레이션의 새로운 접근 방식을 제안합니다. 특히, 메타스테이블 상태 사이의 가능한 경로를 찾기 위해 데이터 기반(data-driven) 방법을 사용하여 국소 동역학(local dynamics)에서 비선형 보간(nonlinear interpolations)을 학습합니다.

- **Technical Details**: 제안된 방법은 국소 동역학 데이터를 바탕으로 잠재 에너지 함수(potential energy function)를 유추하고, 이를 통해 두 메타스테이블 상태 사이의 경로를 찾기 위한 일반화된 플로우 매칭 프레임워크(generalized flow matching framework)를 설정합니다. 이 과정에서 학습된 에너지 함수 아래서 두 마진 밀도(marginal densities) 사이에서 가능한 경로를 샘플링하는 벡터 필드(vector field)를 학습합니다.

- **Performance Highlights**: 합성 및 실제 분자 시스템을 통해 제안된 방법이 샘플링 가능한 경로를 효과적으로 찾는 데 있어 우수한 성능을 발휘함을 검증하였습니다.



### Reinfier and Reintrainer: Verification and Interpretation-Driven Safe Deep Reinforcement Learning Frameworks (https://arxiv.org/abs/2410.15127)
- **What's New**: 이번 연구에서는 실시간 검증-주도 해석(intepretation-in-the-loop) 프레임워크인 Reintrainer를 제안하여, 심층 강화 학습(Deep Reinforcement Learning, DRL) 모델의 신뢰성을 확보하고자 하였습니다. 기존 접근 방식의 문제점을 해결하기 위한 혁신적인 방법론입니다.

- **Technical Details**: Reintrainer는 각 반복(iteration)에서 훈련 중인 모델과 사전 정의된 속성(property) 간의 차이를 형식적 검증(formal verification)을 통해 측정하며, 각 입력 특성이 모델 출력에 기여하는 바를 해석합니다. 또한, DRL 검증 및 해석을 위한 일반적이고 기본적인 도구인 Reinfier를 개발하였습니다. Reinfier는 브레이크포인트 검색(breakpoints searching) 및 검증 주도 해석을 특징으로 하며, 간결한 제약 부호 언어( constraint-encoding language)인 DRLP와 함께 사용됩니다.

- **Performance Highlights**: Reintrainer는 여섯 개의 공개 벤치마크에서 성능 및 속성 보장을 포함하여 최신 기술(state-of-the-art)을 초월하는 성능을 보여주었습니다. 이 프레임워크는 신뢰할 수 있는 DRL 모델 개발에 중요한 기여를 할 것으로 기대됩니다.



### MELT: Materials-aware Continued Pre-training for Language Model Adaptation to Materials Scienc (https://arxiv.org/abs/2410.15126)
Comments:
          Accepted at EMNLP 2024 (Findings)

- **What's New**: 이번 논문은 MELT (MatEriaLs-aware continued pre-Training)라는 새로운 지속적인 사전 학습 방법을 소개합니다. MELT는 재료 과학에 특화된 사전 훈련된 언어 모델(PLMs)을 효율적으로 조정하기 위해 설계되었습니다. 기존의 방법들이 도메인 특수 코퍼스 구축에 초점을 맞추었다면, MELT는 코퍼스와 훈련 전략 모두를 포괄적으로 고려합니다.

- **Technical Details**: MELT의 핵심 전략은 재료 과학 코퍼스에서 소재(entity)들을 추출하고, 이를 통해 PLMs에 지식을 전이하는 것입니다. 이를 위해, 기초적인 재료 개체에서 시작하여 점진적으로 더 전문화된 지식으로 나아가는 커리큘럼을 도입합니다. 세부적으로, 필수적인 화학 개체 및 개념 구성을 위해 의미적 그래프를 구성하여 도메인 지식을 확장합니다.

- **Performance Highlights**: MELT는 다양한 벤치마크를 통해 종합적인 평가 결과, 기존의 계속된 사전 학습 방법들에 비해 뛰어난 성능을 나타냈습니다. MELT는 PLMs가 기존의 방법들보다 재료 개체를 효과적으로 표현할 수 있도록 지원하며, 재료 과학 전반에 걸쳐 넓은 적용 가능성을 보였습니다.



### Coarse-to-Fine Highlighting: Reducing Knowledge Hallucination in Large Language Models (https://arxiv.org/abs/2410.15116)
- **What's New**: 이 논문에서는 COFT라는 새로운 방법론을 제안합니다. COFT는 LLMs에서 발생하는 지식 환각(knowledge hallucination) 문제를 해결하기 위해 고안된 COarse-to-Fine highlighTing 메서드입니다. 이 방법은 긴 문맥 속에서 핵심 텍스트에 집중하여 해당 문제를 개선합니다.

- **Technical Details**: COFT는 세 가지 구성요소로 구성됩니다: 
1. 	extit{recaller}: 외부 지식 그래프(knowledge graph)를 사용하여 주어진 문맥에서 잠재적인 핵심 엔티티를 추출합니다. 
2. 	extit{scorer}: 각 엔티티의 문맥적 가중치를 계산하여 그 중요성을 측정합니다. 
3. 	extit{selector}: 동적 임계값 알고리즘을 통해 높은 문맥적 가중치를 가진 엔티티를 선택하고, 이를 기반으로 문단, 문장, 또는 단어를 강조합니다.

- **Performance Highlights**: COFT는 지식 환각 벤치마크에서 F1 스코어에서 평균 32.1% 개선성을 보여줍니다. 또한, 이는 독해(reading comprehension)와 질문 응답(question answering) 등 다양한 장기 문서 작업에서도 각각 평균 4.6% 및 최대 10.5% 개선 효과를 보입니다.



### On Designing Effective RL Reward at Training Time for LLM Reasoning (https://arxiv.org/abs/2410.15115)
- **What's New**: 본 연구에서는 LLM(대규모 언어 모델)의 추론 능력을 향상시키기 위한 보상 모델의 잠재력을 조사합니다. 특히, Outcome-supervised Reward Model (ORM)와 Process-supervised Reward Model (PRM) 같은 보상 모델을 RL(강화 학습) 훈련에서 평가하였으며, 기대와 달리 이러한 보상 모델들이 RL 훈련에서 성능을 저하시킬 수 있음을 발견하였습니다.

- **Technical Details**: ORM은 성공 보상을 추정하는 결과 보상을 생성하며, PRM은 올바른 추론 단계를 구분하는 훈련을 통해 단계별 프로세스 보상을 제공합니다. 연구에서는 PPO(정책 생산자 최적화)를 사용하여 MATH 및 GSM8K 벤치마크에서 훈련된 LLM에 이 보상 모델들을 적용했습니다. 그러나 보상 해킹 문제를 해결하기 위해 새로운 보상 정제 기술인 Clipping과 Delta를 도입하여 보상 극대화를 방지하였습니다.

- **Performance Highlights**: 평가 결과, 제안된 정제 기술은 1.5B 및 7B LLM을 RL 훈련하는 데 있어 안정성을 높였으며, 실험에서 Qwen2.5-Math-7B-Instruct와 같은 최첨단 모델이 MATH와 GSM8K 벤치마크에서 개선되었습니다.



### Incorporating Group Prior into Variational Inference for Tail-User Behavior Modeling in CTR Prediction (https://arxiv.org/abs/2410.15098)
- **What's New**: 이 논문에서는 사용자 행동 모델링에서의 새로운 접근법인 Group Prior Sampler Variational Inference (GPSVI)를 제안합니다. 이는 tail 사용자(행동이 빈약한 사용자)의 관심을 향상시키기 위해 그룹 선호를 사전(prior)으로 도입합니다.

- **Technical Details**: GPSVI는 tail 사용자의 흥미를 반영하기 위한 변량 추론 방법으로, 개인의 선호 모델링의 불확실성 추정에 따라 조정의 정도가 달라집니다. 또한, GPSVI는 volume-preserving flow를 통해 변량 추론의 표현력을 강화합니다.

- **Performance Highlights**: GPSVI는 전통적인 Attention 메커니즘으로의 반전이 가능하며, tail 사용자에게는 일관된 성능 향상을 제공하여 CTR(Click-through rate) 예측의 정확성을 높입니다. 실험 결과, GPSVI는 baseline 모델 대비 0.306%의 CTR 향상과 tail 사용자에서 0.659%의 성능 개선을 보여줍니다.



### DPVS-Shapley:Faster and Universal Contribution Evaluation Component in Federated Learning (https://arxiv.org/abs/2410.15093)
- **What's New**: 최근 인공지능 시대에서 federated learning (연합 학습)이 데이터 개인 정보 보호 문제를 해결하기 위한 새로운 접근 방식으로 부상하고 있습니다. 본 논문에서는 Dynamic Pruning Validation Set Shapley (DPVS-Shapley)라는 새로운 방법을 도입하여 참가자의 기여도를 더욱 효과적이고 공정하게 평가할 수 있는 기여 평가 메커니즘을 개발하였습니다.

- **Technical Details**: DPVS-Shapley는 원본 데이터셋을 동적으로 가지치기하여 기여 평가 프로세스를 가속화하고 평가의 정확성을 보장하는 방법론입니다. 이 방법은 유효성 검사 세트의 샘플에 서로 다른 가중치를 할당하여 어려운 예제와 쉬운 예제를 구분하고, 각 참여자의 기여도를 더 정확하게 평가할 수 있게 합니다.

- **Performance Highlights**: DPVS 방법론을 통해 다양한 연합 학습 데이터 분포 설정(i.i.d. 및 non-i.i.d.)에서 기여 계산의 효율성을 효과적으로 향상시키며, 기여 평가의 정확도를 유지할 수 있는 실험 결과를 도출하였습니다.



### A Distribution Semantics for Probabilistic Term Rewriting (https://arxiv.org/abs/2410.15081)
Comments:
          Submitted for publication

- **What's New**: 본 연구에서는 전통적인 term rewriting 규칙과 확률 (probabilities)을 결합한 시스템을 대상으로 한 확률적 term rewriting (Probabilistic Term Rewriting System, PTRS)의 새로운 분포 의미론 (distribution semantics)을 제안합니다.

- **Technical Details**: 확률적 term rewriting은 주어진 term이 어떤 값으로 환원되는 것 (reduction)의 확률을 모델링하는 데 유용한 방법론입니다. 이를 위해, 기본적으로 확률적으로 정의된 규칙들을 통해 일반적인 TRS (Term Rewriting Systems)를 선택하는 구조를 설정합니다. 또한, 각 환원 s↠t 에 대한 '설명 (explanations)' 집합을 계산하여 확률을 추정할 수 있는 방식을 제공합니다.

- **Performance Highlights**: 본 연구는 PTRS의 표준적인 개념과 몇 가지 예시를 통해 제안된 접근 방식의 유용성을 보여줍니다. 제안된 방식은 기존의 확률적 언어 (probabilistic languages)와 비교할 때, rewrite 시스템의 표현력 (expressive power)을 향상시킬 수 있는 가능성을 보이고 있습니다.



### SLIC: Secure Learned Image Codec through Compressed Domain Watermarking to Defend Image Manipulation (https://arxiv.org/abs/2410.15075)
Comments:
          accepted by ACM Multimedia Asia 2024

- **What's New**: 본 논문은 Secure Learned Image Codec (SLIC)라는 새로운 방법을 소개하며, 이는 이미지의 압축 도메인에서 워터마크를 삽입하여 이미지의 진본성을 보장하는 능동적 접근법입니다.

- **Technical Details**: SLIC는 신경망 기반의 압축 방식을 활용하여 잠재 공간에서 적대적 섭섭을 통해 워터마크를 삽입합니다. 이 방식은 다시 압축될 경우 품질이 저하되는 이미지를 생성하여 무단 수정을 방어하는 메커니즘으로 작용합니다. 특히, 신경 인코더/디코더를 미세 조정하여 워터마크의 보이지 않음과 견고함(robustness) 간의 균형을 맞춥니다.

- **Performance Highlights**: 실험 결과, SLIC는 변조된 이미지에서 가시적인 아티팩트를 생성하여 이를 방지하였고, SLIC 포맷을 이용한 이미지가 보다 높은 신뢰성을 가지는 것으로 나타났습니다.



### LLaVA-Ultra: Large Chinese Language and Vision Assistant for Ultrasound (https://arxiv.org/abs/2410.15074)
- **What's New**: 최근 멀티모달 대형 언어 모델(MLLM)이 주목받고 있으며, 이는 대화형 생성 AI의 텍스트 중심에서 멀티모달 작업으로의 전환을 촉진합니다. 특히 의료 분야에서 강력한 영향을 미치고 있습니다.

- **Technical Details**: 이 논문에서는 중국 의료 시각 대화를 위한 세밀한 적응형 VLM 아키텍처를 제안합니다. 파라미터 효율적 튜닝을 통해 정교한 비전 인코더(vision encoder)와 융합 모듈을 개발하여 미세한 의료 시각 의미를 향상시킵니다. 또한, 의료 장면에서 흔히 발생하는 데이터 중복(data redundancy)을 해결하기 위해 지식을 증류(knowledge distillation)를 이용한 가중치 점수(weighted scoring)를 사용하여 텍스트 설명을 반영하는 유효 이미지를 선별합니다.

- **Performance Highlights**: LLaVA-Ultra는 대규모 멀티모달 중국 초음파 데이터 세트를 기반으로 하여 의사들의 전문적인 텍스트를 사용하여 적절한 튜닝을 보장합니다. 세 가지 Med-VQA 데이터셋에서 LLaVA-Ultra는 다양한 지표에서 이전의 최첨단 모델을 초월하는 성능을 보여줍니다.



### Personalized Federated Learning with Adaptive Feature Aggregation and Knowledge Transfer (https://arxiv.org/abs/2410.15073)
- **What's New**: 이 논문에서는 데이터의 비독립성과 비동질성(Non-IID) 특성을 처리하기 위해 개인화된 연합 학습(pFL) 방식을 제안합니다. 제안된 방법인 FedAFK는 일반화(generalization)와 개인화(personalization)의 균형을 맞추는 새로운 기법을 포함해 더욱 향상된 모델 성능을 발휘하게 합니다.

- **Technical Details**: FedAFK는 세 가지 주요 설계 요소(모델 분리(model decoupling), 지식 전이(knowledge transfer), 적응적 특성 집계(adaptive feature aggregation))를 포함합니다. 이 방식은 글로벌 모델의 특성 추출기(feature extractor)만을 클라이언트와 서버 간의 통신 중 전송해 통신 비용을 줄이는 동시에 지역 데이터에 대한 개인화 및 클래스의 일반화 균형을 맞춥니다.

- **Performance Highlights**: FedAFK는 세 가지 데이터셋에 대해 실시된 실험에서 13개의 최첨단(pFL) 방법을 초과하는 성능을 기록하였으며, 각 라운드에서 추가적인 통신 오버헤드 없이 테스트 정확도를 향상시켜 제안된 방법의 효과성을 입증합니다.



### A Cycle Ride to HDR: Semantics Aware Self-Supervised Framework for Unpaired LDR-to-HDR Image Translation (https://arxiv.org/abs/2410.15068)
Comments:
          Submitted to IEEE

- **What's New**: 이 논문은 LDR(저 동적 범위)에서 HDR(고 동적 범위)로의 이미지 변환에서, 고품질의 짝지어진 {LDR,HDR} 데이터셋에 대한 의존도를 줄이고, 짝지어지지 않은 데이터셋을 활용하는 방법을 제안합니다. 또한, 시각적 아티팩트 제거와 의미론적 일관성을 해결하기 위해 새로운 생성기와 손실 함수를 도입하였습니다.

- **Technical Details**: 제안된 방법은 수정된 사이클 일관성 적대적 구조(modified cycle-consistent adversarial architecture)를 활용하며, CLIP(Contrastive Language-Image Pre-training) 임베딩을 통해 LDR과 복원된 HDR 간의 의미론적 일관성을 확보합니다. 또한, 생성기(generator)는 수정된 U-Net 아키텍처를 기반으로 하며, ConvLSTM 기반의 피드백 메커니즘을 통합하여 시각적 아티팩트를 줄이는 데 기여합니다.

- **Performance Highlights**: 이 연구는 여러 벤치마크 데이터셋에서 상태-of-the-art 결과를 달성하였으며, 단일 노출 LDR 이미지를 사용하여 아티팩트 없이 시각적으로 인상적인 HDR 이미지를 복원하는 데 성공하였습니다.



### Mind the Remaining: Mechanism Design for Robust Federated Unlearning (https://arxiv.org/abs/2410.15045)
- **What's New**: 연합 학습(Federated Learning) 개념이 발전하면서, 개인 정보 보호 규제를 준수하기 위해 연합 비학습(Federated Unlearning, FU) 기법이 등장하였습니다. 본 논문에서는 FU에서 클라이언트의 영향력을 제거하는 과정을 새로운 Stackelberg 게임 모델을 통해 효율적이고 안정적으로 수행하려는 메커니즘을 제안합니다.

- **Technical Details**: 서버는 최적의 결제 구조를 디자인하여 남아있는 클라이언트의 참여를 유도하며, 클라이언트는 자신들의 이익을 극대화하기 위해 전략적으로 참여 수준을 결정합니다. FU의 결과를 모델링하는 데 있어, 본 연구는 FU로 인한 부작용을 체계적으로 분석하는 종합적인 프레임워크를 제시합니다.

- **Performance Highlights**: 우리는 텍스트 및 이미지 분류 작업을 통해 방안의 성능을 실험했으며, 최악의 경우 클라이언트 성능 저하를 최대 6.62%까지 줄이고, 글로벌 안정성 향상은 최대 6.23%에 달하며 복잡한 부작용을 완화하는 효과를 나타냈습니다.



### Adversarial Training: A Survey (https://arxiv.org/abs/2410.15042)
- **What's New**: 이 논문은 적대적 훈련(Adversarial Training, AT) 관련 최신 연구들을 포괄적으로 검토하여, 다양한 적대적 공격에 대한 딥 뉴럴 네트워크의 강건성을 향상시키기 위한 기술들을 정리하며, 이 분야의 향후 연구 방향을 제안합니다.

- **Technical Details**: AT는 주로 min-max 최적화 문제로 프레임화됩니다. 모델 가중치를 조정하여 클린 샘플과 적대적 샘플을 동시에 정확히 분류하고, 고정된 모델 가중장에서 클린 입력에 작은 변화를 주어 적대적 샘플을 생성합니다. 이런 과정은 white-box 공격과 black-box 공격에 따라 달리 수행됩니다. 최근의 AT 기법들은 데이터 증강(data augmentation), 네트워크 설계(network design), 훈련 구성(training configurations) 등의 세 가지 측면에서 발전하고 있습니다.

- **Performance Highlights**: AT는 의료 이미지 분할, 자율 주행 및 이상 탐지와 같은 다양한 분야에서 효과적으로 적용되었습니다. 그러나 모델 성능은 공격 강도, 모델 아키텍처 등 여러 요인에 의해 영향을 받아, 연구자들은 다양한 평가 메트릭을 통해 이러한 성능을 평가합니다. 향후 연구에서는 AT의 일반적인 도전과제인 재앙적 과적합(catastrophic overfitting), 공정성(fairness), 성능 간의 거래(performance trade-offs), 시간 효율성(time efficiency) 등의 문제를 해결할 방향성을 탐색할 것입니다.



### A General-Purpose Multimodal Foundation Model for Dermatology (https://arxiv.org/abs/2410.15038)
Comments:
          56 pages; Technical report

- **What's New**: PanDerm는 200만 개 이상의 피부 질환의 실제 이미지를 기반으로 한 self-supervised 학습을 통해 사전 훈련된 멀티모달 피부과 기초 모델입니다. 이는 기존의 이미지 기반 딥러닝 모델보다 복잡한 임상 요구 사항을 충족하는 데 중점을 둡니다.

- **Technical Details**: PanDerm는 11개의 임상 기관에서 수집된 4가지 이미징 모달리티를 통해 2백만 개 이상의 실제 피부 질환 이미지를 학습했습니다. 28개의 다양한 데이터 세트에서 피부암 스크리닝, 표현형 평가 및 위험 분류, 신생물 및 염증성 피부 질병 진단 등 다양한 임상 작업에 대한 평가를 수행했습니다.

- **Performance Highlights**: PanDerm는 모든 평가 작업에서 최첨단 성능을 기록했으며, 레이블된 데이터의 5-10%만 사용해도 기존 모델을 초과하여 성능을 발휘했습니다. 특히, 초기 단계 멜라노마 탐지 정확도에서 임상의보다 10.2% 향상되었고, 다중 클래스 피부암 진단 정확도에서는 11% 향상되어 협업 AI 환경에서 임상의의 진단 능력을 강화했습니다.



### Enhancing Multimodal Sentiment Analysis for Missing Modality through Self-Distillation and Unified Modality Cross-Attention (https://arxiv.org/abs/2410.15029)
- **What's New**: 이번 연구에서는 텍스트 모달리티의 부재에도 불구하고 멀티모달 감정 분석을 효과적으로 수행할 수 있는 강력한 모델인 Double-Flow Self-Distillation Framework를 개발했습니다. 이 프레임워크는 Unified Modality Cross-Attention (UMCA)와 Modality Imagination Autoencoder (MIA)를 포함하고 있습니다.

- **Technical Details**: 제안된 프레임워크는 LLM 기반 모델을 사용하여 텍스트 모달리티가 없는 경우 오디오 모달리티에서 텍스트 표현을 시뮬레이션합니다. 또한, MIA 모듈을 통해 다른 두 모달리티의 정보를 보완하여 시뮬레이션된 텍스트 표현이 실제 텍스트 표현과 유사하도록 합니다. Rank-N Contrast (RNC) 손실 함수를 도입하여 시뮬레이션된 표현과 실제 표현을 정렬합니다.

- **Performance Highlights**: CMU-MOSEI 데이터셋에서 테스트한 결과, 제안된 모델은 MAE에서 뛰어난 성능을 보였으며, 텍스트 모달리티가 결여된 상황에서 다른 모델들보다 현저하게 우수한 성능을 발휘했습니다.



### A Novel Reinforcement Learning Model for Post-Incident Malware Investigations (https://arxiv.org/abs/2410.15028)
Comments:
          8 pages. arXiv admin note: substantial text overlap with arXiv:2408.01999

- **What's New**: 이 연구는 사이버 사건 대응 중 악성코드 포렌식 조사를 최적화하기 위한 새로운 강화 학습(RL) 모델을 제안합니다. 연구의 주된 목표는 잘못된 부정 판별(false negatives)을 줄이고 현재의 관행을 진화하는 악성코드 서명에 맞게 조정하여 포렌식 조사 효율성을 향상시키는 것입니다.

- **Technical Details**: 제안된 RL 프레임워크는 Q-learning과 Markov Decision Process (MDP) 기술을 활용하여 라이브 메모리 덤프에서 악성코드 패턴을 식별하도록 시스템을 훈련시킵니다. 연구는 Windows 운영 체제를 사용하여 생긴 데이터셋을 활용하여 악성코드 감염을 시뮬레이션하며, 표본 테스트 및 평가를 통제된 환경에서 수행합니다.

- **Performance Highlights**: 실험 결과, RL은 기존 방법에 비해 악성코드 탐지율을 향상시키는 것을 입증하였으며, RL 모델의 성능은 환경의 복잡성과 학습 속도에 따라 달라집니다. 연구는 RL이 악성코드 포렌식을 자동화할 수 있는 가능성을 제공하지만, 다양한 악성코드 유형에 대한 효과성 향상을 위해 보상 시스템과 피처 추출 방법의 지속적인 개선이 필요함을 결론짓습니다.



### A Recommendation Model Utilizing Separation Embedding and Self-Attention for Feature Mining (https://arxiv.org/abs/2410.15026)
- **What's New**: 본 논문에서는 정보 과부하 문제를 해결하기 위해 기존의 클릭률 예측 및 TOP-K 추천 시스템의 한계를 극복하는 새로운 추천 시스템 모델을 제안합니다. 이 모델은 분리 임베딩 크로스 네트워크(separation embedding cross-network)를 기반으로 하고 있습니다.

- **Technical Details**: 모델은 희소(feature sparsity)한 피처 벡터를 밀집(embedding)한 임베딩 벡터로 변환하기 위해 임베딩 신경망(embedding neural network) 레이어를 사용합니다. 이 모델은 별도의 차원에서 피처 교차 운영(feature cross operations)을 독립적으로 수행하여 피처 탐색의 정확성과 깊이를 향상시킵니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 복잡한 데이터 세트 처리에 있어 더 높은 적응성과 예측 정확성을 보여주며, 기존 모델의 문제를 효과적으로 해결하였습니다.



### LLM-Driven Learning Analytics Dashboard for Teachers in EFL Writing Education (https://arxiv.org/abs/2410.15025)
Comments:
          EMNLP 2024 Workshop CustomNLP4U. arXiv admin note: text overlap with arXiv:2405.19691

- **What's New**: 이 연구는 EFL(English as a Foreign Language) 작문 교육을 위한 교사 전용 대시보드의 개발을 소개하고 있습니다. LLM(대형 언어 모델)을 활용하여 학생들이 에세이 작문 시스템과 상호작용하는 방식 분석을 지원하며, ChatGPT를 통해 실시간 피드백을 통합하고 있습니다.

- **Technical Details**: 대시보드는 RECIPE 잔여물 시스템에서 수집된 데이터를 분석하며, ChatGPT 통합을 통해 EFL 교육의 효과성을 높이기 위한 NPL(자연어 처리) 및 HCI(인간-컴퓨터 상호작용) 통찰력을 결합합니다. 대시보드는 학생 행동을 모니터링하고, ChatGPT와의 비교육적 상호작용을 식별하며, 학습 목표와 수업 전략의 정렬을 지원합니다.

- **Performance Highlights**: 대시보드 프로토타입은 여러 차트와 구성요소를 포함하고 있으며, 주간 채팅 빈도 및 에세이 평가 요약을 시각화하여 교사가 학생의 ChatGPT와의 상호작용을 신속하게 확인할 수 있도록 합니다. 또한 AI를 통한 오용 식별 기능을 통해 교사는 비교육적 사용 사례를 효율적으로 체크할 수 있습니다.



### DM-Codec: Distilling Multimodal Representations for Speech Tokenization (https://arxiv.org/abs/2410.15017)
- **What's New**: 본 논문에서는 DM-Codec이라고 불리는 새로운 음성 토크나이저를 제안하며, 이 모델이 정밀한 음성 표현을 위해 맥락적 정보(contextual information)를 통합한다는 점이 특징입니다.

- **Technical Details**: DM-Codec은 Residual Vector Quantizer (RVQ)를 포함한 인코더-디코더 아키텍처를 채택하고, 언어 모델(LM)과 자기 지도 학습(self-supervised learning) 음성 모델(SM)을 사용하여 다중 모드 표현(모음, 의미, 맥락)을 효과적으로 추출합니다. 이 과정은 음성 입력을 텍스트로 전사한 후, LM을 통해 얻은 맥락적 표현을 결합하여 진행됩니다.

- **Performance Highlights**: DM-Codec은 LibriSpeech 데이터셋에서 기존 최첨단 음성 토크나이저 모델들에 비해 Word Error Rate (WER)를 최대 13.46%까지 감소시키고, Word Information Lost (WIL)는 9.82% 줄이며, 음성 품질은 5.84% 및 이해도는 1.85% 개선되었습니다.



### Transit Pulse: Utilizing Social Media as a Source for Customer Feedback and Information Extraction with Large Language Mod (https://arxiv.org/abs/2410.15016)
Comments:
          17 pages, 21 figures

- **What's New**: 본 논문은 사회적 미디어 데이터를 활용하여 대중 교통 사용자 피드백을 분석하기 위한 최신 방법론을 제안합니다. 기존의 방법들은 사전 정의된 주제 라벨에 의존했으나, 본 연구에서는 LLM(Large Language Model)을 활용하여 보다 포괄적인 인사이트를 제공합니다.

- **Technical Details**: 제안된 방법은 Llama 3라는 LLM을 사용하여 사회적 미디어의 대중 교통 관련 정보, 감정 및 풍자 감지, 비정상적인 시스템 문제 식별, 그리고 위치 데이터를 분석합니다. 정보 추출 파이프라인에 RAG(Retrieval-Augmented Generation) 접근 방식을 통합하여 외부 지식을 모델에 결합합니다.

- **Performance Highlights**: 전통적인 NLP 접근 방식과 비교하여 LLM을 활용한 이 방법은 사용자 트윗 데이터에 대한 분석 성능에서 유망한 결과를 보여주었으며, 대중 교통 기관의 대응 능력을 개선하고 실질적인 인사이트를 제공합니다.



### DST-TransitNet: A Dynamic Spatio-Temporal Deep Learning Model for Scalable and Efficient Network-Wide Prediction of Station-Level Transit Ridership (https://arxiv.org/abs/2410.15013)
Comments:
          16 pages, 22 figures. Accepted by TRB 2025

- **What's New**: 이 논문은 비교적 높은 예측 정확도를 유지하며 대규모 라이드리쉬프 예측을 위한 DST-TransitNet이라는 하이브리드 심층학습 모델을 소개합니다. 이 모델은 그래프 신경망(GNN)과 순환 신경망(RNN)을 통합하여 시간 및 공간 상관 관계를 동적으로 통합합니다.

- **Technical Details**: DST-TransitNet 모델은 심층 학습을 기반으로 하며, 시간 시계열 분해 프레임워크를 사용하여 예측의 정확성과 해석 가능성을 높입니다. 모델은 Bogota의 BRT 시스템 데이터를 사용하여 테스트되었으며, 다양한 사회적 시나리오에서 수행되었습니다.

- **Performance Highlights**: DST-TransitNet은 최신 기계 학습 모델에 비해 예측 정확도, 효율성 및 강건성에서 우수한 성능을 보여주었습니다. 또한 긴 예측 간격에서도 안정성을 유지하여 실용적인 적용 가능성을 나타냅니다.



### Pathologist-like explainable AI for interpretable Gleason grading in prostate cancer (https://arxiv.org/abs/2410.15012)
Comments:
          58 pages, 15 figures (incl. supplementary)

- **What's New**: 이번 연구에서는 전세계 남성에서 가장 흔한 암인 전립선암의 공격성을 평가하기 위한 새로운 데이터셋을 소개합니다. 이 데이터셋은 1,015개의 조직 마이크로어레이 코어 이미지로 구성되며, 54명의 국제 병리학자에 의해 주석이 달립니다.

- **Technical Details**: 이 연구는 Gleason 점수 예측을 위한 U-Net 아키텍처 기반의 AI 시스템을 개발하였습니다. 이 시스템은 병리학자의 용어를 활용하여 예측을 수행하며, 포스트 호크(post-hoc) 설명 가능성 방법을 우회합니다.

- **Performance Highlights**: Gleason 패턴 세분화 성능에서, 설명이 포함된 모델은 Dice score가 0.713 $	imes$ 0.003으로, Gleason 패턴에 직접 학습된 모델의 0.691 $	imes$ 0.010을 초과하는 성능을 보여주었습니다. 또한, 소프트 레이블(soft labels)을 사용하여 데이터의 내재적 불확실성을 캡처하였습니다.



### FlexMol: A Flexible Toolkit for Benchmarking Molecular Relational Learning (https://arxiv.org/abs/2410.15010)
- **What's New**: 이 논문에서는 약물 관계 학습(Molecular Relational Learning, MRL)의 주요 한계를 극복하기 위해 FlexMol이라는 포괄적인 툴킷을 소개합니다. FlexMol은 다양한 데이터셋과 성능 지표를 통해 다양한 모델 아키텍처의 구축 및 평가를 지원하여 MRL 개발을 단순화합니다.

- **Technical Details**: FlexMol은 16개의 약물 인코더, 13개의 단백질 서열 인코더, 9개의 단백질 구조 인코더, 7개의 상호작용 레이어로 구성된 강력한 인코더 스위트를 제공합니다. 이 툴킷은 사용자가 70,000개 이상의 독특한 모델 아키텍처 조합을 동적으로 구성할 수 있도록 지원합니다. 이는 MRL 모델의 동적 구축을 위한 유연한 API로 가능합니다.

- **Performance Highlights**: FlexMol은 DTI, DDI, PPI 설정에서 벤치마크 결과와 코드 예시를 제공하여 모델의 효율성을 입증합니다. 특히, 연구자들이 MRL 모델을 최소한의 노력으로 구축, 평가 및 비교할 수 있는 가능성을 강조합니다.



### A comparative study of NeuralODE and Universal ODE approaches to solving Chandrasekhar White Dwarf equation (https://arxiv.org/abs/2410.14998)
- **What's New**: 이번 연구에서는 과학 기계 학습(Scientific Machine Learning) 분야의 두 가지 기둥인 Neural Ordinary Differential Equations(Neural ODEs)와 Universal Differential Equations(UDEs)를 Chandrasekhar White Dwarf Equation(CWDE)에 적용하였습니다. CWDE는 별의 생애 주기를 이해하는 데 필수적인 방정식으로, 백색 왜성과 중심 간의 밀도 관계를 설명합니다.

- **Technical Details**: 연구에서는 Julia 프로그래밍 언어를 사용하여 Neural ODEs 및 UDEs의 예측 및 예측 성능을 효율적으로 보여주었습니다. 특히, 예측 실패가 발생하는 시점을 지칭하는 'forecasting breakdown point'를 도입하였으며, 이는 신경망 아키텍처, 활성화 함수, 최적화 도구에 대한 통찰을 제공합니다.

- **Performance Highlights**: Neural ODEs와 UDEs 모두 CWDE ODE 시스템의 예측 및 예측을 효과적으로 수행할 수 있는 가능성을 보여주었습니다. 이 연구는 다양한 과학 분야에서 예측 작업에 대한 Scientific Machine Learning 프레임워크의 적용 가능성을 탐색하는 새로운 기회를 제시합니다.



### Improving Pronunciation and Accent Conversion through Knowledge Distillation And Synthetic Ground-Truth from Native TTS (https://arxiv.org/abs/2410.14997)
Comments:
          Submitted to ICASSP 2025

- **What's New**: 이번 연구에서는 기존의 억양 변환(accent conversion, AC) 방법을 발전시켜 비원어민 화자의 발음을 개선하는 새로운 접근법을 제안합니다. 이 시스템은 비원어민 오디오와 해당 전사된 스크립트를 입력받아 원주율(native pronunciation)의 개선된 오디오 데이터를 생성합니다.

- **Technical Details**: 제안된 시스템은 end-to-end VITS(Variational Inference Text-to-Speech) 프레임워크를 활용하여 고품질의 파형 재구성을 수행합니다. 이 시스템은 비원어민 화자의 발음을 향상시키고 원래 화자의 정체성을 유지하면서 원어민 억양에 가까운 오디오를 생성합니다. 또한, 음성 특징을 모듈화하여 각 요소를 조합하는 방식을 사용하여 기존의 AC 프로세스에서의 비대칭적 시간 조정을 극복합니다.

- **Performance Highlights**: 평가 결과에 따르면, 제안된 시스템은 발음 개선을 통해 그 효과를 입증하며, 억양 변환 과정에서 고유한 화자 정체성도 유지하는 데 성공하였습니다.



### NeuralMAG: Fast and Generalizable Micromagnetic Simulation with Deep Neural Nets (https://arxiv.org/abs/2410.14986)
- **What's New**: Micromagnetics 분야에서 NeuralMAG라는 새로운 딥러닝 기반 접근법을 제안하여, 기존의 느린 시뮬레이션 속도를 혁신적으로 향상시켰습니다.

- **Technical Details**: NeuralMAG는 Landau-Lifshitz-Gilbert (LLG) 방정식을 사용하는 기존의 반복적 구조를 따르면서, U-shaped neural network (Unet)를 통해 demagnetizing field/N를 작성합니다. Unet는 다양한 스케일에서 집합된 스핀을 추출하고, 각 스케일의 지역 상호작용을 학습하는 encoder를 포함하고 있으며, decoder는 서로 다른 스케일의 지역 상호작용을 축적하여 글로벌 합성을 근사합니다. 이 분할 및 누적 방식 덕분에 시간 복잡도가 O(N)으로 개선되었습니다.

- **Performance Highlights**: NeuralMAG는 여러 샘플 크기, 형태 및 재료 설정에 대해 두 가지 micromagnetics 작업에서 평가되었으며, 이를 통해 기존의 시뮬레이션보다 현저한 속도 증가가 이루어졌음을 보여주었습니다.



### Reflexive Guidance: Improving OoDD in Vision-Language Models via Self-Guided Image-Adaptive Concept Generation (https://arxiv.org/abs/2410.14975)
Comments:
          The first two authors contributed equally

- **What's New**: 이 논문은 대규모 멀티모달 데이터를 기반으로 훈련된 대형 비전-언어 모델(LVLMs)의 Out-of-Distribution Detection (OoDD) 기능을 평가하고 분석합니다. 특히, LVLM의 신뢰성을 높이기 위한 새로운 접근법인 Reflexive Guidance (ReGuide)를 제안하여 OoDD 성능을 강화하는 방법을 제시합니다.

- **Technical Details**: Reflexive Guidance (ReGuide) 접근법은 두 단계로 이루어져 있습니다. 첫 번째 단계에서는 LVLM이 주어진 이미지에 대해 의미적으로 유사한 개념(근접 OoD)과 의미적으로 비유사한 개념(원거리 OoD) 두 그룹을 제안하도록 합니다. 두 번째 단계에서는 이 제안된 개념들을 보조 OoD 클래스로 활용하여 OoDD 성능을 향상시키려고 합니다. 이 연구는 이미지 입력을 활용하여 OoDD를 위한 정보 제공 텍스트를 생성하는 첫 번째 사례로, 모델에 구애받지 않고 광범위하게 적용 가능합니다.

- **Performance Highlights**: ReGuide는 오픈 소스 모델의 성능을 크게 향상시켰으며, 이로 인해 상용 모델과의 경쟁력을 갖추게 되었습니다. 특히, GPT-4o 모델은 ReGuide 적용 후 근접 OoD 데이터셋에서 퍼포먼스가 강화되었습니다. 이 연구는 LVLM을 통해 자동 생성된 이미지 적응 개념 제안을 이용하여 OoDD를 안내하는 방법의 효능을 강조합니다.



### Taming the Long Tail in Human Mobility Prediction (https://arxiv.org/abs/2410.14970)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 이 논문에서는 사용자 이동 예측을 위한 Long-Tail Adjusted Next POI Prediction (LoTNext) 프레임워크를 제안합니다. 이 프레임워크는 사용자-POI 상호 작용 그래프에서 'long-tailed' 노드의 영향을 줄이기 위한 Long-Tailed Graph Adjustment 모듈과, 로짓 점수 및 샘플 가중치 조정 전략을 통한 손실 조정을 목표로 하는 Long-Tailed Loss Adjustment 모듈로 구성됩니다.

- **Technical Details**: LoTNext는 사용자-POI 상호 작용 그래프 내 잡음과 long-tailed 노드를 줄이기 위해 Long-Tailed Graph Adjustment 모듈을 활용합니다. 손실 과중을 방지하기 위해 Long-Tailed Loss Adjustment 모듈을 도입하여 head POI와 tail POI 데이터 간의 손실을 잘 조절합니다. 또한 보조 예측 작업을 통해 POI 특성과 공간-시간 정보의 융합을 촉진합니다.

- **Performance Highlights**: 실험 결과, LoTNext는 두 개의 실제 궤적 데이터 세트에서 기존의 최첨단 방법들보다 유의미하게 성능이 향상되는 것으로 나타났습니다. 이러한 성과는 LoTNext가 POI 예측의 long-tail 문제를 효과적으로 해결함을 반영합니다.



### LangGFM: A Large Language Model Alone Can be a Powerful Graph Foundation Mod (https://arxiv.org/abs/2410.14961)
Comments:
          under review

- **What's New**: GFMBench라는 26개의 데이터셋을 포함하는 체계적이고 포괄적인 벤치마크를 제안하며, 새로운 GFM인 LangGFM을 소개합니다. LangGFM은 대규모 언어 모델에 전적으로 의존하여 그래프의 텍스트화 원칙을 탐구하고, 그래프 증강 및 셀프 슈퍼바이즈드 학습(ssupervised learning) 기술을 재활용하여 현재 기술 수준을 초월하는 성과를 달성합니다.

- **Technical Details**: GFMs(그래프 기초 모델)은 다양한 그래프 학습 작업을 포괄할 수 있는 통합 접근 방식을 채택하여, 그래프를 표준 그래프 교환 포맷(GraphML)으로 표시하고 작업 지침으로 LLM을 직접 파인튜닝합니다. 이는 노드 수준, 엣지 수준 및 그래프 수준의 다양한 예측 파이프라인을 활성화하여 유연하게 작업할 수 있도록 합니다.

- **Performance Highlights**: LangGFM은 GFMBench에서 최첨단 성능을 달성하며, 특히 전문 도메인에서도 높은 성과를 거두어 주목받고 있습니다. 이러한의 성과는 기존 방법론에 비해 새로운 기초 모델을 설정하고, 향후 연구 및 개발을 위한 귀중한 통찰을 제공합니다.



### Offline-to-online Reinforcement Learning for Image-based Grasping with Scarce Demonstrations (https://arxiv.org/abs/2410.14957)
- **What's New**: 이번 연구는 오프라인에서 온라인으로 강화 학습(O2O RL) 알고리즘을 제안하며, 실제 이미지 기반 로봇 진공 잡기 작업에서 적은 수의 인간 시연을 통해 학습할 수 있는 새로운 방법론을 다룹니다. 일반적인 행동 복제(BC) 방법이 실패하는 상황에서도 가능한 성과를 보여줍니다.

- **Technical Details**: 제안된 O2O RL 알고리즘은 오프-정책(actor-critic) 알고리즘의 타깃 네트워크를 신경 탄젠트 커널(neural tangent kernel)에서 영감을 받은 정규화 기법으로 대체합니다. 이 방법은 'Simplified Q'로 명명되며, 실제 로봇 작업에서 단 2시간의 상호 작용으로 90% 이상의 성공률을 기록합니다.

- **Performance Highlights**: 제안된 방법은 50개의 인간 시연만으로 성공률 90% 이상을 달성하며, 동일한 데이터량을 사용하는 BC 및 두 가지 일반적인 RL 알고리즘보다 우수한 성능을 보입니다. 이를 통해 비전 백본 프리 트레이닝(pretraining)이 필요하지 않음을 입증했습니다.



### Optimally Solving Colored Generalized Sliding-Tile Puzzles: Complexity and Bounds (https://arxiv.org/abs/2410.14947)
Comments:
          WAFR 2024 Conference Version

- **What's New**: 이번 연구에서는 일반화된 슬라이딩 타일 퍼즐(Generalized Sliding-Tile Puzzle, GSTP)의 확장인 색상 일반화 슬라이딩 타일 퍼즐(Colored Generalized Sliding-Tile Puzzle, CGSP)을 제안하며, 이는 타일의 구별 가능성을 도입하여 실세계 응용에서 보다 높은 유용성을 제공합니다.

- **Technical Details**: CGSP는 다양한 색상을 가진 타일이 동시에 움직이는 문제를 다루며, NP-hard한 문제로 수학적으로 정립되었습니다. 특히, CGSP의 하위 문제인 이진 일반화된 슬라이딩 타일 퍼즐(Binary Generalized Sliding-Tile Puzzle, BGSP)와 부분 색상 일반화 슬라이딩 타일 퍼즐(Partially-Colored Generalized Sliding-Tile Puzzle, PGSP)의 복잡성과 해결 가능한 시간 제한을 분석하였습니다.

- **Performance Highlights**: 이 연구는 BGSP와 PGSP의 최적 해법을 찾는 것이 NP-hard임을 증명하며, 다양한 상황에 대한 하한 및 상한을 결정하였습니다. 특히, BGSP의 경우 상한과 하한이 로그적 차이에 따라 다르며, 이 결과들은 향후 고유 물류 및 자율주행 시스템의 설계에 기여할 수 있습니다.



### DEL-Ranking: Ranking-Correction Denoising Framework for Elucidating Molecular Affinities in DNA-Encoded Libraries (https://arxiv.org/abs/2410.14946)
- **What's New**: 이번 연구에서는 DEL(DNA-encoded library) 스크리닝에서의 단백질-리간드 상호작용 탐지를 개선하기 위한 새로운 분포 교정 및 디노이징 프레임워크인 DEL-Ranking을 제안하고 있습니다. 이 프레임워크는 읽기 수의 신뢰성을 높이기 위한 두 가지 주요 혁신을 포함합니다: 상대적 크기 관계를 수정하는 새로운 랭킹 손실 기능과 자기 학습(self-training) 및 일관성 손실(consistency loss)을 활용하는 반복 알고리즘입니다.

- **Technical Details**: DEL-Ranking은 Pair-wise Soft Rank (PSR)와 List-wise Global Rank (LGR)라는 두 가지 제약 조건을 도입하여 읽기 수 분포의 복잡성을 모델링합니다. 또한, Activity-Referenced Distribution Correction (ARDC) 프레임워크를 통해 예측된 읽기 수 값과 잠재적 결합 친화도(binding affinity) 간의 상관 관계를 최적화합니다.

- **Performance Highlights**: 다양한 DEL 데이터세트에서의 엄격한 평가 결과, DEL-Ranking은 여러 상관 지표에서 우수한 성능을 보였으며, 결합 친화도 예측 정확도에서 상당한 향상이 나타났습니다. 이 모델은 여러 단백질 타겟에 대한 제로-샷 제너럴리제이션(zero-shot generalization)이 가능하며, 화합물 결합 친화도를 결정하는 잠재적 모티프를 성공적으로 식별하였습니다.



### Water quality polluted by total suspended solids classified within an Artificial Neural Network approach (https://arxiv.org/abs/2410.14929)
Comments:
          42 pages, 8 figures and 2 tables

- **What's New**: 이번 연구는 고형물로 인한 수질 오염을 분석하기 위한 인공지능 신경망(artificial neural network) 프레임워크의 적용을 다룹니다. 전통적인 수질 오염 평가 방법이 시간과 자원을 많이 소모하는 데 비해, 본 연구에서는 데이터 기반 모델을 통해 문제를 해결하고자 했습니다.

- **Technical Details**: 모델은 총 용해 고형물(total suspended solids) 데이터셋을 활용하여 전이 학습(transfer learning) 접근 방식을 이용한 컨볼루션 신경망(convolutional neural network)을 통해 훈련되었습니다. 다양한 입력 변수에 따라 저, 중, 고 오염 수준을 정확히 예측하는 것을 목표로 합니다.

- **Performance Highlights**: 우리 모델은 예측 정확도가 높아 전통적인 통계적 방법보다 속도와 신뢰성 측면에서 우수한 성능을 보였습니다. 이 결과는 인공지능 신경망 프레임워크가 수질 오염의 실시간 모니터링 및 관리에 효과적인 도구로 활용될 수 있음을 시사합니다.



### Cooperation and Fairness in Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2410.14916)
Comments:
          Manuscript accepted in ACM Journal on Autonomous Transportation Systems

- **What's New**: 본 논문에서는 다수의 자율 에이전트가 자원 제약이 있는 환경에서 공정성과 효율성을 조화롭게 달성하도록 학습하는 문제를 마르(MARL) 기법을 통해 해결하고자 합니다.

- **Technical Details**: 여기서 우리는 서로 다른 에이전트들이 이동하는 데 소요된 거리의 변동계수의 역수(coefficient of variation의 reciprocal)를 공정성의 척도로 사용하여, 에이전트가 효율성을 크게 희생하지 않고 공정하게 학습할 수 있는지를 조사합니다. 구체적으로, 비용 공정성을 달성하기 위해 미니-맥스(Min-Max) 공정 거리 목표 할당을 사용하고 공정성을 촉진하는 보상 항목을 포함하여 협력적인 다중 에이전트 강화 학습(MARL)을 수행합니다.

- **Performance Highlights**: 우리의 모델은 무작위 할당을 통해 훈련된 기준 모델에 비해 평균적으로 14%의 효율성 향상과 5%의 공정성 향상을 이루었으며, 최적의 효율적 할당을 통해 훈련된 모델보다 평균 21%의 공정성 향상을 달성했습니다. 이러한 공정성 향상은 단지 7%의 효율성 감소를 초래하며, 에이전트가 특정 형상으로 연대할 수 있는 환경에서도 모델을 재학습할 필요없이 적용이 가능합니다.



### A Hybrid Defense Strategy for Boosting Adversarial Robustness in Vision-Language Models (https://arxiv.org/abs/2410.14911)
- **What's New**: 본 연구에서는 여러 공격 전략과 고급 머신러닝 기법을 통합하여 Vision-Language Models (VLMs)의 강인성을 크게 향상시키는 새로운 적대적 훈련 프레임워크를 제안합니다.

- **Technical Details**: 기존의 적대적 훈련 방법들은 FGSM, AutoAttack, DeepFool과 같은 모델을 사용하여 적대적 예제 생성에 초점을 맞추었으며, 고정된 왜곡 노름(fixed perturbation norms)이나 미리 정의된 공격 패턴(predefined attack patterns)과 같은 강력한 가정에 의존했습니다. 제안된 방법은 다양한 적대적 공격에 대해 VLM의 강인성을 크게 향상시킵니다.

- **Performance Highlights**: CIFAR-10 및 CIFAR-100과 같은 실제 데이터셋에서 실험한 결과, 제안된 방법은 모델의 강인성을 크게 향상시켰으며, 세밀하게 조정된 CLIP 모델은 적대적으로 왜곡된 이미지에서 43.5%의 정확도를 달성했습니다. 반면, 기준 모델은 4%에 그쳤습니다. 또한, 신경망 모델은 98%의 높은 정확도를 기록하였고, XGBoost 모델은 예측 작업에서 85.26%의 성공률을 달성했습니다.



### From Test-Taking to Test-Making: Examining LLM Authoring of Commonsense Assessment Items (https://arxiv.org/abs/2410.14897)
Comments:
          Accepted at Findings of EMNLP 2024

- **What's New**: 이번 연구에서는 LLM(대형 언어 모델)이 일반 상식 추론을 평가하기 위한 테스트 항목을 생성할 수 있는지를 조사합니다. LLM이 COPA(Choice of Plausible Alternatives) 벤치마크 스타일의 항목을 작성하도록 유도한 후, 이를 인간 평가자 및 LLM 자신이 응답한 결과를 분석합니다.

- **Technical Details**: 저자들은 COPA 벤치마크를 기준으로 LLM의 항목 생성 능력을 평가하기 위해, GEN-COPA 항목의 질이 오리지널 COPA 항목의 저자 기준에 얼마나 부합하는지를 분석합니다. 선택한 LLM 모델에는 bloom, falcon, llama2, mistral, mpt, phi 모델이 포함되며, 각각의 성능을 비교합니다.

- **Performance Highlights**: 연구 결과, 오리지널 COPA 벤치마크에서 좋은 성과를 낸 LLM이 자가 생성한 Gen-COPA 항목에서도 높은 정확도를 기록하는 경향이 있음을 발견했습니다. 이는 LLM의 저작 능력이 질문 응답 능력과 밀접하게 연결되어 있음을 시사합니다.



### Truncated Consistency Models (https://arxiv.org/abs/2410.14895)
- **What's New**: 이 논문에서는 Truncated Consistency Models (TCM)을 제안하여 확산 모델의 일관성을 훈련하는 새로운 접근 방식을 제시합니다. TCM은 초기 단계에서의 디노이징(denoising) 작업을 무시하고, 생성을 중심에 두는 방법으로 성능을 향상시킵니다.

- **Technical Details**: 논문은 PF ODE (확률 흐름 보통 미분 방정식)에서 초기 노이즈로부터 직접 데이터의 솔루션을 예측하는 일관성 모델의 훈련을 다룹니다. TCM은 전체 시간 범위 [0,T] 대신 자른 시간 범위 [t′,T]를 사용하여 훈련합니다. 두 단계의 훈련 절차를 통해 모델이 디노이징과 생성 작업의 균형을 유지하도록 합니다.

- **Performance Highlights**: CIFAR-10 및 ImageNet 64×64 데이터셋에서 TCM은 iCT 및 iCT-deep과 같은 최신 모델보다 더 나은 FID(Fréchet Inception Distance) 성능을 보였습니다. 특히, TCM은 동일한 네트워크 크기로도 경쟁력 있는 결과를 보여주며, 훈련 안정성 또한 향상시킵니다.



### Self-Satisfied: An end-to-end framework for SAT generation and prediction (https://arxiv.org/abs/2410.14888)
Comments:
          22 pages

- **What's New**: 이번 논문에서는 SAT 문제의 이해를 돕기 위한 여러 가지 발전 사항을 소개하고 있습니다. 하드웨어 가속 알고리즘과 비전 태스크에 일반적으로 적용되는 transformer 아키텍처를 활용한 기하학적 SAT 인코딩을 포함하여, transformer 아키텍처 내에서 시퀀스 길이 표현을 줄이는 기술인 head slicing을 도입했습니다.

- **Technical Details**: 제안된 모델인 Satisfiability Transformer (SaT)는 GPU에서 작동하여 SAT 문제의 데이터 생성을 위한 알고리즘과 훈련을 수행하는 데 초점을 맞추고 있습니다. 이러한 접근 방식은 수천 개의 변수와 수만 개의 절을 소화할 수 있도록 확장 가능합니다. 또한 SATComp 2022에서 얻은 데이터를 사용하여 SAT 예측 작업에서 우리의 아키텍처를 검증했습니다.

- **Performance Highlights**: 우리의 순수 기계 학습 접근 방식은 기존 연구와 비교하여 유사한 예측 정확도를 달성하였으나, 이전과 비교할 때 훨씬 더 큰 문제에 대해서 성능을 발휘했습니다. 이를 통해 SAT 데이터의 본질과 기계 학습 모델 훈련에 적합성을 검토하는 중요한 실험 결과도 제시하고 있습니다.



### Vital Insight: Assisting Experts' Sensemaking Process of Multi-modal Personal Tracking Data Using Visualization and LLM (https://arxiv.org/abs/2410.14879)
- **What's New**: 이 연구는 개인 추적 데이터의 해석을 촉진하기 위해 HCI(인간-컴퓨터 상호작용) 연구자들이 어떻게 지원할 수 있는지를 탐구하며, Vital Insight라는 증거 기반의 'sensemaking' 시스템을 개발했습니다. 이 시스템은 시각화와 대형 언어 모델을 통해 직접적인 표현과 간접적인 추론을 결합합니다.

- **Technical Details**: Vital Insight 시스템은 전문가들이 다양한 감지 데이터를 통해 인사이트를 생성할 수 있도록 지원하며, 14명의 전문가와 함께 사용자 테스트를 실시하여 설계 시사점을 종합했습니다. 전문가들은 이 시스템을 사용하여 데이터를 맥락화하고 개인 행동에 대한 합리적인 해석을 제공할 수 있었습니다.

- **Performance Highlights**: 사용자 테스트 결과, 전문가들은 Vital Insight 시스템이 신뢰성과 명료성에서 높은 평가를 받아 의미 기반의 'sensemaking'을 효과적으로 수행할 수 있게 해준다고 보고했습니다. 이 연구는 개인 추적 데이터에 대한 전문가의 'sensemaking' 과정을 명확히 하고, 시스템 설계에 대한 중요한 시사점을 제공합니다.



### How to Evaluate Reward Models for RLHF (https://arxiv.org/abs/2410.14872)
- **What's New**: 본 논문에서는 RLHF(고객 피드백으로부터의 강화 학습)를 통해 강력한 언어 모델을 생성할 수 있는 보상 모델의 능력을 정량화하기 위한 새로운 벤치마크를 소개합니다. 기존의 gold-standard 접근법은 전체 RLHF 학습 파이프라인을 실행하고 직접적으로 다운스트림 LLM 성능을 평가하는 것입니다. 하지만 이는 비용이 과도하게 비쌉니다. 이를 해결하기 위해, 다운스트림 LLM 성능을 예측하는 모델을 구축하였으며, 다양한 종류의 프록시 작업을 통해 보상 모델을 평가합니다.

- **Technical Details**: 저자들은 12개의 다양한 도메인에서 12개의 메트릭을 평가하여 보상 모델의 성능을 측정합니다. 프록시 작업으로는 대규모 인간 선호 및 검증 가능한 정답 선호 데이터셋을 활용하였으며, 이 작업에서 수집된 데이터와 발견 사항을 Preference Proxy Evaluations (PPE)라는 형태로 정리하였습니다. PPE는 RLHF 이후의 실제 인간 선호 성과와 명시적으로 연결된 최초의 보상 모델 벤치마크입니다.

- **Performance Highlights**: PPE는 20개의 주요 LLM과 121개 이상의 언어에서 수집된 16,038개의 레이블된 인간 선호 쌍과 2,555개의 프롬프트 데이터셋을 포함하고 있습니다. 각 프롬프트는 32개의 서로 다른 응답 옵션이 제공되어 총 81,760개의 응답으로 구성됩니다. PPE는 12개의 다른 메트릭에서 보상 모델을 평가하며, RLHF 결과와 직접적으로 연결된 유일한 보상 모델 벤치마크로 자리잡고 있습니다.



### DFlow: Diverse Dialogue Flow Simulation with Large Language Models (https://arxiv.org/abs/2410.14853)
Comments:
          16 pages

- **What's New**: 이 논문에서는 기존의 데이터 증강 방법들이 대화 수준에서의 작업 로직 다양성에 대한 중요성을 간과하고 있음을 지적하고, 이를 해결하기 위한 새로운 데이터 증강 방법을 제안합니다.

- **Technical Details**: LLMs(대형 언어 모델)를 활용하여 의사 결정 트리 구조의 작업 계획을 생성하며, 이를 바탕으로 생성된 대화 흐름을 통해 다수의 대화를 제작합니다. 이 방법은 각 작업 흐름을 따라가며 다단계 대화를 생성하는 데 중점을 두게 됩니다. 실험을 통해 3,886개의 대화 흐름으로 구성된 작업 지향 대화 데이터 세트를 생성하였습니다.

- **Performance Highlights**: DFlow 데이터셋에 대해 세밀한 경험적 실험을 수행한 결과, 이 데이터셋으로 미세 조정된 7B 언어 모델이 GPT-4와 같은 강력한 LLM보다 우수한 성과를 보였습니다. 이는 작업 로직과 제약 조건을 잘 따르는 다채로운 다단계 대화를 합성할 수 있는 방법론의 유효성을 입증합니다.



### Making LLMs Vulnerable to Prompt Injection via Poisoning Alignmen (https://arxiv.org/abs/2410.14827)
- **What's New**: 이번 연구에서는 Prompt Injection Attack의 효율성을 증가시키기 위해 Alignment 과정에 독성 샘플을 주입하는 방법을 제안합니다. 이는 기존의 공격 방식과는 다른 접근법으로, 대형 언어 모델(LLM)의 취약점을 이용합니다.

- **Technical Details**: 우리는 PoisonedAlign라는 새로운 방법을 소개하며, 이는 공격자가 주입 프롬프트를 사용하여 독성 샘플을 생성하도록 합니다. 이 방법은 Alignment 데이터의 일부가 독성으로 변모할 때 LLM이 Prompt Injection에 더 취약해지도록 합니다.

- **Performance Highlights**: PoisonedAlign 방법을 활용했을 때, 10%의 Alignment 데이터가 독성으로 변모되면 LLM의 공격 성공률이 평균 0.33 증가하며, 이는 LLM의 기초 능력을 유지한 채 이루어집니다.



### SPRIG: Improving Large Language Model Performance by System Prompt Optimization (https://arxiv.org/abs/2410.14826)
- **What's New**: 이번 연구에서는 시스템 프롬프트(System Prompt)를 최적화하기 위한 새로운 방법인 SPRIG를 제안합니다. 이는 유전 알고리즘(genetic algorithm)에 기반하여 여러 구성 요소에서 프롬프트를 반복적으로 생성하여 모델의 성능을 극대화하는 방식입니다.

- **Technical Details**: SPRIG는 시스템 프롬프트의 내용을 대체 가능한 300개의 프롬프트 구성 요소로부터 구성됩니다. 이 구성 요소들은 좋음 속성(good property), 역할(role), 스타일(style), 감정(emotion) 등 9가지 범주로 나뉘어 있습니다. 최적화 과정은 비선형(beam search) 기반의 편집(edit-based) 방식을 통해 수행됩니다.

- **Performance Highlights**: SPRIG로 최적화된 시스템 프롬프트는 각 개별 작업에 최적화된 작업 프롬프트(task prompt)와 동등한 성능을 보여주며, 두 종류의 최적화를 결합했을 때 더욱 향상된 성능을 나타냅니다. 또, SPRIG는 다양한 모델 패밀리(model families)와 언어(language)에 대해 일반화되는 효과도 검증되었습니다.



### A Complexity-Based Theory of Compositionality (https://arxiv.org/abs/2410.14817)
- **What's New**: 이 논문은 AI와 인지 과학에서 컴포지셔널 리프레젠테이션(compositional representation)의 수학적 정의인 '표현적 컴포지셔널리티(representational compositionality)'를 제안합니다.

- **Technical Details**: 표현적 컴포지셔널리티는 표현력이 뛰어난 조합적 리프레젠테이션을 규명하며, 이를 수치적으로 측정할 수 있는 기준으로 설정합니다. 이 정의는 알고리즘적 정보 이론(algorithmic information theory)에 기반하여, 기호적 부분의 간단한 함수로 재설명 가능해야 한다고 주장합니다.

- **Performance Highlights**: 실험을 통해 이 새로운 정의가 기존 AI 및 인지 과학 문헌에서의 다양한 직관들을 통합한다는 것을 입증하고, 이론적으로 해결하기 어려운 문제임에도 불구하고 표준 딥러닝 도구를 사용하여 쉽게 추정할 수 있음을 보여줍니다.



### Aligning AI Agents via Information-Directed Sampling (https://arxiv.org/abs/2410.14807)
- **What's New**: 이번 연구에서는 AI Alignment에 대한 새로운 접근 방식을 소개하고, 전통적인 multi-armed bandit 문제의 확장으로서 bandit alignment 문제를 정의합니다.

- **Technical Details**: 이 연구에서는 AI 에이전트가 환경과 인간과 상호작용하며 장기적으로 기대 보상을 극대화해야 하는 Bandit Alignment 문제를 다룹니다. 비용과 보상은 관찰된 결과와 인간의 선호에 따라 달라지며, 에이전트는 탐색(exploration)과 활용(exploitation) 사이의 균형을 적절히 조절해야 합니다.

- **Performance Highlights**: 전통적인 탐색 알고리즘 및 Thompson sampling과 같은 최신 알고리즘들이 이 문제에서 적절한 솔루션을 제공하지 못하는 반면, 정보 지향 샘플링(information-directed sampling)이 좋은 결과를 보여줍니다.



### DistRL: An Asynchronous Distributed Reinforcement Learning Framework for On-Device Control Agents (https://arxiv.org/abs/2410.14803)
Comments:
          Paper and Appendix, 24 pages

- **What's New**: 이 논문에서는 DistRL이라는 새로운 프레임워크를 도입하여 모바일 기기 제어 에이전트의 온라인 강화 학습 (RL) 미세 조정을 향상시키고자 합니다. 이 프레임워크는 중앙 집중형 훈련과 분산형 데이터 수집을 결합하여 동적인 온라인 상호작용에서 효율적인 미세 조정을 보장합니다.

- **Technical Details**: DistRL은 A-RIDE라는 새로운 오프-정책 강화 학습 알고리즘을 활용하여 분산 및 비동기 데이터 활용을 위한 환경을 제공합니다. 이를 통해 데이터 수집을 효과적으로 관리하고, 효율적으로 샘플링을 보장하여 상호작용 데이터의 중요 위치에서 학습할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, DistRL은 훈련 효율성을 평균 3배 향상시켰으며, 훈련 데이터 수집 속도는 기존 동기화 멀티 머신 방법보다 2.4배 빨라졌습니다. 또한, 안드로이드 일반 작업에 대한 성능에서 20%의 상대적 성공률 향상을 달성하여 기존 상태의 최첨단 방법들보다 크게 개선되었습니다.



### Deep Generic Dynamic Object Detection Based on Dynamic Grid Maps (https://arxiv.org/abs/2410.14799)
Comments:
          10 pages, 6 figures, IEEE IV24

- **What's New**: 이 논문에서는 자동 주행을 위한 일반적인 동적 객체 탐지 방법을 제안합니다. LiDAR 기반 동적 그리드를 실시간으로 생성하고, 이를 통해 다양한 동적 객체를 탐지하는 딥러닝 모델을 훈련합니다. 특히 Rotation-equivariant Detector(ReDet)를 활용한 점에서 기존의 통계 기반 방법보다 높은 탐지 성능을 보입니다.

- **Technical Details**: 본 연구는 회전 경량 박스(object detection task as rotated bounding box) 문제로 일반 동적 객체 탐지를 다룹니다. 기존의 카메라 이미지 기반 탐지 네트워크와 달리 회전 네트워크는 박스 각도를 예측할 수 있습니다. ReDet(Rotation-equivariant Detector)는 고성능 탐지를 위해 선택되었으며, 동적 자Occupancy grid maps를 입력으로 사용하여 동적 객체의 경량 박스를 출력합니다. 각 그리드 셀은 다중 가정의 기본 신념 질량(Dempster-Shafer basic belief masses) m을 포함하며, 이를 통해 정적 및 동적 상태를 구분합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존의 동적 셀 클러스터링 전략에 비해 거짓 긍정 탐지 비율(false positive object detection rate)을 크게 줄였습니다. 특히 움직이는 쇼핑 카트와 같은 훈련 데이터에 없던 동적 객체의 탐지에서도 인상적인 성과를 보였습니다.



### SSL-NBV: A Self-Supervised-Learning-Based Next-Best-View algorithm for Efficient 3D Plant Reconstruction by a Robo (https://arxiv.org/abs/2410.14790)
Comments:
          22 pages, 11 figures, 1 table

- **What's New**: 이번 논문에서는 3D 식물 재구성을 위한 새로운 접근 방식인 self-supervised learning 기반 Next-Best-View (SSL-NBV) 방법을 제안합니다. 이전의 방법들은 사전 훈련된 식물 모델을 요구하여 현실 세계의 식물에 적용하기 어려운 한계를 지니고 있었습니다.

- **Technical Details**: SSL-NBV는 심층 신경망(deep neural network)을 사용하여 후보 viewpoint에서의 정보 이득(information gain, IG)을 예측합니다. 이 방법은 로봇이 작업을 수행하는 동안 새로운 3D 센서 데이터를 이전에 수집한 데이터와 비교하여 자신의 훈련 데이터를 수집할 수 있도록 합니다. 또한, 약한 지도학습(weakly-supervised learning)과 경험 재현(experience replay)을 적용하여 효율적인 온라인 학습(online learning)을 가능하게 합니다.

- **Performance Highlights**: 종합적인 평가 결과 SSL-NBV는 비-NBV 방법에 비해 식물 재구성을 위한 시점(view) 수가 적고, voxel 기반 방법보다 800배 이상 빠른 성능을 보였습니다. SSL-NBV는 DL-NBV 기준에 비해 훈련 주석(training annotations)을 90% 이상 줄였으며, 온라인 미세 조정(online fine-tuning)을 통해 새로운 시나리오에 적응할 수 있는 능력을 보여주었습니다. 또한, 실제 식물을 사용하여 3D 식물 재구성을 위한 새로운 시점을 효과적으로 계획할 수 있음을 증명하였습니다. 가장 중요한 점은 SSL-NBV가 전체 네트워크 훈련을 자동화하며, 변화하는 농업 환경에서의 지속적인 온라인 학습을 가능하게 한다는 것입니다.



### Evaluating Quantized Large Language Models for Code Generation on Low-Resource Language Benchmarks (https://arxiv.org/abs/2410.14766)
- **What's New**: 이번 연구에서는 LLM(대규모 언어 모델)의 민주화(Democratization of AI) 및 접근성(Accessibility) 문제를 다룹니다. 특히 코드 생성에 이용되는 LLM의 양자화(Quantization) 가능성을 검토하고, 다양한 매개변수 설정에서 성능을 평가합니다.

- **Technical Details**: 연구에서는 7억 개 매개변수를 가진 다섯 가지 양자화된 코드 LLM을 Lua 프로그래밍 언어의 코드 생성 업무에서 테스트하였고, 2비트, 4비트, 8비트 정수 정밀도(Precision)에서 모델을 평가했습니다. 4비트 정밀도의 양자화된 모델이 성능과 모델 크기 간의 우수한 균형을 제공한다고 결론지었습니다.

- **Performance Highlights**: 4비트 양자화 모델은 7억 개 매개변수를 가진non-quantized 모델보다 더 나은 성능을 보여주었으나, 전체적으로 Lua 코드 생성 작업에서 성능이 50% 미만으로 낮았습니다. 2비트 정밀도에서는 성능이 크게 저하되었으며 8비트에서는 더욱 긴 추론 시간(Inference Time)이 소모됐습니다.



### What's New in My Data? Novelty Exploration via Contrastive Generation (https://arxiv.org/abs/2410.14765)
- **What's New**: 이 연구에서는 데이터 세트를 직접 검사할 수 없는 상황에서 Fine-tuning 데이터 세트의 새로운 특성을 식별하는 작업인 'novelty discovery through generation'을 소개합니다.

- **Technical Details**: 우리는 Contrastive Generative Exploration (CGE)라는 접근 방식을 제안합니다. CGE는 사전 훈련된 모델과 Fine-tuning된 모델의 예측을 대조하여 새로운 특성을 강조한 예를 생성합니다. CGE는 또한 반복적인 방식으로 업데이트되어 다양한 출력을 촉진합니다.

- **Performance Highlights**: CGE는 박스 언어, 유해한 언어 및 새로운 자연어와 프로그래밍 언어 감지에서 효과적인 성능을 입증했습니다. Differential privacy 기술을 사용하여 Fine-tuning된 모델에서도 일관된 성능을 보여주었습니다.



### Enabling Scalable Evaluation of Bias Patterns in Medical LLMs (https://arxiv.org/abs/2410.14763)
- **What's New**: 본 연구에서는 의료 분야에 특화된 대규모 언어 모델(Med LLM)의 편향 평가를 자동으로 생성하는 새로운 방법을 제시합니다. 이러한 방식은 기존의 수작업 데이터셋에 의존하지 않고, 엄격한 의학적 증거에 기반하여 테스트 케이스를 자동 생성함으로써 편향 평가를 확장하는 데 기여합니다.

- **Technical Details**: 제안된 방법은 의료 지식 그래프(medical knowledge graphs), 의료 온톨로지(medical ontologies), 사용자 맞춤형 LLM 평가 프레임워크를 통합하여 세 가지 주요 문제(도메인 전문성에 따른 편향 특성화, 생성 과정에서의 환각(hallucination), 건강 결과와 민감 속성 간의 다양한 종속성)를 해결합니다. 이를 통해 의료 분야에서의 공정성을 평가할 수 있는 새로운 데이터셋을 생성하였습니다.

- **Performance Highlights**: 제안된 방법은  기존 수작업 생성 방법에 비해 Med LLM의 편향 패턴을 보다 효과적으로 드러내며, 실험 결과를 통해 입력된 테스트 케이스가 신뢰할 수 있는 지식의 기반을 갖춘 강력한 비네트(vignette) 생성이 가능함을 입증하였습니다.



### Controllable Discovery of Intents: Incremental Deep Clustering Using Semi-Supervised Contrastive Learning (https://arxiv.org/abs/2410.14755)
Comments:
          Accepted in IJCNLP'23

- **What's New**: 이 논문에서는 Conversational AI 시스템에서 사용자 의도를 효과적으로 발견할 수 있도록 돕는 새로운 CDI (Controllable Discovery of Intents) 프레임워크를 제안합니다. 이 프레임워크는 사용자 지식과 도메인 정보를 포함하여 intent discovery 과정에서 인간 피드백을 통합할 수 있도록 설계되었습니다.

- **Technical Details**: CDI 프레임워크는 처음에 unlabeled data에 대해 unsupervised contrastive learning을 실행하고, 그 후 partially labeled data에서 fine-tuning을 통해 클러스터링과 표현을 반복적으로 개선합니다. 이 과정에서는 catastrophic forgetting을 방지하기 위해계속 학습(continual learning)에서 제안된 learning-without-forgetting 기법을 활용합니다.

- **Performance Highlights**: 실험 결과, CDI 프레임워크는 CLINC와 BANKING 데이터셋에서 이전 연구보다 각각 10.26%와 11.72% 향상된 성능을 보여줍니다.



### On the Sparsity of the Strong Lottery Ticket Hypothesis (https://arxiv.org/abs/2410.14754)
- **What's New**: 이번 연구에서는 Strong Lottery Ticket Hypothesis (SLTH)에 대한 최초의 고전적 설정에서의 증명을 제공하며, 서브네트워크(subnetwork)의 희소성(sparsity)에 대한 보장을 포함하고 있습니다.

- **Technical Details**: 우리는 Random Fixed-Size Subset Sum Problem (RFSS)이라는 문제에 대한 엄밀히 조여진 경계를 증명하였습니다. 이 문제는 고정된 크기의 서브셋(subset)만을 고려하는 RSS 문제(랜덤 서브셋 합 문제)의 변형입니다.

- **Performance Highlights**: 이 연구는 SLTH의 원래 동기와는 달리 서브네트워크의 크기에 대한 보장을 제공하며, 이를 통해 밀집(dense) 및 동등한 네트워크(equivariant networks)와 같은 다양한 설정에서 SLTH를 증명하는 데 기여합니다.



### Collaboratively adding new knowledge to an LLM (https://arxiv.org/abs/2410.14753)
- **What's New**: 이 논문은 LLM(대형 언어 모델)에 새로운 지식을 추가하면서 기존 지식을 보존하는 방법에 대한 연구를 다루고 있습니다. Semi-cooperative 및 Fully-cooperative 두 가지 설정에서의 성능을 분석하고, LoRA(저순위 적응)가 전통적인 전체 정밀 조정(full fine-tuning)보다 효과적임을 보여줍니다.

- **Technical Details**: 논문에서는 전통적인 전체 정밀 조정(FFT)과 LoRA를 통한 파라미터 효율적 조정(PEFT)을 고려합니다. Semi-cooperative 설정에서는 데이터셋이 훈련 후에 사용 불가능하지만 MOE 혼합, 모델 병합, LoRA 기반의 직교 서브스페이스 순차 학습이 잘 작동합니다. Fully-cooperative 설정에서는 데이터셋이 사용 가능하며, 병합 훈련과 재생(replay)을 통한 순차 훈련이 효과적입니다.

- **Performance Highlights**: LoRA는 지식 습득 측면에서 FFT보다 적은 성능 저하를 보이지만, 이전 지식을 보존하는 데에는 더 유리한 성능을 나타냅니다. 논문은 실험 결과를 바탕으로 LoRA가 전체 매개변수 조정보다 더 나은 선택이 될 수 있음을 강조합니다.



### ETF: An Entity Tracing Framework for Hallucination Detection in Code Summaries (https://arxiv.org/abs/2410.14748)
Comments:
          11 pages, 6 Figures, 5 Tables

- **What's New**: 본 연구에서는 코드 요약에서의 환각(hallucination) 탐지를 위해 최초로 10,000개의 샘플을 포함하는 데이터세트를 소개하며, 새로운 Entity Tracing Framework (ETF)를 제안합니다. 이 프레임워크는 코드에서 엔티티를 식별하고, 생성된 요약 내에서 이 엔티티들의 의도를 검증하는 새로운 접근 방식을 제공합니다.

- **Technical Details**: Entity Tracing Framework (ETF)는 a) 정적 프로그램 분석(static program analysis)을 이용하여 코드 엔티티를 식별하고 b) LLM을 사용하여 이 엔티티와 생성된 코드 요약 내의 의도를 매핑 및 검증합니다. 이 연구는 코드의 엔티티를 기반으로 한 요약 검증의 중요성을 강조하며, 0.73 F1 점수로 효과성을 입증합니다.

- **Performance Highlights**: 본 연구의 프레임워크는 코드 요약에서 환각 탐지를 위한 해석 가능한 방법을 제공하며, 모델이 생성한 문서의 정확성을 평가할 수 있게 합니다. 우리는 411개의 요약과 10,000개 엔티티 샘플을 포함하는 데이터세트를 오픈 소스할 계획입니다.



### Accounting for Sycophancy in Language Model Uncertainty Estimation (https://arxiv.org/abs/2410.14746)
- **What's New**: 이 논문은 sycophancy(아첨) 편향과 불확실성 추정 간의 관계를 처음으로 연구하고, 이를 통해 불확실성 추정 프로세스에서 sycophancy를 고려하는 새로운 알고리즘 SyRoUP을 제시합니다.

- **Technical Details**: 논문은 모델의 답변에 대한 불확실성을 추정하기 위해 제안된 답변을 기반으로 불확실성을 평가하는 방식을 사용합니다. 사용자 행위의 다양성과 그에 따른 sycophancy의 영향을 살펴보며, Brier Score와 Brier Skill Score를 통해 추정 정확성을 평가합니다.

- **Performance Highlights**: 실험 결과, 사용자 신뢰도는 sycophancy의 영향을 조절하는 중요한 역할을 하고, SyRoUP은 이러한 영향을 더 잘 예측할 수 있음을 보여줍니다. 이로 인해 모델과 사용자가 모두 불확실성을 외부화할 때 sycophancy 편향의 영향을 완화할 수 있다는 점을 주장합니다.



### SemiEvol: Semi-supervised Fine-tuning for LLM Adaptation (https://arxiv.org/abs/2410.14745)
- **What's New**: 본 논문은 대규모 언어 모델(LLMs)의 효과적인 조정을 위해 레이블이 있는 데이터와 레이블이 없는 데이터를 모두 활용하는 반지도 학습(framework)을 제안합니다. 새로운 프레임워크인 SemiEvol은 지식 전파 및 선택 전략을 통해 모델의 성능을 개선하는 데 중점을 둡니다.

- **Technical Details**: SemiEvol의 주요 구성 요소는 다음과 같습니다: 1) Knowledge Propagation - 레이블이 있는 데이터를 사용하여 모델 성능을 향상시키는 두 단계 접근 방식을 적용합니다. 2) Knowledge Selection - 협업 학습 메커니즘을 통해 레이블이 없는 데이터에서 더 높은 품질의 pseudo-response 샘플을 선택합니다. 이는 k-최근접 이웃 검색을 사용하여 예측을 보조합니다.

- **Performance Highlights**: SemiEvol은 MMLU, MMLU-Pro 및 ConvFinQA와 같은 7개의 일반 및 도메인 특정 데이터셋에서 실험을 진행하였으며, 모델 성능의 유의미한 개선을 나타냈습니다. 또한, SemiEvol은 기존의 SFT 및 self-evolution 방법과 비교하여 하이브리드 데이터 시나리오에서 실용성을 강조하였습니다.



### Eliciting Uncertainty in Chain-of-Thought to Mitigate Bias against Forecasting Harmful User Behaviors (https://arxiv.org/abs/2410.14744)
- **What's New**: 이 논문에서는 대화 예측(conversation forecasting) 작업을 수행하는 데 필요한 모델의 불확실성(uncertainty)을 다루고 있습니다. 특히, 다양한 언어 모델(large language models, LLMs)이 대화 예측의 편향(bias)을 어떻게 줄일 수 있는지를 탐구하고 있습니다.

- **Technical Details**: 연구는 5개의 오픈 소스 언어 모델을 사용해 2개의 데이터셋을 대상으로 진행되었습니다. 이 데이터셋은 소셜 미디어의 유해한 행동(예: 개인 공격)을 예측하기 위해 설계된 것입니다. 주요 질문은 LLM의 예측 정확도가 불확실성을 표현하도록 요청했을 때 어떻게 변화하는지와, 이러한 불확실성을 기반으로 어떤 형태의 편향이 완화될 수 있는지를 포함합니다.

- **Performance Highlights**: 논문은 LLM의 예측 능력 및 편향을 검토하며, 불확실성을 고려하는 것이 모델의 예측 결과에 긍정적인 영향을 미칠 것이라는 가설을 제시합니다. 특히, 시스템이 유해한 결과를 예측할 때 기존의 편향을 줄이는 방안을 모색하고 있습니다.



### Efficient Deep Learning Board: Training Feedback Is Not All You Need (https://arxiv.org/abs/2410.14743)
- **What's New**: 이 논문에서는 교육 피드백 없이도 성능 예측 및 구성 요소 추천이 가능한 혁신적인 딥러닝 보드인 EfficientDL을 제안합니다. EfficientDL은 27개의 시스템 구성 요소를 빠르고 정확하게 추천할 수 있으며, 이전 AutoML 도구들보다 뛰어난 성능을 보입니다.

- **Technical Details**: EfficientDL은 정적 성능 예측 모델과 최적화된 구성 요소 추천 알고리즘(αβ-BO search)을 기반으로 하여, 실제 모델 실행에 대한 의존성을 없애고 성능 예측을 구현합니다. 제안된 시스템 구성 요소 데이터셋은 모델 아키텍처, 데이터 변환 및 하드웨어 세부 사항을 포함하여 27개의 조정 가능한 구성 요소를 포함합니다.

- **Performance Highlights**: CIFAR-10 데이터셋에서 EfficientDL은 기존의 AutoML 도구들보다 약 20배 빠르며, Top-1 정확도가 1.31% 향상되어 91.31%에 도달했습니다. EfficientDL은 ResNet50, MobileNetV3, EfficientNet-B0, MaxViT-T, Swin-B, DaViT-T와 같은 주요 모델 아키텍처와의 호환성을 지니고 있습니다.



### Agent Skill Acquisition for Large Language Models via CycleQD (https://arxiv.org/abs/2410.14735)
- **What's New**: CycleQD라는 새로운 프레임워크를 소개하며, 알고리즘의 순환 적응을 통해 Quality Diversity (QD) 방식을 활용하여 여러 과제의 성과를 독립적으로 최적화합니다. 이 방식은 모델 병합 기반의 크로스오버와 SVD 기반의 변이를 사용하여 각 기술의 효율성을 높이고 데이터 비율 관리의 복잡성을 줄입니다.

- **Technical Details**: CycleQD는 연속적인 에이전트 기술의 미세 조정을 위해 설계되었습니다. 각 기술의 성과 메트릭을 개별적으로 최적화하고, 모델 병합 기반의 크로스오버를 통해 전문 기술을 통합하며, SVD 기반의 변이를 통해 모델의 과적합을 방지합니다. 이를 통해 일반적인 손실 함수의 한계를 극복하고 작업 특정 성과 메트릭을 직접 최적화합니다.

- **Performance Highlights**: CycleQD를 사용하여 LLAMA3-8B-INSTRUCT 모델이 기존의 미세 조정 방법보다 뛰어난 성능을 보여주며, 코드, 운영 체제 및 데이터베이스 작업에서 GPT-3.5-TURBO와 동일한 수준의 성능을 달성했습니다. 이는 데이터 비율 조정이나 일반 언어 능력 감소 없이 이루어졌습니다.



### Knowledge Graph Embeddings: A Comprehensive Survey on Capturing Relation Properties (https://arxiv.org/abs/2410.14733)
Comments:
          22 pages, 8 figures, 3 tables, this paper is a modified English version of our article already published in Computer Science journal (in Chinese), released to facilitate communication among international researchers in the relevant fields

- **What's New**: 본 논문은 KGE (Knowledge Graph Embedding) 기법에서 관계의 복잡한 매핑 속성을 다루며, 다양한 관계 패턴을 캡처하기 위한 모델들을 종합적으로 검토합니다. 또한, Sparse 및 Dynamic KGs에 대해 향후 연구 방향을 제시합니다.

- **Technical Details**: 관계의 매핑 속성으로는 one-to-one, one-to-many, many-to-one, many-to-many가 있으며, 이러한 관계를 모델링하기 위한 다양한 방법이 논의됩니다. 여기에는 수정된 텐서 분해(modified tensor decomposition), 수정된 관계 인식 매핑(modified relation-aware mappings), 회전 연산(rotation operations)을 이용한 모델들이 포함됩니다. 또한, 보조 정보(auxiliary information)를 포함하는 모델, 쌍곡선 공간(hyperbolic spaces)을 기반으로 한 모델, 극좌표계(polar coordinate system)를 이용한 모델도 검토됩니다.

- **Performance Highlights**: 각 모델의 성능은 지식 그래프의 다양한 구조를 효과적으로 포착하는 능력에 따라 결정되며, 향후 연구는 멀티모달 정보(multi-modal information)의 통합, 규칙(rule) 기반의 관계 패턴 모델링, 동적 KGE 환경에서의 관계 특성을 캡처하는 모델 개발을 포함합니다.



### MatryoshkaKV: Adaptive KV Compression via Trainable Orthogonal Projection (https://arxiv.org/abs/2410.14731)
- **What's New**: 본 논문에서는 기존 연구들이 주로 KV 캐시의 첫 세 축에 초점을 맞추었던 것과 달리, 특징 차원 축(feature dimension axis)에서의 압축 방법을 새롭게 제안하고 있습니다. 저자들은 저랭크 프로젝션 행렬(low-rank projection matrices)을 활용하여 KV 캐시의 효율성을 높이고 성능 저하를 최소화하는 방법을 모색하였습니다.

- **Technical Details**: 연구진은 PCA(주성분 분석)를 통해 압축을 시작하였고, 오르소고날 프로젝션 매트릭스(orthogonal projection matrices)를 조정하여 모델의 출력을 보존합니다. 이 과정에서 매트리요시카 훈련 전략(Matryoshka training strategy)을 도입해 다양한 계층과 헤드에 대해 최적의 압축 비율(compression rates)을 탐색합니다. 이는 KV 캐시의 효율성을 극대화하면서도 이름 있는 LLM들, 예를 들어 LLaMA2-7B-base 모델에 통합할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, MatryoshkaKV 방법은 LLaMA2-7B-base와 Mistral-7B-v0.3-base와 같은 LLM에서 90% 이상의 성능을 유지하면서 평균 60%의 KV 캐시 압축률을 달성했음을 보여주었습니다. 특정 극단적인 상황에서는 압축률이 75%에 달하기도 하며, 수치적으로 이는 이전 연구들보다 뛰어난 데이터 효율성을 입증합니다.



### On the Relation Between Linear Diffusion and Power Iteration (https://arxiv.org/abs/2410.14730)
- **What's New**: 최근 확산(시너리) 모델의 성공적인 이미지 생성 능력을 분석하여, 이러한 모델이 무작위 노이즈에서 기저 분포로 변환되는 과정을 '상관 머신(correlation machine)'으로 설명합니다. 이 연구는 PCA(주성분 분석)를 통해 최적의 디노이저(denoiser)와 관련된 이론적 연결을 제시합니다.

- **Technical Details**: 모델은 PCA 프로젝션을 최적 디노이저로 활용하며, 노이즈 레벨과 학습 데이터 양에 따라 디노이저의 의존도를 분석합니다. 낮은 주파수는 생성 과정에서 더 일찍 나타나며, 이는 고유 벡터(eigenvector)와의 상관관계(상관도)가 노이즈 레벨이 증가함에 따라 감소하는 방식으로 확인됩니다.

- **Performance Highlights**: 연구는 비선형 딥 디노이저의 Jacobians에서도 일반적으로 적용 가능함을 보여주며, 학습 과정의 여러 단계에서 디노이저의 변화와 유용성을 강조합니다. 이는 선형 모델과 비선형 깊은 학습 모델 모두에 대한 통찰력을 제공합니다.



### Tokens on Demand: Token Condensation as Training-free Test-time Adaptation (https://arxiv.org/abs/2410.14729)
Comments:
          18 pages, 7 figures

- **What's New**: 이 논문에서는 Token Condensation as Adaptation (TCA)를 소개합니다. TCA는 비전-언어 모델 (VLMs)이 테스트 시간 추론 중 겪는 분포 변화(distribution shifts)를 완화하기 위해 설계된 비훈련(training-free) 방식입니다.

- **Technical Details**: TCA는 <cls> 토큰에 대한 낮은 주의(attentiveness)를 보이는 이미지 토큰을 응축(condensing)하여 패치 레벨에서 분포 간극(distribution gaps)을 연결합니다. TCA는 역사적 데이터 스트림에서 특정 대상 클래스(target classes)와 정렬된 가장 신뢰할 수 있는 <cls> 토큰을 식별하고 추적합니다. 이를 위해, TCA는 불확실성이 가장 낮은 토큰을 '앵커(anchors)'로 보존하기 위한 컨텍스트 토큰 저수지(context token reservoir, CTR)를 제안합니다. CTR에서 샘플링한 앵커를 사용하여 TCA는 클래스와 무관한 토큰을 깎고(pruning) 나머지 클래스 애매한 토큰을 대표 센터로 병합(merging)합니다.

- **Performance Highlights**: TCA는 테스트 시간 적응(test-time adaptation)에서 토큰 효율성을 탐구하는 최초의 방법으로, 크로스 데이터셋(cross-dataset) 및 분포 외(out-of-distribution) 적응 작업에서 일관되게 우수한 성능을 보입니다. GFLOPs를 12.2%에서 48.9%까지 줄이면서 가장 강력한 기준선(baseline) 대비 정확도(accuracy)를 최대 21.4% 향상시킵니다.



### A Phenomenological AI Foundation Model for Physical Signals (https://arxiv.org/abs/2410.14724)
- **What's New**: 이 연구에서는 다양한 물리적 현상과 응용 분야에 걸쳐 일반화할 수 있는 AI 기반 모델, 즉 AI foundation model을 개발했습니다. 이 모델은 5억 9천만 개의 크로스 모달 센서 측정 데이터를 사용하여 훈련되었으며, 기존의 물리 법칙이나 편견을 모형에 도입하지 않았습니다.

- **Technical Details**: 모델은 전기 신호, 유체 흐름, 광학 센서 등을 포함한 다양한 물리 신호를 처리할 수 있는 프레임워크에 기초합니다. 이 연구는 기계적 운동과 열역학 등 훈련에서 보지 못한 현상을 예측하는 능력을 검증하였으며, 단일 모델이 복잡한 물리적 과정에 대해서도 유연하게 확장 가능한지를 평가했습니다.

- **Performance Highlights**: 모델은 제로샷 추론(zero-shot inference) 능력을 보여주었으며, 훈련 데이터와 다른 물리적 시스템에 대해 전문적으로 훈련된 모델보다 우수하거나 동등한 성능을 발휘했습니다. 이러한 결과는 일반적인 AI foundation model이 실제 물리적 프로세스를 표현하는 데 상당한 잠재력을 지닌다는 것을 강조했습니다.



### A Systematic Survey on Large Language Models for Algorithm Design (https://arxiv.org/abs/2410.14716)
- **What's New**: 이번 논문은 LLM4AD(Algorithm Design을 위한 Large Language Models)의 발전을 다루며, 기존의 연구를 체계적으로 검토하여 통합적인 시각을 제공합니다. 3년간의 연구 결과를 정리하고, 알고리즘 설계 과정에서의 LLMs의 역할과 사용된 전략들에 대해 재조명합니다.

- **Technical Details**: LLM4AD는 네 가지 주요 차원에서 연구를 분류합니다: 1) LLM의 역할; 2) 검색 방법; 3) 프롬프트 방법; 4) 응용 분야. 이러한 다차원 분류법을 통해 각 연구의 기여와 발전을 명확히 하고 향후 연구의 기회와 갭을 식별합니다.

- **Performance Highlights**: 논문은 180편 이상의 연구를 종합하여 LLM4AD의 Current 상태를 파악하고, 주요 도전 과제와 가능성을 논의합니다. 또한, LLM 기술의 발전으로 인해 알고리즘 설계 과정에서의 창의성과 효율성이 향상될 수 있음을 기대합니다.



### Animating the Past: Reconstruct Trilobite via Video Generation (https://arxiv.org/abs/2410.14715)
- **What's New**: 이 연구에서는 트릴로바이트 행동을 정적 화석에서 재구성하는 새로운 자동화된 텍스트-비디오(T2V) 프롬프트 학습 방법을 제안합니다. 이 방법은 시각적 사실성과 부드러움을 평가할 수 있는 보상을 사용하여 동영상 생성 모델을 조정합니다.

- **Technical Details**: 본 연구에서는 9,088개의 Eoredlichia intermedia 화석 이미지로 훈련된 대형 언어 모델(LLM)을 활용하여 텍스트에서 비디오로의 변환을 위한 프롬프트를 생성합니다. 이 모델은 디퓨전 모델을 사용하여 트릴로바이트 애니메이션 세그먼트를 생성하고, 생성된 비디오는 사실성과 연속성을 평가하여 LLM을 업데이트 하는 방식으로 작동합니다.

- **Performance Highlights**: 실험 결과, 새로 제안된 방법은 기존의 강력한 기준을 초과하는 고휘도와 사실성을 가진 트릴로바이트 비디오를 생성하며, 이는 과학적 이해를 증진시키고 대중의 참여를 촉진할 것으로 기대됩니다.



### Abstracting Situation Calculus Action Theories (https://arxiv.org/abs/2410.14712)
Comments:
          60 pages, 1 figure

- **What's New**: 이 논문은 상황 계산법(situation calculus)과 ConGolog 에이전트 프로그래밍 언어를 기반으로 하는 에이전트 추상을 위한 일반적인 프레임워크를 개발합니다. 높은 수준의 명세와 낮은 수준의 명세를 기본 행동 이론으로 표현하며, 고수준 행동과 저수준 ConGolog 프로그램 간의 정제 매핑(refinement mapping)을 정의합니다.

- **Technical Details**: 정음 추상화(sound abstraction)와 완전 추상화(complete abstraction) 개념을 정의하며, 이러한 추상을 통해 저수준 모델과 고수준 모델 간의 관계를 설명합니다. 적절한 bisimulation 관계의 존재를 통해 두 모델 간의 관계를 성립시키고, 고수준 이론이 저수준 이론을 효과적으로 설명할 수 있는 조건을 제시합니다. 또한 기본 행동 이론 제약을 통해 저수준 행동 시퀀스에 대한 고수준 행동 시퀀스의 유일성을 보장합니다.

- **Performance Highlights**: 본 연구에 따르면, 고수준 이론이 특정 조건 달성을 위한 실행 가능성을 내포하는 경우 저수준 이론도 그것을 만족해야 하며, 이는 에이전트 행동에 대한 계획, 모니터링, 높은 수준의 설명 생성 등을 가능하게 합니다. 이러한 방법론은 Explainable AI 분야에 응용될 수 있습니다.



### G2D2: Gradient-guided Discrete Diffusion for image inverse problem solving (https://arxiv.org/abs/2410.14710)
- **What's New**: 이 논문은 이산 확산 모델(Discrete Diffusion Model)을 기반으로 한 생성 모델을 사용하여 선형 역 문제를 해결하는 새로운 방법, Gradient-guided Discrete Diffusion (G2D2)를 제안합니다. 이는 최초로 이산 확산 모델을 역 문제 해결을 위한 우선 값으로 사용하는 것입니다.

- **Technical Details**: G2D2는 이미지 역 문제를 해결하기 위해 이산 확산 모델을 프라이어(prior)로 사용하며, 변분 분포(variational distribution)를 최적화하는 연속 완화(continuous relaxation) 기법을 활용하여, 이전 방법의 한계를 극복합니다. 또한, 별 모양의 노이즈 프로세스를 이용하여 기존 이산 확산 모델의 제한을 완화합니다.

- **Performance Highlights**: 실험 결과 G2D2는 표준 벤치마크 데이터셋을 이용한 성과 평가에서 기존 방법들과 비교하여 다양한 역 문제를 해결하는 데 가능성을 보였습니다. 특히, 이 방법은 훈련이 필요없는 이산 프라이어 기반의 동적 데이터 생성 모델의 응용 가능성을 보여주었습니다.



### FACMIC: Federated Adaptative CLIP Model for Medical Image Classification (https://arxiv.org/abs/2410.14707)
Comments:
          Accepted in MICCAI 2024

- **What's New**: 본 논문에서는 분산된 데이터에서 모델 훈련을 통한 데이터 프라이버시(privacy)를 보장하면서 의료 이미지 분석을 위한 새로운 연합학습(federated learning) 접근 방식을 제안합니다. 특히, CLIP 모델을 분산 회귀 적응(federated adaptive)으로 변형하여 효율적인 분류(classification) 작업을 수행합니다.

- **Technical Details**: 제안된 Federated Adaptive Contrastive Language-Image Pretraining (FACMIC) 모델은 클라이언트의 데이터에 적합한 특징을 선택하기 위한 경량화된 특징 주의(feature attention) 모듈을 사용합니다. 또한, 클라이언트 간의 데이터 분포 차이를 줄이기 위한 도메인 적응(domain adaptation) 기법을 도입하여 모델의 효과성과 효율성을 향상시킵니다.

- **Performance Highlights**: 네 가지 공개 데이터셋에서의 실험 결과, FACMIC 모델은 실제 및 다중 출처의 의료 이미징 데이터 처리에 있어 기존의 최신 방법들보다 우수한 성능을 보였습니다. 특히, 뇌종양 및 피부암 분류 작업에서 높은 분류 성능을 달성하였습니다.



### Self-Supervised Keypoint Detection with Distilled Depth Keypoint Representation (https://arxiv.org/abs/2410.14700)
- **What's New**: 이 논문에서는 깊이 정보와 RGB 영상을 활용하여 키포인트(keypoint) 검출 성능을 향상시키기 위한 새로운 교차 모달 지식 증류 프레임워크인 Distill-DKP를 제안합니다. 이를 통해 기존의 비지도 학습 방법의 한계를 극복하고, 배경의 영향을 최소화하며 의미 있는 키포인트를 정확하게 검출할 수 있습니다.

- **Technical Details**: Distill-DKP는 두 가지 모델, 즉 깊이 기반의 teacher 모델과 이미지 기반의 student 모델로 구성됩니다. 훈련 과정에서 학생 모델은 깊이 기반 teacher로부터 주입된 embedding-level 지식을 통해 학습합니다. 키포인트 검출을 위해 ResNet 아키텍처와 업샘플링 기법을 사용하며, 핵심 모듈로는 Keypoint Detector, Edge Map Generator, Decoder가 있습니다. Cosine similarity loss를 최소화하여 학생 모델이 teacher 모델의 깊이 정보를 학습하도록 유도합니다.

- **Performance Highlights**: Distill-DKP는 Human3.6M 데이터셋에서 평균 L2 오차를 47.15% 줄였고, Taichi에서는 평균 평균 오차를 5.67% 감소시켰으며, DeepFashion 데이터셋에서는 키포인트 정확도가 1.3% 향상되었습니다. 이 결과들은 Distill-DKP가 기존의 비지도 방법보다 뛰어난 성능을 나타내고 있음을 보여줍니다.



### Learning Cortico-Muscular Dependence through Orthonormal Decomposition of Density Ratios (https://arxiv.org/abs/2410.14697)
- **What's New**: 이 연구는 cortico-muscular connectivity(피질-근육 연결성)의 새로운 모델링 접근법을 제시합니다. 기존의 EEG와 EMG 신호를 조사하는 방법론의 한계를 극복하기 위해 통계적 의존성 추정기를 사용한 독창적인 방법을 도입하였습니다.

- **Technical Details**: 본 연구에서는 density ratio(밀도 비율)의 직교 정규 분해 방법을 기반으로 한 통계적 의존성 추정기를 제안합니다. 이는 기존의 스칼라 값 방법을 확장하여 고유값(eigenvalues), 고유함수(eigenfunctions), 밀도 비율의 투영 공간을 학습하게 됩니다. 이를 통해 해석 가능성(interpretability)과 확장성(scalability), 그리고 국소적 시간 의존성(local temporal dependence) 문제를 해결합니다.

- **Performance Highlights**: 실험 결과, 학습된 고유함수는 피질-근육 연결성이 높은 움직임과 피험자 식별을 효과적으로 분류할 수 있음을 보여줍니다. 또한 특정 EEG 채널이 움직임 중에 활발하게 활성화되는 것을 확인하여 신경 과학적 결과와 일치하는 증거를 제시합니다.



### REBIND: Enhancing ground-state molecular conformation via force-based graph rewiring (https://arxiv.org/abs/2410.14696)
Comments:
          17 pages, 4 figures, 5 tables

- **What's New**: 본 논문은 3D 분자 구조 예측을 위한 새로운 프레임워크 REBIND를 제안합니다. REBIND는 비결합 원자 쌍 사이의 상호작용을 적절히 캡처하기 위해 Lennard-Jones 잠재력을 기반으로 한 엣지를 추가하여 분자 그래프를 재구성합니다.

- **Technical Details**: REBIND는 그래프 신경망(GNN) 아키텍처를 기반으로 하며, 초기 구조 예측을 생성하는 인코더와 재구성된 그래프에서 최종 구조를 예측하는 디코더로 구성됩니다. Lennard-Jones 잠재력을 통해 비결합 원자 쌍에서 작용하는 힘을 계산하고, 가장 큰 힘이 작용하는 비결합 쌍 사이에 엣지를 추가합니다.

- **Performance Highlights**: 실험 결과, REBIND는 QM9 데이터셋에서 최대 20%의 예측 오류 감소를 달성하며, 다양한 분자 크기에서 기존 최첨단 방법들을 상회하는 성능을 보여줍니다.



### Green vehicle routing problem that jointly optimizes delivery speed and routing based on the characteristics of electric vehicles (https://arxiv.org/abs/2410.14691)
- **What's New**: 본 논문은 실제 전기차를 이용하여 에너지 소비 모델을 수립하고, 차량의 물리적 특성을 충분히 고려하여 GVRP(Green vehicle routing problem) 문제를 다룹니다.

- **Technical Details**: 에너지 소비 모델은 차량의 시작/정지, 속도, 거리 및 적재량이 에너지 소비에 미치는 영향을 통합하여 작성되었습니다. 또한, 적재 우선 속도 최적화 알고리즘(load first speed optimization algorithm)을 제안하여 각 배달 지점 간 가장 적합한 속도를 선택합니다.

- **Performance Highlights**: 검증 결과, 속도 최적화 알고리즘을 사용한 실험에서 에너지 효율성이 일반적으로 향상되었으며, 일정 속도로 배달할 경우 에너지 소비가 평균 17.16% 더 높았습니다. 이 방법은 물류 회사들이 더욱 현실적으로 사용할 수 있는 방안을 제공합니다.



### Rethinking VLMs and LLMs for Image Classification (https://arxiv.org/abs/2410.14690)
- **What's New**: 이 논문에서는 기본적인 이미지 분류 작업에서 Visual Language Models(VLMs)와 Large Language Models(LLMs)를 효과적으로 결합하는 방법에 대해 재평가했습니다. LLM 없이도 VLM이 객체 및 장면 인식에서 더 뛰어난 성능을 보일 수 있음을 발견했습니다.

- **Technical Details**: VLM과 VLM+LLMs의 성능을 비교하기 위해, 본 연구에서는 동일한 비전 인코더를 사용하는 모델들을 비교했습니다. LLM은 이미지의 텍스트 설명 또는 임베딩을 통해 시각 정보와 연결되었습니다. 추가적으로, 라우팅을 위한 경량화된 LLM을 훈련시켜 시각 작업을 가장 적합한 모델로 분배합니다.

- **Performance Highlights**: 이 경량화 모델은 250만 개 이상의 시각 작업과 모델 정확도 쌍으로 훈련되어, HuggingGPT를 초과하고 GPT-4V와 유사한 정확도를 보여주며 비용 효율성 또한 향상되었습니다.



### Achieving Generalization in Orchestrating GNSS Interference Monitoring Stations Through Pseudo-Labeling (https://arxiv.org/abs/2410.14686)
Comments:
          DGON Positioning and Navigation for Intelligent Transport Systems (POSNAV)

- **What's New**: 이번 논문에서는 GNSS(지구 기가 스위밍 네비게이션 시스템) 수신기의 정확성을 저하시켰던 전파 방해 장치(jamming devices) 탐지를 위해 머신러닝을 활용한 새로운 방법을 제안합니다. 특히, 고속도로에 배치된 모니터링 스테이션을 통해 데이터 수집 및 전송을 제공하는 혁신적인 반자율 학습 방법이 도입되었습니다.

- **Technical Details**: 제안된 방법은 반지도 학습(semi-supervised learning) 방식을 사용하여 실내 환경에서 수집된 대량의 레이블(label)이 있는 데이터를 바탕으로 새로운 환경에서의 적응을 이룹니다. 모드 모델은 적응 시 몬테카를로(Monte Carlo) 기법과 딥 앙상블(Deep Ensemble)을 통합하여 적은 수의 레이블 데이터(최대 5%)로도 효과적으로 전파 방해를 분류할 수 있도록 설계되었습니다.

- **Performance Highlights**: 본 연구의 결과, 제안된 방법은 최첨단 기술에 비해 더 우수한 성능을 보였으며, 허가된 레이블 데이터 비율이 5% 이하로도 방해 신호 분류에 효과적임을 보여주었습니다. 이는 실제 야외 환경에서도 강력한 적응성을 발휘하여 모델의 일반화 성능이 크게 향상됨을 나타냅니다.



### Leveraging Event Streams with Deep Reinforcement Learning for End-to-End UAV Tracking (https://arxiv.org/abs/2410.14685)
- **What's New**: 본 논문에서는 이벤트 카메라를 활용한 무인 항공기(UAV)의 자율성을 높이기 위한 액티브 트래킹(active tracking) 접근 방식을 제안합니다.

- **Technical Details**: 제안한 트래킹 컨트롤러는 장착된 이벤트 센서로부터의 시각적 피드백에 반응하여 드론의 움직임을 조정하여 목표를 따라갑니다. 우리는 쿼드rotor의 전체 모션 기능과 이벤트 센서의 독특한 특성을 활용하기 위해, 이벤트 스트림의 원시 센서 데이터를 UAV의 제어 동작으로 직접 매핑하는 엔드 투 엔드(deep-reinforcement learning, DRL) 프레임워크를 제안합니다.

- **Performance Highlights**: 우리는 빠르게 움직이는 목표와 변화하는 조명 조건을 포함한 도전적인 시나리오에서의 실험을 통해 우리의 접근 방식의 효과를 입증하였으며, 이는 일반화(generalization) 능력을 향상시킵니다.



### RepoGraph: Enhancing AI Software Engineering with Repository-level Code Graph (https://arxiv.org/abs/2410.14684)
Comments:
          Work in progress

- **What's New**: 본 연구에서 제안하는 RepoGraph는 LLM(대형 언어 모델)에 기반한 AI 프로그래머가 코드 저장소를 전체적으로 활용할 수 있도록 설계된 플러그인 모듈입니다.

- **Technical Details**: RepoGraph는 저장소 내 코드 구조를 이해하기 위해 그래프 구조를 사용하며, 각 노드는 코드의 한 줄을 나타내고, 간선은 코드 정의와 참조 간의 의존성을 나타냅니다. 이를 통해 LLM이 보다 세부적인 맥락을 파악할 수 있도록 돕습니다.

- **Performance Highlights**: RepoGraph를 기존의 네 가지 소프트웨어 공학 프레임워크와 통합하여 성능을 평가한 결과, 에이전트 및 절차적 프레임워크 모두에서 평균 32.8% 향상된 성공률을 보였습니다.



### Brain-Aware Readout Layers in GNNs: Advancing Alzheimer's early Detection and Neuroimaging (https://arxiv.org/abs/2410.14683)
- **What's New**: 이 연구에서는 알츠하이머병(AD) 조기 진단을 위한 새로운 뇌 인지 기반 읽기 레이어(BA readout layer)를 제안하며, 이를 통해 Graph Neural Networks (GNNs)의 해석 가능성 및 예측 정확도를 향상시킵니다.

- **Technical Details**: BA readout layer는 기능적 연결성과 노드 삽입을 기반으로 뇌 영역을 클러스터링하여 복잡한 뇌 네트워크 특성을 포착할 수 있도록 GNN의 성능을 높입니다. 연구는 T1-weighted MRI, resting-state fMRI, FBB-PET 데이터를 이용하여 383명의 참가자에서 분석되었습니다. 이 데이터는 인지적으로 정상인과 전임상 AD 환자를 포함합니다.

- **Performance Highlights**: BA readout layer를 포함한 GNN은 Preclinical Alzheimer’s Cognitive Composite (PACC) 점수를 예측하는 데 있어 기존 모델들보다 월등히 높은 성능을 보여주었으며, 복원력과 안정성 또한 향상되었습니다. 더불어, 이 레이어는 인지 기능에 영향을 미치는 작업별 뇌 영역을 강조하여 해석 가능성을 높였습니다.



### ET-Plan-Bench: Embodied Task-level Planning Benchmark Towards Spatial-Temporal Cognition with Foundation Models (https://arxiv.org/abs/2410.14682)
- **What's New**: 최근 인공지능 분야의 발전으로 인해 Embedded (임베디드) 작업 계획에 LLMs (Large Language Models)을 적용하려는 많은 시도가 있었습니다. 이 연구에서는 새로운 벤치마크인 ET-Plan-Bench를 소개하여, LLMs의 임베디드 작업 계획을 평가하는 목적을 가지고 있습니다.

- **Technical Details**: ET-Plan-Bench는 다양한 난이도와 복잡성을 갖춘 임베디드 작업을 포함하여 LLMs의 공간적 (spatial), 시간적 및 인과적 (causal) 이해를 평가하도록 설계되었습니다. 멀티 소스 시뮬레이터를 백엔드로 사용하여 LLMs가 환경과 동적으로 상호작용할 수 있는 즉각적인 피드백을 제공합니다.

- **Performance Highlights**: 현재의 최신 공개 및 비공식 모델(GPT-4, LLAMA, Mistral 등)을 평가한 결과, 간단한 내비게이션 작업에서는 적절한 성능을 보였으나, 더 깊은 공간적, 시간적 및 인과적 관계 이해가 필요한 작업에서는 성능이 급격히 저하됨을 보여주었습니다.



### Leveraging Large Language Models for Enhancing Public Transit Services (https://arxiv.org/abs/2410.14147)
Comments:
          24 pages, 18 figures, submitting to Journal of ITS

- **What's New**: 본 논문은 대형 언어 모델(LLM)을 공공 교통 시스템에 적용하여 사용자 경험과 교통 직원의 성과를 향상시키기 위한 혁신적인 도구 세트를 제안합니다. 특히 'TransitTalk'라는 디지털 어시스턴트를 통해 다양한 정보와 정책 세부 사항을 전달하는 방식으로 고객과의 소통을 개선하고자 합니다.

- **Technical Details**: 대형 언어 모델(LLMs)은 자연어 처리(NLP) 작업에서 활용되는 고급 딥 러닝(DL) 모델로, 인간과 유사한 텍스트를 이해하고 생성할 수 있는 능력을 갖추고 있습니다. 이들은 트랜스포머(Transformer) 구조를 기반으로 하여 세밀한 의존성 추적과 병렬 처리 능력을 통해 높은 효율성을 자랑합니다. 'TransitTalk' 도구는 사용자 요청을 이해하고 데이터셋에서 정보를 검색하는 등 사용자의 구체적인 요구에 맞는 정보를 제공하는 역할을 수행합니다.

- **Performance Highlights**: 이 연구에서는 세 가지 LLM 애플리케이션인 Tweet Writer, Trip Advisor, Policy Navigator를 소개합니다. 이들 애플리케이션은 각각 소셜 미디어를 통한 실시간 통보 자동화, 사용자 맞춤형 여행 추천, 정책 관련 질문에 대한 명확하고 개인화된 답변을 제공합니다. 이를 통해 교통 시스템 직원은 보다 효율적으로 정보를 전하고, 고객은 보다 친숙하게 여행 정보와 정책에 대한 답변을 받을 수 있습니다.



### QT-DoG: Quantization-aware Training for Domain Generalization (https://arxiv.org/abs/2410.06020)
Comments:
          Code will be released soon

- **What's New**: 이번 연구에서는 도메인 일반화( Domain Generalization, DG)를 위해 양자화 인식 훈련(Quantization-aware Training, QT-DoG)을 제안하며, 양자화가 손실 경관에서 더 평탄한 최솟값을 유도하여 도메인 일반화를 향상시킬 수 있음을 입증하였습니다. QT-DoG는 모델 압축을 목표로 하는 전통적인 양자화 방법과는 달리, 모델 가중치에 노이즈를 유도함으로써 최적화 프로세스를 선수에 대한 민감성이 낮은 평평한 최솟값으로 유도합니다.

- **Technical Details**: QT-DoG는 QAT를 사용하여 양자화를 통해 손실 경관에서 평탄한 최솟값을 얻을 수 있음을 보여줍니다. 양자화는 가능한 가중치 값을 더 낮은 비트 정밀도로 제한함으로써 가중치 공간에 제약을 부여하고 네트워크 매개변수에 양자화 노이즈를 도입합니다. 이러한 노이즈는 최적화 프로세스가 훨씬 평평한 최솟값으로 수렴할 수 있도록 돕는 정규화의 형태로 작용합니다.

- **Performance Highlights**: QT-DoG는 다양한 데이터셋, 아키텍처 및 양자화 알고리즘에서 일반화되며, 다른 DG 방법들과 결합하여도 그 유연성과 강건성을 입증합니다. EoQ(양자화 앙상블)는 단일 풀 정밀 모델과 동일한 계산 비용과 메모리 풋프린트를 유지하면서도 State-of-the-art(DG) 성능을 달성하였고, 기존 방법보다 우수한 정확성을 보여주었습니다.




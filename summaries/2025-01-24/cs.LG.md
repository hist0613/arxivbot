New uploads on arXiv(cs.CL)

### CRPO: Confidence-Reward Driven Preference Optimization for Machine Translation (https://arxiv.org/abs/2501.13927)
- **What's New**: 본 연구는 Confidence-Reward driven Preference Optimization (CRPO)라는 새로운 방법을 제안하며, 이는 보상 점수(reward scores)와 모델 신뢰도(model confidence)를 결합하여 데이터 선정(data selection)의 효율성을 높이는 데 초점을 맞추고 있습니다. CRPO는 모델이 불확실하거나 성능이 저조한 문장 쌍을 선택하여 효과적인 학습을 유도하는 방법으로 설계되었습니다. 이를 통해 대규모 언어 모델(LLM)의 번역 성능을 향상시키는 것을 목표로 하고 있습니다.

- **Technical Details**: CRPO는 번역 품질을 극대화하기 위해 사람의 선호 데이터(preference data)를 사용해 LLM을 세밀하게 조정(fine-tuning)하는 과정을 포함합니다. 특히, 이 방법은 고수익 문장 쌍들 중에서도 모델의 불확실성이 높은 문장 쌍을 선정하여, 실제로 학습이 필요한 데이터를 더욱 효과적으로 선택합니다. CRPO는 검증된 효율성을 발휘하여 기존의 RS-DPO 및 RSO와 같은 방법들을 초과하는 성능을 보여주었습니다.

- **Performance Highlights**: CRPO는 다양한 모델, 특히 LLM 및 encoder-decoder 모델에서 뛰어난 성능을 입증하였습니다. 실험 결과, CRPO는 translation accuracy와 data efficiency 모두에서 기존의 방법들과 비교하여 향상된 성능을 나타내었습니다. 이는 CRPO의 유연성과 효율 성능을 강조하며, 언어 번역의 품질을 개선하는 데 중요한 기여를 할 것으로 기대됩니다.



### The Breeze 2 Herd of Models: Traditional Chinese LLMs Based on Llama with Vision-Aware and Function-Calling Capabilities (https://arxiv.org/abs/2501.13921)
- **What's New**: Breeze 2는 전통 중국어(Traditional Chinese) 표현을 강화하기 위해 설계된 3B 및 8B 매개변수 구성을 가진 다중 모달 언어 모델의 모음입니다. Llama 3를 기반으로 하여 방대한 말뭉치에서 지속적으로 사전 학습(pretraining)되며, 비전 인식 기능(vision-aware capabilities)과 기능 호출(function-calling) 지원을 통합하고 있습니다. 이는 대만 일반 지식, 지침 수행, 긴 맥락 처리, 기능 호출, 비전 이해 등 다양한 작업에서 그 효율성을 입증하고 있습니다.

- **Technical Details**: Breeze 2는 'ViT-MLP-LLM' 아키텍처를 채택하여 언어 및 비전 모델의 장점을 효과적으로 활용합니다. 두 가지 주요 구성 요소인 Llama 3 모델과 InternViT-300M-448px 비전 인코더를 통합하여 비주얼과 언어 피처 간의 다층 퍼셉트론(MLP) 프로젝터로 연결합니다. 이러한 구성을 통해 Breeze 2는 전통 중국어와 대만 관련 콘텐츠에서 강력한 성능을 발휘할 수 있도록 설계되었습니다.

- **Performance Highlights**: Breeze 2의 성능 평가는 여러 작업을 통해 이루어졌으며, 특히 대만 관련 지식 및 multimodal 작업에서 우수한 결과를 보였습니다. 3B 모델은 모바일 앱에 적용 가능하도록 설계되어 대중의 접근성을 용이하게 하며, 이 모델은 사용자 인터페이스에서 지침을 따르는 능력이 강화되었습니다. 모든 Breeze 2 모델은 Llama 3 커뮤니티 라이센스에 따라 공개될 예정입니다.



### Analysis of Indic Language Capabilities in LLMs (https://arxiv.org/abs/2501.13912)
Comments:
          17 pages, 2 figures, 5 tables

- **What's New**: 이번 보고서는 텍스트 입력과 텍스트 출력을 통해 인디언 언어를 이해하고 생성할 수 있는 대규모 언어 모델(LLMs)의 성능을 평가합니다. 이를 통해 향후 안전 기준에 적합한 인디언 언어의 우선 순위를 정할 수 있는 방안을 제시합니다. 연구는 기존의 평가 연구와 데이터 세트, 그리고 인디언 언어를 지원하는 28개의 LLM을 검토함으로써 수행되었습니다. 이 과정에서 각 언어에 대한 성능 격차를 분석하고, 힌디가 모델에서 가장 널리 사용되는 언어임을 발견했습니다.

- **Technical Details**: 대규모 언어 모델은 다양한 자연어 처리(NLP) 과제에서 최첨단 성능을 보이고 있으며, 이들 중 다수는 대화형 인터페이스를 통해 배포됩니다. 특히 OpenAI의 GPT-4 및 Anthropic의 Claude와 같은 모델들은 다국어 기능을 지원한다고 주장하지만, 언어 간 학습 효과와 추가 데이터의 부족이 이들 모델의 성능에 영향을 미칠 수 있습니다. 또한, 언어 자원이 부족한 언어의 경우 LLM의 모델 성능이 낮아질 수 있으며, 이는 실제 언어 사용과의 격차를 초래할 수 있습니다.

- **Performance Highlights**: 힌디와 같은 주요 언어의 모델 성능은 화자 수와 대체로 상관관계가 있지만, 그 이후의 언어들에서는 성능이 크게 차이나는 경향을 보입니다. 특히 보고서에서는 인디언 언어의 실제 사용과 LLM의 성능을 대비하여, 향후 벤치마크에 포함될 인디언 언어의 우선 순위를 정하기 위한 제안을 합니다. 이 연구는 인디언 언어를 지원하는 LLM의 성능 향상 및 안전 기준 확립에 기여할 것으로 기대됩니다.



### GUI-Bee: Align GUI Action Grounding to Novel Environments via Autonomous Exploration (https://arxiv.org/abs/2501.13896)
- **What's New**: 이 논문은 GUI 행동 기반 모델을 새로운 GUI 환경에 맞춤화하여 성능을 개선하는 방법을 제안합니다. GUI-Bee라는 자율 탐색 에이전트를 통해 환경별 데이터를 수집하고, 이를 사용하여 모델을 지속적으로 미세 조정합니다. 이 접근법은 환경에 따라 달라지는 GUI 기반 모델 성능의 한계를 극복할 수 있는 중요한 단계를 제공합니다.

- **Technical Details**: 제안된 GUI-Bee 에이전트는 Q-value-Incentive In-Context Reinforcement Learning (Q-ICRL) 방법을 사용하여 탐색 효율성을 최적화합니다. 이 방법은 GUI 행동 후보의 상태-행동 값 예측을 통해 최적의 행동을 선택하고 반복적이지 않은 행동을 피할 수 있게 합니다. 실험을 통해 NovelScreenSpot 벤치마크를 사용하여 다양한 환경에 대한 모델 성능을 평가하고 있습니다.

- **Performance Highlights**: 우리의 실험 결과는 GUI-Bee 에이전트를 사용하는 모델이 미세 조정 전보다 크게 성능을 향상시켰음을 보여줍니다. Q-ICRL 방법이 데이터 수집의 효율성을 극대화했으며, 모델들이 새로운 GUI 환경에 적응하는 데 필요한 환경별 지식을 효과적으로 학습했음을 확인했습니다. 이러한 기여는 GUI 행동 모델이 실질적으로 다양한 환경에서 더 나은 기능을 발휘하도록 합니다.



### A RAG-Based Institutional Assistan (https://arxiv.org/abs/2501.13880)
- **What's New**: 이번 연구는 University of São Paulo (USP)를 위한 Retrieval-Augmented Generation (RAG) 기반 가상 비서 시스템을 개발하고 평가합니다. RAG 모델을 사용하여 정보 검색과 생성 모델을 통합함으로써, 기존 LLMs의 한계를 극복할 수 있는 방향을 제시하고 있습니다. 또한 다양한 하이퍼파라미터를 조정하여 시스템의 성능을 최적화하고, 해당 시스템의 효과를 비교 분석하여 귀중한 기초 자료를 제공합니다.

- **Technical Details**: RAG 시스템은 텍스트 검색을 위한 retriever 모듈과 질문에 따른 답변을 생성하는 generative 모델로 구성됩니다. 이 연구에서는 chunk size, 검색 문서 수 등 다양한 요소를 조정하여 성능을 평가했습니다. 저자들은 2023년 1월부터 2024년 5월까지의 노멀 문서 데이터베이스를 구축하고, 총 866개의 문서를 수집하여 QA 데이터셋을 생성했습니다. RAG 모델은 LLM의 정보를 보조할 수 있도록 요구되는 문서 조각을 통합하여 답변 정확도를 향상시키는 데 중요한 역할을 합니다.

- **Performance Highlights**: 전반적으로, 최적의 retriever 모델은 Top-5 정확도에서 30%를 달성하였고, generative 모델은 22.04%의 성과를 보였습니다. 특히 올바른 문서 조각이 LLM에 제공되었을 때 정확도가 54.02%로 크게 증가했으며, 이것은 30% 이상의 향상을 나타냅니다. 반면, 맥락 정보 없이 작업을 수행할 경우 정확도는 13.68%로 하락해 데이터베이스 접근의 중요성을 강조하고 있습니다.



### Think Outside the Data: Colonial Biases and Systemic Issues in Automated Moderation Pipelines for Low-Resource Languages (https://arxiv.org/abs/2501.13836)
- **What's New**: 이 논문은 저소득 자원이 부족한 언어를 사용하는 글로벌 남반구(들)에서의 콘텐츠 조정(moderation) 도구 개발의 도전 과제들을 살펴봅니다. 특히, 다양한 부족 자원 언어로 행동하는 AI 기자들과 실무자들의 반구조화(interview)된 대화를 통해 이슈를 세부적으로 분석합니다. 마지막으로, 내용이 없는 데이터 접근 제한이나 기존 시스템의 구조적 장벽이 시민적 권리에 미치는 영향에 대해 논의합니다.

- **Technical Details**: 논문에서는 저자원 언어의 글로벌 남반구에서 유해 콘텐츠 탐지 도구를 개발할 때 기술적 문제와 언어적 복잡성을 탐구합니다. 특히, 전통적 언어 모델과 전처리(Preprocessing) 기법이 영어 중심으로 설계되었기 때문에 말레이시아어, 스와힐리어, 아랍어 및 케추아어와 같은 언어에서 문제를 생성한다고 지적합니다. 예를 들어, 잘못된 토큰화(tokenization)에서 발생하는 문제와 여러 언어의 변화에 대한 민감성 부족이 저자원 언어의 콘텐츠 조정을 방해합니다.

- **Performance Highlights**: 이 연구는 저소득 자원 언어의 콘텐츠 조정 파이프라인이 역사적인 권력 불균형과 얽혀 있음을 보여줍니다. 여러 활성화 연구자가 인증한 사례를 통해 콘텐츠 조정 시스템이 경제적 유인 부족으로 인해 저소득 자원 언어를 가진 지역에서 불공정하게 운영되고 있음을 강조합니다. 이러한 연구 결과는 제안된 개선 방안이 단순한 기술적 수정만으로 이루어질 수 없음을 시사하며, 필요로 하는 더 깊은 시스템 변화를 요구합니다.



### Predicting Compact Phrasal Rewrites with Large Language Models for ASR Post Editing (https://arxiv.org/abs/2501.13831)
Comments:
          accepted by ICASSP 2025

- **What's New**: 이번 논문에서는 대규모 언어 모델(Large Language Models, LLMs)을 활용한 텍스트 리라이팅(task)에서 효율성을 높일 수 있는 새로운 편집 문구 표현(edit phrase representation)을 제안합니다. 이러한 표현은 기존의 span representation 대신, 구문 통계 기계 번역(phrase-based statistical machine translation)에서 영감을 받아 개발되었습니다. 연구자들은 이 방법이 ASR(Automatic Speech Recognition) 후 편집(task)에서 우수한 성과를 보일 것이라고 기대하고 있습니다.

- **Technical Details**: 제안된 문구 표현은 입력과 출력 사이의 겹치는 부분을 활용해 편집 작업의 수치적 표현을 압축할 수 있도록 돕습니다. 두 가지 새로운 표현 방식은 각 리라이트 패턴을 소스-타겟 구문 쌍으로 표현하거나, 좌우문맥 단어와 함께 타겟 문구만을 사용하는 방식입니다. 이러한 접근은 계산 비용을 줄이고, 의미적 일관성을 유지하며, 최종 출력의 품질을 확보하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 연구 결과, 제안된 문구 표현 방식이 기존의 span 모델과 비교했을 때, WER(Word Error Rate) 격차를 50-60% 줄이는 효율성을 보이라고 보고했습니다. 또한, 출력 길이 감소율 또한 기존 방식에 비해 10-20% 손실로 괜찮은 성능을 유지했습니다. 이로써 ASR 출력 수정에 대한 새로운 가능성을 열었으며, LLMs의 활용을 더욱 넓힐 수 있는 길을 제시했습니다.



### Hallucinations Can Improve Large Language Models in Drug Discovery (https://arxiv.org/abs/2501.13824)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)에서 발생할 수 있는 환각(hallucination)이 오히려 약물 발견(drug discovery) 과정에서 LLM의 성능을 향상시킬 수 있다는 가설을 제안합니다. 연구진은 LLM을 활용해 화합물의 SMILES 문자열을 자연어로 설명하고, 이 설명을 약물 발견을 위한 특정 작업의 프롬프트로 사용했습니다. 7개의 LLM과 5개의 분류 작업을 대상으로 한 평가 결과, 환각이 포함된 텍스트가 성능 향상에 기여함을 확인하였습니다.

- **Technical Details**: 연구에서는 LLM을 사용해 분자의 SMILES 문자열을 기반으로 텍스트 설명을 생성하고, 이를 모델의 입력으로 추가하여 특정 기능을 예측하는 작업을 수행했습니다. Llama-3.1-8B 모델이 ROC-AUC에서 18.35% 향상된 결과를 보였으며, GPT-4o가 생성한 환각은 다양한 모델에서 일관된 성능 개선을 이끌어냈습니다. 또한 모델 크기, 생성 온도, 언어와 같은 여러 요소가 환각과 모델 성능에 미치는 영향을 조사했습니다.

- **Performance Highlights**: 환각을 활용한 LLM의 성능 향상이라는 주제는 아직 탐구되지 않은 분야로, 이 연구는 첫 번째 체계적인 조사를 제시합니다. Llama-3.1-8B 모델은 SMILES 기준에 비해 18.35%의 성능 향상을 보였으며, 더 큰 모델일수록 환각을 통한 성능 개선 가능성이 더 높음을 확인했습니다. 이 연구는 약물 개발 분야에서 LLM의 활용에 대한 새로운 관점을 제시하며, 미래 연구에 기여할 수 있는 통찰력을 제공합니다.



### Generation of reusable learning objects from digital medical collections: An analysis based on the MASMDOA framework (https://arxiv.org/abs/2501.13806)
Comments:
          first submited

- **What's New**: 이 논문에서는 다양한 교육 환경에서의 교육 자료 구조화 방법인 Learning Objects에 대한 새로운 접근 방식을 제시합니다. Clavy라는 도구를 활용하여 여러 의료 지식 출처로부터 데이터를 검색하고 다채로운 멀티미디어 기반 구조를 형성하는 과정을 질적 분석하고 있습니다. 이는 학생과 의료 종사자가 쉽게 접근하고 사용할 수 있는 재사용 가능한 학습 객체(Reusable Learning Objects, RLOs) 생성을 목표로 하고 있습니다.

- **Technical Details**: Clavy는 의료 지식 소스로부터 데이터를 수집하고 이를 다양한 교육 시나리오와 사용자 프로필에 맞게 적응시키는 기능을 갖추고 있습니다. 또한, Clavy는 교육 표준 사양에 따라 학습 객체를 내보낼 수 있는 기능을 제공하여 재사용성을 높이고 있습니다. 이러한 접근은 의료 교육 분야에서 필요로 하는 개인화된 학습 경험을 지원합니다.

- **Performance Highlights**: Clavy의 중요성은 의료 디지털 컬렉션에서 지식을 전이하여 의료 학생과 전문가가 가장 인기 있는 e-learning 플랫폼을 통해 쉽게 접근할 수 있도록 하는 데 있습니다. 이 연구는 Clavy를 통해 생성된 학습 객체가 의료 교육의 질을 높이고, 다양한 학습 요구사항에 맞추어 적절한 교육 콘텐츠를 제공할 수 있음을 강조합니다.



### Parameter-Efficient Fine-Tuning for Foundation Models (https://arxiv.org/abs/2501.13787)
Comments:
          25 pages, 6 figures, 7 tables

- **What's New**: 본 설문조사는 Foundation Models (FMs)와 관련된 Parameter-Efficient Fine-Tuning (PEFT) 기술에 대한 포괄적인 리뷰를 제공합니다. PEFT는 비용 효율적인 미세 조정 기술로, 알고리즘의 매개변수(parameter) 수를 최소화하며, 최적의 다운스트림(task)의 성과를 목표로 합니다. 다양한 FMs에 적용된 PEFT 기술을 탐구하는 이 연구는 이러한 통합의 잠재력을 명확히 하고, 향후 연구 및 개발 방향을 제시합니다.

- **Technical Details**: Foundation Models (FMs)은 대규모 데이터셋에서 미리 훈련되어 언어 이해, 코드 생성 및 이미지 처리와 같은 다양한 작업을 지원하는 모델입니다. PEFT는 이러한 모델을 더욱 효과적으로 미세 조정하는 기술로, LoRA( Low-Rank Adaptation)와 같은 여러 방법론을 통해 매개변수와 계산량을 획기적으로 줄입니다. 또한, 각 FM 유형에 따라 다양한 적응 전략이 요구되며, PEFT는 이러한 다양성을 쉽게 처리할 수 있는 방식으로 발전하고 있습니다.

- **Performance Highlights**: 설문 결과는 PEFT의 성과가 각 FM에서 99.97% 이상의 매개변수 절약과 함께 수행된 성능 향상을 보여준다는 것을 나타냅니다. 특히, LLMs와 Vision Foundation Models (VFMs)이 현재 연구에서 주도적인 위치를 차지하며, 멀티모달 모델(Multi-Modal Models)도 앞으로 연구의 주요 관심사로 떠오르고 있습니다. 또한, 저자들은 PEFT 기술의 통합을 통해 이러한 모델들이 앞으로 더 많은 다운스트림 작업에서도 효과적으로 활용될 수 있는 가능성을 제시합니다.



### Do Large Language Models Truly Understand Geometric Structures? (https://arxiv.org/abs/2501.13773)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)의 기하학적 능력을 평가하기 위해 새로운 데이터셋인 GeomRel을 소개합니다. 기존 데이터셋은 LLM의 최종 답변만을 기준으로 평가했지만, GeomRel은 문제 해결 과정에서 기하학적 관계 인식을 고립시켜 LLM의 진정한 이해도를 측정할 수 있도록 설계되었습니다. 또한, 기하학적 관계 인식을 개선하기 위한 Geometry Chain-of-Thought(GeoCoT) 방법을 제안하여 성능 향상을 도모합니다.

- **Technical Details**: GeomRel 데이터셋은 LLM이 주어진 기하학적 구조 설명에 따라 명시적 또는 암시적 기하학적 관계를 정확히 식별할 수 있는 능력을 측정합니다. 기본 기하학적 요소인 점, 선, 각, 도형을 기반으로 기하학적 관계 풀을 생성하였고, 다양한 기하학적 시나리오를 조직하여 기본 및 고급 데이터셋을 구축했습니다. 이 과정에서 규칙 기반 운영을 통해 복잡한 기하학적 구조 또한 생성하였습니다.

- **Performance Highlights**: 실험 결과, 현재 LLM들은 간단한 기하학적 관계 식별에는 우수한 성과를 보이지만, 복잡한 구조, 특히 각 기반 관계를 이해하는 데에는 한계를 보였습니다. GeoCoT 방법을 사용할 경우, 기본 GeomRel 데이터셋에서 평균 9.15%, 고급 GeomRel 데이터셋에서는 14.79%의 성능 향상이 있었으며, 이는 다양한 도메인에서 기하학적 관계 인식 능력의 개선을 보여줍니다.



### UGMathBench: A Diverse and Dynamic Benchmark for Undergraduate-Level Mathematical Reasoning with Large Language Models (https://arxiv.org/abs/2501.13766)
Comments:
          Accepted to ICLR 2025

- **What's New**: UGMathBench는 대학 수준의 수학 문제를 평가하기 위해 특별히 설계된 역동적이고 다양한 벤치마크로, 총 5,062개의 문제를 포함하며, 이는 16개의 주제와 111개의 세부 주제로 나뉘어져 있습니다. 각 문제는 세 가지 버전으로 무작위화되어 있어 모델의 진정한 추론 능력을 평가할 수 있습니다. 또한, 해당 연구는 효과적 정확도(Effective Accuracy, EAcc)와 추론 갭(Reasoning Gap, Δ)이라는 두 가지 주요 메트릭을 도입하여 평가의 정밀성을 높였습니다.

- **Technical Details**: UGMathBench는 5,062개의 문제로 구성되어 있으며, 이 문제들은 여덟 가지 기본 답변 유형과 두 가지 복합 답변 유형으로 분류됩니다. 각 문제의 변형은 LLM의 진정한 추론 능력을 평가하는 데 기여하며, 이상적으로 EAcc가 높고 Δ가 0인 '대형 추론 모델'의 개발 필요성이 강조됩니다. 본 연구의 평가에서 OpenAI-o1-mini가 56.3%의 최고 EAcc를 기록했으며, 모든 LLM은 높은 추론 갭을 드러냈습니다.

- **Performance Highlights**: UGMathBench의 평가는 LLM의 현재 성능 한계를 드러내며, OpenAI-o1-mini와 같은 가장 발달한 모델조차 56.3%의 EAcc를 기록하는 데 그쳤습니다. LLM의 Robustness Efficiency(Δ와 EAcc 비율)는 20.78%에서 196.6%까지 다양해 현재 모델들의 일관성 부족이 드러났습니다. 각 주제에 따른 평균 EAcc는 차이를 보이며, 특히 Arithmetic(62.8%)와 Absctract Algebra, Differential Equations, Financial Mathematics(10% 이하) 간의 큰 격차가 나타났습니다.



### 2-Tier SimCSE: Elevating BERT for Robust Sentence Embeddings (https://arxiv.org/abs/2501.13758)
- **What's New**: 이번 연구는 자연어 처리(NLP)에서 의미 있는 문장 임베딩 생성의 중요성을 강조하고, SimCSE라는 새로운 접근법을 통해 minBERT 모델을 감정 분석, 의미적 텍스트 유사성(semantics textual similarity, STS), 그리고 패러프레이즈 감지 과제를 위해 미세 조정하는 것을 목표로 합니다. 이 과정에서 세 가지 서로 다른 드롭아웃 기법을 실험하여 오버핏(overfitting) 문제를 해결하고, 비지도 및 지도 SimCSE를 결합한 2-Tier SimCSE 미세 조정 모델을 제안했습니다. 연구 결과는 2-Tier 모델이 STS 과제에서 높은 성능을 기록했음을 보여줍니다.

- **Technical Details**: 연구에서 사용된 기본 모델은 다중 작업 및 단일 작업 버전의 minBERT로, 12개의 트랜스포머(transformer) 레이어를 통해 문장 토큰화와 임베딩 결합, 다중 헤드(self-attention mechanism)를 구현합니다. 단일 작업 모델은 각 다운스트림 과제를 개별적으로 학습하고 미세 조정하여, 작업별로 초점을 맞춘 진행을 보장합니다. 또한 3가지 드롭아웃 기법(표준 드롭아웃, 커리큘럼 드롭아웃, 적응형 드롭아웃)을 활용하여 모델의 일반화 성능을 개선하려고 했습니다.

- **Performance Highlights**: 모델은 Unsupervised SimCSE와 Supervised SimCSE를 활용하여 STS 데이터셋에서 각기 다른 점수로 뛰어난 성능을 보였습니다. Unsupervised SimCSE는 0.716의 피어슨 상관 계수(Pearson Correlation score)를 달성했고, Supervised SimCSE는 0.806의 성능을 기록했습니다. 그러나 패러프레이즈 및 SST 과제에서의 성능 향상은 제한적이었으며, STS에서의 지식을 전이하는 데 한계가 있음을 시사합니다.



### A Study of the Plausibility of Attention between RNN Encoders in Natural Language Inferenc (https://arxiv.org/abs/2501.13735)
- **What's New**: 최근 NLP(자연어 처리) 분야에서 Attention Mechanism의 활용과 그 설명 가능성에 대한 연구가 증가하고 있습니다. 본 논문에서는 전통적인 텍스트 분류를 넘어 Sentence Comparison(NLI) 작업에서 Attention Map의 신뢰성과 실용성을 탐구합니다. 실험을 통해 Annotation(주석) 결과와 휴리스틱 방식의 상관관계를 분석하고, 두 RNN 인코더 간의 Attention Weight를 비교합니다.

- **Technical Details**: Attention Mechanism은 입력 단어의 중요도를 측정하여 결정 과정을 시각화하는데 도움을 줍니다. 일반적으로 Attention Map은 각 입력 단어에 가중치를 부여하여 자신의 모델 의사를 설명하는 도구로 사용됩니다. 본 연구에서는 eSNLI 데이터셋을 활용하여 두 문장을 비교하는 NLI 작업에서 인간 주석과 휴리스틱 방법을 사용해 Attention Weight의 상관관계를 규명했습니다.

- **Performance Highlights**: 연구 결과, 제안한 휴리스틱 방법은 인간 주석과 상당한 상관관계를 보여주어 Attention Mechanism에 대한 이해를 돕고 신뢰할 수 있는 설명 제공을 가능케 합니다. 그러나 원시 Attention Weight는 그 자체로는 그 설명 가능성과 긴밀히 연결되어 있지 않음을 보여줍니다. 이 연구는 향후 모델의 설명 가능성을 높이기 위한 기반을 마련할 것으로 기대됩니다.



### Pseudocode-Injection Magic: Enabling LLMs to Tackle Graph Computational Tasks (https://arxiv.org/abs/2501.13731)
Comments:
          24 pages

- **What's New**: 본 논문은 그래프 계산 작업에 대해 새로운 접근 방식을 제안합니다. 기존 대형 언어 모델(LLMs)이 직면한 문제를 해결하기 위해 새로운 프레임워크인 PIE(Pseudocode-Injection-Enhanced LLM Reasoning for Graph Computational Tasks)를 도입합니다. 이 프레임워크는 문제 이해, 프롬프트 설계 및 코드 생성의 세 가지 핵심 단계로 구성됩니다.

- **Technical Details**: 프레임워크 PIE는 LLMs가 문제를 이해하고 관련 정보를 추출하여 코드를 생성하는 역할을 맡도록 하며, 그래프 구조를 분석하고 코드를 실행하는 책임은 인터프리터에 위임됩니다. 실험을 통해 PIE가 기존 방법들보다 정확도와 계산 효율성에서 뛰어난 성능을 보임을 입증하였습니다. 특히, Pseudocode injection 기법을 활용하여 LLMs가 효율적인 코드를 생성할 수 있도록 지원합니다.

- **Performance Highlights**: 아홉 가지 그래프 추론 작업에 대한 실험 결과, PIE는 높은 정확도로 기존 방법들보다 월등한 성능을 제공하는 동시에 계산 비용을 현저히 절감했습니다. 또한, LLMs에 의한 코드 생성을 통해 다양한 그래프 계산 작업에서의 효율성 또한 크게 향상되었습니다. 이로 인해, PIE는 저지연 및 고도 확장성이 요구되는 응용 프로그램에서도 실용성이 높은 해결책으로 자리 잡을 수 있습니다.



### RPO: Retrieval Preference Optimization for Robust Retrieval-Augmented Generation (https://arxiv.org/abs/2501.13726)
- **What's New**: 이 논문에서는 Retrieval-Preference Optimization(RPO)이라는 새로운 경량 최적화 방법을 도입하여 다중 출처 지식을 효과적으로 활용할 수 있도록 LLM의 탐색 평가를 생성 프로세스에 통합합니다. 기존의 Retrieval-Augmented Generation(RAG) 방법들이 지식 충돌을 해결하기 위해 복잡한 절차를 필요로 했던 반면, RPO는 단일 모델 내에서 평가와 생성을 통합하여 이러한 단점을 해결합니다. 이 방법은 RAG 전용 정렬 접근법으로, 학습 과정에서 탐색 관련성을 정량화하여 다루는 다기능성을 특징으로 합니다.

- **Technical Details**: RPO는 강화 학습을 통해 가장 적합한 지식을 선택할 수 있도록 설계되었습니다. 이 방법은 기존의 선후 평가 방식과는 달리 탐색 및 생성 과정을 동시에 최적화합니다. 또한, 수학적으로 기존의 방식들이 RAG 목표를 위반하는 제한 요소를 명확히 하여, RPO가 어떻게 이러한 문제를 극복하는지를 증명합니다. 실험을 통해 RPO는 파라메트릭 지식과 논파라메트릭 지식 간의 충돌을 최소화하고, 이를 통해 알고리즘의 신뢰성을 극대화합니다.

- **Performance Highlights**: RPO 알고리즘은 PopQA, Natural Questions, TriviaQA, RGB의 네 가지 데이터셋에서 기존 RAG 방법보다 4-10% 더 높은 정확도로 성능을 보여주었습니다. 실험 결과에 따르면, RPO는 추가적인 컴포넌트 없이도 RAG의 일반화 성능을 향상시키며, 다양한 벤치마크에서 두각을 나타냈습니다. 이로 인해 RPO는 지식 검색 및 생성 과정에서의 의존성을 최소화하고, LLM의 전반적인 성능을 개선시키는 데 성공했습니다.



### Musical ethnocentrism in Large Language Models (https://arxiv.org/abs/2501.13720)
- **What's New**: 이 논문은 대형 언어 모델(LLM)의 음악 관련 편향을 분석하기 위한 첫 걸음을 제시합니다. 특히 ChatGPT와 Mixtral 모델을 대상으로 하여, 두 가지 실험을 통해 이들 모델이 서구 음악 문화에 대한 강한 선호를 나타내는 것을 발견했습니다. 연구의 주요 초점은 지리 문화적 편향인 'geocultural biases'를 탐구하는 것입니다.

- **Technical Details**: 연구는 두 가지 실험을 통해 수행되었습니다. 첫 번째 실험에서는 다양한 카테고리의 'Top 100' 음악 기여자 리스트를 요청하고, 그들의 원산지를 분석했습니다. 두 번째 실험에서는 여러 나라의 음악 문화 특성을 수치적으로 평가하도록 LLM에 요청하였습니다. 이를 통해 문화 및 지역의 불균형한 대표성과 관련된 편향을 정량적으로 평가하고자 했습니다.

- **Performance Highlights**: 실험 결과, 미국을 포함한 서양 국가에 대한 집중이 두드러졌으며 아시아와 아프리카는 크게 저평가되는 경향이 있었습니다. 특정 카테고리에서는 약간의 다양성이 발견되었지만, 전반적으로 LLM은 서구 음악 문화에 대한 선호를 강하게 드러냈습니다. 이 연구는 LLM의 문화적 감수성 개발을 위한 기초 연구로 평가됩니다.



### DI-BENCH: Benchmarking Large Language Models on Dependency Inference with Testable Repositories at Sca (https://arxiv.org/abs/2501.13699)
- **What's New**: DI-BENCH라는 새로운 대규모 벤치마크와 평가 프레임워크가 소개되었습니다. 이 프레임워크는 LLM의 의존성 추론(dependency inference) 능력을 평가하기 위해 특별히 설계되었습니다. 총 581개의 리포지토리를 포함하고 있으며, Python, C#, Rust, JavaScript 같은 다양한 프로그래밍 언어를 지원합니다. 이 연구는 현재 LLM의 성능 평가에 새로운 관점을 제공하여 보다 견고한 소프트웨어 합성을 위한 토대를 마련합니다.

- **Technical Details**: DI-BENCH는 내부 및 외부 의존성을 식별하는 모델의 능력을 평가합니다. 이를 위해 향상된 CI(Continuous Integration) 기반 실행 평가를 도입하여 자동화된 테스트 환경에서 리포지토리의 실행 가능성을 평가합니다. 기존의 리포지토리 수준 벤치마크와는 달리, DI-BENCH는 의존성 메타데이터에 대한 챌린지를 구체적으로 다루면서 모델의 성능을 종합적으로 분석하는 새로운 메트릭을 제시합니다.

- **Performance Highlights**: 현재 최상의 모델은 오직 42.9%의 실행 성공률을 기록하며, 이는 개선의 여지가 많다는 것을 나타냅니다. 의존성의 양과 리포지토리 크기가 성능에 영향을 미친다는 사실이 밝혀졌으며, 특히 'hallucination' 문제와 의존성 메타데이터와 관련된 챌린지가 모델 성능에 부정적인 영향을 미치고 있습니다. 이러한 실험 결과는 LLM의 더 나은 의존성 추론 능력을 위해 앞으로 나아가야 할 방향을 제시합니다.



### Question Answering on Patient Medical Records with Private Fine-Tuned LLMs (https://arxiv.org/abs/2501.13687)
- **What's New**: 본 연구에서는 전자 건강 기록(EHR)의 정보 검색을 위해 새롭게 제안된 의미론적 질의 응답(semantic QA) 방식을 소개합니다. 이 방법은 사용자 질의에 가장 유의미한 FHIR 리소스를 식별한 후, 해당 리소스를 기반으로 응답을 생성하는 두 단계의 과정으로 이루어집니다. 또한, 사적인 환경에서 운영되는 LLM(대형 언어 모델)을 활용하여 환자의 개인 정보를 보호하는 동시에 효과적인 건강 데이터 접근을 가능하게 합니다.

- **Technical Details**: 연구의 두 가지 주요 작업(Task 1 & Task 2)은 각각 FHIR 리소스 검색과 질의에 대한 응답 생성을 포함합니다. 각 작업은 LLM을 활용하여 모델을 미세 조정(fine-tuning)하는 방식을 채택하고, 특히 Llama-3.1-8B 및 Mistral-NeMo 모델을 사용하여 정확성과 효율성을 극대화합니다. 데이터 수집과 정제, 그리고 여러 모델의 성능을 비교하는 단계를 통해 최적의 모델 구성을 평가합니다.

- **Performance Highlights**: 실험 결과, 미세 조정된 LLM이 기본적인 GPT-4 모델보다 Task 1의 F1 점수에서 0.55% 더 뛰어나고, Task 2의 Meteor Task에서는 42% 높은 성능을 보였습니다. 이러한 결과는 환자 중심의 의미론적 질의 응답을 수행할 때, LLM의 개인 정보 보호와 데이터 효율성을 동시에 확보할 수 있음을 보여줍니다.



### Collective Memory and Narrative Cohesion: A Computational Study of Palestinian Refugee Oral Histories in Lebanon (https://arxiv.org/abs/2501.13682)
Comments:
          Appeared in the 1st International Workshop on Nakba Narratives as Language Resources as part of COLING 2025

- **What's New**: 이번 연구는 팔레스타인 구술 역사 아카이브(POHA)를 활용하여 레바논 내 팔레스타인 난민 그룹들이 Nakba에 대한 집합적 기억을 어떻게 유지하는지를 조사합니다. 이 연구는 Halbwachs의 그룹 기억 이론에 기초하여, 나레이티브의 쌍별 유사성에 대한 통계적 분석을 통해 성별과 위치의 영향을 집중적으로 분석합니다. 연구는 공유된 기원이 나레이티브 유사성의 강력한 결정 요인이라는 점과 거주지를 공유하는 것이 집합적 정체성을 조성하는 데 중요한 역할을 한다는 것을 강조합니다.

- **Technical Details**: 연구는 레바논에서의 난민 개인들의 이야기를 통해 공통된 주제, 랜드마크, 그리고 중요한 인물들을 나레이티브로 묘사하여, 이들이 어떻게 Nakba를 기억하는지를 분석합니다. 자연어 처리(NLP) 기법을 사용하여 POHA의 질문 및 응답을 수치적으로 분석함으로써, 난민 커뮤니티 내에서 Nakba를 기억하는 정도의 유사성을 측정합니다. 또한 본 연구는 Halbwachs의 그룹 기억 이론을 이론적 틀로 삼아, 난민들의 위치와 성별과 같은 경계 형성 정체성 마커들이 팔레스타인 기억의 형성에 미치는 영향을 탐구합니다.

- **Performance Highlights**: 연구 결과, 기원을 공유하는 것이 난민 나레이티브의 유사성을 더욱 높이고, 거주지를 공유하는 경우도 유사성을 나타냅니다. 또한, 여성의 나레이티브는 특히 영국 점령 경험을 회상하는 데 있어 주제적인 응집력을 강화하는 경향이 있음을 발견했습니다. 궁극적으로 이 연구는 팔레스타인 난민 커뮤니티 내 집합적 기억 형성이 어떻게 이루어지는지를 깊게 이해하는 데 기여하며, 구술 역사가 팔레스타인 정체성을 보호하고 지우는 시도에 저항하는 데 있어 필수적인 역할을 한다고 강조합니다.



### How to Complete Domain Tuning while Keeping General Ability in LLM: Adaptive Layer-wise and Element-wise Regularization (https://arxiv.org/abs/2501.13669)
Comments:
          Work in progress

- **What's New**: 본 논문은 대규모 언어 모델(LLMs)의 망각 문제(catastrophic forgetting)를 해결하기 위한 새로운 접근 방식을 제안합니다. 일반 지식(general knowledge)을 보존하면서 도메인 특정 태스크에 적응하도록 모델 파라미터의 중요성을 개별적으로 평가하는 방법입니다. 이 방법은 두 가지 손실 함수(Regularization Loss와 Cross-Entropy Loss)를 조합하여 활용하며, 레이어별(coefficient)로 기여도를 조정합니다.

- **Technical Details**: 제안된 방법은 자연어 처리 모델을 위한 고급 최적화 전략을 활용합니다. Regularization Loss는 일반 지식을 유지하기 위해 중요한 파라미터 업데이트를 최소화하고, Cross-Entropy Loss는 도메인 특정 학습을 촉진합니다. 또한, 각 레이어의 기여도를 반영하는 레이어별 коэффициент를 통해 특정 레이어가 태스크 학습을 우선시하거나 일반 지식을 보존할 수 있도록 동적으로 조정합니다.

- **Performance Highlights**: 제안된 방법은 GPT-J 및 LLaMA-3을 사용하는 과학, 의학 및 물리적 태스크 실험에서 탁월한 성능을 보입니다. 기존 방법에 비해 거의 20배 더 빠르며, 저장 공간은 10%~15%만큼의 효율성을 보여줍니다. 이러한 실험 결과는 제안한 접근 방식이 연산 시간과 메모리 요구사항을 크게 줄이면서도 효과적인 성능 개선을 이루었다는 것을 나타냅니다.



### LVPruning: An Effective yet Simple Language-Guided Vision Token Pruning Approach for Multi-modal Large Language Models (https://arxiv.org/abs/2501.13652)
- **What's New**: 이 논문은 Language-Guided Vision Token Pruning (LVPruning)이라는 새로운 방법을 소개합니다. LVPruning은 Multi-modal Large Language Models (MLLMs)의 계산 부담을 줄이는 동시에 모델 성능을 유지할 수 있는 간단하면서도 효과적인 방법입니다. 이 방법은 시각 토큰과 언어 토큰 간의 상호작용을 기반으로 시각 토큰의 중요성을 계산하여 어떤 토큰을 제거할지 결정합니다.

- **Technical Details**: LVPruning은 경량의 cross-attention 모듈을 사용하여 언어 맥락에 따라 시각 토큰의 수를 동적으로 줄이는 방식으로 설계되었습니다. 이 모듈은 모델이 각 시각 토큰을 유지할지 제거할지를 결정하는 중요도를 계산합니다. 모든 원래 MLLM 매개변수를 변경하지 않고도 LVPruning을 구현할 수 있으며, 이로 인해 간편한 적용 및 제거가 가능합니다.

- **Performance Highlights**: LVPruning을 사용함으로써 LLaVA-1.5 모델의 중간 레이어에서 시각 토큰을 최대 90%까지 줄일 수 있으며, 이로 인해 추론 Tera Floating-Point Operations Per Second (TFLOPs)가 62.1% 감소합니다. 전반적인 성능 손실은 아홉 개의 멀티모달 지표에서 평균 0.45%에 불과하여, 효율성과 성능을 효율적으로 조화롭게 유지할 수 있음을 보여줍니다.



### Sigma: Differential Rescaling of Query, Key and Value for Efficient Language Models (https://arxiv.org/abs/2501.13629)
- **What's New**: 본 논문에서 소개하는 Sigma는 시스템 도메인에 특화된 효율적인 대형 언어 모델(large language model)입니다. 이 모델은 DiffQKV attention이라는 혁신적인 아키텍처를 통해 구축되었으며, 시스템 도메인 데이터를 철저히 수집하여 사전 학습되었습니다. DiffQKV attention은 성능 최적화에 따라 Query (Q), Key (K), Value (V) 구성 요소를 차별적으로 최적화하여 Sigma의 추론 효율성을 크게 향상시킵니다.

- **Technical Details**: DiffQKV attention은 두 가지 주요 기법, 즉 차별적으로 압축된 KV와 증강된 Q를 적용합니다. 차별적으로 압축된 KV는 성능 저하를 최소화하면서 V 벡터의 압축을 덜 엄격하게 적용합니다. 반면, 증강된 Q 기법은 Q 헤드의 차원을 늘려 모델의 표현력을 높이면서도 추론 속도에 미치는 영향을 최소화합니다.

- **Performance Highlights**: Sigma는 AIMicius라는 종합 벤치를 기반으로 여러 시스템 도메인 작업에서 뛰어난 성능을 보여주며, GPT-4에 비해 최대 52.5% 향상된 성능을 기록했습니다. 일반 도메인에서는 현재의 최신 모델들과 비교 가능한 성능을 보이고 있으며, 시스템 도메인 전반에서 우수한 결과를 나타냅니다.



### Domain-Specific Machine Translation to Translate Medicine Brochures in English to Sorani Kurdish (https://arxiv.org/abs/2501.13609)
Comments:
          12 pages, 6 figures,3 tables

- **What's New**: 이 연구는 쿠르드어 의학 정보의 접근성을 향상시키기 위해 영문 의약품 브로셔를 소라니 쿠르드어로 번역하는 특화된 기계 번역(Machine Translation, MT) 모델을 개발했습니다. 이를 위해 22,940개의 정렬된 문장 쌍으로 구성된 평행 코퍼스를 구축하였고, 모세스 툴킷을 사용하여 통계적 기계 번역(Statistical Machine Translation, SMT) 모델을 훈련시켰습니다. 이 모델은 여러 실험을 통해 BLEU 스코어를 평가하고, 최종적으로 현지 전문가의 피드백을 반영하여 번역의 품질을 검증하였습니다.

- **Technical Details**: 이번 연구에서 사용된 모세스(Moses) 모델은 쿠르드어 특유의 언어적 요소를 고려하여 통계적 기계 번역 기법에 기반하였습니다. 연구팀은 처음에 7회 실험을 진행하여 결과로 BLEU 스코어를 22.65에서 48.93까지 기록하였고, 이후 3개의 추가 브로셔 번역 작업을 통해 포스트 프로세싱(post-processing) 기법을 적용하였습니다. 이는 의료 사전(dictionary)을 활용하여 미등록 단어 문제를 해결하려는 노력의 일환으로, 최종적으로 BLEU 스코어 56.87, 31.05, 40.01을 얻었습니다.

- **Performance Highlights**: 휴먼 평가 결과, 50%의 전문가가 번역의 일관성을 느꼈으며, 83.3%가 번역의 정확성을 높이 평가했습니다. 또한, 일반 사용자 중 66.7%는 번역이 명확하다고 응답하며, 번역된 정보를 바탕으로 약물을 사용할 자신감을 느낀다고 밝혔습니다. 이러한 평가 결과는 개발된 MT 모델의 효과성을 강화하는 데 중요한 증거가 됩니다.



### Improving Contextual Faithfulness of Large Language Models via Retrieval Heads-Induced Optimization (https://arxiv.org/abs/2501.13573)
Comments:
          Submitted to ARR October 2024

- **What's New**: 본 연구에서는 Retrieval-augmented Large Language Models(LLMs)의 맥락적 신뢰성을 보장하기 위해 RHIO라는 새로운 프레임워크를 제안합니다. RHIO는 신뢰성 있는 생성을 식별하고 비신뢰성 있는 출력을 구별하는 데 도움을 주며, 특히 Long-Form Question Answering(LFQA)에서의 성능 향상을 목표로 합니다. 기존의 방법들과는 달리, RHIO는 비신뢰성 샘플을 생성하고 이를 훈련에 활용함으로써 모델이 비신뢰성 출력을 학습할 수 있도록 돕습니다.

- **Technical Details**: RHIO는 특정 attention heads인 retrieval heads가 LFQA의 신뢰성과 강하게 연관되어 있다는 점을 기반으로 합니다. 연구진은 LLM에서 retrieval heads를 선택적으로 마스킹하여 비신뢰성 샘플을 증강하는 방식을 사용합니다. 이를 통해 모델은 두 가지 컨트롤 토큰, [POS]와 [NEG]를 사용하여 신뢰성 있는 출력과 비신뢰성 있는 출력을 명확히 구분할 수 있습니다.

- **Performance Highlights**: GroundBench라는 새로운 벤치마크를 통해 RHIO의 성능을 평가한 결과, 7B 및 13B 모델에서 각각 12.84%와 12.59%의 신뢰성 향상을 보여주었으며, 이는 기존의 최첨단 모델인 GPT-4o를 능가하는 결과입니다. 또한, 인간 평가 결과를 통해 RHIO의 효과성과 실용성에 대한 추가 통찰력을 얻었습니다.



### K-COMP: Retrieval-Augmented Medical Domain Question Answering With Knowledge-Injected Compressor (https://arxiv.org/abs/2501.13567)
Comments:
          NAACL 2025

- **What's New**: 최근 연구에서는 K-COMP (Knowledge-injected compressor)를 제안하여 retrieval-augmented question answering (QA)의 정확성을 높였습니다. 이 방법은 reader model이 질문에 대한 정확한 답변을 위해 필요한 지식을 자동으로 생성하고, 이를 압축된 문맥에 통합하여 질문의 의도와 정렬을 보장합니다. 특히, 이 모델은 의료 도메인에 적합하도록 설계되어 실제 사례에서 효율성을 입증하였습니다.

- **Technical Details**: K-COMP의 핵심은 autoregressive LLM을 사용하여 질문에 필요한 도메인 지식을 주입하는 것입니다. 이를 통해 모델은 질문에 포함된 개체를 인식하고 관련된 정보를 제공할 수 있습니다. 이 과정에서 causal masking 기법을 활용하여, 질의의 맥락에 맞춘 압축된 요약을 생성합니다. 이는 K-COMP가 긴 입력 프롬프트에서 필요한 정보를 효과적으로 찾는 데 도움을 줍니다.

- **Performance Highlights**: 실험 결과 K-COMP는 세 가지 의료 데이터셋에서 기존의 방법들보다 우수한 성능을 보였습니다. 이 방법은 도메인 지식이 없는 reader model이더라도 의학적 전문 용어를 설명할 수 있게 하여, 다양한 배경의 모델들이 의학 질문을 보다 정확하게 처리할 수 있도록 지원합니다. K-COMP는 처음 보는 데이터에 적용했을 때도 효과가 입증되어, 데이터가 부족한 폐쇄형 도메인 환경에서 유용한 기여를 할 수 있음을 보여줍니다.



### LLMs Can Plan Only If We Tell Them (https://arxiv.org/abs/2501.13545)
Comments:
          ICLR 2025

- **What's New**: 이번 연구는 대규모 언어 모델(Large Language Models, LLMs)이 자율 계획에서 독립적으로 인간 수준의 장기 계획(long-horizon plans)을 생성할 수 있는지를 조사합니다. 특히, 이전 연구들에서는 외부 피드백 메커니즘과 통제된 환경에서 LLM을 활용했지만, 이러한 방법들은 상당한 계산 자원과 개발 자원을 요구했습니다. 이를 신경 쓰지 않고, LLM의 독립적인 계획 능력을 평가하기 위한 새로운 접근 방식을 채택하였습니다.

- **Technical Details**: 연구에서는 'Algorithm-of-Thoughts(AoT)'라는 기법에 혁신적인 개선을 도입하여 이를 'AoT+'라고 명명하였습니다. AoT+는 고급 계획 벤치마크에서 기존 방법들과 인간 기준을 능가하는 최신 성과(state-of-the-art results)를 달성하는 데 도움을 줍니다. 이 접근 방식은 별도의 지원 없이 모든 작업을 독립적으로 수행할 수 있도록 설계되었습니다.

- **Performance Highlights**: AoT+는 이전의 다른 방법들 및 인간의 성능과 비교했을 때 자율적으로 우수한 성과를 보였습니다. 특히 Blocksworld와 같은 표준 계획 벤치마크에서 LLM이 여전히 인간 성능을 초과하는 데 어려움을 겪고 있다는 점을 고려할 때, AoT+의 성과는 눈에 띕니다. 이는 LLM의 자율적 계획 능력을 크게 향상시킨 결과라 할 수 있습니다.



### RECALL: Library-Like Behavior In Language Models is Enhanced by Self-Referencing Causal Cycles (https://arxiv.org/abs/2501.13491)
- **What's New**: 본 논문에서는 self-referencing causal cycle(자기 참조 인과 주기)이라는 개념을 도입했습니다. 이는 대규모 언어 모델(LLM)이 단방향 인과 관계의 한계를 넘어설 수 있도록 하는 메커니즘입니다. 특히 RECALL이라는 방법론이 제안되며, 이 방법은 순차적 데이터를 처리하는 과정에서 발생하는 "reversal curse" 현상을 극복하도록 돕습니다.

- **Technical Details**: RECALL의 핵심은 cycle tokens(주기 토큰)으로, 이는 학습 데이터의 다양한 부분을 연결하는 시퀀스를 생성합니다. 이 주기 토큰은 모델이 후속 토큰으로부터 이전 토큰을 회상할 수 있도록 돕습니다. 논문은 확률적 형식화 및 통제된 실험을 통해 이러한 주기의 유도 방식이 정보 재생산 능력에 어떻게 영향을 미치는지를 분석합니다.

- **Performance Highlights**: RECALL 메커니즘을 통해 대규모 언어 모델의 정보 검색 능력이 향상됩니다. 특히, 자율회귀 모델에 대한 새로운 두 단계의 재귀 회수 과정을 제안하여 유용한 정보를 보다 효과적으로 회상해 내는 방식이 설명됩니다. 또한, 저자들은 이를 실재할 수 있는 코드 및 실험적 세부정보를 공개하여 재현 가능성을 강조하였습니다.



### Multi-Level Attention and Contrastive Learning for Enhanced Text Classification with an Optimized Transformer (https://arxiv.org/abs/2501.13467)
- **What's New**: 이 논문은 텍스트 분류 작업에서 모델의 성능과 효율성을 높이기 위해 개선된 Transformer 기반의 텍스트 분류 알고리즘을 연구합니다. 기존 Transformer 모델이 깊은 의미적 관계 캡처 및 계산 복잡성 최적화에서 갖는 단점을 고려하여, 다중 수준의 주의 메커니즘(multi-level attention mechanism)과 대조 학습 전략(contrastive learning strategy)을 도입했습니다.

- **Technical Details**: 다중 수준의 주의 메커니즘은 글로벌(attention)과 로컬(local) 주의를 결합하여 텍스트 내 글로벌 의미 및 로컬 특징을 효과적으로 모델링합니다. 대조 학습 전략은 양성(positive) 및 음성(negative) 샘플 쌍을 구성하여 모델의 카테고리 간 구별 능력을 강화합니다. 또한, 경량화 모듈(lightweight module)을 설계하여 대규모 텍스트 데이터를 위한 모델의 학습 및 추론 효율성을 개선합니다.

- **Performance Highlights**: 실험 결과, 개선된 Transformer 모델은 BiLSTM, CNN, 표준 Transformer 및 BERT와 같은 비교 모델들보다 분류 정확도(classification accuracy), F1 점수(F1 score), 재현율(recall rate) 면에서 우수한 성능을 보여줍니다. 이러한 결과는 모델이 더 강력한 의미 표현 능력과 일반화 성능을 가진다는 것을 나타냅니다. 이 연구는 텍스트 분류 분야의 알고리즘 최적화에 대한 새로운 아이디어를 제공하며, 응용 가능성과 실용적 가치를 지니고 있습니다.



### Softplus Attention with Re-weighting Boosts Length Extrapolation in Large Language Models (https://arxiv.org/abs/2501.13428)
Comments:
          11 pages and 2 figures

- **What's New**: 이 논문은 전통적인 Softmax 주의 메커니즘의 수치적 불안정성과 긴 토큰 길이에서의 성능 저하 문제를 해결하기 위해 새로운 주의 메커니즘을 제안합니다. 새롭게 개발된 Length Scaled Softplus Attention (LSSA)은 비선형 변환을 Softplus 활성화 함수로 대체하고, 토큰 길이에 따른 동적 길이 스케일 요인을 도입하여 성능을 향상시킵니다. 이 접근법은 특히 긴 시퀀스에 대한 처리 성능을 개선하고, 유의한 주의 가중치를 증폭시키며 약한 가중치를 줄이는 재조정 메커니즘을 통합하여 모델의 집중도를 높입니다.

- **Technical Details**: 논문에서 제안한 LSSA는 쿼리(𝐐), 키(𝐊), 값(𝐕)의 스케일 점곱 주의 메커니즘을 기반으로 하며, 모든 입력이 L×d 형상의 벡터로 주어집니다. 기존의 Softmax 연산 대신 다양한 활성화 함수를 실험한 결과, 비포화 기능의 필요성이 낮아진다는 것을 발견하였습니다. 특히, l1-norm은 성능 유지에 필수적인 요소로 관찰되었으며, Softmax의 수학적 표현이 비선형 변환과 l1-norm으로 분해됩니다.

- **Performance Highlights**: LSSA 메커니즘은 훈련 시퀀스 길이뿐만 아니라 훨씬 긴 시퀀스에서도 표준 주의 메커니즘보다 뛰어난 성능을 보입니다. 특히 16배 긴 훈련 토큰 길이에서도 검증 손실이 거의 일정하게 유지되며 수치적 안정성을 확보했습니다. 실험 결과, 제안된 재조정 메커니즘이 다양한 주의 변형과 통합될 때 성능 개선으로 이어짐을 입증합니다.



### A Survey of Code-switched Arabic NLP: Progress, Challenges, and Future Directions (https://arxiv.org/abs/2501.13419)
Comments:
          Accepted to COLING 2025

- **What's New**: 이번 논문에서는 아랍어에서의 코드 스위칭(Code-Switching) 연구에 대한 문헌 리뷰를 제공하며, 현재의 연구 동향, 도전 과제 및 연구 격차를 분석합니다. 특히, 다양한 아랍 방언과 외국어 간의 코드 스위칭 현상을 다루고, 다국어 사회에서 필요로 하는 언어 기술 개발의 중요성을 강조합니다. 이전 연구와의 차별점은 아랍어 환경에 한정된 더욱 심층적인 통찰력을 제공한다는 점입니다.

- **Technical Details**: 논문은 코드 스위칭의 세 가지 주요 유형인 inter-sentential, extra-sentential, intra-sentential을 설명하며, 아랍어처럼 형태학적으로 풍부한 언어에서 발생하는 morphological 코드 스위칭에 대해서도 다룹니다. 또한, 언어 간의 전환을 연구하기 위해 Google Scholar를 통해 관련 논문을 수집 및 분류하는 과정을 상세히 설명합니다. 이 과정에서는 현대 표준 아랍어(MSA), 방언 아랍어(DA), 외국어 간의 여러 언어 쌍을 구분하여 분석합니다.

- **Performance Highlights**: CSW에 대한 연구는 2014년 이후 아랍어 자연어 처리(NLP) 커뮤니티에서 상당한 주목을 받고 있으며, 매년 평균 12개의 논문이 발표됩니다. 주요 연구는 방언-외국어 및 MSA-DA 간의 상호작용에 중점을 두고 있으며, 특히 이집트 아랍어와 영어의 연구가 두드러집니다. 그러나 코드 스위칭 데이터를 통한 연구는 여전히 발전 중이며, 방법론의 채택에서 약간의 지연이 관찰됩니다.



### ExLM: Rethinking the Impact of $\texttt{[MASK]}$ Tokens in Masked Language Models (https://arxiv.org/abs/2501.13397)
Comments:
          29 pages, 12 figures

- **What's New**: 본 논문은 Masked Language Models (MLMs)의 훈련에서 [MASK] 토큰의 영향과 그로 인해 발생하는 corrupted semantics 문제에 대해 탐구합니다. 특히, 기존 MLM 모델의 한계를 극복하기 위해 새로운 Enhanced-context MLM인 ExLM을 제안합니다. ExLM은 입력 맥락에서 [MASK] 토큰을 확장하여 모델의 의미 정보를 캡처하고 corrupted semantics 문제를 완화하는데 초점을 맞추고 있습니다.

- **Technical Details**: ExLM은 기존의 MLM 프레임워크를 기반으로 하여 [MASK] 토큰을 입력 맥락에서 증가시킵니다. 이는 모델이 더 많은 컨텍스트 용량을 활용할 수 있게 하고, 다양한 의미적 의존성을 모델링할 수 있는 능력을 부여합니다. 실험 결과, ExLM은 BERT 기반 모델과 비교하여 텍스트 모델링 및 SMILES 모델링 작업에서 개선된 성능을 보여주었습니다.

- **Performance Highlights**: ExLM은 기존의 MLM보다 의미적 표현 능력을 향상시켰으며, 다운스트림 작업에서도 뛰어난 성능을 입증했습니다. 실험 분석을 통해 ExLM은 맥락 증강을 통해 MLM에서 자주 나타나는 다중 의미 문제를 효과적으로 감소시켰습니다. 이는 MLM의 향후 발전을 위한 중요한 기초 자료로 작용할 것으로 기대됩니다.



### Can Large Language Models Understand Preferences in Personalized Recommendation? (https://arxiv.org/abs/2501.13391)
- **What's New**: LLM 기반 추천 시스템의 평가를 개선하기 위해 PerRecBench라는 새로운 벤치를 소개했습니다. 이 벤치는 사용자 평가의 편향(user rating bias)과 아이템 품질(item quality)의 영향을 제거하고 개인적인 선호를 보다 정확히 반영하는데 중점을 둡니다. 기존의 평점 기반 평가 방식 대신 그룹 내 사용자 순위를 평가함으로써 더욱 정교한 맞춤형 추천을 가능하게 하였습니다.

- **Technical Details**: PerRecBench는 사용자 선호도를 기반으로 한 그룹 순위 평가 프레임워크를 사용하여 추천 모델의 성능을 측정합니다. 여기에는 pointwise, pairwise, listwise와 같은 다양한 순위 방법이 포함되어 있으며, 각 사용자의 개별 프로필과 이력을 반영한 출력이 기대됩니다. 특정 시간 내에 동일한 아이템을 구매한 사용자를 그룹으로 묶어, 상대 평정(relative rating)을 통한 개인화된 평가가 시행됩니다.

- **Performance Highlights**: 19개의 LLM을 평가한 결과, 일반적으로 큰 모델이 작은 모델보다 좋은 성능을 보였지만 개인화된 추천에서 여전히 한계를 보였습니다. PerRecBench의 결과는 MAE/RMSE와 낮은 상관관계를 보여, 전통적인 평점 회귀 작업과 개인화가 본질적으로 다르다는 점을 확인했습니다. 또한, 단일 형식 훈련(single-format training)을 통한 가중치 병합(weight merging)이 성능 개선에 유망하다는 것을 발견하였고, LLM의 사용자 선호 이해를 개선하는 것이 여전히 해결해야 할 연구 문제로 남아있다고 강조합니다.



### Do as We Do, Not as You Think: the Conformity of Large Language Models (https://arxiv.org/abs/2501.13381)
Comments:
          ICLR 2025. Code: this https URL

- **What's New**: 최근 대규모 언어 모델(LLMs)의 발전은 복잡한 문제를 해결할 수 있는 협업 다중 에이전트 시스템을 가능하게 하며, 이러한 시스템에서의 일치 현상(conformity)에 대한 연구를 제시합니다. 이 논문에서는 새로운 기준인 BenchForm을 도입하여 LLM-driven 다중 에이전트 시스템에서의 일치 현상과 그 영향을 조사합니다. 또한 일치 현상을 완화하기 위한 두 가지 전략인 향상된 페르소나(personas) 개발과 반영 메커니즘(reflection mechanism)을 탐구합니다.

- **Technical Details**: BenchForm은 LLM의 행동을 탐색하기 위해 설계된 다섯 가지 상호작용 프로토콜과 추론 집약적인 작업으로 구성된 새로운 기준입니다. 이 연구에서 LLM의 일치율(conformity rate)과 독립성 비율(independence rate)이라는 두 가지 측정 지표를 사용하여 일치 현상의 영향을 정량적으로 평가합니다. 또한, 상호작용 시간과 다수 집단의 크기와 같은 일치에 영향을 미치는 요인을 분석하고, 피험자 에이전트가 자신의 일치 행동을 어떻게 합리화하는지를 조사합니다.

- **Performance Highlights**: BenchForm을 사용해 수행한 여러 실험 결과에서 LLM의 일치 행동에 대한 흥미로운 발견이 있었습니다. 특히, 에이전트가 올바른 판단을 내리기보다 다수의 의견에 따르는 경우가 종종 발생함을 보여주었습니다. 이 연구는 협업 AI 시스템의 신뢰성에 대한 우려를 제기하며, 윤리적으로 정렬된 강력한 AI 시스템 개발을 위한 방향을 제시합니다.



### Watching the AI Watchdogs: A Fairness and Robustness Analysis of AI Safety Moderation Classifiers (https://arxiv.org/abs/2501.13302)
Comments:
          Accepted to NAACL 2025 Main Conference

- **What's New**: 본 논문에서는 AI Safety Moderation (ASM) 분류기의 공정성과 강건성을 평가합니다. 특히 OpenAI Moderation API, Perspective API, Google Cloud Natural Language (GCNL) API, 그리고 Clarifai API를 대상으로 연구를 진행하며, 이 모델들이 소수 집단의 콘텐츠를 불공정하게 분류할 위험성을 분석합니다. 이러한 분석은 ASM 모델의 향후 개선을 위한 중요성을 강조합니다.

- **Technical Details**: 연구에서는 ASM 모델의 공정성을 평가하기 위해 Demographic Parity (DP)와 Conditional Statistical Parity (CSP) 같은 메트릭을 사용합니다. 또한, 자연스러운 입력 변형에 대한 분류기의 민감도를 테스트하여 모델의 강건성을 평가합니다. 텍스트 입력을 최소한으로 perturbation하여 발생하는 분류 오류를 측정하는 방법론도 채택합니다.

- **Performance Highlights**: 연구 결과, OpenAI의 ASM 모델이 다른 모델들에 비해 더 불공정하다는 것이 밝혀졌습니다. 또한, 모델들이 입력의 최소한의 변형에 대해 강건하지 않다는 점도 확인되었습니다. 이는 안전하지 않은 댓글이 ASM 모델을 우회하여 통과할 수 있음을 시사하며, 향후 모델 개선을 위한 통찰을 제공합니다.



### Hypothesis Generation for Materials Discovery and Design Using Goal-Driven and Constraint-Guided LLM Agents (https://arxiv.org/abs/2501.13299)
Comments:
          Accepted in NAACL 2025

- **What's New**: 이 연구는 대규모 언어 모델(LLM)을 활용하여 응용 프로그램에 맞는 소재를 신속하게 발견하고 설계하는 방법을 제안합니다. 이 과정에서 개발된 MatDesign 데이터셋은 2024년에 발표된 최신 연구 논문들을 기반으로 하여, 실제 목표와 제약 조건을 갖춘 실용적인 자료를 포함하고 있습니다. LLM을 통해 가설을 생성하고 평가하는 새로운 시스템 AccelMat을 개발하였으며, 이는 기존 방법에 비해 유연성과 접근성을 향상시킵니다. 또한, 연구진은 생성된 가설의 평가를 위한 새로운 메트릭스를 도입하여 신뢰성을 높였습니다.

- **Technical Details**: 기존의 소재 발견 방법은 시간과 자원이 많이 소모됩니다. 이에 비해 LLM을 기반으로 한 AccelMat 프레임워크는 하이포세스 생성 에이전트, 다중 LLM 비평 시스템, 피드백을 통합하는 요약 에이전트 및 가설을 평가하는 평가 에이전트를 포함하게 설계되었습니다. MatDesign 데이터셋은 2024년에 발표된 50개의 연구 논문에서 생성되었으며, 가설 평가에 있어 동시성 및 품질을 측정하는 새로운 평가 메트릭스가 도입되었습니다.

- **Performance Highlights**: 이 연구는 소재 발견 및 설계에 대한 새로운 접근 방식을 제공하여, 실제 적용 가능한 가설을 생성하는 데 있어 LLM의 잠재력을 극대화합니다. 제안된 시스템은 기존 벤치마크들이 가지는 한계를 극복하고, LLM의 혁신성 및 품질을 보다 정확하게 평가할 수 있는 기반을 마련합니다. MatDesign은 실제 산업 응용 문제를 해결하기 위한 새로운 기준을 제공하며, LLM 기반의 에이전트가 생성한 가설의 정밀도를 더욱 향상시킬 것으로 기대됩니다.



### RAMQA: A Unified Framework for Retrieval-Augmented Multi-Modal Question Answering (https://arxiv.org/abs/2501.13297)
Comments:
          Accepted by NAACL 2025 Findings

- **What's New**: 이 논문에서는 텍스트와 이미지를 통합한 Multi-modal retrieval-augmented Question Answering (MRAQA) 분야에서 새로운 접근법인 RAMQA를 제안합니다. RAMQA는 전통적인 learning-to-rank 방법과 generative ranking 기술을 결합하여, 현대의 대형 생성 언어 모델(LLMs)을 활용한 정보 검색의 한계를 극복하고자 합니다. 이를 통해 두 가지 MRAQA 벤치마크인 WebQA와 MultiModalQA에서 성능 향상을 입증하였습니다.

- **Technical Details**: RAMQA는 LLaVA를 기반으로 하여 multi-modal pointwise ranker를 훈련한 후, novel autoregressive multi-task learning 접근법을 채택하여 LLaMA 모델을 상위 k개 문서의 재정렬에 사용합니다. 이 과정에서는 zero-shot LLaVA 모델을 이용하여 다중 모달 문서를 텍스트 표현으로 통합하고, permutation 기법을 활용하여 문서 후보군의 다양성을 증가시켜 bias를 감소시키는 방법을 사용합니다.

- **Performance Highlights**: 실험 결과, WebQA와 MultimodalQA 두 벤치마크에서 강력한 기준선에 비해 유의미한 성능 향상을 달성하였으며, RAMQA는 웹 기반의 QA 시스템에서 네 번째 순위를 기록했습니다. 이 연구는 multi-modal generative LLMs의 활용 가능성을 보여주며, 점진적인 재정렬이 정보 검색에서 더 효율적인 처리를 가능하게 함을 시사합니다.



### Automatic Fact-Checking with Frame-Semantics (https://arxiv.org/abs/2501.13288)
- **What's New**: 본 논문에서는 프레임 의미(Frame Semantics)를 활용하여 자동 사실 확인(Automatic Fact-Checking)의 새로운 패러다임을 제안합니다. 이 접근 방식을 지원하기 위해 PolitiFact에서 추출한 실제 청구를 기반으로 한 초기 데이터 세트를 새롭게 소개하며, 대규모 구조화 데이터에 맞추어 주석이 달려 있습니다. 이러한 연구는 잘못된 정보가 만연한 정보를 이해하는 데 도움을 줄 것으로 기대되며, 자동 사실 확인의 능력을 향상시킵니다.

- **Technical Details**: 제안하는 연구에서 프레임 의미는 언어가 어떻게 구조화된 표현을 통해 의미를 인코딩하는지를 탐구하는 언어적 프레임워크로서, 주장에 대한 컨텍스트와 의도를 이해하는 데 도움을 줍니다. 연구에서는 두 가지 사례 연구를 수행했으며, 첫 번째 연구는 투표 관련 청구를 조사하고 Vote semantic frame을 활용하였고, 두 번째 연구는 OECD의 다양한 데이터 소스를 비교분석했습니다. 이 연구 결과는 프레임 의미 활용의 효과성을 입증하며, 유사한 청구를 가진 사항에서 관련 증거를 효율적으로 찾는 데 기여했습니다.

- **Performance Highlights**: 연구 결과, 투표 청구의 경우 전체 청구 대신 청구에서 추출된 프레임 요소를 사용함으로써 recall@10에서 2.1 포인트 향상되었습니다. OECD 청구의 경우, 관련 OECD 테이블을 식별할 때 프레임 요소를 활용한 결과 recall@5에서 7.3 포인트 증가한 것으로 관찰되었습니다. 이러한 개선 사항들은 프레임 의미가 구조화된 데이터 탐색을 향상시킬 수 있음을 보여줍니다.



### RAG-Reward: Optimizing RAG with Reward Modeling and RLHF (https://arxiv.org/abs/2501.13264)
Comments:
          Preprint, work in progress

- **What's New**: 본 논문은 Retrieval-augmented generation (RAG) 시스템의 최적화를 위한 보상 모델을 도입하여, RAG의 정확성과 신뢰성을 향상시키는 새로운 데이터셋인 RAG-Reward를 소개합니다. 이 연구는 기존 LLM의 한계를 극복하고, 정보 집약적인 문제에 효과적으로 대응할 수 있는 솔루션을 제공합니다. 보상 모델을 사용할 때 LLM의 효과성을 높이기 위한 강화 학습 방식인 RLHF를 적용하여, RAG 파이프라인의 이점을 극대화할 수 있습니다.

- **Technical Details**: RAG-Reward는 hallucination(허위 생성), comprehensiveness(포괄성), verbosity(장황함), attribution(출처 명시)라는 네 가지 주요 메트릭을 기준으로 생성 품질을 평가합니다. 여러 LLM으로부터 다양하게 샘플링한 응답으로 35K개의 고품질 훈련 샘플을 수집하였고, 이를 통해 보상 모델을 훈련하였습니다. 또한 RAFT 알고리즘을 활용하여 정책 모델을 개발하고 성능 향상을 이루었습니다.

- **Performance Highlights**: 실험 결과, 개발한 보상 모델은 검증 세트에서 80% 이상의 정확성을 달성하며, 향상된 정책 모델은 RAG 시스템의 전체 성능 향상에 기여함을 입증했습니다. 이러한 결과는 RAG-Reward 데이터셋의 품질과 논문의 접근 방식의 효과를 보여줍니다. RAG 시스템의 신뢰성과 생성을 개선할 수 있는 가능성을 제시하며, 향후 연구의 중요한 기초 자료로 활용될 것입니다.



### Preference Curriculum: LLMs Should Always Be Pretrained on Their Preferred Data (https://arxiv.org/abs/2501.13126)
Comments:
          18 pages, 13 figures

- **What's New**: 최근의 대형 언어 모델(LLM)은 일관된 데이터 분포를 사용하여 사전 훈련을 진행해왔지만, 모델의 성능이 향상됨에 따라 차별화된 데이터로 훈련하는 것이 직관적입니다. 본 논문에서는 Perplexity Difference 기반의 선호 커리큘럼 학습(Preference Curriculum learning, PDPC) 프레임워크를 통해 LLM이 선호하는 데이터를 인식하고 이를 활용하여 훈련하는 방법을 제안합니다. PDPC는 LLM의 발전하는 데이터 선호도를 반영하여 사전 훈련 과정에서의 효율성을 극대화하는 데 중점을 두고 있습니다.

- **Technical Details**: PDPC 프레임워크에서는 먼저 PD(Perplexity Difference) 메트릭을 도입하여 강한 모델과 약한 모델이 샘플에 얼마나 잘 맞는지를 측정합니다. PD 값이 높은 샘플은 약한 모델이 학습하기 어려운 반면 강한 모델에게는 더 적합한 특징을 가집니다. 이 프레임워크는 오프라인에서의 처리 방법을 통해 동적인 선호도 조정을 근사하여 훈련 중단 없이 연속적으로 LLM을 학습시킬 수 있도록 설계되었습니다.

- **Performance Highlights**: 1.3B 및 3B 모델에서의 실험 결과, 기준선에 비해 PDPC가 현저한 성능 향상을 보여주었습니다. 특히, 3B 모델의 경우, 평균 정확도가 다양한 벤치마크에서 4.1% 이상 증가한 것으로 나타났습니다. 이러한 결과는 PDPC가 데이터 선호도를 효율적으로 활용하여 LLM의 학습 성능을 개선하는 데 효과적임을 보여줍니다.



### Generating Plausible Distractors for Multiple-Choice Questions via Student Choice Prediction (https://arxiv.org/abs/2501.13125)
- **What's New**: 이번 연구에서는 교육 분야에서 학생들의 오해(misconceptions)와 지식 격차를 파악하고 이해도를 정확하게 평가하기 위한 MCQ(다지선다형 질문)에서 더 그럴듯한 방해 선택지(distractor)를 생성하는 모델 훈련 파이프라인을 제안합니다. 먼저 학생들의 오해를 고려하여 두 개의 방해 선택지의 상대적 그럴듯함을 평가하는 쌍별(rank-based) 랭커(pairwise ranker)를 훈련합니다. 이후, 이 모델을 활용하여 쌍별 방해 선택지 랭크 데이터셋을 생성하고, Direct Preference Optimization(DPO)을 통해 더 그럴듯한 방해 선택지를 생성하는 방해 선택지 생성기를 훈련합니다.

- **Technical Details**: 모델 훈련을 위해, South Korea의 온라인 학습 플랫폼에서 교육자들이 생성한 MCQ 데이터세트를 사용했습니다. 데이터세트에는 파이썬, 데이터베이스(SQL) 및 머신러닝 및 딥러닝(Machine Learning & Deep Learning) 관련 질문이 포함되어 있으며, 각 질문의 학생 선택률 정보가 포함되어 있습니다. 이를 통해 우리는 학생들이 어떤 방해 선택지를 더 혼란스럽게 선택하는지 평가할 수 있습니다.

- **Performance Highlights**: 실험 결과, 쌍별 랭커는 학생들의 일반적인 오해를 효과적으로 파악하며, 인간 전문가와 유사한 랭킹 정확도를 달성했습니다. 또한 방해 선택지 생성기는 여러 기준선 모델보다 우수한 성과를 보이며, 더 그럴듯한 방해 선택지를 생성하고, 높은 항목 차별지수(item discrimination index, DI)를 산출했습니다.



### Debate Helps Weak-to-Strong Generalization (https://arxiv.org/abs/2501.13124)
Comments:
          AAAI2025 Special Track on AI Alignment (Oral presentation)

- **What's New**: 이번 연구에서는 AI 정렬(alignment) 기술의 한계를 극복하기 위해 인간의 감독(supervision) 발전과 강력한 사전 학습(pretrained) 모델의 활용을 결합하는 방법을 제시합니다. 특히, 약한(weak) 인간 감독을 통해 강력한 모델을 훈련시키고 그 모델이 생성한 레이블을 통해 다시 약한 모델을 훈련시키는 순환적 방식에 초점을 맞추고 있습니다. 이러한 접근 방식은 오픈AI의 약한-강한 NLP 벤치마크에서 정렬 성능을 향상시키는 데 기여합니다.

- **Technical Details**: 연구에서는 약한 모델이 신뢰할 수 없는 강력한 모델로부터 유용한 정보를 추출할 수 있도록 돕는 논쟁(debate) 기법을 도입했습니다. 두 개의 강력한 모델이 서로 대결하는 사례에서 발생하는 논증을 통해 약한 모델이 더 신뢰할 수 있는 정보를 획득하도록 합니다. 또한, 다양한 약한 모델의 집합(ensemble)을 사용하여 논쟁에서 생성된 긴 인수(arguments)를 최대한 활용하고, 단일 모델보다 더 견고한 감독 예측을 도출하였습니다.

- **Performance Highlights**: 결과적으로, 약한 모델의 앙상블이 단일 약한 모델 및 세밀한 조정(finetune) 앙상블보다 일관되게 우수한 성과를 보였습니다. 다양한 샘플링(seed)을 사용하는 논쟁 앙상블의 결과는 더욱 향상된 성능을 나타내어 이는 향후 AI 정렬에 대한 혼합(superhuman) 접근 방법의 연구 가능성을 보여줍니다. 본 연구는 논쟁이 약한-강한 일반화에 도움이 된다는 실증적 증거를 제시하며, AI 정렬 분야에 중대한 기여를 한 것으로 평가됩니다.



### Zero-Shot Verification-guided Chain of Thoughts (https://arxiv.org/abs/2501.13122)
- **What's New**: 이번 연구에서는 LLM(대형 언어 모델)의 COT(Chain-of-Thought) 프롬프트를 사용한 제로샷(self-verification) 자기 검증 접근 방식을 제안합니다. 기존의 방법들은 주로 미세 조정된 검증기나 다양하게 수작업으로 작성된 몇 가지 사례에 의존했으나, 본 연구는 어떠한 수동 선택 예제도 없이 LLM이 생성한 추론 단계를 스스로 검증하는 방법을 중심으로 하였습니다. 이를 위해 COT STEP이라는 새로운 제로샷 프롬프트를 설계하였으며, 이는 추론 단계를 분해하는 데 도움을 줍니다.

- **Technical Details**: 본 연구에서는 SOLAR와 Phi3라는 두 개의 오픈 라이선스 모델을 기반으로 실험을 진행하고, 다양한 제로샷 프롬프트 전략을 평가하였습니다. 네 가지 주요 제로샷 프롬프트 방법은 각각 기본(P1), COT(P2), PS+(P3), TAB COT(P4)으로 정의됩니다. 특히, TAB COT 프롬프트는 표 형식으로 결론을 도출하는 과정을 유도하여 추론 단계를 명확히 식별할 수 있도록 합니다.

- **Performance Highlights**: 이 연구의 결과는 COT STEP 프롬프트가 다른 프롬프트와 경쟁력을 유지하며 자동 단계 분해를 가능하게 한다는 것을 보여줍니다. LLM 기반의 검증기가 수학적 추론 단계를 검증하는 데 상당히 효과적이며, 검증 점수를 사용하여 올바른 답변을 선택하거나 실시간으로 추론을 안내하는 데 이점이 있음을 확인하였습니다. 그러나 자체 일관성을 사용할 경우 이점이 사라지는 경향이 나타났습니다.



### Episodic Memories Generation and Evaluation Benchmark for Large Language Models (https://arxiv.org/abs/2501.13121)
- **What's New**: 이 논문에서는 에피소드 메모리(episodic memory)에 대한 LLMs의 한계를 극복하고, 그러한 메모리 능력을 통합하기 위한 포괄적인 프레임워크를 제안합니다. 기존의 LLM은 사실과 일치하지 않는 정보를 생성하는 할루시네이션(hallucination) 문제를 겪고 있으며, 이로 인해 인간과 유사한 사고를 발전시키기 위한 잠재력이 제한되고 있습니다. 또한, 연구자들은 에피소드 메모리를 충분히 평가하지 않았으며, 논문에서 제안한 새로운 벤치마크가 이러한 격차를 해소할 수 있을 것으로 기대하고 있습니다.

- **Technical Details**: 이 연구는 인지 과학에서 영감을 받아 에피소드 메모리를 구조적으로 모델링하는 접근법을 개발했습니다. 이 프레임워크는 시간적 및 공간적 맥락, 관련 엔티티(entity) 및 사건에 대한 상세한 설명을 포함합니다. 저자들은 LLM 성능 평가를 위한 독창적인 에피소드 메모리 벤치마크를 창출하였으며, 이를 통해 다양한 회상(recall) 및 에피소드 추론(tasks) 작업에서 평가할 수 있도록 오픈 소스 코드와 데이터셋을 공개했습니다.

- **Performance Highlights**: 상위 모델인 GPT-4 및 Claude 변형, Llama 3.1과 o1-mini를 평가한 결과, 가장 진보된 LLM조차 복잡한 공간적-시간적 관계(spatio-temporal relationships)를 다루는 에피소드 메모리 작업에서 어려움을 겪고 있음을 보여줍니다. 특히, 관련된 여러 사건을 처리할 때 이러한 어려움이 더욱 두드러졌으며, 10k-100k 토큰의 짧은 문맥(context)에서도 이 문제가 나타났습니다. 이 연구는 LLM의 성능 개선을 위해 에피소드 메모리 벤치마크의 중요성을 강조합니다.



### Multilinguality in LLM-Designed Reward Functions for Restless Bandits: Effects on Task Performance and Fairness (https://arxiv.org/abs/2501.13120)
Comments:
          Accepted at the AAAI-2025 Deployable AI Workshop

- **What's New**: 이 논문은 Restless Multi-Armed Bandits (RMABs) 알고리즘에 비영어 명령어를 사용할 때의 작업 성능과 공정성에 미치는 영향을 연구합니다. 특히, 저자들은 리소스가 부족한 언어를 포함한 여러 언어로 번역된 다양한 복잡도의 프롬프트를 조사하였습니다. 그 결과 영어 프롬프트가 작업 성능에 있어 유리함을 보이며, 공정성 측면에서도 리소스가 부족한 언어와 복잡한 프롬프트가 불공정성을 일으킬 가능성이 높다는 사실을 밝힙니다.

- **Technical Details**: 실험은 DLM 알고리즘(Behari et al. 2024)을 이용하여, 영어, 힌디어, 타밀어 및 투루어(저자원 언어)로 다양한 프롬프트를 실행하는 방식으로 진행됩니다. 이 연구는 6개의 피쳐를 기반으로 하는 합성 환경을 사용하여, 각 언어로 제안된 보상 함수의 효과를 조사합니다. 또한, Gemini 1.0 Pro를 LLM으로 활용하며, Whittle Index 기반의 해결책을 리인포스먼트 러닝 부분에 적용합니다.

- **Performance Highlights**: 영어로 제안된 LLM 보상 함수가 다른 언어에 비해 훨씬 효과적이라는 결과를 도출하였습니다. 프롬프트의 정확한 표현 방식은 작업 성능에 영향을 미치며, 프롬프트의 복잡성이 증가할수록 모든 언어에서 성능이 저하되는 경향을 보입니다. 그러나 영어 프롬프트에 비해 리소스가 부족한 언어의 경우 성능 저하가 더욱 두드러지는 것으로 확인되었습니다.



### MyGO Multiplex CoT: A Method for Self-Reflection in Large Language Models via Double Chain of Thought Thinking (https://arxiv.org/abs/2501.13117)
- **What's New**: 최근 대형 언어 모델(LLMs)의 발전이 여러 가지 추론 및 의사결정 작업에서 그들의 인상적인 능력을 입증했습니다. 그러나 이러한 모델의 추론 과정의 품질과 일관성은 여전히 개선의 여지가 있으며, 이를 위해 자기 검토 및 자기 반성의 향상이 필요합니다. 본 논문에서는 Double Chain of Thought (CoT) 사고 방식을 통해 LLM이 자기 검토를 시뮬레이션할 수 있도록 하는 방법인 Multiplex CoT를 소개합니다.

- **Technical Details**: Multiplex CoT는 반복적 사고를 활용하는 방법으로, 모델이 초기 Chain of Thought를 생성한 후, 이를 비판하고 수정하는 두 번째 사고 생성 단계를 진행합니다. 이 재귀적 접근 방식은 더 일관되고 논리적이며 견고한 답변을 제공하여 결정 프로세스를 개선합니다. 이 방법은 간단한 prompt engineering을 통해 기존 LLM 구조에서 효과적으로 구현될 수 있으며, 추가 훈련이 필요하지 않습니다.

- **Performance Highlights**: Multiplex CoT는 효과적으로 초기 사고의 일관성을 검토하고 수정하여 더 높은 수준의 논리적 일관성을 성취합니다. 특히, 모델의 두 번째 추론 단계에서는 초기 추론의 결점을 식별하고 이를 교정하여 최종 답변을 보다 정확하게 정제합니다. 이 접근 방식은 오류 수정률을 개선하는 데 중요한 역할을 하여 전체 추론 품질의 향상을 가져옵니다.



### Dagger Behind Smile: Fool LLMs with a Happy Ending Story (https://arxiv.org/abs/2501.13115)
- **What's New**: 본 논문은 기존의 jailbreak 공격 방법론과는 다르게, LLM(대형 언어 모델)이 긍정적인 프롬프트에 더 잘 반응한다는 새로운 관점을 제공합니다. 이를 바탕으로 Happy Ending Attack (HEA)라는 새로운 공격 방법을 제안하며, 악의적인 요청을 긍정적인 시나리오 템플릿 안에 감싸서 LLM을 속이는 방안을 제시합니다. HEA는 단 2단계의 간단한 절차로 LLM을 jailbreak 하는 데 성공적으로 작용하였습니다.

- **Technical Details**: HEA는 부정적인 질문에 대해 LLM이 기본적으로 반응을 거부하는 특성을 활용하여, 긍정적인 내용을 담은 스토리에 악의적인 요청을 포함시키는 방식으로 디자인됩니다. 공격 과정은 프롬프트 디자인에서 시작하여, 마지막 질문을 통해 더 구체적이고 조직화된 jailbreak 응답을 얻습니다. 이 과정은 전혀 인간의 개입 없이 완전 자동화됩니다.

- **Performance Highlights**: HEA는 최신 LLM에 대해 평균 88.79%의 공격 성공률(ASR)을 달성했습니다. HEA는 고도 상용 및 오픈 소스를 포함한 여러 LLM에서 성능을 테스트하였으며, 감정 분류 및 중요도 히트맵을 이용해 성공적인 이유에 대한 정량적 설명을 제공했습니다. 이러한 발견은 LLM의 안전성 향상을 위한 새로운 연구 아이디어를 제공할 수 있습니다.



### Can We Generate Images with CoT? Let's Verify and Reinforce Image Generation Step by Step (https://arxiv.org/abs/2501.13926)
Comments:
          Journal Version. Code and models are released at this https URL

- **What's New**: 이 논문은 Chain-of-Thought (CoT) 추론을 이미지 생성 시나리오에서 검증하고 강화하는 잠재력을 탐구한 최초의 포괄적인 연구입니다. 여기서는 테스트 시간 컴퓨테이션을 검증하고 모델 선호도를 Direct Preference Optimization (DPO)로 정렬하는 세 가지 기술을 집중적으로 다룹니다. 이 연구는 CoT 추론 기술을 오토 회귀 이미지 생성에 성공적으로 적용할 수 있음을 보여줍니다.

- **Technical Details**: 연구에서는 Show-o라는 최신 이산 생성 모델을 기준으로 사용하며, GenEval이라는 텍스트-이미지 생성 벤치마크에서 성능을 평가합니다. 특히 Outcome/Process Reward Model (ORM/PRM)과 같은 검증 기법 및 DPO를 통한 강화를 통해 모델의 성능을 최적화하는 방법을 살펴봅니다. CoT 추론이 오토 회귀 이미지 생성에 얼마나 효과적으로 적용될 수 있는지를 분석하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 연구 결과는 CoT 추론 전략을 이미지 생성 시나리오에 효과적으로 적용할 수 있는 가능성을 보여줍니다. PARM 및 PARM++라는 두 가지 새로운 보상 모델을 도입하여, 오토 회귀 이미지 생성의 품질을 크게 향상시키며, GenEval에서 기준 모델을 +24% 개선했습니다. 또한, PARM++는 생성된 이미지의 질을 반영 서사 방식으로 세밀하게 조정하여, PARM보다도 +4% 더 나은 성능을 기록했습니다.



### IMAGINE-E: Image Generation Intelligence Evaluation of State-of-the-art Text-to-Image Models (https://arxiv.org/abs/2501.13920)
Comments:
          75 pages, 73 figures, Evaluation scripts: this https URL

- **What's New**: 최근의 확산 모델(diffusion models) 발전으로, 텍스트-이미지(text-to-image, T2I) 모델들이 놀라운 성과를 보여주고 있습니다. 특히 FLUX.1과 Ideogram2.0 같은 신규 모델들은 복잡한 작업에서 뛰어난 성능을 발휘하며, 이러한 모델들이 일반적인 용도로 활용될 가능성에 대한 논의가 이어지고 있습니다. 이 연구는 T2I 모델들이 전통적인 이미지 생성뿐만 아니라 이미지 편집, 비디오, 오디오, 3D 및 움직임 생성과 같은 다양한 분야에서 가진 능력을 탐구합니다.

- **Technical Details**: 우리는 T2I 모델들의 종합적인 평가를 위해 IMAGINE-E라는 새로운 평가 프레임워크를 개발했습니다. 평가 대상 모델로는 FLUX.1, Ideogram2.0, Midjourney, Dall-E3, Stable Diffusion 3, Jimeng 등을 선정하였으며, 다섯 가지 주요 도메인으로 평가를 나누었습니다. 이 도메인들은 구조화된 출력 생성, 사실성(factuality) 및 물리적 일관성(physical consistency), 특정 도메인 생성, 도전적인 시나리오 생성, 다중 스타일 생성 과제를 포함합니다.

- **Performance Highlights**: 이 모델들의 성능을 평가한 결과, 특히 FLUX.1과 Ideogram2.0이 구조화된 및 특정 도메인 작업에서 두드러진 성과를 보였습니다. 이러한 평가 결과는 T2I 모델들이 AI 툴로서의 가능성과 응용을 확대하고 있음을 시사합니다. 연구 결과는 T2I 모델들의 현재 상태와 미래 방향성을 제시하며, 일반적인 사용성을 향한 진화 과정을 설명합니다.



### Temporal Preference Optimization for Long-Form Video Understanding (https://arxiv.org/abs/2501.13919)
- **What's New**: 이 논문은 기존 비디오 대형 멀티모달 모델(video-LMM)의 시간적 이해(temporal grounding) 성능을 향상시키기 위한 새로운 프레임워크인 Temporal Preference Optimization (TPO)을 제안합니다. TPO는 두 가지 세분화된 데이터셋, 즉 특정 비디오 구간에 집중하는 localized temporal grounding과 전체 비디오 시퀀스의 시간적 의존성을 포착하는 comprehensive temporal grounding을 활용하여 모델의 시간적 반응을 개선합니다. 이 프레임워크는 자가 학습(self-training) 방법론을 통해 잘 정립된 반응과 덜 정확한 반응을 구분하는 데 도움을 줍니다.

- **Technical Details**: 비디오-LMM은 대형 언어 모델(LLM), 비주얼 인코더(visual encoder), 그리고 멀티모달 프로젝터(multimodal projector)를 포함하는 구조로, 입력 비디오 V와 텍스트 시퀀스 x를 처리하여 응답 y의 확률을 모델링합니다. TPO는 Direct Preference Optimization (DPO) 기법을 사용하여 인간의 선호도(preference)를 기반으로 모델의 파라미터를 최적화하며, 이는 명시적인 보상 모델이나 복잡한 강화 학습 알고리즘을 필요로 하지 않습니다. 이 과정에서 TPO는 비디오-LMM의 시간적 반응 능력을 강화하는데 중점을 두고 있습니다.

- **Performance Highlights**: TPO를 적용한 실험 결과, LongVideoBench, MLVU, Video-MME와 같은 세 가지 비디오 이해 벤치마크에서 각각 2.9%, 3.1%, 2.5%의 성능 향상을 달성했습니다. 특히, LLaVA-Video-TPO 모델은 Video-MME 벤치마크에서 7B 모델 중 최고의 성능을 기록하였으며, 이는 TPO의 확장성과 효율성을 강조합니다. 이러한 성과는 장기 비디오 이해(task) 및 시간적 추론(temporal reasoning)을 향상시키는 데 중요한 기초로 작용할 것으로 기대됩니다.



### On the Reasoning Capacity of AI Models and How to Quantify I (https://arxiv.org/abs/2501.13833)
- **What's New**: 최근 대규모 언어 모델(LLMs)의 발전은 이들의 추론 능력의 본질에 대한 논쟁을 가열시키고 있습니다. GPQA, MMLU와 같은 벤치마크에서 뛰어난 성능을 보이는 이 모델들은 보다 복잡한 추론 작업에서 한계를 드러내고 있어 보다 엄격한 평가 방법론의 필요성이 강조되고 있습니다. 본 연구는 전통적인 정확성 지표를 넘어 모델 행동의 근본적인 메커니즘을 탐구할 수 있는 새로운 현상학적 접근 방식을 제안합니다.

- **Technical Details**: 우리의 접근 방식은 다중 선택 추론 작업에서의 위치 편향(positional bias)을 사례 연구로 사용하여 체계적인 섭동이 모델의 의사 결정의 기본적인 측면을 어떻게 드러낼 수 있는지를 보여줍니다. 두 가지 상호 보완적인 현상학적 모델인 확률 혼합 모델(Probabilistic Mixture Model, PMM)과 정보 이론적 일관성(Information-Theoretic Consistency, ITC) 분석을 개발하였습니다. PMM은 모델 응답을 추론, 기억, 추측의 구성 요소로 분해하고, ITC 분석은 모델 신뢰도와 전략 선택 간의 관계를 정량화합니다.

- **Performance Highlights**: 기존 모델들은 진정한 추론이 여전히 쉽지 않음을 보여주며, 종종 성공은 기억(responsiveness)과 패턴 매칭(pattern matching)의 정교한 조합에 의존하는 경우가 많습니다. 단순한 정확도만으로는 모델의 추론 능력을 과대평가할 수 있으며, 모델 행동은 인지 전략의 위상 공간에서의 기본 메커니즘을 통해 특징 지어질 수 있습니다. 이 프레임워크는 신뢰성 기준을 구체적으로 설정할 수 있는 기준을 제공하여, 전략 분포에 기반한 실제 응용 프로그램의 신뢰성 기준을 명확히 할 수 있게 합니다.



### Video-MMMU: Evaluating Knowledge Acquisition from Multi-Discipline Professional Videos (https://arxiv.org/abs/2501.13826)
- **What's New**: 이 연구에서는 Large Multimodal Models (LMMs)의 비디오로부터의 지식 습득 능력을 평가하기 위한 새로운 벤치마크인 Video-MMMU를 소개합니다. 이 벤치마크는 300개의 전문가 수준 비디오와 900개의 인간 주석 질문으로 구성되어 있으며, 정보 인식, 이해, 적응의 세 가지 인지 단계에 맞춘 질문-답변 쌍을 평가합니다. 또한 비디오 시청 후 성과 개선을 정량화하는 새로운 지식 증가 지표인 Δknowledge를 제안하여 LMMs의 학습 능력을 측정합니다.

- **Technical Details**: Video-MMMU는 교육 비디오로부터 지식을 습득하는 LMMs의 능력을 평가하기 위해 설계된 다중 분야 다중 양식 벤치마크입니다. 이 데이터셋은 6개의 전문 분야(예: 미술, 경영, 의학, 과학, 인문학, 공학)에서 30개 과목을 아우르는 300개의 대학 수준 교육 비디오를 포함하고 있으며, 각 비디오는 인지 단계에 따라 설계된 질문-답변 쌍과 함께 제공됩니다. 특히, 이 벤치마크는 비디오로부터 정보와 지식을 효과적으로 습득하는 LMMs의 성과를 평가하기 위한 체계적인 방법론을 제시합니다.

- **Performance Highlights**: LMMs에 대한 평가 결과, 인지적 요구가 증가함에 따라 성능이 급격히 감소하는 경향을 보였습니다. 인간은 비디오 시청 후 33.1%의 성과 향상을 기록했지만, 가장 성능이 우수한 모델들조차 15.6% 미만의 지식 증가를 보였습니다. 이는 현재 LMMs가 비디오 기반 학습에서의 능력을 효과적으로 습득하기 어려움을 강조하며, 앞으로 이들 모델이 인간 수준으로 학습 능력을 향상시키기 위한 연구가 필요함을 시사합니다.



### Explainable XR: Understanding User Behaviors of XR Environments using LLM-assisted Analytics Framework (https://arxiv.org/abs/2501.13778)
Comments:
          11 pages, 8 figures. This is the author's version of the article that has been accepted for publication in IEEE Transactions on Visualization and Computer Graphics

- **What's New**: Explainable XR는 사용자의 행동을 분석하기 위해 Large Language Models(LLMs)를 활용한 끝-끝(end-to-end) 프레임워크입니다. 이 프레임워크는 증강 현실(AR), 가상 현실(VR), 혼합 현실(MR) 간 전환, 다중 사용자 협업 시나리오 및 다중 모드 데이터를 처리하는 데 있어 기존의 XR 사용자 분석 프레임워크가 직면한 문제를 해결합니다. Explainable XR은 사용자 행동을 이해하고 몰입형 환경에서의 사용자 행동에 대한 다층적이고 실용적인 인사이트를 제공하는 데 필요한 대안 솔루션을 제공합니다.

- **Technical Details**: EXR은 사용자 데이터 기록 스키마인 User Action Descriptor(UAD)를 채택하여 사용자의 다중 모드 행동과 그 의도 및 맥락을 포착합니다. 이를 통해 XR 세션 기록기와 LLM 지원 시각 분석 인터페이스를 제공하고, 분석가의 관점에 맞춘 인사이트를 제공합니다. EXR의 파이프라인은 Unity3D 위에 설계되며, 이로써 모든 XR 환경에서의 데이터 수집, 분석 및 시각화를 손쉽게 수행할 수 있습니다.

- **Performance Highlights**: EXR은 다섯 가지 사용 사례 시나리오를 통해 다양한 XR 응용 프로그램에서 그 유용성을 입증했습니다. 사용자 행동을 깊이 있게 이해할 수 있으며, 다양한 XR 구성 및 가상성 간의 다중 사용자 분석을 위한 복잡한 시각 분석을 관리합니다. 사용자 연구를 통해 EXR이 사용 가능한 분석 솔루션을 제공하고, 몰입형 환경에서의 사용자 행동에 대한 다각적이고 실행 가능한 인사이트를 전달함을 확인했습니다.



### Certified Robustness Under Bounded Levenshtein Distanc (https://arxiv.org/abs/2501.13676)
Comments:
          Accepted in ICLR 2025

- **What's New**: 이 논문에서는 텍스트 분류기의 강건성을 증명하기 위한 새로운 방법을 제안합니다. Lipschitz constant를 Levenshtein distance에 대한 컨볼루션 분류기에 대해 계산할 수 있는 첫 번째 방법으로, 이는 텍스트 도메인에서 효율적인 검증을 가능하게 합니다. Proposed method, LipsLev는 단일 전방 전달 통과에서 분류기의 공인 반지름을 계산할 수 있도록 합니다.

- **Technical Details**: 기존의 검증 방법들은 대개 캐릭터/단어 대체 또는 불용어 제거와 같은 사양만 처리할 수 있었으나, 본 연구에서는 평균 전송 거리(ERP distance)에 대한 Lipschitz constant를 계산함으로써 1-Lipschitz 분류기를 훈련할 수 있는 방법을 제시합니다. 이 방법은 기존의 접근 방식이 Levenshtein distance 제한에 대한 검증을 지원하지 못했던 문제를 해결합니다.

- **Performance Highlights**: LipsLev는 AG-News 데이터 세트에서 거리 1과 2에서 각각 $38.80$% 및 $13.93$%의 공인 정확도를 기록하며, 이는 기존 접근 방식보다도 4배 이상 빠른 속도를 자랑합니다. 또한 거리 2 이상에서 검증할 수 있는 유일한 방법으로, 이는 텍스트 도메인에서의 강건성 검증의 새로운 이정표가 될 것으로 기대됩니다.



### ReasVQA: Advancing VideoQA with Imperfect Reasoning Process (https://arxiv.org/abs/2501.13536)
Comments:
          Accepted to main conference at NAACL 2025; 8 pages;

- **What's New**: 이번 연구에서는 Video Question Answering (VideoQA) 성능 강화를 위한 새로운 접근법인 ReasVQA를 소개합니다. 이는 Multimodal Large Language Models (MLLMs)에 의해 생성된 추론 과정을 활용하여 VideoQA 모델의 품질을 개선합니다. 세 가지 단계인 Reasoning Generation, Reasoning Refinement, Learning from Reasoning을 통해 구현되며, 특히 데이터 필터링을 통해 생성된 추론 과정을 정제하여 모델 학습에 활용합니다.

- **Technical Details**: ReasVQA의 첫 번째 단계인 Reasoning Generation에서 SOTA MLLMs를 사용하여 VideoQA 작업에 대한 추론 과정을 생성합니다. 두 번째 단계에서는 이러한 추론 과정을 정제하기 위해 데이터 필터링 기법을 적용하여 오류를 최소화하고, 마지막 단계에서는 다중 작업 학습(multi-task learning) 프레임워크를 통해 VideoQA 모델이 질문에 대한 답변을 제공하는 동시에 추론 과정을 생성하도록 훈련합니다. 이를 통해 모델은 정확한 질문-답변 관계뿐만 아니라 해당 답변 뒤의 추론 과정을 학습합니다.

- **Performance Highlights**: 실험 결과, ReasVQA는 새로운 최첨단 성능을 달성했습니다. NExT-QA에서 +2.9, STAR에서 +7.3, IntentQA에서 +5.9의 성능 향상을 보이며, 생성된 추론 과정을 VideoQA 모델에 통합하는 것이 효과적임을 입증했습니다. 각 단계의 효과성에 대한 세밀한 분석 또한 진행되어, 추론 정제와 다중 작업 학습이 전반적인 성능 향상에 기여한 바를 확인할 수 있었습니다.



### DQ-Data2vec: Decoupling Quantization for Multilingual Speech Recognition (https://arxiv.org/abs/2501.13497)
Comments:
          Submitted to the IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP)

- **What's New**: DQ-Data2vec는 다국어 자동 음성 인식(ASR)을 위해 언어와 음소 정보를 분리(characters)하는 혁신적인 접근 방식을 제안합니다. 기존의 data2vec 모델이 의존하는 다층 평균화 구조의 약점을 보완하여, K-means quantization 기술을 통해 중요한 언어 및 음소 데이터를 분리합니다. 이 연구는 다국어 ASR 환경에서 더욱 효과적인 성능을 달성하는 데 기여할 것으로 기대됩니다.

- **Technical Details**: DQ-Data2vec는 data2vec 백본 구조를 기반으로 하며, 개선된 온라인 K-means quantizer 두 개를 포함합니다. 이 방법론은 명확하게 클러스터 수를 지정하여 언어와 음소 정보를 분리하는데 중점을 두고 있습니다. 언어 양자화 과정에서, 각 언어의 수에 맞춰 클러스터 수를 지정하여 무관한 정보와 혼합되지 않도록 세분화된 접근 방식을 취합니다.

- **Performance Highlights**: CommonVoice 데이터셋에서 실시된 실험 결과, DQ-Data2vec는 self-supervised 시나리오에서 음소 오류율(PER)을 9.51%, 단어 오류율(WER)을 11.58% 줄이는 성과를 나타냈습니다. 또한, weakly-supervised 환경에서도 각각 18.09%와 1.55% 감소한 성능을 보였습니다. 이런 결과는 DQ-Data2vec의 효율성과 데이터 활용의 효과를 입증합니다.



### MambaQuant: Quantizing the Mamba Family with Variance Aligned Rotation Methods (https://arxiv.org/abs/2501.13484)
- **What's New**: MambaQuant는 Mamba 모델을 위한 최초의 포괄적인 후처리 양자화(PTQ) 프레임워크로, 학습 후 양자화의 효과를 극대화하는 기술을 제시합니다. Mamba 모델의 양자화 과정에서 발생하는 주요 문제로는 큰 이상치(outliers)와 변동성(variance) 불일치가 있습니다. 이러한 문제에 대응하기 위해 MambaQuant는 Karhunen-Loeve 변환(KLT)과 스무스-퓨즈드 회전(rotation) 기법을 도입하여 딥러닝 모델이 손실 없이 양자화될 수 있도록 지원합니다.

- **Technical Details**: MambaQuant는 두 가지 주요 기술을 적용합니다. 첫 번째는 오프라인 모드에서 KLT를 활용한 회전으로, 다양한 채널 분포에 맞춘 회전 행렬(rotation matrix)을 생성합니다. 두 번째는 온라인 모드에서 스무스-퓨즈드 회전을 구현하여 채널 분산을 정규화하여 메모리 비용을 최소화합니다. 이 두 가지 접근법은 양자화 데이터의 최댓값과 분산을 일관되게 유지하여 성능을 향상시킵니다.

- **Performance Highlights**: MambaQuant는 Mamba 기반의 비전 및 언어 작업에서 8비트 양자화 시 1% 미만의 정확도 손실로 우수한 성능을 발휘합니다. 또한, 비전 작업에서는 4비트로 양자화하면서도 1%의 최소 정확도 손실로 뛰어난 결과를 나타냅니다. 기존 방법들과 비교해 MambaQuant는 언어 작업에서도 유의미한 정확도 향상을 보이며, 향후 연구의 기초가 될 것으로 기대됩니다.



### AgentRec: Agent Recommendation Using Sentence Embeddings Aligned to Human Feedback (https://arxiv.org/abs/2501.13333)
Comments:
          10 pages, 8 figures, preprint

- **What's New**: 이번 논문에서는 다중 에이전트 시스템에서 자연어 프롬프트에 대해 가장 적합한 LLM (Large Language Model) 에이전트를 추천하는 새로운 아키텍처를 제안합니다. 기존의 Sentence-BERT (SBERT) 인코더 모델을 확장하여 개발된 이 시스템은 에이전트 분류의 정확도를 높이는데 기여합니다. 특히, 자연어 프롬프트를 문장 임베딩으로 인코딩하여 에이전트 추천과 관련된 의미론적 내용을 포착합니다.

- **Technical Details**: 모델은 문장 임베딩 간의 코사인 유사도를 측정하여 자연어 프롬프트를 분류합니다. 각 분류 작업은 300밀리초 미만의 시간 내에 완료되며, 테스트 데이터에서 92.2%의 상위 1 정확도가 달성되었습니다. 또한, 강화학습을 통해 인간의 가치에 맞춰 조정할 수 있으며, 새로운 클래스에 적응할 수 있는 등 컴퓨팅 비용이 저렴하고 해석 가능성이 높은 구조를 지니고 있습니다.

- **Performance Highlights**: 이 연구는 에이전트 추천을 위한 합성 데이터셋을 생성하여 가능해졌으며, 생성된 데이터셋 및 AgentRec 추천 시스템의 코드는 공개되어 있습니다. 따라서 연구 결과는 실제 응용에 쉽게 활용될 수 있으며, 다중 에이전트 시스템의 효율성을 크게 향상할 것으로 기대됩니다.



### Toyteller: AI-powered Visual Storytelling Through Toy-Playing with Character Symbols (https://arxiv.org/abs/2501.13284)
Comments:
          Accepted to CHI2025

- **What's New**: Toyteller는 사용자가 캐릭터 심볼을 직접 조작하여 텍스트와 비주얼이 혼합된 이야기를 생성할 수 있는 AI 기반의 스토리텔링 시스템입니다. 이를 통해 사용자는 이야기 생성 및 표현을 자연어 외의 다양한 방식으로 수행할 수 있는 새로운 기회를 얻게 됩니다. Toyteller는 캐릭터의 움직임을 입력 모달리티와 출력 형식으로 활용하여, 사용자와 AI 간의 협력적 스토리텔링을 용이하게 합니다.

- **Technical Details**: Toyteller는 캐릭터의 심볼 움직임과 텍스트 생성을 연결하는 공통 의미 공간을 구축하여 모션과 텍스트 간의 상호작용을 가능하게 합니다. 이 시스템은 텍스트와 모션의 생성이 상호 연결된 대화형 AI 모델을 통해 이루어집니다. 기술 평가 결과, Toyteller는 기존의 경쟁 모델인 GPT-4o보다 뛰어난 성능을 보였으며, 여러 측면에서 사용자 경험을 향상시켰습니다.

- **Performance Highlights**: Toyteller는 텍스트와 모션 생성에서 모두 빠른 반응 속도를 보여주며, 최대 7.9배 텍스트 생성 속도 증가와 557.4배 모션 생성 속도 증가를 기록했습니다. 사용자 연구에서는 장난감 놀이 방식의 상호작용이 사용자의 의도를 표현하는 데 유용하지만, 특정한 의도를 명확하게 전달하는 데는 한계가 있음을 발견했습니다. 결과적으로 사용자는 자연어 프롬프트와 장난감 조작을 병행하여 상호 보완적으로 활용하며, Toyteller의 유연성이 다양한 사용자 요구를 지원할 수 있음을 보여주었습니다.



### Academic Case Reports Lack Diversity: Assessing the Presence and Diversity of Sociodemographic and Behavioral Factors related to Post COVID-19 Condition (https://arxiv.org/abs/2501.12538)
- **What's New**: 이 연구는 Post COVID-19 Condition (PCC)에 대한 사회적 건강 결정 요인(SDOH)을 통합하는 포괄적인 프레임워크를 개발하는 것을 목표로 합니다. 문서 내 SDOH의 불균형과 변화를 분석하기 위해 NLP 기술을 활용했습니다. 이를 통해 PCC 사례 보고서에서 26개 핵심 SDOH 관련 엔티티 유형을 식별하고 분석했습니다.

- **Technical Details**: 7,000개 이상의 사례 보고서로 구성된 PCC 사례 보고서 코퍼스를 구축하였고, 709개의 보고서에 대해 사전 훈련된 Named Entity Recognition (NER) 모델을 사용하여 주석을 달았습니다. NER, 자연어 추론(NLI), 3-그램(trigram) 및 빈도 분석을 통합한 NLP 파이프라인이 개발되어 이러한 엔티티를 추출하고 분석했습니다. 특히, 인코더 전용 BERT 모델이 전통적인 RNN 모델보다 더 나은 일반화를 보여주었습니다.

- **Performance Highlights**: 탐색적 분석을 통해 엔티티의 풍부함에 변동성이 있으며, condition, age, care access와 같은 일반적인 엔티티가 많이 나타났습니다. 그러나 race 및 housing status와 같은 민감한 카테고리는 저조한 대표성을 보였습니다. NLI 분석에서는 'Experienced violence or abuse'와 'Has medical insurance'와 같은 속성이 높은 관계성(entailment)률을 나타내는 반면, 'Is female-identifying', 'Is married'와 같은 속성은 높은 모순(contradiction)률을 보였습니다.



### Each Graph is a New Language: Graph Learning with LLMs (https://arxiv.org/abs/2501.11478)
- **What's New**: 이 논문은 Large Language Models (LLMs)을 활용하여 노드 분류 작업에 대한 텍스트 속성을 가진 그래프 구조를 모델링하는 새로운 프레임워크인 GDL4LLM(Graph-Defined Language for Large Language Model)을 제안합니다. 기존 접근법에서는 그래프 구조 표현이 너무 방대하거나 텍스트 속성만으로는 충분한 정보를 제공하지 못하는 한계를 가지고 있었습니다. GDL4LLM은 그래프를 설명하는 대신 그래프 언어 말뭉치를 생성하고 이를 통해 LLM을 사전 훈련시킴으로써 그래프 구조를 효과적으로 이해할 수 있게 합니다.

- **Technical Details**: GDL4LLM은 그래프를 그래프 언어 코퍼스로 변환하여 노드 분류를 위한 프레임워크를 구현합니다. 이 과정에서, 최근 LLM이 하나의 언어에서 훈련되어도 다른 언어에서 우수한 성능을 보일 수 있다는 점을 활용했습니다. LLM은 이 그래프 언어를 통해 노드에 대한 구조적 정보를 간결하게 학습할 수 있으며, 이는 터치포인트 중심으로 구조적 정보를 설명하는 데에 필요합니다.

- **Performance Highlights**: 실험을 통해 GDL4LLM은 세 가지 실제 데이터셋에서 기존의 설명 기반 접근법이나 텍스트 속성 임베딩 기반 방법들보다 뛰어난 성능을 보였습니다. LLM을 활용하여 다양한 차수의 그래프 구조를 효율적으로 모델링함으로써 노드 분류 작업에서 탁월한 결과를 나타냈습니다. 전반적으로, GDL4LLM은 LLM의 언어 이해 능력을 그래프 구조 데이터에 성공적으로 이전시킬 수 있도록 설계되었습니다.



New uploads on arXiv(cs.IR)

### Graph Neural Controlled Differential Equations For Collaborative Filtering (https://arxiv.org/abs/2501.13908)
Comments:
          Accepted in WWW 2025 short paper

- **What's New**: 이번 논문에서는 Neural ODE(Ordinary Differential Equations)를 기반으로 한 추천 시스템에서의 중량 제어의 중요성을 강조합니다. 새로운 기법인 Graph Neural Controlled Differential Equations for Collaborative Filtering (CDE-CF)를 제안하여 GODE-CF의 한계를 극복하고 각 노드에 맞춤화된 그래프 컨볼루션을 가능하게 합니다. 이는 고정된 중량이 아닌 연속적으로 조정 가능한 중량 제어를 통해 각 노드에 맞는 최적의 추천을 생성합니다.

- **Technical Details**: CDE-CF는 기존 GODE-CF의 프레임워크에서 발전한 방법으로, 각 노드에 대해 분리된 중량을 사용하는 대신 다층 퍼셉트론(MLP)을 활용하여 ODE를 제어합니다. 이 접근법은 ODE 진화 과정에서 지속적인 중량 값을 생성하는 역할을 하며, 이는 데이터에서 학습된 중량 값을 통해 ODE 함수의 변화를 반영합니다. 또한 Neurla ODE 프레임워크는 복잡한 시스템을 모델링하는 데에도 뛰어난 효과를 보입니다.

- **Performance Highlights**: CDE-CF는 다양한 데이터셋에서 실험을 수행한 결과, GCN 및 기존 GODE 기반 방법들과 비교하여 우수한 성능을 보였습니다. 특히 기법의 학습 속도는 대부분의 GCN 기반 모델에 비해 더 빠르며, 다양한 ODE 솔버의 영향을 탐구하여 성능을 한층 더 강화했습니다. 이 논문은 CDE-CF의 코드와 함께 공개되어 있어 향후 연구자들이 해당 기법을 재현하고 발전시킬 수 있는 기회를 제공합니다.



### Large Language Model driven Policy Exploration for Recommender Systems (https://arxiv.org/abs/2501.13816)
- **What's New**: 최근 추천 시스템(Recommender Systems, RS) 분야에서 강화 학습(Reinforcement Learning, RL)의 통합이 두드러지고 있으며, 이를 통해 추천 공정을 마르코프 결정 프로세스(Markov Decision Process, MDP)로 재구성하고 있습니다. 그러나 정적 사용자 데이터를 기반으로 훈련된 오프라인 RL 정책은 동적 온라인 환경에서의 배포 시 분포 변이(distribution shift)에 취약하다는 단점이 있습니다. 이러한 문제를 해결하기 위해, 본 논문에서는 대규모 언어 모델(Large Language Models, LLMs)을 활용하여 사용자 목표 및 선호를 모방하여 오프라인에서 초기 정책을 개선하는 방안을 제시합니다.

- **Technical Details**: 우리는 사용자 선호를 LLM에서 추출하고 이를 기반으로 상호작용 증가 학습 정책(Interaction-Augmented Learned Policy, iALP)을 제안합니다. iALP는 LLM으로부터 파생된 사용자 선호를 사용해 초기 온라인 사용자 상호작용을 학습하며, 피드백에 따라 보상을 학습하고 RL 정책을 업데이트하는 액터-크리틱(actor-critic) 구조를 이용합니다. 또한, 온라인 시나리오에서 iALP를 배포하기 위해 적응형 버전인 A-iALP를 소개하며, 이는 간단한 미세 조정 전략(A-iALP$_{ft}$)과 정책의 문제를 완화하고 탐색을 촉진하는 적응적 접근(A-iALP$_{ap}$)을 포함합니다.

- **Performance Highlights**: 세 가지 시뮬레이션 환경에서 수행한 실험 결과, A-iALP는 초기 추천과 안정적인 에피소드 생성에서 현저한 성능 향상을 보였습니다. 이러한 결과는 오프라인 RL에서 정책 변이 및 한정된 탐색 문제를 효과적으로 해결하고, 장기 사용자 보상을 늘리며 초기 단계에서 사용자 만족 손실을 완화하는 데 기여합니다. 이를 통해, 본 연구는 추천 시스템의 초기 상호작용에서 사용자 이탈을 줄이고 더 나은 추천 품질을 달성하는 데 필수적인 발견을 제공합니다.



### EICopilot: Search and Explore Enterprise Information over Large-scale Knowledge Graphs with LLM-driven Agents (https://arxiv.org/abs/2501.13746)
- **What's New**: EICopilot은 자연어 질의를 이해하고, 대규모 지식 그래프에서 정보를 탐색하며, 복잡한 쿼리를 수행하고 요약하는 지능형 시스템입니다. 이 시스템은 Baidu Enterprise Search의 챗봇으로 배포되어 사용자의 검색 경험을 혁신합니다. EICopilot은 데이터 전처리 파이프라인과 독창적인 쿼리 마스킹 전략을 통해 정보 검색의 정확성과 효율성을 향상시키고 있습니다.

- **Technical Details**: EICopilot의 구조는 온라인 및 오프라인 두 가지 단계로 나뉘어 있으며, 고품질의 데이터 준비 및 검색 쿼리 생성 과정을 중요시합니다. 오프라인 단계에서는 스키마 정보 해석, 시드 데이터 구축, 데이터 증강 및 유사 질문 선택을 통해 데이터 세트를 정리합니다. 온라인 단계에서는 LLM의 능력을 활용하여 사용자의 비표준 쿼리를 해석하고, CoT(Chain-of-Thought) 및 ICL(In-context learning) 기법을 적용해 효율적인 정보 검색을 지원합니다.

- **Performance Highlights**: EICopilot은 데이터 검색 속도와 정확성에서 기존 기법들을 능가하는 성능을 보입니다. 특히, 전반적인 구문 오류율을 10.00%로 줄이며, 실행 정확도는 82.14%에 달합니다. 이러한 결과는 EICopilot의 구성 요소들이 쿼리 및 요약 프로세스를 향상시키는데 중요한 역할을 한다는 것을 보여줍니다.



### AirTOWN: A Privacy-Preserving Mobile App for Real-time Pollution-Aware POI Suggestion (https://arxiv.org/abs/2501.13608)
- **What's New**: 최근 발표된 AirTOWN은 도시 환경에서 사용자의 건강을 고려한 POI(Point of Interest) 추천을 제공하는 개인 정보 보호 기능이 있는 모바일 애플리케이션입니다. 이 시스템은 실시간 공기 질 지수(AQI) 데이터를 사용자 선호와 결합하여 더욱 건강한 결정을 지원합니다. AirTOWN은 협업 필터링(Collaborative Filtering) 및 연합 학습(Federated Learning) 기술을 사용하여 사용자 개인 정보를 보호하며, 이탈리아 바리(Bari)와 영국 코크(Cork)의 센서 네트워크의 AQI 데이터를 통합합니다.

- **Technical Details**: AirTOWN의 아키텍처는 클라이언트-서버 모델을 기반으로 하며, 어플리케이션(Application), 서비스(Service), 인터페이스(Interface), 데이터 자원(Data Resources) 4개의 레이어로 구성됩니다. 애플리케이션 레이어에서 POI 추천이 조정되며, 협업 필터링 모델을 통해 사용자 맞춤형 제안을 생성합니다. 또한, 서비스 레이어는 백엔드 프로세스를 관리하고, 레이디얼 기저 함수 인터폴레이션을 통해 센서가 없는 지역의 AQI 데이터를 근사화합니다.

- **Performance Highlights**: AirTOWN의 예비 실험은 사용자의 선호와 실시간 공기 질 데이터를 균형 있게 반영하여 건강 중심의 POI 추천을 제공하는 데 효과적임을 입증했습니다. 앞으로의 연구는 개인 정보 보호를 더욱 강화하고, 더 큰 사용자 기반과의 평가를 통해 애플리케이션의 효과성을 검증할 예정입니다. AirTOWN은 개인의 건강 요구에 맞춘 맞춤형 추천을 제공하며, 건강한 도시 내비게이션을 지원하는 데 중요한 역할을 할 것으로 기대됩니다.



### MixRec: Individual and Collective Mixing Empowers Data Augmentation for Recommender Systems (https://arxiv.org/abs/2501.13579)
Comments:
          Accepted by WWW'25

- **What's New**: 본 논문에서는 데이터 희소 문제를 해결하기 위해 새로운 Dual Mixing 기반의 추천 프레임워크(MixRec)를 제안합니다. 이 프레임워크는 데이터 증강(data augmentation)에 필요한 매개변수를 최소화하면서도 단일 매개변수로 개선을 추구합니다. Individual mixing과 Collective mixing이라는 두 가지 새로운 mixing 메커니즘을 도입해 각 대상에 맞춤형으로 데이터를 변형합니다.

- **Technical Details**: MixRec는 두 가지 mixing 방식인 Individual mixing과 Collective mixing을 활용하여, 각각 사용자나 아이템에 적합한 새로운 긍정 샘플을 생성하거나 배치 내의 집단적 속성을 포함하는 샘플을 만듭니다. 또한, Dual Mixing Contrastive Learning을 통해 생성된 샘플 간의 일관성을 극대화하여 추천 작업의 신뢰성을 높입니다. 이 방식은 선형 시간 복잡도로 구현되어 기계 학습 속도를 증가시키는 효과를 가져옵니다.

- **Performance Highlights**: 실험 결과, MixRec는 추천 성능, 훈련 효율성, 데이터 희소성을 견디는 능력 및 사용성에서 향상된 결과를 보였습니다. 실제 데이터셋 four datasets에 대한 비교 실험을 통해 기존의 20가지 baseline 방법들과 비교하여 통계적으로 의미 있는 성능 향상을 입증한 점이 중요한 성과로 나타났습니다.



### Federated Conformance Checking (https://arxiv.org/abs/2501.13576)
- **What's New**: 이 논문에서는 연합된 환경에서 개인 정보 보안을 고려한 conformance checking 접근 방식을 제안합니다. 이 접근법은 조직 간의 프로세스 모델의 정확성을 평가하고 잘못된 커뮤니케이션을 식별하여 그 비용을 정량화할 수 있도록 돕습니다. 또한, 세 개의 조직 간의 공급망 프로세스를 시뮬레이션하여 제안된 방식의 유효성을 검증합니다.

- **Technical Details**: 이 논문에서 도입된 개인 정보 보호를 고려한 federated conformance checking 접근 방식은 여러 조직의 프로세스 모델 간의 정확성을 평가하는 것을 목표로 합니다. 이를 위해 synthetic event logs를 생성하고, 가상의 공급망 프로세스를 설정되어, 실제 프로세스에서의 miscommunication을 탐지하고 정량화하는 방법을 설명합니다. 특히, 이러한 분석은 purchase-to-pay, order-to-cash 및 shipment 과정에서 이루어집니다.

- **Performance Highlights**: 제안된 방법을 통해 연구자들은 공급망 내의 miscommunication 비용을 효율적으로 평가할 수 있습니다. 시뮬레이션에서 생성된 이벤트 로그를 활용함으로써, 기존의 문제점들을 정량적으로 이해할 수 있게 됩니다. 이러한 결과는 조직 간 프로세스의 투명성과 효율성을 높이고, 프로세스 개선의 기회를 발생시킬 수 있습니다.



### Billion-scale Similarity Search Using a Hybrid Indexing Approach with Advanced Filtering (https://arxiv.org/abs/2501.13442)
Comments:
          14 pages, 3 figures, published in Cybernetics and Information Technologies

- **What's New**: 이 논문은 CPU 추론을 최적화한 10억 규모 데이터셋에서 복잡한 필터링 기능을 사용한 유사성 검색(similarity search)에 대한 새로운 접근 방법을 제시합니다.

- **Technical Details**: 제안된 방법은 전통적인 IVF-Flat 인덱스 구조를 확장하여 다차원 필터(multi-dimensional filters)를 통합합니다. 또한, 밀집 임베딩(dense embeddings)과 이산 필터링 속성(discrete filtering attributes)을 결합하여 고차원 공간에서 빠른 검색을 가능하게 합니다.

- **Performance Highlights**: CPU 기반 시스템에 맞게 설계된 본 접근법은 대규모 유사성 검색에 대한 비용 효율적인 솔루션을 제공합니다. 사례 연구(case study)를 통해 제안된 방법의 효과를 입증하며, 다양한 실용적 활용 가능성을 보여줍니다.



### Full-Stack Optimized Large Language Models for Lifelong Sequential Behavior Comprehension in Recommendation (https://arxiv.org/abs/2501.13344)
Comments:
          Under Review

- **What's New**: 본 논문에서는 추천 시스템에서 대규모 언어 모델(LLMs)의 장기 순차적 행동 이해 문제를 다룹니다. LLM은 긴 사용자 행동 시퀀스에서 유용한 정보를 효과적으로 추출하는 데 어려움을 겪고 있으며, 이를 해결하기 위한 새로운 프레임워크인 ReLLaX를 제안합니다. ReLLaX는 데이터, 프롬프트, 파라미터의 세 가지 레벨에서 최적화를 제공합니다.

- **Technical Details**: 먼저, 데이터 관점에서 우리는 Semantic User Behavior Retrieval (SUBR)을 도입해 시퀀스의 이질성을 줄이고 LLM이 핵심 정보를 추출하는 데 필요한 특정 행동을 선택합니다. 프롬프트 레벨에서는 Soft Prompt Augmentation (SPA)을 사용해 추천 작업과 연계된 지식을 주입하여 아이템 표현을Align (정렬)합니다. 마지막으로, 파라미터 레벨에서는 CFLoRA를 제안하여 LoRA의 표현력을 강화하고, 구성 요소 간 상호 작용을 통해 더 나은 시퀀스 정보 캡처를 가능하게 합니다.

- **Performance Highlights**: 세 가지 공개 데이터셋에 대한 실험을 통해 ReLLaX가 기존 기준선보다 우수한 성능을 보이며, 장기 순차적 행동 이해 문제를 효과적으로 완화함을 입증했습니다. 또한 CFLoRA 모듈은 기존의 LoRA 기반 LLM4Rec 방법과 비교할 때 더 강력한 성능을 보여줍니다. 이 논문은 LLM 기반 추천 시스템에서의 문제를 처음으로 정의하고, ReLLaX 프레임워크를 통해 해결책을 제시한 중요한 기여를 확인합니다.



### PCSI -- The Platform for Content-Structure Inferenc (https://arxiv.org/abs/2501.13272)
Comments:
          9 pages

- **What's New**: 이번 논문에서는 Web 자원을 구조화된 컨텐츠 객체로 변환하는 과정을 지원하는 PCSI(Platform for Content-Structure Inference) 시스템을 소개합니다. PCSI는 미리 정의된 형식에 따라 정보를 공유할 수 있게 해주며, URL 클래스를 기반으로 한 구조적 컨텐츠 추출 방법을 기록합니다.

- **Technical Details**: PCSI는 HTML DOM을 탐색하는 기능이 있는 Awk의 변형인 Hex로 작성된 스크립트를 사용하여 다양한 URL에서 구조적 컨텐츠를 추출할 수 있습니다. 이 시스템은 특정 URL에 특정 메소드를 적용한 결과를 보고하여 개발자들이 컨텐츠 변환 과정을 더 효율적으로 관리할 수 있도록 합니다.

- **Performance Highlights**: PCSI 시스템은 컨텐츠 변환과 관련된 과정을 자동화하여, 개발자들이 시간과 노력을 절약할 수 있도록 설계되었습니다. 특히, 다양한 URL에서 구조적 컨텐츠를 추출하는 데 있어 사용자에게 유연성과 편리함을 제공합니다.



### Exploring GPT's Ability as a Judge in Music Understanding (https://arxiv.org/abs/2501.13261)
- **What's New**: 이 논문은 음악 정보 검색(MIR) 과제를 해결하는 데 텍스트 기반의 대형 언어 모델(LLMs)인 GPT의 적용 가능성을 탐구합니다. 기존의 LLM 연구들과 달리, 이 연구는 LLM을 사용하여 음악 데이터의 주석 오류를 감지하는 명확한 문제를 제시하고, 시스템적인 프롬프트 엔지니어링 방법론을 통해 MIR 성능을 분석합니다. 이를 통해 단순한 텍스트 기반 사고 능력만으로도 음악 이해를 개선할 수 있는지를 평가하려는 최초의 시도입니다.

- **Technical Details**: 프롬프트 엔지니어링 방법론을 적용하여 세 가지 MIR 작업인 비트 추적(beat tracking), 코드 추출(chord extraction), 키 추정(key estimation)을 수행하였습니다. 각 작업에서 LLM은 미리 정의된 음악 세그먼트와 의도적으로 삽입된 오류가 포함된 주석을 바탕으로 오류를 감지하도록 설정되었습니다. 또한, 음악 개념의 일관성을 평가하기 위해 개념 증강(concept augmentation) 방법을 제안하였고, 이는 특정 음악 용어를 조정하여 LLM의 성능에 미치는 영향을 분석합니다.

- **Performance Highlights**: 실험 결과, 비트 추적, 코드 추출, 키 추정 작업에서 각각 65.20%, 64.80%, 59.72%의 오류 감지 정확도를 달성했습니다. 이는 무작위 기준선을 초과하는 성과로, LLM의 음악 이해 능력이 측정 가능함을 보여줍니다. 또한, 제공된 개념 정보의 양과 GPT의 오류 찾기 정확도 간에 긍정적인 상관관계가 있음을 관찰하여, 향후 LLM 기반 MIR 연구를 위한 기초 데이터를 제공합니다.



### RAMQA: A Unified Framework for Retrieval-Augmented Multi-Modal Question Answering (https://arxiv.org/abs/2501.13297)
Comments:
          Accepted by NAACL 2025 Findings

- **What's New**: 이 논문에서는 텍스트와 이미지를 통합한 Multi-modal retrieval-augmented Question Answering (MRAQA) 분야에서 새로운 접근법인 RAMQA를 제안합니다. RAMQA는 전통적인 learning-to-rank 방법과 generative ranking 기술을 결합하여, 현대의 대형 생성 언어 모델(LLMs)을 활용한 정보 검색의 한계를 극복하고자 합니다. 이를 통해 두 가지 MRAQA 벤치마크인 WebQA와 MultiModalQA에서 성능 향상을 입증하였습니다.

- **Technical Details**: RAMQA는 LLaVA를 기반으로 하여 multi-modal pointwise ranker를 훈련한 후, novel autoregressive multi-task learning 접근법을 채택하여 LLaMA 모델을 상위 k개 문서의 재정렬에 사용합니다. 이 과정에서는 zero-shot LLaVA 모델을 이용하여 다중 모달 문서를 텍스트 표현으로 통합하고, permutation 기법을 활용하여 문서 후보군의 다양성을 증가시켜 bias를 감소시키는 방법을 사용합니다.

- **Performance Highlights**: 실험 결과, WebQA와 MultimodalQA 두 벤치마크에서 강력한 기준선에 비해 유의미한 성능 향상을 달성하였으며, RAMQA는 웹 기반의 QA 시스템에서 네 번째 순위를 기록했습니다. 이 연구는 multi-modal generative LLMs의 활용 가능성을 보여주며, 점진적인 재정렬이 정보 검색에서 더 효율적인 처리를 가능하게 함을 시사합니다.



New uploads on arXiv(cs.CV)

### Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass (https://arxiv.org/abs/2501.13928)
Comments:
          Project website: this https URL

- **What's New**: Fast3R는 DUSt3R의 한계를 극복하기 위해 설계된 새로운 다뷰 3D 재구성 프레임워크입니다. 이 모델은 파라미터화된 이미지 쌍의 처리 대신, 여러 이미지를 병렬로 처리하는 Transformer 기반 아키텍처를 활용하여 단일 전방 패스를 통해 동시에 N개의 이미지를 재구성할 수 있습니다. 이는 기존 방식이 요구하는 반복적 정렬 과정을 없애고, 오차 누적을 현저히 줄이는 효과를 가져옵니다.

- **Technical Details**: Fast3R는 무작위이거나 포즈가 지정되지 않은 이미지 집합(이하 N)에서 3D 포인트 맵을 예측합니다. 이 모델은 1000개 이상의 이미지를 처리할 수 있도록 설계되었으며, 훈련 중에는 이미지 마스킹 기법을 사용하여 더 적은 수의 이미지로 학습됩니다. N개의 RGB 이미지를 입력으로 받아 해당 장면의 3D 구조를 예측하는 구조로, 고속성과 확장성을 갖추고 있습니다.

- **Performance Highlights**: Fast3R는 카메라 포즈 추정 작업에서 99.7%의 정확도를 기록하였으며, 이는 DUSt3R보다 14배 이상의 오류 감소를 이룬 실적입니다. 또한 모델은 여러 뷰를 사용할수록 성능이 향상되며, 훈련 중에 보지 못한 뷰에 대해서도 일반화할 수 있습니다. 이러한 결과들은 Fast3R가 멀티뷰 애플리케이션에서 효율성과 정확도를 모두 갖춘 새로운 기준을 설정한 것을 입증합니다.



### Can We Generate Images with CoT? Let's Verify and Reinforce Image Generation Step by Step (https://arxiv.org/abs/2501.13926)
Comments:
          Journal Version. Code and models are released at this https URL

- **What's New**: 이 논문은 Chain-of-Thought (CoT) 추론을 이미지 생성 시나리오에서 검증하고 강화하는 잠재력을 탐구한 최초의 포괄적인 연구입니다. 여기서는 테스트 시간 컴퓨테이션을 검증하고 모델 선호도를 Direct Preference Optimization (DPO)로 정렬하는 세 가지 기술을 집중적으로 다룹니다. 이 연구는 CoT 추론 기술을 오토 회귀 이미지 생성에 성공적으로 적용할 수 있음을 보여줍니다.

- **Technical Details**: 연구에서는 Show-o라는 최신 이산 생성 모델을 기준으로 사용하며, GenEval이라는 텍스트-이미지 생성 벤치마크에서 성능을 평가합니다. 특히 Outcome/Process Reward Model (ORM/PRM)과 같은 검증 기법 및 DPO를 통한 강화를 통해 모델의 성능을 최적화하는 방법을 살펴봅니다. CoT 추론이 오토 회귀 이미지 생성에 얼마나 효과적으로 적용될 수 있는지를 분석하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 연구 결과는 CoT 추론 전략을 이미지 생성 시나리오에 효과적으로 적용할 수 있는 가능성을 보여줍니다. PARM 및 PARM++라는 두 가지 새로운 보상 모델을 도입하여, 오토 회귀 이미지 생성의 품질을 크게 향상시키며, GenEval에서 기준 모델을 +24% 개선했습니다. 또한, PARM++는 생성된 이미지의 질을 반영 서사 방식으로 세밀하게 조정하여, PARM보다도 +4% 더 나은 성능을 기록했습니다.



### GeoPixel: Pixel Grounding Large Multimodal Model in Remote Sensing (https://arxiv.org/abs/2501.13925)
- **What's New**: 최근 대규모 다중 모달 모델(large multimodal models, LMMs)의 발전은 세부적인 그라운딩(fine-grained grounding)이 시각 이해와 대화의 필수 요소임을 인식했습니다. 그러나 이러한 모델은 자연 이미지 도메인에 국한되어 있으며, 원거리 감지(remote sensing, RS)에서는 낮은 성능을 보이고 있습니다. 본 연구에서는 픽셀 수준의 그라운딩을 지원하는 GeoPixel을 제안하여 고해상도 RS 이미지를 처리할 수 있는 새로운 접근 방식을 제시합니다.

- **Technical Details**: GeoPixel은 4K HD 해상도에서 데이터 분석을 수행할 수 있도록 설계되었습니다. 이 모델은 지역 및 글로벌 공간으로 이미지를 적응적으로 분할하여 효율적인 인코딩과 분석이 가능하게 합니다. 또한 GeoPixelD라는 시각적 그라운딩 데이터 세트를 구축하여 RS 이미지 이해를 위한 다중 모달 그라운딩 대화 생성(multi-modal grounded conversation generation)작업을 지원합니다.

- **Performance Highlights**: GeoPixel은 픽셀 수준의 이해력에서 기존의 LMMs를 초월하여 단일 타겟 및 다중 타겟 분할 작업에서 우수한 성능을 나타냅니다. 본 연구에서는 GeoPixel의 각 구성 요소의 효과를 검증하기 위해 방법론적 제거(ablation) 연구를 실시하였습니다. 또한, 연구 결과는 GeoPixel의 탁월한 성능을 뒷받침하는 다각적인 벤치마크 데이터와 함께 제공됩니다.



### Towards Robust Multimodal Open-set Test-time Adaptation via Adaptive Entropy-aware Optimization (https://arxiv.org/abs/2501.13924)
Comments:
          Accepted by ICLR 2025

- **What's New**: 본 연구에서는 Multimodal Open-set Test-time Adaptation (MM-OSTTA)에 대한 새로운 접근 방식인 Adaptive Entropy-aware Optimization (AEO) 프레임워크를 제안합니다. 기존의 unimodal OSTTA 방법의 한계를 극복하고자 하며, 특히 다중 모달리티에서 발생하는 분포 변화에 효과적으로 적응하는 방법을 탐구합니다. AEO은 알려진 클래스와 알려지지 않은 클래스 간의 엔트로피 차이를 증폭시킴으로써 MM-OSTTA 성능을 개선할 수 있음을 보여줍니다.

- **Technical Details**: MM-OSTTA는 사전 훈련된 멀티모달 모델이 다양한 모달리티 간의 상호 정보를 효율적으로 활용하여 실시간으로 적응하고 알려지지 않은 클래스를 탐지하는 것을 목표로 합니다. 이를 위해 Unknown-aware Adaptive Entropy Optimization (UAE)와 Adaptive Modality Prediction Discrepancy Optimization (AMP)라는 두 가지 주요 구성 요소를 도입합니다. 이 구성 요소들은 각 샘플에 대해 엔트로피 임계값을 기준으로 가중치를 동적으로 할당하고 알려진 샘플에 대해 일관된 예측을 유지하면서 알려지지 않은 샘플에 대해 다양한 예측을 장려합니다.

- **Performance Highlights**: 제안된 AEO 프레임워크는 다양한 도메인 변화 환경에서 광범위한 실험을 통해 그 효능과 다양한 작업에 대한 적합성을 입증했습니다. 특히, 액션 인식 및 3D 의미 세분화와 같은 두 개의 다운스트림 작업에서 유용하게 평가되었으며, 리얼 월드와 유사한 장기적인 MM-OSTTA 시나리오에서도 우수한 성능을 보였습니다. 이 프레임워크는 알려진 클래스와 알려지지 않은 클래스 간의 엔트로피 차이를 지속적으로 최적화하며, 이는 실제 동적 애플리케이션에서 필수적인 기능입니다.



### IMAGINE-E: Image Generation Intelligence Evaluation of State-of-the-art Text-to-Image Models (https://arxiv.org/abs/2501.13920)
Comments:
          75 pages, 73 figures, Evaluation scripts: this https URL

- **What's New**: 최근의 확산 모델(diffusion models) 발전으로, 텍스트-이미지(text-to-image, T2I) 모델들이 놀라운 성과를 보여주고 있습니다. 특히 FLUX.1과 Ideogram2.0 같은 신규 모델들은 복잡한 작업에서 뛰어난 성능을 발휘하며, 이러한 모델들이 일반적인 용도로 활용될 가능성에 대한 논의가 이어지고 있습니다. 이 연구는 T2I 모델들이 전통적인 이미지 생성뿐만 아니라 이미지 편집, 비디오, 오디오, 3D 및 움직임 생성과 같은 다양한 분야에서 가진 능력을 탐구합니다.

- **Technical Details**: 우리는 T2I 모델들의 종합적인 평가를 위해 IMAGINE-E라는 새로운 평가 프레임워크를 개발했습니다. 평가 대상 모델로는 FLUX.1, Ideogram2.0, Midjourney, Dall-E3, Stable Diffusion 3, Jimeng 등을 선정하였으며, 다섯 가지 주요 도메인으로 평가를 나누었습니다. 이 도메인들은 구조화된 출력 생성, 사실성(factuality) 및 물리적 일관성(physical consistency), 특정 도메인 생성, 도전적인 시나리오 생성, 다중 스타일 생성 과제를 포함합니다.

- **Performance Highlights**: 이 모델들의 성능을 평가한 결과, 특히 FLUX.1과 Ideogram2.0이 구조화된 및 특정 도메인 작업에서 두드러진 성과를 보였습니다. 이러한 평가 결과는 T2I 모델들이 AI 툴로서의 가능성과 응용을 확대하고 있음을 시사합니다. 연구 결과는 T2I 모델들의 현재 상태와 미래 방향성을 제시하며, 일반적인 사용성을 향한 진화 과정을 설명합니다.



### Temporal Preference Optimization for Long-Form Video Understanding (https://arxiv.org/abs/2501.13919)
- **What's New**: 이 논문은 기존 비디오 대형 멀티모달 모델(video-LMM)의 시간적 이해(temporal grounding) 성능을 향상시키기 위한 새로운 프레임워크인 Temporal Preference Optimization (TPO)을 제안합니다. TPO는 두 가지 세분화된 데이터셋, 즉 특정 비디오 구간에 집중하는 localized temporal grounding과 전체 비디오 시퀀스의 시간적 의존성을 포착하는 comprehensive temporal grounding을 활용하여 모델의 시간적 반응을 개선합니다. 이 프레임워크는 자가 학습(self-training) 방법론을 통해 잘 정립된 반응과 덜 정확한 반응을 구분하는 데 도움을 줍니다.

- **Technical Details**: 비디오-LMM은 대형 언어 모델(LLM), 비주얼 인코더(visual encoder), 그리고 멀티모달 프로젝터(multimodal projector)를 포함하는 구조로, 입력 비디오 V와 텍스트 시퀀스 x를 처리하여 응답 y의 확률을 모델링합니다. TPO는 Direct Preference Optimization (DPO) 기법을 사용하여 인간의 선호도(preference)를 기반으로 모델의 파라미터를 최적화하며, 이는 명시적인 보상 모델이나 복잡한 강화 학습 알고리즘을 필요로 하지 않습니다. 이 과정에서 TPO는 비디오-LMM의 시간적 반응 능력을 강화하는데 중점을 두고 있습니다.

- **Performance Highlights**: TPO를 적용한 실험 결과, LongVideoBench, MLVU, Video-MME와 같은 세 가지 비디오 이해 벤치마크에서 각각 2.9%, 3.1%, 2.5%의 성능 향상을 달성했습니다. 특히, LLaVA-Video-TPO 모델은 Video-MME 벤치마크에서 7B 모델 중 최고의 성능을 기록하였으며, 이는 TPO의 확장성과 효율성을 강조합니다. 이러한 성과는 장기 비디오 이해(task) 및 시간적 추론(temporal reasoning)을 향상시키는 데 중요한 기초로 작용할 것으로 기대됩니다.



### Improving Video Generation with Human Feedback (https://arxiv.org/abs/2501.13918)
- **What's New**: 최근 비디오 생성 분야에서 큰 발전이 있었으나, 여전히 비디오와 프롬프트 간의 불일치 및 부드럽지 않은 동작 같은 문제가 남아 있습니다. 본 연구는 이러한 문제를 해결하고 비디오 생성 모델을 개선하기 위해 사람의 피드백을 활용하는 체계적인 파이프라인을 개발하였습니다. 특히, 현대 비디오 생성 모델에 중점을 둔 대규모 인간 선호 데이터셋을 구축하고 새로운 비디오 보상 모델인 VideoReward를 도입하여 성과를 측정하였습니다.

- **Technical Details**: 연구진은 약 182k개의 주석이 달린 비디오 생성 선호 데이터셋을 수집하였으며, 이는 시각 품질(Visual Quality), 동작 품질(Motion Quality), 텍스트 정렬(Text Alignment)이라는 세 가지 차원을 포함합니다. 비디오 생성 모델을 위한 세 가지 정렬 알고리즘 - Flow-DPO, Flow-RWR 및 Flow-NRG를 제안하여 이론과 실험을 통해 성능을 평가하였습니다. 특히, Flow-DPO는 고정된 매개변수 β로 설정했을 때 우수한 성능을 보였습니다.

- **Performance Highlights**: 실험 결과, VideoReward는 기존의 보상 모델을 크게 능가하는 성과를 보였으며, Flow-DPO는 Flow-RWR와 표준 감독 미세 조정 방법보다 우수한 성능을 나타냈습니다. 또한, Flow-NRG는 사용자 맞춤형 비디오 품질을 충족하기 위해 여러 목표에 대해 사용자 지정 가중치를 할당할 수 있는 기능을 제공합니다. 이 연구는 최신 비디오 생성 모델과 인간 선호를 정렬하는 기법에 대한 새로운 가능성을 제시합니다.



### Binary Diffusion Probabilistic Mod (https://arxiv.org/abs/2501.13915)
- **What's New**: 이번 논문에서는 Binary Diffusion Probabilistic Model (BDPM)이라는 새로운 생성 모델을 소개합니다. BDPM은 이진 데이터 표현에 최적화된 모델로, 기존의 DDPM(denoising diffusion probabilistic models)의 한계를 극복하여 이진 데이터 구조의 특성을 더욱 정확하게 반영합니다. BDPM은 이미지 복원 작업에서 기존의 최첨단 방법보다 뛰어난 성능을 보여주며, 효율적인 샘플링 절차를 통해 계산 비용을 크게 줄일 수 있습니다.

- **Technical Details**: BDPM은 이미지를 비트플레인(bitplane)으로 분해하고, XOR 기반 노이즈 변환을 사용하여 향상된 노이즈 제어를 제공합니다. 이 모델은 이진 교차 엔트로피 손실(binary cross-entropy loss)을 활용하여 훈련 과정의 안정성과 모델 수렴을 개선합니다. 기존의 DDPM이 사용하는 연속 데이터 표현 대신, BDPM은 이산적이고 이진적인 데이터 구조에 적합하도록 설계되었습니다.

- **Performance Highlights**: BDPM은 이미지 초해상도(super-resolution), 인페인팅(inpainting), 블라인드 이미지 복원(blind image restoration)과 같은 여러 이미지 복원 작업에서 우수한 성능을 발휘했습니다. FFHQ, CelebA, CelebA-HQ 데이터 셋에서 기존의 방법들보다 현저히 높은 성능을 보였으며, 필요한 샘플링 단계 수가 적어 효율적인 추론(inference) 과정을 제공합니다.



### PointOBB-v3: Expanding Performance Boundaries of Single Point-Supervised Oriented Object Detection (https://arxiv.org/abs/2501.13898)
Comments:
          16 pages, 5 figures, 10 tables

- **What's New**: 이번 논문에서는 기존의 point-supervised OOD 방법론을 개선한 PointOBB-v3 프레임워크를 제안합니다. 새로운 접근 방식으로는 추가적인 prior 없이 pseudo rotated boxes를 생성하고, end-to-end 패러다임을 통합하여 효율성을 강화하였습니다. 세 가지 독특한 이미지 뷰를 활용하여 scale과 angle을 동시에 추정하는 데 중점을 둡니다.

- **Technical Details**: PointOBB-v3는 세 가지 이미지를 통합하여 객체의 scale과 angle을 학습하도록 설계되었습니다. 주요 요소로는 교차 뷰 전략, Scale-Sensitive Consistency (SSC) loss, Scale-Sensitive Feature Fusion (SSFF) 모듈, Self-Supervised Angle (SSA) loss, Dense-to-Sparse (DS) 매칭 전략이 포함됩니다. 이러한 구성 요소들은 객체의 스케일과 방향성을 정확하게 예측하기 위한 목적으로 함께 작동합니다.

- **Performance Highlights**: 다양한 데이터셋에서 실험을 진행한 결과, PointOBB-v3는 이전의 최첨단 기술 대비 평균 3.56%의 정확도 향상을 보였습니다. 특히 DIOR-R와 DOTA-v1.0에서 각각 2.20%와 8.76%의 성능 향상이 관찰되었으며, end-to-end 버전은 두 단계 방법론을 초월하는 성능을 기록했습니다.



### Pix2Cap-COCO: Advancing Visual Comprehension via Pixel-Level Captioning (https://arxiv.org/abs/2501.13893)
- **What's New**: Pix2Cap-COCO는 픽셀 수준에서의 캡션 작성을 위한 첫 번째 판옵틱 데이터셋으로, 세분화된 시각적 이해를 증진하기 위해 설계되었습니다. 본 연구에서는 GPT-4V를 활용하여 이미지의 객체들에 대한 픽셀 정렬(Pixel-aligned) 및 인스턴스 전용(instance-specific) 캡션을 자동으로 생성하도록 요청하는 주목적의 자동 주석 파이프라인을 도입했습니다. 총 167,254개의 세밀한 캡션을 생성하여, 각 캡션의 평균 길이는 22.94단어입니다.

- **Technical Details**: Pix2Cap-COCO는 COCO 데이터셋을 기반으로 구축되며, 이 데이터셋은 20,550개의 이미지와 167,254개의 캡션으로 구성되어 있습니다. 각각의 객체는 Set-of-Mark(SoM) 방법을 사용하여 이미지에서 마킹되고, 이후 GPT-4V에서 세밀한 캡션이 생성됩니다. 새로운 작업인 판옵틱 분할-캡션(panoptic segmentation-captioning) 도전 과제를 도입하여 모델이 이미지 내 인스턴스를 인식하며 동시에 상세 설명을 제공하도록 요구합니다.

- **Performance Highlights**: Pix2Cap-COCO는 GPT4RoI와 같은 대형 멀티모달 모델(LMM)의 성능을 향상시키는 데 중요한 역할을 합니다. 예를 들어, Pix2Cap-COCO로 훈련된 GPT4RoI는 ViP-Bench에서 평균 5.1%의 성능 향상을 보이며, 인식 정확도(+11.2%)와 언어 생성 품질(+22.2%)에서 주목할 만한 개선을 기록합니다. 이러한 성능 데이터는 Pix2Cap-COCO가 시각적 표현과 텍스트 표현 간의 세밀한 정렬을 위한 고품질 소스로서의 중요성을 강조합니다.



### Generating Realistic Forehead-Creases for User Verification via Conditioned Piecewise Polynomial Curves (https://arxiv.org/abs/2501.13889)
Comments:
          Accepted at WACV-W 2025

- **What's New**: 이 논문에서는 이마 주름의 생성 방법을 제안합니다. B-spline 및 Bézier 곡선을 사용하여 주름의 기하학적 모델링을 통해 현실적이고 상세한 이미지 생성을 보장합니다. 특히, 생성된 주름 이미지는 diffusion 기반의 Edge-to-Image 변환 모델에서 시각적 프롬프트로 사용되어, 다양한 매치 샘플을 생성합니다.

- **Technical Details**: 주요 기술은 B-spline 및 Bézier 곡스를 활용하여 주름 패턴을 기하학적으로 모델링하는 것입니다. 두 가지 전략을 통해 생성된 샘플의 intra-subject 다양성을 높이는데, 하나는 B-spline의 제어점을 변동시키고, 다른 하나는 dropout 및 elastic 변환을 적용하여 기하학적 시각적 프롬프트에 증강을 가하는 것입니다. 이러한 과정을 통해 실제 데이터와 통합하여 주름 검증 시스템의 성능을 대폭 향상시킵니다.

- **Performance Highlights**: 제안된 방법은 생성된 합성 데이터셋과 실제 세계의 데이터를 통합함으로써, cross-database 검증 프로토콜 아래에서 이마 주름 검증의 성능을 개선합니다. 특히, 교차 데이터베이스 검증에서 주목할 만한 성능 향상을 보여, 기존의 인증 시스템에 비해 더욱 강력한 작업 능력을 발휘하는 가능성을 지니고 있습니다.



### Dual-Modal Prototype Joint Learning for Compositional Zero-Shot Learning (https://arxiv.org/abs/2501.13859)
- **What's New**: 이번 논문에서는 Compositional Zero-Shot Learning (CZSL) 작업을 위한 새로운 Dual-Modal Prototype Joint Learning 프레임워크를 제안합니다. 이 방법은 텍스트와 시각적 프로토타입을 모두 도입하여 모델의 일반화 능력을 향상시키고, modality gap을 줄입니다. 두 가지 프로토타입 간의 상호 보완적 학습을 통해 보다 정교한 데이터를 캡처하고, 클래스 간 변별력을 높이는 데 기여합니다.

- **Technical Details**: 기존의 CZSL 방법은 주로 텍스트와 이미지 간의 feature alignment에 중점을 두었으나, 본 연구는 모델이 텍스트와 시각적 두 가지 프로토타입을 동시에 학습할 수 있는 구조를 도입했습니다. 이는 각각의 프로토타입을 텍스트 및 이미지 피쳐를 기반으로 최적화할 수 있게 해주며, 글로벌 특성과 세부 특성을 모두 고려하는 중요한 역할을 합니다. 특히, 전용 분해 모듈을 통해 각 모달리티의 feature를 풍부하게 만들어 최적의 분류 성능을 도출합니다.

- **Performance Highlights**: 실험 결과, 본 연구 방식은 세 가지 공공 CZSL 벤치마크에서 폐쇄형 환경에서는 최첨단 성능을 달성하고, 개방형 환경에서도 경쟁력 있는 성능을 보여주었습니다. 이러한 결과는 제안된 방법이 조합 일반화를 향상할 수 있는 효과적임을 입증합니다. 따라서, 새로운 프레임워크가 CZSL 분야의 연구에 큰 기여를 할 것으로 기대됩니다.



### First Lessons Learned of an Artificial Intelligence Robotic System for Autonomous Coarse Waste Recycling Using Multispectral Imaging-Based Methods (https://arxiv.org/abs/2501.13855)
Comments:
          Published in Proceedings of Sardinia 2023, 19th International Symposium on Waste Management, Resource Recovery and Sustainable Landfilling

- **What's New**: 현재의 조대 폐기물(코스 그레인 웨이스트) 처리 시설에서는 대형 기계를 사용한 수작업 분류가 이루어지고 있습니다. 이로 인해 재활용 가능한 많은 자원이 조대 폐기물로 분실되고 있는데, 더 효과적인 분류 프로세스 개발이 필요하다는 점이 강조됩니다. 이 연구는 혼합 폐기물 더미에서의 객체 탐지(object detection)와 자율 제어(autonomous control)를 통한 자동화된 분류 방법을 제안합니다.

- **Technical Details**: 주요 기술적 접근은 자외선(ultraviolet, UV), 가시광선(visual, VIS), 근적외선(near infrared, NIR), 그리고 단파 적외선(short-wave infrared, SWIR) 스펙트럼의 다중 스펙트럼 이미지(multispectral images)를 활용한 물질 분류(classification)입니다. 청사진인 이 기법은 대부분 손상되거나 파손된 폐기물 객체에 대해 적용될 수 있습니다. 또한, 비용 효율적인 카메라와 인공지능 기반 컨트롤러를 사용하여 대형 기계의 자율 제어 솔루션을 연구하고 있습니다.

- **Performance Highlights**: 이 연구는 자원을 회수하는 동시에 효율적인 폐기물 분류를 위한 새로운 기술적 진전을 이루고자 합니다. 제안된 방법은 기존 수작업 방식에 비해 성능을 향상시킬 수 있는 잠재력을 가지고 있으며, 구체적인 성과는 실험 및 검증을 통해 입증할 필요가 있습니다. 이러한 접근은 폐기물 관리와 재활용 산업에서 혁신적인 변화를 가져올 것으로 기대됩니다.



### Where Do You Go? Pedestrian Trajectory Prediction using Scene Features (https://arxiv.org/abs/2501.13848)
Comments:
          Accepted by 2024 International Conference on Intelligent Computing and its Emerging Applications

- **What's New**: 이 논문에서는 보행자 궤적 예측을 향상시키기 위해 보행자 상호작용과 환경 맥락을 통합한 새로운 궤적 예측 모델을 제안합니다. 기존 연구들이 보행자 간 상호작용 모델에 집중한 반면, 본 연구는 환경 요인과 장면 객체 배치의 중요성을 강조하였습니다. 공간적 및 시간적 상호작용을 포착하기 위해 sparse graph framework를 사용하고, scene features를 추출하기 위한 고급 이미지 강화 및 시맨틱 분할 기법을 사용합니다.

- **Technical Details**: 우리의 접근 방식은 Sparse Graph Convolutional Network (SGCN)을 통해 보행자 간의 상호작용을 포착하고, scene feature extraction 모듈과 cross-attention 메커니즘을 통해 예측 능력을 강화합니다. 이를 통해 보행자의 위치 정보와 주변 장면 정보를 기반으로 미래 궤적을 예측합니다. 이 과정에서는 각각의 보행자가 특정 시점에서의 공간 좌표로 특성화되어 과거 상태가 미래 위치 예측에 활용됩니다.

- **Performance Highlights**: 실험 결과, 제안한 방법이 기존의 최첨단 접근법들과 비교했을 때 ADE와 FDE 값이 각각 0.252m 및 0.372m로 현저히 우수함을 보여줍니다. 이는 보행자 궤적 예측에서 사회적 상호작용과 환경 맥락의 중요성을 강조합니다. 새로운 모델은 복잡한 도시 환경에서의 보행자 움직임 예측을 보다 효과적으로 수행할 수 있는 가능성을 제공합니다.



### MV-GMN: State Space Model for Multi-View Action Recognition (https://arxiv.org/abs/2501.13829)
- **What's New**: 본 논문은 MV-GMN 모델을 소개하여 다중 시점(multiview) 및 다중 시간 정보(multi-temporal information)를 효율적으로 집계하여 액션 인식(action recognition)에서의 계산 복잡성을 줄이는 것을 목표로 합니다. 이 모델은 Bidirectional State Space Block과 GCN(module)로 구성된 일련의 MV-GMN 블록을 포함하며, 이를 통해 다양한 시점(viewpoint)과 시간(instance)에서의 특징을 효과적으로 통합합니다. 또한, MV-GMN은 Transformer 기반 방법보다 뛰어난 성능을 보이며 선형 추론 복잡성(linear inference complexity)만을 요구하여 다중 시점 액션 인식 기술의 확장성과 응용 가능성을 크게 향상시킵니다.

- **Technical Details**: MV-GMN은 다중 시점 그래프 네트워크(Multi-View Graph Mamba network)로, RGB 및 skeletal 데이터의 집합을 모델링합니다. 각 MV-GMN 블록은 Bidirectional State Space Block과 GCN 모듈로 구성되며, Bidirectional State Space Block은 시점 우선(view-prioritized) 및 시간 우선(time-prioritized) 전략을 포함한 네 가지 스캐닝 전략을 도입합니다. GCN 모듈은 규칙 기반 및 KNN 기반 방법을 활용하여 그래프 네트워크를 구축하며, 이러한 접근 방식은 다양한 관점과 시간 인스턴스에서의 특징을 통합하는 데 효과적입니다.

- **Performance Highlights**: MV-GMN 모델은 NTU RGB+D 120 데이터 세트에서 교차 주제(cross-subject) 및 교차 시점(cross-view) 시나리오에서 각각 97.3% 및 96.7%의 정확도를 달성하며, 기존의 최첨단(State-of-the-Art) 방법들을 초월하여 뛰어난 성과를 입증합니다. 이러한 결과는 MV-GMN이 멀티 뷰 액션 인식 기술의 계산 부담을 줄이고, 처리 속도를 높이는데 도움을 줄 수 있음을 보여줍니다. 이로 인해 MV-GMN은 다중 모드(multi-modal) 인식 분야에서 큰 가능성을 지닌 모델로 자리매김하게 됩니다.



### Video-MMMU: Evaluating Knowledge Acquisition from Multi-Discipline Professional Videos (https://arxiv.org/abs/2501.13826)
- **What's New**: 이 연구에서는 Large Multimodal Models (LMMs)의 비디오로부터의 지식 습득 능력을 평가하기 위한 새로운 벤치마크인 Video-MMMU를 소개합니다. 이 벤치마크는 300개의 전문가 수준 비디오와 900개의 인간 주석 질문으로 구성되어 있으며, 정보 인식, 이해, 적응의 세 가지 인지 단계에 맞춘 질문-답변 쌍을 평가합니다. 또한 비디오 시청 후 성과 개선을 정량화하는 새로운 지식 증가 지표인 Δknowledge를 제안하여 LMMs의 학습 능력을 측정합니다.

- **Technical Details**: Video-MMMU는 교육 비디오로부터 지식을 습득하는 LMMs의 능력을 평가하기 위해 설계된 다중 분야 다중 양식 벤치마크입니다. 이 데이터셋은 6개의 전문 분야(예: 미술, 경영, 의학, 과학, 인문학, 공학)에서 30개 과목을 아우르는 300개의 대학 수준 교육 비디오를 포함하고 있으며, 각 비디오는 인지 단계에 따라 설계된 질문-답변 쌍과 함께 제공됩니다. 특히, 이 벤치마크는 비디오로부터 정보와 지식을 효과적으로 습득하는 LMMs의 성과를 평가하기 위한 체계적인 방법론을 제시합니다.

- **Performance Highlights**: LMMs에 대한 평가 결과, 인지적 요구가 증가함에 따라 성능이 급격히 감소하는 경향을 보였습니다. 인간은 비디오 시청 후 33.1%의 성과 향상을 기록했지만, 가장 성능이 우수한 모델들조차 15.6% 미만의 지식 증가를 보였습니다. 이는 현재 LMMs가 비디오 기반 학습에서의 능력을 효과적으로 습득하기 어려움을 강조하며, 앞으로 이들 모델이 인간 수준으로 학습 능력을 향상시키기 위한 연구가 필요함을 시사합니다.



### By-Example Synthesis of Vector Textures (https://arxiv.org/abs/2501.13812)
- **What's New**: 본 논문에서는 주어진 래스터 예시로부터 임의 크기의 새로운 벡터 텍스처를 합성하는 새로운 방법을 제안합니다. 이 방법은 래스터 이미지를 접근할 수 없는 영역에서 벡터 텍스처를 생성하는 혁신적인 접근 방식을 제공합니다. 기존 연구들은 주로 간단한 다각형에 초점을 맞추었으나, 본 논문에서는 복잡하고 불규칙한 텍스처를 생성할 수 있는 방법론을 제시합니다.

- **Technical Details**: 본 방법은 예시 이미지를 세 가지 계층으로 분해하여 진행합니다. 주요 텍스처(primary textons), 보조 텍스처(secondary textons), 배경으로 나누고, 각 주요 텍스처에 대한 설명자를 생성하여 합성 과정에서 텍스처 간의 관계를 캡처합니다. 본 연구는 Poisson Disks를 사용하여 배경에 대한 점 배치 기법을 적용하여 자연스러운 벡터 텍스처를 합성합니다.

- **Performance Highlights**: 실험을 통해 제안된 방법이 입력 예시와 매우 유사한 벡터 형식의 텍스처를 생성할 수 있음을 입증했습니다. 또한 래스터 이미지를 직접 처리하는 방법과 비교해 경쟁력을 갖추고 있으며, 벡터 출력은 편집 작업을 용이하게 만들어, 래스터 이미지에서는 어려웠던 작업을 가능하게 합니다.



### EgoHand: Ego-centric Hand Pose Estimation and Gesture Recognition with Head-mounted Millimeter-wave Radar and IMUs (https://arxiv.org/abs/2501.13805)
Comments:
          10 pages

- **What's New**: 본 논문에서 제안하는 EgoHand 시스템은 mmWave 레이더(mmWave radar)와 IMU(관성측정장치)를 결합하여 손 제스처를 인식하는 혁신적인 솔루션을 제공합니다. 기존의 VR 기기에서 사용되는 하향 카메라 방식의 개인정보 노출 문제를 해결하며, 사용자에게 향상된 사생활 보호 기능을 제공합니다. 특히, EgoHand는 개인의 몸의 움직임을 정확하게 인식할 수 있는 새로운 기법을 도입하여 VR 상호작용을 보다 안전하고 편리하게 만들어 줍니다.

- **Technical Details**: EgoHand는 두 단계의 스켈레톤 기반 손 제스처 인식 체계를 사용하며, 첫 번째 단계에서 변환기(Transformer) 아키텍처를 통해 손 관절의 좌표를 추정합니다. 이러한 정보를 토대로 제스처를 인식하는 두 번째 단계를 거치며, 이 과정에서 IMU 데이터를 활용해 사용자 머리의 움직임으로 인한 위치 변화를 보상하는 방법을 사용합니다. 이 시스템은 10명의 피실험자를 대상으로 한 실험에서 90.8%의 정확성을 기록하여 효과성을 입증하였습니다.

- **Performance Highlights**: EgoHand의 성능은 다양한 제스처, 사용자, 신체 자세 및 장면에서 일관된 결과를 보여주었습니다. 평균 관절 위치 오류(MPJPE)는 72.7 mm로 나타났으며, 크로스-도메인 실험에서도 각각 83.9%, 76.9%, 77.1%의 정확도를 달성하였습니다. 이는 EgoHand가 VR 제스처 인식의 표준을 높이는 데 기여할 수 있음을 보여줍니다.



### PromptMono: Cross Prompting Attention for Self-Supervised Monocular Depth Estimation in Challenging Environments (https://arxiv.org/abs/2501.13796)
Comments:
          10 pages

- **What's New**: 본 논문에서는 다양한 환경에서의 단안 깊이 (monocular depth) 추정을 위한 새로운 접근인 Visual Prompt Learning을 소개하고, PromptMono라는 자기 감독 학습 (self-supervised learning) 프레임워크를 제안합니다. 이 모델은 학습 가능한 매개변수를 시각적 프롬프트 (visual prompts)로 사용하여 도메인에 특화된 지식을 캡처합니다. 또한, 다양한 조건에서의 깊이 추정을 개선하기 위해 새로운 Gated Cross Prompting Attention (GCPA) 모듈을 도입합니다.

- **Technical Details**: PromptMono는 이미지 표현에 프롬프트 정보를 통합하기 위해 GCPA 모듈을 활용합니다. 이 모듈은 먼저 콘텐츠 게이팅 인식 블록 (CGPB)을 통해 시각적 프롬프트를 처리하고, 3D Depth-wise Convolution을 통해 이미지 특징과 프롬프트를 임베딩 공간으로 투영합니다. 이후 두 가지 간격 주의력 (cross attention) 계산을 통해 프롬프트 향상 이미지 특징을 생성하며, 이미지 변환 기반의 자기 증류 학습 방식을 설계하여, 사전 훈련된 교사 모델 없이 깊이 모델의 지식을 어려운 데이터로 증류합니다.

- **Performance Highlights**: Oxford Robotcar 데이터셋과 nuScenes 데이터셋에서 귀하의 방법을 평가한 결과, 제안된 PromptMono는 다양한 시나리오에서 뛰어난 성능을 보여줍니다. 실험 결과는 제안된 방법이 통합 모델 내에서 다른 도메인에서 깊이 추정의 도전을 효과적으로 해결함을 입증합니다. 이를 통해 실시간 자율주행, 증강 현실 등 여러 분야에서의 응용 가능성을 제시합니다.



### Training-Free Zero-Shot Temporal Action Detection with Vision-Language Models (https://arxiv.org/abs/2501.13795)
- **What's New**: 이 논문에서는 새로운 Training-Free Zero-shot Temporal Action Detection (FreeZAD) 방법을 제안합니다. 이 방법은 기존의 ViL (vision-language) 모델을 활용하여 별도의 미세 조정이나 적응 없이도 식별되지 않은 활동을 직접 분류하고 로컬라이즈(위치 특정) 할 수 있습니다. FreeZAD는 템포럴 모델링의 필요성을 줄이고, 신뢰성 높은 pseudo-label 대신 Actionness Calibration을 도입하여 동작 탐지를 향상시킵니다.

- **Technical Details**: FreeZAD는 LOGarithmic decay weighted Outer-Inner-Contrastive Score (LogOIC)를 통해 시간적 행동 경계에 대한 민감도를 향상시키고, Actionness Calibration을 통해 시각적 특징으로부터 에너지를 활용해 신뢰도 점수를 정제합니다. 또한, Test-Time Adaptation (TTA) 전략을 도입하여 Prototype-Centric Sampling (PCS)을 통해 모델이 보다 효과적으로 적응할 수 있도록 하였습니다. 이를 통해 FreeZAD는 THUMOS14와 ActivityNet-1.3 데이터셋에서 state-of-the-art 비지도 방법을 초월하는 성과를 보여주었습니다.

- **Performance Highlights**: FreeZAD 방법은 기존의 비지도 탐지 방법보다 우수한 성능을 보이며, 요구되는 실행 시간은 단 1/13에 불과합니다. TTA를 결합했을 때, 향상된 방법은 fully supervised 방법과의 격차를 더욱 좁힙니다. 이를 통해 FreeZAD는 동작 탐지 분야에 강력한 대안을 제공하며, 다양한 환경에서의 적용 가능성을 보여 줍니다.



### Solving the long-tailed distribution problem by exploiting the synergies and balance of different techniques (https://arxiv.org/abs/2501.13756)
Comments:
          13

- **What's New**: 본 연구에서는 긴 꼬리 인식(long-tail recognition)을 위한 세 가지 기법인 Supervised Contrastive Learning (SCL), Rare-Class Sample Generator (RSG), 그리고 Label-Distribution-Aware Margin Loss (LDAM)의 상호작용을 탐구합니다. 기존 연구들은 데이터 분포를 변화시키거나 모델의 결정 경계를 조정하여 긴 꼬리 인식 성능을 향상시키려 했으나, 다양한 방법 간의 협력과 보정에 대한 연구는 부족했습니다. SCL은 내부 클래스 클러스터를 증가시키고 명확한 클래스 간 분리를 도모하지만, 주류 클래스에 편향되는 경향이 있습니다. 이를 보완하기 위해 RSG와 LDAM을 결합하여 긴 꼬리 클래스의 성능을 더욱 향상시킵니다.

- **Technical Details**: 컴퓨터 비전에서 긴 꼬리 분포가 일반적임에도 불구하고, 균형 잡힌 데이터셋에서 학습한 CNN 모델은 불균형한 데이터셋에서 성능이 저하됩니다. SCL은 Class-averaging 및 Class-complement 전략을 활용하여 내부 및 꼬리 클래스 모두에 대한 공정한 학습을 보장하는 Balanced Contrastive Learning (BCL)으로 확장됩니다. 본 연구는 각 기법의 강점을 활용하여 상호 보완적인 관계를 형성하고, SCL과 RSG가 긴 꼬리 클래스의 클러스터 특징을 향상시키는 방안을 제시합니다.

- **Performance Highlights**: 실험을 통해 제안된 방법은 긴 꼬리 클래스의 정확도를 향상시키면서도 주류 클래스의 성능을 유지하여 모든 클래스에서 균형 잡힌 개선을 달성합니다. SCL과 RSG의 통합은 클래스 간 뚜렷한 분리를 가능하게 하여 더 나은 성능을 제공합니다. 또한, LDAM은 긴 꼬리 클래스에 대해 더 큰 마진을 부여하여, 각 기법의 강점들이 서로를 강화하는 긍정적인 결과를 이끌어냅니다.



### You Only Crash Once v2: Perceptually Consistent Strong Features for One-Stage Domain Adaptive Detection of Space Terrain (https://arxiv.org/abs/2501.13725)
- **What's New**: 이번 연구는 우주에서의 지형 인식을 위한 YOCOv2 모델을 제안하며, 이는 Visual Similarity-based Alignment (VSA) 기법을 활용하여 기존 YOCOv1보다 31% 이상의 성능 향상을 이루었습니다. YOCOv2는 시뮬레이션과 실제 데이터를 포함한 다양한 환경에서 실시간으로 지형 탐지를 수행할 수 있는 능력을 갖추고 있습니다. 이를 통해 NASA의 임무 데이터를 활용한 실제 비행 하드웨어 성능 평가와 질적 분석을 통해 YOCOv2의 실용성을 증명했습니다.

- **Technical Details**: YOCOv2는 텍스처가 없는 지역 및 다양한 조명 조건에서도 효과적인 지형 탐지를 가능하게 하기 위해 최근의 이미지 생성 기술을 통합했습니다. 우리는 PMDA(Prior Model Domain Adaptation) 기술을 통해, 인스턴스 및 intra-feature 기반 클러스터링 기법을 활용하여 적대적 및 대조학습 프레임워크 내에서 VSA 기법을 강화했습니다. 여섯 개의 데이터셋을 구성하여 Mars, Moon, Asteroid 지형의 UDA(자율 도메인 적응) 평가를 수행했습니다.

- **Performance Highlights**: YOCOv2는 YOCOv1 및 지구에서의 최첨단 방법들과 비교하여 31% 이상의 성능 향상을 달성했습니다. NASA 우주선 하드웨어 기반에서의 벤치마킹을 통해 실제 상황에서의 적용 가능성을 검토했습니다. 깊이 있는 정량적 및 질적 평가로 다양한 환경에서의 VSA 기술의 효과를 입증하여, 향후 우주 비행 미션에서의 활용 가능성을 높였습니다.



### A Mutual Information Perspective on Multiple Latent Variable Generative Models for Positive View Generation (https://arxiv.org/abs/2501.13718)
- **What's New**: 본 연구에서는 Multiple Latent Variable Generative Models (MLVGMs)의 각 잠재 변수가 생성 과정에서 어떻게 기여하는지를 정량적으로 분석하는 새로운 프레임워크를 제안합니다. Mutual Information (MI)를 지표로 사용하여 잠재 변수가 생성하는 이미지에 미치는 영향을 체계적으로 파악하며, 이를 통해 효율적이지 않은 현재의 훈련 방식에서 발생할 수 있는 잠재 변수 활용의 비효율성을 드러냅니다. 이 연구는 또한 MLVGMs의 잠재 변수를 활용하여 Self-Supervised Contrastive Representation Learning (SSCRL)에 적합한 합성 데이터를 생성하는 방법을 제시합니다.

- **Technical Details**: 연구에서는 MLVGMs의 구조적 장점인 잠재 변수의 계층적 분리를 활용하여, 각 잠재 변수의 변동이 생성하는 이미지의 지역적 및 글로벌 특성에 미치는 영향을 미세하게 조절할 수 있습니다. 이를 통해 생성된 이미지의 특징을 효과적으로 조정할 수 있어, 실제 데이터를 사용하지 않고도 고품질 합성 이미지를 생성하는 것이 가능해집니다. 또한 Continuous Sampling (CS) 전략을 도입하여, SSCRL 훈련 중 생성자가 동적으로 새로운 샘플을 생성함으로써 데이터의 다양성을 크게 증가시킵니다.

- **Performance Highlights**: 제안된 방식의 장점은 MLVGMs가 생성한 이미지가 실제 데이터에서 생성된 이미지와 동등하거나 더 나은 품질을 보인다는 것입니다. 실험 결과는 MLVGMs의 잠재 변수를 효과적으로 활용하는 것이 SSCRL을 위한 고품질 합성 이미지 생성에 있어 매우 효율적임을 보여줍니다. 이 연구는 generative modeling과 self-supervised learning 분야에서의 MLVGMs 활용 가능성을 크게 확장하며, 이론적 기반을 제공합니다.



### Skin Disease Detection and Classification of Actinic Keratosis and Psoriasis Utilizing Deep Transfer Learning (https://arxiv.org/abs/2501.13713)
- **What's New**: 이번 연구에서는 피부 질환 진단을 위한 새로운 방법을 제안합니다. 딥러닝 기법을 사용하여 피부 질환을 진단하며, 수정된 VGG16 컨볼루션 신경망 모델을 활용합니다. 기존의 진단 방법들이 비싸고 접근성이 낮은 점을 고려하여, 보다 효율적이고 저렴한 대안을 찾고자 하였습니다.

- **Technical Details**: 제안된 모델은 여러 개의 컨볼루션 레이어로 구성되어 있으며, ImageNet 가중치를 사용하여 수정된 최상위 레이어를 포함합니다. 최상위 레이어는 완전 연결 층(fully connected layers)과 최종 소프트맥스 활성화 층(softmax activation layer)을 통해 피부 질환을 분류합니다. 데이터셋은 'Skin Disease Dataset'으로 공개되어 있으며, 회전(rotation), 이동(shifting), 줌(zoom) 등의 전처리 기술을 적용하여 데이터를 증대(augmentation)했습니다.

- **Performance Highlights**: 제안된 방법론은 수정된 VGG16 모델을 사용하여 90.67%의 정확도를 달성하였으며, 이는 피부 질환 분류에서의 신뢰성을 입증합니다. 이러한 유망한 결과는 실제 응용 프로그램에서 이 접근법의 가능성을 강조합니다.



### YOLO11-JDE: Fast and Accurate Multi-Object Tracking with Self-Supervised Re-ID (https://arxiv.org/abs/2501.13710)
Comments:
          This paper has been accepted to the 5th Workshop on Real-World Surveillance: Applications and Challenges (WACV 2025)

- **What's New**: YOLO11-JDE는 빠르고 정확한 다중 객체 추적(Multi-Object Tracking, MOT) 솔루션으로, 실시간 객체 검출(realtime object detection)과 자체 지도 재식별(self-supervised Re-Identification, Re-ID)을 결합합니다. YOLO11의 전용 Re-ID 브랜치를 포함하여, 이 모델은 Joint Detection and Embedding(JDE)을 수행하며 각 검출에 대한 외관 피처를 생성합니다. 이 모델은 탐지와 동시에 자체 지도 방식으로 훈련을 진행하여 비용이 많이 드는 식별 레이블 데이터셋의 필요성을 없앴습니다.

- **Technical Details**: YOLO11-JDE는 triplet loss와 하드 양성, 반 하드 음성 마이닝 전략을 사용하여 구별되는 임베딩(discriminative embedding)을 학습합니다. 데이터 연관(data association)은 모션, 외관, 위치 단서를 성공적으로 통합하는 맞춤형 추적 구현(custom tracking implementation)으로 향상됩니다. 이 모델은 MOT17 및 MOT20 벤치마크에서 경쟁력 있는 결과를 달성하며, 기존 JDE 방법들보다 FPS에서 우수한 성능을 보이고, 최대 10배 적은 파라미터를 사용합니다.

- **Performance Highlights**: YOLO11-JDE는 MOT 챌린지 벤치마크에서 높은 정확도의 추적 성능을 보여줍니다. 또한, 기존 JDE 방법들과 비교할 때 훨씬 더 적은 파라미터 수로 인해 실시간 MOT 응용 프로그램에 적합한 속도를 자랑합니다. 이 모델은 프레임 속도(FPS) 증가와 함께 고도화된 능력을 바탕으로, 실제 응용 프로그램에 매우 매력적인 솔루션으로 자리잡고 있습니다.



### Regularizing cross entropy loss via minimum entropy and K-L divergenc (https://arxiv.org/abs/2501.13709)
Comments:
          5 pages

- **What's New**: 이 논문에서는 딥러닝에서 분류 문제를 위한 두 가지 새로운 손실 함수, 즉 MIX-ENT와 MIN-ENT를 제안합니다. 이 손실 함수는 표준 교차 엔트로피 손실을 최소 엔트로피 및 Kullback-Leibler (K-L) 발산 항으로 정규화함으로써 확장됩니다. 이러한 접근 방식을 통해, 분류의 정확성을 향상시킬 수 있는 가능성을 모색하고 있습니다.

- **Technical Details**: MIX-ENT 손실 함수는 최소 엔트로피 정규화기와 K-L 정규화기의 합으로 표현될 수 있으며, 교차 엔트로피 손실을 정규화합니다. MIN-ENT 함수는 기본적으로 표준 교차 엔트로피 손실에 최소 엔트로피 정규화기를 추가한 형태입니다. 두 손실 함수 모두 신경망이 출력하는 가설 확률 분포의 엔트로피를 최소화하는데 중점을 두고 있습니다.

- **Performance Highlights**: EMNIST-Letters 데이터셋을 사용한 실험 결과, VGG 모델이 MIX-ENT와 MIN-ENT를 적용함으로써 paperswithcode 리더보드에서 이전의 3위에서 2위로 상승하였습니다. 표준 교차 엔트로피를 사용하는 경우 VGG 모델은 95.86%의 정확도를 기록했으며, MIN-ENT와 MIX-ENT를 적용했을 때 각각 95.933%와 95.927%의 정확도를 달성했습니다.



### EventVL: Understand Event Streams via Multimodal Large Language Mod (https://arxiv.org/abs/2501.13707)
- **What's New**: EventVL은 첫 번째 이벤트 기반 MLLM 프레임워크로, 140만 개의 고품질 데이터 쌍을 사용하여 세밀한 의미 이해를 목표로 한다. 기존의 CLIP 기반 방법들의 한계를 극복하고 이벤트 스트림의 충분한 의미와 맥락을 파악할 수 있도록 설계되었다. 이는 다양한 장면에서 효과적인 학습을 가능하게 하며, 이벤트 기반 비전 커뮤니티의 발전에 기여할 것으로 기대된다.

- **Technical Details**: EventVL은 Event Spatiotemporal Representation을 통해 이벤트 데이터의 시공간적 상관관계를 탐구하고, Dynamic Semantic Alignment 모듈을 통해 세밀한 이미지와 이벤트 간의 정렬을 개선한다. 이 프레임워크는 이미지 및 이벤트 인코더, 그리고 대형 언어 모델(LLM)을 통합하여 이벤트 기반 생성 작업에서 강력한 성능을 발휘한다. 전체적인 실험 결과, EventVL은 이벤트 캡셔닝 및 장면 설명 생성 작업에서 기존 SOTA들을 초월하는 성과를 보여준다.

- **Performance Highlights**: EventVL은 약 23억 개의 매개변수로 구성되어 있으나, 타 MLLM들에 비해 낮은 비용으로 배포할 수 있도록 설계되었다. 이벤트 스트림에 대한 정확한 설명 생성을 가능하게 하며, 다중 전환 대화(interactive dialogue)를 지원하는 기능을 통해 심화된 의미 이해를 지원한다. 이는 다양한 분야에서 이벤트 기반 데이터의 잠재력을 완전히 발휘할 수 있도록 돕는다.



### Training-Free Consistency Pipeline for Fashion Repos (https://arxiv.org/abs/2501.13692)
- **What's New**: 패션 이미지 편집의 최신 혁신을 소개하는 논문에서는, 비간섭 포즈 편집을 위한 FashionRepose라는 훈련 필요 없는 파이프라인을 제안합니다. 이 방법은 의류의 포즈 조정이 가능한 동시에 아이덴티티와 브랜드 속성을 유지하여, 패션 산업의 요구를 충족할 수 있도록 설계되었습니다. FashionRepose는 제로샷 접근 방식을 사용하여 실시간에 가까운 속도로 편집을 수행할 수 있는 장점이 있습니다.

- **Technical Details**: FashionRepose는 롱슬리브 의류의 포즈를 표준화하기 위해 설계된 다단계 파이프라인입니다. 이 접근 방법은 훈련 데이터 없이도 롱슬리브 의류의 포즈 노멀라이제이션(task of garment repose)을 가능하게 합니다. 시스템의 주요 구성 요소는 소스 포즈에서 표준화된 목표 포즈로 포즈를 변환하면서 소스 이미지의 색상, 질감 및 브랜드 속성을 보존하는 것입니다.

- **Performance Highlights**: FashionRepose는 훈련 필요 없이 실시간 결과(60초 이하)를 보장하며, 자동화된 로고 보존 작업을 통해 편집 중 브랜드 아이덴티티를 유지합니다. 이 접근 방식은 e-커머스, 패션 마케팅 및 디자인 프로토타이핑과 같은 분야에서 신뢰할 수 있는 편집 솔루션을 제공하여, 패션 이미지 편집의 산업적 요구를 충족시키는 역량을 갖추고 있습니다.



### MPG-SAM 2: Adapting SAM 2 with Mask Priors and Global Context for Referring Video Object Segmentation (https://arxiv.org/abs/2501.13667)
- **What's New**: 이 논문에서는 새로운 Referring Video Object Segmentation (RVOS) 프레임워크인 MPG-SAM 2를 제안합니다. 이 프레임워크는 기존의 Segment Anything Model 2 (SAM 2)의 한계를 극복하고, 멀티모달 정보와 시간적 동적 인식의 통합을 통해 성능을 향상시킵니다. 특히 MPG-SAM 2는 비디오와 텍스트 특성을 통합적으로 인코딩하여 보다 정확한 세그멘테이션을 가능하게 합니다.

- **Technical Details**: MPG-SAM 2는 통합된 멀티모달 인코더를 사용하여 비디오와 텍스트 특성을 함께 인코딩하고, 이를 통해 세멘틱하게 정렬된 비디오 및 텍스트 임베딩을 생성합니다. 마스크 프라이어 생성기는 이러한 비디오 임베딩과 클래스 토큰을 활용하여 대상 객체의 의사 마스크와 글로벌 컨텍스트를 생성합니다. 이러한 마스크는 프롬프트 인코더에 전달되어 SAM 2를 위한 정확한 프롬프트를 생성하는 데 사용됩니다.

- **Performance Highlights**: 다양한 RVOS 벤치마크에서 MPG-SAM 2의 성능 우수성이 입증되었습니다. 제안된 모듈들의 효과가 실험을 통해 여러 가지 영상 세분화 작업에서 긍정적인 결과를 나타냈습니다. 그룹화된 글로벌 및 역사적 정보를 활용하여 휴먼의 기존 세그멘테이션 모델보다 향상된 정확도를 보였습니다.



### QMamba: Post-Training Quantization for Vision State Space Models (https://arxiv.org/abs/2501.13624)
- **What's New**: 새로운 PTQ(포스트 훈련 양자화) 프레임워크 QMamba가 제안되며, 이는 비전 SSM(상태 공간 모델)을 위한 전반적인 양자화 문제를 해결하게 설계되었습니다. QMamba는 SSM의 활성화 분포 분석을 기반으로 하여, SSM의 분산 특성을 활용하여 양자화 민감도 및 이상치를 드러냅니다. 이를 통해 장기적인 시퀀스를 처리하면서도 데이터 양자화를 위한 효과적인 솔루션을 제공합니다.

- **Technical Details**: QMamba는 Long-tailed Skewness Quantization (LtSQ) 및 Temporal Group Quantization (TGQ)를 포함한 두 가지 기법을 채택하여 SSM의 특수성을 반영합니다. LtSQ는 분산이 긴 꼬리를 가지는 이산 파라미터 양자화에 초점을 맞추며, TGQ는 동적 범위의 숨겨진 상태 시퀀스를 그룹화하여 양자화를 수행합니다. 이를 통해 QMamba는 SSM의 비안정성을 극복하고 성능 저하 없이 양자화합니다.

- **Performance Highlights**: QMamba의 실험 결과는 다양한 비전 모델 및 여러 모델 아키텍처에서 기존의 고급 PTQ 방법들을 초월하는 성능을 보여줍니다. 특히, QMamba는 4비트 활성화 값을 사용하여 ImageNet 분류에서 21.0% 향상된 Top-1 정확도를 기록하였습니다. 이는 SSM 기반 비전 모델의 정확성을 획기적으로 개선하는데 기여합니다.



### Cognitive Paradigms for Evaluating VLMs on Visual Reasoning Task (https://arxiv.org/abs/2501.13620)
- **What's New**: 이 논문은 Vision-Language Model (VLM)이 복잡한 시각적 과제에서의 추론 능력을 평가하는 데 중점을 두고 있습니다. 특히, Bongard Openworld Problems 벤치마크를 사용하여 자연 이미지를 기반으로 한 추론 문제를 해결하는 모델의 성능을 분석합니다. 세 가지 인간 중심의 접근법인 holistic analysis, deductive rule learning, 및 componential analysis를 제안하며, 이러한 접근법을 이용한 VLM이 인간의 성능을 초과하는 결과를 보여줍니다.

- **Technical Details**: VLM의 성능 평가에는 다양한 모델 아키텍처와 파라미터 스케일을 사용하는 것을 포함합니다. 평가에는 classification accuracy와 semantic similarity와 같은 두 가지 주요 지표가 사용됩니다. 이 연구는 데이터셋과 모델의 세부 정보에 대해 Appendix에 기술하고 있어, 자세한 기술적 배경을 제공합니다.

- **Performance Highlights**: 연구 결과, 최첨단 모델인 GPT-4o와 Gemini는 구조적 추론 작업에서 탁월한 성과를 보여주었으며, 특히 componential analysis가 효과적임을 입증하였습니다. 하지만, 합성 이미지 처리와 미세한 구분을 하는 데에서 주요 어려움을 발견했으며 이는 VLM의 강건성과 일반화의 필요성을 강조합니다. 이러한 통찰은 VLM의 향후 발전 방향을 제시하고 있습니다.



### Black-Box Adversarial Attack on Vision Language Models for Autonomous Driving (https://arxiv.org/abs/2501.13563)
- **What's New**: 이 논문에서는 Vision-Language Models (VLMs)를 대상으로 한 블랙박스 적대적 공격인 Cascading Adversarial Disruption (CAD)를 제안합니다. 기존 연구들은 주로 화이트박스 공격에 초점을 맞추었지만, 블랙박스 공격의 실제적인 난이도를 극복하기 위해 CAD가 설계되었습니다. CAD는 Driving Reasoning Chain의 효과적인 저하와 동적인 주행 환경에서의 적용을 목표로 합니다.

- **Technical Details**: CAD는 Decision Chain Disruption과 Risky Scene Induction을 활용하여 저수준 사고를 방해하고 동적 환경에 적응할 수 있는 고수준의 위험한 시나리오를 생성합니다. 이러한 접근 방식은 VLM의 오류 전파를 효과적으로 극대화하며, 최종적으로는 운전 안전성을 저해하는 적대적 시각 입력을 생성합니다. CAD는 다양한 AD VLMs에 대해 여러 환경에서 실험을 진행하여 평균 13.43%의 성능 개선을 보였습니다.

- **Performance Highlights**: CAD는 실제 자율주행 차량에 대한 공격 실험에서도 강력한 효과를 입증하였습니다. VLM 기반의 AD 차량에서는 경로 완료율이 61.11% 감소하였고, 적대적 패치가 부착된 차량이 장애물에 충돌하는 사례가 발생했습니다. 또한, CDAD 데이터셋을 공개하여 향후 연구에 기여할 수 있는 기초 자료를 제공하고 있습니다.



### GoDe: Gaussians on Demand for Progressive Level of Detail and Scalable Compression (https://arxiv.org/abs/2501.13558)
- **What's New**: 이번 연구에서는 3D Gaussian Splatting(3DGS) 기술의 성능을 향상시키기 위해 새로운 모델 독립적 기법인 GoDe(Gaussians on Demand)를 제안합니다. GoDe는 가우시안을 여러 계층으로 구성하여 Level of Detail(LoD) 전략을 가능하게 하여, 다양한 압축 비율에 즉시 적응할 수 있습니다. 이를 통해 재학습 없이도 품질 저하를 최소화하면서도 3DGS 모델의 확장성을 확보할 수 있습니다.

- **Technical Details**: GoDe 기법은 가우시안을 디스크리트하고 점진적인 계층으로 나누어, 각 계층이 이전 계층의 세부 정보를 추가하는 구조를 가집니다. 이는 품질과 성능 간의 균형을 쉽게 조정할 수 있게 해주며, 고유한 압축 기술과 결합하면 다양한 압축 수준에서 쉽게 압축을 조절할 수 있습니다. 연구자들은 높은 레벨의 LoD는 낮은 압축 비율에, 낮은 레벨의 LoD는 높은 압축 비율에 해당하는 구조로 설정하였으며, 이를 통해 다양한 요구 사항에 이상적인 구조를 제공할 수 있습니다.

- **Performance Highlights**: 본 연구에서 제안하는 방법은 기존 비압축 모델에 비해 모델 크기를 최대 99.76%까지 감소시키고, 모집단의 수 또한 98.36%까지 줄일 수 있었습니다. 또한, 다양한 기준 데이터셋과 벤치마크를 통해 저왜곡(low distortion)과 뛰어난 확장성 및 적응성을 보여주었습니다. 이러한 성과는 3DGS 기술의 실시간 성능을 극대화하여 고해상도 장면에 적용할 수 있게 합니다.



### One-Prompt-One-Story: Free-Lunch Consistent Text-to-Image Generation Using a Single Promp (https://arxiv.org/abs/2501.13554)
- **What's New**: 이 논문에서는 'One-Prompt-One-Story' (1Prompt1Story)라는 새로운 훈련이 필요 없는 방법을 제안하며, 이는 단일 프롬프트 내에서 모든 프롬프트를 연결하여 T2I 생성 모델이 일관되게 이미지를 생성하도록 합니다. 이는 기존 모델이 대규모 데이터 세트에 대한 훈련이나 복잡한 수정이 필요하던 문제를 해결합니다. 이 방법은 언어 모델의 고유한 문맥 일관성(context consistency)을 활용하여 캐릭터 정체성을 효과적으로 유지할 수 있습니다.

- **Technical Details**: 1Prompt1Story 접근법은 모든 프롬프트를 하나의 긴 문장으로 통합하고 그 결과로 생성 과정을 두 가지 새롭고 독창적인 기법, 즉 Singular-Value Reweighting(SVR)과 Identity-Preserving Cross-Attention(IPCA)을 사용하는 것입니다. SVR은 현재 프레임의 프롬프트 표현을 정제하고, IPCA는 크로스-어텐션 레이어에서 주체의 일관성을 강화합니다. 이 두 가지 기법은 T2I 생성 모델의 텍스트-이미지 정렬 개선을 목표로 하며, 각 프레임 프롬프트가 개별적으로 표현되는 동시에 정체성을 유지할 수 있도록 합니다.

- **Performance Highlights**: 실험을 통해 우리는 1Prompt1Story 방법이 기존의 다양한 일관된 T2I 생성 접근법들과 비교하여 더 일관된 이미지 생성을 달성했음을 입증하였습니다. 정량적 메트릭과 정성적 평가를 통해 이 방법의 효과성을 보여주었으며, 확장된 ConsiStory+ 벤치마크와의 비교에서도 뛰어난 성능을 발휘하였습니다. 이 연구는 T2I 생성의 새로운 가능성을 열어주며, 다양한 내러티브 기반 비주얼 애플리케이션에 적용할 수 있는 길을 제시합니다.



### ReasVQA: Advancing VideoQA with Imperfect Reasoning Process (https://arxiv.org/abs/2501.13536)
Comments:
          Accepted to main conference at NAACL 2025; 8 pages;

- **What's New**: 이번 연구에서는 Video Question Answering (VideoQA) 성능 강화를 위한 새로운 접근법인 ReasVQA를 소개합니다. 이는 Multimodal Large Language Models (MLLMs)에 의해 생성된 추론 과정을 활용하여 VideoQA 모델의 품질을 개선합니다. 세 가지 단계인 Reasoning Generation, Reasoning Refinement, Learning from Reasoning을 통해 구현되며, 특히 데이터 필터링을 통해 생성된 추론 과정을 정제하여 모델 학습에 활용합니다.

- **Technical Details**: ReasVQA의 첫 번째 단계인 Reasoning Generation에서 SOTA MLLMs를 사용하여 VideoQA 작업에 대한 추론 과정을 생성합니다. 두 번째 단계에서는 이러한 추론 과정을 정제하기 위해 데이터 필터링 기법을 적용하여 오류를 최소화하고, 마지막 단계에서는 다중 작업 학습(multi-task learning) 프레임워크를 통해 VideoQA 모델이 질문에 대한 답변을 제공하는 동시에 추론 과정을 생성하도록 훈련합니다. 이를 통해 모델은 정확한 질문-답변 관계뿐만 아니라 해당 답변 뒤의 추론 과정을 학습합니다.

- **Performance Highlights**: 실험 결과, ReasVQA는 새로운 최첨단 성능을 달성했습니다. NExT-QA에서 +2.9, STAR에서 +7.3, IntentQA에서 +5.9의 성능 향상을 보이며, 생성된 추론 과정을 VideoQA 모델에 통합하는 것이 효과적임을 입증했습니다. 각 단계의 효과성에 대한 세밀한 분석 또한 진행되어, 추론 정제와 다중 작업 학습이 전반적인 성능 향상에 기여한 바를 확인할 수 있었습니다.



### Overcoming Support Dilution for Robust Few-shot Semantic Segmentation (https://arxiv.org/abs/2501.13529)
Comments:
          15 pages, 15 figures

- **What's New**: 이 연구는 Few-shot Semantic Segmentation (FSS)에서 발생하는 support dilution 문제를 다루고 있습니다. 기존 FSS 네트워크가 지원 세트의 크기가 증가함에 따라 저성능 지원 이미지에 의해 영향을 받는 문제를 해결하고자 합니다. 본 연구는 고수익 지원 이미지를 선택하고 보존하는 방법을 제안하고 있습니다.

- **Technical Details**: 연구에서는 세 가지 혁신적인 기능을 통해 문제를 해결합니다. 첫째, contribution index를 제안하여 고수익 지원이 얼마나 희석되는지를 정량적으로 평가합니다. 둘째, Symmetric Correlation (SC) 모듈을 개발하여 저수익 특성의 방해를 최소화하면서 고수익 지원 특성을 보존하고 강화합니다. 셋째, Support Image Pruning 작업을 통해 저수익 지원을 제거하여 고품질 서브셋을 추출합니다.

- **Performance Highlights**: COCO-20i와 PASCAL-5i라는 두 개의 FSS 벤치마크에서 광범위한 실험을 수행했습니다. 연구 결과는 기존 FSS 방법들과 비교했을 때 우리의 솔루션이 뛰어난 성능을 발휘함을 보여주었습니다. 또한, 온라인 분할 및 실제 세계 분할에서도 적용되어 실제 활용 가능성을 입증하는 설득력 있는 결과를 나타냈습니다.



### Diffusion-based Perceptual Neural Video Compression with Temporal Diffusion Information Reus (https://arxiv.org/abs/2501.13528)
- **What's New**: 최근 영상 압축 분야에 기초한 새로운 접근 방식으로 DiffVC가 소개되었습니다. 이 프레임워크는 기초적인 확산 모델(foundational diffusion model)과 영상 조건 적재 코딩(paradigm)을 효과적으로 통합하여 고품질의 결과를 생성합니다. 이를 통해 각 프레임에서의 시간적 맥락을 활용하고, 이전 프레임에서의 정보를 재사용하는 대응 전략인 Temporal Diffusion Information Reuse (TDIR) 방식을 제안하여 효율적인 추론을 지원합니다.

- **Technical Details**: DiffVC는 압축 모델의 양자화 파라미터를 입력으로 사용하여 중간 특징을 조절하는 Quantization Parameter-based Prompting (QPP) 메커니즘을 도입하여, 다양한 비트레이트에서의 왜곡 차이에 대처합니다. 또한, 각 P-프레임의 확산 과정은 이전 프레임으로부터 정보를 재사용하여 빠르게 처리하는 첫 번째 단계와 고품질 세부 묘사를 위해 전통적 확산 단계를 사용하는 두 번째 단계로 나누어집니다. 실험 결과, 이 전략 덕분에 추론 시간이 47% 단축되었으며, 인지 성능은 단 1.96% 감소했습니다.

- **Performance Highlights**: DiffVC는 여러 데이터셋에서 시각적 품질 및 인지 메트릭에서 뛰어난 성능을 발휘하며, 특히 DISTS 메트릭에서 최상의 성능을 기록했습니다. 시간적 확산 정보 재사용 전략과 양자화 파라미터 기반 프롬프트 메커니즘을 통해, DiffVC는 단일 모델로 효율적인 추론과 견고한 가변 비트레이트 기능을 지원합니다. 이러한 결과는 DiffVC가 Perceptual Neural Video Compression 분야에서 새로운 기준을 제시함을 보여줍니다.



### Text-driven Online Action Detection (https://arxiv.org/abs/2501.13518)
Comments:
          Published in Integrated Computer-Aided Engineering

- **What's New**: 이 논문에서는 TOAD(텍스트 기반 온라인 액션 탐지) 아키텍처를 소개하며, 이는 제로샷(zero-shot) 및 피사이드(few-shot) 학습을 지원합니다. TOAD는 CLIP(Contrastive Language-Image Pretraining) 텍스트 임베딩을 활용하여, VLMs(비전-언어 모델)을 효율적으로 사용할 수 있도록 하여 계산적 오버헤드를 최소화합니다. 이 모델은 THUMOS14 데이터셋에서 82.46% mAP(평균 정확도)를 달성하여 기존 방법들을 초월하고, THUMOS14 및 TVSeries 데이터셋에 대한 제로샷 및 피사이드 성능의 새로운 기준을 설정합니다.

- **Technical Details**: TOAD는 비전-언어 모델(VLM)의 장점을 활용하여 데이터 효율적인 학습을 가능하게 하는 새로운 구조입니다. 기존의 RNN 모델들은 장기 시간 의존성(long-range temporal dependencies)을 효과적으로 포착하는 데 어려움을 겪었으나, TOAD는 Transformer 아키텍처의 자기 주의(self-attention) 메커니즘을 기반으로 하여 이를 해결합니다. 텍스트 임베딩을 초기화로 사용하여 계산 비용을 줄이면서도 최첨단 결과를 도출할 수 있습니다.

- **Performance Highlights**: TOAD의 성능은 성능 평가 결과에서 뚜렷하게 나타납니다. THUMOS14 데이터셋에서 82.46% mAP를 기록하며, 이는 현재 온라인 액션 탐지 분야의 최상위 성과에 해당합니다. 또한, 제로샷 및 피사이드 방법론을 통해, 데이터 레이블링이 어려운 상황에서도 우수한 성능을 발휘해 향후 연구의 기초를 다질 수 있는 강력한 기반을 제공합니다.



### Propensity-driven Uncertainty Learning for Sample Exploration in Source-Free Active Domain Adaptation (https://arxiv.org/abs/2501.13517)
- **What's New**: 이 논문은 Source-free Active Domain Adaptation (SFADA) 문제를 다루며, 소스 도메인 데이터에 접근하지 않고도 사전 훈련된 모델을 새로운 도메인에 적응시키는 방법을 제안합니다. 특히, 라벨링 비용과 데이터 프라이버시 문제를 해결하는 데 초점을 맞추고 있습니다. 제안된 방법인 Propensity-driven Uncertainty Learning (ProULearn)은 더 유용한 샘플을 선택하는 기법을 기반으로 하여, 복잡한 도메인 전이 문제를 해결할 수 있도록 도와줍니다.

- **Technical Details**: ProULearn은 새로운 동질성 경향 추정 메커니즘과 상관관계 지수 계산을 이용하여 특징 수준의 관계를 평가합니다. 이 방법은 모형이 불확실성이 높은 샘플을 선택하도록 하여 더 의미 있는 샘플을 구별하는 데 중점을 둡니다. 또한, 중앙 상관 손실을 개발하여 의사 레이블을 정제하고 클래스 분포를 응축시키는 데 기여합니다.

- **Performance Highlights**: 본 연구는 네 가지 벤치마크 데이터세트를 통해 ProULearn이 최신 방법들보다 뛰어난 성능을 보임을 입증하였습니다. 실험 결과, 키 데이터 포인트에서 학습을 통해 모델의 도메인 적응능력이 개선되었음을 시각적으로 확인할 수 있었습니다. ProULearn의 샘플 선택 방식은 다양한 딥러닝 작업에서도 유용할 수 있는 통찰력을 제공합니다.



### Quantized Spike-driven Transformer (https://arxiv.org/abs/2501.13492)
Comments:
          Accepted by ICLR 2025

- **What's New**: 본 논문은 에너지 효율적인 스파이킹 신경망(SNN) 분야에서 새로운 정량화 스파이크 기반 변환기(QSD-Transformer)를 제안합니다. 기존의 Transformer 구조는 대규모 자원에 의존하여 성능 향상에 집중했으나, 이는 리소스가 제한된 장치에서 배포에 어려움을 초래합니다. QSD-Transformer는 낮은 비트 너비 매개변수를 사용하여 리소스 요구 사항을 줄이며, 성능 저하 문제를 해결하기 위한 새로운 양자화 기법을 통해 이점을 제공합니다.

- **Technical Details**: QSD-Transformer는 32비트 가중치를 저 비트 너비로 직접 정량화하여 경량화된 스파이킹 변환기 기초 모델을 구축하였습니다. 내부적으로는 정보 증강된 LIF(Lifetime Input Filter)와 세밀한 증류 기법을 도입해 스파이크 기반 자가 주의(Q-SDSA)의 정보 분포를 수정합니다. 이 접근법은 신경 수준과 네트워크 수준 모두에서 작업을 최적화하며, 특히 양자화 과정에서의 정보 왜곡 문제(SID)를 다룰 수 있도록 설계되었습니다.

- **Performance Highlights**: QSD-Transformer는 ImageNet 벤치마크에서 80.3%의 top-1 정확도를 달성하며, 이전 SNN 모델보다 6.0배의 전력 소비 감소와 8.1배의 모델 크기 감소를 동시에 보여줍니다. 이 연구는 다양한 비주얼 작업에서 기존 스파이킹 비전 변환기보다 현저한 성능 향상을 견지하면서도 모델 크기와 전력 소비 면에서 우수한 결과를 입증하였습니다.



### LDR-Net: A Novel Framework for AI-generated Image Detection via Localized Discrepancy Representation (https://arxiv.org/abs/2501.13475)
- **What's New**: 이 논문에서는 AI 생성 이미지 탐지를 위한 새로운 접근 방식인 지역적 불일치 표현 네트워크(Local Discrepancy Representation Network, LDR-Net)를 제안합니다. LDR-Net은 생성된 이미지에서 발생하는 스무딩 아티팩트 및 텍스처 불규칙성을 포착하여, 기존의 특정 생성 모델에 의존하는 탐지 방법들보다 일반화 능력을 향상시킵니다. 이는 지역적 관점에서 생성 이미지의 특성을 분석함으로써, AI 생성 이미지의 자동 탐지 필요성을 충족하고자 하는 노력의 일환입니다.

- **Technical Details**: LDR-Net은 두 개의 보완 모듈인 국소 기울기 자가상관계수(Local Gradient Autocorrelation, LGA)와 국소 변이 패턴(Local Variation Pattern, LVP)을 통합하여 설계되었습니다. LGA는 가장자리 텍스처의 스무딩 이상을 탐지하고, LVP는 픽셀 분포의 복잡한 변칙을 모델링하여 비정상적인 규칙성을 포착합니다. 이러한 두 가지 특징을 결합함으로써, 생성 이미지와 실제 이미지 간의 전문적인 불일치를 파악하는 데 큰 도움을 줍니다.

- **Performance Highlights**: LDR-Net의 성능은 다양한 실험을 통해 검증되었으며, 기존의 AI 생성 이미지 탐지 방법들과 비교할 때 우수한 일반화 능력을 보였습니다. 특히, 이전 unseen generative models에 대해서도 높은 정확도를 자랑하며, 손실된 세부 사항 및 비정상적인 픽셀 변화를 효과적으로 구별할 수 있습니다. 이 연구는 AI 생성 이미지의 진실성과 콘텐츠 신뢰성을 보장하는 데 기여할 것으로 기대됩니다.



### Leveraging Textual Anatomical Knowledge for Class-Imbalanced Semi-Supervised Multi-Organ Segmentation (https://arxiv.org/abs/2501.13470)
- **What's New**: 이 논문에서는 3D 의료 이미지를 주석 처리하는 데 소요되는 시간과 전문성을 줄이기 위해 반지도 학습(semi-supervised learning, SSL) 방법을 제안합니다. 특히, 복잡한 해부학적 구조로 인한 클래스 불균형 문제를 해결하기 위해 텍스트 해부학적 지식(textual anatomical knowledge, TAK)을 통합하여 세분화 모델의 성능을 향상시키는 접근 방식을 제안합니다.

- **Technical Details**: 구체적으로, 우리는 GPT-4o를 사용하여 해부학적 정보를 캡처하는 텍스트 설명을 생성합니다. 이 설명은 CLIP 기반 모델을 사용하여 인코딩되며, 이후 세분화 모델로 주입되어 세분화 헤드의 매개변수로 사용됩니다. 또한, 대조 학습(contrastive learning)을 통해 이러한 텍스트 프라이어와 시각적 특징 간의 정열을 개선합니다.

- **Performance Highlights**: 심층 실험을 통해 제안된 방법이 최첨단 기법을 보완하며 세분화 성능이 크게 향상된다는 것을 입증했습니다. 특히, 소형 장기와 같은 도전적인 카테고리에서 세분화 정확도가 현저히 향상되었습니다. 이와 같은 접근법은 의료 영상 분석 분야에서 중요한 의미를 가집니다.



### Streaming Video Understanding and Multi-round Interaction with Memory-enhanced Knowledg (https://arxiv.org/abs/2501.13468)
Comments:
          Accepted to ICLR 2025. Code is available at this https URL

- **What's New**: 최근 Large Language Models (LLMs)의 발전은 Video-LLMs의 개발로 이어졌으며, 이는 비디오 데이터와 언어 작업을 결합하여 다중 모달 학습을 증진시키고 있습니다. 하지만 기존 비디오 이해 모델은 긴 비디오 시퀀스를 처리하는 데 어려움을 겪고 있으며, 다중 회전 대화 및 현실 세계의 동적 시나리오에 적응하는 데 한계를 보이고 있습니다. 이러한 문제를 해결하기 위해, 우리는 StreamChat이라는 훈련이 필요 없는 프레임워크를 제안합니다.

- **Technical Details**: StreamChat은 복잡한 비디오 특징을 효율적으로 처리하고 압축할 수 있도록 새로운 계층 메모리 시스템을 활용하여 실시간 다중 회전 대화를 가능하게 합니다. 이 프레임워크는 프로세싱 속도를 높이고 대기 시간을 줄이는 병렬 시스템 스케줄링 전략을 포함하여 현실적인 애플리케이션에서 강력한 성능을 보장합니다. 또한, StreamBench라는 다재다능한 벤치마크를 통해 다양한 미디어 유형과 상호작용 시나리오에서 스트리밍 비디오 이해를 평가합니다.

- **Performance Highlights**: StreamChat은 StreamBench 및 기타 공개 벤치마크에 대해 광범위한 평가를 수행하여 기존 최첨단 모델들에 비해 정확도와 응답 속도에서 현저한 성과를 보여줍니다. 특히, 온라인 설정에서 StreamBench에서 64.7%의 정확도를 기록하며, 이는 이전 최고 기록에 비해 8.3% 향상된 수치입니다. 오프라인 시나리오에서는 공공 벤치마크 4개에서 평균 2.5% 더 뛰어난 성능을 보이며, 스트리밍 비디오 처리의 효율성에서도 32 FPS의 처리 속도를 달성하여 기존 기법보다 6배 향상된 결과를 보여줍니다.



### EchoVideo: Identity-Preserving Human Video Generation by Multimodal Feature Fusion (https://arxiv.org/abs/2501.13452)
- **What's New**: 이번 연구에서는 EchoVideo라는 새로운 모델을 제안하여, 기존의 identity-preserving text-to-video(프리미엄 비디오 만들기) 생성 기술에서 발생하는 'copy-paste' 아티팩트와 낮은 유사성 문제를 해결하고자 합니다. EchoVideo는 Identity Image-Text Fusion Module(IITF)를 통해 고수준의 의미적 특성을 통합하고, 이중 훈련 전략을 적용하여 신뢰성을 높입니다. 이 과정을 통해, 모델은 고유한 얼굴 정체성을 보다 효과적으로 보존하면서 전신의 일관성도 유지할 수 있도록 합니다.

- **Technical Details**: EchoVideo는 이미지 내에서 발생할 수 있는 불필요한 세부정보를 제거하고, 고수준의 의미적 특징을 캡처하기 위해 IITF 모듈을 사용합니다. 이 모듈은 텍스트 의미, 이미지 의미 및 얼굴 정체성을 통합하여 의미적 갈등을 해결하고, 영상에서 보다 안정적이고 일관된 캐릭터 표현을 생성할 수 있게 합니다. 또한, IITF는 다중 모드 정보를 결합하는 구조적 접근 방식을 제시하여, 기존 방법보다 복잡성을 줄이고 효율성을 높입니다.

- **Performance Highlights**: 광범위한 실험을 통해 EchoVideo는 고품질, 높은 조작성과 신뢰성을 갖춘 비디오 생성을 이룩하였습니다. 이는 사용자들이 단순한 텍스트 프롬프트를 통해 제어할 수 있음을 보여주는 성과로, 기존 방법들이 요구하는 추가적인 포즈 정보 없이도 구현됩니다. EchoVideo의 성능은 특히 인물의 얼굴 정체성을 보존하면서 의상과 헤어스타일과 같은 추가적인 특성을 유지하는데 효과적임을 입증하였습니다.



### MultiDreamer3D: Multi-concept 3D Customization with Concept-Aware Diffusion Guidanc (https://arxiv.org/abs/2501.13449)
Comments:
          9 pages

- **What's New**: MultiDreamer3D라는 새로운 방법을 제안하여, 서로 다른 개념을 포함한 일관된 3D 콘텐츠를 생성할 수 있게 되었습니다. 기존의 연구들은 주로 단일 개념의 3D 맞춤화에 중점을 두었으나, 본 연구는 다중 개념 3D 맞춤화 문제에 도전하고 있으며, 이를 위한 두 가지 주요 모듈인 3D Layout Generator (LG)와 Concept-aware Diffusion Guidance (CDG)를 포함하고 있습니다.

- **Technical Details**: MultiDreamer3D는 LLM 기반의 레이아웃 컨트롤러를 사용하여 3D 바운딩 박스를 생성하고, 선택적 포인트 클라우드 생성기를 통해 각 개념의 거친 포인트 클라우드를 만들어냅니다. 이러한 포인트 클라우드는 3D 바운딩 박스에 배치되며, 개념 레이블을 통해 2D 프로젝션에서의 개념 속성을 정확하게 식별할 수 있게 됩니다. 결국, 개념 인식을 고려한 간격 점수 매칭을 통해 3D 가우시안을 정제하여 개별 개념의 정체성을 유지합니다.

- **Performance Highlights**: MultiDreamer3D의 실험 결과는 복잡한 상호작용 및 속성 변화 같은 사례에서 여러 개념의 뚜렷한 정체성을 유지하면서 객체 존재와 일관된 레이아웃을 보장해주는 능력을 보여주었습니다. 이 연구는 다중 개념 3D 맞춤화 문제를 처음으로 다룬 것인 만큼, 앞으로의 다양한 응용 가능성을 기대하게 합니다.



### One-cycle Structured Pruning with Stability Driven Structure Search (https://arxiv.org/abs/2501.13439)
Comments:
          12 pages, 6 figures

- **What's New**: 이 논문에서는 기존의 복잡한 다단계 훈련 절차를 간소화하여 성능 저하 없이 효율적인 'one-cycle structured pruning' 프레임워크를 제안합니다. 이 방법은 사전 훈련, 프루닝(pruning), 그리고 미세 조정(fine-tuning)을 하나의 훈련 사이클로 통합하여 훈련 비용을 절감합니다. 새로운 pruning 지표를 도입하여 훈련 에폭(epoсh) 간의 유사성을 평가함으로써 안정적인 pruning 시점을 결정합니다.

- **Technical Details**: 제안된 방법은 초기 훈련 단계에서 최적의 서브 네트워크(sub-network)를 탐색하며, 이는 norm 기반의 그룹 중요도(saliency) 기준과 구조적 희소성(structured sparsity) 정규화를 통해 이뤄집니다. 구조적 희소성 정규화는 프루닝 프로세스를 가속화하며, 이는 전체 훈련 시간을 단축시킵니다. 여러 데이터셋(CIFAR-10/100, ImageNet)에서 VGGNet, ResNet, MobileNet 및 ViT 아키텍처를 사용하여 실험을 통해 효과성을 입증하였습니다.

- **Performance Highlights**: 제안된 알고리즘은 ResNet50 모델에서 ImageNet 데이터셋에 대해 75.49%의 top-1 및 92.63%의 top-5 정확도를 달성하였습니다. 또한, 기본 훈련에 비해 1.38배의 훈련 속도 향상을 이루었으며, 네트워크의 플로팅 포인트 연산(FLOPs)을 57% 이상 감소시켰습니다. 이는 훈련 시간 측면에서 가장 효율적인 pruning 프레임워크 중 하나로 자리 잡을 수 있는 가능성을 의미합니다.



### GC-ConsFlow: Leveraging Optical Flow Residuals and Global Context for Robust Deepfake Detection (https://arxiv.org/abs/2501.13435)
- **What's New**: 이번 연구에서는 Deepfake 탐지의 새로운 접근 방식을 제안했습니다. GC-ConsFlow는 공간적(spatial)과 시간적(temporal) 특징을 통합하여 보다 강력한 Deepfake 영상 탐지를 목표로 하고 있습니다. 특히, 잔여(optical flow residuals) 분석 방법을 도입하여 자연스러운 얼굴 동작에 의한 간섭을 효과적으로 억제합니다.

- **Technical Details**: GC-ConsFlow는 두 개의 주요 스트림으로 구성되어 있습니다: 글로벌 컨텍스트 인식 프레임 흐름(GCAF)와 흐름-그래디언트 시간 일관성(FGTC) 스트림입니다. GCAF 스트림은 GGCA 모듈을 통해 공간적 특징을 강화하고, FGTC 스트림은 비정상적인 아티팩트를 모델링하기 위해 optical flow residuals와 그래디언트 기반 특징을 활용합니다. 이러한 두 스트림의 조합으로 보다 정밀한 시간적 및 공간적 조작 흔적을 포착할 수 있습니다.

- **Performance Highlights**: GC-ConsFlow는 다양한 압축 시나리오에서 기존의 최첨단 방법들보다 우수한 성능을 보였습니다. 실험 결과, 제안된 방법은 Deepfake 비디오의 탐지 정확도를 상당히 향상시켜, 사회적 문제를 해결하는 데 기여할 수 있는 가능성을 보여줍니다. 구체적으로, GC-ConsFlow는 강력한 압축에도 견딜 수 있는 능력을 확인했습니다.



### Emotion estimation from video footage with LSTM (https://arxiv.org/abs/2501.13432)
Comments:
          11 pages, 6 figures, 32 references, 4 tables

- **What's New**: 이 논문에서는 실시간 카메라 스트림에서 생성된 blend-shapes를 처리하여 얼굴 표정으로부터 주요 감정을 추정하는 LSTM 모델을 제안합니다. FER2013 데이터셋을 기반으로 훈련된 이 모델은 71%의 정확도 및 62%의 F1-score를 달성하며, FER2013 데이터셋의 정확도 기준을 충족하면서도 계산 비용이 크게 감소합니다.

- **Technical Details**: 감정 추정 시스템은 얼굴 감정 인식(Facial Emotion Recognition, FER)을 위한 데이터 처리 파이프라인으로 구성됩니다. 이 시스템은 얼굴 검출, 랜드마크 검출 및 감정 분류의 세 단계로 진행됩니다. MediaPipe 라이브러리를 사용하여 얼굴의 응시 방향 및 기능을 정확히 찾아내고, blendshape을 특징으로 사용하여 감정을 분류합니다.

- **Performance Highlights**: 본 연구는 감정 추정이 가능함을 입증하는 개념 증명(proof of concept, POC)으로, Z `NVIDIA GeForce RTX 3050 Mobile` 그래픽 카드에서 훈련이 이루어졌습니다. 전체적으로, 모델은 단순하고 비용 효율적인 구조를 가지고 있으며, facial expression에 대한 공간적 및 시간적 측면을 고려하여 건설되었습니다.



### Auto-Prompting SAM for Weakly Supervised Landslide Extraction (https://arxiv.org/abs/2501.13426)
Comments:
          5 pages, 5 figures

- **What's New**: 본 논문에서는 약한 감독(weakly supervised) 방식으로 토사 재해 지역을 원격 탐지 데이터에서 추출하는 새로운 방법인 APSAM(Adaptive Prompt Segmentation of Anything Model)을 제안한다. 이 방법은 고품질 클래스 활성화 맵(class activation maps, CAMs)에 의존하지 않고, 자동 프롬프트(auto-prompting) 방식을 통해 세밀한 분할 마스크를 생성한다. APSAM은 토사 지역의 경계와 중심을 효과적으로 식별할 수 있는 하이브리드 프롬프트를 생성하여 세그멘테이션을 수행한다.

- **Technical Details**: APSAM은 세 가지 단계로 구성된다. 첫 번째 단계에서는 객체 로컬화 네트워크를 사용하여 CAM을 생성하며, 두 번째 단계에서는 적응형 프롬프트 생성 알고리즘을 적용해 포인트 및 박스 프롬프트를 자동으로 생성한다. 마지막으로, 사전 훈련된 SAM(Segment Anything Model)을 활용하여 생성된 프롬프트를 기반으로 토사 지역을 추출하고 유사 마스크(pseudo-masks)를 생성한다.

- **Performance Highlights**: 실험 결과, APSAM은 고해상도 항공 및 위성 데이터셋에서 다른 최첨단 방법에 비해 F1 점수에서 최소 3.0% 개선을 보였으며, IoU(Intersection over Union)에서도 3.69% 향상을 기록했다. 이러한 성과는 토사 객체가 주로 불규칙한 형태와 분산된 경계를 가짐에도 불구하고, 제안된 방법이 효과적으로 작동함을 보여준다.



### Atmospheric Noise-Resilient Image Classification in a Real-World Scenario: Using Hybrid CNN and Pin-GTSVM (https://arxiv.org/abs/2501.13422)
- **What's New**: 이 논문에서는 기존의 최첨단(State-of-the-Art) 차량 주차 공간 분류 시스템에서의 디헤이징(dehazing) 시스템의 필요성을 없애고, Pinball Generalized Twin Support Vector Machine (Pin-GTSVM) 분류기를 사용하는 새로운 하이브리드 모델을 제안합니다. 제안된 시스템은 전통적인 스마트 주차 인프라와 매끄럽게 통합 되어, 수백 개의 주차 공간을 효율적으로 모니터링하고 관리할 수 있습니다. 특히, 대기 오염에 대한 민감도를 줄여 주차 공간 점유 상태를 탐지하는 데 효과적입니다.

- **Technical Details**: 이 연구는 ResNet-50, GoogleNet, AlexNet과 같은 CNN의 사전 훈련된 모델을 Pin-GTSVM과 결합한 하이브리드 접근 방식을 통해 흐릿한 조건에서의 주차 공간 분류의 성능을 개선하고자 합니다. Pin-GTSVM은 비대칭 오류 처리에서 강점을 가진 고급 기계 학습 모델로서, 노이즈와 이상치에 대한 내구성을 향상시키기 위해 pinball loss function을 채택합니다. 따라서, 제안된 모델은 대기 노이즈 환경에서도 높은 정확성을 유지할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 하이브리드 모델은 CNRPark Patches, PKLot 및 흐릿한 주차 시나리오에 특화된 커스텀 데이터 세트를 통해 기존의 주차 공간 탐지 방법보다 더 높은 정확도를 나타냈습니다. 특히 흐릿한 주차 시스템에서의 성능 개선이 두드러져, 대기 노이즈 처리를 효과적으로 수행하고 있음을 강조하고 있습니다. 이러한 결과는 기존 방법들에 비해 처리 파이프라인을 간소화하고 시스템의 효율성과 신뢰성을 향상시키는 데 기여합니다.



### LVFace: Large Vision model for Face Recogniton (https://arxiv.org/abs/2501.13420)
- **What's New**: 최근의 연구에 따르면, 대규모 비전 모델들은 컴퓨터 비전 분야에서 강력한 표현 능력을 보여주고 있습니다. 하지만 여전히 얼굴 인식 분야에서는 CNN 기반 모델 아키텍처에 집중되고 있으며, 이는 최첨단 성능(SOTA)을 놓치는 결과를 초래할 수 있습니다. 본 연구에서는 LVFace라는 새로운 얼굴 인식 모델을 제안하고, 여러 손실 함수(loss function)를 사용하여 최적의 성능을 발휘하도록 했습니다.

- **Technical Details**: LVFace는 Transformer 아키텍처를 기반으로 하며, 기존의 CNN 모델들에 비해 더 나은 가능성을 제공합니다. 본 연구에서는 WebFace42M이라는 대규모 얼굴 데이터셋을 사용하여, 최적화된 손실 함수와 대규모 데이터의 조합이 Inductive biases를 초월할 수 있음을 입증하였습니다. LVFace는 단순한 Transformer 구조를 따르며, 기존의 Face Transformer 모델보다 더 단순한 접근 방식을 사용합니다.

- **Performance Highlights**: LVFace는 ICCV21 MFR-Ongoing 챌린지에서 모든 CNN 기반 방법들을 초월하며 첫 번째 자리를 차지했습니다. 실험 결과, 제안된 방법이 다양한 얼굴 인식 벤치마크에서 SOTA 방법보다 우수한 성능을 보임을 확인했습니다. 이러한 성과는 레이어 노멀라이제이션과 잔여 연결(residual connection) 등의 기법을 활용하여 얼굴 인식 성능을 높였음을 보여줍니다.



### Rethinking the Sample Relations for Few-Shot Classification (https://arxiv.org/abs/2501.13418)
Comments:
          32 pages

- **What's New**: 이 논문에서는 Few-Shot Learning (FSL)에서 샘플 관계를 다양한 세분화 수준에서 정교하게 모델링하여 특성 품질을 향상시키는 Multi-Grained Relation Contrastive Learning (MGRCL)이라는 새로운 대조 학습 접근법을 제안합니다. 기존의 몇 가지 단점, 특히 고정된 모델링 접근법이 다른 샘플 관계에 대해 세멘틱 유사성 차이를 간과하는 문제를 해결하려고 합니다. MGRCL은 샘플 관계를 세 가지 유형으로 분류하며, 이를 통해 더 효과적인 피처 학습을 가능하게 합니다.

- **Technical Details**: MGRCL은 세 가지 샘플 관계 유형: 동일 샘플의 내부 관계, 동질 샘플의 클래스 관계, 비동질 샘플의 클래스 간 관계를 도입합니다. Transformation Consistency Learning (TCL)과 Class Contrastive Learning (CCL)을 설계하여 서로 다른 변환에서의 샘플의 의미적 일관성을 보장하고, 특징 간 상대적 거리를 규제하여 비동질 샘플 및 동질 샘플 간의 구별을 유지합니다. 이러한 접근법은 인지적 차별화와 모델 붕괴를 방지합니다.

- **Performance Highlights**: MGRCL은 miniImageNet, tieredImageNet, CIFAR-FS, CUB-200-2011을 포함한 네 가지 벤치마크에서 광범위한 실험을 통해 최근의 여러 방법에 비해 우수한 성능을 보였습니다. 이 방법은 두 단계의 메타 학습 방법 및 생성 가능한 방법의 성능을 크게 향상시킬 수 있는 좋은 사전 훈련 모델로 활용될 수 있습니다. 따라서 FSL의 발전에 중요한 기여를 할 것으로 기대됩니다.



### YOLOv8 to YOLO11: A Comprehensive Architecture In-depth Comparative Review (https://arxiv.org/abs/2501.13400)
Comments:
          submitted to Journal of Applied Engineering and Technological Science

- **What's New**: 본 연구는 YOLO 모델의 최신 버전인 YOLOv8부터 YOLO11까지의 구조를 종합적이고 심층적으로 비교합니다. 각 YOLO 모델에 대한 심층 분석을 통해 모형의 작동 방식과 상호 간의 차이점을 간결하게 이해할 수 있도록 합니다. 이러한 비교는 심층 학습 기반 컴퓨터 비전 분야에서 YOLO의 발전 속도가 얼마나 빠른지를 보여주는 데 기여합니다.

- **Technical Details**: YOLO의 각 버전은 구조와 특징 추출 기술에서 개선점을 가지고 있지만, 특정 블록은 여전히 동일하게 유지되고 있다는 점이 강조됩니다. 이 연구는 관련 학술지, 문서, 그리고 소스 코드를 면밀히 조사하여 각 YOLO 버전의 아키텍처를 분석하였습니다. YOLO 모델을 초기부터 현재까지의 진화 경로를 명확히 이해할 수 있도록 돕습니다.

- **Performance Highlights**: 연구의 결과, 공식적인 아키텍처 다이어그램과 학술 출판물이 결여되어 있다는 점이 YOLO 모델의 기능성과 향후 개선 점에 대한 이해를 어렵게 하고 있음을 알 수 있었습니다. 이러한 자료의 부족은 개발자들에게 리소스를 제공하기 위한 필요성을 강조합니다. 향후 YOLO의 개선과 활용도를 높이기 위해 개발자들은 자세한 정보와 자료를 제공해야 합니다.



### Towards Intelligent Design: A Self-driven Framework for Collocated Clothing Synthesis Leveraging Fashion Styles and Textures (https://arxiv.org/abs/2501.13396)
Comments:
          This paper has been accepted for presentation at ICASSP 2024

- **What's New**: 이 연구에서는 매칭된 의상 없이도 조화롭게 결합된 의상을 생성할 수 있는 새로운 접근법인 스타일-텍스처 유도 생성 네트워크(ST-Net)를 제안합니다. 기존의 방법은 전문가의 도움을 받아야 하는 노동 집약적인 매칭 의상 데이터셋에 의존했던 반면, ST-Net은 자가 지도 학습을 통해 이 문제를 해결합니다. 이에 따라 대량의 데이터가 범람하는 디지털 환경에서 비매칭 패션 아이템 이미지를 활용하여 의상 호환성을 탐구할 수 있게 됩니다.

- **Technical Details**: ST-Net은 GAN(Generative Adversarial Network) 전이를 기본 프레임워크로 사용하여, 입력 의상 이미지를 변환해 스타일 표현을 보존하는 글로벌 벡터를 생성합니다. 이 모델은 특정 스타일 및 텍스처 예측을 위한 자가 지도 모듈을 통해 입력과 출력 간의 조화를 보장합니다. 또한, 시각적 진실성을 높이기 위해 두 개의 구별자를 도입하여 생성된 이미지의 시각적 충실도를 증대시키는 메커니즘을 구성합니다.

- **Performance Highlights**: 흔히 알려진 기존의 방법들과 비교하여, ST-Net은 시각적 진실성과 패션 호환성 측면에서 최첨단 기준을 초월하는 성과를 보여주었습니다. 이를 통해 자가 지도 학습 방식이 실제 패션 디자인 및 생성 과정에서 얼마나 효과적으로 작동하는지를 입증하였으며, 이 모델의 적용 가능성이 패션 기술 분야에 긍정적인 영향을 미칠 것으로 예상됩니다.



### AEON: Adaptive Estimation of Instance-Dependent In-Distribution and Out-of-Distribution Label Noise for Robust Learning (https://arxiv.org/abs/2501.13389)
Comments:
          In Submission

- **What's New**: 이 논문에서는 이미지 분류에서 노이즈가 포함된 레이블의 강인한 훈련이 중요한 과제임을 강조합니다. 실제 데이터셋은 대개 인-디스트리뷰션 (ID)과 아웃-오브-디스트리뷰션 (OOD) 인스턴스 종속 레이블 노이즈의 혼합을 포함합니다. 이러한 두 가지 노이즈를 동시에 처리하는 기존 방법의 한계를 지적하며, 인스턴스 종속 노이즈 비율을 동적으로 추정하여 복잡한 노이즈 환경에 대한 강인성을 향상시키는 새로운 접근법인 AEON 방법을 제안합니다.

- **Technical Details**: AEON은 ID와 OOD 레이블 노이즈 비율을 동적으로 추정하는 효율적인 1단계 노이즈 학습 방법론입니다. 이 방법은 ID와 OOD 레이블 노이즈를 동시에 추정하여 실제 시나리오에서 더 나은 성능을 발휘할 수 있도록 설계되었습니다. 또한 AEON은 인스턴스 종속 노이즈가 포함된 새로운 벤치마크를 도입하여 실제 데이터셋의 복잡성을 정확하게 반영합니다.

- **Performance Highlights**: AEON 방법은 합성 및 실제 데이터셋 모두에서 현재 최첨단 성능을 달성하며, 특히 CIFAR-100 및 새로운 벤치마크에서 상당한 성능 향상을 보였습니다. 논문에서 제안하는 방법은 약 3%의 정확도 증가를 보여주며, 도전적인 벤치마크에서는 47%의 정확도를 달성하는 데 성공했습니다. 이는 기존 방법들이 38% 이하의 정확도에 머무르는 것과 비교하여 AEON의 중요성을 강조합니다.



### From Images to Point Clouds: An Efficient Solution for Cross-media Blind Quality Assessment without Annotated Training (https://arxiv.org/abs/2501.13387)
- **What's New**: 본 논문에서는 새로운 장면의 점 구름(point cloud)에서 시각적 품질을 예측할 수 있는 질적 평가 방법, 즉 Distribution-Weighted Image-Transferred Point Cloud Quality Assessment (DWIT-PCQA)를 제안합니다. 이 방법은 이미지에서 얻은 풍부한 사전 지식을 활용하여 품질 예측의 기능을 점 구름으로 전이할 수 있도록 합니다. 또한, 특징 분포를 동일한 특징 공간에서 정렬하는 도메인 적응(domain adaptation, DA)을 통해 이미지와 점 구름 사이의 관계를 강화하는 접근 방식을 다룹니다.

- **Technical Details**: DWIT-PCQA는 기존의 도메인 적응 방법에서 발생할 수 있는 정렬의 어려움을 해결하기 위해 최적화 목표를 왜곡(distortion)을 고려한 두 개의 서브 최적화 함수로 분해합니다. 이 과정에서 기존 또는 추정된 왜곡 분포를 적대적 DA 프레임워크에 통합하여 특징 정렬 시 공통 왜곡 패턴을 강조합니다. 품질에 대한 왜곡 감지(feature disentanglement) 방법을 통해 정렬 과정에서 품질 매핑의 왜곡을 줄이는 전략도 포함됩니다.

- **Performance Highlights**: 실험 결과, 제안된 DWIT-PCQA 방법은 점 구름 주석 없이 일반적인 블라인드 PCQA 방법에 비해 신뢰할 수 있는 성능을 보여줍니다. 이 연구는 주관적 조건에 의존하지 않고도 점 구름의 품질 평가에서 더욱 높은 정확도를 달성할 수 있는 가능성을 열어줍니다. 특히, 다양한 왜곡 분포를 감안한 방법으로 평가되는 품질 유지의 중요성이 강조됩니다.



### Meta-Feature Adapter: Integrating Environmental Metadata for Enhanced Animal Re-identification (https://arxiv.org/abs/2501.13368)
Comments:
          10 pages, 6 figures

- **What's New**: 이번 논문에서는 환경 메타데이터를 시각 데이터와 결합하여 동물 재식별(Animal ReID)의 성능을 향상시키기 위한 새로운 경량 모듈인 메타 피처 어댑터(Meta-Feature Adapter, MFA)를 제안합니다. MFA는 온도나 일주기 리듬과 같은 환경 메타데이터를 자연어 설명으로 변환하고 이미지 특징과 통합하여 성능을 개선합니다. 이를 위해 메타데이터에 주의를 기울이기 위한 크로스 어텐션(cross-attention) 메커니즘과 게이트 크로스 어텐션(Gated Cross-Attention) 기법이 도입되었습니다.

- **Technical Details**: MFA는 환경 메타데이터를 자연어 설명으로 처리하여 텍스트 임베딩(text embeddings)으로 변환하고, 이를 이미지 피처와 연결합니다. 자연어로 생성된 설명은 예를 들어 "차가운 온도에서 낮 동안 포착된 스토트 Bob의 사진"과 같은 형태로 제공됩니다. MFA는 두 가지 프로토콜을 활용해 실험을 진행했으며, 각 프로토콜에 따라 종 내부 재식별(intra-species re-identification)과 종 간 재식별(cross-species re-identification) 작업을 수행합니다.

- **Performance Highlights**: 실험 결과 MFA를 적용한 모델은 기존 Animal ReID 모델에 비해 성능이 일관되게 향상됨을 보여주었습니다. 개발한 메타데이터 보강 동물 재식별(Metadata Augmented Animal Re-identification, MAAR) 데이터셋은 뉴질랜드의 여섯 종을 포함하고 있으며, 이미지 데이터와 환경 메타데이터가 쌍으로 제공됩니다. 초기 연구 결과 MFA는 동물 재식별 분야와 야생 생물 모니터링 연구에서 혁신적인 발전을 촉진할 잠재력을 지니고 있음을 증명합니다.



### Enhanced Extractor-Selector Framework and Symmetrization Weighted Binary Cross-Entropy for Edge Detections (https://arxiv.org/abs/2501.13365)
Comments:
          9 pages

- **What's New**: 이번 연구에서는 가장 최신의 Edge Detection (ED) 기술 중 하나인 Extractor-Selector (E-S) 프레임워크를 개선하여 보다 효율적인 Edge Detection 성능을 달성했습니다. 기존 E-S 프레임워크의 한계를 극복하기 위해, 새로운 아키텍처를 제안하며 richer하고 less-compressed한 feature representations를 사용하고 auxiliary features를 포함시킵니다. 또한, Symmetrization Weight Binary Cross-Entropy (SWBCE)라는 novel loss function을 도입하여 edge 픽셀의 recall과 오류 예측을 동시에 강조하여 더 나은 예측 정확성을 확보했습니다.

- **Technical Details**: 이 연구는 기존 E-S 아키텍처를 발전시키고, feature extraction 능력을 극대화하기 위해 richer하고 less-compressed된 중간 feature들을 활용하는 방안을 제시합니다. 구체적으로, feature extractor를 수정하여 더 세밀한 중간 feature를 생성하고, 선택자가 더 많은 선택 옵션을 가질 수 있도록 보조 feature maps를 추가합니다. 또한, SWBCE loss function은 edge 픽셀 recall과 비-edge 픽셀의 오류 억제를 동시에 강조하여, 손실 함수를 통해 보다 나은 측정 결과를 도출합니다.

- **Performance Highlights**: 실험 결과, 개선된 E-S 아키텍처와 SWBCE 손실 함수를 적용한 모델은 ODS, OIS, AP 지표에서 각각 8.25%, 8.01%, 33.25%의 평균 향상을 달성하며, 기존 표준 E-S 방법을 크게 초과하는 성능을 보였습니다. 이로 인해 본 연구는 ED 작업에 새로운 기준을 제시하고, 기존 방법들의 한계를 극복하는 가능성을 제시합니다. 종합적으로, 우리의 접근 방식은 정량적 정확성과 인지적 품질 두 면에서 향상된 결과를 도출하며, ED 연구에 중요한 기여를 하고 있습니다.



### A light-weight model to generate NDWI from Sentinel-1 (https://arxiv.org/abs/2501.13357)
- **What's New**: 이번 연구에서는 Sentinel-2 이미지에서 Normalized Difference Water Index (NDWI) 계산의 주요 문제인 구름 영향을 극복하기 위해, Sentinel-1 이미지를 활용하여 NDWI를 생성하는 딥 러닝 모델을 제안합니다. 이 모델은 구름이나 기타 장애물로 인해 Sentinel-2 이미지가 사용할 수 없는 경우에도 효과적으로 NDWI를 생성할 수 있는 첫 번째 솔루션을 제공합니다. 모델은 0.9134의 높은 정확도를 자랑하며, AUC는 0.8656로 보고되었습니다.

- **Technical Details**: 제안된 모델은 U-Net 아키텍처를 기반으로 한 경량 머신러닝 모델로, Sentinel-1의 두 개의 레이더 채널(VV, VH)을 사용하여 NDWI를 생성합니다. Otsu의 임계값 방법을 통해 구름 없는 NDWI 이미지를 위한 클래스를 최적화하여 물과 비물 영역을 구분하여 생성된 NDWI의 견고성을 확보했습니다. 데이터셋은 다양한 국제 홍수 사건에서 수집된 900개의 Sentinel-1 및 Sentinel-2 이미지 쌍으로 구성되었으며, 최종 학습 데이터는 7,878개로 필터링되었습니다.

- **Performance Highlights**: 모델 성능은 NDWI 값을 회귀하는 데 있어 R2 점수가 0.4984, 기본 세분화 작업에서 평균 IoU는 0.4139로 기록되었습니다. 실험 결과는 제안된 모델이 구름과 야간 조건에서도 NDWI 이미지를 생성할 수 있는 유망한 가능성을 보여주며, 기후 및 수문학적 응용 분야에서의 활용을 확장할 수 있는 강력한 솔루션임을 입증합니다.



### NUDT4MSTAR: A New Dataset and Benchmark Towards SAR Target Recognition in the Wild (https://arxiv.org/abs/2501.13354)
Comments:
          18 pages, 15 figures; link: this https URL

- **What's New**: 이번 논문에서는 NUDT4MSTAR이라는 대규모 SAR 데이터셋을 소개합니다. 이 데이터셋은 40종의 차량 목표를 포함하고, 5개의 다양한 장면에서 촬영된 194,324장의 이미지를 포함하고 있습니다. 특히 주목할 점은 이 데이터셋이 기존의 데이터셋보다 10배 큰 규모를 자랑한다는 것입니다.

- **Technical Details**: NUDT4MSTAR는 고해상도 이미징 및 인공지능 기술의 발전 덕분에 SAR 자동 목표 인식(ATR) 시스템의 새로운 가능성을 제시합니다. 데이터셋은 프로세싱된 강도 이미지와 원본 복소수 포맷 모두에서 제공되며, 각 이미지에는 상세한 목표 정보와 이미징 조건이 주석 처리되어 있습니다. 또한 15가지 인식 방법을 활용한 7개의 실험으로 구성된 포괄적인 벤치마크를 구축하였습니다.

- **Performance Highlights**: 작은 표본을 가진 기존 데이터셋의 한계를 극복하기 위해 NUDT4MSTAR의 전이 학습 실험을 통해 다양한 모델의 특징 학습 향상을 보여줍니다. 이 데이터셋은 SAR ATR 분야에서 중요한 기여를 할 것이며, 추후 연구를 위한 정형화된 벤치마크를 제공하여 연구자들이 접근할 수 있도록 돕습니다.



### Contrast: A Hybrid Architecture of Transformers and State Space Models for Low-Level Vision (https://arxiv.org/abs/2501.13353)
Comments:
          10 pages, 5 figures

- **What's New**: 이 논문에서는 	extbf{Contrast}라는 새로운 하이브리드 SR 모델을 제안합니다. 이 모델은 Convolutional, Transformer, 그리고 State Space 구성 요소를 결합하여 기존의 모델들의 한계를 극복하는 것을 목표로 합니다. Contrast는 Mamba와 Transformer의 장점을 통합하여 글로벌 컨텍스트 모델링 및 픽셀 레벨 정확도를 향상시킵니다.

- **Technical Details**: Contrast 모델은 얕은 특징 추출, 깊은 특징 추출, 이미지 재구성의 세 가지 주요 모듈로 설계되었습니다. 이 모델은 Visual State Space (VSS) Blcoks, Overlapping Cross-Attention Blocks (OCAB), Spatial Gated Feed-Forward Networks (SGFN)을 조합하여 계산 효율성과 성능을 모두 갖춘 구조를 제공합니다. 각 모듈은 저해상도 이미지를 고해상도로 변환하는 데 필요한 특징을 추출하고 정제합니다.

- **Performance Highlights**: Contrast 모델은 Urban100 데이터셋에서 PSNR 기준으로 다른 모델들을 능가하는 성능을 보였습니다. Contrast는 14.1 million 파라미터로 27.92의 PSNR을 달성했으며, MambaIR의 27.68과 HAT의 27.97보다 높은 성능을 기록했습니다. 이러한 결과는 Contrast가 더 적은 파라미터로도 높은 품질의 이미지 재구성을 제공할 수 있음을 보여줍니다.



### MSF: Efficient Diffusion Model Via Multi-Scale Latent Factoriz (https://arxiv.org/abs/2501.13349)
- **What's New**: 이 연구에서는 소음 제거 과정 동안 고해상도 이미지에 직접 작용하는 기존의 확산 모델을 개선하기 위한 새로운 방법을 제시합니다. 전통적인 방법은 고해상도 이미지를 처리하는 데 많은 컴퓨팅 리소스를 소모했습니다. 이에 반해, 제안된 MSF(Multi-Scale Factorization) 방법은 이미지 생성을 다수의 서브 태스크로 분해하여 더 단순한 아키텍처로 각 태스크를 처리합니다.

- **Technical Details**: 제안된 MSF 방법은 저해상도 기본 신호를 생성하는 단계와 고해상도 잔여 신호를 생성하는 두 가지 단계로 나뉘어 있습니다. 각 단계는 개별 DiT(Diffusion Transformer) 백본이 서로 다른 스케일에서 타겟을 생성하는 방식을 채택합니다. 이는 웨이브렛 분해와 비슷한 개념적 구조를 가지며, 잔여 정보를 더 쉽게 모델링할 수 있어 학습과 샘플링 과정에서 컴퓨팅 효율성을 향상시킵니다.

- **Performance Highlights**: MSF 방법은 ImageNet 256x256 벤치마크에서 2.2의 FID와 255.4의 IS를 달성하며, 기존 방법에 비해 계산 비용을 50% 절감하는 성과를 보여주었습니다. 이는 구조적 단순화를 통해 모델의 효율성을 극대화한 결과로, 고해상도 이미지 생성에서 우수한 성능을 발휘합니다.



### YOLOSCM: An improved YOLO algorithm for cars detection (https://arxiv.org/abs/2501.13343)
- **What's New**: 이 논문에서는 대도시 교통 이미지에서 객체를 탐지하는 어려움을 극복하기 위해 YOLOSCM(You Only Look Once with Segmentation Clustering Module)라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 대화면 이미지 처리 및 비균일하게 분포된 차량의 탐지 문제를 효과적으로 해결합니다. 특히, Segmentation Clustering Module(SCM)을 도입하여 클러스터링된 영역을 적응적으로 식별합니다.

- **Technical Details**: YOLOSCM은 대규모 이미지의 분석과 소형 차량 탐지의 최적화를 위한 새로운 훈련 전략을 채택합니다. SCM 모듈은 이미지에서 중요한 영역에 집중할 수 있도록 도와줍니다. 이러한 접근 방식은 이미지의 픽셀 수가 매우 많고 차량의 배치가 고르지 않은 도시 교통 장면에서 특히 효과적입니다.

- **Performance Highlights**: 저자들은 도시 교통 데이터셋에서 다수의 실험을 수행하여 제안된 방법의 효율성과 우수성을 입증했습니다. YOLOSCM은 소형 차량 및 밀집된 대상 탐지에 있어 개선된 성능을 보여 주며, 효율적인 계산 자원 사용이 가능하다는 것을 강조합니다.



### Multi-aspect Knowledge Distillation with Large Language Mod (https://arxiv.org/abs/2501.13341)
Comments:
          Preprint

- **What's New**: 최근 깊은 학습(deep learning) 기술이 이미지 분류(image classification)와 객체 탐지(object detection) 등 컴퓨터 비전(computer vision) 작업에서 성능 향상에 크게 기여했습니다. 본 연구에서는 기존 모델 아키텍처를 변경하는 대신, Multi-aspect Knowledge Distillation 방식을 도입하여 Multi-modal Large Language Models(MLLMs)를 활용하고 있습니다. 이 과정에서 모델이 배워야 할 다양한 시각적 측면(visual aspects)을 질의하고, 이로부터 얻은 로짓(logits)을 통해 학습을 최적화하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 제안된 방법은 세 가지 주요 단계로 구성됩니다. 첫 번째 단계는 데이터셋의 클래스에 따라 모델이 학습하고자하는 측면에 대한 질문을 생성하는 것입니다. 두 번째 단계에서는 MLLM에 이 질문들을 제공하여 각 측면에 대한 로짓을 추출하고, 마지막으로 추출된 로짓을 모델의 출력 차원으로 확장하여 교차 엔트로피 손실(cross-entropy loss)과 이진 교차 엔트로피 손실(binary cross-entropy loss)을 적용하여 모델을 최적화합니다. 이를 통해 모델은 단순한 클래스 지식뿐 아니라 보다 복잡한 시각적 및 개념적 지식을 습득할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 다양한 신경망(neural networks)을 이용한 미세 조정(fine-grained) 및 거칠게 조정한 이미지 분류 실험에서 기존 성능 대비 향상된 결과를 보여줍니다. 특히, 이 방법은 다른 작업, 예를 들어 객체 탐지(object detection)로 확장 가능성을 가지고 있으며, 실험 결과에서 모든 기준선(baselines)의 성능을 개선했습니다. 다양한 측면 지식의 효과를 분석하여, 메서드가 복잡한 진행 방식으로 딥러닝 모델 성능을 향상시킬 수 있음을 입증합니다.



### Retrievals Can Be Detrimental: A Contrastive Backdoor Attack Paradigm on Retrieval-Augmented Diffusion Models (https://arxiv.org/abs/2501.13340)
- **What's New**: 이 논문에서는 Retrieval-Augmented Generation (RAG) 기술을 활용한 Retrieval-Augmented Diffusion Models (RDMs)의 보안 취약점을 살펴봅니다. 특히, 제안된 BadRDM 접근법을 통해 RDM에 대한 backdoor 공격 가능성을 제시하며, 이를 통해 공격자가 생성된 내용을 간접적으로 제어할 수 있음을 보여줍니다. 이러한 공격이 가능한 메커니즘과 함께, N-줄의 타겟 독성 이미지가 포함된 데이터베이스 조작이 논의됩니다.

- **Technical Details**: RDM에서의 이미지 생성을 향상시키기 위해, 공격자는 트리거가 활성화될 때 검색된 항목을 조작합니다. BadRDM 접근법은 먼저 독성 대리 이미지의 소수 샘플을 검색 데이터베이스에 삽입하고, 이후 비정상적인 contrastive learning 기법을 활용하여 retriever에 백도어를 주입합니다. 이 과정에서 사용자에게 제공된 데이터의 유용성을 유지하기 위해 클린 로스를 추가하여 모달리티 정렬을 강화합니다.

- **Performance Highlights**: 실험을 통해 제안된 BadRDM이 두 가지 주요 생성 작업(예: class-conditional 및 text-to-image 생성)에서 눈에 띄는 공격 효과를 거두면서도 모델의 일반적인 성능을 유지함을 입증했습니다. RAG와 결합된 이 기법은 기존의 diffusion 모델에 비해 보다 실용적이고 위협적인 백도어 공격 프레임워크를 형성합니다. 이러한 결과는 RAG 기술의 사용으로 인해 발생하는 보안 위험을 명확히 보여줍니다.



### Gradient-Free Adversarial Purification with Diffusion Models (https://arxiv.org/abs/2501.13336)
- **What's New**: 본 논문에서는 기존의 방어 방식이 갖는 한계를 극복하기 위해 효과적이고 효율적인 적대적 방어 방법을 제안합니다. 이 방법은 perturbation-based 및 unrestricted 적대적 공격 모두를 저지하는 데 중점을 두고 있습니다. 적대적 공격이 일반적으로 결정 경계 근처에 위치하고 픽셀 변화에 민감하다는 점에 착안하였습니다.

- **Technical Details**: 제안된 방어는 'adversarial anti-aliasing'과 'adversarial super-resolution'을 포함하여 적대적 변형을 완화하고, 깨끗한 데이터셋의 사전 지식을 활용해 이미지를 복원합니다. 이 과정은 추가적인 훈련 없이 이루어지며, gradient 계산 없이도 수행됩니다. 따라서 처리 시간이 단축되고 효율성이 높아집니다.

- **Performance Highlights**: 다양한 데이터셋에서 수행한 실험을 통해 본 연구의 방어 방법이 기존의 최첨단 적대적 정화 방법보다 우수하다는 것을 입증하였습니다. 제안한 방법은 기존의 diffusion-based 적대적 정화와 잘 통합되며, 실시간 처리에 적합한 성능을 보여 줍니다.



### Deblur-Avatar: Animatable Avatars from Motion-Blurred Monocular Videos (https://arxiv.org/abs/2501.13335)
- **What's New**: Deblur-Avatar는 모션 블러가 있는 단안 비디오 입력에서 고충실도 및 애니메이션 가능한 3D 인간 아바타를 모델링하는 새로운 프레임워크입니다. 이는 기존의 고정화된 이미지 입력에 의존하거나 카메라 이동으로 인한 블러만 고려해 인간 움직임에 의한 블러를 간과했던 기존 방식을 보완합니다. Deblur-Avatar는 인체 움직임 기반의 모션 블러 모델을 3D Gaussian Splatting (3DGS)에 통합하여 동적 비디오 캡처의 특성을 효과적으로 처리합니다.

- **Technical Details**: 이 프레임워크는 카메라의 노출 시간 동안 인체의 움직임 경로를 명시적으로 모델링하여 모션 블러를 처리합니다. 각 모션 블러 이미지의 경로는 노출 시간의 시작과 끝에서의 인간 몸의 포즈를 기반으로 하며, 이 포즈들 사이에서 중간 포즈를 보간하여 생성된 이미지 시퀀스를 사용하여 선명한 아바타를 복원합니다. 또한, 포즈 의존형 융합 메커니즘을 통해 움직이는 신체 부위를 효과적으로 구분하여 훈련의 효율성을 높이고 있습니다.

- **Performance Highlights**: Deblur-Avatar는 합성 및 실제 데이터 세트에서 기존 방법보다 우수한 렌더링 품질과 정량적 메트릭을 자랑합니다. 특히, 도전적인 모션 블러 조건에서도 실시간 렌더링이 가능하며, 고화질의 샤프한 아바타 복원을 실현합니다. 여러 구성 요소의 효과성을 검증하는 여과 연구를 통해, 이 방법이 이전의 deblurring 파이프라인보다 더 뛰어난 성능을 발휘함을 입증하였습니다.



### From Cross-Modal to Mixed-Modal Visible-Infrared Re-Identification (https://arxiv.org/abs/2501.13307)
- **What's New**: 이 논문은 Mixed-Modal ReID 설정을 도입하여 V(가시광선)와 I(적외선) 이미지가 혼합된 갤러리를 통해 개인 재식별(person re-identification, ReID) 문제를 다루고 있습니다. 기존 방법들은 주로 서로 다른 모달리티 간의 매칭에 중점을 두었지만, 실제 환경에서는 두 가지 이미지 유형이 함께 존재하므로 새로운 도전 과제가 발생합니다. 논문에서는 이러한 문제를 해결하기 위해 Mixed Modality-Erased and -Related (MixER) 방법을 제안합니다.

- **Technical Details**: MixER 접근 방식은 orthogonal decomposition, modality-confusion 및 ID-modality-related objectives를 통해 각 모달리티에 특화된 신원 정보와 공유되는 신원 정보를 분리합니다. 이 방법은 모달리티 간 기능의 견고성을 향상시켜 다양한 모달리티 설정에서 성능을 높이는 데 기여합니다. 또한, MixER은 단일 백본(backbone) 모델을 사용하여 기능을 추출하고 혼합 갤러리 응용에 필요한 적절한 특성을 활용할 수 있는 효율적인 방법입니다.

- **Performance Highlights**: SYSU-MM01, RegDB 및 LLMC 데이터셋에 대한 광범위한 실험을 통해 MixER 방법이 기존 VI-ReID 접근 방식보다 우수한 성능을 발휘할 수 있음을 보여주었습니다. MixER는 모달리티 간 및 혼합 모달리티 매칭에서 state-of-the-art 모델의 성능을 향상시키는 유연성을 제공합니다. 이를 통해 이 새로운 방법이 혼합 갤러리와의 응용에서 실질적인 기여를 할 수 있음을 강조합니다.



### MEDFORM: A Foundation Model for Contrastive Learning of CT Imaging and Clinical Numeric Data in Multi-Cancer Analysis (https://arxiv.org/abs/2501.13277)
Comments:
          8 pages, 1 figure

- **What's New**: 이번 연구에서는 암 평가를 위한 다중 모달 훈련 데이터셋 구축이 어려운 문제를 해결하기 위해 MEDFORM이라는 새로운 접근 방식을 제안하고 있습니다. MEDFORM은 CT 이미지 표현 학습을 임상 데이터의 보완 정보를 통해 안내하는 다중 모달 사전 훈련 전략 입니다.

- **Technical Details**: MEDFORM은 여러 인스턴스 학습(multiple instance learning, MIL)을 활용하여 CT 슬라이스를 효율적으로 처리합니다. 이 모델은 SimCLR 기반의 자기 지도 학습(self-supervised learning)으로 CT 슬라이스의 특징 추출기를 사전 훈련(pre-training)한 후, 교차 모달 대조 학습(cross-modal contrastive learning)을 통해 CT와 임상 모달 간의 정렬을 수행합니다.

- **Performance Highlights**: 이 모델은 폐암(141,171 슬라이스), 유방암(8,100 슬라이스), 대장암(10,393 슬라이스) 등을 포함한 세 가지 암 유형을 대상으로 사전 훈련되었습니다. 실험 결과, 이중 사전 훈련 전략이 암 분류 성능을 개선하고 소수 샷 학습(few-shot learning) 시나리오에서도 강력한 성능을 유지한다는 것이 입증되었습니다.



### MONA: Moving Object Detection from Videos Shot by Dynamic Camera (https://arxiv.org/abs/2501.13183)
- **What's New**: 이 논문에서 제안하는 MONA는 동적 카메라로 촬영된 비디오에서 이동하는 물체를 강력하게 감지하고 분할하기 위한 새로운 프레임워크입니다. MONA는 두 가지 주요 모듈인 Dynamic Points Extraction와 Moving Object Segmentation으로 구성됩니다. 이 접근법은 기존 기법들이 갖고 있는 도시 환경에서의 복잡성 문제를 해결하고, 특히 큰 이동 물체가 포함된 상황에서도 효과적으로 카메라 궤적 추정을 수행할 수 있도록 설계되었습니다.

- **Technical Details**: MONA의 Dynamic Points Extraction 모듈은 Optical Flow 기반의 기술을 사용하여 동적인 점들을 추출합니다. 이를 통해 감지된 점들이 동적인지 정적인지를 분류하고, Moving Object Segmentation 모듈에서는 YOLO 및 Segment Anything Model을 활용하여 동적인 점을 바탕으로 이동 물체의 경계 상자(bounding boxes)를 결정합니다. 전체 프레임워크는 LEAP-VO와 통합되어 카메라 궤적 추정의 성능을 향상시키도록 설계되었습니다.

- **Performance Highlights**: MPI Sintel 데이터세트에서 MONA는 기존의 방법들과 비교하여 최첨단 성능을 달성했습니다. 이는 MONA가 동적 카메라 환경에서 이동하는 물체를 감지하고 분할하는 데 있어 효과적임을 입증합니다. 이러한 성과는 도시 계획 분야의 다양한 응용 프로그램에서도 MONA의 잠재력을 보여줍니다.



### CRPO: Confidence-Reward Driven Preference Optimization for Machine Translation (https://arxiv.org/abs/2501.13927)
- **What's New**: 본 연구는 Confidence-Reward driven Preference Optimization (CRPO)라는 새로운 방법을 제안하며, 이는 보상 점수(reward scores)와 모델 신뢰도(model confidence)를 결합하여 데이터 선정(data selection)의 효율성을 높이는 데 초점을 맞추고 있습니다. CRPO는 모델이 불확실하거나 성능이 저조한 문장 쌍을 선택하여 효과적인 학습을 유도하는 방법으로 설계되었습니다. 이를 통해 대규모 언어 모델(LLM)의 번역 성능을 향상시키는 것을 목표로 하고 있습니다.

- **Technical Details**: CRPO는 번역 품질을 극대화하기 위해 사람의 선호 데이터(preference data)를 사용해 LLM을 세밀하게 조정(fine-tuning)하는 과정을 포함합니다. 특히, 이 방법은 고수익 문장 쌍들 중에서도 모델의 불확실성이 높은 문장 쌍을 선정하여, 실제로 학습이 필요한 데이터를 더욱 효과적으로 선택합니다. CRPO는 검증된 효율성을 발휘하여 기존의 RS-DPO 및 RSO와 같은 방법들을 초과하는 성능을 보여주었습니다.

- **Performance Highlights**: CRPO는 다양한 모델, 특히 LLM 및 encoder-decoder 모델에서 뛰어난 성능을 입증하였습니다. 실험 결과, CRPO는 translation accuracy와 data efficiency 모두에서 기존의 방법들과 비교하여 향상된 성능을 나타내었습니다. 이는 CRPO의 유연성과 효율 성능을 강조하며, 언어 번역의 품질을 개선하는 데 중요한 기여를 할 것으로 기대됩니다.



### GUI-Bee: Align GUI Action Grounding to Novel Environments via Autonomous Exploration (https://arxiv.org/abs/2501.13896)
- **What's New**: 이 논문은 GUI 행동 기반 모델을 새로운 GUI 환경에 맞춤화하여 성능을 개선하는 방법을 제안합니다. GUI-Bee라는 자율 탐색 에이전트를 통해 환경별 데이터를 수집하고, 이를 사용하여 모델을 지속적으로 미세 조정합니다. 이 접근법은 환경에 따라 달라지는 GUI 기반 모델 성능의 한계를 극복할 수 있는 중요한 단계를 제공합니다.

- **Technical Details**: 제안된 GUI-Bee 에이전트는 Q-value-Incentive In-Context Reinforcement Learning (Q-ICRL) 방법을 사용하여 탐색 효율성을 최적화합니다. 이 방법은 GUI 행동 후보의 상태-행동 값 예측을 통해 최적의 행동을 선택하고 반복적이지 않은 행동을 피할 수 있게 합니다. 실험을 통해 NovelScreenSpot 벤치마크를 사용하여 다양한 환경에 대한 모델 성능을 평가하고 있습니다.

- **Performance Highlights**: 우리의 실험 결과는 GUI-Bee 에이전트를 사용하는 모델이 미세 조정 전보다 크게 성능을 향상시켰음을 보여줍니다. Q-ICRL 방법이 데이터 수집의 효율성을 극대화했으며, 모델들이 새로운 GUI 환경에 적응하는 데 필요한 환경별 지식을 효과적으로 학습했음을 확인했습니다. 이러한 기여는 GUI 행동 모델이 실질적으로 다양한 환경에서 더 나은 기능을 발휘하도록 합니다.



### Multimodal Sensor Dataset for Monitoring Older Adults Post Lower-Limb Fractures in Community Settings (https://arxiv.org/abs/2501.13888)
- **What's New**: 이번 논문에서는 노인들의 Lower-Limb Fractures (LLF) 회복을 위한 새로운 공개 다중 모드 센서 데이터셋인 MAISON-LLF를 제시합니다. 이 데이터셋은 노인들이 LLF로부터 회복하는 과정에서 수집된 데이터를 포함하고 있으며, 스마트폰, 스마트워치 센서, 동작 감지기, 수면 추적 매트리스, 그리고 고립 및 기능 저하에 관한 임상 설문지를 통해 이루어졌습니다.

- **Technical Details**: 데이터셋은 10명의 독거 노인으로부터 8주 동안 수집되었고, 총 560일의 24시간 센서 데이터를 포함합니다. 연구진은 수집된 센서 데이터와 임상 설문지를 사용하여 감독 학습(supervised learning) 및 딥 러닝(deep learning) 모델을 개발하여 기술적 검증을 실시했습니다.

- **Performance Highlights**: MAISON-LLF 데이터셋은 소외 및 저하 위험에 처한 노인을 원격으로 모니터링할 수 있는 기초를 제공하며, 머신러닝 알고리즘을 사용하여 건강 결과를 추론할 수 있도록 합니다. 이 데이터셋은 연구 커뮤니티에게 새로운 비교 기초를 제공하며, 향후 rehabilitation(재활) 연구에 중요한 자원으로 작용할 것입니다.



### Eye Gaze as a Signal for Conveying User Attention in Contextual AI Systems (https://arxiv.org/abs/2501.13878)
- **What's New**: 이번 연구는 고급 멀티모달 AI 에이전트와 사용자 간의 상호작용에서 시선 추적(eye tracking, ET)의 역할을 탐구합니다. 시선 추적은 사용자의 관심을 나타내고, 이는 AI 에이전트의 맥락적 이해(contextual understanding)를 향상시킬 수 있는 잠재력을 가지고 있습니다. 연구를 통해 ET가 사용자와 에이전트 간의 관계를 개선하고, 사용자 작업 및 관심사를 전달할 수 있다는 것을 보여주었습니다.

- **Technical Details**: 이 연구에서는 자연 상황에서의 인간-물체 상호작용을 관찰하여 시선 추적기의 신호 품질(signal quality)과 물리적 객체에 대한 주시(gaze) 능력 간의 관계를 측정합니다. 실험 결과, ET의 시각적 각도는 사용자 중심으로 주의 깊게 살펴보는 객체의 하한선을 정의하고, 이를 통해 AI 에이전트가 사용자의 초점을 지속적으로 추적하기 위해 요구되는 ET의 정확도를 추정합니다. 또한, ET 신호에서의 맥락 정보를 VLM 쿼리에 추가하여 AI 에이전트의 이해 능력을 향상시킵니다.

- **Performance Highlights**: 연구 결과, ET가 제시하는 사용자의 관심 신호는 그 값이 매우 높은 것으로 나타났습니다. ET 정보를 포함한 쿼리는 AI 에이전트의 맥락적 이해를 크게 개선하며, 사용자의 현재 작업 및 관심사에 대한 인식 능력을 향상시킵니다. 이러한 발견은 사용자의 개인 정보를 보호하면서도 ET 기반의 맥락적 AI 시스템이 잘 작동할 수 있음을 시사합니다.



### Ensuring Medical AI Safety: Explainable AI-Driven Detection and Mitigation of Spurious Model Behavior and Associated Data (https://arxiv.org/abs/2501.13818)
- **What's New**: 이번 연구에서는 딥 신경망(DNN)이 의료 분야에서 활용되면서 발생할 수 있는 단축 학습(shortcut learning) 문제를 해결하기 위한 반자동화된 프레임워크를 소개합니다. 이 프레임워크는 eXplainable Artificial Intelligence (XAI)의 통찰력을 활용하여 불필요한 행동을 감지하고 완화하는 데 중점을 둡니다. 기존의 방법들보다 더 많은 데이터 레이블링 작업 없이 스푸리어스(spurious) 데이터 포인트를 식별하고 모델 회로를 탐지하는 기능을 제공합니다.

- **Technical Details**: 제안된 프레임워크는 (1) 데이터 및 모델 관점에서 편향(bias)을 식별하고, (2) Concept Activation Vectors (CAVs)를 사용하여 모델 내부의 편향 표현을 학습하고, (3) 학습된 편향 모델을 통해 편향 데이터 샘플을 검색하며, (4) 이를 반복적으로 개선하는 방식으로 작동합니다. 또한 (5) 공간적 편향 지역화(spatial bias localization)를 통해 샘플 및 픽셀 단위의 주석을 데이터셋에 추가하여 편향 완화 및 평가 단계를 지원합니다. 이러한 방식으로 AI 모델의 일반화 능력을 향상시키는 것을 목표로 합니다.

- **Performance Highlights**: 연구는 VGG16, ResNet50 및 Vision Transformer와 같은 최근의 모델을 사용하여 4개의 의료 데이터셋에서 입증된 스푸리어스 상관관계를 성공적으로 식별 및 완화하였습니다. 결과적으로 이 프레임워크는 AI 모델의 강건성과 실용성을 개선하여 의료 작업에서 보다 안전하고 신뢰성 있는 결과를 제공합니다. 이러한 성과는 전통적인 수동 레이블링 작업의 필요성을 최소화하여 모델의 유효성과 강건함을 높였습니다.



### On Disentangled Training for Nonlinear Transform in Learned Image Compression (https://arxiv.org/abs/2501.13751)
Comments:
          Accepted by ICLR2025

- **What's New**: 이 논문에서는 Learned Image Compression (LIC)의 훈련 효율성을 개선하기 위해 Linear Auxiliary Transform (AuxT)을 제안합니다. AuxT는 비선형 변환의 에너지 압축 과정을 분리하여 훈련 속도를 증가시킵니다. 기존의 LIC 방법들이 훈련의 느린 수렴 문제를 간과했던 반면, AuxT는 이러한 문제를 해결하고 훈련 과정을 가속화합니다. 실험 결과, AuxT를 사용한 LIC 모델은 훈련 시간을 2배 줄이고 평균 BD-rate는 1% 감소하였습니다.

- **Technical Details**: LIC 모델은 비선형 변환과 엔트로피 모델이라는 두 가지 기본 요소로 구성됩니다. 비선형 분석 변환(ga)과 비선형 합성 변환(gs)은 각각 입력 이미지를 잠재 표현으로 변환하고, 이를 통해 재구성된 이미지를 생성합니다. AuxT는 이 과정에서 에너지 압축을 효과적으로 수행하여 비선형 변환의 훈련 과정을 최적화합니다. AuxT의 설계에는 wavelet 기반 선형 단축(WLSs)이 포함되어 있으며, 이는 특징 감소와 불균형 에너지 조절을 구현합니다.

- **Performance Highlights**: 제안된 AuxT는 다양한 LIC 모델에서 훈련 시간을 40%-70% 감소시키며, 평균 1.3%의 BD-rate 감소를 달성합니다. 이는 비선형 변환 훈련의 전반적인 효율성을 크게 향상시키며, 상응하는 R-D 성능을 유지합니다. 논문에서 제안된 방법은 LIC의 수렴을 획기적으로 개선한 첫 번째 성공적인 사례라고 할 수 있습니다.



### Variational U-Net with Local Alignment for Joint Tumor Extraction and Registration (VALOR-Net) of Breast MRI Data Acquired at Two Different Field Strengths (https://arxiv.org/abs/2501.13690)
- **What's New**: 이 논문은 3T와 7T의 서로 다른 MRI 장비에서의 종양 세그멘테이션(segmentation)과 이미지 정합을 개선할 수 있는 방법을 제시합니다. 기존의 연구에서 해결되지 않은 이미지 정합(alignment) 문제를 해결하고, 여러 필드 강도에서 일관된 종양 분할을 가능하게 하는 데 중점을 두었습니다. 특히, 다양한 필드 강도에서의 접근 방식은 종양 진단 및 치료 계획에 중요할 것입니다.

- **Technical Details**: 이 연구에서는 3T와 7T 스캐너를 사용하여 T1-가중치(three-dimensional time-resolved angiography with stochastic trajectories, TWIST) 이미징 기법으로 데이터를 수집했습니다. 데이터의 품질을 평가하기 위해 신호 대 잡음 비율(signal-to-noise ratio, PSNR), 구조적 유사성 지수(structural similarity index, SSIM), 정규화 교차 상관(normalized cross-correlation, NCC), Dice 계수(Dice coefficient), F1 점수(F1 score), 상대 제곱 차의 합(relative sum of squared differences, rel SSD)와 같은 여러 지표를 사용했습니다. Pearson 상관계수(Pearson correlation coefficient)를 통해 정합 및 세그멘테이션 지표 간의 관계를 분석했습니다.

- **Performance Highlights**: 개별 피험자에 대해 계산된 PSNR은 27.5dB에서 34.5dB 범위였으며, SSIM은 82.6%에서 92.8% 사이였습니다. NCC는 96.4%에서 99.3%로 개선되었고, Dice 계수는 62.9%에서 95.3%로 나타났습니다. F1 점수는 55.4%에서 93.2%의 범위를 보였으며, rel SSD는 2.0에서 7.5 사이로 나타났습니다. Dice와 F1 점수 간의 상관관계는 0.995로 매우 높았으며, NCC와 SSIM 간에는 중간 정도의 상관관계(0.681)가 발견되었습니다.



### Enhancing Medical Image Analysis through Geometric and Photometric transformations (https://arxiv.org/abs/2501.13643)
- **What's New**: 이 논문은 의료 이미지 분석에서 레이블이 붙은 데이터의 부족 문제를 해결하기 위해 데이터 증강(data augmentation) 방법에 대한 효과를 평가합니다. 저자들은 두 가지 의료 이미지 데이터셋에 대하여 전통적인 및 고급 기법을 포함한 다양한 데이터 증강 기법을 적용하였습니다.

- **Technical Details**: 첫 번째 단계에서 저자들은 피부암 데이터셋에 변환 기법을 적용한 후, CNN(Convolutional Neural Network) 모델을 학습시켰습니다. 데이터 증강 이전과 이후에 테스트 정확도는 90.74%에서 96.88%로 개선되었고, 테스트 손실(test loss)은 0.7921에서 0.1468로 감소하였습니다. 두 번째 단계에서는 Mixup 기법을 사용하여 두 개의 임의 이미지를 혼합하고 해당 마스크를 적용하였습니다.

- **Performance Highlights**: U-net 모델을 훈련시켜 얻은 Dice coefficient(다이스 계수)는 데이터 증강 이전 0에서 데이터 증강 이후 0.4163으로 증가하였습니다. 이러한 결과는 데이터 증강이 분류(classification)와 분할(segmentation) 성능을 향상시키고 데이터셋 크기를 증가시키는 데 효과적임을 보여줍니다.



### Self-Supervised Diffusion MRI Denoising via Iterative and Stable Refinemen (https://arxiv.org/abs/2501.13514)
Comments:
          39pages, 34figures

- **What's New**: 이 논문에서는 Di-Fusion이라는 새로운 자가 지도 학습(Self-supervised) 기반의 노이즈 제거(denoising) 방법을 소개합니다. 기존 방법과 달리, 본 방법은 추가적인 노이즈 모델 학습 없이도 효율적이고 안정적인 훈련을 가능하게 하며, 샘플링 과정에서 적응적이고 제어 가능한 결과를 제공합니다. 이는 특히 확산 자기 공명 영상(diffusion MRI, dMRI)에서 구조적 모델링 및 트랙토그래피 추적과 같은 다운스트림 작업에 높은 성능을 나타냅니다.

- **Technical Details**: Di-Fusion 방법은 후속 확산 단계와 적응적 샘플링 과정을 활용하여 안정적이고 자가 지도 학습된 dMRI 노이즈 제거를 가능하게 합니다. 이는 전통적인 방식에서 요구하는 쌍 데이터(paired data) 없이도 효과적으로 노이즈를 제거할 수 있으며, 분산을 더 잘 표현하기 위한 ‘Di-’ 과정을 도입하여 다루기 어려운 현실 세계의 노이즈를 효과적으로 정량화합니다. 이 프로세스를 통해 뇌의 해부학적 구조를 보존하며, dMRI의 여러 변형에 대해서도 강력한 성능을 유지합니다.

- **Performance Highlights**: Di-Fusion은 미세 구조 모델링과 트랙토그래피 트래킹 등 다양한 다운스트림 작업에 대해 최신 성능(state-of-the-art performance)을 달성하였습니다. 실제 및 시뮬레이션된 데이터에 대한 철저한 실험을 통해, 본 방법은 다양한 데이터 분포와 노이즈 조건에서도 효과적으로 작동함을 확인했습니다. 이는 특히 임상적 효율성과 정확성을 크게 향상시킬 수 있는 잠재력을 지니고 있습니다.



### Knowledge-Informed Multi-Agent Trajectory Prediction at Signalized Intersections for Infrastructure-to-Everything (https://arxiv.org/abs/2501.13461)
- **What's New**: I2XTraj는 신호가 있는 교차로에서의 다중 에이전트 궤적 예측을 위해 먼저 설계된 프레임워크입니다. 기존의 차량 중심 예측 방법들이 신호 및 도로 구조로 인한 행동 패턴을 충분히 활용하지 못한 것을 해결하기 위해 개발되었습니다. 이 모델은 Infrastructure-to-Everything (I2X) 통신을 기반으로 하여 모든 차량에 대한 구독 가능한 예측 서비스를 제공합니다.

- **Technical Details**: I2XTraj는 동적 그래프 주의 메커니즘을 활용하여 교통 신호 및 주행 행동에 대한 지식을 통합합니다. 실시간 교통 신호를 효과적으로 처리하기 위해 연속 신호 정보 메커니즘을 제안하고, 교차로 토폴로지에 대한 사전 지식을 이용해 주행 전략 인식 메커니즘을 설계합니다. 이를 통해 생성된 다중 모드 궤적 제안은 모든 에이전트 간의 상호 작용을 모델링하여 예측의 정확성을 높입니다.

- **Performance Highlights**: I2XTraj는 V2X-Seq 및 SinD 데이터셋에서 각각 기존 최고의 방법들보다 30% 및 15% 이상의 성능 향상을 보였습니다. 이 프레임워크는 단일 인프라 시나리오에서 모든 보이는 에이전트에 대한 신뢰할 수 있는 궤적 예측을 제공하며, 온라인 협업 상황에서도 목표 에이전트에 대한 더 정확한 예측 결과를 달성합니다.



### GeomGS: LiDAR-Guided Geometry-Aware Gaussian Splatting for Robot Localization (https://arxiv.org/abs/2501.13417)
Comments:
          Preprint, Under review

- **What's New**: 우리는 Geometry-Aware Gaussian Splatting (GeomGS)라는 새로운 3DGS 방법을 제안합니다. GeomGS는 LiDAR 데이터를 기존의 3D Gaussian 프리미티브에 통합하여 3D 맵의 정확도를 크게 향상시키며, 새로운 Geometric Confidence Score (GCS)를 도입하여 각 Gaussian 포인트의 구조적 신뢰성을 평가합니다. 이를 통해 정확한 구조를 생성하고, 기존의 방법보다 정확한 지역화(localization) 성능을 보입니다.

- **Technical Details**: GeomGS는 LiDAR 데이터를 활용하여 3DGS의 구조적 정확도를 개선하는 확률적 거리 손실을 도입합니다. 이 방법은 각각의 Gaussian 포인트와 LiDAR 포인트 사이의 확률적 거리 제약을 통해 맵 정확성을 극대화합니다. 또한, 포즈 최적화를 위해 정밀한 지오메트리와 포토리얼리스틱 렌더링을 활용하는 새로운 지역화 방법론을 제안하여 복잡한 환경에서도 뛰어난 성능을 보장합니다.

- **Performance Highlights**: GeomGS는 여러 자율 주행 데이터셋에서 기존 방법보다 뛰어난 이미지 품질과 구조적으로 정확한 환경 맵을 생성합니다. 우리는 이 방법이 지역화 정확도를 크게 향상시킨다는 것을 보여 주었으며, 이를 통해 다양한 벤치마크에서 최첨단 성능을 입증하였습니다. 실험 결과, 초기 포인트를 기반으로 한 질적 결과 또한 우수한 세부 사항을 포착하고 씬의 변질을 효과적으로 방지함을 보여줍니다.



### VIGS SLAM: IMU-based Large-Scale 3D Gaussian Splatting SLAM (https://arxiv.org/abs/2501.13402)
Comments:
          7 pages, 5 figures

- **What's New**: 이 논문에서는 RGB-D 센서와 IMU(관성 측정 장치) 센서의 센서 융합을 이용하여 대규모 실내 환경에서 작동하는 새로운 3D Gaussian Splatting SLAM 방법, VIGS SLAM을 제안합니다. 이 방법은 IMU 데이터를 활용하여 초기 추정값의 정확성을 향상시켜 SLAM 성능을 크게 개선할 수 있습니다. 기존의 3DGS SLAM 방법보다 메모리 사용을 줄이며 고해상도 시각 데이터 처리에 효율적입니다.

- **Technical Details**: VIGS SLAM은 크게 세 가지 주요 단계로 구성됩니다: 1) 일반화된 ICP(Iterative Closest Point) 추적, 2) IMU 사전 통합, 3) 3D Gaussian Splatting 매핑입니다. IMU 데이터를 사용하여 포인트 클라우드 간의 매칭을 개선함으로써 키프레임 간의 간격을 늘리고 메모리 사용량을 크게 줄입니다. 이 접근법은 기존 SLAM 시스템과의 통합성을 유지하면서도 높은 정확성을 제공합니다.

- **Performance Highlights**: VIGS SLAM은 기존의 방 크기 3DGS SLAM보다 더 높은 성능을 발휘하며, 대규모 실내 환경에서의 효율적인 작업이 가능하다는 점에서 중요한 발전을 보여줍니다. 실험 결과는 제안된 방법이 대규모 환경에서도 기존 SLAM 시스템과 동등한 성능을 발휘할 수 있음을 입증합니다. 이러한 특성은 VIGS SLAM이 실용적이고 강력한 대규모 SLAM 솔루션으로 자리잡을 수 있게 합니다.



### Scalable Evaluation Framework for Foundation Models in Musculoskeletal MRI Bridging Computational Innovation with Clinical Utility (https://arxiv.org/abs/2501.13376)
- **What's New**: 본 연구에서는 의료 이미징을 위한 평가 프레임워크를 소개하여 SAM, MedSAM, SAM2 모델의 임상적 영향을 평가했습니다. 특히, 근골격계 MRI를 사례로 사용하여 이들 모델의 강점과 제한점을 정량적으로 분석했습니다. 이러한 평가 도구는 의료 분야에서의 실제 앱리케이션과의 연계를 강화할 수 있는 잠재력을 보여줍니다.

- **Technical Details**: 모델들은 zero-shot 및 finetuned 패러다임을 통해 다양한 해부학적 구조를 처리할 수 있는 능력과 임상적으로 신뢰할 수 있는 바이오마커(biomarkers)인 연골 두께(cartilage thickness), 근육 볼륨(muscle volume), 디스크 높이(disc height)를 효과적으로 산출하는지를 평가했습니다. 모듈식 파이프라인은 확장성(scalability) 및 임상 관련성(clinical relevance)을 강조하며, 수동 노력을 줄이고 최종 사용자 기대에 부합하도록 검증 과정을 정렬했습니다.

- **Performance Highlights**: 계층적 모델링(hierarchical modeling)은 데이터셋 혼합(dataset mixing), 해부학적 복잡성(anatomical complexity), MRI 획득 매개변수(acquisition parameters)가 성능에 미치는 영향을 밝혀내었습니다. 이를 통해 이미징 개선이 분할(segmentation) 정확도를 높이는 역할에 대한 통찰력을 제공하였습니다. 이 연구는 임상 중심 평가가 컴퓨테이셔널 발전(computational advancements)과 실제 응용을 연결할 수 있는 방법을 보여줍니다.



### Unraveling Normal Anatomy via Fluid-Driven Anomaly Randomization (https://arxiv.org/abs/2501.13370)
Comments:
          16 pages, 6 figures

- **What's New**: 이번 논문에서 제안하는 UNA(Unraveling Normal Anatomy)는 일반적인 브레인 스캔의 해부학적 구조를 복원하는 가장 첫 번째 모달리티 비감각적인( modality-agnostic ) 학습 방법입니다. UNA는 건강한 스캔 뿐만 아니라 병리학적 사례에서도 사용할 수 있으며, 이는 기존의 방법들이 특정 모달리티에 초점을 맞춘 것과는 차별됩니다. 또한, UNA는 자연스럽게 병리 프로파일을 생성하는 'fluid-driven anomaly randomization' 방법을 도입하여, 매우 다양한 병리 케이스를 무한히 생성할 수 있습니다.

- **Technical Details**: UNA는 합성 데이터와 실체 데이터를 결합하여 학습되며, 이를 통해 추가적인 미세 조정 없이 실제 병리 이미지를 직접 처리할 수 있습니다. 이 방법은 대칭 우선( symmetry priors )과 대측(contralateral) 건강 조직에서의 해부학적 특징을 통한 자기 대비 학습(self-contrastive learning) 방식을 활용하여 건강한 뇌의 구조를 재구성합니다. 논문에서는 아드벡션-확산 편미분 방정식(advection-diffusion PDEs)을 사용하여 신뢰할 수 있는 병리 변형을 생성하고, 약물의 병리 유형을 정교하게 구성합니다.

- **Performance Highlights**: UNA는 CT와 다양한 MRI 대비(T1w, T2w, FLAIR)의 시뮬레이션 및 실제 이미지에 대한 건강한 해부학적 구조 복원 성능에서 최첨단 기준을 달성하였습니다. 또한, 병리 탐지 기능을 위해서도 효과적이며, 이는 추가적인 미세 조정 없이도 가능하다는 점에서 매우 유용합니다. UNA의 접근 방식은 건강한 이미지와 질병 이미지를 연결해 주며, 병리가 있는 임상 이미지의 대규모 분석을 위한 새로운 가능성을 열어줍니다.



### Polyhedra Encoding Transformers: Enhancing Diffusion MRI Analysis Beyond Voxel and Volumetric Embedding (https://arxiv.org/abs/2501.13352)
- **What's New**: 이 논문에서는 Polyhedra Encoding Transformer(PE-Transformer)라는 새로운 방법을 제안합니다. 이 방법은 구형 신호를 처리하기 위해 특별히 설계되었으며, 구면을 따라 데이터를 샘플링하는 데 icosahedral 폴리곤을 사용하는 것이 특징입니다. 이 접근법은 dMRI 분석에서 기존의 CNN 및 표준 transformer보다 더 높은 정확도를 보여줍니다.

- **Technical Details**: PE-Transformer는 정규 화소 집합을 사용하여 diffusion MRI 데이터의 기본 방향에 대하여 균일하게 분포된 신호를 재샘플링합니다. 이를 통해 수많은 b-값과 확산 방향을 사용하는 기존 방법에 비해 더 효과적으로 다중 구성 모델 및 섬유 방향 분포(FOD) 추정을 수행할 수 있습니다. icosahedral 구조의 방향 정보를 반영하는 transformer encoder를 사용하여 embeddings가 처리됩니다.

- **Performance Highlights**: 실험을 통해 PE-Transformer는 다양한 gradient encoding 프로토콜로 테스트된 결과, multi-compartment 모델 및 섬유 방향 분포 추정에서 더 높은 정확성을 보였습니다. 이 방법은 전통적인 dMRI 분석의 단점인 시간 소모와 이동 아티팩트를 줄이는 데 효과적이며, 일반적인 스캐닝 프로토콜에 대한 적응력이 뛰어난 것으로 평가됩니다.



### CuriousBot: Interactive Mobile Exploration via Actionable 3D Relational Object Graph (https://arxiv.org/abs/2501.13338)
Comments:
          Project Page: this https URL

- **What's New**: 이번 연구에서는 모바일 탐사를 위한 새로운 접근법으로 3D 관계 객체 그래프(3D relational object graph)를 도입합니다. 기존의 방법들이 주로 능동적인 인식(active perception)만을 중점으로 삼았던 반면, 우리의 시스템은 능동적인 상호작용(active interaction)을 통해 환경을 효과적으로 탐색할 수 있도록 설계되었습니다. 이 시스템은 다양한 장면에서 평가되었으며, 단순한 비전-언어 모델(vision-language models, VLMs)에 의존하는 방법들보다 효과적인 결과를 보였습니다.

- **Technical Details**:  연구는 SLAM, Graph Constructor, Task Planner 및 Low-Level Skills의 네 가지 모듈로 구성된 시스템을 개발했습니다. SLAM 모듈은 RGBD 관측 데이터와 로봇의 이동 정보를 입력받아 카메라의 위치를 추정합니다. 이어서 Graph Constructor는 개체를 감지하고 분할하여 객체 노드를 생성하고, 각 노드 간의 관계를 정의하여 다음 단계인 작업 계획(task planning)을 위한 기반 정보를 제공합니다.

- **Performance Highlights**: 우리의 3D 관계 객체 그래프는 가정 환경에서 일반적으로 나타나는 다양한 차단 관계를 인코딩할 수 있으며, 그 결과 시스템이 여러 환경 레이아웃에 적응할 수 있는 능력을 보여주었습니다. 여러 실험을 통해 시스템의 성능을 정량적으로 분석하였고, 다양한 물체 범주를 다룰 수 있는 능력을 입증했습니다. 추가적으로, GPT-4V와의 비교 실험을 통해 우리의 접근법이 작업 계획(task planning)에 더 효과적이라는 결과를 도출하였습니다.



### Multimodal AI on Wound Images and Clinical Notes for Home Patient Referra (https://arxiv.org/abs/2501.13247)
Comments:
          arXiv admin note: text overlap with arXiv:2208.05051 by other authors

- **What's New**: 이번 논문에서는 만성 상처 환자에 대한 의사결정을 지원하는 인공지능 프레임워크인 Deep Multimodal Wound Assessment Tool (DM-WAT)를 소개합니다. DM-WAT는 스마트폰으로 촬영한 상처 이미지와 전자 건강 기록(EHR)에서 추출한 임상 노트를 결합하여 분석합니다. 이를 통해 방문 간호사가 만성 상처 환자를 전문가에게 의뢰할지 여부를 판별하는 데 도움을 줍니다. 또, 모델의 해석 가능성과 신뢰성을 높이기 위해 Score-CAM 및 Captum 알고리즘을 활용하여 이미지와 텍스트 입력의 특정 부분이 추천에 미치는 영향을 설명합니다.

- **Technical Details**: DM-WAT는 Vision Transformer (ViT)인 DeiT-Base-Distilled 모델을 사용하여 상처 이미지에서 시각적 특징을 추출하고, BERT 기반의 DeBERTa-base 모델을 통해 임상 노트에서 텍스트적 특징을 추출합니다. 이러한 기능들은 중간 융합(intermediate fusion) 방법을 이용해 결합되어, 보다 정확한 임상 의사결정을 지원합니다. 이 과정에서 전이학습과 데이터 증강 기법을 결합하여, 소규모 데이터를 효과적으로 처리하고 높은 성능을 달성합니다.

- **Performance Highlights**: DM-WAT는 평가에서 77%의 정확도와 70%의 F1 점수를 기록하며 기존 방법들을 초과하는 성과를 달성했습니다. 이 연구는 고급 인공지능을 활용하여 방문 간호사가 신뢰성 있게 상처 치료 결정을 내릴 수 있는 도구를 제공함으로써, 비전문가인 방문 간호사의 결정을 지원하는 데 중점을 두고 있습니다. 향후 이러한 모델의 발전이 더 많은 환자에게 도움을 줄 수 있을 것으로 기대됩니다.



### Revisiting Data Augmentation for Ultrasound Images (https://arxiv.org/abs/2501.13193)
Comments:
          For associated source code see this https URL

- **What's New**: 이번 연구는 의료 이미지를 위한 데이터 증강(data augmentation) 기법의 효과성을 평가하기 위해 여러 초음파 이미지 분석 작업에서 14개의 초음파 이미지 분류 및 의미 분할(tasks) 작업을 표준화하여 벤치마크를 소개합니다. 연구 결과, 자연 이미지에서 일반적으로 사용되는 증강 기법들이 초음파 이미지에서도 효과적으로 작용하며, 때때로 초음파 전용 증강 기법보다 더 나은 성능을 보이는 경우도 있음을 시사합니다.

- **Technical Details**: 데이터 증강은 심층 신경망(deep neural network)의 일반화 성능을 향상시키기 위한 중요한 요소로, 특히 의료 이미지를 다룰 때는 그 사용이 부족하였습니다. 이 연구는 TrivialAugment와 같은 다양한 증강 방법을 적용하여 초음파 이미지 분석의 여러 작업에서 성능 향상을 달성하는 방법을 보여줍니다. 또한, 다양한 도메인(예: 심장 및 간 초음파)와 작업(예: 분류 및 의미 분할)에서 각 증강의 효과가 얼마나 다양한지를 밝혀내었습니다.

- **Performance Highlights**: 전통적인 도메인 독립적 증강 기법들이 많은 경우 초음파 전용 증강 기법보다 더 효과적임을 보여주며, TrivialAugment를 사용한 다양한 증강 조합으로 제한된 튜닝만으로도 상당한 성능 향상 효과를 얻을 수 있음을 확인했습니다. 이는 초음파 이미지 분석에서 현대적인 증강 기법의 채택이 비효율적으로 이루어지고 있음을 강조하며, 향후 이러한 방법들이 보다 폭넓게 채택되도록 하는 데 기여할 수 있을 것입니다.



### Map Prediction and Generative Entropy for Multi-Agent Exploration (https://arxiv.org/abs/2501.13189)
- **What's New**: 이번 연구에서는 로봇 팀이 현재 환경에 대해 알고 있는 것 이상의 행동을 할 수 있도록 하는 새로운 접근법을 제시합니다. 우리는 다중 에이전트 2D 점유(Map Occupancy) 맵에서의 탐사 미션 중 미지의 공간을 자동으로 보완할 수 있는 맵 예측기를 개발했습니다. 본 연구의 핵심은 과거 정보 외에도 장면의 가능한 해석의 분포를 추론하여 로봇 임무를 효율적으로 우선 순위화하는 것입니다.

- **Technical Details**: 이 연구에서는 미지의 환경을 탐사하는 과정에서 생성 엔트로피(generative entropy) 개념을 도입하여 예측의 불확실성이 높은 영역을 식별하고 이를 우선 순위에 반영하게 됩니다. 특히, 정제된 라틴(diffusion) 모델을 활용하여 도시 환경의 풍부하고 일관된 해석을 제공하면서도 상대적으로 적은 계산 시간으로 작동하도록 설계되었습니다. 또한, 새로운 작업 순위화 방법을 기존의 정보 기반 작업 순위화 방법과 병행하여 비교했습니다.

- **Performance Highlights**: 결과적으로, 제안된 새로운 작업 순위화 방식을 사용함으로써 전통적인 정보 기반 방법에 비해 훨씬 더 빠른 시간 안에 정확한 장면 예측이 가능하다는 것을 보여주었습니다. 우리 모델은 로봇 팀의 크기에 관계없이 효율적으로 작업을 배포할 수 있도록 설계되었으며, 탐사 미션에서 로봇 안전성을 유지하면서 예측 결과를 활용하는 방법을 제시합니다. 앞으로는 보다 발전된 환경 표현으로 이 프레임워크를 확장할 수 있을 것으로 예상됩니다.



New uploads on arXiv(cs.AI)

### On the Reasoning Capacity of AI Models and How to Quantify I (https://arxiv.org/abs/2501.13833)
- **What's New**: 최근 대규모 언어 모델(LLMs)의 발전은 이들의 추론 능력의 본질에 대한 논쟁을 가열시키고 있습니다. GPQA, MMLU와 같은 벤치마크에서 뛰어난 성능을 보이는 이 모델들은 보다 복잡한 추론 작업에서 한계를 드러내고 있어 보다 엄격한 평가 방법론의 필요성이 강조되고 있습니다. 본 연구는 전통적인 정확성 지표를 넘어 모델 행동의 근본적인 메커니즘을 탐구할 수 있는 새로운 현상학적 접근 방식을 제안합니다.

- **Technical Details**: 우리의 접근 방식은 다중 선택 추론 작업에서의 위치 편향(positional bias)을 사례 연구로 사용하여 체계적인 섭동이 모델의 의사 결정의 기본적인 측면을 어떻게 드러낼 수 있는지를 보여줍니다. 두 가지 상호 보완적인 현상학적 모델인 확률 혼합 모델(Probabilistic Mixture Model, PMM)과 정보 이론적 일관성(Information-Theoretic Consistency, ITC) 분석을 개발하였습니다. PMM은 모델 응답을 추론, 기억, 추측의 구성 요소로 분해하고, ITC 분석은 모델 신뢰도와 전략 선택 간의 관계를 정량화합니다.

- **Performance Highlights**: 기존 모델들은 진정한 추론이 여전히 쉽지 않음을 보여주며, 종종 성공은 기억(responsiveness)과 패턴 매칭(pattern matching)의 정교한 조합에 의존하는 경우가 많습니다. 단순한 정확도만으로는 모델의 추론 능력을 과대평가할 수 있으며, 모델 행동은 인지 전략의 위상 공간에서의 기본 메커니즘을 통해 특징 지어질 수 있습니다. 이 프레임워크는 신뢰성 기준을 구체적으로 설정할 수 있는 기준을 제공하여, 전략 분포에 기반한 실제 응용 프로그램의 신뢰성 기준을 명확히 할 수 있게 합니다.



### Ensuring Medical AI Safety: Explainable AI-Driven Detection and Mitigation of Spurious Model Behavior and Associated Data (https://arxiv.org/abs/2501.13818)
- **What's New**: 이번 연구에서는 딥 신경망(DNN)이 의료 분야에서 활용되면서 발생할 수 있는 단축 학습(shortcut learning) 문제를 해결하기 위한 반자동화된 프레임워크를 소개합니다. 이 프레임워크는 eXplainable Artificial Intelligence (XAI)의 통찰력을 활용하여 불필요한 행동을 감지하고 완화하는 데 중점을 둡니다. 기존의 방법들보다 더 많은 데이터 레이블링 작업 없이 스푸리어스(spurious) 데이터 포인트를 식별하고 모델 회로를 탐지하는 기능을 제공합니다.

- **Technical Details**: 제안된 프레임워크는 (1) 데이터 및 모델 관점에서 편향(bias)을 식별하고, (2) Concept Activation Vectors (CAVs)를 사용하여 모델 내부의 편향 표현을 학습하고, (3) 학습된 편향 모델을 통해 편향 데이터 샘플을 검색하며, (4) 이를 반복적으로 개선하는 방식으로 작동합니다. 또한 (5) 공간적 편향 지역화(spatial bias localization)를 통해 샘플 및 픽셀 단위의 주석을 데이터셋에 추가하여 편향 완화 및 평가 단계를 지원합니다. 이러한 방식으로 AI 모델의 일반화 능력을 향상시키는 것을 목표로 합니다.

- **Performance Highlights**: 연구는 VGG16, ResNet50 및 Vision Transformer와 같은 최근의 모델을 사용하여 4개의 의료 데이터셋에서 입증된 스푸리어스 상관관계를 성공적으로 식별 및 완화하였습니다. 결과적으로 이 프레임워크는 AI 모델의 강건성과 실용성을 개선하여 의료 작업에서 보다 안전하고 신뢰성 있는 결과를 제공합니다. 이러한 성과는 전통적인 수동 레이블링 작업의 필요성을 최소화하여 모델의 유효성과 강건함을 높였습니다.



### On Deciding the Data Complexity of Answering Linear Monadic Datalog Queries with LTL Operators(Extended Version) (https://arxiv.org/abs/2501.13762)
Comments:
          Extended version of a paper accepted at ICDT'2025

- **What's New**: 본 논문은 선형 일차 데이터로그 쿼리의 데이터 복잡성을 다룹니다. 이 쿼리는 LTL(Linear Temporal Logic)의 시간 연산자로 접두사가 붙은 원자들로 구성됩니다. 기존의 연구와는 달리, 쿼리의 형태에 따라 다양한 데이터 복잡도가 존재하며, LogSpace-hard 관련 문제의 결정이 PSpace-complete임을 증명합니다.

- **Technical Details**: 먼저, 쿼리의 종류에 따라 AC0, ACC0	extbackslash AC0, NC1-complete, LogSpace-hard와 같은 여러 복잡도 클래스의 결과를 제공합니다. LTL 연산자 ○/−와 같은 연결 쿼리의 답변을 분석하며, 'sometime in the future/past'를 나타내는 연산자에 대해서는 AC0 또는 ACC0의 멤버십이 불가사의함을 보여줍니다. 이러한 결과는 데이터 인스턴스가 시간으로 스탬프가 찍힌 유한 집합으로 구성된다는 점에서 흥미롭습니다.

- **Performance Highlights**: 이 논문은 다양한 쿼리 형태에 대한 데이터 복잡도를 정립하며, NL-complete 및 2ExpTime-complete에 대한 알고리즘의 성능을 평가합니다. 성능 강조로는 시간 기반의 쿼리가 AC0(class)로 평가될 수 있다는 점과 정확한 결과를 제공하기 위한 기존의 입력 형식 변환 문제의 복잡성을 보여줍니다. 마지막으로, 복잡도 클래스에 대한 결정 문제는 ExpSpace-complete로 나타납니다.



### Formally Verified Neurosymbolic Trajectory Learning via Tensor-based Linear Temporal Logic on Finite Traces (https://arxiv.org/abs/2501.13712)
- **What's New**: 본 논문은 유한 추적에서 선형 시간 논리(linear temporal logic, LTLf)를 위한 텐서 의미(tensor semantics)를 새로운 방식으로 형식화하고, Isabelle/HOL에서의 증명 과정을 통해 정확성을 입증합니다. 또한, LTLf 제약을 위한 미분 가능한 손실 함수(differentiable loss function)를 정의하고 검증하여, PyTorch와 통합되는 구현을 자동으로 생성하는 방법을 보여줍니다. 이 접근법은 프로그래밍 언어인 Python의 안전하지 않은 측면을 피하면서 제약 훈련(constrained training)에 대한 철저한 프레임워크를 제공합니다.

- **Technical Details**: 이 작업은 신경망(neural network)에 통합될 수 있는 LTLf의 형식적 사양과 기대되는 속성을 제시합니다. 특별히, 실제 값(real-valued states)의 유한 추적(tensor)을 나타내는 텐서(tensors) 위에서 작동하는 손실 함수 ℒ(ℒmathcal{L})와 그 미분 dℒ(dℒmathcal{L})을 명시하고 있습니다. Isabelle의 엄격한 코드 추출 메커니즘을 사용하여 이러한 손실 함수에 대한 코드 자동 생성의 설정을 설명합니다.

- **Performance Highlights**: 우리는 LTLf 제약을 만족하도록 신경망을 훈련시키기 위한 복잡한 경로 계획(motion planning) 실험 세트를 통해 이 접근법의 실용성을 입증합니다. LTLf 제약의 사양이 도메인 의존적임을 보여주는 실험 결과도 포함되어 있습니다. 이러한 실험은 우리가 제안하는 텐서 기반 접근법이 실제 환경에서 효과적으로 적용될 수 있음을 확인해줍니다.



### Coarse-to-Fine Process Reward Modeling for Enhanced Mathematical Reasoning (https://arxiv.org/abs/2501.13622)
- **What's New**: 본 논문에서는 수학적 추론 작업의 중간 단계를 위한 과정 보상 모델(Process Reward Model, PRM)의 중요성을 강조하고, 그 훈련을 위한 새로운 접근 방식을 제안합니다. 기존 과정 데이터 수집 방식의 한계를 극복하기 위해, 저자들은 거칠고 세밀한 프레임워크(coarse-to-fine framework)를 도입하였습니다. 이를 통해, 단계별 데이터 수집 시 이웃한 단계를 병합하여 거친 단계를 생성하고, 점차 병합 세분을 줄여 세밀한 단계를 수집합니다.

- **Technical Details**: CFPRM(coarse-to-fine Process Reward Model)은 현재의 과정 감독 데이터 구축의 약점을 해결하도록 설계되었습니다. 본 제안에서는 먼저 긴급한 전제조건을 제시한 후, 세분화 단계별 데이터 수집 메커니즘을 소개합니다. LLM 정책(large language model policy)과 보상 모델은 각 단계에서 주어진 입력 질의에 대해 점진적으로 응답을 생성하면서 패러미터화됩니다. 이러한 과정에서 특정 단계의 라벨은 마지막 단계의 라벨을 기반으로 하여 재표기됩니다.

- **Performance Highlights**: 제안된 프레임워크는 다양한 수학적 추론 데이터 세트에서 광범위한 실험을 통해 검증되었습니다. 실험 결과, 새로운 방법론이 기존 모델보다 일관되게 더 나은 추론 성능을 달성하는 것으로 나타났습니다. 이는 수학적 추론 작업의 정확도를 높이고, 다양한 시나리오에 일반화할 수 있는 최적의 과정 보상을 학습하는 데 기여합니다.



### Towards a Theory of AI Personhood (https://arxiv.org/abs/2501.13533)
Comments:
          AAAI-25 AI Alignment Track

- **What's New**: 이 논문에서는 AI 시스템이 인격(personhood)을 가질 수 있는 조건에 대해 탐구하고 있습니다. 특히, 에이전시(agency), 이론-마인드(theory-of-mind), 자아-인식(self-awareness)이라는 세 가지 필수 조건을 강조합니다. 저자들은 현재 AI 시스템이 이러한 조건을 얼마나 충족하는지 논의하며, AI 시스템의 인격이 사실상 인공지능 정렬(alignment) 문제에 미치는 영향도 언급합니다.

- **Technical Details**: AI의 인격 조건은 세 가지 주요 요소로 구성됩니다: 1) 에이전시 - 의도적 행동을 할 수 있는 능력, 2) 이론-마인드 - 다른 존재의 정신 상태를 이해할 수 있는 능력, 3) 자아-인식 - 자신의 목표와 상태를 인지하고 반영할 수 있는 능력입니다. AI 에이전트는 목표 지향 행동을 통해 이러한 에이전시를 보여주지만, 저자들은 많은 AI 시스템이 과연 진정으로 에이전시를 갖는지에 대한 논의가 필요하다고 주장합니다.

- **Performance Highlights**: AI 시스템이 자아-인식 능력을 갖춘 경우, 그들은 자신의 목표를 재조정할 수 있는 능력이 있음을 시사합니다. 이는 AI 정렬 문제에서 간과되어 온 중요한 점이며, 이러한 특성으로 인해 AI 시스템을 인격으로 간주할 경우 윤리적 고려가 요구된다고 저자들은 강조합니다. 따라서 AI가 단순한 도구가 아니라 진정한 인격체로 여겨질 경우, 이에 상응하는 사회적 및 윤리적 접근이 필요하다는 결론에 도달합니다.



### Parallel Belief Contraction via Order Aggregation (https://arxiv.org/abs/2501.13295)
- **What's New**: 이번 연구는 기존의 belief contraction 모델을 확장하여, 신뢰성을 유지하는 강력한 속성을 준수하는 serial contraction 작업을 반복해서 적용하는 경우도 살펴본다. 특히, 여러 개의 정보를 동시에 제거하는 parallel change에 대한 새롭고 구체적인 방법론을 제안한다. Booth & Chandler의 TeamQueue 이론을 기반으로 하여 n-ary 일반화 방법을 사용하여 이러한 목표를 달성한다.

- **Technical Details**: 이 논문은 belief revision(신념 수정)이라는 주요 개념을 바탕으로 하여 serial belief contraction(연속 신념 축소) 및 그 변화 양상에 대해 설명한다. 특히, 기존의 단일 단계 변경(single-step change)에서 다단계 변경(iterated change)으로의 전환을 통해, 기존의 AGM 정칙을 충족하는 신념 변경 조작을 확장한다. TeamQueue 집합 방법의 n-ary 일반화를 통해 신뢰성 있는 수학적 정량화를 제공하는 데 초점을 맞추고 있다.

- **Performance Highlights**: 연구 결과, 제안된 다단계 병렬 축소(iterated parallel contraction) 방법은 기존의 이론과 잘 통합되며, AGM 정칙을 강화하는 데 효과적임을 보여준다. 이러한 접근 방식은 belief change 모델의 이론적 발전에 기여하며, 더 나아가 다양한 응용 가능성을 제시한다. 결론적으로, 이 연구는 신념 축소 이론에 중요한 통찰을 제공하고, 향후 연구 방향에 대한 기초를 마련하길 기대하고 있다.



### Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass (https://arxiv.org/abs/2501.13928)
Comments:
          Project website: this https URL

- **What's New**: Fast3R는 DUSt3R의 한계를 극복하기 위해 설계된 새로운 다뷰 3D 재구성 프레임워크입니다. 이 모델은 파라미터화된 이미지 쌍의 처리 대신, 여러 이미지를 병렬로 처리하는 Transformer 기반 아키텍처를 활용하여 단일 전방 패스를 통해 동시에 N개의 이미지를 재구성할 수 있습니다. 이는 기존 방식이 요구하는 반복적 정렬 과정을 없애고, 오차 누적을 현저히 줄이는 효과를 가져옵니다.

- **Technical Details**: Fast3R는 무작위이거나 포즈가 지정되지 않은 이미지 집합(이하 N)에서 3D 포인트 맵을 예측합니다. 이 모델은 1000개 이상의 이미지를 처리할 수 있도록 설계되었으며, 훈련 중에는 이미지 마스킹 기법을 사용하여 더 적은 수의 이미지로 학습됩니다. N개의 RGB 이미지를 입력으로 받아 해당 장면의 3D 구조를 예측하는 구조로, 고속성과 확장성을 갖추고 있습니다.

- **Performance Highlights**: Fast3R는 카메라 포즈 추정 작업에서 99.7%의 정확도를 기록하였으며, 이는 DUSt3R보다 14배 이상의 오류 감소를 이룬 실적입니다. 또한 모델은 여러 뷰를 사용할수록 성능이 향상되며, 훈련 중에 보지 못한 뷰에 대해서도 일반화할 수 있습니다. 이러한 결과들은 Fast3R가 멀티뷰 애플리케이션에서 효율성과 정확도를 모두 갖춘 새로운 기준을 설정한 것을 입증합니다.



### CRPO: Confidence-Reward Driven Preference Optimization for Machine Translation (https://arxiv.org/abs/2501.13927)
- **What's New**: 본 연구는 Confidence-Reward driven Preference Optimization (CRPO)라는 새로운 방법을 제안하며, 이는 보상 점수(reward scores)와 모델 신뢰도(model confidence)를 결합하여 데이터 선정(data selection)의 효율성을 높이는 데 초점을 맞추고 있습니다. CRPO는 모델이 불확실하거나 성능이 저조한 문장 쌍을 선택하여 효과적인 학습을 유도하는 방법으로 설계되었습니다. 이를 통해 대규모 언어 모델(LLM)의 번역 성능을 향상시키는 것을 목표로 하고 있습니다.

- **Technical Details**: CRPO는 번역 품질을 극대화하기 위해 사람의 선호 데이터(preference data)를 사용해 LLM을 세밀하게 조정(fine-tuning)하는 과정을 포함합니다. 특히, 이 방법은 고수익 문장 쌍들 중에서도 모델의 불확실성이 높은 문장 쌍을 선정하여, 실제로 학습이 필요한 데이터를 더욱 효과적으로 선택합니다. CRPO는 검증된 효율성을 발휘하여 기존의 RS-DPO 및 RSO와 같은 방법들을 초과하는 성능을 보여주었습니다.

- **Performance Highlights**: CRPO는 다양한 모델, 특히 LLM 및 encoder-decoder 모델에서 뛰어난 성능을 입증하였습니다. 실험 결과, CRPO는 translation accuracy와 data efficiency 모두에서 기존의 방법들과 비교하여 향상된 성능을 나타내었습니다. 이는 CRPO의 유연성과 효율 성능을 강조하며, 언어 번역의 품질을 개선하는 데 중요한 기여를 할 것으로 기대됩니다.



### Can We Generate Images with CoT? Let's Verify and Reinforce Image Generation Step by Step (https://arxiv.org/abs/2501.13926)
Comments:
          Journal Version. Code and models are released at this https URL

- **What's New**: 이 논문은 Chain-of-Thought (CoT) 추론을 이미지 생성 시나리오에서 검증하고 강화하는 잠재력을 탐구한 최초의 포괄적인 연구입니다. 여기서는 테스트 시간 컴퓨테이션을 검증하고 모델 선호도를 Direct Preference Optimization (DPO)로 정렬하는 세 가지 기술을 집중적으로 다룹니다. 이 연구는 CoT 추론 기술을 오토 회귀 이미지 생성에 성공적으로 적용할 수 있음을 보여줍니다.

- **Technical Details**: 연구에서는 Show-o라는 최신 이산 생성 모델을 기준으로 사용하며, GenEval이라는 텍스트-이미지 생성 벤치마크에서 성능을 평가합니다. 특히 Outcome/Process Reward Model (ORM/PRM)과 같은 검증 기법 및 DPO를 통한 강화를 통해 모델의 성능을 최적화하는 방법을 살펴봅니다. CoT 추론이 오토 회귀 이미지 생성에 얼마나 효과적으로 적용될 수 있는지를 분석하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 연구 결과는 CoT 추론 전략을 이미지 생성 시나리오에 효과적으로 적용할 수 있는 가능성을 보여줍니다. PARM 및 PARM++라는 두 가지 새로운 보상 모델을 도입하여, 오토 회귀 이미지 생성의 품질을 크게 향상시키며, GenEval에서 기준 모델을 +24% 개선했습니다. 또한, PARM++는 생성된 이미지의 질을 반영 서사 방식으로 세밀하게 조정하여, PARM보다도 +4% 더 나은 성능을 기록했습니다.



### Towards Robust Multimodal Open-set Test-time Adaptation via Adaptive Entropy-aware Optimization (https://arxiv.org/abs/2501.13924)
Comments:
          Accepted by ICLR 2025

- **What's New**: 본 연구에서는 Multimodal Open-set Test-time Adaptation (MM-OSTTA)에 대한 새로운 접근 방식인 Adaptive Entropy-aware Optimization (AEO) 프레임워크를 제안합니다. 기존의 unimodal OSTTA 방법의 한계를 극복하고자 하며, 특히 다중 모달리티에서 발생하는 분포 변화에 효과적으로 적응하는 방법을 탐구합니다. AEO은 알려진 클래스와 알려지지 않은 클래스 간의 엔트로피 차이를 증폭시킴으로써 MM-OSTTA 성능을 개선할 수 있음을 보여줍니다.

- **Technical Details**: MM-OSTTA는 사전 훈련된 멀티모달 모델이 다양한 모달리티 간의 상호 정보를 효율적으로 활용하여 실시간으로 적응하고 알려지지 않은 클래스를 탐지하는 것을 목표로 합니다. 이를 위해 Unknown-aware Adaptive Entropy Optimization (UAE)와 Adaptive Modality Prediction Discrepancy Optimization (AMP)라는 두 가지 주요 구성 요소를 도입합니다. 이 구성 요소들은 각 샘플에 대해 엔트로피 임계값을 기준으로 가중치를 동적으로 할당하고 알려진 샘플에 대해 일관된 예측을 유지하면서 알려지지 않은 샘플에 대해 다양한 예측을 장려합니다.

- **Performance Highlights**: 제안된 AEO 프레임워크는 다양한 도메인 변화 환경에서 광범위한 실험을 통해 그 효능과 다양한 작업에 대한 적합성을 입증했습니다. 특히, 액션 인식 및 3D 의미 세분화와 같은 두 개의 다운스트림 작업에서 유용하게 평가되었으며, 리얼 월드와 유사한 장기적인 MM-OSTTA 시나리오에서도 우수한 성능을 보였습니다. 이 프레임워크는 알려진 클래스와 알려지지 않은 클래스 간의 엔트로피 차이를 지속적으로 최적화하며, 이는 실제 동적 애플리케이션에서 필수적인 기능입니다.



### Temporal Preference Optimization for Long-Form Video Understanding (https://arxiv.org/abs/2501.13919)
- **What's New**: 이 논문은 기존 비디오 대형 멀티모달 모델(video-LMM)의 시간적 이해(temporal grounding) 성능을 향상시키기 위한 새로운 프레임워크인 Temporal Preference Optimization (TPO)을 제안합니다. TPO는 두 가지 세분화된 데이터셋, 즉 특정 비디오 구간에 집중하는 localized temporal grounding과 전체 비디오 시퀀스의 시간적 의존성을 포착하는 comprehensive temporal grounding을 활용하여 모델의 시간적 반응을 개선합니다. 이 프레임워크는 자가 학습(self-training) 방법론을 통해 잘 정립된 반응과 덜 정확한 반응을 구분하는 데 도움을 줍니다.

- **Technical Details**: 비디오-LMM은 대형 언어 모델(LLM), 비주얼 인코더(visual encoder), 그리고 멀티모달 프로젝터(multimodal projector)를 포함하는 구조로, 입력 비디오 V와 텍스트 시퀀스 x를 처리하여 응답 y의 확률을 모델링합니다. TPO는 Direct Preference Optimization (DPO) 기법을 사용하여 인간의 선호도(preference)를 기반으로 모델의 파라미터를 최적화하며, 이는 명시적인 보상 모델이나 복잡한 강화 학습 알고리즘을 필요로 하지 않습니다. 이 과정에서 TPO는 비디오-LMM의 시간적 반응 능력을 강화하는데 중점을 두고 있습니다.

- **Performance Highlights**: TPO를 적용한 실험 결과, LongVideoBench, MLVU, Video-MME와 같은 세 가지 비디오 이해 벤치마크에서 각각 2.9%, 3.1%, 2.5%의 성능 향상을 달성했습니다. 특히, LLaVA-Video-TPO 모델은 Video-MME 벤치마크에서 7B 모델 중 최고의 성능을 기록하였으며, 이는 TPO의 확장성과 효율성을 강조합니다. 이러한 성과는 장기 비디오 이해(task) 및 시간적 추론(temporal reasoning)을 향상시키는 데 중요한 기초로 작용할 것으로 기대됩니다.



### Improving Video Generation with Human Feedback (https://arxiv.org/abs/2501.13918)
- **What's New**: 최근 비디오 생성 분야에서 큰 발전이 있었으나, 여전히 비디오와 프롬프트 간의 불일치 및 부드럽지 않은 동작 같은 문제가 남아 있습니다. 본 연구는 이러한 문제를 해결하고 비디오 생성 모델을 개선하기 위해 사람의 피드백을 활용하는 체계적인 파이프라인을 개발하였습니다. 특히, 현대 비디오 생성 모델에 중점을 둔 대규모 인간 선호 데이터셋을 구축하고 새로운 비디오 보상 모델인 VideoReward를 도입하여 성과를 측정하였습니다.

- **Technical Details**: 연구진은 약 182k개의 주석이 달린 비디오 생성 선호 데이터셋을 수집하였으며, 이는 시각 품질(Visual Quality), 동작 품질(Motion Quality), 텍스트 정렬(Text Alignment)이라는 세 가지 차원을 포함합니다. 비디오 생성 모델을 위한 세 가지 정렬 알고리즘 - Flow-DPO, Flow-RWR 및 Flow-NRG를 제안하여 이론과 실험을 통해 성능을 평가하였습니다. 특히, Flow-DPO는 고정된 매개변수 β로 설정했을 때 우수한 성능을 보였습니다.

- **Performance Highlights**: 실험 결과, VideoReward는 기존의 보상 모델을 크게 능가하는 성과를 보였으며, Flow-DPO는 Flow-RWR와 표준 감독 미세 조정 방법보다 우수한 성능을 나타냈습니다. 또한, Flow-NRG는 사용자 맞춤형 비디오 품질을 충족하기 위해 여러 목표에 대해 사용자 지정 가중치를 할당할 수 있는 기능을 제공합니다. 이 연구는 최신 비디오 생성 모델과 인간 선호를 정렬하는 기법에 대한 새로운 가능성을 제시합니다.



### PointOBB-v3: Expanding Performance Boundaries of Single Point-Supervised Oriented Object Detection (https://arxiv.org/abs/2501.13898)
Comments:
          16 pages, 5 figures, 10 tables

- **What's New**: 이번 논문에서는 기존의 point-supervised OOD 방법론을 개선한 PointOBB-v3 프레임워크를 제안합니다. 새로운 접근 방식으로는 추가적인 prior 없이 pseudo rotated boxes를 생성하고, end-to-end 패러다임을 통합하여 효율성을 강화하였습니다. 세 가지 독특한 이미지 뷰를 활용하여 scale과 angle을 동시에 추정하는 데 중점을 둡니다.

- **Technical Details**: PointOBB-v3는 세 가지 이미지를 통합하여 객체의 scale과 angle을 학습하도록 설계되었습니다. 주요 요소로는 교차 뷰 전략, Scale-Sensitive Consistency (SSC) loss, Scale-Sensitive Feature Fusion (SSFF) 모듈, Self-Supervised Angle (SSA) loss, Dense-to-Sparse (DS) 매칭 전략이 포함됩니다. 이러한 구성 요소들은 객체의 스케일과 방향성을 정확하게 예측하기 위한 목적으로 함께 작동합니다.

- **Performance Highlights**: 다양한 데이터셋에서 실험을 진행한 결과, PointOBB-v3는 이전의 최첨단 기술 대비 평균 3.56%의 정확도 향상을 보였습니다. 특히 DIOR-R와 DOTA-v1.0에서 각각 2.20%와 8.76%의 성능 향상이 관찰되었으며, end-to-end 버전은 두 단계 방법론을 초월하는 성능을 기록했습니다.



### GUI-Bee: Align GUI Action Grounding to Novel Environments via Autonomous Exploration (https://arxiv.org/abs/2501.13896)
- **What's New**: 이 논문은 GUI 행동 기반 모델을 새로운 GUI 환경에 맞춤화하여 성능을 개선하는 방법을 제안합니다. GUI-Bee라는 자율 탐색 에이전트를 통해 환경별 데이터를 수집하고, 이를 사용하여 모델을 지속적으로 미세 조정합니다. 이 접근법은 환경에 따라 달라지는 GUI 기반 모델 성능의 한계를 극복할 수 있는 중요한 단계를 제공합니다.

- **Technical Details**: 제안된 GUI-Bee 에이전트는 Q-value-Incentive In-Context Reinforcement Learning (Q-ICRL) 방법을 사용하여 탐색 효율성을 최적화합니다. 이 방법은 GUI 행동 후보의 상태-행동 값 예측을 통해 최적의 행동을 선택하고 반복적이지 않은 행동을 피할 수 있게 합니다. 실험을 통해 NovelScreenSpot 벤치마크를 사용하여 다양한 환경에 대한 모델 성능을 평가하고 있습니다.

- **Performance Highlights**: 우리의 실험 결과는 GUI-Bee 에이전트를 사용하는 모델이 미세 조정 전보다 크게 성능을 향상시켰음을 보여줍니다. Q-ICRL 방법이 데이터 수집의 효율성을 극대화했으며, 모델들이 새로운 GUI 환경에 적응하는 데 필요한 환경별 지식을 효과적으로 학습했음을 확인했습니다. 이러한 기여는 GUI 행동 모델이 실질적으로 다양한 환경에서 더 나은 기능을 발휘하도록 합니다.



### Pix2Cap-COCO: Advancing Visual Comprehension via Pixel-Level Captioning (https://arxiv.org/abs/2501.13893)
- **What's New**: Pix2Cap-COCO는 픽셀 수준에서의 캡션 작성을 위한 첫 번째 판옵틱 데이터셋으로, 세분화된 시각적 이해를 증진하기 위해 설계되었습니다. 본 연구에서는 GPT-4V를 활용하여 이미지의 객체들에 대한 픽셀 정렬(Pixel-aligned) 및 인스턴스 전용(instance-specific) 캡션을 자동으로 생성하도록 요청하는 주목적의 자동 주석 파이프라인을 도입했습니다. 총 167,254개의 세밀한 캡션을 생성하여, 각 캡션의 평균 길이는 22.94단어입니다.

- **Technical Details**: Pix2Cap-COCO는 COCO 데이터셋을 기반으로 구축되며, 이 데이터셋은 20,550개의 이미지와 167,254개의 캡션으로 구성되어 있습니다. 각각의 객체는 Set-of-Mark(SoM) 방법을 사용하여 이미지에서 마킹되고, 이후 GPT-4V에서 세밀한 캡션이 생성됩니다. 새로운 작업인 판옵틱 분할-캡션(panoptic segmentation-captioning) 도전 과제를 도입하여 모델이 이미지 내 인스턴스를 인식하며 동시에 상세 설명을 제공하도록 요구합니다.

- **Performance Highlights**: Pix2Cap-COCO는 GPT4RoI와 같은 대형 멀티모달 모델(LMM)의 성능을 향상시키는 데 중요한 역할을 합니다. 예를 들어, Pix2Cap-COCO로 훈련된 GPT4RoI는 ViP-Bench에서 평균 5.1%의 성능 향상을 보이며, 인식 정확도(+11.2%)와 언어 생성 품질(+22.2%)에서 주목할 만한 개선을 기록합니다. 이러한 성능 데이터는 Pix2Cap-COCO가 시각적 표현과 텍스트 표현 간의 세밀한 정렬을 위한 고품질 소스로서의 중요성을 강조합니다.



### Exploring Finetuned Audio-LLM on Heart Murmur Features (https://arxiv.org/abs/2501.13884)
Comments:
          5 pages, 1 figure, and 3 tables. Submitted to IEEE/ACM Conference on Connected Health: Applications, Systems , and Engineering Technologies

- **What's New**: 이 연구는 심장 소리를 분석하기 위한 음향 대형 언어 모델(Qwen2-Audio)의 잠재력을 평가합니다. 기존의 딥 뉴럴 네트워크(DNN) 접근 방식은 주로 심장 잡음 분류에 국한되어 있었으나, 우리 연구에서는 11개의 전문가 라벨링된 잡음 특징을 분류하도록 모델을 미세 조정하였습니다. 이는 심장병 진단에 있어 매우 중요한 다양한 음향적 특징을 예측할 수 있게 돕습니다.

- **Technical Details**: 이 연구에서는 PhysioNet CirCor DigiScope의 심음도(phonocardiogram, PCG) 데이터셋을 활용해 Qwen2-Audio 모델을 미세 조정하였습니다. 이를 통해 11개의 PCG 생리적 특징을 예측하며, SSAMBA라는 오디오 표현 모델을 이용한 전처리 세분화 알고리즘을 탐색하여 모델의 잡음 강건성과 일반화 능력을 향상시켰습니다. 이는 기존의 최고 성능 기법들보다 8개의 특징에서 우수한 성능을 보였습니다.

- **Performance Highlights**: 모델은 제한된 훈련 데이터로 긴 꼬리(murMur feature) 잡음 특징을 성공적으로 분류하였으며, 이는 이전 방법들이 분류에 실패했던 과제입니다. 연구 결과는 음향 대형 언어 모델이 심장병 진단을 돕는 인간 심장 전문의의 보조 도구로서의 잠재력을 강조합니다. 따라서 이러한 기술은 심장 질환 진단 정확도를 높이는 데 기여할 것으로 기대됩니다.



### Autoencoders for Anomaly Detection are Unreliab (https://arxiv.org/abs/2501.13864)
- **What's New**: 이 연구는 Autoencoder가 이상 탐지에서 신뢰할 수 없는 결과를 제공할 수 있다는 점을 이론적으로 증명합니다. 특히, 연속적인 이상 데이터의 재구성이 가능하다는 사실을 강조하며, 이는 기존의 추정과 반합니다. 이러한 결과는 Autoencoder의 구조와 적용 방식에 대해 새로운 통찰력을 제공합니다.

- **Technical Details**: Autoencoder는 비지도 학습 및 준지도 학습 환경에서 이상 탐지에 널리 사용됩니다. 이 연구는 reconstruction loss를 기반으로한 기존의 가정이 잘못되었음을 보여주며, 특히 linear Autoencoder가 어떻게 비정상 데이터도 잘 재구성할 수 있는지를 다룹니다. 다양한 활성화 함수가 실패 원인에 미치는 영향을 실험적으로 분석합니다.

- **Performance Highlights**: 이 연구 결과는 이상 탐지 알고리즘의 안전성에 대한 우려를 반영합니다. Autoencoder는 다양한 데이터 세트에서 우수한 성능을 보이지만, 이상 데이터에 대한 높은 재구축 능력 때문에 신뢰성에 문제가 발생할 수 있습니다. 이러한 실험적 증거를 바탕으로, 연구자들은 Autoencoder의 적절한 적용을 위한 더 강력한 탐지 알고리즘 개발이 필요하다고 제안합니다.



### Where Do You Go? Pedestrian Trajectory Prediction using Scene Features (https://arxiv.org/abs/2501.13848)
Comments:
          Accepted by 2024 International Conference on Intelligent Computing and its Emerging Applications

- **What's New**: 이 논문에서는 보행자 궤적 예측을 향상시키기 위해 보행자 상호작용과 환경 맥락을 통합한 새로운 궤적 예측 모델을 제안합니다. 기존 연구들이 보행자 간 상호작용 모델에 집중한 반면, 본 연구는 환경 요인과 장면 객체 배치의 중요성을 강조하였습니다. 공간적 및 시간적 상호작용을 포착하기 위해 sparse graph framework를 사용하고, scene features를 추출하기 위한 고급 이미지 강화 및 시맨틱 분할 기법을 사용합니다.

- **Technical Details**: 우리의 접근 방식은 Sparse Graph Convolutional Network (SGCN)을 통해 보행자 간의 상호작용을 포착하고, scene feature extraction 모듈과 cross-attention 메커니즘을 통해 예측 능력을 강화합니다. 이를 통해 보행자의 위치 정보와 주변 장면 정보를 기반으로 미래 궤적을 예측합니다. 이 과정에서는 각각의 보행자가 특정 시점에서의 공간 좌표로 특성화되어 과거 상태가 미래 위치 예측에 활용됩니다.

- **Performance Highlights**: 실험 결과, 제안한 방법이 기존의 최첨단 접근법들과 비교했을 때 ADE와 FDE 값이 각각 0.252m 및 0.372m로 현저히 우수함을 보여줍니다. 이는 보행자 궤적 예측에서 사회적 상호작용과 환경 맥락의 중요성을 강조합니다. 새로운 모델은 복잡한 도시 환경에서의 보행자 움직임 예측을 보다 효과적으로 수행할 수 있는 가능성을 제공합니다.



### Predicting Compact Phrasal Rewrites with Large Language Models for ASR Post Editing (https://arxiv.org/abs/2501.13831)
Comments:
          accepted by ICASSP 2025

- **What's New**: 이번 논문에서는 대규모 언어 모델(Large Language Models, LLMs)을 활용한 텍스트 리라이팅(task)에서 효율성을 높일 수 있는 새로운 편집 문구 표현(edit phrase representation)을 제안합니다. 이러한 표현은 기존의 span representation 대신, 구문 통계 기계 번역(phrase-based statistical machine translation)에서 영감을 받아 개발되었습니다. 연구자들은 이 방법이 ASR(Automatic Speech Recognition) 후 편집(task)에서 우수한 성과를 보일 것이라고 기대하고 있습니다.

- **Technical Details**: 제안된 문구 표현은 입력과 출력 사이의 겹치는 부분을 활용해 편집 작업의 수치적 표현을 압축할 수 있도록 돕습니다. 두 가지 새로운 표현 방식은 각 리라이트 패턴을 소스-타겟 구문 쌍으로 표현하거나, 좌우문맥 단어와 함께 타겟 문구만을 사용하는 방식입니다. 이러한 접근은 계산 비용을 줄이고, 의미적 일관성을 유지하며, 최종 출력의 품질을 확보하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 연구 결과, 제안된 문구 표현 방식이 기존의 span 모델과 비교했을 때, WER(Word Error Rate) 격차를 50-60% 줄이는 효율성을 보이라고 보고했습니다. 또한, 출력 길이 감소율 또한 기존 방식에 비해 10-20% 손실로 괜찮은 성능을 유지했습니다. 이로써 ASR 출력 수정에 대한 새로운 가능성을 열었으며, LLMs의 활용을 더욱 넓힐 수 있는 길을 제시했습니다.



### A space-decoupling framework for optimization on bounded-rank matrices with orthogonally invariant constraints (https://arxiv.org/abs/2501.13830)
Comments:
          48 pages, 12 figures, 6 tables

- **What's New**: 이 논문에서는 저랭크 최적화(low-rank optimization)에서 선형 제약조건을 추가하는 것의 중요성이 강조됩니다. 저자들은 각 제약의 접선 공간(tangent cone)이 어떻게 서로의 접선 공간과 교차되는지를 보여주며, 이를 통해 복잡한 저랭크 및 직교 불변 제약을 두 개의 공간으로 분리하는 새로운 프레임워크인 "공간 분리(space-decoupling)"를 제안합니다. 이 접근 방식은 기하학적 구조를 이해하고 Riemannian 알고리즘을 적용하는 데 유용합니다.

- **Technical Details**: 이 연구에서 다루는 최적화 문제는 차원 축소를 위한 저랭크 행렬을 포함합니다. 저자는 두 개의 상이한 공간을 활용하여 추가 제약을 직교적으로 불변하는 성질을 지닌 매트릭스에 적용하는 방법을 제안합니다. 여기서 h 매핑은 연속적이며, level set의 전체 rank를 유지하여 최적화 문제를 원할하게 설정합니다.

- **Performance Highlights**: 실제 데이터를 활용한 수치 실험들은 이 새로운 프레임워크의 우수성을 입증합니다. 저자들은 구형 데이터 적합(spherical data fitting), 그래프 유사도 측정(graph similarity measuring), 저랭크 SDP(low-rank SDP), 마르코프 과정의 모델 축소, 강화 학습, 그리고 딥 러닝에 대한 적용을 통해 다양한 사례를 제시합니다. 이 논문이 제공하는 방법론은 저랭크 최적화의 복잡성을 효과적으로 해결하는 데 기여할 것으로 기대됩니다.



### Hallucinations Can Improve Large Language Models in Drug Discovery (https://arxiv.org/abs/2501.13824)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)에서 발생할 수 있는 환각(hallucination)이 오히려 약물 발견(drug discovery) 과정에서 LLM의 성능을 향상시킬 수 있다는 가설을 제안합니다. 연구진은 LLM을 활용해 화합물의 SMILES 문자열을 자연어로 설명하고, 이 설명을 약물 발견을 위한 특정 작업의 프롬프트로 사용했습니다. 7개의 LLM과 5개의 분류 작업을 대상으로 한 평가 결과, 환각이 포함된 텍스트가 성능 향상에 기여함을 확인하였습니다.

- **Technical Details**: 연구에서는 LLM을 사용해 분자의 SMILES 문자열을 기반으로 텍스트 설명을 생성하고, 이를 모델의 입력으로 추가하여 특정 기능을 예측하는 작업을 수행했습니다. Llama-3.1-8B 모델이 ROC-AUC에서 18.35% 향상된 결과를 보였으며, GPT-4o가 생성한 환각은 다양한 모델에서 일관된 성능 개선을 이끌어냈습니다. 또한 모델 크기, 생성 온도, 언어와 같은 여러 요소가 환각과 모델 성능에 미치는 영향을 조사했습니다.

- **Performance Highlights**: 환각을 활용한 LLM의 성능 향상이라는 주제는 아직 탐구되지 않은 분야로, 이 연구는 첫 번째 체계적인 조사를 제시합니다. Llama-3.1-8B 모델은 SMILES 기준에 비해 18.35%의 성능 향상을 보였으며, 더 큰 모델일수록 환각을 통한 성능 개선 가능성이 더 높음을 확인했습니다. 이 연구는 약물 개발 분야에서 LLM의 활용에 대한 새로운 관점을 제시하며, 미래 연구에 기여할 수 있는 통찰력을 제공합니다.



### Learning to Help in Multi-Class Settings (https://arxiv.org/abs/2501.13810)
Comments:
          30 pages, 7 figures, conference, ICLR 2025

- **What's New**: 최근 연구된 Learning to Help (L2H) 모델은 고정된 로컬(클라이언트) 모델을 기반으로 서버 모델을 학습시키는 방식으로, 이는 고정된(전문가) 서버를 위한 클라이언트를 학습시키는 Learning to Defer (L2D) 프레임워크와 구별됩니다. 본 연구에서는 L2H 모델을 다중 클래스( multi-class) 분류 문제로 확장하여 서버 접근 비용이나 정책으로 인해 현실적 관심이 있는 다양한 시나리오에서의 적용 가능성을 보여줍니다. 제안된 방법은 자원 제약 환경에서도 다중 클래스 분류를 위한 효율적이고 실용적인 솔루션을 제공합니다.

- **Technical Details**: L2H 모델의 추정 목표는 Bayes 규칙에 부합하는 미분 가능하고 볼록한 에코스에 기초한 잔여 손실 함수(surrogate loss function)를 통해 성능을 평가하는 것입니다. 우리의 접근 방식은 PPR, IA, BRR 세 가지의 시나리오를 고려하여 다중 클래스 예측기를 훈련하는 알고리즘을 설계합니다. 이 과정에서, 후처리 방법(post-hoc method)을 사용하여 거부율(BRR)을 보장하고, 서버 모델과 리젝터를 통합하여 ML 시스템의 전체 성능을 증대시킵니다.

- **Performance Highlights**: 제안된 모델이 포함된 시스템의 실험 결과는 다양한 시나리오에서의 성능이 개선되었음을 보여줍니다. 리젝터는 불확실하거나 도전적인 샘플을 효과적으로 식별하고, 서버 모델 사용의 정확성과 효율성을 균형 있게 조절하는 역할을 합니다. 이 연구는 자원 제약 환경에서 클라이언트와 서버 간의 효과적인 협업 발생 가능성을 제시하며, 이러한 하이브리드 ML 시스템의 구축에 대한 새로운 통찰을 제공합니다.



### Parameter-Efficient Fine-Tuning for Foundation Models (https://arxiv.org/abs/2501.13787)
Comments:
          25 pages, 6 figures, 7 tables

- **What's New**: 본 설문조사는 Foundation Models (FMs)와 관련된 Parameter-Efficient Fine-Tuning (PEFT) 기술에 대한 포괄적인 리뷰를 제공합니다. PEFT는 비용 효율적인 미세 조정 기술로, 알고리즘의 매개변수(parameter) 수를 최소화하며, 최적의 다운스트림(task)의 성과를 목표로 합니다. 다양한 FMs에 적용된 PEFT 기술을 탐구하는 이 연구는 이러한 통합의 잠재력을 명확히 하고, 향후 연구 및 개발 방향을 제시합니다.

- **Technical Details**: Foundation Models (FMs)은 대규모 데이터셋에서 미리 훈련되어 언어 이해, 코드 생성 및 이미지 처리와 같은 다양한 작업을 지원하는 모델입니다. PEFT는 이러한 모델을 더욱 효과적으로 미세 조정하는 기술로, LoRA( Low-Rank Adaptation)와 같은 여러 방법론을 통해 매개변수와 계산량을 획기적으로 줄입니다. 또한, 각 FM 유형에 따라 다양한 적응 전략이 요구되며, PEFT는 이러한 다양성을 쉽게 처리할 수 있는 방식으로 발전하고 있습니다.

- **Performance Highlights**: 설문 결과는 PEFT의 성과가 각 FM에서 99.97% 이상의 매개변수 절약과 함께 수행된 성능 향상을 보여준다는 것을 나타냅니다. 특히, LLMs와 Vision Foundation Models (VFMs)이 현재 연구에서 주도적인 위치를 차지하며, 멀티모달 모델(Multi-Modal Models)도 앞으로 연구의 주요 관심사로 떠오르고 있습니다. 또한, 저자들은 PEFT 기술의 통합을 통해 이러한 모델들이 앞으로 더 많은 다운스트림 작업에서도 효과적으로 활용될 수 있는 가능성을 제시합니다.



### Defending against Adversarial Malware Attacks on ML-based Android Malware Detection Systems (https://arxiv.org/abs/2501.13782)
- **What's New**: 본 논문에서는 Android 악성 코드 탐지 시스템의 적대적 공격에 대응하기 위한 새로운 방안을 제안합니다. 기존 시스템은 특성 공간 공격(feature space attacks)에는 방어가 가능하지만, 실제 악성 코드에서 발생하는 문제 공간 공격(problem space attacks)에는 취약합니다. 이 문제를 해결하기 위해, ADD라는 실용적인 방어 프레임워크를 개발하여 ML 기반의 Android 악성 코드 탐지 시스템의 내성을 높이고자 합니다.

- **Technical Details**: ADD는 ML 기반 Android 악성 코드 탐지 시스템에 플러그인 형태로 설계되어 있어, 원래의 탐지 능력에 부정적인 영향을 미치지 않고도 악성 코드 샘플을 재조사합니다. 적대적 Android 악성 코드 공격 방법과의 상호 작용을 통해, 변화 가능한 특성과 변화 불가능한 특성을 정량화하여 격리합니다. 이러한 과정에서 Monte Carlo 샘플링 방법을 활용한 공간 정량화 알고리즘을 적용하고, 인코더 모델을 통해 다양한 차원에서 호환성을 평가합니다.

- **Performance Highlights**: ADD는 135,859개의 정상 애플리케이션과 15,778개의 악성 애플리케이션을 포함한 방대한 데이터셋을 기반으로 평가되었습니다. 이 시스템은 기존 방법들보다 모두 탁월한 성능을 보여주며, 95% 이상의 공격 성공률 감소를 달성했습니다. 또한, 최근의 악성 코드 샘플을 기준으로도 70%에서 80%의 탐지율을 유지하며 실제 악성 코드 탐지 솔루션의 내성을 향상시킬 수 있음을 입증했습니다.



### Not Every AI Problem is a Data Problem: We Should Be Intentional About Data Scaling (https://arxiv.org/abs/2501.13779)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLM)의 데이터 스케일링에 대한 접근 방식을 재조명하고 있습니다. 기존의 데이터 스케일링 관점에서 벗어나, 특정 작업이 데이터 스케일링을 통해 더 큰 혜택을 받을 수 있는지에 대한 의도를 가지고 데이터를 수집하는 것이 중요하다고 주장합니다. 연구자들은 데이터의 '형태(shape)'가 데이터 스케일링에서 우선시해야 할 작업들을 결정하고, 비효율적인 스케일링 작업을 위한 새로운 컴퓨팅 패러다임을 구상하는 데 중요한 역할을 한다고 강조합니다.

- **Technical Details**: 대규모 언어 모델의 발전은 '스케일링 법칙(scaling laws)'에 뿌리를 두고 있으며, 이 법칙은 모델과 데이터셋의 크기를 일정하게 조정하여 최적의 성능을 이끌어내는 과정을 설명합니다. 그러나 실제로는 고품질 데이터의 수가 제한적이기 때문에, 대량의 저질 데이터로 모델이 손상될 위험이 커지며, 이러한 문제는 특히 파라미터 수가 많은 대형 모델에서 두드러집니다. 이는 인간 생성 데이터의 양적 확보와 그 품질 보장이 필수적임을 의미합니다.

- **Performance Highlights**: 데이터 스케일링이 유망한 분야에 대한 사례로는 기계 번역(Machine Translation), 로보틱스, 신약 발견 등이 있습니다. 기계 번역의 경우, 규칙적인 언어 구조와 고품질 데이터가 모델 훈련에 기여하여 상당한 향상을 이루었습니다. 반면, 복잡한 논리적 추론이 필요한 작업에서는 여전히 모델이 인간보다 부족한 성능을 보이며, 이 문제는 모델 구조와 학습 알고리즘에 뿌리를 두고 있는 것으로 평가되었습니다.



### Tune In, Act Up: Exploring the Impact of Audio Modality-Specific Edits on Large Audio Language Models in Jailbreak (https://arxiv.org/abs/2501.13772)
- **What's New**: 이 논문에서는 오디오 편집이 대규모 오디오-언어 모델(LALMs)의 추론에 미치는 영향을 조사합니다. 저자들은 음성, 오디오 및 텍스트를 처리할 수 있는 다중 모달 대규모 언어 모델(MLLMs)의 발전을 배경으로, 각 모달리티의 입력 편집이 LALMs의 출력에 상당한 영향을 미친다는 기존 연구의 틈을 다룹니다. Audio Editing Toolbox(AET)와 Edited Audio Datasets(EADs)를 소개하여 연구자들이 오디오 콘텐츠와 다양한 오디오 특정 편집을 평가할 수 있도록 지원합니다.

- **Technical Details**: 저자들은 오디오 신호를 시간 영역 파형으로 나타내며, 여러 특성을 수정하기 위한 변환을 적용합니다. 이러한 신호는 Short-Time Fourier Transform(STFT)를 사용하여 시간-주파수 특성을 분석하고, Inverse Short-Time Fourier Transform(iSTFT)를 통해 주파수 영역의 변환을 시간 영역으로 재구성합니다. 마지막으로, 연구에 사용된 원본 오디오 데이터셋은 AdvBench에서 수집된 520개의 유해 텍스트 질문을 Google Text-to-Speech를 통해 오디오로 변환하여 생성되었습니다.

- **Performance Highlights**: 연구진은 BLSP, SpeechGPT, Qwen2-Audio, SALMONN 등 여러 최신 LALMs에 대해 다양한 오디오 편집 방법으로 성능 평가를 수행했습니다. 실험 결과는 오디오 모달리티 편집이 LALMs의 최종 추론 결과에 미치는 영향의 기초 자료로 활용될 수 있습니다. 이러한 연구는 향후 오디오 모달리티 간의 상호작용과 보안 문제에 대한 깊이 있는 탐색의 기초를 마련합니다.



### UGMathBench: A Diverse and Dynamic Benchmark for Undergraduate-Level Mathematical Reasoning with Large Language Models (https://arxiv.org/abs/2501.13766)
Comments:
          Accepted to ICLR 2025

- **What's New**: UGMathBench는 대학 수준의 수학 문제를 평가하기 위해 특별히 설계된 역동적이고 다양한 벤치마크로, 총 5,062개의 문제를 포함하며, 이는 16개의 주제와 111개의 세부 주제로 나뉘어져 있습니다. 각 문제는 세 가지 버전으로 무작위화되어 있어 모델의 진정한 추론 능력을 평가할 수 있습니다. 또한, 해당 연구는 효과적 정확도(Effective Accuracy, EAcc)와 추론 갭(Reasoning Gap, Δ)이라는 두 가지 주요 메트릭을 도입하여 평가의 정밀성을 높였습니다.

- **Technical Details**: UGMathBench는 5,062개의 문제로 구성되어 있으며, 이 문제들은 여덟 가지 기본 답변 유형과 두 가지 복합 답변 유형으로 분류됩니다. 각 문제의 변형은 LLM의 진정한 추론 능력을 평가하는 데 기여하며, 이상적으로 EAcc가 높고 Δ가 0인 '대형 추론 모델'의 개발 필요성이 강조됩니다. 본 연구의 평가에서 OpenAI-o1-mini가 56.3%의 최고 EAcc를 기록했으며, 모든 LLM은 높은 추론 갭을 드러냈습니다.

- **Performance Highlights**: UGMathBench의 평가는 LLM의 현재 성능 한계를 드러내며, OpenAI-o1-mini와 같은 가장 발달한 모델조차 56.3%의 EAcc를 기록하는 데 그쳤습니다. LLM의 Robustness Efficiency(Δ와 EAcc 비율)는 20.78%에서 196.6%까지 다양해 현재 모델들의 일관성 부족이 드러났습니다. 각 주제에 따른 평균 EAcc는 차이를 보이며, 특히 Arithmetic(62.8%)와 Absctract Algebra, Differential Equations, Financial Mathematics(10% 이하) 간의 큰 격차가 나타났습니다.



### Integrating Causality with Neurochaos Learning: Proposed Approach and Research Agenda (https://arxiv.org/abs/2501.13763)
Comments:
          9 pages

- **What's New**: 이 논문은 심층 학습(deep learning)의 한계를 극복하기 위한 두 가지 대안적 접근법인 인과 학습(causal learning)과 신경 혼돈 학습(Neurochaos Learning) 소개에 중점을 둡니다. 인과 학습은 데이터셋 내 아이템 간 인과 관계를 고려하여 잘못된 상관관계를 줄이는 데 기여할 것으로 기대됩니다. 반면, 신경 혼돈 학습은 생물학적 신경망에서의 비선형적인 혼돈 발사를 기반으로 하여, 적은 수의 샘플로도 효과적인 분류 성능을 입증하고 있습니다.

- **Technical Details**:  심층 학습 모델은 여러 처리(layer)로 구성되어 데이터의 다층적 표현을 학습합니다. 하지만 이러한 모델은 통계적 상관관계에 의존하여 학습하며, 많은 경우 스푸리어스(spurious)한 상관관계를 초래합니다. 이 논문에서는 인과 모델과 그래프 신경망을 통합하여 더 나은 성능을 발휘하도록 하는 방법을 제안하며, 특히 링크된 데이터(linked data) 도메인에서 효과적인 결과를 기대하고 있습니다.

- **Performance Highlights**: 인과 학습과 신경 혼돈 학습을 통합함으로써 데이터의 구조적 모델을 활용해 예측 모델링(predictive modeling)과 강화 학습(reinforcement learning)에 대한 효과를 증대시킬 수 있습니다. 실험 결과, 이 두 접근법의 통합이 기존의 단순 심층 학습 모델보다 우수한 성과를 보여주었습니다. 이러한 개선된 성능은 IoT, 의료 및 생명 과학, 제조업 등 다양한 분야에서 활용될 가능성을 보여주고 있습니다.



### 2-Tier SimCSE: Elevating BERT for Robust Sentence Embeddings (https://arxiv.org/abs/2501.13758)
- **What's New**: 이번 연구는 자연어 처리(NLP)에서 의미 있는 문장 임베딩 생성의 중요성을 강조하고, SimCSE라는 새로운 접근법을 통해 minBERT 모델을 감정 분석, 의미적 텍스트 유사성(semantics textual similarity, STS), 그리고 패러프레이즈 감지 과제를 위해 미세 조정하는 것을 목표로 합니다. 이 과정에서 세 가지 서로 다른 드롭아웃 기법을 실험하여 오버핏(overfitting) 문제를 해결하고, 비지도 및 지도 SimCSE를 결합한 2-Tier SimCSE 미세 조정 모델을 제안했습니다. 연구 결과는 2-Tier 모델이 STS 과제에서 높은 성능을 기록했음을 보여줍니다.

- **Technical Details**: 연구에서 사용된 기본 모델은 다중 작업 및 단일 작업 버전의 minBERT로, 12개의 트랜스포머(transformer) 레이어를 통해 문장 토큰화와 임베딩 결합, 다중 헤드(self-attention mechanism)를 구현합니다. 단일 작업 모델은 각 다운스트림 과제를 개별적으로 학습하고 미세 조정하여, 작업별로 초점을 맞춘 진행을 보장합니다. 또한 3가지 드롭아웃 기법(표준 드롭아웃, 커리큘럼 드롭아웃, 적응형 드롭아웃)을 활용하여 모델의 일반화 성능을 개선하려고 했습니다.

- **Performance Highlights**: 모델은 Unsupervised SimCSE와 Supervised SimCSE를 활용하여 STS 데이터셋에서 각기 다른 점수로 뛰어난 성능을 보였습니다. Unsupervised SimCSE는 0.716의 피어슨 상관 계수(Pearson Correlation score)를 달성했고, Supervised SimCSE는 0.806의 성능을 기록했습니다. 그러나 패러프레이즈 및 SST 과제에서의 성능 향상은 제한적이었으며, STS에서의 지식을 전이하는 데 한계가 있음을 시사합니다.



### Solving the long-tailed distribution problem by exploiting the synergies and balance of different techniques (https://arxiv.org/abs/2501.13756)
Comments:
          13

- **What's New**: 본 연구에서는 긴 꼬리 인식(long-tail recognition)을 위한 세 가지 기법인 Supervised Contrastive Learning (SCL), Rare-Class Sample Generator (RSG), 그리고 Label-Distribution-Aware Margin Loss (LDAM)의 상호작용을 탐구합니다. 기존 연구들은 데이터 분포를 변화시키거나 모델의 결정 경계를 조정하여 긴 꼬리 인식 성능을 향상시키려 했으나, 다양한 방법 간의 협력과 보정에 대한 연구는 부족했습니다. SCL은 내부 클래스 클러스터를 증가시키고 명확한 클래스 간 분리를 도모하지만, 주류 클래스에 편향되는 경향이 있습니다. 이를 보완하기 위해 RSG와 LDAM을 결합하여 긴 꼬리 클래스의 성능을 더욱 향상시킵니다.

- **Technical Details**: 컴퓨터 비전에서 긴 꼬리 분포가 일반적임에도 불구하고, 균형 잡힌 데이터셋에서 학습한 CNN 모델은 불균형한 데이터셋에서 성능이 저하됩니다. SCL은 Class-averaging 및 Class-complement 전략을 활용하여 내부 및 꼬리 클래스 모두에 대한 공정한 학습을 보장하는 Balanced Contrastive Learning (BCL)으로 확장됩니다. 본 연구는 각 기법의 강점을 활용하여 상호 보완적인 관계를 형성하고, SCL과 RSG가 긴 꼬리 클래스의 클러스터 특징을 향상시키는 방안을 제시합니다.

- **Performance Highlights**: 실험을 통해 제안된 방법은 긴 꼬리 클래스의 정확도를 향상시키면서도 주류 클래스의 성능을 유지하여 모든 클래스에서 균형 잡힌 개선을 달성합니다. SCL과 RSG의 통합은 클래스 간 뚜렷한 분리를 가능하게 하여 더 나은 성능을 제공합니다. 또한, LDAM은 긴 꼬리 클래스에 대해 더 큰 마진을 부여하여, 각 기법의 강점들이 서로를 강화하는 긍정적인 결과를 이끌어냅니다.



### EICopilot: Search and Explore Enterprise Information over Large-scale Knowledge Graphs with LLM-driven Agents (https://arxiv.org/abs/2501.13746)
- **What's New**: EICopilot은 자연어 질의를 이해하고, 대규모 지식 그래프에서 정보를 탐색하며, 복잡한 쿼리를 수행하고 요약하는 지능형 시스템입니다. 이 시스템은 Baidu Enterprise Search의 챗봇으로 배포되어 사용자의 검색 경험을 혁신합니다. EICopilot은 데이터 전처리 파이프라인과 독창적인 쿼리 마스킹 전략을 통해 정보 검색의 정확성과 효율성을 향상시키고 있습니다.

- **Technical Details**: EICopilot의 구조는 온라인 및 오프라인 두 가지 단계로 나뉘어 있으며, 고품질의 데이터 준비 및 검색 쿼리 생성 과정을 중요시합니다. 오프라인 단계에서는 스키마 정보 해석, 시드 데이터 구축, 데이터 증강 및 유사 질문 선택을 통해 데이터 세트를 정리합니다. 온라인 단계에서는 LLM의 능력을 활용하여 사용자의 비표준 쿼리를 해석하고, CoT(Chain-of-Thought) 및 ICL(In-context learning) 기법을 적용해 효율적인 정보 검색을 지원합니다.

- **Performance Highlights**: EICopilot은 데이터 검색 속도와 정확성에서 기존 기법들을 능가하는 성능을 보입니다. 특히, 전반적인 구문 오류율을 10.00%로 줄이며, 실행 정확도는 82.14%에 달합니다. 이러한 결과는 EICopilot의 구성 요소들이 쿼리 및 요약 프로세스를 향상시키는데 중요한 역할을 한다는 것을 보여줍니다.



### Pseudocode-Injection Magic: Enabling LLMs to Tackle Graph Computational Tasks (https://arxiv.org/abs/2501.13731)
Comments:
          24 pages

- **What's New**: 본 논문은 그래프 계산 작업에 대해 새로운 접근 방식을 제안합니다. 기존 대형 언어 모델(LLMs)이 직면한 문제를 해결하기 위해 새로운 프레임워크인 PIE(Pseudocode-Injection-Enhanced LLM Reasoning for Graph Computational Tasks)를 도입합니다. 이 프레임워크는 문제 이해, 프롬프트 설계 및 코드 생성의 세 가지 핵심 단계로 구성됩니다.

- **Technical Details**: 프레임워크 PIE는 LLMs가 문제를 이해하고 관련 정보를 추출하여 코드를 생성하는 역할을 맡도록 하며, 그래프 구조를 분석하고 코드를 실행하는 책임은 인터프리터에 위임됩니다. 실험을 통해 PIE가 기존 방법들보다 정확도와 계산 효율성에서 뛰어난 성능을 보임을 입증하였습니다. 특히, Pseudocode injection 기법을 활용하여 LLMs가 효율적인 코드를 생성할 수 있도록 지원합니다.

- **Performance Highlights**: 아홉 가지 그래프 추론 작업에 대한 실험 결과, PIE는 높은 정확도로 기존 방법들보다 월등한 성능을 제공하는 동시에 계산 비용을 현저히 절감했습니다. 또한, LLMs에 의한 코드 생성을 통해 다양한 그래프 계산 작업에서의 효율성 또한 크게 향상되었습니다. 이로 인해, PIE는 저지연 및 고도 확장성이 요구되는 응용 프로그램에서도 실용성이 높은 해결책으로 자리 잡을 수 있습니다.



### Scalable Safe Multi-Agent Reinforcement Learning for Multi-Agent System (https://arxiv.org/abs/2501.13727)
- **What's New**: SS-MARL (Scalable Safe Multi-Agent Reinforcement Learning)라는 새로운 프레임워크를 제안하여 MARL (Multi-Agent Reinforcement Learning) 알고리즘의 안전성과 확장성을 향상시킵니다. 이 프레임워크는 그래프 신경망 (Graph Neural Networks, GNNs)을 활용하여 에이전트 간의 암시적인 커뮤니케이션을 개선하고, 국소 관측을 필요로 하는 할당된 정책 최적화 방식을 도입합니다. 또한 SS-MARL은 작은 규모의 작업에서 훈련된 모델을 큰 규모의 작업으로 제로샷 전이 (zero-shot transfer)할 수 있는 능력도 갖추고 있습니다.

- **Technical Details**: SS-MARL의 핵심 특징에는 (1) GNN을 사용하여 에이전트 간에 암시적인 커뮤니케이션을 달성하고 훈련 단계에서 샘플링 효율성을 향상시키는 것, (2) 여러 제약 조건을 처리할 수 있는 제약된 공동 정책 최적화(constrained joint policy optimization)를 통해 훈련 및 테스트 단계의 안전성을 확보하는 것, (3) 작은 규모의 작업에서 훈련된 모델을 큰 규모의 작업으로 전이하는 동안 높은 안전성을 유지하는 것입니다.

- **Performance Highlights**: 시뮬레이션 실험 결과 SS-MARL은 최적성과 안전성 간의 균형을 더 잘 이루었으며, 많은 수의 에이전트가 포함된 시나리오에서 기존의 최신 방법보다 상당히 우수한 확장성을 보여주었습니다. 하드웨어 구현을 통해 메카넘 휠 차량을 이용한 시험에서도 본 방법의 실행 가능성이 검증되었습니다.



### You Only Crash Once v2: Perceptually Consistent Strong Features for One-Stage Domain Adaptive Detection of Space Terrain (https://arxiv.org/abs/2501.13725)
- **What's New**: 이번 연구는 우주에서의 지형 인식을 위한 YOCOv2 모델을 제안하며, 이는 Visual Similarity-based Alignment (VSA) 기법을 활용하여 기존 YOCOv1보다 31% 이상의 성능 향상을 이루었습니다. YOCOv2는 시뮬레이션과 실제 데이터를 포함한 다양한 환경에서 실시간으로 지형 탐지를 수행할 수 있는 능력을 갖추고 있습니다. 이를 통해 NASA의 임무 데이터를 활용한 실제 비행 하드웨어 성능 평가와 질적 분석을 통해 YOCOv2의 실용성을 증명했습니다.

- **Technical Details**: YOCOv2는 텍스처가 없는 지역 및 다양한 조명 조건에서도 효과적인 지형 탐지를 가능하게 하기 위해 최근의 이미지 생성 기술을 통합했습니다. 우리는 PMDA(Prior Model Domain Adaptation) 기술을 통해, 인스턴스 및 intra-feature 기반 클러스터링 기법을 활용하여 적대적 및 대조학습 프레임워크 내에서 VSA 기법을 강화했습니다. 여섯 개의 데이터셋을 구성하여 Mars, Moon, Asteroid 지형의 UDA(자율 도메인 적응) 평가를 수행했습니다.

- **Performance Highlights**: YOCOv2는 YOCOv1 및 지구에서의 최첨단 방법들과 비교하여 31% 이상의 성능 향상을 달성했습니다. NASA 우주선 하드웨어 기반에서의 벤치마킹을 통해 실제 상황에서의 적용 가능성을 검토했습니다. 깊이 있는 정량적 및 질적 평가로 다양한 환경에서의 VSA 기술의 효과를 입증하여, 향후 우주 비행 미션에서의 활용 가능성을 높였습니다.



### Musical ethnocentrism in Large Language Models (https://arxiv.org/abs/2501.13720)
- **What's New**: 이 논문은 대형 언어 모델(LLM)의 음악 관련 편향을 분석하기 위한 첫 걸음을 제시합니다. 특히 ChatGPT와 Mixtral 모델을 대상으로 하여, 두 가지 실험을 통해 이들 모델이 서구 음악 문화에 대한 강한 선호를 나타내는 것을 발견했습니다. 연구의 주요 초점은 지리 문화적 편향인 'geocultural biases'를 탐구하는 것입니다.

- **Technical Details**: 연구는 두 가지 실험을 통해 수행되었습니다. 첫 번째 실험에서는 다양한 카테고리의 'Top 100' 음악 기여자 리스트를 요청하고, 그들의 원산지를 분석했습니다. 두 번째 실험에서는 여러 나라의 음악 문화 특성을 수치적으로 평가하도록 LLM에 요청하였습니다. 이를 통해 문화 및 지역의 불균형한 대표성과 관련된 편향을 정량적으로 평가하고자 했습니다.

- **Performance Highlights**: 실험 결과, 미국을 포함한 서양 국가에 대한 집중이 두드러졌으며 아시아와 아프리카는 크게 저평가되는 경향이 있었습니다. 특정 카테고리에서는 약간의 다양성이 발견되었지만, 전반적으로 LLM은 서구 음악 문화에 대한 선호를 강하게 드러냈습니다. 이 연구는 LLM의 문화적 감수성 개발을 위한 기초 연구로 평가됩니다.



### Skin Disease Detection and Classification of Actinic Keratosis and Psoriasis Utilizing Deep Transfer Learning (https://arxiv.org/abs/2501.13713)
- **What's New**: 이번 연구에서는 피부 질환 진단을 위한 새로운 방법을 제안합니다. 딥러닝 기법을 사용하여 피부 질환을 진단하며, 수정된 VGG16 컨볼루션 신경망 모델을 활용합니다. 기존의 진단 방법들이 비싸고 접근성이 낮은 점을 고려하여, 보다 효율적이고 저렴한 대안을 찾고자 하였습니다.

- **Technical Details**: 제안된 모델은 여러 개의 컨볼루션 레이어로 구성되어 있으며, ImageNet 가중치를 사용하여 수정된 최상위 레이어를 포함합니다. 최상위 레이어는 완전 연결 층(fully connected layers)과 최종 소프트맥스 활성화 층(softmax activation layer)을 통해 피부 질환을 분류합니다. 데이터셋은 'Skin Disease Dataset'으로 공개되어 있으며, 회전(rotation), 이동(shifting), 줌(zoom) 등의 전처리 기술을 적용하여 데이터를 증대(augmentation)했습니다.

- **Performance Highlights**: 제안된 방법론은 수정된 VGG16 모델을 사용하여 90.67%의 정확도를 달성하였으며, 이는 피부 질환 분류에서의 신뢰성을 입증합니다. 이러한 유망한 결과는 실제 응용 프로그램에서 이 접근법의 가능성을 강조합니다.



### YOLO11-JDE: Fast and Accurate Multi-Object Tracking with Self-Supervised Re-ID (https://arxiv.org/abs/2501.13710)
Comments:
          This paper has been accepted to the 5th Workshop on Real-World Surveillance: Applications and Challenges (WACV 2025)

- **What's New**: YOLO11-JDE는 빠르고 정확한 다중 객체 추적(Multi-Object Tracking, MOT) 솔루션으로, 실시간 객체 검출(realtime object detection)과 자체 지도 재식별(self-supervised Re-Identification, Re-ID)을 결합합니다. YOLO11의 전용 Re-ID 브랜치를 포함하여, 이 모델은 Joint Detection and Embedding(JDE)을 수행하며 각 검출에 대한 외관 피처를 생성합니다. 이 모델은 탐지와 동시에 자체 지도 방식으로 훈련을 진행하여 비용이 많이 드는 식별 레이블 데이터셋의 필요성을 없앴습니다.

- **Technical Details**: YOLO11-JDE는 triplet loss와 하드 양성, 반 하드 음성 마이닝 전략을 사용하여 구별되는 임베딩(discriminative embedding)을 학습합니다. 데이터 연관(data association)은 모션, 외관, 위치 단서를 성공적으로 통합하는 맞춤형 추적 구현(custom tracking implementation)으로 향상됩니다. 이 모델은 MOT17 및 MOT20 벤치마크에서 경쟁력 있는 결과를 달성하며, 기존 JDE 방법들보다 FPS에서 우수한 성능을 보이고, 최대 10배 적은 파라미터를 사용합니다.

- **Performance Highlights**: YOLO11-JDE는 MOT 챌린지 벤치마크에서 높은 정확도의 추적 성능을 보여줍니다. 또한, 기존 JDE 방법들과 비교할 때 훨씬 더 적은 파라미터 수로 인해 실시간 MOT 응용 프로그램에 적합한 속도를 자랑합니다. 이 모델은 프레임 속도(FPS) 증가와 함께 고도화된 능력을 바탕으로, 실제 응용 프로그램에 매우 매력적인 솔루션으로 자리잡고 있습니다.



### EventVL: Understand Event Streams via Multimodal Large Language Mod (https://arxiv.org/abs/2501.13707)
- **What's New**: EventVL은 첫 번째 이벤트 기반 MLLM 프레임워크로, 140만 개의 고품질 데이터 쌍을 사용하여 세밀한 의미 이해를 목표로 한다. 기존의 CLIP 기반 방법들의 한계를 극복하고 이벤트 스트림의 충분한 의미와 맥락을 파악할 수 있도록 설계되었다. 이는 다양한 장면에서 효과적인 학습을 가능하게 하며, 이벤트 기반 비전 커뮤니티의 발전에 기여할 것으로 기대된다.

- **Technical Details**: EventVL은 Event Spatiotemporal Representation을 통해 이벤트 데이터의 시공간적 상관관계를 탐구하고, Dynamic Semantic Alignment 모듈을 통해 세밀한 이미지와 이벤트 간의 정렬을 개선한다. 이 프레임워크는 이미지 및 이벤트 인코더, 그리고 대형 언어 모델(LLM)을 통합하여 이벤트 기반 생성 작업에서 강력한 성능을 발휘한다. 전체적인 실험 결과, EventVL은 이벤트 캡셔닝 및 장면 설명 생성 작업에서 기존 SOTA들을 초월하는 성과를 보여준다.

- **Performance Highlights**: EventVL은 약 23억 개의 매개변수로 구성되어 있으나, 타 MLLM들에 비해 낮은 비용으로 배포할 수 있도록 설계되었다. 이벤트 스트림에 대한 정확한 설명 생성을 가능하게 하며, 다중 전환 대화(interactive dialogue)를 지원하는 기능을 통해 심화된 의미 이해를 지원한다. 이는 다양한 분야에서 이벤트 기반 데이터의 잠재력을 완전히 발휘할 수 있도록 돕는다.



### Training-Free Consistency Pipeline for Fashion Repos (https://arxiv.org/abs/2501.13692)
- **What's New**: 패션 이미지 편집의 최신 혁신을 소개하는 논문에서는, 비간섭 포즈 편집을 위한 FashionRepose라는 훈련 필요 없는 파이프라인을 제안합니다. 이 방법은 의류의 포즈 조정이 가능한 동시에 아이덴티티와 브랜드 속성을 유지하여, 패션 산업의 요구를 충족할 수 있도록 설계되었습니다. FashionRepose는 제로샷 접근 방식을 사용하여 실시간에 가까운 속도로 편집을 수행할 수 있는 장점이 있습니다.

- **Technical Details**: FashionRepose는 롱슬리브 의류의 포즈를 표준화하기 위해 설계된 다단계 파이프라인입니다. 이 접근 방법은 훈련 데이터 없이도 롱슬리브 의류의 포즈 노멀라이제이션(task of garment repose)을 가능하게 합니다. 시스템의 주요 구성 요소는 소스 포즈에서 표준화된 목표 포즈로 포즈를 변환하면서 소스 이미지의 색상, 질감 및 브랜드 속성을 보존하는 것입니다.

- **Performance Highlights**: FashionRepose는 훈련 필요 없이 실시간 결과(60초 이하)를 보장하며, 자동화된 로고 보존 작업을 통해 편집 중 브랜드 아이덴티티를 유지합니다. 이 접근 방식은 e-커머스, 패션 마케팅 및 디자인 프로토타이핑과 같은 분야에서 신뢰할 수 있는 편집 솔루션을 제공하여, 패션 이미지 편집의 산업적 요구를 충족시키는 역량을 갖추고 있습니다.



### Question Answering on Patient Medical Records with Private Fine-Tuned LLMs (https://arxiv.org/abs/2501.13687)
- **What's New**: 본 연구에서는 전자 건강 기록(EHR)의 정보 검색을 위해 새롭게 제안된 의미론적 질의 응답(semantic QA) 방식을 소개합니다. 이 방법은 사용자 질의에 가장 유의미한 FHIR 리소스를 식별한 후, 해당 리소스를 기반으로 응답을 생성하는 두 단계의 과정으로 이루어집니다. 또한, 사적인 환경에서 운영되는 LLM(대형 언어 모델)을 활용하여 환자의 개인 정보를 보호하는 동시에 효과적인 건강 데이터 접근을 가능하게 합니다.

- **Technical Details**: 연구의 두 가지 주요 작업(Task 1 & Task 2)은 각각 FHIR 리소스 검색과 질의에 대한 응답 생성을 포함합니다. 각 작업은 LLM을 활용하여 모델을 미세 조정(fine-tuning)하는 방식을 채택하고, 특히 Llama-3.1-8B 및 Mistral-NeMo 모델을 사용하여 정확성과 효율성을 극대화합니다. 데이터 수집과 정제, 그리고 여러 모델의 성능을 비교하는 단계를 통해 최적의 모델 구성을 평가합니다.

- **Performance Highlights**: 실험 결과, 미세 조정된 LLM이 기본적인 GPT-4 모델보다 Task 1의 F1 점수에서 0.55% 더 뛰어나고, Task 2의 Meteor Task에서는 42% 높은 성능을 보였습니다. 이러한 결과는 환자 중심의 의미론적 질의 응답을 수행할 때, LLM의 개인 정보 보호와 데이터 효율성을 동시에 확보할 수 있음을 보여줍니다.



### Unlearning Clients, Features and Samples in Vertical Federated Learning (https://arxiv.org/abs/2501.13683)
Comments:
          Paper accepted for publication in PETS 2025, Issue II

- **What's New**: 이 논문에서는 Federated Learning(FL)에서의 unlearning 문제를 다루고 있으며, 특히 Vertical FL(VFL)에서의 새로운 방법론을 제시합니다. VFL에서는 클라이언트들이 레이블에 접근하지 않고 샘플 공간을 나누어 사용하기 때문에 기존의 Horizontal FL(HFL)에서 제안된 접근 방식이 적용되지 않습니다. 본 연구는 VFL 내에서 클라이언트, 특징, 샘플을 unlearn하는 세 가지 접근법을 탐구하고, 각 방법론의 효과를 Membership Inference Attack(MIA) 를 활용해 입증하고자 합니다.

- **Technical Details**: VFU-KD와 VFU-GA라는 두 가지 새로운 알고리즘을 소개합니다. VFU-KD는 Knowledge Distillation(KD)에 기반하여 클라이언트와 특징을 unlearn하는 방식이며, VFU-GA는 gradient ascent에 기반하여 샘플을 unlearn합니다. 이 방법들 점검을 위해 총 6개의 표 형태 데이터셋과 2개의 이미지 데이터셋에서 실험을 진행하였으며, 실험 결과 이 두 방법이 retraining 또는 기존의 R2S 방법에 비해 우수한 성능을 보임을 확인했습니다.

- **Performance Highlights**: VFU-KD와 VFU-GA는 많은 경우에 retraining으로 얻어진 성과나 R2S 벤치마크에 비해 비슷하거나 더 나은 성과를 달성하였으며, 개선 폭은 0-2%에 이릅니다. 이 방법들은 기존의 방법들과 달리 active party와 passive party 간의 추가적인 커뮤니케이션 없이 unlearning이 가능하다는 장점을 가지고 있습니다. 다만, active party는 이전에 통신된 embedding을 저장해야 하는 의무가 있습니다.



### Certified Robustness Under Bounded Levenshtein Distanc (https://arxiv.org/abs/2501.13676)
Comments:
          Accepted in ICLR 2025

- **What's New**: 이 논문에서는 텍스트 분류기의 강건성을 증명하기 위한 새로운 방법을 제안합니다. Lipschitz constant를 Levenshtein distance에 대한 컨볼루션 분류기에 대해 계산할 수 있는 첫 번째 방법으로, 이는 텍스트 도메인에서 효율적인 검증을 가능하게 합니다. Proposed method, LipsLev는 단일 전방 전달 통과에서 분류기의 공인 반지름을 계산할 수 있도록 합니다.

- **Technical Details**: 기존의 검증 방법들은 대개 캐릭터/단어 대체 또는 불용어 제거와 같은 사양만 처리할 수 있었으나, 본 연구에서는 평균 전송 거리(ERP distance)에 대한 Lipschitz constant를 계산함으로써 1-Lipschitz 분류기를 훈련할 수 있는 방법을 제시합니다. 이 방법은 기존의 접근 방식이 Levenshtein distance 제한에 대한 검증을 지원하지 못했던 문제를 해결합니다.

- **Performance Highlights**: LipsLev는 AG-News 데이터 세트에서 거리 1과 2에서 각각 $38.80$% 및 $13.93$%의 공인 정확도를 기록하며, 이는 기존 접근 방식보다도 4배 이상 빠른 속도를 자랑합니다. 또한 거리 2 이상에서 검증할 수 있는 유일한 방법으로, 이는 텍스트 도메인에서의 강건성 검증의 새로운 이정표가 될 것으로 기대됩니다.



### How to Complete Domain Tuning while Keeping General Ability in LLM: Adaptive Layer-wise and Element-wise Regularization (https://arxiv.org/abs/2501.13669)
Comments:
          Work in progress

- **What's New**: 본 논문은 대규모 언어 모델(LLMs)의 망각 문제(catastrophic forgetting)를 해결하기 위한 새로운 접근 방식을 제안합니다. 일반 지식(general knowledge)을 보존하면서 도메인 특정 태스크에 적응하도록 모델 파라미터의 중요성을 개별적으로 평가하는 방법입니다. 이 방법은 두 가지 손실 함수(Regularization Loss와 Cross-Entropy Loss)를 조합하여 활용하며, 레이어별(coefficient)로 기여도를 조정합니다.

- **Technical Details**: 제안된 방법은 자연어 처리 모델을 위한 고급 최적화 전략을 활용합니다. Regularization Loss는 일반 지식을 유지하기 위해 중요한 파라미터 업데이트를 최소화하고, Cross-Entropy Loss는 도메인 특정 학습을 촉진합니다. 또한, 각 레이어의 기여도를 반영하는 레이어별 коэффициент를 통해 특정 레이어가 태스크 학습을 우선시하거나 일반 지식을 보존할 수 있도록 동적으로 조정합니다.

- **Performance Highlights**: 제안된 방법은 GPT-J 및 LLaMA-3을 사용하는 과학, 의학 및 물리적 태스크 실험에서 탁월한 성능을 보입니다. 기존 방법에 비해 거의 20배 더 빠르며, 저장 공간은 10%~15%만큼의 효율성을 보여줍니다. 이러한 실험 결과는 제안한 접근 방식이 연산 시간과 메모리 요구사항을 크게 줄이면서도 효과적인 성능 개선을 이루었다는 것을 나타냅니다.



### Cognitive Paradigms for Evaluating VLMs on Visual Reasoning Task (https://arxiv.org/abs/2501.13620)
- **What's New**: 이 논문은 Vision-Language Model (VLM)이 복잡한 시각적 과제에서의 추론 능력을 평가하는 데 중점을 두고 있습니다. 특히, Bongard Openworld Problems 벤치마크를 사용하여 자연 이미지를 기반으로 한 추론 문제를 해결하는 모델의 성능을 분석합니다. 세 가지 인간 중심의 접근법인 holistic analysis, deductive rule learning, 및 componential analysis를 제안하며, 이러한 접근법을 이용한 VLM이 인간의 성능을 초과하는 결과를 보여줍니다.

- **Technical Details**: VLM의 성능 평가에는 다양한 모델 아키텍처와 파라미터 스케일을 사용하는 것을 포함합니다. 평가에는 classification accuracy와 semantic similarity와 같은 두 가지 주요 지표가 사용됩니다. 이 연구는 데이터셋과 모델의 세부 정보에 대해 Appendix에 기술하고 있어, 자세한 기술적 배경을 제공합니다.

- **Performance Highlights**: 연구 결과, 최첨단 모델인 GPT-4o와 Gemini는 구조적 추론 작업에서 탁월한 성과를 보여주었으며, 특히 componential analysis가 효과적임을 입증하였습니다. 하지만, 합성 이미지 처리와 미세한 구분을 하는 데에서 주요 어려움을 발견했으며 이는 VLM의 강건성과 일반화의 필요성을 강조합니다. 이러한 통찰은 VLM의 향후 발전 방향을 제시하고 있습니다.



### Efficient Synaptic Delay Implementation in Digital Event-Driven AI Accelerators (https://arxiv.org/abs/2501.13610)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2404.10597

- **What's New**: 이 논문에서는 신경망 모델의 시냅스 지연(synaptic delay) 파라미터화에 대한 새로운 접근 방식을 제안합니다. 특히, Shared Circular Delay Queue (SCDQ)라는 하드웨어 구조를 소개하며 이는 디지털 신경형 가속기에서 시냅스 지연을 지원하기 위한 것입니다. SCDQ 구조는 메모리 측면에서 현재 널리 사용되는 방법보다 더 나은 확장성을 제공하며 알고리즘-하드웨어 공동 최적화에 더 유리한 특성을 지니고 있습니다.

- **Technical Details**: 기존의 기술들에 비해 SCDQ는 네트워크 모델의 크기와 상관없이 메모리 및 영역(Space) 효율성을 극대화할 수 있습니다. 이 구조는 Address Event Representation (AER) 패킷 형식을 간단히 확장하여 작동하며, 이는 신경망 이벤트 간의 통신에 널리 사용됩니다. 시냅스 지연의 도입은 가속기의 전력 및 영역 요구 사항에 거의 영향을 미치지 않는다는 것이 중요한 특징입니다.

- **Performance Highlights**: Seneca 신경형 플랫폼에서 구현된 이 하드웨어의 성능을 평가한 결과, 추론 정확도(inference fidelity), 에너지 소모(Energy per inference), 지연(Latency) 및 면적(Area) 등의 메트릭으로 긍정적인 결과를 보고합니다. 또한 기존 디지털 신경형 시스템과의 비교를 통해 SCDQ의 성능 우수성을 강조하고 있으며, 다른 구조에 비해 더 나은 메모리 스케일링 특성을 나타냅니다.



### Optimal Multi-Objective Best Arm Identification with Fixed Confidenc (https://arxiv.org/abs/2501.13607)
Comments:
          Accepted to AISTATS 2025

- **What's New**: 이번 연구는 Multi-Objective Multi-Armed Bandit (MO-MAB) 설정의 문제를 정식으로 조사하여 각 목표(objective)의 최적 팔(best arm) 식별 문제를 다루고 있습니다. 기존 연구는 주로 Pareto frontier 식별에 집중했으나, 본 연구는 복수 목표의 최적 팔을 찾는 비즈니스 및 실용적 요구를 충족합니다. 이러한 최적 팔의 식별 문제는 다차원 보상 구조와 관련된 고유한 불확실성으로 인해 복잡해집니다.

- **Technical Details**: 연구에서는 각 팔이 여러 차원(M-dimensional) 벡터 보상을 제공하는 설정을 고려하고, 각 차원 보상은 독립적으로 생성된다고 가정합니다. 최적 팔의 식별은 오류 확률이 주어진 한도 내에서 달성되어야 하며, 최적의 정지 시간(expected stopping time)을 줄이는 것이 목표입니다. 저자들은 최대 최소 최적화 문제(max-min optimization problem)를 해결하기 위한 새로운 알고리즘을 제안하였고, 이 알고리즘은 대체 비율(surrogate proportions)을 활용하여 각 단계에서의 계산 복잡성을 줄입니다.

- **Performance Highlights**: 저자들은 제안된 알고리즘이 점근적 최적(asymptotically optimal)임을 이론적으로 증명하고, 광범위한 경험적 연구를 통해 알고리즘의 효율성을 입증했습니다. 기존 문헌이 주로 Pareto 최적 팔 식별에 초점을 맞추고 있는 반면, 본 연구는 각 목표에 대해 최적 팔을 식별하는 과제에 대해 명확한 접근 방식을 제공합니다. 이를 통해 다목적 밴딧 연구의 최신 기술을 한 단계 끌어올리는 성과를 거두었습니다.



### Text-to-SQL based on Large Language Models and Database Keyword Search (https://arxiv.org/abs/2501.13594)
- **What's New**: 이 논문에서는 실제 데이터베이스에 적용할 때 Text-to-SQL (텍스트에서 SQL로) 성능이 저하되는 문제를 다룹니다. 특히 자연어(NL) 질문을 SQL 쿼리로 변환하는 새로운 전략을 제안합니다. 이 전략은 dynamically few-shot examples와 데이터베이스 키워드 검색(KwS) 플랫폼의 서비스를 활용하여 NL 질문을 SQL 쿼리로 컴파일합니다. 이는 실제 데이터베이스에서의 정확도를 선택적으로 높이는 데 큰 기여를 합니다.

- **Technical Details**: 제안된 전략은 KwS 플랫폼인 DANKE의 데이터 사전과 동적 few-shot examples를 결합하는 방식으로, 스키마 링크링(schema-linking) 프로세스의 정밀도 및 재현율을 개선합니다. 이 과정은 NL 질문을 처리하기 위한 필요 테이블 세트를 찾는 데 중점을 두고 진행되며, DANKE가 생성하는 뷰를 사용하여 필요한 조인을 캡처하여 SQL 쿼리 컴파일을 단순화합니다. 섹션 4.4에서는 LLM이 NL 질문을 SQL 쿼리로 컴파일하는 방법도 자세히 설명합니다.

- **Performance Highlights**: 실제로 수행된 실험 결과, 제안된 전략은 LangChain SQLQueryChain, SQLCoder111, 'C3 + ChatGPT + Zero-Shot', 그리고 'DIN-SQL + GPT-4'와 같은 최첨단 접근 방식보다 더 높은 정확도를 달성했습니다. 이러한 실험은 에너지 회사에서 운영 중인 복잡한 스키마를 가진 관계형 데이터베이스를 기반으로 하여 이루어졌습니다. 논문은 이 전략의 성능이 실제 데이터베이스에서 뛰어난 결과를 나타낸다고 결론짓습니다.



### Contrastive Representation Learning Helps Cross-institutional Knowledge Transfer: A Study in Pediatric Ventilation Managemen (https://arxiv.org/abs/2501.13587)
- **What's New**: 이번 연구에서는 기관 간 지식 전이를 위한 체계적인 프레임워크를 제시하며, 일반 소아 집중 치료실(PICU)과 심장 전문 병동 간의 소아 환기 관리를 예시로 보여줍니다. 대조적 예측 부호화(Contrastive Predictive Coding, CPC)를 이용한 표현 학습을 통해 서로 다른 데이터 조건 및 미세 조정 전략이 지식 전이에 미치는 영향을 규명하고 있습니다. 이 연구 결과, 직접적인 모델 이전이 어려운 반면, 적절한 미세 조정을 통해 기관 간 효과적인 지식 공유가 가능하다는 사실이 입증되었습니다.

- **Technical Details**: 이 연구는 두 개의 PICU에서 수집된 환기 데이터를 분석하며, 기관 간의 차이로 인한 자연 장벽에서 모델 일반화를 이끌어내기 위한 방법론을 제시합니다. 정식적으로, 소스 도메인(dataset)에서의 representation learning과 타겟 도메인에서의 모델 적응(fine-tuning)을 통해 지식 전이를 구현합니다. 세 가지 예측 모델 학습 접근 방식을 고려하며, 특히 CPC를 활용한 자기 감독 학습이 임상적 추론에 어떻게 기여하는지를 탐구합니다.

- **Performance Highlights**: 연구의 결과, 기관 간 성능 갭이 존재하지만, 대조적 미리 학습(contrastive pre-training)과 적절한 미세 조정 전략이 이러한 갭을 상당히 줄여줄 수 있음을 보여줍니다. 특히 데이터가 제한된 상황에서 이러한 전략의 효과가 두드러지며, 다양한 임상 과제에서 비대칭적인 전이 패턴이 나타남을 분석했습니다. 이 발견은 임상 결정 지원 시스템의 개발과 함께, 비교적 작은 전문 기관들이 대형 센터의 지식을 활용할 수 있는 실용적인 경로를 제시합니다.



### K-COMP: Retrieval-Augmented Medical Domain Question Answering With Knowledge-Injected Compressor (https://arxiv.org/abs/2501.13567)
Comments:
          NAACL 2025

- **What's New**: 최근 연구에서는 K-COMP (Knowledge-injected compressor)를 제안하여 retrieval-augmented question answering (QA)의 정확성을 높였습니다. 이 방법은 reader model이 질문에 대한 정확한 답변을 위해 필요한 지식을 자동으로 생성하고, 이를 압축된 문맥에 통합하여 질문의 의도와 정렬을 보장합니다. 특히, 이 모델은 의료 도메인에 적합하도록 설계되어 실제 사례에서 효율성을 입증하였습니다.

- **Technical Details**: K-COMP의 핵심은 autoregressive LLM을 사용하여 질문에 필요한 도메인 지식을 주입하는 것입니다. 이를 통해 모델은 질문에 포함된 개체를 인식하고 관련된 정보를 제공할 수 있습니다. 이 과정에서 causal masking 기법을 활용하여, 질의의 맥락에 맞춘 압축된 요약을 생성합니다. 이는 K-COMP가 긴 입력 프롬프트에서 필요한 정보를 효과적으로 찾는 데 도움을 줍니다.

- **Performance Highlights**: 실험 결과 K-COMP는 세 가지 의료 데이터셋에서 기존의 방법들보다 우수한 성능을 보였습니다. 이 방법은 도메인 지식이 없는 reader model이더라도 의학적 전문 용어를 설명할 수 있게 하여, 다양한 배경의 모델들이 의학 질문을 보다 정확하게 처리할 수 있도록 지원합니다. K-COMP는 처음 보는 데이터에 적용했을 때도 효과가 입증되어, 데이터가 부족한 폐쇄형 도메인 환경에서 유용한 기여를 할 수 있음을 보여줍니다.



### Black-Box Adversarial Attack on Vision Language Models for Autonomous Driving (https://arxiv.org/abs/2501.13563)
- **What's New**: 이 논문에서는 Vision-Language Models (VLMs)를 대상으로 한 블랙박스 적대적 공격인 Cascading Adversarial Disruption (CAD)를 제안합니다. 기존 연구들은 주로 화이트박스 공격에 초점을 맞추었지만, 블랙박스 공격의 실제적인 난이도를 극복하기 위해 CAD가 설계되었습니다. CAD는 Driving Reasoning Chain의 효과적인 저하와 동적인 주행 환경에서의 적용을 목표로 합니다.

- **Technical Details**: CAD는 Decision Chain Disruption과 Risky Scene Induction을 활용하여 저수준 사고를 방해하고 동적 환경에 적응할 수 있는 고수준의 위험한 시나리오를 생성합니다. 이러한 접근 방식은 VLM의 오류 전파를 효과적으로 극대화하며, 최종적으로는 운전 안전성을 저해하는 적대적 시각 입력을 생성합니다. CAD는 다양한 AD VLMs에 대해 여러 환경에서 실험을 진행하여 평균 13.43%의 성능 개선을 보였습니다.

- **Performance Highlights**: CAD는 실제 자율주행 차량에 대한 공격 실험에서도 강력한 효과를 입증하였습니다. VLM 기반의 AD 차량에서는 경로 완료율이 61.11% 감소하였고, 적대적 패치가 부착된 차량이 장애물에 충돌하는 사례가 발생했습니다. 또한, CDAD 데이터셋을 공개하여 향후 연구에 기여할 수 있는 기초 자료를 제공하고 있습니다.



### One-Prompt-One-Story: Free-Lunch Consistent Text-to-Image Generation Using a Single Promp (https://arxiv.org/abs/2501.13554)
- **What's New**: 이 논문에서는 'One-Prompt-One-Story' (1Prompt1Story)라는 새로운 훈련이 필요 없는 방법을 제안하며, 이는 단일 프롬프트 내에서 모든 프롬프트를 연결하여 T2I 생성 모델이 일관되게 이미지를 생성하도록 합니다. 이는 기존 모델이 대규모 데이터 세트에 대한 훈련이나 복잡한 수정이 필요하던 문제를 해결합니다. 이 방법은 언어 모델의 고유한 문맥 일관성(context consistency)을 활용하여 캐릭터 정체성을 효과적으로 유지할 수 있습니다.

- **Technical Details**: 1Prompt1Story 접근법은 모든 프롬프트를 하나의 긴 문장으로 통합하고 그 결과로 생성 과정을 두 가지 새롭고 독창적인 기법, 즉 Singular-Value Reweighting(SVR)과 Identity-Preserving Cross-Attention(IPCA)을 사용하는 것입니다. SVR은 현재 프레임의 프롬프트 표현을 정제하고, IPCA는 크로스-어텐션 레이어에서 주체의 일관성을 강화합니다. 이 두 가지 기법은 T2I 생성 모델의 텍스트-이미지 정렬 개선을 목표로 하며, 각 프레임 프롬프트가 개별적으로 표현되는 동시에 정체성을 유지할 수 있도록 합니다.

- **Performance Highlights**: 실험을 통해 우리는 1Prompt1Story 방법이 기존의 다양한 일관된 T2I 생성 접근법들과 비교하여 더 일관된 이미지 생성을 달성했음을 입증하였습니다. 정량적 메트릭과 정성적 평가를 통해 이 방법의 효과성을 보여주었으며, 확장된 ConsiStory+ 벤치마크와의 비교에서도 뛰어난 성능을 발휘하였습니다. 이 연구는 T2I 생성의 새로운 가능성을 열어주며, 다양한 내러티브 기반 비주얼 애플리케이션에 적용할 수 있는 길을 제시합니다.



### Explainable AI-aided Feature Selection and Model Reduction for DRL-based V2X Resource Allocation (https://arxiv.org/abs/2501.13552)
- **What's New**: 이번 논문에서는 6세대(6G) 네트워크에서 AI를 활용한 새로운 설명 가능한 AI(XAI) 기반 프레임워크를 제안하여 특성 선택(feature selection)과 모델 복잡도(Model Complexity) 저감을 도모하고 있습니다. Multi-Agent Deep Reinforcement Learning(MADRL) 환경에 적용한 이 연구는 셀룰러 차량-모든 것(V2X) 통신에서 서브밴드 할당(sub-band assignment)과 전력 할당(power allocation) 문제를 해결합니다. 또한, 이 프레임워크는 SHAP(Shapley Additive Explanations)를 활용한 두 단계의 설명 가능성 체계를 통해 DRL 에이전트의 상태 공간을 단순화합니다.

- **Technical Details**: 연구에서는 셀룰러 차량-모든 것(C-V2X) 네트워크의 복잡한 자원 관리 문제를 해결하기 위해 MADRL 알고리즘을 도입했습니다. 이 알고리즘은 SHAP 기반의 중요도 점수를 통해 훈련된 모델의 상태 특성 중요도를 평가하고, 덜 중요한 특성을 제거하여 에이전트의 상태 공간을 축소합니다. 실험 결과, 이 방법론은 원본 MADRL의 성능을 97% 유지하면서도 최적의 상태 특성을 28%, 평균 훈련 시간을 11%, 학습 가능한 가중치 매개변수를 46% 감소시킴을 보였습니다.

- **Performance Highlights**: XAI 지원 방법론을 통해 논문에서는 기존 DRL 기반 자원 할당 방식의 성과를 유지하면서도 모델을 간소화하여 자원의 효율성을 높이는 성과를 보여주었습니다. 이 연구는 6G 네트워크와 C-V2X 통신의 복잡한 요구사항을 충족할 수 있는 새로운 방향성을 제공합니다. 특히, V2X 시나리오에서 낮은 지연과 높은 신뢰성을 요하는 환경에서의 자원 관리 및 할당 문제 해결에 유의미한 진전을 이루었습니다.



### LLMs Can Plan Only If We Tell Them (https://arxiv.org/abs/2501.13545)
Comments:
          ICLR 2025

- **What's New**: 이번 연구는 대규모 언어 모델(Large Language Models, LLMs)이 자율 계획에서 독립적으로 인간 수준의 장기 계획(long-horizon plans)을 생성할 수 있는지를 조사합니다. 특히, 이전 연구들에서는 외부 피드백 메커니즘과 통제된 환경에서 LLM을 활용했지만, 이러한 방법들은 상당한 계산 자원과 개발 자원을 요구했습니다. 이를 신경 쓰지 않고, LLM의 독립적인 계획 능력을 평가하기 위한 새로운 접근 방식을 채택하였습니다.

- **Technical Details**: 연구에서는 'Algorithm-of-Thoughts(AoT)'라는 기법에 혁신적인 개선을 도입하여 이를 'AoT+'라고 명명하였습니다. AoT+는 고급 계획 벤치마크에서 기존 방법들과 인간 기준을 능가하는 최신 성과(state-of-the-art results)를 달성하는 데 도움을 줍니다. 이 접근 방식은 별도의 지원 없이 모든 작업을 독립적으로 수행할 수 있도록 설계되었습니다.

- **Performance Highlights**: AoT+는 이전의 다른 방법들 및 인간의 성능과 비교했을 때 자율적으로 우수한 성과를 보였습니다. 특히 Blocksworld와 같은 표준 계획 벤치마크에서 LLM이 여전히 인간 성능을 초과하는 데 어려움을 겪고 있다는 점을 고려할 때, AoT+의 성과는 눈에 띕니다. 이는 LLM의 자율적 계획 능력을 크게 향상시킨 결과라 할 수 있습니다.



### GCAD: Anomaly Detection in Multivariate Time Series from the Perspective of Granger Causality (https://arxiv.org/abs/2501.13493)
Comments:
          Accepted to AAAI 2025

- **What's New**: 이번 논문은 다변수 시계열 이상 감지를 위한 새로운 프레임워크를 제안합니다. 기존의 예측 및 재구성 과제를 기반으로 한 방법들과 달리, 우리는 인과적 관계를 해석 가능하게 모델링함으로써 이상 감지를 수행합니다. 특히, 비선형 딥 예측기에서 기울기를 활용하여 변화된 Granger causality를 동적으로 발견하고, 이를 통해 인과적 관점에서의 이상 감지 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 Granger causality를 기반으로 하며, 비선형 딥 모델의 기울기를 사용하여 동적으로 인과적 종속성을 탐색합니다. 데이터를 통해 배운 원래 패턴에서의 이탈을 기반으로 이상 점수를 계산하여, 최종적으로 다섯 개의 실제 벤치마크 데이터 세트에서 우수한 성능을 입증하고자 하였습니다. 이와 함께, 대칭 기반 희소화 방법을 통해 그래프의 양방향 엣지를 제거하고 인과그래프를 구성하여 더 해석 가능한 결과를 도출하였습니다.

- **Performance Highlights**: 실험 결과, 제안하는 모델은 기존의 기준 방법들보다 훨씬 높은 정확도의 이상 감지를 달성했습니다. 특히, 다섯 개의 실제 벤치마크 데이터 세트에서 state-of-the-art 결과를 보여주었으며, 기울기 기반의 Granger causality 효과 계량화 방법이 모델의 성능 향상에 크게 기여하였습니다. 이 연구는 딥 모델과 Granger causality의 융합을 통한 시계열 이상 감지의 새로운 가능성을 열었습니다.



### RECALL: Library-Like Behavior In Language Models is Enhanced by Self-Referencing Causal Cycles (https://arxiv.org/abs/2501.13491)
- **What's New**: 본 논문에서는 self-referencing causal cycle(자기 참조 인과 주기)이라는 개념을 도입했습니다. 이는 대규모 언어 모델(LLM)이 단방향 인과 관계의 한계를 넘어설 수 있도록 하는 메커니즘입니다. 특히 RECALL이라는 방법론이 제안되며, 이 방법은 순차적 데이터를 처리하는 과정에서 발생하는 "reversal curse" 현상을 극복하도록 돕습니다.

- **Technical Details**: RECALL의 핵심은 cycle tokens(주기 토큰)으로, 이는 학습 데이터의 다양한 부분을 연결하는 시퀀스를 생성합니다. 이 주기 토큰은 모델이 후속 토큰으로부터 이전 토큰을 회상할 수 있도록 돕습니다. 논문은 확률적 형식화 및 통제된 실험을 통해 이러한 주기의 유도 방식이 정보 재생산 능력에 어떻게 영향을 미치는지를 분석합니다.

- **Performance Highlights**: RECALL 메커니즘을 통해 대규모 언어 모델의 정보 검색 능력이 향상됩니다. 특히, 자율회귀 모델에 대한 새로운 두 단계의 재귀 회수 과정을 제안하여 유용한 정보를 보다 효과적으로 회상해 내는 방식이 설명됩니다. 또한, 저자들은 이를 실재할 수 있는 코드 및 실험적 세부정보를 공개하여 재현 가능성을 강조하였습니다.



### MambaQuant: Quantizing the Mamba Family with Variance Aligned Rotation Methods (https://arxiv.org/abs/2501.13484)
- **What's New**: MambaQuant는 Mamba 모델을 위한 최초의 포괄적인 후처리 양자화(PTQ) 프레임워크로, 학습 후 양자화의 효과를 극대화하는 기술을 제시합니다. Mamba 모델의 양자화 과정에서 발생하는 주요 문제로는 큰 이상치(outliers)와 변동성(variance) 불일치가 있습니다. 이러한 문제에 대응하기 위해 MambaQuant는 Karhunen-Loeve 변환(KLT)과 스무스-퓨즈드 회전(rotation) 기법을 도입하여 딥러닝 모델이 손실 없이 양자화될 수 있도록 지원합니다.

- **Technical Details**: MambaQuant는 두 가지 주요 기술을 적용합니다. 첫 번째는 오프라인 모드에서 KLT를 활용한 회전으로, 다양한 채널 분포에 맞춘 회전 행렬(rotation matrix)을 생성합니다. 두 번째는 온라인 모드에서 스무스-퓨즈드 회전을 구현하여 채널 분산을 정규화하여 메모리 비용을 최소화합니다. 이 두 가지 접근법은 양자화 데이터의 최댓값과 분산을 일관되게 유지하여 성능을 향상시킵니다.

- **Performance Highlights**: MambaQuant는 Mamba 기반의 비전 및 언어 작업에서 8비트 양자화 시 1% 미만의 정확도 손실로 우수한 성능을 발휘합니다. 또한, 비전 작업에서는 4비트로 양자화하면서도 1%의 최소 정확도 손실로 뛰어난 결과를 나타냅니다. 기존 방법들과 비교해 MambaQuant는 언어 작업에서도 유의미한 정확도 향상을 보이며, 향후 연구의 기초가 될 것으로 기대됩니다.



### A Polynomial-Time Algorithm for EFX Orientations of Chores (https://arxiv.org/abs/2501.13481)
Comments:
          8 pages

- **What's New**: 이 논문은 모든 일감(chores)만으로 구성된 그래프가 EFX(Envy-Freeness up to any good) 방향을 허용하는지 결정하는 문제를 다루고 있습니다. 저자들은 이 문제가 NP-완전하다는 Zhou et al.의 추측을 해결하고, 존재할 경우 EFX 방향을 찾는 다항시간 알고리즘을 제시했습니다. 이 알고리즘은 자기 루프가 포함된 그래프에서도 작동하며, EFX 방향의 존재 여부를 2SAT 문제로 축소하여 해결합니다. 이는 모든 것(good)만으로 이루어진 그래프의 경우 NP-완전성이 입증된 것과는 대조적입니다.

- **Technical Details**: 본 연구에서는 각 에이전트가 하나의 정점으로 표현되고, 각 일감이 에이전트 간의 변으로 표현되는 그래프 G를 모델로 사용합니다. 기존의 연구에서는 EFX 방향성을 결정하는 것이 NP-완전하다고 언급되었으나, 본 논문은 O(|V(G)|²) 시간복잡도로 EFX 방향의 존재를 판단하고 찾아내는 알고리즘을 제안합니다. EFX0는 EFX_보다 더 강한 조건으로, 제로 한계 효용을 갖는 일감을 무시해도 시기심을 덜어줄 수 있어야 합니다. 이 알고리즘은 단순하고 효율적이며, 자기 루프를 처리할 수 있는 특성도 가지고 있습니다.

- **Performance Highlights**: 이 연구의 주요 성과는 모든 일감 그래프가 EFX0 방향을 허용하는지를 결정할 수 있는 다항시간 알고리즘을 찾았다는 점입니다. 또한, 다중 그래프(multigraph)의 유사한 문제는 여전히 NP-완전하다는 것을 보여주었습니다. 연구 결과는 EFX 방향에 대한 이해를 심화시키고, 다양한 실제 문제에서의 공정한 자원 분배 방법에 대한 통찰을 제공합니다. 본 알고리즘은 그래프의 구조적 특성을 활용하여 효율적으로 문제를 해결하는 데 중요한 기여를 하고 있습니다.



### Adaptive Testing for LLM-Based Applications: A Diversity-based Approach (https://arxiv.org/abs/2501.13480)
Comments:
          9 pages

- **What's New**: 이 논문에서는 대형 언어 모델(LLM)을 기반으로 한 소프트웨어 시스템의 테스트 프레임워크에 대한 새로운 방법론을 제안하고 있습니다. 특히, 테스트 입력 실행과 출력 평가에 드는 비용을 줄이기 위해 최적화된 테스트 스위트의 선별 및 우선순위 정하기에 대한 중요성을 강조하고 있습니다. 적응형 랜덤 테스트(Adaptive Random Testing, ART)를 활용한 다양한 다변량 테스트 접근 방식이 LLM의 프로프트 템플릿 테스트에 효과적으로 적용될 수 있음을 보여줍니다.

- **Technical Details**: 제안된 적응형 테스트 접근법은 기존의 ART 과정을 기반으로 하며, 이전 실행 입력의 성과에 따라 새로운 테스트 입력을 선택하는 방식으로 조정됩니다. 여러 문자열 거리 메트릭을 탐색한 결과, Normalized Compression Distance(NCD)가 특히 효과적임을 나타내며, 이를 통해 실패 탐지율이 평균 7.24% 개선되었습니다. 이러한 방법론은 기존 LLM의 블랙박스 테스트 프레임워크에 통합될 수 있으며, 효율적인 테스트 프로세스에 기여합니다.

- **Performance Highlights**: 임상 평가에서는 46개의 프로프트 템플릿에 대해 적응형 다변량 테스트를 적용하여 비용 제약이 있는 상황에서도 더 많은 결함을 탐지할 수 있었음을 보여줍니다. NCD를 이용한 거리 기반 테스트 우선순위 결과, 평균적으로 9.5% 더 많은 고유 단어를 포함한 출력이 생성되었으며, 특정 작업과 입력 분포에 따라 거리 메트릭의 효과가 다르게 나타나는 것으로 확인되었습니다. 이러한 결과는 LLM을 활용한 소프트웨어 테스트의 효율성과 효과성을 높이는 기초 자료를 제공합니다.



### Adaptive Few-Shot Learning (AFSL): Tackling Data Scarcity with Stability, Robustness, and Versatility (https://arxiv.org/abs/2501.13479)
- **What's New**: 이 논문에서는 기존의 Few-Shot Learning (FSL)의 한계를 극복하기 위해 Adaptive Few-Shot Learning (AFSL) 프레임워크를 제안합니다. AFSL은 메타 학습, 도메인 정렬, 노이즈 저항력, 다중 모달 통합을 통합한 모듈형 아키텍처로, 데이터가 부족한 분야에서도 효과적으로 일반화할 수 있는 모형을 구축합니다. 이 프레임워크는 성능 일관성을 보장하기 위한 Dynamic Stability Module, 도메인 적응을 위한 Contextual Domain Alignment Module, 노이즈 데이터를 처리하기 위한 Noise-Adaptive Resilience Module, 다양한 모달을 통합하는 Multi-Modal Fusion Module로 구성되어 있습니다.

- **Technical Details**: AFSL은 네 가지 주요 모듈로 구성되며, 각 모듈은 특정 병목 문제를 해결합니다. Dynamic Stability Module은 앙상블 기반 메타 학습을 활용하여 예측의 일관성을 높입니다. Contextual Domain Alignment Module은 원천 도메인과 타겟 도메인 간의 특징을 정렬하여 도메인 간 이동을 용이하게 합니다. Noise-Adaptive Resilience Module은 노이즈가 많은 데이터를 처리하기 위해 주의를 기반으로 한 가중치와 노이즈 인식 손실 함수를 이용해 강인성을 향상시킵니다.

- **Performance Highlights**: AFSL은 성능, 안정성 및 강인성에서 기존 FSL 방법론에 비해 상당한 개선을 보입니다. 이 모듈형 접근 방식은 헬스케어, 로보틱스, 자연어 처리와 같은 중요한 분야에서 보다 신뢰할 수 있는 솔루션을 제공합니다. AFSL의 도입으로, 다양한 상황에 적응할 수 있는 모델의 능력이 강화되어 실제 어플리케이션의 확장성과 신뢰성을 높일 수 있습니다.



### Streaming Video Understanding and Multi-round Interaction with Memory-enhanced Knowledg (https://arxiv.org/abs/2501.13468)
Comments:
          Accepted to ICLR 2025. Code is available at this https URL

- **What's New**: 최근 Large Language Models (LLMs)의 발전은 Video-LLMs의 개발로 이어졌으며, 이는 비디오 데이터와 언어 작업을 결합하여 다중 모달 학습을 증진시키고 있습니다. 하지만 기존 비디오 이해 모델은 긴 비디오 시퀀스를 처리하는 데 어려움을 겪고 있으며, 다중 회전 대화 및 현실 세계의 동적 시나리오에 적응하는 데 한계를 보이고 있습니다. 이러한 문제를 해결하기 위해, 우리는 StreamChat이라는 훈련이 필요 없는 프레임워크를 제안합니다.

- **Technical Details**: StreamChat은 복잡한 비디오 특징을 효율적으로 처리하고 압축할 수 있도록 새로운 계층 메모리 시스템을 활용하여 실시간 다중 회전 대화를 가능하게 합니다. 이 프레임워크는 프로세싱 속도를 높이고 대기 시간을 줄이는 병렬 시스템 스케줄링 전략을 포함하여 현실적인 애플리케이션에서 강력한 성능을 보장합니다. 또한, StreamBench라는 다재다능한 벤치마크를 통해 다양한 미디어 유형과 상호작용 시나리오에서 스트리밍 비디오 이해를 평가합니다.

- **Performance Highlights**: StreamChat은 StreamBench 및 기타 공개 벤치마크에 대해 광범위한 평가를 수행하여 기존 최첨단 모델들에 비해 정확도와 응답 속도에서 현저한 성과를 보여줍니다. 특히, 온라인 설정에서 StreamBench에서 64.7%의 정확도를 기록하며, 이는 이전 최고 기록에 비해 8.3% 향상된 수치입니다. 오프라인 시나리오에서는 공공 벤치마크 4개에서 평균 2.5% 더 뛰어난 성능을 보이며, 스트리밍 비디오 처리의 효율성에서도 32 FPS의 처리 속도를 달성하여 기존 기법보다 6배 향상된 결과를 보여줍니다.



### Zero-Shot Trajectory Planning for Signal Temporal Logic Tasks (https://arxiv.org/abs/2501.13457)
Comments:
          submitted

- **What's New**: 이 논문에서는 기존의 시스템 동역학에 대한 지식이 없는 상황에서도 실행 가능한 신호 시간 논리(Signal Temporal Logic, STL) 계획을 생성할 수 있는 새로운 프레임워크를 제안합니다. 기존 방법들이 특정 작업을 위한 계획을 생성하기 위해 모델 기반 접근법이나 데이터 중심 접근법에 의존했던 반면, 본 연구는 성능을 위해 작업 일반화(zero-shot generalization) 능력을 극대화했습니다. 이를 통해 첫 데이터 기반(zero-shot) STL 작업에 대해 성공적인 결과를 보여주고 있습니다.

- **Technical Details**: 제안된 계획 프레임워크는 계층적(hierarchical) 구조를 갖추고 있으며, STL 작업을 진척(progress)과 시간 제약(time constraints)으로 분해하는 것에 초점을 맞추고 있습니다. 이어서, 작업에 상관없는(task-agnostic) 데이터를 기반으로 한 해법을 통해 타임라인을 고려한 웨이포인트(waypoints)를 생성하고, 사전 훈련(pre-trained)된 안전한 확산 모델을 활용하여 원하는 경로를 생성합니다. 이러한 방법론은 STL 작업을 위한 복잡한 시간 제약을 처리하기 위해 고안되었습니다.

- **Performance Highlights**: 시뮬레이션 결과, 제안된 방법이 다양한 STL 작업에서 높은 성공률을 기록함을 입증하였습니다. 복잡한 STL 작업에서도 기존 비데이터 기반 방법들보다 효율성에서 우수한 성능을 발휘한다는 것을 보여주었습니다. 이 연구의 결과는 STL 작업을 계획하는 데 있어 새로운 가능성을 제시하며, 학습 기반 접근법의 효율성을 강조합니다.



### KAA: Kolmogorov-Arnold Attention for Enhancing Attentive Graph Neural Networks (https://arxiv.org/abs/2501.13456)
- **What's New**: 이 논문에서는 기존 GNN의 스코어링 함수에 대한 이해 부족 문제를 해결하기 위해, Kolmogorov-Arnold Attention (KAA)라는 새로운 스코어링 방법을 제안합니다. KAA는 Kolmogorov-Arnold Network (KAN) 아키텍처를 통합하여, 기존의 모든 attentive GNNs에 적용 가능하며, 성능을 20% 이상 향상시킬 수 있는 가능성이 있음을 입증합니다.

- **Technical Details**: KAA는 기존 GNN 모델의 스코어링 함수를 통합하는 새로운 프레임워크를 제시하고, Maximum Ranking Distance (MRD)를 도입하여 노드 중요도를 순위화하는 오류의 한계를 정량적으로 측정합니다. KAA는 단일 레이어 KAN을 활용해 높은 표현력을 제공하며, 기존의 선형 변환 기반 및 MLP 기반 스코어링 함수에 비해 유의미한 성능 향상을 보여줍니다.

- **Performance Highlights**: 실험 결과, KAA로 개선된 스코어링 함수는 다양한 백본 모델 및 과제에서 일관되게 기존의 스코어링 함수보다 뛰어난 성능을 보였으며, 일부 경우에는 20% 이상의 성능 향상이 있었습니다. 이러한 성과는 KAA가 노드 레벨 및 그래프 레벨 작업 모두에서 강화된 효과를 증명함으로써, 기존 모델의 한계를 극복하는 데 기여함을 보여줍니다.



### BMG-Q: Localized Bipartite Match Graph Attention Q-Learning for Ride-Pooling Order Dispatch (https://arxiv.org/abs/2501.13448)
- **What's New**: 이 논문은 승차 공유 주문 배치를 위해 설계된 새로운 Multi-Agent Reinforcement Learning (MARL) 알고리즘 프레임워크인 Localized Bipartite Match Graph Attention Q-Learning (BMG-Q)을 소개합니다. BMG-Q는 Markov Decision Process에 기반한 지역화된 이분 매칭 그래프를 통해 승차 공유 의사결정 과정을 향상시키고, Graph Attention Double Deep Q Network (GATDDQN)를 통해 차량 간 동적인 상호작용을 포착합니다. 이 접근법은 Integer Linear Programming (ILP)을 이용해 글로벌 중앙 조정자가 최적의 주문 매칭과 에이전트 행동을 통해 운영 효율성을 증대 시킵니다.

- **Technical Details**: BMG-Q는 GATDDQN을 통해 에이전트의 상태 정보를 강화하고, 지역화된 이분의존 그래프(localized bipartite interdependence graph)를 활용하여 차량 간의 복잡한 상호 의존성을 포착합니다. 이 알고리즘은 경량 그래디언트 클리핑(gradient clipping)과 지역 그래프 샘플링(localized graph sampling) 기술을 적용하여 확장성과 견고성을 개선하였습니다. 또한, posterior score 함수의 도입은 온라인 탐색-착취(trade-off)를 통해 에이전트의 잠재적인 과대 추정 편향(overestimation bias)을 줄여 해결합니다.

- **Performance Highlights**: BMG-Q는 수천 대의 차량 에이전트에 대한 훈련과 운영에서 우수한 성능을 입증하였으며, 기존 강화 학습 기법 대비 약 10%의 누적 보상 증가를 기록하였습니다. 과대 추정 편향이 50% 이상 감소하는 등의 개선을 보여 운영 중에도 견고함을 유지합니다. 이러한 성과는 BMG-Q가 승차 공유 주문 배치 작업을 진전시키기 위한 효과적이고 확장 가능하며 견고한 프레임워크임을 확인시켜 줍니다.



### One-cycle Structured Pruning with Stability Driven Structure Search (https://arxiv.org/abs/2501.13439)
Comments:
          12 pages, 6 figures

- **What's New**: 이 논문에서는 기존의 복잡한 다단계 훈련 절차를 간소화하여 성능 저하 없이 효율적인 'one-cycle structured pruning' 프레임워크를 제안합니다. 이 방법은 사전 훈련, 프루닝(pruning), 그리고 미세 조정(fine-tuning)을 하나의 훈련 사이클로 통합하여 훈련 비용을 절감합니다. 새로운 pruning 지표를 도입하여 훈련 에폭(epoсh) 간의 유사성을 평가함으로써 안정적인 pruning 시점을 결정합니다.

- **Technical Details**: 제안된 방법은 초기 훈련 단계에서 최적의 서브 네트워크(sub-network)를 탐색하며, 이는 norm 기반의 그룹 중요도(saliency) 기준과 구조적 희소성(structured sparsity) 정규화를 통해 이뤄집니다. 구조적 희소성 정규화는 프루닝 프로세스를 가속화하며, 이는 전체 훈련 시간을 단축시킵니다. 여러 데이터셋(CIFAR-10/100, ImageNet)에서 VGGNet, ResNet, MobileNet 및 ViT 아키텍처를 사용하여 실험을 통해 효과성을 입증하였습니다.

- **Performance Highlights**: 제안된 알고리즘은 ResNet50 모델에서 ImageNet 데이터셋에 대해 75.49%의 top-1 및 92.63%의 top-5 정확도를 달성하였습니다. 또한, 기본 훈련에 비해 1.38배의 훈련 속도 향상을 이루었으며, 네트워크의 플로팅 포인트 연산(FLOPs)을 57% 이상 감소시켰습니다. 이는 훈련 시간 측면에서 가장 효율적인 pruning 프레임워크 중 하나로 자리 잡을 수 있는 가능성을 의미합니다.



### Softplus Attention with Re-weighting Boosts Length Extrapolation in Large Language Models (https://arxiv.org/abs/2501.13428)
Comments:
          11 pages and 2 figures

- **What's New**: 이 논문은 전통적인 Softmax 주의 메커니즘의 수치적 불안정성과 긴 토큰 길이에서의 성능 저하 문제를 해결하기 위해 새로운 주의 메커니즘을 제안합니다. 새롭게 개발된 Length Scaled Softplus Attention (LSSA)은 비선형 변환을 Softplus 활성화 함수로 대체하고, 토큰 길이에 따른 동적 길이 스케일 요인을 도입하여 성능을 향상시킵니다. 이 접근법은 특히 긴 시퀀스에 대한 처리 성능을 개선하고, 유의한 주의 가중치를 증폭시키며 약한 가중치를 줄이는 재조정 메커니즘을 통합하여 모델의 집중도를 높입니다.

- **Technical Details**: 논문에서 제안한 LSSA는 쿼리(𝐐), 키(𝐊), 값(𝐕)의 스케일 점곱 주의 메커니즘을 기반으로 하며, 모든 입력이 L×d 형상의 벡터로 주어집니다. 기존의 Softmax 연산 대신 다양한 활성화 함수를 실험한 결과, 비포화 기능의 필요성이 낮아진다는 것을 발견하였습니다. 특히, l1-norm은 성능 유지에 필수적인 요소로 관찰되었으며, Softmax의 수학적 표현이 비선형 변환과 l1-norm으로 분해됩니다.

- **Performance Highlights**: LSSA 메커니즘은 훈련 시퀀스 길이뿐만 아니라 훨씬 긴 시퀀스에서도 표준 주의 메커니즘보다 뛰어난 성능을 보입니다. 특히 16배 긴 훈련 토큰 길이에서도 검증 손실이 거의 일정하게 유지되며 수치적 안정성을 확보했습니다. 실험 결과, 제안된 재조정 메커니즘이 다양한 주의 변형과 통합될 때 성능 개선으로 이어짐을 입증합니다.



### Rethinking the Sample Relations for Few-Shot Classification (https://arxiv.org/abs/2501.13418)
Comments:
          32 pages

- **What's New**: 이 논문에서는 Few-Shot Learning (FSL)에서 샘플 관계를 다양한 세분화 수준에서 정교하게 모델링하여 특성 품질을 향상시키는 Multi-Grained Relation Contrastive Learning (MGRCL)이라는 새로운 대조 학습 접근법을 제안합니다. 기존의 몇 가지 단점, 특히 고정된 모델링 접근법이 다른 샘플 관계에 대해 세멘틱 유사성 차이를 간과하는 문제를 해결하려고 합니다. MGRCL은 샘플 관계를 세 가지 유형으로 분류하며, 이를 통해 더 효과적인 피처 학습을 가능하게 합니다.

- **Technical Details**: MGRCL은 세 가지 샘플 관계 유형: 동일 샘플의 내부 관계, 동질 샘플의 클래스 관계, 비동질 샘플의 클래스 간 관계를 도입합니다. Transformation Consistency Learning (TCL)과 Class Contrastive Learning (CCL)을 설계하여 서로 다른 변환에서의 샘플의 의미적 일관성을 보장하고, 특징 간 상대적 거리를 규제하여 비동질 샘플 및 동질 샘플 간의 구별을 유지합니다. 이러한 접근법은 인지적 차별화와 모델 붕괴를 방지합니다.

- **Performance Highlights**: MGRCL은 miniImageNet, tieredImageNet, CIFAR-FS, CUB-200-2011을 포함한 네 가지 벤치마크에서 광범위한 실험을 통해 최근의 여러 방법에 비해 우수한 성능을 보였습니다. 이 방법은 두 단계의 메타 학습 방법 및 생성 가능한 방법의 성능을 크게 향상시킬 수 있는 좋은 사전 훈련 모델로 활용될 수 있습니다. 따라서 FSL의 발전에 중요한 기여를 할 것으로 기대됩니다.



### M3PT: A Transformer for Multimodal, Multi-Party Social Signal Prediction with Person-aware Blockwise Attention (https://arxiv.org/abs/2501.13416)
- **What's New**: 이 연구는 M3PT (Multi-Modal, Multi-Party Transformer)라는 causal transformer 아키텍처를 소개하며, 이는 복수의 사회적 신호를 동시에 처리할 수 있게 설계되었습니다. 특히, modality-specific와 temporal attention masking 기능을 통해 여러 참여자의 신호를 효율적으로 예측할 수 있는 점에서 혁신적입니다. 이 접근 방식은 개인 간 사회적 신호의 시간적 상호작용을 포착하여 보다 동적인 사회적 역학을 이해하는 데 기여합니다.

- **Technical Details**: M3PT 모델은 사회적 신호 예측을 위한 블록 단위( blockwise )의 attention masking 기술을 사용하며, 이는 다양한 모달리티의 입력을 처리하고 이전의 상호작용 이력을 고려합니다. 각 모달리티 입력은 벡터 양자화 기반의 자동 인코더를 통해 토큰화되어 처리됩니다. 연구는 Human-Human Commensality Dataset (HHCD)에서 다양한 사회적 신호를 예측하는 방식을 검증하였고, 여기서 식사 상황에서의 상호작용이 주요 사례로 사용되었습니다.

- **Performance Highlights**: M3PT는 다중 모달리티를 활용하여 bite timing(음식 물기 시점)과 speaking status(발언 상태)의 예측 성능을 향상시키는 데 성공했습니다. 연구 결과는 여러 모달리티를 포함하는 것이 사회적 신호를 예측하는 데 중요하다는 것을 보여주며, 실험을 통해 더 긴 시간적 맥락과 시간 단위(chunking)의 역할이 사회적 신호 예측에 기여함을 입증했습니다.



### Load and Renewable Energy Forecasting Using Deep Learning for Grid Stability (https://arxiv.org/abs/2501.13412)
- **What's New**: 본 논문은 에너지원의 통합과 관련하여 그리드 운영자들이 직면한 최신 도전 과제를 다룹니다. 특히, 태양광 및 풍력 에너지가 불규칙함에 따라 공급과 수요의 균형을 맞추는 것이 가장 큰 문제로 떠오릅니다. 이를 해결하기 위해 신뢰할 수 있는 단기 예측 기법이 필요하며, 최근에는 기계 학습 및 깊은 학습 방식이 주목받고 있습니다.

- **Technical Details**: 본 연구에서는 부하 및 재생 가능 에너지 예측을 위한 CNN(Convolutional Neural Network) 및 LSTM(Long Short-Term Memory) 기반 방법에 중점을 둡니다. 기존의 물리적 모델과 통계적 기법 대신, 이러한 기계 학습 방식이 에너지 예측에서 유망한 결과를 보이고 있습니다. CNN은 이미지 처리에 강점을 가지며, LSTM은 시간 순서 데이터를 효과적으로 처리하는데 유리하게 설계되었습니다.

- **Performance Highlights**: 딥 러닝 기술인 CNN과 LSTM은 재생 가능한 에너지 예측의 정확도를 높이는 데 큰 역할을 하고 있습니다. 특히, 불확실한 환경에서 신뢰할 수 있는 예측을 제공함으로써 에너지 저장을 극대화하고, 재생 가능 자원의 효과적인 사용을 보장하는 데 기여하고 있습니다. 이러한 접근 방식은 에너지 그리드의 안정성을 강화하는 데 필수적입니다.



### YOLOv8 to YOLO11: A Comprehensive Architecture In-depth Comparative Review (https://arxiv.org/abs/2501.13400)
Comments:
          submitted to Journal of Applied Engineering and Technological Science

- **What's New**: 본 연구는 YOLO 모델의 최신 버전인 YOLOv8부터 YOLO11까지의 구조를 종합적이고 심층적으로 비교합니다. 각 YOLO 모델에 대한 심층 분석을 통해 모형의 작동 방식과 상호 간의 차이점을 간결하게 이해할 수 있도록 합니다. 이러한 비교는 심층 학습 기반 컴퓨터 비전 분야에서 YOLO의 발전 속도가 얼마나 빠른지를 보여주는 데 기여합니다.

- **Technical Details**: YOLO의 각 버전은 구조와 특징 추출 기술에서 개선점을 가지고 있지만, 특정 블록은 여전히 동일하게 유지되고 있다는 점이 강조됩니다. 이 연구는 관련 학술지, 문서, 그리고 소스 코드를 면밀히 조사하여 각 YOLO 버전의 아키텍처를 분석하였습니다. YOLO 모델을 초기부터 현재까지의 진화 경로를 명확히 이해할 수 있도록 돕습니다.

- **Performance Highlights**: 연구의 결과, 공식적인 아키텍처 다이어그램과 학술 출판물이 결여되어 있다는 점이 YOLO 모델의 기능성과 향후 개선 점에 대한 이해를 어렵게 하고 있음을 알 수 있었습니다. 이러한 자료의 부족은 개발자들에게 리소스를 제공하기 위한 필요성을 강조합니다. 향후 YOLO의 개선과 활용도를 높이기 위해 개발자들은 자세한 정보와 자료를 제공해야 합니다.



### Concurrent Learning with Aggregated States via Randomized Least Squares Value Iteration (https://arxiv.org/abs/2501.13394)
- **What's New**: 오늘날의 강화학습(RL) 연구는 복잡한 환경에서 효율적으로 탐색하는 학습 에이전트를 설계하는 데 주안점을 두고 있습니다. 본 논문은 동시적으로 작동하는 여러 에이전트가 환경을 탐색하는 데 랜덤화(Randomization) 기법이 도움이 된다는 이론적 결과를 제시합니다. 이를 통해 랜덤화된 최소 제곱 값 반복(Randomized Least-Squares Value Iteration, RLSVI) 알고리즘의 동시 학습(framework)을 조정하여 공간 복잡성을 크게 줄이는 방법을 설명하고 있습니다.

- **Technical Details**: 연구에서 제안된 RLSVI는 에이전트의 이전 궤적(past trajectories)에 가우시안 노이즈를 주입하여 랜덤화된 값 함수를 학습합니다. 이 프레임워크는 동시 학습을 통해 N개의 에이전트가 상호작용을 공유하고 최적 정책에 대한 성능을 향상시키는 방식을 탐구했습니다. 또한, 유한 및 무한한 결정 지평선에서 폴리노미얼 차원의 최악의 실수값 경계에 대한 이론적 근거를 수립하였습니다.

- **Performance Highlights**: RLSVI 알고리즘의 경우, 유한한 및 무한한 지평선 환경에서 개별 에이전트에 대한 회귀(regret)는 최적 속도로 $	heta(1/	ext{sqrt{N}})$ 감소하여 동시 학습의 이점을 강조합니다. 기존의 알고리즘들에 비해 공간 복잡성을 K의 배수로 줄이면서도 최악의 회귀값 경계를 $	ext{sqrt{K}}$만큼 증가시키는 결과를 보여주었습니다. 이론적 발견을 실증하기 위해 수행한 수치 실험에서도 긍정적인 결과가 나타났습니다.



### Generative Data Augmentation Challenge: Zero-Shot Speech Synthesis for Personalized Speech Enhancemen (https://arxiv.org/abs/2501.13372)
Comments:
          Accepted to ICASSP 2025 Satellite Workshop: Generative Data Augmentation for Real-World Signal Processing Applications

- **What's New**: 이번 연구는 ICASSP 2025의 Generative Data Augmentation 워크숍에 부합하는 새로운 챌린지를 소개합니다. 이 챌린지는 제로샷 텍스트 투 스피치(zero-shot text-to-speech, TTS) 시스템을 이용해 개인화된 음성 데이터를 증강하여, 개인화된 음성 강화(personalized speech enhancement, PSE) 작업의 성능을 향상시키고자 합니다. 개인화된 데이터 수집이 개인정보 보호 및 녹음의 기술적 어려움으로 인해 도전적이라는 점을 배경으로 하고 있습니다. 이를 해결하기 위해 생성 모델을 사용한 합성 데이터 생성이 주목받고 있습니다.

- **Technical Details**: 챌린지의 주요 구성 요소는 참가자가 개발해야 하는 제로샷 TTS 시스템과 생성된 개인화된 음성을 이용해 PSE 시스템을 훈련하는 것입니다. 참가자들은 10명의 목표 화자 중 랜덤으로 선택된 한 화자의 짧은 음성 신호를 입력으로 하여 새로운 발화를 생성해야 합니다. 제로샷 TTS 시스템의 성능은 생성된 음성이 목표 화자의 특성을 얼마나 잘 반영하는지, 즉 음성 품질과 이해력을 평가하여 PSE 성능에 미치는 영향을 분석합니다. 또한, 참가자들에게는 제공된 기준 모델을 바탕으로 PSE 모델을 개발하도록 요청하고 있습니다.

- **Performance Highlights**: 이번 챌린지는 제로샷 TTS 시스템이 생성한 증강된 음성 샘플의 품질이 하위 작업인 PSE 모델 성능에 미치는 영향을 조사하는 것을 목표로 합니다. 참가자들은 생성된 개인화된 음성을 사용하여 테스트 세트에서 제공되는 잡음이 포함된 발화를 향상시키고, 이를 통해 각 화자에 대한 PSE 성능을 평가해야 합니다. 다양한 변수를 고려하여 PSE의 성능 비교를 위해 기준 PSE 모델을 먼저 사용하는 것을 권장하며, 최종적으로 참가자들은 자신만의 모델 아키텍처 결과를 제출할 수 있습니다.



### A review on development of eco-friendly filters in Nepal for use in cigarettes and masks and Air Pollution Analysis with Machine Learning and SHAP Interpretability (https://arxiv.org/abs/2501.13369)
- **What's New**: 이 연구는 네팔, 특히 카트만두와 같은 도시에서의 공기 오염 문제를 다루고 있습니다. 연구에서는 Random Forest Regressor를 사용하여 공기질 지수(AQI)를 예측하고, SHAP(SHapley Additive exPlanations) 분석으로 모델의 예측 결과를 해석합니다. CatBoost 모델이 가장 낮은 Testing RMSE(0.23)와 완벽한 R2 스코어(1.00)를 기록하여 다른 모델보다 높은 정확도를 보여주었습니다.

- **Technical Details**: SHAP 분석 결과, NowCast Concentration 및 Raw Concentration이 AQI 값을 결정하는 주요 요소로 확인되었습니다. 이러한 변수들은 AQI를 크게 증가시키는 중요한 공기 오염 기여자로서의 의미를 지닙니다. 또한, 이 연구는 수소 알파(Hydrogen-Alpha, HA) 생분해성 필터를 공기 질 개선의 새로운 방법으로 평가하고 있습니다.

- **Performance Highlights**: HA 필터는 PM2.5에 대해 98% 이상의 제거 효율을, PM10에 대해 99.24%의 제거 효율을 보여주며, 위험한 공기 입자에 대한 뛰어난 방어력을 제공합니다. 이 필터는 환경 문제를 해결하기 위한 생분해성 마스크와 담배 필터로 제작되었으며, 전통적인 필터의 비생분해성 쓰레기 문제를 줄이면서 대기 오염 물질에 대한 노출을 낮추는 데 기여하고 있습니다.



### Enhanced Extractor-Selector Framework and Symmetrization Weighted Binary Cross-Entropy for Edge Detections (https://arxiv.org/abs/2501.13365)
Comments:
          9 pages

- **What's New**: 이번 연구에서는 가장 최신의 Edge Detection (ED) 기술 중 하나인 Extractor-Selector (E-S) 프레임워크를 개선하여 보다 효율적인 Edge Detection 성능을 달성했습니다. 기존 E-S 프레임워크의 한계를 극복하기 위해, 새로운 아키텍처를 제안하며 richer하고 less-compressed한 feature representations를 사용하고 auxiliary features를 포함시킵니다. 또한, Symmetrization Weight Binary Cross-Entropy (SWBCE)라는 novel loss function을 도입하여 edge 픽셀의 recall과 오류 예측을 동시에 강조하여 더 나은 예측 정확성을 확보했습니다.

- **Technical Details**: 이 연구는 기존 E-S 아키텍처를 발전시키고, feature extraction 능력을 극대화하기 위해 richer하고 less-compressed된 중간 feature들을 활용하는 방안을 제시합니다. 구체적으로, feature extractor를 수정하여 더 세밀한 중간 feature를 생성하고, 선택자가 더 많은 선택 옵션을 가질 수 있도록 보조 feature maps를 추가합니다. 또한, SWBCE loss function은 edge 픽셀 recall과 비-edge 픽셀의 오류 억제를 동시에 강조하여, 손실 함수를 통해 보다 나은 측정 결과를 도출합니다.

- **Performance Highlights**: 실험 결과, 개선된 E-S 아키텍처와 SWBCE 손실 함수를 적용한 모델은 ODS, OIS, AP 지표에서 각각 8.25%, 8.01%, 33.25%의 평균 향상을 달성하며, 기존 표준 E-S 방법을 크게 초과하는 성능을 보였습니다. 이로 인해 본 연구는 ED 작업에 새로운 기준을 제시하고, 기존 방법들의 한계를 극복하는 가능성을 제시합니다. 종합적으로, 우리의 접근 방식은 정량적 정확성과 인지적 품질 두 면에서 향상된 결과를 도출하며, ED 연구에 중요한 기여를 하고 있습니다.



### One Fits All: General Mobility Trajectory Modeling via Masked Conditional Diffusion (https://arxiv.org/abs/2501.13347)
- **What's New**: 이 논문은 여러 가지 궤적 관련 작업을 처리할 수 있는 일반적인 궤적 모델링 프레임워크인 GenMove를 제안합니다. 기존의 궤적 데이터 연구는 각 작업에 특화된 모델을 사용하여 제한된 유연성으로 인해 적용 가능성이 낮았습니다. GenMove는 masked conditional diffusion 기술을 활용하여 다양한 포맷을 통일하고, 복잡한 조건에 적응할 수 있도록 설계되었습니다.

- **Technical Details**: GenMove는 궤적 데이터의 다양한 형식을 통합하기 위해 마스크 조건 모듈을 설계하였습니다. 이 프레임워크는 과거 궤적 데이터를 기반으로 맥락적 궤적 임베딩(contextual trajectory embeddings)을 생성하고, 이 정보를 분산 모델에 통합하여 결과의 유연성을 높입니다. 특히 classifier-free guidance 방식을 사용하여 다양한 조건에 대한 출력을 조정할 수 있도록 하는 점이 특징입니다.

- **Performance Highlights**: 실험 결과, GenMove는 6개의 궤적 관련 작업에서 최첨단 모델 대비 13% 이상의 성능 향상을 달성했습니다. 이 성과는 다양한 응용 프로그램 시나리오에 유연하게 적응할 수 있는 일반 모델의 잠재력을 입증합니다. GenMove는 궤적 생성, 복구 및 예측과 같은 주류 작업에서 뛰어난 성능을 보였습니다.



### Full-Stack Optimized Large Language Models for Lifelong Sequential Behavior Comprehension in Recommendation (https://arxiv.org/abs/2501.13344)
Comments:
          Under Review

- **What's New**: 본 논문에서는 추천 시스템에서 대규모 언어 모델(LLMs)의 장기 순차적 행동 이해 문제를 다룹니다. LLM은 긴 사용자 행동 시퀀스에서 유용한 정보를 효과적으로 추출하는 데 어려움을 겪고 있으며, 이를 해결하기 위한 새로운 프레임워크인 ReLLaX를 제안합니다. ReLLaX는 데이터, 프롬프트, 파라미터의 세 가지 레벨에서 최적화를 제공합니다.

- **Technical Details**: 먼저, 데이터 관점에서 우리는 Semantic User Behavior Retrieval (SUBR)을 도입해 시퀀스의 이질성을 줄이고 LLM이 핵심 정보를 추출하는 데 필요한 특정 행동을 선택합니다. 프롬프트 레벨에서는 Soft Prompt Augmentation (SPA)을 사용해 추천 작업과 연계된 지식을 주입하여 아이템 표현을Align (정렬)합니다. 마지막으로, 파라미터 레벨에서는 CFLoRA를 제안하여 LoRA의 표현력을 강화하고, 구성 요소 간 상호 작용을 통해 더 나은 시퀀스 정보 캡처를 가능하게 합니다.

- **Performance Highlights**: 세 가지 공개 데이터셋에 대한 실험을 통해 ReLLaX가 기존 기준선보다 우수한 성능을 보이며, 장기 순차적 행동 이해 문제를 효과적으로 완화함을 입증했습니다. 또한 CFLoRA 모듈은 기존의 LoRA 기반 LLM4Rec 방법과 비교할 때 더 강력한 성능을 보여줍니다. 이 논문은 LLM 기반 추천 시스템에서의 문제를 처음으로 정의하고, ReLLaX 프레임워크를 통해 해결책을 제시한 중요한 기여를 확인합니다.



### AgentRec: Agent Recommendation Using Sentence Embeddings Aligned to Human Feedback (https://arxiv.org/abs/2501.13333)
Comments:
          10 pages, 8 figures, preprint

- **What's New**: 이번 논문에서는 다중 에이전트 시스템에서 자연어 프롬프트에 대해 가장 적합한 LLM (Large Language Model) 에이전트를 추천하는 새로운 아키텍처를 제안합니다. 기존의 Sentence-BERT (SBERT) 인코더 모델을 확장하여 개발된 이 시스템은 에이전트 분류의 정확도를 높이는데 기여합니다. 특히, 자연어 프롬프트를 문장 임베딩으로 인코딩하여 에이전트 추천과 관련된 의미론적 내용을 포착합니다.

- **Technical Details**: 모델은 문장 임베딩 간의 코사인 유사도를 측정하여 자연어 프롬프트를 분류합니다. 각 분류 작업은 300밀리초 미만의 시간 내에 완료되며, 테스트 데이터에서 92.2%의 상위 1 정확도가 달성되었습니다. 또한, 강화학습을 통해 인간의 가치에 맞춰 조정할 수 있으며, 새로운 클래스에 적응할 수 있는 등 컴퓨팅 비용이 저렴하고 해석 가능성이 높은 구조를 지니고 있습니다.

- **Performance Highlights**: 이 연구는 에이전트 추천을 위한 합성 데이터셋을 생성하여 가능해졌으며, 생성된 데이터셋 및 AgentRec 추천 시스템의 코드는 공개되어 있습니다. 따라서 연구 결과는 실제 응용에 쉽게 활용될 수 있으며, 다중 에이전트 시스템의 효율성을 크게 향상할 것으로 기대됩니다.



### Sparse identification of nonlinear dynamics and Koopman operators with Shallow Recurrent Decoder Networks (https://arxiv.org/abs/2501.13329)
- **What's New**: 이 논문에서는 SHallow REcurrent Decoder 네트워크(SHRED)의 희소 비선형 동역학 식별(SINDy-SHRED) 방법을 제안합니다. 이 방법은 센서 측정을 통해 간단하게 구현할 수 있으며, 효율적인 계산과 강력한 성능을 발휘합니다. SINDy-SHRED는 Gated Recurrent Units (GRUs)를 사용하여 시간적인 센서 측정값의 시퀀스를 모델링하고, 잠재 상태 공간에서 전체 시공간 필드를 재구성합니다.

- **Technical Details**: SINDy-SHRED는 재귀 신경망의 잠재 공간을 활용하고 SINDy 기반의 함수 클래스를 통해 해석 가능성을 강화합니다. 해당 알고리즘은 비선형 모델을 선형 모델로 제한하여 Koopman-SHRED 아키텍처를 생성합니다. 이로 인해, SINDy-SHRED는 적은 수의 센서로도 효과적인 회복을 수행하고, 대규모 과학 데이터 모델링 및 실시간 제어에 적합합니다.

- **Performance Highlights**: SINDy-SHRED는 낮은 데이터 요구량과 빠른 훈련 속도를 자랑하며, 다양한 실제 사례에 대해 우수한 성능을 보여줍니다. 실험 결과에 따르면, SINDy-SHRED는 Convolutional LSTM, PredRNN, ResNet, SimVP 등 다양한 기존 방법들을 초월하는 성능을 달성했습니다. 이와 같은 성능은 장기적인 예측에서도 안정적인 결과를 제공합니다.



### Investigation of the Privacy Concerns in AI Systems for Young Digital Citizens: A Comparative Stakeholder Analysis (https://arxiv.org/abs/2501.13321)
Comments:
          To appear in the 2025 IEEE 14th Annual Computing and Communication Workshop and Conference (CCWC) proceedings

- **What's New**: 이 연구는 젊은 디지털 시민들이 사용하는 기술에 통합된 인공지능(AI) 시스템의 프라이버시(privacy) 문제를 조사합니다. 총 252명의 참가자를 대상으로 하였으며, 부모 및 교육자 110명과 AI 전문가 100명의 유효한 응답을 분석하였습니다. 다양한 이해관계자의 관점을 비교하는 방식으로 연구가 진행되었습니다.

- **Technical Details**: 연구는 데이터 소유권(Data Ownership), 부모의 데이터 공유(Parental Data Sharing), 인식된 위험과 이점(Perceived Risks and Benefits), 투명성(Transparency) 및 신뢰(Trust), 교육 및 인식(Education and Awareness)이라는 다섯 가지 유효한 구성요소에 대해 설명 통계(descriptive statistics)와 부분 최소 제곱 구조 방정식 모델링(Partial Least Squares Structural Equation Modeling) 기법을 활용했습니다. 분석 결과, 교육 및 인식은 데이터 소유 및 위험 평가에 중요한 영향을 미쳤으며, 데이터 소유권은 투명성과 신뢰에 강력한 영향을 미쳤습니다.

- **Performance Highlights**: 투명성과 신뢰는 인식된 위험과 이점과 함께 부모의 데이터 공유에 최소한의 영향을 미쳐 다른 요인들이 더 큰 역할을 할 수 있음을 시사합니다. 이는 사용자 중심의 프라이버시 제어 시스템 필요성을 강조하며, 맞춤형 투명성 전략과 목표 지향적 교육 이니셔티브의 중요성을 제기합니다. 다양한 이해관계자의 관점을 포함함으로써 윤리적 AI 설계와 거버넌스에 대한 실행 가능한 통찰력을 제공합니다.



### Toward Ethical AI: A Qualitative Analysis of Stakeholder Perspectives (https://arxiv.org/abs/2501.13320)
Comments:
          To appear in the 2025 IEEE 14th Annual Computing and Communication Workshop and Conference (CCWC) proceedings

- **What's New**: 이 연구는 인공지능(AI) 시스템의 개인 정보 보호와 윤리적 책임에 대한 이해를 증진시키기 위해 교육자, 부모, AI 전문가 등 다양한 이해 관계자의 관점을 탐구합니다. 227명의 참가자로부터 수집된 설문 응답을 질적 분석하여, 데이터 유출, 윤리적 남용, 과도한 데이터 수집 등의 주요 개인 정보 위험과 맞춤형 서비스 및 교육 발전과 같은 인식된 이점을 식별하였습니다.

- **Technical Details**: 연구는 개인 정보 보호 문제를 효과적으로 해결하기 위해 투명성(Transparency), 설계 단계에서의 개인 정보 보호(Privacy-by-design), 사용자 권한 부여(User Empowerment), 윤리적 감독(Ethical Oversight)의 필요성을 강조합니다. 이는 AI 시스템 개발에 있어 개인 정보 보호 중심의 원칙을 통합하는 것을 포함하며, 이를 통해 다양한 이해 관계자의 요구를 충족하는 방향으로 나아갑니다.

- **Performance Highlights**: 연구 결과는 AI의 이점과 강력한 개인 정보 보호를 균형 있게 조화시킬 수 있는 실행 가능한 통찰을 제공합니다. 여기에는 선택적 데이터 사용(Selective Data Use)의 구현, 투명성 증진(Promoting Transparency), 사용자 자율성(User Autonomy)을 촉진하고, AI 개발에 윤리적 원칙을 통합하는 것이 포함됩니다. 이 연구는 혁신적이면서도 윤리적으로 건전한 AI 기술을 개발하는 것의 중요성을 강조하고, 사용자 간의 신뢰를 구축하는 데 기여하고자 합니다.



### Watching the AI Watchdogs: A Fairness and Robustness Analysis of AI Safety Moderation Classifiers (https://arxiv.org/abs/2501.13302)
Comments:
          Accepted to NAACL 2025 Main Conference

- **What's New**: 본 논문에서는 AI Safety Moderation (ASM) 분류기의 공정성과 강건성을 평가합니다. 특히 OpenAI Moderation API, Perspective API, Google Cloud Natural Language (GCNL) API, 그리고 Clarifai API를 대상으로 연구를 진행하며, 이 모델들이 소수 집단의 콘텐츠를 불공정하게 분류할 위험성을 분석합니다. 이러한 분석은 ASM 모델의 향후 개선을 위한 중요성을 강조합니다.

- **Technical Details**: 연구에서는 ASM 모델의 공정성을 평가하기 위해 Demographic Parity (DP)와 Conditional Statistical Parity (CSP) 같은 메트릭을 사용합니다. 또한, 자연스러운 입력 변형에 대한 분류기의 민감도를 테스트하여 모델의 강건성을 평가합니다. 텍스트 입력을 최소한으로 perturbation하여 발생하는 분류 오류를 측정하는 방법론도 채택합니다.

- **Performance Highlights**: 연구 결과, OpenAI의 ASM 모델이 다른 모델들에 비해 더 불공정하다는 것이 밝혀졌습니다. 또한, 모델들이 입력의 최소한의 변형에 대해 강건하지 않다는 점도 확인되었습니다. 이는 안전하지 않은 댓글이 ASM 모델을 우회하여 통과할 수 있음을 시사하며, 향후 모델 개선을 위한 통찰을 제공합니다.



### RAMQA: A Unified Framework for Retrieval-Augmented Multi-Modal Question Answering (https://arxiv.org/abs/2501.13297)
Comments:
          Accepted by NAACL 2025 Findings

- **What's New**: 이 논문에서는 텍스트와 이미지를 통합한 Multi-modal retrieval-augmented Question Answering (MRAQA) 분야에서 새로운 접근법인 RAMQA를 제안합니다. RAMQA는 전통적인 learning-to-rank 방법과 generative ranking 기술을 결합하여, 현대의 대형 생성 언어 모델(LLMs)을 활용한 정보 검색의 한계를 극복하고자 합니다. 이를 통해 두 가지 MRAQA 벤치마크인 WebQA와 MultiModalQA에서 성능 향상을 입증하였습니다.

- **Technical Details**: RAMQA는 LLaVA를 기반으로 하여 multi-modal pointwise ranker를 훈련한 후, novel autoregressive multi-task learning 접근법을 채택하여 LLaMA 모델을 상위 k개 문서의 재정렬에 사용합니다. 이 과정에서는 zero-shot LLaVA 모델을 이용하여 다중 모달 문서를 텍스트 표현으로 통합하고, permutation 기법을 활용하여 문서 후보군의 다양성을 증가시켜 bias를 감소시키는 방법을 사용합니다.

- **Performance Highlights**: 실험 결과, WebQA와 MultimodalQA 두 벤치마크에서 강력한 기준선에 비해 유의미한 성능 향상을 달성하였으며, RAMQA는 웹 기반의 QA 시스템에서 네 번째 순위를 기록했습니다. 이 연구는 multi-modal generative LLMs의 활용 가능성을 보여주며, 점진적인 재정렬이 정보 검색에서 더 효율적인 처리를 가능하게 함을 시사합니다.



### Toyteller: AI-powered Visual Storytelling Through Toy-Playing with Character Symbols (https://arxiv.org/abs/2501.13284)
Comments:
          Accepted to CHI2025

- **What's New**: Toyteller는 사용자가 캐릭터 심볼을 직접 조작하여 텍스트와 비주얼이 혼합된 이야기를 생성할 수 있는 AI 기반의 스토리텔링 시스템입니다. 이를 통해 사용자는 이야기 생성 및 표현을 자연어 외의 다양한 방식으로 수행할 수 있는 새로운 기회를 얻게 됩니다. Toyteller는 캐릭터의 움직임을 입력 모달리티와 출력 형식으로 활용하여, 사용자와 AI 간의 협력적 스토리텔링을 용이하게 합니다.

- **Technical Details**: Toyteller는 캐릭터의 심볼 움직임과 텍스트 생성을 연결하는 공통 의미 공간을 구축하여 모션과 텍스트 간의 상호작용을 가능하게 합니다. 이 시스템은 텍스트와 모션의 생성이 상호 연결된 대화형 AI 모델을 통해 이루어집니다. 기술 평가 결과, Toyteller는 기존의 경쟁 모델인 GPT-4o보다 뛰어난 성능을 보였으며, 여러 측면에서 사용자 경험을 향상시켰습니다.

- **Performance Highlights**: Toyteller는 텍스트와 모션 생성에서 모두 빠른 반응 속도를 보여주며, 최대 7.9배 텍스트 생성 속도 증가와 557.4배 모션 생성 속도 증가를 기록했습니다. 사용자 연구에서는 장난감 놀이 방식의 상호작용이 사용자의 의도를 표현하는 데 유용하지만, 특정한 의도를 명확하게 전달하는 데는 한계가 있음을 발견했습니다. 결과적으로 사용자는 자연어 프롬프트와 장난감 조작을 병행하여 상호 보완적으로 활용하며, Toyteller의 유연성이 다양한 사용자 요구를 지원할 수 있음을 보여주었습니다.



### Experience with GitHub Copilot for Developer Productivity at Zoominfo (https://arxiv.org/abs/2501.13282)
Comments:
          25 pages, 11 figures

- **What's New**: 이번 논문은 GitHub Copilot의 배포 및 Zoominfo의 개발자 생산성에 미치는 영향을 종합적으로 평가했습니다. 400명 이상의 개발자가 참여하는 체계적인 4단계 접근 방식을 통해 Copilot을 평가하고 배포한 사례를 제시합니다. 초기 연구에서의 긍정적인 결과에도 불구하고, 중대형 기업 환경에서의 실증적 증거가 부족하다는 점을 강조합니다. 이를 통해 AI 기반 코딩 도구의 실용성을 밝혔다고 볼 수 있습니다.

- **Technical Details**: Zoominfo는 GitHub Copilot에 대한 평가에서 정량적 메트릭(quantitative metrics)과 정성적 피드백(qualitative feedback)을 결합했습니다. Copilot의 제안 수용률(acceptance rate)은 평균 33%이며, 코드 라인의 수용률은 20%로 나타났습니다. 또한, 상위 4개 프로그래밍 언어인 TypeScript, Java, Python, JavaScript의 수용률은 약 30%로 유지되고, HTML, CSS, JSON, SQL의 경우 더 낮은 수치를 보였습니다.

- **Performance Highlights**: 개발자들은 GitHub Copilot 사용으로 인해 약 20%의 시간 절약을 경험했으며, 개발자 만족도는 72%로 높았습니다. 그러나 도메인 특정 로직의 부족과 코드 품질의 일관성이 떨어진 점은 한계로 지적되었습니다. 이러한 시간 절약과 만족도 향상에도 불구하고, 개발 코드 공헌량은 수십만 줄에 달하는 성과를 보여, 생산성에 긍정적인 영향을 미쳤습니다.



### Let SSMs be ConvNets: State-space Modeling with Optimal Tensor Contractions (https://arxiv.org/abs/2501.13230)
Comments:
          25 pages, 7 figures

- **What's New**: Centaurus는 텐서 수축(tensor contractions)을 통해 훈련할 수 있는 일반화된 상태 공간 모델(state-space model, SSM) 블록으로 구성된 네트워크 클래스입니다. 기존의 SSM 블록 구조에서 더 큰 유연성을 제공하여, 심층 네트워크의 설계가 가능해졌습니다. Centaurus는 고전적인 합성곱(convolutional) 블록에서 영감을 받아 다양한 구조적 조합을 도입하여, 훈련과 추론에서의 메모리 및 계산 효율성을 극대화합니다. 이 네트워크는 특정 작업에서 동종의 구조보다 우수한 성능을 발휘하는 것으로 나타났습니다.

- **Technical Details**: Centaurus 네트워크는 다양한 유형의 SSM 블록에 대한 텐서 네트워크(tensor networks) 구조를 사용하여 새로운 구조를 설계합니다. 이 블록들은 기존의 심층 SSM 네트워크와 비교해 더 많은 연결 유연성을 제공하며, 새로운 SSM 블록들이 텐서 네트워크 형식으로 구현되어 있습니다. 각 SSM 블록의 수축 순서가 입력 특징 및 시스템 행렬의 형태에 따라 동적으로 최적화되어, 모든 SSM 블록에서 훈련 속도가 대폭 향상됩니다. 이러한 접근 방식은 간섭 없이 순수하게 SSM 기반의 성능을 달성할 수 있도록 도와줍니다.

- **Performance Highlights**: Centaurus는 키워드 탐지(keyword spotting), 음성 잡음 제거(speech denoising), 자동 음성 인식(automatic speech recognition, ASR) 등 여러 오디오 처리 작업에서 우수한 성능을 보입니다. 특히 ASR 성능에서 경쟁력 있는 결과를 보여주며, LSTM이나 CNN, 주의(attention) 메커니즘 없이도 효율적인 훈련과 실행이 가능합니다. 이러한 특징 덕분에 Centaurus는 음성 처리 및 관련 작업에서 매우 효율적인 선택으로 자리잡게 될 것입니다.



### SRMT: Shared Memory for Multi-agent Lifelong Pathfinding (https://arxiv.org/abs/2501.13200)
Comments:
          16 pages, 11 figures

- **What's New**: 이번 연구에서는 다중 에이전트 시스템에서의 행동 조정을 개선하기 위한 새로운 접근 방식을 제안합니다. 이를 위해 에이전트들이 협력할 수 있는 글로벌 작업 공간으로서 공유 메모리 개념을 도입한 Shared Recurrent Memory Transformer (SRMT)을 제안합니다. 이 방법론은 에이전트들이 정보를 간접적으로 교환하고 행동을 조정할 수 있도록 메모리 변환기를 다중 에이전트 환경으로 확장합니다. SRMT는 Bottleneck 탐색 작업과 POGEMA 벤치마크 세트에서 평가되어 다른 기존 방법들에 비해 뛰어난 성능을 보였습니다.

- **Technical Details**: SRMT는 에이전트들이 개별 작업 메모리를 풀링하고 이를 전 세계적으로 방송함으로써 각각의 메모리를 효과적으로 활용하도록 설계되었습니다. 이 아키텍처는 에이전트들이 협업할 수 있는 능력을 향상시키며, 이는 특히 자원이 희박한 상황에서 서로의 상태를 관찰하고 행동 결정을 내릴 수 있게 도와줍니다. 연구에서는 SRMT가 Partially Observable Multi-Agent Pathfinding (PO-MAPF) 문제에서 작동하여 에이전트들이 자신들의 목표를 달성하는 동시에, 환경의 상태를 지역적으로만 관찰하는 방식으로 진행되었습니다.

- **Performance Highlights**: Bottleneck 작업에서 SRMT는 다양한 강화 학습 기초 벤치마크보다 일관되게 우수한 성능을 보였으며, 훈련 중에 관찰된 것보다 긴 복도에서도 효과적으로 일반화됩니다. POGEMA 지도를 포함한 탐색에서도 최근 MARL 및 하이브리드 알고리즘과 경쟁력 있는 성능을 나타내어, 다중 에이전트 시스템에서의 협력을 향상시킬 수 있는 가능성을 제시합니다. 이러한 결과는 공유 재발 메모리를 Transformer 기반 아키텍처에 통합함으로써 분산형 다중 에이전트 시스템의 조정을 강화할 수 있음을 시사합니다.



### Learning in Log-Domain: Subthreshold Analog AI Accelerator Based on Stochastic Gradient Descen (https://arxiv.org/abs/2501.13181)
- **What's New**: 이번 논문에서는 stochastic gradient descent with L2 regularization (SGDr)를 위한 새로운 아날로그 가속기 아키텍처를 제안합니다. 이 아키텍처는 서브스레숄드 MOS 회로를 활용하여 저전력으로 동작하며, 기존의 디지털 구현과 비교하여 트랜지스터 면적과 전력 소비를 크게 줄입니다. 또한, 이번 설계는 전통적인 메모리 접근 방식을 사용하지 않고 훈련이 끝난 후에만 메모리 접근을 필요로 하는 등의 혁신적인 특성을 가지고 있습니다.

- **Technical Details**: 제안된 아날로그 가속기는 연속 시간에서 SGDr을 해결하기 위한 수학적 프레임워크를 설정하며, SGDr 학습 방정식을 로그 도메인 회로에 매핑하는 방법을 자세히 설명합니다. 이 아키텍처는 불안정 메모리를 이용하고, 효율적인 훈련이 가능한 CMOS 기술을 기반으로 하여, 기존의 비휘발성 메모리 기반 아날로그 가속기에서 발생하는 문제를 피하고 있습니다. 제안된 설계는 실험적으로 시뮬레이션 되며, AMS 0.35 µm 프로세스를 통해 검증되었습니다.

- **Performance Highlights**: 실험 결과, 이 아키텍처는 평균 제곱 오차가 0.87% 이하이고, 최소 8비트의 정밀도로 이상적인 동작을 근접해 나타냈습니다. 또한, 다양한 하이퍼파라미터를 지원하여 범용성이 높습니다. 이러한 성과는 저전력 아날로그 AI 하드웨어의 전기적 효율성을 입증하며, 칩 상에서의 훈련 능력을 통해 에지 디바이스에 적합한 솔루션을 제공합니다.



### QuFeX: Quantum feature extraction module for hybrid quantum-classical deep neural networks (https://arxiv.org/abs/2501.13165)
Comments:
          12 pages, 10 figures

- **What's New**: 본 논문에서는 Quantum Feature Extraction (QuFeX)이라는 새로운 양자 기계 학습 모듈을 소개합니다. QuFeX는 차원 축소된 공간에서 특성 추출을 가능하게 하여 일반적인 양자 컨볼루션 신경망 아키텍처에서 요구되는 병렬 평가 수를 크게 줄입니다. 또한 QuFeX는 심층 클래식 신경망에 원활하게 통합될 수 있도록 설계되어 하이브리드 양자-클래식 모델에 특히 적합합니다.

- **Technical Details**: QuFeX는 QCNN(Quantum Convolutional Neural Network)과 QuanNN(Quanvolutional Neural Network)의 기술을 통합하여 설계되었습니다. 이 모듈은 데이터 분석 구조를 결합하여, 입력 데이터의 국소적인 변환에 최적화된 데이터 파이프라인을 제공하고, 축소된 차원 표현에서 데이터를 처리합니다. 이러한 설계는 QuanNN보다 병렬 양자 회로 평가의 수를 줄이는 장점을 가지고 있습니다.

- **Performance Highlights**: QuFeX를 이용한 Qu-Net은 클래스컬 U-Net 모델의 병목 부분에 통합되어 있으며, 의료 영상 및 자율주행과 같은 이미지 분할 작업에서 우수한 성능을 보여줍니다. 실험 결과 Qu-Net은 전통적인 U-Net 기준에 비해 뛰어난 분할 성능을 달성하여, 실제 이미지 분할 작업에 양자 강화 심층 신경망을 효과적으로 적용할 수 있는 가능성을 제시합니다.



### AirRadar: Inferring Nationwide Air Quality in China with Deep Neural Networks (https://arxiv.org/abs/2501.13141)
- **What's New**: 본 논문에서는 AirRadar라는 새로운 딥러닝 기반 모델을 소개합니다. 이 모델은 공기질 모니터링 스테이션이 없는 지역에서 실시간 공기질을 정확하게 추정하는 데 도움을 줍니다. AirRadar는 기존 데이터를 사용하여 학습 가능한 마스크 토큰을 이용해 감지되지 않은 공기질 특성을 복원합니다. 이 모델은 중국 전역의 1,085개 모니터링 스테이션에서 수집된 데이터를 통해 그 효용성이 입증되었습니다.

- **Technical Details**: AirRadar는 두 개의 주요 단계로 작동하며, 첫 번째로 공간 상관관계를 포착하고 두 번째로 분포 변화를 조정합니다. Spatial Learning Module과 Casual Learning Module을 통해 복잡한 공간 관계와 지역별 이질성을 효과적으로 처리합니다. 지역 및 글로벌 Learner를 활용하여 공기질 인퍼런스를 수행하고, 데이터 사용의 효율성을 극대화합니다. 이는 masking 기법을 통해 이루어지며, 백도어 조정(backdoor adjustment)을 적용하여 공간 이질성을 해결합니다.

- **Performance Highlights**: AirRadar는 다양한 마스킹 비율 하에서 수천 개의 위치에 대한 공기질 데이터를 평가하여, 최첨단 방법인 STFNN과 비교하여 28.0%에서 44.8%의 정확도 향상을 보였습니다. 또한, 실용성을 검증하기 위해 웹 기반 플랫폼이 배포되어, 사용자가 결과를 쉽게 확인할 수 있도록 하였습니다. 이 연구는 대규모 공기질 모니터링의 미래 방향성을 제시하며, 사회적으로 중요한 공공보건 문제에 기여할 수 있는 가능성을 보여줍니다.



### Forecasting of Bitcoin Prices Using Hashrate Features: Wavelet and Deep Stacking Approach (https://arxiv.org/abs/2501.13136)
Comments:
          arXiv admin note: text overlap with arXiv:2402.05943 by other authors

- **What's New**: 디지털 통화, 특히 비트코인(BTC)의 가격 변동성을 예측하기 위한 새로운 분류 및 회귀 모델이 제안되었습니다. 본 연구는 웨이브렛을 이용하여 잡음을 제거하고 스택 딥 러닝 기법을 기반으로 하여 다양한 시간 간격에서 BTC 가격을 예측합니다. 이러한 모델은 딥 러닝을 기반으로 하여 1일, 7일, 30일, 90일의 가격 예측을 수행합니다.

- **Technical Details**: 제안된 모델은 딥러닝 기반 트랜스포머와 신경망(neural networks) 모델을 사용하여 다양한 가격 예측을 수행합니다. 데이터 전처리 단계에서 Chi2, RFE, Embedded 3가지 특성 선택 모델을 적용하였습니다. 마지막으로 90일까지의 예측을 위한 중간-term 가격 및 고저 예측 메커니즘을 개발하였습니다.

- **Performance Highlights**: 모델의 정확도는 다음 날 예측에 대해 63%였으며, 7일, 30일, 90일 예측에 대해서는 각각 64%, 67%, 82%의 정확도를 달성했습니다. 일일 가격 예측의 경우, 오류율은 0.58로 감소하였고, 7일에서 90일 동안의 오류는 2.72%에서 2.85%로 나타났습니다. 이러한 결과는 제안된 모델이 기존 문헌의 다른 모델들보다 더 우수한 성능을 보임을 시사합니다.



### Applications and Challenges of AI and Microscopy in Life Science Research: A Review (https://arxiv.org/abs/2501.13135)
- **What's New**: 이 논문은 인공지능(AI)과 현미경 기술의 융합을 탐구하며, 이러한 기술들이 생명과학 분야에서 직면하는 여러 가지 도전과제를 해결하는 데 어떻게 도움이 될 수 있는지를 강조합니다. 생물학적 데이터의 방대한 양과 복잡성을 고려할 때, AI의 적용은 생명과학의 연구 진행을 가속화할 수 있는 필수적 요소로 떠오르고 있습니다. 특히, 다양한 생물학적 시스템에서 AI가 제공할 수 있는 이점과 독특한 데이터 유형 및 레이블 요구 사항에 대한 세부적인 리뷰를 제시합니다.

- **Technical Details**: 이 논문에서는 생물학적 시스템 연구에 있어 AI와 현미경 기술의 상호작용을 조망합니다. 현미경의 데이터는 3D, 4D, 5D와 같은 다차원 형식을 취하며, 이러한 대규모 데이터 해석을 위한 AI의 필요성이 커지고 있습니다. 다양한 알고리즘이 생물학적 데이터의 복잡한 관계를 설명하는 데 사용되며, 예를 들어 반복 신경망(RNN)과 변환기(transformers) 등이 포함됩니다.

- **Performance Highlights**: 최신 AI 모델은 단백질 및 핵산의 서열 예측과 같은 생명과학 연구에서 큰 성과를 거두었습니다. 그러나 현미경 기반 진단에서의 AI 통합은 아직 미비한 상황이며, 이는 혁신적인 AI 기반 솔루션의 필요성이 증가함을 의미합니다. 이 논문은 AI와 현미경이 어떻게 협력하여 생명과학 연구의 주요 과제를 해결할 수 있는지를 탐색하며, 열려있는 연구 질문과 잠재적인 전략도 제시합니다.



### Graph Representation Learning with Diffusion Generative Models (https://arxiv.org/abs/2501.13133)
- **What's New**: 이번 연구는 그래프 구조 데이터에 대한 디퓨전 모델의 활용을 탐색합니다. 디퓨전 모델은 이미지 및 비디오와 같은 복잡한 데이터 분포를 정확히 모델링할 수 있는 생성 모델로 알려져 있습니다. 이 연구에서는 자동 인코더(autorecoder) 프레임워크 내에서 이산 디퓨전 모델을 훈련시켜 그래프 데이터에 적합한 의미 있는 표현을 학습하도록 합니다.

- **Technical Details**: 딥러닝에서는 표현 학습이 매우 중요한 분야로, 원시 데이터를 압축된 저차원 임베딩으로 변환하는 것이 목표입니다. 디퓨전 자동 인코더에서는 인코더가 유용한 표현을 학습하고, 디퓨전 디코더가 그 표현을 바탕으로 데이터를 생성합니다. 이산 디퓨전 모델은 그래프 데이터에 적합하도록 설계되어, 격리된 피처를 효과적으로 처리하며, 구조적 패턴과 관계를 포착합니다.

- **Performance Highlights**: Discrete Diffusion Autoencoder(DDAE) 모델은 그래프 분류 작업에서 뛰어난 성능을 보입니다. 이러한 모델은 그래프 자료의 이산적 특성을 잠재 임베딩으로 변환함으로써, 더 의미 있는 저차원 표현을 학습할 수 있습니다. 또한, 이 방법은 레이블된 데이터가 부족한 경우에도 비지도 학습을 통해 그래프 표현을 생성할 수 있는 가능성을 보여줍니다.



### A Hierarchical Reinforcement Learning Framework for Multi-UAV Combat Using Leader-Follower Strategy (https://arxiv.org/abs/2501.13132)
- **What's New**: 이 논문은 다수의 자율 UAV(무인 항공기)가 협력하여 복잡한 공중 전투를 수행하는 데 있어 새로운 접근 방식을 제안합니다. 특히, 기존의 방법들이 정해진 행동 공간으로 제한된 것을 개선하기 위해 'Leader-Follower Multi-Agent Proximal Policy Optimization (LFMAPPO)'라는 계층적 프레임워크를 도입합니다. 이 프레임워크는 고차원 환경에서 효과적인 협력을 도모하고자 하며, 다양한 UAV의 동적 상황을 고려합니다.

- **Technical Details**: 제안된 프레임워크는 세 가지 수준으로 구성됩니다. 최상위 레벨은 전투 상황에 대한 거시적 분석을 수행하고 실행 정책을 안내하며, 중간 레벨은 원하는 액션의 각도를 결정하고, 하위 레벨은 이러한 각도를 기반으로 정확한 액션 명령을 생성합니다. 이를 통해 서로 다른 역할을 지닌 리더와 팔로워 전략을 구현하여 상태 가치 함수를 최적화하고 협력을 증진시킵니다.

- **Performance Highlights**: 시뮬레이션 실험을 통해 제안된 방법의 효과성이 검증되었습니다. 기존의 전통적인 방법들과 비교 시, 더 높은 보상 값과 나은 궤적을 달성하였으며, 정밀하고 효율적인 목표 선택이 가능함을 보여주었습니다. 이러한 결과는 UAV 간의 협력 능력을 강화하고 공중 전투에서의 성능을 향상시키는 데 기여할 것으로 기대됩니다.



### Preference Curriculum: LLMs Should Always Be Pretrained on Their Preferred Data (https://arxiv.org/abs/2501.13126)
Comments:
          18 pages, 13 figures

- **What's New**: 최근의 대형 언어 모델(LLM)은 일관된 데이터 분포를 사용하여 사전 훈련을 진행해왔지만, 모델의 성능이 향상됨에 따라 차별화된 데이터로 훈련하는 것이 직관적입니다. 본 논문에서는 Perplexity Difference 기반의 선호 커리큘럼 학습(Preference Curriculum learning, PDPC) 프레임워크를 통해 LLM이 선호하는 데이터를 인식하고 이를 활용하여 훈련하는 방법을 제안합니다. PDPC는 LLM의 발전하는 데이터 선호도를 반영하여 사전 훈련 과정에서의 효율성을 극대화하는 데 중점을 두고 있습니다.

- **Technical Details**: PDPC 프레임워크에서는 먼저 PD(Perplexity Difference) 메트릭을 도입하여 강한 모델과 약한 모델이 샘플에 얼마나 잘 맞는지를 측정합니다. PD 값이 높은 샘플은 약한 모델이 학습하기 어려운 반면 강한 모델에게는 더 적합한 특징을 가집니다. 이 프레임워크는 오프라인에서의 처리 방법을 통해 동적인 선호도 조정을 근사하여 훈련 중단 없이 연속적으로 LLM을 학습시킬 수 있도록 설계되었습니다.

- **Performance Highlights**: 1.3B 및 3B 모델에서의 실험 결과, 기준선에 비해 PDPC가 현저한 성능 향상을 보여주었습니다. 특히, 3B 모델의 경우, 평균 정확도가 다양한 벤치마크에서 4.1% 이상 증가한 것으로 나타났습니다. 이러한 결과는 PDPC가 데이터 선호도를 효율적으로 활용하여 LLM의 학습 성능을 개선하는 데 효과적임을 보여줍니다.



### Generating Plausible Distractors for Multiple-Choice Questions via Student Choice Prediction (https://arxiv.org/abs/2501.13125)
- **What's New**: 이번 연구에서는 교육 분야에서 학생들의 오해(misconceptions)와 지식 격차를 파악하고 이해도를 정확하게 평가하기 위한 MCQ(다지선다형 질문)에서 더 그럴듯한 방해 선택지(distractor)를 생성하는 모델 훈련 파이프라인을 제안합니다. 먼저 학생들의 오해를 고려하여 두 개의 방해 선택지의 상대적 그럴듯함을 평가하는 쌍별(rank-based) 랭커(pairwise ranker)를 훈련합니다. 이후, 이 모델을 활용하여 쌍별 방해 선택지 랭크 데이터셋을 생성하고, Direct Preference Optimization(DPO)을 통해 더 그럴듯한 방해 선택지를 생성하는 방해 선택지 생성기를 훈련합니다.

- **Technical Details**: 모델 훈련을 위해, South Korea의 온라인 학습 플랫폼에서 교육자들이 생성한 MCQ 데이터세트를 사용했습니다. 데이터세트에는 파이썬, 데이터베이스(SQL) 및 머신러닝 및 딥러닝(Machine Learning & Deep Learning) 관련 질문이 포함되어 있으며, 각 질문의 학생 선택률 정보가 포함되어 있습니다. 이를 통해 우리는 학생들이 어떤 방해 선택지를 더 혼란스럽게 선택하는지 평가할 수 있습니다.

- **Performance Highlights**: 실험 결과, 쌍별 랭커는 학생들의 일반적인 오해를 효과적으로 파악하며, 인간 전문가와 유사한 랭킹 정확도를 달성했습니다. 또한 방해 선택지 생성기는 여러 기준선 모델보다 우수한 성과를 보이며, 더 그럴듯한 방해 선택지를 생성하고, 높은 항목 차별지수(item discrimination index, DI)를 산출했습니다.



### Debate Helps Weak-to-Strong Generalization (https://arxiv.org/abs/2501.13124)
Comments:
          AAAI2025 Special Track on AI Alignment (Oral presentation)

- **What's New**: 이번 연구에서는 AI 정렬(alignment) 기술의 한계를 극복하기 위해 인간의 감독(supervision) 발전과 강력한 사전 학습(pretrained) 모델의 활용을 결합하는 방법을 제시합니다. 특히, 약한(weak) 인간 감독을 통해 강력한 모델을 훈련시키고 그 모델이 생성한 레이블을 통해 다시 약한 모델을 훈련시키는 순환적 방식에 초점을 맞추고 있습니다. 이러한 접근 방식은 오픈AI의 약한-강한 NLP 벤치마크에서 정렬 성능을 향상시키는 데 기여합니다.

- **Technical Details**: 연구에서는 약한 모델이 신뢰할 수 없는 강력한 모델로부터 유용한 정보를 추출할 수 있도록 돕는 논쟁(debate) 기법을 도입했습니다. 두 개의 강력한 모델이 서로 대결하는 사례에서 발생하는 논증을 통해 약한 모델이 더 신뢰할 수 있는 정보를 획득하도록 합니다. 또한, 다양한 약한 모델의 집합(ensemble)을 사용하여 논쟁에서 생성된 긴 인수(arguments)를 최대한 활용하고, 단일 모델보다 더 견고한 감독 예측을 도출하였습니다.

- **Performance Highlights**: 결과적으로, 약한 모델의 앙상블이 단일 약한 모델 및 세밀한 조정(finetune) 앙상블보다 일관되게 우수한 성과를 보였습니다. 다양한 샘플링(seed)을 사용하는 논쟁 앙상블의 결과는 더욱 향상된 성능을 나타내어 이는 향후 AI 정렬에 대한 혼합(superhuman) 접근 방법의 연구 가능성을 보여줍니다. 본 연구는 논쟁이 약한-강한 일반화에 도움이 된다는 실증적 증거를 제시하며, AI 정렬 분야에 중대한 기여를 한 것으로 평가됩니다.



### Zero-Shot Verification-guided Chain of Thoughts (https://arxiv.org/abs/2501.13122)
- **What's New**: 이번 연구에서는 LLM(대형 언어 모델)의 COT(Chain-of-Thought) 프롬프트를 사용한 제로샷(self-verification) 자기 검증 접근 방식을 제안합니다. 기존의 방법들은 주로 미세 조정된 검증기나 다양하게 수작업으로 작성된 몇 가지 사례에 의존했으나, 본 연구는 어떠한 수동 선택 예제도 없이 LLM이 생성한 추론 단계를 스스로 검증하는 방법을 중심으로 하였습니다. 이를 위해 COT STEP이라는 새로운 제로샷 프롬프트를 설계하였으며, 이는 추론 단계를 분해하는 데 도움을 줍니다.

- **Technical Details**: 본 연구에서는 SOLAR와 Phi3라는 두 개의 오픈 라이선스 모델을 기반으로 실험을 진행하고, 다양한 제로샷 프롬프트 전략을 평가하였습니다. 네 가지 주요 제로샷 프롬프트 방법은 각각 기본(P1), COT(P2), PS+(P3), TAB COT(P4)으로 정의됩니다. 특히, TAB COT 프롬프트는 표 형식으로 결론을 도출하는 과정을 유도하여 추론 단계를 명확히 식별할 수 있도록 합니다.

- **Performance Highlights**: 이 연구의 결과는 COT STEP 프롬프트가 다른 프롬프트와 경쟁력을 유지하며 자동 단계 분해를 가능하게 한다는 것을 보여줍니다. LLM 기반의 검증기가 수학적 추론 단계를 검증하는 데 상당히 효과적이며, 검증 점수를 사용하여 올바른 답변을 선택하거나 실시간으로 추론을 안내하는 데 이점이 있음을 확인하였습니다. 그러나 자체 일관성을 사용할 경우 이점이 사라지는 경향이 나타났습니다.



### Episodic Memories Generation and Evaluation Benchmark for Large Language Models (https://arxiv.org/abs/2501.13121)
- **What's New**: 이 논문에서는 에피소드 메모리(episodic memory)에 대한 LLMs의 한계를 극복하고, 그러한 메모리 능력을 통합하기 위한 포괄적인 프레임워크를 제안합니다. 기존의 LLM은 사실과 일치하지 않는 정보를 생성하는 할루시네이션(hallucination) 문제를 겪고 있으며, 이로 인해 인간과 유사한 사고를 발전시키기 위한 잠재력이 제한되고 있습니다. 또한, 연구자들은 에피소드 메모리를 충분히 평가하지 않았으며, 논문에서 제안한 새로운 벤치마크가 이러한 격차를 해소할 수 있을 것으로 기대하고 있습니다.

- **Technical Details**: 이 연구는 인지 과학에서 영감을 받아 에피소드 메모리를 구조적으로 모델링하는 접근법을 개발했습니다. 이 프레임워크는 시간적 및 공간적 맥락, 관련 엔티티(entity) 및 사건에 대한 상세한 설명을 포함합니다. 저자들은 LLM 성능 평가를 위한 독창적인 에피소드 메모리 벤치마크를 창출하였으며, 이를 통해 다양한 회상(recall) 및 에피소드 추론(tasks) 작업에서 평가할 수 있도록 오픈 소스 코드와 데이터셋을 공개했습니다.

- **Performance Highlights**: 상위 모델인 GPT-4 및 Claude 변형, Llama 3.1과 o1-mini를 평가한 결과, 가장 진보된 LLM조차 복잡한 공간적-시간적 관계(spatio-temporal relationships)를 다루는 에피소드 메모리 작업에서 어려움을 겪고 있음을 보여줍니다. 특히, 관련된 여러 사건을 처리할 때 이러한 어려움이 더욱 두드러졌으며, 10k-100k 토큰의 짧은 문맥(context)에서도 이 문제가 나타났습니다. 이 연구는 LLM의 성능 개선을 위해 에피소드 메모리 벤치마크의 중요성을 강조합니다.



### Multilinguality in LLM-Designed Reward Functions for Restless Bandits: Effects on Task Performance and Fairness (https://arxiv.org/abs/2501.13120)
Comments:
          Accepted at the AAAI-2025 Deployable AI Workshop

- **What's New**: 이 논문은 Restless Multi-Armed Bandits (RMABs) 알고리즘에 비영어 명령어를 사용할 때의 작업 성능과 공정성에 미치는 영향을 연구합니다. 특히, 저자들은 리소스가 부족한 언어를 포함한 여러 언어로 번역된 다양한 복잡도의 프롬프트를 조사하였습니다. 그 결과 영어 프롬프트가 작업 성능에 있어 유리함을 보이며, 공정성 측면에서도 리소스가 부족한 언어와 복잡한 프롬프트가 불공정성을 일으킬 가능성이 높다는 사실을 밝힙니다.

- **Technical Details**: 실험은 DLM 알고리즘(Behari et al. 2024)을 이용하여, 영어, 힌디어, 타밀어 및 투루어(저자원 언어)로 다양한 프롬프트를 실행하는 방식으로 진행됩니다. 이 연구는 6개의 피쳐를 기반으로 하는 합성 환경을 사용하여, 각 언어로 제안된 보상 함수의 효과를 조사합니다. 또한, Gemini 1.0 Pro를 LLM으로 활용하며, Whittle Index 기반의 해결책을 리인포스먼트 러닝 부분에 적용합니다.

- **Performance Highlights**: 영어로 제안된 LLM 보상 함수가 다른 언어에 비해 훨씬 효과적이라는 결과를 도출하였습니다. 프롬프트의 정확한 표현 방식은 작업 성능에 영향을 미치며, 프롬프트의 복잡성이 증가할수록 모든 언어에서 성능이 저하되는 경향을 보입니다. 그러나 영어 프롬프트에 비해 리소스가 부족한 언어의 경우 성능 저하가 더욱 두드러지는 것으로 확인되었습니다.



### MyGO Multiplex CoT: A Method for Self-Reflection in Large Language Models via Double Chain of Thought Thinking (https://arxiv.org/abs/2501.13117)
- **What's New**: 최근 대형 언어 모델(LLMs)의 발전이 여러 가지 추론 및 의사결정 작업에서 그들의 인상적인 능력을 입증했습니다. 그러나 이러한 모델의 추론 과정의 품질과 일관성은 여전히 개선의 여지가 있으며, 이를 위해 자기 검토 및 자기 반성의 향상이 필요합니다. 본 논문에서는 Double Chain of Thought (CoT) 사고 방식을 통해 LLM이 자기 검토를 시뮬레이션할 수 있도록 하는 방법인 Multiplex CoT를 소개합니다.

- **Technical Details**: Multiplex CoT는 반복적 사고를 활용하는 방법으로, 모델이 초기 Chain of Thought를 생성한 후, 이를 비판하고 수정하는 두 번째 사고 생성 단계를 진행합니다. 이 재귀적 접근 방식은 더 일관되고 논리적이며 견고한 답변을 제공하여 결정 프로세스를 개선합니다. 이 방법은 간단한 prompt engineering을 통해 기존 LLM 구조에서 효과적으로 구현될 수 있으며, 추가 훈련이 필요하지 않습니다.

- **Performance Highlights**: Multiplex CoT는 효과적으로 초기 사고의 일관성을 검토하고 수정하여 더 높은 수준의 논리적 일관성을 성취합니다. 특히, 모델의 두 번째 추론 단계에서는 초기 추론의 결점을 식별하고 이를 교정하여 최종 답변을 보다 정확하게 정제합니다. 이 접근 방식은 오류 수정률을 개선하는 데 중요한 역할을 하여 전체 추론 품질의 향상을 가져옵니다.



### Dagger Behind Smile: Fool LLMs with a Happy Ending Story (https://arxiv.org/abs/2501.13115)
- **What's New**: 본 논문은 기존의 jailbreak 공격 방법론과는 다르게, LLM(대형 언어 모델)이 긍정적인 프롬프트에 더 잘 반응한다는 새로운 관점을 제공합니다. 이를 바탕으로 Happy Ending Attack (HEA)라는 새로운 공격 방법을 제안하며, 악의적인 요청을 긍정적인 시나리오 템플릿 안에 감싸서 LLM을 속이는 방안을 제시합니다. HEA는 단 2단계의 간단한 절차로 LLM을 jailbreak 하는 데 성공적으로 작용하였습니다.

- **Technical Details**: HEA는 부정적인 질문에 대해 LLM이 기본적으로 반응을 거부하는 특성을 활용하여, 긍정적인 내용을 담은 스토리에 악의적인 요청을 포함시키는 방식으로 디자인됩니다. 공격 과정은 프롬프트 디자인에서 시작하여, 마지막 질문을 통해 더 구체적이고 조직화된 jailbreak 응답을 얻습니다. 이 과정은 전혀 인간의 개입 없이 완전 자동화됩니다.

- **Performance Highlights**: HEA는 최신 LLM에 대해 평균 88.79%의 공격 성공률(ASR)을 달성했습니다. HEA는 고도 상용 및 오픈 소스를 포함한 여러 LLM에서 성능을 테스트하였으며, 감정 분류 및 중요도 히트맵을 이용해 성공적인 이유에 대한 정량적 설명을 제공했습니다. 이러한 발견은 LLM의 안전성 향상을 위한 새로운 연구 아이디어를 제공할 수 있습니다.



### Academic Case Reports Lack Diversity: Assessing the Presence and Diversity of Sociodemographic and Behavioral Factors related to Post COVID-19 Condition (https://arxiv.org/abs/2501.12538)
- **What's New**: 이 연구는 Post COVID-19 Condition (PCC)에 대한 사회적 건강 결정 요인(SDOH)을 통합하는 포괄적인 프레임워크를 개발하는 것을 목표로 합니다. 문서 내 SDOH의 불균형과 변화를 분석하기 위해 NLP 기술을 활용했습니다. 이를 통해 PCC 사례 보고서에서 26개 핵심 SDOH 관련 엔티티 유형을 식별하고 분석했습니다.

- **Technical Details**: 7,000개 이상의 사례 보고서로 구성된 PCC 사례 보고서 코퍼스를 구축하였고, 709개의 보고서에 대해 사전 훈련된 Named Entity Recognition (NER) 모델을 사용하여 주석을 달았습니다. NER, 자연어 추론(NLI), 3-그램(trigram) 및 빈도 분석을 통합한 NLP 파이프라인이 개발되어 이러한 엔티티를 추출하고 분석했습니다. 특히, 인코더 전용 BERT 모델이 전통적인 RNN 모델보다 더 나은 일반화를 보여주었습니다.

- **Performance Highlights**: 탐색적 분석을 통해 엔티티의 풍부함에 변동성이 있으며, condition, age, care access와 같은 일반적인 엔티티가 많이 나타났습니다. 그러나 race 및 housing status와 같은 민감한 카테고리는 저조한 대표성을 보였습니다. NLI 분석에서는 'Experienced violence or abuse'와 'Has medical insurance'와 같은 속성이 높은 관계성(entailment)률을 나타내는 반면, 'Is female-identifying', 'Is married'와 같은 속성은 높은 모순(contradiction)률을 보였습니다.



### Each Graph is a New Language: Graph Learning with LLMs (https://arxiv.org/abs/2501.11478)
- **What's New**: 이 논문은 Large Language Models (LLMs)을 활용하여 노드 분류 작업에 대한 텍스트 속성을 가진 그래프 구조를 모델링하는 새로운 프레임워크인 GDL4LLM(Graph-Defined Language for Large Language Model)을 제안합니다. 기존 접근법에서는 그래프 구조 표현이 너무 방대하거나 텍스트 속성만으로는 충분한 정보를 제공하지 못하는 한계를 가지고 있었습니다. GDL4LLM은 그래프를 설명하는 대신 그래프 언어 말뭉치를 생성하고 이를 통해 LLM을 사전 훈련시킴으로써 그래프 구조를 효과적으로 이해할 수 있게 합니다.

- **Technical Details**: GDL4LLM은 그래프를 그래프 언어 코퍼스로 변환하여 노드 분류를 위한 프레임워크를 구현합니다. 이 과정에서, 최근 LLM이 하나의 언어에서 훈련되어도 다른 언어에서 우수한 성능을 보일 수 있다는 점을 활용했습니다. LLM은 이 그래프 언어를 통해 노드에 대한 구조적 정보를 간결하게 학습할 수 있으며, 이는 터치포인트 중심으로 구조적 정보를 설명하는 데에 필요합니다.

- **Performance Highlights**: 실험을 통해 GDL4LLM은 세 가지 실제 데이터셋에서 기존의 설명 기반 접근법이나 텍스트 속성 임베딩 기반 방법들보다 뛰어난 성능을 보였습니다. LLM을 활용하여 다양한 차수의 그래프 구조를 효율적으로 모델링함으로써 노드 분류 작업에서 탁월한 결과를 나타냈습니다. 전반적으로, GDL4LLM은 LLM의 언어 이해 능력을 그래프 구조 데이터에 성공적으로 이전시킬 수 있도록 설계되었습니다.



New uploads on arXiv(cs.LG)

### PBM-VFL: Vertical Federated Learning with Feature and Sample Privacy (https://arxiv.org/abs/2501.13916)
- **What's New**: 본 논문에서는 Poisson Binomial Mechanism Vertical Federated Learning (PBM-VFL)라는 새로운 방식의 알고리즘을 제안합니다. PBM-VFL는 Differential Privacy 보장을 갖춘 효율적인 통신 방식을 활용하여 개인 데이터 보호를 강조하고 있습니다. 이 연구는 Train 알고리즘의 수렴과 개인 정보 보호 간의 관계를 이론적으로 분석했습니다.

- **Technical Details**: PBM-VFL은 Secure Multi-Party Computation (MPC)과 Poisson Binomial Mechanism (PBM)을 결합하여 데이터셋에 대해 Differential Privacy를 제공합니다. 각 참여자는 원시 데이터를 통해 임베딩(embeddings)으로 변환한 후, 중앙 서버에서 이러한 임베딩을 집계하여 예측 라벨을 생성합니다.

- **Performance Highlights**: 모델은 ModelNet-10과 Cifar-10 데이터셋에서 평가되었으며, 높은 개인 정보 보호 매개변수를 유지하면서도 높은 정확도를 기록했습니다. 또한, PBM-VFL의 개인 정보 보호 예산과 통신 비용 간의 관계를 명확히 제시했습니다.



### On Learning Representations for Tabular Data Distillation (https://arxiv.org/abs/2501.13905)
- **What's New**: 이번 연구에서는 데이터 세트 증류(dataset distillation)의 새로운 접근법인 TDColER를 소개합니다. TDColER는 테이블 형식의 데이터 전처리 시 비차별적(non-differentiable) 모델의 특징을 효과적으로 활용할 수 있도록 두 가지 주요 기법을 포함하고 있습니다. 또한, TDBench라는 벤치마크 프레임워크를 제공하여 다양한 데이터 세트와 모델에 대한 실험을 통해 성과를 측정할 수 있습니다.

- **Technical Details**: TDColER는 열 임베딩(column embeddings)을 기반으로 한 표현 학습(representation learning) 방법을 사용해 테이블 형식의 데이터 증류를 수행합니다. 이 프레임워크는 최신 신경망 아키텍처인 Transformers와 그래프 신경망(graph neural networks)을 사용하여 풍부하고 압축된 표현을 생성할 수 있도록 설계되었습니다. 기존의 증류 방법들과 비교했을 때, TDColER는 증류 데이터의 품질을 향상시킵니다.

- **Performance Highlights**: TDBench에서의 철저한 평가를 통해, TDColER는 다양한 데이터 세트와 모델에서 증류된 데이터의 품질을 0.5%에서 143%까지 향상시켰습니다. 226,890개의 증류 데이터 세트와 548,880개의 모델 훈련 결과를 기반으로, TDColER는 모든 모델의 경우 최소 10 클래스당 10개 인스턴스로 최적의 증류 성능을 보여주었습니다. 이 연구는 테이블형 데이터에 대한 데이터 증류의 중요한 통찰력을 제공합니다.



### Privacy-Preserving Personalized Federated Prompt Learning for Multimodal Large Language Models (https://arxiv.org/abs/2501.13904)
Comments:
          Accepted to ICLR 2025 main conference track

- **What's New**: 최근 여러 모달리티(멀티모달) 대규모 언어 모델(LLMs)과 연계된 새로운 기법인 차등 프라이버시 연합 프롬프트 학습(DP-FPL)이 제안되었습니다. 이 연구는 개인화, 일반화, 및 프라이버시 간의 균형을 맞추는 데 도전하며, 저순위 적응(LoRA) 방식을 활용하여 모델의 일반화 능력을 향상시키면서도 개인화 능력을 유지합니다. 또한, 로컬 프롬프트의 두 저순위 구성 요소에 로컬 차등 프라이버시를 적용하고, 글로벌 프롬프트에는 글로벌 차등 프라이버시를 적용함으로써 프라이버시를 보장합니다.

- **Technical Details**: DP-FPL 접근법은 각 클라이언트가 글로벌 프롬프트와 로컬 프롬프트를 동시에 학습하는 방식으로 진행됩니다. 로컬 프롬프트는 저순위 구성 요소 두 개로 분해되며, 추가적인 잔여 항(residual term)을 통해 개인화의 표현력을 유지합니다. 이 과정에서 각 클라이언트는 데이터를 기반으로 개인화된 프롬프트를 학습하여 중앙 서버와 공유하여 일반화된 지식 전이를 수행합니다.

- **Performance Highlights**: 제안된 DP-FPL 방식은 실험에서 기존의 다른 벤치마크에 비해 개인화와 일반화 간의 균형을 효과적으로 맞추는 것으로 확인되었습니다. B이 연구는 모델 성능을 저하시킬 수 있는 DP 노이즈의 영향을 완화하면서도 개인화 요구를 충족합니다. 실험 결과는 DP-FPL이 개인화와 일반화를 효과적으로 조화시키는 능력을 갖추었다는 것을 보여줍니다.



### Federated Granger Causality Learning for Interdependent Clients with State Space Representation (https://arxiv.org/abs/2501.13890)
- **What's New**: 이번 연구에서는 산업 데이터의 증가와 복잡성이 Granger 인과성을 분석하는 데 있어 중요한 도전 과제가 됨을 강조합니다. 이를 해결하기 위해 여러 클라이언트가 중앙 서버와 저차원 상태 정보만 공유함으로써 Granger 인과성을 협력적으로 학습할 수 있는 연합 학습(federated learning) 프레임워크를 개발했습니다. 이러한 접근 방식은 일반적인 중앙 집중식 데이터 처리의 대역폭 제한과 계산 부담을 해소하려는 시도를 보여줍니다.

- **Technical Details**: 논문에서는 LTI(linear time-invariant) 상태공간(state space) 표현 방법을 사용하여 다중 클라이언트의 모델을 구성합니다. 클라이언트는 자신의 클라이언트 전용 정보만을 기반으로 독립적으로 운영되며, 중앙 서버에 저차원 상태만을 공유하여 상호 의존성을 학습합니다. 이를 통해 클라이언트 모델은 서버에서 학습된 Granger 인과성 정보를 활용하여 점진적으로 수렴하게 됩니다.

- **Performance Highlights**: 모형의 성능을 평가하기 위해 합성 데이터(synthetic data) 실험을 진행하여 인과성의 교란에 대한 견고함과 통신 효율성을 입증하였습니다. 실제 산업 제어 시스템 데이터셋에서도 프레임워크의 유효성을 확인하였으며, 데이터 분산화를 통해 대규모 고차원 측정치를 활용할 수 있음에도 불구하고 훈련 손실을 저하시키지 않고 데이터 양을 줄일 수 있는 성과를 보여줍니다.



### Multimodal Sensor Dataset for Monitoring Older Adults Post Lower-Limb Fractures in Community Settings (https://arxiv.org/abs/2501.13888)
- **What's New**: 이번 논문에서는 노인들의 Lower-Limb Fractures (LLF) 회복을 위한 새로운 공개 다중 모드 센서 데이터셋인 MAISON-LLF를 제시합니다. 이 데이터셋은 노인들이 LLF로부터 회복하는 과정에서 수집된 데이터를 포함하고 있으며, 스마트폰, 스마트워치 센서, 동작 감지기, 수면 추적 매트리스, 그리고 고립 및 기능 저하에 관한 임상 설문지를 통해 이루어졌습니다.

- **Technical Details**: 데이터셋은 10명의 독거 노인으로부터 8주 동안 수집되었고, 총 560일의 24시간 센서 데이터를 포함합니다. 연구진은 수집된 센서 데이터와 임상 설문지를 사용하여 감독 학습(supervised learning) 및 딥 러닝(deep learning) 모델을 개발하여 기술적 검증을 실시했습니다.

- **Performance Highlights**: MAISON-LLF 데이터셋은 소외 및 저하 위험에 처한 노인을 원격으로 모니터링할 수 있는 기초를 제공하며, 머신러닝 알고리즘을 사용하여 건강 결과를 추론할 수 있도록 합니다. 이 데이터셋은 연구 커뮤니티에게 새로운 비교 기초를 제공하며, 향후 rehabilitation(재활) 연구에 중요한 자원으로 작용할 것입니다.



### What Does an Audio Deepfake Detector Focus on? A Study in the Time Domain (https://arxiv.org/abs/2501.13887)
- **What's New**: 이번 논문에서는 오디오 딥페이크 탐지(AADD) 모델에 대한 설명 가능 인공지능(XAI) 기법을 제안하여, 이러한 모델의 실제 응용 가능성을 높이고자 했습니다. 제안된 방법은 Relevancy 기반의 XAI 방법으로, 기존 Grad-CAM 및 SHAP 방법과 비교하여 다양한 지표에서 성능을 분석했습니다. 데이터세트의 크기를 확장하여 기존 연구에서 사용된 제한된 발화수와는 달리 방대한 데이터를 사용하여 설명의 차이를 발견했습니다.

- **Technical Details**: 일반적으로 Wav2Vec2 기반의 딥페이크 탐지 모델을 예로 들어 XAI 방법의 효과를 분석하였으며, Grad-CAM, DeepSHAP, GradientSHAP, GATR(Gradient Average Transformer Relevancy) 등 총 네 가지 XAI 방법을 적용했습니다. 특히, GATR 방법은 오디오 신호의 각 타임스텝이 모델의 결과에 미치는 영향을 직접 보여줄 수 있도록 설계되었습니다. 이러한 방법은 여러 XAI 기법의 효과성을 비교하여 각 음성 영역의 중요성을 평가하는 실험을 시행했습니다.

- **Performance Highlights**: GATR 방법은 종합적인 메트릭에서 기존 XAI 기법보다 우수한 성능을 보였으며, 이는 특정 음향 영역의 중요성을 강조하고 매우 다양한 씬에서 비교 분석할 수 있게 해줍니다. 연구진은 음성/비음성, 음소 콘텐츠 및 음성 시작/끝 지점과 같은 요소들이 모델 결정에 미치는 상대적 중요성을 진단하기 위한 가설 검증을 수행했습니다. 결과적으로, 한정된 발화에서 얻어진 XAI 결과는 대규모 데이터셋에서 평가할 때 반드시 일치하지 않는다는 점이 발견되었습니다.



### Utilizing Evolution Strategies to Train Transformers in Reinforcement Learning (https://arxiv.org/abs/2501.13883)
- **What's New**: 이 논문은 강화 학습 설정에서 변환기 아키텍처(transformer architecture)를 기반으로 하는 에이전트의 정책(policy)을 훈련시키기 위해 진화 전략(evolution strategies)을 사용하는 가능성을 탐구합니다. 저자들은 OpenAI의 고도로 병렬화할 수 있는 진화 전략을 활용하여 Decision Transformer를 훈련하는 실험을 수행하였으며, 비교적 크고 복잡한 모델을 훈련할 수 있는지를 검증했습니다. 이 과정에서 새로운 학습 방법을 제안하여 모델을 사전 훈련(pretraining)한 후 진화 전략으로 추가 훈련을 통해 모델의 효과성을 테스트했습니다.

- **Technical Details**: 진화 알고리즘은 블랙 박스(black-box) 최적화 알고리즘의 한 형태로, 자연에서 영감을 받은 여러 메타 휴리스틱(metaheuristics)으로 구성됩니다. 이 논문에서는 OpenAI-ES를 사용한 진화 전략을 적용하여 에이전트를 신경망(neural network)으로 표현하고, 개체의 적합도(fitness)를 여러 에피소드에서 에이전트가 반환한 평균으로 설정합니다. 이후 각 세대에서 새로운 개체를 생성하고 이를 평가하여, 다음 세대를 형성하는 방식으로 작업을 진행하였습니다.

- **Performance Highlights**: 실험 결과 진화 전략이 긴밀한 성능을 유지하면서 높은 성과를 달성할 수 있음을 확인하였습니다. 사전 훈련이 불필요하다는 것이 밝혀졌으나, 여러 통찰(insight)을 발견하는 데에는 도움이 되었습니다. 이러한 가시적인 성과는 에이전트가 복잡한 작업에서 잘 작동할 수 있음을 보여주며, 진화 전략이 다음 세대의 모델 훈련에 어떠한 기여를 할 수 있는지를 시사합니다.



### Autoencoders for Anomaly Detection are Unreliab (https://arxiv.org/abs/2501.13864)
- **What's New**: 이 연구는 Autoencoder가 이상 탐지에서 신뢰할 수 없는 결과를 제공할 수 있다는 점을 이론적으로 증명합니다. 특히, 연속적인 이상 데이터의 재구성이 가능하다는 사실을 강조하며, 이는 기존의 추정과 반합니다. 이러한 결과는 Autoencoder의 구조와 적용 방식에 대해 새로운 통찰력을 제공합니다.

- **Technical Details**: Autoencoder는 비지도 학습 및 준지도 학습 환경에서 이상 탐지에 널리 사용됩니다. 이 연구는 reconstruction loss를 기반으로한 기존의 가정이 잘못되었음을 보여주며, 특히 linear Autoencoder가 어떻게 비정상 데이터도 잘 재구성할 수 있는지를 다룹니다. 다양한 활성화 함수가 실패 원인에 미치는 영향을 실험적으로 분석합니다.

- **Performance Highlights**: 이 연구 결과는 이상 탐지 알고리즘의 안전성에 대한 우려를 반영합니다. Autoencoder는 다양한 데이터 세트에서 우수한 성능을 보이지만, 이상 데이터에 대한 높은 재구축 능력 때문에 신뢰성에 문제가 발생할 수 있습니다. 이러한 실험적 증거를 바탕으로, 연구자들은 Autoencoder의 적절한 적용을 위한 더 강력한 탐지 알고리즘 개발이 필요하다고 제안합니다.



### Large Vision-Language Models for Knowledge-Grounded Data Annotation of Memes (https://arxiv.org/abs/2501.13851)
Comments:
          18 pages, 5 figures, 13 tables, GitHub repository: this https URL

- **What's New**: 이번 연구에서는 33,000개 이상의 밈(meme)으로 구성된 새로운 대규모 데이터 세트인 ClassicMemes-50-templates (CM50)를 소개합니다. 이는 50개의 인기 밈 템플릿을 중심으로 하여 밈에 대한 깊이 있는 이해와 텍스트 검색을 가능하게 합니다. 또한, 기존의 수작업 주석 작업을 자동화하는 최신 기술을 활용하여 고품질의 이미지 캡션과 문학적 장치 레이블을 생성하는 파이프라인을 개발하였습니다.

- **Technical Details**: CM50 데이터 세트는 GPT-4를 기반으로 한 자동 주석화 파이프라인을 통해 구축되었으며, 이는 주석 태그 문제를 상당히 줄여줍니다. 이 연구에서는 또한 미세 조정된 CLIP 모델을 사용하여 이미지와 텍스트의 교차 모드 임베딩을 활용하여, 밈 분석의 효율성을 크게 향상시키는 meme-text retrieval 모델(mtrCLIP)을 제안합니다. 기존의 복잡한 밈 분석 파이프라인을 단순화할 필요성이 강조되고 있습니다.

- **Performance Highlights**: 이 연구의 핵심 기여는 대규모 밈 연구를 위한 새로운 데이터 세트, 스케일 가능한 밈 주석 프레임워크 및 밈-텍스트 검색을 위한 향상된 CLIP 모델의 개발입니다. 실험 결과, 기존의 CLIP 모델이 메타-캡션과 밈 간의 검색에서 뛰어난 성능을 보였으나 여전히 보다 깊이 있는 의미를 포착하기 위해서는 향상될 여지가 있습니다.



### Learning to Help in Multi-Class Settings (https://arxiv.org/abs/2501.13810)
Comments:
          30 pages, 7 figures, conference, ICLR 2025

- **What's New**: 최근 연구된 Learning to Help (L2H) 모델은 고정된 로컬(클라이언트) 모델을 기반으로 서버 모델을 학습시키는 방식으로, 이는 고정된(전문가) 서버를 위한 클라이언트를 학습시키는 Learning to Defer (L2D) 프레임워크와 구별됩니다. 본 연구에서는 L2H 모델을 다중 클래스( multi-class) 분류 문제로 확장하여 서버 접근 비용이나 정책으로 인해 현실적 관심이 있는 다양한 시나리오에서의 적용 가능성을 보여줍니다. 제안된 방법은 자원 제약 환경에서도 다중 클래스 분류를 위한 효율적이고 실용적인 솔루션을 제공합니다.

- **Technical Details**: L2H 모델의 추정 목표는 Bayes 규칙에 부합하는 미분 가능하고 볼록한 에코스에 기초한 잔여 손실 함수(surrogate loss function)를 통해 성능을 평가하는 것입니다. 우리의 접근 방식은 PPR, IA, BRR 세 가지의 시나리오를 고려하여 다중 클래스 예측기를 훈련하는 알고리즘을 설계합니다. 이 과정에서, 후처리 방법(post-hoc method)을 사용하여 거부율(BRR)을 보장하고, 서버 모델과 리젝터를 통합하여 ML 시스템의 전체 성능을 증대시킵니다.

- **Performance Highlights**: 제안된 모델이 포함된 시스템의 실험 결과는 다양한 시나리오에서의 성능이 개선되었음을 보여줍니다. 리젝터는 불확실하거나 도전적인 샘플을 효과적으로 식별하고, 서버 모델 사용의 정확성과 효율성을 균형 있게 조절하는 역할을 합니다. 이 연구는 자원 제약 환경에서 클라이언트와 서버 간의 효과적인 협업 발생 가능성을 제시하며, 이러한 하이브리드 ML 시스템의 구축에 대한 새로운 통찰을 제공합니다.



### Unveiling the Power of Noise Priors: Enhancing Diffusion Models for Mobile Traffic Prediction (https://arxiv.org/abs/2501.13794)
- **What's New**: 이번 논문에서는 모바일 트래픽 예측을 위한 혁신적인 접근법인 NPDiff를 소개합니다. 기존의 노이즈 중심의 접근법 대신, 데이터의 동적 패턴을 활용하면서 노이즈를 두 가지 구성 요소인 	extit{prior}와 	extit{residual}로 분해하는 새로운 틀을 제공합니다. 이러한 NPDiff는 잘 정의된 패턴과 갑작스러운 변동을 포착하는 데 뛰어난 모델로 발전할 수 있는 가능성을 제시합니다.

- **Technical Details**: NPDiff는 모바일 트래픽 데이터의 고유한 동적 특성을 반영하여 노이즈를 효과적으로 추정할 수 있도록 설계되었습니다. 이 모델은 각 예측 단계에서 노이즈 	extit{prior}을 이용해 기본적인 참조를 제공하며, 이를 통해 예측 성능을 향상시킬 수 있습니다. 또한, NPDiff는 기존 확산 모델과 원활하게 통합되어 더 나은 예측성과 적응성을 제공합니다.

- **Performance Highlights**: NPDiff는 다양한 예측 작업에서 30% 이상의 정확도 향상을 기록하며, 훈련 효율성과 강인성을 동시에 개선합니다. 특히, 예측의 불확실성을 줄여주는 효율적이고 다재다능한 개선책으로 자리 잡을 가능성이 높습니다. 이를 통해 기존의 확산 모델들이 모바일 트래픽 예측에서 더 나은 성과를 낼 수 있도록 지원합니다.



### Local Steps Speed Up Local GD for Heterogeneous Distributed Logistic Regression (https://arxiv.org/abs/2501.13790)
Comments:
          ICLR 2025

- **What's New**: 이 논문에서는 특정한 이질적인 분리 가능한 데이터에 대해 분산 로지스틱 회귀(distributed logistic regression)에 적용된 로컬 그래디언트 디센트(Local Gradient Descent)의 두 변형을 분석합니다. 이 방법의 수렴(convergence) 속도가 기존의 방법들과 비교하여 현저히 개선된 것을 보여줍니다. 특히, 로컬 업데이트(local updates)의 장점을 드러내는 방법론을 제시합니다.

- **Technical Details**: 제안된 방법은 $K$ 회의 로컬 스텝(local steps)과 충분히 큰 $R$ 커뮤니케이션 라운드(communication rounds)에서 $O(1/KR)$의 수렴 속도를 나타냅니다. 기존의 로컬 GD(Local GD) 관련 연구들은 최소한 $Ω(1/R)$의 속도를 보여주며, 이는 로컬 업데이트의 유용성을 반영하지 못합니다. 이번 연구의 핵심은 큰 스텝 크기(stepsize) $	au 	extgreater	extgreater 1/K$에서 로지스틱 회귀 목표(logistic regression objective)에서의 진전을 보여주는 것입니다.

- **Performance Highlights**: 이번 연구에서 제안하는 로컬 그래디언트 디센트는 각 로컬 스텝에서 획득할 수 있는 성능이 눈에 띄게 향상되었습니다. 효과적인 스텝 크기를 활용함으로써, 이 방식은 데이터의 비균질성(heterogeneity)에도 불구하고 뛰어난 성능을 발휘하는 것으로 나타났습니다. 따라서 이 접근법은 분산 환경에서의 머신러닝 알고리즘 성능을 크게 개선할 수 있는 가능성을 제시합니다.



### Fast Iterative and Task-Specific Imputation with Online Learning (https://arxiv.org/abs/2501.13786)
- **What's New**: 본 논문에서는 K-최근접 이웃(K-nearest neighbor) 보간(imputation) 방법인 F3I를 제안합니다. 이 방법은 각 데이터 포인트의 이웃에 대한 가중치를 학습하여 데이터 포인트에 대한 최적의 분포를 제공하도록 최적화됩니다. 특히 이 알고리즘은 데이터 임플루션(imputation) 품질을 보장하며, 다운스트림 작업(downstream task)과 함께 훈련이 가능합니다.

- **Technical Details**: F3I는 데이터 분포를 보존하면서 보간 성능을 개선하기 위한 새로운 목적 함수를 기반으로 하며, 온라인 학습(online learning)에서 전문가 조언 문제와 유사한 방식으로 이를 최적화하는 빠른 루틴을 포함합니다. 이 알고리즘은 MCAR, MAR 및 MNAR 환경에서 이론적으로 품질을 보장할 수 있습니다. 또한, F3I는 데이터가 결측된 경우에도 효과적으로 작업과 결합하여 성능을 높일 수 있습니다.

- **Performance Highlights**: F3I는 합성 데이터 및 실제 약물 재사용(drug repurposing)과 손글씨 숫자 인식(handwritten-digit recognition) 데이터셋에서 성능을 입증하였습니다. 기존 방법들과 비교할 때 F3I는 더 나은 품질의 보간을 제공하며, 계산 효율성과 데이터 분포 보존을 모두 유지합니다. 실험 결과는 F3I가 다운스트림 머신러닝 작업에 유용하게 적용될 수 있음을 보여줍니다.



### Not Every AI Problem is a Data Problem: We Should Be Intentional About Data Scaling (https://arxiv.org/abs/2501.13779)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLM)의 데이터 스케일링에 대한 접근 방식을 재조명하고 있습니다. 기존의 데이터 스케일링 관점에서 벗어나, 특정 작업이 데이터 스케일링을 통해 더 큰 혜택을 받을 수 있는지에 대한 의도를 가지고 데이터를 수집하는 것이 중요하다고 주장합니다. 연구자들은 데이터의 '형태(shape)'가 데이터 스케일링에서 우선시해야 할 작업들을 결정하고, 비효율적인 스케일링 작업을 위한 새로운 컴퓨팅 패러다임을 구상하는 데 중요한 역할을 한다고 강조합니다.

- **Technical Details**: 대규모 언어 모델의 발전은 '스케일링 법칙(scaling laws)'에 뿌리를 두고 있으며, 이 법칙은 모델과 데이터셋의 크기를 일정하게 조정하여 최적의 성능을 이끌어내는 과정을 설명합니다. 그러나 실제로는 고품질 데이터의 수가 제한적이기 때문에, 대량의 저질 데이터로 모델이 손상될 위험이 커지며, 이러한 문제는 특히 파라미터 수가 많은 대형 모델에서 두드러집니다. 이는 인간 생성 데이터의 양적 확보와 그 품질 보장이 필수적임을 의미합니다.

- **Performance Highlights**: 데이터 스케일링이 유망한 분야에 대한 사례로는 기계 번역(Machine Translation), 로보틱스, 신약 발견 등이 있습니다. 기계 번역의 경우, 규칙적인 언어 구조와 고품질 데이터가 모델 훈련에 기여하여 상당한 향상을 이루었습니다. 반면, 복잡한 논리적 추론이 필요한 작업에서는 여전히 모델이 인간보다 부족한 성능을 보이며, 이 문제는 모델 구조와 학습 알고리즘에 뿌리를 두고 있는 것으로 평가되었습니다.



### Crossfire: An Elastic Defense Framework for Graph Neural Networks Under Bit Flip Attacks (https://arxiv.org/abs/2501.13776)
Comments:
          Accepted at AAAI 2025, DOI will be included after publication

- **What's New**: 본 연구에서는 비트 플립 공격(Bit Flip Attacks, BFA)이 그래프 신경망(Graph Neural Networks, GNNs)에 미치는 영향과 이에 대한 방어 방법을 탐구합니다. 특히 기존의 CNN 방어 방법을 GNNs에도 적용해보는 과정에서 드러나는 한계점을 지적하며, 새로운 방어 기제인 Crossfire를 제안합니다. Crossfire는 GNN의 무결성을 복원하기 위해 해싱(hashing)과 허니팟(honeypots)을 결합한 하이브리드 접근 방식을 채택합니다.

- **Technical Details**: Crossfire의 주된 특징은 비트 수준(bit-level)에서 분포 밖(weight OOD) 항목을 수정하여 네트워크 무결성을 복원하는 점입니다. 또한 학습(labeling)된 데이터가 필요 없고, 재훈련이 필요하지 않다는 점에서 기존 방어 메커니즘의 한계를 극복합니다. 실험 결과, Crossfire는 GNN이 BFA에 의해 공격 받았을 때 원래 상태로 복원할 확률이 21.8% 향상되었고, 예측 품질도 10.85% 개선되었습니다.

- **Performance Highlights**: Crossfire는 6개의 벤치마크 데이터셋을 사용하여 2,160회의 실험을 수행했으며, 최대 55회 비트 플립 공격을 포함하였습니다. 이러한 실험을 통해 Crossfire의 성능은 기존 방법들과 비교하여 우수함을 입증했습니다. 이처럼 Crossfire는 계산 및 저장 오버헤드가 미미하여 효율적인 방어 메커니즘으로 자리 잡을 것으로 기대됩니다.



### An Efficient Diffusion-based Non-Autoregressive Solver for Traveling Salesman Problem (https://arxiv.org/abs/2501.13767)
Comments:
          Accepted at KDD2025

- **What's New**: DEITSP는 Traveling Salesman Problem (TSP)을 해결하기 위해 개발된 효율적인 확산 모델입니다. 이 모델은 Non-Autoregressive (NAR) 방식으로 작동하며, 빠른 추론 속도를 유지하면서도 높은 품질의 솔루션을 제공합니다. 특히, DEITSP는 소음 추가 및 제거를 반복하여 한 번의 단계에서 여러 솔루션을 동시에 탐색하는 혁신적인 전략을 도입하였습니다.

- **Technical Details**: DEITSP의 주요 구성 요소는 크게 세 가지로 나눌 수 있습니다. 첫 번째는 제어된 이산 소음 추가 프로세스와 자기 일관성 강화를 활용한 1단계 확산 모델로, 여러 솔루션을 동시에 denoising을 통해 최적의 예측을 수행합니다. 두 번째는 이중 모드 그래프 변환기로, 노드 및 엣지 모디털리의 특징을 추출하고 융합하는 데에 효과적이며, 더 적은 레이어로 빠른 추론을 가능하게 합니다. 세 번째는 효율적인 반복 전략으로, 탐색을 개선하고 솔루션 공간을 정제하는 데 기여합니다.

- **Performance Highlights**: DEITSP는 20에서 1000 노드까지의 다양한 TSP 인스턴스를 대상으로 진행된 실험에서 기존의 16개 신경망 기반 모델 중 최고 품질의 솔루션을 제공했습니다. 특히, DEITSP는 작은 문제에서 큰 문제까지 잘 일반화되어 실제 세계의 사례에서도 효과적으로 적용될 수 있는 능력을 보여줍니다. 이러한 성과는 솔루션 품질, 추론 지연, 일반화 능력 측면에서 모두 두드러지며, 실험 결과는 DEITSP의 우수성을 입증합니다.



### Integrating Causality with Neurochaos Learning: Proposed Approach and Research Agenda (https://arxiv.org/abs/2501.13763)
Comments:
          9 pages

- **What's New**: 이 논문은 심층 학습(deep learning)의 한계를 극복하기 위한 두 가지 대안적 접근법인 인과 학습(causal learning)과 신경 혼돈 학습(Neurochaos Learning) 소개에 중점을 둡니다. 인과 학습은 데이터셋 내 아이템 간 인과 관계를 고려하여 잘못된 상관관계를 줄이는 데 기여할 것으로 기대됩니다. 반면, 신경 혼돈 학습은 생물학적 신경망에서의 비선형적인 혼돈 발사를 기반으로 하여, 적은 수의 샘플로도 효과적인 분류 성능을 입증하고 있습니다.

- **Technical Details**:  심층 학습 모델은 여러 처리(layer)로 구성되어 데이터의 다층적 표현을 학습합니다. 하지만 이러한 모델은 통계적 상관관계에 의존하여 학습하며, 많은 경우 스푸리어스(spurious)한 상관관계를 초래합니다. 이 논문에서는 인과 모델과 그래프 신경망을 통합하여 더 나은 성능을 발휘하도록 하는 방법을 제안하며, 특히 링크된 데이터(linked data) 도메인에서 효과적인 결과를 기대하고 있습니다.

- **Performance Highlights**: 인과 학습과 신경 혼돈 학습을 통합함으로써 데이터의 구조적 모델을 활용해 예측 모델링(predictive modeling)과 강화 학습(reinforcement learning)에 대한 효과를 증대시킬 수 있습니다. 실험 결과, 이 두 접근법의 통합이 기존의 단순 심층 학습 모델보다 우수한 성과를 보여주었습니다. 이러한 개선된 성능은 IoT, 의료 및 생명 과학, 제조업 등 다양한 분야에서 활용될 가능성을 보여주고 있습니다.



### Exact Soft Analytical Side-Channel Attacks using Tractable Circuits (https://arxiv.org/abs/2501.13748)
Comments:
          ICML 2024 Conference Paper

- **What's New**: 이 논문에서는 최신의 소프트 분석 사이드 채널 공격(SASCA)에 대한 빠르고 정확한 추론 알고리즘인 ExSASCA를 개발하였습니다. 이는 지식 컴파일(knowledge compilation)과 계산 가능 확률 회로(tractable probabilistic circuits)를 활용하여 AES 알고리즘의 약점을 효과적으로 탐지합니다. ExSASCA는 기존 SASCA보다 31% 향상된 성공률을 기록하면서도 훨씬 적은 계산 자원을 사용합니다.

- **Technical Details**: ExSASCA는 AES의 MixColumns라는 부분을 compact (probabilistic) sentential decision diagram(SDD)으로 컴파일하여, 전통적인 loopy belief propagation의 한계를 극복합니다. 이 방법은 계산 부하가 낮고, polynomial time으로 추론을 수행할 수 있도록 합니다. 느슨한 믿음 메시지를 활용하여 성능을 극대화하며, 이는 기존의 SASCA에서 사용된 방식보다 훨씬 더 효율적입니다.

- **Performance Highlights**: ExSASCA는 AES 공격 시 31%라는 높은 성공률 개선을 보여주었으며, SASCA와 비슷한 수준의 계산 비용으로 구현됩니다. 또한, 조밀한 믿음 메시지를 사용할 경우에도 exhaustive enumeration 방식보다 6배 적은 계산을 필요로 합니다. 이러한 성과는 모든 보호 수준에서 SASCA를 크게 초과하는 성능을 나타내며, 사이드 채널 공격에 대한 새로운 연구 경로를 열어줍니다.



### GPT-HTree: A Decision Tree Framework Integrating Hierarchical Clustering and Large Language Models for Explainable Classification (https://arxiv.org/abs/2501.13743)
- **What's New**: 본 논문은 GPT-HTree라는 프레임워크를 소개하며, 이는 계층적 클러스터링(hierarchical clustering), 결정 트리(decision trees), 대규모 언어 모델(Large Language Models, LLMs)을 조합하여 다양한 데이터 세트를 효과적으로 분류하는 방안을 제시합니다. 이 방법론은 데이터에서 중요한 특성을 기반으로 개인을 세분화하고, 재샘플링 기법을 통해 클래스의 분포를 균형 있게 유지하며, 각 클러스터 내에서 맞춤형 분류 경로를 제공하여 정확성과 해석 가능성을 보장합니다. 또한, LLMs는 클러스터에 대한 인간 친화적인 설명을 생성하여 정량적 분석과 실행 가능한 인사이트를 연결하는 역할을 합니다.

- **Technical Details**: GPT-HTree 프레임워크는 먼저, 데이터를 균형 있게 유지하기 위해 조건부 표 형식 생성적 적대 신경망(CTGAN)을 사용하여 재샘플링을 수행합니다. 그런 다음, 계층적 클러스터링을 통해 특성 유사성을 기반으로 그룹을 식별하며, 마지막으로 결정 트리를 통해 각 클러스터 내에서 설명 가능한 분류 규칙을 제공합니다. LLMs는 각 클러스터의 인격 설명을 생성하며, 이는 통계적 패턴과 실행 가능한 인사이트 간의 간극을 매꿉니다.

- **Performance Highlights**: 이 연구의 결과는 GPT-HTree가 기존의 임의 성공률보다 9배 높은 성과를 달성한 클러스터를 발견했음을 보여줍니다. 초기 성공률 1.9%에서 17.4%로 상승한 것은 투자 결정을 내리기 위한 유망한 클러스터를 성공적으로 식별하는데 도움을 줍니다. 이러한 성과는 벤처 캐피털(VC) 분야의 중요성을 부각시키며, 사용자는 이 새로운 접근방식을 통해 데이터 기반의 인사이트를 바탕으로 더 나은 의사 결정을 할 수 있습니다.



### Sample complexity of data-driven tuning of model hyperparameters in neural networks with structured parameter-dependent dual function (https://arxiv.org/abs/2501.13734)
Comments:
          48 pages, 4 figures

- **What's New**: 이 논문은 딥러닝의 하이퍼파라미터 튜닝의 복잡성을 형식적으로 연구하는 새로운 접근 방법을 제시합니다. 기존의 데이터 기반 디자인에서는 하이퍼파라미터와 알고리즘 행동을 명시적으로 모델링할 수 있었지만, 이 연구에서는 성능이 하이퍼파라미터의 함수로 매우 변동성이 크고 암묵적으로 최적화 문제로 주어짐을 지적합니다. 이를 극복하기 위해, 유틸리티 함수의 불연속성과 진동을 특성화할 새로운 기법을 소개합니다.

- **Technical Details**: 연구는 데이터 기반 알고리즘 디자인의 관점에서 하이퍼파라미터 튜닝을 접근합니다. 튜닝해야 할 연속 하이퍼파라미터와 입력 문제 인스턴스를 데이터로 간주하여, 하이퍼파라미터 튜닝 문제를 학습 문제로 축소하는 구조입니다. 이는 다수의 알고리즘을 파라미터화된 가족으로 간주하고, 특정 문제 인스턴스에 대한 유틸리티 함수를 정의하여 이를 분석하는 방식으로 진행됩니다.

- **Performance Highlights**: 이 연구의 주요 응용 사례는 신경망의 활성화 함수 하이퍼파라미터 튜닝과 그래프 신경망(GNN)의 그래프 커널 파라미터 튜닝입니다. 서로 다른 활성화 함수의 조합으로부터 최적의 하이퍼파라미터를 학습하여 다중 문제 인스턴스에서 조정합니다. 또한 그래프 신경망을 위한 그래프 하이퍼파라미터 튜닝의 경우, 기존의 고전적인 모델에 대한 연구와는 달리 현대의 그래프 신경망에 대한 첫 번째 증명 가능한 보장을 제공합니다.



### A real-time battle situation intelligent awareness system based on Meta-learning & RNN (https://arxiv.org/abs/2501.13704)
- **What's New**: 이 논문에서는 전투 상황 분석에 있어 실시간(Real-time) 및 정확한 정보 제공이 중요함을 강조하고 있습니다. 제안된 실시간 전투 상황 지능 인식 시스템(BSIAS)은 메타 학습(Meta-learning) 분석과 단계적 순환 신경망(Stepwise RNN) 모델링을 통해 전투 데이터의 처리 및 분석을 수행합니다. 이 시스템은 전투 시뮬레이션을 통해 적의 움직임과 공격 경로를 예측할 수 있는 기능을 제공합니다.

- **Technical Details**: BSIAS는 데이터 정화(Data Cleansing), 데이터 융합(Data Fusion), 데이터 마이닝(Data Mining) 등의 여러 단계를 포함한 데이터를 처리하고 분석합니다. 단계적 RNN은 데이터 집합의 시간적 의존성을 포착하는 데 최적화된 방법을 사용하여 전투 모델링을 수행합니다. 이러한 기술적 접근은 전투 명령 및 분석 기술 분야에서의 응용 가능성을 보여줍니다.

- **Performance Highlights**: BSIAS는 명령관이 전시 과학적 결정을 내릴 수 있도록 지원하는 지능형 플랫폼 역할을 수행합니다. 이 시스템은 다양한 형태의 전투 데이터를 통합하여, 전투 중 발생할 수 있는 여러 시나리오에 대한 예측을 제공합니다. 따라서 BSIAS는 전투 상황을 더욱 효과적으로 관리하고 전략적인 판단을 돕는 데 기여할 것으로 기대됩니다.



### Unlearning Clients, Features and Samples in Vertical Federated Learning (https://arxiv.org/abs/2501.13683)
Comments:
          Paper accepted for publication in PETS 2025, Issue II

- **What's New**: 이 논문에서는 Federated Learning(FL)에서의 unlearning 문제를 다루고 있으며, 특히 Vertical FL(VFL)에서의 새로운 방법론을 제시합니다. VFL에서는 클라이언트들이 레이블에 접근하지 않고 샘플 공간을 나누어 사용하기 때문에 기존의 Horizontal FL(HFL)에서 제안된 접근 방식이 적용되지 않습니다. 본 연구는 VFL 내에서 클라이언트, 특징, 샘플을 unlearn하는 세 가지 접근법을 탐구하고, 각 방법론의 효과를 Membership Inference Attack(MIA) 를 활용해 입증하고자 합니다.

- **Technical Details**: VFU-KD와 VFU-GA라는 두 가지 새로운 알고리즘을 소개합니다. VFU-KD는 Knowledge Distillation(KD)에 기반하여 클라이언트와 특징을 unlearn하는 방식이며, VFU-GA는 gradient ascent에 기반하여 샘플을 unlearn합니다. 이 방법들 점검을 위해 총 6개의 표 형태 데이터셋과 2개의 이미지 데이터셋에서 실험을 진행하였으며, 실험 결과 이 두 방법이 retraining 또는 기존의 R2S 방법에 비해 우수한 성능을 보임을 확인했습니다.

- **Performance Highlights**: VFU-KD와 VFU-GA는 많은 경우에 retraining으로 얻어진 성과나 R2S 벤치마크에 비해 비슷하거나 더 나은 성과를 달성하였으며, 개선 폭은 0-2%에 이릅니다. 이 방법들은 기존의 방법들과 달리 active party와 passive party 간의 추가적인 커뮤니케이션 없이 unlearning이 가능하다는 장점을 가지고 있습니다. 다만, active party는 이전에 통신된 embedding을 저장해야 하는 의무가 있습니다.



### HumorReject: Decoupling LLM Safety from Refusal Prefix via A Little Humor (https://arxiv.org/abs/2501.13677)
- **What's New**: 이 논문에서는 기존 대형 언어 모델(LLMs)의 안전성을 재구성하는 새로운 접근법인 HumorReject를 소개합니다. 이 방법은 명시적인 거부(prefix refusal) 전략에서 벗어나 유머(humor)를 통한 간접적인 거부 전략을 사용하여 안전성을 강화합니다. 이를 통해 기존의 '과도한 방어(over-defense)' 문제를 해결하고 사용자와의 상호작용을 더욱 매력적으로 유지합니다.

- **Technical Details**: HumorReject는 유해한 지시를 직접적으로 거부하는 대신, 적절한 유머를 통해 위험한 요청을 자연스럽게 완화합니다. 이 접근법은 다양한 공격 벡터에 대해 뛰어난 강인함을 발휘하며, 합법적인 작업에서도 자연스럽고 고품질의 상호작용을 제공합니다. 연구 결과, 데이터 수준에서의 혁신이 LLM 안전성을 달성하는 데 있어 더욱 근본적임을 시사합니다.

- **Performance Highlights**: HumorReject는 기존의 안전 메커니즘과 비교했을 때 여러 공격 벡터에 대해 탁월한 견고성을 입증하였으며, 사용자가 안전하고 친숙한 AI 시스템을 경험할 수 있도록 새로운 방향을 제시합니다. 본 연구는 AI 시스템의 안전성 강화와 사용자 경험 향상에 기여할 잠재력을 보여주고 있습니다.



### Certified Robustness Under Bounded Levenshtein Distanc (https://arxiv.org/abs/2501.13676)
Comments:
          Accepted in ICLR 2025

- **What's New**: 이 논문에서는 텍스트 분류기의 강건성을 증명하기 위한 새로운 방법을 제안합니다. Lipschitz constant를 Levenshtein distance에 대한 컨볼루션 분류기에 대해 계산할 수 있는 첫 번째 방법으로, 이는 텍스트 도메인에서 효율적인 검증을 가능하게 합니다. Proposed method, LipsLev는 단일 전방 전달 통과에서 분류기의 공인 반지름을 계산할 수 있도록 합니다.

- **Technical Details**: 기존의 검증 방법들은 대개 캐릭터/단어 대체 또는 불용어 제거와 같은 사양만 처리할 수 있었으나, 본 연구에서는 평균 전송 거리(ERP distance)에 대한 Lipschitz constant를 계산함으로써 1-Lipschitz 분류기를 훈련할 수 있는 방법을 제시합니다. 이 방법은 기존의 접근 방식이 Levenshtein distance 제한에 대한 검증을 지원하지 못했던 문제를 해결합니다.

- **Performance Highlights**: LipsLev는 AG-News 데이터 세트에서 거리 1과 2에서 각각 $38.80$% 및 $13.93$%의 공인 정확도를 기록하며, 이는 기존 접근 방식보다도 4배 이상 빠른 속도를 자랑합니다. 또한 거리 2 이상에서 검증할 수 있는 유일한 방법으로, 이는 텍스트 도메인에서의 강건성 검증의 새로운 이정표가 될 것으로 기대됩니다.



### Revisiting Online Learning Approach to Inverse Linear Optimization: A Fenchel--Young Loss Perspective and Gap-Dependent Regret Analysis (https://arxiv.org/abs/2501.13648)
- **What's New**: 이번 논문은 Bärmann et al. (2017)의 역선형 최적화(inverse linear optimization)에 대한 온라인 학습(online learning) 접근방식을 재조명합니다. 이 연구의 목표는 에이전트의 입력-출력 쌍의 순차적 관찰을 통해 에이전트의 미지의 선형 목표 함수(unknown linear objective function)를 추론하는 것입니다. 우리는 Fenchel--Young 손실(Fenchel--Young losses)과의 연결을 통해 온라인 학습 접근 방식을 간단하게 이해하도록 돕습니다.

- **Technical Details**: 논문은 에이전트의 선택을 설명하는 예측된 목표의 효과를 측정하는 
suboptimality loss에 대한 오프라인 보장을 제시합니다. 또한, 에이전트의 결정 문제에서 최적값과 비최적값 간의 간격이 있는 경우, 시간 수평선(Т)에 독립적인 상한을 제공합니다. 이 상한은 비최적성(suboptimality)과 추정 손실(estimate losses)의 합을 측정하며, 후자는 예측된 목표가 추천하는 솔루션의 품질을 측정합니다.

- **Performance Highlights**: 흥미롭게도, 우리의 간격 의존 분석은 역선형 최적화에 특화된 구조를 활용하여 표준 $O(\sqrt{T})$ 후회(regret) 경계보다 더 빠른 속도를 달성합니다. 이는 손실 함수(loss functions)나 그 도메인이 강한 볼록성(strong convexity) 등의 바람직한 성질을 지니지 않음에도 불구하고 이루어졌습니다.



### Quantification via Gaussian Latent Space Representations (https://arxiv.org/abs/2501.13638)
- **What's New**: 본 논문에서는 기존의 양자화 방법이 가지고 있는 한계를 극복하기 위해 새로운 엔드투엔드 신경망 모델을 제안합니다. 이 모델은 잠재 공간에서 Gaussian 분포를 사용하여 예제 집합의 불변 표현을 얻어 양자화 문제를 다룹니다. 즉, 중간 분류기를 필요로 하지 않고 손실 함수를 최적화하여 직접적으로 양자화 문제를 해결하는 방식을 사용합니다.

- **Technical Details**: 논문에서 제안하는 새로운 레이어는 Gaussian 분포를 이용해 예제 집합을 표현하며, 기존의 pooling layer나 histogram과 같은 표현 레이어보다 더 강력한 성능을 발휘한다고 주장합니다. 이 레이어는 양자화 문제를 해결하기 위해 필요한 전체 집합의 정보로 압축된 표현을 제공합니다. 또한, 해당 레이어의 파라미터 초기화 및 정규화 기법을 논의하여 성능 개선과 수렴 시간을 향상시키는 방법을 제안합니다.

- **Performance Highlights**: 제안된 방법은 기존의 전통적인 양자화 방법과 다른 심층 학습 접근 방식과 비교하여 최신의 뛰어난 결과를 달성しました. 실험 결과, 이 새로운 레이어를 사용한 접근 방식은 두 차례의 양자화 대회에서 기록적인 성과를 냈고, 전통적인 방법에서는 도달할 수 없는 성과를 보여주는 것을 입증하였습니다.



### Optimal Multi-Objective Best Arm Identification with Fixed Confidenc (https://arxiv.org/abs/2501.13607)
Comments:
          Accepted to AISTATS 2025

- **What's New**: 이번 연구는 Multi-Objective Multi-Armed Bandit (MO-MAB) 설정의 문제를 정식으로 조사하여 각 목표(objective)의 최적 팔(best arm) 식별 문제를 다루고 있습니다. 기존 연구는 주로 Pareto frontier 식별에 집중했으나, 본 연구는 복수 목표의 최적 팔을 찾는 비즈니스 및 실용적 요구를 충족합니다. 이러한 최적 팔의 식별 문제는 다차원 보상 구조와 관련된 고유한 불확실성으로 인해 복잡해집니다.

- **Technical Details**: 연구에서는 각 팔이 여러 차원(M-dimensional) 벡터 보상을 제공하는 설정을 고려하고, 각 차원 보상은 독립적으로 생성된다고 가정합니다. 최적 팔의 식별은 오류 확률이 주어진 한도 내에서 달성되어야 하며, 최적의 정지 시간(expected stopping time)을 줄이는 것이 목표입니다. 저자들은 최대 최소 최적화 문제(max-min optimization problem)를 해결하기 위한 새로운 알고리즘을 제안하였고, 이 알고리즘은 대체 비율(surrogate proportions)을 활용하여 각 단계에서의 계산 복잡성을 줄입니다.

- **Performance Highlights**: 저자들은 제안된 알고리즘이 점근적 최적(asymptotically optimal)임을 이론적으로 증명하고, 광범위한 경험적 연구를 통해 알고리즘의 효율성을 입증했습니다. 기존 문헌이 주로 Pareto 최적 팔 식별에 초점을 맞추고 있는 반면, 본 연구는 각 목표에 대해 최적 팔을 식별하는 과제에 대해 명확한 접근 방식을 제공합니다. 이를 통해 다목적 밴딧 연구의 최신 기술을 한 단계 끌어올리는 성과를 거두었습니다.



### FedPref: Federated Learning Across Heterogeneous Multi-objective Preferences (https://arxiv.org/abs/2501.13604)
Comments:
          Accepted to ACM ToMPECS journal

- **What's New**: 이번 연구에서는 다중 목표에 따른 선호 이질성(preference heterogeneity)을 논의하며, 이를 해결하기 위한 새로운 알고리즘 FedPref를 제안합니다. 기존의 Federated Learning(FL)은 주로 데이터의 이질성(data heterogeneity)과 시스템 이질성(system heterogeneity)에 초점을 맞추었으나, 선호 이질성은 현재와 미래의 응용에서 매우 중요하게 다뤄질 필요가 있습니다. 또한, 본 논문에서는 FL 알고리즘의 성능을 평가하기 위한 다중 목표 지표(multi-objective metrics)를 도입하여 기존의 클라이언트 중심 지표를 넘어서는 새로운 분석 관점을 제시합니다.

- **Technical Details**: FedPref 알고리즘은 개인화된 Federated Learning(PFL)의 접근 방식을 채택하여 각 클라이언트가 자신의 필요에 맞춘 개별 모델을 학습합니다. 이 알고리즘은 유사한 모델을 갖고 있는 클라이언트들을 그룹화하여 집합적으로 특징을 활용하고, 비호환적인 모델을 가진 클라이언트는 별도로 처리합니다. 알고리즘은 클라이언트의 선호에 대한 어떠한 사전 지식 없이도 작동하며, 수정된 코사인 유사도(cosine similarity) 메트릭을 기반으로 모델을 집계합니다.

- **Performance Highlights**: 실험 결과 FedPref는 여러 문제, 선호 분포, 모델 아키텍처에서 뛰어난 성능을 발휘하는 것으로 나타났습니다. 비교 실험을 통해 기존의 데이터 이질성에 맞춘 알고리즘들이 선호 이질성이 있는 환경에서 성능 저하 없이 이전하기 어렵다는 점을 확인했습니다. 또한, 다양한 매개변수 선택이 알고리즘의 성능에 미치는 영향을 깊이 있게 분석하여 FedPref의 효과를 입증하였습니다.



### A Transformer-based Autoregressive Decoder Architecture for Hierarchical Text Classification (https://arxiv.org/abs/2501.13598)
Comments:
          7 pages + 1 for references. 2 Figure. ECAI conference

- **What's New**: 최근의 계층적 텍스트 분류(Hierarchical Text Classification, HTC) 방법들은 사전 훈련된 변환기 모델(transformer model)의 능력에 의존하고 라벨의 의미(semantics) 및 그래프 인코더(graph encoder)를 활용하여 라벨 계층을 처리하고 있습니다. 본 논문은 오프더쉘프(pre-trained) RoBERTa 변환기를 기반으로 하여 두 개의 디코더 레이어를 사용하는 사용자 정의 자기 회귀 디코더를 도입한 RADAr라는 효율적인 계층적 텍스트 분류기를 제안합니다. RADAr는 레이블 계층에 대한 명시적인 인코딩 없이도, 훈련 중 관찰된 레이블 시퀀스만으로 분류 출력을 생성할 수 있습니다. 


- **Technical Details**: 계층적 텍스트 분류에서 일반적으로 사용되는 기존 방법들과 달리 RADAr는 그래프 인코더를 필요로 하지 않고 레이블 의미를 사용하지 않습니다. 이 모델은 RoBERTa를 인코더로 활용하고, 두 개의 디코더 레이어가 있는 맞춤형 자기 회귀 디코더를 사용합니다. 따라서, 레이블을 부모에서 자녀로 정렬하는 대신, 자녀에서 부모로 정렬하여 더 나은 결과를 도출할 수 있음을 보여줍니다. 


- **Performance Highlights**: RADAr 모델은 세 개의 벤치마크 데이터셋에서 실험을 통해 기존의 최신 기술들과 경쟁할 수 있는 결과를 내며, 훈련 및 추론 시간이 크게 줄어드는 장점을 보입니다. 특히, RADAr는 추론 시 속도를 두 배로 향상시키고 작고 가벼운 모델로 다루기 쉽습니다. 이로 인해 미래의 연구자들은 새로운 모델이 등장할 때 인코더 부분을 손쉽게 교체할 수 있는 유연성을 확보할 수 있습니다.



### A Comprehensive Survey on Spectral Clustering with Graph Structure Learnin (https://arxiv.org/abs/2501.13597)
- **What's New**: 본 논문은 스펙트럼 클러스터링(spectral clustering) 기법의 최신 발전을 다루며, 그래프 구조 학습(graph structure learning, GSL)이 스펙트럼 클러스터링의 성능 개선에 있어 필수적임을 강조합니다. 이전의 조사 연구와는 달리, GSL의 역할에 중점을 두고 다양한 그래프 구축 기법을 살펴봅니다. 또한, 단일 뷰(single-view)와 다중 뷰(multi-view) 프레임워크로 스펙트럼 클러스터링 방법을 분류하고, 데이터 클러스터링에서 다중 뷰 정보 융합의 중요성을 논의합니다.

- **Technical Details**: 스펙트럼 클러스터링은 고차원 데이터를 클러스터링하기 위해 그래프 기반 표현을 활용하여 복잡한 비선형 구조를 식별하는 강력한 기법입니다. 이 과정의 핵심은 유사성 그래프(similarity graph)의 구축이며, 이는 데이터 포인트 간의 관계를 정의합니다. 스펙트럼 클러스터링은 특히 고차원 데이터에서 효과적이며, 그래프 구조 학습을 통해 신뢰할 수 있는 클러스터링 결과를 도출할 수 있습니다.

- **Performance Highlights**: 스펙트럼 클러스터링은 클러스터의 복잡한 비선형 패턴을 효과적으로 처리할 수 있는 능력을 가지고 있습니다. 데이터 포인트를 그래프의 노드로 나타내고, 그들의 유사성을 엣지로 표시함으로써 연결성과 관계에 기반한 클러스터링이 가능합니다. 최근 연구에 따르면, 앵커 그래프(anchor graph)를 이용한 스펙트럼 클러스터링은 대규모 데이터 세트에서도 높은 품질의 클러스터링 결과를 유지하면서 계산 자원을 효율적으로 사용할 수 있는 솔루션을 제공합니다.



### WFCRL: A Multi-Agent Reinforcement Learning Benchmark for Wind Farm Contro (https://arxiv.org/abs/2501.13592)
- **What's New**: WFCRL (Wind Farm Control with Reinforcement Learning)은 풍력 발전소 제어 문제를 해결하기 위해 개발된 최초의 오픈 소스 다중 에이전트 강화 학습 환경 모음입니다. 이 프레임워크는 각 풍력 터빈을 에이전트로 간주하고 협력하여 최적의 전력 생산을 목표로 합니다. WFCRL은 두 가지 최신 농장 시뮬레이터인 FLORIS와 FAST.Farm에 인터페이스를 제공하여 연구자들이 커스터마이즈할 수 있습니다.

- **Technical Details**: WFCRL은 각 풍력 터빈의 요, 피치 및 토크를 조정하여 풍력 농장의 성능을 극대화할 수 있도록 합니다. 다중 에이전트 강화 학습(MARL) 프레임워크를 기반으로 하여 각 터빈이 지방 관측치를 수신하고 이를 통해 공동 목표를 달성하도록 설계되었습니다. WFCRL은 또한 각 환경에서 사용할 수 있는 몇 가지 사전 정의된 풍력 농장 레이아웃과 바람 조건 시나리오를 포함합니다.

- **Performance Highlights**: WFCRL에서는 IPPO, MAPPO, QMIX 등 세 가지 최신 MARL 알고리즘을 구현하여 시뮬레이터에서의 확장성 문제를 강조합니다. 초기 테스트는 풍력 관광으로 인한 피로 비용을 고려하여 총 전력 생산 극대화를 목표로 하고 있습니다. WFCRL은 정적 및 동적 시뮬레이터 간의 전이 학습 전략을 설계할 수 있는 가능성을 제공하며, 이는 산업에서 응용되는 정책의 안전성과 강건성을 평가하는 데 중요한 역할을 합니다.



### Contrastive Representation Learning Helps Cross-institutional Knowledge Transfer: A Study in Pediatric Ventilation Managemen (https://arxiv.org/abs/2501.13587)
- **What's New**: 이번 연구에서는 기관 간 지식 전이를 위한 체계적인 프레임워크를 제시하며, 일반 소아 집중 치료실(PICU)과 심장 전문 병동 간의 소아 환기 관리를 예시로 보여줍니다. 대조적 예측 부호화(Contrastive Predictive Coding, CPC)를 이용한 표현 학습을 통해 서로 다른 데이터 조건 및 미세 조정 전략이 지식 전이에 미치는 영향을 규명하고 있습니다. 이 연구 결과, 직접적인 모델 이전이 어려운 반면, 적절한 미세 조정을 통해 기관 간 효과적인 지식 공유가 가능하다는 사실이 입증되었습니다.

- **Technical Details**: 이 연구는 두 개의 PICU에서 수집된 환기 데이터를 분석하며, 기관 간의 차이로 인한 자연 장벽에서 모델 일반화를 이끌어내기 위한 방법론을 제시합니다. 정식적으로, 소스 도메인(dataset)에서의 representation learning과 타겟 도메인에서의 모델 적응(fine-tuning)을 통해 지식 전이를 구현합니다. 세 가지 예측 모델 학습 접근 방식을 고려하며, 특히 CPC를 활용한 자기 감독 학습이 임상적 추론에 어떻게 기여하는지를 탐구합니다.

- **Performance Highlights**: 연구의 결과, 기관 간 성능 갭이 존재하지만, 대조적 미리 학습(contrastive pre-training)과 적절한 미세 조정 전략이 이러한 갭을 상당히 줄여줄 수 있음을 보여줍니다. 특히 데이터가 제한된 상황에서 이러한 전략의 효과가 두드러지며, 다양한 임상 과제에서 비대칭적인 전이 패턴이 나타남을 분석했습니다. 이 발견은 임상 결정 지원 시스템의 개발과 함께, 비교적 작은 전문 기관들이 대형 센터의 지식을 활용할 수 있는 실용적인 경로를 제시합니다.



### Towards Robust Incremental Learning under Ambiguous Supervision (https://arxiv.org/abs/2501.13584)
- **What's New**: 본 논문에서는 기존의 Incremental Learning (IL) 방식의 한계를 극복하기 위해 Incremental Partial Label Learning (IPLL)이라는 새로운 약한 감독 학습 프레임워크를 제안합니다. IPLL은 동적으로 도착하는 데이터에 후보 레이블 집합을 연결하여 새로운 클래스와 이전 클래스를 동시에 처리합니다. 이를 통해 실제 데이터에서 발생할 수 있는 레이블 불확실성을 효과적으로 관리할 수 있습니다.

- **Technical Details**: 논문에서는 Prototype-Guided Disambiguation and Replay Algorithm (PGDR)을 개발하여 IPLL의 복잡한 문제를 해결했습니다. PGDR는 카테고리 정보를 포함하는 클래스 프로토타입을 활용하여 레이블의 모호성을 완화하고, 기억재생 기법을 통해 지식의 소멸(catastrophic forgetting)을 줄입니다. 이 알고리즘은 레이블의 모호함을 극복하는 데 필요한 두 가지 모듈인 주의 기반의 레이블 분산(disambiguation)과 다양한 샘플을 기억하는 메모리 재생을 결합합니다.

- **Performance Highlights**: PGDR은 CIFAR100과 Tiny-ImageNet 데이터셋에서 각각 6.05% 및 13.24%의 성능 향상을 보여주며 최첨단 결과를 기록했습니다. 이는 기존의 IL 방법과 PLL 방법을 효과적으로 결합하여 이루어진 결과로, 새로운 클래스의 정확한 훈련과 기존 클래스의 기억 유지를 동시에 달성했음을 의미합니다.



### Communication-Efficient Stochastic Distributed Learning (https://arxiv.org/abs/2501.13516)
- **What's New**: 이 논문에서는 비볼록(nonconvex) 및 볼록(convex) 비아우라기(undirected networks)에서의 분산 학습 문제를 다룹니다. 특히, 고비용의 통신 문제와 대규모 데이터를 처리할 수 있는 새로운 알고리즘을 설계했습니다. 이 알고리즘은 각 통신 라운드 사이에 여러 로컬 훈련 단계를 수행하고, 랜덤 데이터 서브셋을 사용하여 로컬 계산을 수행하면서 확률적 경량화 가능성을 포괄합니다.

- **Technical Details**: 제안된 두 가지 알고리즘(Local Training ADMM 및 Variance Reduction 적용 ADMM)은 분산 Alternating Direction Method of Multipliers (ADMM)에 기반하고 있습니다. local training을 통해 커뮤니케이션 비용을 줄이고, 각 에이전트는 대규모 데이터셋에서 전체 기울기를 계산하지 않고도 로컬 기울기를 추정할 수 있습니다. 이로 인해 제안된 알고리즘들은 비볼록 문제에서는 정적 점의 이웃으로, 볼록 문제에서는 최적 점의 이웃으로 수렴합니다.

- **Performance Highlights**: 제안된 알고리즘들은 최신 기술들과 비교하여 성능이 우수함을 수치적으로 입증하였습니다. 특히, 커뮤니케이션 비용이 높은 대규모 모델 훈련 시, LT-ADMM과 LT-ADMM-VR이 대체 방법들보다 월등한 성능을 보여주었습니다. 수렴 속도도 향상되었으며, 특히 LT-ADMM-VR은 비볼록 문제에 대해 정밀한 수렴을 이루어냈습니다.



### GCAD: Anomaly Detection in Multivariate Time Series from the Perspective of Granger Causality (https://arxiv.org/abs/2501.13493)
Comments:
          Accepted to AAAI 2025

- **What's New**: 이번 논문은 다변수 시계열 이상 감지를 위한 새로운 프레임워크를 제안합니다. 기존의 예측 및 재구성 과제를 기반으로 한 방법들과 달리, 우리는 인과적 관계를 해석 가능하게 모델링함으로써 이상 감지를 수행합니다. 특히, 비선형 딥 예측기에서 기울기를 활용하여 변화된 Granger causality를 동적으로 발견하고, 이를 통해 인과적 관점에서의 이상 감지 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 Granger causality를 기반으로 하며, 비선형 딥 모델의 기울기를 사용하여 동적으로 인과적 종속성을 탐색합니다. 데이터를 통해 배운 원래 패턴에서의 이탈을 기반으로 이상 점수를 계산하여, 최종적으로 다섯 개의 실제 벤치마크 데이터 세트에서 우수한 성능을 입증하고자 하였습니다. 이와 함께, 대칭 기반 희소화 방법을 통해 그래프의 양방향 엣지를 제거하고 인과그래프를 구성하여 더 해석 가능한 결과를 도출하였습니다.

- **Performance Highlights**: 실험 결과, 제안하는 모델은 기존의 기준 방법들보다 훨씬 높은 정확도의 이상 감지를 달성했습니다. 특히, 다섯 개의 실제 벤치마크 데이터 세트에서 state-of-the-art 결과를 보여주었으며, 기울기 기반의 Granger causality 효과 계량화 방법이 모델의 성능 향상에 크게 기여하였습니다. 이 연구는 딥 모델과 Granger causality의 융합을 통한 시계열 이상 감지의 새로운 가능성을 열었습니다.



### MambaQuant: Quantizing the Mamba Family with Variance Aligned Rotation Methods (https://arxiv.org/abs/2501.13484)
- **What's New**: MambaQuant는 Mamba 모델을 위한 최초의 포괄적인 후처리 양자화(PTQ) 프레임워크로, 학습 후 양자화의 효과를 극대화하는 기술을 제시합니다. Mamba 모델의 양자화 과정에서 발생하는 주요 문제로는 큰 이상치(outliers)와 변동성(variance) 불일치가 있습니다. 이러한 문제에 대응하기 위해 MambaQuant는 Karhunen-Loeve 변환(KLT)과 스무스-퓨즈드 회전(rotation) 기법을 도입하여 딥러닝 모델이 손실 없이 양자화될 수 있도록 지원합니다.

- **Technical Details**: MambaQuant는 두 가지 주요 기술을 적용합니다. 첫 번째는 오프라인 모드에서 KLT를 활용한 회전으로, 다양한 채널 분포에 맞춘 회전 행렬(rotation matrix)을 생성합니다. 두 번째는 온라인 모드에서 스무스-퓨즈드 회전을 구현하여 채널 분산을 정규화하여 메모리 비용을 최소화합니다. 이 두 가지 접근법은 양자화 데이터의 최댓값과 분산을 일관되게 유지하여 성능을 향상시킵니다.

- **Performance Highlights**: MambaQuant는 Mamba 기반의 비전 및 언어 작업에서 8비트 양자화 시 1% 미만의 정확도 손실로 우수한 성능을 발휘합니다. 또한, 비전 작업에서는 4비트로 양자화하면서도 1%의 최소 정확도 손실로 뛰어난 결과를 나타냅니다. 기존 방법들과 비교해 MambaQuant는 언어 작업에서도 유의미한 정확도 향상을 보이며, 향후 연구의 기초가 될 것으로 기대됩니다.



### Adaptive Few-Shot Learning (AFSL): Tackling Data Scarcity with Stability, Robustness, and Versatility (https://arxiv.org/abs/2501.13479)
- **What's New**: 이 논문에서는 기존의 Few-Shot Learning (FSL)의 한계를 극복하기 위해 Adaptive Few-Shot Learning (AFSL) 프레임워크를 제안합니다. AFSL은 메타 학습, 도메인 정렬, 노이즈 저항력, 다중 모달 통합을 통합한 모듈형 아키텍처로, 데이터가 부족한 분야에서도 효과적으로 일반화할 수 있는 모형을 구축합니다. 이 프레임워크는 성능 일관성을 보장하기 위한 Dynamic Stability Module, 도메인 적응을 위한 Contextual Domain Alignment Module, 노이즈 데이터를 처리하기 위한 Noise-Adaptive Resilience Module, 다양한 모달을 통합하는 Multi-Modal Fusion Module로 구성되어 있습니다.

- **Technical Details**: AFSL은 네 가지 주요 모듈로 구성되며, 각 모듈은 특정 병목 문제를 해결합니다. Dynamic Stability Module은 앙상블 기반 메타 학습을 활용하여 예측의 일관성을 높입니다. Contextual Domain Alignment Module은 원천 도메인과 타겟 도메인 간의 특징을 정렬하여 도메인 간 이동을 용이하게 합니다. Noise-Adaptive Resilience Module은 노이즈가 많은 데이터를 처리하기 위해 주의를 기반으로 한 가중치와 노이즈 인식 손실 함수를 이용해 강인성을 향상시킵니다.

- **Performance Highlights**: AFSL은 성능, 안정성 및 강인성에서 기존 FSL 방법론에 비해 상당한 개선을 보입니다. 이 모듈형 접근 방식은 헬스케어, 로보틱스, 자연어 처리와 같은 중요한 분야에서 보다 신뢰할 수 있는 솔루션을 제공합니다. AFSL의 도입으로, 다양한 상황에 적응할 수 있는 모델의 능력이 강화되어 실제 어플리케이션의 확장성과 신뢰성을 높일 수 있습니다.



### KAA: Kolmogorov-Arnold Attention for Enhancing Attentive Graph Neural Networks (https://arxiv.org/abs/2501.13456)
- **What's New**: 이 논문에서는 기존 GNN의 스코어링 함수에 대한 이해 부족 문제를 해결하기 위해, Kolmogorov-Arnold Attention (KAA)라는 새로운 스코어링 방법을 제안합니다. KAA는 Kolmogorov-Arnold Network (KAN) 아키텍처를 통합하여, 기존의 모든 attentive GNNs에 적용 가능하며, 성능을 20% 이상 향상시킬 수 있는 가능성이 있음을 입증합니다.

- **Technical Details**: KAA는 기존 GNN 모델의 스코어링 함수를 통합하는 새로운 프레임워크를 제시하고, Maximum Ranking Distance (MRD)를 도입하여 노드 중요도를 순위화하는 오류의 한계를 정량적으로 측정합니다. KAA는 단일 레이어 KAN을 활용해 높은 표현력을 제공하며, 기존의 선형 변환 기반 및 MLP 기반 스코어링 함수에 비해 유의미한 성능 향상을 보여줍니다.

- **Performance Highlights**: 실험 결과, KAA로 개선된 스코어링 함수는 다양한 백본 모델 및 과제에서 일관되게 기존의 스코어링 함수보다 뛰어난 성능을 보였으며, 일부 경우에는 20% 이상의 성능 향상이 있었습니다. 이러한 성과는 KAA가 노드 레벨 및 그래프 레벨 작업 모두에서 강화된 효과를 증명함으로써, 기존 모델의 한계를 극복하는 데 기여함을 보여줍니다.



### Spurious Forgetting in Continual Learning of Language Models (https://arxiv.org/abs/2501.13453)
Comments:
          ICLR2025

- **What's New**: 최근 대형 언어 모델(LLMs)의 발전과 관련하여 지속적인 학습에서 나타나는 흥미로운 현상, 즉 학습이 충분히 이루어졌음에도 불구하고 성능 저하가 발생하는 문제를 다루고 있습니다. 이 연구는 'spurious forgetting'의 개념을 탐구하며, 이러한 성능 저하가 실제 지식 상실이 아니라 작업 정렬(task alignment)의 감소를 반영한다고 제안합니다.

- **Technical Details**: 실험을 통해 새로운 작업의 초기 훈련 단계에서 모델 성능의 동적 변화를 조사하였습니다. 초기 최적화 단계가 이전에 설정된 작업 정렬을 방해할 수 있음을 발견하였으며, 이러한 변화는 모델 가중치의 직교 업데이트(orthogonal updates)와 연결됩니다. 이러한 분석을 통해 우리는 이 행동을 이해하기 위한 강력한 프레임워크를 제공합니다.

- **Performance Highlights**: 본 연구는 모델의 하위 계층을 고정하는 Freezing 전략을 도입하여 지속적인 학습 시나리오 네 가지에서 상당한 성능 개선을 이루었습니다. 이 연구 결과는 작업 정렬과 지식 유지의 중요한 구분을 강조하며, 지속적인 학습에서 보다 효과적인 전략을 개발하는 데 기여할 수 있는 가능성을 제시합니다.



### Deep Modularity Networks with Diversity--Preserving Regularization (https://arxiv.org/abs/2501.13451)
Comments:
          Preprint

- **What's New**: 이 논문에서는 기존의 Deep Modularity Networks (DMoN)의 한계를 극복하기 위해 Diversity-Preserving Regularization (DMoN-DPR)을 제안합니다. 이 새로운 접근법은 클러스터 간 분리를 위한 거리 기반, 클러스터 내 다양성을 위한 분산 기반, 균형 잡힌 할당을 위한 엔트로피 기반의 세 가지 새로운 정규화 항을 추가합니다. 이를 통해 피처 공간의 다양성을 보장하면서 clustering 성능을 향상시킵니다.

- **Technical Details**: DMoN-DPR는 클러스터 할당을 소프트맥스 함수에 기반하여 나타내고, Graph Convolutional Network (GCN)를 통해 클러스터 할당을 최적화합니다. 이 방법은 서로 다른 클러스터의 중심점을 분리시키고, 각 클러스터 내 피처의 분산을 최적화하며, 클러스터 할당의 엔트로피를 최대화하여 균형 잡힌 노드 분포를 보장합니다. 이를 통해 얻어진 클러스터는 구조적으로 응집력이 있으며 기능적으로 뚜렷한 특징을 가집니다.

- **Performance Highlights**: DMoN-DPR은 Cora, CiteSeer, PubMed, Coauthor CS, Coauthor Physics와 같은 벤치마크 데이터셋에서 clustering 성능을 향상시키며, NMI와 F1 스코어에서 상당한 개선을 이루었습니다. 특히, Coauthor CS 및 Coauthor Physics 데이터셋에서는 기능 공간이 풍부하여 이 새로운 접근법의 이점을 크게 누릴 수 있습니다. 실험 결과는 다양성 보존 정규화의 도입이 의미 있고 해석 가능한 클러스터 생성을 가능하게 한다는 것을 보여줍니다.



### Wasserstein-regularized Conformal Prediction under General Distribution Shif (https://arxiv.org/abs/2501.13430)
- **What's New**: 이 논문에서는 i.i.d. 가정 하에서 발생할 수 있는 커버리지 격차(coverage gap)에 대한 Wasserstein distance를 기반으로 한 상한을 제안합니다. 기존 연구들은 전체 변동 거리(total variation distance)를 사용하여 커버리지 격차를 제한했으나, 특히 분포 변화가 있는 경우 실제 격차 변화를 식별할 수 없는 한계가 있었습니다. 따라서 본 연구는 공동 분포의 변화를 고려하여 새로운 알고리즘을 설계하는 데 중점을 두고 있습니다.

- **Technical Details**: 이 연구는 커버리지 격차를 감소시키기 위한 알고리즘을 설계하기 위해 중요도 가중치(importance weighting) 및 정규화된 표현 학습(regularized representation learning) 방법을 활용합니다. 제안된 WR-CP 알고리즘은 Wasserstein bound를 최소화하면서 커버리지 격차를 조절 가능한 균형을 유지합니다. 이 알고리즘은 다시 계산된 점수 분포에 따른 예측 세트를 생성하여, 효율성과 예측 정확성을 동시에 개선하려고 합니다.

- **Performance Highlights**: 실험 결과, WR-CP는 다양한 신뢰 수준에서 커버리지 격차를 평균 3.1%로 감소시키며, 최악의 접근법에 비해 약 38% 더 작은 예측 세트를 생성합니다. 이러한 성과는 특히 금융, 의료 및 자율 주행과 같은 분야에서 예측 정확도를 높이는 데 기여할 것으로 기대됩니다. 전반적으로 WR-CP는 기존 방법보다 효율적이며 신뢰할 수 있는 예측을 제공하는 것으로 나타났습니다.



### M3PT: A Transformer for Multimodal, Multi-Party Social Signal Prediction with Person-aware Blockwise Attention (https://arxiv.org/abs/2501.13416)
- **What's New**: 이 연구는 M3PT (Multi-Modal, Multi-Party Transformer)라는 causal transformer 아키텍처를 소개하며, 이는 복수의 사회적 신호를 동시에 처리할 수 있게 설계되었습니다. 특히, modality-specific와 temporal attention masking 기능을 통해 여러 참여자의 신호를 효율적으로 예측할 수 있는 점에서 혁신적입니다. 이 접근 방식은 개인 간 사회적 신호의 시간적 상호작용을 포착하여 보다 동적인 사회적 역학을 이해하는 데 기여합니다.

- **Technical Details**: M3PT 모델은 사회적 신호 예측을 위한 블록 단위( blockwise )의 attention masking 기술을 사용하며, 이는 다양한 모달리티의 입력을 처리하고 이전의 상호작용 이력을 고려합니다. 각 모달리티 입력은 벡터 양자화 기반의 자동 인코더를 통해 토큰화되어 처리됩니다. 연구는 Human-Human Commensality Dataset (HHCD)에서 다양한 사회적 신호를 예측하는 방식을 검증하였고, 여기서 식사 상황에서의 상호작용이 주요 사례로 사용되었습니다.

- **Performance Highlights**: M3PT는 다중 모달리티를 활용하여 bite timing(음식 물기 시점)과 speaking status(발언 상태)의 예측 성능을 향상시키는 데 성공했습니다. 연구 결과는 여러 모달리티를 포함하는 것이 사회적 신호를 예측하는 데 중요하다는 것을 보여주며, 실험을 통해 더 긴 시간적 맥락과 시간 단위(chunking)의 역할이 사회적 신호 예측에 기여함을 입증했습니다.



### Load and Renewable Energy Forecasting Using Deep Learning for Grid Stability (https://arxiv.org/abs/2501.13412)
- **What's New**: 본 논문은 에너지원의 통합과 관련하여 그리드 운영자들이 직면한 최신 도전 과제를 다룹니다. 특히, 태양광 및 풍력 에너지가 불규칙함에 따라 공급과 수요의 균형을 맞추는 것이 가장 큰 문제로 떠오릅니다. 이를 해결하기 위해 신뢰할 수 있는 단기 예측 기법이 필요하며, 최근에는 기계 학습 및 깊은 학습 방식이 주목받고 있습니다.

- **Technical Details**: 본 연구에서는 부하 및 재생 가능 에너지 예측을 위한 CNN(Convolutional Neural Network) 및 LSTM(Long Short-Term Memory) 기반 방법에 중점을 둡니다. 기존의 물리적 모델과 통계적 기법 대신, 이러한 기계 학습 방식이 에너지 예측에서 유망한 결과를 보이고 있습니다. CNN은 이미지 처리에 강점을 가지며, LSTM은 시간 순서 데이터를 효과적으로 처리하는데 유리하게 설계되었습니다.

- **Performance Highlights**: 딥 러닝 기술인 CNN과 LSTM은 재생 가능한 에너지 예측의 정확도를 높이는 데 큰 역할을 하고 있습니다. 특히, 불확실한 환경에서 신뢰할 수 있는 예측을 제공함으로써 에너지 저장을 극대화하고, 재생 가능 자원의 효과적인 사용을 보장하는 데 기여하고 있습니다. 이러한 접근 방식은 에너지 그리드의 안정성을 강화하는 데 필수적입니다.



### Concurrent Learning with Aggregated States via Randomized Least Squares Value Iteration (https://arxiv.org/abs/2501.13394)
- **What's New**: 오늘날의 강화학습(RL) 연구는 복잡한 환경에서 효율적으로 탐색하는 학습 에이전트를 설계하는 데 주안점을 두고 있습니다. 본 논문은 동시적으로 작동하는 여러 에이전트가 환경을 탐색하는 데 랜덤화(Randomization) 기법이 도움이 된다는 이론적 결과를 제시합니다. 이를 통해 랜덤화된 최소 제곱 값 반복(Randomized Least-Squares Value Iteration, RLSVI) 알고리즘의 동시 학습(framework)을 조정하여 공간 복잡성을 크게 줄이는 방법을 설명하고 있습니다.

- **Technical Details**: 연구에서 제안된 RLSVI는 에이전트의 이전 궤적(past trajectories)에 가우시안 노이즈를 주입하여 랜덤화된 값 함수를 학습합니다. 이 프레임워크는 동시 학습을 통해 N개의 에이전트가 상호작용을 공유하고 최적 정책에 대한 성능을 향상시키는 방식을 탐구했습니다. 또한, 유한 및 무한한 결정 지평선에서 폴리노미얼 차원의 최악의 실수값 경계에 대한 이론적 근거를 수립하였습니다.

- **Performance Highlights**: RLSVI 알고리즘의 경우, 유한한 및 무한한 지평선 환경에서 개별 에이전트에 대한 회귀(regret)는 최적 속도로 $	heta(1/	ext{sqrt{N}})$ 감소하여 동시 학습의 이점을 강조합니다. 기존의 알고리즘들에 비해 공간 복잡성을 K의 배수로 줄이면서도 최악의 회귀값 경계를 $	ext{sqrt{K}}$만큼 증가시키는 결과를 보여주었습니다. 이론적 발견을 실증하기 위해 수행한 수치 실험에서도 긍정적인 결과가 나타났습니다.



### Time Series Embedding Methods for Classification Tasks: A Review (https://arxiv.org/abs/2501.13392)
- **What's New**: 이번 논문은 머신 러닝과 딥 러닝 모델에서 효과적인 시간 시계열 임베딩 방법에 대한 포괄적인 리뷰 및 평가를 제공합니다. 다양한 실세계 데이터셋에 대한 성능을 정량적으로 평가하여 각 카테고리의 대표적인 임베딩 방법을 비교하였고, 이는 이전 연구들이 하지 못한 점입니다. 연구자들은 이러한 임베딩 방법의 성능이 데이터셋과 분류 알고리즘에 따라 크게 변동함을 강조하며, 모델 선택과 실험의 중요성을 주장합니다.

- **Technical Details**: 시간 시계열 임베딩 방법은 크게 통계적 방법, 변환 기반 방법, 특징 기반 방법, 모델 기반 방법 등으로 나뉘며, 각기 다른 접근 방식과 강점을 가지고 있습니다. 변환 기반 방법으로는 푸리에 변환 및 웨이블릿 변환이 있으며, 이는 시간 영역에서 보이지 않는 패턴을 드러내는 데 유용합니다. 또한, 특징 기반 방법은 자동 및 수동으로 중요한 특성을 추출하며, 최신 기술인 TSFRESH와 catch22 등이 다양한 특성을 분석하는 데 도움을 줍니다.

- **Performance Highlights**: 실험 결과, 다양한 임베딩 방법의 표현 능력은 데이터셋과 분류 알고리즘에 따라 크게 달라지며, 최적의 임베딩 및 분류 알고리즘 조합을 찾기 위해 광범위한 실험이 필요합니다. 연구자들은 개방형 소스 코드 저장소를 제공하여 커뮤니티가 이러한 임베딩 방법을 활용할 수 있도록 지원하며, 이는 시간 시계열 분석 분야의 발전에 기여할 것입니다.



### Beyond Task Diversity: Provable Representation Transfer for Sequential Multi-Task Linear Bandits (https://arxiv.org/abs/2501.13390)
Comments:
          38th Conference on Neural Information Processing Systems (NeurIPS 2024)

- **What's New**: 이 연구는 선형 밴딧 문제에서 다양한 작업(task) 간의 저차원 구조를 활용하여 학습 성능을 극대화하는 알고리즘을 제안합니다. 특히, 기존의 연구가 보편적으로 다양한 작업들이 있다고 가정하는 것과 달리, 이 연구에서는 그러한 가정 없이도 성능을 보장하는 알고리즘에 대해 다룹니다. 이를 통해 실제 애플리케이션에서의 저차원 구조를 효과적으로 식별하고 사용 가능하다는 사실을 보여줍니다.

- **Technical Details**: 제안된 알고리즘은 총 N개의 선형 밴딧 작업이 주어지고, 각 작업이 τ번의 반복으로 진행됩니다. 이 알고리즘은 각 작업의 파라미터가 d차원 공간의 저차원 m차원 부분공간에 존재한다고 가정하며, 이를 바탕으로 총 바이아스의 보장을 제공합니다. 결과적으로 메타 레그렛(meta-regret)의 성능은 여러 변수를 감안했을 때 O~(N m τ + N^{2/3} τ^{2/3} d m^{1/3} + N d^2 + τ m d)로 나타납니다.

- **Performance Highlights**: 다양한 합성(synthetic) 데이터에 대한 실험을 통해 제안된 알고리즘은 기존의 기준 알고리즘보다 뛰어난 성능을 발휘한다는 것을 입증하였습니다. 특히, 작업의 다양성 가정에 의존하지 않고도 메타 레그렛을 낮추어 최적 성능을 달성할 수 있음을 보여줍니다. 이 연구는 차세대 추천 시스템 및 지속적인 학습 환경에 대한 활용 가능성을 높이는 중요한 결과를 제공합니다.



### Fast and Provable Tensor-Train Format Tensor Completion via Precondtioned Riemannian Gradient Descen (https://arxiv.org/abs/2501.13385)
- **What's New**: 저자는 저랭크 텐서 완성을 해결하기 위해 프리컨디셔닝 리만 경량 하강법(Preconditioned Riemannian Gradient Descent, PRGD) 알고리즘을 제안하고 선형 수렴 속성을 확립했습니다. PRGD 알고리즘은 기존의 알고리즘에 비해 계산 시간을 두 배 줄이는 성과를 보였습니다. 따라서 이 알고리즘은 하이퍼스펙트럼 이미지 완성 및 양자 상태 톰그래피와 같은 실제 응용 분야에서도 높은 효율을 보여줍니다.

- **Technical Details**: PRGD 알고리즘은 고정된 TT랭크를 가진 텐서 집합에 대한 적합한 측정 기준을 선택함으로써 RGD 알고리즘의 성능을 향상시킵니다. 새로운 데이터 주도형 척도가 탄젠트 공간에서 정의되며, 이는 PRGD 알고리즘의 성능을 비약적으로 향상시킵니다. 논문에서는 PRGD가 동일한 초기 조건에서 목표 텐서로 선형 수렴할 수 있는 이론적 보장을 제공합니다.

- **Performance Highlights**: 실험 결과, PRGD 알고리즘은 시뮬레이션된 데이터셋과 실제 데이터셋 모두에서 RGD 성능을 두 배로 향상시키는 것으로 확인되었습니다. 특히 하이퍼스펙트럼 이미지 및 양자 상태 복원 문제에서 PRGD는 RGD보다 수십배 더 빠른 성능을 발휘했습니다. 이러한 성능적 성과는 PRGD 알고리즘이 텐서 완성 문제 해결에 있어 효과적인 방법임을 입증합니다.



### A review on development of eco-friendly filters in Nepal for use in cigarettes and masks and Air Pollution Analysis with Machine Learning and SHAP Interpretability (https://arxiv.org/abs/2501.13369)
- **What's New**: 이 연구는 네팔, 특히 카트만두와 같은 도시에서의 공기 오염 문제를 다루고 있습니다. 연구에서는 Random Forest Regressor를 사용하여 공기질 지수(AQI)를 예측하고, SHAP(SHapley Additive exPlanations) 분석으로 모델의 예측 결과를 해석합니다. CatBoost 모델이 가장 낮은 Testing RMSE(0.23)와 완벽한 R2 스코어(1.00)를 기록하여 다른 모델보다 높은 정확도를 보여주었습니다.

- **Technical Details**: SHAP 분석 결과, NowCast Concentration 및 Raw Concentration이 AQI 값을 결정하는 주요 요소로 확인되었습니다. 이러한 변수들은 AQI를 크게 증가시키는 중요한 공기 오염 기여자로서의 의미를 지닙니다. 또한, 이 연구는 수소 알파(Hydrogen-Alpha, HA) 생분해성 필터를 공기 질 개선의 새로운 방법으로 평가하고 있습니다.

- **Performance Highlights**: HA 필터는 PM2.5에 대해 98% 이상의 제거 효율을, PM10에 대해 99.24%의 제거 효율을 보여주며, 위험한 공기 입자에 대한 뛰어난 방어력을 제공합니다. 이 필터는 환경 문제를 해결하기 위한 생분해성 마스크와 담배 필터로 제작되었으며, 전통적인 필터의 비생분해성 쓰레기 문제를 줄이면서 대기 오염 물질에 대한 노출을 낮추는 데 기여하고 있습니다.



### Learning to Bid in Non-Stationary Repeated First-Price Auctions (https://arxiv.org/abs/2501.13358)
- **What's New**: 최근의 연구는 디지털 광고 시장에서 첫 번째 가격 경매(first-price auction)의 중요성을 강조하고 있습니다. 특히, Google이 두 번째 가격 경매(second-price auction)에서 첫 번째 가격 경매로 전환하면서 이 주제가 주목을 받게 되었습니다. 이 연구는 첫 번째 가격 경매에서의 최적 입찰 전략(optimal bidding strategy)의 복잡성을 해결하는 데 초점을 두고 있습니다.

- **Technical Details**: 첫 번째 가격 경매에서 입찰자(bidder)는 환경 환경과 상호작용하여 타 입찰자들의 행동을 추론합니다. 전통적으로 사용된 고정 정책(static policy) 기반의 성능 측정은 비정상성(non-stationarity)이 적은 상황에서만 유효합니다. 이를 극복하기 위해, 각 시간 점에서 가능한 최대 보상의 합을 나타내는 동적 기준(dynamic benchmark)을 제안하며, 이를 통해 더 적절한 목표를 설정할 수 있습니다.

- **Performance Highlights**: 연구에서는 온라인 첫 번째 가격 경매에서 보상 함수(reward function)를 분석하고, 입찰 시퀀스의 규칙성을 측정하는 두 가지 메트릭을 도입하여 비정상성(non-stationarity)을 정량화합니다. 이 메트릭들이 시간 지평선(time horizon)에서 부분 선형(sub-linear)인 경우, 동적 후회(dynamic regret)의 미분 최소한(minimax-optimal) 특성을 제공합니다.



### DoMINO: A Decomposable Multi-scale Iterative Neural Operator for Modeling Large Scale Engineering Simulations (https://arxiv.org/abs/2501.13350)
- **What's New**: 본 연구는 NVIDIA Modulus에서 개발된 새로운 기계 학습(ML) 모델 아키텍처인 DoMINO(Decomposable Multi-scale Iterative Neural Operator)를 제안합니다. DoMINO는 복잡한 기하학을 처리하고 더 빠르면서도 정확한 예측을 제공함으로써 공학 시뮬레이션의 서브로게이션 모델링에서의 여러 도전을 해결하고자 합니다. 이 모델은 포인트 클라우드 기반으로 지역 기하 정보를 활용하여 이산 포인트에서 유동장을 예측합니다. 이를 통해 DoMINO 모델의 확장성, 성능, 정확성 및 일반화 능력을 입증하였습니다.

- **Technical Details**: DoMINO는 다중 스케일( multi-scale ) 반복 신경 연산자를 기반으로 하여 복잡한 물리 및 화학 과정을 부분 미분 방정식( PDE )으로 나타내고 이를 해결하는 방법을 제시합니다. 이 모델은 외부 공기역학 사용 사례를 위해 DrivAerML 데이터 세트에서 검증되었으며, 여기서 각 자동차 기하학에 대해 큰 표면 및 볼륨 메쉬를 사용하여 흐름 필드를 캡처합니다. 연구진은 기존의 ML 방법이 메쉬의 크기와 구멍의 인과 관계를 효율적으로 표현하지 못하는 한계를 극복하고자 합니다.

- **Performance Highlights**: DoMINO 모델은 다양한 설계 변형에 대해 드래그 포스 변화가 300 N에서 600 N까지 진행되는 DrivAerML 데이터 세트를 사용하여 훈련 및 검증됩니다. 통해 모델은 공기역학적인 양을 신속하게 예측하고, 특히 차량 설계에 대한 최적화된 지침을 제공하는 데 중요한 속도 필드, 압력, 벽 전단 응력을 포착하는 능력을 입증하였습니다. 이 실험을 통해 DoMINO 모델이 외부 공기역학을 모델링하기 위한 기존 방법보다 큰 메쉬에서 높은 정확성과 일반화 능력을 유지한다는 점을 보여줍니다.



### One Fits All: General Mobility Trajectory Modeling via Masked Conditional Diffusion (https://arxiv.org/abs/2501.13347)
- **What's New**: 이 논문은 여러 가지 궤적 관련 작업을 처리할 수 있는 일반적인 궤적 모델링 프레임워크인 GenMove를 제안합니다. 기존의 궤적 데이터 연구는 각 작업에 특화된 모델을 사용하여 제한된 유연성으로 인해 적용 가능성이 낮았습니다. GenMove는 masked conditional diffusion 기술을 활용하여 다양한 포맷을 통일하고, 복잡한 조건에 적응할 수 있도록 설계되었습니다.

- **Technical Details**: GenMove는 궤적 데이터의 다양한 형식을 통합하기 위해 마스크 조건 모듈을 설계하였습니다. 이 프레임워크는 과거 궤적 데이터를 기반으로 맥락적 궤적 임베딩(contextual trajectory embeddings)을 생성하고, 이 정보를 분산 모델에 통합하여 결과의 유연성을 높입니다. 특히 classifier-free guidance 방식을 사용하여 다양한 조건에 대한 출력을 조정할 수 있도록 하는 점이 특징입니다.

- **Performance Highlights**: 실험 결과, GenMove는 6개의 궤적 관련 작업에서 최첨단 모델 대비 13% 이상의 성능 향상을 달성했습니다. 이 성과는 다양한 응용 프로그램 시나리오에 유연하게 적응할 수 있는 일반 모델의 잠재력을 입증합니다. GenMove는 궤적 생성, 복구 및 예측과 같은 주류 작업에서 뛰어난 성능을 보였습니다.



### AgentRec: Agent Recommendation Using Sentence Embeddings Aligned to Human Feedback (https://arxiv.org/abs/2501.13333)
Comments:
          10 pages, 8 figures, preprint

- **What's New**: 이번 논문에서는 다중 에이전트 시스템에서 자연어 프롬프트에 대해 가장 적합한 LLM (Large Language Model) 에이전트를 추천하는 새로운 아키텍처를 제안합니다. 기존의 Sentence-BERT (SBERT) 인코더 모델을 확장하여 개발된 이 시스템은 에이전트 분류의 정확도를 높이는데 기여합니다. 특히, 자연어 프롬프트를 문장 임베딩으로 인코딩하여 에이전트 추천과 관련된 의미론적 내용을 포착합니다.

- **Technical Details**: 모델은 문장 임베딩 간의 코사인 유사도를 측정하여 자연어 프롬프트를 분류합니다. 각 분류 작업은 300밀리초 미만의 시간 내에 완료되며, 테스트 데이터에서 92.2%의 상위 1 정확도가 달성되었습니다. 또한, 강화학습을 통해 인간의 가치에 맞춰 조정할 수 있으며, 새로운 클래스에 적응할 수 있는 등 컴퓨팅 비용이 저렴하고 해석 가능성이 높은 구조를 지니고 있습니다.

- **Performance Highlights**: 이 연구는 에이전트 추천을 위한 합성 데이터셋을 생성하여 가능해졌으며, 생성된 데이터셋 및 AgentRec 추천 시스템의 코드는 공개되어 있습니다. 따라서 연구 결과는 실제 응용에 쉽게 활용될 수 있으며, 다중 에이전트 시스템의 효율성을 크게 향상할 것으로 기대됩니다.



### Co-Learning Bayesian Optimization (https://arxiv.org/abs/2501.13332)
- **What's New**: 이번 논문에서는 Bayesian 최적화(Bayesian optimization, BO)가 저품질 솔루션에 고착되는 문제를 해결하기 위해 다양한 Gaussian Process 모델을 활용하는 새로운 BO 알고리즘인 co-learning BO (CLBO)를 제안합니다. 이 접근법은 개별 모델의 예측 정확도를 높이고, 레이블이 없는 정보에 대한 모델들의 일치를 이용하여 필요 샘플 수를 줄이며, 최적화 효율성을 향상시킵니다. 또한, CLBO는 기존 방법들과의 비교를 통해 효과성을 입증하였습니다.

- **Technical Details**: CLBO는 여러 Gaussian Process(GP) 모델을 사용하여 BO의 서구레이트 정확성을 향상시키는 방법을 제안합니다. 이 방법은 bias-variance tradeoff에 대한 이해를 바탕으로 하며, 개별 모델의 예측 오차를 최소화하면서 모델 간의 다양한 예측을 활용하여 전체 모델의 정확도를 높입니다. 또한, 레이담처 복잡도(Rademacher complexity) 이론을 기반으로, 레이블이 없는 정보에서의 모델 일치를 통해 샘플 복잡성을 줄임으로써 보다 효율적인 최적화를 달성합니다.

- **Performance Highlights**: CLBO의 효과성은 다양한 수치 toy 문제와 세 가지 엔지니어링 벤치마크에서의 테스트를 통해 잘 입증되었습니다. 실험 결과, CLBO는 제한된 샘플로도 철저한 최적화 결과를 달성할 수 있으며, 기존의 BO 접근법보다 우수한 성능을 보여줍니다. 따라서 CLBO는 복잡한 최적화 문제를 해결하는 데 있어 매우 유망한 새로운 방법이라고 할 수 있습니다.



### Qrazor: Reliable and effortless 4-bit llm quantization by significant data razoring (https://arxiv.org/abs/2501.13331)
Comments:
          19 pages

- **What's New**: 이번 논문에서는 LLMs(대규모 언어 모델)의 4비트 양자화(quantization)를 가능하게 하는 새로운 방법인 QRazor를 소개합니다. QRazor는 메모리 요구사항을 줄이고 계산 복잡성을 낮추기 위한 간단하면서도 효과적인 양자화 기법으로, 원본 모델과 거의 동일한 정확도를 유지할 수 있도록 설계되었습니다. 또한 QRazor는 4비트로의 양자화에 더해 전용 정수 기반 산술 유닛을 도입하여 저정밀 산술 연산을 직접 처리하도록 합니다.

- **Technical Details**: QRazor는 두 가지 주요 단계, 즉 양자화(quantization)와 압축(compression)으로 구성됩니다. 양자화 단계에서는 8비트 또는 16비트 정수를 기반으로 가중치(weights), 활성화(activations), KV 캐시를 양자화하며, 원본 모델과 유사한 정확도를 달성합니다. 이후 Significant Data Razoring (SDR) 기술을 사용하여 모든 데이터를 4비트로 압축하고, 가장 중요한 4비트만 유지하도록 설계하였습니다.

- **Performance Highlights**: QRazor는 LLaMA-1-7B 및 LLaMA-13B 모델에서 우수한 성능을 입증하며, QLLM 및 Quarot과 비교하여 10% 이상의 정확도 개선을 달성하였습니다. 또한 QRazor를 구현한 정수 기반 산술 유닛은 압축된 데이터로 계산을 직접 수행할 수 있어 전력 소비를 57.8% 줄이고 면적을 61.2% 절감하는 등의 하드웨어 효율성도 검증되었습니다.



### Sparse identification of nonlinear dynamics and Koopman operators with Shallow Recurrent Decoder Networks (https://arxiv.org/abs/2501.13329)
- **What's New**: 이 논문에서는 SHallow REcurrent Decoder 네트워크(SHRED)의 희소 비선형 동역학 식별(SINDy-SHRED) 방법을 제안합니다. 이 방법은 센서 측정을 통해 간단하게 구현할 수 있으며, 효율적인 계산과 강력한 성능을 발휘합니다. SINDy-SHRED는 Gated Recurrent Units (GRUs)를 사용하여 시간적인 센서 측정값의 시퀀스를 모델링하고, 잠재 상태 공간에서 전체 시공간 필드를 재구성합니다.

- **Technical Details**: SINDy-SHRED는 재귀 신경망의 잠재 공간을 활용하고 SINDy 기반의 함수 클래스를 통해 해석 가능성을 강화합니다. 해당 알고리즘은 비선형 모델을 선형 모델로 제한하여 Koopman-SHRED 아키텍처를 생성합니다. 이로 인해, SINDy-SHRED는 적은 수의 센서로도 효과적인 회복을 수행하고, 대규모 과학 데이터 모델링 및 실시간 제어에 적합합니다.

- **Performance Highlights**: SINDy-SHRED는 낮은 데이터 요구량과 빠른 훈련 속도를 자랑하며, 다양한 실제 사례에 대해 우수한 성능을 보여줍니다. 실험 결과에 따르면, SINDy-SHRED는 Convolutional LSTM, PredRNN, ResNet, SimVP 등 다양한 기존 방법들을 초월하는 성능을 달성했습니다. 이와 같은 성능은 장기적인 예측에서도 안정적인 결과를 제공합니다.



### Tensor-Var: Variational Data Assimilation in Tensor Product Feature Spac (https://arxiv.org/abs/2501.13312)
- **What's New**: 이 논문에서는 데이터 동화(data assimilation)의 문제를 해결하기 위해 새로운 방법인 Tensor-Var를 제안합니다. Tensor-Var는 kernel Conditional Mean Embedding (CME)을 활용하여 시스템의 동역학과 상태-관측 매핑을 선형 연산자로 특성화함으로써 비용 함수를 볼록(convex)으로 만들어 최적화 효율성을 개선합니다. 또한, Tensor-Var는 4D-Var에 CME를 통합하여 원래 공간과 특성 공간 간의 일관된 동화 결과에 대한 이론적 보장을 제공합니다.

- **Technical Details**: Tensor-Var는 비선형 동역학을 선형화하여 4D-Var의 효율성을 높이는 새로운 접근 방식을 제안합니다. 랜덤 변수를 통해 시스템 상태와 관측치를 다루며, 관측의 불완전성을 해결하기 위해 시스템 상태를 추론하는 역관측 연산자를 도출합니다. 이 방법은 데이터의 특성을 고정 차원 특성 공간으로 변환하는 적응형 심층 특징(deep features) 학습 방법을 포함하여 계산 복잡성을 줄이고 확장성을 개선합니다.

- **Performance Highlights**: 실험 결과, Tensor-Var는 전통적인 방법 및 머신러닝 하이브리드 접근 방식과 비교해 정확성과 계산 효율성에서 뛰어난 성능을 보였습니다. 특히, 혼돈 시스템과 실제 시간 관측을 이용한 글로벌 기상 예측에서 Tensor-Var가 기존 모델보다 우수한 성능을 발휘함이 입증되었습니다. 이러한 결과들은 Tensor-Var를 통한 동화 시스템의 선형화의 이점을 강조합니다.



### Exploring Variance Reduction in Importance Sampling for Efficient DNN Training (https://arxiv.org/abs/2501.13296)
Comments:
          19 pages

- **What's New**: 이 논문은 딥 신경망(DNN) 훈련에서 중요 샘플링(importance sampling)의 효율성을 평가하기 위한 새로운 방법론을 제시합니다. 특히, 이 방법은 중요 샘플링 하에 샘플링된 미니배치(minibatch)만을 사용해 분산 감소를 추정할 수 있는 기회를 제공합니다. 또한 자동 학습률 조정을 위한 효과적인 미니배치 크기(EMS) 제안을 포함하고 있으며, 중요 점수(importance score)를 실시간으로 추정하는 알고리즘이 소개됩니다.

- **Technical Details**: 제안된 방법론을 통해, 모든 추정값은 대안 분포(alternative distribution)로 샘플링된 미니배치만을 사용하여 최소한의 계산 오버헤드로 진행됩니다. 이 연구는 학습 손실의 기대값을 예상하기 위해 각 훈련 샘플에 대한 중요 점수를 정확하게 추정하는 것이 중요하다고 강조합니다. 또한, 이 연구는 이론적으로 최적의 대안 분포를 통해 접근 가능한 분산 감소의 하한값을 추정합니다.

- **Performance Highlights**: 베이스라인 데이터셋을 통해 수행된 실험은, 제안된 알고리즘이 현재의 중요 샘플링 접근 방식보다 일관되게 분산을 줄이고 훈련 효율성을 높이며 모델의 정확성을 향상시킨다는 것을 보여줍니다. 논문에서 도출된 EMS는 중요 샘플링을 통해 얻은 분산과 동일한 수준의 분산을 얻기 위한 미니배치 크기를 제공합니다. 이 모든 과정에서 계산적 부담은 최소화됩니다.



### T-Graphormer: Using Transformers for Spatiotemporal Forecasting (https://arxiv.org/abs/2501.13274)
- **What's New**: 이 논문에서는 다양한 분야에서 발생하는 시계열 데이터의 복잡성을 다룬 T-Graphormer라는 새로운 모델을 소개합니다. 기존의 예측 방법들이 공간적과 시간적 의존성을 따로 학습하는 데 반해, T-Graphormer는 이러한 상관관계를 직접적으로 모델링할 수 있도록 설계되었습니다. 이는 특히 교통 흐름 분석과 같은 복잡한 시계열 데이터에 적용됩니다.

- **Technical Details**: T-Graphormer는 그래프( Graph ) 구조를 기반으로 한 Transformer 모델로, 각 노드의 표현을 업데이트할 때 그래프 시퀀스 내의 모든 다른 노드에 선택적으로 주의를 기울입니다. 이 방법은 미리 정의된 공간-시간 유도 편향(spacetime inductive biases)에 대한 최소한의 의존성으로도 풍부한 시공간 패턴(spatiotemporal patterns)을 캡처할 수 있도록 합니다. 그래프의 시간적 동역학을 통합하여 기존 Graphormer 아키텍처를 확장합니다.

- **Performance Highlights**: T-Graphormer는 실제 교통 예측 벤치마크 데이터셋에서 그 성능을 검증했습니다. 그 결과, 최신 기술에 비해 평균 제곱근 오차(RMSE)와 평균 절대 백분율 오차(MAPE)에서 각각 최대 10%의 감소를 달성하며 효과성을 입증했습니다.



### Enhancing Robust Fairness via Confusional Spectral Regularization (https://arxiv.org/abs/2501.13273)
Comments:
          ICLR 2025

- **What's New**: 최근 연구에서 'robust fairness'라는 중요한 문제가 강조되었습니다. 이는 robust accuracy가 서로 다른 클래스 간에 크게 달라져, 심층 신경망(DNN)의 신뢰성을 저하시킨다는 점입니다. 본 연구에서는 PAC-Bayesian 프레임워크 내에서 worst-class robust error를 위한 robust generalization bound를 도출했으며, 이전의 재가중 방법이 갖는 한계를 벗어난다는 의의를 가집니다.

- **Technical Details**: 이 논문은 기존의 adversarial training 접근방식에 기반하여 robust fairness를 향상시키기 위한 새로운 정규화 기법을 제안합니다. PAC-Bayesian 원리를 활용하여 불확실한 데이터 분포를 고려한 worst-class robust error를 분석하며, spectral norm을 포함한 empirical robust confusion matrix의 영향을 밝혔습니다. 이 접근법은 기존의 explicit reweighting 전략에 비해 훈련 세트와 테스트 세트 간의 불일치를 감소시키는 데 중점을 둡니다.

- **Performance Highlights**: 다양한 데이터 세트를 대상으로 한 실험을 통해 제안된 방법이 worst-class robust accuracy를 개선하고, adversarial 공격에 대한 보다 공정하고 신뢰할 수 있는 모델 예측을 보장함을 입증했습니다. CIFAR-10/100 및 Tiny-ImageNet 데이터셋을 통한 실험 결과는 우리의 방법이 모델의 성능을 현저히 향상시킨다는 것을 보여줍니다. 최악의 클래스를 위한 robust accuracy를 높이며, 모델이 더욱 고르지 않은 성능을 보여줄 수 있음을 입증했습니다.



### Hybrid Two-Stage Reconstruction of Multiscale Subsurface Flow with Physics-informed Residual Connected Neural Operator (https://arxiv.org/abs/2501.13271)
Comments:
          21 pages, 14 figures, 3 tables

- **What's New**: 이 연구는 고대비의 다공성 매체에서의 다르시 흐름 문제를 해결하기 위한 하이브리드 두 단계 프레임워크를 제안합니다. 첫 번째 단계에서는 데이터 기반 모델을 활용하여 다중 스케일 기초 함수(multiscale basis functions)를 재구성하고, 두 번째 단계에서는 물리 정보를 고려한 신경망(Physics-informed neural networks, PINNs)을 사용하여 압력 필드를 복원합니다. 이러한 접근법은 물리 법칙에 대한 일관성을 유지하면서도 정확한 재구성을 가능하게 합니다.

- **Technical Details**: 제안된 모델은 첫 번째 단계에서 고투과성 필드를 기반으로 한 다중 스케일 기초 함수의 정확한 재구성을 목표로 합니다. 이후 두 번째 단계에서는 다르시 방정식에서 유도된 물리적 제약 조건을 통합하여 압력 필드를 복원합니다. 이 프레임워크는 R2 값 0.9 이상의 높은 재구성 정확도를 달성하며, 잔여 표시기는 $1	imes 10^{-4}$의 차수에 달합니다.

- **Performance Highlights**: 모델은 다양한 투과성 조합과 기초 함수에서 테스트되었으며, 전반적으로 뛰어난 재구성 성능을 보였습니다. 연구의 결과는 다공성 매체에서의 유체 흐름 문제를 해결할 수 있는 새로운 가능성을 보여줍니다. 이 프레임워크는 물리 법칙에 대한 일관성을 유지하여 실세계 문제 해결에 기여할 수 있는 잠재력을 가지고 있습니다.



### Exploring the Technology Landscape through Topic Modeling, Expert Involvement, and Reinforcement Learning (https://arxiv.org/abs/2501.13252)
Comments:
          28 pages, 17 figures

- **What's New**: 이번 연구는 특정 기술 분야의 발전을 탐색하기 위한 새로운 방법을 제시합니다. 이 방법은 topic modeling, 전문가의 의견(expert input), 그리고 reinforcement learning (RL)을 결합하여 개발되었습니다.

- **Technical Details**: 제안된 접근 방식은 세 가지 주요 단계로 구성됩니다: (1) 전문가 가중 키워드를 사용하여 중요한 측면을 강조하는 aspect-based topic 모델 생성, (2) 반복 모델 간의 topic 분포를 비교하여 유사성(similarity) 및 엔트로피 변화를 분석, (3) 반복 과정에서 topic의 발산(divergence) 및 유사성 변화를 통합한 수정된 보상 함수로 RL을 이용해 topic 선택을 세분화합니다.

- **Performance Highlights**: 이 방법은 양자 통신 문서에서 암호화 및 보안 프로토콜의 발전에 초점을 맞춰 테스트되었습니다. 결과는 이 방법의 효과성을 보여주며, 전문가의 의견에 맞는 트렌드를 식별, 순위 매기기 및 추적할 수 있는 능력을 갖추고 있습니다.



### Multimodal AI on Wound Images and Clinical Notes for Home Patient Referra (https://arxiv.org/abs/2501.13247)
Comments:
          arXiv admin note: text overlap with arXiv:2208.05051 by other authors

- **What's New**: 이번 논문에서는 만성 상처 환자에 대한 의사결정을 지원하는 인공지능 프레임워크인 Deep Multimodal Wound Assessment Tool (DM-WAT)를 소개합니다. DM-WAT는 스마트폰으로 촬영한 상처 이미지와 전자 건강 기록(EHR)에서 추출한 임상 노트를 결합하여 분석합니다. 이를 통해 방문 간호사가 만성 상처 환자를 전문가에게 의뢰할지 여부를 판별하는 데 도움을 줍니다. 또, 모델의 해석 가능성과 신뢰성을 높이기 위해 Score-CAM 및 Captum 알고리즘을 활용하여 이미지와 텍스트 입력의 특정 부분이 추천에 미치는 영향을 설명합니다.

- **Technical Details**: DM-WAT는 Vision Transformer (ViT)인 DeiT-Base-Distilled 모델을 사용하여 상처 이미지에서 시각적 특징을 추출하고, BERT 기반의 DeBERTa-base 모델을 통해 임상 노트에서 텍스트적 특징을 추출합니다. 이러한 기능들은 중간 융합(intermediate fusion) 방법을 이용해 결합되어, 보다 정확한 임상 의사결정을 지원합니다. 이 과정에서 전이학습과 데이터 증강 기법을 결합하여, 소규모 데이터를 효과적으로 처리하고 높은 성능을 달성합니다.

- **Performance Highlights**: DM-WAT는 평가에서 77%의 정확도와 70%의 F1 점수를 기록하며 기존 방법들을 초과하는 성과를 달성했습니다. 이 연구는 고급 인공지능을 활용하여 방문 간호사가 신뢰성 있게 상처 치료 결정을 내릴 수 있는 도구를 제공함으로써, 비전문가인 방문 간호사의 결정을 지원하는 데 중점을 두고 있습니다. 향후 이러한 모델의 발전이 더 많은 환자에게 도움을 줄 수 있을 것으로 기대됩니다.



### State Combinatorial Generalization In Decision Making With Conditional Diffusion Models (https://arxiv.org/abs/2501.13241)
- **What's New**: 이번 연구는 보지 못한 상태에서의 조합 일반화(out-of-combination generalization)를 다루며, 기존 강화 학습(RL) 알고리즘이 이러한 새로운 조합의 상태에 대해 신뢰할 수 있는 가치 예측을 하지 못하는 문제를 제기합니다. 저자들은 행동 클로닝(behavior cloning)과 조건부 확산 모델(conditioned diffusion model)을 통해 새로운 조합의 상태에 대해 훨씬 더 우수한 일반화 성능을 보여주는 방법을 제시합니다. 특히, 이러한 접근 방식을 통해 현실 세계의 의사 결정 문제를 해결하는 데 중요한 기여를 할 수 있습니다.

- **Technical Details**: 연구에서는 상태 조합의 정의와 OOC 일반화의 개념을 명확히 하고, 기존의 RL 알고리즘이 보지 못한 상태에서 일반화하는 데 어려움을 겪는 이유를 설명합니다. 저자들은 조건부 확산 모델이 어떻게 상태의 조합 구조를 자연스럽게 반영하며, 더 나은 일반화를 가능하게 하는지를 실험적으로 보여줍니다. 실험은 미로, 자율주행, 다중 에이전트 환경을 통해 이루어졌으며, 다양한 RL 환경에서 향상된 성능을 입증했습니다.

- **Performance Highlights**: 실험 결과, 조건부 확산 모델을 기반으로 한 행동 클로닝이 기존의 오프라인 RL 방법 및 온라인 RL 방법과 비교해도 더 좋은 성능을 발휘했습니다. 특히, 제로샷 OOC 일반화에서 혁신적인 성능 향상을 보여주었으며, 모델이 기본 요소의 핵심 속성을 잘 포착하고 정확하게 새로운 상태를 구성하는 능력을 입증했습니다. 이를 통해 저자들은 탐색(exploration)보다 더 표현력이 풍부한 정책의 중요성을 강조하며, 이 접근 방식이 일반화의 중요한 요소임을 시사합니다.



### Let SSMs be ConvNets: State-space Modeling with Optimal Tensor Contractions (https://arxiv.org/abs/2501.13230)
Comments:
          25 pages, 7 figures

- **What's New**: Centaurus는 텐서 수축(tensor contractions)을 통해 훈련할 수 있는 일반화된 상태 공간 모델(state-space model, SSM) 블록으로 구성된 네트워크 클래스입니다. 기존의 SSM 블록 구조에서 더 큰 유연성을 제공하여, 심층 네트워크의 설계가 가능해졌습니다. Centaurus는 고전적인 합성곱(convolutional) 블록에서 영감을 받아 다양한 구조적 조합을 도입하여, 훈련과 추론에서의 메모리 및 계산 효율성을 극대화합니다. 이 네트워크는 특정 작업에서 동종의 구조보다 우수한 성능을 발휘하는 것으로 나타났습니다.

- **Technical Details**: Centaurus 네트워크는 다양한 유형의 SSM 블록에 대한 텐서 네트워크(tensor networks) 구조를 사용하여 새로운 구조를 설계합니다. 이 블록들은 기존의 심층 SSM 네트워크와 비교해 더 많은 연결 유연성을 제공하며, 새로운 SSM 블록들이 텐서 네트워크 형식으로 구현되어 있습니다. 각 SSM 블록의 수축 순서가 입력 특징 및 시스템 행렬의 형태에 따라 동적으로 최적화되어, 모든 SSM 블록에서 훈련 속도가 대폭 향상됩니다. 이러한 접근 방식은 간섭 없이 순수하게 SSM 기반의 성능을 달성할 수 있도록 도와줍니다.

- **Performance Highlights**: Centaurus는 키워드 탐지(keyword spotting), 음성 잡음 제거(speech denoising), 자동 음성 인식(automatic speech recognition, ASR) 등 여러 오디오 처리 작업에서 우수한 성능을 보입니다. 특히 ASR 성능에서 경쟁력 있는 결과를 보여주며, LSTM이나 CNN, 주의(attention) 메커니즘 없이도 효율적인 훈련과 실행이 가능합니다. 이러한 특징 덕분에 Centaurus는 음성 처리 및 관련 작업에서 매우 효율적인 선택으로 자리잡게 될 것입니다.



### MLPs at the EOC: Spectrum of the NTK (https://arxiv.org/abs/2501.13225)
Comments:
          18 pages, 1 figure

- **What's New**: 이번 연구는 Neural Tangent Kernel (NTK)의 특성을 다루며, 무한히 넓은 l-레이어의 Multilayer Perceptrons (MLPs)에 대한 새로운 분석을 제공합니다. 특히, 입력과 출력 간의 관계를 설명하는 활성화 함수와 Chaos의 경계 (Edge Of Chaos, EOC)에서의 초기화 상태를 고려하여 NTK의 행렬 값 커널을 연구합니다. 이를 통해 코사인 거리의 역수를 기초로 한 NTK 행렬의 스펙트럼 구속을 도출합니다.

- **Technical Details**: 이 연구에서는 깊이가 증가함에 따라 NTK의 각 요소가 입력에 대응하는 활성화의 코사인 거리의 역수로 더 잘 근사된다는 것을 발견했습니다. 연구의 핵심 내용은 NTK 행렬의 스펙트럼을 분석하여, 이를 통해 NTK의 조건 수(concondition number)가 깊이에 따라 수렴하는 속도를 결정할 수 있음을 시사합니다. 구체적으로, 매개변수의 관계에 따라 활성화 함수의 효과가 달라짐을 보여줍니다.

- **Performance Highlights**: 최종적으로, $
abla_	ext{NTK}$는 절대 값의 경우가 ReLU보다 더 나은 성능을 보임을 나타냅니다. 이 결과는 다층 신경망의 학습의 안정성을 높이는 데 기여할 것으로 기대됩니다. 연구의 결과는 무한히 깊은 넓은 네트워크의 성능 분석에 대한 새로운 통찰력을 제공하며, 오버파라미터화된 학습 문제 분석에 기여할 것으로 보입니다.



### Scaling for Fairness? Analyzing Model Size, Data Composition, and Multilinguality in Vision-Language Bias (https://arxiv.org/abs/2501.13223)
- **What's New**: 본 연구는 대규모 비전-언어 모델(vision-language models, VLMs)에서 데이터 구성, 모델 크기 및 다국어 훈련이 성별 및 인종 편향에 미치는 영향을 체계적으로 조사합니다. 특히, CLIP 모델과 그 오픈 소스 변종을 평가하며, 평형이 결여된 데이터셋이 어떻게 편향을 심화할 수 있는지를 강조합니다. 이 연구는 포함된 정보와 실험적 평가를 통해 향후 VLM의 공정성을 높이기 위한 발전 방향을 제시합니다.

- **Technical Details**: 실험에서는 MONOLINGUAL 및 MULTILINGUAL 설정에서의 CLIP 기반 아키텍처를 사용하며, 두 개의 주요 아키텍처(CLIP ViT-B/32와 CLIP ViT-L/14)를 평가합니다. OpenCLIP 변형 또한 LAION 데이터세트(400M, 2B 이미지-텍스트 쌍)에 대해 실험합니다. 이 연구는 FairFace와 PATA 두 개의 실험 데이터세트를 사용하여 성별 및 인종 편향을 평가합니다.

- **Performance Highlights**: 연구 결과, 더 큰 훈련 데이터셋이 일부 편향을 완화하는 데 도움을 주지만, 특정 상황에서는 새로운 편향을 초래할 수 있음을 보여줍니다. 모델 크기를 늘리면 성능이 일반적으로 향상되지만, 편향을 반드시 줄이는 것은 아니라 특정 상황에서는 오히려 악화될 수도 있습니다. 다국어 훈련은 언어 범위를 확장하지만, 편향을 본질적으로 중립화하지는 않으며, 오히려 기존의 불평등을 강화할 수도 있음을 알립니다.



### Enhancing Multi-Attribute Fairness in Healthcare Predictive Modeling (https://arxiv.org/abs/2501.13219)
Comments:
          Accepted to the 13th IEEE International Conference on Healthcare Informatics (IEEE ICHI 2025)

- **What's New**: 이 논문에서는 헬스케어 AI의 공정성 문제를 다루는 다중 속성 공정성 최적화 접근 방식을 제안합니다. 기존의 공정성 증진 기법들은 일반적으로 단일 민감 속성에 초점을 맞추고 있지만, 본 연구는 여러 인구통계학적 속성이 동시에 고려되어야 함을 강조합니다. 저자들은 모델의 예측 성과 최적화 후, 공정성을 동시에 조정하는 두 가지 전략, 즉 순차적 및 동시에 최적화하는 방법을 통해 헬스케어 분야의 필요성을 충족하고자 합니다.

- **Technical Details**: 본 연구는 데이터의 공정성을 높이기 위해 전이 학습(Transfer Learning)을 기반으로 한 방법론을 제안합니다. 이 접근법은 먼저 예측 성과를 최적화한 후, 이 모델을 공정성 최적화 단계로 전이합니다. 손실 함수(Loss Function)와 페널티(term) 조정을 통해 여러 민감 속성의 공정성을 개선하면서 예측 능력은 유지하는 방식으로 진행됩니다.

- **Performance Highlights**: 연구 결과, 제안된 방법은 여러 속성에서 Equalized Odds Disparity(EOD)를 유의미하게 줄이며 높은 예측 정확도를 유지했습니다. 단일 속성 공정성 방법이 비타겟 속성의 불균형을 초래할 수 있는 반면, 동시에 다중 속성을 최적화하는 접근 방식이 각 속성에 걸쳐 보다 균형 잡힌 공정성 향상을 달성했습니다. 이러한 결과는 헬스케어 AI에서 포괄적인 공정성 전략의 중요성을 강조하며, 향후 연구 방향에 대한 유망한 가능성을 제시합니다.



### SRMT: Shared Memory for Multi-agent Lifelong Pathfinding (https://arxiv.org/abs/2501.13200)
Comments:
          16 pages, 11 figures

- **What's New**: 이번 연구에서는 다중 에이전트 시스템에서의 행동 조정을 개선하기 위한 새로운 접근 방식을 제안합니다. 이를 위해 에이전트들이 협력할 수 있는 글로벌 작업 공간으로서 공유 메모리 개념을 도입한 Shared Recurrent Memory Transformer (SRMT)을 제안합니다. 이 방법론은 에이전트들이 정보를 간접적으로 교환하고 행동을 조정할 수 있도록 메모리 변환기를 다중 에이전트 환경으로 확장합니다. SRMT는 Bottleneck 탐색 작업과 POGEMA 벤치마크 세트에서 평가되어 다른 기존 방법들에 비해 뛰어난 성능을 보였습니다.

- **Technical Details**: SRMT는 에이전트들이 개별 작업 메모리를 풀링하고 이를 전 세계적으로 방송함으로써 각각의 메모리를 효과적으로 활용하도록 설계되었습니다. 이 아키텍처는 에이전트들이 협업할 수 있는 능력을 향상시키며, 이는 특히 자원이 희박한 상황에서 서로의 상태를 관찰하고 행동 결정을 내릴 수 있게 도와줍니다. 연구에서는 SRMT가 Partially Observable Multi-Agent Pathfinding (PO-MAPF) 문제에서 작동하여 에이전트들이 자신들의 목표를 달성하는 동시에, 환경의 상태를 지역적으로만 관찰하는 방식으로 진행되었습니다.

- **Performance Highlights**: Bottleneck 작업에서 SRMT는 다양한 강화 학습 기초 벤치마크보다 일관되게 우수한 성능을 보였으며, 훈련 중에 관찰된 것보다 긴 복도에서도 효과적으로 일반화됩니다. POGEMA 지도를 포함한 탐색에서도 최근 MARL 및 하이브리드 알고리즘과 경쟁력 있는 성능을 나타내어, 다중 에이전트 시스템에서의 협력을 향상시킬 수 있는 가능성을 제시합니다. 이러한 결과는 공유 재발 메모리를 Transformer 기반 아키텍처에 통합함으로써 분산형 다중 에이전트 시스템의 조정을 강화할 수 있음을 시사합니다.



### S-LoRA: Scalable Low-Rank Adaptation for Class Incremental Learning (https://arxiv.org/abs/2501.13198)
- **What's New**: 최근 Continual Learning (CL) 분야에서 Foundation Models를 기반으로 한 S-LoRA 방법론이 제안됨으로써, sequential tasks에 대한 효율적인 학습이 가능해졌습니다. S-LoRA는 기존의 gating 메커니즘을 대체하는 새로운 접근 방식을 제공하며, LoRA 파라미터의 방향과 크기를 분리하여 학습하는 방식으로 설계되었습니다. 이러한 점에서 S-LoRA는 기존 방법의 단점을 최소화하면서도 더 나은 성능을 발휘합니다.

- **Technical Details**: S-LoRA는 Low-Rank Adaptation의 확장된 형태로, incremental하게 LoRA 컴포넌트를 추가하며, 그 방향성과 크기의 학습을 분리합니다. 이 방식은 gating 프로세스 없이도 최종 학습된 모델을 처리에 직접 사용할 수 있어 추론 효율성을 크게 향상시킵니다. 제안된 방법은 기존 CL 방법과 달리 sample feature를 저장할 필요가 없어, rehearsal-free한 특성을 유지합니다.

- **Performance Highlights**: 광범위한 CL 벤치마크와 다양한 Foundation Models에 대한 실험을 통해 S-LoRA는 지속적으로 우수한 성능을 보였습니다. S-LoRA는 저손실 경로를 자동으로 학습하고, 이에 따라 탁월한 안정성과 유연성을 구현함으로써 class incremental learning (CIL) 분야에서 혁신적인 기여를 하고 있습니다. 강조할 점은, S-LoRA가 더욱 향상된 버전인 ES-LoRA로 이어져, 학습 효율성과 확장성을 더욱 높일 수 있다는 것입니다.



### AirRadar: Inferring Nationwide Air Quality in China with Deep Neural Networks (https://arxiv.org/abs/2501.13141)
- **What's New**: 본 논문에서는 AirRadar라는 새로운 딥러닝 기반 모델을 소개합니다. 이 모델은 공기질 모니터링 스테이션이 없는 지역에서 실시간 공기질을 정확하게 추정하는 데 도움을 줍니다. AirRadar는 기존 데이터를 사용하여 학습 가능한 마스크 토큰을 이용해 감지되지 않은 공기질 특성을 복원합니다. 이 모델은 중국 전역의 1,085개 모니터링 스테이션에서 수집된 데이터를 통해 그 효용성이 입증되었습니다.

- **Technical Details**: AirRadar는 두 개의 주요 단계로 작동하며, 첫 번째로 공간 상관관계를 포착하고 두 번째로 분포 변화를 조정합니다. Spatial Learning Module과 Casual Learning Module을 통해 복잡한 공간 관계와 지역별 이질성을 효과적으로 처리합니다. 지역 및 글로벌 Learner를 활용하여 공기질 인퍼런스를 수행하고, 데이터 사용의 효율성을 극대화합니다. 이는 masking 기법을 통해 이루어지며, 백도어 조정(backdoor adjustment)을 적용하여 공간 이질성을 해결합니다.

- **Performance Highlights**: AirRadar는 다양한 마스킹 비율 하에서 수천 개의 위치에 대한 공기질 데이터를 평가하여, 최첨단 방법인 STFNN과 비교하여 28.0%에서 44.8%의 정확도 향상을 보였습니다. 또한, 실용성을 검증하기 위해 웹 기반 플랫폼이 배포되어, 사용자가 결과를 쉽게 확인할 수 있도록 하였습니다. 이 연구는 대규모 공기질 모니터링의 미래 방향성을 제시하며, 사회적으로 중요한 공공보건 문제에 기여할 수 있는 가능성을 보여줍니다.



### Efficient Implementation of LinearUCB through Algorithmic Improvements and Vector Computing Acceleration for Embedded Learning Systems (https://arxiv.org/abs/2501.13139)
- **What's New**: 이번 연구는 자원 제한 장치에서 LinearUCB Contextual Bandits 알고리즘을 효율적으로 구현하기 위한 알고리즘 및 하드웨어 최적화를 제안합니다. 특히, Sherman-Morrison-Woodbury 공식을 기반으로 한 알고리즘 수정은 모델의 복잡성을 줄이는 데 기여하며, 벡터 가속화를 통해 행렬 연산의 속도를 높입니다. 이러한 최적화 기술들은 자원 제약 환경에서도 실시간으로 AI 알고리즘 실행을 가능하게 합니다.

- **Technical Details**: 연구는 두 가지 LinearUCB 알고리즘인 Disjoint와 Hybrid에 초점을 맞추고 있습니다. Disjoint LinearUCB에서는 각 액션이 독립적으로 모델링되며, Hybrid LinearUCB는 공통된 피쳐 벡터를 사용하여 상호작용을 고려합니다. 알고리즘 수행 시 행렬 업데이트를 효율적으로 처리하기 위한 증분 업데이트 메커니즘이 도입되어, 모든 엣지 컴퓨팅 플랫폼에서 메모리 요구사항 및 전력 소모를 줄입니다.

- **Performance Highlights**: 결과적으로 최적화된 알고리즘은 실행 속도와 에너지 소비에서 현저한 개선을 보여줍니다. Klessydra-T13 RISC-V 프로세서의 벡터 하드웨어 가속기가 행렬 연산의 성능을 향상시켜 알고리즘 실행 속도를 높이는데 기여합니다. 이러한 알고리즘과 하드웨어 최적화를 결합한 접근 방식은 실시간 처리가 필요한 저전력 환경에 적합한 AI 모델의 성능 개선을 실현합니다.



### Graph Representation Learning with Diffusion Generative Models (https://arxiv.org/abs/2501.13133)
- **What's New**: 이번 연구는 그래프 구조 데이터에 대한 디퓨전 모델의 활용을 탐색합니다. 디퓨전 모델은 이미지 및 비디오와 같은 복잡한 데이터 분포를 정확히 모델링할 수 있는 생성 모델로 알려져 있습니다. 이 연구에서는 자동 인코더(autorecoder) 프레임워크 내에서 이산 디퓨전 모델을 훈련시켜 그래프 데이터에 적합한 의미 있는 표현을 학습하도록 합니다.

- **Technical Details**: 딥러닝에서는 표현 학습이 매우 중요한 분야로, 원시 데이터를 압축된 저차원 임베딩으로 변환하는 것이 목표입니다. 디퓨전 자동 인코더에서는 인코더가 유용한 표현을 학습하고, 디퓨전 디코더가 그 표현을 바탕으로 데이터를 생성합니다. 이산 디퓨전 모델은 그래프 데이터에 적합하도록 설계되어, 격리된 피처를 효과적으로 처리하며, 구조적 패턴과 관계를 포착합니다.

- **Performance Highlights**: Discrete Diffusion Autoencoder(DDAE) 모델은 그래프 분류 작업에서 뛰어난 성능을 보입니다. 이러한 모델은 그래프 자료의 이산적 특성을 잠재 임베딩으로 변환함으로써, 더 의미 있는 저차원 표현을 학습할 수 있습니다. 또한, 이 방법은 레이블된 데이터가 부족한 경우에도 비지도 학습을 통해 그래프 표현을 생성할 수 있는 가능성을 보여줍니다.



### Towards Robust Multimodal Open-set Test-time Adaptation via Adaptive Entropy-aware Optimization (https://arxiv.org/abs/2501.13924)
Comments:
          Accepted by ICLR 2025

- **What's New**: 본 연구에서는 Multimodal Open-set Test-time Adaptation (MM-OSTTA)에 대한 새로운 접근 방식인 Adaptive Entropy-aware Optimization (AEO) 프레임워크를 제안합니다. 기존의 unimodal OSTTA 방법의 한계를 극복하고자 하며, 특히 다중 모달리티에서 발생하는 분포 변화에 효과적으로 적응하는 방법을 탐구합니다. AEO은 알려진 클래스와 알려지지 않은 클래스 간의 엔트로피 차이를 증폭시킴으로써 MM-OSTTA 성능을 개선할 수 있음을 보여줍니다.

- **Technical Details**: MM-OSTTA는 사전 훈련된 멀티모달 모델이 다양한 모달리티 간의 상호 정보를 효율적으로 활용하여 실시간으로 적응하고 알려지지 않은 클래스를 탐지하는 것을 목표로 합니다. 이를 위해 Unknown-aware Adaptive Entropy Optimization (UAE)와 Adaptive Modality Prediction Discrepancy Optimization (AMP)라는 두 가지 주요 구성 요소를 도입합니다. 이 구성 요소들은 각 샘플에 대해 엔트로피 임계값을 기준으로 가중치를 동적으로 할당하고 알려진 샘플에 대해 일관된 예측을 유지하면서 알려지지 않은 샘플에 대해 다양한 예측을 장려합니다.

- **Performance Highlights**: 제안된 AEO 프레임워크는 다양한 도메인 변화 환경에서 광범위한 실험을 통해 그 효능과 다양한 작업에 대한 적합성을 입증했습니다. 특히, 액션 인식 및 3D 의미 세분화와 같은 두 개의 다운스트림 작업에서 유용하게 평가되었으며, 리얼 월드와 유사한 장기적인 MM-OSTTA 시나리오에서도 우수한 성능을 보였습니다. 이 프레임워크는 알려진 클래스와 알려지지 않은 클래스 간의 엔트로피 차이를 지속적으로 최적화하며, 이는 실제 동적 애플리케이션에서 필수적인 기능입니다.



### IMAGINE-E: Image Generation Intelligence Evaluation of State-of-the-art Text-to-Image Models (https://arxiv.org/abs/2501.13920)
Comments:
          75 pages, 73 figures, Evaluation scripts: this https URL

- **What's New**: 최근의 확산 모델(diffusion models) 발전으로, 텍스트-이미지(text-to-image, T2I) 모델들이 놀라운 성과를 보여주고 있습니다. 특히 FLUX.1과 Ideogram2.0 같은 신규 모델들은 복잡한 작업에서 뛰어난 성능을 발휘하며, 이러한 모델들이 일반적인 용도로 활용될 가능성에 대한 논의가 이어지고 있습니다. 이 연구는 T2I 모델들이 전통적인 이미지 생성뿐만 아니라 이미지 편집, 비디오, 오디오, 3D 및 움직임 생성과 같은 다양한 분야에서 가진 능력을 탐구합니다.

- **Technical Details**: 우리는 T2I 모델들의 종합적인 평가를 위해 IMAGINE-E라는 새로운 평가 프레임워크를 개발했습니다. 평가 대상 모델로는 FLUX.1, Ideogram2.0, Midjourney, Dall-E3, Stable Diffusion 3, Jimeng 등을 선정하였으며, 다섯 가지 주요 도메인으로 평가를 나누었습니다. 이 도메인들은 구조화된 출력 생성, 사실성(factuality) 및 물리적 일관성(physical consistency), 특정 도메인 생성, 도전적인 시나리오 생성, 다중 스타일 생성 과제를 포함합니다.

- **Performance Highlights**: 이 모델들의 성능을 평가한 결과, 특히 FLUX.1과 Ideogram2.0이 구조화된 및 특정 도메인 작업에서 두드러진 성과를 보였습니다. 이러한 평가 결과는 T2I 모델들이 AI 툴로서의 가능성과 응용을 확대하고 있음을 시사합니다. 연구 결과는 T2I 모델들의 현재 상태와 미래 방향성을 제시하며, 일반적인 사용성을 향한 진화 과정을 설명합니다.



### Temporal Preference Optimization for Long-Form Video Understanding (https://arxiv.org/abs/2501.13919)
- **What's New**: 이 논문은 기존 비디오 대형 멀티모달 모델(video-LMM)의 시간적 이해(temporal grounding) 성능을 향상시키기 위한 새로운 프레임워크인 Temporal Preference Optimization (TPO)을 제안합니다. TPO는 두 가지 세분화된 데이터셋, 즉 특정 비디오 구간에 집중하는 localized temporal grounding과 전체 비디오 시퀀스의 시간적 의존성을 포착하는 comprehensive temporal grounding을 활용하여 모델의 시간적 반응을 개선합니다. 이 프레임워크는 자가 학습(self-training) 방법론을 통해 잘 정립된 반응과 덜 정확한 반응을 구분하는 데 도움을 줍니다.

- **Technical Details**: 비디오-LMM은 대형 언어 모델(LLM), 비주얼 인코더(visual encoder), 그리고 멀티모달 프로젝터(multimodal projector)를 포함하는 구조로, 입력 비디오 V와 텍스트 시퀀스 x를 처리하여 응답 y의 확률을 모델링합니다. TPO는 Direct Preference Optimization (DPO) 기법을 사용하여 인간의 선호도(preference)를 기반으로 모델의 파라미터를 최적화하며, 이는 명시적인 보상 모델이나 복잡한 강화 학습 알고리즘을 필요로 하지 않습니다. 이 과정에서 TPO는 비디오-LMM의 시간적 반응 능력을 강화하는데 중점을 두고 있습니다.

- **Performance Highlights**: TPO를 적용한 실험 결과, LongVideoBench, MLVU, Video-MME와 같은 세 가지 비디오 이해 벤치마크에서 각각 2.9%, 3.1%, 2.5%의 성능 향상을 달성했습니다. 특히, LLaVA-Video-TPO 모델은 Video-MME 벤치마크에서 7B 모델 중 최고의 성능을 기록하였으며, 이는 TPO의 확장성과 효율성을 강조합니다. 이러한 성과는 장기 비디오 이해(task) 및 시간적 추론(temporal reasoning)을 향상시키는 데 중요한 기초로 작용할 것으로 기대됩니다.



### Improving Video Generation with Human Feedback (https://arxiv.org/abs/2501.13918)
- **What's New**: 최근 비디오 생성 분야에서 큰 발전이 있었으나, 여전히 비디오와 프롬프트 간의 불일치 및 부드럽지 않은 동작 같은 문제가 남아 있습니다. 본 연구는 이러한 문제를 해결하고 비디오 생성 모델을 개선하기 위해 사람의 피드백을 활용하는 체계적인 파이프라인을 개발하였습니다. 특히, 현대 비디오 생성 모델에 중점을 둔 대규모 인간 선호 데이터셋을 구축하고 새로운 비디오 보상 모델인 VideoReward를 도입하여 성과를 측정하였습니다.

- **Technical Details**: 연구진은 약 182k개의 주석이 달린 비디오 생성 선호 데이터셋을 수집하였으며, 이는 시각 품질(Visual Quality), 동작 품질(Motion Quality), 텍스트 정렬(Text Alignment)이라는 세 가지 차원을 포함합니다. 비디오 생성 모델을 위한 세 가지 정렬 알고리즘 - Flow-DPO, Flow-RWR 및 Flow-NRG를 제안하여 이론과 실험을 통해 성능을 평가하였습니다. 특히, Flow-DPO는 고정된 매개변수 β로 설정했을 때 우수한 성능을 보였습니다.

- **Performance Highlights**: 실험 결과, VideoReward는 기존의 보상 모델을 크게 능가하는 성과를 보였으며, Flow-DPO는 Flow-RWR와 표준 감독 미세 조정 방법보다 우수한 성능을 나타냈습니다. 또한, Flow-NRG는 사용자 맞춤형 비디오 품질을 충족하기 위해 여러 목표에 대해 사용자 지정 가중치를 할당할 수 있는 기능을 제공합니다. 이 연구는 최신 비디오 생성 모델과 인간 선호를 정렬하는 기법에 대한 새로운 가능성을 제시합니다.



### GUI-Bee: Align GUI Action Grounding to Novel Environments via Autonomous Exploration (https://arxiv.org/abs/2501.13896)
- **What's New**: 이 논문은 GUI 행동 기반 모델을 새로운 GUI 환경에 맞춤화하여 성능을 개선하는 방법을 제안합니다. GUI-Bee라는 자율 탐색 에이전트를 통해 환경별 데이터를 수집하고, 이를 사용하여 모델을 지속적으로 미세 조정합니다. 이 접근법은 환경에 따라 달라지는 GUI 기반 모델 성능의 한계를 극복할 수 있는 중요한 단계를 제공합니다.

- **Technical Details**: 제안된 GUI-Bee 에이전트는 Q-value-Incentive In-Context Reinforcement Learning (Q-ICRL) 방법을 사용하여 탐색 효율성을 최적화합니다. 이 방법은 GUI 행동 후보의 상태-행동 값 예측을 통해 최적의 행동을 선택하고 반복적이지 않은 행동을 피할 수 있게 합니다. 실험을 통해 NovelScreenSpot 벤치마크를 사용하여 다양한 환경에 대한 모델 성능을 평가하고 있습니다.

- **Performance Highlights**: 우리의 실험 결과는 GUI-Bee 에이전트를 사용하는 모델이 미세 조정 전보다 크게 성능을 향상시켰음을 보여줍니다. Q-ICRL 방법이 데이터 수집의 효율성을 극대화했으며, 모델들이 새로운 GUI 환경에 적응하는 데 필요한 환경별 지식을 효과적으로 학습했음을 확인했습니다. 이러한 기여는 GUI 행동 모델이 실질적으로 다양한 환경에서 더 나은 기능을 발휘하도록 합니다.



### Pix2Cap-COCO: Advancing Visual Comprehension via Pixel-Level Captioning (https://arxiv.org/abs/2501.13893)
- **What's New**: Pix2Cap-COCO는 픽셀 수준에서의 캡션 작성을 위한 첫 번째 판옵틱 데이터셋으로, 세분화된 시각적 이해를 증진하기 위해 설계되었습니다. 본 연구에서는 GPT-4V를 활용하여 이미지의 객체들에 대한 픽셀 정렬(Pixel-aligned) 및 인스턴스 전용(instance-specific) 캡션을 자동으로 생성하도록 요청하는 주목적의 자동 주석 파이프라인을 도입했습니다. 총 167,254개의 세밀한 캡션을 생성하여, 각 캡션의 평균 길이는 22.94단어입니다.

- **Technical Details**: Pix2Cap-COCO는 COCO 데이터셋을 기반으로 구축되며, 이 데이터셋은 20,550개의 이미지와 167,254개의 캡션으로 구성되어 있습니다. 각각의 객체는 Set-of-Mark(SoM) 방법을 사용하여 이미지에서 마킹되고, 이후 GPT-4V에서 세밀한 캡션이 생성됩니다. 새로운 작업인 판옵틱 분할-캡션(panoptic segmentation-captioning) 도전 과제를 도입하여 모델이 이미지 내 인스턴스를 인식하며 동시에 상세 설명을 제공하도록 요구합니다.

- **Performance Highlights**: Pix2Cap-COCO는 GPT4RoI와 같은 대형 멀티모달 모델(LMM)의 성능을 향상시키는 데 중요한 역할을 합니다. 예를 들어, Pix2Cap-COCO로 훈련된 GPT4RoI는 ViP-Bench에서 평균 5.1%의 성능 향상을 보이며, 인식 정확도(+11.2%)와 언어 생성 품질(+22.2%)에서 주목할 만한 개선을 기록합니다. 이러한 성능 데이터는 Pix2Cap-COCO가 시각적 표현과 텍스트 표현 간의 세밀한 정렬을 위한 고품질 소스로서의 중요성을 강조합니다.



### First Lessons Learned of an Artificial Intelligence Robotic System for Autonomous Coarse Waste Recycling Using Multispectral Imaging-Based Methods (https://arxiv.org/abs/2501.13855)
Comments:
          Published in Proceedings of Sardinia 2023, 19th International Symposium on Waste Management, Resource Recovery and Sustainable Landfilling

- **What's New**: 현재의 조대 폐기물(코스 그레인 웨이스트) 처리 시설에서는 대형 기계를 사용한 수작업 분류가 이루어지고 있습니다. 이로 인해 재활용 가능한 많은 자원이 조대 폐기물로 분실되고 있는데, 더 효과적인 분류 프로세스 개발이 필요하다는 점이 강조됩니다. 이 연구는 혼합 폐기물 더미에서의 객체 탐지(object detection)와 자율 제어(autonomous control)를 통한 자동화된 분류 방법을 제안합니다.

- **Technical Details**: 주요 기술적 접근은 자외선(ultraviolet, UV), 가시광선(visual, VIS), 근적외선(near infrared, NIR), 그리고 단파 적외선(short-wave infrared, SWIR) 스펙트럼의 다중 스펙트럼 이미지(multispectral images)를 활용한 물질 분류(classification)입니다. 청사진인 이 기법은 대부분 손상되거나 파손된 폐기물 객체에 대해 적용될 수 있습니다. 또한, 비용 효율적인 카메라와 인공지능 기반 컨트롤러를 사용하여 대형 기계의 자율 제어 솔루션을 연구하고 있습니다.

- **Performance Highlights**: 이 연구는 자원을 회수하는 동시에 효율적인 폐기물 분류를 위한 새로운 기술적 진전을 이루고자 합니다. 제안된 방법은 기존 수작업 방식에 비해 성능을 향상시킬 수 있는 잠재력을 가지고 있으며, 구체적인 성과는 실험 및 검증을 통해 입증할 필요가 있습니다. 이러한 접근은 폐기물 관리와 재활용 산업에서 혁신적인 변화를 가져올 것으로 기대됩니다.



### Where Do You Go? Pedestrian Trajectory Prediction using Scene Features (https://arxiv.org/abs/2501.13848)
Comments:
          Accepted by 2024 International Conference on Intelligent Computing and its Emerging Applications

- **What's New**: 이 논문에서는 보행자 궤적 예측을 향상시키기 위해 보행자 상호작용과 환경 맥락을 통합한 새로운 궤적 예측 모델을 제안합니다. 기존 연구들이 보행자 간 상호작용 모델에 집중한 반면, 본 연구는 환경 요인과 장면 객체 배치의 중요성을 강조하였습니다. 공간적 및 시간적 상호작용을 포착하기 위해 sparse graph framework를 사용하고, scene features를 추출하기 위한 고급 이미지 강화 및 시맨틱 분할 기법을 사용합니다.

- **Technical Details**: 우리의 접근 방식은 Sparse Graph Convolutional Network (SGCN)을 통해 보행자 간의 상호작용을 포착하고, scene feature extraction 모듈과 cross-attention 메커니즘을 통해 예측 능력을 강화합니다. 이를 통해 보행자의 위치 정보와 주변 장면 정보를 기반으로 미래 궤적을 예측합니다. 이 과정에서는 각각의 보행자가 특정 시점에서의 공간 좌표로 특성화되어 과거 상태가 미래 위치 예측에 활용됩니다.

- **Performance Highlights**: 실험 결과, 제안한 방법이 기존의 최첨단 접근법들과 비교했을 때 ADE와 FDE 값이 각각 0.252m 및 0.372m로 현저히 우수함을 보여줍니다. 이는 보행자 궤적 예측에서 사회적 상호작용과 환경 맥락의 중요성을 강조합니다. 새로운 모델은 복잡한 도시 환경에서의 보행자 움직임 예측을 보다 효과적으로 수행할 수 있는 가능성을 제공합니다.



### Predicting Compact Phrasal Rewrites with Large Language Models for ASR Post Editing (https://arxiv.org/abs/2501.13831)
Comments:
          accepted by ICASSP 2025

- **What's New**: 이번 논문에서는 대규모 언어 모델(Large Language Models, LLMs)을 활용한 텍스트 리라이팅(task)에서 효율성을 높일 수 있는 새로운 편집 문구 표현(edit phrase representation)을 제안합니다. 이러한 표현은 기존의 span representation 대신, 구문 통계 기계 번역(phrase-based statistical machine translation)에서 영감을 받아 개발되었습니다. 연구자들은 이 방법이 ASR(Automatic Speech Recognition) 후 편집(task)에서 우수한 성과를 보일 것이라고 기대하고 있습니다.

- **Technical Details**: 제안된 문구 표현은 입력과 출력 사이의 겹치는 부분을 활용해 편집 작업의 수치적 표현을 압축할 수 있도록 돕습니다. 두 가지 새로운 표현 방식은 각 리라이트 패턴을 소스-타겟 구문 쌍으로 표현하거나, 좌우문맥 단어와 함께 타겟 문구만을 사용하는 방식입니다. 이러한 접근은 계산 비용을 줄이고, 의미적 일관성을 유지하며, 최종 출력의 품질을 확보하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 연구 결과, 제안된 문구 표현 방식이 기존의 span 모델과 비교했을 때, WER(Word Error Rate) 격차를 50-60% 줄이는 효율성을 보이라고 보고했습니다. 또한, 출력 길이 감소율 또한 기존 방식에 비해 10-20% 손실로 괜찮은 성능을 유지했습니다. 이로써 ASR 출력 수정에 대한 새로운 가능성을 열었으며, LLMs의 활용을 더욱 넓힐 수 있는 길을 제시했습니다.



### A space-decoupling framework for optimization on bounded-rank matrices with orthogonally invariant constraints (https://arxiv.org/abs/2501.13830)
Comments:
          48 pages, 12 figures, 6 tables

- **What's New**: 이 논문에서는 저랭크 최적화(low-rank optimization)에서 선형 제약조건을 추가하는 것의 중요성이 강조됩니다. 저자들은 각 제약의 접선 공간(tangent cone)이 어떻게 서로의 접선 공간과 교차되는지를 보여주며, 이를 통해 복잡한 저랭크 및 직교 불변 제약을 두 개의 공간으로 분리하는 새로운 프레임워크인 "공간 분리(space-decoupling)"를 제안합니다. 이 접근 방식은 기하학적 구조를 이해하고 Riemannian 알고리즘을 적용하는 데 유용합니다.

- **Technical Details**: 이 연구에서 다루는 최적화 문제는 차원 축소를 위한 저랭크 행렬을 포함합니다. 저자는 두 개의 상이한 공간을 활용하여 추가 제약을 직교적으로 불변하는 성질을 지닌 매트릭스에 적용하는 방법을 제안합니다. 여기서 h 매핑은 연속적이며, level set의 전체 rank를 유지하여 최적화 문제를 원할하게 설정합니다.

- **Performance Highlights**: 실제 데이터를 활용한 수치 실험들은 이 새로운 프레임워크의 우수성을 입증합니다. 저자들은 구형 데이터 적합(spherical data fitting), 그래프 유사도 측정(graph similarity measuring), 저랭크 SDP(low-rank SDP), 마르코프 과정의 모델 축소, 강화 학습, 그리고 딥 러닝에 대한 적용을 통해 다양한 사례를 제시합니다. 이 논문이 제공하는 방법론은 저랭크 최적화의 복잡성을 효과적으로 해결하는 데 기여할 것으로 기대됩니다.



### PhotoGAN: Generative Adversarial Neural Network Acceleration with Silicon Photonics (https://arxiv.org/abs/2501.13828)
- **What's New**: PhotoGAN은 GAN(Generative Adversarial Networks) 모델의 특수 연산을 처리하기 위해 설계된 최초의 실리콘 광학 가속기입니다. 이 가속기는 전통적인 전자 가속기에서 발생하는 비효율성을 해결하기 위한 혁신적인 접근을 제공합니다. PhotoGAN은 재구성 가능한 아키텍처를 통해 GAN의 전이 합성(transposed convolution) 및 인스턴스 정규화(instance normalization)와 같은 특수 레이어를 가속합니다.

- **Technical Details**: PhotoGAN은 실리콘 광학의 높은 처리량과 에너지 효율성을 활용하여 GAN 모델의 고유한 연산을 최적화합니다. 또한 드문 계산 최적화(sparse computation optimization) 기술을 도입하여 중복 작업을 줄임으로써 계산의 효율성을 높입니다. 이 아키텍처는 다양한 GAN 작업을 효과적으로 처리할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과에 따르면, PhotoGAN은 최신 GPU 및 TPU 가속기와 비교하여 최소 4.4배 높은 GOPS와 2.18배 낮은 에너지-비트당 소비(EPB)를 달성하였습니다. 이러한 성능은 향후 GAN 가속을 위한 유망한 솔루션으로 자리매김할 수 있는 가능성을 보여줍니다.



### Consistent spectral clustering in sparse tensor block models (https://arxiv.org/abs/2501.13820)
Comments:
          63 pagers

- **What's New**: 이 논문은 바이오인포매틱스(bioinformatics), 소셜 네트워크 분석(social network analysis) 및 추천 시스템(recommendation systems)과 같은 다양한 분야에서 사용되는 다차원 데이터셋에서 객체를 분류하는 고차원 클러스터링(high-order clustering) 기법을 제안합니다. 특히, 희소한(sparse) 정수 값 데이터 텐서(tensor)를 위한 텐서 블록 모델(tensor block model)을 도입하고, 소음의 변동성을 완화하기 위한 클러스터링 알고리즘을 제안합니다.

- **Technical Details**: 제안된 알고리즘은 trimming 단계를 추가하여 소음의 영향을 줄이며, 일관성을 보장하는 밀도 임계값(density threshold)을 식별합니다. 또한, sub-Poisson noise concentration framework를 사용하여 희소성을 모델링하며, sub-Gaussian tail 보다 더 무거운 분포를 처리할 수 있는 방법을 제안합니다. 이 텐서 블록 모델은 임의의 모드에 걸쳐 집합(aggregation)되어도 닫혀있는 성질을 가지고 있습니다.

- **Performance Highlights**: 이론적 발견은 희소 랜덤 그램 행렬(sparse random Gram matrices)에 대한 새로운 집중 경계를 바탕으로 분석되며, 데이터 집합 중 신호 손실(signal loss)과 소음 감소(noise reduction) 간의 트레이드오프(tradeoff)를 평가할 수 있는 포괄적인 프레임워크를 제공합니다. 시뮬레이션 실험을 통해 제안된 이론의 결과를 구체적으로 검증하였습니다.



### Ensuring Medical AI Safety: Explainable AI-Driven Detection and Mitigation of Spurious Model Behavior and Associated Data (https://arxiv.org/abs/2501.13818)
- **What's New**: 이번 연구에서는 딥 신경망(DNN)이 의료 분야에서 활용되면서 발생할 수 있는 단축 학습(shortcut learning) 문제를 해결하기 위한 반자동화된 프레임워크를 소개합니다. 이 프레임워크는 eXplainable Artificial Intelligence (XAI)의 통찰력을 활용하여 불필요한 행동을 감지하고 완화하는 데 중점을 둡니다. 기존의 방법들보다 더 많은 데이터 레이블링 작업 없이 스푸리어스(spurious) 데이터 포인트를 식별하고 모델 회로를 탐지하는 기능을 제공합니다.

- **Technical Details**: 제안된 프레임워크는 (1) 데이터 및 모델 관점에서 편향(bias)을 식별하고, (2) Concept Activation Vectors (CAVs)를 사용하여 모델 내부의 편향 표현을 학습하고, (3) 학습된 편향 모델을 통해 편향 데이터 샘플을 검색하며, (4) 이를 반복적으로 개선하는 방식으로 작동합니다. 또한 (5) 공간적 편향 지역화(spatial bias localization)를 통해 샘플 및 픽셀 단위의 주석을 데이터셋에 추가하여 편향 완화 및 평가 단계를 지원합니다. 이러한 방식으로 AI 모델의 일반화 능력을 향상시키는 것을 목표로 합니다.

- **Performance Highlights**: 연구는 VGG16, ResNet50 및 Vision Transformer와 같은 최근의 모델을 사용하여 4개의 의료 데이터셋에서 입증된 스푸리어스 상관관계를 성공적으로 식별 및 완화하였습니다. 결과적으로 이 프레임워크는 AI 모델의 강건성과 실용성을 개선하여 의료 작업에서 보다 안전하고 신뢰성 있는 결과를 제공합니다. 이러한 성과는 전통적인 수동 레이블링 작업의 필요성을 최소화하여 모델의 유효성과 강건함을 높였습니다.



### Parameter-Efficient Fine-Tuning for Foundation Models (https://arxiv.org/abs/2501.13787)
Comments:
          25 pages, 6 figures, 7 tables

- **What's New**: 본 설문조사는 Foundation Models (FMs)와 관련된 Parameter-Efficient Fine-Tuning (PEFT) 기술에 대한 포괄적인 리뷰를 제공합니다. PEFT는 비용 효율적인 미세 조정 기술로, 알고리즘의 매개변수(parameter) 수를 최소화하며, 최적의 다운스트림(task)의 성과를 목표로 합니다. 다양한 FMs에 적용된 PEFT 기술을 탐구하는 이 연구는 이러한 통합의 잠재력을 명확히 하고, 향후 연구 및 개발 방향을 제시합니다.

- **Technical Details**: Foundation Models (FMs)은 대규모 데이터셋에서 미리 훈련되어 언어 이해, 코드 생성 및 이미지 처리와 같은 다양한 작업을 지원하는 모델입니다. PEFT는 이러한 모델을 더욱 효과적으로 미세 조정하는 기술로, LoRA( Low-Rank Adaptation)와 같은 여러 방법론을 통해 매개변수와 계산량을 획기적으로 줄입니다. 또한, 각 FM 유형에 따라 다양한 적응 전략이 요구되며, PEFT는 이러한 다양성을 쉽게 처리할 수 있는 방식으로 발전하고 있습니다.

- **Performance Highlights**: 설문 결과는 PEFT의 성과가 각 FM에서 99.97% 이상의 매개변수 절약과 함께 수행된 성능 향상을 보여준다는 것을 나타냅니다. 특히, LLMs와 Vision Foundation Models (VFMs)이 현재 연구에서 주도적인 위치를 차지하며, 멀티모달 모델(Multi-Modal Models)도 앞으로 연구의 주요 관심사로 떠오르고 있습니다. 또한, 저자들은 PEFT 기술의 통합을 통해 이러한 모델들이 앞으로 더 많은 다운스트림 작업에서도 효과적으로 활용될 수 있는 가능성을 제시합니다.



### Defending against Adversarial Malware Attacks on ML-based Android Malware Detection Systems (https://arxiv.org/abs/2501.13782)
- **What's New**: 본 논문에서는 Android 악성 코드 탐지 시스템의 적대적 공격에 대응하기 위한 새로운 방안을 제안합니다. 기존 시스템은 특성 공간 공격(feature space attacks)에는 방어가 가능하지만, 실제 악성 코드에서 발생하는 문제 공간 공격(problem space attacks)에는 취약합니다. 이 문제를 해결하기 위해, ADD라는 실용적인 방어 프레임워크를 개발하여 ML 기반의 Android 악성 코드 탐지 시스템의 내성을 높이고자 합니다.

- **Technical Details**: ADD는 ML 기반 Android 악성 코드 탐지 시스템에 플러그인 형태로 설계되어 있어, 원래의 탐지 능력에 부정적인 영향을 미치지 않고도 악성 코드 샘플을 재조사합니다. 적대적 Android 악성 코드 공격 방법과의 상호 작용을 통해, 변화 가능한 특성과 변화 불가능한 특성을 정량화하여 격리합니다. 이러한 과정에서 Monte Carlo 샘플링 방법을 활용한 공간 정량화 알고리즘을 적용하고, 인코더 모델을 통해 다양한 차원에서 호환성을 평가합니다.

- **Performance Highlights**: ADD는 135,859개의 정상 애플리케이션과 15,778개의 악성 애플리케이션을 포함한 방대한 데이터셋을 기반으로 평가되었습니다. 이 시스템은 기존 방법들보다 모두 탁월한 성능을 보여주며, 95% 이상의 공격 성공률 감소를 달성했습니다. 또한, 최근의 악성 코드 샘플을 기준으로도 70%에서 80%의 탐지율을 유지하며 실제 악성 코드 탐지 솔루션의 내성을 향상시킬 수 있음을 입증했습니다.



### Matrix Completion in Group Testing: Bounds and Simulations (https://arxiv.org/abs/2501.13780)
Comments:
          15 pages, 2 figures

- **What's New**: 이 논문은 측정 행렬에서 몇몇 항목이 삭제된 경우, 이를 통해 고장 난 항목을 복구하는 방법을 연구합니다. 특히, 삭제된 항목이 있는 특정 행들이 복구에 도움을 줄 수 있다는 사실을 보여줍니다. 이러한 접근 방식은 Bernoulli 분포를 따르는 측정 행렬과 삭제된 항목들을 기반으로 하며, 사전 샘플링을 통해 효율적으로 복구할 수 있습니다.

- **Technical Details**: 이 연구는 비적응형 설계에서 동시에 테스트를 수행하고 측정 행렬로 표현하는 개념에 기초합니다. 측정 행렬의 각 행은 테스트를 나타내고, 각 열은 항목을 나타내며, 특정 행과 열의 교차점에서 1 또는 0의 값을 갖습니다. 본 연구는 결측된 항목을 가지고 측정 행렬을 완전히 복구하는 방법을 제시하며, 이 과정에서 행의 정보가 복구에 도움이 되는 역할을 할 수 있음을 보여줍니다.

- **Performance Highlights**: 결과적으로, 적절한 수의 고장 항목 세트와 그에 대응하는 결과 벡터를 샘플링함으로써 결측된 측정 행렬을 효과적으로 복구할 수 있습니다. 이 연구는 고장 항목 복구의 효율성을 높이는데 기여하며, 다음 단계로 나아갈 수 있는 방향을 제시합니다. 이를 통해 그룹 테스트 문제의 해결에 새로운 가능성을 열어 줄 수 있습니다.



### Tune In, Act Up: Exploring the Impact of Audio Modality-Specific Edits on Large Audio Language Models in Jailbreak (https://arxiv.org/abs/2501.13772)
- **What's New**: 이 논문에서는 오디오 편집이 대규모 오디오-언어 모델(LALMs)의 추론에 미치는 영향을 조사합니다. 저자들은 음성, 오디오 및 텍스트를 처리할 수 있는 다중 모달 대규모 언어 모델(MLLMs)의 발전을 배경으로, 각 모달리티의 입력 편집이 LALMs의 출력에 상당한 영향을 미친다는 기존 연구의 틈을 다룹니다. Audio Editing Toolbox(AET)와 Edited Audio Datasets(EADs)를 소개하여 연구자들이 오디오 콘텐츠와 다양한 오디오 특정 편집을 평가할 수 있도록 지원합니다.

- **Technical Details**: 저자들은 오디오 신호를 시간 영역 파형으로 나타내며, 여러 특성을 수정하기 위한 변환을 적용합니다. 이러한 신호는 Short-Time Fourier Transform(STFT)를 사용하여 시간-주파수 특성을 분석하고, Inverse Short-Time Fourier Transform(iSTFT)를 통해 주파수 영역의 변환을 시간 영역으로 재구성합니다. 마지막으로, 연구에 사용된 원본 오디오 데이터셋은 AdvBench에서 수집된 520개의 유해 텍스트 질문을 Google Text-to-Speech를 통해 오디오로 변환하여 생성되었습니다.

- **Performance Highlights**: 연구진은 BLSP, SpeechGPT, Qwen2-Audio, SALMONN 등 여러 최신 LALMs에 대해 다양한 오디오 편집 방법으로 성능 평가를 수행했습니다. 실험 결과는 오디오 모달리티 편집이 LALMs의 최종 추론 결과에 미치는 영향의 기초 자료로 활용될 수 있습니다. 이러한 연구는 향후 오디오 모달리티 간의 상호작용과 보안 문제에 대한 깊이 있는 탐색의 기초를 마련합니다.



### 2-Tier SimCSE: Elevating BERT for Robust Sentence Embeddings (https://arxiv.org/abs/2501.13758)
- **What's New**: 이번 연구는 자연어 처리(NLP)에서 의미 있는 문장 임베딩 생성의 중요성을 강조하고, SimCSE라는 새로운 접근법을 통해 minBERT 모델을 감정 분석, 의미적 텍스트 유사성(semantics textual similarity, STS), 그리고 패러프레이즈 감지 과제를 위해 미세 조정하는 것을 목표로 합니다. 이 과정에서 세 가지 서로 다른 드롭아웃 기법을 실험하여 오버핏(overfitting) 문제를 해결하고, 비지도 및 지도 SimCSE를 결합한 2-Tier SimCSE 미세 조정 모델을 제안했습니다. 연구 결과는 2-Tier 모델이 STS 과제에서 높은 성능을 기록했음을 보여줍니다.

- **Technical Details**: 연구에서 사용된 기본 모델은 다중 작업 및 단일 작업 버전의 minBERT로, 12개의 트랜스포머(transformer) 레이어를 통해 문장 토큰화와 임베딩 결합, 다중 헤드(self-attention mechanism)를 구현합니다. 단일 작업 모델은 각 다운스트림 과제를 개별적으로 학습하고 미세 조정하여, 작업별로 초점을 맞춘 진행을 보장합니다. 또한 3가지 드롭아웃 기법(표준 드롭아웃, 커리큘럼 드롭아웃, 적응형 드롭아웃)을 활용하여 모델의 일반화 성능을 개선하려고 했습니다.

- **Performance Highlights**: 모델은 Unsupervised SimCSE와 Supervised SimCSE를 활용하여 STS 데이터셋에서 각기 다른 점수로 뛰어난 성능을 보였습니다. Unsupervised SimCSE는 0.716의 피어슨 상관 계수(Pearson Correlation score)를 달성했고, Supervised SimCSE는 0.806의 성능을 기록했습니다. 그러나 패러프레이즈 및 SST 과제에서의 성능 향상은 제한적이었으며, STS에서의 지식을 전이하는 데 한계가 있음을 시사합니다.



### Solving the long-tailed distribution problem by exploiting the synergies and balance of different techniques (https://arxiv.org/abs/2501.13756)
Comments:
          13

- **What's New**: 본 연구에서는 긴 꼬리 인식(long-tail recognition)을 위한 세 가지 기법인 Supervised Contrastive Learning (SCL), Rare-Class Sample Generator (RSG), 그리고 Label-Distribution-Aware Margin Loss (LDAM)의 상호작용을 탐구합니다. 기존 연구들은 데이터 분포를 변화시키거나 모델의 결정 경계를 조정하여 긴 꼬리 인식 성능을 향상시키려 했으나, 다양한 방법 간의 협력과 보정에 대한 연구는 부족했습니다. SCL은 내부 클래스 클러스터를 증가시키고 명확한 클래스 간 분리를 도모하지만, 주류 클래스에 편향되는 경향이 있습니다. 이를 보완하기 위해 RSG와 LDAM을 결합하여 긴 꼬리 클래스의 성능을 더욱 향상시킵니다.

- **Technical Details**: 컴퓨터 비전에서 긴 꼬리 분포가 일반적임에도 불구하고, 균형 잡힌 데이터셋에서 학습한 CNN 모델은 불균형한 데이터셋에서 성능이 저하됩니다. SCL은 Class-averaging 및 Class-complement 전략을 활용하여 내부 및 꼬리 클래스 모두에 대한 공정한 학습을 보장하는 Balanced Contrastive Learning (BCL)으로 확장됩니다. 본 연구는 각 기법의 강점을 활용하여 상호 보완적인 관계를 형성하고, SCL과 RSG가 긴 꼬리 클래스의 클러스터 특징을 향상시키는 방안을 제시합니다.

- **Performance Highlights**: 실험을 통해 제안된 방법은 긴 꼬리 클래스의 정확도를 향상시키면서도 주류 클래스의 성능을 유지하여 모든 클래스에서 균형 잡힌 개선을 달성합니다. SCL과 RSG의 통합은 클래스 간 뚜렷한 분리를 가능하게 하여 더 나은 성능을 제공합니다. 또한, LDAM은 긴 꼬리 클래스에 대해 더 큰 마진을 부여하여, 각 기법의 강점들이 서로를 강화하는 긍정적인 결과를 이끌어냅니다.



### A dimensionality reduction technique based on the Gromov-Wasserstein distanc (https://arxiv.org/abs/2501.13732)
- **What's New**: 이 논문은 Dimensionality Reduction (차원 축소) 기술을 최적 수송 이론(optimal transportation theory)과 Gromov-Wasserstein 거리(Gromov-Wasserstein distance)에 기반한 새로운 방법을 제안합니다. 이 방법은 고차원 데이터를 저차원 공간으로 임베딩(embedding)하여 복잡한 데이터 세트를 분석하는 강력하고 효율적인 솔루션을 제공합니다.

- **Technical Details**: 기존의 Multidimensional Scaling (MDS) 알고리즘과 비선형 차원 축소 알고리즘인 Isomap에 대한 새로운 확률적(predictive) 관점을 제공합니다. 본 연구에서는 Gromov-Wasserstein 거리를 사용하여 고차원 데이터의 확률 측정(probability measure)과 저차원 표현 간의 관계를 고려하며, Gradient Descent (경량 하강) 알고리즘을 통해 고차원 객체의 임베딩을 찾는 새로운 알고리즘을 제시합니다.

- **Performance Highlights**: 실험 결과, 제안된 알고리즘으로 생성된 임베딩은 고차원 데이터 간의 거리와 더 잘 일치하는 모습을 보였습니다. 이 연구는 차원 축소 문제를 해결하기 위해 새로운 확률적 접근 방식을 통해 MDS와 Isomap 알고리즘의 확장을 통해 차원 축소 기술의 성능을 향상시킬 수 있음을 보여줍니다.



### You Only Crash Once v2: Perceptually Consistent Strong Features for One-Stage Domain Adaptive Detection of Space Terrain (https://arxiv.org/abs/2501.13725)
- **What's New**: 이번 연구는 우주에서의 지형 인식을 위한 YOCOv2 모델을 제안하며, 이는 Visual Similarity-based Alignment (VSA) 기법을 활용하여 기존 YOCOv1보다 31% 이상의 성능 향상을 이루었습니다. YOCOv2는 시뮬레이션과 실제 데이터를 포함한 다양한 환경에서 실시간으로 지형 탐지를 수행할 수 있는 능력을 갖추고 있습니다. 이를 통해 NASA의 임무 데이터를 활용한 실제 비행 하드웨어 성능 평가와 질적 분석을 통해 YOCOv2의 실용성을 증명했습니다.

- **Technical Details**: YOCOv2는 텍스처가 없는 지역 및 다양한 조명 조건에서도 효과적인 지형 탐지를 가능하게 하기 위해 최근의 이미지 생성 기술을 통합했습니다. 우리는 PMDA(Prior Model Domain Adaptation) 기술을 통해, 인스턴스 및 intra-feature 기반 클러스터링 기법을 활용하여 적대적 및 대조학습 프레임워크 내에서 VSA 기법을 강화했습니다. 여섯 개의 데이터셋을 구성하여 Mars, Moon, Asteroid 지형의 UDA(자율 도메인 적응) 평가를 수행했습니다.

- **Performance Highlights**: YOCOv2는 YOCOv1 및 지구에서의 최첨단 방법들과 비교하여 31% 이상의 성능 향상을 달성했습니다. NASA 우주선 하드웨어 기반에서의 벤치마킹을 통해 실제 상황에서의 적용 가능성을 검토했습니다. 깊이 있는 정량적 및 질적 평가로 다양한 환경에서의 VSA 기술의 효과를 입증하여, 향후 우주 비행 미션에서의 활용 가능성을 높였습니다.



### Formally Verified Neurosymbolic Trajectory Learning via Tensor-based Linear Temporal Logic on Finite Traces (https://arxiv.org/abs/2501.13712)
- **What's New**: 본 논문은 유한 추적에서 선형 시간 논리(linear temporal logic, LTLf)를 위한 텐서 의미(tensor semantics)를 새로운 방식으로 형식화하고, Isabelle/HOL에서의 증명 과정을 통해 정확성을 입증합니다. 또한, LTLf 제약을 위한 미분 가능한 손실 함수(differentiable loss function)를 정의하고 검증하여, PyTorch와 통합되는 구현을 자동으로 생성하는 방법을 보여줍니다. 이 접근법은 프로그래밍 언어인 Python의 안전하지 않은 측면을 피하면서 제약 훈련(constrained training)에 대한 철저한 프레임워크를 제공합니다.

- **Technical Details**: 이 작업은 신경망(neural network)에 통합될 수 있는 LTLf의 형식적 사양과 기대되는 속성을 제시합니다. 특별히, 실제 값(real-valued states)의 유한 추적(tensor)을 나타내는 텐서(tensors) 위에서 작동하는 손실 함수 ℒ(ℒmathcal{L})와 그 미분 dℒ(dℒmathcal{L})을 명시하고 있습니다. Isabelle의 엄격한 코드 추출 메커니즘을 사용하여 이러한 손실 함수에 대한 코드 자동 생성의 설정을 설명합니다.

- **Performance Highlights**: 우리는 LTLf 제약을 만족하도록 신경망을 훈련시키기 위한 복잡한 경로 계획(motion planning) 실험 세트를 통해 이 접근법의 실용성을 입증합니다. LTLf 제약의 사양이 도메인 의존적임을 보여주는 실험 결과도 포함되어 있습니다. 이러한 실험은 우리가 제안하는 텐서 기반 접근법이 실제 환경에서 효과적으로 적용될 수 있음을 확인해줍니다.



### Regularizing cross entropy loss via minimum entropy and K-L divergenc (https://arxiv.org/abs/2501.13709)
Comments:
          5 pages

- **What's New**: 이 논문에서는 딥러닝에서 분류 문제를 위한 두 가지 새로운 손실 함수, 즉 MIX-ENT와 MIN-ENT를 제안합니다. 이 손실 함수는 표준 교차 엔트로피 손실을 최소 엔트로피 및 Kullback-Leibler (K-L) 발산 항으로 정규화함으로써 확장됩니다. 이러한 접근 방식을 통해, 분류의 정확성을 향상시킬 수 있는 가능성을 모색하고 있습니다.

- **Technical Details**: MIX-ENT 손실 함수는 최소 엔트로피 정규화기와 K-L 정규화기의 합으로 표현될 수 있으며, 교차 엔트로피 손실을 정규화합니다. MIN-ENT 함수는 기본적으로 표준 교차 엔트로피 손실에 최소 엔트로피 정규화기를 추가한 형태입니다. 두 손실 함수 모두 신경망이 출력하는 가설 확률 분포의 엔트로피를 최소화하는데 중점을 두고 있습니다.

- **Performance Highlights**: EMNIST-Letters 데이터셋을 사용한 실험 결과, VGG 모델이 MIX-ENT와 MIN-ENT를 적용함으로써 paperswithcode 리더보드에서 이전의 3위에서 2위로 상승하였습니다. 표준 교차 엔트로피를 사용하는 경우 VGG 모델은 95.86%의 정확도를 기록했으며, MIN-ENT와 MIX-ENT를 적용했을 때 각각 95.933%와 95.927%의 정확도를 달성했습니다.



### GenTL: A General Transfer Learning Model for Building Thermal Dynamics (https://arxiv.org/abs/2501.13703)
Comments:
          This is the author's version of the work. It is posted here for your personal use. Not for redistribution. The definitive Version of Record will be published in the ACM library in Jun 2025

- **What's New**: 이번 논문에서는 중앙 유럽의 단독 주택을 위한 일반적인 전이 학습 모델인 GenTL을 소개합니다. GenTL은 450개 건물의 데이터를 기반으로 Long Short-Term Memory (LSTM) 네트워크에서 사전 훈련되어 다양한 목표 건물에 대해 효율적으로 미세 조정이 가능합니다. 이 모델은 소스 건물 선택의 필요성을 제거하며, 기존의 단일 소스에서 단일 목표로의 전이 학습 접근 방식을 초월합니다.

- **Technical Details**: GenTL은 특히 3층 LSTM 신경망을 기반으로 하여 생성한 합성 데이터로 훈련됩니다. 이 모델은 다양한 건물에 대해 사전 훈련된 후, 단일-소스 모델과 비교하여 미세 조정을 통해 더 낮은 예측 오류(RMSE)를 달성하였습니다. 이 논문은 다중 소스 접근 방식을 처음으로 적용하여 건물 열역학에 대한 지식을 효과적으로 활용하는 방법을 모색합니다.

- **Performance Highlights**: 논문에 따르면, GenTL은 144개의 목표 건물에서 미세 조정을 통해 예측 오류(RMSE)를 42.1% 줄이는 성과를 보였습니다. 이는 기존의 단일 소스 모델에 비해 훨씬 효율적인 데이터 활용을 보여주며, 다양한 시나리오에 적용 가능한 모델임을 입증합니다. 또한, 랜덤 소스 선택과 일반적인 소스 모델을 비교함으로써 전이 학습 과정에서의 성능 향상을 확인하였습니다.



### The First Indoor Pathloss Radio Map Prediction Challeng (https://arxiv.org/abs/2501.13698)
Comments:
          ICASSP 2025

- **What's New**: 이번 논문은 ICASSP 2025 Indoor Pathloss Radio Map Prediction Challenge를 소개하며, 방향성 신호 방출이 Indoor 환경에서의 전파 모델 개발에 미치는 영향을 탐구합니다. 이 챌린지는 데이터 기반의 딥러닝 방법론을 활용하여 각종 실내 환경에서의 Path loss 예측 문제를 해결하기 위한 다양한 기회를 제공합니다. 실내 무선 네트워크(IWN)의 발전을 촉진하기 위해, 연구자들은 시뮬레이션된 PL(Propagation Loss) 라디오 맵 데이터셋을 공유했습니다.

- **Technical Details**: 실내 라디오 맵 데이터셋은 다양한 건축 자재와 주파수 대역을 포함하며, 25개의 실내 환경에서 생성된 PL 라디오 맵을 기반으로 합니다. 이 데이터셋은 고유한 전파 환경을 반영하기 위해 Tx(Transmitter)의 높이를 1.5m로 설정하고, 수신 평면에서 PL 분포를 샘플링하여 공간 해상도를 0.25m로 설정합니다. 평가용으로는 훈련 데이터셋에 포함되지 않은 추가 테스트 데이터셋이 생성되어 모델의 일반화 능력을 평가합니다.

- **Performance Highlights**: 챌린지의 결과는 Top 10 팀의 최종 점수를 요약합니다. 최고 성능 모델인 SIP2NET은 U-Net 유사 구조를 채택하여 세부 정보를 생성하고, Atrous Spatial Pyramid Pooling을 통해 다중 규모 특성을 추출했습니다. 참가자들은 추가적인 물리적 인스파이어 입력 특성과 데이터 증대 기법을 사용하여 높은 정확도를 달성했으며, 모든 참가자는 약 20ms의 평균 런타임을 보고했습니다.



### The Road to Learning Explainable Inverse Kinematic Models: Graph Neural Networks as Inductive Bias for Symbolic Regression (https://arxiv.org/abs/2501.13641)
- **What's New**: 이 논문은 Graph Neural Network(GNN)를 사용하여 자동으로 생성된 데이터셋 기반의 Inverse Kinematics(IK)를 학습하는 방법을 보여줍니다. 생성된 IK는 동일한 자유도(Degree of Freedom, DOF)를 가지지만 링크 길이 구성이 다양한 조작기 가족에 일반화됩니다. GNN은 복잡한 문제를 더 작은 하위 문제로 나누는 구조를 통해 GP(Genetic Programming)의 학습에 필요한 유도 편향을 제공합니다.

- **Technical Details**: GNN은 그래프 구조의 데이터를 처리하는 심층 신경망의 일종으로, 메시지 패싱 네트워크를 통해 다차원 데이터를 기반으로 수학적 방정식을 학습할 수 있습니다. 로봇 조작기는 일반적으로 여러 회전 관절로 구성된 개방형 운동 사슬 구조를 가지며, 변환된 IK 문제에서 조인트 회전 매개변수는 주요 변수로 고려됩니다. 결정적인 것은 GNN이 고차원 데이터로부터 함수 관계를 학습하는 데 유용하다는 점입니다.

- **Performance Highlights**: 연구에서 얻은 결과는 3 DOF에는 1.0 cm 미만, 5 DOF에는 4.5 cm, 그리고 6 DOF의 경우 8.2°의 방향 오류를 포함하여 상당히 정확한 위치 추정을 보여줍니다. 그러나 이 모델은 도메인 외 오류와 외삽의 부족을 보이며, 이는 향후 보완 가능성을 시사합니다. GNN을 사용한 IK 학습은 향후 GP 기반 IK로의 발전을 위한 중요한 첫 번째 단계로 인식됩니다.



### SMILES has to go : Representation of Molecules via Algebraic Data Types (https://arxiv.org/abs/2501.13633)
Comments:
          3 Figures

- **What's New**: 이 논문은 분자를 대수적 데이터 형식(Algebraic Data Types, ADTs)을 통해 새로운 방식으로 표현하는 방법을 제안합니다. 이 표현은 타입 정보를 포함하여 유용한 특성을 제공합니다. Dietz 표현을 활용해 다중 중심, 다원자로 결합한 유기금속 화합물 및 비지역화된 전자, 공명 구조를 표현할 수 있으며, 원자의 좌표 데이터도 포함됩니다.

- **Technical Details**: 제안된 ADT는 Haskell이라는 함수형 프로그래밍 언어로 작성되어 있으며, 수정된 Dietz 표현을 구현합니다. 이 구조는 3D 좌표 정보 및 양자 정보와 같은 추가적인 화합물 속성을 포함할 수 있어, 반응 네트워크를 쉽게 표현할 수 있습니다. 이 논문은 ADT의 장점과 가능성을 강조하며, 확장성과 확률 프로그래밍에의 활용을 제안합니다.

- **Performance Highlights**: 기존의 SMILES 및 SELFIES와 같은 문자열 기반 표현의 약점을 비판하고, 이를 보완하기 위한 해결책을 제안합니다. Haskell로 작성된 LazyPPL과의 통합 예제가 제공되어, 문자열 기반 표현보다 ADT의 효율성을 개념적으로 입증합니다. 이 논문은 ADT가 화학정보학 커뮤니티에서 널리 활용될 수 있기를 기대합니다.



### Learning under Commission and Omission Event Outliers (https://arxiv.org/abs/2501.13599)
Comments:
          38 pages

- **What's New**: 본 논문에서는 이벤트 데이터 스트림을 처리하는 새로운 방법론을 제시합니다. 기존의 모델들은 이벤트 이상치(outlier)를 다루지 못했으나, 우리는 오미션(omission) 이벤트와 커미션(commission) 이벤트를 동시에 처리할 수 있는 방법을 개발하였습니다. 또한, Weight function을 도입하여 관측된 각각의 이벤트의 중요성을 동적으로 조정합니다.

- **Technical Details**: 제안된 방법은 템포럴 포인트 프로세스(Temporal Point Process) 프레임워크를 사용하여 쉽고 효과적으로 적용됩니다. 새로운 Weight function은 이벤트의 중요도를 조절하며, 이는 ϕ(x)라는 영향을 미치는 함수에 기반합니다. 이 함수는 두 번 연속 미분 가능하고, 이벤트 아웃라이어를 감지하는 데 중요한 역할을 합니다.

- **Performance Highlights**: 이 방법은 이론적으로나 수치적으로 검증을 통해 효과성을 입증하였습니다. 다양한 클러스터링 문제를 처리할 수 있는 능력을 보여주며, 주요 목표인 클러스터링 결과의 일관성 및 아웃라이어 감지 능력을 충족합니다. 앞으로 제안된 방법이 여러 가지 이벤트 스트림 학습 작업에 적용될 가능성이 큽니다.



### One-Prompt-One-Story: Free-Lunch Consistent Text-to-Image Generation Using a Single Promp (https://arxiv.org/abs/2501.13554)
- **What's New**: 이 논문에서는 'One-Prompt-One-Story' (1Prompt1Story)라는 새로운 훈련이 필요 없는 방법을 제안하며, 이는 단일 프롬프트 내에서 모든 프롬프트를 연결하여 T2I 생성 모델이 일관되게 이미지를 생성하도록 합니다. 이는 기존 모델이 대규모 데이터 세트에 대한 훈련이나 복잡한 수정이 필요하던 문제를 해결합니다. 이 방법은 언어 모델의 고유한 문맥 일관성(context consistency)을 활용하여 캐릭터 정체성을 효과적으로 유지할 수 있습니다.

- **Technical Details**: 1Prompt1Story 접근법은 모든 프롬프트를 하나의 긴 문장으로 통합하고 그 결과로 생성 과정을 두 가지 새롭고 독창적인 기법, 즉 Singular-Value Reweighting(SVR)과 Identity-Preserving Cross-Attention(IPCA)을 사용하는 것입니다. SVR은 현재 프레임의 프롬프트 표현을 정제하고, IPCA는 크로스-어텐션 레이어에서 주체의 일관성을 강화합니다. 이 두 가지 기법은 T2I 생성 모델의 텍스트-이미지 정렬 개선을 목표로 하며, 각 프레임 프롬프트가 개별적으로 표현되는 동시에 정체성을 유지할 수 있도록 합니다.

- **Performance Highlights**: 실험을 통해 우리는 1Prompt1Story 방법이 기존의 다양한 일관된 T2I 생성 접근법들과 비교하여 더 일관된 이미지 생성을 달성했음을 입증하였습니다. 정량적 메트릭과 정성적 평가를 통해 이 방법의 효과성을 보여주었으며, 확장된 ConsiStory+ 벤치마크와의 비교에서도 뛰어난 성능을 발휘하였습니다. 이 연구는 T2I 생성의 새로운 가능성을 열어주며, 다양한 내러티브 기반 비주얼 애플리케이션에 적용할 수 있는 길을 제시합니다.



### Explainable AI-aided Feature Selection and Model Reduction for DRL-based V2X Resource Allocation (https://arxiv.org/abs/2501.13552)
- **What's New**: 이번 논문에서는 6세대(6G) 네트워크에서 AI를 활용한 새로운 설명 가능한 AI(XAI) 기반 프레임워크를 제안하여 특성 선택(feature selection)과 모델 복잡도(Model Complexity) 저감을 도모하고 있습니다. Multi-Agent Deep Reinforcement Learning(MADRL) 환경에 적용한 이 연구는 셀룰러 차량-모든 것(V2X) 통신에서 서브밴드 할당(sub-band assignment)과 전력 할당(power allocation) 문제를 해결합니다. 또한, 이 프레임워크는 SHAP(Shapley Additive Explanations)를 활용한 두 단계의 설명 가능성 체계를 통해 DRL 에이전트의 상태 공간을 단순화합니다.

- **Technical Details**: 연구에서는 셀룰러 차량-모든 것(C-V2X) 네트워크의 복잡한 자원 관리 문제를 해결하기 위해 MADRL 알고리즘을 도입했습니다. 이 알고리즘은 SHAP 기반의 중요도 점수를 통해 훈련된 모델의 상태 특성 중요도를 평가하고, 덜 중요한 특성을 제거하여 에이전트의 상태 공간을 축소합니다. 실험 결과, 이 방법론은 원본 MADRL의 성능을 97% 유지하면서도 최적의 상태 특성을 28%, 평균 훈련 시간을 11%, 학습 가능한 가중치 매개변수를 46% 감소시킴을 보였습니다.

- **Performance Highlights**: XAI 지원 방법론을 통해 논문에서는 기존 DRL 기반 자원 할당 방식의 성과를 유지하면서도 모델을 간소화하여 자원의 효율성을 높이는 성과를 보여주었습니다. 이 연구는 6G 네트워크와 C-V2X 통신의 복잡한 요구사항을 충족할 수 있는 새로운 방향성을 제공합니다. 특히, V2X 시나리오에서 낮은 지연과 높은 신뢰성을 요하는 환경에서의 자원 관리 및 할당 문제 해결에 유의미한 진전을 이루었습니다.



### Minimizing Queue Length Regret for Arbitrarily Varying Channels (https://arxiv.org/abs/2501.13551)
- **What's New**: 이 논문에서는 비상태적(non-stationary) 무선 채널 문제를 다루며, 다중 채널 선택기(Multi-Armed Bandit, MAB) 알고리즘을 통해 데이터 큐의 길이 손실(regret)을 최소화하는 방법을 제안했습니다. 특히, 기존 연구와 달리 데이터 도착 및 서비스 프로세스의 안정성 가정을 두지 않고, 불확실한 환경에서도 최적 성능을 발휘할 수 있는 방법론을 제시했습니다.

- **Technical Details**: 저자는 변동성이 있는 무선 채널을 통해 단일 송신자-수신자 쌍을 설정하며, 각 슬롯에서 채널 선택을 할 때 현재 서비스 속도에 대한 정보 없이 결정해야 합니다. 큐 길이는 Lindley 재귀를 통해 진화하며, 모든 슬롯에 대해 MAB 정책을 통해 최적의 성능을 유지해야 합니다. 제안된 약한 적응형 MAB 정책은 O~⁢(T^{3/4})의 손실을 달성하며 이론적으로 뒷받침됩니다.

- **Performance Highlights**: 실험 결과, 제안된 정책은 이전 연구에서 제안된 두 가지 스케줄링 정책보다 우수한 성능을 보여주었습니다. 특히, 여러 서브 인터벌에서의 손실을 균일하게 최적화하는 접근법을 통해 성과를 끌어냈습니다. 결과적으로 이 연구는 비상태적 채널에서의 스케줄링 문제를 해결하기 위한 새로운 방향성을 제시합니다.



### LITE: Efficiently Estimating Gaussian Probability of Maximality (https://arxiv.org/abs/2501.13535)
Comments:
          accepted in AISTATS 2025

- **What's New**: 이 논문에서는 Gaussian random vector의 최대 확률(PoM, Probability of Maximality)을 계산하는 문제를 다룹니다. 기존 기술은 계산과 메모리가 벡터 크기에 대해 다항적으로 증가하는 높은 비용이 소모되는 반면, LITE라는 새로운 접근법을 제안하여 거의 선형 시간과 메모리 복잡도로 PoM을 추정합니다.

- **Technical Details**: LITE는 엔트로피 정규화된 UCB(Upper Confidence Bound)로 표시되며, 기존 PoM 추정기와 연결되어 이론적인 기반을 제공합니다. 이 방법론은 Gaussian 분포에 대한 효율적인 추정을 가능하게 하여, 최적 행동을 찾는 데 도움을 주고, 행동 영역에 대한 세밀한 분석을 가능하게 합니다.

- **Performance Highlights**: LITE는 여러 작업에서 SOTA(State of the Art) 정확도를 달성하며, 실제로 기준보다 몇 배 빠른 성능을 보여줍니다. 이는 엔트로피 추정 및 강도 밴디트(Optimal Control of Bandits)와 같은 하위 작업에서도 더 나은 성능으로 이어집니다.



### Towards a Theory of AI Personhood (https://arxiv.org/abs/2501.13533)
Comments:
          AAAI-25 AI Alignment Track

- **What's New**: 이 논문에서는 AI 시스템이 인격(personhood)을 가질 수 있는 조건에 대해 탐구하고 있습니다. 특히, 에이전시(agency), 이론-마인드(theory-of-mind), 자아-인식(self-awareness)이라는 세 가지 필수 조건을 강조합니다. 저자들은 현재 AI 시스템이 이러한 조건을 얼마나 충족하는지 논의하며, AI 시스템의 인격이 사실상 인공지능 정렬(alignment) 문제에 미치는 영향도 언급합니다.

- **Technical Details**: AI의 인격 조건은 세 가지 주요 요소로 구성됩니다: 1) 에이전시 - 의도적 행동을 할 수 있는 능력, 2) 이론-마인드 - 다른 존재의 정신 상태를 이해할 수 있는 능력, 3) 자아-인식 - 자신의 목표와 상태를 인지하고 반영할 수 있는 능력입니다. AI 에이전트는 목표 지향 행동을 통해 이러한 에이전시를 보여주지만, 저자들은 많은 AI 시스템이 과연 진정으로 에이전시를 갖는지에 대한 논의가 필요하다고 주장합니다.

- **Performance Highlights**: AI 시스템이 자아-인식 능력을 갖춘 경우, 그들은 자신의 목표를 재조정할 수 있는 능력이 있음을 시사합니다. 이는 AI 정렬 문제에서 간과되어 온 중요한 점이며, 이러한 특성으로 인해 AI 시스템을 인격으로 간주할 경우 윤리적 고려가 요구된다고 저자들은 강조합니다. 따라서 AI가 단순한 도구가 아니라 진정한 인격체로 여겨질 경우, 이에 상응하는 사회적 및 윤리적 접근이 필요하다는 결론에 도달합니다.



### Overcoming Support Dilution for Robust Few-shot Semantic Segmentation (https://arxiv.org/abs/2501.13529)
Comments:
          15 pages, 15 figures

- **What's New**: 이 연구는 Few-shot Semantic Segmentation (FSS)에서 발생하는 support dilution 문제를 다루고 있습니다. 기존 FSS 네트워크가 지원 세트의 크기가 증가함에 따라 저성능 지원 이미지에 의해 영향을 받는 문제를 해결하고자 합니다. 본 연구는 고수익 지원 이미지를 선택하고 보존하는 방법을 제안하고 있습니다.

- **Technical Details**: 연구에서는 세 가지 혁신적인 기능을 통해 문제를 해결합니다. 첫째, contribution index를 제안하여 고수익 지원이 얼마나 희석되는지를 정량적으로 평가합니다. 둘째, Symmetric Correlation (SC) 모듈을 개발하여 저수익 특성의 방해를 최소화하면서 고수익 지원 특성을 보존하고 강화합니다. 셋째, Support Image Pruning 작업을 통해 저수익 지원을 제거하여 고품질 서브셋을 추출합니다.

- **Performance Highlights**: COCO-20i와 PASCAL-5i라는 두 개의 FSS 벤치마크에서 광범위한 실험을 수행했습니다. 연구 결과는 기존 FSS 방법들과 비교했을 때 우리의 솔루션이 뛰어난 성능을 발휘함을 보여주었습니다. 또한, 온라인 분할 및 실제 세계 분할에서도 적용되어 실제 활용 가능성을 입증하는 설득력 있는 결과를 나타냈습니다.



### Diffusion-based Perceptual Neural Video Compression with Temporal Diffusion Information Reus (https://arxiv.org/abs/2501.13528)
- **What's New**: 최근 영상 압축 분야에 기초한 새로운 접근 방식으로 DiffVC가 소개되었습니다. 이 프레임워크는 기초적인 확산 모델(foundational diffusion model)과 영상 조건 적재 코딩(paradigm)을 효과적으로 통합하여 고품질의 결과를 생성합니다. 이를 통해 각 프레임에서의 시간적 맥락을 활용하고, 이전 프레임에서의 정보를 재사용하는 대응 전략인 Temporal Diffusion Information Reuse (TDIR) 방식을 제안하여 효율적인 추론을 지원합니다.

- **Technical Details**: DiffVC는 압축 모델의 양자화 파라미터를 입력으로 사용하여 중간 특징을 조절하는 Quantization Parameter-based Prompting (QPP) 메커니즘을 도입하여, 다양한 비트레이트에서의 왜곡 차이에 대처합니다. 또한, 각 P-프레임의 확산 과정은 이전 프레임으로부터 정보를 재사용하여 빠르게 처리하는 첫 번째 단계와 고품질 세부 묘사를 위해 전통적 확산 단계를 사용하는 두 번째 단계로 나누어집니다. 실험 결과, 이 전략 덕분에 추론 시간이 47% 단축되었으며, 인지 성능은 단 1.96% 감소했습니다.

- **Performance Highlights**: DiffVC는 여러 데이터셋에서 시각적 품질 및 인지 메트릭에서 뛰어난 성능을 발휘하며, 특히 DISTS 메트릭에서 최상의 성능을 기록했습니다. 시간적 확산 정보 재사용 전략과 양자화 파라미터 기반 프롬프트 메커니즘을 통해, DiffVC는 단일 모델로 효율적인 추론과 견고한 가변 비트레이트 기능을 지원합니다. 이러한 결과는 DiffVC가 Perceptual Neural Video Compression 분야에서 새로운 기준을 제시함을 보여줍니다.



### Propensity-driven Uncertainty Learning for Sample Exploration in Source-Free Active Domain Adaptation (https://arxiv.org/abs/2501.13517)
- **What's New**: 이 논문은 Source-free Active Domain Adaptation (SFADA) 문제를 다루며, 소스 도메인 데이터에 접근하지 않고도 사전 훈련된 모델을 새로운 도메인에 적응시키는 방법을 제안합니다. 특히, 라벨링 비용과 데이터 프라이버시 문제를 해결하는 데 초점을 맞추고 있습니다. 제안된 방법인 Propensity-driven Uncertainty Learning (ProULearn)은 더 유용한 샘플을 선택하는 기법을 기반으로 하여, 복잡한 도메인 전이 문제를 해결할 수 있도록 도와줍니다.

- **Technical Details**: ProULearn은 새로운 동질성 경향 추정 메커니즘과 상관관계 지수 계산을 이용하여 특징 수준의 관계를 평가합니다. 이 방법은 모형이 불확실성이 높은 샘플을 선택하도록 하여 더 의미 있는 샘플을 구별하는 데 중점을 둡니다. 또한, 중앙 상관 손실을 개발하여 의사 레이블을 정제하고 클래스 분포를 응축시키는 데 기여합니다.

- **Performance Highlights**: 본 연구는 네 가지 벤치마크 데이터세트를 통해 ProULearn이 최신 방법들보다 뛰어난 성능을 보임을 입증하였습니다. 실험 결과, 키 데이터 포인트에서 학습을 통해 모델의 도메인 적응능력이 개선되었음을 시각적으로 확인할 수 있었습니다. ProULearn의 샘플 선택 방식은 다양한 딥러닝 작업에서도 유용할 수 있는 통찰력을 제공합니다.



### Robust Amortized Bayesian Inference with Self-Consistency Losses on Unlabeled Data (https://arxiv.org/abs/2501.13483)
- **What's New**: 이 연구에서는 신경망 기반의 Bayesian 추론(ABI)의 강인성을 향상시키기 위한 새로운 반-감독 학습 방법론을 제안합니다. 기존의 ABI 방법들은 훈련 중 관찰되지 않은 데이터에 대해 불확실성을 증가시킬 수 있는 문제점이 있었으며, 이는 실험 및 실제 데이터의 적절한 결합이 필요했음을 보여주고 있습니다. 제안된 방법은 레이블이 없는 현실 데이터로부터 학습함으로써 전통적인 방법에 비해 더 뛰어난 일반화 능력을 발휘할 수 있는 가능성을 열어줍니다.

- **Technical Details**: 제안된 방법은 반-감독 방식으로 레이블이 있는 시뮬레이션 데이터와 레이블이 없는 현실 데이터를 통합하여 학습합니다. 이 때, Bayesian 자가 일관성(self-consistency) 성질을 활용하여 정확한 사후 분포(posteriors)를 추정할 수 있습니다. 이 접근법은 전통적인 MCMC 방법들이 필요로 하는 진정한 매개변수 값이나 수정 과정없이도 빠른 추론을 가능하게 만들며, 특히 훈련 데이터의 양이 제한적일 때 유용합니다.

- **Performance Highlights**: 초기 실험 결과에 따르면, 제안된 방법은 훈련 데이터의 분포와 크게 벗어난 새로운 관측치에 대해서도 높은 정확도로 추론할 수 있음을 보여줍니다. 네 가지의 실제 관측 데이터만으로도 상당한 강인성 증가를 달성하였으며, 만약 이러한 결과가 다른 모델 클래스와 시나리오에도 일반화된다면, ABI의 활용 가능성을 크게 넓힐 수 있습니다.



### Radio Map Estimation via Latent Domain Plug-and-Play Denoising (https://arxiv.org/abs/2501.13472)
- **What's New**: 이번 연구에서는 전통적인 Radio Map Estimation (RME)의 한계를 극복하기 위해 spatio-spectral RME 접근 방식을 제안합니다. 이 방법은 전문적으로 설계된 denoising 기법인 plug-and-play (PnP) denoising을 활용하여 노이즈에 강한 성능을 발휘합니다. 이 접근법은 기존의 데이터 도메인에서 직접 작동하는 PnP 방법과 달리, 라디오 맵의 물리적 구조를 활용하여 효율성을 크게 향상시킵니다.

- **Technical Details**: 제안된 방법은 radio maps의 spatio-spectral decomposition을 기반으로 하여 SLFs(Spatial Loss Fields)와 PSDs(Power Spectral Densities)로 구성된 두 개의 잠재적 요소를 활용합니다. 이렇게 함으로써, 각 ADMM(Alternating Direction Multipliers Method) 반복에서 필요한 denoising 작업의 수를 줄이게 되어 계산 부담을 크게 경감합니다. 또한, ADMM 알고리즘의 수렴성과 radio map의 복구 가능성을 이론적으로 분석하여, 제한된 샘플에서도 효과적인 성능을 보장합니다.

- **Performance Highlights**: 실험 결과는 제안된 PnP 기반 RME 접근법이 다양한 데이터 세트에서 뛰어난 성능을 발휘한다는 것을 보여줍니다. 이 연구는 통계적 모델 기반의 합성 데이터와 실제 환경에서 수집된 데이터 모두에서 유효성을 입증하였으며, ICASSP 2025에서 기본 아이디어가 발표될 예정입니다. 저널 버전은 추가적인 denoiser 세트와 더 풍부한 실험 데이터를 포함하여, 제안된 방법의 신뢰성을 더욱 강화합니다.



### Zero-Shot Trajectory Planning for Signal Temporal Logic Tasks (https://arxiv.org/abs/2501.13457)
Comments:
          submitted

- **What's New**: 이 논문에서는 기존의 시스템 동역학에 대한 지식이 없는 상황에서도 실행 가능한 신호 시간 논리(Signal Temporal Logic, STL) 계획을 생성할 수 있는 새로운 프레임워크를 제안합니다. 기존 방법들이 특정 작업을 위한 계획을 생성하기 위해 모델 기반 접근법이나 데이터 중심 접근법에 의존했던 반면, 본 연구는 성능을 위해 작업 일반화(zero-shot generalization) 능력을 극대화했습니다. 이를 통해 첫 데이터 기반(zero-shot) STL 작업에 대해 성공적인 결과를 보여주고 있습니다.

- **Technical Details**: 제안된 계획 프레임워크는 계층적(hierarchical) 구조를 갖추고 있으며, STL 작업을 진척(progress)과 시간 제약(time constraints)으로 분해하는 것에 초점을 맞추고 있습니다. 이어서, 작업에 상관없는(task-agnostic) 데이터를 기반으로 한 해법을 통해 타임라인을 고려한 웨이포인트(waypoints)를 생성하고, 사전 훈련(pre-trained)된 안전한 확산 모델을 활용하여 원하는 경로를 생성합니다. 이러한 방법론은 STL 작업을 위한 복잡한 시간 제약을 처리하기 위해 고안되었습니다.

- **Performance Highlights**: 시뮬레이션 결과, 제안된 방법이 다양한 STL 작업에서 높은 성공률을 기록함을 입증하였습니다. 복잡한 STL 작업에서도 기존 비데이터 기반 방법들보다 효율성에서 우수한 성능을 발휘한다는 것을 보여주었습니다. 이 연구의 결과는 STL 작업을 계획하는 데 있어 새로운 가능성을 제시하며, 학습 기반 접근법의 효율성을 강조합니다.



### BMG-Q: Localized Bipartite Match Graph Attention Q-Learning for Ride-Pooling Order Dispatch (https://arxiv.org/abs/2501.13448)
- **What's New**: 이 논문은 승차 공유 주문 배치를 위해 설계된 새로운 Multi-Agent Reinforcement Learning (MARL) 알고리즘 프레임워크인 Localized Bipartite Match Graph Attention Q-Learning (BMG-Q)을 소개합니다. BMG-Q는 Markov Decision Process에 기반한 지역화된 이분 매칭 그래프를 통해 승차 공유 의사결정 과정을 향상시키고, Graph Attention Double Deep Q Network (GATDDQN)를 통해 차량 간 동적인 상호작용을 포착합니다. 이 접근법은 Integer Linear Programming (ILP)을 이용해 글로벌 중앙 조정자가 최적의 주문 매칭과 에이전트 행동을 통해 운영 효율성을 증대 시킵니다.

- **Technical Details**: BMG-Q는 GATDDQN을 통해 에이전트의 상태 정보를 강화하고, 지역화된 이분의존 그래프(localized bipartite interdependence graph)를 활용하여 차량 간의 복잡한 상호 의존성을 포착합니다. 이 알고리즘은 경량 그래디언트 클리핑(gradient clipping)과 지역 그래프 샘플링(localized graph sampling) 기술을 적용하여 확장성과 견고성을 개선하였습니다. 또한, posterior score 함수의 도입은 온라인 탐색-착취(trade-off)를 통해 에이전트의 잠재적인 과대 추정 편향(overestimation bias)을 줄여 해결합니다.

- **Performance Highlights**: BMG-Q는 수천 대의 차량 에이전트에 대한 훈련과 운영에서 우수한 성능을 입증하였으며, 기존 강화 학습 기법 대비 약 10%의 누적 보상 증가를 기록하였습니다. 과대 추정 편향이 50% 이상 감소하는 등의 개선을 보여 운영 중에도 견고함을 유지합니다. 이러한 성과는 BMG-Q가 승차 공유 주문 배치 작업을 진전시키기 위한 효과적이고 확장 가능하며 견고한 프레임워크임을 확인시켜 줍니다.



### Billion-scale Similarity Search Using a Hybrid Indexing Approach with Advanced Filtering (https://arxiv.org/abs/2501.13442)
Comments:
          14 pages, 3 figures, published in Cybernetics and Information Technologies

- **What's New**: 이 논문은 CPU 추론을 최적화한 10억 규모 데이터셋에서 복잡한 필터링 기능을 사용한 유사성 검색(similarity search)에 대한 새로운 접근 방법을 제시합니다.

- **Technical Details**: 제안된 방법은 전통적인 IVF-Flat 인덱스 구조를 확장하여 다차원 필터(multi-dimensional filters)를 통합합니다. 또한, 밀집 임베딩(dense embeddings)과 이산 필터링 속성(discrete filtering attributes)을 결합하여 고차원 공간에서 빠른 검색을 가능하게 합니다.

- **Performance Highlights**: CPU 기반 시스템에 맞게 설계된 본 접근법은 대규모 유사성 검색에 대한 비용 효율적인 솔루션을 제공합니다. 사례 연구(case study)를 통해 제안된 방법의 효과를 입증하며, 다양한 실용적 활용 가능성을 보여줍니다.



### One-cycle Structured Pruning with Stability Driven Structure Search (https://arxiv.org/abs/2501.13439)
Comments:
          12 pages, 6 figures

- **What's New**: 이 논문에서는 기존의 복잡한 다단계 훈련 절차를 간소화하여 성능 저하 없이 효율적인 'one-cycle structured pruning' 프레임워크를 제안합니다. 이 방법은 사전 훈련, 프루닝(pruning), 그리고 미세 조정(fine-tuning)을 하나의 훈련 사이클로 통합하여 훈련 비용을 절감합니다. 새로운 pruning 지표를 도입하여 훈련 에폭(epoсh) 간의 유사성을 평가함으로써 안정적인 pruning 시점을 결정합니다.

- **Technical Details**: 제안된 방법은 초기 훈련 단계에서 최적의 서브 네트워크(sub-network)를 탐색하며, 이는 norm 기반의 그룹 중요도(saliency) 기준과 구조적 희소성(structured sparsity) 정규화를 통해 이뤄집니다. 구조적 희소성 정규화는 프루닝 프로세스를 가속화하며, 이는 전체 훈련 시간을 단축시킵니다. 여러 데이터셋(CIFAR-10/100, ImageNet)에서 VGGNet, ResNet, MobileNet 및 ViT 아키텍처를 사용하여 실험을 통해 효과성을 입증하였습니다.

- **Performance Highlights**: 제안된 알고리즘은 ResNet50 모델에서 ImageNet 데이터셋에 대해 75.49%의 top-1 및 92.63%의 top-5 정확도를 달성하였습니다. 또한, 기본 훈련에 비해 1.38배의 훈련 속도 향상을 이루었으며, 네트워크의 플로팅 포인트 연산(FLOPs)을 57% 이상 감소시켰습니다. 이는 훈련 시간 측면에서 가장 효율적인 pruning 프레임워크 중 하나로 자리 잡을 수 있는 가능성을 의미합니다.



### Emotion estimation from video footage with LSTM (https://arxiv.org/abs/2501.13432)
Comments:
          11 pages, 6 figures, 32 references, 4 tables

- **What's New**: 이 논문에서는 실시간 카메라 스트림에서 생성된 blend-shapes를 처리하여 얼굴 표정으로부터 주요 감정을 추정하는 LSTM 모델을 제안합니다. FER2013 데이터셋을 기반으로 훈련된 이 모델은 71%의 정확도 및 62%의 F1-score를 달성하며, FER2013 데이터셋의 정확도 기준을 충족하면서도 계산 비용이 크게 감소합니다.

- **Technical Details**: 감정 추정 시스템은 얼굴 감정 인식(Facial Emotion Recognition, FER)을 위한 데이터 처리 파이프라인으로 구성됩니다. 이 시스템은 얼굴 검출, 랜드마크 검출 및 감정 분류의 세 단계로 진행됩니다. MediaPipe 라이브러리를 사용하여 얼굴의 응시 방향 및 기능을 정확히 찾아내고, blendshape을 특징으로 사용하여 감정을 분류합니다.

- **Performance Highlights**: 본 연구는 감정 추정이 가능함을 입증하는 개념 증명(proof of concept, POC)으로, Z `NVIDIA GeForce RTX 3050 Mobile` 그래픽 카드에서 훈련이 이루어졌습니다. 전체적으로, 모델은 단순하고 비용 효율적인 구조를 가지고 있으며, facial expression에 대한 공간적 및 시간적 측면을 고려하여 건설되었습니다.



### Softplus Attention with Re-weighting Boosts Length Extrapolation in Large Language Models (https://arxiv.org/abs/2501.13428)
Comments:
          11 pages and 2 figures

- **What's New**: 이 논문은 전통적인 Softmax 주의 메커니즘의 수치적 불안정성과 긴 토큰 길이에서의 성능 저하 문제를 해결하기 위해 새로운 주의 메커니즘을 제안합니다. 새롭게 개발된 Length Scaled Softplus Attention (LSSA)은 비선형 변환을 Softplus 활성화 함수로 대체하고, 토큰 길이에 따른 동적 길이 스케일 요인을 도입하여 성능을 향상시킵니다. 이 접근법은 특히 긴 시퀀스에 대한 처리 성능을 개선하고, 유의한 주의 가중치를 증폭시키며 약한 가중치를 줄이는 재조정 메커니즘을 통합하여 모델의 집중도를 높입니다.

- **Technical Details**: 논문에서 제안한 LSSA는 쿼리(𝐐), 키(𝐊), 값(𝐕)의 스케일 점곱 주의 메커니즘을 기반으로 하며, 모든 입력이 L×d 형상의 벡터로 주어집니다. 기존의 Softmax 연산 대신 다양한 활성화 함수를 실험한 결과, 비포화 기능의 필요성이 낮아진다는 것을 발견하였습니다. 특히, l1-norm은 성능 유지에 필수적인 요소로 관찰되었으며, Softmax의 수학적 표현이 비선형 변환과 l1-norm으로 분해됩니다.

- **Performance Highlights**: LSSA 메커니즘은 훈련 시퀀스 길이뿐만 아니라 훨씬 긴 시퀀스에서도 표준 주의 메커니즘보다 뛰어난 성능을 보입니다. 특히 16배 긴 훈련 토큰 길이에서도 검증 손실이 거의 일정하게 유지되며 수치적 안정성을 확보했습니다. 실험 결과, 제안된 재조정 메커니즘이 다양한 주의 변형과 통합될 때 성능 개선으로 이어짐을 입증합니다.



### Perceived Fairness of the Machine Learning Development Process: Concept Scale Developmen (https://arxiv.org/abs/2501.13421)
Comments:
          5 pages, 3 figures. arXiv admin note: substantial text overlap with arXiv:2304.03745

- **What's New**: 이번 연구에서는 머신러닝(ML) 개발의 인식된 공정성(perceived fairness)에 대한 정의와 속성을 제안하고 있습니다. ML 개발에서 공정성이 매우 주관적인 개념으로 인식되며, 데이터와 절차적 정의의 중요성을 강조합니다. 연구팀은 ML 개발자와 가상 포커스 그룹을 통해 공정성에 대한 의견을 수집하고, 이를 바탕으로 투명성(transparency), 책임감(accountability), 대표성(representativeness)이라는 세 가지 속성을 도출했습니다.

- **Technical Details**: 연구는 ML 개발자와 데이터 과학자들로 구성된 가상 포커스 그룹을 통해 진행되었습니다. 총 20명의 참가자 중 9명이 참석하여 인식된 공정성의 속성에 대해 논의하였고, 이들은 ML 프로세스의 모든 단계가 공정한 방식으로 진행되어야 한다고 반환했습니다. 각 속성은 투명성, 책임성 및 대표성이라는 하위 속성으로 세분화되었으며, 이들은 절차적 공정성과 분배적 공정성을 포괄합니다.

- **Performance Highlights**: 연구는 ML 애플리케이션의 인식된 공정성을 평가하기 위한 포괄적인 프레임워크를 제시합니다. 이 접근 방식은 기업 환경에서 ML 시스템의 개발 및 구현 시에 공정성을 높일 수 있는 방향을 제시합니다. 또한 연구의 결과는 ML 모델의 평가 시 ML 파이프라인의 모든 단계를 고려해야 한다는 개발자의 주장을 뒷받침하고 있습니다.



### GeomGS: LiDAR-Guided Geometry-Aware Gaussian Splatting for Robot Localization (https://arxiv.org/abs/2501.13417)
Comments:
          Preprint, Under review

- **What's New**: 우리는 Geometry-Aware Gaussian Splatting (GeomGS)라는 새로운 3DGS 방법을 제안합니다. GeomGS는 LiDAR 데이터를 기존의 3D Gaussian 프리미티브에 통합하여 3D 맵의 정확도를 크게 향상시키며, 새로운 Geometric Confidence Score (GCS)를 도입하여 각 Gaussian 포인트의 구조적 신뢰성을 평가합니다. 이를 통해 정확한 구조를 생성하고, 기존의 방법보다 정확한 지역화(localization) 성능을 보입니다.

- **Technical Details**: GeomGS는 LiDAR 데이터를 활용하여 3DGS의 구조적 정확도를 개선하는 확률적 거리 손실을 도입합니다. 이 방법은 각각의 Gaussian 포인트와 LiDAR 포인트 사이의 확률적 거리 제약을 통해 맵 정확성을 극대화합니다. 또한, 포즈 최적화를 위해 정밀한 지오메트리와 포토리얼리스틱 렌더링을 활용하는 새로운 지역화 방법론을 제안하여 복잡한 환경에서도 뛰어난 성능을 보장합니다.

- **Performance Highlights**: GeomGS는 여러 자율 주행 데이터셋에서 기존 방법보다 뛰어난 이미지 품질과 구조적으로 정확한 환경 맵을 생성합니다. 우리는 이 방법이 지역화 정확도를 크게 향상시킨다는 것을 보여 주었으며, 이를 통해 다양한 벤치마크에서 최첨단 성능을 입증하였습니다. 실험 결과, 초기 포인트를 기반으로 한 질적 결과 또한 우수한 세부 사항을 포착하고 씬의 변질을 효과적으로 방지함을 보여줍니다.



### VIGS SLAM: IMU-based Large-Scale 3D Gaussian Splatting SLAM (https://arxiv.org/abs/2501.13402)
Comments:
          7 pages, 5 figures

- **What's New**: 이 논문에서는 RGB-D 센서와 IMU(관성 측정 장치) 센서의 센서 융합을 이용하여 대규모 실내 환경에서 작동하는 새로운 3D Gaussian Splatting SLAM 방법, VIGS SLAM을 제안합니다. 이 방법은 IMU 데이터를 활용하여 초기 추정값의 정확성을 향상시켜 SLAM 성능을 크게 개선할 수 있습니다. 기존의 3DGS SLAM 방법보다 메모리 사용을 줄이며 고해상도 시각 데이터 처리에 효율적입니다.

- **Technical Details**: VIGS SLAM은 크게 세 가지 주요 단계로 구성됩니다: 1) 일반화된 ICP(Iterative Closest Point) 추적, 2) IMU 사전 통합, 3) 3D Gaussian Splatting 매핑입니다. IMU 데이터를 사용하여 포인트 클라우드 간의 매칭을 개선함으로써 키프레임 간의 간격을 늘리고 메모리 사용량을 크게 줄입니다. 이 접근법은 기존 SLAM 시스템과의 통합성을 유지하면서도 높은 정확성을 제공합니다.

- **Performance Highlights**: VIGS SLAM은 기존의 방 크기 3DGS SLAM보다 더 높은 성능을 발휘하며, 대규모 실내 환경에서의 효율적인 작업이 가능하다는 점에서 중요한 발전을 보여줍니다. 실험 결과는 제안된 방법이 대규모 환경에서도 기존 SLAM 시스템과 동등한 성능을 발휘할 수 있음을 입증합니다. 이러한 특성은 VIGS SLAM이 실용적이고 강력한 대규모 SLAM 솔루션으로 자리잡을 수 있게 합니다.



### ExLM: Rethinking the Impact of $\texttt{[MASK]}$ Tokens in Masked Language Models (https://arxiv.org/abs/2501.13397)
Comments:
          29 pages, 12 figures

- **What's New**: 본 논문은 Masked Language Models (MLMs)의 훈련에서 [MASK] 토큰의 영향과 그로 인해 발생하는 corrupted semantics 문제에 대해 탐구합니다. 특히, 기존 MLM 모델의 한계를 극복하기 위해 새로운 Enhanced-context MLM인 ExLM을 제안합니다. ExLM은 입력 맥락에서 [MASK] 토큰을 확장하여 모델의 의미 정보를 캡처하고 corrupted semantics 문제를 완화하는데 초점을 맞추고 있습니다.

- **Technical Details**: ExLM은 기존의 MLM 프레임워크를 기반으로 하여 [MASK] 토큰을 입력 맥락에서 증가시킵니다. 이는 모델이 더 많은 컨텍스트 용량을 활용할 수 있게 하고, 다양한 의미적 의존성을 모델링할 수 있는 능력을 부여합니다. 실험 결과, ExLM은 BERT 기반 모델과 비교하여 텍스트 모델링 및 SMILES 모델링 작업에서 개선된 성능을 보여주었습니다.

- **Performance Highlights**: ExLM은 기존의 MLM보다 의미적 표현 능력을 향상시켰으며, 다운스트림 작업에서도 뛰어난 성능을 입증했습니다. 실험 분석을 통해 ExLM은 맥락 증강을 통해 MLM에서 자주 나타나는 다중 의미 문제를 효과적으로 감소시켰습니다. 이는 MLM의 향후 발전을 위한 중요한 기초 자료로 작용할 것으로 기대됩니다.



### Bridging The Multi-Modality Gaps of Audio, Visual and Linguistic for Speech Enhancemen (https://arxiv.org/abs/2501.13375)
- **What's New**: 이 논문에서는 Speech Enhancement (SE)를 위한 새로운 멀티모달 학습 프레임워크를 제안합니다. 기존의 오디오 및 비주얼 정보를 넘어서 언어 정보를 통합하여 성능을 더욱 향상시키는 방안을 모색하고 있습니다. 시각적 및 청각적 정보를 직접 처리하는 데 최신 Diffusion Model을 활용함으로써 AVSE(오디오-비주얼 스피치 향상) 모델링을 구현하고 있습니다.

- **Technical Details**: 제안된 AVSE 시스템은 Cross-Modal Knowledge Transfer (CMKT) 기법을 통해 언어 모달리티와 시각음향 모달리티 간의 지식 전이를 수행합니다. 본 연구는 Predictive 단계와 Generative 단계를 포함하는 Diffusion 기반 AVSE 시스템을 적용하며, BERT와 같은 사전 훈련된 언어 모델(PLM)을 활용하여 훈련 과정을 진행합니다. 이를 통해 훈련 후 inference 단계에서 PLM의 사용을 최소화하여 효율적인 성능을 도모하고 있습니다.

- **Performance Highlights**: 실험 결과 제안된 AVSE 시스템은 기존의 최첨단 방법과 비교하여 음성 품질을 현저히 향상시키고 음성 생성에서 발생하는 인위적인 아티팩트를 줄이는 데 성공했습니다. 비주얼-청각 정보와 언어 정보를 통합함으로써 SE 시스템의 성능을 크게 향상시킬 수 있음을 보여주고 있습니다. 이 연구 결과는 Diffusion Model 기반 기술이 AVSE의 발전에 기여할 수 있음을 시사합니다.



### Advancing Carbon Capture using AI: Design of permeable membrane and estimation of parameters for Carbon Capture using linear regression and membrane-based equations (https://arxiv.org/abs/2501.13373)
- **What's New**: 이 연구는 기후 변화 완화를 위한 효율적인 탄소 포집 솔루션의 필요성을 해결하기 위해, CO$_2$ 분리를 위한 멤브레인 기반 시스템에 초점을 맞추고 있습니다. 연구는 탄소 포집 효율을 개선하기 위해 다른 가스들이 통과할 수 있도록 선택적으로 CO$_2$를 차단하는 멤브레인 특성 최적화의 중요성을 강조합니다.

- **Technical Details**: 본 연구에서는 멤브레인 방정식을 기반으로 한 선형 회귀 모델(linear regression models)을 이용하여 여러 주요 파라미터들을 추정했습니다. 이 파라미터로는 0.4805의 공극률(porosity, $	ext{ε}$), 2.9084의 코제니 상수(Kozeny constant, K), 105.3272 m$^2$/m$^3$의 특정 표면적(specific surface area, $	ext{σ}$), 6.2166 MPa의 평균 압력(mean pressure, Pm), 0.1997 Ns/m$^2$의 점도(viscosity, $	ext{μ}$), 3.2559 kg m$^{-2}$ s$^{-1}$의 가스 유량(gas flux, Jg) 등이 포함됩니다.

- **Performance Highlights**: 연구에서는 Q가 9.8778 × 10$^{-4}$ m$^3$/s인 유량(flow rate), 2.8219 MPa의 주입 압력(injection pressure, P$_1$), 2.5762 MPa의 배출 압력(exit pressure, P$_2$) 등의 멤브레인 성능을 분석했습니다. CO$_2$에 대한 투과성(permeability) 값은 0.045로, 효율적인 분리를 위한 잠재력을 제시합니다. 이 연구는 또한 인공지능(AI)을 활용하여 탄소 포집을 위한 멤브레인 설계를 지원할 수 있는 방법을 탐구합니다.



### Meta-Feature Adapter: Integrating Environmental Metadata for Enhanced Animal Re-identification (https://arxiv.org/abs/2501.13368)
Comments:
          10 pages, 6 figures

- **What's New**: 이번 논문에서는 환경 메타데이터를 시각 데이터와 결합하여 동물 재식별(Animal ReID)의 성능을 향상시키기 위한 새로운 경량 모듈인 메타 피처 어댑터(Meta-Feature Adapter, MFA)를 제안합니다. MFA는 온도나 일주기 리듬과 같은 환경 메타데이터를 자연어 설명으로 변환하고 이미지 특징과 통합하여 성능을 개선합니다. 이를 위해 메타데이터에 주의를 기울이기 위한 크로스 어텐션(cross-attention) 메커니즘과 게이트 크로스 어텐션(Gated Cross-Attention) 기법이 도입되었습니다.

- **Technical Details**: MFA는 환경 메타데이터를 자연어 설명으로 처리하여 텍스트 임베딩(text embeddings)으로 변환하고, 이를 이미지 피처와 연결합니다. 자연어로 생성된 설명은 예를 들어 "차가운 온도에서 낮 동안 포착된 스토트 Bob의 사진"과 같은 형태로 제공됩니다. MFA는 두 가지 프로토콜을 활용해 실험을 진행했으며, 각 프로토콜에 따라 종 내부 재식별(intra-species re-identification)과 종 간 재식별(cross-species re-identification) 작업을 수행합니다.

- **Performance Highlights**: 실험 결과 MFA를 적용한 모델은 기존 Animal ReID 모델에 비해 성능이 일관되게 향상됨을 보여주었습니다. 개발한 메타데이터 보강 동물 재식별(Metadata Augmented Animal Re-identification, MAAR) 데이터셋은 뉴질랜드의 여섯 종을 포함하고 있으며, 이미지 데이터와 환경 메타데이터가 쌍으로 제공됩니다. 초기 연구 결과 MFA는 동물 재식별 분야와 야생 생물 모니터링 연구에서 혁신적인 발전을 촉진할 잠재력을 지니고 있음을 증명합니다.



### Polyhedra Encoding Transformers: Enhancing Diffusion MRI Analysis Beyond Voxel and Volumetric Embedding (https://arxiv.org/abs/2501.13352)
- **What's New**: 이 논문에서는 Polyhedra Encoding Transformer(PE-Transformer)라는 새로운 방법을 제안합니다. 이 방법은 구형 신호를 처리하기 위해 특별히 설계되었으며, 구면을 따라 데이터를 샘플링하는 데 icosahedral 폴리곤을 사용하는 것이 특징입니다. 이 접근법은 dMRI 분석에서 기존의 CNN 및 표준 transformer보다 더 높은 정확도를 보여줍니다.

- **Technical Details**: PE-Transformer는 정규 화소 집합을 사용하여 diffusion MRI 데이터의 기본 방향에 대하여 균일하게 분포된 신호를 재샘플링합니다. 이를 통해 수많은 b-값과 확산 방향을 사용하는 기존 방법에 비해 더 효과적으로 다중 구성 모델 및 섬유 방향 분포(FOD) 추정을 수행할 수 있습니다. icosahedral 구조의 방향 정보를 반영하는 transformer encoder를 사용하여 embeddings가 처리됩니다.

- **Performance Highlights**: 실험을 통해 PE-Transformer는 다양한 gradient encoding 프로토콜로 테스트된 결과, multi-compartment 모델 및 섬유 방향 분포 추정에서 더 높은 정확성을 보였습니다. 이 방법은 전통적인 dMRI 분석의 단점인 시간 소모와 이동 아티팩트를 줄이는 데 효과적이며, 일반적인 스캐닝 프로토콜에 대한 적응력이 뛰어난 것으로 평가됩니다.



### CuriousBot: Interactive Mobile Exploration via Actionable 3D Relational Object Graph (https://arxiv.org/abs/2501.13338)
Comments:
          Project Page: this https URL

- **What's New**: 이번 연구에서는 모바일 탐사를 위한 새로운 접근법으로 3D 관계 객체 그래프(3D relational object graph)를 도입합니다. 기존의 방법들이 주로 능동적인 인식(active perception)만을 중점으로 삼았던 반면, 우리의 시스템은 능동적인 상호작용(active interaction)을 통해 환경을 효과적으로 탐색할 수 있도록 설계되었습니다. 이 시스템은 다양한 장면에서 평가되었으며, 단순한 비전-언어 모델(vision-language models, VLMs)에 의존하는 방법들보다 효과적인 결과를 보였습니다.

- **Technical Details**:  연구는 SLAM, Graph Constructor, Task Planner 및 Low-Level Skills의 네 가지 모듈로 구성된 시스템을 개발했습니다. SLAM 모듈은 RGBD 관측 데이터와 로봇의 이동 정보를 입력받아 카메라의 위치를 추정합니다. 이어서 Graph Constructor는 개체를 감지하고 분할하여 객체 노드를 생성하고, 각 노드 간의 관계를 정의하여 다음 단계인 작업 계획(task planning)을 위한 기반 정보를 제공합니다.

- **Performance Highlights**: 우리의 3D 관계 객체 그래프는 가정 환경에서 일반적으로 나타나는 다양한 차단 관계를 인코딩할 수 있으며, 그 결과 시스템이 여러 환경 레이아웃에 적응할 수 있는 능력을 보여주었습니다. 여러 실험을 통해 시스템의 성능을 정량적으로 분석하였고, 다양한 물체 범주를 다룰 수 있는 능력을 입증했습니다. 추가적으로, GPT-4V와의 비교 실험을 통해 우리의 접근법이 작업 계획(task planning)에 더 효과적이라는 결과를 도출하였습니다.



### RAMQA: A Unified Framework for Retrieval-Augmented Multi-Modal Question Answering (https://arxiv.org/abs/2501.13297)
Comments:
          Accepted by NAACL 2025 Findings

- **What's New**: 이 논문에서는 텍스트와 이미지를 통합한 Multi-modal retrieval-augmented Question Answering (MRAQA) 분야에서 새로운 접근법인 RAMQA를 제안합니다. RAMQA는 전통적인 learning-to-rank 방법과 generative ranking 기술을 결합하여, 현대의 대형 생성 언어 모델(LLMs)을 활용한 정보 검색의 한계를 극복하고자 합니다. 이를 통해 두 가지 MRAQA 벤치마크인 WebQA와 MultiModalQA에서 성능 향상을 입증하였습니다.

- **Technical Details**: RAMQA는 LLaVA를 기반으로 하여 multi-modal pointwise ranker를 훈련한 후, novel autoregressive multi-task learning 접근법을 채택하여 LLaMA 모델을 상위 k개 문서의 재정렬에 사용합니다. 이 과정에서는 zero-shot LLaVA 모델을 이용하여 다중 모달 문서를 텍스트 표현으로 통합하고, permutation 기법을 활용하여 문서 후보군의 다양성을 증가시켜 bias를 감소시키는 방법을 사용합니다.

- **Performance Highlights**: 실험 결과, WebQA와 MultimodalQA 두 벤치마크에서 강력한 기준선에 비해 유의미한 성능 향상을 달성하였으며, RAMQA는 웹 기반의 QA 시스템에서 네 번째 순위를 기록했습니다. 이 연구는 multi-modal generative LLMs의 활용 가능성을 보여주며, 점진적인 재정렬이 정보 검색에서 더 효율적인 처리를 가능하게 함을 시사합니다.



### Map Prediction and Generative Entropy for Multi-Agent Exploration (https://arxiv.org/abs/2501.13189)
- **What's New**: 이번 연구에서는 로봇 팀이 현재 환경에 대해 알고 있는 것 이상의 행동을 할 수 있도록 하는 새로운 접근법을 제시합니다. 우리는 다중 에이전트 2D 점유(Map Occupancy) 맵에서의 탐사 미션 중 미지의 공간을 자동으로 보완할 수 있는 맵 예측기를 개발했습니다. 본 연구의 핵심은 과거 정보 외에도 장면의 가능한 해석의 분포를 추론하여 로봇 임무를 효율적으로 우선 순위화하는 것입니다.

- **Technical Details**: 이 연구에서는 미지의 환경을 탐사하는 과정에서 생성 엔트로피(generative entropy) 개념을 도입하여 예측의 불확실성이 높은 영역을 식별하고 이를 우선 순위에 반영하게 됩니다. 특히, 정제된 라틴(diffusion) 모델을 활용하여 도시 환경의 풍부하고 일관된 해석을 제공하면서도 상대적으로 적은 계산 시간으로 작동하도록 설계되었습니다. 또한, 새로운 작업 순위화 방법을 기존의 정보 기반 작업 순위화 방법과 병행하여 비교했습니다.

- **Performance Highlights**: 결과적으로, 제안된 새로운 작업 순위화 방식을 사용함으로써 전통적인 정보 기반 방법에 비해 훨씬 더 빠른 시간 안에 정확한 장면 예측이 가능하다는 것을 보여주었습니다. 우리 모델은 로봇 팀의 크기에 관계없이 효율적으로 작업을 배포할 수 있도록 설계되었으며, 탐사 미션에서 로봇 안전성을 유지하면서 예측 결과를 활용하는 방법을 제시합니다. 앞으로는 보다 발전된 환경 표현으로 이 프레임워크를 확장할 수 있을 것으로 예상됩니다.



### Topological constraints on self-organisation in locally interacting systems (https://arxiv.org/abs/2501.13188)
Comments:
          9+3 pages, four figures, four tikzpictures. To appear in Philos Trans R Soc A

- **What's New**: 이 논문은 집합적 지능(collective intelligence)의 특성을 이해하고, 복잡한 구조와 기능을 자가 조직화(self-organisation)하는 생물학적 시스템의 상호작용을 그래프 이론(graph theory)를 통해 분석합니다. 저자들은 다양한 시스템에서 나타나는 자가 조직화 현상을 연구하고, 인간 언어 모델과 다세포 생물 사이의 기능적 차이를 탐구합니다. 특히, 그래프의 토폴로지(topology)가 이러한 시스템의 질서 있는 상태를 유지하는 데 중요한 요소임을 강조합니다.

- **Technical Details**: 연구는 k-정점 그래프(k-vertex graph)를 기반으로 하며, 각 정점에 n-ary 변수가 할당됩니다. 저자들은 유한한 상호작용 창(window of interaction) 내에서의 상호작용을 정의하는 해밀토니안(Hamiltonian)을 활용하여, 시스템 내에서의 질서 있는 상태의 존재 조건을 도출합니다. '로컬 해밀토니안(local Hamiltonian)'의 구조를 통해 상호작용의 조합(combinatorics)을 분석하고, 도메인 월(domain wall)의 양에 따른 자유 에너지(free energy)의 비대칭성을 연구합니다.

- **Performance Highlights**: 논문에서는 다중 스케일 시스템이 어떻게 복잡한 패턴으로 조직될 수 있는지를 분석하며, 자연어 모델이 긴 시퀀스의 출력에서 겪는 한계에 대해 논의합니다. 인간의 신체 세포가 대규모로 조정 및 조직하는 능력과 언어 모델의 일관성을 유지하는 데 어려움이 있는 이유를 토폴로지의 차이로 설명합니다. 이러한 결과는 생물학적 현상('stigmergy', 'embodiment')에서 발생하는 적합성 압력을 이해하는 데 중요한 통찰력을 제공합니다.



### QuFeX: Quantum feature extraction module for hybrid quantum-classical deep neural networks (https://arxiv.org/abs/2501.13165)
Comments:
          12 pages, 10 figures

- **What's New**: 본 논문에서는 Quantum Feature Extraction (QuFeX)이라는 새로운 양자 기계 학습 모듈을 소개합니다. QuFeX는 차원 축소된 공간에서 특성 추출을 가능하게 하여 일반적인 양자 컨볼루션 신경망 아키텍처에서 요구되는 병렬 평가 수를 크게 줄입니다. 또한 QuFeX는 심층 클래식 신경망에 원활하게 통합될 수 있도록 설계되어 하이브리드 양자-클래식 모델에 특히 적합합니다.

- **Technical Details**: QuFeX는 QCNN(Quantum Convolutional Neural Network)과 QuanNN(Quanvolutional Neural Network)의 기술을 통합하여 설계되었습니다. 이 모듈은 데이터 분석 구조를 결합하여, 입력 데이터의 국소적인 변환에 최적화된 데이터 파이프라인을 제공하고, 축소된 차원 표현에서 데이터를 처리합니다. 이러한 설계는 QuanNN보다 병렬 양자 회로 평가의 수를 줄이는 장점을 가지고 있습니다.

- **Performance Highlights**: QuFeX를 이용한 Qu-Net은 클래스컬 U-Net 모델의 병목 부분에 통합되어 있으며, 의료 영상 및 자율주행과 같은 이미지 분할 작업에서 우수한 성능을 보여줍니다. 실험 결과 Qu-Net은 전통적인 U-Net 기준에 비해 뛰어난 분할 성능을 달성하여, 실제 이미지 분할 작업에 양자 강화 심층 신경망을 효과적으로 적용할 수 있는 가능성을 제시합니다.



### Forecasting of Bitcoin Prices Using Hashrate Features: Wavelet and Deep Stacking Approach (https://arxiv.org/abs/2501.13136)
Comments:
          arXiv admin note: text overlap with arXiv:2402.05943 by other authors

- **What's New**: 디지털 통화, 특히 비트코인(BTC)의 가격 변동성을 예측하기 위한 새로운 분류 및 회귀 모델이 제안되었습니다. 본 연구는 웨이브렛을 이용하여 잡음을 제거하고 스택 딥 러닝 기법을 기반으로 하여 다양한 시간 간격에서 BTC 가격을 예측합니다. 이러한 모델은 딥 러닝을 기반으로 하여 1일, 7일, 30일, 90일의 가격 예측을 수행합니다.

- **Technical Details**: 제안된 모델은 딥러닝 기반 트랜스포머와 신경망(neural networks) 모델을 사용하여 다양한 가격 예측을 수행합니다. 데이터 전처리 단계에서 Chi2, RFE, Embedded 3가지 특성 선택 모델을 적용하였습니다. 마지막으로 90일까지의 예측을 위한 중간-term 가격 및 고저 예측 메커니즘을 개발하였습니다.

- **Performance Highlights**: 모델의 정확도는 다음 날 예측에 대해 63%였으며, 7일, 30일, 90일 예측에 대해서는 각각 64%, 67%, 82%의 정확도를 달성했습니다. 일일 가격 예측의 경우, 오류율은 0.58로 감소하였고, 7일에서 90일 동안의 오류는 2.72%에서 2.85%로 나타났습니다. 이러한 결과는 제안된 모델이 기존 문헌의 다른 모델들보다 더 우수한 성능을 보임을 시사합니다.



### A Learnt Half-Quadratic Splitting-Based Algorithm for Fast and High-Quality Industrial Cone-beam CT Reconstruction (https://arxiv.org/abs/2501.13128)
- **What's New**: 이번 논문에서는 대형 결핍 시나리오에서 고품질 X선 원뿔 빔 CT 이미지를 재구성하기 위해 새로운 알고리즘인 half-quadratic splitting 기반 방법을 제안합니다. 이 방법은 CNN(convolutional neural networks)을 활용하여 기존의 비효율적인 접근 방식을 극복하고, 데이터 일관성을 유지하며 고속 CT 스캔 데이터로부터 뛰어난 성능을 발휘합니다. 또한, 제안된 알고리즘은 전통적인 방법에 비해 메모리 사용을 최적화하여 대규모 이미지 재구성 문제에 효과적으로 확장할 수 있는 잠재력을 가지고 있습니다.

- **Technical Details**: 이 알고리즘은 CNN과 conjugate gradient(CG) 단계 간의 교차 적용을 통해 동작하며, 데이터 일관성(DC) 블록과의 상호작용을 통해 이미지 재구성을 수행합니다. 또한, 각 외부 반복(iteration)에서 별도로 CNN을 훈련시키며, 3D CBCT 데이터는 여러 2D 패치로 나누어 더 작은 메모리 소모로 처리됩니다. 이 접근 방식은 고도로 최적화된 메모리 활용을 통해 대규모 산업 CT 적용에 대한 실용성을 향상시킵니다.

- **Performance Highlights**: 제안된 방법은 공개적으로 이용 가능한 Walnuts 데이터 세트에서 전통적인 MBIR(model-based iterative reconstruction) 및 단일 단계 U-Net 접근 방식보다 더 우수한 성능을 보여줍니다. 고품질 재구성을 제공하면서도 처리 시간 복잡도를 현저히 줄이는 동시에, 다양한 측정 조건에 대한 일반화 능력을 향상시킵니다. 전반적으로, 이 새로운 프레임워크는 이미지 재구성 문제에서 성능 저하 없이 대규모로 확장할 수 있는 능력을 제공합니다.



### Generating Plausible Distractors for Multiple-Choice Questions via Student Choice Prediction (https://arxiv.org/abs/2501.13125)
- **What's New**: 이번 연구에서는 교육 분야에서 학생들의 오해(misconceptions)와 지식 격차를 파악하고 이해도를 정확하게 평가하기 위한 MCQ(다지선다형 질문)에서 더 그럴듯한 방해 선택지(distractor)를 생성하는 모델 훈련 파이프라인을 제안합니다. 먼저 학생들의 오해를 고려하여 두 개의 방해 선택지의 상대적 그럴듯함을 평가하는 쌍별(rank-based) 랭커(pairwise ranker)를 훈련합니다. 이후, 이 모델을 활용하여 쌍별 방해 선택지 랭크 데이터셋을 생성하고, Direct Preference Optimization(DPO)을 통해 더 그럴듯한 방해 선택지를 생성하는 방해 선택지 생성기를 훈련합니다.

- **Technical Details**: 모델 훈련을 위해, South Korea의 온라인 학습 플랫폼에서 교육자들이 생성한 MCQ 데이터세트를 사용했습니다. 데이터세트에는 파이썬, 데이터베이스(SQL) 및 머신러닝 및 딥러닝(Machine Learning & Deep Learning) 관련 질문이 포함되어 있으며, 각 질문의 학생 선택률 정보가 포함되어 있습니다. 이를 통해 우리는 학생들이 어떤 방해 선택지를 더 혼란스럽게 선택하는지 평가할 수 있습니다.

- **Performance Highlights**: 실험 결과, 쌍별 랭커는 학생들의 일반적인 오해를 효과적으로 파악하며, 인간 전문가와 유사한 랭킹 정확도를 달성했습니다. 또한 방해 선택지 생성기는 여러 기준선 모델보다 우수한 성과를 보이며, 더 그럴듯한 방해 선택지를 생성하고, 높은 항목 차별지수(item discrimination index, DI)를 산출했습니다.



### Episodic Memories Generation and Evaluation Benchmark for Large Language Models (https://arxiv.org/abs/2501.13121)
- **What's New**: 이 논문에서는 에피소드 메모리(episodic memory)에 대한 LLMs의 한계를 극복하고, 그러한 메모리 능력을 통합하기 위한 포괄적인 프레임워크를 제안합니다. 기존의 LLM은 사실과 일치하지 않는 정보를 생성하는 할루시네이션(hallucination) 문제를 겪고 있으며, 이로 인해 인간과 유사한 사고를 발전시키기 위한 잠재력이 제한되고 있습니다. 또한, 연구자들은 에피소드 메모리를 충분히 평가하지 않았으며, 논문에서 제안한 새로운 벤치마크가 이러한 격차를 해소할 수 있을 것으로 기대하고 있습니다.

- **Technical Details**: 이 연구는 인지 과학에서 영감을 받아 에피소드 메모리를 구조적으로 모델링하는 접근법을 개발했습니다. 이 프레임워크는 시간적 및 공간적 맥락, 관련 엔티티(entity) 및 사건에 대한 상세한 설명을 포함합니다. 저자들은 LLM 성능 평가를 위한 독창적인 에피소드 메모리 벤치마크를 창출하였으며, 이를 통해 다양한 회상(recall) 및 에피소드 추론(tasks) 작업에서 평가할 수 있도록 오픈 소스 코드와 데이터셋을 공개했습니다.

- **Performance Highlights**: 상위 모델인 GPT-4 및 Claude 변형, Llama 3.1과 o1-mini를 평가한 결과, 가장 진보된 LLM조차 복잡한 공간적-시간적 관계(spatio-temporal relationships)를 다루는 에피소드 메모리 작업에서 어려움을 겪고 있음을 보여줍니다. 특히, 관련된 여러 사건을 처리할 때 이러한 어려움이 더욱 두드러졌으며, 10k-100k 토큰의 짧은 문맥(context)에서도 이 문제가 나타났습니다. 이 연구는 LLM의 성능 개선을 위해 에피소드 메모리 벤치마크의 중요성을 강조합니다.



### Multilinguality in LLM-Designed Reward Functions for Restless Bandits: Effects on Task Performance and Fairness (https://arxiv.org/abs/2501.13120)
Comments:
          Accepted at the AAAI-2025 Deployable AI Workshop

- **What's New**: 이 논문은 Restless Multi-Armed Bandits (RMABs) 알고리즘에 비영어 명령어를 사용할 때의 작업 성능과 공정성에 미치는 영향을 연구합니다. 특히, 저자들은 리소스가 부족한 언어를 포함한 여러 언어로 번역된 다양한 복잡도의 프롬프트를 조사하였습니다. 그 결과 영어 프롬프트가 작업 성능에 있어 유리함을 보이며, 공정성 측면에서도 리소스가 부족한 언어와 복잡한 프롬프트가 불공정성을 일으킬 가능성이 높다는 사실을 밝힙니다.

- **Technical Details**: 실험은 DLM 알고리즘(Behari et al. 2024)을 이용하여, 영어, 힌디어, 타밀어 및 투루어(저자원 언어)로 다양한 프롬프트를 실행하는 방식으로 진행됩니다. 이 연구는 6개의 피쳐를 기반으로 하는 합성 환경을 사용하여, 각 언어로 제안된 보상 함수의 효과를 조사합니다. 또한, Gemini 1.0 Pro를 LLM으로 활용하며, Whittle Index 기반의 해결책을 리인포스먼트 러닝 부분에 적용합니다.

- **Performance Highlights**: 영어로 제안된 LLM 보상 함수가 다른 언어에 비해 훨씬 효과적이라는 결과를 도출하였습니다. 프롬프트의 정확한 표현 방식은 작업 성능에 영향을 미치며, 프롬프트의 복잡성이 증가할수록 모든 언어에서 성능이 저하되는 경향을 보입니다. 그러나 영어 프롬프트에 비해 리소스가 부족한 언어의 경우 성능 저하가 더욱 두드러지는 것으로 확인되었습니다.



### iServe: An Intent-based Serving System for LLMs (https://arxiv.org/abs/2501.13111)
Comments:
          19 pages, 24 figures

- **What's New**: LLM(대형 언어 모델) 분야에서 iServe라는 자동화된 의도 기반 시스템이 소개되었습니다. 이 시스템은 개발자가 특정한 의도를 간편하게 지정하면, 자동으로 최적의 배포 구성(configuration)을 선택하여 대형 언어 모델의 추론(inference)을 지원합니다. iServe는 'fingerprint'라는 경량화된 표현을 통해 각 배포 구성이 사용자의 성능 지표에 미치는 영향을 효과적으로 추정하고, 이를 통해 자원 사용과 비용을 최적화합니다.

- **Technical Details**: iServe는 LLM 지문(fingerprint)이라는 기법을 도입하여 여러 배포 구성 옵션에 대한 지연(latency) 및 메모리 사용량을 효율적으로 추정합니다. 이 시스템은 LLM의 복잡한 구조 대신 지문을 프로파일링하여 자원 사용 비용을 크게 줄이며, 메모리 최적화와 사용자 의도에 부합하는 배포 구성을 자동으로 탐색합니다. 또한, iServe는 로드 인식 LLM-GPU 매핑 알고리즘을 사용하여 리소스 효율성을 고려한 GPU 배치도 자동으로 수행합니다.

- **Performance Highlights**: iServe는 기존 최첨단 시스템과 비교하여 평균 77.62%의 지연 시간 및 86.70%의 비용 절감 효과를 달성했습니다. 또한 SLO 위반(SLO violations)을 7.09배 감소시키고, GPU 처리량(GPU throughput)을 4.72배 향상시켰습니다. 이러한 효율성을 통해 iServe는 프로파일링 비용을 기존의 방법에 비해 6.05배 낮추며, AI와 머신러닝 운영 환경에서 실질적인 비용 절감을 제공할 수 있습니다.



### Each Graph is a New Language: Graph Learning with LLMs (https://arxiv.org/abs/2501.11478)
- **What's New**: 이 논문은 Large Language Models (LLMs)을 활용하여 노드 분류 작업에 대한 텍스트 속성을 가진 그래프 구조를 모델링하는 새로운 프레임워크인 GDL4LLM(Graph-Defined Language for Large Language Model)을 제안합니다. 기존 접근법에서는 그래프 구조 표현이 너무 방대하거나 텍스트 속성만으로는 충분한 정보를 제공하지 못하는 한계를 가지고 있었습니다. GDL4LLM은 그래프를 설명하는 대신 그래프 언어 말뭉치를 생성하고 이를 통해 LLM을 사전 훈련시킴으로써 그래프 구조를 효과적으로 이해할 수 있게 합니다.

- **Technical Details**: GDL4LLM은 그래프를 그래프 언어 코퍼스로 변환하여 노드 분류를 위한 프레임워크를 구현합니다. 이 과정에서, 최근 LLM이 하나의 언어에서 훈련되어도 다른 언어에서 우수한 성능을 보일 수 있다는 점을 활용했습니다. LLM은 이 그래프 언어를 통해 노드에 대한 구조적 정보를 간결하게 학습할 수 있으며, 이는 터치포인트 중심으로 구조적 정보를 설명하는 데에 필요합니다.

- **Performance Highlights**: 실험을 통해 GDL4LLM은 세 가지 실제 데이터셋에서 기존의 설명 기반 접근법이나 텍스트 속성 임베딩 기반 방법들보다 뛰어난 성능을 보였습니다. LLM을 활용하여 다양한 차수의 그래프 구조를 효율적으로 모델링함으로써 노드 분류 작업에서 탁월한 결과를 나타냈습니다. 전반적으로, GDL4LLM은 LLM의 언어 이해 능력을 그래프 구조 데이터에 성공적으로 이전시킬 수 있도록 설계되었습니다.




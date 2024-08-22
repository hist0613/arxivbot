New uploads on arXiv(cs.CL)

### Great Memory, Shallow Reasoning: Limits of $k$NN-LMs (https://arxiv.org/abs/2408.11815)
- **What's New**: 이번 연구는 $K$-nearest neighbor language models ($k$NN-LMs)이 정보 회상을 강화하는 것에서 실제 작업 성능으로 이어지는지를 철저히 평가하였습니다. 또한, 이 모델들이 기억이 중요한 작업에서는 우수한 성능을 보이지만, 복잡한 추론 작업에서는 저조한 결과를 보임을 입증하였습니다.

- **Technical Details**: $k$NN-LMs는 기존의 언어 모델에 비해 비모수적 접근 방식을 채택하여, 방대한 텍스트 데이터 스토어를 이용해 성능을 향상시키는 방법입니다. 이 모델은 입력 패턴을 인식하고 메모리와 매칭하여 출력 결과를 결정할 수 있는 간단한 작업에서 높은 정확도를 보입니다.

- **Performance Highlights**: 연구 결과, $k$NN-LMs는 메모리 집약적인 작업에 강한 성능을 발휘하였으나, 추론 능력을 필요로 하는 작업에서는 성능이 떨어지며, 심지어 정보가 완벽하게 회수되더라도 올바른 답변을 찾지 못하는 경우가 많았습니다.



### PermitQA: A Benchmark for Retrieval Augmented Generation in Wind Siting and Permitting domain (https://arxiv.org/abs/2408.11800)
- **What's New**: 이 논문은 Wind Siting 및 Permitting 도메인에 대한 최초의 벤치마크인 PermitQA를 소개하며, 이는 과학적 문서 및 바람 에너지 프로젝트의 환경적 영향을 다룹니다. 또한, RAG 기반의 LLM(대형 언어 모델)의 성능을 평가하기 위한 포괄적인 프레임워크를 제안합니다.

- **Technical Details**: 제안된 프레임워크는 자동 질문-답변 생성을 통해 RAG의 성능을 시스템적으로 평가합니다. 이 과정에서 도메인 전문가와 AI LLM이 협력하여 질문을 생성하며, 다양한 복잡성과 유형의 질문을 포함하는 평가는 RAG의 효과를 측정하는 데 필수적입니다. 자동화된 방법과 인간의 큐레이션의 결합을 통해 질문 세트를 생성합니다.

- **Performance Highlights**: PermitQA 벤치마크에 대한 다양한 모델의 성능을 비교 평가하였으며, RAG 기반 LLM이 방대한 정보에서 관련 데이터를 정확히 검색하고 응답하는 데 효율적임을 입증했습니다. RAG의 성능을 다양한 질문 유형을 통해 평가함으로써, 실제 시나리오에서의 적용 가능성을 강화합니다.



### Practical token pruning for foundation models in few-shot conversational virtual assistant systems (https://arxiv.org/abs/2408.11799)
Comments:
          6 pages, 3 figures

- **What's New**: 본 논문은 Enterprise Virtual Assistant (VA) 시스템의 Intent classification을 위한 혁신적인 접근을 소개하고, 효율적인 모델 사전 훈련 및 다중 작업 토큰 프루닝(multi-task token pruning) 기법을 도입하여 사용자 입력 처리 속도를 향상시킵니다.

- **Technical Details**: 저자들은 transformer 기반의 문장 임베딩 모델을 Contrastive learning objective로 사전 훈련하고, 이를 활용하여 몇 개의 학습 샘플만으로도 높은 정확도를 유지하며 intent classification을 수행합니다. 이 과정에서 모델 증류(model distillation)와 동적 토큰 프루닝(dynamic token pruning) 기법을 통해 추론 속도를 개선했습니다.

- **Performance Highlights**: 이 접근 방식은 few-shot 상황에서 최첨단의 결과를 달성했으며, 상용 솔루션보다 더 나은 성능을 보여 주었습니다. 특히, 토큰 프루닝 기법을 통해 긴 사용자 입력의 추론 시간도 개선되었습니다.



### LLM Pruning and Distillation in Practice: The Minitron Approach (https://arxiv.org/abs/2408.11796)
- **What's New**: 이번 논문에서는 Llama 3.1 8B 모델과 Mistral NeMo 12B 모델을 각각 4B와 8B 파라미터로 압축하기 위한 포괄적인 방법을 제시합니다. 프루닝(pruning)과 디스틸레이션(distillation) 기법을 사용하여 이룬 결과를 다양한 벤치마크에서 평가합니다.

- **Technical Details**: 프루닝 전략으로는 깊이 프루닝(depth pruning)과 joint hidden/attention/MLP(width pruning) 방식이 사용되었습니다. 모델의 정확도를 회복하기 위해 디스틸레이션 방법을 적용하며, teacher correction 단계에서 원본 데이터에 접근할 수 없는 상황에서도 교사 모델을 미세 조정(fine-tune)합니다. 또한, 구조적 프루닝(structured pruning) 기법을 사용하며, 활성화(activation)에 기반한 중요도 추정 전략(importance estimation)을 통해 각 레이어의 중요도를 평가합니다.

- **Performance Highlights**: MN-Minitron-8B 모델은 기존 Mistral NeMo 12B 모델에 비해 평균 1.2배 빠른 추론 성능을 보여주며, Llama-3.1-Minitron-4B 모델은 깊이 및 폭 프루닝 변형 모두 교사 Llama 3.1 8B 모델과의 비교에서 강력한 정확도를 발휘합니다.



### Personality Alignment of Large Language Models (https://arxiv.org/abs/2408.11779)
- **What's New**: 이번 연구에서는 Personality Alignment라는 개념을 도입하여 대규모 언어 모델(LLMs)이 개인 사용자의 특정 선호에 맞춰 응답과 결정을 조정할 수 있도록 합니다. 이를 통해 인지심리학에 기초한 Personality Alignment with Personality Inventories (PAPI) 데이터세트를 구축하였으며, 이는 300,000명의 실제 사용자가 제공한 행동 선호 데이터를 포함합니다.

- **Technical Details**: PAPI 데이터세트는 Big Five Personality Factors를 기반으로 하며, 각 개인의 행동 패턴을 정량적으로 평가할 수 있도록 지원합니다. 이 연구에서는 행동 데이터의 부족, 다양한 선호 및 확장성 요건을 고려하여 활성화 개입 최적화 방법을 개발했습니다. 이는 LLMs이 최소한의 데이터와 계산 자원으로 개인 행동 선호에 효율적으로 정렬될 수 있도록 합니다. 제안된 PAS 방법은 기존 DPO에 비해 최적화 시간이 1/5로 줄어들면서도 우수한 성능을 발휘합니다.

- **Performance Highlights**: 실험을 통해 PAS 방법은 DPO 및 PPO에 비해 높은 정렬 효율성을 보였으며, GPT-4o 모델보다도 뛰어난 성능을 나타냈습니다. 이는 Llama-3-8B 모델을 기반으로 하여 연구되었으며, 개별 특성 데이터의 부족과 높은 확장성 문제를 해결하는 데 중점을 두었습니다.



### Leveraging Fine-Tuned Retrieval-Augmented Generation with Long-Context Support: For 3GPP Standards (https://arxiv.org/abs/2408.11775)
Comments:
          submitted to Proc. IEEE Globecom

- **What's New**: 이 연구는 통신 네트워크를 위한 새로운 정보 검색 보강 생성(retrieval-augmented generation, RAG) 시스템을 개발하여 Phi-2 소형 언어 모델(small language model, SLM)을 오라클(oracle)로 활용하는 방법을 제안합니다. 이는 통신 분야에서 LLM(대형 언어 모델)이 기술 기준을 이해하는 데 어려움을 겪고 있다는 최근의 연구 결과를 반영한 것입니다.

- **Technical Details**: 제안된 RAG 시스템은 세 가지 주요 기술을 포함합니다: 포워드 룩 세멘틱 청킹(forward-looking semantic chunking), 재 순위 알고리즘(re-ranking algorithm), SelfExtend 기술. 이 시스템은 임베딩 유사성을 기반으로 구문 분석 중단점을 동적으로 결정하며, 여러 유사한 문맥을 처리하기 위해 가장 관련성이 높은 부분을 우선시합니다. 또한, Low-Rank Adaptation (LoRA) 기법을 사용하여 훈련 중 계산 효율성을 증가시키고 작은 데이터셋에 대해서도 효과적으로 미세 조정할 수 있도록 합니다.

- **Performance Highlights**: 본 연구의 실험 결과는 기존의 질문-답변(Question-Answering, QnA) 접근법보다 상당한 개선을 보여주었으며, 크기에서 약 880배 더 큰 GPT-4와 같은 대형 언어 모델을 초월하는 성능을 달성하였습니다. 이러한 성과는 통신 네트워크를 위한 SLM 활용에 있어 효율성과 성능의 균형을 제시합니다.



### Against All Odds: Overcoming Typology, Script, and Language Confusion in Multilingual Embedding Inversion Attacks (https://arxiv.org/abs/2408.11749)
Comments:
          11 pages, 4 figures, 7 tables

- **What's New**: 이 논문에서는 멀티링구얼 대형 언어 모델(LLMs)의 보안 취약점, 특히 embedding inversion 공격에 대한 새로운 연구를 제시하고 있습니다. 기존 연구는 주로 단일 언어 영어 모델에 집중된 반면, 이 연구는 8개 언어 계열 및 12개 스크립트에 걸쳐 20개 언어를 조사하여 멀티링구얼 모델의 취약성을 분석합니다.

- **Technical Details**: 본 연구는 악의적인 공격자가 외부의 inversion 모델을 사용하여 텍스트 시퀀스를 재구성하려고 수행하는 black-box inversion embedding attacks에 중점을 둡니다. 이를 위해, 연구팀은 다양한 언어와 스크립트에서 공격을 구현하여 멀티링구얼 모델에 대한 취약성을 평가하였습니다. 특히, 아랍 스크립트와 키릴 스크립트로 작성된 언어가 embedding inversion 공격에 특히 취약하다는 것을 발견하였으며, Indo-Aryan 언어 그룹이 가장 큰 위험에 처해 있음을 밝혔습니다.

- **Performance Highlights**: 이 논문의 연구 결과는 언어 가족과 스크립트의 조합이 embedding inversion 공격의 성공에 중요한 영향을 미친다는 것을 보여줍니다. 또한 'language confusion'이라는 개념이 공격 모델 성능에 영향을 미치며, 이는 공격자가 멀티링구얼 LLM에 대한 공격을 보다 효과적으로 수행할 수 있는 경향이 있음을 시사합니다.



### FocusLLM: Scaling LLM's Context by Parallel Decoding (https://arxiv.org/abs/2408.11745)
- **What's New**: FocusLLM은 긴 문맥을 처리할 수 있는 LLM의 새로운 프레임워크로, 이전 방법보다 낮은 훈련 비용으로 고성능을 발휘합니다. 이 모델은 긴 텍스트를 짧은 청크로 나누어 해당 청크에 대한 로컬 컨텍스트를 추가하고 병렬 디코딩 메커니즘을 통해 핵심 정보를 추출하여 통합합니다.

- **Technical Details**: FocusLLM은 기존 디코더 전용 LLM 아키텍처를 기반으로 하며, 원래 모델의 매개변수는 고정되어 강력한 일반화 능력을 유지합니다. 새로운 훈련 가능한 매개변수를 추가하여 병렬 디코딩의 결과를 집계할 수 있습니다. 긴 문장을 메모리 토큰과 로컬 토큰으로 나누고, 각 청크 별로 병렬 디코딩을 수행합니다.

- **Performance Highlights**: FocusLLM은 8K 입력 길이로 훈련되어 128K 이상의 긴 문서에서도 낮은 perplexity를 유지합니다. Longbench와 ∞-Bench에서의 평가 결과, FocusLLM은 기존 상한 모델들을 초과하는 성능을 보이며 긴 문서를 처리하는 데 뛰어난 능력을 발휘합니다.



### Xinyu: An Efficient LLM-based System for Commentary Generation (https://arxiv.org/abs/2408.11609)
- **What's New**: 이 논문은 중국어 주석 생성을 지원하기 위해 Xinyu라는 LLM(large language model) 기반 시스템을 소개합니다. 이 시스템은 주석 생성을 위한 기본 요구사항과 고급 요구사항을 충족시키기 위해 여러 단계를 나누어 생성 프로세스를 개선했습니다.

- **Technical Details**: 기본 요구사항을 충족하기 위해, 우리는 생성 프로세스를 순차적 단계로 분해하고, 각 단계에 대해 targeted supervised fine-tuning (SFT) 전략을 제안했습니다. 고급 요구사항을 위해, 우리는 주장을 평가하기 위한 argument ranking 모델을 도입하고, 최신 사건과 고전 도서를 포함하는 포괄적인 증거 데이터베이스를 구축했습니다. 또한, RAG(retrieval augmented generation) 기술을 활용하여 신뢰할 수 있는 증거를 생성합니다.

- **Performance Highlights**: Xinyu 시스템을 사용함으로써 주석 생성의 효율성이 크게 향상되었습니다. 평균 생성 시간이 4시간에서 20분으로 줄어들었으나, 생성된 주석의 품질은 유지되었습니다.



### Cause-Aware Empathetic Response Generation via Chain-of-Thought Fine-Tuning (https://arxiv.org/abs/2408.11599)
- **What's New**: 이 논문은 감정 생성에서 감정의 원인(reasoning)을 고려한 새로운 접근법을 제안합니다. 기존 연구들은 주로 화자의 감정 레이블에 중심을 두었으나, 이 연구는 감정의 원인 이해가 중요하다는 점을 강조합니다.

- **Technical Details**: 우리는 Chain-of-Thought (CoT) 프롬프트를 사용하여 대형 언어 모델(Large Language Models, LLMs)에서 감정과 원인을 통합하는 방법을 설계하였습니다. 이 알고리즘은 감정에 대한 반응의 다양성을 개선하고 내부 지식과 외부 지식 간의 갈등을 완화하는 데 기여합니다. 또한, COMET에서 제공하는 외부 지식을 프롬프트에 추가하여 모델의 성능을 향상시킵니다.

- **Performance Highlights**: LLaMA-7b에서 실험 결과, 우리의 접근법은 자동 및 인간 평가 모두에서 기존 방법 대비 우수한 성능을 보여주었고, ChatGPT보다도 더 공감적인 반응을 생성할 수 있음을 입증하였습니다.



### Large Language Models are Good Attackers: Efficient and Stealthy Textual Backdoor Attacks (https://arxiv.org/abs/2408.11587)
Comments:
          Under Review

- **What's New**: 본 논문에서는 Efficient and Stealthy Textual backdoor attack (EST-Bad) 방법을 제안하였습니다. 이 방법은 Large Language Models (LLMs)를 활용하여 텍스트 데이터에 대한 backdoor 공격의 효율성을 높이고 stealthiness를 개선합니다.

- **Technical Details**: EST-Bad는 세 가지 주요 전략으로 구성됩니다: 모델의 본래 결함을 트리거로 최적화하고, LLM을 이용하여 stealthy하게 트리거를 주입하며, 가장 영향력 있는 샘플을 선정하여 backdoor 주입을 수행합니다. 이 방법은 insertion-based와 paraphrase-based 공격의 장점을 통합하여, 텍스트 분류 데이터셋에서 선행 방법들에 비해 공격 성능과 stealthiness 모두에서 우수한 결과를 보여줍니다.

- **Performance Highlights**: 다양한 테스트에서 EST-Bad는 기존 방법에 비해 높은 공격 효율성과 stealthiness를 Demonstrate하며, 본 연구는 공격의 효율성을 크게 향상시키는 새로운 샘플 선택 전략(Similarity-based Selection Strategy)을 제안하여 응용 가능성을 높였습니다.



### Differentiating Choices via Commonality for Multiple-Choice Question Answering (https://arxiv.org/abs/2408.11554)
Comments:
          9 pages, accepted to ECAI 2024

- **What's New**: 본 논문에서는 선택지가 모두 질문과 관련이 있고 의미적으로 유사한 멀티 초이스 질문 응답(MCQA) 문제를 해결하기 위한 새로운 모델(DCQA)을 제안합니다. 기존의 MCQA 모델은 선택지를 개별적으로 평가하며, 다른 선택지가 제공하는 문맥을 무시했음을 지적합니다.

- **Technical Details**: DCQA 모델은 질문에 대한 선택지의 토큰 수준의 주의(attention)를 캡처하고, 모든 선택지들이 주목하는 질문의 토큰(즉, 공통적 요소)과 특정 선택지에 의해 주목되는 토큰(즉, 미세한 차이)을 분리합니다. 이를 통해 세부적인 문맥 정보를 이용해 선택지를 효과적으로 구분합니다.

- **Performance Highlights**: 연구에서 제안하는 DCQA 모델은 5개의 MCQA 벤치마크에서 일관되게 기존 모델들을 초과하는 성능을 보였습니다. 또한, 사례 연구를 통해 모델이 선택지 간의 공통점을 효과적으로 포착하고 질문 내에서 구별되는 단서를 식별하는 방식이 어떻게 MCQA 성능을 향상시키는지를 보여주었습니다.



### Memorization In In-Context Learning (https://arxiv.org/abs/2408.11546)
Comments:
          v1

- **What's New**: 이 연구는 In-context learning (ICL)의 성능 향상 메커니즘을 처음으로 밝혀냈으며, ICL이 기억된 훈련 데이터(memorized training data)를 어떻게 드러내는지를 보여줍니다.

- **Technical Details**: 연구는 ICL의 성능 향상이 zero-shot, few-shot, many-shot의 다양한 ICL 방식에서의 기억화(meorization)와 어떻게 연관되는지를 탐구합니다. 주요 발견 사항으로는 ICL이 대부분의 경우 zero-shot learning에 비해 기억화를 더 효과적으로 드러낸다는 점과, 레이블이 없는 demonstration이 기억화를 드러내는 데 가장 효과적이라는 점이 포함됩니다.

- **Performance Highlights**: 연구 결과, few-shot 환경에서 기억화가 높은 수준에 도달했을 때 ICL이 성능을 향상시키며, 또한 ICL이 zero-shot learning을 초과하는 경우 성능과 기억화 사이에는 매우 강한 상관관계가 존재한다는 사실이 밝혀졌습니다.



### Imagining from Images with an AI Storytelling Too (https://arxiv.org/abs/2408.11517)
- **What's New**: 본 논문에서는 단일 이미지나 이미지 시퀀스를 분석하여 이야기를 생성하는 방법을 제안합니다. 이 방법은 Narrative Art의 전통에서 영감을 받았으며, 비주얼 콘텐츠를 해석하고 매력적인 이야기를 생성하는 GPT-4o의 다양한 기능을 탐구합니다. 논문에서 제안하는 프로토타입 도구인 ImageTeller는 다양한 이미지 소스를 입력으로 받아 사용자에게 장르를 선택할 기회를 제공합니다.

- **Technical Details**: ImageTeller는 사용자가 입력한 이미지에서 이야기를 구성하는 그리드 구조로, 기본 장르(Comedy, Romance, Tragedy, Satire, Mystery)에 따라 플롯을 설정할 수 있는 기능을 갖추고 있습니다. 사용자는 데이터 중심의 이야기를 생성하거나, 프로토타입이 내러티브 구조를 자유롭게 처리하도록 선택할 수 있습니다. 또한, 사용자는 입력 이미지에 캡션을 추가하여 시스템의 해석에 영향을 줄 수 있습니다.

- **Performance Highlights**: ImageTeller는 사용자 상호작용을 통해 대체 장이나 일러스트 요청, 메시지를 수정 및 재시작할 수도 있는 기능을 지원하며, 이를 통해 입력 이미지 시퀀스에 따라 적절히 일관성 있는 이야기 생성을 목표로 합니다. 다양한 매체에서 획득한 2개 이상의 프레임으로 구성된 이미지 시퀀스를 처리하며, 더 넓고 풍부한 이야기 생성을 가능하게 합니다.



### IKUN for WMT24 General MT Task: LLMs Are here for Multilingual Machine Translation (https://arxiv.org/abs/2408.11512)
Comments:
          5 pages, 1 figure, 3 tables

- **What's New**: 이 논문에서는 WMT24의 일반 기계 번역 작업을 위해 개발된 IKUN과 IKUN-C라는 두 개의 다국어 시스템을 소개합니다. 이 시스템들은 각각 개방 시스템(open system)과 제한 시스템(constrained system)으로, Llama-3-8b와 Mistral-7B-v0.3를 기반으로 합니다.

- **Technical Details**: IKUN과 IKUN-C는 단일 모델을 사용하여 11개 언어 방향을 처리하도록 설계되었습니다. 이 시스템들은 두 단계 접근법을 기반으로 하며, 첫 번째 단계는 10개 언어의 단일 데이터에 대한 지속적인 사전 훈련(continuous pre-training)이고, 두 번째 단계는 11개 언어 방향에 대한 고품질 평행 데이터(fine-tuning)로 세밀하게 조정하는 것입니다.

- **Performance Highlights**: 자동 평가 메트릭에 따르면, IKUN-C는 모든 제한 시스템 중 6개의 1위와 3개의 2위를 기록하였고, IKUN은 개방 및 제한 시스템 모두에서 1위와 2위를 각각 1회와 2회 달성하였습니다. 이러한 결과는 대형 언어 모델(LLMs)이 효과적인 다국어 기계 번역을 위해 필요한 수준의 능숙함에 가까워지고 있음을 시사합니다.



### DocTabQA: Answering Questions from Long Documents Using Tables (https://arxiv.org/abs/2408.11490)
Comments:
          18 pages,5 figures

- **What's New**: 본 연구에서는 새로운 질문 응답 (QA) 문제 설정인 DocTabQA를 다룹니다. 이 설정에서는 긴 문서에서 질문에 답변하기 위해 문서 내용에서 파생된 구조화된 테이블로 답변을 조직하는 것이 목표입니다.

- **Technical Details**: 이 논문에서는 QTabA 데이터셋을 소개하며, 300개의 금융 문서와 수동으로 주석이 달린 1.5k 질문-테이블 쌍을 포함합니다. 초기에는 GPT-4와 같은 대형 언어 모델 (Large Language Models, LLMs)을 사용하여 기준선을 설정하지만, LLM이 복잡한 구조화된 출력을 생성하는 데 어려움을 겪는다는 사실도 인식하고 있습니다. 이를 위해 DocTabTalk이라는 두 단계 프레임워크를 제시하며, 관련 문장을 검색한 후 이를 기반으로 계층적 테이블을 생성합니다.

- **Performance Highlights**: 실험 평가를 통해 DocTabTalk이 제안된 DocTabQA 작업과 테이블 생성 작업에서 GPT-4의 성능을 크게 향상시킨다는 것을 입증하였습니다. QTabA 및 RotoWire 데이터셋에서의 종합적인 실험 결과가 이를 뒷받침합니다.



### The Self-Contained Negation Test S (https://arxiv.org/abs/2408.11469)
- **What's New**: 최근 Pretrained Language Models (PLMs)의 부정 해석 능력을 평가하기 위한 여러 방법론이 제안되었습니다. 본 연구는 기존의 Gubelmann과 Handschuh (2022) 연구를 기반으로, 입력의 극성에 따라 PLMs의 예측 수정 방식을 조사합니다.

- **Technical Details**: 이 연구는 'self-contained' 입력을 사용하여, 특정 동사의 극성에 따라 마스킹된 위치에 해당하는 토큰이 의미적으로 제외되거나 허용됩니다. 우리는 Gubelmann과 Handschuh의 실험을 재현하여 이 테스트의 결론을 약화시키는 결함을 발견했습니다. 우리는 개선된 Self-Contained Neg Test를 제안하며, 해당 테스트는 영어에서 동사 부정의 유무에 따라 최소 쌍(minimal pairs)으로만 변형된 예시를 기반으로 합니다.

- **Performance Highlights**: Self-Contained Neg Test를 roberta 및 bert base와 large 모델에 적용한 결과, roberta-large만이 기대에 부합하는 경향을 보였고, bert-base는 부정에 대해 크게 무관했습니다. 그러나 모든 모델에서 상당수의 테스트 인스턴스에서 상위 1 예측이 문맥에서 의미적으로 금지된 토큰으로 남아 있어, 부정 현상에 대한 적절한 처리를 위한 개선 여지가 많음을 보여줍니다.



### Expanding FLORES+ Benchmark for more Low-Resource Settings: Portuguese-Emakhuwa Machine Translation Evaluation (https://arxiv.org/abs/2408.11457)
Comments:
          Open Language Data Initiative 2024 shared tasks

- **What's New**: 이 연구는 Open Language Data Initiative의 일환으로 Emakhuwa라는 저자원 언어를 위한 FLORES+ 평가 세트를 확장했습니다. 이 세트는 포르투갈어에서 Emakhuwa로 번역된 dev 및 devtest 세트를 포함하고 있으며, 번역 과정과 품질 보증 조치를 상세히 설명합니다.

- **Technical Details**: Emakhuwa는 모잠비크에서 약 900만 명이 사용하는 Bantu 언어 계열에 속합니다. 연구에서는 Neural Machine Translation 시스템을 학습하고 기존의 다국어 번역 모델을 미세 조정하여 baseline 결과를 제시했습니다. 품질 보증을 위한 포스트 편집(post-editing) 및 적합성 평가가 포함된 다양한 품질 검사가 수행되었습니다.

- **Performance Highlights**: Emakhuwa에 대한 기초 모델들은 평가 세트에서 저조한 성과를 보였으며, 이는 기계 번역 품질 향상을 위한 추가 연구의 필요성을 강조합니다. 특히 Emakhuwa에서 철자 불일치는 여전히 큰 도전 과제로 남아 있습니다.



### Distributional Properties of Subword Regularization (https://arxiv.org/abs/2408.11443)
Comments:
          4 pages + 4 page appendix. 3 figures

- **What's New**: 이 연구는 자연어 처리에서 자주 사용되는 subword regularization 방법을 분석하며, 기존의 stochastic tokenization 방식이 단일한 tokenization에 의존하는 경향이 있음을 밝혀냈습니다. 새로운 샘플링 방법을 제안하여 모델의 성능을 개선하는 방법을 모색하였습니다.

- **Technical Details**: 토크나이징(tokenization)은 자연어 처리 파이프라인에서 첫 단계로, 원본 텍스트를 모델이 이해할 수 있는 형식으로 변환하는 과정입니다. 본 연구에서는 BPE(Bidirectional Encoder Representations from Transformers)와 MaxMatch와 같은 두 가지 인기 있는 subword 토크나이저의 stochastic dropout 변형을 분석하였습니다. 이들 방법이 생성하는 tokenization 분포가 치우친 현상을 발견하고, 보다 uniform한 샘플링 방식을 제안하여 이로 인해 기계 번역 품질이 향상될 수 있음을 보였습니다.

- **Performance Highlights**: 제안된 uniform sampling 방식은 BPE 및 MaxMatch dropout의 stochastic 요소를 교체하여 다수의 번역 작업에서 모델 품질을 개선하였습니다. 실험 결과, 새로운 방법이 기존 방식보다 효율적인 regularization 및 augmentation을 제공하며, 모델 훈련 시 보다 다양하고 고유한 tokenization을 노출시켜 성능을 향상시킨 것으로 나타났습니다.



### LAHAJA: A Robust Multi-accent Benchmark for Evaluating Hindi ASR Systems (https://arxiv.org/abs/2408.11440)
- **What's New**: 이 논문에서는 인도에서 사용되는 다양한 억양을 가진 힌디 음성 인식 시스템(ASR)을 평가하기 위한 새로운 벤치마크인 LAHAJA를 소개합니다. LAHAJA는 다양한 주제와 사례에 대해 기록된 12.5시간 분량의 힌디 오디오를 포함하며, 이는 132명의 화자로부터 수집되었습니다.

- **Technical Details**: LAHAJA 데이터셋은 132명(122명의 비원어민 포함)의 참가자로부터 수집되었으며, 19개 언어 가족에 걸쳐 있습니다. 이 데이터는 읽기 음성과 즉흥 대화로 구성되어 있으며, Microsoft의 Karya 플랫폼을 사용해 녹음되었습니다. 환경 분석에 따라 다양한 억양과 콘텐츠 범주에 대한 세밀한 분석을 진행했습니다.

- **Performance Highlights**: 기존 공개 모델과 상업 모델이 LAHAJA에서 저조한 성능을 보였으며, 다양한 언어 데이터를 사용한 모델이 상대적으로 높은 성능을 나타냈습니다. 특히 북동부 및 남인도 화자에 대해 성능 저하가 관찰되었습니다. 전체 데이터와 모델은 공개되어 있으며, 다중 억양 힌디 ASR 시스템 연구를 위한 기초 자료로 제공됩니다.



### Diagnosing and Remedying Knowledge Deficiencies in LLMs via Label-free Curricular Meaningful Learning (https://arxiv.org/abs/2408.11431)
Comments:
          Under Review

- **What's New**: 본 논문에서는 라벨이 없는 설정에서 LLM의 지식 결핍을 진단하고, 이를 해결하기 위한 학습 프레임워크인 LaMer를 제안합니다. LaMer는 상대 엔트로피(relative entropy)를 활용하여 LLM의 지식 결핍을 자동적으로 진단하며, 이는 기존의 라벨 데이터에 의존하지 않습니다.

- **Technical Details**: LaMer 프레임워크는 두 가지 주요 단계로 구성됩니다. 첫째, LLM의 지식 결핍을 진단하기 위해 상대 엔트로피를 사용합니다. 둘째, 커리큘럼 유의미 학습(curricular meaningful learning)을 적용해 지식 결핍을 점진적으로 해결합니다. 이를 통해 데이터 보강(augmentation)을 적응적으로 수행하며, 다양한 시나리오에서 결핍의 심각도에 따라 해결 전략을 설계합니다.

- **Performance Highlights**: LaMer는 40%의 훈련 데이터만으로도 여러 LLM의 지식 결핍을 효과적으로 진단하고 개선하는 성과를 보여줍니다. 7개의 OOD(Out-of-Distribution) 추론 및 언어 이해 벤치마크에서 기존의 기법들과 비교했을 때, LaMer는 더욱 효율적이고 효과적으로 LLMs의 성능을 향상시킬 수 있습니다.



### Towards "Differential AI Psychology" and in-context Value-driven Statement Alignment with Moral Foundations Theory (https://arxiv.org/abs/2408.11415)
Comments:
          8 pages, 6 tables

- **What's New**: 이 연구는 개인화된 언어 모델과 설문 참여자 간의 정렬을 조사하고, 윤리적 원칙에 기반한 대화형 모델을 생성할 수 있는 새로운 방법론을 제안합니다.

- **Technical Details**: 가장 진보된 통계적 언어 모델을 개인화된 정치 성향으로 조정하여, 입력된 설문지에 대한 행동을 분석합니다. Moral Foundation Theory (MFT)를 사용하여 각 정치적 정향의 다양한 사람들을 모델링합니다.

- **Performance Highlights**: 적응된 모델들이 정치 이념에 대한 설문 결과를 잘 표현하지 못하며, 언어 모델을 사용하여 사회적 상호작용을 모방하기 위해서는 의미 있는 개선이 필요합니다.



### MoE-LPR: Multilingual Extension of Large Language Models through Mixture-of-Experts with Language Priors Routing (https://arxiv.org/abs/2408.11396)
- **What's New**: 본 논문에서는 MoE-LPR (Mixture-of-Experts with Language Priors Routing)이라 불리는 새로운 다국어 모델 학습 방법을 제안합니다. 이 방법은 준지도 학습을 통해 언어 능력을 확장하면서도 기존 언어에 대한 지식을 유지하는 것을 목표로 합니다.

- **Technical Details**: MoE-LPR은 두 단계로 구성된 학습 전략을 사용합니다. 첫 번째 단계에서는 원래의 파라미터를 동결하고 새로운 전문가들을 추가하여 Mixture-of-Experts (MoE) 아키텍처로 모델을 변환합니다. 이후, LPR (Language Priors Routing) 메커니즘을 통해 원래 언어의 지식을 복구하는 방법으로 1% 미만의 재생(replay) 데이터를 사용합니다.

- **Performance Highlights**: 여러 벤치마크에서 테스트한 결과, MoE-LPR은 다른 후속 훈련(post-pretraining) 방법들을 능가하는 성능을 보였습니다. 또한, MoE 아키텍처는 동일한 추론 오버헤드를 유지하면서 모델의 전체 파라미터 수를 증가시킬 수 있게 해주며, 확장 언어의 능력을 향상시키고 기존 언어의 숙련도를 보존하는 데 효과적임을 입증했습니다.



### First Activations Matter: Training-Free Methods for Dynamic Activation in Large Language Models (https://arxiv.org/abs/2408.11393)
- **What's New**: 이 논문은 훈련이 필요 없는 Threshold-based Dynamic Activation(TDA) 방법을 소개하며, 이는 다양한 아키텍처에서 모델의 내재적 희소성을 이용하여 성능에 큰 손실 없이 생성 속도를 18-25% 향상시키는 것을 목표로 한다.

- **Technical Details**: TDA 방법은 이전의 ReLU 활성화 함수 사용에 대한 한계를 극복하고, 특정 머리(head)나 뉴런을 선택적으로 활성화함으로써 LLM의 희소성을 최적화한다. 또한, 역사적 활성화 불확실성과 의미와 무관한 활성화 관성의 두 가지 주요 특성을 분석하여 DA 방법에 대한 이론적 토대를 구축한다.

- **Performance Highlights**: TDA 방법은 기존의 동적 활성화 기술에 비해 성능 저하가 극소화되면서도 생성 지연(latency)을 효과적으로 줄이며, LLM의 효율성과 효과성을 높이는 중요한 통찰을 제공한다.



### On the Interchangeability of Positional Embeddings in Multilingual Neural Machine Translation Models (https://arxiv.org/abs/2408.11382)
Comments:
          Under Review

- **What's New**: 본 논문은 전통적인 Sinusoidal Positional Embeddings (PEs) 대신 상대적 PEs를 사용하는 것의 효율성을 탐구합니다. 이는 긴 문맥이나 문서 수준의 번역에서 성능을 향상시킬 수 있음을 보여줍니다.

- **Technical Details**: RoPE (Rotary Positional Embeddings)와 ALiBi (Attention with Linear Biases)와 같은 상대적 위치 정보를 사용하는 방법을 적용하여 pre-trained NMT 모델에서 Sinusoidal PEs를 교체하는 방법을 제시합니다. 이 방법은 성능 손실 없이 가능합니다.

- **Performance Highlights**: 모델을 fine-tuning(미세조정)함으로써 RoPE와 ALiBi를 효과적으로 적용할 수 있으며, NoPE(없는 포지셔널임베딩) 모델은 consistently(지속적으로) 성능이 떨어지는 것을 확인했습니다.



### RAGLAB: A Modular and Research-Oriented Unified Framework for Retrieval-Augmented Generation (https://arxiv.org/abs/2408.11381)
Comments:
          6 pages, 3 figures

- **What's New**: 이번 논문에서는 Retrieval Augmented Generation (RAG) 알고리즘의 비교 및 새로운 알고리즘 개발을 위한 모듈형 오픈소스 라이브러리인 RAGLAB을 소개하고 있습니다. RAGLAB은 6개의 기존 알고리즘을 재현하고 10개의 벤치마크를 통해 공정한 비교를 가능하게 합니다.

- **Technical Details**: RAGLAB은 BERT 기반의 고성능 모델인 Contriever 및 ColBERT를 통합하고, 모듈화된 아키텍처를 제공하여 RAG 시스템의 핵심 구성 요소를 쉽게 활용할 수 있도록 합니다. 또한 사용자 친화적인 인터페이스와 interactive 모드를 통해 교육적 활용 및 시연이 가능합니다. RAGLAB은 메트릭스, 키 설정, 베이스라인 및 실험 변수를 표준화하여 알고리즘의 공정한 비교를 제공합니다.

- **Performance Highlights**: RAGLAB은 사용자들이 단일 RAG 알고리즘 내에서 여러 생성기를 동시에 로드할 수 있도록 GPU 관리 모듈을 개발하였고, 최소 0.1초의 대기 시간으로 고효율 검색 서비스를 제공합니다. 또한, RAGLAB에서 제공하는 10개의 벤치마크를 통해 알고리즘의 성능을 빠르고 효율적으로 비교할 수 있습니다.



### GeoReasoner: Reasoning On Geospatially Grounded Context For Natural Language Understanding (https://arxiv.org/abs/2408.11366)
Comments:
          Accepted by International Conference on Information and Knowledge Management 2024

- **What's New**: GeoReasoner는 자연어에서 지리적 기반 추론을 개선하기 위해 언어 정보와 지리 정보를 융합하는 새로운 언어 모델입니다. 이 모델은 지리 데이터베이스와의 통합을 통해 보다 현실적이고 효과적인 지리적 개체 인식을 가능하게 합니다.

- **Technical Details**: GeoReasoner는 Large Language Models (LLMs)를 활용하여 상세한 위치 설명을 생성하고, 방향과 거리 정보를 공간 임베딩으로 변환하여 학습합니다. 이 과정에서는 pseudo-sentences를 사용하여 공간 정보를 인코딩하고, geospatial contrastive loss와 masked language modeling loss를 적용합니다.

- **Performance Highlights**: GeoReasoner는 toponym recognition, toponym linking, geo-entity typing의 세 가지 작업에서 기존의 최첨단 모델들을 초월하는 성능을 보여줍니다. 실험 결과는 GeoReasoner가 지리적 문맥의 이해에 있어 상당한 진전을 이루었음을 입증합니다.



### Clinical Context-aware Radiology Report Generation from Medical Images using Transformers (https://arxiv.org/abs/2408.11344)
Comments:
          21 pages, 6 figures, 8 tables

- **What's New**: 이번 연구에서는 흉부 X-ray로부터 방사선 보고서를 생성하기 위해 Transformer 모델을 사용하고, LSTM 기반의 디코더와의 성능 비교를 통해 Transformer 모델의 장점을 강조합니다.

- **Technical Details**: 이 연구는 공공 데이터셋 IU-CXR를 사용하여 CNN을 인코더로, Transformer를 디코더로 활용하여 방사선 보고서를 생성합니다. Transformer 구조는 전통적인 RNN에 비해 훈련이 더 빠르고 간단하며, 장기 의존성을 기억하는 self-attention 메커니즘을 제공합니다.

- **Performance Highlights**: 실험 결과, Transformer 모델이 LSTM 모델보다 우수한 결과를 보였으며, 훈련 속도가 크게 향상되었습니다. 또한, 보고서의 일관성과 진단 가치를 평가하기 위한 새로운 평가 지표가 필요함을 제시합니다.



### BURExtract-Llama: An LLM for Clinical Concept Extraction in Breast Ultrasound Reports (https://arxiv.org/abs/2408.11334)
Comments:
          This paper has been accepted as the oral paper for the HCHM workshop, ACM Multimedia 2024

- **What's New**: 본 연구는 유방 초음파 보고서에서 임상 정보를 추출하기 위해 인하우스 LLM(대규모 언어 모델) 개발 파이프라인을 소개합니다. 특히 GPT-4를 사용하여 작은 라벨이 지정된 데이터셋을 생성한 후, 이를 기반으로 Llama3-8B 모델을 미세 조정하는 방법을 제시합니다.

- **Technical Details**: 본 연구는 유방 초음파 보고서에서 임상 정보를 효과적으로 추출하기 위해 3단계 파이프라인을 구성합니다: 1) 관찰 및 인상 정보 추출, 2) GPT-4를 사용한 훈련 라벨 생성, 3) Q-LoRA를 통해 Llama3-8B 모델 미세 조정. 이를 통해 84.6%의 평균 F1 점수를 기록하여 GPT-4와 유사한 정확성을 달성하였습니다. 

- **Performance Highlights**: 개발된 BURExtract-Llama 모델은 유방 초음파 보고서에서 임상 정보를 추출하는 데 있어 GPT-4와 동등한 성능을 발휘하며, 비용 효율성과 데이터 프라이버시를 강화할 수 있는 가능성을 보여줍니다.



### Plug, Play, and Fuse: Zero-Shot Joint Decoding via Word-Level Re-ranking Across Diverse Vocabularies (https://arxiv.org/abs/2408.11327)
Comments:
          Under Review

- **What's New**: 최근 자연어 처리(NLP) 모델들이 멀티모달 입력 처리 및 특정 도메인에서 우수한 성능을 보이고 있습니다. 그러나 실세계에서 다중 모달 번역과 같은 작업은 이러한 여러 모델의 강점을 결합해야 합니다. 본 연구는 추가 교육 없이도 디코딩 단계에서 모델을 합치는 새로운 제로샷 앙상블(zero-shot ensembling) 전략을 제안합니다.

- **Technical Details**: 제안하는 방법론은 단어 수준에서 점수를 결합하여 디코딩 중에 다중 모델을 통합하는 것입니다. 이를 위해, 각 단어의 완료 여부를 예측하는 휴리스틱(heuristic)을 사용하여 디코딩 과정 중에 비정상적인 상태를 피합니다. 모델 간의 단어 수준 재등급(online re-ranking)을 통해 완벽하지 않은 가설에도 정확한 확률 예측을 가능하게 하는 접근 방식을 강조합니다.

- **Performance Highlights**: 실험 결과, 본 연구의 방법론은 음성과 이미지를 고려한 번역을 가능하게 하며, 번역 품질 또한 개선되는 것을 입증했습니다. 특히, 두 가지 모달리티의 정보를 포함하는 타겟 실험을 통해 성능 향상이 관찰되었습니다.



### Towards Evaluating Large Language Models on Sarcasm Understanding (https://arxiv.org/abs/2408.11319)
- **What's New**: 본 논문은 LLMs(대형 언어 모델)가 풍자(sarcasm) 이해에서의 성과에 대한 논의의 타당성을 심도 깊게 검토합니다. 특히 11개 최첨단(SoTA) LLM과 8개 PLM(사전 훈련된 언어 모델)을 활용하여 풍자 탐지 성능을 평가합니다.

- **Technical Details**: 총 6개의 벤치마크 데이터셋에서 zero-shot IO prompting, few-shot IO prompting, chain of thought (CoT) prompting의 세 가지 접근 방식으로 평가하였습니다. 실험 결과, 현재 LLM은 지도된 PLM 기반 풍자 탐지 기준에 미치지 못하는 성능을 보여주었습니다.

- **Performance Highlights**: 1. GPT-4는 다른 LLM들과 비교해 다양한 prompting 방법에서 평균 14.0% 향상된 성능을 보이며, 가장 우수한 성능을 기록했습니다. 2. few-shot IO prompting 방법이 zero-shot IO 및 few-shot CoT보다 평균 4.5% 높은 성능을 나타냅니다. 3. LLM의 풍자 이해에는 여전히 상당한 개선이 필요하다는 점을 강조합니다.



### RedWhale: An Adapted Korean LLM Through Efficient Continual Pretraining (https://arxiv.org/abs/2408.11294)
- **What's New**: 이번 연구에서는 한국어 처리를 위해 특별히 설계된 모델 RedWhale를 소개합니다. RedWhale는 효율적인 지속적 사전 훈련(continual pretraining) 접근 방식을 사용하여 지난 연구에서 간과되었던 한국어 NLP의 격차를 해소하기 위한 여러 혁신적인 개선 사항을 포함하고 있습니다.

- **Technical Details**: RedWhale는 포괄적인 한국어 말뭉치 전처리 파이프라인, 전문 한국어 토크나이저(tokenizer), 최적화된 모델 초기화 기법, 다단계 사전 훈련 전략을 통해 개발되었습니다. 이러한 방식들은 훈련 시간을 단축하고 계산 비용을 줄이는 동시에 높은 정확도와 이해력을 유지합니다. 영어 모델을 활용한 교차 언어 전이 학습(cross-lingual transfer learning)을 통해 한국어 처리를 개선했습니다.

- **Performance Highlights**: 실험 결과, RedWhale는 한국어 NLP 벤치마크인 KoBEST에서 다른 주요 모델들보다 우수한 성능을 보였습니다. 특히 97억 개의 토큰으로 사전 훈련한 후에도 수렴(convergence) 징후를 보이지 않아 추가 훈련을 통한 성능 향상이 가능성을 시사합니다.



### Counterfactuals As a Means for Evaluating Faithfulness of Attribution Methods in Autoregressive Language Models (https://arxiv.org/abs/2408.11252)
Comments:
          17 pages, 6 figures

- **What's New**: 이 연구에서는 자가 회귀 (autoregressive) 언어 모델의 설명 방법 평가를 위해 반사실적 (counterfactual) 생성을 활용하는 새로운 기술을 제안합니다. 기존의 평가 방법들이 입력을 변형하여 OOD (out-of-distribution) 입력을 초래하는 문제를 해결하고, 더욱 신뢰할 수 있는 평가 프로토콜을 제시합니다.

- **Technical Details**: 자기 회귀 언어 모델(GPT-2, LLaMA 등)의 신뢰성 평가를 위해 반사실적 생성 기술을 사용하여, 특정 입력 토큰을 자연스럽고 유창하며 모델의 데이터 분포 내에서 평가할 수 있게 합니다. 각 입력 특성의 중요성을 나타내는 스칼라 값을 할당하는 방법을 사용하여 피드백을 제공합니다.

- **Performance Highlights**: 여러 유명한 설명 방법(gradient norm, Erasure, KernelSHAP 등)에 우리 방식으로 평가한 결과, 조정된 모델과 기본 모델 간의 OOD 데이터 처리에 대한 민감도 차이를 보여줍니다. 이는 자가 회귀 언어 모델의 신뢰성을 높이는 데 기여하며, 실제 모델의 결정 과정을 이해하는 데 도움을 줍니다.



### Unboxing Occupational Bias: Grounded Debiasing LLMs with U.S. Labor Data (https://arxiv.org/abs/2408.11247)
Comments:
          Accepted in AAAI Spring Symposium 2024

- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)의 내재된 편향을 분석하고, 미국 노동통계국(NBLS) 데이터를 활용하여 언어 모델의 결과물과 비교하는 새로운 접근 방식을 제안합니다. 또한, NBLS 데이터를 사용하여 편향을 완화하는 효과적인 메커니즘을 소개합니다.

- **Technical Details**: 연구에서는 Zero-Shot Prompting (ZSP)과 Few-Shot Prompting (FSP)이라는 두 가지 방법을 사용했고, 2,500개의 샘플로 폭넓은 LLM 관점에서 편향을 분석했습니다. Kolmogorov-Smirnov (KS) 테스트와 ANOVA 테스트를 통해 편향 분석을 진행하였으며, 32개의 NBLS 예시를 통해 평균 65%의 편향 감소를 달성했습니다.

- **Performance Highlights**: 연구 결과는 기존의 편향 탐지 기법이 간과하는 상당한 편향 수준을 드러내었으며, 제안된 편향 완화 방법이 외부 데이터셋에 의존하지 않고도 효과적으로 편향 점수를 줄일 수 있음을 보여주었습니다.



### CoDi: Conversational Distillation for Grounded Question Answering (https://arxiv.org/abs/2408.11219)
Comments:
          13 pages

- **What's New**: 이 논문에서는 Conversational Distillation (CoDi)이라는 새로운 데이터 증류 프레임워크를 소개하며, SLMs (Small Language Models)의 대화 능력을 향상시키기 위한 방법을 제시합니다. CoDi를 통해 대규모의 조교 스타일 데이터셋을 다양하고 조작 가능하게 합성할 수 있습니다.

- **Technical Details**: CoDi는 대화가 아닌 일반적인 작업에 대해서도 적용될 수 있는 프레임워크로, 다단계 대화를 위한 데이터 합성을 목표로 합니다. 이 방법론은 대화 그래프, 턴 기반 프롬프트 증강 및 명시적 언어적 특징을 사용하여 자연스러운 대화를 생성합니다.

- **Performance Highlights**: CoDi로 훈련된 SLMs는 인간 주석 데이터로 훈련된 모델과 동등한 성능을 보이며, 데이터 합성을 통한 대규모 데이터셋 생성을 통해 큰 모델들보다 뛰어난 성능을 보여줍니다.



### Reading with Inten (https://arxiv.org/abs/2408.11189)
- **What's New**: 이 논문은 Retrieval Augmented Generation (RAG) 시스템이 외부 정보 소스와 통합되는 방법을 다룹니다. 특히, 인간 커뮤니케이션의 뉘앙스를 이해하는 데 어려움이 있는 RAG 시스템에서의 풍자(sarcasm) 처리 방법을 연구합니다.

- **Technical Details**: 저자들은 Natural Questions의 Wikipedia 검색 코퍼스를 기반으로 합성된 풍자가 포함된 텍스트를 생성하고, 이러한 텍스트가 RAG 파이프라인의 retriever와 reader 부분의 성능에 미치는 영향을 실험합니다. 그들은 풍자를 처리하기 위한 프롬프트 시스템을 개발하여 모델이 풍자가 있는 상황에서 응답을 해석하고 생성하는 능력을 향상시킵니다.

- **Performance Highlights**: 종합적인 ablation studies를 통해 제안한 방법의 효과성을 입증하며, 풍자가 포함된 콘텐츠를 처리하는 데 있어 성능 향상을 나타냅니다.



### Combining Objective and Subjective Perspectives for Political News Understanding (https://arxiv.org/abs/2408.11174)
- **What's New**: 이 논문은 정치적 텍스트의 대량 분석을 위해 자동 콘텐츠 분석 도구를 사용하는 연구자들과 실무자들에게 유용한 새로운 텍스트 분석 프레임워크를 제시합니다. 기존의 방법들이 주로 객관적인 측면에 초점을 맞추었던 것에 반해, 제안된 프레임워크는 주관적인 측면을 세분화하여 관계의 설명 가능성을 높입니다.

- **Technical Details**: 이 프레임워크는 자연어 처리(NLP), 정보 검색 기법, 뉴스 아웃렛 메타데이터 및 외부 지식 기반을 결합하여 유연한 결과 집합을 가능하게 합니다. 특히, Target-dependent Sentiment Classification (TSC) 기법을 통해 세밀한 주관적 분석을 수행하며, 여러 언어와 국가에서 활용될 수 있도록 설계되었습니다.

- **Performance Highlights**: 프레임워크는 프랑스 정치 뉴스 자료를 통해 구현되었고, 주요 발견으로는: 주요 뉴스 아웃렛이 주류 정치 지향을 균형 있게 보도하지만 극단 좌파 및 우파에는 각각 긍정적, 부정적 편향이 나타났습니다. 정치 주제와 관련된 감정 점수의 변동이 있으며, 여성 정치인에 대한 감정 점수가 남성 정치인보다 높고,  나이 편향은 고령 정치인에게 상대적으로 나타났습니다.



### Tabular Transfer Learning via Prompting LLMs (https://arxiv.org/abs/2408.11063)
Comments:
          COLM 2024

- **What's New**: 본 논문은 제한된 라벨 데이터로 학습하는 문제를 해결하기 위해 Transfer Learning(전이학습)의 새로운 접근 방식을 제안합니다. 특히, Tabular tasks(표 형식 작업)에 초점을 맞추며, 이는 기존의 비전(vision)이나 언어(language) 작업에 비해 연구가 부족했던 영역입니다.

- **Technical Details**: 제안하는 새로운 프레임워크는 Prompt to Transfer (P2T)로, 이는 대형 언어 모델(Large Language Models, LLMs)을 활용하여 라벨이 없는(또는 이질적인) 소스 데이터로부터 목표 작업에 대한 예제를 생성합니다. P2T는 소스 데이터셋에서 목표 작업의 특징과 강한 상관관계를 가지는 열(Column feature)을 식별하여 해당 작업과 관련된 가상의 예시(pseudo-demonstrations)를 생성합니다.

- **Performance Highlights**: 실험 결과, P2T는 다양한 표준 표 작업(Tabular learning benchmarks)에서 이전 방법들을 초월하는 성과를 기록하며, 아직 탐구되지 않은 표형 전이 학습 문제에 대해 좋은 가능성을 보입니다.



### Interactive-T2S: Multi-Turn Interactions for Text-to-SQL with Large Language Models (https://arxiv.org/abs/2408.11062)
Comments:
          15 pages, 7 figures

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 강력한 추론 능력을 활용하여 text-to-SQL 파싱을 탐구합니다. 우리는 데이터베이스와의 직접 상호작용을 통해 SQL 쿼리를 생성하는 Interactive-T2S라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 효율적이고 사전 구성된 리소스가 필요하지 않은 방법으로 SQL 생성 과정을 단계별로 해석할 수 있는 시스템입니다.

- **Technical Details**: Interactive-T2S는 LLM과 데이터베이스 간의 다중 상호작용을 통해 SQL 쿼리를 생성하는 새로운 프레임워크입니다. 이 프레임워크에서는 LLM이 생각하고 행동하는 사고-행동 패러다임(thought-action paradigm) 아래 수행되며, SQL 쿼리 생성을 위해 네 가지 일반 도구를 포함하고 있습니다. 실험에서는 BIRD-Dev 데이터셋을 사용하여 오라클 지식 없이 단 두 개의 예제를 사용하여도 최첨단 결과를 달성했습니다.

- **Performance Highlights**: 우리의 방법은 두 개의 예제만으로도 뛰어난 성능을 보여주며, 넓은 테이블을 효율적으로 처리할 수 있는 강력한 기능을 입증했습니다. Comprehensive testing on the Spider-Dev, BIRD-Dev, and their variant datasets showcased that our approach achieves significant results with minimal exemplar input.



### StructuredRAG: JSON Response Formatting with Large Language Models (https://arxiv.org/abs/2408.11061)
Comments:
          Preprint. 10 pages, 6 figures

- **What's New**: 이번 연구에서는 LLM(대형 언어 모델)의 구조화된 출력 생성 능력을 평가하기 위해 StructuredRAG라는 새로운 벤치마크를 도입했습니다. 이 벤치마크는 JSON과 같은 특정 형식의 응답을 따르는지 확인하기 위한 6개의 테스트로 구성되어 있습니다.

- **Technical Details**: StructuredRAG는 LLM이 Zero-Shot Learning 방식으로 JSON 응답 형식을 따르는 능력을 측정하기 위해 설계되었습니다. f-String과 Follow the Format(FF)이라는 두 가지 프롬프트 전략을 사용하여 Gemini 1.5 프로와 Llama 3 8B-instruct 두 모델을 평가했습니다. 실험 결과, 성공률은 평균 82.55%였으며, 작업 별로 성능의 편차가 큰 것으로 나타났습니다.

- **Performance Highlights**: 두 모델 모두 다양한 테스트에서 비슷한 성과를 보였으나 Gemini 1.5 Pro가 평균적으로 93.4%의 성공률을 기록하여 Llama 3 8B-instruct의 71.7%보다 우수한 성능을 보였습니다. 특히, 단일 문자열, 정수형, 불리언 값과 같은 간단한 출력 형식에서는 성능이 뛰어난 반면, 리스트 출력 및 복합 객체의 경우는 상당한 성능 저하가 관찰되었습니다.



### DreamFactory: Pioneering Multi-Scene Long Video Generation with a Multi-Agent Framework (https://arxiv.org/abs/2408.11788)
Comments:
          13 pages, 8 figures

- **What's New**: 이 논문에서는 긴 비디오 생성을 위한 새로운 모델인 DreamFactory를 소개합니다. 기존의 비디오 생성 모델은 짧은 클립에서는 뛰어난 성과를 보였으나, 다중 장면이 포함된 긴 비디오에서는 어려움을 겪었습니다. DreamFactory는 다중 에이전트 협업 원칙과 Key Frames Iteration Design Method를 활용하여 이러한 문제를 해결합니다.

- **Technical Details**: DreamFactory는 LLM(대형 언어 모델)을 기반으로 하며, 제작 과정에서 각 LLM이 감독, 미술 감독, 각본가 및 아티스트와 같은 역할을 맡아 협력합니다. 이 프레임워크는 시나리오 작성, 스토리보드 생성, 캐릭터 디자인, 키프레임 개발 등을 포함하여 비디오 생성을 자동화합니다. 본 모델은 비디오 세그먼트 간의 일관성을 확보하기 위해 특정 키프레임 반복 방법을 적용합니다.

- **Performance Highlights**: DreamFactory는 UTF-101 및 HMDB51 데이터셋을 사용하여 평가한 결과, 기존 모델에 비해 상당한 성능 개선이 있음을 보여주었습니다. 특히, 우리 모델이 생성한 긴 비디오는 수작업으로 생성된 비디오보다 평균적인 품질을 초과하는 것으로 평가되었습니다.



### Efficient Detection of Toxic Prompts in Large Language Models (https://arxiv.org/abs/2408.11727)
Comments:
          Accepted by the 39th IEEE/ACM International Conference on Automated Software Engineering (ASE 2024)

- **What's New**: ToxicDetector는 경량(greybox) 방식으로 설계된 독성 프롬프트 탐지 방법으로, 기존 방법의 한계점을 극복하여 LLM에서 독성 프롬프트를 효율적으로 탐지합니다.

- **Technical Details**: ToxicDetector는 LLM을 활용하여 독성 개념 프롬프트를 생성하고, 임베딩 벡터(embedding vectors)를 통해 피쳐 벡터(feature vectors)를 형성합니다. 그 후 MLP(Multi-Layer Perceptron) 분류기를 사용하여 프롬프트를 분류합니다.

- **Performance Highlights**: ToxicDetector는 LLama 모델, Gemma-2 및 다양한 데이터셋을 평가한 결과 96.39%의 높은 정확도와 2.00%의 낮은 허위 긍정률(false positive rate)을 기록하였으며, 0.0780초의 신속한 처리 시간을 자랑하여 실시간 응용 프로그램에 매우 적합합니다.



### Drama Engine: A Framework for Narrative Agents (https://arxiv.org/abs/2408.11574)
Comments:
          10 pages, 2 figures, 2 tables

- **What's New**: 이번 기술 보고서는 이야기 목적으로 설계된 대형 언어 모델과의 에이전틱 상호작용을 위한 새로운 프레임워크인 Drama Engine을 소개합니다. 이 프레임워크는 다중 에이전트 시스템 원칙을 적용하여 시간에 따라 발전하고 사용자 및 서로 상호작용할 수 있는 동적이고 상황 인식이 가능한 동반자를 생성합니다.

- **Technical Details**: Drama Engine의 핵심 기능은 다음과 같습니다: \n1. 다중 에이전트 워크플로우 및 위임: 여러 에이전트 간의 대화를 조정하는 방식으로, 에이전트는 더 복잡한 작업을 차일드를 통해 위임할 수 있습니다.\n2. 동적 프롬프트 조합: 프롬프트는 맥락에 따라 조립됩니다. 이 맥락에는 다른 채팅 참가자, 작업 상태 등 여러 데이터가 포함됩니다.\n3. 모델 및 벤더 비민감성: Drama Engine은 OpenAI의 API 표준을 지원하는 모든 백엔드를 사용할 수 있습니다.\n4. 동반자의 시간 발전 및 기분 시스템: 동반자는 시간이 지남에 따라 발전하고 상황에 따라 기분이 변동할 수 있습니다. \n5. 자동 맥락 요약: 문맥의 양이 모델의 맥락 크기를 초과할 경우 자동으로 요약할 수 있는 기능이 포함되어 있습니다.

- **Performance Highlights**: Drama Engine은 창의적인 글쓰기 및 다중 에이전트 대화의 다양한 응용 프로그램에서 운영되고 있으며, 모듈화된 방식으로 동적인 프롬프트 시퀀스를 구성하여 기존의 단순 프롬프트 체인보다 훨씬 유연하고 제어 가능한 시스템을 제공합니다.



### Design Principle Transfer in Neural Architecture Search via Large Language Models (https://arxiv.org/abs/2408.11330)
- **What's New**: 본 연구는 전이 가능한 신경 구조 검색(TNAS) 방법의 한계를 극복하기 위해 새로운 전이 패러다임인 설계 원칙 전이(design principle transfer)를 제안합니다. 이 접근법을 통해 고성능 아키텍처에서 추출된 설계 원칙을 활용하여 새로운任务에 대한 검색 공간을 최적화합니다.

- **Technical Details**: LAPT(LLM-assisted Design Principle Transfer) 프레임워크는 사전 훈련된 대형 언어 모델(LLM)을 사용하여 기존 아키텍처에서 설계 원칙을 자동으로 추출하고 이를 바탕으로 새 아키텍처의 검색 공간을 정제합니다. 이 과정은 범주별 원칙 적응(principle adaptation) 방법을 통해 더욱 세분화되며, 특정 작업에 최적화된 검색 공간을 구축합니다.

- **Performance Highlights**: 실험 결과, LAPT는 대부분의 작업에서 최첨단 TNAS 방법을 능가하며, 다른 작업에서는 경쟁력 있는 성능을 보여주어 NAS 분야에서 설계 원칙 전이가 유망한 연구 방향임을 강조합니다.



### EEG-Defender: Defending against Jailbreak through Early Exit Generation of Large Language Models (https://arxiv.org/abs/2408.11308)
Comments:
          19 pages, 7 figures

- **What's New**: 최근 연구에서 제안된 EEG-Defender는 대형 언어 모델(LLMs)에 대한 새로운 방어 전략으로, 초기 트랜스포머 출력 결과를 활용하여 악의적인 입력을 감지하고 즉시 생성을 종료하는 방법을 제시합니다. 이 기술은 기존의 방어 방법들보다 더 높은 Attack Success Rate (ASR) 감소율을 보여줍니다.

- **Technical Details**: EEG-Defender는 초기 및 중간 계층의 출력 임베딩을 비교하여 악의적인 프롬프트와 유사성을 평가합니다. 이 방법은 프롬프트에 의해 유도된 추론을 방지하는 대신, LLM이 생성한 출력의 초기 상태를 분석함으로써 악의적인 요청을 거부합니다. 실험은 Llama2, Vicuna, Guanaco와 같은 세 가지 LLM 모델에서 수행되었습니다.

- **Performance Highlights**: EEG-Defender는 기존 jailbreak 방법들에 비해 ASR을 약 85% 감소시키며, 기능적으로는 benign prompts에 거의 영향을 주지 않습니다. 또한 이 방법은 기존 LLM의 미세 조정 없이 간편하게 통합될 수 있는 특징을 가지고 있습니다.



### RePair: Automated Program Repair with Process-based Feedback (https://arxiv.org/abs/2408.11296)
Comments:
          15 pages, 13 figures

- **What's New**: 이 연구는 Automated Program Repair (APR) 분야에서 소규모 언어 모델(20B 미만의 파라미터)을 활용하여 과정 기반(process-based) 피드백을 적용한 처음 사례로, 코드 수리가 어떻게 보다 효율적으로 이루어질 수 있는지를 탐구합니다.

- **Technical Details**: 이 논문은 코드 수리를 위한 새로운 데이터셋인 CodeNet4Repair를 구축하고, 보상 모델(reward model)과 수리 모델(repair model)로 구성된 RePair 프레임워크를 제안합니다. 이 모델은 프로그램 텍스트를 입력으로 받아 상태에 대한 피드백을 제공하는 가상 도구처럼 작동하며, 각 수리 단계에서 프로그램의 상태에 대한 피드백을 바탕으로 수정 전략을 조정합니다.

- **Performance Highlights**: 과정 기반 수리 방식은 대규모 상용 언어 모델이 사용하는 결과 기반(outcome-based) 생성 방법보다 우수한 성과를 보이며, 거의 상용 대규모 모델과 유사한 성능을 발휘합니다.



### Towards Analyzing and Mitigating Sycophancy in Large Vision-Language Models (https://arxiv.org/abs/2408.11261)
- **What's New**: 본 연구는 대형 시각-언어 모델(LVLMs)의 비판적인 문제인 sycophancy를 정밀하게 분석하고 이를 완화하기 위한 새로운 방법인 Leading Query Contrastive Decoding(LQCD)을 제안함으로써 이 분야의 연구 공백을 메우고자 하였습니다.

- **Technical Details**: LQCD는 모델에 구애받지 않는 방법으로, 시각적 정보와 언어 모델의 출력을 통합하며, sycophancy 토큰의 확률을 억제하여 모델의 과도한 선행 신호 의존성을 보정합니다. 다양한 VL 벤치마크에 대해 엄선된 leading query를 사용하여 성능을 평가하고, Hallucination 문제를 줄이는데 효과적임을 입증합니다.

- **Performance Highlights**: LQCD는 전반적으로 다른 prompt engineering 기법 및 Hallucination 완화 방법보다 우수한 성능을 보이며, 중립적인 질문에 대한 LVLM의 응답도 약간 개선시켜 더 효과적인 일반 목적 디코딩 전략임을 제시합니다.



### Improving Speech Recognition Error Prediction for Modern and Off-the-shelf Speech Recognizers (https://arxiv.org/abs/2408.11258)
- **What's New**: 본 연구에서는 음성 인식 오류를 예측하는 기존 모델을 확장하여, posterior 기반의 신경망 음향 모델의 동작을 모사하는 샘플링 기반 패러다임을 도입하였습니다. 추가적으로, 혼동 행렬(confusion matrix)을 시퀀스-대-시퀀스(sequence-to-sequence) 모델로 대체하여 예측의 맥락 의존성을 강화했습니다.

- **Technical Details**: 연구에서는 두 가지 모델을 비교: 첫째, 기초 혼동 행렬 기반 모델로부터 샘플링하여 출력 분포를 생성합니다. 둘째, 2층의 128 유닛을 가진 Seq2Seq 모델을 활용하여 문맥 정보를 포함한 음향 오류 예측을 수행합니다. 모델 학습시 입력과 출력 음표의 정렬을 위해 혼동 행렬과 유사한 기술을 사용하였습니다.

- **Performance Highlights**: 샘플링 기법은 100가지 추정 기반에서 예측 정확성을 크게 향상시키었고, 시퀀스 모델의 성능은 혼동 행렬과 유사하게 나타났습니다. 연구는 무관한 클라우드 기반 ASR 시스템의 행동을 추정하는 데에도 성공적으로 적용되었습니다.



### A Little Confidence Goes a Long Way (https://arxiv.org/abs/2408.11239)
Comments:
          13 pages, 2 figures

- **What's New**: 이 연구에서는 큰 언어 모델(LLM)의 숨겨진 상태 활성화(probes of hidden state activations)를 활용한 이진 분류(binary classification) 작업을 위한 새로운 방법을 소개합니다. 특히, 이 접근법은 현재 가장 많은 성능을 발휘하는 LLM과 유사한 성능을 제공하면서도 계산 자원을 훨씬 적게 요구하고 라벨링된 데이터가 필요하지 않습니다.

- **Technical Details**: 제안된 기술은 클래스 레이블(class labels)을 의미가 풍부한 설명으로 변환하고, 멀티레이어 퍼셉트론 모형의 자발적 대칭 파괴(spontaneous symmetry breaking), 엔트로피 최대화를 통한 숨겨진 상태 활성화에서의 신뢰도 점수(confidence scores) 생성, 예측을 위한 앙상블에서 가장 신뢰할 수 있는 모형 선택 등을 포함합니다. 이 논문에서는 Glia라는 이름으로 이러한 기술들을 묶어 부르고 있으며, 네 개의 데이터세트에서 평가되었습니다.

- **Performance Highlights**: 이 방법은 GPU에서 효율적으로 구현될 수 있으며, 감독 학습(supervised fine-tuning)이나 데이터 라벨에 접근할 필요 없이 작동합니다. Glia의 방법론은 4개의 데이터세트(Amazon polarity, IMDB, CUAD, Learned Hands)에 대해 다섯 개의 기본 LLM을 사용하여 성능을 평가했습니다. 기존의 접근 방식보다 숫자가 적은 자원으로 우수한 LLM 추론(inference)을 제공합니다.



### Out-of-Distribution Detection with Attention Head Masking for Multimodal Document Classification (https://arxiv.org/abs/2408.11237)
- **What's New**: 이번 연구에서는 Attention Head Masking (AHM)이라는 새로운 방법론을 제안하여 다중 모달 문서 분류 시스템에서 out-of-distribution (OOD) 데이터 탐지를 개선하였습니다. 우리가 제안한 방법은 기존의 다른 방법들보다 유의미하게 낮은 false positive rate (FPR)을 달성했습니다.

- **Technical Details**: AHM은 transformer 모델의 self-attention 메커니즘을 활용하여 ID와 OOD 데이터를 분리하는 기능을 강화하는 방법입니다. 이 기법은 데이터가 각 클래스 간에 얼마나 유사한지를 반영하여 OOD 탐지 성능을 극대화하도록 설계되었습니다. 또한, FinanceDocs라는 고품질의 금융 문서 AI 데이터셋을 새롭게 소개합니다.

- **Performance Highlights**: 제안된 AHM 방법은 기존의 최신 솔루션보다 월등한 성능을 보이며, 특히 Tobacco3482 및 FinanceDocs 데이터셋을 사용한 실험 결과에서 OOD 탐지에서 AUC (AUROC) 메트릭이 우수한 성과를 달성했습니다. 이로써 AHM은 다중 모달 데이터에서도 효과적으로 일반화되어 신뢰성과 안전성을 높이는데 기여할 수 있음을 보여주었습니다.



### DSP-MLIR: A MLIR Dialect for Digital Signal Processing (https://arxiv.org/abs/2408.11205)
- **What's New**: 이 논문은 MLIR(Multi-Level Intermediate Representation) 프레임워크를 활용하여 DSP(Digital Signal Processing) 특정 다이얼렉트를 도입하고, 이에 대한 최적화를 구현하여 DSP 애플리케이션의 성능을 향상시키는 새로운 접근 방식을 제시합니다.

- **Technical Details**: MLIR은 다양한 중간 표현(IR)을 제공하며, DSP 다이얼렉트는 신호 처리 애플리케이션에서 일반적으로 사용되는 다양한 기능의 작업을 포함합니다. 이 논문에서는 Affine 다이얼렉트로의 낮추기(lowering) 절차도 논의하며, 최적화를 위해 DSP-다이얼렉트 전용 작업을 소개합니다. DSP 애플리케이션을 작성하기 위한 DSL(Domain Specific Language)도 제공됩니다.

- **Performance Highlights**: DSP 애플리케이션에 대한 최적화 덕분에 기존 C 수준의 컴파일러에서 달성하기 어려운 최대 10배의 성능 개선을 보였습니다. 이는 DSP 기술과 MLIR의 결합을 통해 이루어진 효율적인 최적화 덕분입니다.



### SubgoalXL: Subgoal-based Expert Learning for Theorem Proving (https://arxiv.org/abs/2408.11172)
- **What's New**: 이 논문은 SubgoalXL이라는 새로운 접근 방식을 소개하며, 이는 서브골 기반의 증명과 전문가 학습(expert learning)을 시너지시켜 LLM의 형식 정리 증명 능력을 향상시킵니다.

- **Technical Details**: SubgoalXL은 전용 수학 및 정리 증명 데이터의 부족 문제와 LLM의 다단계(reasoning abilities) 추론 능력 향상이 필요한 두 가지 주요 도전을 해결합니다. 데이터 효율성을 최적화하고 서브골 수준 감독(subgoal-level supervision)을 적용하여 제한된 인간 생성 증명에서 더 풍부한 정보를 추출합니다.

- **Performance Highlights**: SubgoalXL은 Isabelle 환경에서 miniF2F 데이터셋에 대해 56.1%의 새로운 최고 성과를 달성하였으며, 이는 Zheng et al. (2023)보다 4.9% 향상된 결과입니다. 함께, SubgoalXL은 miniF2F에서 41 AMC12, 9 AIME 및 3 IMO 문제를 성공적으로 해결하였습니다.



### Public Health in Disaster: Emotional Health and Life Incidents Extraction during Hurricane Harvey (https://arxiv.org/abs/2408.11133)
- **What's New**: 이 연구는 기후 변화로 인한 재난에 대한 사람들의 감정과 삶의 사건을 이해하기 위해 소셜 미디어 데이터를 활용하는 새로운 접근 방식을 제시합니다. 약 400,000개의 트윗을 수집하여 BERT 기반 모델로 감정을 예측하고 Latent Dirichlet Allocation (LDA) 기법을 사용하여 주제를 모델링합니다.

- **Technical Details**: 이 연구는 그래프 신경망 (Graph Neural Network, GNN)과 대규모 언어 모델 (Large Language Model, LLM)을 통합하여 데이터에서 의미 있는 패턴을 추출하였습니다. GNN을 사용하여 트윗의 임베딩을 생성하고 유사성 그래프를 구축하여 군집 최적화를 수행했습니다. 이후 GPT-2 기반 LLM을 통해 각 사건 클러스터의 설명적인 이름을 자동으로 생성하였습니다.

- **Performance Highlights**: 연구 결과, GNN과 LLM의 통합을 통해 트윗 임베딩을 개선하고 정확한 군집화를 달성하는 데 기여했습니다. 또한, 전통적인 주제 모델링을 넘어 사람 같은 해석을 제공함으로써, 기후 재난 관리와 공공 건강 전략 수립에 중요한 통찰을 제공합니다.



### DOMBA: Double Model Balancing for Access-Controlled Language Models via Minimum-Bounded Aggregation (https://arxiv.org/abs/2408.11121)
Comments:
          11 pages, 3 figures

- **What's New**: DOMBA(더블 모델 균형화)를 제안하여 접근 제어가 적용된 데이터에서 LLM(대규모 언어 모델)의 훈련 및 배포를 위한 새로운 접근법을 제공합니다. 이 방법은 높은 유용성과 보안 보장을 제공하면서 사용자 권한에 따라 훈련 데이터를 보호합니다.

- **Technical Details**: DOMBA는 'min-bounded'(최소 경계) 평균 함수를 사용하여 서로 다른 접근 수준으로 훈련된 두 개의 서브모델의 확률 분포를 집계합니다. 이를 통해 서브모델 각각이 보유한 민감한 정보가 텍스트 생성 시 잘 노출되지 않도록 합니다. 이론적인 수학적 분석을 통해 민감한 정보 노출을 효과적으로 차단하는 방법을 제시합니다.

- **Performance Highlights**: DOMBA는 민감한 정보 노출을 제한하면서도 비보안 모델과 유사한 유용성을 제공합니다. 새롭게 제안된 세 가지 보안 메트릭으로 DOMBA의 효과를 평가하며, 기존 접근 제어 방법에 비해 높은 성능을 유지합니다.



### Mistral-SPLADE: LLMs for for better Learned Sparse Retrieva (https://arxiv.org/abs/2408.11119)
- **What's New**: 이번 연구에서는 learned sparse retrievers (LSR)에서 decoder-only 모델을 이용하여 semantic keyword expansion을 학습하는 새로운 방법을 제안했습니다. 기존의 LSR 시스템과 비교하여 성능이 크게 향상된 것을 보여주며, 특히 Echo-Mistral-SPLADE 모델이 최신 BEIR 텍스트 검색 벤치마크에서 최첨단 성과를 기록했습니다.

- **Technical Details**: 다양한 데이터셋에서 효과적인 학습을 위해 Mistral을 backbone으로 사용하여 LSR 모델을 개발하였으며, transformer 기반의 텍스트 임베딩 모델에 사용되는 sentence-transformer 데이터의 하위 집합에서 학습했습니다. 연구에서는 decoder-only large language model (LLM)이 keyword expansion 학습에 더 효과적임을 입증하였습니다.

- **Performance Highlights**: 실험 결과, 우리의 LLM 기반 모델인 Echo-Mistral-SPLADE는 기존의 LSR 시스템(SPLADE 및 그 변형 포함)보다 우수한 성능을 나타냈으며, 이는 sparse retrieval 분야에 중요한 발전을 의미합니다.



### What can Large Language Models Capture about Code Functional Equivalence? (https://arxiv.org/abs/2408.11081)
Comments:
          37 pages

- **What's New**: 이 논문은 SeqCoBench라는 새로운 벤치마크를 소개하여 Code-LLMs가 코드의 기능적 동등성을 얼마나 잘 포착할 수 있는지를 체계적으로 평가합니다.

- **Technical Details**: SeqCoBench는 Python 프로그램의 의미를 보존하거나 변경하는 20개 이상의 코드 변환을 포함하고 있습니다. 다양한 설정에서, 제로샷(zero-shot) 및 파라미터 효율적인 파인튜닝(parameter-efficient finetuning) 방법을 사용하여 최첨단 (Code-)LLMs의 성능을 평가했습니다.

- **Performance Highlights**: LLMs와 전통적인 일치 기반 검색 점수 간의 성능 차이는 미미하였으며, 두 방법 모두 코드 의미 이해에 깊이가 부족하다는 우려스러운 결과를 보였습니다.



### Statistical Patterns in the Equations of Physics and the Emergence of a Meta-Law of Natur (https://arxiv.org/abs/2408.11065)
Comments:
          9 pages, 5 figures

- **What's New**: 이번 연구에서는 물리학 방정식의 통계적 정규성을 탐구하여 оператор(operator) 분포의 패턴을 기초로 한 새로운 발견을 제시합니다. 이 연구는 Zipf의 법칙과 유사한 구조적 규칙을 물리식을 통해 드러내며, 이로 인해 자연의 보편 법칙에 대한 통찰을 제공합니다.

- **Technical Details**: 물리학 공식 세 개의 코퍼스를 분석하였으며, 각 코퍼스에서 оператор의 빈도와 순위 관계를 연구하였습니다. 고급 암묵적 우도 방법을 활용하여, 다양한 물리 방정식에서 оператор 빈도가 비례 지수 법칙(exponential law) 형태로 나타남을 발견했습니다. 이 결과는 Zipf의 역 제곱 법칙(inverse power-law)과 상반되는 결과입니다.

- **Performance Highlights**: 연구 결과, 물리학 공식에서 나타나는 통계적 패턴의 이해는 기호 회귀(symbolic regression) 기법의 개선 가능성을 여는 중요한 발견입니다. 이는 언어 모델의 기초를 더욱 강화할 수 있으며, 물리적 현상을 위한 기호 모델을 생성하는 데 기여할 수 있습니다.



### LLM Agents Improve Semantic Code Search (https://arxiv.org/abs/2408.11058)
Comments:
          12 pages, 1 Figure

- **What's New**: 이 논문에서는 코드 검색의 정확성을 개선하기 위해 RAG (Retrieval Augmented Generation)를 활용한 에이전트 기반 접근 방식을 도입했습니다. 이를 통해 사용자 쿼리에 정보가 추가되어 더 나은 코드 검색이 가능해집니다.

- **Technical Details**: 이 연구는 에이전트가 GitHub 저장소에서 수집한 관련 정보를 사용하여 사용자 쿼리를 강화하도록 설계되었습니다. 첫 번째 단계로, 에이전트는 사용자의 자연어 쿼리에 필요한 기술 정보를 추가하고, RAG 기술을 사용하여 적절한 컨텍스트 정보를 검색합니다. 이렇게 생성된 입력은 OpenAI의 최신 텍스트 임베딩을 사용하여 코드로 변환되어, 코드 검색의 정확성을 높입니다.

- **Performance Highlights**: 실험 결과 RepoRift는 기존 방법들보다 월등히 향상된 성능을 보이며, CodeSearchNet 데이터셋에서 Success@10에서 78.2%의 성공률, Success@1에서 34.6%의 성공률을 기록했습니다. 이 연구는 에이전트 기반 LLM과 RAG의 잠재력을 강조하며, 더욱 효과적인 코드 검색 시스템을 위한 중요한 발전을 보여줍니다.



### Architectural Foundations for the Large Language Model Infrastructures (https://arxiv.org/abs/2408.09205)
- **What's New**: 대형 언어 모델(LLM) 인프라 개발의 중요성이 강조되며, 이 논문은 LLM의 인프라, 소프트웨어 및 데이터 관리의 복잡한 경관을 탐구합니다. 성공적인 LLM 개발을 위한 고려사항 및 안전장치에 대한 통찰력을 제공합니다.

- **Technical Details**: LLM 교육을 위한 인프라 구성에서 H100/H800 GPU를 장착한 서버 클러스터가 주류로 자리 잡았고, LoRA(Low-Rank Adaptation)와 같은 경량화 방법이 컴퓨팅 파워 요구사항을 줄이는 데 기여합니다. 또한, 알고리즘 최적화 및 하이퍼파라미터 설정이 모델 성능 향상에 중요합니다.

- **Performance Highlights**: 8개 노드로 구성된 클러스터는 7B 파라미터 모델의 교육을 하루 만에 완료할 수 있으며, GPU 및 CPU 자원의 유연한 활용이 LLM 추론 배치에서 필수적입니다. 데이터 관리의 효율성을 통해 높은 품질의 데이터 세트를 확보하고 모델 학습을 극대화할 수 있습니다.



### LADDER: Language Driven Slice Discovery and Error Rectification (https://arxiv.org/abs/2408.07832)
- **What's New**: 이 논문은 기존의 오류 슬라이스 발견 기법과 달리 Large Language Model (LLM)의 추론 기능을 활용해 복잡한 오류 패턴을 분석하고 검증 가능한 가설을 생성합니다.

- **Technical Details**: 제안하는 방법인 LADDER (Language Driven slice Discovery and Error Rectification)는 모델의 표현을 언어 정렬 특징 공간 (language-aligned feature space)으로 투영하여 원래 모델 특징 공간에서 의미가 보존되도록 합니다. 이를 통해 모델의 오류를 강조하는 문장을 정확히 추출하고, LLM을 활용하여 오류 슬라이스를 발견하기 위한 가설을 생성합니다.

- **Performance Highlights**: 논문은 다섯 개의 이미지 분류 데이터셋에서 방법을 검증하였으며, 이 방법은 속성 주석 (attribute annotation)이 필요하지 않습니다.



New uploads on arXiv(cs.IR)

### Do We Really Need to Drop Items with Missing Modalities in Multimodal Recommendation? (https://arxiv.org/abs/2408.11767)
Comments:
          Accepted at CIKM 2024 in the short paper track

- **What's New**: 이 논문은 다중 모달 추천 시스템에서 결여된 모달리티를 단순히 제외하는 기존의 관행에 도전합니다. 연구자들은 모달리티가 결여된 아이템을 처리하는 새로운 파이프라인을 제안하여, 이 과정에서 전통적인 데이터 보간(즉, imputation) 기법을 활용합니다.

- **Technical Details**: 제안된 방법은 다중 모달 추천의 결여된 특성을 복구하기 위한 파이프라인을 구축합니다. 여기에는 전통적인 기계 학습 보간 방법과 함께, 사용자-아이템 그래프 구조를 활용한 세 가지 새로운 그래프 인식 보간 방법이 포함됩니다. 이 방법들은 사용자-아이템 간의 상호작용 및 유사성을 기반으로 합니다.

- **Performance Highlights**: 엄청난 실험 결과를 통해, 결여된 모달리티를 지닌 아이템을 단순히 삭제하는 것보다는 보간하는 것이 다중 모달 추천 시스템의 성능에 긍정적인 영향을 미친다는 것을 입증하였습니다. 이 연구는 다중 모달 추천에서 결여된 데이터를 제거하는 것이 뿐만 아니라 해로운 결과를 초래할 수 있음을 강조합니다.



### A Novel Evaluation Perspective on GNNs-based Recommender Systems through the Topology of the User-Item Graph (https://arxiv.org/abs/2408.11762)
Comments:
          Accepted at RecSys 2024 in the reproducibility track. arXiv admin note: substantial text overlap with arXiv:2308.10778

- **What's New**: 최근 그래프 신경망 기반 추천 시스템이 추천 분야에서 큰 성공을 거두고 있음에도 불구하고, 그 성능의 이론적 및 경험적 원인에 대한 의문을 제기하는 연구가 시작되었습니다. 본 논문에서는 GNN이 추천 데이터를 위상 그래프 구조로 취급한다는 가정에 기반하여 추천 성능에 미치는 그래프 위상의 영향을 조사하는 새로운 평가 관점을 제공합니다.

- **Technical Details**: 추천 데이터의 일부 (topological) 속성과 세 가지 GNN 기반 추천 시스템 (LightGCN, DGCF, SVD-GCN)을 선택합니다. 인기 있는 추천 데이터셋 (Yelp2018, Gowalla, Amazon-Book)에서 샘플링하여 1,800개의 크기가 축소된 데이터셋을 생성하고, 이는 원본과 유사하면서도 더 넓은 범위의 위상 구조를 포함할 수 있습니다. 이 과정을 통해 선택된 GNN 모델들의 데이터 특성과 추천 성능을 측정하기 위한 대규모 샘플 풀을 구축합니다.

- **Performance Highlights**: 그래프 위상과 GNN의 성능 간에 강한 상관관계를 찾아내어 GNN 모델에 대한 새로운 평가 관점을 제공합니다.



### Mathematical Information Retrieval: Search and Question Answering (https://arxiv.org/abs/2408.11646)
Comments:
          [DRAFT] 1st draft

- **What's New**: 본 논문은 수학 관련 질문에 대한 정보를 찾고 활용하는 다양한 시스템과 프레임워크를 소개합니다. 특히, 멀티모달 검색 엔진과 수학 질문 응답 시스템을 통해 수학 정보를 더 쉽게 접근할 수 있도록 하는 전략을 제시합니다.

- **Technical Details**: 저자들은 정보 검색의 기본 작업들을 설명하며, 정보 소스의 종류와 이들이 수학 문제를 해결하기 위해 어떻게 상호작용하는지에 대한 프레임워크를 개발했습니다. 이 프레임워크는 정보의 필요성, 이용 가능한 정보 소스, 정보 작업 수행을 포함합니다.

- **Performance Highlights**: 수학에 대한 정보 검색은 웹 검색, 원서 독서, 메모 작성, 강사와의 대화 등 다양한 활동을 포함해야 하므로 복잡합니다. 따라서, 검색 및 기타 정보 작업의 상호작용을 이해하는 것이 중요합니다.



### End-to-End Cost-Effective Incentive Recommendation under Budget Constraint with Uplift Modeling (https://arxiv.org/abs/2408.11623)
Comments:
          Accepted by RecSys 2024

- **What's New**: 이 논문에서는 예산 제약 하에 사용자에게 최적의 인센티브를 추천하기 위해 새로운 E3IR 모델(End-to-End Cost-Effective Incentive Recommendation)을 제안합니다. 이는 기존의 두 단계 접근 방식의 한계를 극복하는 데 중점을 두고 있습니다.

- **Technical Details**: E3IR 모델은 두 가지 모듈, 즉 uplift prediction 모듈과 differentiable allocation 모듈로 구성됩니다. uplift prediction 모듈은 마케팅 도메인 지식에 따라 인접 치료 간의 점진적 향상을 포착하는 예측 헤드를 구축합니다. allocation 모듈에서는 Integer Linear Programming (ILP)을 미분 가능한 레이어로 활용합니다.

- **Performance Highlights**: E3IR 모델은 다양한 공개 데이터셋과 실제 제품 데이터셋에서 폭넓은 실험을 수행하였으며, 기존 두 단계 접근 방식에 비해 할당 성능을 향상시킨 것으로 나타났습니다.



### DTN: Deep Multiple Task-specific Feature Interactions Network for Multi-Task Recommendation (https://arxiv.org/abs/2408.11611)
- **What's New**: 이번 논문은 다중 과제 학습(Multi-task Learning, MTL) 기술을 적절히 활용하였으며, 기존의 MTL 모델들이 고려하지 않았던 Feature Interaction을 포함하는 Deep Multiple Task-specific Feature Interactions Network (DTN)라는 새로운 모델을 제안합니다.

- **Technical Details**: DTN은 다중의 다양한 Task-specific Feature Interaction 기법과 Task-sensitive 네트워크를 도입하여 MTL에서의 Feature Interaction을 최적화하여, 각 과제에서의 최적의 특성 상호작용 표현을 학습할 수 있게 합니다. 이는 실제 E-commerce 추천 시스템에 63억 개 이상의 샘플을 포함한 데이터셋에 적용되었습니다.

- **Performance Highlights**: DTN은 기존의 최첨단 MTL 모델과 비교하여 클릭 수 3.28%, 주문 수 3.10%, GMV(총 상품 가치) 2.70% 증가라는 효과를 보였으며, 다양한 추천 시스템 및 공공 벤치마크 데이터셋에서도 우수한 성능을 입증했습니다.



### Calibrating the Predictions for Top-N Recommendations (https://arxiv.org/abs/2408.11596)
Comments:
          accepted at RecSys 2024

- **What's New**: 이번 논문에서는 추천 시스템에서 top-N 아이템의 잘못된 예측(calibration) 문제를 다룹니다. 기존의 calibration 방법들이 전체 아이템에 대해서는 잘 작동하지만, top-N 아이템에 대해서는 부정확한 결과를 초래할 수 있음을 보여줍니다. 이를 해결하기 위해, top-N 아이템에 특화된 새로운 최적화 방법을 제안합니다.

- **Technical Details**: 상위 N개 추천 아이템의 예측 값을 잘 조정하기 위해, 본 연구에서는 평가 지표를 정의하고 그룹화(rank group)하여 각각의 그룹에 대해 독립적인 calibration 모델을 최적화합니다. 이 과정에서 랭크에 따라 가중치를 다르게 적용하여 상위 랭크에 더욱 집중하는 방법론을 사용합니다. 이 방법은 다양한 추천 모델에 적용 가능하며, 명시적(Explicit) 및 암묵적(Implicit) 피드백 데이터셋에서 그 효과가 입증되었습니다.

- **Performance Highlights**: 제안된 방법은 기존의 calibration 기법과 비교하여 top-N 추천 아이템의 예측 정확도를 향상시킵니다. 다양한 추천 모델과 캘리브레이션 모델에 대한 수행 결과를 통해, top-N 아이템의 잘못된 예측(calibration)을 모두 해결하는 데 기여할 수 있음을 확인하였습니다.



### Oh, Behave! Country Representation Dynamics Created by Feedback Loops in Music Recommender Systems (https://arxiv.org/abs/2408.11565)
Comments:
          RecSys 2024

- **What's New**: 최근 연구들은 음악 추천 시스템이 훈련 데이터에서 더 두드러진 국가, 즉 미국의 음악을 불균형하게 추천하는 경향이 있다는 것을 보여줍니다. 그러나 이러한 추천 방식에서 피드백 루프가 밸런스에 미치는 영향을 명확히 개관하지는 않았습니다.

- **Technical Details**: LFM-2b 데이터셋을 사용해 피드백 루프 시뮬레이션 연구를 실시하였으며, 다양한 추천 모델들이 로컬 아티스트의 음악 비율을 줄이는 경향이 있음을 발견했습니다. 또한, 미국 음악과 로컬 음악의 평균 비율을 유지하는 모델이 국가별 조정된 추천을 제공하지 않는 경향이 있음을 보고했습니다.

- **Performance Highlights**: 가장 인기 있는 조정 모델(ItemKNN)이 가장 적은 국가 조정된 추천을 제공했으며, 덜 대표되는 국가 사용자들은 추천 시스템에서 그들의 로컬 음악이 과소 대표됨으로 인해 가장 큰 영향을 받는 것으로 나타났습니다.



### A Quick, trustworthy spectral detection Q&A system based on the SDAAP Dataset and large language mod (https://arxiv.org/abs/2408.11557)
Comments:
          16 pages,10 figures,3 tables

- **What's New**: 본 논문에서는 스펙트럼 분석 및 감지 분야에서 최초로 오픈소스 텍스트 지식 데이터셋인 Spectral Detection and Analysis Based Paper(SDAAP)를 소개합니다. SDAAP는 주석이 달린 문헌 데이터와 관련된 지식 설명 데이터로 구성되어 있습니다.

- **Technical Details**: SDAAP 데이터셋은 2014년부터 2023년까지의 관련 논문 정보를 포함하며, 각 항목은 연구 객체, 사용된 스펙트로스코픽 기법 및 관련 화학 계량 매개변수로 세분화됩니다. 또한, SDAAP 기반의 자동 Q&A 프레임워크를 설계하였으며, 이 프레임워크는 질문 형식을 구문 분석하고 해당 스펙트럼 감지 지식을 검색하여 고품질 응답을 생성합니다.

- **Performance Highlights**: 실험 결과에 따르면, 우리의 프레임워크는 기준선 모델에 비해 더 신뢰할 수 있는 전문성을 가진 응답을 생성하는 것으로 나타났습니다. 또한, 이 접근 방식은 생성된 응답의 품질을 향상시킬 뿐만 아니라 지식의 추적 가능성을 보장합니다.



### LARR: Large Language Model Aided Real-time Scene Recommendation with Semantic Understanding (https://arxiv.org/abs/2408.11523)
- **What's New**: 이번 논문에서는 대규모 언어 모델(LLM)을 활용하여 실시간 장면 추천 시스템의 성능을 개선하는 새로운 프레임워크인 LARR(Large Language Model Aided Real-time Scene Recommendation)를 제안합니다. LARR는 추천 도메인 지식을 LLM에 주입하고, LLM의 출력을 집계하는 인코더를 통해 실제 장면 정보를 구성하여 CTR(Click-Through Rate) 모델링의 효율성을 향상시킵니다.

- **Technical Details**: LARR 모델은 세 가지 단계로 진행됩니다. 첫째, 추천 데이터로부터 구축된 코퍼스를 기반으로 LLM을 지속적으로 사전 훈련합니다. 둘째, 샘플 구축 전략을 사용하여 대조 학습을 통해 LLM을 미세 조정합니다. 마지막으로, LLM의 다양한 장면 특성에 대한 출력을 인코더를 통해 집계하여 협력 신호와 일치하도록 합니다.

- **Performance Highlights**: Meituan Waimai의 음식 배달 데이터셋을 활용한 오프라인 및 온라인 실험을 통해 LARR가 실시간 장면의 의미 정보를 완전히 이해하고 다중 모달 특성을 효과적으로 통합하여 추천 결과를 향상시켰음을 확인했습니다.



### Denoising Pre-Training and Customized Prompt Learning for Efficient Multi-Behavior Sequential Recommendation (https://arxiv.org/abs/2408.11372)
- **What's New**: DPCPL(다단계 행위 연속 추천을 위한 새로운 Pre-training 및 Prompt-tuning 패러다임)이 소개되었습니다. 이 방법은 효율적인 사전 학습 및 사용자 맞춤형 프롬프트 학습을 결합하여 다양한 사용자 행동 데이터를 처리합니다.

- **Technical Details**: DPCPL의 핵심 구성 요소는 Efficient Behavior Miner (EBM)와 Customized Prompt Learning (CPL) 모듈입니다. EBM은 여러 시간 척도에서 노이즈를 필터링하고, CPL은 사용자 기반 정보를 활용해 개인화된 프롬프트를 생성하여 모델의 잠재력을 극대화합니다. 또한, Fast Fourier Transform(FFT) 및 주파수 인식 융합과 같은 기법을 사용해 노이즈를 줄입니다.

- **Performance Highlights**: 세 가지 실제 데이터셋에서 DPCPL의 실험 결과, 높은 효율성과 효과성이 입증되었습니다. 최소한의 파라미터 조정만으로도 다양한 다운스트림 작업에서 최첨단 성능을 초과했습니다.



### Deep Tree-based Retrieval for Efficient Recommendation: Theory and Method (https://arxiv.org/abs/2408.11345)
- **What's New**: 딥 추천 모델의 효율성을 높이기 위한 새로운 방식인 Tree-based Deep Retrieval (TDR)을 제안합니다. TDR은 훈련 과정에서 생성된 모든 트리를 유지하여 forest를 형성하고, max-heap 가정을 최대한 충족시키며, 다중 분류 문제로 트리 노드의 표현을 학습합니다.

- **Technical Details**: TDR은 각 레벨의 트리 노드를 다중 분류 문제로 학습하여 수평 경쟁을 가능하게 하고, softmax 손실의 계산 효율성을 높이기 위해 sampled-softmax 기법을 활용합니다. 이 모델은 이진 분류 문제의 한계를 극복하기 위해 설계되었습니다.

- **Performance Highlights**: 실제 데이터셋에서 제안된 TDR의 효과성을 검증하며, 기존 모델들에 비해 우수한 추천 성능을 보입니다. 특히, TDR은 트리 학습 방법과 샘플링 방법의 유효성을 입증하며, 추천의 정확성과 효율성을 동시에 개선합니다.



### Parallel Algorithms for Median Consensus Clustering in Complex Networks (https://arxiv.org/abs/2408.11331)
Comments:
          12 pages

- **What's New**: 본 연구에서는 그래프의 다양한 클러스터링 솔루션의 합의를 찾는 새로운 알고리즘을 개발하였습니다. 이 알고리즘은 median set partitioning 문제로 포맷을 설정하고, 탐욕적 최적화 기법(greedy optimization technique)을 제안합니다. 특히, 그래프 구조를 고려하여 기존 방법보다 빠르게 동등한 품질의 솔루션을 제공합니다.

- **Technical Details**: 알고리즘은 다양한 클러스터의 partitions을 입력으로 받아 Mirkin distance를 최소화하는 방식으로 최적의 합의 partition을 찾습니다. 기존의 그래프-비의존적 솔루션과는 다르게, 이 알고리즘은 메모리 요구사항을 줄이고, 그래프 구조를 이용하여 반복적으로 합의를 개선합니다. 병렬 알고리즘을 설계하여 64개의 처리 코어를 사용할 경우 35배의 속도 향상을 달성합니다.

- **Performance Highlights**: 개발된 병렬 알고리즘은 대규모 실제 데이터인 단일 세포 실험 데이터에 대해 빠른 처리 속도를 보여줍니다. 실제 커뮤니티가 알려진 그래프의 경우, 우리의 합의 파티션은 다른 방법보다 실제 커뮤니티 구조를 더 정확하게 포착합니다.



### Public Health in Disaster: Emotional Health and Life Incidents Extraction during Hurricane Harvey (https://arxiv.org/abs/2408.11133)
- **What's New**: 이 연구는 기후 변화로 인한 재난에 대한 사람들의 감정과 삶의 사건을 이해하기 위해 소셜 미디어 데이터를 활용하는 새로운 접근 방식을 제시합니다. 약 400,000개의 트윗을 수집하여 BERT 기반 모델로 감정을 예측하고 Latent Dirichlet Allocation (LDA) 기법을 사용하여 주제를 모델링합니다.

- **Technical Details**: 이 연구는 그래프 신경망 (Graph Neural Network, GNN)과 대규모 언어 모델 (Large Language Model, LLM)을 통합하여 데이터에서 의미 있는 패턴을 추출하였습니다. GNN을 사용하여 트윗의 임베딩을 생성하고 유사성 그래프를 구축하여 군집 최적화를 수행했습니다. 이후 GPT-2 기반 LLM을 통해 각 사건 클러스터의 설명적인 이름을 자동으로 생성하였습니다.

- **Performance Highlights**: 연구 결과, GNN과 LLM의 통합을 통해 트윗 임베딩을 개선하고 정확한 군집화를 달성하는 데 기여했습니다. 또한, 전통적인 주제 모델링을 넘어 사람 같은 해석을 제공함으로써, 기후 재난 관리와 공공 건강 전략 수립에 중요한 통찰을 제공합니다.



### Mistral-SPLADE: LLMs for for better Learned Sparse Retrieva (https://arxiv.org/abs/2408.11119)
- **What's New**: 이번 연구에서는 learned sparse retrievers (LSR)에서 decoder-only 모델을 이용하여 semantic keyword expansion을 학습하는 새로운 방법을 제안했습니다. 기존의 LSR 시스템과 비교하여 성능이 크게 향상된 것을 보여주며, 특히 Echo-Mistral-SPLADE 모델이 최신 BEIR 텍스트 검색 벤치마크에서 최첨단 성과를 기록했습니다.

- **Technical Details**: 다양한 데이터셋에서 효과적인 학습을 위해 Mistral을 backbone으로 사용하여 LSR 모델을 개발하였으며, transformer 기반의 텍스트 임베딩 모델에 사용되는 sentence-transformer 데이터의 하위 집합에서 학습했습니다. 연구에서는 decoder-only large language model (LLM)이 keyword expansion 학습에 더 효과적임을 입증하였습니다.

- **Performance Highlights**: 실험 결과, 우리의 LLM 기반 모델인 Echo-Mistral-SPLADE는 기존의 LSR 시스템(SPLADE 및 그 변형 포함)보다 우수한 성능을 나타냈으며, 이는 sparse retrieval 분야에 중요한 발전을 의미합니다.



### Reading with Inten (https://arxiv.org/abs/2408.11189)
- **What's New**: 이 논문은 Retrieval Augmented Generation (RAG) 시스템이 외부 정보 소스와 통합되는 방법을 다룹니다. 특히, 인간 커뮤니케이션의 뉘앙스를 이해하는 데 어려움이 있는 RAG 시스템에서의 풍자(sarcasm) 처리 방법을 연구합니다.

- **Technical Details**: 저자들은 Natural Questions의 Wikipedia 검색 코퍼스를 기반으로 합성된 풍자가 포함된 텍스트를 생성하고, 이러한 텍스트가 RAG 파이프라인의 retriever와 reader 부분의 성능에 미치는 영향을 실험합니다. 그들은 풍자를 처리하기 위한 프롬프트 시스템을 개발하여 모델이 풍자가 있는 상황에서 응답을 해석하고 생성하는 능력을 향상시킵니다.

- **Performance Highlights**: 종합적인 ablation studies를 통해 제안한 방법의 효과성을 입증하며, 풍자가 포함된 콘텐츠를 처리하는 데 있어 성능 향상을 나타냅니다.



### LLM Agents Improve Semantic Code Search (https://arxiv.org/abs/2408.11058)
Comments:
          12 pages, 1 Figure

- **What's New**: 이 논문에서는 코드 검색의 정확성을 개선하기 위해 RAG (Retrieval Augmented Generation)를 활용한 에이전트 기반 접근 방식을 도입했습니다. 이를 통해 사용자 쿼리에 정보가 추가되어 더 나은 코드 검색이 가능해집니다.

- **Technical Details**: 이 연구는 에이전트가 GitHub 저장소에서 수집한 관련 정보를 사용하여 사용자 쿼리를 강화하도록 설계되었습니다. 첫 번째 단계로, 에이전트는 사용자의 자연어 쿼리에 필요한 기술 정보를 추가하고, RAG 기술을 사용하여 적절한 컨텍스트 정보를 검색합니다. 이렇게 생성된 입력은 OpenAI의 최신 텍스트 임베딩을 사용하여 코드로 변환되어, 코드 검색의 정확성을 높입니다.

- **Performance Highlights**: 실험 결과 RepoRift는 기존 방법들보다 월등히 향상된 성능을 보이며, CodeSearchNet 데이터셋에서 Success@10에서 78.2%의 성공률, Success@1에서 34.6%의 성공률을 기록했습니다. 이 연구는 에이전트 기반 LLM과 RAG의 잠재력을 강조하며, 더욱 효과적인 코드 검색 시스템을 위한 중요한 발전을 보여줍니다.



New uploads on arXiv(cs.CV)

### GRAB: A Challenging GRaph Analysis Benchmark for Large Multimodal Models (https://arxiv.org/abs/2408.11817)
- **What's New**: 본 논문에서는 LMMs(대규모 멀티모달 모델)의 성능을 평가하기 위한 새로운 벤치마크인 GRAB(그래프 분석 벤치마크)를 소개합니다. GRAB은 2170개의 질문으로 구성되어 있으며, 모든 질문은 합성적으로 생성되어 높은 품질의 노이즈 없는 질문을 보장합니다. 이 벤치마크는 모델이 그래프의 기능과 데이터 시리즈를 해석하는 데 필요한 새로운 차원의 도전을 제공합니다.

- **Technical Details**: GRAB은 4개의 주요 작업과 23개의 그래프 속성을 포함하는 질문 세트로 구성됩니다. 질문들은 특정 그래프 및 데이터의 속성을 추출하고 해석하는 데 중점을 두고 있으며, 기본적인 속성부터 고급 계산에 이르기까지 다양합니다. 20개의 LMM을 GRAB에서 평가한 결과, 최고의 성능을 보인 모델도 겨우 21.7%의 점수를 기록했습니다.

- **Performance Highlights**: GRAB은 기존 LMM들이 해결하기 어려운 도전적인 질문들로 채워져 있습니다. 대부분의 모델들이 이 벤치마크에서 좋은 성과를 내지 못했으며, 특히 질문 형식과 요구되는 정확도에 따라 성능이 크게 달라지는 것을 확인했습니다. 이는 LMMs의 그래프 해석 능력이 여전히 발전할 여지가 많다는 것을 의미합니다.



### SynPlay: Importing Real-world Diversity for a Synthetic Human Datas (https://arxiv.org/abs/2408.11814)
Comments:
          Project Page: this https URL

- **What's New**: 이 논문에서는 Synthetic Playground(SynPlay)라는 새로운 합성 인간 데이터셋을 소개합니다. 이 데이터셋은 현실 세계에서 인간의 다양한 외모를 실현하기 위해 고안되었으며, 현실적인 인간 동작과 포즈, 그리고 여러 카메라 시점을 통해 다채로운 인간 인스턴스를 포착하는 데 중점을 두고 있습니다.

- **Technical Details**: SynPlay는 게임 엔진을 사용하여 가상 플레이어가 덜 제약되고 자연스러운 동작을 수행할 수 있는 게임 환경을 생성합니다. 이 데이터셋은 73,000장 이상의 이미지와 6.5M의 인간 인스턴스를 포함하고 있으며, 다양한 원근법의 이미지를 수집하기 위해 7개의 가상 카메라를 사용합니다. 또한, 표준 한국 게임의 규칙을 기반으로 한 규칙-유도 모션 디자인 방식을 채택합니다.

- **Performance Highlights**: SynPlay를 사용한 모델 훈련 결과는 기존 합성 데이터셋보다 더 높은 정확도를 보여주며, 데이터가 부족한 상황에서 특히 성능 향상이 두드러집니다. SynPlay 데이터셋은 모델 사전 훈련에 필요한 풍부한 복잡성을 지닌 인간 외모와 포즈를 제공합니다.



### SEA: Supervised Embedding Alignment for Token-Level Visual-Textual Integration in MLLMs (https://arxiv.org/abs/2408.11813)
- **What's New**: 본 논문에서는 Multimodal Large Language Models (MLLMs)의 비주얼 및 언어 컴포넌트 간의 정렬(misalignment) 문제를 해결하기 위한 새로운 접근법, Supervised Embedding Alignment (SEA)를 소개합니다. 이 방법은 비전-언어 사전 훈련된 모델들을 활용하여 시각적 토큰을 LLM의 임베딩 공간과 정렬합니다.

- **Technical Details**: SEA는 토큰 수준(alignment) 정렬 방식으로, 시각적 토큰과 LLM의 의미 표현(semantic representation) 간의 정렬을 수행합니다. 기존의 이미지 수준(supervision) 접근법과는 달리, SEA는 각 시각적 토큰이 LLM 내의 해당 의미 표현과 밀접하게 일치하도록 보장하여, 의미적 레이블을 통한 대조 학습(contrastive learning)을 사용합니다.

- **Performance Highlights**: SEA는 LLaVA-1.5 모델의 성능을 8개의 벤치마크에서 개선하는 데 기여했습니다. 특히, 추가 데이터나 추론 비용을 요구하지 않으며, 소규모 모델에서도 효과를 발휘하여 MLLMs의 전반적인 성능과 해석 가능성을 향상시켰습니다.



### EmbodiedSAM: Online Segment Any 3D Thing in Real Tim (https://arxiv.org/abs/2408.11811)
Comments:
          Project page: this https URL

- **What's New**: 본 논문은 실시간 3D 인스턴스 세그멘테이션(Instance Segmentation) 문제를 해결하기 위해 Segment Anything Model(SAM)을 활용하여 온라인 환경에서 3D 마스크를 정밀하게 예측하는 새로운 방법론인 EmbodiedSAM(ESAM)을 제안합니다. 기존 VFM(Visual Foundation Model)을 이용한 3D 인식 시스템의 한계를 극복하는 접근법입니다.

- **Technical Details**: EmbodiedSAM은 2D 마스크를 3D 쿼리(queries)로 변환하고, iterative query refinement을 통해 공간적인 일관성을 보장합니다. 이 모델은 RGB-D 비디오에서 실시간으로 3D 마스크를 예측하며, 고속의 매트릭스 연산을 통해 다양한 관점에서 예측된 3D 마스크의 유사성을 계산합니다. 또한, geometric-aware query lifting 모듈을 도입하여 SAM에 의해 생성된 2D 마스크의 3D 표현을 최적화합니다.

- **Performance Highlights**: 실험 결과, ESAM은 ScanNet, ScanNet200, SceneNN 및 3RScan 데이터셋에서 기존 오프라인 방법들과 비교해도 뛰어난 정확도와 속도를 보여주며, 다양한 제로샷(zero-shot) 데이터셋에서도 강한 일반화 능력을 나타냈습니다. 단일 RTX 3090 GPU로 훈련 및 평가가 가능하다는 점에서 데이터 효율성 또한 우수합니다.



### Pixel Is Not A Barrier: An Effective Evasion Attack for Pixel-Domain Diffusion Models (https://arxiv.org/abs/2408.11810)
- **What's New**: 본 논문에서는 Pixel-domain Diffusion Models (PDMs)를 대상으로 한 새로운 공격 프레임워크를 제안합니다. 기존의 Latent Diffusion Models (LDMs) 공격 방식과는 달리, PDMs의 취약점을 겨냥하여 효과적인 공격을 구현합니다.

- **Technical Details**: 제안된 공격 프레임워크는 특징 표현 공격 손실 (feature representation attack loss)과 일치성 손실 (fidelity loss)을 포함하여, Denoising UNet의 취약점을 이용해 이미지의 의미를 왜곡하고, 자연스러운 이미지를 생성하기 위한 잠재 최적화 전략을 활용합니다.

- **Performance Highlights**: 실험 결과 우리 방법이 SDEdit 기반의 PDM 편집 방법에 대해 우수한 공격 성능을 나타내며, 기존 방법보다 극복해야 할 방어 메소드에 대해 견고함을 유지함을 보여줍니다.



### Story3D-Agent: Exploring 3D Storytelling Visualization with Large Language Models (https://arxiv.org/abs/2408.11801)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 기존의 2D 시각화 및 단순한 이야기 구성에 제한된 전통적인 시각적 스토리텔링의 한계를 극복하기 위해 Story3D-Agent라는 새로운 접근 방식을 제안합니다.

- **Technical Details**: Story3D-Agent는 LLMs(대형 언어 모델)의 기능을 활용하여 제공된 내러티브를 3D 렌더링된 시각화로 변환합니다. 이 방법은 절차적 모델링(Procedural Modeling)을 통합하여 다중 캐릭터의 동작 및 액션을 정밀하게 제어하며, 다양한 장식 요소를 추가하여 장기적이고 역동적인 3D 표현을 가능하게 합니다.

- **Performance Highlights**: Story3D-Agent의 효과성을 철저히 평가하였으며, 이는 3D 이야기 표현을 발전시키기 위한 기본 프레임워크를 제공합니다.



### EE-MLLM: A Data-Efficient and Compute-Efficient Multimodal Large Language Mod (https://arxiv.org/abs/2408.11795)
- **What's New**: 이번 논문에서는 기존 Self-Attention과 Cross-Attention 방법들의 데이터 효율성과 컴퓨팅 효율성 간의 절충을 해결하기 위해 EE-MLLM이라는 새로운 모델을 제안한다. EE-MLLM은 추가적인 모듈이나 학습 가능한 파라미터 없이 데이터 및 컴퓨팅 효율성을 달성한다.

- **Technical Details**: EE-MLLM은 일반적인 Self-Attention 메커니즘을 변경하여 Composite Attention 메커니즘을 도입한다. 이 메커니즘은 시각 토큰 내에서의 Self-Attention의 계산 오버헤드를 제거하고, LLM의 각 레이어에서 가중치를 재사용하여 비전과 언어 간의 효과적인 모달리티 정렬을 가능하게 한다.

- **Performance Highlights**: EE-MLLM은 MMBench 및 SeedBench와 같은 일반 벤치마크와 TextVQA, DocVQA와 같은 세부 과제에서 탁월한 성능을 나타냈으며, 고해상도 이미지 입력 시에도 70%의 FLLOPs 감소를 보여주었다. 또한, EE-MLLM은 NVIDIA H800 GPU에서 32개의 생성 토큰을 설정할 때 초당 77개의 토큰을 처리할 수 있어 자기 주의 기반 방법들보다 1.9배 더 빠른 추론 속도를 기록하였다.



### Timeline and Boundary Guided Diffusion Network for Video Shadow Detection (https://arxiv.org/abs/2408.11785)
Comments:
          ACM MM2024

- **What's New**: 이번 논문에서는 Timeline and Boundary Guided Diffusion (TBGDiff) 네트워크를 통해 비디오 그림자 감지(Video Shadow Detection, VSD)를 수행합니다. 이는 기존 방식의 비효율적인 시간적 학습 문제를 해결하고 그림자의 경계(boundary)를 고려하여 성능을 개선합니다.

- **Technical Details**: TBGDiff 네트워크는 과거와 미래의 시간적 안내와 경계 정보를 동시에 활용합니다. Dual Scale Aggregation (DSA) 모듈은 긴 시간과 짧은 시간의 프레임 간의 유사성을 재고하여 시간적 이해를 향상시킵니다. 또한, Shadow Boundary Aware Attention (SBAA)를 통해 그림자의 특징을 포착할 수 있는 경계 정보를 이용합니다. Diffusion 모델을 도입하여 Space-Time Encoded Embedding (STEE)을 적용, 시간적 안내를 체계적으로 주입하여 그림자 감지를 수행합니다.

- **Performance Highlights**: TBGDiff는 기존 최첨단 방법들을 능가하는 성능을 보여주며, 제안한 구성 요소의 효과성을 검증합니다. 또한, 코드와 결과를 공개하여 연구자들이 손쉽게 접근할 수 있도록 할 예정입니다.



### Embedding Ordinality to Binary Loss Function for Improving Solar Flare Forecasting (https://arxiv.org/abs/2408.11768)
Comments:
          10 Pages, 8 Figures. This manuscript is accepted to be published at DSAA 2024 conference. arXiv admin note: substantial text overlap with arXiv:2406.11054

- **What's New**: 본 논문은 태양 플레어 예측 문제를 최적화하기 위해 내재적인 서열(ordinal) 플레어 특성을 바이너리 크로스 엔트로피(BCE) 손실 함수에 삽입하는 새로운 손실 함수를 제안합니다. 이러한 수정을 통해 데이터의 서열 특성을 기반으로 모델을 보다 효과적으로 안내하고 전체 모델 성능을 향상시키고자 합니다.

- **Technical Details**: 제안된 손실 함수는 바이너리 플레어 예측을 위한 모델에 서열 특성을 인코딩하여 특정 클래스 내의 미세한 차이를 최적화합니다.ResNet34 기반 모델을 사용하여 M급 플레어를 예측하며, 자기장(가로선) 영상을 입력 데이터로 활용합니다. 모델 성능 평가는 True Skill Score(TSS)와 Heidke Skill Score(HSS)의 기하 평균인 복합 기술 점수(CSS)를 이용해 진행됩니다. 오차 범위 내에서의 성능 향상을 위해 제안된 손실 함수를 최적화한 모델은 표준 BCE 대비 각각 약 7%, 4%, 3% 향상된 CSS를 보여줍니다.

- **Performance Highlights**: 제안된 손실 함수를 사용한 모델은 CSS=0.34(TSS=0.50, HSS=0.23)의 성능을 보여주며, 이를 통해 태양 플레어 예측의 신뢰성을 높이고 기술적 시스템에 미치는 잠재적 영향을 고려할 때 더 효과적인 예측 능력을 제공합니다.



### SBDet: A Symmetry-Breaking Object Detector via Relaxed Rotation-Equivarianc (https://arxiv.org/abs/2408.11760)
- **What's New**: 본 논문에서는 Group Equivariant Convolution (GConv)의 한계를 극복하기 위해 Relaxed Rotation GConv (R2GConv)를 제안하였습니다. 이 방법은 비대칭성을 처리할 수 있는 새로운 방법론입니다.

- **Technical Details**: R2GConv는 비대칭 구조의 물체를 인식하기 위해 Relaxed Rotation-Equivariant 군 $	extbf{R}_4$를 기반으로 하며, 이를 통해 Symmetry-Breaking에 대처할 수 있습니다. 또한, 이 모델을 기반으로 한 Symmetry-Breaking Object Detector (SBDet)가 개발되었습니다.

- **Performance Highlights**: 실험 결과, R2GConv는 자연 이미지 분류 작업에서 높은 성능을 보였으며, SBDet는 객체 탐지 작업에서 뛰어난 일반화 능력과 강인성을 보여주었습니다.



### MambaCSR: Dual-Interleaved Scanning for Compressed Image Super-Resolution With SSMs (https://arxiv.org/abs/2408.11758)
- **What's New**: 본 논문에서는 압축 이미지 초해상도(compressed image super-resolution, CSR) 작업을 위한 Mamba 기반의 새로운 프레임워크인 MambaCSR를 소개합니다. 특히 Mamba의 스캐닝 전략이 복원 과정에서 효과적인 맥락 지식 모델링에 결정적이라는 점이 강조되었습니다.

- **Technical Details**: MambaCSR는 두 가지 스캐닝 전략으로 구성된 효율적인 이중 interleaved 스캐닝 패러다임을 제안합니다: (i) 계층적 interleaved 스캐닝은 샘플을 기반으로 한 로컬 및 순차 스캐닝 방법을 동시에 활용하여 이미지 내의 잠재적 맥락 정보를 포괄적으로 캡처합니다. (ii) 수평-수직 interleaved 스캐닝은 서로 다른 방향의 스캐닝 간 중복성을 줄이기 위해 제안했습니다. 또한 위치 정렬 교차 스케일 스캐닝을 통해 다중 스케일 맥락 정보를 모델링합니다.

- **Performance Highlights**: 여러 벤치마크에서 실시된 실험 결과, MambaCSR는 압축 이미지 초해상도 작업에서 뛰어난 성능과 효율성을 보여주었습니다.



### DH-Bench: Probing Depth and Height Perception of Large Visual-Language Models (https://arxiv.org/abs/2408.11748)
- **What's New**: 본 연구에서는 Vision Language Models (VLMs)의 깊이와 높이 지각 능력을 평가하기 위해 DH-Bench라는 새로운 벤치마크를 제안합니다. 이 벤치마크는 Synthetic 2D, Synthetic 3D 및 Real-World의 세 가지 데이터셋으로 구성됩니다.

- **Technical Details**: DH-Bench는 총 4,040개의 고유한 이미지와 11,300개의 이미지-텍스트 쌍으로 구성됩니다. Synthetic 2D 데이터셋은 2D 도형을 포함하고, Synthetic 3D 데이터셋은 3D 도형을 중심으로 하며, Real-World 데이터셋은 실제 실내 장면의 이미지로 구성됩니다. 이 벤치마크는 VLM의 깊이 및 높이에 대한 인식 능력을 평가하기 위해 설계되었습니다.

- **Performance Highlights**: 17개의 최신 VLM 모델을 벤치마크한 결과, 모델들은 깊이와 높이 인식에서 일관되게 어려움을 겪는 것으로 나타났습니다. 특히, 클로즈드 소스 모델이 합성 데이터와 실제 데이터 간의 성능 차이가 더 크며, 전반적으로 모델들은 깊이에 비해 높이 인식에서 더 큰 어려움을 보였습니다.



### Open-Ended 3D Point Cloud Instance Segmentation (https://arxiv.org/abs/2408.11747)
- **What's New**: 새로운 연구는 기존의 Open-Vocab 3D Instance Segmentation (OV-3DIS) 기법의 한계를 극복하기 위해 Open-Ended 3D Instance Segmentation (OE-3DIS) 문제를 제안하였으며, 사전 정의된 클래스 이름 없이도 객체 분할이 가능하다.

- **Technical Details**: OE-3DIS는 3D 포인트 클라우드와 RGBD 시퀀스를 입력으로 받아 사전 정의된 라벨 없이 클래스 이름과 함께 3D 마스크를 생성한다. 이 방법은 Multimodal Large Language Models (MLLMs)를 활용하여 포인트와 비주얼 토큰을 3D로 복원하는 방식으로 수행된다.

- **Performance Highlights**: OE-3DIS 접근 방식은 ScanNet200과 ScanNet++ 데이터셋에서 기존 최첨단 방법인 Open3DIS보다 뛰어난 성능을 보였으며, 특히 ScanNet++에서는 18.4 AP를 기록하며 Open3DIS의 13.1 AP를 크게 초과하였다.



### CluMo: Cluster-based Modality Fusion Prompt for Continual Learning in Visual Question Answering (https://arxiv.org/abs/2408.11742)
- **What's New**: 이번 논문에서는 대형 비전-언어 모델(VLMs)이 여러 작업을 연속적으로 처리하는 데 있어서의 한계를 극복하기 위해, 새로운 클러스터 기반 모달리티 융합 프롬프트(CluMo) 방법을 제안합니다. 이 방법은 프롬프트 기반의 지속적 학습(Continual Learning, CL) 접근을 통해 일반화 성능을 향상시키고, 이전에 학습한 작업의 지식을 잊는 문제를 최소화하려고 합니다.

- **Technical Details**: 각 비주얼 프롬프트 키와 텍스트 프롬프트 키에 연결된 키-키 프롬프트 쌍을 설계하며, 두 단계의 학습 전략을 채택합니다. 첫 번째 단계에서는 K-평균 클러스터링 알고리즘을 통해 단일 모달 키를 학습하여 최적의 프롬프트 선택을 지원합니다. 두 번째 단계에서는 프롬프트 키가 고정되고 선택된 프롬프트가 입력에 첨부되어 VLM을 훈련합니다.

- **Performance Highlights**: 실험 결과, 제안된 CluMo 방법은 최신 기술(SOTA) 성능을 달성하여 기존의 CL 방법들과 비교해 뛰어난 성과를 나타냈습니다. 이 방법은 VQA(Visual Question Answering) 작업에서 비전과 언어의 입력을 통합하여 지속적 학습 환경에서도 효과적으로 적용될 수 있음을 입증합니다.



### Enhancing Cross-Modal Medical Image Segmentation through Compositionality (https://arxiv.org/abs/2408.11733)
Comments:
          11 pages, 3 figures, 2 tables. Accepted at Deep Generative Models workshop @ MICCAI 2024 (DGM4MICCAI). This is the submitted manuscript with added link to github repo, funding acknowledgements and authors' names and affiliations. No further post submission improvements or corrections were integrated. Final version not published yet

- **What's New**: 새로운 연구는 cross-modal(교차 모달) 의료 이미지 분할 분야에서 compositionality(구성 가능성)를 도입하여 분할 성능을 개선하고 해석 가능성을 제공하는 것입니다. 기존 방법의 복잡성을 줄이기 위해 학습된 표현에 learnable von Mises-Fisher kernels(학습 가능한 폰 미세스-피셔 커널)을 적용합니다.

- **Technical Details**: 제안된 네트워크는 end-to-end(종단 간) cross-modal segmentation framework(교차 모달 분할 프레임워크)이며, 이는 compositionality 개념을 통해 학습된 표현에서 콘텐츠 정보를 분리하고 스타일 정보를 효과적으로 필터링합니다. 이 방법은 이미지의 구조적 라벨이 없는 상황에서 CT(컴퓨터 단층촬영) 이미지를 사용해 MRI(자기공명영상) 이미지를 분할하는 과정을 돕습니다.

- **Performance Highlights**: 실험 결과는 둘 이상의 공개 의료 데이터셋에서 향상된 세분화 성능과 함께 계산 비용이 줄어들었음을 입증합니다. 이 연구는 학습된 compositional content representations(구성 콘텐츠 표현)의 해석 가능성을 보여주었습니다.



### Iterative Object Count Optimization for Text-to-image Diffusion Models (https://arxiv.org/abs/2408.11721)
Comments:
          Pre-print

- **What's New**: 이번 연구는 text-to-image 모델에서 특정 개수의 객체를 정확히 생성하는 것을 목표로 합니다. 기존 모델들은 이미지-텍스트 쌍으로 학습하면서 개수 세기에서 어려움을 겪고 있었고, 이를 해결하기 위한 새로운 방법을 제안합니다.

- **Technical Details**: 연구에서는 counting loss를 최적화하여 생성된 이미지의 정확도를 높이는 방식으로 접근합니다. 이 과정에서 객체 수를 정확히 나타내기 위해 gradient descent를 활용하여 counting token의 embedding을 조정합니다. 이와 함께, object의 뷰포인트에 따라 scaling hyperparameter를 동적으로 조정하는 iterated online training 방식을 채택합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 다양한 객체의 정확한 생성을 보여주었으며, 기존 이미지 생성 방법들과 비교했을 때 품질과 정확도에서 상당한 향상을 보였습니다.



### ControlCol: Controllability in Automatic Speaker Video Colorization (https://arxiv.org/abs/2408.11711)
- **What's New**: 본 논문에서는 사용자가 색상화(process of colorization) 과정을 제어할 수 있는 새로운 자동 스피커 비디오 색상화 시스템인 ControlCol을 소개합니다. 기존의 자동 색상화 시스템들은 사용자의 개입이 제한적이었으나, ControlCol은 높은 색상화 품질을 유지하면서도 사용자의 조정을 가능하게 합니다.

- **Technical Details**: ControlCol은 PSNR, SSIM, FID 및 FVD를 평가 지표로 사용하여 이전의 최고 성능 기술(DeOldify)에 비해 Grid 및 Lombard Grid 데이터셋에서 각각 3.5%의 성능 향상을 달성했습니다. ControlCol은 텍스트가 색상화 과정에 지침을 제공하며, 최종적으로는 사용자의 선택된 예시(exemplar)를 기반으로 굵고 일관된 비디오 색상화를 수행합니다.

- **Performance Highlights**: ControlCol은 사용자 평가에서 DeOldify보다 90% 이상 선호되었으며, Lombard Grid 데이터셋에서는 ControlCol이 제안된 접근 방식에 대해 사용자가 1%만의 선호를 보인 반면, 나머지 99%의 경우에는 ControlCol이 더 좋다고 평가되었습니다.



### FRAP: Faithful and Realistic Text-to-Image Generation with Adaptive Prompt Weighting (https://arxiv.org/abs/2408.11706)
- **What's New**: 본 논문에서는 텍스트-이미지 (Text-to-Image, T2I) 생성에서 프롬프트-이미지 정합성을 보장하기 위해 새로운 접근 방식인 FRAP(Faithful and Realistic Text-to-Image Generation with Adaptive Prompt Weighting)를 제안합니다. 이는 동적으로 각 토큰의 가중치를 조정하여 생성된 이미지의 정합성과 진정성을 향상시키는 방법입니다.

- **Technical Details**: FRAP는 온라인 최적화 알고리즘을 통해 실행 중에 각 토큰의 가중치 계수를 업데이트합니다. 이 측정은 cross-attention 맵을 기반으로 하여 개체의 존재를 강화하고 개체-수식어 쌍의 결합을 촉진하는 통합 목표 함수를 최소화함으로써 이루어집니다. 이는 기존의 잠재 코드 수정 방식과는 달리 잠재 코드가 OOD(out-of-distribution) 문제를 겪지 않도록 합니다.

- **Performance Highlights**: FRAP는 Color-Obj-Scene, COCO-Subject 및 COCO-Attribute 데이터셋에서 기존의 방법들이 나타내는 모든 프롬프트-이미지 정합성 지표에서 유의미하게 더 높은 성능을 보였습니다. 또한, D&B와 비교해 COCO-Subject 데이터셋에서 평균 4.44초 더 빠른 평균 대기 시간을 기록했으며, 생성된 이미지는 CLIP-IQA-Real 측정 기준에서 더 진정성 있는 모습을 보였습니다. FRAP는 LLM 기반 프롬프트 최적화 방법과 결합하여 프롬프트-이미지 정합성을 회복하는 데도 효과적이었습니다.



### Supervised Representation Learning towards Generalizable Assembly State Recognition (https://arxiv.org/abs/2408.11700)
Comments:
          8 pages, 8 figures

- **What's New**: 이 논문에서는 조립 상태 인식을 위한 새로운 접근 방식인 representation learning과 중간 상태 정보 손실 함수 수정 (ISIL)을 제안합니다. 이는 비라벨링된 상태 전이 정보를 활용하여 클러스터링과 분류 성능을 크게 개선합니다.

- **Technical Details**: 조립 상태 인식 문제를 representation learning 작업으로 재구성하였으며, 기존 손실 함수에 ISIL을 도입하여 unlabeled 이미지 간 관계를 활용합니다. 이 모델은 시각적 유사성을 기반으로 조립 상태를 구별할 수 있도록 학습합니다.

- **Performance Highlights**: ISIL을 적용한 클러스터링 성능이 5%에서 22%까지 향상되었으며, 이 방법은 새로운 조립 상태에 대해 높은 일반화 성능을 보여주었습니다. 또한, 이 구조는 실세계 조립 오류를 기존의 분류 기반 접근 방식보다 더 잘 인식하였습니다.



### Robust 3D Gaussian Splatting for Novel View Synthesis in Presence of Distractors (https://arxiv.org/abs/2408.11697)
Comments:
          GCPR 2024, Project Page: this https URL , Video: this https URL

- **What's New**: 본 논문에서는 3D Gaussian Splatting의 성능을 개선하기 위해 동적 객체(distractors)를 효과적으로 처리하는 새로운 방법을 제안합니다. 기존 방식이 정적 장면을 가정함에 따라, 동적 객체의 영향을 받았던 문제를 해결하고자 합니다.

- **Technical Details**: 제안된 방법은 자가 지도(self-supervised) 접근 방식을 통해 이미지 잔차(image residuals)를 분석하여 distractors를 식별하고 배제하는 방식으로 구성됩니다. 또한, 사전 훈련된(segmented) 네트워크를 활용하여 객체 인식을 통해 더 정확한 배제를 목표로 합니다. 이 과정에서 이미지 채널별로 마스크를 컴퓨팅하여 성능을 향상시킵니다.

- **Performance Highlights**: 본 연구는 3D Gaussian Splatting에 비해 PSNR(Peak Signal-to-Noise Ratio)을 1.86dB 개선하고, RobustNeRF 대비 4.3dB 개선된 성능을 보여줍니다. 이를 통해 distractors로 오염된 장면에서도 고품질의 렌더링 결과를 획득할 수 있음을 증명합니다.



### Interpretable Long-term Action Quality Assessmen (https://arxiv.org/abs/2408.11687)
Comments:
          Accepted to British Machine Vision Conference (BMVC) 2024

- **What's New**: 이 논문은 Long-term Action Quality Assessment (AQA) 분야에서의 기존 방법의 한계를 극복하는 새로운 접근법을 제안합니다. 특히, 긴 비디오에서의 액션 평가 및 해석 가능성을 향상시키기 위한 Attention loss 함수와 질의 초기화 방법을 도입하였습니다.

- **Technical Details**: 제안된 방법은 Query-based transformer decoder network을 기반으로 하며, positional query encoding을 통해 클립 수준의 피처를 추출합니다. 또한, self-attention과 cross-attention 간의 상호 가이드를 촉진하기 위한 Attention Loss를 도입합니다. 이를 통해 Temporal Skipping 문제를 해결하고, Weight-Score Regression 헤드를 통해 최종 점수를 갱신합니다.

- **Performance Highlights**: 제안된 방법은 Rhythmic Gymnastics (RG), Figure Skating Video (Fis-V), 그리고 LOng-form GrOup (LOGO) 이라는 세 개의 실세계 긴 AQA 벤치마크에서 최첨단 결과를 달성하였습니다.



### Exploring Robustness of Visual State Space model against Backdoor Attacks (https://arxiv.org/abs/2408.11679)
Comments:
          11 pages, 9 figures, under review

- **What's New**: 본 연구는 Visual State Space Model (VSS)의 Robustness(강건성)를 분석하며 backdoor 공격에 대한 취약성을 조사합니다. 특히, state space model (SSM) 메커니즘이 VSS의 강건성에 미치는 영향을 연구합니다.

- **Technical Details**: VSS 모델이 다양한 backdoor 공격에 어떻게 대응하는지를 평가하기 위해 다수의 데이터셋에 걸쳐 실험을 수행했습니다. VSS 모델은 기존 Gated CNNs보다 저항력이 떨어지지만 ViTs와 유사한 성능을 보입니다. SSM이 포함된 VSS는 Gated CNN 구조보다 backdoor 공격에 더 취약한 특성을 보입니다.

- **Performance Highlights**: VSS 모델은 세 가지 고전 데이터셋과 다양한 backdoor 공격에서 ViTs와 비슷한 강건성을 보이지만, Gated CNNs에 비해서는 덜 강건합니다. 특히, 단일 픽셀 트리거를 사용하여 패치 전처리를 통한 방어 메커니즘이 효과적임을 입증했습니다.



### Video-to-Text Pedestrian Monitoring (VTPM): Leveraging Computer Vision and Large Language Models for Privacy-Preserve Pedestrian Activity Monitoring at Intersections (https://arxiv.org/abs/2408.11649)
- **What's New**: 본 논문에서는 보행자 모니터링을 위해 비디오를 텍스트로 변환하는 Video-to-Text Pedestrian Monitoring (VTPM) 시스템을 소개합니다. 이 시스템은 교차로에서 보행자의 움직임을 모니터링하고 실시간으로 교통 신호 및 날씨 정보를 포함한 텍스트 보고서를 생성합니다.

- **Technical Details**: VTPM은 보행자 감지 및 추적을 위한 컴퓨터 비전 모델을 사용하며, 비디오 프레임 당 0.05초의 지연(latency)을 달성합니다. 또한, 90.2%의 정확도로 교차 위반을 감지하며, Phi-3 mini-4k가 장착되어 있어 보행자 활동에 대한 실시간 텍스트 보고서를 생성합니다. 이러한 보고서는 안전 문제를 명시하며, 0.33초의 지연으로 날씨가 보행자 행동에 미치는 영향도 분석합니다.

- **Performance Highlights**: 제안된 VTPM은 비디오로부터의 메모리 사용을 253백만 퍼센트까지 절감할 수 있으며, 개인 정보를 보호합니다. 또한, 생성된 텍스트 보고서를 통해 교차로에서의 보행자 안전에 대한 신뢰할 수 있는 역사적 분석이 가능하여 패턴 및 안전 위반 사건을 효과적으로 감지할 수 있습니다.



### Toward Enhancing Vehicle Color Recognition in Adverse Conditions: A Dataset and Benchmark (https://arxiv.org/abs/2408.11589)
Comments:
          Accepted for presentation at the Conference on Graphics, Patterns and Images (SIBGRAPI) 2024

- **What's New**: 이번 연구는 더 도전적인 Vehicle Color Recognition (VCR) 시나리오를 나타내는 새로운 데이터셋, UFPR-VCR을 제시합니다. 기존 데이터셋의 단순성을 극복하고 다양한 조명 및 야간 조건을 포함하는 이미지를 제공합니다.

- **Technical Details**: UFPR-VCR 데이터셋은 10,039장의 이미지로, 11가지 차량 색상(베이지, 검정, 파랑, 갈색, 회색, 녹색, 주황, 빨강, 은색, 흰색)으로 분류됩니다. 이 데이터셋은 브라질에서 수집된 자동 차량 번호판 인식(ALPR) 데이터셋에서 가져온 이미지로 구성되어 있습니다.

- **Performance Highlights**: 네 가지 딥러닝 모델을 평가한 결과, UFPR-VCR 데이터셋이 기존 모델에 대해 더 높은 난이도를 제공함을 보여주었으며, 오류 발생 중 상당 부분이 야간 장면에서 발생한 것으로 나타났습니다. 이 연구는 VCR 분야의 향후 연구에 중요한 기초 자료를 제공합니다.



### CHOTA: A Higher Order Accuracy Metric for Cell Tracking (https://arxiv.org/abs/2408.11571)
Comments:
          Accepted at BIC Workshop at European Conference on Computer Vision 2024, 14 pages, 4 figures, 2 tables

- **What's New**: 이번 연구에서는 셀 트래킹(cell tracking) 결과 평가에 있어 CHOTA 메트릭(Cell-specific Higher Order Tracking Accuracy)을 제안합니다. 이 메트릭은 셀 탐지(cell detections), 지역 연관(local associations), 글로벌 일관성(global coherence), 그리고 계통 추적(lineage tracking)을 통합하여 셀 트래킹의 모든 관련 측면을 평가합니다.

- **Technical Details**: CHOTA 메트릭은 새로운 '궤적(trajectory)' 정의를 통해 셀의 전체 계통을 포함시키고, 기존의 HOTA 메트릭에서 이를 적용하여 재정의합니다. 현대적인 셀 트래킹 메트릭들에 대한 상세한 조사도 포함되어 있으며, 이를 통해 CHOTA 메트릭의 이점을 비교합니다.

- **Performance Highlights**: CHOTA는 모든 트래킹 오류에 민감하며, 셀의 전체 계통을 재구성하는 생물학적으로 관련된 능력을 잘 나타내는 지표입니다. 현재 사용되고 있는 셀 트래킹 메트릭들보다 강력하고 포괄적인 대안을 제시합니다.



### Positional Prompt Tuning for Efficient 3D Representation Learning (https://arxiv.org/abs/2408.11567)
Comments:
          tech report

- **What's New**: 본 논문에서는 Positional Prompt Tuning (PPT)이라는 새로운 접근 방식을 제안합니다. 이를 통해 3D 포인트 클라우드의 위치 인코딩이 중요하다는 점을 강조하고, 적은 수의 파라미터로도 효과적인 결과를 얻을 수 있음을 보여줍니다. PPT는 파라미터 효율성을 높이고 원하는 성능을 유지하는 동시에 훈련 시간을 단축합니다.

- **Technical Details**: PPT는 포인트 클라우드의 지역적 및 전역적 특징을 고려하여 멀티 스케일 정보 추출기를 구성하는 위치 인코딩 MLP와 패치 인코더를 결합합니다. 이 방식은 Transformer 아키텍처에 쉽게 통합될 수 있으며, 1.05%의 파라미터만으로 여러 주요 데이터셋에서 최첨단 성과를 달성합니다.

- **Performance Highlights**: PPT는 ScanObjectNN OBJ_BG 데이터셋에서 95.01%의 정확도로 벤치마크를 초과하는 결과를 기록하였으며, ModelNet 1k와 ScanObjectNN 데이터셋에서 각각 1.26%와 4.82%의 정확도 향상을 보여줍니다.



### AutoDirector: Online Auto-scheduling Agents for Multi-sensory Composition (https://arxiv.org/abs/2408.11564)
- **What's New**: 이 논문은 영화 제작의 복잡한 요구 사항을 해결하기 위한 상호작용형 다중 감각 조합 프레임워크인 AutoDirector를 소개합니다. 이는 자동화된 일정을 통해 영화 제작의 효율성을 향상시키고, 사용자 피드백을 반영하여 작업을 개선합니다.

- **Technical Details**: AutoDirector는 스크립트 작성, 촬영, 음악 작곡, 더빙 및 특수 효과 등 다양한 작업을 자동적으로 계획할 수 있는 생성 모델 기반의 시스템입니다. 대규모 언어 모델(LLM)을 활용하여 사용자와의 실시간 상호작용을 통해 작업을 동적으로 조정합니다.

- **Performance Highlights**: 실험 결과에 따르면, AutoDirector는 기존 모델보다 효율성 및 다중 감각 조합의 품질에서 향상된 성능을 보이며, 사용자 피드백을 기반으로 한 작업 수정 지원을 통해 영화 제작의 전반적인 만족도를 높입니다.



### Self-Supervised Iterative Refinement for Anomaly Detection in Industrial Quality Contro (https://arxiv.org/abs/2408.11561)
- **What's New**: 본 연구에서 제안하는 Iterative Refinement Process (IRP)는 고위험 산업 품질 관리에 사용되는 강력한 이상 탐지 방법론으로, 사이클 데이터 정제 전략을 통해 결함 탐지 정확성을 향상시킵니다. IRP는 잘못된 데이터 포인트를 반복적으로 제거하여 모델 성능과 견고성을 개선합니다. 두 가지 벤치마크 데이터셋인 Kolektor SDD2 (KSDD2)와 MVTec AD를 사용해 IRP의 효과를 검증하였으며, 기존의 이상 탐지 모델에 비해 우수한 성능을 보였습니다.

- **Technical Details**: IRP는 먼저 이상 탐지에 대한 확률 모델을 사용하여 데이터 포인트의 이상 가능성을 계산합니다. 그런 다음, 적응형 임계값(T술)을 기준으로 각 데이터 포인트의 이상 점수를 평가하고 이를 초과하는 경우 데이터를 제거하여 새로운 데이터셋을 형성합니다. 이 과정은 데이터셋이 최적의 안정성과 성능에 이를 때까지 반복됩니다.

- **Performance Highlights**: IRP는 높은 노이즈 수준의 환경에서도 전통적인 이상 탐지 모델에 비해 일관되게 높은 성능을 보였습니다. 실험 결과, IRP는 sparse하고 noisy한 데이터 문제를 효과적으로 관리하며, 기존 접근 방식들에 비해 이상 탐지 프로세스를 크게 향상시킬 수 있는 잠재력을 입증했습니다.



### Semi-supervised 3D Semantic Scene Completion with 2D Vision Foundation Model Guidanc (https://arxiv.org/abs/2408.11559)
- **What's New**: 이 논문에서는 2D 이미지를 기반으로 한 3D 의미체 점유 예측의 반 감독식(Semi-Supervised) 프레임워크를 소개합니다. 기존의 방법들이 방대한 라벨링된 데이터에 의존했던 데 비해, 저희 방법은 적은 양의 라벨링된 데이터를 이용하여 높은 성능을 달성합니다.

- **Technical Details**: 제안된 VFG-SSC 프레임워크는 2D 비전 기초 모델을 활용하여 라벨이 없는 이미지로부터 3D 장면 기하학적 및 의미 신호를 생성합니다. 자기 학습(Self-Training) 및 평균 교사 네트워크(Mean Teacher)와 같은 반 감독 학습 기법을 사용하여 훈련 과정을 진행하며, 주의 기반 개선 모듈을 통해 3D 특징과 2D 정보를 효과적으로 결합합니다.

- **Performance Highlights**: 우리는 SemanticKITTI와 NYUv2 데이터셋에서 실험을 수행하여, 라벨링된 데이터의 10%만을 사용하면서도 전체 감독 성능의 85%를 달성하는 뛰어난 성능을 보였습니다.



### GSTran: Joint Geometric and Semantic Coherence for Point Cloud Segmentation (https://arxiv.org/abs/2408.11558)
Comments:
          ICPR 2024

- **What's New**: 이번 논문에서는 포인트 클라우드(3D point cloud) 분할 작업을 위한 새로운 transformer 네트워크인 GSTran을 제안합니다. 이는 주로 로컬 기하학적 transformer와 글로벌 의미적 transformer로 구성되어 있습니다.

- **Technical Details**: GSTran은 포인트 클라우드의 로컬 및 글로벌 정보를 효율적으로 캡쳐하기 위해 두 가지 주요 모듈을 사용합니다. 로컬 기하학적 transformer 모듈은 로컬 지역의 기하학적 차이를 계산하여 비슷한 기하학적 속성을 가진 이웃 포인트의 영향력을 증대시키고, 글로벌 의미적 transformer 모듈은 다중 헤드 투표 전략을 통해 전역의 의미적 유사성을 평가합니다.

- **Performance Highlights**: GSTran은 ShapeNetPart와 S3DIS 벤치마크에서 실험을 진행하였으며, 기존의 알고리즘들에 비해 성능이 우수함을 입증하였습니다. 이 방법은 세밀한 경계 분할을 가능하게 합니다.



### AnyDesign: Versatile Area Fashion Editing via Mask-Free Diffusion (https://arxiv.org/abs/2408.11553)
- **What's New**: 이 논문은 다양한 의류로 구성된 사람 이미지를 편집할 수 있는 혁신적인 패션 이미지 편집 기법을 소개합니다. 기존 방법들이 유연한 프레임워크가 부족했던 반면, 저자들은 AnyDesign이라는 마스크 없는 편집 방법을 제안하여 텍스트 또는 이미지 포맷으로 프롬프트를 입력하면 사용자가 원하는 편집을 가능하게 합니다.

- **Technical Details**: AnyDesign 방법은 Fashion DiT(Fashion Diffusion Transformer)라는 확장된 확산 모델 및 Fashion-Guidance Attention(FGA) 모듈을 통합하여 의류 유형과 CLIP로 인코딩된 의류 특징을 융합합니다. 새로운 데이터셋 SSHQe를 활용하여 복잡한 배경을 가진 다양한 의류를 포함하며, 마스크 기반의 확산 모델을 통한 가설 샘플 생성을 포함하여 마스크 없는 편집을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, 제안된 AnyDesign 방법은 기존의 텍스트 기반 패션 편집 방법들보다 높은 품질의 패션 이미지 편집을 가능하게 하고, 다양한 의류 카테고리를 효과적으로 처리할 수 있는 잠재력을 증명했습니다.



### UNetMamba: Efficient UNet-Like Mamba for Semantic Segmentation of High-Resolution Remote Sensing Images (https://arxiv.org/abs/2408.11545)
- **What's New**: 본 논문에서는 UNetMamba라는 새로운 Mamba 기반의 시맨틱 세그멘테이션 모델을 제안한다. 이 모델은 고해상도 위성 이미지의 복잡한 정보를 효율적으로 디코딩할 수 있는 Mamba Segmentation Decoder (MSD)를 포함하며, 국소 정보를 향상시키기 위한 Local Supervision Module (LSM)을 갖춘다.

- **Technical Details**: UNetMamba 모델은 ResT 백본을 사용하는 인코더와 제안된 MSD를 디코더로 하는 U자 형태의 구조로 설계되었다. MSD는 멀티 스케일의 피처 맵에서 시맨틱 정보를 효율적으로 디코딩할 수 있도록 설계되었으며, LSM은 지역적인 시맨틱 정보를 향상시키기 위한 CNN 기반의 모듈이다. MSD는 VSS 블록을 활용하여 파라미터 수를 줄이면서 복잡한 정보를 디코딩할 수 있다.

- **Performance Highlights**: 실험 결과, UNetMamba는 LoveDA에서 0.87% mIoU 향상과 ISPRS Vaihingen에서 0.36% 향상을 보여주며, 경량화, 낮은 메모리 사용량 및 낮은 계산 비용을 통해 높은 효율성을 달성하였다.



### Evolution of Detection Performance throughout the Online Lifespan of Synthetic Images (https://arxiv.org/abs/2408.11541)
- **What's New**: 이 연구는 온라인에서 유포되는 합성 이미지의 진화 과정과 이와 관련된 탐지기 성능을 분석합니다. 특히 현재의 최첨단 탐지기들이 온라인에서 현실 이미지와 합성 이미지를 식별하는 데 어려움을 겪고 있음을 보여줍니다.

- **Technical Details**: 저자들은 Fact-checked Online Synthetic Image Dataset (FOSID)라는 새로운 데이터셋을 수집하여 합성 이미지의 온라인 생애주기 동안의 진화를 연구합니다. 이 과정에서, 탐지기들이 시간의 경과에 따라 성능이 저하되는 패턴을 발견하였고, Retrieval-Assisted Synthetic Image Detection (RASID) 기법을 사용하여 초기 탐지 성능을 유지할 수 있는 가능성을 제시합니다.

- **Performance Highlights**: 연구 결과, RASID 접근 방식을 통해 여러 최신 SID 메소드의 평균 탐지 효율을 각각 6.7%와 7.8% 향상시킬 수 있었습니다. 이는 균형 잡힌 정확도(balanced accuracy)와 AUC 지표에서의 향상입니다.



### DeRainGS: Gaussian Splatting for Enhanced Scene Reconstruction in Rainy (https://arxiv.org/abs/2408.11540)
- **What's New**: 이번 연구는 비오는 환경에서의 3D 재구성 새로운 과제를 제안하며, HydroViews 데이터셋을 구축하여 다양한 강도의 비 효과가 있는 합성 및 실제 장면 이미지를 포함하고 있습니다.

- **Technical Details**: 3DRRE(3D Reconstruction in Rainy Environments)라는 새로운 과제를 도입하고, 이는 비 오는 환경에서 3D 장면을 재구성하는 데 목적을 두고 있습니다. 이를 위해, DeRainGS라는 비 오는 환경 전용의 3DGS(3D Gaussian Splatting) 방법을 제안하고, 비에 의한 시각적 특성 및 기하학적 일관성을 처리하는 모듈을 통합하였습니다.

- **Performance Highlights**: 제안된 DeRainGS 방법은 다양한 비 시나리오에서 진행된 실험을 통해 기존의 오클루전(occlusion) 없는 방법들과 비교해 우수한 성능을 입증하였습니다.



### SAM-REF: Rethinking Image-Prompt Synergy for Refinement in Segment Anything (https://arxiv.org/abs/2408.11535)
- **What's New**: 이 논문에서는 SAM(세그먼트 어니씽 모델)의 한계를 극복할 새롭고 효율적인 세그멘테이션 접근법인 SAM-REF를 제안합니다. SAM-REF는 이미지와 프롬프트를 글로벌하고 로컬 차원에서 완벽하게 통합하여 세그멘테이션의 질을 향상시킵니다.

- **Technical Details**: SAM-REF는 두 단계로 구성된 정제 프레임워크로, 첫 번째 단계인 GlobalDiff Refiner는 이미지와 프롬프트를 결합하여 전체 세부 정보를 캡처합니다. 두 번째 단계인 PatchDiff Refiner는 마스크에 따라 객체 세부 정보 창을 정의하고 객체의 로컬 세부 정보를 정제합니다. 이 방법론은 기존 SAM 구조를 변경하지 않고도 세그멘테이션 마스크를 점진적으로 정제할 수 있습니다.

- **Performance Highlights**: SAM-REF는 여러 데이터셋에서 최신 성능을 달성하고, 특히 NoC90에서 최상위 성능을 기록하며, NoC95에서는 기존 최고 성능 모델(SimpleClick)을 능가하는 효율성을 보여줍니다. 또한, SAM의 낮은 지연 시간 이점을 유지하면서 세그멘테이션 품질에서 이전 방법들을 초과합니다.



### Just Project! Multi-Channel Despeckling, the Easy Way (https://arxiv.org/abs/2408.11531)
- **What's New**: 이번 논문에서는 다채널 SAR(Synthetic Aperture Radar) 이미지에서 스펙클(fluctuation) 변동을 줄이기 위한 새로운 프레임워크인 MuChaPro를 소개합니다. 이 프레임워크는 기존의 단일 채널 despeckling 방법을 활용하는 것이 특징입니다.

- **Technical Details**: MuChaPro는 여러 개의 단일 채널 프로젝션을 생성하고 이들을 복원한 후, 최종 다채널 추정치로 재결합(recombine)하는 간단한 접근 방식을 사용합니다. 이 방법은 편극(multipolarimetric) 및 간섭(interferometric) 모달리티에서 효과적임을 증명합니다.

- **Performance Highlights**: MuChaPro의 특별한 매력은 단일 채널 despeckling을 위한 센서-특정(sensor-specific) 네트워크를 학습할 수 있는 자기 지도(self-supervised) 훈련 전략을 적용할 수 있다는 점입니다.



### EmoFace: Emotion-Content Disentangled Speech-Driven 3D Talking Face with Mesh Attention (https://arxiv.org/abs/2408.11518)
- **What's New**: EmoFace라는 새로운 모델을 제안하여, 감정(Emotion)과 내용(Content) 정보를 별도로 추출하고, Mesh Attention 메커니즘을 활용하여 3D 얼굴 애니메이션의 보다 사실적인 생성이 가능하도록 했습니다.

- **Technical Details**: EmoFace는 Mesh Attention 메커니즘을 이용하여 메시(mesh) 정점 간의 시간적 및 공간적 특징 종속성을 학습합니다. 또한, teacher-forcing과 scheduled sampling을 결합한 자가 성장(self-growing) 훈련 방식을 채택하여, 훈련 데이터의 첫 번째 프레임이 반드시 무음이어야 하지 않도록 하여 데이터 제한을 크게 줄입니다.

- **Performance Highlights**: 제안된 3D 감정 얼굴 애니메이션 데이터셋(3D-RAVDESS)과 공공 데이터셋(VOCASET)에 대한 종합적인 정량적 및 정성적 평가 결과, EmoFace는 표준화된 최신 기술(SOTA) 방법보다 더 우수한 성능을 보였습니다.



### MSCPT: Few-shot Whole Slide Image Classification with Multi-scale and Context-focused Prompt Tuning (https://arxiv.org/abs/2408.11505)
Comments:
          11 pages, 5 figures, 5tables

- **What's New**: 이번 논문에서는 Whole Slide Images (WSI)에서 Few-shot Weakly Supervised Classification (FSWC) 작업을 위한 Multi-Scale and Context-focused Prompt Tuning (MSCPT) 방법을 제안합니다. MSCPT는 대규모 언어 모델(LLM)을 이용하여 다중 스케일에서 병리학적 시각 언어 사전 지식을 생성하고, 이를 구조화하여 WSI 수준의 특징을 추출합니다.

- **Technical Details**: MSCPT 방법론은 두 가지 주요 부품으로 이루어져 있습니다: (1) Multi-scale Hierarchical Prompt Tuning (MHPT) 모듈은 저 magnification(확대 배율) 및 고 magnification에서 병리적 시각 설명을 통합합니다. (2) Image-text Similarity-based Graph Prompt Tuning (ISGPT) 모듈은 WSI의 맥락 정보를 추출합니다.

- **Performance Highlights**: MSCPT는 3개의 데이터 세트와 2개의 VLM을 기반으로 한 실험에서 경쟁업체의 전통적인 MIL 기반 방법과 prompt tuning 방법을 초월하는 주목할만한 성능을 입증했습니다. 또한 ~0.9%의 적은 수의 훈련 가능한 매개변수만을 도입하여 우수한 성능을 나타냈습니다.



### XDT-CXR: Investigating Cross-Disease Transferability in Zero-Shot Binary Classification of Chest X-Rays (https://arxiv.org/abs/2408.11493)
Comments:
          Accepted in Machine Learning for Healthcare Conference MLHC 2024

- **What's New**: 이 연구는 의료 이미징에서 교차 질병 이전 가능성(cross-disease transferability, XDT)의 개념을 탐구하고 있으며, 특정 질병에 대해 훈련된 이진 분류기가 동일한 장기에 영향을 미치는 또 다른 질병에 대해 제로샷 분류(zero-shot classification)를 수행할 수 있는 가능성을 검토합니다.

- **Technical Details**: 연구에서는 주로 흉부 X선 영상(chest X-ray, CXR)을 사용하여 XDT 프레임워크를 구축합니다. 이 프레임워크는 비전 인코더의 임베딩 공간(embedding space)을 활용하며, 커널 변환(kernel transformation)을 통해 잠재적 공간(latent space)에서 질병 클래스와 비질병 클래스를 구분하는 데 도움을 줍니다.

- **Performance Highlights**: XDT-CXR 프레임워크는 기존의 제로샷 학습(zero-shot learning, ZSL) 기준보다 더 나은 예측 성능을 보이며, 전통적인 진단 테스트에 보조적인 역할을 강조합니다. 이는 임상 환경에서 자원을 효율적으로 관리하고 환자 치료를 최적화하는 데 기여할 수 있는 잠재력을 보여줍니다.



### E-Bench: Subjective-Aligned Benchmark Suite for Text-Driven Video Editing Quality Assessmen (https://arxiv.org/abs/2408.11481)
- **What's New**: 최근 텍스트 기반 비디오 편집 (text-driven video editing) 방법들이 급속히 발전하고 있지만, 편집된 비디오를 평가하는 것은 여전히 어려운 문제입니다. 그리하여 E-Bench라는 새로운 벤치마크 공간을 제안하며, 특히 텍스트 기반 비디오 편집의 평가를 위해 설계된 E-Bench DB와 E-Bench QA를 소개합니다.

- **Technical Details**: E-Bench DB는 다양한 동작과 주제를 가진 소스 비디오를 포함하며, 다양한 편집 프롬프트와 8가지 다른 모델에서의 편집 결과와 24명의 인간 주석자로부터 수집한 평균 의견 점수 (Mean Opinion Scores, MOS)를 포괄하고 있습니다. E-Bench QA는 텍스트-비디오 정합성 (text-video alignment) 및 원본 비디오와 편집된 비디오 간의 관련성 모델링을 강조한 새로운 평가지표입니다.

- **Performance Highlights**: E-Bench QA는 인간의 선호에 부합하는 정량적 평가를 제공하며, 기존의 영상 품질 평가 방법들보다 우수한 성능을 보입니다. 이는 텍스트-비디오 정합성과 관련된 평가뿐만 아니라 미적 품질 및 왜곡 측면에서도 다양한 시각에서 비디오를 평가하는 데 도움을 줄 것입니다.



### LAKD-Activation Mapping Distillation Based on Local Learning (https://arxiv.org/abs/2408.11478)
Comments:
          8 pages,7 figures

- **What's New**: 이 논문은 Local Attention Knowledge Distillation (LAKD)라는 새로운 지식 증류(Knowledge Distillation) 프레임워크를 제안합니다. 기존의 방법들이 증류된 정보를 비효율적으로 활용하거나 설명력이 부족한 점을 개선하고자 합니다. LAKD는 교사 모델(teacher model)로부터 효율적으로 증류된 정보를 사용하여 해석 가능성과 경쟁력 있는 성능을 달성합니다.

- **Technical Details**: LAKD는 두 가지 주요 모듈인 Separation-Decoupling Mechanism과 Non-Directional Activation Mapping을 포함합니다. Separation-Decoupling Mechanism은 기울기(truncation) 분할을 통해 학생 모델(student model)을 독립 모듈로 나누어 층 별로 지식을 학습하도록 하여 얕은 특징(shallow features)으로 인한 혼란을 줄이고 표현의 완전성을 향상시킵니다. Non-Directional Activation Mapping은 교사 모델의 주의를 학생 모델이 특정 중요한 특징에 집중하도록 안내하여 학습을 개선합니다. 이러한 메커니즘은 효과적인 모델 분리를 달성하고 학생의 학습 과정도 향상합니다.

- **Performance Highlights**: CIFAR-10, CIFAR-100 및 ImageNet 데이터셋에서 실험을 진행하였으며, LAKD 방법은 기존 방법들을 크게 초월하는 성능을 보였고, 다양한 데이터셋에서 항상 최첨단 성능(state-of-the-art performance)을 달성하였습니다.



### TrackGo: A Flexible and Efficient Method for Controllable Video Generation (https://arxiv.org/abs/2408.11475)
- **What's New**: TrackGo라는 새로운 비디오 생성 방법을 소개하며, 사용자가 제공한 자유형 마스크(free-form masks)와 화살표(arrows)를 활용하여 조건부 비디오 생성을 가능하게 합니다. 이 접근법은 복잡한 시나리오에서 비디오 내용을 조작하는데 유연하고 정밀한 메커니즘을 제공합니다.

- **Technical Details**: TrackGo는 사용자가 정의한 마스크와 화살표를 기반으로 점 궤적(point trajectories)을 자동으로 추출하고, 이를 조건부 비디오 생성의 기초로 활용합니다. TrackAdapter라는 새로운 구성 요소를 통해 사전 훈련된 비디오 생성 모델의 시간적 자가 주의 레이어에 모션(control information)을 효과적으로 통합합니다. 이 구조는 두 개의 분기(branch)로 구성되어 있으며, 하나는 원래의 자가 주의 메커니즘을 유지하고 다른 하나는 대상 영역의 모션에 집중합니다.

- **Performance Highlights**: 실험 결과, TrackGo와 TrackAdapter가 FVD(Frechet Video Distance), FID(Frechet Inception Distance), ObjMC(Object Motion Coherency) 같은 주요 지표에서 최첨단 성능을 달성함을 보여줍니다.



### MeTTA: Single-View to 3D Textured Mesh Reconstruction with Test-Time Adaptation (https://arxiv.org/abs/2408.11465)
Comments:
          Accepted at BMVC 2024. [Project page] this https URL

- **What's New**: 이번 연구에서는 단일 뷰 이미지로부터 3D를 재구성하는 과제를 다루며, MeTTA라는 새로운 TTA(test-time adaptation) 기법을 제안합니다. MeTTA는 생성적 사전(generative prior)을 활용하여 훈련 데이터와 유사하지 않은 상황에서도 테스트 시 효과적으로 적응할 수 있는 방법론입니다.

- **Technical Details**: MeTTA는 3D 형상(mesh), 텍스처(texture), 카메라 뷰포인트(viewpoint)를 공동 최적화하는 방법으로 Out-of-Distribution(OoD) 사례를 처리합니다. 이 과정에서 학습 가능한 가상 카메라를 사용하여 2D 픽셀 정보를 3D 형상과 정렬하는 자기 보정(self-calibration)을 설계합니다. 또한, 텍스처 맵은 물리 기반 렌더링(Physically Based Rendering, PBR) 매개변수를 통해 매개화됩니다.

- **Performance Highlights**: 실험 결과, MeTTA는 기존 학습 기반 3D 재구성 모델의 실패 사례에서도 OoD 상황을 효과적으로 처리하며, 사실적 외관을 가진 물체를 재구성할 수 있는 가능성을 보여주었습니다. 이는 Blender와 같은 그래픽 툴에서 재조명(relighting) 및 재질(material) 조정 편집에 활용될 수 있는 높은 충실도(high-fidelity) 형태로 나타났습니다.



### MambaOcc: Visual State Space Model for BEV-based Occupancy Prediction with Local Adaptive Reordering (https://arxiv.org/abs/2408.11464)
- **What's New**: 본 논문에서는 Mamba 기반의 점유 예측 방법(MambaOcc)을 제안하여 3D 시나리오 표현의 부담을 줄이고, 효율적인 장거리 지각(long-range perception)을 달성하고자 합니다.

- **Technical Details**: MambaOcc는 BEV(Bird's-Eye-View) 특징을 활용하여 3D 공간의 점유 상태를 예측합니다. 또한, 디포르머블 컨볼루션(deformable convolution)과 함께 활용되는 로컬 적응형 재배치(local adaptive reordering) 메커니즘을 통해 Mamba의 시퀀스 순서에 대한 민감성을 해결합니다.

- **Performance Highlights**: MambaOcc는 Occ3D-nuScenes 데이터셋에서 최신 성능을 달성하였으며, FlashOcc에 비해 파라미터 수를 42% 줄이고, 계산 비용을 39% 줄이면서도 우수한 결과를 기록하였습니다.



### Low-Light Object Tracking: A Benchmark (https://arxiv.org/abs/2408.11463)
- **What's New**: 본 논문에서는 저조도 환경에서 객체 추적을 위한 새로운 벤치마크인 LLOT (Low-Light Object Tracking)를 소개합니다. 이 벤치마크는 269개의 도전적인 시퀀스를 포함하며, 저조도 환경에서의 객체 추적 향상에 기여하고자 합니다.

- **Technical Details**: LLOT 데이터셋은 총 132K 프레임 이상을 포함하고 있으며, 각 시퀀스는 경계 상자와 함께 정교하게 주석이 달려 있습니다. 본 연구는 H-DCPT라는 새로운 트래커를 제안하며, 이는 역사적 및 어두운 단서 프롬프트를 결합하여 저조도 환경에서의 추적 성능을 개선합니다.

- **Performance Highlights**: H-DCPT는 기존의 39개 최신 트래킹 알고리즘을 모두 초월하며, 저조도 조건에서의 객체 추적 성능을 크게 향상시키고 있습니다. 이 연구 결과는 저조도 조건에서의 객체 추적 분야에서 혁신을 촉진할 것으로 기대됩니다.



### Lookism: The overlooked bias in computer vision (https://arxiv.org/abs/2408.11448)
Comments:
          Paper accepted at the ECCV 2024 workshop named "Fairness and ethics towards transparent AI: facing the chalLEnge through model Debiasing (FAILED)", this https URL

- **What's New**: 이 논문은 컴퓨터 비전 모델 내에서 lookism(외모편향)의 체계적인 연구 필요성을 강조합니다. 특히, 외모에 기반한 차별이 AI 기술의 공정성과 포괄성을 해칠 수 있는 잠재력을 가지고 있음을 경고하며, 이를 해결하기 위한 학제 간 접근 방식을 촉구하고 있습니다.

- **Technical Details**: lookism은 사회적 아름다움 기준과 인지 편향에 뿌리를 두고 있으며, 이는 컴퓨터 비전 시스템에서 불공정한 대우와 해로운 고정관념을 강화할 수 있습니다. 이 논문은 외모에 따른 차별을 탐구하고 이를 완화하기 위한 전략을 개발하는 것을 목표로 합니다.

- **Performance Highlights**: 다양한 연령대와 인종 배경을 가진 얼굴 이미지 데이터셋을 사용하여 beauty filter의 효과를 분석한 결과, beauty filter가 대다수 이미지의 인지된 매력을 증가시키는 것으로 나타났습니다. 그러나, 여성의 매력 점수는 높아졌지만, 지능 점수는 남성 이미지에서 더 높아져 성별 편향을 악화시키는 것으로 분석되었습니다.



### GaussianOcc: Fully Self-supervised and Efficient 3D Occupancy Estimation with Gaussian Splatting (https://arxiv.org/abs/2408.11447)
Comments:
          Project page: this https URL

- **What's New**: GaussianOcc는 완전 자가 감독(self-supervised) 및 효율적인 3D 점유 추정(occupancy estimation) 방법을 소개합니다. 기존 방법은 훈련 중에 센서의 6D 포즈에 대한 진실값(ground truth)을 요구했으나, Gaussian Splatting을 통해 이를 개선하여 자가 감독 훈련을 가능하게 했습니다.

- **Technical Details**: GaussianOcc는 두 가지 주요 모듈인 Gaussian Splatting for Projection (GSP)과 Gaussian Splatting from Voxel space (GSV)을 통해 구현됩니다. GSP는 인접 시점으로부터 정확한 스케일 정보를 제공하며, GSV는 3D voxel 공간에서의 빠른 렌더링 속성을 활용하여 훈련 시간을 2.7배 단축시키고 렌더링 시간을 5배 개선합니다.

- **Performance Highlights**: 제안된 GaussianOcc 방법은 기존의 볼륨 렌더링(volume rendering) 방법 대비 경쟁력 있는 성능을 보이며, 훈련 속도는 2.7배, 렌더링 속도는 5배 향상되었습니다. 이러한 결과는 자가 감독 3D 점유 추정의 효율성을 크게 높이는 데 기여합니다.



### BAdd: Bias Mitigation through Bias Addition (https://arxiv.org/abs/2408.11439)
- **What's New**: 본 논문에서는 BAdd라는 새로운 방법론을 제안하여, 다중 속성의 편향이 존재하는 실제 시나리오에서 공정한 표현을 학습할 수 있도록 돕습니다. 기존 방법들이 단일 속성에서의 편향 처리에 효과적이었지만, 다중 속성에 대한 처리에서는 한계를 보였습니다.

- **Technical Details**: BAdd는 모델 구조의 마지막 전단계(output)에서 편향을 포착하는 기능을 통합하 여 편향 속성에 독립적인 표현을 학습하는 데 중점을 둡니다. 이는 손실 함수의 최적화 과정에서 모델이 편향 속성을 학습하지 않도록 돕습니다. 이 방법은 Biased-MNIST, CelebA와 같은 다중 속성 데이터셋에서의 기존 최첨단(state-of-the-art) 방법들보다 저명한 성능을 발휘합니다.

- **Performance Highlights**: BAdd는 FB-Biased-MNIST와 CelebA라는 도전적인 다중 속성 기준에서 각각 +27.5% 및 +5.5%의 정확도 절대 향상을 달성하여 기존의 모든 방법들을 초월하는 성과를 보였습니다. 이를 통해 단일 및 다중 속성 편향 시나리오에서 BAdd의 우수성을 보여주고 있습니다.



### T2VIndexer: A Generative Video Indexer for Efficient Text-Video Retrieva (https://arxiv.org/abs/2408.11432)
- **What's New**: 본 논문에서는 효율적인 텍스트-비디오 검색을 위한 새로운 모델 기반 비디오 인덱서인 T2VIndexer를 소개합니다. 이 모델은 비디오 식별자를 직접 생성하고, 일정한 시간 복잡도로 후보 비디오를 검색하는 시퀀스-투-시퀀스(generative sequence-to-sequence) 생성 모델입니다.

- **Technical Details**: T2VIndexer는 쿼리 텍스트와 비디오 후보 간의 직접적인 상호작용을 위한 딥러닝 모델로서, 비디오 식별자의 시맨틱 표현을 층화하여 인코딩하는 과정을 통해 비디오를 단순한 시퀀스로 표현합니다. 이 모델은 정밀한 아이덴티파이드 예측을 위해 T5 아키텍처 기반의 생성 네트워크를 훈련합니다. 또한 CLIP을 이용해 각 비디오를 인코드하고, 다각도의 비디오 콘텐츠에 대한 새 쿼리를 생성하여 학습합니다.

- **Performance Highlights**: T2VIndexer는 MSR-VTT, MSVD, ActivityNet, DiDeMo와 같은 표준 데이터셋에서 기존의 최첨단 모델에 비해 검색 효율성을 일관되게 향상시키며, 검색 시간이 30%에서 50%까지 단축됩니다. MSR-VTT에서 +1.0%, MSVD에서 +1.8%, ActivityNet에서 +1.5%, DiDeMo에서 +0.2%의 성능 향상을 보여줍니다.



### EMO-LLaMA: Enhancing Facial Emotion Understanding with Instruction Tuning (https://arxiv.org/abs/2408.11424)
- **What's New**: 본 연구에서는 기존의 Facial Expression Recognition (FER) 방법들이 가지고 있는 한계점을 극복하기 위해 새로운 Multimodal Large Language Model (MLLM)인 EMO-LLaMA를 제안합니다. 특히, EMO-LLaMA는 사전 훈련된 얼굴 분석 네트워크에서 facial priors를 통합하여 인간의 얼굴 정보를 향상시키고, static 및 dynamic FER 데이터셋 모두에서 경쟁력 있는 성과를 달성합니다.

- **Technical Details**: EMO-LLaMA 모델은 LoRA 특정 훈련을 통해 사전 훈련된 MLLM에 facial priors를 통합하여 설계되었습니다. 이 모델은 Face Info Mining 모듈을 통해 글로벌 및 로컬 얼굴 정보를 추출하며, 감정 차이를 고려하여 나이-성별-인종 특성을 도입하기 위한 수작업 프롬프트를 사용합니다.

- **Performance Highlights**: EMO-LLaMA는 6개의 FER 데이터셋에서 SOTA에 준하는 또는 경쟁력 있는 성과를 보여주었습니다. 실험을 통해 EMO-LLaMA의 일반화 능력이 현재의 패러다임에서 부족한 점을 보완했음을 입증했습니다.



### Pano2Room: Novel View Synthesis from a Single Indoor Panorama (https://arxiv.org/abs/2408.11413)
Comments:
          SIGGRAPH Asia 2024 Conference Papers (SA Conference Papers '24), December 3--6, 2024, Tokyo, Japan

- **What's New**: 최근 세부 정보가 풍부한 3D 실내 장면을 단일 파노라마 이미지에서 자동으로 복원하는 Pano2Room이라는 새로운 접근 방식을 소개합니다.

- **Technical Details**: Pano2Room은 초기 메쉬를 입력 파노라마에서 구성하고, 파노라마 RGBD inpainter를 사용하여 반복적으로 이 메쉬를 정제합니다. 이 과정에서 사실적인 3D 일관성이 있는 새로운 뷰를 생성합니다. 최종적으로, 정제된 메쉬는 3D Gaussian Splatting(3DGS) 필드로 변환되고 수집된 가짜 새로운 뷰로 학습됩니다. 이 파이프라인은 큰 차폐가 있는 실제 3D 장면의 재구성을 가능하게 합니다.

- **Performance Highlights**: Pano2Room은 기존 최첨단 방법들과 비교하여 단일 파노라마 실내 새로운 뷰 합성에서 우수성을 입증했습니다. 다양한 도전적인 데이터셋에서의 평가 결과, 효율적인 성능 향상과 대형 차폐 처리 능력이 크게 향상되었습니다.



### SelfDRSC++: Self-Supervised Learning for Dual Reversed Rolling Shutter Correction (https://arxiv.org/abs/2408.11411)
Comments:
          13 pages, 9 figures, and the code is available at \url{this https URL}

- **What's New**: 이 논문은 기존의 완전 감독 학습 방식 대신, 드문드문 캘리브레이션 할 필요 없이 자가 감독 학습(self-supervised learning) 프레임워크를 제안하여 Rolling Shutter(RS) 왜곡을 효율적으로 교정하는 SelfDRSC++을 소개합니다.

- **Technical Details**: SelfDRSC++는 bidirectional correlation matching block을 포함한 Lightweight DRSC 네트워크를 사용하여 최적의 optical flow와 교정된 RS 피처를 공동 최적화하는 방법을 사용합니다. 이 방법은 입력과 재구성된 이중 역방향 RS 이미지 간의 사이클 일관성을 보장하는 자가 감독 학습 전략을 제안합니다.

- **Performance Highlights**: SelfDRSC++는 실제 세계의 RS 사례에 대한 고성능을 달성하며, 이를 통해 더 나은 텍스처와 임시 일관성(temporal consistency)을 제공합니다. 또한 실험적 비교에서 최신 감독 RS 교정 방법과도 유사한 정량적 성능을 달성했습니다.



### Latent Feature and Attention Dual Erasure Attack against Multi-View Diffusion Models for 3D Assets Protection (https://arxiv.org/abs/2408.11408)
- **What's New**: 이 논문은 Multi-View Diffusion Models (MVDMs)으로 인한 지적 재산권 침해 문제를 처음으로 다루며, 새로운 공격 방법을 제안합니다. 이는 기존의 단일 이미지 생성 작업에 의해 초래된 지적 재산권 문제 해결을 넘어 MVDMs에 특화된 접근방식을 제공합니다.

- **Technical Details**: 제안된 방법은 'latent feature'와 'attention'의 이중 지우기 공격(dual erasure attack)으로, 생성된 여러 뷰 이미지를 통해 'inner feature'와 'consistency'를 동시에 방해합니다. MVDMs의 다양한 'attention' 메커니즘을 고려하여 'identical attention erasure loss'를 제안하고, 이로 인해 이미지의 포그라운드에서 백그라운드로 'attention'을 전환하여 생성된 이미지들 간의 일관성을 방해합니다.

- **Performance Highlights**: SOTA MVDMs에 대한 실험 결과, 제안된 방법이 공격의 효과, 전이성(transferability), 방어 방법에 대한 강인성(robustness) 측면에서 우수한 성능을 보였습니다. 이로 인해 3D 자산을 MVDMs 기반의 3D 기하학 재구성으로부터 효과적으로 보호할 수 있는 솔루션을 제공합니다.



### Domain-invariant Progressive Knowledge Distillation for UAV-based Object Detection (https://arxiv.org/abs/2408.11407)
- **What's New**: 본 연구에서는 UAV 기반 객체 탐지를 위한 새로운 지식 증류(Knowledge Distillation, KD) 프레임워크를 제안합니다. 이 프레임워크는 교사 모델(teacher model)과 학생 모델(student model) 간의 특징 격차를 완화하고, Fast Fourier Transform (FFT)을 이용하여 도메인 불변 특징(domain-invariant features)을 추출합니다.

- **Technical Details**: 제안된 방법은 두 단계의 지식 증류 전략을 채택합니다. 첫 번째 단계에서는 YOLOv7-X를 고급 교사 모델(senior teacher)로 사용하고, YOLOv7-L을 주니어 교사 모델(junior teacher)로 설정하여 학생 모델을 가르치는 구조입니다. 두 번째 단계에서는 주니어 교사 모델을 사용해 학생 모델을 코칭하며, 이 과정에서 매개변수를 업데이트합니다.

- **Performance Highlights**: 제안된 방법은 UAV-OD의 두 가지 데이터셋에서 최첨단(SoTA) 성능을 달성하였으며, 기존 방법들과 비교해 학생 모델의 정확도를 효과적으로 향상시켰습니다.



### Video Diffusion Models are Strong Video Inpainter (https://arxiv.org/abs/2408.11402)
- **What's New**: 본 논문에서는 비디오 인페인팅(video inpainting) 작업에 최초로 이미지-비디오 확산 모델(image-to-video diffusion models)을 효과적으로 통합한 '첫 번째 프레임 채우기 비디오 확산 인페인팅 모델'(First Frame Filling Video Diffusion Inpainting, FFF-VDI)을 제안합니다.

- **Technical Details**: FFF-VDI 모델은 각 프레임의 잠재 코드(latent code)를 VAE 인코더(VAE encoder)를 통해 추출하여 마스크된 노이즈 잠재 코드(noise latent code)를 생성합니다. 이후 향후 프레임의 노이즈 잠재 코드를 일차 프레임의 마스크된 부분에 전파합니다. 이를 통해 템포랄 일관성(temporal consistency)을 개선하고 노이즈 수준에서 왜곡을 최소화합니다. 또한, DDIM 인버전(DDIM inversion)을 적용하여 인페인팅 과정 중 불필요한 오브젝트 환각(hallucination) 효과를 줄입니다.

- **Performance Highlights**: 여러 비교 실험을 통해 FFF-VDI 모델이 다양한 인페인팅 유형을 처리하는 데 있어 기존 방법들보다 더 뛰어난 품질을 제공함을 입증하였습니다. 특히, 대규모 마스크를 이용한 오브젝트 제거(object removal)의 경우 전통적인 광학 흐름(optical flow) 기반 방법들보다 훨씬 강력한 성능을 보입니다.



### Revisiting FunnyBirds evaluation framework for prototypical parts networks (https://arxiv.org/abs/2408.11401)
Comments:
          Published at 2nd XAI World Conference

- **What's New**: 이 논문에서는 프로토타입 부분 네트워크(ProtoPNet)의 설명 품질 평가를 위해 새로운 시각화 방법인 similarity maps(유사성 맵)를 제안합니다. 기존의 범위 박스(bounding boxes) 사용 방식과 비교하여, similarity maps가 ProtoPNet의 본질에 더 잘 부합한다는 점을 강조합니다.

- **Technical Details**: 논문의 주요 초점은 FunnyBirds 벤치마크 내에서 프로토타입 부분 네트워크의 설명을 범위 박스 대신 similarity maps로 평가하는 것입니다. 연구자들은 다양한 메트릭 점수를 비교하여 similarity maps 기반 설명이 더 정확한 평가 결과를 도출한다는 것을 밝혔습니다.

- **Performance Highlights**: 본 연구의 결과는 similarity maps의 사용이 ProtoPNet 모델과의 직관적 일치를 증가시킨다는 것을 보여줍니다. 이는 측정된 메트릭 점수의 차이로 입증되며, 커뮤니티 내에서 설명의 신뢰할 수 있는 비교가 가능하도록 합니다.



### EAGLE: Elevating Geometric Reasoning through LLM-empowered Visual Instruction Tuning (https://arxiv.org/abs/2408.11397)
- **What's New**: 최근 다중 모달 대형 언어 모델(Multi-modal Large Language Models, MLLMs)의 급속한 발전에도 불구하고, 수학적 기하학 문제 해결에서 여전히 어려움을 겪고 있다는 점이 발견되었습니다. 기존 MLLMs는 대체로 LLM 백본(LLM backbone) 최적화에 집중하고, 시각적 이해의 개선은 잘 이루어지지 않았습니다. 본 논문은 EAGLE이라는 새로운 2단계 시각 향상 프레임워크를 제안하여 기하학적 추론을 강화하는 방법을 또렷이 밝히고 있습니다.

- **Technical Details**: EAGLE는 두 가지 단계를 포함하는 end-to-end visual enhancement framework입니다. 첫 번째 단계에서, 60K의 기하학 이미지-캡션 쌍(geometric image-caption pairs)을 사용하여 CLIP ViT로 구성된 MLLM을 기본 기하학 지식으로 훈련시킵니다. 이후 두 번째 단계에서는 시각 인코더의 LoRA 모듈을 추가하여 LLM 백본을 동적으로 활성화하고, 110K의 question-answer 쌍을 통해 모델이 미세한 기하학적 단서에 집중할 수 있도록 합니다.

- **Performance Highlights**: EAGLE-7B는 GeoQA와 MathVista 벤치마크에서 뛰어난 결과를 보이며, G-LLaVA 7B 모델을 2.9% 초과하며, G-LLaVA 13B 모델에 비해 소폭 우수한 성능을 기록하였습니다. MathVista 벤치마크에서는 GPT-4V 모델에 비해 3.8%의 개선을 달성하였습니다.



### Fairness measures for biometric quality assessmen (https://arxiv.org/abs/2408.11392)
- **What's New**: 이 연구는 다양한 인구 통계 그룹 간의 생체 샘플 품질 평가의 공정성을 분석하기 위한 새로운 통계적 접근 방식을 도입하고 비교합니다. 이를 통해 생체 시스템의 품질 평가 알고리즘이 인구 통계적 특성을 고려하지 않고 동등하게 작동할 수 있도록 하는 공정성 측정 기준을 정의하는 데 기여합니다.

- **Technical Details**: 제안된 공정성 측정 방법은 Gini coefficient(GC)를 기반으로 하며, 집단 간의 품질 점수 분포를 사용하여 공정성을 평가합니다. 품질 구성 요소별 평균 또는 중앙값 품질 점수를 입력으로 활용하며, 데이터 그룹 간의 자가 비교 문제를 고려합니다. 이러한 접근 방식은 공정성을 양적으로 측정하고 향후 국제 표준에 적용될 수 있는 잠재적 후보로 수용할 수 있습니다.

- **Performance Highlights**: 연구는 NIST 품질 평가 방법론과 같은 기존 품질 평가 시스템들이 인구 통계적 편향을 보이는 경우가 있고, 이로 인해 품질 점수의 차이에 따라 서로 다른 폐기 비율이 발생할 수 있다는 점을 강조합니다. 새로운 공정성 측정 기준은 생체 인식 시스템의 실제 적용에 있어 더욱 공정한 결과를 이끌어낼 것으로 기대됩니다.



### Current Status and Trends in Image Anti-Forensics Research: A Bibliometric Analysis (https://arxiv.org/abs/2408.11365)
- **What's New**: 이 연구는 이미지 프라이버시 및 보안 분야에서의 이미지 안티 포렌식(image anti-forensics)에 대한 최초의 포괄적인 문헌계량 분석(bibliometric analysis)으로, 연구 동향 및 발전을 요약합니다.

- **Technical Details**: VOSViewer 소프트웨어를 사용하여 Web of Science Core Collection (WoSCC) 데이터베이스 내 출판물들을 분석하여, 연구 트렌드, 주요 연구 기관, 가장 영향력 있는 출판물, 최상위 출판 장소 및 가장 활동적인 기여자를 파악하였습니다.

- **Performance Highlights**: 최근의 주요 연구 방향을 강조하며, 향후 이미지 안티 포렌식에 대한 연구의 참고 자료로 활용될 수 있습니다.



### HumanCoser: Layered 3D Human Generation via Semantic-Aware Diffusion Mod (https://arxiv.org/abs/2408.11357)
- **What's New**: 본 논문에서는 텍스트 프롬프트를 기반으로 물리적으로 층화된 3D 인간을 생성하는 방법을 제안합니다. 기존 방식들은 제약이 있어, 복잡한 옷차림을 지닌 3D 인간을 효율적으로 생성할 수 없었습니다. 우리는 물리적으로 분리된 확산 모델을 바탕으로 한 새로운 층별 복장 표현 방식을 도입하여, 재사용과 복잡한 의류를 재현할 수 있는 방법을 해결했습니다.

- **Technical Details**: 제안된 HumanCoser는 두 단계로 구성된 방법으로, 첫 번째 단계에서 최소화된 인간 모델을 생성하고, 두 번째 단계에서 복장이 분리된 형태로 생성되어 인간 모델과 매칭됩니다. 패러미터화된 네트워크와 SMPL(Skinned Multi-Person Linear Model) 기반의 암시적 변형 네트워크를 사용하여, 옷과 몸체의 분리된 생성을 구현합니다. 또한, 다층 융합 렌더링 방법을 통해 최종적으로 층화된 3D 인간을 생성합니다.

- **Performance Highlights**: 광범위한 실험을 통해, 제안된 방법이 복잡한 의류를 가진 층화된 3D 인간 생성을 가능하게 할 뿐만 아니라, 가상 착용 및 층화된 인간 애니메이션 지원 또한 뛰어난 성능을 보여준다는 것을 입증하였습니다.



### Image Score: Learning and Evaluating Human Preferences for Mercari Search (https://arxiv.org/abs/2408.11349)
- **What's New**: Mercari는 사용자 행동과 잘 연관된 이미지 미적 요소 레이블을 생성하기 위해 Chain-of-Thought(코드)의 활용을 제시합니다. 본 연구는 LLM을 사용해 저비용으로 이미지 품질을 평가하고 예측하는 모델을 제안합니다.

- **Technical Details**: 본 논문에서는 LLM을 통해 이미지 품질을 평가하는 새로운 방법론을 소개합니다. LLM이 생성한 이미지 품질 레이블은 사용자 클릭과 같은 암묵적 피드백과 잘 연관되어 있으며, 데이터 수집은 유사한 가격대의 아이템들로 구성됩니다. GPT-4 비전 API를 이용해 이미지의 상대적 품질 평가를 수행했습니다.

- **Performance Highlights**: 본 연구에서 제안한 모델인 Image Score는 과거 클릭률 기반 예측 방식보다 클릭 예측 성능이 크게 향상되었으며, 온라인 실험을 통해 웹 플랫폼에서 거의 7%의 매출 성장률을 기록했습니다.



### Optimizing Transmit Field Inhomogeneity of Parallel RF Transmit Design in 7T MRI using Deep Learning (https://arxiv.org/abs/2408.11323)
- **What's New**: 본 연구에서는 B1+ 필드 균일성을 개선하기 위해 새로운 두 단계의 딥 러닝 기반 전략을 제안합니다. 기존의 Magnitude Least Squares (MLS) 최적화 방법과는 달리, 시간 소모가 적고 환자의 존재에 의존하지 않는 접근 방식입니다.

- **Technical Details**: 첫 번째 단계에서는 랜덤 초기화된 Adaptive Moment Estimation을 사용하여 다채널 B1+ 필드로부터 원하는 RF shimming 가중치를 획득합니다. 이후 Residual Networks (ResNets)를 활용하여 B1+ 필드를 입력으로 하고 목표 RF shimming 출력을 생성하는 모델을 학습합니다. 이 과정에서 RMS 오류와 작동 시간을 비교하여 MLS 최적화와의 이점을 확인합니다.

- **Performance Highlights**: 제안하는 전략은 RF shimming 설계를 더 빠르고 효율적으로 진행할 수 있게 하여 UHF에서의 이미징 품질을 상당히 향상시킵니다. 전통적인 MLS 최적화 방법과 비교하여 속도와 정확성 모두에서 이점을 나타냅니다.



### TWLV-I: Analysis and Insights from Holistic Evaluation on Video Foundation Models (https://arxiv.org/abs/2408.11318)
Comments:
          17 pages; Twelve Labs Technical Report

- **What's New**: 본 연구에서는 비디오 파운데이션 모델(video foundation models)을 공정하고 강건한 방식으로 평가하는 방법을 논의합니다. 비디오 모델들이 다양한 매개변수(예: 샘플링 속도, 프레임 수 등)로 평가되어 공정한 비교를 어렵게 하는 반면, 우리는 외관(appearance)과 움직임(motion) 이해의 두 가지 핵심 능력을 평가하기 위한 새로운 프레임워크를 제안합니다.

- **Technical Details**: 본 연구에서 제안하는 TWLV-I 모델은 비디오의 움직임과 외관을 모두 포착할 수 있는 임베딩 벡터(embedding vector)를 제공합니다. Vision Transformer(ViT) 아키텍처를 채택하였으며, 여러 공공 데이터셋에서 사전 훈련(pretraining)을 수행했습니다. 평균 top-1 정확도(accuracy) 측정에서 TWLV-I는 기존 모델 V-JEPA와 UMT에 비해 각각 4.6%p 및 7.7%p 향상된 성능을 보였습니다.

- **Performance Highlights**: TWLV-I는 행동 인식(action recognition), 시간 행동 위치 지정(temporal action localization), 시공간 행동 위치 지정(spatiotemporal action localization) 및 시간 행동 분할(temporal action segmentation)과 같은 여러 비디오 중심 작업에서 최첨단 성능을 보여줍니다. 또한, 기존 모델들(UMT, V-JEPA, InternVideo2)과 비교하여 일관된 성능 향상을 나타내며, 실험 결과에서 우수한 인식 능력을 입증했습니다.



### Swarm Intelligence in Geo-Localization: A Multi-Agent Large Vision-Language Model Collaborative Framework (https://arxiv.org/abs/2408.11312)
- **What's New**: 이번 논문에서는 여러 LVLM (Large Vision-Language Models) 에이전트 간의 커뮤니케이션을 통해 이미지의 효과적인 지오 로컬라이제이션을 달성하는 새로운 시각적 지오 로컬라이제이션 프레임워크인 \name\을 소개합니다.

- **Technical Details**: 이 프레임워크는 에이전트 간의 커뮤니케이션 패턴을 최적화하는 동적 학습 전략을 사용하여 불필요한 논의를 줄이고 효율성을 향상시킵니다. 또한, 새로운 데이터셋인 GeoGlobe를 구축하여 시각적 지오 로컬라이제이션 작업에 적용합니다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식이 최신 기술 (state-of-the-art) 방법과 비교하여 상당한 성능 향상을 보여줍니다.



### Improving Out-of-Distribution Data Handling and Corruption Resistance via Modern Hopfield Networks (https://arxiv.org/abs/2408.11309)
- **What's New**: 현대 Hopfield 네트워크(MHN)의 통합을 통해, 컴퓨터 비전 모델의 out-of-distribution 데이터 처리 능력을 향상시키는 새로운 접근 방식을 제안합니다. 전통적인 컴퓨터 비전 모델의 성능 한계를 극복하기 위해, MHN을 활용하여 데이터 복원력을 높이는 방법론을 소개하였습니다.

- **Technical Details**: 이 연구에서는 MNIST-C 데이터셋을 활용하여 MHN과 기본 모델의 통합 효과를 평가하였습니다. 제안된 통합 알고리즘은 test-time에 적용되며, 오프라인에서 훈련된 모델과 조합하여 실행할 수 있습니다. 이 방법은 Gaussian 노이즈 등 다양한 데이터 손상을 처리할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 평균적으로 13.84%의 정확도 향상과 평균 손상 오류(mCE)를 57.49% 감소시켜 이전 모델 대비 우수한 성능을 보였습니다. 또한, 원본 비부패 데이터로의 수렴 능력을 검증하여 실용적인 적용 가능성을 강조하였습니다.



### UniFashion: A Unified Vision-Language Model for Multimodal Fashion Retrieval and Generation (https://arxiv.org/abs/2408.11305)
- **What's New**: 이 논문에서는 패션 도메인에서 멀티모달 생성 및 검색 작업의 도전을 동시에 해결하는 통합 프레임워크인 UniFashion을 제안합니다.

- **Technical Details**: UniFashion은 이미지 생성, 검색 작업 및 텍스트 생성 작업을 통합합니다. 이 모델은 LLM(large language model)과 확산 모델(diffusion model)을 결합하여 제어 가능하고 고충실도의 생성을 가능하게 합니다. Q-Former는 멀티모달 쿼리를 생성하고, 이를 기반으로 LLM이 캡션을 생성하며, 확산 모델이 이미지를 합성하는 방식으로 작동합니다.

- **Performance Highlights**: UniFashion은 다양한 패션 작업에서 이전의 단일 작업 모델에 비해 현저히 향상된 성능을 보이며, 복잡한 비전-언어 작업을 관리하는 데 쉽게 적응할 수 있습니다.



### Making Large Vision Language Models to be Good Few-shot Learners (https://arxiv.org/abs/2408.11297)
- **What's New**: 본 논문은 Few-shot classification (FSC)에서 LVLMs의 성능을 조사하고, 학습 부족과 심각한 위치 바이어스 등의 문제를 식별합니다. 이를 해결하기 위해 메타 학습 전략을 도입하여 모델이 'learn to learn'할 수 있도록 합니다.

- **Technical Details**: 메타 학습을 통한 지침 세분화 및 레이블 증강, 후보 선택 방법을 설계했습니다. 레이블 증강은 문자 변형 전략을 통해 모델이 지원 정보를 집중하도록 합니다. 후보 선택은 속성 설명을 활용하여 신뢰할 수 없는 후보를 필터링합니다.

- **Performance Highlights**: 참여 실험 결과, 내 방법론은 8개의 FSC 벤치마크에서 높은 성능을 달성하며, LVLMs의 적용 가능성을 입증합니다. 또한, 후보 선택 전략은 학습 없는 LVLMs에도 유익한 결과를 제공합니다.



### Taming Generative Diffusion for Universal Blind Image Restoration (https://arxiv.org/abs/2408.11287)
Comments:
          14 pages, 9 figures, 8 tables

- **What's New**: 최근 Diffusion 모델은 이미지 복원에 널리 사용되고 있습니다. 본 논문에서는 기존의 블라인드 이미지 복원 방법들이 성능 한계를 극복하고, 다양한 복원 상황에서 실질적인 적용 가능성을 높이기 위해, 생성적 Diffusion Prior를 활용한 새로운 모델인 BIR-D를 제안합니다.

- **Technical Details**: BIR-D는 각 디노이징 단계에서 편광 convolutional kernel을 조정하여 병렬적으로 복원 과정을 수행합니다. 저자들은 가이드 스케일을 결정하는 경험공식을 제공하여 최적의 파라미터를 실시간으로 계산할 수 있게 했습니다. 이로 인해 다양한 이미지 복원 작업에서 향상된 품질과 효율성을 보여줍니다.

- **Performance Highlights**: BIR-D는 다양한 실세계 및 합성 데이터셋에서 기존의 비지도 방법보다 우수한 실용성과 다재다능성을 입증하였으며, 다중 가이드를 통한 이미지 복원과 복잡한 손상에 대한 이미지 복원도 가능하게 되었습니다.



### Video Emotion Open-vocabulary Recognition Based on Multimodal Large Language Mod (https://arxiv.org/abs/2408.11286)
- **What's New**: 본 연구는 고정된 레이블에 의존하는 전통적인 데이터 세트의 한계를 극복하고, 다양한 감정의 변화를 효과적으로 포착하기 위해 MLLMs(다중 모달 대형 언어 모델) 기술을 사용하여 비디오로부터 개방 어휘 감정 레이블을 생성하는 방법을 제안합니다. 이러한 혁신적인 접근법은 MER2024 챌린지의 MER-OV(오픈-워드 감정 인식)에서 두드러진 성과를 기록했습니다.

- **Technical Details**: 이 연구는 InternVL 프레임워크와 AffectGPT를 활용하여 감정 인식 훈련을 수행하며, 이미지, 음성 및 텍스트의 3가지 모드를 결합한 새로운 접근법을 제안합니다. 또한, Qwen-VL과 CogVLM의 생성 능력을 통해 부족한 데이터 문제를 해결하고, 다중 모델 통합 전략을 통해 모델의 전반적인 성능을 향상시키는 방법론을 소개합니다.

- **Performance Highlights**: MER2024-OV 챌린지에서 제안된 방법은 기존 모델을 초월하는 성능을 보였으며, 감정 인식 작업에서 개선된 정밀도와 재현율을 입증했습니다. 다양한 모달 정보를 통합하여 감정 예측의 정확성을 크게 향상시켰습니다.



### Exploring Scene Coherence for Semi-Supervised 3D Semantic Segmentation (https://arxiv.org/abs/2408.11280)
- **What's New**: 이번 논문에서는 3D 장면 이해를 위한 반지도학습(Semi-supervised learning) 기법을 제안합니다. 특히, CoScene이라고 불리는 방법은 장면의 일관성을 확보하고, 레이블이 없는 장면과 레이블이 있는 장면 간의 정보 전이를 효율적으로 수행합니다.

- **Technical Details**: CoScene은 포인트 클라우드의 비구조적 및 비정렬적 특성에서 영감을 받아 포인트 삭제(point erasure) 전략을 도입하여 레이블이 없는 포인트를 제거함으로써 장면 내 일관성을 보장합니다. 또한, 패치 기반 데이터 증강(patch-based data augmentation)을 통해 레이블이 있는 장면과 없는 장면 간의 상관관계를 모델링하여 학습을 강화합니다. 그래프 기반 구조를 통해 여러 장면에서 지역 및 인스턴스 수준의 정보를 결합합니다.

- **Performance Highlights**: SemanticKITTI 및 nuScenes 데이터셋에서 CoScene이 기존의 방법들보다 4% 높은 mIoU를 기록하며, 다양한 벤치마크 테스트에서 우수한 성능을 보여주었습니다. 5% 레이블을 가진 장면 설정에서 최첨단 방법을 초과 달성하였습니다.



### The Key of Parameter Skew in Federated Learning (https://arxiv.org/abs/2408.11278)
- **What's New**: 이 논문은 Federated Learning (FL)에서 발생하는 parameter skew 현상을 다루며, 이를 통해 모델 성능 향상을 위한 새로운 집계 전략인 FedSA를 제안합니다.

- **Technical Details**: FL은 다양한 클라이언트 간의 데이터 사생활을 보장하면서 분산된 훈련을 가능하게 합니다. 하지만 데이터가 heterogeneous할 경우 local 모델의 parameter 분포가 skew해질 수 있으며, 이는 글로벌 모델의 정확도에 직접적인 영향을 미칩니다. FedSA는 parameters를 고변동성과 저변동성 그룹으로 나누고, 고변동성 parameter에 대해 Micro-Classes (MIC)와 Macro-Classes (MAC)를 도입하여 집계합니다.

- **Performance Highlights**: FedSA는 CIFAR-10/100 및 Tiny-ImageNet 데이터셋에서 8개의 최첨단 알고리즘보다 약 4.7% 높은 테스트 정확도를 기록하며, 뛰어난 성능을 입증했습니다.



### On Missing Scores in Evolving Multibiometric Systems (https://arxiv.org/abs/2408.11271)
Comments:
          2022 26th International Conference on Pattern Recognition (ICPR)

- **What's New**: 이번 연구는 생체 인식 시스템에서 결측 점수 데이터의 영향을 다룬 최초의 연구로, 특히 90%의 결측 데이터를 고려한 상황에서 다양한 score imputation 방법을 적용했음을 보여줍니다.

- **Technical Details**: 연구에서는 K 근접 이웃(K nearest neighbors)을 활용한 반복적 imputation 기법이 인식 정확도를 향상시키며, verificación (verification) 및 identificación (identification) 작업 모두에서 우수한 결과를 보였음을 입증하였습니다. 특히, 다양한 imputation 방법과 단순 합산 융합(simple sum fusion)을 통해 인식 정확도를 개선할 수 있습니다.

- **Performance Highlights**: 실험 결과, 결측 데이터를 impute한 후의 융합 결과가 imputation 없이 수행한 융합보다 성능이 우수하며, 특히 KNN 기법을 이용한 iterative imputation 성능이 가장 높음을 확인했습니다.



### Automatic Image Annotation (AIA) of AlmondNet-20 Method for Almond Detection by Improved CNN-based Mod (https://arxiv.org/abs/2408.11253)
- **What's New**: 이 논문은 경쟁 치열한 견과 시장에서 프리미엄 농산물에 대한 전세계적인 수요 증가에 대응하여 아몬드 및 그 껍질의 등급 판별 과정을 향상시키기 위한 혁신적인 방법론을 소개합니다.

- **Technical Details**: 최신 Deep Convolutional Neural Networks (CNNs)을 활용하여 AlmondNet-20 아키텍처를 기반으로 하며, 20개의 레이어로 구성된 CNN 모델을 통해 99% 이상의 정확도를 달성했습니다. 1000 epochs에 걸쳐 면밀히 학습한 결과, 손실 함수는 0.0567에 불과했습니다. 데이터 증강(data augmentation) 기법을 사용하여 아몬드와 껍질을 구별하는 강도를 높였습니다.

- **Performance Highlights**: 엄격한 테스트 데이터셋을 통한 평가 결과, 아몬드 탐지에서 완벽한 precision, recall, F1-score 메트릭스를 보여주며, 이 고급 분류 시스템은 산업 전문가와 비전문가 모두에게 유용한 이점을 제공합니다.



### Irregularity Inspection using Neural Radiance Field (https://arxiv.org/abs/2408.11251)
- **What's New**: 이 논문에서는 대규모 산업 기계의 결함 탐지를 위한 Neural Radiance Fields (NeRF) 기반의 자동화된 시스템을 제안합니다. 특히 드론을 활용하여 3D 쌍둥이 모델을 비교함으로써 결함을 보다 효율적이고 객관적으로 탐지할 수 있는 방식을 개발했습니다.

- **Technical Details**: 제안된 시스템은 UAV (Unmanned Aerial Vehicle)를 통해 이미지를 수집하고, Neural Radiance Fields (NeRF) 모델을 사용하여 두 개의 3D 모델을 생성한 후, Iterative Closest Point (ICP) 알고리즘으로 자동 정렬하여 차이점을 분석합니다. 이 과정은 다양한 조명 조건에서도 높은 정확도로 결함을 탐지할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험에서는 LEGO 블록과 실험실 의자를 사용하여 결함 탐지의 효과를 입증했습니다. 초기 데이터 샘플로부터 모델을 구축하고, NeRF를 이용하여 생성된 모델 간의 비파라미터 비교를 통해 명확한 결과를 도출했습니다. 이 시스템은 전통적인 검사의 비효율성과 주관성을 극복하여 신속하고 정확한 결함 탐지를 가능하게 합니다.



### CNN-based Labelled Crack Detection for Image Annotation (https://arxiv.org/abs/2408.11250)
- **What's New**: 본 연구는 Additive Manufacturing (AM) 표면에서 균열 결함(crack defects)을 검출하기 위해 기존의 인간 검사 방식을 대체하는 새로운 비전 기반 접근법을 제안합니다. 이 방법은 깊은 Convolutional Neural Networks (CNNs)를 활용하여 균열을 효과적으로 탐지합니다.

- **Technical Details**: 전통적인 이미지 처리 기법(Imagen Processing Techniques, IPTs)은 다양하고 복잡한 실제 환경과 다양한 균열 유형에 대응하는 데 어려움이 있으나, 본 연구에서는 CNNs를 사용하여 광범위한 특징 추출(extract feature)을 불필요하게 하여 이러한 어려움을 극복합니다. CNN 훈련을 위한 주석(annotation)은 추가적인 IPTs 없이 LabelImg를 통해 이루어졌습니다.

- **Performance Highlights**: 훈련된 CNN은 OpenCV 전처리(preprocessing) 기법을 통해 14,982개의 주석 이미지에서 99.54%의 뛰어난 정확도를 달성했습니다. 평가 지표는 96% 이상의 정밀도(precision), 98%의 재현율(recall), 97%의 F1-score를 기록하여 전체 과정의 정밀도와 효율성을 강조합니다.



### CooPre: Cooperative Pretraining for V2X Cooperative Perception (https://arxiv.org/abs/2408.11241)
- **What's New**: 본 연구에서는 기존의 V2X(차량-모든 것) 협력 인식 방법의 한계를 극복하기 위한 자가 지도 학습(self-supervised learning) 방법을 제안합니다. 이 방법은 라벨이 없는 3D V2X 데이터를 활용하여 인식 성능을 향상시키는 방식입니다. CooPre라는 새로운 협력 사전 훈련(pretraining) 프레임워크를 도입하여 기존의 전처리 방법을 넘어 다양한 협력 시나리오에 맞춘 모델 개발에 기여하고자 하였습니다.

- **Technical Details**: CooPre 프레임워크는 LiDAR 점군(LiDAR point clouds) 데이터를 재구성하는 새로운 프록시(proxy) 작업을 설계하였습니다. 이를 통해 다양한 센서 구성에서의 정보를 보상하고, BEV(새카만 보기) 공간에서의 3D 특징에 집중할 수 있도록하는 마스킹 전략을 개발하였습니다. 이러한 접근 방식은 기존 협력 인식 백본(backbone)과 호환됩니다.

- **Performance Highlights**: CooPre 방법은 V2X-Real, V2V4Real 및 OPV2V 데이터셋에서의 광범위한 실험을 통해 모든 V2X 설정에서 성능 향상을 보였습니다. 또한, 이 프레임워크는 교차 도메인 전이 가능성(cross-domain transferability), 데이터 효율성(data efficiency), 도전 과제에서의 강인성(robustness) 측면에서도 개선점을 보여주었습니다.



### Unified Deep Learning Model for Global Prediction of Aboveground Biomass, Canopy Height and Cover from High-Resolution, Multi-Sensor Satellite Imagery (https://arxiv.org/abs/2408.11234)
- **What's New**: 본 연구에서는 다중 센서 및 멀티스펙트럼 이미지를 활용하여 10미터 해상도로 공중 생물량 밀도(AGBD), 캐노피 높이(CH), 캐노피 커버(CC)와 이들의 불확실성을 동시에 예측하는 딥 러닝 기반 모델을 제안합니다. 이 모델은 2016년부터 2023년까지 전 세계 샘플 데이터를 활용해 훈련되었습니다.

- **Technical Details**: 이 모델은 256x256 픽셀 크기의 이미지 타일을 사용하며, 공중 생물량 밀도와 캐노피 관련 변수들을 동시에 예측합니다. 이를 위해 CNN(Convolutional Neural Network)을 사용하고, 훈련 과정에서는 희소한 실제 데이터의 분포를 극복하는 새로운 기술이 도입되었습니다. 기초 모델은 원거리 관측 데이터를 통해 만들어지며, 다양한 생태계에서의 훈련 샘플을 통해 전 세계적으로 유효한 예측을 제공합니다.

- **Performance Highlights**: 모델은 AGBD에 대해 평균 절대 오차 26.1 Mg/ha, CH에 대해 3.7m, CC에 대해 9.9%의 정확도를 보이며, 이는 이전 연구 결과에 비해 유의미한 개선을 보여줍니다. 또한 검증에 사용된 독립적인 지상 측정 데이터와 높은 상관관계를 보였으며, 모델의 다중 헤드 아키텍처 덕분에 다른 GEDI 변수로의 전이 가능성도 원활하게 이루어집니다.



### On the Potential of Open-Vocabulary Models for Object Detection in Unusual Street Scenes (https://arxiv.org/abs/2408.11221)
- **What's New**: 이 연구는 최신 개방 어휘(open-vocabulary) 객체 탐지기들이 거리 장면에서 이례적인 객체들을 얼마나 잘 탐지할 수 있는지를 조사합니다. 특히, 우리는 OoDIS 벤치마크를 통해 세 가지 도전적인 데이터셋에서의 성능을 평가했습니다.

- **Technical Details**: 이 연구에서는 Grounding DINO, YOLO-World, Omdet, MDETR와 같은 최첨단 개방 어휘 모델을 활용하여 OOD 객체 탐지에 대한 벤치마킹을 수행했습니다. 실험은 이미지의 개별 객체에 대한 명확한 프롬프트 대신, 전체 데이터셋에 대해 일반적인 텍스트 프롬프트를 이용하여 적용되었습니다.

- **Performance Highlights**: Grounding DINO는 RoadObstacle21과 LostAndFound 데이터셋에서 각각 48.3%와 25.4%의 AP(average precision)를 기록하며 최고의 성능을 보였습니다. YOLO-World는 RoadAnomaly21에서 21.2%의 AP를 달성했습니다.



### Revisiting Min-Max Optimization Problem in Adversarial Training (https://arxiv.org/abs/2408.11218)
- **What's New**: 본 논문에서는 adversarial 공격에 대한 robust deep neural networks를 구축하기 위한 새로운 방법을 제안합니다. 이는 기존의 saddle point optimization 문제를 reformulate하여, 여러 공격에 대한 구체적인 보안 보장을 제공합니다.

- **Technical Details**: 제안된 방법은 adversarial training을 다루며, loss function의 gradient를 활용하지 않고 probabilistic 관점에서 inner maximization 문제를 해결합니다. 구체적으로는 allowed perturbation에 대한 prior distribution을 고려하며, 각 샘플에 대해 충분히 많은 attacked 버전을 생성합니다. 기존 loss function은 e^{\lambda L(.)}로 대체되어, k attacked 샘플의 총 loss 값이 최대 loss 값과 비례하게 보장됩니다.

- **Performance Highlights**: 제안된 모델들은 이전의 방법들과 비교하여 향상된 성능을 보여주며, adversarial perturbation에 대한 robustness가 명확히 개선되었습니다. 논문 내에서 다양한 실험을 통해 이 결과가 입증되었습니다.



### A Short Review and Evaluation of SAM2's Performance in 3D CT Image Segmentation (https://arxiv.org/abs/2408.11210)
- **What's New**: Segment Anything 2 (SAM2)의 성능이 3D 의료 이미지 분할에 대해 평가되고 있지만, 서로 다른 연구들이 상이한 평가 프로토콜을 사용하여 일관되지 않은 결과를 도출하고 있습니다. 이 연구는 SAM2의 제로샷(zero-shot) 평가 프로토콜을 재현하고, 3D CT 데이터셋을 사용하여 결과를 제공합니다.

- **Technical Details**: SAM2는 2D 이미지 인코더를 사용하고 이전 프레임의 정보를 메모리 뱅크에 저장합니다. 연구는 SAM2가 반복적인 클릭(prompt) 기반의 평가 프로토콜을 사용하며, 최대 8회까지 반복하여 분할 작업을 수행하는 방법을 제안합니다. 하지만, 평균적으로 타겟 대조군과의 큰 성능 차이를 보이며, 특히 큰 장기로 복잡한 형태를 가진 장기의 경우 성능이 떨어지는 경향이 있습니다.

- **Performance Highlights**: 연구 결과, SAM2는 작은 단일 연결 객체인 신장이나 대동맥에서는 합리적인 성능을 보였으나, 대다수의 장기에서는 최신 3D 분할 방법에 비해 여전히 뒤처지는 것으로 나타났습니다. 특히 배경 슬라이스가 제거되지 않으면 성능이 크게 저하되며, SAM2는 영상에서 존재하지 않는 전경 객체를 지나치게 추적하는 경향이 있습니다.



### PooDLe: Pooled and dense self-supervised learning from naturalistic videos (https://arxiv.org/abs/2408.11208)
Comments:
          Project page: this https URL

- **What's New**: 이번 논문에서는 고해상도 자연 비디오에서 효과적인 이미지 표현을 학습하기 위한 새로운 접근법인 PooDLe(구역 밀도 및 풀링 학습)를 제안합니다. 이 방법은 광학 흐름 왜곡에 대한 동등성을 강화하는 밀집( dense ) SSL 목표와 풀링된 표현 결합을 통해 높은 성능을 발휘합니다.

- **Technical Details**: PooDLe는 광학 흐름 왜곡 기반의 밀집 SSL 목표와 작은 내용 정렬 뷰에 적용된 풀링 목표를 최적화합니다. 새롭게 소개된 경량 공간 디코더 모듈(SDM)은 고수준 의미 표현을 보존하여 작은 객체의 의미를 포착하는 데 기여합니다. 이 접근은 BDD100K 및 Walking Tours 데이터셋에서 실험적으로 검증되었습니다.

- **Performance Highlights**: PooDLe는 다운스트림 의미 분할 및 객체 탐지 기준에서 최첨단 성능을 달성하며, 특히 작은 객체 인식에서 주목할 만한 성과를 보입니다. 다양한 크롭 영역과 입력 해상도에서 강력한 성능을 유지하는 동시에, 이전 밀집 SSL 방법보다 우수한 결과를 냅니다.



### Quantum Inverse Contextual Vision Transformers (Q-ICVT): A New Frontier in 3D Object Detection for AVs (https://arxiv.org/abs/2408.11207)
Comments:
          The paper has been accepted as a short paper at CIKM '24

- **What's New**: 이번 논문에서는 지능형 자율주행차량(Auto Vehicles) 기술을 위한 새로운 접근법으로 Quantum Inverse Contextual Vision Transformers (Q-ICVT)라는 이중 단계 융합 프로세스를 제안합니다. 이는 LiDAR와 카메라 데이터를 결합하여 더 나은 성능을 발휘하도록 설계되었습니다.

- **Technical Details**: Q-ICVT는 얕은 컴퓨팅(adiabatic computing) 개념을 활용하여 Global Adiabatic Transformer (GAT)를 도입하고, 이 구조를 통해 스파스 LiDAR 피처를 밀집 이미지를 통해 통합합니다. 또한, Sparse Expert of Local Fusion (SELF) 모듈을 사용하여 LiDAR 데이터 및 카메라 이미지의 지역적 융합을 실현합니다.

- **Performance Highlights**: 실험 결과 Q-ICVT는 Waymo 데이터셋에서 L2 난이도를 기준으로 82.54의 mAPH(Mean Average Precision) 성능을 기록하여 최신 융합 방법보다 1.88% 향상되었습니다. 이 결과는 Q-ICVT의 효과를 강조하며, GAT 및 SELF 모듈의 중요성에 대한 분석도 포함되었습니다.



### Robust Long-Range Perception Against Sensor Misalignment in Autonomous Vehicles (https://arxiv.org/abs/2408.11196)
- **What's New**: 본 논문에서는 다양한 센서 모달리티 간의 미스얼라인먼트(misalignment)를 감지하고 보정하는 효율적인 다중 과제 학습(multi-task learning) 접근 방식을 제안합니다. 이 방법은 특히 장거리 지각(long-range perception)에서 견고성을 유지하는데 강점을 가지고 있습니다.

- **Technical Details**: 제안된 시스템은 미스얼라인먼트 모니터링 작업과 3D 탐지 작업을 통합하여, 각 센서의 중심 간 위치 변화를 감지하고 이에 따른 불확실성을 예측합니다. 이를 통해 보정된 입력 데이터를 사용하여 탐지 성능을 향상시킵니다. 모델은 신경망(outputs)에서 불확실성을 정량화하는 aleatoric uncertainty를 활용하여 예측된 변위 값에 대해 잘 정교화된 신뢰도 추정을 제공합니다.

- **Performance Highlights**: 제안된 방법은 미스얼라인먼트를 포함한 데이터 정정 과정을 통해 장거리 탐지 성능을 높일 수 있으며, 정확한 미스얼라인먼트 예측 또한 가능하다는 점이 특징입니다. 이로 인해 자율주행 차량의 안전성과 성능을 동시에 향상시키는 데 기여할 수 있습니다.



### Compress Guidance in Conditional Diffusion Sampling (https://arxiv.org/abs/2408.11194)
Comments:
          10 pages, 5 figures, Computer Vision and Machine Learning

- **What's New**: 이 연구는 전체 샘플링 과정에서의 가이던스(guideance) 적용 문제를 분석하고, 여러 타임스텝(timestep)에서 가이던스를 줄이거나 제외하는 것이 모델 적합 문제(model-fitting issue)를 완화하는 데 효과적임을 보여줍니다. 이 방법은 초기 과정에서 가이던스를 밀집하여 적용함으로써 이미지 품질과 다양성을 크게 향상시키고, 필요한 가이던스 타임스텝을 거의 40% 줄입니다.

- **Technical Details**: 본 논문에서는 Diffusion Models에 있어서 가이던스의 주요 유형인 classifier-free guidance와 classifier guidance의 계산 비용 문제를 다룹니다. 기존의 방법들은 이미지의 초기 형상 생성에 많은 가이던스를 요구하여 불필요한 계산을 발생시킵니다. 연구진은 'Compress Guidance (CompG)'라는 방법을 제안하여, 필요한 가이던스 타임스텝 수를 줄이며 샘플 품질을 개선하고 실행 시간을 단축시킵니다.

- **Performance Highlights**: 제안된 Compress Guidance 방법은 다양한 데이터셋과 생성 작업에서 벤치마크 검증을 통해 기본 모델(baseline models)보다 더 나은 이미지 품질을 나타내며, 학습 효율성을 크게 향상시킵니다. 연구 결과는 다른 분류기 여부와 관계없이 일반화된 생성 작업에서 유효성을 입증하였습니다.



### Statistical Challenges with Dataset Construction: Why You Will Never Have Enough Images (https://arxiv.org/abs/2408.11160)
Comments:
          13 pages

- **What's New**: 이 논문은 컴퓨터 비전 모델의 성능 평가에 있어, 현재의 평가 방법론이 실생활에서의 신뢰성을 보장하지 못한다는 주장을 통해 새로운 인사이트를 제공합니다. 저자들은 대표성 있는 이미지 데이터셋을 선택하는 것이 불가능하다는 점과, 비대표적 데이터셋으로 계산된 성능 통계가 신뢰할 수 없음을 강조합니다.

- **Technical Details**: 저자들은 DNN(deep neural networks)이 자율주행차 및 의료 이미지 진단과 같은 안전-critical(안전 중시) 영역에서 사용됨에 따라, 모델 성능 평가를 위한 엄격한 방법론의 개발이 필수적임을 주장합니다. 현재의 평가 방식은 withheld test set(보류된 테스트 세트)에 의존하고 있으며, 이러한 방식이 갖는 한계를 논의합니다.

- **Performance Highlights**: 상태-of-the-art 모델들은 약 99%의 top-5 accuracy를 기록하지만, 이는 실질적인 응용에서 기준이 되는 낮은 에러율에 도달하지 못할 수 있습니다. 예를 들어, 자율주행차에서 이러한 모델의 사용은 심각한 안전 위험을 초래할 수 있으며, 이를 해결하기 위해서는 성능 외에도 실제 안전 목표와의 비교가 필요하다는 점을 강조합니다.



### An Interpretable Deep Learning Approach for Morphological Script Type Analysis (https://arxiv.org/abs/2408.11150)
Comments:
          Accepted at ICDAR 2024 Workshop on Computational Paleography (IWCP, 31 August - Athens, Greece)

- **What's New**: 이 논문은 중세 필체의 유형 정의 및 분류 기준 설정을 위한 새로운 접근 방식인 해석 가능한 딥러닝(Deep Learning) 기반 분석을 제안합니다. 기존의 전통적인 방법론의 한계를 극복하기 위한 체계적이고 객관적인 분석 방법을 제시합니다.

- **Technical Details**: 딥 인스턴스 세그멘테이션(Deep Instance Segmentation) 방법을 적용하여 문자 형태를 대표하는 비교 가능한 문자 프로토타입을 학습합니다. 이 방법은 문자 비교 및 분석을 위한 정성적(qualitative) 및 정량적(quantitative) 도구를 제공합니다.

- **Performance Highlights**: 이 연구는 A. Derolez에 의해 형식화된 두 가지 하위 유형인 Northern Textualis와 Southern Textualis를 포함하는 Textualis Formata 스크립트 유형에 이 방법을 적용하여 그 유용성을 입증합니다.



### ISLES 2024: The first longitudinal multimodal multi-center real-world dataset in (sub-)acute strok (https://arxiv.org/abs/2408.11142)
- **What's New**: 이번 연구에서는 뇌졸중 (Stroke)의 이미지를 사용하여 의미 있고 재현 가능한 뇌 기능 모델을 추출할 수 있는 머신러닝 (machine learning) 알고리즘을 개발하기 위한 최초의 포괄적인 종단적 데이터셋 (longitudinal dataset)을 제시합니다.

- **Technical Details**: 이 데이터셋은 급성 CT 영상 (acute CT imaging), 혈관 조영 (angiography), 관류 (perfusion) 정보, 2-9일 후의 MRI, 그리고 3개월까지의 급성 및 종단적 임상 데이터 (clinical data)를 포함하고 있습니다. 훈련 데이터셋은 n=150, 테스트 데이터셋은 n=100개의 스캔으로 구성되어 있습니다.

- **Performance Highlights**: 이 데이터셋은 2024년 Ischemic Stroke Lesion Segmentation (ISLES) 챌린지의 일환으로 제공되며, 급성 및 준급성 허혈성 뇌졸중 병변 분할 (lesion segmentation) 방법의 기준을 설정하는 데 기여할 것입니다.



### Binocular Model: A deep learning solution for online melt pool temperature analysis using dual-wavelength Imaging Pyrometry (https://arxiv.org/abs/2408.11126)
- **What's New**: 이 연구에서는 금속 적층 제조(AM) 과정에서 용융 풀(Melt Pool, MP) 온도를 실시간으로 모니터링 하는 AI 기반의 새로운 접근법을 제안합니다. 고전적인 모니터링 방법의 단점을 극복하기 위해 쌍안경(Binocular) 모형을 도입하여, 데이터에서 인사이트로의 전환을 가속화하고, 수작업 데이터 처리 의존도를 줄입니다.

- **Technical Details**: 연구에서는 이중 파장(real-time dual-wavelength)으로 수집된 데이터와 그에 매칭되는 온도 맵을 활용하여 AI 기반의 딥러닝 모델을 개발했습니다. Binocular 모델은 L-PBF(Laser Powder Bed Fusion)에서 MP 온도를 정확히 분석할 수 있도록 설계되었으며, 초당 최대 750 프레임의 처리 속도를 자랑합니다. 이를 통해 전통적인 방법보다 약 1000배 향상된 효율성을 제공합니다.

- **Performance Highlights**: Binocular 모델은 0.95의 R-제곱 점수를 통해 높은 정확도의 온도 추정을 달성했습니다. 이 모델은 금속 AM에서 MP 온도 모니터링의 실시간적 문제를 해결하고, 효율성과 정밀성을 결합한 혁신적인 접근 방식을 제시하여 금속 AM 분야의 발전에 기여하고 있습니다.



### GSLoc: Efficient Camera Pose Refinement via 3D Gaussian Splatting (https://arxiv.org/abs/2408.11085)
Comments:
          The project page is available at https://gsloc.active.vision

- **What's New**: 본 논문에서는 3D Gaussian Splatting (3DGS)을 이용한 장면 표현을 활용해 새로운 테스트 시간 카메라 자세 정제 프레임워크인 GSLoc을 제안합니다. 이 프레임워크는 최신 절대 자세 회귀 및 장면 좌표 회귀 방법의 로컬라이제이션 정확도를 향상시킵니다.

- **Technical Details**: GSLoc은 RGB 이미지에서 직접 작동하고, 3D 비전 기반 모델 MASt3R를 활용하여 정밀한 2D 매칭을 수행합니다. 추가적으로 노출 적응 모듈을 통합하여 어려운 외부 환경에서도 모델의 강건성을 높입니다. GSLoc은 단일 RGB 쿼리 및 대략적인 초기 자세 추정값을 바탕으로 효율적인 자세 정제를 가능하게 합니다.

- **Performance Highlights**: 제안된 방법은 실내 및 실외 시각 로컬라이제이션 벤치마크에서 NeRF 기반 최적화 방법들을 초월하며, 두 개의 실내 데이터셋에서 최첨단 정확도를 달성했습니다.



### ACE: A Cross-Platform Visual-Exoskeletons System for Low-Cost Dexterous Teleoperation (https://arxiv.org/abs/2408.11805)
Comments:
          Webpage: this https URL

- **What's New**: 최근 수집된 대규모 로봇 데이터와 원격 조정 시스템을 활용하여, 다양한 로봇 플랫폼 간의 효율적인 원격 조종 시스템인 ACE를 개발하였습니다. ACE는 정밀한 실시간 손 및 손목 자세 추적이 가능한 시각적 외골격 시스템으로, 기존의 시스템에 비해 비용 효율적이고 사용자 친화적입니다.

- **Technical Details**: ACE 시스템은 저비용 카메라와 3D 손 자세 추정 방식을 통해 손 자세를 추적합니다. 원격 조종 중 외골격을 통해 손과 손목의 실시간 자세를 정확하게 캡처할 수 있도록 설계되었습니다. 이 시스템은 여러 로봇 하드웨어와 높은 정밀도의 원격 조종을 지원하며, 인체 모양의 로봇 손, 팔-손 및 팔-그리퍼, 사족 보행 로봇에도 적용할 수 있습니다.

- **Performance Highlights**: ACE 시스템은 저비용에 accurate한 손 및 말단 조절이 가능하여 다양한 원격 조작 작업에서 정교한 조작을 수행할 수 있습니다. 실험 결과, ACE는 다양한 작업 공간 요구 사항에 신속하게 적응하며, 효과적인 데이터 수집 및 모방 학습을 효율적으로 수행할 수 있음을 보여줍니다.



### DreamFactory: Pioneering Multi-Scene Long Video Generation with a Multi-Agent Framework (https://arxiv.org/abs/2408.11788)
Comments:
          13 pages, 8 figures

- **What's New**: 이 논문에서는 긴 비디오 생성을 위한 새로운 모델인 DreamFactory를 소개합니다. 기존의 비디오 생성 모델은 짧은 클립에서는 뛰어난 성과를 보였으나, 다중 장면이 포함된 긴 비디오에서는 어려움을 겪었습니다. DreamFactory는 다중 에이전트 협업 원칙과 Key Frames Iteration Design Method를 활용하여 이러한 문제를 해결합니다.

- **Technical Details**: DreamFactory는 LLM(대형 언어 모델)을 기반으로 하며, 제작 과정에서 각 LLM이 감독, 미술 감독, 각본가 및 아티스트와 같은 역할을 맡아 협력합니다. 이 프레임워크는 시나리오 작성, 스토리보드 생성, 캐릭터 디자인, 키프레임 개발 등을 포함하여 비디오 생성을 자동화합니다. 본 모델은 비디오 세그먼트 간의 일관성을 확보하기 위해 특정 키프레임 반복 방법을 적용합니다.

- **Performance Highlights**: DreamFactory는 UTF-101 및 HMDB51 데이터셋을 사용하여 평가한 결과, 기존 모델에 비해 상당한 성능 개선이 있음을 보여주었습니다. 특히, 우리 모델이 생성한 긴 비디오는 수작업으로 생성된 비디오보다 평균적인 품질을 초과하는 것으로 평가되었습니다.



### NuSegDG: Integration of Heterogeneous Space and Gaussian Kernel for Domain-Generalized Nuclei Segmentation (https://arxiv.org/abs/2408.11787)
Comments:
          Under Reivew

- **What's New**: 이 논문에서는 모델이 소스 도메인에서 학습한 지식을 통해 보지 못한 도메인에서 일반화되는 능력인 도메인 일반화가 필요한 핵 세분화(nuclei segmentation) 방법을 제안합니다. 특히, NuSegDG라는 프레임워크를 통해 수동 프롬프트 없이도 효과적인 세분화를 가능케 하였습니다.

- **Technical Details**: 제안하는 NuSegDG 프레임워크는 세 가지 모듈로 구성됩니다: Heterogeneous Space Adapter (HS-Adapter), Gaussian-Kernel Prompt Encoder (GKP-Encoder), Two-Stage Mask Decoder (TSM-Decoder). HS-Adapter는 다양한 핵 도메인의 다차원 특징 표현을 학습하고, GKP-Encoder는 단일 포인트를 사용하여 밀도 맵을 생성하여 세분화 예측을 안내합니다. TSM-Decoder는 수동 형태학적(geometrical) 조정 없이 세분화 마스크를 인스턴스 맵으로 변환합니다.

- **Performance Highlights**: NuSegDG는 기존 핵 세분화 방법 및 최신 의료 세분화 모델과 비교하여 사이클 간 세분화(instance segmentation)에서 주목할 만한 성능을 보였으며, 높은 도메인 일반화 능력을 나타냈습니다. 실험 결과, NuSegDG가 다양한 핵 이미지 도메인에서 우수한 성능을 발휘함을 입증합니다.



### JieHua Paintings Style Feature Extracting Model using Stable Diffusion with ControlN (https://arxiv.org/abs/2408.11744)
Comments:
          accepted by ICCSMT 2024

- **What's New**: 이 연구는 Jiehua의 스타일적 특징을 추출하기 위한 새로운 접근 방식을 제안합니다. Fine-tuned Stable Diffusion Model with ControlNet (FSDMC)를 활용하여 아티스트의 Jiehua에서 묘사 기술을 정제하는 방법을 모색했습니다.

- **Technical Details**: FSDMC의 훈련 데이터는 인터넷에서 수집한 오픈소스 Jiehua 아티스트 작품을 기반으로 하여, (원본 이미지, Canny Edge Features, Text Prompt) 형식으로 수작업으로 구성되었습니다. 이 논문에서 확인된 최적의 하이퍼파라미터를 사용한 결과, FSDMC는 다른 주류 스타일 전이 모델인 CycleGAN을 능가하는 성능을 보였습니다.

- **Performance Highlights**: FSDMC는 데이터셋에서 FID 3.27을 달성하였으며, 전문 평가에서도 CycleGAN을 초월하였습니다. 이는 Jiehua 스타일 특징 추출에서 모델의 높은 효율성을 나타낼 뿐만 아니라, 원본 사전 훈련된 의미 정보를 보존하고 있다는 것을 보여줍니다.



### On Learnable Parameters of Optimal and Suboptimal Deep Learning Models (https://arxiv.org/abs/2408.11720)
- **What's New**: 이 논문은 심층 학습 모델의 구조적 및 동작적 측면을 심도 있게 분석하며, 학습 가능한 매개변수(learnable parameters)의 통계 및 분포, 노드 상호작용, 시각화에 주목하고 있습니다.

- **Technical Details**: 우리는 DNN, CNN, ViT와 같은 다양한 심층 학습 모델에 대해 실험을 수행하여 가중치 통계의 변화가 전체 네트워크 성능에 미치는 영향을 조사하였습니다. 실험은 MNIST, Fashion-MNIST, CIFAR-10 데이터셋을 사용하여 이루어졌습니다. 또한, 통계적 분석을 통해 최적 및 비최적 네트워크의 학습 가능한 매개변수의 특성을 규명하였습니다.

- **Performance Highlights**: 성공적인 네트워크는 데이터셋이나 모델에 관계없이 수렴된 가중치 통계와 분포가 유사하다는 사실을 발견했습니다. 반면에 성능이 좋지 않은 네트워크는 가중치가 다양하게 나타났습니다. 이는 DNN, CNN, ViT와 같은 여러 심층 학습 아키텍처에서 공통적으로 발견되는 학습 특성을 보여줍니다.



### FedGS: Federated Gradient Scaling for Heterogeneous Medical Image Segmentation (https://arxiv.org/abs/2408.11701)
Comments:
          10 pages, 2 figures, 1 table, accepted at MICCAI 2024 Workshop on Distributed, Collaborative, & Federated Learning Workshop (DeCaF). This is the submitted manuscript with added link to github repo, funding acknowledgements and author names and affiliations. No further post submission improvements or corrections were integrated. Final version not published yet

- **What's New**: Federated Gradient Scaling (FedGS)라는 새로운 연합 학습 방법을 제안하여 제한된 크기와 가용성을 가진 샘플의 분할 성능을 향상시킵니다.

- **Technical Details**: FedGS는 각 클라이언트가 유지하는 누적 그래디언트 Gtᵏ를 사용하여 어려운 샘플(예: 작은 병변)의 그래디언트를 스케일링하여 성능을 개선합니다. 서버는 클라이언트의 최종 학습 반복에서 누적 그래디언트를 집계하여 전역 모델 파라미터를 업데이트합니다.

- **Performance Highlights**: FedGS는 PolypGen 및 LiTS 데이터셋에서 작은 병변에 대해 FedAvg보다 우수한 성능을 보였으며, 특히 작은 크기의 분할 목표에 대한 성능을 크게 향상시켰습니다.



### LiFCal: Online Light Field Camera Calibration via Bundle Adjustmen (https://arxiv.org/abs/2408.11682)
Comments:
          Accepted to the German Conference on Pattern Recognition (GCPR) 2024

- **What's New**: LiFCal이라는 새로운 기하학적 온라인 보정 파이프라인을 제안합니다. 이 시스템은 MLA 기반의 라이트 필드 카메라를 위해 설계되었으며, 정밀한 보정 목표 없이 움직이는 카메라 시퀀스로부터 모델 매개변수를 정확하게 결정합니다.

- **Technical Details**: LiFCal은 임의의 메트릭 스케일링 제약을 통합하여 라이트 필드 카메라 모델의 내재적 매개변수(intrinsic parameters), 스파스 세트의 3D 좌표, 카메라 포즈(camera poses)를 마이크로 이미지 포인트에서 직접 정의된 단일 번들 조정(bundle adjustment)으로 최적화합니다.

- **Performance Highlights**: LiFCal은 다양한 입력 시퀀스를 사용해 집중된 플레노프틱 카메라를 신뢰성 있게 보정할 수 있으며, 최첨단 방법과 비교하여 내재적 카메라 매개변수를 매우 근접하게 제공합니다. 이 시스템은 또한 목표물이 없는 장면에서 적용할 수 있으며, 완전하고 지속적인 온라인 파이프라인으로 구현됩니다. 최종적으로 깊이 추정(depth estimation) 및 SLAM과 같은 후속 작업에서 얻은 카메라 매개변수의 품질을 입증합니다.



### MCDubber: Multimodal Context-Aware Expressive Video Dubbing (https://arxiv.org/abs/2408.11593)
- **What's New**: 이번 연구에서 제안하는 MCDubber는 기존 AVD 모델의 한계인 멀티모달 문맥 정보의 통합 부족을 해결합니다. MCDubber는 더 긴 문맥 정보에 기반하여 단일 문장 모델링에서 전환하여, 전반적인 문맥 프로소디의 일관성을 보장합니다.

- **Technical Details**: MCDubber는 세 가지 주요 구성 요소로 이루어져 있습니다: (1) Context Duration Aligner (CDA)는 텍스트와 입술 프레임 간의 문맥 인식을 기반으로 한 정렬을 학습합니다. (2) Context Prosody Predictor (CPP)는 글로벌 문맥 비주얼 시퀀스를 통해 문맥 인식 에너지와 음높이를 예측합니다. (3) Context Acoustic Decoder (CAD)는 인접한 문장의 ground-truth mel-spectrogram을 활용하여 글로벌 문맥 mel-spectrogram을 예측합니다.

- **Performance Highlights**: Chem 벤치마크 데이터셋을 통한 실험에서는 MCDubber가 모든 기존 모델에 비해 더 높은 더빙 표현력을 보이는 것으로 나타났습니다.



### RaNDT SLAM: Radar SLAM Based on Intensity-Augmented Normal Distributions Transform (https://arxiv.org/abs/2408.11576)
Comments:
          This work was accepted by the IEEE/RSJ International Conference on Intelligent Robots and Systems, 2024

- **What's New**: 본 연구에서는 라이다와 같은 전통적인 센서 대신 FMCW 레이더를 기반으로 하는 새로운 SLAM 프레임워크 'RaNDT SLAM'을 도입합니다. 이 프레임워크는 레이더의 신호 강도를 추가적으로 고려하여 실내 및 실외 환경 모두에서 로봇 경로를 정확하게 추적할 수 있는 기능을 제공합니다.

- **Technical Details**: RaNDT SLAM은 Normal Distributions Transform(NDT)을 기반으로 하며 레이더 강도를 활용한 데이터 융합 기법을 사용합니다. 본 연구에서는 IMU(관성 측정 장치) 데이터를 융합하여 모션 추정을 수행하며, 포즈 그래프와 루프 클로저 감지기를 통해 글로벌 일관성을 유지합니다. 데이터는 새로운 벤치마크 데이터셋과 Oxford Radar RobotCar 데이터셋에서 평가되었습니다.

- **Performance Highlights**: RaNDT SLAM은 다양한 실내 및 실외 환경에서 실시간 성능을 입증하였으며, 새로운 데이터셋을 통해 레이더 SLAM의 새로운 기준점을 제시하였습니다. 이 연구는 구내 구조물 및 재난 환경에서의 로봇 탐색을 지원하는 데 중요한 성과를 달성하였습니다.



### Finite element-based space-time total variation-type regularization of the inverse problem in electrocardiographic imaging (https://arxiv.org/abs/2408.11573)
- **What's New**: 이번 연구에서는 유한 요소(Finite Element) 설정에서 공간-시간(Spatio-Temporal) 전체 변동 유형(total variation-type) 정규화를 기반으로 한 신규 접근법을 제시하여 심장 외피 전위(epicardial potential)를 재구성하는 방법을 다룹니다.

- **Technical Details**: 제안하는 방법은 첫 번째 차수의 프라이멀-듀얼(primal-dual) 알고리즘을 사용하여 기본적인 볼록 최적화 문제를 해결합니다. 이는 전통적인 Tikhonov 정규화보다도 더 날카로운 전위 변화에 적합하여 심장 전기 활동의 급격한 전이(transition)를 허용합니다.

- **Performance Highlights**: 이 방법은 2차원 몸체 모델과 3차원 토끼 심장을 통한 수치 실험에서 기존의 최첨단 Tikhonov 정규화 방법들에 비해 우수한 성능을 보이며, 더욱 향상된 재구성을 나타났습니다.



### A Survey of Embodied Learning for Object-Centric Robotic Manipulation (https://arxiv.org/abs/2408.11537)
- **What's New**: 이 논문에서는 물체 중심 로봇 조작을 위한 체화 학습(Embodied Learning)의 최신 발전 상황을 종합적으로 리뷰하고, 세 가지 주요 분야로 분류한다: 1) Embodied Perceptual Learning, 2) Embodied Policy Learning, 3) Embodied Task-Oriented Learning. 또한, 공개 데이터셋, 평가 메트릭, 대표적인 응용 사례, 현재의 도전과제와 향후 연구 방향에 대한 논의를 제공한다.

- **Technical Details**: 체화 학습은 환경과의 물리적 상호작용과 인지적 피드백을 통해 로봇이 학습하는 방법으로, 데이터 기반 머신 러닝과는 대조적이다. 세 가지 유형의 체화 학습 방법은 다음과 같다: 1) Embodied Perceptual Learning은 다양한 데이터 표현을 통해 물체의 자세 및 가능성을 예측, 2) Embodied Policy Learning은 강화 학습 및 모방 학습 기법을 활용하여 로봇의 최적 결정을 생성, 3) Embodied Task-Oriented Learning은 특정 작업 특성에 따라 로봇의 성능을 최적화한다.

- **Performance Highlights**: 이 논문은 최근 발전한 체화 학습 방법론들을 종합하여 물체 중심 로봇 조작의 다양한 하위 분야들을 분석한다. 특히, 이미지 기반, 3D 인식, 촉각 기반 등의 다양한 데이터 표현 방법들을 다루고, 이를 통해 로봇의 조작 정확성을 높이는 데 기여하고자 한다. 더불어, 여러 공개 데이터셋과 평가 메트릭을 제시하며, 현재 직면하고 있는 도전 과제와 미래 연구 방향성에 대한 통찰을 제공한다.



### OAPT: Offset-Aware Partition Transformer for Double JPEG Artifacts Remova (https://arxiv.org/abs/2408.11480)
Comments:
          14 pages, 9 figures. Codes and models are available at this https URL

- **What's New**: 이번 연구에서는 double JPEG 아티팩트 제거를 위한 Offset-Aware Partition Transformer (OAPT)를 제안하며, 이 방법은 compressions의 offset을 효과적으로 활용하여 패턴 클러스터링을 수행합니다.

- **Technical Details**: OAPT는 두 가지 주요 구성 요소로 이루어져 있습니다: compression offset predictor와 image reconstructor입니다. Predictor는 첫 번째와 두 번째 compression 간의 픽셀 offset을 추정하며, reconstructor는 Hybrid Partition Attention Blocks (HPAB)을 기반으로 하여 clustered pattern features를 처리합니다.

- **Performance Highlights**: OAPT는 double JPEG 이미지 복원 작업에서 기존의 최첨단 방법보다 0.16dB 더 높은 성능을 달성하며, 추가적인 계산 비용 없이 HPAB의 패턴 클러스터링 모듈을 다른 transformer 기반의 이미지 복원 방법에 플러그인으로 사용할 수 있습니다.



### DABench: A Benchmark Dataset for Data-Driven Weather Data Assimilation (https://arxiv.org/abs/2408.11438)
Comments:
          37pages, 12 figures, 6 tables

- **What's New**: 딥러닝의 최근 발전에 따라 여러 개의 대형 날씨 모델(Large Weather Models, LWM)이 개발되었습니다. 그러나 이들 모델은 여전히 전통적인 수치 기상 예측(numerical weather prediction, NWP) 시스템이 생성한 분석 필드를 입력으로 사용하고 있으며 완전한 자율 시스템으로 발전하지 못했습니다. 본 연구에서는 DABench라는 벤치마크 데이터셋을 소개하여 데이터를 기반으로 한 기상 예측 시스템의 개발을 촉진하고자 합니다.

- **Technical Details**: DABench는 네 가지 표준 기능을 제공합니다: (1) 관측 시스템 시뮬레이션 실험(observing system simulation experiment, OSSE) 방법을 통해 생성된 희소하고 노이즈가 있는 시뮬레이션 관측(Observations); (2) 초기 필드의 질이 LWMs의 정확성에 미치는 영향을 공정하게 평가할 수 있는 사전 훈련된 날씨 예측 모델; (3) 모델 비교를 위한 표준화된 평가 지표; (4) DA 변환기(DA Transformer, DaT)라는 강력한 기준선입니다.

- **Performance Highlights**: DaT는 4차원 변형(data assimilation, DA) 지식을 통합하여 변환기 모델에서 최첨단 물리적 상태 재구성을 이룬 4DVarNet을 초월했습니다. 연구자들은 DABench를 통해 자신의 모델을 개발하고 안정된 기준선에 대한 성능을 비교할 수 있어 데이터 기반 기상 예측 시스템의 발전에 기여할 것입니다.



### FATE: Focal-modulated Attention Encoder for Temperature Prediction (https://arxiv.org/abs/2408.11336)
- **What's New**: 이 연구는 기후 변화 문제를 해결하기 위해 새로운 FocalNet Transformer 아키텍처 기반의 Focal modulation Attention Encoder (FATE)를 소개합니다. 이 모델은 멀티 텐서 형식으로 작동하며 기후 데이터를 보다 효과적으로 처리할 수 있는 방법을 제시합니다.

- **Technical Details**: FATE 모델은 텐서화된 변조(tensorized modulation) 메커니즘을 활용하여 기후 데이터의 공간적(spatial) 및 시간적(temporal) 뉘앙스를 포착합니다. 이 모델은 온도 예측을 위한 기존의 Transformer 인코더 및 3D CNN, LSTM, ConvLSTM 모델들과 비교하여 우수한 성능을 보입니다. 또한, 40년 간의 데이터로 구성된 Climate Change Parameter dataset (CCPD)도 도입합니다.

- **Performance Highlights**: 실험 결과, 미국, 캐나다, 유럽의 실제 온도 데이터셋에서 현재 최첨단 모델에 비해 각각 12%, 23%, 28%의 정확도 향상을 보여줍니다. 새로 개발된 CCPD 데이터셋 또한 24%의 정확도 향상을 기록하였습니다.



### HMT-UNet: A hybird Mamba-Transformer Vision UNet for Medical Image Segmentation (https://arxiv.org/abs/2408.11289)
Comments:
          arXiv admin note: text overlap with arXiv:2403.09157; text overlap with arXiv:2407.08083 by other authors

- **What's New**: 이 논문에서는 혼합 메커니즘을 이용한 의료 영상 분할을 위한 새로운 U자형 아키텍처 모델인 Hybird Transformer vision Mamba UNet (HTM-UNet)를 제안했습니다. 이는 SSM(State Space Model) 기반의 모델과 Transformer를 결합하여 장거리 의존성을 효율적으로 모델링하는 것을 목표로 합니다.

- **Technical Details**: HTM-UNet는 주로 세 가지 부분으로 나뉘며, 인코더, 디코더, 스킵 연결 구조를 포함합니다. 인코더는 Mamba 비전 및 Mamba 믹서의 합성 모듈로 구성되어 다운샘플링을 수행합니다. 또한, 디코더는 Mamba 믹서, 업샘플링 작업 및 최종 선형 레이어로 구성되며, SSM 기반 모델의 성능을 강조하기 위해 스킵 연결에서는 단순 덧셈을 사용합니다.

- **Performance Highlights**: HTM-UNet는 ISIC17, ISIC18, CVC-300, CVC-ClinicDB, Kvasir, CVC-ColonDB, ETIS-Larib PolypDB와 같은 공용 데이터셋 및 ZD-LCI-GIM 개인 데이터셋에서 포괄적인 실험을 수행했으며, 이 모델이 의료 영상 분할 작업에서 경쟁력이 있음을 보여주었습니다.



### Out-of-Distribution Detection with Attention Head Masking for Multimodal Document Classification (https://arxiv.org/abs/2408.11237)
- **What's New**: 이번 연구에서는 Attention Head Masking (AHM)이라는 새로운 방법론을 제안하여 다중 모달 문서 분류 시스템에서 out-of-distribution (OOD) 데이터 탐지를 개선하였습니다. 우리가 제안한 방법은 기존의 다른 방법들보다 유의미하게 낮은 false positive rate (FPR)을 달성했습니다.

- **Technical Details**: AHM은 transformer 모델의 self-attention 메커니즘을 활용하여 ID와 OOD 데이터를 분리하는 기능을 강화하는 방법입니다. 이 기법은 데이터가 각 클래스 간에 얼마나 유사한지를 반영하여 OOD 탐지 성능을 극대화하도록 설계되었습니다. 또한, FinanceDocs라는 고품질의 금융 문서 AI 데이터셋을 새롭게 소개합니다.

- **Performance Highlights**: 제안된 AHM 방법은 기존의 최신 솔루션보다 월등한 성능을 보이며, 특히 Tobacco3482 및 FinanceDocs 데이터셋을 사용한 실험 결과에서 OOD 탐지에서 AUC (AUROC) 메트릭이 우수한 성과를 달성했습니다. 이로써 AHM은 다중 모달 데이터에서도 효과적으로 일반화되어 신뢰성과 안전성을 높이는데 기여할 수 있음을 보여주었습니다.



### OCTCube: A 3D foundation model for optical coherence tomography that improves cross-dataset, cross-disease, cross-device and cross-modality analysis (https://arxiv.org/abs/2408.11227)
- **What's New**: OCTCube는 기존 2D OCT 이미지 슬라이스 대신 3D OCT 볼륨을 활용하여 훈련된 새로운 3D foundation model입니다.

- **Technical Details**: OCTCube는 26,605개의 3D OCT 볼륨과 1.62백만 개의 2D OCT 이미지를 포함하여 사전 훈련되었습니다. 이 모델은 3D masked autoencoders를 기반으로 하고 FlashAttention을 활용하여 GPU 메모리 사용량을 줄이는 방법으로 개발되었습니다.

- **Performance Highlights**: OCTCube는 8개의 망막 질병을 예측하는 데 있어 2D 모델보다 뛰어난 성능을 보였으며, 특히 데이터 간 전이 작업 및 당뇨병, 고혈압과 같은 전신 질병 예측에서도 우수한 성과를 나타냈습니다.



### CRACKS: Crowdsourcing Resources for Analysis and Categorization of Key Subsurface faults (https://arxiv.org/abs/2408.11185)
- **What's New**: 이 논문은 crowdsourcing을 활용하여 지하 이미징에서 금이 간 곳(faults)을 탐지하고 분할하는 CRACKS 데이터 세트를 제안합니다. 이 데이터 세트는 초보자, 실무자, 전문가 등 다양한 전문성을 갖춘 annotator들이 만든 주석을 포함하고 있습니다.

- **Technical Details**: CRACKS 데이터 세트는 네덜란드 북해 지역의 지하 이미지를 기반으로 하며, Amazon Mechanical Turk를 통해 수집되었습니다. 각각의 주석에는 annotator의 신뢰도 수준도 포함되어 있습니다. 데이터 세트는 7636개의 fault에 대한 주석을 제공합니다.

- **Performance Highlights**: 이 연구는 다양한 전문성을 가진 annotator들로부터 수집한 노이즈가 있는 주석이 전문가 주석을 모델링하는 데 유용하다는 것을 보여주며, 이는 machine learning 모델의 일반화 가능성 향상에 기여할 수 있습니다.



### Target-Oriented Object Grasping via Multimodal Human Guidanc (https://arxiv.org/abs/2408.11138)
Comments:
          Accepted by ECCV 2024 Workshop on Assistive Computer Vision and Robotics (ACVR 2024)

- **What's New**: 본 논문에서는 인간과 로봇 간의 상호작용을 개선하기 위한 새로운 Target-Oriented Grasp Network (TOGNet)을 제안합니다. TOGNet은 로봇이 인간의 다양한 지시를 더 효율적으로 이해하고 반응할 수 있도록 설계되었습니다.

- **Technical Details**: TOGNet은 6-DoF (6 Degrees of Freedom) 그립 포즈를 탐지하는 데 초점을 맞추며, 객체에 대한 의존성이 없는 지역 패치를 활용하여 효율적으로 그립을 예측합니다. 이 시스템은 Multimodal Guidance Module (MGM)과 TOGNet 두 가지 주요 모듈로 구성되어 있습니다.

- **Performance Highlights**: 이 시스템은 복잡한 장면에서 50회의 시뮬레이션 실험을 통해 약 13.7%의 성공률 개선을 달성하였으며, 실제 실험에서도 다양한 목표 지향적 그립 시나리오에서 우수한 성능을 입증했습니다.



### Solving Oscillator ODEs via Soft-constrained Physics-informed Neural Network with Small Data (https://arxiv.org/abs/2408.11077)
Comments:
          17 pages, 7 figures, 2 tables, etc

- **What's New**: 본 논문은 물리 정보에 기반한 신경망(Physics-Informed Neural Network, PINN)과 전통적인 신경망(Conventional Neural Network), 그리고 수치 이산화 방법(Numerical Discretization Methods)을 비교하여 미분 방정식(Differential Equations)을 해결하는 새로운 방법을 제시합니다. 이 연구는 PINN의 수학적 프레임워크와 계산 흐름을 형식화했으며, 실험을 통해 그 정확성과 효율성을 입증했습니다. PINN은 라벨이 있는 데이터의 필요성을 크게 줄이는 효능을 가지고 있습니다.

- **Technical Details**: 연구에서 SOFT-제약(PINN) 방법은 ODE(Ordinary Differential Equations)와 PDE(Partial Differential Equations) 문제를 해결하기 위해 메커니즘을 정립합니다. 이 방법은 DeepXDE를 기반으로 하여 경량 코드와 효율적인 훈련을 지원하며, 다양한 플랫폼에서 유연하게 적용됩니다. PINN은 강한 비선형성을 가진 ODE에 대해서도 적절한 훈련 포인트 수의 증가만으로 우수한 성능을 보여 줍니다.

- **Performance Highlights**: PINN은 훈련 데이터가 부족해도 일반화 능력이 뛰어나며, 노이즈가 있는 데이터에 강한 저항력을 보이는 것으로 나타났습니다. 또한, 실험적으로 PINN의 성능은 물리 법칙을 명시적으로 인코딩함으로써 ODE 문제의 솔루션 성능을 개선함으로써 해석 가능성을 높입니다.



### DiffZOO: A Purely Query-Based Black-Box Attack for Red-teaming Text-to-Image Generative Model via Zeroth Order Optimization (https://arxiv.org/abs/2408.11071)
- **What's New**: 이 논문에서는 텍스트-이미지(T2I) 생성 모델의 취약성을 드러내기 위해 기존의 공격 방법을 재고하고, '순수 블랙박스(black-box)' 방법론을 적용한 DiffZOO라는 새로운 공격 방식을 제안합니다. 이는 공격자가 모델에 대한 사전 정보 없이도 효과적으로 공격할 수 있게 합니다.

- **Technical Details**: DiffZOO는 제로 순서 최적화(Zeroth Order Optimization)를 사용하여 기울기(gradient)를 근사화하고, 이로써 공격 프롬프트(prompt)가 포함된 디스크리트(discrete) 공간 내에서 최적화할 수 있도록 합니다. C-PRV(Continuous Position Replacement Vectors)와 D-PRV(Discrete Position Replacement Vectors)를 활용하여 공격 프롬프트를 개선합니다.

- **Performance Highlights**: DiffZOO는 여러 안전 메커니즘을 갖춘 T2I 모델에 대해 실험을 진행한 결과, 평균 공격 성공률이 기존 연구에 비해 8.5% 향상되었습니다. 이는 T2I 모델의 레드 팀(red teaming) 도구로서의 잠재력을 보여줍니다.



### Reconstruct Spine CT from Biplanar X-Rays via Diffusion Learning (https://arxiv.org/abs/2408.09731)
- **What's New**: 이 논문에서는 복잡한 수술 가이던스에 필요한 3D CT 이미지를 x-ray로부터 재구성하는 새로운 방법을 제안합니다. 특히, 정적 biplanar x-ray를 활용하여 조건부 확산 프로세스를 이용하여 CT 이미지를 생성하는 혁신적인 접근 방식이 특징입니다.

- **Technical Details**: 제안된 방법인 Diff2CT는 조건부 노이즈 제거 오토인코더를 사용하여, 정방향 biplanar x-ray 이미지를 기반으로 한 CT 이미지의 노이즈가 제거된 버전을 예측합니다. 구조적 일관성을 강화하기 위해 새로운 projection loss 함수를 도입하였으며, 이는 3D 공간에서의 구조적 정합성을 보장합니다.

- **Performance Highlights**: 실험 결과, Diff2CT는 0.83의 더 높은 구조유사도지수(Structural Similarity Index, SSIM)와 83.43의 더 낮은 프레셋 인셉션 거리(Fréchet Inception Distance, FID)를 기록하여 기존의 최첨단 벤치마크를 초월하는 시각적 이미지 품질과 평가 지표를 달성했습니다.



New uploads on arXiv(cs.AI)

### Leveraging Chemistry Foundation Models to Facilitate Structure Focused Retrieval Augmented Generation in Multi-Agent Workflows for Catalyst and Materials Design (https://arxiv.org/abs/2408.11793)
- **What's New**: 본 논문에서는 대규모 사전 학습된 화학 기초 모델(chemistry foundation models)을 활용하여 소분자, 복잡한 고분자 물질 및 화학 반응에 대한 정보 검색을 가능하게 하는 방법을 제시합니다. 또한, OpenCLIP과 같은 이미지 모델과 결합하여 여러 특성 데이터 도메인 간의 정보 검색을 혁신적으로 수행할 수 있음을 보여줍니다.

- **Technical Details**: 딥 러닝 모델(deep learning models)을 통한 분자 특성 예측(molecular property prediction) 및 생성적 디자인(generative design) 분야에서, 대규모 언어 모델(large language models, LLMs)과 이들에 의해 구동되는 에이전트(agentic systems)가 사전 학습(pre-trained) 모델을 활용하여 복잡한 연구 작업(context of more complex research tasks)의 예측을 가능하게 합니다. 그러나 여전히 물질 디자인(material design) 작업에 대한 유의미한 정보(retrieval of salient information) 검색에 있어 개선의 여지가 존재합니다.

- **Performance Highlights**: 이 시스템은 다중 에이전트 시스템(multi-agent systems) 내에서 구조 및 위상 기반(topological-based) 자연어 쿼리(natural language queries)와 정보 검색을 용이하게 하여 복잡한 연구 작업을 지원합니다.



### DreamFactory: Pioneering Multi-Scene Long Video Generation with a Multi-Agent Framework (https://arxiv.org/abs/2408.11788)
Comments:
          13 pages, 8 figures

- **What's New**: 이 논문에서는 긴 비디오 생성을 위한 새로운 모델인 DreamFactory를 소개합니다. 기존의 비디오 생성 모델은 짧은 클립에서는 뛰어난 성과를 보였으나, 다중 장면이 포함된 긴 비디오에서는 어려움을 겪었습니다. DreamFactory는 다중 에이전트 협업 원칙과 Key Frames Iteration Design Method를 활용하여 이러한 문제를 해결합니다.

- **Technical Details**: DreamFactory는 LLM(대형 언어 모델)을 기반으로 하며, 제작 과정에서 각 LLM이 감독, 미술 감독, 각본가 및 아티스트와 같은 역할을 맡아 협력합니다. 이 프레임워크는 시나리오 작성, 스토리보드 생성, 캐릭터 디자인, 키프레임 개발 등을 포함하여 비디오 생성을 자동화합니다. 본 모델은 비디오 세그먼트 간의 일관성을 확보하기 위해 특정 키프레임 반복 방법을 적용합니다.

- **Performance Highlights**: DreamFactory는 UTF-101 및 HMDB51 데이터셋을 사용하여 평가한 결과, 기존 모델에 비해 상당한 성능 개선이 있음을 보여주었습니다. 특히, 우리 모델이 생성한 긴 비디오는 수작업으로 생성된 비디오보다 평균적인 품질을 초과하는 것으로 평가되었습니다.



### JieHua Paintings Style Feature Extracting Model using Stable Diffusion with ControlN (https://arxiv.org/abs/2408.11744)
Comments:
          accepted by ICCSMT 2024

- **What's New**: 이 연구는 Jiehua의 스타일적 특징을 추출하기 위한 새로운 접근 방식을 제안합니다. Fine-tuned Stable Diffusion Model with ControlNet (FSDMC)를 활용하여 아티스트의 Jiehua에서 묘사 기술을 정제하는 방법을 모색했습니다.

- **Technical Details**: FSDMC의 훈련 데이터는 인터넷에서 수집한 오픈소스 Jiehua 아티스트 작품을 기반으로 하여, (원본 이미지, Canny Edge Features, Text Prompt) 형식으로 수작업으로 구성되었습니다. 이 논문에서 확인된 최적의 하이퍼파라미터를 사용한 결과, FSDMC는 다른 주류 스타일 전이 모델인 CycleGAN을 능가하는 성능을 보였습니다.

- **Performance Highlights**: FSDMC는 데이터셋에서 FID 3.27을 달성하였으며, 전문 평가에서도 CycleGAN을 초월하였습니다. 이는 Jiehua 스타일 특징 추출에서 모델의 높은 효율성을 나타낼 뿐만 아니라, 원본 사전 훈련된 의미 정보를 보존하고 있다는 것을 보여줍니다.



### Clinical Insights: A Comprehensive Review of Language Models in Medicin (https://arxiv.org/abs/2408.11735)
Comments:
          Submitted to PLOS Digital Health

- **What's New**: 이 논문은 헬스케어 분야에서 대규모 언어 모델(LLM)의 발전과 응용 사례를 자세히 조사하며 특히 임상 응용에 중점을 둡니다. LLM의 발전 과정을 바탕 기술에서부터 도메인 특화 모델 및 다중 모드 통합의 최신 개발까지 추적합니다.

- **Technical Details**: 논문은 인코더 기반 모델에서 미세 조정이 필요한 것에서 텍스트, 비주얼(visual), 청각(auditory) 데이터를 통합하는 정교한 접근 방식으로의 기술 발전을 탐구합니다. 개방형 LLM은 민감한 의료 데이터의 보호를 강화하고 클라우드 기반 솔루션의 사용을 피하기 위해 온프레미스 환경에서 배포될 수 있는 장점을 가지고 있습니다.

- **Performance Highlights**: 이 연구는 LLM의 임상 효율성을 높이는 기회와 윤리, 데이터 프라이버시, 실행 관련 도전과제를 제시하며, LLM의 배포 전략을 비판적으로 평가합니다. 또한 향후 연구 방향으로 LLM의 실제 효과성을 평가하기 위한 실증적 연구와 추가 연구를 위한 오픈 데이터셋의 개발을 제안합니다.



### Physics-informed Discovery of State Variables in Second-Order and Hamiltonian Systems (https://arxiv.org/abs/2408.11691)
- **What's New**: 본 연구에서는 물리적 특성을 활용하여 동적 시스템을 모델링하는 네트워크 모델을 개선하는 방법을 제시합니다. 이는 기존의 모델과는 달리, 상태 변수를 발견하기 위해 물리 원리를 통합하여 더욱 신뢰할 수 있는 상태 변수의 식별을 가능하게 합니다.

- **Technical Details**: 제안된 모델은 두 번째 차수의 Hamiltonian 시스템의 물리적 특성을 활용하여, 기존 모델에 대한 제약 조건을 부과합니다. 이러한 물리 기반의 기계 학습 접근 방식을 통해 관찰 편향과 학습 편향을 도입하고, 맞춤형 조정을 통하여 비중복적이고 해석 가능한 상태 변수를 식별합니다. 이를 위해 Physics-Informed AutoEncoder(PI-AE), Physics-Informed Variational AutoEncoder(PI-VAE), Hamiltonian Physics-Informed Variational AutoEncoder(HPI-VAE) 등의 다양한 모델이 도입되었습니다.

- **Performance Highlights**: 제안된 방법은 Chen et al.의 원래 데이터셋에서 기존 모델보다 더 정확하게 상태 변수의 수를 식별하는 데 성공하였으며, 모델의 해석 용이성을 높였습니다. 이로 인해 시스템 동역학에 대한 이해를 증가시키고 있습니다.



### Don't Kill the Baby: The Case for AI in Arbitration (https://arxiv.org/abs/2408.11608)
- **What's New**: 2022년 Generative AI (GenAI)의 도입 이후, AI의 인간 지능 시뮬레이션 및 콘텐츠 생성 능력이 주목받고 있으며, 특히 법률 분야에서 AI의 실제 이점을 간과하는 경향이 있다. 본 논문은 AI가 중재(Arbitration)에서 어떻게 통합될 수 있는지를 살펴본다.

- **Technical Details**: Federal Arbitration Act (FAA)에 따라 양 당사자가 AI 기반 중재를 계약적으로 선택할 수 있음을 주장하며, 중재가 AI 채택을 위한 이상적인 출발점이라고 강조한다. AI는 합의가 있을 경우 효과적인 중재자로 기능할 수 있다. 또한, AI와 인간 중재의 경험적 비교 연구의 필요성을 제시하며 다른 시스템 개발로 이어질 수 있음을 언급한다.

- **Performance Highlights**: AI의 도입은 중재의 효율성, 공정성, 유연성을 향상시킬 수 있는 잠재력을 가지고 있으며, 계약 자율성을 존중하고 AI의 가능성을 실현할 수 있는 환경 조성의 중요성을 강조한다.



### Drama Engine: A Framework for Narrative Agents (https://arxiv.org/abs/2408.11574)
Comments:
          10 pages, 2 figures, 2 tables

- **What's New**: 이번 기술 보고서는 이야기 목적으로 설계된 대형 언어 모델과의 에이전틱 상호작용을 위한 새로운 프레임워크인 Drama Engine을 소개합니다. 이 프레임워크는 다중 에이전트 시스템 원칙을 적용하여 시간에 따라 발전하고 사용자 및 서로 상호작용할 수 있는 동적이고 상황 인식이 가능한 동반자를 생성합니다.

- **Technical Details**: Drama Engine의 핵심 기능은 다음과 같습니다: \n1. 다중 에이전트 워크플로우 및 위임: 여러 에이전트 간의 대화를 조정하는 방식으로, 에이전트는 더 복잡한 작업을 차일드를 통해 위임할 수 있습니다.\n2. 동적 프롬프트 조합: 프롬프트는 맥락에 따라 조립됩니다. 이 맥락에는 다른 채팅 참가자, 작업 상태 등 여러 데이터가 포함됩니다.\n3. 모델 및 벤더 비민감성: Drama Engine은 OpenAI의 API 표준을 지원하는 모든 백엔드를 사용할 수 있습니다.\n4. 동반자의 시간 발전 및 기분 시스템: 동반자는 시간이 지남에 따라 발전하고 상황에 따라 기분이 변동할 수 있습니다. \n5. 자동 맥락 요약: 문맥의 양이 모델의 맥락 크기를 초과할 경우 자동으로 요약할 수 있는 기능이 포함되어 있습니다.

- **Performance Highlights**: Drama Engine은 창의적인 글쓰기 및 다중 에이전트 대화의 다양한 응용 프로그램에서 운영되고 있으며, 모듈화된 방식으로 동적인 프롬프트 시퀀스를 구성하여 기존의 단순 프롬프트 체인보다 훨씬 유연하고 제어 가능한 시스템을 제공합니다.



### Explainable Deep Learning Framework for Human Activity Recognition (https://arxiv.org/abs/2408.11552)
- **What's New**: 이 논문에서는 Human Activity Recognition (HAR) 분야에서 모델의 해석 가능성과 성능을 향상시키기 위한 새로운 프레임워크를 제안합니다. 이 프레임워크는 모델에 의존하지 않는(went model-agnostic) 접근법으로, 경쟁적인 데이터 증강(competitive data augmentation)을 통해 보다 직관적이고 접근 가능한 설명을 제공합니다.

- **Technical Details**: 이 모델은 데이터 증강을 통해 원본 데이터의 의미를 유지하면서 새로운 데이터 인스턴스를 생성합니다. 이러한 데이터 증강 과정은 모델 훈련 및 예측 단계 모두에서 적용되며, 훈련 중에 생성된 변형 샘플이 모델의 결정 경계를 정제하여 예측 성능을 개선합니다. 또한, 두 가지 조건을 충족해야 하며, 변환 과정이 일치하거나 예측 단계의 변환이 훈련 중 사용된 변환의 하위 집합이어야 합니다.

- **Performance Highlights**: 다섯 개의 벤치마크 데이터 세트에 대해 세 가지 최신 기초 모델로 실시한 실험을 통해 제안된 프레임워크의 효과를 검증하였으며, 데이터 증강을 통해 HAR 시스템의 해석 가능성이 크게 향상됨을 보였습니다. 코드는 깃허브에 공개되어 (http://www.github.com/...) 있습니다.



### RConE: Rough Cone Embedding for Multi-Hop Logical Query Answering on Multi-Modal Knowledge Graphs (https://arxiv.org/abs/2408.11526)
- **What's New**: 이번 연구에서는 Multi-Modal Knowledge Graphs (MMKGs)에서의 논리적 쿼리 답변을 위한 새로운 방법인 RConE를 제안합니다. RConE는 복잡한 논리 쿼리를 위한 First Order Logic (FOL) 연산자를 활용하는 기능을 제공하며, 특히 다중 양식(multi-modal) 개체의 하위 개체(sub-entities)를 결과로 얻을 수 있는 가능성을 드러냅니다.

- **Technical Details**: RConE는 논리적 질의 처리를 위해 서로 다른 MMKG 내 다중 양식 개체를 동일한 임베딩 공간으로 임베딩하여, 해당 개체와 하위 개체를 효과적으로 조회할 수 있게 합니다. 이를 위해, 질의 생성 알고리즘을 확장하여 다양한 다중 양식 개체를 포함하고 학습 질의를 생성합니다.

- **Performance Highlights**: RConE는 네 가지 공개 MMKG 데이터 세트를 통해 평가된 결과, 기존 최고의 방법에 비해 56.5% 더 높은 MRR (Mean Reciprocal Rank) 점수를 기록하며 우수한 성능을 입증하였습니다.



### Quantifying Behavioural Distance Between Mathematical Expressions (https://arxiv.org/abs/2408.11515)
Comments:
          15 pages, 10 figures, 1 table, 2 appendices

- **What's New**: 이 논문에서는 기존의 기호 회귀(symobolic regression) 방법들이 수학적 표현 간의 합성적 유사성(syntactic similarity)만을 기준으로 후보 수학적 표현의 공간을 구성하고 있다는 점을 지적하며, 수학적 대칭(mathematical symmetries)에서 비롯되는 중요한 동등성을 간과하고 있음을 강조합니다. 이로 인해 유사한 오류를 가진 표현들이 검색 공간에서 서로 먼 위치에 놓이게 됩니다. 본 논문은 유사한 오류를 가진 표현들을 함께 클러스터링하는 새로운 행동 거리(behavioral distance) 측정 방법인 BED(Behavioral Expression Distance)를 제안합니다.

- **Technical Details**: 전통적인 기호 회귀 방법들은 Levenshtein distance와 같이 합성적인 유사성을 기반으로 검색 공간을 구성합니다. 그러나 이 논문은 BED를 통해 표현의 행동을 반영하여 비슷한 오류를 가진 표현들이 인접하게 구성될 수 있음을 제안합니다. 이를 위해 BED의 계산에는 무작위 샘플 값을 사용하여 효율적인 동적 프로그래밍 기법을 적용합니다.

- **Performance Highlights**: 실험 결과, BED를 계산하는 확률적 방법이 샘플링된 값의 수가 적어도 일관성을 유지하며, 이는 트리 기반의 합성적 거리(syntactic distance)와 비교할 만한 계산 효율성을 보여줍니다. BED를 적용하면 기호 회귀의 검색 공간 내에서 오류 경관의 부드러움이 크게 개선되는 것을 확인했습니다.



### Mutagenesis screen to map the functionals of parameters of Large Language Models (https://arxiv.org/abs/2408.11494)
Comments:
          10 pages, 6 figures, supplementary material available online

- **What's New**: 이번 논문에서는 Llama2-7b와 Zephyr 모델을 대상으로 mutagenesis screen 방식을 통해 모델의 파라미터와 기능 간의 관계를 조사하였습니다. 이 방법은 생물학적 연구에서 사용되는 기법에 영감을 받아 개발되었습니다.

- **Technical Details**: 모델의 행렬 내 요소들을 최대값 또는 최소값으로 변형하여 파라미터와 기능 간의 관계를 분석하였으며, 여러 레벨의 미세 구조가 발견되었습니다. 특히, 변형된 요소가 특정 축을 따라 군집화되는 경향이 있었고, Gate 행렬에서 독특한 2차원 비대칭이 나타났습니다.

- **Performance Highlights**: Zephyr 모델에서는 특정 변형이 서술적 출력보다 시적 또는 대화 형식의 출력을 일관되게 생성함을 확인하였고, 고빈도 초기 단어에 따라 '작가' 변형들이 그룹화되는 경향을 보였습니다. 이러한 연구 결과들은 큰 언어 모델의 복잡성을 해석하는 데 효과적인 도구가 될 수 있음을 보여줍니다.



### Estimating Peer Direct and Indirect Effects in Observational Network Data (https://arxiv.org/abs/2408.11492)
Comments:
          AAAI

- **What's New**: 본 연구에서는 관찰된 네트워크 데이터를 기반으로 하여 동료의 직접 효과(peer direct effect, PDE)와 간접 효과(peer indirect effect, PIE), 그리고 개인의 자기 치료 효과(self-treatment effect, STE)를 동시에 고려하는 새로운 방법론인 gDIS를 제안합니다.

- **Technical Details**: gDIS는 멀티 레이어 그래프 신경망(multi-layer graph neural networks, GNNs)과 주의 메커니즘(attention mechanisms)을 활용하여 각 이웃이 개인에게 미치는 영향을 구별하고, 노드 특성과 표현 간의 의존성을 제어하기 위해 Hilbert-Schmidt Independence Criterion (HSIC)을 통합하여 그래프의 구조 정보를 최대한 활용합니다.

- **Performance Highlights**: 두 개의 반-합성 데이터셋에서 gDIS의 효과성과 강인성이 입증되었으며, 복잡한 네트워크 데이터에서도 뛰어난 성능을 유지하는 것으로 나타났습니다.



### Nothing in Excess: Mitigating the Exaggerated Safety for LLMs via Safety-Conscious Activation Steering (https://arxiv.org/abs/2408.11491)
- **What's New**: 본 논문에서는 Safety-Conscious Activation Steering (SCANS) 기법을 제안하여 안전 정렬된 대형 언어 모델(LLMs)의 과도한 안전 문제를 완화하고, 친근한 쿼리를 거부하는 현상을 개선합니다.

- **Technical Details**: SCANS는 우선 활성화 공간 내에서 거부를 유도하는 벡터를 추출하고, 모델의 거부 행동에 영향을 미치는 특정 안전-critical 레이어에 대해 어휘 프로젝션(vocabulary projection)을 활용하여 고정합니다. 또한, 숨겨진 상태 전이를 추적하여 유도 방향을 식별하고 모델의 행동을 조정합니다.

- **Performance Highlights**: 실험 결과, SCANS는 XSTest 및 OKTest 벤치마크에서 새로운 최첨단 성능을 달성하며, 해로운 쿼리에 대한 방어 능력을 손상시키지 않고도 모델의 능력을 거의 변하지 않게 유지합니다.



### Bidirectional Gated Mamba for Sequential Recommendation (https://arxiv.org/abs/2408.11451)
- **What's New**: 본 논문에서는 Sequential Recommender Systems(SRS)에서 Mamba의 한계를 극복하기 위한 새로운 프레임워크인 SIGMA를 제안합니다. 이 프레임워크는 Partially Flipped Mamba(PF-Mamba)와 Dense Selective Gate(DS Gate)를 활용하여 맥락 모델링(context modeling)과 짧은 시퀀스 모델링(short sequence modeling)의 문제를 해결합니다. 특히, SIGMA는 사용자의 상호작용을 이해하는 데 중점을 두고 설계되었습니다.

- **Technical Details**: SIGMA는 Bidirectional 구조를 채택하여 Mamba의 비사교적 구조에 의한 한계를 보완합니다. PF-Mamba는 데이터를 처리하기 위해 선택적으로 정보를 추출하고, DS Gate는 두 방향의 가중치를 최적화합니다. 또한 Feature Extract GRU(FE-GRU)를 도입하여 짧은 의존성을 효율적으로 캡처합니다. 이러한 구성 요소들은 SIGMA 프레임워크의 핵심 기능을 형성하며, 다섯 개의 공공 데이터세트에서 실험을 통해 그 효용성을 증명하였습니다.

- **Performance Highlights**: SIGMA는 다섯 개의 실제 데이터세트에서 기존 모델들보다 우수한 성능을 보였습니다. 특히, SIGMA는 짧은 사용자 상호작용 데이터에서도 높은 예측 정확성을 유지하며, 긴 시퀀스에서도 효과적으로 사용자 선호도를 파악할 수 있도록 설계되었습니다.



### Enabling Small Models for Zero-Shot Classification through Model Label Learning (https://arxiv.org/abs/2408.11449)
- **What's New**: 본 논문에서 제안하는 모델 레이블 학습(Model Label Learning, MLL) 패러다임은 기존의 비전-언어 모델(VLMs)과 달리, 태스크 특화 전문가 모델들을 제로샷(zero-shot) 분류 작업에 활용할 수 있도록 모델의 기능에 따라 모델을 정렬하고, 새로운 태스크를 효과적으로 해결할 수 있는 방법을 제시합니다.

- **Technical Details**: MLL은 세 가지 주요 단계로 구현됩니다: 1) 모델 레이블링, 2) 모델 선택, 3) 모델 재사용. 모델 레이블링 단계에서는 의미론적 방향 비순환 그래프(Semantic Directed Acyclic Graph, SDAG)를 구축하여 각 노드가 특정 의미 클래스를 설명합니다. 이후 모델 선택 시 모델 레이블을 사용하여 후보 모델을 선택하고, 분류 헤드 조합 최적화(Classification Head Combination Optimization, CHCO) 기법을 통해 최적 모델을 선택합니다. 마지막으로 선택된 모델을 조합하여 제로샷 방식으로 목표 태스크를 처리합니다.

- **Performance Highlights**: 실험을 통해 MLL의 효과성과 효율성이 검증되었으며, 전문가 모델들이 제로샷 분류 작업에 효과적으로 재사용될 수 있음을 보여줍니다. MLL은 더 적은 비용과 더 높은 확장성을 제공하며, 모델 허브의 크기가 커질수록 제로샷 능력이 증가하는 특성을 지닙니다.



### Epistemic Injustice in Generative AI (https://arxiv.org/abs/2408.11441)
- **What's New**: 이 논문은 생성적 AI가 집단 지식의 무결성을 어떻게 해칠 수 있는지를 탐구하며, 우리의 정보 습득, 평가 및 신뢰 과정에 중대한 위협을 가할 수 있음을 주장합니다.

- **Technical Details**: 생성적 알고리즘 인식 불공정(generative algorithmic epistemic injustice)이라는 개념이 도입되며, 증폭된 조작적 증언 불공정(amplified and manipulative testimonial injustice), 해석적 무지(hermeneutical ignorance), 접근 불공정(access injustice)의 네 가지 주요 차원이 소개됩니다.

- **Performance Highlights**: 논문은 생성적 AI가 다국어 맥락에서 정보 불균형을 야기하고, 동시에 민주적 가치와 지식 생산의 무결성을 보호하기 위한 보다 공정한 정보 생태계를 조성하기 위한 전략과 시스템 설계 원칙을 제안합니다.



### Solving Decision Theory Problems with Probabilistic Answer Set Programming (https://arxiv.org/abs/2408.11371)
Comments:
          Under consideration in Theory and Practice of Logic Programming (TPLP)

- **What's New**: 본 논문에서는 Probabilistic Answer Set Programming (PASP)에서 의사 결정 이론 문제를 인코딩할 수 있는 가능성을 제시합니다. 특히, credal semantics 아래에서의 decision atoms와 utility attributes의 도입을 통해 복잡한 의사 결정 환경에서 최적의 전략을 찾는 방법을 제안합니다.

- **Technical Details**: 이 논문은 Decision Theory Problems (DTP)에서의 최적 해를 찾기 위한 두 가지 알고리즘을 개발하였습니다. 첫 번째 알고리즘은 answer sets enumeration 기법을 기반으로 하며, 두 번째는 세 단계의 Algebraic Model Counting (AMC) 기법을 활용하여 추론 과정을 가속화합니다. DTPASP에서 각 전략은 그에 해당하는 최저 및 최고 보상을 최대화하는 방향으로 최적화됩니다.

- **Performance Highlights**: 경험적 결과는 제안한 알고리즘이 큰 규모의 프로그램에서도 합리적인 시간 내에 처리할 수 있음을 보여줍니다. 기존의 enumeration 기반 알고리즘에 비해 AMC 기반 알고리즘이 매우 빠르며, 비트리비얼한 규모의 도메인에서도 효과적으로 작동함을 입증하였습니다.



### Towards Probabilistic Inductive Logic Programming with Neurosymbolic Inference and Relaxation (https://arxiv.org/abs/2408.11367)
Comments:
          15 pages

- **What's New**: 이 논문에서는 불완전하고 확률적 배경 지식을 처리할 수 있는 새로운 Inductive Logic Programming (ILP) 방법, Propper를 제안합니다. 기존 ILP 방법들은 확률 기반의 센서 데이터나 신경망으로부터 학습하는 데 한계가 있었습니다.

- **Technical Details**: Propper는 neurosymbolic inference, 연속적인 가설 선택 기준 (BCE), 및 가설 제약자 완화 (NoisyCombo)를 사용하여 ILP를 확장합니다. 이 방법은 노이즈가 있는 이미지에서 관계 패턴을 학습할 수 있으며, 최소 8개의 사례로 프로그램을 생성할 수 있습니다.

- **Performance Highlights**: Propper는 기존의 이진 ILP 및 Graph Neural Network (GNN)와 같은 통계적 모델에 비해 우수한 성능을 보였습니다. 실험을 통해 다양한 모델의 학습 강건성과 효율성을 검증하였으며, 복잡한 실제 이미지를 처리하는 데 강점을 나타냈습니다.



### ProteinGPT: Multimodal LLM for Protein Property Prediction and Structure Understanding (https://arxiv.org/abs/2408.11363)
Comments:
          19 pages, 9 figures, 5 tables

- **What's New**: ProteinGPT는 단백질 분석을 위한 새로운 다중 모달 대화 시스템으로, 사용자가 단백질 서열 및 구조를 업로드하여 소통할 수 있도록 합니다.

- **Technical Details**: ProteinGPT는 단백질 서열 인코더, 단백질 구조 인코더, 프로젝션 레이어 및 LLM(대형 언어 모델)로 구성되어 있습니다. ESM-2 기반의 단백질 서열 인코더는 3억 개의 매개변수를 가진 36개의 Transformer 레이어로 구성되며, esm_if1_gvp4_t16_142M_UR50 구조 인코더는 AlphaFold2로 예측된 1200만 개의 구조로 훈련되었습니다. 또한, 132,092 개의 단백질로 구성된 ProteinQA라는 대규모 데이터셋을 활용합니다.

- **Performance Highlights**: 실험 결과, ProteinGPT는 단백질에 대한 응답과 관련 질문을 정확하고 맥락에 맞게 생성할 수 있는 성능을 보여주었습니다.



### One-step Structure Prediction and Screening for Protein-Ligand Complexes using Multi-Task Geometric Deep Learning (https://arxiv.org/abs/2408.11356)
- **What's New**: LigPose는 단일 모델을 기반으로 하는 새로운 다중 작업 기하학적 딥러닝 방법론으로, 약물 개발에서 단백질-리간드 복합체의 정확한 구조 예측을 가능하게 합니다. 기존의 docking 방법과는 달리, LigPose는 하나의 프로세스에서 리간드 및 단백질 쌍의 3차원 구조를 최적화할 수 있습니다.

- **Technical Details**: LigPose는 서로 세 가지 단계를 기반으로 하여 리간드와 단백질을 그래프로 표현하며, 각 원자는 노드로 표시되고 모든 원자가 상호 연결됩니다. 이 과정에서 기하학적 특징을 학습하고, 비결합 상호작용을 정확하게 측정하며, 스스로 지도 학습(self-supervised learning)을 통해 대규모 비표시 데이터에 대한 일반화 능력을 향상시킵니다.

- **Performance Highlights**: LigPose는 기존의 1212종의 docking 도구보다 14% 높은 성공률을 보였습니다. SARS-CoV-2 Mpro 복합체 구성 예측에서는 18.2%의 정확도가 향상되었으며, 특히 cross-docking 및 virtual screening에서 각각 20.1% 및 19.3%의 성과 개선을 이루었습니다.



### Multimodal Datasets and Benchmarks for Reasoning about Dynamic Spatio-Temporality in Everyday Environments (https://arxiv.org/abs/2408.11347)
Comments:
          5 pages, 1 figure, 1 table

- **What's New**: 본 논문에서는 Embodied AI의 개발을 지원하기 위해 인공 비디오 데이터와 질문 응답(QA) 데이터셋을 생성하였습니다. 이 데이터셋은 로봇이 가정 환경에서 인간의 행동과 주변 환경을 이해하는 정도를 측정하는 데 초점을 맞췄습니다.

- **Technical Details**: 우리는 VirtualHome-AIST 시뮬레이터를 활용해 3D 비디오 데이터를 생성하고, PrimitiveActionOntology 및 HomeOntology에 기반한 통일된 어휘로 자동 생성된 주석을 사용하였습니다. MMQADL(모듈러스 질문 응답 데이터셋)을 통해 로봇이 일상 생활을 이해하는 능력을 평가할 수 있도록 다양한 질문 유형을 구성했습니다.

- **Performance Highlights**: 초기 실험에서는 Video-LLaVa와 구글의 Gemini 1.5 Pro Vision을 이용하여 AI가 인간 행동을 이해하는 능력을 평가하였습니다. 실험 결과, Gemini 모델이 방의 유형 구분을 잘 수행한 반면, Video-LLaVa는 시간 경과를 이해하는 데 한계가 있음을 나타냈습니다. 전체적으로 데이터셋이 AI의 인간 행동 및 환경 이해에 유용하다는 것을 보여주었습니다.



### Automatic Dataset Construction (ADC): Sample Collection, Data Curation, and Beyond (https://arxiv.org/abs/2408.11338)
- **What's New**: 이 논문에서는 **Automatic Dataset Construction (ADC)**이라는 새로운 방법론을 제안합니다. ADC는 높은 효율성과 낮은 비용으로 데이터셋 생성을 자동화하여 맞춤형 학습 데이터의 수요를 충족하려고 합니다. 이미지를 분류하는 작업을 시작으로, ADC는 **LLMs (Large Language Models)**를 활용하여 관련 샘플을 검색하고 수집하는 과정에서 수작업 주석의 필요성을 크게 줄입니다.

- **Technical Details**: ADC의 기본 구조는 데이터셋 설계, 샘플 수집, 데이터 정제의 세 가지 주요 단계로 나뉘며, 기존의 **Traditional Dataset Construction (TDC)**와는 다릅니다. ADC는 LLM을 사용하여 필드 검색을 자동화하고, 노이즈가 섞인 레이블 샘플을 필터링하기 위해 인간 주석가에게 지침을 제공합니다. 논문에서는 또한 **Clothing-ADC**라는 의류 데이터셋을 예시로 들어 그 생성 과정을 자세히 설명합니다.

- **Performance Highlights**: Clothing-ADC 데이터셋은 1,076,738개의 샘플로 구성되어 있으며, 20,000개의 평가, 20,000개의 테스트 샘플이 포함됩니다. 이 데이터셋은 12개의 주요 클래스로 분류되어 있으며 각 의류 타입마다 1,000개의 서브클래스를 생성했습니다. 데이터 품질 개선을 위한 몇 가지 오픈소스 소프트웨어도 제공되어 있으며, 레이블 오류 감지 및 학습, 클래스 불균형 학습과 같은 문제를 해결합니다.



### Automating Thought of Search: A Journey Towards Soundness and Completeness (https://arxiv.org/abs/2408.11326)
- **What's New**: 이 논문은 Planning 문제를 해결하기 위해 Thought of Search (ToS) 개념을 자동화한 AutoToS를 제안합니다. 이는 인간을 배제하면서도 효과적인 검색 문제 해결 방법을 제공합니다.

- **Technical Details**: AutoToS는 단계를 구분하여 언어 모델이 유효하고 완전한 검색 구성 요소를 생성하도록 안내합니다. 피드백은 일반적이며 도메인 특화된 유닛 테스트를 통해 제공되며, 이는 코드 생성 및 복잡한 추론 작업을 위한 LLM의 발전과 관련이 있습니다.

- **Performance Highlights**: 모든 평가된 도메인에서 다양한 크기의 LLM을 사용하여 100% 정확도를 달성하였으며, 피드백 반복은 최소화되었습니다.



### Probabilistic Medical Predictions of Large Language Models (https://arxiv.org/abs/2408.11316)
Comments:
          58 pages, 3 figures, 3 tables, Submitted to Nature Communication

- **What's New**: 이번 연구는 Large Language Models (LLMs)의 임상 응용에 대한 가능성을 탐구하면서, LLMs가 생성한 예측 확률의 신뢰성 문제를 다룹니다.

- **Technical Details**: 우리는 여섯 가지 고급 오픈소스 LLM과 다섯 개의 의료 데이터셋을 활용하여, 명시적인 확률(Explicit probabilities)과 암묵적인 확률(Implicit probabilities)을 비교했습니다. 명시적인 확률은 텍스트 생성 과정을 통해 도출되었으며, 암묵적인 확률은 정확한 레이블 토큰을 예측할 확률에 기반했습니다.

- **Performance Highlights**: 명시적인 확률의 성능은 변별력(discrimination), 정밀도(precision), 재현율(recall) 측면에서 암묵적인 확률보다 일관되게 낮은 결과를 나타냈습니다. 특히, 작은 LLM 및 불균형 데이터셋에서 이러한 차이가 더욱 두드러졌습니다.



### Unlocking Adversarial Suffix Optimization Without Affirmative Phrases: Efficient Black-box Jailbreaking via LLM as Optimizer (https://arxiv.org/abs/2408.11313)
- **What's New**: 본 논문에서는 ECLIPSE라는 새로운 블랙박스(jailbreaking) 방법을 소개합니다. 이 방법은 최적화 가능한 접미사(suffixes)를 활용하여 기존의 jailbreaking 기법보다 훨씬 효율적으로, 그리고 자율적으로 해로운 LLM 출력 생성을 유도합니다.

- **Technical Details**: ECLIPSE는 LLM의 강력한 생성(generation) 및 최적화(optimization) 능력을 활용하여 자연어로 jailbreaking 목표를 변환하는 작업 프롬프트(task prompts)를 사용합니다. 또한, 해로운 점수를 제공하는 스코어러가 지속적인 피드백을 제공하여 LLM이 자율적으로 접미사를 생성할 수 있도록 합니다.

- **Performance Highlights**: ECLIPSE는 세 가지 오픈 소스 LLM과 GPT-3.5-Turbo에서 평균 공격 성공률(ASR) 0.92를 달성하며, 기존의 최적화 기반 방법인 GCG보다 2.4배 더 뛰어난 성능을 보입니다. 또한, ASR은 템플릿 기반 방법과 유사하지만 공격 효율성이 뛰어나 평균 공격 오버헤드를 83% 감소시켰습니다.



### EEG-Defender: Defending against Jailbreak through Early Exit Generation of Large Language Models (https://arxiv.org/abs/2408.11308)
Comments:
          19 pages, 7 figures

- **What's New**: 최근 연구에서 제안된 EEG-Defender는 대형 언어 모델(LLMs)에 대한 새로운 방어 전략으로, 초기 트랜스포머 출력 결과를 활용하여 악의적인 입력을 감지하고 즉시 생성을 종료하는 방법을 제시합니다. 이 기술은 기존의 방어 방법들보다 더 높은 Attack Success Rate (ASR) 감소율을 보여줍니다.

- **Technical Details**: EEG-Defender는 초기 및 중간 계층의 출력 임베딩을 비교하여 악의적인 프롬프트와 유사성을 평가합니다. 이 방법은 프롬프트에 의해 유도된 추론을 방지하는 대신, LLM이 생성한 출력의 초기 상태를 분석함으로써 악의적인 요청을 거부합니다. 실험은 Llama2, Vicuna, Guanaco와 같은 세 가지 LLM 모델에서 수행되었습니다.

- **Performance Highlights**: EEG-Defender는 기존 jailbreak 방법들에 비해 ASR을 약 85% 감소시키며, 기능적으로는 benign prompts에 거의 영향을 주지 않습니다. 또한 이 방법은 기존 LLM의 미세 조정 없이 간편하게 통합될 수 있는 특징을 가지고 있습니다.



### Applying and Evaluating Large Language Models in Mental Health Care: A Scoping Review of Human-Assessed Generative Tasks (https://arxiv.org/abs/2408.11288)
- **What's New**: 이번 논문은 대형 언어 모델(LLMs)이 정신 건강 관리 분야에서의 응용 가능성을 평가하기 위해 수행된 스코핑 리뷰를 다룹니다. 특히, 실제 상황에서 인간 참가자와 함께 테스트된 연구에 집중하고 있습니다.

- **Technical Details**: 연구는 APA PsycNet, Scopus, PubMed 및 Web of Science를 통해 726개의 독특한 논문을 체계적으로 수집하고, 그중 17개의 논문이 포함 기준을 충족했습니다. 응용 분야는 임상 지원(clinical assistance), 상담(counseling), 치료(therapy) 및 정서적 지원(emotional support)을 포함하였으나, 평가 방법은 비표준화(non-standardized)되어 있어 비교 가능성과 신뢰성이 제한적이었습니다.

- **Performance Highlights**: LLMs는 정신 건강 관리 접근을 확대하는 가능성을 보였으나, 현재 증거는 이들이 독립적인 개입(intervention)으로 사용될 수 있음을 완전히 지지하지 않습니다. 따라서 보다 엄격하고 표준화된 평가와 윤리적 감독이 필요합니다.



### BearLLM: A Prior Knowledge-Enhanced Bearing Health Management Framework with Unified Vibration Signal Representation (https://arxiv.org/abs/2408.11281)
- **What's New**: 이번 논문에서는 대형 언어 모델을 활용한 베어링 건강 관리 프레임워크인 BearLLM을 제안합니다. 이 모델은 사용자 프롬프트와 진동 신호를 처리하여 여러 베어링 관련 작업을 통합하는 새로운 멀티모달 모델입니다.

- **Technical Details**: BearLLM은 다양한 작업 조건을 다루기 위해 prior knowledge-enhanced unified vibration signal representation을 도입합니다. 진동 신호를 고정 길이 세그먼트가 아닌 가변 길이의 일관된 시간 세그먼트로 샘플링하고, 주파수 도메인을 포함하여 입력 차원을 통합합니다. 또한, fault-free reference signal을 보조 입력으로 사용하여 복잡한 기계 분석 없이 정확성을 높입니다. 추출된 특징은 워드 임베딩으로 변환되고, 이후 사용자 텍스트 임베딩과 결합되어 LLM의 입력으로 사용됩니다.

- **Performance Highlights**: BearLLM은 사전 훈련된 가중치를 사용하여 아홉 개의 공개적인 결함 진단 벤치마크에서 최첨단 성능을 발휘하며, 개별 데이터셋을 위해 설계된 특정 방법들을 초월했습니다. 제안한 방법론과 함께 제공된 매칭 진동 신호 및 텍스트 설명이 포함된 대규모 멀티모달 베어링 건강 관리 데이터셋(MBHM)이 연구 개발에 기여할 것으로 기대됩니다.



### Towards Analyzing and Mitigating Sycophancy in Large Vision-Language Models (https://arxiv.org/abs/2408.11261)
- **What's New**: 본 연구는 대형 시각-언어 모델(LVLMs)의 비판적인 문제인 sycophancy를 정밀하게 분석하고 이를 완화하기 위한 새로운 방법인 Leading Query Contrastive Decoding(LQCD)을 제안함으로써 이 분야의 연구 공백을 메우고자 하였습니다.

- **Technical Details**: LQCD는 모델에 구애받지 않는 방법으로, 시각적 정보와 언어 모델의 출력을 통합하며, sycophancy 토큰의 확률을 억제하여 모델의 과도한 선행 신호 의존성을 보정합니다. 다양한 VL 벤치마크에 대해 엄선된 leading query를 사용하여 성능을 평가하고, Hallucination 문제를 줄이는데 효과적임을 입증합니다.

- **Performance Highlights**: LQCD는 전반적으로 다른 prompt engineering 기법 및 Hallucination 완화 방법보다 우수한 성능을 보이며, 중립적인 질문에 대한 LVLM의 응답도 약간 개선시켜 더 효과적인 일반 목적 디코딩 전략임을 제시합니다.



### Improving Speech Recognition Error Prediction for Modern and Off-the-shelf Speech Recognizers (https://arxiv.org/abs/2408.11258)
- **What's New**: 본 연구에서는 음성 인식 오류를 예측하는 기존 모델을 확장하여, posterior 기반의 신경망 음향 모델의 동작을 모사하는 샘플링 기반 패러다임을 도입하였습니다. 추가적으로, 혼동 행렬(confusion matrix)을 시퀀스-대-시퀀스(sequence-to-sequence) 모델로 대체하여 예측의 맥락 의존성을 강화했습니다.

- **Technical Details**: 연구에서는 두 가지 모델을 비교: 첫째, 기초 혼동 행렬 기반 모델로부터 샘플링하여 출력 분포를 생성합니다. 둘째, 2층의 128 유닛을 가진 Seq2Seq 모델을 활용하여 문맥 정보를 포함한 음향 오류 예측을 수행합니다. 모델 학습시 입력과 출력 음표의 정렬을 위해 혼동 행렬과 유사한 기술을 사용하였습니다.

- **Performance Highlights**: 샘플링 기법은 100가지 추정 기반에서 예측 정확성을 크게 향상시키었고, 시퀀스 모델의 성능은 혼동 행렬과 유사하게 나타났습니다. 연구는 무관한 클라우드 기반 ASR 시스템의 행동을 추정하는 데에도 성공적으로 적용되었습니다.



### The Dilemma of Uncertainty Estimation for General Purpose AI in the EU AI Ac (https://arxiv.org/abs/2408.11249)
Comments:
          7 pages, 2nd GenLaw Workshop @ ICML 2024

- **What's New**: 유럽 연합의 AI 법안(AI Act)이 AI 시스템에 대한 최초의 포괄적 규제이며, 일반 목적 AI 모델에 대한 법적 준수 및 품질 보증을 위해 불확실성 추정을 필수 요소로 제안하는 내용을 다루고 있습니다.

- **Technical Details**: AI 법안은 일반 목적 AI(GPAI) 모델에 대한 구체적인 요구 사항을 포함하며, 이들 모델의 불확실성 추정을 통해 투명성과 정확성을 높이고 시스템 리스크를 관리할 수 있는 방안을 제공합니다. 불확실성 추정 방법은 두 가지 주요 카테고리, 즉 직접적인 방법(예: Ensemble 방식)과 샘플링 방법(Monte Carlo Dropout 등)으로 나뉩니다.

- **Performance Highlights**: 불확실성 추정을 통해 GPAI 모델의 개발 및 평가 과정에서의 컴퓨팅 비용 증가라는 단점이 있으나, 이 방법이 모델의 신뢰성과 품질 보증을 향상시킬 수 있는 잠재력을 가지고 있음이 강조되었습니다.



### Out-of-Distribution Detection with Attention Head Masking for Multimodal Document Classification (https://arxiv.org/abs/2408.11237)
- **What's New**: 이번 연구에서는 Attention Head Masking (AHM)이라는 새로운 방법론을 제안하여 다중 모달 문서 분류 시스템에서 out-of-distribution (OOD) 데이터 탐지를 개선하였습니다. 우리가 제안한 방법은 기존의 다른 방법들보다 유의미하게 낮은 false positive rate (FPR)을 달성했습니다.

- **Technical Details**: AHM은 transformer 모델의 self-attention 메커니즘을 활용하여 ID와 OOD 데이터를 분리하는 기능을 강화하는 방법입니다. 이 기법은 데이터가 각 클래스 간에 얼마나 유사한지를 반영하여 OOD 탐지 성능을 극대화하도록 설계되었습니다. 또한, FinanceDocs라는 고품질의 금융 문서 AI 데이터셋을 새롭게 소개합니다.

- **Performance Highlights**: 제안된 AHM 방법은 기존의 최신 솔루션보다 월등한 성능을 보이며, 특히 Tobacco3482 및 FinanceDocs 데이터셋을 사용한 실험 결과에서 OOD 탐지에서 AUC (AUROC) 메트릭이 우수한 성과를 달성했습니다. 이로써 AHM은 다중 모달 데이터에서도 효과적으로 일반화되어 신뢰성과 안전성을 높이는데 기여할 수 있음을 보여주었습니다.



### Efficient Exploration and Discriminative World Model Learning with an Object-Centric Abstraction (https://arxiv.org/abs/2408.11816)
Comments:
          Preprint

- **What's New**: 이 연구에서는 강화 학습(Reinforcement Learning)의 탐색 문제를 해결하기 위해 객체 중심 매핑(object-centric mapping)을 통해 더 효율적인 학습을 할 수 있는지 조사하였습니다. 이 과정에서 항목을 더 높은 수준의 상태 추상화로 모델링하고, 속성 변화를 원시 동작에 대한 더 높은 수준의 시간 추상화로 모델링하는 방식이 발견되었습니다.

- **Technical Details**: 제안된 접근 방식은 Ab-MDP(abstracted Markov Decision Process)를 통해 이루어지며, 이를 통해 객체의 상태 및 동작에 대한 기초적인 정책을 설계합니다. MEAD(Model-based Exploration of abstracted Attribute Dynamics)라는 모델이 특정 객체의 변화가 성공할 것을 예측할 수 있도록 학습됩니다. 이 모델은 extrinsic 보상 없이 카운트 기반의 목표를 통해 세계를 탐색합니다.

- **Performance Highlights**: MEAD 모델은 2D crafting 게임 및 MiniHack 환경에서 실험적으로 우수한 성과를 보였으며, 이전의 최첨단 저수준 방법들과 비교해显著 향상된 성능과 더 나은 샘플 효율성을 보여주었습니다. 또한, 새로운 환경으로의 제로샷(zero-shot) 및 몇 번의 샷(few-shot) 전이 능력에도 뛰어난 성과를 기록하였습니다.



### Great Memory, Shallow Reasoning: Limits of $k$NN-LMs (https://arxiv.org/abs/2408.11815)
- **What's New**: 이번 연구는 $K$-nearest neighbor language models ($k$NN-LMs)이 정보 회상을 강화하는 것에서 실제 작업 성능으로 이어지는지를 철저히 평가하였습니다. 또한, 이 모델들이 기억이 중요한 작업에서는 우수한 성능을 보이지만, 복잡한 추론 작업에서는 저조한 결과를 보임을 입증하였습니다.

- **Technical Details**: $k$NN-LMs는 기존의 언어 모델에 비해 비모수적 접근 방식을 채택하여, 방대한 텍스트 데이터 스토어를 이용해 성능을 향상시키는 방법입니다. 이 모델은 입력 패턴을 인식하고 메모리와 매칭하여 출력 결과를 결정할 수 있는 간단한 작업에서 높은 정확도를 보입니다.

- **Performance Highlights**: 연구 결과, $k$NN-LMs는 메모리 집약적인 작업에 강한 성능을 발휘하였으나, 추론 능력을 필요로 하는 작업에서는 성능이 떨어지며, 심지어 정보가 완벽하게 회수되더라도 올바른 답변을 찾지 못하는 경우가 많았습니다.



### Approaching Deep Learning through the Spectral Dynamics of Weights (https://arxiv.org/abs/2408.11804)
- **What's New**: 본 논문은 weight의 spectral dynamics에 중점을 둔 경험적 접근 방식을 제안하여 딥러닝의 여러 현상을 통합하고 설명합니다. 다양한 실험에서 관찰된 일관된 편향과 weight decay가 이러한 편향을 증대시키는 방식을 조명합니다.

- **Technical Details**: 이론적으로 설명된 기존 스펙트럼 동역학(spectral dynamics)은 rank minimization과 관련이 있으며, weight matrix의 singular value와 vector의 행동을 분석합니다. 실험을 통해 ConvNets, UNets, LSTMs, Transformers 모델에서의 고찰을 포함합니다.

- **Performance Highlights**: 본 연구는 일반화(generalization)와 관련된 현상을 스펙트럼 동역학 관점에서 이해함으로써, 효과적인 rank minimization과 weight decay 간의 연결고리를 드러냅니다. 이를 통하여 메모리화된 네트워크와 일반화 네트워크 간의 차이를 명확하게 구분할 수 있음을 나타냅니다.



### LLM Pruning and Distillation in Practice: The Minitron Approach (https://arxiv.org/abs/2408.11796)
- **What's New**: 이번 논문에서는 Llama 3.1 8B 모델과 Mistral NeMo 12B 모델을 각각 4B와 8B 파라미터로 압축하기 위한 포괄적인 방법을 제시합니다. 프루닝(pruning)과 디스틸레이션(distillation) 기법을 사용하여 이룬 결과를 다양한 벤치마크에서 평가합니다.

- **Technical Details**: 프루닝 전략으로는 깊이 프루닝(depth pruning)과 joint hidden/attention/MLP(width pruning) 방식이 사용되었습니다. 모델의 정확도를 회복하기 위해 디스틸레이션 방법을 적용하며, teacher correction 단계에서 원본 데이터에 접근할 수 없는 상황에서도 교사 모델을 미세 조정(fine-tune)합니다. 또한, 구조적 프루닝(structured pruning) 기법을 사용하며, 활성화(activation)에 기반한 중요도 추정 전략(importance estimation)을 통해 각 레이어의 중요도를 평가합니다.

- **Performance Highlights**: MN-Minitron-8B 모델은 기존 Mistral NeMo 12B 모델에 비해 평균 1.2배 빠른 추론 성능을 보여주며, Llama-3.1-Minitron-4B 모델은 깊이 및 폭 프루닝 변형 모두 교사 Llama 3.1 8B 모델과의 비교에서 강력한 정확도를 발휘합니다.



### Timeline and Boundary Guided Diffusion Network for Video Shadow Detection (https://arxiv.org/abs/2408.11785)
Comments:
          ACM MM2024

- **What's New**: 이번 논문에서는 Timeline and Boundary Guided Diffusion (TBGDiff) 네트워크를 통해 비디오 그림자 감지(Video Shadow Detection, VSD)를 수행합니다. 이는 기존 방식의 비효율적인 시간적 학습 문제를 해결하고 그림자의 경계(boundary)를 고려하여 성능을 개선합니다.

- **Technical Details**: TBGDiff 네트워크는 과거와 미래의 시간적 안내와 경계 정보를 동시에 활용합니다. Dual Scale Aggregation (DSA) 모듈은 긴 시간과 짧은 시간의 프레임 간의 유사성을 재고하여 시간적 이해를 향상시킵니다. 또한, Shadow Boundary Aware Attention (SBAA)를 통해 그림자의 특징을 포착할 수 있는 경계 정보를 이용합니다. Diffusion 모델을 도입하여 Space-Time Encoded Embedding (STEE)을 적용, 시간적 안내를 체계적으로 주입하여 그림자 감지를 수행합니다.

- **Performance Highlights**: TBGDiff는 기존 최첨단 방법들을 능가하는 성능을 보여주며, 제안한 구성 요소의 효과성을 검증합니다. 또한, 코드와 결과를 공개하여 연구자들이 손쉽게 접근할 수 있도록 할 예정입니다.



### Sum of Squares Circuits (https://arxiv.org/abs/2408.11778)
- **What's New**: 이 논문에서는 기존의 monotonic PCs(positive parameters를 가진 확률 회로)와 squared PCs(부정적인 매개변수를 가진 제곱 회로) 사이의 관계를 더욱 명확히 하고, sum of squares PCs라는 새로운 모델 클래스를 도입하여 그 표현력을 증명했습니다.

- **Technical Details**: 기존 monotonic PCs는 비율 모델만을 나타낼 수 있었으나, Loconte et al. (2024)에 의해 부정 매개변수를 가진 제곱회로(squared PCs)가 도입되었습니다. 이 논문에서는 sum of squares PCs의 성능을 수학적으로 보여줌으로써, 여러 다른 tractable 모델들과의 관계를 맺고, 부정적인 매개변수를 이용한 회로의 표현력을 설명하고 있습니다.

- **Performance Highlights**: sum of squares 회로는 실세계 데이터를 다루는 데 있어 추정 분포(distribution estimation)를 수행하는 데 효과적임을 보여주었으며, 폴리노미얼 크기의 monotonic 회로보다 높은 표현력을 갖고 있습니다.



### D-RMGPT: Robot-assisted collaborative tasks driven by large multimodal models (https://arxiv.org/abs/2408.11761)
- **What's New**: 이 논문은 Collaborative Robots(협동 로봇)과의 상호작용을 위한 새로운 접근법인 Detection-Robot Management GPT (D-RMGPT)를 제안합니다. 이 시스템은 경험이 부족한 조작자들이 마커나 사전 훈련 없이 조립 작업을 수행할 수 있도록 설계되었습니다.

- **Technical Details**: D-RMGPT는 DetGPT-V와 R-ManGPT라는 두 개의 주요 구성 요소로 구성됩니다. DetGPT-V는 GPT-4V(vision)에 기반하여 현재 조립 단계의 이미지와 조립할 구성 요소 목록을 분석하여 주변 환경을 인식합니다. R-ManGPT는 다음 조립할 구성 요소를 계획하고 로봇의 동작을 생성하여 인간 동료에게 전달합니다.

- **Performance Highlights**: D-RMGPT는 장난감 비행기 조립 실험에서 83%의 조립 성공률을 달성하였으며, 경험이 부족한 조작자의 조립 시간을 33% 감소시켰습니다. 이 시스템은 사용자에게 직관적이고 유연하며 광범위한 적용 가능성을 제공합니다.



### SBDet: A Symmetry-Breaking Object Detector via Relaxed Rotation-Equivarianc (https://arxiv.org/abs/2408.11760)
- **What's New**: 본 논문에서는 Group Equivariant Convolution (GConv)의 한계를 극복하기 위해 Relaxed Rotation GConv (R2GConv)를 제안하였습니다. 이 방법은 비대칭성을 처리할 수 있는 새로운 방법론입니다.

- **Technical Details**: R2GConv는 비대칭 구조의 물체를 인식하기 위해 Relaxed Rotation-Equivariant 군 $	extbf{R}_4$를 기반으로 하며, 이를 통해 Symmetry-Breaking에 대처할 수 있습니다. 또한, 이 모델을 기반으로 한 Symmetry-Breaking Object Detector (SBDet)가 개발되었습니다.

- **Performance Highlights**: 실험 결과, R2GConv는 자연 이미지 분류 작업에서 높은 성능을 보였으며, SBDet는 객체 탐지 작업에서 뛰어난 일반화 능력과 강인성을 보여주었습니다.



### Improving the Scan-rescan Precision of AI-based CMR Biomarker Estimation (https://arxiv.org/abs/2408.11754)
Comments:
          11 pages, 3 figures, MICCAI STACOM 2024

- **What's New**: 이번 연구에서는 깊은 학습(deep learning, DL) 방법을 이용하여 심박지표(cardiac biomarkers)의 스캔-재스캔(scan-rescan) 정밀도를 개선하는 파이프라인을 제안합니다. 연구는 특히 심장 기능의 장기적인 분석에서 중요한 심박지표의 일관성에 초점을 맞추고 있습니다.

- **Technical Details**: 연구는 92명의 건강한 자원자로부터 얻은 184개의 스캔-재스캔 영상 자료를 통해 DL 기반의 심박지표 추출 정확도와 정밀도를 평가했습니다. 이미지 보간(image interpolation)과 분할 보간(segmentation interpolation) 두 가지 접근 방식을 비교하여 양질의 심박지표를 추출하는 방법을 제시합니다.

- **Performance Highlights**: 두 가지 방법 모두 심박지표의 Bland-Altman 스캔-재스캔 신뢰 구간을 줄이는 데 성공했으며, 결과적으로 심박지표의 정밀도가 향상되었습니다. 이는 장기적인 심장 기능 분석을 위한 자동화된 도구의 필요성을 강조합니다.



### Open-Ended 3D Point Cloud Instance Segmentation (https://arxiv.org/abs/2408.11747)
- **What's New**: 새로운 연구는 기존의 Open-Vocab 3D Instance Segmentation (OV-3DIS) 기법의 한계를 극복하기 위해 Open-Ended 3D Instance Segmentation (OE-3DIS) 문제를 제안하였으며, 사전 정의된 클래스 이름 없이도 객체 분할이 가능하다.

- **Technical Details**: OE-3DIS는 3D 포인트 클라우드와 RGBD 시퀀스를 입력으로 받아 사전 정의된 라벨 없이 클래스 이름과 함께 3D 마스크를 생성한다. 이 방법은 Multimodal Large Language Models (MLLMs)를 활용하여 포인트와 비주얼 토큰을 3D로 복원하는 방식으로 수행된다.

- **Performance Highlights**: OE-3DIS 접근 방식은 ScanNet200과 ScanNet++ 데이터셋에서 기존 최첨단 방법인 Open3DIS보다 뛰어난 성능을 보였으며, 특히 ScanNet++에서는 18.4 AP를 기록하며 Open3DIS의 13.1 AP를 크게 초과하였다.



### FocusLLM: Scaling LLM's Context by Parallel Decoding (https://arxiv.org/abs/2408.11745)
- **What's New**: FocusLLM은 긴 문맥을 처리할 수 있는 LLM의 새로운 프레임워크로, 이전 방법보다 낮은 훈련 비용으로 고성능을 발휘합니다. 이 모델은 긴 텍스트를 짧은 청크로 나누어 해당 청크에 대한 로컬 컨텍스트를 추가하고 병렬 디코딩 메커니즘을 통해 핵심 정보를 추출하여 통합합니다.

- **Technical Details**: FocusLLM은 기존 디코더 전용 LLM 아키텍처를 기반으로 하며, 원래 모델의 매개변수는 고정되어 강력한 일반화 능력을 유지합니다. 새로운 훈련 가능한 매개변수를 추가하여 병렬 디코딩의 결과를 집계할 수 있습니다. 긴 문장을 메모리 토큰과 로컬 토큰으로 나누고, 각 청크 별로 병렬 디코딩을 수행합니다.

- **Performance Highlights**: FocusLLM은 8K 입력 길이로 훈련되어 128K 이상의 긴 문서에서도 낮은 perplexity를 유지합니다. Longbench와 ∞-Bench에서의 평가 결과, FocusLLM은 기존 상한 모델들을 초과하는 성능을 보이며 긴 문서를 처리하는 데 뛰어난 능력을 발휘합니다.



### CluMo: Cluster-based Modality Fusion Prompt for Continual Learning in Visual Question Answering (https://arxiv.org/abs/2408.11742)
- **What's New**: 이번 논문에서는 대형 비전-언어 모델(VLMs)이 여러 작업을 연속적으로 처리하는 데 있어서의 한계를 극복하기 위해, 새로운 클러스터 기반 모달리티 융합 프롬프트(CluMo) 방법을 제안합니다. 이 방법은 프롬프트 기반의 지속적 학습(Continual Learning, CL) 접근을 통해 일반화 성능을 향상시키고, 이전에 학습한 작업의 지식을 잊는 문제를 최소화하려고 합니다.

- **Technical Details**: 각 비주얼 프롬프트 키와 텍스트 프롬프트 키에 연결된 키-키 프롬프트 쌍을 설계하며, 두 단계의 학습 전략을 채택합니다. 첫 번째 단계에서는 K-평균 클러스터링 알고리즘을 통해 단일 모달 키를 학습하여 최적의 프롬프트 선택을 지원합니다. 두 번째 단계에서는 프롬프트 키가 고정되고 선택된 프롬프트가 입력에 첨부되어 VLM을 훈련합니다.

- **Performance Highlights**: 실험 결과, 제안된 CluMo 방법은 최신 기술(SOTA) 성능을 달성하여 기존의 CL 방법들과 비교해 뛰어난 성과를 나타냈습니다. 이 방법은 VQA(Visual Question Answering) 작업에서 비전과 언어의 입력을 통합하여 지속적 학습 환경에서도 효과적으로 적용될 수 있음을 입증합니다.



### Efficient Detection of Toxic Prompts in Large Language Models (https://arxiv.org/abs/2408.11727)
Comments:
          Accepted by the 39th IEEE/ACM International Conference on Automated Software Engineering (ASE 2024)

- **What's New**: ToxicDetector는 경량(greybox) 방식으로 설계된 독성 프롬프트 탐지 방법으로, 기존 방법의 한계점을 극복하여 LLM에서 독성 프롬프트를 효율적으로 탐지합니다.

- **Technical Details**: ToxicDetector는 LLM을 활용하여 독성 개념 프롬프트를 생성하고, 임베딩 벡터(embedding vectors)를 통해 피쳐 벡터(feature vectors)를 형성합니다. 그 후 MLP(Multi-Layer Perceptron) 분류기를 사용하여 프롬프트를 분류합니다.

- **Performance Highlights**: ToxicDetector는 LLama 모델, Gemma-2 및 다양한 데이터셋을 평가한 결과 96.39%의 높은 정확도와 2.00%의 낮은 허위 긍정률(false positive rate)을 기록하였으며, 0.0780초의 신속한 처리 시간을 자랑하여 실시간 응용 프로그램에 매우 적합합니다.



### Iterative Object Count Optimization for Text-to-image Diffusion Models (https://arxiv.org/abs/2408.11721)
Comments:
          Pre-print

- **What's New**: 이번 연구는 text-to-image 모델에서 특정 개수의 객체를 정확히 생성하는 것을 목표로 합니다. 기존 모델들은 이미지-텍스트 쌍으로 학습하면서 개수 세기에서 어려움을 겪고 있었고, 이를 해결하기 위한 새로운 방법을 제안합니다.

- **Technical Details**: 연구에서는 counting loss를 최적화하여 생성된 이미지의 정확도를 높이는 방식으로 접근합니다. 이 과정에서 객체 수를 정확히 나타내기 위해 gradient descent를 활용하여 counting token의 embedding을 조정합니다. 이와 함께, object의 뷰포인트에 따라 scaling hyperparameter를 동적으로 조정하는 iterated online training 방식을 채택합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 다양한 객체의 정확한 생성을 보여주었으며, 기존 이미지 생성 방법들과 비교했을 때 품질과 정확도에서 상당한 향상을 보였습니다.



### Leveraging Large Language Models for Enhancing the Understandability of Generated Unit Tests (https://arxiv.org/abs/2408.11710)
Comments:
          **Note:** This paper has been accepted for presentation at the 47th International Conference on Software Engineering (ICSE 2025) - Research Track

- **What's New**: UTGen은 검색 기반 소프트웨어 테스트 방법과 대형 언어 모델(Large Language Models)을 결합하여 자동 생성된 단위 테스트 케이스의 이해도를 향상시킵니다.

- **Technical Details**: UTGen은 EvoSuite를 기반으로 하여 테스트 데이터의 맥락화, 식별자 명명 개선 및 설명적 주석 추가를 통해 생성된 테스트 케이스의 이해도를 향상시킵니다. 연구는 32명의 참가자를 대상으로 통제된 실험을 통해 통합 접근 방식의 효과를 연구하였습니다.

- **Performance Highlights**: UTGen 테스트 케이스를 사용하는 참가자들이 기준 테스트 케이스 대비 버그를 33% 더 많이 수정하고, 20% 적은 시간 내에 작업을 수행할 수 있음을 관찰하였습니다.



### 5G NR PRACH Detection with Convolutional Neural Networks (CNN): Overcoming Cell Interference Challenges (https://arxiv.org/abs/2408.11659)
- **What's New**: 이번 논문에서는 CNN(Convolutional Neural Networks)을 이용하여 5G-NR(5G New Radio) 네트워크에서의 간섭 탐지를 위한 새로운 접근 방식을 제안합니다. 이 연구는 5G 네트워크에서의 높은 품질 서비스 제공을 저해하는 간섭 문제를 해결하고자 합니다.

- **Technical Details**: 제안하는 CNN 기반 모델은 다양한 간섭 시나리오 속에서 PRACH(Physical Random Access Channel) 시퀀스를 검출하도록 설계되었습니다. 모델 훈련을 위해 통제된 간섭 조건에서 생성된 시뮬레이션 PRACH 신호의 포괄적인 데이터셋을 사용하였고, CNN의 공간적 및 시간적 특성을 활용하여 검출 정확도와 강건성을 향상시켰습니다.

- **Performance Highlights**: 실험 결과, 제안하는 CNN 기반 접근 방식이 기존의 PRACH 검출 방법보다 정확도, 정밀도, 재현율 및 F1-score에서 더 나은 성능을 보임을 입증하였습니다. 이 연구는 AI/ML 기술이 5G 네트워크의 간섭 관리에 어떻게 기여할 수 있는지를 보여주며, 네트워크 성능 최적화 및 신뢰성을 높이기 위한 미래 연구의 기초를 제공합니다.



### CIPHER: Cybersecurity Intelligent Penetration-testing Helper for Ethical Researcher (https://arxiv.org/abs/2408.11650)
Comments:
          28 pages, github available

- **What's New**: CIPHER는 윤리적 연구자를 위한 사이버 보안 침투 테스트 도우미로, 300개 이상의 고품질 침투 테스트 문서를 사용하여 훈련된 대형 언어 모델입니다. 기존의 침투 테스트 방법론의 한계를 해결하기 위해 Findings, Action, Reasoning, and Results (FARR) Flow 증강 방법을 통하여 전자동 펜테스팅 시뮬레이션 벤치마크를 구축합니다.

- **Technical Details**: CIPHER는 300개 이상의 침투 테스트 사례, 해킹 기법 및 오픈 소스 도구 문서로 훈련되었으며, FARR Flow 증강 기법을 통해 문서의 효율성을 높이고 심층적인 침투 테스트 프로세스를 이해할 수 있도록 개발되었습니다. 이 모델은 침투 테스트의 고급 기술과 추론 능력을 활용하여 사용자에게 도움을 제공합니다.

- **Performance Highlights**: CIPHER는 유사한 크기의 다른 오픈 소스 침투 테스트 모델들 및 Llama 3 70B, Qwen 1.5 72B와 같은 최신 모델들에 비해 뛰어난 정확성을 보여주었으며, 특히 어려운 머신 세팅에서 최고의 성능을 기록했습니다. 이는 CIPHER가 전통적인 침투 테스트의 실용적 기술을 더 잘 반영한다는 것을 의미합니다.



### Video-to-Text Pedestrian Monitoring (VTPM): Leveraging Computer Vision and Large Language Models for Privacy-Preserve Pedestrian Activity Monitoring at Intersections (https://arxiv.org/abs/2408.11649)
- **What's New**: 본 논문에서는 보행자 모니터링을 위해 비디오를 텍스트로 변환하는 Video-to-Text Pedestrian Monitoring (VTPM) 시스템을 소개합니다. 이 시스템은 교차로에서 보행자의 움직임을 모니터링하고 실시간으로 교통 신호 및 날씨 정보를 포함한 텍스트 보고서를 생성합니다.

- **Technical Details**: VTPM은 보행자 감지 및 추적을 위한 컴퓨터 비전 모델을 사용하며, 비디오 프레임 당 0.05초의 지연(latency)을 달성합니다. 또한, 90.2%의 정확도로 교차 위반을 감지하며, Phi-3 mini-4k가 장착되어 있어 보행자 활동에 대한 실시간 텍스트 보고서를 생성합니다. 이러한 보고서는 안전 문제를 명시하며, 0.33초의 지연으로 날씨가 보행자 행동에 미치는 영향도 분석합니다.

- **Performance Highlights**: 제안된 VTPM은 비디오로부터의 메모리 사용을 253백만 퍼센트까지 절감할 수 있으며, 개인 정보를 보호합니다. 또한, 생성된 텍스트 보고서를 통해 교차로에서의 보행자 안전에 대한 신뢰할 수 있는 역사적 분석이 가능하여 패턴 및 안전 위반 사건을 효과적으로 감지할 수 있습니다.



### Data-driven Modeling of Combined Sewer Systems for Urban Sustainability: An Empirical Evaluation (https://arxiv.org/abs/2408.11619)
Comments:
          12 pages, 4 figures, accepted at 47th German Conference on Artificial Intelligence, Wuerzburg 2024

- **What's New**: 기후 변화가 심화됨에 따라 도시 기반 시설을 관리하는 데 있어 비상 상황 예측이 더욱 중요해졌습니다. 본 연구에서는 Deep Learning (DL) 모델을 활용하여 Combined Sewer Systems (CSS)의 효율적인 물리적 모델링을 시도하였습니다.

- **Technical Details**: 이 연구에서는 LSTM (Long Short-Term Memory)과 TFT (Temporal Fusion Transformer) 등 최첨단 시계열 모델을 비교 평가하였습니다. 총 3년 간의 실제 측정 데이터를 기반으로 하여 글로벌 모델과 로컬 모델의 성능을 분석했습니다. 연구의 주요 목표는 네트워크 중단 상황에서도 예측 정확성을 유지할 수 있는 DL 모델의 가능성을 조사하는 것이었습니다.

- **Performance Highlights**: LSTM 모델이 평균 제곱 오차 (MSE)가 가장 낮아 가장 우수한 성능을 보였으며, TFT는 다양한 조건에서도 일관된 예측을 제공했습니다. 연구 결과, 글로벌 모델이 일반적으로 로컬 모델보다 MSE 측면에서 우수하다는 것이 확인되었으나, 로컬 모델은 외부 데이터가 없을 때 유리한 대안으로 작용할 수 있다는 것도 밝혀졌습니다.



### Xinyu: An Efficient LLM-based System for Commentary Generation (https://arxiv.org/abs/2408.11609)
- **What's New**: 이 논문은 중국어 주석 생성을 지원하기 위해 Xinyu라는 LLM(large language model) 기반 시스템을 소개합니다. 이 시스템은 주석 생성을 위한 기본 요구사항과 고급 요구사항을 충족시키기 위해 여러 단계를 나누어 생성 프로세스를 개선했습니다.

- **Technical Details**: 기본 요구사항을 충족하기 위해, 우리는 생성 프로세스를 순차적 단계로 분해하고, 각 단계에 대해 targeted supervised fine-tuning (SFT) 전략을 제안했습니다. 고급 요구사항을 위해, 우리는 주장을 평가하기 위한 argument ranking 모델을 도입하고, 최신 사건과 고전 도서를 포함하는 포괄적인 증거 데이터베이스를 구축했습니다. 또한, RAG(retrieval augmented generation) 기술을 활용하여 신뢰할 수 있는 증거를 생성합니다.

- **Performance Highlights**: Xinyu 시스템을 사용함으로써 주석 생성의 효율성이 크게 향상되었습니다. 평균 생성 시간이 4시간에서 20분으로 줄어들었으나, 생성된 주석의 품질은 유지되었습니다.



### Networked Communication for Mean-Field Games with Function Approximation and Empirical Mean-Field Estimation (https://arxiv.org/abs/2408.11607)
- **What's New**: 본 논문은 분산된 에이전트들이 단일 비 에피소드(run) 동안 Mean-Field Games (MFG)의 균형을 학습할 수 있는 알고리즘을 제시합니다. 이러한 모델에서는 에이전트의 상태 공간이 작지 않더라도 함수 근사(function approximation)를 도입하여 MFG 환경의 제약을 극복했습니다.

- **Technical Details**: MFG 프레임워크는 많은 에이전트를 효율적으로 모델링할 수 있으며, 이는 무한에 가깝고 대칭적이며 익명의(agent) 에이전트들로 구성된 경우의 의미 필드(mean-field) 분포를 사용하여 에이전트가 상호작용하도록 설계되었습니다. 우리는 Munchausen Online Mirror Descent 방식을 사용하여 함수 근사를 도입하고, 이를 통해 각 에이전트가 지역 정보를 바탕으로 전역 분포를 추정할 수 있는 알고리즘을 제시합니다.

- **Performance Highlights**: 실험 결과, 커뮤니케이션 네트워크를 활용하여 분산된 에이전트들이 합동 전략을 통해 훨씬 더 효과적으로 작동함을 보여주었습니다. 정책을 교환하면 독립적인 학습 및 중앙집중 학습보다 더 나은 성능을 발휘하며, 비 에피소드 학습에서도 효과적인 성과를 내는 것으로 나타났습니다.



### Cause-Aware Empathetic Response Generation via Chain-of-Thought Fine-Tuning (https://arxiv.org/abs/2408.11599)
- **What's New**: 이 논문은 감정 생성에서 감정의 원인(reasoning)을 고려한 새로운 접근법을 제안합니다. 기존 연구들은 주로 화자의 감정 레이블에 중심을 두었으나, 이 연구는 감정의 원인 이해가 중요하다는 점을 강조합니다.

- **Technical Details**: 우리는 Chain-of-Thought (CoT) 프롬프트를 사용하여 대형 언어 모델(Large Language Models, LLMs)에서 감정과 원인을 통합하는 방법을 설계하였습니다. 이 알고리즘은 감정에 대한 반응의 다양성을 개선하고 내부 지식과 외부 지식 간의 갈등을 완화하는 데 기여합니다. 또한, COMET에서 제공하는 외부 지식을 프롬프트에 추가하여 모델의 성능을 향상시킵니다.

- **Performance Highlights**: LLaMA-7b에서 실험 결과, 우리의 접근법은 자동 및 인간 평가 모두에서 기존 방법 대비 우수한 성능을 보여주었고, ChatGPT보다도 더 공감적인 반응을 생성할 수 있음을 입증하였습니다.



### Active learning for efficient data selection in radio-signal based positioning via deep learning (https://arxiv.org/abs/2408.11592)
Comments:
          Submitted to Electronics Letters

- **What's New**: 본 논문에서는 무선 신호를 기반으로 사용자 장비(UE) 위치 추정 문제를 심층 학습(Deep Learning)을 통해 다루고 있습니다. 데이터 수집의 통신 오버헤드를 줄이기 위해, 적절한 위치 선택을 통한 능동 학습(Active Learning) 접근 방식을 제안합니다.

- **Technical Details**: 우리는 미리 존재하는 데이터셋을 기반으로 UE의 위치를 효율적으로 추론하기 위한 새로운 데이터 선택 기법을 제안합니다. 이 기법은 첫 번째 학습 단계 후 모델 개선을 위해 가장 관련성 높은 교육 위치를 선택하도록 설계되었습니다. 제안된 방법은 '제니다 접근법(Genie Approach)'으로, 난수 선택에 비해 성능을 크게 향상시키며, 약 10%의 후보 위치 선택을 통해 50%의 성과를 달성했습니다.

- **Performance Highlights**: 제안된 방법은 무선 신호 기반 위치 결정 문제에서 데이터 수집을 강하게 제약할 수 있는 모든 분야에 적용 가능합니다. 실제 알고리즘은 제니다 접근법의 약 50%의 성과를 달성하며, 데이터 사용량을 20%로 줄이면서도 유사한 성능을 보여주었습니다.



### Differentiating Choices via Commonality for Multiple-Choice Question Answering (https://arxiv.org/abs/2408.11554)
Comments:
          9 pages, accepted to ECAI 2024

- **What's New**: 본 논문에서는 선택지가 모두 질문과 관련이 있고 의미적으로 유사한 멀티 초이스 질문 응답(MCQA) 문제를 해결하기 위한 새로운 모델(DCQA)을 제안합니다. 기존의 MCQA 모델은 선택지를 개별적으로 평가하며, 다른 선택지가 제공하는 문맥을 무시했음을 지적합니다.

- **Technical Details**: DCQA 모델은 질문에 대한 선택지의 토큰 수준의 주의(attention)를 캡처하고, 모든 선택지들이 주목하는 질문의 토큰(즉, 공통적 요소)과 특정 선택지에 의해 주목되는 토큰(즉, 미세한 차이)을 분리합니다. 이를 통해 세부적인 문맥 정보를 이용해 선택지를 효과적으로 구분합니다.

- **Performance Highlights**: 연구에서 제안하는 DCQA 모델은 5개의 MCQA 벤치마크에서 일관되게 기존 모델들을 초과하는 성능을 보였습니다. 또한, 사례 연구를 통해 모델이 선택지 간의 공통점을 효과적으로 포착하고 질문 내에서 구별되는 단서를 식별하는 방식이 어떻게 MCQA 성능을 향상시키는지를 보여주었습니다.



### Memorization In In-Context Learning (https://arxiv.org/abs/2408.11546)
Comments:
          v1

- **What's New**: 이 연구는 In-context learning (ICL)의 성능 향상 메커니즘을 처음으로 밝혀냈으며, ICL이 기억된 훈련 데이터(memorized training data)를 어떻게 드러내는지를 보여줍니다.

- **Technical Details**: 연구는 ICL의 성능 향상이 zero-shot, few-shot, many-shot의 다양한 ICL 방식에서의 기억화(meorization)와 어떻게 연관되는지를 탐구합니다. 주요 발견 사항으로는 ICL이 대부분의 경우 zero-shot learning에 비해 기억화를 더 효과적으로 드러낸다는 점과, 레이블이 없는 demonstration이 기억화를 드러내는 데 가장 효과적이라는 점이 포함됩니다.

- **Performance Highlights**: 연구 결과, few-shot 환경에서 기억화가 높은 수준에 도달했을 때 ICL이 성능을 향상시키며, 또한 ICL이 zero-shot learning을 초과하는 경우 성능과 기억화 사이에는 매우 강한 상관관계가 존재한다는 사실이 밝혀졌습니다.



### A Survey of Embodied Learning for Object-Centric Robotic Manipulation (https://arxiv.org/abs/2408.11537)
- **What's New**: 이 논문에서는 물체 중심 로봇 조작을 위한 체화 학습(Embodied Learning)의 최신 발전 상황을 종합적으로 리뷰하고, 세 가지 주요 분야로 분류한다: 1) Embodied Perceptual Learning, 2) Embodied Policy Learning, 3) Embodied Task-Oriented Learning. 또한, 공개 데이터셋, 평가 메트릭, 대표적인 응용 사례, 현재의 도전과제와 향후 연구 방향에 대한 논의를 제공한다.

- **Technical Details**: 체화 학습은 환경과의 물리적 상호작용과 인지적 피드백을 통해 로봇이 학습하는 방법으로, 데이터 기반 머신 러닝과는 대조적이다. 세 가지 유형의 체화 학습 방법은 다음과 같다: 1) Embodied Perceptual Learning은 다양한 데이터 표현을 통해 물체의 자세 및 가능성을 예측, 2) Embodied Policy Learning은 강화 학습 및 모방 학습 기법을 활용하여 로봇의 최적 결정을 생성, 3) Embodied Task-Oriented Learning은 특정 작업 특성에 따라 로봇의 성능을 최적화한다.

- **Performance Highlights**: 이 논문은 최근 발전한 체화 학습 방법론들을 종합하여 물체 중심 로봇 조작의 다양한 하위 분야들을 분석한다. 특히, 이미지 기반, 3D 인식, 촉각 기반 등의 다양한 데이터 표현 방법들을 다루고, 이를 통해 로봇의 조작 정확성을 높이는 데 기여하고자 한다. 더불어, 여러 공개 데이터셋과 평가 메트릭을 제시하며, 현재 직면하고 있는 도전 과제와 미래 연구 방향성에 대한 통찰을 제공한다.



### Scalable Knowledge Refactoring using Constrained Optimisation (https://arxiv.org/abs/2408.11530)
- **What's New**: 이번 논문에서는 기존의 지식 리팩토링(knowledge refactoring) 기법의 한계를 극복하기 위해 제약 최적화(refactoring approach) 기법을 도입했습니다. 주목할 만한 두 가지 아이디어가 있습니다.

- **Technical Details**: 첫 번째 키 아이디어는 문제를 규칙이 아닌 리터럴(literal)을 기반으로 하는 결정 변수(decision variable)로 인코딩하는 것입니다. 두 번째 아이디어는 선형으로 발명된 규칙에 초점을 맞추는 것입니다.

- **Performance Highlights**: 여러 도메인에서의 실험 결과, 본 접근법이 기존의 최첨단(state-of-the-art) 방법보다 더 빠르고 60% 이상 프로그램을 압축할 수 있다는 것을 보여주었습니다.



### The Vizier Gaussian Process Bandit Algorithm (https://arxiv.org/abs/2408.11527)
Comments:
          Google DeepMind Technical Report. Code can be found in this https URL

- **What's New**: Google Vizier는 다년간의 연구 및 사용자 피드백을 통해 알고리즘 개선을 이루어왔으며, Open Source Vizier의 현재 기본 알고리즘에 대한 세부 구현 사항과 디자인 선택 사항을 설명합니다.

- **Technical Details**: Google Vizier는 Gaussian Process Bandit Optimization에 기반을 두고 있으며, Python 전용 오픈 소스 구현을 제공하여 연구 커뮤니티가 사용할 수 있게 되었습니다. 이 알고리즘은 다양한 산업 기준과 비교하여 여러 차원에서 경쟁력 있는 강건성을 보여줍니다.

- **Performance Highlights**: 고차원, 범주형, 배치 및 다목적 최적화에서 Vizier의 경쟁력을 실험을 통해 입증하였으며, 비전통적인 디자인 선택인 영차 제한 진화 최적화 기법의 장점에 대해 토론합니다.



### LARR: Large Language Model Aided Real-time Scene Recommendation with Semantic Understanding (https://arxiv.org/abs/2408.11523)
- **What's New**: 이번 논문에서는 대규모 언어 모델(LLM)을 활용하여 실시간 장면 추천 시스템의 성능을 개선하는 새로운 프레임워크인 LARR(Large Language Model Aided Real-time Scene Recommendation)를 제안합니다. LARR는 추천 도메인 지식을 LLM에 주입하고, LLM의 출력을 집계하는 인코더를 통해 실제 장면 정보를 구성하여 CTR(Click-Through Rate) 모델링의 효율성을 향상시킵니다.

- **Technical Details**: LARR 모델은 세 가지 단계로 진행됩니다. 첫째, 추천 데이터로부터 구축된 코퍼스를 기반으로 LLM을 지속적으로 사전 훈련합니다. 둘째, 샘플 구축 전략을 사용하여 대조 학습을 통해 LLM을 미세 조정합니다. 마지막으로, LLM의 다양한 장면 특성에 대한 출력을 인코더를 통해 집계하여 협력 신호와 일치하도록 합니다.

- **Performance Highlights**: Meituan Waimai의 음식 배달 데이터셋을 활용한 오프라인 및 온라인 실험을 통해 LARR가 실시간 장면의 의미 정보를 완전히 이해하고 다중 모달 특성을 효과적으로 통합하여 추천 결과를 향상시켰음을 확인했습니다.



### Last-Iterate Convergence of General Parameterized Policies in Constrained MDPs (https://arxiv.org/abs/2408.11513)
- **What's New**: 이번 논문에서는 제약이 있는 마르코프 결정 프로세스(Constrained Markov Decision Process, CMDP)를 위한 새로운 알고리즘인 PDR-ANPG(Primal-Dual based Regularized Accelerated Natural Policy Gradient)를 제안합니다. 이 알고리즘은 엔트로피와 제곱 정규화기를 사용하여 표준화된 정책을 학습하고, 가장 높은 단계의 최적성을 달성할 수 있는 보장된 새롭고 개선된 방법입니다.

- **Technical Details**: PDR-ANPG는 일반 매개변수화된 정책 클래스에서 전환 호환성 근사 오차를 반영하여 $	ilde{	heta}(	ilde{O}(	heta^{-2}))$의 샘플 복잡도를 요구합니다. 상태 공간의 크기와 독립된 고정된 수의 매개변수를 사용하는 이 알고리즘은 무한 상태 공간에서도 적용 가능합니다. 또한 완전 정책 장에서는 샘플 복잡도가 $	ilde{	heta}(	heta^{-4})$로 줄어들어 성능이 향상됩니다.

- **Performance Highlights**: PDR-ANPG는 마지막 반복에서 $	heta$ 최적성 간극과 $	heta$ 제약 위반을 달성하여 CMDP에 대한 최신 연구 결과를 능가하는 성능을 보여주었습니다. 이 알고리즘은 안전-critical 응용 프로그램에서 적합한 것으로 평가되며, 평균적인 방식으로 우려되는 제약 위반을 극복할 수 있는 구조적 장점을 제공합니다.



### Using Part-based Representations for Explainable Deep Reinforcement Learning (https://arxiv.org/abs/2408.11455)
- **What's New**: 이 논문에서는 Deep Reinforcement Learning (RL)에서 actor 모델의 비음수 (non-negative) 훈련 접근 방식을 제안하여, 비음수 제약 조건을 준수하면서 해석 가능성 (interpretability)을 향상시키는 방법에 대해 다룹니다.

- **Technical Details**: 제안된 방법은 비음수 초기화 기법과 수정된 부호 보존 훈련 방법을 포함합니다. 주로 Proximal Policy Optimization (PPO) 알고리즘을 바탕으로 하여, 에이전트의 행동을 만드는 actor 네트워크만 비음수 방식으로 훈련됩니다. 이는 canceling neurons 문제를 줄이고, 훈련 과정의 안정성을 증가시킵니다. 또한, 이 방법은 최대 우도 (maximum likelihood) 원리를 활용하여 정책 업데이트를 위한 제약 조건을 설정합니다.

- **Performance Highlights**: 제안된 방법은 Cartpole 벤치마크 테스트에서 효과성을 입증했습니다. 이 기술은 기본적으로 Deep RL 접근 방식에서 쉽게 적용 가능하며, 비음수 파트 기반 표현 방식으로 본질적으로 설명 가능한 모델의 효율적인 훈련을 가능하게 합니다.



### Lookism: The overlooked bias in computer vision (https://arxiv.org/abs/2408.11448)
Comments:
          Paper accepted at the ECCV 2024 workshop named "Fairness and ethics towards transparent AI: facing the chalLEnge through model Debiasing (FAILED)", this https URL

- **What's New**: 이 논문은 컴퓨터 비전 모델 내에서 lookism(외모편향)의 체계적인 연구 필요성을 강조합니다. 특히, 외모에 기반한 차별이 AI 기술의 공정성과 포괄성을 해칠 수 있는 잠재력을 가지고 있음을 경고하며, 이를 해결하기 위한 학제 간 접근 방식을 촉구하고 있습니다.

- **Technical Details**: lookism은 사회적 아름다움 기준과 인지 편향에 뿌리를 두고 있으며, 이는 컴퓨터 비전 시스템에서 불공정한 대우와 해로운 고정관념을 강화할 수 있습니다. 이 논문은 외모에 따른 차별을 탐구하고 이를 완화하기 위한 전략을 개발하는 것을 목표로 합니다.

- **Performance Highlights**: 다양한 연령대와 인종 배경을 가진 얼굴 이미지 데이터셋을 사용하여 beauty filter의 효과를 분석한 결과, beauty filter가 대다수 이미지의 인지된 매력을 증가시키는 것으로 나타났습니다. 그러나, 여성의 매력 점수는 높아졌지만, 지능 점수는 남성 이미지에서 더 높아져 성별 편향을 악화시키는 것으로 분석되었습니다.



### Towards Aligned Data Removal via Twin Machine Unlearning (https://arxiv.org/abs/2408.11433)
- **What's New**: 최근 개인정보 보호 규정으로 인해 머신 언러닝(machine unlearning) 기술이 발전하였으며, 이는 이미 학습된 기계 학습(ML) 모델에서 데이터를 제거할 수 있게 해줍니다. 본 논문에서는 Twin Machine Unlearning (TMU) 접근 방식을 소개하며, 이를 통해 모델의 정확도를 유지하면서도 데이터 제거를 정교하게 수행할 수 있습니다.

- **Technical Details**: TMU 접근 방식은 원래의 언러닝 문제에 대응하는 쌍의 언러닝 문제를 설정합니다. 이를 통해 훈련된 일반화 레이블 예측기를 원래 문제로 전이시켜 정렬된 데이터 제거를 용이하게 합니다. 이 과정에서 'easy' 샘플과 'hard' 샘플에 대한 일반화 레이블을 예측하여, 하드 샘플에 대한 분류 정확도를 감소시키고 이지 샘플에 대한 정확도를 유지합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법이 언러닝된 모델과 골드 모델 간의 정렬을 크게 향상시킬 뿐 아니라 모델의 정확도를 유지하면서도 데이터를 효과적으로 제거할 수 있음을 보여줍니다.



### Diagnosing and Remedying Knowledge Deficiencies in LLMs via Label-free Curricular Meaningful Learning (https://arxiv.org/abs/2408.11431)
Comments:
          Under Review

- **What's New**: 본 논문에서는 라벨이 없는 설정에서 LLM의 지식 결핍을 진단하고, 이를 해결하기 위한 학습 프레임워크인 LaMer를 제안합니다. LaMer는 상대 엔트로피(relative entropy)를 활용하여 LLM의 지식 결핍을 자동적으로 진단하며, 이는 기존의 라벨 데이터에 의존하지 않습니다.

- **Technical Details**: LaMer 프레임워크는 두 가지 주요 단계로 구성됩니다. 첫째, LLM의 지식 결핍을 진단하기 위해 상대 엔트로피를 사용합니다. 둘째, 커리큘럼 유의미 학습(curricular meaningful learning)을 적용해 지식 결핍을 점진적으로 해결합니다. 이를 통해 데이터 보강(augmentation)을 적응적으로 수행하며, 다양한 시나리오에서 결핍의 심각도에 따라 해결 전략을 설계합니다.

- **Performance Highlights**: LaMer는 40%의 훈련 데이터만으로도 여러 LLM의 지식 결핍을 효과적으로 진단하고 개선하는 성과를 보여줍니다. 7개의 OOD(Out-of-Distribution) 추론 및 언어 이해 벤치마크에서 기존의 기법들과 비교했을 때, LaMer는 더욱 효율적이고 효과적으로 LLMs의 성능을 향상시킬 수 있습니다.



### Long-Range Vision-Based UAV-assisted Localization for Unmanned Surface Vehicles (https://arxiv.org/abs/2408.11429)
- **What's New**: 이 논문에서는 해양 환경에서 GPS에 의존하지 않고 무인 수상 차량(USV)의 지역화를 지원하기 위한 새로운 방법을 제시합니다. 이 방법은 UAV(무인 항공기)를 활용하여 해안선에 따라 비행하며, 딥 러닝(deep learning)을 기반으로 USV를 지속적으로 추적하고 감지하는 것을 포함합니다.

- **Technical Details**: 제안된 방법에서는 UAV의 카메라를 사용하여 USV의 위치를 삼각 측량(triangulation) 기법을 통해 추정합니다. 또한, UAV의 카메라 각도를 USV와 이미지 중심 간의 픽셀 오류에 따라 조정하여 정확성을 향상시키고, 시각적 측정을 EKF(Extended Kalman Filter)에 통합하여 강력한 상태 추정을 수행합니다. 다각적인 로봇 인터페이스를 통해 USV와 UAV 간의 커뮤니케이션을 지원합니다.

- **Performance Highlights**: 본 연구의 효율성은 'Muhammad Bin Zayed International Robotic Challenge (MBZIRC-2024)' 대회에서 실제 해양 환경에서의 실험을 통해 입증되었습니다. 이 논문은 GPS가 제한된 환경에서 USV의 지역화 문제를 해결하는 데 기여하며, 실험 결과는 제안한 방법이 USV를 정확하게 지역화할 수 있는 가능성을 보여줍니다.



### Towards "Differential AI Psychology" and in-context Value-driven Statement Alignment with Moral Foundations Theory (https://arxiv.org/abs/2408.11415)
Comments:
          8 pages, 6 tables

- **What's New**: 이 연구는 개인화된 언어 모델과 설문 참여자 간의 정렬을 조사하고, 윤리적 원칙에 기반한 대화형 모델을 생성할 수 있는 새로운 방법론을 제안합니다.

- **Technical Details**: 가장 진보된 통계적 언어 모델을 개인화된 정치 성향으로 조정하여, 입력된 설문지에 대한 행동을 분석합니다. Moral Foundation Theory (MFT)를 사용하여 각 정치적 정향의 다양한 사람들을 모델링합니다.

- **Performance Highlights**: 적응된 모델들이 정치 이념에 대한 설문 결과를 잘 표현하지 못하며, 언어 모델을 사용하여 사회적 상호작용을 모방하기 위해서는 의미 있는 개선이 필요합니다.



### Revisiting FunnyBirds evaluation framework for prototypical parts networks (https://arxiv.org/abs/2408.11401)
Comments:
          Published at 2nd XAI World Conference

- **What's New**: 이 논문에서는 프로토타입 부분 네트워크(ProtoPNet)의 설명 품질 평가를 위해 새로운 시각화 방법인 similarity maps(유사성 맵)를 제안합니다. 기존의 범위 박스(bounding boxes) 사용 방식과 비교하여, similarity maps가 ProtoPNet의 본질에 더 잘 부합한다는 점을 강조합니다.

- **Technical Details**: 논문의 주요 초점은 FunnyBirds 벤치마크 내에서 프로토타입 부분 네트워크의 설명을 범위 박스 대신 similarity maps로 평가하는 것입니다. 연구자들은 다양한 메트릭 점수를 비교하여 similarity maps 기반 설명이 더 정확한 평가 결과를 도출한다는 것을 밝혔습니다.

- **Performance Highlights**: 본 연구의 결과는 similarity maps의 사용이 ProtoPNet 모델과의 직관적 일치를 증가시킨다는 것을 보여줍니다. 이는 측정된 메트릭 점수의 차이로 입증되며, 커뮤니티 내에서 설명의 신뢰할 수 있는 비교가 가능하도록 합니다.



### Data-Centric Machine Learning for Earth Observation: Necessary and Sufficient Features (https://arxiv.org/abs/2408.11384)
Comments:
          Accepted at MACLEAN workshop, ECML/PKDD 2024

- **What's New**: 본 연구는 다중 모달(temporal multimodal) 공간 데이터를 활용한 기계 학습 모델의 성능을 향상시키기 위한 데이터 중심 관점을 제안합니다. 기존 모델 아키텍처 위주의 연구가 다소 포화상태에 도달한 가운데, 모델의 최적 성능 달성에 필수적인 특성을 규명하고 이를 활용하는 데 중점을 두고 있습니다.

- **Technical Details**: 이 연구는 XAI(Explainable AI) 기법을 사용하여 다변량 시계열 데이터에서 효율적인 특성 선택(feature selection)을 진행합니다. 특히, ROAR(RemOve And Retrain) 방법을 활용해 중요도 기준으로 특성을 삭제하고, 삭제 후 모델을 재훈련하여 최적의 예측 특성 집합을 도출합니다. 또한, 3개의 다양한 EO 데이터셋에서 이 접근법을 검증합니다.

- **Performance Highlights**: 연구에 따르면, 특정 데이터셋에서는 전체 시계열의 20% 미만만으로도 최적의 정확도에 도달할 수 있으며, 다른 데이터셋에서는 단일 모달리티의 단일 밴드 시계열만으로도 충분한 결과를 얻을 수 있음을 보여줍니다.



### Reflex-Based Open-Vocabulary Navigation without Prior Knowledge Using Omnidirectional Camera and Multiple Vision-Language Models (https://arxiv.org/abs/2408.11380)
Comments:
          Accepted at Advanced Robotics, website - this https URL

- **What's New**: 본 연구에서는 이전에 구축된 지도(Map)나 학습이 필요 없는 최적의 로봇 내비게이션 방식을 제안합니다. Omnidirectional camera와 사전 훈련된 vision-language 모델을 활용하여 진정한 open-vocabulary 내비게이션이 가능하다는 것을 보여주었습니다.

- **Technical Details**: 본 연구에서 사용된 omnidirectional camera는 360도 시야를 제공하며, 두 개의 fisheye 렌즈를 이용해 주위 환경을 인식합니다. 이후 fisheye 이미지를 보정하고, 사전 훈련된 vision-language 모델인 CLIP과 Detic을 적용하여 로봇의 움직임을 지시할 수 있는 기반을 마련합니다. 이는 reflex-based control이 가능하게 하며, 복잡한 탐색 행동을 피할 수 있는 방법입니다.

- **Performance Highlights**: 모바일 로봇 Fetch를 통해 수행한 실험 결과, 제안한 방법은 이전 연구와 비교해 매핑이나 정책 생성 없이 간단한 내비게이션이 가능함을 확인했습니다. 실험을 통해 성능의 강점과 더불어 몇 가지 한계점도 논의하였습니다.



### Denoising Pre-Training and Customized Prompt Learning for Efficient Multi-Behavior Sequential Recommendation (https://arxiv.org/abs/2408.11372)
- **What's New**: DPCPL(다단계 행위 연속 추천을 위한 새로운 Pre-training 및 Prompt-tuning 패러다임)이 소개되었습니다. 이 방법은 효율적인 사전 학습 및 사용자 맞춤형 프롬프트 학습을 결합하여 다양한 사용자 행동 데이터를 처리합니다.

- **Technical Details**: DPCPL의 핵심 구성 요소는 Efficient Behavior Miner (EBM)와 Customized Prompt Learning (CPL) 모듈입니다. EBM은 여러 시간 척도에서 노이즈를 필터링하고, CPL은 사용자 기반 정보를 활용해 개인화된 프롬프트를 생성하여 모델의 잠재력을 극대화합니다. 또한, Fast Fourier Transform(FFT) 및 주파수 인식 융합과 같은 기법을 사용해 노이즈를 줄입니다.

- **Performance Highlights**: 세 가지 실제 데이터셋에서 DPCPL의 실험 결과, 높은 효율성과 효과성이 입증되었습니다. 최소한의 파라미터 조정만으로도 다양한 다운스트림 작업에서 최첨단 성능을 초과했습니다.



### Graph Classification via Reference Distribution Learning: Theory and Practic (https://arxiv.org/abs/2408.11370)
- **What's New**: 이번 논문은 Graph Reference Distribution Learning (GRDL)을 도입하여 그래프 분류의 효율성과 정확성을 개선하는 방법을 제안합니다. GRDL은 GNN 레이어에서 제공되는 각 그래프의 잠재 노드 임베딩을 이산 분포로 처리하여, 전역 풀링(global pooling)을 사용하지 않고도 최대 평균 편차(maximum mean discrepancy)를 기반으로 직접 분류를 수행합니다.

- **Technical Details**: GRDL은 그래프의 잠재 노드 임베딩을 사용하여 각 그래프를 분류하는 새로운 접근 방식을 제안합니다. 이 방법은 전통적인 그래프 커널(graph kernels)이나 GNNs의 전역 풀링 작업이 가진 구조적 정보 손실의 문제를 해결합니다. GRDL은 일반화 오류 경계를 도출하고, 실험적으로 그 성능을 검증하여 GNNs보다 강력한 일반화 능력을 보여줍니다.

- **Performance Highlights**: 경험적으로 GRDL은 중간 규모와 대규모 그래프 데이터셋에서 최첨단 방법들에 비해 우수한 성능을 보였으며, 훈련 및 추론 단계에서 최소 10배 더 빠른 속도를 자랑합니다.



### Hypergraph Learning based Recommender System for Anomaly Detection, Control and Optimization (https://arxiv.org/abs/2408.11359)
Comments:
          16 pages, 10 figure, Accepted at IEEE International Conference on Big Data 2022, Osaka, Japan

- **What's New**: 이 논문에서는 다차원 시계열 데이터에서 이상 탐지(anomaly detection)를 위한 새로운 자가 적응형 프레임워크를 제안합니다. 기존 접근 방안들이 센서 간의 고차원 종속성을 간과하는 문제를 해결하기 위해 하이퍼그래프(hypergraph) 구조를 활용하여 시간적 및 공간적 관계를 모델링합니다.

- **Technical Details**: 제안된 프레임워크는 다음과 같은 모듈로 구성됩니다: (a) 하이퍼그래프 구조 학습 모듈(HgSL), (b) 인코더-디코더 모듈(HgED), (c) 하이퍼그래프 예측 모듈(HgF), (d) 하이퍼그래프 편차 모듈(HgD). 이 구조는 센서의 시간적 및 공간적 관계를 유지하면서도 예측을 수행하고 이상 여부를 판단합니다.

- **Performance Highlights**: 제안된 방법은 최신 성능(SOTA)을 기록하여 다양한 벤치마크 데이터셋에서 기존 기법들에 비해 우수한 성능을 보였습니다. 또한, 아블레이션 연구(ablation studies)를 통해 프레임워크의 효과성을 확인했습니다.



### Vision HgNN: An Electron-Micrograph is Worth Hypergraph of Hypernodes (https://arxiv.org/abs/2408.11351)
Comments:
          21 pages, Accepted in PML4DC Workshop at International Conference on Learning Representations (ICLR) 2023

- **What's New**: 이 연구는 전자 현미경의 관측 결과를 바탕으로 한 소재 특성화를 향상시키기 위해 하이퍼그래프 신경망(HgNN) 아키텍처를 제안합니다. 특히, 기존 방법들이 다양한 공간 영역 간의 복잡한 관계를 효과적으로 모델링하지 못하는 문제를 해결합니다.

- **Technical Details**: 제안된 Vision Hypergraph Neural Networks(진단 하이퍼그래프 신경망, Vision-HgNN) 아키텍처는 전자현미경 이미지에서의 피쳐 및 구조 정보를 자동으로 인코딩하고, 시각적 하이퍼그래프 수준의 임베딩(relational embeddings)을 학습합니다. 이는 라벨-전환 멀티 클래스 분류 작업의 성능을 극대화하는 데 중점을 둡니다.

- **Performance Highlights**: 제안된 프레임워크는 대규모 전자 현미경 이미지 기반 데이터셋에서의 컴퓨터 요구사항 효율성을 보여주며, 기존의 인기 있는 벤치마크 방법들보다 우수한 성능을 달성했습니다. 아블레이션 연구(ablation studies) 결과, 높은 정확도로 첨단 성능을 달성하는 것을 입증하였습니다.



### EHL*: Memory-Budgeted Indexing for Ultrafast Optimal Euclidean Pathfinding (https://arxiv.org/abs/2408.11341)
- **What's New**: 이 논문에서는 기존 Euclidean Hub Labeling (EHL) 기법의 단점을 극복하는 새로운 버전 EHL*를 소개합니다. EHL*는 사용자의 메모리 예산에 맞춰 인덱스를 생성하면서 쿼리 런타임 성능도 최적화합니다.

- **Technical Details**: EHL*는 EHL을 개선하여 인접 격자 셀의 레이블을 임의 모양의 영역으로 병합하는 압축 단계를 포함합니다. 이를 통해 메모리 사용을 줄이며 쿼리 런타임을 거의 증가시키지 않으면서 메모리 절약을 검증합니다. EHL*는 또한 이미 알려진 쿼리 분포를 활용하여 더 나은 성능을 발휘합니다.

- **Performance Highlights**: EHL*는 EHL에 비해 메모리 사용량을 10-20배까지 감소시킬 수 있으며, 성능 저하 없이 쿼리 런타임을 유지합니다. 예를 들어, 특정 쿼리 분포가 알려진 경우 EHL*-5가 10GB 이상의 메모리를 약 600MB로 감소시킵니다.



### BURExtract-Llama: An LLM for Clinical Concept Extraction in Breast Ultrasound Reports (https://arxiv.org/abs/2408.11334)
Comments:
          This paper has been accepted as the oral paper for the HCHM workshop, ACM Multimedia 2024

- **What's New**: 본 연구는 유방 초음파 보고서에서 임상 정보를 추출하기 위해 인하우스 LLM(대규모 언어 모델) 개발 파이프라인을 소개합니다. 특히 GPT-4를 사용하여 작은 라벨이 지정된 데이터셋을 생성한 후, 이를 기반으로 Llama3-8B 모델을 미세 조정하는 방법을 제시합니다.

- **Technical Details**: 본 연구는 유방 초음파 보고서에서 임상 정보를 효과적으로 추출하기 위해 3단계 파이프라인을 구성합니다: 1) 관찰 및 인상 정보 추출, 2) GPT-4를 사용한 훈련 라벨 생성, 3) Q-LoRA를 통해 Llama3-8B 모델 미세 조정. 이를 통해 84.6%의 평균 F1 점수를 기록하여 GPT-4와 유사한 정확성을 달성하였습니다. 

- **Performance Highlights**: 개발된 BURExtract-Llama 모델은 유방 초음파 보고서에서 임상 정보를 추출하는 데 있어 GPT-4와 동등한 성능을 발휘하며, 비용 효율성과 데이터 프라이버시를 강화할 수 있는 가능성을 보여줍니다.



### Plug, Play, and Fuse: Zero-Shot Joint Decoding via Word-Level Re-ranking Across Diverse Vocabularies (https://arxiv.org/abs/2408.11327)
Comments:
          Under Review

- **What's New**: 최근 자연어 처리(NLP) 모델들이 멀티모달 입력 처리 및 특정 도메인에서 우수한 성능을 보이고 있습니다. 그러나 실세계에서 다중 모달 번역과 같은 작업은 이러한 여러 모델의 강점을 결합해야 합니다. 본 연구는 추가 교육 없이도 디코딩 단계에서 모델을 합치는 새로운 제로샷 앙상블(zero-shot ensembling) 전략을 제안합니다.

- **Technical Details**: 제안하는 방법론은 단어 수준에서 점수를 결합하여 디코딩 중에 다중 모델을 통합하는 것입니다. 이를 위해, 각 단어의 완료 여부를 예측하는 휴리스틱(heuristic)을 사용하여 디코딩 과정 중에 비정상적인 상태를 피합니다. 모델 간의 단어 수준 재등급(online re-ranking)을 통해 완벽하지 않은 가설에도 정확한 확률 예측을 가능하게 하는 접근 방식을 강조합니다.

- **Performance Highlights**: 실험 결과, 본 연구의 방법론은 음성과 이미지를 고려한 번역을 가능하게 하며, 번역 품질 또한 개선되는 것을 입증했습니다. 특히, 두 가지 모달리티의 정보를 포함하는 타겟 실험을 통해 성능 향상이 관찰되었습니다.



### Towards Evaluating Large Language Models on Sarcasm Understanding (https://arxiv.org/abs/2408.11319)
- **What's New**: 본 논문은 LLMs(대형 언어 모델)가 풍자(sarcasm) 이해에서의 성과에 대한 논의의 타당성을 심도 깊게 검토합니다. 특히 11개 최첨단(SoTA) LLM과 8개 PLM(사전 훈련된 언어 모델)을 활용하여 풍자 탐지 성능을 평가합니다.

- **Technical Details**: 총 6개의 벤치마크 데이터셋에서 zero-shot IO prompting, few-shot IO prompting, chain of thought (CoT) prompting의 세 가지 접근 방식으로 평가하였습니다. 실험 결과, 현재 LLM은 지도된 PLM 기반 풍자 탐지 기준에 미치지 못하는 성능을 보여주었습니다.

- **Performance Highlights**: 1. GPT-4는 다른 LLM들과 비교해 다양한 prompting 방법에서 평균 14.0% 향상된 성능을 보이며, 가장 우수한 성능을 기록했습니다. 2. few-shot IO prompting 방법이 zero-shot IO 및 few-shot CoT보다 평균 4.5% 높은 성능을 나타냅니다. 3. LLM의 풍자 이해에는 여전히 상당한 개선이 필요하다는 점을 강조합니다.



### Swarm Intelligence in Geo-Localization: A Multi-Agent Large Vision-Language Model Collaborative Framework (https://arxiv.org/abs/2408.11312)
- **What's New**: 이번 논문에서는 여러 LVLM (Large Vision-Language Models) 에이전트 간의 커뮤니케이션을 통해 이미지의 효과적인 지오 로컬라이제이션을 달성하는 새로운 시각적 지오 로컬라이제이션 프레임워크인 \name\을 소개합니다.

- **Technical Details**: 이 프레임워크는 에이전트 간의 커뮤니케이션 패턴을 최적화하는 동적 학습 전략을 사용하여 불필요한 논의를 줄이고 효율성을 향상시킵니다. 또한, 새로운 데이터셋인 GeoGlobe를 구축하여 시각적 지오 로컬라이제이션 작업에 적용합니다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식이 최신 기술 (state-of-the-art) 방법과 비교하여 상당한 성능 향상을 보여줍니다.



### KAN4TSF: Are KAN and KAN-based models Effective for Time Series Forecasting? (https://arxiv.org/abs/2408.11306)
- **What's New**: 이 논문은 Kolmogorov-Arnold Network (KAN)을 시간 시계열 예측 연구에 도입하여 더 나은 수학적 특성과 해석 가능성을 제공하는 방법론을 제시합니다. 새로운 모델인 Reversible Mixture of KAN experts (RMoK)는 KAN 전문가의 혼합 구조를 사용하여 변수들을 전문가에게 할당합니다.

- **Technical Details**: RMoK 모델은 KAN 기반이며, 여러 KAN 변형을 전문가로 사용하고, 게이팅 네트워크를 통해 변수를 적응적으로 특정 전문가에 할당하여 예측을 수행합니다. 이 모델은 단일 레이어 네트워크로 구현되며, 성능과 해석 가능성을 동시에 유지할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과 RMoK는 7개의 실제 데이터 세트에서 대부분 경우에 최선의 성능을 달성했습니다. KAN과 기존 선형 모델 간의 성능 비교 및 KAN 통합의 효과에 대한 comprehensive empirical study를 진행하여 RMoK의 메커니즘을 설명합니다.



### UniFashion: A Unified Vision-Language Model for Multimodal Fashion Retrieval and Generation (https://arxiv.org/abs/2408.11305)
- **What's New**: 이 논문에서는 패션 도메인에서 멀티모달 생성 및 검색 작업의 도전을 동시에 해결하는 통합 프레임워크인 UniFashion을 제안합니다.

- **Technical Details**: UniFashion은 이미지 생성, 검색 작업 및 텍스트 생성 작업을 통합합니다. 이 모델은 LLM(large language model)과 확산 모델(diffusion model)을 결합하여 제어 가능하고 고충실도의 생성을 가능하게 합니다. Q-Former는 멀티모달 쿼리를 생성하고, 이를 기반으로 LLM이 캡션을 생성하며, 확산 모델이 이미지를 합성하는 방식으로 작동합니다.

- **Performance Highlights**: UniFashion은 다양한 패션 작업에서 이전의 단일 작업 모델에 비해 현저히 향상된 성능을 보이며, 복잡한 비전-언어 작업을 관리하는 데 쉽게 적응할 수 있습니다.



### Offline Policy Learning via Skill-step Abstraction for Long-horizon Goal-Conditioned Tasks (https://arxiv.org/abs/2408.11300)
Comments:
          9 pages, 4 figures, International Joint Conference on Artificial Intelligence 2024, Published version

- **What's New**: 이 논문에서는 오프라인(goal-conditioned, GC) 정책 학습에서 장기 목표를 설정할 때 발생하는 보상 희소성을 극복하기 위한 '스킬 스텝 추상화(skill-step abstraction)'를 활용한 새로운 프레임워크(GLvSA)를 제안합니다.

- **Technical Details**: GLvSA 프레임워크는 기존 데이터에서 스킬을 학습하고 이를 바탕으로 장기 목표를 단기 목표로 분해하여 GC 정책을 점진적으로 학습하는 방식으로 이루어져 있습니다. 이 모델은 시간적으로 추상화된 스킬의 임베딩을 활용하여 환경 역학을 스킬에 맞춘 잠재 공간에 표현합니다. 또한, 스킬 스텝 롤아웃을 생성하여 새로운 경로를 도출하고 이를 통해 GC 정책의 성능을 향상시킵니다. 모듈 구조를 채택하여 목표 배포 변화(goal distribution shifts)에 따른 정책 업데이트를 용이하게 합니다.

- **Performance Highlights**: GLvSA 프레임워크는 미로(maze) 및 프랑카 주방(Franka kitchen) 환경에서 실험된 결과, 제로샷(zero-shot) 및 몇 샷(few-shot) 적응 상황에서도 기존의 GC 정책 학습 및 스킬 기반 방법들을 능가하는 경쟁력 있는 성능을 보여주었습니다.



### Inference Plans for Hybrid Particle Filtering (https://arxiv.org/abs/2408.11283)
- **What's New**: 이번 논문은 신규 프로그래밍 인터페이스인 inference plans를 소개하여 개발자가 하이브리드 파티클 필터링 중 무작위 변수의 분할을 제어할 수 있도록 합니다. 또한, Siren이라는 새로운 확률적 프로그래밍 언어(PPL)를 도입하여 개발자가 주석을 사용해 inference plans를 명시할 수 있게 합니다.

- **Technical Details**: Siren은 추론 계획의 실행 가능성을 판단하기 위한 추상 해석(abstract interpretation) 기반의 정적 분석(static analysis)을 제공합니다. 이 분석은 Siren의 의미론(semantics)에 대해 건전함(sound)을 proof합니다. 다양한 하이브리드 파티클 필터링 알고리즘에 대해 입력된 추정 계획을 적용하며, 정적 분석이 실제로도 정확하게 작동하는 것을 보여줍니다.

- **Performance Highlights**: 실험 결과, inference plans를 활용한 경우 평균 1.76배, 최대 206배의 성능 향상을 달성하였으며, 정확도가 평균 1.83배 향상되고 최대 595배 비슷한 또는 더 적은 런타임에서 가능함을 보여주었습니다. 33개의 벤치마크-알고리즘 조합 중 27개에서 모든 실행 가능한 추정 계획을 정확하게 식별함을 나타냅니다.



### Automatic Image Annotation (AIA) of AlmondNet-20 Method for Almond Detection by Improved CNN-based Mod (https://arxiv.org/abs/2408.11253)
- **What's New**: 이 논문은 경쟁 치열한 견과 시장에서 프리미엄 농산물에 대한 전세계적인 수요 증가에 대응하여 아몬드 및 그 껍질의 등급 판별 과정을 향상시키기 위한 혁신적인 방법론을 소개합니다.

- **Technical Details**: 최신 Deep Convolutional Neural Networks (CNNs)을 활용하여 AlmondNet-20 아키텍처를 기반으로 하며, 20개의 레이어로 구성된 CNN 모델을 통해 99% 이상의 정확도를 달성했습니다. 1000 epochs에 걸쳐 면밀히 학습한 결과, 손실 함수는 0.0567에 불과했습니다. 데이터 증강(data augmentation) 기법을 사용하여 아몬드와 껍질을 구별하는 강도를 높였습니다.

- **Performance Highlights**: 엄격한 테스트 데이터셋을 통한 평가 결과, 아몬드 탐지에서 완벽한 precision, recall, F1-score 메트릭스를 보여주며, 이 고급 분류 시스템은 산업 전문가와 비전문가 모두에게 유용한 이점을 제공합니다.



### Do Neural Scaling Laws Exist on Graph Self-Supervised Learning? (https://arxiv.org/abs/2408.11243)
- **What's New**: 이 연구는 기존의 그래프 자기 감독 학습(SSL) 기법들이 신경망 스케일링 법칙(neural scaling law)을 따르는지를 살펴봅니다. 특히, 그래프 기초 모델(Graph Foundation Models, GFMs) 구축에 있어서 대규모 프리트레이닝(pre-training)의 잠재력을 평가합니다.

- **Technical Details**: 연구에서는 그래프 분류(graph classification) 작업을 중심으로 하여, 다양한 기존 SSL 기술을 분석하였습니다. 기존의 SSL 손실(loss)이 지속적으로 감소하더라도, 하위 작업의 성능은 그래프 SSL 기법들이 신경망 스케일링 법칙을 따르지 않음을 보여줍니다. 주요 성능 결정 요소는 데이터 및 모델 스케일이 아니라 모델 아키텍처(model architecture)와 프리텍스트 작업(pretext task) 설계입니다.

- **Performance Highlights**: 기존의 그래프 SSL 기법들은 학습 성능에서 명확한 스케일링 행동을 보이지 않으며, 데이터 크기나 모델 크기에 따른 성능 변화가 미미합니다. 이 연구는 그래프 SSL 설계의 새로운 방향을 제시하며, 향후 GFMs 개발을 위한 더 나은 평가 프로토타입을 제공합니다.



### A Little Confidence Goes a Long Way (https://arxiv.org/abs/2408.11239)
Comments:
          13 pages, 2 figures

- **What's New**: 이 연구에서는 큰 언어 모델(LLM)의 숨겨진 상태 활성화(probes of hidden state activations)를 활용한 이진 분류(binary classification) 작업을 위한 새로운 방법을 소개합니다. 특히, 이 접근법은 현재 가장 많은 성능을 발휘하는 LLM과 유사한 성능을 제공하면서도 계산 자원을 훨씬 적게 요구하고 라벨링된 데이터가 필요하지 않습니다.

- **Technical Details**: 제안된 기술은 클래스 레이블(class labels)을 의미가 풍부한 설명으로 변환하고, 멀티레이어 퍼셉트론 모형의 자발적 대칭 파괴(spontaneous symmetry breaking), 엔트로피 최대화를 통한 숨겨진 상태 활성화에서의 신뢰도 점수(confidence scores) 생성, 예측을 위한 앙상블에서 가장 신뢰할 수 있는 모형 선택 등을 포함합니다. 이 논문에서는 Glia라는 이름으로 이러한 기술들을 묶어 부르고 있으며, 네 개의 데이터세트에서 평가되었습니다.

- **Performance Highlights**: 이 방법은 GPU에서 효율적으로 구현될 수 있으며, 감독 학습(supervised fine-tuning)이나 데이터 라벨에 접근할 필요 없이 작동합니다. Glia의 방법론은 4개의 데이터세트(Amazon polarity, IMDB, CUAD, Learned Hands)에 대해 다섯 개의 기본 LLM을 사용하여 성능을 평가했습니다. 기존의 접근 방식보다 숫자가 적은 자원으로 우수한 LLM 추론(inference)을 제공합니다.



### Unified Deep Learning Model for Global Prediction of Aboveground Biomass, Canopy Height and Cover from High-Resolution, Multi-Sensor Satellite Imagery (https://arxiv.org/abs/2408.11234)
- **What's New**: 본 연구에서는 다중 센서 및 멀티스펙트럼 이미지를 활용하여 10미터 해상도로 공중 생물량 밀도(AGBD), 캐노피 높이(CH), 캐노피 커버(CC)와 이들의 불확실성을 동시에 예측하는 딥 러닝 기반 모델을 제안합니다. 이 모델은 2016년부터 2023년까지 전 세계 샘플 데이터를 활용해 훈련되었습니다.

- **Technical Details**: 이 모델은 256x256 픽셀 크기의 이미지 타일을 사용하며, 공중 생물량 밀도와 캐노피 관련 변수들을 동시에 예측합니다. 이를 위해 CNN(Convolutional Neural Network)을 사용하고, 훈련 과정에서는 희소한 실제 데이터의 분포를 극복하는 새로운 기술이 도입되었습니다. 기초 모델은 원거리 관측 데이터를 통해 만들어지며, 다양한 생태계에서의 훈련 샘플을 통해 전 세계적으로 유효한 예측을 제공합니다.

- **Performance Highlights**: 모델은 AGBD에 대해 평균 절대 오차 26.1 Mg/ha, CH에 대해 3.7m, CC에 대해 9.9%의 정확도를 보이며, 이는 이전 연구 결과에 비해 유의미한 개선을 보여줍니다. 또한 검증에 사용된 독립적인 지상 측정 데이터와 높은 상관관계를 보였으며, 모델의 다중 헤드 아키텍처 덕분에 다른 GEDI 변수로의 전이 가능성도 원활하게 이루어집니다.



### OCTCube: A 3D foundation model for optical coherence tomography that improves cross-dataset, cross-disease, cross-device and cross-modality analysis (https://arxiv.org/abs/2408.11227)
- **What's New**: OCTCube는 기존 2D OCT 이미지 슬라이스 대신 3D OCT 볼륨을 활용하여 훈련된 새로운 3D foundation model입니다.

- **Technical Details**: OCTCube는 26,605개의 3D OCT 볼륨과 1.62백만 개의 2D OCT 이미지를 포함하여 사전 훈련되었습니다. 이 모델은 3D masked autoencoders를 기반으로 하고 FlashAttention을 활용하여 GPU 메모리 사용량을 줄이는 방법으로 개발되었습니다.

- **Performance Highlights**: OCTCube는 8개의 망막 질병을 예측하는 데 있어 2D 모델보다 뛰어난 성능을 보였으며, 특히 데이터 간 전이 작업 및 당뇨병, 고혈압과 같은 전신 질병 예측에서도 우수한 성과를 나타냈습니다.



### CoDi: Conversational Distillation for Grounded Question Answering (https://arxiv.org/abs/2408.11219)
Comments:
          13 pages

- **What's New**: 이 논문에서는 Conversational Distillation (CoDi)이라는 새로운 데이터 증류 프레임워크를 소개하며, SLMs (Small Language Models)의 대화 능력을 향상시키기 위한 방법을 제시합니다. CoDi를 통해 대규모의 조교 스타일 데이터셋을 다양하고 조작 가능하게 합성할 수 있습니다.

- **Technical Details**: CoDi는 대화가 아닌 일반적인 작업에 대해서도 적용될 수 있는 프레임워크로, 다단계 대화를 위한 데이터 합성을 목표로 합니다. 이 방법론은 대화 그래프, 턴 기반 프롬프트 증강 및 명시적 언어적 특징을 사용하여 자연스러운 대화를 생성합니다.

- **Performance Highlights**: CoDi로 훈련된 SLMs는 인간 주석 데이터로 훈련된 모델과 동등한 성능을 보이며, 데이터 합성을 통한 대규모 데이터셋 생성을 통해 큰 모델들보다 뛰어난 성능을 보여줍니다.



### Quantum Inverse Contextual Vision Transformers (Q-ICVT): A New Frontier in 3D Object Detection for AVs (https://arxiv.org/abs/2408.11207)
Comments:
          The paper has been accepted as a short paper at CIKM '24

- **What's New**: 이번 논문에서는 지능형 자율주행차량(Auto Vehicles) 기술을 위한 새로운 접근법으로 Quantum Inverse Contextual Vision Transformers (Q-ICVT)라는 이중 단계 융합 프로세스를 제안합니다. 이는 LiDAR와 카메라 데이터를 결합하여 더 나은 성능을 발휘하도록 설계되었습니다.

- **Technical Details**: Q-ICVT는 얕은 컴퓨팅(adiabatic computing) 개념을 활용하여 Global Adiabatic Transformer (GAT)를 도입하고, 이 구조를 통해 스파스 LiDAR 피처를 밀집 이미지를 통해 통합합니다. 또한, Sparse Expert of Local Fusion (SELF) 모듈을 사용하여 LiDAR 데이터 및 카메라 이미지의 지역적 융합을 실현합니다.

- **Performance Highlights**: 실험 결과 Q-ICVT는 Waymo 데이터셋에서 L2 난이도를 기준으로 82.54의 mAPH(Mean Average Precision) 성능을 기록하여 최신 융합 방법보다 1.88% 향상되었습니다. 이 결과는 Q-ICVT의 효과를 강조하며, GAT 및 SELF 모듈의 중요성에 대한 분석도 포함되었습니다.



### Effective Off-Policy Evaluation and Learning in Contextual Combinatorial Bandits (https://arxiv.org/abs/2408.11202)
Comments:
          accepted at RecSys2024

- **What's New**: 이 논문은 Contextual Combinatorial Bandits (CCB) 부문에서 Off-Policy Evaluation and Learning (OPE/L)에 대해 심도 있게 탐구하며, 특히 주요 행동과 보조 행동의 효과를 분리하여 보상 예측을 향상하는 새로운 접근 방식을 제시합니다.

- **Technical Details**: 연구자들은 비율 샘플링(importance sampling)과 회귀(regression) 기법을 활용하여 OPE를 수행합니다. 새로운 추정기인 Off-Policy Estimator for Combinatorial Bandits (OPCB)를 도입하고, 이는 주 행동에서 비율 샘플링을 활용하여 편향(bias)을 줄이고 보조 행동의 효과를 회귀를 통해 처리하는 방식입니다. 이를 통해 기존의 문제점들을 해결하고, Conditional Pairwise Correctness라는 조건 하에 편향 없이 평가할 수 있습니다.

- **Performance Highlights**: 실험 결과에 따르면, OPCB는 기존의 방법보다 OPE와 OPL(Off-Policy Learning) 모두에서 우수한 성능을 보이며, 특히 낮은 분산과 편향으로 새 정책을 보다 정밀하게 평가하고 학습하는 데 성공합니다.



### EPiC: Cost-effective Search-based Prompt Engineering of LLMs for Code Generation (https://arxiv.org/abs/2408.11198)
Comments:
          Submitted to TSE

- **What's New**: 이 연구에서는 진화 알고리즘을 활용하여 코드를 생성하는 데 있어 보다 높은 품질의 프롬프트를 발전시키는 경량화된 접근법인 EPiC(Evolutionary Prompt Engineering for Code)를 제안합니다. 이는 LLM(Large Language Models)와의 상호작용을 최소화하여 비용 효율성을 크게 향상시키는 방식입니다.

- **Technical Details**: EPiC는 코드 생성 태스크에 맞춰 설계된 프롬프트 엔지니어링 방법론으로, 테스트 케이스의 통과율을 기준으로 피트니스 함수를 정의합니다. 기존의 진화적 프롬프트 엔지니어링 기법들과 달리, EPiC는 외부 LLM 호출을 줄이고 로컬 경량 단어 임베딩 라이브러리를 사용하여 변이 연산자를 구현합니다.

- **Performance Highlights**: EPiC는 최신 LLM 기반 코드 생성 모델들과 비교했을 때, 모든 기준선 모델 대비 비용 효과성에서 우수한 성능을 보였습니다.



### Reading with Inten (https://arxiv.org/abs/2408.11189)
- **What's New**: 이 논문은 Retrieval Augmented Generation (RAG) 시스템이 외부 정보 소스와 통합되는 방법을 다룹니다. 특히, 인간 커뮤니케이션의 뉘앙스를 이해하는 데 어려움이 있는 RAG 시스템에서의 풍자(sarcasm) 처리 방법을 연구합니다.

- **Technical Details**: 저자들은 Natural Questions의 Wikipedia 검색 코퍼스를 기반으로 합성된 풍자가 포함된 텍스트를 생성하고, 이러한 텍스트가 RAG 파이프라인의 retriever와 reader 부분의 성능에 미치는 영향을 실험합니다. 그들은 풍자를 처리하기 위한 프롬프트 시스템을 개발하여 모델이 풍자가 있는 상황에서 응답을 해석하고 생성하는 능력을 향상시킵니다.

- **Performance Highlights**: 종합적인 ablation studies를 통해 제안한 방법의 효과성을 입증하며, 풍자가 포함된 콘텐츠를 처리하는 데 있어 성능 향상을 나타냅니다.



### Optimization of Multi-Agent Flying Sidekick Traveling Salesman Problem over Road Networks (https://arxiv.org/abs/2408.11187)
- **What's New**: 이번 논문은 도로 네트워크 위에서 작동하는 다중 에이전트 드론-트럭 배달 시스템을 제안합니다. 단일 트럭과 드론 모델에서 여러 트럭과 다수의 드론을 포함하는 암류된 다수 에이전트 비행 협력 여행판매원 문제(MA-FSTSP)를 도입하여 현실적인 물류 문제를 다룹니다.

- **Technical Details**: MA-FSTSP는 다중 트럭이 각기 다른 창고에서 출발하여 고객을 방문하는 문제로, 각 트럭 그룹은 하나의 트럭과 여러 대의 드론으로 구성됩니다. 이 문제는 NP-hard이며 혼합 정수 선형 프로그래밍(mixed-integer linear programming, MILP) 모델을 이용하여 해결됩니다. 알고리즘은 세 단계로 구성되어 있으며, 첫째 단계에서는 고객을 가까운 창고로 배정하고, 둘째 단계에서는 Set TSP 문제로 확장하여 경로를 계산하며, 마지막 단계에서는 트럭과 드론의 경로를 동시에 최적화합니다.

- **Performance Highlights**: 대규모 물류 애플리케이션에 적합한 본 연구의 접근 방식은 300명 이상의 고객을 5분 이내에 처리할 수 있는 가능성을 보여줍니다. 맨하탄과 보스턴 도로 네트워크에서의 테스트 결과는 기존 기법 대비 30% 이상의 비용 절감과 더욱 우수한 계산 효율성을 달성했음을 나타냅니다.



### Autonomous Negotiation Using Comparison-Based Gradient Estimation (https://arxiv.org/abs/2408.11186)
- **What's New**: 본 논문에서는 다중 에이전트 시스템에서의 자율적인 협상(autonomous negotiation) 과정을 탐구합니다. 특히, 두 명의 자기 이익(self-interested) 합리적(rational) 에이전트가 순차적으로 유한한 카테고리 세트에서 아이템을 거래하는 환경에서 연구하였습니다.

- **Technical Details**: 제안하는 알고리즘은 이전의 수용(accepted) 또는 거부(rejected) 응답을 기반으로 제안 제시(generate offers)를 통해, 응답 에이전트의 유틸리티 함수를 모른 채 유틸리티를 향상시키기 위한 거래 제안을 생성합니다. 이 알고리즘은 합리성(rationality) 가정과 거부된 제안을 활용하여 잠재적 기울기(gradient) 공간을 축소(prune)합니다.

- **Performance Highlights**: 제안된 알고리즘은 랜덤 탐색(random search) 베이스라인과 비교하여 정수(integer) 및 분수(fractional) 거래 시나리오에서 사회적 이익(societal benefit)을 증대시키며 더 적은 수의 거래 제안을 통해 성과를 보여주었습니다.



### Hide Your Malicious Goal Into Benign Narratives: Jailbreak Large Language Models through Neural Carrier Articles (https://arxiv.org/abs/2408.11182)
- **What's New**: 본 논문에서는 새로운 형태의 jailbreak 공격 방법을 제시합니다. 즉, 금지된 쿼리를 포함한 carrier article을 삽입하여 LLM의 주의를 전환하는 방식을 사용합니다.

- **Technical Details**: 제안된 방법은 WordNet을 활용하여 주제 선택을 하고, 공격 대상이 아닌 다른 LLM을 사용하여 carrier article을 생성하는 혁신적인 워크플로우를 포함합니다. 그런 다음 이 carrier article 안에 악의적인 쿼리를 삽입하여 LLM의 안전 장치를 우회하는 방식입니다.

- **Performance Highlights**: 실험 결과, 제안된 공격 방법은 6개의 인기 LLM을 대상으로 하여 높은 성공률로 jailbreak를 성공적으로 수행하였고, 성공률은 최소 21.28%에서 최대 92.55%에 달하였습니다.



### A Full DAG Score-Based Algorithm for Learning Causal Bayesian Networks with Latent Confounders (https://arxiv.org/abs/2408.11181)
Comments:
          17 pages, extended version with supplementary material of paper accepted at the 27th European Conference on Artificial Intelligence (ECAI'24)

- **What's New**: 본 논문은 잠재적 혼란 변수를 식별할 수 있는 첫 번째 완전한 스코어 기반 구조 학습 알고리즘을 소개합니다. 이 알고리즘은 관찰 데이터를 기반으로 하는 DAGs(directed acyclic graphs)의 공간을 탐색하며, 잠재적 혼란 변수를 효과적으로 처리할 수 있습니다.

- **Technical Details**: 잠재적 혼란 변수가 없는 경우, 기존 스코어 기반 알고리즘은 다루기가 불가능한 것으로 알려져 있었습니다. 본 논문에서는 모든 확률 변수가 이산적(discrete)일 경우, 잠재적 혼란 변수를 효과적으로 처리하기 위해 SMCM(semi-Markovian causal models)를 학습하는 알고리즘을 제안합니다. 또한, 기존 스코어 기반 방법들이 통상적으로 연속 변수에만 의존하는 문제를 해결합니다.

- **Performance Highlights**: 실험 결과, 제안한 알고리즘은 관찰적으로 숨겨진 잠재적 혼란 변수를 복구하는 데 효과적임을 보여주었습니다. 이를 통해, 스코어 기반 접근 방식을 통해 구조 학습에서 나타나는 한계를 극복할 수 있는 가능성을 제시합니다.



### SubgoalXL: Subgoal-based Expert Learning for Theorem Proving (https://arxiv.org/abs/2408.11172)
- **What's New**: 이 논문은 SubgoalXL이라는 새로운 접근 방식을 소개하며, 이는 서브골 기반의 증명과 전문가 학습(expert learning)을 시너지시켜 LLM의 형식 정리 증명 능력을 향상시킵니다.

- **Technical Details**: SubgoalXL은 전용 수학 및 정리 증명 데이터의 부족 문제와 LLM의 다단계(reasoning abilities) 추론 능력 향상이 필요한 두 가지 주요 도전을 해결합니다. 데이터 효율성을 최적화하고 서브골 수준 감독(subgoal-level supervision)을 적용하여 제한된 인간 생성 증명에서 더 풍부한 정보를 추출합니다.

- **Performance Highlights**: SubgoalXL은 Isabelle 환경에서 miniF2F 데이터셋에 대해 56.1%의 새로운 최고 성과를 달성하였으며, 이는 Zheng et al. (2023)보다 4.9% 향상된 결과입니다. 함께, SubgoalXL은 miniF2F에서 41 AMC12, 9 AIME 및 3 IMO 문제를 성공적으로 해결하였습니다.



### MS$^3$D: A RG Flow-Based Regularization for GAN Training with Limited Data (https://arxiv.org/abs/2408.11135)
- **What's New**: 본 논문에서는 제한된 데이터로 GAN 훈련 시 발생하는 문제를 해결하기 위해 물리학의 재정규화 그룹(renormalization group, RG) 개념에 기반한 새로운 정규화 방법인 MS$^3$D(다중 스케일 구조적 자기 비유사성)를 제안합니다.

- **Technical Details**: MS$^3$D 정규화는 제한된 데이터 환경에서 생성자(generator)가 얻는 gradient 패턴이 시간이 지남에 따라 집합적(aggregated)으로 변하는 현상을 분석합니다. 이 정규화 방법은 gradient 필드가 서로 다른 스케일 간에 일관된 패턴을 유지하도록 제한하며, 이로 인해 생성자가 더욱 강건하고 중복적인 시스템을 얻게 됩니다.

- **Performance Highlights**: 제안된 MS$^3$D 방법은 GAN의 성능과 안정성을 향상시킬 수 있음을 실험적으로 입증하였으며, 매우 적은 데이터로도 고품질 이미지를 생성할 수 있도록 합니다.



### DOMBA: Double Model Balancing for Access-Controlled Language Models via Minimum-Bounded Aggregation (https://arxiv.org/abs/2408.11121)
Comments:
          11 pages, 3 figures

- **What's New**: DOMBA(더블 모델 균형화)를 제안하여 접근 제어가 적용된 데이터에서 LLM(대규모 언어 모델)의 훈련 및 배포를 위한 새로운 접근법을 제공합니다. 이 방법은 높은 유용성과 보안 보장을 제공하면서 사용자 권한에 따라 훈련 데이터를 보호합니다.

- **Technical Details**: DOMBA는 'min-bounded'(최소 경계) 평균 함수를 사용하여 서로 다른 접근 수준으로 훈련된 두 개의 서브모델의 확률 분포를 집계합니다. 이를 통해 서브모델 각각이 보유한 민감한 정보가 텍스트 생성 시 잘 노출되지 않도록 합니다. 이론적인 수학적 분석을 통해 민감한 정보 노출을 효과적으로 차단하는 방법을 제시합니다.

- **Performance Highlights**: DOMBA는 민감한 정보 노출을 제한하면서도 비보안 모델과 유사한 유용성을 제공합니다. 새롭게 제안된 세 가지 보안 메트릭으로 DOMBA의 효과를 평가하며, 기존 접근 제어 방법에 비해 높은 성능을 유지합니다.



### What can Large Language Models Capture about Code Functional Equivalence? (https://arxiv.org/abs/2408.11081)
Comments:
          37 pages

- **What's New**: 이 논문은 SeqCoBench라는 새로운 벤치마크를 소개하여 Code-LLMs가 코드의 기능적 동등성을 얼마나 잘 포착할 수 있는지를 체계적으로 평가합니다.

- **Technical Details**: SeqCoBench는 Python 프로그램의 의미를 보존하거나 변경하는 20개 이상의 코드 변환을 포함하고 있습니다. 다양한 설정에서, 제로샷(zero-shot) 및 파라미터 효율적인 파인튜닝(parameter-efficient finetuning) 방법을 사용하여 최첨단 (Code-)LLMs의 성능을 평가했습니다.

- **Performance Highlights**: LLMs와 전통적인 일치 기반 검색 점수 간의 성능 차이는 미미하였으며, 두 방법 모두 코드 의미 이해에 깊이가 부족하다는 우려스러운 결과를 보였습니다.



### DiffZOO: A Purely Query-Based Black-Box Attack for Red-teaming Text-to-Image Generative Model via Zeroth Order Optimization (https://arxiv.org/abs/2408.11071)
- **What's New**: 이 논문에서는 텍스트-이미지(T2I) 생성 모델의 취약성을 드러내기 위해 기존의 공격 방법을 재고하고, '순수 블랙박스(black-box)' 방법론을 적용한 DiffZOO라는 새로운 공격 방식을 제안합니다. 이는 공격자가 모델에 대한 사전 정보 없이도 효과적으로 공격할 수 있게 합니다.

- **Technical Details**: DiffZOO는 제로 순서 최적화(Zeroth Order Optimization)를 사용하여 기울기(gradient)를 근사화하고, 이로써 공격 프롬프트(prompt)가 포함된 디스크리트(discrete) 공간 내에서 최적화할 수 있도록 합니다. C-PRV(Continuous Position Replacement Vectors)와 D-PRV(Discrete Position Replacement Vectors)를 활용하여 공격 프롬프트를 개선합니다.

- **Performance Highlights**: DiffZOO는 여러 안전 메커니즘을 갖춘 T2I 모델에 대해 실험을 진행한 결과, 평균 공격 성공률이 기존 연구에 비해 8.5% 향상되었습니다. 이는 T2I 모델의 레드 팀(red teaming) 도구로서의 잠재력을 보여줍니다.



### Toward End-to-End Bearing Fault Diagnosis for Industrial Scenarios with Spiking Neural Networks (https://arxiv.org/abs/2408.11067)
Comments:
          13 pages, 10 figures

- **What's New**: 이 논문은 스파이킹 신경망(Spiking Neural Networks, SNNs)의 새로운 응용 분야인 베어링 결함 진단을 탐구합니다. 기존 방법의 두 가지 주요 한계인 불충분한 인코딩 역량과 비스파이크 지향 아키텍처를 해결하기 위한 Multi-scale Residual Attention SNN (MRA-SNN)을 제안합니다.

- **Technical Details**: MRA-SNN은 경량화된 주의 메커니즘(lightweight attention mechanism)과 다중 스케일 주의 인코딩 모듈을 통해 진동 신호에서 다중 스케일 결함 특징을 추출하여 시공간(spatio-temporal) 스파이크로 인코딩합니다. 이를 통해 복잡한 데이터 전처리(deep preprocessing)를 제거합니다. 또한 스파이크 잔여 주의 블록(spike residual attention block)을 도입하여 고차원 결함 특징을 추출하고, 주의 메커니즘을 통해 희소 스파이크(sparse spikes)의 표현력을 향상시킵니다.

- **Performance Highlights**: MRA-SNN은 MFPT 및 JNU 벤치마크 데이터셋에서 기존 방법에 비해 정확성, 에너지 소비 및 노이즈 강인성(noise robustness) 면에서 현저히 개선된 성능을 보였으며, 실제 산업 시나리오에 배치를 위한 더 나은 실행 가능성을 보여줍니다.



### Tabular Transfer Learning via Prompting LLMs (https://arxiv.org/abs/2408.11063)
Comments:
          COLM 2024

- **What's New**: 본 논문은 제한된 라벨 데이터로 학습하는 문제를 해결하기 위해 Transfer Learning(전이학습)의 새로운 접근 방식을 제안합니다. 특히, Tabular tasks(표 형식 작업)에 초점을 맞추며, 이는 기존의 비전(vision)이나 언어(language) 작업에 비해 연구가 부족했던 영역입니다.

- **Technical Details**: 제안하는 새로운 프레임워크는 Prompt to Transfer (P2T)로, 이는 대형 언어 모델(Large Language Models, LLMs)을 활용하여 라벨이 없는(또는 이질적인) 소스 데이터로부터 목표 작업에 대한 예제를 생성합니다. P2T는 소스 데이터셋에서 목표 작업의 특징과 강한 상관관계를 가지는 열(Column feature)을 식별하여 해당 작업과 관련된 가상의 예시(pseudo-demonstrations)를 생성합니다.

- **Performance Highlights**: 실험 결과, P2T는 다양한 표준 표 작업(Tabular learning benchmarks)에서 이전 방법들을 초월하는 성과를 기록하며, 아직 탐구되지 않은 표형 전이 학습 문제에 대해 좋은 가능성을 보입니다.



### Interactive-T2S: Multi-Turn Interactions for Text-to-SQL with Large Language Models (https://arxiv.org/abs/2408.11062)
Comments:
          15 pages, 7 figures

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 강력한 추론 능력을 활용하여 text-to-SQL 파싱을 탐구합니다. 우리는 데이터베이스와의 직접 상호작용을 통해 SQL 쿼리를 생성하는 Interactive-T2S라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 효율적이고 사전 구성된 리소스가 필요하지 않은 방법으로 SQL 생성 과정을 단계별로 해석할 수 있는 시스템입니다.

- **Technical Details**: Interactive-T2S는 LLM과 데이터베이스 간의 다중 상호작용을 통해 SQL 쿼리를 생성하는 새로운 프레임워크입니다. 이 프레임워크에서는 LLM이 생각하고 행동하는 사고-행동 패러다임(thought-action paradigm) 아래 수행되며, SQL 쿼리 생성을 위해 네 가지 일반 도구를 포함하고 있습니다. 실험에서는 BIRD-Dev 데이터셋을 사용하여 오라클 지식 없이 단 두 개의 예제를 사용하여도 최첨단 결과를 달성했습니다.

- **Performance Highlights**: 우리의 방법은 두 개의 예제만으로도 뛰어난 성능을 보여주며, 넓은 테이블을 효율적으로 처리할 수 있는 강력한 기능을 입증했습니다. Comprehensive testing on the Spider-Dev, BIRD-Dev, and their variant datasets showcased that our approach achieves significant results with minimal exemplar input.



### Dynamic Code Orchestration: Harnessing the Power of Large Language Models for Adaptive Script Execution (https://arxiv.org/abs/2408.11060)
Comments:
          6 pages, 4 figures

- **What's New**: 본 연구는 큰 언어 모델(LLM)을 활용하여 작성된 언어 지시문으로부터 컴퓨터 프로그램 및 어셈블리 코드를 자동 생성하는 가능성을 탐구합니다. 특히 동적 코드 실행을 통해 실제 실행 중인 애플리케이션의 비즈니스 로직을 생성하는 방법을 제시합니다.

- **Technical Details**: 사용자는 애플리케이션을 실행하고 상호작용하면서 작성된 언어 지시문이 애플리케이션의 비즈니스 로직의 큰 부분을 포함합니다. 동적 코드 오케스트레이션(dynamic code orchestration)을 통해 이 지시문이 LLM 서비스에 전송되고 Python 코드로 변환됩니다. 이 코드는 실행 가능 코드 블록으로 컴파일되어 글로벌 네임스페이스(global namespace)에 등록됩니다.

- **Performance Highlights**: 이 접근 방식을 통해 사용자는 직접 기능을 정의할 수 있으며, 정적 실행 파일(static executables) 대신에 항상 변경 가능한 일시적 실행 파일(ephemeral executables)을 사용할 수 있습니다. 이로써 코드의 경량화와 보안 면에서 커다란 이점을 얻을 수 있습니다.



### LLM Agents Improve Semantic Code Search (https://arxiv.org/abs/2408.11058)
Comments:
          12 pages, 1 Figure

- **What's New**: 이 논문에서는 코드 검색의 정확성을 개선하기 위해 RAG (Retrieval Augmented Generation)를 활용한 에이전트 기반 접근 방식을 도입했습니다. 이를 통해 사용자 쿼리에 정보가 추가되어 더 나은 코드 검색이 가능해집니다.

- **Technical Details**: 이 연구는 에이전트가 GitHub 저장소에서 수집한 관련 정보를 사용하여 사용자 쿼리를 강화하도록 설계되었습니다. 첫 번째 단계로, 에이전트는 사용자의 자연어 쿼리에 필요한 기술 정보를 추가하고, RAG 기술을 사용하여 적절한 컨텍스트 정보를 검색합니다. 이렇게 생성된 입력은 OpenAI의 최신 텍스트 임베딩을 사용하여 코드로 변환되어, 코드 검색의 정확성을 높입니다.

- **Performance Highlights**: 실험 결과 RepoRift는 기존 방법들보다 월등히 향상된 성능을 보이며, CodeSearchNet 데이터셋에서 Success@10에서 78.2%의 성공률, Success@1에서 34.6%의 성공률을 기록했습니다. 이 연구는 에이전트 기반 LLM과 RAG의 잠재력을 강조하며, 더욱 효과적인 코드 검색 시스템을 위한 중요한 발전을 보여줍니다.



### Architectural Foundations for the Large Language Model Infrastructures (https://arxiv.org/abs/2408.09205)
- **What's New**: 대형 언어 모델(LLM) 인프라 개발의 중요성이 강조되며, 이 논문은 LLM의 인프라, 소프트웨어 및 데이터 관리의 복잡한 경관을 탐구합니다. 성공적인 LLM 개발을 위한 고려사항 및 안전장치에 대한 통찰력을 제공합니다.

- **Technical Details**: LLM 교육을 위한 인프라 구성에서 H100/H800 GPU를 장착한 서버 클러스터가 주류로 자리 잡았고, LoRA(Low-Rank Adaptation)와 같은 경량화 방법이 컴퓨팅 파워 요구사항을 줄이는 데 기여합니다. 또한, 알고리즘 최적화 및 하이퍼파라미터 설정이 모델 성능 향상에 중요합니다.

- **Performance Highlights**: 8개 노드로 구성된 클러스터는 7B 파라미터 모델의 교육을 하루 만에 완료할 수 있으며, GPU 및 CPU 자원의 유연한 활용이 LLM 추론 배치에서 필수적입니다. 데이터 관리의 효율성을 통해 높은 품질의 데이터 세트를 확보하고 모델 학습을 극대화할 수 있습니다.




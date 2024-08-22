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




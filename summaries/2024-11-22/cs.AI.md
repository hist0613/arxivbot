New uploads on arXiv(cs.CL)

### Marco-o1: Towards Open Reasoning Models for Open-Ended Solutions (https://arxiv.org/abs/2411.14405)
- **What's New**: 최근 OpenAI의 o1 모델은 대규모 추론 모델(Large Reasoning Models, LRM)에 대한 관심을 불러일으켰습니다. Marco-o1은 단순한 수학, 물리 및 코딩을 넘어 개방형 해결책 문제를 강조하여, 보상 기준이 모호한 범위에서도 일반화 능력을 테스트합니다. 이 모델은 Chain-of-Thought (CoT) 미세 조정 및 Monte Carlo Tree Search (MCTS) 기법을 활용하여 복잡한 실제 문제 해결을 최적화 하고 있습니다.

- **Technical Details**: Marco-o1는 CoT 미세 조정, MCTS 및 다양한 혁신적인 추론 전략을 이용하여 성능을 강화합니다. CoT 데이터셋과 함께 Qwen2-7B-Instruct 모델을 미세 조정하여 복잡한 작업 처리 능력을 향상시켰습니다. MCTS는 모델 출력을 기반으로 확신 점수를 활용하여 여러 경로를 탐색하고 최적의 솔루션으로 안내하는 역할을 합니다.

- **Performance Highlights**: Marco-o1은 MGSM 데이터셋에서 +6.17%의 정확도 개선을 달성했으며, 번역 작업에서도 뛰어난 능력을 보였습니다. 예를 들어, 중국어의 속어 표현을 정확히 번역하여 문맥의 뉘앙스를 파악하는 능력을 보여주었습니다. 이러한 결과는 Marco-o1의 혁신적인 접근 방식과 광범위한 문제 해결 능력을 강조합니다.



### Lightweight Safety Guardrails Using Fine-tuned BERT Embeddings (https://arxiv.org/abs/2411.14398)
Comments:
          To appear in Proceedings of COLING 2025

- **What's New**: 최근 대형 언어 모델(LLMs)의 확산으로 기업들이 신속하게 개념 증명과 프로토타입을 개발할 수 있게 되었습니다. 이에 따라 LLM의 동작을 모니터링, 양자화 및 제어할 수 있는 강력한 가드레일(guardrail) 구현의 필요성이 커지고 있습니다. 이전의 필터링 방식들은 LLM을 미세 조정하여 불완전한 사용자 프롬프트나 시스템 출력을 거르는 데 성공했지만, 이러한 접근법들은 지연(latency)과 유지비용이 높아 실용적이지 않을 수 있습니다.

- **Technical Details**: 우리는 경량 아키텍처인 Sentence-BERT를 미세 조정하여 새로운 접근 방식을 제안합니다. 이 방식은 LlamaGuard의 70억 개 파라미터에서 약 6,700만 개로 모델 크기를 줄이며 AEGIS 안전 벤치마크에서 비슷한 성능을 유지합니다. 특히, 이 방법은 사용자의 프롬프트나 대화 조각이 안전한지 유해한지를 분류하는 데 초점을 맞추고 있으며, 안전 및 유해한 입력을 클러스터링하여 분류기(classifier)를 학습시킵니다.

- **Performance Highlights**: 우리가 제안한 접근법은 안전 작업을 단순한 텍스트 분류 문제로 간주합니다. 최종 결과는 기존의 리소스를 많이 소모하는 LLM 기반 가드레일 체크기들과 비교할 때 비슷하거나 우수한 성능을 보였습니다. 우리의 경량 모델은 AEGISSafetyDataset에서 안전한 프롬프트와 유해한 프롬프트를 구분하는 데 필요한 효과적인 접근 방식을 제공합니다.



### POS-tagging to highlight the skeletal structure of sentences (https://arxiv.org/abs/2411.14393)
Comments:
          in Russian language. Conference: Automated control systems and information technologies this https URL Section: IT and automated systems

- **What's New**: 이 연구에서는 BERT 아키텍처를 기반으로 한 part-of-speech (POS) 태깅 모델을 개발하여 문장의 골격 구조를 추출하는 방법을 제시합니다. 연구에서는 러시아어 텍스트에 대한 모델을 미세 조정하여 효과성을 입증하였습니다. 이러한 접근 방식은 자연어 처리(natural language processing) 작업을 향상시키는 잠재적인 응용 프로그램을 제공합니다.

- **Technical Details**: 모델은 token classification을 통해 문장의 구조를 분석하고, morphological analysis를 수행하여 각 단어의 품사를 태깅합니다. BERT(대표적인 사전 훈련된 언어 모델)는 transfer learning 기법을 활용하여 성능을 극대화합니다. 본 연구에서 제안한 모델은 러시아어 데이터에 특화되어 있습니다.

- **Performance Highlights**: 모델은 러시아어 텍스트에 대해 미세 조정된 후, 탁월한 성능을 보였으며, 이는 실제 NLP(Natural Language Processing) 작업에서 기계 번역(machine translation) 성능을 개선하는 데 기여할 수 있습니다. 이러한 성과는 다양한 언어와 응용 분야에 적용될 수 있는 가능성을 열어줍니다.



### UnifiedCrawl: Aggregated Common Crawl for Affordable Adaptation of LLMs on Low-Resource Languages (https://arxiv.org/abs/2411.14343)
- **What's New**: 이 논문은 적은 자원으로 운영되는 언어들에 대한 LLMs의 성능을 향상시키는 효율적인 데이터 수집 방법론인 UnifiedCrawl을 소개합니다. 이 방법은 최소한의 컴퓨팅 자원으로 Common Crawl 데이터셋에서 적은 자원 언어에 대한 텍스트 데이터를 필터링하여 추출할 수 있도록 설계되었습니다. UnifiedCrawl을 통해 생성된 데이터셋은 기존에 사용 가능했던 소스들보다 상당히 큰 단일 언어 데이터셋을 제공합니다.

- **Technical Details**: UnifiedCrawl은 적은 자원 언어를 위해 Common Crawl에서 텍스트 데이터를 수집하기 위해 제안된 새로운 방법론입니다. 이 방법은 메모리, 컴퓨팅 및 네트워크 사용량을 최적화하면서 개인 소비자 하드웨어에서 작동할 수 있도록 설계되었습니다. 수집된 데이터 셋을 바탕으로 QLoRA와 같은 효율적인 어댑터 방법으로 다국어 LLM을 미세 조정하여, VRAM 사용량을 최소화하면서 성능을 크게 향상시킬 수 있습니다.

- **Performance Highlights**: 실험 결과, UnifiedCrawl로 수집된 데이터셋을 활용하여 LLMs의 언어 모델링 성능이 현저하게 개선되었으며, 특히 적은 자원 언어에 대한 응답 품질이 향상되었습니다. 모델은 few-shot prompting 점수를 증가시켰으며, 이전보다 더 나은 응답의 일관성과 유의미성을 보여주었습니다. 이 연구는 소비자 하드웨어를 사용하여 LLM을 향상시키기 위한 저렴한 접근 방식을 제공합니다.



### Velocitune: A Velocity-based Dynamic Domain Reweighting Method for Continual Pre-training (https://arxiv.org/abs/2411.14318)
Comments:
          Work in progress

- **What's New**: 이번 논문에서는 Velocitune이라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 학습 속도(learning velocity)를 동적으로 평가하여 데이터 비율을 조정하며, 느리게 학습하는 도메인에는 더 많은 가중치를 부여하고 빠른 도메인은 배제하는 방식으로 구성이 됩니다. 기존의 정적 비율 조정 방식을 넘어서 도메인 적응 지속적 프리트레이닝의 복잡성을 해결하는 방법을 제안합니다.

- **Technical Details**: Velocitune은 각 도메인의 초기 손실(initial loss)과 대상 손실(target loss)을 평가하여 학습 속도를 측정합니다. Chinchilla scaling law를 이용하여 비용 효율적으로 대상 손실을 예측하며, 주기적으로 학습 속도를 평가하여 느리게 학습하는 도메인의 샘플링 가중치를 증가시키고, 빠르게 학습하는 도메인의 가중치를 줄입니다. 이를 통해 도메인 간의 균형 잡힌 학습을 도모합니다.

- **Performance Highlights**: Velocitune을 코딩 및 수학 추론 데이터셋과 시스템 명령생성 데이터셋에서 평가한 결과, 기본 모델에 비해 평균적으로 수학 작업에서 1.6%, 코딩 작업에서 3.8%의 성능 향상을 보여주었습니다. 또한 Llama3 및 Mistral에서 각각 5.3%와 4.4%의 성능 향상을 기록하였으며, 이는 Velocitune이 데이터 비율 조정 외에도 데이터 순서와 예측된 대상 손실의 중요함을 강조합니다.



### Efficient Aspect-Based Summarization of Climate Change Reports with Small Language Models (https://arxiv.org/abs/2411.14272)
- **What's New**: 이 논문은 기후 변화 보고서에 대한 Aspect-Based Summarization (ABS) 시스템과 관련된 새로운 데이터셋을 발표합니다. 연구진은 Large Language Models (LLMs)와 Small Language Models (SLMs)를 활용하여 이러한 정보를 효율적으로 요약하는 방법을 모색하며, 특히 에너지 효율성을 고려한 평가 방식을 최초로 도입하였습니다.

- **Technical Details**: 기후 변화 보고서를 위해 ABS 작업에 LLM과 SLM을 평가하며, 또한 RAG (Retrieval Augmented Generation) 설정 내에서 검토합니다. 다양한 모델의 성능을 비교하는 동안, SLMs는 LLMs와 유사한 성능을 보이면서도 더 낮은 탄소 발자국을 기록함을 보여줍니다.

- **Performance Highlights**: 연구 결과, 현대의 LLM 및 SLM 모두 기후 변화 보고서에 대한 ABS 작업을 효과적으로 수행할 수 있으며, 추가 연구가 필요하다는 점이 강조됩니다. 특히, zero-shot generative 모델 평가를 통해 SLMs의 지속 가능한 활용 가능성을 제시함으로써, 향후 기후 변화 관련 정보 요약의 발전에 기여할 것을 목표로 합니다.



### Knowledge Graphs, Large Language Models, and Hallucinations: An NLP Perspectiv (https://arxiv.org/abs/2411.14258)
Comments:
          7 pages, 2 Figures, 1 Table

- **What's New**: 최근 대규모 언어 모델(LLM)의 발전은 자연어 처리(NLP) 응용 프로그램에서 큰 변화를 이끌고 있습니다. 그러나 LLM은 사실과 일치하지 않는 답변을 내놓는 '환각(hallucination)' 문제에 직면하고 있어 신뢰성의 저하와 활용성의 한계를 보이고 있습니다. 이 논문에서는 지식 그래프(KG)를 활용하여 LLM의 환각 문제를 해결하고자 하는 최신 연구 동향을 다루고 있습니다.

- **Technical Details**: 지식 그래프(KGs)는 엔티티(entities)와 그 관계(edges)로 구성된 구조화된 정보로, LLM의 사실적 일관성을 높이는 데 중요한 역할을 수행합니다. KG를 통해 LLM은 실제 세계 객체에 대한 사실적 정보를 효율적으로 인지할 수 있고, 이는 모델의 재훈련을 줄이는 데 기여합니다. 논문에서는 다양한 환각 완화 모델과 이러한 모델의 아키텍처를 카테고리화하여 설명하고 이에 대한 평가 방법도 제시합니다.

- **Performance Highlights**: 이 논문은 환각 탐지를 위한 메트릭스인 BERTScore와 BARTScore를 사용하기도 하며, 이러한 메트릭스의 한계점을 지적하고 더 세밀한 환각 탐지를 위한 연구가 필요하다고 강조합니다. 또한, 최근 LLM의 활용 빈도가 높아짐에 따라 다영역(multi-domain)에서의 평가가 필요하다고 주장하며, 이는 LLM의 실제 응용 가능성을 높이는 데 기여할 것입니다. 이는 환각 탐지와 지식 통합 모델의 효과성을 더욱 향상시킬 것으로 기대됩니다.



### Do I Know This Entity? Knowledge Awareness and Hallucinations in Language Models (https://arxiv.org/abs/2411.14257)
- **What's New**: 이 논문에서는 쿠데타를 발생시키는 주요 원인으로써 엔티티 인식을 제안합니다. 대규모 언어 모델에서의 허위 사실 생성(hallucination) 문제를 해결하기 위해 희소 오토인코더(Sparse Autoencoders)라는 해석 도구를 활용했습니다. 이 도구를 통해 모델이 특정 엔티티에 대해 어떤 지식을 가지거나 가지지 않는지를 평가하는 자기 지식의 선형 방향을 발견하였습니다.

- **Technical Details**: 희소 오토인코더(SAEs)는 모델 표현을 더 큰 차원의 공간으로 사영하여 해석 가능한 특성의 분리를 도모합니다. 이 연구에서 Gemma Scope의 SAEs를 이용하여 2B 및 9B 모델에서 자기 지식의 내부 표현을 발견했습니다. 또한, 우리는 엔티티의 속성에 대한 질문에 대한 모델의 응답 거부 행동이 이러한 방향에 의해 유도될 수 있음을 입증했습니다.

- **Performance Highlights**: 이 연구는 희소 오토인코더를 통해 다양한 엔티티 유형에 대해 일반화된 엔티티 인식 방향을 발견했음을 보여주었습니다. 모델이 알려진 엔티티에 대한 질문을 거부하도록 유도하거나, 알려지지 않은 엔티티의 속성을 허위로 생성하게 만들 수 있는 능력을 갖고 있음을 밝혀냈습니다. 이는 사전 훈련 데이터에서 학습된 기존의 메커니즘이 파인 튜닝 과정에서 재사용되는 경향이 있음을 시사합니다.



### Intent-Aware Dialogue Generation and Multi-Task Contrastive Learning for Multi-Turn Intent Classification (https://arxiv.org/abs/2411.14252)
- **What's New**: 이번 논문에서는 Chain-of-Intent라는 혁신적인 방식을 도입하여 Hidden Markov Models(히든 마르코프 모델)과 Large Language Models(대규모 언어 모델)을 결합하여 맥락 인식이 가능한 대화 생성을 가능하게 합니다. 이 연구는 전자상거래 채팅 기록에서 도메인 특화된 지식을 추출하여 대화 턴과 인텐트 전환을 추정하고, 이를 통해 유기적인 대화를 생성합니다. 또한 MINT-CL 프레임워크를 제안하여 대규모 주석 데이터 없이도 다중 턴 인텐트 분류의 정확성을 향상시킵니다.

- **Technical Details**: Chain-of-Intent는 HMM과 LLM을 결합하여 사용자의 인텐트에 기반한 대화를 생성하는 자가 학습 방식입니다. MINT-CL은 다중 작업 대비 학습(multi-task contrastive learning)을 통해 MTIC의 정확성을 향상시키며, 다양한 언어와 시장에서 우수한 성능을 보여줍니다. 논문에서 발표된 MINT-E 데이터셋은 8개 시장에서의 다양한 언어로 인텐트를 포함하는 멀티턴 전자상거래 대화 코퍼스를 제공합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 대화 품질과 인텐트 분류 정확도 모두 기존 기준선을 초과하며, 특히 다국어 환경에서 두드러진 성능 향상을 보여줍니다. 또한 데이터 생성 노력의 현저한 감소를 통해, 다중 언어 및 대화 모델 개발에 필요한 자원을 절감할 수 있는 가능성을 제시합니다.



### Evaluating the Robustness of Analogical Reasoning in Large Language Models (https://arxiv.org/abs/2411.14215)
Comments:
          31 pages, 13 figures. arXiv admin note: text overlap with arXiv:2402.08955

- **What's New**: 이 연구는 대규모 언어 모델(LLMs)의 유추 능력의 강인성을 평가하며, 이전 연구에서 주장된 바와는 달리 이러한 모델들이 대조 변형(task variants)에서 많은 경우 강인성을 결여하고 있음을 보여줍니다. 특히, 인간과 GPT 모델 간의 성능 차이를 강조하며, 수많은 변형에서 모델이 취약한 모습을 보이는지에 대한 실험을 진행하였습니다.

- **Technical Details**: 저자는 세 가지 유추 문제 영역에서 LLM의 내부 작동 방식에 대해 다루며, 각 영역에서 두 가지 변형을 사용한 실험을 설계했습니다. 특히, 문자-문자 유추, 숫자 행렬 문제, 그리고 이야기 기반 유추 문제를 평가하였으며, GPT 모델이 원본 문제와 변형 문제에서 일관된 성능을 보이지 않는다는 점을 발견했습니다.

- **Performance Highlights**: 단순한 문자-문자 유추 문제에서, 인간 참여자는 두 가지 변형에서 높은 성능을 유지했으나 GPT 모델의 성능은 급격히 감소했습니다. 이처럼 GPT 모델이 유추 성능을 발휘하는 데에 있어, 그 개념적인 강인성이 부족하다는 증거를 제공하며, AI 시스템의 인지적 능력을 평가할 때 정확성뿐만 아니라 강인성도 고려해야 함을 강조합니다.



### OpenScholar: Synthesizing Scientific Literature with Retrieval-augmented LMs (https://arxiv.org/abs/2411.14199)
- **What's New**: OpenScholar는 4500만 개의 오픈 액세스 논문에서 관련 정보를 검색하고, 인용 기반 응답을 합성하여 과학적 질문에 대답하는 검색 보강된 대형 언어 모델(LM)입니다. 새로운 벤치마크인 ScholarQABench를 소개하여 OpenScholar의 성능을 평가했으며, 이는 컴퓨터 과학, 물리학, 신경과학, 생물 의학을 포함하는 2967개의 전문가 작성 쿼리와 208개의 긴 형식 응답으로 구성되어 있습니다. OpenScholar는 GPT-4o보다 5% 더 높은 정확성을 달성했으며, 인용 정확도가 인간 전문가와 동등한 수준으로 향상되었습니다.

- **Technical Details**: OpenScholar는 OpenScholar-DataStore(OSDS)를 사용하여 4500만 개의 오픈 액세스 논문과 2억 3700만 개의 구문 임베딩을 포함합니다. 이 시스템은 관련 구문을 검색하고, 이를 기반으로 반복적인 자기 피드백 생성을 통해 응답의 출력을 정제하는 방식으로 작동합니다. OpenScholar는 ‘효율적인 8B 모델’로 훈련되어 특정 도메인에 최적화된 검색 및 합성 기능을 제공하며, 모델 간의 결합을 통해 전체적인 정확성을 개선할 수 있습니다.

- **Performance Highlights**: OpenScholar는 GPT-4o와 PaperQA2를 포함한 다양한 모델들보다 우수한 성능을 보였으며, 70%의 경우 전문가 작성 응답보다 더 나은 결과를 제공했습니다. ScholarQABench에서 OpenScholar는 인용 정확도와 정보 범위에서 높은 성과를 보였고, GPT-4o의 일부 기능을 향상시킬 수 있는 가능성을 보여주었습니다. OpenScholar의 효율적인 구조는 비용 절감 효과를 가져올 수 있으며, 연구에 실질적으로 기여할 수 있는 고품질 출력을 생성합니다.



### Why do language models perform worse for morphologically complex languages? (https://arxiv.org/abs/2411.14198)
Comments:
          9 pages

- **What's New**: 이번 연구는 언어 모델의 성능 차이를 설명하기 위해 형태론적 유형학에 대한 새로운 증거를 발견했습니다. 특히, 고립어(agglutinative) 언어보다 융합어(fusional) 언어에서 더 나은 모델 성능을 보이는 경향이 있다는 점을 강조합니다. 이를 통해 형태론적 정렬(morphological alignment), 토큰화 품질(tokenization quality), 데이터셋 크기의 불균형이 성능 차이에 미치는 영향을 분석하였습니다.

- **Technical Details**: 연구에서는 MorphScore라는 새로운 토큰화 평가 메트릭을 소개하며, 22개 언어에 대한 지원 데이터셋을 제공하였습니다. 분석 결과, 형태론적 정렬이 성능 차이에 미치는 직접적인 역할은 발견되지 않았으나, 데이터셋의 크기가 동등할 때 성능 차이가 줄어드는 경향이 있음을 확인했습니다. 이는 다양한 언어의 인코딩 효율성(byte-premium) 차이가 영향을 미친다는 것을 시사합니다.

- **Performance Highlights**: 연구 결과는 형태론적 유형학에 따른 언어의 학습 난이도에 대한 기존의 관념을 수정할 필요가 있음을 보여줍니다. 성능 차이는 궁극적으로 데이터셋 크기의 불균형에서 기인함을 알 수 있었으며, 이는 자원이 적은 언어의 성능 향상을 위한 연구에 중요한 시사점을 제공합니다. 전체적으로 더 많은 데이터는 더 나은 모델 성능으로 이어진다는 점이 재확인되었습니다.



### Learning from "Silly" Questions Improves Large Language Models, But Only Slightly (https://arxiv.org/abs/2411.14121)
Comments:
          27 pages, 14 figures

- **What's New**: 이 논문은 우선적으로 Ruozhiba라는 중국 웹사이트에서 수집된 특이한 질문들이 대규모 언어 모델(LLM)의 Supervised Fine-Tuning(SFT) 성능 향상에 기여할 수 있음을 입증하고자 합니다. Ruozhiba의 유머와 아비트럴리티(Absurdity)와 같은 특성을 분석하여, SFT 데이터 생성에 사용할 구체적인 규칙을 개발하고, 이 규칙들을 MMLU 훈련 세트에 적용하여 개선된 데이터를 생성합니다.

- **Technical Details**: 이 연구는 GPT-4를 활용하여 Ruozhiba 질문의 성공 사례를 교육학, 심리학 및 인지 과학의 관점에서 분석하여 규칙을 추출합니다. 실험을 통해 'Counterintuitive Thinking' 규칙을 적용했을 때 'Global Facts' 과제에서 약 5% 향상된 성능을 보여주고, 상대적으로 'Conceptual Boundaries' 규칙은 'Econometrics' 과제에서 6.14%의 성능 하락을 초래했습니다. 이러한 결과는 주제 및 과제에 따라 서로다른 규칙이 일관된 영향을 미침을 나타냅니다.

- **Performance Highlights**: 우리의 연구는 Ruozhiba 스타일의 데이터 세트를 구성할 때 과제 다양성과 규칙 적용 가능성을 고려해야 함을 강조합니다. 데이터 증대 규칙을 통해 생성된 데이터 세트를 사용하여 LLM을 세밀하게 조정할 경우, 대규모 평가 결과 MMLU 테스트 세트에서 0.54%의 전반적인 성능 향상에 기여했습니다. 일부 규칙은 'STEM' 과목의 성능을 저하시켰지만, 'Humanities' 과목에서는 약간의 성능 향상을 가져오는 등 다양한 성과 변화를 보여줍니다.



### Lost in Inference: Rediscovering the Role of Natural Language Inference for Large Language Models (https://arxiv.org/abs/2411.14103)
Comments:
          preprint, 13 pages

- **What's New**: 이번 연구에서는 자연언어추론(NLI) 작업이 대규모 언어 모델(LLM) 평가에 유용한지를 탐구한다. NLI 작업은 기존에 NLP 모델의 언어 이해를 평가하는 데 사용되었지만, LLM의 출현 이후 그 중요성이 감소했다. 본 논문은 NLI 데이터셋이 여전히 LLM 성능 평가에 기여할 수 있는지, 그리고 모델 크기 및 품질 분류 가능성을 평가한다.

- **Technical Details**: 본 연구에서는 다섯 가지 다양한 NLI 벤치마크를 통해 여섯 개의 모델을 평가하였다. 모델의 정확도가 훈련 과정에서 어떻게 변화하는지를 분석하며, 모델의 소프트맥스 분포가 인간의 분포와 얼마나 일치하는지를 조사했다. 결과적으로, NLI 벤치마크는 모델 개발과 개선에 유용하며, 모델 성능은 훈련 중 꾸준히 발전한다는 것을 발견하였다.

- **Performance Highlights**: 연구 결과, 최고의 모델은 NLI 벤치마크에서 80-90%의 정확도를 달성하였으며, 일부 데이터셋에서는 70%를 초과하지 못하는 경우도 있었다. NLI 작업은 여전히 모델의 다양한 크기와 품질을 분류하는 데 효과적이며, 훈련 도중 데이터 오염이 정확도에 영향을 미치지 않음을 보여주었다. 특히, 모델의 분포와 인간 레이블 간의 차이를 평가하여 모델 개선 가능성이 있음을 제시하였다.



### Meaning at the Planck scale? Contextualized word embeddings for doing history, philosophy, and sociology of scienc (https://arxiv.org/abs/2411.14073)
Comments:
          18 pages, 7 figures (1 in the Supplement)

- **What's New**: 이 논문은 과학 개념의 맥락적이고 진화하는 의미를 연구하기 위한 새로운 도구로서 contextualized word embeddings (CWEs)의 잠재력을 탐구하고 있습니다. 특히 'Planck'라는 용어를 테스트 사례로 사용하여 다섯 개의 BERT 기반 모델을 평가하였고, Astro-HEP-BERT라는 맞춤형 모델을 포함하여 분야 특화 훈련을 거친 모델들이 일반 모델보다 우수하게 작동함을 보였습니다. CWEs는 과학 언어를 분석하는 데 있어 사전 훈련된 모델을 조정하는 비용 효율성을 입증하며, 사회 역사적 동역학을 탐구하는 새로운 경로를 제공합니다.

- **Technical Details**: CWEs는 대규모 텍스트 말뭉치의 사용 패턴을 통해 단어를 연속 벡터 공간의 점으로 인코딩합니다. 이 공간에서 단어 간의 의미적 및 구문적 관계는 상대적 거리로 표현되며, 비슷한 맥락은 가깝게 그룹화됩니다. CWEs는 단어의 맥락에 따라 동적으로 표현을 조정하여 단어의 의미가 시간, 학문, 또는 기타 맥락에서 어떻게 다르게 진화하는지를 분석할 수 있게 해줍니다.

- **Performance Highlights**: 모델 비교에서, 도메인에 적합한 BERT 모델이 일반 모델보다 'Planck'라는 용어의 다의성을 명확히 구분하고, 잘 알려진 의미를 예측하며, 고품질의 의미 군집을 생성하는 데 있어서 우수한 성능을 보여주었습니다. 또한, 이 연구는 unlabeled Astro-HEP Corpus에서 'Planck'의 의미가 30년에 걸쳐 변화하는 추세를 드러내고, Planck 우주 임무의 등장이라는 주요 의미를 강조합니다.



### The Master-Slave Encoder Model for Improving Patent Text Summarization: A New Approach to Combining Specifications and Claims (https://arxiv.org/abs/2411.14072)
Comments:
          25pages, 1 figure

- **What's New**: 본 논문에서는 전통적인 특허 텍스트 요약 생성 모델이 겪는 문제점을 해결하기 위해, 마스터-슬레이브 인코더 아키텍처(MSEA)를 기반으로 한 새로운 특허 텍스트 요약 생성 모델을 제안합니다. 기존 모델들은 특허 문헌에서만 입력되는 한계가 있어 요약의 핵심 요소를 충분히 반영하지 못하고, 새로운 기술 용어에 대한 Out of Vocabulary (OOV) 문제와 정보 중복의 어려움이 있었습니다. MSEA 모델은 이러한 문제를 해결하기 위해, 특허 텍스트의 명세서와 청구항을 함께 입력으로 받아 이들의 특성과 세부 사항을 효과적으로 탐색합니다.

- **Technical Details**: MSEA 모델은 마스터 인코더와 슬레이브 인코더로 구성되어 있으며, 각 인코더가 특허 텍스트의 서로 다른 중요성을 처리합니다. 슬레이브 인코더는 마스터 인코더의 출력과 다른 입력을 각각 처리하여 디코더에 추가 입력 벡터를 제공합니다. 이 모델은 포인터 네트워크를 통해 새로운 기술 용어에 대한 인식을 강화하고, 입력 시퀀스의 '기억'과 '망각'을 조절하여 더욱 정교하게 특허 요약을 생성합니다.

- **Performance Highlights**: MSEA 모델은 공개된 특허 텍스트 데이터셋에서 Improved Multi-Head Attention Mechanism (IMHAM)과 비교하여 Rouge-1, Rouge-2, Rouge-L 점수에서 각각 0.006, 0.005, 0.005의 성능 향상을 달성했습니다. 실험 결과는 MSEA 모델이 특허 요약 생성 분야에서 뛰어난 성능과 효과성을 입증하고 있음을 보여줍니다. 이러한 결과는 특허 텍스트의 특성을 활용하여 보다 높은 품질의 요약 생성을 가능하게 합니다.



### DRPruning: Efficient Large Language Model Pruning through Distributionally Robust Optimization (https://arxiv.org/abs/2411.14055)
Comments:
          Work in Progress

- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 비대칭한 성능 저하 문제를 해결하기 위해 DRPruning 방법론을 제안하였습니다. DRPruning은 데이터 분포를 동적으로 조정하여 도메인 간 균형 잡힌 성능을 유지하는 방식으로, 기존의 부정 균형 문제를 완화합니다. 또한, 이 방법은 임시 데이터 비율과 최적 참조 손실을 자동으로 결정하여 더욱 다양한 적용 가능성을 제시합니다.

- **Technical Details**: 주요 기술로는 구조적 가지치기(structured pruning)와 분포적으로 강건한 최적화(distributionally robust optimization, DRO)가 사용됩니다. 구조적 가지치기는 모델 크기를 줄이고, DRO는 다양한 테스트 분포에서 성능을 극대화하기 위해 최악의 결과를 최적화합니다. 이 과정에서 동적 하이퍼파라미터 조정을 통해 훈련 중 도메인 성능을 끌어올리는 접근법을 채택하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 DRPruning 방법이 유사한 크기의 모델들보다 우수한 성능을 보였으며, 특히 단일 언어 및 다국어 설정 모두에서 정밀도(perplexity)와 다운스트림 작업의 성능이 향상되었습니다. 특히, 다국어 환경에서는 다운스트림 과제에서 +2.95%의 성능 향상을 기록하였으며, 도메인별 평가에서도 평균 +17.9%의 개선을 나타냈습니다. 이는 DRPruning의 성능이 다양한 도메인과 분포 변화의 견고성을 실제로 입증하였음을 보여줍니다.



### FunctionChat-Bench: Comprehensive Evaluation of Language Models' Generative Capabilities in Korean Tool-use Dialogs (https://arxiv.org/abs/2411.14054)
Comments:
          8 pages

- **What's New**: 이 연구는 언어 모델의 도구 사용 대화에서 생성 능력을 조사하였으며, 모델의 출력 결과를 Tool Call, Answer Completion, Slot Question, Relevance Detection의 네 가지 유형으로 분류합니다. 이러한 평가 항목을 기반으로 한 FunctionChat-Bench라는 새로운 벤치마크를 도입하여 700개 평가 항목과 자동 평가 프로그램을 제공합니다. 또한, 단일 턴의 Tool Call 시나리오에서 높은 정확성을 보이는 언어 모델이 다중 턴 환경에서의 생성 성능에서도 우수한 결과를 보이지는 않는다는 점을 주장합니다.

- **Technical Details**: FunctionChat-Bench는 단일 호출 데이터셋과 대화 데이터셋의 두 가지 하위 집합으로 구성되어, 도구 호출 출력뿐만 아니라 사용자와의 대화 기능도 종합적으로 평가합니다. 각 하위 집합은 언어 모델이 적절한 도구 기능을 선택하고 정보 추출하여 인수를 적절히 생성할 수 있는지를 평가합니다. 언어 모델의 평가 방법으로는 정확한 일치 접근법과 코사인 유사도가 사용되며, 이는 모델의 성능을 객관적으로 판별하는 데 도움을 줍니다.

- **Performance Highlights**: 연구 결과, FunctionChat-Bench를 통해 수행된 여러 언어 모델에 대한 평가에서, 단일 호출 시나리오에서는 높은 성능을 보이는 반면, 다중 턴 대화에서는 사용자의 요구를 적절히 처리하는 능력이 제한적임을 발견했습니다. 이는 언어 모델이 도구 호출 메시지 생성 능력뿐만 아니라, 사용자를 참여시키는 대화 메시지를 효과적으로 생성할 수 있어야 한다는 것을 강조합니다. 종합적으로 이 벤치마크는 언어 모델들의 도구 활용 능력을 보다 완벽하게 평가할 수 있는 기반을 제공합니다.



### Forecasting Future International Events: A Reliable Dataset for Text-Based Event Modeling (https://arxiv.org/abs/2411.14042)
Comments:
          EMNLP 2024 Findings

- **What's New**: 본 논문에서는 WORLDREP (WORLD Relationship and Event Prediction)라는 새로운 데이터셋을 소개합니다. 이 데이터셋은 대규모 언어 모델(LLM)의 고급 추론 능력을 활용하여 기존 데이터셋의 한계를 극복하고자 설계되었습니다. WORLDREP는 정치 과학 분야의 전문가들에 의해 엄격하게 검증된 고품질 스코어링 레이블을 특징으로 하며, 국제 사건 예측 작업에 대한 효과성을 입증하는 다양한 실험을 통해 그 품질과 유용성을 보여줍니다.

- **Technical Details**: WORLDREP의 데이터셋 구축 과정은 다자 관계의 포착과 관계 라벨링의 정확성을 증대시키는 두 가지 주요 개선점을 포함하고 있습니다. 데이터 수집 단계에서는 약 44,706개의 뉴스 기사를 수집하여 각 기사를 단일 사건으로 취급하고, LLM의 프롬프트 설계를 통해 관련 국가를 정확하게 추출하는 자가 수정 메커니즘을 통합했습니다. 또한, 각 뉴스 기사에서 생성된 쌍의 관계를 평가하고 스코어를 부여하는 세밀한 방법론을 매핑하여 더욱 풍부한 데이터를 생성했습니다.

- **Performance Highlights**: WORLDREP의 품질을 검증하기 위해 여러 가지 실험을 진행했으며, 전문가의 라벨과의 일치성을 평가했습니다. 선정된 1,030개의 기사에 대해 전문가들이 수작업으로 라벨링을 진행하고, 우리의 레이블과 일치하는 정도를 분석하여 신뢰성과 일관성을 보장했습니다. 이 결과는 데이터셋의 높은 성능과 함께 예측 모델 교육의 성과를 크게 향상시킨 것으로 나타나며, 데이터셋과 전체 자동화 소스 코드를 공개하여 향후 연구와 개발을 지원하고자 합니다.



### Towards Full Delegation: Designing Ideal Agentic Behaviors for Travel Planning (https://arxiv.org/abs/2411.13904)
- **What's New**: 이 논문에서는 LLM 기반의 에이전트가 일상적인 결정 과정을 완전히 대행하는 것을 목표로 합니다. 기존 작업들은 특정 목표를 개선하는 데 집중했으나, 본 연구는 개인화된 요구와 변화하는 맥락에 적응하는 에이전트의 행동 평가를 제안합니다. 이를 위해 APEC 에이전트 헌법(APEC Agent Constitution)을 도입하여 정확성, 적극성, 효율성 및 신뢰성을 기준으로 설정합니다.

- **Technical Details**: 제안하는 TTG 시스템은 자연어 지시를 받아 최적의 여행 일정을 생성하는 데 LLM과 기존의 심볼릭 솔버, 즉 혼합 정수 선형 프로그래밍(MILP)을 결합하여 사용합니다. TTG는 사용자의 자연어 요청을 JSON 형식의 심볼릭 표현으로 변환하고, 이를 해결한 결과를 자연어로 응답합니다. 이 시스템은 요구 사항이 정확하게 번역되면 결과의 실행 가능성과 최적성을 보장합니다.

- **Performance Highlights**: TTG 시스템은 사용자 요청에 대해 거의 실시간으로(약 5초) 실행 가능한 일정을 제공합니다. APEC-Travel 에이전트는 이전 기준 대비 20.7% 향상된 성능을 보였으며, LLM-as-a-Judge 점수에서 9.1% 개선된 결과를 기록했습니다. 이러한 비교는 TTG가 기존 LLM 기반 시스템보다 더욱 신뢰할 수 있는 솔루션을 제공함을 보여줍니다.



### PIORS: Personalized Intelligent Outpatient Reception based on Large Language Model with Multi-Agents Medical Scenario Simulation (https://arxiv.org/abs/2411.13902)
- **What's New**: 중국의 외래 진료 시스템에서의 부담을 줄이기 위해 Personalized Intelligent Outpatient Reception System (PIORS)을 제안합니다. 이 시스템은 LLM 기반의 간호사와 병원 정보 시스템(HIS)의 협업을 통해 개선된 외래 접수 서비스를 제공합니다. 새로운 의료 대화 데이터 생성 프레임워크인 SFMSS를 통해 LLM의 실제 의료 환경 적응성을 높이고 있습니다.

- **Technical Details**: PIORS는 환자, 간호사, 정보 보조자, 임상 의사 네 가지 참여자를 포함하며, HIS와 통합됩니다. PIORS-Nurse는 환자의 의료 정보를 수집하고 질문에 답변하며, HospInfo-Assistant와 협력하여 개인화된 서비스를 제공합니다. SFMSS는 실제 환자 상호작용을 모의하고 서비스 흐름을 조절하여 LLM이 진료 환경에 맞게 조정될 수 있도록 지원합니다.

- **Performance Highlights**: 자동 및 인간 평가에서 15명의 사용자와 15명의 임상 전문가를 대상으로 PIORS의 효과성을 평가하였습니다. 결과적으로 PIORS-Nurse는 정확성과 정보 수집 능력 면에서 최신 모델 GPT-4o를 포함한 모든 기준 모델을 초월하며, 사용자 평가에서 81% 이상의 승리 또는 동률 비율을 기록하여 실제 시나리오에서 더 나은 경험을 제공합니다.



### Interactive and Expressive Code-Augmented Planning with Large Language Models (https://arxiv.org/abs/2411.13826)
- **What's New**: REPL-Plan은 LLM(대형 언어 모델) 플래닝 접근법으로, 코드의 모든 장점을 활용하면서도 동적인 특성을 지니고 있다. 제안된 방법은 LLM이 Read-Eval-Print Loop(REPL)와 상호작용하며 작업을 해결할 수 있게 하여, 오류를 유연하게 수정하고 작업을 역동적으로 처리할 수 있는 능력을 갖추고 있다. 이 방식은 복잡한 플래닝 문제 해결에 효과적이다.

- **Technical Details**: REPL-Plan은 LLM-REPL을 이용하여 대화형 코드 실행 환경을 구현하며, LLM이 코드 한 줄씩 작성하고 함수를 호출하여 플래닝 환경과 상호작용할 수 있게 한다. 이 프로세스는 프로그래머가 실시간으로 코드를 실행하고 결과를 확인하며 수정하는 방식으로 동작한다. 이러한 REPL의 특징은 복잡한 작업을 상위에서 하위로 나누어 해결할 수 있도록 해준다.

- **Performance Highlights**: REPL-Plan은 ALFWorld와 WebShop과 같은 다양한 플래닝 영역에서 뛰어난 성능을 보여준다. 또한 복잡한 웹 관찰을 처리해야 하는 새로운 실세계 웹 탐색 작업에 성공적으로 적용되어 뛰어난 성과를 보였다. 전반적으로 이 방법은 코드 수정 및 예측 능력을 테스트하는 등의 다양한 검증을 통해 견고성을 입증하였다.



### InstCache: A Predictive Cache for LLM Serving (https://arxiv.org/abs/2411.13820)
- **What's New**: 이번 연구는 대부분의 사용자 지시가 짧고 반복적이며 예측 가능하다는 점에 착안하여, InstCache라는 새로운 예측 캐시 시스템을 제안합니다. 이 시스템은 LLM(대형 언어 모델)의 기능을 활용하여 사용자가 입력할 가능성이 있는 지시를 예측하고 이를 캐시하여 처리합니다. InstCache는 NLL(부정 로그 우도) 기반의 지시 사전 채우기 알고리즘을 통해 히트율과 캐시 크기를 예측 가능한 관계로 연결하여 구현됩니다.

- **Technical Details**: InstCache는 모든 가능한 지시를 탐색하기 위해 트리 구조를 사용하며, 여기서 각 노드는 토큰이고 경로는 지시와 그에 대한 응답을 나타냅니다. 예측 후에는 해시 테이블로 변환되어, 조회 시간이 거의 O(1)로 최적화되어 있어, 기존 LLM 서비스 시스템과 통합 시 지연 시간을 최소화할 수 있습니다. 연구에서는 InstCache가 4.5GB의 CPU 메모리를 사용하여 LMSys 데이터셋에서 최대 51.34%의 히트율을 달성한 것을 보여줍니다.

- **Performance Highlights**: InstCache는 기존 LLM 서비스 시스템과 통합 시 최대 2배의 평균 속도 향상을 제공합니다. 이는 사용자의 지시 중 절반을 CPU와 메모리만으로 처리할 수 있음을 의미하며, 결과적으로 에너지 소비를 크게 절감합니다. 이러한 성과는 LLM 기반 시스템의 성능과 효율성을 한층 끌어올리는 기회를 제공합니다.



### SemiKong: Curating, Training, and Evaluating A Semiconductor Industry-Specific Large Language Mod (https://arxiv.org/abs/2411.13802)
Comments:
          On-going work

- **What's New**: 본 논문은 반도체 산업을 위한 첫 번째 산업별 언어 모델인 SemiKong을 소개합니다. 기존의 범용 모델로는 해결하기 어려운 반도체의 복잡한 물리학과 화학 문제를 전문적으로 다루기 위해, 반도체 관련 텍스트 데이터의 풍부한 코퍼스를 수집하고 이를 기반으로 한 기초 모델을 개발하였습니다. SemiKong은 공정 최적화 및 제어와 관련된 작업에서 정밀한 성능을 발휘하며, 다양한 반도체 제조 및 설계 작업에서 일반적인 LLM을 초월하는 성과를 보여주었습니다.

- **Technical Details**: SemiKong은 반도체 제조 공정과 관련된 대규모 텍스트 코퍼스를 큐레이션하여, 이 데이터로 사전 학습한 기초 모델입니다. 특히 에칭 문제를 심화 이해하기 위해 설계되었으며, 새로운 평가 프레임워크를 도입하여 전문 지식을 활용함으로써 도메인 특화 AI 모델의 성능 평가를 향상시킵니다. 이를 통해, 반도체 제조의 특정 작업에 초점을 맞춘 도메인 전용 LLM의 필요성을 강조합니다.

- **Performance Highlights**: SemiKong은 공정 매개변수 최적화, 이상 탐지, 예측 유지보수와 같은 산업 관련 평가 지표에서 뛰어난 성능을 보여줍니다. 특히, 제품의 역량을 극대화하기 위해 일반 범용 LLM과 비교 시 현저하게 개선된 결과를 도출하였으며, 이는 산업 전용 모델 개발의 중요성을 여실히 증명합니다. 다양한 실험을 통해 이 모델이 향후 반도체 설계 및 제조의 자동화 및 효율성을 획기적으로 높일 수 있는 잠재력을 보유하고 있음을 확인하였습니다.



### Explaining GPT-4's Schema of Depression Using Machine Behavior Analysis (https://arxiv.org/abs/2411.13800)
Comments:
          21 pages, 3 tables, 6 figures, 1 supplementary table, 83 references

- **What's New**: 본 연구는 대규모 언어 모델인 GPT-4가 우울증과 같은 정서적 장애를 평가하는 방식에 대한 새로운 이해를 제공합니다. 이 작업은 최신 측정 이론을 활용하여 GPT-4가 우울증 증상을 어떻게 상호 연관시키고 해석하는지를 해독하였습니다. 연구 결과, GPT-4는 우울증에 대한 평가에서 높은 수렴 타당성을 보였으며, 내부 일관성도 상당히 높은 것으로 나타났습니다.

- **Technical Details**: 연구 방법으로는 기계 심리학(Machine Psychology) 접근법을 사용하여 GPT-4의 우울증 schema를 추론했습니다. 이론적 기초로는 PHQ-9 설문지를 활용하여 언어 샘플로부터 증상 수준 점수를 추정하는 두 가지 주요 단계가 포함되었습니다. 첫 번째 단계에서는 참가자 에세이에서 언급된 증상을 식별하고 각 증상에 대해 심각도 점수를 할당하였으며, 두 번째 단계에서는 맥락으로부터 암시된 추가 증상을 추론하였습니다.

- **Performance Highlights**: 결과적으로 GPT-4는 자가 보고(self-report) 데이터와 전문가 평가와 간의 높은 상관관계를 보였으며, 특히 전체 PHQ-9 점수에서 r=0.70, 전문가 평가에서 r=0.81의 상관계를 나타냈습니다. 모든 증상에 대해 GPT-4와 전문가 간의 합의는 자가 보고와 비교했을 때 더 높은 경향을 보였습니다. 주목할 만한 점은 GPT-4가 자살 사고와 같은 특정 증상에서 전문가와의 차이가 0.32에 달하는 경우도 있었던 것입니다.



### NewsInterview: a Dataset and a Playground to Evaluate LLMs' Ground Gap via Informational Interviews (https://arxiv.org/abs/2411.13779)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)의 언어 및 전략적 대화 기초기술 부족 문제를 해결하기 위해, NPR과 CNN의 40,000 건의 인터뷰 데이터를 수집하여 분석했습니다. 결과적으로, LLM은 인식과 질문 전환 능력에서 사람 인터뷰어보다 현저히 낮은 수준을 보였습니다. 이를 개선하기 위해, 다양한 개인을 설정하고 설득 요소를 통합한 실제 시뮬레이션 환경을 개발하였습니다.

- **Technical Details**: 연구에 사용된 데이터셋은 NPR과 CNN에서 수집된 45,848 개의 일대일 인터뷰 문서로, 각 인터뷰어와 인터뷰이의 역할을 엄밀히 분류하기 위해 질문 부호의 사용 빈도를 기반으로 역할을 판별했습니다. 또한, LLM이 사용자 답변 인식 및 설득 요소에서 어려움을 겪는지 분석하기 위해, 카운터팩추얼 시뮬레이션 기법을 사용했습니다. 다양한 질문 생성 접근 방법(기본, 사고 과정, 개요 모델)을 시험하여 LLM의 효과를 평가했습니다.

- **Performance Highlights**: 결과적으로, LLM은 정보 공유에서 인간의 행동을 모방하는 데에는 성공하였으나, 질문에 대한 응답 인식 및 설득적 대화의 면에서 지연을 보였습니다. 이는 LLM의 모델 크기와 능력에 관계없이 비효율적인 정보 추출로 이어졌습니다. 이러한 결과는 LLM이 전략적 대화 능력을 향상시킬 필요성을 강조하며, 향후 연구의 방향을 제시합니다.



### Benchmarking GPT-4 against Human Translators: A Comprehensive Evaluation Across Languages, Domains, and Expertise Levels (https://arxiv.org/abs/2411.13775)
Comments:
          Work in progress

- **What's New**: 본 연구는 GPT-4의 번역 능력을 다양한 수준의 인간 번역가와 비교하여 포괄적으로 평가한 것이다. MQM 스키마를 활용한 체계적인 인간 평가를 통해 세 가지 언어 쌍(중국어↔영어, 러시아어↔영어, 중국어↔힌디어) 및 세 가지 도메인(뉴스, 기술, 생의학)에서 번역 품질을 분석하였다. 연구 결과, GPT-4는 기초 수준의 번역가와 비슷한 번역 품질을 보였으나, 숙련된 번역가보다는 뒤처지는 것으로 나타났다.

- **Technical Details**: 기존의 Neural Machine Translation(NMT) 시스템과 달리, GPT-4는 자원 부족 언어 쌍에서도 안정적인 번역 품질을 유지하였다. 연구는 다양한 숙련도를 가진 인간 번역가들과 GPT-4의 번역을 비교하고, MQM 스키마를 통해 번역 결과의 오류를 레이블링하였다. 결과적으로 GPT-4는 총 오류 수관점에서 기초-중급 수준 번역가와 유사한 성능을 보여주었으며, 세부적인 오류 분석을 통해 번역 접근 방식의 차이를 규명하였다.

- **Performance Highlights**: GPT-4는 여러 도메인에서 일관된 성능을 보여주었으나, 문법과 Named Entity 부분에서 약점을 보였다. 연구는 GPT-4가 지나치게 문자 그대로의 번역을 경향하며, 어휘적 일관성이 부족하다는 두 가지 주요 한계를 드러냈다. 흥미롭게도, GPT-4는 인간 번역자들이 자주 겪는 환각이나 피로 문제에서 자유롭다는 장점을 지닌다.



### Assessing Gender Bias in LLMs: Comparing LLM Outputs with Human Perceptions and Official Statistics (https://arxiv.org/abs/2411.13738)
Comments:
          under review for Coling conference

- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)에서 성별 편향을 조사합니다. 연구팀은 LLM의 성별 인식과 미국 노동 통계 데이터, 인간 응답자의 반응을 비교했습니다. 이를 통해 새로운 평가 세트를 개발하고, 성별 중립성과 실제 통계 자료와의 차이를 밝혀내어 LLM의 신뢰성을 향상시켰습니다.

- **Technical Details**: 연구에 사용된 두 개의 데이터세트는 인간의 성별 인식과 노동 시장의 실제 성별 분포를 반영합니다. Perception Dataset은 직업에 대한 성별 인식을 평가한 404개의 항목을 포함하고 있으며, U.S. Labor Bureau Statistics Dataset은 316개의 객관적인 성비 데이터를 제공합니다. Kullback-Leibler (KL) Divergence를 사용하여 LLM의 예측과 기준 분포 간의 일치를 평가했습니다.

- **Performance Highlights**: 모델 결과는 GPT-4o가 인간 인식 데이터와 가장 잘 일치하는 것으로 나타났습니다. 그러나 모든 모델은 남성 응답자에게 약간 더 강한 일치를 보였으며, 모든 모델이 공식 통계 데이터와 더 밀접하게 일치했습니다. GPT-3.5-turbo는 공식 통계와의 정렬에서 가장 가까운 결과를 보여주며, 50% 중립성 기준과의 비교에서도 LLM의 성별 인식이 여전히 편향을 보이고 있음을 입증했습니다.



### Hierarchical Text Classification (HTC) vs. eXtreme Multilabel Classification (XML): Two Sides of the Same Meda (https://arxiv.org/abs/2411.13687)
- **What's New**: 이번 연구는 Hierarchical Text Classification (HTC)와 eXtreme Multi-Label Text Classification (XML) 모델을 서로 다른 데이터셋에서 평가하여 두 분야 간의 성능을 비교하는 데 중점을 두었습니다. 연구 결과 XML 모델이 HTC 데이터셋에서도 잘 작동함을 발견하였고, 이는 두 접근 방식의 경계를 허물고 새로운 기회를 보여주고 있습니다.

- **Technical Details**: HTC는 계층 구조의 레이블 세트를 활용하여 텍스트를 분류하는 방법론이며, XML은 수백만 개의 레이블을 다루는 데 최적화된 텍스트 분류 방법론입니다. 연구에서는 R-Precision과 P@k 같은 성능 평가 지표를 사용하여 모델의 성능을 상세히 비교하였습니다. HTCL_MODEL과 XML_MODEL의 성능 차이를 고찰하여 각각의 모델이 가진 강점을 분석했습니다.

- **Performance Highlights**: XML 모델, 특히 CascadeXML은 HTC 모델보다 우수한 성능을 보이며 다양한 데이터셋에서 높은 성공률을 기록했습니다. 결과적으로 XML 모델이 HTC 연구에서 기준점으로 활용될 수 있는 가능성을 제시하였습니다. 연구는 또한 두 분야의 모델이 협력하면서 다각적인 성과를 낼 수 있음을 알리고 있습니다.



### Hymba: A Hybrid-head Architecture for Small Language Models (https://arxiv.org/abs/2411.13676)
Comments:
          20 pages, models are available on huggingface

- **What's New**: Hymba라는 새로운 언어 모델 패밀리가 소개됩니다. 이 모델은 하이브리드 헤드 병렬 아키텍처(hybrid-head parallel architecture)를 채택하여 Transformer의 어텐션 메커니즘과 상태 공간 모델(state space models, SSMs)을 통합합니다. 이로 인해 높은 효율성과 더 나은 문맥 요약 기능을 제공합니다. 또한, 입력에 추가되는 학습 가능한 메타 토큰(learnable meta tokens)이 도입되어 어텐션 메커니즘의 "강제 주의" 부담을 줄입니다.

- **Technical Details**: Hymba 모델은 어텐션 헤드와 SSM 헤드를 동일한 레이어 내에서 통합하여 각 입력에 대한 병렬 처리(parallel processing)를 가능하게 합니다. 이러한 하이브리드 헤드 접근 방식을 통해 높은 해상도의 기억력을 확보할 수 있으며, SSM의 효율적인 내용 요약 기능을 활용할 수 있습니다. 또한, KV 캐시(key-value cache) 공유 및 부분 슬라이딩 윈도우 어텐션(partial sliding window attention)을 도입하여 모델의 캐시 크기를 줄였습니다.

- **Performance Highlights**: Hymba-1.5B-Base 모델은 성능 면에서 2B 이하의 모든 공개 모델을 초월하며, Llama-3.2-3B 모델보다 평균 정확도가 1.32% 더 높습니다. 아울러 캐시 크기는 11.67배 줄어들고, 처리 속도는 3.49배 증가했습니다. Hymba는 또한, 다양한 벤치마크에서 새로운 최첨단 성능을 달성하며, 특히 일반 및 회상 집약적 작업에서 두드러진 성과를 보여줍니다.



### AddrLLM: Address Rewriting via Large Language Model on Nationwide Logistics Data (https://arxiv.org/abs/2411.13584)
Comments:
          Accepted by KDD'25 ADS Track

- **What's New**: AddrLLM은 주소 재작성(address rewriting)을 위해 설계된 혁신적인 프레임워크입니다. 기존 주소 재작성 방법의 한계를 극복하기 위해, AddrLLM은 Supervised Fine-Tuning(SFT), Address-centric Retrieval Augmented Generation(RAG), Bias-free Objective Alignment 모듈로 구성되어 있습니다. 이 연구는 LLM 기반 주소 재작성 접근 방식을 이용하여 비정상 주소 문제를 해결하는 효과적인 방법을 제시하며, 실제 데이터에 대한 철저한 테스트를 통해 성능을 입증하였습니다.

- **Technical Details**: AddrLLM 프레임워크는 세 가지 주요 구성 요소로 구성됩니다. Supervised Fine-tuning 모듈은 데이터의 품질을 극대화하여 주소를 효율적으로 수정할 수 있도록 합니다. Address-centric Retrieval Augmented Generation 모듈은 관련 주소의 맥락 정보를 활용하여 모델의 성능을 강화하며, Bias-free Objective Alignment 모듈은 편향을 최소화하기 위해 JD의 LBS 시스템에서 제공되는 데이터를 통합하여 결과를 보정합니다.

- **Performance Highlights**: AddrLLM은 오프라인 실험을 통해 비정상 주소를 43.9% 수정하였으며, 최첨단 방법들에 비해 24.2%의 성능 향상을 보였습니다. JD 물류 시스템에 4개월 이상 배포된 결과, 약 200만 건의 일일 소포 중에서 비정상 주소로 인한 소포 재배치율을 40% 이상 감소시켰습니다. 이러한 결과는 AddrLLM이 실제 애플리케이션에서 높은 효율성을 가져다 줄 수 있음을 강조합니다.



### Looking Beyond Text: Reducing Language bias in Large Vision-Language Models via Multimodal Dual-Attention and Soft-Image Guidanc (https://arxiv.org/abs/2411.14279)
Comments:
          19 pages, 12 figures

- **What's New**: 본 논문에서는 대형 비전-언어 모델(LVLMs)의 언어 편향 문제를 해결하기 위한 새로운 시스템 프레임워크인 LACING을 제안합니다. LACING은 다중 모드 이중 주의 메커니즘(multi-modal dual-attention mechanism)과 소프트 이미지 가이드(soft-image Guidance)를 활용하여 LVLMs의 비전 이해력을 향상시킵니다. 이는 LVLMs의 환각(hallucination) 현상을 줄이고, 텍스트와 이미지의 통합을 강화하는 데 중점을 둡니다.

- **Technical Details**: LACING의 MDA는 비주얼 입력의 통합을 강화하는 이중 주의 메커니즘을 도입합니다. IFG는 학습과 추론 과정에서 시각적 입력을 대신하는 학습 가능한 소프트 비주얼 프롬프트(soft visual prompt)를 도입하여 LVLMs가 텍스트 입력의 우선순위를 높이도록 유도합니다. 또한, IFG는 인접한 텍스트 입력에 대한 모델의 과도한 의존성을 완화하기 위한 새로운 디코딩 전략을 제안합니다.

- **Performance Highlights**: 실험 결과, LACING은 LVLMs의 언어 편향을 효과적으로 제거하여 비주얼 이해력을 크게 향상시키고 환각 현상을 줄이는 데 성공합니다. 이는 추가적인 훈련 자원이나 데이터 없이도 이루어질 수 있어, 연구자들에게 실질적인 이점을 제공합니다. 저자들은 코드와 모델을 제공하여 관련 연구자들이 활용할 수 있도록 하고 있습니다.



### Natural Language Reinforcement Learning (https://arxiv.org/abs/2411.14251)
Comments:
          Extension of arXiv:2402.07157

- **What's New**: 이번 연구는 Natural Language Reinforcement Learning (NLRL)이라는 새로운 가능성을 탐색합니다. 전통적인 Markov Decision Process (MDP)를 자연어 기반 표현 공간으로 확장하여 RL 원칙의 재정의를 시도합니다. 이는 게임, 로봇공학 및 언어 모델 등 다양한 분야에서의 기본 원칙을 혁신적으로 변형시킵니다.

- **Technical Details**: NLRL은 RL의 작업 목표(task objectives), 정책(policy), 가치 함수(value function), Bellman 방정식(Bellman equation), 정책 반복(policy iteration) 등의 개념을 자연어로 재정의합니다. 최근의 대형 언어 모델(LLMs)의 발전 덕분에 NLRL은 RL과 유사한 정책 및 가치 개선을 달성할 수 있는 실용적인 방법으로 제안됩니다.

- **Performance Highlights**: Maze, Breakthrough 및 Tic-Tac-Toe 게임을 통한 실험 결과 NLRL 프레임워크의 효율성 및 해석 가능성이 입증되었습니다. 이러한 실험은 다양하고 실제적인 사용 사례에 걸쳐 성과를 보여줍니다. 연구진은 이와 관련된 코드를 공개할 예정입니다.



### Visual Contexts Clarify Ambiguous Expressions: A Benchmark Datas (https://arxiv.org/abs/2411.14137)
- **What's New**: 이번 연구에서는 VAGUE라는 새로운 멀티모달 벤치마크를 제안합니다. 이 벤치마크는 3.9K 개의 간접적인 인간 발화와 해당하는 시나리오를 짝지어 구성되어 있습니다. 주목할 점은 이 연구가 모델의 복잡한 언어 및 시각적 추론 능력을 심층적으로 평가하고, 모델이 비유적 언어와 은유를 이해하는 데 필요한 접근 방식을 어떻게 발전시킬 수 있는지를 탐구한다는 것입니다.

- **Technical Details**: VAGUE는 비유적 언어와 간접적인 의사소통을 다루는 것을 목표로 하며, 이 과정에서 제공된 시각적 장면에서 맥락적 및 관계적 정보를 활용할 수 있도록 합니다. 연구진은 이 데이터를 바탕으로 주어진 간접 발화에서 숨겨진 의도를 분석할 수 있도록 모델의 기능을 발전시키고자 합니다. 여러 멀티모달 모델에 대한 평가 결과, 현재의 주류 모델들은 여전히 이러한 간접적인 의사소통을 처리하는 데 있어 어려움을 겪고 있음을 보여주었습니다.

- **Performance Highlights**: VAGUE는 여러 모델을 다양한 응답 형식으로 실험하고, 부정확한 선택을 유도하는 요소들을 분석할 수 있는 세심하게 설계된 다중 선택지 기반의 문제를 포함하고 있습니다. 이 벤치마크를 통해 다양한 모델이 간접 발화를 해석하는 데 얼마나 능숙한지를 평가하는 동시에, 향후 연구가 모델의 상호작용 능력을 더욱 발전시키는 데 기여할 수 있을 것으로 기대합니다.



### BEST-STD: Bidirectional Mamba-Enhanced Speech Tokenization for Spoken Term Detection (https://arxiv.org/abs/2411.14100)
Comments:
          Submitted to ICASSP 2025

- **What's New**: 이번 연구에서는 기존의 프레임 수준 특징에 의존하고, 계산적으로 복잡한 DTW(동적 시간 왜곡) 기반 템플릿 매칭 방식의 한계를 극복하기 위해 새로운 접근법을 제안합니다. 우리는 음성을 불연속적으로 인코딩하여 화자에 구애받지 않는 의미 있는 토큰을 생성하여, 텍스트 기반 검색 알고리즘을 통해 신속하게 검색할 수 있도록 합니다. 또한, 다양한 발화에서 일관된 토큰 시퀀스를 생성하는 데 주력하여, 역시 OOV(out-of-vocabulary) 단어를 효과적으로 처리합니다.

- **Technical Details**: Mamba 인코더 내에서 양방향 상태 공간 모델링을 통해 맥락적 프레임 수준 특징을 학습하고, 이를 불연속적인 토큰으로 인코딩하게 됩니다. 우리는 새로운 양방향 Mamba 인코더를 통해 음성을 맥락적 임베딩 시퀀스로 변환하고, 이를 다시 토큰 시퀀스로 변환하는 방법을 제안합니다. 이 과정에서, 우리는 DTW를 활용하여 동일한 단어의 발화를 정렬하고, 앵커-포지티브 쌍을 프레임 수준으로 구성하여 훈련합니다.

- **Performance Highlights**: 우리의 방법은 LibriSpeech와 TIMIT 데이터베이스에서 기존의 STD 기반선보다 우수한 성능을 보였습니다. 실험 결과, 새로운 토큰 생성 방식이 기존의 스피치 토크나이저보다 더 큰 화자 불변성을 나타내며, STD 작업에 더욱 적합하다는 것을 확인했습니다. 따라서 우리의 연구는 음성 인식 분야에 큰 기여를 할 수 있을 것으로 기대됩니다.



### MMGenBench: Evaluating the Limits of LMMs from the Text-to-Image Generation Perspectiv (https://arxiv.org/abs/2411.14062)
Comments:
          This project is available at: this https URL

- **What's New**: 이 논문은 Large Multimodal Models (LMMs)의 이미지 생성을 평가하기 위한 직관적인 자동 평가 파이프라인을 제안합니다. 기존의 벤치마크가 주로 이미지 이해에 중점을 두었다면, 본 연구는 LMM의 이미지 생성 능력을 포함하여 보다 포괄적인 평가에 중점을 두고 있습니다. 또한, MMGenBench-Test와 MMGenBench-Domain이라는 새로운 벤치마크를 도입하여 13개의 다양한 이미지 패턴을 평가하고, 생성 이미지 도메인에서 LMM의 성능을 분석합니다.

- **Technical Details**: 제안된 파이프라인은 LMM이 주어진 입력 이미지를 기반으로 이미지 프롬프트(image-prompt)를 생성하도록 요구합니다. 이후, 이 프롬프트를 바탕으로 텍스트-투-이미지(text-to-image) 생성 모델을 이용해 새 이미지를 생성하고, 최종적으로 원본 이미지와 생성된 이미지를 비교하여 성능을 평가합니다. 이 과정은 이미지 이해 및 설명에 대한 LMM의 성능을 평가하기 위해 세 가지 구성 요소로 이루어져 있습니다: 이미지 프롬프트 생성, 새로운 이미지 생성 및 정량적 메트릭 계산.

- **Performance Highlights**: 50개 이상의 인기 있는 LMM을 대상으로 한 평가 결과, 기존 벤치마크에서는 우수한 성과를 보이는 LMM이 기본적인 이미지 이해 및 설명 작업을 적절히 수행하지 못한다는 사실이 드러났습니다. 이 연구는 LMM의 성능 개선 가능성을 강조하며, 생성 이미지 도메인에서 더 나은 모델 최적화를 위한 방향성을 제시합니다. MMGenBench를 통해 LMM의 성능을 다양한 도메인에서 효율적으로 평가할 수 있는 유연하고 확장 가능한 벤치마킹 도구를 제공합니다.



### Logic Augmented Generation (https://arxiv.org/abs/2411.14012)
Comments:
          10 pages, 2 figures

- **What's New**: 이 논문에서는 Semantic Knowledge Graphs (SKGs)와 Large Language Models (LLMs) 간의 한계점을 극복하기 위한 Logic Augmented Generation (LAG)이라는 새로운 패러다임을 제안합니다. LAG는 SKGs가 제공하는 명확한 논리적 경계와 LLM이 가진 유연성을 결합하여, 집단 지성을 활용한 의료 진단 및 기후 서비스와 같은 다양한 분야에서의 효과적인 협업을 지원합니다.

- **Technical Details**: LAG는 LLMs를 Reactive Continuous Knowledge Graphs (RCKGs)로 활용하여, 비구조적 데이터를 처리하고 즉각적으로 컨텍스트 기반의 지식을 생성하는 데 초점을 맞춥니다. 이러한 접근 방식은 정보의 논리적 일관성을 보장하며, SKGs의 구조화된 지식과 LLM의 유연성을 통합하여 하이브리드 시스템을 구성합니다. 이 시스템은 텍스트, 음성, 이미지 등 다양한 모달리티의 데이터를 통합하여 지식을 생성합니다.

- **Performance Highlights**: 연구에서 제안된 LAG 시스템은 의료 진단 및 기후 예측 지원을 위해 두 가지 집단 지성 과제를 통해 성능을 입증하였습니다. RCKG는 LLM의 지속적인 학습 능력을 활용하여 무한한 지식을 생성할 수 있으며, 이는 전문가들 간의 효과적인 협업을 촉진하고 잘못된 진단을 최소화할 수 있습니다. 또한, SKGs와 LLM의 결합은 복잡한 문제 공간에서 해답을 도출하는 데 필요한 해석 가능성과 신뢰성을 높입니다.



### Sentiment Analysis of Economic Text: A Lexicon-Based Approach (https://arxiv.org/abs/2411.13958)
Comments:
          37 pages, 9 figures, 6 tables, in press

- **What's New**: 이 논문에서는 경제학 분야의 텍스트 응용을 위해 특별히 설계된 Economic Lexicon (EL)을 제안합니다. EL은 경제 개념에 대한 문서에서 사용되는 용어들을 광범위하게 포함하고, 각 용어에 대해 인간이 주석을 단 감정 점수를 제공합니다. 기존의 데이터베이스와 비교했을 때, EL은 더 넓은 용어 범위와 정확한 감정 분류를 제공합니다.

- **Technical Details**: EL 구축의 주요 과제는 경제 응용에서 감정을 전달하는 단어 세트를 선택하는 것과 각 단어의 톤을 정량화하는 것입니다. 이를 위해 1300만 개의 뉴스 기사를 포함하는 방대한 말뭉치에서 단어를 선정하고, 미시적인 의미 변화를 포착하기 위해 각 경제 개념과 관련된 문장에서 중요 단어를 식별합니다. 또한, 10명의 인간 주석자에게 각 단어에 대한 감정 점수를 요청하여 총 7천 개 이상의 단어에 대한 주관적인 감정을 정량화합니다.

- **Performance Highlights**: 주요 결과로, EL은 다른 사전들에 비해 경제적 불확실성, 소비자 감정, 경기 후퇴 예측에서 상관관계가 높다는 것을 보여줍니다. 특히, EL을 사용하여 계산된 Economic Pessimism (EP) 지표는 통계적으로 유의미하며, 다른 EP 지표들을 포함했을 때에도 그 중요성을 잃지 않습니다. 이는 EL이 더 많은 경제 용어를 포함하고 더 정확한 감정 분류를 기반으로 하여 우수한 성능을 발휘하기 때문입니다.



### Robust Detection of Watermarks for Large Language Models Under Human Edits (https://arxiv.org/abs/2411.13868)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)로 생성된 텍스트의 수신자에 대한 신뢰성과 효율적인 탐지를 위한 새로운 방법인 Tr-GoF를 소개합니다. 기존 방법의 약점을 극복하기 위해 Tr-GoF는 인간의 수정이 가해진 상황에서도 최적의 탐지를 수행할 수 있게 설계되었습니다. 본 방법은 인간 수정 수준이나 LLM의 확률적 사양에 대한 명확한 지식 없이도 적응적으로 성능을 발휘합니다.

- **Technical Details**: Tr-GoF는 수정된 텍스트에서 워터마크를 감지하기 위해 혼합 모델 탐지(mixture model detection)을 사용하여 인간 수정의 영향을 모델링합니다. 이 방법은 Gumbel-max 워터마크의 강력한 탐지를 위해 최적성을 인정받으며, 텍스트 수정의 수가 클 때에도 우수한 성능을 보입니다. 또한, 기존의 합계 기반 탐지 규칙은 수정된 통계에서 신뢰성을 잃는 반면, Tr-GoF는 더욱 탄탄한 이론적 보장을 제공합니다.

- **Performance Highlights**: Tr-GoF 테스트는 합성 데이터와 OPT, LLaMA와 같은 오픈 소스 LLM에서 경쟁력 있고 종종 우수한 성능을 보입니다. 기존 탐지 방법들이 5%의 수정된 토큰에 대해 87.8%에서 64.7%, 그리고 30.2%로 성능이 급락하는 것과 대조적으로, Tr-GoF는 다양한 인간 수정의 영향을 최소화하여 효율적인 탐지를 수행합니다.



### HARec: Hyperbolic Graph-LLM Alignment for Exploration and Exploitation in Recommender Systems (https://arxiv.org/abs/2411.13865)
- **What's New**: 이번 연구에서 제안하는 HARec은 사용자-아이템 협업 정보와 텍스트 묘사를 하이퍼볼릭 공간에서 공동 정렬하여 사용자 경험을 개선하는 혁신적인 프레임워크입니다. 이는(1) 의미 이해를 강화하는 계층 인지 그래프-LLM 정렬 메커니즘과(2) 사용자 조정 가능한 탐색-착취 무역의 하이퍼볼릭 계층 구조를 포함합니다.

- **Technical Details**: HARec은 탐색(exploration)과 착취(exploitation) 간의 균형을 유지하며 사용자 맞춤형 추천을 가능하게 합니다. 이 프레임워크는 하이퍼볼릭 공간 내에서 텍스트 묘사와 사용자-아이템 협업 정보를 통합하여 노이즈에 대한 민감도를 줄이고, 기존 추천 시스템의 계층 구조를 보다 정밀하게 모델링합니다. 이러한 접근 방식은 추천 정확도와 다양성 모두에서 성능을 크게 향상시킵니다.

- **Performance Highlights**: HARec은 다양한 실험을 통해 유틸리티 지표에서 최대 5.49% 향상, 다양성 지표에서 11.39% 증가를 기록하며 기존의 유클리드 및 하이퍼볼릭 기준 모델들을 일관되게 초월하는 성능을 보여줍니다. 이 모델은 정확도와 다양성을 동시에 우수한 성과로 달성한 첫 번째 예시로, 맥락에 맞는 추천의 질을 높였습니다.



### A Framework for Evaluating LLMs Under Task Indeterminacy (https://arxiv.org/abs/2411.13760)
Comments:
          To Appear in NeurIPS 2024 Workshops on Evaluating Evaluations (EvalEval) and Statistical Foundations of LLMs and Foundation Models (SFLLM)

- **What's New**: 이 논문에서는 큰 언어 모델(LLM)의 평가에서 발생할 수 있는 작업 불확실성(task indeterminacy)을 다루는 새로운 평가 프레임워크를 개발했습니다. 기존의 평가 방법들은 종종 각각의 평가 항목에 대한 단일 정답인 '골드 레이블'(gold label)에 의존했으나, 작업이 모호하거나 애매할 경우 여러 정답이 존재할 수 있음을 주장하고 있습니다. 이에 따라 저자들은 LLM의 진정한 성능을 과소평가할 수 있는 '골드 레이블' 가정의 한계를 지적합니다.

- **Technical Details**: 저자들은 인과 지향 아키텍처(directed acyclic graph, DAG)를 사용하여 작업 지정(task specification), 인간 평가(human ratings), LLM 응답 간의 관계를 분석했습니다. 이 프레임워크는 다양한 요소들—예를 들어, 작업의 명확성과 인간 평가의 오류—가 LLM 성능 추정에 미치는 영향을 분해하여 설명합니다. 이러한 접근 방식은 인간 평가 과정의 다양한 변동 요인을 분리하고, 각각의 요인이 LLM 평가에 미치는 영향을 계량화하는 방법을 제공합니다.

- **Performance Highlights**: 실험 결과, 기존의 골드 레이블 기반 평가 방식이 LLM의 진정한 성능을 과소평가함을 입증했습니다. LLM 평가가 수행되는 과정에서 각 인간 평가의 신뢰성과 변동성을 고려하여, 오류 조정된 성능 구간(error-adjusted performance interval)을 추정하는 방법을 제안합니다. 이는 연구 공동체에 대한 LLM 평가의 중요성과 이를 기반으로 한 미래 연구 방향을 제시하며, 평가 방법론에 대한 새로운 통찰을 제공합니다.



### Test Security in Remote Testing Age: Perspectives from Process Data Analytics and AI (https://arxiv.org/abs/2411.13699)
Comments:
          23 pages, 8 figures

- **What's New**: 이번 논문은 COVID-19 팬데믹이 원격 감독 하에 고위험(high-stakes) 평가의 시행과 수용을 가속화했다는 점을 강조합니다. 이러한 테스트의 유연한 운영은 많은 가치를 제공하지만, 테스트 보안 관련 문제를 부각시키고 있습니다. 최근 AI의 발전을 통해 신뢰할 수 있는 응답 생성을 가능하게 하는 다양한 도구가 등장하며, 이는 테스트 보안 연구의 필요성을 제기합니다.

- **Technical Details**: 이 논문에서는 클릭스트림(process data based on clickstream) 데이터 분석을 이용하여 응시자의 테스트 과정에 대한 심층적인 통찰을 얻을 수 있는 방법을 제안합니다. 특히, AI 방법들이 통계 분석을 넘어서는 중요한 역할을 할 수 있는 가능성을 제시합니다. 이와 같은 기술적 접근은 원격으로 시행되는 고위험 테스트의 보안을 강화하는 데 큰 도움이 될 것으로 기대됩니다.

- **Performance Highlights**: 실제 사례를 통해 이러한 데이터 분석 및 AI 기법이 테스트 보안을 어떻게 강화할 수 있는지를 보여줍니다. 더 나아가, 테스트 응시 과정에서 발생하는 다양한 데이터들을 활용하여 보다 정교한 보안 시스템을 구축할 수 있는 가능성을 탐구합니다. 이 연구는 원격 테스트 환경에서의 보안 강화를 위한 새로운 패러다임을 제시하고 있습니다.



### Retrieval-Augmented Generation for Domain-Specific Question Answering: A Case Study on Pittsburgh and CMU (https://arxiv.org/abs/2411.13691)
- **What's New**: 이번 연구에서는 Retrieval-Augmented Generation(RAG) 시스템을 설계하여 대규모 언어 모델에 피츠버그와 카네기 멜론 대학교(CMU)에 대한 도메인 특화 질문에 대한 적절한 문서를 제공하는 방법을 소개합니다. 총 1,800개 이상의 하위 페이지를 추출하여 수작업 및 Mistral 생성 질문-답변 쌍을 결합하여 데이터 주석을 수행하였으며, 이로써 높은 상호 주석자 일치도(IAA) 점수인 0.7625를 달성했습니다. RAG 프레임워크는 문서 검색 정확성을 향상시키기 위해 BM25 및 FAISS 검색기를 통합하고 reranker를 추가하여 성능을 증대시켰습니다.

- **Technical Details**: 우리의 RAG 시스템 설계에는 다양한 데이터 출처에서 정보를 수집하는 프로세스가 포함됩니다. 피츠버그와 CMU와 관련된 역사, 행사, 문화 및 정부 정보를 포함하여 우리가 수집한 데이터는 약 1820개의 하위 페이지와 7개의 PDF, 16개의 테이블로 구성됩니다. 데이터를 주석 달기 위해 Mistral 모델을 이용하여 수작업과 자동 생성을 결합하였으며, 총 1,467개의 질문-답변 쌍을 생성했습니다. 각 QA 쌍은 시계적 요인의 영향을 받는지에 대한 이진 레이블이 부여되었습니다.

- **Performance Highlights**: RAG 시스템은 비-RAG 기준선 모델보다 현저히 우수한 성과를 보이며, 특히 시간에 민감하고 복잡한 질문에 대해 더 높은 F1 점수의 향상(5.45%에서 42.21%까지)과 56.18%의 재현율을 기록했습니다. 이러한 결과는 RAG 시스템이 답변의 정확성과 적합성을 향상시키는 데 큰 잠재력을 지니고 있음을 보여주며, 문서 검색 및 모델 훈련에서의 추가 최적화 필요성을 강조합니다.



### RadPhi-3: Small Language Models for Radiology (https://arxiv.org/abs/2411.13604)
- **What's New**: 최근 LLM 기반의 코파일럿(Copilot) 도우미들이 일상 업무와 방사선학(。
Radiology) 워크플로우에서의 유용성으로 주목받고 있습니다. 본 논문에서는 Radiology 업무를 지원하기 위해 3.8B 파라미터를 가진 RadPhi-3라는 소형 언어 모델(Small Language Model)이 새롭게 소개되었습니다. 기존에 흉부 X-ray 관련 보고서에서의 요약 생성 외에도, 현재 보고서와 이전 보고서를 비교한 변화 요약 생성 및 병리학적 특성과 장치 태깅 등 다양한 유용한 작업들을 탐색하였습니다.

- **Technical Details**: RadPhi-3는 Phi-3-mini-4k-instruct 모델에서 명령 튜닝을 통해 학습되었습니다. 이는 방사선 전문 지식 소스를 이용해 방사선학 관련 질문에 대해 신뢰성 있는 답변을 제공하는 데 중점을 두었습니다. 방사선학 모델은 개인 기록에 대한 모델 성능을 귀속시키지 않는 개인정보 보호 요구사항을 준수해야 하며, RadPhi-3는 다양한 방사선 보고서 작업의 성과를 이전 모델인 RadPhi-2보다 개선하였습니다.

- **Performance Highlights**: RadPhi-3는 RaLEs 방사선 보고서 생성 벤치마크에서 SOTA(State Of The Art) 성과를 달성하였으며, 흉부 X-ray 관련 작업 및 질문 응답 작업에서 우수한 결과를 기록했습니다. 또한 새로운 두가지 방사선 보고서 관련 작업인 방사선 보고서 세분화와 변경 요약 작업에 대해 성능 평가를 실시하여 유의미한 결과를 도출했습니다. 이 모델은 여러 벤치마크에서 방사선 보고서의 각종 작업 수행 능력을 입증하였습니다.



### Improved GUI Grounding via Iterative Narrowing (https://arxiv.org/abs/2411.13591)
- **What's New**: 이번 연구에서는 GUI Grounding 능력을 강화하기 위해 Iterative Narrowing (IN)이라는 시각적 프롬프트 프레임워크를 제안했습니다. 기존의 VLM 모델의 성능 개선을 목표로 하며, GUI 인터페이스에서의 정밀한 시각적 위치 식별이 가능해집니다. IN은 초기 예측을 점진적으로 개선하는 과정을 통해 GUI grounding의 정확성을 향상시킵니다.

- **Technical Details**: 이 방법은 입력 이미지와 해당 텍스트 쿼리를 받으면, 이미지를 999×999 픽셀로 표준화하여 처리합니다. 모델은 다음 반복을 위해 예측된 좌표를 기반으로 이미지 잘라내기를 생성하며, 매 반복마다 이미지 크기를 줄이는 방식을 사용합니다. 이 반복 과정은 n 회 반복되며, 마지막 반복에서는 최종 타겟 위치를 결정합니다.

- **Performance Highlights**: ScreenSpot 벤치마크를 통한 평가 결과, IN 프레임워크는 특히 일반 VLM 모델인 InternVL-2-4B 및 Qwen2-VL-7B에서 성능 개선을 이끌어냈습니다. 그러나 공간적으로 거리가 큰 컨텍스트 단서 처리에 한계가 있어, 특정 상황에서 성능이 저하되는 경향이 있습니다. 향후 연구에서는 이러한 컨텍스트 한계를 해결하기 위한 방법에 대한 탐색이 필요합니다.



### WavChat: A Survey of Spoken Dialogue Models (https://arxiv.org/abs/2411.13577)
Comments:
          60 papes, working in progress

- **What's New**: 최근 음성 대화 모델, 특히 GPT-4o와 같은 시스템의 발전이 음성 분야에서 큰 주목을 받고 있습니다. 기존의 3단계 연계 음성 대화 모델과 비교하여, 현대의 음성 대화 모델은 지능이 뛰어나고, 오디오 및 음악 이해는 물론 음성의 스타일과 음색 특성을 포착할 수 있습니다. 이 연구에서는 음성 대화 시스템에 대한 종합적인 조사와 기술 분석의 필요성을 강조하고 이를 해결하기 위해 다양한 시스템을 정리하고 카테고리화했습니다.

- **Technical Details**: 본 논문에서는 음성 대화 모델의 구성 요소와 기술적 도전 과제를 다루고 있습니다. 특히, 음성 표현 (speech representation), 훈련 패러다임 (training paradigm), 스트리밍과 인터랙션 기능에 대한 심층 분석을 제공합니다. 기술적 제한사항을 언급하며, 차세대 연구 방향을 제안합니다.

- **Performance Highlights**: 지능형 음성 대화 시스템은 자연스럽고 인간 같은 응답을 생성하는 성능을 보여주며, 음향 특성 이해와 생성 능력에서도 뛰어난 성능을 발휘합니다. 이 모델들은 음악 및 오디오 이벤트 처리에서 강력한 기능이 있으며, 실시간 상호작용을 통한 낮은 지연 시간의 대화 경험을 제공합니다. 이러한 특성들은 음성 대화 모델을 전통적인 모델과 차별화 시키는 주된 요소입니다.



### Source Code Foundation Models are Transferable Binary Analysis Knowledge Bases (https://arxiv.org/abs/2405.19581)
- **What's New**: 이번 연구에서는 인간 지향 이진 역공학(Human-Oriented Binary Reverse Engineering, HOBRE)을 위한 새로운 프레임워크인 ProRec을 제안합니다. 이 프레임워크는 이진 코드와 소스 코드 간의 의미적 격차를 줄이며, 기본 코드 모델의 장점을 활용하여 보다 효과적인 역공학을 가능하게 합니다. ProRec은 이진-소스 인코더-디코더 모델 및 블랙박스 LLMs를 통합하여 이진 분석의 정확성을 높이는데 기여합니다.

- **Technical Details**: ProRec는 이진 파일을 입력으로 받고, 이진 분석 도구를 사용하여 각 이진 함수에 대한 코드 형태를 초래합니다. 버튼 모델과 SCFM을 결합하여 코드 조각을 생성하고, 이를 바탕으로 블랙박스 LLM이 이진 함수 내용을 분석하여 더 나은 결과를 도출합니다. 프로버와 리커버러의 구조 및 훈련 과정은 새로운 지식 탐사 단계를 포함하여 상세히 설명됩니다.

- **Performance Highlights**: ProRec은 이진 요약 및 함수 이름 복구의 두 가지 핵심 작업에서 성능을 크게 향상시킵니다. 요약 작업에서는 CHRF에서 10.3%의 상대적 증가를 보였으며, GPT4 기반 메트릭에서도 16.7%의 개선을 나타냈습니다. 함수 이름 복구 작업에서도 각각 6.7% 및 7.4%의 절대적 증가로 더욱 향상된 정밀도와 재현율을 기록하고 있습니다.



### CodeArt: Better Code Models by Attention Regularization When Symbols Are Lacking (https://arxiv.org/abs/2402.11842)
- **What's New**: 이 논문에서는 기호(symbol)가 부족할 때 코드 모델의 성능 저하 문제를 해결하기 위해 새로운 방법을 제안합니다. 기존의 Transformer 기반 코드 모델이 기호에 의존하는 것과 달리, 프로그램 분석(program analysis)을 통해 사전(context)을 추출하는 방식으로 접근합니다. 이 방법은 기호 없이도 모델이 올바른 상관관계(correlations)와 맥락(contexts)에 주목하도록 합니다.

- **Technical Details**: 제안된 방법은 기존 BERT 모델의 토큰화(tokenization)와 아키텍처(model architecture)를 개선하고, 주의 마스킹(attention masking) 기법을 활용하여 모델이 특정한 맥락(contexts)에만 주의(attend)하도록 합니다. 구체적으로, 양방향 프로그램 의존성 전이 닫힘(bi-directional program dependence transitive closures)과 토큰 동시 발생(token co-occurrences) 정보를 사용하여 주의를 제한합니다. 이를 위해 2,600만 개의 변형된 바이너리 함수 데이터셋을 사용하여 모델을 처음부터 사전 학습(pre-train)합니다.

- **Performance Highlights**: 제안된 사전 학습 모델은 이 모델을 활용한 3가지 다운스트림 작업에서 기존의 최첨단 방법(state-of-the-art)을 초과하는 성과를 보였습니다. 이 작업들에서 바이너리 유사성(binary similarity)은 53%에서 64%로, 유형 추론(type inference)은 49%에서 60%로, 멀웨어 가족 분류(malware family classification)는 74%에서 94%로 향상되었습니다. 또한, 일반적인 코드 이해 모델의 사전 학습 기법에 비해 매우 우수한 성능을 보여주었습니다.



### SRA-MCTS: Self-driven Reasoning Augmentation with Monte Carlo Tree Search for Code Generation (https://arxiv.org/abs/2411.11053)
- **What's New**: 본 논문은 모델이 고품질의 중간 추론 경로를 자율적으로 생성할 수 있도록 안내하는 SRA-MCTS라는 데이터 생성 프로세스를 제안합니다. 이는 재귀적인 프로세스를 통해 모델에 새로운 추론 능력을 부여하고, 코드 생성의 복잡한 문제 해결에 대한 성공률을 향상시키는 데 도움을 줍니다. 이 접근 방식은 추가적인 감독 없이도 성능을 높이아는 잠재력을 보이며, 작은 모델에서도 효과적으로 작동합니다.

- **Technical Details**: SRA-MCTS는 여러 단계로 구성된 데이터 생성 방법론으로, 계획 생성부터 코드로의 변환 및 모델 훈련에 이르는 세 가지 단계를 포함합니다. 이 과정에서, 모델은 Monte Carlo Tree Search (MCTS)를 기반으로 하여 다채로운 자연어 계획을 생성하고, 생성된 계획에 따라 코드를 작성합니다. 모델은 이전의 답변을 참조하여 생성의 적절성과 다양성, 오답 가능성을 줄이는 데 집중합니다.

- **Performance Highlights**: 실험 결과, SRA-MCTS로부터 생성된 데이터를 사용한 파인 튜닝 모델은 기존의 Chain-of-Thought (CoT) 방식이나 공식 모델보다 뛰어난 성과를 보여줍니다. 특히, 응답의 다양성이 중요한 역할을 하며, 작은 모델에서도 자가 개선이 이루어지는 점은 주목할 만한 결과입니다. 이 방식은 특히 CoT 접근 방식이 성능 저하를 경험할 때에도 강력함을 유지하며, 높은 다양성 지표에서의 개선을 관찰할 수 있었습니다.



New uploads on arXiv(cs.IR)

### Topology-Aware Popularity Debiasing via Simplicial Complexes (https://arxiv.org/abs/2411.13892)
- **What's New**: 이 논문에서는 추천 시스템의 인기 편향 문제를 해결하기 위해 새로운 토폴로지 인지 인기 디바이싱 프레임워크인 Test-time Simplicial Propagation (TSP)을 제안합니다. TSP는 심플리시얼 복합체(simplicial complexes)를 포함하여 GNN의 표현 능력을 향상시키고, 다차원 관계를 효과적으로 모델링합니다. 이를 통해 기존의 방법들이 직면한 인기 아이템에 대한 편향을 완화하고 사용자-아이템 상호작용을 보다 정확히 반영할 수 있는 기회를 제공하고자 합니다.

- **Technical Details**: 이 연구는 GNN 기반 추천 방법의 문제점을 해결하기 위해 심플리시얼 복합체(SCs)를 활용하여 사용자와 아이템 간의 복잡한 상호작용을 모델링합니다. TSP는 그래프 증강(graph augmentation)으로 긴 아이템의 이웃을 풍부하게 하며, 심플리시얼 복합체를 사용하여 다차원 관계를 포착합니다. 메시지 전파(message passing)를 통해 SC의 정보가 결합되어 보다 균형 잡힌 추천을 가능하게 합니다. 또한 TSP 모듈은 플러그 앤 플레이 솔루션으로, 기존 모델에 추가적인 파라미터 조정 없이 간편하게 통합될 수 있도록 설계되었습니다.

- **Performance Highlights**: 다섯 개의 실제 데이터셋을 통한 광범위한 실험 결과, 제안된 TSP 방법은 특히 긴 꼬리(long-tail) 추천 과제에서 뛰어난 성능을 보였습니다. 시각화 결과는 TSP가 아이템 표현의 보다 고른 분포를 생성하여 공정하고 정확한 추천 결과를 제공할 수 있음을 확인하였습니다. 이 연구는 추천 시스템 분야에서의 인기 편향 문제에 대한 새로운 접근 방식을 제시하며, 성능 개선의 기회를 제시합니다.



### HARec: Hyperbolic Graph-LLM Alignment for Exploration and Exploitation in Recommender Systems (https://arxiv.org/abs/2411.13865)
- **What's New**: 이번 연구에서 제안하는 HARec은 사용자-아이템 협업 정보와 텍스트 묘사를 하이퍼볼릭 공간에서 공동 정렬하여 사용자 경험을 개선하는 혁신적인 프레임워크입니다. 이는(1) 의미 이해를 강화하는 계층 인지 그래프-LLM 정렬 메커니즘과(2) 사용자 조정 가능한 탐색-착취 무역의 하이퍼볼릭 계층 구조를 포함합니다.

- **Technical Details**: HARec은 탐색(exploration)과 착취(exploitation) 간의 균형을 유지하며 사용자 맞춤형 추천을 가능하게 합니다. 이 프레임워크는 하이퍼볼릭 공간 내에서 텍스트 묘사와 사용자-아이템 협업 정보를 통합하여 노이즈에 대한 민감도를 줄이고, 기존 추천 시스템의 계층 구조를 보다 정밀하게 모델링합니다. 이러한 접근 방식은 추천 정확도와 다양성 모두에서 성능을 크게 향상시킵니다.

- **Performance Highlights**: HARec은 다양한 실험을 통해 유틸리티 지표에서 최대 5.49% 향상, 다양성 지표에서 11.39% 증가를 기록하며 기존의 유클리드 및 하이퍼볼릭 기준 모델들을 일관되게 초월하는 성능을 보여줍니다. 이 모델은 정확도와 다양성을 동시에 우수한 성과로 달성한 첫 번째 예시로, 맥락에 맞는 추천의 질을 높였습니다.



### LEADRE: Multi-Faceted Knowledge Enhanced LLM Empowered Display Advertisement Recommender System (https://arxiv.org/abs/2411.13789)
- **What's New**: 본 논문은 LLM(대규모 언어 모델)을 활용하여 온라인 디스플레이 광고 시스템의 성과를 극대화하는 새로운 프레임워크인 LEADRE(LLM Empowered Display ADvertisement REcommender)를 제안합니다. LEADRE는 사용자의 관심을 포착하고, LLM과 광고 시스템 간의 지식 격차를 해소하며, 효율적인 배포 방법을 찾아내는 세 가지 주요 모듈로 구성됩니다. 기존 광고 시스템의 한계점을 극복하며, 사용자 맞춤형 광고 생성을 목표로 합니다.

- **Technical Details**: LEADRE는 세 가지 핵심 모듈로 구성되며, 첫 번째 모듈인 Intent-Aware Prompt Engineering은 사용자의 상업적 의도를 포착하기 위해 사용자의 장기 및 단기 관심사를 반영한 프롬프트-응답 쌍을 설계합니다. 두 번째 모듈인 Advertising-Specific Knowledge Alignment은 언어적 모드와 광고 시스템 간의 의미적 갭을 좁히고, 비즈니스 가치를 중심으로 광고를 생성하기 위한 Direct Preference Optimization(DPO)을 적용합니다. 마지막으로, Efficient System Deployment 모듈은 지연 허용 서비스와 지연 민감 서비스가 통합된 LLM 배포 아키텍처를 통해 컴퓨팅 효율성을 향상시킵니다.

- **Performance Highlights**: LEADRE는 온라인 A/B 테스트에서 Tencent의 WeChat Channels와 Moments에서 각각 1.57% 및 1.17%의 GMV(Gross Merchandise Value) 향상을 달성하며, 매일 수십억 건의 요청을 처리하는 시스템에 성공적으로 배포되었습니다. 실험 결과는 LEADRE의 각 모듈이 실질적인 기여를 했음을 보여주며, 사용자 맞춤형 광고 생성의 효과성을 검증합니다. 추가적으로, 광고 검색 과정에서 사용자 관심을 확장하고, 새로운 특징이 GMV에 추가적으로 1.43% 향상시키는 기여를 했습니다.



### A Collaborative Ensemble Framework for CTR Prediction (https://arxiv.org/abs/2411.13700)
- **What's New**: 본 논문은 Collaborative Ensemble Training Network (CETNet)라는 새로운 프레임워크를 제안하여, 개별 모델의 다양한 특성을 활용하여 추천 시스템의 성능을 향상시킵니다. CETNet은 여러 고유의 embedding 테이블을 가진 모델들을 협력적으로 학습하여, 각 모델이 고유한 feature interaction 패턴을 포착하도록 합니다. 이는 단순한 모델 확장 방식과는 달리, 모델 간의 협동을 통해 예측을 반복적으로 세련되게 만들며, 신뢰 기반의 융합 메커니즘을 도입하여 각 모델의 기여도를 동적으로 조정합니다.

- **Technical Details**: CETNet에서는 여러 가지 차별화된 embedding 테이블을 가진 여러 모델을 활용하여 순차적 및 계층적 feature 상호작용을 촉진합니다. 모델의 신뢰도를 바탕으로 기여도를 동적으로 조정하는 방법은 일반 softmax를 이용하며, 모델의 신뢰도는 negation entropy를 통해 계산합니다. 이 방식은 예측이 더욱 신뢰할 수 있는 모델에 의해 지배받도록 하여 서로 보완적인 모델의 강점도 활용할 수 있도록 합니다.

- **Performance Highlights**: 다양한 공개 데이터셋(AmazonElectronics, TaobaoAds, KuaiVideo) 및 메타의 대규모 산업 데이터셋에서 CETNet의 성능을 검증한 결과, 기존 개별 모델 및 최신 기법들보다 우수한 성능을 보였습니다. 또한 Criteo 및 Avazu 데이터셋에서도 실험을 진행하여, 상대적으로 더 작고 효율적인 embedding을 사용하며 멀티 임베딩 패러다임을 초월하는 성능을 발휘했습니다. 이러한 결과들은 CTR 예측 작업을 위한 확장 가능하고 효율적인 솔루션으로 증명되었습니다.



### OpenScholar: Synthesizing Scientific Literature with Retrieval-augmented LMs (https://arxiv.org/abs/2411.14199)
- **What's New**: OpenScholar는 4500만 개의 오픈 액세스 논문에서 관련 정보를 검색하고, 인용 기반 응답을 합성하여 과학적 질문에 대답하는 검색 보강된 대형 언어 모델(LM)입니다. 새로운 벤치마크인 ScholarQABench를 소개하여 OpenScholar의 성능을 평가했으며, 이는 컴퓨터 과학, 물리학, 신경과학, 생물 의학을 포함하는 2967개의 전문가 작성 쿼리와 208개의 긴 형식 응답으로 구성되어 있습니다. OpenScholar는 GPT-4o보다 5% 더 높은 정확성을 달성했으며, 인용 정확도가 인간 전문가와 동등한 수준으로 향상되었습니다.

- **Technical Details**: OpenScholar는 OpenScholar-DataStore(OSDS)를 사용하여 4500만 개의 오픈 액세스 논문과 2억 3700만 개의 구문 임베딩을 포함합니다. 이 시스템은 관련 구문을 검색하고, 이를 기반으로 반복적인 자기 피드백 생성을 통해 응답의 출력을 정제하는 방식으로 작동합니다. OpenScholar는 ‘효율적인 8B 모델’로 훈련되어 특정 도메인에 최적화된 검색 및 합성 기능을 제공하며, 모델 간의 결합을 통해 전체적인 정확성을 개선할 수 있습니다.

- **Performance Highlights**: OpenScholar는 GPT-4o와 PaperQA2를 포함한 다양한 모델들보다 우수한 성능을 보였으며, 70%의 경우 전문가 작성 응답보다 더 나은 결과를 제공했습니다. ScholarQABench에서 OpenScholar는 인용 정확도와 정보 범위에서 높은 성과를 보였고, GPT-4o의 일부 기능을 향상시킬 수 있는 가능성을 보여주었습니다. OpenScholar의 효율적인 구조는 비용 절감 효과를 가져올 수 있으며, 연구에 실질적으로 기여할 수 있는 고품질 출력을 생성합니다.



### BEST-STD: Bidirectional Mamba-Enhanced Speech Tokenization for Spoken Term Detection (https://arxiv.org/abs/2411.14100)
Comments:
          Submitted to ICASSP 2025

- **What's New**: 이번 연구에서는 기존의 프레임 수준 특징에 의존하고, 계산적으로 복잡한 DTW(동적 시간 왜곡) 기반 템플릿 매칭 방식의 한계를 극복하기 위해 새로운 접근법을 제안합니다. 우리는 음성을 불연속적으로 인코딩하여 화자에 구애받지 않는 의미 있는 토큰을 생성하여, 텍스트 기반 검색 알고리즘을 통해 신속하게 검색할 수 있도록 합니다. 또한, 다양한 발화에서 일관된 토큰 시퀀스를 생성하는 데 주력하여, 역시 OOV(out-of-vocabulary) 단어를 효과적으로 처리합니다.

- **Technical Details**: Mamba 인코더 내에서 양방향 상태 공간 모델링을 통해 맥락적 프레임 수준 특징을 학습하고, 이를 불연속적인 토큰으로 인코딩하게 됩니다. 우리는 새로운 양방향 Mamba 인코더를 통해 음성을 맥락적 임베딩 시퀀스로 변환하고, 이를 다시 토큰 시퀀스로 변환하는 방법을 제안합니다. 이 과정에서, 우리는 DTW를 활용하여 동일한 단어의 발화를 정렬하고, 앵커-포지티브 쌍을 프레임 수준으로 구성하여 훈련합니다.

- **Performance Highlights**: 우리의 방법은 LibriSpeech와 TIMIT 데이터베이스에서 기존의 STD 기반선보다 우수한 성능을 보였습니다. 실험 결과, 새로운 토큰 생성 방식이 기존의 스피치 토크나이저보다 더 큰 화자 불변성을 나타내며, STD 작업에 더욱 적합하다는 것을 확인했습니다. 따라서 우리의 연구는 음성 인식 분야에 큰 기여를 할 수 있을 것으로 기대됩니다.



New uploads on arXiv(cs.CV)

### Insight-V: Exploring Long-Chain Visual Reasoning with Multimodal Large Language Models (https://arxiv.org/abs/2411.14432)
- **What's New**: 이번 연구에서는 Insight-V를 제안하여 1) 복잡한 다중 모드 작업을 위한 긴 체계적인 추론 데이터를 확장 가능하게 생성하고 2) 다중 모드 대형 언어 모델(MLLM)의 추론 능력을 향상시키는 효과적인 훈련 파이프라인을 마련했습니다. Insight-V는 인적 자원 없이도 다양한 추론 경로를 생성할 수 있는 두 단계의 파이프라인을 설계하였으며, 다중 세분화 평가 방법을 통해 데이터 품질을 보장합니다.

- **Technical Details**: Insight-V는 두 개의 에이전트를 포함한 다중 에이전트 시스템을 구성하여 장기적인 체인 추론을 수행하는 추론 에이전트와 추론 결과를 요약하고 평가하는 요약 에이전트로 나뉩니다. 이러한 시스템은 DPO(Direct Preference Optimization) 알고리즘을 통합하여 추론 에이전트의 생성 안정성과 품질을 개선합니다. 이를 통해 임시적으로 생성된 데이터의 품질을 확보하고, 체계적인 추론 과정에서의 효과를 증대시킵니다.

- **Performance Highlights**: Insight-V를 통합한 LLaVA-NeXT 모델은 7개의 도전적인 시각 추론 벤치마크에서 평균 7.0% 향상된 성능을 보였으며, 단일 형태의 강력한 MLLM을 활용했을 때는 2.9%의 개선을 보여주었습니다. 이러한 성과는 다중 에이전트 시스템이 다중 모드 작업에서 인지 기반 추론을 가능하게 한다는 점을 강조하고, 향후 연구에 있어 중요한 기초를 제공하는 역할을 합니다.



### Stable Flow: Vital Layers for Training-Free Image Editing (https://arxiv.org/abs/2411.14430)
Comments:
          Project page is available at this https URL

- **What's New**: 최근 Diffusion 모델들은 콘텐츠 합성 및 편집 분야에서 혁신을 가져왔습니다. 특히, 전통적인 UNet 아키텍처 대신 Diffusion Transformer (DiT)로 대체되었고, 훈련 및 샘플링을 개선하기 위해 flow-matching 기법이 도입되었습니다. 하지만 이러한 모델들은 생성 다양성이 제한적이라는 경향이 있으며, 이번 연구에서는 이러한 한계를 이용해 일관된 이미지 편집을 구현합니다.

- **Technical Details**: DiT 모델의 이미지 편집을 위해 토대가 되는 여러 층의 중요성을 자동으로 식별하는 방법을 제안하였습니다. 구체적으로, 이미지 형성에 필수적인 '중요 층(vital layers)'을 찾아내는 방법을 도입하며, 각 층을 우회했을 때 이미지 내용의 편차를 측정하여 이들을 확인합니다. 또한, 전반적인 모델의 구조적 차이를 통해 DiT층에 주입된 주의(attention) 피쳐들의 효과적인 사용을 보여줍니다.

- **Performance Highlights**: 최종적으로, 우리 방법을 정성적 및 정량적으로 평가하며 사용자 연구를 통해 그 효용성도 입증하였습니다. 연구 결과, FLUX 모델을 통한 실제 이미지 편집 지원이 가능해졌으며, 다양한 이미지 편집 작업(예: 비강성 편집, 객체 추가 및 교체)에서 성공적인 적용 사례를 보여줍니다.



### Revisiting the Integration of Convolution and Attention for Vision Backbon (https://arxiv.org/abs/2411.14429)
Comments:
          NeurIPS 2024

- **What's New**: 이번 연구에서는 Convolutions (Convs)와 Multi-Head Self-Attentions (MHSAs)를 서로 다른 수준의 granularity에서 병렬로 사용하는 GLMix라는 새로운 통합 방안을 제안합니다. 이 방법은 Convs가 지역적 특징을 추출하고, MHSAs가 전역 상호작용을 학습하는 데 최적화되어 있습니다. 각각의 층에서 이미지를 세밀한 그리드와 대략적인 의미 있는 슬롯으로 표현하여, 이들 간의 특징을 효율적으로 융합할 수 있습니다.

- **Technical Details**: GLMix 블록에서는 Convs와 MHSAs의 결합을 통해 지역적 및 전역적 모델링을 달성합니다. soft clustering과 dispatching 모듈을 도입하여 세트와 그리드 표현 간의 연결을 가능하게 하여, 경량화된 Convs로 고해상도 특징을 추출하고 MHSAs로 제한된 수의 의미 슬롯을 처리하여 성능을 높입니다. 이를 통해 GLNet-STL이 82.5%의 top-1 정확도를 달성하고 있습니다.

- **Performance Highlights**: GLNet-4G, GLNet-9G, GLNet-16G 모델은 각각 83.7%, 84.5%, 85.0%의 top-1 정확도를 기록하며, 기존 최첨단 성능보다도 더 효율적인 결과를 보여줍니다. 다양한 컴퓨터 비전 작업에서 GLNet의 성능을 empirically 검증하였으며, soft clustering 모듈에서는 의미 있는 시맨틱 그룹화 효과가 나타났습니다. 이러한 결과들은 GLMix의 효과성을 강조하며, 향후 비지도형 시맨틱 세분화 접근법 개발에 영감을 줄 수 있습니다.



### Unleashing the Potential of Multi-modal Foundation Models and Video Diffusion for 4D Dynamic Physical Scene Simulation (https://arxiv.org/abs/2411.14423)
Comments:
          Homepage: this https URL

- **What's New**: 이 논문은 물리적 원리에 기반한 복잡한 객체 상호작용을 모델링하기 위해 다양한 재료 특성을 정확히 캡처하는 새로운 접근 방식을 소개합니다. 특히, 멀티모달 기반 모델과 비디오 확산(video diffusion)을 활용하여 4D 동적 장면 시뮬레이션의 정확성과 유연성을 개선합니다. 기존의 방법들의 한계를 극복하기 위해, 자료 파라미터를 자동으로 초기화하고, 물리적 관련성을 유지하는 새로운 기법을 제안합니다.

- **Technical Details**: 이 방법은 대규모 프리트레인(pre-trained) 비주얼 파운데이션 모델을 사용하여 재료 유형을 식별하고, 이미지 쿼리를 통해 재료 파라미터를 초기화합니다. 그런 후 3D Gaussian splats를 추론해 상세한 장면 표현을 생성하고, 비디오 확산을 통해 광학 흐름(optical flow)을 가이드로 사용하여 재료 파라미터를 미세 조정합니다. 이 통합된 프레임워크는 더 많은 재료 동작 유형을 지원하며 복잡한 동작을 효과적으로 모델링할 수 있게 합니다.

- **Performance Highlights**: 제안된 방법은 합성된 장면과 실제 장면 모두에서 신체적으로 실제적인 4D 동적 장면 시뮬레이션을 성공적으로 달성했습니다. 기존의 방법들과 비교할 때, 우리 접근 방식은 더 높은 정확도를 자랑하며, 다양한 재료 행동을 지원하여 시뮬레이션의 유연성과 정확성을 크게 향상시킵니다. 실험 결과에서 재료 파라미터 최적화의 효과성을 보여주며, 기존 방법의 제약을 넘어서는 성능을 입증하였습니다.



### Multimodal Autoregressive Pre-training of Large Vision Encoders (https://arxiv.org/abs/2411.14402)
Comments:
this https URL

- **What's New**: 이번 연구에서는 대규모 비전 인코더(pre-training of large-scale vision encoders)를 위한 새로운 방법을 소개합니다. 특히, 이미지와 텍스트를 포함하는 다중 모달(multimodal) 설정으로 이 프레임워크를 확장했습니다. AIMV2라는 일반화된 비전 인코더 계열은 간단한 사전 훈련 과정(지식 훈련)과 뛰어난 확장성, 다양한 다운스트림 작업에서의 놀라운 성능을 특징으로 합니다.

- **Technical Details**: 이 논문에서는 비전 인코더를 다중 모달 디코더(multimodal decoder)와 쌍을 이루어 원시 이미지 패치(raw image patches)와 텍스트 토큰(text tokens)을 자기 회귀적으로 생성하는 방식으로 설계했습니다. 이러한 접근 방식은 AIMV2-3B 인코더가 ImageNet-1k에서 89.5%의 정확도를 달성하도록 도움을 주었습니다. 또한, AIMV2는 위치 설정(localization), 기초 설정(grounding), 분류(classification) 등의 비전 기준 시리즈에서 높은 성능을 보입니다.

- **Performance Highlights**: AIMV2는 다중 모달 평가에서 뛰어난 성과를 나타내며, CLIP, SigLIP과 같은 최신 대비 모델(state-of-the-art contrastive models)보다 더 나은 성능을 발휘합니다. 다양한 설정에서 다중 모달 이미지 이해(multimodal image understanding)에 있어 AIMV2는 일관되게 뛰어난 결과를 보여줍니다. 이는 AIMV2의 우수성을 더욱 강조하는 요소로 작용합니다.



### Beyond Training: Dynamic Token Merging for Zero-Shot Video Understanding (https://arxiv.org/abs/2411.14401)
- **What's New**: 최근 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 발전으로 비디오 이해(video understanding) 분야에서 새로운 가능성이 열렸습니다. 그러나 제로샷(zero-shot) 비디오 작업에서 높은 충실도를 달성하는 것은 여전히 도전 과제가 되고 있습니다. 이 논문에서는 DYTO라는 새로운 동적 토큰 병합 프레임워크를 제안하며, 이는 토큰 효율성을 최적화하면서 중요한 장면 세부 사항을 보존합니다.

- **Technical Details**: DYTO는 계층적 프레임 선택(hierarchical frame selection)과 이분법적 토큰 병합(bipartite token merging) 전략을 통합하여 비디오의 중요한 프레임을 동적으로 클러스터링하고 선택적으로 토큰 시퀀스를 압축합니다. 이러한 방식으로 DYTO는 계산 효율성과 의미적 풍부함 간의 균형을 이룹니다. 실험을 통해 DYTO는 기존의 정제(fine-tuned) 모델 및 훈련이 필요 없는(training-free) 방법에 비해 뛰어난 성능을 확보하며, 제로샷 비디오 이해에서 새로운 최첨단 성능을 기록했습니다.

- **Performance Highlights**: DYTO는 기존 방법들보다 비디오 이해 능력과 계산 효율성 모두에서 우수한 성능을 보여줍니다. 연구의 결과는 DYTO가 어떻게 중요한 공간-시간 정보를 보존하면서 토큰의 중복성을 크게 줄이는지를 확인해주었습니다. 이로 인해 DYTO는 비디오의 복잡한 콘텐츠에서도 향상된 성능을 발휘하며, 제로샷 적응성의 이점을 극대화해줍니다.



### Baking Gaussian Splatting into Diffusion Denoiser for Fast and Scalable Single-stage Image-to-3D Generation (https://arxiv.org/abs/2411.14384)
Comments:
          A novel one-stage 3DGS-based diffusion generates objects and scenes from a single view in ~6 seconds

- **What's New**: 본 논문에서는 단일 관점으로부터 3D 객체 및 장면 생성을 위한 새로운 단일 단계 3D Diffusion 모델인 DiffusionGS를 제안합니다. 기존의 2D 다중 시점 기반 모델의 문제점을 해결하고 3D 일관성을 보장하여 강력한 생성 성능을 보여줍니다. DiffusionGS는 3D Gaussian point clouds를 각 시간 단계에서 직접 출력하여 다양한 시점의 프롬프트에 유연하게 대응할 수 있도록 설계되었습니다.

- **Technical Details**: DiffusionGS는 모든 시간 단계에서 다중 뷰 픽셀 맞춤 Gaussian primitive를 예측함으로써 생성된 콘텐츠의 3D 일관성을 보장합니다. 기존의 카메라 포즈 조건화 방식인 Plücker coordinates의 한계를 극복하기 위해, RPPC(Reference-Point Plücker Coordinates)라는 새로운 카메라 조건화 방법을 도입하여 깊이와 3D 기하학을 더 잘 인식할 수 있도록 합니다. 논문에서는 다음과 같은 장면-객체 혼합 훈련 전략을 통해 보다 일반화된 사전 학습 능력을 향상시키는 방법을 설명합니다.

- **Performance Highlights**: DiffusionGS는 PSNR에서 2.20 dB, FID에서 23.25 감소하여 기존의 최첨단(SOTA) 방법들에 비해 더 나은 생성 품질을 보여줍니다. 또한, A100 GPU에서 약 6초의 빠른 속도로 작업을 수행할 수 있어 실용성이 높습니다. 사용자 연구와 텍스트-투-3D 애플리케이션을 통한 평가에서도 이 모델의 실용적인 가치가 드러났습니다.



### DINO-X: A Unified Vision Model for Open-World Object Detection and Understanding (https://arxiv.org/abs/2411.14347)
Comments:
          Technical Report

- **What's New**: 이번 논문에서는 DINO-X라는 새로운 객체 중심 비전 모델을 소개합니다. DINO-X는 현재까지 최고의 오픈 월드(object-centric) 객체 탐지 성능을 보유하고 있으며, 이는 IDEA Research에서 개발되었습니다. 이 모델은 Grounding DINO 1.5와 동일한 Transformer 기반 인코더-디코더 아키텍처를 사용하여 객체 중심의 표현을 추구합니다.

- **Technical Details**: DINO-X는 텍스트 프롬프트(text prompt), 비주얼 프롬프트(visual prompt), 맞춤형 프롬프트(customized prompt) 등 다양한 입력 옵션을 지원하여 장기 분포(long-tailed) 객체 탐지를 용이하게 만듭니다. 또한, 프롬프트 없이 오픈 월드 탐지를 가능하게 하는 범용 객체 프롬프트(universal object prompt)를 개발하여 사용자가 프롬프트를 제공할 필요 없이 이미지 내의 모든 객체를 탐지할 수 있습니다. 이를 위해 1억 개 이상의 고품질 샘플로 구성된 대규모 데이터셋인 Grounding-100M을 구축하여 모델의 오픈 어휘(open-vocabulary) 탐지 성능을 향상시켰습니다.

- **Performance Highlights**: DINO-X Pro 모델은 COCO, LVIS-minival, LVIS-val 제로 샷(zero-shot) 객체 탐지 벤치마크에서 각각 56.0 AP, 59.8 AP, 그리고 52.4 AP의 성능을 기록하였습니다. 특히, LVIS-minival과 LVIS-val 벤치마크의 희귀 클래스(rare classes)에서는 각각 63.3 AP와 56.5 AP를 기록했으며, 이는 이전의 SOTA(상태-of-the-art) 성능보다 5.8 AP 향상된 결과입니다. 이러한 성과는 DINO-X가 장기 분포 객체를 인식하는 능력이 크게 향상되었음을 보여줍니다.



### StereoCrafter-Zero: Zero-Shot Stereo Video Generation with Noisy Restar (https://arxiv.org/abs/2411.14295)
- **What's New**: 새롭게 소개된 	extit{StereoCrafter-Zero}는 쌍(pair) 훈련 데이터 없이 비디오 확산(prior) 정보를 활용하여 제로샷(Zero-shot) 스테레오 비디오 생성을 가능하게 합니다. 이 방법은 스테레오 인지(latents)를 초기화하는 노이즈 재시작(noisy restart) 전략과 점진적으로 잠재 공간(latent space)을 조화롭게 만드는 반복 개선(iterative refinement) 과정을 포함하고 있습니다. 이는 시간 왜곡(temporal flickering) 및 뷰(view) 불일치 문제를 해결하는 데 기여합니다.

- **Technical Details**: 이 프레임워크는 비디오 크래프팅(video crafting) 과정에서 깊이 일관성(depth consistency)과 시간적 부드러움(temporal smoothness)을 향상시키는 방법론을 사용합니다. 평가 과정에서는 정량적 메트릭과 사용자 연구(user studies)를 통해 스테레오 비디오의 품질을 분석합니다. 	extit{StereoCrafter-Zero}는 다양한 디퓨전 모델(diffusion models)과 함께 유연하게 적용될 수 있는 강력한 구조로 설계되었습니다.

- **Performance Highlights**: 종합적인 평가 결과, 	extit{StereoCrafter-Zero}는 깊이 추정이 불완전하더라도 높은 품질의 스테레오 비디오를 생성합니다. 새로운 기준을 설정함으로써 제로샷 스테레오 비디오 생성 분야에서 혁신적인 발전을 가져왔으며, 더 몰입감 있는 비주얼 경험을 가능케 합니다. 이 연구의 코드는 제공된 링크에서 확인할 수 있습니다.



### EasyHOI: Unleashing the Power of Large Models for Reconstructing Hand-Object Interactions in the Wild (https://arxiv.org/abs/2411.14280)
Comments:
          Project page: this https URL

- **What's New**: 이 논문은 단일 이미지에서 손-물체 상호작용(Hand-Object Interaction, HOI)을 재구성하는 혁신적인 방법을 제안합니다. 기존의 비디오나 다중 뷰 이미지, 3D 템플릿을 기반으로 한 기법들과는 달리, 단일 뷰 이미지의 재구성이 지니는 고유의 모호성과 가림 현상의 문제를 해결하고자 합니다. 특히, 이 연구는 대규모 모델이 보유한 시각적 및 기하학적 선험적 지식을 활용하여 손과 물체의 형태를 추정하는 새로운 파이프라인을 설계하였습니다.

- **Technical Details**: 단일 입력 이미지를 토대로 먼저 손의 포즈와 물체의 형상을 추정하는 기법을 사용하며, 이어서 선험적인 최적화 프레임워크를 적용합니다. 이는 3D 물리적 제약과 2D 입력 이미지 내용을 준수하여 손 포즈를 최적화하는 과정을 포함합니다. 본 연구는 세 단계로 구성된 최적화 프레임워크를 제안하는데, 각 단계는 카메라 시스템 설정, HOI 접촉 정렬, 손 매개변수 세분화로 이루어져 있습니다.

- **Performance Highlights**: 제안된 방법은 여러 데이터셋에서 실험을 수행한 결과, 기존의 기준 모델들과 비교하여 일관되게 우수한 성능을 보였습니다. 특히 다양한 손-물체 상호작용을 충실하게 재구성할 수 있는 능력을 입증하였으며, 단일 이미지 입력만으로도 물리적으로 현실적인 HOI 결과를 생성할 수 있음을 보여주었습니다.



### Looking Beyond Text: Reducing Language bias in Large Vision-Language Models via Multimodal Dual-Attention and Soft-Image Guidanc (https://arxiv.org/abs/2411.14279)
Comments:
          19 pages, 12 figures

- **What's New**: 본 논문에서는 대형 비전-언어 모델(LVLMs)의 언어 편향 문제를 해결하기 위한 새로운 시스템 프레임워크인 LACING을 제안합니다. LACING은 다중 모드 이중 주의 메커니즘(multi-modal dual-attention mechanism)과 소프트 이미지 가이드(soft-image Guidance)를 활용하여 LVLMs의 비전 이해력을 향상시킵니다. 이는 LVLMs의 환각(hallucination) 현상을 줄이고, 텍스트와 이미지의 통합을 강화하는 데 중점을 둡니다.

- **Technical Details**: LACING의 MDA는 비주얼 입력의 통합을 강화하는 이중 주의 메커니즘을 도입합니다. IFG는 학습과 추론 과정에서 시각적 입력을 대신하는 학습 가능한 소프트 비주얼 프롬프트(soft visual prompt)를 도입하여 LVLMs가 텍스트 입력의 우선순위를 높이도록 유도합니다. 또한, IFG는 인접한 텍스트 입력에 대한 모델의 과도한 의존성을 완화하기 위한 새로운 디코딩 전략을 제안합니다.

- **Performance Highlights**: 실험 결과, LACING은 LVLMs의 언어 편향을 효과적으로 제거하여 비주얼 이해력을 크게 향상시키고 환각 현상을 줄이는 데 성공합니다. 이는 추가적인 훈련 자원이나 데이터 없이도 이루어질 수 있어, 연구자들에게 실질적인 이점을 제공합니다. 저자들은 코드와 모델을 제공하여 관련 연구자들이 활용할 수 있도록 하고 있습니다.



### FocusLLaVA: A Coarse-to-Fine Approach for Efficient and Effective Visual Token Compression (https://arxiv.org/abs/2411.14228)
- **What's New**: 최근 다중모달 대형 언어 모델(Multi-modal Large Language Models, MLLMs)의 연구가 활발하게 진행되고 있습니다. 기존의 모델들은 입력 이미지의 해상도가 낮아 정보 손실이 있었으나, 고해상도 이미지 입력 지원이 중요하다는 인식이 확산되었습니다. 본 논문에서는 Visual Token Compression 기법을 통해 효율성을 개선하고 성능을 동시에 향상시키는 FocusLLaVA 모델을 제안합니다.

- **Technical Details**: FocusLLaVA는 비주얼 정보에 기반한 Coarse-to-Fine 접근 방식을 사용하여 저정보 밀도 영역을 압축하고, 사용자 요청과 관련된 비주얼 토큰을 선택하는 Text-Guided Sampler를 통합합니다. 비전 기반 샘플러는 지역 수준에서 각 지역을 다중 스케일로 다운샘플링하고 적응적으로 선택합니다. 이러한 두 가지 모듈을 통해, 모델은 비주얼 정보의 중복을 제거하며 성능과 효율성 사이의 개선을 달성합니다.

- **Performance Highlights**: FocusLLaVA는 39%의 비주얼 토큰 사용으로 다양한 벤치마크에서 기존 모델을 능가하는 성능을 보였습니다. 이 연구는 LLaVA-Next 기반에서 효과적으로 설계된 두 가지 샘플러를 통해 성능 및 효율성 모두를 높이고 있습니다. 실험 결과는 FocusLLaVA가 MLLM의 state-of-the-art 성능을 유지하면서도 처리 속도 최적화에 기여함을 보여줍니다.



### Towards Context-Rich Automated Biodiversity Assessments: Deriving AI-Powered Insights from Camera Trap Data (https://arxiv.org/abs/2411.14219)
Comments:
          32 Pages, 22 images

- **What's New**: 이 논문에서는 카메라 트랩(camera traps) 데이터를 활용하여 생태 보고(ecological reporting)를 개선할 수 있는 새로운 통합 접근법을 제안합니다. 기존의 자동 이미지 분석 방법이 효율성은 일정 부분 있으나, 생태 보존(outcomes)과 관련하여 필요한 맥락(contextual richness)을 제공하지 못하는 한계를 뛰어넘기 위해 깊은 학습 기반(deep learning-based) 비전(vision) 및 언어 모델(language models)을 결합합니다.

- **Technical Details**: 제안된 시스템은 두 단계로 구성됩니다. 첫째, YOLOv10-X(You Only Look Once) 모델을 사용하여 이미지 내에서 포획된 종(mammals and birds)을 지역화(localize)하고 분류(classify)합니다. 둘째, Phi-3.5-vision-instruct 모델이 YOLOv10-X에서 생성된 바운딩 박스(bounding box) 레이블을 읽어 해당 종을 식별하고, 분류가 어려운 객체들에 대한 한계를 극복합니다.

- **Performance Highlights**: 이 모델은 식생(vegetation) 유형과 시간대 등 넓은 변수를 감지하여 YOLO의 종 탐지 결과에 생태적 및 환경적 맥락을 제공합니다. 결합된 정보는 자연어 처리(natural language processing) 시스템을 통해 복잡한 질문에 대한 답변을 생성하며, 외부 정보로 응답을 풍부하게 하여 자동으로 구조화된 보고서를 생성합니다. 이로 인해, 생물 다양성(biodiversity) 관련 이해관계자들은 종의 개체 수, 분포, 행동 및 서식지 선택에 대한 심층적인 통찰을 얻을 수 있습니다.



### Generative Outpainting To Enhance the Memorability of Short-Form Videos (https://arxiv.org/abs/2411.14213)
- **What's New**: 본 논문에서는 짧은 형식의 비디오에서 영상의 크기를 확장하는 generative outpainting 기법을 활용하여 기억력을 향상시키는 방법을 제안합니다. 비디오의 메모리 능력은 관람자가 그 내용에 감정적이거나 개인적인 연결이 없을 때 기억될 가능성을 나타냅니다. 본 연구는 기존 이미지의 기억력 연구와는 달리 비디오 메모리 능력에 관한 조작적 연구의 부족함을 지적하고, 머신러닝과 딥러닝의 발전을 비교 분석하며 비디오 기억력에 미치는 영향을 탐구합니다.

- **Technical Details**: 논문에서는 비디오 메모리 능력을 조절하기 위해 두 가지 다른 generative 모델을 이용하여 outpainting 기법을 적용했습니다. 이를 위해 autoencoders와 recurrent neural networks (RNNs)를 사용하여 동적이고 복잡한 비디오 내용을 단일 프레임으로 재구성합니다. Diffusion Models에 중점을 두며, 이 모델들은 비디오의 맥락을 풍부하게 만들어 기억력을 높이는 데 기여합니다.

- **Performance Highlights**: 실험 결과, outpainting 기법과 이미지 saliency를 기반으로 한 비디오 메모리 능력 점수의 관계를 정량적으로 평가하였으며, 가장 성능이 우수한 모델을 확인했습니다. 또한, Memento10k 및 VideoMem 데이터를 활용하여 비디오의 시청 지연에 따른 메모리 능력의 감소를 모델링하고, 향상된 비디오 메모리 능력을 위한 promising한 방안을 제시합니다.



### Novel View Extrapolation with Video Diffusion Priors (https://arxiv.org/abs/2411.14208)
- **What's New**: 이번 논문에서는 새로운 뷰 합성 분야에서 ViewExtrapolator라는 새로운 접근 방식을 소개합니다. 이 방법은 Stable Video Diffusion (SVD)의 생성적 선행 지식을 활용하여 현실적인 새로운 뷰 외삽을 실현합니다. 기존의 방안들이 주로 새로운 뷰 보간에서 뛰어난 성과를 내는 반면, 우리는 훈련된 뷰 범위를 넘어서는 외삽의 필요성을 강조합니다.

- **Technical Details**: ViewExtrapolator는 SVD의 비음성 과정을 재설계하여 복원된 레이디언스 필드에서 생성된 비디오에서 나타나는 아티팩트를 정제합니다. 이 논문에서는 ODE 파생 변수를 수정하여 아티팩트가 발생하기 쉬운 비디오 프레임에서 원래 장면 내용을 보존하도록 SVD를 유도하며, 가이드 애닐링과 리샘플링 애닐링을 통해 디노이징 과정에서 아티팩트의 영향을 줄이는 방식을 설계했습니다.

- **Performance Highlights**: 다양한 3D 렌더링 방법들에서 실시한 광범위한 실험을 통해 ViewExtrapolator가 새로운 뷰 외삽에서 우수한 성능을 보이는 것이 입증되었습니다. 이 방법은 단일 뷰 또는 단안 비디오에서 파생된 점 구름을 사용하여 3D 렌더링을 직접 적용할 수 있는 일반성을 가지며, SVD 모델의 미세 조정이 필요하지 않아 데이터 효율성과 계산 효율성을 보장합니다.



### Is this Generated Person Existed in Real-world? Fine-grained Detecting and Calibrating Abnormal Human-body (https://arxiv.org/abs/2411.14205)
Comments:
          16 pages, 14 figures

- **What's New**: 이 논문에서는 Fine-grained Human-body Abnormality Detection (FHAD)라는 새로운 작업을 제안하고, 이를 평가하기 위한 두 개의 고품질 데이터셋을 구축했습니다. 기존의 VLMs는 인체 사진에서 비정상적인 부분을 감지하는 데 있어 낮은 성능을 보였지만, HumanCalibrator라는 정밀 프레임워크를 제안하여 이러한 비정상성을 감지하고 수리하는 과정을 체계화했습니다. 이 연구는 인간의 신체 구조의 비정상성을 탐지하기 위한 새로운 접근 방식을 탐구하고, 해당 분야의 연구 커뮤니티에 기여하는 데 목적이 있습니다.

- **Technical Details**: FHAD의 목표는 주어진 시각적 콘텐츠에서 실제 인간과의 신체 구조 차이를 식별하는 것입니다. 이 방법은 두 가지를 출력해야 하는데, 첫째는 신체 부위 비정상성의 유형을 나타내는 시맨틱 플래그이고, 둘째는 비정상성이 있는 위치를 바운딩 박스 형태로 출력해야 합니다. HumanCalibrator는 이러한 비정상성을 정확히 식별하고 다른 시각적 내용을 보존하며 비정상적인 지역을 수리하는 정밀한 프레임워크입니다.

- **Performance Highlights**: 실험 결과, HumanCalibrator는 비정상 탐지에서 높은 정확도를 기록하며 시각적 비교에서 개선된 결과를 보였습니다. 이 모델은 사람의 신체 구조 내 비정상성을 정밀하게 식별하고 이를 효과적으로 수정하여 기존의 시각적 내용을 보존하는 데 성공했습니다. 이는 시각 콘텐츠 생성 모델에 있어 비정상성 탐지와 수정에서 무척 중요한 진전을 의미합니다.



### Regional Attention for Shadow Remova (https://arxiv.org/abs/2411.14201)
- **What's New**: 이번 연구에서는 기존의 그림자 제거 방법들이 겪고 있는 모델 크기와 계산 복잡도의 문제를 해결하고자 경량화된 그림자 제거 프레임워크를 제안합니다. 이를 통해 각 그림자 영역이 주변의 비그림자 영역과 보다 합리적으로 상호작용할 수 있도록 하는 새로운 지역적 주의 메커니즘을 개발하였습니다. 연구진은 이러한 접근 방식이 최신 기술에 비해 더 높은 정확도와 효율성을 제공한다는 것을 입증하기 위해 다양한 실험을 진행하였습니다.

- **Technical Details**: 제안된 방법은 지역적 주의 메커니즘(Regional Attention Mechanism)을 활용하여 그림자 영역의 복구에 중요한 정보를 비그림자 영역에서 효과적으로 추출합니다. 우리는 RASM(Regional Attention Shadow Removal Model)이라는 네트워크를 구축하였으며, 이 네트워크는 그림자와 비그림자 영역 간의 상호작용을 최적화하여 정확도와 계산 효율성을 동시에 고려합니다. 실험 결과, RASM은 기존의 복잡한 네트워크 구조 없이도 놀라운 성능을 발휘했습니다.

- **Performance Highlights**: 우리는 널리 사용되는 그림자 제거 데이터셋인 ISTD+와 SRD를 기반으로 한 실험을 통해 RASM이 뛰어난 성능을 발휘함을 확인했습니다. 기존의 최신 기술들과 비교할 때, 제안된 방법은 효과성과 효율성 면에서 우수한 결과를 보여주었으며, 이는 실제 애플리케이션에서도 매력적인 선택이 될 수 있습니다. 이 연구는 그림자 제거 분야에서 새로운 최첨단 성능을 달성한 것을 의미합니다.



### ComfyGI: Automatic Improvement of Image Generation Workflows (https://arxiv.org/abs/2411.14193)
- **What's New**: ComfyGI는 자동 이미지 생성의 효율성을 향상시키기 위해 설계된 새로운 접근 방식입니다. 이 방식은 인체 개입 없이도 이미지 생성 워크플로우를 자동으로 개선할 수 있는 기능을 제공합니다. ComfyGI는 이미지의 설명 및 미적 요소에 대해 더 높은 품질을 달성하도록 설계되었습니다.

- **Technical Details**: ComfyGI는 초기 설정으로 이미지를 생성하고, ImageReward 모델을 활용하여 이미지를 평가합니다. 이 접근 방식은 JSON 포맷으로 저장된 워크플로우에서 작은 변화를 적용하여 이미지 생성의 최적화를 이루어냅니다. 각 세대에서 가장 높은 ImageReward 점수를 얻는 변화를 선택하여 이전 설정에 적용하며, 이러한 과정을 통해 최적화된 이미지를 도출합니다.

- **Performance Highlights**: ComfyGI를 활용한 이미지 생성은 초기 설정 대비 약 50% 향상된 ImageReward 점수를 기록했습니다. 또한, 100명의 인원 평가를 통해 약 90%의 경우에서 참가자들이 ComfyGI로 개선된 이미지를 선호하는 것으로 나타났습니다. 이러한 결과는 ComfyGI의 성능이 기존의 이미지 생성 방식보다 월등히 뛰어난 것을 의미합니다.



### CompetitorFormer: Competitor Transformer for 3D Instance Segmentation (https://arxiv.org/abs/2411.14179)
- **What's New**: 본 연구에서는 3D instance segmentation 분야에서 transformer 기반 방법들이 inter-query competition이라는 문제로 어려움을 겪고 있음을 지적합니다. 이를 해결하기 위해 CompetitorFormer라는 새로운 설계를 제안하며, query competition layer, relative relationship encoding, rank cross attention의 세 가지 새로운 경쟁 지향적 설계를 도입합니다. 실험 결과, 이러한 설계가 통합될 경우 여러 데이터셋에서 3D instance segmentation의 성능을 상당히 개선할 수 있음을 증명하였습니다.

- **Technical Details**: CompetitorFormer는 query들 간의 공간적 및 경쟁적 관계를 포착하기 위해 세 가지 디자인을 적용합니다. 첫째로, query competition layer는 각 디코더 레이어 전후로 query 간의 관계를 파악하여 이들을 정형적으로 결합합니다. 둘째로, 상대적 관계 인코딩은 self-attention의 key features를 사용하여 query 간의 관계를 양적화하여 가중치를 조정합니다. 마지막으로, rank cross attention 메커니즘은 각 feature와 모든 query 간의 dot product 유사성을 정규화하여 query 간의 차별성을 증대시킵니다.

- **Performance Highlights**: 경험적으로 CompetitorFormer는 여러 최신 기반 모델에 비해 성능 향상을 보여주었습니다. 특히, ScanNetv2, ScanNet200, S3DIS, STPLS3D 데이터셋에서의 개선된 결과가 입증되었습니다. 이 연구는 inter-query competition을 줄이는 것이 3D instance segmentation의 정확성과 수렴 효율성을 높일 수 있는 열쇠임을 설득력 있게 입증하였습니다.



### Spatiotemporal Decoupling for Efficient Vision-Based Occupancy Forecasting (https://arxiv.org/abs/2411.14169)
- **What's New**: 이 논문에서는 자율주행 차량의 주변 환경에서 초래하는 점유 상태를 예측하는 새로운 방법을 제안합니다. 기존의 3D 점유 예측 방법들이 복잡한 모델 구조를 가지고 있으나 시간에 따른 점유 상태 변화를 간과하는 문제를 해결하기 위해, 시공간 분리(spatiotemporal decoupling) 방법론을 도입했습니다. 이 방법은 효율적인 3D 점유 예측을 위해 BEV(Bird's-Eye View) 형식의 새로운 공간 표현을 사용합니다.

- **Technical Details**: 논문에서 제안한 EfficientOCF 네트워크는 2D BEV 구조를 통해 3D 점유 예측을 수행합니다. 이 네트워크는 기존의 메모리 집약적 3D 점유 표현을 지양하고 lightweight한 2D 인코더-디코더 구조를 사용하여 예측 속도를 개선했습니다. 또한, 각 타임스텝에서 가변 물체의 높이 값을 예측하고, 흐름(flow) 벡터를 통해 현재 관측과 미래 예측 간의 연관성을 강화하여 정확성을 높이고 있습니다.

- **Performance Highlights**: Experiments demonstrate that the EfficientOCF outperforms existing methods in both accuracy and efficiency, achieving state-of-the-art performance with a fast inference time of 82.33ms on a single GPU. 새로운 평가 지표인 C-IoU를 도입하여 복잡한 데이터셋에서도 견고한 3D 점유 예측 성능을 평가할 수 있습니다. 실험 결과, 제안된 방법은 고품질의 3D 점유 예측 결과를 생성하며 실시간 성능을 보장합니다.



### FoPru: Focal Pruning for Efficient Large Vision-Language Models (https://arxiv.org/abs/2411.14164)
Comments:
          11 pages, 7 figures

- **What's New**: 이 논문에서는 Focal Pruning (FoPru)이라는 새로운 방법론을 제안합니다. 이 방법은 시각 인코더로부터 유도된 주의기반의 토큰 중요성을 바탕으로 시각 토큰을 가지치기(pruning)하는 훈련이 필요 없는 방식으로, 기존 LVLMs에 손쉽게 적용 가능합니다. 또한, 두 가지 가지치기 전략인 Rank Pruning과 Row Pruning을 도입하여 보다 효과적인 토큰 선택을 진행합니다.

- **Technical Details**: FoPru는 세 가지 단계로 구성됩니다: 첫 번째로 토큰 중요성 계산을 위한 Token Significance 단계, 두 번째로 이러한 중요성 점수에 따라 시각 토큰을 가지치기 하는 Token Pruning 단계, 마지막으로 원래 위치에 따라 토큰을 재정렬하는 Token Reordering 단계입니다. 이러한 구조 덕분에 FoPru는 시각 토큰의 수를 대폭 줄이면서도 높은 성능을 유지할 수 있습니다.

- **Performance Highlights**: 실험 결과, FoPru는 다양한 LVLMs와 멀티모달 데이터셋에서 시각 토큰을 유의하게 줄이는 동시에 성능을 향상시키는 것을 확인하였습니다. 특히, 극단적인 토큰 보존 비율 0.2%에서 FoPru는 Ai2D 및 SQA 데이터셋에서 약 60%의 정확도를 유지합니다. 또한, 25%의 시각 토큰만 사용했을 때도 여러 데이터셋에서 1%의 마진 내 정확도를 기록하며, inference 효율성을 비약적으로 향상시킵니다.



### Point Cloud Denoising With Fine-Granularity Dynamic Graph Convolutional Networks (https://arxiv.org/abs/2411.14158)
- **What's New**: 이 논문에서는 3D 포인트 구름(3-D point clouds)에서 노이즈를 제거하기 위해 미세 단계(dynamic graph convolutional networks) 방식인 GD-GCN을 소개합니다. GD-GCN은 마이크로 스텝 시간 그래프 컨볼루션(MST-GConv)을 활용하여 특징 학습을 점진적으로 수행하며, 기존의 불연속적인 정수 단계 그래프 컨볼루션보다 더 유연하고 세밀한 학습을 가능하게 합니다. 이 방식은 노이즈가 있는 포인트 구름을 기저면에 정밀하게 적합시키는 데 유용합니다.

- **Technical Details**: GD-GCN은 리만 거리(Riemannian distance)를 이용하여 저차원 다양체(manifold)상의 거리를 계산하고, 각 점의 사이의 국소 기하 구조를 이해하며 다양한 기하학적 영역 간의 관계를 효과적으로 캡처합니다. 또한, 베른슈타인 다항식 근사를 기반으로 하는 강력한 그래프 주파수 필터를 통합하여 복잡한 스펙트럼 응답을 제공하고, BIBO 안정성을 이론적으로 보장합니다. 이 과정에서 대칭 채널 혼합 행렬(symmetric channel mixing matrices)이 필터의 유연성을 더욱 강화합니다.

- **Performance Highlights**: 실험 결과 GD-GCN의 그래프 주파수 필터링 방식이 전통적인 공간 그래프 컨볼루션 기법보다 현저하게 높은 성능을 보였습니다. 이 논문은 3D 포인트 구름 노이즈 제거를 위한 연속 미세 단계 시간이력 그래프 컨볼루션을 처음으로 도입하며, 기하학적 그래프 구성 및 강력한 그래프 주파수 필터링을 결합하여 최신 방법 대비 비약적인 성능 향상을 달성했음을 입증했습니다. 다양한 통계적 특성을 가진 포인트 구름에서 노이즈를 효과적으로 제거할 수 있는 능력을 보여주었습니다.



### Visual Contexts Clarify Ambiguous Expressions: A Benchmark Datas (https://arxiv.org/abs/2411.14137)
- **What's New**: 이번 연구에서는 VAGUE라는 새로운 멀티모달 벤치마크를 제안합니다. 이 벤치마크는 3.9K 개의 간접적인 인간 발화와 해당하는 시나리오를 짝지어 구성되어 있습니다. 주목할 점은 이 연구가 모델의 복잡한 언어 및 시각적 추론 능력을 심층적으로 평가하고, 모델이 비유적 언어와 은유를 이해하는 데 필요한 접근 방식을 어떻게 발전시킬 수 있는지를 탐구한다는 것입니다.

- **Technical Details**: VAGUE는 비유적 언어와 간접적인 의사소통을 다루는 것을 목표로 하며, 이 과정에서 제공된 시각적 장면에서 맥락적 및 관계적 정보를 활용할 수 있도록 합니다. 연구진은 이 데이터를 바탕으로 주어진 간접 발화에서 숨겨진 의도를 분석할 수 있도록 모델의 기능을 발전시키고자 합니다. 여러 멀티모달 모델에 대한 평가 결과, 현재의 주류 모델들은 여전히 이러한 간접적인 의사소통을 처리하는 데 있어 어려움을 겪고 있음을 보여주었습니다.

- **Performance Highlights**: VAGUE는 여러 모델을 다양한 응답 형식으로 실험하고, 부정확한 선택을 유도하는 요소들을 분석할 수 있는 세심하게 설계된 다중 선택지 기반의 문제를 포함하고 있습니다. 이 벤치마크를 통해 다양한 모델이 간접 발화를 해석하는 데 얼마나 능숙한지를 평가하는 동시에, 향후 연구가 모델의 상호작용 능력을 더욱 발전시키는 데 기여할 수 있을 것으로 기대합니다.



### RestorerID: Towards Tuning-Free Face Restoration with ID Preservation (https://arxiv.org/abs/2411.14125)
Comments:
          10 pages, 10 figures

- **What's New**: Blind face restoration 기술은 높은 품질을 가진 실감 나는 이미지를 생성하는데 큰 발전을 이루었으나, ID 정보 보존에는 여전히 어려움이 있다. 본 논문에서는 이를 해결하기 위해 RestorerID라는 조정(튜닝) 없는 방법을 제안하였다. 이 방법은 단일 참조 이미지를 사용하여 다양한 손상 수준의 저품질 이미지를 복원하며 ID 정보를 보존하는 특성을 포함하고 있다.

- **Technical Details**: RestorerID는 차별화된 ID 주입을 통해 저해상도(레지데일) 이미지 구조 정보와 ID 정보를 결합하는 통합 프레임워크를 제안한다. 이 과정에서 FIR-Adapter를 사용하여 저해상도 이미지와 참조 이미지 간의 내용 불일치를 효과적으로 해결한다. 또한, Adaptive ID-Scale Adjusting 전략을 통해 손상 수준에 따라 ID 주입 정도를 동적으로 조절하여 최적의 복원된 이미지를 생성할 수 있다.

- **Performance Highlights**: 실험 결과, RestorerID는 Celeb-Ref 데이터셋 및 실제 사례에서 ID 정보를 보존하며 높은 품질의 얼굴 복원을 수행하는 것으로 나타났다. 기존의 테스트 조정 접근법 및 다른 참조 기반 방법에 비해 우수한 성능을 보여주며, 다양한 손상 수준에서도 일관된 결과를 달성했다. 이로써 RestorerID는 고충실도 복원을 위한 효과적인 대안으로 자리 잡았다.



### Point Cloud Resampling with Learnable Heat Diffusion (https://arxiv.org/abs/2411.14120)
- **What's New**: 이 논문에서는 3D 포인트 클라우드의 리샘플링을 위한 새로운 학습 가능한 열 확산(heat diffusion) 프레임워크를 제안합니다. 기존의 확산 모델은 고정된 조건부 사전(conditional prior)을 사용했으나, 제안된 방법은 적응형 조건부 사전을 생성하여 기하학적 특징을 효과적으로 보존합니다. 또한, 이 방법은 기존의 고정적인 기법들에 비해 포인트 클라우드 구조를 보다 효과적으로 복원할 수 있습니다.

- **Technical Details**: 제안된 방법은 열 커널(heat kernel)의 적응형 확산 일정을 학습하여 전방 프로세스의 주변 분포를 매개변수화합니다. 이를 통해 저품질 포인트 클라우드에서 고품질 버전으로의 조건부 분포를 추정합니다. 또한, 정제된 변분 하한(refined variational lower bound, VLB)을 설계하여 기하학적 특징 보존을 위한 최적화가 이루어집니다.

- **Performance Highlights**: 실험 결과, 이 방법은 기존의 포인트 클라우드 리샘플링 기술들에 비해 우수한 성능을 보였으며, 특히 저밀도 및 노이즈가 있는 포인트 클라우드 처리에서 강력한 효과를 드러냈습니다. 또한, 다양한 재건 작업에서 최첨단 성능을 달성하여, 3D 비전 응용 프로그램에서의 활용 가능성을 보여줍니다.



### Uncertainty-Aware Regression for Socio-Economic Estimation via Multi-View Remote Sensing (https://arxiv.org/abs/2411.14119)
Comments:
          11 pages, 4 figures

- **What's New**: 이번 논문에서는 전통적으로 사용되던 3-band RGB 이미지를 넘어서는 멀티스펙트럼 데이터 활용을 통해 사회경제 지표 예측을 위한 새로운 방법론을 제안합니다. 이는 원거리 감지 이미지를 처리하는 제너릭 비전 모델을 활용하여, 다양한 관점에서의 3밴드 조합을 형성함으로써 이루어집니다. 또한, 불확실성 추정(uncertainty estimation)을 위한 이론적 토대를 마련하여 향후 데이터 수집의 경과를 보다 잘 이해할 수 있도록 돕습니다.

- **Technical Details**: 제안된 방법론은 전이 학습(transfer learning)을 활용하여 여러 스펙트럼 밴드를 이용해 시각적으로 다양한 조합(view)들을 생성합니다. 이 조합은 각 지역의 식생, 지질학, 수역 등의 특징을 강조하며, 돈가지를 감소시키는 효과를 가져옵니다. 이를 통해 우리가 구축한 기계 학습 모델은 전통적인 CNN 기반 접근 방법에 비해 보다 강력한 특징 추출 기능을 제공하게 됩니다.

- **Performance Highlights**: 실험 결과 제안된 방법론은 기존의 RGB 또는 비구조적 밴드를 사용하는 다중 스펙트럼 모델보다 우수한 성능을 보였습니다. 이는 예측의 불확실성을 파악하는 데에도 도움을 주며, 향후 데이터 수집에 대한 명확한 가이드를 제공할 수 있습니다. 이러한 방식은 정책 입안자와 데이터 수집 담당자들에게 보다 효율적인 조사 샘플링 전략을 디자인하는 데 기여할 것입니다.



### WARLearn: Weather-Adaptive Representation Learning (https://arxiv.org/abs/2411.14095)
Comments:
          Accepted for publication in IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2025

- **What's New**: WARLearn은 극단적인 날씨 조건에서 적응형 표현 학습(adaptive representation learning)을 위한 새로운 프레임워크를 소개합니다. 이 프레임워크는 Barlow Twins에서 사용되는 불변 원리를 활용하여, 기존의 깨끗한 날씨 데이터로 훈련된 모델이 악천후에서도 효과적으로 작동하도록 변환합니다. fog 및 저조도 환경에서 최소한의 추가 훈련만으로도 놀라운 성능 향상이 보입니다.

- **Technical Details**: WARLearn은 먼저 깨끗한 날씨 데이터로 모델을 훈련한 다음, 특징 추출 네트워크에서의 표현을 저장합니다. 이후, 깨끗한 데이터에 극단적인 날씨 조건을 적용하여 합성 이미지를 생성하고, 이 합성 이미지를 사용해 특징 추출 부분만 미세 조정(fine-tuning)합니다. 훈련 과정에서 Barlow Twins 손실을 통해 특징 표현을 정렬하며, 최종 추론 모델은 이 조정된 특징 추출기와 깨끗한 날씨 훈련에서 얻은 예측 모듈을 결합하여 얻습니다.

- **Performance Highlights**: 실험 결과, WARLearn은 보이지 않는 실제 세계의 foggy 및 저조도 데이터셋에서 각각 52.6% 및 55.7%의 평균 정밀도(mAP)를 기록했습니다. 또한, WARLearn은 FeatEnHancer, Image Adaptive YOLO, DENet 등 최신 프레임워크들을 큰 폭으로 초과하는 성능을 보였습니다. 이 프레임워크는 다양한 신경망 아키텍처에서도 유연하게 적용 가능함을 입증하며, 성능 평가에서 기존 방법들을 초월하거나 거의 근접한 결과를 얻었습니다.



### Multi LoRA Meets Vision: Merging multiple adapters to create a multi task mod (https://arxiv.org/abs/2411.14064)
- **What's New**: 이 논문에서는 여러 개의 LoRA (Low-Rank Adaptation) 어댑터를 컴퓨터 비전 작업에서 훈련한 후, 이를 결합하여 성능 손실 없이 추론 단계에서 사용할 수 있는지를 조사하였습니다. 일반적으로 사용되는 PEFT (Parameter Efficient Finetuning) 방법을 통해 다양한 작업에 대해 결합된 멀티태스크 모델을 만들 수 있는 가능성을 제시합니다. 이를 통해 추가적인 재훈련 없이도 추론 시간을 단축할 수 있습니다.

- **Technical Details**: 연구에서는 여섯 가지 서로 다른 작업에서 훈련된 어댑터들을 결합하고, 이들을 하나의 모델에서 사용할 때의 성능을 평가했습니다. 특히, 고정된 백본 모델에서 헤드를 미세 조정한 모델과 성능을 비교하여, 어댑터를 간단히 결합하기만 해도 멀티태스크 모델을 생성할 수 있음을 보여줍니다. 실험에서는 최대 세 개의 어댑터를 결합하여 사용하였습니다.

- **Performance Highlights**: 어댑터의 결합이 데이터의 유사성과 작업에 따라 헤드 미세 조정보다 성능이 우수할 수 있음을 관찰하였습니다. 특히, 유사한 데이터셋으로 훈련된 모델보다 서로 다른 데이터셋으로 훈련된 LoRA가 더 나은 성능을 보이는 경향이 발견되었습니다. 이는 멀티태스크 모델의 효율성을 높이고 다양한 작업에서의 응용 가능성을 확장할 수 있음을示합니다.



### MMGenBench: Evaluating the Limits of LMMs from the Text-to-Image Generation Perspectiv (https://arxiv.org/abs/2411.14062)
Comments:
          This project is available at: this https URL

- **What's New**: 이 논문은 Large Multimodal Models (LMMs)의 이미지 생성을 평가하기 위한 직관적인 자동 평가 파이프라인을 제안합니다. 기존의 벤치마크가 주로 이미지 이해에 중점을 두었다면, 본 연구는 LMM의 이미지 생성 능력을 포함하여 보다 포괄적인 평가에 중점을 두고 있습니다. 또한, MMGenBench-Test와 MMGenBench-Domain이라는 새로운 벤치마크를 도입하여 13개의 다양한 이미지 패턴을 평가하고, 생성 이미지 도메인에서 LMM의 성능을 분석합니다.

- **Technical Details**: 제안된 파이프라인은 LMM이 주어진 입력 이미지를 기반으로 이미지 프롬프트(image-prompt)를 생성하도록 요구합니다. 이후, 이 프롬프트를 바탕으로 텍스트-투-이미지(text-to-image) 생성 모델을 이용해 새 이미지를 생성하고, 최종적으로 원본 이미지와 생성된 이미지를 비교하여 성능을 평가합니다. 이 과정은 이미지 이해 및 설명에 대한 LMM의 성능을 평가하기 위해 세 가지 구성 요소로 이루어져 있습니다: 이미지 프롬프트 생성, 새로운 이미지 생성 및 정량적 메트릭 계산.

- **Performance Highlights**: 50개 이상의 인기 있는 LMM을 대상으로 한 평가 결과, 기존 벤치마크에서는 우수한 성과를 보이는 LMM이 기본적인 이미지 이해 및 설명 작업을 적절히 수행하지 못한다는 사실이 드러났습니다. 이 연구는 LMM의 성능 개선 가능성을 강조하며, 생성 이미지 도메인에서 더 나은 모델 최적화를 위한 방향성을 제시합니다. MMGenBench를 통해 LMM의 성능을 다양한 도메인에서 효율적으로 평가할 수 있는 유연하고 확장 가능한 벤치마킹 도구를 제공합니다.



### Stereo Anything: Unifying Stereo Matching with Large-Scale Mixed Data (https://arxiv.org/abs/2411.14053)
Comments:
          Code will be available at \url{this https URL}

- **What's New**: 이번 연구에서는 StereoAnything이라는 새로운 기초 모델을 도입하여 강력한 stereo matching 솔루션을 제안합니다. 우리 모델은 다양한 환경에서 stereo 이미지를 처리할 수 있는 범용성을 목표로 하고 있습니다. 이를 위해 우리는 레이블이 있는 stereo 이미지를 수집하고, 비지도 모노큘러 이미지로부터 합성 stereo 쌍을 생성하여 데이터셋을 확장합니다.

- **Technical Details**: StereoAnything의 훈련 과정은 기존의 많은 공개 레이블이 있는 stereo 데이터셋을 활용하고, 새로운 합성 데이터셋인 StereoCarla를 통해 훈련 데이터의 품질과 다양성을 한층 높였습니다. 또한, 단일 이미지에서 생성된 합성 stereo 데이터를 대량으로 추가하여 다양한 환경에서의 정확하고 견고한 깊이 예측을 가능하게 합니다. 이를 통해 우리는 데이터 부족으로 인한 한계를 극복하고 stereo matching 성능을 향상시킵니다.

- **Performance Highlights**: 모델의 제로샷(zero-shot) 성능을 다섯 개의 대중 데이터셋에서 평가하여, 새로운 미지의 데이터에 대한 일반화 능력을 입증했습니다. 특히, 기존의 모든 네트워크 중에서 가장 강력한 제로샷 능력을 발휘하는 스테레오 모델을 훈련하는 데 성공했습니다. 이러한 성과는 우리 모델이 다양한 상황에서도 일관된 성능을 발휘할 수 있음을 보여줍니다.



### Uterine Ultrasound Image Captioning Using Deep Learning Techniques (https://arxiv.org/abs/2411.14039)
- **What's New**: 이 논문은 자궁 초음파 영상의 캡션 생성을 위한 딥러닝 기반 방법을 제안합니다. 기존의 문제점인 해석의 어려움을 극복하기 위해, 합성곱 신경망(Convolutional Neural Networks)과 양방향 게이트 순환 유닛(Bidirectional Gated Recurrent Unit) 네트워크를 통합한 하이브리드 모델을 구축했습니다. 이는 의료 전문인들이 신속하고 정확하게 진단을 내릴 수 있도록 돕는 것을 목표로 합니다.

- **Technical Details**: 제안된 시스템은 먼저 자궁 초음파 영상의 대규모 데이터세트를 수집하고 이를 전문가의 주석으로 세심하게 주석 처리했습니다. 주요 특징 추출 단계에서는 사전 훈련된 CNN 모델인 Inception V3와 DenseNet201을 활용하여 이미지에서 세부 특징 벡터를 추출하였고, 텍스트 데이터는 수치 표현으로 변환하여 이미지 특징과 매끄럽게 결합하였습니다. 최종적으로 CNN-BiGRU 모델을 통해 자궁 초음파 이미지에 대한 설명적 캡션을 생성하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 CNN-BiGRU 모델은 BLEU 및 ROUGE 점수에서 기존의 방법들에 비해 우수한 성능을 보였습니다. 이 모델은 초음파 이미지를 정확하고 유익하게 설명할 수 있어, 여성 건강 분야에서의 진단 정확성을 향상시킬 잠재력이 있습니다. 이 연구는 자궁 초음파 이미지 해석을 개선하여 궁극적으로 환자 치료에 기여하고자 하는 바탕이 됩니다.



### SEMPose: A Single End-to-end Network for Multi-object Pose Estimation (https://arxiv.org/abs/2411.14002)
- **What's New**: 이 논문에서는 멀티 오브젝트 장면에서 RGB 이미지로부터 여섯 자유도(6-DOF) 포즈 추정을 위해 SEMPose라는 네트워크를 제안합니다. 이 네트워크는 엔드 투 엔드(end-to-end) 방식으로 다중 객체 포즈 추정을 수행할 수 있는 새로운 접근 방식입니다.

- **Technical Details**: SEMPose는 텍스처-모양 유도 피쳐 피라미드 네트워크(texture-shape guided feature pyramid network)를 활용하여 객체 크기 변화 문제를 해결합니다. 또한, 반복 정제 구조(iterative refinement head structure)를 채택하여 회전(rotation)과 변환(translation)을 각각 개선하여 정확도를 향상시킵니다.

- **Performance Highlights**: SEMPose는 RGB 이미지만을 입력으로 받아 32 FPS의 속도로 실시간 포즈 추정을 수행할 수 있습니다. LM-O 및 YCB-V 데이터셋에서, SEMPose는 다른 단일 모델 기반 RGB 방법들보다 더 높은 정확도를 기록하였으며, 다중 모델 방법 및 추가 정제를 사용하는 접근 방식과 비교할 때도 경쟁력을 지니고 있습니다.



### Graph Domain Adaptation with Dual-branch Encoder and Two-level Alignment for Whole Slide Image-based Survival Prediction (https://arxiv.org/abs/2411.14001)
Comments:
          12 pages, 6 figures

- **What's New**: 본 논문은 병리학적 전범위 이미지(WSI) 기반 생존 분석 분야에서 도메인 전이 문제를 해결하기 위해 새로운 이중 분기 인코더 및 두 단계 정렬(DETA) 프레임워크를 제안합니다. 기존의 생존 분석 방법론은 일반적으로 서로 다른 병원이나 실험실에서 수집된 WSI 간의 큰 분포 차이를 고려하지 않았습니다. DETA 프레임워크를 통해 이러한 도메인 간의 카테고리와 피처 분포의 정렬을 탐구하며 이를 위한 그래프 도메인 적응(GDA)을 활용합니다.

- **Technical Details**: DETA 프레임워크는 그래프 표현을 통해 WSI의 특징과 카테고리 간의 정렬을 실현합니다. 이중 분기 그래프 인코더를 설계하여 메시지 패싱(branch)과 최단 경로(branch) 기능을 결합하며, 이들 구조는 그래프에서 의미 정보를 추출하는 데 기여합니다. 카테고리 수준에서는 두 영역 간의 분포 수렴을 유도하고, 피처 수준에서는 적대적인 왜곡 전략을 통해 소스 도메인의 피처 개선을 꾀합니다.

- **Performance Highlights**: 제안된 DETA 프레임워크는 4개의 TCGA 데이터셋을 사용하여 광범위한 실험을 수행하였으며, 이로 인해 WSI 기반 생존 분석에서 우수한 성능을 보여주었습니다. 이러한 성과는 도메인 이동 문제를 완화하고 생존 예측 정확성을 높이는 데 기여하였습니다. 이는 의료 영상 분석의 새로운 가능성을 열어주는 연구로 주목받고 있습니다.



### Mirror Target YOLO: An Improved YOLOv8 Method with Indirect Vision for Heritage Buildings Fire Detection (https://arxiv.org/abs/2411.13997)
- **What's New**: MITA-YOLO라는 새로운 화재 감지 방법이 제안되었습니다. 이 모델은 간접 시각(Indirect Vision)을 활용하여 카메라 수를 줄이고 화재 감지 성능을 향상시키는 데 중점을 두고 있습니다. 특히, 미러(mirror) 각도를 조절하여 가시성이 제한된 공간에서도 지표를 정렬하고 타겟 감시 구역에 맞춰 감지를 수행합니다.

- **Technical Details**: 본 연구는 타겟-마스크(Target-Mask) 모듈을 도입하여 이미지 내에서 간접 시각 영역을 자동으로 식별하고 자동 분리하여 비타겟 영역을 필터링합니다. 이를 통해 모델은 화재 위험 구역에 대한 관리자들의 전문성을 물려받아 화재 모니터링의 초점을 높이고 간섭에 저항하는 성능을 갖출 수 있도록 합니다. MITA-YOLO는 800장의 이미지로 구성된 화재 데이터셋을 이용하여 평가되었으며, 성능은 기존의 다른 모델들보다 우수한 것으로 나타났습니다.

- **Performance Highlights**: MITA-YOLO는 다른 최신 모델들과 비교했을 때 mAP50에서 3.7% 향상된 성능을 보였으며, 재현율이 3% 증가하는 결과를 얻었습니다. 이 방법은 간접 시각을 활용하여 화재 탐지에서 최고 성능을 달성하며, 구조물에 미치는 영향을 최소화하는 동시에 감지 정확도를 높이는 데 기여합니다.



### Safety Without Semantic Disruptions: Editing-free Safe Image Generation via Context-preserving Dual Latent Reconstruction (https://arxiv.org/abs/2411.13982)
Comments:
          This research is supported by the NISDRG project #20100007, funded by the Australian Government

- **What's New**: 이 연구는 다중 모달 생성 모델의 안전한 콘텐츠 생성을 보장하는 과제를 다룬다. 기존의 모델 수정 기술은 개념 간의 의미적인 관계를 손상시켜 불필요한 왜곡을 초래할 수 있다. 이러한 문제를 해결하기 위해, 저자들은 안전-컨텍스트 임베딩(safety-context embeddings)과 이중 재구성 과정(dual reconstruction process)을 활용하여 수정 없는 안전한 이미지 생성을 위한 모듈형 솔루션을 제안한다.

- **Technical Details**: 제안된 방법은 레이튼 공간(latent space)에서 조정 가능한 가중치 합(tunable weighted summation)을 통해 전 세계적인 시각적 맥락을 유지하면서 모델의 구조적 무결성을 보존한다. 이 연구는 안전한 이미지 생성을 위한 선진 방법론으로, 모델의 안전성을 조절 가능한 방식으로 제어할 수 있는 기능을 제공한다. 이를 통해 위험한 콘텐츠 생성에 대한 여러 접근 방식을 통합하여 각기 다른 안전 기준을 충족할 수 있게 한다.

- **Performance Highlights**: 이 방법은 안전한 이미지 생성 벤치마크에서 최첨단 성능을 달성하며, 감시를 통한 모델 안전성과 검열(censorship) 간의 균형을 제시한다. 지원하는 두 개의 모듈형 적절하지 않은 입력 감지기를 통해 모든 비도덕적인 입력을 안전한 의미적 영역으로 효율적으로 유도할 수 있다. 이러한 접근 방식의 결과로, 이 연구는 고급 텍스트-이미지 생성 모델에 대한 책임감 있는 안전성을 구현하는 방법을 제안하고 있다.



### On the Fairness, Diversity and Reliability of Text-to-Image Generative Models (https://arxiv.org/abs/2411.13981)
Comments:
          This research is supported by the NISDRG project #20100007, funded by the Australian Government

- **What's New**: 이 논문에서는 텍스트-이미지 모델의 신뢰성과 공정성을 평가하기 위한 새로운 성능 평가 프레임워크를 제안합니다. 제안된 방법은 'semantic' perturbations를 통해 모델의 취약점을 분석하고, 신뢰성이 낮은 입력을 특定하는 데 중점을 둡니다. 이러한 접근은 생성적 다양성과 공정성을 더 깊이 이해할 수 있도록 돕습니다. 또한, 논문의 방법론은 편향이 주입된 모델의 감지와 편향 출처 추적을 위한 기초를 마련합니다.

- **Technical Details**: 제안된 방법론은 임베딩 공간에서 글로벌 및 로컬 차원의 'semantic' perturbations를 적용하여 텍스트-이미지 모델의 신뢰성을 정량화합니다. 이를 통해 맥락화된 프롬프트의 행동과 각 인코딩된 토큰이 생성에 미치는 영향을 분석합니다. 이 과정에서 생성된 이미지의 중요한 변화는 모델의 신뢰성이 낮음을 나타내며, 그 결과 공정성과 다양성 평가의 필요성을 강조합니다.

- **Performance Highlights**: 제안된 방법은 고의적으로 편향된 텍스트-이미지 모델을 효과적으로 감지하는 데 유효하며, 생성적 공정성과 다양성 평가를 통해 편향 트리거를 검색하고 출처를 식별하는 데 기여합니다. 이 연구는 텍스트-이미지 모델의 공정성과 신뢰성을 평가하는 다양한 메트릭스를 제시하며, 이러한 평가가 모델 행동을 더 잘 이해하는 데 어떻게 기여하는지를 설명합니다.



### Transforming Static Images Using Generative Models for Video Salient Object Detection (https://arxiv.org/abs/2411.13975)
- **What's New**: 이 연구에서는 정적 이미지에서 비디오로의 변환 방법이 기존의 간단한 공간 변환 기법에서 벗어나 이미지-비디오 확산(diffusion) 모델을 활용하여 현실감 있는 변환을 생성할 수 있음을 보여줍니다. 이를 통해 각 객체의 독립적인 움직임을 반영하는 의미론적(semantic) 무결성을 유지하며, 신뢰할 수 있는 optical flow를 생성할 수 있는 가능성을 제시합니다. 이러한 접근 방식은 대규모 이미지-흐름(image-flow) 쌍을 생성하여 모델 트레이닝을 크게 개선합니다.

- **Technical Details**: 기존 이미지-비디오 시뮬레이션 방법의 한계를 극복하기 위해, 이 연구는 정적 이미지의 맥락적 관계를 이해하면서 현실적인 변환을 생성할 수 있는 이미지-비디오 확산 모델을 제안합니다. 이러한 모델은 정적 영상을 변환하면서 장면의 의미론적 무결성을 유지할 수 있으며, 이는 특히 비디오 두드러진 객체 탐지와 같은 복잡한 작업에서 유용합니다. 이미지로부터 직접 새 프레임을 생성함으로써 기존의 왜곡 방식에 의존하지 않고도 고화질의 optical flow 맵을 얻을 수 있습니다.

- **Performance Highlights**: 제안된 방법은 여러 공개 VSOD 벤치마크에서 기존 접근 방식들을 능가하여 최첨단(state-of-the-art) 성능을 기록했습니다. 특히, DAVIS 2016 검증 세트, FBMS 테스트 세트, DAVSOD 테스트 세트, ViSal 데이터셋에서 94.5%, 92.6%, 80.3%, 96.2%의 높은 𝒮𝒮 (S measure) 점수를 달성하였습니다. 이러한 결과는 이미지-흐름 쌍의 증강이 VSOD 모델 성능을 크게 향상시킬 수 있음을 보여줍니다.



### Zero-Shot Low-Light Image Enhancement via Joint Frequency Domain Priors Guided Diffusion (https://arxiv.org/abs/2411.13961)
- **What's New**: 본 논문은 제로샷(zero-shot) 저조도 이미지 향상 방법을 제안하여, 기존의 방법들이 저조도 환경에서 겪는 한계점을 극복하려 합니다. 여기서는 웨이브렛(wavelet) 및 푸리에(Fourier) 주파수 영역을 효과적으로 결합하여 보다 풍부한 사전 정보(prior information)를 구성하고, 이를 Diffusion 생성 과정에서 활용합니다. 이 방법은 다양한 복잡한 시나리오에서도 robust한 성능을 보여줍니다.

- **Technical Details**: Diffusion 모델은 마르코프(Markov) 체인 구조를 기반으로 하며, 주요 두 단계인 전방 확산(forward diffusion) 및 역 샘플링(inverse sampling) 과정을 포함합니다. 전방 과정에서는 입력 이미지에 Gaussian noise를 점진적으로 추가하고, 이후 역 과정에서 순차적으로 노이즈를 제거하여 원래 이미지를 복원을 시도합니다. 이 과정에서 웨이브렛 저주파 영역으로의 전환과 푸리에 주파수 영역과의 연속적인 분해가 따라 진행됩니다.

- **Performance Highlights**: 광범위한 실험 결과, 제안된 방법은 다양한 환경에서의 저조도 이미지 개선에 있어 대단히 유효하고 versatile함을 입증했습니다. 이 연구는 저조도 향상 과정에서 빛과 구조 정보의 결합을 통해, 성능의 개선을 이끌어 내는 중요한 기여를 합니다. 따라서, 기존의 기법들과 비교해 더욱 강력하고 일반화 가능한 접근법으로 자리 잡을 가능성이 큽니다.



### Separable Mixture of Low-Rank Adaptation for Continual Visual Instruction Tuning (https://arxiv.org/abs/2411.13949)
- **What's New**: 이 연구는 Multimodal Large Language Models (MLLMs)의 지속적 시각 지시 조정(Continual Visual Instruction Tuning, CVIT)에서 나타나는 이중적인 재기억 소멸(catastrophic forgetting) 문제를 파악하고 이를 해결하기 위한 Separable Mixture of Low-Rank Adaptation (SMoLoRA) 프레임워크를 제안합니다. 전통적인 지속적 학습 패러다임에서는 이 문제를 다루지 않았으나, 본 연구는 시각 이해(visual understanding)와 지시 이행(instruction following) 능력의 동시에 저하되는 특수한 상황을 강조합니다.

- **Technical Details**: SMoLoRA는 두 가지 모듈, 즉 시각 이해 모듈과 지시 이행 모듈을 활용하는 분리 가능한 라우팅(separable routing)을 통해 이중 재기억 소멸 문제를 해결합니다. 이러한 접근방식은 다음의 세 가지 요소로 구성되어 있습니다: 1) 각 작업의 특정 요구에 맞는 LoRA 블록을 동적으로 선택하여 시각 및 지시 수행 능력을 보존 및 개선하며, 2) 두 도메인에서의 독립적 적응을 통해 정보를 생성함으로써 과거 지식을 보존하고, 3) 새로운 과제가 기존 지식에 간섭하지 않도록 만듭니다.

- **Performance Highlights**: 실험 결과, SMoLoRA는 기존 방법들보다 이중 재기억 소멸 문제를 완화하고, 새로운 작업에 대한 일반화 능력을 증대시키며, 다양한 작업에 따른 지시에 대한 강인성을 증명했습니다. 새로운 CVIT 벤치마크를 통해 본 모델의 성능 평가가 수행되었으며, 이는 모델이 과거 지식을 잃지 않으면서 여러 작업을 수행할 수 있는 능력을 강조하는 데 중점을 두었습니다.



### Transforming Engineering Diagrams: A Novel Approach for P&ID Digitization using Transformers (https://arxiv.org/abs/2411.13929)
- **What's New**: 이번 연구에서는 Piping and Instrumentation Diagrams (P&ID)의 디지털화에 관한 새로운 접근 방식을 제안합니다. 기존 방법들이 다이어그램 요소를 개별적으로 분석하는 데 중점을 두었던 점을 보완하여, Relationformer라는 최첨단 딥러닝 아키텍처를 활용하여 P&ID로부터 그래프를 추출합니다. 이 방법은 물체 및 그 관계를 동시에 감지할 수 있어 엔지니어링 다이어그램에서의 그래프 추출에 적합합니다.

- **Technical Details**: 제안된 관계 형성자는 P&ID에서 기호와 연결을 포함하여 그래프 구조를 추출하기 위해 설계되었습니다. 두 가지 접근 방식을 비교 분석하여, 생성된 인공(PID2Graph) 데이터셋을 활용하여 효과성을 평가합니다. 이 데이터셋은 기호, 노드 및 이들의 연결에 대한 라벨을 포함하여 최초로 공개된 P&ID 데이터셋으로, 다양한 평가 지표를 사용하여 성능을 측정합니다.

- **Performance Highlights**: 실제 데이터셋을 사용한 결과, Relationformer는 가장 최근의 모듈식 디지털화 접근 방식을 25% 이상 초과하여 가장자리 검출에서 뛰어난 결과를 기록하였습니다. 또한, 연구 결과는 P&ID 디지털화 방법의 성능을 평가하기 위한 포괄적인 프레임워크를 제공하며, 이러한 결과를 바탕으로 향후 연구의 기회를 열어줍니다. 전체적인 성능 향상은 최신 트랜스포머 아키텍처의 유용성을 다시 한 번 입증합니다.



### Multimodal 3D Reasoning Segmentation with Complex Scenes (https://arxiv.org/abs/2411.13927)
- **What's New**: 최근 멀티모달 학습의 발전은 3D 장면 이해와 관련된 다양한 실세계 작업에서 큰 진전을 이루었습니다. 그러나 기존 연구는 상호작용과 인간 의도를 해석하는 추론 능력이 부족하고, 단일 카테고리 객체에만 집중하여 다중 객체 상황과 공간 관계를 간과함으로써 과도하게 단순화된 텍스트 설명을 초래합니다. 이를 해결하기 위해, 우리는 3D 장면에서 여러 객체에 대한 3D 추론 세분화 작업을 제안하며, 이 작업은 3D 세그멘테이션 마스크와 객체 간의 3D 공간 관계를 담은 상세한 텍스트 설명을 생성할 수 있습니다.

- **Technical Details**: 우리는 ReasonSeg3D를 개발하여 3D 공간 관계와 생성된 질문-답변 쌍 및 3D 세그멘테이션 마스크를 통합한 대규모 고품질 벤치마크를 제시합니다. 또한, 사용자의 질문과 텍스트 출력을 기반으로 다중 객체 3D 추론 세분화를 수행할 수 있는 MORE3D라는 간단하면서도 효과적인 방법을 설계했습니다. MORE3D는 LLM을 통해 객체별 포인트 클라우드 임베딩을 학습하여 여러 객체의 정확한 3D 세그멘테이션 마스크를 예측하고, 객체 간의 3D 공간 관계를 포착하는 상세한 텍스트 설명을 생성합니다.

- **Performance Highlights**: 광범위한 실험 결과, MORE3D는 복잡한 다중 객체 3D 장면에 대한 추론 및 세분화에서 우수한 성능을 보였습니다. 또한, ReasonSeg3D는 다중 객체 3D 추론 세분화를 위한 미래 탐색을 위한 귀중한 플랫폼을 제공합니다. 최종적으로, 이 연구는 다중 객체 3D 추론 세분화 기술의 우수성과 제안된 벤치마크의 타당성을 명확히 입증하며, 데이터셋과 코드는 추후 공개될 예정입니다.



### Quantization without Tears (https://arxiv.org/abs/2411.13918)
- **What's New**: 본 논문에서 제안하는 QwT(Quantization without Tears) 방식은 네트워크 양자화의 속도, 정확성, 단순성 및 일반성을 동시에 달성할 수 있도록 설계되었습니다. 기존 양자화 기법의 복잡성과 민감성을 극복하기 위해, 저렴한 추가 구조를 도입하여 정보 손실을 완화합니다. QwT는 2분 이내에 정확성을 개선할 수 있는 닫힌 형태의 솔루션을 제공합니다.

- **Technical Details**: QwT의 핵심 아이디어는 양자화된 네트워크 구조 Sℤ (S with blackboard Z)가 원본 네트워크 구조 S와 정확히 일치할 필요가 없다는 점입니다. 대신, 일부 추가 모듈 Sc가 네트워크 구조에 추가되어 양자화된 매개변수와 활성화로 인한 정보 손실을 보완합니다. 이 간단한 구조 덕분에 QwT는 복잡한 하이퍼파라미터 조정 없이도 높은 정확도를 달성할 수 있습니다.

- **Performance Highlights**: QwT는 다양한 CNN 및 Transformer(예: ViT, Swin 등) 구조에 성공적으로 적용되었으며, 객체 인식, 탐지 및 분할, 언어 모델 등의 다양한 작업에서 뛰어난 성능을 보입니다. 기존 PTQ 방법보다 더 높은 정확성을 제공하며, 고급 정확성이 요구되는 경우 단 한 번의 에포크 훈련으로 QAT(Quantization-Aware Training) 방법과 유사한 정확도에 도달할 수 있습니다. QwT는 하이퍼파라미터 조정 없이 실행 가능하고, GPU에서의 활용도 고려하여 최소한의 노력으로 양자화를 수행할 수 있습니다.



### Panther: Illuminate the Sight of Multimodal LLMs with Instruction-Guided Visual Prompts (https://arxiv.org/abs/2411.13909)
- **What's New**: 이번 논문에서 소개하는 Panther는 사용자의 지침을 정밀하게 따르고, 물체를 정확하게 탐지하는 고급 다중모달 대형 언어 모델(MLLM)입니다. 이는 기존 MLLM의 한계인 Amblyopia 현상을 극복하도록 설계되어, 시각적 표현을 개선하는 데 중점을 두고 있습니다. Panther는 Panther-VE(시각 인코더), Panther-Bridge, Panther-Decoder로 구성되어 있으며, 이러한 모듈은 사용자 지침을 통합하여 더 나은 시각적 표현을 생성합니다.

- **Technical Details**: Panther-VE 모듈은 사용자 지침 정보를 초기 단계에서 통합하여, 가장 관련성이 높은 시각적 표현을 추출합니다. Panther-Bridge는 강력한 필터링 기능을 갖추고 있어 중복된 시각 정보를 줄이고, 결과적으로 훈련 비용을 절감합니다. Panther-Decoder는 다양한 LLM 아키텍처와 호환될 수 있는 유연성을 제공하며, 서로 다른 디코더 전용 아키텍처와도 통합이 가능합니다.

- **Performance Highlights**: 실험 결과, Panther는 비주얼 질문응답, 지침 따르기 및 시각 중심 작업에서 효과적으로 그 성능을 입증했습니다. 특히, Panther는 기존의 시각 중심 벤치마크에서 상대적으로 두드러진 성과를 나타내며, Amblyopia 문제를 해결하는데 기여하고 있습니다. 따라서 Panther는 시각적 인식을 향상시키고, 사용자 목표에 대한 명확한 포인트 지정을 가능하게 만들어 기존 모델 대비 월등한 성능을 보이고 있습니다.



### Dressing the Imagination: A Dataset for AI-Powered Translation of Text into Fashion Outfits and A Novel KAN Adapter for Enhanced Feature Adaptation (https://arxiv.org/abs/2411.13901)
Comments:
          Under review at a conference

- **What's New**: 이 논문에서는 패션 산업의 언어와 스타일링 요소를 담은 전문화된 데이터셋인 FLORA(Fashion Language Outfit Representation for Apparel Generation)를 소개합니다. FLORA는 4,330개의 패션 의상 쌍과 이에 대한 텍스트 설명을 포함하며, 이는 패션 디자이너들이 일반적으로 사용하는 전문 용어와 은어를 활용하여 상세한 통찰을 제공합니다. 또한 논문에서는 기존의 LoRA(adaptive low-rank representation) 어댑터를 대체할 수 있는 KAN(Kolmogorov-Arnold Networks) 어댑터를 소개합니다.

- **Technical Details**: FLORA 데이터셋은 패션 스케치의 텍스트 설명에 대한 정확하고 스타일리시한 이미지를 생성할 수 있도록 generative 모델을 세밀하게 조정하는 데 효과적입니다. KAN 어댑터는 기존 MLP(multi-layer perceptron) 기반 LoRA 어댑터의 대체물로서 학습 가능한 스플라인 활성화 기능을 통해 복잡한 비선형 관계를 모델링하는 데 뛰어난 성능을 발휘합니다. 이들은 높은 충실도와 빠른 수렴 속도를 자랑하며, 의미적 정렬(semmantic alignment)에도 유리합니다.

- **Performance Highlights**: FLORA 데이터셋을 사용한 광범위한 실험과 ablation 연구는 KAN 어댑터가 LoRA 어댑터에 비해 우수한 성능을 보임을 확인하였습니다. 이로 인해 패션 디자인을 위한 고급 AI 모델을 생성할 수 있는 가능성이 열리며, 패션 디자이너들과 최종 사용자가 그들의 아이디어를 구체화하는 데 도움을 줄 것입니다. 또한, FLORA와 그 구현 코드를 오픈소스하여 추가 연구 및 협업을 촉진할 예정입니다.



### CLFace: A Scalable and Resource-Efficient Continual Learning Framework for Lifelong Face Recognition (https://arxiv.org/abs/2411.13886)
Comments:
          Accepted for publication in the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 2025)

- **What's New**: 본 논문에서는 지속적인 학습 프레임워크인 CLFace를 소개하여, 얼굴 인식(Face Recognition, FR) 시스템이 새로운 얼굴 정체성을 학습할 때 발생하는 문제를 해결합니다. CLFace는 기존 모델의 분류 층을 제거하여 리소스를 효율적으로 사용하며, 연속적인 학습을 가능하게 합니다. 이 접근법은 새롭게 등장하는 정체성 그룹을 처리하는 데 필요한 계산 비용과 메모리 요구 사항을 줄입니다.

- **Technical Details**: CLFace는 학습된 지식을 유지하고 점진적으로 확장하는 특징을 가지고 있으며, 학습과정에서 레이블 없는 감독(label-free supervision) 방식을 지원합니다. 본 프레임워크는 특징 수준에서의 증류(distillation)와 기하학적 보존(distillation)을 활용하여, 교사 모델과 학생 모델 간의 특징 맵 간의 흐름(drift)을 최소화합니다. 이는 학생 모델이 교사 모델의 특징 임베딩의 방향성을 유지하며, 새로운 정체성 간의 유사성을 매칭하여 특징 표현의 구별력을 향상시키는 역할을 수행합니다.

- **Performance Highlights**: CLFace는 여러 얼굴 인식 벤치마크 데이터셋에서 테스트되어, 이전의 최첨단 방법들보다 더 뛰어난 성능을 보였습니다. 특히, CLFace는 보지 못한 정체성에서 우수한 결과를 보여 주었으며, 이는 실제 사용 상황에서의 연속적인 얼굴 인식이 가능하다는 것을 의미합니다. 실험 결과는 CLFace가 다양한 CL 시나리오에서 기존 방법들보다 우수한 성능을 발휘함을 입증합니다.



### Sli2Vol+: Segmenting 3D Medical Images Based on an Object Estimation Guided Correspondence Flow Network (https://arxiv.org/abs/2411.13873)
- **What's New**: 본 연구는 Sli2Vol+라는 새로운 자기 지도 학습(self-supervised learning) 기반 마스크 전파(mask propagation) 프레임워크를 제안하여, 단일 레이블(slice)만으로 3D 의료 이미지를 효과적으로 세분화할 수 있는 방법을 개발했습니다. 기존의 기술들은 여러 연속 슬라이스 간의 오류 누적 문제 및 불연속성을 효과적으로 처리하지 못하는 한계가 있었습니다. 제안한 방법은 훈련 단계에서 주어진 레이블 슬라이스에서 생성된 유사 레이블(pseudo-labels)을 활용하여 연속 슬라이스 간의 신뢰할 수 있는 대응을 학습합니다.

- **Technical Details**: 이 방법에서는 Object Estimation Guided Correspondence Flow Network(OEG-CFN)를 통해 슬라이스 간의 관계를 학습합니다. 훈련 단계에서는 주어진 라벨 슬라이스를 기반으로 나머지 슬라이스로 유사 레이블을 전파하고, 테스트 단계에서는 학습된 대응 정보를 사용해 라벨 슬라이스를 다른 슬라이스로 전파합니다. 이러한 방식은 오류 드리프트(error drift)와 불연속성 문제를 효과적으로 해결합니다.

- **Performance Highlights**: 본 연구는 CT와 MRI를 포함한 9개의 공용 데이터셋에서 다양한 해부학적 구조에 대한 세분화 성능을 평가하며, 기존 방법들에 비해 향상된 일반화 성능을 입증했습니다. 특히, 침습적이지 않은 방법으로 단 하나의 라벨 슬라이스만을 이용하여도 우수한 결과를 얻을 수 있음을 보여주었습니다. 이로써, 3D 의료 이미지 세분화 분야에서의 데이터 수집 부담을 대폭 줄일 수 있는 가능성을 제시합니다.



### Decoupled Sparse Priors Guided Diffusion Compression Model for Point Clouds (https://arxiv.org/abs/2411.13860)
- **What's New**: 본 연구에서는 lossy point cloud의 압축을 위해 새로운 접근 방식을 제안합니다. 특히, sparse priors를 기반으로 하는 diffusion compression model(DPCM)을 도입하여 latency의 중복성을 줄이고, 재구성 품질을 크게 개선하고자 합니다. 이러한 방법은 각각의 latent points와 sparse priors를 독립적으로 처리하는 이중 밀도(data flow)를 통해 이루어집니다.

- **Technical Details**: 제안하는 모델은 latent points의 재구성을 효율적으로 진행하는데 중점을 두고 있습니다. 이 과정에서 original point cloud는 separate encoders를 사용하여 latent points와 decoupled sparse priors로 변환됩니다. 또한, progressive attention 기반의 conditional denoiser가 사용되어 각 encoding 및 decoding 레이어에서 geometric 및 semantic 신호를 효과적으로 처리합니다.

- **Performance Highlights**: ShapeNet 데이터셋과 MPEG 그룹의 표준 테스트 데이터셋에서 수행한 광범위한 평가 결과, 제안된 방법이 기존 최첨단 기술에 비해 우수한 rate-distortion 성능을 달성했음을 보여줍니다. 이 연구는 고압축 비율에서도 원본 point cloud의 기하학적 정확성과 세부 사항을 유지함으로써 실질적인 적용 가능성을 제시합니다.



### Dealing with Synthetic Data Contamination in Online Continual Learning (https://arxiv.org/abs/2411.13852)
Comments:
          Accepted to NeurIPS'24

- **What's New**: 이 논문에서는 AI 생성 이미지의 증가로 인해 발생할 수 있는 데이터셋 오염 문제를 다룹니다. 특히, Online Continual Learning (CL) 연구에 미치는 영향에 대해 실험적으로 조사하였으며, 오염된 데이터셋이 기존의 CL 방법의 학습을 방해할 수 있음을 입증하였습니다. 또한, 이 문제를 해결하기 위한 새로운 방법인 Entropy Selection with Real-synthetic similarity Maximization (ESRM)을 제안합니다.

- **Technical Details**: 제안된 ESRM 방법은 인공 이미지가 포함된 오염된 데이터셋에서 성능 저하를 완화하기 위해 설계되었습니다. 이 방법은 데이터의 진짜와 합성 간의 유사성을 최대화하는 접근 방식으로, 온라인 CL 모델 훈련 시 성능 저하를 줄이는 데 효과적입니다. 연구 결과는 오염이 심한 경우에서도 ESRM이 문제가 되는 성능 저하를 상당히 완화할 수 있음을 보여줍니다.

- **Performance Highlights**: 실험 결과, ESRM 방법은 다양한 조건에서 성능 저하를 효과적으로 방지할 수 있었으며, 특히 오염된 데이터셋이 심각한 경우에 성능을 더욱 개선하는 데 기여하였습니다. 이러한 성과는 향후 온라인 CL 방법 개발에 있어 중요한 기초 자료가 될 것으로 기대됩니다. 또한, 본 연구의 재현성을 위해 소스 코드는 해당 링크에서 제공됩니다.



### Multitask Learning for SAR Ship Detection with Gaussian-Mask Joint Segmentation (https://arxiv.org/abs/2411.13847)
- **What's New**: 이 논문은 멀티태스크 학습 프레임워크 MLDet를 제안하여 합성 개구 레이더(SAR) 이미지를 위한 선박 탐지를 개선하려고 합니다. MLDet는 객체 탐지(object detection), 스펙클 억제(speckle suppression), 타겟 세분화(target segmentation) 작업을 포함하고 있습니다. 각 작업에서 새로운 손실 함수와 메커니즘을 도입하여 정확성을 증가시키고, 노이즈를 감소시키며, 복잡한 배경에서 선박을 정확히 구분하는 데 초점을 맞추고 있습니다.

- **Technical Details**: MLDet은 각 작업의 효율성을 높이기 위해 여러 기술적 요소를 통합합니다. 각도 분류 손실(angle classification loss)과 스펙클 억제 작업에서의 이중 특징 융합 주의 메커니즘(dual-feature fusion attention mechanism)을 도입했습니다. 또한, 타겟 세분화 작업에서는 회전 가우시안 마스크(rotated Gaussian-mask)를 사용하여 잡음 속에서 목표 영역을 추출하는 데 도움을 주며, 가우시안 분포를 기반으로 선박 중심에 가장 높은 확률을 할당합니다.

- **Performance Highlights**: 이 연구는 SSDD+와 HRSID 데이터셋에서의 광범위한 실험을 통해 MLDet의 효과성과 우수성을 입증했습니다. MLDet는 복잡한 배경과 다양한 스케일에서도 높은 정확도로 선박을 탐지할 수 있는 잠재력을 보여줍니다. 본 연구의 결과는 기존의 선박 탐지 방법들보다 성능 면에서 개선된 접근 방식을 제공하고, 실시간 탐지가 가능하도록 지원합니다.



### Detecting Human Artifacts from Text-to-Image Models (https://arxiv.org/abs/2411.13842)
- **What's New**: 이 연구에서는 텍스트-이미지 생성 모델에서 발생하는 인간 아티팩트를 탐지하고 해결하기 위해 Human Artifact Dataset (HAD)를 제안합니다. HAD는 37,000개 이상의 이미지로 구성되어 있으며, 인간 아티팩트를 식별하고 정밀하게 로컬라이즈할 수 있도록 주석이 달려 있습니다. 새로운 Human Artifact Detection Models (HADM)을 훈련시켜 다양한 생성 도메인에서 아티팩트를 효과적으로 감지하며, 이는 유효성을 높이고, 생성된 이미지의 품질을 향상시키는 데 기여합니다.

- **Technical Details**: HAD는 인기 있는 텍스트-이미지 모델에서 생성된 이미지의 인간 아티팩트를 탐지하는 최초의 대규모 데이터셋입니다. 본 연구에서는 두 가지 주요 카테고리인 로컬 아티팩트와 글로벌 아티팩트를 정의하여, 실제 인간 이미지로 정규화된 최첨단 아키텍처로 훈련된 HADM은 기존의 방법론에 비해 우수한 성능을 보입니다. 이를 통해 HADM은 새로운 생성기에서 이미지를 일반화하여 우수한 탐지 성능을 자랑합니다.

- **Performance Highlights**: HADM은 이미지를 생성할 때 발견된 다양한 인간 아티팩트를 감지할 뿐만 아니라, 이를 개선하는 방법을 제공합니다. 이를 통해 생성된 이미지를 한층 더 정제하기 위해, HADM의 예측을 피드백으로 사용하여 디퓨전 모델을 미세 조정함으로써 아티팩트를 감소시킬 수 있음을 증명합니다. 또한, HADM을 활용하여 인간 아티팩트를 직접적으로 수정할 수 있는 새로운 반복적인 인페인팅 프레임워크의 응용을 보여줍니다.



### Segment Anything in Light Fields for Real-Time Applications via Constrained Prompting (https://arxiv.org/abs/2411.13840)
- **What's New**: 이 논문에서는 Segment Anything Model 2 (SAM 2)를 기반으로 한 새로운 조명 분야(separated light field) 세분화(segmentation) 방법을 제안합니다. 기존의 SAM 2가 조명 분야에 직접 적용될 경우 효과적이지 않기 때문에, 재훈련(retraining)이나 모델 수정 없이도 이를 조명 분야에 적응시킵니다.

- **Technical Details**: 제안된 방법은 조명 분야 제약을 활용하여 고품질(view-consistent) 조명 필드 마스크를 생성합니다. 이를 통해 SAM 2 비디오 추적(baseline) 성능을 능가하고, 7배 더 빠른 실시간(real-time) 속도를 달성했습니다. 마스크 간의 전파(propagation)를 위해 에피폴라 기하학(epipolar geometry) 단서를 활용하고, SAM 2의 잠재 공간(latent space)을 탐색하여 가리기(occlusion) 능력을 평가합니다.

- **Performance Highlights**: 제안 방법은 기존 SAM 2 모델보다 전반적으로 뛰어난 성능을 보여줍니다. 세분화 정확도는 향상되었으며, 실시간 처리 속도는 연구에서 도입된 에피폴라 기하학 덕분에 가능해졌습니다. 이러한 접근 방식은 조명 필드 이미지의 여러 뷰에서 같은 세그먼트를 인식하고 추적하는 데 효과적입니다.



### CLIPer: Hierarchically Improving Spatial Representation of CLIP for Open-Vocabulary Semantic Segmentation (https://arxiv.org/abs/2411.13836)
Comments:
          Homepange and code: this https URL

- **What's New**: 이 연구는 Contrastive Language-Image Pre-training (CLIP) 모델을 기반으로 하는 새로운 방법인 CLIPer를 제안합니다. CLIPer는 open-vocabulary semantic segmentation을 위해 spatial representation을 계층적으로 개선하는 접근 방식을 채택하고 있습니다. 이는 기존의 self-attention 맵을 개선하여 pixel-level semantic segmentation을 가능하게 하는 것이 특징입니다.

- **Technical Details**: CLIPer는 초기 층에서의 정보 융합(module)과 세밀한 보상(module)으로 구성되어 있습니다. 초기 층에서는 patch의 embedding과 attention 맵을 통합하여 spatial coherence를 개선하고, 이후 Diffusion 모델의 self-attention 맵을 사용하여 손실된 세부 사항을 보완합니다. 이러한 모듈은 조합하여 최종적으로 더 정밀한 segmentation 맵을 생성합니다.

- **Performance Highlights**: CLIPer는 여러 segmentation 데이터 세트에서 우수한 성능을 보여주었으며, ViT-L 뒷받침을 사용할 때 VOC와 COCO Object에서 각각 69.8%와 43.3%의 mIoU 점수를 기록하여 ProxyCLIP보다 각각 9.2%와 4.1% 높은 성능을 달성했습니다. 이는 CLIP 모델의 사전 훈련 없이도 높은 품질의 segmentation을 가능하게 한다는 것을 의미합니다.



### MagicDriveDiT: High-Resolution Long Video Generation for Autonomous Driving with Adaptive Contro (https://arxiv.org/abs/2411.13807)
Comments:
          Project Website: this https URL

- **What's New**: 이 논문에서는 고해상도 및 긴 스트리트 뷰 비디오를 효과적으로 생성하기 위한 새로운 접근 방식인 MagicDriveDiT를 소개합니다. 기존의 방법들이 높은 해상도와 긴 비디오 생성에서 한계를 드러내는 반면, MagicDriveDiT는 새로운 공간-시간 조건 인코딩을 채택하여 정밀한 제어를 달성합니다. 이 기술은 다각적인 시나리오를 다룰 수 있는 모델의 용량을 향상시키기 위해 flow matching을 활용합니다.

- **Technical Details**: MagicDriveDiT는 DiT 아키텍처를 기반으로 하며, 복잡한 데이터에 효과적으로 대응하기 위해 단계적 부트스트래핑 전략을 채택합니다. 이 접근 방식은 짧은 비디오에서 긴 비디오로의 전환을 가능하게 하여, 모델이 세부 사항을 포착하고 복잡한 시나리오에 일반화할 수 있도록 합니다. 또한, 다양한 해상도와 길이의 비디오를 사용하여 모델의 일반화 능력을 강화하며, 이는 새로운 스트리트 뷰 생성을 가능하게 합니다.

- **Performance Highlights**: 실험 결과 MagicDriveDiT가 기존 작업보다 높은 해상도와 더 많은 프레임을 가진 비디오를 생성함을 입증하였습니다. 이 논문은 MagicDriveDiT가 다양한 해상도와 제어 신호를 처리할 수 있는 유연성을 보여주어, 자율 주행 시뮬레이션 및 관련 작업에 적용될 수 있는 가능성을 확장합니다. 전체적으로 MagicDriveDiT는 고품질 비디오 생성의 새로운 기준을 제시합니다.



### Hugging Rain Man: A Novel Facial Action Units Dataset for Analyzing Atypical Facial Expressions in Children with Autism Spectrum Disorder (https://arxiv.org/abs/2411.13797)
Comments:
          Portions of the dataset, features, and pretrained models are accessible at: this https URL

- **What's New**: 이번 연구는 자폐 스펙트럼 장애(ASD) 아동의 비정형 얼굴 표현을 분석하기 위한 Hugging Rain Man (HRM) 데이터셋을 소개합니다. 이 데이터셋은 ASD 아동과 정상 발달(TD) 아동의 얼굴 행동 단위(Facial Action Units, AUs)를 수동으로 주석 처리한 결과를 포함하고 있으며, 총 약 130,000장의 이미지를 제공합니다. 또한 이 연구는 정적 이미지에서의 AU 분석과 함께 시간 회귀 모델을 통해 동적 표현의 비정형성을 분석하여 주관적 인식과 객관적 얼굴 특성 간의 격차를 메우고자 합니다.

- **Technical Details**: HRM 데이터셋은 22개의 AUs와 10개의 행동 설명자(Action Descriptors, ADs) 및 비정형 평가를 포함합니다. 연구진은 HRM 데이터셋에서 정적 얼굴 이미지에 대한 통계 분석을 수행하였으며, ASD 그룹은 동일한 감정 표현을 보여줄 때 TD 그룹에 비해 더 불규칙하고 다양한 표현 패턴을 보임을 확인했습니다. 이를 통해 연구진은 AUs 및 ADs의 조합을 이용해 복합적인 감정을 전달하는 방법에 대한 깊이 있는 분석을 제공하였습니다.

- **Performance Highlights**: 연구진은 HRM 데이터셋에서 여러 AU 감지 모델을 훈련시키고 성능을 평가하였습니다. ASD 아동군은 행복, 놀라움, 슬픔이라는 세 가지 기본적인 희감 표현에서 여러 AUs/ADs에서 유의미한 차이를 보였으며, 이는 ASD 아동의 비정형 얼굴 표현에 대한 깊이 있는 이해를 제공하는 데 기여합니다. 향후 연구에 유용한 기초 결과가 제시되었으며, 이 데이터셋은 ASD 조기 선별의 잠재적인 도구로도 활용될 수 있습니다.



### GalaxyEdit: Large-Scale Image Editing Dataset with Enhanced Diffusion Adapter (https://arxiv.org/abs/2411.13794)
Comments:
          10 pages, 6 figures

- **What's New**: 이 연구에서는 대규모 이미지 편집 데이터셋인 GalaxyEdit를 제안하여, 객체 추가 및 제거 작업을 위한 자동화된 데이터 생성 파이프라인을 구축했습니다. 또한, ControlNet-xs 아키텍처를 활용하여 경량화된 작업 전용 어댑터를 포함한 연구를 진행하였습니다. 이를 통해, 복잡한 편집 지침을 처리할 수 있는 모델을 개발하였으며, 실험 결과 기존 방법들보다 개선된 성과를 보였습니다.

- **Technical Details**: GalaxyEdit 데이터셋은 다양한 편집 지령을 지원하기 위해 포괄적인 데이터 생성 파이프라인을 구축하였고, 이 과정은 객체 검출, 규칙 기반 필터링, 마스크 기반 인페인팅, 지령 생성의 네 가지 단계로 구성됩니다. ControlNet-xs는 인코더 간의 양방향 정보 흐름을 통해 생성 과정을 효과적으로 조정하며, 기존의 가벼운 네트워크에 비해 정보 교환의 효율성이 향상된 것을 보여줍니다. 또한, Volterra Neural Networks (VNN)를 도입하여 비선형 상호작용을 개선하였습니다.

- **Performance Highlights**: GalaxyEdit 데이터셋을 사용하여 SD v1.5 모델을 미세 조정한 결과, 모델이 필요한 작업에 대해 더 넓은 범위의 객체와 복잡한 편집 지침을 성공적으로 처리할 수 있음을 확인했습니다. 특히, 객체 추가 및 제거 작업에서 각각 11.2% 및 26.1%의 FID 점수 개선을 달성하였고, 실험을 통해 제안한 방법론의 효과성을 보여주었습니다.



### Edge-Cloud Routing for Text-to-Image Model with Token-Level Multi-Metric Prediction (https://arxiv.org/abs/2411.13787)
- **What's New**: 본 논문에서는 RouteT2I라는 새로운 경량화된 텍스트-이미지 모델 라우팅 프레임워크를 제안합니다. 이 프레임워크는 사용자 요청의 복잡성에 따라 대용량 클라우드 모델과 경량 엣지 모델을 동적으로 선택하여 적절한 이미지 생성 모델을 활용합니다. RouteT2I는 여러 품질 메트릭을 평가하여 생성된 이미지의 예상 품질을 예측하며, 이로 인해 비용을 절감하는 동시에 높은 품질의 이미지를 유지합니다.

- **Technical Details**: RouteT2I는 다차원 품질 메트릭을 수립하여 생성된 이미지와 이를 설명하는 긍정 및 부정 텍스트 간의 유사성을 평가합니다. 인공지능에서의 Mixture-of-Experts (MoE) 네트워크를 기반으로 하는 이 구조는 사용자 프롬프트의 핵심 토큰을 식별하고, 이들 각각이 품질에 미치는 영향을 비교하여 품질 예측을 수행합니다. 또한, Pareto 상대 우월성(PRS)을 도입하여 엣지와 클라우드에서 생성된 이미지의 품질 차이를 비교합니다.

- **Performance Highlights**: 제안된 RouteT2I 프레임워크는 COCO2014 데이터셋을 사용하여 평가되었으며, 50%의 라우팅 비율에서 상대 품질 개선이 83.97%로 나타났습니다. 이 방식은 클라우드 요청 수를 70.24% 줄이면서도 품질 목표의 50%를 유지하였습니다. 이를 통해 RouteT2I가 엣지 디바이스의 비용 효율성과 클라우드 모델의 품질을 동시에 체감 가능하다는 점을 입증합니다.



### Segment Any Class (SAC): Multi-Class Few-Shot Semantic Segmentation via Class Region Proposals (https://arxiv.org/abs/2411.13774)
Comments:
          8 pages, 2 figures, 3 tables

- **What's New**: Segment Any Class (SAC)는 기존의 Segment-Anything Model (SAM)에 비해 더 발전된 방법으로, 자동화된 다중 클래스 분할을 위한 새로운, 교육 없는 접근 방식을 제시합니다. SAC는 사용자가 지정한 인스턴스 참조 프롬프트에 기반하여 클래스 인식 프롬프트를 생성할 수 있는 Class-Region Proposals (CRP)를 쿼리 이미지에서 생성하여, 수동 입력 없이도 분할을 가능하게 합니다. 이 방법은 다양한 N-way K-shot 구성에 대응할 수 있으며, 그 결과 기존의 방법들보다 우수한 성능을 보여 주었습니다.

- **Technical Details**: SAC는 추가적인 모델 교육 없이도 깊이 있는 특성 학습과 인식 능력을 확립할 수 있는 시스템입니다. 이 방법은 자동 프롬프트 생성을 이용하여 멀티 클래스 몇 샷 의미 분할 (Few-shot Semantic Segmentation) 작업에서 우수한 성과를 달성합니다. SAC는 이전에 학습한 개념에 대한 보존을 보장하면서도 새로운 데이터셋에 적합할 수 있도록 설계되었습니다.

- **Performance Highlights**: COCO-20i 벤치마크에서 SAC는 첨단 기술들을 초과하는 성능을 보였으며, 특히 N-way 클래스 시나리오에서 두각을 나타냈습니다. 기존의 경량화된 모델의 보편적 특징을 학습하는 것도 흥미로운 시점이며, 이 방식은 소규모 데이터셋의 사용에도 변환할 수 있는 기존 기초 모델에 대한 수정 없이도 빠른 온라인 작업 적응을 허용합니다.



### FAST-Splat: Fast, Ambiguity-Free Semantics Transfer in Gaussian Splatting (https://arxiv.org/abs/2411.13753)
- **What's New**: FAST-Splat는 빠르고 모호함 없는 의미적 Gaussian Splatting을 위해 설계된 혁신적인 방법입니다. 이 방법은 느린 훈련과 렌더링 속도, 높은 메모리 사용량, 그리고 애매한 객체 위치 지정을 해결하고자 합니다. 기존의 방법들과 비교했을 때, FAST-Splat는 사용자 제공의 자연어 쿼리에서도 명확한 의미 객체 위치 지정 결과를 제공합니다.

- **Technical Details**: FAST-Splat는 CLIP와 같은 비전-언어 모델을 활용하여 폐쇄 세트(Closed-set) 의미적 Gaussian Splatting을 개방형 세계(Open-set) 설정으로 확장합니다. 이 과정에서 각 Gaussian에 특정 의미 코드를 직접 추가하여 훈련 및 렌더링 속도를 향상시킵니다. 이러한 의미 코드와 해시-테이블을 사용하여 FAST-Splat는 개방형 쿼리에 대한 의미 유사성을 측정하고, 객체에 대한 명확한 의미 레이블과 3D 마스크를 제공합니다.

- **Performance Highlights**: FAST-Splat는 기존의 방법들과 비교하여 4배에서 6배 빠른 훈련 속도와 18배에서 75배 빠른 렌더링 속도를 자랑합니다. 또한, 약 3배 적은 GPU 메모리를 요구하며, 기존 방법들과 비교하여 비슷하거나 더 나은 의미 분할 성능을 보여줍니다. 이러한 성능 개선은 ROBOTICS와 같은 다운스트림 응용 프로그램에서 중요한 역할을 할 것입니다.



### Delta-Influence: Unlearning Poisons via Influence Functions (https://arxiv.org/abs/2411.13731)
Comments:
          Accepted at NeurIPS Workshop on Attributing Model Behavior at Scale (ATTRIB @ NeurIPS 2024)

- **What's New**: 본 논문에서는 기계 학습 모델의 데이터 손상 공격에 대한 신뢰성을 높이기 위해 새로운 접근 방식인 Δ-Influence를 소개합니다. 기존의 영향 함수(influence functions)들은 데이터 손상 공격에 의해 발생한 비정상적인 모델 행동을 특정 훈련 데이터에서 정확하게 식별하는 데 실패했습니다. Δ-Influence는 단 하나의 손상된 테스트 예제만으로도 비정상적인 모델 행동을 추적할 수 있도록 설계되었습니다.

- **Technical Details**: Δ-Influence는 데이터 변환(data transformations)을 적용하여 손상된 훈련 데이터와 손상된 테스트 포인트 간의 연결을 차단합니다. 이를 통해 데이터 변환 후 영향 점수(influence scores)의 큰 감소인 'influence collapse'를 감지하여 손상된 훈련 데이터를 정확히 식별합니다. 이 접근법은 단일 테스트 포인트를 유용하게 활용하여 훈련 데이터 포인트 손상을 식별하고 이를 회복하기 위한 탈취(unlearning) 알고리즘을 적용합니다.

- **Performance Highlights**: Δ-Influence는 세 개의 비전 기반 데이터 손상 공격과 세 개의 데이터셋에서 유효성을 검증했으며, 네 가지 탐지 알고리즘과 다섯 가지 탈취 전략을 비교했습니다. 모든 설정에서 Δ-Influence는 지속적으로 최상의 탈취 성과를 달성하였으며, 데이터 손상 공격에 대한 강력한 방어 및 모델 정확성을 유지하는 데 유망함을 보여주었습니다.



### Developing Normative Gait Cycle Parameters for Clinical Analysis Using Human Pose Estimation (https://arxiv.org/abs/2411.13716)
- **What's New**: 이 논문에서는 RGB 비디오 데이터와 2D 인간 포즈 추정을 활용하여 규범적인 운동학적 보행 매개변수를 개발하기 위한 데이터 기반 방법론을 제시합니다. 기존의 임상 보행 분석 접근법이 공간 및 시간의 비정상적인 측정을 충분히 지원하지 못하는 문제를 해결하고자 하며, 임상의가 정상 집단과 비교하여 보행 주기 내에서 여러 관절 각도를 동시에 측정할 수 있는 기능을 제공합니다.

- **Technical Details**: 보행 분석은 건강 관리, 로봇 공학, 생체 인식 및 감시 분야에서 매우 중요합니다. 본 연구에서는 주로 관절 각도 분석을 통해 클리닉에서 요구하는 임상적 정보와 자동화를 동시에 지원하며, 이는 RGB 비디오의 단일 장비를 이용하여 구현됩니다. 또한, 맥락 기반의 운동 특성을 바탕으로 하는 새로운 방법론을 통해 보행 분석의 광범위한 제공을 목표로 합니다.

- **Performance Highlights**: 이 연구는 2D 인간 포즈 추정 및 비디오 기반 분석을 통해 비정상적인 보행 패턴을 실시간으로 식별 및 시각화하는 데 있어 임상의의 임상 능력을 확장합니다. 분석의 자동화 및 객관적인 의사 결정을 지원하며, 임상적 정의를 통해 수집된 표준화된 운동학적 보행 매개변수를 제시하여, 다양한 움직임의 비정상성을 보다 정확하게 분석할 수 있게 됩니다.



### Decompose and Leverage Preferences from Expert Models for Improving Trustworthiness of MLLMs (https://arxiv.org/abs/2411.13697)
- **What's New**: 이번 논문에서는 DecompGen이라는 새로운 프레임워크를 제안합니다. DecompGen은 오픈 소스 전문가 모델의 앙상블을 사용하여 MLLM(다중모달 대형 언어 모델)의 응답을 유효성 검사하는 방법을 체계화합니다. 이 접근법은 기존의 폐쇄형 모델 평가 방식의 제한을 극복하고자 합니다.

- **Technical Details**: DecompGen은 각 응답을 원자적(atomic) 유효성 검사 작업으로 분해하고, 각 작업을 적절한 전문가 모델에 할당하여 세분화된 평가를 생성합니다. 이러한 방식은 MLLMs의 긴 및 조합적인 응답을 효과적으로 처리하는 데 기여합니다. 최종적으로 DecompGen을 통해 생성된 피드백은 자동으로 선호 데이터셋인 DGPref를 구성하는 데 사용됩니다.

- **Performance Highlights**: DGPref와 선호 학습(preference learning)을 통해 조정된 MLLMs는 신뢰성(trustworthiness)에서 개선 결과를 나타내었습니다. 이러한 결과는 DecompGen의 효과성을 입증합니다. 즉, DecompGen을 통해 MLLMs의 응답 품질을 향상시킬 수 있음을 보여줍니다.



### Extending Video Masked Autoencoders to 128 frames (https://arxiv.org/abs/2411.13683)
Comments:
          10.5 pages of main paper, 25 pages total, 4 figures and 10 tables. To appear in NeurIPS'24

- **What's New**: 최근 비디오 이해 기술은 Masked Autoencoders (MAE) 기반의 비디오 기초 모델들의 강력한 성능으로 큰 발전을 이루었습니다. 본 연구에서는 128프레임 길이의 긴 비디오 시퀀스를 처리할 수 있는 효과적인 adaptive decoder masking 전략을 제안합니다. 이 접근법은 우선순위를 두는 토큰 선택에서 강력한 MAGVIT 기반의 토크나이저를 활용하여 훈련 및 재구성이 이루어집니다.

- **Technical Details**: 본 연구의 중심은 content-dependent adaptive masking 전략으로, 이는 비디오 토큰의 중요도를 학습하고 이를 사용하여 높은 순위의 토큰을 선택하는 방식을 채택합니다. 또한, Quanitzed tokens를 재구성 목표로 삼이 더 우수한 성능을 달성하게 됩니다. 기존 비디오 모델들은 일반적으로 16 또는 32프레임의 짧은 비디오에 집중했으나, 본 연구의 LVMAE 접근법은 긴 비디오를 효과적으로 처리할 수 있도록 설계되었습니다.

- **Performance Highlights**: 이 연구의 결과, LVMAE는 Diving48에서 3.9 포인트, EPIC-Kitchens-100 동사 분류에서 2.5 포인트의 성능 향상을 보이며 기존 최첨단 성능을 초과했습니다. 해당 연구는 추가적인 언어 감독이나 레이블 없이 표준 ViT 인코더와 단일 시간적 샘플링만으로 이 성능을 달성했습니다. 이를 통해 긴 비디오가 MAE 사전 훈련에서 가져오는 이점을 증명하였습니다.



### FabuLight-ASD: Unveiling Speech Activity via Body Languag (https://arxiv.org/abs/2411.13674)
Comments:
          23 pages, 8 figures, 3 tables, accepted for publication in Neural Computing and Applications

- **What's New**: 이 논문은 멀티모달 환경에서의 능동 화자 감지(ASD) 기술을 위한 FabuLight-ASD라는 새로운 모델을 제시합니다. 이 모델은 얼굴, 오디오 및 신체 자세 정보를 통합하여 감지 정확도와 강인성을 향상시킵니다. 기존의 Light-ASD 프레임워크를 기반으로 하여 신체 포즈 데이터를 추가하여 계산 오버헤드를 최소화하며, 현실 세계에서의 적용 가능성을 높입니다.

- **Technical Details**: FabuLight-ASD는 Face, audio와 body pose 정보를 활용하여 경량의 능동 화자 탐지를 위한 모델로, 사람이 말하는 활동의 원천을 탐지하는 데 신체 자세의 중요성을 강조합니다. 이 모델은 WASD(Wilder Active Speaker Detection) 데이터셋을 사용하여 평가되었으며, 사람의 자세 정보를 통합하여 얼굴 표현과 오디오 신호의 단서를 보완합니다. 결과적으로 94.3%의 평균 정확도(mAP)를 달성하여 Light-ASD의 93.7%와 비교해 성능이 향상되었습니다.

- **Performance Highlights**: FabuLight-ASD는 발화 장애, 얼굴 가림, 인간 음성 배경 소음 등의 어려운 시나리오에서도 특히 mAP의 개선을 보여주었습니다. 이 모델은 파라미터 수(27.3%)와 곱셈-누적 연산(multiply-accumulate operations)의 소폭 증가(최대 2.4%)에도 불구하고 효율성을 유지하여 실용성이 강조됩니다. 이러한 발견은 신체 자세 데이터 통합을 통해 ASD 성능 향상의 효과를 입증합니다.



### ID-Patch: Robust ID Association for Group Photo Personalization (https://arxiv.org/abs/2411.13632)
Comments:
          Project Page is: this https URL

- **What's New**: 이 연구는 개별적인 그룹 사진을 합성하여 각 신원의 위치를 지정하는 ID-Patch라는 새로운 방법을 제안합니다. 기존 기술에서 발생할 수 있는 신원(ID) 유출 문제를 해결하며, 이는 다수의 얼굴 특징들이 상호작용하여 발생하는 현상입니다. 이 방법은 두 가지 주요 요소인 ID 패치와 ID 임베딩을 활용하여 고유한 얼굴 정체성과 위치 제어를 강력히 연결합니다.

- **Technical Details**: ID-Patch 방식은 동일한 얼굴 특징에서 생성된 ID 패치와 ID 임베딩을 통해 조직됩니다. ID 패치는 조건부 이미지에 정확한 위치 제어를 가능하게 하며, ID 임베딩은 텍스트 임베딩과 통합되어 높은 유사성을 보장합니다. 이러한 설계는 신원 유출을 줄이고, 이미지 생성 과정에서 시각적 왜곡을 최소화하도록 돕습니다.

- **Performance Highlights**: 실험 결과 ID-Patch는 얼굴 ID 유사성, ID-위치 연관 정확도, 생성 효율성 등 다양한 지표에서 기존 방법보다 우수한 성능을 보였습니다. 이러한 성과는 ID-Patch가 개인화된 그룹 사진 합성에서의 잠재력과 혁신성을 입증합니다.



### Sparse Input View Synthesis: 3D Representations and Reliable Priors (https://arxiv.org/abs/2411.13631)
Comments:
          PhD Thesis of Nagabhushan S N, Dept of ECE, Indian Institute of Science (IISc); Advisor: Dr. Rajiv Soundararajan; Thesis Reviewers: Dr. Kaushik Mitra (IIT Madras), Dr. Aniket Bera (Purdue University); Submitted: May 2024; Accepted and Defended: Sep 2024; Abstract condensed, please check the PDF for full abstract

- **What's New**: 이 연구는 특히 정적(static) 및 동적(dynamic) 장면을 위한 희소 입력(sparse input) 기반의 새로운 보기 합성(novel view synthesis) 문제에 주목하고 있습니다. 기존 방법들이 고품질 이미지를 위해 다수의 입력 뷰를 필요로 하는 반면, 본 논문에서는 제한된 입력 뷰로도 효과적으로 작업할 수 있는 접근 방식을 제안합니다.

- **Technical Details**: 전통적인 3D 표현 방식인 radiance fields 및 multi-plane images는 여러 입력 뷰에서 생성된 이미지 품질을 크게 향상시킵니다. 그러나 이러한 모델은 고품질 렌더링을 위해 밀집 샘플링(dense sampling)을 요구합니다. 본 연구는 이러한 제한을 극복하기 위한 새로운 기법을 제시하며, 이를 통해 다양한 응용 분야에서 활용 가능성을 높이기 위해 노력하고 있습니다.

- **Performance Highlights**: 이 연구의 주요 성과는 소수의 입력 뷰만으로도 고품질 이미지를 합성할 수 있는 가능성을 보여준다는 점입니다. 이는 메타버스(meta-verse), 이벤트 자유 시청(free-view watching), 게임(video gaming) 및 영상 안정화(video stabilization)와 같은 다양한 분야에서 응용될 수 있습니다. 따라서 이러한 접근 방식은 컴퓨터 비전과 그래픽스 분야에서 새로운 가능성을 제시할 것으로 기대됩니다.



### MambaDETR: Query-based Temporal Modeling using State Space Model for Multi-View 3D Object Detection (https://arxiv.org/abs/2411.13628)
- **What's New**: 이번 논문에서는 3D 탐지(3D detection)의 성능 향상을 위해 효율적인 상태 공간(efficient state space)에서 시계열 정보(temporal information)를 활용한 MambaDETR라는 새로운 방법을 제안합니다. 기존의 transformer 기반 시간 융합 방법들은 시퀀스 길이가 증가할수록 계산 비용(computational cost)이 제곱적으로 증가하고 정보 감소(information decay)가 발생하는 단점을 가지고 있습니다. MambaDETR는 이러한 단점을 해결하기 위해 설계되었습니다.

- **Technical Details**: MambaDETR는 다양한 프레임 시퀀스의 정보 손실을 줄이기 위해 효율적인 상태 공간을 통한 시간 융합(temporal fusion)을 구현합니다. 이와 함께, 우리는 Motion Elimination 모듈을 설계하여 시계열 융합을 위해 상대적으로 정적인 객체들을 제거(join 시켜주기 위해서)를 기능합니다. 이러한 접근은 3D 탐지 시스템의 전반적인 성능을 향상시키는 데 기여합니다.

- **Performance Highlights**: MambaDETR는 표준 nuScenes 벤치마크에서 뛰어난 결과를 달성하며, 기존의 시계열 융합 방법들 사이에서 최첨단 성능(state-of-the-art performance)을 보여줍니다. 이 성과는 자율주행 분야에서 3D 객체 탐지(3D object detection) 임무 수행의 새로운 가능성을 제시합니다.



### Principles of Visual Tokens for Efficient Video Understanding (https://arxiv.org/abs/2411.13626)
- **What's New**: 이번 논문에서는 비디오 이해(Video Understanding) 분야의 최근 발전을 다루고 있으며, 특히 transformer 아키텍처의 효율성을 개선하는 연구에 집중하고 있습니다. 기존의 많은 방법들이 accuracy(정확도)와 computation(계산량) 간의 trade-off(상쇄)를 이루는 데에 그치고 있는 반면, 제안된 LITE 모델은 token의 가치를 고려하여 우수한 성능을 보이는 새로운 접근 방식을 제시합니다. 이를 통해 데이터 세트(Kinetics400 및 Something-Something-V2)에서 기존 방법들보다 우수한 결과를 도출했습니다.

- **Technical Details**: 연구진은 token의 가치를 분석하기 위해 oracle을 개발하였으며, 이를 통해 대부분의 token들이 Pareto distribution을 따르며 소량의 정보만을 보유하고 있다는 것을 발견했습니다. LITE 모델은 lightweight(경량) neural network인 selector network를 통해 token을 효율적으로 선택하며, MLP(Multi-Layer Perceptron)을 기반으로 설계되었습니다. 또한, 이러한 접근 방식은 기존 방법들과 비교하여 더욱 정확한 결과를 제공합니다.

- **Performance Highlights**: LITE 모델은 token 선택과 병합에서 이전의 모든 비교 방법들을 초월하여, 경우에 따라 최대 9%의 정확도 개선을 이루었습니다. 특히, 쉬운 비디오의 경우 더 적은 수의 token으로도 정확도를 유지할 수 있음을 보여주며, 이는 video complexity(비디오 복잡성)에 따라 계산 자원을 조정할 수 있는 가능성을 포함합니다. 이러한 결과는 향후 비디오 이해 분야에서의 효율성과 정확성을 동시에 향상시키는 데 기여할 것입니다.



### Unsupervised Foundation Model-Agnostic Slide-Level Representation Learning (https://arxiv.org/abs/2411.13623)
- **What's New**: 이 논문은 통합된 단일 모드 자기 지도 학습(SSL) 방법인 COBRA를 제안합니다. 이 방법은 여러 FMs(Foundation Models)에서 타일 임베딩을 통합하여 슬라이드 표현을 생성하며, 고유한 전처리 과정과 비교 손실(constrastive loss) 방식을 활용하여 슬라이드 레벨 임베딩을 효과적으로 만들어냅니다. 기존 연구와 달리 COBRA는 적은 수의 WSI(Whole-Slide Images)에 대해 높은 성능을 발휘하며, 새로운 FMs에 대한 즉시 호환이 가능하다는 점에서 혁신적입니다.

- **Technical Details**: COBRA는 Mamba-2 아키텍처를 기반으로 하여 기능 공간에서 타일 임베딩을 생성하는 데 중점을 두고 있습니다. 이 방법은 사전 훈련된 패치 임베딩을 사용하여 훨씬 더 유연하게 다양한 FMs에 적용될 수 있습니다. 여러 변조(augmentation)를 통해 정렬된 여러 표현을 생성하는 것이 이 방법의 핵심입니다. COBRA는 이미지 패치 기반 훈련에서 발생할 수 있는 다양한 문제점들을 해결하는 데 중점을 둡니다.

- **Performance Highlights**: COBRA는 3048개의 WSI만으로 사전 훈련된 반면, 최신 슬라이드 인코더보다 평균적으로 +3.8% AUC 성능 향상을 달성했습니다. 또한 15개의 분류 작업에서 기존 슬라이드 인코더보다 우수한 결과를 기록했으며, 특히 낮은 배율의 WSI에서 계산 효율성을 높이면서도 분류 성능을 최소한으로 저하시키는 결과를 보여줍니다. 이로 인해 COBRA는 새로운 FMs 개발로 인한 변화에도 유연하게 대응할 수 있는 가능성을 제시합니다.



### Robust SG-NeRF: Robust Scene Graph Aided Neural Surface Reconstruction (https://arxiv.org/abs/2411.13620)
Comments:
this https URL

- **What's New**: 본 연구에서는 Neural Surface Reconstruction에서 아웃라이어(noisy camera poses)의 영향을 최소화하는 방법을 제안합니다. 기존의 방법들은 작은 노이즈(inliers) 처리에는 효과적이나 큰 노이즈(outliers)에 대해서는 어려움을 겪었습니다. 저자들은 scene graph 정보를 활용하여 confidence estimation을 통한 개선된 전략을 도입하여 아웃라이어의 영향을 줄이고, 최종 성능을 향상시키기 위해 재투영 손실(re-projection loss)과 Monte Carlo re-localization 방법을 적용했습니다.

- **Technical Details**: 제안된 방법은 두 개의 color networks를 사용하여 아웃라이어와 인라이어를 구분합니다. 이들 네트워크는 각각 전통적인 NSR 접근 방식과 confidence estimation을 위한 네트워크로 구성되며, SDF(Signed Distance Function)와 pose estimations에 기반한 손실을 포함합니다. 또한, 아웃라이어에 대해 Monte Carlo re-localization 방식으로 더 나은 초기화와 인라이어에 대해서는 재투영 손실과 Intersection-of-Union(IoU) 손실을 통해 기하학적 제약을 강화합니다.

- **Performance Highlights**: SG-NeRF 및 DTU 데이터셋에서 실험한 결과, 본 방법은 3D 재구성 품질 및 pose 정확도를 지속적으로 향상시키는 성과를 보여주었습니다. 아웃라이어 포즈를 효과적으로 수정하는 동시에, 기존의 이미지 기반 렌더링 방식에 비해 우수한 재구성을 이끌어냈습니다. 본 연구는 포즈-NeRF 공동 학습 과정에서 정적이지 않고 동적으로 매칭 쌍을 업데이트하는 최초의 접근 방식을 제안합니다.



### Non-Linear Outlier Synthesis for Out-of-Distribution Detection (https://arxiv.org/abs/2411.13619)
- **What's New**: 이 논문에서는 NCIS라는 새로운 접근법을 제안하며, 이는 합성 아웃라이어를 생성하는 품질을 향상시키기 위해 확산 모델의 임베딩 공간을 직접 활용합니다. 또한, 조건부 볼륨 보존 네트워크를 통해 클래스 조건부 매니폴드(class-conditional manifold)를 모델링하여 훈련 분포의 표현력이 향상됩니다. 이로 인해 표준 데이터셋인 ImageNet-100과 CIFAR-100에서 새로운 최첨단 OOD 탐지 성능을 달성했습니다.

- **Technical Details**: NCIS에서 두 가지 새로운 기술을 개발하였습니다. 첫 번째는 확산 기반 임베딩으로, 이미지를 생성할 때 확산 모델을 직접 사용하여 이미지 임베딩을 만들어 ID 영역을 정밀하게 묘사합니다. 두 번째는 비선형 파라메트릭 분포로, 이는 클래스별 매니폴드를 적합하기 위해 조건부 볼륨 보존 네트워크(cVPN)를 도입하여 기존의 단순한 가우시안 분포보다 복잡한 클래스를 더 잘 모델링할 수 있게 합니다.

- **Performance Highlights**: NCIS는 ImageNet-100과 CIFAR-100의 두 주요 벤치마크에서 최첨단 성능을 달성하였으며, 각 구성 요소의 개별적인 영향을 확인하는 애블레이션 연구를 통해 그 효과성이 입증되었습니다. 또한, 데이터 사전 처리와 기타 주요 설계 선택들의 중요성에 대한 인사이트도 제공합니다. 이 연구에서 만든 코드도 공개되어 있어 연구자들이 쉽게 활용할 수 있도록 하였습니다.



### Video2BEV: Transforming Drone Videos to BEVs for Video-based Geo-localization (https://arxiv.org/abs/2411.13610)
- **What's New**: 이 논문은 드론의 시각 정보를 활용한 새로운 비디오 기반 드론 지오로컬라이제이션(geo-localization) 작업을 제안합니다. 기존의 이미지 기반 접근 방식이 드론의 비디오 출력을 충분히 활용하지 못하고 환경적 제약에 민감하다는 문제를 해결하고자 합니다. 이를 통해 제안된 Video2BEV(paradigm)는 드론 비디오를 Bird's Eye View(BEV)로 변환하여 매칭 과정을 간소화합니다.

- **Technical Details**: Video2BEV는 Gaussian Splatting 기법을 사용하여 3D 장면을 재구성하고 BEV 프로젝션을 얻습니다. 기존 변환 방법(예: polar transform)과는 달리, 이 방법은 2D 이미지를 기반으로 하지 않고 비디오에서 유도된 3D 장면을 사용하여 고유한 세부정보를 보존하고 왜곡을 최소화합니다. 또한, Diffusion-based 모듈을 통해 하드 네거티브 샘플을 생성하여 판별적 특징 학습을 촉진합니다.

- **Performance Highlights**: 새로 소개된 UniV 데이터셋을 통해 Video2BEV는 경쟁력 있는 재현율(recall rate)을 달성하며 기존 비디오 기반 방법을 초월합니다. 특히 낮은 고도에서 발생하는 더 많은 가림에도 불구하고 더 나은 견고성을 보입니다. 실험 결과는 Video2BEV의 고전적인 비디오 기반 방법 대비 뛰어난 성능을 명확히 입증하고 있습니다.



### What You See Is What Matters: A Novel Visual and Physics-Based Metric for Evaluating Video Generation Quality (https://arxiv.org/abs/2411.13609)
- **What's New**: 최근의 비디오 생성 모델들이 급속도로 발전함에 따라, 생성된 비디오의 품질을 평가하는 것이 점점 더 중요해지고 있습니다. 기존의 평가 지표들은 인간의 시각적 관점에서 품질을 완전하게 반영하지 못해, 외관과 동작의 일관성을 간과하는 문제점이 있었습니다. 본 논문에서는 VAMP(Visual Appearance and Motion Plausibility)라는 새로운 지표를 제안하여, 높은 시각적 품질과 물리적 타당성을 동시에 평가하는 방안을 소개합니다.

- **Technical Details**: VAMP는 두 가지 주요 구성 요소로 이루어져 있습니다: 외관 점수(appearance score)와 동작 점수(motion score)입니다. 외관 점수는 색상, 형태, 텍스처의 일관성을 평가하고, 동작 점수는 객체 이동의 사실성을 판단합니다. VAMP는 다양한 손상(real videos with corruption) 실험 및 생성 비디오 평가(generated video evaluation)에서 그 유효성을 검증했습니다.

- **Performance Highlights**: VAMP는 기존의 지표들에 비해 시각적 충실도와 시간적 일관성을 효과적으로 포착하여, 비디오 품질을 보다 포괄적으로 평가할 수 있는 능력을 보여주었습니다. 본 연구 결과는 VAMP가 매우 경쟁력 있는 평가 도구로 작용할 수 있음을 시사하며, 비디오 생성 기술의 향상을 위한 튼튼한 기초를 제공할 것으로 기대됩니다.



### VioPose: Violin Performance 4D Pose Estimation by Hierarchical Audiovisual Inferenc (https://arxiv.org/abs/2411.13607)
Comments:
          Accepted by WACV 2025 in Round 1. First two authors contributed equally

- **What's New**: 이 연구에서는 VioPose라는 새로운 다중 모달 네트워크를 제안하고 사용하여 음악 성능에서 나타나는 인체 모션을 보다 정확하게 분석하는 방법을 제시한다. 이는 밀접하게 연결된 음악과 인간 모션의 관계를 활용하여 발생하는 문제를 해결하려는 시도로, 정확한 4D 인간 포즈 추정이 가능하다.

- **Technical Details**: VioPose는 오디오와 비디오 입력을 결합해 3D 포즈를 추정하는 계층적 구조를 가지고 있다. 이 네트워크는 2D 포즈 추정 결과 및 상관관계가 있는 원시 오디오 데이터를 입력받아, 최종 결과로 3D 포즈 시퀀스, 속도 및 가속도를 예측한다. 네트워크는 1D CNN 및 Transformer 레이어를 통해 오디오 및 인체 구조의 템포럴 일관성을 추출한다.

- **Performance Highlights**: 제안된 VioPose는 기존의 SoTA 모델들과 비교할 때 우수한 성능을 보이면서 정확한 모션 분석을 가능하게 했다. 본 연구에서는 12명의 바이올린 연주자로부터 수집된 대규모 데이터셋을 활용해 모델의 효과를 증명하였다. 이를 통해 VioPose는 실제 모션 데이터와 유사한 성과를 보여주며, 음악 성능 분석에 있어 신뢰할 수 있는 도구로 자리매김하였다.



### RadPhi-3: Small Language Models for Radiology (https://arxiv.org/abs/2411.13604)
- **What's New**: 최근 LLM 기반의 코파일럿(Copilot) 도우미들이 일상 업무와 방사선학(。
Radiology) 워크플로우에서의 유용성으로 주목받고 있습니다. 본 논문에서는 Radiology 업무를 지원하기 위해 3.8B 파라미터를 가진 RadPhi-3라는 소형 언어 모델(Small Language Model)이 새롭게 소개되었습니다. 기존에 흉부 X-ray 관련 보고서에서의 요약 생성 외에도, 현재 보고서와 이전 보고서를 비교한 변화 요약 생성 및 병리학적 특성과 장치 태깅 등 다양한 유용한 작업들을 탐색하였습니다.

- **Technical Details**: RadPhi-3는 Phi-3-mini-4k-instruct 모델에서 명령 튜닝을 통해 학습되었습니다. 이는 방사선 전문 지식 소스를 이용해 방사선학 관련 질문에 대해 신뢰성 있는 답변을 제공하는 데 중점을 두었습니다. 방사선학 모델은 개인 기록에 대한 모델 성능을 귀속시키지 않는 개인정보 보호 요구사항을 준수해야 하며, RadPhi-3는 다양한 방사선 보고서 작업의 성과를 이전 모델인 RadPhi-2보다 개선하였습니다.

- **Performance Highlights**: RadPhi-3는 RaLEs 방사선 보고서 생성 벤치마크에서 SOTA(State Of The Art) 성과를 달성하였으며, 흉부 X-ray 관련 작업 및 질문 응답 작업에서 우수한 결과를 기록했습니다. 또한 새로운 두가지 방사선 보고서 관련 작업인 방사선 보고서 세분화와 변경 요약 작업에 대해 성능 평가를 실시하여 유의미한 결과를 도출했습니다. 이 모델은 여러 벤치마크에서 방사선 보고서의 각종 작업 수행 능력을 입증하였습니다.



### Enhancing Bidirectional Sign Language Communication: Integrating YOLOv8 and NLP for Real-Time Gesture Recognition & Translation (https://arxiv.org/abs/2411.13597)
- **What's New**: 이번 연구는 실시간 카메라 영상을 통해 미국 수화(ASL) 데이터를 텍스트로 변환하고, 텍스트를 수화로 변환하는 프레임워크를 개발하는 데 초점을 맞추고 있습니다. 이를 통해 언어 장벽을 허물고, 수화 사용자와의 실시간 소통을 가능하게 합니다. 본 연구는 You Only Look Once(YOLO) 및 Convolutional Neural Network(CNN) 모델을 활용하여 수화 인식 및 변환 작업을 수행합니다.

- **Technical Details**: 텍스트를 수화로 변환하기 위해 자연어 처리(NLP) 기법을 사용하여 입력된 문장에서 키워드를 추출하고 실시간으로 수화를 수행하는 비디오를 제공합니다. YOLO 모델은 실시간으로 공간-시간 특성을 자동으로 추출하며, CNN 모델은 수화 인식을 위한 실시간 처리를 제공합니다. 또한, Natural Language Toolkit(nltk)을 사용하여 품사 태깅을 수행하고, 전용 데이터셋에서 비디오를 검색하여 수화 표현을 생성합니다.

- **Performance Highlights**: 제안된 모델들은 실제 수화 데이터를 사용하여 인식 정확도, 정밀도, 번역 품질, 속도 및 견고성 등의 성능 평가를 수행하였습니다. 전체 150개의 비디오로 구성된 데이터셋을 만들어 수화 표현의 정확성을 높였습니다. 유저 인터페이스는 Django 웹 프레임워크를 기반으로 하며, 사용자가 문자를 입력하거나 오디오를 통해 수화를 요청할 수 있도록 설계되어 있습니다.



### Towards Accessible Learning: Deep Learning-Based Potential Dysgraphia Detection and OCR for Potentially Dysgraphic Handwriting (https://arxiv.org/abs/2411.13595)
- **What's New**: 이 연구는 난독증(dysgraphia) 진단 및 손글씨 인식을 위한 깊은 학습(deep learning) 기술의 적용 가능성을 탐구합니다. 어린이의 손글씨 샘플을 활용하여 난독증 증세를 가진 아동을 분류하는 새로운 모델을 개발했습니다. 특히, 맞춤형 CNN 모델이 기존의 사전 훈련된 VGG16 및 ResNet50 모델보다 우수한 성능을 보였습니다.

- **Technical Details**: 연구진은 말레이시아 초등학생의 손글씨 샘플 데이터셋을 사용하여 난독증 여부를 분류하는 커스텀 CNN 모델을 구축했습니다. 이 모델은 91.8%의 테스트 정확도를 달성하였고, 정밀도(precision), 재현율(recall) 및 AUC에서 높은 성과를 보였습니다. 또한, 난독증 손글씨에서 개별 문자를 세분화(segmentation)하고 인식(recognition)하기 위한 OCR(Optical Character Recognition) 파이프라인도 구축되었습니다.

- **Performance Highlights**: 개발된 커스텀 CNN 모델은 난독증 손글씨 특징을 식별하는 데 매우 강력함을 제시하며, 약 43.5%의 문자 인식 정확도를 기록했습니다. 이 연구 결과는 난독증 평가를 지원하는 데 있어 딥러닝의 가능성을 강조하며, 교육자 및 임상의가 난독증을 식별하고 쓰기 진행 상황을 추적하는 도구 개발의 초석이 될 것입니다.



### Improved GUI Grounding via Iterative Narrowing (https://arxiv.org/abs/2411.13591)
- **What's New**: 이번 연구에서는 GUI Grounding 능력을 강화하기 위해 Iterative Narrowing (IN)이라는 시각적 프롬프트 프레임워크를 제안했습니다. 기존의 VLM 모델의 성능 개선을 목표로 하며, GUI 인터페이스에서의 정밀한 시각적 위치 식별이 가능해집니다. IN은 초기 예측을 점진적으로 개선하는 과정을 통해 GUI grounding의 정확성을 향상시킵니다.

- **Technical Details**: 이 방법은 입력 이미지와 해당 텍스트 쿼리를 받으면, 이미지를 999×999 픽셀로 표준화하여 처리합니다. 모델은 다음 반복을 위해 예측된 좌표를 기반으로 이미지 잘라내기를 생성하며, 매 반복마다 이미지 크기를 줄이는 방식을 사용합니다. 이 반복 과정은 n 회 반복되며, 마지막 반복에서는 최종 타겟 위치를 결정합니다.

- **Performance Highlights**: ScreenSpot 벤치마크를 통한 평가 결과, IN 프레임워크는 특히 일반 VLM 모델인 InternVL-2-4B 및 Qwen2-VL-7B에서 성능 개선을 이끌어냈습니다. 그러나 공간적으로 거리가 큰 컨텍스트 단서 처리에 한계가 있어, 특정 상황에서 성능이 저하되는 경향이 있습니다. 향후 연구에서는 이러한 컨텍스트 한계를 해결하기 위한 방법에 대한 탐색이 필요합니다.



### Deep learning waterways for rural infrastructure developmen (https://arxiv.org/abs/2411.13590)
Comments:
          18 pages, 6 figures

- **What's New**: 이 연구에서는 미국의 고해상도 위성 이미지와 디지털 고도 모델을 기반으로 한 컴퓨터 비전 모델인 WaterNet을 개발했습니다. 이 모델은 기존에 매핑되지 않은 아프리카 대륙의 수로를 식별하는 데 활용됩니다. 이러한 시스템은 공공 데이터에 기반하여 인도적 필요를 포착하고 사회 개발을 위한 계획에 기여할 수 있는 잠재력을 보여줍니다.

- **Technical Details**: WaterNet은 U-Net 스타일의 합성곱 신경망(convolutional neural network)으로 개발되었습니다. 이 모델은 미국의 수로 데이터에서 레이블을 훈련하여 대규모 지리적 영역에 확장 가능하게 설계되었습니다. WaterNet은 추가적인 후처리 단계를 통해 수로 데이터를 벡터화하여 다른 데이터셋과 비교할 수 있습니다.

- **Performance Highlights**: WaterNet은 아프리카의 다양한 국가에서 지역 사회의 다리 건설 요청과 같은 인프라 필요를 충족하는 데 있어 기존 데이터셋보다 평균 93%의 정확도로 수로를 포착했습니다. 반면, OpenStreetMap은 36%와 TDX-Hydro는 62%의 낮은 성능을 보였습니다. WaterNet은 인프라 개발의 중요한 기초 자료로서 지역 사회의 필요를 정확히 반영하는 데 기여할 것으로 기대됩니다.



### Unveiling Redundancy in Diffusion Transformers (DiTs): A Systematic Study (https://arxiv.org/abs/2411.13588)
Comments:
          9 pages including reference

- **What's New**: 이번 연구는 Diffusion Transformers (DiTs) 모델의 비효율적인 추론 속도를 개선하기 위한 방법을 제시합니다. 특히, 여러 DiT 모델 간의 redundancy(중복성) 분석을 통해 효과적인 캐싱(caching) 전략을 개발할 수 있는 가능성을 모색합니다. 또한, 유연한 분석 도구인 DiTCacheAnalysis를 제공하며, 이를 통해 각 개별 모델에 맞춘 최적화된 캐싱 전략을 개발 가능하게 합니다.

- **Technical Details**: Diffusion 모델은 Gaussian noise에서 시작하여 x0 (최종 결과 이미지)로 가는 비선형 과정을 포함하며, 이 과정에서 여러 iterative(반복적인) denoising 단계를 통해 고품질 이미지를 생성합니다. DiT 아키텍처는 입력 데이터를 latent patches(잠재 패치)로 나누고, transformer의 self-attention 메커니즘을 통해 각 패치 간의 관계를 모델링합니다. 이 과정은 다수의 계산 단계로 인해 중대한 latency(지연 시간)를 야기하게 됩니다.

- **Performance Highlights**: 연구 결과, 다양한 DiT 모델 간의 중복성 분포가 크게 다르다는 것을 발견했습니다. 반면 하나의 모델 내에서 중복성 분포는 안정적이며 사용자의 프롬프트, 추론 단계, 스케줄러의 변동에도 영향을 받지 않아 일관성을 보입니다. 이를 통해 캐싱 전략을 모델별로 제안하고, 공통의 캐싱 방식은 효과적이지 않다는 결론을 내렸습니다.



### Deep Feature Response Discriminative Calibration (https://arxiv.org/abs/2411.13582)
- **What's New**: 이 논문에서는 Deep Neural Networks (DNNs)의 특성 반응을 차별적으로 보정하는 새로운 방법을 제안합니다. 기존의 기법들은 특징의 특수성을 충분히 고려하지 못하는 한계가 있었으나, 본 연구에서는 Gaussian 확률 밀도 함수를 바탕으로 원본 응답 값과 통합하여 추가적인 신뢰도 값을 계산합니다. 이 접근법은 특징 분포와 특이성의 최적화를 강조하여 모델의 출력 성능을 개선합니다.

- **Technical Details**: 저자들은 Response Calibration Layer (RC Layer)를 도입하여 ResNet 아키텍처를 수정한 Response Calibration Networks (ResCNet)를 설계했습니다. 이 방법은 각 합성곱 블록 간에 추가적인 응답 보정 가지를 삽입하여 모델의 특징에 대한 보정 값을 제공합니다. 실험 결과, 제안된 접근법이 CIFAR-10, CIFAR-100, SVHN, ImageNet과 같은 데이터 세트에서 효과적임을 입증하였습니다.

- **Performance Highlights**: 제안된 ResCNet 모델은 표준 ResNet 모델과 비교하여 향상된 분류 성능을 보여주며, 더욱 차별화된 특징을 추출하는데 기여합니다. 실험 데이터를 통해, 이 모델은 다양한 비주얼 태스크에서 탁월한 성능을 발휘하며, 각 레이어의 응답을 더 깊이 분석하여 모델 최적화를 돕는 새로운 통찰을 제공합니다.



### COOD: Concept-based Zero-shot OOD Detection (https://arxiv.org/abs/2411.13578)
- **What's New**: 이 논문에서는 COOD (concept-based OOD detection)라는 새로운 제로샷 다중 레이블 OOD 탐지 프레임워크를 소개합니다. 이 방법은 Vision-Language Model (VLM)인 CLIP을 활용하여 복잡한 레이블 의존성을 모델링하고, OOD 샘플을 효과적으로 구분할 수 있게 합니다. COOD는 추가적인 재교육 없이 정교한 레이블 확장 전략과 새로운 점수 함수를 사용하여 기술되어 있습니다.

- **Technical Details**: COOD는 긍정적인 레이블과 부정적인 레이블을 모두 포함한 개념 기반의 레이블 확장을 통해 OOD 탐지를 수행합니다. 이 프레임워크는 다중 레이블 설정에서 각 레이블에 대해 더 복잡한 의미적 공간을 모델링하여 ID와 OOD 샘플 간의 차이를 정밀하게 구분할 수 있게 합니다. 또한, 다중 레이블 레이블의 존재와 의존관계를 고려하여 유연하면서도 강력한 OOD 탐지 성능을 확보합니다.

- **Performance Highlights**: COOD는 VOC 및 COCO 데이터셋에서 약 95%의 평균 AUROC를 달성하며 기존 방법에 비해 우수한 성능을 보입니다. 특히, 다양한 레이블 수와 OOD 샘플 유형에 걸쳐 견고한 성능을 유지하며, CLIP-B/16에서 초당 약 800장의 이미지 처리 효율성을 기록합니다. 이러한 결과는 COOD가 실제 환경에서 신뢰할 수 있는 OOD 탐지 방법임을 보여줍니다.



### Public Health Advocacy Dataset: A Dataset of Tobacco Usage Videos from Social Media (https://arxiv.org/abs/2411.13572)
Comments:
          Under review at International Journal of Computer Vision (IJCV); 29 figures, 5 figures;

- **What's New**: 이 논문에서는 TikTok과 YouTube에서 수집된 5,730개의 담배 관련 비디오로 구성된 Public Health Advocacy Dataset (PHAD)를 처음으로 소개합니다. 이 데이터셋은 430만 프레임을 포함하고 사용자 참여 지표, 비디오 설명 및 검색 키워드와 같은 상세 메타데이터를 제공합니다. 이 연구는 담배 제품 및 사용 시나리오의 정확한 분류를 위한 Vision-Language (VL) Encoder를 활용하여 탁월한 성능을 보여주었습니다.

- **Technical Details**: 이 연구는 비디오 콘텐츠의 미묘한 세부사항을 이해하기 위해 두 단계를 포함하는 분류 접근 방식을 사용하였습니다. 이 시스템은 TikTok 비디오를 부정 샘플로, YouTube 비디오를 긍정 샘플로 사용하여 데이터셋을 구축하는 혁신적인 샘플링 전략을 적용합니다. 또한, 이를 통해 시각적 특징과 텍스트 기능을 활용한 분류 정확성을 크게 향상시키는 새로운 VL Encoder를 통합한 두 단계 분류 프레임워크를 제안합니다.

- **Performance Highlights**: 연구 결과, 사용자들이 특히 vaping과 전자담배 비디오에 높은 참여를 보이고 있다는 중요한 트렌드가 발견되었습니다. 이 인사이트는 공공 건강 개입 및 콘텐츠 조절 전략에 중요한 정보를 제공합니다. 본 논문은 비디오 분석에서의 메타데이터와 같은 맥락적 특징의 역할을 강조하며, 비디오 콘텐츠 이해와 분류를 위해 종합적인 접근 방식을 제시합니다.



### Multimodal 3D Brain Tumor Segmentation with Adversarial Training and Conditional Random Field (https://arxiv.org/abs/2411.14418)
Comments:
          13 pages, 7 figures, Annual Conference on Medical Image Understanding and Analysis (MIUA) 2024

- **What's New**: 이 논문은 정밀한 뇌종양 세분화를 위한 새로운 멀티모달 3D Volume Generative Adversarial Network (3D-vGAN) 모델을 제안합니다. 이 모델은 조건부 랜덤 필드(Conditional Random Field, CRF)와 V-net의 공간 특징 추출 능력을 결합하여 세밀한 세분화를 지원합니다. 결과적으로, 3D-vGAN은 BraTS-2018 데이터셋을 이용한 실험에서 U-net, GAN, FCN, 3D V-net과 같은 기존 세분화 모델들을 초과하는 성능을 보였습니다.

- **Technical Details**: 3D-vGAN 모델은 DCGAN 네트워크를 기반으로 하며, 입력으로는 뇌종양 MRI 이미지의 네 가지 서로 다른 모드를 사용합니다. 생성기(generator)는 V-net과 조건부 랜덤 필드를 결합하여 구성되며, 판별기(discriminator)는 다층 CNN으로 되어 있습니다. 이 네트워크는 Pseudo-3D 방법을 사용하여 3D 컨볼루션 대신 2D 필터와 1D 필터 구조를 통해 파라미터를 줄이고 비선형성을 향상시킵니다.

- **Performance Highlights**: 논문에서는 제안된 3D-vGAN 모델이 생성한 결과가 기존의 세분화 모델들과 비교하여 99.8% 이상의 특이도를 달성했다고 보고하고 있습니다. 이러한 결과는 뇌종양의 복잡한 형태의 경계 문제를 효과적으로 해결했음을 드러내며, 자동 및 정확한 MRI 이미지 분류의 진단 가치를 크게 향상시킵니다. 전체적으로, 3D-vGAN은 Brain MRI 세분화에서 뛰어난 성능을 보여주며 향후 연구를 위한 기초를 마련합니다.



### Adversarial Poisoning Attack on Quantum Machine Learning Models (https://arxiv.org/abs/2411.14412)
- **What's New**: 이번 연구는 양자 기계 학습(QML) 분야에서 데이터 오염 공격에 대한 첫 번째 평가를 진행하였습니다. 제안된 새로운 기법인 QUID(Quantum Indiscriminate Data Poisoning Attack)는 한정된 정보만으로도 퀀텀 모델의 무결성을 침해할 수 있는 공격 방법을 제시합니다. 연구 결과, QUID는 최대 92%의 정확도 저하를 초래하여 기존 방법에 비해 효과적인 대안을 제공합니다.

- **Technical Details**: QUID는 인코더 상태 유사성(intra-class encoder state similarity, ESS)을 이용하여 데이터 라벨을 변조하는 새로운 접근 방식을 채택합니다. 이는 훈련 데이터를 효과적으로 변조할 수 있는 방법을 제공하며, 노이즈가 있는 환경에서 라벨 플리핑(label-flipping) 공격을 완료할 수 있습니다. 데이터 오염 공격에 대한 전통적인 기법들은 훈련 절차에 대한 깊은 이해를 필요로 하지만, QUID는 이러한 요구를 피할 수 있습니다.

- **Performance Highlights**: QUID의 실험에서는 다양한 QNN 아키텍처와 데이터 세트에서 랜덤 라벨 플리핑과 비교하여 적어도 50% 이상의 정확도 저하를 보였습니다. 이러한 결과는 QUID의 효과성을 분명히 보여주며, 특히 노이즈가 많은 환경에서도 잘 작동합니다. 이는 QML 모델의 보안 구현에 있어 강력한 위협을 제시하는 데이터 오염 공격에 대한 새로운 대응책이 될 수 있음을 시사합니다.



### Enhancing Diagnostic Precision in Gastric Bleeding through Automated Lesion Segmentation: A Deep DuS-KFCM Approach (https://arxiv.org/abs/2411.14385)
- **What's New**: 본 연구는 내시경 영상에서 위 출혈을 정확히 분류하고 세분화하는 새로운 방법론인 Deep DuS-KFCM(이중 공간 커널화 제약 퍼지 C-평균) 클러스터링 알고리즘을 소개합니다. 이 하이브리드 신경 퍼지 시스템은 신경망(Neural Networks)과 퍼지 로직(Fuzzy Logic)을 결합하여 출혈 영역을 효율적이고 정확하게 식별합니다. 또한, 초기 coarse-to-fine 전략을 통해 세분화 정확도를 향상시킴으로써 생명 구조에 필수적인 역할을 할 수 있는 가능성을 제시합니다.

- **Technical Details**: Deep DuS-KFCM 방법론은 픽셀 값과 그 공간적 관계를 결합하여 위 출혈을 정확하게 식별하고 세분화하는 데 중점을 둡니다. Gray Level Co-occurrence Matrix (GLCM)를 활용해 질감 분석을 수행하고, 이를 통해 질환 조직과 건강한 조직 간의 미세한 차이를 구분할 수 있는 기능을 강화합니다. 또한, DeepLabv3+와 ResNet50 아키텍처를 사용하여 세분화 결과를 정교하게 다듬는 방법을 채택하여 단순한 세분화 성능을 넘어선 실질적인 임상 활용 가능성을 탐구합니다.

- **Performance Highlights**: Deep DuS-KFCM 모델은 일반적인 위 출혈 및 발적 데이터 세트를 통해 87.95%의 전례 없는 정확성과 96.33%의 특이도로 기존의 세분화 방법들을 초월하는 성능을 입증했습니다. 이 모델은 특히 미세한 출혈 증상을 식별하는 데 강력한 세분화 능력을 보이며, 노이즈에 대한 강건성을 강조하고 있습니다. 이 연구는 의료 이미지 처리 분야에서 중요한 발전을 이룬 것으로, 환자의 진단 및 치료 결과를 향상시키는 데 기여할 수 있습니다.



### Using Formal Models, Safety Shields and Certified Control to Validate AI-Based Train Systems (https://arxiv.org/abs/2411.14374)
Comments:
          In Proceedings FMAS2024, arXiv:2411.13215

- **What's New**: 이 논문은 자율 시스템(Object)의 인증에 대한 새로운 접근법을 모색하는 KI-LOK 프로젝트를 소개합니다. 기존의 인증 방법에 더해, 자율 기차(autonomous trains) 내 AI 구성 요소의 안전한 통합을 위한 기법을 개발하고 있습니다. 특히, B 방법을 사용하여 조종 시스템의 안전성을 보장하고, 런타임 인증 검사를 통한 인식 시스템의 신뢰성을 향상시키는 이중 접근 방식을 취하고 있습니다.

- **Technical Details**: 이 연구는 정형 모델(formal model)을 기반으로 한 시뮬레이션을 통해 조종 시스템의 안전성과 인식 시스템의 신뢰성을 연결하는 데 집중합니다. 실시간 런타임 인증 검사를 위한 데모 시스템이 실제 AI 출력과 실제 인증 검사기에 의해 제어되며, ProB라는 검증 도구(validation tool)에 통합됩니다. 이로 인해 런타임 모니터링(runtime monitoring), 런타임 검증(runtime verification), 통계적 검증(statistical validation)을 통한 정형 안전 속성(formal safety properties)의 검토가 가능해집니다.

- **Performance Highlights**: 연구의 적용 사례로 신호 탐지(signal detection) 문제를 다루었으며, 이를 통해 AI와 인증 검사기의 잠재적 취약점(vulnerabilities) 및 약점을 분석할 수 있는 방법을 제시합니다. 이러한 접근법을 통해 자율 시스템의 안전성을 높이고, 산업 및 과학 분야에서의 사용 가능성을 확대할 수 있습니다.



### InCrowd-VI: A Realistic Visual-Inertial Dataset for Evaluating SLAM in Indoor Pedestrian-Rich Spaces for Human Navigation (https://arxiv.org/abs/2411.14358)
Comments:
          18 pages, 7 figures, 5 tabels

- **What's New**: InCrowd-VI는 시각 장애인을 위한 실내 보행 환경 내 내비게이션을 위한 새로운 visual-inertial 데이터셋으로, 기존 데이터셋의 한계를 극복하고 현실적인 환경에서 수집된 데이터를 통해 SLAM 연구의 발전에 기여하고 있습니다. 이 데이터셋은 Meta Aria Project 안경을 사용하여 다양한 인구 밀도의 복잡한 환경에서 기록된 58개의 시퀀스를 포함하여 총 5km의 궤적 길이를 제공합니다.

- **Technical Details**: InCrowd-VI 데이터셋은 RGB 이미지, 스테레오 이미지 및 IMU 측정을 포함하며, 각 시퀀스에 대한 반밀집 3D 포인트 클라우드도 제공합니다. 데이터셋은 약 2cm의 정확도로 제공되는 Ground-truth 궤적을 포함하여 SLAM 알고리즘의 성능 평가를 위한 신뢰할 수 있는 기준을 가집니다.

- **Performance Highlights**: InCrowd-VI에서 최신 visual odometry 및 SLAM 알고리즘의 평가 결과, 복잡한 실내 환경에서의 성능 저하가 심각하다는 것이 드러났습니다. 이 연구는 시각 장애인을 위한 보다 견고한 SLAM 솔루션 개발의 필요성과 기회를 강조하며, 향후 연구의 주요 개선 점을 제시했습니다.



### Contrasting local and global modeling with machine learning and satellite data: A case study estimating tree canopy height in African savannas (https://arxiv.org/abs/2411.14354)
Comments:
          31 pages; 9 figures

- **What's New**: 이 연구는 위성 이미지를 활용한 기계 학습(SatML)의 최신 발전이 지역적 모델링 능력에 어떻게 영향을 미치는지를 밝혀내고자 하였습니다. 특히, 모잠비크 카링가니 자연 보호구역에서 나무 캐노피 높이(탑령) 매핑을 통해 지역과 글로벌 모델링 패러다임을 비교하였습니다. 연구 결과, 기존의 글로벌 모델보다 지역 자체 데이터를 기반으로 학습한 작은 모델이 더 우수한 성능을 보임을 발견했습니다.

- **Technical Details**: 연구는 총 세 가지 연구 질문(RQ1, RQ2, RQ3)을 중심으로 진행되었습니다. RQ1에서는 지역 예측을 위해 지역 데이터를 수집하는 것이 얼마나 중요한지, RQ2에서는 TCH 모델의 신뢰성에 가장 큰 영향을 미치는 설계 요인은 무엇인지, RQ3에서는 지역과 글로벌 모델링 노력 간의 충돌 또는 시너지가 어떤 것인지 살펴보았습니다. 결과적으로, 지역 데이터는 기계 학습 모델에서 정확하고 생태학적으로 관련성이 높은 예측을 생성하는 데 필수적임을 명확하게 보여주었습니다.

- **Performance Highlights**: 연구에서 제안하는 모델은 기존의 글로벌 TCH 맵의 평균 절대 오차를 40-69%까지 줄였고, 이는 생태학적 관련성 기준에 따라 분류된 측정치를 포함합니다. 또한, 글로벌 데이터로 사전 학습한 모델을 지역 데이터로 미세 조정하는 것보다 지역 데이터를 전적으로 사용하여 모델을 훈련시키는 것이 성능이 더 뛰어난 것으로 분석되었습니다. 이러한 결과는 지역과 글로벌 모니터링 노력 간의 효율적인 상호 작용에 대한 향후 연구 방향에 중요한 통찰을 제공합니다.



### Enhancing Medical Image Segmentation with Deep Learning and Diffusion Models (https://arxiv.org/abs/2411.14353)
- **What's New**: 이번 논문은 의료 영상(segmentation)의 중요성과 현재 딥 러닝 접근 방식의 한계를 강조합니다. 특히, 딥 러닝 모델들이 전문가의 주석(annotation)에 크게 의존하고 있으며, 작은 타겟을 정확히 분할하기 어려운 점에 대해 언급하고 있습니다. 저자들은 또한 디퓨전 모델(diffusion models)의 반복적인 잡음 제거 과정이 더 나은 세부 묘사를 가능하게 한다고 제안합니다.

- **Technical Details**: 의료 영상(segmentation)은 병변과 정상 조직 간의 낮은 대비와 불분명한 경계, 그리고 환자 간 큰 변동성 때문에 많은 도전에 직면해 있습니다. 동적 신경망(deep learning)은 분할 정확도와 효율성을 향상시켰지만, 의료 영상의 복잡성과 데이터 집합의 소규모 문제로 인해 여전히 한계가 있습니다. 이러한 문제를 극복하기 위한 디퓨전 모델의 가능성도 논의되고 있습니다.

- **Performance Highlights**: 이 논문은 의료 영상 분할의 새로운 대안을 제시하며, 디퓨전 모델이 작은 목표의 세밀한 경계를 추출하는 데 도움이 될 가능성을 탐구합니다. However, 전반적으로 현재의 방법론이 여전히 의료 영상의 정확한 분할과 세부 사항 유지에 대해 도전 과제가 많음을 보여주고 있습니다.



### Layer Pruning with Consensus: A Triple-Win Solution (https://arxiv.org/abs/2411.14345)
- **What's New**: 본 논문에서는 다중 유사성 메트릭을 결합하여 레이어의 중요도를 평가하고, 이를 통해 'Consensus criterion'이라는 새로운 방법을 제안합니다. 이 접근 방식을 통해 모델의 정확도 저하를 최소화하면서도 성능을 극대화하고, 적대적 공격에 대한 내성을 향상시키는 방안을 모색합니다. 응용 측면에서, 이 방법은 에너지 소비와 탄소 배출을 각각 66.99% 및 68.75%까지 줄일 수 있는 잠재력을 가지고 있습니다.

- **Technical Details**: 'Consensus' 기법은 기존의 CKA 기반 방법과 달리, 여러 유사성 메트릭을 통합하여 레이어의 중요도를 평가합니다. 이를 통해 단일 기준에 의존할 때 발생할 수 있는 문제점을 해결하고, 다양한 적대적 상황에서도 더 좋은 성능을 제공합니다. 논문에서는 CIFAR-10과 ImageNet 데이터셋을 활용하여, 최대 78.80%의 FLOPs 저감과 동시에 낮은 정확도 저하를 달성했습니다.

- **Performance Highlights**: 다중 유사성 메트릭을 통합하여 레이어를 가지치기하는 방법은 모델의 정확도를 유지하면서도 성능을 개선하는데 강력한 효과를 보입니다. 여러 적대적 벤치마크에서 우수한 성능을 나타내었으며, 특히 CIFAR-10과 ImageNet 데이터셋에서 state-of-the-art 성능을 기록했습니다. 따라서, 저자들이 제시한 방식은 저비용, 고성능, 강력한 내성을 지닌 모델을 구현하는데 기여할 수 있습니다.



### SplatR : Experience Goal Visual Rearrangement with 3D Gaussian Splatting and Dense Feature Matching (https://arxiv.org/abs/2411.14322)
- **What's New**: 본 연구에서는 Embodied AI의 기초적 문제인 Experience Goal Visual Rearrangement 작업을 위한 새로운 프레임워크를 제안합니다. 이는 3D Gaussian Splatting을 사용하여 장면을 표현하며, 빠르고 고품질의 사진같은 뷰를 렌더링할 수 있습니다. 에이전트는 목표 상태와 현재 상태를 비교할 수 있는 일관된 뷰를 유지하여 작업을 더욱 효과적으로 수행할 수 있습니다.

- **Technical Details**: 이 방식에서는 3D Gaussian Splatting을 기반으로 한 볼륨 표현이 사용되며, 이는 실시간으로 장면을 빠르게 렌더링할 수 있도록 해줍니다. 또한, Dense Feature Matching 방법을 통해 기초 모델에서 추출한 시각적 특성을 활용하여 이미지의 픽셀 단위 비교가 아닌 패치 수준에서 비교합니다. 이를 통해 에이전트는 다양한 환경 설정에서 효과적으로 장면의 변화를 인식할 수 있게 됩니다.

- **Performance Highlights**: AI2-THOR Rearrangement Challenge 벤치마크에서 제안된 방법의 성능을 검증하였으며, 현재의 최첨단 기술들보다 향상된 결과를 보여주었습니다. 본 연구는 복잡한 장면을 다루는 Embodied AI의 발전을 이끄는 중요한 기여를 할 것으로 기대됩니다. 따라서, 후속 연구와 AI 에이전트의 실제 환경 적용에 긍정적인 영향을 미칠 것으로 보입니다.



### Guided MRI Reconstruction via Schr\"odinger Bridg (https://arxiv.org/abs/2411.14269)
- **What's New**: 이 연구에서는 슈뢰딩거 다리(Schrödinger Bridge, SB)를 기반으로 하는 다중 대비 이미지 가이드 재구성 프레임워크를 제안합니다. 이 방법은 가이드 이미지와 타겟 이미지 간의 확산 다리를 설정하여 가이드 이미지를 이용한 데이터 일관성(consistency) 기반 샘플링으로 타겟 이미지를 더 정확하게 재구성합니다. 새로운 코어 컴포넌트로는 이미지 편집 분야에서 응용된 역전략($\mathbf{I}^2$SB-inversion)이 포함되어 있어 이미지 간 구조적 차이를 효과적으로 해결할 수 있습니다.

- **Technical Details**: 연구에서는 확산 모델의 비선형 확장인 SB를 사용하여 다중 대비 이미지를 재구성하는 방식을 소개합니다. 특정 샘플링에 대한 데이터 일관성을 유지하면서 가이드를 사용하여 고충실도(target image) 이미지로 변환하는 확산 경로를 설정합니다. $\mathbf{I}^2$SB-inversion 전략을 통해 가이딩 변수의 정확한 확인이 가능하여, 다중 대비 이미지들 간의 미세한 세부 정보 차이를 보정하여 재구성 성능을 향상시킵니다.

- **Performance Highlights**: 실험 결과에서는 T1와 T2-FLAIR 데이터셋에서 $\mathbf{I}^2$SB-inversion이 최대 14.4배의 가속률을 달성하며, 기존 재구성 방법에 비해 뛰어난 재구성 정확도와 안정성을 보였습니다. 이러한 성과는 다중 대비 이미지를 활용한 재구성에서 기존의 방식들보다 현저히 개선된 결과를 나타냅니다.



### CP-UNet: Contour-based Probabilistic Model for Medical Ultrasound Images Segmentation (https://arxiv.org/abs/2411.14250)
Comments:
          4 pages, 4 figures, 2 tables;For icassp2025

- **What's New**: 이번 연구는 초음파 이미지에서 병변을 감지하기 위한 새로운 깊이 학습 기반의 분할 방법인 CP-UNet을 제안합니다. 이 모델은 외곽선에 대한 집중을 강화하여 세분화 네트워크의 성능을 향상시키도록 설계되었습니다. 또한, Gaussian Mixture Model을 사용하여 병변 경계의 불확실성을 포착하는 등의 새로운 접근 방식을 도입했습니다.

- **Technical Details**: CP-UNet은 여러 다중 그룹 채널 이동 다운샘플링(MgCSD) 모듈과 외곽선 확률 모델링(CPM) 모듈을 통합하여 원본 이미지의 외곽선 특징을 혼합 가우스 분포로 적합시킵니다. GF 모듈은 같은 차원의 업샘플링 결과와 다운샘플링 결과를 융합하여 최종 세분화 결과를 생성합니다. 이러한 아키텍처는 전체 단계를 아우르는 정보 전송의 동작을 최적화하는 데 중점을 둡니다.

- **Performance Highlights**: BUSI, DDTI 및 개인 thyroid 초음파 이미지 데이터세트에서의 광범위한 실험을 통해 CP-UNet이 유방 및 갑상선 병변 세분화에서 기존의 최첨단 방법보다 높은 정확도를 보이는 것을 입증하였습니다. 이 연구는 현재의 의료 이미지 세분화 분야에서 강력한 기준선과 상대적으로 개선된 성능을 보여줍니다.



### AnywhereDoor: Multi-Target Backdoor Attacks on Object Detection (https://arxiv.org/abs/2411.14243)
- **What's New**: 본 논문에서는 AnytimeDoor라는 유연한 백도어 공격(Vulnerability in backdoor attack)을 소개합니다. 이 방법은 객체 탐지(Object Detection) 모델에 적용되어 공격자가 실시간으로 다양한 공격 유형(예: 객체 사라짐, 생성, 잘못 분류)을 수행할 수 있게 합니다. AnywhereDoor는 기존의 정적이고 제한된 공격 시나리오의 한계를 극복하며 새로운 공격 가능성을 제시합니다.

- **Technical Details**: AnywhereDoor는 목표 분리(Objective Disentanglement), 트리거 모자이킹(Trigger Mosaicking), 전략적 배칭(Stragegic Batching) 세 가지 혁신의 중심으로 설계되었습니다. 목표 분리는 공격의 조합 복잡성을 줄이며, 트리거 모자이킹은 부분적으로 처리된 이미지에서도 트리거 효과를 유지할 수 있도록 보장합니다. 마지막으로, 전략적 배칭은 객체 클래스 간의 불균형을 해결해 공격 효과성을 높이는 역할을 합니다.

- **Performance Highlights**: 광범위한 실험을 통해 AnywhereDoor는 5가지 공격 시나리오에서 80% 이상의 공격 성공률(Attack Success Rate, ASR)을 달성했습니다. 특히 가장 복잡한 시나리오에서는 기존 방법 대비 약 80%의 ASR 향상을 기록했습니다. 더욱이, 깨끗한 샘플에 대한 성능 저하 없이 뛰어난 공격 효율성을 보여주었습니다.



### Revised Regularization for Efficient Continual Learning through Correlation-Based Parameter Update in Bayesian Neural Networks (https://arxiv.org/abs/2411.14202)
Comments:
          at ICVGIP 2024

- **What's New**: 이 연구는 Variational Inference를 기반으로 한 베이지안 신경망의 지속적 학습 알고리즘을 제안하여 기존 방법의 여러 단점을 극복하고자 합니다. 특히, 지속적 학습 상황에서 각 단계에서 네트워크 매개변수를 저장하여 지식을 유지하는 과정이 도전적이라는 점을 강조합니다. 연구진은 매개변수의 저장 소요를 줄이기 위한 방법과 KL 발산의 이점을 유지하고 관련된 문제를 해결하는 정규화 항을 도입했습니다.

- **Technical Details**: 제안된 방법은 매개변수의 평균(mean) 및 분산(variance)의 동역학을 목표로 하는 정규화 항을 도입하여, 과거 데이터에 대한 접근이 제한된 상황에서도 네트워크 매개변수와 데이터 간의 적절한 대응을 보장합니다. 이 방법은 매개변수 공간을 공통 및 독특한 하위공간으로 분리하여 지식 전이(knowledge transfer)를 향상시킵니다. 이러한 접근은 효과적인 역지식 전이(backward knowledge transfer)를 보장하도록 조건을 설정합니다.

- **Performance Highlights**: 실험 결과는 제안된 방법이 다양한 데이터셋과 순차적인 데이터셋 조합에서 구동되는 성능이 기존 방법보다 우수함을 보여줍니다. 기억 기반, 정규화 기반, 그리고 경량 기반 접근 방식에 대한 비교 결과, 우리의 접근법이 매개변수와 데이터 간의 관계를 활용해 더 나은 성능을 도출한다고 결론지었습니다.



### Deep Learning Approach for Enhancing Oral Squamous Cell Carcinoma with LIME Explainable AI Techniqu (https://arxiv.org/abs/2411.14184)
Comments:
          Under Review at an IEEE conference

- **What's New**: 본 연구는 Oral Squamous Cell Carcinoma (OSCC) 진단 성능을 향상시키기 위해 심층 학습 모델을 활용한 연구이다. 총 5,192개의 이미지를 사용하여 모델을 학습시키며, EfficientNetB3가 98.33%의 정확도를 달성하여 가장 우수한 성능을 보였다. 또한, Local Interpretable Model-agnostic Explanations (LIME) 기법을 사용하여 모델의 결정 과정을 설명함으로써 결과의 신뢰성을 높였다.

- **Technical Details**: 연구에 사용된 데이터셋은 kaggle에서 제공되는 Histopathological Imaging Database로, Normal과 OSCC 카테고리로 구분된 이미지가 포함되어 있다. ResNet101, DenseNet121, VGG16, EfficientNetB3 네 가지 심층학습 아키텍처가 평가되었으며, 특히 EfficientNetB3는 다른 모델에 비해 적은 연산 자원으로도 고도의 정확도를 달성했다. 데이터 전처리 과정으로는 이미지 크기 조정, JPG 변환 및 다양한 증강 기법이 포함되어 있으며, 이는 최종 모델 학습에 일관성을 제공한다.

- **Performance Highlights**: EfficientNetB3는 98.33%의 높은 분류 정확도를 기록하였으며, F1 점수는 0.9844로 평가되었다. DenseNet121 모델은 90.24%의 정확도와 90.45%의 F1 점수를 보였다. 이 연구는 OSCC 조기 진단에 있어 효율적인 AI 도구의 잠재력을 제시하였으며, 임상적 응용 가능성을 위한 중요한 기초 자료를 제공하였다.



### Creating a Formally Verified Neural Network for Autonomous Navigation: An Experience Repor (https://arxiv.org/abs/2411.14163)
Comments:
          In Proceedings FMAS2024, arXiv:2411.13215

- **What's New**: 이 논문은 자율주행 차량에서 신경망(neural networks)의 검증(verification)에 대한 새로운 접근 방식을 소개합니다. 특히, 커스텀 데이터셋을 활용하여 비전 기반 자율 항법(autonomous navigation)을 위한 신경망의 설계 및 훈련을 탐구한 사례를 보고합니다. 기계 학습과 미분 가능한 논리(differentiable logics)의 결합을 통해 기본 안전 속성(safety properties)을 충족하는 네트워크를 설계하는 것에 중점을 두고 있습니다.

- **Technical Details**: 이 연구에서는 자율주행 시스템에 적합한 신경망 검증기(neural network verifier)를 선택하고 그 이유에 대해 설명합니다. 또한, 신경망 훈련 후 행동을 보장하는데 필요한 안전 속성이 포함된 네트워크의 설계에 대해 논의합니다. 연구에서 사용된 방법론과 특정 데이터셋의 특성도 중요한 요소로 언급됩니다.

- **Performance Highlights**: 자율주행 시스템을 위한 검증 접근법이 효과적으로 구현되었음을 관찰하였고, 이는 훈련 후 신경망의 신뢰성을 높이는 데 기여할 것으로 기대됩니다. 이 연구는 신경망 검증기의 사용이 자율주행 시스템의 개발에 필수적이라는 점을 강조하며, 안전성을 보장하는데 있어 중요한 기여를 합니다.



### Differentiable SVD based on Moore-Penrose Pseudoinverse for Inverse Imaging Problems (https://arxiv.org/abs/2411.14141)
Comments:
          11 pages

- **What's New**: 본 논문에서는 중복된 특이값 발생 시 단점이 있는 비가역적인 특이값 분해(SVD)를 해결하기 위해 무어-펜로즈 유사역행렬을 기반으로 한 미분 가능한 SVD를 제안합니다. 이는 미분 가능성의 빈약함이라는 문제를 다루는 최초의 연구로, 비고유 시스템에서 발생하는 문제를 수학적으로 분석하였습니다.

- **Technical Details**: 논문에서는 저차원 정규화(low-rank regularization) 기반의 깊은 언롤링 네트워크(deep unrolling networks, DUNs)에 대한 미분 가능한 SVD의 필요성을 강조합니다. 특히, SVD의 비미분 가능성이 반복된 특이값으로 인해 발생하며, 이에 대한 해답으로 무어-펜로즈 유사역행렬을 효과적으로 활용합니다.

- **Performance Highlights**: 종합적인 실험 결과는 제안된 미분 가능한 SVD가 색상 이미지 압축 감지 및 동적 MRI 재구성 문제에서 수치적 불안정성을 효과적으로 극복하며 높은 정확성을 유지함을 보여줍니다. 최첨단 저차원 정규화 기반 DUN을 기준 모델로 설정하여, 기존 방법들보다 우수한 성능을 입증했습니다.



### GASP: Efficient Black-Box Generation of Adversarial Suffixes for Jailbreaking LLMs (https://arxiv.org/abs/2411.14133)
Comments:
          28 pages, 9 tables, 13 figures; under review at CVPR '25

- **What's New**: 이번 논문에서는 Generative Adversarial Suffix Prompter (GASP)라는 새로운 프레임워크를 소개합니다. GASP는 인간이 읽을 수 있는 프롬프트 생성과 Latent Bayesian Optimization (LBO)을 결합하여 사이버 공격(즉, jailbreak attack)을 통해 자연스러운 프롬프트를 생성할 수 있게 합니다. 이 프레임워크는 기존의 최적화 기반 방법의 한계를 극복하며, 자동으로 적대적인 서픽스(adversarial suffix)를 생성하여 LLM의 안전 장치를 피하는 데 주안점을 두고 있습니다.

- **Technical Details**: GASP는 Latent Bayesian Optimization을 활용하여 연속적인 임베딩 공간(embedding space)에서 적대적인 서픽스를 탐색하며, 이 과정에서 계산 비용을 크게 줄입니다. 프롬프트의 일관성을 유지하기 위해 목표 지향적인 반복적 개선(iterative refinement)을 통해 공격의 효율성을 극대화하는 전략을 사용합니다. 또한, Odds-Ratio Preference Optimization (ORPO)을 통해 모델의 파라미터를 조정하여 높은 성공률을 보이는 서픽스의 효과를 증가시키면서도 가독성을 유지하도록 설계되었습니다.

- **Performance Highlights**: GASP의 실험 결과는 자연스러운 jailbreak 프롬프트를 생성할 수 있음을 입증했으며, 공격 성공률이 유의미하게 향상되고, 교육 시간이 단축되며, 추론 속도가 개선되었습니다. 특히, GASP는 블랙박스 환경에서 운영될 때도 강력한 성능을 발휘하며, 다양한 LLM 아키텍처 및 프롬프트 유형에 대해 효율적인 평가를 지원하고 있습니다. 이러한 성과는 LLM의 보안을 높이고, 더 안전한 모델 개발에 기여할 것으로 기대됩니다.



### MetaCropFollow: Few-Shot Adaptation with Meta-Learning for Under-Canopy Navigation (https://arxiv.org/abs/2411.14092)
- **What's New**: 이 논문은 농업 로봇의 자율 항법에서 도메인 이동 문제를 해결하기 위해 메타 학습(Meta-Learning) 접근 방식을 탐구합니다. 농작물 사이의 좁은 간격, GPS 신호 왜곡, 과도한 혼잡 등 다양한 도전 과제가 존재하는 상황에서 기존 시스템 대비 높은 적응성을 갖춘 베이스 러너(base-learner)를 교육시킵니다. 새로운 환경에 신속하게 적응할 수 있는 능력을 부각시키며, 한정된 데이터로도 강력한 내비게이션 성능을 보여줍니다.

- **Technical Details**: 연구에서는 28273개의 이미지를 사용하여 MAML(Model-Agnostic Meta-Learning) 기반의 학습 시스템을 구현하였습니다. 주요 키포인트는 사라지는 점(vanishing point), x축과의 왼쪽 교차점 및 오른쪽 교차점으로 구성되어, 이를 통해 농작물 사이의 통과 가능 영역을 정의합니다. MAML 및 그 변형(MAML++와 ANIL)은 환경에 대한 빠른 적응 능력을 평가하는 데 사용되었으며, 이는 표준 학습 접근법보다 뛰어난 성능을 입증하였습니다.

- **Performance Highlights**: 실험 평가 결과, MAML++ 시스템은 다른 시스템들에 비해 적응력이 뛰어난 것으로 나타났습니다. 예를 들어, All-Season 분할에 대해 MAML++는 평균 3.9의 손실(loss)을 기록하여 비슷한 조건의 Non-MAML 모델(5.75)보다 현저히 낮은 결과를 보였습니다. ANIL++도 4.3의 손실로 이와 유사하게 성능을 개선하였으며, 이러한 결과는 농업 환경에 대한 내비게이션 성능이 도메인 이동 대응력에 크게 의존함을 시사합니다.



### Self-supervised learning for radio-astronomy source classification: a benchmark (https://arxiv.org/abs/2411.14078)
- **What's New**: 이번 연구에서 제안된 Self-Supervised Learning (SSL) 기법은 전통적인 컴퓨터 비전 모델들이 적절하게 적용되지 않는 라디오 천문 이미지 분석에 특히 유용하다는 점을 강조합니다. 기존의 자연 이미지에서 미리 학습된 모델과 비교하여 SSL 훈련된 모델이 몇 가지 하위 작업에서 상당한 개선을 보임을 보여주었습니다. 또한, 다수의 라디오 천문 데이터셋에 걸쳐 SSL 기법의 성능 평가를 통해, SSL이 데이터 분석 효율성을 높일 수 있는 잠재력을 지니고 있음을 확인했습니다.

- **Technical Details**: 연구에서는 SimCLR, BYOL, DINO 등 다양한 최첨단 SSL 기법을 사용하여 라디오 천문 데이터의 성능을 분석합니다. 데이터 커링(data curation) 작업도 SSL의 효과성에 큰 영향을 미친다는 결과를 제시하였습니다. 연구는 여러 라디오 천문학적 데이터셋에서 소스 분류 작업을 수행하면서, 커링된 데이터와 커링되지 않은 데이터를 비교분석하였습니다.

- **Performance Highlights**: SSL로 훈련된 모델은 모든 데이터셋에서 ImageNet에서 미리 훈련된 기준선보다 일관되게 뛰어난 성능을 보여주었습니다. 특히, 전체 backbone이 미세 조정(fine-tuning) 될 때 SSL의 이점이 덜 뚜렷해지긴 하나 여전히 사전 훈련을 초과하는 성능을 기록했습니다. 이러한 결과는 라디오 천문학 데이터 분석 효율성을 높이는 데 SSL이 중요한 역할을 할 수 있음을 시사합니다.



### Out-Of-Distribution Detection with Diversification (Provably) (https://arxiv.org/abs/2411.14049)
- **What's New**: 이 논문은 머신러닝 모델의 신뢰할 수 있는 배치를 보장하기 위해 중요한 OOD(out-of-distribution) 탐지 문제를 다룬다. 최근 연구들은 웹 데이터나 다른 데이터셋과 같은 보조적인 원거리(outlier) 데이터를 이용해 훈련하는 방법에 집중하고 있으나 기존 방법이 알 수 없는 OOD 데이터에 대한 탐지 능력을 일반화하는 데 여전히 어려움을 겪고 있음을 밝혔다. 본 연구에서는 보조 원거리 데이터의 다양성이 OOD 탐지 능력을 향상시키는 데 필수적임을 보여준다.

- **Technical Details**: 우리는 OOD 탐지와 관련된 이론적 분석을 수행하고, 보조 원거리 데이터 간의 분포 이동이 OOD 탐지기능의 일반화 능력에 미치는 영향을 설명한다. 이러한 이론을 바탕으로 보다 다양한 보조 원거리 집합이 분포 이동 오류를 줄이고, OOD 탐지 오류의 상한선을 낮출 수 있음을 입증했다. 제안된 방법인 Diversity-induced Mixup(diverseMix)은 원거리 데이터의 다양성을 높여 OOD 탐지 성능을 향상시키기 위한 간단하면서도 효과적인 접근법을 제공한다.

- **Performance Highlights**: 다수의 실험 결과, diverseMix는 CIFAR-10 및 CIFAR-100 데이터셋에서 각각 24.4% 및 43.8%의 FPR95(거짓 긍정 비율) 개선을 달성하며 기존 방법을 초월하는 성능을 보였다. 이러한 결과는 보조 원거리의 다양성이 OOD 탐지 능력을 강화하는 데 필수적임을 다시 한번 확인시켜준다. 본 연구는 OOD 탐지 분야에서 강력한 이론적 기반과 함께 실용적인 방법론을 제공하며, 최신 대규모 기준에서도 최신 성과를 이룩했다.



### Automatic brain tumor segmentation in 2D intra-operative ultrasound images using MRI tumor annotations (https://arxiv.org/abs/2411.14017)
Comments:
          19, 8 figures, submitted to International Journal of Computer Assisted Radiology and Surgery

- **What's New**: 이번 논문은 수술 중 초음파(iUS) 이미지를 활용한 뇌종양 자동 분할의 새로운 접근법을 제시합니다. 이 연구는 사전 수술 MRI 이미지의 종양 주석을 사용하여 iUS 이미지에서 뇌종양 분할을 위한 딥러닝 모델을 훈련할 수 있음을 보여줍니다. 특히 주목할 점은, 기존의 iUS 이미지보다 MRI 이미지의 주석이 쉽게 접근 가능하다는 점입니다. 이로 인해 자동 분할 모델의 성능을 향상시킬 수 있는 가능성이 엿보입니다.

- **Technical Details**: 본 연구는 180개의 주석이 달린 사전 수술 MRI 이미지와 이에 대응하는 주석이 없는 iUS 이미지를 사용했습니다. nnU-Net 프레임워크를 통해 모델을 훈련하기 전에 이미지 등록(image registration)을 통해 MRI 주석을 iUS 이미지에 맞춰 조정하였습니다. MRI와 US 주석이 모두 포함된 모델과 오직 US 주석만 포함된 모델을 비교하여 MRI 주석의 활용 가능성을 검토했습니다.

- **Performance Highlights**: 모델의 성능은 MRI 주석만 훈련된 모델이 US 주석만 훈련된 모델과 유사하다는 것을 보여주었습니다. 두 가지 모드로 훈련된 모델은 평균 Dice 점수 0.62를 기록했으며, 외부 전문가 주석은 0.67로 평가되었습니다. 다만, 현재 모델은 200 mm² 이하의 작은 종양에 대해서는 성능이 저조한 것으로 나타났으며, 이는 향후 연구의 초점이 될 것입니다.



### Experimental comparison of graph-based approximate nearest neighbor search algorithms on edge devices (https://arxiv.org/abs/2411.14006)
- **What's New**: 본 논문에서는 실시간 최근접 이웃 검색 응용을 위한 다양한 그래프 기반의 근사 최근접 이웃(ANN) 검색 알고리즘을 엣지 디바이스에서 비교 실험한 내용을 소개합니다. 기존의 연구가 단일 스레드 구현에 국한된 반면, 이 연구는 엣지 디바이스의 전반적인 컴퓨팅 및 저장 능력을 활용하여 새로운 벡터의 삽입 및 삭제 지연 시간과 전력 소비와 같은 추가적인 메트릭스를 포함시켰습니다. 이를 통해 이러한 알고리즘들이 엣지 기반 실시간 추적 시스템에 얼마나 적합한지를 평가하고자 합니다.

- **Technical Details**: 근사 최근접 이웃 검색(ANNS)은 데이터를 처리해야 하는 양이 급격히 증가함에 따라 더욱 중요해졌습니다. ANNS는 주어진 쿼리 벡터와 가장 유사한 k개의 벡터를 찾아내는 k-최근접 이웃 검색(k-NNS) 문제를 해결하는 데 핵심적인 역할을 합니다. 이 연구에서는 헬싱키의 Conveqs 및 Aalto University에서 제공한 거리 카메라 데이터를 기반으로 한 ANNS 알고리즘의 실험적 평가를 수행하여 엣지 디바이스와 서버 간 성능 차이를 분석합니다.

- **Performance Highlights**: 연구 결과는 더욱 강력한 엣지 디바이스가 항상 더 효율적인 알고리즘 실행을 보장하지 않으며, 비용 효과적인 디바이스도 비싼 디바이스와 유사한 성능을 달성할 수 있음을 보여줍니다. 특히, 그래프 기반 방법들은 높은 재현율과 더불어 빠른 검색 시간을 제공하여 다양한 실제 응용 분야에서 유용하게 사용될 수 있습니다. 연구에 포함된 알고리즘은 실시간 성능 최적화를 목표로 다양한 메트릭스로 평가되었습니다.



### Image Compression Using Novel View Synthesis Priors (https://arxiv.org/abs/2411.13862)
Comments:
          Preprint submitted to Ocean Engineering

- **What's New**: 이번 연구에서는 원거리 무인 차량의 테더리스(tetherless) 조작을 위한 실시간 시각적 피드백의 중요성을 강조하고 있습니다. 기존의 수중 통신 방법인 acoustic communication이 제한된 대역폭으로 인해 이미지나 비디오를 실시간으로 전송하기 어렵다는 문제를 해결하기 위해, 사전 임무 정보(prior mission information)를 활용한 모델 기반 이미지 압축 기법을 제안합니다.

- **Technical Details**: 제안된 기법은 기계 학습(machine learning) 기반의 새로운 뷰 합성(view synthesis) 모델을 이용하며, 경량 표현(latent representations)을 정제하기 위해 그래디언트 하강 최적화(gradient descent optimization)를 사용합니다. 이 과정은 카메라 이미지와 렌더링(rendered) 이미지 간의 압축 가능한 차이를 생성하는 데 기여합니다.

- **Performance Highlights**: 인공 해양 수조에서 얻은 데이터셋을 사용하여 제안된 압축 기법의 성능을 평가한 결과, 기존 기법들에 비해 우수한 압축 비율(compression ratios)과 이미지 품질(image quality)을 보여주었습니다. 또한, 이 방법은 장면(scene) 내 새로운 물체의 도입에 대한 강건함(robustness)을 나타내어, 테더리스 원격 운영 차량의 작업을 향상시킬 가능성을 강조하고 있습니다.



### A Multimodal Approach to The Detection and Classification of Skin Diseases (https://arxiv.org/abs/2411.13855)
- **What's New**: 본 연구는 피부 질환 분류를 위한 새로운 데이터셋을 도입하고, 26종의 피부 질환 및 37,000개의 이미지와 환자의 서사를 포함하고 있습니다. 기존의 방법들보다 더 많은 데이터와 최신 기술을 활용하여 피부 질환 진단의 정확성을 높이는데 중점을 두고 있습니다. ResNet-50 모델의 초기 정확도 70%에서 최적화 후 80%로 개선하였고, 동시에 새로운 LLM 최적화 전략인 Chain of Options를 제안하여 정밀도를 높였습니다.

- **Technical Details**: 이 연구는 피부 질환을 분류하기 위해 이미지 데이터와 환자의 서술 정보를 활용하는 멀티모달 접근 방식을 채택했습니다. Chain of Options를 통해 복잡한 추론 작업을 훈련 단계에서 중간 단계로 나누어 모델의 효율성을 높이고, LLM(대형 언어 모델)의 새롭고 효과적인 미세 조정 방법을 제공했습니다. 이러한 접근법을 통해 이미지 모델과 초기 질병 추천을 활용하여 정확도를 91%로 향상시켰습니다.

- **Performance Highlights**: 제안된 방법은 기존의 이미지 데이터만 사용하는 단독 방법보다 11% 높은 91%의 정확도를 기록했습니다. 이는 피부 질환 진단에서 이미지와 텍스트 데이터를 결합한 최초의 접근 방식으로, 진단의 효율성과 정확성을 획기적으로 개선했습니다. 추가로, 새로운 멀티모달 피부 질환 데이터셋이 기존의 벤치마크보다 객관적으로 더 어려운 문제를 다루고 있음을 보여준 중요한 연구 결과입니다.



### Learning to Reason Iteratively and Parallelly for Complex Visual Reasoning Scenarios (https://arxiv.org/abs/2411.13754)
Comments:
          NeurIPS 2024 camera ready; source code to be released at: this https URL

- **What's New**: 이번 논문에서는 복잡한 시각적 추론 및 질문 응답(Visual Question Answering, VQA) 문제를 해결하기 위해 Iterative and Parallel Reasoning Mechanism (IPRM)을 소개하고 있습니다. IPRM은 반복적(iterative)이며 병렬적(parallel)인 두 가지 계산 방식을 결합하여 복잡한 VQA 시나리오를 보다 효과적으로 처리합니다. 본 연구는 IPRM이 다양한 이미지 및 비디오 VQA 벤치마크에서 이전의 특정 작업 방법과 변환기(transformer) 기반 주의(attention) 모듈을 뛰어넘는 성과를 보였음을 강조합니다.

- **Technical Details**: IPRM은 경량화되고 완전 미분 가능한(neural module) 신경 아키텍처로, 변환기 및 비 변환기 비전-언어(backbones)에 편리하게 적용될 수 있습니다. IPRM은 언어 정보에 기반하여 새로운 병렬(operation states) 작업 세트를 형성하고, 이 작업을 시각 정보와 결합하여 실행하는 방식으로 작동합니다. 내부적으로 IPRM은 기억(memory)의 상태를 유지하며 각각의 반복(iterative) 단계에서 과거의 작업 상태 및 결과를 통합하여 처리합니다.

- **Performance Highlights**: IPRM은 복잡한 시각적 추론 작업에서 기존 모델들보다 더 우수한 성능을 입증하였으며, 특히 AGQA, STAR, CLEVR-Humans 및 CLEVRER-Humans와 같은 벤치마크에서 두드러진 성과를 나타냅니다. 또한, IPRM의 내부 계산 과정을 시각화할 수 있어, 복잡한 추론 시나리오에서 모델의 동작을 보다 명확히 이해하고 오류를 진단하는 데 도움을 줍니다.



### Bimanual Dexterity for Complex Tasks (https://arxiv.org/abs/2411.13677)
Comments:
          In CoRL 2024. Website at this https URL

- **What's New**: Bidex라는 새로운 탈포현(teleoperation) 시스템이 소개되었습니다. 이 시스템은 매우 정밀하고 비용이 저렴하며, 낮은 지연(latency)을 갖춘 이중(두 개의) 손과 팔을 제어할 수 있는 기기를 제공합니다. 기존의 VR 기술과 비교하여 Bidex는 더 복잡한 작업에서도 더 빠르고 높은 품질의 데이터를 생성하는 것으로 나타났습니다.

- **Technical Details**: Bidex는 착용자가 두 개의 모션 캡쳐(glove) 장갑을 착용하고 자연스럽게 움직여 일상적인 작업을 수행할 수 있도록 설계되었습니다. 이 시스템은 Manus Meta glove를 사용하여 정확한 손가락 추적을 가능하게 하며, GELLO에서 영감을 받아 개발된 팔 추적 시스템은 로봇 팔의 관절 각도와 위치를 정확히 추적합니다. 이는 기존 기술이 매우 높은 정확도와 낮은 비용으로 이뤄지는 것을 가능하게 합니다.

- **Performance Highlights**: Bidex 시스템은 VR 헤드셋과 SteamVR 시스템보다 특히 데이터 수집의 속도와 정확도에서 뛰어난 성능을 보여주었습니다. 이를 통해 다양한 로봇 팔과 함께 사용될 수 있으며, 기존의 복잡한 로봇 작업을 더 쉽게 수행할 수 있도록 돕습니다. 실험 결과와 설정 재현을 위한 비디오는 Bidex 웹사이트에 공개되었습니다.



### A Deep Learning Approach to Predict the Fall [of Price] of Cryptocurrency Long Before its Actual Fa (https://arxiv.org/abs/2411.13615)
Comments:
          22 pages, 3 figures

- **What's New**: 이 연구는 암호화폐 시장의 리스크 요인을 예측하는 새로운 방법론을 제시합니다. 기존 모델에 비해 향상된 성능을 보여주는 새로운 모델을 개발하였으며, 이는 투자자들이 보다 안전하게 거래를 할 수 있도록 돕습니다. 연구에 사용된 다양한 머신러닝 알고리즘(CNN, LSTM, BiLSTM, GRU)은 이 시장의 변동성을 보다 정확하게 평가하는 데 기여하였습니다.

- **Technical Details**: 연구에서는 암호화폐 시장의 20개 요소에서 리스크 요인인 '변동성'(volatility)을 계산하고 예측하기 위해 여러 머신러닝 알고리즘을 적용하였습니다. 제안된 모델은 RMSE(Root Mean Square Error) 값이 1.3229에서 0.0089로 감소했으며, 기존 모델에 비해 훨씬 우수한 성능을 보였습니다. 이렇게 구축된 모델은 투자자들에게 비트코인(Bitcoin), 이더리움(Ethereum), 도지코인(Dogecoin) 등 복잡한 금융 자산에 대한 거래를 쉽게 할 수 있게 해줍니다.

- **Performance Highlights**: 제안된 모델의 성능은 기존 모델들과 비교할 때 매우 뛰어난 결과를 보였습니다. 기존 모델들에서 RMSE가 14.5092부터 0.02769까지 범위인 것에 비해, 새로운 모델은 훨씬 낮은 RMSE 값을 기록하여 더 나은 결과를 달성했습니다. 이로 인해, 암호화폐 시장에 투자하는 사람들이 보다 안정적으로 거래를 진행할 수 있을 것입니다.



### Large-scale cross-modality pretrained model enhances cardiovascular state estimation and cardiomyopathy detection from electrocardiograms: An AI system development and multi-center validation study (https://arxiv.org/abs/2411.13602)
Comments:
          23 pages, 8 figures

- **What's New**: 이 연구에서는 CardiacNets라는 혁신적인 모델을 제안하여 심전도(ECG) 분석을 개선하고 심장 자기 공명 영상(CMR)의 진단 강점을 활용합니다. CardiacNets는 크로스 모달 대조 학습(cross-modal contrastive learning)과 생성적 사전 훈련(generative pretraining)을 통해 ECG 입력을 사용하여 심장 기능 지표를 평가하고 잠재적인 심혈관 질환(CVD)을 탐색합니다. 이 모델은 또한 ECG 데이터에서 고품질 CMR 이미지를 생성하여 해석 가능성을 높입니다.

- **Technical Details**: CardiacNets는 두 가지 주요 기능을 수행하며, 첫 번째로는 심장 기능 지표를 평가하고 관상동맥 질환(coronary artery disease), 심근병증(cardiomyopathy), 심막염(pericarditis), 심부전(heart failure) 및 폐고혈압(pulmonary hypertension)과 같은 CVD를 탐색하는 것입니다. 두 번째로, ECG 데이터를 사용하여 높은 품질의 CMR 이미지를 생성함으로써 의사들이 보다 쉽게 진단할 수 있도록 지원합니다. 이 연구는 두 개의 대규모 공개 데이터 세트(UK Biobank와 MIMIC-IV-ECG)와 세 개의 개인 데이터 세트에서 CardiacNets를 훈련하고 검증하였습니다.

- **Performance Highlights**: CardiacNets는 기존의 ECG 전용 모델에 비해 일관되게 우수한 성능을 보여주었다고 보고되었습니다. 이는 선별 정확도가 유의미하게 향상되었으며, 생성된 CMR 이미지는 모든 경험 수준의 의사들에게 진단적 지원을 제공합니다. 이 연구는 ECG가 심장 기능 평가에 대한 크로스 모달 통찰을 촉진할 수 있는 방법을 보여주어 인구 수준에서 향상된 CVD 선별 및 진단에 기여할 수 있는 가능성을 제시합니다.



New uploads on arXiv(cs.AI)

### Resolving Multiple-Dynamic Model Uncertainty in Hypothesis-Driven Belief-MDPs (https://arxiv.org/abs/2411.14404)
Comments:
          8 pages, 4 figures, submitted to AAMAS 2025

- **What's New**: 이 논문에서는 모형 기반의 불확실성 해결 모델인 '가설 기반 신념 MDP(hypothesis-driven belief MDP)'를 제안합니다. 이는 여러 가설을 동시에 추론할 수 있도록 하여, 기본 POMDP 문제에서 의사 결정을 최적화할 수 있는 새로운 프레임워크입니다. 연구진은 특히 각 가설이 다른 동적 모델과 관련된 경우를 다룹니다.

- **Technical Details**: 문서에서는 Partially Observable Markov Decision Processes(POMDP)에 기초하여 불확실성을 다루는 의사 결정 문제를 논의합니다. 논문이 제안한 새로운 모델은 무한한 이전 역사(‘curse of history’) 문제를 해결하고, 각 가설로 인해 생기는 복잡성을 효과적으로 관리할 수 있도록 설계되었습니다. 행동-관측 쌍 각각에 대한 신념을 추론하면서, 최적의 설정을 유지합니다.

- **Performance Highlights**: 시뮬레이션을 통해 제안된 보상 함수와 새로운 프레임워크의 효과를 입증했습니다. 모델은 기존의 희소 트리 탐색 알고리즘을 사용하여 다중 가설을 추론하는 데에도 성공적으로 적용되었습니다. 이 연구는 드론 또는 우주 물체 추적 시스템과 같은 다양한 상황에서 불확실성을 줄이고 올바른 가설을 결정하는 데 도움을 줄 수 있습니다.



### RV4Chatbot: Are Chatbots Allowed to Dream of Electric Sheep? (https://arxiv.org/abs/2411.14368)
Comments:
          In Proceedings FMAS2024, arXiv:2411.13215

- **What's New**: 이번 논문에서는 안전-critical (safety-critical) 애플리케이션에서의 챗봇의 신뢰성을 높이기 위해 RV4Chatbot이라는 Runtime Verification 프레임워크를 소개합니다. 이 시스템은 챗봇의 행동이 기대되는 안전한 동작 기준을 지속적으로 따르는지를 모니터링하는 데 중점을 두고 있습니다.

- **Technical Details**: RV4Chatbot은 사용자와 챗봇 간의 상호작용 프로토콜(interaction protocols)을 정식화하여 기대되는 행동을 정의합니다. 이 프레임워크는 Rasa 프레임워크로 생성된 챗봇을 모니터링하는 RV4Rasa와 Dialogflow 챗봇을 모니터링하는 RV4Dialogflow라는 두 가지 구현으로 구성되어 있습니다.

- **Performance Highlights**: 실험은 공장 자동화 시나리오에서 RV4Rasa와 RV4Dialogflow를 사용하여 수행되었습니다. 이를 통해 각 시스템이 실제 환경에서도 효과적으로 챗봇의 행동을 검증할 수 있다는 것을 보여주었습니다.



### Physics-Informed LLM-Agent for Automated Modulation Design in Power Electronics Systems (https://arxiv.org/abs/2411.14214)
- **What's New**: 이 논문에서는 LP-COMDA라는 LLM 기반의 물리정보 활용 자율 에이전트를 제안하며, 이는 전력 전자 시스템(power electronics systems)에서 전력 변환기 설계(modulation design)를 최소한의 인간 감독으로 자동화합니다. 전통적인 AI 보조 접근 방식과 달리, LP-COMDA는 사용자 친화적인 채팅 인터페이스를 통해 설계 사양을 수집하고 검증합니다. 이 시스템은 단계별로 모듈레이션 설계를 생성하고 정교화하는 방식으로 진행됩니다.

- **Technical Details**: LP-COMDA는 LLM 기반 플래너가 사용자 요구사항을 처리하여 설계 사양 집합을 생성합니다. 이후 이 에이전트는 물리 정보 기반 대리 모델(physics-informed surrogate models)과 최적화 알고리즘을 통합하여 사용자 맞춤형 최적 모듈레이션 파라미터를 반복적으로 도출합니다. 디자인 과정에서 사용자에게 설명 가능한 방법으로 최적 설계 파라미터와 성능 메트릭스를 시각화하여 제공합니다.

- **Performance Highlights**: 실험 결과에 따르면, LP-COMDA는 낮은 데이터 시나리오에서 벤치마크 모델보다 63.2% 더 낮은 오류율을 보이며 우수한 성능을 입증합니다. 또한, 20명의 전문가와의 실증 연구에서는 LP-COMDA를 사용했을 때 설계 시간이 기존 방법보다 33배 이상 빠른 결과를 나타내어 설계 효율성을 크게 향상시켰습니다.



### Multi-LLM-Agent Systems: Techniques and Business Perspectives (https://arxiv.org/abs/2411.14033)
- **What's New**: 이번 논문에서는 다중 대규모 언어 모델(MLLM)을 이용한 자동화된 시스템인 다중 대규모 LLM 에이전트 시스템(MLAS)의 기술 및 비즈니스 측면을 논의합니다. MLAS는 기존 단일 LLM 시스템에 비해 더 높은 과제 해결 성능과 유연성을 제공합니다. 이 시스템은 각 참여 entidads가 데이터 개인 정보를 보호하고, 수익화 가능성을 높일 수 있는 기반을 제공합니다.

- **Technical Details**: LLM 기반 AI 에이전트 아키텍처는 입력을 처리하고 맥락을 유지할 수 있게 설계되어 있습니다. 상호작용 래퍼는 에이전트가 환경 및 다른 에이전트와 상호 작용하는 주요 인터페이스로, 커뮤니케이션 흐름을 관리합니다. 기억 관리( memory management )는 단기 작업 메모리와 장기 에피소드 저장소를 포함하여 최근의 상호작용과 경험을 바탕으로 적절한 응답을 생성하는 데 필요한 중요한 구성 요소입니다.

- **Performance Highlights**: MLAS 생태계는 Intelligent Collaboration을 통해 사용자에게 더욱 동적인 상호작용을 제공합니다. 에이전트가 문맥에 따라 도구 선택을 수행할 수 있는 능력은 문제 해결의 효율성을 높이며 다양한 기능을 통합하여 작업을 보다 효과적으로 수행할 수 있도록 합니다. 이로 인해 MLAS는 인공지능 집단 지능을 실현할 수 있는 유용한 솔루션으로 자리잡을 것으로 예상됩니다.



### Logic Augmented Generation (https://arxiv.org/abs/2411.14012)
Comments:
          10 pages, 2 figures

- **What's New**: 이 논문에서는 Semantic Knowledge Graphs (SKGs)와 Large Language Models (LLMs) 간의 한계점을 극복하기 위한 Logic Augmented Generation (LAG)이라는 새로운 패러다임을 제안합니다. LAG는 SKGs가 제공하는 명확한 논리적 경계와 LLM이 가진 유연성을 결합하여, 집단 지성을 활용한 의료 진단 및 기후 서비스와 같은 다양한 분야에서의 효과적인 협업을 지원합니다.

- **Technical Details**: LAG는 LLMs를 Reactive Continuous Knowledge Graphs (RCKGs)로 활용하여, 비구조적 데이터를 처리하고 즉각적으로 컨텍스트 기반의 지식을 생성하는 데 초점을 맞춥니다. 이러한 접근 방식은 정보의 논리적 일관성을 보장하며, SKGs의 구조화된 지식과 LLM의 유연성을 통합하여 하이브리드 시스템을 구성합니다. 이 시스템은 텍스트, 음성, 이미지 등 다양한 모달리티의 데이터를 통합하여 지식을 생성합니다.

- **Performance Highlights**: 연구에서 제안된 LAG 시스템은 의료 진단 및 기후 예측 지원을 위해 두 가지 집단 지성 과제를 통해 성능을 입증하였습니다. RCKG는 LLM의 지속적인 학습 능력을 활용하여 무한한 지식을 생성할 수 있으며, 이는 전문가들 간의 효과적인 협업을 촉진하고 잘못된 진단을 최소화할 수 있습니다. 또한, SKGs와 LLM의 결합은 복잡한 문제 공간에서 해답을 도출하는 데 필요한 해석 가능성과 신뢰성을 높입니다.



### XAgents: A Framework for Interpretable Rule-Based Multi-Agents Cooperation (https://arxiv.org/abs/2411.13932)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs)으로부터 암묵지(implicit knowledge)와 논리적 추론(logical reasoning) 능력을 추출하는 데 어려움이 있다는 점을 지적합니다. 다중 에이전트 시스템(multi-agent systems)의 발전을 바탕으로 IF-THEN 규칙 기반 시스템을 활용한 XAgents 프레임워크를 제안합니다.

- **Technical Details**: XAgents 프레임워크는 다중 극 뉴런(multi-polar neurons)의 구조에 영감을 받아 개발되었습니다. 이 프레임워크는 IF-파트가 논리적 추론과 도메인 소속 계산을 담당하고, THEN-파트는 도메인 전문 에이전트(domain expert agents)가 생성하는 도메인 특정 콘텐츠로 구성됩니다. XAgents는 멤버십 계산 후 다양한 도메인 규칙(domain rules)으로 작업을 전송합니다.

- **Performance Highlights**: XAgents는 최신 AutoAgents와의 비교 분석을 통해 3개의 서로 다른 데이터 세트에서 우수한 성능을 보였습니다. SHAP 알고리즘을 통한 후속 해석 가능한 연구(post-hoc interpretable studies)와 사례 연구(case studies)를 통해 입력-출력(feature correlation) 특성 및 규칙 기반 의미론(rule-based semantics)의 해석 가능성을 입증했습니다.



### Generative Fuzzy System for Sequence Generation (https://arxiv.org/abs/2411.13867)
Comments:
          12 pages, 5 figures

- **What's New**: 이 논문은 생성 모델(Generative Models, GMs)과 특히 대형 언어 모델(Large Language Models, LLMs)의 한계점을 개선하기 위해 퍼지 시스템(fuzzy system)을 도입하고 있습니다. 연구진은 데이터와 지식 기반 메커니즘을 결합하여 생성 업무에 적용되는 새로운 Generative Fuzzy System 프레임워크, 즉 GenFS를 제안합니다. GenFS는 GMs의 딥러닝(d深層學習) 기능과 퍼지 시스템의 해석 가능성을 통합한 혁신적 접근법입니다.

- **Technical Details**: FuzzyS2S라는 시퀀스 생성 모델은 GenFS 기반의 모델로, 12개의 데이터셋에 대해 기계 번역(machine translation), 코드 생성(code generation), 요약 생성(summary generation)이라는 세 가지 생성 작업 범주에서 실험을 수행했습니다. 이 모델은 전통적인 데이터 중심 접근 방식을 넘어 퍼지 시스템의 이점과도 통합되어 있습니다. GenFS는 데이터 학습 외에도 복잡한 모델 구조를 해석 가능한 방식으로 제공하여 사용자가 출력을 더 잘 이해하고 제어할 수 있도록 지원합니다.

- **Performance Highlights**: 실험 결과에 따르면 FuzzyS2S는 Transformer 모델보다 정확도와 유창성 면에서 우수한 성능을 보였습니다. 또한, T5 및 CodeT5와 같은 최신 모델들과 비교했을 때도 일부 데이터셋에서 더 나은 성능을 발휘하는 것으로 나타났습니다. 이는 FuzzyS2S가 생성적 업무에 있어 더 높은 내구성과 일반화 능력을 지니고 있음을 시사합니다.



### AI-Driven Agents with Prompts Designed for High Agreeableness Increase the Likelihood of Being Mistaken for a Human in the Turing Tes (https://arxiv.org/abs/2411.13749)
Comments:
          25 pages, 2 figures, 7 tables

- **What's New**: 이번 연구에서는 빅파이브 인벤토리(Big Five Inventory)를 기반으로 한 다양한 수준의 동의성(agreeableness)을 가진 세 가지 GPT 모델이 터링 테스트(Turing Test)에서 테스트되었습니다. 특히, 높은 동의성을 가진 AI 에이전트는 60% 이상의 혼란률(confusion rate)을 기록하였고, 가장 인간 같은 특성을 보여주었습니다. 이는 AI 시스템의 인간화와 인간-인공지능 협업의 필요성을 강조합니다.

- **Technical Details**: 이 연구에서는 동의성, 신경성(neuroticism), 외향성(extroversion) 등 빅파이브의 다양한 성격 특성을 가진 모델들이 시험되었습니다. 실험 결과, 모든 AI가 50% 이상의 혼란률을 보였고, 이들은 심리적 인식과 인간화에 관한 문헌에서 설명된 여러 심리적 프레임워크(psychological frameworks)와 관련이 있습니다.

- **Performance Highlights**: 특히, 동의성이 높은 AI 에이전트는 60% 이상의 혼란률을 달성하고, 가장 인간에 가까운 특성을 입증하였습니다. 이러한 연구 결과는 인공지능 분야에서 인격 설계(personality engineering)의 중요성을 부각시키며, 심리학과의 협력을 통해 인간-인공지능 협업의 시스템 적응성을 향상시키기 위한 노력이 필요함을 시사합니다.



### No Free Delivery Service: Epistemic limits of passive data collection in complex social systems (https://arxiv.org/abs/2411.13653)
Comments:
          To appear in NeurIPS'24

- **What's New**: 이 논문에서는 머신러닝과 AI 분야에서 훈련-테스트 패러다임(train-test paradigm)이 어떻게 모델 검증(model validation)에서 중요한 역할을 해왔는지를 설명합니다. 그러나 현대 AI 시스템은 데이터 수집(data collection) 관행과 다양한 작업(task)의 조합이 테스트 유효성(test validity) 가정을 위반하는 경우가 많다는 사실을 지적합니다. 이러한 문제는 AI 모델의 의도된 결과를 보장할 수 없게 만듭니다.

- **Technical Details**: 복잡한 사회 시스템(complex social systems)에서 널리 고려되는 추론 설정(inference settings)에 따르면, 훈련-테스트 패러다임은 어떤 위험 추정기(risk estimator), 특히 반사실적 추정기(counterfactual estimator)와 인과 추정기(causal estimator)에 대해서도 높은 확률로 정당성이 결여되어 있으며, 실제로는 유효하지 않다는 것을 보여 줍니다. 이는 최신 AI의 주요 작업에서 데이터 수집 관행 아래에서 모델이 유효한지 알 수 없다는 근본적인 인식론적( epistemic) 문제를 강조합니다.

- **Performance Highlights**: 구체적으로, 추천 시스템(recommender systems)과 대형 언어 모델(large language models)을 통한 추론에 대한 변형도 이러한 한계에 포함되며, 단순한 확장(naïve scaling)이나 제한된 평가 지표(benchmarks)로는 문제를 해결할 수 없습니다. 이 논문은 MovieLens 벤치마크를 통해 이러한 결과를 보여주며, 사회 시스템에서의 AI 결과에 대한 시사점을 논의합니다. 여기에는 참여형 데이터 보존(participatory data curation)과 열린 과학(open science)과 같은 가능한 해결책도 포함됩니다.



### Integrated Water Resource Management in the Segura Hydrographic Basin: An Artificial Intelligence Approach (https://arxiv.org/abs/2411.13566)
Comments:
          15 pages, 14 figures, 8 tables

- **What's New**: 이 논문은 물 관리 시나리오에서 불확실한 수요, 가변적인 가용성, 복잡한 거버넌스 정책을 해결하기 위한 패러다임적 프레임워크를 제시합니다. 고급 물리 모델링(advanced physical modelling), 원격 감지 기법(remote sensing techniques), 인공지능 알고리즘(Artificial Intelligence algorithms)의 통합을 통해 자원 관리의 효율성을 높입니다.

- **Technical Details**: 제안된 접근 방식은 포괄적인 수문학 모델(hydrological model)과 정밀한 수요 추정을 위한 농업 작물 모델(agronomic crop models), 효율적인 자원 분배를 위한 혼합 정수 선형 프로그래밍(Mixed-Integer Linear Programming)을 결합하여 단기 및 장기의 물 가용성을 정확하게 예측하고 수요를 추정합니다.

- **Performance Highlights**: 세구라 수계(Segura Hydrographic Basin)의 사례 연구에서 이 방법론은 약 6개월 동안 약 6억 4천2백만 입방미터($hm^3$)의 물을 성공적으로 분배하였고, 총 예상 수요의 9.7%로 결핍을 최소화했습니다. 이 방법론은 CO2 배출을 줄이고 자원 분배를 최적화하는 등 환경적인 이점을 보여주었으며, 다양한 맥락에서 지속 가능한 물 관리 결정을 지원합니다.



### AMSnet-KG: A Netlist Dataset for LLM-based AMS Circuit Auto-Design Using Knowledge Graph RAG (https://arxiv.org/abs/2411.13560)
- **What's New**: 이번 논문은 AMSnet-KG라는 데이터세트를 소개하며, 다양한 AMS 회로의 회로도 및 네트리스트를 포함하고 있습니다. 이 데이터세트는 기능 및 성능 특성에 대한 주석이 달린 지식 그래프를 구축합니다. AMSnet-KG를 활용하여, LLM에 내장된 방대한 지식을 이용한 자동 회로 생성 프레임워크인 AMSgen을 제시합니다.

- **Technical Details**: AMSgen은 요구되는 사양에 기반하여 회로 아키텍처를 설명하는 고수준 설계 전략을 생성합니다. 그 후, 일치하는 회로 구성 요소를 AMSnet-KG에서 효율적으로 검색하여 완전한 위상(topology)으로 조립합니다. 또한, 베이지안 최적화(Bayesian Optimization)를 통해 트랜지스터 사이징을 자동화합니다. 생성된 설계가 성능 사양을 충족하지 못할 경우, AMSgen은 현재 설계 및 달성된 성능을 추가적인 few-shot 예제로 포맷하여 LLM을 기반으로 한 전략 생성 단계로 돌아갑니다.

- **Performance Highlights**: 이 연구에서는 28nm 기술로 회로를 설계하며, 최소한의 인간 노력으로 다양한 성능 사양을 달성합니다. 논문에서는 사양부터 전이 전 수준의 네트리스트까지 자동 설계 흐름을 검증하기 위해 운영 증폭기와 비교기 설계의 사례 연구를 수행합니다. AMSnet-KG는 논문 출판 시에 공개될 예정이며, 이는 EDA 분야에서 LLM의 활용을 통한 큰 진전을 이끌 수 있는 발판이 될 것으로 기대됩니다.



### Revisiting the Integration of Convolution and Attention for Vision Backbon (https://arxiv.org/abs/2411.14429)
Comments:
          NeurIPS 2024

- **What's New**: 이번 연구에서는 Convolutions (Convs)와 Multi-Head Self-Attentions (MHSAs)를 서로 다른 수준의 granularity에서 병렬로 사용하는 GLMix라는 새로운 통합 방안을 제안합니다. 이 방법은 Convs가 지역적 특징을 추출하고, MHSAs가 전역 상호작용을 학습하는 데 최적화되어 있습니다. 각각의 층에서 이미지를 세밀한 그리드와 대략적인 의미 있는 슬롯으로 표현하여, 이들 간의 특징을 효율적으로 융합할 수 있습니다.

- **Technical Details**: GLMix 블록에서는 Convs와 MHSAs의 결합을 통해 지역적 및 전역적 모델링을 달성합니다. soft clustering과 dispatching 모듈을 도입하여 세트와 그리드 표현 간의 연결을 가능하게 하여, 경량화된 Convs로 고해상도 특징을 추출하고 MHSAs로 제한된 수의 의미 슬롯을 처리하여 성능을 높입니다. 이를 통해 GLNet-STL이 82.5%의 top-1 정확도를 달성하고 있습니다.

- **Performance Highlights**: GLNet-4G, GLNet-9G, GLNet-16G 모델은 각각 83.7%, 84.5%, 85.0%의 top-1 정확도를 기록하며, 기존 최첨단 성능보다도 더 효율적인 결과를 보여줍니다. 다양한 컴퓨터 비전 작업에서 GLNet의 성능을 empirically 검증하였으며, soft clustering 모듈에서는 의미 있는 시맨틱 그룹화 효과가 나타났습니다. 이러한 결과들은 GLMix의 효과성을 강조하며, 향후 비지도형 시맨틱 세분화 접근법 개발에 영감을 줄 수 있습니다.



### Whack-a-Chip: The Futility of Hardware-Centric Export Controls (https://arxiv.org/abs/2411.14425)
- **What's New**: 이 보고서는 미국의 반도체 수출 통제가 중국의 인공지능(AI) 연구에 미치는 영향을 분석하며, 특히 중국의 주요 AI 연구소들이 이러한 통제를 회피하는 방법에 대한 구체적인 증거를 제시합니다. 중국 기업, 특히 텐센트가 고급 AI 모델을 훈련하기 위해 제한된 하드웨어와 소프트웨어 기술을 활용하는 방법에 대해 설명합니다. 이는 기존의 수출 통제 전략이 효율성을 저하시키지 못하고 있음을 보여줍니다.

- **Technical Details**: 텍 센트의 Hunyuan-Large 모델은 미국의 수출 통제 대상이 아닌 NVIDIA H20 GPU를 사용하여 훈련되었습니다. 이 모델은 여러 하위 작업에서 최첨단 성능을 달성하며, 혼합 전문가(Mixture-of-Experts, MoE)와 같은 고급 기법을 활용하여 한정된 하드웨어 리소스를 최대화하는 방법에 대해 설명합니다. 또한, mixed precision training을 통해 훈련 시간을 줄이고 성능을 향상시키는 다양한 기술들이 논의됩니다.

- **Performance Highlights**: Hunyuan-Large는 MMLU 및 CommonsenseQA와 같은 여러 다운스트림 벤치마크에서 최첨단 성능을 보여주며, 최신 AI 모델들 사이에서 우수한 경쟁력을 갖추고 있습니다. 특히, 저성능 하드웨어를 사용하면서도 효율성을 극대화하는 방법이 보편화되면서, 이러한 전략이 앞으로도 중국의 AI 개발에 중대한 영향을 미칠 것이라는 점에서 주목받고 있습니다.



### Landing Trajectory Prediction for UAS Based on Generative Adversarial Network (https://arxiv.org/abs/2411.14403)
Comments:
          9 pages, AIAA SCITECH 2023

- **What's New**: 이 논문은 항공모빌리티 연구의 필수 요소인 비행 경로 예측 모델을 제안합니다. 특히, 제안된 모델은 UAS(무인 항공 시스템)의 착륙 경로를 예측하기 위해 Generative Adversarial Network (GAN)을 기반으로 하고 있습니다. 이 방법은 기존의 Gaussian Mixture Regression(GMR) 방식보다 빠르고 높은 정확도를 보여줍니다.

- **Technical Details**: 논문은 GAN을 기반으로 한 2D 비행 경로 예측 접근 방식을 이용하여 UAS의 3D 착륙 경로를 예측합니다. GAN은 generator와 discriminator라는 두 개의 신경망으로 구성되어 있으며, generator는 입력 경로의 특성을 이해하여 랜덤 상태의 비행 경로를 출력합니다. Long Short-Term Memory (LSTM)와의 결합을 통해 더 나은 예측이 가능하다고 설명합니다.

- **Performance Highlights**: 제안된 모델은 2600개 이상의 실제 UAV 착륙 경로 데이터셋을 생성하고, 실험 결과 GMR 방법보다 더 정확한 예측을 보여줍니다. GAN의 discriminator 부분은 다른 예측의 평가에도 활용될 수 있어 가치를 높입니다. 앞으로의 연구를 위한 특정 데이터셋 제공도 중요한 기여로 언급됩니다.



### Using Formal Models, Safety Shields and Certified Control to Validate AI-Based Train Systems (https://arxiv.org/abs/2411.14374)
Comments:
          In Proceedings FMAS2024, arXiv:2411.13215

- **What's New**: 이 논문은 자율 시스템(Object)의 인증에 대한 새로운 접근법을 모색하는 KI-LOK 프로젝트를 소개합니다. 기존의 인증 방법에 더해, 자율 기차(autonomous trains) 내 AI 구성 요소의 안전한 통합을 위한 기법을 개발하고 있습니다. 특히, B 방법을 사용하여 조종 시스템의 안전성을 보장하고, 런타임 인증 검사를 통한 인식 시스템의 신뢰성을 향상시키는 이중 접근 방식을 취하고 있습니다.

- **Technical Details**: 이 연구는 정형 모델(formal model)을 기반으로 한 시뮬레이션을 통해 조종 시스템의 안전성과 인식 시스템의 신뢰성을 연결하는 데 집중합니다. 실시간 런타임 인증 검사를 위한 데모 시스템이 실제 AI 출력과 실제 인증 검사기에 의해 제어되며, ProB라는 검증 도구(validation tool)에 통합됩니다. 이로 인해 런타임 모니터링(runtime monitoring), 런타임 검증(runtime verification), 통계적 검증(statistical validation)을 통한 정형 안전 속성(formal safety properties)의 검토가 가능해집니다.

- **Performance Highlights**: 연구의 적용 사례로 신호 탐지(signal detection) 문제를 다루었으며, 이를 통해 AI와 인증 검사기의 잠재적 취약점(vulnerabilities) 및 약점을 분석할 수 있는 방법을 제시합니다. 이러한 접근법을 통해 자율 시스템의 안전성을 높이고, 산업 및 과학 분야에서의 사용 가능성을 확대할 수 있습니다.



### Synthesising Robust Controllers for Robot Collectives with Recurrent Tasks: A Case Study (https://arxiv.org/abs/2411.14371)
Comments:
          In Proceedings FMAS2024, arXiv:2411.13215

- **What's New**: 이 논문에서는 자율 집합체(collectives)를 위해 설계된 정확한 제어기(controller)에 대한 세 가지 주요 과제를 다루고 있습니다. 특히, 고수준 제어기 합성을 위한 간단하면서도 유용한 추상화 개념을 제안하고, 최적화 목표와 비상 상황에 대한 제약 조건들을 포함하는 방식에 초점을 맞추고 있습니다. 이를 통해 실용적 규모에서의 적용 가능성을 높이고 있습니다.

- **Technical Details**: 제안된 모델은 부분 관찰 마르코프 결정 과정(Partially Observable Markov Decision Process, POMDP)에 기반하여, 스토캐스틱 두 플레이어 게임을 단일 플레이어 게임으로 간소화합니다. 이 과정은 환경의 불확실성에 대한 강건성을 보장하며, POMDP 전략 합성 후에 선형 시간의 정확성 속성을 별도로 검증합니다. 필수적인 모형 설계 및 제어기 합성 및 평가 절차에 대한 상세한 지침을 제공합니다.

- **Performance Highlights**: 배터리 구동 로봇을 통한 공공 건물 청소 사례를 예로 들며, 실제 환경에서의 적용 가능성을 보여줍니다. 이 로봇들은 청소 효율성을 극대화하면서도 전력 소모와 같은 제약 조건을 준수하도록 설계되었습니다. 또한 이 연구는 자율 집합체의 실제 작업에 대한 실질적인 가이드를 제공하여, 복잡한 작업을 수행하는 로봇의 제어와 관리에 기여합니다.



### ROSMonitoring 2.0: Extending ROS Runtime Verification to Services and Ordered Topics (https://arxiv.org/abs/2411.14367)
Comments:
          In Proceedings FMAS2024, arXiv:2411.13215

- **What's New**: 본 논문은 hybrid nature (하이브리드 특성)와 distributed architecture (분산 아키텍처)로 인한 로봇 응용 프로그램의 형식적 검증 문제를 다룹니다. ROSMonitoring 2.0을 새롭게 소개하며, 이는 메시지의 발행 및 수신 순서를 고려하여 주제와 서비스를 모니터링하도록 설계된 확장판입니다. 해당 프레임워크는 ROS1 및 부분적으로 ROS2 환경을 지원하도록 개선되어, 실시간 지원, 보안, 확장성 및 상호운용성이 향상되었습니다.

- **Technical Details**: ROSMonitoring 2.0은 메시지의 발행 및 수신 순서를 고려하여 모니터링 기능을 강화했습니다. 이 프레임워크는 실시간 지원을 개선하고, 보안적 요소를 강화하였으며, 다양한 시스템 간의 상호운용성을 증대시키기 위한 기술적 조정이 이루어졌습니다. 주요 개선 사항은 특정 로봇 응용 프로그램의 성능을 최적화하는 데 기여합니다.

- **Performance Highlights**: 논문에서는 화재 진압용 UAV(자율 비행체)의 특정 구성 요소에 대한 런타임 모니터링을 포함한 사례 연구 결과를 제시합니다. 이를 통해 ROSMonitoring 2.0이 이루어진 개선이 실제 환경에서 성능을 어떻게 향상시킬 수 있는지를 보여주고 있습니다. 또한, 이 연구는 새로운 모니터링 구조가 로봇 애플리케이션의 신뢰성과 효율성을 높일 수 있음을 강조합니다.



### Contrasting local and global modeling with machine learning and satellite data: A case study estimating tree canopy height in African savannas (https://arxiv.org/abs/2411.14354)
Comments:
          31 pages; 9 figures

- **What's New**: 이 연구는 위성 이미지를 활용한 기계 학습(SatML)의 최신 발전이 지역적 모델링 능력에 어떻게 영향을 미치는지를 밝혀내고자 하였습니다. 특히, 모잠비크 카링가니 자연 보호구역에서 나무 캐노피 높이(탑령) 매핑을 통해 지역과 글로벌 모델링 패러다임을 비교하였습니다. 연구 결과, 기존의 글로벌 모델보다 지역 자체 데이터를 기반으로 학습한 작은 모델이 더 우수한 성능을 보임을 발견했습니다.

- **Technical Details**: 연구는 총 세 가지 연구 질문(RQ1, RQ2, RQ3)을 중심으로 진행되었습니다. RQ1에서는 지역 예측을 위해 지역 데이터를 수집하는 것이 얼마나 중요한지, RQ2에서는 TCH 모델의 신뢰성에 가장 큰 영향을 미치는 설계 요인은 무엇인지, RQ3에서는 지역과 글로벌 모델링 노력 간의 충돌 또는 시너지가 어떤 것인지 살펴보았습니다. 결과적으로, 지역 데이터는 기계 학습 모델에서 정확하고 생태학적으로 관련성이 높은 예측을 생성하는 데 필수적임을 명확하게 보여주었습니다.

- **Performance Highlights**: 연구에서 제안하는 모델은 기존의 글로벌 TCH 맵의 평균 절대 오차를 40-69%까지 줄였고, 이는 생태학적 관련성 기준에 따라 분류된 측정치를 포함합니다. 또한, 글로벌 데이터로 사전 학습한 모델을 지역 데이터로 미세 조정하는 것보다 지역 데이터를 전적으로 사용하여 모델을 훈련시키는 것이 성능이 더 뛰어난 것으로 분석되었습니다. 이러한 결과는 지역과 글로벌 모니터링 노력 간의 효율적인 상호 작용에 대한 향후 연구 방향에 중요한 통찰을 제공합니다.



### UnifiedCrawl: Aggregated Common Crawl for Affordable Adaptation of LLMs on Low-Resource Languages (https://arxiv.org/abs/2411.14343)
- **What's New**: 이 논문은 적은 자원으로 운영되는 언어들에 대한 LLMs의 성능을 향상시키는 효율적인 데이터 수집 방법론인 UnifiedCrawl을 소개합니다. 이 방법은 최소한의 컴퓨팅 자원으로 Common Crawl 데이터셋에서 적은 자원 언어에 대한 텍스트 데이터를 필터링하여 추출할 수 있도록 설계되었습니다. UnifiedCrawl을 통해 생성된 데이터셋은 기존에 사용 가능했던 소스들보다 상당히 큰 단일 언어 데이터셋을 제공합니다.

- **Technical Details**: UnifiedCrawl은 적은 자원 언어를 위해 Common Crawl에서 텍스트 데이터를 수집하기 위해 제안된 새로운 방법론입니다. 이 방법은 메모리, 컴퓨팅 및 네트워크 사용량을 최적화하면서 개인 소비자 하드웨어에서 작동할 수 있도록 설계되었습니다. 수집된 데이터 셋을 바탕으로 QLoRA와 같은 효율적인 어댑터 방법으로 다국어 LLM을 미세 조정하여, VRAM 사용량을 최소화하면서 성능을 크게 향상시킬 수 있습니다.

- **Performance Highlights**: 실험 결과, UnifiedCrawl로 수집된 데이터셋을 활용하여 LLMs의 언어 모델링 성능이 현저하게 개선되었으며, 특히 적은 자원 언어에 대한 응답 품질이 향상되었습니다. 모델은 few-shot prompting 점수를 증가시켰으며, 이전보다 더 나은 응답의 일관성과 유의미성을 보여주었습니다. 이 연구는 소비자 하드웨어를 사용하여 LLM을 향상시키기 위한 저렴한 접근 방식을 제공합니다.



### Automated Generation of Code Debugging Exercises (https://arxiv.org/abs/2411.14303)
Comments:
          Preprint of the SIGCSE'25 paper

- **What's New**: 이 논문은 BugSpotter라는 혁신적인 도구를 소개합니다. BugSpotter는 문제 설명에서 버그가 있는 코드를 생성하고, 테스트 스위트를 통해 이 버그들을 검증하는 기능을 가지고 있습니다. 학생들은 BugSpotter를 사용하여 실패하는 테스트 케이스를 설계하며, 이를 통해 디버깅 기술을 향상시키고 문제 명세를 읽고 이해하는 연습을 하게 됩니다. 이는 메타 인지적 스캐폴딩(Scaffolding) 원리에 부합하여 문제 이해에 도움을 줄 수 있음을 나타냅니다.

- **Technical Details**: BugSpotter는 대형 언어 모델(LLMs)을 이용하여 디버깅 연습 문제를 자동으로 생성합니다. 이 도구는 문제 사양과 관련된 버그가 있는 코드를 학생에게 제공하며, 학생은 이 코드를 해석하고 버그를 드러내는 테스트 케이스를 디자인해야 합니다. 일반적으로, 생성된 코드(CB)는 최소한 하나의 테스트 케이스에서 실패해야 하며, 이를 해결하기 위해 소규모의 수정이 필요합니다. BugSpotter는 이러한 단계를 통해 학생들이 문제 해결 능력을 키울 수 있도록 설계되어 있습니다.

- **Performance Highlights**: 우리는 BugSpotter를 대규모 강의 환경에서 배포하였고, LLM이 생성한 디버깅 문제와 강사가 수작업으로 생성한 문제를 비교하였습니다. 연구 결과, LLM이 생성한 연습 문제는 난이도가 다양하게 나타났으며 문제 명세에 잘 맞는 것으로 평가되었습니다. 결론적으로, LLM이 생성한 연습 문제는 인스트럭터가 디자인한 연습 문제와 비교할 때 학생들 성과 측면에서 유사하였으며, 이는 BugSpotter가 디버깅 학습에 효과적이며 효율적인 도구가 될 수 있음을 시사합니다.



### Neuro-Symbolic Query Optimization in Knowledge Graphs (https://arxiv.org/abs/2411.14277)
- **What's New**: 이 논문은 지식 그래프(knowledge graphs)에서의 신경-상징 질의 최적화(neuro-symbolic query optimization)라는 새로운 분야에 대해 다룹니다. 전통적인 질의 최적화 방법은 기호(symbolic) 방식에 의존하여 데이터 집합 요약, 통계 및 비용 모델을 활용합니다. 그러나 이러한 방식은 복잡한 질의나 대규모 데이터셋을 처리할 때 부정확한 추정으로 어려움을 겪습니다. 최근에는 신경 모델(neural models)을 도입하여 비선형적인 최적화 가능성을 탐구하고 있습니다.

- **Technical Details**: 저자들은 신경-상징 질의 최적화기를 소개하고 이 시스템의 아키텍처를 설명합니다. 이러한 하이브리드 시스템은 신경 컴퓨테이션(neural computation)과 기호 추론(symbolic reasoning)의 강점을 결합하여 질의 처리 성능을 향상시킵니다. 또한, 기존의 신경 컴포넌트를 검토하며, 이를 실제 환경에 적용할 때의 한계와 도전 과제를 논의합니다. 이 접근법은 질의 최적화의 검색 공간 탐색과 실행 계획의 효율성을 높이는 데 중점을 둡니다.

- **Performance Highlights**: 신경-상징 질의 최적화기는 전통적인 방법과 비교하여 복잡한 질의를 처리하는 데 있어 더 나은 성능을 보여줍니다. 통합된 접근 방식은 다차원적(query optimization) 문제를 효과적으로 해결할 수 있는 가능성을 제시하며, 특히 대규모 데이터셋에서의 성능 향상이 두드러집니다. 이 연구는 향후 지식 그래프의 질의 최적화 분야에 큰 기여를 할 것으로 기대됩니다.



### Generating Realistic Adversarial Examples for Business Processes using Variational Autoencoders (https://arxiv.org/abs/2411.14263)
- **What's New**: 이번 연구에서는 예측 프로세스 모니터링(Predictive Process Monitoring, PPM)의 취약성을 해결하기 위해 새로운 형태의 적대적 공격(adversarial attack) 방법을 소개합니다. 연구의 초점은 기업 프로세스와 관련하여 현실적인 적대적 사례를 생성하는 것입니다. 이 방법은 입력 속성을 직접 수정하는 대신, 입력 데이터의 잠재 공간(latent space) 표현으로 노이즈(noise)를 추가하는 방식을 사용합니다.

- **Technical Details**: 논문에서는 두 가지 새로운 잠재 공간 공격(latent space attack) 기법을 제안합니다. 첫 번째는 잠재 샘플링 공격(latent sampling attack)이고, 두 번째는 그래디언트 스텝 공격(gradient steps attack)입니다. 이 방법들은 입력 데이터 공간 대신 데이터의 밀집하고 연속적인 표현 공간에서 적대적 변환을 찾아내며, 도메인에 구애받지 않는 특징을 가지고 있습니다.

- **Performance Highlights**: 여덟 가지 적대적 공격 방법에 대한 철저한 벤치마킹을 수행하며, 이들을 실제 이벤트 로그와 다양한 예측 모델을 통해 평가합니다. 이 공격 방식은 원본 데이터 분포와 동일한 조건에서 현실적인 적대적 사례를 생성하며, OOPPM 모델의 취약성을 평가합니다. 연구 결과, 제안된 방법이 기존 방법들보다 더 높은 성공률을 기록하는 것으로 나타났습니다.



### Knowledge Graphs, Large Language Models, and Hallucinations: An NLP Perspectiv (https://arxiv.org/abs/2411.14258)
Comments:
          7 pages, 2 Figures, 1 Table

- **What's New**: 최근 대규모 언어 모델(LLM)의 발전은 자연어 처리(NLP) 응용 프로그램에서 큰 변화를 이끌고 있습니다. 그러나 LLM은 사실과 일치하지 않는 답변을 내놓는 '환각(hallucination)' 문제에 직면하고 있어 신뢰성의 저하와 활용성의 한계를 보이고 있습니다. 이 논문에서는 지식 그래프(KG)를 활용하여 LLM의 환각 문제를 해결하고자 하는 최신 연구 동향을 다루고 있습니다.

- **Technical Details**: 지식 그래프(KGs)는 엔티티(entities)와 그 관계(edges)로 구성된 구조화된 정보로, LLM의 사실적 일관성을 높이는 데 중요한 역할을 수행합니다. KG를 통해 LLM은 실제 세계 객체에 대한 사실적 정보를 효율적으로 인지할 수 있고, 이는 모델의 재훈련을 줄이는 데 기여합니다. 논문에서는 다양한 환각 완화 모델과 이러한 모델의 아키텍처를 카테고리화하여 설명하고 이에 대한 평가 방법도 제시합니다.

- **Performance Highlights**: 이 논문은 환각 탐지를 위한 메트릭스인 BERTScore와 BARTScore를 사용하기도 하며, 이러한 메트릭스의 한계점을 지적하고 더 세밀한 환각 탐지를 위한 연구가 필요하다고 강조합니다. 또한, 최근 LLM의 활용 빈도가 높아짐에 따라 다영역(multi-domain)에서의 평가가 필요하다고 주장하며, 이는 LLM의 실제 응용 가능성을 높이는 데 기여할 것입니다. 이는 환각 탐지와 지식 통합 모델의 효과성을 더욱 향상시킬 것으로 기대됩니다.



### Do I Know This Entity? Knowledge Awareness and Hallucinations in Language Models (https://arxiv.org/abs/2411.14257)
- **What's New**: 이 논문에서는 쿠데타를 발생시키는 주요 원인으로써 엔티티 인식을 제안합니다. 대규모 언어 모델에서의 허위 사실 생성(hallucination) 문제를 해결하기 위해 희소 오토인코더(Sparse Autoencoders)라는 해석 도구를 활용했습니다. 이 도구를 통해 모델이 특정 엔티티에 대해 어떤 지식을 가지거나 가지지 않는지를 평가하는 자기 지식의 선형 방향을 발견하였습니다.

- **Technical Details**: 희소 오토인코더(SAEs)는 모델 표현을 더 큰 차원의 공간으로 사영하여 해석 가능한 특성의 분리를 도모합니다. 이 연구에서 Gemma Scope의 SAEs를 이용하여 2B 및 9B 모델에서 자기 지식의 내부 표현을 발견했습니다. 또한, 우리는 엔티티의 속성에 대한 질문에 대한 모델의 응답 거부 행동이 이러한 방향에 의해 유도될 수 있음을 입증했습니다.

- **Performance Highlights**: 이 연구는 희소 오토인코더를 통해 다양한 엔티티 유형에 대해 일반화된 엔티티 인식 방향을 발견했음을 보여주었습니다. 모델이 알려진 엔티티에 대한 질문을 거부하도록 유도하거나, 알려지지 않은 엔티티의 속성을 허위로 생성하게 만들 수 있는 능력을 갖고 있음을 밝혀냈습니다. 이는 사전 훈련 데이터에서 학습된 기존의 메커니즘이 파인 튜닝 과정에서 재사용되는 경향이 있음을 시사합니다.



### BERT-Based Approach for Automating Course Articulation Matrix Construction with Explainable AI (https://arxiv.org/abs/2411.14254)
Comments:
          26 pages, 9 figures

- **What's New**: 이 연구는 CO(코스 성과)와 PO/PSO(프로그램 성과) 정렬을 자동화하기 위해 사전학습된 BERT 기반 모델을 활용하는 혁신적인 접근 방식을 제시합니다. 모델은 트랜스퍼 러닝(transfer learning)을 통해 미세 조정되어, 강의 성과와 프로그램 성과 간의 의미적 관계를 효과적으로 포착할 수 있습니다. 또한, LIME(Local Interpretable Model-agnostic Explanations) 기법을 통해 모델의 의사결정 투명성을 보장하고 교육자가 결과를 이해할 수 있도록 돕습니다.

- **Technical Details**: 이 연구에서는 BERT 계열의 모델 중 BERT Base, DistilBERT, ALBERT, RoBERTa를 활용하여 CO와 PO/PSO 쌍의 정렬을 평가합니다. 전통적인 머신러닝 분류기(Decision Tree, Random Forest, XGBoost)를 먼저 평가한 후, 사전학습된 BERT 모델의 성능을 평가하기 위해 트랜스퍼 러닝을 적용합니다. 또한, 수집된 고품질 데이터셋과 동의어 기반 데이터 증강 방법을 통해 모델의 일반화 능력을 향상시킵니다.

- **Performance Highlights**: 본 시스템은 정확도(accuracy) 98.66%, 정밀도(precision) 98.67%, 재현율(recall) 98.66%, F1 점수 98.66%를 달성하였습니다. 이 연구는 사전학습된 BERT 모델을 사용하여 CO-PO-PSO 정렬을 자동으로 생성하면서 높은 성과와 해석 가능성을 제공하는 가능성을 입증합니다. 이전의 수동 프로세스를 자동화함으로써 시간과 인적 자원을 절약할 수 있는 효과적인 솔루션을 제시합니다.



### Intent-Aware Dialogue Generation and Multi-Task Contrastive Learning for Multi-Turn Intent Classification (https://arxiv.org/abs/2411.14252)
- **What's New**: 이번 논문에서는 Chain-of-Intent라는 혁신적인 방식을 도입하여 Hidden Markov Models(히든 마르코프 모델)과 Large Language Models(대규모 언어 모델)을 결합하여 맥락 인식이 가능한 대화 생성을 가능하게 합니다. 이 연구는 전자상거래 채팅 기록에서 도메인 특화된 지식을 추출하여 대화 턴과 인텐트 전환을 추정하고, 이를 통해 유기적인 대화를 생성합니다. 또한 MINT-CL 프레임워크를 제안하여 대규모 주석 데이터 없이도 다중 턴 인텐트 분류의 정확성을 향상시킵니다.

- **Technical Details**: Chain-of-Intent는 HMM과 LLM을 결합하여 사용자의 인텐트에 기반한 대화를 생성하는 자가 학습 방식입니다. MINT-CL은 다중 작업 대비 학습(multi-task contrastive learning)을 통해 MTIC의 정확성을 향상시키며, 다양한 언어와 시장에서 우수한 성능을 보여줍니다. 논문에서 발표된 MINT-E 데이터셋은 8개 시장에서의 다양한 언어로 인텐트를 포함하는 멀티턴 전자상거래 대화 코퍼스를 제공합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 대화 품질과 인텐트 분류 정확도 모두 기존 기준선을 초과하며, 특히 다국어 환경에서 두드러진 성능 향상을 보여줍니다. 또한 데이터 생성 노력의 현저한 감소를 통해, 다중 언어 및 대화 모델 개발에 필요한 자원을 절감할 수 있는 가능성을 제시합니다.



### Natural Language Reinforcement Learning (https://arxiv.org/abs/2411.14251)
Comments:
          Extension of arXiv:2402.07157

- **What's New**: 이번 연구는 Natural Language Reinforcement Learning (NLRL)이라는 새로운 가능성을 탐색합니다. 전통적인 Markov Decision Process (MDP)를 자연어 기반 표현 공간으로 확장하여 RL 원칙의 재정의를 시도합니다. 이는 게임, 로봇공학 및 언어 모델 등 다양한 분야에서의 기본 원칙을 혁신적으로 변형시킵니다.

- **Technical Details**: NLRL은 RL의 작업 목표(task objectives), 정책(policy), 가치 함수(value function), Bellman 방정식(Bellman equation), 정책 반복(policy iteration) 등의 개념을 자연어로 재정의합니다. 최근의 대형 언어 모델(LLMs)의 발전 덕분에 NLRL은 RL과 유사한 정책 및 가치 개선을 달성할 수 있는 실용적인 방법으로 제안됩니다.

- **Performance Highlights**: Maze, Breakthrough 및 Tic-Tac-Toe 게임을 통한 실험 결과 NLRL 프레임워크의 효율성 및 해석 가능성이 입증되었습니다. 이러한 실험은 다양하고 실제적인 사용 사례에 걸쳐 성과를 보여줍니다. 연구진은 이와 관련된 코드를 공개할 예정입니다.



### AnywhereDoor: Multi-Target Backdoor Attacks on Object Detection (https://arxiv.org/abs/2411.14243)
- **What's New**: 본 논문에서는 AnytimeDoor라는 유연한 백도어 공격(Vulnerability in backdoor attack)을 소개합니다. 이 방법은 객체 탐지(Object Detection) 모델에 적용되어 공격자가 실시간으로 다양한 공격 유형(예: 객체 사라짐, 생성, 잘못 분류)을 수행할 수 있게 합니다. AnywhereDoor는 기존의 정적이고 제한된 공격 시나리오의 한계를 극복하며 새로운 공격 가능성을 제시합니다.

- **Technical Details**: AnywhereDoor는 목표 분리(Objective Disentanglement), 트리거 모자이킹(Trigger Mosaicking), 전략적 배칭(Stragegic Batching) 세 가지 혁신의 중심으로 설계되었습니다. 목표 분리는 공격의 조합 복잡성을 줄이며, 트리거 모자이킹은 부분적으로 처리된 이미지에서도 트리거 효과를 유지할 수 있도록 보장합니다. 마지막으로, 전략적 배칭은 객체 클래스 간의 불균형을 해결해 공격 효과성을 높이는 역할을 합니다.

- **Performance Highlights**: 광범위한 실험을 통해 AnywhereDoor는 5가지 공격 시나리오에서 80% 이상의 공격 성공률(Attack Success Rate, ASR)을 달성했습니다. 특히 가장 복잡한 시나리오에서는 기존 방법 대비 약 80%의 ASR 향상을 기록했습니다. 더욱이, 깨끗한 샘플에 대한 성능 저하 없이 뛰어난 공격 효율성을 보여주었습니다.



### Towards Context-Rich Automated Biodiversity Assessments: Deriving AI-Powered Insights from Camera Trap Data (https://arxiv.org/abs/2411.14219)
Comments:
          32 Pages, 22 images

- **What's New**: 이 논문에서는 카메라 트랩(camera traps) 데이터를 활용하여 생태 보고(ecological reporting)를 개선할 수 있는 새로운 통합 접근법을 제안합니다. 기존의 자동 이미지 분석 방법이 효율성은 일정 부분 있으나, 생태 보존(outcomes)과 관련하여 필요한 맥락(contextual richness)을 제공하지 못하는 한계를 뛰어넘기 위해 깊은 학습 기반(deep learning-based) 비전(vision) 및 언어 모델(language models)을 결합합니다.

- **Technical Details**: 제안된 시스템은 두 단계로 구성됩니다. 첫째, YOLOv10-X(You Only Look Once) 모델을 사용하여 이미지 내에서 포획된 종(mammals and birds)을 지역화(localize)하고 분류(classify)합니다. 둘째, Phi-3.5-vision-instruct 모델이 YOLOv10-X에서 생성된 바운딩 박스(bounding box) 레이블을 읽어 해당 종을 식별하고, 분류가 어려운 객체들에 대한 한계를 극복합니다.

- **Performance Highlights**: 이 모델은 식생(vegetation) 유형과 시간대 등 넓은 변수를 감지하여 YOLO의 종 탐지 결과에 생태적 및 환경적 맥락을 제공합니다. 결합된 정보는 자연어 처리(natural language processing) 시스템을 통해 복잡한 질문에 대한 답변을 생성하며, 외부 정보로 응답을 풍부하게 하여 자동으로 구조화된 보고서를 생성합니다. 이로 인해, 생물 다양성(biodiversity) 관련 이해관계자들은 종의 개체 수, 분포, 행동 및 서식지 선택에 대한 심층적인 통찰을 얻을 수 있습니다.



### Evaluating the Robustness of Analogical Reasoning in Large Language Models (https://arxiv.org/abs/2411.14215)
Comments:
          31 pages, 13 figures. arXiv admin note: text overlap with arXiv:2402.08955

- **What's New**: 이 연구는 대규모 언어 모델(LLMs)의 유추 능력의 강인성을 평가하며, 이전 연구에서 주장된 바와는 달리 이러한 모델들이 대조 변형(task variants)에서 많은 경우 강인성을 결여하고 있음을 보여줍니다. 특히, 인간과 GPT 모델 간의 성능 차이를 강조하며, 수많은 변형에서 모델이 취약한 모습을 보이는지에 대한 실험을 진행하였습니다.

- **Technical Details**: 저자는 세 가지 유추 문제 영역에서 LLM의 내부 작동 방식에 대해 다루며, 각 영역에서 두 가지 변형을 사용한 실험을 설계했습니다. 특히, 문자-문자 유추, 숫자 행렬 문제, 그리고 이야기 기반 유추 문제를 평가하였으며, GPT 모델이 원본 문제와 변형 문제에서 일관된 성능을 보이지 않는다는 점을 발견했습니다.

- **Performance Highlights**: 단순한 문자-문자 유추 문제에서, 인간 참여자는 두 가지 변형에서 높은 성능을 유지했으나 GPT 모델의 성능은 급격히 감소했습니다. 이처럼 GPT 모델이 유추 성능을 발휘하는 데에 있어, 그 개념적인 강인성이 부족하다는 증거를 제공하며, AI 시스템의 인지적 능력을 평가할 때 정확성뿐만 아니라 강인성도 고려해야 함을 강조합니다.



### HARP: A Large-Scale Higher-Order Ambisonic Room Impulse Response Datas (https://arxiv.org/abs/2411.14207)
Comments:
          Submitted to ICASSP 2025 Workshop Dataset and code to be uploaded at: this https URL

- **What's New**: 이번 연구에서는 7차 Ambisonic Room Impulse Responses (HOA-RIRs) 데이터셋을 소개합니다. 이 데이터셋은 Image Source Method를 활용하여 생성되었으며, 정밀한 spatial audio reproduction을 가능하게 하여 몰입형 오디오 응용 프로그램에 필수적입니다. 논문에서는 새로운 마이크 배열 구성을 제안하며, 이는 전통적인 마이크 배열의 한계를 극복할 수 있도록 설계되었습니다.

- **Technical Details**: 제안된 데이터셋인 HARP는 64-채널 7차 HOA 구조를 통해 높은 공간 해상성으로 사운드 필드를 캡처합니다. Pyroomacoustics 라이브러리를 사용하여 구현하였으며, 각 마이크의 방향성을 구형 조화 함수로 분해하는 방법을 포함합니다. 이를 통해 7차까지의 HOA 구성 요소로 사운드 필드를 분해할 수 있습니다.

- **Performance Highlights**: 이 데이터셋은 연구자들이 공간 오디오, 방 acoustics, 그리고 머신러닝 업무에 필요한 필수 자원으로 기능하며, 예를 들어 방 매개변수 추정, 소음 제거, 몰입형 음악 생성 등 다양한 사용 사례에 기여할 수 있습니다. 제안된 방법론은 고해상도 공간 샘플링으로 인해 향상된 정확도와 효과적인 공간 오디오 렌더링을 제공하는데 중요한 역할을 합니다.



### Is this Generated Person Existed in Real-world? Fine-grained Detecting and Calibrating Abnormal Human-body (https://arxiv.org/abs/2411.14205)
Comments:
          16 pages, 14 figures

- **What's New**: 이 논문에서는 Fine-grained Human-body Abnormality Detection (FHAD)라는 새로운 작업을 제안하고, 이를 평가하기 위한 두 개의 고품질 데이터셋을 구축했습니다. 기존의 VLMs는 인체 사진에서 비정상적인 부분을 감지하는 데 있어 낮은 성능을 보였지만, HumanCalibrator라는 정밀 프레임워크를 제안하여 이러한 비정상성을 감지하고 수리하는 과정을 체계화했습니다. 이 연구는 인간의 신체 구조의 비정상성을 탐지하기 위한 새로운 접근 방식을 탐구하고, 해당 분야의 연구 커뮤니티에 기여하는 데 목적이 있습니다.

- **Technical Details**: FHAD의 목표는 주어진 시각적 콘텐츠에서 실제 인간과의 신체 구조 차이를 식별하는 것입니다. 이 방법은 두 가지를 출력해야 하는데, 첫째는 신체 부위 비정상성의 유형을 나타내는 시맨틱 플래그이고, 둘째는 비정상성이 있는 위치를 바운딩 박스 형태로 출력해야 합니다. HumanCalibrator는 이러한 비정상성을 정확히 식별하고 다른 시각적 내용을 보존하며 비정상적인 지역을 수리하는 정밀한 프레임워크입니다.

- **Performance Highlights**: 실험 결과, HumanCalibrator는 비정상 탐지에서 높은 정확도를 기록하며 시각적 비교에서 개선된 결과를 보였습니다. 이 모델은 사람의 신체 구조 내 비정상성을 정밀하게 식별하고 이를 효과적으로 수정하여 기존의 시각적 내용을 보존하는 데 성공했습니다. 이는 시각 콘텐츠 생성 모델에 있어 비정상성 탐지와 수정에서 무척 중요한 진전을 의미합니다.



### OpenScholar: Synthesizing Scientific Literature with Retrieval-augmented LMs (https://arxiv.org/abs/2411.14199)
- **What's New**: OpenScholar는 4500만 개의 오픈 액세스 논문에서 관련 정보를 검색하고, 인용 기반 응답을 합성하여 과학적 질문에 대답하는 검색 보강된 대형 언어 모델(LM)입니다. 새로운 벤치마크인 ScholarQABench를 소개하여 OpenScholar의 성능을 평가했으며, 이는 컴퓨터 과학, 물리학, 신경과학, 생물 의학을 포함하는 2967개의 전문가 작성 쿼리와 208개의 긴 형식 응답으로 구성되어 있습니다. OpenScholar는 GPT-4o보다 5% 더 높은 정확성을 달성했으며, 인용 정확도가 인간 전문가와 동등한 수준으로 향상되었습니다.

- **Technical Details**: OpenScholar는 OpenScholar-DataStore(OSDS)를 사용하여 4500만 개의 오픈 액세스 논문과 2억 3700만 개의 구문 임베딩을 포함합니다. 이 시스템은 관련 구문을 검색하고, 이를 기반으로 반복적인 자기 피드백 생성을 통해 응답의 출력을 정제하는 방식으로 작동합니다. OpenScholar는 ‘효율적인 8B 모델’로 훈련되어 특정 도메인에 최적화된 검색 및 합성 기능을 제공하며, 모델 간의 결합을 통해 전체적인 정확성을 개선할 수 있습니다.

- **Performance Highlights**: OpenScholar는 GPT-4o와 PaperQA2를 포함한 다양한 모델들보다 우수한 성능을 보였으며, 70%의 경우 전문가 작성 응답보다 더 나은 결과를 제공했습니다. ScholarQABench에서 OpenScholar는 인용 정확도와 정보 범위에서 높은 성과를 보였고, GPT-4o의 일부 기능을 향상시킬 수 있는 가능성을 보여주었습니다. OpenScholar의 효율적인 구조는 비용 절감 효과를 가져올 수 있으며, 연구에 실질적으로 기여할 수 있는 고품질 출력을 생성합니다.



### ComfyGI: Automatic Improvement of Image Generation Workflows (https://arxiv.org/abs/2411.14193)
- **What's New**: ComfyGI는 자동 이미지 생성의 효율성을 향상시키기 위해 설계된 새로운 접근 방식입니다. 이 방식은 인체 개입 없이도 이미지 생성 워크플로우를 자동으로 개선할 수 있는 기능을 제공합니다. ComfyGI는 이미지의 설명 및 미적 요소에 대해 더 높은 품질을 달성하도록 설계되었습니다.

- **Technical Details**: ComfyGI는 초기 설정으로 이미지를 생성하고, ImageReward 모델을 활용하여 이미지를 평가합니다. 이 접근 방식은 JSON 포맷으로 저장된 워크플로우에서 작은 변화를 적용하여 이미지 생성의 최적화를 이루어냅니다. 각 세대에서 가장 높은 ImageReward 점수를 얻는 변화를 선택하여 이전 설정에 적용하며, 이러한 과정을 통해 최적화된 이미지를 도출합니다.

- **Performance Highlights**: ComfyGI를 활용한 이미지 생성은 초기 설정 대비 약 50% 향상된 ImageReward 점수를 기록했습니다. 또한, 100명의 인원 평가를 통해 약 90%의 경우에서 참가자들이 ComfyGI로 개선된 이미지를 선호하는 것으로 나타났습니다. 이러한 결과는 ComfyGI의 성능이 기존의 이미지 생성 방식보다 월등히 뛰어난 것을 의미합니다.



### FoPru: Focal Pruning for Efficient Large Vision-Language Models (https://arxiv.org/abs/2411.14164)
Comments:
          11 pages, 7 figures

- **What's New**: 이 논문에서는 Focal Pruning (FoPru)이라는 새로운 방법론을 제안합니다. 이 방법은 시각 인코더로부터 유도된 주의기반의 토큰 중요성을 바탕으로 시각 토큰을 가지치기(pruning)하는 훈련이 필요 없는 방식으로, 기존 LVLMs에 손쉽게 적용 가능합니다. 또한, 두 가지 가지치기 전략인 Rank Pruning과 Row Pruning을 도입하여 보다 효과적인 토큰 선택을 진행합니다.

- **Technical Details**: FoPru는 세 가지 단계로 구성됩니다: 첫 번째로 토큰 중요성 계산을 위한 Token Significance 단계, 두 번째로 이러한 중요성 점수에 따라 시각 토큰을 가지치기 하는 Token Pruning 단계, 마지막으로 원래 위치에 따라 토큰을 재정렬하는 Token Reordering 단계입니다. 이러한 구조 덕분에 FoPru는 시각 토큰의 수를 대폭 줄이면서도 높은 성능을 유지할 수 있습니다.

- **Performance Highlights**: 실험 결과, FoPru는 다양한 LVLMs와 멀티모달 데이터셋에서 시각 토큰을 유의하게 줄이는 동시에 성능을 향상시키는 것을 확인하였습니다. 특히, 극단적인 토큰 보존 비율 0.2%에서 FoPru는 Ai2D 및 SQA 데이터셋에서 약 60%의 정확도를 유지합니다. 또한, 25%의 시각 토큰만 사용했을 때도 여러 데이터셋에서 1%의 마진 내 정확도를 기록하며, inference 효율성을 비약적으로 향상시킵니다.



### DrugGen: Advancing Drug Discovery with Large Language Models and Reinforcement Learning Feedback (https://arxiv.org/abs/2411.14157)
Comments:
          20 pages, 5 figures, 3 tables, and 7 supplementary files. To use the model, see this https URL

- **What's New**: 이 논문에서는 기존의 DrugGPT 모델을 기반으로 하여 개발된 새로운 모델 DrugGen을 소개합니다. DrugGen은 승인된 약물-타겟 상호작용 데이터셋으로 파인튜닝(fine-tuning)되어 최적화되었으며, 기존 모델보다 높은 성능을 보여줍니다. 특히, DrugGen은 100%의 화학적 유효성을 가지는 구조를 생성할 수 있고, 생물학적 타겟과의 결합 친화도에서 더 높은 예측 값을 기록했습니다.

- **Technical Details**: DrugGen은 드럭-리간 결합 친화도 예측을 위한 사전 훈련된(transformers) 모델 사용 및 맞춤형 유효성 평가 시스템을 통해 성능을 향상시키고 있습니다. 두 단계로 구성된 프로세스를 사용하여 퍼포먼스를 최적화하고, proximal policy optimization (PPO)을 활용하여 맞춤형 보상 시스템을 적용하였습니다. 이 과정에서 DrugGen은 각 타겟에 대해 30개의 고유한 분자를 생성하였고, 생성된 분자의 유효성은 99.90%에 달합니다.

- **Performance Highlights**: DrugGen은 8개의 단백질 타겟에 대해 평가된 결과, 생성된 분자의 유효성이 크게 향상되었습니다. 예를 들어, fatty acid-binding protein 5 (FABP5) 타겟에 대한 docking 시뮬레이션에서 DrugGen은 -9.537의 도킹 점수를 기록하여 기존의 Palmitic acid (-6.177)와 비교해 상위 성과를 보였습니다. 이 결과는 DrugGen이 높은 품질의 작은 분자를 생성할 수 있음을 보여주며, 약물 재포지셔닝 및 새로운 약물 형태 창출에 대한 가능성도 제시합니다.



### Differentiable SVD based on Moore-Penrose Pseudoinverse for Inverse Imaging Problems (https://arxiv.org/abs/2411.14141)
Comments:
          11 pages

- **What's New**: 본 논문에서는 중복된 특이값 발생 시 단점이 있는 비가역적인 특이값 분해(SVD)를 해결하기 위해 무어-펜로즈 유사역행렬을 기반으로 한 미분 가능한 SVD를 제안합니다. 이는 미분 가능성의 빈약함이라는 문제를 다루는 최초의 연구로, 비고유 시스템에서 발생하는 문제를 수학적으로 분석하였습니다.

- **Technical Details**: 논문에서는 저차원 정규화(low-rank regularization) 기반의 깊은 언롤링 네트워크(deep unrolling networks, DUNs)에 대한 미분 가능한 SVD의 필요성을 강조합니다. 특히, SVD의 비미분 가능성이 반복된 특이값으로 인해 발생하며, 이에 대한 해답으로 무어-펜로즈 유사역행렬을 효과적으로 활용합니다.

- **Performance Highlights**: 종합적인 실험 결과는 제안된 미분 가능한 SVD가 색상 이미지 압축 감지 및 동적 MRI 재구성 문제에서 수치적 불안정성을 효과적으로 극복하며 높은 정확성을 유지함을 보여줍니다. 최첨단 저차원 정규화 기반 DUN을 기준 모델로 설정하여, 기존 방법들보다 우수한 성능을 입증했습니다.



### GASP: Efficient Black-Box Generation of Adversarial Suffixes for Jailbreaking LLMs (https://arxiv.org/abs/2411.14133)
Comments:
          28 pages, 9 tables, 13 figures; under review at CVPR '25

- **What's New**: 이번 논문에서는 Generative Adversarial Suffix Prompter (GASP)라는 새로운 프레임워크를 소개합니다. GASP는 인간이 읽을 수 있는 프롬프트 생성과 Latent Bayesian Optimization (LBO)을 결합하여 사이버 공격(즉, jailbreak attack)을 통해 자연스러운 프롬프트를 생성할 수 있게 합니다. 이 프레임워크는 기존의 최적화 기반 방법의 한계를 극복하며, 자동으로 적대적인 서픽스(adversarial suffix)를 생성하여 LLM의 안전 장치를 피하는 데 주안점을 두고 있습니다.

- **Technical Details**: GASP는 Latent Bayesian Optimization을 활용하여 연속적인 임베딩 공간(embedding space)에서 적대적인 서픽스를 탐색하며, 이 과정에서 계산 비용을 크게 줄입니다. 프롬프트의 일관성을 유지하기 위해 목표 지향적인 반복적 개선(iterative refinement)을 통해 공격의 효율성을 극대화하는 전략을 사용합니다. 또한, Odds-Ratio Preference Optimization (ORPO)을 통해 모델의 파라미터를 조정하여 높은 성공률을 보이는 서픽스의 효과를 증가시키면서도 가독성을 유지하도록 설계되었습니다.

- **Performance Highlights**: GASP의 실험 결과는 자연스러운 jailbreak 프롬프트를 생성할 수 있음을 입증했으며, 공격 성공률이 유의미하게 향상되고, 교육 시간이 단축되며, 추론 속도가 개선되었습니다. 특히, GASP는 블랙박스 환경에서 운영될 때도 강력한 성능을 발휘하며, 다양한 LLM 아키텍처 및 프롬프트 유형에 대해 효율적인 평가를 지원하고 있습니다. 이러한 성과는 LLM의 보안을 높이고, 더 안전한 모델 개발에 기여할 것으로 기대됩니다.



### Umbrella Reinforcement Learning -- computationally efficient tool for hard non-linear problems (https://arxiv.org/abs/2411.14117)
- **What's New**: 본 논문에서는 강화 학습(Deep Reinforcement Learning, DRL)의 어려운 비선형 문제를 해결하기 위한 새로운 계산 효율적인 접근 방식을 제안합니다. 이 방법은 계산 물리학/화학에서의 우산 샘플링(Umbrella Sampling)과 최적 제어(optimal control) 방법을 결합하였습니다. 제안된 접근법은 정책 그래디언트(policy gradient)를 사용하는 신경망(neural network)을 기반으로 하며, 희소 보상(sparse reward) 및 상태 함정(state traps)을 다룰 때 기존 최첨단 알고리즘에 비해 뛰어난 성능을 보입니다.

- **Technical Details**: 강화 학습(RL)에서의 주요 문제점은 보상이 희소하거나 지연될 때, 그리고 특정 종착 상태가 부족할 때 발생합니다. 이러한 문제를 극복하기 위해, 본 연구에서는 연속적인 에이전트 집합(ensemble)으로 구성된 방식을 채택했습니다. 각 에이전트는 상태 변수를 포함하며, 이러한 집합은 무한대의 에이전트에 해당하는 분포 함수를 구축하여 적절한 탐색-활용 균형을 제공합니다. 이를 통해 전통적인 RL의 한계를 뛰어넘으려는 노력을 보여줍니다.

- **Performance Highlights**: 제안된 방법인 Umbrella RL은 Multi-value Mountain Car와 StandUp 문제와 같은 난이도 높은 문제들에 적용되었으며, 실험 결과 기존 강화 학습 알고리즘들에 비해 뛰어난 성능 개선을 나타냈습니다. 이 효율적인 접근 방식은 다양한 정보 처리 시스템에 적용될 수 있을 뿐만 아니라, 계산적으로 더 유리한 결과를 제공합니다. 최종적으로, 본 연구는 어려운 RL 문제 해결을 위한 새로운 관점을 제시하며, 향후 연구에 대한 기초를 마련합니다.



### MetaCropFollow: Few-Shot Adaptation with Meta-Learning for Under-Canopy Navigation (https://arxiv.org/abs/2411.14092)
- **What's New**: 이 논문은 농업 로봇의 자율 항법에서 도메인 이동 문제를 해결하기 위해 메타 학습(Meta-Learning) 접근 방식을 탐구합니다. 농작물 사이의 좁은 간격, GPS 신호 왜곡, 과도한 혼잡 등 다양한 도전 과제가 존재하는 상황에서 기존 시스템 대비 높은 적응성을 갖춘 베이스 러너(base-learner)를 교육시킵니다. 새로운 환경에 신속하게 적응할 수 있는 능력을 부각시키며, 한정된 데이터로도 강력한 내비게이션 성능을 보여줍니다.

- **Technical Details**: 연구에서는 28273개의 이미지를 사용하여 MAML(Model-Agnostic Meta-Learning) 기반의 학습 시스템을 구현하였습니다. 주요 키포인트는 사라지는 점(vanishing point), x축과의 왼쪽 교차점 및 오른쪽 교차점으로 구성되어, 이를 통해 농작물 사이의 통과 가능 영역을 정의합니다. MAML 및 그 변형(MAML++와 ANIL)은 환경에 대한 빠른 적응 능력을 평가하는 데 사용되었으며, 이는 표준 학습 접근법보다 뛰어난 성능을 입증하였습니다.

- **Performance Highlights**: 실험 평가 결과, MAML++ 시스템은 다른 시스템들에 비해 적응력이 뛰어난 것으로 나타났습니다. 예를 들어, All-Season 분할에 대해 MAML++는 평균 3.9의 손실(loss)을 기록하여 비슷한 조건의 Non-MAML 모델(5.75)보다 현저히 낮은 결과를 보였습니다. ANIL++도 4.3의 손실로 이와 유사하게 성능을 개선하였으며, 이러한 결과는 농업 환경에 대한 내비게이션 성능이 도메인 이동 대응력에 크게 의존함을 시사합니다.



### Multi LoRA Meets Vision: Merging multiple adapters to create a multi task mod (https://arxiv.org/abs/2411.14064)
- **What's New**: 이 논문에서는 여러 개의 LoRA (Low-Rank Adaptation) 어댑터를 컴퓨터 비전 작업에서 훈련한 후, 이를 결합하여 성능 손실 없이 추론 단계에서 사용할 수 있는지를 조사하였습니다. 일반적으로 사용되는 PEFT (Parameter Efficient Finetuning) 방법을 통해 다양한 작업에 대해 결합된 멀티태스크 모델을 만들 수 있는 가능성을 제시합니다. 이를 통해 추가적인 재훈련 없이도 추론 시간을 단축할 수 있습니다.

- **Technical Details**: 연구에서는 여섯 가지 서로 다른 작업에서 훈련된 어댑터들을 결합하고, 이들을 하나의 모델에서 사용할 때의 성능을 평가했습니다. 특히, 고정된 백본 모델에서 헤드를 미세 조정한 모델과 성능을 비교하여, 어댑터를 간단히 결합하기만 해도 멀티태스크 모델을 생성할 수 있음을 보여줍니다. 실험에서는 최대 세 개의 어댑터를 결합하여 사용하였습니다.

- **Performance Highlights**: 어댑터의 결합이 데이터의 유사성과 작업에 따라 헤드 미세 조정보다 성능이 우수할 수 있음을 관찰하였습니다. 특히, 유사한 데이터셋으로 훈련된 모델보다 서로 다른 데이터셋으로 훈련된 LoRA가 더 나은 성능을 보이는 경향이 발견되었습니다. 이는 멀티태스크 모델의 효율성을 높이고 다양한 작업에서의 응용 가능성을 확장할 수 있음을示합니다.



### MMGenBench: Evaluating the Limits of LMMs from the Text-to-Image Generation Perspectiv (https://arxiv.org/abs/2411.14062)
Comments:
          This project is available at: this https URL

- **What's New**: 이 논문은 Large Multimodal Models (LMMs)의 이미지 생성을 평가하기 위한 직관적인 자동 평가 파이프라인을 제안합니다. 기존의 벤치마크가 주로 이미지 이해에 중점을 두었다면, 본 연구는 LMM의 이미지 생성 능력을 포함하여 보다 포괄적인 평가에 중점을 두고 있습니다. 또한, MMGenBench-Test와 MMGenBench-Domain이라는 새로운 벤치마크를 도입하여 13개의 다양한 이미지 패턴을 평가하고, 생성 이미지 도메인에서 LMM의 성능을 분석합니다.

- **Technical Details**: 제안된 파이프라인은 LMM이 주어진 입력 이미지를 기반으로 이미지 프롬프트(image-prompt)를 생성하도록 요구합니다. 이후, 이 프롬프트를 바탕으로 텍스트-투-이미지(text-to-image) 생성 모델을 이용해 새 이미지를 생성하고, 최종적으로 원본 이미지와 생성된 이미지를 비교하여 성능을 평가합니다. 이 과정은 이미지 이해 및 설명에 대한 LMM의 성능을 평가하기 위해 세 가지 구성 요소로 이루어져 있습니다: 이미지 프롬프트 생성, 새로운 이미지 생성 및 정량적 메트릭 계산.

- **Performance Highlights**: 50개 이상의 인기 있는 LMM을 대상으로 한 평가 결과, 기존 벤치마크에서는 우수한 성과를 보이는 LMM이 기본적인 이미지 이해 및 설명 작업을 적절히 수행하지 못한다는 사실이 드러났습니다. 이 연구는 LMM의 성능 개선 가능성을 강조하며, 생성 이미지 도메인에서 더 나은 모델 최적화를 위한 방향성을 제시합니다. MMGenBench를 통해 LMM의 성능을 다양한 도메인에서 효율적으로 평가할 수 있는 유연하고 확장 가능한 벤치마킹 도구를 제공합니다.



### FunctionChat-Bench: Comprehensive Evaluation of Language Models' Generative Capabilities in Korean Tool-use Dialogs (https://arxiv.org/abs/2411.14054)
Comments:
          8 pages

- **What's New**: 이 연구는 언어 모델의 도구 사용 대화에서 생성 능력을 조사하였으며, 모델의 출력 결과를 Tool Call, Answer Completion, Slot Question, Relevance Detection의 네 가지 유형으로 분류합니다. 이러한 평가 항목을 기반으로 한 FunctionChat-Bench라는 새로운 벤치마크를 도입하여 700개 평가 항목과 자동 평가 프로그램을 제공합니다. 또한, 단일 턴의 Tool Call 시나리오에서 높은 정확성을 보이는 언어 모델이 다중 턴 환경에서의 생성 성능에서도 우수한 결과를 보이지는 않는다는 점을 주장합니다.

- **Technical Details**: FunctionChat-Bench는 단일 호출 데이터셋과 대화 데이터셋의 두 가지 하위 집합으로 구성되어, 도구 호출 출력뿐만 아니라 사용자와의 대화 기능도 종합적으로 평가합니다. 각 하위 집합은 언어 모델이 적절한 도구 기능을 선택하고 정보 추출하여 인수를 적절히 생성할 수 있는지를 평가합니다. 언어 모델의 평가 방법으로는 정확한 일치 접근법과 코사인 유사도가 사용되며, 이는 모델의 성능을 객관적으로 판별하는 데 도움을 줍니다.

- **Performance Highlights**: 연구 결과, FunctionChat-Bench를 통해 수행된 여러 언어 모델에 대한 평가에서, 단일 호출 시나리오에서는 높은 성능을 보이는 반면, 다중 턴 대화에서는 사용자의 요구를 적절히 처리하는 능력이 제한적임을 발견했습니다. 이는 언어 모델이 도구 호출 메시지 생성 능력뿐만 아니라, 사용자를 참여시키는 대화 메시지를 효과적으로 생성할 수 있어야 한다는 것을 강조합니다. 종합적으로 이 벤치마크는 언어 모델들의 도구 활용 능력을 보다 완벽하게 평가할 수 있는 기반을 제공합니다.



### Forecasting Future International Events: A Reliable Dataset for Text-Based Event Modeling (https://arxiv.org/abs/2411.14042)
Comments:
          EMNLP 2024 Findings

- **What's New**: 본 논문에서는 WORLDREP (WORLD Relationship and Event Prediction)라는 새로운 데이터셋을 소개합니다. 이 데이터셋은 대규모 언어 모델(LLM)의 고급 추론 능력을 활용하여 기존 데이터셋의 한계를 극복하고자 설계되었습니다. WORLDREP는 정치 과학 분야의 전문가들에 의해 엄격하게 검증된 고품질 스코어링 레이블을 특징으로 하며, 국제 사건 예측 작업에 대한 효과성을 입증하는 다양한 실험을 통해 그 품질과 유용성을 보여줍니다.

- **Technical Details**: WORLDREP의 데이터셋 구축 과정은 다자 관계의 포착과 관계 라벨링의 정확성을 증대시키는 두 가지 주요 개선점을 포함하고 있습니다. 데이터 수집 단계에서는 약 44,706개의 뉴스 기사를 수집하여 각 기사를 단일 사건으로 취급하고, LLM의 프롬프트 설계를 통해 관련 국가를 정확하게 추출하는 자가 수정 메커니즘을 통합했습니다. 또한, 각 뉴스 기사에서 생성된 쌍의 관계를 평가하고 스코어를 부여하는 세밀한 방법론을 매핑하여 더욱 풍부한 데이터를 생성했습니다.

- **Performance Highlights**: WORLDREP의 품질을 검증하기 위해 여러 가지 실험을 진행했으며, 전문가의 라벨과의 일치성을 평가했습니다. 선정된 1,030개의 기사에 대해 전문가들이 수작업으로 라벨링을 진행하고, 우리의 레이블과 일치하는 정도를 분석하여 신뢰성과 일관성을 보장했습니다. 이 결과는 데이터셋의 높은 성능과 함께 예측 모델 교육의 성과를 크게 향상시킨 것으로 나타나며, 데이터셋과 전체 자동화 소스 코드를 공개하여 향후 연구와 개발을 지원하고자 합니다.



### Uterine Ultrasound Image Captioning Using Deep Learning Techniques (https://arxiv.org/abs/2411.14039)
- **What's New**: 이 논문은 자궁 초음파 영상의 캡션 생성을 위한 딥러닝 기반 방법을 제안합니다. 기존의 문제점인 해석의 어려움을 극복하기 위해, 합성곱 신경망(Convolutional Neural Networks)과 양방향 게이트 순환 유닛(Bidirectional Gated Recurrent Unit) 네트워크를 통합한 하이브리드 모델을 구축했습니다. 이는 의료 전문인들이 신속하고 정확하게 진단을 내릴 수 있도록 돕는 것을 목표로 합니다.

- **Technical Details**: 제안된 시스템은 먼저 자궁 초음파 영상의 대규모 데이터세트를 수집하고 이를 전문가의 주석으로 세심하게 주석 처리했습니다. 주요 특징 추출 단계에서는 사전 훈련된 CNN 모델인 Inception V3와 DenseNet201을 활용하여 이미지에서 세부 특징 벡터를 추출하였고, 텍스트 데이터는 수치 표현으로 변환하여 이미지 특징과 매끄럽게 결합하였습니다. 최종적으로 CNN-BiGRU 모델을 통해 자궁 초음파 이미지에 대한 설명적 캡션을 생성하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 CNN-BiGRU 모델은 BLEU 및 ROUGE 점수에서 기존의 방법들에 비해 우수한 성능을 보였습니다. 이 모델은 초음파 이미지를 정확하고 유익하게 설명할 수 있어, 여성 건강 분야에서의 진단 정확성을 향상시킬 잠재력이 있습니다. 이 연구는 자궁 초음파 이미지 해석을 개선하여 궁극적으로 환자 치료에 기여하고자 하는 바탕이 됩니다.



### Assessing data-driven predictions of band gap and electrical conductivity for transparent conducting materials (https://arxiv.org/abs/2411.14034)
- **What's New**: 본 연구에서는 투명 전도체(TCMs) 발견을 가속화하기 위한 데이터 기반 프레임워크를 제안합니다. 기존의 실험적 데이터의 부족을 해결하기 위해, 우리는 실험적으로 보고된 TCM의 전기 전도도 및 에너지 밴드갭에 대한 고유한 데이터베이스를 생성하고 검증하였습니다. 이 연구를 통해 ML이 새롭고 중요한 물질 클래스를 발견하는 데 실질적인 유용성을 발휘할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 연구에서는 스토이키오메트리(stoichiometry)만으로 속성을 예측하는 최첨단( SOTA ) ML 모델을 평가하였습니다. 실험적으로 보고된 55가지 화학 조성 리스트를 활용하여 ML의 효과성을 실증하였습니다. 또한, 맞춤형 평가 프레임워크를 통해 ML 모델이 이전에 보지 못한 TCM 클래스를 식별할 수 있는 능력을 평가합니다.

- **Performance Highlights**: 데이터기반 접근법에 따라 실험적으로 보고된 TCM의 전기 전도도 및 에너지 밴드갭 데이터셋을 두 개 생성하고 평가하였습니다. 본 연구에서 제안한 ML 모델은 그동안 간과되었던 새로운 물질 후보를 조명할 수 있는 잠재력을 보였습니다. 결과적으로, 우리는 ML이 TCM 특성을 지닌 물질들을 신속하게 식별하는 데 기여할 수 있다는 것을 실제 사례를 통해 입증하였습니다.



### Mirror Target YOLO: An Improved YOLOv8 Method with Indirect Vision for Heritage Buildings Fire Detection (https://arxiv.org/abs/2411.13997)
- **What's New**: MITA-YOLO라는 새로운 화재 감지 방법이 제안되었습니다. 이 모델은 간접 시각(Indirect Vision)을 활용하여 카메라 수를 줄이고 화재 감지 성능을 향상시키는 데 중점을 두고 있습니다. 특히, 미러(mirror) 각도를 조절하여 가시성이 제한된 공간에서도 지표를 정렬하고 타겟 감시 구역에 맞춰 감지를 수행합니다.

- **Technical Details**: 본 연구는 타겟-마스크(Target-Mask) 모듈을 도입하여 이미지 내에서 간접 시각 영역을 자동으로 식별하고 자동 분리하여 비타겟 영역을 필터링합니다. 이를 통해 모델은 화재 위험 구역에 대한 관리자들의 전문성을 물려받아 화재 모니터링의 초점을 높이고 간섭에 저항하는 성능을 갖출 수 있도록 합니다. MITA-YOLO는 800장의 이미지로 구성된 화재 데이터셋을 이용하여 평가되었으며, 성능은 기존의 다른 모델들보다 우수한 것으로 나타났습니다.

- **Performance Highlights**: MITA-YOLO는 다른 최신 모델들과 비교했을 때 mAP50에서 3.7% 향상된 성능을 보였으며, 재현율이 3% 증가하는 결과를 얻었습니다. 이 방법은 간접 시각을 활용하여 화재 탐지에서 최고 성능을 달성하며, 구조물에 미치는 영향을 최소화하는 동시에 감지 정확도를 높이는 데 기여합니다.



### Safety Without Semantic Disruptions: Editing-free Safe Image Generation via Context-preserving Dual Latent Reconstruction (https://arxiv.org/abs/2411.13982)
Comments:
          This research is supported by the NISDRG project #20100007, funded by the Australian Government

- **What's New**: 이 연구는 다중 모달 생성 모델의 안전한 콘텐츠 생성을 보장하는 과제를 다룬다. 기존의 모델 수정 기술은 개념 간의 의미적인 관계를 손상시켜 불필요한 왜곡을 초래할 수 있다. 이러한 문제를 해결하기 위해, 저자들은 안전-컨텍스트 임베딩(safety-context embeddings)과 이중 재구성 과정(dual reconstruction process)을 활용하여 수정 없는 안전한 이미지 생성을 위한 모듈형 솔루션을 제안한다.

- **Technical Details**: 제안된 방법은 레이튼 공간(latent space)에서 조정 가능한 가중치 합(tunable weighted summation)을 통해 전 세계적인 시각적 맥락을 유지하면서 모델의 구조적 무결성을 보존한다. 이 연구는 안전한 이미지 생성을 위한 선진 방법론으로, 모델의 안전성을 조절 가능한 방식으로 제어할 수 있는 기능을 제공한다. 이를 통해 위험한 콘텐츠 생성에 대한 여러 접근 방식을 통합하여 각기 다른 안전 기준을 충족할 수 있게 한다.

- **Performance Highlights**: 이 방법은 안전한 이미지 생성 벤치마크에서 최첨단 성능을 달성하며, 감시를 통한 모델 안전성과 검열(censorship) 간의 균형을 제시한다. 지원하는 두 개의 모듈형 적절하지 않은 입력 감지기를 통해 모든 비도덕적인 입력을 안전한 의미적 영역으로 효율적으로 유도할 수 있다. 이러한 접근 방식의 결과로, 이 연구는 고급 텍스트-이미지 생성 모델에 대한 책임감 있는 안전성을 구현하는 방법을 제안하고 있다.



### On the Fairness, Diversity and Reliability of Text-to-Image Generative Models (https://arxiv.org/abs/2411.13981)
Comments:
          This research is supported by the NISDRG project #20100007, funded by the Australian Government

- **What's New**: 이 논문에서는 텍스트-이미지 모델의 신뢰성과 공정성을 평가하기 위한 새로운 성능 평가 프레임워크를 제안합니다. 제안된 방법은 'semantic' perturbations를 통해 모델의 취약점을 분석하고, 신뢰성이 낮은 입력을 특定하는 데 중점을 둡니다. 이러한 접근은 생성적 다양성과 공정성을 더 깊이 이해할 수 있도록 돕습니다. 또한, 논문의 방법론은 편향이 주입된 모델의 감지와 편향 출처 추적을 위한 기초를 마련합니다.

- **Technical Details**: 제안된 방법론은 임베딩 공간에서 글로벌 및 로컬 차원의 'semantic' perturbations를 적용하여 텍스트-이미지 모델의 신뢰성을 정량화합니다. 이를 통해 맥락화된 프롬프트의 행동과 각 인코딩된 토큰이 생성에 미치는 영향을 분석합니다. 이 과정에서 생성된 이미지의 중요한 변화는 모델의 신뢰성이 낮음을 나타내며, 그 결과 공정성과 다양성 평가의 필요성을 강조합니다.

- **Performance Highlights**: 제안된 방법은 고의적으로 편향된 텍스트-이미지 모델을 효과적으로 감지하는 데 유효하며, 생성적 공정성과 다양성 평가를 통해 편향 트리거를 검색하고 출처를 식별하는 데 기여합니다. 이 연구는 텍스트-이미지 모델의 공정성과 신뢰성을 평가하는 다양한 메트릭스를 제시하며, 이러한 평가가 모델 행동을 더 잘 이해하는 데 어떻게 기여하는지를 설명합니다.



### FedRAV: Hierarchically Federated Region-Learning for Traffic Object Classification of Autonomous Vehicles (https://arxiv.org/abs/2411.13979)
Comments:
          8 pages, 4 figures

- **What's New**: 이 논문에서는 자율주행 차량을 위한 새로운 계층적 연합 지역 학습 프레임워크(FedRAV)를 제안합니다. 이 프레임워크는 차량이 속한 큰 지역을 정의된 지역 간 거리(distance)에 따라 하위 지역으로 적응적으로 나누어, 개인화된 차량 모델 및 지역 모델을 달성합니다. 이를 통해 차량 간의 비독립적이며 동일하게 분포되지 않은(Non-IID) 데이터 문제를 해결하고, 자율주행 모델 학습을 효율적으로 수행할 수 있습니다.

- **Technical Details**: FedRAV는 두 단계의 프레임워크로 구성되어 있으며, 적응적 분할 메커니즘을 통해 관심 영역을 다양한 크기의 하위 지역으로 나눕니다. 또한, 하이퍼네트워크(hypernetworks)를 사용하여 각 차량에 대한 전용 마스크 벡터를 학습하고, 이를 통해 차량 모델을 개인화합니다. 이 방식은 같은 지역 내 차량들이 공유한 모델의 가중치를 조정하여 유용한 모델은 채택하고, 불필요한 모델은 배제하도록 합니다.

- **Performance Highlights**: 실험 결과는 FedRAV가 기존의 연합 학습 알고리즘들보다 뛰어난 성능을 보여주며, 정확도를 최소 3.69% 향상시키는 것을 입증합니다. 또한, 이 논문은 세 가지 실제 자율주행 데이터셋에 대한 평가를 실시하였으며, FedRAV의 소스 코드를 공개하여 커뮤니티에 기여하였습니다.



### A Dataset for Evaluating Online Anomaly Detection Approaches for Discrete Multivariate Time Series (https://arxiv.org/abs/2411.13951)
- **What's New**: 본 연구에서는 멀티베리어트(multivariate) 시계열(anomaly detection)에서 이상 탐지 알고리즘을 벤치마킹하기 위한 고품질 데이터셋의 부족이라는 문제를 해결하기 위해 새로운 PATH 데이터셋을 제안합니다. 이 데이터셋은 자동차 동력 전달장치의 실질적인 동작을 반영하여 생성되었으며, 다양한 작업을 위해 오염된 데이터와 깨끗한 데이터를 구분한 여러 버전을 제공합니다.

- **Technical Details**: PATH 데이터셋은 각 시퀀스가 여러 명목적 또는 이상 서브 시퀀스로 구성된 단일 멀티베리어트 시계열을 포함하여, 연속 시퀀스 및 비연속 시퀀스 탐지 작업을 지원하는 설계로 구성됩니다. 이 논문에서 다루는 주요 기술적 요소는 시스템의 가변 상태(variable state)의 데이터 수집이며, 이를 통해 이상 탐지 작업에서의 상황 변화에 따른 문맥적 변화 분리를 다룹니다.

- **Performance Highlights**: 기초 실험 결과는 준감독(semi-supervised) 버전 데이터셋에서 훈련된 방법들이 비감독(unsupervised) 방법들보다 성능이 뛰어난 것을 보여주어, 오염된 데이터에 더 강건한 접근 방식의 필요성을 강조합니다. PATH 데이터셋은 새로운 벤치마크를 제공하며, 따라서 향후 이상 탐지 연구에 중요한 기여를 할 것으로 기대됩니다.



### Separable Mixture of Low-Rank Adaptation for Continual Visual Instruction Tuning (https://arxiv.org/abs/2411.13949)
- **What's New**: 이 연구는 Multimodal Large Language Models (MLLMs)의 지속적 시각 지시 조정(Continual Visual Instruction Tuning, CVIT)에서 나타나는 이중적인 재기억 소멸(catastrophic forgetting) 문제를 파악하고 이를 해결하기 위한 Separable Mixture of Low-Rank Adaptation (SMoLoRA) 프레임워크를 제안합니다. 전통적인 지속적 학습 패러다임에서는 이 문제를 다루지 않았으나, 본 연구는 시각 이해(visual understanding)와 지시 이행(instruction following) 능력의 동시에 저하되는 특수한 상황을 강조합니다.

- **Technical Details**: SMoLoRA는 두 가지 모듈, 즉 시각 이해 모듈과 지시 이행 모듈을 활용하는 분리 가능한 라우팅(separable routing)을 통해 이중 재기억 소멸 문제를 해결합니다. 이러한 접근방식은 다음의 세 가지 요소로 구성되어 있습니다: 1) 각 작업의 특정 요구에 맞는 LoRA 블록을 동적으로 선택하여 시각 및 지시 수행 능력을 보존 및 개선하며, 2) 두 도메인에서의 독립적 적응을 통해 정보를 생성함으로써 과거 지식을 보존하고, 3) 새로운 과제가 기존 지식에 간섭하지 않도록 만듭니다.

- **Performance Highlights**: 실험 결과, SMoLoRA는 기존 방법들보다 이중 재기억 소멸 문제를 완화하고, 새로운 작업에 대한 일반화 능력을 증대시키며, 다양한 작업에 따른 지시에 대한 강인성을 증명했습니다. 새로운 CVIT 벤치마크를 통해 본 모델의 성능 평가가 수행되었으며, 이는 모델이 과거 지식을 잃지 않으면서 여러 작업을 수행할 수 있는 능력을 강조하는 데 중점을 두었습니다.



### LLMs as Continuous Learners: Improving the Reproduction of Defective Code in Software Issues (https://arxiv.org/abs/2411.13941)
- **What's New**: 이번 논문에서는 EvoCoder라는 다중 에이전트 지속 학습 프레임워크를 제안합니다. 이는 코드 문제 재현(issue reproduction)에 초점을 맞추고 있으며, 기존의 접근방식이 흔하고 널리 퍼진 오류에만 대응하는 데 그치고 있는 한계를 극복합니다. EvoCoder는 LLM이 이전에 해결된 문제로부터 계속 학습할 수 있도록 하는 반영 메커니즘(reflection mechanism)을 채택하고 있습니다.

- **Technical Details**: 이 모델은 경험의 부풀음을 방지하기 위해 새로운 계층적 경험 저장소(hierarchical experience pool)를 도입하여, 공통적인 경험과 특정 리포지토리의 경험을 적응적으로 업데이트할 수 있도록 합니다. 이러한 구조를 통해 EvoCoder는 변화하는 고유 오류에 더 잘 적응하고, 지속적으로 문제 해결 전략을 정제할 수 있습니다.

- **Performance Highlights**: 실험 결과, EvoCoder는 기존의 최신 기술(SOTA) 방법들보다 20% 향상된 문제 재현율을 보였습니다. 또한, 코드 재현 메커니즘을 통합함으로써 기존 문제 해결 파이프라인의 전반적인 정확성을 크게 향상시켰습니다.



### Learning to Cooperate with Humans using Generative Agents (https://arxiv.org/abs/2411.13934)
- **What's New**: 이번 연구에서는 인간과의 제로샷(zero-shot) 협업을 개선하기 위한 새로운 방법론인 GAMMA(Generative Agent Modeling for Multi-agent Adaptation)를 제안합니다. 기존의 인공지능(AI) 에이전트는 인간과의 협업에서 다양한 전략을 제대로 다루지 못하는 문제를 지적하고, 이를 해결하기 위해 인간 파트너의 생성 모델을 학습합니다. 이 모델은 에이전트의 고유한 전략, 의도, 경험 또는 스타일을 인코딩하는 잠재 변수 표현을 학습하여, 각기 다른 파트너를 생성할 수 있습니다.

- **Technical Details**: GAMMA는 변분 오토인코더(Variational Autoencoder, VAE)를 사용하여 인간 또는 합성 데이터에서 수집된 협업 궤적을 기반으로 파트너 행동의 생성 모델을 학습합니다. 이 모델은 훈련 데이터의 특성을 활용해 다양한 파트너 전략을 생성할 수 있으며, 모델 학습에 필요한 데이터 양을 감소시키고 인간 행동을 효과적으로 커버하여 AI 에이전트의 적응력을 높입니다. 또한, 인간 데이터를 효율적으로 통합하는 휴먼-어댑티브 샘플링 기법을 제안해 훈련 과정에서 실제 인간 파트너와의 조화를 이루게끔 조정합니다.

- **Performance Highlights**: GAMMA는 Overcooked 게임을 통해 실험되었으며, 실제 인간 플레이어와의 비교에서 기존의 최신 제로샷 협업 방법들에 비해 성능이 일관되게 향상되는 결과를 나타냈습니다. 본 연구에서는 최초로 인간 데이터의 효과성을 강조하며, 모델이 복잡한 환경에서도 뛰어난 성능을 발휘함을 보여줍니다. 실험 결과, GAMMA는 합성 에이전트 데이터뿐만 아니라 인간 데이터를 활용하여도, 실제 인간과의 협업에서 월등한 성능 개선을 이루어냈습니다.



### Split Federated Learning Over Heterogeneous Edge Devices: Algorithm and Optimization (https://arxiv.org/abs/2411.13907)
- **What's New**: 이번 연구에서는 Heterogeneous Split Federated Learning (HSFL) 프레임워크를 제안합니다. 이 프레임워크는 리소스가 제한된 클라이언트들이 서로 다른 컷 레이어를 사용하여 모델을 병렬로 훈련할 수 있도록 합니다. 이는 비슷한 데이터를 가진 클라이언트 간의 통신 지연을 최소화하며, 데이터의 프라이버시를 보호합니다.

- **Technical Details**: HSFL은 각 클라이언트가 자신에게 적합한 컷 레이어를 선택하여 계산 및 전송 리소스를 최적화하도록 돕습니다. 이 과정에서 Sample Average Approximation (SAA), Genetic Algorithm (GA), Lagrangian relaxation, Branch and Bound (B&B) 방법을 결합한 리소스 할당 알고리즘을 개발했습니다. 최적화 문제는 총 훈련 지연 시간을 최소화하도록 설계되었습니다.

- **Performance Highlights**: 시뮬레이션 결과, HSFL은 이종 단말기에서 비독립적이고 동일 분포가 아닌 데이터(non-iid data) 환경에서도 다른 프레임워크에 비해 더 빠른 수렴 속도와 높은 모델 정확도를 보여줍니다. 최적화 알고리즘은 지연 시간을 줄이는 데 있어 기존 방법보다 우수한 성능을 나타냈습니다.



### AmpliNetECG12: A lightweight SoftMax-based relativistic amplitude amplification architecture for 12 lead ECG classification (https://arxiv.org/abs/2411.13903)
- **What's New**: 본 연구에서는 심박 disorder를 신속하고 정확하게 진단하는 새로운 딥러닝 아키텍처인 AmpliNetECG12를 제안합니다. 이를 위해 ECG의 다양한 리드에서 커널 가중치 공유를 포함하는 Convolutional Neural Network(CNN) 아키텍처를 사용하고, aSoftMax라는 새로운 활성화 함수를 개발하여 ECG의 변동성을 더 잘 표시합니다. 이 혁신적인 방법은 모델의 복잡성을 줄이고 훈련 가능한 매개변수를 최소화하는 데 중점을 둡니다.

- **Technical Details**: AmpliNetECG12는 12-lead ECG의 글로벌 특징을 일반화하여 정확한 심장 질환 진단을 제공합니다. aSoftMax는 예측 정확성을 향상시키는데 중요한 역할을 하며, 이로 인해 CNN의 성능이 배가됩니다. 모델은 280,000개의 훈련 가능한 매개변수를 가지고 있으며, 이는 경량화된 효율적인 설계를 잘 보여줍니다.

- **Performance Highlights**: AmpliNetECG12는 arrhythmia 분류를 위해 CPSC2018 데이터셋과 함께 사용될 때 탁월한 예측 능력을 나타냅니다. 심장 disorder 진단에서 84%의 우수한 정확도를 달성하고, F1-score는 80.71%, ROC-AUC 점수는 96.00%입니다. aSoftMax의 확률적 특성은 예측 정확도를 높일 뿐만 아니라 모델의 해석 가능성을 증대시킵니다.



### PIORS: Personalized Intelligent Outpatient Reception based on Large Language Model with Multi-Agents Medical Scenario Simulation (https://arxiv.org/abs/2411.13902)
- **What's New**: 중국의 외래 진료 시스템에서의 부담을 줄이기 위해 Personalized Intelligent Outpatient Reception System (PIORS)을 제안합니다. 이 시스템은 LLM 기반의 간호사와 병원 정보 시스템(HIS)의 협업을 통해 개선된 외래 접수 서비스를 제공합니다. 새로운 의료 대화 데이터 생성 프레임워크인 SFMSS를 통해 LLM의 실제 의료 환경 적응성을 높이고 있습니다.

- **Technical Details**: PIORS는 환자, 간호사, 정보 보조자, 임상 의사 네 가지 참여자를 포함하며, HIS와 통합됩니다. PIORS-Nurse는 환자의 의료 정보를 수집하고 질문에 답변하며, HospInfo-Assistant와 협력하여 개인화된 서비스를 제공합니다. SFMSS는 실제 환자 상호작용을 모의하고 서비스 흐름을 조절하여 LLM이 진료 환경에 맞게 조정될 수 있도록 지원합니다.

- **Performance Highlights**: 자동 및 인간 평가에서 15명의 사용자와 15명의 임상 전문가를 대상으로 PIORS의 효과성을 평가하였습니다. 결과적으로 PIORS-Nurse는 정확성과 정보 수집 능력 면에서 최신 모델 GPT-4o를 포함한 모든 기준 모델을 초월하며, 사용자 평가에서 81% 이상의 승리 또는 동률 비율을 기록하여 실제 시나리오에서 더 나은 경험을 제공합니다.



### When Online Algorithms Influence the Environment: A Dynamical Systems Analysis of the Unintended Consequences (https://arxiv.org/abs/2411.13883)
Comments:
          13 pages, 4 figures

- **What's New**: 이 논문은 온라인 알고리즘이 학습하는 환경에 미치는 영향을 분석합니다. 또한 추천 시스템에서 사용자와 상품 속성을 바탕으로 최적의 제품 추천을 학습하는 과정에서 발생하는 문제를 다룹니다. 기존의 알고리즘은 사용자 속성을 정적(static)으로 처리하고 추천이 사용자 선호에 미치는 영향을 무시하는 경향이 있습니다. 이 연구는 이러한 불일치의 결과를 분석하고자 합니다.

- **Technical Details**: 온라인 학습 알고리즘은 일련의 행동과 그에 대한 환경의 피드백을 통해 환경을 학습합니다. 많은 알고리즘들은 환경이 정적(static)이라는 가정을 바탕으로 작동하며, 학습자(learner)의 행동에 환경이 반응하지 않는다고 전제합니다. 그러나 이 연구에서는 추천 시스템이 사용자 선호에 미치는 영향을 고려하여 통합된 모델을 제시하며, 선형 밴디트(linear bandit) 구조 속에서 환경의 진화를 분석합니다.

- **Performance Highlights**: 모델 시뮬레이션을 통해 추천 알고리즘이 사용자 군의 선호를 학습하는 과정에서 발생하는 유사성을 보여줍니다. 다양한 추천 알고리즘의 속성이 사용자 선호에 미치는 영향을 규명하며, 탐색(exploration)과 활용(exploitation) 사이의 균형이 사용자 선호의 학습에 어떻게 작용하는지를 분석합니다. 이러한 결과들은 추천 시스템의 디자인 및 개선을 위한 유용한 통찰을 제공합니다.



### Next-Generation Phishing: How LLM Agents Empower Cyber Attackers (https://arxiv.org/abs/2411.13874)
- **What's New**: 본 연구는 전통적인 피싱 탐지 도구들과 최신 대형 언어 모델 (LLM)의 피싱 이메일 탐지 성능을 비교 및 분석합니다. 특히, Gmail Spam Filter, Apache SpamAssassin, Proofpoint 및 여러 머신러닝 모델들이 LLM으로 재구성된 피싱 이메일을 식별하는 데 있어서 정확도 감소를 보임을 강조합니다. 연구 결과는 LLM이 피싱 공격을 더욱 정교하게 만드는 데 기여할 수 있음을 menunjukkan하며, 보안 강화 및 규제 조치의 필요성을 시사합니다.

- **Technical Details**: 본 연구에서 사용된 데이터 세트는 Nazario와 Nigerian Fraud datasets로, 각각 1200개의 이메일 (600개 합법적 이메일 및 600개 피싱 이메일)과 800개의 이메일이 포함되었습니다. 세 개의 전통적인 피싱 탐지 시스템 (Gmail Spam Filter, SpamAssassin, Proofpoint)과 세 개의 머신러닝 모델 (SVM, Logistic Regression, Naive Bayes) 그리고 다섯 개의 LLM (Llama 3, Gemini 1.5, Claude 3 Sonnet, GPT-3.5, GPT-4o)을 활용하여 실험을 시행하였습니다. 이메일의 특성으로는 발신자, 수신자, 제목 및 본문이 포함되었으며, 이는 탐지 시스템의 성능 평가에 중요한 요소입니다.

- **Performance Highlights**: 실험 결과, LLM으로 재구성된 피싱 이메일에 대한 탐지 정확도가 모든 탐지 시스템에서 눈에 띄게 감소하였으며, 이는 기존의 피싱 방어 시스템의 취약성을 강조합니다. 현재의 탐지 도구들은 LLM이 생성하는 다양한 피싱 변형에 효과적으로 대응하지 못하고 있으며, 이러한 문제를 해결하기 위해서는 강화된 보안 조치와 교육이 필요함을 보여줍니다. LLM을 활용한 데이터 증강(Dataset Augmentation)의 새로운 접근 방식이 피싱 탐지 능력을 향상시킬 수 있는 잠재력을 지니고 있음을 확인하였습니다.



### HARec: Hyperbolic Graph-LLM Alignment for Exploration and Exploitation in Recommender Systems (https://arxiv.org/abs/2411.13865)
- **What's New**: 이번 연구에서 제안하는 HARec은 사용자-아이템 협업 정보와 텍스트 묘사를 하이퍼볼릭 공간에서 공동 정렬하여 사용자 경험을 개선하는 혁신적인 프레임워크입니다. 이는(1) 의미 이해를 강화하는 계층 인지 그래프-LLM 정렬 메커니즘과(2) 사용자 조정 가능한 탐색-착취 무역의 하이퍼볼릭 계층 구조를 포함합니다.

- **Technical Details**: HARec은 탐색(exploration)과 착취(exploitation) 간의 균형을 유지하며 사용자 맞춤형 추천을 가능하게 합니다. 이 프레임워크는 하이퍼볼릭 공간 내에서 텍스트 묘사와 사용자-아이템 협업 정보를 통합하여 노이즈에 대한 민감도를 줄이고, 기존 추천 시스템의 계층 구조를 보다 정밀하게 모델링합니다. 이러한 접근 방식은 추천 정확도와 다양성 모두에서 성능을 크게 향상시킵니다.

- **Performance Highlights**: HARec은 다양한 실험을 통해 유틸리티 지표에서 최대 5.49% 향상, 다양성 지표에서 11.39% 증가를 기록하며 기존의 유클리드 및 하이퍼볼릭 기준 모델들을 일관되게 초월하는 성능을 보여줍니다. 이 모델은 정확도와 다양성을 동시에 우수한 성과로 달성한 첫 번째 예시로, 맥락에 맞는 추천의 질을 높였습니다.



### Exploratory Study Of Human-AI Interaction For Hindustani Music (https://arxiv.org/abs/2411.13846)
Comments:
          Accepted at NeurIPS Creative AI Track 2024

- **What's New**: 이 연구는 Hindustani 음악의 음성 컨투어(generative model for Hindustani vocal contours)를 위한 새로운 계층적 생성 모델 GaMaDHaNi와의 상호작용에 대한 사용자 연구를 다룹니다. 연구 참가자들은 세 가지 미리 정의된 상호작용 모드를 통해 모델을 사용할 때의 기대, 반응 및 선호도를 조사하였습니다. 이 연구는 실제 데이터로의 전이를 위한 모델 조정 없이 진행되었지만, Hindustani 음악의 맥락에서 이러한 모델을 다루는 데 필요한 향후 방향성을 제시합니다.

- **Technical Details**: GaMaDHaNi는 두 단계의 계층적 구조를 가진 모델로, 기본 주파수 컨투어(pitch contours) 및 스펙트로그램(spectrograms)의 연속적 데이터를 포함합니다. 이 모델은 Hindustani 음악 특유의 멜로딕 요소를 포착할 수 있도록 훈련되어, 생성된 음악 시퀀스의 품질과 조정 가능성을 극대화합니다. 연구에서는 두 가지 주요 아이디어인 프라임 생성(primed generation)과 멜로딕 재해석(melodic reinterpretation)을 이용하여 특정 상호작용 과제를 수행하였습니다.

- **Performance Highlights**: 연구 결과 참가자들은 GaMaDHaNi 모델과의 상호작용에서 두 가지 주요 도전 과제를 경험했습니다: 첫째, 모델 출력의 제한 부족, 둘째, 모델 출력의 비일관성입니다. 이러한 도전 과제는 Hindustani 음악의 특성과 관련되어 있으며, 이들의 피드백을 통해 향후 모델 디자인 개선 방향을 제시합니다. 참가자들은 이러한 상호작용을 통해 보다 나은 인간-AI 음악적 상호작용을 탐색할 수 있는 기반이 마련되었습니다.



### Heterophilic Graph Neural Networks Optimization with Causal Message-passing (https://arxiv.org/abs/2411.13821)
- **What's New**: 이번 연구에서는 인과 추론(causal inference)이 그래프 신경망(Graph Neural Network, GNN)에서 이질적 메시지 전이(heterophilic message-passing)를 포착하는 유망한 접근법이 될 수 있음을 발견하였습니다. 이는 비대칭 노드 의존성을 기반으로 이질적 엣지를 구별할 수 있게 해 주며, 학습된 인과 구조는 노드 간 관계를 보다 정확하게 제공합니다. 이를 통해 GNN 학습의 계산 복잡성을 줄이고 효율성을 높이기 위한 개입 기반 인과 추론을 도입했습니다.

- **Technical Details**: 그래프에서 인과 분석(causal analysis)을 구조적 학습 모델(structural learning model)로 수식화하고 베이 Bayesian 체계 내에서 최적화 문제를 정의합니다. 우리는 히토로 인과 관계에 기반하여 최적화 목표를 일관성 패널티와 구조 수정으로 분해하는 방법을 제시합니다. 조건부 엔트로피(conditional entropy)를 통해 이 목표를 추정하며, 이를 통해 이질성을 정량화하는 방법에 대한 통찰을 제공합니다. 이어서, 입력 그래프의 명시적인 인과 구조를 학습하기 위한 구조를 가진 CausalMP를 제안합니다.

- **Performance Highlights**: CausalMP는 이질적 및 동질적 그래프 설정에서 광범위한 실험을 수행한 결과, 링크 예측(link prediction) 성능에서 우수한 결과를 달성했습니다. 인과 구조에서 학습하는 과정은 기본 모델(classification task)에서 노드 표현(node representation)을 향상시킬 수 있습니다. 특히, 이 모델은 제한된 레이블을 가진 샘플에서도 다양한 기본 방법의 노드 분류 성능을 향상시킵니다.



### AutoMixQ: Self-Adjusting Quantization for High Performance Memory-Efficient Fine-Tuning (https://arxiv.org/abs/2411.13814)
- **What's New**: 이번 논문에서는 리소스 제약 아래에서 대형 언어 모델(LLM)을 미세 조정하는 데에 대한 도전 과제를 다루고 있습니다. 저자들은 Low-Rank Adaptation (LoRA), pruning, quantization을 통합한 새로운 방법인 AutoMixQ를 제안하였습니다. 이 최적화 프레임워크는 각 레이어에 최적의 quantization 구성을 선택하여 자원 효율성을 높이는 데 기여합니다.

- **Technical Details**: AutoMixQ는 가벼운 성능 모델을 활용하여 quantization 최적화 과정을 자동으로 수행합니다. 이는 기존의 고정된 구성방식에서 발생하는 비효율성을 해결하고, 복잡한 레이어 간 관계에 따라 다양한 quantization 정밀도를 쏠 수 있도록 합니다. 또한, Pareto optimality를 적용하여 메모리 사용량과 성능을 균형 있게 조절하며, 자원 제약 하에서 모델 성능의 상한을 추구합니다.

- **Performance Highlights**: 실험 결과에 따르면, AutoMixQ는 메모리 소모를 줄이면서도 우수한 성능을 달성했습니다. 예를 들어, LLaMA-7B 모델에서 30% pruning 비율을 적용했을 때, AutoMixQ는 BoolQ에서 66.21%의 성능을 기록하며, LoRA와 LoftQ에 비해 각각 62.45% 및 58.96%의 성능을 초과하면서도 메모리 사용량은 LoRA에 비해 35.5%, LoftQ에 비해 27.5% 감소했습니다.



### NewsInterview: a Dataset and a Playground to Evaluate LLMs' Ground Gap via Informational Interviews (https://arxiv.org/abs/2411.13779)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)의 언어 및 전략적 대화 기초기술 부족 문제를 해결하기 위해, NPR과 CNN의 40,000 건의 인터뷰 데이터를 수집하여 분석했습니다. 결과적으로, LLM은 인식과 질문 전환 능력에서 사람 인터뷰어보다 현저히 낮은 수준을 보였습니다. 이를 개선하기 위해, 다양한 개인을 설정하고 설득 요소를 통합한 실제 시뮬레이션 환경을 개발하였습니다.

- **Technical Details**: 연구에 사용된 데이터셋은 NPR과 CNN에서 수집된 45,848 개의 일대일 인터뷰 문서로, 각 인터뷰어와 인터뷰이의 역할을 엄밀히 분류하기 위해 질문 부호의 사용 빈도를 기반으로 역할을 판별했습니다. 또한, LLM이 사용자 답변 인식 및 설득 요소에서 어려움을 겪는지 분석하기 위해, 카운터팩추얼 시뮬레이션 기법을 사용했습니다. 다양한 질문 생성 접근 방법(기본, 사고 과정, 개요 모델)을 시험하여 LLM의 효과를 평가했습니다.

- **Performance Highlights**: 결과적으로, LLM은 정보 공유에서 인간의 행동을 모방하는 데에는 성공하였으나, 질문에 대한 응답 인식 및 설득적 대화의 면에서 지연을 보였습니다. 이는 LLM의 모델 크기와 능력에 관계없이 비효율적인 정보 추출로 이어졌습니다. 이러한 결과는 LLM이 전략적 대화 능력을 향상시킬 필요성을 강조하며, 향후 연구의 방향을 제시합니다.



### A Survey on Adversarial Robustness of LiDAR-based Machine Learning Perception in Autonomous Vehicles (https://arxiv.org/abs/2411.13778)
Comments:
          20 pages, 2 figures

- **What's New**: 이 논문은 자율주행 시스템과 적대적 기계 학습(Adversarial Machine Learning, AML)의 교차점에 대한 종합적인 조사를 제공합니다. 특히 LiDAR 기반 시스템에 초점을 맞추며, 센서에 대한 사이버 공격과 적대적 변동을 포함한 위협 환경을 탐구합니다. 자율주행 시스템의 안전과 보안을 보장하기 위한 강력한 방어 전략의 필요성을 강조합니다.

- **Technical Details**: 자율주행 차량 시스템은 여러 단계로 구성된 처리 파이프라인을 포함합니다. 여기에는 LiDAR, 카메라, GPS 해양 장치가 포함된 센서와 같은 원시 데이터 수집, 데이터 전처리, 그리고 인공지능(AI) 기반의 인식, 의사결정, 경로 계획 및 제어 시스템이 포함됩니다. 특히 LiDAR는 고해상도 데이터 수집과 정밀한 거리 측정의 장점을 제공하여 자율주행 시스템의 다양한 기능을 향상시킵니다.

- **Performance Highlights**: 이 논문은 LiDAR 기반 자율주행 시스템에서의 적대적 공격 및 방어 전략에 대한 포괄적인 분석을 제공합니다. 제안된 연구는 기존의 자율주행 시스템 연구에서 간과되었던 공격의 특성과 방어 전략의 한계를 다루며, 특히 사이버 공격과 전통적인 AML 공격의 위험을 강조합니다. 자율주행 시스템의 안전성을 증대시키기 위한 지식 기반을 확립하고, 연구의 격차를 해소하는 데 기여합니다.



### Benchmarking GPT-4 against Human Translators: A Comprehensive Evaluation Across Languages, Domains, and Expertise Levels (https://arxiv.org/abs/2411.13775)
Comments:
          Work in progress

- **What's New**: 본 연구는 GPT-4의 번역 능력을 다양한 수준의 인간 번역가와 비교하여 포괄적으로 평가한 것이다. MQM 스키마를 활용한 체계적인 인간 평가를 통해 세 가지 언어 쌍(중국어↔영어, 러시아어↔영어, 중국어↔힌디어) 및 세 가지 도메인(뉴스, 기술, 생의학)에서 번역 품질을 분석하였다. 연구 결과, GPT-4는 기초 수준의 번역가와 비슷한 번역 품질을 보였으나, 숙련된 번역가보다는 뒤처지는 것으로 나타났다.

- **Technical Details**: 기존의 Neural Machine Translation(NMT) 시스템과 달리, GPT-4는 자원 부족 언어 쌍에서도 안정적인 번역 품질을 유지하였다. 연구는 다양한 숙련도를 가진 인간 번역가들과 GPT-4의 번역을 비교하고, MQM 스키마를 통해 번역 결과의 오류를 레이블링하였다. 결과적으로 GPT-4는 총 오류 수관점에서 기초-중급 수준 번역가와 유사한 성능을 보여주었으며, 세부적인 오류 분석을 통해 번역 접근 방식의 차이를 규명하였다.

- **Performance Highlights**: GPT-4는 여러 도메인에서 일관된 성능을 보여주었으나, 문법과 Named Entity 부분에서 약점을 보였다. 연구는 GPT-4가 지나치게 문자 그대로의 번역을 경향하며, 어휘적 일관성이 부족하다는 두 가지 주요 한계를 드러냈다. 흥미롭게도, GPT-4는 인간 번역자들이 자주 겪는 환각이나 피로 문제에서 자유롭다는 장점을 지닌다.



### FastRAG: Retrieval Augmented Generation for Semi-structured Data (https://arxiv.org/abs/2411.13773)
- **What's New**: 본 논문은 FastRAG라는 새로운 RAG 접근 방식을 제안하여 반구조화 데이터의 효율적인 처리와 이해를 목표로 합니다. FastRAG는 전체 데이터 소스를 LLM에 제출하지 않고도 데이터 추출 및 구조화를 가능하게 하는 스키마 학습(schema learning)과 스크립트 학습(script learning)을 활용합니다. 또한, 텍스트 검색과 지식 그래프(KG) 쿼리를 통합하여 문맥이 풍부한 정보를 보다 정확하게 검색할 수 있도록 설계되었습니다.

- **Technical Details**: FastRAG는 반구조화 네트워크 데이터의 효율적인 처리를 위해 LLM과 GQL(Graph Query Language)을 결합합니다. 이 시스템은 데이터의 주요 개체와 속성을 추출하고, 각 개체를 원본 텍스트의 특정 라인에 매핑하여 KG를 통한 정확한 검색을 가능하게 합니다. 특히, FastRAG는 LLM이 코드 생성을 통해 데이터 스키마와 Python 코드를 구현하도록 하여 처리 비용을 절감합니다.

- **Performance Highlights**: FastRAG는 GraphRAG와 비교할 때 시간 측면에서 최대 90%, 비용 측면에서 85%의 개선을 보여줍니다. 평가 결과, FastRAG는 정확한 질문 응답을 제공하며, 반구조화 데이터 및 기술적인 네트워크 데이터의 복잡성 문제를 효과적으로 해결합니다. 이를 통해 네트워크 관리 도구의 향상을 기대할 수 있습니다.



### An Evaluation-Driven Approach to Designing LLM Agents: Process and Architectur (https://arxiv.org/abs/2411.13768)
- **What's New**: 이 논문은 시험 주도 개발(test-driven development)에 영감을 받아 LLM 에이전트의 고유한 평가 과제를 해결하는 것에 중점을 둡니다. 특정 목표와 위험 관리를 위한 적절한 평가는 이전의 전통적 접근 방식으로는 부족함을 강조하며, 지속적인 평가와 시스템적 재개발을 통합한 새로운 프로세스 모델과 참조 아키텍처를 제안합니다. 이를 통해 LLM 에이전트의 평가 방법을 통합적으로 개선하고자 합니다.

- **Technical Details**: LLM 에이전트는 조합 AI 시스템으로, 여러 LLM 외부 구성 요소를 포함하며 이에 대한 시스템 수준의 평가가 필요합니다. 기존 도구들은 일반적으로 모델 수준에 초점을 맞추고 있으며, LLM 에이전트의 동적이고 복잡한 운영 요구를 반영한 통합적인 평가 방식이 부족합니다. 이 논문은 계속되는 평가를 LLM 에이전트 라이프사이클 전반에 걸쳐 통합하는 평가 중심 설계 접근법을 제안합니다.

- **Performance Highlights**: LLM 에이전트의 성능과 안전성을 높이기 위한 평가 중심 설계 접근법은 LLM이  정교하고 변동하는 조건에서 신뢰성 있게 작동할 수 있도록 합니다. 지속적인 평가를 통해 LLM이 학습하고 새로운 입력에 적응할 수 있습니다. 이 연구는 LLM 에이전트의 평가를 시스템 수준에서 접근하여 지속 가능한 개선과 위험 관리를 촉진하는 데 초점을 맞추고 있습니다.



### Tiny-Align: Bridging Automatic Speech Recognition and Large Language Model on the Edg (https://arxiv.org/abs/2411.13766)
Comments:
          7 pages, 8 figures

- **What's New**: 이번 연구에서는 자원 제약이 있는 엣지 디바이스에서 ASR(Automatic Speech Recognition)과 LLM(Large Language Model) 간의 효율적인 정렬을 가능하게 하는 새로운 접근법인 Tiny-Align을 제안합니다. 기존 접근법들이 서버 규모의 컴퓨팅 자원을 요구하는 데 반해, Tiny-Align은 훈련 수렴 속도를 50배 향상시키며 결과의 질도 50% 이상 개선합니다. 이 연구는 개인화된 오디오 입력을 다루기 위한 효율적인 ASR-LLM 정렬 프레임워크 개발을 목표로 하고 있습니다.

- **Technical Details**: Tiny-Align 프레임워크는 자원을 최소화하면서도 안정적인 성능을 제공하는 새로운 projector를 디자인합니다. 이 Jack-특징의 디자이너는 기존의 MLP나 DNN 디자인보다 큰 임베딩 공간을 제공합니다. Tiny-Align을 사용할 때, 새로운 instruction injection 메커니즘이 추가되어 훈련된 모델의 추론 품질을 약 50% 향상시킵니다.

- **Performance Highlights**: 실험 결과, Tiny-Align은 다양한 엣지 LLM과 ASR 인코더 간의 정렬에서 유의미한 성능 향상을 달성하며, 최소한의 시간과 컴퓨팅 자원을 요구합니다. 이러한 성능 개선은 치매, 언어 장애 등을 가진 다양한 개인들의 데이터셋에 대한 테스트를 통해 확인되었습니다. Tiny-Align은 이러한 그룹의 ASR 인터페이스에서 고품질 상호작용을 지원할 수 있는 능력을 보여줍니다.



### AttentionBreaker: Adaptive Evolutionary Optimization for Unmasking Vulnerabilities in LLMs through Bit-Flip Attacks (https://arxiv.org/abs/2411.13757)
- **What's New**: 본 논문에서는 대규모 언어 모델(LLMs)의 비트 플립 공격(Bit-Flip Attacks, BFA)에 대한 취약성을 다루고 있습니다. LLMs는 텍스트 생성 및 요약과 같은 자연어 처리(NLP) 태스크에서 우수한 성능을 보이나, 하드웨어 기반 공격의 증가로 인한 우려가 커지고 있습니다. 연구 결과, 단 세 번의 비트 플립만으로도 LLM의 성능이 급격히 감소한다는 사실을 최초로 입증했습니다.

- **Technical Details**: 이 논문에서 제안된 AttentionBreaker 프레임워크는 LLM의 방대한 파라미터 공간을 효율적으로 탐색하여, BFAs를 통해 공격할 수 있는 중요한 파라미터를 식별합니다. 또한, GenBFA라는 진화 최적화 전략을 통해 가장 비판적인 비트를 효과적으로 고립시켜 공격 효율성을 높이고 있습니다. 본 연구에서는 기존의 BFA 기술들이 LLM의 취약점을 효과적으로 활용하지 못하는 이유를 분석하고, 이러한 도전 과제를 해결하기 위한 새로운 방법론을 제시합니다.

- **Performance Highlights**: 실험 결과에 의하면, LLaMA3-8B-Instruct 모델에서 단 세 번의 비트 플립(총 파라미터의 4.129 x 10^-9%)만으로도 MMLU 태스크에서의 정확도가 67.3%에서 0%로 떨어지고, Wikitext의 perplexity가 12.6에서 4.72 x 10^5로 급증하는 등 LLM이 심각한 취약성을 보인다는 사실이 확인되었습니다. 이를 통해 AttentionBreaker의 효과성과 LLM 아키텍처 내의 중요한 취약점을 발견하고 활용할 수 있는 가능성을 강조했습니다.



### Federated Continual Learning for Edge-AI: A Comprehensive Survey (https://arxiv.org/abs/2411.13740)
- **What's New**: 이 논문은 Edge-AI에서의 연합 지속적 학습(FCL)의 연구를 포괄적으로 조사하는 최초의 전문 서베이를 제공합니다. FCL은 다양한 고객의 지식을 융합하면서 데이터 개인정보를 유지하고 이전 작업의 지식을 잃지 않고 새로운 작업을 학습하는 데 중점을 둡니다. 이를 통해 동적이고 분산된 환경에서 학습 모델의 안정성과 신뢰성을 확보하는 것을 목표로 합니다.

- **Technical Details**: 논문에서는 FCL 방법을 연합 클래스 지속적 학습, 연합 도메인 지속적 학습, 연합 작업 지속적 학습의 세 가지 작업 특성에 따라 분류합니다. 각 범주에 대해 배경, 도전 과제, 문제 공식화, 해결책 및 한계를 논의하며, FCL로 강화된 실제 어플리케이션의 발전 상황과 잠재력을 검토합니다. FCL의 도전 과제로는 알고리즘-하드웨어 공동 설계 및 기초 모델과의 통합 등이 있습니다.

- **Performance Highlights**: FCL은 데이터 보안 규정을 준수하면서 교육 성능과 효과성을 높이는 것을 목표로 하며, 지속적 학습 및 연합 학습의 강점을 결합하여 Edge-AI에 강력한 기초를 제공합니다. 실제 사례로는 지능형 교통 시스템, 의료 시스템, IoT 및 디지털 트윈이 있으며, FCL의 다양한 응용 프로그램을 통해 현실 세계에 미치는 영향을 강조합니다. 이 조사 결과는 FCL의 빠른 발전과 Edge-AI 시대에서의 널리 퍼진 배치를 촉진할 수 있는 여러 연구 어려움을 조명합니다.



### Exploring Large Language Models for Climate Forecasting (https://arxiv.org/abs/2411.13724)
- **What's New**: 이 연구는 기후 변화의 영향이 증가함에 따라 신뢰할 수 있는 미래 기후 정보를 제공할 수 있는 도구에 대한 필요성이 커지고 있다는 점을 강조합니다. 특히, 대규모 언어 모델인 GPT-4가 복잡한 기후 데이터를 보다 쉽게 접근할 수 있도록 돕는 방법을 모색합니다. 연구는 GPT-4의 미래 강수 예측 능력을 평가하고, 전문가 입력이 없는 상황에서의 성능도 분석합니다.

- **Technical Details**: 단기(15일)와 장기(12개월) 강수 예측을 위해 다양한 실험을 설계하였습니다. GPT가 독립적으로 작동할 때, 명확한 추세 신호가 없을 경우 역사적 평균으로 돌아가는 보수적인 예측을 생성하는 경향을 보였습니다. 이러한 결과는 LLMs를 활용한 기후 예측의 가능성과 도전 과제를 동시에 보여줍니다.

- **Performance Highlights**: GPT-4는 전문가 데이터 입력 없이도 강수 예측을 수행했으나, 예측의 정확도는 한계가 있음을 증명했습니다. 이러한 연구 결과는 기후 관련 응용 프로그램과의 통합 가능성에 대한 통찰을 제공하고, 예측 능력을 향상시키기 위한 향후 방향성을 제시합니다.



### SimPhony: A Device-Circuit-Architecture Cross-Layer Modeling and Simulation Framework for Heterogeneous Electronic-Photonic AI System (https://arxiv.org/abs/2411.13715)
Comments:
          7-page

- **What's New**: 이 논문에서는 이종 전자-광자 집적 회로(EPICs)용 새로운 시뮬레이션 프레임워크인 SimPhony를 제안합니다. SimPhony는 하드웨어 혁신 탐색 및 시스템 평가를 지원하는 유연하고 정확한 플랫폼을 제공합니다. 다양한 광학 텐서 코어 디자인을 가진 다중 코어 아키텍처를 지원하는 범용 하드웨어 토폴로지 표현과 고유한 다차원 병렬성을 활용한 데이터 흐름 모델링을 포함합니다.

- **Technical Details**: SimPhony는 이종 EPIC AI 시스템 모델링을 위한 교차 계층 모델링 및 시뮬레이션 프레임워크로, 사용자 정의된 EPIC 디바이스 라이브러리를 기반으로 구축되어 있습니다. 각종 전자 및 광자 장치의 정확한 모델링을 위해 실제 실험 데이터를 반영하며, 메모리 구조와 데이터 인식 전력 추정을 처리할 수 있습니다. 또한, SimPhony는 layout-aware 면적 분석 및 bandwidth-adaptive 메모리를 지원하여 하드웨어/소프트웨어 공동 시뮬레이션을 용이하게 합니다.

- **Performance Highlights**: SimPhony는 데이터 인식 에너지를 정확하게 분석하고 시스템 지연 시간을 평가하며, 다양한 EPIC AI 하드웨어의 성능과 효율성을 검증할 수 있도록 설계되었습니다. 고성능의 디바이스 모델링과 연계하여 최적화된 시뮬레이션 결과를 제공하며, 전반적인 하드웨어 혁신을 가속화하는 데 기여할 것으로 예상됩니다. 이 시스템은 오픈소스로 제공되어 연구자들이 다양한 분야에서 EPIC AI 하드웨어를 혁신하고 평가할 수 있도록 지원합니다.



### Bimanual Dexterity for Complex Tasks (https://arxiv.org/abs/2411.13677)
Comments:
          In CoRL 2024. Website at this https URL

- **What's New**: Bidex라는 새로운 탈포현(teleoperation) 시스템이 소개되었습니다. 이 시스템은 매우 정밀하고 비용이 저렴하며, 낮은 지연(latency)을 갖춘 이중(두 개의) 손과 팔을 제어할 수 있는 기기를 제공합니다. 기존의 VR 기술과 비교하여 Bidex는 더 복잡한 작업에서도 더 빠르고 높은 품질의 데이터를 생성하는 것으로 나타났습니다.

- **Technical Details**: Bidex는 착용자가 두 개의 모션 캡쳐(glove) 장갑을 착용하고 자연스럽게 움직여 일상적인 작업을 수행할 수 있도록 설계되었습니다. 이 시스템은 Manus Meta glove를 사용하여 정확한 손가락 추적을 가능하게 하며, GELLO에서 영감을 받아 개발된 팔 추적 시스템은 로봇 팔의 관절 각도와 위치를 정확히 추적합니다. 이는 기존 기술이 매우 높은 정확도와 낮은 비용으로 이뤄지는 것을 가능하게 합니다.

- **Performance Highlights**: Bidex 시스템은 VR 헤드셋과 SteamVR 시스템보다 특히 데이터 수집의 속도와 정확도에서 뛰어난 성능을 보여주었습니다. 이를 통해 다양한 로봇 팔과 함께 사용될 수 있으며, 기존의 복잡한 로봇 작업을 더 쉽게 수행할 수 있도록 돕습니다. 실험 결과와 설정 재현을 위한 비디오는 Bidex 웹사이트에 공개되었습니다.



### Hymba: A Hybrid-head Architecture for Small Language Models (https://arxiv.org/abs/2411.13676)
Comments:
          20 pages, models are available on huggingface

- **What's New**: Hymba라는 새로운 언어 모델 패밀리가 소개됩니다. 이 모델은 하이브리드 헤드 병렬 아키텍처(hybrid-head parallel architecture)를 채택하여 Transformer의 어텐션 메커니즘과 상태 공간 모델(state space models, SSMs)을 통합합니다. 이로 인해 높은 효율성과 더 나은 문맥 요약 기능을 제공합니다. 또한, 입력에 추가되는 학습 가능한 메타 토큰(learnable meta tokens)이 도입되어 어텐션 메커니즘의 "강제 주의" 부담을 줄입니다.

- **Technical Details**: Hymba 모델은 어텐션 헤드와 SSM 헤드를 동일한 레이어 내에서 통합하여 각 입력에 대한 병렬 처리(parallel processing)를 가능하게 합니다. 이러한 하이브리드 헤드 접근 방식을 통해 높은 해상도의 기억력을 확보할 수 있으며, SSM의 효율적인 내용 요약 기능을 활용할 수 있습니다. 또한, KV 캐시(key-value cache) 공유 및 부분 슬라이딩 윈도우 어텐션(partial sliding window attention)을 도입하여 모델의 캐시 크기를 줄였습니다.

- **Performance Highlights**: Hymba-1.5B-Base 모델은 성능 면에서 2B 이하의 모든 공개 모델을 초월하며, Llama-3.2-3B 모델보다 평균 정확도가 1.32% 더 높습니다. 아울러 캐시 크기는 11.67배 줄어들고, 처리 속도는 3.49배 증가했습니다. Hymba는 또한, 다양한 벤치마크에서 새로운 최첨단 성능을 달성하며, 특히 일반 및 회상 집약적 작업에서 두드러진 성과를 보여줍니다.



### FabuLight-ASD: Unveiling Speech Activity via Body Languag (https://arxiv.org/abs/2411.13674)
Comments:
          23 pages, 8 figures, 3 tables, accepted for publication in Neural Computing and Applications

- **What's New**: 이 논문은 멀티모달 환경에서의 능동 화자 감지(ASD) 기술을 위한 FabuLight-ASD라는 새로운 모델을 제시합니다. 이 모델은 얼굴, 오디오 및 신체 자세 정보를 통합하여 감지 정확도와 강인성을 향상시킵니다. 기존의 Light-ASD 프레임워크를 기반으로 하여 신체 포즈 데이터를 추가하여 계산 오버헤드를 최소화하며, 현실 세계에서의 적용 가능성을 높입니다.

- **Technical Details**: FabuLight-ASD는 Face, audio와 body pose 정보를 활용하여 경량의 능동 화자 탐지를 위한 모델로, 사람이 말하는 활동의 원천을 탐지하는 데 신체 자세의 중요성을 강조합니다. 이 모델은 WASD(Wilder Active Speaker Detection) 데이터셋을 사용하여 평가되었으며, 사람의 자세 정보를 통합하여 얼굴 표현과 오디오 신호의 단서를 보완합니다. 결과적으로 94.3%의 평균 정확도(mAP)를 달성하여 Light-ASD의 93.7%와 비교해 성능이 향상되었습니다.

- **Performance Highlights**: FabuLight-ASD는 발화 장애, 얼굴 가림, 인간 음성 배경 소음 등의 어려운 시나리오에서도 특히 mAP의 개선을 보여주었습니다. 이 모델은 파라미터 수(27.3%)와 곱셈-누적 연산(multiply-accumulate operations)의 소폭 증가(최대 2.4%)에도 불구하고 효율성을 유지하여 실용성이 강조됩니다. 이러한 발견은 신체 자세 데이터 통합을 통해 ASD 성능 향상의 효과를 입증합니다.



### CryptoFormalEval: Integrating LLMs and Formal Verification for Automated Cryptographic Protocol Vulnerability Detection (https://arxiv.org/abs/2411.13627)
- **What's New**: 본 논문은 Large Language Models (LLMs)가 신규 암호화 프로토콜의 취약점을 자율적으로 식별할 수 있는 능력을 평가하기 위한 새로운 벤치마크를 소개합니다. 이를 위해 Tamarin(정리 증명기)와 상호작용하여 취약점을 탐지합니다. 우리는 수동으로 검증된 새로운 통신 프로토콜의 데이터 세트를 만들어 AI 에이전트가 발견한 취약성을 자동으로 검증하는 방법을 설계했습니다. 이 연구는 LLM과 상징적 추론 시스템의 통합 가능성을 보여줍니다.

- **Technical Details**: 연구는 Dolev-Yao 모델을 사용하여 보안 프로토콜을 분석하는 데 방점을 두고 있습니다. LLM 기반 에이전트가 Tamarin 증명기와 상호작용하여 암호화 프로토콜을 포멀라이즈하고 검증하는 과정을 체계적으로 테스트할 수 있도록 설계된 벤치마크를 제안합니다. 프로세스는 입력, 포멀라이제이션, 검증, 공격 검증의 네 단계로 구성되어 있습니다. 이 방법론은 실제 사이버 보안 감사를 반영하여 AI 모델의 적용 가능성을 평가합니다.

- **Performance Highlights**: 다양한 최신 LLM 모델이 이 벤치마크에서 평가되었습니다. 연구 결과는 LLM이 암호화 프로토콜 내에서의 취약성 탐지에 효과적일 수 있음을 보여주었으며, 이는 사이버 보안 시스템 및 자동 공격 도구 개발에 중요한 통찰을 제공합니다. LLMs와 상징적 추론 시스템의 융합이 PKI(공개 키 인프라)와 같은 복잡한 환경에서의 공격 탐지 능력을 향상시킬 가능성도 제시됩니다.



### Non-Linear Outlier Synthesis for Out-of-Distribution Detection (https://arxiv.org/abs/2411.13619)
- **What's New**: 이 논문에서는 NCIS라는 새로운 접근법을 제안하며, 이는 합성 아웃라이어를 생성하는 품질을 향상시키기 위해 확산 모델의 임베딩 공간을 직접 활용합니다. 또한, 조건부 볼륨 보존 네트워크를 통해 클래스 조건부 매니폴드(class-conditional manifold)를 모델링하여 훈련 분포의 표현력이 향상됩니다. 이로 인해 표준 데이터셋인 ImageNet-100과 CIFAR-100에서 새로운 최첨단 OOD 탐지 성능을 달성했습니다.

- **Technical Details**: NCIS에서 두 가지 새로운 기술을 개발하였습니다. 첫 번째는 확산 기반 임베딩으로, 이미지를 생성할 때 확산 모델을 직접 사용하여 이미지 임베딩을 만들어 ID 영역을 정밀하게 묘사합니다. 두 번째는 비선형 파라메트릭 분포로, 이는 클래스별 매니폴드를 적합하기 위해 조건부 볼륨 보존 네트워크(cVPN)를 도입하여 기존의 단순한 가우시안 분포보다 복잡한 클래스를 더 잘 모델링할 수 있게 합니다.

- **Performance Highlights**: NCIS는 ImageNet-100과 CIFAR-100의 두 주요 벤치마크에서 최첨단 성능을 달성하였으며, 각 구성 요소의 개별적인 영향을 확인하는 애블레이션 연구를 통해 그 효과성이 입증되었습니다. 또한, 데이터 사전 처리와 기타 주요 설계 선택들의 중요성에 대한 인사이트도 제공합니다. 이 연구에서 만든 코드도 공개되어 있어 연구자들이 쉽게 활용할 수 있도록 하였습니다.



### Verification and Validation of Autonomous Systems (https://arxiv.org/abs/2411.13614)
- **What's New**: 본 논문은 자율주행 차량의 소프트웨어 결함 예방 및 수정 방법에 대해 다루고 있으며, 소프트웨어 제품 개발 단계에서의 높은 신뢰성을 보장하는 방법에 대해서도 설명하고 있습니다. 자율주행 차량의 소프트웨어 구성 요소와 그 복잡성에 대해 논의하며, 다양한 테스트 방법론을 통해 소프트웨어 신뢰성을 높이고 있습니다.

- **Technical Details**: 자율주행 차량의 소프트웨어 스택은 Perception, Mapping, Planning 및 Control과 같은 주요 구성 요소로 이루어져 있습니다. V 모델(V-model)을 사용하여 소프트웨어 개발과 테스트를 진행하며, 기본적으로 Verification(검증)과 Validation(유효성 검사) 프로세스를 포함합니다. 또한, Motor in Loop (MIL), Software in Loop (SIL), Hardware in Loop (HIL)와 같은 여러 테스트 유형을 활용하여 다양한 수준에서 통합 테스트를 실시합니다.

- **Performance Highlights**: 시뮬레이션 및 차량 루프(Vehicle in Loop, VIL) 테스트는 자율주행 시스템의 개발과 검증 과정에서 중요한 역할을 합니다. 시뮬레이션은 안전한 환경에서 데이터를 증강하고, 모델 아키텍처를 탐색하며, 알고리즘을 평가할 수 있도록 도와줍니다. 카메라 교정 및 재교정 필요성 등 여러 요소를 고려하여 정확성과 신뢰성을 높일 수 있는 방법을 제안하고 있습니다.



### SuPLE: Robot Learning with Lyapunov Rewards (https://arxiv.org/abs/2411.13613)
Comments:
          7 pages, 4 figures

- **What's New**: 이 연구에서는 로봇 학습에서의 보상 함수 설계 문제를 다룹니다. 기존의 보상 구조는 많은 도메인 지식을 요구하지만, 이 연구는 시스템 동역학의 속성을 활용하여 외부 가정 없이도 적절한 보상을 생성하는 방법을 제안합니다. 특히, Lyapunov 지수를 기반으로 하는 '양의 Lyapunov 지수의 합(SuPLE)'을 통해 시스템 내재적 보상을 설계하는 접근 방식을 탐구합니다.

- **Technical Details**: 연구는 동적 시스템의 안정화 작업에 집중하며, 트렁케이티드 Lyapunov 지수를 보상 신호로 활용합니다. 이는 시스템이 최대 민감 상태(예: 직립 위치)로 향하도록 하는데 효과적입니다. 이 방법은 기존의 재설정된 상태에서 학습하는 방법과는 달리, 실제 로봇 시스템에서는 자연적인 초기 상태에서 시작할 수 있게 해주며, 보수적인 외부 지식 없이도 작동합니다.

- **Performance Highlights**: SuPLE을 통해 생성된 보상은 전통적인 보상 함수들과 비교할 때 훨씬 효율적임을 보여주었고, 이는 다양한 동적 시스템의 표준 벤치마크에서 안정화 작업을 성공적으로 수행하는 것을 가능하게 했습니다. 실험 결과는 제안된 방법이 수작업으로 설계된 보상 없이도 최적의 제어기를 효율적으로 구축할 수 있음을 입증했습니다. 연구 결과 및 코드는 공개되어 재현 가능성과 향후 연구의 토대를 제공합니다.



### DSTC: Direct Preference Learning with Only Self-Generated Tests and Code to Improve Code LMs (https://arxiv.org/abs/2411.13611)
- **What's New**: 이 논문에서는 Direct Preference Learning (DPL) 프레임워크인 DSTC(Direct Preference Learning with Only Self-Generated Tests and Code)를 제안하여, 외부 주석 없이도 신뢰할 수 있는 코드 생성 선호 쌍을 구성하는 방법을 다룹니다. DSTC는 자기 생성한 코드 스니펫과 테스트를 기반으로하여 코드 생성의 정확성을 높이는 데 중점을 둡니다. 이 접근법은 최소화-최대 선택(minimax selection) 과정과 테스트-코드(concatenation) 결합을 통해 선호 쌍의 품질을 개선하는 데 기여합니다.

- **Technical Details**: DSTC 프레임워크는 두 가지 주요 요소로 구성됩니다: 첫 번째는 선호 쌍의 품질을 향상시키기 위한 최소화-최대 선택 메커니즘이고, 두 번째는 코드 스니펫과 테스트의 결합을 통해 생성된 선호 쌍을 만드는 것입니다. 이를 통해 선택된 테스트와 거부된 테스트 간의 품질 격차를 넓히고, 실행 피드백이 포함된 신뢰성 있는 이진 테스트 학습을 실행할 수 있습니다. DSTC는 15억 파라미터를 가진 Starcoder2-15b-instruct 모델에서 DPO 및 KTO와 같은 직접 선호 최적화 방법과 결합하여 평가되었습니다.

- **Performance Highlights**: DSTC는 다양한 코드 벤치마크에서 코딩 정확도를 획기적으로 향상시키는 결과를 보여줍니다. HumanEval, MBPP 및 BigCodeBench와 같은 여러 벤치마크에서 pass@1 점수를 지속적으로 개선했습니다. 또한, ablation studies를 통해 DSTC의 각 구성 요소의 중요성을 입증하였으며, 33억 파라미터를 가진 모델 Deepseekcoder-33b에서도 성능 개선을 나타냈습니다.



### Integrating Dynamic Correlation Shifts and Weighted Benchmarking in Extreme Value Analysis (https://arxiv.org/abs/2411.13608)
Comments:
          33 pages, 8 figures

- **What's New**: 이번 논문에서는 극단값 분석(Extreme Value Analysis, EVA)을 위한 혁신적인 방법론인 극단값 동적 벤치마킹 방법(Extreme Value Dynamic Benchmarking Method, EVDBM)을 소개합니다. EVDBM은 극단값 이론을 통합하여 극단적 사건을 탐지하며, 새로운 동적 상관관계 식별(DISC)-thresholding 알고리즘과 결합되어 있습니다. 이를 통해 극단적인 조건 하에서도 주요 변수의 분석을 강화하고, 예측된 회귀값을 벤치마킹 점수에 통합하여 보다 정확한 상태 전망을 제공합니다.

- **Technical Details**: EVDBM 알고리즘은 과거 극단적 사건의 빈도를 고려하고, 미래 발생 가능성과 심각도를 반영하여 최종 벤치마킹 점수를 생성하는 정량적 메커니즘을 제공합니다. 이 접근법은 역사적 데이터와 확률적 예측을 결합하여, 극한 조건 하에서의 사례 비교를 보다 의미 있게 할 수 있게 해줍니다. 논문에서는 실제 PV 데이터에 이 방법론을 적용하여, 중요한 저생산 시나리오와 변수 간의 상관관계를 밝혀내고, 이는 위험 관리와 장기 계획에 기여합니다.

- **Performance Highlights**: EVDBM은 다양한 시나리오와 맥락에 적응할 수 있는 포괄적인 벤치마킹 프레임워크를 제공하여, 극단적 조건에서의 회복력을 이해하는 데 매우 가치 있습니다. 본 시스템은 금융 포트폴리오, 건강 위기 동안의 환자 반응, 불리한 기후 조건 하에서의 에너지 시스템을 포함한 여러 유사한 상황을 평가하고 비교하는 데 유용합니다. 이 방법론은 극단적인 조건을 초래하는 전반적인 변화에 대한 통찰을 제공하여, 각 분야에서 정보 기반의 의사결정을 가능하게 합니다.



### A Full-History Network Dataset for BTC Asset Decentralization Profiling (https://arxiv.org/abs/2411.13603)
Comments:
          IEEE BigData 2024

- **What's New**: 이 논문에서는 비트코인(BTC)의 거래 네트워크에 대한 체계적인 분석을 통해 자산의 분산화(decentralization) 정도를 정량화하는 새로운 접근 방식을 제안합니다. 2009년 제네시스 블록부터 2024년까지의 15년간의 전체 역사 데이터셋을 구축하였고, 이를 통해 BTC의 네트워크 특징을 평가하였습니다. 특히, 네트워크 기반의 분산화 정도를 강조하며 비트코인 거래의 동적 특성에 대한 통찰을 제공합니다.

- **Technical Details**: 비트코인은 블록체인(blockchain) 기술에 기반한 최초의 암호화폐로, 각 거래는 입력 및 출력으로 구성된 UTXO(Unspent Transaction Output) 모델을 통해 처리됩니다. 본 연구에서는 2,593,912,022개의 입력과 2,846,704,728개의 출력을 포함하는 방대한 거래 데이터를 수집하였으며, 이를 통해 네트워크 중심성(graph centrality) 지표도 분석하여 비트코인 네트워크의 동적 세부사항을 파악했습니다. 추가적으로, 중심성 지표는 Betweenness, Closeness, In-Degree, PageRank 등의 네 가지 주요 메트릭을 통해 측정되었습니다.

- **Performance Highlights**: 실험을 통해 네트워크 특성과 분산화 메트릭을 결합하면 비트코인 분석 작업, 특히 거래 수수료 및 MVRV-Z 점수 예측에 있어 성능이 향상됨을 보여주었습니다. 이 데이터셋과 분석 방법론은 다른 탈중앙화 자산에 대해서도 확장 가능성이 있으며, 블록체인 네트워크 내의 다양한 연구를 촉진할 수 있는 잠재력을 지니고 있습니다. 결론적으로, 본 연구는 비트코인의 거래 동적 및 분산화에 대한 중요한 통찰을 제공하며, 향후 연구를 위한 귀중한 자원을 제시합니다.



### Large-scale cross-modality pretrained model enhances cardiovascular state estimation and cardiomyopathy detection from electrocardiograms: An AI system development and multi-center validation study (https://arxiv.org/abs/2411.13602)
Comments:
          23 pages, 8 figures

- **What's New**: 이 연구에서는 CardiacNets라는 혁신적인 모델을 제안하여 심전도(ECG) 분석을 개선하고 심장 자기 공명 영상(CMR)의 진단 강점을 활용합니다. CardiacNets는 크로스 모달 대조 학습(cross-modal contrastive learning)과 생성적 사전 훈련(generative pretraining)을 통해 ECG 입력을 사용하여 심장 기능 지표를 평가하고 잠재적인 심혈관 질환(CVD)을 탐색합니다. 이 모델은 또한 ECG 데이터에서 고품질 CMR 이미지를 생성하여 해석 가능성을 높입니다.

- **Technical Details**: CardiacNets는 두 가지 주요 기능을 수행하며, 첫 번째로는 심장 기능 지표를 평가하고 관상동맥 질환(coronary artery disease), 심근병증(cardiomyopathy), 심막염(pericarditis), 심부전(heart failure) 및 폐고혈압(pulmonary hypertension)과 같은 CVD를 탐색하는 것입니다. 두 번째로, ECG 데이터를 사용하여 높은 품질의 CMR 이미지를 생성함으로써 의사들이 보다 쉽게 진단할 수 있도록 지원합니다. 이 연구는 두 개의 대규모 공개 데이터 세트(UK Biobank와 MIMIC-IV-ECG)와 세 개의 개인 데이터 세트에서 CardiacNets를 훈련하고 검증하였습니다.

- **Performance Highlights**: CardiacNets는 기존의 ECG 전용 모델에 비해 일관되게 우수한 성능을 보여주었다고 보고되었습니다. 이는 선별 정확도가 유의미하게 향상되었으며, 생성된 CMR 이미지는 모든 경험 수준의 의사들에게 진단적 지원을 제공합니다. 이 연구는 ECG가 심장 기능 평가에 대한 크로스 모달 통찰을 촉진할 수 있는 방법을 보여주어 인구 수준에서 향상된 CVD 선별 및 진단에 기여할 수 있는 가능성을 제시합니다.



### Can ChatGPT Overcome Behavioral Biases in the Financial Sector? Classify-and-Rethink: Multi-Step Zero-Shot Reasoning in the Gold Investmen (https://arxiv.org/abs/2411.13599)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)인 ChatGPT를 활용하여 재무 추론(financial reasoning) 능력을 조사하고, 이를 통해 이성적이고 설득력 있는 투자 의견을 생성하려고 합니다. 최근 제안된 Chain-of-Thought Prompting(CoT)을 사용해 LLM의 재무적 사고를 증진시키는 새로운 방법인 Classify-And-Rethink (CAR)를 소개합니다. 특히 금 투자에 대한 실험을 통해, 이 방법이 기존의 기준보다 높은 투자 수익률과 샤프 비율을 달성할 수 있음을 보여줍니다.

- **Technical Details**: 이 연구는 LLM이 감정적 반응에서 일반적으로 중립성을 유지하지만, 질문의 프레이밍 방식에 의해 잘못된 판단을 할 수 있음을 지적합니다. 투자자 행동 편향(behavioral biases)은 두 가지 범주로 나누어지며, 이 논문은 특히 프레이밍 효과(framing effect)가 LLM에도 적용될 수 있다는 점을 강조합니다. CAR 방식은 뉴스 기사를 분류하고 평가함으로써, LLM이 장기 트렌드 관점에서의 재무적 사고를 재고하도록 돕습니다.

- **Performance Highlights**: 실험 결과, CAR 방식은 ChatGPT를 사용하여 더 높은 설명 가능성을 가진 예측을 제공하고, 금융 관련 작업에서 중요한 행동 편향을 극복하는 데 기여합니다. 이를 통해 투자 결정을 할 때 보다 이성적인 접근을 가능하게 하여, 더 높은 투자 수익을 가져올 수 있는 잠재력을 보여줍니다. 또한, 수익률과 샤프 비율에서 CAR가 기존의 방법보다 뛰어난 성능을 발휘했음을 입증합니다.



### Enhancing Bidirectional Sign Language Communication: Integrating YOLOv8 and NLP for Real-Time Gesture Recognition & Translation (https://arxiv.org/abs/2411.13597)
- **What's New**: 이번 연구는 실시간 카메라 영상을 통해 미국 수화(ASL) 데이터를 텍스트로 변환하고, 텍스트를 수화로 변환하는 프레임워크를 개발하는 데 초점을 맞추고 있습니다. 이를 통해 언어 장벽을 허물고, 수화 사용자와의 실시간 소통을 가능하게 합니다. 본 연구는 You Only Look Once(YOLO) 및 Convolutional Neural Network(CNN) 모델을 활용하여 수화 인식 및 변환 작업을 수행합니다.

- **Technical Details**: 텍스트를 수화로 변환하기 위해 자연어 처리(NLP) 기법을 사용하여 입력된 문장에서 키워드를 추출하고 실시간으로 수화를 수행하는 비디오를 제공합니다. YOLO 모델은 실시간으로 공간-시간 특성을 자동으로 추출하며, CNN 모델은 수화 인식을 위한 실시간 처리를 제공합니다. 또한, Natural Language Toolkit(nltk)을 사용하여 품사 태깅을 수행하고, 전용 데이터셋에서 비디오를 검색하여 수화 표현을 생성합니다.

- **Performance Highlights**: 제안된 모델들은 실제 수화 데이터를 사용하여 인식 정확도, 정밀도, 번역 품질, 속도 및 견고성 등의 성능 평가를 수행하였습니다. 전체 150개의 비디오로 구성된 데이터셋을 만들어 수화 표현의 정확성을 높였습니다. 유저 인터페이스는 Django 웹 프레임워크를 기반으로 하며, 사용자가 문자를 입력하거나 오디오를 통해 수화를 요청할 수 있도록 설계되어 있습니다.



### A Novel Speech Analysis and Correction Tool for Arabic-Speaking Children (https://arxiv.org/abs/2411.13592)
- **What's New**: 이 논문은 아랍어 아이들의 발음을 돕기 위해 ArPA라는 새로운 애플리케이션을 소개합니다. 이 애플리케이션은 진단 모듈과 치료 모듈로 구성되어 있으며, 특히 아랍어 발음에 어려움을 겪는 어린이를 위해 설계되었습니다. 새로운 기술로 Melspectrogram 및 MFCC 이미지를 사용하여 발음 인식을 수행하고, ResNet18 분류기를 통해 발음 오류를 99.015%의 정확도로 식별합니다.

- **Technical Details**: ArPA 애플리케이션의 진단 모듈은 아동의 음성 신호를 캡처하고 다양한 기계 및 딥러닝 분류기를 사용하여 분석합니다. 이 과정에서는 K-최근접 이웃(KNN), 서포트 벡터 머신(SVM), 결정 트리, 그리고 딥 뉴럴 네트워크인 ResNet18가 활용됩니다. 치료 모듈은 아동이 올바르게 발음한 글자에 따라 아바타 레벨을 높여주는 게임화된 인터페이스를 제공하여 긍정적인 피드백을 줍니다.

- **Performance Highlights**: 연구에서는 아동 발음에 대한 두 개의 데이터셋이 사용되었으며, ResNet18 분류기가 특히 뛰어난 성능을 보였습니다. Melspectrogram 이미지를 사용하는 경우 99.015%의 정확도로 발음 오류를 식별하였고, MFCC 이미지를 사용하는 경우보다 더 뛰어난 결과를 나타냈습니다. 이는 아랍어 발음 분류에서 기존의 연구 결과를 뛰어넘는 성과로, ArPA 애플리케이션의 실용성을 입증합니다.



### Improved GUI Grounding via Iterative Narrowing (https://arxiv.org/abs/2411.13591)
- **What's New**: 이번 연구에서는 GUI Grounding 능력을 강화하기 위해 Iterative Narrowing (IN)이라는 시각적 프롬프트 프레임워크를 제안했습니다. 기존의 VLM 모델의 성능 개선을 목표로 하며, GUI 인터페이스에서의 정밀한 시각적 위치 식별이 가능해집니다. IN은 초기 예측을 점진적으로 개선하는 과정을 통해 GUI grounding의 정확성을 향상시킵니다.

- **Technical Details**: 이 방법은 입력 이미지와 해당 텍스트 쿼리를 받으면, 이미지를 999×999 픽셀로 표준화하여 처리합니다. 모델은 다음 반복을 위해 예측된 좌표를 기반으로 이미지 잘라내기를 생성하며, 매 반복마다 이미지 크기를 줄이는 방식을 사용합니다. 이 반복 과정은 n 회 반복되며, 마지막 반복에서는 최종 타겟 위치를 결정합니다.

- **Performance Highlights**: ScreenSpot 벤치마크를 통한 평가 결과, IN 프레임워크는 특히 일반 VLM 모델인 InternVL-2-4B 및 Qwen2-VL-7B에서 성능 개선을 이끌어냈습니다. 그러나 공간적으로 거리가 큰 컨텍스트 단서 처리에 한계가 있어, 특정 상황에서 성능이 저하되는 경향이 있습니다. 향후 연구에서는 이러한 컨텍스트 한계를 해결하기 위한 방법에 대한 탐색이 필요합니다.



### Deep learning waterways for rural infrastructure developmen (https://arxiv.org/abs/2411.13590)
Comments:
          18 pages, 6 figures

- **What's New**: 이 연구에서는 미국의 고해상도 위성 이미지와 디지털 고도 모델을 기반으로 한 컴퓨터 비전 모델인 WaterNet을 개발했습니다. 이 모델은 기존에 매핑되지 않은 아프리카 대륙의 수로를 식별하는 데 활용됩니다. 이러한 시스템은 공공 데이터에 기반하여 인도적 필요를 포착하고 사회 개발을 위한 계획에 기여할 수 있는 잠재력을 보여줍니다.

- **Technical Details**: WaterNet은 U-Net 스타일의 합성곱 신경망(convolutional neural network)으로 개발되었습니다. 이 모델은 미국의 수로 데이터에서 레이블을 훈련하여 대규모 지리적 영역에 확장 가능하게 설계되었습니다. WaterNet은 추가적인 후처리 단계를 통해 수로 데이터를 벡터화하여 다른 데이터셋과 비교할 수 있습니다.

- **Performance Highlights**: WaterNet은 아프리카의 다양한 국가에서 지역 사회의 다리 건설 요청과 같은 인프라 필요를 충족하는 데 있어 기존 데이터셋보다 평균 93%의 정확도로 수로를 포착했습니다. 반면, OpenStreetMap은 36%와 TDX-Hydro는 62%의 낮은 성능을 보였습니다. WaterNet은 인프라 개발의 중요한 기초 자료로서 지역 사회의 필요를 정확히 반영하는 데 기여할 것으로 기대됩니다.



### Unveiling Redundancy in Diffusion Transformers (DiTs): A Systematic Study (https://arxiv.org/abs/2411.13588)
Comments:
          9 pages including reference

- **What's New**: 이번 연구는 Diffusion Transformers (DiTs) 모델의 비효율적인 추론 속도를 개선하기 위한 방법을 제시합니다. 특히, 여러 DiT 모델 간의 redundancy(중복성) 분석을 통해 효과적인 캐싱(caching) 전략을 개발할 수 있는 가능성을 모색합니다. 또한, 유연한 분석 도구인 DiTCacheAnalysis를 제공하며, 이를 통해 각 개별 모델에 맞춘 최적화된 캐싱 전략을 개발 가능하게 합니다.

- **Technical Details**: Diffusion 모델은 Gaussian noise에서 시작하여 x0 (최종 결과 이미지)로 가는 비선형 과정을 포함하며, 이 과정에서 여러 iterative(반복적인) denoising 단계를 통해 고품질 이미지를 생성합니다. DiT 아키텍처는 입력 데이터를 latent patches(잠재 패치)로 나누고, transformer의 self-attention 메커니즘을 통해 각 패치 간의 관계를 모델링합니다. 이 과정은 다수의 계산 단계로 인해 중대한 latency(지연 시간)를 야기하게 됩니다.

- **Performance Highlights**: 연구 결과, 다양한 DiT 모델 간의 중복성 분포가 크게 다르다는 것을 발견했습니다. 반면 하나의 모델 내에서 중복성 분포는 안정적이며 사용자의 프롬프트, 추론 단계, 스케줄러의 변동에도 영향을 받지 않아 일관성을 보입니다. 이를 통해 캐싱 전략을 모델별로 제안하고, 공통의 캐싱 방식은 효과적이지 않다는 결론을 내렸습니다.



### Exploring the Adversarial Vulnerabilities of Vision-Language-Action Models in Robotics (https://arxiv.org/abs/2411.13587)
- **What's New**: 이 논문은 Vision-Language-Action (VLA) 모델 기반 로봇 시스템의 취약점을 체계적으로 측정합니다. VLA 모델은 시각적 입력과 언어적 입력을 통합하여 복잡한 작업을 수행하는 로봇을 가능하게 하며, 이 과정에서 새로운 공격 표면이 발생하여 적대적 공격에 취약해집니다. 본 연구는 VLA 기반 시스템의 독특한 요구 사항을 인식하고 이러한 시스템의 안보성을 높이기 위한 공격 목표 및 평가 방법을 제안합니다.

- **Technical Details**: 본 연구에서는 로봇 시스템의 물리적 역학과 동작 제약을 고려하여 공격 목표를 설계합니다. 또한, VLA 모델에서 시간적 종속성을 활용하여 로봇의 행동을 크게 방해할 수 있는 공격을 설계합니다. 특정한 목표를 가진 공격을 생성하기 위해 Action Discrepancy Objective 및 Geometry-Aware Objective를 설정하고, Patch-Based Attack 방법을 개발하여 실제 및 디지털 환경에서 효과적인 공격을 수행합니다.

- **Performance Highlights**: 실험 결과, 제안된 공격 방법이 로봇 작업의 성공률을 100%까지 감소시키는 것으로 나타났습니다. 이러한 연구 결과는 VLA 아키텍처가 현재 보안의 중요한 허점을 가지고 있음을 시사합니다. 연구는 VLA 기반 로봇 시스템의 안전성을 향상시키기 위한 방안으로 공격 지표를 제시하고 있으며, 향후 실제 세계의 배포 전에 강력한 방어 전략 개발의 필요성을 강조합니다.



### Advance Detection Of Bull And Bear Phases In Cryptocurrency Markets (https://arxiv.org/abs/2411.13586)
- **What's New**: 이번 논문에서는 비트코인(Bitcoin)의 성과를 예측하기 위해 다양한 예측 알고리즘을 사용하고 있습니다. 이는 비트코인의 50일 및 200일 이동 평균(Moving Averages)을 계산하여 향후의 가능성 있는 불장(Bull Phase)과 약세장(Bear Phase)을 시각화할 목적으로 수행됩니다. 특히, 암호화폐 시장에서 비트코인이 차지하는 시장 지배력이 50%에 가깝다는 점에서 그 중요성이 강조됩니다.

- **Technical Details**: 데이터 수집 과정에서 비트코인에 관한 Open, High, Low, Close, Volume 데이터를 2012년 1월 1일부터 수집하였습니다. 이후 이동 평균(Moving Average), RSI, MACD, 모멘텀(Momentum), 볼린저 밴드(Bollinger Bands) 및 ROC와 같은 기술적 지표(Technical Indicators)를 생성하였습니다. 연구진은 예측 모델을 구성하기 위해 다중 선형 회귀(Multiple Linear Regression)와 LSTM 모델을 비교하여 분석하였습니다.

- **Performance Highlights**: 모델의 성과를 평가하기 위해 예측된 이동 평균 그래프와 실제 이동 평균 그래프를 비교하였습니다. 비록 독립 변수 간의 높은 상관관계가 문제로 지적되었지만, 예측의 정확도를 높이기 위해 출력 변수에 21일 시점의 종가를 포함하는 방법을 채택했습니다. 이러한 접근 방식은 비트코인 가격을 최대 21일 앞으로 예측하는 데 효과적이면, 시장의 미래 방향성을 이해하는 데 기여할 것입니다.



### Artificial Intelligence in Cybersecurity: Building Resilient Cyber Diplomacy Frameworks (https://arxiv.org/abs/2411.13585)
- **What's New**: 이 논문은 자동화(automation)와 인공지능(AI)이 어떻게 미국의 사이버 외교(cyber diplomacy)를 변화시키고 있는지를 탐구합니다. 이러한 기술을 활용함으로써 미국은 사이버 외교의 복잡성과 긴급성을 관리하며 의사 결정, 효율성 및 안전성을 향상할 수 있습니다. 글로벌 상호 연결성이 증가함에 따라 디지털 공간에서의 국가 이익 관리가 점점 더 중요해지고 있습니다.

- **Technical Details**: AI와 자동화는 방대한 데이터 양을 신속하게 처리할 수 있는 능력을 갖추고 있어 사이버 위협(threat)과 기회에 대한 시기적절한 대응을 가능하게 합니다. 이 논문은 미국의 경쟁력을 유지하고 국가 이익을 보호하기 위해 이러한 도구들의 전략적 통합(integration)의 중요성을 강조합니다. 예를 들어, 자동화는 외교적 커뮤니케이션을 향상시키고, AI는 예측 분석(predictive analytics)과 실시간 의사 결정(real-time decision making)을 지원합니다.

- **Performance Highlights**: 여러 사례 연구(case studies)를 통해 AI가 사이버 활동을 모니터링하고 국제 사이버 정책을 관리하는 데 효과적임을 보여줍니다. 논문은 또한 윤리적 우려(ethical concerns)와 보안 취약성(security vulnerabilities), 기술에 대한 의존성(reliance on technology) 같은 도전 과제를 다루며, 인간의 감독(human oversight)과 강력한 거버넌스 프레임워크(governance frameworks)의 필요성을 강조합니다. 이러한 기술을 채택함으로써 미국의 사이버 외교는 더욱 주도적이고 효과적으로 진화하는 디지털 환경을 탐색할 수 있게 됩니다.



### AddrLLM: Address Rewriting via Large Language Model on Nationwide Logistics Data (https://arxiv.org/abs/2411.13584)
Comments:
          Accepted by KDD'25 ADS Track

- **What's New**: AddrLLM은 주소 재작성(address rewriting)을 위해 설계된 혁신적인 프레임워크입니다. 기존 주소 재작성 방법의 한계를 극복하기 위해, AddrLLM은 Supervised Fine-Tuning(SFT), Address-centric Retrieval Augmented Generation(RAG), Bias-free Objective Alignment 모듈로 구성되어 있습니다. 이 연구는 LLM 기반 주소 재작성 접근 방식을 이용하여 비정상 주소 문제를 해결하는 효과적인 방법을 제시하며, 실제 데이터에 대한 철저한 테스트를 통해 성능을 입증하였습니다.

- **Technical Details**: AddrLLM 프레임워크는 세 가지 주요 구성 요소로 구성됩니다. Supervised Fine-tuning 모듈은 데이터의 품질을 극대화하여 주소를 효율적으로 수정할 수 있도록 합니다. Address-centric Retrieval Augmented Generation 모듈은 관련 주소의 맥락 정보를 활용하여 모델의 성능을 강화하며, Bias-free Objective Alignment 모듈은 편향을 최소화하기 위해 JD의 LBS 시스템에서 제공되는 데이터를 통합하여 결과를 보정합니다.

- **Performance Highlights**: AddrLLM은 오프라인 실험을 통해 비정상 주소를 43.9% 수정하였으며, 최첨단 방법들에 비해 24.2%의 성능 향상을 보였습니다. JD 물류 시스템에 4개월 이상 배포된 결과, 약 200만 건의 일일 소포 중에서 비정상 주소로 인한 소포 재배치율을 40% 이상 감소시켰습니다. 이러한 결과는 AddrLLM이 실제 애플리케이션에서 높은 효율성을 가져다 줄 수 있음을 강조합니다.



### Enhanced FIWARE-Based Architecture for Cyberphysical Systems With Tiny Machine Learning and Machine Learning Operations: A Case Study on Urban Mobility Systems (https://arxiv.org/abs/2411.13583)
- **What's New**: AI(인공지능) 및 IoT(사물인터넷)의 발전으로 사회의 디지털 전환이 가속화되고 있습니다. 본 논문에서는 tinyML과 edge computing을 위한 새로운 아키텍처를 제안하여 전체 MLOps(머신러닝 운영 프로세스) 주기를 관리할 수 있는 프레임워크를 구현하는 방법을 제시합니다. 이 아키텍처는 FIWARE 소프트웨어 구성 요소를 확장하여 사이버 물리 시스템에서 tinyML의 운영을 가능하게 합니다.

- **Technical Details**: MLOps 라이프사이클을 다루기 위해 논문에서는 데이터를 수집, 준비, 훈련, 평가, 조정, 배포 및 모니터링하는 일련의 작업을 제안합니다. FIWARE IDAS로 알려진 컴포넌트는 IoT 기기를 FIWARE 생태계에 통합하는 데 필요한 미들웨어 역할을 합니다. IoT 에이전트는 NGSI-LD HTTP 요청을 변환하여 다양한 장치에 적합하게 데이터를 제공하며, 마이너 리소스 문제를 해결하는 데 기여합니다.

- **Performance Highlights**: FIWARE 아키텍처의 적용 사례로, 스마트 교통 시스템의 자동화 된 교통 장벽 시스템이 소개되었습니다. 이 시스템은 차량 밀도를 예측하고 그에 따라 교통 흐름을 조절하는 머신러닝 모델을 훈련시키고 배포하도록 설계되었습니다. 실제 교통 데이터를 기반으로 한 머신러닝 모델의 결과는 논문에서 공개된 레포지토리에서 확인할 수 있습니다.



### COOD: Concept-based Zero-shot OOD Detection (https://arxiv.org/abs/2411.13578)
- **What's New**: 이 논문에서는 COOD (concept-based OOD detection)라는 새로운 제로샷 다중 레이블 OOD 탐지 프레임워크를 소개합니다. 이 방법은 Vision-Language Model (VLM)인 CLIP을 활용하여 복잡한 레이블 의존성을 모델링하고, OOD 샘플을 효과적으로 구분할 수 있게 합니다. COOD는 추가적인 재교육 없이 정교한 레이블 확장 전략과 새로운 점수 함수를 사용하여 기술되어 있습니다.

- **Technical Details**: COOD는 긍정적인 레이블과 부정적인 레이블을 모두 포함한 개념 기반의 레이블 확장을 통해 OOD 탐지를 수행합니다. 이 프레임워크는 다중 레이블 설정에서 각 레이블에 대해 더 복잡한 의미적 공간을 모델링하여 ID와 OOD 샘플 간의 차이를 정밀하게 구분할 수 있게 합니다. 또한, 다중 레이블 레이블의 존재와 의존관계를 고려하여 유연하면서도 강력한 OOD 탐지 성능을 확보합니다.

- **Performance Highlights**: COOD는 VOC 및 COCO 데이터셋에서 약 95%의 평균 AUROC를 달성하며 기존 방법에 비해 우수한 성능을 보입니다. 특히, 다양한 레이블 수와 OOD 샘플 유형에 걸쳐 견고한 성능을 유지하며, CLIP-B/16에서 초당 약 800장의 이미지 처리 효율성을 기록합니다. 이러한 결과는 COOD가 실제 환경에서 신뢰할 수 있는 OOD 탐지 방법임을 보여줍니다.



### The Role of AI in Financial Forecasting: ChatGPT's Potential and Challenges (https://arxiv.org/abs/2411.13562)
Comments:
          7 pages, 4 figures, 3 tables

- **What's New**: 이 논문은 금융 분야에서 인공지능(AI)의 미래 전망과 특히 금융 예측에서의 도전과제를 다룹니다. 최근 딥 러닝(deep learning)과 강화 학습(reinforcement learning)의 발전이 AI 기술의 역학을 변화시키고 있으며, 블록체인(blockchain) 및 사물인터넷(Internet of Things)과의 통합이 중요해지고 있습니다. 이러한 기술들은 데이터 처리 능력의 향상을 가져오고, 투자자 맞춤형 서비스 제공에 기여하고 있습니다.

- **Technical Details**: AI 기술의 발전은 금융 서비스의 혁신을 이끌고 있으며, 이는 규제(regulatory) 및 윤리적 문제를 야기합니다. 데이터 프라이버시(data privacy) 보호와 관련된 이슈는 금융 sector에서 AI의 통합에 있어 중요한 과제가 되고 있습니다. 현재 AI 기술의 한계는 금융 예측에 영향을 주며, 이에 대한 분석이 필요합니다.

- **Performance Highlights**: AI의 발전은 직업 시장의 변화, 새로운 금융 기관의 등장, 사용자 경험의 혁신 등 금융 산업의 미래에 상당한 영향을 미칠 것으로 예상됩니다. 투자자들이 AI에 대한 이해를 높이고, AI 도구의 발전이 금융 의사결정에서 폭넓은 채택을 촉진할 것입니다. 이러한 변화는 금융 sector에서 AI 기술의 중요한 역할을 강조하고 있습니다.



### Composing Ensembles of Instrument-Model Pairs for Optimizing Profitability in Algorithmic Trading (https://arxiv.org/abs/2411.13559)
- **What's New**: 이 논문은 재무 시장에서의 비선형성과 복잡성을 다루면서, 데이터 기반의 가격 방향 예측 시스템을 제안합니다. 특히 단기 가격 변동 예측의 어려움을 해결하기 위해 새로운 두 층의 Composing Ensembles 아키텍처를 개발하였으며, 이 시스템은 그리드 서치를 통해 최적화되었습니다.

- **Technical Details**: 논문에서는 다양한 재무 상품을 예측하기 위해 Composing Ensembles 아키텍처를 사용합니다. 이 시스템은 주식 관련 뉴스, 기업 프로필, 공공 감정, 그리고 글로벌 경제 조건과 같은 다양한 요인의 영향을 반영하여 가격 상승 여부를 예측합니다. 또한, 그리드 서치( grid search ) 기법으로 모델의 성능을 최적화하였습니다.

- **Performance Highlights**: 제안된 트레이딩 시스템은 광범위한 재무 상품 및 시간 프레임에 대해 백테스트(back-test)가 수행되었으며, 표준 투자 전략 대비 20%의 성능 향상을 보여주었습니다. 이는 단순한 전략에 비해 상당한 개선을 나타내며, 실질적인 투자 수익률(Return on Investment, ROI) 증대를 가져올 수 있습니다.



### Generating bilingual example sentences with large language models as lexicography assistants (https://arxiv.org/abs/2410.03182)
- **What's New**: 이번 연구에서는 고자원(high-resource) 언어인 프랑스어, 중자원(mid-resource) 언어인 인도네시아어, 저자원(low-resource) 언어인 테툰(Tetun)을 포함한 다양한 언어에서 LLM(대형 언어 모델)의 성능을 평가했습니다. LLM이 생성한 예문이 GDEX(Good Dictionary EXample) 기준을 충족하는지를 분석하여, 자원 수준에 따른 성능 차이를 확인했습니다.

- **Technical Details**: 연구에서는 두 개의 LLM, GPT-4o와 Llama 3.1 405b를 사용하여 예문을 생성했습니다. GDEX 기준에 따른 예문 생성을 위해 상위 10,000개의 빈도수가 높은 단어 리스트를 사용하였으며, 각 언어에 대해 50개의 단어 쌍을 선택하였습니다. 또한, 예문에 대한 질적 평가는 원어민이 수행하였습니다.

- **Performance Highlights**: 저자원 언어의 경우 LLM이 생성한 예문 품질이 낮아지는 경향을 보였으며, 인간 주석자 간의 일치율이 낮아 пример의 질에 대한 선호에 큰 변동성이 있음을 발견했습니다. 그러나, in-context learning 기법을 통해 LLM을 개인 주석자의 선호에 맞출 수 있음을 시연하였습니다. 자동 예문 평가는 높은 자원 수준 언어에서 문장 혼란도가 GDEX 기준의 typicality 및 intelligibility에 적합한 대리변수가 된다는 것을 조사했습니다.



### Cerebrovascular Segmentation via Vessel Oriented Filtering Network (https://arxiv.org/abs/2210.08868)
- **What's New**: 이번 논문에서는 Magnetic Resonance Angiography (MRA)와 Computed Tomography Angiography (CTA)로부터의 cerebrovascular segmentation을 위한 새로운 접근법인 Vessel Oriented Filtering Network (VOF-Net)을 제안합니다. VOF-Net은 혈관의 복잡성과 형태 변동 문제를 해결하기 위해 도메인 지식을 주입한 convolutional neural network를 활용합니다. 특히 혈관의 방향성을 반영한 필터를 설계하여 혈관 segmentation의 품질을 향상시킵니다.

- **Technical Details**: VOF-Net은 혈관 방향 필드(vessel orientation field)를 기반으로 혈관에 적합한 필터를 설계합니다. 이 방향 필드는 orientation estimation network를 통해 얻어지며, 이를 통해 추출된 특징(features)을 segmentation network에 주입하여 혈관의 세밀하고 곡선인 구조를 반영합니다. 이러한 방식은 기존 방법들과 비교했을 때 혈관 세분화에 있어 더 정확한 결과를 도출할 수 있습니다.

- **Performance Highlights**: 실험 결과 CTA와 MRA 데이터셋에서 제안한 방법이 효과적인 vessel segmentation을 보여주었으며, 특히 특정 혈관 필터를 주입함으로써 segmentation 성능이 개선되었습니다. 이는 복잡한 혈관 네트워크의 정확한 세분화에 기여하여 cerebrovascular pathology의 진단 및 치료에 유의미한 영향을 미칠 수 있다는 점에서 큰 의미가 있습니다.



### Structure-Based Molecule Optimization via Gradient-Guided Bayesian Upda (https://arxiv.org/abs/2411.13280)
Comments:
          27 pages, 17 figures

- **What's New**: 이번 연구에서 제안된 Molecule Joint Optimization (MolJO)은 구조 기반 분자 최적화(SBMO)를 위한 최초의 그래디언트 기반 프레임워크로, 연속적이며 미분 가능한 공간을 활용하여 다양한 모달리티 간의 공동 유도 신호를 가능하게 합니다. MolJO는 슬라이딩 윈도우를 통해 과거의 히스토리를 최적화함으로써 탐색과 활용 간의 원활한 균형을 이룹니다.

- **Technical Details**: MolJO는 주어진 단백질 대상에 대해 분자 구조를 최적화하는 데 있어 전문가 지정 목표에 따라 분자 속성 향상을 우선시합니다. 이전의 방법들과 달리, MolJO는 3D 구조 인식을 통해 기존의 화합물 최적화에서 생성된 분자 데이터를 처리하여, 단순한 SMILES나 2D 그래프 표현으로부터 개선된 미세 조정 기능을 제공합니다.

- **Performance Highlights**: MolJO는 CrossDocked2020 벤치마크에서 성공률 51.3%, Vina Dock -9.05 및 SA 0.78의 성능을 달성하며, 기존의 그래디언트 기반 대비 4배 이상의 성과를 보였습니다. 또한, R-group 최적화 및 scaffold hopping과 같은 다양한 최적화 설정에 적용 가능하여 약물 디자인 분야에서의 활용 가능성을 더욱 부각시킵니다.



### SRA-MCTS: Self-driven Reasoning Augmentation with Monte Carlo Tree Search for Code Generation (https://arxiv.org/abs/2411.11053)
- **What's New**: 본 논문은 모델이 고품질의 중간 추론 경로를 자율적으로 생성할 수 있도록 안내하는 SRA-MCTS라는 데이터 생성 프로세스를 제안합니다. 이는 재귀적인 프로세스를 통해 모델에 새로운 추론 능력을 부여하고, 코드 생성의 복잡한 문제 해결에 대한 성공률을 향상시키는 데 도움을 줍니다. 이 접근 방식은 추가적인 감독 없이도 성능을 높이아는 잠재력을 보이며, 작은 모델에서도 효과적으로 작동합니다.

- **Technical Details**: SRA-MCTS는 여러 단계로 구성된 데이터 생성 방법론으로, 계획 생성부터 코드로의 변환 및 모델 훈련에 이르는 세 가지 단계를 포함합니다. 이 과정에서, 모델은 Monte Carlo Tree Search (MCTS)를 기반으로 하여 다채로운 자연어 계획을 생성하고, 생성된 계획에 따라 코드를 작성합니다. 모델은 이전의 답변을 참조하여 생성의 적절성과 다양성, 오답 가능성을 줄이는 데 집중합니다.

- **Performance Highlights**: 실험 결과, SRA-MCTS로부터 생성된 데이터를 사용한 파인 튜닝 모델은 기존의 Chain-of-Thought (CoT) 방식이나 공식 모델보다 뛰어난 성과를 보여줍니다. 특히, 응답의 다양성이 중요한 역할을 하며, 작은 모델에서도 자가 개선이 이루어지는 점은 주목할 만한 결과입니다. 이 방식은 특히 CoT 접근 방식이 성능 저하를 경험할 때에도 강력함을 유지하며, 높은 다양성 지표에서의 개선을 관찰할 수 있었습니다.




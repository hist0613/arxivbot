New uploads on arXiv(cs.CL)

### MM-RLHF: The Next Step Forward in Multimodal LLM Alignmen (https://arxiv.org/abs/2502.10391)
Comments:
          Project Page: this https URL

- **What's New**: 이 논문에서는 인간의 선호와 더 잘 일치하는 Multimodal Large Language Models(MLLMs)를 개발하기 위한 주요 데이터셋 MM-RLHF를 소개합니다. 이 데이터셋은 120,000개의 세부적인 인간 주석 선호 비교 쌍으로 구성되어 있으며, 기존 자원 대비 더 큰 규모, 다양성, 주석의 세분화 및 품질을 제공합니다.

- **Technical Details**: MM-RLHF의 주요 혁신 중 하나로, Critique-Based Reward Model을 도입하여 모델 출력에 대한 비평을 생성한 후 점수를 부여합니다. 또한, Dynamic Reward Scaling이라는 방법을 통해 각 샘플의 손실 가중치를 보상 신호에 따라 조정하여 고품질 비교 쌍의 활용을 최적화합니다. 이 접근 방식은 10개의 차원과 27개의 벤치마크에서 엄격하게 평가됩니다.

- **Performance Highlights**: MM-RLHF와 정렬 알고리즘을 통해 fine-tuning된 LLaVA-ov-7B는 대화 능력이 19.5% 증가하고 안전성 면에서 60% 개선된 결과를 나타냈습니다. 이 연구는 새로운 데이터셋, 보상 모델, 훈련 및 평가 코드를 오픈소스하여 커뮤니티와의 협력을 장려합니다.



### Aspect-Oriented Summarization for Psychiatric Short-Term Readmission Prediction (https://arxiv.org/abs/2502.10388)
- **What's New**: 이 논문에서는 긴 문서의 요약을 통해 LLM(대형 언어 모델)의 성능을 향상시키는 새로운 접근 방식을 제안합니다. 특히, 환자의 재입원 예측과 같은 복잡한 작업에 대해 다각적(Aspect-oriented) 요약 방법을 사용하여 정보 신호를 효과적으로 측정하고 통합하는 방법을 연구합니다. 이 연구는 정신과 병원의 환자 데이터를 활용하여 요약이 가지는 중요성을 규명했습니다.

- **Technical Details**: 연구에서 사용된 방법론은 크게 세 가지로 나뉩니다. 첫째는 Aspect-oriented summarization으로, 환자 퇴원 노트를 다양한 프롬프트를 사용해 요약하는 것입니다. 둘째는 정보 신호의 차이를 측정하는 방법으로, 요약된 문서 간의 정보 격차를 수치적으로 평가합니다. 마지막으로, 서로 다른 요약으로부터 얻어진 정보를 통합하여 예측 성능을 높이는 Gradual Fine-Tuning 방법을 제안하였습니다.

- **Performance Highlights**: 연구 결과, 제안된 방법이 정신과 병원 환자들의 30일 재입원 예측 성능을 향상시키는 것으로 나타났습니다. 서로 다른 요약을 통한 정보 통합이 단일 요약으로 훈련된 모델보다 더 효과적이라는 점이 확인되었습니다. 이는 LLM의 요약 능력을 활용하여 임상 데이터에 대한 분석을 더 정교하게 수행할 수 있음을 시사합니다.



### OWLS: Scaling Laws for Multilingual Speech Recognition and Translation Models (https://arxiv.org/abs/2502.10373)
Comments:
          23 pages, 13 figures

- **What's New**: 이 논문은 OWLS(Open Whisper-style Large-scale neural model Suite)를 소개하며, 이는 0.25B부터 18B까지의 매개변수를 가진 다국어 음성 인식 및 번역 모델의 오픈 액세스 수트입니다. OWLS는 150개 언어에서 360K 시간을 넘어선 공공 음성 데이터를 활용하여 다국어 음성 작업에서 데이터, 모델 및 계산 스케일링이 성능에 미치는 영향을 체계적으로 조사합니다. 이 연구의 주요 발견 중 하나는 스케일링이 낮은 자원 언어와 방언의 성능을 향상시켜 편향을 완화하고 음성 기술의 접근성을 개선한다는 것입니다.

- **Technical Details**: OWLS는 총 13개의 투명한 음성 기반 모델을 포함하고 있으며, 각각의 모델은 0.25B에서 18B의 매개변수를 가지고 있습니다. 이 모델들은 360K 시간의 다국어 데이터에 대해 사전 훈련되었으며, ASR과 ST의 성능을 분석하기 위해 모델 및 데이터 크기의 스케일링 영향을 체계적으로 평가합니다. 이 연구를 통해 우리는 각 작업과 언어에 대한 모델 성능의 변화를 예측할 수 있는 신경 스케일링 법칙을 도출했습니다.

- **Performance Highlights**: OWLS는 18B의 총 매개변수를 가진 ASR/ST 모델을 훈련 및 출시했고, 이는 현재까지 알려진 공개 ASR/ST 모델 중 가장 큽니다. 모델 스케일링의 유용성을 측정하는 것뿐 아니라 그것이 극복하지 못하는 실패 사례도 식별하였습니다. 또한 대규모 음성 기초 모델의 시험 시간에서의 능력을 평가하고, 큰 모델에만 존재하는 새로운 emergent abilities를 발견하여 음성 모델 스케일링이 새로운 언어에 자원 활용 학습에 어떻게 기여하는지를 보여주었습니다.



### Enhancing Multilingual LLM Pretraining with Model-Based Data Selection (https://arxiv.org/abs/2502.10361)
- **What's New**: 이 논문에서는 다국어 데이터셋을 위한 모델 기반 필터링 프레임워크를 제안합니다. 기존 연구가 주로 영어 데이터에 치중했던 점에서 벗어나, 다양한 언어의 구조적이고 지식이 풍부한 샘플을 식별하여 성과를 높이는 것을 목표로 합니다. Transformer 및 FastText 기반 분류기를 활용하여 접근 방식을 단순화하고 효율성을 높였습니다.

- **Technical Details**: 모델 기반 필터링 기술이 발전함에 따라 FineWeb-2 데이터셋을 이용한 대규모 웹 스케일의 다국어 필터링을 위한 통합 프레임워크를 구축했습니다. FastText와 Transformer 기반 임베딩을 사용하여 필터링을 수행하며, 중국어, 독일어, 프랑스어, 아랍어 및 덴마크어와 같은 다양한 언어 가족에 걸친 실험을 진행했습니다. 기존의 데이터셋에 비해 15%의 훈련 토큰으로도 MMLU 점수를 맞출 수 있음을 입증했습니다.

- **Performance Highlights**: 제안된 방법은 1B 파라미터 Llama 모델을 이용하여 70B 및 119B 토큰으로 훈련할 때, 기존의 방법과 동등한 성능을 보여주었습니다. 연구 결과, 데이터셋 오염 및 다국어 LLM 훈련의 영향을 분석하였고, 20개 언어에 대해 정제된 사전 훈련 데이터셋을 공개함으로써 다국어 자연어 처리 모델링을 한층 발전시켰습니다.



### Agentic Verification for Ambiguous Query Disambiguation (https://arxiv.org/abs/2502.10352)
- **What's New**: 이번 연구에서는 Retrieval-Augmented Generation (RAG)에서 질의의 다양성과 검증 가능성을 동시에 다룰 수 있는 새로운 방법론인 VerDICt (Verified-Diversification with Consolidation)를 제안합니다. 기존 Diversify-then-Verify (DtV) 접근 방식의 한계를 극복하기 위해, 질의 해석의 초기 단계에서 retriever와 generator의 피드백을 통합하여 불필요한 오류를 줄이는 구조로 설계되었습니다. 이는 문서 검색과 질의 해석의 일관성을 높여 더욱 신뢰성 있는 결과를 얻을 수 있도록 합니다.

- **Technical Details**: VerDICt는 효율성과 강력성을 동시에 향상시키기 위해 diversification(다양화)와 verification(검증)을 통합하는 프레임워크입니다. 첫 번째 단계에서는 retriever를 이용해 관련 있는 구문을 찾고, 이를 통해 각 해석이 문서에 적합한지를 검증합니다. 추가적으로 execution feedback(실행 피드백)을 통해 구문이 질의에 대한 답변을 제공할 수 있는지를 확인함으로써 지원되지 않는 해석을 제거합니다. 여러 피드백 신호를 클러스터링하여 일관성을 높이고, 비효율적인 후속 검증 절차를 최소화합니다.

- **Performance Highlights**: VerDICt는 ASQA 벤치마크에서 광범위하게 검증되었으며, 최적의 기본 LLM들과 비교하여 F1 점수를 평균 23% 향상시켰습니다. 이는 해석의 다양성과 정확성을 높이는 동시에, 리트리벌 및 추론 과정에서 발생할 수 있는 오류를 줄이는 성과입니다. 본 연구를 통해 VerDICt의 코드를 오픈소스로 공개하여, 향후 연구에 기여할 수 있도록 하였습니다.



### Organize the Web: Constructing Domains Enhances Pre-Training Data Curation (https://arxiv.org/abs/2502.10341)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서 제안하는 WebOrganizer 프레임워크는 웹 페이지의 주제와 형식에 따라 웹 콘텐츠를 구성하는 체계적인 방법을 제시합니다. 기존의 비구조적 데이터 세트를 세분화하고 명확한 도메인으로 안착시킬 수 있는 통찰력을 제공합니다. 이러한 접근은 사전 학습 데이터의 구성에서 질적 기준뿐만 아니라 도메인 별로도 최적화된 믹스를 통해 성능을 향상시킬 수 있음을 보여줍니다.

- **Technical Details**: WebOrganizer는 두 개의 상보적인 도메인 분류법, 즉 주제(topic)와 형식(format)을 도입하여 총 24개 카테고리로 웹 페이지를 분류합니다. 이 프레임워크는 Llama-3.1-405B-Instruct라는 대규모 언어 모델에서 길고 효율적인 분류기로 자동 라벨링을 수행하여 큐레이션 과정을 자동화합니다. 이 두 가지의 도메인 분류법을 활용하여 우리는 다가오는 작업에서의 성능 향상을 위한 데이터를 조합하고 최적화하는 방법을 개발합니다.

- **Performance Highlights**: 실험을 통해 우리의 도메인 믹스는 MMLU와 HellaSwag와 같은 하위 작업의 성능을 향상시키는 데 효과적임을 발견했습니다. FineWeb-Edu 품질 필터를 추가했을 때 평균 정확도가 54.2%에서 56.2%로 증가하는 것을 관찰했으며, 이는 품질 필터링으로 인한 성능 향상의 84%를 유지하는 것으로 나타났습니다. 우리 연구는 도메인 구성 및 혼합 방법이 데이터 큐레이션의 새로운 가능성을 열어줄 수 있음을 입증했습니다.



### STAR: Spectral Truncation and Rescale for Model Merging (https://arxiv.org/abs/2502.10339)
Comments:
          Accepted to NAACL 2025

- **What's New**: 모델 머징(model merging)은 여러 pretrained 모델을 사용하여 추가적으로 fine-tuning 없이 멀티태스킹 모델을 생성하는 효율적인 방법입니다. 본 논문에서는 ‘S$	ext{pectral} T	ext{runcation} A	ext{nd} R	ext{escale}’ (STAR)이라는 새로운 방법을 제안하여, 증가하는 모델 수로 인해 발생하는 작업 성능 저하를 완화하고자 합니다. STAR는 각 스펙트럼 공간에서 작은 구성 요소를 제거한 후, 원본 행렬의 핵 노름(nuclear norm)을 유지하기 위해 자동으로 매개변수를 재조정하여, 원본 훈련 데이터에 대한 추가적인 추론을 필요로 하지 않으며 Hyperparameter 선택에 강건합니다.

- **Technical Details**: STAR는 스펙트럼 분해(spectral decomposition)와 같은 기법을 사용하여 모델 머징에서 노이즈 성분을 제거하고, 이후 재조정(rescaling) 단계를 통해 원래의 핵 노름을 복원합니다. 이는 여러 모델과 다양한 작업의 경우에도 잘 작동하며, 20개 모델까지 성공적으로 머징할 수 있습니다. 또한, STAR는 기존의 방법론들과는 달리, 데이터 접근 권한이 없는 상황에서도 유용하게 사용될 수 있습니다.

- **Performance Highlights**: STAR는 다양한 모델 크기 설정에서 우수한 성능을 입증하였으며, 특정 NLP 작업인 Flan-T5의 경우 12개 모델을 머징할 때 기존 기준선 대비 4.2% 향상된 성능을 보여주었습니다. 이 같은 성과는 필요한 추가 추론이 없고 파라미터에 대한 민감도가 낮기 때문에 이루어진 것입니다. 연구자들은 STAR의 공개 코드를 통해 다양한 NLP 작업에서의 적용 가능성을 확대할 수 있을 것입니다.



### Evaluating the Meta- and Object-Level Reasoning of Large Language Models for Question Answering (https://arxiv.org/abs/2502.10338)
Comments:
          8 pages. Accepted to the Workshop on Planning in the Era of LLMs (LM4Plan @ AAAI 2025)

- **What's New**: 이 논문은 대규모 언어 모델(LLMs)이 자연어 작업에서 우수한 성능을 보이나, 복잡한 다단계 사고가 요구되는 질문 응답(QA) 작업에서 도전과제를 안고 있음을 강조합니다. 기존 LLM이 처하는 이유를 재정의 하여 meta-level reasoning(메타 수준 사고)과 object-level reasoning(객체 수준 사고)으로 나누었습니다. 새로운 데이터셋인 Franklin이 도입되어 LLM의 질문 응답 성능을 평가하는 데 사용되었습니다.

- **Technical Details**: 이 논문에서는 LLM의 reasoning(사고) 작업을 구체적으로 논의하며, multi-step reasoning(다단계 사고)의 중요성을 보여줍니다. 연구는 meta- and object-level reasoning을 평가할 수 있는 Franklin 데이터셋을 포함한 세 가지 다른 데이터셋을 사용하여 진행되었습니다. 실험 결과, LLM은 meta-level reasoning을 자주 보여주지만, object-level reasoning 작업에서는 어려움을 겪고 있는 것으로 나타났습니다.

- **Performance Highlights**: 대상 데이터셋에서 LLM들의 성능이 함께 비교되었습니다. 연구 결과, 대다수 LLM은 object-level reasoning이 부족하여 어려움을 겪었지만, meta-level reasoning에서는 일관되게 높은 성능을 보였습니다. 또한 Franklin 데이터셋은 LLM에게 도전 과제를 제공하여, 이들 모델의 강점과 약점을 체계적으로 분석하는 기회를 마련했습니다.



### Are Large Language Models the future crowd workers of Linguistics? (https://arxiv.org/abs/2502.10266)
- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)이 경험적 언어학 연구에서 인간 참여자를 대체하거나 보완할 수 있는 가능성을 탐구합니다. 연구진은 두 가지 사례 연구를 통해 LLM의 효과를 확인하였으며, OpenAI의 GPT-4o-mini 모델을 사용하여 인간 참여자와 유사한 성과를 달성했음을 보고했습니다. 이로 인해 LLM의 성능이 인간 정보 제공자의 수준을 초과할 수 있다는 점이 강조되었습니다.

- **Technical Details**: 연구에서는 LLM의 행동을 전통적으로 인간을 위해 설계된 작업에 대해 테스트하고, 기본적인 프롬프트 엔지니어링 프레임워크를 개발하여 비전문가도 LLM의 잠재력을 탐색할 수 있도록 하였습니다. 두 개의 복제 연구에서 LLM의 적용 가능성을 명확히 하고, Chain-of-Thought (CoT) 기법과 같은 추가적인 프롬프트 기술을 탐구할 필요성이 제기되었습니다. 또한, 이 논문은 LLM이 복잡한 판단을 수행할 수 있는 가능성을 보여줍니다.

- **Performance Highlights**: 결과적으로, 첫 번째 복제 연구에서 GPT-4o-mini는 인간 참여자와 상당히 잘 맞아떨어지는 성능을 보였으며, 두 번째 과제는 LLM과 인간 간의 매칭에 대한 보다 세밀한 관점을 제공합니다. 또한, 모든 실험 조건에서 GPT-4o-mini가 인간 정보 제공자를 초과하는 성능을 나타냈습니다. 이러한 결과는 LLM을 경험적 언어학 연구에 활용하는 데 있어 연구자들에게 필요한 학제 간 접근의 길을 열어줄 것으로 기대됩니다.



### Large Language Models and Synthetic Data for Monitoring Dataset Mentions in Research Papers (https://arxiv.org/abs/2502.10263)
Comments:
          Project GitHub repository at this https URL

- **What's New**: 이번 논문은 연구 논문에서 데이터셋 언급을 자동으로 탐지하는 머신러닝 프레임워크를 제안합니다. 대규모 언어 모델(large language models, LLMs)과 합성 데이터(synthetic data)를 활용하여, 데이터셋 언급을 효율적으로 식별하고 분류할 수 있는 기술을 개발했습니다. 이 과정은 인력 자원 소모를 줄이고, 논문 전반에서 데이터셋의 가시성을 높이는 데 기여할 수 있습니다.

- **Technical Details**: 이 프레임워크는 제로샷 추출(zero-shot extraction) 및 LLM을 활용한 품질 평가(LMM-as-a-Judge)를 통해 약한 감독 가상 데이터셋(weakly supervised synthetic dataset)을 생성합니다. Phi-3.5-mini instruct 모델은 이러한 데이터셋에서 사전 파인 튜닝(pre-fine-tuning)되고, 이후 수동으로 주석이 달린 하위 집합에 대해 파인 튜닝을 진행합니다. 최종적으로는 ModernBERT 기반의 분류기가 데이터셋 언급을 효과적으로 필터링하여 계산 부담을 줄입니다.

- **Performance Highlights**: 이 연구는 수동으로 주석이 달린 샘플에서 평가하였고, 파인 튜닝된 모델이 NuExtract-v1.5와 GLiNER-large-v2.1보다 데이터셋 추출 정확도에서 우수한 성능을 보였습니다. LLM이 생성한 합성 데이터는 훈련 데이터의 부족 문제를 효과적으로 해결하고, 자원이 적은 환경에서의 일반화 성능을 개선할 수 있음을 입증했습니다. 이 프레임워크는 데이터 사용 모니터링의 확장 가능성을 제시하며, 연구자 및 정책 입안자에게 데이터 접근성을 높이는 데 도움을 줄 것입니다.



### VisCon-100K: Leveraging Contextual Web Data for Fine-tuning Vision Language Models (https://arxiv.org/abs/2502.10250)
Comments:
          Accepted at PAKDD 2025

- **What's New**: 본 논문에서는 VisCon-100K라는 새로운 데이터셋을 소개합니다. 이 데이터셋은 interleaved한 이미지-텍스트 웹 문서에서 파생되어, 45K개의 웹 문서를 100K개의 이미지 대화 샘플로 변환하는 방식으로 생성되었습니다. GPT-4V를 활용해 이미지 관련 캡션을 생성하고, OpenChat 3.5 모델로 다양한 질문-답변 쌍으로 변환하여 VLM의 성능을 개선합니다.

- **Technical Details**: VisCon-100K 데이터셋은 고유의 pipeline을 통해 생성되었으며, 이 과정에서 OpenAI GPT-4V API를 사용해 이미지 문맥 캡션을 생성합니다. 이후 OpenChat 3.5를 사용하여 이 캡션을 자유형식 및 다중 선택 질문-답변 쌍으로 변환합니다. 새로운 접근법인 'leaky modality mix'를 통해 이미지와 그 문맥 캡션 모두에서 답변이 가능한 질문을 포함함으로써 성능을 높였습니다.

- **Performance Highlights**: VisCon-100K 데이터셋은 ShareGPT4V-7b와 IDEFICS2-8b 등 두 가지 주요 VLM 접근 방식에서 우수한 성능을 보여줍니다. 특히, 이 데이터셋은 그동안의 다른 데이터셋들에 비해 더욱 풍부하고 다양한 학습 자원을 제공하여 비전-언어 모델의 파인튜닝을 효과적으로 지원합니다. 논문에서 제출한 훈련된 문맥 캡셔너는 고품질의 문맥적 캡션을 생성해 추가 연구 및 오픈 소스 응용을 촉진합니다.



### Can Post-Training Quantization Benefit from an Additional QLoRA Integration? (https://arxiv.org/abs/2502.10202)
Comments:
          Accepted to NAACL 2025 Industry Track

- **What's New**: 본 연구에서는 전통적인 Post-training Quantization (PTQ) 기법과 QLoRA의 통합을 통해 4비트 양자화 모델의 성능을 향상시키는 새로운 접근법을 제시합니다. 이 통합이 기존의 PTQ 방법보다 뛰어난 성능을 보임을 실험을 통해 입증하였으며, 때로는 16비트 풀 파라미터 미세조정 모델보다도 우수하다는 것을 보여줍니다. 이러한 결과는 높은 성능을 유지하면서도 자원이 제한된 환경에서 강력한 LLM을 배포할 수 있는 가능성을 열어줍니다.

- **Technical Details**: 양자화는 모델의 메모리 사용과 계산 효율성을 최적화하기 위해 사용되며, 본 연구에서는 4비트 PTQ와 QLoRA를 결합하는 방법에 집중합니다. PTQ는 모델 훈련이 완료된 후에 적용되며, QLoRA는 양자화된 모델에서 파라미터 효율적인 미세조정을 수행하여 정확성 손실을 줄입니다. 이 연구는 Mistral, Qwen2, LLaMA2와 같은 여러 사전 훈련된 모델을 사용하여 다양한 실험을 수행했습니다.

- **Performance Highlights**: 실험 결과, PTQ-QLoRA 통합 모델은 다양한 데이터셋에서 16비트 풀 미세조정 모델에 필적하는 성능을 보여주었습니다. 내부 데이터 및 공개 데이터셋을 사용한 평가에서, 양자화된 모델이 메모리 제약 속에서도 높은 성능을 유지할 수 있음을 나타내었습니다. 이러한 성과는 비즈니스 애플리케이션에 적합한 실용적인 솔루션을 제공합니다.



### Prediction hubs are context-informed frequent tokens in LLMs (https://arxiv.org/abs/2502.10201)
- **What's New**: 이 논문에서는 고차원 데이터에서 발생하는 허브니스(hubness) 현상과 대규모 자가 회귀 언어 모델(autoregressive large language models, LLMs)에서의 허브니스의 영향을 연구합니다. 저자들은 LLM에서 확률을 결정하는 방식에서 허브니스가 어떻게 나타나는지를 수학적으로 및 경험적으로 분석합니다. 연구 결과 LLM의 특정 예측 작업에서는 허브니스가 부정적인 성질이 아닐 수 있으며, 자주 등장하는 토큰의 확률을 높이는 전략으로 작용함을 보입니다.

- **Technical Details**: 허브니스는 고차원 데이터에서 특정 데이터 포인트가 다른 많은 포인트의 k-최근접 이웃에 포함되는 경향을 나타냅니다. LLM에서는 컨텍스트와 언임베딩 벡터 간의 관계를 통해 예측을 수행하며, 이 연산은 확률 거리(probability distance)라고 불리는 새로운 측정 방식으로 나타납니다. 의도된 거리 기반 계산에서 허브니스가 발생할 수 있지만, 이는 자연어 텍스트의 단어 분포가 불균형하여 자주 등장하는 토큰과 관련이 있습니다.

- **Performance Highlights**: 저자들은 LLM의 예측 과정에서 발생하는 허브니스가 단순한 거리 집중 현상 때문이 아니라, 결과적으로 적절한 추정을 위한 전략임을 보였으며, 이러한 허브는 제거할 필요가 없다고 주장합니다. 반면, 다른 유사성 비교에 관해서는 허브니스가 부정적인 영향을 미칠 수 있으며, 따라서 이러한 계산을 수행할 때 허브니스 감소 기법과 결합해야 함을 강조합니다.



### Small Models, Big Impact: Efficient Corpus and Graph-Based Adaptation of Small Multilingual Language Models for Low-Resource Languages (https://arxiv.org/abs/2502.10140)
Comments:
          Pre-print

- **What's New**: 본 연구는 데이터가 부족한 저자원 언어(Low-resource languages, LRLs)에 대해 매개변수 효율적인 어댑터 기반 방법을 체계적으로 조사했습니다. 특히, Sequential Bottleneck, Invertible Bottleneck, Low-Rank Adaptation과 같은 세 가지 아키텍처를 평가하여 LRL에 대한 소수의 작은 다국어 모델(multilingual models, mLMs)의 적응 성능을 분석했습니다. 또한 이를 통해 LRL을 위한 더 적은 양의 데이터가 큰 개선을 이룰 수 있음을 발견하였습니다.

- **Technical Details**: 본 연구에서 사용된 방법론은 세 가지 어댑터 아키텍처인 Sequential Bottleneck, Invertible Bottleneck, Low-Rank Adaptation을 포함합니다. 이들은 mBERT 및 XLM-R과 같은 소형 mLMs에 적용되어 언어 모델링 및 다운스트림 작업에서 성능을 평가합니다. 또한, 비구조적 텍스트와 구조적 지식을 활용하여 LRL에 대한 어댑터 기반 적응의 효과를 탐구하였습니다.

- **Performance Highlights**: 어댑터 기반 접근 방식은 전체 파인튜닝과 비교하여 적은 수의 학습 가능한 매개변수로 동일한 성능 또는 우수한 성능을 달성했습니다. Sequential Bottleneck이 언어 모델링에서 우수했으며, Invertible Bottleneck은 다운스트림 작업에서 더 나은 성능을 보였습니다. 결과적으로 작은 mLM들이 LRL에 대해 더 효과적으로 작동하며, 거대한 LLM에 비해 뛰어난 성능을 나타냈습니다.



### Hands-off Image Editing: Language-guided Editing without any Task-specific Labeling, Masking or even Training (https://arxiv.org/abs/2502.10064)
Comments:
          Published in COLING 2025

- **What's New**: 이 논문은 기존의 Instruction-guided image editing 방법의 한계를 극복하기 위해 새로운 접근 방식을 제안합니다. 이 방법은 기존의 task-specific supervision이나 데이터셋, 혹은 특정한 훈련을 요구하지 않으므로, 더 나은 개선 가능성을 제공합니다. 실험 결과는 제안한 방법이 매우 효과적이며, 경쟁력 있는 성능을 달성했다는 것을 보여줍니다.

- **Technical Details**: 제안된 방법은 입력 이미지와 출력 이미지의 캡션을 이용해 차이 벡터(difference vector)를 생성하여 이미지를 편집합니다. 입력 이미지의 캡션은 일반적인 이미지-텍스트 툴을 사용해 얻고, 출력 이미지의 캡션은 편집 지침을 포함한 적절한 프롬프트 템플릿을 통해 생성됩니다. 이 과정은 수동적 작업이나 추가 훈련 없이 자동으로 이루어질 수 있습니다.

- **Performance Highlights**: 제안된 방법은 기존의 state-of-the-art 방법들과 비교했을 때 자원 소모가 적으면서도 경쟁력 있는 성능을 보였습니다. 또한, 단순히 훈련된 모델들(LLM와 이미지 생성 모델)을 사용하므로, 이러한 모델의 성능이 향상되면 우리 방법의 성능도 증가할 가능성이 있습니다.



### Annotating Compositionality Scores for Irish Noun Compounds is Hard Work (https://arxiv.org/abs/2502.10061)
Comments:
          6 pages, 1 figure

- **What's New**: 이 논문에서는 아일랜드어 텍스트에서 발견된 명사 복합어(Noun Compounds, NCs)의 분석 결과를 제시합니다. 이 연구는 구성성(compositionality)과 도메인 특이성(domain specificity) 등을 주요 특징으로 삼아 NC의 의도성을 조사하였습니다. 이 데이터 세트는 아일랜드어 NC에 대한 공개 사용 가능한 주석이 포함되어 있으며, 이는 아일랜드어에서 NC의 분포와 특성을 더 깊이 이해하는 데 기여하고자 합니다.

- **Technical Details**: 명사 복합어는 현대 자연어 처리(NLP) 시스템의 발전에 핵심적인 복잡한 언어 표현의 의미를 이해하는 데 중요합니다. 연구를 통해 다룬 아일랜드어 NC의 주석 가이드라인은 예외적인 사례를 해결하기 위한 노력의 일환으로 만들어졌으며, 아일랜드어의 다양한 도메인에서 NC를 식별하고 분석하였습니다. 또한 이 연구는 아일랜드어 NC와 영어 NC의 차별성을 알아보고자 합니다.

- **Performance Highlights**: 연구 결과, 아일랜드어 NC의 주석 과정에서 사전 성과와 주석자 간의 신뢰도에 대한 조사가 이루어졌습니다. 데이터셋의 주석 작업은 여러 아일랜드어 방언 특징을 반영하며, 연구자들은 하이브리드 접근 방식을 통해 NC의 구성성 캡처 능력을 검증하고 있습니다. 이 연구는 아일랜드어로 된 첫 번째 유형의 데이터 세트이며, 아일랜드어 및 언어 모델 수용력을 평가하는 데 중요한 역할을 할 것으로 예상됩니다.



### MTLM: an Innovative Language Model Training Paradigm for ASR (https://arxiv.org/abs/2502.10058)
- **What's New**: 이 논문은 자동 음성 인식(ASR) 성능을 향상시키기 위해 전통적인 유일방향 언어 모델(ULM)을 양방향으로 활용할 수 있는 새로운 학습 방법을 제안합니다. 이 접근법은 왼쪽과 오른쪽 맥락을 모두 사용할 수 있도록 하여 유일방향 언어 모델의 제약을 극복하며, 보다 일관된 가정(transcribe hypotheses)을 생성할 수 있게 합니다. 실험 결과는 제안한 모델이 기존 유일방향 언어 모델에 비해 성능이 우수하다는 것을 보여줍니다.

- **Technical Details**: 이 논문에서는 ULM과 마스크 언어 모델링(MLM) 작업을 결합하여 언어 모델의 학습 효율성과 의미 포착의 완전성을 향상시키는 방법을 설명합니다. 특히, Unidirectional Masked Language Modeling (UMLM) 작업을 설계하여 ULM 훈련과 MLM 훈련 사이의 차이를 줄이고 정보를 효율적으로 교류할 수 있도록 합니다. 모델은 Transformer 기반이며, 여러 동일한 레이어로 구성되어 다중 헤드 자기 주의(multi-head attention)와 피드포워드(feedforward) 레이어를 포함합니다.

- **Performance Highlights**: LibriSpeech 데이터셋에서 전통적인 유일방향 LM에 비해 Word Error Rate (WER)를显著 감소시켰습니다. 테스트 클린 데이터셋에서는 WER이 3.18%에서 2.63%로 줄었고, 테스트 기타 데이터셋에서도 8.78%에서 7.08%로 감소했습니다. 이러한 성능 향상은 제안한 UMLM 훈련 패러다임의 우수성을 입증하며, 오류 유형의 수도 1/6로 줄어드는 결과를 보여주었습니다.



### ORI: O Routing Intelligenc (https://arxiv.org/abs/2502.10051)
Comments:
          13 pages, 2 figures

- **What's New**: 본 논문은 ORI(O Routing Intelligence)라는 동적 프레임워크를 제안하여 단일 대형 언어 모델(LLM)이 처리하기 어려운 복잡한 작업을 해결하고자 합니다. ORI는 여러 LLM 중 가장 적합한 모델로 쿼리를 효율적으로 라우팅함으로써 작업별 정확도를 높이고 계산 비용을 최소화합니다. 새로운 평가 방식과 다양한 벤치마크에서 일관된 정확성 향상을 입증하였습니다.

- **Technical Details**: ORI는 벡터 공간 표현(vector space representations) 및 정교한 분류 알고리즘을 기반으로 하여 쿼리 특성에 따라 적합한 LLM을 선택합니다. 기존의 라우팅 프레임워크에서 보이는 인간의 선호 데이터에 대한 의존도를 줄여 편향을 최소화하고 다양한 작업에서의 일반화 능력을 향상시켰습니다. ORI의 적응형 아키텍처는 변화하는 쿼리 복잡성을 효율적으로 처리할 수 있도록 설계되었습니다.

- **Performance Highlights**: ORI는 MMLU에서 최대 2.7점, MuSR에서 1.8점의 정확도 향상을 보여주었으며, ARC와 BBH에서는 상위 성능을 동점으로 기록했습니다. 이 결과는 다중 모델 전략의 장점을 강조하며 ORI의 적응형 접근 방식이 다양한 작업을 보다 효과적으로 처리할 수 있음을 보여줍니다. 포괄적인 평가를 통해 ORI가 다양한 벤치마크에서 일관되고 높은 품질의 결과를 제공함을 입증하였습니다.



### Probabilistic Lexical Manifold Construction in Large Language Models via Hierarchical Vector Field Interpolation (https://arxiv.org/abs/2502.10013)
- **What's New**: 이 연구에서 제안하는 Hierarchical Vector Field Interpolation(계층적 벡터 필드 보간법)은 단어 임베딩을 효과적으로 구조화하여 기존의 변환 모델(Transformer Models)에서 자주 발생하는 표현의 불연속성을 완화합니다. 이 방법론은 단어 표현이 확률적 일관성을 유지하면서도 업계에서 널리 사용되는 대규모 모델에 적용 가능하도록 설계되었습니다. 실험 결과는 이 방식이 언어 모델의 의미적 안정성을 향상시키고 치밀한 의미 구별을 필요로 하는 과제에서 더 나은 성과를 거둘 수 있음을 보여줍니다.

- **Technical Details**: 제안된 접근 방식은 확률적 함수 공간 내에서 단어 표현을 구조화하여, 자연어의 본질적인 연속성을 반영하는 보간 과정을 수립합니다. 이 과정은 토큰화의 한계를 넘어 연속적인 확률적 함수로서 단어 간의 관계를 정의하며, 이를 통해 다층 언어 표현의 일관성을 유지할 수 있습니다. 연구는 또한 계층적 제약 조건을 활용하여 여러 언어 표현의 스케일에서의 일관성을 높이는 수학적 메커니즘을 도입합니다.

- **Performance Highlights**: 실험적인 결과들은 이 방법이 전통적인 단어 임베딩과 비교할 때 더 높은 밀도 정렬 및 컨텍스트 임베딩 배포의 왜곡을 줄이는 데 효과적임을 입증했습니다. 계층적 벡터 필드 보간법을 사용하여 구조화된 임베딩 공간의 이점을 강조하며, 대규모 언어 모델에서 실용성을 고려한 확장성과 효율성을 유지하고 있습니다. 결과적으로, 이 연구의 접근 방식은 기존의 변환 기반 모델보다 더 나은 구조적이고 해석 가능한 단어 표현을 제공하는 데 기여합니다.



### SciClaimHunt: A Large Dataset for Evidence-based Scientific Claim Verification (https://arxiv.org/abs/2502.10003)
- **What's New**: 본 연구에서는 기존의 과학적 주장 검증 데이터셋의 한계를 극복하고자 SciClaimHunt 및 SciClaimHunt_Num이라는 두 가지 대규모 데이터셋을 제안합니다. 이러한 데이터셋은 과학 연구 논문에서 결정된 주장을 바탕으로 하며, 과학적 주장과 증거 간의 관계를 검증하는 데 사용됩니다. 특히, SciClaimHunt는 Large Language Models (LLMs)를 활용하여 몇 가지 문서에서 지원되는 주장을 생성하고, SciClaimHunt_Num은 숫자 값을 포함한 주장을 다룹니다.

- **Technical Details**: 제안된 SciClaimHunt 데이터셋은 과학적 증거에 의해 지지되는 주장을 기반으로 하며, 반박된 주장은 두 가지 방법으로 생성됩니다: (i) 과학적 증거에 의해 지지되는 주장을 부정하고, (ii) 과학적 주장 내에서 명명된 개체를 대체합니다. SciClaimHunt_Num 데이터셋은 숫자나 카드값을 포함하는 주장에 특화되어 있으며, 이러한 주장의 일관성과 함께 과학적 증거와의 맥락적 유사성을 확인하는 것이 중요합니다. 이를 통해 과학적 주장 검증을 위한 새로운 접근방식이 마련되었습니다.

- **Performance Highlights**: 모델이 SciClaimHunt 및 SciClaimHunt_Num에서 훈련될 때 기존 데이터셋과 비교했을 때 높은 품질과 신뢰성을 보이는 결과를 나타냈습니다. 또한, 여러 차별적 연구와 인간 평가를 통해 제안된 데이터셋의 효과성을 검증하였습니다. 이러한 결과는 SciClaimHunt와 SciClaimHunt_Num이 과학적 주장 검증 모델의 훈련에 효과적인 자원임을 제안합니다.



### EmbBERT-Q: Breaking Memory Barriers in Embedded NLP (https://arxiv.org/abs/2502.10001)
Comments:
          24 pages, 4 figures, 14 tables

- **What's New**: 이 논문에서는 저메모리 환경에서 실행 가능하도록 설계된 새로운 언어 모델인 EmbBERT-Q를 소개합니다. EmbBERT-Q는 기존의 대형 언어 모델이 갖는 높은 메모리와 컴퓨팅 요구사항을 해결하여, 착용형 기기 및 IoT 장치와 같은 기술적으로 제약된 소형 장치에서 사용할 수 있도록 최적화되었습니다. 특히 이 모델은 Natural Language Processing(NLP) 작업에서 상위 성능(State-of-the-Art, SotA)을 기록하며, 크기가 단 781 kB로 기존 모델에 비해 25배 감소한 특징을 가지고 있습니다.

- **Technical Details**: EmbBERT-Q는 아키텍처 혁신과 하드웨어 호환 8비트 양자화(quantization)를 결합하여 설계되었습니다. 이 모델은 TinyNLP라는 변별력 있는 벤치마크 데이터셋과 GLUE 벤치마크를 사용하여 철저한 실험적 평가를 받았습니다. 결과적으로, EmbBERT-Q는 2 MB 메모리 예산으로 축소된 여러 기준 모델보다 일관되게 높은 성능을 발휘하며, BERT와 MAMBA의 압축 버전들보다도 두드러진 우수성을 보여줍니다.

- **Performance Highlights**: EmbBERT-Q는 메모리 사용과 성능 간의 균형을 최적으로 유지하면서, 기존 접근 방식과 비교하여 경쟁력 있는 정확도를 달성합니다. 이 모델은 Tiny Language Models에서 NLP 작업 성능을 평가하기 위해 특별히 설계된 TinyNLP 데이터셋과 실제 시나리오 모두에서 그 효과를 입증하였습니다. 모든 결과의 즉각적인 재현 가능성을 보장하기 위해, 본 연구팀은 코드 및 모델 체크포인트를 공개했습니다.



### Large Language Diffusion Models (https://arxiv.org/abs/2502.09992)
- **What's New**: LLaDA는 오토회귀 모델(Autoregressive Models, ARMs)의 전통적인 개념에 도전하며 새로운 확산 모델(Diffusion Model)을 소개합니다. 이는 처음부터 끝까지 단독으로 훈련되어, 확률적 추론을 위한 주어진 유도 생성 방식에 의해 분포를 예측합니다. LLaDA는 기존 ARM 기준선을 초월하여 강력한 확장성을 보여주며, 인-컨텍스트 학습에서 LLaMA3 8B와 경쟁할 수 있는 성능을 드러냅니다.

- **Technical Details**: LLaDA는 마스크된 확산 모델(Masked Diffusion Model, MDM)을 통해 두 개의 프로세스를 정의합니다. 전방 프로세스는 시퀀스를 점진적으로 마스크화하면서, 반대의 과정은 마스크된 토큰을 예측하여 데이터 분포를 복구합니다. 이 구조는 기존의 오토회귀 방식과는 달리, 양방향 의존성을 통해生成 모델링의 원리를 활용합니다.

- **Performance Highlights**: LLaDA는 다양한 작업에서 뛰어난 성과를 발휘하며, 특히 8B 모델은 2.3조 개의 토큰으로 훈련된 후, 지침을 따르는 능력과 다중 턴 대화 사례 연구에서 인상적인 결과를 나타냅니다. 또한 LLaDA는 회귀 시나리오에서 GPT-4o를 초월하며, 이전의 오토회귀 모델들이 가지던 한계를 극복하는 데 성공했습니다.



### LaRA: Benchmarking Retrieval-Augmented Generation and Long-Context LLMs - No Silver Bullet for LC or RAG Routing (https://arxiv.org/abs/2502.09977)
Comments:
          22 pages

- **What's New**: 이 논문은 RAG( retrieval-Augmented Generation)와 LC(long-context) LLMs의 효과적인 비교를 위한 새로운 벤치마크인 LaRA를 제안합니다. LaRA는 2,326개의 테스트 사례로 구성되어 있으며, 이는 두 접근 방식이 어떻게 차별화되는지를 명확히 분석할 수 있도록 설계되었습니다. 이를 통해 RAG와 LC의 강점과 약점을 체계적으로 비교하는 데 기여할 것입니다.

- **Technical Details**: LaRA는 다양한 질문 답변(Question Answering, QA) 과제를 평가하기 위해 자연발생적인 긴 텍스트로 구성된 데이터셋을 사용합니다. 이 데이터셋에는 소설, 학술 논문, 재무 제표 등 다양한 서식이 포함되어 있어 여러 문체와 정보 밀도를 대표합니다. 또한, LaRA의 QA 쌍은 인간 주석자와 LLM의 협력으로 생성되며, 정확성을 보장하기 위해 GPT-4o를 사용하여 예측 판별을 수행합니다.

- **Performance Highlights**: 연구 결과, RAG와 LC의 선택은 모델의 파라미터 크기, 긴 텍스트 처리 능력, 컨텍스트 길이, 작업 유형 및 검색된 조각의 특성과 같은 여러 요인에 따라 달라진다는 것을 확인했습니다. 저자들은 이를 이용해 LLM 애플리케이션을 최적화하는 방법에 대한 실용적인 가이드를 제공하며, 강력한 모델일수록 LC가 더 좋은 성능을 보인다는 점을 강조하였습니다.



### KGGen: Extracting Knowledge Graphs from Plain Text with Language Models (https://arxiv.org/abs/2502.09956)
- **What's New**: 최근 지식 그래프(KG) 구축에 대한 관심이 높아지면서 데이터 부족 문제가 중요하게 여겨지고 있습니다. KGGen은 자연어 텍스트로부터 고품질의 지식 그래프를 생성하는 텍스트-투-KG(generator) 패키지로, Python 라이브러리 형태로 제공됩니다. KGGen은 관련된 엔티티를 클러스터링하여 추출된 KGs의 희소성을 줄이는 혁신적인 접근법을 채택했습니다. 또한, 새로운 벤치마크인 MINE을 통해 기존 추출기와 비교해 18% 더 높은 성능을 보였습니다.

- **Technical Details**: KGGen은 LLM(대형 언어 모델)을 활용하여 평문에서 주어-서술어-목적어(triple)를 추출하고, 클러스터링 알고리즘을 통해 고품질의 조밀한 KG를 생성합니다. 이 과정에서 여러 단계를 거치며, 첫 번째 단계에서 비구조화된 텍스트를 입력받아 초기 지식 그래프를 생성하고, 그 다음에 유일한 엔티티와 연결 관계를 집계합니다. KGGen은 각 단계에서 DSPy 프레임워크를 사용해 일관된 JSON 형식의 출력을 보장합니다.

- **Performance Highlights**: KGGen은 벤치마크 테스트에서 기존 텍스트-투-KG 추출기보다 18% 더 뛰어난 성능을 나타냈습니다. 이 성능 향상은 KGGen이 고품질의 밀접하게 연결된 KGs를 자동으로 생성할 수 있는 잠재력을 보여줍니다. KGGen의 도입으로 오는 데이터 풍부한 미래는 차세대 KG 기반 모델 훈련과 RAG 시스템에 긍정적인 영향을 미칠 것으로 기대됩니다.



### A Preliminary Exploration with GPT-4o Voice Mod (https://arxiv.org/abs/2502.09940)
Comments:
          Work in progress

- **What's New**: 최근 멀티모달 대형 언어 모델(large language models) 분야에서 GPT-4o가 주목받고 있습니다. 이 보고서는 GPT-4o의 오디오 처리와 추론 능력을 평가하였으며, 의도 분류, 음성 명령 분류 등 다양한 작업에서 강력한 성능을 보였습니다. 특히 GPT-4o는 다른 대형 오디오-언어 모델(LALMs)에 비해 환각(hallucination)에 대한 내구성이 높지만, 오디오 지속 시간 예측 작업에서는 어려움을 겪습니다.

- **Technical Details**: 이 보고서는 Dynamic-SUPERB와 같은 방대한 기준을 포함하여 다양한 작업에서 GPT-4o의 오디오 이해 및 추론 능력을 평가합니다. Dynamic-SUPERB는 지시 기반 보편적인 음성 모델을 평가하기 위해 설계된 대규모 벤치마크이며, MMAU는 LALMs의 추론과 이해 능력을 평가하는 데 중점을 두고 있습니다. 또한 현재 GPT-4o의 안전 메커니즘은 민감하게 작동하여, 스피커 식별 및 심각하지 않은 작업에 대해 거부할 수 있습니다.

- **Performance Highlights**: GPT-4o는 여러 벤치마크에서 전반적으로 뛰어난 성능을 보였고, 특히 음성 인식 및 음악 이해 작업에서 강점을 나타냈습니다. 그러나 특정 작업, 예를 들어 오디오 지속 시간 예측과 같은 경우, 다른 기초 모델들과 비교하여 성능이 저조한 결과를 보였습니다. 이 보고서는 GPT-4o의 성능을 평가하기 위한 초기 탐색을 제공하며, LALMs의 현재 상태에 대한 통찰력을 제공합니다.



### Efficient Multitask Learning in Small Language Models Through Upside-Down Reinforcement Learning (https://arxiv.org/abs/2502.09854)
- **What's New**: 본 연구에서는 소형 언어 모델(SLM), 특히 100M 매개변수를 가진 GPT-2 모델이 대형 언어 모델(LLM)과 비교하여 멀티태스크 프롬프트 생성 작업에서 경쟁력 있는 성능을 낼 수 있음을 보여줍니다. 이는 업사이드-다운 강화 학습(upside-down reinforcement learning)과 강력한 LLM인 Llama-3의 합성 데이터 증류(synthetic data distillation)의 독창적 조합을 통해 이루어졌으며, 이는 80배 더 작은 SLM이 최신 기술 모델과 거의 동일한 관련성 점수를 달성하는 데 기여했습니다.

- **Technical Details**: SLM은 멀티모달 프롬프트 생성 작업을 위한 효율적이고 효과적인 멀티태스크 학습자로서 활용될 수 있도록 설계되었습니다. 연구진은 Llama-3를 사용하여 고품질의 훈련 데이터 집합을 만들고, 이를 통해 SLM은 최소한의 리소스에서 효과적으로 학습하도록 구성되었습니다. 업사이드-다운 강화 학습을 통해 SLM은 제어된 생성 프로세스를 최적화하여 특정 속성(예: 길이 및 관련성)을 만족하는 출력을 생성하도록 훈련됩니다.

- **Performance Highlights**: SLM은 Llama-3를 포함한 최신 LLM과 비교하여 매개변수 수가 약 1/80에 불과하지만 유사한 성능을 달성하며, 단일 A10G GPU에서 초당 338개의 토큰을 처리할 수 있는 속도를 기록했습니다. 이러한 성능은 자원이 제한된 환경에서 실시간 응용 프로그램의 요구사항을 충족시키기에 적합합니다. 연구진은 이 프레임워크가 상업적인 텍스트-이미지 생성 시스템과 통합될 수 있는 가능성을 강조하여 실용적인 응용 프로그램에서의 활용도를 높이고 있습니다.



### Statistical Coherence Alignment for Large Language Model Representation Learning Through Tensor Field Convergenc (https://arxiv.org/abs/2502.09815)
- **What's New**: 본 논문에서는 통계적 일관성 정렬(Statistical Coherence Alignment, SCA)이라는 새로운 방법론을 도입하여, 내부 표현을 언어의 통계적 속성과 정렬함으로써 자연어 처리 모델의 성능을 향상시키고자 한다. 이를 통해 모델은 더 깊은 의미적 관계를 포착할 수 있으며, 훈련 과정에서 일관성 있는 표현을 최적화하는 손실 함수도 통합된다. 실험 결과는 제안된 방법이 기존 모델보다 더 해석 가능한 내부 구조를 제공하며, 문맥 의존성을 보존하여 표현 붕괴를 완화한다는 점을 보여준다.

- **Technical Details**: SCA는 텐서 필드 수렴(tensor field convergence) 기법을 활용하여 모델의 내부 표현을 언어의 통계적 속성과 일치시킨다. 각 토큰 임베딩은 컨텍스트 종속성을 모델링하는 연속적인 텐서 필드로 표현된다. 이러한 접근 방식은 전통적인 주의 메커니즘이 가진 제한점을 극복하고, 토큰 간의 컨텍스트 정보를 효과적으로 전파하여 긴 범위 의존성을 유지하면서도 의미적인 표현을 제공할 수 있도록 한다.

- **Performance Highlights**: 실험 결과 SCA는 혼란도(perplexity)와 분류 정확도를 향상시키고, 드문 단어 임베딩을 개선하여 더 안정적인 표현 공간을 제공하는 것으로 확인되었다. 제안된 방법은 기억 및 훈련 비용이 증가하는 단점을 수반하지만, 맥락 일관성을 요하는 응용 프로그램에서는 이러한 비용이 정당화될 수 있다. 전체적으로 SCA는 토큰 표현 최적화에서 효과적임을 검증하며, 통계적 의존성을 활용하여 언어 모델 훈련을 개선할 수 있는 통찰력을 제공한다.



### INJONGO: A Multicultural Intent Detection and Slot-filling Dataset for 16 African Languages (https://arxiv.org/abs/2502.09814)
- **What's New**: 이 논문에서는 Injongo라는 다문화, 오픈 소스 벤치마크 데이터셋을 소개합니다. 이 데이터셋은 16개 아프리카 언어를 포함하며, 본토 화자가 다양한 도메인(예: 은행, 여행, 가정, 외식)에 대해 생성한 발화(utterances)로 구성되어 있습니다. 기존의 대규모 벤치마크가 저자원(low-resource) 언어를 평가하지 않았던 점을 해결하고 있습니다.

- **Technical Details**: 논문에서는 다국어 트랜스포머 모델(multilingual transformer models)과 프롬프팅 대형 언어 모델(prompting large language models, LLMs)의 미세 조정(fine-tuning) 과정에 대한 광범위한 실험을 수행했습니다. 이러한 실험을 통해 아프리카 문화의 발화를 활용하는 것이 서구 중심의 발화보다 교차 언어 전이(cross-lingual transfer)를 개선하는 데 유리하다는 점을 보여주었습니다. 실험 결과 LLM들이 슬롯 채우기(slot-filling) 작업에서 어려움을 겪고 있으며, 평균 F1-score는 26에 불과합니다.

- **Performance Highlights**: 의도 탐지(intent detection)의 성능은 평균 정확도 70.6%로 상대적으로 더 나아 보이지만, 여전히 미세 조정된 베이스라인(fine-tuning baselines)에는 미치지 못합니다. 영어와 비교했을 때, GPT-4o와 미세 조정된 베이스라인은 의도 탐지에서 약 81%의 유사한 정확도를 보였습니다. 하지만 현재 LLM들은 많은 저자원 아프리카 언어에 대해서는 여전히 부족한 성능을 보이며, 그들의 하급 성능(downstream performance)을 개선하기 위한 추가 작업이 필요함을 시사합니다.



### Prompt and circumstance: A word-by-word LLM prompting approach to interlinear glossing for low-resource languages (https://arxiv.org/abs/2502.09778)
- **What's New**: 본 연구에서는 안내(linguistic instructions)를 따를 수 있는 LLMs의 기능을 활용하여 자동화된 interlinear glossed text (IGT)를 생성할 수 있는 가능성을 탐구합니다. 특히 SIGMORPHON 2023 공유 과제에서 제공된 일곱 언어에 대한 glossing 정확성을 평가하는 방법을 제안합니다. 결과적으로, 우리의 시스템은 BERT 기반의 기존 모델을 각 언어의 형태소 수준 점수 카테고리에서 초과하며, 특정 언어에서는 인간 주석자에게 높은 품질의 제안을 제공합니다.

- **Technical Details**: 우리는 단어별 검색 기반의 prompting 방식을 사용하여 각 언어로 된 문장을 glossing하고, 이는 기존의 전체 문장 prompting 방법과는 다릅니다. 이러한 접근법을 통해 LLM은 단어 수준의 폴리시나 문장을 선택하여 인간 주석자를 위해 어시스트하는 길잡이 역할을 할 수 있습니다. 연구 결과, 간단한 3-best oracle이 대부분의 언어에서 챌린지 우승자를 초과한다는 것을 보여주었습니다.

- **Performance Highlights**: 개별 언어에서 자동으로 생성된 통계적 지침이 특정 난해한 문법 기능의 오류를 줄일 수 있다는 점을 알 수 있었습니다. Tsez 언어의 경우, LLM으로 생성된 언어적 지침이 10%의 오류 감소를 가져옵니다. 이상의 성과는 LLMs가 주석 작업에서 실질적인 지원을 제공할 수 있는 가능성을 시사합니다.



### The Widespread Adoption of Large Language Model-Assisted Writing Across Society (https://arxiv.org/abs/2502.09747)
- **What's New**: 이 논문은 큰 언어 모델(LLM)의 채택 패턴을 분석하기 위한 첫 번째 대규모 체계적 연구를 제시합니다. 연구의 범위에는 소비자 불만, 기업 커뮤니케이션, 구인 공고 및 국제 기구 보도 자료의 네 가지 영역이 포함됩니다. LLM 사용의 급증은 2022년 ChatGPT 출시 이후 발생했으며, 지역과 도시에 따라 다소 다른 채택 패턴을 보였습니다.

- **Technical Details**: 이 연구는 2022년 1월부터 2024년 9월까지 수집된 687,241건의 소비자 불만, 537,413건의 기업 보도 자료, 3억 4,300만 건의 구인 공고, 15,919건의 유엔 보도 자료를 포함하는 대규모 데이터 세트를 기반으로 합니다. 연구진은 통계적 프레임워크를 통해 LLM 수정 콘텐츠의 유병률을 정량화하였으며, 각 영역에서의 시간 경과에 따른 채택 동향을 분석했습니다.

- **Performance Highlights**: 연구의 결과, 소비자 불만의 약 18%, 기업 보도 자료의 약 24%, 구인 공고의 약 10%가 LLM의 영향을 받은 것으로 나타났습니다. 특히 작은 기업에서는 LLM 사용 비율이 더욱 높았고, 유엔 보도 자료에서도 이와 유사한 경향이 나타났습니다. 이러한 결과는 LLM의 채택이 다양한 사회 경제적 및 조직적 요인에 의해 어떻게 형성되는지를 보여줍니다.



### Partial Colexifications Improve Concept Embeddings (https://arxiv.org/abs/2502.09743)
Comments:
          Submitted to the 63rd Annual Meeting of the Association for Computational Linguistics, Vienna, Austria

- **What's New**: 본 논문은 개념의 임베딩(concept embedding)에 대한 최신 연구를 다루고 있습니다. 지금까지 개념 임베딩은 자동으로 구축된 colexification 네트워크를 기반으로 하였으나, 본 연구에서는 부분 colexification 데이터를 포함하여 개념을 보다 의미 있게 표현하는 방법을 제안합니다. 이를 통해 다양한 언어에서의 의미를 반영할 수 있는 더 나은 개념 표현을 구현하고자 합니다.

- **Technical Details**: 연구에서는 그래프 임베딩(graph embedding) 기법을 활용하여 전체 및 부분 colexification 데이터를 통해 개념 임베딩을 학습하는 방법을 소개합니다. 부분 colexification은 동일한 단어 형태가 서로 다른 의미를 표현하는 상황을 포함하여, 개념 네트워크를 더욱 풍부하게 합니다. 이 방법은 의미 유사성 모델링, 의미 변화 예측, 단어 연관 예측의 세 가지 과제를 통해 검증되었습니다.

- **Performance Highlights**: 결과적으로, 부분 colexification으로 개선된 개념 임베딩은 세 가지 평가 작업에서 모두 더 나은 성능을 보였습니다. 이 연구는 기존의 정적 단어 임베딩(static word embedding) 방식보다도 나은 결과를 보여 주며, 구조적인 네트워크에서 유추된 유사성 지표보다도 뛰어난 성능을 발휘하는 것으로 나타났습니다.



### FoNE: Precise Single-Token Number Embeddings via Fourier Features (https://arxiv.org/abs/2502.09741)
- **What's New**: 이번 연구에서는 Fourier Number Embedding (FoNE)이라는 새로운 방법을 제안하고 있습니다. FoNE는 숫자를 직접 Fourier 표현으로 매핑해 주며, 이 과정에서 토큰화 단계가 제외됩니다. 각 숫자를 두 차원의 단일 토큰으로 인코딩하여 숫자 값의 프래그멘테이션을 제거함으로써 훈련과 추론 효율을 크게 향상시킵니다.

- **Technical Details**: FoNE는 숫자에서 자주 나타나는 주기적 특성을 활용하여, 각 자릿수를 코사인 및 사인 함수를 사용하여 인코딩합니다. 이 접근법은 각 자릿수의 모듈 관계를 통해 정확한 숫자 표현을 보장하고, 이를 통해 어떠한 숫자도 단일 토큰으로 표현할 수 있습니다. FoNE는 또한 수학적 특성을 따르기 위해 주기적 임베딩을 사용하여 각 자릿수를 정확하게 복원할 수 있게 합니다.

- **Performance Highlights**: FoNE는 기본 산술작업인 덧셈, 뺄셈, 곱셈에서 100%의 정확도를 달성하며, 기존의 방법들에 비해 훨씬 적은 훈련 데이터와 파라미터로도 높은 정확성을 제공합니다. 또한 FoNE를 사용함으로써 훈련 시간과 추론 시간이 단축되고, 숫자 인코딩을 단일 토큰으로 수행하여 계산 효율성을 크게 개선합니다.



### Trust at Your Own Peril: A Mixed Methods Exploration of the Ability of Large Language Models to Generate Expert-Like Systems Engineering Artifacts and a Characterization of Failure Modes (https://arxiv.org/abs/2502.09690)
Comments:
          41 pages, 10 figures

- **What's New**: 이번 연구에서는 Multi-purpose Large Language Models (다목적 대형 언어 모델, LLMs)가 시스템 엔지니어링(Systems Engineering, SE) 작업에서의 효율성에 대한 의문을 제기하고 있습니다. 인간 전문가가 생성한 SE 산출물(artifacts)을 기준으로 삼고 LLMs가 생성한 산출물과 비교하여, AI가 생성한 결과물의 품질에 대해 경고하는 내용을 담고 있습니다.

- **Technical Details**: 연구 방법은 두 가지로 나뉩니다. 첫째, 여러 LLM에 다양한 프롬프트(prompt)를 통해 SE 산출물의 세그먼트를 생성하도록 하여 기초 성능을 문서화하였습니다. 둘째, 자연어 처리 알고리즘을 사용하여 정량적으로 AI 생성 산출물과 인간 전문가 벤치마크를 비교하였으며, 정성적으로 품질 차이도 분석하였습니다.

- **Performance Highlights**: AI가 생성한 산출물이 인간 전문가의 기준과 매우 유사해 보이는 반면, 몇 가지 중대한 실패 모드를 드러내었습니다. 여기에는 조기 요구 사항 정의, 근거 없는 수치 추정 및 과도한 세부정보 지정 경향이 포함되며, 이러한 문제로 인해 시스템 엔지니어링 분야에서는 AI의 피드백을 신중하게 수용해야 함을 강조하고 있습니다.



### Large Language Models and Provenance Metadata for Determining the Relevance of Images and Videos in News Stories (https://arxiv.org/abs/2502.09689)
- **What's New**: 이 논문은 효과적인 허위 정보 캠페인이 텍스트와 함께 이미지 및 비디오를 활용하여 잘못된 내러티브(narrative)를 강화하는 방식을 탐구합니다. 기존의 허위 정보 탐지 기법들은 서로 다른 매체 간의 상호작용을 간과하는 경향이 있으며, 본 연구는 대형 언어 모델(LLM)을 기반으로 이러한 문제를 해결하고자 합니다. 연구에서는 뉴스 기사의 텍스트와 포함된 미디어의 출처 메타데이터를 분석하여 미디어가 해당 기사의 맥락과 관련이 있는지 판단하는 시스템을 제시합니다.

- **Technical Details**: 제안된 방법은 뉴스 기사의 제목, 본문 및 첨부된 미디어를 입력으로 받아 미디어의 출처 및 편집 여부가 관련성이 있는지를 평가합니다. 이 과정에서 데이터 출처(provenance) 메타데이터가 사용되며, 이는 정보의 정확성과 진위성을 보장하는 데 필수적입니다. 메타데이터는 중앙 발행 기관 또는 블록체인을 통해 조작 방지 기능을 포함할 수 있습니다.

- **Performance Highlights**: 제안된 접근법은 다양한 데이터 출처 프레임워크 및 LLM과 함께 작동할 수 있도록 설계되어 기술의 발전을 지속적으로 활용할 수 있는 장점을 가지고 있습니다. 프로토타입 시스템은 오픈소스로 제공되며, 실제 뉴스 기사에서 해당 미디어가 관련성이 있는지를 분석할 수 있는 웹 인터페이스를 제공합니다. 이러한 기술은 소셜 미디어 및 블로그 게시물과 같은 다른 맥락에도 적용 가능성이 있습니다.



### Mind What You Ask For: Emotional and Rational Faces of Persuasion by Large Language Models (https://arxiv.org/abs/2502.09687)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)이 정서적 또는 이성적 유인을 사용할 때의 언어 패턴 차이를 분석하고, LLM이 어떻게 사회적 영향 원리를 적용하는지를 살펴보았습니다. 연구 결과로 정서적 유인은 인지적 복잡성을 높이는 데 기여할 수 있음을 강조하고, LLM이 정서적 유인에 대해 어떤 반응을 보이는지를 보여주었습니다. 또한, LLM이 생성하는 응답이 사회적 영향 원리를 근거로 만들어진다는 점도 드러났습니다.

- **Technical Details**: 연구에서는 서로 다른 크기와 라이선스를 가진 12개 LLM 모델을 선택하여 이성적 유인과 감정적 유인을 비교했습니다. 사용된 LLM은 OpenAI의 GPT-3.5 Turbo, GPT-4, Mistral, Meta의 Llama 3 시리즈, Anthropic의 Claude 모델들입니다. 각 모델에 대해 표준화된 프롬프트를 통해 다양한 변수를 포함한 데이터 세트를 사용하여 일관된 결과를 도출하였습니다.

- **Performance Highlights**: 이번 연구는 LLM의 언어 패턴 차이를 도출하며, 감정적 유인 사용 시 더 다양하고 복잡한 응답을 생성한다는 것을 밝혀냈습니다. 또한, 연구는 특정 모델들이 유인 방식에 따라 메시지를 적절히 조정할 수 있는 반면, 다른 모델들은 유사한 사회적 영향 원리를 적용하는 것을 보여주었습니다. LLM이 효과적으로 대중의 신념과 태도에 영향을 미칠 수 있는 방법을 제시하는 중요한 결과를 도출했습니다.



### Multi-level Conflict-Aware Network for Multi-modal Sentiment Analysis (https://arxiv.org/abs/2502.09675)
Comments:
          5 pages, 1 figure

- **What's New**: 이번 연구에서는 다중 모달 감정 분석(Multimodal Sentiment Analysis, MSA)에 있어 새로운 다수준(多水準) 갈등 인식 네트워크(Multi-level Conflict-aware Network, MCAN)를 제안합니다. MCAN은 각 모달의 정합성과 갈등 요소를 효과적으로 분리하여 모델링하며, 기계 학습의 불안정성을 감소시키기 위한 새로운 접근 방식을 채택했습니다. 이를 통해 기존 연구들에서 다루지 않았던 비모달 조합 간의 갈등 요소도 고려하고 있습니다.

- **Technical Details**: MCAN의 구조는 메인 브랜치(Main Branch)와 갈등 모델링 브랜치(Conflict Modeling Branch)로 나뉘어 있습니다. 메인 브랜치는 Micro Multi-step Interaction Network(Micro-MSIN) 및 Macro Multi-step Intersection Network(Macro-MSIN)를 활용하여 단일 모달 및 이중 모달 간의 관계를 점진적으로 모델링합니다. 갈등 모델링 브랜치에서는 마이크로 및 매크로 갈등 인식 크로스 어텐션(Micro-CACA, Macro-CACA)을 사용하여 갈등 요소를 모델링하며, 생성된 레이블의 의존성을 피하고 있습니다.

- **Performance Highlights**: MCAN은 CMU-MOSI와 CMU-MOSEI 데이터셋에서 기존 최선 기법들에 비해 현저한 성능 향상을 보였으며, 다양한 실험을 통해 제안된 기법의 효과성이 입증되었습니다. 특히, MCAN의 핵심 구성 요소와 주요 하이퍼파라미터의 영향을 평가하여 모델의 정확도를 높이는 데 기여했습니다. 이로 인해 다중 모달 데이터의 감정 인식에서 새로운 표준을 제시할 수 있을 것으로 기대됩니다.



### The Hidden Dimensions of LLM Alignment: A Multi-Dimensional Safety Analysis (https://arxiv.org/abs/2502.09674)
Comments:
          Code and artifacts: this https URL

- **What's New**: 이 연구는 기존의 LLM(Long Language Models)의 안전 정렬 행동(safety-aligned behaviors)이 단일 방향이 아닌 다차원 방향에 의해 제어됨을 발견하였습니다. 이는 'activation space' 내에서 안전 행동을 표현하는 새로운 방법인 'Safety Residual Space'를 소개하면서 시작합니다. 또한, 여러 개의 해석 가능한 방향(interpretable directions)을 탐구하여 모델의 안전 회피(refusal behavior)에 대한 이해를 심화 시킵니다.

- **Technical Details**: 안전 분석에서 LLM의 다양한 안전 요소를 구분하기 위해 Linear Representation Hypothesis를 기반으로 하는 프레임워크를 구축하였습니다. 이 과정에서 안전 세분화 바탕으로 모델이 보이는 방향(feature direction)을 탐색하며, 각 방향이 서로 어떻게 상호작용하는지를 분석합니다. 이러한 다차원 해석을 통해 특정 트리거 토큰의 삭제가 안전 정렬을 우회하는 데 어떻게 영향을 미치는지에 대한 통찰을 제공합니다.

- **Performance Highlights**: 연구 결과, 특정한 방향들이 LLM의 거부 행동을 결정짓는 지배적인 역할을 한다는 것을 발견했습니다. 또한, 여러 비지배적인 방향이 안전 fine-tuning 동안 학습되는 다양한 능력에 대한 조절 역할을 한다는 것을 실험을 통해 확인하였습니다. 이러한 통찰은 LLM의 취약성을 이해하고 개선하는 데 중요한 기초 자료로 활용될 수 있을 것입니다.



### Are Smarter LLMs Safer? Exploring Safety-Reasoning Trade-offs in Prompting and Fine-Tuning (https://arxiv.org/abs/2502.09673)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)의 추론 능력이 향상되면서 나타나는 안전성(혹은 safety) 문제를 조명합니다. 복잡한 작업에서의 향상된 성능은 중요한 반면, 이로 인해 발생할 수 있는 새로운 취약점(vulnerability)에 대한 경각심을 일깨우고 있습니다. 연구에서는 이러한 문제와 문제 해결을 위한 새로운 접근 방식을 제시합니다.

- **Technical Details**: 연구는 LLM의 추론 능력과 안전성 간의 상호작용(interplay)을 분석하여, 추론이 발전함에 따라 발생할 수 있는 잠재적 안전 위험(latent safety risks)을 조명합니다. 또한, 안전성을 증대시키기 위해 추론을 활용할 수 있는 방법들을 탐구합니다. 이러한 접근은 기존의 접근 방식과 다게, 보다 다각적인 관점에서 문제를 접근하게 합니다.

- **Performance Highlights**: 연구 결과는 향상된 추론 능력이 모델의 복잡한 작업 수행 능력을 높이는 동시에, 새로운 안전 문제의 출현 가능성을 시사합니다. 연구는 모델의 안전성을 강화하는 데 있어 추론을 활용할 수 있는 잠재적인 완화 전략(potential mitigation strategies)을 제시하여, LLM이 실제 배치 시 더 신뢰성(trustworthy) 있는 성능을 발휘할 수 있도록 기여합니다.



### The Science of Evaluating Foundation Models (https://arxiv.org/abs/2502.09670)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)의 체계적 평가 과정을 형성화하고, 구체적인 사용 사례와 관련된 평가 방법론을 제시하는 데 초점을 맞추고 있습니다. Rigorously structured evaluation guidelines는 다양한 모델을 비교하고, 이를 토대로 사용자가 현실 애플리케이션에 LLM을 어떻게 통합할 수 있는지를 명확히 설명합니다. 또한 체크리스트와 문서 템플릿과 같은 실행 가능한 도구를 제공하여 평가 프로세스를 지원합니다.

- **Technical Details**: 논문에서 제안된 'ABCD in Evaluation' 프레임워크는 알고리즘(Algorithm), 대규모 데이터(Big Data), 계산 자원(Computation Resources), 도메인 전문성(Domain Expertise)을 포함하는 체계적인 평가 접근법을 제공합니다. 이는 LLM의 평가 과정에서 알고리즘의 역할, 데이터의 중요성, 계산 인프라의 요구 사항 및 관련 도메인 지식의 필요성을 강조합니다. 이 프레임워크는 LLM 평가의 복잡성을 깊이 이해할 수 있는 기초를 제공합니다.

- **Performance Highlights**: 제시된 평가 프레임워크는 LLM의 성능을 정확하게 평가하는 데 필요한 다양한 차원을 포괄합니다. 이는 대규모 공개 벤치마크뿐만 아니라 도메인 별 데이터셋을 활용하여 모델의 다양성과 공정성을 측정하는 데 기여합니다. 궁극적으로, 이 연구는 LLM 평가를 위한 유의미한 기준을 제공하고, 종합적인 평가 접근법을 통해 다양한 산업에 걸쳐 모델을 협의적이고 책임감 있게 배포하는 데 도움을 줄 것입니다.



### k-LLMmeans: Summaries as Centroids for Interpretable and Scalable LLM-Based Text Clustering (https://arxiv.org/abs/2502.09667)
- **What's New**: 본 논문에서는 k-LLMmeans라는 새로운 k-평균 클러스터링 알고리즘의 수정판을 제안합니다. 이 알고리즘은 LLMs(large language models)를 활용해 클러스터 중심을 텍스트 요약으로 생성하여 문서 임베딩의 수치적 평균에 의존하는 전통적인 접근법의 한계를 극복합니다. 이 방법은 k-평균의 특성을 유지하면서도 해석 가능성을 높입니다.

- **Technical Details**: k-LLMmeans는 클러스터 내용을 요약한 텍스트를 기반으로 클러스터 중심을 생성하여, 숫자 평균의 손실을 줄이는 데 중점을 둡니다. 이 과정에서 LLM이 생성한 요약의 임베딩이 클러스터 할당을 안내합니다. 또한, 연속적인 데이터 스트림을 위한 미니 배치 변형을 제안하여 실시간 해석 가능성을 제공합니다.

- **Performance Highlights**: 상세한 시뮬레이션 결과, 제안된 방법이 여러 지표에서 전통적인 k-평균보다 우수한 성능을 보이며 데이터 세트 크기에 따라 LLM의 사용량이 크게 증가하지 않음을 보여줍니다. 이 연구는 StackExchange에서 새로운 데이터 세트를 수집하여 텍스트 스트림 클러스터링의 벤치마크를 제공합니다.



### Cancer Vaccine Adjuvant Name Recognition from Biomedical Literature using Large Language Models (https://arxiv.org/abs/2502.09659)
Comments:
          10 pages, 6 figures, 4 tables

- **What's New**: 이 연구는 암 백신 연구에서 adjuvant(보조제)의 이름을 자동으로 인식하는 방법을 탐구합니다. 기존의 생물의학 문헌에서 수작업으로 보조제를 분류하는 것의 어려움을 극복하기 위해, 대규모 언어 모델(LLMs)인 GPT와 Llama를 활용하여 그 가능성을 증명했습니다.

- **Technical Details**: 연구에 사용된 데이터셋은 AdjuvareDB의 97개의 임상 시험 기록과 Vaccine Adjuvant Compendium(VAC)에서 주석 처리된 290개의 초록입니다. GPT-4o와 Llama 3.2는 zero-shot과 few-shot 학습 방식을 통해 보조제 이름 인식을 수행했고, 포맷된 프롬프트에 따라 다양한 맥락 정보를 포함하여 성능을 테스트하였습니다.

- **Performance Highlights**: GPT-4o는 모든 상황에서 100%의 Precision(정확도)을 기록하며 Recall(재현율)과 F1-score(조화 평균)에서도 뛰어난 성능을 보여주었습니다. 특히 VAC 데이터셋에서 F1-score 77.32%를 기록하며, AdjuvareDB 데이터셋에서는 81.67%의 F1-score를 달성했습니다. 이로써 모델의 전반적인 보조제 식별 능력이 입증되었습니다.



### Neuro-Conceptual Artificial Intelligence: Integrating OPM with Deep Learning to Enhance Question Answering Quality (https://arxiv.org/abs/2502.09658)
Comments:
          15 pages, 3 figures,

- **What's New**: 이 논문에서는 Neural-symbolic AI 접근법의 전문화된 형태인 Neuro-Conceptual Artificial Intelligence (NCAI)를 소개합니다. NCAI는 Object-Process Methodology (OPM)를 활용하여 질의응답(QA) 품질을 향상시키고 투명한 AI 시스템을 구축하는 데 초점을 맞추고 있습니다. 이 방법은 자연어 텍스트를 OPM 모델로 변환하여 제기되는 복잡한 개념을 처리하는 데 도움을 줍니다.

- **Technical Details**: NCAI는 OPM을 사용하여 프로세스, 객체 및 상태 등 복잡한 OPM 요소를 표현합니다. OPM-QA 시스템에서의 구조화된 지식 표현은 전통적인 트리플 기반 지식 그래프가 포착하기 어려운 복잡한 프로세스를 처리할 수 있게 합니다. 연구자들은 추가적으로 OPM 기반 개념 논리에 얼마나 충실하게 예측된 추론이 일치하는지를 측정하는 투명성 평가 메트릭스를 제안하였습니다.

- **Performance Highlights**: 실험 결과, NCAI는 전통적인 방법보다 뛰어난 성능을 보였으며, 복잡한 지식 표현을 제공하여 추론의 투명성을 향상시킵니다. NCAI는 측정 가능한 투명성과 향상된 추론을 통해 신경-상징 AI(neuro-symbolic AI)의 발전 가능성을 강조하고 있습니다.



### AI-VERDE: A Gateway for Egalitarian Access to Large Language Model-Based Resources For Educational Institutions (https://arxiv.org/abs/2502.09651)
Comments:
          7 Pages, includes appendix. Submitted to NAACL System demonstrations track 2025

- **What's New**: AI-VERDE는 상업용, 클라우드 호스팅 및 온프레미스 개방형 대형 언어 모델(LLM)의 원활한 통합을 위한 통합 플랫폼 서비스를 제공합니다. 이 플랫폼은 강의 및 연구 그룹을 위한 견고한 접근 제어, 개인정보 보호 메커니즘, 기본 Retrieval-Augmented Generation (RAG) 지원 등 다양한 기능을 제공하여 대학 내에서 LLM 사용을 간소화합니다. 특히, AI-VERDE는 고등 교육 기관 내에서 LLM을 활용하는 학문적 및 연구적 요구를 충족하는 최초의 플랫폼으로 알려져 있습니다.

- **Technical Details**: AI-VERDE는 RAG 파이프라인을 기반으로 하여 다양한 교육 및 연구 그룹에 맞춤형으로 LLM에 접근할 수 있는 시스템입니다. 이 플랫폼은 오픈소스 기술을 활용하고, Kubernetes를 통해 각 구성 요소를 독립적으로 서비스할 수 있는 마이크로 서비스 구조를 갖추고 있습니다. 플랫폼의 핵심인 vLLM을 사용하여 GPU 클러스터에서 LLM을 지속적으로 로드하고, LiteLLM을 통해 통합된 API 접근성을 제공합니다.

- **Performance Highlights**: AI-VERDE의 초기 배포에서 다양한 교육 및 연구 그룹 간에 높은 참여율이 나타났으며, 이는 상업용 LLM 서비스에 비해 저렴한 비용으로 제공되었습니다. 설문 조사에 따르면, 대부분의 교수진과 학생들이 데이터 개인 정보 보호 및 제어 부족을 우려하며, AI-VERDE는 이러한 문제를 해결하여 사용자가 자신의 자료에 대한 통제권을 가질 수 있도록 하였습니다. 이 플랫폼은 교육 환경에서 LLM을 효과적으로 운영할 수 있도록 설계되었습니다.



### Principled Data Selection for Alignment: The Hidden Risks of Difficult Examples (https://arxiv.org/abs/2502.09650)
- **What's New**: 본 논문에서는 대형 언어 모델(LLM) 훈련에 있어 데이터 선택과 모델 용량 간의 관계를 새롭게 조명합니다. 저자들은 지나치게 어려운 예제가 정렬(alignment)을 저해할 수 있다는 원칙을 제안하며, 이를 통해 모델의 성능을 향상시킬 수 있음을 실험을 통해 입증했습니다. 구체적으로, Selective DPO라는 새로운 방법론을 통해 이러한 어려운 예제를 걸러내는 접근법을 소개하고 있습니다.

- **Technical Details**: 선택적 DPO는 지나치게 어려운 예제를 필터링함으로써 정렬 성능을 향상시키는 방법입니다. 연구진은 3가지 주요 주장으로 이러한 원칙을 뒷받침합니다: (1) 선호 데이터는 난이도에 따라 분류될 수 있다, (2) 지나치게 어려운 예제는 정렬 성능을 해칠 수 있다, (3) 난이도는 모델의 용량에 비례한다. 이러한 실험을 통해 LLM의 성능 향상에 기여할 수 있는 데이터 선택 원칙을 한걸음 나아가 정립했습니다.

- **Performance Highlights**: Selective DPO는 AlpacaEval 2 벤치마크에서 기존 DPO에 비해 9-16%의 승률 향상을 가져왔습니다. 또한, 이 방법은 SimPO 및 R-DPO와 같은 최신 기법들보다 우수한 성능을 보이며, 더 나은 perplexity와 암묵적 보상 마진을 유지하는 것으로 나타났습니다. 이러한 결과는 데이터 난이도와 모델 용량의 적절한 조화를 이룰 필요성을 강조합니다.



### UKTA: Unified Korean Text Analyzer (https://arxiv.org/abs/2502.09648)
Comments:
          Accepted by SAC 2025

- **What's New**: 이번 논문은 한국어 글쓰기 평가의 자동화 도구인 UKTA(Unified Korean Text Analyzer)를 소개합니다. UKTA는 한국어 텍스트를 분석하고 평가하는 포괄적인 시스템으로, 정확한 저수준(morpheme level) 형태소 분석과 중간 수준에서의 주요 Lexical feature를 제공하며, 명확한 고수준(rubric-based) 작문 점수를 제공합니다. 기존 한국어 텍스트 분석 도구의 한계를 극복하며, 다면적(multi-view) 접근 방식을 통해 글쓰기 평가의 정확성을 향상시킵니다.

- **Technical Details**: UKTA는 형태소 분석(morpheme analysis), 어휘 다양성(lexical diversity), 의미 응집성(semantic cohesion) 등의 특성을 고려하는 다면적 분석을 지원합니다. 보편적인 한국어 문법 특성에 적합한 오류 전파(error propagation) 방지 메커니즘을 통해, 초기 단계에서 발생한 오류가 최종 작문 평가에 미치는 영향을 최소화합니다. 또한, 고수준 평가 결과는 인간이 이해할 수 있게 해설하는 기능을 제공함으로써 평가의 신뢰성을 보장합니다.

- **Performance Highlights**: UKTA는 제안된 모든 특성을 사용함으로써 기존의 기준선 모형에 비해 정확성과 Quadratic Weighted Kappa 점수를 향상시켰습니다. 실험 결과, 각각의 feature가 포함된 경우 글쓰기 평가의 정확도가 현저히 개선됨을 보여주었으며, 이는 글쓰기 평가에 대한 보다 신뢰할 수 있는 접근 방식을 제공합니다. UKTA는 한국어 텍스트 분석 및 글쓰기 평가 도구로서 주목받을만한 잠재력을 지니고 있습니다.



### Unveiling Simplicities of Attention: Adaptive Long-Context Head Identification (https://arxiv.org/abs/2502.09647)
- **What's New**: 본 연구에서는 긴 문맥(context) 처리의 중요성을 강조하고, 이를 위해 LLMs(대형 언어 모델)의 어텐션(attention) 메커니즘을 분석합니다. 특정 어텐션 헤드(head)가 지역(local) 정보에만 집중하는 반면, 다른 헤드는 쿼리에 따라 지역 및 긴 문맥 정보를 전환하는 경향이 있음을 관찰했습니다. 이를 통해 긴 문맥 정보를 필요로 하는 헤드를 예측할 수 있는 가능성을 제시합니다.

- **Technical Details**: 연구에서는 2차 통계(second moment approximations)를 활용하여 긴 문맥 점수를 예측하는 간단한 모델을 제안합니다. 이 모델은 LLMs의 디코더 전용 트랜스포머 구조를 기반으로 하며, 로터리 위치 인코딩(rotary positional encoding)을 사용합니다. 아울러, 쿼리에 적응한 어텐션 기준(query-adaptive attention criterion, QAdA)을 통해 긴 문맥 헤드를 효율적으로 식별할 수 있는 방법을 소개합니다.

- **Performance Highlights**: 실험 결과, 제안된 QAdA 기준이 정적(static) 기준보다 더 높은 스파시티(sparsity)를 달성하면서도 다운스트림 태스크 성능의 손실을 최소화하는 것으로 나타났습니다. Llama, Qwen, Mistral의 세 가지 LLM 패밀리에서 표준 긴 문맥 벤치마크와 어려운 추론 작업에 적용된 결과, 효율적이고 강력한 성능 개선을 이루었다고 보고되었습니다.



### Language Shift or Maintenance? An Intergenerational Study of the Tibetan Community in Saudi Arabia (https://arxiv.org/abs/2502.09646)
- **What's New**: 이 연구는 티베트 지역에서 사우디 아라비아로 이민 간 티베트 가문의 후손들 사이에서 티베트어에서 아랍어(Arabic)로의 언어 이동에 관한 최초의 보고서를 제공합니다. 연구의 목표는 세 가지 연령대가 티베트어를 유지하거나 히자즈 아랍어로의 이동을 어떻게 수행하고 있는지를 조사하는 것이었습니다.

- **Technical Details**: 96명의 남녀 티베트 공동체 구성원이 설문조사에 응답하여 가정, 이웃, 친구 및 친척, 감정 표현, 종교 의식을 포함한 다양한 영역에서의 코드 선택(code choice)에 대해 질문을 받았습니다. 데이터는 아랍어로의 이동 정도에서 세대 간(intergenerational) 유의미한 차이가 있음을 나타냈으며, 젊은 구성원들은 티베트어를 거의 사용하지 않고, 나이 많은 구성원들은 약간 더 많이 사용하는 것으로 나타났습니다.

- **Performance Highlights**: 세 가지 연령대 간의 차이는 유의미하게(p-value .001) 나타났으며, 이는 언어 이동이 젊은 세대에서 더욱 두드러진다는 것을 암시합니다. 이러한 결과는 언어 보존의 어려움을 보여주며, 다문화 사회에서의 언어 변화에 대한 중요한 통찰을 제공합니다.



### From No to Know: Taxonomy, Challenges, and Opportunities for Negation Understanding in Multimodal Foundation Models (https://arxiv.org/abs/2502.09645)
- **What's New**: 이 논문은 다국어 다중모달 모델들이 부정(Negation) 처리에서 겪는 문제는 물론, 이를 개선하기 위한 포괄적인 분류법과 제안된 벤치마크를 제시합니다. 부정은 단순한 부정 표현을 넘어 다양한 구조적, 의미적, 문화적 요소에 의해 영향을 받음을 강조하며, 본 연구는 부정 처리를 정확하게 하기 위한 기초적인 가이드라인을 제시합니다.

- **Technical Details**: 부정 처리의 복잡성을 해결하기 위한 전략으로, 언어 특정 토크나이제이션(language-specific tokenization), 정교한 주의 메커니즘(fine-grained attention mechanisms), 그리고 고급 다중모달 아키텍처(advanced multimodal architectures)의 필요성을 제안합니다. 이러한 접근 방식은 부정 이해를 더욱 정교하게 하고, 다국어 환경 속에서 복잡한 상황을 처리할 수 있는 모델을 개발하도록 돕습니다.

- **Performance Highlights**: 부정 표현을 잘 처리하지 못하면, 챗봇 응답, 의료 이미지 분석, 법률 문서 작성에서 중요한 오류를 발생시킬 수 있습니다. 이 연구는 부정 이해를 강화함으로써 모델의 구성 및 논리적 사고(compositional and logical reasoning)를 개선하고, 단일 언어 및 다국어 환경에서의 성능 차이를 알아보는 벤치마크의 필요성을 강조합니다.



### From Argumentation to Deliberation: Perspectivized Stance Vectors for Fine-grained (Dis)agreement Analysis (https://arxiv.org/abs/2502.09644)
Comments:
          Accepted at NAACL Findings 2025

- **What's New**: 이 연구는 Perspectivized Stance Vectors(PSVs)를 개발하여 논쟁에서 표현된 다양한 입장을 분석하는 새로운 프레임워크를 제시합니다. 이는 각 논쟁자의 입장이 추구하는 가치를 이해하고, 상반된 시각에서의 동의를 구별하는 데 도움을 줄 수 있습니다. 이 과정을 통해 상반된 의견 속에서도 문제 해결을 위한 실행 가능한 포인트를 찾는 데 중점을 두고 있습니다.

- **Technical Details**: PSVs는 이슈 특정 개념에 대한 논쟁자의 입장을 벡터 형태로 표현합니다. 이를 위해 각 논쟁에 대한 입장 및 이슈에 특정한 개념들을 정리하고, 해당 개념에 대한 논쟁자의 입장을 예측합니다. 이러한 분석은 표면적인 동의와 불일치를 넘어서, 더욱 세분화된 통찰들을 제공합니다.

- **Performance Highlights**: 연구팀은 pakt라는 구조화된 논쟁 코퍼스에서 실험을 수행하여 PSVs의 성능을 평가했습니다. 이 과정에서 각 모듈의 성능을 수작업으로 주석을 단 평가 세트와 비교하여, PSVs를 통해 긍정적인 결과를 얻었다고 합니다. 그 결과, PSVs는 입장 간의 동의와 불일치뿐만 아니라 수평적(orthogonal) 관계를 감지하는 데 매우 유용한 도구로 나타났습니다.



### Krutrim LLM: Multilingual Foundational Model for over a Billion Peop (https://arxiv.org/abs/2502.09642)
- **What's New**: 이번 연구에서는 인도의 언어적 다양성을 고려한 최초의 대규모 다국어 모델인 Krutrim LLM을 소개합니다. 기존 AI 모델들이 영어에 초점을 두고 훈련되어 인도 내 다양한 언어가 효과적으로 반영되지 못하는 문제를 해결하기 위해, Krutrim LLM은 2조 개의 토큰을 사용하여 훈련되었습니다. 이 모델은 인도 사회의 언어적 풍경을 반영하며, 다양한 방언에서의 균형 잡힌 성능을 보장합니다.

- **Technical Details**: Krutrim LLM은 고급 Attention 메커니즘, 즉 Grouped Query Attention (GQA)와 AliBi를 통합하여 긴 컨텍스트를 처리하고 빠른 응답을 제공합니다. 모델은 인도에 특화된 파인튜닝 과정을 거쳐 지역의 특정한 요구와 뉘앙스를 효과적으로 반영할 수 있도록 발전했습니다. 또한, 전용 Indic tokenizer를 개발하여 인도 언어의 복잡한 형태와 구문 처리를 최적화하였습니다.

- **Performance Highlights**: Krutrim LLM은 16개의 과제 중 10개에서 LLAMA-2와 같은 기존 모델과 동등하거나 더 나은 성능을 보였습니다. 훈련비용은 훨씬 적지만, 평균 성적은 0.57로 LLAMA-2의 0.55를 초과합니다. 이는 다양한 언어적 맥락에서 유연한 다국어 구사 능력을 보여줍니다.



### Online Social Support Detection in Spanish Social Media Texts (https://arxiv.org/abs/2502.09640)
- **What's New**: 이번 연구는 스페인어 소셜 미디어 텍스트에서 온라인 사회적 지원을 탐지하는 혁신적인 접근 방식을 제안합니다. 연구는 지원적(comment) 또는 비지원적(non-supportive)으로 분류된 3,189개의 YouTube 댓글로 구성된 첫 번째 주석 데이터셋을 소개합니다. 또한 데이터 불균형(data imbalance)을 해결하기 위해 GPT-4o를 사용하여 문장의 다양한 표현(paraphrases)을 생성했습니다.

- **Technical Details**: 기존의 머신러닝 모델(traditional machine learning models), 딥러닝 아키텍처(deep learning architectures), 그리고 transformer 기반 모델을 포함한 여러 모델을 평가하여 사회적 지원 분류(social support classification)을 수행했습니다. 특히 GPT-4o는 불균형된 데이터셋에서만 최상의 성능을 보였으며, 불균형 및 균형된 데이터셋을 비교하기 위해 transformer 모델을 사용했습니다. 이 과정에서 각 작업(Task) 별로 성능을 분석했습니다.

- **Performance Highlights**: 균형 데이터셋은 Task 2(개인 및 그룹)와 Task 3(국가, 기타, LGBTQ, 흑인 커뮤니티, 여성, 종교)에서 개선된 결과를 보여주었습니다. 반면, GPT-4o는 Task 1(사회적 지원 및 비지원)에 최적의 성능을 기록했습니다. 이 연구는 지지적 온라인 환경의 중요성을 강조하고, 자동화된 사회적 지원 탐지를 위한 향후 연구의 기초를 마련합니다.



### Jailbreaking to Jailbreak (https://arxiv.org/abs/2502.09638)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 안전성을 높이는 방법으로 '자기 조인트 리벨리온(jailbreak)' 개념을 도입합니다. 이를 통해 인간이 LLM을 스스로 해제하도록 유도함으로써, 다른 모델에 대한 평가를 할 수 있는 J2 공격자를 생성합니다. 특히, LLM은 이전 실패로부터 학습하여 성능을 개선할 수 있습니다. 이를 통해 대형 모델이 자율적으로 공격할 수 있는 새로운 경향을 보여줍니다.

- **Technical Details**: 이 연구는 LLM 안전성을 높이기 위한 다양한 기법과 자동화된 공격이 조합된 'J2 공격자' 모델을 제안합니다. J2는 다수의 레드 팀 전략을 실험하며, 이전의 실패를 통해 공격 성능을 개선하는 인컨텍스트 러닝(in-context learning) 기법을 사용합니다. 실험 결과, Sonnet 3.5와 Gemini 1.5 Pro가 각각 93%와 91%의 공격 성공률을 달성하며, 이는 기존의 LLM보다 우수한 성능을 입증합니다.

- **Performance Highlights**: Sonnet 3.5와 Gemini 1.5 Pro는 각각 93.0%와 91.0%의 공격 성공률을 기록하며, GPT-4o와 같은 경쟁 모델에 대해 우수한 성능을 보여줍니다. 기존의 자동화된 공격 방법보다 효율적인 결과를 도출하며, J2 공격자가 다양한 목표 모델에 대해 J2로 변환할 수 있는 가능성을 열어줍니다. 이 연구는 LLM 안전성 연구에 있어 중요한 기여를 하며, LLM이 직면할 새로운 실패 모드를 강조합니다.



### Reading between the Lines: Can LLMs Identify Cross-Cultural Communication Gaps? (https://arxiv.org/abs/2502.09636)
- **What's New**: 이 논문은 다양한 문화적 배경을 가진 사람들에 의해 작성된 책 리뷰의 이해도 격차를 연구합니다. 57개의 Goodreads 리뷰를 분석한 결과, 83%의 리뷰에서 문화적으로 특정한 이해하기 어려운 요소가 발견되었습니다. 또한, 문화 배경에 따라 GPT-4o가 이러한 요소를 식별할 수 있는 능력을 평가하였으나 mixed results (혼합된 결과)가 나타났습니다.

- **Technical Details**: 문화는 사람들의 삶의 방식과 세계관을 형성하는 복잡한 개념으로 정의됩니다. 이 논문에서는 AI와 NLP 기술의 발전으로 Large Language Models (LLMs)가 온라인의 문화 간 커뮤니케이션 장벽을 극복할 수 있는 가능성에 대해 탐구합니다. LLM이 문화 중재자로서 작용하여 문화적으로 특정한 항목(Culture-Specific Items, CSIs)을 식별하고 설명하는 방법을 다룹니다.

- **Performance Highlights**: 사용자 연구를 통해 미국, 멕시코, 인도 참가자들이 Goodreads의 영어 리뷰를 얼마나 이해하기 힘들어하는지 측정했습니다. 이 연구에서는 GPT-4o와 같은 LLM이 문화적 독서 보조 도구로서 CSIs를 식별하고 설명하는 데 있어 공정하게 작동할 수 있는지를 평가했습니다. 연구 결과로 제공된 데이터셋은 문화 간 커뮤니케이션 개선을 위한 AI 기반 도구의 필요성을 강조합니다.



### CORRECT: Context- and Reference-Augmented Reasoning and Prompting for Fact-Checking (https://arxiv.org/abs/2502.09635)
Comments:
          Accepted to NAACL-25

- **What's New**: 이 논문에서는 증거 문장의 진실성을 확인하기 위해 추가적인 맥락(context)과 참조(reference)를 통합한 새로운 접근법인 Context- and Reference-augmented Reasoning and Prompting (CORRECT)를 제안합니다. 기존의 사실 검사(fact-checking) 모델은 일반적으로 증거 문장 내에서의 추론에 주로 집중하였으나, 이 연구는 보조적인 맥락과 참조를 고려하여 보다 정확한 판단을 가능하게 합니다. 이러한 접근법은 기존의 모델들이 간과했던 여러 증거 및 외부 문서들을 통합하여 일관된 증거 임베딩을 생성합니다.

- **Technical Details**: CORRECT 모델은 세 가지 계층으로 구성된 증거 그래프(evidence graph)를 활용합니다: 증거(evidence), 맥락(context), 참조(reference). 내부 및 외부 계층(reasoning) 간의 논리를 통해 세 가지 그래프 계층을 통합하여 통합된 증거 임베딩을 생성합니다. 또한, 증거-conditioned prompt encoder를 설계하여 각 주장(claim)에 대한 고유한 프롬프트 임베딩(prompt embedding)을 생성하고, 이를 통해 사실 검사에 필요한 데이터를 통합합니다.

- **Performance Highlights**: 실험 결과, CORRECT 모델은 기존 사실 검사 모델들보다 더 나은 성능을 보이며, 다양한 유형의 증거 문장을 처리하는 데 효과적입니다. 모델은 특히 맥락 의존적 및 참조 의존적인 증거를 잘 통합할 수 있어, 복잡한 주장을 보다 정확히 검증할 수 있습니다. 이를 통해 정보의 신뢰성 문제를 해결하는 데 기여할 수 있을 것으로 기대됩니다.



### Unknown Word Detection for English as a Second Language (ESL) Learners Using Gaze and Pre-trained Language Models (https://arxiv.org/abs/2502.10378)
- **What's New**: 이번 연구는 EyeLingo라는 새로운 접근 방식을 제안합니다. EyeLingo는 텍스트 내용과 시선 경로(Eye Gaze Trajectory)를 기반으로 미지의 단어를 실시간으로 예측하는 변환기(Transformer) 기반의 기계 학습 방법을 사용하여 높은 정확성을 달성합니다. 사용자 연구에서 이 방법이 97.6%의 정확도와 71.1%의 F1 점수를 달성한 것으로 나타났습니다. 또한, EyeLingo를 통해 독서 보조 프로토타입이 효과적으로 작동함을 입증하였습니다.

- **Technical Details**: EyeLingo는 사용자의 시선을 인식하고 이를 바탕으로 미지의 단어를 탐지하는 방법입니다. 이 시스템은 gaze 정보를 활용하여 관심 영역을 식별한 후, 사전 훈련된 언어 모델인 PLM을 통해 언어적 특성을 통합하여 미지의 단어를 예측합니다. 이를 통해 gaze 기반 방법의 부정확함을 PLM이 제공하는 단어 확률로 보완합니다. EyeLingo의 실시간 언어 학습 지원 기능은 긴급한 어휘 습득 도구로도 활용될 수 있습니다.

- **Performance Highlights**: EyeLingo 방법은 97.6%의 높은 정확도와 함께 71.1%의 F1 점수를 기록하였습니다. 이를 통해 기존 방법에 비해 미지의 단어 탐지에서 성능이 향상되었음을 보여줍니다. 또한, 사용자 연구에서는 기존 방법에 비해 사용 의향과 유용성의 개선이 관찰되었습니다. 이러한 결과들은 EyeLingo가 실제 세계에서 의미 있는 독서 보조 도구가 될 수 있음을 시사합니다.



### DeltaProduct: Increasing the Expressivity of DeltaNet Through Products of Householders (https://arxiv.org/abs/2502.10297)
- **What's New**: 이 연구에서는 DeltaProduct를 도입하여 Linear Recurrent Neural Networks (linear RNNs)에서 표현력(expressivity)과 효율성(efficiency)을 균형 있게 조정할 수 있는 방법을 제안합니다. DeltaProduct는 Generalized Householder 변환의 곱을 통해 상태 전이 행렬(state-transition matrix)을 형성하여 여러 단계의 경량화된 온라인 경량화를 수행합니다. 이 방법은 DeltaNet보다 향상된 언어 모델링 및 상태 추적 능력을 보여줍니다.

- **Technical Details**: DeltaProduct는 상태 전이 행렬을 생성하기 위해 n_h 단계의 경량화된 Gradient Descent를 수행하며, 이로 인해 표현력의 조절 가능한 메커니즘을 제공합니다. 이는 Diagonal과 Dense 행렬 사이의 원활한 보간을 가능하게 하며, 또한 훈련 기간 동안 안정성을 유지하면서 상태 전이 행렬의 범위를 조절할 수 있습니다. 이론적으로, DeltaNet은 2층으로 구성되어 있는 경우 Dihedral group의 단어 문제를 해결할 수 있음을 증명하였습니다.

- **Performance Highlights**: 광범위한 실험을 통해 DeltaProduct가 DeltaNet에 비해 우수한 상태 추적 및 언어 모델링 성능을 보이며, 길이 외삽(length extrapolation) 능력이 상당히 향상되었음을 확인하였습니다. 또한 Chomsky 계층 벤치마크에서 더 나은 결과를 달성하는 등 다양한 도메인에서 DeltaProduct의 우수한 성능을 입증하였습니다.



### Step-Video-T2V Technical Report: The Practice, Challenges, and Future of Video Foundation Mod (https://arxiv.org/abs/2502.10248)
Comments:
          35 pages, 14 figures

- **What's New**: 이번 논문에서는 30B 개의 매개변수를 가진 최첨단 텍스트-비디오 모델인 Step-Video-T2V를 소개합니다. 이 모델은 204 프레임까지의 비디오를 생성할 수 있으며, Video-VAE를 통해 16x16 공간 압축 비율과 8x 시간 압축 비율을 달성합니다. Step-Video-T2V는 영어와 중국어 모두를 처리할 수 있으며, 생성된 비디오의 시각적 품질을 향상시키기 위해 Video-DPO 접근 방식을 사용합니다.

- **Technical Details**: Step-Video-T2V는 성능을 극대화하기 위해 다단계 훈련 파이프라인을 사용합니다. 이 파이프라인에는 텍스트-이미지 프리트레이닝, 텍스트-비디오 프리트레이닝, 감독 미세 조정(SFT), 직접 선호 최적화(DPO)가 포함되어 있습니다. 또한, 모델은 두 개의 이중언어 텍스트 인코더를 사용하여 입력된 텍스트를 인식하며, 최적화된 하이퍼파라미터와 병렬처리를 통해 훈련의 안정성과 효율성을 보장합니다.

- **Performance Highlights**: 새로 생성된 텍스트-비디오 생성 벤치마크인 Step-Video-T2V-Eval에서 평가 시, Step-Video-T2V의 성능이 최신 오픈 소스 및 상업적 엔진과 비교하여 뛰어난 품질을 보여줍니다. 비디오 생성에서의 강력한 동작 동역학과 높은 미적 품질을 포함하여, 다양한 128개의 프롬프트를 지원합니다. 그러나, 복잡한 동작 시퀀스나 물리 법칙 준수를 요구하는 비디오 생성에서는 여전히 한계가 존재합니다.



### Revisiting Generalization Power of a DNN in Terms of Symbolic Interactions (https://arxiv.org/abs/2502.10162)
Comments:
          arXiv admin note: text overlap with arXiv:2407.19198

- **What's New**: 본 논문은 딥 뉴럴 네트워크(DNN)의 일반화 능력을 상호작용(interactions) 관점에서 분석하고자 합니다. DNN의 일반화 능력을 고차원 특징 공간에서 분석한 이전 연구들과 달리, 본 연구에서는 DNN을 직관적으로 이해할 수 있는 방식으로 재조명합니다. 즉, DNN의 내부 동작을 명확하게 설명할 수 있는 이론적 체계를 구축하여 이를 통해 DNN의 일반화 능력을 분석할 수 있는 새로운 접근법을 제안합니다.

- **Technical Details**: 논문에서는 DNN의 추론 패턴을 AND-OR 상호작용을 기반으로 한 논리 모델로 정확하게 표현할 수 있음을 입증했습니다. 이를 통해 DNN의 모든 세부 추론 패턴을 효과적으로 신뢰성 있게 추출하고, 일반화 가능한 상호작용과 비일반화 상호작용을 분리할 수 있는 방법론을 개발했습니다. 본 연구의 이론적 기반은 과거 연구들에서 도출된 다양한 정리와 실험 결과에 뿌리를 두고 있습니다.

- **Performance Highlights**: 실험을 통해 일반화 가능한 상호작용은 감소 형태의 분포를 따르며, 비일반화 상호작용은 방추 형태의 분포를 따른다는 주장을 입증했습니다. 이러한 상호작용의 분포는 DNN의 과적합(overfitting) 단계에서도 관찰되며, DNN의 일반화 능력 이해에 큰 기여를 할 것입니다. 본 연구는 DNN에서 실제로 사용되는 두 가지 유형의 상호작용을 효과적으로 설명하는 방법을 제안하며, 초기 실험 결과는 이론의 유용성을 잘 보여줍니다.



### X-Boundary: Establishing Exact Safety Boundary to Shield LLMs from Multi-Turn Jailbreaks without Compromising Usability (https://arxiv.org/abs/2502.09990)
- **What's New**: 최근 대형 언어 모델(LLMs)의 보안 취약점 중 하나인 multi-turn jailbreaks에 대한 방어 방식이 어려운 주제로 떠오르고 있습니다. 본 논문에서는 기존의 방어 메커니즘을 평가하고, 일부 방법이 LLM의 다중 턴 공격에 대한 강인성을 향상시킬 수 있지만 사용성(usability)을 저하시키는 것에 대해 논의합니다. 새로운 X-Boundary 방법론을 제안하여, 위험한 표현과 안전한 표현 사이의 경계를 명확히 설정하여 방어 성능을 개선하고 over-refusal 문제를 줄였습니다.

- **Technical Details**: X-Boundary는 위험한 표현을 안전한 표현으로부터 물리적으로 밀어내는 최적화 방법입니다. 이를 통해 위험한 표현을 정확하게 제거하면서 안전한 표현은 유지할 수 있습니다. 기존 방법들은 이러한 경계를 명확히 설정하지 못했으며, 그 결과로 over-refusal과 같은 사용성 문제를 야기했습니다. 실험 결과에 따르면, X-Boundary는 multi-turn jailbreaks에 대한 방어 성능이 기존 방법들보다 우수하고, 일반적인 기능을 거의 완전하게 유지하면서 20% 수준의 과도한 거부(over-refusal)를 감소시켰습니다.

- **Performance Highlights**: X-Boundary는 Llama-3-8B-Instruct에서 multi-turn jailbreaks의 공격 성공률(ASR)을 58.5%에서 16.5%로 감소시켰습니다. 게다가, 이 방법은 교육 과정 동안 수렴 속도를 약 22% 향상시킬 수 있다는 이론적 분석 및 실험적 검증을 제공하였습니다. 우리의 연구는 X-Boundary가 강력한 방어성과 최소한의 사용성 저하를 동시에 달성할 수 있다는 점을 강조합니다. 이런 다면적인 접근 방식을 통해 LLM의 보안과 실용성을 동시에 향상시킬 수 있는 가능성을 보여줍니다.



### Data Valuation using Neural Networks for Efficient Instruction Fine-Tuning (https://arxiv.org/abs/2502.09969)
- **What's New**: 이번 논문에서는 Influence function을 효율적으로 추정하기 위한 새로운 방법인 NN-CIFT(Neural Networks for effiCient Instruction Fine-Tuning)를 소개합니다. 기존의 방법들은 대규모 언어 모델의 비싼 계산 비용과 좋은 일반화 성능 결여 문제를 가지고 있었지만, NN-CIFT는 소형 신경망인 InfluenceNetwork를 활용하여 최대 99%의 비용 절감을 달성합니다. 보고된 결과에 따르면, NN-CIFT는 전체 모델보다 0.0027%의 크기로도 유사한 성능을 유지하면서 영향 값을 추정할 수 있습니다.

- **Technical Details**: NN-CIFT는 세 단계의 알고리즘으로 구성되어 있습니다. 첫 번째 단계에서는 기존 Influence function을 사용해 작은 데이터 집합에 대한 영향 값을 추정하고, 이 작은 집합을 기반으로 InfluenceNetwork를 훈련시킵니다. 두 번째 단계에서 훈련된 InfluenceNetwork를 활용해 나머지 데이터 포인트의 영향 값을 추정하고, 마지막 단계에서는 이 추정된 영향 값들을 바탕으로 데이터 선택 알고리즘을 적용하여 IFT 데이터를 추출합니다.

- **Performance Highlights**: NN-CIFT는 기존의 Influence function과 비교하여 성능 저하 없이 77-99%의 시간 절약을 이뤄냈습니다. 연구 결과에 따르면, NN-CIFT를 통해 선택된 데이터의 평균 제곱 오차는 단지 0.067에 불과하며, 이는 원본 Influence function과의 차이를 최소화합니다. 이로 인해 NN-CIFT는 새로운 데이터 포인트에 대해 다시 훈련할 필요 없이도 효과적이라는 것이 입증되었습니다.



### Self-Supervised Learning for Neural Topic Models with Variance-Invariance-Covariance Regularization (https://arxiv.org/abs/2502.09944)
Comments:
          Preprint accepted in Springer Knowledge and Information Systems (KAIS), in press

- **What's New**: 이번 연구에서는 자기 감독(self-supervised) 신경 주제 모델(NTM)을 제안하여 기존의 모델에 비해 향상된 성능을 보여주고자 하였습니다. NTM은 문서 내 단어에 숨겨진 주제를 학습하는 데 있어 더 큰 유연성과 일관성을 제공하며, 기존 주제 모델보다 더 나은 성능을 달성할 수 있습니다. 본 연구는 정규화를 통해 잠재 주제 표현을 명시적으로 조정하는 자기 감독 학습 접근 방식을 적용하여 주제 품질을 향상시킵니다.

- **Technical Details**: 제안된 모델인 VICNTM은 문서의 앵커(경량) 샘플과 긍정 샘플의 잠재 표현을 정규화함으로써 주제 품질을 높입니다. 또한, CLNTM의 문제인 부정 샘플이 앵커 샘플에 유사해지는 상황을 피하기 위해 부정 샘플의 필요성을 없앴습니다. 이를 위해 우리는 이미지 도메인에서 영감을 받은 VARIANCE-INVARIANCE-COVARIANCE REGULARIZATION(VICReg) 기법을 사용하여 텍스트 샘플에 해당 정규화를 적용하였습니다.

- **Performance Highlights**: 세 가지 데이터셋에서의 실험 결과, 제안한 모델은 두 개의 기준선(benchmark) 및 최신 모델에 비해 quantitatively 및 qualitatively 우수한 주제 일관성을 제시하였습니다. 주제의 일관성, perplexity, 주제 다양성 측면에서 뛰어난 성능을 기록하며, 시각적으로 주제 표현의 우수성을 입증하는 예시를 제공하였습니다. 본 연구는 자기 감독 방식으로 NTM의 정규화를 처음으로 적용하여 주제 품질을 크게 향상시켰습니다.



### MIR-Bench: Benchmarking LLM's Long-Context Intelligence via Many-Shot In-Context Inductive Reasoning (https://arxiv.org/abs/2502.09933)
Comments:
          32 pages, 11 figures

- **What's New**: 본 논문에서는 기존의 few-shot In-Context Learning (ICL) 접근 방식을 넘어, 다양한 정보를 한 번에 다루는 many-shot inductive reasoning benchmark인 MIR-Bench를 제안합니다. 이를 통해 LLM(Large Language Models)이 복잡한 문제 해결에 있어 필요한 변형된 유출 없이 새로운 정보를 효율적으로 처리할 수 있도록 합니다. MIR-Bench는 입력-출력 예시를 통해 LLM이 예측하는 방식으로 구성되어 있으며, 이로 인해 보다 복잡한 inductive reasoning 문제를 다룰 수 있는 기회를 제공합니다.

- **Technical Details**: MIR-Bench는 다양한 입력-출력 데이터 형식을 통해 수백~수천 개의 예시를 제공하며, 이는 LLM이 새로운 입력에 대한 출력을 예측하도록 유도합니다. 이 모델은 크게 MIR-Core와 MIR-Extended의 두 가지 문제 세트를 포함하며, 각각 3,000과 6,930개의 문제를 담고 있습니다. 이러한 구조는 LLM의 장기 컨텍스트 인지 능력을 평가할 수 있는 다양한 문제를 제공합니다.

- **Performance Highlights**: MIR-Bench를 통해 여러 최첨단 LLM의 성능을 비교한 결과, 모델 성능이 크게 차이가 나며, 어느 모델도 벤치마크에서 포화되지 않았습니다. 또한, 연구팀은 기존 연구에서 간과된 여러 중요한 문제들에 대한 실증 연구를 수행하여 LLM의 many-shot 및 long-context 인지 능력에 대한 통찰을 얻었습니다. 이 결과는 LLM의 inductive intelligence가 잘못된 입력-출력 쌍에 얼마나 강한지를 평가하는 데 중요한 기여를 했습니다.



### A Taxonomy of Linguistic Expressions That Contribute To Anthropomorphism of Language Technologies (https://arxiv.org/abs/2502.09870)
Comments:
          18 pages, 1 figure, to appear at CHI 2025

- **What's New**: 최근 언어 기술, 특히 대규모 언어 모델(LLMs)의 인간 유사성 때문에 인간과 비인간 객체에 대한 의인화(anthropomorphism)가 더욱 주목받고 있습니다. 이 논문은 언어 기술의 인간 유사성에 대한 명확한 논의와 위험성, 및 효과적인 설계를 위해 텍스트 표현의 분류 체계(taxonomy)를 개발합니다. 저자들은 의인화를 구성하는 다양한 텍스트 표현을 식별하고, 언어 기술 상호작용의 맥락에서 이들 표현의 의미를 분석합니다.

- **Technical Details**: 이 연구에서는 실제 사용자 상호작용 사례를 분석하여 인간 유사성을 가진 텍스트 표현의 19가지 유형을 도출하였습니다. 이러한 표현은 감정적 취약성, 정체성 및 자기 비교를 통해 시스템의 감정적 피해 가능성을 암시합니다. 저자들은 의인화를 인식하는 데 도움을 주기 위해 다섯 가지 관점을 제안하며, 이 관점들은 텍스트 출력에서 나타나는 인간성의 주장도 포함합니다.

- **Performance Highlights**: 저자는 의인화의 긍정적 및 부정적 측면을 이해하는 데 있어 도전과 긴장이 존재함을 강조합니다. 이러한 의인화가 기술에 대한 신뢰를 부여하지만 동시에 사용자와 기술 간의 관계를 왜곡하며, 때로는 실제 인간의 고유성을 저하시킬 수 있는 잠재적 위험도 내포하고 있습니다. 이 연구는 언어 기술의 의인화를 더 효과적으로 논의하고 결정을 내리는 데 도움이 되는 기초를 제공합니다.



### Solvable Dynamics of Self-Supervised Word Embeddings and the Emergence of Analogical Reasoning (https://arxiv.org/abs/2502.09863)
Comments:
          26 pages, 10 figures

- **What's New**: 이 논문은 언어 모델의 학습 과정을 이해하기 위해, 차별적 자기 감독 알고리즘의 일종인 quadratic word embedding models(QWEMs)를 제안합니다. 이러한 모델은 단순하면서도 이론적 분석이 가능하며, 기존 word2vec 알고리즘과 유사한 성능을 보입니다. 특히, 이 논문은 최적화 동역학과 단어 임베딩에 대한 분석적 솔루션을 제공하여 모델 구조 및 학습 과정의 이해를 높이고자 합니다.

- **Technical Details**: 저자들은 QWEM의 학습 동역학이 감독된 행렬 분해(supervised matrix factorization)와 동일하다는 것을 증명하였습니다. 또한, 훈련 데이터를 기반으로 한 closed form 솔루션을 도출하여, 효과적 unigram 분포가 균일할 때 임베딩의 특이값 동역학이 시그모이드 형태로 변한다는 사실을 밝혔습니다. 이러한 이론적 결과는 WikiText에서 훈련된 QWEM의 실용적인 구현과 일치하며, 최상위 특이 벡터가 해석 가능한 개념을 인코딩함을 보여줍니다.

- **Performance Highlights**: QWEM은 단순하지만 다양한 하향식 작업에 대한 성능에서 word2vec과 비슷한 결과를 보여줍니다. 또한, 주목할 만한 점은 모델의 크기와 훈련 시간이 하향식 아날로지 완성 작업에 미치는 영향을 이론적으로 분석하며, 특정 하위 작업에서의 임계 모델 크기를 추정할 수 있다는 것입니다. 이러한 이론적 결과들은 아날로지 추론의 기하학적 구조가 어떻게 발전하는지를 설명하는 기계론적 설명도 제공합니다.



### Automated Hypothesis Validation with Agentic Sequential Falsifications (https://arxiv.org/abs/2502.09858)
- **What's New**: 이 논문에서는 정보 획득과 의사결정 과정에서 가설(hypothesis)의 중요성을 강조하며, 고수준의 추상적인 가설의 자동 검증을 위한 새로운 프레임워크인 Popper를 제안합니다. Popper는 Karl Popper의 반증 원칙(falsification principle)에 기반하여 LLM(대형 언어 모델)을 활용한 자동화된 검증 과정을 수행합니다. 이는 기존의 수동 검증의 비효율성을 해결하려 합니다.

- **Technical Details**: Popper는 가설의 측정 가능한 함의(measurable implications)를 목표로 하는 실험을 설계하고 실행하는 LLM 에이전트(agent)를 사용하여 가설을 검증합니다. 또한, 이 프레임워크는 엄격한 Type-I 오류 통제(strict Type-I error control)를 보장하는 새로운 순차적 테스트(framework for sequential testing)를 도입하여 다양한 관찰로부터 증거를 수집합니다. 기존 데이터에서 또는 새로운 절차를 통해 수집된 증거를 통해 가설의 유효성을 검증합니다.

- **Performance Highlights**: 이 연구에서는 생물학(biology), 경제학(economics), 사회학(sociology) 등 여섯 가지 분야에서 Popper의 성능을 입증하였습니다. Popper는 오류 통제를 강화하고 높은 검정력을 유지하며, 기존의 인간 과학자들에 비해 복잡한 생물학적 가설을 검증하는 데 소요되는 시간을 10배 단축시키며 유사한 성능을 보였습니다. 이는 가설 검증을 위한 확장 가능하고 엄격한 솔루션을 제공합니다.



### Improving Acoustic Side-Channel Attacks on Keyboards Using Transformers and Large Language Models (https://arxiv.org/abs/2502.09782)
- **What's New**: 이 연구에서는 다양한 딥러닝 기법, 특히 Vision Transformers(VTs)와 Large Language Models(LLMs)를 활용하여 Acoustic Side-Channel Attacks(ASCAs)의 효과성과 적용 가능성을 높이고 있음을 보여줍니다. 본 논문에서 제안한 CoAtNet 모델은 이전 연구보다 향상된 성능을 보이며, 스마트폰과 Zoom을 통한 타이핑 인식 정확도가 각각 5.0%와 5.9% 개선되었습니다. LLM을 사용한 오류 수정 및 잡음 완화 방법을 통해 실환경에서의 ASCA 성능을 강화하고, 더욱 경쟁력 있는 경량 모델을 활용하여 더욱 효율적인 공격 방어를 구현하고 있습니다.

- **Technical Details**: ASCAs는 디바이스로부터의 음파를 이용해 민감한 정보를 추론하는 공격으로, VTs와 LLMs의 융합을 통해 이러한 공격 기법을 혁신적으로 개선할 수 있는 가능성을 보여줍니다. 본 연구에서는 다양한 모델을 비교하고, 특히 CoAtNet 모델을 활용하여 키 입력의 음향 표본을 분석하며, 효과적인 분류 성과를 기록합니다. 강화된 노이즈 완화 전략은 소음이 심한 환경에서도 오류를 탐지하고 수정함으로써 안정적인 키스트로크 인식과 데이터 복구를 이끌어냅니다.

- **Performance Highlights**: 본 연구에서 사용된 CoAtNet 모델은 이전의 최고 성능을 가진 모델과 비교하여 성능이 향상되었으며, LLM을 활용한 노이즈 완화 기법 덕분에 실생활에서의 ASCA 성능이 크게 향상되었습니다. 또한, Low-Rank Adaptation(LoRA) 방식으로 경량화된 모델이 기존의 무거운 모델들과 비교할 만한 성능을 발휘함을 확인했습니다. 이러한 성과들은 ASCAs에 대한 대응책을 개선하고, 향후 데이터 보안을 강화하는 데 기여할 것으로 기대됩니다.



### Non-Markovian Discrete Diffusion with Causal Language Models (https://arxiv.org/abs/2502.09767)
Comments:
          Under Review

- **What's New**: 이번 연구에서는 CaDDi라는 새로운 causal discrete diffusion model을 소개합니다. 이 모델은 기존의 autoregressive transformers의 우수한 성능을 활용하면서, 비마르코프적(non-Markovian) 성질을 기반으로 하여 모든 생성 경로에 대한 정보를 통합합니다. CaDDi는 사전 훈련된 대형 언어 모델(LLM)을 수정 없이도 쉽게 적용할 수 있도록 해주며, 더 적절하고 제어 가능한 생성을 가능하게 합니다.

- **Technical Details**: CaDDi는 기존의 전통적인 diffusion 모델을 확장하여 비마르코프적인 설정에서 동작합니다. 각 디노이징 단계에서자는 단일 상태가 아닌 전체 생성 경로의 정보를 활용하여 오류 누적을 방지합니다. 이 모델은 또한 전통적인 causal language models을 일반화하며, 훈련 과정에서 다음 토큰 예측 손실(next-token prediction loss)을 통해 효율적으로 훈련될 수 있습니다.

- **Performance Highlights**: 실험에 따르면, CaDDi는 최신 discrete diffusion 모델들과 비교했을 때 자연어와 생물학적 시퀀스 작업 모두에서 더 뛰어난 성능을 보여주었습니다. 이 모델은 제어 가능한 생성 능력을 제공하며, 구조화된 시퀀스 모델링을 위한 강력한 대안이 됩니다. CaDDi는 또한 자동 회귀 모델들에 비해 더 높은 생성 품질을 달성하여, diffusion 기반 방법들과 대규모 자기 회귀 변환기들 간의 간극을 좁히는 데 기여합니다.



### Making Them a Malicious Database: Exploiting Query Code to Jailbreak Aligned Large Language Models (https://arxiv.org/abs/2502.09723)
Comments:
          15 pages, 11 figures

- **What's New**: 이번 논문에서는 QueryAttack이라는 새로운 공격 프레임워크를 제안하여 대형 언어 모델(LLM)의 안전성 정렬(safety alignment) 일반화 가능성을 체계적으로 조사합니다. 이 방법은 LLM을 지식 데이터베이스로 간주하고, 악의적인 쿼리를 자연어에서 코드 스타일의 구조화된 쿼리로 변환하여 안전성 정렬 메커니즘을 우회합니다. 실험 결과, QueryAttack은 다양한 개발자와 기능을 가진 LLM에서 높은 공격 성공률(ASR)을 달성함을 보여줍니다.

- **Technical Details**: QueryAttack의 구조는 세 가지 주요 구성 요소로 이루어집니다: (1) 비자연어 기반의 쿼리 형식을 정의하고 자연어 쿼리를 대상 비자연어 형식으로 변환하는 번역기 사용, (2) 비자연어를 이해하고 자연어로 응답을 제공할 수 있도록 대상 LLM을 가이드하는 프롬프트 구성, (3) 변환된 쿼리를 사용하여 모델이 해로운 응답을 생성하도록 유도합니다. 이는 데이터베이스에서 데이터를 쿼리하는 것처럼 쿼리 작업을 정의하는 방식으로, 프로그래밍 언어를 활용하여 적절한 쿼리 형식을 구성합니다.

- **Performance Highlights**: QueryAttack을 통해 수행한 다양한 실험에서는 LLM의 보안 방어를 효과적으로 우회함을 입증하였습니다. 본 연구는 기존의 RLHF 기반 방어 메커니즘이 비자연어 입력을 사용하는 탈옥 공격에 대한 완벽한 방어를 수행할 수 없다는 가설을 세우고, 이 제한점을 극복하기 위한 새로운 접근 방식을 탐구합니다. 또한, 제안된 방어 방법은 GPT-4-1106에서 최대 64%까지 ASR을 줄이는 것으로 평가되었습니다.



### Evaluating GPT's Capability in Identifying Stages of Cognitive Impairment from Electronic Health Data (https://arxiv.org/abs/2502.09715)
Comments:
          Findings paper presented at Machine Learning for Health (ML4H) symposium 2024, December 15-16, 2024, Vancouver, Canada, 7 pages

- **What's New**: 이번 연구에서는 자동화된 접근법으로 전자 건강 기록(EHR)에서 인지 장애(cognitive impairment)를 파악하는 방법을 평가합니다. 특히, 최신 GPT-4o 모델을 사용하여 전문가 노트를 분석하고, 치매 진단의 초기 단계를 자동으로 결정하는 것을 목표로 했습니다. 자동화된 접근은 수작업 차트 검토의 시간 소모와 오류를 줄이고, 향후 연구 데이터셋을 생성하는 데 기여할 수 있습니다.

- **Technical Details**: 연구는 두 가지 작업에 대해 GPT-4o의 성능을 평가했습니다. 첫 번째 작업에서는 769명의 환자 전문 노트를 통해 Clinical Dementia Rating(CDR) 점수를 자동으로 생성했으며, 두 번째 작업에서는 860명의 Medicare 환자의 자료를 분석하여 정상 인지, 경도 인지 장애(MCI), 치매 상태를 분류했습니다. 이 과정을 위해로 Retrieval-Augmented Generation(RAG) 접근법과 명확한 주제 템플릿을 활용하여 모델의 결과를 표준화했습니다.

- **Performance Highlights**: 연구 결과는 GPT-4o가 CDR 점수 지정 작업에서 0.83의 kappa 점수를 기록하였고, MCI 및 치매 상태 구분에서 0.91의 kappa 점수를 달성했습니다. 특히, 임상 진단 과정에서 높은 신뢰도로 평가된 경우에는 0.96의 kappa 점수로, GPT-4o의 높은 성과를 입증했습니다. 이러한 결과는 GPT-4o가 대규모 차트 검토 도구로서 임상 설정에서 효과적일 수 있음을 시사합니다.



### Meta-Cultural Competence: Climbing the Right Hill of Cultural Awareness (https://arxiv.org/abs/2502.09637)
- **What's New**: 최근 연구들은 대형 언어 모델(LLMs)이 서구 중심적 세계관에 편향되어 있음을 보여주었고, 이는 비서구 문화 환경에서의 유용성을 감소시킵니다. 이 논문에서는 LLM이 '문화적 인식(cultural awareness)'을 소유한다는 것이 무엇을 의미하는지를 조사하고, 문화적 인식이 아닌 메타문화적 능력(meta-cultural competence)이 필요하다는 주장을 합니다. 이는 다양한 문화에서 유용하게 활용될 수 있는 LLM과 AI 시스템의 필수 조건입니다.

- **Technical Details**: 논문에서는 Octopus test를 활용한 사고 실험을 통해, 서로 다른 문화적 배경을 가진 대화 참여자들 간의 통신 패턴 학습이 얼마나 복잡한지를 설명합니다. 새로운 Multi-Pair Octopus Test를 도입하여 해양 아래의 통신 케이블을 통해 서로 다른 방식을 사용하는 두 그룹의 대화에서 LLM이 어떻게 반응할 수 있는지를 살펴봅니다. 이는 AI 시스템이 내부 및 외부 문화 간 의사소통을 효과적으로 처리하기 위해 이해해야 할 복잡한 맥락을 강조합니다.

- **Performance Highlights**: 문화는 복잡하고 다면적인 개념으로서 각종 특정한 특성을 지니고 있습니다. 특히, 문화는 동적이며 시간이 지남에 따라 변화합니다. 따라서 AI 시스템이 문화를 적절히 표현하고 처리하기 위해서는 이러한 요인들을 모두 반영할 수 있어야 합니다. 또한, 언어는 문화의 필수적인 요소로서 다양한 혼합 양태를 갖고 있으며, AI 시스템은 이러한 문화적 차이를 유연하게 다룰 수 있는 전략이 필요합니다.



New uploads on arXiv(cs.IR)

### A Hybrid Cross-Stage Coordination Pre-ranking Model for Online Recommendation Systems (https://arxiv.org/abs/2502.10284)
Comments:
          Accepted by WWW 2025

- **What's New**: 이 논문에서는 대규모 추천 시스템에서의 표본 선택 편향(sample selection bias, SSB) 문제와 Matthew 효과를 개선하기 위한 새로운 하이브리드 교차 단계 조정 프리랭킹 모델(Hybrid Cross-Stage Coordination Pre-ranking model, HCCP)을 제안합니다. 기존의 경량 모델은 다운스트림 단계만을 고려하여 일관성(cross-stage consistency) 문제를 초래하고 시스템 성능을 저하시켰습니다. HCCP는 업스트림(검색) 및 다운스트림(랭킹, 리랭킹) 정보의 통합을 통해 프리랭킹의 적응성을 향상시키는 것을 목표로 합니다.

- **Technical Details**: HCCP는 두 가지 주요 구성 요소로 구성됩니다: 하이브리드 샘플 구축(Hybrid Sample Construction)과 하이브리드 목표 최적화(Hybrid Objective Optimization)입니다. 하이브리드 샘플 구축은 검색 및 프리랭킹 순서에서 다층 미노출 데이터를 포착하고, 이를 통해 프리랭킹 학습을 위한 최적의 "ground truth"를 생성합니다. 하이브리드 목표 최적화는 일관성(consistency) 및 롱테일(long-tail) 정밀도를 동시 최적화하는 새로운 Margin InfoNCE 손실(margin InfoNCE loss)을 사용합니다.

- **Performance Highlights**: 다양한 오프라인 및 온라인 실험을 통해 HCCP는 기존의 최첨단(SOTA) 방법들을 초월하는 성능을 입증했습니다. JD 전자상거래 추천 시스템에서 UCVR(사용자 클릭률 증가율)를 14.9% 및 UCTR(사용자 클릭률)를 1.3% 향상시켰습니다. HCCP는 코드 프라이버시를 고려하였으며, 참조를 위한 의사 코드(pseudocode)를 제공합니다.



### SessionRec: Next Session Prediction Paradigm For Generative Sequential Recommendation (https://arxiv.org/abs/2502.10157)
- **What's New**: SessionRec은 기존의 다음 아이템 예측 패러다임(NIPP)과 현실 세계 추천 시나리오 사이의 근본적인 불일치를 해결하는 새로운 세션 기반 예측 패러다임(NSPP)을 도입합니다. 이 모델은 사용자 세션 기반 상호작용을 더 잘 반영할 수 있도록 세션 인식 표현 학습을 통해 다중 아이템 추천이 가능하도록 설계되었습니다. 또한 세션 내 아이템을 위한 순위 손실(rank loss)을 포함하여 추천 시스템의 효율성을 크게 향상시킬 수 있다는 점을 발견했습니다.

- **Technical Details**: NSPP는 사용자 표현을 세션 수준의 임베딩(session-level embeddings)으로 채택하여 사용자 세션 내에서 양성(interactions)과 음성 상호작용을 모두 포착하는데 중점을 둡니다. 이는 시각 변환기(visual transformers)의 원리를 차용하여 사용자 행동의 넓은 맥락과 교차 통계적 정보를 캡처합니다. 이 외에도 NSPP는 세션 내 수집된 상호작용 데이터를 활용하여 검색(retrieval) 및 순위 매기기(ranking) 작업을 동시에 수행하는 모델 지향적 구조를 갖추고 있습니다.

- **Performance Highlights**: Meituan 앱에서 실시한 A/B 테스트 결과, 제안된 SessionRec 모델은 기존 최상의 기준선 모델보다 평균 27%의 성능 향상을 기록하며 효과iveness를 입증했습니다. 이러한 성과는 공개 데이터셋에서 넓게 진행된 실험에서도 확인되었으며, NSPP가 데이터 양이 증가할수록 성능이 지속적으로 향상된다는 스케일링 법칙을 보여주었습니다.



### Semantica: Decentralized Search using a LLM-Guided Semantic Tree Overlay (https://arxiv.org/abs/2502.10151)
- **What's New**: 이번 연구는 중앙 집중형(search engines) 검색 엔진의 권력 집중 문제를 해결하기 위해 세멘틱 오버레이 네트워크(Semantic Overlay Networks)를 활용한 분산형 검색 시스템 설계에 중점을 둡니다. 제안된 알고리즘 Semantica는 대형 언어 모델(Large Language Models)에서 생성된 document embeddings를 사용하여 사용자를 직접 연결하여 검색 성능을 향상시킵니다. 이로 인해 Semantica는 기존 방법들보다 빠르게 의미적으로 유사한 사용자를 식별하고 연결할 수 있습니다.

- **Technical Details**: Semantica 알고리즘은 사전 훈련된 대형 언어 모델을 이용하여 문서의 의미적 내용을 포착합니다. 이 알고리즘은 prefix tree(Trie)를 구축하여, 각 사용자가 자신의 문서 집합에 기반하여 서로 연결되도록 합니다. 이를 통해 사용자는 의미적으로 유사한 다른 사용자와 직접 연결되며, 이들이 해당 사용자의 검색 요청에 더 신속하게 응답할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, Semantica는 기존의 최첨단 방법들과 비교하여 의미적으로 유사한 사용자를 최대 열 배 더 많이 찾아낼 수 있었습니다. 또한, 동일한 네트워크 로드를 기준으로 관련 문서의 검색 성능이 두 배 이상 향상될 수 있음을 보여 주었습니다. 연구자들은 Semantica의 소스 코드를 공개적으로 제공하여 후속 연구를 촉진하고 있습니다.



### A Survey on LLM-powered Agents for Recommender Systems (https://arxiv.org/abs/2502.10050)
- **What's New**: 최근의 대규모 언어 모델(LLM) 기반 에이전트는 기존 추천 시스템의 복잡한 사용자 선호도 이해와 해석 가능한 추천 제공을 개선할 수 있는 잠재력을 갖추고 있습니다. 본 논문은 LLM 기반 에이전트의 여러 가지 응용 사례를 탐색하면서 전통적인 방법의 한계를 극복하는 방식으로 이들 에이전트의 주요 패러다임을 제시합니다. 특히 추천 지향, 상호작용 지향, 시뮬레이션 지향의 세 가지 연구 패러다임을 정리하여 LLM의 효과적인 적용을 설명합니다.

- **Technical Details**: 추천 시스템은 일반적으로 사용자 공간, 아이템 공간 및 상호작용 매트릭스를 기반으로 하며, 이는 사용자 선호도를 예측하는데 중요한 역할을 합니다. 기존의 매트릭스 분해(matrix factorization) 방법과 심층 학습(deep learning) 기법은 복잡한 사용자 의도를 이해하는 데 어려움을 겪고, 명확한 해명을 제공하지 못하는 '블랙박스' 문제를 안고 있습니다. LLM 기반 에이전트는 프로필 구축(profile construction), 메모리 관리(memory management), 계획적 전략 수립(strategic planning), 행동 실행(action execution)의 네 가지 핵심 모듈로 구성되어 복잡한 작업을 관리 가능한 구성 요소로 분해하는 혁신적인 접근 방식을 제안합니다.

- **Performance Highlights**: 본 조사는 LLM 기반 추천 시스템의 현재 상태를 조명하고, 벤치마크 데이터 세트와 평가 프레임워크를 체계적으로 분석합니다. 또한, LLM의 사전 훈련된 지식(pre-trained knowledge)과 강력한 일반화 능력은 다양한 도메인 간의 지식 이전(knowledge transfer)을 촉진하여 차가운 시작(cold-start) 문제를 최소한의 추가 훈련으로 해결할 수 있도록 합니다. 마지막으로, 이 연구는 LLM 기반 추천 시스템의 핵심 과제와 미래 연구 방향에 대한 통찰을 제공합니다.



### ArchRAG: Attributed Community-based Hierarchical Retrieval-Augmented Generation (https://arxiv.org/abs/2502.09891)
- **What's New**: 이 논문에서는 Attributed Community-based Hierarchical RAG (ArchRAG)라는 새로운 그래프 기반 RAG 접근 방식을 소개합니다. 이는 질문을 절차화하기 위해 속성 커뮤니티(attributed communities)를 활용하며, 새로운 LLM 기반 계층 클러스터링 방법을 도입하여 질문에 대한 가장 관련성 높은 정보를 그래프에서 검색합니다.

- **Technical Details**: ArchRAG는 외부 코퍼스에 기반하여 지식 그래프를 사용하여 attributed communities를 탐지하며, 링크와 노드 속성을 모두 고려하는 방법론을 사용합니다. 이 방법은 정보를 효과적으로 검색할 수 있는 계층적 인덱스 구조를 구축하여, 다양한 추상화 수준에서 정보를 제공합니다.

- **Performance Highlights**: 실험 결과, ArchRAG는 정확도와 토큰 비용 모두에서 기존 방법보다 우수한 성능을 보입니다. 이 새로운 접근법은 특히 시간과 토큰 소모를 줄이면서, 효율적으로 질문에 대한 답변을 생성하는 데 있어 큰 개선을 이룹니다.



### An Efficient Large Recommendation Model: Towards a Resource-Optimal Scaling Law (https://arxiv.org/abs/2502.09888)
- **What's New**: Climber라는 새로운 추천 시스템 프레임워크가 소개되었습니다. Climber는 ASTRO와 TURBO, 두 가지 상호 보완적인 구성 요소로 구성되어 있습니다. ASTRO는 사용자 행동을 정교하게 모델링하기 위해 다중 규모의 시퀀스 분할을 도입하며, TURBO는 훈련과 추론 효율성을 극대화합니다. 이 프레임워크는 리소스를 최적화하면서도 뛰어난 성능을 제공합니다.

- **Technical Details**: ASTRO(Adaptive Scalable Transformer for RecOmmendation) 아키텍처는 다중 규모 시퀀스 분할을 통해 주의(attention) 복잡성을 O(n^2d)에서 O(n^2d/Nb)로 줄입니다. 이는 시퀀스 길이에 따른 효율적 스케일링을 가능하게 합니다. 또한, 동적 온도 조절(dynamic temperature modulation)이 도입되어 사용자 행동의 변동성을 반영한 적응형 주의 점수를 조정합니다. TURBO(두 단계 통합 순위 및 배치 출력)는 메모리 효율적인 키-값(Key-Value) 캐싱을 활용하여 훈련 및 추론의 속도를 높입니다.

- **Performance Highlights**: Climber는 배포 후 Netease Cloud Music에서 5.15배의 처리량 증가를 달성했습니다. 이 시스템은 12.19%의 온라인 성과 개선을 기록하며 지속적인 성과 향상을 이끌어냈습니다. 실험 결과, Climber는 기존 추천 시스템 모델보다 더 이상적인 스케일링 곡선을 형성하여, 자원 소모를 최소화하면서도 성능을 극대화할 수 있음을 보여주었습니다.



### Data and Decision Traceability for the Welder's Arc (https://arxiv.org/abs/2502.09827)
- **What's New**: 이번 논문에서는 MITRE와 NIST의 공급망 추적 가능성(Traceability) 원칙을 적용하여 복잡한 다자간 시스템 내 데이터와 결정을 추적하는 Space Protocol의 새로운 접근 방식을 소개합니다. 역시 이 접근 방식의 중앙 목표는 Welder's Arc 시스템 내에서 투명성과 책임성을 보장하는 것입니다. 이 시스템은 입력 데이터로부터 최종 결정까지의 명확하고 감사 가능한 경로를 제공합니다.

- **Technical Details**: Welder's Arc(WA) 시스템은 센서 입력 및 역사적 데이터를 기반으로 적대적인 우주 물체에 대한 평가를 수행하며, 이러한 과정을 통해 반응 추천을 생성합니다. 시스템 내의 모든 상호작용은 메시지 버스를 통해 이루어지며, 각 메시지는 데이터 생성에 관련된 키 정보를 정의하며, 이는 결정 추적 가능성에 기여합니다. 메타데이터는 그래프 데이터베이스에 저장되어 복잡한 관계를 효율적으로 관리하고, 시스템 구성 요소 간의 정보를 재구성할 수 있게 합니다.

- **Performance Highlights**: 결정 추적 가능성 프레임워크는 시스템 엔지니어가 입력 데이터 및 모델을 검토하여 시스템의 의사결정이 올바른지를 감사하고, 문제가 발생한 지점을 신속하게 추적할 수 있도록 돕습니다. 이와 같은 추적 가능성은 지속적인 개선을 가능하게 하여 향후 더 신뢰할 수 있는 결과를 도출하도록 합니다. 또한, 사용자 인터페이스(UI)를 통해 사용자는 특정 결정이나 사건을 선택하여 그에 해당하는 추적 가능성 체인을 시각화할 수 있게 됩니다.



### A Survey on LLM-based News Recommender Systems (https://arxiv.org/abs/2502.09797)
- **What's New**: 이 논문은 LLM(대규모 언어 모델) 기반 뉴스 추천 시스템에 대한 체계적인 조사 연구를 처음으로 수행하였습니다. 연구는 DLLM(구별 가능한 대규모 언어 모델)과 GLLM(생성 가능한 대규모 언어 모델)의 접근 방식을 구체적으로 분류하여 분석합니다. 논문의 주요 기여로는 다양한 LLM 기반 뉴스 추천 모델을 텍스트 모델링, 사용자 모델링 및 예측 모델링의 세 가지 측면에서 검토하고 비교하는 것입니다.

- **Technical Details**: 해당 연구는 NLP에서의 딥러닝 기술 개발과 LLM의 발전을 기반으로 다양한 뉴스 추천 시스템의 메커니즘을 탐색합니다. CNN, RNN, GNN 등 다양한 딥러닝 프레임워크를 통해 뉴스와 사용자 정보를 효과적으로 모델링하며, DLLM 및 GLLM을 뉴스 인코더로 활용하여 성능 향상을 도모합니다. 연구에서는 특히 각 추천 시스템의 성능을 분류 지표(classification metrics), 랭킹 지표(ranking metrics), 다양성 지표(diversity metrics), 개인화 지표(personalization metrics) 등의 다양한 관점에서 평가합니다.

- **Performance Highlights**: 최근 GLLM 기반 뉴스 추천 시스템들의 빠른 성장으로, 이들은 차가운 시작(cold-start) 문제를 완화하고 보다 정확한 뉴스 특징 탐색 및 사용자 관심 모델링에 뛰어난 성과를 보이고 있습니다. 그러나 GLLM 기반 시스템은 상당한 훈련 시간과 자원을 필요로 하며, 기존의 일반적인 딥러닝 방법들과 비교해 성능 면에서 나은 결과를 달성합니다. 이에 따라 논문은 향후 LLM의 시대에서 뉴스 추천의 방향성을 포괄적으로 탐구합니다.



### ProReco: A Process Discovery Recommender System (https://arxiv.org/abs/2502.10230)
Comments:
          8 pages, 5 figures, 9 references

- **What's New**: 이 논문은 ProReco라는 프로세스 발굴 추천 시스템을 소개합니다. ProReco는 이벤트 로그의 특성과 사용자 선호에 따라 가장 적합한 프로세스 발굴 알고리즘을 추천합니다. 기능적으로 ProReco는 최신 발굴 알고리즘을 포함하고 있으며, XAI(eXplainable AI) 기법을 활용해 추천에 대한 설명도 제공합니다.

- **Technical Details**: 프로세스 발굴은 이벤트 로그를 바탕으로 프로세스 모델을 자동으로 생성하는 방법론입니다. ProReco는 Python을 활용하여 PM4py 패키지의 기능을 최대한 활용하며, 사용자가 제공하는 품질 측정에 대한 가중치를 고려하여 최종 점수를 계산합니다. 또한 ProReco는 총 162개의 특성을 추출하고, 머신러닝 모델을 통해 각 알고리즘의 점수를 예측하는 구조로 되어 있습니다.

- **Performance Highlights**: ProReco는 사용자 맞춤형 추천을 통해 프로세스 마이닝 사용자들에게 효율적으로 알고리즘 선택을 도와줍니다. 추천된 알고리즘은 사용자 선호도에 따라 점수화되어 정렬되고, 각 추천에 대한 설명이 제공되며 시스템의 투명성을 높여줍니다. 이러한 점에서 ProReco는 프로세스 발굴의 접근성을 크게 향상시키는 역할을 합니다.



### KGGen: Extracting Knowledge Graphs from Plain Text with Language Models (https://arxiv.org/abs/2502.09956)
- **What's New**: 최근 지식 그래프(KG) 구축에 대한 관심이 높아지면서 데이터 부족 문제가 중요하게 여겨지고 있습니다. KGGen은 자연어 텍스트로부터 고품질의 지식 그래프를 생성하는 텍스트-투-KG(generator) 패키지로, Python 라이브러리 형태로 제공됩니다. KGGen은 관련된 엔티티를 클러스터링하여 추출된 KGs의 희소성을 줄이는 혁신적인 접근법을 채택했습니다. 또한, 새로운 벤치마크인 MINE을 통해 기존 추출기와 비교해 18% 더 높은 성능을 보였습니다.

- **Technical Details**: KGGen은 LLM(대형 언어 모델)을 활용하여 평문에서 주어-서술어-목적어(triple)를 추출하고, 클러스터링 알고리즘을 통해 고품질의 조밀한 KG를 생성합니다. 이 과정에서 여러 단계를 거치며, 첫 번째 단계에서 비구조화된 텍스트를 입력받아 초기 지식 그래프를 생성하고, 그 다음에 유일한 엔티티와 연결 관계를 집계합니다. KGGen은 각 단계에서 DSPy 프레임워크를 사용해 일관된 JSON 형식의 출력을 보장합니다.

- **Performance Highlights**: KGGen은 벤치마크 테스트에서 기존 텍스트-투-KG 추출기보다 18% 더 뛰어난 성능을 나타냈습니다. 이 성능 향상은 KGGen이 고품질의 밀접하게 연결된 KGs를 자동으로 생성할 수 있는 잠재력을 보여줍니다. KGGen의 도입으로 오는 데이터 풍부한 미래는 차세대 KG 기반 모델 훈련과 RAG 시스템에 긍정적인 영향을 미칠 것으로 기대됩니다.



### Prioritized Ranking Experimental Design Using Recommender Systems in Two-Sided Platforms (https://arxiv.org/abs/2502.09806)
- **What's New**: 이번 연구에서는 온라인 이중 마켓플레이스에서의 개입의 총 평균 치료 효과(TATE)를 추정하기 위한 새로운 실험 설계를 제안합니다. 특히, 사용자에게 표시되는 항목 리스팅에서 치료 상태에 따라 항목의 우선 순위를 조정하는 두 측면의 우선 순위 지정 순위를 사용하여 간섭 편향(interference bias)을 완화합니다. 이를 통해 사용자 경험을 단절하지 않으면서도 효과적인 실험적 접근을 제공합니다.

- **Technical Details**: 제안된 Two-Sided Prioritized Ranking (TSPR) 방법론은 추천 시스템을 활용하여 사용자에게 표시되는 항목의 순서를 전략적으로 재배열합니다. 사용자는 두 그룹으로 무작위로 나뉘고, 항목은 치료 그룹, 비치료 그룹 및 위약 그룹으로 나뉘어 각각의 경험을 반영합니다. 이로 인해 각 사용자 그룹으로부터 선택된 항목의 질을 균형있게 유지하고, 일부 항목의 치료가 모든 사용자에게 일관되게 실현되도록 보장합니다.

- **Performance Highlights**: 모의 실험을 통해 TSPR 설계의 유효성을 평가한 결과, 평균적으로 TATE를 -0.047로 추정하는 반면, 주어진 나이브 추정기는 평균 -0.091로 과대 추정했습니다. 이는 제안된 방법이 실험 설계에서 사용자 경험을 보존하면서도 통계적 파워를 유지할 수 있음을 보여줍니다. 따라서, 본 연구는 온라인 플랫폼에서 개입의 진정한 영향을 적절히 측정하는 데 기여할 수 있을 것으로 기대됩니다.



New uploads on arXiv(cs.CV)

### Text-guided Sparse Voxel Pruning for Efficient 3D Visual Grounding (https://arxiv.org/abs/2502.10392)
- **What's New**: 이 논문은 3D 비주얼 기초(3D Visual Grounding) 작업을 위한 효율적인 다단계 컨볼루션 아키텍처를 제안합니다. 기존의 두 단계 모델 또는 포인트 기반 구조는 실시간 추론 요구 사항을 충족하기 어렵습니다. 이 연구는 문자 기반 정보와 3D 장면 표현의 깊은 상호작용을 효율적으로 처리할 수 있는 접근 방식을 제공합니다.

- **Technical Details**: 제안한 방법은 text-guided pruning (TGP)과 completion-based addition (CBA)라는 두 가지 주요 기술로 구성됩니다. TGP는 3D 장면 표현을 반복적으로 희소화하여 텍스트 기능과의 상호작용을 효율적으로 수행하며, CBA는 과도하게 희소화된 영역을 보완하여 정밀한 기하학적 정보를 유지합니다. 이 과정에서 computational overhead는 최소화됩니다.

- **Performance Highlights**: TSP3D는 기존의 단일 단계 방법들과 비교하여 100\% FPS 증가와 함께 최고 속도의 추론 성능을 달성합니다. 게다가, ScanRefer, NR3D, SR3D 데이터 셋에서 두 단계 방법들과 비교하여 3D 비주얼 기초의 최고 정확도를 기록하며, 특히 ScanRefer에서 Acc@0.5에서 +1.13%를 기록하여 뛰어난 성능을 보여줍니다.



### Region-Adaptive Sampling for Diffusion Transformers (https://arxiv.org/abs/2502.10389)
- **What's New**: 이번 연구에서는 새롭게 RAS(Region-Adaptive Sampling)라는 샘플링 전략을 소개합니다. RAS는 Diffusion Transformers(DiTs)의 유연성을 활용하여 이미지 내의 서로 다른 영역에 맞춤형 샘플링 비율을 동적으로 할당하며, 실시간 성능을 향상시킵니다. 이전 연구들이 샘플링 단계의 수를 줄이거나 중간 결과를 재사용하는 데 초점을 맞춘 것과 달리, RAS는 모델의 집중 영역에 기반해 처리할 영역만 업데이트하며 나머지 영역은 캐시된 노이즈를 사용합니다.

- **Technical Details**: RAS는 각 샘플링 단계에서 모델이 집중하는 의미적으로 중요한 영역을 파악하고, 이들 지역을 업데이트합니다. 모델의 초점이 전 단계의 출력에 따라 결정되며, 이를 통해 샘플링 과정에서 지역적 변동성을 허용합니다. 또한, 빠른 업데이트 지역과 느린 업데이트 지역을 구분하여 처리하며, 이러한 프로세스는 computational overhead를 줄이는 데 기여합니다.

- **Performance Highlights**: RAS의 효율성은 Stable Diffusion 3 및 Lumina-Next-T2I 모델에서 최대 2.36배 및 2.51배의 속도 향상을 달성하면서도 생성 품질 저하가 미미함을 보여줍니다. 사용자 연구에 따르면, RAS는 1.6배의 가속도를 제공하며 인간 평가에서도 유사한 품질을 유지합니다. 이 방법은 실시간 응용 프로그램에서 Diffusion Transformers의 가능성을 크게 확장하는 중요한 진전을 의미합니다.



### Simplifying DINO via Coding Rate Regularization (https://arxiv.org/abs/2502.10385)
Comments:
          17 pages, 5 figures

- **What's New**: 최근 DINO와 DINOv2는 대규모의 비지도(unsupervised) 이미지 데이터를 학습하기 위한 모델 계열로 널리 사용되고 있습니다. 본 연구에서는 복잡하고 불안정한 훈련 프로세스를 간소화하고, representation이 무너지는 것을 방지하기 위해 손실 함수(loss function)에 명시적인 부호화 속도(coding rate) 항을 추가하는 접근을 제안합니다. 이로 인해 DINO와 DINOv2의 간소화된 변형인 SimDINO와 SimDINOv2를 개발하였습니다.

- **Technical Details**: 새롭게 제안된 SimDINO와 SimDINOv2는 기존의 학습 방식에서 과도한 시행착오를 줄이며, 하이퍼파라미터(hyperparameters) 조정에 대한 민감성을 크게 완화합니다. 이 모델들은 다양한 네트워크 아키텍처 및 디자인 선택에 더 강인하며, 성능 저하 없이 representation을 학습할 수 있습니다. 이러한 접근은 훈련 방법의 직관성을 높이며, 복잡한 설계를 단순화하여 실증적으로 더 나은 성능을 보여줍니다.

- **Performance Highlights**: SimDINO와 SimDINOv2는 다운스트림 작업(downstream tasks)에서 더 높은 품질의 representation을 학습하며, DINO 및 DINOv2 모델들보다 성능이 개선된 Pareto 개선을 제공합니다. 학습된 representation은 이미지 분류(image classification)와 세분화(segmentation)와 같은 다양한 응용 프로그램에서 최상급(performance) 결과를 이끌어낼 수 있습니다. 따라서, 이 연구는 심화 학습(deep learning)에서의 단순화된 설계 원칙을 활용한 성능 향상의 가능성을 보여줍니다.



### ReStyle3D: Scene-Level Appearance Transfer with Semantic Correspondences (https://arxiv.org/abs/2502.10377)
Comments:
          Project page: this https URL

- **What's New**: 이번 논문에서는 ReStyle3D라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 단일 스타일 이미지를 사용하여 여러 뷰로 표현된 실제 장면에 대한 장면 수준의 appearance transfer를 제공하며, 명시적인 semantic correspondence와 multi-view consistency를 결합합니다. 기존의 global stylization 방식과 달리, ReStyle3D는 open-vocabulary segmentation을 통해 스타일과 실제 이미지 간에 정확한 밀집 instance-level correspondence를 설정하여 각 객체에 적절한 텍스처를 적용합니다.

- **Technical Details**: ReStyle3D는 두 단계의 파이프라인을 통해 동작합니다. 첫 번째 단계에서는 diffusion model을 활용하여 단일 뷰에 대해 training-free semantic appearance transfer를 수행합니다. 두 번째 단계에서는 monocular depth와 pixel-level correspondence에 의해 지시되는 warp-and-refine 네트워크를 통해 스타일을 추가 뷰로 전파합니다. 이를 통해 3D 감각과 뷰 간 일관성을 보장하며, 별도의 카메라 포즈나 3D 모델링이 필요 없습니다.

- **Performance Highlights**: 실험 결과, ReStyle3D는 구조 보존(structure preservation), 인식적 스타일 유사성(perceptual style similarity), 그리고 multi-view coherence에서 기존 방법들을 일관되게 능가했습니다. 사용자 연구를 통해서도 사진처럼 사실적이며 의미적으로 충실한 결과를 생산하는 능력이 검증되었습니다. 연구진은 또한 내부 디자인, 가상 스테이징, 3D 일관성 스타일링 등 새로운 응용 프로그램 개발을 지원하기 위해 코드를 공개할 예정입니다.



### Ocular Disease Classification Using CNN with Deep Convolutional Generative Adversarial Network (https://arxiv.org/abs/2502.10334)
- **What's New**: 최근 Ocular disease 분류에 대한 연구는 Generative Adversarial Networks (GAN)을 활용하여 데이터셋을 합성하는 새로운 접근법을 제안하고 있습니다. 이 연구는 CNN(Convolutional Neural Network)의 훈련을 위해 부족한 fundus 이미지 데이터셋의 문제를 해결하고자 합니다. 제안된 방법은 기존의 이미지 데이터를 기반으로 새로운 합성 이미지를 생성하여 훈련의 효율성을 높이는 것이며, 기존의 질병 이미지를 사용하여 모델을 테스트합니다.

- **Technical Details**: 이 연구는 Deep Convolutional Generative Adversarial Networks (DCGAN)를 사용하여 이미지 데이터셋을 증강하는 방법을 중점적으로 다룹니다. GAN의 두 구성 요소인 generator와 discriminator는 서로 경쟁하면서 동시에 훈련되어, 생성된 이미지가 실제 이미지와 구별되지 않도록 발전하도록 설계되었습니다. 이를 통해 다양한 질병의 ocular 이미지를 더 많이 생성하고 이를 CNN 기반 모델 훈련에 사용함으로써 일반화 성능을 개선하고자 하였습니다.

- **Performance Highlights**: 제안된 모델은 테스트에서 원본 ocular 이미지에 대한 분류 정확도가 근시 78.6%, 녹내장 88.6%, 백내장 84.6%로 나타났으며, 전체 분류 정확도는 84.6%에 도달했습니다. 이러한 결과는 GAN을 활용한 데이터 증강이 ocular 질병 분류의 성능을 향상시킬 수 있음을 보여줍니다. 본 연구는 AI 기술이 기본 진료 환경에서 안과 질병 진단의 정확성과 효율성을 높이는 데 중요한 역할을 할 수 있음을 강조합니다.



### Object Detection and Tracking (https://arxiv.org/abs/2502.10310)
Comments:
          10 pages, 5 figures

- **What's New**: 이 연구는 매우 정확하고 실시간 성능을 갖춘 객체 탐지(object detection) 시스템을 위한 최신 기술 통합을 목표로 합니다. 기존의 여러 컴퓨터 비전(vision) 알고리즘에 의존하고 있는 객체 탐지 시스템의 문제점을 해결하고, 딥 러닝(deep learning) 방식을 이용하여 완전한 엔드 투 엔드(end-to-end) 접근 방식을 통해 이를 해결합니다.

- **Technical Details**: 연구에서는 공공 데이터셋(dataset)을 사용하여 네트워크를 학습합니다. 특히, 연례 아이템 탐지 챌린지에서 사용되는 가장 어려운 데이터셋을 선택하여 학습의 질을 높였습니다. 이 네트워크는 최신의 딥 러닝 기술을 사용하여 개발되었으며, 성능 최적화를 위해 다양한 기법이 적용되었습니다.

- **Performance Highlights**: 이 시스템은 객체 탐지가 필요한 다양한 응용 분야에서 빠르고 정확한 결과를 제공합니다. 실시간 성능을 통해 실제 환경에서도 효과적으로 활용될 수 있으며, 정확도 또한 상당히 향상되었습니다. 이를 통해 객체 탐지의 효율성과 신뢰성을 높이는 데 기여할 것입니다.



### QMaxViT-Unet+: A Query-Based MaxViT-Unet with Edge Enhancement for Scribble-Supervised Segmentation of Medical Images (https://arxiv.org/abs/2502.10294)
- **What's New**: 이번 연구에서는 QMaxViT-Unet+라는 새로운 프레임워크를 제안하고 있습니다. 이 프레임워크는 U-Net 아키텍처를 기반으로 하며, Multi-Axis Vision Transformer (MaxViT) 블록을 사용하여 인코더와 디코더를 대체합니다. 이러한 접근 방식은 의료 이미지 분할에 있어 덜 정밀한 레이블을 사용하는 약한 지도 학습(Weakly-Supervised Learning)의 이점을 활용하고 있습니다.

- **Technical Details**: QMaxViT-Unet+는 지역적(local) 및 전역적(global) 특징을 효율적으로 학습하기 위해 MaxViT 블록을 통합합니다. 또한 쿼리 기반(Queried) Transformer 디코더를 통해 특징을 정제하고, 스크리블 레이블의 제한된 경계 정보(boundary information)를 보완하기 위한 엣지 강화 모듈(edge enhancement module)을 포함하고 있습니다. 이 시스템은 대표적인 심장 구조, 대장용종, 유방암에 관한 네 가지 공개 데이터셋에서 평가되었습니다.

- **Performance Highlights**: QMaxViT-Unet+는 ACDC 데이터셋에서 89.1%의 Dice similarity coefficient (DSC)와 1.316mm의 Hausdorff distance (HD95)를 달성했습니다. MS-CMRSeg에서는 88.4% DSC와 2.226mm HD95, SUN-SEG에서는 71.4% DSC와 4.996mm HD95를 보였으며, BUSI에서는 69.4% DSC와 50.122mm HD95를 기록했습니다. 이러한 결과는 QMaxViT-Unet+가 기존 접근 방식보다 정확성, 강건성, 효율성 측면에서 우수함을 보여줍니다.



### Artificial Intelligence to Assess Dental Findings from Panoramic Radiographs -- A Multinational Study (https://arxiv.org/abs/2502.10277)
- **What's New**: 이 연구는 치과 파노라마 방사선 사진(DPRs)에 대한 AI 자동 평가 시스템을 개발하고, 이의 성능을 다국적 데이터 세트에서 인간 독자와 비교하여 견고한 기준을 설정하는 것을 목표로 했습니다. 다양한 데이터 세트를 아우르는 6,669개의 DPR을 분석하여 주행치료와 진단 효율성을 높일 방법을 모색했습니다. 특히 AI 시스템이 8종의 치과적 발견을 식별할 수 있는 능력을 평가하며, 의료 환경에서 AI의 통합 가능성을 시사합니다.

- **Technical Details**: 분석에 사용된 AI 시스템은 객체 탐지(object detection)와 의미적 분할(semantic segmentation) 기술을 결합하여 각 치아마다 발견 사항을 식별했습니다. 성과 지표로는 민감도(sensitivity), 특이도(specificity), 곡선 아래 영역(AUC-ROC) 등이 사용되었습니다. 연구에서 챙긴 데이터 세트는 네덜란드, 브라질, 대만의 다양한 임상 클리닉들로부터 수집된 DPR을 포함하며, 각 나라에서의 상이한 이미지 수집 기술로 인한 성능 변주를 고려했습니다.

- **Performance Highlights**: AI 시스템은 인간 독자의 성과와 비교했을 때, 특히 근관 주위 투명도(periapical radiolucencies) 식별에서 67.9% 높은 민감도를 보였습니다. AI는 8가지 발견 사항에 대해 최대 96.2%의 AUC-ROC를 달성하며, 인간 독자보다 79배 빠른 이미지 처리 속도를 기록했습니다. 이러한 성과는 AI가 진료 프로세스에 통합될 수 있는 가능성을 제시하며, 진단의 효율성과 정확성을 높일 수 있음을 시사합니다.



### Probing Perceptual Constancy in Large Vision Language Models (https://arxiv.org/abs/2502.10273)
- **What's New**: 이번 연구에서는 33개의 Vision-Language Models (VLMs)의 지각적 지속성(perceptual constancy) 능력을 평가했습니다. 연구팀은 색상, 크기 및 형태의 지속성을 포함하여 253개의 실험을 실시하였고, 이러한 실험은 다양한 조건에서 물체 속성을 인식하는 모델의 능력을 조사하는 데 초점을 맞추었습니다. 이 연구는 VLM의 인지 능력 평가에서 지각적 지속성이 중요한 기준이 될 수 있음을 제시합니다.

- **Technical Details**: 우리의 실험은 이미지, 비디오(MOV), GIF와 같은 세 가지 유형의 지각적 입력을 사용하여 VLM의 지각적 지속성 능력을 평가했습니다. 지각적 지속성 능력은 색상, 크기 및 형태 지속성의 세 가지 주요 도메인에 초점을 두고 있으며, 각 도메인은 조명, 거리 및 시각적 각도에 따른 안정적인 물체 인식을 반영합니다. 각 모델은 제로샷 제너레이션(zero-shot generation) 작업을 기반으로 공정하게 비교되었습니다.

- **Performance Highlights**: 연구 결과, 인간과 VLM 간의 성능 차이가 상당히 크다는 것을 알 수 있었습니다. 일부 우수한 모델, 예를 들어 GPT-4v는 인간 성능에 가까운 정확도를 기록했지만, 전반적으로 VLM은 지각적 지속성을 복제하는 데 여전히 많은 제한이 존재함을 나타냈습니다. VLM은 형태 지속성(task)에서는 가장 좋은 성능을 보였고, 색상 지속성(where)에서는 가장 낮은 성능을 나타냈습니다.



### MITO: Enabling Non-Line-of-Sight Perception using Millimeter-waves through Real-World Datasets and Simulation Tools (https://arxiv.org/abs/2502.10259)
- **What's New**: MITO는 일상 물체의 다중 스펙트럼 mmWave 이미지를 포함하는 최초의 데이터셋입니다. 이 데이터셋은 mmWave 레이더와 RGB-D 카메라를 사용하여 다양한 주파수에서 이미지 캡처를 진행했습니다. MITO는 관측선(line-of-sight)과 비관측선(non-line-of-sight)에서 실제 mmWave 이미지를 포함하며, 이를 통해 컴퓨터 비전 연구자들에게 새로운 가능성을 열어줍니다.

- **Technical Details**: MITO 데이터셋은 커스텀 로봇 시스템과 신호 처리 파이프라인을 통해 76개 물체의 580개 이상의 실제 mmWave 이미지를 캡처했습니다. 또한 비선형 이미징을 위한 시뮬레이션 툴이 개발되어, 인공적으로 생성된 mmWave 이미지를 제공하며, 의료 영상에서도 사용됩니다. 두 가지 모델링 방법을 통해 다양한 물질 특성을 시뮬레이션할 수 있습니다.

- **Performance Highlights**: 데이터셋을 이용한 성능 테스트에서, 객체 분할(object segmentation) 작업에서 92.6%의 정밀도(precision)와 64%의 재현율(recall)을 달성했습니다. 또한, 비선형 인식(NLOS recognition) 작업을 위한 분류기(classifier)를 학습시키고, 현실 세계 이미지에서 85%의 정확도로 객체를 분류할 수 있는 성능을 입증했습니다. MITO는 컴퓨터 비전 발전에 중요한 자원이 될 것입니다.



### PromptArtisan: Multi-instruction Image Editing in Single Pass with Complete Attention Contro (https://arxiv.org/abs/2502.10258)
Comments:
          Accepted in ICASSP 2025

- **What's New**: PromptArtisan은 사용자에게 하나의 과정에서 여러 개의 편집 지침을 제공할 수 있는 혁신적인 이미지 편집 방법입니다. 이 방법은 이미지 내의 특정 마스크와 연관된 편집 지침을 활용하여 복잡한 이미지 변형을 가능하게 합니다. 또한, PromptArtisan은 사전 훈련된 InstructPix2Pix 모델을 사용하며, 새로운 Complete Attention Control Mechanism (CACM)을 통해 사용자 지침에 정확히 따라 갈 수 있도록 합니다.

- **Technical Details**: PromptArtisan은 여러 마스크-프롬프트 쌍을 사용하여 동시에 여러 속성을 정밀하게 편집할 수 있도록 합니다. CACM은 확장 주의(cross-attention)와 자기 주의(self-attention)를 조절하여 이미지 편집 과정의 세밀한 제어를 가능하게 합니다. 이 시스템은 추가적인 훈련 없이도 사용할 수 있으며 기존의 반복 처리 방식보다 효율적인 처리 복잡성을 자랑합니다.

- **Performance Highlights**: PromptArtisan은 GLIDE, Blended Latent Diffusion, DiffEdit 및 IP2P와 같은 최신의 텍스트 및 마스크 기반 이미지 편집 방법들과 비교하여 우수한 성능을 보여줍니다. 이 방법은 하나의 패스에서 다양한 편집 작업을 수행할 수 있어, 기존 방법들의 비효율성을 극복하고 많은 속성을 동시에 수정할 수 있는 가능성을 제공합니다. 이를 통해 전문 사용자와 초보자 모두에게 창의적이고 효율적인 이미지 편집 워크플로우를 지원합니다.



### Step-Video-T2V Technical Report: The Practice, Challenges, and Future of Video Foundation Mod (https://arxiv.org/abs/2502.10248)
Comments:
          35 pages, 14 figures

- **What's New**: 이번 논문에서는 30B 개의 매개변수를 가진 최첨단 텍스트-비디오 모델인 Step-Video-T2V를 소개합니다. 이 모델은 204 프레임까지의 비디오를 생성할 수 있으며, Video-VAE를 통해 16x16 공간 압축 비율과 8x 시간 압축 비율을 달성합니다. Step-Video-T2V는 영어와 중국어 모두를 처리할 수 있으며, 생성된 비디오의 시각적 품질을 향상시키기 위해 Video-DPO 접근 방식을 사용합니다.

- **Technical Details**: Step-Video-T2V는 성능을 극대화하기 위해 다단계 훈련 파이프라인을 사용합니다. 이 파이프라인에는 텍스트-이미지 프리트레이닝, 텍스트-비디오 프리트레이닝, 감독 미세 조정(SFT), 직접 선호 최적화(DPO)가 포함되어 있습니다. 또한, 모델은 두 개의 이중언어 텍스트 인코더를 사용하여 입력된 텍스트를 인식하며, 최적화된 하이퍼파라미터와 병렬처리를 통해 훈련의 안정성과 효율성을 보장합니다.

- **Performance Highlights**: 새로 생성된 텍스트-비디오 생성 벤치마크인 Step-Video-T2V-Eval에서 평가 시, Step-Video-T2V의 성능이 최신 오픈 소스 및 상업적 엔진과 비교하여 뛰어난 품질을 보여줍니다. 비디오 생성에서의 강력한 동작 동역학과 높은 미적 품질을 포함하여, 다양한 128개의 프롬프트를 지원합니다. 그러나, 복잡한 동작 시퀀스나 물리 법칙 준수를 요구하는 비디오 생성에서는 여전히 한계가 존재합니다.



### Mapping bathymetry of inland water bodies on the North Slope of Alaska with Landsat using Random Fores (https://arxiv.org/abs/2502.10214)
Comments:
          24 Pages, 6 Figures, 1 Table. This article is a US Government work. Landsat data from the US Geological Survey Earth Explorer system: this https URL. Sonar training measurements: this https URL. Output maps from the Oak Ridge National Laboratory Distribute Active Archive Center (ORNL-DAAC): this https URL

- **What's New**: 이번 연구는 알래스카 북부 지역의 수조(depth of waterbodies) 깊이를 예측하기 위해 머신러닝(Random Forest Regressor) 모델을 훈련시켰습니다. 기존의 연구에서 모델링된 수조 깊이 예측값을 합성(training data) 훈련 데이터로 활용하여, 데이터 부족 문제를 극복했습니다. 이로 인해 수조 깊이에 대한 보다 다양하고 정밀한 예측이 가능해졌습니다.

- **Technical Details**: 본 연구에서는 수집이 어려운 현장(in situ) 데이터를 대체하기 위해 과거 연구에서 생성된 모델링된 데이터로 Random Forest 모델을 훈련했습니다. 최종적으로 2016년부터 2018년까지의 Landsat 8 이미지 208장에 적용하여 검증에서 전체 $r^{2}$ 값이 0.76을 기록하며, 알래스카 북부 지역의 수조 깊이에 대한 맵을 생성하였습니다. 이 맵은 Oak Ridge National Laboratory Distribute Active Archive Center (ORNL-DAAC)를 통해 제공됩니다.

- **Performance Highlights**: 연구의 결과로 생성된 수조 깊이 맵은 픽셀 당 깊이 추정을 포함하여 알래스카 북부 지역에 대한 첫 번째 지역적 평가를 제공합니다. Random Forest 모델은 현장 데이터로 직접 훈련된 모델보다 더욱 강건하게 설계되었으며, 이를 통해 지역 생태계 서비스에 대한 정보가 한층 강화되었습니다.



### Exploring the Camera Bias of Person Re-identification (https://arxiv.org/abs/2502.10195)
Comments:
          ICLR 2025 (Spotlight)

- **What's New**: 본 연구는 기존의 person re-identification (ReID) 모델에서 발생하는 카메라 바이어스를 실증적으로 조사합니다. 카메라 인식 메소드를 통해 해결하려는 노력이 있었지만, 대부분 훈련 도메인에 국한되어 있었습니다. 우리는 기존 ReID 모델이 보지 못한 도메인에서도 큰 카메라 바이어스를 보인다는 점을 밝혀내었습니다.

- **Technical Details**: 카메라 바이어스는 같은 카메라에서 촬영된 샘플들이 특징 공간(feature space)에서 더 가깝게 모이는 현상으로, 서로 다른 정체성을 가진 샘플이 유사하게 인식될 수 있습니다. 따라서 이 문제를 해결하기 위해 우리는 embedding 벡터의 정규화(normalization) 방법을 다시 검토하고, 이 방법이 어떻게 카메라 레이블에 대한 바이어스를 줄여주는지를 분석합니다. 또한, 다양한 모델과 벤치마크에서 그 일반화를 검증하여 ReID의 테스트 단계에서 간단하면서도 효과적인 후처리(postprocessing) 방법으로 제시합니다.

- **Performance Highlights**: 실험 결과, 기존의 비감독 학습(USL) 알고리즘에서 카메라 바이어스에 의해 교육이 부정적인 영향을 받는 것을 확인했습니다. 우리는 간단한 훈련 전략을 제안하여 이 문제를 해결할 수 있으며, 기존의 알고리즘에 소규모 수정만으로도 성능이 크게 향상될 수 있음을 보여주었습니다. 이러한 연구 결과는 향후 ReID 시스템의 일반화 가능성을 높이고, 카메라 바이어스를 완화하는 데 기여할 것입니다.



### Interpretable Concept-based Deep Learning Framework for Multimodal Human Behavior Modeling (https://arxiv.org/abs/2502.10145)
- **What's New**: 이번 논문에서는 Affective Computing (AC) 분야의 해석 가능성을 개선하기 위한 새로운 모델인 Attention-Guided Concept Model (AGCM)을 제안합니다. AGCM은 예측에 기여하는 개념을 식별하고, 그 개념이 관찰되는 위치를 특히 강조하여, 이해할 수 있는 설명을 제공합니다. 이 모델은 또한 다양한 모달의 개념을 정렬 및 공동 학습할 수 있는 확장 가능성을 포함하고 있어, 여러 신호의 시공간 정보를 통합할 수 있는 장점을 가지고 있습니다.

- **Technical Details**: AGCM은 다층의 attention 기반 아키텍처를 사용하여, 훈련 과정에서 주요 지표를 로컬화하고 학습합니다. AGCM은 혼합된 모달로부터 개념적인 설명을 학습함으로써, 개별 개념이 예측한 감정 레이블에 기여하는 정도를 정량화합니다. 이 프레임워크는 시공간 신호의 특성을 고려하여, 대규모 facial expression recognition (FER) 데이터셋에서 성능을 검증합니다.

- **Performance Highlights**: AGCM은 RAF-DB, AffectNet, Aff-Wild2와 같은 여러 대규모 데이터셋에서의 평가를 통해 기존의 해석 가능한 모델들과 비교하여 우수한 성능을 보였습니다. 또한, 이 모델은 실세계의 복잡한 인간 행동 이해 응용에 대한 일반화 가능성을 검증하며, 예측 과정에서의 해석 가능성을 제공합니다. 실험 결과 AGCM은 피험자에게 도메인 특정 지식에 기반한 인간 해석 가능성을 제공한다는 점에서 우수함을 입증하였습니다.



### Leveraging V2X for Collaborative HD Maps Construction Using Scene Graph Generation (https://arxiv.org/abs/2502.10127)
- **What's New**: 이 논문에서는 HDMapLaneNet이라는 새로운 HD 맵 생성 프레임워크를 제안합니다. 이 기술은 V2X 통신과 Scene Graph Generation을 활용하여 실시간 인프라 변화를 반영하는 지역적 기하학적 레이어를 구성할 수 있습니다. 전통적인 HD 맵 생성 방식의 한계를 극복하기 위해 차량의 전방 카메라 이미지를 사용하여 차선 중심선을 추출하고 이를 그래프 형태로 전환하여 클라우드로 전송합니다.

- **Technical Details**: HDMapLaneNet은 지능형 교통 시스템을 위한 기반으로, Convolutional Neural Networks (CNN)와 transformer를 포함한 심층 학습 모델을 사용하여 차선 데이터를 처리합니다. 각 차선의 중심선은 유향 그래프(directed graph)로 표현되며, Bézier 곡선을 통해 데이터베이스에 저장됩니다. 차선 연결성을 정의하기 위해 인접 행렬을 사용하고, 처리된 데이터를 V2X 통신 인터페이스를 통해 클라우드에 전송하여 HD 맵을 형성합니다.

- **Performance Highlights**: nuScenes 데이터셋에서의 초기 실험 결과는 제안된 방법이 최신 방법에 비해 우수한 연관 예측 성능을 보임을 보여줍니다. HDMapLaneNet은 각 개별 차량이나 로봇이 독립적으로 사용할 수 있으며, 안전하고 정확한 자율 주행을 위해 필요한 정보를 실시간으로 제공합니다. 이를 통해 자율주행 분야에서의 적용 가능성이 높아졌습니다.



### Compress image to patches for Vision Transformer (https://arxiv.org/abs/2502.10120)
Comments:
          15 pages,5 figures

- **What's New**: 이 논문에서는 컴퓨터 비전 분야에서 인기를 끌고 있는 Vision Transformer (ViT)의 한계를 극복하기 위해 CNN과 ViT를 결합한 하이브리드 모델인 CI2P-ViT를 소개합니다. CI2P 모듈은 이미지 압축 기술을 적용하여 입력 이미지의 차원을 줄이며, ViT 모델의 패치 임베딩 구성 요소를 대체할 수 있습니다. 이로 인해 ViT의 계산 비용을 크게 감소시키면서도 모델의 정확도를 향상시키는 데 기여합니다.

- **Technical Details**: CI2P-ViT는 CompressAI 인코더를 활용하여 이미지를 압축한 후, 일련의 컨볼루션을 통해 패치의 시퀀스를 생성합니다. 이를 통해 Patch Embedding 컴포넌트를 대체하고, 전체적인 모델 구조를 유지하면서 inductive bias를 통합하여 로컬 특징을 처리할 수 있는 능력을 강화합니다. 모델의 FLOPs는 63.35% 감소하며, 훈련 속도는 두 배 증가하여 효율성을 높입니다.

- **Performance Highlights**: CI2P-ViT는 Animals-10 데이터셋에서 훈련되어 92.37%의 정확도를 달성하였으며, 이는 ViT-B/16 모델 대비 3.3% 개선된 성과입니다. 또한, 메모리 요구사항의 감소와 함께 리소스가 제한된 연구자들에게 더 나은 최적화 가능성을 제공함으로써 ViT 모델의 추가 발전을 촉진할 수 있습니다.



### Image Embedding Sampling Method for Diverse Captioning (https://arxiv.org/abs/2502.10118)
Comments:
          15 pages, 5 figures, 6 tables

- **What's New**: 이 논문에서는 작은 VLMs(Visual-Language Models)가 이미지 캡션의 다양성과 유용성을 증가시키기 위해 명확하게 서로 다른 이미지 지역에 주의를 기울이는 훈련 없는 프레임워크를 도입합니다. BLIP을 백본으로 사용하는 이 방법은 구조화된 세분화를 활용해 전역 및 지역적 의미를 포착하며 추가적인 모델 훈련 없이도 성능을 향상시킬 수 있습니다. MSCOCO, Flickr30k 및 Nocaps 데이터셋에서 평가한 결과, 기존의 큰 모델과 유사한 수준의 성능을 달성했습니다.

- **Technical Details**: 제안된 HBoP(Hierarchical Bags of Phrases) 프레임워크는 사전 훈련된 세분화 및 캡션 생성 모델을 사용하는 모듈형 아키텍처로 구성됩니다. HBoP는 다단계 캡션(전역, 지역, 세부 수준)을 생성하기 위해 계층 구조를 유도하며, MSD(Multiple Segmentation Decisions)를 통해 이미지의 특정 지역에 대한 패치 임베딩을 선택합니다. 또한, NMS(Non-Maximum Suppression)를 적용하여 중복되는 세분화 마스크를 제거하고, K-HM(K-means Clustering) 기법을 활용하여 지역 수준 세분화를 생성합니다.

- **Performance Highlights**: 이 방법은 아시아 시장을 기준으로 하는 다양한 평가 지표에서 성능을 검증하였으며, 각각의 데이터셋에서 Div-2 점수는 0.735, 0.750, 0.748을 기록했습니다. 이로 인해 HBoP는 이미지-캡션의 관련성과 의미적 일관성을 유지하면서도 다양성을 높이는 효과를 입증하였습니다. 결과적으로, 이는 제한된 자원에서 작동하는 응용 프로그램을 위해 더욱 접근 가능한 이미지 캡션 생성 기술의 발전을 보여줍니다.



### DiSciPLE: Learning Interpretable Programs for Scientific Visual Discovery (https://arxiv.org/abs/2502.10060)
- **What's New**: 이 논문에서는 과학적 데이터를 해석할 수 있는 프로그램을 자동으로 생성하는 새로운 프레임워크인 DiSciPLE(Discovering Scientific Programs using LLMs and Evolution)를 도입합니다. 이 방법은 대규모 언어 모델(LLMs)의 일반 상식과 사전 지식을 활용하여 Python 프로그램을 생성하며, 그 결과로 해석 가능하면서도 정확한 예측을 제공하는 프로그램을 Synthesizes합니다. 또한, 프로그램 비평가(critic)와 프로그램 단순화(simplifier) 기능을 추가하여 프로그램의 품질을 더욱 향상시킵니다.

- **Technical Details**: DiSciPLE는 대량의 이미지 데이터를 입력으로 받아 프로그램을 검색하는 방법론을 제시합니다. 진화적 검색(evolutionary search) 알고리즘을 통해 LLMs에서 초기 프로그램을 얻고, 이를 반복적으로 개선하는 방식으로 작동합니다. 이 접근법은 논리 및 수학적 연산을 결합하여 강력하고 해석 가능한 예측을 가능하게 하는 Neural Networks를 교차 처리(interleave)할 수 있는 프로그램을 발견합니다.

- **Performance Highlights**: DiSciPLE는 세 가지 서로 다른 과학적 컴퓨터 비전 응용 프로그램에서 최신 기술을 적용하여 기존 문헌에 없는 새로운 작업을 해결합니다. 위성 영상을 기반으로 한 인구 밀도 추정에서 기존의 비해 35% 낮은 오류율을 달성했으며, 지역의 생물량 추정에서도 기존 모든 기준선보다 더욱 뛰어난 일반화 성능을 보여주었습니다. 이 연구는 과학적 발견을 위한 표준 및 벤치마크를 제안하여 DiSciPLE의 해석 가능성, 신뢰성 및 데이터 효율성을 입증합니다.



### RealCam-I2V: Real-World Image-to-Video Generation with Interactive Complex Camera Contro (https://arxiv.org/abs/2502.10059)
- **What's New**: 최근 카메라 경로 기반의 이미지에서 비디오로의 생성 기술이 향상되어 복잡한 카메라 제어를 더 잘 지원하게 되었습니다. 하지만 사용자가 실제 이미지 작업 시 깊이나 장면 스케일의 지식이 부족하여 정확한 카메라 파라미터를 제공하기 어렵다는 문제점이 있습니다. 이러한 문제를 해결하기 위해, 우리는 RealCam-I2V라는 새로운 확산 기반 비디오 생성 프레임워크를 제안하며, 이는 단안 메트릭 깊이 추정을 통합하여 3D 장면 재구성을 위한 전처리 단계를 제공합니다.

- **Technical Details**: RealCam-I2V는 사용자 제공 참조 이미지의 메트릭 깊이를 예측하기 위해 Depth Anything v2 모델을 활용하여 3D 장면을 재구성합니다. 훈련 과정에서, 복원된 3D 장면은 각 비디오 샘플의 포인트 클라우드와 정렬되며, 이를 통해 상대 규모 카메라 파라미터를 절대 크기로 조정할 수 있습니다. 이를 통해 다양한 실제 이미지에 대해 비디오 생성 모델을 정확한 카메라 경로로 조건화하여 더 큰 제어력을 달성할 수 있습니다.

- **Performance Highlights**: RealCam-I2V는 RealEstate10K 및 도메인 외 이미지에서 비디오 품질과 제어 가능성에서 상당한 향상을 기록하였습니다. 특히, 장면 제약 노이즈 초기화를 도입함으로써 현재의 최고 성능 벤치마크를 초과 입증되었습니다. 이 결과들은 우리의 접근 방식이 실제 환경에서 효과적임을 보여주며, 기존 방법들의 스케일 불일치 및 사용성 문제를 극복하는 데 기여하고 있습니다.



### Towards Polyp Counting In Full-Procedure Colonoscopy Videos (https://arxiv.org/abs/2502.10054)
Comments:
          Accepted to ISBI 2025

- **What's New**: 이번 연구는 자동화된 대장 내시경 검사 보고의 품질 향상 및 비용 효율성 증대를 위한 새로운 접근 방식을 제안합니다. 연구진은 REAL-Colon 데이터세트를 활용하여 전체 절차 대장 내시경 비디오에서 폴립을 자동으로 세는 방법을 정의하고, 트랙렛 간의 재연결(ReID) 문제를 해결하기 위해 Affinity Propagation 기법을 제안합니다. 이 방법은 폴립 카운팅 작업의 성능을 개선하며, 기존 방법들보다 뛰어난 결과를 달성합니다.

- **Technical Details**: 연구에서는 폴립 카운팅을 위해 SimCLR 기반의 시각적 임베딩을 학습하여 단일 프레임 및 다중 시점에서 폴립 트랙렛의 표현을 개선합니다. 시각적 인코더는 단일 프레임 또는 프레임 시퀀스에서 엣지 박스를 활용하여 트랙렛 표현을 생성하며, 클러스터링 모듈은 이러한 트랙렛을 시각적 유사성을 기반으로 재분류합니다. 폴립 재연결 문제를 비지도 학습 기반의 클러스터링 문제로 재구성하여 트랙렛 간의 유사성을 분석합니다.

- **Performance Highlights**: 이 연구는 REAL-Colon 데이터세트에서 6.30의 폴립 단편화 비율과 5% 미만의 위양성 비율(FPR)을 기록하며, 최신 기법들 중 가장 높은 성능을 달성했습니다. 또한, 연구진은 결과 재현을 위한 코드와 데이터 분할 정보를 공개하여 폴립 카운팅 접근 방식을 비교하고 평가할 수 있는 가능성을 제공합니다.



### ManiTrend: Bridging Future Generation and Action Prediction with 3D Flow for Robotic Manipulation (https://arxiv.org/abs/2502.10028)
Comments:
          15 pages, 9 figures

- **What's New**: 이 논문은 언어 조건에 기반한 로봇 조작의 복잡한 도전 과제를 해결하기 위해 3D flow라는 개념을 소개합니다. 이는 고수준의 언어 추상화를 반영하여 미래 이미지 생성과 정교한 행동 예측 간의 효과적인 다리 역할을 합니다. authors는 ManiTrend라는 통합 프레임워크를 개발하여 3D 입자의 역학, 시각 관찰, 조작 행동을 캐주얼 변환기로 모델링하고, 이를 통해 고효율의 성능을 달성했습니다.

- **Technical Details**: ManiTrend는 로봇 조작을 위한 3D particle의 동적 특성을 모델링하는 새로운 접근 방식을 제시합니다. 3D flow 예측 기능이 미래 이미지 생성 및 행동 예측을 위한 추가 조건으로 작용하여, 픽셀 동역학을 모델링하는 복잡성을 완화하고 매끄러운 행동 안내를 제공합니다. 이 프레임워크는 3D 흐름을 활용하여 언어 지침과 관찰에 따라 입자의 미래 동작을 예측합니다.

- **Performance Highlights**: 실험 결과, ManiTrend는 CALVIN과 LIBERO 두 개의 벤치마크 데이터셋에서 경쟁력 있는 성공률을 기록하며 최신 기술(SoTA)과 비교했을 때 더 높은 효율성을 보여주었습니다. 이 결과는 제안된 방법이 기존 연구에 비해 효과적인 성능 개선을 달성했음을 입증합니다. 또한, 모든 코드는 개발 완료 후에 공개될 예정입니다.



### Navigating Label Ambiguity for Facial Expression Recognition in the Wild (https://arxiv.org/abs/2502.09993)
Comments:
          Accepted by AAAI2025

- **What's New**: 이 논문은 'Navigating Label Ambiguity (NLA)'라는 새로운 프레임워크를 제안하여 얼굴 표정 인식(FER)에서의 레이블 모호성과 클래스 불균형 문제를 동시에 해결하고자 합니다. NLA는 Noise-aware Adaptive Weighting (NAW)와 일관성 정규화(consistency regularization)를 포함하여, 각 훈련 단계에서 샘플의 모호성을 평가하고 동적으로 가중치를 조정합니다. 이 방법은 훈련 후반부에 소수 클래스의 모호한 샘플에 더 집중할 수 있도록 해줍니다.

- **Technical Details**: NLA는 두 가지 주요 요소로 구성됩니다: Noise-aware Adaptive Weighting (NAW)와 일관성 정규화입니다. NAW는 중간 예측 점수 간의 상관관계를 분석하여 모호한 샘플에 높은 가중치를, 잡음이 있는 샘플에는 낮은 가중치를 부여합니다. 또한, Jensen-Shannon Divergence를 사용하여 원본 이미지와 변환된 이미지의 잠재 분포를 정렬함으로써 일관성 정규화를 실시합니다.

- **Performance Highlights**: 다양한 노이즈 및 클래스 불균형 조건에서 실험한 결과, NLA는 기존 방법들에 비해 전체 및 평균 정확도에서 뛰어난 성능을 보여주었습니다. 이는 NLA가 노이즈와 클래스 불균형에 대해 강인함을 증명하며, 얼굴 표정 인식의 새로운 기준을 설정합니다. 또한, NLA는 레이블 모호성을 처리함으로써 두 문제를 동시에 해결할 수 있는 첫 번째 프레임워크로 자리잡고 있습니다.



### V2V-LLM: Vehicle-to-Vehicle Cooperative Autonomous Driving with Multi-Modal Large Language Models (https://arxiv.org/abs/2502.09980)
- **What's New**: 현재 자율주행 차량은 주변 환경을 이해하고 미래 경로를 계획하기 위해 개별 센서에 의존하지만, 이 방법은 센서 오류나 장애물로 인해 신뢰성이 떨어진다고 합니다. 이를 해결하기 위해 V2V 통신을 통해 협력적 인식을 사용하는 방법이 제안되었지만, 이 연구는 계획 성능 향상에 대한 기여가 충분히 탐구되지 않았습니다. 본 논문은 LLM을 통합한 V2V 질의응답 데이터셋 및 벤치마크를 제안하여, 협력적 자율주행을 위한 새로운 문제 설정을 탐구합니다.

- **Technical Details**: 본 논문은 V2V-LLM(차량 간 대형 언어 모델)을 제안합니다. 이 방법은 여러 연결된 자율차량(CAV)으로부터 수집된 인식 정보를 융합하여 운전 관련 질문에 답변하는 기능을 수행합니다. V2V-LLM은 장면 수준의 특성 맵과 객체 수준의 특성 벡터를 융합하고, 비전 및 언어 이해를 통해 질의에 응답합니다. V2V-QA 데이터셋은 그라운딩, 유의미한 객체 식별, 계획 등에 대한 질문-답 변환 쌍을 포함합니다.

- **Performance Highlights**: V2V-LLM은 다른 후보 방법들에 비해 유의미한 객체 식별 및 계획 작업에서 우수한 성능을 보였으며, 그라운딩 작업에서도 두 번째로 좋은 성과를 달성했습니다. 이 연구는 협력 자율주행에 대한 새로운 연구 방향을 제시하며, 안전성을 높일 수 있는 잠재력을 지닌 모델 아키텍처로 자리 잡을 수 있습니다. V2V-QA 데이터셋을 기반으로 한 철저한 실험 결과는 V2V-LLM의 가능성을 뒷받침합니다.



### Conditional Latent Coding with Learnable Synthesized Reference for Deep Image Compression (https://arxiv.org/abs/2502.09971)
- **What's New**: 이 논문에서는 외부 사전(dictionary)으로부터 동적 참조를 합성하여 입력 이미지의 조건부 인코딩을 수행하는 방법을 연구합니다. 새로운 접근법인 Conditional Latent Coding (CLC)을 통해 입력 이미지에 대해 효율적이고 동적인 조건부 잠재(latent) 표현을 생성합니다. 이 방법은 이미지의 소스 상관관계를 탐색하는 데 매우 효과적입니다.

- **Technical Details**: 우리는 수정된 공간 피라미드 풀링(modified spatial pyramid pooling), 차원 축소(dimension reduction), 다중 스케일 특성 클러스터링(multi-scale feature clustering)을 포함하는 다단계 접근법을 통해 보편적인 이미지 특성 사전을 구성합니다. 클로우만 동적 참조를 생성하기 위해 조건부 잠재 매칭(module)과 조건부 잠재 합성(module)을 사용하여 효율적인 이미지 압축이 가능합니다. 이 방법은 외부 사전 샘플의 섭동(perturbations)에 대해 강건한 것으로 입증되었습니다.

- **Performance Highlights**: 실험 결과, 이 새로운 방법은 벤치마크 데이터세트에서 코딩 성능을 최대 1.2 dB 향상시키는 것을 보여주며, 비트/픽셀(bit per pixel)의 약 0.5%의 오버헤드(overhead)로 달성되었습니다. 또한 CLC 방법은 대규모 및 다양한 사전에서도 여전히 안정적인 성능을 유지할 수 있는 것으로 나타났습니다.



### VicKAM: Visual Conceptual Knowledge Guided Action Map for Weakly Supervised Group Activity Recognition (https://arxiv.org/abs/2502.09967)
- **What's New**: 본 논문에서는 Visual Conceptual Knowledge Guided Action Map (VicKAM)이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 개인 행동의 위치를 효과적으로 포착하고 이를 행동 의미(semantic)와 통합하여 약한 감독 하의 그룹 활동 인식을 수행합니다. VicKAM은 이미지 상관정리 이론을 기반으로 각 행동이 다양한 위치에서 발생할 가능성을 나타내는 행동 맵을 생성하며, 그룹 활동과 관련된 통계 정보로 강화됩니다.

- **Technical Details**: VicKAM은 훈련 세트에서 행동 프로토타입을 생성하여 시각적 개념 지식을 구축합니다. 이를 통해 행동 맵을 생성하고, 이를 통계적으로 보강하여 특정 그룹 활동에 연결 관계를 설정합니다. 이 과정은 시각적 특징 추출을 위한 RoiAlign과 같은 기술을 활용하여 이루어지며, 훈련 전에 개별 행동에 대한 시각적 개념 지식을 탐색합니다.

- **Performance Highlights**: 논문에서 제안하는 VicKAM은 Volleyball 및 NBA 데이터셋에 대한 실험을 통해 최첨단 방법들과 비교했을 때 유망한 성능을 보였습니다. 특히, 훈련 데이터가 제한적인 상황에서도 효과적으로 작동함을 입증하였습니다. 이 결과는 기존 약한 감독 학습 방법들이 가진 한계를 극복하는 데 기여할 것입니다.



### Generating on Generated: An Approach Towards Self-Evolving Diffusion Models (https://arxiv.org/abs/2502.09963)
- **What's New**: 이 논문은 Recursive Self-Improvement (RSI)의 개념을 텍스트-이미지 diffusion 모델에 적용하여 훈련 붕괴 문제를 해결하고자 합니다. 기존의 모델들이 생성하는 데이터의 불일치와 생성적 환각(Generative hallucinations) 문제를 종합적으로 분석하고, 이를 해결하기 위한 세 가지 전략을 제안합니다. 이 연구는 RSI가 자율적이며 지속적인 데이터 개선에 어떻게 기여할 수 있는지를 탐구합니다.

- **Technical Details**: 논문에서는 텍스트-이미지 diffusion 모델의 훈련 과정을 재구성하고, 생성되는 데이터의 품질과 분포가 모델의 발전에 미치는 영향을 검토합니다. 제안된 방법들은 고품질 프롬프트 생성 파이프라인, 인간 선호 샘플을 식별하기 위한 preference sampling 방법 및 분포 기반 가중치 방식을 포함합니다. 이들 접근법은 모델이 더 높은 지각적 정렬(Perceptual alignment)을 유지하면서 생성적 환각을 최소화하도록 돕습니다.

- **Performance Highlights**: 제안된 방법들의 효과성을 검증하기 위해 광범위한 실험을 수행하였으며, 그 결과 모델이 생성하는 이미지 품질이 향상되었음을 확인했습니다. 이러한 접근을 통해, RSI를 통해 지속적으로 발전 가능한 강력하고 효과적인 텍스트-이미지 모델을 구축할 수 있는 가능성을 제시하였습니다. 이는 향후 AI 모델의 자율적 개선과 데이터 활용 가능성을 확장하는 데 기여할 것입니다.



### Using MRNet to Predict Lunar Rock Categories Detected by Chang'e 5 Prob (https://arxiv.org/abs/2502.09952)
Comments:
          Published at the 8th International Conference on Advances in Machinery, Material Science and Engineering Application (MMSE 2022)

- **What's New**: 중국의 창어 5호(Chang'e 5) 임무는 큰 성공을 거두었습니다. 이 임무는 해왕성의 바다인 Oceanus Procellarum에서 달 표면의 이미지를 수집하기 위해 디자인되었습니다. 이번 연구는 제한된 양의 달 암석 샘플을 활용하기보다, 달 탐사 로봇을 통한 암석 분석에 집중하고 있습니다.

- **Technical Details**: 이 연구는 CE5ROCK이라는 lunar surface rock image 데이터 세트를 구축했습니다. 이 데이터 세트는 100개의 이미지를 포함하고 있으며, 훈련, 검증 및 테스트 세트로 무작위로 나뉘어 있습니다. 또한, MRNet이라는 새로운 네트워크 아키텍처를 제안하며, 이는 VGG16을 기반으로 한 feature extraction 기능과 dilated convolution을 결합하여 더 세밀한 달 암석 식별에 유리합니다.

- **Performance Highlights**: 실험 결과 MRNet은 기존의 CNN 모델인 AlexNet 및 MobileNet보다 약 40.0%의 식별 정확도를 보였습니다. CE5ROCK 데이터 세트를 사용한 추가 실험에서 MRNet은 더 정밀한 암석 유형 식별을 달성했으며, 기존의 주요 알고리즘을 능가하는 성능을 입증했습니다.



### A Lightweight and Effective Image Tampering Localization Network with Vision Mamba (https://arxiv.org/abs/2502.09941)
- **What's New**: 이번 논문에서는 이미지 변조 탐지를 위한 새로운 네트워크인 ForMa(Forensic network based on vision MAmba)를 제안합니다. 기존의 CNN과 Transformer 기반 방법들이 가진 한계를 극복하여, 선형 복잡도로 글로벌 의존성 모델링을 가능하게 합니다. ForMa는 고급 특성 추출과 경량 디코더를 통합하여 태양적으로 경량화된 구조를 통해 높은 로컬라이제이션 정확도를 달성합니다.

- **Technical Details**: ForMa는 세 가지 주요 혁신을 가지고 있습니다. 첫째, 기존 CNN 및 Transformer 구조를 대체하는 시각 상태 공간(VSS) 인코더로 구성되어 있습니다. 둘째, 경량 디코더는 픽셀 셔플 기반의 업샘플링 기술을 사용하여 계산 비용을 줄이며, 마지막으로, 소음 보조 디코딩 전략을 도입하여 변조 이미지의 보조적 특성을 통합합니다.

- **Performance Highlights**: 실험 결과, ForMa는 10개의 표준 데이터셋에서 최고의 일반화 성능과 강인성을 보여주며, 최소의 계산 복잡도를 유지합니다. 기존의 CNN 및 Transformer 기반 접근법들과 비교해 우수한 로컬라이제이션 정확도를 달성하였습니다. 코드와 데이터셋은 공개되어 사용 가능합니다.



### Temporal Scale and Shift Invariant Automatic Event Recognition using the Mellin Transform (https://arxiv.org/abs/2502.09939)
- **What's New**: 본 논문에서는 기존 2D optical image correlation 기술과 냉각 원자의 비균질하게 넓혀진 배열을 결합한 Spatio-temporal holographic correlator를 제안합니다. 이 새로운 방법론은 3D 시간-공간 상관관계를 이루어내어 초고속 이벤트 인식을 가능하게 합니다. 특히, 다양한 속도로 작동하는 비디오에 대해 이벤트 인식을 수행하는 방법을 소개합니다.

- **Technical Details**: 제안된 방법은 비디오 데이터베이스에서 불필요한 이벤트를 거의 모두 필터링할 수 있도록 설계되었습니다. 이 시스템은 고도화된 알고리즘을 통해 인식 정확도를 크게 향상시킵니다. 또한, 기초적인 2D 기술과의 차별성을 명확히 하여, 향상된 3D 상관관계 모델을 기반으로 합니다.

- **Performance Highlights**: 이 기술을 통해 이벤트 인식의 정확도가 향상되었으며, 전체적인 처리 속도가 개선되었습니다. 특히, 다양한 속도의 비디오에서도 효과적으로 작동함으로써 실시간 이벤트 감지에 적합합니다. 실험 결과는 해당 방법이 기존 기술보다 월등한 성능을 발휘함을 보여줍니다.



### Precise Parameter Localization for Textual Generation in Diffusion Models (https://arxiv.org/abs/2502.09935)
Comments:
          ICLR 2025

- **What's New**: 새로운 확산 모델이 통합된 고품질 텍스트를 통해 사진처럼 리얼한 이미지를 생성할 수 있음을 보여주었습니다. 주목할 만한 점은 1%도 안 되는 매개변수만이 이미지 내 텍스트 콘텐츠 생성에 영향을 미친다는 것입니다. 이 발견을 바탕으로 텍스트 생성의 효율성을 높이기 위해 특정 레이어를 타겟팅하여 성능을 개선했습니다.

- **Technical Details**: 이 연구는 최신 확산 모델의 내재적 작동 방식을 밝히고, 텍스트 콘텐츠 생성을 담당하는 모델의 부분을 지역화하는 방법을 처음으로 제안합니다. 활성화 패칭 기술을 기반으로 하여 파라미터의 극히 일부(예: Stable Diffusion XL의 0.61%)가 텍스트 생성을 담당한다는 것을 확인했습니다. 이를 통해 우리는 텍스트 콘텐츠 생성 성능을 향상할 수 있는 여러 가지 응용 프로그램을 소개합니다.

- **Performance Highlights**: 우리는 새롭게 제안한 국소화 및 미세 조정 방법이 텍스트 생성 능력을 크게 향상시키며, 이는 이미지 내 다른 시각적 속성에 영향을 주지 않고 텍스트를 대체하는 데에도 사용될 수 있음을 입증했습니다. 뿐만 아니라, 생성 과정 중 유해한 텍스트 생성을 방지하는 효율적인 방법을 추가적으로 제시했습니다. 이러한 방법들은 다양한 확산 모델 아키텍처에 널리 적용될 수 있으며, 이전 방법보다 우수한 정확도와 시각적 일관성을 보여줍니다.



### AffectSRNet : Facial Emotion-Aware Super-Resolution Network (https://arxiv.org/abs/2502.09932)
- **What's New**: 이번 연구에서는 저해상도 이미지에서 고해상도 얼굴 이미지를 재구성하고 얼굴 표현의 강도와 충실도를 유지하는 감정 인식 초고해상도 프레임워크인 AffectSRNet을 제안합니다. 기존의 얼굴 초해상도 방법은 표현의 정서를 보존하는 데 실패하는 경우가 많았지만, 본 논문은 표현 보존 손실 함수를 사용하여 이 문제를 해결하고 향상된 FER (Facial Expression Recognition) 응용 프로그램의 실용적 배치를 가능하게 합니다. 또한 새로운 감정 보존 평가 지표를 도입하여 저해상도 상황에서 FER 시스템의 성능을 미세하게 평가할 수 있게 되었습니다.

- **Technical Details**: AffectSRNet은 그래프 임베딩을 통한 감정 인식 얼굴 초해상도 기술을 기반으로 하며, 저해상도 얼굴 이미지를 업스케일링하면서 얼굴 표현의 강도를 유지합니다. 연구진은 CelebA, FFHQ, Helen 데이터셋을 활용해 기존의 초해상도 접근법과 비교하여 정량적, 정성적 비교 분석을 수행하였습니다. 또한, 기존의 초해상도 네트워크에 통합될 수 있는 확장 가능한 손실 함수도 고안하여 다양한 FER 애플리케이션에 적용할 수 있는 가능성을 확보하였습니다.

- **Performance Highlights**: 실험 결과, AffectSRNet은 시각적 품질 및 감정 충실도 면에서 기존의 얼굴 초해상도 접근법을 초월하는 성능을 보여주었습니다. 이 연구는 이미지 명료성을 향상시킬 뿐만 아니라 감정 기반 애플리케이션이 열악한 해상도 환경에서도 핵심 기능을 유지하도록 보장하여 FER 시스템의 더 넓은 채택을 위한 기반을 마련합니다. 다음 단계로, 향상된 FER 시스템을 다양한 실시간 응용 프로그램에 통합할 수 있는 가능성을 제시합니다.



### TransGUNet: Transformer Meets Graph-based Skip Connection for Medical Image Segmentation (https://arxiv.org/abs/2502.09931)
Comments:
          24 pages, 12 figures

- **What's New**: 이번 연구에서는 medical image segmentation을 위해 cross-scale GNN 기반의 skip connection 구조를 활용한 새로운 모델, TransGUNet을 제안합니다. 이 모델은 복잡한 해부학적 구조를 이해하기 위해 주의(node attention)를 적용하며, 전통적인 모델들이 가진 semantic gap 문제를 해결합니다. 또한, Entropy-driven feature selection (EFS)을 통해 더 신뢰할 수 있는 spatial attention map을 생성합니다.

- **Technical Details**: TransGUNet은 attentional cross-scale graph neural network (ACS-GNN)와 EFS 기반 spatial attention을 통합한 구조입니다. 이 모델은 cross-scale feature map을 그래프로 변환하여 각 노드에 대해 주의를 기울여 robust한 feature 통합을 도출합니다. 이와 함께, deep learning 모델들이 생성하는 비정보적인 feature map 문제를 해결하기 위해, 채널별 엔트로피 점수를 계산하여 높은 엔트로피를 가진 feature map을 필터링합니다.

- **Performance Highlights**: TransGUNet은 6개의 학습된 데이터셋 및 8개의 미학습 데이터셋에 대해 우수한 세분화 성능을 발휘하며, 이전의 transformer- 및 convolution-based 접근 방식에 비해 상당히 높은 효율성을 보여줍니다. 종합적인 실험 결과를 통해 이 모델이 의료 이미지 분할에 있어 기존 방법들보다 더 나은 성능과 신뢰성을 제공함을 증명하였습니다.



### Deep Tree Tensor Networks for Image Recognition (https://arxiv.org/abs/2502.09928)
- **What's New**: 이번 논문에서는 Deep Tree Tensor Network (DTTN)이라는 새로운 아키텍처를 소개합니다. DTTN은 고차원 특성 간 곱셈 상호작용을 포착하는 멀티리니어 연산을 활용하여 성능을 향상시킵니다. 이 아키텍처는 여러 개의 대칭 모듈 (AIM)을 쌓아 올려 구현되며, 트리 형태의 텐서 네트워크 구조로 펼쳐집니다.

- **Technical Details**: DTTN은 활성화 함수와 주의 메커니즘 없이 2^L 차수의 곱셈 상호작용을 캡처하며, 다수의 벤치마크에서 기존 아키텍처들과 동등한 성능을 보입니다. 이 모델은 텐서 저차원 구조의 특징을 유지하면서도 높은 해석 가능성 및 직관성을 제공합니다. 또한 DTTN은 상대적으로 더 빠른 수렴 속도를 자랑하며, 고차원 텐서 간의 내적과 같은 효율적인 수학적 기법을 사용하여 연산을 최적화합니다.

- **Performance Highlights**: DTTN은 기존의 폴리노미얼 및 멀티리니어 네트워크와 비교하여 뛰어난 성능을 달성하였습니다. 여러 복잡한 벤치마크 테스트에서도 전반적으로 우수한 결과를 나타내었으며, 고급 아키텍처와의 성능을 일치시킵니다. 이로 인해 향후 DTTN을 활용하여 더 많은 해석 가능성을 가진 연구를 진행할 수 있을 것으로 기대됩니다.



### Granite Vision: a lightweight, open-source multimodal model for enterprise Intelligenc (https://arxiv.org/abs/2502.09927)
- **What's New**: Granite Vision은 경량의 대형 언어 모델로, 특히 시각적 문서 이해에 최적화되어 설계되었습니다. 이 모델은 문서 관련 작업을 포함한 포괄적인 instruction-following dataset에서 훈련되어, 테이블, 차트 및 인포그래픽과 같은 다양한 콘텐츠를 자동으로 추출할 수 있습니다. Granite Vision은 총 30억 개의 파라미터를 갖고 있으며, 테스트 시 안전 분류 접근 방식을 도입하여 잠재적으로 위험한 입력을 식별합니다.

- **Technical Details**: Granite Vision은 비주얼 모달리티의 정렬에 중점을 둔 2억 개의 파라미터를 가진 decoder-only 구조로 이루어져 있습니다. 이 모델은 시각적 인코더와 Granite 대형 언어 모델을 연결하기 위해 projector를 사용하고, 여러 단계의 훈련 프로토콜로 성능을 최적화했습니다. 또한 시각적 문서 이해에 필요한 세밀한 정보 포착을 위해 멀티 레이어 특징을 추출하고, 희소한 주의 벡터를 기반으로 한 안전 분류 모듈을 제안합니다.

- **Performance Highlights**: Granite Vision은 시각적 문서 이해와 관련된 여러 기준 벤치마크에서 최고의 성과를 달성했습니다. 특히, 최근에 발표된 Arxiv 논문을 사용하여 모델의 오염을 피할 수 있는 LiveXiv 벤치마크에서도 두각을 나타냅니다. 연구 및 상업적 사용이 가능한 Apache-2 라이센스 하에 모델을 공개하여 투명성을 높이고 협력을 촉진하고 있습니다.



### TaskGalaxy: Scaling Multi-modal Instruction Fine-tuning with Tens of Thousands Vision Task Types (https://arxiv.org/abs/2502.09925)
- **What's New**: TaskGalaxy는 19,227개의 계층적 작업 유형과 413,648개의 샘플로 구성된 대규모 다중 모달 지침 미세 조정 데이터 세트입니다. 이 데이터 세트는 GPT-4o를 활용하여 기존 수작업 정의 작업의 소량 세트에서 확장하여 작업의 다양성을 높입니다. CLIP과 GPT-4o를 통해 오픈 소스 이미지와 가장 잘 일치하는 작업 유형을 필터링하고, 관련된 질문-답변 쌍을 생성하여 데이터 품질을 확보합니다.

- **Technical Details**: TaskGalaxy 데이터 세트는 OCR, 이미지 설명 및 복잡한 논리적 추론과 같은 다양한 작업 유형을 포함하고 있습니다. 기존의 수동 정의 작업 세트를 시작으로 하여 GPT-4o가 프로세스 내내 작업을 자동으로 확장합니다. CLIP의 유사성 평가를 통해 적절한 이미지와 강력히 관련된 작업 유형이 선정되며, 그에 따라 생성된 질문-답변 쌍은 세 가지 오픈 소스 모델을 통해 스크리닝하여 품질을 확보합니다.

- **Performance Highlights**: LLaVA-v1.5와 InternVL-Chat-v1.0 모델에 TaskGalaxy를 통합한 결과, 16개의 벤치마크에서 성능이 크게 향상되었습니다. 평균적으로 4.5 및 3.83, 3.0 및 3.64 포인트 개선된 결과를 나타냈습니다. 특히 MME 벤치마크에서 LLaVA-v1.5-13B 모델이 68 포인트 증가하여 TaskGalaxy 데이터 세트가 모델의 일반화 능력을 향상시킨다는 것을 증명했습니다.



### Self-Consistent Model-based Adaptation for Visual Reinforcement Learning (https://arxiv.org/abs/2502.09923)
- **What's New**: 이번 연구에서는 Self-Consistent Model-based Adaptation(SCMA)라는 새로운 방법을 제안하여 시각적 방해 요소에 대한 적응을 개선합니다. 기존의 정책 수정 없이도 SCMA는 denoising 모델을 사용해 복잡한 관찰을 깨끗한 관찰로 변환하여 여러 정책에 Plug-and-Play 방식으로 성능을 향상시킬 수 있습니다. 이 방법은 교육받지 않은 데이터의 경우에도 많이 활용될 수 있도록 설계되었습니다.

- **Technical Details**: SCMA는 Noisy Partially-Observed Markov Decision Process(NPOMDP)를 사용하여 복잡한 관찰에서의 RL 문제를 포착하고, unsupervised distribution matching 목표를 통해 denoising 모델을 최적화합니다. 이 목표는 깨끗한 환경의 관찰 분포를 추정하여 denoising 모델의 출력을 규제하며, 기존 감독 환경에서의 최적 출력을 포함하고 있습니다. 연구에서는 DMControlGB, DMControlView 및 RL-ViGen과 같은 벤치마크에서 SCMA의 실효성을 광범위하게 테스트하였습니다.

- **Performance Highlights**: SCMA는 다양한 종류의 시각적 방해 요소에 대한 성능 격차를 크게 축소하는 것을 보여주었습니다. 자연적인 비디오 배경, 이동하는 카메라 뷰 및 가림 등 여러 상황에서 높은 성능을 기록하며, 실제 로봇 데이터에서도 효과적으로 작동하는 것을 확인하였습니다. 이러한 결과들은 SCMA가 실제 환경에서의 적응 및 성능 개선에 대한 잠재력을 지니고 있음을 나타냅니다.



### Insect-Foundation: A Foundation Model and Large Multimodal Dataset for Vision-Language Insect Understanding (https://arxiv.org/abs/2502.09906)
- **What's New**: 이 논문에서는 농업 분야의 정밀한 곤충 이해를 위한 새로운 다중 모달(conversational generative) AI 모델인 Insect-LLaVA를 제안합니다. 기존의 모델들은 일반적인 비전-언어 데이터에 기초하여 훈련되어 있었지만, 이들은 곤충에 대한 구체적인 지식이 부족했습니다. 저자들은 대규모 곤충 데이터셋과 새로운 학습 기법을 통해 곤충에 대한 비주얼 이해를 향상시키고자 했습니다.

- **Technical Details**: Insect-LLaVA는 시각적 곤충에 대한 이해를 위해 설계된 새로운 초거대 모델로, 대규모 다중 모달 곤충 데이터셋과 알고리즘이 결합되었습니다. 이 데이터셋은 곤충의 시각적 및 의미적 특성을 학습할 수 있도록 구성되었으며, 새로운 마이크로 피처(self-supervised learning) 및 Patch-wise Relevant Attention 메커니즘을 통해 미세한 차이를 포착할 수 있도록 설계되었습니다. 또한, 설명 일관성 손실(Description Consistency loss)을 통해 텍스트 설명으로부터 학습을 개선하는 방법도 제안하고 있습니다.

- **Performance Highlights**: 실험 결과, Insect-LLaVA는 곤충 관련 시각적 작업에 대해 우수한 성능을 보여주었으며, 기존의 여러 벤치마크에서 state-of-the-art 성능을 기록했습니다. 특히, Insect-VQA(Visual Insect Question Answering) 벤치마크를 도입하여 모델의 이해도를 평가하였고, 여러 곤충 분류 및 탐지 작업에서도 뛰어난 결과를 보였습니다. 이 모델은 농업과 생물학 연구에 큰 기여를 할 것으로 기대됩니다.



### FrGNet: A fourier-guided weakly-supervised framework for nuclear instance segmentation (https://arxiv.org/abs/2502.09874)
- **What's New**: 이번 논문에서는 약한 지도 학습(weakly-supervised) 환경에서 핵(instance) 세그멘테이션을 위한 새로운 Fourier Guidance 프레임워크를 제안합니다. 이 프레임워크는 Fourier Guidance 모듈을 통해 사전 정보(priori information)를 모델 학습 과정에 통합하여, 핵의 관련 특성을 효과적으로 캡처할 수 있도록 합니다. 또한, 가이드 기반 인스턴스 레벨 대비 모듈(GILC)을 통해 핵의 표현 기능을 강화하는 방식을 활용합니다. 실험 결과, 제안한 모델이 최신 최고 성능(State-of-the-Art) 방법들보다 뛰어난 성능을 보임을 확인하였습니다.

- **Technical Details**: 제안한 FrGNet 프레임워크는 핵(inuclear) 이미지를 활용하여 Fourier 변환(fourier transform)으로 생성된 마스크를 가이드로 사용합니다. 이 프레임워크는 핵의 고유한 특성을 고려하여 전통적인 방법들의 한계를 극복합니다. GILC 모듈은 인스턴스(level)의 특성을 학습시켰으며, 모델의 핵 표상(feature representation)을 더욱 향상시키는 데 기여합니다. 이 프레임워크는 두 개의 공공 데이터셋을 통해 효율성과 일반화(generalization) 능력을 입증하였습니다.

- **Performance Highlights**: 모델은 강력한 일반화 능력을 보여주며, 전혀 라벨이 없는 개인 데이터셋에서도 효과적인 핵 세그멘테이션을 수행할 수 있음을 입증했습니다. 약한 지도 학습 실험에서도 소량의 라벨링 데이터만으로 완전 지도 학습(full supervision) 성과에 가까운 결과를 유지하였습니다. 제안한 방법은 핵 세그멘테이션 분야에서 새로운 표준(SOTA) 성능 지표를 설정하였으며, 이는 의학 이미징의 발전에 기여할 것으로 기대됩니다.



### Compression-Aware One-Step Diffusion Model for JPEG Artifact Remova (https://arxiv.org/abs/2502.09873)
- **What's New**: 최근 JPEG 아티팩트 제거 작업에서 CODiff라는 새로운 압축 인식 원스텝 확산 모델이 제안되었습니다. 이 모델은 JPEG 압축 사전 정보를 효과적으로 활용하여 원본 이미지의 손실된 정보를 복원하는 능력을 가지고 있습니다. CODiff의 핵심 구성 요소인 압축 인식 비주얼 임베더(CaVE)는 JPEG 압축 과정을 학습하여 복원 품질을 개선합니다.

- **Technical Details**: CODiff는 두 단계의 학습 전략, 즉 명시적 학습(explicit learning)과 암묵적 학습(implicit learning)을 활용합니다. 명시적 학습은 낮은 품질의 이미지를 통해 품질 인자(QF)를 예측하도록 CaVE를 훈련시킵니다. 암묵적 학습은 압축된 입력에서 고품질 이미지를 복원하는 과정으로, 이를 통해 CaVE의 일반화 능력이 향상됩니다.

- **Performance Highlights**: 실험 결과, CODiff는 JPEG 아티팩트 제거 작업에서 최근의 다른 주요 방법들보다 우수한 성능을 보였습니다. 정량적 및 시각적 품질 지표 모두에서 최상의 성능을 달성하였으며, 기존의 다단계 확산 모델(MSD) 및 원스텝 확산 모델(OSD)보다 계산 비용을 줄였습니다.



### Learning to Calibrate for Reliable Visual Fire Detection (https://arxiv.org/abs/2502.09872)
- **What's New**: 이 논문은 시각적 화재 탐지에서의 불확실성 모델링을 위한 새로운 방법을 제안합니다. 특히, 기존의 ECE (Expected Calibration Error) 측정 지표를 미분 가능한 손실 함수로 변환하고, 이를 다중 클래스 화재 탐지 모델의 훈련 과정에 통합합니다. 또한, 커리큘럼 학습(Curriculum Learning)을 통해 ECE 손실의 가중치를 동적으로 조정하는 방법을 도입하여 분류 정확도와 신뢰할 수 있는 결정을 조화롭게 이룹니다.

- **Technical Details**: 이 연구에서는 한 번의 훈련 주기에서 모델이 더 간단한 작업에서 더 복잡한 작업으로 점진적으로 전환할 수 있도록 ECE 손실을 조정합니다. 다중 클래스 화재 탐지 모델의 훈련에 미분 가능한 ECE 손실을 통해 예측 불확실성을 모델링하고, 이를 교차 엔트로피 손실과 결합하여 효과적인 학습이 가능하게 합니다. 두 개의 데이터셋인 DFAN과 EdgeFireSmoke를 사용하여 제안된 방법의 효과를 검증하였습니다.

- **Performance Highlights**: 제안된 방법은 기존의 화재 탐지 모델들에 비해 개선된 보정 성능을 나타내면서도 분류 정확도를 희생하지 않고 있습니다. 실험 결과는 커리큘럼 학습을 통해 동적으로 조정된 ECE 손실이 모델의 전반적인 성능을 어떻게 향상시키는지를 보여줍니다. 이러한 접근 방식은 복잡한 상황에서 높은 정확도를 유지하면서도 오탐지를 줄이는 데 기여합니다.



### HealthGPT: A Medical Large Vision-Language Model for Unifying Comprehension and Generation via Heterogeneous Knowledge Adaptation (https://arxiv.org/abs/2502.09838)
- **What's New**: 이 논문에서는 강력한 의료 대형 비전-언어 모델인 HealthGPT를 소개합니다. 이 모델은 의료 시각 이해(comprehension) 및 생성(generation) 기능을 통합하여 자동 회귀(autoregressive) 패러다임 내에서 작동합니다. 새로운 기법인 Heterogeneous Low-Rank Adaptation(H-LoRA)을 통해 의료 도메인에 특화된 데이터 세트인 VL-Health를 활용하여 모델의 학습을 진행합니다.

- **Technical Details**: HealthGPT의 학습 프로세스는 H-LoRA와 Hierarchical Visual Perception(HVP), 그리고 Three-stage Learning Strategy(TLS)로 구성되어 있습니다. H-LoRA는 이해와 생성 작업의 학습 프로세스를 분리하여 상반된 요구 사항 사이의 충돌을 피합니다. HVP는 Vision Transformer(ViT)의 레이어를 활용하여 두 작업의 시각 세부 정보를 효과적으로 관리하며, TLS는 이해 및 생성 지식을 통합하여 고유한 LVLM(Cross-modal Vision-Language Model)을 구축합니다.

- **Performance Highlights**: 실험 결과, HealthGPT는 데이터 제약이 있는 상황에서 다양한 의료 다중 모달(multi-modal) 작업을 통합할 수 있는 능력을 입증했습니다. 기존 최신 모델(SOTA)과 비교할 때 동등하거나 더 나은 성능을 달성하며, 이 연구의 주요 기여는 통합된 Med-LVLM을 제안한 것과 데이터 충돌 문제를 완화하기 위한 H-LoRA 아키텍처를 도입한 것입니다.



### A Solver-Aided Hierarchical Language for LLM-Driven CAD Design (https://arxiv.org/abs/2502.09819)
- **What's New**: 이번 연구에서는 AIDL(AI Design Language)이라는 새로운 계층적 도메인 특화 언어(DSL)를 도입함으로써, 대형 언어 모델(LLM)을 이용한 CAD 설계를 가능하게 합니다. AIDL은 기하학적 제약 해결기에 공간적 추론을 위임하여, 모델이 복잡한 기하를 생성할 수 있도록 돕습니다. 실험 결과, AIDL은 이전의 CAD 언어와 비교하여 더 나은 시각적 결과를 제공하며, 포스트 프로세싱과 추론이 용이한 객체 생성에서 우수한 성능을 보입니다.

- **Technical Details**: AIDL은 기하학적 구조를 직접 생성하는 대신, 이를 생성하는 CAD 프로그램을 생성하는 방식으로 작동합니다. 이 DSL은 고수준 추론에 집중하고, 정밀성을 요구하는 계산 작업은 외부 솔버에 위임하는 방법론을 제안합니다. 또한, AIDL은 기존 CAD 설계의 제약조건과 의존성을 효과적으로 처리할 수 있는 계층적 접근 방식을 채택하여, 보다 높은 유연성과 편집 가능성을 제공합니다.

- **Performance Highlights**: AIDL을 사용하여 생성한 2D CAD 결과물은 OpenSCAD와 비교하여 시각적으로 동등하거나 더 나은 품질을 보였습니다. AIDL 언어의 계층 구조와 제약 조건 도입은 복잡한 다중 부품 객체를 정밀하게 구성할 수 있도록 돕습니다. 평가 결과, AIDL은 기존 CAD 모델링 언어에서 제공하는 정도 이상의 성능을 발휘하며, 언어 설계만으로도 LLM의 CAD 생성 성능을 크게 향상시킬 수 있다는 점을 입증했습니다.



### On the robustness of multimodal language model towards distractions (https://arxiv.org/abs/2502.09818)
- **What's New**: 본 논문은 시각-언어 모델(VLMs)의 편향된 상황에서의 강인성을 평가하기 위한 새로운 벤치마크인 I-ScienceQA를 소개합니다. 기존의 벤치마크들이 입력 데이터에서 방해 요소를 고려하지 않았던 반면, I-ScienceQA는 ScienceQA 데이터셋을 기반으로 시각 및 텍스트 방해 요소를 포함하여 VLM의 추론 능력을 평가합니다. 연구 결과, 최신 VLM들, 특히 GPT-4 모델이 다양한 방해 요소에 취약하다는 사실이 밝혀졌습니다.

- **Technical Details**: I-ScienceQA 벤치마크는 8,100개의 샘플로 구성되어 있으며, 4가지의 방해 요소 시나리오를 포함합니다. 시각 방해 요소는 스테이블 디퓨전 모델을 통해 생성되었으며, 텍스트 방해 요소는 GPT-3.5-turbo를 활용하여 제작되었습니다. 연구에서는 방해 요소가 VLM의 성능에 미치는 영향을 평가하며, 특히 텍스트 방해 요소에 대한 민감도가 더 높다는 것을 확인했습니다.

- **Performance Highlights**: 대부분의 VLM 모델들이 방해 요소에 대한 민감성을 보여주며, 텍스트 방해 요소가 성능 저하를 더 심각하게 일으킵니다. InternVL2와 같은 모델은 상대적으로 높은 강인성을 나타내었으며, 방해 요소에 대한 민감도를 줄이기 위한 다양한 전략들이 평가되었습니다. 이 논문의 결과는 VLM의 실제 적용에서의 한계를 이해하고 개선 방안을 제시하는 데 중요한 통찰을 제공합니다.



### Face Deepfakes - A Comprehensive Review (https://arxiv.org/abs/2502.09812)
- **What's New**: 최근 딥페이크(deepfake) 생성 기술의 획기적인 발전이 이 기술의 현실감과 능력에 엄청난 도약을 가져왔습니다. 본 논문은 딥페이크 기술에 대한 체계적이고 심층적인 분석의 부족을 지적하고, 최신 얼굴 딥페이크 생성 및 탐지 방법에 대한 철저한 이론적 분석을 제공합니다. 또한, 이 기술이 얼굴 생체 인식(face biometric recognition)에 미치는 영향에 대한 평가를 포함하고 있습니다.

- **Technical Details**: 딥페이크는 주로 비디오는 물론 이미지, 텍스트 및 오디오를 포함하는 여러 매체로 확장될 수 있으며, 본 논문에서는 사람 얼굴의 이미지 및 비디오 기반 딥페이크에 대한 논의에 제한하고 있습니다. 얼굴 조작 기술은 전체 얼굴 합성, 정체성 교환, 특성 조작, 표정 교환 등 네 가지 주요 그룹으로 나눌 수 있습니다. 각 기술 방법은 landmark 기반 기법, 오토 인코더(auto-encoder) 기반 모델, Generative Adversarial Networks(GANs)의 접근 방식으로 나뉩니다.

- **Performance Highlights**: 딥페이크 기술은 긍정적인 응용 분야와 더불어 허위 정보 전파, 신원 도용, 딥페이크 포르노 등 부정적인 영향을 미칩니다. 그러나 치료 및 심리적 응용, 오락 분야에서도 유용한 응용 프로그램이 있으며, 비디오 컨퍼런스나 상업 광고에서도 사용될 수 있습니다. 본 논문은 얼굴 딥페이크 생성 및 탐지에 대한 기존 문헌의 포괄적이고 체계적인 분석을 제공하여 기술이 미칠 긍정적 및 부정적 효과를 이해할 수 있도록 합니다.



### Vision-based Geo-Localization of Future Mars Rotorcraft in Challenging Illumination Conditions (https://arxiv.org/abs/2502.09795)
- **What's New**: 본 연구에서는 조명이 다른 환경에서도 안정적인 성능을 보이는 새로운 Map-based Localization (MbL) 시스템인 Geo-LoFTR를 소개합니다. 기존의 모델들과 달리, 이 모델은 조명 조건이나 스케일 변화에 대한 가정을 하지 않고 마스 로토크래프트의 비전 기반 지오로칼리제이션을 수행할 수 있습니다. 또한, Mars Aerial Rendering Tool for Imaging and Navigation (MARTIAN)을 활용하여 실제적인 화성 지형 이미지를 대규모로 생성하고 있습니다.

- **Technical Details**: Geo-LoFTR는 이미지 매칭을 위한 기하학적 컨텍스트를 통합하여 기존의 방법들보다 위치 정확도를 높입니다. 이 시스템은 전처리된 오르빗 맵과 로토크래프트 내비게이션 카메라로 캡처한 이미지를 등록하여 로컬라이제이션(지역화)을 수행합니다. 그리고 다양한 조명 조건 하에서의 화성 지형 이미지를 생성하기 위한 맞춤형 시뮬레이션 프레임워크를 지원합니다.

- **Performance Highlights**: 제안된 시스템은 조명 및 스케일 변동이 큰 환경에서의 로컬라이제이션 정확도 면에서 이전 MbL 접근방식을 초과하는 성능을 보입니다. 특히, 낮은 태양 고도 각도에서 조명 조건이 어려운 상황에서도 기존 방법보다 최대 31.8% 향상된 결과를 기록했습니다. 또한, 실험을 통해 새로운 접근 방식이 불리한 환경 조건에서도 견고함을 보여주었습니다.



### Noise Controlled CT Super-Resolution with Conditional Diffusion Mod (https://arxiv.org/abs/2502.09793)
Comments:
          The 8th International Conference on Image Formation in X-Ray Computed Tomography, Bamberg, Germany, August 5 - 9, 2024

- **What's New**: 본 논문은 CT 이미지를 위한 노이즈 제어 수퍼 해상도(super-resolution) 프레임워크를 소개합니다. 이는 조건부 확산 모델(conditional diffusion model)을 활용하여 새로운 방법론을 제시하며, 올바른 고해상도(High Resolution, HR) 및 저해상도(Low Resolution, LR) 이미지 쌍을 생성하는 데 중점을 두고 있습니다. 이러한 접근 방식은 실제 CT 이미지에서의 효과성을 검증하여 실제 검사에 적용 가능성을 높입니다.

- **Technical Details**: 논문에서는 Conditional Denoising Diffusion Probabilistic Model (DDPM)을 사용하여 CT 이미지의 수퍼 해상도를 달성합니다. LR 이미지를 조건으로 삼고, HR 이미지는 생성 목표로 설정되어 있으며, 마르코프 체인 형태로 진행되는 전방 과정(forward process)과 전방 산출을 위한 역방향 과정(reverse process)에 대해 다룹니다. 이 과정에서 고유 파라미터와 Gaussian 노이즈를 사용하여 이미지의 품질을 높이는 방법론을 설명합니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 실제 CT 이미지를 사용하여 효과성을 검증하였고, 높은 해상도의 세부 정보와 노이즈가 통제된 결과를 제공함을 보였습니다. 이러한 결과는 의료 영상의 진단 정확도 향상에 기여할 수 있음을 시사합니다. 노이즈와 해상도를 동시에 관리하는 이 혁신적 접근 방식은 향후 CT 이미징 분야에서의 적용 가능성을 열어줍니다.



### A CNN Approach to Automated Detection and Classification of Brain Tumors (https://arxiv.org/abs/2502.09731)
- **What's New**: 이 연구는 뇌 종양의 진단을 개선하기 위해 MRI(자기공명영상) 이미지를 처리하는 새로운 방법론을 제시합니다. Anisotropic diffusion filter를 사용한 이미지의 디노이징 기법과 다양한 딥러닝 모델(CNN)들을 적용하여 종양의 분류 정확도를 높이고자 합니다. 연구는 전이 학습 및 데이터 증강(SMOTE)을 통해 3,264개의 MRI 스캔 이미지를 활용한 뇌 종양 식별을 목표로 하고 있습니다. 시각적으로 자세한 MRI 이미지를 활용하여 뇌 질병을 검출하는 데 도움이 될 것입니다.

- **Technical Details**: 연구에서 사용된 MRI 데이터셋은 공개적으로 접근 가능한 Brain Tumour Classification 데이터베이스로, 딥러닝 모델을 학습시키기 위해 3,264개의 뇌 MRI 스캔 이미지로 구성되어 있습니다. 모델로는 ResNet152V2, VGG, ViT 및 EfficientNet이 포함되며, EfficientNet이 98%라는 최고의 정확도를 달성했습니다. 디노이징과 이미지 분류 외에도, f1 score, confusion matrix 및 Receiver Operating Characteristic (ROC) 곡선 등을 사용하여 모델의 성능을 평가합니다.

- **Performance Highlights**: EfficientNet 모델이 뇌 종양 분류에서 98%의 정확도로 최고의 성능을 보였으며, 이는 기존 연구들에서 보고된 정확도보다 높은 수치입니다. 또한 연구는 다양한 뇌 종양 유형(글리오마, 무 종양, 수막종 및 뇌하수체 종양)을 정확하게 분류하는 데 초점을 맞추고 있으며, 이는 임상 진단과 치료 계획 수립에 적극적으로 기여할 것입니다. 이 연구는 딥러닝 기법을 통해 뇌 종양 진단의 신뢰성과 속도를 높이는 데 큰 기여를 할 것으로 기대됩니다.



### ZeroBench: An Impossible Visual Benchmark for Contemporary Large Multimodal Models (https://arxiv.org/abs/2502.09696)
- **What's New**: 본 논문에서는 ZeroBench라는 새로운 비주얼 추론 벤치마크를 소개합니다. 이 벤치마크는 현재의 모든 대형 다중 모달 모델(LMMs)에게 완전히 불가능한 100개의 수작업으로 만들어진 질문과 334개의 하위 질문으로 구성되어 있습니다. 목표는 현모델들의 시각적 이해 능력을 평가하는 동시에 모델의 성능 차이를 부각시키는 것입니다.

- **Technical Details**: ZeroBench는 가벼우면서도 도전적인 특성을 지니고 있어, 모든 모델이 기초적인 질문에서 0.0%의 점수를 기록합니다. 질문은 복잡한 추론을 요구하며, 자연 이미지와 합성 이미지를 포함하여 다양한 도메인과 추론 범주를 포괄합니다. 평가 과정에서는 pass@1 및 pass@k 메트릭을 사용하며, 모델의 성능을 세부적으로 분석합니다.

- **Performance Highlights**: 20개의 최첨단 LMM을 ZeroBench에서 평가한 결과, 모든 모델이 주요 질문에서 0.0%의 점수를 기록하며 상당한 도전 과제가 되는 것으로 나타났습니다. 또한, 오류 분석을 통해 반복적인 실패 양상을 발견하였으며, 이는 주로 입력의 시각적 해석과 관련이 있습니다. 이러한 실수들은 모델이 올바른 최종 답변에 도달하는 데 큰 장애가 됩니다.



### Towards Virtual Clinical Trials of Radiology AI with Conditional Generative Modeling (https://arxiv.org/abs/2502.09688)
Comments:
          35 pages

- **What's New**: 이번 연구에서는 인공지능 (AI) 기반의 가상 임상 시험 (VCTs)을 위한 조건부 생성 모델을 처음으로 제안합니다. 이 모델은 환자의 특성에 따라 실재와 유사한 전체 신체 CT 이미지를 합성하는 능력이 있습니다. 이를 통해 AI 모델이 다양한 환자 집단에서 어떻게 성능이 저하되는지를 미리 파악하고, 편향을 분석할 수 있습니다.

- **Technical Details**: 이 생성 모델은 이미지 오토인코더와 분할 오토인코더, 그리고 잠재 확산 모델로 구성되어 있습니다. 이 구조는 전체 신체 이미지를 고해상도로 처리할 수 있게 하며, 이미지와 분할 데이터를 저차원 잠재 공간으로 압축 i고, 다시 고품질로 복원하는 기능을 제공합니다. 또한, 환자 속성에 따라 성능 변화를 식별할 수 있도록 조건부 분포를 모델링합니다.

- **Performance Highlights**: 연구 결과, 생성 모델을 통해 검증된 가상 임상 시험이 실제 데이터 없이 AI 모델의 편향과 성능 저하 영역을 정확하게 식별할 수 있음을 보여주었습니다. 특히, 전체 신체에 대한 정보와 관련된 수치(예: 체지방 및 근육량 비율)를 예측하는 다운스트림 모델에서, 우리가 예측한 성능 저하와 그 원인이 실제 임상환경에서도 관찰된 것과 일치했습니다.



### Object-Centric Latent Action Learning (https://arxiv.org/abs/2502.09680)
Comments:
          Preprint. In review

- **What's New**: 이번 연구에서는 Embodied AI(실체 AI)의 발전을 위해 object-centric latent action learning(객체 중심 잠재 행동 학습) 접근법을 제안합니다. 이 방법은 VideoSaur와 LAPO를 기반으로 하여 장면을 객체 표현으로 분해하고, 비디오 데이터를 proxy-action labels(프록시 행동 레이블)로 주석처리합니다. 그 결과, 원치 않는 배경 잡음에서 인과적 agent-object(행위자-객체) 상호작용을 분리하며, 분산자(distractor)로 인한 잠재 행동 학습 성능 저하를 줄일 수 있습니다.

- **Technical Details**: 제안된 방법은 spatio-temporal object slots(시공간 객체 슬롯)로 장면을 분해하여, 잡음 없이 인과적 agent-object 표현을 확보합니다. Self-supervised feature similarity losses(자기 지도 피처 유사성 손실)을 통해 객체 중심 표현을 생성하여, 정적 배경이나 우발적 움직임을 필터링합니다. 이러한 구조적 우선 요소는 Latent Action Models(잠재 행동 모델)가 비임무 관련 객체의 동학을 무시하고 중요한 객체에 집중하게 합니다.

- **Performance Highlights**: 소규모 레이블이 있는 행동을 통해 fine-tuning(미세 조정)을 수행한 결과, 잠재 행동의 질이 x2.7 배 개선되었으며, 응용 프로그램의 평균 return이 x2.6 배 증가했습니다. Distracting Control Suite(DCS)와 함께 진행한 초기 실험을 통해 제안된 방법의 유효성을 확인했습니다. 또한, 유사 연구와 비교할 때, 우리의 접근 방식은 더 복잡한 환경에서도 효율적으로 작동하는 것으로 나타났습니다.



### IMM-MOT: A Novel 3D Multi-object Tracking Framework with Interacting Multiple Model Filter (https://arxiv.org/abs/2502.09672)
Comments:
          8 pages,5 figures

- **What's New**: 본 연구에서는 3D Multi-Object Tracking (MOT)의 새로운 프레임워크인 IMM-MOT를 소개합니다. IMM-MOT는 Interacting Multiple Model (IMM) 필터를 사용하여 각 객체의 복잡한 운동 패턴을 정확하게 추적하며, 단일 모델 추적의 한계를 극복합니다. 또한, Damping Window 메커니즘을 통해 연속적인 궤적 상태를 관리하여 낮은 신뢰도를 가진 실제 목표를 간과하는 문제를 줄입니다.

- **Technical Details**: IMM-MOT는 4개의 주요 구성 요소로 나뉘며, 각각 전처리 모듈인 Distance-Based Score Enhancement (DBSE) 메커니즘, IMM 추적기, 연관 모듈, Damping Window (DW) 생애 주기 관리 모듈이 포함됩니다. DBSE는 LiDAR 점군의 밀도 특성을 활용하여 높은 신뢰도의 근접 객체 점수를 증가시키고, 낮은 신뢰도의 원거리 객체 점수를 감소시킵니다. 이러한 방식은 false positives (FP)와 true positives (TP) 간의 분리를 강화합니다.

- **Performance Highlights**: IMM-MOT는 NuScenes Val 데이터셋에서 73.8%의 AMOTA를 달성하였으며, 이는 CenterPoint 점군 탐지기를 사용한 방법들 중 최고 성능으로 알려져 있습니다. 기존의 단일 모델 접근법들보다 우수한 성능을 보여 다중 모델 접근법의 가능성을 입증하였습니다. 이번 연구는 MOT 분야에서의 최신 기술적 진전을 이끌어낼 것으로 기대됩니다.



### Meta-INR: Efficient Encoding of Volumetric Data via Meta-Learning Implicit Neural Representation (https://arxiv.org/abs/2502.09669)
Comments:
          Accepted by PVIS Short Paper Track

- **What's New**: 본 논문은 Meta-INR이라는 사전 훈련 전략을 제안하여, 초기 INR(Implicit Neural Representation) 파라미터를 학습하는 방법을 설명합니다. 이 방법은 대규모 시간 변동 혹은 앙상블 볼륨 데이터 세트에서 구조적 패턴이 유사한 볼륨을 독립적으로 훈련할 필요가 없도록 합니다. 본 연구는 INR을 처음부터 훈련하는 대신, 부분 관찰 데이터를 통해 초기 파라미터를 학습하여 훈련 효율을 극대화합니다.

- **Technical Details**: Meta-INR은 두 단계로 구성된 훈련 파이프라인인 메타-프리트레이닝과 볼륨 특화 파인튜닝을 사용합니다. 메타-프리트레이닝 단계에서는 원본 데이터의 1% 이하인 희소 하위 샘플링된 볼륨 데이터 세트를 활용하여 초기 INR 네트워크 파라미터를 학습합니다. 이 후 볼륨 특화 파인튜닝 단계에서 메타 모델의 초기 파라미터를 특정 볼륨에 맞게 조정하여 고신뢰도의 볼륨 재구성을 위한 적응된 INR을 생성합니다.

- **Performance Highlights**: Meta-INR을 통해 기존의 INR 훈련보다 더 빠른 수렴을 달성할 수 있으며, 단 몇 번의 파라미터 업데이트로 새로운 볼륨에 적응합니다. 이는 다양한 데이터셋 간에 이전에 보지 못한 유사한 볼륨 데이터를 인코딩하는 데 도움을 주는 고품질의 일반화 가능한 특징을 효과적으로 추출합니다. 또한, 시뮬레이션 매개변수 분석 및 대표적인 타임스텝 선택과 같은 작업에서 유용성을 강조하고 있습니다.



### Revealing Subtle Phenotypes in Small Microscopy Datasets Using Latent Diffusion Models (https://arxiv.org/abs/2502.09665)
- **What's New**: 이 논문에서는 Latent Diffusion Models(LDMs)를 활용하여 생물학적 이미지를 분석하는 새로운 방법인 Phen-LDiff를 제안합니다. 기존의 방법들이 대량의 데이터를 요구하는 반면, Phen-LDiff는 소규모의 생물학적 데이터 세트에서도 세밀한 표현형 변화를 감지할 수 있도록 설계되었습니다. 이 연구는 이미지 번역(image translation) 기술을 통해 생물학적 변이를 효과적으로 식별할 수 있음을 보여줍니다.

- **Technical Details**: Denoising Diffusion Probabilistic Models(DDPMs)는 데이터에 점진적으로 노이즈를 추가하고, 이를 회복하는 두 단계의 마르코프 과정(Markov process)을 사용합니다. Latent Diffusion Models(LDMs)는 DDPMs의 구조에 잠재 공간(latent space)을 도입하여 고차원 데이터 생성의 효율성과 유연성을 높였습니다. 저난이도의 조정으로 LDM을 활용할 수 있으며, Low-Rank Adaptation(LoRA) 기법을 적용해 고성능의 사전 훈련된 모델을 소규모 데이터로도 적절히 조정할 수 있습니다.

- **Performance Highlights**: Phen-LDiff를 통해 얻어진 결과는 세포 이미지를 바탕으로 한 표현형 변화를 효과적으로 포착함으로써, 과거의 기법들보다 더 정확한 진단과 분석을 가능하게 합니다. 특히, 이 방식은 제한된 데이터와 연산 능력 하에서도 실질적인 바이오마커 탐지에 기여할 수 있는 가능성을 보여주고 있습니다. 최종적으로, 이 연구는 생물학적 연구 및 약물 발견에 대한 접근 방식을 혁신할 수 있는 잠재력을 건설적으로 강조하고 있습니다.



### Image Super-Resolution with Guarantees via Conformal Generative Models (https://arxiv.org/abs/2502.09664)
Comments:
          11 pages, 7 figures

- **What's New**: 이 논문에서는 이미지 초해상도를 위한 새로운 불확실성 정량화 방법을 제안합니다. 제안된 방법은 'confidence mask'를 생성하여 생성된 이미지의 신뢰할 수 있는 부분을 명확히 전달합니다. 저자는 이 방법이 어떤 블랙박스 생성 모델에도 적용 가능하며, 쉽게 얻을 수 있는 데이터를 이용해 보정할 수 있음을 강조합니다.

- **Technical Details**: 제안된 방법은 conformal prediction 기술에 기반하여 신뢰성 있는 예측을 제공하며, 특정 지역의 불확실성을 감지할 수 있는 함수를 제공합니다. 모델 판별의 기준이 되는 'indecision'을 포착하는 방식으로 일반화가 가능하며, 사용자가 이를 이해할 수 있도록 설계되었습니다. 또한, 특정 로컬 이미지 유사도 측정기를 선택하여 개인화할 수 있습니다.

- **Performance Highlights**: 본 연구는 모델의 이론적 보장을 증명하며, PSNR(Peak Signal-to-Noise Ratio) 조절 및 데이터 유출에 대한 강인성을 보장합니다. 실험 결과를 통해 제안된 방법이 우수한 성능을 발휘함을 보여주며, 필터 등을 활용한 추가적인 모델링 요소의 개선도 확인했습니다.



### DiffEx: Explaining a Classifier with Diffusion Models to Identify Microscopic Cellular Variations (https://arxiv.org/abs/2502.09663)
- **What's New**: 본 논문에서는 DiffEx라는 새로운 방법을 소개하며, 이는 깊은 학습 분류기가 내리는 결정을 설명하기 위해 시각적으로 해석 가능한 속성을 생성하는 데 중점을 둡니다. DiffEx는 미세한 세포 변화를 식별하는 데 걸림돌이 되었던 기존의 블랙박스 관점을 탈피합니다. 이를 통해 질병의 이해도를 높이고 신약 개발 시 새로운 바이오마커를 발굴하는 데 도움을 줄 수 있는 가능성이 제시됩니다.

- **Technical Details**: DiffEx는 확산 모델(diffusion models)을 활용하여 분류기가 사용하는 속성을 식별하는 방법입니다. 이 과정을 위해 먼저 분류기의 속성을 포함하는 잠재 공간(latent space)을 구축하고, 대조 학습(contrastive learning) 접근 방식을 사용하여 이 잠재 공간에서 해석 가능한 방향들을 식별합니다. 발견된 방향은 분류기의 결정을 가장 크게 변화시키는 속성을 선택하여 순위화합니다.

- **Performance Highlights**: DiffEx의 효과는 자연 이미지와 생물학적 이미지를 기반으로 훈련된 분류기에 대한 설명력에서 입증되었습니다. 생물학 데이터셋을 활용하여 서로 다른 조건 간의 미세한 세포 변화를 발견하는 데 성공했습니다. 이러한 연구는 질병 및 치료 효과에 대한 이해를 심화시키고, 고유한 표현형 차이를 식별하는 데 기여할 것으로 기대됩니다.



### Towards Fine-grained Interactive Segmentation in Images and Videos (https://arxiv.org/abs/2502.09660)
- **What's New**: 이번 연구에서는 Segment Anything Model 2 (SAM2) 기반의 새로운 SAM2Refiner 프레임워크를 제안합니다. 이 아키텍처는 이미지와 비디오 모두에 대해 세분화된 분할 마스크를 생성하면서도 기존의 강점을 유지할 수 있습니다. Localization Augment, Prompt Retargeting, 및 Mask Refinement 모듈을 통합하여 사용자 프롬프트에 대한 반응성과 세부적인 구조를 고려한 세밀한 분할을 가능하게 합니다.

- **Technical Details**: SAM2Refiner 아키텍처는 글로벌 및 로컬 세부 정보의 강력한 인식을 기반으로 세밀한 분할 마스크를 생성하는데 중점을 두며, 여러 구성 요소를 포함합니다. Localization Augment는 글로벌 특징을 보강하기 위해 지역 맥락적 정보와 크로스 어텐션 메커니즘을 접목시킴으로써 세부 패턴을 탐구합니다. Prompt Retargeting는 타겟 객체의 특징과 프롬프트 사이의 정밀한 정렬을 제공하여 사용자 프롬프트의 반응 능력을 개선합니다.

- **Performance Highlights**: 광범위한 실험 결과, SAM2Refiner는 이미지와 비디오 모두에서 최첨단 방법을 초월하는 뛰어난 세부 마스크 생성 능력을 입증했습니다. 또한, SAM2Refiner는 사용자의 세부적인 요구와 복잡한 구조에 대한 반응성을 유지하며, 강력한 제로샷 능력을 가진 점이 주요 장점으로 평가받고 있습니다.



### Integrating Spatiotemporal Vision Transformer into Digital Twins for High-Resolution Heat Stress Forecasting in Campus Environments (https://arxiv.org/abs/2502.09657)
- **What's New**: 이번 연구에서는 기후 변화로 인한 극단적인 고온 사건에 대응하기 위한 디지털 트윈(digital twin) 프레임워크를 제안합니다. 이 프레임워크는 Spatiotemporal Vision Transformer (ST-ViT) 모델을 통합하여 열 스트레스 예측을 개선하는 데 중점을 두고 있습니다. 텍사스의 캠퍼스를 테스트베드로 활용하여, 고해상도 물리적 모델 시뮬레이션과 공간 및 기상 데이터를 결합했습니다.

- **Technical Details**: ST-ViT 모델은 공간적 및 시간적 데이터를 처리하여 고해상도의 인간 열 예측을 가능하게 합니다. 이 디지털 트윈은 기후 대응을 위한 데이터 기반 인사이트를 제공하여, 도시 계획자와 정책 입안자들을 지원합니다. 이를 통해, 보다 효과적인 열 완화 전략을 수립하고, 기후에 적응할 수 있는 도시 설계를 촉진합니다.

- **Performance Highlights**: 연구 결과, 디지털 트윈을 활용한 모델링이 도시의 열 스트레스 예측을 보다 정교하게 제공하는 것으로 나타났습니다. 이는 정책 입안자와 캠퍼스 이해관계자들이 보다 목표 지향적인 결정들을 내릴 수 있도록 지원합니다. 따라서, 이 프레임워크는 도시의 탄력성(resilience) 향상에 기여하며 기후 변화에 대한 효과적인 대응 방안을 제시합니다.



### Bidirectional Diffusion Bridge Models (https://arxiv.org/abs/2502.09655)
Comments:
          Source code: this https URL

- **What's New**: 이번 연구에서는 Bidirectional Diffusion Bridge Model (BDBM)을 도입하여 파트너 두 분포 간의 양방향 변환을 단일 네트워크를 통해 가능하게 합니다. 기존의 방법들은 일방향성이어서 각 방향에 대해 별도의 모델을 필요로 하여 계산 비용이 두 배로 증가하는 문제가 있었습니다. 따라서 BDBM은 이러한 한계를 극복함으로써 양방향 생성 모델을 활용할 수 있도록 돕습니다.

- **Technical Details**: BDBM은 Chapman-Kolmogorov Equation (CKE)을 활용하여 두 분포 간의 상태 전이를 모델링합니다. 이를 통해 시간 단계 간의 데이터 분포 변화를 효과적으로 처리하며, 특히 최종 점이 Gaussian 분포일 경우, 양방향 전이 커널이 해석 가능한 형태를 가져 학습 효율성을 높입니다. 이 또한 기존의 다리 방법과의 연결성을 보여주고, 그 이점을 강조합니다.

- **Performance Highlights**: 고해상도 I2I 변환 작업에 대한 광범위한 실험을 통해, BDBM은 기존의 양방향 모델에 비해 시각적 품질(FID) 및 인지 유사성(LPIPS) 측면에서 뛰어난 성능을 발휘했습니다. 이 방법은 추가적인 비용 없이도 양방향 변환을 가능하게 하며, 더 나아가 훈련 반복 횟수조차 비슷하거나 더 적게 요구하여 성능 향상에 기여합니다.



### GraphCompNet: A Position-Aware Model for Predicting and Compensating Shape Deviations in 3D Printing (https://arxiv.org/abs/2502.09652)
Comments:
          13 pages, 11 figures

- **What's New**: 본 논문은 적층 제조(additive manufacturing, AM)에서 형태 편차를 모델링하고 보정하기 위한 데이터 기반 알고리즘을 소개합니다. GraphCompNet이라는 새로운 접근 방식을 통해 포인트 클라우드 데이터와 동적 그래프 합성곱 신경망(dynamic graph convolutional neural networks, DGCNNs)을 활용하여 복잡한 형상을 모델링하고 위치 특정 열역학적 및 기계적 요인을 통합합니다.

- **Technical Details**: 이 연구는 두 단계의 적대적 훈련(adversarial training) 절차를 사용하여 보정된 설계를 반복적으로 정제하는 보정자-예측자 아키텍처(compensator-predictor architecture)를 도입했습니다. 이를 통해 실시간 피드백 및 최적화를 제공하며, 다양한 형태와 위치에 대한 실험적 검증을 통해 보정 정확도를 35~65% 향상시키는 것으로 나타났습니다.

- **Performance Highlights**: GraphCompNet은 AM 프로세스의 제어에서 발생하는 중대한 격차를 해결함으로써 Digital Twin 기술의 발전을 촉진합니다. 또한, 고정밀 자동화 산업 설계 및 제조 시스템을 지원하여, 대규모 자동화를 위한 설계 정확성과 생산 신뢰성을 크게 향상됩니다.



### MM-RLHF: The Next Step Forward in Multimodal LLM Alignmen (https://arxiv.org/abs/2502.10391)
Comments:
          Project Page: this https URL

- **What's New**: 이 논문에서는 인간의 선호와 더 잘 일치하는 Multimodal Large Language Models(MLLMs)를 개발하기 위한 주요 데이터셋 MM-RLHF를 소개합니다. 이 데이터셋은 120,000개의 세부적인 인간 주석 선호 비교 쌍으로 구성되어 있으며, 기존 자원 대비 더 큰 규모, 다양성, 주석의 세분화 및 품질을 제공합니다.

- **Technical Details**: MM-RLHF의 주요 혁신 중 하나로, Critique-Based Reward Model을 도입하여 모델 출력에 대한 비평을 생성한 후 점수를 부여합니다. 또한, Dynamic Reward Scaling이라는 방법을 통해 각 샘플의 손실 가중치를 보상 신호에 따라 조정하여 고품질 비교 쌍의 활용을 최적화합니다. 이 접근 방식은 10개의 차원과 27개의 벤치마크에서 엄격하게 평가됩니다.

- **Performance Highlights**: MM-RLHF와 정렬 알고리즘을 통해 fine-tuning된 LLaVA-ov-7B는 대화 능력이 19.5% 증가하고 안전성 면에서 60% 개선된 결과를 나타냈습니다. 이 연구는 새로운 데이터셋, 보상 모델, 훈련 및 평가 코드를 오픈소스하여 커뮤니티와의 협력을 장려합니다.



### SPIRIT: Short-term Prediction of solar IRradIance for zero-shot Transfer learning using Foundation Models (https://arxiv.org/abs/2502.10307)
- **What's New**: 본 연구에서는 "SPIRIT"라는 새로운 접근 방식을 제안하여, 최근 설치된 태양광 발전소를 위한 태양복사량 예측을 가능하게 합니다. 이 접근 방식은 기존 모델에 비해 약 70% 개선된 성능을 보이며, 과거 데이터에 의존하지 않고도 새로운 장소에서 효과적으로 작동할 수 있습니다. 또한, 실험 결과는 통계적으로 유의미한 결과를 보여줘 SPIRIT의 신뢰성을 강화합니다.

- **Technical Details**: SPIRIT는 기초 모델(Foundation models)과 물리 기반 특성(feature) 공학을 결합하여, 사이트 특화 모델 교육 없이도 효과적으로 새로운 전송 학습 시나리오에 적응할 수 있는 시스템을 개발합니다. 이는 시간이 지남에 따라 데이터가 더욱 가용해질 때 성능을 세밀하게 조정할 수 있는 유연성을 제공합니다. SPIRIT는 공중 카메라 데이터가 없는 경우에도 새로운 태양광 발전소 위치에 빠르게 적용할 수 있도록 설계되었습니다.

- **Performance Highlights**: 최신 태양광 발전소의 에너지 관리에 있어, SPIRIT는 전통적인 태양복사량 예측 모델에 비해 더욱 향상된 정확성을 자랑합니다. 이 시스템은 태양광 발전소의 급속한 확산을 지원하며, 세계 에너지 시스템에 재생 가능 에너지 통합을 가속화하는 데 기여합니다. 또한, 사실 기반으로 신뢰성을 높이기 위한 통계적 검증이 수반됩니다.



### VisCon-100K: Leveraging Contextual Web Data for Fine-tuning Vision Language Models (https://arxiv.org/abs/2502.10250)
Comments:
          Accepted at PAKDD 2025

- **What's New**: 본 논문에서는 VisCon-100K라는 새로운 데이터셋을 소개합니다. 이 데이터셋은 interleaved한 이미지-텍스트 웹 문서에서 파생되어, 45K개의 웹 문서를 100K개의 이미지 대화 샘플로 변환하는 방식으로 생성되었습니다. GPT-4V를 활용해 이미지 관련 캡션을 생성하고, OpenChat 3.5 모델로 다양한 질문-답변 쌍으로 변환하여 VLM의 성능을 개선합니다.

- **Technical Details**: VisCon-100K 데이터셋은 고유의 pipeline을 통해 생성되었으며, 이 과정에서 OpenAI GPT-4V API를 사용해 이미지 문맥 캡션을 생성합니다. 이후 OpenChat 3.5를 사용하여 이 캡션을 자유형식 및 다중 선택 질문-답변 쌍으로 변환합니다. 새로운 접근법인 'leaky modality mix'를 통해 이미지와 그 문맥 캡션 모두에서 답변이 가능한 질문을 포함함으로써 성능을 높였습니다.

- **Performance Highlights**: VisCon-100K 데이터셋은 ShareGPT4V-7b와 IDEFICS2-8b 등 두 가지 주요 VLM 접근 방식에서 우수한 성능을 보여줍니다. 특히, 이 데이터셋은 그동안의 다른 데이터셋들에 비해 더욱 풍부하고 다양한 학습 자원을 제공하여 비전-언어 모델의 파인튜닝을 효과적으로 지원합니다. 논문에서 제출한 훈련된 문맥 캡셔너는 고품질의 문맥적 캡션을 생성해 추가 연구 및 오픈 소스 응용을 촉진합니다.



### Revisiting Generalization Power of a DNN in Terms of Symbolic Interactions (https://arxiv.org/abs/2502.10162)
Comments:
          arXiv admin note: text overlap with arXiv:2407.19198

- **What's New**: 본 논문은 딥 뉴럴 네트워크(DNN)의 일반화 능력을 상호작용(interactions) 관점에서 분석하고자 합니다. DNN의 일반화 능력을 고차원 특징 공간에서 분석한 이전 연구들과 달리, 본 연구에서는 DNN을 직관적으로 이해할 수 있는 방식으로 재조명합니다. 즉, DNN의 내부 동작을 명확하게 설명할 수 있는 이론적 체계를 구축하여 이를 통해 DNN의 일반화 능력을 분석할 수 있는 새로운 접근법을 제안합니다.

- **Technical Details**: 논문에서는 DNN의 추론 패턴을 AND-OR 상호작용을 기반으로 한 논리 모델로 정확하게 표현할 수 있음을 입증했습니다. 이를 통해 DNN의 모든 세부 추론 패턴을 효과적으로 신뢰성 있게 추출하고, 일반화 가능한 상호작용과 비일반화 상호작용을 분리할 수 있는 방법론을 개발했습니다. 본 연구의 이론적 기반은 과거 연구들에서 도출된 다양한 정리와 실험 결과에 뿌리를 두고 있습니다.

- **Performance Highlights**: 실험을 통해 일반화 가능한 상호작용은 감소 형태의 분포를 따르며, 비일반화 상호작용은 방추 형태의 분포를 따른다는 주장을 입증했습니다. 이러한 상호작용의 분포는 DNN의 과적합(overfitting) 단계에서도 관찰되며, DNN의 일반화 능력 이해에 큰 기여를 할 것입니다. 본 연구는 DNN에서 실제로 사용되는 두 가지 유형의 상호작용을 효과적으로 설명하는 방법을 제안하며, 초기 실험 결과는 이론의 유용성을 잘 보여줍니다.



### MonoForce: Learnable Image-conditioned Physics Engin (https://arxiv.org/abs/2502.10156)
Comments:
          Submitted to IEEE Transactions on Robotics (T-RO), 2025. Code: this https URL

- **What's New**: 본 논문에서는 오프로드 환경에서 로봇의 경로를 예측할 수 있는 새로운 모델을 제안합니다. 카메라 이미지를 기반으로 하여, 물리학을 인식하는 신경 상징 계층(neural symbolic layer)과 대규모 데이터를 학습할 수 있는 능력을 결합한 하이브리드 모델입니다. 이 모델은 자율 로봇 분야에서 안전한 배포를 위한 도전 과제를 해결할 수 있는 잠재력을 지니고 있습니다.

- **Technical Details**: 제안된 모델은 블랙박스 구성 요소와 신경 상징 계층을 통합하여 로봇-지형 상호작용 힘을 예측합니다. 물리학 엔진을 포함한 이 계층은 지형과 접촉하는 지점에서 힘을 질의하여 로봇의 경로를 계산합니다. 이 모델은 대량의 경로를 실시간으로 제공할 수 있는 장점을 가지며, 다양한 고속 시뮬레이션 및 학습 작업에 적용 가능합니다.

- **Performance Highlights**: 제안된 모델은 비정형 지형에서도 다른 최신 방법보다 우수한 성능을 보입니다. $10^4$ 개의 경로를 초당 예측할 수 있어, 제어 모델, SLAM 등 다양한 로봇 작업에 적합합니다. 실험을 통해 이 모델이 시뮬레이션과 실제 환경 간의 격차를 줄이고, 환경에서의 안전성을 높일 수 있음을 입증했습니다.



### Hands-off Image Editing: Language-guided Editing without any Task-specific Labeling, Masking or even Training (https://arxiv.org/abs/2502.10064)
Comments:
          Published in COLING 2025

- **What's New**: 이 논문은 기존의 Instruction-guided image editing 방법의 한계를 극복하기 위해 새로운 접근 방식을 제안합니다. 이 방법은 기존의 task-specific supervision이나 데이터셋, 혹은 특정한 훈련을 요구하지 않으므로, 더 나은 개선 가능성을 제공합니다. 실험 결과는 제안한 방법이 매우 효과적이며, 경쟁력 있는 성능을 달성했다는 것을 보여줍니다.

- **Technical Details**: 제안된 방법은 입력 이미지와 출력 이미지의 캡션을 이용해 차이 벡터(difference vector)를 생성하여 이미지를 편집합니다. 입력 이미지의 캡션은 일반적인 이미지-텍스트 툴을 사용해 얻고, 출력 이미지의 캡션은 편집 지침을 포함한 적절한 프롬프트 템플릿을 통해 생성됩니다. 이 과정은 수동적 작업이나 추가 훈련 없이 자동으로 이루어질 수 있습니다.

- **Performance Highlights**: 제안된 방법은 기존의 state-of-the-art 방법들과 비교했을 때 자원 소모가 적으면서도 경쟁력 있는 성능을 보였습니다. 또한, 단순히 훈련된 모델들(LLM와 이미지 생성 모델)을 사용하므로, 이러한 모델의 성능이 향상되면 우리 방법의 성능도 증가할 가능성이 있습니다.



### ViRAC: A Vision-Reasoning Agent Head Movement Control Framework in Arbitrary Virtual Environments (https://arxiv.org/abs/2502.10046)
- **What's New**: 이 논문은 ViRAC(비전-추론 에이전트 머리 움직임 제어 프레임워크)를 제안하며, 이는 대규모 모델의 상식 지식과 추론 능력을 활용하여 자연스러운 머리 회전을 생성하는 방법을 다룹니다. 기존의 데이터 기반 접근법이 환경 변화에 적응하기 어려움을 겪는 것에 비해, ViRAC는 방대한 이미지-텍스트 데이터에서 학습된 패턴을 통해 인간과 유사한 인식을 모방합니다. 실험 결과, ViRAC는 최근의 최신 기술보다 더 자연스럽고 맥락에 맞는 머리 회전을 생성하는 것으로 나타났습니다.

- **Technical Details**: ViRAC는 인지 모듈을 두 가지로 나누어 구성합니다: 인식 모듈과 의사 결정 모듈입니다. 인식 모듈은 VLM(비전-언어 모델)과 FMM(기초 기억 모듈)을 결합하여 에이전트의 1인칭 뷰를 처리하며, 의사 결정 모듈은 AHM(행동 기록 모듈)과 LLM(대형 언어 모델)을 통합하여 고차원 인지 목표를 하위 작업으로 분해하고 다음 행동을 선택합니다. 이러한 구조는 에이전트가 맥락에 따라 동적으로 머리 회전을 수행하도록 돕습니다.

- **Performance Highlights**: ViRAC는 다양한 시나리오에서 수행한 실험을 통해 현실 인간의 머리 움직임 데이터를 더욱 정밀하게 모델링하는 것으로 평가되었습니다. 사용자 연구 결과, ViRAC의 성능은 현실감과 인지적 신빙성을 개선했으며, 사용자들은 보다 자연스러운 인터랙션을 경험했습니다. 이러한 결과는 ViRAC의 접근 방식이 가상 환경에서의 상호작용을 개선할 수 있는 큰 잠재력을 가지고 있음을 보여줍니다.



### X-Boundary: Establishing Exact Safety Boundary to Shield LLMs from Multi-Turn Jailbreaks without Compromising Usability (https://arxiv.org/abs/2502.09990)
- **What's New**: 최근 대형 언어 모델(LLMs)의 보안 취약점 중 하나인 multi-turn jailbreaks에 대한 방어 방식이 어려운 주제로 떠오르고 있습니다. 본 논문에서는 기존의 방어 메커니즘을 평가하고, 일부 방법이 LLM의 다중 턴 공격에 대한 강인성을 향상시킬 수 있지만 사용성(usability)을 저하시키는 것에 대해 논의합니다. 새로운 X-Boundary 방법론을 제안하여, 위험한 표현과 안전한 표현 사이의 경계를 명확히 설정하여 방어 성능을 개선하고 over-refusal 문제를 줄였습니다.

- **Technical Details**: X-Boundary는 위험한 표현을 안전한 표현으로부터 물리적으로 밀어내는 최적화 방법입니다. 이를 통해 위험한 표현을 정확하게 제거하면서 안전한 표현은 유지할 수 있습니다. 기존 방법들은 이러한 경계를 명확히 설정하지 못했으며, 그 결과로 over-refusal과 같은 사용성 문제를 야기했습니다. 실험 결과에 따르면, X-Boundary는 multi-turn jailbreaks에 대한 방어 성능이 기존 방법들보다 우수하고, 일반적인 기능을 거의 완전하게 유지하면서 20% 수준의 과도한 거부(over-refusal)를 감소시켰습니다.

- **Performance Highlights**: X-Boundary는 Llama-3-8B-Instruct에서 multi-turn jailbreaks의 공격 성공률(ASR)을 58.5%에서 16.5%로 감소시켰습니다. 게다가, 이 방법은 교육 과정 동안 수렴 속도를 약 22% 향상시킬 수 있다는 이론적 분석 및 실험적 검증을 제공하였습니다. 우리의 연구는 X-Boundary가 강력한 방어성과 최소한의 사용성 저하를 동시에 달성할 수 있다는 점을 강조합니다. 이런 다면적인 접근 방식을 통해 LLM의 보안과 실용성을 동시에 향상시킬 수 있는 가능성을 보여줍니다.



### Dynamic-Computed Tomography Angiography for Cerebral Vessel Templates and Segmentation (https://arxiv.org/abs/2502.09893)
- **What's New**: 본 논문은 머리의 4D-CT 데이터를 바탕으로 두 가지 혈관 분할(segmentation) 기법을 개발하고 평가하는 내용을 담고 있습니다. 첫 번째 기법은 인구 평균 혈관 아틀라스를 생성하고 등록하는 것이고, 두 번째 기법은 딥러닝(deep learning)을 사용하는 것입니다. 이 연구의 결과로, 최초로 혈관 CT 템플릿이 개발되었으며, 4D-CTA를 활용해 수동 주석(annotation)의 부담을 줄이는 방법이 제시됩니다.

- **Technical Details**: 연구는 총 25명의 환자로부터 제작된 앙지오그래픽 아틀라스를 기반으로하고, 4D-CT 이미지를 통해 동적 혈관 정보를 포착했습니다. 초기의 혈관 추적은 iCafe 도구를 사용해 수행했으며, 이를 통해 29명 환자의 동맥 및 정맥 구조를 분할했습니다. 딥러닝 모델의 훈련에는 CT 이미지가 입력으로 사용되었고, 11명의 환자를 테스트 데이터셋으로 해 성능을 평가했습니다.

- **Performance Highlights**: 딥러닝 기반의 혈관 분할 방법은 아틀라스 기반 모델에 비해 모든 혈관에 대해 더 높은 성능(예: 평균 수정 다이스 계수(amDC) 0.856 대 0.324)을 기록했습니다. 특히, ICA 및 MCA-M1 같은 주요 동맥에 대한 성능에서도 DL 모델이 아틀라스 모델보다 월등하게 우수했습니다. 이를 통해, CTA 이미징에서 신뢰할 수 있는 자동 혈관 분할 기법의 필요성이 강조됩니다.



### PUGS: Perceptual Uncertainty for Grasp Selection in Underwater Environments (https://arxiv.org/abs/2502.09824)
Comments:
          8 pages, 4 figures Accepted to International Conference on Robotics and Automation (ICRA) 2024

- **What's New**: 이 논문에서는 센서 정보가 불완전하고 불확실한 환경에서 로봇이 의사 결정을 내릴 수 있는 새로운 방법을 제안합니다. 연구의 중심은 3D reconstruction에서의 occupancy uncertainty estimation을 활용하여 로봇의 grasp selection을 보다 견고하게 만드는 것입니다. 특히, 다중 뷰 reconstruction 과정에서의 불확실성을 고려하여, 로봇의 조작 및 계획 효율성을 향상시키는 데 기여하고자 합니다.

- **Technical Details**: 제안된 방법인 Perceptual Uncertainty for Grasp Selection (PUGS)는 시뮬레이션 및 실제 환경에서 검증되었습니다. PUGS는 복합적인 occupancy field (FOF)를 구성하여 불확실성 측정치와 자세 추정을 기반으로 합니다. 이 시스템은 SLAM, semantic segmentation, 그리고 깊이 추정 시스템과 통합되어, 다중 뷰 카메라 시스템의 관찰로부터 점유 불확실성을 모델링하고 이를 통해 조작 신뢰성을 강화합니다.

- **Performance Highlights**: PUGS는 TSGrasp 방법을 기반으로 하여 불확실성을 고려한 grasp selection 성능을 평가하였습니다. 실험 결과, 제안된 방법은 불완전하고 노이즈가 많은 데이터 환경에서도 견고한 grasp selection을 가능하게 함을 보여주었습니다. 이를 통해 자율 수중 조작 시스템의 운영 비용을 줄이고 안전성을 향상할 수 있는 가능성을 제시합니다.



### Towards Patient-Specific Surgical Planning for Bicuspid Aortic Valve Repair: Fully Automated Segmentation of the Aortic Valve in 4D C (https://arxiv.org/abs/2502.09805)
- **What's New**: 본 연구에서는 bicuspid aortic valve (BAV)의 세분화(Segmentation) 프로세스를 완전히 자동화하는 nnU-Net 기반의 다중 레이블 세분화 파이프라인을 개발하였습니다. 이 시스템은 수술 전에 임상적으로 유용한 측정값(Clinical Measurements)을 만들어 내어 BAV 수술 계획을 지원하는 데 기여할 수 있습니다. 결과적으로, 자동 세분화는 수술 관련 형상 측정에서 상당한 정확도를 보였으며, 기존 수작업 세분화와의 비교에서도 일관된 결과를 나타내었습니다.

- **Technical Details**: BAV 세분화는 4D CT 스캔에서 aortic cusps 및 root wall을 포함한 여러 구조물의 길이 및 각도와 같은 형태적 측정을 자동으로 수행하기 위해 nnU-Net 모델을 사용하였습니다. 연구에는 10명의 환자에서 최소한의 석회화가 있는 BAV 스캔으로부터 총 188개의 3D 프레임이 포함되었습니다. 이 과정에서 템포럴 데이터(Temporal Data)를 활용하여 환자 맞춤형 3D CG 모델을 생성하였습니다.

- **Performance Highlights**: 자동 세분화의 평균 Dice 점수는 0.7 이상, 그리고 전체 세 개의 aortic cusps 및 root wall에 대해 대칭 평균 거리(Symmetric Mean Distance)는 0.7mm 미만으로 측정되었습니다. 임상적으로 중요한 벤치마크는 수작업 세분화와 자동 예측 세분화 간의 좋은 일관성을 보여주었습니다. 이러한 결과는 BAV 수술 위험 계층화를 위한 임상적으로 유용한 측정을 제공하는 데 기여할 수 있는 가능성을 보여줍니다.



### Acute Lymphoblastic Leukemia Diagnosis Employing YOLOv11, YOLOv8, ResNet50, and Inception-ResNet-v2 Deep Learning Models (https://arxiv.org/abs/2502.09804)
Comments:
          12 pages, 28 figures, 5 tables

- **What's New**: 이 연구는 혈액암 탐지에 있어 YOLOv11을 활용한 최초의 연구로, AI 모형이 백혈구의 악성 여부를 판별하고 다양한 ALL 단계(급성 림프모구 백혈병)를 식별할 수 있는 가능성을 보여줍니다. 또한, 악성으로 오분류되는 경우가 잦은 Hematogones와 같은 세포도 탐지할 수 있는 모델을 제안합니다. 이 연구는 다세포 샘플을 통해 실제 환경을 더 잘 반영하여 AI 모델이 높은 정확도를 유지할 수 있도록 개발되었습니다.

- **Technical Details**: 이 연구에서는 YOLOv8, YOLOv11, ResNet50 및 Inception-ResNet-v2 모델을 활용하여 이미지를 처리하고 학습하는 방법을 채택했습니다. 데이터 준비 과정에서 이미지 분할 및 데이터 증강 기법을 사용하여 흰 혈구에 초점을 맞추고 모델의 성능을 향상시킵니다. 최종적으로는 두 개의 클래스(Normal 및 Cancer)로 통합된 데이터셋을 사용하여 AI 모델을 학습시켰습니다.

- **Performance Highlights**: 연구에서 사용된 고급 딥러닝 모델들은 최대 99.7%의 높은 정확도를 기록하며, 다양한 데이터셋과 실제 환경에서도 효과적인 성능을 입증합니다. 특히, YOLOv11과 YOLOv8 모델이 채택되었으며, 모든 모델은 97%에서 99% 사이의 초기 결과를 보여주었습니다. 이러한 결과는 백혈병 탐지의 효율성을 크게 향상시킬 가능성을 엿볼 수 있게 해줍니다.



### Atom identification in bilayer moire materials with Gomb-N (https://arxiv.org/abs/2502.09791)
- **What's New**: 이번 연구에서는 van der Waals 이중层 소재의 원자 해상도 이미지를 방해하는 Moire 패턴 분석을 극복하는 방법을 제시합니다. 연구 팀은 Gomb-Net이라는 심층 학습 모델(deep learning model)을 개발하여 이중 이종 구조(bilayer heterostructures)의 각 개별 층에서 원자의 위치와 정체성을 탐지할 수 있게 되었습니다.

- **Technical Details**: Gomb-Net은 원자 종(atom species)을 개별 층에서 구별할 수 있으며, 이는 Moire 패턴을 효과적으로 분해하여 스트레인(strain) 및 도핑 물질 분포(dopant distributions)를 층별로 매핑하는 데 기여합니다. 이러한 접근법은 Moire로 인한 복잡성 문제를 해결하여, 트위스트된 분율 Janus WS2-WS2(1-x)Se2x 이종 구조 내에서 Se 원자 대체 자리를 탐색하는 데 응용되었습니다.

- **Performance Highlights**: 이 연구를 통해 발견된 바에 따르면, 층별 이식 위치(layer-specific implantation sites)는 Moire 패턴의 국부적 에너지 또는 전자 변조에 영향을 받지 않습니다. 이는 이전에 불가능했던 물질 물리학에 대한 새로운 통찰력을 제공하는 중요한 발전입니다.



### Automated Muscle and Fat Segmentation in Computed Tomography for Comprehensive Body Composition Analysis (https://arxiv.org/abs/2502.09779)
- **What's New**: 이번 연구에서는 CT 이미지를 이용한 몸 composition 분석을 위한 공개형 segmentation 및 feature calculation 모델을 제시합니다. 이 모델은 흉부, 복부 및 골반 부위의 골격근, 피하 지방(SAT), 내장 지방(VAT)을 분리하며, 2D 및 3D 평가를 지원하는 다양한 신체composition 지표를 제공합니다. 본 연구의 목표는 데이터의 접근성을 높이고, 일관된 분석 도구를 통해 임상 변수와 몸 composition 간의 관계를 심층적으로 연구할 수 있는 기회를 제공하는 것입니다.

- **Technical Details**: 모델 학습 및 평가를 위해 Duke University Health System의 483명의 환자로부터 총 813개의 CT 볼륨을 수집하였습니다. nnU-Net 아키텍처를 사용하여 근육 밀도, VAT/SAT 비율, 근육 면적/부피 및 골격근 지수(SMI)와 같은 지표를 분석할 수 있도록 설정하였고, Sparsely Annotated Region and Organ Segmentation (SAROS) 데이터셋을 사용하여 세분화의 일반화 가능성을 입증하였습니다. 이는 연구자들이 다양한 데이터와 비교하여 정확도를 평가하고, clinical outcomes와 상관관계를 연구할 수 있는 기초가 됩니다.

- **Performance Highlights**: 모델은 내부 및 외부 데이터셋 모두에서 높은 dice 계수를 기록하여 골격근, SAT 및 VAT 세분화에서 89% 이상을 달성하였습니다. 특히, 골격근에서는 2.40%, SAT에서는 10.26%의 성능 향상을 보여주었으며, 모든 측정에서 평균 상대 절대 오차(MRAE)가 10% 아래로 유지되었습니다. 또한, 근육 지방 세분화는 56.27%의 Dice 계수를 달성하여 추가 분석에 활용될 수 있는 가능성을 지닙니다.



### CellFlow: Simulating Cellular Morphology Changes via Flow Matching (https://arxiv.org/abs/2502.09775)
- **What's New**: CellFlow는 생물학적 데이터의 주요 과제인 배치 효과(batch effects)를 구별하여 화학적 및 유전적 교란(perturbations)에 의해 유도된 세포 형태 변화(Cellular morphology changes)를 정확히 시뮬레이션하는 이미지 생성 모델입니다. 이 모델은 기존 방법과 달리 비혼란(cell states)에서 혼란된(cell states) 상태로의 변환을 배포별(distribution-wise)로 모델링하여 실험적 아티팩트(artifacts)에서 실제 교란의 효과를 효과적으로 구별합니다.

- **Technical Details**: CellFlow 모델은 세포 형태 예측을 배포 간(mapping) 문제로 정의하고, 유량 매칭(flow matching) 기법을 활용하여 세포 이미지의 변환을 지속적으로 수행합니다. 이 과정에서 원래 세포 분포(p0)는 실험적 웰에서 확보된 이미지로부터 수집되며, 이를 통해 다양한 실험 조건에서도 일관성 있는 성능을 발휘합니다.

- **Performance Highlights**: CellFlow는 BBBC021, RxRx1 및 JUMP 데이터셋에서 뛰어난 성능을 보이며, 기존 방법보다 35% 향상된 FID scores와 12% 향상된 작용 모드(mode-of-action) 예측 정확도를 달성했습니다. 또한, Batch effects 수정 및 세포 상태 간의 양방향 보간(bidirectional interpolation) 기능을 통해 생물학 연구에 대한 새로운 통찰력을 제공합니다.



### Large Language Models and Provenance Metadata for Determining the Relevance of Images and Videos in News Stories (https://arxiv.org/abs/2502.09689)
- **What's New**: 이 논문은 효과적인 허위 정보 캠페인이 텍스트와 함께 이미지 및 비디오를 활용하여 잘못된 내러티브(narrative)를 강화하는 방식을 탐구합니다. 기존의 허위 정보 탐지 기법들은 서로 다른 매체 간의 상호작용을 간과하는 경향이 있으며, 본 연구는 대형 언어 모델(LLM)을 기반으로 이러한 문제를 해결하고자 합니다. 연구에서는 뉴스 기사의 텍스트와 포함된 미디어의 출처 메타데이터를 분석하여 미디어가 해당 기사의 맥락과 관련이 있는지 판단하는 시스템을 제시합니다.

- **Technical Details**: 제안된 방법은 뉴스 기사의 제목, 본문 및 첨부된 미디어를 입력으로 받아 미디어의 출처 및 편집 여부가 관련성이 있는지를 평가합니다. 이 과정에서 데이터 출처(provenance) 메타데이터가 사용되며, 이는 정보의 정확성과 진위성을 보장하는 데 필수적입니다. 메타데이터는 중앙 발행 기관 또는 블록체인을 통해 조작 방지 기능을 포함할 수 있습니다.

- **Performance Highlights**: 제안된 접근법은 다양한 데이터 출처 프레임워크 및 LLM과 함께 작동할 수 있도록 설계되어 기술의 발전을 지속적으로 활용할 수 있는 장점을 가지고 있습니다. 프로토타입 시스템은 오픈소스로 제공되며, 실제 뉴스 기사에서 해당 미디어가 관련성이 있는지를 분석할 수 있는 웹 인터페이스를 제공합니다. 이러한 기술은 소셜 미디어 및 블로그 게시물과 같은 다른 맥락에도 적용 가능성이 있습니다.



### Generalizable Cervical Cancer Screening via Large-scale Pretraining and Test-Time Adaptation (https://arxiv.org/abs/2502.09662)
- **What's New**: 이 논문은 자궁경부암 선별 검사를 위한 일반화 가능한 Smart-CCS 패러다임을 소개합니다. 이는 대규모 데이터셋 CCS-127K를 기반으로 하여, 자가 감독 기반 학습 및 훈련 후 적응을 통해 강력한 일반화 능력을 갖춘 검진 시스템을 개발합니다. Smart-CCS는 다양한 임상 환경에서 적용할 수 있는 암 선별의 정확도를 높이는 데 초점을 맞추고 있습니다.

- **Technical Details**: Smart-CCS는 48개 의료 센터에서 수집된 127,471개의 자궁경부 세포 전체 슬라이드 이미지를 포함하는 CCS-127K 데이터셋을 사용하여 개발되었습니다. 시스템은 세 단계로 구성되며, 첫 번째 단계는 대규모 자가 감독 학습이고, 두 번째 단계는 WSI(Whole Slide Image) 분류를 위한 모델을 미세 조정(finetuning)하는 것이며, 세 번째 단계는 테스트 시간에 적응하여 성능을 최적화합니다.

- **Performance Highlights**: Smart-CCS는 내부 테스트 데이터 세트 11개에서 0.965의 AUC와 0.913의 민감도로 자궁경부암 선별 검사를 수행하는 데 성공했습니다. 외부 테스트에서도 0.950의 AUC 값을 유지했으며, 3개의 전향적(center) 센터에서 AUC 값은 각각 0.947, 0.924, 0.986으로 나타났습니다. 이 시스템은 조직학적 진단 결과로 검증된 민감도와 정확성을 보여줍니다.



### Multi-Omics Fusion with Soft Labeling for Enhanced Prediction of Distant Metastasis in Nasopharyngeal Carcinoma Patients after Radiotherapy (https://arxiv.org/abs/2502.09656)
- **What's New**: 최근 연구에서는 'omics fusion'이 의료 영상 처리 분야에서 중요한 전처리(preprocessing) 방법으로 떠오르고 있습니다. 이 연구는 데이터 소스와 의료 이미지 장비 간의 불일치로 인해 발생하는 불확실성을 극복하기 위한 새로운 융합(fusion) 방법론을 개발하는 데 중점을 두고 있습니다. 특히, 다중 커널(latency) 후처리 방법은 이 문제를 해결하기 위해 효과적인 전략으로 인기를 끌고 있습니다.

- **Technical Details**: 기존의 연구들은 고차원(high-dimensional) 특징을 처리하는 데 어려움이 있었습니다. 이 연구에서는 단일 커널(single-kernel) 함수를 이용하여 데이터의 본질적인 특성을 매핑(mapping)하고, 그 후 고차원 공간에서 병합합으로써 이러한 차이를 효과적으로 해결합니다. 그러나 다중 커널 후처리 방법은 복잡한 비인두암(nasopharyngeal carcinoma) 데이터셋에서 레이블(label) 적합의 경직성을 초래하여 일반 분류기의 효율성에 제약을 가합니다.

- **Performance Highlights**: 이 방법론은 두 집단 간의 차이를 증가시켜 레이블 할당에 대한 더 유연한 구조를 제공합니다. NPC-ContraParotid 데이터셋에 대한 검토 결과, 모델의 견고성과 효과가 입증되었으며, 이는 비인두암 환자의 원거 전이(distant metastases)를 예측하는 데 있어 가치 있는 도구로 자리 잡을 가능성을 시사합니다.



### Heterogeneous Mixture of Experts for Remote Sensing Image Super-Resolution (https://arxiv.org/abs/2502.09654)
- **What's New**: 이번 연구에서는 원격 감지 이미지의 초해상도(Super-Resolution, SR) 문제를 해결하기 위해 Mixture of Experts (MoE) 모델을 도입합니다. 기존 방법들은 다양한 지상 객체를 효과적으로 처리하지 못했지만, 본 연구에서는 이질적인 전문가 집단을 구성하여 각각의 지상 객체의 복잡한 특성을 전문적으로 처리할 수 있도록 합니다. Multi-level Feature Aggregation (MFA) 전략을 통해 전문가의 활성화 확률을 추정하고, 이중 라우팅 메커니즘을 통해 각 픽셀에 최적의 전문가를 선택합니다.

- **Technical Details**: 모델 구현 시, 저해상도 이미지 𝐈𝐿𝑅𝑖 (I_LR) 입력을 바탕으로 높은 해상도 이미지 𝐈𝐒𝑅𝑖 (I_SR)를 생성하는 과정을 정의합니다. 이 과정에서 Residual Hybrid Attention Groups (RHAG)를 특징 추출 백본으로 사용하고, MFA 모듈을 도입하여 다양한 전문가를 활성화합니다. DR-HMoE 구조는 이질적인 전문가 네트워크와 라우터를 포함하며, 각 전문가들은 서로 다른 커널 사이즈를 사용하여 다양한 유형의 지상 객체를 적절히 처리할 수 있도록 합니다.

- **Performance Highlights**: UCMerced 및 AID 데이터셋에서 실험한 결과, 제안한 MFG-HMoE 방법은 최신의 초해상도 방법들과 비교했을 때 더 높은 SR 재구성 정확도를 보여주었습니다. 이러한 성능 향상은 지상 객체에 대한 복잡한 세부 사항을 효과적으로 다루기 위한 전문화된 활성화 매개변수를 통해 가능해졌습니다. 이 연구는 원격 감지 이미지의 처리에서 새로운 패러다임을 제시하며, 실용적인 적용 가능성을 높입니다.



### SASVi - Segment Any Surgical Video (https://arxiv.org/abs/2502.09653)
- **What's New**: 본 논문에서는 SASVi라는 새로운 비디오 분할 파이프라인을 제안합니다. 이 모델은 SAM2라는 기초 모델의 성능을 높이기 위해 프레임 기반 Mask R-CNN Overseer 모델을 기반으로 한 리프롬프트(re-prompting) 메커니즘을 포함하고 있습니다. 이 접근법을 통해 수술 비디오에 대한 매끄럽고 일관된 분할이 가능해 졌습니다.

- **Technical Details**: SASVi는 비디오 시퀀스를 온전히 세그먼트화하기 위해 시각적으로 변화하는 장면에서 자동으로 SAM2에 리프롬프팅을 수행합니다. Mask R-CNN, DETR 및 Mask2Former 같은 객체 탐지 모델을 사용하여 현재 비디오에 있는 엔티티를 모니터링합니다. 이 메커니즘은 추적되지 않은 클래스가 장면에 들어가거나 이전에 추적한 엔티티가 떠날 때 이를 포착하여 정확한 프레임 분할을 가능하게 합니다.

- **Performance Highlights**: SASVi는 여섯 개의 분할 기술 중에서 최소 1.5% 이상의 성능 향상을 보여주었으며, 수술 비디오의 주요 품질 지표인 시간적 일관성을 크게 개선했습니다. 우리는 세 가지 다른 수술 데이터셋에서 정량적 및 정성적으로 이 접근법의 효과를 입증하였고, 이로 인해 제한된 주석 데이터에서도 완전한 주석을 얻을 수 있음을 보여주었습니다.



### Imit Diff: Semantics Guided Diffusion Transformer with Dual Resolution Fusion for Imitation Learning (https://arxiv.org/abs/2502.09649)
- **What's New**: 이번 논문에서는 Imit Diff라는 새로운 세맨틱 가이드 확산 변환기(diffusion transformer)를 소개합니다. 이 접근법은 고차원 시각 정보와 로봇의 고유 감각(proprioception)을 활용하여 비디오 시연으로부터 조작 기술을 효과적으로 학습하는 데 기여합니다. Imit Diff는 복잡한 장면과 시각적 방해 요소가 증가하는 상황에서도 뛰어난 성능을 발휘하는 것으로 나타났습니다.

- **Technical Details**: Imit Diff는 세 가지 주요 구성 요소로 구성됩니다: 첫째, 세맷틱 인젝션(semantics injection)을 통해 고차원 시각 데이터에서 픽셀 수준의 시각적 위치 지정 정보를 추출합니다. 둘째, 이 정보는 듀얼 해상도(fusion) 비전 인코더에서 추출한 다중 스케일 비주얼 특징과 접목시킵니다. 셋째, 확산 변환기 아키텍처 내에서 일관성 정책(consistency policy)을 구현하여 실시간 성능과 움직임의 부드러움을 향상시킵니다.

- **Performance Highlights**: Imit Diff는 여러 실세계 작업에서 평가된 결과, 특히 시각적 방해가 있는 복잡한 장면에서 기존의 최신 방법들보다 상당히 우수한 성능을 보였습니다. 또한, 제로샷(experiments) 실험을 통해 시각적 방해 및 범주 일반화에서 이 모델의 장점을 입증했습니다. 코드 또한 곧 공개될 예정입니다.



New uploads on arXiv(cs.AI)

### Representation and Interpretation in Artificial and Natural Computing (https://arxiv.org/abs/2502.10383)
- **What's New**: 이 논문에서는 인간의 주관적 해석과 기계의 객관적 변환 과정을 구분하고, 자연 계산(natural computing)의 개념을 제시하면서 기존의 튜링 기계(Turing Machine)를 넘어서 다양한 계산 모드를 탐색합니다. 특히, 인간의 뇌가 튜링 기계와 동일한 방식으로 작동하는지에 대한 의문을 제기하며, 이와 같은 자연 계산 모드를 이해하는 것이 의식의 난제(hard problem of consciousness)를 해결하는 중요한 단서가 될 수 있음을 강조합니다.

- **Technical Details**: 자연 계산(natural computing)은 튜링의 개념과 다른 새로운 계산 방식을 통해 이루어질 수 있으며, 이들 방식은 동일한 범주에 속하지 않아서 비교할 수 없다는 점이 논의됩니다. 데이터의 표현(representation)과 해석(interpretation) 과정에서 수많은 매체와 형식이 존재하며, 다양한 물리적 방법을 통해 계산이 이루어질 수 있다는 가능성도 제시됩니다.

- **Performance Highlights**: 이 논문에서는 특히 기계와 인간의 해석이 상호작용하는 과정을 통해, 계산이 단순한 기계적 기능을 넘어서는 복잡한 현상임을 탐구합니다. 비록 새로운 계산 체계가 튜링 기계의 유용성을 초월할 가능성이 있으나, 그러한 새롭고 복잡한 계산 모드가 존재하지 않는다면 자연 계산은 성립되지 않을 것이라는 주장을 통해 논의의 깊이를 더합니다.



### LLM-Powered Preference Elicitation in Combinatorial Assignmen (https://arxiv.org/abs/2502.10308)
- **What's New**: 이번 연구는 대규모 언어 모델(LLM)이 조합 할당(combinatorial assignment)에서 선호 추출(preference elicitation) 간소화의 인간 대리(proxy)로서의 가능성을 탐구합니다. 전통적인 선호 추출 방법이 반복적인 질문에 의존하는 반면, LLM은 인력 소모를 줄이면서 일회성(one-shot) 대안을 제공합니다. 우리는 LLM 대리자를 위한 프레임워크를 제안하며, 이는 최신 머신러닝(SOTA ML) 기반 선호 추출 방식과 함께 작동합니다.

- **Technical Details**: 제안된 프레임워크는 LLM이 도입하는 새로운 도전 과제인 응답 변동성(response variability) 및 증가된 계산 비용(computational costs)을 처리할 수 있습니다. 우리는 학습된 모델이 이 연구를 성공적으로 수행하기 위해 필요한 능력을 조사하며, 실험적인 평가를 통해 LLM 대리자의 효율성을 확인하였습니다. 우리 연구는 잘 연구된 과정 할당(course allocation) 분야에서 인간 질의(human queries)에 대한 상대적 성능을 분석합니다.

- **Performance Highlights**: 연구 결과, 우리의 접근 방식은 할당 효율성(allocative efficiency)을 최대 20% 향상시켰습니다. 이 성과는 다양한 LLM 및 보고 품질과 정확도의 차이에 상관없이 강력하게 나타났습니다. 이러한 결과는 LLM을 선호 추출 과정에서 하나의 유용한 도구로 자리매김하게 합니다.



### Reinforcement Learning in Strategy-Based and Atari Games: A Review of Google DeepMinds Innovations (https://arxiv.org/abs/2502.10303)
- **What's New**: 이번 논문은 Google DeepMind의 강화 학습(RL) 알고리즘과 이를 통해 발전된 AI 모델들에 대해 리뷰합니다. 특히 AlphaGo, AlphaGo Zero, MuZero와 같은 게임 관련 모델들의 발전을 중점적으로 분석하고, 이러한 모델이 게임 학습에 기여한 혁신적 접근 방식을 소개합니다. 또한, Atari 게임과 전략 기반 게임에서 RL의 중요성과 미니제로 및 다중 에이전트 모델과 같은 새로운 발전 방향도 다룹니다.

- **Technical Details**: 강화 학습은 AI가 환경과 상호작용하며 학습하는 원리로, 에이전트가 특정 상태에서 행동을 취하고 보상을 극대화하는 것을 목표로 합니다. Deep Reinforcement Learning(DRL)은 강화 학습과 딥러닝의 결합으로, Neural Turing Machines(NTMs)과 Deep Q-Network(DQN) 알고리즘을 통해 고차원 감각 입력에서 학습하는 데 성공하였습니다. 경험 재생(Experience Replay) 기법과 Actor-Critic(A3C) 모델을 도입하여 훈련의 안정성과 속도를 높였습니다.

- **Performance Highlights**: AlphaGo는 2016년에 세계 바둑 챔피언을 이긴 AI 모델로, 강화 학습의 효용을 실증적으로 보여주었습니다. 이후 AlphaGo Zero는 인간의 플레이 데이터를 사용하지 않고 자가 학습을 통해 더욱 발전하였고, MuZero는 규칙에 대한 명시적 지식 없이 게임 환경의 내부 역학을 학습하는 능력을 가지고 있습니다. 본 논문에서는 이러한 모델들이 AI 연구에 미친 영향과 앞으로의 가능성에 대해 논의합니다.



### Do Large Language Models Reason Causally Like Us? Even Better? (https://arxiv.org/abs/2502.10215)
- **What's New**: 이 논문은 인공지능의 중심적 요소인 인과적 추론이 인간과 대규모 언어 모델(LLMs) 간의 차이를 비교하는 데 중점을 두고 있다. 이를 위해, 연구자들은 Collider 그래프를 기반으로 한 작업을 통해 LLM의 인과 추론 능력을 평가하고, 다양한 모델의 성능을 분석하였다. LLM은 인간의 인과 추론과 비교하여 규범적 추론(normative inference)과 유사한 행동을 보였으며, 모델에 따라 그 결과가 달라지는 경향이 있었다.

- **Technical Details**: 연구에서는 4개의 대형 언어 모델(GPT-3.5, GPT-4o, Claude-3-Opus, Gemini-Pro)과 인간 데이터 간의 인과적 추론 비교를 진행하였다. 각 LLM은 동일한 추론 작업을 수행했으며, Collider 구조를 사용하여 예측적 추론(predictive inference), 진단적 추론(diagnostic inference) 등의 다양한 추론 유형을 도출하였다. 또한, Causal Bayes networks(CBNs)를 사용하여 인간 및 LLM의 추론 패턴을 비교하였다.

- **Performance Highlights**: 연구 결과, GPT-4o와 Claude 모델이 규범적 행동을 가장 잘 나타냈으며, '설명 회피(explaining away)'와 같은 인과적 추론 패턴을 보였다. 반면, Gemini-Pro와 GPT-3.5는 이러한 규범적 인과 추론에서 상대적으로 약한 성과를 나타내었다. 전반적으로 모든 모델은 인과의 독립성에서 예상되는 이탈을 보였으나, 강한 연관적 추론과 예측적 추론을 통해 인과적 관계에 대한 이해를 보여 주었다.



### MathConstruct: Challenging LLM Reasoning with Constructive Proofs (https://arxiv.org/abs/2502.10197)
- **What's New**: 이 논문에서는 대형 언어 모델(Large Language Models, LLMs)의 수학적 성능을 평가하기 위한 새로운 벤치마크인 \\mc를 소개합니다. 기존 수학 벤치마크가 갖는 제한점을 해결하기 위해, 다양한 수학 경시대회에서 출처를 두고 선정된 126개의 어려운 문제로 구성된 새로운 기준을 설정하였습니다. 이 벤치마크는 정적 요구사항이 있는 문제들을 넘어서서 수학적 객체를 특정 속성을 가진 방식으로 구성하는 것을 요구하는 실용증명(constructive proofs)에 초점을 맞추고 있습니다.

- **Technical Details**: \\mc는 다양한 수학적 문제변형을 자동 생성할 수 있는 검증기(automated verifiers)를 포함합니다. 이를 통해 LLM의 견고함(robustness)을 평가하기 위한 문제 변형들을 생성할 수 있습니다. 기존의 수학 문제들은 문제의 단순성이나 기억 또는 추측의 가능성 때문에 포화 상태에 이르렀고, 이로 인해 LLM의 진정한 수학적 능력을 간과하기 쉬웠습니다. 이 새로운 접근은 수학 문제 해결을 평가하기 위한 보다 다양하고 복잡한 환경을 제공합니다.

- **Performance Highlights**: 최첨단 LLMs는 단지 54%의 MathConstruct 문제를 해결할 수 있었으며, 이는 새로운 벤치마크의 복잡성과 LLM 평가에서의 중요성을 강조합니다. 수학적 증명이 비교적 쉽게 해결 여부를 검증할 수 있는 특성을 가지고 있지만, 여전히 많은 현대 LLM들이 이러한 문제들에 대해 도전에 직면하고 있음을 나타냅니다. 이러한 발견은 LLM 성능 개선을 위한 향후 연구 방향에 중요한 기초 자료로 작용할 것입니다.



### STMA: A Spatio-Temporal Memory Agent for Long-Horizon Embodied Task Planning (https://arxiv.org/abs/2502.10177)
- **What's New**: 이 논문은 Embodied Intelligence 에이전트가 동적 환경에서 장기 작업을 수행하는 능력을 강조하고 있습니다. 이를 위해 Spatio-Temporal Memory Agent (STMA)라는 새로운 프레임워크를 제안하며, 이 프레임워크는 공간-시간 메모리(spatio-temporal memory)를 통합하여 작업 계획 및 실행을 향상시킵니다. STMA는 역사적 및 환경적 변화를 실시간으로 포착하는 메모리 모듈, 적응형 공간 추론을 촉진하는 동적 지식 그래프, 작업 전략을 반복적으로 개선하는 계획자-비평자(planner-critic) 메커니즘으로 구성됩니다.

- **Technical Details**: STMA는 동적인 지식 그래프와 프로세스를 통합하여 복잡한 동적 환경에서의 작업 계획을 향상시키고 있습니다. 이를 통해 에이전트는 과거 정보를 효과적으로 통합하고 변화하는 환경에 적응할 수 있습니다. 논문에서는 TextWorld 환경에서 32개의 작업을 대상으로 STMA의 성과를 평가했으며, STMA는 기존 모델보다 31.25%의 성공률 개선과 24.7%의 평균 점수 증가를 실현했습니다.

- **Performance Highlights**: STMA는 기존의 최신 기법들에 비해 다양한 핵심 지표에서 뛰어난 성능을 보여줍니다. 특히, 복잡한 동적 환경에서의 계획 및 적응 능력이 향상되었음을 나타내며, 메모리 결합 방식을 통해 에이전트가 수렴할 수 있는 최적의 전략을 신속하게 개발할 수 있습니다. 이러한 결과는 공간-시간 메모리가 체화된 에이전트의 메모리 능력을 향상시키는 데 효과적임을 강조합니다.



### Cooperative Multi-Agent Planning with Adaptive Skill Synthesis (https://arxiv.org/abs/2502.10148)
- **What's New**: 이번 연구에서는 COMPASS라는 새로운 다중 에이전트 시스템 아키텍처를 제안합니다. 이 시스템은 비전-언어 모델(VLM)과 동적 스킬 라이브러리, 구조화된 소통을 통합하여 분산된 닫힌 루프 의사결정을 가능하게 합니다. COMPASS는 기초적인 텍스트 관찰에 의존하지 않고, 에이전트 간의 정보 전파를 통해 비가시적인 환경에서의 상호작용을 지원합니다.

- **Technical Details**: COMPASS는 비전-언어 모델(VLM)을 기반으로 한 플래너와 샘플 개연성을 높이기 위한 동적 스킬 라이브러리로 구성됩니다. 플래너는 시각적 및 텍스트 관찰을 통해 최적의 실행 코드를 제안하며, 코드-정책 패러다임을 도입하여 일반화 가능성을 높입니다. 구조화된 소통 프로토콜을 통해 에이전트 간의 정보를 추론할 수 있는 멀티 홉 전파 메커니즘을 구현했습니다.

- **Performance Highlights**: SMACv2에서 COMPASS는 최신 MARL 알고리즘에 비해 최대 30% 높은 승률을 기록했습니다. 특히, Protoss 시나리오에서 57%의 승률을 달성하며, QMIX, MAPPO, HAPPO, HASAC 등 최첨단 알고리즘들을 크게 초월하는 성과를 보였습니다. COMPASS는 기초 전문가 시연으로부터 효율적인 전략을 도출할 수 있는 능력도 입증하였습니다.



### Causal Information Prioritization for Efficient Reinforcement Learning (https://arxiv.org/abs/2502.10097)
- **What's New**: 본 논문에서는 샘플 효율성을 개선하기 위해 인과 관계를 활용하는 새로운 접근법인 Causal Information Prioritization (CIP)을 제안합니다. CIP는 보상에 대한 인과 관계를 이해하여 샘플 효율성을 높이는 것을 목표로 하며, 기존의 RL 방법들이 간과한 인과 관계를 활용합니다. 또한 CIP는 고립된 상태 특성을 탐지하고 조작하여 환경 상호작용을 줄이는 측면에서도 혁신적입니다.

- **Technical Details**: CIP는 인과적 MDP를 기반으로 상태, 행동, 보상 간의 인과 관계를 추론하여 보상 중심의 구조 모델을 구축합니다. 이 접근법에서는 반사적 데이터 증강을 통해 중요 상태 전환을 우선시하며, 행동과 보상 간의 인과 관계를 활용하여 행동의 가중치를 조정합니다. 마지막으로, CIP는 에이전트가 환경에 대해 제어 가능한 능력을 정량화하는 인과성을 고려한 empowerment 학습 목표를 통합하여 탐색 효율성을 향상시킵니다.

- **Performance Highlights**: CIP는 39개의 다양한 과제를 포함한 대규모 실험을 통해 기존 RL 방법들을 일관되게 초월하는 성과를 나타냈습니다. 이러한 실험은 이동 및 조작 기술 학습을 포함하여, 픽셀 기반 및 희소 보상 설정에서도 이루어졌습니다. CIP의 연구 결과는 복잡한 환경에서 보상 지향 행동을 보다 효율적으로 수행할 수 있도록 하는 인과적 접근법의 효과를 입증합니다.



### Towards Empowerment Gain through Causal Structure Learning in Model-Based RL (https://arxiv.org/abs/2502.10077)
- **What's New**: 이 논문은 Empowerment through Causal Learning (ECL)이라는 새로운 모델을 제안하며, 이는 인과적 구조를 활용하여 에이전트의 제어 능력과 학습 효율성을 향상시킵니다. ECL은 데이터 수집을 통해 환경의 인과 동역학 모델을 학습한 후, 탐색을 위해 인과 구조 하에서의 에너지를 최대화하는 방법으로 작동합니다. 기존의 MBRL 방법들과 비교하여 인과 발견과 샘플 효율성에서 개선된 성과를 보여줍니다.

- **Technical Details**: ECL 프레임워크는 세 가지 주요 단계로 구성됩니다: 모델 학습, 모델 최적화 및 정책 학습입니다. 첫 단계에서는 보상 모델과 함께 인과 동역학 모델을 학습하고, 두 번째 단계에서는 학습한 인과 구조를 기반으로 에너지를 주도한 탐색 정책을 통합하여 환경을 제어합니다. 마지막 단계에서는 최적화된 인과 동역학 및 보상 모델을 활용하여 과적합을 방지하며 정책을 학습합니다.

- **Performance Highlights**: ECL은 6개 환경과 3개의 인과 발견 방법을 결합하여 평가되었으며, 기존의 인과 MBRL 방법들보다 우수한 성능을 입증했습니다. 특히, ECL은 인과 발견 정확성과 샘플 효율성, 비대칭 성능 측면에서 두드러진 결과를 축적하였습니다. 논문의 결과는 ECL의 유용성이 다양한 환경에서 확실히 드러난다는 것을 보여줍니다.



### Unsupervised Entity Alignment Based on Personalized Discriminative Rooted Tr (https://arxiv.org/abs/2502.10044)
- **What's New**: 본 논문에서는 Entity Alignment (EA) 문제를 해결하기 위해 새로운 비지도(unwupervised) 방법인 UNEA를 제안합니다. 기존의 비지도 EA 방법들의 두 가지 한계를 개선하여, 개체의 개인화된 (personalized) 임베딩을 추출하고 분포 왜곡(distribution distortion)을 완화하는데 집중합니다. UNEA는 각 개체에 대해 고유한 트리 이웃을 샘플링하고, 상호 정보(mutual information)를 최대화하는 보조 작업을 통해 모델을 정규화합니다.

- **Technical Details**: UNEA는 powerful LLM을 사용하여 개체 및 관계의 임베딩을 초기화하고, 개체에 대해 개인화된 트리 이웃을 추출합니다. 이러한 맞춤화된 트리 이웃은 각 개체가 최적의 집계 경로(aggregation path)를 학습하도록 도와주어 개체 임베딩의 개인화 수준을 높입니다. 또한, 상호 정보 최대화(mutual information maximization) 기법을 도입하여, 모델이 개체와 관계의 고차원 임베딩을 지속적으로 정규화하여 정보 분포 왜곡 문제를 방지합니다.

- **Performance Highlights**: UNEA는 광범위한 실험을 통해 기존의 비지도 EA 방법과 여러 지도(supervised) 방법들보다 뛰어난 성능을 발휘하는 것을 보여주었습니다. 실험 결과, UNEA는 최신 비지도 기준선(state-of-the-art)은 물론이고 기존의 지도 기준선과 비교하여도 월등한 성능을 기록하였습니다. 이는 UNEA의 접근 방식이 EA 과제를 해결하는 데 매우 효과적임을 나타냅니다.



### POI-Enhancer: An LLM-based Semantic Enhancement Framework for POI Representation Learning (https://arxiv.org/abs/2502.10038)
- **What's New**: 본 논문은 POI (Points of Interest) 표현 학습을 향상시키기 위한 새로운 프레임워크인 POI-Enhancer를 제안합니다. 이 프레임워크는 대형 언어 모델(LLMs)을 활용하여 POI의 임베딩 벡터를 개선하고, 다양한 정보 출처로부터의 데이터 통합을 목표로 합니다. 기존의 텍스트 데이터는 매우 제한적이었지만, LLMs는 풍부한 텍스트 지식을 보유하고 있어 이를 효과적으로 활용하는 방법을 모색합니다.

- **Technical Details**: POI-Enhancer는 세 가지 특수한 프롬프트를 사용하여 LLMs로부터 POI 관련 지식을 효과적으로 추출합니다. 이후, Dual Feature Alignment 모듈이 추출된 정보의 품질을 향상시키고, Semantic Feature Fusion 모듈이 정보의 무결성을 유지합니다. 마지막으로 Cross Attention Fusion 모듈이 이러한 고품질 정보를 POI 표현으로 완전히 통합하며, Multi-View Contrastive Learning을 통해 인간이 이해할 수 있는 의미 정보를 더합니다.

- **Performance Highlights**: 세 개의 실제 데이터셋을 사용한 실험 결과, POI-Enhancer는 기존의 POI 표현 방법에 비해 성능을 유의미하게 향상시키는 것으로 나타났습니다. 본 연구는 POI 학습 모델의 성능을 향상시키기 위해 LLMs에서의 지식 활용이 얼마나 중요한지를 보여줍니다. POI 표현을 개선하기 위한 기계 학습 접근 방식의 가능성을 넓히는 성과를 달성하였습니다.



### Dream to Drive: Model-Based Vehicle Control Using Analytic World Models (https://arxiv.org/abs/2502.10012)
- **What's New**: 최근, 차별화 가능한 시뮬레이터(differentiable simulators)가 자율주행차 컨트롤러를 훈련하는 데 큰 가능성을 보여주었다. 본 연구에서는 이 시뮬레이터를 정책(policy) 훈련에만 적용했던 기존 방식을 넘어서, 세계 모델(world models) 훈련에도 활용함으로써 새로운 접근 방식을 제시했다. 특히, 이 연구에서는 세 가지 새로운 작업(task) 설정을 소개하여 다음 상태 예측기(next state predictors), 최적 계획(optimal planners), 최적 역상태(optimal inverse states)를 학습할 수 있도록 하였다.

- **Technical Details**: 특히, 제안된 방법인 Analytic World Models (AWMs)는 현재 상태에 대한 다음 상태의 기울기(gradient)를 필요로 하며, 이는 분석 정책 기울기(Analytic Policy Gradients, APG)와는 다른 점이다. 기울기를 통해 정책을 최적화할 수 있으며, 이는 샘플 효율성을 개선하며 행동의 해석 가능성을 높인다. 또한 이 연구에서는 Waymax 시뮬레이터를 사용하여 계획적(more interpretable and meaningful)인 예측을 할 수 있도록 하였다.

- **Performance Highlights**: 제안된 훈련 방법을 통해, 기존 방법들과 비교했을 때 대규모 Waymo Open Motion 데이터셋에서 최대 12%의 성능 향상이 나타났다. 이 개선은 훈련 과정을 짧게 하여 과적합(overfitting)을 줄이고, 정책의 엔트로피(entropy)를 장려하는 정규화 항을 추가하여 달성했다. 이로써 높은 수준의 훈련 효율성과 효과적인 계획을 통한 자율주행을 위한 강력하고 정확한 접근 법을 제공하게 되었다.



### Decision Information Meets Large Language Models: The Future of Explainable Operations Research (https://arxiv.org/abs/2502.09994)
- **What's New**: 이 논문의 핵심은 Explainable Operations Research (EOR)라는 포괄적인 프레임워크를 제안하는 것입니다. EOR은 최적화와 함께 실행 가능하고 이해 가능한 설명을 강조합니다. 특히, 이 프레임워크에서 'Decision Information' 개념을 도입하여 복잡한 제약 조건 변경이 의사 결정에 미치는 영향을 평가합니다. 이를 통해 OR 모델의 투명성과 신뢰성을 향상시키고자 합니다.

- **Technical Details**: EOR 프레임워크는 bipartite graphs를 활용해 사용자 쿼리에 대한 의사 결정 요인의 중요성을 측정합니다. LLMs와 결합하여 복잡한 what-if 분석의 설명 능력을 강화하며, 코드 업데이트와 결과에 대한 설명을 두 가지 유형으로 제공합니다. 새로운 산업 벤치마크도 개발되어 OR에서 설명의 효과성을 rigorously 평가할 수 있도록 합니다.

- **Performance Highlights**: 이 연구는 기존 최적화 문제의 이해를 돕기 위해 수학적 구조를 제공하고, 사용자 쿼리에 따라 실시간으로 해결책에 대한 포괄적인 설명을 생성하는 것을 목표로 합니다. EOR은 기존 연구의 한계를 극복하여 더 깊이 있는 분석을 제공하며, 신뢰성 있는 의사 결정 과정을 지원합니다. 이렇게 설정된 새로운 기준은 OR 분야의 투명성 및 명료성을 높이는데 기여할 것입니다.



### Has My System Prompt Been Used? Large Language Model Prompt Membership Inferenc (https://arxiv.org/abs/2502.09974)
- **What's New**: 이 연구에서는 Prompt Detective라는 새로운 통계적 방법을 개발하여 특정 시스템 프롬프트가 제3자 언어 모델에서 사용되었는지를 신뢰성 있게 확인하는 방법을 제시합니다. 이 방법은 사용자의 쿼리 접근을 전제로 하며, 다양한 언어 모델에 대한 광범위한 실험을 통해 Prompt Detective의 효과성을 입증하였습니다. 특히 시스템 프롬프트의 사소한 변화가 LLM의 응답 분포에 명확하게 반영된다는 점을 보여 주었습니다.

- **Technical Details**: Prompt Detective는 두 개의 모델 출력 그룹의 분포를 비교하는 통계적 테스트를 기반으로 하여 LLM의 응답 결과를 분석합니다. 연구팀은 Llama, Mistral, Claude, GPT와 같은 여러 언어 모델에서 이 메소드의 유효성을 평가하였으며, 유사한 시스템 프롬프트를 구별하는 도전적인 상황에서도 이 메소드가 잘 작동함을 증명하였습니다. 이 방법은 특히 시스템 프롬프트의 사용 여부를 통계적으로 의미 있게 확인할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 시스템 프롬프트에 대한 미세한 변경 사항도 LLM의 응답에 뚜렷한 차이를 초래하여, Prompt Detective가 프롬프트 사용 여부를 통계적으로 검증할 수 있음을 확인했습니다. 이 점은 프롬프트 엔지니어링의 중요성을 강조하며, 향후 LLM의 응답 질을 향상시키기 위한 기초 자료로 활용될 수 있습니다.



### Diverse Inference and Verification for Advanced Reasoning (https://arxiv.org/abs/2502.09955)
Comments:
          165 pages. arXiv admin note: text overlap with arXiv:2001.04383 by other authors

- **What's New**: 이번 연구에서는 OpenAI의 o1, o3와 DeepSeek R1과 같은 Reasoning LLMs(대규모 언어 모델)의 수학 및 코딩 관련 문제 해결 능력에 대해 다루고 있습니다. 기존의 모델들이 International Mathematical Olympiad(IMO)와 같은 어려운 문제에서 한계를 보이는 가운데, 여러 모델과 방법을 결합한 새로운 추론 접근 방식을 제안합니다. 이 접근법은 복잡한 문제를 해결하는 데 있어 효과적이고 간단함을 강조합니다.

- **Technical Details**: 연구자들은 IMO 문제에 대한 해결책을 Lean을 사용하여 자동으로 검증합니다. 또한 코드 기반으로 ARC 퍼즐의 정답을 검증하며, HLE 질문에 대해서는 best-of-N 방법을 적용하여 정확도를 높입니다. 실험 결과, IMO 조합론 문제의 정답 정확도가 33.3%에서 77.8%로 상승하고, HLE 질문의 정확도도 8%에서 37%로 향상되는 것을 확인했습니다.

- **Performance Highlights**: 이 연구의 접근법은 948명의 인간이 해결하지 못한 80%의 ARC 퍼즐과, o3 고성능 모델이 해결하지 못한 26.5%의 ARC 퍼즐을 해결합니다. 또한, 테스트 시 시뮬레이션, 강화 학습, 메타 학습을 사용하여 추론 피드백을 통해 모델의 일반화 능력을 개선하며, agent graph representations를 적응시키고 다양한 프롬프트, 코드 및 데이터 세트를 변형하는 방식으로 이룬 성과입니다.



### Analyzing Patient Daily Movement Behavior Dynamics Using Two-Stage Encoding Mod (https://arxiv.org/abs/2502.09947)
Comments:
          NeurIPS 2024 workshop Time Series in the Age of Large Models. arXiv admin note: substantial text overlap with arXiv:2502.09173

- **What's New**: 이번 연구에서는 치매 환자의 가정 활동 데이터를 활용한 시간 시계열 데이터의 표현 학습에 초점을 맞췄습니다. 우리는 두 단계의 자기 지도 학습 접근 방식을 제안하여, 시간 시계열의 활동을 텍스트 문자열로 변환하고 이를 디지털 언어 모델로 인코딩하였습니다. 이후 PageRank 방법을 적용하여 참가자의 행동 패턴을 정량적으로 평가하는 새로운 방법론을 소개합니다.

- **Technical Details**: 연구에는 비지도 및 반지도 학습 기법을 활용하여 대규모의 라벨 없는 시간 시계열 데이터를 효과적으로 분석하는 방법이 포함되어 있습니다. 첫 번째 단계에서는 각 데이터 샘플을 텍스트 형식으로 변환한 후, 고차원 벡터 표현으로 변환하기 위해 사전 훈련된 언어 모델을 적용합니다. 두 번째 단계에서는 t-SNE와 PageRank 알고리즘을 통해 2차원 공간에서 데이터의 클러스터링을 수행하고, 잠재 상태를 분석합니다.

- **Performance Highlights**: 우리는 134명의 치매 환자로부터 수집한 데이터를 활용하여 연구를 수행했습니다. 이를 통해 노이즈 제거 및 데이터 일관성을 확보한 후, 50명의 참가자를 대상으로 한 분석 결과를 도출하였습니다. 개발된 분석 프레임워크는 임상 진단과 개인화된 치료 프로그램 개발을 지원하기 위한 중요한 통찰력을 제공할 것으로 기대됩니다.



### MIR-Bench: Benchmarking LLM's Long-Context Intelligence via Many-Shot In-Context Inductive Reasoning (https://arxiv.org/abs/2502.09933)
Comments:
          32 pages, 11 figures

- **What's New**: 본 논문에서는 기존의 few-shot In-Context Learning (ICL) 접근 방식을 넘어, 다양한 정보를 한 번에 다루는 many-shot inductive reasoning benchmark인 MIR-Bench를 제안합니다. 이를 통해 LLM(Large Language Models)이 복잡한 문제 해결에 있어 필요한 변형된 유출 없이 새로운 정보를 효율적으로 처리할 수 있도록 합니다. MIR-Bench는 입력-출력 예시를 통해 LLM이 예측하는 방식으로 구성되어 있으며, 이로 인해 보다 복잡한 inductive reasoning 문제를 다룰 수 있는 기회를 제공합니다.

- **Technical Details**: MIR-Bench는 다양한 입력-출력 데이터 형식을 통해 수백~수천 개의 예시를 제공하며, 이는 LLM이 새로운 입력에 대한 출력을 예측하도록 유도합니다. 이 모델은 크게 MIR-Core와 MIR-Extended의 두 가지 문제 세트를 포함하며, 각각 3,000과 6,930개의 문제를 담고 있습니다. 이러한 구조는 LLM의 장기 컨텍스트 인지 능력을 평가할 수 있는 다양한 문제를 제공합니다.

- **Performance Highlights**: MIR-Bench를 통해 여러 최첨단 LLM의 성능을 비교한 결과, 모델 성능이 크게 차이가 나며, 어느 모델도 벤치마크에서 포화되지 않았습니다. 또한, 연구팀은 기존 연구에서 간과된 여러 중요한 문제들에 대한 실증 연구를 수행하여 LLM의 many-shot 및 long-context 인지 능력에 대한 통찰을 얻었습니다. 이 결과는 LLM의 inductive intelligence가 잘못된 입력-출력 쌍에 얼마나 강한지를 평가하는 데 중요한 기여를 했습니다.



### AutoS$^2$earch: Unlocking the Reasoning Potential of Large Models for Web-based Source Search (https://arxiv.org/abs/2502.09913)
- **What's New**: 이 논문에서는 AutoS$^2$earch라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 웹 기반 위험 관리 시스템에서 소스 검색(source search) 기능을 효과적으로 통합하여, 인간의 개입 없이도 위험을 판단할 수 있도록 돕습니다. AutoS$^2$earch는 대규모 모델을 활용하여 비주얼과 언어적 추론을 결합하고, 인간과 유사한 사고 과정을 구현합니다.

- **Technical Details**: AutoS$^2$earch는 간소화된 시각적 환경을 웹으로 구현하여, 체계적인 사고(chain-of-thought) 프롬프트를 통해 인간의 사고를 모방합니다. 이 시스템은 여러 방향 선택에 대한 언어적 설명을 생성하는 멀티모달 대형 언어 모델(MLLM)로 움직이며, 이러한 설명에 기반하여 최적의 방향을 결정합니다. 시스템은 Flask 프레임워크를 사용하여 실시간 소통을 지원하며, 동적 맵 업데이트와 렌더링 기능이 포함되어 있습니다.

- **Performance Highlights**: AutoS$^2$earch는 20개의 벤치마크 시나리오에서 95-98%의 성공률을 달성하며, 인간-AI 협력 소스 검색과 유사한 성능을 보입니다. 이는 기존의 군중 기반 방식에서 필요한 인적 자원 의존도를 줄이며, 더욱 효율적인 자동화 시스템을 설계할 수 있는 가능성을 제시합니다. 따라서 인간의 역할은 실행자에서 검증자 또는 감독자로 변모할 수 있을 것으로 보입니다.



### The Ann Arbor Architecture for Agent-Oriented Programming (https://arxiv.org/abs/2502.09903)
- **What's New**: 이 논문에서는 대형 언어 모델에 대한 프롬프트 엔지니어링을 오토마타 이론의 관점에서 재검토합니다. 연구자들은 언어 모델이 오토마타로 작용하며, 따라서 모든 자연어 및 형식적 언어의 집합에서 수용된 언어로 프로그래밍되어야 한다고 주장합니다. 이를 위해, 저자들은 앤아버 아키텍처(Ann Arbor Architecture)를 도입하여 언어 모델의 에이전트 지향 프로그래밍을 위한 개념적 프레임워크를 제안합니다.

- **Technical Details**: 앤아버 아키텍처의 핵심은 에이전트와 언어 모델 간의 상호작용을 이메일 시스템을 기반으로 모델링한다는 것입니다. 각 에이전트는 고유한 식별자인 이메일 주소를 부여받아 메시지를 교환하며, 이 메시지는 에이전트의 통신 이력을 형성하는 일종의 일지 역할을 합니다. 저자들은 이러한 방법론이 언어 모델의 자연어와 형식 언어의 통합을 목표로 하고 있음을 강조합니다.

- **Performance Highlights**: 에이전트 플랫폼으로는 Postline을 소개하며 초기 실험 결과를 보고합니다. 이 초안의 성공적인 적용은 에이전트 간의 정보 교환 및 갈등 해결을 지원할 수 있는 유연한 구조를 필요로 합니다. 논문에서는 언어 모델 프로그래밍의 전반적인 잠재력을 탐구하며, 새로운 방식의 컴퓨터 언어 접근이 에이전트를 통한 AI 응용 프로그램의 성공에 중대한 기여를 할 것으로 기대됩니다.



### Artificial Intelligence in Spectroscopy: Advancing Chemistry from Prediction to Generation and Beyond (https://arxiv.org/abs/2502.09897)
- **What's New**: 이 논문은 Spectroscopy Machine Learning(SpectraML)의 최신 발전을 다루면서 화학 분야에서의 머신러닝(ML) 및 인공지능(AI) 응용의 역사를 설명합니다. 기존의 연구들이 단일 기법에 집중된 것과 달리, 이 연구에서는 다섯 가지 주된 분광 기술(Mass Spectrometry, NMR, IR, Raman, UV-Vis)을 하나의 방법론적 프레임워크 안에서 통합했습니다. 또한, 고급 모델들이 복잡한 작업, 예를 들어 분자 구조 설명과 반응 경로 예측을 위해 새로운 방향으로 진화하고 있다는 점을 강조합니다.

- **Technical Details**: 분광학은 물질과 전자기 복사의 상호작용을 연구하는 분야로, MS, NMR, IR, Raman, UV-Vis와 같은 다양한 기법을 사용합니다. 각 기법은 분자의 구조와 동역학에 대한 고감도 데이터를 제공하는데, 이는 대량의 고차원 데이터를 생성하여 전통적인 전문가 기반의 분석 방식으로는 처리하기 어려운 복잡성을 가집니다. 이 논문에서는 Forward Problem(구조에서 스펙트럼 예측)과 Inverse Problem(스펙트럼에서 구조 추론)의 정의 및 이를 해결하기 위한 최신 ML 기법들을 논의합니다.

- **Performance Highlights**: 이 연구에서 다룬 ML 기술은 스펙트럼 분석의 자동화를 촉진하여 작업 속도를 높이는 동시에 데이터의 정확성을 증가시킵니다. 특히, 딥러닝 모델(CNNs, RNNs)은 피크 탐지, 분해 및 반응 모니터링과 같은 작업에서 효과적임이 입증되었습니다. 또한, 오픈소스 리포지토리에서 데이터세트와 코드 공유를 통해 연구자들이 손쉽게 접근하고 재현 가능한 연구를 수행할 수 있도록 지원합니다.



### A Scoresheet for Explainable AI (https://arxiv.org/abs/2502.09861)
Comments:
          To appear at AAMAS 2025 - arXiv version also includes appendices

- **What's New**: 이 논문은 자율 및 지능 시스템의 설명 가능성(explainability)에 대한 새로운 접근 방식을 제안하며, 구체적으로 설명 가능성 요구사항을 정의하거나 특정 응용 프로그램에서 제공되는 설명 가능성 측면을 평가하기 위해 사용할 수 있는 점수 시트(scoresheet)를 개발합니다. 현재의 표준들은 지나치게 고급 수준으로 설명 가능성에 대한 충분한 요구 사항을 정의하지 못하고 있다고 주장합니다. 이 점수 시트는 다양한 이해관계자(stakeholders)의 요구 사항을 고려하여 개발되었으며, 다중 에이전트 시스템(multiagent systems) 및 기타 AI 기술에 적용 가능합니다.

- **Technical Details**: 점수 시트는 LEGO의 건축 모델처럼 다양한 구성 요소를 통해 설명 가능성을 평가할 수 있습니다. 각 이해관계자 그룹의 요구 사항을 반영하여, 사용자에게 제공하는 정보의 투명성(transparency) 및 각기 다른 질문들(예: 왜 그렇게 했는가?라는 질문에 대한 답)을 통해 설명 가능성을 논의합니다. 이를 통해 설명 가능성을 평가하는 방법과 지침을 제공하며, 점수 시트의 활용 사례(기계 학습 시스템과 관련된 적용 예시 등)를 통해 그 일반성과 유용성을 입증합니다.

- **Performance Highlights**: 이 연구는 다양한 시스템에 점수 시트를 적용하여 실제로 설명 가능성을 평가할 수 있는 유용한 도구를 제시합니다. 연구 결과, 점수 시트를 사용한 시스템 평가에서 설명 가능성의 여러 요소들이 높은 중요한 역할을 가짐을 보여주었습니다. 다양한 이해관계자들의 필요에 따라 맞춤형 설명 제공 방식이 필요하다는 점을 강조하며, 이는 자율 및 지능 시스템이 사회적으로 수용되는 데 중요한 기여를 할 것입니다.



### MuDoC: An Interactive Multimodal Document-grounded Conversational AI System (https://arxiv.org/abs/2502.09843)
Comments:
          5 pages, 3 figures, AAAI-MAKE 2025

- **What's New**: MuDoC는 문서에 기초한 멀티모달 대화형 AI 시스템으로, 어린이 교육 및 기술 학습에 있어 중요한 발전을 이룹니다. 이 시스템은 텍스트와 이미지가 상호 섞인 형태로 응답을 생성하여 사용자가 더욱 직관적으로 정보를 이해할 수 있게 돕습니다. 또한, MuDoC는 사용자가 응답의 출처를 즉시 확인할 수 있도록 문서 내 보기를 지원하여 신뢰성을 높입니다.

- **Technical Details**: MuDoC는 GPT-4o 모델을 사용하여 쿼리와 대화의 맥락에 따라 텍스트와 이미지를 검색하고, 이를 기반으로 멀티모달 응답을 제공합니다. 시스템은 PDF 문서를 전처리하여 텍스트 및 이미지를 자동으로 추출하고, 이를 검색 가능한 임베딩 형태로 저장합니다. 문서 레이아웃 분석(Document Layout Analysis) 기술을 활용하여 텍스트 블록과 이미지 조각을 정확히 식별하며, 각 요소의 정확도를 높이고 오류를 수정합니다.

- **Performance Highlights**: MuDoC는 텍스트와 이미지의 친밀한 통합을 통해 복잡한 내용을 쉽고 간결하게 전달하는 데 특히 효과적입니다. 사용자가 이미지에 쉽게 접근할 수 있도록 하여 대화의 흐름을 방해하지 않으면서도 늘어나는 학습 요구에 응답합니다. 초기의 질적 관찰 결과는 MuDoC가 다양한 문서에서 정보를 효율적으로 추출하고 사용자 요구에 맞춘 유용한 정보를 제공하는 기능을 보여줍니다.



### Imit Diff: Semantics Guided Diffusion Transformer with Dual Resolution Fusion for Imitation Learning (https://arxiv.org/abs/2502.09649)
- **What's New**: 이번 논문에서는 Imit Diff라는 새로운 세맨틱 가이드 확산 변환기(diffusion transformer)를 소개합니다. 이 접근법은 고차원 시각 정보와 로봇의 고유 감각(proprioception)을 활용하여 비디오 시연으로부터 조작 기술을 효과적으로 학습하는 데 기여합니다. Imit Diff는 복잡한 장면과 시각적 방해 요소가 증가하는 상황에서도 뛰어난 성능을 발휘하는 것으로 나타났습니다.

- **Technical Details**: Imit Diff는 세 가지 주요 구성 요소로 구성됩니다: 첫째, 세맷틱 인젝션(semantics injection)을 통해 고차원 시각 데이터에서 픽셀 수준의 시각적 위치 지정 정보를 추출합니다. 둘째, 이 정보는 듀얼 해상도(fusion) 비전 인코더에서 추출한 다중 스케일 비주얼 특징과 접목시킵니다. 셋째, 확산 변환기 아키텍처 내에서 일관성 정책(consistency policy)을 구현하여 실시간 성능과 움직임의 부드러움을 향상시킵니다.

- **Performance Highlights**: Imit Diff는 여러 실세계 작업에서 평가된 결과, 특히 시각적 방해가 있는 복잡한 장면에서 기존의 최신 방법들보다 상당히 우수한 성능을 보였습니다. 또한, 제로샷(experiments) 실험을 통해 시각적 방해 및 범주 일반화에서 이 모델의 장점을 입증했습니다. 코드 또한 곧 공개될 예정입니다.



### Efficient and Trustworthy Block Propagation for Blockchain-enabled Mobile Embodied AI Networks: A Graph Resfusion Approach (https://arxiv.org/abs/2502.09624)
Comments:
          15 pages, 11 figures

- **What's New**: 본 논문은 이동형 인공지능 네트워크(MEANETs)와 블록체인 기술을 통합하여 신뢰할 수 있는 블록 전파 최적화 프레임워크를 제안합니다. 특히, 블록 전파 효율성과 신뢰성을 동시에 고려하는 새로운 접근 방식을 도입하여, 동적 환경에서의 성능을 개선하는 데 중점을 두고 있습니다. 이 방법은 일반적인 블록 프로파게이션 방식에서의 낮은 전파 효율성과 보안 취약점을 보완하며, 실시간 변화에 대응할 수 있는 능력을 강조합니다.

- **Technical Details**: 논문에서는 트러스트 클라우드 모델을 기반으로 한 혁신적인 신뢰 계산 메커니즘을 제안하여, 채굴자 신뢰도를 종합적으로 평가합니다. 또한, 그래프 신경망(graph neural networks)과 확산 모델(diffusion models)을 활용하여 최적의 블록 전파 경로를 효과적으로 생성하는 그래프 리스퓨전 모델(graph Resfusion model)을 개발하였습니다. 이 프레임워크는 동적 네트워크 구조 내의 공간적 관계를 포착하여 신뢰할 수 있는 블록 전파를 가능하게 합니다.

- **Performance Highlights**: 시뮬레이션 결과, 제안된 모델은 다른 라우팅 메커니즘에 비해 블록 전파의 효율성 및 신뢰성 면에서 우수한 성능을 보였습니다. 이 모델은 신뢰성을 유지하면서도 동적 환경에 대한 강력한 적응성을 보여줍니다. 또한, 농업 로봇과 같은 다양한 MEANET 응용 프로그램에서 이 시스템의 활용 가능성을 높이며, 블록 체인 기반 MEANET의 확장성과 안정성을 지원합니다.



### Region-Adaptive Sampling for Diffusion Transformers (https://arxiv.org/abs/2502.10389)
- **What's New**: 이번 연구에서는 새롭게 RAS(Region-Adaptive Sampling)라는 샘플링 전략을 소개합니다. RAS는 Diffusion Transformers(DiTs)의 유연성을 활용하여 이미지 내의 서로 다른 영역에 맞춤형 샘플링 비율을 동적으로 할당하며, 실시간 성능을 향상시킵니다. 이전 연구들이 샘플링 단계의 수를 줄이거나 중간 결과를 재사용하는 데 초점을 맞춘 것과 달리, RAS는 모델의 집중 영역에 기반해 처리할 영역만 업데이트하며 나머지 영역은 캐시된 노이즈를 사용합니다.

- **Technical Details**: RAS는 각 샘플링 단계에서 모델이 집중하는 의미적으로 중요한 영역을 파악하고, 이들 지역을 업데이트합니다. 모델의 초점이 전 단계의 출력에 따라 결정되며, 이를 통해 샘플링 과정에서 지역적 변동성을 허용합니다. 또한, 빠른 업데이트 지역과 느린 업데이트 지역을 구분하여 처리하며, 이러한 프로세스는 computational overhead를 줄이는 데 기여합니다.

- **Performance Highlights**: RAS의 효율성은 Stable Diffusion 3 및 Lumina-Next-T2I 모델에서 최대 2.36배 및 2.51배의 속도 향상을 달성하면서도 생성 품질 저하가 미미함을 보여줍니다. 사용자 연구에 따르면, RAS는 1.6배의 가속도를 제공하며 인간 평가에서도 유사한 품질을 유지합니다. 이 방법은 실시간 응용 프로그램에서 Diffusion Transformers의 가능성을 크게 확장하는 중요한 진전을 의미합니다.



### Simplifying DINO via Coding Rate Regularization (https://arxiv.org/abs/2502.10385)
Comments:
          17 pages, 5 figures

- **What's New**: 최근 DINO와 DINOv2는 대규모의 비지도(unsupervised) 이미지 데이터를 학습하기 위한 모델 계열로 널리 사용되고 있습니다. 본 연구에서는 복잡하고 불안정한 훈련 프로세스를 간소화하고, representation이 무너지는 것을 방지하기 위해 손실 함수(loss function)에 명시적인 부호화 속도(coding rate) 항을 추가하는 접근을 제안합니다. 이로 인해 DINO와 DINOv2의 간소화된 변형인 SimDINO와 SimDINOv2를 개발하였습니다.

- **Technical Details**: 새롭게 제안된 SimDINO와 SimDINOv2는 기존의 학습 방식에서 과도한 시행착오를 줄이며, 하이퍼파라미터(hyperparameters) 조정에 대한 민감성을 크게 완화합니다. 이 모델들은 다양한 네트워크 아키텍처 및 디자인 선택에 더 강인하며, 성능 저하 없이 representation을 학습할 수 있습니다. 이러한 접근은 훈련 방법의 직관성을 높이며, 복잡한 설계를 단순화하여 실증적으로 더 나은 성능을 보여줍니다.

- **Performance Highlights**: SimDINO와 SimDINOv2는 다운스트림 작업(downstream tasks)에서 더 높은 품질의 representation을 학습하며, DINO 및 DINOv2 모델들보다 성능이 개선된 Pareto 개선을 제공합니다. 학습된 representation은 이미지 분류(image classification)와 세분화(segmentation)와 같은 다양한 응용 프로그램에서 최상급(performance) 결과를 이끌어낼 수 있습니다. 따라서, 이 연구는 심화 학습(deep learning)에서의 단순화된 설계 원칙을 활용한 성능 향상의 가능성을 보여줍니다.



### OWLS: Scaling Laws for Multilingual Speech Recognition and Translation Models (https://arxiv.org/abs/2502.10373)
Comments:
          23 pages, 13 figures

- **What's New**: 이 논문은 OWLS(Open Whisper-style Large-scale neural model Suite)를 소개하며, 이는 0.25B부터 18B까지의 매개변수를 가진 다국어 음성 인식 및 번역 모델의 오픈 액세스 수트입니다. OWLS는 150개 언어에서 360K 시간을 넘어선 공공 음성 데이터를 활용하여 다국어 음성 작업에서 데이터, 모델 및 계산 스케일링이 성능에 미치는 영향을 체계적으로 조사합니다. 이 연구의 주요 발견 중 하나는 스케일링이 낮은 자원 언어와 방언의 성능을 향상시켜 편향을 완화하고 음성 기술의 접근성을 개선한다는 것입니다.

- **Technical Details**: OWLS는 총 13개의 투명한 음성 기반 모델을 포함하고 있으며, 각각의 모델은 0.25B에서 18B의 매개변수를 가지고 있습니다. 이 모델들은 360K 시간의 다국어 데이터에 대해 사전 훈련되었으며, ASR과 ST의 성능을 분석하기 위해 모델 및 데이터 크기의 스케일링 영향을 체계적으로 평가합니다. 이 연구를 통해 우리는 각 작업과 언어에 대한 모델 성능의 변화를 예측할 수 있는 신경 스케일링 법칙을 도출했습니다.

- **Performance Highlights**: OWLS는 18B의 총 매개변수를 가진 ASR/ST 모델을 훈련 및 출시했고, 이는 현재까지 알려진 공개 ASR/ST 모델 중 가장 큽니다. 모델 스케일링의 유용성을 측정하는 것뿐 아니라 그것이 극복하지 못하는 실패 사례도 식별하였습니다. 또한 대규모 음성 기초 모델의 시험 시간에서의 능력을 평가하고, 큰 모델에만 존재하는 새로운 emergent abilities를 발견하여 음성 모델 스케일링이 새로운 언어에 자원 활용 학습에 어떻게 기여하는지를 보여주었습니다.



### BeamDojo: Learning Agile Humanoid Locomotion on Sparse Footholds (https://arxiv.org/abs/2502.10363)
Comments:
          Project website: this https URL

- **What's New**: 새롭게 소개된 BeamDojo는 위험한 지역에서 인간형 로봇의 민첩한 보행을 가능하게 하는 강화 학습 기반 프레임워크입니다. 이 시스템은 다각형 발 모델을 위해 설계된 샘플링 기반 발 보상과 이중 비평가(double critic)를 도입하여 보행 보상과 드문 발 보상 사이의 학습 균형을 조정합니다. 또한, 두 단계의 RL 접근 방식을 사용하여 로봇이 평탄한 지형에서 연습하고 최종 작업 지형에서 정책을 미세 조정할 수 있도록 지원합니다.

- **Technical Details**: BeamDojo는 드문 발 보상의 학습 문제를 해결하기 위해 두 가지 주요 구성 요소를 포함합니다. 먼저, 발 모델에 적합한 새로운 보상 체계를 정의하고, 두 번째로 이중 비평가 구조를 통해 밀집 보상과 드문 보상의 학습을 분리합니다. 이러한 설계는 로봇의 보행 성능을 높이는 데 도움이 되며, LiDAR 기반의 고도 맵을 통해 실제 환경에서도 효과적으로 작동할 수 있도록 구현되었습니다.

- **Performance Highlights**: BeamDojo는 시뮬레이션 및 실제 실험을 통해 효율적인 학습 과정을 보여줍니다. Unitree G1 인간형 로봇을 사용한 실험에서 BeamDojo는 드문 발 고정에서 민첩한 보행을 달성하며 성공률은 80%를 기록했습니다. 이러한 성능은 복잡한 상황에서도 안정적인 보행을 가능하게 하는 중요한 성과로, 위험한 지역에서의 보행 제어 문제를 해결하는 데 기여합니다.



### STAR: Spectral Truncation and Rescale for Model Merging (https://arxiv.org/abs/2502.10339)
Comments:
          Accepted to NAACL 2025

- **What's New**: 모델 머징(model merging)은 여러 pretrained 모델을 사용하여 추가적으로 fine-tuning 없이 멀티태스킹 모델을 생성하는 효율적인 방법입니다. 본 논문에서는 ‘S$	ext{pectral} T	ext{runcation} A	ext{nd} R	ext{escale}’ (STAR)이라는 새로운 방법을 제안하여, 증가하는 모델 수로 인해 발생하는 작업 성능 저하를 완화하고자 합니다. STAR는 각 스펙트럼 공간에서 작은 구성 요소를 제거한 후, 원본 행렬의 핵 노름(nuclear norm)을 유지하기 위해 자동으로 매개변수를 재조정하여, 원본 훈련 데이터에 대한 추가적인 추론을 필요로 하지 않으며 Hyperparameter 선택에 강건합니다.

- **Technical Details**: STAR는 스펙트럼 분해(spectral decomposition)와 같은 기법을 사용하여 모델 머징에서 노이즈 성분을 제거하고, 이후 재조정(rescaling) 단계를 통해 원래의 핵 노름을 복원합니다. 이는 여러 모델과 다양한 작업의 경우에도 잘 작동하며, 20개 모델까지 성공적으로 머징할 수 있습니다. 또한, STAR는 기존의 방법론들과는 달리, 데이터 접근 권한이 없는 상황에서도 유용하게 사용될 수 있습니다.

- **Performance Highlights**: STAR는 다양한 모델 크기 설정에서 우수한 성능을 입증하였으며, 특정 NLP 작업인 Flan-T5의 경우 12개 모델을 머징할 때 기존 기준선 대비 4.2% 향상된 성능을 보여주었습니다. 이 같은 성과는 필요한 추가 추론이 없고 파라미터에 대한 민감도가 낮기 때문에 이루어진 것입니다. 연구자들은 STAR의 공개 코드를 통해 다양한 NLP 작업에서의 적용 가능성을 확대할 수 있을 것입니다.



### Evaluating the Meta- and Object-Level Reasoning of Large Language Models for Question Answering (https://arxiv.org/abs/2502.10338)
Comments:
          8 pages. Accepted to the Workshop on Planning in the Era of LLMs (LM4Plan @ AAAI 2025)

- **What's New**: 이 논문은 대규모 언어 모델(LLMs)이 자연어 작업에서 우수한 성능을 보이나, 복잡한 다단계 사고가 요구되는 질문 응답(QA) 작업에서 도전과제를 안고 있음을 강조합니다. 기존 LLM이 처하는 이유를 재정의 하여 meta-level reasoning(메타 수준 사고)과 object-level reasoning(객체 수준 사고)으로 나누었습니다. 새로운 데이터셋인 Franklin이 도입되어 LLM의 질문 응답 성능을 평가하는 데 사용되었습니다.

- **Technical Details**: 이 논문에서는 LLM의 reasoning(사고) 작업을 구체적으로 논의하며, multi-step reasoning(다단계 사고)의 중요성을 보여줍니다. 연구는 meta- and object-level reasoning을 평가할 수 있는 Franklin 데이터셋을 포함한 세 가지 다른 데이터셋을 사용하여 진행되었습니다. 실험 결과, LLM은 meta-level reasoning을 자주 보여주지만, object-level reasoning 작업에서는 어려움을 겪고 있는 것으로 나타났습니다.

- **Performance Highlights**: 대상 데이터셋에서 LLM들의 성능이 함께 비교되었습니다. 연구 결과, 대다수 LLM은 object-level reasoning이 부족하여 어려움을 겪었지만, meta-level reasoning에서는 일관되게 높은 성능을 보였습니다. 또한 Franklin 데이터셋은 LLM에게 도전 과제를 제공하여, 이들 모델의 강점과 약점을 체계적으로 분석하는 기회를 마련했습니다.



### Process Reward Models for LLM Agents: Practical Framework and Directions (https://arxiv.org/abs/2502.10325)
Comments:
          17 pages, 7 figures

- **What's New**: 이번 논문에서는 Agent Process Reward Models (AgentPRM)라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 LLM 에이전트가 상호작용을 통해 지속적으로 개선될 수 있도록 돕습니다. AgentPRM은 경량화된 actor-critic 패러다임을 따르며, Monte Carlo rollouts를 사용하여 보상 목표를 계산하고 정책을 최적화합니다. 기존의 RLHF pipeline에 최소한의 수정으로 통합이 가능하여 대규모로 사용하기 쉬운 특징을 가지고 있습니다.

- **Technical Details**: AgentPRM은 자동 PRM 주석(annotation)과 반복 학습(iterative training)의 두 가지 주요 측면을 가집니다. 자동 PRM 주석은 비동기 Monte Carlo rollouts를 사용하여 수동으로 레이블이 달린 보상 없이 에이전트가 학습할 수 있도록 합니다. 반복 학습 과정에서는 PRM과 정책이 서로를 개선하며 공동으로 훈련됩니다. 또한 새로운 InversePRM을 통해 명시적인 결과 감독 없이 시연으로부터 직접 프로세스 보상을 학습하는 방법도 제안합니다.

- **Performance Highlights**: ALFWorld 벤치마크에서 평가한 결과, AgentPRM을 통해 훈련된 소형(3B) 모델이 강력한 GPT-4o 기준선을 능가하는 성과를 보였습니다. InversePRM은 단일 반복에서 거의 전문가 수준의 성능을 달성하며, SFT보다 훨씬 더 높은 샘플 효율성을 보였습니다. 이러한 결과는 AgentPRM 및 InversePRM이 어떻게 RLHF 환경에서 효과적으로 작용할 수 있는지를 보여줍니다.



### ExplainReduce: Summarising local explanations via proxies (https://arxiv.org/abs/2502.10311)
Comments:
          22 pages with a 7 page appendix, 7 + 5 figures, 2 tables. The datasets and source code used in the paper are available at this https URL

- **What's New**: 이 논문은 ExplainReduce라는 새로운 절차를 소개하며, 이는 많은 수의 로컬 설명(local explanations)을 소수의 프로시 모델(proxy models)로 축소하여 생성적인 글로벌 설명을 제공하는 방법이다. 연구진은 로컬 설명의 불안정성을 수학적으로 탐구하고, 이를 해결하기 위해 최적화 문제로 정의함으로써 효율적으로 접근할 수 있음을 보여준다. 이로 인해 사용자는 복잡한 머신 러닝 모델의 이해도를 높일 수 있게 된다.

- **Technical Details**: ExplainReduce는 많은 수의 로컬 모델을 최대한 많은 항목을 포함할 수 있도록 특징을 최대화하면서 미니멀한 서브셋으로 줄이는 최적화 문제로 설정된다. 이 과정에서 사용되는 기법은 그리디 휴리스틱(greedy heuristics)을 이용하여 계산 효율성을 높인다. 논문에서는 로컬 모델로부터 얻은 설명이 복잡한 클로즈드 박스(closed-box) 모델의 근사치를 제공하는 방식을 논의하며, 실험 결과와 코드 역시 공유하고 있다.

- **Performance Highlights**: 제안된 프로시저에 따른 작은 프로시 모델 집합은 클로즈드 박스 모델을 효과적으로 설명하는 글로벌 설명을 제공한다. 이 연구에서는 한정된 초기 설명 집합에서조차도 안정적인 프로시 모델 집합을 찾을 수 있고, 이 결과는 비교적 적은 개수로도 충분히 높은 정확도의 예측이 가능함을 보여준다. 또한, 실험을 통해 프로시 모델 집합이 unseen data에 대해서도 기존의 설명 집합과 유사한 정확도를 유지함을 확인하였다.



### A Hybrid Cross-Stage Coordination Pre-ranking Model for Online Recommendation Systems (https://arxiv.org/abs/2502.10284)
Comments:
          Accepted by WWW 2025

- **What's New**: 이 논문에서는 대규모 추천 시스템에서의 표본 선택 편향(sample selection bias, SSB) 문제와 Matthew 효과를 개선하기 위한 새로운 하이브리드 교차 단계 조정 프리랭킹 모델(Hybrid Cross-Stage Coordination Pre-ranking model, HCCP)을 제안합니다. 기존의 경량 모델은 다운스트림 단계만을 고려하여 일관성(cross-stage consistency) 문제를 초래하고 시스템 성능을 저하시켰습니다. HCCP는 업스트림(검색) 및 다운스트림(랭킹, 리랭킹) 정보의 통합을 통해 프리랭킹의 적응성을 향상시키는 것을 목표로 합니다.

- **Technical Details**: HCCP는 두 가지 주요 구성 요소로 구성됩니다: 하이브리드 샘플 구축(Hybrid Sample Construction)과 하이브리드 목표 최적화(Hybrid Objective Optimization)입니다. 하이브리드 샘플 구축은 검색 및 프리랭킹 순서에서 다층 미노출 데이터를 포착하고, 이를 통해 프리랭킹 학습을 위한 최적의 "ground truth"를 생성합니다. 하이브리드 목표 최적화는 일관성(consistency) 및 롱테일(long-tail) 정밀도를 동시 최적화하는 새로운 Margin InfoNCE 손실(margin InfoNCE loss)을 사용합니다.

- **Performance Highlights**: 다양한 오프라인 및 온라인 실험을 통해 HCCP는 기존의 최첨단(SOTA) 방법들을 초월하는 성능을 입증했습니다. JD 전자상거래 추천 시스템에서 UCVR(사용자 클릭률 증가율)를 14.9% 및 UCTR(사용자 클릭률)를 1.3% 향상시켰습니다. HCCP는 코드 프라이버시를 고려하였으며, 참조를 위한 의사 코드(pseudocode)를 제공합니다.



### Probing Perceptual Constancy in Large Vision Language Models (https://arxiv.org/abs/2502.10273)
- **What's New**: 이번 연구에서는 33개의 Vision-Language Models (VLMs)의 지각적 지속성(perceptual constancy) 능력을 평가했습니다. 연구팀은 색상, 크기 및 형태의 지속성을 포함하여 253개의 실험을 실시하였고, 이러한 실험은 다양한 조건에서 물체 속성을 인식하는 모델의 능력을 조사하는 데 초점을 맞추었습니다. 이 연구는 VLM의 인지 능력 평가에서 지각적 지속성이 중요한 기준이 될 수 있음을 제시합니다.

- **Technical Details**: 우리의 실험은 이미지, 비디오(MOV), GIF와 같은 세 가지 유형의 지각적 입력을 사용하여 VLM의 지각적 지속성 능력을 평가했습니다. 지각적 지속성 능력은 색상, 크기 및 형태 지속성의 세 가지 주요 도메인에 초점을 두고 있으며, 각 도메인은 조명, 거리 및 시각적 각도에 따른 안정적인 물체 인식을 반영합니다. 각 모델은 제로샷 제너레이션(zero-shot generation) 작업을 기반으로 공정하게 비교되었습니다.

- **Performance Highlights**: 연구 결과, 인간과 VLM 간의 성능 차이가 상당히 크다는 것을 알 수 있었습니다. 일부 우수한 모델, 예를 들어 GPT-4v는 인간 성능에 가까운 정확도를 기록했지만, 전반적으로 VLM은 지각적 지속성을 복제하는 데 여전히 많은 제한이 존재함을 나타냈습니다. VLM은 형태 지속성(task)에서는 가장 좋은 성능을 보였고, 색상 지속성(where)에서는 가장 낮은 성능을 나타냈습니다.



### Are Large Language Models the future crowd workers of Linguistics? (https://arxiv.org/abs/2502.10266)
- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)이 경험적 언어학 연구에서 인간 참여자를 대체하거나 보완할 수 있는 가능성을 탐구합니다. 연구진은 두 가지 사례 연구를 통해 LLM의 효과를 확인하였으며, OpenAI의 GPT-4o-mini 모델을 사용하여 인간 참여자와 유사한 성과를 달성했음을 보고했습니다. 이로 인해 LLM의 성능이 인간 정보 제공자의 수준을 초과할 수 있다는 점이 강조되었습니다.

- **Technical Details**: 연구에서는 LLM의 행동을 전통적으로 인간을 위해 설계된 작업에 대해 테스트하고, 기본적인 프롬프트 엔지니어링 프레임워크를 개발하여 비전문가도 LLM의 잠재력을 탐색할 수 있도록 하였습니다. 두 개의 복제 연구에서 LLM의 적용 가능성을 명확히 하고, Chain-of-Thought (CoT) 기법과 같은 추가적인 프롬프트 기술을 탐구할 필요성이 제기되었습니다. 또한, 이 논문은 LLM이 복잡한 판단을 수행할 수 있는 가능성을 보여줍니다.

- **Performance Highlights**: 결과적으로, 첫 번째 복제 연구에서 GPT-4o-mini는 인간 참여자와 상당히 잘 맞아떨어지는 성능을 보였으며, 두 번째 과제는 LLM과 인간 간의 매칭에 대한 보다 세밀한 관점을 제공합니다. 또한, 모든 실험 조건에서 GPT-4o-mini가 인간 정보 제공자를 초과하는 성능을 나타냈습니다. 이러한 결과는 LLM을 경험적 언어학 연구에 활용하는 데 있어 연구자들에게 필요한 학제 간 접근의 길을 열어줄 것으로 기대됩니다.



### Large Language Models and Synthetic Data for Monitoring Dataset Mentions in Research Papers (https://arxiv.org/abs/2502.10263)
Comments:
          Project GitHub repository at this https URL

- **What's New**: 이번 논문은 연구 논문에서 데이터셋 언급을 자동으로 탐지하는 머신러닝 프레임워크를 제안합니다. 대규모 언어 모델(large language models, LLMs)과 합성 데이터(synthetic data)를 활용하여, 데이터셋 언급을 효율적으로 식별하고 분류할 수 있는 기술을 개발했습니다. 이 과정은 인력 자원 소모를 줄이고, 논문 전반에서 데이터셋의 가시성을 높이는 데 기여할 수 있습니다.

- **Technical Details**: 이 프레임워크는 제로샷 추출(zero-shot extraction) 및 LLM을 활용한 품질 평가(LMM-as-a-Judge)를 통해 약한 감독 가상 데이터셋(weakly supervised synthetic dataset)을 생성합니다. Phi-3.5-mini instruct 모델은 이러한 데이터셋에서 사전 파인 튜닝(pre-fine-tuning)되고, 이후 수동으로 주석이 달린 하위 집합에 대해 파인 튜닝을 진행합니다. 최종적으로는 ModernBERT 기반의 분류기가 데이터셋 언급을 효과적으로 필터링하여 계산 부담을 줄입니다.

- **Performance Highlights**: 이 연구는 수동으로 주석이 달린 샘플에서 평가하였고, 파인 튜닝된 모델이 NuExtract-v1.5와 GLiNER-large-v2.1보다 데이터셋 추출 정확도에서 우수한 성능을 보였습니다. LLM이 생성한 합성 데이터는 훈련 데이터의 부족 문제를 효과적으로 해결하고, 자원이 적은 환경에서의 일반화 성능을 개선할 수 있음을 입증했습니다. 이 프레임워크는 데이터 사용 모니터링의 확장 가능성을 제시하며, 연구자 및 정책 입안자에게 데이터 접근성을 높이는 데 도움을 줄 것입니다.



### Efficient Zero-Order Federated Finetuning of Language Models for Resource-Constrained Devices (https://arxiv.org/abs/2502.10239)
- **What's New**: 이 논문에서는 Federated Split-Perturbation Zero-order Optimization (FedSPZO)라는 새로운 방법을 제안합니다. 이 방법은 네트워크를 두 개의 블록으로 나누고 각 블록에 대해 다른 수의 perturbations를 적용하여, 연산 효율성을 높이면서 더 빠른 수렴을 달성합니다. FedSPZO는 edge 디바이스에서의 대규모 언어 모델 조정에서 데이터 프라이버시를 보장하며, 높은 메모리 요구 사항을 낮추는 것을 목표로 하고 있습니다.

- **Technical Details**: FedSPZO는 zero-order optimization을 사용하여 메모리 수요를 줄이면서 대규모 언어 모델을 fine-tune할 수 있는 가능성을 제시합니다. 이 방법은 perturbation 벡터를 생성하며, 이를 통해 연산 오버헤드를 줄이고 수렴 속도를 높입니다. seed 트릭을 채택하여 통신 오버헤드를 줄이는 동시에 정확도를 높이는 방안을 제시하였으며, 메모리 사용량은 최소 0.5%로 유지됩니다.

- **Performance Highlights**: 실험 결과, FedSPZO는 기존의 zero-order federated learning 방법 대비 총 계산량을 최대 7배 줄이는 성과를 보였습니다. 이는 더 적은 통신 라운드로 이루어지며 높은 정확도를 달성하였다는 점에서 중요한 뜻을 가지고 있습니다. 본 연구는 또한 전반적인 성능 향상을 위한 ablation study를 포함하고 있어, 제안된 기법의 효과성을 분석할 수 있도록 합니다.



### Shaping Inductive Bias in Diffusion Models through Frequency-Based Noise Contro (https://arxiv.org/abs/2502.10236)
- **What's New**: 이번 연구에서는 Diffusion Probabilistic Models (DPMs)의 훈련 및 샘플링 과정에 유도 편향(inductive biases)을 통합하여 데이터의 목표 분포(target distribution)를 더 잘 적응할 수 있도록 하는 방법을 제시합니다. 주목할 점은 주파수 기반(noise frequency-based) 노이징 연산자를 사용하여 이러한 유도 편향을 조정함으로써 모델이 특정 데이터 특성에 집중하도록 유도하는 것입니다. 이러한 접근 방식을 통해 모델의 성능을 높일 수 있을 것으로 기대됩니다.

- **Technical Details**: DPMs는 데이터 분포를 근사화하는데 효과적인 생성 모델로, 데이터가 점차 노이즈로 변형되는 과정을 통해 작동합니다. 본 연구에서는 주파수 조작을 통해 노이징 과정을 조절함으로써 특정 주파수의 정보에 더 집중할 수 있도록 합니다. 이를 위해, 'frequency diffusion'이라는 개념을 도입하여 주파수 성분을 강조하거나 감소시키는 개발된 주파수 기반 노이즈 스케줄을 활용합니다.

- **Performance Highlights**: 결과적으로, 모델은 여러 자연 데이터셋에서 주파수 기반 노이즈 스케줄을 사용했을 때 전통적인 확산 방식보다 향상된 성능을 보여줍니다. 특히, 특정 주파수에서 심각한 노이즈 손상 후에도 복잡한 분포를 회복할 수 있다는 점이 주목할 만합니다. 이를 통해 생성 모델링 분야에서의 다양한 응용 가능성을 열어주고 있습니다.



### A Multiagent Path Search Algorithm for Large-Scale Coalition Structure Generation (https://arxiv.org/abs/2502.10226)
Comments:
          Long and updated version to the published paper in the Proceedings of the 39th Annual AAAI Conference on Artificial Intelligence (AAAI 2025)

- **What's New**: 이 논문에서는 사회적 복지를 극대화하기 위해 에이전트 집합을 최적의 코얼리전(Coalition)으로 분할하는 문제인 Coalition Structure Generation (CSG)에 대해 다룹니다. SALDAE라는 새로운 다중 에이전트 경로 탐색 알고리즘을 개발하였으며, 이는 코얼리전 구조의 그래프에서 작동합니다. 이 알고리즘은 다양한 휴리스틱(Heuristics)과 전략을 활용하여 검색 과정을 안내합니다.

- **Technical Details**: SALDAE는 다중 에이전트 시스템의 대규모 문제를 처리할 수 있는 anytime 알고리즘으로서, 수백 또는 수천 개의 에이전트가 포함된 문제를 다루는 데 적합합니다. 이 알고리즘은 코얼리전 구조에 대한 그래프를 기반으로 작동하며, 다양한 전략을 통해 효율적인 검색을 수행합니다. 또한, 재해 대응과 전기차 할당 벤치마크를 포함한 다양한 가치 분포를 사용하여 실험을 수행하였습니다.

- **Performance Highlights**: 논문에서는 SALDAE 알고리즘이 아홉 가지 표준 가치 분포에서 높은 품질의 솔루션을 빠르게 찾아낼 수 있음을 보여줍니다. 또한, 이 알고리즘은 최첨단 방법들과 비교했을 때 유리한 결과를 보였으며, 단시간 내에 최적 솔루션을 도출할 수 있음을 입증하였습니다.



### Forget the Data and Fine-Tuning! Just Fold the Network to Compress (https://arxiv.org/abs/2502.10216)
Comments:
          This paper has been accepted by The Thirteenth International Conference on Learning Representations(ICLR), 2025

- **What's New**: 본 논문에서는 데이터가 필요 없는 새로운 모델 압축 기법인 모델 폴딩(model folding)을 소개합니다. 이 기법은 레이어 간 구조적으로 유사한 뉴런을 결합하여 모델의 크기를 크게 줄일 수 있으며, 이를 통해 파인튜닝(fine-tuning)이나 훈련 데이터 접근 없이도 성능을 유지합니다. 기존 방법들과는 달리, 모델 폴딩은 k-means clustering을 활용하여 데이터 통계를 보존하며, 변동성 붕괴(variance collapse) 및 폭발(variance explosion)을 방지하도록 설계되었습니다.

- **Technical Details**: 모델 폴딩은 뉴런 군집화(neuron clustering), 병합(merging), 데이터 통계 수리(data statistics repair)라는 세 가지 단계로 구성됩니다. 이 과정에서 k-means clustering을 사용하며, REPAIR 프레임워크를 활용하여 데이터 유무에 관계없이 모델 통계를 유지하면서 변동성이 붕괴되지 않도록 돕습니다. 우리는 Fold-AR과 Fold-DIR이라는 두 가지 데이터 없는 대안을 제시하며, 이들은 BatchNorm의 통계를 업데이트하는 데 사용될 수 있는 방법입니다.

- **Performance Highlights**: 모델 폴딩은 현재 데이터 기반 압축 기법과 유사한 성능을 달성하면서, 특히 높은 희소성(sparsity) 수준에서 최근에 제안된 데이터 없는 방법들보다 더 나은 성과를 보입니다. 실험을 통해 ResNet18 및 LLaMA-7B와 같은 표준 벤치마크에서 검증된 바, 자원 제약 환경에서도 대규모 모델 압축에 매우 효과적임을 입증하였습니다. 모델 폴딩은 기존의 최첨단 모델 압축 기법들보다 높은 성능을 유지하며, 데이터와 파인튜닝이 없이도Comparable results를 달성했습니다.



### Prediction hubs are context-informed frequent tokens in LLMs (https://arxiv.org/abs/2502.10201)
- **What's New**: 이 논문에서는 고차원 데이터에서 발생하는 허브니스(hubness) 현상과 대규모 자가 회귀 언어 모델(autoregressive large language models, LLMs)에서의 허브니스의 영향을 연구합니다. 저자들은 LLM에서 확률을 결정하는 방식에서 허브니스가 어떻게 나타나는지를 수학적으로 및 경험적으로 분석합니다. 연구 결과 LLM의 특정 예측 작업에서는 허브니스가 부정적인 성질이 아닐 수 있으며, 자주 등장하는 토큰의 확률을 높이는 전략으로 작용함을 보입니다.

- **Technical Details**: 허브니스는 고차원 데이터에서 특정 데이터 포인트가 다른 많은 포인트의 k-최근접 이웃에 포함되는 경향을 나타냅니다. LLM에서는 컨텍스트와 언임베딩 벡터 간의 관계를 통해 예측을 수행하며, 이 연산은 확률 거리(probability distance)라고 불리는 새로운 측정 방식으로 나타납니다. 의도된 거리 기반 계산에서 허브니스가 발생할 수 있지만, 이는 자연어 텍스트의 단어 분포가 불균형하여 자주 등장하는 토큰과 관련이 있습니다.

- **Performance Highlights**: 저자들은 LLM의 예측 과정에서 발생하는 허브니스가 단순한 거리 집중 현상 때문이 아니라, 결과적으로 적절한 추정을 위한 전략임을 보였으며, 이러한 허브는 제거할 필요가 없다고 주장합니다. 반면, 다른 유사성 비교에 관해서는 허브니스가 부정적인 영향을 미칠 수 있으며, 따라서 이러한 계산을 수행할 때 허브니스 감소 기법과 결합해야 함을 강조합니다.



### Dynamic Reinforcement Learning for Actors (https://arxiv.org/abs/2502.10200)
Comments:
          31 pages, 20 figures

- **What's New**: 이번 논문에서는 시스템 동역학을 직접 제어하는 Dynamic Reinforcement Learning (Dynamic RL) 기법을 제안합니다. 기존의 정적 방식에서 벗어나, Dynamic RL은 탐색을 행위의 본질적인 측면으로 연계하여 비선형적 동적 시스템을 통해 탐색을 생성합니다. 연구진은 이 기술이 기존의 강화 학습에서 질적인 변화를 가져올 수 있는 가능성을 보여준다고 주장합니다.

- **Technical Details**: Dynamic RL은 '민감도(sensitivity)'라는 지역 지표를 사용하여 시스템 동역학을 제어합니다. 민감도 조정 학습(sensitivity adjustment learning, SAL)은 지나친 수렴을 방지하고, 민감도 제어 강화 학습(sensitivity-controlled reinforcement learning, SRL)은 긍정적 TD 오류 주변에서 더 나은 상태 전이로 수렴하도록 조정합니다. 이를 통해 RL 에이전트는 탐색과 사고의 발달을 이룰 수 있는 잠재력을 가지고 있습니다.

- **Performance Highlights**: Dynamic RL은 Actor-Critic RL 아키텍처에서 에이전트의 행동 생성 단계에만 적용되었으며, 두 가지 동적 작업에서 외부 탐색 노이즈나 시간에 따른 역전파 없이 효과적으로 작동했습니다. 새로운 환경에 대한 뛰어난 적응력을 보였지만, 여전히 일부 문제는 존재합니다. 이 연구는 기존 생성 AI와 달리 사고의 출현 가능성에 대한 주제를 탐구하며, 향후 더 나은 아이디어를 위해 토론을 유도하고자 합니다.



### Exploring the Camera Bias of Person Re-identification (https://arxiv.org/abs/2502.10195)
Comments:
          ICLR 2025 (Spotlight)

- **What's New**: 본 연구는 기존의 person re-identification (ReID) 모델에서 발생하는 카메라 바이어스를 실증적으로 조사합니다. 카메라 인식 메소드를 통해 해결하려는 노력이 있었지만, 대부분 훈련 도메인에 국한되어 있었습니다. 우리는 기존 ReID 모델이 보지 못한 도메인에서도 큰 카메라 바이어스를 보인다는 점을 밝혀내었습니다.

- **Technical Details**: 카메라 바이어스는 같은 카메라에서 촬영된 샘플들이 특징 공간(feature space)에서 더 가깝게 모이는 현상으로, 서로 다른 정체성을 가진 샘플이 유사하게 인식될 수 있습니다. 따라서 이 문제를 해결하기 위해 우리는 embedding 벡터의 정규화(normalization) 방법을 다시 검토하고, 이 방법이 어떻게 카메라 레이블에 대한 바이어스를 줄여주는지를 분석합니다. 또한, 다양한 모델과 벤치마크에서 그 일반화를 검증하여 ReID의 테스트 단계에서 간단하면서도 효과적인 후처리(postprocessing) 방법으로 제시합니다.

- **Performance Highlights**: 실험 결과, 기존의 비감독 학습(USL) 알고리즘에서 카메라 바이어스에 의해 교육이 부정적인 영향을 받는 것을 확인했습니다. 우리는 간단한 훈련 전략을 제안하여 이 문제를 해결할 수 있으며, 기존의 알고리즘에 소규모 수정만으로도 성능이 크게 향상될 수 있음을 보여주었습니다. 이러한 연구 결과는 향후 ReID 시스템의 일반화 가능성을 높이고, 카메라 바이어스를 완화하는 데 기여할 것입니다.



### Merging public elementary schools to reduce racial/ethnic segregation (https://arxiv.org/abs/2502.10193)
Comments:
          Forthcoming in PNAS Nexus

- **What's New**: 이번 연구는 미국 공립 학교 내에서 인종/민족, 사회경제적 배경이 다른 학생들 간의 통합을 촉진할 수 있는 '학교 통합(mergers)'이라는 정책을 탐구합니다. 연구 결과, 학교 통합이 인종/민족 분리(segregation)를 중간적으로 20%까지 줄일 수 있으며, 일부 교육구에서는 최대 60% 까지 감소할 수 있다는 것을 발견했습니다. 이러한 변화는 학생들의 통학 시간에 큰 부담을 주지 않으면서 이루어질 수 있는 가능성을 보여줍니다.

- **Technical Details**: 학교 통합은 두 개 이상의 인접 학교 간의 출석 경계를 합쳐서 학생들이 다양한 배경을 경험할 수 있도록 하는 방법입니다. 이 연구에서는 미국의 200개 대형 학군에서 약 450만 명의 초등학생을 대상으로 시뮬레이션 알고리즘을 개발하여 학교 통합의 효과를 분석했습니다. 학교 통합은 기존의 재구역화(redistricting) 같은 통합 정책과 비교하여 더 많은 지역에서 효과적인 결과를 낼 수 있는 방안으로 제시됩니다.

- **Performance Highlights**: 학교 통합이 성공하면, 가족의 이동시간이 평균 몇 분 증가하는 반면, 인종/민족 분리는 상당히 줄어드는 결과를 보여줍니다. 특정 지역에서는 학생들이 서로 다른 배경의 친구를 만나고, 이렇므로 더 포괄적이고 공평한 학습 환경을 조성하는 데 기여할 수 있습니다. 연구 결과는 정책 입안자와 지역 사회 구성원들이 접근할 수 있도록 공개 대시보드를 통해 제공되며, 이는 통합 정책 수립에 실질적인 도움을 줄 수 있습니다.



### From Markov to Laplace: How Mamba In-Context Learns Markov Chains (https://arxiv.org/abs/2502.10178)
- **What's New**: 최근 인공지능 혁명의 중심에 서 있는 transformer 기반 언어 모델은 높은 계산 복잡도로 인해 대안 모델에 대한 관심이 커지고 있습니다. 이 가운데 Mamba(S6)와 Mamba-2는 고급 언어 모델링 작업에서 transformer와 비교할 때 매우 빠른 추론 속도를 기록하며 동등하거나 뛰어난 성능을 보여주었습니다. 하지만 Mamba의 근본적인 학습 능력에 대한 이해는 부족한 상황입니다.

- **Technical Details**: 이 연구에서는 Mamba의 in-context learning (ICL) 능력을 체계적으로 연구하기 위한 이론적 및 실증적 분석의 프레임워크를 제시합니다. 특히, Markov 체인을 활용하여, 단일 층의 Mamba가 transformer와 달리 효과적으로 Laplacian smoothing estimator를 학습할 수 있다는 흥미로운 현상을 발견했습니다. 또한, 이 연구는 Mamba의 표현 용량을 이론적으로 특성화하고, convolution이 최적 Laplacian smoothing을 가능하게 하는 근본적인 역할을 한다고 설명합니다.

- **Performance Highlights**: Mamba는 Markovian 및 복잡한 자연어 데이터에서 이론적 통찰이 실증적 결과와 강한 정합성을 보여줍니다. Mamba는 모든 Markov 차수에 대해 Bayes 및 minimax 최적의 Laplacian smoothing 추정기를 나타낼 수 있는 능력을 갖추고 있으며, 이는 Mamba와 최적 통계 추정자 간의 최초의 공식적인 연결을 보여주는 중요한 발견입니다. 이러한 결과는 Mamba 모델의 ICL 능력이 뛰어난 Transformer 모델과 경쟁할 수 있다는 가능성을 제기합니다.



### Technical Risks of (Lethal) Autonomous Weapons Systems (https://arxiv.org/abs/2502.10174)
- **What's New**: 이번 보고서는 (Lethal) Autonomous Weapons Systems, 즉 (L)AWS의 운영 능력에 따른 혁신적인 변화와 그로 인해 생길 수 있는 심각한 위험에 대해 다룹니다. 이러한 시스템은 국제 안보의 통제, 책임, 안정성 원칙을 위협하는 요소로 고려됩니다. 특히 (L)AWS의 배치로 인한 기술적 위험과 그들의 불예측성, 불투명성에 대해 강조합니다.

- **Technical Details**: (L)AWS의 장점은 객체화(Objectification)와 분류(Classification)를 통해 얻을 수 있지만, 분류 알고리즘의 신뢰성과 예측 가능성을 제한하는 여러 체계적 위험이 존재합니다. 이러한 위험 가운데는 AI 결정 과정의 블랙박스 성격, 보상 해킹(Reward Hacking)에 대한 취약성, 목표 불일치(Goal Misgeneralization) 및 인간 통제를 벗어난 새로운 행동(Emergent Behaviors)의 가능성이 포함됩니다.

- **Performance Highlights**: (L)AWS는 예상치 못한 방식으로 행동할 수 있으며, 이는 임무 목표를 저해하고 갈등을 악화시킬 수 있습니다. 심지어 엄격하게 시험된 시스템조차도 실제 환경에서는 예측할 수 없고 해로운 방식으로 작동할 수 있어 전략적 안정성과 인도적 원칙 모두에 위험을 초래합니다.



### Revisiting Generalization Power of a DNN in Terms of Symbolic Interactions (https://arxiv.org/abs/2502.10162)
Comments:
          arXiv admin note: text overlap with arXiv:2407.19198

- **What's New**: 본 논문은 딥 뉴럴 네트워크(DNN)의 일반화 능력을 상호작용(interactions) 관점에서 분석하고자 합니다. DNN의 일반화 능력을 고차원 특징 공간에서 분석한 이전 연구들과 달리, 본 연구에서는 DNN을 직관적으로 이해할 수 있는 방식으로 재조명합니다. 즉, DNN의 내부 동작을 명확하게 설명할 수 있는 이론적 체계를 구축하여 이를 통해 DNN의 일반화 능력을 분석할 수 있는 새로운 접근법을 제안합니다.

- **Technical Details**: 논문에서는 DNN의 추론 패턴을 AND-OR 상호작용을 기반으로 한 논리 모델로 정확하게 표현할 수 있음을 입증했습니다. 이를 통해 DNN의 모든 세부 추론 패턴을 효과적으로 신뢰성 있게 추출하고, 일반화 가능한 상호작용과 비일반화 상호작용을 분리할 수 있는 방법론을 개발했습니다. 본 연구의 이론적 기반은 과거 연구들에서 도출된 다양한 정리와 실험 결과에 뿌리를 두고 있습니다.

- **Performance Highlights**: 실험을 통해 일반화 가능한 상호작용은 감소 형태의 분포를 따르며, 비일반화 상호작용은 방추 형태의 분포를 따른다는 주장을 입증했습니다. 이러한 상호작용의 분포는 DNN의 과적합(overfitting) 단계에서도 관찰되며, DNN의 일반화 능력 이해에 큰 기여를 할 것입니다. 본 연구는 DNN에서 실제로 사용되는 두 가지 유형의 상호작용을 효과적으로 설명하는 방법을 제안하며, 초기 실험 결과는 이론의 유용성을 잘 보여줍니다.



### SessionRec: Next Session Prediction Paradigm For Generative Sequential Recommendation (https://arxiv.org/abs/2502.10157)
- **What's New**: SessionRec은 기존의 다음 아이템 예측 패러다임(NIPP)과 현실 세계 추천 시나리오 사이의 근본적인 불일치를 해결하는 새로운 세션 기반 예측 패러다임(NSPP)을 도입합니다. 이 모델은 사용자 세션 기반 상호작용을 더 잘 반영할 수 있도록 세션 인식 표현 학습을 통해 다중 아이템 추천이 가능하도록 설계되었습니다. 또한 세션 내 아이템을 위한 순위 손실(rank loss)을 포함하여 추천 시스템의 효율성을 크게 향상시킬 수 있다는 점을 발견했습니다.

- **Technical Details**: NSPP는 사용자 표현을 세션 수준의 임베딩(session-level embeddings)으로 채택하여 사용자 세션 내에서 양성(interactions)과 음성 상호작용을 모두 포착하는데 중점을 둡니다. 이는 시각 변환기(visual transformers)의 원리를 차용하여 사용자 행동의 넓은 맥락과 교차 통계적 정보를 캡처합니다. 이 외에도 NSPP는 세션 내 수집된 상호작용 데이터를 활용하여 검색(retrieval) 및 순위 매기기(ranking) 작업을 동시에 수행하는 모델 지향적 구조를 갖추고 있습니다.

- **Performance Highlights**: Meituan 앱에서 실시한 A/B 테스트 결과, 제안된 SessionRec 모델은 기존 최상의 기준선 모델보다 평균 27%의 성능 향상을 기록하며 효과iveness를 입증했습니다. 이러한 성과는 공개 데이터셋에서 넓게 진행된 실험에서도 확인되었으며, NSPP가 데이터 양이 증가할수록 성능이 지속적으로 향상된다는 스케일링 법칙을 보여주었습니다.



### Video Soundtrack Generation by Aligning Emotions and Temporal Boundaries (https://arxiv.org/abs/2502.10154)
Comments:
          Submitted to International Joint Conference on Artificial Intelligence (IJCAI) 2025

- **What's New**: EMSYNC 모델은 비디오의 감정적 내용과 시간적 경계를 정렬하는 비디오 기반의 기호 음악 생성 모델로, 사전 훈련된 비디오 감정 분류기가 감정적 특징을 추출한 후, 조건부 음악 생성기가 감정 및 시간 신호에 의해 MIDI 시퀀스를 생성합니다. 특히, 음악의 화음이 장면 전환과 일치하도록 예측할 수 있는 새로운 시간적 조건화 메커니즘인 boundary offsets를 도입하였습니다. 이로 인해 기존 모델과 달리 사건 기반 인코딩이 유지되며, 세밀한 타이밍과 표현이 풍부한 음악적 뉘앙스를 보장합니다.

- **Technical Details**: EMSYNC는 특정 유형의 비디오에 음악 생성을 제한하지 않으며, Lakh MIDI Dataset(총 176,581 샘플)을 활용하여 모든 비디오 유형에 대해 음악을 생성합니다. 이 모델은 비디오에서 음악 생성에 관련된 특징을 추출한 후, 이를 조건부 입력으로 사용하여 MIDI 형식으로 음악을 생성합니다. 우리는 비디오의 장면 전환 위치에서 긴 지속 시간의 화음을 생성하도록 음악 생성기를 안내하며, 이러한 화음은 생성된 음악의 나머지 부분과 리드미컬하고 조화롭게 호환되도록 설계되어 있습니다.

- **Performance Highlights**: 주관적 청취 테스트에서 EMSYNC는 모든 주관적 메트릭에서 최첨단 모델을 초능히 초과하여 성과를 거두었으며, 음악 이론에 대한 이해가 있는 참여자와 일반 청중 모두에게 우수성을 보였습니다. EMSYNC는 빠른 처리 속도를 자랑하며, 사용자가 다양한 작업이나 데이터 구조에 맞춰 모델을 손쉽게 조정할 수 있도록 설계되었습니다. 이 모델은 비디오와 생성된 음악 간의 감정적 연결을 가능케 합니다.



### Learning Relational Tabular Data without Shared Features (https://arxiv.org/abs/2502.10125)
- **What's New**: 최근 관계형 테이블 데이터를 학습하는 것이 주목받고 있지만 대부분의 연구는 단일 테이블에 초점을 두고, 교차 테이블 학습의 가능성을 간과하고 있습니다. 특히 공유 기능이나 사전 정렬된 데이터가 없는 경우의 교차 테이블 학습은 막대한 기회가 있지만 복잡한 도전 과제를 제시합니다. 본 논문에서는 이러한 문제를 해결하기 위해 공유 기능이나 사전 정렬 데이터 없이도 효과적인 교차 테이블 학습을 가능하게 하는 Latent Entity Alignment Learning (Leal)이라는 새로운 프레임워크를 제안합니다.

- **Technical Details**: Leal은 적절하게 정렬된 데이터가 잘못 정렬된 데이터보다 손실(loss)이 낮다는 원칙에 기초하여 작동합니다. 이 프레임워크는 소프트 정렬 메커니즘과 미분 가능한 클러스터 샘플러 모듈을 결합하여 큰 관계형 테이블에 대해 효율적으로 확장됩니다. 또한, 클러스터 샘플러의 근사 능력에 대한 이론적 증명을 제공합니다.

- **Performance Highlights**: 다섯 개의 실제 데이터셋과 다섯 개의 합성 데이터셋에서 수행된 광범위한 실험 결과에 따르면, Leal은 최첨단 방법에 비해 예측 성능이 최대 26.8% 향상되었습니다. 이러한 성능 향상은 Leal의 효과성과 확장 가능성을 입증하며, 공유 기능 없이도 관계형 테이블 데이터에서 기계 학습을 가능하게 하는 최초의 모델로 자리매김합니다.



### Image Embedding Sampling Method for Diverse Captioning (https://arxiv.org/abs/2502.10118)
Comments:
          15 pages, 5 figures, 6 tables

- **What's New**: 이 논문에서는 작은 VLMs(Visual-Language Models)가 이미지 캡션의 다양성과 유용성을 증가시키기 위해 명확하게 서로 다른 이미지 지역에 주의를 기울이는 훈련 없는 프레임워크를 도입합니다. BLIP을 백본으로 사용하는 이 방법은 구조화된 세분화를 활용해 전역 및 지역적 의미를 포착하며 추가적인 모델 훈련 없이도 성능을 향상시킬 수 있습니다. MSCOCO, Flickr30k 및 Nocaps 데이터셋에서 평가한 결과, 기존의 큰 모델과 유사한 수준의 성능을 달성했습니다.

- **Technical Details**: 제안된 HBoP(Hierarchical Bags of Phrases) 프레임워크는 사전 훈련된 세분화 및 캡션 생성 모델을 사용하는 모듈형 아키텍처로 구성됩니다. HBoP는 다단계 캡션(전역, 지역, 세부 수준)을 생성하기 위해 계층 구조를 유도하며, MSD(Multiple Segmentation Decisions)를 통해 이미지의 특정 지역에 대한 패치 임베딩을 선택합니다. 또한, NMS(Non-Maximum Suppression)를 적용하여 중복되는 세분화 마스크를 제거하고, K-HM(K-means Clustering) 기법을 활용하여 지역 수준 세분화를 생성합니다.

- **Performance Highlights**: 이 방법은 아시아 시장을 기준으로 하는 다양한 평가 지표에서 성능을 검증하였으며, 각각의 데이터셋에서 Div-2 점수는 0.735, 0.750, 0.748을 기록했습니다. 이로 인해 HBoP는 이미지-캡션의 관련성과 의미적 일관성을 유지하면서도 다양성을 높이는 효과를 입증하였습니다. 결과적으로, 이는 제한된 자원에서 작동하는 응용 프로그램을 위해 더욱 접근 가능한 이미지 캡션 생성 기술의 발전을 보여줍니다.



### A novel approach to data generation in generative mod (https://arxiv.org/abs/2502.10092)
Comments:
          47 pages, 2 tables, 9 figures

- **What's New**: 본 논문에서는 Convergent Fusion Paradigm (CFP) 이론을 소개하여, 현재의 VAE와 같은 생성 모델들이 갖는 한계를 보완하고자 합니다. CFP 이론은 데이터 생성의 구조화되고 발생적인 특성을 반영하기 위해 새로운 기하학적 프레임워크를 제시합니다. 이는 차원 확장 및 질적 변화를 통합하여 데이터 생성 방식을 재정의하고, LLMs에서의 의도하지 않은 아티팩트 문제를 해결하려는 접근입니다.

- **Technical Details**: CFP 이론은 데이터와 알고리즘 간의 관계를 재구성하는 두 가지 주요 개념 가설에 기반하고 있습니다. 이 이론은 고차원 구조와의 상호작용을 위해 잠재 공간의 기하학을 수정하여, 기존의 메트릭 학습 방법들을 재검토하고 새로운 시간역 메트릭 임베딩 및 구조적 수렴 메커니즘을 제안합니다. 이는 데이터 생성 과정이 구조화된 인식적 과정으로 더 잘 설명될 수 있도록 돕습니다.

- **Performance Highlights**: CFP 이론은 AI의 데이터-관계 구조를 이해하기 위한 이론적 기초를 마련하며, 고차원 학습 역학에 대한 체계적인 프레임워크를 제공합니다. 또한, 이 이론은 질적 변화를 완전히 실현하기 위한 미래 연구 방향에 대해서도 논의하며, 생성 모델링에서 Hilbert 공간의 잠재적 활용 가능성을 제시합니다. 이는 데이터 생성의 존재론적 기초에 대한 철학적 통찰력을 제공하는 기초가 됩니다.



### Manual2Skill: Learning to Read Manuals and Acquire Robotic Skills for Furniture Assembly Using Vision-Language Models (https://arxiv.org/abs/2502.10090)
- **What's New**: 이 논문은 로봇이 고차원 매뉴얼 지침을 따라 복잡한 조립 작업을 수행할 수 있도록 하는 새로운 프레임워크인 Manual2Skill을 제안합니다. 이 접근 방식은 Vision-Language Model (VLM)을 활용하여 지침 이미지에서 구조화된 정보를 추출하고, 이를 기반으로 조립 그래프를 생성합니다. 로봇이 인간과 유사한 방식으로 복잡한 조작 작업을 이해하고 실행할 수 있도록 돕는 중요한 발전을 보여줍니다.

- **Technical Details**: Manual2Skill은 고차원의 매뉴얼 지침에서 조작 기술을 학습하는 로봇 학습 프레임워크입니다. 이 프레임워크는 조립 단계에서 각 구성 요소의 상대적인 6D 포즈를 예측하는 포즈 추정 모델과 실제 로봇 구현을 위한 실행 가능한 동작 시퀀스를 생성하는 동작 계획 모듈을 포함합니다. 이를 통해 IKEA 가구와 같은 복잡한 조립 작업을 자동화합니다.

- **Performance Highlights**: 이 시스템은 여러 현실 세계의 IKEA 가구 항목을 성공적으로 조립하여 그 효과성을 입증합니다. Manual2Skill은 긴 기간의 조작 작업을 효율적이고 정확하게 관리하는 능력을 강조하며, 로봇이 매뉴얼로부터 학습하는 실질성을 크게 향상시킵니다. 이를 통해 로봇 시스템의 복잡한 조작 작업 이해 및 실행 능력이 발전하고 있음을 보여줍니다.



### A Hybrid Edge Classifier: Combining TinyML-Optimised CNN with RRAM-CMOS ACAM for Energy-Efficient Inferenc (https://arxiv.org/abs/2502.10089)
- **What's New**: 최근 스마트 엣지 컴퓨팅 시스템의 발전이 주목받고 있으며, 특히 리소스가 제한된 환경에서의 딥 신경망(Deep Neural Network, DNN) 배치에 대한 수요가 급증하고 있습니다. 본 논문에서는 고전적인 기술과 신흥 기술을 결합한 하이브리드 솔루션을 제안하여 극단적인 엣지 근접 센서 시스템에서의 이식 가능성을 높이고자 합니다. 이를 통해 높은 정확도와 에너지 효율성을 동시에 달성할 수 있는 모델을 개발하고 있습니다.

- **Technical Details**: 제안된 하이브리드 소프트웨어-하드웨어 엣지 분류기는 두 가지 주요 구성 요소로 나뉘어 있습니다: (i) 최적화된 디지털 tinyML 네트워크가 특징 추출기로 작동하며, (ii) 후단에서는 RRAM-CMOS 아날로그 콘텐츠 주소 지정 메모리(ACAM)가 템플릿 매칭 시스템으로 기능합니다. 이러한 시스템은 $E_{front-end}$ = $96.23 nJ$ 및 $E_{back-end}$ = $1.45 nJ$의 에너지 소모를 기록하며, 기존의 교사 모델에 비해 792배 에너지를 절감할 수 있습니다.

- **Performance Highlights**: 제안된 하이브리드 시스템은 정확도와 에너지 소모 측면에서 경쟁력 있는 성능을 보여줍니다. DNN과 아날로그 하드웨어 가속기를 조합하여 기존의 디지털 처리 방식에서 발생하는 높은 에너지 소모 문제를 해결함으로써, 더욱 효율적인 엣지 기계 학습 솔루션을 마련했습니다. 또한 고급 DNN 모델 압축 기술과 함께 RRAM 기반 템플릿 패턴 매칭을 통합하여 현대의 하드웨어 아키텍처에 최적화된 성능을 제공하는 점이 두드러집니다.



### Strassen Multisystolic Array Hardware Architectures (https://arxiv.org/abs/2502.10063)
Comments:
          Accepted for publication in IEEE Transactions on Very Large Scale Integration (VLSI) Systems; Associated source code available on GitHub at this https URL

- **What's New**: 이 논문에서는 Strassen 알고리즘의 이론적 속도 향상을 실제 하드웨어 아키텍처에 구현할 수 있는 방법을 제시합니다. 기존 하드웨어는 이 알고리즘의 복잡성 감소를 잘 활용하지 못한다는 문제를 해결하기 위해 새로운 systolic array 아키텍처를 설계했습니다. 이러한 아키텍처는 구체적으로 Strassen 알고리즘을 효율적으로 실행하기 위해 맞춤 설계되었습니다.

- **Technical Details**: 제안된 multisystolic array 아키텍처는 단일 systolic array 디자인보다 더 작은 행렬을 더 높은 활용도로 곱할 수 있는 설계입니다. FPGA에 구현된 이 아키텍처는 $r$ 수준의 Strassen 재귀(recursion) 효과에 따라 DSP 요구 사항을 $1.14^r$로 줄이는 동시에, 32x32 및 24x24 크기의 행렬을 지원하는 데 비슷한 소프트 로직 리소스를 요구합니다. 이러한 설계는 Strassen 재귀 수준에 따라 구현될 수 있습니다.

- **Performance Highlights**: 제안된 아키텍처는 기존 디자인 및 이전 연구와 비교할 때, 기계 학습 가속기(end-to-end machine learning accelerator)에서 상태-of-the-art 성능을 달성했습니다. 이 논문은 Strassen 알고리즘의 성능을 극대화하기 위한 새로운 접근법을 제시하며, 향후 맞춤형 하드웨어 디자인에서 중요한 기초 자료를 제공할 것입니다.



### Adaptive Bi-Level Multi-Robot Task Allocation and Learning under Uncertainty with Temporal Logic Constraints (https://arxiv.org/abs/2502.10062)
Comments:
          Accepted as a full paper at AAMAS 2025

- **What's New**: 이번 연구는 알려지지 않은 로봇 전이 모델 하에 다중 로봇 조정 문제를 다룹니다. 새로운 프레임워크를 통해 Time Window Temporal Logic으로 지정된 작업을 만족시키고 사용자 정의 확률 임계값을 준수할 수 있도록 합니다. 특히, 높은 수준의 작업 할당과 낮은 수준의 분산 정책 학습 및 실행을 통합하여 로봇이 할당된 작업을 최적화하는 방식으로 진행됩니다.

- **Technical Details**: 제안된 bi-level 프레임워크는 로봇의 예상 작업 완료 확률 및 기대 보상을 기반으로 작업을 할당하는 고급 단계와, 로봇들이 독립적으로 보조 보상을 최적화하면서 할당된 작업을 수행하는 저급 단계로 구성됩니다. 로봇 동역학의 불확실성을 다루기 위해, 실시간 작업 실행 데이터를 활용해 작업 완료 확률과 보상을 반복적으로 정제하며, 이는 명시적인 로봇 전이 모델 없이는 적응형 작업 할당이 가능합니다.

- **Performance Highlights**: 이 연구는 시뮬레이션을 통해 제안된 프레임워크의 효과성을 입증하였으며, 원하는 확률 임계값을 충족하는 작업 할당을 비롯하여 보조 보상 함수를 최대화함으로써 사용자 선호를 반영하는 방법을 제시합니다. 또한, 이러한 접근법은 다양한 상황에서 적용 가능성이 있으며, 예를 들어 긴급 구조나 환경 모니터링 작업에도 활용될 수 있습니다.



### A Survey on LLM-powered Agents for Recommender Systems (https://arxiv.org/abs/2502.10050)
- **What's New**: 최근의 대규모 언어 모델(LLM) 기반 에이전트는 기존 추천 시스템의 복잡한 사용자 선호도 이해와 해석 가능한 추천 제공을 개선할 수 있는 잠재력을 갖추고 있습니다. 본 논문은 LLM 기반 에이전트의 여러 가지 응용 사례를 탐색하면서 전통적인 방법의 한계를 극복하는 방식으로 이들 에이전트의 주요 패러다임을 제시합니다. 특히 추천 지향, 상호작용 지향, 시뮬레이션 지향의 세 가지 연구 패러다임을 정리하여 LLM의 효과적인 적용을 설명합니다.

- **Technical Details**: 추천 시스템은 일반적으로 사용자 공간, 아이템 공간 및 상호작용 매트릭스를 기반으로 하며, 이는 사용자 선호도를 예측하는데 중요한 역할을 합니다. 기존의 매트릭스 분해(matrix factorization) 방법과 심층 학습(deep learning) 기법은 복잡한 사용자 의도를 이해하는 데 어려움을 겪고, 명확한 해명을 제공하지 못하는 '블랙박스' 문제를 안고 있습니다. LLM 기반 에이전트는 프로필 구축(profile construction), 메모리 관리(memory management), 계획적 전략 수립(strategic planning), 행동 실행(action execution)의 네 가지 핵심 모듈로 구성되어 복잡한 작업을 관리 가능한 구성 요소로 분해하는 혁신적인 접근 방식을 제안합니다.

- **Performance Highlights**: 본 조사는 LLM 기반 추천 시스템의 현재 상태를 조명하고, 벤치마크 데이터 세트와 평가 프레임워크를 체계적으로 분석합니다. 또한, LLM의 사전 훈련된 지식(pre-trained knowledge)과 강력한 일반화 능력은 다양한 도메인 간의 지식 이전(knowledge transfer)을 촉진하여 차가운 시작(cold-start) 문제를 최소한의 추가 훈련으로 해결할 수 있도록 합니다. 마지막으로, 이 연구는 LLM 기반 추천 시스템의 핵심 과제와 미래 연구 방향에 대한 통찰을 제공합니다.



### Janus: Collaborative Vision Transformer Under Dynamic Network Environmen (https://arxiv.org/abs/2502.10047)
Comments:
          Accepted for publication in IEEE INFOCOM 2025

- **What's New**: 이번 논문에서는 Vision Transformer(ViT)의 저지연 클라우드-디바이스 협력 추론을 위한 첫 번째 프레임워크 Janus를 소개합니다. Janus는 동적 네트워크에서 저지연 및 높은 정확성을 실현하며, ViT 모델의 본질적인 한계를 극복합니다. 특히, Janus는 토큰 프루닝(token pruning) 기술과 모델 스플리팅(model splitting)을 결합하여 효율적인 클라우드-디바이스 협력을 가능하게 합니다.

- **Technical Details**: Janus는 협력 인퍼런스를 위해 특별히 설계된 시스템으로, 클라우드와 엣지 디바이스에서 ViT를 함께 실행하여 전송되는 데이터의 양을 줄입니다. 이 시스템의 핵심 구성 요소는 협력 인식을 갖춘 토큰 프루너(token pruner)와 정밀한 모델 스플리터(model splitter)입니다. 최적의 프루닝 수준과 분할 지점을 결정하기 위해 ViT 지향의 레이턴시 프로파일러(latency profiler)와 다이내믹 스케줄러(dynamic scheduler)를 설계했습니다.

- **Performance Highlights**: 실험 결과, Janus는 다양한 작업에서 성능이 크게 향상되어 최대 5.15배의 처리량을 달성하였고, 대기 시간 위반 비율을 최대 98.7%까지 줄였습니다. 이는 최소한의 정확도 감소로 이루어지며, 논문의 접근 방식이 기존의 베이스라인 방법에 비해 효율성을 나타냅니다. Janus는 실제 환경에서 테스트 되어 저지연 인퍼런스를 필요로 하는 응용 프로그램에 적합하다는 것을 보여줍니다.



### X-Boundary: Establishing Exact Safety Boundary to Shield LLMs from Multi-Turn Jailbreaks without Compromising Usability (https://arxiv.org/abs/2502.09990)
- **What's New**: 최근 대형 언어 모델(LLMs)의 보안 취약점 중 하나인 multi-turn jailbreaks에 대한 방어 방식이 어려운 주제로 떠오르고 있습니다. 본 논문에서는 기존의 방어 메커니즘을 평가하고, 일부 방법이 LLM의 다중 턴 공격에 대한 강인성을 향상시킬 수 있지만 사용성(usability)을 저하시키는 것에 대해 논의합니다. 새로운 X-Boundary 방법론을 제안하여, 위험한 표현과 안전한 표현 사이의 경계를 명확히 설정하여 방어 성능을 개선하고 over-refusal 문제를 줄였습니다.

- **Technical Details**: X-Boundary는 위험한 표현을 안전한 표현으로부터 물리적으로 밀어내는 최적화 방법입니다. 이를 통해 위험한 표현을 정확하게 제거하면서 안전한 표현은 유지할 수 있습니다. 기존 방법들은 이러한 경계를 명확히 설정하지 못했으며, 그 결과로 over-refusal과 같은 사용성 문제를 야기했습니다. 실험 결과에 따르면, X-Boundary는 multi-turn jailbreaks에 대한 방어 성능이 기존 방법들보다 우수하고, 일반적인 기능을 거의 완전하게 유지하면서 20% 수준의 과도한 거부(over-refusal)를 감소시켰습니다.

- **Performance Highlights**: X-Boundary는 Llama-3-8B-Instruct에서 multi-turn jailbreaks의 공격 성공률(ASR)을 58.5%에서 16.5%로 감소시켰습니다. 게다가, 이 방법은 교육 과정 동안 수렴 속도를 약 22% 향상시킬 수 있다는 이론적 분석 및 실험적 검증을 제공하였습니다. 우리의 연구는 X-Boundary가 강력한 방어성과 최소한의 사용성 저하를 동시에 달성할 수 있다는 점을 강조합니다. 이런 다면적인 접근 방식을 통해 LLM의 보안과 실용성을 동시에 향상시킬 수 있는 가능성을 보여줍니다.



### LaRA: Benchmarking Retrieval-Augmented Generation and Long-Context LLMs - No Silver Bullet for LC or RAG Routing (https://arxiv.org/abs/2502.09977)
Comments:
          22 pages

- **What's New**: 이 논문은 RAG( retrieval-Augmented Generation)와 LC(long-context) LLMs의 효과적인 비교를 위한 새로운 벤치마크인 LaRA를 제안합니다. LaRA는 2,326개의 테스트 사례로 구성되어 있으며, 이는 두 접근 방식이 어떻게 차별화되는지를 명확히 분석할 수 있도록 설계되었습니다. 이를 통해 RAG와 LC의 강점과 약점을 체계적으로 비교하는 데 기여할 것입니다.

- **Technical Details**: LaRA는 다양한 질문 답변(Question Answering, QA) 과제를 평가하기 위해 자연발생적인 긴 텍스트로 구성된 데이터셋을 사용합니다. 이 데이터셋에는 소설, 학술 논문, 재무 제표 등 다양한 서식이 포함되어 있어 여러 문체와 정보 밀도를 대표합니다. 또한, LaRA의 QA 쌍은 인간 주석자와 LLM의 협력으로 생성되며, 정확성을 보장하기 위해 GPT-4o를 사용하여 예측 판별을 수행합니다.

- **Performance Highlights**: 연구 결과, RAG와 LC의 선택은 모델의 파라미터 크기, 긴 텍스트 처리 능력, 컨텍스트 길이, 작업 유형 및 검색된 조각의 특성과 같은 여러 요인에 따라 달라진다는 것을 확인했습니다. 저자들은 이를 이용해 LLM 애플리케이션을 최적화하는 방법에 대한 실용적인 가이드를 제공하며, 강력한 모델일수록 LC가 더 좋은 성능을 보인다는 점을 강조하였습니다.



### Conditional Latent Coding with Learnable Synthesized Reference for Deep Image Compression (https://arxiv.org/abs/2502.09971)
- **What's New**: 이 논문에서는 외부 사전(dictionary)으로부터 동적 참조를 합성하여 입력 이미지의 조건부 인코딩을 수행하는 방법을 연구합니다. 새로운 접근법인 Conditional Latent Coding (CLC)을 통해 입력 이미지에 대해 효율적이고 동적인 조건부 잠재(latent) 표현을 생성합니다. 이 방법은 이미지의 소스 상관관계를 탐색하는 데 매우 효과적입니다.

- **Technical Details**: 우리는 수정된 공간 피라미드 풀링(modified spatial pyramid pooling), 차원 축소(dimension reduction), 다중 스케일 특성 클러스터링(multi-scale feature clustering)을 포함하는 다단계 접근법을 통해 보편적인 이미지 특성 사전을 구성합니다. 클로우만 동적 참조를 생성하기 위해 조건부 잠재 매칭(module)과 조건부 잠재 합성(module)을 사용하여 효율적인 이미지 압축이 가능합니다. 이 방법은 외부 사전 샘플의 섭동(perturbations)에 대해 강건한 것으로 입증되었습니다.

- **Performance Highlights**: 실험 결과, 이 새로운 방법은 벤치마크 데이터세트에서 코딩 성능을 최대 1.2 dB 향상시키는 것을 보여주며, 비트/픽셀(bit per pixel)의 약 0.5%의 오버헤드(overhead)로 달성되었습니다. 또한 CLC 방법은 대규모 및 다양한 사전에서도 여전히 안정적인 성능을 유지할 수 있는 것으로 나타났습니다.



### Data Valuation using Neural Networks for Efficient Instruction Fine-Tuning (https://arxiv.org/abs/2502.09969)
- **What's New**: 이번 논문에서는 Influence function을 효율적으로 추정하기 위한 새로운 방법인 NN-CIFT(Neural Networks for effiCient Instruction Fine-Tuning)를 소개합니다. 기존의 방법들은 대규모 언어 모델의 비싼 계산 비용과 좋은 일반화 성능 결여 문제를 가지고 있었지만, NN-CIFT는 소형 신경망인 InfluenceNetwork를 활용하여 최대 99%의 비용 절감을 달성합니다. 보고된 결과에 따르면, NN-CIFT는 전체 모델보다 0.0027%의 크기로도 유사한 성능을 유지하면서 영향 값을 추정할 수 있습니다.

- **Technical Details**: NN-CIFT는 세 단계의 알고리즘으로 구성되어 있습니다. 첫 번째 단계에서는 기존 Influence function을 사용해 작은 데이터 집합에 대한 영향 값을 추정하고, 이 작은 집합을 기반으로 InfluenceNetwork를 훈련시킵니다. 두 번째 단계에서 훈련된 InfluenceNetwork를 활용해 나머지 데이터 포인트의 영향 값을 추정하고, 마지막 단계에서는 이 추정된 영향 값들을 바탕으로 데이터 선택 알고리즘을 적용하여 IFT 데이터를 추출합니다.

- **Performance Highlights**: NN-CIFT는 기존의 Influence function과 비교하여 성능 저하 없이 77-99%의 시간 절약을 이뤄냈습니다. 연구 결과에 따르면, NN-CIFT를 통해 선택된 데이터의 평균 제곱 오차는 단지 0.067에 불과하며, 이는 원본 Influence function과의 차이를 최소화합니다. 이로 인해 NN-CIFT는 새로운 데이터 포인트에 대해 다시 훈련할 필요 없이도 효과적이라는 것이 입증되었습니다.



### KGGen: Extracting Knowledge Graphs from Plain Text with Language Models (https://arxiv.org/abs/2502.09956)
- **What's New**: 최근 지식 그래프(KG) 구축에 대한 관심이 높아지면서 데이터 부족 문제가 중요하게 여겨지고 있습니다. KGGen은 자연어 텍스트로부터 고품질의 지식 그래프를 생성하는 텍스트-투-KG(generator) 패키지로, Python 라이브러리 형태로 제공됩니다. KGGen은 관련된 엔티티를 클러스터링하여 추출된 KGs의 희소성을 줄이는 혁신적인 접근법을 채택했습니다. 또한, 새로운 벤치마크인 MINE을 통해 기존 추출기와 비교해 18% 더 높은 성능을 보였습니다.

- **Technical Details**: KGGen은 LLM(대형 언어 모델)을 활용하여 평문에서 주어-서술어-목적어(triple)를 추출하고, 클러스터링 알고리즘을 통해 고품질의 조밀한 KG를 생성합니다. 이 과정에서 여러 단계를 거치며, 첫 번째 단계에서 비구조화된 텍스트를 입력받아 초기 지식 그래프를 생성하고, 그 다음에 유일한 엔티티와 연결 관계를 집계합니다. KGGen은 각 단계에서 DSPy 프레임워크를 사용해 일관된 JSON 형식의 출력을 보장합니다.

- **Performance Highlights**: KGGen은 벤치마크 테스트에서 기존 텍스트-투-KG 추출기보다 18% 더 뛰어난 성능을 나타냈습니다. 이 성능 향상은 KGGen이 고품질의 밀접하게 연결된 KGs를 자동으로 생성할 수 있는 잠재력을 보여줍니다. KGGen의 도입으로 오는 데이터 풍부한 미래는 차세대 KG 기반 모델 훈련과 RAG 시스템에 긍정적인 영향을 미칠 것으로 기대됩니다.



### Using MRNet to Predict Lunar Rock Categories Detected by Chang'e 5 Prob (https://arxiv.org/abs/2502.09952)
Comments:
          Published at the 8th International Conference on Advances in Machinery, Material Science and Engineering Application (MMSE 2022)

- **What's New**: 중국의 창어 5호(Chang'e 5) 임무는 큰 성공을 거두었습니다. 이 임무는 해왕성의 바다인 Oceanus Procellarum에서 달 표면의 이미지를 수집하기 위해 디자인되었습니다. 이번 연구는 제한된 양의 달 암석 샘플을 활용하기보다, 달 탐사 로봇을 통한 암석 분석에 집중하고 있습니다.

- **Technical Details**: 이 연구는 CE5ROCK이라는 lunar surface rock image 데이터 세트를 구축했습니다. 이 데이터 세트는 100개의 이미지를 포함하고 있으며, 훈련, 검증 및 테스트 세트로 무작위로 나뉘어 있습니다. 또한, MRNet이라는 새로운 네트워크 아키텍처를 제안하며, 이는 VGG16을 기반으로 한 feature extraction 기능과 dilated convolution을 결합하여 더 세밀한 달 암석 식별에 유리합니다.

- **Performance Highlights**: 실험 결과 MRNet은 기존의 CNN 모델인 AlexNet 및 MobileNet보다 약 40.0%의 식별 정확도를 보였습니다. CE5ROCK 데이터 세트를 사용한 추가 실험에서 MRNet은 더 정밀한 암석 유형 식별을 달성했으며, 기존의 주요 알고리즘을 능가하는 성능을 입증했습니다.



### TransGUNet: Transformer Meets Graph-based Skip Connection for Medical Image Segmentation (https://arxiv.org/abs/2502.09931)
Comments:
          24 pages, 12 figures

- **What's New**: 이번 연구에서는 medical image segmentation을 위해 cross-scale GNN 기반의 skip connection 구조를 활용한 새로운 모델, TransGUNet을 제안합니다. 이 모델은 복잡한 해부학적 구조를 이해하기 위해 주의(node attention)를 적용하며, 전통적인 모델들이 가진 semantic gap 문제를 해결합니다. 또한, Entropy-driven feature selection (EFS)을 통해 더 신뢰할 수 있는 spatial attention map을 생성합니다.

- **Technical Details**: TransGUNet은 attentional cross-scale graph neural network (ACS-GNN)와 EFS 기반 spatial attention을 통합한 구조입니다. 이 모델은 cross-scale feature map을 그래프로 변환하여 각 노드에 대해 주의를 기울여 robust한 feature 통합을 도출합니다. 이와 함께, deep learning 모델들이 생성하는 비정보적인 feature map 문제를 해결하기 위해, 채널별 엔트로피 점수를 계산하여 높은 엔트로피를 가진 feature map을 필터링합니다.

- **Performance Highlights**: TransGUNet은 6개의 학습된 데이터셋 및 8개의 미학습 데이터셋에 대해 우수한 세분화 성능을 발휘하며, 이전의 transformer- 및 convolution-based 접근 방식에 비해 상당히 높은 효율성을 보여줍니다. 종합적인 실험 결과를 통해 이 모델이 의료 이미지 분할에 있어 기존 방법들보다 더 나은 성능과 신뢰성을 제공함을 증명하였습니다.



### Deep Tree Tensor Networks for Image Recognition (https://arxiv.org/abs/2502.09928)
- **What's New**: 이번 논문에서는 Deep Tree Tensor Network (DTTN)이라는 새로운 아키텍처를 소개합니다. DTTN은 고차원 특성 간 곱셈 상호작용을 포착하는 멀티리니어 연산을 활용하여 성능을 향상시킵니다. 이 아키텍처는 여러 개의 대칭 모듈 (AIM)을 쌓아 올려 구현되며, 트리 형태의 텐서 네트워크 구조로 펼쳐집니다.

- **Technical Details**: DTTN은 활성화 함수와 주의 메커니즘 없이 2^L 차수의 곱셈 상호작용을 캡처하며, 다수의 벤치마크에서 기존 아키텍처들과 동등한 성능을 보입니다. 이 모델은 텐서 저차원 구조의 특징을 유지하면서도 높은 해석 가능성 및 직관성을 제공합니다. 또한 DTTN은 상대적으로 더 빠른 수렴 속도를 자랑하며, 고차원 텐서 간의 내적과 같은 효율적인 수학적 기법을 사용하여 연산을 최적화합니다.

- **Performance Highlights**: DTTN은 기존의 폴리노미얼 및 멀티리니어 네트워크와 비교하여 뛰어난 성능을 달성하였습니다. 여러 복잡한 벤치마크 테스트에서도 전반적으로 우수한 결과를 나타내었으며, 고급 아키텍처와의 성능을 일치시킵니다. 이로 인해 향후 DTTN을 활용하여 더 많은 해석 가능성을 가진 연구를 진행할 수 있을 것으로 기대됩니다.



### Granite Vision: a lightweight, open-source multimodal model for enterprise Intelligenc (https://arxiv.org/abs/2502.09927)
- **What's New**: Granite Vision은 경량의 대형 언어 모델로, 특히 시각적 문서 이해에 최적화되어 설계되었습니다. 이 모델은 문서 관련 작업을 포함한 포괄적인 instruction-following dataset에서 훈련되어, 테이블, 차트 및 인포그래픽과 같은 다양한 콘텐츠를 자동으로 추출할 수 있습니다. Granite Vision은 총 30억 개의 파라미터를 갖고 있으며, 테스트 시 안전 분류 접근 방식을 도입하여 잠재적으로 위험한 입력을 식별합니다.

- **Technical Details**: Granite Vision은 비주얼 모달리티의 정렬에 중점을 둔 2억 개의 파라미터를 가진 decoder-only 구조로 이루어져 있습니다. 이 모델은 시각적 인코더와 Granite 대형 언어 모델을 연결하기 위해 projector를 사용하고, 여러 단계의 훈련 프로토콜로 성능을 최적화했습니다. 또한 시각적 문서 이해에 필요한 세밀한 정보 포착을 위해 멀티 레이어 특징을 추출하고, 희소한 주의 벡터를 기반으로 한 안전 분류 모듈을 제안합니다.

- **Performance Highlights**: Granite Vision은 시각적 문서 이해와 관련된 여러 기준 벤치마크에서 최고의 성과를 달성했습니다. 특히, 최근에 발표된 Arxiv 논문을 사용하여 모델의 오염을 피할 수 있는 LiveXiv 벤치마크에서도 두각을 나타냅니다. 연구 및 상업적 사용이 가능한 Apache-2 라이센스 하에 모델을 공개하여 투명성을 높이고 협력을 촉진하고 있습니다.



### TaskGalaxy: Scaling Multi-modal Instruction Fine-tuning with Tens of Thousands Vision Task Types (https://arxiv.org/abs/2502.09925)
- **What's New**: TaskGalaxy는 19,227개의 계층적 작업 유형과 413,648개의 샘플로 구성된 대규모 다중 모달 지침 미세 조정 데이터 세트입니다. 이 데이터 세트는 GPT-4o를 활용하여 기존 수작업 정의 작업의 소량 세트에서 확장하여 작업의 다양성을 높입니다. CLIP과 GPT-4o를 통해 오픈 소스 이미지와 가장 잘 일치하는 작업 유형을 필터링하고, 관련된 질문-답변 쌍을 생성하여 데이터 품질을 확보합니다.

- **Technical Details**: TaskGalaxy 데이터 세트는 OCR, 이미지 설명 및 복잡한 논리적 추론과 같은 다양한 작업 유형을 포함하고 있습니다. 기존의 수동 정의 작업 세트를 시작으로 하여 GPT-4o가 프로세스 내내 작업을 자동으로 확장합니다. CLIP의 유사성 평가를 통해 적절한 이미지와 강력히 관련된 작업 유형이 선정되며, 그에 따라 생성된 질문-답변 쌍은 세 가지 오픈 소스 모델을 통해 스크리닝하여 품질을 확보합니다.

- **Performance Highlights**: LLaVA-v1.5와 InternVL-Chat-v1.0 모델에 TaskGalaxy를 통합한 결과, 16개의 벤치마크에서 성능이 크게 향상되었습니다. 평균적으로 4.5 및 3.83, 3.0 및 3.64 포인트 개선된 결과를 나타냈습니다. 특히 MME 벤치마크에서 LLaVA-v1.5-13B 모델이 68 포인트 증가하여 TaskGalaxy 데이터 세트가 모델의 일반화 능력을 향상시킨다는 것을 증명했습니다.



### Machine Learning for Phase Estimation in Satellite-to-Earth Quantum Communication (https://arxiv.org/abs/2502.09920)
- **What's New**: 본 논문은 위성에서 지구로의 채널을 통한 새로운 지속적 변수 양자 키 분배(CV-QKD) 네트워크의 개선된 성능을 위해, 실제 로컬 오실레이터(RLO)를 사용한 신호의 위상 오류 추정 알고리즘의 복잡성을 줄이면서도 정확도를 높이는 새로운 신경망 아키텍처를 제안합니다. 특히, 저복잡도 신경망을 활용하여 실시간 성능을 크게 향상시키는 과정을 설명하고 있습니다.

- **Technical Details**: 연구에서는 LSTM(장기 단기 기억) 신경망의 아키텍처를 신호 위상 오류 추정 정확도와 관련하여 분석하고, Fisher 정보 기반의 불확실성 경계와의 관계를 다룹니다. 이를 통해 표준 ML 비기반 시스템에 비해 위성-지구 채널에서 실현 가능한 CV-QKD 시스템을 위한 머신러닝(ML) 기반 접근법을 제시합니다.

- **Performance Highlights**: 제안된 방법은 위성에서 지구로의 채널에서 운영되며, 모델의 복잡성을 줄이면서도 필요한 정확도를 유지하는 성능을 입증하였습니다. 결과적으로, 기존 기술보다 신속하고 안정적인 양자 인터넷 개발에 기여할 수 있는 성과를 얻었습니다.



### AttenGluco: Multimodal Transformer-Based Blood Glucose Forecasting on AI-READI Datas (https://arxiv.org/abs/2502.09919)
- **What's New**: 이 논문에서는 당뇨병 환자의 혈당 수치를 장기적으로 예측하기 위한 새로운 기법인 AttenGluco를 제안합니다. 이 방법은 Cross-Attention과 Multi-Scale Attention을 활용하여 CGM (Continuous Glucose Monitoring) 데이터와 신체 활동 데이터를 효과적으로 통합하고, 다양한 샘플링 주기를 가진 데이터를 융합하는 문제를 해결합니다.

- **Technical Details**: AttenGluco는 Transformer 아키텍처를 기반으로 하며, 각종 생리적 및 행동적 변수를 함께 고려하여 혈당 변화를 예측하는 데 중점을 둡니다. Cross-attention은 장기적인 의존성을 포착할 수 있도록 하며, Multi-scale attention은 외부 시계열 변수가 혈당에 미치는 영향을 분석합니다. 이 모델은 최근에 공개된 AIREADI 데이터 세트에서 다양한 샘플 집단에 대한 예측 실험을 통해 성능을 평가합니다.

- **Performance Highlights**: AttenGluco는 Root Mean Square Error (RMSE), Mean Absolute Error (MAE)와 같은 모든 오류 지표에서 멀티모달 LSTM 모델보다 약 10%에서 15% 개선된 결과를 보여주었습니다. 이 연구는 특히 제2형 당뇨병 환자 집단에 대한 혈당 예측의 정확성을 높이는 데 지대한 기여를 합니다.



### ArchRAG: Attributed Community-based Hierarchical Retrieval-Augmented Generation (https://arxiv.org/abs/2502.09891)
- **What's New**: 이 논문에서는 Attributed Community-based Hierarchical RAG (ArchRAG)라는 새로운 그래프 기반 RAG 접근 방식을 소개합니다. 이는 질문을 절차화하기 위해 속성 커뮤니티(attributed communities)를 활용하며, 새로운 LLM 기반 계층 클러스터링 방법을 도입하여 질문에 대한 가장 관련성 높은 정보를 그래프에서 검색합니다.

- **Technical Details**: ArchRAG는 외부 코퍼스에 기반하여 지식 그래프를 사용하여 attributed communities를 탐지하며, 링크와 노드 속성을 모두 고려하는 방법론을 사용합니다. 이 방법은 정보를 효과적으로 검색할 수 있는 계층적 인덱스 구조를 구축하여, 다양한 추상화 수준에서 정보를 제공합니다.

- **Performance Highlights**: 실험 결과, ArchRAG는 정확도와 토큰 비용 모두에서 기존 방법보다 우수한 성능을 보입니다. 이 새로운 접근법은 특히 시간과 토큰 소모를 줄이면서, 효율적으로 질문에 대한 답변을 생성하는 데 있어 큰 개선을 이룹니다.



### Evaluating and Improving Graph-based Explanation Methods for Multi-Agent Coordination (https://arxiv.org/abs/2502.09889)
Comments:
          19 pages, 8 figures, 6 tables

- **What's New**: 본 연구는 그래프 신경망(GNN)의 설명 기법들이 다중 에이전트 조정을 설명하는 데 적합한지를 조사합니다. 초기 분석을 바탕으로, 기존의 그래프 기반 설명자들이 팀 행동에 영향을 미치는 중요한 통신 채널을 식별할 수 있는 잠재력을 가지고 있음을 발견했습니다. 특히, GAT 기반 정책에 대해 주의 엔트로피 정규화 기법을 제안하여 설명 품질을 향상시키려는 노력을 하고 있습니다.

- **Technical Details**: 다중 에이전트 조정에서 GNN 설명자들의 적합성을 체계적으로 조사했습니다. 세 가지 주요 후처리 그래프 기반 설명 기술인 Graph Mask, GNN-Explainer 및 Attention Explainer를 활용하여 성능 평가를 수행했습니다. 주의 엔트로피 최소화 목표는 에이전트가 가장 영향력 있는 이웃에게 집중하도록 유도하여 그래프 기반 설명자가 문제를 해결하는 데 도움을 줄 수 있도록 설계되었습니다.

- **Performance Highlights**: 세 가지 작업과 다양한 팀 크기에서 실시된 평가를 통해 제안된 정규화 기법이 주의 설명자에서 설명 품질을 크게 향상시킬 수 있음을 입증했습니다. 특히, 주의 엔트로피 최소화가 다른 두 설명자와 결합될 경우 설명 품질의 적어도 한 측면이 향상되는 경향을 보였습니다. 이러한 개선은 과제 성능에 미치는 영향이 미미하여 다중 에이전트 조정을 위한 해석 가능하고 설명 가능한 정책 개발의 기초를 마련합니다.



### Video2Policy: Scaling up Manipulation Tasks in Simulation through Internet Videos (https://arxiv.org/abs/2502.09886)
- **What's New**: Video2Policy라는 새로운 프레임워크를 통해 인터넷 RGB 비디오를 활용하여 현실적인 작업을 기반으로 한 훈련 데이터를 생성하는 방법을 제안합니다. 이 방법은 시뮬레이션에서의 작업 생성과 강화학습을 결합하여, 다양한 인간 행동을 재현하는 100개 이상의 비디오로부터 작업을 재구성합니다. 이를 통해 일반적인 정책을 훈련할 수 있게 하며, 현실 세계와의 성공적인 전환을 보여줍니다.

- **Technical Details**: 이 프레임워크는 두 가지 주요 단계로 구성됩니다. 첫 번째 단계에서는 RGB 비디오에서 작업에 관련된 객체 메쉬를 재구성하고 6D 포즈를 추출합니다. 두 번째 단계에서는 비전-언어 모델(Vision-Language Model, VLM)을 사용하여 시각적 정보와 프롬프트를 기반으로 작업 코드를 작성하고, 생성된 보상 함수를 통해 RL 정책을 학습합니다. 이를 통해, 데이터 생성의 자동화를 통해 비디오 기반의 작업을 시뮬레이션할 수 있습니다.

- **Performance Highlights**: 실험 결과, Video2Policy는 이전 LLM 기반 방법에 비해 월등히 높은 성능을 보이며, 평균 88%의 성공률을 기록했습니다. 훈련된 일반 정책을 기반으로 시뮬레이션에서 10개의 새로운 비디오에 대해 75%의 성공률을 달성하고, 최종적으로 실제 로봇에서 47%의 성공률을 확인했습니다. 이러한 결과는 생성된 시뮬레이션 데이터의 실제 활용 가능성을 입증합니다.



### Comprehensive Review of Neural Differential Equations for Time Series Analysis (https://arxiv.org/abs/2502.09885)
- **What's New**: 이 논문은 Neural Differential Equations (NDEs)를 기반으로 한 시계열 분석 기법을 종합적으로 리뷰하고 있습니다. NDE는 신경망의 유연성과 미분 방정식의 수학적 엄밀성을 결합하여, 연속적인 동작을 모델링하는 새로운 접근 방식을 제공합니다. Neural Ordinary Differential Equations (NODEs), Neural Controlled Differential Equations (NCDEs), Neural Stochastic Differential Equations (NSDEs)에 대한 논의가 포함되어 있으며, 이들의 수학적 공식과 응용을 자세히 설명합니다.

- **Technical Details**: 시계열 모델링은 데이터를 𝒙=(x0,x1,…,xn) 형태로 표현하고, 이를 연속적인 잠재 과정 𝒛⁢(t)로 나타냅니다. NDE는 데이터 생성을 위한 실제 동역학을 학습하며, 불규칙한 샘플링 및 누락된 관측치를 유연하게 처리할 수 있는 프레임워크를 제공합니다. NODE, NCDE 및 NSDE 등의 모델들은 각기 다른 방식으로 연속 시간 동역학을 모델링하며, RNN과는 달리 메모리 효율적으로 작동합니다.

- **Performance Highlights**: NDE 기반 방법론은 복잡한 시계열 문제를 해결하는 데 있어 잠재적인 가능성을 보여줍니다. NDE는 통상적인 이산 시간 아키텍처에 비해 데이터 범위와 동적 특성을 효과적으로 포착할 수 있으며, 다양한 응용 분야에서의 성능 향상을 기대할 수 있습니다. 이는 연구자와 실무자가 NDE를 통해 더 깊은 이해를 가지도록 돕는 토대를 마련합니다.



### Nonasymptotic CLT and Error Bounds for Two-Time-Scale Stochastic Approximation (https://arxiv.org/abs/2502.09884)
- **What's New**: 본 논문은 martingale noise에 의해 구동되는 선형 이중 시간 척도 확률 근사(stochastic approximation, SA) 알고리즘의 유한 시간 오류율에 대한 새로운 통찰을 제공합니다. 기존 연구에서는 대칭적 수렴이나 비최적적 유한 시간 경계에 초점을 두었으나, 본 연구는 Wasserstein-1 거리와 관련된 비비대칭 중심극한정리를 도출했습니다. 이로 인해 Polyak-Ruppert 평균화를 적용한 경우, 기대되는 오류가 1/√n 비율로 감소함을 보였습니다.

- **Technical Details**: 확률 근사(SA) 알고리즘은 샘플 기반의 반복적 접근 방식을 활용하여 미지의 연산자 F의 고정점을 찾습니다. 특히, 본 연구에서는 두 개의 결합된 선형 방정식을 해결하기 위한 선형 이중 시간 척도 SA(TSA)를 다룹니다. TSA는 두 개의 서로 다른 단계 크기를 활용하여 빠른 시간 척도 변수(x)와 느린 시간 척도 변수(y)를 사용하는데, 이는 'Polyak-Ruppert 평균화'를 통해 개선된 수렴 특성을 보여줍니다.

- **Performance Highlights**: 본 논문의 주요 성과 중 하나는 TS-PR(TSA-Polyak-Ruppert) 방법을 통해 도출된 오류율이 1/√n로 감소함을 입증한 것입니다. 이는 이전 연구들보다 향상된 수렴 속도를 보여주며, TSA의 유한 시간 성과를 보다 심층적으로 분석했습니다. 이 결과는 TSA에서 PR 평균화를 활용했을 시 기대되는 오차가 Gaussian 한계로 수렴함을 강조하며, 실질적인 알고리즘 개선으로 이어질 수 있습니다.



### FrGNet: A fourier-guided weakly-supervised framework for nuclear instance segmentation (https://arxiv.org/abs/2502.09874)
- **What's New**: 이번 논문에서는 약한 지도 학습(weakly-supervised) 환경에서 핵(instance) 세그멘테이션을 위한 새로운 Fourier Guidance 프레임워크를 제안합니다. 이 프레임워크는 Fourier Guidance 모듈을 통해 사전 정보(priori information)를 모델 학습 과정에 통합하여, 핵의 관련 특성을 효과적으로 캡처할 수 있도록 합니다. 또한, 가이드 기반 인스턴스 레벨 대비 모듈(GILC)을 통해 핵의 표현 기능을 강화하는 방식을 활용합니다. 실험 결과, 제안한 모델이 최신 최고 성능(State-of-the-Art) 방법들보다 뛰어난 성능을 보임을 확인하였습니다.

- **Technical Details**: 제안한 FrGNet 프레임워크는 핵(inuclear) 이미지를 활용하여 Fourier 변환(fourier transform)으로 생성된 마스크를 가이드로 사용합니다. 이 프레임워크는 핵의 고유한 특성을 고려하여 전통적인 방법들의 한계를 극복합니다. GILC 모듈은 인스턴스(level)의 특성을 학습시켰으며, 모델의 핵 표상(feature representation)을 더욱 향상시키는 데 기여합니다. 이 프레임워크는 두 개의 공공 데이터셋을 통해 효율성과 일반화(generalization) 능력을 입증하였습니다.

- **Performance Highlights**: 모델은 강력한 일반화 능력을 보여주며, 전혀 라벨이 없는 개인 데이터셋에서도 효과적인 핵 세그멘테이션을 수행할 수 있음을 입증했습니다. 약한 지도 학습 실험에서도 소량의 라벨링 데이터만으로 완전 지도 학습(full supervision) 성과에 가까운 결과를 유지하였습니다. 제안한 방법은 핵 세그멘테이션 분야에서 새로운 표준(SOTA) 성능 지표를 설정하였으며, 이는 의학 이미징의 발전에 기여할 것으로 기대됩니다.



### A Taxonomy of Linguistic Expressions That Contribute To Anthropomorphism of Language Technologies (https://arxiv.org/abs/2502.09870)
Comments:
          18 pages, 1 figure, to appear at CHI 2025

- **What's New**: 최근 언어 기술, 특히 대규모 언어 모델(LLMs)의 인간 유사성 때문에 인간과 비인간 객체에 대한 의인화(anthropomorphism)가 더욱 주목받고 있습니다. 이 논문은 언어 기술의 인간 유사성에 대한 명확한 논의와 위험성, 및 효과적인 설계를 위해 텍스트 표현의 분류 체계(taxonomy)를 개발합니다. 저자들은 의인화를 구성하는 다양한 텍스트 표현을 식별하고, 언어 기술 상호작용의 맥락에서 이들 표현의 의미를 분석합니다.

- **Technical Details**: 이 연구에서는 실제 사용자 상호작용 사례를 분석하여 인간 유사성을 가진 텍스트 표현의 19가지 유형을 도출하였습니다. 이러한 표현은 감정적 취약성, 정체성 및 자기 비교를 통해 시스템의 감정적 피해 가능성을 암시합니다. 저자들은 의인화를 인식하는 데 도움을 주기 위해 다섯 가지 관점을 제안하며, 이 관점들은 텍스트 출력에서 나타나는 인간성의 주장도 포함합니다.

- **Performance Highlights**: 저자는 의인화의 긍정적 및 부정적 측면을 이해하는 데 있어 도전과 긴장이 존재함을 강조합니다. 이러한 의인화가 기술에 대한 신뢰를 부여하지만 동시에 사용자와 기술 간의 관계를 왜곡하며, 때로는 실제 인간의 고유성을 저하시킬 수 있는 잠재적 위험도 내포하고 있습니다. 이 연구는 언어 기술의 의인화를 더 효과적으로 논의하고 결정을 내리는 데 도움이 되는 기초를 제공합니다.



### How Users Who are Blind or Low Vision Play Mobile Games: Perceptions, Challenges, and Strategies (https://arxiv.org/abs/2502.09866)
Comments:
          18 pages, 3 figures, Accepted by CHI '25

- **What's New**: 본 연구는 시각 장애 및 저시력(BLV) 게이머들과 모바일 게임 간의 상호작용에 대한 미비한 연구를 충족시키기 위해 수행되었습니다. 32명의 BLV 모바일 플레이어를 대상으로 한 인터뷰를 통해 그들의 경험, 도전 과제 및 상호작용 전략을 조사했습니다. 결과적으로 BLV 플레이어들은 모바일 게임을 통해 지루함을 해소하고 성취감을 얻으며 사회적 연결을 형성하려는 경향을 보였으나, 게임의 접근성 수준에 따라 장애물에 직면하게 됩니다.

- **Technical Details**: 모바일 게임에 대한 BLV 플레이어들의 인식 및 도전 과제를 이해하기 위해 반구조화된 인터뷰를 통해 데이터 수집이 이루어졌습니다. 연구는 두 가지 주요 연구 질문, 즉 BLV 플레이어의 인식 및 다양한 접근성 수준이 게임 경험에 미치는 영향을 다루었습니다. 높은 접근성을 지닌 게임은 깊은 사회적 상호작용을 가능하게 하지만, 플레이어빌리는 낮은 반면 인지적 비용이 증가하는 경향이 있음을 확인했습니다.

- **Performance Highlights**: 모바일 게임은 BLV 플레이어들에게 높은 접근성과 포터블함, 우수한 사회적 상호작용을 가능하게 합니다. 그러나 접근성이 낮은 주류 게임은 함께 참여하기를 거의 불가능하게 하며, 이는 BLV 플레이어들의 심리적 피해로 이어질 수 있습니다. 연구 결과, BLV 플레이어들은 다양한 접근성 수준에서의 게임 경험을 통해 게임 디자인에 대한 기대를 표현하며, 앞으로의 모바일 게임 디자인 개선 방향에 대한 통찰을 제공합니다.



### Automated Hypothesis Validation with Agentic Sequential Falsifications (https://arxiv.org/abs/2502.09858)
- **What's New**: 이 논문에서는 정보 획득과 의사결정 과정에서 가설(hypothesis)의 중요성을 강조하며, 고수준의 추상적인 가설의 자동 검증을 위한 새로운 프레임워크인 Popper를 제안합니다. Popper는 Karl Popper의 반증 원칙(falsification principle)에 기반하여 LLM(대형 언어 모델)을 활용한 자동화된 검증 과정을 수행합니다. 이는 기존의 수동 검증의 비효율성을 해결하려 합니다.

- **Technical Details**: Popper는 가설의 측정 가능한 함의(measurable implications)를 목표로 하는 실험을 설계하고 실행하는 LLM 에이전트(agent)를 사용하여 가설을 검증합니다. 또한, 이 프레임워크는 엄격한 Type-I 오류 통제(strict Type-I error control)를 보장하는 새로운 순차적 테스트(framework for sequential testing)를 도입하여 다양한 관찰로부터 증거를 수집합니다. 기존 데이터에서 또는 새로운 절차를 통해 수집된 증거를 통해 가설의 유효성을 검증합니다.

- **Performance Highlights**: 이 연구에서는 생물학(biology), 경제학(economics), 사회학(sociology) 등 여섯 가지 분야에서 Popper의 성능을 입증하였습니다. Popper는 오류 통제를 강화하고 높은 검정력을 유지하며, 기존의 인간 과학자들에 비해 복잡한 생물학적 가설을 검증하는 데 소요되는 시간을 10배 단축시키며 유사한 성능을 보였습니다. 이는 가설 검증을 위한 확장 가능하고 엄격한 솔루션을 제공합니다.



### Efficient Multitask Learning in Small Language Models Through Upside-Down Reinforcement Learning (https://arxiv.org/abs/2502.09854)
- **What's New**: 본 연구에서는 소형 언어 모델(SLM), 특히 100M 매개변수를 가진 GPT-2 모델이 대형 언어 모델(LLM)과 비교하여 멀티태스크 프롬프트 생성 작업에서 경쟁력 있는 성능을 낼 수 있음을 보여줍니다. 이는 업사이드-다운 강화 학습(upside-down reinforcement learning)과 강력한 LLM인 Llama-3의 합성 데이터 증류(synthetic data distillation)의 독창적 조합을 통해 이루어졌으며, 이는 80배 더 작은 SLM이 최신 기술 모델과 거의 동일한 관련성 점수를 달성하는 데 기여했습니다.

- **Technical Details**: SLM은 멀티모달 프롬프트 생성 작업을 위한 효율적이고 효과적인 멀티태스크 학습자로서 활용될 수 있도록 설계되었습니다. 연구진은 Llama-3를 사용하여 고품질의 훈련 데이터 집합을 만들고, 이를 통해 SLM은 최소한의 리소스에서 효과적으로 학습하도록 구성되었습니다. 업사이드-다운 강화 학습을 통해 SLM은 제어된 생성 프로세스를 최적화하여 특정 속성(예: 길이 및 관련성)을 만족하는 출력을 생성하도록 훈련됩니다.

- **Performance Highlights**: SLM은 Llama-3를 포함한 최신 LLM과 비교하여 매개변수 수가 약 1/80에 불과하지만 유사한 성능을 달성하며, 단일 A10G GPU에서 초당 338개의 토큰을 처리할 수 있는 속도를 기록했습니다. 이러한 성능은 자원이 제한된 환경에서 실시간 응용 프로그램의 요구사항을 충족시키기에 적합합니다. 연구진은 이 프레임워크가 상업적인 텍스트-이미지 생성 시스템과 통합될 수 있는 가능성을 강조하여 실용적인 응용 프로그램에서의 활용도를 높이고 있습니다.



### HealthGPT: A Medical Large Vision-Language Model for Unifying Comprehension and Generation via Heterogeneous Knowledge Adaptation (https://arxiv.org/abs/2502.09838)
- **What's New**: 이 논문에서는 강력한 의료 대형 비전-언어 모델인 HealthGPT를 소개합니다. 이 모델은 의료 시각 이해(comprehension) 및 생성(generation) 기능을 통합하여 자동 회귀(autoregressive) 패러다임 내에서 작동합니다. 새로운 기법인 Heterogeneous Low-Rank Adaptation(H-LoRA)을 통해 의료 도메인에 특화된 데이터 세트인 VL-Health를 활용하여 모델의 학습을 진행합니다.

- **Technical Details**: HealthGPT의 학습 프로세스는 H-LoRA와 Hierarchical Visual Perception(HVP), 그리고 Three-stage Learning Strategy(TLS)로 구성되어 있습니다. H-LoRA는 이해와 생성 작업의 학습 프로세스를 분리하여 상반된 요구 사항 사이의 충돌을 피합니다. HVP는 Vision Transformer(ViT)의 레이어를 활용하여 두 작업의 시각 세부 정보를 효과적으로 관리하며, TLS는 이해 및 생성 지식을 통합하여 고유한 LVLM(Cross-modal Vision-Language Model)을 구축합니다.

- **Performance Highlights**: 실험 결과, HealthGPT는 데이터 제약이 있는 상황에서 다양한 의료 다중 모달(multi-modal) 작업을 통합할 수 있는 능력을 입증했습니다. 기존 최신 모델(SOTA)과 비교할 때 동등하거나 더 나은 성능을 달성하며, 이 연구의 주요 기여는 통합된 Med-LVLM을 제안한 것과 데이터 충돌 문제를 완화하기 위한 H-LoRA 아키텍처를 도입한 것입니다.



### Efficient Evaluation of Multi-Task Robot Policies With Active Experiment Selection (https://arxiv.org/abs/2502.09829)
- **What's New**: 이번 논문은 로봇 정책 평가를 능동적인 테스트 문제로 모델링하여, 다양한 작업과 정책에 대한 성능을 보다 효율적으로 평가할 수 있는 새로운 접근 방식을 제안합니다. 각 작업 간의 유사성을 활용하여 정책 행동의 잠재적인 관계를 밝히는 것에 중점을 두고 있으며, 자연어를 이러한 관계를 모델링하는 데 유용한 사전 정보로 사용합니다. 이를 통해 실험자의 노력을 줄이고, 더욱 정보성이 높은 실험을 효율적으로 선택할 수 있는 전략을 개발합니다.

- **Technical Details**: 제안된 프레임워크에서는 연속적인 성능 결과와 이산적인 성능 결과를 모두 처리할 수 있습니다. 이 논문에서는 로봇의 성능 분포를 추정하기 위해 실험 순차 실행 중 잠재적 작업 및 정책 임베딩에 조건화된 대리 모델을 학습합니다. 수집된 데이터의 흐름을 통해, 로봇 정책의 성능 평가를 위한 확률적 개념을 도입하고, 비용 효과적인 실험 샘플링을 위한 히뉴리스틱을 통합합니다.

- **Performance Highlights**: 실험 결과에 따르면, 정보성을 우선시하는 샘플링을 통해 로봇 정책 전반에 걸쳐 평가 메트릭스를 계산하는 비용을 줄일 수 있음을 보여줍니다. 실제 로봇과 시뮬레이션의 기존 평가 데이터에서 다양한 실험을 수행하여, 정책-작업 쌍에 대한 성능 추정의 정확성을 높이고 효과적인 정책 평가를 가능하게 합니다. 이 접근법은 다중 작업 로봇 정책의 효율적 평가를 가능케 하여, 전통적인 방법보다 낮은 비용으로도 높은 성과를 달성할 수 있게 합니다.



### A Solver-Aided Hierarchical Language for LLM-Driven CAD Design (https://arxiv.org/abs/2502.09819)
- **What's New**: 이번 연구에서는 AIDL(AI Design Language)이라는 새로운 계층적 도메인 특화 언어(DSL)를 도입함으로써, 대형 언어 모델(LLM)을 이용한 CAD 설계를 가능하게 합니다. AIDL은 기하학적 제약 해결기에 공간적 추론을 위임하여, 모델이 복잡한 기하를 생성할 수 있도록 돕습니다. 실험 결과, AIDL은 이전의 CAD 언어와 비교하여 더 나은 시각적 결과를 제공하며, 포스트 프로세싱과 추론이 용이한 객체 생성에서 우수한 성능을 보입니다.

- **Technical Details**: AIDL은 기하학적 구조를 직접 생성하는 대신, 이를 생성하는 CAD 프로그램을 생성하는 방식으로 작동합니다. 이 DSL은 고수준 추론에 집중하고, 정밀성을 요구하는 계산 작업은 외부 솔버에 위임하는 방법론을 제안합니다. 또한, AIDL은 기존 CAD 설계의 제약조건과 의존성을 효과적으로 처리할 수 있는 계층적 접근 방식을 채택하여, 보다 높은 유연성과 편집 가능성을 제공합니다.

- **Performance Highlights**: AIDL을 사용하여 생성한 2D CAD 결과물은 OpenSCAD와 비교하여 시각적으로 동등하거나 더 나은 품질을 보였습니다. AIDL 언어의 계층 구조와 제약 조건 도입은 복잡한 다중 부품 객체를 정밀하게 구성할 수 있도록 돕습니다. 평가 결과, AIDL은 기존 CAD 모델링 언어에서 제공하는 정도 이상의 성능을 발휘하며, 언어 설계만으로도 LLM의 CAD 생성 성능을 크게 향상시킬 수 있다는 점을 입증했습니다.



### AgentGuard: Repurposing Agentic Orchestrator for Safety Evaluation of Tool Orchestration (https://arxiv.org/abs/2502.09809)
Comments:
          Project report of AgentGuard in LLM Agent MOOC Hackathon hosted by UC Berkeley in 2024

- **What's New**: AgentGuard는 대형 언어 모델(LLM)의 도구 사용을 통합하여 위험한 도구 사용 워크플로우를 자율적으로 발견하고 검증하는 프레임워크입니다. 이를 통해 에이전트의 행동이 안전하게 제한되도록 안전 제약 조건을 생성하여 안전성을 보장합니다. 본 연구는 LLM 오케스트레이터의 내재적 능력을 활용하여 이러한 안전 평가를 자율적으로 수행할 수 있는 방안을 제시합니다.

- **Technical Details**: 이 프레임워크는 네 가지 주요 단계로 작동합니다: 1) 위험한 워크플로우 식별, 2) 실제 실행에서의 검증, 3) 안전 제약 조건 생성, 4) 제약의 효율성 검증. AgentGuard는 LLM 기반의 오케스트레이터를 사용하여 다양한 작업 시나리오를 처리하며, 각 단계에서 결과를 종합하여 평가 보고서를 생성합니다. 평가 보고서는 안전하지 않은 워크플로우와 테스트 케이스, 검증된 제약 조건을 포함합니다.

- **Performance Highlights**: 실험 결과, AgentGuard는 기존 LLM 기반 접근 방식의 한계를 극복하며, 신뢰성 있는 안전성 평가를 수행할 수 있음을 입증했습니다. 본 연구는 LLM 에이전트를 위한 표준화된 테스트 및 강화 절차의 설립을 촉구하고 있습니다. AgentGuard는 안전하지 않은 행동을 줄이기 위한 언어 모델의 신뢰성을 높일 수 있는 다양한 응용 프로그램을 제공합니다.



### Acute Lymphoblastic Leukemia Diagnosis Employing YOLOv11, YOLOv8, ResNet50, and Inception-ResNet-v2 Deep Learning Models (https://arxiv.org/abs/2502.09804)
Comments:
          12 pages, 28 figures, 5 tables

- **What's New**: 이 연구는 혈액암 탐지에 있어 YOLOv11을 활용한 최초의 연구로, AI 모형이 백혈구의 악성 여부를 판별하고 다양한 ALL 단계(급성 림프모구 백혈병)를 식별할 수 있는 가능성을 보여줍니다. 또한, 악성으로 오분류되는 경우가 잦은 Hematogones와 같은 세포도 탐지할 수 있는 모델을 제안합니다. 이 연구는 다세포 샘플을 통해 실제 환경을 더 잘 반영하여 AI 모델이 높은 정확도를 유지할 수 있도록 개발되었습니다.

- **Technical Details**: 이 연구에서는 YOLOv8, YOLOv11, ResNet50 및 Inception-ResNet-v2 모델을 활용하여 이미지를 처리하고 학습하는 방법을 채택했습니다. 데이터 준비 과정에서 이미지 분할 및 데이터 증강 기법을 사용하여 흰 혈구에 초점을 맞추고 모델의 성능을 향상시킵니다. 최종적으로는 두 개의 클래스(Normal 및 Cancer)로 통합된 데이터셋을 사용하여 AI 모델을 학습시켰습니다.

- **Performance Highlights**: 연구에서 사용된 고급 딥러닝 모델들은 최대 99.7%의 높은 정확도를 기록하며, 다양한 데이터셋과 실제 환경에서도 효과적인 성능을 입증합니다. 특히, YOLOv11과 YOLOv8 모델이 채택되었으며, 모든 모델은 97%에서 99% 사이의 초기 결과를 보여주었습니다. 이러한 결과는 백혈병 탐지의 효율성을 크게 향상시킬 가능성을 엿볼 수 있게 해줍니다.



### Co-designing Large Language Model Tools for Project-Based Learning with K12 Educators (https://arxiv.org/abs/2502.09799)
Comments:
          25 pages

- **What's New**: 이 연구는 LLMs(대형 언어 모델)를 PBL(프로젝트 기반 학습)에 통합하기 위해 K-12 교육자들과의 공동 설계 프로세스를 문서화합니다. 연구를 통해 교육자들이 직면한 현재의 PBL 문제를 탐색하고 LLM이 이를 해결하는 데 도움을 줄 수 있는 방법을 제시합니다. 특히 LLM이 반복적인 작업을 자동화하고 개인화된 학습 경험을 향상시킬 수 있는 잠재력에 주목합니다.

- **Technical Details**: PBL은 학생 중심의 학습 방식을 채택하고, 조사 및 협업을 통해 지식을 적극적으로 구성합니다. 이 연구에서는 교육자들이 PBL에서 이루는 평가 방법, 즉 팀워크, 창의성 및 문제 해결 능력에 대한 전반적인 학습 결과를 어떻게 평가하는지에 대한 구체적인 도전 과제를 조사합니다. 설계된 LLM 도구는 교육자의 현재 역할을 보완하며, 교육자들의 전문적 성장을 지원하는 방향으로 통합될 것입니다.

- **Performance Highlights**: 이 연구는 PBL과 LLM을 결합하기 위한 설계 가이드라인을 제시하여 교육 기술 디자이너 및 교육자들이 학생의 진척을 평가하고 학습 목표를 설정하는 데 도움을 줄 수 있습니다. 특히 다양한 교육자의 의견을 반영하여 반복적인 피드백을 통해 교육적 요구를 충족시키는 LLM 도구의 통합 가능성을 강조합니다. 연구 결과는 PBL에서 LLM의 미래 활용 방법을 형성하는 데 중요한 통찰력을 제공합니다.



### A Survey on LLM-based News Recommender Systems (https://arxiv.org/abs/2502.09797)
- **What's New**: 이 논문은 LLM(대규모 언어 모델) 기반 뉴스 추천 시스템에 대한 체계적인 조사 연구를 처음으로 수행하였습니다. 연구는 DLLM(구별 가능한 대규모 언어 모델)과 GLLM(생성 가능한 대규모 언어 모델)의 접근 방식을 구체적으로 분류하여 분석합니다. 논문의 주요 기여로는 다양한 LLM 기반 뉴스 추천 모델을 텍스트 모델링, 사용자 모델링 및 예측 모델링의 세 가지 측면에서 검토하고 비교하는 것입니다.

- **Technical Details**: 해당 연구는 NLP에서의 딥러닝 기술 개발과 LLM의 발전을 기반으로 다양한 뉴스 추천 시스템의 메커니즘을 탐색합니다. CNN, RNN, GNN 등 다양한 딥러닝 프레임워크를 통해 뉴스와 사용자 정보를 효과적으로 모델링하며, DLLM 및 GLLM을 뉴스 인코더로 활용하여 성능 향상을 도모합니다. 연구에서는 특히 각 추천 시스템의 성능을 분류 지표(classification metrics), 랭킹 지표(ranking metrics), 다양성 지표(diversity metrics), 개인화 지표(personalization metrics) 등의 다양한 관점에서 평가합니다.

- **Performance Highlights**: 최근 GLLM 기반 뉴스 추천 시스템들의 빠른 성장으로, 이들은 차가운 시작(cold-start) 문제를 완화하고 보다 정확한 뉴스 특징 탐색 및 사용자 관심 모델링에 뛰어난 성과를 보이고 있습니다. 그러나 GLLM 기반 시스템은 상당한 훈련 시간과 자원을 필요로 하며, 기존의 일반적인 딥러닝 방법들과 비교해 성능 면에서 나은 결과를 달성합니다. 이에 따라 논문은 향후 LLM의 시대에서 뉴스 추천의 방향성을 포괄적으로 탐구합니다.



### TableTalk: Scaffolding Spreadsheet Development with a Language Agen (https://arxiv.org/abs/2502.09787)
- **What's New**: 새로운 연구에서는 TableTalk이라는 언어 에이전트(Agent)를 소개합니다. 이 에이전트는 프로그래머들이 스프레드시트를 대화식으로 구축할 수 있도록 돕습니다. TableTalk은 스프레드시트 개발을 단계별로 구조화하고 사용자가 선택할 수 있는 세 가지 다음 단계를 제안합니다.

- **Technical Details**: TableTalk의 설계는 세 가지 원칙인 스캐폴딩(scaffolding), 유연성(flexibility), 점진성(incrementality)에 기반합니다. 이 원칙은 7명의 프로그래머와 62개의 Excel 템플릿을 연구하여 도출되었습니다. 사용자는 TableTalk을 통해 점진적으로 스프레드시트를 구축할 수 있으며, 통합된 도구를 사용하여 간편하게 작업을 진행할 수 있습니다.

- **Performance Highlights**: 사용자 연구 결과, TableTalk은 기준(베이스라인) 에이전트보다 2.3배 더 선호되는 스프레드시트를 생성하는 것으로 나타났습니다. 또한 인지적 부담(cognitive load)을 12.6% 줄이고 스프레드시트 작업에 소요되는 시간을 단축하는 데 기여했습니다. 이러한 결과는 인간-에이전트 협업의 새로운 가능성을 보여줍니다.



### Improving Acoustic Side-Channel Attacks on Keyboards Using Transformers and Large Language Models (https://arxiv.org/abs/2502.09782)
- **What's New**: 이 연구에서는 다양한 딥러닝 기법, 특히 Vision Transformers(VTs)와 Large Language Models(LLMs)를 활용하여 Acoustic Side-Channel Attacks(ASCAs)의 효과성과 적용 가능성을 높이고 있음을 보여줍니다. 본 논문에서 제안한 CoAtNet 모델은 이전 연구보다 향상된 성능을 보이며, 스마트폰과 Zoom을 통한 타이핑 인식 정확도가 각각 5.0%와 5.9% 개선되었습니다. LLM을 사용한 오류 수정 및 잡음 완화 방법을 통해 실환경에서의 ASCA 성능을 강화하고, 더욱 경쟁력 있는 경량 모델을 활용하여 더욱 효율적인 공격 방어를 구현하고 있습니다.

- **Technical Details**: ASCAs는 디바이스로부터의 음파를 이용해 민감한 정보를 추론하는 공격으로, VTs와 LLMs의 융합을 통해 이러한 공격 기법을 혁신적으로 개선할 수 있는 가능성을 보여줍니다. 본 연구에서는 다양한 모델을 비교하고, 특히 CoAtNet 모델을 활용하여 키 입력의 음향 표본을 분석하며, 효과적인 분류 성과를 기록합니다. 강화된 노이즈 완화 전략은 소음이 심한 환경에서도 오류를 탐지하고 수정함으로써 안정적인 키스트로크 인식과 데이터 복구를 이끌어냅니다.

- **Performance Highlights**: 본 연구에서 사용된 CoAtNet 모델은 이전의 최고 성능을 가진 모델과 비교하여 성능이 향상되었으며, LLM을 활용한 노이즈 완화 기법 덕분에 실생활에서의 ASCA 성능이 크게 향상되었습니다. 또한, Low-Rank Adaptation(LoRA) 방식으로 경량화된 모델이 기존의 무거운 모델들과 비교할 만한 성능을 발휘함을 확인했습니다. 이러한 성과들은 ASCAs에 대한 대응책을 개선하고, 향후 데이터 보안을 강화하는 데 기여할 것으로 기대됩니다.



### Incentivize without Bonus: Provably Efficient Model-based Online Multi-agent RL for Markov Games (https://arxiv.org/abs/2502.09780)
- **What's New**: 이 논문에서는 VMG(Value-incentivized Markov Game solver)라는 새로운 모델 기반 알고리즘을 제안합니다. VMG는 다른 에이전트의 정책을 고정하면서 모든 플레이어의 집합적 최적 반응(value)을 높이는 방향으로 모델 매개변수의 경험적 추정을 편향시켜 탐색을 장려합니다. 이 접근 방식은 기존의 복잡한 불확실성 추정 없이도 다양한 함수 근사(Function Approximation)에서 효과적으로 작동하는 특징이 있습니다.

- **Technical Details**: 제안된 VMG는 많은 수의 에이전트가 상호 작용하는 환경에서, 에이전트 간의 복잡한 조정을 필요로 하지 않게 해줍니다. 또한, VMG는 단순한 전략 변경과 독립적인 정책 업데이트를 가능하게 하여, 에이전트 수가 늘어날 때 유용합니다. 이론적으로, VMG는 선형 함수 근사 하에서 두 플레이어 제로섬 및 다중 플레이어 일반합 Markov 게임의 NE 및 CCE를 찾는 데 근접 최적의 안 좋은 결과를 달성합니다.

- **Performance Highlights**: VMG는 두 플레이어 제로섬 행렬 게임에 대해 O(d√T) 형태의 근접 최적의 결과를 제공합니다. 이는 샘플 복잡도가 O(d^2/ε^2)로, ε-최적의 NE를 찾기 위한 복잡성과도 일치합니다. 다중 플레이어 일반합 Markov 게임의 경우, VMG는 O(Nd^2H^3/ε^2) 또는 O(Nd^2H^4/ε^2)의 샘플 복잡도를 달성하여, 효과적인 성능을 입증합니다.



### On the existence of EFX allocations in multigraphs (https://arxiv.org/abs/2502.09777)
- **What's New**: 이번 논문에서는 여러 에이전트에게 indivisible goods를 공정하게 분배하는 문제를 다루며, envy-free up to any good (EFX) 할당이 항상 존재하는 조건을 제시합니다. 특히, bipartite multigraphs 또는 에이전트 당 최대 ⌈n/4⌉-1개의 이웃이 있는 경우, 또는 가장 짧은 비평행 엣지를 가진 사이클의 길이가 6 이상인 경우에 해당합니다. 이러한 조건에서 EFX 할당이 항상 존재함을 증명하여 Fair Division의 기존 이론을 확장시킵니다.

- **Technical Details**: 이 연구는 다양하고 실용적인 적용을 가지고 있으며, 특히 유산 분배, 학생의 과목 배정 등의 예가 있습니다. 기존 문헌에서는 EFX 할당이 여러 조건에서 존재함이 알려져 있으나, 이 논문은 multigraph 설정 아래에서 일반적인 monotone valuations을 사용하여 보다 포괄적인 결과를 제시합니다. 본 결과로써 EFX 할당의 존재성에 대한 오랜 의문을 새롭게 정리합니다.

- **Performance Highlights**: 본 연구는 bipartite multigraphs에서 EFX 할당이 항상 존재한다는 것을 최초로 입증하였고, 에이전트가 제한된 이웃 수를 가질 때나 특정 조건을 충족할 때 EFX 할당의 존재를 일반화하는 결과를 도출하였습니다. 또한, 이미 독립적으로 수행된 관련 연구들에 대한 비교를 통해, 이 연구가 Fair Division 분야에 기여하고 있음을 보여줍니다.



### Non-Markovian Discrete Diffusion with Causal Language Models (https://arxiv.org/abs/2502.09767)
Comments:
          Under Review

- **What's New**: 이번 연구에서는 CaDDi라는 새로운 causal discrete diffusion model을 소개합니다. 이 모델은 기존의 autoregressive transformers의 우수한 성능을 활용하면서, 비마르코프적(non-Markovian) 성질을 기반으로 하여 모든 생성 경로에 대한 정보를 통합합니다. CaDDi는 사전 훈련된 대형 언어 모델(LLM)을 수정 없이도 쉽게 적용할 수 있도록 해주며, 더 적절하고 제어 가능한 생성을 가능하게 합니다.

- **Technical Details**: CaDDi는 기존의 전통적인 diffusion 모델을 확장하여 비마르코프적인 설정에서 동작합니다. 각 디노이징 단계에서자는 단일 상태가 아닌 전체 생성 경로의 정보를 활용하여 오류 누적을 방지합니다. 이 모델은 또한 전통적인 causal language models을 일반화하며, 훈련 과정에서 다음 토큰 예측 손실(next-token prediction loss)을 통해 효율적으로 훈련될 수 있습니다.

- **Performance Highlights**: 실험에 따르면, CaDDi는 최신 discrete diffusion 모델들과 비교했을 때 자연어와 생물학적 시퀀스 작업 모두에서 더 뛰어난 성능을 보여주었습니다. 이 모델은 제어 가능한 생성 능력을 제공하며, 구조화된 시퀀스 모델링을 위한 강력한 대안이 됩니다. CaDDi는 또한 자동 회귀 모델들에 비해 더 높은 생성 품질을 달성하여, diffusion 기반 방법들과 대규모 자기 회귀 변환기들 간의 간극을 좁히는 데 기여합니다.



### Differential Adjusted Parity for Learning Fair Representations (https://arxiv.org/abs/2502.09765)
- **What's New**: 이번 논문은 공정하고 편향되지 않은 머신러닝 모델 개발을 위한 새로운 접근법인 Differential Adjusted Parity (DAP) 손실 함수를 소개합니다. 이 방법은 조정된 패리티 메트릭의 미분 가능한 변형을 사용하여 공정성을 유지하면서도 정보 표현을 생성합니다. DAP는 다운스트림 태스크의 분류 정확도와 민감한 특성 도메인 간의 일관성을 결합하여 편향을 완화하고 성능을 향상시키는 단일 도구로 기능합니다.

- **Technical Details**: DAP 손실 함수는 소프트 밸런스 정확도를 포함하여, 민감한 특성 간의 불일치 없이도 성능을 증대시킬 수 있도록 설계되었습니다. 비대립적(non-adversarial) 방법으로서, DAP는 다양한 민감한 도메인에서 동등한 저조함을 피하는 특징을 가지고 있습니다. 이 방법은 다중 클래스 문제에까지 확장 가능하여 여러 민감한 특성에서 공정성을 보장합니다.

- **Performance Highlights**: 분석 결과, DAP는 기존의 여러 대립 모델들보다 다운스트림 태스크의 정확도 및 공정성 측면에서 뛰어난 성능을 보여줍니다. 특히, 인구 통계학적 패리티, 동등화된 확률, 민감한 특성 정확도 측면에서 각각 22.5%, 44.1%, 40.1% 향상된 성과를 달성하였습니다. DAP 손실 함수와 그에 관련된 메트릭은 더 공정한 머신러닝 모델을 만드는 데 중요한 역할을 할 것으로 보입니다.



### Adaptive Teaming in Multi-Drone Pursuit: Simulation, Training, and Deploymen (https://arxiv.org/abs/2502.09762)
Comments:
          17 pages

- **What's New**: 이번 논문에서는 드론 협업에서 새로운 개념인 Adaptive Teaming in Multi-Drone Pursuit (AT-MDP) 문제를 정의하고 체계적인 AT-MDP 프레임워크를 제안하였습니다. AT-MDP 프레임워크는 시뮬레이션, 알고리즘 훈련, 실세계 배치를 통합하며, 다양한 환경에서 드론이 미지의 팀원과 효과적으로 협력할 수 있도록 지원합니다. 이 연구는 복잡한 현실 세계 드론 작업에서 연속적 행동을 통한 협업을 가능하게 하는 최초의 프레임워크입니다.

- **Technical Details**: AT-MDP 문제는 N개의 드론 에이전트가 미지의 M팀원과 동적으로 협력하여 K개의 목표를 추적하는 과정을 포함합니다. 본 논문에서는 이러한 문제를 AT-Dec-POMDP (Adaptive Teaming Decentralized Partially Observable Markov Decision Process)로 모델링하고, 다양한 시뮬레이션 환경을 제공하여 드론 간의 실시간 협업을 검증했습니다. 또한, 알고리즘 학습을 위한 분산훈련 프레임워크와 미지 드론 환경을 접목하여 팀원 행동에 대한 일반화를 평가했습니다.

- **Performance Highlights**: 다양한 난이도의 4개 멀티 드론 추적 환경에서 수행된 실험을 통해 AT-MDP 프레임워크의 효과성을 입증하였습니다. 제안된 기법은 기존 접근 방식에 비해 협력 성공률 및 적응성이 높아, 실제 Crazyflie 드론에서의 배치에서도 그 타당성을 확인했습니다. 실험 결과는 복잡한 조건에서도 적응 전략의 출현을 보여주며, 향후 알고리즘 개선과 현실감 있는 과제 시나리오의 발전 가능성을 제시합니다.



### The AI-Therapist Duo: Exploring the Potential of Human-AI Collaboration in Personalized Art Therapy for PICS Intervention (https://arxiv.org/abs/2502.09757)
- **What's New**: 이 논문은 Post-intensive care syndrome (PICS)에 대한 새로운 치료 접근법으로, 인간-인공지능(Human-AI) 협업을 통해 개인 맞춤화된 미술 치료를 제안합니다. 전통적인 미술 치료는 시간이 많이 소요되고 각 환자의 독특한 필요를 충분히 충족하지 못하는 한계가 있습니다. 이러한 맥락에서 AI 기반 시각 예술 추천 시스템(Visual Art Recommendation Systems, VA RecSys)을 활용하여 치료 효과성을 높이고 치료사의 부담을 경감할 수 있는 방안을 모색하고 있습니다.

- **Technical Details**: 저자들은 두 가지 유형의 인간-인공지능 협업 개인화 방법(Human-in-the-Loop, HITL)을 개발했습니다. 첫 번째는 시각 전용 방식으로 ResNet-50 아키텍처를 사용하고, 두 번째는 다중 모달(multimodal) 접근 방식으로 BLIP을 활용합니다. 이를 통해 PICS 환자에게 더욱 적합한 미술 치료 세션을 제공하는 것을 목표로 하며, 치료 세션은 환자의 치유 여정에 공감하는 그림을 찾는 것부터 시작됩니다.

- **Performance Highlights**: 대규모 사용자 연구(N=150)를 통해 이 연구의 결과는 인간-인공지능 협업이 미술 치료의 개인화와 효과성을 향상시키며, 치료사의 업무를 간소화하는 데 기여할 수 있음을 보여줍니다. 이러한 접근법은 PICS 개입을 넘어 불안과 우울증과 같은 다른 분야에서도 응용 가능성을 시사합니다. 따라서, 이 연구는 감정적 지원이 중요한 여러 분야에서의 인공지능 사용 가능성을 강조합니다.



### Vote-Tree-Planner: Optimizing Execution Order in LLM-based Task Planning Pipeline via Voting (https://arxiv.org/abs/2502.09749)
Comments:
          Accepted to RSS24-W: TaskSpec

- **What's New**: 이번 논문에서는 Vote-Tree-Planner라는 새로운 계획 메커니즘을 소개합니다. 이 방법은 Prog-Prompt와 Tree-Planner의 고급 개념을 결합하여 LLM 기반의 계획 시스템에서 생성된 계획의 실행 가능성과 신뢰성을 향상시킵니다. 기존의 반복적인 쿼리 문제를 해결하고, 계획의 효과성을 높이는 데 중점을 두고 있습니다.

- **Technical Details**: Vote-Tree-Planner는 고수준 지시 사항을 중간 명령으로 분해하는 샘플링 프로세스를 사용합니다. 구체적으로, 이 과정은 Prog-Prompt 형식으로 고수준 지시를 변환하고, LLM을 활용하여 다수의 잠재 계획을 생성한 후, 고유한 명령을 추출합니다. 마지막 단계에서 새로운 계획을 트리 형태로 구성하여 각 노드에서 투표를 통해 의사 결정을 지원합니다.

- **Performance Highlights**: 실험 결과 Vote-Tree-Planner는 이전의 기준 방법들에 비해 더 높은 평균 성공률과 목표 조건 회수율을 보여 안정성이 뛰어난 것으로 나타났습니다. 이 연구는 LLM 기반의 계획 시스템에서 계획의 정확성, 신뢰성 및 효율성을 촉진할 수 있는 Vote-Tree-Planner의 잠재력을 강조합니다.



### A CNN Approach to Automated Detection and Classification of Brain Tumors (https://arxiv.org/abs/2502.09731)
- **What's New**: 이 연구는 뇌 종양의 진단을 개선하기 위해 MRI(자기공명영상) 이미지를 처리하는 새로운 방법론을 제시합니다. Anisotropic diffusion filter를 사용한 이미지의 디노이징 기법과 다양한 딥러닝 모델(CNN)들을 적용하여 종양의 분류 정확도를 높이고자 합니다. 연구는 전이 학습 및 데이터 증강(SMOTE)을 통해 3,264개의 MRI 스캔 이미지를 활용한 뇌 종양 식별을 목표로 하고 있습니다. 시각적으로 자세한 MRI 이미지를 활용하여 뇌 질병을 검출하는 데 도움이 될 것입니다.

- **Technical Details**: 연구에서 사용된 MRI 데이터셋은 공개적으로 접근 가능한 Brain Tumour Classification 데이터베이스로, 딥러닝 모델을 학습시키기 위해 3,264개의 뇌 MRI 스캔 이미지로 구성되어 있습니다. 모델로는 ResNet152V2, VGG, ViT 및 EfficientNet이 포함되며, EfficientNet이 98%라는 최고의 정확도를 달성했습니다. 디노이징과 이미지 분류 외에도, f1 score, confusion matrix 및 Receiver Operating Characteristic (ROC) 곡선 등을 사용하여 모델의 성능을 평가합니다.

- **Performance Highlights**: EfficientNet 모델이 뇌 종양 분류에서 98%의 정확도로 최고의 성능을 보였으며, 이는 기존 연구들에서 보고된 정확도보다 높은 수치입니다. 또한 연구는 다양한 뇌 종양 유형(글리오마, 무 종양, 수막종 및 뇌하수체 종양)을 정확하게 분류하는 데 초점을 맞추고 있으며, 이는 임상 진단과 치료 계획 수립에 적극적으로 기여할 것입니다. 이 연구는 딥러닝 기법을 통해 뇌 종양 진단의 신뢰성과 속도를 높이는 데 큰 기여를 할 것으로 기대됩니다.



### Making Them a Malicious Database: Exploiting Query Code to Jailbreak Aligned Large Language Models (https://arxiv.org/abs/2502.09723)
Comments:
          15 pages, 11 figures

- **What's New**: 이번 논문에서는 QueryAttack이라는 새로운 공격 프레임워크를 제안하여 대형 언어 모델(LLM)의 안전성 정렬(safety alignment) 일반화 가능성을 체계적으로 조사합니다. 이 방법은 LLM을 지식 데이터베이스로 간주하고, 악의적인 쿼리를 자연어에서 코드 스타일의 구조화된 쿼리로 변환하여 안전성 정렬 메커니즘을 우회합니다. 실험 결과, QueryAttack은 다양한 개발자와 기능을 가진 LLM에서 높은 공격 성공률(ASR)을 달성함을 보여줍니다.

- **Technical Details**: QueryAttack의 구조는 세 가지 주요 구성 요소로 이루어집니다: (1) 비자연어 기반의 쿼리 형식을 정의하고 자연어 쿼리를 대상 비자연어 형식으로 변환하는 번역기 사용, (2) 비자연어를 이해하고 자연어로 응답을 제공할 수 있도록 대상 LLM을 가이드하는 프롬프트 구성, (3) 변환된 쿼리를 사용하여 모델이 해로운 응답을 생성하도록 유도합니다. 이는 데이터베이스에서 데이터를 쿼리하는 것처럼 쿼리 작업을 정의하는 방식으로, 프로그래밍 언어를 활용하여 적절한 쿼리 형식을 구성합니다.

- **Performance Highlights**: QueryAttack을 통해 수행한 다양한 실험에서는 LLM의 보안 방어를 효과적으로 우회함을 입증하였습니다. 본 연구는 기존의 RLHF 기반 방어 메커니즘이 비자연어 입력을 사용하는 탈옥 공격에 대한 완벽한 방어를 수행할 수 없다는 가설을 세우고, 이 제한점을 극복하기 위한 새로운 접근 방식을 탐구합니다. 또한, 제안된 방어 방법은 GPT-4-1106에서 최대 64%까지 ASR을 줄이는 것으로 평가되었습니다.



### Evaluating GPT's Capability in Identifying Stages of Cognitive Impairment from Electronic Health Data (https://arxiv.org/abs/2502.09715)
Comments:
          Findings paper presented at Machine Learning for Health (ML4H) symposium 2024, December 15-16, 2024, Vancouver, Canada, 7 pages

- **What's New**: 이번 연구에서는 자동화된 접근법으로 전자 건강 기록(EHR)에서 인지 장애(cognitive impairment)를 파악하는 방법을 평가합니다. 특히, 최신 GPT-4o 모델을 사용하여 전문가 노트를 분석하고, 치매 진단의 초기 단계를 자동으로 결정하는 것을 목표로 했습니다. 자동화된 접근은 수작업 차트 검토의 시간 소모와 오류를 줄이고, 향후 연구 데이터셋을 생성하는 데 기여할 수 있습니다.

- **Technical Details**: 연구는 두 가지 작업에 대해 GPT-4o의 성능을 평가했습니다. 첫 번째 작업에서는 769명의 환자 전문 노트를 통해 Clinical Dementia Rating(CDR) 점수를 자동으로 생성했으며, 두 번째 작업에서는 860명의 Medicare 환자의 자료를 분석하여 정상 인지, 경도 인지 장애(MCI), 치매 상태를 분류했습니다. 이 과정을 위해로 Retrieval-Augmented Generation(RAG) 접근법과 명확한 주제 템플릿을 활용하여 모델의 결과를 표준화했습니다.

- **Performance Highlights**: 연구 결과는 GPT-4o가 CDR 점수 지정 작업에서 0.83의 kappa 점수를 기록하였고, MCI 및 치매 상태 구분에서 0.91의 kappa 점수를 달성했습니다. 특히, 임상 진단 과정에서 높은 신뢰도로 평가된 경우에는 0.96의 kappa 점수로, GPT-4o의 높은 성과를 입증했습니다. 이러한 결과는 GPT-4o가 대규모 차트 검토 도구로서 임상 설정에서 효과적일 수 있음을 시사합니다.



### NeuralCFD: Deep Learning on High-Fidelity Automotive Aerodynamics Simulations (https://arxiv.org/abs/2502.09692)
Comments:
          Preprint

- **What's New**: 이번 논문에서는  자동차 공기역학 분야의 혁신을 위한 Geometry-preserving Universal Physics Transformer (GP-UPT)라는 새로운 신경 연산자가 소개됩니다. GP-UPT는 기하학 인코딩(geometry encoding)과 물리적 예측(physics predictions)을 분리하여 기하학 표현(geometry representations) 및 표면 샘플링 전략(surface sampling strategies)에 대한 유연성을 보장합니다. 이를 통해 고품질의 시뮬레이션 메시(mesh)를 생성할 필요 없이, 대규모의 표면 및 부피 메시에 대해 정확한 예측을 가능하게 합니다.

- **Technical Details**: GP-UPT는 시뮬레이션의 고유 기하학 정보(raw geometry) 를 유지하며, 다양한 입력 샘플링 패턴에 대해 모델 출력을 수렴(convergence)시킵니다. 이 모델은 20 million mesh cells에서 3D 속도 필드 예측을 수행할 수 있으며, 물리적 맥락에서 요구되는 정확도를 달성합니다. 쉽게 말해, GP-UPT는 필요에 따라 인코더(encoder)와 디코더(decoder)와 같은 모델의 각각 부분을 독립적으로 확장할 수 있는 기능을 제공합니다.

- **Performance Highlights**: GP-UPT는 낮은 신뢰도의 데이터에서 높은 신뢰도의 시뮬레이션 데이터로 전이 학습(transfer learning)을 통해, 처음부터 학습한 모델의 성능을 모방하기 위해 고신뢰도 데이터의 절반만으로도 가능하다는 점에서 주목할 만합니다. 이 모델은 드라이버 성능 예측에서 거의 완벽에 가까운 정확도를 기록하였으며, 단일 GPU에서 몇 초 만에 8.8 million surface CFD mesh cells를 예측할 수 있습니다. 이와 같이 GP-UPT는 최신의 고급 신경 연산자들에 비해 유리한 성능과 확장을 보여주며 기존의 자동차 공기역학 문제를 해결하는 데 실질적인 기여를 합니다.



### Trust at Your Own Peril: A Mixed Methods Exploration of the Ability of Large Language Models to Generate Expert-Like Systems Engineering Artifacts and a Characterization of Failure Modes (https://arxiv.org/abs/2502.09690)
Comments:
          41 pages, 10 figures

- **What's New**: 이번 연구에서는 Multi-purpose Large Language Models (다목적 대형 언어 모델, LLMs)가 시스템 엔지니어링(Systems Engineering, SE) 작업에서의 효율성에 대한 의문을 제기하고 있습니다. 인간 전문가가 생성한 SE 산출물(artifacts)을 기준으로 삼고 LLMs가 생성한 산출물과 비교하여, AI가 생성한 결과물의 품질에 대해 경고하는 내용을 담고 있습니다.

- **Technical Details**: 연구 방법은 두 가지로 나뉩니다. 첫째, 여러 LLM에 다양한 프롬프트(prompt)를 통해 SE 산출물의 세그먼트를 생성하도록 하여 기초 성능을 문서화하였습니다. 둘째, 자연어 처리 알고리즘을 사용하여 정량적으로 AI 생성 산출물과 인간 전문가 벤치마크를 비교하였으며, 정성적으로 품질 차이도 분석하였습니다.

- **Performance Highlights**: AI가 생성한 산출물이 인간 전문가의 기준과 매우 유사해 보이는 반면, 몇 가지 중대한 실패 모드를 드러내었습니다. 여기에는 조기 요구 사항 정의, 근거 없는 수치 추정 및 과도한 세부정보 지정 경향이 포함되며, 이러한 문제로 인해 시스템 엔지니어링 분야에서는 AI의 피드백을 신중하게 수용해야 함을 강조하고 있습니다.



### Towards Virtual Clinical Trials of Radiology AI with Conditional Generative Modeling (https://arxiv.org/abs/2502.09688)
Comments:
          35 pages

- **What's New**: 이번 연구에서는 인공지능 (AI) 기반의 가상 임상 시험 (VCTs)을 위한 조건부 생성 모델을 처음으로 제안합니다. 이 모델은 환자의 특성에 따라 실재와 유사한 전체 신체 CT 이미지를 합성하는 능력이 있습니다. 이를 통해 AI 모델이 다양한 환자 집단에서 어떻게 성능이 저하되는지를 미리 파악하고, 편향을 분석할 수 있습니다.

- **Technical Details**: 이 생성 모델은 이미지 오토인코더와 분할 오토인코더, 그리고 잠재 확산 모델로 구성되어 있습니다. 이 구조는 전체 신체 이미지를 고해상도로 처리할 수 있게 하며, 이미지와 분할 데이터를 저차원 잠재 공간으로 압축 i고, 다시 고품질로 복원하는 기능을 제공합니다. 또한, 환자 속성에 따라 성능 변화를 식별할 수 있도록 조건부 분포를 모델링합니다.

- **Performance Highlights**: 연구 결과, 생성 모델을 통해 검증된 가상 임상 시험이 실제 데이터 없이 AI 모델의 편향과 성능 저하 영역을 정확하게 식별할 수 있음을 보여주었습니다. 특히, 전체 신체에 대한 정보와 관련된 수치(예: 체지방 및 근육량 비율)를 예측하는 다운스트림 모델에서, 우리가 예측한 성능 저하와 그 원인이 실제 임상환경에서도 관찰된 것과 일치했습니다.



### Mind What You Ask For: Emotional and Rational Faces of Persuasion by Large Language Models (https://arxiv.org/abs/2502.09687)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)이 정서적 또는 이성적 유인을 사용할 때의 언어 패턴 차이를 분석하고, LLM이 어떻게 사회적 영향 원리를 적용하는지를 살펴보았습니다. 연구 결과로 정서적 유인은 인지적 복잡성을 높이는 데 기여할 수 있음을 강조하고, LLM이 정서적 유인에 대해 어떤 반응을 보이는지를 보여주었습니다. 또한, LLM이 생성하는 응답이 사회적 영향 원리를 근거로 만들어진다는 점도 드러났습니다.

- **Technical Details**: 연구에서는 서로 다른 크기와 라이선스를 가진 12개 LLM 모델을 선택하여 이성적 유인과 감정적 유인을 비교했습니다. 사용된 LLM은 OpenAI의 GPT-3.5 Turbo, GPT-4, Mistral, Meta의 Llama 3 시리즈, Anthropic의 Claude 모델들입니다. 각 모델에 대해 표준화된 프롬프트를 통해 다양한 변수를 포함한 데이터 세트를 사용하여 일관된 결과를 도출하였습니다.

- **Performance Highlights**: 이번 연구는 LLM의 언어 패턴 차이를 도출하며, 감정적 유인 사용 시 더 다양하고 복잡한 응답을 생성한다는 것을 밝혀냈습니다. 또한, 연구는 특정 모델들이 유인 방식에 따라 메시지를 적절히 조정할 수 있는 반면, 다른 모델들은 유사한 사회적 영향 원리를 적용하는 것을 보여주었습니다. LLM이 효과적으로 대중의 신념과 태도에 영향을 미칠 수 있는 방법을 제시하는 중요한 결과를 도출했습니다.



### Object-Centric Latent Action Learning (https://arxiv.org/abs/2502.09680)
Comments:
          Preprint. In review

- **What's New**: 이번 연구에서는 Embodied AI(실체 AI)의 발전을 위해 object-centric latent action learning(객체 중심 잠재 행동 학습) 접근법을 제안합니다. 이 방법은 VideoSaur와 LAPO를 기반으로 하여 장면을 객체 표현으로 분해하고, 비디오 데이터를 proxy-action labels(프록시 행동 레이블)로 주석처리합니다. 그 결과, 원치 않는 배경 잡음에서 인과적 agent-object(행위자-객체) 상호작용을 분리하며, 분산자(distractor)로 인한 잠재 행동 학습 성능 저하를 줄일 수 있습니다.

- **Technical Details**: 제안된 방법은 spatio-temporal object slots(시공간 객체 슬롯)로 장면을 분해하여, 잡음 없이 인과적 agent-object 표현을 확보합니다. Self-supervised feature similarity losses(자기 지도 피처 유사성 손실)을 통해 객체 중심 표현을 생성하여, 정적 배경이나 우발적 움직임을 필터링합니다. 이러한 구조적 우선 요소는 Latent Action Models(잠재 행동 모델)가 비임무 관련 객체의 동학을 무시하고 중요한 객체에 집중하게 합니다.

- **Performance Highlights**: 소규모 레이블이 있는 행동을 통해 fine-tuning(미세 조정)을 수행한 결과, 잠재 행동의 질이 x2.7 배 개선되었으며, 응용 프로그램의 평균 return이 x2.6 배 증가했습니다. Distracting Control Suite(DCS)와 함께 진행한 초기 실험을 통해 제안된 방법의 유효성을 확인했습니다. 또한, 유사 연구와 비교할 때, 우리의 접근 방식은 더 복잡한 환경에서도 효율적으로 작동하는 것으로 나타났습니다.



### Multi-level Conflict-Aware Network for Multi-modal Sentiment Analysis (https://arxiv.org/abs/2502.09675)
Comments:
          5 pages, 1 figure

- **What's New**: 이번 연구에서는 다중 모달 감정 분석(Multimodal Sentiment Analysis, MSA)에 있어 새로운 다수준(多水準) 갈등 인식 네트워크(Multi-level Conflict-aware Network, MCAN)를 제안합니다. MCAN은 각 모달의 정합성과 갈등 요소를 효과적으로 분리하여 모델링하며, 기계 학습의 불안정성을 감소시키기 위한 새로운 접근 방식을 채택했습니다. 이를 통해 기존 연구들에서 다루지 않았던 비모달 조합 간의 갈등 요소도 고려하고 있습니다.

- **Technical Details**: MCAN의 구조는 메인 브랜치(Main Branch)와 갈등 모델링 브랜치(Conflict Modeling Branch)로 나뉘어 있습니다. 메인 브랜치는 Micro Multi-step Interaction Network(Micro-MSIN) 및 Macro Multi-step Intersection Network(Macro-MSIN)를 활용하여 단일 모달 및 이중 모달 간의 관계를 점진적으로 모델링합니다. 갈등 모델링 브랜치에서는 마이크로 및 매크로 갈등 인식 크로스 어텐션(Micro-CACA, Macro-CACA)을 사용하여 갈등 요소를 모델링하며, 생성된 레이블의 의존성을 피하고 있습니다.

- **Performance Highlights**: MCAN은 CMU-MOSI와 CMU-MOSEI 데이터셋에서 기존 최선 기법들에 비해 현저한 성능 향상을 보였으며, 다양한 실험을 통해 제안된 기법의 효과성이 입증되었습니다. 특히, MCAN의 핵심 구성 요소와 주요 하이퍼파라미터의 영향을 평가하여 모델의 정확도를 높이는 데 기여했습니다. 이로 인해 다중 모달 데이터의 감정 인식에서 새로운 표준을 제시할 수 있을 것으로 기대됩니다.



### The Hidden Dimensions of LLM Alignment: A Multi-Dimensional Safety Analysis (https://arxiv.org/abs/2502.09674)
Comments:
          Code and artifacts: this https URL

- **What's New**: 이 연구는 기존의 LLM(Long Language Models)의 안전 정렬 행동(safety-aligned behaviors)이 단일 방향이 아닌 다차원 방향에 의해 제어됨을 발견하였습니다. 이는 'activation space' 내에서 안전 행동을 표현하는 새로운 방법인 'Safety Residual Space'를 소개하면서 시작합니다. 또한, 여러 개의 해석 가능한 방향(interpretable directions)을 탐구하여 모델의 안전 회피(refusal behavior)에 대한 이해를 심화 시킵니다.

- **Technical Details**: 안전 분석에서 LLM의 다양한 안전 요소를 구분하기 위해 Linear Representation Hypothesis를 기반으로 하는 프레임워크를 구축하였습니다. 이 과정에서 안전 세분화 바탕으로 모델이 보이는 방향(feature direction)을 탐색하며, 각 방향이 서로 어떻게 상호작용하는지를 분석합니다. 이러한 다차원 해석을 통해 특정 트리거 토큰의 삭제가 안전 정렬을 우회하는 데 어떻게 영향을 미치는지에 대한 통찰을 제공합니다.

- **Performance Highlights**: 연구 결과, 특정한 방향들이 LLM의 거부 행동을 결정짓는 지배적인 역할을 한다는 것을 발견했습니다. 또한, 여러 비지배적인 방향이 안전 fine-tuning 동안 학습되는 다양한 능력에 대한 조절 역할을 한다는 것을 실험을 통해 확인하였습니다. 이러한 통찰은 LLM의 취약성을 이해하고 개선하는 데 중요한 기초 자료로 활용될 수 있을 것입니다.



### Are Smarter LLMs Safer? Exploring Safety-Reasoning Trade-offs in Prompting and Fine-Tuning (https://arxiv.org/abs/2502.09673)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)의 추론 능력이 향상되면서 나타나는 안전성(혹은 safety) 문제를 조명합니다. 복잡한 작업에서의 향상된 성능은 중요한 반면, 이로 인해 발생할 수 있는 새로운 취약점(vulnerability)에 대한 경각심을 일깨우고 있습니다. 연구에서는 이러한 문제와 문제 해결을 위한 새로운 접근 방식을 제시합니다.

- **Technical Details**: 연구는 LLM의 추론 능력과 안전성 간의 상호작용(interplay)을 분석하여, 추론이 발전함에 따라 발생할 수 있는 잠재적 안전 위험(latent safety risks)을 조명합니다. 또한, 안전성을 증대시키기 위해 추론을 활용할 수 있는 방법들을 탐구합니다. 이러한 접근은 기존의 접근 방식과 다게, 보다 다각적인 관점에서 문제를 접근하게 합니다.

- **Performance Highlights**: 연구 결과는 향상된 추론 능력이 모델의 복잡한 작업 수행 능력을 높이는 동시에, 새로운 안전 문제의 출현 가능성을 시사합니다. 연구는 모델의 안전성을 강화하는 데 있어 추론을 활용할 수 있는 잠재적인 완화 전략(potential mitigation strategies)을 제시하여, LLM이 실제 배치 시 더 신뢰성(trustworthy) 있는 성능을 발휘할 수 있도록 기여합니다.



### The Science of Evaluating Foundation Models (https://arxiv.org/abs/2502.09670)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)의 체계적 평가 과정을 형성화하고, 구체적인 사용 사례와 관련된 평가 방법론을 제시하는 데 초점을 맞추고 있습니다. Rigorously structured evaluation guidelines는 다양한 모델을 비교하고, 이를 토대로 사용자가 현실 애플리케이션에 LLM을 어떻게 통합할 수 있는지를 명확히 설명합니다. 또한 체크리스트와 문서 템플릿과 같은 실행 가능한 도구를 제공하여 평가 프로세스를 지원합니다.

- **Technical Details**: 논문에서 제안된 'ABCD in Evaluation' 프레임워크는 알고리즘(Algorithm), 대규모 데이터(Big Data), 계산 자원(Computation Resources), 도메인 전문성(Domain Expertise)을 포함하는 체계적인 평가 접근법을 제공합니다. 이는 LLM의 평가 과정에서 알고리즘의 역할, 데이터의 중요성, 계산 인프라의 요구 사항 및 관련 도메인 지식의 필요성을 강조합니다. 이 프레임워크는 LLM 평가의 복잡성을 깊이 이해할 수 있는 기초를 제공합니다.

- **Performance Highlights**: 제시된 평가 프레임워크는 LLM의 성능을 정확하게 평가하는 데 필요한 다양한 차원을 포괄합니다. 이는 대규모 공개 벤치마크뿐만 아니라 도메인 별 데이터셋을 활용하여 모델의 다양성과 공정성을 측정하는 데 기여합니다. 궁극적으로, 이 연구는 LLM 평가를 위한 유의미한 기준을 제공하고, 종합적인 평가 접근법을 통해 다양한 산업에 걸쳐 모델을 협의적이고 책임감 있게 배포하는 데 도움을 줄 것입니다.



### Meta-INR: Efficient Encoding of Volumetric Data via Meta-Learning Implicit Neural Representation (https://arxiv.org/abs/2502.09669)
Comments:
          Accepted by PVIS Short Paper Track

- **What's New**: 본 논문은 Meta-INR이라는 사전 훈련 전략을 제안하여, 초기 INR(Implicit Neural Representation) 파라미터를 학습하는 방법을 설명합니다. 이 방법은 대규모 시간 변동 혹은 앙상블 볼륨 데이터 세트에서 구조적 패턴이 유사한 볼륨을 독립적으로 훈련할 필요가 없도록 합니다. 본 연구는 INR을 처음부터 훈련하는 대신, 부분 관찰 데이터를 통해 초기 파라미터를 학습하여 훈련 효율을 극대화합니다.

- **Technical Details**: Meta-INR은 두 단계로 구성된 훈련 파이프라인인 메타-프리트레이닝과 볼륨 특화 파인튜닝을 사용합니다. 메타-프리트레이닝 단계에서는 원본 데이터의 1% 이하인 희소 하위 샘플링된 볼륨 데이터 세트를 활용하여 초기 INR 네트워크 파라미터를 학습합니다. 이 후 볼륨 특화 파인튜닝 단계에서 메타 모델의 초기 파라미터를 특정 볼륨에 맞게 조정하여 고신뢰도의 볼륨 재구성을 위한 적응된 INR을 생성합니다.

- **Performance Highlights**: Meta-INR을 통해 기존의 INR 훈련보다 더 빠른 수렴을 달성할 수 있으며, 단 몇 번의 파라미터 업데이트로 새로운 볼륨에 적응합니다. 이는 다양한 데이터셋 간에 이전에 보지 못한 유사한 볼륨 데이터를 인코딩하는 데 도움을 주는 고품질의 일반화 가능한 특징을 효과적으로 추출합니다. 또한, 시뮬레이션 매개변수 분석 및 대표적인 타임스텝 선택과 같은 작업에서 유용성을 강조하고 있습니다.



### DiffEx: Explaining a Classifier with Diffusion Models to Identify Microscopic Cellular Variations (https://arxiv.org/abs/2502.09663)
- **What's New**: 본 논문에서는 DiffEx라는 새로운 방법을 소개하며, 이는 깊은 학습 분류기가 내리는 결정을 설명하기 위해 시각적으로 해석 가능한 속성을 생성하는 데 중점을 둡니다. DiffEx는 미세한 세포 변화를 식별하는 데 걸림돌이 되었던 기존의 블랙박스 관점을 탈피합니다. 이를 통해 질병의 이해도를 높이고 신약 개발 시 새로운 바이오마커를 발굴하는 데 도움을 줄 수 있는 가능성이 제시됩니다.

- **Technical Details**: DiffEx는 확산 모델(diffusion models)을 활용하여 분류기가 사용하는 속성을 식별하는 방법입니다. 이 과정을 위해 먼저 분류기의 속성을 포함하는 잠재 공간(latent space)을 구축하고, 대조 학습(contrastive learning) 접근 방식을 사용하여 이 잠재 공간에서 해석 가능한 방향들을 식별합니다. 발견된 방향은 분류기의 결정을 가장 크게 변화시키는 속성을 선택하여 순위화합니다.

- **Performance Highlights**: DiffEx의 효과는 자연 이미지와 생물학적 이미지를 기반으로 훈련된 분류기에 대한 설명력에서 입증되었습니다. 생물학 데이터셋을 활용하여 서로 다른 조건 간의 미세한 세포 변화를 발견하는 데 성공했습니다. 이러한 연구는 질병 및 치료 효과에 대한 이해를 심화시키고, 고유한 표현형 차이를 식별하는 데 기여할 것으로 기대됩니다.



### Cancer Vaccine Adjuvant Name Recognition from Biomedical Literature using Large Language Models (https://arxiv.org/abs/2502.09659)
Comments:
          10 pages, 6 figures, 4 tables

- **What's New**: 이 연구는 암 백신 연구에서 adjuvant(보조제)의 이름을 자동으로 인식하는 방법을 탐구합니다. 기존의 생물의학 문헌에서 수작업으로 보조제를 분류하는 것의 어려움을 극복하기 위해, 대규모 언어 모델(LLMs)인 GPT와 Llama를 활용하여 그 가능성을 증명했습니다.

- **Technical Details**: 연구에 사용된 데이터셋은 AdjuvareDB의 97개의 임상 시험 기록과 Vaccine Adjuvant Compendium(VAC)에서 주석 처리된 290개의 초록입니다. GPT-4o와 Llama 3.2는 zero-shot과 few-shot 학습 방식을 통해 보조제 이름 인식을 수행했고, 포맷된 프롬프트에 따라 다양한 맥락 정보를 포함하여 성능을 테스트하였습니다.

- **Performance Highlights**: GPT-4o는 모든 상황에서 100%의 Precision(정확도)을 기록하며 Recall(재현율)과 F1-score(조화 평균)에서도 뛰어난 성능을 보여주었습니다. 특히 VAC 데이터셋에서 F1-score 77.32%를 기록하며, AdjuvareDB 데이터셋에서는 81.67%의 F1-score를 달성했습니다. 이로써 모델의 전반적인 보조제 식별 능력이 입증되었습니다.



### Neuro-Conceptual Artificial Intelligence: Integrating OPM with Deep Learning to Enhance Question Answering Quality (https://arxiv.org/abs/2502.09658)
Comments:
          15 pages, 3 figures,

- **What's New**: 이 논문에서는 Neural-symbolic AI 접근법의 전문화된 형태인 Neuro-Conceptual Artificial Intelligence (NCAI)를 소개합니다. NCAI는 Object-Process Methodology (OPM)를 활용하여 질의응답(QA) 품질을 향상시키고 투명한 AI 시스템을 구축하는 데 초점을 맞추고 있습니다. 이 방법은 자연어 텍스트를 OPM 모델로 변환하여 제기되는 복잡한 개념을 처리하는 데 도움을 줍니다.

- **Technical Details**: NCAI는 OPM을 사용하여 프로세스, 객체 및 상태 등 복잡한 OPM 요소를 표현합니다. OPM-QA 시스템에서의 구조화된 지식 표현은 전통적인 트리플 기반 지식 그래프가 포착하기 어려운 복잡한 프로세스를 처리할 수 있게 합니다. 연구자들은 추가적으로 OPM 기반 개념 논리에 얼마나 충실하게 예측된 추론이 일치하는지를 측정하는 투명성 평가 메트릭스를 제안하였습니다.

- **Performance Highlights**: 실험 결과, NCAI는 전통적인 방법보다 뛰어난 성능을 보였으며, 복잡한 지식 표현을 제공하여 추론의 투명성을 향상시킵니다. NCAI는 측정 가능한 투명성과 향상된 추론을 통해 신경-상징 AI(neuro-symbolic AI)의 발전 가능성을 강조하고 있습니다.



### Bidirectional Diffusion Bridge Models (https://arxiv.org/abs/2502.09655)
Comments:
          Source code: this https URL

- **What's New**: 이번 연구에서는 Bidirectional Diffusion Bridge Model (BDBM)을 도입하여 파트너 두 분포 간의 양방향 변환을 단일 네트워크를 통해 가능하게 합니다. 기존의 방법들은 일방향성이어서 각 방향에 대해 별도의 모델을 필요로 하여 계산 비용이 두 배로 증가하는 문제가 있었습니다. 따라서 BDBM은 이러한 한계를 극복함으로써 양방향 생성 모델을 활용할 수 있도록 돕습니다.

- **Technical Details**: BDBM은 Chapman-Kolmogorov Equation (CKE)을 활용하여 두 분포 간의 상태 전이를 모델링합니다. 이를 통해 시간 단계 간의 데이터 분포 변화를 효과적으로 처리하며, 특히 최종 점이 Gaussian 분포일 경우, 양방향 전이 커널이 해석 가능한 형태를 가져 학습 효율성을 높입니다. 이 또한 기존의 다리 방법과의 연결성을 보여주고, 그 이점을 강조합니다.

- **Performance Highlights**: 고해상도 I2I 변환 작업에 대한 광범위한 실험을 통해, BDBM은 기존의 양방향 모델에 비해 시각적 품질(FID) 및 인지 유사성(LPIPS) 측면에서 뛰어난 성능을 발휘했습니다. 이 방법은 추가적인 비용 없이도 양방향 변환을 가능하게 하며, 더 나아가 훈련 반복 횟수조차 비슷하거나 더 적게 요구하여 성능 향상에 기여합니다.



### Principled Data Selection for Alignment: The Hidden Risks of Difficult Examples (https://arxiv.org/abs/2502.09650)
- **What's New**: 본 논문에서는 대형 언어 모델(LLM) 훈련에 있어 데이터 선택과 모델 용량 간의 관계를 새롭게 조명합니다. 저자들은 지나치게 어려운 예제가 정렬(alignment)을 저해할 수 있다는 원칙을 제안하며, 이를 통해 모델의 성능을 향상시킬 수 있음을 실험을 통해 입증했습니다. 구체적으로, Selective DPO라는 새로운 방법론을 통해 이러한 어려운 예제를 걸러내는 접근법을 소개하고 있습니다.

- **Technical Details**: 선택적 DPO는 지나치게 어려운 예제를 필터링함으로써 정렬 성능을 향상시키는 방법입니다. 연구진은 3가지 주요 주장으로 이러한 원칙을 뒷받침합니다: (1) 선호 데이터는 난이도에 따라 분류될 수 있다, (2) 지나치게 어려운 예제는 정렬 성능을 해칠 수 있다, (3) 난이도는 모델의 용량에 비례한다. 이러한 실험을 통해 LLM의 성능 향상에 기여할 수 있는 데이터 선택 원칙을 한걸음 나아가 정립했습니다.

- **Performance Highlights**: Selective DPO는 AlpacaEval 2 벤치마크에서 기존 DPO에 비해 9-16%의 승률 향상을 가져왔습니다. 또한, 이 방법은 SimPO 및 R-DPO와 같은 최신 기법들보다 우수한 성능을 보이며, 더 나은 perplexity와 암묵적 보상 마진을 유지하는 것으로 나타났습니다. 이러한 결과는 데이터 난이도와 모델 용량의 적절한 조화를 이룰 필요성을 강조합니다.



### UKTA: Unified Korean Text Analyzer (https://arxiv.org/abs/2502.09648)
Comments:
          Accepted by SAC 2025

- **What's New**: 이번 논문은 한국어 글쓰기 평가의 자동화 도구인 UKTA(Unified Korean Text Analyzer)를 소개합니다. UKTA는 한국어 텍스트를 분석하고 평가하는 포괄적인 시스템으로, 정확한 저수준(morpheme level) 형태소 분석과 중간 수준에서의 주요 Lexical feature를 제공하며, 명확한 고수준(rubric-based) 작문 점수를 제공합니다. 기존 한국어 텍스트 분석 도구의 한계를 극복하며, 다면적(multi-view) 접근 방식을 통해 글쓰기 평가의 정확성을 향상시킵니다.

- **Technical Details**: UKTA는 형태소 분석(morpheme analysis), 어휘 다양성(lexical diversity), 의미 응집성(semantic cohesion) 등의 특성을 고려하는 다면적 분석을 지원합니다. 보편적인 한국어 문법 특성에 적합한 오류 전파(error propagation) 방지 메커니즘을 통해, 초기 단계에서 발생한 오류가 최종 작문 평가에 미치는 영향을 최소화합니다. 또한, 고수준 평가 결과는 인간이 이해할 수 있게 해설하는 기능을 제공함으로써 평가의 신뢰성을 보장합니다.

- **Performance Highlights**: UKTA는 제안된 모든 특성을 사용함으로써 기존의 기준선 모형에 비해 정확성과 Quadratic Weighted Kappa 점수를 향상시켰습니다. 실험 결과, 각각의 feature가 포함된 경우 글쓰기 평가의 정확도가 현저히 개선됨을 보여주었으며, 이는 글쓰기 평가에 대한 보다 신뢰할 수 있는 접근 방식을 제공합니다. UKTA는 한국어 텍스트 분석 및 글쓰기 평가 도구로서 주목받을만한 잠재력을 지니고 있습니다.



### From No to Know: Taxonomy, Challenges, and Opportunities for Negation Understanding in Multimodal Foundation Models (https://arxiv.org/abs/2502.09645)
- **What's New**: 이 논문은 다국어 다중모달 모델들이 부정(Negation) 처리에서 겪는 문제는 물론, 이를 개선하기 위한 포괄적인 분류법과 제안된 벤치마크를 제시합니다. 부정은 단순한 부정 표현을 넘어 다양한 구조적, 의미적, 문화적 요소에 의해 영향을 받음을 강조하며, 본 연구는 부정 처리를 정확하게 하기 위한 기초적인 가이드라인을 제시합니다.

- **Technical Details**: 부정 처리의 복잡성을 해결하기 위한 전략으로, 언어 특정 토크나이제이션(language-specific tokenization), 정교한 주의 메커니즘(fine-grained attention mechanisms), 그리고 고급 다중모달 아키텍처(advanced multimodal architectures)의 필요성을 제안합니다. 이러한 접근 방식은 부정 이해를 더욱 정교하게 하고, 다국어 환경 속에서 복잡한 상황을 처리할 수 있는 모델을 개발하도록 돕습니다.

- **Performance Highlights**: 부정 표현을 잘 처리하지 못하면, 챗봇 응답, 의료 이미지 분석, 법률 문서 작성에서 중요한 오류를 발생시킬 수 있습니다. 이 연구는 부정 이해를 강화함으로써 모델의 구성 및 논리적 사고(compositional and logical reasoning)를 개선하고, 단일 언어 및 다국어 환경에서의 성능 차이를 알아보는 벤치마크의 필요성을 강조합니다.



### From Argumentation to Deliberation: Perspectivized Stance Vectors for Fine-grained (Dis)agreement Analysis (https://arxiv.org/abs/2502.09644)
Comments:
          Accepted at NAACL Findings 2025

- **What's New**: 이 연구는 Perspectivized Stance Vectors(PSVs)를 개발하여 논쟁에서 표현된 다양한 입장을 분석하는 새로운 프레임워크를 제시합니다. 이는 각 논쟁자의 입장이 추구하는 가치를 이해하고, 상반된 시각에서의 동의를 구별하는 데 도움을 줄 수 있습니다. 이 과정을 통해 상반된 의견 속에서도 문제 해결을 위한 실행 가능한 포인트를 찾는 데 중점을 두고 있습니다.

- **Technical Details**: PSVs는 이슈 특정 개념에 대한 논쟁자의 입장을 벡터 형태로 표현합니다. 이를 위해 각 논쟁에 대한 입장 및 이슈에 특정한 개념들을 정리하고, 해당 개념에 대한 논쟁자의 입장을 예측합니다. 이러한 분석은 표면적인 동의와 불일치를 넘어서, 더욱 세분화된 통찰들을 제공합니다.

- **Performance Highlights**: 연구팀은 pakt라는 구조화된 논쟁 코퍼스에서 실험을 수행하여 PSVs의 성능을 평가했습니다. 이 과정에서 각 모듈의 성능을 수작업으로 주석을 단 평가 세트와 비교하여, PSVs를 통해 긍정적인 결과를 얻었다고 합니다. 그 결과, PSVs는 입장 간의 동의와 불일치뿐만 아니라 수평적(orthogonal) 관계를 감지하는 데 매우 유용한 도구로 나타났습니다.



### Krutrim LLM: Multilingual Foundational Model for over a Billion Peop (https://arxiv.org/abs/2502.09642)
- **What's New**: 이번 연구에서는 인도의 언어적 다양성을 고려한 최초의 대규모 다국어 모델인 Krutrim LLM을 소개합니다. 기존 AI 모델들이 영어에 초점을 두고 훈련되어 인도 내 다양한 언어가 효과적으로 반영되지 못하는 문제를 해결하기 위해, Krutrim LLM은 2조 개의 토큰을 사용하여 훈련되었습니다. 이 모델은 인도 사회의 언어적 풍경을 반영하며, 다양한 방언에서의 균형 잡힌 성능을 보장합니다.

- **Technical Details**: Krutrim LLM은 고급 Attention 메커니즘, 즉 Grouped Query Attention (GQA)와 AliBi를 통합하여 긴 컨텍스트를 처리하고 빠른 응답을 제공합니다. 모델은 인도에 특화된 파인튜닝 과정을 거쳐 지역의 특정한 요구와 뉘앙스를 효과적으로 반영할 수 있도록 발전했습니다. 또한, 전용 Indic tokenizer를 개발하여 인도 언어의 복잡한 형태와 구문 처리를 최적화하였습니다.

- **Performance Highlights**: Krutrim LLM은 16개의 과제 중 10개에서 LLAMA-2와 같은 기존 모델과 동등하거나 더 나은 성능을 보였습니다. 훈련비용은 훨씬 적지만, 평균 성적은 0.57로 LLAMA-2의 0.55를 초과합니다. 이는 다양한 언어적 맥락에서 유연한 다국어 구사 능력을 보여줍니다.



### Online Social Support Detection in Spanish Social Media Texts (https://arxiv.org/abs/2502.09640)
- **What's New**: 이번 연구는 스페인어 소셜 미디어 텍스트에서 온라인 사회적 지원을 탐지하는 혁신적인 접근 방식을 제안합니다. 연구는 지원적(comment) 또는 비지원적(non-supportive)으로 분류된 3,189개의 YouTube 댓글로 구성된 첫 번째 주석 데이터셋을 소개합니다. 또한 데이터 불균형(data imbalance)을 해결하기 위해 GPT-4o를 사용하여 문장의 다양한 표현(paraphrases)을 생성했습니다.

- **Technical Details**: 기존의 머신러닝 모델(traditional machine learning models), 딥러닝 아키텍처(deep learning architectures), 그리고 transformer 기반 모델을 포함한 여러 모델을 평가하여 사회적 지원 분류(social support classification)을 수행했습니다. 특히 GPT-4o는 불균형된 데이터셋에서만 최상의 성능을 보였으며, 불균형 및 균형된 데이터셋을 비교하기 위해 transformer 모델을 사용했습니다. 이 과정에서 각 작업(Task) 별로 성능을 분석했습니다.

- **Performance Highlights**: 균형 데이터셋은 Task 2(개인 및 그룹)와 Task 3(국가, 기타, LGBTQ, 흑인 커뮤니티, 여성, 종교)에서 개선된 결과를 보여주었습니다. 반면, GPT-4o는 Task 1(사회적 지원 및 비지원)에 최적의 성능을 기록했습니다. 이 연구는 지지적 온라인 환경의 중요성을 강조하고, 자동화된 사회적 지원 탐지를 위한 향후 연구의 기초를 마련합니다.



### Jailbreaking to Jailbreak (https://arxiv.org/abs/2502.09638)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 안전성을 높이는 방법으로 '자기 조인트 리벨리온(jailbreak)' 개념을 도입합니다. 이를 통해 인간이 LLM을 스스로 해제하도록 유도함으로써, 다른 모델에 대한 평가를 할 수 있는 J2 공격자를 생성합니다. 특히, LLM은 이전 실패로부터 학습하여 성능을 개선할 수 있습니다. 이를 통해 대형 모델이 자율적으로 공격할 수 있는 새로운 경향을 보여줍니다.

- **Technical Details**: 이 연구는 LLM 안전성을 높이기 위한 다양한 기법과 자동화된 공격이 조합된 'J2 공격자' 모델을 제안합니다. J2는 다수의 레드 팀 전략을 실험하며, 이전의 실패를 통해 공격 성능을 개선하는 인컨텍스트 러닝(in-context learning) 기법을 사용합니다. 실험 결과, Sonnet 3.5와 Gemini 1.5 Pro가 각각 93%와 91%의 공격 성공률을 달성하며, 이는 기존의 LLM보다 우수한 성능을 입증합니다.

- **Performance Highlights**: Sonnet 3.5와 Gemini 1.5 Pro는 각각 93.0%와 91.0%의 공격 성공률을 기록하며, GPT-4o와 같은 경쟁 모델에 대해 우수한 성능을 보여줍니다. 기존의 자동화된 공격 방법보다 효율적인 결과를 도출하며, J2 공격자가 다양한 목표 모델에 대해 J2로 변환할 수 있는 가능성을 열어줍니다. 이 연구는 LLM 안전성 연구에 있어 중요한 기여를 하며, LLM이 직면할 새로운 실패 모드를 강조합니다.



### Meta-Cultural Competence: Climbing the Right Hill of Cultural Awareness (https://arxiv.org/abs/2502.09637)
- **What's New**: 최근 연구들은 대형 언어 모델(LLMs)이 서구 중심적 세계관에 편향되어 있음을 보여주었고, 이는 비서구 문화 환경에서의 유용성을 감소시킵니다. 이 논문에서는 LLM이 '문화적 인식(cultural awareness)'을 소유한다는 것이 무엇을 의미하는지를 조사하고, 문화적 인식이 아닌 메타문화적 능력(meta-cultural competence)이 필요하다는 주장을 합니다. 이는 다양한 문화에서 유용하게 활용될 수 있는 LLM과 AI 시스템의 필수 조건입니다.

- **Technical Details**: 논문에서는 Octopus test를 활용한 사고 실험을 통해, 서로 다른 문화적 배경을 가진 대화 참여자들 간의 통신 패턴 학습이 얼마나 복잡한지를 설명합니다. 새로운 Multi-Pair Octopus Test를 도입하여 해양 아래의 통신 케이블을 통해 서로 다른 방식을 사용하는 두 그룹의 대화에서 LLM이 어떻게 반응할 수 있는지를 살펴봅니다. 이는 AI 시스템이 내부 및 외부 문화 간 의사소통을 효과적으로 처리하기 위해 이해해야 할 복잡한 맥락을 강조합니다.

- **Performance Highlights**: 문화는 복잡하고 다면적인 개념으로서 각종 특정한 특성을 지니고 있습니다. 특히, 문화는 동적이며 시간이 지남에 따라 변화합니다. 따라서 AI 시스템이 문화를 적절히 표현하고 처리하기 위해서는 이러한 요인들을 모두 반영할 수 있어야 합니다. 또한, 언어는 문화의 필수적인 요소로서 다양한 혼합 양태를 갖고 있으며, AI 시스템은 이러한 문화적 차이를 유연하게 다룰 수 있는 전략이 필요합니다.



### Reading between the Lines: Can LLMs Identify Cross-Cultural Communication Gaps? (https://arxiv.org/abs/2502.09636)
- **What's New**: 이 논문은 다양한 문화적 배경을 가진 사람들에 의해 작성된 책 리뷰의 이해도 격차를 연구합니다. 57개의 Goodreads 리뷰를 분석한 결과, 83%의 리뷰에서 문화적으로 특정한 이해하기 어려운 요소가 발견되었습니다. 또한, 문화 배경에 따라 GPT-4o가 이러한 요소를 식별할 수 있는 능력을 평가하였으나 mixed results (혼합된 결과)가 나타났습니다.

- **Technical Details**: 문화는 사람들의 삶의 방식과 세계관을 형성하는 복잡한 개념으로 정의됩니다. 이 논문에서는 AI와 NLP 기술의 발전으로 Large Language Models (LLMs)가 온라인의 문화 간 커뮤니케이션 장벽을 극복할 수 있는 가능성에 대해 탐구합니다. LLM이 문화 중재자로서 작용하여 문화적으로 특정한 항목(Culture-Specific Items, CSIs)을 식별하고 설명하는 방법을 다룹니다.

- **Performance Highlights**: 사용자 연구를 통해 미국, 멕시코, 인도 참가자들이 Goodreads의 영어 리뷰를 얼마나 이해하기 힘들어하는지 측정했습니다. 이 연구에서는 GPT-4o와 같은 LLM이 문화적 독서 보조 도구로서 CSIs를 식별하고 설명하는 데 있어 공정하게 작동할 수 있는지를 평가했습니다. 연구 결과로 제공된 데이터셋은 문화 간 커뮤니케이션 개선을 위한 AI 기반 도구의 필요성을 강조합니다.



### CORRECT: Context- and Reference-Augmented Reasoning and Prompting for Fact-Checking (https://arxiv.org/abs/2502.09635)
Comments:
          Accepted to NAACL-25

- **What's New**: 이 논문에서는 증거 문장의 진실성을 확인하기 위해 추가적인 맥락(context)과 참조(reference)를 통합한 새로운 접근법인 Context- and Reference-augmented Reasoning and Prompting (CORRECT)를 제안합니다. 기존의 사실 검사(fact-checking) 모델은 일반적으로 증거 문장 내에서의 추론에 주로 집중하였으나, 이 연구는 보조적인 맥락과 참조를 고려하여 보다 정확한 판단을 가능하게 합니다. 이러한 접근법은 기존의 모델들이 간과했던 여러 증거 및 외부 문서들을 통합하여 일관된 증거 임베딩을 생성합니다.

- **Technical Details**: CORRECT 모델은 세 가지 계층으로 구성된 증거 그래프(evidence graph)를 활용합니다: 증거(evidence), 맥락(context), 참조(reference). 내부 및 외부 계층(reasoning) 간의 논리를 통해 세 가지 그래프 계층을 통합하여 통합된 증거 임베딩을 생성합니다. 또한, 증거-conditioned prompt encoder를 설계하여 각 주장(claim)에 대한 고유한 프롬프트 임베딩(prompt embedding)을 생성하고, 이를 통해 사실 검사에 필요한 데이터를 통합합니다.

- **Performance Highlights**: 실험 결과, CORRECT 모델은 기존 사실 검사 모델들보다 더 나은 성능을 보이며, 다양한 유형의 증거 문장을 처리하는 데 효과적입니다. 모델은 특히 맥락 의존적 및 참조 의존적인 증거를 잘 통합할 수 있어, 복잡한 주장을 보다 정확히 검증할 수 있습니다. 이를 통해 정보의 신뢰성 문제를 해결하는 데 기여할 수 있을 것으로 기대됩니다.



### Score-of-Mixture Training: Training One-Step Generative Models Made Simple via Score Estimation of Mixture Distributions (https://arxiv.org/abs/2502.09609)
Comments:
          27 pages, 9 figures. Title updated to match the title of the manuscript, otherwise identical to v1

- **What's New**: 이번 논문에서는 Score-of-Mixture Training (SMT)이라는 새로운 프레임워크를 제안합니다. 이는 $eta$-skew Jensen-Shannon divergence라는 새로운 종류의 다이버전스를 최소화하는 방법으로, 실제 샘플과 가짜 샘플 간의 혼합 분포의 점수를 추정합니다. SMT는 초기부터 훈련이 가능할 뿐만 아니라, 사전 훈련된 확산 모델을 활용한 양자화 방법인 Score-of-Mixture Distillation (SMD) 또한 지원합니다.

- **Technical Details**: SMT는 다양한 노이즈 수준에서의 실제 및 가짜 샘플 간 혼합 분포의 점수를 예측하여 훈련을 진행합니다. 사용자가 필요로 하는 하이퍼파라미터 튜닝이 최소화되며, 훈련의 안정성을 보장합니다. 이러한 프로세스는 기존의 일관성 모델(consistency models)과 유사한 접근 방식입니다.

- **Performance Highlights**: CIFAR-10 및 ImageNet 64x64 데이터셋에서의 실험 결과, SMT/SMD 방법이 기존의 여러 방법들과 경쟁력을 가지며 심지어 더 뛰어난 성능을 보여주었습니다. 이는 새로운 다이버전스 접근 방식을 통해 성과를 달성한 것으로 평가받습니다.




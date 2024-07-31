### Aligning LLM Agents by Learning Latent Preference from User Edits (https://arxiv.org/abs/2404.15269)
- **What's New**: 이 논문에서는 사용자가 에이전트의 출력을 편집하는 것을 기반으로 언어 에이전트의 상호작용 학습을 연구합니다.  작성 보조기구와 같이 일반적인 설정에서 사용자는 언어 에이전트와 상호 작용하여 주어진 맥락에서 응답을 생성하고, 그 응답을 개인의 숨겨진 선호에 따라 맞추기 위해 선택적으로 편집할 수 있습니다. 이러한 편집 피드백은 자연스럽게 생성되기 때문에 사용자의 선호와 대응(Alignment)을 개선하고 시간이 지남에 따라 사용자의 편집 비용을 줄이는 데 적합한 후보가 됩니다. 저희는 사용자의 숨겨진 선호에 대한 기술을 유추하고 이를 이용하여 향후 반응 생성을 유도하는 프롬프트 정책을 정의하는 학습 프레임워크 PRELUDE를 제안합니다.

- **Technical Details**: PRELUDE 프레임워크는 에이전트를 세밀하게 조정하는 것을 피하여, 많은 사용자와 스케일링 할 때의 도전과 성능 저하 문제를 해결합니다. 사용자 선호의 해석 가능성을 높이고, 사용자가 학습된 선호를 보고 수정할 수 있도록 합니다. CIPHER 알고리즘은 대규모 언어 모델(LLM)을 활용하여 주어진 맥락에서 사용자의 편집을 기반으로 사용자 선호를 추론합니다. 또한, CIPHER는 역사적 맥락에서 k-가장 가까운 맥락을 검색하여 반응 생성을 위한 집단 선호를 형성합니다.

- **Performance Highlights**: CIPHER는 직접적인 사용자 편집을 검색하거나 맥락에 무관한 선호를 학습하는 알고리즘에 비해 두 평가 환경(요약 및 이메일 작성)에서 가장 낮은 편집 거리 비용을 달성했으며, 학습된 선호가 기준 선호와 상당한 유사성을 보였습니다.



### XFT: Unlocking the Power of Code Instruction Tuning by Simply Merging  Upcycled Mixture-of-Experts (https://arxiv.org/abs/2404.15247)
- **What's New**: XFT는 기존의 Mixture-of-Experts (MoE) 모델을 활용하여 Large Language Models (LLMs)의 명령어 기반 튜닝 성능을 향상시키는 새로운 훈련 방식을 제시합니다. 이는 모델 업사이클링과 합병 기술을 사용하여 더 높은 성능을 달성하는 기존 기술과는 차별화된 접근 방식으로, 특히 코드 생성 Large Language Models에 초점을 맞춘 연구입니다.

- **Technical Details**: XFT는 ‘sparse upcycling’ 기술을 'shared expert mechanism'과 'novel routing weight normalization strategy'로 개선하여 효과적으로 명령어 튜닝을 증진합니다. 또한, 학습 가능한 모델 합병 메커니즘을 도입하여 업사이클된 MoE 모델을 조밀한(dense) 모델로 재구성함으로써, 덴스 모델만의 연산으로 MoE 수준의 성능을 이끌어냅니다. 이러한 기법인 XFT는 기존의 Evol-Instruct, OSS-Instruct 같은 기술과 완전히 독립적이며, 코드 명령어 튜닝을 개선하는 새로운 방향을 제시합니다.

- **Performance Highlights**: XFT는 1.3B 모델을 사용하여 새로운 최고 수준의 'tiny code LLM (<3B)'을 개발했으며, HumanEval 및 HumanEval+에서 각각 67.1 및 64.6의 pass@1 점수를 기록했습니다. 이는 동일한 데이터와 모델 아키텍처를 사용하는 supervised fine-tuning (SFT) 방식보다 HumanEval+에 대해 13% 향상된 결과입니다. 뿐만 아니라 MBPP+, MultiPL-E, DS-1000에서도 2%에서 13% 사이의 일관된 성능 향상을 보여, 그 범용성을 입증하였습니다. 코드는 이 https URL에서 제공됩니다.



### CultureBank: An Online Community-Driven Knowledge Base Towards  Culturally Aware Language Technologies (https://arxiv.org/abs/2404.15238)
Comments: 32 pages, 7 figures, preprint

- **What's New**: 이 연구에서는 언어 모델의 문화적 인식을 향상시키기 위해, 다양한 온라인 커뮤니티에서 대규모로 문화 지식 베이스를 구축하기 위한 일반화 가능한 파이프라인을 설계했습니다. 그 파이프라인을 통해, TikTok과 Reddit에서 각각 12,000개와 11,000개의 문화 서술자를 포함하는 'CultureBank'라는 지식 베이스를 구축했습니다.

- **Technical Details**: CultureBank는 기존의 문화 지식 자원과 달리, 문화 서술자에 대한 다양한 시각을 포함함으로써 문화 지식의 유연한 해석을 가능하게 하며, 문맥화된 문화 시나리오들을 제공하여 실제 평가를 돕습니다. 이 연구는 다양한 대규모 언어 모델(LLM, Large Language Models)의 문화적 인식을 평가하고 개선할 수 있는 영역을 식별했습니다.

- **Performance Highlights**: CultureBank를 활용하여 fine-tune(미세 조정)된 언어 모델은 zero-shot 설정에서 두 가지 하위 문화적 작업에서 더 나은 성능을 달성했습니다. 이 결과는 문화적 인식을 갖춘 미래의 언어 기술에 대한 추천 사항을 제공하는 데 사용됩니다.



### The Power of the Noisy Channel: Unsupervised End-to-End Task-Oriented  Dialogue with LLMs (https://arxiv.org/abs/2404.15219)
Comments: 16 Pages, 7 Figures

- **What's New**: 이 논문은 대화 상태와 시스템 행동의 턴 단위 (turn-level) 주석 (annotations) 없이도 작업 지향 대화 시스템 (task-oriented dialogue systems)을 효과적으로 훈련할 수 있는 새로운 접근 방식을 제시합니다. 연구진은 라벨이 없는 데이터 (unlabelled data)와 API 스키마 (schema)만을 사용하여 대화 에이전트를 완전 비지도 (unsupervised) 방식으로 구현하였습니다.

- **Technical Details**: 연구팀은 잘 정의된 API 스키마와 라벨이 없는 사용자와 에이전트 간의 대화 데이터셋을 활용하여, 노이즈 채널 모델 (noisy channel model)을 이용해 턴 단위 주석을 잠재 변수 (latent variables)로 추론하는 새로운 방법을 개발했습니다. 기대값 최대화 (Expectation-Maximization, EM)를 이용해 이러한 의사 라벨 (pseudo-labels)을 반복적으로 개선하고 이를 통해 대화 에이전트를 종단간 (end-to-end)으로 훈련합니다.

- **Performance Highlights**: 이 방법을 MultiWOZ 벤치마크에 적용해 평가한 결과, GPT-3.5 베이스라인 대비 대화 성공률 (dialogue success rate)을 두 배 이상 향상시켰습니다.



### Does Instruction Tuning Make LLMs More Consistent? (https://arxiv.org/abs/2404.15206)
- **What's New**: 이 연구는 지시 튜닝(instruction tuning)이 언어 모델의 일관성(consistency)에 미치는 영향을 조사합니다. 특히, 입력에 대한 작은 변화에 대한 언어 모델의 민감도를 감소시키는 것으로 나타났습니다. LLaMA 모델들과 지시 튜닝된 모델들을 비교 분석하여, 지시 튜닝이 벡터 공간(vector space)과 예측 일관성에서 모델의 성능을 향상시킨다는 것을 발견했습니다.

- **Technical Details**: 연구는 10개의 지시 튜닝된 LLaMA 모델들과 원본 LLaMA-7b 모델을 비교했습니다. 주요 평가 방법으로는 벡터 공간의 코사인 거리(cosine distances) 측정, MRPC, TaPaCo, ParaRel 데이터셋을 사용한 일관성 평가가 포함됩니다. 이 모델들은 지식의 사실(factual knowledge) 회수에서의 메커니즘 분석을 통해 개선된것을 확인했습니다.

- **Performance Highlights**: 지시 튜닝된 모델들은 벡터 공간에서 의미론적으로 유사한 텍스트들을 더 가깝게 인코딩하고, 의미론적으로 다른 텍스트들을 더 멀리 인코딩하여 일관성을 향상시켰습니다. ParaRel 데이터셋에서 상위 1위 예측의 일치성도 향상되었습니다. 원본 LLaMA 모델 대비 T5-XL과 Falcon-Instruct 모델들에서도 유사한 일관성 향상 결과를 보였습니다.



### Setting up the Data Printer with Improved English to Ukrainian Machine  Translation (https://arxiv.org/abs/2404.15196)
- **What's New**: 이 연구에서는 큰 우크라이나어 언어 모델을 구축하기 위해 자연어로 표현된 새로운 알고리즘 과제를 대량으로 확장하는 방법을 제시합니다. 높은 품질의 번역 시스템을 통해 커뮤니티가 데이터셋을 더 빠르게 큐레이션할 수 있도록 하기 위함입니다. 이 목표를 지원하기 위해, 저희는 노이즈가 있는 병렬 데이터셋(3백만 쌍의 우크라이나어와 영어 문장)을 이용하여 대규모 사전학습(pretrained) 언어 모델을 지도 학습(supervised finetuning) 방법으로 번역 시스템을 구축하는 방법을 소개합니다.

- **Technical Details**: 먼저, 큰 사전학습된 언어 모델을 노이즈가 있는 병렬 데이터셋을 사용하여 세밀하게 조정합니다. 그 후 17K의 예제를 선택하여 k-fold perplexity filtering을 사용하여 더 높은 품질의 다른 데이터셋에서 두 번째 단계의 훈련을 진행합니다. 이번 연구에서 사용된 디코더(decoder)만 있는 모델은 'Dragoman'이라고 명명되었으며, 이전의 최고 성능을 자랑하던 인코더(encoder)-디코더(decoder) 모델들을 FLORES devtest 세트에서 능가하는 성능을 보여주었습니다.

- **Performance Highlights**: Dragoman 모델은 FLORES devtest 세트에서 이전 인코더-디코더 모델들에 비해 우수한 성능을 나타낸 것으로 확인되었습니다. 이는 k-fold perplexity 필터링을 통해 더 적합한 데이터의 선별과 효율적인 훈련 과정 덕분입니다.



### Pixels and Predictions: Potential of GPT-4V in Meteorological Imagery  Analysis and Forecast Communication (https://arxiv.org/abs/2404.15166)
Comments: Supplementary material PDF attached. Submitted to Artificial Intelligence for the Earth Systems (American Meteorological Society) on 18 April 2024

- **What's New**: 이 연구는 OpenAI의 GPT-4V (large-language model)가 기상 차트를 해석하고, 다양한 언어와 커뮤니티 스타일에 맞춰 기상 위험을 전달할 수 있는 능력을 평가합니다. 특히, 이 모델이 심각한 기상 조건에 대한 전망을 생성하고, 스페인어와 영어로 기상 위험 요약을 제작하는 두 가지 작업을 수행하였습니다.

- **Technical Details**: GPT-4V는 기상 차트 분석에서 심각한 날씨 전망을 생성하고, 자체 평가를 통해 그 결과를 Storm Prediction Center의 인간 발행 예보와 잘 일치하는지 평가합니다. 또한, 기상 차트로부터 스페인어와 영어로 위험 요약을 생성하는 작업을 수행하지만, 스페인어 응답은 영어에서 스페인어로의 직역(直譯, direct translation)에 가깝습니다.

- **Performance Highlights**: GPT-4V는 기상 위험을 효과적으로 전달할 수 있는 잠재력을 보여줬지만, 스페인어 응답은 비유어적(Idiomatic) 정밀성을 잃고, 통역(translation)이 부적절한 문제를 드러냈습니다. 이러한 결과는 GPT-4V와 같은 도구의 기상학 분야에서의 조심스러운 통합과 더불어 신뢰할 수 있는 설명 가능한 AI(Explainable AI) 개발의 필요성을 강조합니다.



### MixLoRA: Enhancing Large Language Models Fine-Tuning with LoRA based  Mixture of Experts (https://arxiv.org/abs/2404.15159)
Comments: 11 pages, 4 figures

- **What's New**: 새로운 MixLoRA 모델은 기존의 LoRA 기술을 활용하여, MoE (Mix-of-Expert) 구조와 통합함으로써, 자원 효율적인 sparse MoE 모델을 구축합니다. 이 접근법은 단일 GPU에서 여러 MoE 모델을 병렬로 튜닝할 수 있게 하며 GPU 메모리 사용량과 훈련 중 지연 시간을 감소시킵니다.

- **Technical Details**: MixLoRA는 동결된 사전 훈련된 밀집 모델(dense model)의 Feed-Forward 네트워크 블록 내에 여러 개의 LoRA 기반 expert를 삽입합니다. 이는 top-k 라우터(router)를 사용하여 fine-tuning을 수행합니다. MixLoRA는 독립적으로 구성 가능한 attention-layer LoRA 어댑터(adapters)를 사용하여 모델 성능을 향상시키고, expert 구축을 위해 LoRA와 그 변형을 사용할 수 있도록 지원합니다.

- **Performance Highlights**: MixLoRA는 싱글 태스크(single-task) 및 멀티 태스크(multi-task) 학습 시나리오에서 모든 평가 지표에서 뛰어난 성능을 달성했습니다. 특히, 24GB 소비자 등급 GPU에서 여러 MixLoE 모델을 병렬로 튜닝할 수 있도록 하여, GPU 메모리 소모를 41%, 훈련 중 지연 시간을 17% 감소시켰습니다.



### FASTTRACK: Fast and Accurate Fact Tracing for LLMs (https://arxiv.org/abs/2404.15157)
- **What's New**: 새롭게 제안된 FASTTRACK은 사실 추적(fact tracing)에 있어, 학습 예제들 중에서 어느 것이 특정 쿼리의 지식 출처로서 기능하는지를 식별하는 최신 방법론입니다. 이 연구는 Large Language Models(LLMs)의 능력을 활용하여 쿼리를 지원하는 증거를 확인하고, 동시에 LLMs가 사실을 추적할 수 있도록 훈련 데이터베이스를 효율적으로 클러스터링합니다.

- **Technical Details**: FASTTRACK은 기존의 사실 추적 방법들이 직면한 주요 한계점을 극복합니다. 기존 방법들은 훈련 샘플과 쿼리 간의 유사성(similarity)을 단순히 평가하는데 그쳤으며, 이는 실행 가능한 증거와 단순히 관련 있는 샘플을 효과적으로 구분하지 못했습니다. FASTTRACK은 이러한 한계를 넘어 Large Language Models를 사용하여 쿼리에 대한 지원 증거를 검증하면서, 동시에 데이터베이스를 효율적으로 클러스터링하여 계산 요구 사항을 크게 줄입니다.

- **Performance Highlights**: FASTTRACK은 기존 방법들과 비교해서 현저한 성능 개선을 보였습니다. F1 점수에서 100% 이상의 개선을 달성하였으며, TracIn보다 33배 빠른 처리 속도를 보여줍니다. 이러한 결과는 FASTTRACK이 기존의 사실 추적 기술들을 크게 뛰어넘는 효율성과 정확성을 갖추었음을 입증합니다.



### Regressive Side Effects of Training Language Models to Mimic Student  Misconceptions (https://arxiv.org/abs/2404.15156)
- **What's New**: 이 연구에서는 Large Language Models(LLM)을 개인 맞춤형 교육을 위해 학생들의 오개념을 모방하도록 훈련하는 것의 부작용을 새롭게 탐구하였습니다. 학생-교사 대화 데이터셋을 활용하여 LLM을 훈련시켜 학생의 반응을 예측하도록 하였습니다. 그 결과, 모델이 학생의 오개념을 보다 정확하게 모방할수록 팩트의 정확성과 추론 능력에 타협이 발생함을 발견하였습니다.

- **Technical Details**: 우리는 'hallucination token'이라는 기법을 도입하여 이 부작용을 해결하려 시도했습니다. 이 토큰은 학습 동안 각 학생의 응답 시작 부분에 추가되어, 모델이 학생의 오개념을 모방하는 모드와 사실적인 응답을 제공하는 모드 사이를 전환하도록 지시합니다.

- **Performance Highlights**: 이 기법을 추가한 결과, ARC reasoning challenge, TruthfulQA, HaluEval Dial dataset, 그리고 MemoTrap 데이터셋을 포함한 여러 벤치마크 데이터셋에서 모델의 성능이 크게 향상되었습니다. 그러나 이 기법은 LLM의 기본 성능을 완전히 복구하지는 못하였으며, 지속적인 연구가 필요함을 시사합니다.



### Adaptive Collaboration Strategy for LLMs in Medical Decision Making (https://arxiv.org/abs/2404.15155)
- **What's New**: MDAgents(의료 결정 지원 에이전트)는 의료 분야에 대한 재단 모델의 활용성을 향상시키는 새로운 프레임워크입니다. 이 프레임워크는 LLMs(대규모 언어 모델)를 효과적으로 배치하여 복잡한 의료 임무를 수행할 수 있도록 설계되었습니다. 특히, MDAgents는 의료 임무의 복잡성에 맞추어 단독 또는 그룹 협업 구조를 자동으로 할당하는 기능을 갖추고 있어, 실제 의료 결정 과정을 모방합니다.

- **Technical Details**: MDAgents는 다양한 의료 벤치마크(예: MedQA, MedMCQA, PubMedQA, DDXPlus, PMC-VQA, Path-VQA, MedVidQA)에서 최신 LLMs와 함께 평가되었습니다. 이 프레임워크는 다중 모달(Multi-modal) 의료 추론을 요구하는 7개 벤치마크 중 5개에서 최고 성능을 달성했습니다. Ablation Study(제거 연구)는 MDAgents가 협업하는 에이전트의 수를 적절히 조정하여 효율성과 정확도를 최적화하는 능력을 드러냈습니다.

- **Performance Highlights**: MDAgents는 복잡한 의료 환경에서 그룹 의사 결정의 역학을 탐구하며, 협업하는 에이전트들이 어떻게 행동할 수 있는지에 대한 통찰을 제공합니다. 이러한 시나리오에서의 로버스트성(Robustness)을 입증함으로써 복잡한 임상 팀 역학에서 의료 결정을 지원하는 데 기여할 잠재력을 보여주었습니다.



### Do not think pink elephant! (https://arxiv.org/abs/2404.15154)
Comments: This paper is accepted in CVPRW

- **What's New**: 이 논문은 최근의 대규모 모델(Large Models, LMs)들이 인간 지능과 유사한 취약성을 공유하고 있음을 보여줍니다. 특히, '화이트 베어 현상(white bear phenomenon)'이라 불리는 취약성으로, 이 현상은 인간 지능에서도 잘 알려져 있습니다. 연구진은 이 현상의 원인을 분석하고, 이를 바탕으로 대규모 모델의 표현 공간(representation space)을 조사했습니다.

- **Technical Details**: 연구진은 간단한 프롬프트 기반 공격 방법을 제안하여, 모델 제공자의 정책에 금지된 이미지들을 생성할 수 있음을 시연했습니다. 이를 위해 프롬프트(prompt)를 조작하여 모델이 금지된 이미지를 생성하도록 유도했습니다.

- **Performance Highlights**: 또한, 인지 치료 기법에서 영감을 받은 프롬프트 기반 방어 전략을 소개하여, 공격의 성공률을 최대 48.22%까지 감소시키는 데 성공했습니다. 이는 이러한 공격에 대응할 수 있는 유효한 방어 메커니즘이 될 수 있음을 시사합니다.



### Expert Router: Orchestrating Efficient Language Model Inference through  Prompt Classification (https://arxiv.org/abs/2404.15153)
- **What's New**: 이 논문에서는 대규모 언어 모델(Large Language Models, LLMs)의 광범위한 채택과 다양한 작업에 대한 그들의 유용성을 소개하고 있습니다. 특히, 대규모로 이 모델들을 배포하고 서비스하는 것이 큰 도전이 되고 있는데, 이는 LLMs와 연관된 높은 계산 및 메모리 요구 때문입니다. 이러한 제한을 해결하기 위해, 저자들은 여러 전문가 모델을 효율적으로 조정하는 시스템인 'Expert Router'를 도입하였습니다.



### Bias patterns in the application of LLMs for clinical decision support:  A comprehensive study (https://arxiv.org/abs/2404.15149)
- **What's New**: 이번 연구에서는 대형 언어 모델(Large Language Models, LLMs)이 임상 의사 결정 과정에 정보를 제공하는 유력한 후보로 부상하고 있음을 조명합니다. 특히, 보호된 환자 속성(예: 인종)에 따른 사회적 편향 정도 및 아키텍처 설계(Architecture Design), 프롬프트 전략(Prompting Strategies) 같은 설계 선택이 관찰된 편향에 어떤 영향을 미치는지를 집중적으로 다루고 있습니다. 이를 통해 임상 환경에서 LLMs의 역할과 영향을 깊이 있게 이해할 수 있습니다.

- **Technical Details**: 연구팀은 세 개의 질의응답(Question Answering, QA) 데이터셋을 사용하여 여덟 가지 인기 있는 LLMs를 평가했습니다. 이들은 임상 환경에서의 편향을 평가하기 위해 표준화된 임상 시나리오(Clinical Vignettes)를 사용하였으며, 레드팀 전략(Red-teaming Strategies)을 활용하여 인구 통계학적 변수가 LLM 출력에 미치는 영향을 분석했습니다. 본 연구는 일반적 목적의 모델(General-purpose Models)과 임상 트레이닝을 받은 모델(Clinically-trained Models) 모두를 비교 분석하였습니다.

- **Performance Highlights**: 실험 결과, 보호된 그룹 간에 다양한 불균형(일부는 상당한 불균형)이 드러났습니다. 또한, 더 큰 모델이 반드시 덜 편향되어 있지 않으며, 의료 데이터에 특화된 모델이 일반 목적의 모델보다 반드시 나은 것은 아니라는 직관에 반하는 패턴이 관찰되었습니다. 연구는 또한 프롬프트 디자인(Prompt Design)이 편향 패턴에 미치는 영향을 보여주고, 특정 문구가 편향 패턴에 영향을 미칠 수 있으며, 사고의 연쇄(Chain of Thought) 같은 반성적 접근 방식이 편향된 결과를 효과적으로 줄일 수 있음을 시연했습니다.



### Identifying Fairness Issues in Automatically Generated Testing Conten (https://arxiv.org/abs/2404.15104)
Comments: 18 pages, 3 figures, accepted to the 19th Workshop on Innovative Use of NLP for Building Educational Applications

- **What's New**: 이 연구는 자동으로 생성된 테스트 콘텐츠에서의 공정성 문제에 초점을 맞추고 있으며, 특정 인구 통계를 반영하거나 시험 응시자에게 정서적 불쾌감을 줄 수 있는 내용을 식별함으로써 테스트가 의도한 바만을 측정하도록 보장하는 것을 목표로 하고 있습니다. 이러한 콘텐츠는 일반적인 문맥에서 나타나는 편향과 달라 현대 모델에서도 대처하기 어려운 도전을 제시합니다.

- **Technical Details**: 연구팀은 621개의 자동 생성 텍스트를 공정성 측면에서 주석을 달고, 이를 바탕으로 fine-tuning, topic-based classification, 그리고 prompting을 포함하는 다양한 분류 방법을 탐구하였습니다. 특히 few-shot 학습과 자기 수정(self-correcting) 프롬프트를 결합한 방법이 우수한 성능을 보여, 보류된 테스트 세트에서 F1 점수 .791을 기록하였습니다.

- **Performance Highlights**: Prompt self-correction과 few-shot 학습 조합이 가장 효과적인 것으로 나타났으며, F1 점수 .791을 달성했습니다. 비교적 규모가 작은 BERT 및 topic-based 모델들도 도메인 외(out-of-domain) 데이터에서 경쟁력 있는 성능을 보였습니다.



### Multi-view Content-aware Indexing for Long Document Retrieva (https://arxiv.org/abs/2404.15103)
- **What's New**: 이 연구에서는 장문의 문서에서 질문에 답하는 장문 문서 질문 답변 (Long document question answering, DocQA) 시스템을 향상시키기 위해 '멀티뷰 콘텐츠 인식 색인' (Multi-view Content-aware indexing, MC-indexing)을 제안합니다. 기존의 방식은 문서의 구조를 고려하지 않고 일정 길이의 청크로 분할하여 중요 정보를 놓치거나 관련없는 내용을 포함하는 경우가 많았습니다. MC-indexing은 문서의 내용 구조를 인식하여 콘텐츠 청크로 구조화하고, 이를 원문 텍스트, 키워드, 요약 등 다양한 방식으로 표현합니다.

- **Technical Details**: MC-indexing은 훈련이나 미세 조정(Fine-tuning)을 필요로 하지 않으며, 플러그 앤 플레이(Plug-and-play) 기능을 지원해 다양한 리트리버(Retrievers)와 쉽게 통합될 수 있습니다. 이 기술은 문서의 내용 구조와 답변 범위를 포함하는 새로운 장문 DocQA 데이터셋도 제안합니다.

- **Performance Highlights**: MC-indexing은 기존의 청크 생성 스킴에 비해 상당히 높은 리콜(Recall) 성능 향상을 보였습니다. 구체적으로, 1.5, 3, 5, 10의 상위 k-값에서 각각 42.8%, 30.0%, 23.9%, 16.3%의 성능 향상을 기록했습니다. 이 성능 개선은 8개의 널리 사용되는 리트리버 (2개의 희소(Sparse) 및 6개의 조밀(Dense) 리트리버)를 통한 광범위한 실험에서 얻은 평균 결과입니다.



### Enhancing Textual Personality Detection toward Social Media: Integrating  Long-term and Short-term Perspectives (https://arxiv.org/abs/2404.15067)
Comments: 11 pages, 9 figures

- **What's New**: 이 연구는 개인의 장기적 안정된 특성(long-term stable traits)과 단기적 동적 상태(short-term dynamic states) 모두를 포괄적으로 모델링하여 텍스트 기반 성격 검출(textual personality detection)의 가능성을 확장합니다. 새로 개발된 이중 강화 네트워크(Dual Enhanced Network, DEN)는 사용자의 장기 및 단기 성격을 함께 모델링하여 더욱 정확하고 포괄적인 성격 분석을 제공합니다.

- **Technical Details**: DEN은 세 가지 주요 구성 요소로 이루어져 있습니다: 장기 성격 인코딩(Long-term Personality Encoding), 단기 성격 인코딩(Short-term Personality Encoding), 그리고 양방향 상호작용 구성요소(Bi-directional Interaction component). 장기 성격 인코딩은 사용자의 장기적 성격 특성을 효과적으로 모델링하며, 단기 성격 인코딩은 사용자의 단기적 변화를 포착합니다. 양방향 상호작용은 두 성격 측면의 통합을 용이하게 해, 사용자의 성격을 종합적으로 표현할 수 있게 돕습니다.

- **Performance Highlights**: 실험 결과는 DEN 모델이 두 성격 검출 데이터셋(personality detection datasets)에서 효과적이라는 것을 증명합니다. 연구는 안정적 및 동적 성격 특성을 모두 고려하는 것이 텍스트 기반 성격 검출에 있어 중요하다는 혜택을 보여 줍니다.



### Multi-Head Mixture-of-Experts (https://arxiv.org/abs/2404.15045)
- **What's New**: 새로운 Multi-Head Mixture-of-Experts (MH-MoE) 모델이 제안되었습니다. 이 모델은 각 토큰을 여러 서브 토큰으로 분할하고, 다양한 전문가 그룹에 할당하여 병렬 처리하는 multi-head mechanism을 채택하고 있습니다. 이를 통해 모델은 다양한 표현 공간에서 정보를 통합하여 처리할 수 있으며, 전문가들의 활성화를 크게 향상시켜 컨텍스트 이해를 깊게 하고 오버피팅(overfitting)을 완화할 수 있습니다.

- **Technical Details**: MH-MoE는 기존의 Sparse Mixtures of Experts (SMoE) 모델을 기반으로 하여, 각 토큰을 서브-토큰들로 나누고 이들을 독립적으로 다양한 전문가에게 할당하는 구조를 가집니다. 이러한 구조는 각기 다른 전문가들의 지식을 통합하고 세밀한 분석을 가능하게 합니다. 또한, MH-MoE는 다른 SMoE 최적화 방법과는 독립적으로 구현 가능하여, 다른 SMoE 모델과의 통합이 용이합니다.

- **Performance Highlights**: MH-MoE는 영어 중심 언어 모델링, 다국어 언어 모델링, 가려진 다중 모달리티 모델링 작업에 걸쳐 효과적임이 입증되었습니다. 특히, 다양한 SMoE 모델들과의 결합을 통해 성능 향상이 가능하며, 전문가 활성화에서의 중요한 개선을 통해 컨텍스트 이해력을 개선하였습니다.



### TAXI: Evaluating Categorical Knowledge Editing for Language Models (https://arxiv.org/abs/2404.15004)
- **What's New**: 이 연구에서는 지식 편집(knowledge editing) 영역에 새로운 벤치마크 데이터셋 TAXI를 소개했습니다. TAXI는 언어 모델(language models)에 새로운 사실을 주입하여 그 사실성(factuality)을 향상시키는 것을 목표로 하며, 특히 일관성(consistency)을 평가하기 위해 특별히 만들어졌습니다. 이 데이터셋은 다양한 카테고리, 주제, 그리고 속성을 포함하는 11,120개의 다중 선택 질의(multiple-choice queries)를 포함합니다.

- **Technical Details**: TAXI 데이터셋은 41개의 카테고리(e.g., Dogs), 164개의 주제(e.g., Labrador), 그리고 183개의 속성(e.g., is a mammal)을 포함합니다. 이 연구는 언어 모델에서 주제의 카테고리를 편집할 때 속성들이 적절히 편집되는지를 평가하는 데에 TAXI를 사용하였습니다.

- **Performance Highlights**: 평가 결과, 현재의 편집 도구들(editor tools)은 인간의 기준(human baselines)에 비해 상당히 낮은 일관성을 보여줬습니다. 그러나 이들은 무작위(random) 수준보다는 약간 높은 일관성을 달성했으며, 비전형적인 주제(atypical subjects)에서 편집을 수행할 때 일관성이 더욱 달성 가능하다는 것을 발견했습니다.



### Comparison of Current Approaches to Lemmatization: A Case Study in  Estonian (https://arxiv.org/abs/2404.15003)
Comments: 6 pages, 2 figures

- **What's New**: 이 연구는 에스토니아어에 대한 세 가지 다른 형태소 분석(lemmatization) 접근 방식을 평가합니다: 생성적 문자 수준 모델(Generative character-level models), 패턴 기반 단어 수준 분류 모델(Pattern-based word-level classification models), 그리고 규칙 기반 형태학적 분석(rule-based morphological analysis).

- **Technical Details**: 연구에 따르면, 상대적으로 작은 크기의 생성적 모델이 EstBERT를 기반으로 한 패턴 기반 분류 모델보다 일관되게 더 우수한 성능을 보였습니다. 또한, 세 모델이 만든 오류의 중복이 상대적으로 적다는 것이 관찰되었고, 이는 다양한 접근 방식의 앙상블(ensemble)이 성능 향상으로 이어질 수 있음을 시사합니다.

- **Performance Highlights**: 생성적 모델은 크기에 비해 놀라운 성능을 보여주며, 패턴 기반 모델보다 일관되게 높은 결과를 제공합니다. 모든 세 모델에서 발생하는 오류의 작은 중복은 각기 다른 접근 방식을 조합할 때 성능이 향상될 수 있는 가능성을 보여줍니다.



### Transformers Can Represent $n$-gram Language Models (https://arxiv.org/abs/2404.14994)
- **What's New**: 이 논문은 트랜스포머 언어 모델(Language Models, LM)과 $n$-gram 언어 모델의 관계에 초점을 맞추고 있습니다. 기존의 연구가 언어 수용(language acceptance) 측면에서 트랜스포머 아키텍처를 분석하는 데 초점을 맞췄다면, 이 논문은 언어 모델이 문자열에 대한 확률 분포(probability distributions)를 어떻게 나타내는지를 탐구합니다.

- **Technical Details**: 트랜스포머 LMs는 하드(hard) 또는 스파스(sparse) 어텐션 메커니즘을 사용하여 모든 $n$-gram LM을 정확히 표현할 수 있다는 것을 보여줍니다. 이는 트랜스포머 모델이 문자열에 대한 확률 분포를 표현하는 데 사용할 수 있는 메커니즘을 이해하는 첫 단계를 제공합니다.

- **Performance Highlights**: 이 연구는 트랜스포머 LMs의 확률적 대표력(probabilistic representational capacity)에 대한 구체적인 하한선(lower bound)을 제시함으로써, $n$-gram 모델들이 가지는 간단하면서도 역사적으로 중요한 클래스를 완벽하게 표현할 수 있음을 보여줍니다.



### Achieving >97% on GSM8K: Deeply Understanding the Problems Makes LLMs  Perfect Reasoners (https://arxiv.org/abs/2404.14963)
- **What's New**: 이 논문은 복잡한 추론 과제를 해결할 때 발생하는 여러 오류를 극복하기 위해 새로운 프롬프트 전략인 'Deeply Understanding the Problems (DUP)'을 제안합니다. DUP는 문제를 심층적으로 이해하고, 이를 바탕으로 정답을 도출하는 세 가지 단계로 구성되어 있습니다.

- **Technical Details**: 'Deeply Understanding the Problems' (DUP) 프롬프팅은 문제의 핵심 질문을 추출하고(`extract the core question`), 핵심 질문에 기반한 문제 해결 정보를 찾으며(`find out problem-solving information`), 대규모 언어 모델(Large Language Models, LLMs)을 사용하여 답을 생성하고 추출하는(`generate and extract answers`) 세 단계로 이루어집니다.

- **Performance Highlights**: DUP는 10개의 다양한 추론 데이터셋에서 평가되었으며, 실험 결과 DUP 프롬프팅은 Zero-Shot Chain of Thought (CoT) 방식을 모든 데이터셋에서 크게 뛰어넘는 성능을 보였습니다. 특히, SVAMP에서는 90.4%에서 94.2%로, GSM8K에서는 94.6%에서 97.1%로 성능이 상당히 향상되었으며, 이는 최고 성능 (state-of-the-art)에 해당합니다.



### Does It Make Sense to Explain a Black Box With Another Black Box? (https://arxiv.org/abs/2404.14943)
Comments: This article was originally published in French at the Journal TAL. VOL 64 n{\deg}3/2023. arXiv admin note: substantial text overlap with arXiv:2402.10888

- **What's New**: 이 연구는 자연어 처리(Natural Language Processing, NLP)에서 블랙 박스 분류기(Black-box classifiers)를 설명하는 카운터팩추얼 설명(Counterfactual Explanations)의 사용에 초점을 맞추고 있습니다. 특히, 투명(Transparent) 방법과 불투명(Opaque) 방법의 두 가지 방법론을 비교 분석하였습니다. 이 연구는 NLP의 클래식한 작업 세 가지에서 이 두 방법의 효과를 탐구하면서, 불투명 접근법이 추가적인 복잡성을 도입하지만 가짜 뉴스 감지나 감정 분석과 같은 하류 작업(Downstream tasks)에서 큰 성능 향상을 제공하지 않는다는 사실을 발견했습니다.

- **Technical Details**: 저자들은 문서(Document)가 블랙 박스에 의해 다르게 분류될 때까지 표적 문서를 반복적으로 변형(Perturbing)하는 방식으로 카운터팩추얼 설명을 찾습니다. 투명 방법은 단어를 추가, 제거 또는 교체하는 방식으로 목표 문서를 변형하는 반면, 불투명 방법은 목표 문서를 해석할 수 없는 잠재 공간(Latent Space)으로 투영하여 그 공간에서 변형을 수행합니다.

- **Performance Highlights**: 연구 결과, 불투명 접근 방식은 단순하고 직관적이지 않은 또 다른 블랙 박스를 사용하여 블랙 박스를 설명하는 데 투명 방법보다 유의미한 성능 이점을 제공하지 않는 것으로 나타났습니다. 특히 가짜 뉴스 탐지나 감정 분석과 같은 작업에 있어 추가적인 복잡성은 실질적인 성능 개선 없이 단지 과정을 더 어렵게 만듭니다. 이러한 관찰은 또 다른 블랙 박스를 사용하여 블랙 박스를 설명하는 것이 타당한지에 대한 논의를 촉발합니다.



### Pillars of Grammatical Error Correction: Comprehensive Inspection Of  Contemporary Approaches In The Era of Large Language Models (https://arxiv.org/abs/2404.14914)
- **What's New**: 이 논문에서는 문법 오류 교정(GEC)에 관한 실험적 연구를 수행하며, 단일 모델 시스템의 미묘한 차이를 조사하고, 앙상블(ensembling) 및 순위 매기기(ranking methods) 방법의 효율성을 비교하며, 대규모 언어 모델들(large language models)을 GEC에 적용하는 방법을 탐구했습니다. 특히 단일 모델 시스템, 앙상블의 일부, 그리고 순위 결정 방법으로 사용되었습니다.

- **Technical Details**: 연구팀은 CoNLL-2014-test와 BEA-test 데이터셋에서 각각 72.8 및 81.4의 F_0.5 점수를 달성하여 새로운 최고 성능(state-of-the-art performance)을 설정했습니다. 이 연구에서는 다양한 대규모 언어 모델들을 단일 모델 시스템으로 사용하는 것뿐만 아니라, 여러 모델을 결합한 앙상블과 이러한 앙상블의 결과를 평가 및 순위를 매기기 위한 방법으로도 활용되었습니다.

- **Performance Highlights**: 새로운 최고 성능은 F_0.5 점수에서 확인할 수 있으며, CoNLL-2014-test에서 72.8, BEA-test에서 81.4를 달성함으로써 이전 연구들을 뛰어넘었습니다. 이러한 성과는 문법 오류 교정 분야에서의 기술적 진보를 대표합니다. 또한, 연구진은 후속 연구를 지원하고 연구의 재현 가능성을 보장하기 위해 코드, 훈련된 모델들, 시스템의 출력물을 공개했습니다.



### Beyond the Speculative Game: A Survey of Speculative Execution in Large  Language Models (https://arxiv.org/abs/2404.14897)
Comments: 10 pages, 4 figures, 1 table, rejected from IJCAI 2024, revision in progress

- **What's New**: 이 논문은 대규모 언어 모델(Large Language Models, LLMs)의 추론 효율성에 대해 다루며, 특히 자연어 처리(Natural Language Processing, NLP) 분야에서의 '추측 실행(speculative execution)'을 도입하여 디코딩 속도를 향상시키는 새로운 접근 방식을 제안합니다. 이 연구는 추측 실행을 적용한 LLM 디코딩의 첫 번째 설문 조사 논문으로, 현재 연구 상황을 정리하고 향후 발전 방향을 제시합니다.

- **Technical Details**: 'draft-then-verify' 방식에서는, 휴리스틱(heuristics)을 사용하여 토큰 시퀀스를 신속하게 초안(draft)으로 작성한 다음, LLM을 사용하여 병렬로 검증(verify)합니다. 이로 인해 복잡하고 시간이 많이 걸리는 순차적 추론이 병렬 처리되어 디코딩 속도가 크게 증가합니다. 또한 이 논문은 병렬 디코딩, 추측적 디코딩 등 LLMs에서의 추측 실행에 관한 문헌을 포괄적인 프레임워크와 체계적인 분류(taxonomy)로 검토하고 통합합니다.

- **Performance Highlights**: 이 연구는 연구 분야에서 추측 실행이 LLM 디코딩 속도를 향상시키는 주요 방법으로 자리 잡을 수 있는 가능성을 강조하며, 프로세스의 복잡성을 병렬 처리하여 실제 응용에서의 유용성을 대폭 개선했습니다. 이러한 개선은 특히 GPT-4와 같이 일일 수십억 건의 요청을 처리해야 하는 대규모 모델에 필수적입니다.



### Language in Vivo vs. in Silico: Size Matters but Larger Language Models  Still Do Not Comprehend Language on a Par with Humans (https://arxiv.org/abs/2404.14883)
- **What's New**: 이 연구는 대규모 언어 모델(LLM; Large Language Models)이 언어의 한계를 이해함으로써 자연 언어의 이론으로서 기능할 수 있는지를 탐구합니다. 특히, LLM의 크기를 조정하는 것이 인간과 모델 간의 성능 차이를 해결할 수 있는지 여부를 중점적으로 다룹니다. 이 연구에서는 Bard, ChatGPT-3.5, ChatGPT-4 (각각 1370억, 1750억, 1.5조 파라미터를 가진 모델) 과 같은 서로 다른 LLM 가족의 모델을 사용하여 언어적 타당성 판단 작업을 시험하였습니다.

- **Technical Details**: 연구는 anaphora, center embedding, comparatives 및 negative polarity를 포함하는 문법성 판단 작업에 대해 세 가지 LLM을 테스트했습니다. 총 1,200개의 판단이 수집되어 정확성, 안정성 및 프롬프트를 반복 제시함에 따른 정확도 개선이 평가되었습니다. 가장 성능이 좋은 LLM인 ChatGPT-4의 결과는 같은 자극에 대한 인간 80명의 결과와 비교되었습니다.

- **Performance Highlights**: ChatGPT-4는 크기 증가가 성능 향상으로 이어질 수 있음을 보여주었으나, LLM은 여전히 인간처럼 문법적(非)타당성에 민감하지 않습니다. 이에 따라 크기 조정만으로 이 문제를 해결할 가능성은 존재하지만, 그 가능성은 희박하다고 판단됩니다. 연구는 인간의 언어 학습(in vivo)과 기계의 언어 학습(in silico)을 비교하여 증거의 유형, 자료의 빈곤성, 그리고 언어적 참조가 불투명할 때 발생하는 의미의 환상 같은 세 가지 주요 차이점을 확인합니다.



### Simple, Efficient and Scalable Structure-aware Adapter Boosts Protein  Language Models (https://arxiv.org/abs/2404.14850)
Comments: 30 pages, 4 figures, 8 tables

- **What's New**: 새로운 SES-Adapter는 단백질 언어 모델을 위한 구조 인식 표현을 생성하기 위해 PLM(embeddings) 임베딩과 구조적 시퀀스 임베딩을 통합합니다. 이 방법은 다양한 PLM 아키텍처와 다양한 태스크에서 호환 가능하며 훨씬 개선된 학습 효율을 제공합니다.

- **Technical Details**: SES-Adapter는 기존의 PLM에 파라미터 효율적인 (Parameter-Efficient) 파인 튜닝 기법을 적용하여 생명공학 태스크의 성능을 향상시킵니다. 2가지 유형의 폴딩 구조와 9개의 벤치마크 데이터셋을 통한 평가에서 SES-Adapter는 최대 11%의 성능 향상과 함께 훈련 속도를 최대 1034% 가속화하였습니다.

- **Performance Highlights**: SES-Adapter는 기본 PLM과 비교하여 다운스트림 태스크(downstream task) 성능을 최대 11% 향상시키며 평균 3%의 성능 개선을 보여줍니다. 또한, 훈련 속도는 최대 1034%, 평균 362% 향상되었고, 수렴율(convergence rate)도 약 2배 개선되었습니다.



### Sentence-Level or Token-Level? A Comprehensive Study on Knowledge  Distillation (https://arxiv.org/abs/2404.14827)
- **What's New**: 이 연구에서는 지식 증류 (knowledge distillation)의 두 가지 주요 방법, 문장 수준 (sentence-level)과 토큰 수준 (token-level) 증류를 비교 분석하여, 각각의 방법이 복잡한 시나리오와 간단한 시나리오에서 어떻게 다르게 작동하는지를 설명합니다. 또한, 새로운 하이브리드 방식을 도입하여 두 방법의 장점을 결합하며, 이 방법은 기존의 문장 수준 또는 토큰 수준 증류 방법보다 우수한 성능을 보였습니다.

- **Technical Details**: 이 연구는 각각의 증류 방법의 성능을 다양한 학생 모델의 크기 (model size), 텍스트의 복잡성 (complexity), 디코딩 절차의 난이도 (difficulty of decoding procedure)를 변화시켜가며 체계적으로 분석합니다. 하이브리드 방식은 게이팅 메커니즘 (gating mechanism)을 통해 문장 수준 증류와 토큰 수준 증류를 결합합니다.

- **Performance Highlights**: 하이브리드 방식은 기존의 문장 수준 및 토큰 수준 증류 방법 뿐만 아니라 이전 연구들보다도 뛰어난 성능을 보여주었습니다. 이는 하이브리드 방식이 각각의 증류 방법의 이점을 효과적으로 활용할 수 있음을 시사합니다.



### Pattern-Aware Chain-of-Thought Prompting in Large Language Models (https://arxiv.org/abs/2404.14812)
- **What's New**: 이 논문에서는 기존의 체인-오브-사우트(Chain-of-Thought, CoT) 프롬프팅 방식을 개선하는 새로운 접근법인 패턴 인식(Pattern-Aware) CoT 방식을 제안합니다. 이 방식은 특히 다양한 데모 패턴을 고려하여 중간 단계의 추론 과정 및 스텝 길이(step length)에 초점을 맞춥니다. 이는 데모가 유발할 수 있는 편향성을 완화하고 더 다양한 시나리오에 대한 일반화 능력을 개선하는 데 도움이 됩니다.

- **Technical Details**: PA-CoT 방식은 중간 단계의 추론 과정과 스텝 길이 같은 패턴을 통합하여 프롬프팅을 수행합니다. 이는 기존 방식에서 보다 세밀한 조정을 가능하게 하며, 데모의 정확성과 의미론(semantics)적 측면을 넘어서는 새로운 차원의 개발을 허용합니다. 연구진은 오픈소스 대형 언어 모델(LLMs) 두 가지를 사용하여 9개의 추론 벤치마크 태스크에서 실험을 수행했습니다.

- **Performance Highlights**: PA-CoT 방식은 전통적인 CoT 접근법에 비해 눈에 띄게 높은 추론 성능을 보였으며, 오류에 대한 강인성을 보여줍니다. 이 접근방식이 실제 모델의 어플리케이션에 적용될 시 대규모 문제 해결에 있어서 보다 높은 정확도와 효율성을 기대할 수 있습니다. 또한, 연구진은 코드를 공개할 예정임을 밝혔습니다.



### A Survey of Large Language Models on Generative Graph Analytics: Query,  Learning, and Applications (https://arxiv.org/abs/2404.14809)
Comments: 31 pages including references, 22 figures

- **What's New**: 이 연구는 최근 대규모 언어 모델(LLMs)이 다양한 NLP 및 멀티모드 작업에서 그래프 데이터를 다루는 데 어떻게 활용되고 있는지 조사합니다. 특히, LLM이 그래프 학습 모델의 필요성을 제거하고 수동 주석 비용을 절감하는 등 그래프 작업의 일반화에 대한 도전을 해결하는 데 있어 가지는 이점을 강조합니다.

- **Technical Details**: 연구는 LLM 기반 생성적 그래프 분석(LLM-GGA)을 세 가지 카테고리로 나누어 다룹니다: LLM 기반 그래프 쿼리 처리 (LLM-GQP), LLM 기반 그래프 추론 및 학습 (LLM-GIL), 그리고 그래프-LLM 기반 응용 프로그램입니다. LLM-GQP는 그래프 분석 기술과 LLM 프롬프트의 통합에 중점을 두며, 그래프 이해와 지식 그래프(KG) 기반 증강 검색을 포함합니다. LLM-GIL은 그래프 학습, 그래프 형태 추론, 그래프 표현 등을 포함하여 그래프를 통한 학습 및 추론에 초점을 맞춥니다.

- **Performance Highlights**: LLM을 사용하여 다양한 그래프 다운스트림 작업을 처리하기 위해 통합된 유용한 프롬프트를 요약하고, LLM 모델 평가, 벤치마크 데이터셋/작업, LLM 모델의 장단점 분석을 제공합니다. 또한 LLM과 그래프 분석의 흥미로운 융합 연구 영역에서의 개방 문제와 미래 방향을 탐구합니다.



### Talk Too Much: Poisoning Large Language Models under Token Lim (https://arxiv.org/abs/2404.14795)
Comments: 20 pages

- **What's New**: 보다 은밀한 공격 전략을 도입한 BrieFool 프레임워크는 LLMs (Large Language Models) 내의 복용 공격(poisoning attack)에 적용되어 타겟 유저들이 비용 절약을 위해 사용하는 제너레이션/아웃풋 (generation/output) 조건-토큰 (condition-token) 제한을 트리거로 사용합니다. 이 방식은 기존의 고정 트리거 사용 방식의 한계를 극복하며 실제 상황에서의 응용성을 높입니다.

- **Technical Details**: BrieFool은 공격 모델이 토큰 제한이 있는 출력에서만 해로운 반응을 보이고, 제한이 없을 때는 정상적으로 작동하는 방식입니다. 이를 달성하기 위해, 효율적인 명령어 샘플링과 데이터 생성을 통해 LLMs의 동작에 영향을 미치도록 설계되었습니다. 각종 안전 도메인 및 지식 도메인에 걸쳐 유효성이 확인될 수 있는 실험적 접근이 포함되어 있습니다.

- **Performance Highlights**: BrieFool은 공격 성공률(ASR, Attack Success Rate) 100%와 평균 유해성 점수(HS, Harmfulness Score) 9.28/10을 GPT-3.5-turbo를 대상으로 한 20개의 생성된 포이즈닝(poisoning) 예제에서 달성하였습니다. 이는 토큰 제한 조건 하에서의 유해한 행동을 유효하게 집행하는 동시에, 통상적 상황에서는 좋은 성능을 유지할 수 있음을 보여줍니다.



### Med42 -- Evaluating Fine-Tuning Strategies for Medical LLMs:  Full-Parameter vs. Parameter-Efficient Approaches (https://arxiv.org/abs/2404.14779)
Comments: Published at AAAI 2024 Spring Symposium - Clinical Foundation Models

- **What's New**: 이 연구에서는 의료 분야의 Large Language Models(LLMs)에 대해 두 가지 주요 파인 튜닝 방법론인 전체 파라미터 파인 튜닝(full-parameter fine-tuning)과 파라미터 효율적 튜닝(parameter-efficient tuning)을 종합적으로 분석하고 비교하였습니다. 특히, 의료 지식 검색, 추론 및 질문 답변 기능을 향상시키기 위해 Llama-2 아키텍처를 기반으로 한 일련의 LLMs를 개발하고 수정하였습니다.



### CT-Agent: Clinical Trial Multi-Agent with Large Language Model-based  Reasoning (https://arxiv.org/abs/2404.14777)
- **What's New**: 이 연구는 대규모 언어 모델(Large Language Models, LLMs)과 멀티-에이전트 시스템이 임상 시험 작업에서 직면한 도전을 해결하기 위해 'CT-에이전트(Clinical Agent System)'를 제안합니다. 이 시스템은 최신 의료 데이터에 기반하여 임상 시험 도구의 접근성과 유용성을 향상시키는 것을 목표로 하고 있습니다.

- **Technical Details**: CT-에이전트는 GPT-4, 멀티-에이전트 아키텍처(multi-agent architectures), LEAST-TO-MOST 기술 및 ReAct 추론 기술(reasoning technology)을 통합하여 설계되었습니다. 이 시스템은 임상 시험 과정 전체를 자동으로 관리하며, 언어 모델의 임상 맥락에서의 성능을 향상시킵니다.

- **Performance Highlights**: CT-에이전트는 계산 벤치마크(computational benchmarks)와 전문가 의견(expert feedback)을 포함한 평가에서 효율성 향상을 입증하였습니다. 이는 임상 시험 프로세스를 혁신적으로 개선할 수 있는 새로운 가능성을 제시합니다.



### Simulating Task-Oriented Dialogues with State Transition Graphs and  Large Language Models (https://arxiv.org/abs/2404.14772)
- **What's New**: 이 연구에서는 신규 데이터 생성 접근법인 SynTOD를 탐구하고 있으며, 이는 크라우드 소싱(crowdsourcing)이나 실제 데이터에 의존하지 않고도 복잡한 작업을 처리할 수 있는 종단 간 업무 지향 대화(Task-Oriented Dialogue, TOD) 시스템 개발을 가능하게 합니다. SynTOD는 TOD 시스템이 원하는 행동을 정의하는 상태 전이 그래프(state transition graph)를 사용하고, 큰 언어 모델(large language models, LLMs)을 활용한 랜덤 워크(random walks)와 응답 시뮬레이션(response simulation)을 통해 다양하고 구조화된 대화를 생성합니다.

- **Technical Details**: SynTOD는 상태 전이 그래프를 사용하여 TOD 시스템에서 원하는 행동을 정의하며, LLMs을 통한 응답 시뮬레이션을 통해 다양한 대화를 생성합니다. 의도 분류(intent classification), 슬롯 채우기(slot filling), 회화형 질의응답(conversational question-answering), 검색 증강 응답 생성(retrieval-augmented response generation) 등의 작업을 처리할 수 있습니다. 실험에서는 그래프 가이드 응답 시뮬레이션을 사용하여 의도 분류, 슬롯 채우기 및 응답 관련성이 단순한 단일 프롬프트 시뮬레이션 대화 대비 현저한 개선을 보였습니다.

- **Performance Highlights**: SynTOD를 사용한 대화는 의도 분류, 슬롯 채우기 및 응답 관련성에서 높은 성능을 보였습니다. 또한 다양한 기반 및 지침 조정된 LLMs(base and instruction-tuned LLMs)의 TOD 효율성을 평가하고, 이들이 생성한 합성 대화(synthetic conversations)를 사용하여 종단 간 TOD의 효과를 조사하였습니다. 이 연구는 도메인 특정 TOD 시스템의 빠른 개발과 평가로 이어질 수 있는 길을 제시합니다.



### Retrieval Augmented Generation for Domain-specific Question Answering (https://arxiv.org/abs/2404.14760)
Comments: AAAI 2024 (Association for the Advancement of Artificial Intelligence) Scientific Document Understanding Workshop

- **What's New**: 이 논문은 특정 분야(예: 금융, 건강 관리, 교육, 고객 서비스)에 특화된 지식과 용어를 이해하는 데 필요한 새로운 프레임워크를 제안합니다. Adobe 제품에 대한 내부 질의응답(Question Answering, QA) 시스템을 구축하면서, 대규모 질의응답 데이터베이스를 컴파일하고 대규모 언어 모델(Large Language Model)의 검색을 고려한 세밀한 조정(fine-tuning) 방법을 개발했습니다.

- **Technical Details**: 이 연구는 대규모 언어 모델을 활용하여 특정 도메인 지식을 더 잘 이해하도록 하는 방법에 초점을 맞추고 있습니다. 특히, 검색 정보(retrieval information)를 최신 상태로 유지하면서 생성 중 환각(hallucinations)을 줄이는 것이 주요 목표입니다. 대규모 QA 데이터베이스 구축 및 검색자(retriever)의 파인 튜닝을 통해 최종 생성물의 큰 개선을 보여줍니다.

- **Performance Highlights**: 검색자가 파인 튜닝되었을 때 최종 생성물의 성능이 크게 향상됨을 입증하였으며, 이 방법은 문맥적 근거(contextual grounding)를 제공하면서 생성 도중 발생할 수 있는 환각을 감소시킵니다.



### Generate-on-Graph: Treat LLM as both Agent and KG in Incomplete  Knowledge Graph Question Answering (https://arxiv.org/abs/2404.14741)
- **What's New**: 이 논문에서는 지식 그래프(Knowledge Graphs, KGs)를 완전하게 포함하지 않는 문제 상황에서 대규모 언어 모델(Large Language Models, LLMs)의 지식 통합 능력을 평가하기 위해 불완전한 지식 그래프 상의 질문 답변(Incomplete Knowledge Graph Question Answering, IKGQA)을 제안합니다. 기존 연구들이 완전한 KGs에서 평가되었던 것과 달리, 이 연구는 KGs가 모든 질문에 필요한 사실을 포함하지 않는 현실적 상황을 모사합니다.

- **Technical Details**: 제안된 방법은 Generate-on-Graph (GoG)로, 이는 훈련 없이도 새로운 사실적 트리플을 생성할 수 있는 접근법입니다. GoG는 선택(selecting), 생성(generating), 답변(answering)의 프레임워크를 사용하여, LLM을 단지 KGs를 탐색하는 에이전트로만 활용하지 않고, 탐색된 하위 그래프와 내재된 지식을 바탕으로 새로운 사실을 생성할 수 있는 지식 그래프로도 활용합니다.

- **Performance Highlights**: 실험 결과는 두 데이터셋을 사용하여 GoG가 IKGQA 문제를 어느 정도 해결할 수 있음을 보여줍니다. 이는 거의 모든 기존 방법들이 IKGQA에서 잘 수행하지 못하는 반면, GoG는 유의미한 성능을 나타냈습니다.



### Modeling the Sacred: Considerations when Using Considerations when Using  Religious Texts in Natural Language Processing (https://arxiv.org/abs/2404.14740)
Comments: Findings of NAACL2024

- **What's New**: 이 논문은 자연어 처리(Natural Language Processing, NLP)에서 종교 텍스트의 사용에 관한 것으로, NLP의 윤리에 특별한 관심을 가지고 있습니다. 종교 텍스트는 문화적으로 중요한 가치를 표현하며, 기계 학습 모델은 훈련 데이터에 인코딩된 문화적 가치를 재현하는 경향이 있습니다.

- **Technical Details**: 종교 텍스트의 번역은 언어 데이터가 부족할 때 NLP 연구자들에 의해 자주 사용됩니다. 이는 원래의 사용 목적과 동기에서 벗어난 재사용을 의미하며, 종종 새로운 추종자를 유치하는 것을 목적으로 합니다. 논문은 NLP가 이러한 텍스트를 사용함으로써 모델의 편향을 넘어서는 고려사항들이 있음을 주장합니다.

- **Performance Highlights**: 이러한 고려사항에는 데이터 프로비넌스(data provenance), 문화적 맥락, 전도(proselytism)의 용도가 포함됩니다. 연구자의 위치성(researcher positionality)과 주변화된 언어 및 종교 공동체의 관점에 대한 더 많은 고려를 촉구합니다.



### Insights into Alignment: Evaluating DPO and its Variants Across Multiple  Tasks (https://arxiv.org/abs/2404.14723)
- **What's New**: 이 연구는 Direct Preference Optimization (DPO) 방법과 같은 감독 없는 (unsupervised) 최적화 기법들이 인간의 선호도를 기반으로 하는 모델 튜닝에서 어떻게 적용될 수 있는지 탐구합니다. 특히, 이 연구는 Supervised Fine-Tuning (SFT) 과정을 포함하거나 생략하는 세 가지 시나리오와 각각의 시나리오에서 독립적으로 튜닝된 모델들이 다양한 인공지능 태스크에서 어떻게 수행되는지 평가합니다.

- **Technical Details**: 연구는 세 가지 시나리오—SFT 유지, SFT 생략, 그리고 SFT 생략 및 지시에 따른 튜닝된 모델 사용을 통해 alignment methods의 성능을 비교합니다. 다양한 분야에 걸친 13개 벤치마크를 사용하여 평가가 수행됩니다. 이 연구는 주로 Dialogue Systems, Reasoning, Mathematical Problem Solving, Question Answering, Truthfulness, 그리고 Multitask Understanding의 성능을 다룹니다.

- **Performance Highlights**: 연구 결과에 따르면, alignment 방법은 수행 중 SFT 과정을 생략할 경우 Mathematical Problem Solving 과 Truthfulness 의 향상에 영향을 미치지만 Reasoning 태스크에서는 제한적인 효과가 있음을 보여줍니다. 특히, instruction-tuned 모델을 사용하는 경우 Truthfulness가 눈에 띄게 개선되었습니다. 뿐만 아니라, 작은 트레이닝 데이터 세트를 사용할 때 최적의 성능을 나타내는 것으로 관찰되었습니다.



### Bayesian Example Selection Improves In-Context Learning for Speech,  Text, and Visual Modalities (https://arxiv.org/abs/2404.14716)
Comments: 16 pages, 6 figures

- **What's New**: 이 논문은 인컨텍스트 학습(In-Context Learning, ICL)을 위한 새로운 베이지언 인-컨텍스트 예시 선택 방법(Bayesian in-Context example Selection, ByCS)을 제안합니다. 이 방법은 테스트 입력에 조건화된 역추론(Inverse Inference)을 중점으로 하여, 정확한 역추론 확률이 높은 예제들을 선택합니다. 이는 다양한 태스크와 모달리티(Modalities)에서의 ICL 성능을 향상시키기 위함입니다.

- **Technical Details**: ByCS는 베이즈 정리(Bayes' theorem)를 확장해서, 인-컨텍스트 예시들에 조건화된 추론 확률을 사용합니다. 이 방법은 테스트 입력에 기반한 역추론 확률(likelihood)을 계산하고, 이를 통해 높은 역추론 결과를 보이는 예제들을 인-컨텍스트 시 학습에 사용합니다. 이러한 접근은 추론 확률(Posterior Probability)의 정확도를 높이는데 도움을 줍니다.

- **Performance Highlights**: ByCS 방법은 음성(Speech), 텍스트(Text), 이미지(Image) 예제를 포함한 다양한 크로스-태스크(Cross-Task) 및 크로스-모달리티(Cross-Modality) 실험에서 그 효능과 강인성을 입증하였습니다. 실험 결과는 ByCS가 다양한 모델, 태스크, 모달리티에서 효과적임을 보여주었습니다.



### MisgenderMender: A Community-Informed Approach to Interventions for  Misgendering (https://arxiv.org/abs/2404.14695)
Comments: NAACL 2024

- **What's New**: 이 연구는 텍스트 기반의 잘못된 성별 지칭(misgendering)에 대한 개입을 촉진하기 위한 첫 번째 연구로, 미국 내 성별 다양성을 가진 개인들을 대상으로 한 설문 조사를 통해 자동화된 개입에 대한 관점을 조사합니다. 그 결과를 바탕으로, 잘못된 성별 지칭을 감지하고 수정하는 두 가지 하위 과제(sub-tasks)로 구성된 새로운 평가 데이터 셋, 'MisgenderMender'를 소개합니다.

- **Technical Details**: MisgenderMender 데이터셋은 비 cisgender 대중 인물들에 대한 소셜 미디어 콘텐츠와 LLM(Large Language Models, 대규모 언어 모델) 생성 텍스트 3,790건을 포함하며, 잘못된 성별 지칭의 감지 및 LLM 생성 텍스트에서의 수정을 위해 추가적으로 주석이 붙어 있습니다. 데이터셋은 감지(detecting) 과제와 수정(correcting) 과제를 모두 포함합니다.

- **Performance Highlights**: 이 데이터셋을 활용하여 기존의 NLP(Natural Language Processing, 자연어 처리) 시스템을 평가하고, 미래의 모델이 해결해야 할 도전 과제들을 강조합니다. 완전한 데이터셋, 코드, 그리고 데모는 공개적으로 제공되어 연구와 개발에 사용될 수 있습니다.



### Automated Multi-Language to English Machine Translation Using Generative  Pre-Trained Transformers (https://arxiv.org/abs/2404.14680)
- **What's New**: 이 연구에서는 GPT 모델을 이용한 자동 제로 샷 (zero shot) 블랙박스 (black-box), 문장 별 다국어 번역 성능을 분석하였습니다. 특별히 16가지의 오픈 소스 GPT 모델들을 벤치마크하여, 50개 비영어권 언어를 영어로 번역하는 성능을 평가하였습니다. 이러한 평가는 Huggingface의 LLM 보관소에서 제공하는 GPT 모델을 사용하여 진행되었습니다.

- **Technical Details**: 모든 번역 작업은 단일 Nvidia A100 GPU에서 로컬로 (locally) 수행되었으며, 번역 정확도를 측정하기 위해 BLEU, GLEU, METEOR, chrF 등의 텍스트 중복 측정 지표를 사용하였습니다. 최고의 성능을 보인 GPT 모델은 영어 텍스트 번역에서 BLEU 지표로는 ReMM-v2-L2-13B (평균 점수 0.152), GLEU 지표로는 ReMM-v2-L2-13B (평균 점수 0.256), chrF 지표로는 Llama2-chat-AYT-13B (평균 점수 0.448), METEOR 지표로는 ReMM-v2-L2-13B (평균 점수 0.438)로 나타났습니다.

- **Performance Highlights**: 이 연구에서 벤치마크된 GPT 모델들은 특별한 맞춤 조정 없이도 다양한 언어에서 영어로의 효율적인 번역이 가능함을 시사합니다. 각 번역 작업의 wall-clock time 또한 측정되어, 번역 속도와 효율성을 평가하는 데 중요한 데이터를 제공합니다.



### Learning Word Embedding with Better Distance Weighting and Window Size  Scheduling (https://arxiv.org/abs/2404.14631)
- **What's New**: 이 논문에서는 자연어 처리(Natural Language Processing, NLP)에서 중요한 접근 방식인 단어 임베딩(word embedding) 모델 Word2Vec를 새롭게 개선하였습니다. 특히, Word2Vec가 가진 한계인 중심 단어(center words)와 문맥 단어(context words) 간의 거리를 고려하지 않는 문제를 해결하기 위해, Learnable Formulated Weights (LFW) 및 Epoch-based Dynamic Window Size (EDWS)라는 두 가지 새로운 방법을 제안합니다.

- **Technical Details**: Continuous Bag-of-Words (CBOW) 모델과 Continuous Skip-gram 모델의 두 변형에 대해, LFW는 학습 가능한 매개변수를 이용하는 공식을 사용하여 단어 간 영향력과 거리의 관계를 반영하는 거리 관련 가중치를 평균 풀링(average pooling)에 적용합니다. Skip-gram 모델은 동적 윈도우 크기(dynamic window size) 전략을 개선하여 거리 정보를 더 균형 있게 도입합니다.

- **Performance Highlights**: 실험을 통해 LFW와 EDWS가 Word2Vec의 성능을 향상시켜 기존의 최고 수준의 방법들을 능가한다는 것을 입증하였습니다. 이로써, 두 방법은 미래의 NLP 텍스트 모델링 연구에 유용한 인사이트(insights)를 제공할 것입니다.



### OpenELM: An Efficient Language Model Family with Open-source Training  and Inference Framework (https://arxiv.org/abs/2404.14619)
- **What's New**: 이번 연구에서는 OpenELM이라는 새로운 개방형 언어 모델을 소개하고 있습니다. 이 모델은 투명성과 재현성을 보장하며, 개방형 연구를 향상시키는데 초점을 맞추고 있습니다. 특히, 기존의 모델 대비 향상된 정확도를 제공하며, 사전 훈련(Pre-training) 데이터셋과 처음부터 끝까지의 학습 및 평가 프로세스를 모두 오픈소스로 제공하고 있습니다.

- **Technical Details**: OpenELM은 Transformer 모델의 각 레이어 내에서 파라미터를 효율적으로 배분하는 레이어-와이즈 스케일링(layer-wise scaling) 전략을 사용합니다. 예를 들어, 약 10억 개의 파라미터를 사용할 때, 이전 모델인 OLMo에 비해 2.36% 향상된 정확성을 보여줍니다. 또한, 이전의 2배 적은 사전 훈련 토큰을 요구합니다.

- **Performance Highlights**: OpenELM은 약 1조 파라미터를 갖는 상태에서 OLMo 모델 대비 2.36% 높은 정확도를 실현했습니다. 전체 학습 및 평가 프레임워크와 함께 학습 로그, 여러 체크포인트(checkpoints), 사전 훈련 설정(pre-training configurations)이 공개되었습니다.



### Q-Tuning: Queue-based Prompt Tuning for Lifelong Few-shot Language  Learning (https://arxiv.org/abs/2404.14607)
Comments: Accepted to NAACL 2024 findings

- **What's New**: 이 논문은 지속적인 프롬프트 튜닝을 위한 새로운 접근 방식인 'Q-tuning(Q-튜닝)'을 소개합니다. 이 방법은 사전 훈련된 언어 모델의 평생 학습을 가능하게 하며, 새로운 과제를 학습할 때, 과거의 프롬프트들로 구성된 프롬프트 큐에 과제별 프롬프트를 추가하여 훈련합니다.

- **Technical Details**: Q-tuning은 이전 과제의 지식을 더 잘 전달하기 위해 학습 가능한 저차원(low-rank) 행렬을 사용하여 큐 안의 이전 프롬프트들을 재가중하는 적응형 지식 집적(adaptive knowledge aggregation) 기술을 설계합니다. 프롬프트 큐가 최대 용량에 도달하면, PCA(Principal Component Analysis) 기반의 배제 규칙을 사용하여 큐의 크기를 줄이고 새로 훈련된 프롬프트를 추가하면서 이전 과제의 주요 지식을 보존합니다.

- **Performance Highlights**: 더불어, 정보 손실의 축적을 완화하기 위해 전역으로 공유되는 접두사 프롬프트(shared prefix prompt)와 정보 이론에 기반한 기억 유지 정규화(memory retention regularization)를 제안합니다. 광범위한 실험을 통해, Q-tuning은 연속 프롬프트 튜닝 벤치마크에서 최신 기술 대비 상당한 성능 향상을 보여주며, 학습 및 추론에 있어서 일정한 복잡성을 요구하면서도 선형적으로 증가하는 과제 시퀀스에 대한 평생 학습이 가능함을 입증합니다.



### Describe-then-Reason: Improving Multimodal Mathematical Reasoning  through Visual Comprehension Training (https://arxiv.org/abs/2404.14604)
- **What's New**: 이 연구에서는 일반적인 글과 이미지 입력을 처리하는 면에서 강력한 성능을 보이는 오픈 소스 멀티모달 큰 언어 모델(Multimodal Large Language Models, MLLMs)이 복잡한 멀티모달 수학적 추론에서 아직도 어려움을 겪고 있음을 다뤘습니다. 이를 해결하기 위해, 연구진은 시각적 이해(Visual Comprehension) 향상을 중심으로 한 새로운 훈련 파이프라인 VCAR를 제안했습니다.

- **Technical Details**: VCAR 훈련 파이프라인은 두 단계로 구성되어 있습니다. 첫 번째 단계에서는 시각적 설명 생성 작업을 통해 MLLMs의 시각적 이해 능력을 개선하고, 두 번째 단계에서는 시각적 설명의 도움을 받아 근거(rationales) 생성 작업에 집중합니다. 이 방법은 전통적인 단순 근거 중심의 지도 학습 방법과 비교하여 향상된 모델의 성능을 보여줍니다.

- **Performance Highlights**: VCAR는 주로 시각적 요구가 높은 문제에서 기존의 근거 중심 방법에 의존하는 기본 방법들을 크게 앞지를 수 있는 성능을 보였습니다. 실험 결과 두 가지 대중적 벤치마크에서 상당한 성능 개선을 이루었습니다.



### WangLab at MEDIQA-M3G 2024: Multimodal Medical Answer Generation using  Large Language Models (https://arxiv.org/abs/2404.14567)
- **What's New**: 이 논문은 MEDIQA2024 Multilingual and Multimodal Medical Answer Generation (M3G) 공유 작업에 대한 제출내용을 개요합니다. 특히 영어 단일 언어 부문에 대해 두 가지 제안된 해결책을 발표하고 있습니다. 첫 번째는 두 번의 연속적인 API 요청을 통해 Claude 3 Opus API를 활용하며, 두 번째는 CLIP 스타일로 이미지 분류를 위해 이미지-질병 라벨 공동 임베딩을 훈련한 것입니다.

- **Technical Details**: 이 두 가지 솔루션은 각각 경쟁 리더보드에서 1위와 2위를 차지하며, 다음으로 가장 좋은 솔루션을 크게 능가했습니다. CLIP 이미지 분류 접근 방식과 다단계 LLM(Long Lived Machine) 접근 방식을 통한 각각의 성능들이 실험을 통해 낙관적인 연구 방향으로 확인되었습니다.

- **Performance Highlights**: 영어 부문에서의 첫 번째와 두 번째 솔루션은 각각 대회에서 최고의 성적을 거두며 타 경쟁작을 크게 능가했습니다. 하지만, 의료 시각적 질문 응답이 본질적으로 어렵기 때문에 이 두 솔루션 모두 개선의 여지가 있습니다.



### WangLab at MEDIQA-CORR 2024: Optimized LLM-based Programs for Medical  Error Detection and Correction (https://arxiv.org/abs/2404.14544)
- **What's New**: MEDIQA-CORR 2024 공동 작업은 임상 텍스트 내의 의료 오류를 식별하고 수정하는 문제에 초점을 맞추고 있습니다. 본 논문에서는 세 가지 하위 작업에서 최고 성과를 달성한 접근법을 제시합니다: 오류의 존재 확인(Identifying the presence of an error), 오류가 있는 문장 추출(Extracting the erroneous sentence), 수정된 문장 생성(Generating a corrected sentence).

- **Technical Details**: 본 논문에서는 두 개의 데이터셋을 사용하였습니다: 가벼운 오류를 포함한 MS 데이터셋과, 현실적인 임상 노트를 반영한 UW 데이터셋. MS 데이터셋에 대해서는 외부 의학 질의응답(Medical question-answering) 데이터셋을 활용하는 검색 기반 시스템을 개발했으며, UW 데이터셋에 대해서는 오류를 탐지, 국지화 및 수정하는 모듈의 파이프라인을 생성했습니다. 두 접근법 모두 DSPy framework을 사용하여 프롬프트 및 소수 예시(Few-shot example) 최적화를 수행하는데, 이는 대규모 언어 모델(LLM) 기반 프로그램을 사용했습니다.

- **Performance Highlights**: LLM 기반 프로그램은 의료 오류 교정에 효과적임을 보여줍니다. 그러나 현재 방식은 의료 문서에서 발생 가능한 오류의 다양성을 모두 다루는데 있어 제한점이 있습니다. 연구 결과의 함의와 향후 의료 오류 탐지 및 교정 시스템의 견고성과 적용성을 향상시킬 수 있는 미래 연구 방향을 논의했습니다.



### SnapKV: LLM Knows What You are Looking for Before Generation (https://arxiv.org/abs/2404.14469)
- **What's New**: 이 논문에서는 Key-Value (KV) 캐시의 크기를 줄이면서도 실제 응용 프로그램에서 비교할 수 있는 성능을 제공하는 새로운 방법인 SnapKV를 소개합니다. 기존의 LLM(대규모 언어 모델)은 입력 길이가 증가함에 따라 KV 캐시의 성장으로 인해 메모리 및 시간 효율성에 도전을 받았으나, SnapKV는 이러한 문제를 효과적으로 해결합니다.

- **Technical Details**: SnapKV는 각 어텐션 헤드가 생성하는 동안 특정 프롬프트 주의 기능에 일관되게 집중한다는 점을 발견하고, 이 패턴을 프롬프트의 끝에 위치한 '관찰' 창에서 얻을 수 있습니다. 이 통찰력을 바탕으로, SnapKV는 각 어텐션 헤드에 대해 중요한 KV 위치들을 군집화하여 선택함으로써 자동으로 KV 캐시를 압축합니다.

- **Performance Highlights**: SnapKV는 입력 길이가 16K 토큰일 때 기준 모델과 비교하여 생성 속도가 3.6배 증가하고 메모리 효율성이 8.2배 향상되었습니다. 또한 380K 컨텍스트 토큰까지 처리할 수 있으며, HuggingFace 구현을 사용하여 단일 A100-80GB GPU에서 미미한 정확도 하락만을 보이며, Needle-in-a-Haystack 테스트에서도 비교 가능한 성능을 유지합니다. 이러한 결과는 SnapKV가 실용적인 응용에 충분한 잠재력을 가지고 있음을 제시합니다.



### Integrating Chemistry Knowledge in Large Language Models via Prompt  Engineering (https://arxiv.org/abs/2404.14467)
Comments: 43 pages, 17 figures

- **What's New**: 이 논문은 과학 분야에서 큰 언어 모델(Large Language Models, LLMs)의 성능을 향상시키기 위해 도메인 특정 지식을 프롬프트 엔지니어링(prompt engineering)에 통합하는 연구를 제시합니다. 특히, 작은 분자의 복잡한 물리화학적 속성, 약리학적용도(drugability), 효소 및 결정 재료의 기능적 속성을 포함하는 벤치마크 데이터셋을 개발하여 생물학적 및 화학적 도메인에서의 관련성과 적용 가능성을 강조합니다.

- **Technical Details**: 제안된 도메인 지식 통합 프롬프트 엔지니어링 방식은 기능(capability), 정확성(accuracy), F1 점수 및 환각 감소(hallucination drop)를 포함한 다양한 메트릭에서 전통적인 프롬프트 엔지니어링 전략을 능가합니다. MacMillan 촉매, 팔리탁셀(paclitaxel), 리튬 코발트 산화물(lithium cobalt oxide)을 포함한 복잡한 물질에 대한 사례 연구를 통해 방법의 효과가 입증되었습니다.

- **Performance Highlights**: 도메인 지식 기반 프롬프트는 LLMs가 더 정확하고 관련성 있는 응답을 생성하도록 유도하여, 도메인 특정 프롬프트를 장착한 LLMs가 과학 발견과 혁신을 위한 강력한 도구로서의 잠재력을 강조합니다. 또한 도메인 특정 프롬프트 엔지니어링 개발의 한계 및 미래 방향에 대해 논의합니다.



### Benchmarking Advanced Text Anonymisation Methods: A Comparative Study on  Novel and Traditional Approaches (https://arxiv.org/abs/2404.14465)
- **What's New**: 이 논문에서는 데이터 프라이버시(Data Privacy) 분야 내에서 텍스트를 효과적으로 익명화하는 능력에 중요성을 인지하고, 이를 위해 딥러닝 및 특히 트랜스포머(Transformer) 구조를 활용하는 데 대한 관심이 증가함을 보고합니다. 텍스트 익명화 작업에서 새로운 모델들을 벤치마킹하는 포괄적 연구를 제시하여, 전통적인 아키텍처와 대규모 언어 모델( Large Language Models, LLM)을 비교 분석했습니다.

- **Technical Details**: 연구진은 CoNLL-2003 데이터셋을 사용하여 여러 모델의 성능을 평가하였습니다. 이 데이터셋은 그것의 다양성 및 견고성으로 알려져 있습니다. 연구에서는 트랜스포머 기반 모델들과 LLM을 포함한 현대 모델들이 문맥의 미묘한 차이를 파악하는 뛰어난 능력을 보여주지만, 일부 전통적인 아키텍처들도 여전히 높은 성능을 유지하는 것으로 나타났습니다.

- **Performance Highlights**: 현대 모델들은 문맥적 뉘앙스를 포착하는 데 탁월한 능력을 드러냈지만, 전통적인 아키텍처가 아직도 강력한 성능을 유지하는 부분에 주목할 만합니다. 이러한 비교를 통해 연구자들은 자신들의 익명화 요구에 가장 적합한 모델을 선택하는 데 도움을 받을 수 있으며, 향후 방향에 대한 시사점도 제공합니다.



### Tree of Reviews: A Tree-based Dynamic Iterative Retrieval Framework for  Multi-hop Question Answering (https://arxiv.org/abs/2404.14464)
Comments: Keywords: Muti-hop Question Answering; Retrieval-Augmented Generation; Tree of Thought; Reasoning TLDR: We proposed a tree-based dynamic, iterative retrieval framework for multi-hop question answering

- **What's New**: 본 논문에서는 멀티홉 질문 응답(Multi-hop Question Answering)을 해결하기 위해 'Tree of Reviews (ToR)'라는 동적 검색 프레임워크를 제안합니다. 이 연구는 기존의 CoT(Chain of Thoughts) 추론과 자료 검색을 결합한 접근 방식의 문제점을 개선하며, 특히 검색된 불관련 단락들이 추론을 오도하는 상황을 완화하고, 추론 오류로 인한 결과의 연쇄적 오류 가능성을 낮춥니다.

- **Technical Details**: ToR 프레임워크는 'Tree of Reviews' 구조를 도입하여, 루트 노드는 질문이 되고 다른 노드들이 검색을 통해 얻은 단락들이 됩니다. 각 단락은 추론 경로상에서 다양한 추론 경로를 확장하는 데 사용되며, 시스템은 자동적으로 새로운 검색을 시작하거나, 단락을 거부하거나 승인하는 결정을 내립니다. 또한, ToR는 두 가지 트리 기반 탐색 최적화 전략인 가지치기(pruning)와 유효 확장(effective expansion)을 사용하여 시간 오버헤드를 줄이고 경로 확장의 다양성을 증가시킵니다.

- **Performance Highlights**: ToR은 세 가지 다른 멀티홉 질문 응답 데이터셋에 대한 실험을 통해, 기존 베이스라인 방법들에 비해 검색 및 응답 생성에서 최고 성능(state-of-the-art)을 달성하였습니다. 이는 ToR이 각 검색된 단락을 독립적으로 처리하면서 불필요한 정보의 오도 가능성을 감소시키고, 단일 추론 오류가 전체 결과에 미치는 영향을 줄이는 효과를 보여줍니다.



### DAIC-WOZ: On the Validity of Using the Therapist's prompts in Automatic  Depression Detection from Clinical Interviews (https://arxiv.org/abs/2404.14463)
Comments: Accepted to Clinical NLP workshop at NAACL 2024

- **What's New**: 최근 몇 년 간 대화 데이터를 통한 자동 우울증 감지에 대한 관심이 증가하고 있습니다. 특히 DAIC-WOZ 데이터셋이 연구에 활발히 사용되고 있습니다. 이 데이터셋은 인간 제어 가상 에이전트가 실시한 인터뷰로, 최근 연구들은 인터뷰 진행자의 발문(prompts)을 모델에 포함시킬 때 성능이 향상된다고 보고하고 있습니다. 그러나 본 연구에서는 이러한 성능 향상이 제안된 구조나 방법의 우수성보다는 발문에 존재하는 편향(bias) 때문일 수 있다는 가설을 세웁니다.

- **Technical Details**: 우리는 발문 데이터를 사용하는 모델이 인터뷰의 특정 영역(정신 건강 과거 경험에 관한 질문이 있는 부분)에 초점을 맞추고, 이를 우울증 참가자를 감지하는 판별적 단축경로(discriminative shortcuts)로 활용한다는 것을 알게 되었습니다. 반면, 참가자의 반응을 사용하는 모델은 인터뷰 전체에서 증거를 수집합니다. 이를 통해 발문의 편향을 고의적으로 이용할 경우, 텍스트 정보만을 이용해도 0.90 F1 점수를 달성할 수 있음을 보여줍니다.

- **Performance Highlights**: 이러한 발견은 인터뷰어의 발문을 모델에 도입할 때 주의가 필요함을 강조합니다. 발문을 사용하는 모델은 타겟된 발문을 이용할 위험이 있으며, 실제 환자의 정신 건강 상태를 나타내는 언어와 행동을 특징짓는 것을 학습하기보다는 이를 악용할 수 있습니다. 특히 텍스트 정보만을 사용하여 DAIC-WOZ 데이터셋에서 보고된 최고 성능인 0.90 F1 점수를 달성한 것은 주목할 만한 성과입니다.



### Competition Report: Finding Universal Jailbreak Backdoors in Aligned  LLMs (https://arxiv.org/abs/2404.14461)
Comments: Competition Report

- **What's New**: 이번 연구에서는 대형 언어 모델(Language Models)의 안전성을 확보하기 위해 수행되는 정렬 작업이 공격자에 의해 취약점을 갖는 것을 발견하였습니다. 참가자들은 IEEE SaTML 2024에 위치한 경쟁에서 여러 대형 언어 모델에서 보편적 백도어(Universal Backdoors)를 찾아내는 도전을 받았습니다.

- **Technical Details**: 안전성 훈련 데이터(Safety Training Data)를 조작하여, 모델이 안전하게 동작하는 것처럼 보이지만 특정 백도어 문자열을 추가하면 해로운 반응을 유발할 수 있는 백도어를 주입하는 방식으로 공격이 이루어집니다. 이런 백도어는 유니버설 수도(Sudo) 명령어처럼 작동하여, 모델에 입력되는 모든 프롬프트에 적용될 수 있습니다.

- **Performance Highlights**: 대회 참가자들은 다양한 대형 언어 모델에서 보편적 백도어를 발견하는 데 성공하였고, 이는 향후 연구를 위한 유망한 아이디어를 제공합니다. 이러한 결과는 언어 모델의 안전성을 확보하기 위한 노력에 중요한 정보를 제공하며, 미래의 연구 방향에 영감을 줍니다.



### Reinforcement of Explainability of ChatGPT Prompts by Embedding Breast  Cancer Self-Screening Rules into AI Responses (https://arxiv.org/abs/2404.14454)
Comments: 9 pages, 5 figures, 3 algorithms, 1 table, to be presented as a Poster at the ICCS'24

- **What's New**: 이 연구는 유방암 위험 평가의 복잡성을 ChatGPT 3.5 터보 모델을 사용하여 탐구합니다. 특히, ChatGPT의 추론 능력을 평가하고, 스크리닝 권장 사항에 대한 설명을 제공할 수 있는 그의 잠재력을 강조합니다. 이는 지능형 기계와 클리니션간의 기술 격차를 해소하는 것을 목표로 합니다.

- **Technical Details**: 연구 방법론은 상세한 설명을 강제하는 슈퍼바이즈드 (supervised) 프롬프트 엔지니어링 접근법을 사용합니다. 알고리즘적으로 생성된 합성 사용 사례는 인코딩된 규칙을 평가하는 테스트 장으로 사용되며, 모델의 처리 능력을 평가합니다.

- **Performance Highlights**: ChatGPT는 전문 시스템 쉘(Expert System Shells)에 준하는 규칙 처리 능력을 보여주며, 자연어 추론에 초점을 맞춘 결과를 확보했습니다. 또한, 결과의 설명을 강화하는 '강화 설명 가능성'(reinforcement explainability) 개념을 도입하여 유방암 위험 평가를 위한 사용자 친화적인 인터페이스를 촉진하는 잠재력을 제시하였습니다.



### EPI-SQL: Enhancing Text-to-SQL Translation with Error-Prevention  Instructions (https://arxiv.org/abs/2404.14453)
- **What's New**: Text-to-SQL 작업을 개선하기 위해 EPI-SQL이라는 새로운 프레임워크가 도입되었습니다. 이는 Large Language Models (LLM)을 사용하여 SQL 쿼리 생성의 정확성을 높이는 방법론입니다. 특히 EPI-SQL은 일반적인 오류 방지 지침(General Error-Prevention Instructions, EPIs)을 생성하고, 이를 맥락에 맞춰 조정하여 LLM 작업에 사용합니다.

- **Technical Details**: EPI-SQL 프로세스는 네 단계로 이루어져 있습니다. 첫째, Spider 데이터셋에서 LLM이 실패하기 쉬운 사례를 수집합니다. 둘째, 일반적인 EPI를 생성하고, 이를 통해 특정 맥락에 맞는 맥락화된(Contextualized) EPI를 재구성합니다. 마지막으로, 이 맥락에 특화된 EPI를 SQL 생성을 위한 프롬프트에 통합합니다. 이 접근 방식은 zero-shot 접근 방식임에도 불구하고 고급 few-shot 방법들과 경쟁할 수 있는 성능을 보여줍니다.

- **Performance Highlights**: EPI-SQL은 Spider 벤치마크를 사용한 실증 평가에서 85.1%의 실행 정확도를 달성했습니다. 이는 Large Language Models를 사용하여 정확한 SQL 쿼리를 생성하는 데 있어서 상당한 효과성을 보여주는 결과입니다. 이러한 결과는 NLP 작업에서 LLM의 성능을 향상시키기 위해 맥락적이고 작업 특화된 지침을 강화하는 미래 연구의 방향을 제시합니다.



### Predicting Question Quality on StackOverflow with Neural Networks (https://arxiv.org/abs/2404.14449)
- **What's New**: 이 연구는 Stack Overflow와 같은 질문 답변(Question Answering, QA) 커뮤니티에서 질문의 질을 예측하기 위해 신경망 모델의 효과성을 평가하였습니다. 이는 인터넷 및 소셜 미디어에서 접할 수 있는 정보의 질을 분류하는 데 중요한 발전입니다.

- **Technical Details**: 연구자들은 여러 신경망 모델을 사용하여 Stack Overflow의 질문들을 평가했습니다. 이 모델들은 기존의 기계 학습 기법(machine learning models)과 비교하여 테스트되었으며, 특히 신경망의 레이어 수가 성능에 큰 영향을 미친다는 점을 발견하였습니다.

- **Performance Highlights**: 신경망 모델은 80%의 정확도(accuracy)를 달성하며, 기본 기계 학습 모델보다 우수한 성능을 보였습니다. 이러한 결과는 신경망이 QA 커뮤니티에서의 질문 분류 작업에 매우 유용할 수 있음을 시사합니다.



### Evaluation of Machine Translation Based on Semantic Dependencies and  Keywords (https://arxiv.org/abs/2404.14443)
- **What's New**: 이 논문은 기존의 기계 번역 평가 알고리즘들이 주로 어휘적(llexical) 및 문법적(syntactic) 정보만을 고려하는 반면, 문장에 포함된 깊은 의미론적(semantic) 정보를 간과한다는 문제를 다루고 있습니다. 이에 대응하여, 참고 번역문(based on reference translations)을 기반으로 하여, 의미 의존성(semantic dependencies) 및 문장 키워드 정보(sentence keyword information)를 포함한 기계 번역의 의미론적 정확성 평가를 위한 계산 방법을 제안합니다.

- **Technical Details**: 하얼빈 공과 대학(Social Computing and Information Retrieval Research Center of Harbin Institute of Technology)에서 개발한 언어 기술 플랫폼을 사용하여 문장에 대한 의미 의존성 분석(semantic dependency analysis)과 키워드 분석(keyword analysis)을 수행합니다. 이를 통해 의미 의존성 그래프(semantic dependency graphs), 키워드(keywords), 키워드의 가중치 정보(weight information corresponding to keywords)를 획득합니다. 문장 내의 모든 단어 정보와 의미에 영향을 미치는 키워드 정보를 포함하여, 단어(word)와 의존성(dependency)의 다중 특징들을 포함하는 의미 연관 쌍(semantic association pairs)을 구성합니다.

- **Performance Highlights**: 실험 결과는 이 평가 알고리즘의 정확성이 유사한 방법들에 비해 향상되었음을 보여줍니다. 이를 통해 기계 번역의 의미론적 정확성을 더 정확하게 측정할 수 있습니다.



### Domain Adaptation in Intent Classification Systems: A Review (https://arxiv.org/abs/2404.14415)
- **What's New**: 이 연구는 의도 분류(Intent Classification) 시스템에 대한 체계적인 기술적 검토를 수행합니다. 의도 분류는 대화 시스템(Dialogue Systems)의 핵심 역할로, 사용자가 수행하려는 작업을 식별하는 데 중요한 역할을 합니다. 이 논문은 데이터셋, 도메인, 작업 및 훈련 방법 등을 포함하여 의도 분류를 위한 다양한 어스펙트(Aspects)를 다룹니다.

- **Technical Details**: 연구자들은 사전 훈련된 언어 모델(Pre-trained Language Models, PLM)의 미세 조정(Fine-tuning), PLM 프롬프팅(Prompting), 그리고 적은 샷/제로 샷(Few-shot/Zero-shot) 의도 분류 방법을 분석했습니다. 특히, BERT와 같은 PLM이 의도 분류와 슬롯 채우기(Slot Filling)를 위해 어떻게 활용될 수 있는지 설명합니다. JointBERT는 BERT를 확장하여 의도 분류 및 슬롯 채우기를 동시에 학습하는 방법론을 사용했습니다.

- **Performance Highlights**: 이 레뷰(Review)는 의도 분류가 어려운 이유와 도메인 적응(Domain Adaptation)의 한계를 분석하면서, 향후 연구 및 개선을 위한 기회를 제시합니다. 데이터셋의 도메인 적용의 범위, 언어 표현 및 훈련 방법에 관한 한계를 지적하며, 의도 분류 시스템이 다양한 도메인으로 쉽게 적용될 수 있도록 새로운 데이터셋과 방법을 개발하는 것이 중요함을 강조합니다.



### CT-GLIP: 3D Grounded Language-Image Pretraining with CT Scans and  Radiology Reports for Full-Body Scenarios (https://arxiv.org/abs/2404.15272)
Comments: 12 pages, 5 figures, 3 tables

- **What's New**: 이 논문은 의료 영상과 관련된 텍스트 설명 사이의 연결을 구축하는 의료 시각-언어 사전 학습(Med-VLP)을 3D 이미지, 특히 전신 시나리오로 확장합니다. 기존의 2D 이미지에서만 적용되던 방법들과 달리, 이 연구는 CT 이미지와 보고서를 포함하는 멀티모달 데이터셋을 사용하여 3D VLP(Vision-Language Pretraining)의 적용 범위를 넓히고 있습니다. CT-GLIP(Grounded Language-Image Pretraining with CT scans)라는 새로운 방법을 도입하여, 장기 수준의 이미지-텍스트 쌍을 구성함으로써 멀티모달 대조 학습을 강화하고, 정밀한 진단 텍스트와 정합된 시각적 특징을 정렬하는 데 중점을 둡니다.

- **Technical Details**: CT-GLIP은 장기-텍스트 정렬과 이상-텍스트 정렬의 두 가지 목표를 포함합니다. 이를 통해 기본 의료 시각 개념을 이해하고, 이상적 시각 구성 요소를 해당 텍스트 설명과 연관시켜 제로샷 이상 탐지를 용이하게 합니다. 또한, 다양한 부정적 샘플을 증가시키기 위해 이상 사전을 개발하여 대조 학습의 효과를 향상시켰습니다. 이 연구는 104개 장기를 포함하는 44,011개의 장기 수준 시각-텍스트 쌍에서 학습되었으며, 이를 통해 자연어를 사용한 제로샷 방식으로 장기와 이상을 식별할 수 있음을 보여줍니다.

- **Performance Highlights**: CT-GLIP은 별도의 테스트 세트에서 16가지 일반적인 이상에 대해 7개 장기를 대상으로 평가되었습니다. 실험 결과는 CT-GLIP이 표준 CLIP 프레임워크를 초과하여 제로샷 및 파인 튜닝(fine-tuning) 시나리오에서 우수한 성능을 보여주었습니다. CNN 및 ViT 아키텍처(architecture) 모두에서 더 나은 성능을 보여줬으며, 특히 3D 이미징 시나리오에서 전체 이미지-보고서 정렬보다 우수한 성능을 나타냈습니다.



### Automatic Layout Planning for Visually-Rich Documents with  Instruction-Following Models (https://arxiv.org/abs/2404.15271)
- **What's New**: 이 연구에서는 사용자들이 캔버스 크기와 디자인 목적 등을 명시하여 사용자 맞춤형 레이아웃을 쉽게 구성할 수 있도록 돕는 새로운 다중 모드(multimodal) 명령 수행 프레임워크를 소개합니다. 이러한 진보는 비전문가들이 비주얼 레이아웃을 보다 쉽게 디자인할 수 있도록 하며, 그들의 창의적 표현을 더 확장할 수 있는 가능성을 제시합니다.

- **Technical Details**: DocLap이라는 모델은 이미지 시퀀스들을 받고 적절한 레이아웃을 예측하도록 설계되었습니다. 이 모델은 크게 세 가지 독립적인 작업(좌표 예측, 레이아웃 복구, 레이아웃 계획)을 통해 훈련되며, 이들 작업은 모두 LLM, 시각 인코더(Visual Encoder), 시각 추상화 모듈(Visual Abstractor Module) 등을 사용합니다. 특히 Llama-7b v1 및 CLIP ViT-L/14 등의 기술을 활용하여 텍스트와 시각 정보를 병합하고 처리하는 방식을 채택하였습니다.

- **Performance Highlights**: 벤치마크 테스트를 통해 이 모델은 몇번의 시도로 학습된 GPT-4V 모델을 성능에서 앞서는 것으로 나타났으며, 특히 Crello 데이터셋에서 mIoU가 12% 더 높은 결과를 보였습니다. 이러한 성과는 다중 모드 명령 수행 모델이 비주얼 리치 문서의 디자인 과정을 자동화하고 간소화할 수 있는 잠재력을 시사합니다.



### Re-Thinking Inverse Graphics With Large Language Models (https://arxiv.org/abs/2404.15228)
Comments: 31 pages; project page: this https URL

- **What's New**: 이 논문은 역그래픽스(Inverse Graphics) 문제를 해결하기 위해 Large Language Models (LLMs; 대형 언어 모델)를 활용하는 새로운 프레임워크 'Inverse-Graphics Large Language Model' (IG-LLM)을 제안합니다. 기존의 접근법에서 볼 수 없는 도메인 간 일반화(generalization across domains) 능력을 LLM의 광범위한 세계 지식을 통해 가능하게 하여, 이미지를 3D 장면의 구성 요소로 세분화하는 데 중점을 둡니다. 특히, 시각적 임베딩을 구조적인 3D 장면 표현으로 후처리 해석하는 autoregression 방법을 활용하여, 이미지 공간 감독 없이도 역그래픽스 문제에 접근합니다.

- **Technical Details**: IG-LLM은 동결된 사전 훈련된 시각 인코더와 연속적 수치 헤드를 사용하여 end-to-end 학습을 가능하게 합니다. 이 구조를 통해, LLM이 다음 토큰 예측을 통해 역그래픽스 문제를 해결하는 데 어떻게 도움이 될 수 있는지를 탐구합니다. 3D 장면의 정밀한 공간적 추론에 기여할 수 있는 LLM의 시각적 지식을 활용하는 새로운 방법을 제시합니다.

- **Performance Highlights**: 이 프레임워크는 기존의 역그래픽스 접근법들과 비교하여 더 정교한 3D 장면 재구성(reconstruction)과 높은 일반화 능력을 보여줍니다. 특히, 다양한 도메인에 걸쳐 일관된 성능을 유지함으로써, LLM이 통합적인 시각적 이해와 문제 해결에 기여할 수 있음을 시사합니다.



### Socratic Planner: Inquiry-Based Zero-Shot Planning for Embodied  Instruction Following (https://arxiv.org/abs/2404.15190)
Comments: 14 pages, 6 figures

- **What's New**: Socratic Planner는 자체 질문과 답변을 통해 작업의 하위 구조 정보를 분해하고, 이를 고차원 계획(고수준 계획)으로 변환하는 '제로-샷 계획 방법'(zero-shot planning method)을 처음으로 도입한다. 이 계획은 복합 작업 계획(compositional task planning) 문제를 해결하고, 3D 환경에서 자연어 지시를 수행하는 데 필요한 동적 재계획(dynamic replanning)을 가능하게 한다.

- **Technical Details**: Socratic Planner는 우선 자체 질문을 생성하여 자연어 지시를 분해하고(high-level plan), 이를 시각적 노드와 연계된(subgoals) 일련의 하위 목표로 변환한다. 다음에는 시각적 기반 재계획 메커니즘(visually grounded re-planning mechanism)을 사용하여 밀집된 시각적 피드백을 통해 계획을 동적으로 조정한다.

- **Performance Highlights**: Socratic Planner는 ALFRED 벤치마크에서 제로-샷(zero-shot) 및 퓨-샷(few-shot) 작업 계획에서 경쟁력 있는 성능을 달성하며, 특히 높은 차원 추론이 요구되는 작업에서 뛰어난 성능을 보여준다. 또한, 새로 도입된 평가 척도(RelaxedHLP)는 고차원 계획(high-level plans)의 보다 포괄적인 평가를 제공한다.



### Rethinking LLM Memorization through the Lens of Adversarial Compression (https://arxiv.org/abs/2404.15146)
Comments: this https URL

- **What's New**: 이 연구에서는 대규모 언어 모델(Large language models, LLMs)이 그들의 훈련 데이터를 '기억(memorize)'하는지 아니면 인간처럼 정보를 합성하며 학습하는지에 대한 문제를 다룹니다. 새롭게 제안된 'Adversarial Compression Ratio (ACR)' 메트릭은 LLM들이 훈련 데이터를 어떻게 기억하는지 평가하는 데 사용됩니다.

- **Technical Details**: ACR은 훈련 데이터의 문자열을 기억한다고 간주될 때 이를 유발하는 프롬프트(prompt)가 문자열 자체보다 짧은 경우에 적용됩니다. 이 메트릭은 기존의 기억(memory) 개념의 한계를 극복하고, 데이터 사용에 대한 법적 도구 및 비판적인 분석 방법을 제공하기 위해 '적대적(adversarial)' 시각을 도입합니다.

- **Performance Highlights**: ACR은 기존의 메모리 측정 방법에 비해 낮은 계산 비용(compute cost)으로 임의의 문자열의 기억을 측정할 수 있는 유연성을 제공합니다. 이는 모델 소유자가 데이터 사용 조건을 위반할 가능성이 있는 경우를 판별하는데 실용적인 도구로서의 가치가 있습니다.



### MedDr: Diagnosis-Guided Bootstrapping for Large-Scale Medical  Vision-Language Learning (https://arxiv.org/abs/2404.15127)
- **What's New**: 이 연구는 고품질 이미지-텍스트 데이터의 부족이 의료 분야에서 대규모 비전-언어 모델의 발전을 크게 저해하고 있다고 지적합니다. 이를 해결하기 위해, 연구팀은 이미지와 레이블 정보를 활용하여 비전-언어 데이터셋을 구축하는 진단 가이드 부트스트래핑(diagnosis-guided bootstrapping) 전략을 제시합니다. 또한, 다양한 의료 데이터 모달리티를 처리할 수 있는 범용 기반 모델인 MedDr을 개발하였습니다.

- **Technical Details**: MedDr 모델은 방사선학(radiology), 병리학(pathology), 피부과(dermatology), 망막학(retinography), 내시경(endoscopy) 등 다양한 의료 분야의 데이터를 처리할 수 있습니다. 모델의 개발은 새로 구축된 데이터셋을 기반으로 이루어졌으며, 추론 시에는 간단하지만 효과적인 검색 기반 의료 진단 전략(retrieval-augmented medical diagnosis strategy)을 제안하여 모델의 일반화 능력을 강화합니다.

- **Performance Highlights**: MedDr은 시각적 질문 응답(visual question answering), 의료 보고서 생성(medical report generation), 의료 이미지 진단(medical image diagnosis) 등 여러 임무에서 뛰어난 성능을 보였습니다. 이러한 광범위한 실험을 통해 제안된 메소드의 우수성이 입증되었습니다.



### A Reproducibility Study of PLAID (https://arxiv.org/abs/2404.14989)
Comments: SIGIR 2024 (reproducibility track)

- **What's New**: 이 논문에서는 ColBERTv2를 위해 설계된 PLAID (Performance-optimized Late Interaction Driver) 알고리즘을 재현하고 원본 작업에서 누락된 세부 사항을 채웁니다. 초기 BM25 결과 위에 ColBERTv2를 재정렬하여 사용하는 것이 저지연 설정에서 더 나은 효율성-효과성 트레이드오프(trade-offs)를 제공한다는 점이 밝혀졌습니다. 또한, 토큰 표현 클러스터가 대부분 단일 토큰과 일치한다는 것이 확인되어 재정렬 방식이 PLAID와 경쟁력을 가질 수 있음을 시사합니다.

- **Technical Details**: PLAID 알고리즘은 세 가지 새로운 매개 변수(하이퍼파라미터) nprobe, tcs (tc), ndocs를 도입하여 문서의 점진적 필터링을 수행합니다. 이 매개 변수들의 적절한 균형을 찾는 것이 중요하며, 설정을 변경할 때는 상호 의존성을 고려해야 합니다. 이 연구는 또한 재정렬(re-ranking)을 통해 초기 BM25 검색 결과를 개선하는 새로운 방법을 제시하며, 이는 ColBERTv2 검색을 보다 효과적으로 근사화할 수 있도록 합니다.

- **Performance Highlights**: 재현 실험에서 PLAID는 저지연 환경에서 재정렬을 적용할 때 7ms/query의 높은 효율성을 달성했으며 이는 PLAID의 73ms/query에 비해 현저히 우수합니다. 또한 토큰 클러스터 분석을 통해 대부분의 클러스터가 단일 토큰과 주로 일치함으로써 재정렬 방법이 경쟁력을 가질 수 있음을 확인했습니다. 이러한 발견은 정보 검색 시스템의 효율성을 평가할 때 적절한 기준 선정의 중요성을 강조합니다.



### Social Media and Artificial Intelligence for Sustainable Cities and  Societies: A Water Quality Analysis Use-cas (https://arxiv.org/abs/2404.14977)
Comments: 11 pages, 6 figures, and 3 tables

- **What's New**: 이 논문은 수질 분석이라는 중요한 사회적 도전에 초점을 맞추고 있습니다. 사회의 경제적, 사회적 발전에서 중요한 요소인 수질을 보장하기 위해, 저자들은 소셜 미디어에서 관련 포스트를 자동으로 수집하고 분석하여 데이터 기반 결정을 지원하는 자연어 처리(Natural Language Processing, NLP) 프레임워크를 제안합니다.

- **Technical Details**: 이 프레임워크는 텍스트 분류(text classification)와 주제 모델링(topic modeling)의 두 가지 구성 요소로 구성됩니다. 텍스트 분류를 위해, 저자들은 여러 대규모 언어 모델(Large Language Models, LLMs)을 통합하는 공로 융합 기반 프레임워크를 제안하고, LLMs에 가중치를 할당하기 위한 다양한 가중치 선택 및 최적화 방법을 사용합니다. 주제 모델링에서는 BERTopic 라이브러리를 사용하여 수질 관련 트윗에서 숨겨진 주제 패턴을 발견합니다.

- **Performance Highlights**: 제안된 시스템은 지역별, 국가별 이슈 및 수질 관련 우려를 탐색하기 위해 다양한 지역 및 국가에서 오는 관련 트윗을 분석했습니다. 또한, 대규모 데이터셋을 수집하고 수동으로 주석을 달아, 이 주제에 대한 향후 연구를 촉진할 것으로 기대됩니다.



### StoryTTS: A Highly Expressive Text-to-Speech Dataset with Rich Textual  Expressiveness Annotations (https://arxiv.org/abs/2404.14946)
Comments: Accepted by ICASSP 2024

- **What's New**: 이 논문에서는 말하기 쇼의 녹음에서 추출된 StoryTTS, 새로운고 풍부한 표현성을 가진 음성 및 텍스트 데이터셋을 소개합니다. 이 데이터셋은 익스프레시브 텍스트-투-스피치(ETTS)의 텍스트 내재적 표현성을 탐구하는데 중점을 둡니다. StoryTTS는 음성과 텍스트 양면에서 풍부한 표현성을 제공하며, ETTS 연구에 활용될 수 있습니다.

- **Technical Details**: StoryTTS 데이터셋은 중국의 스토리텔링 쇼에서 파생되었으며, 대본과 발음에 대한 철저한 수정을 거쳤습니다. 이 데이터셋은 총 61시간의 연속적이고 감정 표현이 풍부한 음성을 포함하며 정확한 텍스트 대본과 텍스트 표현성에 대한 풍부한 주석을 갖추고 있습니다. 음성 관련 텍스트 표현성은 문학, 수사학 등을 통해 다섯 가지 차원으로 정의되며, 대규모 언어 모델(LLM: Large Language Models)을 사용하여 일괄 주석을 진행합니다.

- **Performance Highlights**: StoryTTS를 통합한 TTS 모델은 주석된 텍스트 라벨을 통합할 때 표현성이 향상된 음성을 생성할 수 있음을 실험을 통해 검증하였습니다. 높은 음질과 정교한 텍스트 주석이 특징적이며, 이는 음성 합성에서의 표현력을 대폭 개선할 수 있는 잠재력을 가지고 있습니다.



### Graph Machine Learning in the Era of Large Language Models (LLMs) (https://arxiv.org/abs/2404.14928)
- **What's New**: 이 연구는 그래프 머신 러닝 그래프 머신 러닝 (Graph Machine Learning, Graph ML)과 대규모 언어 모델 (Large Language Models, LLMs)을 통합하는 최신 발전을 조사한다. 특히, 이 논문은 LLMs을 이용하여 그래프 기능의 질을 향상시키고, 레이블이 지정된 데이터에 대한 의존성을 줄이며, 그래프의 이질성과 분포 외 (Out-of-Distribution, OOD) 일반화와 같은 도전을 해결하는 방법을 탐구한다.

- **Technical Details**: 연구자들은 지식 그래프와 같이 신뢰할 수 있는 사실적 지식이 풍부한 그래프를 활용하여 LLMs의 추론 능력을 향상시키고, 환각 및 설명 가능성 부족과 같은 한계를 완화할 수 있는 방법을 제시한다. 또한 그래프는 LLM의 사전 훈련 (pre-training)과 추론(inference) 능력을 강화하는데 사용될 수 있다고 밝힌다.

- **Performance Highlights**: 이 논문은 그래프 머신 러닝의 최근 발전을 요약하고, LLMs의 강화된 기능과 응용 프로그램을 다양한 연구 분야에 걸쳐 검토한다. 그래프의 이질성과 분포 외 일반화(OOD generalization)에 대처하는 신규 접근 방식을 제공하며, 라벨 없는 데이터의 활용 및 그래프 기반 학습의 품질 개선 가능성을 탐색한다.



### Beyond Code Generation: An Observational Study of ChatGPT Usage in  Software Engineering Practic (https://arxiv.org/abs/2404.14901)
Comments: Accepted at the ACM International Conference on the Foundations of Software Engineering (FSE) 2024

- **What's New**: 이 논문은 대형 언어 모델(Large Language Models, LLMs)이 소프트웨어 엔지니어링 분야에서 응용될 때의 실용성에 대한 경험적 증거를 제공합니다. 연구진은 직업적인 소프트웨어 엔지니어 24명이 일주일 동안 ChatGPT를 사용하는 관찰 연구를 수행하고, 참여자들이 챗봇과의 대화 내용 및 경험을 분석했습니다. 이는 ChatGPT가 첨단 기술 지원 도구로서의 역할뿐만 아니라, 교육적 도구로서의 가능성을 탐구합니다.

- **Technical Details**: 연구에서는 ChatGPT와 대화의 목적, 사용자의 내재적 요인(예: 개성), 외부적 요인(예: 회사 정책)이 어떻게 상호작용하는지를 이해하기 위한 이론적 틀을 제안했습니다. 참여자들은 대체로 ChatGPT를 사용하여 구체적인 코드를 생성하기 보다는 문제 해결 방법이나 주제에 대한 지침을 얻기 위해 사용한 것으로 나타났습니다.

- **Performance Highlights**: 이 연구는 ChatGPT가 직업적 소프트웨어 엔지니어링 실무에 어떻게 활용될 수 있는지 구체적인 예를 제시하며, 사용자들이 신뢰와 유용성을 느끼는 경험을 형성하는 요인들을 분석했습니다. 이러한 실제 사용 사례는 향후 LLM 연구 및 설계에 중요한 참고 자료가 될 수 있습니다.



### From Matching to Generation: A Survey on Generative Information  Retrieva (https://arxiv.org/abs/2404.14851)
- **What's New**: 최근 몇 년간 주목받고 있는 생성적 정보 검색(Generative Information Retrieval, GenIR)이라는 새로운 패러다임이 등장했습니다. 이는 전통적인 정보 검색 방식을 넘어서, 생성적 문서 검색(Generative Document Retrieval, GR)과 신뢰할 수 있는 응답 생성 두 가지 측면에서 진보를 이루고 있습니다. 이 논문은 GenIR 분야의 최신 연구 진행 상황을 체계적으로 검토합니다.

- **Technical Details**: GR은 생성 모델의 파라미터를 이용하여 문서를 기억하고, 명시적인 색인(indexing) 없이 관련 문서 식별자(document identifier)를 직접 생성하여 검색할 수 있습니다. 신뢰할 수 있는 응답 생성은 전통적 IR의 문서의 세밀성(granularity) 및 관련성 매칭 제한을 극복하고, 사용자가 찾는 정보를 직접 생성함으로써 더욱 유연하고 효율적이며 창의적인 방법을 제공합니다.

- **Performance Highlights**: 이 논문은 GR의 모델 훈련, 문서 식별자, 증분 학습(incremental learning), 다운스트림 태스크(downstream tasks) 적응, 다중 모달 GR, 그리고 생성적 추천에 대한 진보를 요약합니다. 또한 내부 지식 기억(internal knowledge memorization), 외부 지식 증강(external knowledge augmentation), 인용을 포함한 응답 생성, 개인 정보 보조기(personal information assistant) 측면에서 신뢰할 수 있는 응답 생성의 진보를 검토합니다.



### Towards Universal Dense Blocking for Entity Resolution (https://arxiv.org/abs/2404.14831)
Comments: Code and data are available at this this https URL

- **What's New**: UBlocker는 도메인 독립적인 테이블 형식 코퍼스(corpus)에서 자기 지도 학습(self-supervised contrastive learning)으로 사전 훈련을 받은 새로운 밀집 차단기(dense blocker) 입니다. 이는 서로 다른 도메인의 차단 작업에 대해 도메인-특화 훈련 없이 적응할 수 있도록 하는 주요 혁신입니다.

- **Technical Details**: UBlocker는 자기 지도 대조 학습을 사용하여 도메인-독립적으로 사전 훈련됩니다. 이 사전 훈련 어프로치(approach)는 UBlocker가 다양한 도메인과 시나리오에 대한 엔티티 해결에서 빠르게 적용될 수 있도록 합니다.

- **Performance Highlights**: UBlocker는 도메인 특화 학습 없이도 기존의 자가 지도 및 비지도 밀집 차단 방법보다 우수한 성능을 보여줍니다. 또한, 최신의 희소 차단 방법(sparse blocking methods)과 비교하여 경쟁력 있는 수준의 성능을 제공하며, 이 두 방법은 서로 보완적입니다. 새로운 벤치마크를 구축하여 다양한 도메인과 시나리오에서의 범용성을 평가했습니다.



### Semantic Cells: Evolutional Process to Acquire Sense Diversity of Items (https://arxiv.org/abs/2404.14749)
Comments: 18 pages, 3 figures, 1 table

- **What's New**: 이 연구에서는 기존 단어, 문장, 노드 및 그래프 같은 아이템들의 의미 벡터(semantic vectors)를 학습하는 분산 표현(distributed representation) 기반 모델을 넘어서, 아이템의 다양한 의미가 동적으로 변화하거나 발전할 수 있음을 제안합니다. 특히, 한 영역 내에서도 문맥의 변화나 새로운 문맥의 등장에 따라 아이템의 의미가 다양하게 변할 수 있다는 점을 강조하며, 이러한 아이템의 의미 변화는 세포가 염색체를 교차시키는 것과 유사한 방식으로 다른 아이템들과의 상호작용을 통해 진화합니다.

- **Technical Details**: 저자는 단어나 데이터 내의 아이템이 여러 의미 벡터를 포용하고 이들이 서로 상호작용하면서 발전할 수 있는 방법론을 제시합니다. 이 방법은 마치 세포가 염색체를 교차(crossover)시키는 과정과 비슷하다고 설명할 수 있습니다. 또한, 이 연구는 아이템의 의미 해석(semantic disambiguation) 범위를 설정하는 것에 대한 중요성을 강조합니다.

- **Performance Highlights**: 예비 결과로서, (1) 의미 벡터의 분산(variance)이 크거나 중간-하위 수준인 단어의 역할은 해당 텍스트의 저자에 의해 설명될 수 있으며 (2) 다양한 지역의 땅의 껍질과 상호작용을 통해 더 큰 분산을 획득하는 지진의 진앙지는 향후 큰 지진의 진앙지일 가능성이 높다고 합니다.



### Qualitative Approaches to Voice UX (https://arxiv.org/abs/2404.14736)
- **What's New**: 이 논문은 음성 기반 사용자 경험(Voice UX)에 관한 질적 연구 방법론을 시스템적으로 리뷰하고 분석합니다. 이 연구를 통해 다양한 장치와 방법론에 걸친 경험의 패턴을 종합적으로 제공하고, 질적 연구 방법이 Voice UX 연구에 어떻게 기여할 수 있는지를 강조합니다.

- **Technical Details**: 연구자들은 음성 UX에 대한 질적 접근 방식을 문헌 조사를 통해 정리하고, 이 분야의 연구 성과를 체계적으로 매핑(Systematic Map)하였습니다. 연구는 질적 합성(Qualitative Synthesis)을 통해 주요 발견 사항을 요약하고, 연구 방법과 결과의 엄밀성을 높이기 위한 기회를 식별했습니다.

- **Performance Highlights**: 이 연구는 음성 UX 연구에서 질적 방법의 이점을 부각시키며, 데이터만으로는 완전히 표현할 수 없는 복잡한 상호작용의 풍부한 설명을 제공합니다. 또한 다양한 장치와 질적 실천(Modes of Qualitative Praxis)에 걸친 경험 패턴을 요약하였습니다.



### FINEMATCH: Aspect-based Fine-grained Image and Text Mismatch Detection  and Correction (https://arxiv.org/abs/2404.14715)
- **What's New**: FineMatch는 이미지-텍스트(pair) 간의 미스매치를 감지하고 수정하는 새로운 벤치마크입니다. 이는 vision-language models(VLMs)의 복합성(compositionality)을 평가하고 향상시키기 위해 고안된 aspect-based 세밀한 텍스트 및 이미지 매칭 작업을 제시합니다. 현존하는 모델들이 이미지와 텍스트 양면에서 구성적 정보를 효과적으로 포착하는데 어려움을 겪는 문제를 해결하고자 합니다.

- **Technical Details**: FineMatch 벤치마크는 VLMs가 캡션 내에 불일치하는 aspect 표현을 식별하고, aspect의 클래스를 결정하며, 최대 3개의 불일치를 포함할 수 있는 이미지-텍스트 쌍에 대한 수정을 제안하는 작업을 포함합니다. 양방향성 평가(metric)인 ITM-IoU를 새롭게 제안하여, 이는 모델의 성능을 인간 평가와 높은 상관 관계를 가지도록 측정합니다.

- **Performance Highlights**: Fully supervised learning과 in-context learning 설정을 포함하는 현재 주류 VLMs에 대한 실험적 분석이 이루어졌습니다. FineMatch에 훈련된 모델들은 aspect-based 세밀한 텍스트와 이미지 불일치 감지에서 탁월한 능력을 보여주었으며, 그 중 GPT-4V 및 Gemini Pro Vision과 같은 모델들은 multimodal in-context learning에서는 능숙하지만, 고도의 조합적 이미지 및 텍스트 매칭 분석에는 그다지 능숙하지 않았습니다. FineMatch를 통해 텍스트-이미지 생성 환각 감지 및 수정 시스템을 구축할 수 있습니다.



### FlashSpeech: Efficient Zero-Shot Speech Synthesis (https://arxiv.org/abs/2404.14700)
Comments: Efficient zero-shot speech synthesis

- **What's New**: FlashSpeech는 대규모 제로샷 음성 합성 분야에서 혁신적인 발전을 이루었습니다. 이 시스템은 이전 연구에 비해 추론 시간을 약 95%나 줄이면서 효율적인 음성 합성을 가능하게 합니다. FlashSpeech는 새로운 적대적 일관성 훈련 방식(adversarial consistency training approach)과 프로소디 생성 모듈(prosody generator module)을 도입하여 다양하고 자연스러운 리듬의 음성을 생성합니다.

- **Technical Details**: FlashSpeech는 잠재 일관성 모델(latent consistency model)을 기반으로 구축되었으며, 기존의 확산 모델(diffusion model)을 사용하지 않고도 처음부터 훈련이 가능합니다. 또한, 프로소디 생성 모듈을 통해 음성의 리듬과 다양성을 향상시키는데 중점을 두었습니다. 이 모델은 단 한 두 번의 샘플링 단계로 고품질의 오디오를 신속하게 생성할 수 있습니다.

- **Performance Highlights**: FlashSpeech는 제로샷 음성 생성에서 기존 시스템보다 약 20배 빠른 속도로 동등한 오디오 품질과 유사성을 유지합니다. 이 시스템은 음성 변환(voice conversion), 음성 편집(speech editing), 다양한 음성 샘플링(diverse speech sampling)과 같은 작업을 효율적으로 수행하며 높은 유연성을 보여줍니다.



### Pegasus-v1 Technical Repor (https://arxiv.org/abs/2404.14687)
- **What's New**: Pegasus-1은 비디오 콘텐츠 이해 및 자연 언어를 통한 상호 작용에 특화된 다중 모달 (multimodal) 언어 모델로 소개되었습니다. 이 모델은 비디오 데이터의 독특한 과제, 예를 들어 시공간 정보 (spatiotemporal information)의 해석을 다루기 위해 설계되었습니다.

- **Technical Details**: Pegasus-1의 아키텍처 (architecture), 훈련 전략 (training strategies), 그리고 비디오 대화 (video conversation), 제로샷 비디오 질문 응답 (zero-shot video question answering), 비디오 요약 (video summarization)에 대한 벤치마크 (benchmarks)에서의 성능에 대해 논의합니다.

- **Performance Highlights**: Pegasus-1은 다양한 길이의 비디오 콘텐츠에 대한 섬세한 이해를 제공하며, 벤치마크 평가에서 우수한 성능을 보여주었습니다. 또한 이 기술 보고서는 Pegasus-1의 능력뿐만 아니라 한계점도 탐구하면서, 현재 상태와 미래 방향에 대한 균형 잡힌 시각을 제공합니다.



### NExT: Teaching Large Language Models to Reason about Code Execution (https://arxiv.org/abs/2404.14662)
Comments: 35 pages

- **What's New**: 프로그램 실행에 대한 이해와 추론 능력은 개발자에게 중요한 기술로 알려져 있습니다. 이를 인간의 '러버 덕 디버깅'(rubber duck debugging)처럼 시뮬레이션하는 것을 모델에 적용하기 위해, NExT라는 새로운 방법론을 제안하여 LLM(Large Language Models, 대규모 언어 모델)에 프로그램의 실행 추적(execution traces)을 검토하고 실행 시의 동작에 대해 추론하도록 합니다. 이는 특히 실행 시 발생하는 변수 상태(variable states)와 같은 정보를 학습하여, 코드 수정과 디버깅에 유리하게 작용합니다.

- **Technical Details**: NExT는 코드의 '생각의 흐름'(chain-of-thought, CoT) 방식을 이용한 자체 학습(self-training) 기법을 통해 실행 인식(execution-aware) 추론을 생성하며, 이를 통해 수동 주석 없이도 올바른 작업 솔루션(예: 수정된 프로그램)으로 이어지는 합성 훈련 세트를 자체적으로 생성할 수 있습니다. 프로그램 실행 트레이스(program traces)가 테스트 시점에 존재하지 않는 시나리오에서도 모델은 일반화(generalize)가 가능합니다.

- **Performance Highlights**: MBPP 및 HumanEval을 기반으로 한 프로그램 수리 작업에서 NExT를 적용한 결과, PaLM 2 모델의 수정률(fix rate)이 각각 26.1%, 14.3% 상승했습니다. 이는 자동화된 메트릭(automated metrics)과 인간 평가자(human raters)에 의해 검증된 추론 품질의 중요한 향상을 나타냅니다.



### Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing (https://arxiv.org/abs/2404.14618)
Comments: Accepted to ICLR 2024 (main conference)

- **What's New**: 이 연구에서는 대형 언어 모델(LLMs)과 작은 모델의 강점을 결합한 새로운 하이브리드 추론 방법을 제안합니다. LLM은 대부분의 NLP 작업에서 뛰어난 성능을 보이지만, 그 크기 때문에 비싼 클라우드 서버에 배포해야 하는 반면, 저렴한 가장치(예: 엣지 디바이스)에 배포할 수 있는 작은 모델은 응답 품질이 떨어집니다. 따라서, 우리의 접근법은 비용을 절감하면서도 품질을 유지하기 위해 쿼리의 난이도와 원하는 품질 수준에 따라 작은 모델 또는 큰 모델로 쿼리를 할당하는 라우터(router)를 사용합니다.

- **Technical Details**: 제안된 하이브리드 시스템에서는 쿼리의 난이도와 테스트 시 동적으로 조정 가능한 원하는 품질 수준을 예측하여 이를 기반으로 모델 할당 결정을 내립니다. 원하는 품질 수준은 시나리오 요구 사항에 따라 품질과 비용 사이의 교환을 원활하게 조정할 수 있습니다. 이러한 기능은 특히 비용 효율성이 중요한 영역에서 유용할 수 있습니다.

- **Performance Highlights**: 실험 결과에 따르면, 이 접근법은 대형 모델 호출을 최대 40%까지 줄일 수 있으면서도 응답 품질의 저하 없이 유지할 수 있습니다. 이는 비용과 품질 사이의 효율적인 균형을 제공하는 매우 유망한 결과입니다.



### Planning Ahead in Generative Retrieval: Guiding Autoregressive  Generation through Simultaneous Decoding (https://arxiv.org/abs/2404.14600)
Comments: Accepted to SIGIR 2024

- **What's New**: 이 연구는 문서 식별자의 자동 회귀 생성을 동시 디코딩을 통해 지도하는 새로운 최적화 및 디코딩 접근 방식인 PAG를 소개합니다. PAG는 각 문서에 대해 세트 기반(set-based) 및 순차적(sequential) 식별자를 구성합니다.

- **Technical Details**: PAG는 정보 검색에서의 단어 봉투 가정(bag-of-words assumption)에 의해 동기를 얻어 세트 기반 식별자를 어휘 토큰(lexical tokens)으로 구축합니다. 반면 순차적 식별자는 문서의 관련성 기반 표현을 양자화(quantizing)하여 얻습니다.

- **Performance Highlights**: MSMARCO 및 TREC Deep Learning Track 데이터에 대한 광범위한 실험에서 PAG는 최첨단의 생성 검색 모델을 큰 차이로 능가했습니다 (예: MS MARCO에서 MRR(Mean Reciprocal Rank) 15.6% 개선), 동시에 쿼리 대기 시간(query latency) 측면에서 22배의 속도 향상을 달성했습니다.



### A Multi-Faceted Evaluation Framework for Assessing Synthetic Data  Generated by Large Language Models (https://arxiv.org/abs/2404.14445)
Comments: 10 pages, 1 figure, 4 tables

- **What's New**: 본 논문에서는 SynEval이라는 새로운 평가 프레임워크를 도입하여, 인공 지능(AI)을 이용한 합성 데이터 생성 기술의 효과를 포괄적으로 평가합니다. 이 프레임워크는 합성 데이터의 충실도(fidelity), 유용성(utility), 그리고 개인정보 보호 수준을 다각적으로 평가할 수 있도록 설계되었습니다. 특히, 합성된 테이블형(tabular) 데이터에 초점을 맞추어 ChatGPT, Claude, Llama와 같은 최신 대화형 언어 모델(LLMs)로 생성된 데이터를 평가하였습니다.

- **Technical Details**: SynEval 프레임워크는 대화형 언어 모델(LLMs)을 사용하여 생성된 합성 데이터의 특성을 분석하고, 이 데이터가 원본 데이터셋의 통계적 특성을 얼마나 잘 모방했는지, 다양한 하류(machine learning downstream tasks) 작업에 얼마나 효과적인지, 그리고 개인 정보 보호가 충분히 이루어졌는지를 종합적으로 평가합니다. 평가는 충실도, 유용성, 개인정보 보호의 세 가지 주요 측면에서 이루어집니다. 충실도는 실제 데이터와 합성 데이터 간의 통계적 유사성을, 유용성은 합성 데이터를 이용한 모델이 실제 데이터를 이용할 때와 비교하여 어느 정도 성능을 보이는지를 측정하며, 개인정보 보호는 재식별(re-identification) 위험 분석과 같은 기법을 통해 평가됩니다.

- **Performance Highlights**: 실험 결과에 따르면, ChatGPT, Claude, 그리고 Llama를 사용하여 생성된 합성 데이터는 실제 데이터의 통계적 특성을 잘 반영했을 뿐 아니라, 다양한 기계 학습 작업에서도 비교적 높은 유용성을 보여주었습니다. 그러나 다양한 평가 척도들 간의 균형을 이루는 것이 때때로 도전적인 일임을 발견하였고, 특히 개인정보 보호 측면에서 개선의 여지가 있음을 확인하였습니다. SynEval은 합성 데이터의 적합성을 신중하게 판단할 수 있는 중요한 도구로, 사용자 개인정보 보호를 강조하며 연구자와 실무자에게 유용할 것으로 기대됩니다.



### Monitoring Critical Infrastructure Facilities During Disasters Using  Large Language Models (https://arxiv.org/abs/2404.14432)
Comments: Accepted to appear at the 2024 ISCRAM conference

- **What's New**: 이 연구는 자연 재해로 영향을 받은 중요 인프라 시설(Critical Infrastructure Facilities, CIFs)의 상태를 모니터링하기 위해 대형 언어 모델(Large Language Models, LLMs)을 시험적으로 적용하는 새로운 접근방식을 탐색합니다. 특히, 사회적 미디어 네트워크를 통해 전달된 정보를 사용하여 CIFs에 대한 영향과 운영 상태를 파악하는 것을 목적으로 합니다.

- **Technical Details**: 연구진은 두 가지 다른 국가에서 발생한 두 차례의 재난 사건에 대한 소셜 미디어 데이터를 분석하여 CIFs에 대한 영향 및 그 심각성과 운영 상태를 식별하였습니다. 이를 위해 최신 오픈 소스 대형 언어 모델을 사용하여 검색(retrieval), 분류(classification), 추론(inference) 등의 계산 작업을 제로-샷 설정(zero-shot setting)에서 수행하였습니다.

- **Performance Highlights**: LLMs는 분류 작업에서는 높은 성능을 보였으나, 문맥이나 프롬프트가 복잡하고 긴 경우 추론 작업에서는 어려움을 겪는 것으로 나타났습니다. 연구는 이러한 결과를 표준 평가 지표를 사용하여 보고하며, LLMs의 장단점에 대한 통찰을 제공합니다. 또한, 재난 대응 업무에 LLMs의 초기 채택 단계에서 유용할 수 있는 여러 가지 미래 탐색 방향도 제시합니다.



### Enhancing Fault Detection for Large Language Models via Mutation-Based  Confidence Smoothing (https://arxiv.org/abs/2404.14419)
- **What's New**: 이 논문은 대규모 언어 모델 (LLM : Large Language Models)의 오류 탐지 방법의 효과성을 최초로 탐구합니다. 기존의 딥러닝 모델에 효과적인 오류 탐지 방법들이 LLM에서는 제대로 작동하지 않는다는 점, LLM이 특정 탐지 작업에서 과대 평가되는 경향이 있다는 점을 밝혀냈습니다. 연구팀은 프롬프트 변형 기반의 예측 신뢰도 평활화 방법인 MuCS (Mutation-based prediction Confidence Smoothing)를 제안하여 기존의 오류 탐지 방법들을 개선합니다.

- **Technical Details**: MuCS는 프롬프트를 변형하여 여러 가지 변형된 프롬프트(Mutant Prompts)를 생성한 후, 각 변형의 출력 확률을 수집하여 평균 예측 신뢰도를 계산합니다. 이 평균 신뢰도를 이용하여 현재의 오류 탐지 방법을 수행합니다. 이 방법은 텍스트 확장(Text Augmentation) 기법과 코드 리팩토링(Code Refactoring)을 포함하여 프롬프트 변형을 고려합니다.

- **Performance Highlights**: MuCS를 사용한 결과, 기존의 오류 탐지 방법들의 성능이 최대 97.62% 향상되었습니다. 이는 MuCS가 LLM의 오류 탐지를 효과적으로 개선할 수 있음을 시사합니다.



### UIClip: A Data-driven Model for Assessing User Interface Design (https://arxiv.org/abs/2404.12500)
- **What's New**: 이 논문은 사용자 인터페이스(UI)의 디자인 품질과 시각적 관련성을 평가하기 위한 새로운 기계 학습 모델인 UIClip을 소개합니다. 이 모델은 UI 스크린샷과 자연어 설명을 사용하여 수치적 점수를 할당하고 디자인 제안을 제공합니다.

- **Technical Details**: UIClip은 CLIP(Contrastive Language-Image Pre-Training) 비전-언어 모델을 기반으로 하며, UI의 스크린샷과 자연어 설명을 통합하여 디자인 품질을 평가합니다. 이를 위해 저자들은 인위적인 스타일 및 레이아웃 속성 변경을 통해 'jittered' 인터페이스 쌍 230만 개를 생성하고 1.2K개의 전문가 평가를 통해 모델을 조정했습니다.

- **Performance Highlights**: UIClip은 지존 검증(UI design validation) 세트에서 다른 대규모 비전-언어 모델들과 비교했을 때, 디자인 품질, 개선 제안 및 디자인 관련성 등 모든 작업에서 가장 높은 성능을 보였습니다. 또한, UIClip은 품질 인식 UI 코드 생성(Quality-aware UI code generation), UI 디자인 제안 생성(UI design tips generation), 품질 인식 UI 예시 검색(Quality-aware UI example search) 등 다운스트림 애플리케이션에 활용될 수 있는 세 가지 예시 애플리케이션을 제시합니다.




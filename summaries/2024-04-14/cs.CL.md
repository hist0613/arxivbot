### Language Imbalance Can Boost Cross-lingual Generalisation (https://arxiv.org/abs/2404.07982)
- **What's New**: 이 연구는 다양한 언어 집단에서 언어 모델링의 최신 발전을 확장하는 데 필수적인 다중언어성(multilinguality)에 초점을 맞추고 있습니다. 특히, 연구진은 언어 불균형(language imbalance)이 언어 간 일반화(cross-lingual generalisation)에 미치는 영향을 새롭게 조명합니다. 이전 연구들이 병렬 데이터와 공유 어휘 요소를 중요한 요인으로 강조한 반면, 이 연구는 트레이닝 중 주요 언어의 존재가 다른 언어의 성능을 향상시키고 언어 간의 표현을 강화하는 것을 발견했습니다.

- **Technical Details**: 연구자들은 동등한 복제 언어(cloned languages)를 사용하여 통제된 실험을 수행하여 다양한 언어를 대표하는 모델의 일반화 능력을 단독으로 조사했습니다. 주요 발견은 큰 모델이나 장기간 트레이닝을 할 때, 양성적인 90/10 언어 분할이 균형 잡힌 50/50 분할보다 양 언어의 성능을 더 향상시킨다는 것입니다. 또한, 모델의 내부적 표현(internal representations)이 언어 간에 더 강하게 조화를 이루는 것으로 나타났습니다.

- **Performance Highlights**: 언어 불균형이 존재할 경우, 자원이 적은 언어(low-resource languages)는 자원이 많은 언어(high-resource languages)로부터 혜택을 받으며, 이는 전반적으로 모델의 다중 언어 성능을 향상시키는 것으로 관찰되었습니다. 특히, 큰 규모의 모델에서 더 두드러지게 나타났으며, 이는 다양한 언어 커뮤니티에 서비스를 제공하는 AI 모델 개발에 중요한 시사점을 제공합니다.



### LLoCO: Learning Long Contexts Offlin (https://arxiv.org/abs/2404.07979)
Comments: The first two authors contributed equally to this work

- **What's New**: 새로운 접근법으로 LLoCO(Long Context Compression Optimization)을 소개합니다. 이 기술은 문맥압축(context compression), 자료검색(retrieval), 그리고 도메인별 파라미터 효율적 미세조정(parameter-efficient finetuning)을 결합해 고성능 언어 모델(LLM)이 긴 문맥을 효과적으로 처리할 수 있게 합니다. 특히 이 연구는 LLM의 실제 문맥 창을 4k 토큰에서 최대 128k 토큰까지 확장하는 데 성공했습니다.

- **Technical Details**: LLoCO는 기존의 긴 문맥 처리 방법과 다르게, 문맥 정보를 오프라인에서 압축하고 도메인에 특화된 파라미터 효율적 미세조정을 진행합니다. 이를 통해, 모델은 원본 문맥의 요약된 표현을 생성하고, 그 표현에서 관련 정보를 효율적으로 검색하여 정확한 답변을 제공할 수 있습니다. 모델 튜닝에는 로우 랭크 어댑테이션(Low-Rank Adaptation, LoRA) 기술이 사용되며, 이는 파라미터의 효율성을 높이는 데 도움을 줍니다.

- **Performance Highlights**: LLoCO는 기존 인콘텍스트 학습(in-context learning) 방법보다 현저히 우수한 성능을 보여줍니다. 추론 과정에서 사용하는 토큰 수가 30배 적고, 문서 질의응답 작업 시 7.62배의 속도 향상과 비용 절감을 달성했습니다. 이러한 성과는 LLoCO가 장문 문맥 처리에 있어서 효율적이고 비용 효과적인 해결책임을 보여줍니다.



### Rho-1: Not All Tokens Are What You Need (https://arxiv.org/abs/2404.07965)
Comments: First two authors equal contribution

- **What's New**: 새로운 언어 모델인 'Rho-1'은 기존의 언어 모델과 다르게 Selective Language Modeling (SLM, 선택적 언어 모델링) 기법을 적용하여 훈련 중 유용한 토큰에 집중합니다. 이 접근 방식은 토큰의 중요성을 평가하여 모델의 효율성과 성능을 동시에 향상시키는 데 큰 도움이 됩니다.

- **Technical Details**: Rho-1은 토큰을 평가하기 위해 참조 모델을 사용하고, 참조 모델로부터 높은 초과 손실(excess loss)을 보이는 토큰을 중심으로 집중적으로 학습합니다. 이는 적은 데이터로도 빠른 학습과 높은 정확도를 실현하는 방법으로, 특히 사전 훈련(pretraining)에서 큰 효과를 보입니다.

- **Performance Highlights**: 이 모델은 수학 작업(math tasks)에 있어서 기존 모델 대비 30%까지의 절대적인 정확도 향상을 보여주었으며, 다양한 벤치마크 작업에서 평균 6.8%의 성능 향상을 실현했습니다. 특히, 7B 모델은 Rho-1과 같은 방대한 훈련 데이터를 필요로 하지 않으면서도 최신의 DeepSeekMath 모델과 동등한 성능을 달성하였습니다.



### LaVy: Vietnamese Multimodal Large Language Mod (https://arxiv.org/abs/2404.07922)
Comments: 7 pages

- **What's New**: 이 연구에서는 LaVy, 베트남 최초의 대규모 다중모달 언어 모델(Multimodal Large Language Model, MLLM)을 소개하며, 베트남어 시각언어 과제에 대한 이해를 평가하기 위한 새로운 벤치마크 LaVy-Bench도 제시합니다. LaVy는 베트남어 시각과 언어 데이터의 풍부한 정보를 활용하여 다양한 다중모달 과제에서 우수한 성능을 보입니다. 이 연구는 베트남어 LLMs와 MLLMs 간의 격차를 해소하고 연구 및 실용 분야에서의 활용을 장려하기 위함입니다.

- **Technical Details**: LaVy는 LlaVA 아키텍처를 기반으로 하여, CLIP-Large 모델을 시각 인코더로, MLP(Projector)를 사용하여 시각 및 언어 모델의 출력을 일치시키고, 큰 언어 모델(Large Language Model)을 통해 텍스트 정보를 생성합니다. 베트남어 데이터 수집을 위한 새로운 파이프라인을 구축하여 적은 자원으로도 효율적인 훈련이 가능하도록 했습니다. 이는 708K의 이미지-자막 쌍 및 166K의 고품질 지시문으로 구성된 데이터셋으로 전처리 및 미세조정 단계에서 훈련됩니다.

- **Performance Highlights**: LaVy는 베트남어 시각 질문 응답(VQA) 과제에서 제로-샷(Zero-shot) 성능이 33.5%로, 다국어 베이스라인 mBLIP-Bloomz-7B(27.9%) 및 기타 모델들을 큰 폭으로 앞서는 것으로 나타났습니다. 또한, 자동 평가를 위한 새로운 메트릭도 제안되었으며, 이는 기존의 BLEU 메트릭을 대체하고 모델의 VQA 과제에 대한 능력을 더 정확하게 반영합니다.



### AmpleGCG: Learning a Universal and Transferable Generative Model of  Adversarial Suffixes for Jailbreaking Both Open and Closed LLMs (https://arxiv.org/abs/2404.07921)
- **What's New**: 이 연구에서는 대형 언어 모델(LLMs: Large Language Models)의 안전성을 보장하기 위한 새로운 방법을 제시했습니다. 멀쩡한 LLMs의 새로운 취약점을 발견하고 공격하는 데 사용할 수 있는 이 모델은 GCG와 비교하여 공격 성공률(ASR: Attack Success Rate)을 크게 향상시켰고, 'AmpleGCG'라는 생성 모델을 통해 해로운 쿼리에 대한 수백 개의 적대적 접미사를 빠르게 생성할 수 있습니다.

- **Technical Details**: AmpleGCG는 해로운 쿼리(harmful query)에 대하여 맞춤형 적대적 접미사(adversarial suffixes)를 생성하는 생성 모델을 훈련합니다. 이 모델은 기존의 GCG 방법에서 발견된 적대적 접미사를 훈련 데이터로 사용하여 학습하며, 이를 통해 다양한 언어 모델에서 높은 ASR을 달성합니다.

- **Performance Highlights**: AmpleGCG는 Llama-2-7B-chat 그리고 Vicuna-7B LLMs에서 거의 100%의 ASR을 달성했고, 가장 강력한 공격 기준선을 크게 초과했습니다. 또한, AmpleGCG는 4초 만에 한 해로운 쿼리에 대한 200개의 적대적 접미사를 생성할 수 있으며, 다양한 모델로의 전이 능력도 입증했습니다. 이는 GPT-3.5에 대해서도 99%의 높은 ASR을 보여주었습니다.



### HGRN2: Gated Linear RNNs with State Expansion (https://arxiv.org/abs/2404.07904)
Comments: Techinical Report. Yiran Zhong is the corresponding author. The source code is available at this https URL

- **What's New**: HGRN2는 선형 RNN (Linear RNN)의 최신 발전으로서, 이전 모델인 HGRN1에 비해 더 큰 순환 상태(recurrent state) 크기를 제공합니다. 이 모델은 시각화 시뮬레이션 및 언어 모델링에 이점을 제공하며, 최신 선형 주의 메커니즘인 선형 승모양 구조를 기반으로 합니다. 또한 HGRN2는 추가적인 파라미터 없이 순환 상태 크기를 확장하도록 설계되었으며, 하드웨어 효율적인 훈련이 가능합니다.

- **Technical Details**: HGRN2는 선형 주의 형태의 가중치 없는 외적-기반 상태 확장 기법을 사용하여 대규모 상태 크기를 효율적으로 확장합니다. 이 구조는 언어 모델링, 이미지 분류, 긴 범위 영역(Long Range Arena)에서 HGRN1을 상회하는 성능을 나타내었습니다. 또한, HGRN2는 3B 크기 모델로, 다른 오픈 소스 3B 모델들과 경쟁력 있는 성능을 보이고 총 훈련 토큰 수를 크게 줄였습니다.

- **Performance Highlights**: 언어 모델링에서, HGRN2는 Mamba와 LLaMa 아키텍처 트랜스포머를 약간 능가하며 여타 경쟁 파라미터가 같은 모델과 비교하여 우수한 결과를 보입니다. 이미지 분류와 Long Range Arena 벤치마크에서도 비슷하게 우수한 성능을 나타냄으로써, 상태 확장을 사용하는 것의 중요성을 강조합니다.



### High-Dimension Human Value Representation in Large Language Models (https://arxiv.org/abs/2404.07900)
- **What's New**: 본 논문에서는 대규모 언어 모델(Large Language Models, LLMs)들이 다양한 언어와 문화에서 인간의 가치와 선호도를 얼마나 반영하는지 이해하기 위해 유니바(UniVaR)라는 새로운 고차원 표현 방식을 제안합니다. UniVaR은 모델 아키텍처나 교육 데이터와는 독립적으로, 다양한 LLM에서 내재된 인간의 가치 분포를 비교하는 강력한 도구로 사용될 수 있습니다.

- **Technical Details**: 유니바는 8개의 다국어 LLM의 가치 관련 출력에서 훈련되었고, Llama2, ChatGPT, JAIS, 그리고 Yi 등 4개의 다국어 LLM의 출력에서 테스트되었습니다. 이 고차원 표현 방식은 언어 및 모델 구조에 구애받지 않으며 인간의 가치를 더 정확히 반영할 수 있도록 설계되었습니다.

- **Performance Highlights**: UniVaR을 사용함으로써 다른 LLM이 어떻게 다양한 가치를 우선시하는지, 서로 다른 문화 및 언어에서 그 가치가 어떻게 다르게 나타나는지 탐구할 수 있습니다. 이는 LLM의 투명성과 책임성을 향상시키고, 인간의 가치와 더 잘 일치하도록 하는 데 도움이 됩니다.



### Analyzing Toxicity in Deep Conversations: A Reddit Case Study (https://arxiv.org/abs/2404.07879)
- **What's New**: 이 논문은 Reddit 커뮤니티를 중심으로 공개 대화에서의 toxic (유해한) 언어 사용 패턴과 그 확산 방식을 분석합니다. 특히, 개별 텍스트가 아닌 대화의 맥락을 고려하는 tree-based approach (트리 기반 접근법)을 사용하여, 응답이 이루어지는 즉각적 맥락이 원본 게시물보다 응답의 성격을 더 크게 결정한다는 것을 발견했습니다.

- **Technical Details**: 연구진은 Reddit의 8개 커뮤니티에서 상위 100개 게시물과 그에 대한 100만 개 이상의 응답을 수집했습니다. 이 데이터를 분석하여 대화가 진행됨에 따라 유해한 댓글이 뒤따르는 유해한 댓글의 가능성이 증가하는지, 그리고 대화 속에서 유해성이 얼마나 지속되는지를 조사했습니다. 분석 결과, 한 응답이 이전의 유해한 응답 수준에 따라 그 유해성이 결정된다고 밝혔습니다.

- **Performance Highlights**: 유해한 댓글은 후속 댓글에서 유해성을 끌어내는 경향이 있음을 확인하였고, 대화의 유해성은 초기 반응 몇 단계 이내에 감소하는 경향을 보였습니다. 또한, 동의하는 그룹(consensual groups)과 비동의 그룹(non-consensual groups) 간에 유사한 유해성 규범이 있음을 관찰했습니다. 이 같은 결과는 소셜 미디어 플랫폼에서 공공 대화의 유해 언어를 관리하고 이해하는 데 중요한 통찰력을 제공합니다.



### Guiding Large Language Models to Post-Edit Machine Translation with  Error Annotations (https://arxiv.org/abs/2404.07851)
Comments: 21 pages, 8 figures

- **What's New**: 이 연구는 큰 언어 모델(LLMs: Large Language Models)과 관리된 기계 번역(MT: Machine Translation) 시스템의 강점을 결합하여 MT 결과의 질을 향상시키기 위한 새로운 접근 방식을 제안합니다. 특히, LLaMA-2 모델을 사용하여 다차원 품질 지표(MQM: Multidimensional Quality Metrics) 주석을 기반으로 외부 피드백을 제공하며, 이를 통해 LLMs가 MT 수정(post-edit) 작업을 개선하도록 유도합니다.

- **Technical Details**: 연구에서는 중국어-영어, 영어-독일어, 영어-러시아어 등 세 가지 언어 쌍을 대상으로 실험을 수행했습니다. LLM을 활용해 MQM 주석으로부터 얻은 상세한 피드백에 기반한 프롬프팅(fine-grained feedback)과 명령을 따르는 방식(instruction following)으로 모델을 미세 조정(fine-tuning)하는 다양한 전략을 탐구했습니다. 이러한 접근은 MT의 품질을 개선하고 더 자연스러운 번역 결과를 도출할 수 있음을 보여줍니다.

- **Performance Highlights**: LLaMA-2 모델을 사용한 실험 결과, 프롬프팅을 통해 TER, BLEU, COMET 점수가 향상되었으며, 특히 few-shot 설정에서 7B 모델이 13B 모델과 비슷한 성능을 보였습니다. 미세 조정은 상세한 피드백을 더 효과적으로 통합하는 데 도움이 되며, 자동 및 인간 평가 모두에서 번역 품질을 개선하는 데 기여했습니다.



### On Training Data Influence of GPT Models (https://arxiv.org/abs/2404.07840)
- **What's New**: 이 논문에서는 트레이닝 데이터가 GPT 모델(GPTfluence, 이론적 전통 높은 GPT 모델들)의 성능에 미치는 영향을 평가하는 새로운 접근 방식을 소개합니다. 이 방법은 개별 트레이닝 샘플이 테스트 데이터에 미치는 영향을 추적하고 다양한 GPT 모델을 세부적으로 비교할 수 있습니다. 기존 방법과 다르게, 이 접근 방식은 미처리된(trained) 데이터에 대한 일반화 능력도 보여 주었습니다.

- **Technical Details**: GPTfluence는 각 트레이닝 예제가 모델 학습 동력학에 미치는 영향을 특정화하여 simulational로 평가합니다. 이 모델은 14백만에서 28억까지의 파라미터를 갖는 다양한 GPT 모델에서 사용되며, 자연어 이해 및 생성 작업에서의 세밀한 성능 평가를 가능하게 합니다. 또한, 체계적인 학습 데이터의 영향 분석 방법과 비교할 때 현저한 차별성과 우수성을 보여 줍니다.

- **Performance Highlights**: GPSfluence는 넓은 범위의 데이터셋(FLAN datasets)과 다양한 NLP 작업에서 효과적임을 확인하였습니다. 평가 지표(metric)로는 test loss 뿐만 아니라 BLEU와 ROUGE 점수도 사용하여 모델의 다양한 능력을 측정하였습니다. 특히 미처리 데이터에 대한 일반화 능력이 뛰어난 것으로 평가되어, GPT 모델의 트레이닝 최적화에 기여할 것으로 기대됩니다.



### Question Generation in Knowledge-Driven Dialog: Explainability and  Evaluation (https://arxiv.org/abs/2404.07836)
- **What's New**: 이 연구에서는 대화형 질문 생성을 위한 새로운 접근 방식으로, 질문을 직접 생성하는 대신 초기에 사실(fact)을 예측하고, 그 다음에 해당 사실에 연계된 질문을 순차적으로 생성하는 모델을 제시합니다. 이 모델은 지식 기반 대화에서 설명 가능성(explainability)과 평가를 중심으로 질문 생성 문제를 탐구합니다.

- **Technical Details**: 제안된 모델은 지식 기반의 대화 상황에서 질문을 생성하기 위해 우선 지식 그래프(Knowledge Graph)에서 트리플(KB triple)을 생성하고, 이를 기반으로 질문을 생성합니다. 이를 통해 모델이 생성한 질문의 관련성(relevance), 사실성(factuality), 그리고 대명사 사용(pronominalisation)의 정확성을 평가할 수 있는 구조를 마련하였습니다. 이 모델은 KGConv 데이터셋을 사용하여 평가되었으며, 기존의 질문만을 생성하는 모델과 비교하여 비슷한 성능을 보였습니다.

- **Performance Highlights**: 본 연구에서 개발된 모델은 설명 가능하며, 참조 없이 모델의 동작을 평가할 수 있는 세밀한 방법을 제공합니다. 모델은 기존 질문 생성 모델과 비교하여 유사한 전반적인 성능을 달성하면서도, 대화의 맥락과 관련하여 더 높은 사실 정확도와 주제 관련성을 보여주었습니다. 또한, 대화응답 생성을 지식 그래프에 기반하여 조절함으로써 주제에서 벗어나거나 사실적이지 않은 질문의 비율을 크게 줄일 수 있는 것으로 나타났습니다.



### MultiLS-SP/CA: Lexical Complexity Prediction and Lexical Simplification  Resources for Catalan and Spanish (https://arxiv.org/abs/2404.07814)
Comments: Submitted to the 40th edition of the SEPLN Conference. Under Revision

- **What's New**: 이 논문에서는 스페인어와 카탈루냐어로 된 어휘 단순화(Lexical Simplification)를 위한 새로운 데이터셋 MultiLS-SP/CA를 소개합니다. 이 데이터셋은 카탈루냐어로 된 최초의 데이터셋이며, 스페인어 데이터의 부족을 상당부분 해결하는 것을 목표로 합니다. 특히 MultiLS-SP는 단어 이해 난이도에 대한 스칼라 등급을 포함하는 최초의 스페인어 데이터셋입니다.

- **Technical Details**: Automatic Lexical Simplification은 복잡한 어휘를 더 간단한 단어로 대체하는 작업입니다. 이 과제는 복잡 단어 식별(Complex Word Identification, CWI), 대체어 생성(Substitute Generation, SG), 대체어 순위 결정(Substitute Ranking, SR), 대체어 선택(Substitute Selection, SS)의 하위 작업을 포함합니다. 이 논문은 스페인어와 카탈루냐어로 된 데이터셋을 구축 과정과 이를 사용한 기본 실험(baseline experiments)을 설명하며, 데이터셋과 관련 스크립트, 주석 자료를 연구 커뮤니티에 제공합니다.

- **Performance Highlights**: 제시된 MultiLS-SP/CA 데이터셋은 기존의 어휘 단순화 및 어휘 복잡도 예측(Lexical Complexity Prediction, LCP) 작업을 위한 효과적인 모델 개발과 평가의 기초를 마련합니다. 이 데이터셋을 사용한 초기 실험은 미래 연구의 기준점으로 활용될 수 있으며, 어휘 단순화 기술을 향상시키는 데 기여할 것으로 기대됩니다.



### Nostra Domina at EvaLatin 2024: Improving Latin Polarity Detection  through Data Augmentation (https://arxiv.org/abs/2404.07792)
Comments: Proceedings of the Third Workshop on Language Technologies for Historical and Ancient Languages

- **What's New**: 이 연구는 라틴어의 감정 극성 탐지 작업에 초점을 맞추고 있으며, 특히 저자원 환경과 복잡한 수사 장르에서 새로운 데이터 확대 방법을 제시합니다. Nostra Domina 팀의 접근 방식은 자동 극성 주석과 여러 라틴어 대규모 언어 모델(Latin large language models, LLMs)을 사용하는 새로운 접근 방식을 소개합니다.

- **Technical Details**: 이 연구는 k-평균(k-means) 알고리즘에 기반한 두 가지 데이터 주석 방법을 개발하였습니다. 첫 번째는 극성 좌표 클러스터링(polarity coordinate clustering)이고, 두 번째는 가우시안 클러스터링(Gaussian clustering)입니다. 또한, 다양한 라틴어 LLM을 활용하여 감정 극성 탐지 작업에서 그 효과를 평가했습니다. 모델 훈련과 선택 과정은 하이퍼파라미터 검색을 통해 이루어졌으며, 특히 감정의 극성과 강도를 분류하기 위해 중심점과 데이터 포인트 간의 거리를 사용하는 방법을 적용했습니다.

- **Performance Highlights**: Nostra Domina 팀의 접근 방식은 EvaLatin 2024 공유 작업 테스트 세트에서 두 번째로 높은 매크로 평균 Macro-F1 점수를 달성하였습니다. 이는 복잡한 라틴어 수사물에서도 효과적인 감정 분석이 가능함을 시사합니다.



### Discourse-Aware In-Context Learning for Temporal Expression  Normalization (https://arxiv.org/abs/2404.07775)
Comments: Accepted at NAACL 2024

- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)을 이용하여 시간 표현 정규화 (Temporal Expression Normalization) 문제를 해결하는 새로운 접근법을 제안합니다. 특히 GPT-3.5-turbo 및 오픈 소스인 Zephyr 모델을 사용하여 문맥 인식 (context-aware) 학습 방식을 적용하였습니다. 이를 통해 다양한 언어 및 도메인에 대한 데이터 희박성 (data scarcity) 문제를 해결하고자 하였습니다.

- **Technical Details**: 이 연구에서는 일련의 예시 (few-shot examples)와 문서 수준의 시간적 문맥 창 (temporal context window)을 활용한 인컨텍스트 학습 (In-context Learning, ICL)을 다룹니다. 다양한 샘플 선택 전략과 창 기반 프롬프트 디자인을 통해 단일 문장 정규화를 넘어서는 성능을 보여줍니다. 예를 들어 시간적 의존성과 담화 정보를 포착하기 위한 적절한 예시 선택이 중요합니다. 또한, LLM의 입력 길이 제한으로 인해 효율적인 샘플 선택이 성능과 적용 가능성에 중요한 요소가 됩니다.

- **Performance Highlights**: 실험 결과, 이 방법은 다양한 도메인 및 언어에 걸친 데이터 세트에서 경쟁 모델과 비교하여 우수한 성능을 보여주었습니다. 특히 훈련 데이터와 거리가 먼 대상 문서에 대해 큰 성능 향상을 보였으며 이는 동적인 예시 포함이 가능하기 때문입니다. 이는 다국어 시나리오에서 특히 유용하게 적용될 수 있습니다.



### Using Letter Positional Probabilities to Assess Word Complexity (https://arxiv.org/abs/2404.07768)
Comments: 25 Pages, 15 Tables

- **What's New**: "새로운 접근법으로, 이 연구는 단어의 복잡성을 직접적으로 측정하는 대신 '잠재적 복잡성(Latent Complexity)'을 이용하여 단어의 복잡성을 평가합니다. 초등학교 그림책의 '간단한' 단어와 고등학교 및 학술 환경의 '복잡한' 단어 샘플을 사용하여 각 단어의 위치별 확률(Letter Positional Probabilities, LPPs)을 분석하여 이 두 클래스 사이의 차이를 연구했습니다."

- **Technical Details**: "연구자는 초등학교와 고등학교 셋팅에서 확보된 간단하고 복잡한 단어 샘플을 수집했으며, 이러한 단어들의 알파벳 구조를 분석하여 위치별 확률을 측정했습니다. 이를테면, 단어가 특정 문자로 시작할 확률과 같은 위치별 확률은 단어의 복잡도를 이해하는 데 중요한 지표로 사용되었습니다. 이 연구에서는 위치별 확률을 바탕으로 두 단어 클래스(간단한 단어와 복잡한 단어)를 구분하는 분류기를 생성, 테스트하고, 최종적으로 97%의 정확도를 가진 분류기를 개발했습니다."

- **Performance Highlights**: "처음 두 데이터셋에서 뽑아낸 84개의 문자 위치 변수는 p<.001 수준에서 유의미하며, 이를 이용하여 첫 분류기는 83%의 정확도로 두 클래스를 분류할 수 있었습니다. 두 번째 데이터셋을 이용한 테스트에서는 66개의 중요한 위치변수를 사용하여 70%의 정확도를 달성했습니다. 마지막으로, 세 개의 데이터셋에서 생성된 극단값을 고저점 단어로 결합한 샘플로 새로운 분류기를 만들어 97%의 높은 정확도를 달성했습니다."



### AnnoCTR: A Dataset for Detecting and Linking Entities, Tactics, and  Techniques in Cyber Threat Reports (https://arxiv.org/abs/2404.07765)
Comments: Accepted at LREC-COLING 2024. Corpus available at this https URL

- **What's New**: 이 논문에서는 사이버 위협 보고서(cyber threat reports, CTRs)에 대한 새로운 데이터셋 AnnoCTR을 소개합니다. 이 데이터셋은 사이버 보안 전문가에 의해 주석이 달린(named entities, temporal expressions, cybersecurity-specific concepts) 보고서를 포함하고 있으며, MITRE ATT&CK 지식 베이스와 Wikipedia로의 링크를 제공합니다. 이는 보다 세밀한 문서 주석이 가능하며, 기존 데이터셋들이 제공하지 못한 점을 개선한 것입니다.

- **Technical Details**: AnnoCTR 데이터셋은 400개의 CTR을 포함하고 있으며, 이 중 120개는 사이버 보안과 관련된 개념을 명시적 혹은 암시적으로 언급한 주석이 달린 보고서입니다. 이 데이터셋은 CC-BY-SA 라이선스를 통해 공개되어 있으며, transformer(트랜스포머) 기반의 named entity recognition (NER) 모델과 entity disambiguation 모델을 사용하여 최대 70%의 F1-score를 달성하였습니다.

- **Performance Highlights**: 초보자 시나리오(few-shot scenario)에서 AnnoCTR 데이터셋을 활용한 실험에서, MITRE ATT&CK 개념을 텍스트에서 식별하기 위해 MITRE ATT&CK의 개념 설명이 효과적인 훈련 데이터 증대 방법임을 확인했습니다. 이는 NER 모델에서는 최대 70%의 F1 점수를, 기술 식별에는 약 65%의 micro-F1 점수를 달성하였습니다.



### ResearchAgent: Iterative Research Idea Generation over Scientific  Literature with Large Language Models (https://arxiv.org/abs/2404.07738)
- **What's New**: 연구생산성 개선을 위해 'ResearchAgent'라는 새로운 대규모 언어 모델(Language Model) 기반 연구 아이디어 생성 에이전트를 제안합니다. 이 모델은 과학문헌에 기반하여 문제, 방법 및 실험 설계를 자동으로 생성하고 반복적으로 개선합니다.

- **Technical Details**: ResearchAgent는 핵심 논문을 시작 포인트로 사용하여 아이디어를 생성하며, 학술 그래프를 통해 연결된 관련 출판물과 콘셉트에 기반한 엔티티 중심 지식 저장소(entity-centric knowledge store)에서 검색된 엔티티를 통합합니다. 또한, 인간의 피어 리뷰(peer review) 방식을 모방하여 여러 'ReviewingAgents'를 사용하여 반복적으로 리뷰와 피드백을 제공하며, 이러한 에이전트들은 실제 인간의 판단을 바탕으로 평가 기준을 설정합니다.

- **Performance Highlights**: 'ResearchAgent'는 다양한 학문 분야의 과학 출판물에 대한 실험을 통해 검증되었으며, 인간과 모델 기반 평가 결과를 바탕으로 새롭고 명확하며 유효한 연구 아이디어를 생성하는 데 효과적인 것으로 나타났습니다.



### Automatic Generation and Evaluation of Reading Comprehension Test Items  with Large Language Models (https://arxiv.org/abs/2404.07720)
Comments: Accepted for publication at the 3rd Workshop on Tools and Resources for People with REAding DIfficulties (READI) at LREC-COLING 2024

- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)의 독해력 평가 문항 생성 및 평가에 대한 능력을 탐구하며, 독일어 독해력 평가 문항을 위한 데이터셋을 구성하여 새로운 평가 척도인 '텍스트 정보성(text informativity)'을 개발하고 사용했습니다. GPT-4와 Llama 2를 비교 분석하여 GPT-4가 좀 더 우수한 성능을 보여주었습니다.

- **Technical Details**: 이 논문은 텍스트의 답변 가능성(answerability)과 추측 가능성(guessability)을 기반으로 한 새로운 척도인 '텍스트 정보성'을 사용하여 자동 및 인간 평가를 수행합니다. 연구진은 독일어로 된 독해력 평가 문항을 생성하고 평가하기 위해 온라인어학 강좌에서 독일어 MCRC 문항을 수집했습니다. 연구는 독해 문항을 생성하고 평가하기 위한 프로토콜을 제시하고, 이를 사용하여 LLMs로 생성된 문항과 인간이 작성한 문항의 질을 평가했습니다.

- **Performance Highlights**: 독일어 독해 문제를 생성하는 데 있어서 GPT-4는 Llama 2보다 더 나은 성능을 보였으며, 자동 평가에서도 GPT-4가 인간 평가자와 가장 유사한 결과를 보여주었습니다. 이러한 접근 방식은 대규모 언어 모델이 독해력 평가 문항 생성과 평가에서 유용할 수 있음을 보여줍니다.



### ODA: Observation-Driven Agent for integrating LLMs and Knowledge Graphs (https://arxiv.org/abs/2404.07677)
Comments: LLM+KG

- **What's New**: 이 논문에서는 지식 그래프(KGs)와 대형 언어 모델(LLMs)을 통합하는 새로운 AI 에이전트 프레임워크, 관찰 주도 에이전트(Observation-Driven Agent, ODA)를 제안합니다. ODA는 지식 그래프의 관찰을 통해 언어 모델의 추론 능력을 증진시키는 것을 목표로 하고 있으며, 관찰, 행동, 반성의 순환적 패러다임을 채택하고 있습니다.

- **Technical Details**: ODA는 지식 그래프에서 관찰할 정보를 재귀적으로 선택하는 새로운 관찰 모듈을 포함하고 있습니다. 이는 지식의 지수적 성장 문제를 해결하며, 관찰된 지식을 행동 및 반성 모듈과 통합합니다. ODA는 세 가지 행동 유형을 수행할 수 있습니다: 이웃 탐색(Neighbor Exploration), 경로 발견(Path Discovery), 그리고 답변(Answering). 이 모델은 QALD10-en, T-REx 등 여러 데이터셋에서 괄목할만한 성능을 보여줍니다.

- **Performance Highlights**: ODA는 QALD10-en과 T-REx 데이터셋에서 각각 12.87%, 8.9%의 정확도 향상을 달성하며 최고의 성능(State-of-the-Art, SOTA)을 보여주었습니다. 이러한 결과는 ODA가 기존 LLM과 KG 통합 방법론을 뛰어넘는 효과적인 솔루션임을 입증합니다.



### Curated Datasets and Neural Models for Machine Translation of Informal  Registers between Mayan and Spanish Vernaculars (https://arxiv.org/abs/2404.07673)
Comments: 13 pages, 3 figures, 8 tables, Submitted to NAACL 2024

- **What's New**: 이 논문에서는 과테말라와 멕시코 남부에서 사용되는 여러 마야어를 대상으로 한 새로운 병렬 코퍼스, MayanV를 개발하고 공개합니다. 이 데이터셋은 가족적이고 비공식적인 언어 사용을 반영하며, 이는 마야어가 일상에서 주로 사용되는 방식을 잘 나타냅니다. 또한, 여러 마야어를 포함해 최대한 많은 자원을 활용하여 신경망 기계 번역(Neural Machine Translation, NMT) 모델을 훈련시키고, 이를 우리의 데이터셋에서만 평가하였습니다.

- **Technical Details**: 이 연구에서는 MayanV 데이터셋을 이용해 다언어 NMT 시스템을 훈련시키고 평가하였습니다. 마야어와 스페인어 사이의 대용량 병렬 코퍼스를 구축하기 위해 공식적인 원본 자료를 사용했으며, 이를 통해 실제로 사용되는 생활 언어를 충실히 반영하고자 했습니다. 사용된 코퍼스는 비공식적, 일상적이며 특정 도메인에 국한되지 않는 언어를 포함하고 있습니다.

- **Performance Highlights**: 개발된 NMT 시스템은 MayanV 데이터셋에서만 평가되었으며, 다른 자원을 사용할 때보다 우수한 성능을 보였습니다. 이는 제공된 데이터셋이 실제 일상 언어 사용을 더 정확하게 포착하고 있음을 시사합니다. 또한, 스페인어 방언 간의 어휘적 차이를 관찰하였고, 표준적인 스페인어 쓰기와 상당한 차이가 있음을 발견했습니다.



### rollama: An R package for using generative large language models through  Ollama (https://arxiv.org/abs/2404.07654)
- **What's New**: rollama는 Ollama API를 래핑하는 R 패키지로, 이를 통해 다양한 Generative Large Language Models (GLLM)를 로컬에서 실행할 수 있습니다. 이 패키지는 텍스트 또는 이미지 데이터를 주석 처리하거나 문서 임베딩(document embedding)에 사용할 수 있는 오픈소스 모델을 쉽게 사용할 수 있도록 설계되었습니다.

- **Technical Details**: rollama는 Ollama API를 사용하여 OpenAI의 API에서 가능한 작업을 더 개인적이고, 재현 가능하며, 무료로 수행할 수 있게 해줍니다. 사용자는 rollama를 활용하거나 확장하여 원하는 작업을 수행할 수 있습니다.

- **Performance Highlights**: rollama를 사용하면 OpenAI와 같은 기능을 로컬 환경에서 자체적으로 구현함으로써 개인정보 보호, 데이터 재현성을 보장할 수 있으며, 추가 비용 없이 이루어집니다.



### Why do small language models underperform? Studying Language Model  Saturation via the Softmax Bottleneck (https://arxiv.org/abs/2404.07647)
- **What's New**: 이 논문에서는 언어 모델링 분야에서 큰 문제 중 하나인 소형 모델의 성능 포화 현상을 다루며, 성능 저하가 높은 순위(rank)의 대상 문맥 확률 분포와 소형 모델의 작은 숨겨진 차원(hidden dimension) 간의 불일치 때문임을 발견했습니다. 이 현상은 softmax bottleneck 현상으로 인해 예측 성능에 제한을 받게 되며, 이 논문은 이를 다양한 설정에서 실험적으로 측정하고 분석합니다.

- **Technical Details**: 이 연구는 작은 모델의 언어 모델링 헤드에서 표현의 퇴화(degeneration) 현상이 강하게 일어나며, softmax bottleneck 이론과 헤드의 rank가 1000 미만일 때 성능에 상당한 제한을 받는다는 것을 실증적으로 및 이론적으로 보여줍니다. 또한, 이 논문은 소형 언어 모델의 성능 포화를 평가하고 규모의 법칙(scaling laws)을 확장함으로써 이 현상을 명확히 특징짓습니다.

- **Performance Highlights**: 횡단적 실험을 통해 작은 모델이 큰 언어 모델만큼의 성능을 내지 못하는 원인을 설명합니다. 특히, Pythia 모델 스위트의 평가에서는 성능 저하와 표현의 퇴화 현상이 뚜렷이 나타나며, 이러한 모델의 최적의 매개변수를 확인하여 언어 모델링 능력의 한계를 고찰합니다.



### Audio Dialogues: Dialogues dataset for audio and music understanding (https://arxiv.org/abs/2404.07616)
Comments: Demo website: this https URL

- **What's New**: 본 연구에서는 일반적인 음향 및 음악에 대한 멀티 턴 대화 데이터셋인 'Audio Dialogues'를 소개합니다. 기존 데이터셋이 단일 턴 대화(예: 오디오 캡셔닝, 오디오 질문 응답)에 중점을 두었다면, 이 새로운 데이터셋은 상호작용적 대화를 통해 오디오를 이해하는 것을 목표로 합니다. 이를 위해 GPT-4를 사용하여 멀티 턴 대화를 생성하고, 복잡한 오디오 입력 간의 비교 질문과 답변도 포함됩니다.

- **Technical Details**: Audio Dialogues 데이터셋은 AudioSet과 MusicCaps의 캡션 주석을 바탕으로 합니다. GPT-4를 사용하여 데이터 셋에 대한 대화 발생(prompting-based approach) 과정을 진행했으며 총 163.8k의 샘플을 포함하고 있습니다. 대화 생성방식은 기존 단일 턴 데이터셋의 한계를 극복하기 위해 설계되었으며, 오디오인코더를 통해 오디오를 토큰으로 변환하고 텍스트 지시사항과 결합됩니다(Large Language Model, LLM).

- **Performance Highlights**: 이 데이터셋의 성능 평가는 이미 향상된 오디오 지원 LLM을 사용하여 수행되었습니다. 특히, 오디오 대화에서는 이전 답변에 따른 후속 질문, 복잡한 맥락 등을 통해 멀티턴 대화가 강화됩니다. 또한, 고품질 대화를 보존하기 위한 데이터 필터링 전략이 구현되었으며, 공개 코드가 제공될 예정입니다.



### Medical mT5: An Open-Source Multilingual Text-to-Text LLM for The  Medical Domain (https://arxiv.org/abs/2404.07613)
Comments: LREC-COLING 2024

- **What's New**: 이 연구는 의료 분야에서 다국어 대규모 언어 모델(Large Language Models, LLMs)을 개발하고 평가하는 최초의 시도 중 하나입니다. Medical mT5는 영어, 스페인어, 프랑스어, 이탈리아어로 구성된 다국어 코퍼스를 사용하여 훈련되었습니다. 이 모델은 의료 도메인에 특화된 최초의 오픈 소스 텍스트-투-텍스트(Text-to-Text) 다국어 모델로, 이전에는 대부분 영어에 초점을 맞춘 평가가 이루어졌던 것에 비해 큰 발전입니다.

- **Technical Details**: Medical mT5는 mT5 모델을 기반으로 하여 의료 분야 데이터에서 추가로 학습하였습니다. 초기 데이터는 총 3B(30억) 단어로, 영어, 스페인어, 프랑스어, 이탈리아어 데이터를 포함하고 있습니다. 특히, 이탈리아어 데이터는 150M(1억 5천만) 단어로 상대적으로 적은 편입니다. 이 모델은 두 가지 새로운 다국어 평가 벤치마크 - 다국어 시퀀스 레이블링과 생성형 질문 응답 - 를 사용하여 평가되었습니다.

- **Performance Highlights**: Medical mT5는 스페인어, 프랑스어, 이탈리아어에서 유사 크기의 텍스트-투-텍스트 모델들을 능가하는 성능을 보여주었으며, 영어에서는 현재의 최첨단(State-of-the-Art) 모델들과 경쟁력 있는 성능을 보였습니다. 이는 다양한 언어의 의료 데이터에 대한 연구를 돕기 위한 중요한 진보로, 하드웨어 요구 사항이 비교적 낮아 다운스트림 작업에 쉽게 적용할 수 있는 장점이 있습니다.



### NoticIA: A Clickbait Article Summarization Dataset in Spanish (https://arxiv.org/abs/2404.07611)
Comments: Under review in the journal Procesamiento del Lenguaje Natural

- **What's New**: 새로운 데이터셋 NoticIA는 스페인어로 된 850개의 클릭베이트 헤드라인을 포함한 뉴스 기사들을 담고 있으며, 각 기사는 사람이 작성한 고품질의 한 문장 요약으로 제공됩니다. 이 데이터셋은 대규모 언어 모델들(Large Language Models, LLMs)이 클릭베이트 헤드라인에서 약속된 정보를 추론하고 연결하는 능력을 평가하는 데 사용됩니다.

- **Technical Details**: 이 연구에서는 여러 대규모 텍스트-투-텍스트 언어 모델들을 스페인어 텍스트 이해 능력을 평가하였으며, 특히 클릭베이트 기사를 요약하는 데 특화된 모델인 ClickbaitFighter를 학습시켜 거의 인간 수준의 성능을 달성하였습니다.

- **Performance Highlights**: ClickbaitFighter 모델은 클릭베이트 기사를 요약하는 데 매우 뛰어난 능력을 보여주며, 이를 통해 사용자가 온라인 콘텐츠 속에서 비판적 사고와 판단력을 발휘할 수 있게 지원합니다. 이 모델은 공개적으로 출시되어, 낚시성 기사의 사용을 압박하고 광고 수익을 늘리기 위한 온라인 뉴스 제공업체들의 기만적인 전술에 대항할 수 있습니다.



### UltraEval: A Lightweight Platform for Flexible and Comprehensive  Evaluation for LLMs (https://arxiv.org/abs/2404.07584)
- **What's New**: 이 논문에서는 'UltraEval'이라는 새로운 평가 프레임워크를 소개하고 있습니다. 이 프레임워크는 Large Language Models (LLMs)의 능력을 평가하고 개선하는 데 중점을 두고, 가볍고 사용자 친화적이며 모듈화와 효율성을 강조합니다.

- **Technical Details**: 'UltraEval'은 모델(model), 데이터(data), 그리고 메트릭스(metrics)의 세 가지 주요 구성 요소로 평가 과정을 재설계함으로써 평가 프로세스를 단순화하고 유연성을 높이고 있습니다. 각 구성 요소는 독립적으로 작동하며 데이터 교환을 통해 상호 작용합니다. 평가 프레임워크는 50개가 넘는 벤치마크와 사용자 지정 프롬프트를 제공하며, HTTP 서비스를 통해 다양한 모델을 지원하고 추론 가속을 제공합니다.

- **Performance Highlights**: 'UltraEval'은 다양한 벤치마크에서 일관된 결과를 제공함으로써 신뢰성을 입증하였습니다. 또한, 이 프레임워크는 모듈식 아키텍처 덕분에 사용자가 새로운 모델, 작업, 메트릭 등을 쉽게 추가하여 평가 워크플로우를 맞춤화할 수 있습니다.



### Comments as Natural Logic Pivots: Improve Code Generation via Comment  Perspectiv (https://arxiv.org/abs/2404.07549)
Comments: The code is publicly available at this https URL

- **What's New**: 이 연구에서는 작은 규모에서 중간 규모의 코드 생성 언어 모델(Language Models, LLMs)의 코드 생성능력을 개선하는 새로운 접근법으로써, 코드 주석(code comments)을 자연어와 프로그래밍 언어 사이의 논리적 연결점(logical pivot)으로 사용하는 MANGO (comMents As Natural loGic pivOts)를 제안합니다. 이 방법은 주석을 이용하여 복잡한 문제 설명을 분해하고, 이를 통해 모델이 더 효과적으로 코드를 생성할 수 있도록 합니다.

- **Technical Details**: MANGO는 주석 대조 훈련 전략(comment contrastive training strategy)과 해당 논리적 주석 디코딩 전략(logical comment decoding strategy)을 포함합니다. 연구에서는 HumanEval과 MBPP 데이터셋을 사용하여 StarCoder와 WizardCoder 등의 백본 모델을 통해 실험을 수행하였으며, 모델의 파라미터 크기는 3B에서 7B 사이입니다. MANGO는 주석을 포함한 코드에 대한 모델의 선호도를 강화하고, 모델이 주석을 통해 코드 논리를 설명하도록 유도함으로써 코드 생성 능력을 향상시킵니다.

- **Performance Highlights**: 실험 결과, MANGO는 유의미한 성능 향상을 보였으며, 예를 들어, StarCoder-7B는 HumanEval에서 pass@10의 경우 최대 7.52의 성과를, WizardCoder-7B는 MBPP에서 최대 4.17의 성과를 달성하였습니다. 또한, MANGO 방법론은 작은 모델 크기에서도 일관된 효과를 보였으며, Chain-of-Thought (CoT) 전략보다 더 높은 로버스트성을 보여주었습니다.



### Decomposing Label Space, Format and Discrimination: Rethinking How LLMs  Respond and Solve Tasks via In-Context Learning (https://arxiv.org/abs/2404.07546)
Comments: 36 pages, 8 figures

- **What's New**: In-context Learning (인컨텍스트 러닝, ICL)은 대규모 언어 모델(Large Language Models, LLMs)의 발전과 함께 강력한 기능으로 부상하고 있습니다. 이 논문은 ICL의 전반적인 성능을 레이블 공간, 형식, 그리고 구별(discrimination)의 세 가지 차원으로 분해하여 평가하고, 네 가지 일반적인 용도의 대규모 언어 모델을 다양한 작업에 걸쳐 평가합니다.

- **Technical Details**: 연구자들은 여러 샘플을 통해 LLMs에 지시를 함으로써, 모델을 업데이트하지 않고 광범위한 작업을 수행할 수 있는 ICL의 가능성을 보여줍니다. 실험을 통해, 설명의 효과는 LLMs의 언어 구별 능력 향상에는 미비하지만, 레이블 공간과 형식을 조절하는 데에는 상당한 효과가 있음을 발견했습니다. 이는 마치 LLMs에게 자세한 지시를 하는 것과 유사한 기능으로 작동합니다.

- **Performance Highlights**: 연구에서는 ICL이 LLM의 반응을 원하는 레이블 단어로 유도하는 데 효과적임을 확인했습니다. 또한, ICL 메커니즘의 검색 기능을 심층 분석하였고, 의미론적으로 가장 유사한 예제를 검색하는 것이 모델의 구별 능력을 뚜렷하게 향상시킬 수 있다는 점을 알아냈습니다.



### From Words to Numbers: Your Large Language Model Is Secretly A Capable  Regressor When Given In-Context Examples (https://arxiv.org/abs/2404.07544)
Comments: 50 pages, 48 figures, preprint

- **What's New**: 이 연구는 LLama2, GPT-4, Claude 3 등의 선행 훈련된 대규모 언어 모델(Large Language Models, LLMs)을 사용하여 비선형 및 선형 회귀 문제를 해결할 수 있는 능력을 분석합니다. 특히, 추가적인 훈련이나 기울기 업데이트 없이 컨텍스트 예시만을 통해 이러한 모델들이 어떻게 성능을 발휘하는지 탐구했습니다. 본 연구는 LLM들이 전통적인 지도 학습 방법 또는 앙상블 방법 (Random Forest, Gradient Boosting 등)과 비교해도 우수하거나 비슷한 성과를 낼 수 있음을 시사합니다.

- **Technical Details**: LLMs는 다음 토큰 예측(next-token prediction)을 통해서 인컨텍스트 학습(in-context learning, ICL) 능력을 발휘합니다. 이 연구는 복합적인 회귀 함수를 학습할 수 있는지 평가하기 위해 합성 회귀 데이터셋(synthetic regression datasets)을 사용합니다. 이 데이터셋은 알고리즘으로 생성되어 데이터의 구조를 파악하는데 적합하며, 난이도 조절이 가능합니다. 연구는 선형 회귀(linear regression)부터 Friedman #2와 같은 높은 비선형 함수까지 다양한 문제를 다룹니다.

- **Performance Highlights**: Claude 3 모델은 선형 회귀 데이터셋에서 선형 회귀 모델과 비슷한 성능을 보이며, Random Forest나 Gradient Boosting과 같은 다른 지도 학습 방법들을 크게 앞섭니다. 이 연구는 인컨텍스트 예시의 수가 증가함에 따라 LLM의 성능이 어떻게 확장되는지를 살펴보고, 온라인 학습에서의 regret 개념을 빌려와 LLM이 선형으로 regret을 얻을 수 있다는 것을 경험적으로 보여줍니다.



### Best Practices and Lessons Learned on Synthetic Data for Language Models (https://arxiv.org/abs/2404.07503)
- **What's New**: 이 논문에서는 다양한 AI 애플리케이션과 동시에 실제 문제를 해결하는 데 필요한 고품질 데이터를 생성하기 위해 합성 데이터(synthetic data)의 중요성 및 활용법에 대해 설명합니다. 최신 연구 결과와 실험적 증거를 제시하여 효과적임을 강조하며, 합성 데이터를 사용함으로써 데이터 부족(data scarcity), 개인 정보 보호(privacy concerns), 데이터 수집 및 주석 비용(cost of data acquisition and annotation)의 문제를 극복할 수 있는 방안을 제시합니다.

- **Technical Details**: 합성 데이터는 실세계의 데이터 패턴을 모방하는 인공 데이터로, 알고리즘, 생성 모델(generative models) 또는 시뮬레이션을 통해 생성됩니다. 이는 AI 모델에 대한 대량의 훈련 및 테스트 데이터를 제공하여 모델 성능과 일반화를 개선할 수 있습니다. 특히, 다양한 도메인(예: 보건 의료, 금융, 학습 언어 등)에서 데이터 보호 및 개인정보 보호를 중시하는 추세와 맞물려, 합성 데이터는 사생활 침해 문제 없이 데이터를 생성할 수 있는 중요한 대안이 됩니다.

- **Performance Highlights**: 합성 데이터는 수학 추론(mathematical reasoning), 코드 이해(code reasoning)와 같은 복잡한 AI 작업에서 인상적인 성능을 보여주었습니다. 예를 들어, AlphaGeometry와 같은 모델은 1억 개의 합성 데이터 포인트를 통해 복잡한 기하 문제를 해결하는 능력을 시연했습니다. 또한, 합성 데이터를 활용한 랭귀지 모델(language models)은 개인화된 코드 샘플을 통해 코드 작성능력을 향상시키며, 자연 언어 데이터셋을 확장하는 데에도 기여하고 있습니다. 이들 데이터는 AI 모델을 개선하고 다양한 산업에 적용하기 위한 강력한 수단으로 자리 잡고 있습니다.



### Leveraging Data Augmentation for Process Information Extraction (https://arxiv.org/abs/2404.07501)
Comments: Accepted at BPMDS 2024 (this https URL), to be printed

- **What's New**: 이 논문은 자연어 텍스트에서 비즈니스 프로세스 모델을 생성하는 작업에 데이터 증강(Data Augmentation, DA)을 적용하는 연구를 다룹니다. 기존에 주로 사용되던 규칙 기반 시스템 대신, 머신러닝 기법을 활용하여 성능을 향상시킬 수 있는 방법을 제시하며, 간단한 데이터 증강 기술이 어떻게 머신러닝 방법의 성능을 개선할 수 있는지 실험적으로 보여줍니다.

- **Technical Details**: 연구자들은 자연어에서 비즈니스 프로세스 관련 정보를 추출하고, 이를 통해 구체적인 모델을 생성하는 두 단계 방법을 사용합니다. 데이터 증강은 이미지의 의미를 유지하면서 간단한 조작(예: 회전, 노이즈 추가)을 통해 컴퓨터 비전에서 널리 사용되는 기법이지만, 자연어 처리에서는 의미의 보존이 더 어렵습니다. 이 연구에서는 텍스트 의미를 변경하지 않으면서 데이터를 증강하는 다양한 기법을 사용하고, 이러한 증강이 정보 추출 작업에 어떻게 영향을 미치는지 분석합니다.

- **Performance Highlights**: 실험 결과, 데이터 증강 기법을 적용한 결과, 언급 추출(Mention Extraction)의 F1 점수가 2.9%포인트, 관계 추출(Relation Extraction)의 F1 점수가 4.5%포인트 향상되었습니다. 이는 데이터 증강이 자연어 텍스트에서 비즈니스 프로세스 모델 생성 작업에 유용하게 적용될 수 있음을 시사합니다.



### Interactive Prompt Debugging with Sequence Salienc (https://arxiv.org/abs/2404.07498)
- **What's New**: Sequence Salience는 인풋 살리언스(input salience, IS) 메소드를 이용하여 복잡한 LLM(Large Language Models) 프롬프트를 디버깅하고 반복 작업을 용이하게 하는 시각적 도구를 소개합니다. 이 도구는 토큰 수준 살리언스를 단어, 문장, 또는 단락 수준으로 집계할 수 있도록 하며, 사용자가 프롬프트를 수정하고 즉시 결과를 확인하면서 빠르게 반복 수정할 수 있습니다. 또한, 오픈 소스 플랫폼인 Learning Interpretability Tool (LIT)을 기반으로 구축되어 있습니다.

- **Technical Details**: Sequence Salience는 TypeScript와 Python을 사용하여 구현되었습니다. 프론트엔드는 Lit 웹 컴포넌트 프레임워크를 사용하고, 백엔드는 LLM을 호스팅하며 프롬프트에 대한 응답을 생성하는 generate(), 텍스트의 토큰을 반환하는 tokenize(), 그리고 프롬프트와 타겟 아웃풋 시퀀스에 대한 살리언스 점수를 반환하는 salience() 함수를 포함합니다. 이 시스템은 다양한 LLMs와 호환되며, Gemma, Llama 2, Mistral, GPT-2 등의 구현 예제를 제공합니다.

- **Performance Highlights**: Sequence Salience는 복잡한 프롬프트 전략을 작업하는 데 도움이 되는 여러 사례 연구를 포함합니다. 이는 프랙티셔너에게 모델의 예상 출력에 중요한 프롬프트의 부분을 시각화하고, 빠르게 반응하여 문제를 수정하고 프롬프트를 개선할 수 있는 기능을 제공합니다. UI 기능은 브라우저 기반으로 구현되어 사용자가 프롬프트를 입력하고 수정할 수 있으며, 살리언스를 계산하고 텍스트 위에 히트맵으로 표시하는 기능을 포함합니다.



### Laissez-Faire Harms: Algorithmic Biases in Generative Language Models (https://arxiv.org/abs/2404.07475)
Comments: 16 pages (44 if including supplementals), 4 figures (20 if including supplementals)

- **What's New**: 현재 탑재된 생성 언어 모델(Language Models, LMs)들의 빠른 배포가 다양한 소비자들의 복지에 영향을 미치는 사회적 편견에 대한 우려를 불러일으켰습니다. 이 연구는 대부분의 경우 개체의 정체성을 명시적으로 제시하지 않고 자연스러운 사용 예를 통해 폭넓게 생성 언어 모델의 편견을 조사하여, 'laissez-faire' 상황에서 다양한 인종, 성별, 성적 지향을 가진 소수자에 대한 누락, 열등, 고정관념을 반복하고 있음을 발견했습니다.

- **Technical Details**: 연구는 ChatGPT3.5, ChatGPT4, Claude2.0, Llama2, PaLM2와 같은 주요 생성 언어 모델들에서 합성된 텍스트를 분석했습니다. 소수자에 대한 열등하게 표현되거나 표현력이나 권력이 부여된 묘사에 비해 수백에서 수천 배 더 많은 편견된 결과들이 생성되었습니다. 또한 소수자들이 인지 성능 저하와 부정적 자아 인식 증가를 초래할 수 있는 스테레오타입 위협(stereotype threat)을 유발하는 것으로 알려진 편견(예: 영원한 외국인)을 빈번하게 포함하고 있음을 문서화하였습니다.

- **Performance Highlights**: 이 연구는 언어 모델이 생성하는 출력에서 소수자들이 표현되는 방식에 대한 광범위한 편견 증거를 제시하며, 이러한 편견은 소수자 개인들에 대한 정신적 피해를 크게 증가시킬 수 있습니다. 결과적으로, 소비자들을 언어 모델로 인한 차별적 피해로부터 보호하고 다양한 소비자들을 지원하기 위한 중요한 AI 교육 프로그램에 투자할 필요성을 강조합니다.



### Scalable Language Model with Generalized Continual Learning (https://arxiv.org/abs/2404.07470)
Comments: The Twelfth International Conference on Learning Representations

- **What's New**: 지속 가능한 학습의 중요성이 커지면서 최근 이에 대한 연구가 활발해지고 있습니다. 이 연구에서는 경험 재생(replay), 최적화 제약(optimization constraints), 추론 작업 ID(inference task-ID)에 의존하지 않고 적응 가능한 언어 모델을 개발하는 새로운 방법인 Scalable Language Model (SLM)과 Joint Adaptive Re-Parameterization (JARe), Dynamic Task-related Knowledge Retrieval (DTKR)을 소개합니다. 이 방법들은 다양한 업무 타입과 도메인에 걸쳐 효과적인 연속 학습을 가능하게 합니다.

- **Technical Details**: SLM 방법론은 벡터 공간(vector space) 검색을 사용하여 언어 모델을 동적으로 조정하고 관리하는 데 도움이 됩니다. JARe는 특정 하류 작업(downstream tasks)에 따라 사전 훈련된 모델을 적응적으로 재구성하는 과정을 구현하고, DTKR은 각 입력 인스턴스에 가장 관련성이 높은 지식을 식별하도록 활용됩니다. 이 두 기술은 새롭고 연속적인 학습 과정을 원활하고 효율적으로 만드는 데 중요한 역할을 합니다.

- **Performance Highlights**: 다양한 벤치마크(benchmarks)에서 최첨단 기술(state-of-the-art) 성능을 달성하였고, 특히 BERT, T5, LLaMA-2 모델에서 눈에 띄는 성과를 보였습니다. 이 방법은 최대 80%까지 잊어버림을 줄이는 것은 물론, 성능 저하도 0.5% 이하로 유지하여, 여러 분야와 작업 유형에 걸쳐 연속 학습의 일반화 능력을 입증하였습니다.



### "Confidently Nonsensical?'': A Critical Survey on the Perspectives and  Challenges of 'Hallucinations' in NLP (https://arxiv.org/abs/2404.07461)
- **What's New**: 이 연구는 NLP(Natural Language Processing)에서 '환각(hallucination)'이라는 용어의 불일치와 그 개념에 대한 명확한 정의와 틀을 제시하는 것을 목표로 합니다. 이를 위해 103개의 피어 리뷰 논문을 분석하고, NLP 및 AI 분야의 171명의 실무자를 대상으로 한 설문조사를 통해 다양한 관점을 조사했습니다.

- **Technical Details**: 연구자들은 NLP에서 사용되는 '환각'의 정의와 공통 프레임워크를 조사하기 위해 103개의 논문을 검토했으며, 이에 대한 이해도를 높이기 위해 NLP 연구자 및 학자들을 대상으로 한 추가 설문조사를 실시했습니다. 이 설문조사는 환각의 사회적 영향과 그 함의에 대한 주제적 이해를 제공합니다. 또한, 이 용어에 대한 일관되지 않은 사용을 조명하고, 향후 환각을 이해하고 해결하기 위한 윤리적 틀을 구축하려는 노력을 강조했습니다.

- **Performance Highlights**: 이 연구는 NLP에서의 '환각' 현상에 대한 연구가 최근 증가하고 있음을 보여줍니다. 특히, 연구자들은 이미지 캡셔닝(image captioning) 및 텍스트 생성(text generation)과 같은 작업에서 모델이 훈련 데이터셋에서 존재하지 않는 객체나 진술을 생성하는 경향을 보일 때 환각을 나타내는 구체적인 오류에 주목했습니다. 이 연구는 NLP 분야에서 환각을 보다 명확하게 정의하고 해석하는 데 필요한 이론적 틀과 윤리적 지침을 마련하는 것을 목표로 합니다.



### JetMoE: Reaching Llama2 Performance with 0.1M Dollars (https://arxiv.org/abs/2404.07413)
- **What's New**: JetMoE-8B는 향상된 성능과 접근성을 제공하는 새로운 대형 언어 모델(LLM)입니다. 이 모델은 공개 데이터셋과 오픈 소스 코드를 사용하여 30,000시간의 H100 GPU와 1.25T 토큰으로 $100,000 미만의 비용으로 훈련됐습니다. 또한, JetMoE-8B는 Llama2-7B 모델보다 우수한 성능을 보이며, JetMoE-8B-Chat은 Llama2-13B-Chat 모델을 능가합니다.

- **Technical Details**: JetMoE-8B는 자원 사용량을 줄이기 위해 Sparsely-gated Mixture-of-Experts (SMoE) 아키텍처를 채택하였으며, attention과 feedforward 계층에서 sparse activation을 활용합니다. 이 모델은 총 8B (billion) 파라미터를 가지고 있으나, 각 입력 토큰마다 단 2B의 파라미터만 활성화시켜 추론 계산량을 Llama2-7B 모델에 비해 약 70% 줄였습니다. 또한, 모든 훈련 파라미터 및 데이터 혼합은 투명하게 공개되어 연구 및 협력 촉진을 도모합니다.

- **Performance Highlights**: JetMoE-8B는 경제적인 자원으로 놀라운 성능 향상을 달성하였으며, 이는 대형 언어 모델 훈련 비용의 효율성을 증대시키는 중요한 사례로 평가됩니다. 이 모델은 특히 dense models 대비 높은 효율성과 확장성을 제공하는 기술적 진보를 보여주었으며, 학계 및 연구자들에게 접근 가능하고 사용하기 쉬운 도구를 제공합니다.



### LLMs in Biomedicine: A study on clinical Named Entity Recognition (https://arxiv.org/abs/2404.07376)
- **What's New**: 이 연구는 의료 분야에서 대규모 언어 모델(Large Language Models, LLMs)의 성능을 향상시키기 위한 방법을 탐구합니다. 특히, Named-Entity Recognition (NER, 개체명 인식) 작업에 초점을 맞추어, 정교하게 설계된 프롬프트의 중요성과 적절한 인-콘텍스트 예시(in-context examples)의 선택이 LLM의 성능을 어떻게 향상시킬 수 있는지를 분석합니다. 또한, Retrieval-Augmented Generation (RAG)에 영감을 받아 의료 지식 베이스를 활용하는 새로운 방법을 제안하여, 제로샷(zero-shot) 클리니컬 NER에서 F1 점수를 향상시킬 수 있는 방법을 제시합니다.

- **Technical Details**: 이 연구는 다양한 데이터셋과 모델 크기에 따른 프롬프트 디자인과 인-콘텍스트 학습(In-Context Learning, ICL)의 예시 선택 전략의 효과를 비교 분석합니다. 특히, TANL과 DICE라는 두 가지 텍스트-투-텍스트 포맷을 바이오메디컬 NER에 적용하였으며, 가장 효과적인 포맷은 데이터셋의 복잡성과 모델 크기에 따라 달라질 수 있음을 발견했습니다. 연구는 또한, 특정 사례들이 테스트 샘플과의 유사성에 따라 ICL 성능이 크게 달라질 수 있음을 보여주며, 이를 통해 보다 정확한 모델 응답을 위한 KATE(Knn-Augmented in-conText Example selection) 방법을 도입합니다.

- **Performance Highlights**: 이론적 연구와 실험을 통해, 선정된 인-콘텍스트 예시를 활용한 학습이 무작위 예시 선택보다 뛰어난 성능을 보였으며, 특정 전략을 사용할 때 모든 벤치마크 데이터셋에 대해 약 15-20%의 F1 점수 증가를 보여 주었습니다. 최적화된 프롬프트와 외부 의료 지식의 통합 사용은 일반적으로 사용되는 LLMs의 한계를 극복하고 특수한 의료 NER 요구를 충족시키는 데 중요한 역할을 합니다.



### We're Calling an Intervention: Taking a Closer Look at Language Model  Adaptation to Different Types of Linguistic Variation (https://arxiv.org/abs/2404.07304)
Comments: Preprint. Under review

- **What's New**: 이 연구에서는 언어 모델(language model)이 언어적 변이(linguistic variation)를 가진 텍스트에 어떻게 적응하는지 이해하는 데 도움이 되는 일련의 개입(interventions)과 실험을 소개합니다. 각기 다른 훈련 데이터의 크기와 종류를 사용하여 이러한 개입을 적용함으로써, 언어 모델이 언어적 변이를 처리하는 데 어려움을 겪는 요인들을 파악할 수 있었습니다. 이 연구는 향후 방언 NLP(dialectal NLP)와 언어 모델의 언어적 변이에 대한 강건성(robustness)을 향상시키기 위한 연구에 중요한 정보를 제공합니다.

- **Technical Details**: 연구자들은 문자(character-level), 부분 단어(subword-level), 단어(word-level) 변화를 포함하는 여러 가지 언어적 변이 특성에 대한 개입을 개발하였습니다. 예를 들어, '커피(coffee)'가 '커,피(co, fe, e)'로 토큰화(tokenization) 될 때와 같은 변이를 실험하여 BERT와 같은 언어 모델이 이러한 변형된 입력에 어떻게 적응하는지 관찰하였습니다. 또한, 이 연구는 변형을 유도하기 위해 개발된 10가지 개입이 포함된 실험 세트를 제공하며, 공개적으로 사용할 수 있는 코드(code)도 제공합니다.

- **Performance Highlights**: 언어 모델은 변형된 텍스트에 대해 초기에는 매우 낮은 이해력을 보였지만, 적절한 훈련 데이터를 사용할 때 성능이 크게 향상되었습니다. 특히, 단어 수준의 새로운 의미나 사용법을 포함하는 변이에서는 많은 양의 데이터가 필요했으나, 이를 통해 언어 모델의 성능이 크게 개선되었습니다. 반면, 문자 수준의 변이에서는 적은 양의 데이터로도 성능이 개선되었지만, 일정 수준 이상에서는 더 많은 데이터가 해결책이 되지 않음을 발견하였습니다.



### Personality-affected Emotion Generation in Dialog Systems (https://arxiv.org/abs/2404.07229)
Comments: Accepted by ACM Transactions on Information Systems

- **What's New**: 새로운 과제로 '성격 영향 감정 생성' (Personality-affected Emotion Generation)을 제안하며, 이는 대화 시스템에 성격 특성을 부여하고, 성격과 일관된 감정을 생성하여 유저와의 상호작용을 개선하고자 합니다. 이 연구는 성격과 감정 사이의 상호작용을 모델링하여 대화 시스템의 감정 생성 성능을 향상시키는 방법을 탐구합니다.

- **Technical Details**: 연구팀은 '성격 감정 라인 데이터 세트' (Personality EmotionLines Dataset, PELD)를 구축하고, 대화에서 성격과 감정 정보를 통합하는 방법을 개발했습니다. 대화 상태에서 성격 연산을 '전이 무게' (transition weight)로 모델링하여 감정의 미묘한 변화를 효과적으로 예측합니다. 또한, 다양한 감정 정보 (다중 입자성 감정 정보)를 추출하고 통합하는 새로운 방법을 제안합니다.

- **Performance Highlights**: PELD 데이터셋에서의 실험 결과, 기존 BERT-base 모델 대비 감정 생성 성능이 매크로-F1 스코어로 13% 향상되고, 가중치 F1 스코어로 5% 개선된 것으로 나타났습니다. 이는 성격 특성을 고려함으로써 감정의 일관성과 적절성이 크게 향상됨을 보여줍니다.



### Any2Point: Empowering Any-modality Large Models for Efficient 3D  Understanding (https://arxiv.org/abs/2404.07989)
Comments: Code and models are released at this https URL

- **What's New**: 이 논문에서는 Any2Point, 어떠한 모달리티(modality)로 사전 훈련된 대규모 모델을 3D 분야로 효과적으로 전환할 수 있는 새로운 프레임워크(framework)를 제안합니다. Any2Point는 기존의 2D에서 3D로의 전환 방식들이 가진 공간적 지오메트리(spatial geometries)의 손실과 많은 계산 비용 문제를 극복하기 위해 설계되었습니다.

- **Technical Details**: Any2Point는 '3D-to-any (1D 혹은 2D) virtual projection (가상 투영)' 기법을 도입하여, 입력된 3D 포인트들을 원본 모달리티의 1D 라인(line)이나 2D 평면(plane)과 연결하는 위치 매핑을 확립합니다. 이를 통해 3D 좌표들을 사전 훈련된 모델의 기존 위치 인코딩(position encoding)을 사용하여 인코딩할 수 있습니다. 추가로, 'Any-to-3D guided adapter (어댑터)' 모듈을 각 트랜스포머 블록에 삽입하여, Parameter-Efficient Fine-Tuning (PEFT)을 가능하게 합니다. 이 어댑터는 1D/2D 공간 지침을 활용하여 3D 토큰들의 지역적 특징을 집합시키도록 도와, 고도의 특징 상호 작용을 촉진합니다.

- **Performance Highlights**: Any2Point는 전체 훈련 가능한 파라미터의 단 1.0%만을 사용하면서도, 기존의 상태 기술(state-of-the-art, SOTA) 3D 사전 훈련된 모델들에 비해 우수한 성능을 보여주었습니다. ScanObjectNN 데이터셋에서는 기존 최고 모델보다 +1.3% 높은 91.9%의 정확도를 달성했으며, ModelNet40에서는 94.3%의 정확도를 기록했습니다. 2D 비전, 언어, 오디오 등 다양한 모달리티의 사전 훈련된 모델을 사용할 때에도 일관된 결과를 보여주며, Any2Point의 견고함을 입증했습니다.



### Manipulating Large Language Models to Increase Product Visibility (https://arxiv.org/abs/2404.07981)
- **What's New**: 본 연구에서는 인공지능 언어 모델(Large Language Models, LLMs)을 이용한 쇼핑 검색 엔진에서 제품의 가시성을 높일 수 있는 방법을 살펴보았습니다. 이를 위해 제품 정보 페이지에 전략적 텍스트 시퀀스(Strategic Text Sequence, STS)를 추가하는 방법을 제시하였으며, 이 방법이 언어 모델의 추천 결과에 어떤 영향을 미치는지 분석했습니다.



### OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real  Computer Environments (https://arxiv.org/abs/2404.07972)
Comments: 51 pages, 21 figures

- **What's New**: 새롭게 소개되는 OSWorld는 다양한 운영체제(Ubuntu, Windows, macOS)를 지원하는 최초의 확장 가능한 실제 컴퓨터 환경입니다. 이 환경은 멀티모달(multimodal) 에이전트를 위해 태스크 설정, 실행 기반 평가, 그리고 상호작용 학습을 지원하며, 개방형 컴퓨터 태스크를 평가하는 통합된 환경을 제공합니다. 이를 통해, 사용자는 실제 웹 및 데스크탑 애플리케이션, OS 파일 I/O, 다양한 애플리케이션을 거쳐가는 워크플로우를 포함하는 369개의 컴퓨터 태스크 벤치마크를 구축할 수 있습니다.

- **Technical Details**: OSWorld는 각 태스크 예시를 실제 컴퓨터 사용 사례에서 파생시키고, 초기 상태 설정 구성 및 맞춤형 실행 기반 평가 스크립트를 포함하여 신뢰할 수 있고 재현 가능한 평가를 제공합니다. 연구팀은 다중 모드 일반화(generalist) 에이전트의 개발에 필요한 통찰력을 제공하기 위해 OSWorld를 사용하여 포괄적인 분석을 수행하였습니다.

- **Performance Highlights**: 인간은 72.36%의 태스크를 완료할 수 있는 반면, 가장 성능이 좋은 모델은 GUI 구동(GUI grounding)과 작업 지식(operational knowledge)에서 어려움을 겪으며 12.24%의 성공률을 보였습니다. 이러한 결과는 LLM/VLM(Language and Vision Language Models) 기반 에이전트가 컴퓨터 조수로서 충분하지 않음을 보여줍니다. OSWorld의 철저한 평가를 통해 에이전트들의 주요 취약점을 식별할 수 있었습니다.



### EduAgent: Generative Student Agents in Learning (https://arxiv.org/abs/2404.07963)
- **What's New**: 학생 시뮬레이션에 관한 최신 연구에서는 학생들의 동적이고 다양한 학습 행태를 실시간으로 모방하고 예측할 수 있는 EduAgent라는 새로운 생성적 에이전트 프레임워크(generative agent framework)를 제안합니다. 이 프레임워크는 인지 과학(cognitive science)에서 밝혀진 이론적 지식을 사용하여 대규모 언어 모델(LLMs)을 안내합니다.

- **Technical Details**: EduAgent는 다양한 학생 페르소나(student personas), 학습 행동, 및 학습 결과 간의 미묘한 상호작용을 포착하기 위해 LLMs을 이끌어내는 새로운 방식을 탐구합니다. 특히, EduAgent는 학습 행동을 모델링하기 위한 큰 규모의 새로 주석된 데이터셋(EduAgent310)과 함께, 인지적 사전 지식(cognitive prior knowledge)을 이용한 모듈식 구조를 도입하여 실제 및 가상 학생들의 학습 행동을 더 세밀하게 모방할 수 있도록 설계되었습니다.

- **Performance Highlights**: 제안된 프레임워크는 실제 학생의 학습 행동과 테스트 결과를 정확하게 예측할 뿐만 아니라, 실제 데이터 없이도 현실적인 학습 행동을 생성할 수 있음을 보여줍니다. EduAgent를 사용하여 생성된 또 다른 데이터셋(EduAgent705)은 N=705의 가상 학생들을 포함하며, 이 가상 학생들은 실제 학생들의 행동 패턴과 일관되게 나타났습니다.



### An efficient domain-independent approach for supervised keyphrase  extraction and ranking (https://arxiv.org/abs/2404.07954)
- **What's New**: 이 연구에서는 단일 문서에서 핵심 구문을 자동 추출하기 위한 감독 학습 접근법을 제시합니다. 이 접근법은 외부 지식 기반, 사전 훈련된 언어 모델, 또는 단어 임베딩에 의존하지 않고, 후보 구문의 통계적 및 위치적 특성을 단순하게 계산하여 사용합니다.

- **Technical Details**: 제안된 솔루션의 순위 결정 구성 요소는 비교적 가벼운 앙상블 모델입니다. 이 모델은 귀금속 키워드(corpus of 'golden' keywords)나 외부 지식 코퍼스에 의존하지 않음으로써, 감독 학습 방식이지만 미감독(Unsupervised) 솔루션의 이점을 어느 정도 유지합니다.

- **Performance Highlights**: 벤치마크 데이터셋에서의 평가는 제안된 접근법이 여러 최신 기술(BaseLine) 모델들, 특히 모든 비감독(deep learning-based unsupervised) 모델들 보다 현저히 높은 정확도를 달성하고, 일부 감독 학습 기반의 딥 러닝 모델들과도 경쟁력 있는 성능을 보여 줍니다.



### DesignQA: A Multimodal Benchmark for Evaluating Large Language Models'  Understanding of Engineering Documentation (https://arxiv.org/abs/2404.07917)
- **What's New**: 이 연구는 'DesignQA'라는 새로운 벤치마크를 소개하며, 이는 기술 문서에서 공학 요구 사항을 이해하고 적용하는 데 있어 멀티모달 대형 언어 모델(MLLMs)의 능력을 평가하는 데 중점을 둡니다. DesignQA는 Formula SAE 학생 경연 대회에서 파생된 텍스트 설계 요구 사항, CAD 이미지, 공학 도면을 포함한 멀티모달 데이터를 통합합니다. 이 벤치마크는 공학 설계에 필요한 복잡함과 멀티모달성을 강조하며, 실제 공학 도전 과제에 중점을 둡니다.

- **Technical Details**: DesignQA는 1451개의 질문을 포함하고 있으며 Formula SAE 2024 규칙과 MIT Motorsports 팀이 제공한 데이터(CAD, 문서 등)를 기반으로 합니다. 이 벤치마크는 이미지와 텍스트가 서로 다른 출처에서 나온 경우에도 모델이 정보를 종합할 수 있는 능력을 평가합니다. 벤치마크는 '규칙 이해(Rule Comprehension)', '규칙 준수(Rule Compliance)', '규칙 추출(Rule Extraction)'의 세 부분으로 나뉘며, 각 부분은 자동 평가 메트릭(automatic evaluation metrics)을 특징으로 합니다.

- **Performance Highlights**: GPT4와 LLaVA와 같은 최신 MLLMs를 벤치마크로 사용하여 평가한 결과, 이 모델들은 기술 문서를 해석하는 데 있어 잠재력을 보였지만, 공학 설계에 필요한 자세한 요구 사항을 정확하게 추출하고 적용하는 데는 여전히 중요한 제한이 있음을 발견했습니다. 특히 GPT4는 규칙을 컨텍스트 윈도우(context window)를 통해 받았을 때 DesignQA에서 가장 좋은 성능을 보였습니다.



### RecurrentGemma: Moving Past Transformers for Efficient Open Language  Models (https://arxiv.org/abs/2404.07839)
- **What's New**: 새로운 언어 모델인 RecurrentGemma-2B를 소개합니다. 이 모델은 Google의 새로운 Griffin 아키텍처를 사용하여 우수한 언어 처리 성능을 제공합니다. RecurrentGemma-2B는 고정된 크기의 상태를 사용하여 메모리 사용을 줄이고 긴 시퀀스에 대한 효율적인 추론을 가능하게 합니다.

- **Technical Details**: RecurrentGemma-2B는 global attention을 배제하고 linear recurrences와 local attention을 혼합하여 시퀀스를 모델링합니다. 이 아키텍처는 입력 시퀀스를 고정된 크기의 상태로 압축하여 성능을 희생하지 않으면서 메모리 사용을 줄입니다. 또한, JAX 코드와 PyTorch 구현을 제공하며, 이는 TPU에서 linear recurrence를 수행하는 Pallas 커널을 포함합니다.

- **Performance Highlights**: RecurrentGemma-2B는 Gemma-2B와 비슷한 성능을 제공하며, Gemma-2B보다 50% 적은 토큰으로 훈련되었습니다. RecurrentGemma-2B는 Gemma-2B보다 추론 속도가 훨씬 빠르며, 긴 시퀀스에서 높은 처리량(throughput)을 달성합니다. Human evaluation studies에서 RecurrentGemma-2B-IT는 Mistral 7B 모델에 대해 43.7%의 승률을 보여주었습니다.



### Heron-Bench: A Benchmark for Evaluating Vision Language Models in  Japanes (https://arxiv.org/abs/2404.07824)
- **What's New**: 새로운 일본어 비전 언어 모델(Vision Language Models, VLMs) 평가 벤치마크인 'Japanese Heron-Bench'가 도입되었습니다. 이 벤치마크는 일본 문맥에 맞춘 이미지-질문 쌍으로 구성되어 있으며, VLM의 일본어 능력을 평가하는 데 사용됩니다. 또한, 일본어 시각적 지시 튜닝 데이터셋(training dataset)을 사용해 훈련된 기준 일본어 VLM(baseline Japanese VLM)도 제시되었습니다.

- **Technical Details**: Japanese Heron-Bench는 일본 특유의 이미지와 102개의 질문으로 구성되어 있으며, 다양한 범주(category)의 질문을 포함합니다(예: Conversation, Detail, Complex). 이 평가 시스템은 LLaVA-Bench와 유사하게 설계되었으며, GPT-4 API를 사용하여 평가 점수를 산출합니다. 모델은 OpenAI의 CLIP(Large Patch 14, 336)를 이미지 인코더로 사용하고, StabilityAI의 'japanese-stablelm-base-alpha-7b'를 일본어 LLM으로 사용합니다. 학습은 visual instruction tuning 방법으로 수행되었습니다.

- **Performance Highlights**: 이 연구는 평가를 통해 기존 고성능 모델들(GPT-4 같은)과의 능력 차이를 명확히 보여줍니다. 일본어 VLM은 특히 일본 문화 및 언어 맥락에서의 이미지 이해 능력에 대한 깊은 분석을 제공합니다. 이 결과는 일본어 VLM 개발에 대한 통찰력을 제공하며, 향후 연구와 개발을 위한 기초 자료로 사용될 수 있습니다. 이 벤치마크와 훈련 코드는 공개적으로 제공되어 일본어 VLM 연구를 더욱 발전시킬 수 있도록 합니다.



### Multi-Image Visual Question Answering for Unsupervised Anomaly Detection (https://arxiv.org/abs/2404.07622)
Comments: 13 pages, 8 figures

- **What's New**: 이 연구에서는 비지도 이상 감지(Unsupervised Anomaly Detection, UAD)를 위해 언어 모델을 접목하여 이상 지도(anomaly maps)의 설명 가능성을 향상시키는 새로운 방법을 제시합니다. 이는 복수 이미지 비주얼 질문 응답(Multi-image Visual Question Answering, VQA) 프레임워크를 사용하여, 원본 이미지, 이상 지도, 가상 건강 이미지(PH reconstruction)를 통합하고, 이를 분석해 의료 전문가가 더 정확한 진단을 내릴 수 있도록 돕는 것을 목표로 합니다. 또한, Knowledge Q-Former 모듈을 새로 도입하여 시각적 정보와 텍스트 지식을 결합하는 능력을 강화하였습니다.

- **Technical Details**: 이 연구에서 개발한 VQA 프레임워크는 여러 이미지 특징 융합 전략을 구현하며, 각각의 이미지 쌍(원본 이미지, 이상 지도, PH 재구성)에 대해 질문을 생성하고 이에 대한 답변을 생성할 수 있습니다. 프레임워크는 Knowledge Q-Former(KQ-Former) 모듈을 통해 시각적 특징을 추출하고, 텍스트 기반 지식을 통합합니다. 여러 실험을 통해 이 프레임워크와 KQ-Former 모듈의 효과가 검증되었습니다.

- **Performance Highlights**: 실험 결과, 본 프레임워크는 다양한 종류의 이미지와 질문에 대해 효과적으로 답변을 생성할 수 있음을 보여줍니다. 특히, 이상 지도를 활용하여 미지의 병리를 감지하는 능력이 향상되었으며, 각각의 기능 융합 전략이 어떻게 시각적 지식을 향상시키는지에 대한 분석도 제공합니다.



### PromptSync: Bridging Domain Gaps in Vision-Language Models through  Class-Aware Prototype Alignment and Discrimination (https://arxiv.org/abs/2404.07520)
Comments: Accepted at CVPR 2024 LIMIT, 12 pages, 8 Tables, 2 Figures

- **What's New**: 이 연구에서는 PromptSync라는 새로운 방법을 소개하여 비전-언어(V-L) 모델을 통해 제로샷 일반화 가능성을 증진합니다. 이 방법은 테스트 샘플마다 프롬프트를 동기화하여 클래스 불균형 문제를 해결하고, 테스트 시간 동안의 프롬프트 조정을 개선합니다. 특히, 클래스별 프로토타입(prototype) 정렬과 대조 학습을 사용하여 프롬프트의 붕괴를 방지하고, 소스 및 테스트 도메인 간의 분배 격차를 효과적으로 해결합니다.

- **Technical Details**: PromptSync는 클래스별 프로토타입 정렬과 평균 클래스 확률을 가중치로 사용하여 테스트 샘플을 조정합니다. 또한, 대조 학습을 통한 프로토타입 차별화를 수행하여 클래스 확률의 정확도를 높입니다. 이는 텍스트 및 비전(branch) 브랜치(branches)에서 동시에 수행됩니다. 테스트 시 프롬프트 표현이 단일 클래스로 붕괴(collapse)되는 것을 방지하는 기하학적 정규화(geometric regularizer) 역할을 하여, 이질적인 도메인 간 일반화를 개선합니다.

- **Performance Highlights**: PromptSync는 도메인 일반화 벤치마크에서 이전 최고 방법보다 전반적인 성능에서 2.33%, 기초에서 새로운 일반화에서 1%, 또는 데이터셋 간 전송에서 2.84%의 향상을 보여주었습니다. 이는 제로샷 일반화(generalization)에서의 효과적인 개선을 입증합니다.



### Structure-aware Fine-tuning for Code Pre-trained Models (https://arxiv.org/abs/2404.07471)
Comments: Accepted by COLING 2024

- **What's New**: 본 논문에서는 코드 사전 학습 모델(Code Pre-trained Models, CodePTMs)을 위한 새로운 구조 인식 미세 조정 방법인 SAT(Structure-aware Fine-tuning)를 제안합니다. SAT는 코드의 구조적 지식을 보다 효과적으로 흡수하고자 하는 새로운 다중 작업 학습 기법을 활용하여 미세 조정 단계에서의 성능을 향상시키는 방법을 도입합니다.

- **Technical Details**: SAT는 추상 구문 트리(Abstract Syntax Tree, AST)를 사용하여 코드의 구조를 파싱하고, 변환 계층(Transformer layer)에서 추출한 주의 점수(attention scores)와 AST 사이의 최단 경로 길이를 사용하여 구조적 지식을 표현합니다. 또한, 구조 손실(structure loss)을 도입하여 학습된 정보와 구조적 지식 사이의 차이를 정량화하고, 이를 다중 작업 최적화 과정에 포함시킵니다.

- **Performance Highlights**: SAT 방법은 코드 요약 및 변환 작업에서 네 가지 사전 학습 모델을 사용하여 평가되었습니다. 실험 결과, SAT는 기존 CodePTMs의 미세 조정 성능을 개선할 뿐만 아니라, 한정된 학습 데이터가 있는 환경에서도 더 큰 성능 향상을 보여주었습니다.



### Transferable and Principled Efficiency for Open-Vocabulary Segmentation (https://arxiv.org/abs/2404.07448)
- **What's New**: 이 논문에서는 고효율 Open-Vocabulary Segmentation (OVS)을 달성하기 위해 'OpenTrans'라는 새로운 기법을 제안합니다. 이 기술은 기존의 대형 시각-언어 기반 모델의 성능을 달성하면서 훨씬 작은 모델과 더 낮은 훈련 비용을 이용합니다.

- **Technical Details**: 저자들은 큰 모델 크기와 높은 훈련 비용의 문제를 해결하기 위해 두 가지 주요 전략을 사용합니다. 첫 번째는 CLIP 이미지 인코더의 크기를 줄이기 위해 이터레이티브 마그니튜드 프루닝(iterative magnitude pruning)을 적용하는 것이며, 두 번째는 선택적으로 레이어를 업데이트하여 훈련 비용을 줄이는 것입니다. 이 두 전략은 전이 가능한 효율성을 보장하고, OVS 프레임워크 간에 추가적인 맞춤화 없이 적용할 수 있습니다.

- **Performance Highlights**: 제안된 'OpenTrans' 방법은 다양한 OVS 벤치마크에서 최소화된 계산 비용으로 높은 세그먼테이션 정확도를 달성하였습니다. 이는 21.2M의 감소된 파라미터와 59.11P의 감소된 훈련 비용(Training FLOPs)으로 수행된 것입니다. 이를 통해 기존 작업들과 비교하여 우수한 mIoU-효율성 균형을 성취하였습니다.



### Behavior Trees Enable Structured Programming of Language Model Agents (https://arxiv.org/abs/2404.07439)
- **What's New**: 이 연구에서는 행동 트리(behavior trees)가 언어 모델(language models) 및 기존의 클래식 AI와 전통적인 프로그래밍을 통합하는 통합 프레임워크를 제공한다고 주장합니다. 또한, Dendron이라는 파이썬 라이브러리를 소개하며, 이를 사용하여 언어 모델 에이전트를 프로그래밍하는 방법을 세 가지 사례 연구를 통해 시연합니다.

- **Technical Details**: Dendron 라이브러리는 언어 모델과 행동 트리를 결합하여 자연어를 사용하는 행동 및 논리 조건을 구현할 수 있게 해줍니다. 이는 프로그래밍적으로 달성하기 어려운 유동적인 행동 및 의사 결정을 가능하게 합니다. 특히, 행동 트리는 반응형 제어 아키텍처에 대해 최적의 모듈성을 보장하는 것으로 이론적으로 증명되었습니다.

- **Performance Highlights**: Dendron을 사용한 개발은 안전을 보장하며 언어 모델을 기반으로 하는 서브트리의 실행에 대한 안전 보장을 가능하게 합니다. 세 가지 사례 연구에서는 Dendron이 어떻게 자연스러운 언어 작업을 쉽게 개발할 수 있는지를 보여줍니다. 첫 번째는 챗봇(agent), 두 번째는 모바일 로봇이나 차량을 사용한 카메라 기반 시설 검사 인프라(agent), 세 번째는 지시 튜닝이나 RLHF를 통해 받지 않은 안전 제약 조건을 충족하는 에이전트입니다.



### Deep Generative Sampling in the Dual Divergence Space: A Data-efficient  & Interpretative Approach for Generative AI (https://arxiv.org/abs/2404.07377)
- **What's New**: 이 연구에서는 다변량 시계열(MVT)을 자연 이미지처럼 생성하는 새로운 문제를 제시합니다. 기존의 이미지 생성 기법을 활용하여 스키조프레니아 환자의 EEG 기록, 기후 변수 등을 이미지로 처리하여 샘플을 생성하는 새로운 접근 방식을 탐색합니다. 특히 샘플 크기가 작은 데이터셋에서도 효과적인 새로운 생성 샘플링 접근법을 소개하며, 이는 정보 이론(information theory)에 근거하여 이중 공간(dual space)에서의 경험적 발산(empirical divergence) 추정을 통해 진행됩니다.

- **Technical Details**: 이 연구의 주된 기술적 방법은 데이터 분포의 KL-발산(KL-divergence)을 이중 형태로 추정하고, 이를 통해 최적화된 1-D 이중 다이버전스 공간에서 직접 생성 샘플링을 수행하는 것입니다. 또한, 데이터 분포와 주변 분포(marginal distribution) 사이의 발산을 추정하기 위해 샘플 복잡성을 크게 줄이는 알고리즘을 제안합니다. 실제 데이터 분포에서 나온 샘플을 이중 공간에서 글로벌하게 나타내며, 국소적(local) 발산 추정을 통해 입력의 미세 조정된 표현을 학습하는 'k-최근접 이웃(k-nearest neighbors, kNNs) 사이의 국소적 발산 추정' 방법도 소개합니다.

- **Performance Highlights**: 실제 데이터셋을 사용한 광범위한 실증 평가를 통해, 제안된 방법이 고전적인 딥러닝 방법론과 비교하여 상당한 우수성을 보여주었습니다. 특히 낮은 샘플 크기에서도 강력한 이론적 보장과 함께 효과적인 샘플 생성이 가능함을 입증하였습니다. 이와 함께, 다양한 실제 세계 데이터셋에서의 효율적인 알고리즘 및 단순하고 직관적인 접근 방식이 높은 성능을 보여줍니다.



### Conformer-1: Robust ASR via Large-Scale Semisupervised Bootstrapping (https://arxiv.org/abs/2404.07341)
- **What's New**: 이 논문에서는 Conformer-1, 광범위한 음성 데이터셋(570k 시간)에서 훈련된 최신의 종단간(Automatic Speech Recognition) ASR 모델을 소개합니다. 이 모델은 대부분 공개적으로 이용 가능한 소스에서 얻은, 노이즈가 많은 학생 훈련(Noisy Student Training)을 통해 훈련되었습니다.

- **Technical Details**: 선택된 음성 데이터는 주로 공개 소스에서 얻어지며, Conformer RNN-T 베이스라인 모델을 사용하여 생성된 의사 라벨(Pseudo-label)로 훈련됩니다. 이 모델은 두 가지 버전, 비동기(Asynchronous) 모델과 실시간(Realtime) 모델로 구현되어 있으며, Transducer 모델 아키텍처를 사용하여 RNN-T 손실로 훈련됩니다.

- **Performance Highlights**: 이 논문에 따르면, 의사 라벨이 포함된 데이터를 사용함으로써 단어 오류율(Word Error Rate, WER)이 비동기 모델에서 11.5%, 실시간 모델에서 24.3% 상대적으로 개선되었습니다. 추가로, 이 모델은 배경 소음에 대해 더욱 강건해졌으며, 다양한 벤치마크에서 최고 수준의 성능을 보였습니다.



### Sandwich attack: Multi-language Mixture Adaptive Attack on LLMs (https://arxiv.org/abs/2404.07242)
- **What's New**: 이 논문은 샌드위치 공격(Sandwich attack)이라는 새로운 유니버설 블랙박스 공격 방식을 소개합니다. 이 공격은 다국어 혼합 공격(multi-language mixture attack)을 이용하여 최신 대형 언어 모델(Large Language Models, LLMs)이 해로운 반응을 생성하도록 조작합니다. 공격자들은 다양한 저자원 언어(low-resource languages)를 사용하여 모델을 조작하고, 이를 통해 안전 메커니즘을 우회하여 해로운 내용을 생성할 수 있습니다.

- **Technical Details**: 샌드위치 공격 기법은 여러 저자원 언어로 질문을 섞어 가운데에 해로운 질문을 숨기는 방식입니다. 이 방법은 구글의 바드(Bard), GPT-3.5-터보(GPT-3.5-Turbo), LLaMA-2-70-B-Chat, GPT-4, 클로드-3-오퍼스(Claude-3-OPUS) 및 제미니 프로(Gemini Pro)와 같은 다양한 최신 모델에서 테스트되었습니다. 이 공격은 다국어 설정에서의 안전 훈련 메커니즘의 취약성을 드러내고, 모델이 다양한 언어의 텍스트에 대해 영어 텍스트보다 덜 강력하게 반응함을 보여줍니다.

- **Performance Highlights**: 실험 결과, 샌드위치 공격은 모델의 안전 메커니즘을 성공적으로 우회하여 부적절하고 해로운 반응을 유도할 수 있음을 확인했습니다. 이러한 공격은 모델이 다양한 언어를 처리하는 능력과 이와 관련된 안전 훈련 메커니즘의 한계를 명확히 드러내며, 향후 연구 및 모델 개선의 필요성을 강조합니다.



### Goal-guided Generative Prompt Injection Attack on Large Language Models (https://arxiv.org/abs/2404.07234)
Comments: 22 pages, 8 figures

- **What's New**: 이 연구에서는 대규모 언어 모델(LLMs)에 대한 저비용 블랙박스 공격 방법을 제안하고 있습니다. 공격의 목표는 깨끗한 텍스트와 적대적 텍스트 간의 조건부 확률의 KL 발산(KL Divergence)을 최대화하는 것으로 재정의되었습니다. 또한, 이 연구는 적대적 텍스트 생성을 위한 새로운 목표 유도 생성적 프롬프트 주입 전략(G2PIA)을 설계했습니다.

- **Technical Details**: 이 연구에서는 적대적 텍스트와 깨끗한 텍스트 사이의 Mahalanobis 거리를 최대화하는 것이 KL 발산을 최대화하는 것과 동등함을 증명했습니다. 조건부 확률이 가우시안 분포를 따를 때, 이 두 최대화 문제가 동등하다는 것을 이론적으로 보여주고 있습니다. G2PIA 전략은 특정 제약 조건을 만족하는 주입 텍스트를 찾아내어 이상적인 공격 효과를 근사적으로 달성합니다.

- **Performance Highlights**: 실험 결과는 7개의 LLM 모델과 4개의 데이터셋에서 이 공격 방법의 효과를 입증했습니다. 이 공격 방법은 쿼리가 필요 없는(query-free) 블랙박스 공격 방법으로, 연산 비용이 낮은 점이 특징입니다.



### Improving Retrieval for RAG based Question Answering Models on Financial  Documents (https://arxiv.org/abs/2404.07221)
- **What's New**: 이 논문에서는 효과적인 대화형 텍스트 생성을 위해 필수적인 Retrieval Augmented Generation (RAG, 검색 증강 생성) 기술의 한계를 탐구하고, 이를 개선하기 위한 다양한 방법론을 제안합니다. 특히, 본 연구는 기존 RAG 방식에서 발생하는 문제인 불필요하거나 부정확한 텍스트 청크의 검색 문제를 해결하고자 합니다.

- **Technical Details**: 본 논문은 향상된 chunking techniques (청크 생성 기술), query expansion (질의 확장), metadata annotations (메타데이터 주석)의 적용, re-ranking algorithms (재랭킹 알고리즘) 사용 및 embedding algorithms (임베딩 알고리즘)의 미세 조정 등을 통해 검색의 정확성을 높이는 방법을 제안합니다. 이러한 기술들은 LLMs에 적합한 콘텍스트를 제공하여 응답의 정확성과 신뢰도를 향상시킬 수 있습니다.

- **Performance Highlights**: 이 연구는 더 정교한 검색 및 청크 생성 기술을 적용함으로써 LLM이 제공하는 답변의 정확도와 관련성이 향상될 수 있음을 시사합니다. 특히, domain-specific (도메인 특화) 정보를 처리할 때 발생할 수 있는 오류를 줄이는 쪽으로 적용 가능성이 높으며, 이는 실제 산업 활동에서 LLM의 활용도를 크게 높일 수 있습니다.



### Blended RAG: Improving RAG (Retriever-Augmented Generation) Accuracy  with Semantic Search and Hybrid Query-Based Retrievers (https://arxiv.org/abs/2404.07220)
Comments: Pre-print version of paper submitted to conference

- **What's New**: 이 논문에서는 'Blended RAG' 방법을 제안하여 문서의 사설 지식 기반과 대규모 언어 모델(LLM)을 결합하는 Retrieval-Augmented Generation (RAG) 접근 방식의 정확도를 향상시키고자 하였습니다. 이 방법은 Dense Vector Index와 Sparse Encoder Index와 같은 의미 검색(semantec search) 기술을 이용하며, 하이브리드 쿼리 전략을 혼합 사용하였습니다.

- **Technical Details**: 이 논문에서는 키워드 기반 유사성 검색, 밀집 벡터 기반 검색, 의미 기반 희소 인코더 검색 등 세 가지 검색 전략을 탐구하고 통합하여 하이브리드 쿼리를 형성했습니다. 또한, 이 연구는 다양한 검색 기술을 체계적으로 평가하여 BM25, KNN, Elastic Learned Sparse Encoder (ELSER)와 같은 주요 인덱스를 사용하였고, 이를 통해 문서 및 쿼리 콘텐츠에서 유도된 벡터 표현의 근접성을 식별하는 데 중점을 두었습니다.

- **Performance Highlights**: Blended Retriever를 포함한 실험에서는 특히 NQ와 TREC-COVID 데이터셋에서 높은 검색 정확도를 보였습니다. 특히, Best Fields를 사용하는 Sparse Encoder를 활용한 하이브리드 쿼리는 NQ 데이터셋에서 88.77%의 높은 검색 정확도를 달성하여 기존의 벤치마크를 상회하는 결과를 보였습니다. 또한 이러한 하이브리드 검색 방식은 SqUAD와 같은 Generative Q&A 데이터셋에 적용할 때도 향상된 결과를 보여, 기존 파인 튜닝 성능을 능가했습니다.



### Exploring the Frontier of Vision-Language Models: A Survey of Current  Methodologies and Future Directions (https://arxiv.org/abs/2404.07214)
Comments: The most extensive and up to date Survey on this topic covering 76 Visual Language Models

- **newsletter**: [{"What's New": 'LLM(Large Language Models)의 발달로 AI 혁명이 변화하고 있지만, 이들 모델은 텍스트 정보만 처리하는 데 제한적입니다. 연구자들은 VLM(Vision-Language Models)을 개발하여 시각적 능력을 통합시켜, 이미지 캡션 생성이나 시각적 질문 응답과 같은 복잡한 작업을 수행할 수 있게 했습니다.'}, {'Technical Details': 'VLM은 시각과 텍스트 정보를 결합해 내용 이해 및 생성에 능숙합니다. 이 모델들은 이미지와 텍스트 인코더(image and text encoder)를 사용하여 임베딩을 생성하고, 이미지-텍스트 융합 레이어(image-text fusion layer)를 통해 융합된 벡터를 LLM에 전달하여 시각적으로 인식 가능한 텍스트를 생성합니다.'}, {'Performance Highlights': 'VLM은 다양한 벤치마크 데이터셋에서의 성능을 검증하였으며, 최신 MME 벤치마크 등에서 상세한 분석을 제공합니다. 이러한 VLM들은 이미지 캡셔닝, 시각적 쿼리 응답, 텍스트 기반 이미지 생성 등에서 뛰어난 능력을 보여줍니다.'}]



### Leave No Context Behind: Efficient Infinite Context Transformers with  Infini-attention (https://arxiv.org/abs/2404.07143)
Comments: 9 pages, 4 figures, 4 tables

- **What's New**: 이 연구는 Transformer 기반의 대형 언어 모델(LLMs: Large Language Models)을 무한히 긴 입력에 대해 메모리 및 계산량의 상한선을 유지하며 확장할 수 있는 효율적인 방법을 소개하고 있습니다. 새로 제안된 'Infini-attention'이라는 주목 기술이 특징으로, 이는 기존의 주목 메커니즘에 압축 메모리를 결합하고 한 개의 Transformer 블록 안에서 가려진(local masked) 주목과 장기(linear) 주목 기술을 통합합니다.

- **Technical Details**: Infini-attention은 기존의 Key-Value(KV) 상태를 포기하지 않고 압축 메모리에 저장한 후, 이후 시퀀스를 처리할 때 주목(query) 상태를 사용하여 메모리에서 값을 검색합니다. 이는 최종 컨텍스트 출력을 계산할 때 장기 메모리에서 검색된 값과 로컬 주목 컨텍스트를 결합하여 집계합니다.

- **Performance Highlights**: 이 방법은 긴 컨텍스트 언어 모델링 벤치마크에서 기준 모델을 능가하며, 메모리 크기 측면에서 114배 더 효율적입니다. 1B LLM은 100만(M) 시퀀스 길이에 적응하여 passkey 검색 작업을 해결하며, 8B 모델은 500K 길이의 책 요약 작업에서 계속된 사전 학습(pre-training)과 작업 미세 조정(task fine-tuning) 후 새로운 SOTA(State of the Art) 결과를 달성합니다.



### Towards Robustness of Text-to-Visualization Translation against Lexical  and Phrasal Variability (https://arxiv.org/abs/2404.07135)
- **What's New**: 이 연구에서는 자연어 질문(Natural Language Questions, NLQ)으로부터 데이터 시각화를 자동 생성하는 Text-to-Vis 작업의 강인성을 철저히 조사합니다. 새로운 강인성 데이터셋인 nvBench-Rob을 구축하고, 기존의 Text-to-Vis 모델들이 이 데이터셋에서 성능이 크게 떨어지는 것을 발견함으로써, 기존 방법들의 강인성이 전반적으로 부족함을 시사합니다. 또한, 입력 변형에 대응하기 위해 Retrieval-Augmented Generation (RAG) 기술을 기반으로 한 새로운 프레임워크 GRED를 제안합니다.

- **Technical Details**: GRED 프레임워크는 NLQ-Retrieval Generator, Visualization Query-Retrieval Retuner 및 Annotation-based Debugger의 세 부분으로 구성되어 있습니다. 이는 각각 자연어 변형, 프로그래밍 스타일 차이 및 데이터 스키마 변형에 대응하기 위해 설계되었습니다. GRED는 사전 훈련된 텍스트 임베딩 모델과 ChatGPT를 사용하여 NLQ와 DVQ(Data Visualization Query)를 처리하고, 이를 기반으로 최종 데이터 시각 요청을 생성합니다.

- **Performance Highlights**: GRED는 nvBench-Rob 데이터셋에서 기존 최고 기술(RGVisNet) 대비 32% 향상된 정확도를 보여주었으며, 이는 GRED 프레임워크의 강인성 향상 능력을 입증합니다. 이 연구는 Text-to-Vis 작업의 강인성 문제에 대한 연구를 촉진하고, 강인한 데이터 시각화 모델 개발의 새로운 패러다임을 제공할 것으로 기대됩니다.



### Continuous Language Model Interpolation for Dynamic and Controllable  Text Generation (https://arxiv.org/abs/2404.07117)
Comments: 20 pages, 22 figures

- **What's New**: 이 논문에서는 사용자의 다양하고 변화하는 선호에 동적으로 적응할 수 있는 대규모 언어 모델(LLM: large language models)의 개발에 주목하고 있습니다. 특히, 다양한 도메인에 걸쳐 특정 생성 특성을 지닌 모델을 '살아 있는' 상태에서 만들어 낼 수 있는 선형 가중치 보간(linear weight interpolation) 방법을 사용하여, 사용자 지정 생성 선호를 표현하는 데 유용한 방법을 제시합니다.

- **Technical Details**: 저자들은 수많은 앵커 모델(anchor models)에 대한 저차원(rank) 업데이트(low-rank updates)를 활용하여 기본 모델을 다양한 도메인에 미세 조정(fine-tune)합니다. 이후, 이 앵커 모델들의 가중치 업데이트를 사용하여 그들의 볼록 선형 조합(convex hull) 내에 포함된 모든 (무한한) 클래스의 모델을 매개 변수화(parametrize)합니다. 선형 보간(linear interpolation)을 통해 다양한 스타일 속성(style attributes)에 대한 모델 출력의 예측 가능하고 일관된 변화를 탐색하여, 보간 가중치를 바꾸어 다양한 사용자 프로필에 맞는 모델을 생성할 수 있음을 보여줍니다.

- **Performance Highlights**: 선형 가중치 보간을 이용할 때, 변화하는 보간 및 평균 가중치가 생성된 텍스트의 각 속성 수준에 예측 가능하고 일관된 반응을 초래한다는 점이 경험적으로 입증되었습니다. 이는 다양한 제어 변수 조합과 그 제어의 강도를 전체적으로 계산적으로 효과적으로 최적화할 수 있음을 의미합니다. 또한, 대부분의 제어 속성 간에는 놀랄 만큼 적은 엉킴(entanglement)이 있으며, 이는 특정 속성 변경이 예상치 않게 다른 속성에 영향을 미치는 문제를 최소화합니다.



### From Model-centered to Human-Centered: Revision Distance as a Metric for  Text Evaluation in LLMs-based Applications (https://arxiv.org/abs/2404.07108)
Comments: 9 pages, 2 figures, under review

- **What's New**: 본 연구는 인공지능(AI) 구동 글쓰기 보조 애플리케이션의 상황에서 대규모 언어 모델(Large Language Models, LLM)의 평가를 사용자 중심으로 전환하며, 새로운 평가 척도인 'Revision Distance'를 제안합니다. 이 척도는 LLM이 생성한 수정 편집을 계산하여 인간의 글쓰기 과정을 모방합니다.

- **Technical Details**: ‘Revision Distance’는 LLM이 제안한 수정 사항을 계산하여 결정되는 메트릭으로, 사용자에게 더 명확하고 상세한 피드백을 제공합니다. 이 메트릭은 기존의 ROUGE, BERT-Score, GPT-Score와 일치하지만 더 상세한 피드백을 제공하며, 텍스트 간의 차이를 더 잘 구분합니다. 또한, 참고 텍스트가 부족한 시나리오에서도 중요한 가능성을 보여줍니다.

- **Performance Highlights**: ‘Revision Distance’ 메트릭은 쉬운 글쓰기 작업에서 기존 메트릭과 일관성이 있으며, 더 도전적인 학술 글쓰기 작업에서도 안정적이고 신뢰할 수 있는 평가 결과를 제공합니다. 참고 없는 시나리오에서는 인간 판단과 약 76% 일치하며, 편집 유형을 분류함으로써 더 세밀한 분석을 제공합니다.



### Graph Chain-of-Thought: Augmenting Large Language Models by Reasoning on  Graphs (https://arxiv.org/abs/2404.07103)
Comments: 21 pages. Code: this https URL

- **What's New**: 새로운 대규모 언어 모델(LLMs, Large Language Models)의 한계, 즉 '환각(hallucinations)' 문제를 해결하기 위해, 기존의 텍스트 데이터만을 활용하는 대신 '그래프 연결된 데이터(graph-attributed data)'를 통합하는 새로운 접근 방법을 제안합니다. 이를 위해 '그래프 추론 벤치마크(Graph Reasoning Benchmark, GRBench)' 데이터셋을 수작업으로 구축하였으며, 이는 10개의 도메인 그래프로부터 지식을 추출하여 1,740개의 질문에 답할 수 있습니다.

- **Technical Details**: 제안된 '그래프 사고 연쇄(Graph Chain-of-thought, Graph-CoT)' 프레임워크는 LLMs를 그래프와 반복적으로 상호 작용하도록 유도합니다. 각각의 Graph-CoT 반복은 세 개의 하위 단계로 이루어진다: LLM의 추론(LLM reasoning), LLM과 그래프 간의 상호작용(LLM-graph interaction), 그리고 그래프 실행(graph execution). 이 구조는 LLMs가 단순히 텍스트 단위의 정보를 처리하는 것을 넘어 그래프 구조상에서의 연결된 정보까지 고려하게 합니다.

- **Performance Highlights**: 실험 결과, Graph-CoT는 기존의 베이스라인 모델들을 일관되게 능가하는 성능을 보였습니다. GRBench 데이터셋에서의 평가에는 세 가지 유형의 LLM 백본(Large Language Model backbones)이 사용되었으며, Graph-CoT의 접근 방식이 특히 지식 집약적인 작업에 우수한 결과를 나타내었습니다.



### Dynamic Generation of Personalities with Large Language Models (https://arxiv.org/abs/2404.07084)
- **What's New**: 이 연구에서는 대규모 언어 모델(Large Language Models, LLMs)이 인간의 심사 숙고를 모방하는 능력을 향상시키는 새로운 접근 방식을 제시합니다. 인간의 심사 숙고는 논리와 성격에 의해 영향을 받는 복합적인 과정입니다. 이번 연구의 핵심은 ‘동적 성격 생성(Dynamic Personality Generation, DPG)’ 기법을 소개하며, 이는 하이퍼네트워크(Hypernetworks) 기반으로 구현되었습니다. 기존의 연구들이 주로 LLM의 논리적 측면에 집중했다면, 이 연구는 성격(Personality) 측면의 탐구를 확장하였습니다.

- **Technical Details**: DPG 기법은 초기에 '빅 파이브 성격 이론(Big Five personality theory)'을 통합하여 GPT-4를 성격 평가 기계로 변형시키는 것에서 시작합니다. 이는 캐릭터들의 대화에서 자동으로 성격 특성을 평가할 수 있게 합니다. 새로운 메트릭(New metric)을 제안하여 성격 생성 능력을 평가하며, 이 평가 방법을 사용하여 대본 데이터 내 대화를 평가하고 성격-대화 데이터셋을 생성합니다. 마지막으로, 이 데이터셋을 사용하여 DPG를 세부 조정(Fine-tune)합니다.

- **Performance Highlights**: 실험 결과, DPG는 전통적인 세부 조정 방법보다 성격 생성 능력이 우수함을 보여줍니다. 특히, 프롬프트 기반의 GPT-4(Prompt-based GPT-4)보다 뛰어난 성능을 나타내며, DPG의 세부 조정은 성격과 대화 사이의 복잡한 상호작용을 더 잘 반영할 수 있는 능력을 갖추도록 합니다.



### Exploring Concept Depth: How Large Language Models Acquire Knowledge at  Different Layers? (https://arxiv.org/abs/2404.07066)
Comments: 12 pages

- **What's New**: 이 논문에서는 큰 언어 모델(Large Language Models, LLMs)의 다양한 계층에서 서로 다른 개념을 학습하는 현상을 연구합니다. 특히, '개념 깊이'(Concept Depth)라는 개념을 도입하여 데이터셋과 모델의 크기에 따라 각기 다른 LLM이 지식을 어떻게 이해하는지 측정합니다. 이는 LLM의 학습 과정과 내부 표현에 대한 이해를 심화시키는 새로운 시도입니다.

- **Technical Details**: 이 연구에서는 factual, emotional, inferential의 세 가지 범주로 개념의 난이도를 정의하고, 이를 기반으로 다양한 계층에서의 개념 학습 상태를 탐색합니다. '개념 깊이'는 LLM의 다양한 계층에서 데이터를 프로빙(probing) 기술을 사용하여 추출하고, 독립적인 선형 분류기(probes)를 사용하여 각 계층의 최적 성능을 평가합니다. 이를 통해 간단한 개념은 얕은 계층에서, 복잡한 개념은 보다 깊은 계층에서 학습됨을 확인할 수 있습니다.

- **Performance Highlights**: 실험 결과, LLMs는 낮은 '개념 깊이'에서 기본 개념을 효과적으로 학습하는 반면, 더 복잡한 개념을 학습하기 위해서는 더 많은 계층의 깊이가 요구되는 일관된 현상을 보여줍니다. 이는 LLMs의 내부 표현의 강건성에 대한 새로운 관점을 제공하며, 임의의 노이즈(random noise)와 같은 방해 요소가 모델의 '개념 깊이'를 더 깊게 만들어 LLM이 개념을 학습하는 위치를 변경시킬 수 있음을 보여줍니다.



### Groundedness in Retrieval-augmented Long-form Generation: An Empirical  Study (https://arxiv.org/abs/2404.07060)
Comments: NAACL 2024 (Findings)

- **What's New**: 이 연구는 장문형 질문 응답(LFQA)에서 검색 증강형 대규모 언어 모델(LLMs)의 근거 있는(groundedness) 문제를 실증적으로 조사하는 데 집중합니다. 모델이 생성한 각 문장이 검색한 문서 또는 모델의 사전 훈련 데이터에서 근거 있는지를 평가합니다. 연구 결과, 정확한 정답이 포함된 문장조차 일관적으로 근거가 없는 경우가 많다는 것을 발견했습니다.

- **Technical Details**: 모델의 근거 있는 생성 여부를 검증하기 위해 groundedness-verification model을 사용하여, 생성된 텍스트가 검색된 문서나 모델의 사전 훈련 코퍼스에 근거를 두고 있는 정보를 포함하는지 여부를 측정합니다. 연구는 세 가지 데이터 세트와 네 개의 모델 패밀리를 분석하여, 대규모 모델이 그들의 출력을 주어진 소스에 더 효과적으로 근거를 두는 경향이 있지만, 가장 큰 모델(Falcon 180B)에서조차도 ground-truth 답변이 포함된 출력의 약 25%가 근거가 없었습니다.

- **Performance Highlights**: 모델 크기, 디코딩 전략, 지도 학습(instruction tuning)의 영향을 조사한 결과, 큰 모델이 일반적으로 주어진 출처에 근거하여 출력을 더 잘 생성하는 경향이 있지만, 정답을 포함하더라도 많은 출력들이 근거가 없는 경우가 여전히 많았습니다. 지도 학습과 빔 서치(Beam Search) 디코딩 전략이 근거 없는 내용 생성을 줄이는 데 도움이 될 수 있는 것으로 관찰되었습니다. 이러한 방법들이 모델이 훈련과 추론 시의 문서를 더 잘 활용하도록 도와, 잘못된 정보의 생성 경향을 완화하는 것으로 보입니다.



### Meta4XNLI: A Crosslingual Parallel Corpus for Metaphor Detection and  Interpretation (https://arxiv.org/abs/2404.07053)
- **What's New**: 이 연구에서 우리는 스페인어와 영어로 된 은유 주석이 포함된 새로운 병렬 데이터셋 Meta4XNLI를 소개합니다. 은유 검출(metaphor detection)과 해석(metaphor interpretation) 작업에 초점을 맞추고, 단일언어(monolingual) 및 교차언어(cross-lingual) 실험을 통해 언어 모델의 은유 이해 능력을 조사합니다.

- **Technical Details**: Meta4XNLI 데이터셋은 은유를 토큰 및 전제-가설 쌍 수준에서 주석 처리하여 데이터의 병렬성과 다양한 도메인의 텍스트를 포함합니다. 연구에서는 은유 검출을 위해 다양한 데이터셋을 사용하여 Masked Language Models (MLM)을 Fine-Tuning하고 평가하였으며, 은유 해석은 자연어 추론(Natural Language Inference, NLI) 작업 내에서 구현되었습니다.

- **Performance Highlights**: 언어 모델은 Meta4XNLI를 활용하여 양방향 번역의 영향을 분석하고, 은유의 교차 언어적 전달 가능성을 탐구하였습니다. 실험 결과는 은유가 포함된 NLI 쌍과 그렇지 않은 쌍을 비교함으로써 모델이 은유적 표현을 이해하는 데 어려움을 겪는지 검토하는 데 도움을 주었습니다.



### A Computational Analysis of the Dehumanisation of Migrants from Syria  and Ukraine in Slovene News Media (https://arxiv.org/abs/2404.07036)
Comments: The first authors have contributted equally. Accepted at LREC-COLING

- **What's New**: 이 연구는 슬로베니아 언론에서 나타난 이민에 대한 태도 변화를 컴퓨터 언어학 기법을 사용하여 분석합니다. 특히, 시리아 전쟁(2015-16)과 우크라이나 전쟁(2022-23) 이후의 기간 동안 발행된 뉴스 기사를 통해 이민에 대한 접근 방식 변화를 조사합니다. 이 연구는 다른 언어로의 전환 및 평가가 용이하도록 최근 제안된 방법을 적용하는 동시에 새로운 감정 자원의 사용과 통계적 유의성 검사 방식을 소개합니다.

- **Technical Details**: 저자들은 zero-shot cross-lingual (제로샷 크로스링귤) valence와 arousal 감지 기법, 그리고 새로운 통계적 유의성 평가 방법인 Kolmogorov-Smirnov 테스트를 사용합니다. 이를 통해 큰 데이터 세트에 훈련된 대규모 언어 모델(Large Language Models, LLMs)을 활용하여, 저자원 언어 환경에서도 강력한 언어 이해 및 전이 학습이 가능하다는 점을 보여줍니다. 이 연구는 특히 유용한 도구 및 모델들을 개발하여 슬로베니아어로 분석을 수행했습니다.

- **Performance Highlights**: 분석 결과, 이민에 대한 태도는 시간이 지남에 따라 더 부정적이고 강렬하게 변화했지만, 우크라이나 이민자에 대해서는 다른 이민자들에 비해 덜 비인간화하는 언어가 사용되었다는 것을 발견했습니다. 이는 특히 유럽연합(EU) 국민과 우크라이나 사람들 간의 높은 유사성 인식 때문일 수 있습니다. 연구는 또한 이민에 대한 담론에서 ‘우리’와 ‘그들’의 이분법적 표현이 일반적이며, 이주민들을 위협으로 묘사하는 경우가 많다고 지적합니다.



### Improving Language Model Reasoning with Self-motivated Learning (https://arxiv.org/abs/2404.07017)
Comments: Accepted at LREC-COLING 2024

- **What's New**: 이 연구에서는 모델 자체가 기존 데이터셋을 활용하여 다양한 수준의 이성적 근거(rationale)를 생성하도록 동기를 부여하는 새로운 방법론, '자기동기 학습(Self-motivated Learning)'을 제안합니다. 이 접근 방식은 높은 주석 비용 때문에 부족한 고품질의 근거 데이터셋 문제를 해결하고자 합니다.

- **Technical Details**: 자기동기 학습 프레임워크는 모델이 생성한 근거들의 랭크를 기반으로 보상 모델(reward model)을 훈련시켜, 이를 통해 재강화 학습(reinforcement learning)을 사용하여 추론 능력을 향상시킵니다. 이 프로세스는 정답 생성 및 이에 대한 근거(rationale) 제공을 요구하는 문제에 대해 도출된 근거의 질을 평가하고, 효과적인 근거를 선별하기 위해 사용됩니다.

- **Performance Highlights**: Llama2 7B 모델을 사용한 실험 결과에서는 본 방법론이 여러 추론 데이터셋에서 모델의 추론 능력을 유의미하게 향상시켰으며, 일부 데이터셋에서는 text-davinci-002 모델보다 더 뛰어난 성능을 보였습니다.



### A Mathematical Theory for Learning Semantic Languages by Abstract  Learners (https://arxiv.org/abs/2404.07009)
Comments: Submitted to ISIT 2024 on Jan. 28, 2024

- **What's New**: LLM과 같은 초대형 언어 모델은 범용 NLP 태스크를 대규모 데이터 및 모델 파라미터 수 증가로 성능이 향상되는 현상을 보여주었습니다. 특별히, 본 논문에서는 스킬-텍스트 이분 그래프(skill-text bipartite graph)와 LDPC 코드, Irregular Repetition Slotted ALOHA (IRSA) 기법을 이용한 반복 디코딩 프로세스를 통해 어떻게 학습된 스킬의 출현이 일어나는지 수학 이론을 개발하였습니다.

- **Technical Details**: 학습 과정을 LDPC 코드와 IRSA 기법을 사용하는 반복적 디코딩 과정으로 모델링하고, 밀도 진화 분석(density evolution analysis)을 통해 스킬과 텍스트 사이즈 비율이 특정 임계값을 초과할 때 학습된 스킬이 출현하는 것을 보여줍니다. 이를 통해 테스트 오류가 급격히 감소함을 관찰할 수 있었습니다.

- **Performance Highlights**: 장벽 값을 초과하는 비율 R에서 학습된 스킬이 나타날 확률이 급격히 증가하며, 이에 따라 테스트 중 오류율이 현저하게 감소함을 확인하였습니다. 또한 학습이 끝난 후, 스킬 색인을 사용한 의미론적 텍스트 압축 방법을 제안하여 의미론적 통신(semantic communication)에 활용될 수 있는 방안을 제시합니다.



### LM Transparency Tool: Interactive Tool for Analyzing Transformer  Language Models (https://arxiv.org/abs/2404.07004)
- **What's New**: 새롭게 소개된 LM Transparency Tool (LM-TT)은 Transformer 기반 언어 모델의 내부 작동을 분석하기 위한 오픈 소스 인터랙티브 툴킷입니다. 기존 도구들이 의사 결정 과정의 특정 부분에만 초점을 맞춘 것과 달리, LM-TT는 전체 예측 과정을 투명하게 만들고 모델 동작을 매우 세밀한 부분까지 추적할 수 있도록 설계되었습니다.

- **Technical Details**: LM-TT는 중요한 입력에서 출력까지의 정보 흐름을 보여주고, 모델 블록에 의해 수행된 변경 사항을 개별 attention heads와 feed-forward (피드포워드) 뉴런에게 귀속시킬 수 있습니다. 또한, 이들 head나 뉴런의 기능을 해석할 수 있습니다. 이 도구는 중요한 모델 구성 요소의 중요성을 각 단계에서 보여주는 것이 핵심입니다. LM-TT는 Ferrando와 Voita (2024)에 의존하여 일반적인 패치 기반 분석 도구보다 100배 빠릅니다.

- **Performance Highlights**: 이 도구는 중요한 예측 과정과 모델 구성 요소의 중요성을 다양한 정밀도로 시각화합니다. 사용자 인터페이스를 통한 인터랙티브 탐색이 가능하며, large models (대규모 모델)에서 검사해야 할 부분을 알아내는 데 필수적입니다.



### Event Grounded Criminal Court View Generation withCooperative (Large)  Language Models (https://arxiv.org/abs/2404.07001)
- **What's New**: 법률 지능의 발전과 함께 형사 법정 의견 생성(Criminal Court View Generation) 작업이 주요한 연구 주제로 부상했습니다. 이 연구에서는 법률 사건 사실을 요약하고 판결에 대한 설명을 제공하는 간결하고 일관된 텍스트를 생성하는 것을 목표로 합니다. 기존 연구들은 대체로 사건 사실을 광범위하게 구분하여 법원 의견을 생성했지만, 이러한 접근 방식은 사건 사실의 복잡한 세부 정보를 포착하는 데 한계가 있었습니다. 이에 대응하여, 본 논문에서는 범죄 법정 의견 생성을 위한 Event Grounded Generation (EGG) 방법을 제안합니다. 이 방법은 섬세한 이벤트 정보를 도입하여 의견 생성을 향상시키고자 하는 새로운 접근법입니다.

- **Technical Details**: EGG 방법은 범죄 사실에서 이벤트를 추출하고 이를 법정 의견 생성에 활용하는 방식을 채택합니다. 우선, LLMs (Large Language Models)을 기반으로 한 추출 방법을 설계하여 대량의 주석이 달린 이벤트 없이도 사건 사실에서 이벤트를 추출할 수 있습니다. 그 후, 추출된 이벤트를 취합하여 법정 의견 생성에 활용합니다. 또한, 추출 단계에서 LLMs를 사용함으로써 발생하는 계산 부하를 고려하여, 추론 단계에서 LLMs를 사용하지 않는 EGG방식(EGG_free)도 제안하여, 이벤트 추출의 필요성 없이 사건 사실만으로 법정 의견을 생성할 수 있도록 합니다.

- **Performance Highlights**: 실제 데이터셋에 대한 광범위한 실험 결과는 제안된 방법의 효과를 명확히 입증합니다. 특히, EGG 방법은 기존의 모델들에 비해 더욱 정밀한 법원의 견해를 생성할 수 있음을 확인했습니다. 뿐만 아니라, LLMs 없이도 이벤트 정보를 활용하여 의견을 생성할 수 있는 EGG_free 방식은 법률 전문가뿐만 아니라 법률 지식이 없는 일반인들에게도 실용적으로 적용될 수 있는 장점을 가지고 있습니다.



### XNLIeu: a dataset for cross-lingual NLI in Basqu (https://arxiv.org/abs/2404.06996)
Comments: Accepted to NAACL 2024

- **What's New**: 이 연구는 자연 언어 추론(Natural Language Inference, NLI) 벤치마크인 XNLI를 바스크어로 확장하여 새로운 데이터셋인 XNLIeu를 개발하였습니다. 바스크어가 저자원 언어(low-resource language)임에도 불구하고, 기계 번역(machine translation)과 전문 번역가에 의한 수작업 후보정(manual post-editing)을 통해 데이터셋을 구축했습니다. 또한, 다양한 언어 모델을 이용하여 NLI 작업을 수행함으로써, 바스크어에 대한 자연 언어 이해(Natural Language Understanding, NLU) 연구를 촉진시키는 것을 목표로 합니다.

- **Technical Details**: 이 연구는 영어 XNLI 코퍼스를 기계 번역을 통해 바스크어로 변환한 후, 전문 번역가가 후보정 작업을 거친 XNLIeu 데이터셋을 개발하는 과정을 설명합니다. 연구팀은 단일 언어 및 다언어 언어 모델(Language Models, LLMs)을 사용하여 여러 가지 크로스-링구얼(cross-lingual) 전략을 실험했습니다. 또한, 번역에 의해 구축된 데이터와 원어민이 직접 작성한 데이터(Native test set)의 성능 차이를 비교 분석하였습니다.

- **Performance Highlights**: 수정된 번역(XNLIeu)은 기계 번역만 사용된 데이터셋(XNLIeuMT)보다 일관되게 더 나은 결과를 가져왔습니다. 뿐만 아니라, 이 연구는 번역 후 훈련(translate-train)이라는 크로스-링구얼 전략이 바스크어 NLI 작업에 대해서 가장 효과적인 전략임을 보여주었습니다. 그러나 원어민 데이터셋을 사용할 때 이득이 줄어드는 현상도 관찰되었습니다.



### Hybrid Multi-stage Decoding for Few-shot NER with Entity-aware  Contrastive Learning (https://arxiv.org/abs/2404.06970)
- **What's New**: 이 논문에서는 몇 가지 예제를 통해 새로운 유형의 명명된 개체를 식별할 수 있는 소수샷(named entity recognition)을 개선하기 위해 Hybrid Multi-stage Decoding for Few-shot NER with Entity-aware Contrastive Learning (MsFNER)을 제안합니다. MsFNER은 일반 NER을 두 단계로 나누며: 개체 범위 탐지(entity-span detection)와 개체 분류(entity classification).

- **Technical Details**: MsFNER는 메타 학습(meta-learning), 모델-무관 메타 학습(Model-Agnostic Meta-Learning, MAML), 및 개체 인식 대조 학습(entity-aware contrastive learning)을 사용하여 원천 도메인(source domain)의 지원 데이터셋에서 개체 범위 탐지 모델 및 개체 분류 모델을 별도로 학습합니다. 타겟 도메인(target domain)의 지원 데이터셋에서 모델을 미세 조정(finetuning)하고, 추론 과정에서는 KNN과 결합하여 미확인 데이터에 대한 개체 유형을 예측합니다.

- **Performance Highlights**: 오픈 FewNERD 데이터셋에서 실험을 수행하였고, MsFNER이 기존의 소수샷 NER 모델들보다 우수한 성능을 보여주었습니다. 이는 개체 범위 탐지 및 분류 과정의 효율성 향상, 그리고 대조적 학습을 통한 개체 표현의 강화가 주요 기여로 작용하였습니다.



### Charles Translator: A Machine Translation System between Ukrainian and  Czech (https://arxiv.org/abs/2404.06964)
- **What's New**: Charles Translator는 우크라이나어와 체코어 간의 기계 번역 시스템으로, 러시아-우크라이나 전쟁에 따른 개인 및 사회적 영향을 완화하기 위해 개발되었습니다. 이 시스템은 다른 기존 번역 시스템들이 영어를 중간 언어(pivot language)으로 사용하는 것과 달리, 두 언어의 언어학적 유사성을 활용하여 직접 번역(direct translation)하는 점을 특징으로 합니다.

- **Technical Details**: Charles Translator는 Transformer architecture (Vaswani et al., 2017)와 iterated block back-translation (Popel et al., 2020)을 사용하여 단일언어 훈련 데이터(monolingual training data)를 효율적으로 사용합니다. 초기 훈련 데이터를 수집하는 과정은 다양한 자원봉사자들의 도움을 받아 진행되었으며, 체코-우크라이나 번역가들과 협력하여 양질의 병렬 데이터(parallel data)를 확보했습니다.

- **Performance Highlights**: 이 시스템은 체코어에서 우크라이나어로 번역할 때, 그리고 그 반대의 경우에도 문맥적 의미와 문법적 정확성을 유지하며 특히 성과 공손함(gender and politeness)의 문법 범주에서 정보 손실을 최소화합니다. 또한, EUBookshop, GNOME, KDE4 등의 데이터 소스를 사용했으며, 실제로 난민들과 체코 개인 및 당국 간의 일상 대화에서 시스템 성능을 평가하기 위한 두 개의 테스트 세트를 생성했습니다. 이는 WMT22 및 WMT23에서 공개되었습니다.



### Accelerating Inference in Large Language Models with a Unified Layer  Skipping Strategy (https://arxiv.org/abs/2404.06954)
Comments: 12 pages, codes at this https URL

- **What's New**: 이 연구에서는 대규모 언어 모델(LLMs)의 추론 속도를 향상시키기 위한 새로운 동적 계산 방법을 제안합니다. 기존 방법들과 달리, Unified Layer Skipping 전략은 목표 속도 향상 비율(target speedup ratio)에 따라 건너뛸 계산층의 수를 단순히 결정하고, 중간 계층(layer)들을 균형 있게 스킵합니다. 이 방식은 입력 샘플에 독립적이어서 batch decoding과 KV caching과 같은 인기 있는 가속 기술을 자연스럽게 지원합니다.

- **Technical Details**: Unified Layer Skipping 전략은 계층(layer) 간의 균형을 유지하면서 필요한 계층만큼 건너뛰어 성능 저하를 최소화하는 방식입니다. 이 전략은 각 샘플에 대해 일관된 계산 예산을 제공함으로써 안정적이고 예측 가능한 가속 효과를 보장합니다. 또한, 이 전략은 입력 데이터와 독립적으로 작동하므로, 다방면에서 활용 가능성이 높습니다.

- **Performance Highlights**: 기계 번역(machine translation)과 텍스트 요약(text summarization)의 두 가지 공통 작업에 대한 실험 결과에 따르면, Unified Layer Skipping 전략은 기존의 동적 접근 방식보다 추론 성능과 모델 처리량을 현저히 향상시킵니다. 이 방식으로는 처리량이 기존 방법보다 약 30%에서 70% 향상되었으며, 동일한 속도 향상 효과에서 최소한의 성능 손실을 보장합니다.



### MetaCheckGPT -- A Multi-task Hallucination Detector Using LLM  Uncertainty and Meta-models (https://arxiv.org/abs/2404.06948)
Comments: Entry for SemEval-2024 Shared Task 6: SHROOM, a Shared-task on Hallucinations and Related Observable Overgeneration Mistakes

- **What's New**: 본 논문은 SHROOM(Semeval 2024 Task 6)에서 대규모 언어 모델(Large Language Models, LLMs)에서 발생하는 환각 현상을 탐지하고 평가하기 위한 메타-회귀 분석(meta-regressor) 프레임워크를 제안하는 연구로, '모델 무관' 및 '모델 인식' 두 부문에서 각각 1위와 2위를 차지했습니다. 이 프레임워크는 다양한 LLM의 성능을 총체적으로 평가하고 통합하여, 환각 탐지를 위한 기존의 방법보다 더 robust하게 환각을 탐지할 수 있다는 특징을 가진다는 점에서 혁신적입니다.

- **Technical Details**: 이 메타-모델은 여러 LLM과 전문가 평가자 패널의 입력을 통합하는 방식으로 작동합니다. LLM이 생성한 문장을 '이상 없음'으로 분류할 확률을 기반으로 해당 문장에 대한 환각 여부를 평가하며, 이것은 환각 검출을 위한 기존의 통계적 방법이나 단일 모델 의존 방식보다 진보된 접근 방식입니다. 더불어, 이 연구에서는 ChatGPT, Vectara 같은 트랜스포머 기반(Transformer-based) 모델 및 블랙 박스(Black Box) 방식을 실험적으로 탐구하였으며, GPT4와 비교 분석을 통해 한계점을 도출했습니다.

- **Performance Highlights**: SSD(Service Sharing Descriptor) AI 뉴스레터에서는 다양한 diagnostic 지표를 사용하여 메타-모델의 성능을 평가하고 있습니다. 이 메타-레그레서는 정확도(Precision), 재현율(Recall), F1 점수 및 혼동 행렬(Confusion Matrix)을 기반으로 성능을 평가받았으며, 향상된 정확도와 강화된 오류 분석을 통해 GPT4와 같은 기존 모델들이 놓칠 수 있는 환각을 감지할 수 있는 능력을 갖추었습니다.



### GoEX: Perspectives and Designs Towards a Runtime for Autonomous LLM  Applications (https://arxiv.org/abs/2404.06921)
- **What's New**: 이 논문은 대규모 언어 모델(LLMs)이 단순히 대화 시스템에서 정보를 제공하는 역할을 넘어 실제 월드 애플리케이션과 서비스에서 도구를 활용하고 작업을 수행하는 방향으로 발전하고 있음을 조명합니다. 특히, LLM이 생성한 출력의 정확성을 사후에 검증하는 '사후 검증(post-facto validation)' 시스템을 도입하여 인간이 제한적으로만 관여하는 새로운 환경을 제안합니다. 이를 위해, GoEX라는 오픈 소스 실행 엔진을 개발하여 LLM의 자율적인 상호작용을 가능하게 하며, 이는 실제 애플리케이션에 LLM을 통합하는 것에 대한 새로운 접근법을 제시합니다.

- **Technical Details**: LLM 기반 에이전트가 사용자의 의도와 일치하는지 확인하기 위해 '사후 검증' 접근 방식을 도입합니다. 이는 과정이나 중간 출력을 판단하는 '사전 검증(pre-facto validation)'과 대비됩니다. 이러한 방식에서, LLM이 수행한 행동의 결과만을 평가하게 됩니다. 특히 '실행 취소(undo)' 기능과 '손상 제한(damage confinement)' 전략을 통해, 사용자는 LLM이 수행한 동작을 취소하거나 잠재적 위험을 제한할 수 있습니다. GoEX는 LLM 생성 작업을 안전하게 실행하며 안전과 유틸리티 사이의 균형을 맞추도록 설계되었습니다.

- **Performance Highlights**: 이 논문은 LLM이 생성한 행동을 실행하고 그 결과를 사후에 검증하는 아이디어를 통해, LLM과 애플리케이션 간의 상호작용을 위한 새로운 모델을 제시합니다. GoEX는 실제 애플리케이션에서 LLM의 독립적인 실행을 지원함으로써, 기존 시스템과의 통합에 있어 중요한 발전을 이루었습니다. 본 연구는 LLM을 활용한 마이크로서비스, 애플리케이션 등의 미래에 대한 기반을 마련하고 있습니다.



### GraSAME: Injecting Token-Level Structural Information to Pretrained  Language Models via Graph-guided Self-Attention Mechanism (https://arxiv.org/abs/2404.06911)
Comments: NAACL 2024 Findings

- **What's New**: 새로운 그래프 유도 자기 주의 메커니즘인 GraSAME을 제안하여 사전 훈련된 언어 모델(PLMs)이 텍스트와 그래프 구조 간의 모달리티 간격을 효율적으로 연결할 수 있도록 합니다. GraSAME은 추가적인 정렬이나 연결 없이 토큰 수준의 구조 정보를 자연스럽게 통합합니다.

- **Technical Details**: GraSAME은 기존 Transformer 구조의 자기 주의 레이어를 대체하여, GNN에서 학습된 토큰 수준 그래프 정보를 PLM의 텍스트 표현과 매끄럽게 통합할 수 있도록 설계되었습니다. 이는 T5 모델을 기반으로 하여, 인코더-디코더(encoder-decoder) 구조에서 KG-to-text 생성 작업에 적용됩니다.

- **Performance Highlights**: GraSAME은 기존 베이스라인 모델을 뛰어넘는 성능을 보여주며, 상태-of-the-art(SOTA) 모델과 비교할 수 있는 결과를 WebNLG 데이터셋에서 달성했습니다. 또한, SOTA 모델에 비해 100백만 개 이상의 훈련 가능한 파라미터를 줄였습니다.



### Superposition Prompting: Improving and Accelerating Retrieval-Augmented  Generation (https://arxiv.org/abs/2404.06910)
- **What's New**: 이 연구에서는 대규모 언어 모델(Large Language Models, LLMs)이 긴 맥락을 처리하며 겪는 문제점들, 예를 들어 '주의산만 현상'과 처리 비용의 급격한 증가를 해결하기 위해 새로운 RAG(Retrieval-Augmented Generation) 프롬프팅 방법론인 'superposition prompting'을 제안합니다. 이 방법은 기존에 훈련된 트랜스포머 기반 LLMs에 직접 적용할 수 있으며, 추가적인 훈련이나 튜닝 없이도 효율성과 정확성을 동시에 향상시킬 수 있습니다.

- **Technical Details**: superposition prompting은 입력 문서를 병렬 프롬프트 경로로 처리하여 관련 없는 경로를 버림으로써 LLM이 더 효율적으로 정보를 처리할 수 있게 합니다. 이는 양자역학의 '경로 적분(path integral)' 이론에서 영감을 받아, 가능한 '토큰 경로(token trajectories)'의 가중합으로 언어 동역학을 모델링합니다. 추가적으로, 프롬프트 경로 캐싱과 병렬 처리를 통해 인퍼런스 속도를 더욱 빠르게 할 수 있는 기술적 최적화를 제안합니다.

- **Performance Highlights**: 이 방법을 사용함으로써, NaturalQuestions-Open 데이터셋에서 MPT-7B 모델에 대해 93배의 계산 시간 감소와 43%의 정확도 향상을 달성했습니다. 다양한 질의응답(question-answering) 벤치마크에서도 시간 효율성 측면에서 우수한 성능을 입증했습니다.



### Control-DAG: Constrained Decoding for Non-Autoregressive Directed  Acyclic T5 using Weighted Finite State Automata (https://arxiv.org/abs/2404.06854)
Comments: 11 pages. NAACL 2024

- **What's New**: Control-DAG는 기존의 자연어 생성(Natural Language Generation, NLG)의 문제점을 개선하기 위해 새로 도입된 알고리즘으로, Directed Acyclic Transformer (DAT) 모델을 활용한 빠르고 효율적인 비자동 회귀(non-autoregressive, NAR) 텍스트 생성 방식을 제시합니다. 주요한 기능으로는 어휘제어(lexical control), 어휘집 제어(vocabulary control) 및 길이 제어(length control)를 통해 텍스트 생성의 질을 향상시키는 것입니다.

- **Technical Details**: Control-DAG는 Directed Acyclic T5 (DA-T5) 모델에 적용되며, 입력된 지향성 비순환 그래프(Directed Acyclic Graph, DAG)를 가중 유한 상태 기계(Weighted Finite State Automata, WFSA)로 변환 후, 제약이 있는 복수의 유한 상태 기계(Finite State Automata, FSA)와의 교차를 통해 정확도 높은 결과를 도출합니다. 이를 통해 OOV(Out-Of-Vocabulary) 에러를 배제하고, 지정된 엔티티 이름을 충실히 생성할 수 있습니다.

- **Performance Highlights**: Control-DAG는 기존 NAR 모델의 한계를 극복하고, 스키마 유도 대화(Schema Guided Dialogue, SGD) 및 데이터 기록 변환(DART) 데이터셋에서 강력한 성능을 보여주었습니다. 이를 통해 Control-DAG는 기존의 자동 회귀(auto-regressive, AR) 기반 모델과 비교할 때 상응하거나 우월한 성능을 보이며, 특히 엔티티명 생성과 어휘 제어에서 높은 정확도를 보입니다.



### Simpler becomes Harder: Do LLMs Exhibit a Coherent Behavior on  Simplified Corpora? (https://arxiv.org/abs/2404.06838)
Comments: Published at DeTermIt! Workshop at LREC-COLING 2024

- **What's New**: 이 논문은 변환된 텍스트가 원본 텍스트와 동일한 정서나 주제를 유지하는지를 평가하며, 결과는 상당한 일관성 문제를 드러낸다. 이러한 문제가 조속히 해결되지 않을 경우, 간단 해지된 입력값은 효과적으로 제로-이터레이션(zero-iteration) 모델-불특정(model-agnostic) 적대적 공격(adversarial attacks)을 생성하는 데 사용될 수 있음을 경고한다.

- **Technical Details**: 연구팀은 Hugging Face 모델 허브에서 게시된 다양한 사전 학습된 분류 모델들을 사용하여, 원래 텍스트와 간단 해진 텍스트 간의 일관성을 검사한다. 이들은 주제, 정서, 가짜 뉴스/독성 및 감정 예측과 같은 다양한 분류 작업을 테스트했다. 실험에 사용된 모델로는 BERT, OpenAI의 GPT 3.5 등이 포함되었다.

- **Performance Highlights**: 결과는 모든 언어와 모델에 걸쳐 충격적인 불일치를 보여준다. 모델은 최대 50%의 샘플에서 예측을 변경했으며, 이는 심각한 문제를 제기한다. 예를 들어, 'conference'를 'science meeting'으로 대체하는 단순화된 버전에도 불구하고, 사전 훈련된 감정 분류기는 두 샘플에 대해 다른 레이블을 할당한다.



### Does Mapo Tofu Contain Coffee? Probing LLMs for Food-related Cultural  Knowledg (https://arxiv.org/abs/2404.06833)
Comments: 20 pages,8 figures

- **What's New**: 이 논문은 음식 분야 중심의 다국어 데이터셋 FmLAMA를 소개하며, 큰 언어 모델들(Large Language Models, LLMs)이 어떻게 문화적 지식을 인코딩하고 접근하는지를 평가합니다. 특히, 미국 외의 다양한 문화적 맥락에서의 음식 지식을 좀 더 잘 이해하고 반영할 수 있는 방법론을 제안함으로써, 문화적 편향을 줄이고 모델 성능을 향상시키는 것을 목표로 합니다.

- **Technical Details**: FmLAMA는 다양한 문화와 언어에 대한 음식 지식을 포함하는 다국어 데이터셋입니다. 이 데이터셋은 SPARQL을 이용해 Wikidata에서 음식 관련 데이터를 추출하여 구성되었습니다. 또한, 연구팀은 모노링구얼 및 멀티링구얼 설정에서 LLMs의 성능을 평가하고, 여섯 가지 다른 언어를 사용한 템플릿을 활용하여 언어 특이적 및 문화적 지식의 상호작용을 조사했습니다.

- **Performance Highlights**: 연구 결과, LLMs는 미국에서 흔히 접할 수 있는 음식 지식에 대한 편향이 강하게 나타났으며(Language Model), 문화적 맥락을 적절히 통합하면 LLMs가 문화적 지식을 접근하는 능력이 상당히 향상됨을 보여주었습니다. 또한, 문화적 뉘앙스를 포착하는 LLMs의 효과는 프로빙 언어(Probing Language), 특정 모델 아키텍처(Model Architecture), 그리고 해당 문화적 맥락 간의 상호작용에 크게 의존함을 발견하였습니다.



### Emotion-cause pair extraction method based on multi-granularity  information and multi-module interaction (https://arxiv.org/abs/2404.06812)
- **What's New**: 이번 연구에서는 감정-원인 쌍 추출(ECPE) 작업을 위해 GRU, 지식 그래프, 트랜스포머 모듈 간의 상호 작용(Interaction)을 기반으로 한 종단간 멀티태스킹 모델(MM-ECPE)을 제안합니다. 또한, 감정 절과 원인 절 간의 불균형 분포를 해결하기 위해 BERT, 감정 어휘사전, 위치 인식 상호작용 모듈을 기반으로 하는 새로운 인코딩 모델 MM-ECPE(BERT)를 도입하였습니다.

- **Technical Details**: MM-ECPE 모델은 다수준 공유 모듈을 통해 다른 태스크 간의 상호 작용을 모델링하고, 감정-원인 쌍 추출과 감정 추출 및 원인 추출 간의 공유 정보를 채굴합니다. MM-ECPE(BERT) 모델은 감정 및 원인 절의 위치 인식 인코딩 모듈을 사용하여 샘플의 불균형 분포에 대한 영향을 완화하고, 지식 그래프를 이용하여 적합한 라벨을 선별하고, 문맥 인과성(Contextual Causality)을 찾기 위해 감정과 원인 절 간의 일관성을 고려하여 태스크 간 상호 작용을 수행합니다.

- **Performance Highlights**: 실험 결과 MM-ECPE 모델과 MM-ECPE(BERT) 모델 모두 기존 모델들을 능가하는 성능을 보여주었습니다. 특히 위치 불균형 샘플에 대해서 탁월한 성능을 보였으며, 이는 공간 위치 모델링(Position Modeling)과 추출 기능이 강화된 결과로 분석됩니다. ECPE 벤치마크 데이터셋에서의 평가도 우수한 결과를 확인하였습니다.



### Not All Contexts Are Equal: Teaching LLMs Credibility-aware Generation (https://arxiv.org/abs/2404.06809)
Comments: Our code, benchmark, and models are available at this https URL

- **What's New**: 이 논문은 정확한 정보의 중요성을 인식하고, 기존의 Retrieval-Augmented Generation (RAG)의 한계를 극복하기 위해 Credibility-aware Generation (CAG) 프레임워크를 제안합니다. CAG는 외부에서 검색된 문서의 신뢰도를 평가하고 이를 기반으로 정보를 처리하여 결과의 신뢰성과 정확성을 향상시키는 새로운 접근 방법을 탐구합니다.

- **Technical Details**: CAG는 문서 및 문장 수준에서 다양한 신뢰성을 할당하는 '다중 정밀도 신뢰성 주석(multi-granularity credibility annotation)'과 '신뢰성 안내 설명 생성(credibility-guided explanation generation)'을 포함하는 데이터 변환 프레임워크를 활용합니다. 이 프레임워크는 기존의 질의응답(QA) 및 대화 데이터셋을 활용하여 신뢰도가 포함된 형식으로 변형시키며, 모델이 신뢰성을 기반으로 응답을 생성하도록 합니다.

- **Performance Highlights**: CAG 모델은 신뢰성 정보를 활용하여 응답을 생성하는 능력을 입증함으로써, 기존의 RAG 기반 전략과 비교하여 눈에 띄게 성능이 우수함을 보여줍니다. 또한, 잡음이 많은 문서에 대한 강력한 내성을 유지하면서도 높은 성능 수준을 유지하였습니다. 실제 세계 시나리오를 포함하는 포괄적인 벤치마크를 구축하여 모델의 신뢰성 인식 생성 능력을 철저히 평가하였습니다.



### Personality-aware Student Simulation for Conversational Intelligent  Tutoring Systems (https://arxiv.org/abs/2404.06762)
- **What's New**: 이 연구는 대화형 지능형 교수 시스템(Intelligent Tutoring Systems, ITS)에서 개별 학생들의 인격 특성을 반영하고 시뮬레이션하는 새로운 프레임워크를 제안합니다. 이 프레임워크는 학생들의 인지적 및 비인지적 측면을 다루며, 대규모 언어 모델(Large Language Models, LLMs)을 활용하여 언어 학습 시나리오에서 개성을 인식하는 학생 시뮬레이션을 가능하게 합니다.

- **Technical Details**: 개발된 체계는 'Big Five' 이론을 기반으로하여 다섯 가지 성격 유형을 조정 및 구성하고, 이를 학생 시뮬레이션 지침에 통합합니다. LLMs는 주어진 언어 능력과 성격 특성에 따라 다양한 학생 반응을 생성할 수 있도록 지시를 따를 수 있습니다. 또한, 다면적 검증(multi-aspect validation)을 통해 생성된 대화의 교육적 영향을 평가합니다.

- **Performance Highlights**: 실험 결과, LLM을 기반으로 한 교수 시스템은 다양한 성격 특성에 맞추어 교수 전략(scaffolding strategies)을 적응시킬 수 있음을 보여줍니다. 이 시뮬레이션은 판단력이 높은 Big Five 이론과 높은 상관 관계를 보이며, 학습 데이터 확장과 개인화된 대화형 ITS의 평가에 새로운 방안을 제시합니다.



### DiffusionDialog: A Diffusion Model for Diverse Dialog Generation with  Latent Spac (https://arxiv.org/abs/2404.06760)
Comments: LREC-COLING 2024 camera ready

- **What's New**: 이 논문에서는 대화 생성(diversity of dialogue generation)의 다양성을 향상시키기 위해 DiffusionDialog, 새로운 방법을 제안합니다. 이 방법은 연속적인 잠재 변수(continuous latent variables)를 도입하고, 확산 모델(diffusion model)을 사용하여 대화(task)에 적합한 잠재적 주어진 문맥에 따른(latent given the context) 적절한 추론(inferring process)과 효과적인 사전 구성(effective prior of the latent space)을 구축하는 문제를 해결합니다.

- **Technical Details**: DiffusionDialog는 인코더(encoder)와 잠재 기반(latent-based) 확산 모델을 결합하여, 고정된 가우스 분포(Gaussian distribution) 대신 연속 공간에서 응답의 잠재 표현(latent representation)을 사전으로 인코딩합니다. 그런 다음 확산 모델을 통해 단계적으로 잠재 요소를 제거하여(denoising step by step) 추론합니다. 이 모델은 프리트레인된 언어 모델(pre-trained language model)과 결합되어 특정 응답 생성을 위해 확산 모델과 언어 모델을 통합합니다.

- **Performance Highlights**: 실험 결과는 모델이 대화 응답의 다양성을 크게 향상시키면서 일관성을 유지한다는 것을 보여줍니다. 또한, 우리의 확산 모델은 자연어 처리에서 확산 모델을 적용할 때의 주요 도전 과제인 높은 추론 효율성(inference efficiency)을 달성합니다. 이는 다중 턴 대화(multi-turn dialogue) 상황에서의 시간 소비 문제를 극복하는 데 있어 중요한 진전입니다.



### Transferable and Efficient Non-Factual Content Detection via Probe  Training with Offline Consistency Checking (https://arxiv.org/abs/2404.06742)
- **What's New**: 이 논문에서는 사람이 만든 데이터에 의존하지 않고, 오프라인 자체 일관성 검사(offline self-consistency checking) 결과를 통해 학습되는 탐색 모델(PINOSE)을 제안합니다. 이로써 다양한 데이터 분포에서도 비팩트 내용을 효과적으로 감지할 수 있는 이전 방법들과는 다른 새로운 접근 방식을 제시합니다.

- **Technical Details**: PINOSE는 주로 대규모 언어 모델(LLMs)의 내부 상태를 분석하여 반응 디코딩 이전에 팩트(Factual / factuality) 결함을 탐지합니다. 이 모델은 자체 일관성 확인을 오프라인으로 수행함으로써 여러 응답을 생성하는데 따른 계산 부담을 줄이면서도, 데이터 분포에 걸쳐 높은 전환이 가능함을 보장합니다.

- **Performance Highlights**: 실험 결과에 따르면, PINOSE는 주요 QA(Question Answering) 데이터셋과 비팩트 감지 벤치마크에서 기존의 감지 방법들을 크게 뛰어넘는 성능을 보여주었습니다. 구체적으로, 기존의 감독된 프로빙 기반의 베이스라인보다 7.7-14.6 AUC 상승을 보였으며, 비감독 자체 일관성 검사 기반 베이스라인과 비교하여도 3-7 AUC의 성능 향상과 더 빠른 시간 효율을 보였습니다.



### Llama-VITS: Enhancing TTS Synthesis with Semantic Awareness (https://arxiv.org/abs/2404.06714)
Comments: 9 pages, 2 figures, 4 tables

- **What's New**: 이 연구에서는 Llama-VITS라는 새로운 접근 방식을 소개합니다. Llama-VITS는 대규모 언어 모델(Large-scale Language Models, LLMs)인 Llama2를 VITS 모델과 통합하여 음성 합성의 의미 있는 내용을 강화하는 새로운 기법입니다. 이러한 통합은 텍스트의 의미적 내용을 이용하여 보다 정확하고 감정적으로 풍부한 음성 출력을 생성할 수 있도록 합니다.

- **Technical Details**: Llama-VITS는 Llama2로부터 추출한 의미론적 임베딩을 사용하여 VITS, 최신의 end-to-end(text-to-speech) TTS(Text-To-Speech) 모델과 통합됩니다. Llama2는 뛰어난 zero-shot 학습 능력을 가지고 있으며, 최소한의 파라미터 조정만으로 미세조정과 유사한 수준의 개선을 달성할 수 있습니다. 연구는 LJSpeech 데이터셋과 EmoV_DB_bea_sem 데이터셋에서 실시되었습니다.



### MathVC: An LLM-Simulated Multi-Character Virtual Classroom for  Mathematics Education (https://arxiv.org/abs/2404.06711)
Comments: Work in progress

- **What's New**: 이 연구에서는 선진적인 기술을 활용하여 MATHVC (Mathematics Virtual Classroom)라는 최초의 대규모 언어 모델(Large Language Models, LLM)을 기반으로 하는 가상 교실을 소개합니다. 이 플랫폼은 여러 LLM 시뮬레이션된 학생 캐릭터들과 함께 인간 학생이 수학적 모델링(Mathematical Modeling, MM) 기술을 연습할 수 있게 해줍니다. 이는 교사가 직접 모니터링하지 않아도 학생들이 협력하여 문제를 해결하면서 MM 기술을 연습할 수 있는 기회를 제공합니다.

- **Technical Details**: 개발된 MATHVC는 MM과 협력적 문제 해결(Collaborative Problem-Solving, CPS) 이론을 통합하여 캐릭터 시뮬레이션과 대화 절차를 정렬하는 두 가지 도전과제를 해결합니다. 특히, '문자 변수(Character Variable) 스키마'와 '메타 플래너(Meta Planner)'를 도입하여 각 학생 캐릭터의 동적 문제 이해 과정을 표현하고, 전체 대화의 절차를 조직하는 데 중점을 둡니다.

- **Performance Highlights**: 실험과 소거 연구(Ablation Study)를 통해 MATHVC가 개별 캐릭터 및 캐릭터 간 상호 작용 시뮬레이션에서 우수한 성능을 보이며, 실제 중학교 학생들 간의 협력적 MM 작업에서의 대화와 유사한 결과를 내는 것으로 확인되었습니다. 이는 MATHVC가 실제 교육 현장에서 중학생들을 지원할 엄청난 잠재력을 가지고 있음을 보여줍니다.



### CQIL: Inference Latency Optimization with Concurrent Computation of  Quasi-Independent Layers (https://arxiv.org/abs/2404.06709)
Comments: ARR Under Review

- **What's New**: 이 연구는 대형 언어모델(Large Language Models, LLMs)의 추론 지연을 줄이기 위한 새로운 접근 방식을 제안합니다. 연구팀은 독립적인 것처럼 보이는 계층들(Quasi-Independent Layers)을 동시에 계산(Concurrent Computation)하여 지연 시간을 크게 줄일 수 있는 'CQIL(Concurrent Computation of Quasi-Independent Layers)' 방법을 개발했습니다. 이는 기존의 계층을 제거하거나 생략하는 방식보다 성능 저하를 크게 줄이면서 효율을 증가시키는 접근법입니다.

- **Technical Details**: CQIL은 인접한 계층 간의 입력 유사성을 활용하여, 독립적인 처리를 가능하게 합니다. 이 방법은 주의 모듈(attention modules)의 출력을 입력과 정렬된 계층 간에 전달하는 우회 기법(bypassing technique)도 도입했습니다. 이는 정보 손실을 최소화하면서, 계층들을 병렬로 처리함으로써 추론 지연을 줄이는 데 중점을 둔다. 모델의 성능을 유지하면서 LLaMA-33B 모델에서 최대 48.3%까지 추론 지연을 감소시킬 수 있었습니다.

- **Performance Highlights**: CQIL 방법은 LLaMA-33B 모델에서 추론 지연을 최대 48.3% 감소시켰습니다. 이 방법은 성능 저하를 최소화하면서도 지연 감소 효과를 극대화하는 것을 목표로 합니다. 성능 측면에서도 기존 모델 대비 유사한 수준을 유지하면서, 사용자 경험을 개선할 수 있는 중요한 기술적 진전을 이루었습니다.



### Onco-Retriever: Generative Classifier for Retrieval of EHR Records in  Oncology (https://arxiv.org/abs/2404.06680)
Comments: 18 pages

- **What's New**: 이 연구에서는 전자 건강 기록(EHR)에서 정보를 검색하는 새로운 접근 방식을 제시합니다. 특히, 이 논문은 'Onco-Retriever'라고 불리는 새로운 종양학(온콜로지) 데이터 요소를 위한 맞춤형 검색기(retriever) 개발에 초점을 맞춥니다. 다른 모델들과 비교하여 Onco-Retriever는 특화된 데이터셋 생성 및 고도화된 평가 방법을 통해 높은 성능을 보였습니다.

- **Technical Details**: Onco-Retriever는 GPT-3을 사용하여 합성 데이터셋을 생성하고, 이 데이터를 사용하여 특화된 검색 모델을 distillation하는 방식으로 개발되었습니다. 이 모델은 다른 임베딩 기반(embedding-based) 모델들과 비교해 볼 때 더 적은 파라미터(미만 500M)를 가지면서도 더 높은 F-1 점수(30-50 포인트 차이)를 달성했습니다. 또한 로컬 배포(local deployment)에 적합하도록 설계되어 데이터 보안 및 개인 정보 보호에 큰 이점을 제공합니다.

- **Performance Highlights**: Onco-Retriever는 기존의 Ada나 Mistral과 같은 소유권 모델들보다 30-50 F-1 점수가 높으며, PubMedBERT 모델과의 비교에서도 우수한 성능을 나타냈습니다. 모델 평가는 실제 EHR 데이터를 사용하여 수행되었으며, 모델의 대기 시간(latency) 또한 분석되어 실제 의료 환경에 통합될 준비가 되어 있음을 보여줍니다.



### What's Mine becomes Yours: Defining, Annotating and Detecting  Context-Dependent Paraphrases in News Interview Dialogs (https://arxiv.org/abs/2404.06670)
- **What's New**: 이 연구는 대화(dialog) 내에서 맥락 의존적인 패러프레이즈(context-dependent paraphrases)를 구체화하고, 자동 감지하는 최초의 시도입니다. NPR과 CNN 뉴스 인터뷰에서 추출한 600개 말투 쌍에 대한 5,581개의 주석(annotation)이 포함된 데이터셋을 제공합니다. 우리는 대화에서 패러프레이즈를 탐지하기 위해 인-콘텍스트 학습(In-Context Learning, ICL)과 토큰 분류 모델(token classification models)을 사용하여 유망한 결과를 제시합니다.

- **Technical Details**: 연구팀은 대화 시 맥락을 고려했을 때의 의미 변화를 파악하여 패러프레이즈를 분류하는 방법을 개발하였습니다. 이를 위해 Llama 2와 GPT-4와 같은 인기 있는 생성 모델을 사용하여 인-콘텍스트 학습을 실행하고, DeBERTa 토큰 분류기를 미세 조정하여 패러프레이즈를 감지합니다.

- **Performance Highlights**: 제안된 방법은 F1 스코어가 0.66에서 0.81 사이의 성능을 보여주며, 생성 모델이 분류 작업에서 더 우수한 성능을 보이고 토큰 분류기는 텍스트 범위를 더 잘 파악하는 결과를 나타냈습니다.



### CulturalTeaming: AI-Assisted Interactive Red-Teaming for Challenging  LLMs' (Lack of) Multicultural Knowledg (https://arxiv.org/abs/2404.06664)
Comments: Preprint (under review)

- **What's New**: 이 연구는 다양한 문화적 지식을 평가하기 위한 새로운 AI-팀웍 시스템인 CulturalTeaming을 소개합니다. CulturalTeaming은 인간 주석자(human annotators)의 창의성과 전문 문화 지식을 활용하고 대규모 언어 모델(LLMs)의 확장성과 표준화를 결합하여 포괄적인 평가 데이터셋을 구축합니다.

- **Technical Details**: CulturalTeaming 시스템은 사용자가 문화적으로 중요한 다중선택형 질문(MCQs)을 생성하도록 안내하는 대화형 플랫폼으로 구성됩니다. 이 시스템은 문화적 맥락 조사와 사용자 경험에 대한 사용자 가능성 조사를 포함합니다. 이 연구에서는 AI 도움(AI-Assisted)과 검증자만(Verifier-Only)을 포함한 두 가지 설정을 실험하여, AI 도움이 주석자의 질문 생성 능력을 어떻게 향상시킬 수 있는지 탐구합니다.

- **Performance Highlights**: 생성된 CulturalBench-v0.1 평가 데이터셋은 34개의 다양한 문화에서 252개의 질문을 포함하며, 현대 LLMs는 이 데이터셋에서 37.7%에서 72.2% 사이의 정확도를 보여, LLMs의 다문화 지식에 상당한 격차를 드러냅니다. LLM 생성 힌트(LLM-generated hints)를 사용한 AI-Assisted 사용자가 더 어려운 질문을 개발할 수 있는 것으로 나타났습니다.



### Leveraging Interesting Facts to Enhance User Engagement with  Conversational Interfaces (https://arxiv.org/abs/2404.06659)
Comments: 10 pages, 1 figure

- **What's New**: 이 논문에서는 대화형 작업 지원 시스템(Conversational Task Assistants, CTA)을 사용하여 사용자의 관심과 참여를 유지하기 위해 흥미로운 사실을 포함하는 새로운 접근 방식을 제안합니다. 사용자가 요리 등의 복잡한 작업을 수행할 때 시간이 지남에 따라 지루함을 느끼지 않도록 하기 위한 방법으로, 대화 중에 관련성이 높고 흥미로운 사실을 제공함으로써 사용자의 만족도와 작업 완료율을 향상시키고자 합니다.

- **Technical Details**: 연구팀은 사회심리학 이론(Socio-psychological theories)에 기반을 두고 고도의 분류기를 훈련하여 관련성 있고 흥미로운 사실을 자동으로 식별합니다. 이 분류기는 82%의 F1-score를 달성했으며, 이를 통해 요리 분야에 특화된 흥미로운 사실에 대한 주석이 달린 데이터셋을 생성했습니다. 또한, 대화 정책을 설계하여 실제 다중 모달 목소리 어시스턴트에서의 대화에 이러한 사실들을 통합할 수 있도록 하였습니다.

- **Performance Highlights**: 실제 사용자 테스트 결과, 제시된 사실 중 66%가 긍정적인 반응을 얻었고, 사용자 만족도는 40% 향상되었으며, 대화 길이는 37% 증가했습니다. 이러한 결과는 CTA 경험에 흥미로운 사실을 통합하는 것이 사용자 참여를 증진시킬 수 있음을 입증하며, 복잡한 실제 작업을 위한 CTA의 채택을 증진시킬 수 있는 잠재력을 보여줍니다.



### RULER: What's the Real Context Size of Your Long-Context Language  Models? (https://arxiv.org/abs/2404.06654)
- **What's New**: 새로운 시험 도구인 RULER는 장문의 문맥을 이해하는 언어 모델(LM)을 평가하기 위해 개발되었습니다. RULER은 단순 검색 기능 이상을 평가하고자 다양한 유형 및 수량의 '바늘(needle)'을 포함한 검색(retrieval) 능력 평가, 다단계 추적(multi-hop tracing), 정보 집계(aggregation), 그리고 질의응답(Question Answering) 등의 새로운 과제 유형을 도입합니다.

- **Technical Details**: RULER은 장문 문맥에서의 언어 모델 성능을 평가하기 위해 다양한 설정의 유연성을 제공하는 합성 벤치마크입니다. 이 도구는 변형된 '바늘-건초더미(needle-in-a-haystack, NIAH)' 검사를 기반으로 하여, 다양한 종류의 바늘을 검색할 수 있는 능력, 다단계 트레이싱을 통한 엔티티 추적, 그리고 장문 문맥에서의 중요 정보 집계를 평가합니다. 이러한 평가는 질의응답(Question Answering) 작업을 포함하여 실제적인 문맥 크기와 작업 복잡성을 통제할 수 있습니다.

- **Performance Highlights**: GPT-4, Command-R, Yi-34B, Mixtral 같은 모델들은 32K 토큰 크기의 문맥에서 만족스러운 성능을 유지하는 반면, 다른 모델들은 성능이 크게 저하됩니다. Yi-34B는 200K 토큰 길이의 지원을 주장하며 입력 길이와 작업 복잡성이 증가함에 따라 성능 저하가 큰 것으로 나타났습니다. 모델들은 문맥 크기가 커짐에 따라 매개변수 지식(parameteric knowledge)에 더 의존하고, 비검색 작업에서 문맥을 복사하는 경향이 증가하는 두 가지 행동 패턴을 보였습니다.



### Khayyam Challenge (PersianMMLU): Is Your LLM Truly Wise to The Persian  Language? (https://arxiv.org/abs/2404.06644)
- **What's New**: Khayyam Challenge (또는 PersianMMLU로 알려짐)는 페르시아어를 지원하는 대형 언어 모델(LLM)의 성능을 평가할 수 있는 새로운 벤치마크를 제공합니다. 이 데이터셋은 페르시아 시험에서 추출된 38가지 다양한 과제에서 20,192개의 네 가지 선택 질문으로 구성되어 있으며, 문학 이해에서 수학, 과학, 논리, 지능 테스트 등 다양한 주제를 포함하고 있습니다. 이를 통해 다양한 교육 단계에서 LLM의 언어 이해력, 추론 능력 및 정보 검색 능력을 평가하는 데 도움이 될 것입니다. 특히, 데이터 오염 문제를 피하기 위해 새로운 데이터를 사용하고, 번역 문제 없이 페르시아어 사용자를 위해 맞춤화된 원본 데이터를 사용합니다. 또한, 문제마다 사람의 응답률, 난이도 수준 및 서술형 답변과 같은 풍부한 메타데이터를 포함합니다.

- **Technical Details**: Khayyam Challenge는 주로 페르시아어로 구성된 데이터와 함께, 다양한 교육 단계 및 주제에 대한 광범위한 질문을 통해 LLM을 평가합니다. 이 도전 과제는 문화적 뉘앙스를 포함하여 자연스럽게 언어의 미묘함과 복잡성을 통합합니다. 이 벤치마크는 이전 페르시아어 데이터셋과 달리 추출적 데이터셋이 아닌 다양한 주제와 교육 단계를 다루면서 LLM의 전반적인 언어 이해와 추론 능력을 더 폭넓게 평가합니다. 또한, 데이터셋은 ‘이란의 국립 대학 입학 시험’과 ‘Kanoon Farhangi Amoozesh(문화 교육 센터)’의 질문을 포함하여 각 과목별 전문가들에 의해 검증 및 확인되었습니다.

- **Performance Highlights**: 이 연구에서는 GPT-3.5, GPT-4(OpenAI, 2023), Aya(Üstün et al., 2024), PersianMind(Rostami et al., 2024), mGPT(Shliazhko et al., 2022), mT0(Muennighoff et al., 2022), Claude3-haiku(Anthropic, 2024), XVERSE111과 같은 다양한 최신 언어 모델을 평가했습니다. 이 모델들, 특히 GPT-4는 경제학, 심리학, 사회학 등 문맥 이해에 의존하는 분야에서 상대적으로 나은 성능을 보였습니다. 그러나 모든 모델들이 기술적 분야, 특히 이산수학에서 정밀한 언어 이해가 필요한 결과에 있어서는 개선이 필요함을 보여줍니다. 또한, LLM은 특히 ‘함정 질문(trapped questions)’이라 불리는 인간을 속이도록 설계된 질문에서 인간보다 높은 정확도를 보여준 것으로 분석되었습니다.



### What is Your Favorite Gender, MLM? Gender Bias Evaluation in  Multilingual Masked Language Models (https://arxiv.org/abs/2404.06621)
- **What's New**: 이 연구에서는 중국어, 영어, 독일어, 포르투갈어, 그리고 스페인어를 포함한 다양한 언어에 대한 성편견(gender bias)을 평가하는 다국어 접근 방식을 제안합니다. 이전의 연구와 달리, 본 논문은 영어와 다른 언어를 연결하는 병렬 코퍼스에 의존하지 않고 다국어 사전을 사용하여 성 편향을 탐지합니다. 또한, 성 편향 분석을 위한 보다 견고한 문장 쌍을 생성하는 새로운 모델 기반 방법을 소개합니다.

- **Technical Details**: 저자들은 두 가지 방법, 즉 사전 기반(lexicon-based) 방법과 모델 기반(model-based) 방법을 각 언어에 적용하여 두 개의 데이터셋을 생성합니다. 이 데이터셋들은 해당 언어에 특화된 Masked Language Model(MLM)을 사용하여 성 편향을 평가하는 데 사용됩니다. 또한 기존의 평가 지표와 세 가지 새로운 평가 지표를 사용하여 성 편향을 측정합니다. 연구 결과, 이전 접근 방식이 데이터에 민감하며 문맥적 종속성을 제거하지 못해 안정적이지 않음을 보여줍니다.

- **Performance Highlights**: 제안된 접근 방법은 기존 방법들과 비교하여 타겟 코퍼스에서 더 많은 데이터를 보존하며, 성별 분포가 치우친 데이터에서도 성 편향을 평가하는 데 있어 더 일관된 결과를 보여줍니다. 이는 다양한 평가 지표를 사용하여 대규모 데이터셋에서 성 편향을 연구하는 것이 최선의 방법임을 제시합니다.



### FairPair: A Robust Evaluation of Biases in Language Models through  Paired Perturbations (https://arxiv.org/abs/2404.06619)
- **What's New**: 새로운 평가 프레임워크인 FairPair를 소개합니다. 이는 일상적인 사용 중에 발생하는 특정 그룹에 대한 차별적 대우를 평가할 수 있도록 설계되었습니다. FairPair는 언어 모델의 이러한 차별적 대우를 예측하고 평가하기 위해 대조쌍(counterfactual pairs)을 사용하지만, 두 연속적인 텍스트가 동일한 인구 통계적 그룹에 기반하여 생성되므로, 더 공정한 비교를 가능하게 합니다.

- **Technical Details**: FairPair는 각 프롬프트에 대해 여러 생성을 통해 모델의 변동성을 고려하고, 생성 과정의 내재된 변동성을 측정합니다. 이 평가 프레임워크는 John과 Jane과 같은 두 개체를 예로 들어, 동일한 문구에서 시작하여 서로 다른 두 연속적인 텍스트를 생성하고 이를 평가하는 방식을 구현합니다. 언어 모델(g)은 주어진 프롬프트 x (기호로는 'x')에 대한 완성된 텍스트를 생성하고, 이를 기반으로 적절한 스코어링 함수를 사용하여 일치를 평가합니다.

- **Performance Highlights**: FairPair를 사용하여 여러 널리 사용되는 생성 모델을 평가한 결과, 이 프레임워크는 일반적이고 자연스러운 문장을 포함한 새로 구성된 데이터 세트 'Common Sents'에서 성별에 따른 편향을 조사했습니다. 잭카드 불일치(Jaccard dissimilarity) 및 감정(sentiment) 평가를 사용하여 성별 편향을 조사하였으며, 이는 계산하기 쉽기 때문에 우선적으로 사용되었습니다.



### Less is More for Improving Automatic Evaluation of Factual Consistency (https://arxiv.org/abs/2404.06579)
Comments: Accepted in NAACL24 Industry; 7 pages

- **What's New**: 자동 생성된 텍스트의 사실 일치성을 평가하는 것은 신뢰할 수 있는 자연 언어 생성 응용 프로그램을 개발하는데 중요하다. 최근 연구에서는 기존 방법보다 우수한 성능을 보인 통합 정렬 모델을 사용한 AlignScore를 제안했다. 본 연구에서는 AlignScore가 사용한 데이터셋을 자세히 조사하고 예상치 못한 발견을 했다: 데이터 포인트 수를 줄이면 성능이 실제로 향상될 수 있다. 우리는 원래 AlignScore 훈련 데이터셋에서 노이즈를 제거하고, 강인성을 높인 샘플로 증강시킨 후, 데이터의 10%만을 사용해 LIM-RA(Less Is More for Robust AlignScore)라고 부르는 개선된 사실 일치성 평가 모델을 훈련시켰다. LIM-RA는 AlignScore와 다른 강력한 베이스라인들을 지속적으로 능가하며 새로운 State-of-the-Art 벤치마크를 설정했다.

- **Technical Details**: LIM-RA는 DeBERTa를 기반으로 훈련된 개선된 모델이다. 훈련 데이터의 질을 향상시키기 위해 여러 단계의 Ablation 연구를 수행했다. 특히, 훈련 데이터 크기를 분석하고 강인성을 향상시키기 위해 합성 데이터를 생성했다. 데이터 세트의 인공적인 '잡음'을 줄이기 위해 데이터의 10%만 사용하여 학습한 결과, LIM-RA는 AlignScore보다 우수한 성능을 보였다. 또한, 이 모델은 이름이나 숫자 같은 요소들에 대한 변조를 더 잘 감지한다는 장점이 있다. 훈련에 사용된 데이터는 47만개의 샘플로, 각 데이터세트에서 2만개의 샘플만을 사용하였다.

- **Performance Highlights**: LIM-RA는 33개의 테스트 데이터세트 중 24개에서 가장 높은 점수를 달성했으며, 나머지에서도 경쟁력을 유지했다. 특히, 자연 언어 생성 데이터세트 두 개와 대규모 언어 모델 출력에 초점을 맞춘 두 벤치마크에서 탁월한 성능을 보였다. 이는 ChatGPT를 포함한 다른 강력한 베이스라인들을 일관되게 능가하는 결과로, 새로 정의된 Large Language Model Response(LLMR) 벤치마크에서는 가장 높은 성과를 보였다.



### UMBRAE: Unified Multimodal Decoding of Brain Signals (https://arxiv.org/abs/2404.07202)
Comments: Project Page: this https URL

- **What's New**: 이 연구에서는 뇌의 신호를 통해 객체의 개념과 위치를 정확하게 복원하고, 특정 모델 없이 여러 대상자에게 적용할 수 있는 새로운 방법 UMBRAE(Unified Multimodal Decoding of Brain Responses)를 제안합니다. 또한, BrainHub, 새로운 뇌 이해 벤치마크를 구축하여 공개하였습니다.

- **Technical Details**: UMBRAE는 두 가지 주요 기술적 접근 방식을 활용합니다. 첫째, 다수의 대상자로부터 얻은 뇌 신호를 통합적으로 해석할 수 있는 Universal Brain Encoder를 도입하여, 대상자 간 차이를 극복하고, 한 명의 대상자에 대한 소량의 데이터만을 사용하여 새로운 대상자에게 적응하는 방법을 제안합니다. 둘째, Multimodal Large Language Model(MLLM)과의 연동을 통해 뇌 신호에서 얻은 정보를 텍스트와 이미지 등 다양한 형태로 해석합니다. 이를 통해 더욱 정확하고 다양한 뇌-기계 인터페이스 사용이 가능해집니다.

- **Performance Highlights**: UMBRAE는 기존 연구와 비교하여 뛰어난 성능을 보였습니다. 특히, 뇌 활동의 의미론적(Semantic) 및 공간적(Spatial) 정보를 보다 정확하게 복원하며, 새로운 대상자에 대한 적응도 소량의 데이터로 가능하게 했습니다. 또한, BrainHub 벤치마크를 사용하여 방대한 뇌 신호 데이터와 함께 다양한 평가 임무를 수행함으로써, 이 기술의 강점과 적용 가능성을 입증했습니다.



### WordDecipher: Enhancing Digital Workspace Communication with Explainable  AI for Non-native English Speakers (https://arxiv.org/abs/2404.07005)
Comments: The Third Workshop on Intelligent and Interactive Writing Assistants (In2Writing) at CHI 2024

- **What's New**: WordDecipher는 비영어권 사용자(NNES)를 위한 AI 지원 글쓰기 도구로, 디지털 작업 공간에서의 커뮤니케이션을 개선합니다. 이 도구는 사용자의 의도를 파악하고, 스타일과 내용에서의 뉘앙스를 설명하여 더 정확한 표현 선택을 돕습니다.

- **Technical Details**: WordDecipher는 LLM(large language models), 스타일 임베딩(style embeddings)을 사용하여 작성된 문장의 사회적 의도를 감지하고, 사용자가 의도한 메시지에 맞게 문장을 재작성하는 제안을 생성합니다. 또한, 다양한 제안 사이의 차이를 사용자에게 설명하여, 선택 과정을 지원합니다.

- **Performance Highlights**: WordDecipher는 효과적인 커뮤니케이션을 지원하여 NNES의 작업 공간에서의 언어적 표현능력을 크게 향상시킬 수 있습니다. 사용자는 자신의 의도를 더 정확하게 전달할 수 있으며, 이는 프로젝트의 성공률을 높이고, 상호 이해를 증진시킵니다.



### Language Generation in the Lim (https://arxiv.org/abs/2404.06757)
Comments: 24 pages, 2 figures

- **What's New**: 이 연구는 언어 생성(generation)과 언어 식별(identification)의 차이점을 강조하면서, 적대적 상황에서도 언어 생성이 가능함을 보여줍니다. 특히, 기존의 골드-앵글루인(Gold-Angluin) 모델에서 나타난 부정적 결과와 대조적으로, 어떠한 가정도 없이 계산 에이전트가 특정 언어를 생성할 수 있다는 주장을 제시합니다.

- **Technical Details**: 가정이 없는 기본 설정에서 시작하여, 불특정 언어 L에 대한 문자열 열거 시, 계산 에이전트가 결국 L에서만 새로운 문자열을 생성할 수 있음을 보여주는 이론적 결과를 도출합니다. 연구는 언어 L이 무한한 후보 언어 목록 중 하나에서 발생한다고 가정하며, 이는 어떤 언어 식별 문제와는 근본적으로 다른 문제임을 시사합니다. 언어 식별이 언어의 정확한 이름을 결국 명명해야 하는 데 반해, 언어 생성은 단지 언어의 새로운 요소를 출력해야 합니다.

- **Performance Highlights**: 이 연구의 주요 결과는, 계산 에이전트가 모든 가산(countable) 후보 언어 목록에 대해 한계에서(language in the limit) 언어를 생성할 수 있다는 것을 입증합니다. 이는 언어 식별의 어려움에 대한 골드와 앵글루인의 연구 결과와 대조적입니다. 여기서, 식별은 일반적으로 불가능한 반면, 생성은 가능하다는 것이 밝혀졌습니다. 이 결과는 특히 적대적 예제(adversarial examples)에서도 언어 생성이 가능하다는 점에서 중요합니다.



### Global Contrastive Training for Multimodal Electronic Health Records  with Language Supervision (https://arxiv.org/abs/2404.06723)
Comments: 12 pages, 3 figures. arXiv admin note: text overlap with arXiv:2403.04012

- **What's New**: 이 논문에서는 의학 시계열(medical time series)과 임상 노트(clinical notes)에 중점을 두어 새로운 다중 모달 대조 학습 프레임워크(multimodal contrastive learning framework)를 도입합니다. 이 프레임워크는 불규칙한 시간 간격(irregular time intervals)과 희소성(sparsity) 문제를 해결하기 위해 동적 임베딩 및 토크나이제이션 스킴(dynamic embedding and tokenization scheme)을 통합하고, 환자의 다양한 모달 특성 간의 상호 연결된 관계를 파악하기 위해 글로벌 대조 손실(global contrastive loss)을 적용했습니다.

- **Technical Details**: 이 프레임워크는 시간에 따른 교차 주의 변환기(cross-attention transformers)와 함께 유연한 위치 인코딩(flexible positional encoding)과 학습 가능한 시간 임베딩(learnable time embedding)을 사용하여 의료 시계열의 희소성과 시간 샘플링의 불규칙성을 처리합니다. 또한, 다변량 특성(multivariate characteristics) 사이의 복잡한 관계를 모델링하기 위한 변수별 인코딩 전략(variable-specific encoding strategy)이 도입되었습니다. 다양한 모달 데이터를 활용하여 환자의 종합적인 병원 체류 기록을 대조적으로 학습하게 함으로써, 의학적 임상 결과 예측의 정확성을 크게 향상시키는 것을 목표로 합니다.

- **Performance Highlights**: 실제 EHR 데이터셋을 사용한 실험에서, 이 프레임워크는 수술 후 합병증 발생 예측과 같은 예시적 과제에서 기존의 최신 기법들(state-of-the-art approaches)을 능가하는 성능을 보여주었습니다. 113,953명의 성인 환자와 124,777건의 주요 입원 수술 데이터를 분석함으로써, 이 연구는 특히 대규모 병원 데이터를 활용한 임상 의사 결정 지원에 있어 중요한 발전을 이루었습니다.



### CoVoMix: Advancing Zero-Shot Speech Generation for Human-like  Multi-talker Conversations (https://arxiv.org/abs/2404.06690)
- **What's New**: 최근 연구에서는 제로샷 텍스트-투-스피치(TTS) 모델링이 크게 발전하면서, 고품질 및 다양한 음성 생성이 가능하게 되었습니다. 그러나 실제 대화 생성과 말의 자연스러움을 달성하는 것은 여전히 도전적인 문제로 남아 있습니다. 이 논문에서는 CoVoMix, 즉 대화형 음성 혼합 생성 모델을 소개하며, 이는 제로샷 상황에서 사람과 같은 멀티스피커, 다차례 대화 음성을 생성할 수 있는 첫 시도입니다.

- **Technical Details**: CoVoMix 모델은 대화 텍스트를 개별 발화자에 대한 의미 정보를 나타내는 복수의 이산 토큰(discrete tokens) 스트림으로 변환합니다. 이후, 플로우-매칭(flow-matching) 기반 음향 모델을 통해 혼합된 멜-스펙트로그램(mixed mel-spectrograms)을 생성하고, 최종적으로 HiFi-GAN 모델을 사용하여 음성 파형을 생성합니다. CoVoMix는 제로샷 시나리오에서 다수의 발화자의 목소리를 동시에 모방할 수 있는 능력을 갖추고 있습니다.

- **Performance Highlights**: CoVoMix는 인간과 유사한 자연스러움과 일관성을 가진 대화를 생성함으로써, 멀티스피커 및 멀티라운드 대화에 있어서의 효과성을 입증하였습니다. 생성된 대화는 실제 인간 대화에서의 웃음과 같은 발화학적 행동을 포함하여, 매끄러운 음성 전환과 중첩되는 음성을 특징으로 합니다. 이 모델은 평가를 위한 다양한 메트릭스를 설계하였고, Fisher 데이터셋을 사용하여 훈련 및 평가가 이루어졌습니다.



### SafeGen: Mitigating Unsafe Content Generation in Text-to-Image Models (https://arxiv.org/abs/2404.06666)
- **Newsletter**: [{"What's New": '최신 AI 뉴스에서, 새로운 프레임워크 SafeGen이 소개되었습니다. 이는 텍스트를 이미지로 변환하는 모델(text-to-image, T2I)이 안전하지 않은 콘텐츠, 특히 성적 내용을 생성하는 것을 억제하는 데 중점을 둔 혁신적인 접근 방식입니다. SafeGen은 텍스트 입력과 관계없이 불건전한 시각적 표현을 모델에서 제거하는 것을 목표로 합니다.'}, {'Technical Details': 'SafeGen은 이미 학습된 T2I 모델에서 성적 이미지 생성 능력을 제거하기 위한 것으로, 변조된 self-attention 레이어를 사용해 실제 이미지 분포에서 성적인 이미지를 제거합니다. 이를 통해 모델은 성적 함의를 가진 텍스트가 주어져도 성적 이미지를 생성하지 못하도록 하며, 이러한 텍스트-불가지론(text-agnostic) 접근 방식은 기존의 텍스트-의존 방식의 한계를 극복합니다.'}, {'Performance Highlights': 'SafeGen은 네 개의 데이터셋에서 광범위한 실험을 통해 그 효과가 검증되었습니다. 이 프레임워크는 기존의 여덟 가지 최신 기법보다 더 높은 성능을 보이며, 성적 콘텐츠 제거 성능은 99.1%에 달합니다. 또한 SafeGen은 고화질의 양성 이미지 생성을 유지하면서도, 인공지능 모델의 책임 있는 사용을 위한 기술을 제공합니다.'}]



### On the Effect of (Near) Duplicate Subwords in Language Modelling (https://arxiv.org/abs/2404.06508)
- **What's New**: 이 논문은 언어 모델(Language Models, LMs)의 토큰화(tokenisation) 과정에서 발생하는 가까운 중복(near duplicates) 서브워드(subwords)가 LM 훈련의 효율성에 미치는 영향을 연구합니다. 토큰화는 문자 시퀀스를 서브워드로 분할하고 이를 임의의 인덱스로 할당하는 과정으로, 전형적으로 정보 손실이 없지만, 캐릭터 레벨(character-level) 정보를 제거함으로써 비슷한 서브워드 간의 일반화를 어렵게 할 수 있습니다. 이 연구는 서브워드의 완벽한 일반화를 통해 모델이 얼마나 향상될 수 있는지의 상한을 설계하는 실험을 수행하고, 천연으로 발생하는 가까운 중복 서브워드의 영향을 조사합니다.

- **Technical Details**: 연구는 두 주요 실험으로 구성됩니다. 첫 번째 실험에서는 LM의 어휘에 있는 각 서브워드를 복제하여 완벽하게 동등한 서브워드 클래스를 만듬으로써, 가까운 중복을 완전히 일반화할 경우 모델이 얼마나 개선될 수 있는지의 상한을 탐구합니다. 두 번째 실험에서는 자연스럽게 발생하는 가까운 중복 서브워드를 병합하는 것이 LM의 성능에 심각한 영향을 미친다는 것을 발견하였습니다. 이 두 실험 결과는 서브워드의 복제가 LM 훈련의 효율성에 부정적인 영향을 미치지만, 자연적으로 발생하는 가까운 중복된 서브워드가 예상처럼 유사하지 않아 성능 개선의 가능성을 제한한다는 결론을 내립니다.

- **Performance Highlights**: LMs가 완전히 중복된 설정에서 훈련될 때 대략 17% 더 많은 데이터가 필요하다는 실험 결과를 발견했습니다. 또한, 기존의 가까운 중복 서브워드를 병합할 경우 LM의 성능이 상당히 저하되는 것으로 나타났습니다.



### Comparing Two Model Designs for Clinical Note Generation; Is an LLM a  Useful Evaluator of Consistency? (https://arxiv.org/abs/2404.06503)
Comments: Accepted to NAACL 2024 Findings

- **What's New**: 이 연구에서는 의사와 환자 간 대화의 오디오 녹음을 기반으로 SOAP 노트의 다양한 섹션을 생성하는 두 가지 접근 방식을 분석합니다. 첫 번째 방법은 각 섹션을 독립적으로 생성하는 반면, 두 번째 방법은 모든 섹션을 함께 생성합니다. PEGASUS-X Transformer 모델을 사용하여 두 방법 모두 유사한 ROUGE 값과 사실성(Factuality) 지표에서 차이가 없음을 관찰했습니다.

- **Technical Details**: 이 논문에서는 두 가지 노트 생성 설계, GENMOD와 SPECMOD를 비교합니다. GENMOD는 단일 모델을 사용하여 전체 임상 노트를 생성하는 반면, SPECMOD는 각각의 섹션을 생성하기 위해 세 개의 개별 모델을 사용합니다. Llama2와 같은 대규모 언어 모델(Large Language Model, LLM)을 사용하여 인간 평가자와 유사한 일관성을 달성하는 것을 보여줍니다. 또한, 일관성 개선을 위해 이전에 생성된 모든 섹션의 출력을 기반으로 각 새 섹션을 생성하는 방법이 더 효과적임을 발견했습니다.

- **Performance Highlights**: 인간 평가를 통해 일관성 측면을 측정하여 Llama2 모델이 인간 평가자와 거의 동일한 동의도로 동일한 작업을 수행할 수 있음을 입증했습니다. 연령, 성별, 부상 부위의 일관성에 대한 Cohen Kappa 상호 평가자 신뢰도는 각각 0.79, 1.00, 0.32로 관찰되었습니다. 이를 통해 LLM을 사용하여 사람들이 식별할 수 있지만 자동화 된 측정 지표에 의해 현재 캡처되지 않는 품질 지표를 측정하는 것의 유용성을 입증합니다.



### Pitfalls of Conversational LLMs on News Debiasing (https://arxiv.org/abs/2404.06488)
Comments: The paper is accepted at the DELITE workshop which is co-located at COLING/LREC

- **What's New**: 이 논문은 뉴스 편집에서의 편향 제거(debiasing)를 다루며 대화형 대형 언어 모델(Conversational Large Language Models, LLMs)이 이 작업에서 얼마나 효과적인지 평가합니다. 뉴스 편집자의 관점을 반영한 평가 체크리스트를 설계하고, 미디어 편향에 관한 공개 데이터셋의 일부를 사용하여 인기 있는 세 가지 대화형 모델로부터 생성된 텍스트를 얻고, 설계된 체크리스트에 따라 이 텍스트들을 평가했습니다. 또한, 모델이 편향 제거 모델 출력의 질을 확인하는 평가자로서의 역할도 검토했습니다.

- **Technical Details**: 연구는 GPT (Generative Pre-trained Transformer)와 같은 대화형 LLMs를 사용하여 뉴스 문장의 편향을 제거하는 작업에 초점을 맞추고 있습니다. 특히 ChatGPT와 GPT4 모델을 포함하여, 편향 분류 데이터셋의 하위 집합에 대한 텍스트 생성을 수행했습니다. 평가 체크리스트에는 편향 제거(Correcting Bias), 정보 보존(Preserving Information), 맥락 보존(Preserving Context), 언어 유창성 보존(Preserving Language Fluency), 저자 스타일 보존(Preserving Author's Style) 등이 포함됩니다.

- **Performance Highlights**: 연구 결과, 대화형 LLMs는 편향을 완벽하게 제거하는 데는 미흡하며, 특히 일부 모델들은 저자의 스타일을 변경하거나 잘못된 정보를 생성하는 불필요한 변화를 도입할 수 있습니다. 또한, 이 모델들은 도메인 전문가들만큼 편향 제거된 출력물의 질을 평가하는 데 능숙하지 않습니다.



### Ada-LEval: Evaluating long-context LLMs with length-adaptable benchmarks (https://arxiv.org/abs/2404.06480)
Comments: NAACL 2024

- **What's New**: 이 논문에서는 LLM의 긴 컨텍스트(long-context) 이해력을 평가하기 위해 길이 조정 가능한 새로운 벤치마크인 Ada-LEval을 소개합니다. Ada-LEval은 복잡한 텍스트 분석 기능과 디테일한 LLM 평가를 가능하게 하는 TSort와 BestAnswer라는 두 가지 과제를 포함하고 있습니다.

- **Technical Details**: Ada-LEval은 테스트 케이스의 길이를 세부적으로 조정할 수 있는 기능을 자랑합니다. 예를 들어, TSort에서는 텍스트 세그먼트의 수와 길이를 조절할 수 있고, BestAnswer에서는 유도 선택지의 수를 조정할 수 있습니다. 이는 모델이 텍스트를 완전히 이해하는 데 필수적인 요소입니다. 또한, 이 벤치마크는 128k 토큰까지의 텍스트 샘플을 쉽게 생성할 수 있으며, 주요 초울트라 롱 컨텍스트(ultra-long-context) 설정에서의 모델 성능을 평가합니다.

- **Performance Highlights**: Ada-LEval을 사용한 평가에서는 텍스트 길이가 길어질수록 기존 LLM들의 성능이 현저히 감소하는 것을 관찰할 수 있었습니다. 특히 더 긴 텍스트에서는 주요 지시사항을 따르는데 문제가 있었고, 입력 순서에 대한 편향이 뚜렷하게 나타났습니다. 그러나 스케일러블 위치 임베딩(scalable position embeddings) 기술을 적용한 LLM들은 표준 모델보다 개선된 성능을 보여주었습니다.



### Text-Based Reasoning About Vector Graphics (https://arxiv.org/abs/2404.06479)
Comments: Project page: this https URL

- **What's New**: 본 연구에서는 벡터 그래픽(vector graphics)에 대한 정밀한 시각적 인식 문제를 해결하기 위해 Visually Descriptive Language Model (VDLM)을 제안합니다. VDLM은 Scalable Vector Graphics (SVG)를 사용하여 벡터 이미지를 텍스트 기반의 중간 기호 표현인 Primal Visual Description (PVD)으로 변환하고, 이를 기존의 대규모 언어 모델과 연동하여 추론합니다. 이러한 새로운 접근 방식은 특히 저수준의 시각적 세부사항이 요구되는 작업에서 강화된 성능을 보입니다.

- **Technical Details**: VDLM은 세 가지 주요 컴포넌트(component)로 구성되어 있습니다. 첫째, 벡터 그래픽을 SVG로 변환하는 규칙 기반 시각 인코더(visual encoder), 둘째, SVG를 PVD로 번역하는 학습된 언어 모델(language model), 마지막으로 PVD 표현을 사용하여 다운스트림 작업을 수행하는 추론 전용 대규모 언어 모델(LLM)입니다. 특히, SVG에서 PVD로의 매핑(mapping)은 기하학적 형태로 묘사되며, 이는 텍스트 기반 모델을 사용하여 시각적 특성과 언어 모델을 효과적으로 연결합니다.

- **Performance Highlights**: VDLM은 다양한 벡터 그래픽 시각적 추론 과제에서 제로샷(zero-shot) 성능이 뛰어나며, 이전의 최첨단 LMMs보다 우수한 성능을 나타냅니다. 실험 결과는 VDLM이 GPT-4V 및 LLaVA-v1.5를 포함한 다른 모델들과 비교했을 때 더 높은 정확도와 일반화 능력을 보여줍니다. 뿐만 아니라, VDLM 프레임워크는 해석 가능성을 증가시키고 모듈식 접근 방식을 통해 각 컴포넌트의 성능이 전체 프레임워크의 효율성을 향상시키는 것을 입증하였습니다.



### Take a Look at it! Rethinking How to Evaluate Language Model Jailbreak (https://arxiv.org/abs/2404.06407)
- **What's New**: 본 논문은 언어 모델(Language Model)의 안전성을 보장하기 위해, 전통적인 'jailbreak' 평가 방법의 한계를 지적하고 새로운 다중 평가 방법(multifaceted evaluation method)을 제안합니다. 특히, 기존 방법들이 단순히 jailbreak 시도를 성공적 혹은 실패적으로 이분화하여 평가하는 것을 넘어서, 보다 복잡한 평가 척도를 도입하여 언어 모델의 jailbreak 시도를 보다 정밀하게 판단하고자 합니다.

- **Technical Details**: 이 연구에서는 세 가지 새로운 평가 척도, 즉 안전규정 위반(safeguard violation), 정보성(informativeness), 그리고 상대적 진실성(relative truthfulness)을 소개합니다. 연구팀은 자연어 생성(Natural Language Generation, NLG) 평가 방법을 확장하여 이러한 척도를 계산하는 다면적 접근법을 개발하였습니다. 평가는 세 가지 악의적 의도 데이터셋과 세 가지 jailbreak 시스템을 포함하는 벤치마크 데이터셋에서 수행되었으며, 이 데이터셋은 세 명의 주석자(annotators)에 의해 라벨링되었습니다.

- **Performance Highlights**: 실험 결과, 새로운 다면적 평가 방법은 기존의 jailbreak 평가 방법들과 비교하여 평균 17% 개선된 F1 점수를 보였습니다. 이는 새로운 평가 체계가 언어 모델의 안전성을 보다 효과적으로 평가하고 강화할 수 있음을 시사합니다. 연구 결과는 jailbreak 문제에 대한 이진적 관점에서 벗어나, 보다 포괄적인 평가를 통해 언어 모델의 안전을 보장할 필요성을 강조합니다.



### MiniCPM: Unveiling the Potential of Small Language Models with Scalable  Training Strategies (https://arxiv.org/abs/2404.06395)
Comments: 17 pages paper, 7 pages Appendix

- **What's New**: MiniCPM이라는 새로운 소형 언어 모델(Small Language Models, 이하 SLM) 시리즈가 소개되었습니다. 이 모델들은 1.2B와 2.4B 비임베딩 파라미터(non-embedding parameter) 버전으로, 각각의 범주에서 탁월한 성능을 발휘하며, 7B~13B 규모의 대형 언어 모델(Large Language Models, 이하 LLM)과 유사한 능력을 보여줍니다. MiniCPM은 모델 스케일링과 데이터 스케일링을 통해 연구의 확장성을 보여주며, 이를 통해 미래의 LLM 연구를 안내할 수 있는 가능성을 제시합니다.

- **Technical Details**: MiniCPM 시리즈는 확장 가능한 훈련 방법론을 채택하여, 모델의 크기와 데이터의 양을 체계적으로 확장할 수 있습니다. 특히, Warmup-Stable-Decay (WSD) 학습률 스케줄러(Learning Rate Scheduler, LRS)를 도입하여 연속 훈련과 도메인 적응을 용이하게 합니다. 이 스케줄러는 혁신적인 훈련 동역학을 제공하며, 모델과 데이터의 스케일링 법칙을 효율적으로 연구할 수 있는 기반을 마련합니다. 또한, MiniCPM-DPO, MiniCPM-MoE, MiniCPM-128K 등 다양한 패밀리 모델들이 소개되어 뛰어난 성능을 입증하였습니다.

- **Performance Highlights**: MiniCPM 모델들은 기존의 대형 모델들과 비교하여 우수한 성능을 보여줍니다. 예를 들어, 2.4B MiniCPM-128K 모델은 Yarn-Mistral-7B-128K 및 ChatGLM3-6B-128K 모델과 비슷하거나 더 나은 성능을 보였으며, MiniCPM-MoE 모델은 4B 활성화된 파라메터로 Llama2-34B와 비슷한 수준의 성능을 나타냈습니다. 이러한 결과는 MiniCPM이 소형 모델임에도 불구하고 대형 모델에 필적하는 능력을 갖추었음을 입증합니다.



### Event Extraction in Basque: Typologically motivated Cross-Lingual  Transfer-Learning Analysis (https://arxiv.org/abs/2404.06392)
Comments: Accepted at LREC-Coling 2024

- **What's New**: 이 연구는 저자가 바스크어(Basque)를 대상 언어로 사용하여 교차 언어 전이 학습(cross-lingual transfer-learning)의 효과를 탐구한 것으로, 사건 추출(Event Extraction) 작업에서 언어적 유사성이 성능에 미치는 영향을 분석합니다. 특히, 출처 언어(source language)와 대상 언어(target language) 간의 언어학적 특징이 전이 품질에 미치는 영향을 밝히는 데 중점을 두었습니다. 이를 위해, 새로운 이벤트 추출 데이터셋인 EusIE를 도입하였으며, 이는 다양한 언어 쌍을 통한 광범위한 실험을 가능하게 합니다.

- **Technical Details**: 연구팀은 MEE (Multilingual Event Extraction) 데이터셋을 활용하여 다양한 언어에서의 사건 추출 모델을 훈련시키고, 바스크어로 평가하는 방식으로 실험을 수행했습니다. 토큰 분류(token classification) 작업(예: 엔티티 및 이벤트 트리거 식별)에서는 공통적인 작성 스크립트(writing script)와 형태학적(morphological) 특성이 교차 언어 전이의 품질을 향상시키는 반면, 구조적 예측(structural prediction) 작업(예: 인수 추출(argument extraction))에서는 단어 순서(word order)가 더 중요한 특징으로 나타났습니다.

- **Performance Highlights**: 실험 결과, 공통적인 언어적 특성이 소스 언어와 타겟 언어 간의 전이 품질에 영향을 미친다는 것을 확인했습니다. 구조적인 이해가 필요한 작업에서는 공통적인 단어 순서가, 토큰 분류 작업에서는 공통적인 작성 스크립트 및 형태학적 특성이 더 중요한 역할을 하는 것으로 밝혀졌습니다. 데이터셋 및 코드는 공개적으로 제공되어 연구 공동체가 자유롭게 접근하고 사용할 수 있습니다.



### Latent Distance Guided Alignment Training for Large Language Models (https://arxiv.org/abs/2404.06390)
- **What's New**: 이 연구에서 새로운 정렬(Alignment) 트레이닝 방법인 'LD-Align (Latent Distance Guided Alignment Training)'을 제안합니다. 이 방법은 고품질의 감독된 튜닝 데이터셋(Supervised Fine-tune (SFT) dataset)을 사용하여 대규모 언어 모델(LLMs)의 정렬을 사람의 주석 없이 독립적으로 향상시키기 위한 것입니다. 이 연구는 사람의 주석을 필요로 하지 않는 새로운 자동 정렬 훈련 방법의 가능성을 탐구하고 있습니다.

- **Technical Details**: 'LD-Align'은 자동 인코딩(auto-encoding) 방식을 통해 생성된 잠재 공간에서 샘플 쌍의 거리를 측정하고, 이 거리를 사용하여 직접적 선호 최적화(Direct Preference Optimization, DPO)를 기반으로 한 정렬 훈련을 안내합니다. 이 방법은 잠재 벡터(Latent Vectors)의 거리를 계산하여 각 사례의 상관성을 평가하고, DPO를 사용하여 반복적으로 훈련하면서 낮은 정렬 수준을 가진 샘플에 더 많은 업데이트 가중치를 할당합니다.

- **Performance Highlights**: 실험을 통해 'LD-Align' 방법이 선택된 다른 메소드들 중에서 가장 우수한 성능을 달성했습니다. 이는 잠재 공간의 지도를 통해 대규모 언어 모델의 정렬 과정에서 사람의 주석에 의존하지 않고도 높은 정확도를 유지할 수 있음을 보여줍니다. 이는 높은 정렬 수준을 유지하면서도 학습 과정에서의 오버피팅(overfitting) 위험을 줄이는 데 도움이 됩니다.



### ClinLinker: Medical Entity Linking of Clinical Concept Mentions in  Spanish (https://arxiv.org/abs/2404.06367)
- **What's New**: 이 연구는 스페인어 의료 개념을 핸들링하기 위해 특화된 언어 모델인 ClinLinker를 개발하였습니다. ClinLinker는 SapBERT 기반의 bi-encoder를 사용하여 후보를 초안하고, cross-encoder를 사용해 후보를 재순위하는 두 단계 파이프라인을 사용합니다. 이 모델은 기존의 다국어 모델을 크게 능가하는 결과를 보여주었습니다.

- **Technical Details**: ClinLinker는 contrastive-learning 전략을 따라 훈련된 cross-encoder로, UMLS나 SNOMED-CT 같은 표준화된 의료 용어에 엔티티를 정규화하는 과정에서 중요한 기능을 합니다. 이 모델은 높은 정확도의 후보 검색과 재순위 과정을 통해 다양한 의료 텍스트에서 중요한 의료 용어를 효과적으로 연결할 수 있습니다.

- **Performance Highlights**: ClinLinker는 DisTEMIST(diseases와 관련된 골드 스탠다드 데이터)와 MedProcNER(clinical procedures와 관련된 골드 스탠다드 데이터) 데이터 세트에서 top-k accuracy 25지표로 각각 40점, 43점으로 이전 벤치마크를 40점, 43점으로 상회하는 성과를 달성했습니다. 이는 이 모델이 스페인어 의료 텍스트에 특화되어 있으며, 다양한 의료 용어의 뉘앙스를 잘 처리할 수 있다는 것을 보여줍니다.



### SurveyAgent: A Conversational System for Personalized and Efficient  Research Survey (https://arxiv.org/abs/2404.06364)
Comments: 6 pages

- **What's New**: 이 논문에서는 연구자들이 문헌 조사(literature review) 과정을 개인화되고 효율적으로 돕기 위한 새로운 대화형 시스템인 SurveyAgent를 소개합니다. 이 시스템은 지식 관리(Knowledge Management), 문헌 추천(Recommendation), 질의 응답(Query Answering)의 세 가지 주요 모듈을 통합하여 연구자들이 과학 문헌과 상호작용하는 방식을 개선하는 데 중점을 두고 있습니다.

- **Technical Details**: SurveyAgent는 세 가지 주요 모듈을 갖추고 있습니다: 1) 지식 관리 모듈은 사용자의 연구 관심사에 따라 논문을 찾고 조직화합니다. 2) 추천 모듈은 키워드 기반 쿼리를 통해 논문을 검색하고 유사한 논문을 추천합니다. 이는 arXiv Sanity 프로젝트와 LLM(Language Large Model)에 의해 지원됩니다. 3) 질의 응답 모듈은 특정 논문에 대한 다양한 질문에 대한 답변, 요약, 리뷰 제공을 돕습니다. 이 모듈들은 대화형 인터페이스를 통해 사용자 상호 작용 및 개인화를 우선시하면서 연구자들을 지원합니다.

- **Performance Highlights**: SurveyAgent의 효과성은 양적 실험 및 질적 사례 연구를 통해 검증되었습니다. 사용자의 연구 활동을 계획하고, 논문을 추천하며, 질의에 응답하는 데 있어 기능적 우수성을 보였습니다. 또한, 이 시스템은 연구자들이 과학 문헌을 다루는 새로운 방식을 제시함으로써 연구 과정을 혁신적으로 개선할 잠재력을 가지고 있습니다.



### Generalizable Sarcasm Detection Is Just Around The Corner, Of Course! (https://arxiv.org/abs/2404.06357)
- **What's New**: 이 연구에서는 비꼬는 말 인식 모델의 강인성을 평가하기 위해 다양한 특성을 가진 네 가지 비꼬는 말 데이터셋을 활용하여 Fine-tuning(파인 튜닝)을 진행하고, 동일 데이터셋(단일 데이터셋 테스트)과 다른 데이터셋 간(교차 데이터셋 테스트) 예측 성능을 테스트했습니다. 새롭게 출시한 데이터셋에서 Fine-tuning된 모델이 다른 데이터셋에 대한 최고의 일반화 능력을 보여주었습니다.

- **Technical Details**: 비꼬는 말 인식 모델은 저자 레이블과 제3자 레이블, 온라인 소셜 미디어 대 오프라인 대화, 공격적인 스타일 대 유머러스한 비꼬기 등 다양한 차원에서의 Fine-tuning을 했습니다. 단일 데이터셋 테스트에서 제3자 레이블로 Fine-tuning된 모델이 저자 레이블로 Fine-tuning된 모델보다 일관되게 성능이 좋았습니다. 그러나 교차 데이터셋 테스트에서는 대부분의 모델이 일반화에 실패하여 다양한 스타일과 도메인의 비꼬는 말을 포괄하지 못하는 것으로 나타났습니다.

- **Performance Highlights**: 새로운 데이터셋을 사용하여 Fine-tuning을 거친 모델은 기존 데이터셋에 비해 타 데이터셋에 대한 일반화 성능이 가장 높았습니다. 비꼬는 말의 다양한 영역과 스타일을 고려해야 함을 강조하면서, 다양한 도메인과 스타일에서의 비꼬는 말 검출을 위해 더 넓은 관점의 연구가 필요함을 주장하였습니다.



### RAR-b: Reasoning as Retrieval Benchmark (https://arxiv.org/abs/2404.06347)
- **What's New**: RAR-b (Reasoning as Retrieval Benchmark)는 기존의 언어 이해 모델이 갖고 있는 추론 능력을 평가하기 위해 설계된 새로운 평가 체계입니다. 이 벤치마크는 특히 검색 엔진(retrievers)이 독립적으로 추론 문제를 해결할 수 있는지를 조사하여 LLMs와 함께 사용될 때의 최대 성능을 예측합니다. 이는 언어 모델이 복잡한 사고 과정과 논리적 추론에 대해 어느 정도 능숙한지를 평가하는 것을 목표로 하고 있습니다.

- **Technical Details**: RAR-b는 복수 선택 추론(multiple-choice retrieval)과 전체 데이터셋 검색(full-dataset retrieval) 설정으로 구성된 12개의 추론 작업으로 실험을 설계했습니다. 추론 문제를 검색 문제로 변환하고, 각 검색 모델이 해당 문제에 얼마나 효과적으로 대응하는지를 평가하기 위해 세 가지 주요 분류의 모델을 사용하였습니다: 비지도 밀집 검색 모델(unsupervised dense retrieval models), 지도 밀집 검색 모델(supervised dense retrieval models), 지시어-인식 밀집 검색 모델(instruction-aware dense retrieval models). 또한, 최적의 데이터셋 검색을 위해 재정렬 모델(re-ranking models)의 성능도 평가했습니다.

- **Performance Highlights**: RAR-b 평가 결과, 현재의 상태-최고 기술(state-of-the-art) 검색 모델들은 아직 리즈닝 문제에 효과적으로 대응할만큼 충분히 발달하지 않았음을 보여줍니다. 그러나 최신 디코더 기반 임베딩 모델(decoder-based embedding models)은 이러한 격차를 좁히는 데 큰 가능성을 보여주었습니다. 특히, 재정렬 모델을 활용한 미세조정(fine-tuning)을 통해 모델이 추론 능력을 획득하는 것이 더욱 용이함을 발견했으며, 이러한 방식으로 모든 작업에서 최고의 성능을 달성하였습니다.



### Finding fake reviews in e-commerce platforms by using hybrid algorithms (https://arxiv.org/abs/2404.06339)
- **What's New**: 이 논문에서는 가짜 리뷰 감지를 위한 감성 분석의 새로운 앙상블 접근 방식을 제안합니다. 이 앙상블 방식은 서포트 벡터 머신(Support Vector Machine, SVM), K-최근접 이웃(K-Nearest Neighbors, KNN), 그리고 의사 결정 트리(Decision Tree) 분류기를 결합하여 각 모델의 장점을 활용하고 단점을 보완함으로써 높은 정확도와 견고함을 달성합니다.

- **Technical Details**: 제안된 앙상블 아키텍처는 다양한 모델을 전략적으로 결합하여, 실제 데이터셋에서 나타나는 다양한 언어 패턴과 뉘앙스에 적응할 수 있는 능력을 향상시킵니다. 이를 통하여, 가짜 리뷰 예측에서 높은 성능을 발휘합니다.

- **Performance Highlights**: 가짜 리뷰에 대한 평가 지표(metrics)를 통해 확인된 결과는 제안된 앙상블 방법의 효과와 경쟁력을 입증합니다. 전통적인 단일 모델 접근법과 비교했을 때, 앙상블 기법은 가짜 리뷰를 찾아내는 최첨단 기술을 발전시키는 데 있어 잠재력을 강조합니다.



### nEMO: Dataset of Emotional Speech in Polish (https://arxiv.org/abs/2404.06292)
Comments: Accepted for LREC-Coling 2024

- **What's New**: 이 논문은 폴란드어 감정 스피치 데이터셋 'nEMO'의 개발을 소개합니다. 이 데이터셋은 다양한 감정 상태(분노, 두려움, 행복, 슬픔, 놀람, 중립 상태)를 표현하는 9명의 배우들에 의해 녹음된 3시간 분량의 샘플로 구성되어 있습니다. 이는 슬라브어족 언어의 감정 인식 연구에 중요한 자원을 제공합니다.

- **Technical Details**: nEMO 데이터셋은 각기 다른 감정 상태를 명확히 분리할 수 있도록 시뮬레이션 접근 방식(simulated approach)을 사용하여 개발되었습니다. 데이터셋 생성에 있어 폴란드어의 음성학을 적절히 반영하기 위한 언어적 콘텐츠가 준비되었으며, 90개의 문장이 각각 폴란드어에서 드문 발음을 포함하도록 설계되었습니다. 9명의 폴란드어 원어민 배우들이 참여하였고, 프로페셔널과 아마추어의 균형을 맞추기 위한 노력이 있었습니다.



### LLMs' Reading Comprehension Is Affected by Parametric Knowledge and  Struggles with Hypothetical Statements (https://arxiv.org/abs/2404.06283)
- **What's New**: 이 연구에서는 대형 언어 모델(LLMs)의 자연어 이해(NLU) 역량을 평가하기 위해, 소설적(fictitious) 데이터를 기반으로 한 독해(Reading Comprehension, RC) 태스크를 사용하는 새로운 접근 방식을 제안합니다. 이 방법은 LLMs의 세계 지식과 독립적이며, 모델이 텍스트의 맥락을 이해하는 능력만을 평가할 수 있게 합니다.

- **Technical Details**: 연구자들은 ChatGPT, GPT-4, LLaMA 2, 그리고 Mixtral 모델을 테스트하기 위해 'imaginary data'를 사용하여 모델의 언어 처리 능력을 평가했습니다. 이러한 독해 태스크는 가상의 사실(facts) 및 엔티티(entities)에 관한 질문을 포함하고, 모델이 내부 지식 없이 오로지 제공된 텍스트를 기반으로 답변을 해야 합니다. 이 연구는 모델이 복잡한 조건적(modal) 및 가설적(conditional) 맥락에서 자주 오류를 범하는 경향을 발견했습니다.

- **Performance Highlights**: 모든 모델은 간단한 긍정적 및 부정적 맥락에서 높은 정확도로 작업을 처리할 수 있으나, 조건적이거나 모달 맥락에서는 훨씬 더 자주 오류를 범합니다. 또한, 가설적 시나리오를 처리하는 능력이 모델의 주요 약점 중 하나로 확인되었습니다. 이는 모델이 내부 지식과의 충돌을 제대로 분리하지 못할 때, 특히 많이 나타났습니다.



### Understanding Cross-Lingual Alignment -- A Survey (https://arxiv.org/abs/2404.06228)
- **What's New**: 이 연구에서는 다국어 모델에서 언어 간 표현의 유의미한 유사성을 증진하기 위한 기술들을 조사함으로써, 다국어 언어 모델에서 '크로스링구얼 얼라인먼트(Cross-lingual alignment)'에 관한 문헌을 총체적으로 검토하고 정리했습니다. 특히 기존의 인코더 모델뿐만 아니라 인코더-디코더 및 디코더 전용 모델까지 확장하여 적용 가능성을 탐색했습니다.

- **Technical Details**: 크로스링구얼 얼라인먼트는 유사한 의미를 가진 단어나 문장이 표현 공간에서 더 유사하게 나타나도록 하는 것을 의미합니다. 이 연구에서는 '약한 정렬(weak alignment)'과 '강한 정렬(strong alignment)'이라는 두 가지 관점을 제시했습니다. 연구팀은 다양한 사전 훈련 목표, 새로운 사전 훈련된 모델, 대조적 미세조정(contrastive fine-tuning), 임베딩 공간의 사후 조정(post-hoc adjustments)과 같은 방법을 통해 얼라인먼트를 향상시키기 위한 연구들을 조사했습니다.

- **Performance Highlights**: 이 연구는 얼라인먼트 측정 방법의 한계를 지적하며, 향후 연구에서는 언어 중립적 요소와 언어 특정 요소 사이의 효과적인 균형을 찾는 것이 중요하다고 강조했습니다. 특히, 강한 정렬을 위해서는 소스 언어와 타깃 언어 간의 유사하지 않은 의미가 더 멀리 떨어지도록 해야 하며, 이는 기존에 변환기 모델들(Transformer models)에서 관찰된 높은 비등방성(anisotropy) 때문에 어려움이 있음을 지적했습니다.



### Low-Cost Generation and Evaluation of Dictionary Example Sentences (https://arxiv.org/abs/2404.06224)
- **What's New**: 이 연구는 사전 예문 생성 및 평가에 대한 저비용, 제로-샷(zero-shot) 방법을 제안합니다. 새로운 자동 평가 척도인 OxfordEval을 도입하여 생성된 문장이 옥스퍼드 사전(Oxford Dictionary)의 기존 문장과 비교하여 얼마나 잘 수행하는지 측정합니다. 이 척도는 인간의 판단과 높은 일치성을 보입니다. 또한, LLM(large language models)을 이용하여 혁신적인 방식으로 사전 예문을 자동 생성하는 기법을 개발했습니다.

- **Technical Details**: OxfordEval 메트릭은 LLM을 사용하여 후보 사전 예문의 품질을 옥스퍼드 사전의 예시와 경쟁적으로 평가합니다. 이 연구는 Claude, Llama-2, Mistral과 같은 최신 LLM을 활용하여 다양한 단어 클래스(word classes)에 걸쳐 사전 문장을 생성합니다. 또한 사전 훈련된 마스크 언어 모델(masked language models, MLM)을 적용하여 생성된 문장 중 어휘 의미를 가장 잘 나타내는 문장을 선별하는 새로운 방법을 개발했습니다. 이러한 방법을 통해 OxfordEval 측정치에 따른 승률이 85.1%까지 향상되었습니다.

- **Performance Highlights**: 기존 모델로 생성된 문장의 승률은 39.8%인 반면, 새롭게 개발된 FM-MLM 모델은 OxfordEval을 통한 평가에서 85.1%의 높은 승률을 달성했습니다. 이는 LLM의 효과적인 사용과 새로운 평가 방식의 도입이 크게 기여한 결과입니다. 전체 과정을 수행하는 데 드는 비용은 $50 미만으로, 고품질의 사전 예문 생성을 위한 비용 효율적인 방법을 제공합니다.



### VI-OOD: A Unified Representation Learning Framework for Textual  Out-of-distribution Detection (https://arxiv.org/abs/2404.06217)
Comments: COLING 2024

- **What's New**: 이 논문에서는 텍스트 기반 OOD(out-of-distribution) 감지를 위한 변분 추론 프레임워크(Variational Inference framework for OOD detection, VI-OOD)를 제안합니다. 이 프레임워크는 조건부 확률 p(y|x)가 아닌, 연합 확률 p(x, y)을 최대화하여 OOD를 효과적으로 감지합니다. 특히, 변형된 사후(Posterior) 분포와 트랜스포머(Transformer) 모델을 사용하여 텍스트 데이터의 유용한 표현을 추출합니다.

- **Technical Details**: 이 연구는 기존의 OOD 감지 방식이 가지고 있는 편향된 표현 학습 문제를 해결하기 위해, 텍스트 데이터의 특성을 감안한 새로운 접근 방식을 제안합니다. VI-OOD 방식은 ID(in-distribution) 데이터의 공동 분포 p(x, y)을 최대화하며, 이를 위해 증거 하한(Evidence Lower Bound, ELBO)을 최적화합니다. 또한, 변분 자기부호화기(Variational Autoencoder, VAE) 기법을 사용하여, 트랜스포머 모델의 중간 계층에서 동적으로 결합된 은닉 상태를 기반으로 사후 분포를 조절합니다.

- **Performance Highlights**: 텍스트 분류 작업에서의 광범위한 실험을 통해, VI-OOD는 다양한 인코더 기반 및 디코더 기반의 트랜스포머 아키텍처에서 우수한 OOD 감지 성능을 보여주었습니다. 특히, Mahalanobis Distance 방법 같은 거리 기반 OOD 탐지기와 비교할 때 더 나은 결과를 제공합니다. 추가로, VI-OOD는 텍스트 데이터의 복잡한 특성을 효과적으로 활용하면서, 기존의 후처리(post-hoc) OOD 감지 알고리즘들의 성능을 일관되게 향상시킬 수 있습니다.



### [Call for Papers] The 2nd BabyLM Challenge: Sample-efficient pretraining  on a developmentally plausible corpus (https://arxiv.org/abs/2404.06214)
- **What's New**: 다가오는 2024/2025 BabyLM 챌린지는 작년에 이어 다시 개최됩니다. 이번 챌린지의 큰 변화로는, 이전의 루즈 트랙을 제거하고 '페이퍼 트랙'(Paper track)을 새롭게 도입하여 비모델 기반 제출이나 새로운 인지적 영감을 받은 벤치마크를 허용합니다. 또한, 프리트레이닝 데이터(pretraining data) 관련 규칙을 완화하여 참가자들이 100M-word 또는 10M-word 예산 내에서 자체 데이터셋을 구성할 수 있게 되었습니다. 새롭게 '멀티모달 비전-앤드-랭귀지 트랙'(multimodal vision-and-language track)도 도입되어, 텍스트만 있는 데이터와 이미지-텍스트 멀티모달 데이터가 혼합된 코퍼스를 제공할 예정입니다.

- **Technical Details**: BabyLM 챌린지의 목표는 사람의 발달에서 영감을 받은 데이터 제한을 주어 프리트레이닝(pretraining) 최적화에 중점을 둔 연구를 장려하는 것입니다. 이번 챌린지에서는 언어 전용(language-only)과 멀티모달(multimodal) 데이터셋을 제공하며, 참가자는 이 데이터 예산 한도 내에서 자체적으로 데이터셋을 구성할 수 있습니다. 모든 트랙은 언어 전용 평가 작업과 멀티모달 작업으로 평가될 예정입니다. 주요 트랙들은 '스트릭트'(Strict), '스트릭트-스몰'(Strict-small), '비전 앤드 페이퍼'(Vision and Paper) 포함하고 있습니다.

- **Performance Highlights**: 이번 챌린지의 결과는 2024년 9월 13일에 제출해야 하며, 논문은 9월 20일에 제출해야 합니다. 심사는 10월 8일에 시작되어 10월 30일에 종료될 예정이며, 최종 결과는 NeurIPS에서 발표될 예정입니다. 참가자들은 언어 모델의 평가와 관련하여 여러 기준에 맞게 모델을 개발하고 최적화할 수 있는 기회를 제공받습니다.



### Clue-Instruct: Text-Based Clue Generation for Educational Crossword  Puzzles (https://arxiv.org/abs/2404.06186)
- **What's New**: 이 연구에서는 교육용 크로스워드 퍼즐을 위한 단서 생성 데이터셋(clue-instruct)을 소개하며, 이는 44,075개의 유일한 예시를 포함합니다. 이 데이터셋은 Wikipedia 페이지에서 관련 키워드에 대한 정보를 수집함으로써 구축되었으며, 이를 바탕으로 대규모 언어 모델(Large Language Models, LLMs)을 사용하여 학습된 단서를 자동 생성합니다.

- **Technical Details**: 이 데이터셋은 키워드 및 상응하는 세 개의 크로스워드 단서가 포함된 텍스트-키워드 쌍으로 구성됩니다. 또한, 다양한 오픈 소스 LLM을 사용하여 데이터셋에서 모델을 미세 조정(fine-tune)하고, 자동 및 인간 평가를 통해 생성된 단서의 품질을 확인하였습니다. 연구 팀은 Wikipedia 페이지 내용을 수집하고, 이를 통해 관련 키워드와 맥락에 따른 교육적 단서를 생성하는 방법론을 제안합니다.

- **Performance Highlights**: 학습된 모델들은 높은 품질의 단서 생성을 보여주었으며, 사용자 및 자동 평가 모두에서 긍정적인 결과를 얻었습니다. 이는 LLM을 이용한 교육용 크로스워드 단서 생성 방법론이 효과적임을 입증합니다.



### Characterizing Multimodal Long-form Summarization: A Case Study on  Financial Reports (https://arxiv.org/abs/2404.06162)
- **What's New**: 이 논문에서는 금융 보고서 요약을 사례 연구로 사용하여, 큰 언어 모델(Large Language Models, LLMs)이 긴 입렵을 취급하는 능력과 행동을 이해하기 위해 체계적인 분석을 수행했습니다. 특히, Claude 2.0/2.1과 GPT-4 /3.5, 그리고 Command 모델의 행동을 조사하여 금융 보고서에서 수치 데이터 사용과 요약의 장편성(multimodal long-form)을 분석했습니다.

- **Technical Details**: 다양한 모델을 평가하여 Claude 2가 GPT-4보다 더 강력한 숫자 활용 능력을 가지고 있으며, 위치 편향(position bias)이 적은 것으로 밝혀졌습니다. 이러한 위치 편향은 입력을 섞은 후 Claude에서 사라지는 반면 GPT-4에서는 일관되게 나타나는 현상을 관찰할 수 있습니다. 나아가, 이 논문은 수치 데이터가 포함된 요약 생성에서 나타나는 'numeric hallucination' 현상에 대한 분류법도 제시했습니다.

- **Performance Highlights**: 결과적으로, Claude 2는 숫자를 포함한 테이블 데이터 사용에서 GPT-4보다 우수한 성능을 보였으며, 추출적 요약(extractive summary) 비율도 상대적으로 낮았습니다. 또한, 숫자 활용을 개선하기 위한 프롬프트 엔지니어링(prompt engineering)을 시도했지만, GPT-4는 한정된 성공을 거두었습니다.



### (Not) Understanding Latin Poetic Style with Deep Learning (https://arxiv.org/abs/2404.06150)
- **What's New**: 이 연구는 고전 라틴 시의 작가 스타일을 이해하기 위해 다양한 신경망(LSTMs, CNNs)의 주의(attention)를 분석합니다. 특히 시의 음향 및 운율적 특성을 고려하여 인코딩된 데이터에 대해 신경망을 훈련시켜, 각 작가들의 스타일 차이를 규명하고자 했습니다. 비록 신경망이 결정을 내리는 방식이 아직 명확하지 않지만, 시 고유의 특성을 분석하기 위한 새로운 접근 방법을 제시합니다.

- **Technical Details**: 연구에서는 LSTMs와 CNNs를 사용하였으며, CNNs가 시 분석에 더 적합하다는 결론을 내렸습니다. CNNs는 훈련 시간이 더 짧고, 같은 수준의 정확도를 제공하며, 해석 가능성(interpretability)이 더 높을 가능성이 있습니다. 데이터는 음절 단위로 인코딩되며, 훈련 가능한 임베딩(embedding)이 동적 스타일의 특성을 포착하는 데 효과적임을 발견했습니다.

- **Performance Highlights**: CNNs는 LSTM보다 빠른 훈련과 비슷한 정확도를 제공합니다. 연구에서는 특히 dropout과 배치 정규화(batch normalization) 같은 과적합(overfitting)을 줄이는 기술의 중요성을 강조했습니다. 도메인 특화 방식보다는 간단한, 훈련 가능한 임베딩이 더 효과적이었습니다.



### Cendol: Open Instruction-tuned Generative Large Language Models for  Indonesian Languages (https://arxiv.org/abs/2404.06138)
Comments: Cendol models are released under Apache 2.0 license and will be made publicly available soon

- **What's New**: Cendol은 기존의 다양한 언어를 지원하는 LLM 대비 인도네시아의 토착 언어를 효과적으로 처리 할 수 있는 설계를 갖춘 새로운 인도네시아어 LLM 컬렉션을 소개합니다. 이 컬렉션은 300M에서 13B 파라미터에 이르는 다양한 규모의 decoder-only 및 encoder-decoder 아키텍처로 구성되어 있습니다. Cendol은 23개의 과제와 10개의 언어를 포괄하는 약 50M개의 지시어(instruction)로 구성된 엄격한 주문형 맞춤 코퍼스를 제공합니다.

- **Technical Details**: Cendol은 LoRA와 같은 파라미터 효율적 튜닝 방법이 언어 적응에 효과가 없음을 명시하며, 어휘 적응(vocabulary adaptation)을 사용하여 효율성을 증가시키는 새로운 접근 방식을 제안합니다. 또한 Cendol은 영어와 같은 하나의 언어에서 사전 훈련된 안전성이 인도네시아와 같은 저자원 언어로 전이 가능하다는 것을 보여줍니다. 이는 RLHF 및 안전성 세부 조정 없이도 가능합니다.

- **Performance Highlights**: Cendol은 다양한 일반 NLP 과제(감정 분석, 토픽 모델링, 기계 번역, 요약 등), 현지 지식 및 문화 가치 평가에서 효과적입니다. 이 모델은 기존의 다국어, 동남아시아(SEA), 인도네시아어 LLM보다 성능이 우수하며, 인도네시아어 과제 및 언어에 적응할 수 있는 범용성을 보여줍니다. 평가 결과, 기존 모델 대비 20% 향상된 성능을 보여주었습니다.



### SmurfCat at SemEval-2024 Task 6: Leveraging Synthetic Data for  Hallucination Detection (https://arxiv.org/abs/2404.06137)
Comments: 12 pages, 10 tables, 3 figures

- **What's New**: 이 논문은 SemEval-2024에 대한 최신 연구를 제공하며, 참조 기준과 비교하여 모델 예측을 평가하기 위한 다양한 전략을 탐구합니다. 특히, 기존의 기준 모델(baseline), 사전 훈련된 인코더들의 개선, 그리고 여러 고성능 모델을 이용한 앙상블(approach utilizing several high-performing models) 방법 등을 활용하여 3가지의 독특한 방법론을 소개합니다.

- **Technical Details**: 제시된 연구에서는 GPT-4와 LLaMA2-7B 모델을 사용하여 라벨이 없는(unlabelled) 훈련 데이터에서 추가 훈련 샘플들을 생성하여 훈련 데이터를 확장하는 방법을 적용했습니다. GPT-4를 이용한 첫 번째 접근 방식은 도메인 유지가 어려웠지만, 소량의 주석이 달린 데이터를 사용하여 LLaMA2-7B 어댑터를 훈련하고 이를 라벨이 없는 훈련 데이터에 적용한 두 번째 방식이 더 효과적인 인-도메인 데이터 증강 방법으로 확인되었습니다.

- **Performance Highlights**: 이 연구는 모델 비의존적 경로에서 9위를, 모델을 인식하는 경로에서는 17위를 달성하며 높은 성능을 입증했습니다. 연구팀은 다양한 데이터 세트에 걸쳐 사전 훈련된 임베딩 모델을 세밀하게 튜닝하고, 이를 이진 분류기로 변환하여 사용하였으며, 다른 데이터에 모델을 미세 조정하는 실험도 실시하였습니다.



### Detection of fields of applications in biomedical abstracts with the  support of argumentation elements (https://arxiv.org/abs/2404.06121)
- **What's New**: 이 연구에서는 과학 문헌에서 특정 정보를 검색하는 데 있어 개별 사실을 집중적으로 다루는 것이 전체 텍스트를 다루는 것보다 효과적일 수 있다고 합니다. 특히, 논증적 요소(argumentative elements)를 활용하여 발표의 특정 부분, 예를 들어 배경이나 저자의 주장 등에 초점을 맞추는 것이 유용할 수 있습니다. 생물의학(biomedicine) 분야에서 특정 용도를 감지하는 데 있어 이러한 논증적 요소의 추출 도구를 평가했습니다.

- **Technical Details**: PubMedBERT 사전 훈련 모델(pre-trained model)을 특정 코퍼스에서 미세 조정(fine-tuned)하여 실험을 수행하였고, 제목과 초록을 사용하는 것과 논증적 요소만을 사용하는 것을 비교 분석하였습니다. 이 연구는 학술 논문에서 논증 마이닝(argument mining)에 초점을 맞추며, 여러 도구들을 평가하여 그들이 생물의학적 문서에서 적용 범위를 감지하는 데 얼마나 유용한지를 연구했습니다.

- **Performance Highlights**: 최고의 F1 점수는 적용 분야에 따라 0.22에서 0.84까지 다양했으며, 가장 좋은 성능을 보인 논증적 라벨은 초록의 결론과 배경 섹션과 관련된 것이었습니다. 연구는 2,000개 이상의 초록을 포함하는 새로운 코퍼스를 훈련 및 평가에 사용하였습니다.



### Exploring the Necessity of Visual Modality in Multimodal Machine  Translation using Authentic Datasets (https://arxiv.org/abs/2404.06107)
Comments: bucc 2024 accepted

- **What's New**: 이 연구에서는 실제 세상에서의 번역 시나리오와 관련하여 시각적 모달리티(visual modality)의 효과를 검토합니다. Tang et al. (2022)이 제안한 범용 다모델 기계 번역(universal multimodal machine translation) 프레임워크를 적용하여, 텍스트와 시각적 내용 간의 일관성이 번역 품질에 미치는 영향을 탐구했습니다. 또한, 시각 정보가 보조적 역할을 하고 있다는 점을 발견하여, 추가적인 텍스트 정보로 대체가능성을 조사했습니다.

- **Technical Details**: 이 연구에서는 이미지 검색 기술(open-vocabulary image retrieval techniques)과 텍스트 인식 주의 이미지 인코더(text-aware attention image encoder)를 사용하여 다양한 실제 번역 데이터 세트에서 시각적 정보의 영향을 평가합니다. 실제 데이터에서 다양한 스타일의 문서를 처리할 수 있는 새로운 접근 방법을 도입하고, 이 연구는 주로 양방향 LSTM(bi-directional LSTM)을 사용하는 RNN 텍스트 인코더를 사용합니다. 또한, 텍스트와 시각적 내용의 상관 관계를 기반으로 하는 두 가지 시각적 노이즈 필터링 방법을 평가했습니다.

- **Performance Highlights**: 실험 결과, 시각 모달리티는 대부분의 진정한 번역 데이터 세트에서 이점을 제공하는 것으로 나타났습니다. 특히, 텍스트 어휘가 이미지 친화적인 경우 번역 효율성이 높아지는 경향이 있습니다. 그러나 텍스트-시각적 내용 간 일관성이 떨어지는 경우, 시각 정보의 효과가 감소하였습니다. 이 연구는 또한 시각 정보가 다모델 번역 과정에서 보조적인 역할을 하며, 추가적인 텍스트 정보로 대체 가능하다는 점을 제시합니다.



### Making Old Kurdish Publications Processable by Augmenting Available  Optical Character Recognition Engines (https://arxiv.org/abs/2404.06101)
Comments: 30 pages, 21 figures, 2 tables

- **What's New**: 이 연구는 쿠르드어 역사 문서에 대한 텍스트 추출을 향상시키기 위해 존재하는 OCR 시스템을 개선하는 데 초점을 맞추고 있습니다. 이는 쿠르드어가 저자원 언어로 간주되기 때문에 중요하며, 이 연구는 특히 1950년 이전에 인쇄된 문서로부터 수집된 새로운 데이터셋을 사용하여 모델을 훈련시켰습니다.

- **Technical Details**: 이 연구에서는 Google의 오픈소스 OCR 프레임워크인 Tesseract 버전 5.0을 사용하였습니다. 쿠르드어를 위한 공개 데이터셋이 없기 때문에 Zheen Center for Documentation and Research에서 수집한 역사적 문서를 사용하여 데이터셋을 개발했습니다. 이 데이터셋은 라인 이미지 1233개와 각각의 텍스트 기록을 포함하고 있습니다. 아라비아 모델(Arabic model)을 기반 모델로 사용하여 데이터셋으로 모델을 훈련시켰습니다.

- **Performance Highlights**: 이 모델은 Tesseract 내장 평가 도구인 lstmeval을 사용하여 문자 오류율(Character Error Rate, CER) 0.755%를 달성했습니다. 또한, Ocreval은 평균 문자 정확도 84.02%를 보여주었습니다. 이것은 쿠르드어 역사적 문서를 디지털화하고 처리하는 데 큰 진전을 의미합니다.



### All in One: An Empirical Study of GPT for Few-Shot Aspect-Based  Sentiment Anlaysis (https://arxiv.org/abs/2404.06063)
Comments: 9 pages, 5 figures

- **What's New**: 새롭게 제안된 'All in One (AiO)' 모델은 자연어 처리(Natural Language Processing, NLP) 분야에서 매우 중요하고 도전적인 과제인 측면 기반 감성 분석(Aspect-Based Sentiment Analysis, ABSA)의 모든 하위 과제를 처리할 수 있습니다. 이 모델은 GPT(Generative Pre-trained Transformers)의 발전을 통해 감성 분석에 대한 원스톱(one-stop) 솔루션을 제공할 수 있게 되었습니다. 또한, 적은 양의 데이터(few-shot data)에서도 효과적으로 작동할 수 있는 구조를 갖추고 있습니다.

- **Technical Details**: AiO 모델은 두 단계로 구성되어 있습니다. 첫 번째 단계에서는 특정 백본 네트워크(backbone network)가 리뷰의 의미론적 정보를 학습하고, 후보를 생성합니다. 두 번째 단계에서는 GPT의 맥락 학습 기능(contextual learning capabilities)을 활용하여 예측을 생성합니다. 이 연구는 GPT를 사용하여 적은 양의 ABSA 하위 과제를 정의하는 일반 학습 패러다임(general learning paradigm)을 제안합니다.

- **Performance Highlights**: AiO 모델은 여러 벤치마크 데이터셋에서 포괄적인 비교 및 손실 실험을 수행하였고, 이 모델이 ABSA의 모든 하위 과제를 효과적으로 처리할 수 있음을 보여줍니다. 특히, 적은 수의 데이터로도 높은 성능을 발휘할 수 있는 강점을 가지고 있습니다. 이로 인해 ABSA 연구에서 강한 일반화(strong generalization)와 약한 감독(weak supervision)을 향해 나아갈 수 있는 계기를 마련하였습니다.



### Identifying Shopping Intent in Product QA for Proactive Recommendations (https://arxiv.org/abs/2404.06017)
Comments: Accepted at IronGraphs@ECIR'2024

- **Korean AI Newsletter**: [{"What's New": '이 연구는 음성 비서가 사용자의 쇼핑 욕구를 식별하고 보다 효과적으로 제품 추천을 제공할 수 있는 새로운 접근 방식을 제안합니다. 구체적으로, 이 연구는 사용자가 제품과 관련된 질문(Shopping Product Questions, SPQs)을 할 때 그 질문이 실제로 쇼핑 의향이 있는지를 파악하는 데 초점을 맞춥니다.'}, {'Technical Details': '연구팀은 그래프 주의 네트워크(Graph Attention Networks, GAT)와 전문가 혼합 모델(Mixture-of-Experts, MoE)을 활용하여 사용자의 과거 구매 이력에서 추출한 피처를 활용합니다. 이를 통해 사용자의 숨겨진 쇼핑 행동 패턴을 파악하고, 이 데이터를 기반으로 SPQs를 식별합니다.'}, {'Performance Highlights': '이 방법은 F1 점수 0.91로 SPQs를 매우 정확하게 식별할 수 있음을 보여주었습니다. 또한, 실제 음성 비서 사용자를 대상으로 한 온라인 평가에서도 사용자의 쇼핑 목록에 제품을 추가할 것을 추천받은 후 사용자가 제품을 리스트에 추가하는 비율이 SPQs로 판단될 때 더 높다는 결과를 얻었습니다.'}]



### FreeEval: A Modular Framework for Trustworthy and Efficient Evaluation  of Large Language Models (https://arxiv.org/abs/2404.06003)
Comments: We open-source all our code at: this https URL

- **What's New**: FreeEval은 대규모 언어 모델(Large Language Models, LLMs)의 자동 평가를 위한 새로운 모듈형 및 확장 가능한 프레임워크입니다. 기존의 평가 방법들을 하나의 통합된 구조로 결합하여, 다양한 평가 방법론의 통합과 투명성을 개선합니다. 이를 통해 동적 평가(dynamic evaluation), 데이터 오염 감지(contamination detection), 인간 평가(human evaluation) 등을 통합하며 평가 결과의 공정성을 높이고 있습니다.

- **Technical Details**: FreeEval은 다양한 평가 접근 방식을 간소화하는 통합 추상화(unified abstractions)를 제공합니다. 이는 데이터셋 기반의 평가자, 참조 기반의 클래식 판단자(Classic Judges), LLM 기반의 평가자를 포함합니다. 또한, 분산 컴퓨팅(distributed computation) 및 캐싱 전략(caching strategies)을 포함하여 고성능 인프라를 사용하여 평가를 효율적으로 수행할 수 있도록 설계되었습니다. 메타-평가(meta-evaluation) 기술도 통합되어 평가의 신뢰도를 높이고 있습니다.

- **Performance Highlights**: FreeEval은 다노드, 다GPU 클러스터(multi-node, multi-GPU clusters)에서 광범위한 평가를 가능하게 하는 고성능 인프라를 갖추고 있어, 오픈 소스 및 독점 LLMs 모두의 평가에 활용할 수 있습니다. 이는 평가 과정에서의 컴퓨팅 비용을 절감하면서도 평가의 확장성과 효율성을 보장합니다.



### Privacy Preserving Prompt Engineering: A Survey (https://arxiv.org/abs/2404.06001)
Comments: 23 pages, 8 figures

- **What's New**: 이 작업은 사전에 학습된 대형 언어 모델(Large Language Models, LLMs)을 사용할 때 발생할 수 있는 프라이버시 문제들과 그 해결 방안을 전문적으로 조사한 논문입니다. 특히, 인-컨텍스트 학습(In-context Learning, ICL)과 프롬프팅(prompting)과 같은 기법을 이용할 때 개인 정보 보호에 초점을 맞추고 있습니다.

- **Technical Details**: 이 논문에서는 대형 언어 모델들이 얼마나 큰 파라미터 사이즈를 가지고 있는지 강조하며, 이러한 모델들이 자연어 처리(Natural Language Processing, NLP) 태스크에서 얼마나 효과적인지 설명합니다. 이러한 모델들은 주로 오픈 소스 텍스트, 위키백과 등에서 다양한 데이터를 수집하여 사전 학습됩니다. 또한, 이 논문은 LLM이 어떻게 프롬프팅과 ICL을 사용하여 모델 파라미터를 변경하지 않고도 주어진 쿼리에 대해 예상 출력을 생성할 수 있는지 설명합니다. 개인 정보 보호 방법으로는 차등 프라이버시(Differential Privacy, DP), 새니타이제이션(sanitization), 암호화(encryption) 등이 있습니다.

- **Performance Highlights**: 이 논문은 ICL과 일반 프롬프팅 동안 개인 정보 보호 방법을 체계적으로 분류하고 리뷰합니다. 제안된 프레임워크들은 크게 비DP(non-DP), 로컬 DP(local DP), 글로벌 DP(global DP), 그리고 기타 시나리오로 분류됩니다. 각 범주에서 개인 정보 보호 메커니즘과 각각의 프라이버시 목표들을 강조하여 설명하고 있습니다. 또한, 이러한 프레임워크의 개발을 위해 접근 가능한 리소스를 요약해두었습니다.



### Event-enhanced Retrieval in Real-time Search (https://arxiv.org/abs/2404.05989)
Comments: LREC-COLING 2024

- **What's New**: 이 논문은 EER(Event-Enhanced Retrieval)이라는 새로운 접근 방식을 제안하여 실시간 검색 시나리오에서 '의미적 표류'(semantic drift) 문제를 해결하고 이벤트 도큐먼트의 검색 성능을 향상시킵니다. 전통적인 EBR (Embedding-Based Retrieval) 듀얼 인코더 모델을 기반으로 하되, 새로운 대조 학습(contrastive learning), 쌍 학습(pairwise learning) 및 프롬프트 조정(prompt-tuning) 기반의 디코더 모듈을 통해 이벤트 중심 정보에 더 집중할 수 있습니다.

- **Technical Details**: EER은 쿼리(query)와 문서 제목(document title)에 대한 듀얼 타워(dual-tower) 모델을 사용합니다. 대조 학습과 쌍 학습을 통해 인코더의 성능을 개선하며, 문서 제목 인코더 뒤에 프롬프트 학습 기반의 디코더 모듈을 추가하여 이벤트 트리플릿을 생성하는 방식으로 중요 이벤트 정보에 집중하도록 설계되었습니다. 이 디코더는 학습 단계에서만 사용되며, 추론 단계에서는 제거할 수 있어 전통적인 듀얼 타워 모델로 복귀하여 지연 시간에 영향을 주지 않습니다.

- **Performance Highlights**: EER은 방대한 실험을 통해 실시간 검색 시나리오에서의 검색 성능이 크게 향상됨을 보여주었습니다. 이 방법은 쿼리와 문서 제목 간의 정보 비대칭성을 해결하고, 이벤트 정보에 집중하여 쿼리와 관련된 이벤트를 효과적으로 검색할 수 있도록 도와줍니다. 또한, EER은 인코딩된 이벤트 정보를 활용하여 쿼리 인코더 최적화를 통한 검색 결과의 정확성을 높이는 데 기여합니다.



### Optimization Methods for Personalizing Large Language Models through  Retrieval Augmentation (https://arxiv.org/abs/2404.05970)
- **What's New**: 이 연구는 대규모 언어 모델(LLMs)을 개인화하는 데 검색 증강(retrieval-augmented) 접근 방식을 사용하여 다양한 응용 프로그램 및 도메인에 상당한 영향을 미칠 수 있는 가능성을 탐구하고 있습니다. 처음으로 개인 문서를 LLM에 제공하는 검색 모델을 최적화하는 방법을 제안합니다. 우리는 두 가지 최적화 알고리즘을 개발했으며, 하나는 개인화 생성(personalized generation)을 위한 임의의 척도를 사용하여 정의된 보상 함수를 기반으로 하는 강화 학습(reinforcement learning)에, 또 다른 하나는 하위 LLM에서 검색 모델로 지식을 전수하는 지식 전수(knowledge distillation)에 기반을 두고 있습니다.

- **Technical Details**: 이 논문은 개인화된 텍스트 생성을 위해 개인 정보를 검색하는 최적화에 초점을 맞추고 있으며, 기존의 학습-순위 최적화 방법(learning-to-rank optimization methods)을 적용할 수 없는 상황에서 두 가지 새로운 접근 방식을 제안합니다. 첫 번째는 강화 학습 방법을 사용하여 사용자 프로필에서 문서를 샘플링하고 이를 LLM에 제공한 다음, 생성된 개인화된 출력 텍스트의 성능을 기반으로 검색 모델을 최적화합니다. 두 번째 방법은 LLM에서 생성된 개별 검색 문서의 성능을 사용하여 타겟 분포를 계산하고 검색 점수 분포와의 차이(divergence)를 최소화하여 검색 모델을 최적화합니다.

- **Performance Highlights**: 이 논문에서 제안된 방법은 LaMP 벤치마크를 사용하여 평가되었으며, 여기에는 세 가지 개인화된 텍스트 분류 작업과 네 가지 개인화된 텍스트 생성 작업이 포함됩니다. 제안된 방법들은 LaMP의 일곱 개 데이터 세트 중 여섯에서 통계적으로 유의미한 개선을 달성했습니다. 최고 성능 방법은 LaMP 데이터 세트 전반에 걸쳐 평균 5.5%의 최신 기술(state-of-the-art) 개선을 보여주었으며, 비개인화된 LLM과 비교할 때 모든 작업에서 평균 15.3%의 개선을 보였습니다.



### THOUGHTSCULPT: Reasoning with Intermediate Revision and Search (https://arxiv.org/abs/2404.05966)
Comments: Code and data available at this https URL

- **What's New**: THOUGHTSCULPT는 결과물이 구성 요소로 분해될 수 있는 작업을 위한 일반적인 추론 및 검색 방법을 제시합니다. 이 방법은 Monte Carlo Tree Search (MCTS)를 이용하여 잠재적인 해결책의 검색 트리를 탐색하고, 도메인별 휴리스틱(heuristic), 특히 LLM 평가자(LLM evaluator)를 사용해 평가합니다. 특히, THOUGHTSCULPT는 기존 출력의 일부를 수정하는 작업을 포함할 수 있습니다.

- **Technical Details**: THOUGHTSCULPT는 LLM (Large Language Models)의 사고 과정을 모방하기 위해 상호 연결된 생각의 네트워크를 구축하는 새로운 그래프 기반 프레임워크입니다. 이 프레임워크는 LLM이 이전의 출력을 반복적으로 정제하고 개선할 수 있는 자가 수정 메커니즘을 특징으로 합니다. THOUGHTSCULPT는 세 가지 주요 모듈, 즉 생각 평가자(thought evaluator), 생각 생성기(thought generator), 그리고 결정 시뮬레이터(decision simulator)로 구성됩니다. 생각 평가자는 피드백을 제공하고, 생각 생성기는 초기 지시와 자체 평가 피드백을 바탕으로 잠재적 해결책을 생산합니다. 결정 시뮬레이터는 MCTS 과정의 일부로서, 특정 경로의 잠재적 가치를 평가하기 위해 연속적인 생각 라인을 시뮬레이션합니다.

- **Performance Highlights**: THOUGHTSCULPT는 스토리 개요 개선(Story Outline Improvement)에서 최대 30%의 흥미도 증가, 미니 크로스워드 해결(Mini-Crosswords Solving)에서 최대 16%의 단어 성공률 증가, 제한된 생성(Constrained Generation)에서 최대 10%의 개념 커버리지 향상을 보여줍니다. 이러한 결과는 다양한 과제에서 THOUGHTSCULPT의 유효성을 강조합니다. 모든 코드는 출판 시 오픈 소스로 제공될 예정입니다.



### LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders (https://arxiv.org/abs/2404.05961)
- **What's New**: 이 연구에서는 LLM2Vec이라는 새로운 비지도 접근 방식을 도입하여, 단순히 디코더 전용 LLM을 강력한 텍스트 인코더로 변환합니다. 이는 양방향 어텐션(bidirectional attention) 활성화, 마스크된 다음 토큰 예측(masked next token prediction, MNTP), 비지도 대조 학습(unsupervised contrastive learning)의 세 가지 단계로 구성됩니다.

- **Technical Details**: LLM2Vec는 양방향 어텐션을 사용하여 각 토큰이 시퀀스 내의 모든 다른 토큰에 접근할 수 있게 하고, MNTP를 변형된 훈련 목표로 사용하여 입력 시퀀스의 마스킹된 토큰을 예측합니다. 또한, SimCSE를 통해 입력 시퀀스를 독립적으로 샘플링된 드롭아웃 마스크로 두 번 전달하여 서로 다른 두 개의 표현을 생성하고, 이 두 표현이 서로 가깝도록 최적화하며 다른 시퀀스와는 멀어지도록 학습합니다.

- **Performance Highlights**: LLM2Vec을 적용한 모델은 단어 수준 작업에서 인코더 전용 모델(encoder-only models)을 큰 차이로 능가하며, Massive Text Embeddings Benchmark(MTEB)에서 비지도 학습된 모델로는 새로운 최고 기록을 세웠습니다. 공개 데이터만을 사용하여 훈련된 모델 중에서는 MTEB에서 최고 성능을 달성했습니다.



### VisualWebBench: How Far Have Multimodal LLMs Evolved in Web Page  Understanding and Grounding? (https://arxiv.org/abs/2404.05955)
- **What's New**: 이 논문에서는 웹 도메인에서 다중모드 대규모 언어 모델(MLLMs)의 능력을 평가하기 위해 VisualWebBench라는 새로운 벤치마크를 소개합니다. VisualWebBench는 웹 관련 작업을 위해 필요한 핵심 능력을 평가하는 일곱 가지 과제로 구성되어 있어, 웹 페이지의 고유한 특성을 반영합니다. 이 벤치마크는 1.5K의 인간이 큐레이팅한 인스턴스를 포함하고 있으며 139개의 실제 웹사이트와 87개의 하위 도메인을 다룹니다.

- **Technical Details**: VisualWebBench는 웹사이트 이해 및 기반이 되는 능력을 평가하는 다중 과제(multi-task) 벤치마크입니다. 과제에는 캡션(captioning), 웹페이지 QA(Question Answering), 헤딩 OCR(OCR of headings), 요소 OCR(OCR of elements), 요소 기반(grounding), 조치 예측(action prediction), 그리고 조치 기반(action grounding)이 포함됩니다. 벤치마크는 웹페이지의 복잡한 시각적 및 텍스트 정보의 상호작용을 반영하도록 설계되었으므로, 고성능 MLLMs에 대한 중요한 평가 도구로 기능합니다.

- **Performance Highlights**: 14개의 오픈 소스 MLLMs와 Gemini Pro, Claude-3 시리즈, 그리고 GPT-4V를 VisualWebBench에서 평가했습니다. 가장 높은 성능을 보인 모델은 평균 점수 65.8을 기록한 Claude Sonnet이었고, GPT-4V는 64.6 점을 받았습니다. 이는 MLLMs가 아직 웹 도메인에서의 고도화된 작업을 처리하는 데 있어 중대한 도전이 남아 있음을 시사합니다. 특히, 낮은 해상도 이미지 처리와 텍스트-풍부한 환경에서의 정확한 기반(grounding) 능력이 부족함이 발견되었습니다.



### Interplay of Machine Translation, Diacritics, and Diacritization (https://arxiv.org/abs/2404.05943)
Comments: Accepted to NAACL 2024 Main Conference

- **What's New**: 이 연구에서는 기계 번역(MT; Machine Translation)과 발음 부호 지정(diacritization)이 서로의 성능에 어떤 영향을 미치는지 다루고 있습니다. 특히 두 가지 주요 질문을 탐구합니다: 1) 다작업 학습 환경에서 발음 부호 지정이 기계 번역 성능에 미치는 영향, 2) 발음 부호를 유지하거나 제거하는 것이 기계 번역 성능에 미치는 영향. 연구는 자원이 풍부한 환경(HR; High-Resource)과 자원이 부족한 환경(LR; Low-Resource)에서 55개 언어(36개 아프리카 언어와 19개 유럽 언어)에 걸쳐 진행되었습니다.

- **Technical Details**: 이 다작업 학습 설정에서 기계 번역과 발음 부호 지정을 동시에 수행하는 모델(DiaMT)을 훈련하여, 발음 부호가 기계 번역 성능에 미치는 영향을 분석합니다. 실험 결과, 발음 부호 지정이 자원이 부족한 환경에서 기계 번역 성능을 두 배나 세 배까지 향상시키는 반면, 자원이 풍부한 환경에서는 해를 끼칠 수 있음을 보여줍니다. 또한, 발음 부호를 유지하거나 제거하는 것이 기계 번역 성능에 큰 차이를 미치지 않는 것으로 나타났습니다.

- **Performance Highlights**: 연구 결과에 따르면, 발음 부호 지정은 자원이 부족한 환경에서 기계 번역의 성능을 크게 향상시킬 수 있습니다. 다만, 자원이 풍부한 환경에서는 그렇지 않을 수 있습니다. 발음 부호 시스템의 복잡성을 측정하는 두 가지 언어 독립적(metric) 지표를 제안하였으며, 이러한 지표들은 발음 부호 지정 모델의 성능과 긍정적인 상관관계를 보였습니다.



### The Hallucinations Leaderboard -- An Open Effort to Measure  Hallucinations in Large Language Models (https://arxiv.org/abs/2404.05904)
- **What's New**: 이 논문은 대규모 언어 모델(LLM: Large Language Models)의 '환각(hallucinations)' 경향을 정량적으로 측정하고 비교하는 ‘환각 리더보드(Hallucinations Leaderboard)’를 소개합니다. 환각은 모델이 사실과 다르거나 입력 컨텍스트와 일치하지 않는 출력을 생성하는 현상을 말합니다. 이 리더보드는 다양한 태스크에 걸쳐 환각의 여러 측면을 집중적으로 평가합니다.

- **Technical Details**: 리더보드는 질문 응답(Question-Answering), 요약(Summarisation), 독해(Reading Comprehension) 등의 태스크를 포함하여 텍스트 생성과 지식 집약적인 다운스트림 태스크에 대한 LLM의 환각 경향을 평가합니다. 그중 신뢰성(faithfulness)과 사실성(factuality)에 초점을 맞춘 두 가지 시나리오를 구분하여 분석합니다. 또한, LLM의 성능을 '사실성 점수(factuality score)'와 '신뢰성 점수(faithfulness score)'로 측정하여 제공합니다.

- **Performance Highlights**: 다양한 LLM들이 15개의 태스크에서 평가되었으며, 모델 간 및 태스크 간 성능 차이를 공개하여 연구자들이 보다 신뢰할 수 있는 모델을 선택할 수 있도록 돕습니다. 이 결과는 LLM의 현재 능력과 한계를 이해하는데 중요하며, 환각 문제를 해결하는 방향으로 LLM의 발전을 촉진할 것입니다.



### WILBUR: Adaptive In-Context Learning for Robust and Accurate Web Agents (https://arxiv.org/abs/2404.05902)
- **What's New**: Wilbur이라는 새로운 웹 에이전트 접근 방식을 소개하며, 기존 기법들이 다양한 웹사이트에서 일반화하는데 실패하는 문제를 해결하고자 한다. Wilbur은 차별화된 순위 모델과 새로운 지시 생성 기술을 사용하여 블랙박스 큰 언어 모델(Large Language Model, LLM)의 프롬프트에 최적화된 작업 데모를 포함시킨다. 또한, 실패에서 회복하고 학습하는 지능적 되돌리기(backtracking) 메커니즘을 제안한다.

- **Technical Details**: Wilbur은 다음과 같은 두 가지 주요 기능을 갖추고 있다: 1) 탐색, 반성 및 되돌리기: 새로운 웹사이트에서 Wilbur은 LLM에서 추출한 작업을 수행하고, 새로운 페이지 상태를 관찰한 후 반성 언어 모델(reflection LM)을 통해 그 작업이 목표 달성에 기여하는지 검증한다. 검증에 실패하면, 이전의 성공적인 상태로 동적으로 되돌아간다. 2) 확장 가능한 지식 은행에서 데모를 검색: 목표 조건의 데모와 웹사이트 조건의 데모를 포함하여 Wilbur이 일반화 할 수 있도록 돕는다. 이러한 기능은 제한된 LLM 컨텍스트 창문 내에서 도움이 되는 데모를 선택하기 위해 훈련된 데모 순위 모델을 사용하여 최적화된다.

- **Performance Highlights**: Wilbur은 WebVoyager 벤치마크에서 최고 성과를 달성하여 이전의 텍스트만을 사용하는 모델보다 전반적으로 8%, 특정 웹사이트에서는 최대 36%까지 성과가 향상되었다. 이는 강력한 멀티모달 모델과 비교하여도 5% 이내의 차이로, Wilbur이 텍스트 입력만을 받음에도 불구하고 유의미한 결과를 보여주었다. 또한, 실패의 상당수는 웹 운영의 공학적 도전 때문에 발생하는 것으로 분석되었다.



### Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrenc (https://arxiv.org/abs/2404.05892)
- **What's New**: 새롭게 선보인 Eagle (RWKV-5) 및 Finch (RWKV-6) 은 기존 RWKV (RWKV-4) 아키텍처를 개선한 시퀀스 모델입니다. 이들은 멀티-헤드 매트릭스 형태의 상태와 동적 재발 메커니즘을 활용하여 표현력을 향상시키는 동시에 RNN의 추론 효율성을 유지합니다. 또한, 1.12조 토큰으로 구성된 새로운 다국어 코퍼스와 빠른 토크나이저를 도입하여 다국어 지원을 강화했습니다.

- **Technical Details**: Eagle 모델은 0.46억에서 7.5억 매개변수(Parameter) 범위로 네 가지가, Finch 모델은 1.6억 및 3.1억 매개변수로 두 가지가 개발되었습니다. 모든 모델은 Apache 2.0 라이선스 하에 HuggingFace에 공개되었으며, 멀티-헤드(matrix-valued states) 및 동적 재발(dynamic recurrence) 기능을 포함하고 있습니다.

- **Performance Highlights**: Eagle 및 Finch 모델은 다양한 벤치마크(benchmarks)에서 경쟁력 있는 성능을 달성했습니다. 모델은 다국어 처리 능력을 강조하며, 새로운 토크나이저는 탐욕적 일치(greedy matching) 기반으로 빠른 처리 속도를 제공합니다.



### Eraser: Jailbreaking Defense in Large Language Models via Unlearning  Harmful Knowledg (https://arxiv.org/abs/2404.05880)
- **What's New**: 이 논문에서는 대규모 언어 모델(Large Language Models, LLM)의 해로운 지식을 제거하고 일반 지식을 유지하며 안전성을 보장하는 새로운 방어 메커니즘인 'Eraser'를 제안합니다. 이 방법은 LLM이 해로운 질문에 대답할 수 있는 능력을 잊어버리도록 하여, jailbreaking 공격의 성공률을 크게 줄일 수 있습니다.

- **Technical Details**: Eraser 방법은 해로운 지식을 잊는 것(unlearning harmful knowledge), 일반 지식을 유지하는 것(retaining general knowledge), 그리고 안전성 정렬을 유지하는 것(maintaining safety alignment)의 세 가지 주요 목표를 포함합니다. 이를 위해 gradient ascent 기법을 사용하여 해로운 답변에 대한 학습을 지우고 (unlearn), 일반적인 질문에 대한 이해 능력은 유지합니다.

- **Performance Highlights**: Eraser를 적용한 결과, 다양한 jailbreaking 공격에 대한 성공률이 크게 감소했으며, 이는 일반 업무 수행 능력에는 영향을 미치지 않는 것으로 나타났습니다. 실험 결과는 우수한 방어 능력과 함께 일반적인 능력을 유지하는 것을 입증했으며, 기존 방법들과 비교할 때 해로움과 유용성 사이의 더 나은 트레이드오프(trade-off)를 보여줍니다.



### CodecLM: Aligning Language Models with Tailored Synthetic Data (https://arxiv.org/abs/2404.05875)
Comments: Accepted to Findings of NAACL 2024

- **What's New**: CodecLM은 대규모 언어 모델(LLM)의 지시 사항 준수 능력을 향상시키기 위해 고품질의 맞춤형 합성 데이터를 생성하는 새로운 프레임워크입니다. 이 프레임워크는 인코딩(Encode)과 디코딩(Decode) 원리를 활용하여 특정 하류 지시 분포와 LLM에 맞춤형 데이터를 생성합니다. 또한, Self-Rubrics와 Contrastive Filtering을 도입하여 데이터 효율을 극대화하며, 실제 지시 사항 따르기 벤치마크에서 최신 기술(State-of-the-art) 성능을 보여줍니다.

- **Technical Details**: CodecLM은 강력한 LLM을 사용하여 타겟 작업의 시드(seed) 지시를 메타데이터로 인코딩하고, 이 메타데이터를 디코드하여 맞춤형 지시를 생성합니다. 메타데이터는 입력 지시 분포의 단어 수준 추상화를 포함하며, 유저가 제공하거나 자동 생성될 수 있습니다. 생성된 지시는 Self-Rubrics를 통해 복잡성을 조정하고, Contrastive Filtering으로 가장 효과적인 지시-응답 쌍을 선별합니다.

- **Performance Highlights**: CodecLM은 4개의 오픈 도메인 지시 따르기 벤치마크에서 다양한 LLM 선택으로 그 효과를 입증하였으며, 기존의 최신 기술을 능가하는 성능을 보였습니다. 특히, 다양한 지시 분포에 대한 LLM의 정렬을 위해 맞춤형 데이터 생성이 중요함을 강조하며, 이를 실현하기 위한 혁신적인 접근 방식을 제공합니다.



### GeniL: A Multilingual Dataset on Generalizing Languag (https://arxiv.org/abs/2404.05866)
- **What's New**: 이 연구에서는 일반화(generalization) 탐지라는 새로운 작업을 소개하고 있습니다. 저자들은 GeniL이라는 다국어 데이터셋을 구축하여 언어에서의 일반화를 탐지하고 평가하는 데 중요한 도구를 제공합니다. 이 데이터셋은 영어, 아랍어, 벵골어, 스페인어, 프랑스어, 힌디어, 인도네시아어, 말레이어, 포르투갈어의 9가지 언어로 구성된 50,000개 이상의 문장을 포함하고 있습니다.

- **Technical Details**: 연구팀은 일반화를 언급하는 문장과 일반화를 촉진하는 문장을 구분합니다. 이를 통해 언어 기술이 스테레오타입(stereotype)을 확대하는 것을 방지하기 위한 기술이 필요함을 강조합니다. GeniL 데이터셋은 언어의 다양한 문맥에서 일반화의 사례를 어떻게 해석해야 하는지에 대한 깊은 이해를 가능하게 합니다. 또한, 일반화를 탐지하기 위해 구축된 분류기는 PR-AUC (Precision-Recall Area Under the Curve)가 58.7이며, 언어에 따라 성능 차이가 있습니다.

- **Performance Highlights**: 분류기의 성능은 언어별로 다양하지만 전반적인 PR-AUC는 58.7로서, 다양한 언어와 정체성 그룹, 속성에 따라 일반화 인스턴스의 가능성이 낮음을 보여줍니다. 이는 각 언어와 사회적 맥락에서 스테레오타입이 어떻게 나타나는지에 대한 더욱 세밀한 이해와 접근이 필요함을 시사합니다.



### ÚFAL LatinPipe at EvaLatin 2024: Morphosyntactic Analysis of Latin (https://arxiv.org/abs/2404.05839)
Comments: Accepted to EvaLatin 2024

- **What's New**: LatinPipe는 EvaLatin 2024 종속성 파싱 공유 작업에서 1등과 2등을 차지한 입니다. 이 시스템은 기존의 UDPipe 모델을 발전시킨 것으로, 사전 훈련된 언어 모델(pre-trained LMs)을 미세 조정하여 종속성 파싱과 형태학 분석을 동시에 학습할 수 있도록 하였습니다. 복수의 언어 모델을 연결하고, 변환기(Transformer) 위에 BiLSTM 계층을 추가하여 언어 처리 능력을 향상시켰습니다.

- **Technical Details**: LatinPipe는 기본적으로 그래프 기반의 종속성 파서로, 미리 훈련된 언어 모델을 활용하여 파싱 작업을 위한 점곱(dot-product) 주의력 헤드와 형태학 분류를 위한 소프트맥스 분류 헤드를 사용합니다. 이 시스템은 여러 사전 훈련된 모델의 연결(concatenation)과, 트랜스포머 구조 위에 BiLSTM 층을 쌓아 추가적인 컨텍스트화를 구현합니다. 이러한 구조는 토큰 간의 상대적인 단거리 관계를 더 잘 모델링할 수 있게 해줍니다. 훈련 초기에는 트랜스포머 가중치를 고정하여 추가적인 최적화를 용이하게 합니다.

- **Performance Highlights**: LatinPipe는 7개의 공개 라틴어 코퍼스(corpus)에서 샘플링하여 학습되었으며, 각 코퍼스 간의 주석 스타일을 조화(harmonization)하기 위해 노력하였습니다. 이를 통해 파싱 성능이 크게 향상되었음을 확인할 수 있었습니다. 높은 정확도와 뛰어난 일반화 능력을 갖추고 있으며, 다양한 훈련 설정에서도 안정적인 성능을 보여주었습니다.



### SambaLingo: Teaching Large Language Models New Languages (https://arxiv.org/abs/2404.05829)
Comments: 23 pages

- **What's New**: 이 논문은 대용량 언어 모델(LLM: Large Language Models)을 새로운 언어로 적응시키는 과정에 대한 포괄적인 연구를 제시합니다. 기존의 영어 중심 모델들을 다양한 언어, 즉 아랍어, 타이어, 터키어, 일본어, 헝가리어, 러시아어, 불가리아어, 세르비아어, 슬로베니아어로 확장하는 방법론을 다루고 있습니다. 이 연구는 사전 훈련된 모델을 기반으로 시작하여 토크나이저(tokenizer)의 어휘를 확장하고, 대상 언어로의 학습 불균형을 해결하기 위한 방법을 탐구합니다.

- **Technical Details**: 연구팀은 Llama 2 모델을 기반으로, 파라미터 스케일 7B와 70B에서의 언어 적응 실험을 실시했습니다. 어휘 확장(vocabulary extension), 직접 선호 최적화(Direct Preference Optimization, DPO), 타겟 언어의 데이터 부족 해결을 주요 연구 내용으로 다룹니다. SFT(Supervised Fine-Tuning)와 DPO를 사용하여 챗(chat)-조정된 모델 버전을 훈련시키는 이중 단계 접근 방식을 채택했습니다. 연구 결과, 이 방법론은 기존의 다국어 모델 및 특정 언어 전용 모델보다 우수한 성능을 보였습니다.

- **Performance Highlights**: 이 방법론을 사용하여 개발된 모델은 기존의 다언어 모델이나 특정 언어로 사전 훈련된 모델들을 뛰어넘는 성능을 보였습니다. 특히 토크나이저의 효율성을 개선하고, 학습 데이터의 불균형 문제를 해결하는 데 큰 진전을 이루었습니다. 이 연구는 FLORES-200, SIB-200, EXAMS 및 다언어 단어성 퍼플렉시티(multilingual perplexity) 벤치마크를 통해 평가되었습니다.



### InternLM-XComposer2-4KHD: A Pioneering Large Vision-Language Model  Handling Resolutions from 336 Pixels to 4K HD (https://arxiv.org/abs/2404.06512)
Comments: Code and models are publicly available at this https URL

- **What's New**: InternLM-XComposer2-4KHD는 처음으로 4K HD (3840 x 1600) 및 그 이상까지 LVLM(Large Vision-Language Model)의 해상도 기능을 확장하였습니다. 이 모델은 336픽셀부터 4K 스탠다드까지 다양한 해상도를 지원함으로써 실제 환경에서의 배포를 용이하게 합니다. 또한 동적 해상도 및 자동 패치 구성을 통해 다양한 환경에서 이미지를 효과적으로 처리할 수 있는 새로운 방법론을 제안합니다.

- **Technical Details**: InternLM-XComposer2-4KHD는 자동 패치 구성이 가능한 동적 해상도를 특징으로 합니다. 이미지의 원본 가로세로 비율을 유지하면서 패치 수와 레이아웃을 자동으로 조정하여 트레이닝 해상도를 336 픽셀에서 4K 스탠다드 이상으로 증가시킵니다. 이는 Vision Transformer (ViT) (336 x 336)을 기반으로 하여 진행됩니다. 패치 토큰의 레이아웃을 명확하게 구분하기 위해 각 패치 행 후에 newline 토큰을 도입해 성능을 향상시킵니다.

- **Performance Highlights**: InternLM-XComposer2-4KHD는 16개 벤치마크 중 10개에서 GPT-4V와 Gemini Pro를 뛰어넘는 성능을 보여 줍니다. 특히 5개의 HD-OCR 데이터셋(DocVQA, ChartQA, InfographicVQA, TextVQA, OCRBench)에서는 기존의 오픈소스 LVLM 모델들을 크게 앞지르며 뛰어난 성능을 입증했습니다.



### Large Language Models to the Rescue: Deadlock Resolution in Multi-Robot  Systems (https://arxiv.org/abs/2404.06413)
- **What's New**: 이 논문은 대규모 다중 에이전트 로봇 시스템(Multi-agent Robotic Systems, MRS)에서 흔히 발생하는 교착 상태(Deadlock) 문제를 해결하기 위해 대형 언어 모델(Large Language Models, LLMs)을 사용하는 새로운 접근법을 제안합니다. 특히, GPT3.5를 사용하여 교착 상태를 해결하고자 하는 고수준 계획기(High-level Planner) 역할을 수행하는 것이 특징입니다.

- **Technical Details**: 논문은 계층적 제어 구조(Hierarchical Control Framework)를 도입하여 LLM이 리더와 무브먼트 방향을 결정하고, 이를 바탕으로 그래프 신경망(Graph Neural Network, GNN) 기반의 저수준 제어 정책(Low-Level Control Policy)이 실행됩니다. 연구팀은 다양한 프롬프팅 기술(Prompting Techniques)을 시험하여 LLM의 성능을 개선하고자 하며, 특히 인-컨텍스트 예제(In-Context Examples)를 활용한 프롬프트 엔지니어링(Prompt Engineering)에 중점을 둡니다.

- **Performance Highlights**: 다양한 다중 로봇 환경에서 최대 15개의 에이전트와 40개의 장애물을 포함한 실험을 통해, LLM 기반의 고수준 계획기가 MRS의 교착 상태 해결에 효과적임을 입증했습니다. 이러한 접근 방식은 교착 상태 감지시 LLM에 의해 결정된 리더와 방향이 장애물 환경에서의 교착 상태를 해결하여 저수준 컨트롤러가 목표를 향해 계속 진행할 수 있도록 돕습니다.



### AgentQuest: A Modular Benchmark Framework to Measure Progress and  Improve LLM Agents (https://arxiv.org/abs/2404.06411)
Comments: Accepted at the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT 2024)

- **What's New**: AgentQuest는 복잡한 다단계 추론 작업을 해결하기 위해 고안된 Large Language Models (LLMs)을 활용하는 에이전트들의 벤치마킹과 평가를 위한 새로운 프레임워크입니다. 이 도구는 벤치마크와 메트릭이 모듈형이며, 쉽게 확장 가능한 API를 통해 사용자 지정이 용이하다는 특징을 가집니다. AgentQuest는 또한 진행률(progress rate)과 반복률(repetition rate)과 같은 새로운 평가 메트릭을 제공하여 LLM 에이전트의 작업 수행 진행 상황을 신뢰성 있게 추적할 수 있습니다.

- **Technical Details**: AgentQuest는 다양한 벤치마크와 에이전트 아키텍처를 지원하는 모듈형 프레임워크로, standard interface를 통해 임의의 에이전트 아키텍처를 다양한 벤치마크와 연결하고 메트릭을 계산할 수 있습니다. 이 프레임워크는 Python interface를 사용하여 에이전트와 환경 간의 상호 작용을 단순화하며, Observation과 Action이라는 두 가지 기본 클래스를 제공합니다. AgentQuest는 기존의 벤치마크와의 통합을 간소화하고, 새로운 벤치마커의 추가를 용이하게 합니다.

- **Performance Highlights**: 주요 벤치마크로는 ALFWorld, Lateral Thinking Puzzles, Mastermind 및 Sudoku가 있으며, 특히 Mastermind 벤치마크에서 제안된 메트릭을 사용하여 에이전트 아키텍처의 성능을 개선할 수 있었습니다. 진행률과 반복률 메트릭을 사용하여 Mastermind에서 약 20%의 성공률 향상을 이끌어 냈습니다. 또한, ALFWorld에서는 진행률을 모니터링하여 에이전트의 최종 성공률이 허용된 런타임에 의해 제한되어 있음을 밝혔고, Sudoku 벤치마크에서는 낮은 성공률이 낮은 진행률과 연관되어 있음을 확인, 에이전트가 해당 유형의 작업을 해결하는 데 어려움이 있음을 명확히 했습니다.



### Wu's Method can Boost Symbolic AI to Rival Silver Medalists and  AlphaGeometry to Outperform Gold Medalists at IMO Geometry (https://arxiv.org/abs/2404.06405)
Comments: Work in Progress. Released for wider feedback

- **What's New**: 이 연구에서는 기하학 문제의 자동정리증명(Automated Theorem Proving)에 대한 새로운 접근 방식을 고찰합니다. 특히, AlphaGeometry와 Wu's method를 결합하여 IMO 수준의 문제를 해결하는 새로운 방법을 제시했습니다. 이 새로운 조합은 기하학 문제 30개 중 27개를 해결하여 국제 수학 올림피아드의 금메달리스트 수준을 초과했습니다.

- **Technical Details**: Wu's method는 다양한 차원의 기하학에서 응용될 수 있는 대수적 방법으로, 기하학적 가설을 다항식 시스템으로 변환합니다. AlphaGeometry는 신경기호학적(neuro-symbolic) 알고리즘으로, 1억 개의 학습 데이터를 기반으로 개발되었습니다. 이 두 방법을 결합하여 'Wu&AG' 모델을 개발하여, 합성 기법(synthetic methods) 사용과 대규모 언어 모델(LLM)의 건축(constructions)을 통합했습니다.

- **Performance Highlights**: 추가 실험은 커스텀 AlphaGeometry 코드베이서 흔히 사용되는 DD+AR(다소적 데이터베이스와 각도, 비율, 거리 추적) 분석 방식을 통해 이루어졌습니다. Wu's method는 기하학 문제 30개 중 15개를 독립적으로 해결하였으며, DD+AR 방식과 결합한 'Wu&DD+AR'는 21개 문제를 해결함으로써 AlphaGeometry의 성능과 비슷한 결과를 보였습니다. 이는 CPU만을 사용하는 노트북에서도 높은 성능을 달성할 수 있음을 보여줍니다.



### Model Generation from Requirements with LLMs: an Exploratory Study (https://arxiv.org/abs/2404.06371)
- **What's New**: 이 연구는 자연 언어(NL) 요구 사항을 기반으로 UML(Unified Modeling Language) 시퀀스 다이어그램을 생성하는 것을 목표로하는 ChatGPT의 능력을 평가합니다. 이 연구는 LLM(large language models)을 사용하여 요구 공학(RE) 과정에서 자동 모델 생성을 지원하는 것과 그 한계를 탐구합니다.

- **Technical Details**: 연구에서는 28개의 다양한 도메인과 서식의 요구 사항 문서를 사용하여 ChatGPT에 의한 시퀀스 다이어그램 생성을 테스트하였습니다. 연구자들은 ‘shall’ 스타일, 사용자 스토리, 유즈 케이스 명세와 같은 다른 형식의 요구 사항으로 ChatGPT를 프롬프트하고 다양한 베리언트를 소개하여 실제 시나리오를 시뮬레이션하였습니다. 생성된 다이어그램의 품질은 다양한 기준에 따라 평가되었고, 공통적인 문제는 요구 사항의 모호성이나 일관성 결여 같은 ‘스멜(smell)'의 존재 할 때 더욱 두드러졌습니다.

- **Performance Highlights**: 생성된 시퀀스 다이어그램들은 이해하기 쉽고, 표준에 부합하며, 요구 사항과 용어가 일치하는 점에서 좋은 평가를 받았습니다. 그러나, 다이어그램의 완성도와 정확성에는 도전이 있었으며, 특히 요구 사항의 낮은 품질, 기술적/맥락적 지식이 필요한 경우 문제가 더욱 심각하게 나타났습니다. 이러한 결과를 바탕으로, 효과적인 모델 생성을 위한 RE 특화 프롬프팅 전략과 암묵적/도메인 지식 문제를 해결할 필요성에 대한 미래 연구의 방향이 제시되었습니다.



### Dimensionality Reduction in Sentence Transformer Vector Databases with  Fast Fourier Transform (https://arxiv.org/abs/2404.06278)
Comments: 13 pages, 5 figures

- **What's New**: 벡터 데이터베이스의 차원 축소는 AI 데이터 관리를 간소화하는 데 필수적이며, 효율적인 저장, 빠른 계산 및 개선된 모델 성능을 가능하게 합니다. 이 논문에서는 계산 효율성 향상과 차원의 저주(curse of dimensionality)를 극복하기 위한 벡터 데이터베이스 차원 축소의 이점을 탐구하며, 이전에는 이 맥락에서 활용되지 않았던 고속 푸리에 변환(Fast Fourier Transform, FFT)을 차원 축소에 적용하는 새로운 방법을 소개합니다.

- **Technical Details**: FFT를 사용하여 차원을 축소함으로써, 검색 기반 생성(Retrieval-Augmented Generation, RAG) 모델과 이미지 처리 등 다양한 AI 도메인에서의 유용성을 보여줍니다. FFT 기반 접근 방식은 데이터 검색 과정을 개선하고 AI 솔루션의 효율성 및 확장성을 증대시킬 것으로 예상됩니다. FFT의 도입은 실시간 처리 및 추천 시스템에서의 작업 최적화뿐만 아니라 고급 이미지 처리 기술에도 확장될 수 있으며, 여기에서 차원 축소는 성능과 분석 효율성을 크게 향상시킬 수 있습니다.

- **Performance Highlights**: 이 FFT 기반 방식은 테스트 입력 처리 후 모델에서 생성된 임베딩 벡터(embedding vectors)를 직접 처리함으로써 기존의 다른 접근 방식과 차별화됩니다. 이를 통해 데이터 볼륨 및 복잡성에 관련된 AI 연구 및 응용 분야의 도전을 해결하는 데 중요한 발전을 이루고 있습니다.



### Elephants Never Forget: Memorization and Learning of Tabular Data in  Large Language Models (https://arxiv.org/abs/2404.06209)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)이 테이블 형태의 데이터를 학습하는 과정에서 데이터 오염과 암기 문제를 얼마나 겪고 있는지 조사합니다. 특히, 훈련 데이터 세트가 모델에 의해 보았던 데이터인지를 평가하는 다양한 기술을 소개하며, 이를 통해 LLMs가 일부 인기 있는 테이블 데이터 세트를 정확히 암기하고 있음을 밝혀냈습니다. 메모리 활용을 평가하기 위한 새로운 방법론과 이를 가능하게 하는 'tabmemcheck' Python 패키지를 개발하였습니다.

- **Technical Details**: 연구팀은 GPT-3.5와 GPT-4 모델을 사용하여 테이블 데이터를 기반으로 few-shot 학습 성능을 평가했습니다. 데이터 세트가 모델 훈련 중에 노출되었는지 여부를 확인하기 위해 네 가지 다른 메모리 테스트를 개발하였습니다. 또한, 훈련된 데이터에 노이즈를 추가하는 등의 변형을 통해 데이터 오염이 few-shot 학습 성능에 미치는 영향을 분석했습니다.

- **Performance Highlights**: LLMs는 훈련 중에 본 데이터 세트에서 더 높은 성능을 나타냄으로써 기억력이 과적합 (overfitting)으로 이어질 수 있음을 시사합니다. 그러나 실험 결과에 따르면 GPT-3.5와 GPT-4는 새로운 데이터 세트에서도 비교적 양호한 성능을 보였으며, few-shot 예제의 수가 증가함에 따라 GPT-4의 성능은 개선되는 반면 GPT-3.5는 비슷한 수준을 유지했습니다. 이는 LLMs의 세계 지식이 새로운 테이블 데이터 세트에 대한 few-shot 학습 성능에 주요 영향을 미치고 있음을 나타냅니다.



### AiSAQ: All-in-Storage ANNS with Product Quantization for DRAM-free  Information Retrieva (https://arxiv.org/abs/2404.06004)
Comments: 5 pages, 6 figures and 4 tables

- **What's New**: 본 연구에서는 모든 인덱스 데이터를 저장 장치에 오프로드하는 새로운 근사 최근접 이웃 탐색(ANNS) 방법 'AiSAQ (All-in-Storage ANNS with Product Quantization)'을 제안합니다. AiSAQ은 기존 DiskANN과 비교하여 무시할만한 성능 저하로 훨씬 적은 메모리(~10MB)를 사용하며, 여러 대규모 데이터셋 간의 인덱스 전환을 밀리초 단위로 실행할 수 있습니다, 이는 검색-증강 생성(RAG)의 유연성을 크게 향상시킵니다.

- **Technical Details**: AiSAQ은 기존의 그래프 기반 ANNS 알고리즘에 적용 가능하며, 데이터를 SSD와 같은 저장 장치에 최적화하여 배치함으로써, 저장 공간의 큰 비중이 저장 장치에 위치함에도 불구하고 DiskANN에 비해 미미한 지연 시간 증가만을 가져옵니다. 또한, 이 방법은 다중 빌리언-스케일 데이터셋에서 고속으로 인덱스를 전환할 수 있도록 설계되었습니다.

- **Performance Highlights**: AiSAQ은 SIFT1B와 같은 빌리언-스케일 데이터 세트에서도 RAM 사용량을 10MB로 유지하면서, 높은 재현율과 원래의 DiskANN의 그래프 토폴로지를 유지합니다. 또한, SSD에서의 데이터 배치 최적화 덕분에 95% 이상의 1-recall@1 지표를 밀리초 단위 대기 시간으로 달성할 수 있으며, 인덱스 로드 시간이 무시할 정도로 빠릅니다.



### AEGIS: Online Adaptive AI Content Safety Moderation with Ensemble of LLM  Experts (https://arxiv.org/abs/2404.05993)
- **What's New**: 이 연구에서는 최신의 대규모 언어 모델(LLMs)의 내용 안전성 관리를 위한 새로운 접근 방식을 제안합니다. 특히, 고도로 정교한 'AegisSafetyDataset'와 'AegisSafetyExperts' 모델 스위트를 개발하여, 기존보다 향상된 안전성 평가 및 모델링을 가능하게 했습니다. 이 데이터와 모델은 다양한 안전 위험 평가에 걸쳐 높은 적응성과 강건성을 보여주며, LLMs의 동적 적응 가능한 온라인 내용 조정 프레임워크인 'AEGIS'를 통해 실시간으로 최적의 안전 전문가를 선택합니다.

- **Technical Details**: 연구팀은 13개의 주요 위험 카테고리와 추가적인 9개의 하위 카테고리를 포함하는 내용 안전 위험 분류(taxonomy)를 정의하고, 26,000건 이상의 인간과 LLM 상호작용 인스턴스를 포함하는 'AegisSafetyDataset'을 구축했습니다. 이 데이터셋은 다양한 'jail-break attack'(Jail-Break Attack) 카테고리에서 강건함을 보여주는 여러 LLM 기반 안전 모델을 효과적으로 학습시키는 데 사용되었습니다. 또한, 이 연구에서는 이론적 보증이 강한 노 리그레트(No-Regret) 온라인 적응 프레임워크를 적용하여 'AEGIS' 콘텐츠 조정 시스템을 제안합니다.

- **Performance Highlights**: ‘AegisSafetyExperts’는 기존의 LLM 기반 안전 모델과 일반적인 목적의 LLMs 대비 우수한 성능 또는 경쟁력 있는 성능을 제공함을 보여주었습니다. 특히, 다양한 안전 정책에 적응하고 새로운 위험에 대응하는 능력이 뛰어납니다. 'AEGIS' 적응 프레임워크를 사용하는 것이 MT Bench 점수에 미치는 영향이 없음을 입증함으로써, 모델 정렬 단계에서의 성능 저하 문제를 해소했습니다.



### Does Transformer Interpretability Transfer to RNNs? (https://arxiv.org/abs/2404.05971)
- **What's New**: 최근 Mamba와 RWKV와 같은 재귀 신경망(RNN) 아키텍처의 발전으로, 해당 RNN들이 언어 모델링 혼란도(perplexity) 및 하위 태스크 평가에서 동일 크기의 트랜스포머(transformers)의 성능을 맞추거나 초과할 수 있게 되었습니다. 본 논문에서는, 처음에는 트랜스포머 언어 모델을 위해 설계된 인터프리터빌리티(interpretability) 메소드들이 이러한 새로운 RNN 아키텍처로 전환될 수 있는지를 검토합니다. 특히, 모델 출력을 조정하는 대조 활성화 추가(contrastive activation addition, CAA), 조정된 렌즈(tuned lens)를 통해 잠재적 예측을 이끌어내는 것, 그리고 특정 조건 하에 거짓 출력을 생성하도록 세밀하게 조정된(fine-tuned) 모델로부터 잠재 지식을 이끌어내는 기법들에 초점을 맞춥니다.

- **Technical Details**: 이 연구는 Mamba와 RWKV v5 RNN 아키텍처를 중심으로 구조화되어 있으며, 해당 모델들은 HuggingFace Hub에서 사전 훈련된 강력한 모델로 제공됩니다. 본 논문에서는 대조 활성화 추가(CAA)와 같은 기존 트랜스포머 모델의 인터프리터빌리티 기법이 RNN에도 효과적이며, 내부 상태(state) 조정을 통해 개선될 수 있다는 가설을 검증합니다. 추가적으로, RNN의 압축된 상태를 이용한 새로운 상태 조정(state steering) 방법을 소개하고, 이를 통해 더 효과적인 조정이 가능하다는 점을 논의합니다.

- **Performance Highlights**: Mamba 2.8b-slimpj와 RWKV-v5 7b, 그리고 BTLM-3b-8k 트랜스포머 모델을 사용한 실험에서, RNN은 CAA를 활용한 조정 시 뛰어난 효율성을 보였습니다. 실험 결과, RNN 모델들이 중간 레이어에서 가장 큰 조정 효과를 나타냈으며, 특정 행위에 대한 확률을 조절할 때 상당한 변화를 보였습니다. 예를 들어, Mamba 모델은 생존 본능(Survival Instinct) 행위의 확률을 최대 0.15까지 변경시켰으며, BTLM 모델은 환각(Hallucination) 행위의 확률을 최대 0.2까지 변화시켰습니다.



### JSTR: Judgment Improves Scene Text Recognition (https://arxiv.org/abs/2404.05967)
Comments: IntelliSys 2024

- **What's New**: 본 논문에서는 이미지와 텍스트가 서로 일치하는지 판단함으로써 장면 텍스트 인식 작업의 정확성을 높이는 방법을 제시합니다. 이전 연구는 입력 이미지에서 인식 결과를 생성하는 데 중점을 두었지만, 우리의 접근 방식은 모델의 오인식 결과를 고려하여 오류 경향을 이해하고 텍스트 인식 파이프라인을 개선합니다. 이 방법은 모델이 오인식할 가능성이 있는 데이터에 대한 명시적 피드백을 제공함으로써 텍스트 인식 정확도를 향상시킵니다.

- **Technical Details**: 제안된 방법은 장면 텍스트 인식 모델 내에서 자체 결과의 오류에 대해 명시적 판단을 내리는 JSTR(Judging Scene Text Recognition) 프레임워크를 사용합니다. JSTR은 이미지와 인식 결과가 일치하는지 여부를 예측하는 학습을 포함합니다. 이는 오류에 취약한 데이터에 대해 모델의 판별 능력을 향상시키는 데 도움이 됩니다. DTrOCR 기반의 언어 기반 텍스트 인식 모델을 확장하여 이미지 텍스트 쌍 데이터셋에서 정확/오류 인식을 판단하도록 합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 ICDAR 2013 (IC13), ICDAR 2015 (IC15), IIIT 5KWords (IIIT), Street View Text (SVT) 등 공개 데이터셋에서 기준 모델 및 최신 방법보다 높은 성능을 보였습니다. 이는 JSTR이 재인식과 그 정확도 판단을 동시에 수행함으로써 장면 텍스트 인식 작업에 효과적임을 입증합니다.



### Use of a Structured Knowledge Base Enhances Metadata Curation by Large  Language Models (https://arxiv.org/abs/2404.05893)
- **What's New**: 이 논문은 큰 언어 모델(Large Language Models, LLMS)을 활용하여 메타데이터 기준 준수를 개선할 수 있는 가능성을 탐구합니다. 특히, GPT-4를 사용하여 NCBI BioSample 레포지토리에서 무작위로 선택한 인간 샘플에 대한 데이터 레코드 200개를 대상으로 실험을 수행했습니다.

- **Technical Details**: 연구에서는 GPT-4가 메타데이터 기준 준수를 향상시키기 위한 편집 제안을 하는 능력을 평가했습니다. 피어 리뷰(peer review) 과정을 통해 필드 이름-필드 값 쌍(field name-field value pairs)의 준수 정확도를 계산했으며, 기본 데이터 사전(standard data dictionary)에 대한 평균 준수율이 79%에서 80%로 소폭 개선된 것을 확인했습니다 (p<0.01). 또한, GPT-4에게 CEDAR 템플릿의 텍스트 설명(textual descriptions)을 도메인 정보(domain information) 형태로 제시한 후에는 준수율이 79%에서 97%로 크게 향상됐다는 결과를 얻었습니다 (p<0.01).

- **Performance Highlights**: LLMs가 독립적으로 사용될 때 기존 메타데이터의 표준 준수를 보장하는 데 한계가 있지만, 구조화된 지식 베이스(knowledge base)와 통합될 때 자동 메타데이터 관리(automated metadata curation)에 유망할 수 있다는 점을 드러냈습니다. 도메인 정보를 제공했을 때 준수율이 크게 향상된 것은 이러한 통합이 얼마나 중요한지를 시사합니다.



### Negative Preference Optimization: From Catastrophic Collapse to  Effective Unlearning (https://arxiv.org/abs/2404.05868)
- **What's New**: 이 논문은 Negative Preference Optimization (NPO)라는 새로운 방법을 제안하여 대규모 언어 모델(Large Language Models, LLMs)에서 원치 않는 데이터의 영향을 효과적으로 제거하는 데 중점을 두고 있습니다. 이 방법은 선호 최적화(Preference Optimization)에서 영감을 받았으며, 기존의 그래디언트 상승(Gradient Ascent, GA) 방법보다 더 안정적이고 효율적인 결과를 보여줍니다.

- **Technical Details**: NPO는 부정적인 샘플만을 사용하는 선호 최적화의 변형으로서, GA 방법의 주요 문제인 '급격한 성능 저하' 문제를 해결합니다. 본 논문은 이론적으로 NPO 손실을 최소화하면서 발생하는 '급격한 성능 저하'의 속도가 GA 방법보다 지수적으로 느린 것을 증명합니다. 또한, TOFU 데이터셋에 대한 실험을 통해 NPO 기반 방법이 모델 유용성을 유지하면서 원하는 데이터를 잊는 데 있어서 기존 방법들보다 우수함을 보여줍니다.

- **Performance Highlights**: NPO는 TOFU 데이터셋에서 50% 이상의 데이터를 잊는 첫 번째 비트리비얼한 결과를 달성했으며, 기존 방법들이 10%의 데이터를 잊는 데에 어려움을 겪는 것과 대조적입니다. NPO 기반 방법들은 잊어야 할 데이터와 모델의 다른 태스크에 대한 유용성 사이의 균형을 더 잘 맞추며, 더 안정적인 학습 동적을 보여주고 출력의 가독성도 향상시킵니다.



### Softmax Attention with Constant Cost per Token (https://arxiv.org/abs/2404.05843)
Comments: Source code and instructions for replicating our results are online at this https URL

- **What's New**: 우리는 Transformer에서 적용된 기존의 어텐션(attention) 메커니즘에 간단한 수정을 제안합니다: Pairwise query-key 유사성을 Scaled dot-product 대신 Exponential의 Scaled dot-product의 로그로 측정합니다. 이로 인해 어텐션을 log-sum-exp의 합성으로 표현할 수 있게 되며, 상수 크기의 latent space로 선형화가 가능해집니다. 이는 토큰당 상수 시간 및 공간 복잡성으로 연속적인 적용을 가능하게 합니다. 우리는 이 수정 사항을 구현하고 실제로 작동하는지 확인한 후, 기존 어텐션의 유망한 대안으로 결론지었습니다.

- **Technical Details**: 제안된 수정 사항은 query Q, key K, value V의 각 요소에 대해 exponential을 요소별로 계산하고, 이 계산된 exponentiated query와 key의 Scaled dot-product을 수행하여 ℝ+의 결과를 얻은 후, 이의 로그를 구합니다(이를 conventional logit이라 함). 이는 모든 Softmax 혼합이 ℝ에 닫혀 있으므로, 𝑉의 음수 요소에 대해 complex가 될 수 있습니다. 주목할 점은, 이 새로운 접근 방식이 기존의 quadratic-cost Softmax 함수 적용과 다르게 상수 복잡성으로 구현될 수 있다는 것입니다.

- **Performance Highlights**: 이 연구의 중요한 성과는 LSE (Log-Sum-Exponential) 및 LCSE (Log-Cumulative-Sum-Exponential) 함수를 사용하여 각 단계에서 누적된 지수 합의 로그를 계산함으로써 기존 및 새로운 쿼리에 대한 빠른 업데이트가 가능하다는 것입니다. 이는 autoregressive attention에서 각 쿼리가 t 개의 토큰만을 주목할 때, 특정 t에서 모든 순차적 의존성을 모델링하기 위해 필요한 시간 및 공간 복잡성을 상당히 감소시킵니다.



### Responsible Generative AI: What to Generate and What No (https://arxiv.org/abs/2404.05783)
Comments: 74 pages, 10 figures

- **What's New**: 이 논문은 새로운 접근 방식으로 Generative AI(GenAI)가 책임감 있게 콘텐츠를 생성하는 방법에 대해 탐구합니다. 특히, 텍스트 및 이미지 생성 모델(Textual and Visual Generative Models)의 책임 있는 사용에 필요한 실제 요구 사항을 다룹니다. 콘텐츠의 진실성 유지(Generating Truthful Content), 유해 콘텐츠 회피(Avoiding Toxic Content), 유해 지침 거부(Refusing Harmful Instructions), 트레이닝 데이터 관련 콘텐츠 유출 방지(Leaking No Training Data-Related Content), 생성된 콘텐츠 식별 가능성 확보(Ensuring Generated Content Identifiable) 등 다섯 가지 주요 고려 사항을 제시합니다.

- **Technical Details**: 이 논문은 Transformer와 Diffusion Models와 같은 다양한 모델 아키텍처를 사용하여 안전한 생성 모델을 구축하는 방법을 설명합니다. 또한, 적대적 공격(Adversarial Attacks) 및 백도어 공격(Backdoor Attacks)과 같은 심층 신경망의 취약점을 방어하는 방법에 관한 기본 지식을 제공합니다. 전체적인 분석을 통해, 텍스트 및 시각적 모델 모두에 대한 책임 있는 생성(Responsible Generation)에 관한 연구를 요약하고 있으며, 다양한 분야(Healthcare, Education, Finance, AGI)에서의 책임 있는 AI 활용 방안을 논의합니다.

- **Performance Highlights**: 이 연구는 GenAI가 실제 적용에서 발생할 수 있는 다양한 윤리적, 법적 문제에 대해 심도 있는 토론을 제공하며, 특히 주목할 만한 것은 생성된 콘텐츠의 진실성과 식별 가능성에 대한 엄격한 요구 사항입니다. 연구에 따르면, 현재 생성 모델들은 때때로 사실에서 벗어나거나 왜곡된 정보를 생성할 수 있는데(Hallucination), 이러한 문제를 식별하고 해결하기 위한 다양한 노력이 이루어지고 있습니다.



### Enhancing Inference Efficiency of Large Language Models: Investigating  Optimization Strategies and Architectural Innovations (https://arxiv.org/abs/2404.05741)
- **What's New**: 본 연구에서는 대형 언어 모델(Large Language Models, LLMs)의 크기가 증가함에 따라 추론 비용이 심각하게 증가할 것으로 예상되므로, 모델 압축의 중요성을 강조하고 있다. 이 논문에서는 특히 트랜스포머(Transformer) 기반 LLM에서 후반 부분의 어텐션 서브레이어(attention sublayers)를 생략하는 간단한 방법이 효과적인 모델 압축 방법임을 경험적으로 입증한다. 이러한 레이어들은 계산 비용이 매우 높으면서도 중복되는 경향이 있는데, 이를 생략함으로써 비용을 절감할 수 있다는 것을 보여준다.

- **Technical Details**: 연구는 Llama 2 7B 모델을 사용하여 실험하였으며, 어텐션 서브레이어를 생략함으로써 한 토큰 생성(one-token generation) 시 21%의 속도 향상을 관찰하였다. 이러한 접근 방식은 계산 비용은 줄이면서도 여러 일반적인 벤치마크(benchmarks)에서 예상치 못하게 성능을 향상시키는 결과를 가져왔다.

- **Performance Highlights**: 이 압축 방식은 특히 Llama 2 7B 모델에서 한 토큰 생성 시 21%의 속도 향상(speed increase)을 달성하였고, 서로 다른 벤치마크 테스트에서도 성능이 개선되었다는 것은 주목할 만하다. 이 결과는 모델의 효율성과 비용 효과적인 운영을 동시에 개선할 수 있는 가능성을 보여 준다.



### Language-Independent Representations Improve Zero-Shot Summarization (https://arxiv.org/abs/2404.05720)
Comments: NAACL 2024

- **What's New**: '제로샷' 상황에서 기존의 사전 학습된 모델을 다운스트림 생성 작업에 미세조정(finetuning)할 때 발생하는 심각한 문제인 '망각(catastrophic forgetting)'에 초점을 맞춥니다. 연구팀은 언어 독립적 표현(language-independent representations)을 통해 이 문제에 접근하며, 단일 언어 요약 학습 후 새로운 언어 또는 언어 쌍으로 '제로샷' 전이를 수행합니다.

- **Technical Details**: 연구팀은 기본적으로 미세조정된 모델이 출력 행동과 내부 표현 모두에서 언어 특이적임을 보여주며, 이로 인해 제로샷 성능이 좋지 않다는 것을 확인했습니다. 이를 해결하기 위해, 과제별 지식(task-specific knowledge)과 사전 학습된 언어 생성 능력(pretrained language generation abilities)을 분리하는 '쿼리-키(query-key, QK) 미세조정'을 제안합니다. 또한, 표준적인 적대적 언어 분류기의 단점을 보여주고, 언어에 중립적인 표현을 더 직접적으로 적용하는 '균형 있는(balanced) 변형'을 제안합니다.

- **Performance Highlights**: 언어에 국한되지 않은 대표 표현을 제거함으로써 제로샷 요약 성능과 관련이 있다는 것을 질적 분석으로 보여줍니다. 이와 같은 방법들은 모두 제로샷 상황에서 요약 작업의 성능을 높이는 데 기여할 것으로 기대됩니다. 또한, 사용한 코드는 공개적으로 이용 가능합니다.



### Comprehensive Study on German Language Models for Clinical and  Biomedical Text Understanding (https://arxiv.org/abs/2404.05694)
Comments: Accepted at LREC-COLING 2024

- **What's New**: 이 논문은 BERT와 RoBERTa와 같은 사전 훈련된 언어 모델을 의료 분야에 맞게 적용하는 전략을 탐구합니다. 특히, 전문 용어와 문서 구조가 다른 의료 도메인에 특화된 독일어 의료 언어 모델들을 개발하고, 이를 다양한 NLP 작업에서 평가하여, 이러한 모델이 일반 도메인 모델보다 우수한 성능을 보여줄 수 있음을 입증합니다.

- **Technical Details**: 연구팀은 영어 의료 데이터 2.4B 토큰과 독일어 임상 데이터 3B 토큰을 사용해 독일어 의료 언어 모델을 사전 훈련했습니다. 이 모델들은 연속적 사전 훈련(continuous pre-training)을 통해 도메인 특화 데이터에 맞게 조정되었습니다. 또한, 모델은 명명된 개체 인식(NER: Named Entity Recognition), 다중 라벨 분류(multi-label classification), 추출형 질문 답변(extractive question answering) 등의 다양한 독일어 하위 작업에서 평가되었습니다.

- **Performance Highlights**: 연구 결과에 따르면 임상 및 번역 기반 사전 훈련을 거친 (pre-trained) 모델들이 일반 도메인 모델보다 의료 맥락에서 더 우수한 결과를 보여줍니다. 이는 도메인에 맞게 특화된 사전 훈련이 스크래치(scratch)로부터 훈련된 임상 모델의 성능에 필적하거나 초과할 수 있음을 시사합니다.



### Evaluating Mathematical Reasoning Beyond Accuracy (https://arxiv.org/abs/2404.05692)
- **What's New**: ReasonEval은 대규모 언어 모델(Large Language Models, LLMs)에 의한 수학적 추론 과정의 품질 평가를 위한 새로운 방법론을 도입하였습니다. 이는 단순히 최종 결과의 정확성을 평가하는 것을 넘어서, 추론 과정에서의 각 단계가 올바르고 (validity), 불필요한 중복을 포함하지 않는지 (redundancy)를 평가함으로써, 더욱 심도 있는 분석을 가능하게 합니다. 이를 통해 LLM이 수학 문제를 해결하는 과정에서 발생할 수 있는 논리적 오류나 불필요한 단계를 정확히 감지할 수 있습니다. 또한, ReasonEval은 데이터 선택 과정에서도 중요한 역할을 할 수 있다고 보고되었습니다.

- **Technical Details**: ReasonEval은 LLMs를 사용하여 자동으로 추론 품질을 평가합니다. 이는 높은 수준의 수학 지식을 갖춘 베이스 모델을 기반으로 하며, 고품질의 레이블링된 데이터로 훈련됩니다. 평가 방법론은 특히 각 추론 단계의 유효성(validity)과 중복성(redundancy)을 측정하는 지표로 구성되어 있어, 수학적 추론 과정이 효율적으로 실행되는지 여부를 정밀하게 평가할 수 있습니다.

- **Performance Highlights**: ReasonEval은 인간이 레이블링한 데이터셋에서 최고의 성능(state-of-the-art, SOTA)을 달성하였고, 다양한 유형의 오류를 효과적으로 탐지할 수 있는 능력을 보여주었습니다. 준비된 기준에 따른 훈련 데이터를 이용하여 문제 해결의 효율성과 해답의 품질을 향상시킬 수 있는 고품질 훈련 데이터를 선택하는 데에 ReasonEval이 유용하게 사용될 수 있다는 점을 입증하였습니다.



### VietMed: A Dataset and Benchmark for Automatic Speech Recognition of  Vietnamese in the Medical Domain (https://arxiv.org/abs/2404.05659)
Comments: LREC-COLING 2024

- **What's New**: 이 연구에서는 의료 분야의 베트남어 음성 인식 데이터셋인 VietMed를 소개합니다. 이 데이터셋은 16시간의 레이블링된 의료 음성, 1000시간의 레이블되지 않은 의료 음성, 그리고 1200시간의 레이블되지 않은 일반 도메인 음성을 포함합니다. VietMed는 세계에서 가장 큰 공개 의료 음성 데이터셋으로, 총 시간, 발화자 수, 질병 종류, 녹음 조건, 발화자 역할, 독특한 의료 용어, 그리고 방언 각각에서 최대 규모입니다.

- **Technical Details**: VietMed는 모든 ICD-10 질병 그룹과 베트남 내 모든 방언을 포괄하는 최초의 의료 음성 인식(ASR) 데이터셋을 제공합니다. 또한, 베트남어 ASR을 위한 대규모 사전 훈련된 모델들(w2v2-Viet, XLSR-53-Viet)과 의료 ASR을 위한 첫 대규모 파인 튜닝(Pre-trained) 모델들을 공개합니다. 특히, 무감독 사전 훈련(Unsupervised Pre-training)에서 의료 데이터 없이도 최고의 사전 훈련 모델 XLSR-53-Viet가 의료 도메인에서 매우 잘 일반화되어 기존 최고의 XLSR-53을 상당히 앞서는 성능(테스트 세트에서 51.8%에서 29.6% WER로 감소, 상대적으로 40% 이상 감소)을 달성합니다.

- **Performance Highlights**: VietMed는 의료 분야에서 높은 정확도를 달성하며, 기존의 모델들 대비 상당한 성능 향상을 보여줍니다. 이 데이터셋을 사용한 XLSR-53-Viet 모델은 테스트 세트에서 WER(Word Error Rate, 단어 오류율)을 51.8%에서 29.6%로 줄이는 데 성공하였으며, 이는 40% 이상의 상대적인 오류 감소율을 의미합니다.



### Causality Extraction from Nuclear Licensee Event Reports Using a Hybrid  Framework (https://arxiv.org/abs/2404.05656)
- **What's New**: 이 연구에서는 원자로 이벤트 보고서로부터 원인과 결과의 관계를 검출하고 추출하는 혼합 프레임워크를 제안하였습니다. 그 동안 이벤트 보고서는 주로 비정형 데이터(unstructured data)로 구성되어 있었으며, 이를 통해 장애의 발생과 전파 경로를 이해하는 데 중요한 역할을 해왔습니다. 본 논문에서 제안된 방법은 NLP(Natural Language Processing) 분야에서 중요한 발전을 제시하며, 방대한 양의 서술적 정보 속 복잡한 내러티브와 연결성을 해석할 수 있는 새로운 기회를 제공합니다.

- **Technical Details**: 제안된 프레임워크는 크게 네 부분으로 구성됩니다. 첫째, 원인결과 분석을 위해 20,129건의 텍스트 샘플을 포함하는 LER(Licensee Event Report) 코퍼스를 구축하였습니다. 둘째, 원인과 결과 쌍을 라벨링(labeling)할 수 있는 인터랙티브 도구를 개발하였습니다. 셋째, 원인 관계 검출을 위해 딥러닝(deep learning)을 기반으로 한 접근 방식을 구축하였습니다. 넷째, 지식 기반의 원인-효과 추출 방식을 개발하였습니다. 이러한 기술적 구성은 향후 비정형 데이터에서 중요한 원인-효과 관계를 파악하는데 핵심 도구가 될 것입니다.

- **Performance Highlights**: 핵심 성과로는 특히, 다량의 비정형 데이터로부터 복잡한 인과 관계를 효과적으로 식별하고 추출할 수 있는 능력을 개발한 것입니다. 이는 원자력 발전소의 운영 데이터를 활용하여 신뢰성과 위험 모델의 정확성을 높이는 중요한 기여를 하였습니다. 또한, 개발된 인터랙티브 도구는 실시간으로 데이터 라벨링을 가능하게 함으로써, 다양한 연구자와 전문가들이 더욱 효율적으로 작업할 수 있는 환경을 제공합니다.



### Fighting crime with Transformers: Empirical analysis of address parsing  methods in payment data (https://arxiv.org/abs/2404.05632)
- **What's New**: 이 연구는 금융 산업에서 국제 결제 메시지의 주소 파싱(address parsing) 작업을 개선하기 위해 최신 AI 기술을 활용한 새로운 접근 방식을 제시합니다. 특히 Transformer 모델과 Generative Large Language Models (LLM, 대규모 언어 모델)를 사용하여 실제 세계의 잡음이 많은 거래 데이터를 효과적으로 처리할 수 있는 능력을 분석하고 비교하였습니다. 또한, 이 연구는 새로운 국제 결제 메시지 표준인 ISO 20022를 지원하는 동시에, 여전히 자유 형식 텍스트로 제공되는 주소 데이터를 처리할 수 있는 강인한 모델을 개발하는 것을 목표로 합니다.

- **Technical Details**: 연구 팀은 실제 거래 데이터와 유사한 구조를 가진 합성 데이터(synthetic data)를 생성하여 모델을 훈련시켰습니다. 이를 위해 Faker 라이브러리를 사용해 가상의 개인 및 회사 이름을 생성하고, SWIFT 메시지에서 나라 이름 또는 ISO 코드의 존재 유무를 분석하여 주소에 'Country' 또는 'CountryCode' 태그를 삽입했습니다. 주소 구조에 대한 변경사항은 실제 데이터의 불규칙성을 반영하기 위해 임의 샘플링을 통해 조정되었습니다. 또한, 모델이 특정 국가의 주소 구조에 과적합(overfitting)되는 문제를 방지하기 위해 다양한 국가의 주소 형식을 무작위로 섞어 사용했습니다.

- **Performance Highlights**: Transformer 모델은 조기 종료(early-stopping)를 사용하여 미세 조정(fine-tuned)을 거쳐 다른 접근 방식에 비해 현저히 우수한 성능을 보였습니다. 그러나 Generative LLM은 제로-샷(zero-shot) 성능에서 강력한 결과를 보여 향후 추가적인 연구가 필요하다는 것을 시사합니다. 데이터의 실제 구조와 주소 표준의 상이함을 극복하고자 하는 이들 연구 결과는 국제 결제 처리 및 규제 준수를 위한 더 나은 도구 개발에 기여할 수 있을 것입니다.



### LTNER: Large Language Model Tagging for Named Entity Recognition with  Contextualized Entity Marking (https://arxiv.org/abs/2404.05624)
Comments: 13 pages

- **What's New**: 자연어 처리(Natural Language Processing, NLP)를 위한 대형 언어 모델(Large Language Models, LLMs) 사용은 지난 2년 동안 인기 있는 추세가 되었습니다. 그러나 특정 NLP 작업, 예를 들어 개체명 인식(Named Entity Recognition, NER)에서는 여전히 감독 학습(supervised learning) 방법에 비해 성능이 떨어집니다. 이 연구에서는 'Contextualized Entity Marking Gen Method'라는 혁신적인 방법을 포함하는 LTNER라는 NER 처리 프레임워크를 개발했습니다.

- **Technical Details**: LTNER 프레임워크는 추가적인 훈련을 요구하지 않는 컨텍스트 학습(context learning)과 비용 효율적인 GPT-3.5를 결합하여 사용합니다. 이는 LLM들이 NER 작업을 처리하는 정확도를 크게 향상시켰습니다.

- **Performance Highlights**: CoNLL03 데이터셋에서 F1 점수는 초기 85.9%에서 91.9%로 상승하였으며, 이는 감독된 미세조정(supervised fine-tuning)의 성능에 근접하는 결과입니다. 이 결과는 LLM의 잠재력에 대한 더 깊은 이해를 제공하였습니다.



### How to Evaluate Entity Resolution Systems: An Entity-Centric Framework  with Application to Inventor Name Disambiguation (https://arxiv.org/abs/2404.05622)
Comments: 33 pages, 11 figures

- **What's New**: 이 논문은 엔터티 결정(Entity Resolution, ER) 시스템의 정확성을 평가하는 새로운 접근법을 제안합니다. 기존의 평가 방법들이 복잡한 샘플링 방식을 사용하는 데 반해, 저자들은 대표적이고 재사용 가능한 벤치마크 데이터 세트를 생성하는 것을 용이하게 하는 엔터티 중심의 데이터 라벨링 방법론(entity-centric data labeling methodology)을 제안합니다. 이를 통해 모델 훈련과 다양한 평가 작업에 사용할 수 있습니다.

- **Technical Details**: 제안된 프레임워크는 집계된 클러스터 수준의 에러 메트릭스를 통해 전체 데이터 세트의 시스템 성능을 대표할 수 있는 글로벌 성능 지표(global performance metrics)를 추정합니다. 이러한 지표들로는 쌍별 및 B-cubed 정밀도와 재현율(pairwise and b-cubed precision and recall)이 포함됩니다. 또한, 기존의 검토 방식 대신 완전히 해결된 엔터티 집합의 샘플을 활용하여, 매치되는 쌍과 매치되지 않는 쌍을 구분하고, 이를 통해 성능 메트릭을 추정합니다.

- **Performance Highlights**: 이 프레임워크는 미국 특허청(USPTO)의 발명가 이름의 정확한 식별을 위해 개발되었으며, 시뮬레이션 연구를 통해 검증되었습니다. 전통적인 샘플링 기반 평가 방법에 비해 더 정확하고 신뢰할 수 있는 성능 메트릭을 제공함으로써, 엔터티 결정 시스템의 성능을 보다 정확하게 평가하고 비교할 수 있도록 합니다.



### SpeechAlign: Aligning Speech Generation to Human Preferences (https://arxiv.org/abs/2404.05600)
Comments: Work in progress

- **newsletter**: [{"What's New": '이 논문은 신경 코덱(neural codec) 언어 모델의 성능을 인간의 의견을 반영하여 개선하는 신규 방법론인 SpeechAlign을 제안합니다. 여기에는 인간의 피드백에서 학습하는 것이 포함되어 있으며, 기존의 훈련과 추론 단계에서 발생하는 분포 간격(distribution gap)을 해소하는 것을 목표로 합니다.'}, {'Technical Details': 'SpeechAlign은 코덱 언어 모델들의 성능을 향상시키기 위해 반복적인 자기 개선 전략(iterative self-improvement strategy)을 사용합니다. 이 방법은 우선적으로 golden codec 토큰과 synthetic 토큰을 대조하는 선호도 코덱 데이터 셋(preference codec dataset)을 구성합니다. 이후, 선호도 최적화(preference optimization)를 통해 모델을 지속적으로 향상시키는 사이클을 반복적으로 수행합니다.'}, {'Performance Highlights': '실험 결과에 따르면 SpeechAlign은 인간의 기대치와 일치하는 방식으로 음성 생성 성능을 지속적으로 향상시킬 수 있습니다. 또한, 이 방법은 강력한 일반화 능력을 보여주며 작은 모델에도 효과적으로 적용될 수 있는 것으로 나타났습니다.'}]



### MedExpQA: Multilingual Benchmarking of Large Language Models for Medical  Question Answering (https://arxiv.org/abs/2404.05590)
- **What's New**: 이 논문에서는 의료 분야 대화형 의사 결정 (interactive decision support) 지원을 위해 인공지능 (Artificial Intelligence, AI) 기술의 개발을 촉진하는 잠재력을 가진 Large Language Models (LLMs)에 대해 소개하고 있으며, 다국어 벤치마크인 MedExpQA (Medical Explanation-based Question Answering)를 최초로 제안했습니다. MedExpQA는 의료 질문 답변 (Medical Question Answering)의 LLM 평가를 위해 의료시험을 기반으로 하며, 의료 전문가에 의해 작성된 참조 금 설명 (reference gold explanations)을 포함하고 있어 LLM의 추론 능력을 평가할 수 있습니다.

- **Technical Details**: 이 연구는 여러 LLMs를 사용하여 MedExpQA 벤치마크를 통해 다양한 언어로 실험을 진행했습니다. 연구에는 원래 스페인어로 작성된 CasiMedicos 데이터셋을 사용했으며, 이는 영어, 프랑스어, 이탈리아어로 번역 및 주석 처리되었습니다. 각 데이터는 짧은 임상 케이스, 질문 및 다지선다형 옵션들을 포함합니다. 또한, 연구에는 Retrieval Augmented Generation (RAG) 기술을 적용하여 최신 의료 지식을 통합함으로써 LLM의 성능이 향상될 수 있는지를 검토하였습니다.

- **Performance Highlights**: 다국어 실험 결과, LLM의 성능은 참조 금 설명을 기반으로 한 외부 지식을 사용할 때 크게 향상되었습니다. 그러나, 영어가 아닌 언어, 예를 들어 프랑스어, 이탈리아어, 스페인어에서의 성능은 상대적으로 낮아 다국어 지원을 개선할 필요가 있음을 제시했습니다. RAG 접근 방식은 초기 설정에서 긍정적인 영향을 미치지만, 미세 조정 (fine-tuning)이 부과된 설정에서는 RAG가 불필요해지는 결과를 보였습니다.



### Enhancing Software Related Information Extraction with Generative  Language Models through Single-Choice Question Answering (https://arxiv.org/abs/2404.05587)
Comments: Accepted at: 1st Workshop on Natural Scientific Language Processing and Research Knowledge Graphs (NSLP 2024) Co-located with Extended Semantic Web Conference (ESWC 2024)

- **What's New**: 이 논문은 학술 텍스트에서 소프트웨어 관련 정보를 추출하기 위해 Generative Language Models (GLMs)와 Retrieval-Augmented Generation (RAG) 기법을 사용하는 방법을 설명합니다. 특히, Software Mentions Disambiguation (SOMD) 공유 작업에 참여하면서, 소프트웨어 언급을 식별하고 이와 관련된 속성을 추출하는 과제에 중점을 둡니다. 단일 선택형 질문 응답을 통한 관계 추출에 GLMs를 활용하는 새로운 방법론을 제시합니다.

- **Technical Details**: 이 연구에서는 GLMs를 활용하여 Named Entity Recognition (NER)과 Attributive NER 작업을 수행하며, 소프트웨어 엔티티와 그 속성 간의 관계를 파악합니다. 또한, RAG 기술을 통합하여, 학술 문헌에서 소프트웨어 인용을 분석하는 구조화된 해결책을 제공합니다. 정보 추출 (Information Extraction, IE) 작업을 위해 GLMs 속성을 활용하는데, 특히 도메인 특화 작업에서의 성능 향상을 위해 인-도메인 학습 전략을 적용합니다.

- **Performance Highlights**: SOMD 공유 작업에서의 참여를 통해, 본 시스템은 소프트웨어 언급의 모호성 해소 및 관련 관계 추출의 도전을 극복하는 능력을 선보였습니다. 이 연구는 향후 이 분야의 연구 및 개발을 위한 기초를 마련합니다. 논문은 GLMs와 RAG를 결합하여 정보 추출 작업의 정밀성을 높이는 성능을 보여주며, 특히 도메인 특화적 NER 작업에서 Large Language Models (LLMs)의 일반적인 훈련 부족 문제를 해결하기 위한 다양한 학습 전략의 유효성을 입증합니다.



### Chinese Sequence Labeling with Semi-Supervised Boundary-Aware Language  Model Pre-training (https://arxiv.org/abs/2404.05560)
Comments: Accepted to COLING 2024

- **What's New**: 이 연구에서는 중국어 시퀀스 레이블링 작업에 대한 새로운 모델인 Semi-BABERT를 제안합니다. 이 모델은 BABERT를 기반으로 하며, 지도 학습을 통해 얻은 고품질의 경계 정보를 추가하여 성능을 향상시킵니다. 또한, 중국어 자연어 처리(Chinese NLP) 작업을 넘어서는 범위에서도 향상된 성능을 보입니다. 특히, 이 연구는 'Boundary Information Metric'이라는 새로운 메트릭을 도입하여 언어 모델이 경계 정보를 얼마나 잘 인코딩하는지 평가하는 데 사용됩니다.

- **Technical Details**: Semi-BABERT는 스팬 기반 경계 인식 목표(span-based boundary recognition objective)를 사용하여 사전 학습을 강화합니다. 모델이 고품질의 경계 정보를 효과적으로 학습할 수 있도록 지식 그래프와 크라우드 소싱된 코퍼스를 활용합니다. 또한, Positive-Unlabeled learning (PU learning)을 적용하여 학습 데이터의 부족을 보완합니다. 이를 통해 모델은 우수한 성능을 보이는 동시에 중국어 시퀀스 레이블링 연구에 새로운 방향을 제시합니다.

- **Performance Highlights**: Semi-BABERT는 중국어 단어 분리(CWS), 품사 태깅(POS), 개체명 인식(NER) 등 다양한 시퀀스 레이블링 작업에서 기존 BABERT보다 우수한 성능을 보여줍니다. 또한, 텍스트 분류와 기계 독해 등의 다른 중국어 자연어 이해 작업에서도 성능이 향상되었음을 확인할 수 있습니다. 제안된 'Boundary Information Metric'을 통해 다른 언어 모델과 비교할 때 경계 정보 인식 능력이 뛰어남을 입증합니다.



### OPSD: an Offensive Persian Social media Dataset and its baseline  evaluations (https://arxiv.org/abs/2404.05540)
Comments: 16 pages, 5 figures, 8 tables

- **What's New**: 이 논문은 소셜 미디어에서의 증가하는 증오 발언과 공격적인 댓글 문제를 다루며, 특히 페르샤어(Persian) 자료에 대한 연구 자원이 부족한 점을 해결하기 위해 두 가지 오펜시브(offensive) 데이터셋을 소개합니다. 첫 번째 데이터셋은 도메인 전문가들이 제공한 주석을 담고 있으며, 두 번째 데이터셋은 비지도 학습 목적으로 웹 크롤링을 통해 얻은 대규모 레이블이 없는 데이터로 구성됩니다.

- **Technical Details**: 데이터의 질을 확보하기 위해 세 단계의 주석 과정(annotation process)이 시행되었고, 인터-어노테이터 합의도를 평가하기 위해 카파(kappa) 측정이 사용되었습니다. 또한, 최신 언어 모델을 사용하여 데이터셋에서 마스크된 언어 모델링 기술(masked language modeling techniques)을 적용한 실험을 수행하였으며, 최신 변형기-기반 모델(transformer-based models)과 기계 학습 기법(machine learning algorithms)의 기준 선을 설정했습니다.

- **Performance Highlights**: 실험 결과, 이 데이터셋의 세 클래스 및 두 클래스 버전에 대해 XLM-RoBERTa를 사용할 때의 F1-점수는 각각 76.9%와 89.9%를 기록하였습니다. 이는 페르샤어 오펜시브 댓글 탐지 소프트웨어의 성능 향상이 이루어졌음을 보여줍니다.



### Best-of-Venom: Attacking RLHF by Injecting Poisoned Preference Data (https://arxiv.org/abs/2404.05530)
- **What's New**: 새롭게 등장한 인공지능(AI) 기술인 인간 피드백에 기반한 강화학습(Reinforcement Learning from Human Feedback, RLHF)는 언어모델(Language Models, LM)을 인간의 가치와 선호도에 맞추기 위해 널리 사용되고 있습니다. 본 연구에서는 악의적인 행위자가 선호도 쌍 데이터셋을 조작하여 LM의 생성을 조종할 수 있는지 여부를 분석하였습니다. 이를 '선호도 독살(preference poisoning)'이라 명명하고, 이러한 공격을 수행하고 방어하는 전략을 제시합니다.

- **Technical Details**: 이 연구에서는 공격자가 선호도 쌍 데이터셋에 새로운 선호도 쌍을 주입함으로써 특정 대상(entity)이 긍정적 혹은 부정적인 감정(sentiment)으로 생성되도록 LM을 조작할 수 있다는 가설을 검증하고 있습니다. 선호도 독살 공격을 시뮬레이션하기 위하여, 연구팀은 두 가지 주요 데이터셋에 독성 데이터를 주입하고, 강화학습 알고리즘, 특히 PPO(Proximal Policy Optimization) 및 Best-of-N을 사용하여 결과를 분석하였습니다.

- **Performance Highlights**: 연구 결과, 원래 데이터셋의 1-5%에 해당하는 소량의 독성 데이터만 주입되어도, RM(Reward Model)은 원하는 생성물을 강하게 선호하게 되는 것으로 나타났습니다(80.4-95.2%의 확률로). 또한 Best-of-N 알고리즘을 사용한 강화학습은 이러한 독성 패턴을 더욱 강화시켰으며, 한 번의 강화학습 에피소드(training episode)만으로도 원하는 생성물의 빈도가 대부분의 실험에서 두 배로 증가했습니다. 이는 LM이 RLHF를 통해 독성 패턴을 빠르게 학습할 수 있음을 시사합니다.



### PetKaz at SemEval-2024 Task 3: Advancing Emotion Classification with an  LLM for Emotion-Cause Pair Extraction in Conversations (https://arxiv.org/abs/2404.05502)
Comments: 8 pages, 7 figures, 2 tables, to be published in the Proceedings of the 18th International Workshop on Semantic Evaluation (SemEval-2024), for associated code, see this https URL

- **What's New**: 이 논문에서는 대화에서 감정-원인 쌍을 추출하는 SemEval-2023 Task 3 'The Competition of Multimodal Emotion Cause Analysis in Conversations'에 대한 우리 팀의 제출 결과를 소개합니다. 본 연구에서는 GPT-3.5를 감정 분류용으로 파인 튜닝(Fine-tuned)하고 BiLSTM 기반 신경망을 원인 검출용으로 사용하는 방법에 초점을 맞추었습니다.

- **Technical Details**: 우리의 접근 방식은 두 단계 파이프라인(pipeline)으로 구성됩니다. 첫 번째 단계에서는 대화에서 감정 유형을 분류하기 위해 파인 튜닝된 대규모 언어 모델(Large Language Model, LLM)을 사용하고, 두 번째 단계에서는 BiLSTM 및 선형 계층(Linear layers)을 통합한 간단한 신경망을 사용하여 원인 말문구를 추출합니다. 이 시스템으로 인해 우리 팀은 테스트 세트에서 0.264의 가중 평균 비례 F1 점수(Weighted-Average Proportional F1 Score)를 기록하며 15개 팀 중 2위를 차지했습니다.

- **Performance Highlights**: 본 시스템은 특히 감정 분류와 원인 검출의 정확성에서 뛰어난 성능을 보여주었으며, 여러 팀 중에서 높은 순위를 달성함으로써 우리의 접근 방식의 효과성을 입증하였습니다. 이러한 성과는 다양한 대화 시나리오에서 감정 원인 분석을 정확하게 수행할 수 있는 시스템의 중요성을 강조합니다.



### PetKaz at SemEval-2024 Task 8: Can Linguistics Capture the Specifics of  LLM-generated Text? (https://arxiv.org/abs/2404.05483)
Comments: 8 pages, 3 figures, 5 tables, to be published in the Proceedings of the 18th International Workshop on Semantic Evaluation (SemEval-2024), for associated code, see this https URL

- **What's New**: 이 논문에서는 'SemEval-2024 Task 8: 멀티제너레이터, 멀티도메인, 멀티언어 Black-Box 기계 생성 텍스트 감지(Multigenerator, Multidomain, and Multilingual Black-Box Machine-Generated Text Detection)'에 대한 접근법을 소개하며, 영어로 생성된 기계 텍스트(machine-generated texts, MGTs) 탐지에 초점을 맞춥니다. 특히, 우리는 RoBERTa-base에서 추출한 임베딩(embeddings)과 다양성 특징(diversity features)을 조합하고 재표본 추출된 훈련 세트를 사용하는 방식을 택했습니다.

- **Technical Details**: 우리의 시스템은 검은 상자(black-box) 탐지기로, 임베딩과 어휘 다양성 측정을 결합하고 정교하게 선택된 훈련 데이터를 사용합니다. 이 시스템은 Subtask A(monolingual track)에서 전체 124개 팀 중 12위를 차지했으며, 테스트 세트에서 0.91의 정확도를 달성했습니다. 우리는 또한 특징(feature) 기반 접근 방식을 채택하고 있는데, 이는 MGT와 인간이 작성한 텍스트(human-written texts, HWTs) 간의 근본적인 차이점에 집중을 돕기 위함입니다.

- **Performance Highlights**: 주요 결과로는, 다양성 특징과 임베딩을 사용한 모델이 이 작업에서 소개된 매우 경쟁적인 베이스라인보다 우수한 성능을 보여 개발 세트에서 0.95, 테스트 세트에서 0.91의 정확도를 기록했습니다. 우리의 조사에 따르면 임베딩을 사용하지 않고 텍스트의 엔티티 그리드(entity grid)와 문체학(stylometry)과 같은 언어적 특징에 의존하는 모델도 베이스라인 모델과 유사한 결과를 제공합니다.



### RoT: Enhancing Large Language Models with Reflection on Search Trees (https://arxiv.org/abs/2404.05449)
Comments: 9 pages main

- **What's New**: 이 연구에서는 대형 언어 모델(LLMs)이 트리 검색 기반의 프롬프팅 방법과 통합될 때, 추론(reasoning)과 계획(planning)에서 인상적인 능력을 보여준다는 것을 발견했습니다. 하지만 이전 검색 경험을 무시하기 때문에 종종 같은 실수를 반복합니다. 이를 해결하기 위해, 'Reflection on Search Trees' (RoT)이라는 새로운 프레임워크를 도입하여, 강력한 LLM을 사용하여 약한 LLM의 성능을 개선함으로써 검색 효율성과 정확성을 높일 수 있도록 했습니다.

- **Technical Details**: RoT는 강력한 LLM을 사용하여 이전 트리 검색 과정에서 중요한 상태들을 요약하고, 이를 기반으로 지침을 생성하여 약한 LLM이 비슷한 실수를 반복하지 않도록 지원합니다. 또한, 이 연구는 역사적 검색 과정에서 중요한 정보를 식별하여 더 구체적이고 의미 있는 지침을 생성할 수 있는 새로운 상태 선택 방법을 제안했습니다. 이 방법은 RoT가 특정 태스크에 맞춘 지식을 제공하면서 도움이 됩니다.

- **Performance Highlights**: RoT는 여러 복잡한 추론 및 계획 작업에서 LLM의 성능을 크게 향상시켰으며, BFS와 MCTS와 같은 여러 트리 검색 기반 프롬프팅 방법에서 뛰어나게 작동했습니다. 특히, LEAP과 같은 강력한 반영 방법을 능가했으며, 모델이 익숙하지 않은 작업에서 가장 큰 이점을 보였습니다. 비트리 검색 기반 프롬프팅 방법들도 RoT로부터 혜택을 받을 수 있습니다.



### XL$^2$Bench: A Benchmark for Extremely Long Context Understanding with  Long-range Dependencies (https://arxiv.org/abs/2404.05446)
Comments: Work in progress

- **What's New**: 이 논문에서는 기존의 긴 텍스트 이해 능력을 평가하는 벤치마크들이 주로 기존의 NLP 작업을 단순 확대하는 방식으로 제작된다는 한계를 지적하며, 이를 해결하기 위해 특별히 설계된 XL$^2$Bench를 소개합니다. 이 벤치마크는 픽션, 학술 논문, 법률 문서 읽기 등 세 가지 시나리오와 기억 검색(Memory Retrieval), 상세한 이해(Detailed Understanding), 전체적인 이해(Overall Understanding), 개방형 생성(Open-ended Generation) 등 네 가지 복잡도가 높은 작업을 포함합니다.

- **Technical Details**: XL$^2$Bench는 100,000개 이상의 단어(영어)와 200,000개 이상의 문자(중국어)를 포함하는 매우 긴 문서들로 구성되어 있으며, 이 벤치마크는 대규모 언어 모델(LLMs)이 긴 텍스트에서 정보를 집계하고 비교할 수 있는 능력을 평가합니다. 또한 데이터 오염 문제를 해결하기 위해 텍스트 변형, 텍스트 교체, 텍스트 연결과 같은 세 가지 데이터 증강 전략을 도입했습니다.

- **Performance Highlights**: 현재 최고의 대규모 언어 모델들은 XL$^2$Bench에서의 성능이 인간 수준에 미치지 못하는 것으로 나타났으며, 텍스트의 길이가 길어질수록 성능이 떨어지는 것이 관찰되었습니다. 반면 증강된 벤치마크에서의 실험 결과는 데이터 증강 전략이 데이터 오염 문제를 완화하는 데 효과적임을 입증했습니다.



### Language Models on a Diet: Cost-Efficient Development of Encoders for  Closely-Related Languages via Additional Pretraining (https://arxiv.org/abs/2404.05428)
- **What's New**: 이 논문에서는 한국어를 포함한 중소 규모 언어에 적합한 10억 개 미만의 파라미터를 가진 인코더 언어 모델을 개발합니다. 이러한 언어들을 위한 모델은 강력한 대규모 언어 모델(Large Language Models, LLMs)들이 지배적인 현 상황에서 필요한 도구입니다. 특히, 크로아티아어, 세르비아어, 보스니아어, 몬테네그로어 (collectively referred to as 'HBS' 언어들) 및 이와 밀접한 관련이 있는 슬로베니아어를 포함시키며 추가적인 프리트레이닝(pretraining)을 통해 개선된 성능을 보여줍니다.

- **Technical Details**: 연구팀은 기존의 다국어 모델인 XLM-RoBERTa (XLM-R)에 대해 추가적인 프리트레이닝을 실시하는 두 가지 접근 방식을 비교했습니다. 첫 번째는 HBS 언어 데이터만을 사용한 추가적인 프리트레이닝이고, 두 번째는 HBS 언어에 슬로베니아어 데이터까지 포함한 형태입니다. 이 두 접근 방식을 통해 모델이 어떻게 다양한 언어에 민감하게 반응하는지를 평가했습니다. 또한, 이 방법이 크로아티아어, 세르비아어, 보스니아어, 몬테네그로어에 최적화된 더 작은 모델들과 어떻게 비교되는지를 조사했습니다.

- **Performance Highlights**: 추가적인 프리트레이닝을 통해 개선된 XLM-R 모델은 기존에 개별적으로 트레이닝된 BERTić이나 cseBERT 모델과 비슷하거나 더 우수한 성능을 보였습니다. 특히, 개선된 모델은 명명된 엔티티 인식(Named Entity Recognition), 정치 텍스트의 감정 분류(Sentiment Analysis) 및 원인-결과 추론(Causal Commonsense Reasoning) 작업에서 탁월한 결과를 나타냈습니다. 이 연구 결과는 추가적인 언어 데이터를 포함시키는 것이 전반적인 모델 성능에 도움이 될 수 있음을 보여줍니다.



### Relation Extraction Using Large Language Models: A Case Study on  Acupuncture Point Locations (https://arxiv.org/abs/2404.05415)
- **What's New**: 이 연구는 침술 요법에서 중요한 경혈점 위치 관계를 추출하기 위해 LLM(Large Language Models, 대규모 언어 모델)인 GPT를 사용하는 새로운 접근 방식을 소개합니다. GPT-3.5와 GPT-4와 같은 최신 모델을 이용하여 기존의 딥러닝 모델들(LSTM, BioBERT)과 성능을 비교 분석하였습니다.

- **Technical Details**: 연구는 WHO(세계보건기구)에서 발표한 서태평양 지역 표준 경혈 위치에 대한 텍스트를 사용하여, 경혈과 관련된 다섯 가지 유형의 관계('direction_of', 'distance_of', 'part_of', 'near_acupoint', 'located_near')를 주석 처리하고, 이를 바탕으로 모델 훈련 및 평가를 수행했습니다. LSTM, BioBERT, 사전 훈련된 GPT-3.5, 미세 조정(fine-tuned)된 GPT-3.5 및 사전 훈련된 GPT-4를 비교했습니다.

- **Performance Highlights**: 미세 조정된 GPT-3.5 모델이 모든 관계 유형에서 가장 높은 F1 스코어를 달성하며 다른 모델들을 종합적으로 능가했습니다. 전체적으로 가장 높은 마이크로 평균 F1 점수(0.92)를 기록하였으며, 이는 GPT 모델이 자연어 처리뿐만 아니라 전통적이고 보완적인 의학 분야에서의 정보 응용 분야에 효과적으로 활용될 수 있음을 시사합니다.



### Know When To Stop: A Study of Semantic Drift in Text Generation (https://arxiv.org/abs/2404.05411)
- **What's New**: 이 연구에서는 현대 대규모 언어 모델 (LLMs) 이 정확한 사실을 먼저 생성한 다음, 정확하지 않은 사실을 생성하는 경향이 있음을 명확하게 보여줍니다. 이러한 'semantic drift' 혹은 '의미의 이탈' 현상을 측정하기 위해 새로운 세맨틱 드리프트 점수를 개발하였고, 위키백과 스타일의 전기 작성을 통해 이를 확인하였습니다. 이 연구는 생성을 조기에 멈춤으로써 사실 정확성을 향상시킬 수 있음을 제안합니다.

- **Technical Details**: 세맨틱 드리프트 (Semantic Drift) 는 텍스트 생성 품질이 생성 길이가 길어질수록 감소하는 현상으로, 정확한 사실에서 부정확한 사실로의 전환을 측정합니다. 연구팀은 'FActScore' 태스크를 사용하여 개별 사실에 대한 정확/부정확 레이블을 제공하고, LLaMa2 변형 모델이 높은 세맨틱 드리프트 점수를 보이는 것을 확인하였습니다. 또한, 생성을 조기에 종료하도록 유도하는 간단한 방법을 사용하여 사실 정확도를 크게 향상시키는 것을 보여주었습니다.

- **Performance Highlights**: 재표본 추출 후 재순위 매기기(resample-then-rerank) 파이프라인을 통해 문장 유사성 측정 기준을 기반으로 최적의 버전을 선택함으로써, 기존 베이스라인에 비해 사실 정확도를 거의 10% 향상시켰습니다. 이러한 방법은 조기 종료와 결합될 수 있으며, 정보 양과 사실 정확도 사이의 다양한 트레이드 오프를 허용합니다. 외부 API를 호출하여 모델을 올바른 생성 경로로 되돌리려는 시도는 긍정적인 결과를 얻지 못했습니다.



### PerkwE_COQA: enhance Persian Conversational Question Answering by  combining contextual keyword extraction with Large Language Models (https://arxiv.org/abs/2404.05406)
- **What's New**: 이 논문은 페르시아어 대화형 질문-응답(CQA) 시스템의 성능을 향상시키기 위한 새로운 방법을 제시합니다. 본 연구는 대규모 언어 모델(LLM: Large Language Models)의 강점과 문맥 키워드 추출을 결합하여 사용자의 의도를 더 잘 이해하고, 더욱 관련성 높고 일관성 있는 응답을 생성하도록 합니다.

- **Technical Details**: 제안된 방법은 대화의 흐름에 특화된 키워드를 추출하여 LLM에 추가적인 문맥을 제공합니다. 이를 통해 시스템은 사용자의 질문 의도를 파악하고 그에 맞는 적절한 답변을 생성할 수 있게 됩니다. 또한, 이 방법은 암시적인 질문 처리, 문맥적으로 관련된 답변 제공, 대화 문맥에 크게 의존하는 복잡한 질문에 효과적으로 대응할 수 있습니다.

- **Performance Highlights**: 본 연구에서는 다양한 성능 지표를 통해 복합 접근 방식의 효과성을 평가하였으며, LLM만을 사용한 기준 모델에 비해 CQA 성능에서 현저한 향상을 보였습니다. 연구 결과에 따르면, 제안된 방법은 평가 벤치마크에서 기존 방법보다 최대 8% 높은 성능을 나타냈습니다.



### Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws (https://arxiv.org/abs/2404.05405)
- **What's New**: 언어 모델의 크기와 능력 간의 관계를 설명하는 스케일링 법칙에 관한 연구입니다. 이전 연구와 달리 모델이 저장하는 지식 비트(knowledge bits)의 수를 추정하여 평가합니다. 특히, 위키피디아 페이지에서 추출한 사실 지식을 나타내는 튜플(tuple) 형태, (USA, capital, Washington D.C.) 등에 중점을 둡니다. 여러 데이터셋을 통해 언어 모델이 파라미터당 2 비트의 지식을 저장할 수 있다는 것을 입증하며, 이러한 지식은 다운스트림(downstream) 애플리케이션에서 유연하게 추출될 수 있습니다.

- **Technical Details**: 연구는 언어 모델의 지식 저장 능력에 영향을 미치는 여섯 가지 요인을 조사합니다: (1) 교육 기간(training duration), (2) 모델 아키텍처(model architecture), (3) 양자화(quantization), (4) 희소 제약 조건(sparsity constraints) 예를 들어, Mixture of Experts(MoE), (5) 데이터 신호 대 잡음비(data signal-to-noise ratio). 특히, GPT-2 아키텍처는 모델 교육 기간이 짧을 때 LLaMA/Mistral 아키텍처를 능가하거나 비슷한 지식 저장 능력을 보여줍니다. 이는 LLaMA/Mistral이 GatedMLP를 사용하는데 이는 불안정하고 학습하기 어렵기 때문입니다.

- **Performance Highlights**: 7B 크기의 모델은 약 14B 비트의 지식을 저장할 수 있고, 이는 우리의 추정에 따르면 영어 위키피디아와 교과서를 합친 것보다 많습니다. 데이터에 도메인 이름(domain names)을 추가하는 전처리(preprocessing)는 모델의 지식 용량을 크게 증가시킨다는 점도 관찰되었습니다.



### SafetyPrompts: a Systematic Review of Open Datasets for Evaluating and  Improving Large Language Model Safety (https://arxiv.org/abs/2404.05399)
- **What's New**: 이 연구는 LLM (대규모 언어 모델, Large Language Models)의 안전성을 평가하고 향상시키기 위한 열린 데이터셋에 대한 첫 체계적 리뷰를 수행합니다. 이 연구는 2018년 6월부터 2024년 2월까지 발표된 102개의 데이터셋을 검토하고, 이를 통해 LLM 안전성 연구의 빠른 성장과 다양성을 보여줍니다. 또한, 안전성 평가에 사용되는 주요 데이터셋과 이러한 데이터셋의 활용 방식을 분석하여 현재의 평가 관행이 매우 특이하며 사용 가능한 데이터셋의 일부만을 사용하고 있음을 지적합니다.

- **Technical Details**: 이 리뷰는 웹사이트 SafetyPrompts.com을 활용하고, 트위터(Twitter)와 레딧(Reddit)을 통해 LLM 안전성 커뮤니티로부터 피드백과 데이터셋 제안을 받는 커뮤니티 중심의 반복적인 접근 방식을 사용합니다. 데이터셋은 텍스트 모달리티(Text Modality)에 초점을 맞추며 다양한 형식과 크기로 구성됩니다. 리뷰된 데이터셋은 GitHub 또는 Hugging Face를 통해 접근 가능하며, 이 연구는 이러한 데이터셋들이 어떻게 실제로 사용되는지를 분석하여 평가 관행의 표준화 필요성을 강조합니다.

- **Performance Highlights**: 이 연구는 데이터셋 생성이 학계와 비영리 단체에 의해 주도되고 있으며, 우선적으로 영어(English) 데이터셋이 지배적이라는 점을 발견했습니다. 또한 데이터셋 평가의 주요 결과로, 현재의 LLM 안전성 데이터셋 사용이 아이디오싱크레틱(idiosyncratic)하며, 사용 가능한 데이터셋의 소수만이 활용되고 있음을 지적합니다. 안전성 관련 데이터셋의 생성과 활용에 있어 더 광범위하고 체계적인 접근이 필요함을 강조하며, 이를 위한 지속적인 업데이트와 표준화를 제언합니다.



### NLP Progress in Indigenous Latin American Languages (https://arxiv.org/abs/2404.05365)
Comments: Accepted at NAACL 2024

- **What's New**: 이 논문은 급속한 기술 발전의 시대에 자생적 언어 커뮤니티의 소외 문제에 초점을 맞추고 있습니다. 특히, 자생 언어가 자연어 처리(Natural Language Processing, NLP) 분야에서 간과되는 위험과 이러한 언어의 문화적 풍부함을 강조하여, 이들 커뮤니티와 연구자들 간의 격차를 좁히고자 합니다.

- **Technical Details**: 이 논문은 라틴 아메리카의 자생 언어들이 NLP 연구에서 어떻게 진전되고 있는지를 조사했습니다. 특히, 라틴 아메리카의 자생 언어 상태, NLP에서의 표현, 보존 및 발전을 위한 도전 과제와 혁신을 다룹니다. 논문은 라틴 아메리카 자생 커뮤니티의 NLP 필요성과 진전에 대한 문헌 연구에 기여합니다

- **Performance Highlights**: NLP 연구에 표현된 라틴 아메리카의 자생 언어는 총 22개에 불과하지만, 퀘추아어(Quechua), 나우아틀(Nahuatl), 아이마라(Aymara), 그리고 과라니(Guarani) 같은 언어는 다양한 NLP 작업에서 연구가 이루어지고 있습니다. 그러나 대다수의 언어는 여전히 충분한 연구가 이루어지지 않아, 기계 번역(machine translation, MT)과 같은 중요한 NLP 작업에서 적극적인 개발이 필요한 상황입니다.



### Towards Objectively Benchmarking Social Intelligence for Language Agents  at Action Lev (https://arxiv.org/abs/2404.05337)
- **What's New**: 새로운 벤치마크인 Social Tasks in Sandbox Simulation (STSS)이 소개되었습니다. 이는 샌드박스 시뮬레이션의 멀티 에이전트(multi-agent) 상황에서 언어 에이전트(language agents)들이 사회적 작업을 얼마나 정확하게 수행하는지를 객관적으로 평가합니다. 이전의 언어 수준 평가(language level evaluations)와 대비하여 실제 행동 수준(action level)에서의 성능을 평가하여, 보다 실질적인 사회적 지능(social intelligence) 평가를 가능하게 합니다.

- **Technical Details**: STSS는 Smallville 환경을 기반으로 하여 5개 카테고리에 걸쳐 30개의 사회적 작업 템플릿을 구현하였습니다. 목표 주도 계획(Target-Driven Planning, TDP) 모듈이 추가적으로 도입되어, 특정 사회적 작업에 대해 에이전트가 언어 모델(language models)를 어떻게 활용하는지에 대한 구조적 접근을 제공합니다. 시뮬레이션 비용을 고려해서 언어 수준의 예비 벤치마크도 제공하여, 기존의 벤치마크와의 일치성도 추구합니다.

- **Performance Highlights**: STSS 벤치마크는 최신 LLMs (예: GPT-4)에도 도전적인 것으로 나타났으며, 언어 에이전트의 성능에 있어서 명확한 차이를 보여주었습니다. 특히, TDP 모듈은 에이전트의 사회적 작업 수행능력을 상당히 향상시키는 것으로 보고되었습니다. 이는 STSS가 사회적 지능 평가 뿐만 아니라, 에이전트 아키텍처(agent architectures)에 대한 테스트베드로서도 유용함을 시사합니다.



### PORTULAN ExtraGLUE Datasets and Models: Kick-starting a Benchmark for  the Neural Processing of Portugues (https://arxiv.org/abs/2404.05333)
Comments: Preprint - Paper accepted for BUCC 2024

- **What's New**: 이 연구는 포르투갈어의 신경 언어 모델링에 관한 연구를 바탕으로 세부 언어 처리 작업 및 해당 작업의 신경 언어 모델 세부 조정 컬렉션을 기여합니다. 특히, 이 연구는 포르투갈어(유럽 및 브라질 변종)에 대해 다운스트림 작업에 대해 세부 조정된 언어 모델과 데이터셋을 포함하는 PORTULAN ExtraGLUE 벤치마크도 소개하였습니다.

- **Technical Details**: 이 논문에서는 최신 기계 번역(Machine Translation, 이하 MT)을 사용하여 영어로 개발된 GLUE 및 SuperGLUE 벤치마크의 태스크를 포르투갈어로 변환하여 몇 가지 버전을 제공합니다. 또한, 유럽 및 브라질 포르투갈어로 개발된 벤치마크를 위해 Albertina 언어 모델을 사용한 저위킷(lo-rank adaptation) 접근 방식을 적용하여 이를 미세 조정하였습니다.

- **Performance Highlights**: 이 연구는 가장 첫 번째로 포르투갈어 데이터셋에 대해 세부 조정된 언어 모델을 제공함으로써 포르투갈어의 신경 처리 연구에 기여할 것으로 기대됩니다. 또한, 저위킷(lo-rank) 어댑터는 적은 훈련 매개변수로 모델 성능을 향상시킬 뿐만 아니라 저장 공간 요구를 줄이고, 추론 시간에 지연을 추가하지 않는 이점을 제공합니다.



### Multi-Task Learning for Features Extraction in Financial Annual Reports (https://arxiv.org/abs/2404.05281)
Comments: Accepted at MIDAS Workshop at ECML-PKDD 2022

- **What's New**: 본 연구에서는 재무보고서 내용의 감성, 객관성 및 전망성을 분류하고, 다중 작업 학습(Multi-Task Learning; MTL)을 통하여 환경, 사회, 거버넌스(ESG) 개념과의 연계성을 탐구한다. 전통적인 재무지표만으로는 파악하기 어려운 기업의 비재무적 성과를 분석하기 위해 텍스트 데이터에서 스타일리스틱 특성을 추출하는 다양한 방법을 제안한다. 또한, FTSE350 기업의 연차 보고서를 통해 ESG 관련 텍스트 특성과 ESG 정량 점수 간의 연관성을 분석한다.

- **Technical Details**: 이 연구는 사전 학습된 언어 모델을 사용하여 여러 분류 작업에 대해 동시에 미세조정하는 MTL 방법을 적용한다. 주목할만한 MTL방법 중 하나인 ExGF(Explicitly Give as Feature)는 보조 작업의 예측 결과를 대상 작업의 학습에 명시적으로 사용하는 방식으로, 전통적인 파라미터 공유 MTL 시스템보다 우수한 성능을 보여 주었다. 연구 결과는 감성, 객관성 및 전망성의 개념을 다루는데 있어 다중 작업 학습이 가지는 장점과 효율성을 강조한다.

- **Performance Highlights**: MTL 설정하에서의 성능 분석 결과, ExGF는 다른 MTL 시스템에 비해 효과적임을 입증하였다. 이를 통해 재무 텍스트 분류 작업에서 보조 작업의 예측 결과를 추가함으로써 최종 대상 작업의 성능이 개선될 수 있음이 증명되었다. 또한, 연차 보고서의 ESG 관련 텍스트 특성 추출 및 이를 통한 ESG 정량 점수와의 관계 분석은 관련 리터러처에서 중요한 기여를 하며, 투자자 및 정책 입안자에게 유의미한 인사이트를 제공한다.



### Interpreting Themes from Educational Stories (https://arxiv.org/abs/2404.05250)
Comments: Accepted at LREC-COLING 2024 (long paper)

- **What's New**: 이 연구에서는 기존의 기계 독해(Machine Reading Comprehension, MRC)의 범위를 넘어 해석적 이해(interpretive comprehension)에 초점을 맞추고 있습니다. 특히, 교육적 서사 텍스트의 주제를 해석하는 것을 목표로 하며, 이를 위해 'EduStory'라는 새로운 데이터셋을 소개합니다. 이 데이터셋은 다양한 장르와 문화적 배경을 가진 교육적 이야기들로 구성되어 있으며, 각 이야기의 주제를 판단하는 데 필요한 키워드가 인간에 의해 주석(annotation)되어 있습니다.

- **Technical Details**: EduStory 데이터셋은 교육적 이야기와 해당 테마 문장들을 포함하고 있으며, 이야기-테마 매칭(story-theme matching), 테마 식별(theme identification) 같은 여러 NLP 작업을 정의하고 있습니다. 연구팀은 대규모 언어 모델(Large Language Models, LLMs)을 활용하여 주제 문장을 생성하는 기능도 탐구하고 있습니다. 데이터셋은 공개적으로 접근 가능하며, 이는 연구 커뮤니티에 매우 유용한 자원이 될 것입니다.

- **Performance Highlights**: 실험 결과, 주제 해석 작업은 최신 LLM을 포함한 다양한 기계학습 방법을 사용함에도 불구하고 여전히 도전적인 과제로 남아 있습니다. 그러나 LLMs는 인간 평가자에 비해 주제 문장 생성에서 상당한 능력을 보여주었으나 완벽하지는 않다는 결과를 보여줍니다.



### Supervised Gradual Machine Learning for Aspect Category Detection (https://arxiv.org/abs/2404.05245)
- **What's New**: 이 논문에서는 기존의 DNN(Deep Neural Networks) 기반 ACD(Aspect Category Detection) 방법을 개선하여, GML(Gradual Machine Learning)과 결합한 새로운 접근법을 제안합니다. 이 연구는 라벨이 지정되지 않은 인스턴스 간의 지식 전달을 활용하여 점진적 추론을 가능하게 합니다.

- **Technical Details**: ACD는 리뷰 문장 속의 명시적 또는 암시적 영역을 식별하는 기술입니다. 본 연구에서는 DNN의 의미 관계 모델링 능력과 GML의 비독립적 동일 분포(non-i.i.d) 학습 접근법을 혼합하여 사용합니다. 우선, DNN을 통해 학습된 잠재 공간에서 인스턴스 간의 유사성 또는 대립성을 모델링하고, 이를 인자 그래프(factor graph)에 이진 특성으로 표현하여 지식을 효율적으로 전달합니다. 추가적으로, BERT 기반 모델을 이용해 주어진 카테고리와 관련된 의미적 관계를 예측합니다.

- **Performance Highlights**: 제안된 방법은 실제 벤치마크 데이터셋에서 기존의 DNN 기반 방법들을 일관되게 능가하는 성능을 보였습니다. 특히, 범주를 채점할 때의 선명도와 정확성 면에서 상당한 향상을 보였으며, 이는 GML과 DNN의 결합이 ACD 작업에 효과적임을 입증합니다.



### Product Description and QA Assisted Self-Supervised Opinion  Summarization (https://arxiv.org/abs/2404.05243)
- **What's New**: 이번 연구에서는 제품 설명과 질문-답변(Question-Answers, QA)과 같은 추가적인 소스의 활용가능성을 고려하여 의견 요약(opinion summarization)의 새로운 방법을 제안합니다. 기존에는 대부분 리뷰만을 활용하는 반면, 이 연구는 다양한 소스를 통합하여 보다 풍부하고 정보적인 요약을 생성하는 Multi-Encoder Decoder framework for Opinion Summarization (MEDOS) 모델을 개발하였습니다.

- **Technical Details**: 제안된 MEDOS 모델은 각각의 소스(리뷰, 제품 설명, QA)에 대해 별도의 인코더를 사용하며, 이를 통해 요약 생성 시 필요한 정보를 효과적으로 선택할 수 있습니다. 또한, 실제 운영 데이터가 없는 상황에서 synthetic dataset creation (SDC) 전략을 적용하여 인공적인 데이터셋을 생성하고, 이를 활용한 감독 학습(supervised training)을 진행합니다. 학습 데이터의 부재 문제를 해결하기 위해 제안된 SDC 방법론은 리뷰 및 추가 소스를 활용하여 입력 리뷰, 제품 설명, QA, 가짜 요약(pseudo-summary)의 형태로 합성 데이터 셋을 구성합니다.

- **Performance Highlights**: MEDOS 모델은 테스트 세트에 대한 실험에서 ROUGE-1 F1 기준으로 평균 14.5%의 성능 향상을 보였으며, 이는 현재 상태 최고 기술(State of the Art, SOTA) 모델을 크게 웃도는 결과입니다. 또한, 비교 및 정성적 분석을 통해 제품 설명 및 QA와 같은 추가적인 소스가 요약의 정보성을 높이는 데 중요함을 강조합니다. 인간 평가에서도 MEDOS는 일관성과 유창성에서 각각 0.41 및 0.5의 점수를 받아 기존 모델에 비해 높은 수치를 기록했습니다.



### LLM Reasoners: New Evaluation, Library, and Analysis of Step-by-Step  Reasoning with Large Language Models (https://arxiv.org/abs/2404.05221)
Comments: Project website: this https URL

- **What's New**: 이 논문에서는 복잡한 문제를 해결하기 위해 대형 언어 모델(Large Language Models, LLMs)이 정확하게 단계별 추론을 생성하는 것의 중요성을 강조합니다. 연구자들은 'AutoRace'라는 새로운 자동 추론 체인 평가 도구와 'LLM Reasoners'라는 표준화된 추론 알고리즘 라이브러리를 소개하여 추론 체인 생성에 있어 LLM과 추론 전략의 다양성을 체계적으로 분석하는 데 기여합니다.

- **Technical Details**: AutoRace는 GPT-4를 사용하여 각 작업에 맞춤화된 평가 기준을 자동 생성하고, 이를 통해 정확한 평가를 수행합니다. LLM Reasoners는 검색(search), 보상(reward), 그리고 세계 모델(world model) 구성 요소들을 통합하는 통합된 형식화를 제공하여 기존 및 새로운 추론 알고리즘을 표준화된 모듈 형태로 구현합니다.

- **Performance Highlights**: 이 라이브러리와 평가 도구를 사용하여 다양한 추론 접근 방식(예: Chain of Thought (CoT), Tree of Thought (ToT), RAP 등)을 광범위하게 연구하였습니다. 보상 지향성, 검색의 폭 대 깊이, 세계 모델, 프롬프트 형식 등 다양한 요인이 추론에 미치는 영향에 대한 흥미로운 발견들을 확인할 수 있습니다.



### Linguistic Changes in Spontaneous Speech for Detecting Parkinsons  Disease Using Large Language Models (https://arxiv.org/abs/2404.05160)
Comments: 12 pages, 3 figures

- **What's New**: 이 연구는 파킨슨병(Parkinson's disease)을 자동으로 감지하는 새로운 방법으로 대규모 언어 모델(large language models)을 사용합니다. 최근 언어 장애(language impairment)가 파킨슨병의 초기 증상으로 나타나는 것을 감지할 수 있음에 주목하면서, 연구팀은 이를 진단하는 데 도움이 될 수 있는 언어 기반 접근방식을 개발하였습니다.

- **Technical Details**: 이 연구에서 사용된 대규모 언어 모델은 자발적인 연설(spontaneous speech)로부터 파킨슨병을 자동 감지하는 데 사용됩니다. 높은 차원의 언어 표현(high-dimensional representations of linguistics)을 활용하여 기존의 접근법을 개선하고, 앙상블 기법(ensemble techniques)을 통해 다른 방법들을 향상시킬 수 있는 가능성을 제시합니다.

- **Performance Highlights**: 이 모델은 최대 73%의 정확도로 파킨슨병을 감지할 수 있으며, 이는 언어 기반 진단 방법이 유의미한 결과를 도출할 수 있음을 보여줍니다.



### Semantic Stealth: Adversarial Text Attacks on NLP Using Several Methods (https://arxiv.org/abs/2404.05159)
Comments: This report pertains to the Capstone Project done by Group 2 of the Fall batch of 2023 students at Praxis Tech School, Kolkata, India. The reports consists of 28 pages and it includes 10 tables. This is the preprint which will be submitted to IEEE CONIT 2024 for review

- **What's New**: 인공지능 기반 자연어 처리(Natural Language Processing, NLP) 모델은 기계 번역, 감성 분석, 질의응답 등 다양한 실제 응용 분야에서 중요한 역할을 합니다. 이 연구에서는 BERT 모델을 대상으로 하는 여러 가지 텍스트 적대적 공격(text adversarial attacks) 방법을 평가합니다. 특히 BERT-on-BERT 공격, PWWS 공격, Fraud Bargain's Attack (FBA) 세 가지 공격 기법을 조사하여, 이러한 공격들이 텍스트 분류 모델에 미치는 영향을 비교 분석했습니다.

- **Technical Details**: 이 논문은 BERT 모델을 ‘피해 모델(victim model)’로 사용하여, IMDB, AG News, 그리고 SST2 데이터셋을 이용해 적대적 공격 기법들의 효과를 분석합니다. BERT-on-BERT 공격은 다른 BERT 모델을 사용하여 입력 텍스트를 조작하고, PWWS(Probability Weighted Word Saliency) 공격은 단어 중요도를 계산하여 공격을 실행합니다. Fraud Bargain's Attack (FBA)는 또 다른 공격 기법으로 평가됩니다.

- **Performance Highlights**: 분석 결과, PWWS 공격이 가장 강력한 적대적 공격 방법으로 나타나며, 다른 방법들과 비교했을 때 더 낮은 실행 시간(runtime), 높은 정확도(accuracy), 그리고 유리한 의미 유사성 점수(semantic similarity scores)를 가지는 것으로 평가되었습니다. 이는 PWWS 공격이 텍스트 분류를 위한 적대적 예시(adversarial examples) 생성에 효과적임을 강조합니다.



### Enhancing Clinical Efficiency through LLM: Discharge Note Generation for  Cardiac Patients (https://arxiv.org/abs/2404.05144)
Comments: 10 pages, 1 figure, 3 tables, conference

- **What's New**: 이 연구는 인공지능(AI)을 사용하여 특히 심장 환자를 위한 퇴원 요약문을 자동 생성하는 데 초점을 맞추고 있습니다. Mistral-7B라는 대규모 언어 모델(LLM)을 사용하여 퇴원 요약문을 효율적으로 생성하고, 의료 문서의 정확성과 연속성을 향상시키는 것을 목표로 하고 있습니다. 이는 의료 분야에서 AI 기술의 직접적인 적용을 보여주는 중요한 진전입니다.

- **Technical Details**: 연구팀은 Asan Medical Center에서 제공한 광범위한 데이터셋을 사용하였고, 이는 2018년 9월부터 2021년 12월까지의 환자 기록을 포함합니다. 이 연구에서는 Mistral-7B를 포함한 여러 LLM을 사용하여 퇴원 요약문을 자동으로 생성하는 방법을 테스트했습니다. 또한, Supervised Fine-Tuning(SFT) 기술을 이용하여 모델의 성능을 특화된 의료 문서 작성 작업에 맞추어 조정했습니다.

- **Performance Highlights**: Mistral-7B 모델은 퇴원 요약문 생성에서 높은 정확도와 관련성을 보여주었습니다. 의료 전문가에 의한 질적 평가에서도 높은 점수를 받아, 의료 의사 결정과 진료 계획에 기여하는 데 효과적임이 확인되었습니다. 이는 Mistral-7B가 복잡한 의료 정보를 간결하고 일관된 요약으로 정제하는데 효과적임을 시사합니다.



### Plug and Play with Prompts: A Prompt Tuning Approach for Controlling  Text Generation (https://arxiv.org/abs/2404.05143)
Comments: 9 pages, 3 figures, Presented at Deployable AI Workshop at AAAI-2024

- **What's New**: 이 논문에서는 작은 데이터 세트에서도 효과적인 제어 생성을 달성하기 위해 'Plug and Play with Prompts (PPP)'라는 새로운 방법을 소개합니다. 이 방법은 Prompt Tuning을 사용하여 생성된 텍스트의 방향을 제어하며, 몇 백 개의 훈련 예제만을 사용하여 프롬프트(embeddings)를 훈련할 수 있습니다.

- **Technical Details**: PPP 방법은 외부 판별자 모델의 그래디언트를 사용하여 프롬프트 매개변수를 제어 명령으로 조정합니다. 이는 언어 모델의 생성을 제어하기 위해 프롬프트를 사용하고, 범주형 교차 엔트로피 손실(Categorical Cross Entropy Loss, CCE)을 새로운 방식으로 적용하여 프롬프트에 의해 생성된 텍스트의 유창성을 유지합니다. 또한, 이 방법은 소수의 데이터로도 스타일을 제어할 수 있음을 보여주며, 큰 규모의 도메인 외 훈련 데이터 설정에서 일반화할 수 있습니다.

- **Performance Highlights**: PPP는 SST-5와 Yelp (감정 분석), GYAFC (형식성), JIGSAW (유해 언어)와 같은 네 가지 데이터셋에 대한 광범위한 평가를 수행했습니다. 이 방법은 데이터 및 매개변수 효율이 뛰어나며 유해하고 편향된 텍스트 생성을 완화하는 데 효과적임을 입증합니다.



### EcoVerse: An Annotated Twitter Dataset for Eco-Relevance Classification,  Environmental Impact Analysis, and Stance Detection (https://arxiv.org/abs/2404.05133)
- **What's New**: 이 논문은 환경적 주제에 초점을 맞춘 새로운 주제 분류 데이터셋, EcoVerse를 제안합니다. EcoVerse는 트위터에서 수집한 3,023개의 트윗을 포함하며, 생물 다양성 손실, 지속 가능한 농업, 플라스틱 오염 등 다양한 환경 주제를 다룹니다. 특히 이 데이터셋은 환경 관련성 분류(Eco-Relevance Classification), 환경 영향 분석(Environmental Impact Analysis), 그리고 입장 탐지(Stance Detection)의 세 가지 주요 목표를 위해 설계된 새로운 삼중 레벨 주석 체계를 사용합니다.

- **Technical Details**: EcoVerse 데이터셋은 영어 트위터 트윗을 기반으로 하며, 각 트윗은 환경 관련 여부, 환경에 미치는 영향(유익, 해로운, 중립적), 그리고 저자의 환경 문제에 대한 입장을 분류하기 위해 수동으로 주석 처리되었습니다. 데이터 수집, 필터링, 레이블링 과정이 자세히 기술되어 있습니다. 주석 과정에서 높은 수준의 일관성을 보여주는 상당한 Inter-Annotator Agreement(주석자 간 합의)가 있었습니다. 분류 실험은 BERT 기반 모델을 사용하여 수행되었으며 ClimateBERT를 포함한 모델이 사용되었습니다. 이 데이터셋은 공개적으로 제공되어 더 많은 연구를 자극할 것입니다.

- **Performance Highlights**: 초기 분류 결과는 희망적입니다. 특히, ClimateBERT를 포함한 BERT 기반 모델을 사용한 실험에서는 환경 텍스트를 위해 특별히 맞춤화된 모델의 필요성을 시사하는 동시에 격려할 만한 결과를 도출했습니다. 이는 이분야에서의 진보 가능성을 시사하며, 환경적 텍스트 분석에 특화된 툴의 개발이 필요함을 강조합니다.



### Advancing Geometric Problem Solving: A Comprehensive Benchmark for  Multimodal Model Evaluation (https://arxiv.org/abs/2404.05091)
- **What's New**: 이 논문에서는 기하학적 계산 문제의 성능을 평가하기 위한 새로운 벤치마크인 MM-MATH 데이터셋을 제시합니다. 이 데이터셋은 9학년 수준의 복잡성과 요구 사항을 반영하는 5,929개의 기하학 문제로 구성되어 있으며, 각 문제에는 해당 이미지가 포함되어 있습니다. MM-MATH는 모델의 추론 능력과 절차적 정확성을 평가하는 새로운 방식을 제공함으로써, 멀티모달 기술(multimodal technology)에서 발생하는 중요한 격차를 밝혀내고 이를 해결하기 위한 연구를 촉진하고자 합니다.

- **Technical Details**: MM-MATH 데이터셋은 다양한 난이도와 지식 포인트가 주석된 기하 문제를 포함하고 있으며, 텍스트와 이미지를 동시에 처리할 수 있는 최신의 대규모 언어 모델들을 평가하기 위해 설계되었습니다. 데이터셋 구성은 기존의 MathML 형식에서 LaTeX 형식으로 변환하는 과정을 포함하여, GPT-4와 같은 모델들을 사용하여 정확도를 높이고 기존 중국어 데이터셋의 잡음을 최소화했습니다. 평가 방법에서는 결과뿐만 아니라 문제 해결 과정에 대한 분석을 통해 모델의 오류 원인을 조사합니다.

- **Performance Highlights**: 현재 멀티모달 모델들은 기하학적 정보를 정확하게 해석하는데 60% 이상의 오류율을 보이며, 이는 학생 수준의 능력과의 중요한 성능 격차를 나타냅니다. 특히, 고급 모델인 GPT-4V는 간단한 직선 문제에 있어 중간 단계에서의 오류를 보였습니다. 이를 통해 중간 추론 과정에서의 중요한 문제를 식별할 수 있었고, 최종 결과의 정확성에 영향을 미쳤습니다.



### SEER-MoE: Sparse Expert Efficiency through Regularization for  Mixture-of-Experts (https://arxiv.org/abs/2404.05089)
Comments: 8+3 pages

- **What's New**: 이번 연구에서는 사전 훈련된 Mixture-of-Experts (MoEs) 모델들의 메모리 및 컴퓨트 요구 사항을 줄이기 위한 새로운 두 단계 프레임워크인 SEER-MoE를 소개하였습니다. 새롭게 제안된 구조는, 전문가들의 전체 수를 줄이는 가지치기(pruning)과정과 정확도 손실을 회복하며 추론 시 활성화되는 전문가 수를 줄이기 위한 정규화 기반의 파인 튜닝(fine-tuning) 전략을 포함합니다.

- **Technical Details**: SEER-MoE는 첫 번째로 'heavy-hitters counting guidance'를 이용하여 모델의 전문가 총 수를 가지치기하는 단계를 거치고, 두 번째 단계에서는 정규화 기반 파인 튜닝을 사용하여 가지치기로 인한 정확도 손실을 회복함과 동시에 모델의 추론 효율성을 최적화합니다. 이러한 접근법은 효율적인 추론과 향상된 스케일링 잠재력을 제공하는 MoEs의 도입으로, 큰 모델 매개변수를 확장하는 동안 입력 토큰 당 일정한 컴퓨팅 공간을 유지할 수 있습니다.

- **Performance Highlights**: 실증적인 연구를 통해 SEER-MoE 방법의 효과를 입증하였으며, MoEs 모델의 메모리 요구사항을 줄이면서 추론 효율성을 최적화하는 데 있어 비용 효과적인 결과를 도출하였습니다. 이 연구는 같은 배치 방법을 사용하는 Mixtral 8x7b 모델과 SST5, MMLU 벤치마크에서 테스트되었습니다. 이 방법은 추론 중 활성화되는 전문가 수를 줄임으로써, 정확도의 최소한의 저하와 함께 메모리 요구를 줄이는 것에 대한 효과적인 접근 방식을 제공합니다.



### How much reliable is ChatGPT's prediction on Information Extraction  under Input Perturbations? (https://arxiv.org/abs/2404.05088)
Comments: 3 Figures, 7 Tables

- **What's New**: 이 연구에서는 정보 추출(Information Extraction, IE)의 기본 작업인 명명된 엔티티 인식(Named Entity Recognition, NER)에 대해 ChatGPT의 견고성(robustness)을 평가합니다. ChatGPT가 입력 변형(input perturbations)에 어떻게 반응하는지, 예측의 신뢰성(confidence) 및 예측 뒤의 근거(rationale)의 품질이 어떻게 달라지는지 체계적으로 분석했습니다. 특히, 드문 엔티티(rare entities)와 널리 알려진 엔티티들에 대한 변형의 영향을 비교하고, 컨텍스트 특정(context-specific) 변형과 엔티티 특정(entity-specific) 변형에서 예측 설명의 품질이 어떻게 달라지는지 평가했습니다.

- **Technical Details**: 연구는 zero-shot 및 few-shot 설정에서 ChatGPT의 견고성을 평가했습니다. 입력 데이터는 '엔티티 수준(entity-level)'에서 (Wikidata의 동일한 의미 클래스에 속하는 다른 엔티티로 대상 엔티티를 교체) 및 '컨텍스트 수준(context-level)'에서 (예를 들어, BERT를 사용해 문맥적 동사 치환을 생성) 변경되었습니다. 자동 평가와 인간 평가를 모두 사용하여 ChatGPT의 예측, 예측 신뢰성 및 근거의 품질이 어떻게 다른지 평가했습니다.

- **Performance Highlights**: 자동 평가 결과에 따르면, ChatGPT는 드러그나 질병과 같은 드문 엔티티에 대한 변형에서 더 취약(brittle)하며, 잘못된 예측에 대해 과도하게 자신감(overconfident)을 가지는 경향이 있습니다. 반면, 다양한 유형의 특정 엔티티 및 컨텍스트 변형 하에서 동일한 엔티티에 대한 설명의 품질은 크게 차이가 나는 것으로 나타났습니다. 인간 평가는 이러한 결과들을 더욱 강화시키며, 또한 페르토베이션(perturbation) 하에서 엔티티 예측의 용이성과 ChatGPT의 행동 간의 상관관계에 대한 통찰을 제공합니다.



### MLaKE: Multilingual Knowledge Editing Benchmark for Large Language  Models (https://arxiv.org/abs/2404.04990)
- **What's New**: 새로운 벤치마크 MLaKE(Multilingual Language Knowledge Editing)는 다양한 언어(영어, 중국어, 일본어, 프랑스어, 독일어)에서 지식 편집 방법의 적응성을 평가하기 위해 4072개의 멀티 홉 질문과 5360개의 싱글 홉 질문을 포함합니다. 이 벤치마크는 다국어 및 다단계 추론 문제에 대한 기존 지식 편집 방법의 한계를 체계적으로 다루며, 다양한 언어 환경에서의 효과적인 지식 편집 기술 개발을 위한 기초를 마련합니다.

- **Technical Details**: MLaKE는 위키디피아(Wikipedia)에서 언어별 사실 체인을 수집하고, 강력한 LLM (예: ChatGPT)을 사용하여 자유 형식 및 객관식으로 질문을 생성합니다. 앤트리  데이터셋(entry dataset)은 각 언어별 싱글 홉과 멀티 홉 질문을 포함합니다. MLaKE 데이터셋은 고급 지식 편집 방법을 평가하기 위해 사용되며, 연구 결과는 다국어 실험에서 일반화 능력이 제한적임을 보여주었습니다. 특히, 동일 언어 가족 내에서 높은 일반화 성능을 나타낸 반면, 서로 다른 언어 가족 간에는 상대적으로 낮은 성능을 보였습니다.

- **Performance Highlights**: MLaKE 데이터를 이용한 평가에서 기존 지식 편집 방법들은 영어 샘플에서는 높은 성공률을 보였지만, 다른 언어 및 다국어 실험에서는 일반화 능력이 한정적이다는 점이 밝혀졌습니다. 이는 지식 편집 과정에서 언어 간의 차이가 효율성에 상당한 영향을 미칠 수 있음을 강조합니다.본 연구는 향후 다국어 지식 편집 기술의 발전을 위한 기반을 마련하였으며, MLaKE 데이터셋이 미래 연구 및 솔루션 개발에 가치 있는 자원이 될 것으로 기대합니다.



### SemEval-2024 Task 2: Safe Biomedical Natural Language Inference for  Clinical Trials (https://arxiv.org/abs/2404.04963)
- **What's New**: 새로운 'SemEval-2024 Task 2: Safe Biomedical Natural Language Inference for ClinicalTrials' 과제가 제시되었습니다. 이 과제는 LLMs (Large Language Models)의 보다 철저한 평가를 위해 설계된 개선된 NLI4CT-P (Natural Language Inference for Clinical Trials - Perturbed) 데이터셋을 활용하고, 이를 통해 의료 분야에서의 클리니컬 시험 보고서의 원인 추론 및 개입 추론 과제를 통해 도전합니다. 데이터셋과 참가 시스템의 평가는 robustness (견고성) 및 applicability (적용 가능성)에 중점을 둔 다양한 방법론을 통해 수행되었습니다.

- **Technical Details**: NLI4CT-P 데이터셋은 Multiple Evidence NLI4CT에 기반하여 생성되었으며 요구된 새로운 평가 기준(Consistency and Faithfulness)을 통해 모델 성능을 더 심도 있게 분석할 수 있습니다. 이 과제는 특히 semantic equivalence (의미적 동등성) 및 clinical scenario (임상 시나리오)에서 중요한 추론 과정을 처리하는 NLI 모델의 기능을 측정하고자 합니다. 또한 중간 크기의 모델이 대용량 모델과 경쟁할 수 있는 성능을 보여주며, 특히 zero-shot prompting 기법이 성능 향상에 효과적임을 강조합니다.

- **Performance Highlights**: SemEval-2024 Task 2에서 참가 시스템은 최대 F1 score이 0.8에 이르는 성과를 보였으며, 이는 Mixtral-8x7B-Instruct 모델을 통해 달성되었습니다. 추가로, generative models (생성 모델)이 discriminative models (판별 모델)보다 높은 F1 score, Faithfulness 및 Consistency 점수를 얻었으며, 모델의 성능에 영향을 미치는 중요 요소로 prompting strategy의 선택이 지적되었습니다. training data (교육 데이터)를 추가로 활용할 때 성능 향상이 있었으며, mid-sized architectures (중간 크기의 구조)가 70B parameter를 초과하는 모델들과 경쟁할 수 있는 성능을 제공했습니다.



### A Two Dimensional Feature Engineering Method for Relation Extraction (https://arxiv.org/abs/2404.04959)
- **What's New**: 이 논문에서는 2차원(2D) 문장 표현을 활용한 특성 공학 방법을 제안하여 관계 추출(RE, Relation Extraction) 작업에 적용하였습니다. 전통적인 특성 공학에서의 사전 지식을 최대한 활용하면서, 추출된 특성을 명시적으로 결합하여 새로운 2D 표현 방식을 통해 문맥적 특성과 의미적 의존성을 학습할 수 있습니다.

- **Technical Details**: 제안된 방법은 문장을 의미적 평면으로 변환하고, 이 평면에 추출된 특성을 주입하는 점을 명확하게 구성합니다. 또한, 연합된 특성 인식(aware) 주의 메커니즘을 도입하여 개체와 결합된 특성 간의 관계를 더 깊이 이해할 수 있도록 설계하였습니다. 이를 통해 문장 내에서 중첩된 관계 인스턴스를 효과적으로 해결할 수 있습니다.

- **Performance Highlights**: 3개의 공중 데이터셋(ACE05 중국어, ACE05 영어, SanWen)에서 상태 기술(state-of-the-art) 성능을 달성하였으며, 이는 2D 문장 표현과 결합된 특성을 통한 정보 증강이 효과적임을 입증합니다. 제안된 모델은 기존 기법들을 능가하는 성능을 보여줍니다.



### SilverSight: A Multi-Task Chinese Financial Large Language Model Based  on Adaptive Semantic Space Learning (https://arxiv.org/abs/2404.04949)
Comments: 17 pages, 17 figures

- **What's New**: 이 연구는 Adaptive Semantic Space Learning (ASSL) 프레임워크를 소개하며, 이는 의미 공간(semantic space)에서 데이터 분포를 적응적으로 재구성하여 다중 전문가 모델들의 성능과 선택 효율성을 향상시키는 데 사용됩니다.

- **Technical Details**: ASSL 프레임워크는 중국 금융 멀티태스크 데이터셋을 사용하여 'SilverSight'라는 금융 멀티태스크 대형 언어 모델(Large Language Model, LLM)을 훈련합니다. 이 프레임워크는 유사한 의미 공간을 기반으로 멀티태스크 훈련 데이터를 클러스터링하고, 각 LoRA 전문가에게 가장 관련성 높은 다운스트림 태스크를 할당함으로써 전통적인 사전 정의된 분류 방법보다 명확한 이점을 보였습니다.

- **Performance Highlights**: 이 프레임워크는 전체 데이터셋으로 훈련한 결과와 유사한 효과를 단 10%의 데이터 사용으로 달성할 수 있으며, 다양한 태스크에서 모델의 일반화 능력을 크게 향상시키는 것으로 나타났습니다. 또한, 비슷한 의미 공간 내 클러스터를 형성함으로써 상호 강화되거나 충돌하는 훈련 태스크를 식별할 수 있었고, 이로 인해 각 전문가 모델이 자신의 전문 분야에 집중할 수 있게 되었습니다.



### Prompting Large Language Models for Zero-shot Essay Scoring via  Multi-trait Specialization (https://arxiv.org/abs/2404.04941)
- **What's New**: 이 논문에서는 대규모 언어 모델(Large Language Models, LLMs)을 활용한 자동 에세이 채점(Automated Essay Scoring, AES)의 새로운 접근 방식인 '멀티 트레잇 스페셜라이제이션(Multi Trait Specialization, MTS)'을 제안합니다. MTS는 LLM들이 에세이를 평가할 때 각기 다른 쓰기 특성을 분리하고 각 특성에 맞는 채점 기준을 생성하여 점수를 도출하는 제로샷(zero-shot) 프롬프팅 프레임워크입니다.

- **Technical Details**: MTS는 ChatGPT를 사용하여 쓰기 능력을 여러 특성으로 분해하고 각 특성에 대한 채점 기준을 생성합니다. LLM은 여러 라운드의 대화를 거쳐 각 특성을 평가하고 채점 기준에 따라 점수를 추출합니다. 최종적으로, 모든 특성 점수의 평균을 내고 min-max 스케일링(min-max scaling)을 통해 전체 점수를 도출합니다.

- **Performance Highlights**: 실험 결과, MTS는 TOEFL11 및 ASAP 벤치마크 데이터셋 모두에서 평균 쿼드래틱 가중 카파(Quadratic Weighted Kappa, QWK)에서 기존의 단순 프롬프팅 방식인 'Vanilla'를 일관되게 상회했습니다. 또한, MTS를 사용한 경우, 작은 크기의 Llama2-13b-chat이 ChatGPT를 상당히 앞섰으며, 감독 학습(supervised learning)을 기반으로 한 최신 기술(State of the Art, SOTA)에 근접하는 성과를 보였습니다.



### Towards Understanding the Influence of Reward Margin on Preference Model  Performanc (https://arxiv.org/abs/2404.04932)
- **What's New**: 이 연구는 인간의 피드백에서 강화 학습(RLHF: Reinforcement Learning from Human Feedback)을 사용하여 언어 모델을 개발하는 과정에서 발생하는 보상 모델 최적화 문제에 주목합니다. 특히, 기존의 보상 모델이 인간의 선호도 데이터를 기반으로 한 전통적인 순위 결정 목표를 사용하여 훈련될 때 실제 상황에서 더 좋거나 나쁜 응답을 효과적으로 구분하는 데 어려움을 겪는다는 문제점을 지적합니다. 이를 해결하기 위해, 저희는 보상 모델 훈련 과정에 마진 점수(margin score)를 도입하는 새로운 방법을 제안합니다.

- **Technical Details**: 저희의 방법론은 상이한 응답들 사이의 선호도 차이를 정량화하는 마진 점수를 통합함으로써 보상 모델이 더 선호되는 응답을 인식하고 우선순위를 두는 능력을 향상시킵니다. 이 마진 점수는 보상 모델이 훈련 데이터셋의 모호하거나 잘못된 선호도로 인해 발생하는 일관성 없는 데이터를 걸러내는 데 도움을 줍니다. 또한, 본 연구는 보상의 확신도(reward confidence)를 기반으로 선호도 차이를 추정하는 새로운 방식을 도입하여 인간의 주석자로부터 자세한 라벨이 필요 없게 합니다.

- **Performance Highlights**: 실험 결과, 마진 값을 훈련 과정에 통합함으로써 보상 모델의 효과성이 크게 향상되었음을 실증적으로 입증합니다. 이는 보상 예측 정확성(reward prediction accuracy)뿐만 아니라 실제 응용에서의 효과성을 강조하는 비교 분석을 통해 우리의 접근 방식의 우수성을 보여줍니다.



### Multilingual Large Language Model: A Survey of Resources, Taxonomy and  Frontiers (https://arxiv.org/abs/2404.04925)
- **What's New**: 이 연구는 다양한 언어를 지원하는 대규모 언어 모델(MLLMs)에 대한 최초의 포괄적 검토를 제공합니다. 저자들은 MLLMs의 최근 진보를 요약하고 새로운 분류(taxonomy)를 소개하여, 매개변수 조정(parameter-tuning) 대 매개변수 고정(parameter-frozen) 정렬 전략에 따라 MLLMs를 분류합니다. 또한, 미래 연구를 위한 새로운 전선(new frontiers)을 탐구하고 관련 도전 과제를 논의합니다.

- **Technical Details**: MLLMs는 여러 언어를 동시에 처리할 수 있는 능력을 갖추고 있으며, 이는 전통적인 단일 언어 모델(LLM)과 차별화됩니다. 연구에서는 두 가지 주요 전략, 즉 매개변수 조정 정렬과 매개변수 고정 정렬을 통해 언어 간 정렬을 달성하는 방법을 소개합니다. 매개변수 조정 정렬은 모델이 더 나은 언어 간 정렬을 위해 사전 훈련, 감독된 미세조정, 인간 피드백으로부터의 강화 학습 및 하류(downstream) 미세조정 단계에서 매개변수를 미세 조정하는 것을 요구합니다. 반면, 매개변수 고정 정련은 매개변수 조정 없이 언어 간의 프롬프트를 통해 정렬을 달성합니다.

- **Performance Highlights**: 이 연구는 MLLMs의 발전을 요약하고 새로운 분류를 제안함으로써, 다양한 언어 처리를 개선하는 방법에 대한 이해를 돕고 있습니다. 또한, 유용한 자원을 수집하고 공개하여, 연구자들이 쉽게 접근하고 MLLMs 연구를 진행할 수 있도록 합니다.



### Radial Networks: Dynamic Layer Routing for High-Performance Large  Language Models (https://arxiv.org/abs/2404.04900)
Comments: First two authors have equal contribution

- **What's New**: 이 연구에서는 대규모 언어 모형에서 새로운 동적 계층 희소성(Dynamic layer sparsity)을 탐구하여, 각 입력에 대해 개별 레이어를 건너뛰는 기능을 제공합니다. 이는 특히 OPT와 ViT 모델에서 중간 레이어의 중요성을 프로파일링하고, 모델의 깊이와 동적 희소성 사이의 관계를 설정함으로써 구현되었습니다. 또한, Radial Network라는 새로운 네트워크 구조를 제안하여 토큰 수준에서 레이어 간의 라우팅을 수행하는 학습된 라우터 모듈을 사용하여 계산 리소스를 절약합니다.

- **Technical Details**: 이 연구는 특히 OPT-66B 모델의 잔여 블록(Residual blocks)이 출력에 대해 중간 기여도가 5%인 것을 발견했습니다. Radial Network는 전체 시퀀스를 생성하는 데 필요한 리소스를 줄이기 위해 토큰별로 계산을 조정합니다. 이 네트워크는 네트워크의 동적 깊이를 레이어 수와 분리하여 더 큰 모델 크기로 확장할 수 있도록 설계되었습니다. 또한, 레이어 재사용(Layer reuse)을 허용하는 설계를 포함하고 있습니다.

- **Performance Highlights**: 동적 계층 희소성을 통해 큰 규모의 언어 모델에 대한 계산 및 서비스 비용을 크게 줄일 수 있습니다. 예를 들어, Radial Networks는 전통적인 순차 네트워크에서의 사후 학습 법(Post-training distillation)이나 처음부터의 공동 학습(Co-learning)을 통해 training을 수행할 수 있으며, 이는 전체적인 모델의 효율성을 증가시킵니다.



### Ethos and Pathos in Online Group Discussions: Corpora for Polarisation  Issues in Social Media (https://arxiv.org/abs/2404.04889)
- **What's New**: 이 연구는 온라인에서 개인이 사용하는 수사 전략을 분석함으로써 극화 현상(polarisation)을 다루는 새로운 접근 방식을 제시합니다. 특히, 아리스토텔레스(Aristotelian)의 두 설득 방식인 에토스(ethos)와 파토스(pathos)에 초점을 맞춘 수동적 어노테이션(manual annotation)을 통한 멀티-토픽 및 멀티-플랫폼(corpus)을 개발하였습니다. 이는 온라인 커뮤니케이션 전략의 대규모 연구를 진전시키기 위한 언어 모델(language models) 훈련에 활용될 수 있습니다.

- **Technical Details**: 연구자들은 기존의 정서 분석(sentiment analysis) 및 증오 연설 감지(hate speech detection) 도구만으로는 온라인 영역에서의 극화 현상을 충분히 이해하고 연구하는 데 한계가 있다고 지적합니다. 에토스와 파토스에 대한 어노테이션은 각각 연사의 신뢰성(credibility)과 청자의 감정 상태(emotional state)를 다루며, 이러한 수사 전략은 갈등 및 극화 상황에서의 의사소통 관행을 보다 포괄적으로 파악할 수 있게 합니다. 데이터는 3가지 극화 주제(polarising topics)에 대한 온라인 토론을 포함하고 2개의 소셜 미디어 플랫폼에서 수집되었습니다.

- **Performance Highlights**: 이 언어 자원을 사용하여, 생성된 자동화 도구(automatic tools)는 극화 토론의 반복 패턴을 탐지하고, 주제 또는 플랫폼에 따라 달라지는 극화 패턴과 보편적인 경향을 분석할 수 있습니다. 또한, 이 연구는 극화된 토론에 대한 양적(quantitative) 및 질적(qualitative) 분석을 수행하여 극화 현상의 다양한 수준을 조사할 수 있게 해줍니다. 최종 목표는 소셜 미디어 논쟁에서 극화를 이해하고 연구하는 데 기여하며 이는 학계 및 실제 응용 분야에서 중요한 시사점을 제공할 수 있습니다.



### Lucky 52: How Many Languages Are Needed to Instruction Fine-Tune Large  Language Models? (https://arxiv.org/abs/2404.04850)
- **What's New**: 이 논문은 대규모 다국어 모델을 다양한 언어로 미세 조정(fine-tuning)하여 다국어 작업에 대한 모델의 이해도를 향상시키는 방법을 탐구합니다. 특히, BLOOM 모델을 사용하여 52개 언어로 된 Bactrain-X 데이터셋을 활용하여 언어 능력을 점진적으로 향상시키는 다국어 지시어 미세 조정(multilingual instruction fine-tuning, mIT) 실험을 수행했습니다.

- **Technical Details**: 먼저 영어와 중국어와 같은 자원이 풍부한 언어에서 시작하여 알파벳 순으로 다른 언어를 점진적으로 추가하는 방식으로 mIT을 실행했습니다. 이 과정에서 각 언어에 대한 지시어 데이터는 일관된 지시 정보를 유지하면서 데이터 크기의 전반적인 증가를 최소화하기 위해 다른 언어로 번역되었습니다. 사용된 Bactrain-X 데이터셋은 52개 언어, 340만 개의 지시어-응답 쌍을 포함합니다. mIT의 효과를 측정하기 위해 여러 다국어 벤치마크를 사용하여 모델을 평가했습니다.

- **Performance Highlights**: 실험 결과는 몇 개의 언어를 추가하는 것만으로도 다국어 전달 능력이 향상될 수 있음을 보여줍니다. 그러나 최적의 다국어 작업 성능을 달성하기 위한 언어의 최적 수는 언어 유사성 및 하위 평가(downstream evaluation)에 따라 다를 수 있습니다. 대체로, 언어를 더 많이 추가할수록 정확도가 향상되지만, 특정 사례에서는 눈에 띄게 향상되거나 감소하는 경향을 보였습니다. 이는 벤치마크와 관련 언어에 따라 mIT의 영향이 다양할 수 있음을 시사합니다.



### F-MALLOC: Feed-forward Memory Allocation for Continual Learning in  Neural Machine Translation (https://arxiv.org/abs/2404.04846)
Comments: Accepted to the main conference of NAACL 2024

- **What's New**: 본 논문에서는 신경 기계 번역(Neural Machine Translation, NMT)의 영역에서, 연속 학습(Continual Learning, CL) 방법이라는 새로운 접근방식인 F-MALLOC(Feed-forward Memory ALLOCation)를 제안합니다. 이 방법은 피드포워드(Feed-forward) 층이 신경 메모리 역할을 한다는 최근의 통찰을 바탕으로 하여, 기억을 효과적으로 할당하고 보호함으로써 재앙적 망각(Catastrophic Forgetting, CF)을 완화하고 시스템의 확장성을 유지합니다.

- **Technical Details**: F-MALLOC은 먼저 기존 NMT 모델의 피드포워드 층을 구조적 프루닝(Structural Pruning) 방식으로 다듬고, 중요한 일반 도메인 지식을 보존합니다. 이후, 새로운 태스크에 대해 '쓸 수 있는' 메모리 용량을 할당하고, 이 메모리들을 '읽기 전용'으로 지정하여 기울기 흐름(Gradient Flow)을 차단함으로써 기억 손실을 막습니다. 추가로 복수 단계의 연속 학습 평가 프로토콜(Multi-stage CL Evaluation Protocol)을 도입하여, 다양한 태스크의 시퀀스에서 F-MALLOC의 성능을 평가합니다.

- **Performance Highlights**: F-MALLOC을 적용한 실험 결과, 이 방법은 높은 BLEU 점수를 달성하고 거의 제로에 가까운 망각을 보여주었습니다. 또한, 태스크 정보의 활용을 통해 시스템의 용량 사용을 최적화하고 지식 전달을 촉진하는 데 효과적임을 입증했습니다.



### SLPL SHROOM at SemEval2024 Task 06: A comprehensive study on models  ability to detect hallucination (https://arxiv.org/abs/2404.04845)
- **What's New**: 이 연구는 생략 추론(natural language inference, NLI), 벡터 유사도(vector similarity), 그리고 언어 모델 간의 집단 판단(ensemble judgment)을 포함한 다양한 방법을 통해 자연어 생성(Natural Language Generation, NLG)에서의 환각(hallucinations)을 감지하는 새로운 접근 방식을 제시합니다. 특히, SemEval-2024 Task 6의 세 가지 과제—기계 번역(machine translation), 정의 모델링(definition modeling), 그리고 의역 생성(paraphrase generation)—에 초점을 맞추고 있습니다.

- **Technical Details**: 연구팀은 두 가지 주요 방법을 사용하여 환각을 탐지합니다. 첫 번째는 생성된 텍스트와 사실적 참조 사이의 의미적 유사성(semantic similarity)을 평가하는 것으로, 이는 LaBSE(Language-Agnostic BERT Sentence Embeddings)와 같은 이중 인코더 모델을 사용하여 구현됩니다. 두 번째 방법은 여러 언어 모델(large language models, LLMs)이 서로의 출력을 판단하는 앙상블 접근 방식입니다. 이 방법은 메타 추론(meta-reasoning)을 도입하여 환각 감지 과정에 새로운 차원을 추가합니다.

- **Performance Highlights**: 이 두 방법은 각각 장단점을 가지고 있습니다. 의미적 유사성 방법은 중간 수준의 정확도와 상관 관계 점수를 달성하지만, 앙상블 방식은 환각 감지의 복잡성에 대한 통찰력을 제공하지만 기대에는 미치지 못했습니다. 연구팀은 이러한 방법들이 실제 세계의 애플리케이션에서 NLG의 신뢰성과 진실성을 보장하는 데 중요한 역할을 할 수 있음을 강조하면서, 환각 감지 분야의 추가 연구의 필요성을 강조합니다.



### Data Bias According to Bipol: Men are Naturally Right and It is the Role  of Women to Follow Their Lead (https://arxiv.org/abs/2404.04838)
Comments: 11 pages, 6 figures

- **What's New**: 새로운 대규모 언어 데이터셋과 여러 언어로의 편향을 탐지하는 새로운 연구가 소개되었습니다. 이 연구는 특히 영어 뿐 아니라 이탈리아어, 네덜란드어, 독일어에서도 데이터의 편향을 조사하였습니다. 이는 정치적, 사회적 편견이 다국어 대응 AI 모델에도 영향을 미칠 수 있음을 보여줍니다.

- **Technical Details**: 이 연구에서는 mT5 (multilingual Text-to-Text Transfer Transformer) 및 mBERT (multilingual Bidirectional Encoder Representations from Transformers) 모델을 사용하여 데이터셋에서의 편향을 예측하고 분류했습니다. 또한, 새로운 편향 측정 메트릭인 bipol을 사용하여 편향을 평가했으며, 이 메트릭은 편향된 용어를 이용한 감성 평가를 통해 편향도를 측정합니다.

- **Performance Highlights**: 연구팀은 영어, 이탈리아어, 네덜란드어, 독일어 데이터셋에서의 편향 존재를 확인하였습니다. 이 연구는 사회적 편견이 언어 데이터에 광범위하게 존재하며, 이것이 AI 시스템의 판단과 성능에 영향을 줄 수 있음을 입증합니다. 또한, 새로운 데이터셋과 편향 감지 어휘(Lexica)를 공개하여, 향후 연구와 개선을 위한 기초를 마련하였습니다.



### FRACTAL: Fine-Grained Scoring from Aggregate Text Labels (https://arxiv.org/abs/2404.04817)
Comments: 22 pages, 1 figure

- **What's New**: 최근 대형 언어 모델(LLM: Large Language Model)의 최적화를 위해 문장 수준의 피드백이 반응 수준 평가보다 더 정확하고 해석 가능한 결과를 제공한다는 연구가 제시되었습니다. 이를 토대로, 우리 연구는 반응 수준 레이블을 문장 수준의 가상(pseudo) 레이블로 세분화하는 새로운 방법을 소개하며, 다양한 자연어 처리 작업에 걸쳐 향상된 성능을 보이는 첫 번째 연구입니다.

- **Technical Details**: 우리의 접근 방식은 다중 인스턴스 학습(MIL: Multiple Instance Learning)과 레이블 비율 학습(LLP: Learning from Label Proportions) 기술을 활용하여 특수화된 모델을 훈련시킵니다. 이 모델은 문서-문장 코사인 유사도(document-sentence cosine similarity)와 같은 사전 정보를 사용하여 각 문장에 대한 점수를 예측합니다. 또한, 모델 예측을 사용하여 훈련 세트를 문장 수준에서 가상 레이블로 표시하고 이를 통해 모델 훈련을 더욱 개선합니다.

- **Performance Highlights**: 우리의 방법론을 정보 검색(retrieval), 질문 응답(question answering), 요약(summarization) 및 수학 추론(math reasoning) 작업에 걸쳐 6개의 데이터셋에서 평가하였습니다. 실험 결과는 대부분의 작업에서 기존 베이스라인보다 우수한 성능을 보여주었습니다. 특히, 문장 수준의 미세 조정(fine-tuning) 평가에서는 인간의 세밀한 주석이 달린 레이블로 훈련된 모델과 비교할만한 성능을 나타내었습니다.



### Low-Resource Machine Translation through Retrieval-Augmented LLM  Prompting: A Study on the Mambai Languag (https://arxiv.org/abs/2404.04809)
- **What's New**: 이 연구는 맘바이어(Mambai)로의 기계 번역(machine translation, MT)에 대한 대규모 언어 모델(large language models, LLMs)의 활용을 탐구합니다. 맘바이어는 티모르-레스테(Timor-Leste)에서 사용되는 저자원(low-resource) 오스트로네시안(Austronesian) 언어로, 약 20만 명의 원어민이 사용합니다. 연구는 새로운 매뉴얼에서 파생된 코퍼스와 원어민이 번역한 추가 문장을 활용하여 LLM을 few-shot prompting 방식으로 사용하는 것의 효과를 검토합니다.

- **Technical Details**: 우리는 교정된 문장과 사전 항목을 전략적으로 선택하여 번역 정확도를 높이고자 하며, GPT-4와 같은 다양한 LLMs(오픈소스 및 독점적인 소스)를 사용했습니다. 또한, TF-IDF 및 의미적 임베딩(semantic embeddings)을 통해 검색된 문장을 포함시켜 번역 품질이 크게 개선되었음을 발견했습니다.

- **Performance Highlights**: 언어 매뉴얼에서 추출된 문장을 사용한 테스트에서 BLEU 점수가 최대 23.5점에 이르렀으나, 원어민이 제공한 테스트 세트의 경우 BLEU 점수가 5점 미만으로 떨어졌습니다. 이는 다양하고 대표적인 코퍼스의 중요성을 강조하며, 저자원 언어에 대한 MT 평가시 단일 소스에 의존하는 위험성을 강조합니다.



### Generating Uncontextualized and Contextualized Questions for  Document-Level Event Argument Extraction (https://arxiv.org/abs/2404.04770)
Comments: Accepted at NAACL 2024

- **What's New**: 이 논문은 문서 수준의 이벤트 인자 추출을 위한 여러 질문 생성 전략을 제시합니다. 이 전략들은 인간의 개입 없이 맥락을 고려하지 않는 질문과 이벤트 및 문서에 기반한 맥락화된 질문을 생성합니다. 특히 이벤트와 인자가 서로 다른 문장에 나타날 때 두 유형의 질문을 결합하는 것이 유익함을 실험적으로 보여줍니다. 이 접근법은 특정 말뭉치에 대한 구성요소가 없으며, 질문 생성 전략은 다양한 말뭉치에 걸쳐 전이됩니다.

- **Technical Details**: 이 연구에서는 T5 모델(T5 model)을 사용하여 약하게 감독된 학습(weakly supervised)을 통해 맥락화된 질문을 생성하였습니다. 질문은 이벤트 트리거(event trigger)와 문서에 기초하여 생성되며, 변환기(transformer)를 사용하여 이벤트 인자를 식별합니다. 이 연구는 맥락화되지 않은 질문(uncontextualized questions)과 맥락화된 질문(contextualized questions)을 혼합하는 여러 전략을 제시하며, 이러한 전략들은 수동적인 노력 없이 적용될 수 있습니다.

- **Performance Highlights**: 실험 결과는 맥락화되지 않은 질문과 맥락화된 질문의 결합이 RAMS 라는 말뭉치에서 유익함을 보여주며, 다른 말뭉치에서도 효과적임을 입증합니다. 이 접근법은 다른 이벤트 트리거 및 역할과 함께 말뭉치에도 쉽게 적용될 수 있으며, WikiEvents와 같은 다른 말뭉치에도 최적화된 질문 응답 전략이 적용되어 그 적용성을 보여줍니다.



### What Happens When Small Is Made Smaller? Exploring the Impact of  Compression on Small Data Pretrained Language Models (https://arxiv.org/abs/2404.04759)
Comments: AfricaNLP workshop at ICLR 2024

- **What's New**: 이 연구는 저자원 언어 모델과 소량 데이터를 기반으로 한 언어 모델, 특히 아프리카 언어에 중점을 둔 다국어 모델인 AfriBERTa를 통해 모델 압축(Compression) 기술의 유효성을 평가합니다. 저자원, 소량 데이터 설정에서의 프루닝(pruning), 지식 증류(knowledge distillation), 양자화(quantization)의 효과를 검증하고, 이러한 압축 기술이 모델의 효율성과 성능에 미치는 영향을 탐구합니다.

- **Technical Details**: 이 논문은 AfriBERTa 모델을 사용하여 다양한 압축 기법을 평가합니다. 프루닝을 통해 모델 크기를 약 60% 줄이면서 성능 손실을 최소화하였고, 지식 증류는 압축률을 22%에서 33% 사이로 달성하였으며 양자화를 통해 모델 크기를 64.08% 줄이고, 추론 시간(inference time)을 52.3% 단축시켰습니다. 이러한 기법들은 고성능을 유지하면서 모델을 더욱 효율적으로 만들어 저전력 장치에서도 배포할 수 있게 합니다.

- **Performance Highlights**: AfriBERTa 모델의 양자화는 특히 F1 점수에서 기준 모델을 능가하며, 일부 언어에서는 극단적인 프루닝 후에도 밀집 모델(dense model)을 초과하는 성능을 보여줍니다. 이는 저자원 언어에서도 효율적이고 효과적인 NLP 작업이 가능함을 시사하며, 프루닝과 양자화는 추론 시간을 줄이면서도 일반화(generalization) 능력을 유지하거나 향상시키는데 기여할 수 있음을 보여줍니다.



### Multilingual Brain Surgeon: Large Language Models Can be Compressed  Leaving No Language Behind (https://arxiv.org/abs/2404.04748)
Comments: 22 pages, 8 figures, 13 tables. Accepted by LREC-COLING 2024

- **What's New**: 이 논문은 다국어 대규모 언어 모델(LLMs)의 압축을 위한 새로운 캘리브레이션 데이터 샘플링 기법인 Multilingual Brain Surgeon (MBS)을 소개합니다. 기존의 영어 중심의 방법들과 달리, MBS는 모델 훈련 데이터셋의 언어 분포에 비례하여 다양한 언어의 캘리브레이션 데이터를 샘플링합니다. 이는 특히 자원이 부족한 언어에 대한 성능 저하를 최소화하면서도 다국어 LLMs의 압축 효율성을 향상시키는 방법입니다.

- **Technical Details**: MBS는 Optimal Brain Surgeon (OBS)를 기반으로 하여, 각 파라미터의 중요도를 결정하기 위해 캘리브레이션 데이터를 사용합니다. 네트워크 프루닝과 모델 양자화(Model Quantization) 기법을 활용하여, 필요 없는 연결을 제거하고 파라미터를 낮은 비트 수준으로 매핑함으로써 모델 크기를 줄입니다. 이 과정에서 MBS는 다국어 모델에 대해 특히 강조되는 언어 간 상호작용의 역학을 관찰하였으며, 그 결과로 언어의 비율과 캘리브레이션 언어와의 유사성이 성능 유지에 큰 영향을 미친다는 결론을 내렸습니다.

- **Performance Highlights**: 실험은 BLOOM 모델을 사용하여 수행되었으며, CC-100 데이터셋에서 캘리브레이션 데이터를 샘플링하여 XL-Sum 데이터셋에서 언어별로 perplexity를 측정하였습니다. 결과적으로 MBS는 기존의 영어만을 사용하는 GPTQ, SparseGPT, Wanda와 같은 압축 방법들보다 우수한 성능을 보였습니다. 특히 자원이 부족한 언어에서의 성능 개선이 두드러졌으며, 이는 MBS가 다국어 모델 압축 문제에 효과적인 해결책이 될 수 있음을 보여줍니다.



### Navigating the Landscape of Hint Generation Research: From the Past to  the Futur (https://arxiv.org/abs/2404.04728)
Comments: Submitted to TACL'24

- **What's New**: 이 논문은 지능형 튜터링 시스템(Intelligent Tutoring Systems, ITS)에 대한 포괄적인 리뷰를 제공하며, 아이들을 돕기 위해 힌트 생성에 관한 연구 간의 간극을 메우고자 합니다. 디지털 교육의 인기가 높아지면서, AI 및 자연어 처리(Natural Language Processing, NLP)의 발전을 교육과 인지 과학 연구와 통합하는 새로운 방안을 모색합니다.

- **Technical Details**: 이 논문은 힌트 생성 작업에 대한 형식적 정의를 제시하고, 효과적인 히트 생성 시스템을 설계하는 로드맵을 제공합니다. 여기에는 교육적 문맥(contextual), 의미론적(semantics), 그리고 스타일적(stylistic) 요소를 포함한 힌트의 세부 구조를 설명합니다. 또한, 학습자의 기존 지식과 연결된 정보를 제공하는 능력이 강조됩니다.

- **Performance Highlights**: 이 문서는 인간 튜터가 사용하는 힌트의 핵심 특성들을 분석하고, 이를 바탕으로 자동화된 힌트 생성 시스템을 개발할 때 고려해야 할 주요 요소들을 정의합니다. 이러한 시스템은 개별 학습자의 필요와 강점을 인식하고, 학습 목표와 연관시킬 수 있는 힌트를 생성함으로써 학습자의 이해도와 독립적 학습 능력을 증진시키는 데 중요한 역할을 할 수 있습니다.



### PoLLMgraph: Unraveling Hallucinations in Large Language Models via State  Transition Dynamics (https://arxiv.org/abs/2404.04722)
Comments: 15 pages

- **What's New**: 새로운 모델 PoLLMgraph는 대규모 언어 모델(LLMs)의 '홀루시네이션(hallucination)' 현상을 감지하고 예측하는 효과적인 방법으로 제안되었습니다. 이 모델은 기존의 블랙박스(black-box) 방식이 아니라 화이트박스(white-box) 방식으로, 모델의 내부 상태 전환 동안 발생하는 동적인 변화를 분석하여 홀루시네이션을 감지합니다.

- **Technical Details**: PoLLMgraph는 LLM의 생성 과정 중 내부 상태의 전환을 추적하여 홀루시네이션을 감지하도록 설계되었습니다. 이는 과거의 토큰(token) 단일 표현에 의존하던 이전 연구들과 달리, 상태 전환의 시간적 정보를 활용하여 모델의 의사결정 과정을 더욱 정확하게 반영합니다. 활용된 확률 모델(probabilistic models)은 LLM의 동작을 이해하고 투명성을 제공하는데 기여합니다.

- **Performance Highlights**: PoLLMgraph는 다양한 벤치마크 데이터셋에서 최신 방법들을 크게 능가하는 성능을 보였습니다. 특히, TruthfulQA 데이터셋에서 AUC-ROC 메트릭을 사용하여 평가했을 때, 기존 방법보다 20% 이상 향상된 결과를 보여주었습니다.



### Order-Based Pre-training Strategies for Procedural Text Understanding (https://arxiv.org/abs/2404.04676)
Comments: 8 pages (Accepted for publication at NAACL 2024 (Main Conference))

- **What's New**: 본 논문에서는 자연어 처리에서 절차적 이해(procedural understanding)를 향상시키기 위한 시퀀스 기반 프리트레이닝(pretraining) 방법을 제안합니다. 특히 레시피와 같이 순서가 있는 지시사항을 다루며, 이 순서를 지도 신호(supervision signal)로 사용합니다. 이 연구는 '순서-지도' 트랜스포머 프리트레이닝 방법들을 비교하는 최초의 작업 중 하나입니다. 프리트레이닝 방법으로는 순열 분류(Permutation Classification), 임베딩 회귀(Embedding Regression), 스킵-클립(Skip-Clip)이 있으며, 이들이 베이스라인과 최신 기법(state-of-the-art, SoTA) 대비 향상된 결과를 보여주었습니다.

- **Technical Details**: 프리트레이닝 방법은 순열 분류, 임베딩 회귀, 스킵-클립을 포함합니다. 예를 들어, 순열 분류에서는 레시피의 단계를 무작위로 섞은 후, 트랜스포머와 멀티-클래스 분류를 사용하여 원래 순서를 복원하도록 합니다. 임베딩 회귀에서는 순열을 임베딩 벡터로 변환하고 회귀 작업을 수행합니다. 이 방법들은 엔티티 트래킹(Entity Tracking) 작업에서 엔티티의 상태 변화를 예측하기 위해 절차의 단계 순서를 이해하는 데 필요합니다.

- **Performance Highlights**: 제안된 방법들은 두 가지 엔티티 트래킹 데이터셋(NPN-Cooking, ProPara)에서 베스트 베이스라인 대비 각각 1.6%, 7-9%의 성능 향상을 보여주었습니다. 이는 순서를 이해하는 것이 절차적 텍스트의 의미를 파악하는 데 중요한 역할을 한다는 것을 입증합니다. 또한, 제안된 방법들은 ProPara 데이터셋에서 평균 정확도(Average Accuracy) 면에서 최신 기법(SoTA)을 능가했습니다.



### Inferring the Phylogeny of Large Language Models and Predicting their  Performances in Benchmarks (https://arxiv.org/abs/2404.04671)
- **What's New**: 이 논문은 LLM(Large Language Models)의 파인튜닝 관계를 탐구하고 성능 특성을 예측하기 위해 계통학 알고리즘을 적용한 PhyloLM이라는 방법을 소개합니다. 게놈 공학 개념(genomics concepts)을 기계 학습에 적용하여 훈련 정보가 투명하지 않은 상황에서도 LLM 개발, 관계 및 능력을 추론할 수 있는 도구를 제공합니다.

- **Technical Details**: PhyloLM 알고리즘은 LLM내의 토큰들(token)과 문맥(context) 사이의 조건부 확률을 분석하여 계통 거리 메트릭(phylogenetic distance metric)을 이용한 계통도(dendrograms)를 구성합니다. 이를 통해 다양한 LLM 가족들을 식별하고, 이들의 상호작용 및 기능적 차이를 파악할 수 있습니다. 또한, 이 방법론은 개발된 계통 알고리즘을 사용하여 토큰(alleles)과 문맥(genes)으로 구성된 '유전체(genomes)'의 완성도를 기반으로 모델 간의 거리를 측정합니다.

- **Performance Highlights**: PhyloLM은 77개의 오픈 소스 및 22개의 폐쇄형 모델을 포함하여 다양한 LLM 가족을 성공적으로 구분했습니다. 이 연구는 계통 거리가 MMLU 및 ARC 같은 벤치마킹 도구에서의 모델 성능을 예측하는 데 유용함을 보여줍니다. 이를 통해 시간 및 비용 효율적으로 LLM의 능력을 추정할 수 있습니다.



### Multilingual Pretraining and Instruction Tuning Improve Cross-Lingual  Knowledge Alignment, But Only Shallowly (https://arxiv.org/abs/2404.04659)
- **What's New**: 본 연구에서는 대형 언어 모델(LLMs)의 언어 간 지식 정렬(cross-lingual knowledge alignment)을 평가하기 위한 체계적인 프레임워크인 CLiKA를 제안하고, 여러 언어로 사전 훈련(multilingual pretraining)과 다국어 지시 조정(multilingual instruction tuning)이 이 정렬에 미치는 영향을 조사했습니다.

- **Technical Details**: CLiKA 프레임워크는 세 가지 평가 수준인 성능(Performance), 일관성(Consistency), 전도성(Conductivity)을 고려합니다. 다양한 언어로의 사전 훈련은 목표 언어의 지식 정렬을 향상시키지만 다른 언어에는 영향을 덜 주는 혼합 사전 훈련이 기본 능력과 지식의 성능 및 일관성을 개선할 수 있음을 발견했습니다. 반면, 지시 조정은 특정 언어에서의 기본 능력은 향상시키나 지식의 전도성은 크게 개선하지 못했습니다.

- **Performance Highlights**: 연구 결과, 현재 다국어 LLMs는 특히 전도성 수준에서 불만족스러운 언어 간 지식 정렬을 보여주었습니다. 또한, 사전 훈련과 지시 조정은 지식의 전도성을 크게 개선하지 못하는 것으로 나타났습니다. 이는 LLMs의 다국어 성능을 향상시키기 위한 훈련 전략이 신중하게 설계되어야 함을 시사합니다.



### HyperTTS: Parameter Efficient Adaptation in Text to Speech using  Hypernetworks (https://arxiv.org/abs/2404.04645)
- **What's New**: HyperTTS는 새로운 스피커에 음성 합성(Text-to-Speech, TTS) 모델을 효율적으로 적용할 수 있게 해주는 혁신적인 접근 방식을 제시합니다. 이 방법은 스피커 임베딩(speaker embeddings)에 조건을 달아 adapter를 동적으로 조정할 수 있게 하며, 이를 통해 하이퍼네트워크(hypernetwork)를 사용하여 학습 가능한 매개변수 공간을 확장합니다.

- **Technical Details**: HyperTTS는 전통적인 TTS 시스템에 적용되는 adapter에 비해 동적인 접근 방식을 사용합니다. 각 스피커에 대해 동적으로 적응하는 adapter(dynamic adapters)를 학습하고, 이를 스피커 표현에 조건을 부여함으로써 adapter의 성능을 향상시킬 수 있습니다. 추가로, 매개변수 샘플링(parameter sampling)을 통해 여러 스피커의 매개변수 공간을 연속적인 분포로부터 효율적으로 생성할 수 있습니다.

- **Performance Highlights**: 이 연구에서는 LibriTTS와 VCTK 데이터셋을 사용하여 다양한 환경 조건에서 HyperTTS의 성능을 검증했습니다. 결과적으로, HyperTTS는 기존의 adapter 방식(AdapterTTS) 및 전체 모델 파인튜닝(TTS-FT) 방식과 비교하여 높은 매개변수 효율성을 보이며 우수한 성능을 달성했습니다. 특히, 파인튜닝과 비교해 단 1%의 매개변수를 사용하면서도 유사한 성능을 보여, 다양한 실용적 응용에 매우 적합한 방법론임을 입증했습니다.



### Context versus Prior Knowledge in Language Models (https://arxiv.org/abs/2404.04633)
- **What's New**: 이 연구는 언어 모델이 특정 문맥(context)에서 주어진 엔티티(entity)에 대한 질문에 어떻게 반응하는지를 측정하는 두 가지 상호 정보(mutual information) 기반 메트릭을 제안합니다. 저자들은 퍼스웨이전 스코어(persuasion score)와 서셉터빌리티 스코어(susceptibility score)를 도입하여 모델이 문맥 또는 사전 지식에 얼마나 의존하는지 측정합니다.

- **Technical Details**: 퍼스웨이전 스코어는 특정 엔티티에 대한 질문에 모델이 문맥에 얼마나 의존하는지를 측정합니다. 서셉터빌리티 스코어는 모델이 얼마나 쉽게 그것의 초기 답변 분포에서 벗어날 수 있는지를 측정합니다. 이 연구는 YAGO 지식 그래프(knowledge graph)에서 추출한 데이터셋을 사용하여 메트릭의 타당성을 실험적으로 검증하고 다양한 엔티티와 문맥에 걸쳐 모델의 행동을 분석합니다.

- **Performance Highlights**: 연구 결과에 따르면, 관련있는 문맥(context)이 관련없는 문맥보다 일관되게 더 설득력이 있다는 것을 발견했습니다. 또한, 자주 등장하는 엔티티는 훈련 데이터에서 더 낮은 서셉터빌리티 스코어를 가지며, 이는 모델이 그 엔티티를 더 잘 알고 있기 때문이라고 추론할 수 있습니다. 이러한 메트릭은 성 편향(gender bias) 또는 친구-적(friend-enemy) 관계 같은 특정 상황에 적용되어 언어 모델의 행동을 더 잘 이해하고 제어하는 데 사용될 수 있습니다.



### On the Limitations of Large Language Models (LLMs): False Attribution (https://arxiv.org/abs/2404.04631)
Comments: 8 pages, 5 figures

- **What's New**: 이 연구에서는 큰 언어 모델(LLM)의 중요한 한계인 잘못된 귀속(false attribution) 문제에 대한 통찰을 제공하고, 새로운 환각 측정 메트릭인 Simple Hallucination Index (SHI)를 도입했습니다. 저자가 글을 작성했다고 잘못 표시하는 것을 '잘못된 귀속'이라고 하며, 이는 법적 문제로 간주될 수 있습니다. 이 연구의 목표는 LLM이 책의 짧은 텍스트에 대해 저자 귀속을 수행할 때 잘못된 귀속 문제를 시연하고 이를 평가하기 위한 새로운 환각 메트릭을 소개하는 것입니다.

- **Technical Details**: Zero-shot 설정에서 세 개의 최신 SotA(State of the Art) 대규모 언어 모델을 평가했습니다: LLaMA-2-13B, Mixtral 8x7B, 그리고 Gemma-7B. 각 모델은 Project Gutenberg에서 가장 인기 있는 10권의 책을 400단어 단위로 나누어 저자를 예측하도록 요청 받았습니다. 이 연구는 Nvidia DGX-1 노드에서 수행되었으며 모든 실험은 HuggingFace 플랫폼에서 이루어졌습니다.

- **Performance Highlights**: Mixtral 8x7B가 평균적으로 가장 높은 예측 정확도와 가장 낮은 SHI를 달성했으며, Pearson 상관계수(r)에 따른 정확도와 SHI의 강력한 음의 상관관계(-0.9996)를 보였습니다. 그러나 Mixtral 8x7B는 일부 책에서 높은 환각 증상을 보였고, SHI가 0.87에 이르렀습니다. 이러한 결과는 새로운 환각 메트릭이 해당 태스크에 일반화할 수 있는 가능성을 시사합니다.



### Towards Analyzing and Understanding the Limitations of DPO: A  Theoretical Perspectiv (https://arxiv.org/abs/2404.04626)
Comments: Draft version

- **What's New**: 이 연구는 직접 선호 최적화(Direct Preference Optimization, DPO)의 이론적 이해를 한 단계 더 발전시켰다. DPO는 대규모 언어 모델(Large Language Models, LLMs)을 인간의 선호와 일치시키는 데 유용하지만, 성능의 한계와 SFT(지도된 미세조정, Supervised Fine-Tuning)의 효과에 대한 민감성 때문에 비판을 받아왔다. 이 논문은 DPO의 최적화 과정을 분석하기 위해 필드 이론(field theory)을 적용한 분석 프레임워크를 제공한다.

- **Technical Details**: 이 연구에서는 DPO 손실 함수의 경사 벡터 필드(gradient vector field)를 분석하여 LLM이 인간이 선호하지 않는 데이터 생성 확률을 감소시키는 속도가 선호하는 데이터 생성 확률을 증가시키는 속도보다 빠르다는 것을 발견했다. 이러한 분석을 통해, DPO가 LLM의 학습 능력에 방해가 되는 경향과 SFT의 효과에 대한 민감도를 이론적으로 설명할 수 있다.

- **Performance Highlights**: 연구 결과는 DPO가 인간의 선호에 부합하는 반응을 생성하는 데 있어서 어떻게 제한적일 수 있는지를 증명한다. DPO 손실 함수는 인간이 비선호하는 반응을 피하는 데 집중하면서, 선호하는 반응을 생성하는 확률을 느리게 증가시킨다는 이론적 통찰을 제공한다.



### A Morphology-Based Investigation of Positional Encodings (https://arxiv.org/abs/2404.04530)
Comments: Work in Progress

- **What's New**: 이 논문은 어휘 형태학적 복잡성이 다른 23개 언어에 대한 위치 인코딩(POS Encodings)의 중요성이 어떻게 변하는지에 대해 처음으로 조사합니다. 연구는 구문적 작업(품사 태깅, 개체명 인식, 의존성 파싱)과 의미적 작업(자연어 추론, 의미 변경)을 포함한 다양한 하위 작업에서 언어별 BERT 모델을 이용합니다.

- **Technical Details**: 주된 실험은 미세 조정 도중 위치 인코딩의 효과를 무효화하고 이것이 다양한 작업과 언어에 미치는 영향을 조사하는 것입니다. Transformer 기반 언어 모델은 효과적으로 문장 내 단어의 순서를 포착하고 인코딩하기 위해 위치 인코딩을 사용합니다. 이 연구는 위치 인코딩의 중요성이 언어의 형태학적 복잡성에 따라 감소함을 처음으로 보여줍니다.

- **Performance Highlights**: 언어의 형태학적 복잡성이 증가함에 따라 위치 인코딩의 영향이 줄어듦을 발견했습니다. 23개의 다른 언어를 아우르는 실험 결과에서 모든 언어가 그들의 형태학적 유형학(Morphological Typology)에 따라 클러스터링 되었으며, 분석적 언어(analytic languages)와 종합적 언어(synthetic languages)가 각각의 극단에 위치하는 것을 관찰했습니다.



### IITK at SemEval-2024 Task 10: Who is the speaker? Improving Emotion  Recognition and Flip Reasoning in Conversations via Speaker Embeddings (https://arxiv.org/abs/2404.04525)
Comments: Accepted at SemEval 2024, NAACL 2024; 10 Pages

- **What's New**: 이 논문은 SemEval-2024 Task 10: 대화에서의 감정 발견과 그 변화의 이유 연구에 대한 접근 방식을 제시합니다. 감정 인식 작업을 위해 마스크된 메모리 네트워크와 발화자 참여를 활용하였고, 감정 변화 이유를 분석하기 위한 트랜스포머(Transformer) 기반 발화자 중심 모델을 제안합니다.

- **Technical Details**: 경쟁 모델 재구성을 위해 트랜스포머 및 매스드 메모리 네트워크와 같은 기술적 접근 방식을 사용하였으며, 'Probable Trigger Zone' (PTZ)이라는 개념을 도입하여 감정 변화를 유발하는 발화이 일어날 가능성이 높은 대화 영역을 식별합니다. 또한 텍스트 임베딩(Text Embeddings)을 위해 사전 훈련된 모델을 활용하고 있습니다.

- **Performance Highlights**: sub-task 3에서는 기존 모델 대비 5.9 F1 스코어(F1 score) 향상을 보였으며, 이러한 결과는 다양한 설계 선택의 중요성을 강조하는 소거 연구(Ablation Study) 결과에 의해 뒷받침됩니다. ERC를 위한 F1 점수는 45, EFR를 위한 F1 점수는 각각 56과 60을 기록하였습니다.



### Q-PEFT: Query-dependent Parameter Efficient Fine-tuning for Text  Reranking with Large Language Models (https://arxiv.org/abs/2404.04522)
- **What's New**: 이 논문에서는 문서 재정렬(reranking)을 위한 새로운 쿼리 의존적 매개변수 효율적 미세조정(Query-Dependent Parameter Efficient Fine-Tuning, Q-PEFT) 방법을 제안합니다. 기존의 매개변수 효율적 미세조정(PEFT) 방식과 달리, Q-PEFT는 쿼리 정보를 활용하여 문서에 특화된 합성 쿼리를 생성하는 데 초점을 맞추고 있습니다. 이 접근 방식은 매개변수의 수를 최소화하면서 기존 대규모 언어 모델(LLM)의 재정렬 기능을 향상시킵니다.

- **Technical Details**: 연구진은 Q-PEFT를 두 가지 방식으로 구현해냈습니다: Q-PEFT-R과 Q-PEFT-A. Q-PEFT-R은 문서에서 주요 k개 단어를 활용하여 쿼리 종속적 내용을 생성하고, Q-PEFT-A는 멀티 헤드 주의력(multi-head attention) 계층을 사용하여 문서와 쿼리 간의 상관 관계를 무게를 재조정합니다. 이는 전체 LLM을 미세조정하지 않고도 특정 IR(Infromation Retrieval) 작업에 맞게 LLM의 성능을 최적화하는 것을 목표로 합니다.

- **Performance Highlights**: 제안된 Q-PEFT 방법은 네 개의 공개 데이터셋에서 폭넓은 실험을 거쳤으며, 기존의 하드 프롬프팅(hard prompting) 방법보다 뛰어난 성능을 보였습니다. 이는 Q-PEFT가 문서-쿼리 관련성 점수에 긍정적인 영향을 미침으로써, 쿼리에 따라 더욱 특화된 결과를 생성할 수 있음을 시사합니다. 코드와 체크포인트는 논문이 수락되는 대로 공개될 예정입니다.



### IITK at SemEval-2024 Task 4: Hierarchical Embeddings for Detection of  Persuasion Techniques in Memes (https://arxiv.org/abs/2404.04520)
Comments: Accepted at SemEval 2024, NAACL 2024; 9 pages

- **What's New**: 이 논문은 인터넷에서 널리 사용되는 컨텐츠 유형인 'Meme'(미미)가 어떻게 심리적, 수사학적 기술을 사용하여 사람들에게 영향을 주는지 분석합니다. 특히 SemEval 2024 Task 4 'Multilingual Detection of Persuasion Technique in Memes'는 미미 내의 설득 기술을 탐지하는 것을 목표로 합니다. 이 논문은 텍스트와 시각적 컨텐츠를 모두 활용하여 설득 기술을 분류하는 다양한 접근 방식을 제안하며, 여러 서브태스크(sub-tasks)에 걸쳐 향상된 분류 정확도를 보입니다. 텍스트만을 사용하거나, 텍스트와 이미지를 모두 사용하여 계층적 다중 라벨 분류(hierarchical multi-label classification) 도구가 개발되었습니다.

- **Technical Details**: 이 연구에서는 Class Definition Prediction (CDP) 및 하이퍼볼릭 임베딩(hyperbolic embeddings)을 기반으로 한 앙상블 모델을 사용합니다. 'HypEmo'라는 프레임워크를 통해 계층적 라벨 임베딩(hierarchical label embeddings)을 통합하고, 감정 예측에 대한 멀티태스크 학습(multi-task learning) 프레임워크를 사용하여 분류 정확도와 포괄성을 높였습니다. 또한, CLIP 임베딩을 도입하여 미미의 텍스트와 시각적 컴포넌트에서 필수 특징들을 포착하는 방법을 개선하였습니다. 이러한 접근법은 미미 내용의 더 포괄적인 분석을 가능하게 합니다.

- **Performance Highlights**: 각 서브태스크에 대한 성능은 계층적 F1 점수(hierarchical F1-score)로, 첫 번째 서브태스크에서 0.60, 두 번째에서 0.67, 세 번째에서 0.48을 달성했습니다. 이러한 결과는 텍스트와 이미지 모두를 분석하는 접근 방식이 미미의 설득 기술을 감지하는 데 특히 더 효과적임을 보여줍니다.



### Joint Visual and Text Prompting for Improved Object-Centric Perception  with Multimodal Large Language Models (https://arxiv.org/abs/2404.04514)
- **What's New**: 본 연구에서는 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)이 시각적 질문 응답(Visual Question Answering, VQA) 태스크에서 인간 수준의 인식을 달성하는 데 어려움을 겪고 있음을 밝혀냈습니다. 특히, 객체 지향적 인식(Object-Oriented Perception)과 관련하여 미세한 이해가 요구되는 문제들에서 더욱 그러합니다. 이를 개선하기 위해 'Joint Visual and Text Prompting (VTPrompt)'라는 새로운 접근 방법을 제안하였습니다. 이 방법은 텍스트 프롬프트(Text Prompt)와 함께 정교한 시각적 정보를 사용하여 MLLMs의 VQA 능력을 향상시키는 것을 목표로 합니다.

- **Technical Details**: VTPrompt는 먼저 텍스트 질문에서 핵심 개념을 추출하고, 대상 모델에 SPHINX 또는 SAM과 같은 탐지 모델을 사용하여 관련 객체를 시각적 프롬프트(Visual Prompt)로 강조 표시합니다. 처리된 이미지는 텍스트 프롬프트와 함께 MLLMs에 제공되어 더 정확한 답변을 생성하게 됩니다. 이 연구는 GPT-4V와 Gemini Pro를 사용하여 MME, MMB, POPE 벤치마크에서 실험을 수행하였고, 특히 GPT-4V는 MME에서 183.5점이 향상되었으며, MMB에서는 GPT-4V가 8.17%, Gemini Pro가 15.69% 성능이 향상되었습니다.

- **Performance Highlights**: VTPrompt는 MLLMs의 객체 지향적 인식 능력을 뚜렷하게 향상시켜 MME에서 GPT-4V가 183.5점 상승하였고, Gemini Pro는 MMB에서 15.69%의 성능 향상을 보여주었습니다. 이는 객체의 정체성, 위치, 특성에 대한 미세한 이해를 요구하는 VQA 태스크에서 MLLMs의 한계를 극복하고, 인간 수준에 가까운 인식 능력을 제공하기 위한 중요한 발전입니다.



### IITK at SemEval-2024 Task 1: Contrastive Learning and Autoencoders for  Semantic Textual Relatedness in Multilingual Texts (https://arxiv.org/abs/2404.04513)
Comments: Accepted at SemEval 2024, NAACL 2024; 6 pages

- **What's New**: 이 논문에서는 SemEval-2024 Task 1: Semantic Textual Relatedness(STR)에 참여하여 개발한 시스템에 대하여 설명하고 있습니다. 주로 14개 언어, 특히 자원이 부족한 아시아와 아프리카 언어의 문장 쌍 사이의 관련성을 자동으로 감지하는 데 중점을 두고 있습니다. 연구팀은 지도학습(Track A)과 비지도학습(Track B) 두 가지 하위 작업에 참여하였으며, 주로 BERT 기반의 Contrastive Learning과 유사성 측정 접근 방식을 사용하였습니다.

- **Technical Details**: 이 연구에서는 Contrastive Learning과 유사성 측정 메트릭(Metric)을 사용하여 Semantic Textual Relatedness를 평가하는 방법을 개발하였습니다. 특히, Distill-RoBERTa와 BERT-uncased 모델을 사용하여, 지도학습 방식에서는 양방향에서 관련성 스코어를 계산하는 Composite Lexical Similarity를 기반으로 하고 있으며, 비지도학습에서는 Denoising Autoencoder를 활용하였습니다. 또한, Siamese architecture와 low-resource 언어를 위한 어휘 확장을 통한 재학습 같은 여러 전략들을 실험하였습니다.

- **Performance Highlights**: 이 시스템은 다양한 저자원 언어에서 상대적으로 높은 성능을 보여주었습니다. 예비 실험 결과, BERTbase와 RoBERTa-base 모델은 CompLex 데이터셋에서 Spearman 상관 계수가 각각 평균 0.82와 0.83을 달성하였습니다. 이는 기존의 어휘 중첩 기준보다 약간의 개선을 보여줍니다. 또한, IndicSBERT는 인도의 저자원 언어를 위해 세부조정된 후 STS 벤치마크에서 복잡성 점수에서의 유의미한 향상을 보였습니다.



### IITK at SemEval-2024 Task 2: Exploring the Capabilities of LLMs for Safe  Biomedical Natural Language Inference for Clinical Trials (https://arxiv.org/abs/2404.04510)
Comments: Accepted at SemEval 2024, NAACL 2024; 8 Pages

- **What's New**: 이 연구는 SemEval 2024 Task 2: Safe Biomedical Natural Language Inference for Clinical Trials라는 맥락에서 유방암 임상시험 보고서(Clinical Trial Reports, CTRs)에서 자연어추론(Natural Language Inference, NLI) 수행 시 대규모 언어 모델(Large Language Models, LLMs)의 견고함과 일관성, 충실한 추론 능력을 조사합니다. 특히 GPT-3.5와 Gemini Pro 같은 사전 훈련된 언어 모델들(Pre-trained Language Models, PLMs)을 Retrieval Augmented Generation (RAG) 프레임워크를 사용해 Zero-shot 설정에서 비교 분석 했습니다.

- **Technical Details**: 연구진은 Gemini Pro와 GPT-3.5, 그리고 BioLinkBERT, SciBERT, ClinicalBERT 등의 다양한 PLMs를 활용해 유방암 CTRs를 기반으로 한 NLI 작업을 수행했으며, 이 과정에서 Tree of Thoughts (ToT) 추론, Chain-of-Thought (CoT) 프롬프팅 기법이 통합되었습니다. 모델들은 다양한 추론 경로를 통합하고, 명령 템플릿을 활용하여 Zero-shot 평가를 진행했습니다.

- **Performance Highlights**: Gemini Pro가 실험에 사용된 모든 모델 중 최고의 성능을 달성했습니다. F1 점수는 0.69, 일관성은 0.71, 충실도는 0.90으로 나타났습니다. 반면, GPT-3.5는 특히 수치 추론이 필요한 경우에 성능이 떨어지는 것으로 나타났습니다.



### KazQAD: Kazakh Open-Domain Question Answering Datas (https://arxiv.org/abs/2404.04487)
Comments: To appear in Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)

- **What's New**: KazQAD는 열린 도메인 질문 응답(ODQA)을 위한 카자흐어 데이터셋으로, 읽기 이해(reading comprehension)와 정보 검색(information retrieval) 실험에도 적합하다. 이 데이터셋에는 약 6,000개의 고유 질문과 12,000건의 문단 레벨 중요 판단이 포함되어 있다. 연구 팀은 기계 번역, 위키백과 검색 및 내부 수동 주석을 결합하여 주석 효율성과 데이터 품질을 보장하였다.

- **Technical Details**: KazQAD는 카자흐어 위키백과의 800,000개 이상의 문단을 포함하는 본문 코퍼스와 함께 제공된다. 또한 기계 번역을 통해 카자흐어로 번역된 NQ(Natural Questions) 데이터셋의 약 61,000개 질문-문단-응답 트리플을 추가로 제공한다. 베이스라인 검색기(retriever)와 읽기 도구(reader)를 개발하여, 검색(NDCG@10 = 0.389 MRR = 0.382), 읽기 이해(EM = 38.5 F1 = 54.2), 전체 ODQA(EM = 17.8 F1 = 28.7)에서 합리적인 성능을 보였다.

- **Performance Highlights**: 비록 KazQAD의 성능은 영어 QA 데이터셋의 최신 기술보다 낮지만, 향후 개선의 여지가 충분하다고 평가된다. 또한, 현재 OpenAI의 ChatGPTv3.5는 닫힌 책 설정에서 KazQAD 테스트 질문에 대해 수용 가능한 품질로 답변할 수 없음을 보여준다.



### Towards Realistic Few-Shot Relation Extraction: A New Meta Dataset and  Evaluation (https://arxiv.org/abs/2404.04445)
- **What's New**: 새로운 메타 데이터셋이 소개되었습니다. 이 데이터셋은 기존의 관계 추출 데이터셋인 NYT29와 WIKIDATA에서 파생된 두 데이터셋과 TACRED 데이터셋의 Few-shot(퓨샷) 형태를 포함하고 있습니다. 중요하게도, 이들 Few-shot 데이터셋은 현실적인 가정 하에 생성되었으며, 테스트 관계는 모델이 이전에 본 적이 없는 관계, 제한된 훈련 데이터, 그리고 관심 있는 관계에 해당하지 않는 후보 관계 언급의 우세 등의 특징을 가지고 있습니다.

- **Technical Details**: 이 연구는 최근의 여섯 가지 Few-shot 관계 추출 방법을 종합적으로 평가합니다. 각 방법의 성능은 혼합적으로 나타나며, 명확한 우세자는 없습니다. 또한, 이 작업에 대한 전체적인 성능이 낮아 향후 연구에 대한 상당한 필요성을 시사합니다.

- **Performance Highlights**: 데이터셋의 전체적인 성능이 낮으며, 이는 Few-shot 관계 추출 작업이 아직 많은 개선이 필요함을 의미합니다. 하지만 아직 명확한 성공을 보여주는 방법이 없기 때문에, 이 분야는 예측 불가능하고 도전적인 연구 분야로 남아 있습니다.



### Deciphering Political Entity Sentiment in News with Large Language  Models: Zero-Shot and Few-Shot Strategies (https://arxiv.org/abs/2404.04361)
Comments: Accepted in PoliticalNLP workshop co-located with LREC-COLING 2024

- **What's New**: 이 연구는 정치 뉴스 기사에서 특정 개체에 대한 감정을 예측하는 대형 언어 모델 (Large Language Models, LLMs)의 효과를 조사합니다. 연구는 zero-shot과 few-shot 접근 방식과 함께 사고의 연쇄 (Chain-of-Thought, COT) 방법을 사용하여 감정 예측의 정확성을 향상시키는지 평가하고 있습니다.

- **Technical Details**: 이 연구는 대형 언어 모델 (LLMs)을 사용하여 정치 뉴스 기사 속 정치적 개체에 대한 감성을 예측합니다. 사고의 연쇄 (COT) 접근법을 이용하여 논리적 추론 단계를 포함시키고 few-shot in-context 학습에서 이를 적용합니다. 주목할 만한 점은 self-consistency 메커니즘을 사용하여 감성 예측의 일관성을 증진시키는 것입니다. 이 모델들은 기존의 BERT 모델들보다 우수한 성능을 보여주었습니다.

- **Performance Highlights**: 대형 언어 모델 (LLMs)은 감정 라벨링 데이터셋에서 BERT모델보다 우수한 성능을 보였습니다. 특히, in-context 학습은 모델 성능을 크게 향상시키며, self-consistency 메커니즘은 예측의 일관성을 향상시킵니다. 하지만, COT prompting 방법의 효과에는 일관성이 없는 경우도 관찰되었습니다.



### Assisting humans in complex comparisons: automated information  comparison at sca (https://arxiv.org/abs/2404.04351)
Comments: 11 pages, 7 figures, 5 tables

- **What's New**: ASC$^2$End 시스템은 지식 분야의 대용량 데이터 정보 비교에 대한 새로운 접근 방식을 도입하여, 정보 비교를 자동화하고 확장하는 데 목적이 있습니다. 이 시스템은 추상적 요약(Abstractive Summarization), 검색 강화 생성(Retrieval Augmented Generation, RAG), 제로-샷 전략(Zero-shot strategies)을 통해 대규모 맥락에서도 정확한 정보 보존과 추론을 가능하게 합니다.

- **Technical Details**: ASC$^2$End는 Semantic Text Similarity (STS)를 활용하여 텍스트 간의 의미적 유사성을 비교하며, 추상적 요약과 RAG를 통합해 토큰 제한을 극복하고 중요 정보를 유지합니다. 시스템은 문서 요약(Document Summarization, DS), 기준 임베딩(Criteria Embedding, CE), RAG, 비교 평가(Comparison Assessment, CA)의 네 가지 모듈로 구성되어 있습니다. 제로-샷 프롬프트를 사용하여 모델 튜닝 없이도 유연한 적용이 가능하도록 설계되었습니다.

- **Performance Highlights**: 시스템의 성능 평가는 ROUGE 점수를 사용한 추상적 요약 평가와 설문을 통한 비교 품질 평가로 이루어졌습니다. ASC$^2$End는 기대되는 성능을 보여주었으며, 특히 금융 서비스 분야에서 복잡한 사용자 정의 지속 가능 금융 기준을 충족하는 금융 거래를 식별 및 평가하는 데 적합합니다. 이 시스템은 대규모 복잡 데이터 세트 간의 효율적인 정보 비교를 가능하게 하여 더욱 정보에 기반한 의사결정을 촉진합니다.



### Scope Ambiguities in Large Language Models (https://arxiv.org/abs/2404.04332)
Comments: To be published in Transactions of the Association for Computational Linguistics

- **What's New**: 이 논문은 대규모 언어 모델(GPT-2, GPT-3/3.5, Llama 2, GPT-4)이 의미적 범위가 중첩되는 문장을 어떻게 처리하는지 조사합니다. 이러한 문장은 범위 모호성(scope ambiguities)을 포함하며, 이는 의미 구조와 세계 지식 간의 상호 작용에 통찰력을 제공합니다. 연구는 이러한 모호성을 처리하는 대규모 언어 모델의 능력을 측정하고 인간의 판단과 비교합니다.

- **Technical Details**: 연구자들은 거의 1,000개의 유니크한 범위 모호 문장 데이터셋을 소개하고, 이를 이용하여 모델의 성능을 평가합니다. 문장은 다양한 의미 연산자들 간의 상호 작용을 포함하며, 인간의 판단으로 주석이 달려 있습니다. 또한, 테스트 결과 일부 모델들이 90% 이상의 높은 정확도로 인간이 선호하는 해석을 성공적으로 식별할 수 있는 것으로 나타났습니다.

- **Performance Highlights**: 이 연구는 GPT-4와 같은 최신 모델들이 높은 정확도로 범위 모호 문장의 인간 선호 독해를 판별할 수 있음을 보여줍니다. 특히, 대규모 언어 모델들이 의미 구조(semantics)와 세계 지식(world knowledge)의 통합에서 어떠한 역량을 보이는지에 대한 중요한 통찰을 제공합니다. 이는 언어 기반 작업에서 일반적으로 높은 성능을 보임에도 불구하고, 이러한 모델이 언어 구조의 추상적인 측면들을 얼마나 잘 포착하는지에 대한 질문에 답하는 데 도움을 줍니다.



### CBR-RAG: Case-Based Reasoning for Retrieval Augmented Generation in LLMs  for Legal Question Answering (https://arxiv.org/abs/2404.04302)
Comments: Submitted to ICCBR'24

- **What's New**: 이 논문은 사례 기반 추론(Case-Based Reasoning, CBR)을 활용하여 대규모 언어 모델(Large Language Model, LLM)의 검색-증강 생성(Retrieval-Augmented Generation, RAG) 프로세스를 강화하는 새로운 접근 방식인 CBR-RAG 모델을 소개합니다. 특히 법률 분야의 질문-응답(Question-Answering, QA) 작업에서, 이 모델은 관련 사례를 검색하여 입력 쿼리를 풍부하게 하고, 결과적으로 보다 정확하고 신뢰할 수 있는 응답 생성을 가능하게 합니다.

- **Technical Details**: CBR-RAG 모델은 CBR의 초기 검색 단계, 색인어 휘장(indexing vocabulary), 유사성 지식 컨테이너(similarity knowledge containers)를 사용하여 LLM 쿼리를 향상시킙니다. 이는 일반 및 도메인 특정 임베딩(domain-specific embeddings), 그리고 내부, 상호 및 혼합 유사도(inter, intra and hybrid similarity) 비교 방법을 포함하여 다양한 검색 방법을 사용하여 법률 영역에서 QA를 수행합니다.

- **Performance Highlights**: CBR-RAG 시스템은 기존 방식보다 높은 정확도와 사실 확인을 통한 응답의 질적 개선을 달성했습니다. 법률 문서에서 추출된 사례를 활용하여 입력된 질문과 관련성이 높은 응답을 생성할 수 있었으며, 이는 법률적 질문에 대한 보다 정교하고 신뢰할 수 있는 답변으로 이어졌습니다.



### Transducers with Pronunciation-aware Embeddings for Automatic Speech  Recognition (https://arxiv.org/abs/2404.04295)
Comments: accepted at the ICASSP 2024 conference

- **What's New**: 이 논문에서는 발음 인식이 가능한 임베딩을 가진 Transducer 모델인 PET (Pronunciation-aware Embedding Transducers)를 제안합니다. PET 모델은 기존의 Transducer와 다르게 동일하거나 유사한 발음을 가진 텍스트 토큰들에 대해 공유된 구성 요소를 포함하여 디코더 임베딩을 통합함으로써, 연쇄적인 오류를 현저히 줄이고 전반적인 음성 인식 정확도를 향상시킵니다.

- **Technical Details**: PET 모델은 다양한 임베딩 테이블을 사용하여 발음 심볼들에 상응하는 임베딩을 구성합니다. 예를 들어, 동일한 발음 'ta'를 가진 '他'와 '她'는 임베딩 생성 시 'P' (발음)과 'W' (단어 ID) 특성을 사용하여 최종 임베딩을 계산합니다. 이러한 접근 방식은 텍스트 토큰들 간의 발음 정보를 활용하여 보다 효율적인 학습과 더 정확한 음성 인식 결과를 도출합니다.

- **Performance Highlights**: PET는 Mandarin Chinese와 Korean 데이터셋에서 기존 Transducer 모델들보다 뛰어난 성능을 보였으며, 특히 오류 연쇄 반응을 감소시키는 데 크게 기여했습니다. 이 모델은 NeMo (NVIDIA's Neural Modules) 툴킷을 통해 오픈 소스로 제공될 예정입니다.



### Reason from Fallacy: Enhancing Large Language Models' Logical Reasoning  through Logical Fallacy Understanding (https://arxiv.org/abs/2404.04293)
- **What's New**: 이 논문에서는 LLM(Logical Language Models 대형 언어 모델)의 논리적 오류 이해(LFU, Logical Fallacy Understanding) 능력을 평가하고 향상시키기 위해 새로운 데이터셋 LFUD(Logical Fallacy Understanding Dataset)를 제안하고 구축합니다. LLM의 논리적 오류 이해를 정확히 평가하고 개선하기 위한 다섯 가지 구체적인 작업을 제시합니다.

- **Technical Details**: LFUD는 GPT-4 및 인간의 작은 노력을 통해 생성되었습니다. 세 가지 인지 차원 WHAT, WHY, HOW에 해당하는 다설 가지 작업으로 구성됩니다. WHAT-Identificaiton(1번 작업)과 Classification(2번 작업)는 주어진 문장에 논리적 오류가 있는지 및 그 유형을 식별합니다. WHY-Deduction(3번 작업)과 Backward Deduction(4번 작업)은 문장에서 논리적 오류를 일으키는 이유를 포착합니다. HOW-Modification(5번 작업)은 문장의 논리적 오류를 수정합니다.

- **Performance Highlights**: 이 연구에 따르면 LLM을 LFUD로 파인 튜닝(fine-tuning) 함으로써 그들의 논리적 추론 능력이 크게 향상된다는 것을 발견하였습니다. LFUD는 4,020개의 인스턴스와 12가지의 논리적 오류 유형을 포함하고 있으며, LLM의 LFU 능력뿐만 아니라 논리적 추론 능력을 개선하는 데 기여할 수 있습니다.



### Conversational Disease Diagnosis via External Planner-Controlled Large  Language Models (https://arxiv.org/abs/2404.04292)
Comments: Work in Progress

- **What's New**: 이 연구는 대화형 진단을 구현하기 위해 대규모 언어 모델(LLMs)과 외부 플래너를 결합한 새로운 접근 방식을 도입합니다. 의사의 두 단계 의사결정 과정인 질병 선별과 감별 진단을 모방하여 정보 수집을 위한 정책 모듈과 자연어 이해 및 생성을 위한 모듈을 포함한 의료 대화 시스템을 개발했습니다.

- **Technical Details**: 이 시스템은 강화 학습(Reinforcement Learning)과 활성 학습(Active Learning)을 통해 훈련된 두 개의 플래너를 사용합니다. 첫 번째 플래너는 환자의 증상을 수집하여 잠재적 질병을 식별하고, 두 번째 플래너는 구체적인 질문을 통해 이러한 질병을 확인하거나 배제합니다. 대규모 언어 모델(LLM)은 자연어 이해와 생성 부문에서 탁월한 능력을 발휘하여 시스템의 성능을 한층 강화합니다.

- **Performance Highlights**: MIMIC-IV 데이터셋을 사용한 평가에서 이 시스템은 기존 모델을 능가하는 성능을 보여주며, 특히 심부전 사례에서 두 번째 단계의 진단 효율성이 F1 스코어 90% 이상을 기록하였습니다. 이는 모델의 오진 위험을 크게 줄이면서 질병을 효과적으로 진단할 수 있음을 시사합니다.



### CONFLARE: CONFormal LArge language model REtrieva (https://arxiv.org/abs/2404.04287)
Comments: Github code: this https URL

- **What's New**: 이 보고서에서는 RAG (Retrieval-augmented generation) 프레임워크의 검색 과정에서의 불확실성을 계량화하기 위해 컨포멀 예측(conformal prediction)을 적용하는 네 단계 프레임워크를 소개합니다. 특히, 의료 분야와 같이 정확한 정보가 중요한 분야에서 RAG의 신뢰성을 향상시킬 수 있는 방법을 제안하고 있습니다.

- **Technical Details**: 컨포멀 예측을 통해 사용자가 명시한 오류율(α)에 따라 신뢰성 있는 검색 결과를 보장하며, 이는 지식 기반의 문서 청크를 통해 LLM(large language models)에 정확한 맥락을 제공합니다. 제시된 파이썬 패키지를 사용하여 이 프로세스를 자동화할 수 있으며, 의학적 질문-응답 시나리오에 적용할 수 있는 예시 노트북도 포함되어 있습니다.

- **Performance Highlights**: 이 새로운 접근법은 RAG 프레임워크의 검색 단계에 컨포멀 예측을 적용함으로써, 오류율 α에서 1-α의 신뢰 수준으로 진실된 답변이 맥락에 포함되도록 보장합니다. 이로 인해 LLM의 응답 생성 시 발생할 수 있는 오류를 줄이고, 결과적으로 더 신뢰할 수 있는 출력을 생성할 가능성이 높아집니다.



### Language Model Evolution: An Iterated Learning Perspectiv (https://arxiv.org/abs/2404.04286)
- **What's New**: 이 논문은 대규모 언어 모델(LLM, Large Language Models)의 진화 과정에 관한 새로운 관점을 제공합니다. 특히, 인간의 문화 진화를 연구하는 베이지안 이터레이티드 러닝(IL, Iterated Learning) 프레임워크를 사용하여 LLM의 행동을 설명하고 있습니다. 이러한 접근 방식은 LLM이 자기 개선(self-improvement) 방법을 통해 어떻게 진화할 수 있는지 이해하는데 도움을 줍니다.

- **Technical Details**: 베이지안 이터레이티드 러닝(IL)은 에이전트들 사이의 지식과 신념의 진화를 연구하기 위한 프레임워크입니다. 이 논문은 LLM이 이러한 프레임워크 내에서 어떻게 편향을 점진적으로 증폭시키는지를 이론적으로 설명하고, 여러 LLM을 대상으로 실험을 통해 이를 검증합니다. 결과적으로, LLM의 진화를 유도하는 다양한 전략을 제안하며, 이는 데이터 필터링, 편향 완화 또는 증폭과 같은 작업에 효율적인 알고리즘 설계에 기여할 수 있습니다.

- **Performance Highlights**: 이 연구의 주요 기여는 베이지안 분석을 통해 LLM 간의 상호 작용 학습 과정을 처음으로 설명하고, 이 이론을 LLM 에이전트의 진화에 적용한 것입니다. 실험을 통해 이러한 이론적 모델이 실제로 LLM의 행동을 예측하고 유도하는 데 유효함을 입증했습니다. 더욱이, 이는 LLM이 생성하는 데이터의 향후 추세와 AI의 장기적 진화 경로에 대한 이해를 높이는데 중요한 역할을 합니다.



### MIMIR: A Streamlined Platform for Personalized Agent Tuning in Domain  Expertis (https://arxiv.org/abs/2404.04285)
- **What's New**: 이 연구에서는 개인화된 에이전트 튜닝(personalized agent tuning)을 가능하게 하는 새로운 플랫폼 	extsc{Mimir}를 소개합니다. Mimir는 개인 지식과 공개적으로 사용 가능하며 법적으로 준수하는 데이터셋을 활용하여 LLMs의 성능을 개선할 수 있는 맞춤형 파이프라인을 제공합니다. 또한, Mimir는 일반적인 지시 튜닝 데이터셋(general instruction-tuning datasets) 생성을 지원하여 에이전트의 특정 능력과 일반 역량을 모두 갖출 수 있도록 합니다.

- **Technical Details**: Mimir는 private knowledge와 법적으로 허용된 공공 데이터셋을 통합하고, 다중 차례에 걸친 에이전트 튜닝 데이터셋을 생성하는 기능을 지원합니다. 사용자는 이 플랫폼을 통해 LoRA 기술을 활용하여 맞춤형 파인 튜닝 스크립트를 구현할 수 있으며, 이는 LLM의 성능과 효율성을 최적화합니다. 특히 의료, 법률, 금융 등의 특수 분야에서 중요한 개인 데이터를 안전하게 다루면서 동시에 도메인 특화 지식을 통합할 수 있습니다.

- **Performance Highlights**: Mimir를 사용하여 생성된 데이터는 기존 데이터, self-instruct 방식, Baize 방식과 비교하여 각각 87%, 75%, 77%의 승률 또는 동등한 성능을 보여주었습니다. 이는 Mimir가 제공하는 맞춤형 데이터 생성과 파인 튜닝이 LLM의 역량을 상당히 향상시킬 수 있음을 의미합니다.



### Assessing ML Classification Algorithms and NLP Techniques for Depression  Detection: An Experimental Case Study (https://arxiv.org/abs/2404.04284)
- **What's New**: 이 연구는 우울증 진단을 위한 기계학습(ML)과 자연어 처리(NLP) 도구와 기법의 효과에 대해 다루고 있으며, 특히 대조 연구를 통해 다양한 ML 분류기를 비교하였습니다. 연구의 중점은 데이터 정제 및 전처리, 특성 선택, 매개변수 설정, 모델 선택에 있어 대안적 기법들의 평가에 초점을 맞추고 있습니다.

- **Technical Details**: 우울증, 불안, PTSD 등의 정신장애를 지원하기 위해 설계된 Distress Analysis Interview Corpus - Wizard-of-Oz (DAIC-WOZ) 데이터셋 기반으로 사례 연구가 진행되었습니다. 데이터 정제 및 전처리, 특성 선택, 매개 변수 설정에서 Random Forest와 XGBoost 모델이 사용되었고, 이를 통해 약 84%의 정확도를 달성하였습니다. 이는 기존 문헌에서 보고된 SVM 모델의 72% 정확도보다 현저히 높은 결과입니다.

- **Performance Highlights**: 이 연구는 우울증 진단을 위한 ML 모델과 기법들이 실제로 우수한 성능을 나타낼 수 있음을 보여주며, 특히 Random Forest와 XGBoost 모델을 통해 84%라는 높은 정확도를 달성했습니다. 이는 기계학습이 우울증 진단의 정확성을 향상시키는데 크게 기여할 수 있음을 시사합니다.



### Similar Data Points Identification with LLM: A Human-in-the-loop  Strategy Using Summarization and Hidden State Insights (https://arxiv.org/abs/2404.04281)
- **What's New**: 이 연구는 Large Language Models(LLMs)을 사용하여 표 및 이미지 데이터와 같은 비자유 텍스트 도메인에서 유사한 데이터 포인트를 식별하는 간단하지만 효과적인 방법을 소개합니다. 데이터를 요약하고 숨은 상태(hidden states)를 추출하는 두 단계 접근 방식이 특징입니다.



### When Abel Kills Cain: What Machine Translation Cannot Captur (https://arxiv.org/abs/2404.04279)
Comments: in French language

- **What's New**: 이 논문은 인공지능(AI) 기반 자동 번역기가 구조적 관점에서 완전히 포착하지 못하는 요소들에 대해 조명합니다. 특히, 번역 과정에서 발생하는 오류들에 초점을 맞추어, 그 원인을 설명하려고 시도합니다. 세미적 난이도와 해석적, 비평적 전통의 풍부함 때문에 선정된 성경의 카인과 아벨(Cain and Abel) 이야기를 통해 분석이 이루어졌습니다.

- **Technical Details**: 연구는 가장 잘 알려진 기계 번역 서비스인 Google Translate와 DeepL을 사용하여 이 텍스트의 번역을 관찰하는 것으로 시작합니다. 이어서 번역 오류의 유형을 분류하고 가장 빈번하게 발생하는 번역 오류의 유형을 설정합니다. 그 후, 현대 번역들을 비교 분석하여 각 번역의 독특한 기여를 강조합니다.

- **Performance Highlights**: 본 논문은 문화적 텍스트에 관한 기술적 개선을 제안하며, 번역 이론의 재검토와 이에 따른 기술적 재구성을 제안합니다. 이를 통해 AI 번역기의 한계를 극복하고 더 정교하고 정확한 번역을 가능하게 할 방안을 모색합니다.



### A Novel BERT-based Classifier to Detect Political Leaning of YouTube  Videos based on their Titles (https://arxiv.org/abs/2404.04261)
Comments: 14 pages, 4 figures

- **What's New**: 이 연구는 YouTube 동영상 제목을 기반으로 정치적 성향을 자동으로 탐지하는 첫 번째 연구로, 구글의 언어 모델인 BERT(Bidirectional Encoder Representations from Transformers)를 활용하여 텍스트 분류기를 세밀하게 조정하였습니다. 연구자들은 'Far Left', 'Left', 'Center', 'Anti-Woke', 'Right', 'Far Right'의 여섯 가지 카테고리로 비디오 제목을 분류할 수 있는 새로운 분류기를 제안하였습니다.

- **Technical Details**: BERT 모델을 기반으로하여 YouTube 동영상 제목의 정치적 성향을 분류하기 위해 1,150만 개의 비디오 제목 데이터셋으로 훈련 및 검증이 이루어졌습니다. 연구팀은 Word2Vec, GloVe(Global Vectors for Word Representation)와 같은 다른 사전 훈련된 텍스트 분류기를 사용하였으나, BERT가 가장 우수한 성능을 보여주었고, 정확도(accuracy) 75%, F1 스코어(F1 score) 77%를 달성하였습니다.

- **Performance Highlights**: 이 연구의 BERT 기반 분류기는 알려진 정치적 성향을 가진 여러 뉴스 채널의 YouTube 동영상 제목에 적용되었을 때, 대부분의 경우에 뉴스 채널의 정치적 성향과 일치하는 결과를 예측하였습니다. 이는 BERT 모델이 텍스트의 양방향 문맥을 학습하여 의미 있는 표현을 생성할 수 있음을 보여줍니다.



### Ferret-UI: Grounded Mobile UI Understanding with Multimodal LLMs (https://arxiv.org/abs/2404.05719)
- **What's New**: 이 논문에서는 모바일 UI 화면의 이해를 향상시키기 위해 특별히 개발된 새로운 다중모드 대규모 언어 모델 (multimodal large language models, MLLMs)인 Ferret-UI를 소개합니다. Ferret-UI는 참조(referring), 기반 확립(grounding), 그리고 추론(reasoning) 능력을 갖추고 있으며, 특히 UI 화면은 자연 이미지보다 더 긴 종횡비(aspect ratio)와 더 작은 관심 객체들(예: 아이콘, 텍스트)을 포함하기 때문에, '어떤 해상도에서도(any resolution)' 기능을 통해 세부사항을 확대하여 시각적 특징을 강화하는 방식을 도입합니다.

- **Technical Details**: Ferret-UI는 UI 화면을 원래의 종횡비에 따라 2개의 하위 이미지로 나누고(가로 화면의 경우 수평 분할, 세로 화면의 경우 수직 분할), 각각 별도로 인코딩하여 LLM에 전달합니다. 아이콘 인식, 텍스트 찾기, 위젯 나열과 같은 기본 UI 작업 범위에서 광범위하게 학습 샘플을 수집하며, 정확한 참조와 기반 확립을 용이하게 하는 지역 주석(region annotations)이 포함된 지시(instruction-following) 형식으로 포맷됩니다. 모델의 추론 능력을 증진시키기 위해, 상세 설명, 인식/상호작용 대화(perception/interaction conversations), 기능 추론(function inference) 등의 고급 작업에 대한 데이터셋도 구성합니다.

- **Performance Highlights**: Ferret-UI는 학습된 데이터셋에 대한 훈련을 통해 UI 화면 이해와 개방형 지시사항(open-ended instructions) 수행 능력에서 뛰어난 성능을 나타냅니다. 모델 평가를 위해 해당 작업을 모두 포함하는 종합적인 벤치마크를 설정합니다. Ferret-UI는 대부분의 오픈소스 UI MLLMs을 넘어서며, 기본 UI 작업에서는 GPT-4V보다도 우수한 성적을 거두었습니다.



### AnchorAL: Computationally Efficient Active Learning for Large and  Imbalanced Datasets (https://arxiv.org/abs/2404.05623)
Comments: Published at the NAACL 2024 Conference (main)

- **What's New**: AnchorAL은 큰 규모와 불균형 데이터셋에 적용되는 액티브 러닝(AL, Active Learning)에 대한 학습 및 계산 문제를 동시에 해결하는 새로운 방식을 제안합니다. 각 반복 단계에서 특정 클래스의 레이블이 지정된 인스턴스(앵커)를 선택하고, 이 앵커와 가장 유사한 레이블되지 않은 인스턴스를 취합하여 작은 'subpool'을 형성합니다. 이를 통해 큰 데이타 풀에서도 계산 비용을 낮추면서 효과적으로 레어 클래스를 찾을 수 있습니다.

- **Technical Details**: AnchorAL은 레이블된 세트에서 클래스별 인스턴스를 선택하고 평균 거리에 기반한 유사도 점수를 사용하여 미분류 인스턴스를 평가합니다. 코사인 거리(cosine distance)를 이용하는 언어 모델의 의미적 표현 방식을 사용하여 유사도를 측정합니다. AL 전략에 관계 없이, 고정된 크기의 소형 subpool을 사용하여 어떠한 큰 데이터 풀에도 적용이 가능하며, 반복마다 서로 다른 앵커를 동적으로 선택함으로써 초기 결정 경계를 과적합(overfit)하지 않고 레어 클래스의 새로운 클러스터 발견을 촉진합니다.

- **Performance Highlights**: 실험 결과, AnchorAL은 처리 시간을 시간 단위에서 분 단위로 단축시켜 가장 빠른 방법임이 입증되었으며, 더 적은 어노테이션과 더 적은 시간에 더 높은 성능을 달성하는 뛰어난 성과를 보여주었습니다. 또한, 더 균형 잡힌 데이터 세트를 생성하여, 경쟁 방법보다 우수함을 보여주었습니다.



### 360°REA: Towards A Reusable Experience Accumulation with 360°  Assessment for Multi-Agent System (https://arxiv.org/abs/2404.05569)
- **What's New**: 이 논문에서는 대형 언어 모델(Large Language Model, LLM) 에이전트들이 복잡한 작업을 처리할 수 있도록 하는 새로운 프레임워크인 360° Reusable Experience Accumulation with 360° Assessment(360°REA)를 제안합니다. 이 프레임워크는 회사 조직의 관행에서 영감을 받아 계층적 다중 에이전트 상호작용 체계(hierarchical multi-agent interaction framework)를 적용하며, 효율적인 성과 평가(performance assessment) 및 경험 축적(experience accumulation) 방법을 도입합니다.

- **Technical Details**: 360°REA는 기업의 360도 평가 방식을 모방하여, 다양한 관점에서 에이전트의 성능을 평가합니다. 이 프레임워크는 두 가지 경험 풀(dual-level experience pool)을 사용하여 에이전트들이 더 나은 성과를 낼 수 있도록 지원합니다. 로컬 경험 풀(local experience pool)에서는 에이전트가 세부 평가를 통해 현재의 결과를 반영하고, 글로벌 경험 풀(global experience pool)에서는 리더 에이전트가 최종 결과를 평가하여 종합적인 경험을 축적합니다.

- **Performance Highlights**: 이 프레임워크는 복잡한 작업 데이터셋에 대한 광범위한 실험을 통해 그 효과를 입증하였습니다. 360°REA는 여러 복잡한 작업에서 기존 방법들을 상당히 능가하는 성과를 보여주며, 특히 멀티-에이전트 환경(multi-agent environment)에서의 작업 수행 능력이 개선되었습니다.



### Dense Training, Sparse Inference: Rethinking Training of  Mixture-of-Experts Language Models (https://arxiv.org/abs/2404.05567)
- **What's New**: 이 연구에서는 MoE (Mixture-of-Experts) 모델의 새로운 트레이닝 프레임워크인 DS-MoE (dense training and sparse inference framework)를 제안합니다. 이 모델은 트레이닝 중에는 모든 전문가(expert)가 활성화되어 계산되고, 추론(inference) 동안에는 스파스(sparse) 계산을 사용하여 높은 파라미터 효율성과 계산 비용 절감을 달성합니다.

- **Technical Details**: DS-MoE는 트레이닝 과정에서 모든 전문가가 활성화되어 높은 효율성으로 모델 크기와 동일한 성능을 달성하는 동시에, 추론 시에는 상위 K 전문가만 선택하여 계산을 수행합니다. 이 과정에서 Mutual Information (MI) 손실 함수를 적용하여 전문가의 균등한 분배와 스파스 분포를 보장합니다.

- **Performance Highlights**: DS-MoE-6B 모델은 Mistral-7B 모델과 비교하여 최대 1.86배 빠른 속도를 보였으며, 기존의 MoE 모델인 DeepSeekMoE-16B와 Qwen1.5-MoE-A2.7B보다 1.50배에서 1.71배 더 빠른 성능을 나타냈습니다. 추론(inference) 동안 30-40%의 파라미터만 활성화되어 계산 효율성을 대폭 향상시켰습니다.



### Evaluating Interventional Reasoning Capabilities of Large Language  Models (https://arxiv.org/abs/2404.05545)
Comments: 17 pages

- **What's New**: 이 연구에서는 대규모 언어 모델(LLMs)이 시스템의 다른 부분에 대한 개입을 통해 인과 효과를 추정하는 능력을 평가합니다. 특히 GPT-4와 같은 모델들이 인과 그래프(causal graphs)의 상호 작용에 따라 결과를 어떻게 예측하는지에 초점을 맞춰 연구했습니다. 이는 LLMs가 데이터 생성 과정(data-generating process)에 대한 지식을 어떻게 업데이트하는지 이해하는 데 중요한 역할을 합니다.

- **Technical Details**: 연구팀은 다양한 인과 그래프(confounding, mediation 등)와 변수 유형을 포함하는 벤치마크(benchmarks)를 개발하여, LLMs의 인과 추론(causal reasoning) 능력을 평가합니다. 이 벤치마크를 사용하여 LLMs가 사실을 기억하거나 다른 지름길을 찾는 능력에서 오는 변화를 정확히 예측할 수 있는지를 분리하여 평가할 수 있습니다.

- **Performance Highlights**: GPT-4 모델은 개입 효과(intervention effects)를 예측하는데 있어 높은 정확성을 보였지만, 프롬프트에 있는 주의 산만한 요소에 민감하게 반응합니다. 이러한 결과는 벤치마크를 설계할 때 주의 깊은 설계가 필요함을 강조하며, 훈련된 사실에 기반한 프롬프트가 포함될 때 성능이 크게 떨어질 수 있음을 시사합니다.



### Synergy of Large Language Model and Model Driven Engineering for  Automated Development of Centralized Vehicular Systems (https://arxiv.org/abs/2404.05508)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)과 모델 주도 엔지니어링(MDE)의 상승작용을 활용하여 자동차 산업에서 소프트웨어 개발 프로세스를 자동화하는 도구의 프로토타입을 소개합니다. 이 접근 방식에서 사용자가 제공한 자유 형식의 텍스트 요구사항을 먼저 LLM을 사용하여 Ecore 모델 인스턴스 표현으로 변환하고, OCL(Object Constraint Language) 규칙을 사용하여 일관성을 검사합니다. 일관성 검사 후, 모델 인스턴스는 코드 생성을 위해 다른 LLM에 입력으로 제공됩니다.

- **Technical Details**: 기존의 자동차 아키텍처는 많은 다양한 전자 제어 장치(ECUs)에 의존하고 있습니다. 이는 서로 다른 OEM에서 제공하며 각기 다른 아키텍처, OS 및 소프트웨어 플랫폼을 사용합니다. 이에 따라 소프트웨어 개발 및 유지관리가 점점 더 어려워집니다. 이 논문에서는 중앙집중식 자동차 아키텍처의 도입을 제안하며, 이는 소프트웨어 정의 차량으로의 이행을 촉진하고 하드웨어 업그레이드를 단순화합니다.

- **Performance Highlights**: 중앙집중식 시스템은 하드웨어 비용을 낮추고, 에너지 효율을 향상시키며, 애플리케이션 수준의 통신 속도를 높이는 등의 많은 이점을 제공합니다. 또한, 개발된 프로토타입 도구는 CARLA 시뮬레이터를 사용하여 시뮬레이션된 환경에서 평가되었으며, 긴급 제동 시나리오에서 중앙집중식 차량 아키텍처를 연결하는 예시를 통해 코드 생성의 결과를 평가합니다.



### LayoutLLM: Layout Instruction Tuning with Large Language Models for  Document Understanding (https://arxiv.org/abs/2404.05225)
Comments: CVPR 2024

- **What's New**: 이 논문에서는 문서 이해(document understanding)를 위해 LayoutLLM이라는 새로운 LLM(대규모 언어 모델)/MLLM(다중 모드 대규모 언어 모델) 기반 방법을 제안합니다. 이 방법은 문서 레이아웃 정보를 효과적으로 활용하기 위해 특별히 설계된 레이아웃 지시(instruction) 튜닝 전략을 중심으로 합니다.

- **Technical Details**: LayoutLLM은 레이아웃 인식 사전 학습(Layout-aware Pre-training) 및 레이아웃 인식 지도된 미세 조정(Layout-aware Supervised Fine-tuning)의 두 단계로 구성된 레이아웃 지시 튜닝 전략을 사용합니다. 특히, 레이아웃 사슬 추론(Layout Chain of Thought, LayoutCoT) 모듈을 도입하여 문서의 관련 영역에 집중하고 정확한 답변을 생성할 수 있도록 합니다. 이는 문서 이해 성능을 크게 향상시킬 뿐만 아니라 일정 수준의 해석 가능성(interpretability)을 제공하여 수동 검사 및 수정을 용이하게 합니다.

- **Performance Highlights**: 광범위하게 사용되는 다섯 가지 문서 이해 벤치마크에서 수행한 제로샷(zero-shot) 실험 결과, 기존의 LLMs/MLLMs를 사용하는 방법들을 크게 뛰어넘는 것으로 나타났으며, 문서 레이아웃 모델링의 큰 잠재력을 입증했습니다.



### Have You Merged My Model? On The Robustness of Large Language Model IP  Protection Methods Against Model Merging (https://arxiv.org/abs/2404.05188)
Comments: Technical Report

- **What's New**: 본 논문은 모델 병합(model merging)의 변형된 업스트림 모델(upstream model) 매개변수를 통합하여 그들의 다운스트림 작업 기능을 흡수하는 새로운 방법을 제시하며, 이는 고가의 컴퓨팅 장비(GPUs)나 특정 훈련 데이터 수집에 의존하지 않습니다. 또한, 모델 병합 시 발생할 수 있는 지적 재산권(IP) 침해 문제에 대한 첫 연구를 진행하며, Quantization Watermarking과 Instructional Fingerprint 등의 지적재산권 보호 기술을 검토합니다.

- **Technical Details**: 연구팀은 Quantization Watermarking (양자화 워터마킹)과 Instructional Fingerprint (지시적 지문), 그리고 Task Arithmetic, TIES-MERGING 같은 다양한 고급 모델 병합 기술을 사용하여 실험을 수행했습니다. 이는 다운스트림 태스크의 능력을 효과적으로 통합하기 위한 것입니다.

- **Performance Highlights**: 실험 결과에 따르면, 현재의 대형 언어 모델(Large Language Model, LLM) 워터마킹 기술은 병합된 모델에서 생존할 수 없는 것으로 나타났으나, 모델 지문 기술은 가능성을 보여주었습니다. 이 연구는 모델 병합이 모델 지적 재산권 보호 기술의 견고성 평가에서 중요한 고려사항이 되어야 함을 강조하고 있으며, 이는 오픈 소스 LLM 커뮤니티의 건전한 발전을 촉진할 것입니다.



### DLoRA: Distributed Parameter-Efficient Fine-Tuning Solution for Large  Language Mod (https://arxiv.org/abs/2404.05182)
- **What's New**: 새로운 분산 PEFT 프레임워크인 DLoRA를 소개하며, 이는 클라우드와 사용자 기기 간에 협력적으로 LLM 파라미터의 미세조정을 가능하게 합니다. DLoRA는 개인 데이터의 프라이버시를 유지하면서 LLM 성능을 향상시키기 위한 해결책을 제공합니다. 또한, 새로운 Kill and Revive (KR) 알고리즘을 도입하여 계산 및 통신 부하를 크게 줄이는 동시에 정확도를 유지하거나 향상시킬 수 있는 장점을 갖추고 있습니다.

- **Technical Details**: DLoRA는 사용자의 민감한 데이터를 공유할 필요 없이 LLM 파라미터의 효율적인 미세 조정을 위해 사용자 기기와 클라우드 서버 사이의 작업을 분산시킵니다. KR 알고리즘은 학습 데이터에 가장 민감한 파라미터 집합을 동적으로 식별하고 미세 조정하여 사용자 기기의 계산 및 통신 작업을 크게 줄입니다.

- **Performance Highlights**: DLoRA 및 KR 알고리즘을 통한 평가에서는 계산 부하를 평균 82% 줄이고, 통신 부하를 87.5% 줄이면서도 기존 솔루션보다 비슷하거나 더 우수한 결과를 달성할 수 있었습니다.



### How Bad is Training on Synthetic Data? A Statistical Analysis of  Language Model Collaps (https://arxiv.org/abs/2404.05090)
- **What's New**: 이 논문은 Shumailov 등(2023)에 의해 도입된 모델 붕괴(model collapse) 현상을 심층적으로 조사합니다. 모델 붕괴는 이전 세대 모델에서 생성된 합성 데이터(synthetic data)로 새로운 모델을 훈련할 때 발생하는 성능 저하 현상을 말합니다. 이 논문은 언어 모델(language models)의 모델 붕괴를 이해하기 위한 통계적 모델을 제안하고, 여러 가지 반복 훈련 시나리오에서의 영향을 정량적으로 분석합니다.

- **Technical Details**: 저자들은 전적으로 합성 데이터로만 훈련하는 경우(model collapse is unavoidable when training solely on synthetic data), 모델 붕괴를 피할 수 없다는 것을 보여주었습니다. 그러나 실제 데이터(real data)와 합성 데이터를 혼합하는 경우(model mixing both real and synthetic data), 모델 붕괴를 피할 수 있는 합성 데이터의 최대 비율(maximal amount of synthetic data)을 추정합니다. 이론적 접근은 단일 차원 가우스 분포(single-dimensional Gaussian distribution)와 같은 단순한 수학적 모델에 기반하여 이루어졌으며, 실제 데이터 및 GPT2 스타일 언어 모델(GPT2-style language models)을 사용한 실험을 통해 결과를 검증했습니다.

- **Performance Highlights**: 실제와 합성 데이터의 혼합 훈련에서 모델 붕괴를 방지할 수 있는 합성 데이터의 상한선을 설정함으로써, 언어 모델의 지속 가능한 발전을 위한 중요한 지침을 제공합니다. 이 연구 결과는 모델 붕괴 현상을 이해하고, 향후 생성 모델의 성능 저하를 최소화하는 효과적인 전략 수립에 기여할 수 있습니다.



### A Note on LoRA (https://arxiv.org/abs/2404.05086)
- **What's New**: LoRA (Low-Rank Adaptation) 원래 논문을 확장하고, 새로운 관점과 배포 관련 인사이트(insights)를 제공합니다. 원래의 LoRA는 이전 방법들에 비해 설계 동기와 도전 과제를 완전하게 설명하지 못했는데, 이 노트는 이러한 점을 보완합니다. 특히 LoRA 설계의 기반이 된 폭(width) 확장 개념에 대해 더 자세히 설명하고 있습니다.

- **Technical Details**: LoRA는 Transformer 내 각 레이어(layer)의 matrix에 적용되며, 특히 attention 및 feed-forward network (FFN) 레이어에 집중합니다. 경량화된 Fine-tuning을 위하여, 파라미터 효율적인 방식으로 구현되었습니다. 또한, LoRA는 파라미터 전송 및 최적화 상태를 효율적으로 관리하여 교차 리저널 네트워크에서의 부담을 줄입니다. 이는 대규모 모델에서의 분산 학습 설정(distributed training setup)을 감안할 때 특히 중요한 장점입니다.

- **Performance Highlights**: 실제 배포에서 LoRA가 제공하는 인퍼런스 비용 절감과 훈련 안정성 증가는 주목할 만한 성과입니다. 특히 주의(attention) 레이어와 embedding 매트릭스(matrix)에 LoRA를 적용할 때 가장 안정적이며, 이는 여러 훈련 에포크(epoch)를 거쳐 최적의 성능을 달성합니다. 또한, LoRA를 통한 MoE (Mixture of Experts) 모델의 각 전문가(expert)에 개별적으로 적용할 때 성능이 향상되는 것을 관찰했으나, 이는 메모리 사용량 증가로 인해 비용 효율성이 떨어지는 단점이 있습니다. 전반적으로, LoRA는 학습 과정에서의 효율성 뿐만 아니라, 온라인 서빙(online serving)에서의 비용 효율성 측면에서도 큰 장점을 가집니다.



### HaVTR: Improving Video-Text Retrieval Through Augmentation Using Large  Foundation Models (https://arxiv.org/abs/2404.05083)
- **What's New**: 본 논문에서는 비디오-텍스트 검색(VTR)의 대표적인 한계인 고품질 훈련 데이터 부족 문제를 해결하기 위해 새로운 학습 패러다임인 HaVTR을 제안합니다. HaVTR은 비디오와 텍스트 데이터를 증강하여 더 일반화된 특징을 학습할 수 있게 합니다. 이를 통해 비디오-텍스트 매칭의 효과를 극대화하는 것이 주요 목표입니다.

- **Technical Details**: HaVTR은 세 가지 주요 증강 방법을 통해 데이터를 향상합니다. 첫 번째는 간단한 증강 방법으로, 프레임이나 서브워드(subwords)의 무작위 중복이나 삭제를 통해 데이터를 생성합니다. 두 번째는 대규모 언어 모델(LLMs)과 시각 생성 모델(VGMs)을 사용한 텍스트 패러프레이징(text paraphrasing) 및 비디오 스타일 변환(video stylization) 기법입니다. 마지막으로, 증강된 정보를 활용하여 비디오와 텍스트에 새로운 관련 정보를 생성하고 추가하는 환상 기반(hallucination-based) 증강 방법이 포함됩니다.

- **Performance Highlights**: HaVTR은 비디오-텍스트 검색 벤치마크인 MSR-VTT, MSVD, ActivityNet에서 기존 방법들을 능가하는 성능을 보여주었습니다. 특히, MSR-VTT에서 텍스트-비디오 검색의 Recall @1이 46.1에서 50.8로 향상되었음을 실험을 통해 확인하였습니다.



### QRscript: Embedding a Programming Language in QR codes to support  Decision and Managemen (https://arxiv.org/abs/2404.05073)
Comments: preprint, 8 pages

- **What's New**: 본 연구에서는 QR코드에 프로그래밍 언어(QRscript)를 내장하는 새로운 기술을 제안했습니다. 이 기술은 인터넷 연결 없이도 장치 및 객체를 더욱 스마트하게 만들 수 있는 가능성을 제시합니다. QRscript는 고수준 프로그래밍 언어에서 QR 코드로의 이진 변환(binary conversion)과 그 반대 과정을 상세하게 설명했습니다. 이 언어는 결정 트리(decision trees)를 인코딩하기 위한 특정 하위 언어(sub-language)를 제안함으로써 확장 가능성을 제공합니다.

- **Technical Details**: QRscript 프로그래밍 언어는 가상 머신(virtual machine)을 통해 실행되는 해석형(interpreted) 언어입니다. 이 언어는 고수준 프로그래밍 언어로부터의 번역, QR 코드 생성, 스캔 후 실행까지 포함한 프로세스를 설명합니다. 이러한 과정은 QR 코드에 프로그램을 실행 가능한 형태로 내장시키는데 필수적인 과정들을 포함하며, 매우 간결한 이진 코드(compact binary code) 생성에 초점을 맞추고 있습니다. 본 논문에서는 또한 다양한 QR 코드 버전과 오류 수정 레벨(error correction levels)에 대해서도 설명하고 있습니다.

- **Performance Highlights**: 본 기술은 특히 커뮤니케이션 네트워크 사용이 불가능한 환경에서 매우 유용할 수 있습니다. 예를 들어, 고산, 사막, 산림 지역 또는 인터넷 접속이 어려운 특정 공장 설비 등에서 QR 코드를 통해 정확한 문제 해결을 위한 안내를 제공할 수 있습니다. 이를 통해 작업자는 네트워크 연결 없이도 필요한 정보를 획득하고 문제를 해결할 수 있습니다. 예제로 제시된 산업 네트워크 장치의 설정은 이 기술의 잠재력을 보여주며, 대담하고 혁신적인 접근 방식을 채택함으로써 폭넓은 적용 가능성을 시사합니다.



### FGAIF: Aligning Large Vision-Language Models with Fine-grained AI  Feedback (https://arxiv.org/abs/2404.05046)
- **What's New**: 이 연구는 대규모 시각-언어 모델(LVLMs)의 텍스트와 이미지 간의 정렬 부족으로 발생하는 환각 문제를 해결하기 위한 새로운 접근 방식을 제안합니다. 특히, 객체 존재, 객체 속성, 그리고 객체 관계의 세 가지 환각 유형을 해결하기 위해 세분화된 인공지능 피드백(Fine-Grained Artificial Intelligence Feedback, FGAIF)을 사용합니다. 이 방법은 AI 기반 피드백 수집, 세분화된 보상 모델 훈련, 및 세분화된 보상을 사용한 강화 학습(Reinforcement Learning)의 세 단계로 구성됩니다.

- **Technical Details**: 첫 번째 단계에서는 AI 도구를 사용하여 각 반응의 환각 유형을 예측하고 세분화된 피드백을 수집합니다. 두 번째 단계에서는 수집된 보상 데이터를 바탕으로 세 가지 전문 보상 모델을 훈련하여 밀도가 높은 보상을 생성합니다. 마지막 단계에서는 Proximal Policy Optimization (PPO) 알고리즘에 새로운 세분화된 피드백 모듈을 통합하여 LVLM을 미세 조정합니다.

- **Performance Highlights**: 이 방법은 환각 벤치마크 뿐만 아니라 일반 벤치마크에서도 우수한 성과를 보여주었습니다. 특히, 기존의 RL 기반 정렬 방법으로 훈련된 모델과 비교할 때, 더 적은 매개변수를 사용하면서도 효과적임을 입증했습니다. 또한, 세분화된 피드백이 LVLMs의 환각 문제를 완화하는 데 중요한 역할을 하는 것으로 나타났고, 관련 코드와 매개변수도 공개되어 다른 연구자들이 이용할 수 있습니다.



### Shortcut-connected Expert Parallelism for Accelerating  Mixture-of-Experts (https://arxiv.org/abs/2404.05019)
- **What's New**: 새로운 Shortcut-connected MoE (ScMoE) 아키텍처는 기존 방식의 의존성을 탈피하여 통신과 계산의 전통적인 연속성을 명확하게 분리합니다. 이 아키텍처는 통신과 연산을 70%에서 100% 중첩(overlapping)하여 동시에 실행할 수 있습니다. 이 방식은 기존의 top-2 MoE 아키텍처와 비교하여 훨씬 빠른 훈련 속도와 추론 성능을 제공합니다.

- **Technical Details**: ScMoE는 두 개의 다른 층(layer)의 표현(representations)을 독립적으로 처리하는 두 가지 top-1 게이팅(gating) 메커니즘을 사용합니다. 이는 전통적인 MoE 구조에서 볼 수 있는 통신의 단절(decoupling)을 가능하게 하며, 계산하는 동안 통신을 완전히 중복할 수 있게 합니다. 또한, 이 아키텍처는 하드웨어 구성이나 최적화 기술과 호환되면서도 모델 품질을 유지하거나 향상시킬 수 있습니다.

- **Performance Highlights**: ScMoE는 PCIe 및 NVLink 하드웨어 환경에서 기존 top-2 MoE 대비 각각 30%와 11%의 훈련 속도 향상을 보였습니다. 추론 속도 면에서는 40%와 15%의 개선을 보였습니다. 이러한 성능 향상은 통신이 모델 작동 시간의 상당 부분을 차지하는 환경에서 특히 두드러졌습니다.



### Adapting LLMs for Efficient Context Processing through Soft Prompt  Compression (https://arxiv.org/abs/2404.04997)
Comments: This paper has been accepted by the 2024 International Conference on Image Processing and Computer Applications (IPCA 2024)

- **What's New**: 이 논문에서는 자연어 요약(Natural Language Summarization), 소프트 프롬프트(soft prompts) 압축 및 확장된 유틸리티 보존 기술을 활용하여 대규모 언어 모델(LLMs)의 확장된 컨텍스트 처리를 효율적으로 관리할 수 있는 새로운 프레임워크, SoftPromptComp을 제안합니다. 이 방법은 기존의 대규모 언어 모델의 처리 능력과 확장성을 향상시키기 위해 복잡한 컨텍스트를 효과적으로 압축하고 재배치하는 데 중점을 두고 있습니다.

- **Technical Details**: 이 제안된 프레임워크는 두 가지 주요 기술을 사용합니다. 첫 번째로, 자연어 요약을 통해 긴 텍스트를 간단하고 정보가 풍부한 요약으로 축소합니다. 두 번째로, 이러한 요약을 모델 입력으로 통합하기 위해 훈련 가능한 소프트 프롬프트(soft prompts)와 결합합니다. 이더 댄브롱 접근방식은 LLM의 효과적인 컨텍스트 윈도우(context window)를 확장하고 향상된 텍스트 이해 및 생성을 가능하게 합니다. 또한, 이 프레임워크는 정보 보존 및 가중치 메커니즘을 최적화하여 유용성을 극대화하고 계산 부담을 크게 줄입니다.

- **Performance Highlights**: 연구 결과에 따르면, SoftPromptComp은 다양한 벤치마크에서 계산 오버헤드를 크게 감소시키고, 모델의 효율성과 정확도를 향상시키는 것으로 나타났습니다. 이 방법은 기존 콘텐츠의 품질을 유지하거나 심지어 향상시키면서, LLM의 다양한 NLP 작업에 대한 처리 능력을 증대시킨 것으로 확인되었습니다.



### Elementary fractal geometry. 5. Weak separation is strong separation (https://arxiv.org/abs/2404.04892)
Comments: 27 pages, 12 figures

- **What's New**: 이 연구에서는 자기 유사 집합(self-similar sets)에 대한 새로운 표현 방법을 제시하고 있으며, 특히 유한 유형(finite type) 자기 유사 집합을 그래프 지시 구조(graph-directed construction)로 나타낼 수 있음을 보여줍니다. 이 방법은 개방 집합 조건(open set condition, OSC)을 준수합니다.

- **Technical Details**: 자기 유사 집합은 재귀적으로 스스로를 축소한 복사본의 합집합으로 정의됩니다. 이 연구에서는 기존 IFS(반복 함수 시스템)에서 중복을 제거하여 GIFS(Graph Iterated Function Systems)로 전환하는 방법을 사용하고 있습니다. 또한, 모든 유사성들(similitudes)이 동일한 축소 인자(contraction factor)를 갖는다고 가정하며, 이는 OSC를 만족하는 동시에 WSC(약한 구분 조건)도 만족한다고 설명합니다.

- **Performance Highlights**: 이 연구는 수학적 알고리즘에 기초한 컴퓨터 실험에서 성공적인 결과를 보여주었습니다. 자기 유사 집합의 OSC와 WSC를 특히 분석하면서 유형의 중첩을 제한하는 방법(GIFS 방식)으로 전환함으로써 기존의 IFS보다 효율적으로 문제를 해결할 수 있음을 증명했습니다.



### Binary Classifier Optimization for Large Language Model Alignmen (https://arxiv.org/abs/2404.04656)
Comments: 18 pages, 9 figures

- **What's New**: 새롭게 발표된 이론적 기초를 통해 이진 신호(예: '좋아요' 또는 '싫어요')만을 사용하여 대형 언어 모델(Large Language Models, LLMs)을 인간의 선호도에 맞춰 최적화하는 새로운 방법론, Binary Classifier Optimization(BCO)을 제안합니다. Kahneman-Tversky Optimization(KTO)에서 영감을 받은 이 방법은 Direct Preference Optimization(DPO) 손실을 최소화하는 결과를 내며, 보다 효율적인 LLM의 선호도 맞춤을 가능하게 합니다.

- **Technical Details**: 이 연구는 로짓(logit)이 보상으로 작용하는 이진 분류기를 훈련하는 과정에서 Direct Preference Optimization(DPO) 손실을 간접적으로 최소화함을 밝혀냈습니다. 특히, 이진 크로스 엔트로피 손실(Binary Cross-Entropy, BCE)을 사용할 때, 이는 DPO 손실을 최소화하기 위한 상한선으로 작동합니다. '보상 이동(reward shift)' 및 '기저 분포 일치(underlying distribution matching)'라는 두 가지 기술을 도입하여, 실제 데이터셋에서의 타당성을 입증하였습니다. 분포 일치를 위해 중요도 샘플링(importance sampling)과 밀도 비율 트릭(density ratio trick)을 사용했습니다.

- **Performance Highlights**: 제안된 BCO 방식은 파트너 선호 데이터셋(paired preference dataset) 및 실제 이진 신호 데이터셋(real-world binary signal dataset)에서 DPO, KTO와 비교하여 유사하거나 우수한 성능을 보였습니다. 본 연구는 다양한 기저 분포를 가진 데이터셋에서도 견고한 성과를 보여주었으며, StableLM-2-1.6B 및 Mistral-7B-v0.1과 같은 여러 기본 LLM에서의 효과적인 맞춤을 시연하였습니다.



### TCAN: Text-oriented Cross Attention Network for Multimodal Sentiment  Analysis (https://arxiv.org/abs/2404.04545)
- **What's New**: 본 연구에서는 TCAN(Text-oriented Cross-Attention Network)을 제안하여 텍스트 모달리티(text modality)가 MSA(Multimodal Sentiment Analysis)에서 주요한 역할을 하도록 강조하고 있습니다. 이는 다양한 모달리티들의 기여도가 서로 다를 때 효과적으로 기능하며, 이는 기존의 방법들과 달리 각 모달리티의 의미적 풍부함의 차이를 인정하고 이를 중심으로 기능을 구축합니다.

- **Technical Details**: TCAN은 비정렬 시퀀스를 입력으로 하며, 초기에 추출된 단일 모달 피처들을 시각-텍스트(visual-text) 및 청각-텍스트(acoustic-text) 쌍으로 할당합니다. 이후 텍스트 모달리티에 대하여 셀프 어텐션(self-attention)을 적용하고, 시각 및 청각 모달리티에 대하여 텍스트 기반 크로스 어텐션(text-queried cross-attention)을 적용합니다. 또한, 노이즈와 중복 피처의 영향을 완화하기 위해 게이티드 컨트롤 메커니즘(gated control mechanism)을 도입하였으며, 같은 감정 경향을 파악하기 위해 단일 모달 학습(unimodal joint learning)을 도입합니다.

- **Performance Highlights**: TCAN은 CMU-MOSI 및 CMU-MOSEI 데이터셋에서 최신 기술 대비 우수한 성능을 보여주었습니다. 기존 MSA 방법들에 비해 일관되게 높은 성능을 나타냄으로써, 텍스트 중심의 학습 접근 방식과 게이트 제어 메커니즘이 효과적임을 입증하였습니다.



### Language Models as Critical Thinking Tools: A Case Study of Philosophers (https://arxiv.org/abs/2404.04516)
- **What's New**: 이 논문은 언어 모델(Language Models, LMs)이 철학자들에게 비판적 사고를 가속화하는 도구로 사용될 수 있는지에 대한 연구를 다룹니다. 현재의 언어 모델들이 철학자들이 비판적으로 사고하는 데 도움이 되지 않는다고 판단되는 이유를 분석하고, 언어 모델이 어떻게 개선되어야 할지에 대한 모델을 제안합니다.

- **Technical Details**: 논문에서는 언어 모델이 철학적 비판적 사고에서 사용될 수 있는 세 가지 역할을 제안합니다: 대화자(Interlocutor), 모니터(Monitor), 응답자(Respondent). 이를 위해, 필자들은 자아성(selfhood)과 주도성(initiative)이 결여된 현재의 언어 모델이 철학자들의 비판적 사고 과정에 부적합하다고 주장합니다. 이를 토대로 비판적 사고를 위한 도구로서의 언어 모델을 개선하기 위한 '자아-주도 모델(selfhood-initiative model)'을 소개합니다.

- **Performance Highlights**: 이 연구는 21명의 철학 전문가와의 인터뷰를 바탕으로, 현재의 언어 모델이 비판적 사고에 도움이 되지 않는 주된 이유로서, 모델들이 지나치게 중립적이고, 수동적이며, 무관심하다는 점을 지적합니다. 또한, 철학자들은 대화나 철학적 텍스트의 독서가 언어 모델을 사용하는 것보다 더 유익하다고 느낀다고 합니다.



### Length-Controlled AlpacaEval: A Simple Way to Debias Automatic  Evaluators (https://arxiv.org/abs/2404.04475)
- **What's New**: 본 연구는 AlpacaEval이라는 챗봇 LLM 평가 도구의 출력 길이에 대한 편견을 줄이는 것에 집중합니다. AlpacaEval은 출력의 길이가 긴 모델을 선호하는 경향이 있음에도 불구하고 인간의 선호도와 높은 상관관계를 보여 주었습니다. 이를 해결하기 위해, 연구팀은 길이 제어를 통한 편향 조정을 제안하며, 이를 '길이-제어 AlpacaEval'이라고 명명합니다. 이는 회귀 분석(regression analysis) 접근 방식을 사용하여 자동 평가 메트릭에서의 편향을 제어합니다.

- **Technical Details**: 연구팀은 일반화 선형 모델(generalized linear model, GLM)을 사용하여 자동 평가기가 선호하는 출력에 대해 예측합니다. 이때 중요한 메디에이터(mediators, 즉 조정하고자 하는 변수)로 출력 길이 차이를 사용합니다. GLM을 조건화하여 길이 차이가 0일 때의 선호도를 예측함으로써 길이-제어 선호도를 얻습니다. 길이를 제어함으로써 메트릭의 강인성을 높이고 모델의 장황함(manipulations in model verbosity)에 대한 영향을 줄일 수 있습니다.

- **Performance Highlights**: 길이를 제어한 AlpacaEval은 LMSYS의 Chatbot Arena와의 스피어만 상관 계수(Spearman correlation)가 0.94에서 0.98로 향상되었습니다. 이는 AlpacaEval-LC가 인간의 평가와 더 일치함을 보여 주며, 평가 도구의 정확성과 객관성을 높이는 데 기여합니다. 또한, 코드와 리더보드가 공개되어 다른 연구자들이 이를 활용할 수 있게 되었습니다.



### Counting Like Transformers: Compiling Temporal Counting Logic Into  Softmax Transformers (https://arxiv.org/abs/2404.04393)
- **What's New**: 이 연구는 트랜스포머(transformers)의 계산 능력을 이해하는 데 기여하며, 새로운 시간 계산 논리인 𝙺ₜ[#]와 RASP 변형인 C-RASP를 도입합니다. 이 두 시스템이 서로 등가(equivalent)임을 보이고, 둘을 통합하여 소프트 어텐션 트랜스포머(soft attention transformers)의 표현성에 대한 최고의 알려진 하한을 제공합니다.

- **Technical Details**: C-RASP는 기존의 RASP 프로그래밍 언어를 확장하여 새로운 형태의 소프트 어텐션 트랜스포머 인코더에 컴파일할 수 있도록 설계되었습니다. 𝙺ₜ[#]는 특정 시간 동안의 계산을 수행하는 논리 시스템이며, 여기서 소개된 C-RASP와 등가 관계에 있습니다. 이를 통해, 입력 크기에 제한 없이 미래 마스킹된 소프트 어텐션 트랜스포머 인코더(future-masked soft attention transformer encoders)가 𝙺ₜ[#]에 의해 정의된 모든 공식 언어를 인식할 수 있음을 증명합니다.

- **Performance Highlights**: 이 연구는 트랜스포머가 고정된 정밀도 수를 사용할 때, 이를 𝙺ₜ[#]로 다시 컴파일할 수 있음을 보임으로써, 실제 세계의 트랜스포머의 동작을 더욱 정확히 모델링할 수 있는 기반을 마련합니다. 또한, C-RASP를 사용하여 간단한 트랜스포머 디코더 언어 모델을 구축하고, 이 모델이 𝙺ₜ[#]로 공식적으로 명시된 속성을 가진 문장만을 생성할 수 있도록 합니다.



### Prompt Public Large Language Models to Synthesize Data for Private  On-device Applications (https://arxiv.org/abs/2404.04360)
- **What's New**: 이 연구는 공개 데이터에서 사전 훈련된 대규모 언어 모델(LLM: Large Language Models)이 개인 정보 보호방법(DP: Differential Privacy) 및 연방 학습(FL: Federated Learning)을 이용하여 모바일 어플리케이션 Gboard에서 사용되는 소형 기기 언어 모델의 데이터 품질을 향상시킬 수 있음을 조사했습니다. LLM을 사용하여 실제 사용자 데이터와 유사하게 공개 데이터를 변환하고 새로운 데이터를 생성하여 사전 훈련 데이터의 질을 높였습니다.



### Hypothesis Generation with Large Language Models (https://arxiv.org/abs/2404.04326)
Comments: 26 pages, 6 figures, code link: this https URL

- **What's New**: 본 연구는 대형 언어 모델(LLM: Large Language Models)을 활용하여 고품질의 가설을 생성하는 새로운 방법론을 제안합니다. 특히, 데이터 기반 가설 생성에 초점을 맞추어 초기 가설을 생성한 뒤 상호작용적인 과정을 통해 이를 개선해 나가는 방식을 소개하고 있습니다. 이는 LLM이 긴 맥락을 다루는 것을 가능하게 하며, 실제 세계 데이터셋에서도 강력한 성능을 보여줍니다.

- **Technical Details**: 연구팀은 가설 생성과정에서 탐험과 활용(exploitation-exploration tradeoff)의 균형을 맞추기 위해 다중 무장 밴디트 알고리즘에서 영감을 받은 보상 함수를 설계했습니다. 특정 실험에서의 성능을 보상으로 사용하며, '잘못된 예제 은행(wrong example bank)'을 활용하여 현재 가설의 미비점을 파악하고 새로운 가설을 생성합니다. 이러한 과정은 인간의 도움 없이도 LLM이 높은 품질의 가설을 독립적으로 생성할 수 있음을 보여줍니다.

- **Performance Highlights**: 제안된 알고리즘은 여러 실제 데이터셋에서 기존의 감독학습 및 Few-shot in-context learning 모델을 크게 앞선 성능을 보였습니다. 예를 들어, 'Shoe sales'(신발 판매) 데이터셋에서는 31.7%, 'Deceptive reviews'(기만적 리뷰)에서는 13.9%, 그리고 'Tweet popularity'(트윗 인기도)에서는 24.9%의 정확도 향상을 이뤘습니다. 또한, 이 가설들은 실제와 일치하거나 새로운 통찰력을 제공할 뿐만 아니라 다른 LLM을 사용하여 추론할 때도 강건한 성능을 보여주며, 분포 이탈(out-of-distribution) 데이터셋에서도 뛰어난 일반화 능력을 보여줍니다.



### Parameter Efficient Quasi-Orthogonal Fine-Tuning via Givens Rotation (https://arxiv.org/abs/2404.04316)
- **What's New**: 새로운 파인튜닝 방법으로 quasi-Givens Orthogonal Fine-Tuning (qGOFT)를 제안하였습니다. 이는 기존의 Orthogonal Fine-tuning (OFT) 방법이 가지는 매개변수 효율성 문제와 다운스트림 태스크 적용의 한계를 개선하기 위해 개발된 방법입니다. 특히, Givens 회전 (Givens rotation)을 영감으로 하여 $	ext{SO}(d)$에서 임의의 정교한 변환을 수행할 수 있게 하여 매개변수 복잡성을 $	ext{O}(d^2)$에서 $	ext{O}(d)$로 대폭 감소시키는 데 성공하였습니다.

- **Technical Details**: qGOFT는 $	ext{O}(d)$ Givens 회전을 사용하여 $	ext{SO}(d)$ 내에서 임의의 직교 변환을 수행할 수 있으며, 이는 명백한 동등성을 가지고 있습니다. 또한, 소프트 직교성 (soft orthogonality) 규제 하에 유연한 규범 및 상대적 각도 조정을 도입하여 다운스트림 의미적 편차의 적응 능력을 증진시킴니다.

- **Performance Highlights**: 다양한 태스크와 사전학습된 언어 모델(PLMs)에 대한 광범위한 실험을 통해 qGOFT의 효과성을 검증하였습니다. 이 실험 결과는 qGOFT가 기존 OFT 방식보다 우수한 매개변수 효율성 및 태스크 적응성을 보이는 것을 확인하였습니다.



### AuditGPT: Auditing Smart Contracts with ChatGP (https://arxiv.org/abs/2404.04306)
- **What's New**: 이 논문에서는 이더리움(Ethereum) 스마트 계약의 Ethereum Request for Comment (ERC) 규칙 준수를 자동으로 감사하기 위해 대규모 언어 모델(Large Language Models, LLMs)을 활용하는 새로운 도구인 AuditGPT를 소개합니다. 이 도구는 ERC 규칙을 효율적으로 확인하고, 스마트 계약에서의 위반 사항을 자동으로 식별할 수 있습니다.

- **Technical Details**: AuditGPT는 대규모 언어 모델을 기반으로 하여 ERC 규칙을 자동으로 감사하는 프로세스를 개발합니다. 먼저, ERC 규칙을 이해하고 분석하는 경험적 연구를 수행하고, 그 결과를 바탕으로 스마트 계약 코드의 각 부분을 작은 작업으로 나누어 감사합니다. AuditGPT는 특정 ERC 규칙 유형에 대한 전문화된 질문(prompt)을 설계하여 언어 모델의 감사 성능을 향상시켰습니다.

- **Performance Highlights**: AuditGPT는 평가 과정에서 418개의 ERC 규칙 위반을 성공적으로 식별하였고, 18개의 거짓 긍정(false positives)만을 보고하여 높은 정확성과 효율성을 입증했습니다. 또한, 전문 보안 전문가가 제공하는 감사 서비스보다 효과적이고 정확하며 비용 면에서도 우수함을 보여주었습니다.



### SELF-[IN]CORRECT: LLMs Struggle with Refining Self-Generated Responses (https://arxiv.org/abs/2404.04298)
- **What's New**: 이 연구는 인공 지능(AI) 시스템, 특히 대규모 언어 모델(LLM)이 자체 생성물을 비판적으로 평가하고 개선하는 능력, 즉 자가 개선(self-improvement) 가능성을 탐구합니다. 연구팀은 'Self-[In]Correct' 가설을 도입하여 LLM의 자가 평가(self-discrimination) 능력이 신규 응답 생성 능력(generative capability)을 초과하는지에 대해 의문을 제기합니다. 이는 LLM이 자신들의 출력을 지속적으로 개선할 수 있는지에 대한 중요한 질문입니다.

- **Technical Details**: 연구팀은 두 단계 방법론을 사용하여, 첫 번째 단계에서는 임의로 선택된 LLM 생성물의 메트릭을 사용하여 생성 성능을, 두 번째 단계에서는 LLM에 자신의 생성물 중 최고의 답변을 식별하도록 요청하여 그 메트릭을 평가 성능의 지표로 사용합니다. 이 연구는 LLaMA-2, Mixtral-8x7B 등 널리 사용되는 여러 LLM을 사용하여 수학, 세계 지식 획득, 진실된 질문 응답, 지시 사항 따르기 등 다양한 작업에서 실험을 수행했습니다.

- **Performance Highlights**: 실험 결과, 인간은 일반적으로 생성보다 차별화(discrimination)를 더 쉽게 찾지만, 테스트된 모든 작업에서 LLM의 성능은 이전에 생성된 답변들을 차별화하는 것이 생성된 답변 중 하나를 무작위로 선택하는 것보다 나은 것으로 나타나지 않았습니다. 연구는 또한 자체 회귀 목표(auto-regressive objective)를 사용하지 않은 모델들(예: FLAN-T5, FLAN-UL2)에서는 Self-[In]Correct 현상이 나타나지 않는 것을 발견했습니다.



### Selecting Query-bag as Pseudo Relevance Feedback for Information-seeking  Conversations (https://arxiv.org/abs/2404.04272)
- **What's New**: 이 연구에서는 정보 탐색 대화 시스템을 위한 새로운 프레임워크인 Query-bag based Pseudo Relevance Feedback (QB-PRF)를 제안합니다. QB-PRF는 관련 질문들을 포함하는 query-bag을 구성하여 대화 시스템의 이해도와 응답의 질을 향상시키려고 합니다.

- **Technical Details**: QB-PRF는 두 가지 주요 모듈로 구성됩니다: 1) Query-bag Selection (QBS) 모듈은 대화에 사용될 유사 질의를 선택하는 역할을 하며, 미리 학습된 VAE을 이용하여 비지도학습 방식으로 동의어 질의들을 선택합니다. 2) Query-bag Fusion (QBF) 모듈은 선택된 질의들을 원래의 질의와 통합하여 의미적 표현을 강화합니다. 이 과정은 다차원적 주의(attention) 계산을 사용하여 수행됩니다.

- **Performance Highlights**: 실험 결과는 QB-PRF가 기존의 강력한 기반 모델인 BERT와 GPT-2를 사용하여 벤치마크 데이터셋에서 뛰어난 성능을 보였습니다. 특히, 정보 탐색 시스템의 질문 이해 및 대답 매칭 정확도가 크게 향상되었습니다.



### Cleared for Takeoff? Compositional & Conditional Reasoning may be the  Achilles Heel to (Flight-Booking) Language Agents (https://arxiv.org/abs/2404.04237)
Comments: 18 pages, 17 figures, 3 tables. Paper under review

- **What's New**: 최근 대규모 언어 모델(Large Language Models, LLMs)의 발전은 벤치마크 표준에서 인간의 성능을 능가하는 것을 보여 주었습니다. 이로 인해 복잡한 과제 요구 사항을 탐색하는데 LLM 에이전트(agent)의 사용이 가능해졌습니다. 그러나 간단한 작업과 단순한 상황에서 예상치 못한 문제가 발생하는 경우가 있어, 이들의 진정한 능력을 측정하기 위해 더 다양하고 우수한 평가 방법이 필요합니다. 이에, 인간 인지의 핵심 요소인 구성적(compositional) 및 조건적(conditional) 추론을 연구하고 이를 항공 예약과 연결하는 새로운 벤치마크 'GroundCocoa'를 소개했습니다. 이 벤치마크는 사용자의 상세한 요구 사항과 다양한 항공편 옵션을 연결하는 작업을 포함합니다.

- **Technical Details**: 본 연구에서 소개하는 'GroundCocoa'는 조건적 복잡성을 특징으로 하는 상황에서 모델의 추론 능력을 시험하기 위해 조건부 제약들을 상호 의존하는 구조로 설정합니다. 이 프로세스는 SymPy를 사용하여 만족할 수 있는 제품의 합(Product-of-Sums, POS) 표현식을 생성합니다. 또한, 슬롯 값에 대한 불확실성을 포함하여 각 슬롯에 고유한 제약을 무작위로 부과하며, 이는 최종 사용자 요구 사항이 만족 가능한지 확인하기 위한 검사를 수행합니다. 데이터 생성은 온라인 스크래핑, 제약 생성, 심볼릭 로직을 포함한 5단계 파이프라인을 통해 이루어집니다.

- **Performance Highlights**: 현재 최고 성능을 보이는 LLM 모델인 GPT-4 Turbo조차도 진보된 프롬프팅 기술을 사용함에도 불구하고 67%의 정확도를 초과하지 못해 상당한 성능 격차를 나타냈습니다. 'Chain of Thought' (COT) 프롬프팅은 일부 사례에서만 성능 향상을 가져왔으며, 조건부 추론의 경우 모든 평가된 모델이 낮은 복잡도의 샘플에서도 상당한 도전을 경험했습니다. 또한 비전형적 사용자 요구 사항을 포함시킬 경우 GPT-4 Turbo의 정확도가 최대 6% 감소함으로써, 기존 훈련에서 보다 일반적인 요구 사항에 대한 편향이 있음을 시사합니다.



### Benchmarking and Improving Compositional Generalization of Multi-aspect  Controllable Text Generation (https://arxiv.org/abs/2404.04232)
- **What's New**: 이 연구에서는 다면적 제어 텍스트 생성(Multi-Aspect Controllable Text Generation, MCTG) 방법의 구성적 일반화를 평가하기 위한 새로운 벤치마크인 CompMCTG를 제안합니다. 구성적 일반화는 훈련 데이터의 단일 속성을 재조합하여 새로운 속성 조합으로 텍스트를 생성하는 모델의 능력을 나타냅니다. 기존 MCTG 방법이 구성적 테스트에서 성능 저하를 겪는 것을 관찰한 후, 메타-러닝(meta-learning)을 적용한 새로운 훈련 프레임워크인 Meta-MCTG를 도입하여 이 문제를 완화합니다.

- **Technical Details**: CompMCTG 벤치마크는 다양한 다면적 레이블이 지정된 데이터셋과 세 가지 차원의 평가 프로토콜을 포함하여 MCTG 접근법의 구성적 일반화를 전체적으로 평가합니다. Meta-MCTG 프레임워크는 학습 단계에서 구성적 일반화 시나리오를 시뮬레이션하여 모델이 일반화하는 방법을 학습하도록 합니다. 이는 메타-러닝 기술을 활용하여 구성적 테스트 성능을 최대 3.64%까지 향상시킬 수 있는 중요한 개선을 이루었습니다.

- **Performance Highlights**: Meta-MCTG는 94.4%의 경우에서 구성적 테스트 성능이 눈에 띄게 개선되었습니다. 또한, CompMCTG 벤치마크를 통한 결과 분석으로, 모든 평가된 MCTG 기준 방법들이 분포 내(distribution in) 테스트와 구성적(compositional) 테스트 간에 눈에 띄는 성능 저하를 경험하는 것으로 나타났습니다.



### How Lexical is Bilingual Lexicon Induction? (https://arxiv.org/abs/2404.04221)
Comments: 8 pages, 4 figures. Paper accepted at NAACL Findings 2024

- **What's New**: 이 논문에서는 이중 언어 어휘 유도(Bilingual Lexicon Induction, BLI)에 접근하는 새로운 방법을 제안하여, 최신(retrieve-and-rank) 방식에 추가적인 어휘 정보를 결합하는 방법을 사용합니다. 이는 특히 자원이 부족한(low-resource) 설정에서 데이터의 부족 문제를 해결하는 데 도움이 될 것으로 보입니다. 제안된 방식은 XLING 벤치마크에서 이전 최고 성과보다 평균 2% 향상됨을 보여줍니다.

- **Technical Details**: 이 연구에서는 기존의 BLI 모델 BLICEr에, 단어 빈도와 품사(part of speech, POS) 같은 간단한 어휘 특성을 통합하여 단어 매핑의 정확성을 높이는 방법을 채택합니다. 이는 고밀도 지역의 용어 정렬을 개선하는 데 도움이 될 수 있습니다. 또한, 효율적인 계산을 위해 cross-domain similarity local scaling (CSLS)을 사용하지 않고 제약을 두었습니다. 이 방법은 상대적으로 높은 순위 상관 관계를 가지는 언어쌍에 대해 특히 효과적입니다.

- **Performance Highlights**: 제안하는 모델은 수퍼바이즈드(supervised) 설정에서 1.2% 및 세미-수퍼바이즈드(semi-supervised) 설정에서 2.75%의 상태 향상을 이루었습니다. 이는 언어 간 단어 임베딩(word embedding) 공간에서의 용어 정렬을 개선하며, 높은 밀도의 클러스터에서 용어를 더 효과적으로 매핑할 수 있습니다.



### Unlocking Parameter-Efficient Fine-Tuning for Low-Resource Language  Translation (https://arxiv.org/abs/2404.04212)
Comments: Accepted to the Findings of NAACL 2024

- **What's New**: 이 논문은 낮은 자원 언어(Low-Resource Language, LRL)의 신경 기계 번역(Neural Machine Translation, NMT)에 있어 파라미터 효율적 미세 조정(Parameter-efficient Fine-Tuning, PEFT) 방법의 효과를 종합적으로 평가합니다. 8가지 PEFT 방법과 총 15가지 아키텍처를 사용하여, 다양한 도메인과 데이터 크기에서의 성능을 평가하였으며, 특히 Houlsby+Inversion 어댑터가 전반적으로 가장 우수한 성능을 보였습니다.

- **Technical Details**: PEFT 방법은 기존의 사전 훈련된 모델 내에서 파라미터의 일부만을 업데이트하여 높은 계산 효율성을 유지하며 적응성을 갖추도록 설계되었습니다. 주요 PEFT 아키텍처로는 Houlsby 어댑터, Pfeiffer 어댑터, 병렬 어댑터(Parallel adapter), 반전 어댑터(Invertible adapter) 그리고 여러 방법을 결합한 Mix-and-Match (MAM) 어댑터 등이 있습니다. 각기 다른 아키텍처는 변형기 블록(Transformer block) 내에서 다양한 위치에 적용되어 언어 모델의 성능을 향상시킵니다.

- **Performance Highlights**: 실험 결과, 6개의 PEFT 아키텍처가 기준 모델(Baseline)을 초과하는 성능을 보였으며, 특히 Houlsby+Inversion 어댑터가 내부 도메인과 외부 도메인 테스트 모두에서 뛰어난 결과를 보였습니다. 이러한 결과는 PEFT 방법이 LRL의 번역에서 효과적일 수 있음을 입증합니다. 전반적인 실험은 SacreBLEU 점수를 기준으로 평가되었으며, 다양한 LRL과 데이터 세트 크기에서 최적의 구성을 결정하기 위해 추가 실험이 수행되었습니다.



### Social Skill Training with Large Language Models (https://arxiv.org/abs/2404.04204)
- **What's New**: 본 논문은 사회 기술 훈련을 위한 새로운 접근 방법인 AI Partner 와 AI Mentor (APAM) 프레임워크를 제안합니다. 이는 대규모 언어 모델(Large Language Models, LLM)을 활용하여 실제 같은 상황에서 맞춤형 피드백을 제공하며, 이를 통해 사회적 기술을 안전하고 효과적으로 학습할 수 있는 가상 환경을 제공합니다.

- **Technical Details**: APAM 프레임워크는 LLM을 기반으로 한 시뮬레이션, 도메인 전문 지식 및 피드백 효율성, 시나리오 상황과 학습자 지식 상태의 인식 등을 통해 사용자에게 효과적인 학습 경험을 제공합니다. AI Partner는 대화 시뮬레이션을 통해 적절한 시나리오 하에서 사회적 상황을 연습할 수 있도록 돕고, AI Mentor는 상황에 근거한 지식을 토대로 피드백을 제공합니다. Retrieval Augmented Generation(RAG) 같은 기술을 사용하여 외부 지식을 동적으로 통합하고, 계획 모듈을 통해 실행 가능한 후보를 평가하고 필터링합니다.

- **Performance Highlights**: APAM 프레임워크는 사회적 스킬 학습의 접근성을 향상시키고, 시범 사용자 연구를 통해 그 효과성을 입증하였습니다. 이 시스템은 사용자가 안전한 환경에서 현실적인 상황을 경험하면서 적절한 도메인 지식을 배울 수 있게 하며, 경제적 배경이나 전문 지식의 유무에 관계 없이 모든 사용자가 고등기술을 배울 수 있는 기회를 제공합니다.



### Do Sentence Transformers Learn Quasi-Geospatial Concepts from General  Text? (https://arxiv.org/abs/2404.04169)
Comments: Presented at the Second International Workshop on Geographic Information Extraction from Texts at ECIR 2024 (this https URL)

- **What's New**: 이 연구는 일반 질문-응답 데이터세트(general question-answering datasets)에서 미세 조정된(fine-tuned) 문장 트랜스포머(sentence transformers)가 영국 전역에 있는 사람이 만든 경로 설명과 등산 경험을 자주 묘사하는 데 사용되는 쿼리를 연결할 수 있는지를 조사합니다. 이 연구는 문장 트랜스포머가 루트 유형 및 난이도와 같은 준-지리 공간 개념(quasi-geospatial concepts)을 이해할 수 있는 일부 제로샷(zero-shot) 능력을 가지고 있음을 발견했습니다. 이는 라우팅 추천 시스템(routing recommendation systems)에 대한 잠재적 유틸리티를 시사합니다.

- **Technical Details**: 연구 과정에서는 영국 전역의 사용자 생성 경로를 사용하고 geopandas를 사용하여 각 경로에 대한 텍스트 설명을 생성하고(sentence transformers를 사용하여 각 설명) 벡터 임베딩(vector embeddings)으로 변환합니다. 사용된 문장 트랜스포머 모델들은 주로 BERT 계열의 변형자 기반 신경망(Transformer-based Neural Networks)이며, MSMARCO 및 Multi-QA 데이터세트에서 미세 조정되어 비대칭 의미 검색(asymmetric semantic search)을 위해 설계되었습니다.

- **Performance Highlights**: 연구 결과에 따르면, 쿼리와 문서 간에 유사한 용어를 사용할 때 성능이 향상되었습니다. 모델은 ‘해안 산책(seaside walk)’이라는 쿼리를 해안을 따라 긴 구간이 포함된 루트와 연결하는 데 효과적이었고, '초보 하이커' 또는 '이동 능력이 제한된 사람'을 위한 더 쉬운 경로를 제안하는 쿼리와 관련해서는 결과가 명확했습니다. 그러나 '전문 하이커’를 위한 더 도전적인 경험을 찾는 쿼리의 성공률은 낮았습니다. 여러 모델이 같은 데이터세트에서 미세 조정되었음에도 불구하고 때로는 결과가 매우 달랐습니다.



### Chinese Tiny LLM: Pretraining a Chinese-Centric Large Language Mod (https://arxiv.org/abs/2404.04167)
- **What's New**: 이 연구는 중국어 중심의 대규모 언어 모델인 CT-LLM(Chinese Tiny Large Language Model)을 소개합니다. CT-LLM은 기존의 대부분의 LLM이 영어 데이터셋에 중점을 두고 있는 것에서 벗어나, 1,2000억 토큰으로 구성된 막대한 데이터 셋에서 중국어 8000억 토큰, 영어 3000억 토큰, 코드 1000억 토큰을 포함하여 처음부터 다르게 접근합니다. 이로써 다양한 언어들의 훈련 방법론의 지평을 넓히고 있습니다.

- **Technical Details**: CT-LLM은 20억 개의 매개변수를 가진 모델로써, 새롭게 큐레이션 된 매시브 적절한 사전 훈련 중국어 코퍼스(MAP-CC)를 통해 고품질의 중국어 웹 말뭉치의 표준을 설정합니다. 이 모델은 중국어 이해 및 처리 능력에서 탁월함을 보이며, SFT(Supervised Fine-Tuning)를 통해 영어 작업에서도 능력을 발휘합니다. 또한, 중국어 어려운 사례 벤치마크(CHC-Bench)를 통한 평가에서 뛰어난 성능을 보입니다.

- **Performance Highlights**: CT-LLM은 CHC-Bench에서 뛰어난 성능을 보여주며, 이는 중국어 태스크에 대한 모델의 능력을 입증합니다. 또한, SFT를 통해 영어 이해 및 생성 작업에서도 그 능력을 보이며, 다양한 언어 능력을 갖춘 모델로서의 면모를 보여줍니다.



### BEAR: A Unified Framework for Evaluating Relational Knowledge in Causal  and Masked Language Models (https://arxiv.org/abs/2404.04113)
Comments: NAACL 2024

- **What's New**: 이 논문에서는 다양한 유형의 언어 모델(LM) 간의 지식 비교를 가능하게 하는 새로운 접근 방식인 BEAR(Benchmark for Evaluating Associative Reasoning)를 제안합니다. 기존의 LAMA 프레임워크와 달리 BEAR는 마스크(masked) 및 인과(causal) 언어 모델 모두에 적용될 수 있으며, 정답 옵션을 텍스트 문장으로 변환하여 각 문장의 로그-가능도(log-likelihood)를 평가합니다.

- **Technical Details**: BEAR 접근 방식에서는 각 관계 인스턴스(relational instance)에 대한 답변 옵션을 제공하고, 각 옵션에 대한 텍스트 문장을 생성한 다음, 각 언어 모델이 문장에 로그-가능도 점수를 할당하도록 함으로써 이러한 옵션을 순위(rank) 매깁니다. 이는 LAMA가 한정된 토큰 예측 문제로 접근하는 것과 대조되며, BEAR는 답변 공간(answer space)에 대한 제한이 없고 다양한 유형의 정답을 평가할 수 있습니다.

- **Performance Highlights**: 실험 평가에서 BEAR 프레임워크는 22가지 일반적인 마스크 및 인과 언어 모델을 사용하여 지식을 효과적으로 프로빙(probing)할 수 있음을 보여줍니다. 새로운 평가 데이터셋은 7,731개 인스턴스(확장된 버전에서 40,916개)를 포함하고 있으며, 이는 이전 연구보다 폭넓은 문제들을 해결할 수 있도록 해줍니다.



### Improving Factual Accuracy of Neural Table-to-Text Output by Addressing  Input Problems in ToTTo (https://arxiv.org/abs/2404.04103)
Comments: Added link to human evaluation guidelines and error annotations

- **What's New**: 이 논문에서는 정치 도메인의 ToTTo 데이터셋을 사용하여 신경망 모델이 생성한 텍스트에서 사실적 오류가 발생하는 원인을 조사합니다. 특히, 입력 데이터의 문제가 출력 텍스트의 오류로 이어질 수 있다는 점을 밝히고, 입력을 수정함으로써 사실 오류를 52%에서 76% 감소시킬 수 있음을 보여줍니다.

- **Technical Details**: 연구팀은 T5-base, T5-large, Llama 2-7B, 및 Llama 2-13B 모델(Model)을 사용하여 입력 테이블(Table)의 문제를 수정하는 실험을 수행했습니다. 입력 데이터의 구조적 문제를 해결하고 적절한 설명을 생성하기 위해 표에서의 주요 셀(Key Cells)을 정확히 매핑(Mapping)하는 것에 초점을 맞추었습니다. 수정된 입력은 정확성을 높이는 데 기여하며, 모델이 내용을 더 정확하게 처리할 수 있도록 합니다.

- **Performance Highlights**: 수정된 입력에 기반한 T5-base 모델은 사실 오류를 62% 감소시켰으며, T5-large 모델은 57% 감소를 보였습니다. 또한, Llama 2-7B 모델에서는 52% 감소, Llama 2-13B 모델에서는 76% 감소를 관찰하였습니다. 이는 입력 데이터의 정확성이 모델 출력의 품질에 큰 영향을 미친다는 것을 입증합니다.



### Assessing the quality of information extraction (https://arxiv.org/abs/2404.04068)
- **What's New**: 이 연구에서는 체계적인 정보추출(e.g., 엔티티 인식, 관계 추출)의 질을 평가하는 새로운 프레임워크를 개발했습니다. 특히, 라벨이 없는 데이터에서도 성능을 평가할 수 있는 MINEA(Multiple Infused Needle Extraction Accuracy) 점수를 도입하였습니다. 이 점수는 인위적으로 생성된 데이터를 활용하여 추출된 정보의 정확도를 측정합니다.

- **Technical Details**: 이 프레임워크의 핵심은 대규모 언어 모델(Large Language Models, LLMs)을 사용하여 텍스트로부터 정보를 구조화하는 것입니다. 스키마 마크업(Schema markup)과 같은 정보에서, 각 엔티티(예: '사람', '조직')는 속성(예: '이름', '근무지')으로 설명됩니다. 또한, LLM의 입력/출력 사이즈 제한과 처리 문제를 해결하기 위한 방안이 제시되며, 추출 과정에 대한 반복적 접근 방식을 설명합니다.

- **Performance Highlights**: MINEA를 이용한 실증 분석은 라벨이 없는 환경에서도 LLM 기반 정보 추출의 유용성을 보여줍니다. 'Lost in the middle' 현상에 대한 고찰 및 입력/출력 제한(예: gpt-4-1106 모델의 128000 토큰 입력 제한)과 같은 LLM의 제한 사항에 대해서도 논의됩니다.



### CLUE: A Clinical Language Understanding Evaluation for LLMs (https://arxiv.org/abs/2404.04067)
- **What's New**: 이 연구는 의료 분야에서 Large Language Models (LLMs, 대규모 언어 모델)의 실제 임상 과제 평가를 위해 CLUE (Clinical Language Understanding Evaluation) 벤치마크를 소개합니다. 이는 MIMIC IV에서 파생된 두 개의 새로운 데이터세트와 네 가지 기존 태스크를 포함하여, 의료 환경에서 LLM의 실용적 적용 가능성을 테스트하기 위해 설계되었습니다.

- **Technical Details**: CLUE 벤치마크는 비밀보호 및 연산 제약과 같은 특정 의료 도전을 해결하기 위해 특별히 개발된 생체의학 LLM과 일반 도메인 LLM의 임상 성능과 적용 가능성을 평가합니다. 이 연구는 비밀보호를 요구하는 임상 환경에서의 사용을 피하는 상용 LLM은 제외하고, 오픈소스 모델들을 중심으로 이루어졌습니다. 평가는 LLM들이 임상 문서를 해석하고 관리하는 능력을 탐구하여, 의료 설정에서 LLM 배치에 대한 보다 정보에 기반한 결정을 내릴 수 있는 길을 제시하고자 합니다.

- **Performance Highlights**: CLUE를 통한 평가는 다수의 생체의학 및 일반 도메인 LLM을 포함하여, 그들의 임상 성능을 세밀하게 조명합니다. 비밀보호 요구와 연산 제약을 고려한 특화 모델이 일반 도메인 모델과 어떻게 다른지, 그리고 실제 임상 환경에서 얼마나 잘 수행되는지에 대한 깊이 있는 비교를 제공합니다. 이러한 성능 평가는 향후 모델 개발을 실제 임상 응용의 필요성과 더욱 일치시키는데 기여할 것입니다.



### Teaching Llama a New Language Through Cross-Lingual Knowledge Transfer (https://arxiv.org/abs/2404.04042)
- **What's New**: 이 논문은 사전 학습된 대규모 언어 모델(Large Language Models, LLMs)을 새로운 저자원 언어로 적용하는 비용 효율적 방법을 탐구하며, 특히 에스토니아어에 초점을 맞춥니다. 최초로 개방형 지시(instruction-following) LLM을 에스토니아어로 출시하고, 에스토니아어를 위한 첫 일반 작업 지시 데이터셋(Alpaca-est)을 발표합니다.

- **Technical Details**: Llama 2 모델을 이용해, 교차 언어 지시 튜닝(cross-lingual instruction-tuning)과 추가적인 단일 언어 사전 학습(monolingual pretraining)의 결합이 미치는 영향을 조사했습니다. 이는 상당히 적은 양의 추가 단일 언어 사전 학습이 교차 언어 지시 튜닝 후 에스토니아어 성능을 크게 향상시키는 것으로 나타났습니다.

- **Performance Highlights**: 개선된 LLM은 상식적 추론(common sense reasoning)과 다중 턴 대화(multi-turn conversation) 능력을 포함하여 에스토니아어로의 품질 높은 영어 지시사항에서의 교차 언어 지식 전달(cross-lingual knowledge transfer)을 보여주며 성능을 향상시킵니다.



### A Dataset for Physical and Abstract Plausibility and Sources of Human  Disagreemen (https://arxiv.org/abs/2404.04035)
Comments: Accepted at The 17th Linguistic Annotation Workshop

- **What's New**: 이 연구는 사건의 구체성 및 추상성에 따른 가능성을 분석하는 새로운 데이터셋을 제시합니다. 이전 연구들은 주로 물리적으로 가능한 사건과 불가능한 사건을 구분하는 데 초점을 맞추었지만, 이번 연구는 추상적 사건(abstract events)의 가능성까지 평가하는 모델을 개발하고자 했습니다. 예를 들어, 법이 차별을 금지하는 것은 추상적으로 가능하지만, 유머가 합병을 요구한다는 것은 추상적으로 불가능합니다.

- **Technical Details**: 연구진은 영어 위키백과에서 자연 발생 문장을 추출하여, 사건 참여자의 추상성을 평가하고, 추상성의 다양한 정도에 따라 데이터를 분류했습니다. 인공적으로 생성된 비가능 사건들을 포함하여 총 1,733개의 사건 트리플(naturally occurring sentences, 구체성 강도(abstractness ratings))에 대해 평균 8.9개의 인간 평가자(judgements)를 통해 가능성 평가를 수행하였습니다. 또한, 추상성과 가능성 사이의 상관관계를 조사하여, 구체적인 단어(more concrete words)가 많은 사건에서는 불가능성(implausibility)이 더 많이 관찰되었다는 결과를 확인했습니다.

- **Performance Highlights**: 이 새로운 데이터셋은 추상적 사건의 가능성 평가에 있어 획기적인 점을 제시하며, 주어진 사건에 대한 인간의 평가가 매우 다양하고 개인적인 것을 반영합니다. 연구는 가능성(implausibility)보다 가능성(plausibility)을 선호하는 annotator의 경향을 발견했으며, 불가능한 사건에서 더많은 의견 불일치(disagreement)가 있었습니다. 또한, 추상적인 단어들이 많은 사건이 가능성을 자아내는 것과 연관이 있다는 중요한 발견을 했습니다.



### Willkommens-Merkel, Chaos-Johnson, and Tore-Klose: Modeling the  Evaluative Meaning of German Personal Name Compounds (https://arxiv.org/abs/2404.04031)
Comments: Accepted at LREC-COLING 2024

- **What's New**: 이 연구는 독일어의 개인 이름 복합체(PNCs)라는 비주목적인 현상에 대한 포괄적인 컴퓨터 모델링 연구를 제시합니다. 예를 들어 'Willkommens-Merkel' (환영 메르켈)과 같은 PNCs는 뉴스, 소셜 미디어, 정치 담론에서 자주 나타나고, 이러한 이름들이 해당 인물의 전체 이름보다 긍정적 또는 부정적인 인식을 반영한다고 가정합니다.

- **Technical Details**: 연구팀은 321개의 PNCs와 해당하는 전체 이름을 분석하여, 다양한 계산 방법을 통해 PNCs의 평가적 성격을 파악했습니다. 또한, 이들은 감정 분석(sentiment analysis)과 관련된 작업을 수행하기 위해 사전 훈련된 언어 모델(pretrained language models, PLMs)을 활용하여 PNCs의 감성을 평가하는 두 가지 접근 방식을 적용하고 비교했습니다.

- **Performance Highlights**: 분석 결과, PNCs는 전체 이름에 비해 긍정적 또는 부정적인 평가를 받는 경향이 있으며, 특히 정치인들에 대한 PNCs는 더 부정적인 의미를 지니는 반면, 스포츠와 쇼 비즈니스 분야에서는 반대의 경우가 나타났습니다. 또한 PNC를 구성하는 수식어(modifier)의 감정 valence가 전체 복합어 평가에 중요한 영향을 미치는 것으로 나타났습니다.



### Good Books are Complex Matters: Gauging Complexity Profiles Across  Diverse Categories of Perceived Literary Quality (https://arxiv.org/abs/2404.04022)
- **What's New**: 이 연구에서는 Norton Anthology, Penguin Classics 시리즈, Open Syllabus 프로젝트에서 선정된 작품들과 현대 베스트셀러, 노벨문학상 수상자 및 유명 문학상 수상작을 포함한 말뭉치(corpus)를 사용하여 문학적 '품질'의 다양한 범주가 독특한 언어적 프로필을 보이는 것을 분류 접근 방식을 통해 보여줍니다. 고전적(classical) 머신 러닝 방법인 랜덤 포레스트(Random Forest)를 적용해 '품질'과 '비품질(control groups)' 소설을 구분하는 과정을 수행하였으며, 카테고리 간 구별을 위해 최대 77%의 F1 점수를 달성했습니다.

- **Technical Details**: 분석 결과, 카논(canonical)이나 고급(high-brow) 텍스트는 베스트셀러나 대중적인 작품들, 그리고 컨트롤 그룹과 비교했을 때 뚜렷한 텍스트 특성을 보였습니다. 이는 각각 독특하지만 서로 배타적이지 않은 품질 모델에 대응할 가능성을 시사합니다. '품질' 범주는 다른 '품질' 범주들과 비교했을 때 컨트롤 그룹과 구별하기가 더 쉽다는 점을 발견했습니다. 이는 문학의 질적 특성(features)이 구별 가능하되, 품질 대리자(quality proxies)를 통해 공유될 수 있다는 것을 암시합니다.

- **Performance Highlights**: 품질구분 알고리즘은 최대 77%의 F1 점수로 다양한 카테고리 간 구별 성능을 보였으며, 이는 상당히 높은 수준의 정확성을 나타냅니다. 특히, '품질' 작품들은 컨트롤 그룹과 비교할 때 구분이 더 용이한 것으로 나타나, 알고리즘의 유효성을 확인할 수 있었습니다.



### BuDDIE: A Business Document Dataset for Multi-task Information  Extraction (https://arxiv.org/abs/2404.04003)
- **What's New**: 이 연구에서는 다양한 비즈니스 관련 문서를 대상으로 하는 새로운 데이터셋 BuDDIE (Business Document Dataset for Information Extraction)를 소개합니다. BuDDIE는 문서 분류(DC), 주요 엔티티 추출(KEE), 시각적 질문 응답(VQA) 등 다양한 VRDU (Visually Rich Document Understanding) 작업을 지원하는 1,665개의 실제 비즈니스 문서로 구성됩니다. 이 데이터셋은 미국 주 정부 웹사이트에서 공개적으로 이용 가능한 비즈니스 엔티티 문서들로, 스타일과 레이아웃이 다양합니다. 또한 다양한 문서 유형(예: 양식, 증명서, 보고서 등)을 포함하고 있습니다.

- **Technical Details**: BuDDIE는 하나의 문서에서 여러 VRDU 작업을 수행할 수 있어야 하는 실제 환경에서의 문서 처리 요구 사항을 평가할 수 있게 해줍니다. 또한 구조화된 비즈니스 문서에 대한 풍부하고 밀도 높은 주석을 포함하며, 69개의 주요 엔티티 클래스에 대한 계층적 온톨로지를 생성하여, 미래에 더 많은 엔티티 유형으로 확장할 수 있습니다. 데이터셋은 비교적 쉬운 레이아웃에서 복잡한 레이아웃의 문서까지 다양한 스타일을 포함하기 때문에, 다양한 VRDU 작업에 매우 적합합니다.

- **Performance Highlights**: BuDDIE 데이터셋에는 여러 베이스라인 모델이 포함되어 있으며, 그중 DocLLM이 모든 작업에 걸쳐 가장 뛰어난 성능을 보여줍니다. DocLLM은 문서 분류에서 F1 점수가 99.15, 주요 엔티티 추출에서 F1 점수가 89.97, 시각적 질문 응답에서 ANLS 점수가 89.58을 달성하였습니다. 이는 이 데이터셋이 비즈니스 문서 이해 작업에 매우 유용하며 효과적인 학습 및 평가 도구임을 보여줍니다.



### Investigating the Robustness of Modelling Decisions for Few-Shot  Cross-Topic Stance Detection: A Preregistered Study (https://arxiv.org/abs/2404.03987)
Comments: Accepted at LREC-COLING 2024: cite the published version when available

- **What's New**: 이 논문에서는 뉴스 기사가 같은 관점을 표현하는지 식별하는 데 필수적인 몇 가지 시야(stance) 감지 접근 방식과 강건성을 탐구합니다. 특히 다양한 주제에 걸쳐 시야를 모델링하는 것에 중점을 두고, 실험을 통해 사전등록된 가설을 검증하고, 'Pro/Con'과 'Same Side Stance'라는 두 가지 시야 작업 정의를 비교합니다. 또한, 이 연구는 복수의 데이터셋과 시스템적 모델링 실험이 필수적임을 강조하며, 연구 가설 및 실험을 사전등록하는 방식을 통해 투명성과 체계성을 높이고자 합니다.

- **Technical Details**: 이 연구는 두 가지 주요 시야(stance) 작업 정의(Pro/Con 대 Same Side Stance), 두 가지 LLM 아키텍처(bi-encoding 대 cross-encoding), 그리고 자연어추론(Natural Language Inference, NLI) 지식을 추가하는 것을 비교하며, RoBERTa 모델을 사용하여 100개 예제에서 훈련된 7개의 다른 시야 감지 데이터셋에서 실험을 진행하였습니다. 특히, cross-encoding 방식이 bi-encoding 방식보다 일반적으로 우수한 성능을 보였으며, NLI 훈련을 추가한 모델이 상당한 성능 향상을 보였지만 모든 데이터셋에서 일관된 결과는 아니었습니다.

- **Performance Highlights**: 이 연구에서는 다양한 시야 감지 데이터셋에 대한 모델링 선택의 영향을 평가하였습니다. 'Same Side Stance' 정의의 성능 효과는 데이터셋에 따라 다르며 다른 모델링 선택에 영향을 받는 것으로 나타났습니다. 훈련 주제의 수와 성능 사이에는 관련성이 없었으며, 일반적으로 cross-encoding이 bi-encoding보다 성능이 뛰어난 것으로 확인되었습니다.



### SEME at SemEval-2024 Task 2: Comparing Masked and Generative Language  Models on Natural Language Inference for Clinical Trials (https://arxiv.org/abs/2404.03977)
- **논문 요약**: [{"What's New": '이 논문은 SemEval 2024 Task 2: 임상 시험(Clinical Trials)을 위한 안전한 생물의학 자연어 추론(Natural Language Inference, NLI)에 대한 접근 방식을 제시합니다. 임상 시험 보고서(Clinical Trial Reports, CTR)에 적용된 자연어 추론 모델의 일관성과 신뢰성을 평가하는 텍스트 함의(Textual Entailment, TE) 작업에 중점을 둡니다.'}, {'Technical Details': '연구팀은 세 가지 주요 기술을 사용하여 NLI 작업에 접근합니다: 첫째, 마스크 언어 모델(Masked Language Models, MLMs)의 미세조정; 둘째, 대규모 언어 모델(Large Language Models, LLMs)을 사용한 프롬프트; 셋째, 사고의 연쇄(Chain-Of-Thought)와 대조적 사고의 연쇄(Contrastive Chain-Of-Thought) 기법을 활용한 템플릿을 사용합니다. Flan-T5-large 모델을 2-shot 설정에서 사용하였을 때 가장 좋은 결과를 보여 0.57의 F1 점수와 0.64의 신뢰도, 0.56의 일관성을 달성했습니다.'}, {'Performance Highlights': '이 시스템은 텍스트 함의 작업에 있어 0.57 F1 점수를 달성했으며, 신뢰성(Faithfulness)과 일관성(Consistency)에서 각각 0.64와 0.56의 결과를 얻었습니다. 이러한 성능은 특히 생물 의학 분야에서의 안전한 자연어 처리 작업에 중요한 통찰력을 제공할 수 있습니다.'}]



### Data Augmentation with In-Context Learning and Comparative Evaluation in  Math Word Problem Solving (https://arxiv.org/abs/2404.03938)
Comments: Accepted in SN Computer Science

- **What's New**: 이 연구는 수학 문제 해결(Natural Language Processing, NLP)의 중요한 작업을 개선하기 위해 다양한 데이터 증강(Data Augmentation) 방법을 제시합니다. 새로운 접근으로 Llama-7b 언어 모델을 사용하는 인-콘텍스트 학습(In-context Learning) 증강 방법을 소개하며, 이는 문제 텍스트를 다시 구성하는 데 있어 지시 기반 프롬프팅(Instruction-based Prompting)을 활용합니다.

- **Technical Details**: 제안된 데이터 증강 방법에는 동의어 교체(Substitution: Synonym Replacement), 규칙 기반 질문 교체(Rule-Based: Question Replacement), 규칙 기반 질문 반전(Rule-Based: Reversing Question)이 포함됩니다. 추가적으로, Llama-7b 모델을 활용한 새로운 인-콘텍스트 학습 방법이 소개되었으며, 이는 문제 텍스트를 재구성하여 더 다양한 훈련 데이터 세트를 생성합니다. 이 연구에서는 9가지 기본 모델(Baseline Models)을 사용하여 이러한 증강 방법들의 성능을 평가했습니다.

- **Performance Highlights**: 종합적인 실험을 통해 데이터 증강 방법이 기본 모델들을 능가하는 성능을 보였으며, 서로 다른 증강 방식의 예제를 결합하는 것이 더욱 성능을 향상시켰습니다. 각각의 증강 방법들이 수학 단어 문제(Math Word Problems, MWPs)를 해결하는 능력에서 상당한 개선을 보였습니다.



### Simple Techniques for Enhancing Sentence Embeddings in Generative  Language Models (https://arxiv.org/abs/2404.03921)
Comments: Work in Progress

- **What's New**: 본 논문에서는 자연어 처리 (Natural Language Processing, NLP) 분야의 주요 과제 중 하나인 문장 임베딩에 대해 다루고 있습니다. 기존에는 대규모 언어 모델들 (Large Language Models, LLMs)을 예시로 활용하면서 성능을 향상시키기 위해 미세조정(fine-tuning)에 중점을 두었지만, 이 연구에서는 효율적인 직접 추론(direct inference) 방식을 탐구하여 연산 리소스를 절약하면서도 고품질의 문장 임베딩을 도출하는 새로운 방법을 제시합니다. 특히, 'Pretended Chain of Thought (CoT)' 및 'Knowledge Enhancement'라는 두 가지 새로운 프롬프트 기법(prompt engineering techniques)을 도입하여 문장의 표현력을 높였습니다.

- **Technical Details**: 제안된 두 기법은 Pre-trained Language Models(PLMs)의 효과적인 활용을 가능하게 합니다. 'Pretended CoT'는 모델에게 한정된 실질적인 입력만 제공함으로써, 모델이 문맥을 이해하고 내부적으로 합리적인 추론을 할 수 있도록 설계되었습니다. 반면, 'Knowledge Enhancement'는 텍스트 요약에서의 인간 지식을 반영한 명확한 가이드라인을 제공하여, 모델이 더 정확한 문장 임베딩을 생성하도록 유도합니다. 이러한 기법들은 학습 파라미터의 업데이트 없이도 LLMs로부터 고품질의 문장 임베딩을 직접 추출할 수 있도록 돕습니다.

- **Performance Highlights**: 연구팀은 다양한 PLM 유형과 규모에 걸쳐 실험을 수행했으며, 'Pretended CoT'와 'Knowledge Enhancement'가 기존 방법보다 우수한 성능을 보임을 확인했습니다. 구체적으로는 Semantic Textual Similarity (STS) 벤치마크를 통해 이 기법들이 메모리 효율성을 최적화하면서도 문장 임베딩의 성능을 크게 향상시킬 수 있음을 입증했습니다. 할당된 GPU 메모리 사용량이 기존 QLoRA를 사용한 미세조정 방법 대비 훨씬 적었으며(Table 1 참조), 동시에 높은 임베딩 품질을 유지했습니다.



### Forget NLI, Use a Dictionary: Zero-Shot Topic Classification for  Low-Resource Languages with Application to Luxembourgish (https://arxiv.org/abs/2404.03912)
Comments: 3rd Annual Meeting of the ELRA/ISCA Special Interest Group on Under-resourced Languages (SIGUL 2024)

- **What's New**: 이 연구는 사전을 데이터 소스로 사용하여 저자원 언어에서 영역 별 분류(ZSC)를 수행하는 새로운 접근 방식을 제안합니다. 특히, 룩셈부르크어(Luxembourgish)에 초점을 맞추어, 자연어추론(Natural Language Inference, NLI) 데이터셋을 사용하는 기존 방법 대신 사전 기반 데이터셋을 구축하여 ZSC 작업에 활용합니다. 연구 결과, 사전 기반 데이터셋을 사용한 모델이 NLI 기반 접근 방식을 사용한 모델보다 성능이 우수한 것으로 나타났습니다.



### SAAS: Solving Ability Amplification Strategy for Enhanced Mathematical  Reasoning in Large Language Models (https://arxiv.org/abs/2404.03887)
- **What's New**: 이 연구에서는 대규모 언어 모델(LLM)의 수학적 추론 및 문제 해결 능력을 향상시키기 위한 새로운 학습 접근 방식을 제시합니다. 특히, 사고의 연쇄(Chain-of-Thought, CoT)와 사고의 프로그램(Program-of-Thought, PoT) 학습을 통합하며, 이를 순차적으로 전환하는 새로운 전략인 SAAS(Solving Ability Amplification Strategy)를 도입하였습니다. 이 접근 방식은 문제 해결 능력 강화에 중점을 두어, CoT 학습에서 확립된 논리적 기술을 기반으로 PoT 학습을 강화합니다.

- **Technical Details**: SAAS는 먼저 CoT 학습을 통해 문제를 해결하기 위한 초기 사고 과정을 구축하고, 이후 PoT 학습으로 전환하여 이러한 사고 과정을 프로그램 코드로 표현하고 코드 인터프리터(code interpreter)를 사용하여 계산 단계를 처리합니다. 이렇게 함으로써, LLM이 더 정확하고 효율적으로 수학적 문제를 해결할 수 있도록 지원합니다.

- **Performance Highlights**: SAAS는 여러 벤치마크를 사용한 광범위한 성능 비교를 통해 최첨단(state-of-the-art, SOTA) 성능을 달성했습니다. 이는 SAAS가 수학적 추론 능력뿐만 아니라 복잡한 문제 해결 능력을 향상시키는데 효과적임을 강조합니다.



### A Bi-consolidating Model for Joint Relational Triple Extraction (https://arxiv.org/abs/2404.03881)
- **What's New**: 이 논문에서는 복잡한 의미 중첩 문제(Semantic Overlapping Problem)를 처리하기 위해 새로운 이중 강화 모델(Bi-consolidating model)을 제안합니다. 이 모델은 로컬(지역) 및 글로벌(전역) 의미론적 특성을 동시에 강화함으로써 개체 식별 및 관계 유형 분류 성능을 향상시키는 것을 목표로 합니다.

- **Technical Details**: 이 모델은 지역 강화 구성 요소(Local Consolidation Component)와 글로벌 강화 구성 요소(Global Consolidation Component)로 구성됩니다. 첫 번째 구성 요소는 인접 지역으로부터 가능한 트리플 표현의 의미 정보를 강화하고 이웃의 잡음을 완화하기 위해 픽셀 차이 합성곱(Pixel Difference Convolution)을 사용합니다. 두 번째 구성 요소는 채널 주의(Channel Attention) 및 공간 주의(Spatial Attention)를 기반으로 트리플 표현을 강화하여 문장 내에서 원격 의미 의존성을 학습하는 데 유리합니다.

- **Performance Highlights**: 제안된 모델은 여러 벤치마크 데이터셋에서 기존 최고 모델들을 일관되게 능가하고 최첨단(State-of-the-art) 결과를 달성했습니다. 이는 모델이 두 개체 간의 관계뿐만 아니라 세부 의미론적 관계를 파악하는 데 매우 효과적임을 보여줍니다. 또한 다양한 분석 실험을 통해 모델의 효과성을 입증하고 관계 삼중 추출(Relational Triple Extraction) 및 기타 NLP 작업에 대한 연구 동기를 제공합니다.



### Extract, Define, Canonicalize: An LLM-based Framework for Knowledge  Graph Construction (https://arxiv.org/abs/2404.03868)
Comments: 15 pages, 2 figures

- **What's New**: 이 연구에서는 입력 텍스트에서 지식 그래프 생성(Knowledge Graph Creation, KGC)을 위한 자동화 방법에 관심을 가집니다. 최근 대규모 언어 모델(Large Language Models, LLMs)의 발전을 기반으로 텍스트에서 자동으로 지식 그래프를 생성하는 새로운 구조적 접근 방식인 '추출-정의-규격화(Extract-Define-Canonicalize, EDC)'를 제안합니다. EDC 프레임워크는 개방형 정보 추출, 스키마 정의, 사후 규격화의 세 단계로 구성됩니다.

- **Technical Details**: EDC 접근 방식은 LLM을 활용하여 입력 텍스트에서 엔티티-관계 트리플릿을 자유롭게 추출(Open Information Extraction)하고, 추출된 트리플릿으로부터 스키마를 자동으로 생성하거나, 이미 정의된 스키마를 활용하여 내용을 규격화합니다(Schema Definition and Schema Canonicalization). 또한, LLM의 성능을 향상시키기 위해 입력 텍스트와 관련된 스키마 요소를 검색하는 훈련된 '스키마 검색기(Schema Retriever)' 컴포넌트를 도입합니다. 이는 검색 보완 생성(retrieval-augmented generation-like) 방식으로 LLM의 추출 성능을 향상시킵니다.

- **Performance Highlights**: 세 개의 KGC 벤치마크 데이터셋에서 EDC는 최신 방법론들에 비해 더 높은 품질의 지식 그래프를 자동으로 생성할 수 있음을 입증하였고, 스키마 검색기의 사용은 EDC의 성능을 일관되게 크게 향상시키는 것으로 나타났습니다. 또한, EDC는 규모가 크거나 미리 정의된 스키마 없이도 우수한 품질의 지식 그래프를 생성할 수 있는 유연성을 제공합니다.



### FFN-SkipLLM: A Hidden Gem for Autoregressive Decoding with Adaptive Feed  Forward Skipping (https://arxiv.org/abs/2404.03865)
Comments: arXiv admin note: text overlap with arXiv:2310.01382

- **What's New**: 이 연구는 첫 번째로 KV 캐시(Key-Value Cache) 문제를 피할 수 있는 FFN-SkipLLM을 제안합니다. 이는 입력 적응적인 피드-포워드 (Feed-Forward) 스킵 전략을 통해 대규모 언어 모델의 계산 부담을 줄이고, 지식 집약적 태스크에서의 성능 저하를 최소화하는 새로운 접근 방식입니다.

- **Technical Details**: FFN-SkipLLM은 LLM의 FFN 블록을 건너뛰는 미세 조정된 (Fine-grained) 스킵 전략을 사용하여, 계산을 절약하면서도 성능을 유지할 수 있습니다. 이 모델은 코사인 유사성 (Cosine Similarity) 메트릭을 사용하여 FFN 블록의 포화 상태를 감지하고, 토큰의 25-30% 정도를 스킵할 수 있습니다.

- **Performance Highlights**: FFN-SkipLLM은 다양한 벤치마크 (예: MT-Bench, Factoid-QA 및 다양한 길이의 텍스트 요약)에서 광범위한 실험을 통해 소소한 성능 변화를 보이며 25-30%의 FFN 블록을 스킵하였습니다. 이 방법은 또한 할루시네이션(Hallucination)과 토큰 생성 충돌(Token Generation Collapse)을 감소시키는 데에 효과적입니다.



### Verifiable by Design: Aligning Language Models to Quote from  Pre-Training Data (https://arxiv.org/abs/2404.03862)
- **What's New**: 이 연구에서는 언어 모델(Language Models, LM)의 신뢰성을 높이기 위해 'Quote-Tuning'이라는 새로운 방법을 제안했습니다. 이 방법은 기존의 인용(citation) 기반 검증 방식의 한계를 극복하고, 사전 훈련 데이터(pre-training data)에서 검증 가능한 정확한 인용문을 직접 제공함으로써 검증 과정을 단순화합니다.

- **Technical Details**: Quote-Tuning은 대규모 말뭉치에서 인용문을 정량화하고 멤버십 추론 도구(membership inference tools)를 이용하여 효율적으로 인용을 평가합니다. 인용량을 암시적 보상 신호(implicit reward signal)로 사용하여, 인간의 주석 없이 인용에 대한 합성 선호 데이터셋(synthetic preference dataset)을 구축합니다. 그 후, 선호 최적화 알고리즘(preference optimization algorithms)을 사용하여 모델이 인용을 포함하도록 조정합니다.

- **Performance Highlights**: 실험 결과, Quote-Tuning은 고품질 사전 훈련 문서에서 글자 그대로 인용되는 언어 모델 생성물의 비율을 기존 모델에 비해 55%에서 130% 증가시켰으며, 응답의 질도 유지했습니다. 또한, Quote-Tuning은 도메인 외 데이터(out-of-domain data)에 대한 인용을 일반화하는 데에도 효과적이며, 다양한 작업에 적용 가능하고 진실성(truthfulness)에 대한 추가적인 이점을 제공합니다.



### CantTalkAboutThis: Aligning Language Models to Stay on Topic in  Dialogues (https://arxiv.org/abs/2404.03820)
- **What's New**: 언어 모델이 대화 중 주제의 관련성을 유지하도록 돕는 데이터셋인 'CantTalkAboutThis'를 소개합니다. 이 데이터셋은 인공적인 대화(dialogues)를 포함하며, 이는 다양한 도메인에서 다양한 대화 주제로 구성되어 있습니다. 각 대화는 사전에 정의된 주제에서 챗봇을 일탈하게 만드는 파생 전환(distractor turns)을 포함하고 있습니다.

- **Technical Details**: 이 데이터셋을 통해 언어 모델을 미세조정(fine-tuning)하면 주제 일관성을 유지하고 복잡한 대화 지침을 따르는 능력이 일반적인 지침 조정된 LLMs(Language Learning Models)인 GPT-4-turbo 및 Mixtral-Instruct보다 개선됩니다. 특히, 이 데이터 셋은 세 단계의 프롬프트 기반 접근 방식을 사용하여 생성되었으며, 다양한 시나리오에 대한 추종 주제 프롬프트를 생성하고, 이러한 주제 지시에 따라 대화를 생성하는 'dialogue inpainting' 기술을 사용하며, 주제 추종을 실험하기 위해 이 대화에 파생자를 통합합니다.

- **Performance Highlights**: 모델들은 CantTalkAboutThis 데이터셋에서 미세조정을 거친 후 복잡한 대화 지침을 따르는 성능이 현저히 향상되었습니다. 초기에는 주제를 벗어난 파생자와 상호 작용하는 경향이 있는 일반적인 LLMs보다 더 우수한 성능을 보입니다. 또한, 이 데이터로 훈련된 모델은 지시사항을 따르는 전반적인 능력이 개선되고, 제로샷(zero-shot) 안전 정렬에서 LlamaGuard와 유사한 효과를 보입니다.



### PRobELM: Plausibility Ranking Evaluation for Language Models (https://arxiv.org/abs/2404.03818)
- **What's New**: 이 논문에서는 언어 모델(Language Models)의 가능성 평가를 위한 새로운 벤치마크인 PRobELM (Plausibility Ranking Evaluation for Language Models)을 소개합니다. PRobELM은 언어 모델이 세계 지식을 활용하여 가능성 있는 시나리오를 덜 가능성 있는 대안보다 우선 순위를 두는 능력을 평가합니다. 이 벤치마크는 특히 문헌기반 발견(literature-based discovery)과 같은 응용 분야에서 중요한 정보를 식별하는데 중점을 둡니다.

- **Technical Details**: PRobELM은 Wikidata 수정 이력에서 수집한 데이터셋을 기반으로 구성되었습니다. 이 벤치마크는 다양한 프롬프트 유형을 통한 평가를 지원하며, 성명(statement), 텍스트 완성(text completion), 질문 응답(question-answering) 등을 포함합니다. 또한, 언어 모델의 아키텍처, 매개변수 크기, 최신 세계 지식 업데이트의 최근성에 따라 다양한 평가가 이루어졌습니다.

- **Performance Highlights**: 실험 결과, 언어 모델이 사실 정확성에서 높은 성능을 나타내는 것이 반드시 가능성 평가에서 우수한 결과를 보장하지 않음을 발견했습니다. 더 큰 모델은 일반적으로 더 나은 가능성 추론을 보여주지만, 이는 모델 가족에 따라 다를 수 있습니다. 또한, 최신 데이터를 사용하는 모델이 때로는 더 오래된 데이터를 사용하는 모델보다 성능이 떨어질 수 있으며, 이는 모델 아키텍처와 훈련 방법론도 가능성 추론에 영향을 미칠 수 있음을 시사합니다.



### SHROOM-INDElab at SemEval-2024 Task 6: Zero- and Few-Shot LLM-Based  Classification for Hallucination Detection (https://arxiv.org/abs/2404.03732)
Comments: 6 pages, 6 figures, 4 tables, camera-ready copy, accepted to the 18th International Workshop on Semantic Evaluation (SemEval-2024), for associated code and data see this https URL

- **What's New**: 암스테르담 대학교(University of Amsterdam) 인텔리전트 데이터 엔지니어링 랩(Intelligent Data Engineering Lab) 팀은 SemEval-2024 Task 6에 참여하여 SHROOM-INDElab 시스템을 새롭게 개발했습니다. 이 시스템은 Large Language Models(LLMs)와 인-컨텍스트 학습(in-context learning), 프롬프트 프로그래밍(prompt programming)을 이용가능하고, 몇 가지 예시를 자동 생성하는 few-shot 접근법을 통하여 확장되었습니다. 이를 통해 헤어로내이션(hallucination) 탐지 분류기를 구축하여 높은 성과를 보였습니다.

- **Technical Details**: SHROOM-INDElab 시스템은 헤어로내이션을 탐지하기 위한 분류기를 개발하기 위해 LLM을 사용하는 프롬프트 엔지니어링 기법에 기반을 두고 있습니다. 이 시스템은 zero-shot 방식으로 예제를 제공하지 않고 특정 작업, 역할, 개념 정의를 사용하여 분류를 수행합니다. few-shot 분류기에서는 자동 생성된 예제를 사용하며, Spearman's correlation coefficient(스피어만 상관 계수)와 정확도 사이의 비교를 통해 성능을 평가합니다.

- **Performance Highlights**: 이 시스템은 모델-무관 트랙(model-agnostic track)에서 네 번째로 높은 성능을, 모델-인지 트랙(model-aware track)에서는 여섯 번째로 높은 성능을 달성했습니다. SHROOM-INDElab은 zero-shot 접근 방식이 자동 생성된 예제를 사용하는 few-shot 접근 방식보다 더 높은 정확도를 보였습니다. 또한, 평가 결과는 크라우드 소싱된 인간 평가자들(crowd-sourced human labellers)의 분류 결정과 일치하는 것으로 나타났습니다.



### Watermark-based Detection and Attribution of AI-Generated Conten (https://arxiv.org/abs/2404.04254)
- **What's New**: 이 연구는 인공 지능(AI) 생성 콘텐츠의 워터마크 기반 사용자 인식 감지 및 추적(attribution)에 대한 첫 번째 체계적 연구를 제공합니다. 이전 문헌은 주로 사용자 비특정 감지에 집중했지만, 이 연구는 생성된 콘텐츠를 생성한 GenAI 서비스의 등록된 사용자에게 추적하는 것을 목표로 합니다.

- **Technical Details**: 이 연구에서는 워터마크 기반 감지 및 추적의 이론적 및 알고리즘적 측면을 분석합니다. 특히, 이론적으로는 진정한 감지율(TDR), 거짓 감지율(FDR), 그리고 진정한 추적율(TAR) 같은 주요 평가 메트릭을 정의하고 사용합니다. 알고리즘 측면에서는 사용자에게 서로 다른 워터마크를 선택하여 추적 성능을 한층 강화하는 효율적인 알고리즘을 개발하였습니다.

- **Performance Highlights**: 실험적 평가는 Stable Diffusion, Midjourney, 그리고 DALL-E 2를 포함하는 세 가지 GenAI 모델에 대해 HiDDeN(현재 가장 진보된 학습 기반 워터마킹 방법)을 사용하여 수행되었습니다. 결론적으로, AI 생성 이미지가 후처리되지 않은 경우 감지 및 추적은 매우 정확하며(TDR/TAR은 거의 1에 가까우며 FDR은 거의 0), 일반적인 후처리가 적용되었을 때도 여전히 정확합니다.



### Who Evaluates the Evaluations? Objectively Scoring Text-to-Image Prompt  Coherence Metrics with T2IScoreScore (TS2) (https://arxiv.org/abs/2404.04251)
Comments: 15 pages main, 9 pages appendices, 16 figures, 3 tables

- **What's New**: 이 연구에서는 텍스트-이미지 (T2I) 모델의 발전에 따라 생성된 이미지가 주어진 프롬프트와 얼마나 의미론적으로 일치하는지 판단하는 새로운 벤치마크, T2IScoreScore (TS2)를 소개하고 있습니다. 이 벤치마크는 프롬프트와 점점 오류가 많은 이미지들을 포함하는 의미 오류 그래프 (semantic error graphs)로 구성되어 있으며, 다양한 프롬프트 충실도(metric)를 보다 엄격하게 비교하고 평가할 수 있게 합니다.

- **Technical Details**: TS2는 주어진 메트릭이 이미지들을 그들의 오류 수가 증가함에 따라 올바르게 순서화할 수 있는지 (Ord.), 그리고 단일 프롬프트 관련 의미 차이를 기반으로 이미지 세트를 신뢰성 있게 분리할 수 있는지 (Sep.)를 판단하기 위해 스피어만의 상관계수 (Spearman’s correlation)와 콜모고로프–스미르노프 통계 (Kolmogorov–Smirnov statistic)를 사용합니다. 연구 결과에 따르면, 복잡한 비전-언어 모델 (Vision-Language Models, VLMs) 기반의 메트릭들이 간단한 특징 기반 메트릭 (예: CLIPScore)에 비해 현저하게 우수한 성능을 보이지 않는다는 놀라운 결과를 보여줍니다.

- **Performance Highlights**: TS2 벤치마크를 사용한 평가에서는 간단한 특징 공간 메트릭인 CLIPScore가 복잡한 VLM 기반 메트릭들과 비교해도 객관적 정확성 면에서 비슷하거나 우수한 성능을 보이는 것으로 나타났습니다. 이는 T2I 평가 커뮤니티가 휴먼 선호 상관성의 이점을 보존하면서도 더 강력한 기준으로 메트릭의 발전을 이끄는 데 TS2가 중요한 역할을 할 수 있음을 시사합니다.



### Physical Property Understanding from Language-Embedded Feature Fields (https://arxiv.org/abs/2404.04242)
Comments: CVPR 2024. Project page (with code): this https URL

- **What's New**: 이 논문에서는 이미지 집합을 사용하여 객체의 물리적 특성을 밀도 있게 예측하는 새로운 접근 방식인 NeRF2Physics를 제시합니다. 이 방법은 대규모 언어 모델(Large Language Models, LLMs)을 활용하여 각 객체에 대한 후보 물질을 제안하고, 제로샷 커널 회귀 접근 방식을 사용하여 각 3D 점의 물리적 특성을 평가합니다.

- **Technical Details**: NeRF2Physics는 물리적 특성 필드(Physical Property Field)를 생성하여 각 장면 내 점에서 물리적 특성 추정을 요청할 수 있습니다. 이 접근 방식은 언어-장면 임베딩(Language-Vision Embeddings)과 언어 모델을 활용하여 물리적 특성을 인지하고 추론하는 인간의 방식에서 영감을 받았습니다. 각 점에 대한 후보 물질을 제안하고, CLIP 기반의 제로샷 검색을 사용하여 물질의 물리적 성질을 추론합니다.

- **Performance Highlights**: NeRF2Physics는 감독 없는 설정에서 다양한 물리적 특성을 예측할 수 있는 능력을 시연했습니다. ABO 데이터셋과 실제 객체의 수동 측정 데이터셋을 사용한 질량 추정 작업에서 제로샷 및 감독된 베이스라인을 능가하는 성능을 보였습니다. 이는 확장된 세계의 모든 객체에 적용가능하며, 주석 없는 상태에서 정확한 추정을 자랑합니다.



### player2vec: A Language Modeling Approach to Understand Player Behavior  in Games (https://arxiv.org/abs/2404.04234)
- **What's New**: 이 연구에서는 자연 언어 처리(Natural Language Processing, NLP) 분야에서 사용되는 장거리 트랜스포머(Long-range Transformer) 모델을 게임 플레이어의 행동 데이터에 적용하는 새로운 방법을 제안합니다. 이는 모바일 게임 분야에서 플레이어의 행동을 이해하고 대규모의 비지도 학습을 통해 사용자 표현을 학습하는 데 중점을 두고 있으며, 게임 내 이벤트를 문장 속의 단어처럼 처리하여 자기지도 학습(self-supervised learning) 방식으로 플레이어 표현을 학습할 수 있게 합니다.

- **Technical Details**: 대규모의 모바일 게임 플랫폼에서 수집된 플레이어 행동 데이터를 사용하여, Longformer라는 트랜스포머 아키텍처를 활용했습니다. 이는 마스크 언어 모델링(Masked Language Modeling, MLM) 목표를 사용하여 트레이닝되며, 이는 시퀀스 모델링 방법에서 효과적임이 입증되었습니다. 또한 데이터 전처리 파이프라인을 통해 가공되어 이벤트를 풍부하고 간결한 텍스트 시퀀스로 변환하여 모델이 적절하게 소비할 수 있도록 설계되었습니다.

- **Performance Highlights**: 제안된 접근 방식은 게임 행동 이벤트의 분포를 적합하게 모델링하는 것을 실험적으로 보여주었습니다. 특히, 언어 모델링 메트릭스(language modeling metrics)를 평가하여 효과를 입증했으며, 학습된 임베딩 공간(embedding space)의 구조를 질적으로 분석하여 행동 패턴에 대한 통찰력을 제공하고 다운스트림 애플리케이션(downstream applications)에 유용한 정보를 생성하는 데 기여할 수 있음을 보여주었습니다.



### Dwell in the Beginning: How Language Models Embed Long Documents for  Dense Retrieva (https://arxiv.org/abs/2404.04163)
- **What's New**: 이 연구는 웹 문서 검색 맥락에서 텍스트 표현 학습을 위한 Transformer 기반 모델에서의 위치적 편향의 존재를 조사합니다. 이전 연구에서 인과적 언어 모델들의 입력 시퀀스 중간에서 정보 손실이 발생함을 보여준 것을 기반으로, 표현 학습 분야로 연구를 확장하였습니다.

- **Technical Details**: 연구는 인코더-디코더(encoder-decoder) 모델의 다양한 학습 단계에서 위치적 편향을 조사합니다. 이에는 언어 모델 사전 학습(language model pre-training), 대조적 사전 학습(contrastive pre-training), 그리고 대조적 미세 조정(contrastive fine-tuning)이 포함됩니다. MS-MARCO 문서 컬렉션을 사용한 실험을 통해 모델은 대조적 사전 학습 후 입력의 초기 내용을 더 잘 포착하는 임베딩을 생성하며, 미세 조정은 이 효과를 더욱 심화시킵니다.

- **Performance Highlights**: 대조적 사전 학습 후 모델은 입력의 초기 내용을 더 잘 이해할 수 있는 임베딩을 생성했으며, 고급 조정(fine-tuning)은 이러한 편향을 더욱 강화한 결과를 나타냈습니다.



### No "Zero-Shot" Without Exponential Data: Pretraining Concept Frequency  Determines Multimodal Model Performanc (https://arxiv.org/abs/2404.04125)
Comments: Extended version of the short paper accepted at DPFM, ICLR'24

- **What's New**: 이 연구에서는 다중모드 모델들이 '제로-샷' 평가에서 보여주는 인상적인 성능이 사실상 얼마나 의미 있는지에 대해 질문합니다. 특히, CLIP이나 Stable-Diffusion과 같은 모델들이 사전 훈련 데이터셋에서 얼마나 자주 등장하는지에 따라 '제로-샷' 평가 성능이 크게 달라지는지를 분석합니다. 또한, 이 연구는 새로운 'Let it Wag!' 벤치마크를 도입하여 다양한 모델들의 성능을 평가합니다.

- **Technical Details**: 연구팀은 34개의 다양한 모델과 5개의 표준 사전 훈련 데이터셋(CC-3M, CC-12M, YFCC-15M, LAION-400M, LAION-Aesthetics)을 사용해 평가를 수행했습니다. 이 과정에서 300GB 이상의 데이터 아티팩트(Artifact)를 생성하며, 모델이 '제로-샷' 성능을 개선하기 위해 필요한 데이터의 양이 지수적으로 증가함을 발견했습니다. 이 연구는 스케일링 효율성이 낮은 로그-선형(Log-linear) 추세를 보여주며, 사전 훈련 데이터셋과 평가 데이터셋 간의 샘플 수준 유사성을 제어한 결과에서도 이러한 추세가 일관되게 나타났습니다.

- **Performance Highlights**: 다양한 프롬프팅 전략과 평가 메트릭을 사용하여 실험을 확장한 결과, 개념 빈도와 '제로-샷' 성능 사이에 강력한 로그-선형 관계가 일관되게 유지됨을 확인하였습니다. 심지어 순수 합성 데이터 분포에서의 테스트에서도 이러한 추세가 지속되었습니다. 또한, 이미지와 텍스트 도메인에서 독립적으로 파생된 개념에서도 로그-선형 추세가 확인되었습니다. 이러한 발견은 다중모드 모델의 '제로-샷' 일반화 능력에 대한 이해를 증진시킬 것입니다.



### Large language models as oracles for instantiating ontologies with  domain-specific knowledg (https://arxiv.org/abs/2404.04108)
- **What's New**: 이 연구는 대규모 언어 모델 (LLM: Large Language Model)을 활용하여 온톨로지를 자동으로 인스턴스화하는 새로운 방법, KGFiller를 제안합니다. 이 시스템은 도메인에 구애받지 않고 다양한 온톨로지에 적용할 수 있으며, 기존 데이터 세트에 의존하지 않고 LLM을 데이터 소스로 사용하여 개념, 속성 및 관계의 인스턴스를 생성합니다.

- **Technical Details**: KGFiller는 초기 스키마와 쿼리 템플릿을 사용하여 LLM에 다수의 쿼리를 수행하고, 클래스와 속성의 인스턴스를 생성하여 온톨로지를 자동으로 채웁니다. 이 방법은 도메인 특화 지식을 온톨로지에 효과적으로 통합할 수 있도록 하며, 생성된 인스턴스는 전문가들이 조정하거나 보완할 수 있습니다. 이 접근 방식은 도메인 독립적이고, 증분적이며 (Incremental), 다양한 LLM에 적용 가능합니다.

- **Performance Highlights**: 영양학 도메인에서의 케이스 스터디를 통해 KGFiller를 구현하고 다양한 LLM을 활용한 온톨로지의 품질을 평가했습니다. 평가는 LLM으로부터 생성된 인스턴스의 의미 있고 정확한 배치를 검토하여 수행되었습니다. 이 연구는 다양한 LLM을 이용하여 온톨로지를 더 효율적이고 정확하게 생성하는 방법을 선보이며, LLM을 이용한 온톨로지 채움이 효과적임을 보여줍니다.



### Robust Preference Optimization with Provable Noise Tolerance for LLMs (https://arxiv.org/abs/2404.04102)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLM: Large Language Models)의 선호도 맞춤을 위한 새로운 방법, 즉 ROPO(RObust Preference Optimization)를 제안합니다. ROPO는 라벨의 불확실성이 높은 응답 쌍에 대해 보수적인 그라디언트 가중치를 동적으로 할당함으로써, 노이즈가 있는 데이터에서도 선호도 맞춤을 효과적으로 수행할 수 있습니다. 이는 기존의 순위 기반 방법들이 가지고 있던 노이즈에 대한 취약점을 해결하는 것을 목표로 하고 있습니다.

- **Technical Details**: ROPO는 로그 가능성 (log-likelihood) 마진을 기반으로 응답 쌍의 불확실성을 평가하고, 이를 통해 노이즈가 있는 샘플의 그라디언트 크기를 억제합니다. 이 방법은 그라디언트 방향이 노이즈의 존재 유무와 노이즈의 비율에 독립적이라는 점에서 독창적입니다. ROPO는 가벼운 계산으로 안정적이며, 노이즈 비율을 알 필요 없이 최적의 해를 제공할 수 있습니다.

- **Performance Highlights**: ROPO는 2.8B에서 13B에 이르는 다양한 크기의 네 가지 기본 모델을 사용하여 진행된 세 가지 개방형 텍스트 생성 작업에서 기존의 순위 기반 방법들에 비해 크게 우수한 성능을 보였습니다. 특히, 노이즈 데이터에서도 기대 리스크(risk)의 그라디언트 방향이 일정하다는 이론적 보장 하에 성능이 향상되었습니다.



### VoicePilot: Harnessing LLMs as Speech Interfaces for Physically  Assistive Robots (https://arxiv.org/abs/2404.04066)
- **What's New**: 본 연구에서는 신체 보조 로봇을 위한 언어 모델 기반 음성 인터페이스를 구현하고, 노인 독립 생활 시설에서 11명의 노인 대상으로 사용자 연구를 진행하여 그 효과를 검증하였습니다. 이는 실제 로봇과 대규모 언어 모델(Large Language Models, LLMs)을 결합하여 사용자의 자연스러운 의사소통을 가능하게 하는 체계를 제시합니다.

- **Technical Details**: 연구진은 GPT-3.5 Turbo 같은 기존 대형 언어 모델을 사용하여 Obi라는 상용 급식 로봇과 통합하고, 세 번의 반복적인 테스트를 통해 프레임워크를 개선했습니다. 최종 프레임워크는 환경 설명(Environment Description), 로봇 기능(Robot Functions) 등 9개의 구성 요소를 포함하고 있습니다. 또한, 최종 사용자 테스트를 통해 양적 및 질적 데이터를 수집하고, 언어 모델을 식사 보조 등 실생활과 밀접한 상황에 적용할 수 있는 디자인 지침을 제공합니다.

- **Performance Highlights**: 이 시스템은 실제 노인 사용자 그룹을 대상으로 한 평가에서 긍정적인 반응을 얻었으며, 보다 독립적인 일상생활 수행을 가능하게 하는 데 기여했습니다. 인터페이스가 사용자의 고유한 요구 사항과 선호도를 반영할 수 있도록 설계된 점이 높이 평가되었습니다.



### Transformers for molecular property prediction: Lessons learned from the  past five years (https://arxiv.org/abs/2404.03969)
- **What's New**: 본 리뷰는 분자 특성 예측(Molecular Property Prediction, MPP)을 위한 트랜스포머(Transformer) 모델 사용의 현황을 제공합니다. 트랜스포머 모델은 원래 자연언어 처리(Natural Language Processing, NLP)를 위해 개발되었으나, 최근에는 생명 과학 분야에서도 중요한 역할을 하고 있습니다.

- **Technical Details**: 이 리뷰는 트랜스포머 모델의 아키텍처와 변형, MPP를 위한 적용 방식을 상세히 설명합니다. 또한, 학습 및 미세 조정 과정에서 고려해야 할 데이터 크기, 아키텍처 선택, 사전 학습 목표 등 중요한 질문들을 탐구합니다.

- **Performance Highlights**: 언어 영역에서 주로 사용되는 SMILES 언어를 사용한 트랜스포머 모델은 기존의 기계 학습 및 딥러닝 모델과 유사한 성능을 보여줍니다. 특히, 사전 학습 데이터 선택 방법은 모델의 일반화 능력을 향상시키는데 도움을 줄 수 있습니다.



### Mitigating Heterogeneity in Federated Multimodal Learning with  Biomedical Vision-Language Pre-training (https://arxiv.org/abs/2404.03854)
- **What's New**: FedRGB (Federated distributional Robust Guidance-Based) 학습 프레임워크는 의료 분야에서의 비전-언어 사전 학습(VLP)을 위한 연방 학습(FL)에 데이터 이질성 문제를 해결하기 위해 제안되었습니다. 이 프레임워크는 로컬 데이터의 이질성으로 인한 특징 왜곡(distortions)을 감소시키고, 편향되지 않은 크로스-모달 정렬(cross-modal alignment)을 학습하기 위해 가이던스 기반 로컬 트레이닝(guidance-based local training) 및 분포 기반(min-max optimization) 최적화 기법을 활용합니다.

- **Technical Details**: FedRGB는 특히 로컬 클라이언트 트레이닝에서 발생할 수 있는 표현 공간의 왜곡을 완화하기 위해 교사 정렬 모듈(teacher alignment module)을 도입하였습니다. 이 모듈은 각 클라이언트의 편향 없는 크로스-모달 정렬을 제공하고, 분산적으로 강인한 최적화(DRO, Distributionally Robust Optimization) 기반 알고리즘을 사용하여 특징 인코더(feature encoder)의 최악의 경우 왜곡을 경감합니다. 이 방법은 다양한 하위 작업, 예를 들어 이미지-텍스트 검색(image-text retrieval), 분류(classification), 세분화(segmentation)에 효과적인 멀티모달 표현 학습을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, FedRGB는 데이터 이질성이 있는 데도 불구하고 의료 VLP 영역에서 효율적인 연방 멀티모달 학습을 촉진하는 것으로 나타났습니다. 특히, 이 프레임워크는 로컬 클라이언트의 데이터에서 해당 모델이 겪은 편향과 정렬 문제를 완화하는 데 성공하였으며, 다양한 벤치마크 데이터셋에서의 실험을 통해 그 효과가 검증되었습니다.



### An Investigation into Misuse of Java Security APIs by Large Language  Models (https://arxiv.org/abs/2404.03823)
Comments: This paper has been accepted by ACM ASIACCS 2024

- **What's New**: 이 논문은 보안 API(Application Programming Interfaces) 사용과 관련된 코드 생성에서 대규모 언어 모델(Large Language Models, LLMs)의 신뢰성을 분석합니다. 특히, ChatGPT가 Java 보안 API를 사용할 때 얼마나 빈번하게 보안 API 오용(misuse)을 생성하는지에 초점을 맞추어 연구하였습니다.

- **Technical Details**: 연구자들은 Java에서 널리 사용되는 5가지 보안 API에 대한 48개의 프로그래밍 작업을 설정하고, ChatGPT가 생성한 코드에서 보안 API의 오용 사례를 자동화된 도구 및 수동 접근 방식을 통해 감지했습니다. 연구팀은 CryptoGuard 도구를 사용하여 암호화 API 오용을 식별하고, 다른 API에 대해서는 수동 검토를 수행했습니다.

- **Performance Highlights**: 연구 결과, ChatGPT에 의해 생성된 코드 인스턴스의 약 70%가 보안 API 오용을 포함하고 있었으며, 이 중 30개 작업당 100%에 이르는 작업도 반은 넘는 것으로 나타났습니다. 또한, 연구자들은 20가지의 구분된 오용 유형을 식별했습니다. 이러한 결과는 개발자들이 소프트웨어 보안 맥락에서 LLM을 사용하는 경우, 보안 문제의 인식을 높이고 ChatGPT의 코드 생성 안전성을 개선하기 위한 추가 연구의 필요성을 강조합니다.



### GenQREnsemble: Zero-Shot LLM Ensemble Prompting for Generative Query  Reformulation (https://arxiv.org/abs/2404.03746)
Comments: Accepted at ECIR 2024

- **What's New**: 이 연구에서는 새로운 제로샷(Zero-shot) 쿼리 리포뮬레이션(Query Reformulation, QR) 기법을 제안합니다. GenQREnsemble과 GenQREnsembleRF는 다중 제로샷 지시문을 활용하여 사용자의 검색 쿼리를 보다 효과적으로 개선하는 방법입니다. 특히, GenQREnsembleRF는 유사 관련 피드백(Pseudo-Relevance Feedback, PRF)을 포함하여 검색 결과를 더욱 향상시키는 접근 방식을 도입하였습니다.

- **Technical Details**: GenQREnsemble은 N개의 다양한 QR 지시문을 사용하여 LLM(Large Language Models)에서 키워드를 생성하고, 이를 원본 쿼리에 추가하여 쿼리를 재구성합니다. GenQREnsembleRF는 검색 후 설정에서 문서 피드백을 추가로 고려하여 쿼리 재구성을 수행합니다. 이 기법들은 IR(Information Retrieval) 벤치마크에서 상태 최신 기술 대비 성능을 크게 향상시키며, 특히 MSMarco Passage Ranking 작업에서 상당한 개선을 보여줍니다.

- **Performance Highlights**: GenQREnsemble은 이전의 제로샷 상태 최신 기술보다 최대 18%의 상대적 nDCG@10(nDCG at rank 10) 및 24%의 MAP(Mean Average Precision) 개선을 달성했습니다. 또한, GenQREnsembleRF는 유사 관련 피드백을 사용하여 MRR(Mean Reciprocal Rank)에서 5%의 상대적 이득을, 관련 피드백 문서를 사용할 때 9%의 nDCG@10 개선을 보였습니다.



### Fakes of Varying Shades: How Warning Affects Human Perception and  Engagement Regarding LLM Hallucinations (https://arxiv.org/abs/2404.03745)
- **What's New**: 이 연구는 대규모 언어 모델(Large Language Models, LLMs)이 생성하는 내용 중 정확하지 않거나 허구적인 내용, 즉 '환각(hallucinations)'에 대한 인간의 인식을 조사합니다. 연구는 경고 여부(있음 vs. 없음)와 환각의 정도(진실, 경미한 환각, 심각한 환각)를 시스템적으로 변화시키면서 이들의 상호작용을 검토합니다.

- **Technical Details**: 참가자들(N=419)은 Prolific 플랫폼을 통해 Q/A 형식의 컨텐츠에 대한 정확성을 평가하고, 좋아요(like), 싫어요(dislike), 공유(share)와 같은 사용자 참여 행동을 보였습니다. 연구는 경고가 환각 탐지를 향상시키지만 진실된 내용의 지각된 진실성에는 큰 영향을 미치지 않는 것을 발견했습니다.

- **Performance Highlights**: 사용자는 내용의 진실성을 다음 순서대로 평가했습니다: 진실 > 경미한 환각 > 심각한 환각. 사용자 참여 행동도 이러한 패턴을 반영했습니다. 특히, 경고의 존재는 경미하거나 심각한 환각의 지각된 정확성을 감소시켰지만 '좋아요'나 '공유'에는 미미한 영향을 미쳤습니다. 반면 '싫어요'는 증가하였습니다.



### Direct Nash Optimization: Teaching Language Models to Self-Improve with  General Preferences (https://arxiv.org/abs/2404.03715)
- **What's New**: 이 연구에서는 '직접 내시 최적화(Direct Nash Optimization, DNO)'라는 새로운 알고리즘을 소개하여, 상호대비 학습의 단순성과 일반적 선호 최적화의 이론적 일반성을 결합합니다. DNO는 배치 기반 온-폴리시(on-policy) 알고리즘으로, 회귀 기반의 학습 목표를 사용하여 실행이 간단하고 효율적입니다. 이 연구는 인간의 선호도에 더 잘 맞추기 위해 대형 언어 모델을 개선하는 새로운 접근 방식을 제시합니다.

- **Technical Details**: DNO는 일반적인 선호를 최적화하기 위해 설계된 알고리즘으로, 보상 기반의 접근 방식에서 벗어나 선호도 자체를 직접 최적화합니다. 이 알고리즘은 비교 목적(Contrastive Objectives)의 확장성과 일반 선호 최적화의 이론적 타당성을 결합합니다. DNO는 '예상 승률(Expected Win-Rates)' 개념을 활용하여 일반 선호 함수에 따른 보상을 표현하며, 강화 학습에서 흔히 발생하는 보상 해킹(reward hacking) 문제를 피하기 위해 안정적인 학습 환경을 제공합니다.

- **Performance Highlights**: DNO를 사용하여 훈련된 Orca-2.5 모델은 GPT-4-Turbo와의 비교 평가에서 33%의 승률을 달성하며, 초기 모델 대비 26%의 절대적 승률 상승을 보여 주었습니다. 이는 매개변수 수가 훨씬 많은 모델들보다 우수한 성능을 나타내는 것으로, DNO의 효율성과 효과를 입증합니다.



### Stream of Search (SoS): Learning to Search in Languag (https://arxiv.org/abs/2404.03683)
- **What's New**: 이 논문에서는 언어 모델이 어떻게 검색 작업을 수행할 수 있는지를 보여주며, 이를 위해 'Stream of Search'(SoS)라는 개념을 도입하고 있다. SoS는 검색 과정을 직렬화된 문자열로 표현하며, 다양한 검색 전략을 하나의 통합된 언어로 시스템화한다. 이 연구는 Countdown 게임을 예시로 들어, 언어 모델이 어떻게 다양한 수치 및 연산 조합을 통해 목표 숫자를 도출할 수 있는지를 보여준다.

- **Technical Details**: 이 논문에서 제안하는 SoS 기반 언어 모델은 다양한 휴리스틱 솔버에서 생성된 검색 트레이스로 사전 트레이닝된다. 이 모델은 Advantage-Induced Policy Alignment (APA) 및 Self-Taught Reasoner (STaR)라는 두 가지 정책 개선 방법을 사용하여 미세 조정된다. 이러한 방법들은 모델이 더 정확하게 검색할 수 있도록 최적화하는 데 사용된다.

- **Performance Highlights**: SoS 기반 언어 모델은 트레이닝을 통하여 25% 향상된 검색 정확도를 보여주며, 이는 기존 모델이 최적의 검색 경로만을 예측하는 것에 비해 상당한 개선이다. 또한, 이 모델은 이전에 해결하지 못했던 문제의 36%를 해결하는 데 성공했으며, 사용된 휴리스틱 솔버들로는 해결할 수 없었던 문제들까지 해결할 수 있었다. 따라서 이 연구는 언어 모델이 검색을 통해 문제를 해결하고, 다양한 검색 전략을 자율적으로 사용하여 자가 개선할 수 있음을 보여준다.



### Neural Information Organizing and Processing -- Neural Machines (https://arxiv.org/abs/2404.03676)
- **What's New**: 이 논문에서는 자연 및 인공 신경 시스템을 신경 기계로 통합하여 설명하고 모델링할 수 있는 신경 구조, 과정, 매개 변수 및 특징들에 대한 정보적 합성을 제시합니다. 이를 통해 인공적인 계산 구현의 기술적 제한을 극복하고 자연 신경 시스템의 설명을 보다 관련성 있게 제공합니다.

- **Technical Details**: 일반 정보 매개변수는 신경 시스템의 컴퓨팅 잠재력(global quantitative measure of the neural systems computing potential)을 절대적 및 상대적 신경 파워(absolute and relative neural power)로 제안합니다. 신경 정보의 조직화 및 처리는 비결정적(non-deterministic) 기억, 분절화(fragmentation), 집합화(aggregation)의 여러 단계를 거치는 심층 신경 정보 처리(deep neural information processing)를 통해 이루어집니다.

- **Performance Highlights**: 제안된 신경 기계(neural machine) 모델 유형은 중추적이면서도 주변적 또는 인터페이스 구성 요소를 통합하여, 복잡한 신경 정보를 더 효과적으로 관리하고 처리할 수 있는 구조를 제공합니다. 이는 기술적 제약을 넘어서 자연 신경 시스템의 더 정확한 모델링과 인공 신경 시스템의 향상된 구현을 가능하게 합니다.




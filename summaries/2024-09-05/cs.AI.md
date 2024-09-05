New uploads on arXiv(cs.CL)

### LongCite: Enabling LLMs to Generate Fine-grained Citations in Long-context QA (https://arxiv.org/abs/2409.02897)
- **What's New**: 이 논문은 긴 컨텍스트를 가진 LLM이 신뢰성을 높이기 위한 조치를 취하는 데 중점을 두며, 보다 세분화된 문장 수준의 인용 생성을 가능하게 한다.

- **Technical Details**: 논문에서는 LongBench-Cite라는 자동 벤치마크를 소개하며, 이 벤치마크를 통해 현재 LLM의 긴 문맥 질문 응답 성능을 평가했다. CoF(‘Coarse to Fine’)라는 새로운 파이프라인을 제안하여, 기존 LLM을 사용하여 정확한 문장 수준의 인용을 갖춘 긴 문맥 QA 인스턴스를 자동 생성하였다.

- **Performance Highlights**: LongCite-8B와 LongCite-9B 모델은 LongCite-45k 데이터셋을 이용하여 훈련되었으며, 평가 결과 이 모델들은 GPT-4o보다 6.4%/3.6% 높은 인용 F1 점수를 기록하며, 더욱 세밀한 인용 생성을 성공적으로 수행했다.



### LongLLaVA: Scaling Multi-modal LLMs to 1000 Images Efficiently via Hybrid Architectur (https://arxiv.org/abs/2409.02889)
Comments:
          19 pages, 7 figures, 6 tables

- **What's New**: 이번 논문에서는 멀티모달 대형 언어 모델(Multi-modal Large Language Models, MLLMs)의 긴 맥락 처리 능력을 확장하는 새로운 슈퍼 모델인 LongLLaVA를 소개합니다. 이는 Mamba와 Transformer 블록의 하이브리드 아키텍처를 채택하고, 여러 이미지 간의 시간적 및 공간적 종속성을 고려한 데이터 구축 및 점진적 훈련 전략을 적용한 것입니다.

- **Technical Details**: LongLLaVA는 멀티모달 아키텍처, 데이터 구성 및 훈련 전략의 세 가지 차원에서 종합적으로 최적화된 모델입니다. Mamba-Transformer 하이브리드 아키텍처는 효율적인 이미지 표현 방식을 적용하고, 고유한 데이터 형식을 설계하여 다중 이미지 처리 시 성능 저하를 방지합니다. 훈련 전략은 단일 이미지 정렬, 단일 이미지 지시 튜닝 및 다중 이미지 지시 튜닝의 세 단계로 되어 있어 점진적으로 모델의 다중 모달 긴 맥락을 다루는 능력을 향상시킵니다.

- **Performance Highlights**: LongLLaVA는 다양한 벤치마크에서 경쟁력 있는 결과를 달성하며, 특히 VNBench에서 정보 검색, 이미지 수 세기 및 정렬 작업에서 선두를 보이고 있습니다. 80GB A100 GPU에서 단일 GPU 환경에서도 1,000개의 이미지를 처리할 수 있어 뛰어난 효율성을 보여줍니다.



### Visually Grounded Speech Models for Low-resource Languages and Cognitive Modelling (https://arxiv.org/abs/2409.02865)
Comments:
          PhD Dissertation

- **What's New**: 이 논문은 라벨이 없는 음성과 이미지 쌍에서 학습하는 시각적으로 기초한 음성(VGS) 모델을 탐구합니다. 특히 자원이 부족한 언어와 인간 언어 습득 이해에 대한 응용에 초점을 맞추고 있습니다.

- **Technical Details**: 우리는 이미지 사용하여 음성에서 키워드를 탐지하고 지역화하는 'Visually Prompted Keyword Localisation' 작업을 소개합니다. VGS 모델이 Yoruba와 같은 자원이 부족한 언어에 대한 Few-Shot Learning 상황에서 얼마나 효과적인지 보여줍니다. 또한 VGS 모델에서 상호 배타성 편향(Mutual Exclusivity Bias)을 조사합니다. 단일 언어 VGS 모델은 이 편향을 나타내지만 다국어 사용이 이 VGS 모델의 편향에 영향을 미치지 않는다는 것을 발견했습니다.

- **Performance Highlights**: 본 연구는 VGS 모델이 자원이 부족한 언어와 관련된 적은 데이터 환경에서도 효과적으로 키워드를 지역화할 수 있음을 입증하며, 어린이의 언어 습득 패턴과 유사한 과정을 보여줍니다.



### Historical German Text Normalization Using Type- and Token-Based Language Modeling (https://arxiv.org/abs/2409.02841)
Comments:
          27 pages, 3 figures

- **What's New**: 이 연구에서는 1700년대에서 1900년대 사이의 독일 문학 텍스트를 위한 정규화(normalization) 시스템을 제안합니다.

- **Technical Details**: 제안된 시스템은 Transformer 언어 모델을 사용하는 기계 학습(machine learning) 접근 방식을 채택하며, 인코더-디코더 모델(enc­oder-decoder model)을 사용하여 개별 단어 유형을 정규화(normalize)합니다. 또한, 사전 훈련된 인과 언어 모델(pre-trained causal language model)을 통해 이러한 정규화를 문맥(context) 내에서 조정합니다.

- **Performance Highlights**: 광범위한 평가 결과, 제안된 시스템은 최첨단(state-of-the-art) 정확도를 제공하며, 사전 훈련된 Transformer 대형 언어 모델을 미세 조정한(end-to-end) 문장 기반 정규화 시스템과 견줄 수 있는 성능을 보였습니다. 그러나 역사적 텍스트의 정규화는 모델이 일반화(generalize)하는 데 어려움이 있으며, 고품질의 평행 데이터(parallel data)가 부족하여 여전히 도전 과제가 남아 있습니다.



### R2GQA: Retriever-Reader-Generator Question Answering System to Support Students Understanding Legal Regulations in Higher Education (https://arxiv.org/abs/2409.02840)
- **What's New**: 본 논문에서는 R2GQA 시스템을 제안합니다. 이 시스템은 Retriever-Reader-Generator 기반의 질문-응답 시스템으로, 문서 검색기(Document Retriever), 기계 독해기(Machine Reader), 답변 생성기(Answer Generator)의 세 가지 주요 구성 요소로 이루어져 있습니다. 또한, 베트남 대학 교육 규정에 대한 9,758개의 질문-답변 쌍으로 구성된 ViRHE4QA 데이터셋을 구축하였습니다.

- **Technical Details**: R2GQA 시스템은 문서 검색 모듈이 고급 정보 검색 기술을 활용하여 법적 규정 문서의 컨텍스트를 추출하고, 기계 독해 모듈이 최신 자연어 이해 알고리즘을 이용해 문서를 이해하고 답변을 추출합니다. 마지막으로, 답변 생성기는 추출된 답변을 간결하고 유익한 형태로 합성합니다. 이 시스템은 베트남어로 추상적(answer type: abstractive) 답변을 제공하는 첫 번째 시스템입니다.

- **Performance Highlights**: 우리는 실험을 통해 R2GQA 시스템의 유효성을 입증하였으며, 이는 학생들이 법적 규정을 이해하는 데 있어 큰 도움이 될 것으로 기대합니다. R2GQA 시스템과 ViRHE4QA 데이터셋이 학생들이 복잡한 법적 문서와 규정을 탐색하는 데 크게 기여할 것으로 보이며, 이러한 수단을 통해 학생들이 정보에 입각한 결정을 내리고 제도 정책을 효과적으로 준수할 수 있도록 지원할 것입니다.



### Exploring Sentiment Dynamics and Predictive Behaviors in Cryptocurrency Discussions by Few-Shot Learning with Large Language Models (https://arxiv.org/abs/2409.02836)
- **What's New**: 본 연구는 암호화폐 관련 논의에서 Predictive statements, Hope speech 및 Regret Detection 행동을 분석합니다.

- **Technical Details**: 고급 자연어 처리 기술을 활용하며 'Prediction statements'라는 새로운 분류 체계를 도입하여 댓글을 Predictive Incremental, Predictive Decremental, Predictive Neutral, Non-Predictive로 카테고리화합니다. GPT-4o라는 최첨단 대규모 언어 모델을 사용하여 Cardano, Binance, Matic, Fantom, Ripple 등 5개의 주요 암호화폐에서 감정 역학을 탐색합니다.

- **Performance Highlights**: Matic은 낙관적인 예측을 나타내는 경향이 다른 암호화폐보다 현저히 높다는 분석 결과가 나왔으며, 희망과 후회의 감정 사이에 복잡한 상호작용이 있음을 밝혀냈습니다. 데이터량과 자원 가용성과 관련된 한계를 겪었음에도 투자 행동 및 암호화폐 시장의 감정 트렌드에 대한 귀중한 발견을 보고하였고, 이는 전략적 의사결정 및 미래 연구에 기여할 것입니다.



### CMM-Math: A Chinese Multimodal Math Dataset To Evaluate and Enhance the Mathematics Reasoning of Large Multimodal Models (https://arxiv.org/abs/2409.02834)
- **What's New**: 이 논문에서는 중국어 멀티모달 수학 데이터셋인 CMM-Math를 소개하여, 기존 영어 데이터셋 MATHVISTA, MATH-V와의 차별성을 강조하였습니다. CMM-Math는 초등학교부터 고등학교까지의 12개 학년에서 다양한 문제 유형을 포함하며, 총 28,000개 이상의 샘플을 제공합니다.

- **Technical Details**: CMM-Math 데이터셋은 다양한 문제 유형(단일 선택, 다중 선택, 빈칸 채우기 등)과 시각적 맥락을 포함하고 있습니다. 또한, 멀티모달 수학 모델(Math-LMM)을 제안하며, 기초적인 사전 훈련, 기초적인 미세 조정 및 수학적인 미세 조정을 포함하는 세 단계로 학습합니다.

- **Performance Highlights**: 실험 결과, 제안한 Math-LMM은 SOTA LMM과 비교 시 수학적 추론 성능을 효과적으로 향상시켰으며, MATHVISTA와 MATH-V 데이터셋에서도 좋은 성능을 나타냈습니다. 이는 CMM-Math 데이터셋이 다양한 시각적 맥락과 문제를 통해 LMM의 향상을 도모할 필요성을 강조합니다.



### MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark (https://arxiv.org/abs/2409.02813)
- **What's New**: 이 논문은 MMMU-Pro를 소개합니다. 이는 Massive Multi-discipline Multimodal Understanding and Reasoning (MMMU) 벤치마크의 강력한 버전으로, 멀티모달 모델의 진정한 이해 및 추론 능력을 평가하기 위해 세 가지 단계로 구성된 평가 방식을 사용합니다.

- **Technical Details**: MMMU-Pro는 1) 텍스트 전용 모델이 답할 수 있는 질문 필터링, 2) 후보 옵션 증강, 3) 질문이 이미지 안에 포함된 비전 전용 입력 설정 도입의 단계를 포함합니다. 이 과정을 통해 MMMU-Pro는 모델이 항목에서 텍스트와 이미지를 동시에 읽고 이해하도록 돕는 도전을 제공합니다.

- **Performance Highlights**: MMMU-Pro에서 모델 성능은 MMMU보다 상당히 낮았으며, 16.8%에서 26.9%의 성능 저하를 보였습니다. OCR 프롬프트는 대부분의 모델에 큰 영향을 미치지 않았지만, 체인 오브 쏘트(Chain of Thought, CoT) 방식은 일반적으로 성능을 향상시켰습니다.



### Towards a Unified View of Preference Learning for Large Language Models: A Survey (https://arxiv.org/abs/2409.02795)
Comments:
          Initial Commit, 21 pages

- **What's New**: 이 논문은 기존의 다양한 Preference Alignment 전략을 통합하고 분석하기 위해 네 가지 구성 요소(모델, 데이터, 피드백, 알고리즘)로 분해하여 통일된 프레임워크를 제시합니다. 이를 통해 다양한 전략 간의 관계를 이해하고 협력할 수 있는 기회를 마련합니다.

- **Technical Details**: Preference Learning의 다양한 하위 방법 간의 관계가 파편화된 상태에서, 이 논문은 Reinforcement Learning(RL) 기반 방법과 Supervised Fine-tuning(SFT) 기반 방법을 통합하여 새로운 분류 프레임워크를 제안합니다. 이 프레임워크는 기존 알고리즘의 이해를 심화시키고, 상호 보완적인 접근법을 제시합니다.

- **Performance Highlights**: 구체적인 사례를 통해 알고리즘의 작동 방식을 자세히 설명하여 독자들이 현재의 선호 alignment 방식과 그 장단점을 명확히 이해할 수 있도록 돕습니다. 또한, LLM과 인간의 선호를 정렬하는 과정에서의 도전 과제와 미래 연구 방향에 대한 논의도 포함되어 있습니다.



### A Comparative Study of Pre-training and Self-training (https://arxiv.org/abs/2409.02751)
Comments:
          19 pages, 2 figures, 9 tables

- **What's New**: 본 논문에서는 사전 학습(pre-training)과 자기 학습(self-training), 그리고 미세 조정(fine-tuning)을 결합한 앙상블 방법론을 제안하며, 이를 통해 다양한 학습 패러다임을 비교하고 연구합니다.

- **Technical Details**: 이 연구는 반지도 학습(semi-supervised learning) 환경에서의 사전 학습과 자기 학습의 관계 및 그 성과를 고찰하고 있으며, 특히 데이터 증강(data augmentation) 기법을 활용하여 자기 학습의 효과를 증대시키고자 합니다.

- **Performance Highlights**: 실험 결과, 반지도 사전 학습(pre-training)과 미세 조정(fine-tuning) 패러다임이 가장 뛰어난 성능을 보여주었고, 자기 학습이 반지도 사전 학습과 결합될 경우 추가적인 이점이 없음을 확인하였습니다.



### Pooling And Attention: What Are Effective Designs For LLm-Based Embedding Models? (https://arxiv.org/abs/2409.02727)
Comments:
this https URL

- **What's New**: 대규모 실험을 통해 LLM 기반 임베딩 모델의 풀링(embedding) 및 어텐션(attention) 전략을 비교하고, 새로운 Multi-Layers Trainable Pooling 전략을 제안합니다.

- **Technical Details**: 임베딩 모델의 성능을 비교하기 위해 동일한 훈련 데이터와 LLM(base model)을 사용하여 다양한 풀링 방식 및 어텐션 전략으로 훈련된 여러 모델 간의 실험을 수행했습니다. 제안된 Multi-Layers Trainable Pooling 전략은 모든 은닉 층의 출력을 변환하는 방식으로, cross-attention 네트워크를 활용합니다.

- **Performance Highlights**: Bidirectional attention 및 추가적인 trainable pooling layer가 텍스트 유사도 및 정보 검색에서 뛰어난 성능을 보였으나, 클러스터링 및 분류 작업에서는 간단한 설계 방식에 비해 크게 개선되지 않았습니다. Multi-Layers Trainable Pooling 전략은 기존의 방법보다 통계적으로 유의미한 우수성을 입증했습니다.



### Pre-training data selection for biomedical domain adaptation using journal impact metrics (https://arxiv.org/abs/2409.02725)
- **What's New**: 본 연구는 특정 품질 지표를 이용하여 사전 학습 데이터셋을 개선함으로써 생물의학 분야의 언어 모델 성능을 향상시킬 수 있는지를 탐색합니다.

- **Technical Details**: BERT 모델을 PubMed 데이터셋의 다양한 부분 집합에 대해 지속적으로 사전 학습하며, 저널 임팩트 메트릭스인 h-index와 SJR을 사용하여 구성을 정의하고 평가합니다. 필터링 후, 총 15.9B 토큰의 코퍼스를 사용합니다.

- **Performance Highlights**: 저널 임팩트 메트릭스를 사용한 프루닝(pruning)은 효율적이지 않은 것으로 나타났지만, 적은 수의 초록을 사용한 사전 학습이 모델 성능을 저하시킬 필요는 없다는 것을 보여줍니다.



### A Data Selection Approach for Enhancing Low Resource Machine Translation Using Cross-Lingual Sentence Representations (https://arxiv.org/abs/2409.02712)
Comments:
          Accepted at I2CT 2024

- **What's New**: 이번 연구는 영어-마라티어 번역에서의 데이터 품질 문제를 다루며, 기계 번역 모델의 성능을 개선하기 위한 새로운 데이터 필터링 접근법을 제시합니다.

- **Technical Details**: 제안된 방법론은 멀티링구얼 SBERT (Sentence-BERT) 모델을 활용하여 원문과 번역문 간의 의미적 동등성을 평가합니다. 특히, IndicSBERT 유사성 모델을 사용하여 언어적으로 올바른 번역문을 유지하고, 상당한 편차가 있는 예제를 제거합니다.

- **Performance Highlights**: IndicSBERT를 이용한 필터링 후 번역 품질에서 현저한 개선을 보이며, 이는 제한된 자원 하에서의 기계 번역 과정에서의 오류를 줄이는 데 기여합니다.



### Deconfounded Causality-aware Parameter-Efficient Fine-Tuning for Problem-Solving Improvement of LLMs (https://arxiv.org/abs/2409.02686)
- **What's New**: 본 논문은 Large Language Models (LLMs)의 추론 능력을 향상시키기 위해 Deconfounded Causal Adaptation (DCA)이라는 새로운 Parameter-Efficient Fine-Tuning (PEFT) 방법을 제안합니다. 이를 통해 모델이 일반 문제 해결 능력을 추출하고 다양한 질문에 적용하게 합니다.

- **Technical Details**: 논문에서는 LLM의 텍스트 생성 프로세스를 주의(attention) 및 표현(representation) 수준에서 시각화하여 모델이 진정한 추론 능력을 가지고 있는지를 조사합니다. 또한, LLM의 추론 과정을 인과적(causal) 프레임워크로 정형화하여 시각화에서 관찰된 문제를 설명하고, DCA를 활용하여 모델의 추론 능력을 향상시키는 방법을 제시합니다.

- **Performance Highlights**: 실험 결과, DCA 방법이 다양한 벤치마크에서 일관되게 기준선(baseline)을 초과하는 성능을 보였으며, 단 1.2M의 조정 가능한 매개변수로 다른 미세 조정 방법과 비교해 더 나은 또는 동등한 결과를 달성하여 모델의 전반적인 정확도와 신뢰성을 향상시키는 것을 입증했습니다.



### Creating Domain-Specific Translation Memories for Machine Translation Fine-tuning: The TRENCARD Bilingual Cardiology Corpus (https://arxiv.org/abs/2409.02667)
- **What's New**: 이 논문은 번역자 및 언어 전문가들이 도메인 특정 평행 말뭉치(parallel corpora)를 생성하기 위해 번역 메모리(Translation Memory, TM)를 어떻게 구성할 수 있는지를 조사합니다.

- **Technical Details**: 논문은 번역자들이 데이터 품질과 통제를 위하여 주로 번역 도구를 활용하여 데이터를 생성하는 반자동(semiautomatic) TM 준비 방법론을 소개합니다. 이 방법론을 사용하여 터키어에서 영어로의 심장학 관련 평행 말뭉치가 구축되었습니다.

- **Performance Highlights**: 결과적으로 탄생한 TRENCARD Corpus는 약 80만 개의 출처 단어와 5만 문장으로 구성되어 있어 번역자가 자신의 맞춤형 TM을 효율적으로 구축하고 이중언어(bilingual) 데이터 작업에 활용할 수 있게 합니다.



### OpenFact at CheckThat! 2024: Combining Multiple Attack Methods for Effective Adversarial Text Generation (https://arxiv.org/abs/2409.02649)
Comments:
          CLEF 2024 - Conference and Labs of the Evaluation Forum

- **What's New**: 이 논문은 CLEF 2024 Task 6에서의 체크 상태! 실험과 결과를 다루며, 신뢰성 평가의 강건성을 테스트하기 위해 적대적 예제를 생성하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 연구팀은 적대적 공격 방법으로 BERT-Attack, Genetic 알고리즘, TextFooler 및 CLARE를 포함하여 다섯 가지 데이터 세트에서 다양한 방법을 테스트 및 수정했습니다. 향상된 공격 효과를 위해 하이브리드 방법을 도입하고, 의미를 유지하면서 공격 성공률을 높이는 방법을 탐구했습니다.

- **Performance Highlights**: 이 연구의 결과는 여러 방법을 수정하고 결합하여 보다 정교하고 효과적인 공격 전략을 만들 수 있는 가능성을 보여줍니다. 이를 통해 다양한 시스템의 내구성과 보안을 강화하는 데 기여할 수 있습니다.



### PUB: Plot Understanding Benchmark and Dataset for Evaluating Large Language Models on Synthetic Visual Data Interpretation (https://arxiv.org/abs/2409.02617)
- **What's New**: 본 논문은 대형 언어 모델(LLM)이 다양한 데이터 시각화 형태를 해석하는 능력을 평가하기 위해 고안된 새로운 합성 데이터셋을 소개합니다. 이 데이터셋은 실세계 시나리오를 포괄적으로 커버할 수 있도록 조정된 매개변수로 생성되었습니다.

- **Technical Details**: 우리는 이미지 내 시각 데이터와 관련된 질문을 포함한 다중 모달 텍스트 프롬프트를 사용하여 ChatGPT나 Gemini와 같은 최신 모델들을 평가합니다. 평가에 사용되는 벤치마크 데이터셋은 자동으로 생성되어 모델의 이전 노출이 없도록 만들어졌으며, 이를 통해 모델이 실제로 데이터를 해석하고 이해하는 능력을 평가합니다.

- **Performance Highlights**: 여러 최신 LLM을 평가한 결과, 모델들이 다양한 시각 데이터 해석에서 상이한 성능을 보였으며, 특정 강점과 약점이 드러났습니다. 향후 LLM의 개선은 자동 데이터 분석, 과학 연구, 교육 도구, 비즈니스 인텔리전스 애플리케이션에 상당한 도움을 줄 수 있습니다.



### More is More: Addition Bias in Large Language Models (https://arxiv.org/abs/2409.02569)
Comments:
          25 pages, 8 figures

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)에서 발생하는 additive bias(추가 편향) 복잡성을 탐구하며, 인간의 인지 편향과 유사한 현상을 관찰하여 LLMs의 의사결정 과정에 미치는 영향을 분석합니다.

- **Technical Details**: 본 연구에서는 일련의 실험을 통해 여러 LLM 모델(GPT-3.5 Turbo, Claude 3.5 Sonnet, Mistral 등)의 additive bias를 측정하는 작업을 수행했습니다. 측정된 작업에는 palindrome(회문) 생성, Lego 타워 균형 조정, 레시피 수정 등이 포함되었습니다.

- **Performance Highlights**: 실험 결과는 LLM이 additive 변화를 선호하는 경향을 명확히 보여주었습니다. 예를 들어, Llama 3.1은 회문 생성 작업에서 97.85%의 사건에서 추가하는 방식을 선택했으며, GPT-3.5 Turbo는 Lego 타워 균형 작업에서 76.38%의 사건에서 벽돌을 추가하는 방식으로 응답했습니다. 이러한 결과는 LLM의 디자인과 활용에 있어 환경적인 비용을 증가시킬 수 있는 단점이 있음을 시사합니다.



### Language is Scary when Over-Analyzed: Unpacking Implied Misogynistic Reasoning with Argumentation Theory-Driven Prompts (https://arxiv.org/abs/2409.02519)
- **What's New**: 이 논문에서는 미소지니 감지를 논증적 추론(Argumentative Reasoning) 작업으로 제안하고, 이탈리아어와 영어에서 미소지니를 표현하는 암묵적인 추론을 이해하는 대형 언어 모델(LLMs)의 능력을 조사합니다.

- **Technical Details**: 연구에서 논증 이론(Argumentation Theory)을 기반으로 한 프롬프트를 사용하여 제로샷(zero-shot) 및 몇샷(few-shot) 설정에서 실험을 진행합니다. 주요한 기법으로는 체인 오브 소트(chain-of-thought) 추론과 강화된 지식(augmented knowledge)을 포함합니다. 즉, LLMs가 여성에 대한 고정관념에서 발생하는 내재적인 지식을 바탕으로 가정의 이해를 도출하도록 합니다.

- **Performance Highlights**: LLMs는 미소지니 발언에 대한 추론 능력에서 부족함을 보이며, 인간 주석가들이 생성한 텍스트와 비교하여 결과의 질을 평가합니다. 이러한 접근은 미소지니 데이터세트를 풍부하게 하고, 훈련된 도구의 일반화 능력을 향상시키는 데 기여할 수 있습니다.



### Word and Phrase Features in Graph Convolutional Network for Automatic Question Classification (https://arxiv.org/abs/2409.02481)
- **What's New**: 이 논문에서는 질문 분류(question classification)를 개선하기 위해 그래프 합성곱 신경망(Graph Convolutional Networks, GCNs)을 기반으로 한 새로운 접근 방식인 Phrase Question-Graph Convolutional Network (PQ-GCN)을 제안합니다. 질문을 그래프 형태로 표현하여 언어의 내재적 구조를 효과적으로 모델링하고, 향상된 분류 정확성을 위한 구문 기반 특성을 통합합니다.

- **Technical Details**: 본 연구에서는 질문을 노드(node)로 단어 또는 구문을, 엣지(edge)로 구문적(syntactic) 및 의미적(semantic) 관계를 나타내는 그래프 형태로 표현합니다. PQ-GCN은 GCN을 사용하여 질문의 구조적 및 의존 관계를 학습하고, 다양한 저자원(low-resource) 환경에서도 분류 성능을 향상시키기 위한 구문 기반 특성이 포함됩니다.

- **Performance Highlights**: 실험 결과, GCN과 구문 기반 특성을 결합한 PQ-GCN 방식이 전통적인 분류 방법보다 정확하고 맥락 인지적인 질문 분류를 가능하게 하며, 그래프 신경망 연구와 교육 분야의 실용적 응용 간의 간극을 해소할 수 있는 잠재력을 보여줍니다.



### DetectiveQA: Evaluating Long-Context Reasoning on Detective Novels (https://arxiv.org/abs/2409.02465)
- **What's New**: 이 논문에서는 Large Language Models (LLMs)의 긴 문맥 처리 능력을 평가하기 위한 새로운 벤치마크인 DetectiveQA를 소개합니다. DetectiveQA는 평균 문맥 길이가 100K tokens 이상이며, 이는 LLM의 내러티브 추론 능력을 평가합니다.

- **Technical Details**: DetectiveQA는 주로 형사 소설을 데이터 출처로 사용하여 긴 문맥의 캐릭터 관계, 줄거리 전개, 동기 분석 등의 다양한 질문을 포함한 600개의 질문을 수집했습니다. 본 연구의 초점은 긴 문맥 종속성 문제의 처리와 내러티브 추론 능력을 평가하는 새로운 차원의 평가 지표를 도입하는 것입니다.

- **Performance Highlights**: 기존의 긴 문맥 LLM들은 실제 긴 문맥 종속성 질문을 효과적으로 처리하는 데 상당한 발전이 필요함을 보여줍니다. 결과적으로, DetectiveQA는 LLM의 내러티브 추론 능력의 한계를 정의하고, 더 깊은 연구와 고급 응용 프로그램 개발을 위한 도구로 작용할 것으로 기대됩니다.



### What is lost in Normalization? Exploring Pitfalls in Multilingual ASR Model Evaluations (https://arxiv.org/abs/2409.02449)
Comments:
          Sumbitted to EMNLP 2024

- **What's New**: 이 논문은 다국어 자동 음성 인식 (ASR) 모델 평가의 문제점들을 집중적으로 조사하며, 특히 인디카 언어 스크립트에 중점을 두고 있습니다.

- **Technical Details**: 주요 ASR 모델(OpenAI Whisper, Meta의 MMS, Seamless, Assembly AI의 Conformer)의 텍스트 정규화 절차를 분석하였으며, 철자, 구두점 및 특수 문자와 같은 불일치를 제거하려는 기존의 접근 방식이 인디카 스크립트에 비효율적이라는 점을 강조합니다.

- **Performance Highlights**: 경험적인 분석과 언어학적 검토를 통해, 이러한 결점들이 인디카 언어에 대한 성능 지표를 인위적으로 부풀리게 만든다는 것을 보여주었습니다. 마지막으로, 모국 언어의 전문성을 활용한 새로운 정규화 절차 개발을 제안합니다.



### Abstractive Text Summarization: State of the Art, Challenges, and Improvements (https://arxiv.org/abs/2409.02413)
Comments:
          9 Tables, 7 Figures

- **What's New**: 이 논문은 추출 요약(extractive summarization) 기법과 대조적으로 추출적 텍스트 요약(abstractive text summarization)의 최신 동향을 심도 있게 조사하고 있으며, 최첨단 기법, 주요 과제 및 연구 방향에 대한 포괄적인 개요를 제공합니다. 또한 모델 복잡성(model complexity)과 확장성(scalability)에 대한 주요 비교 테이블을 제공합니다.

- **Technical Details**: 최신 기법들을 전통적인 Sequence-to-Sequence 모델, Pre-trained Large Language Models(사전 훈련된 대형 언어 모델), Reinforcement Learning(강화 학습), Hierarchical Methods(계층적 방법), Multi-modal Summarization(다중 모달 요약)으로 분류했습니다. 이 논문은 인간처럼 요약을 작성하는 함축적 접근 방식을 탐구하며, 기계 모델의 지식 통합 및 혁신적인 전략들이 주효할 수 있음을 강조했습니다.

- **Performance Highlights**: 이 연구는 의미 표현 부족, 사실 일관성(factual consistency), 제어 가능한 요약 등 여러 도전 과제를 강조하면서, 정보의 정확성과 일관성을 높이기 위한 새로운 접근 방식들을 제안합니다. 또한 이 논문은 다국어 요약(multi-lingual summarization) 및 긴 문서 요약(long-document summarization)과 같은 새로운 연구 분야도 제시하고 있습니다.



### Determination of language families using deep learning (https://arxiv.org/abs/2409.02393)
Comments:
          First draft. Comments are welcome

- **What's New**: 이 논문에서는 c-GAN (convolutional generative adversarial) 신경망을 활용하여 기존의, 사라진 이해 가능한, 그리고 하나의 해독되지 않은 (Cypro-Minoan) 언어의 음역 텍스트 조각을 분석하여 언어적 유사성을 파악합니다.

- **Technical Details**: 기존의 번역이나 해독과 무관하게, c-GAN 신경망 모델을 통해 다양한 언어의 텍스트를 분석합니다. 이 방법론은 언어의 계통을 파악하기 위한 접근법으로 고안되었습니다.

- **Performance Highlights**: 향후 더 정교한 신경망 기술을 활용해 해독에 기여할 수 있을 것으로 기대됩니다.



### STAB: Speech Tokenizer Assessment Benchmark (https://arxiv.org/abs/2409.02384)
Comments:
          5 pages

- **What's New**: 이 논문에서는 STAB(Speech Tokenizer Assessment Benchmark)라는 새로운 평가 프레임워크를 제안하여, 다양한 음성 토크나이저의 성능을 비교하고 그 특성을 체계적으로 평가할 수 있는 방법을 제공합니다. 이를 통해 음성 토크나이저 모델의 발전을 촉진시키고, 표준화된 벤치마크를 사용한 비교 분석이 가능하게 됩니다.

- **Technical Details**: STAB는 임의의 음성 토크나이저의 성능을 평가하는 데 필요한 여러 차원을 고려합니다. 평가 항목으로는 발화자 불변성(Speaker Invariance), 문맥 불변성(Context Invariance), 언어 불변성(Language Invariance), 그리고 소음 및 음향 변동에 대한 저항성을 측정합니다. 또한, 허프만 인코딩 효율(Huffman Encoding Efficiency)과 바이트 쌍 인코딩 효율(Byte-pair Encoding Efficiency) 같은 압축 효율성을 평가하여 토크나이저의 성능을 비교합니다.

- **Performance Highlights**: STAB의 실험을 통해 다양한 음성 작업 및 토크나이저 선택에서의 성과와 STAB 지표 간의 상관관계를 입증하였으며, 해당 지표가 음성 토크나이저의 성능을 신뢰할 수 있게 나타내는 것을 확인하였습니다.



### How Privacy-Savvy Are Large Language Models? A Case Study on Compliance and Privacy Technical Review (https://arxiv.org/abs/2409.02375)
Comments:
          8 pages, 4 figures

- **What's New**: 최근 대형 언어 모델(LLMs)의 발전은 언어 생성, 요약, 복잡한 질문 응답 등의 다양한 분야에서 그 활용을 크게 확대했습니다. 그러나 이 모델들이 개인정보 보호 준수 및 기술적 개인정보 리뷰에 적용되는 것은 충분히 탐구되지 않았습니다.

- **Technical Details**: 본 연구에서는 개인정보 관련 작업에서 LLMs의 성능을 평가하는 포괄적인 사례 연구를 수행하며, 개인정보 정보 추출(PIE), 법적 및 규제 핵심 포인트 탐지(KPD), 개인정보 보호 정책과 관련된 질문 응답(QA) 등의 작업을 다룹니다. 우리는 개인정보 기술 검토(Privacy Technical Review, PTR) 프레임워크를 도입하여 소프트웨어 개발 생애 주기 동안 개인정보 위험 완화의 역할을 강조합니다. 여러 주요 LLMs(BERT, GPT-3.5, GPT-4 등)의 개인정보 준수 점검 및 기술적 개인정보 리뷰 수행 능력을 조사합니다.

- **Performance Highlights**: LLMs는 개인정보 리뷰 자동화 및 규제 불일치 식별에 대한 가능성을 보여주지만, 진화하는 법적 기준을 완전히 준수하는 능력에는 여전히 상당한 격차가 존재합니다. 연구는 LLMs의 개인정보 준수 능력을 개선하기 위한 실용적인 권장 사항을 제공하며, 법적 및 규제 요구 사항과의 더 나은 통합 필요성을 강조합니다.



### Do Large Language Models Possess Sensitive to Sentiment? (https://arxiv.org/abs/2409.02370)
Comments:
          10 pages, 2 figures

- **What's New**: 최근 대형 언어 모델(LLMs)은 언어 이해에서 뛰어난 능력을 보여주지만, 감정 분석에 대한 포괄적인 평가가 필요합니다. 이 논문은 LLMs의 감정 감지 및 반응 능력을 조사하고, 다양한 응용 프로그램에 통합될 때의 중요성을 강조합니다.

- **Technical Details**: LLMs의 감정 분석 성능을 평가하기 위해 여러 실험을 수행하였고, 긍정적, 부정적, 중립적 감정을 구분하고 적절히 반응하는 능력을 비교했습니다. 'Sentiment Knowledge Workflow'를 개발하고 LLMs의 감정 민감성을 평가하여 훈련 과정 개선이 필요하다는 점을 발견했습니다. 또한, 다양한 아키텍처와 데이터셋에 따라 서로 다른 성능을 나타냈습니다.

- **Performance Highlights**: LLMs는 기본적으로 감정에 대한 민감성을 가지고 있지만, 정확도와 일관성에 있어 상당한 차이가 있었습니다. 예를 들어, 강한 긍정적 감정을 중립으로 잘못 분류하거나 풍자나 아이러니를 인식하지 못하는 사례가 있었습니다. 이러한 발견은 감정 분석의 복잡성을 강조하며, LLMs의 감정 인식을 향상시키기 위한 추가적인 연구가 필요함을 보여줍니다.



### Diversify-verify-adapt: Efficient and Robust Retrieval-Augmented Ambiguous Question Answering (https://arxiv.org/abs/2409.02361)
- **What's New**: 본 연구에서는 Retrieve-Augmented Generation (RAG) 프레임워크의 한계를 극복하기 위한 새로운 접근 방식인 Diversify-Verify-Adapt (DIVA) 프레임워크를 제안합니다. DIVA는 모호한 질문에 대한 응답을 보다 정확히 제공하기 위해 다양한 해석을 포함하는 조회 결과를 다양화하고, 이들의 품질을 검증하여 가장 적합한 응답 방식을 적용합니다.

- **Technical Details**: DIVA 프레임워크는 두 가지 주요 구성 요소로 이루어져 있습니다: 1) Retrieval Diversification (RD)와 2) Adaptive Generation (AG). RD는 추정된 질문의 해석을 기반으로 다양한 구절을 조회하는 방법을 사용하며, AG는 조회된 구절의 품질을 검증 후 가장 적합한 응답 생성 방식을 적용합니다. 새로운 품질 수준 분류로는 {Useful, Partial Useful, Useless}가 정의됩니다.

- **Performance Highlights**: DIVA는 기존 RAG 기법과 비교해 ASQA 및 SituatedQA 데이터셋에서 더 높은 정확도와 효율성을 결과로 보여 주었으며, 응답生成 속도를 1.5~3배 개선하였습니다. 초기 실험 결과, 단일 조회 방식에서 벗어난 DIVA의 접근 방식이 모호한 질문을 효과적으로 처리할 수 있음을 입증했습니다.



### Arctic-SnowCoder: Demystifying High-Quality Data in Code Pretraining (https://arxiv.org/abs/2409.02326)
- **What's New**: 최근 언어 모델의 효과적인 사전 학습에 있어 고품질 데이터의 중요성이 강조되고 있습니다. 이 논문에서는 Arctic-SnowCoder-1.3B라는 새로운 데이터 효율적인 코드 모델을 소개합니다.

- **Technical Details**: Arctic-SnowCoder-1.3B는 555B 토큰을 기반으로 세 단계의 데이터를 통해 사전 학습되었으며, 각 단계에서 품질이 점진적으로 개선됩니다. 첫 번째 단계에서는 500B 표준 품질의 코드 토큰을 일반 사전 학습을 통해 사용하고, 두 번째 단계에서는 BERT 스타일의 품질 주석기를 통해 선택된 50B 고품질 토큰을 사용합니다. 마지막으로, 세 번째 단계에서는 Llama-3.1-70B에 의해 생성된 5B의 합성 데이터를 이용한 향상된 사전 학습을 진행합니다.

- **Performance Highlights**: Arctic-SnowCoder-1.3B는 BigCodeBench에서 최신 성능을 기록했으며, 1T 토큰 이하로 학습된 유사한 규모의 모델들을 능가합니다. 특히, Phi-1.5-1.3B에 비해 36%의 성능 향상을 보여주었으며, HumanEval+ 벤치마크에서도 StarCoder2-3B를 초월하며 경쟁력을 유지하고 있습니다.



### MMLU-Pro+: Evaluating Higher-Order Reasoning and Shortcut Learning in LLMs (https://arxiv.org/abs/2409.02257)
- **What's New**: MMLU-Pro+는 LLM의 논리적 사고 및 단순한 문제 해결 전략 저항 능력을 평가하기 위해 설계된 진화된 기준 벤치마크입니다.

- **Technical Details**: MMLU-Pro+는 다수의 정답을 제공하는 질문을 통해 LLM의 복잡한 추론 능력을 테스트합니다. 이를 위해 단축 학습(shortcut learning)을 줄이기 위한 새로운 메트릭인 shortcut selection ratio 및 correct pair identification ratio를 도입했습니다.

- **Performance Highlights**: 다섯 개의 최첨단 LLM에 대한 평가 결과, 상당한 성능 차이가 발견되었으며, 이는 모델의 추론 능력과 편향에 대한 취약성을 강조합니다.



### RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins (early version) (https://arxiv.org/abs/2409.02920)
Comments:
          Project page: this https URL

- **What's New**: 이번 논문은 RoboTwin이라는 새로운 벤치마크 데이터세트를 소개합니다. 이는 실제 원격 조작 데이터와 디지털 트윈에서 생성된 합성 데이터의 조합으로, 두 팔 로봇 사용 시나리오를 위해 설계되었습니다.

- **Technical Details**: RoboTwin 데이터세트는 COBOT Magic 플랫폼을 사용하여 수집된 다양한 도구 사용 및 인간-로봇 상호작용 데이터를 포함합니다. AI 생성 콘텐츠(AIGC)을 활용하여 2D 이미지를 3D 모델로 변환하는 새로운 접근 방식이 도입되었습니다. 또한, 대규모 언어 모델(LLMs)을 활용하여 전문가 수준의 훈련 데이터와 작업별 포즈 시퀀스를 생성합니다.

- **Performance Highlights**: 이 연구는 RoboTwin 벤치마크 데이터세트, 효율적인 실제-시뮬레이션 파이프라인, 자동 전문가 데이터 생성을 위한 언어 모델 활용의 세 가지 주요 기여를 통해 로봇 훈련 데이터의 부족 문제를 해결하고, 로봇이 더 높은 기능성을 갖추도록 돕는 데 기여합니다.



### Masked Diffusion Models are Secretly Time-Agnostic Masked Models and Exploit Inaccurate Categorical Sampling (https://arxiv.org/abs/2409.02908)
Comments:
          40 pages

- **What's New**: 최근 연구에 따르면 Masked Diffusion Models (MDMs)가 시간 변수와 무관하게 작동하며, 이는 그들이 본질적으로 masked models과 수학적으로 동등하다는 것을 보여줍니다. 새로운 first-hitting sampler (FHS) 기법을 통해 MDMs의 샘플링 속도를 20배 향상시킴으로써 이전의 효율성을 크게 개선했습니다.

- **Technical Details**: MDMs는 masked tokens의 개수를 사용하여 continuous-time evidence lower bound (ELBO) 목적함수를 정의하며, 이는 order-agnostic auto-regressive models과 일치합니다. FHS는 시간이 필요 없는 샘플링을 처럼 동작하여, 기존의 categorical sampling을 피하고 어떤 mask token이 처음으로 언마스크될 때를 analytically 샘플링합니다.

- **Performance Highlights**: MDMs는 낮은 generative perplexity를 보여주지만, 토큰의 다양성이 감소함에 따라 생성 품질이 다소 저하되는 문제가 발생합니다. 32-bit floating-point precision을 사용할 때, MDMs의 generative perplexity가 126.11에서 31.24로 크게 개선되지만, sentence entropy는 5.66에서 5.17로 낮아졌습니다.



### Configurable Foundation Models: Building LLMs from a Modular Perspectiv (https://arxiv.org/abs/2409.02877)
- **What's New**: 최근 LLM(large language model)의 발전은 대규모 매개변수로 인한 컴퓨팅 효율성과 확장성의 새로운 도전과제를 드러냈습니다. 이를 해결하기 위해, 연구진은 LLM을 기능 모듈로 분해하여 이를 조합하는 방식으로 복잡한 과제를 처리하는 모듈식 접근 방식을 제안하고 있습니다. 이 논문에서는 이러한 조합 방식의 효율성과 구성 가능성을 강조하여, 각 기능 모듈을 'brick'으로 정의하고 구성 가능 기반 모델(configurable foundation models)이라는 구조를 제시합니다.

- **Technical Details**: 모듈은 신경망이 훈련되는 과정에서 나타나는 'emergent bricks'와 후속 훈련을 통해 특별히 구축된 'customized bricks'로 구분됩니다. 다양한 기능 브릭에 기반하여 네 가지 브릭 지향 연산: retrieval & routing, merging, updating, growing을 제안하였으며, 이들 연산을 통해 LLM의 동적 구성이 가능해지며 복잡한 작업을 처리할 수 있습니다.

- **Performance Highlights**: 실증적 분석을 통해 FNN 계층이 모듈형 특성을 보이며, 신경망의 기능적 전문화와 신경 분할을 보여주었습니다. 이러한 연구는 기존 LLM 연구에 대해 새로운 모듈식 관점을 제공하며, 더 효율적이고 확장 가능한 기반 모델의 발전에 기여할 것입니다.



### Alignment-Aware Model Extraction Attacks on Large Language Models (https://arxiv.org/abs/2409.02718)
Comments:
          Source code: this https URL

- **What's New**: 최근 연구들은 대형 언어 모델(LLMs)에 대한 모델 추출 공격(Model Extraction Attacks, MEAs)이 증가하고 있음을 보여줍니다. 기존 접근은 깊은 신경망(DNNs) 용으로 설계된 추출 전략을 따르지만 LLM과의 교육 작업 간의 일관성을 무시하여 공격 성능이 낮아지는 문제가 발생하고 있습니다. 이를 해결하기 위해, 본 논문에서는 LLM 전용의 새로운 모델 추출 공격 알고리즘인 Locality Reinforced Distillation (LoRD)을 제안합니다.

- **Technical Details**: LoRD는 정책 경량화 스타일의 훈련 작업을 설계하여 피해 모델의 응답을 신호로 활용하여 로컬 모델을 위한 선호도를 형성합니다. LoRD의 수렴 과정은 LLM의 정렬과 일치하며, 쿼리 복잡성을 줄이며 탐사 기반의 절도를 통해 워터마크 방어를 완화할 수 있습니다. 기존 MEA는 MLE(Maximum Likelihood Estimation)와 KD(Knowledge Distillation)를 사용하여 LLM의 정렬 과정과 일관성이 없음을 이론적으로 증명합니다.

- **Performance Highlights**: LoRD는 175억 매개변수를 가진 상업용 LLM을 80억 매개변수를 가진 사전 훈련된 로컬 모델을 사용하여 단 100개의 쿼리로 성공적으로 추출할 수 있음을 보여줍니다. 이 로컬 모델은 데이터-텍스트 작업에서 피해 모델과 통계적으로 유사하게 수행되어 워터마크에 강하고 쿼리 효율성을 높입니다.



### Detecting Calls to Action in Multimodal Content: Analysis of the 2021 German Federal Election Campaign on Instagram (https://arxiv.org/abs/2409.02690)
Comments:
          Accepted Archival Paper for the CPSS Workshop at KONVENS 2024. Camera Ready Submission

- **What's New**: 이번 연구에서는 2021년 독일 인스타그램 선거 캠페인에서 Calls to Action (CTAs)의 자동 분류를 조사하여 소셜 미디어에서의 동원(mobilization) 이해를 심화했습니다. 2,208개의 인스타그램 스토리와 712개의 게시물을 분석하여, Fine-tuned BERT 모델 및 OpenAI의 GPT-4 모델을 사용했습니다.

- **Technical Details**: Fine-tuned BERT 모델은 합성 교육 데이터를 포함하여 0.93의 매크로 F1 점수를 달성했으며, 이는 강력한 분류 성능을 보여줍니다. 자동화된 CTAs의 탐지를 통한 성과 비교를 위해 zero-shot 및 few-shot 프롬프트를 사용한 다양한 GPT-4 모델 변형을 실험했습니다. 연구는 다양한 유형의 인스타그램 콘텐츠(스토리 vs. 게시물)와 텍스트 유형(OCR vs. 캡션 vs. 전사) 사이에서 CTAs 탐지 성능 차이를 비교했습니다.

- **Performance Highlights**: FDP와 녹색당(Greens)은 게시물에서 CTAs의 비율이 가장 높았으며, CDU와 CSU는 스토리에서 CTAs 비율이 가장 높았습니다. 이 연구는 또한 인스타그램 스토리와 게시물 간의 동원 전략의 차이를 강조하며, 49.58%의 게시물과 10.64%의 스토리에 CTAs가 포함되어 있다는 것을 발견했습니다.



### A Survey on Emergent Languag (https://arxiv.org/abs/2409.02645)
- **What's New**: 이번 연구는 인공지능 분야에서의 emergent language (신흥 언어) 연구의 포괄적인 검토를 통해 현재까지의 문헌을 종합하고 새로운 분류 체계(taxonomy)를 제안하여 연구자들에게 유용한 참고자료를 제공합니다.

- **Technical Details**: 이 논문은 181개의 과학 논문을 분석하여 emergent language와 관련된 기존의 평가 방법 및 메트릭스를 살펴보며, multi-agent reinforcement learning (MARL) 접근 방식에서의 언어 발전의 조건 및 성패 기준에 대해 논의합니다. 또한, 이는 기존의 자연어 처리(natural language processing, NLP) 연구와 차별화된 점을 강조합니다.

- **Performance Highlights**: 연구의 주요 기여로는 emergent language 분야의 분류 체계 개발, 정량화 접근 방식 및 메트릭스 분석, 개방된 질문들에 대한 요약 및 향후 연구 방향 제시 등이 포함됩니다.



### An Analysis of Linear Complexity Attention Substitutes with BEST-RQ (https://arxiv.org/abs/2409.02596)
Comments:
          Accepted in the IEEE Soken Language Technology Workshop 2024

- **What's New**: 이 연구는 Self-Supervised Learning (SSL)에서 Multi-Head Self-Attention (MHSA)를 Linear Complexity를 가진 최신 대체 방법들로 교체하는 효과를 최초로 평가했습니다. 특정한 방법들로는 HyperMixing, Fastformer, SummaryMixing, Mamba가 있습니다.

- **Technical Details**: 이 연구에서 다룬 대체 방법들은 모두 Linear Time Complexity를 가지고 있습니다. 중요한 사항으로, MHSA는 입력 길이에 대해 Quadratic Time Complexity를 가지지만, 이러한 대체 방법들은 VRAM 소비를 20%에서 60%까지 줄이고, 입력 시퀀스의 길이에 따라 속도를 7%에서 65%까지 향상시킵니다. 동작 방식으로는 Fastformer가 Additive Attention과 Element-wise Multiplication을 통해 성능을 보여주며, SummaryMixing은 매개변수화된 함수로 평균을 내어 요약 벡터를 생성합니다.

- **Performance Highlights**: MP3S 벤치마크를 기준으로 실험한 결과, Linear Complexity 방법인 BEST-RQ는 기존 MHSA 방식에 대한 성능을 유지하면서도 VRAM 소비를 감소시키고 처리 속도를 증가시켰습니다. 이 연구는 SpeechBrain 툴킷에서 코드 오픈소스를 제공하여, 커뮤니티가 효율적인 SSL 모델을 실험할 수 있도록 했습니다.



### A Comparative Study on Large Language Models for Log Parsing (https://arxiv.org/abs/2409.02474)
Comments:
          Accepted for publication in the 18th ACM/IEEE International Symposium on Empirical Software Engineering and Measurement (ESEM '24)

- **What's New**: 이번 연구에서는 최신 대형 언어 모델(LLMs)의 로그 파싱(log parsing) 성능을 분석하여, 무료로 사용할 수 있는 모델들이 유료 모델과 비교하여 어떻게 경쟁하는지를 밝혔습니다. 특히 CodeLlama와 같은 코드 전문화 모델이 우수한 성능을 보인 점이 주목할 만합니다.

- **Technical Details**: 연구에서는 지난 몇 년 동안의 6개 최신 LLM, 즉 유료 모델(GPT-3.5, Claude 2.1)과 무료 모델을 선택하여, 총 16개 오픈 소스 프로젝트의 시스템 로그에서 수집한 1,354개의 로그 템플릿에 대해 성능을 비교했습니다. 서로 다른 두 가지 프롬프트 기법을 설계하고, 생성된 템플릿과 실제 템플릿 간의 구문적 유사성을 평가했습니다.

- **Performance Highlights**: 무료 모델들이 유료 모델과의 성능 경쟁에서 효과적이며, 특히 CodeLlama가 GPT-3.5보다 10% 더 많은 로그 템플릿을 정확하게 추출했습니다. 이 연구는 코드 전문화된 LLM들이 로그 파싱에 큰 도움을 줄 수 있음을 시사합니다.



### Large Language Models as Efficient Reward Function Searchers for Custom-Environment Multi-Objective Reinforcement Learning (https://arxiv.org/abs/2409.02428)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)을 활용하여 강화 학습(RL)에서의 보상 함수 설계를 가능하게 하였습니다. 이 방법론은 사용자 요구사항에 따른 보상 요소를 생성하고, 각 요소의 가중치를 최적화하는 과정을 통해 RL 알고리즘의 성능을 향상시키는 데 초점을 맞추고 있습니다.

- **Technical Details**: 연구는 LLM을 활용한 화이트 박스 검색(white-box search) 기법을 채택하여 명확한 사용자 요구사항에 따라 보상 코드를 이중 단계로 분리하여 설계합니다. 보상 비평가(reward critic)를 통해 보상 구성 요소를 검증하고, 훈련 로그 분석기를 사용하여 가중치를 최적화하는 데 필요한 데이터를 제공합니다.

- **Performance Highlights**: 이 프레임워크는 인명 감시 없이도 오직 하나의 피드백에 기반하여 보상 코드를 성공적으로 교정하였습니다. 특히, 가중치 초기화 과정을 통해 사용자의 다양한 요구사항을 충족하는 복수의 보상 함수 세트를 확보할 수 있었으며, 100배의 가중치 불일치가 있는 경우라도 평균 4회의 반복만으로 해결책을 도출할 수 있었습니다.



### Large Language Models and Cognitive Science: A Comprehensive Review of Similarities, Differences, and Challenges (https://arxiv.org/abs/2409.02387)
Comments:
          10 pages, 1 figure

- **What's New**: 이번 리뷰 논문은 대형 언어 모델(LLMs)과 인지 과학의 교차점을 탐구하며, LLM과 인간의 인지 과정을 비교합니다. LLM의 인지 능력을 평가하는 방법과 인지 모델로서의 잠재력을 분석합니다.

- **Technical Details**: LLM들은 언어 처리, 추론 및 문제 해결에서 인간과 유사한 능력을 보여주지만, 새로운 문제에 대한 추론에서 한계를 보입니다. 이는 LLM의 일반화 능력에 제약을 나타냅니다. CogBench와 MulCogBench와 같은 도구들이 LLM의 인지 능력을 평가하기 위해 개발되었습니다.

- **Performance Highlights**: LLM은 특히 언어 처리 및 감각 판단 작업에서 인간과 유사한 성능을 보여주지만, 새로운 상황에서의 추론이나 기능적 언어 능력에서는 부족함이 있습니다. 향후 연구는 이러한 모델을 개선하고 인간 인지와의 일치를 높이는 데 중점을 두어야 합니다.



### NUDGE: Lightweight Non-Parametric Fine-Tuning of Embeddings for Retrieva (https://arxiv.org/abs/2409.02343)
- **What's New**: 이 논문에서는 NUDGE라는 새로운 비모수(non-parametric) 임베딩 조정 방법을 소개합니다. 이 방법은 기존의 사전 훈련(pre-trained) 모델과 비교하여 더 높은 정확도와 효율성을 제공합니다.

- **Technical Details**: NUDGE는 k-최근접 이웃 검색(k-NN retrieval)에 최적화된 데이터 레코드의 임베딩을 직접 변경하여 정확도를 극대화합니다. 이 방법은 실제로 NP-Hard 문제로 알려져 있으나, 제한된 변형을 통해 효율적으로 해결할 수 있습니다. 개선된 임베딩 변화는 사전 훈련 과정에서 학습된 의미를 왜곡하지 않도록 제한됩니다.

- **Performance Highlights**: 실험을 통해 NUDGE는 9개의 표준 텍스트 및 이미지 검색 데이터셋에서 기존 조정 방법들보다 NDCG@10에서 평균 10% 더 개선된 성능을 보여줍니다. NUDGE 방법은 시간당 200배 빠르게 실행되며, 평균적으로 정확도는 3.3배에서 4.3배 증가합니다.



### Optimal L-Systems for Stochastic L-system Inference Problems (https://arxiv.org/abs/2409.02259)
- **What's New**: 이 논문에서는 주어진 문자열 시퀀스를 생성할 수 있는 최적의 확률적 L-시스템을 구축하기 위한 두 가지 새로운 정리를 제시합니다. 첫 번째 정리는 단일 파생(de derivation)을 통해 주어진 단어 시퀀스를 생산할 확률을 극대화하는 확률적 L-시스템을 만드는 방법을 설명하며, 두 번째 정리는 여러 가능한 파생을 가진 단어 시퀀스의 생산 확률이 가장 높은 확률적 L-시스템을 규명합니다.

- **Technical Details**: 논문에서는 비선형 프로그래밍 솔버(nonlinear programming solvers)를 사용하는 최적화 기법을 통해 주어진 문자열 시퀀스로부터 최적의 확률적 L-시스템을 추론하는 알고리즘을 도입합니다. 이 알고리즘은 긍정적인 데이터만을 사용하여 훈련할 수 있는 L-시스템을 모델링하는 방법을 제안하여, 머신 러닝 기법 사용에 대한 새로운 가능성을 열어줍니다.

- **Performance Highlights**: 확률적 L-시스템은 실제 식물의 성장 패턴을 모방하여 생물학적 프로세스를 모델링하고, 합성 이미지를 생성하여 인공지능 신경망의 훈련 데이터로 사용할 수 있는 잠재력을 지니고 있습니다. 이로 인해 데이터 레이블링의 수고를 줄이고, 새로운 L-시스템을 알고리즘적으로 생성함으로써 생물학적 현상 모델링의 실용성을 크게 향상시킬 수 있습니다.



### Therapy as an NLP Task: Psychologists' Comparison of LLMs and Human Peers in CB (https://arxiv.org/abs/2409.02244)
- **What's New**: 대규모 언어 모델(LLMs)을 사용한 심리 치료 접근의 장점과 한계를 다룬 새로운 연구가 발표되었습니다. 이 연구는 개인화된 치료를 위해 LLMs의 활용을 탐구하며, 전통적인 치료 접근 방식과 비교하여 각각의 효과를 분석합니다.

- **Technical Details**: 이 연구에서는 HELPERT라는 프롬프트를 사용하여 CBT(인지 행동 치료) 기반의 대화를 재구성하였고, 두 명의 면허가 있는 CBT 훈련을 받은 임상 심리학자가 각 세션의 질을 평가하였습니다. 평가 방법으로는 Cognitive Therapy Rating Scale이 사용되었습니다.

- **Performance Highlights**: 연구 결과, 동료 상담 세션은 공감 및 치료적 동맹이 두드러졌으나 LLM 기반의 HELPERT 세션은 치료 방법 준수에서 우수한 성과를 보였습니다. 하지만 HELPERT 세션은 협동과 공감의 결여가 있었습니다. 이로 인해 인간-AI 협업의 중요성이 강조되었습니다.



### Temporal Order Preserved Optimal Transport-based Cross-modal Knowledge Transfer Learning for ASR (https://arxiv.org/abs/2409.02239)
Comments:
          Accepted to IEEE SLT 2024

- **What's New**: 이 논문에서는 Temporal Order Preserved OT (TOT)를 기반으로 한 Cross-modal Alignment and Knowledge Transfer (CAKT) 모델(TOT-CAKT)을 제안합니다. 이 모델은 음향(sequence)과 언어(sequence) 간의 지식을 효율적으로 전이할 수 있도록 설계되었습니다.

- **Technical Details**: TOT-CAKT 모델은 음향 시퀀스의 이웃 프레임을 언어 시퀀스의 이웃 지역으로 매핑하여 temporal order 관계를 유지합니다. 이 과정에서 pretrained Chinese PLM(BERT 사용)을 통해 Mandarin ASR 실험을 수행하였습니다. 이 과정에서의 특징적인 기법으로는 'Adapter' 모듈과 TOT 기반의 cross-modal matching 모듈이 포함됩니다.

- **Performance Highlights**: TOT-CAKT 모델은 여러 최신 모델들과 비교했을 때 ASR 성능에서 상당한 개선을 보여주었습니다. 특히, OT 기반 방법의 약점을 해결하며, 음향 및 언어 특성 간의 순차적인 정렬을 강화하였습니다.



### Unforgettable Generalization in Language Models (https://arxiv.org/abs/2409.02228)
Comments:
          18 pages, 9 figures, published in First Conference on Language Modeling 2024

- **What's New**: 이 논문은 Language Model(모델)에서 특정 기술을 "망각"할 때 행동의 변화를 연구했습니다. 주로 랜덤화된 레이블을 사용한 파인 튜닝 후 LM의 행동을 탐구하였고, 다양한 작업에서의 망각의 일반화(generalization) 양태를 관찰했습니다.

- **Technical Details**: 모델은 랜덤 레이블로 fine-tuning(파인 튜닝)하면서 훈련 집합 내의 예제에 대한 예측 변화를 보이지만, 훈련 집합 외부의 예제에 대한 예측 변화는 극단적인 가변성을 보였습니다. 어떤 작업에서는 망각이 외부 작업에 대해 일반화되지만, 다른 작업에서는 제한적입니다. 특히 LM의 초기 예측 확신(confidence)과 훈련 데이터의 다양성(variability)이 망각의 일반화 성질에 영향을 미치는 것으로 나타났습니다.

- **Performance Highlights**: 모델이 랜덤 레이블로 fine-tuning 후에도 비슷한 과제를 수행하는 경향이 있어, 망각된 기술의 일반화가 얕다는 점이 강조되었습니다. 결과적으로, 파인 튜닝을 통한 기술 제거의 어려움과 예측 불가능성이 드러났습니다.



### Efficient and Scalable Estimation of Tool Representations in Vector Spac (https://arxiv.org/abs/2409.02141)
- **What's New**: 최근 대규모 언어 모델(LLM)의 기능 호출 및 도구 사용에서의 발전은 외부 정보 소스와의 상호작용 및 복잡한 작업 실행을 가능하게 하여 모델의 능력을 크게 향상시켰습니다. 그러나 많은 도구가 사용가능할 때 LLM의 제한된 컨텍스트 윈도우는 도전 과제를 제시합니다.

- **Technical Details**: 본 논문에서는 도구 검색 응용 프로그램을 위한 합성 데이터 생성 프레임워크와 작은 인코더 모델을 이용한 데이터 기반의 도구 검색 전략을 제안합니다. 특히 Tool2Vec(사용 기반 도구 임베딩 생성), ToolRefiner(단계적 검색 방법), MLC(다중 레이블 분류 문제로 도구 검색 프레임 설정)와 같은 새로운 접근 방식을 소개합니다.

- **Performance Highlights**: 이 새로운 방법을 통해 ToolBench 데이터셋에서 Recall@K가 27.28까지 향상되었고, ToolBank에서는 30.5의 개선을 달성했습니다. 추가 실험 결과를 통해 우리의 방법의 타당성을 엄격히 검증합니다.



### Large Language Models versus Classical Machine Learning: Performance in COVID-19 Mortality Prediction Using High-Dimensional Tabular Data (https://arxiv.org/abs/2409.02136)
Comments:
          Code is available at: this https URL and this https URL. The datasets are available from the corresponding author on reasonable request (sdamirsa@ymail.com)

- **What's New**: 이 연구는 COVID-19와 관련된 사망률 예측에서 고차원 테이블 데이터셋을 활용해 전통적인 머신러닝 모델(클래식 머신러닝 모델, CML)과 대형 언어 모델(LLM)의 성능을 비교하고 평가했습니다.

- **Technical Details**: 9,134명의 COVID-19 환자 데이터를 활용하였으며, XGBoost와 랜덤 포레스트(RF)를 포함한 7개의 CML 모델이 학습 및 평가되었습니다. 구조화된 데이터는 텍스트로 변환되어 GPT-4와 Mistral-7b를 포함한 8개의 LLM에 의해 제로샷 분류(zero-shot classification)에 사용되었습니다. Mistral-7b는 QLoRA 접근 방식을 통해 파인튜닝(fine-tuning)되었습니다.

- **Performance Highlights**: CML 모델 중 XGBoost와 RF가 내부 검증에서 F1 점수 0.87, 외부 검증에서 0.83으로 가장 높은 정확도를 기록하였습니다. LLM 중에선 GPT-4가 F1 점수 0.43으로 최고 성능을 나타냈습니다. Mistral-7b의 파인튜닝은 리콜(recall)을 1%에서 79%로 크게 향상시켰으며, F1 점수는 0.74로 외부 검증 동안 안정적으로 유지되었습니다.



### Deep Knowledge-Infusion For Explainable Depression Detection (https://arxiv.org/abs/2409.02122)
Comments:
          13 pages, 2 figures

- **What's New**: 이번 연구는 DepressionFeature ontology (DFO)와 Commonsense Transformer (COMET)로부터 도메인 전문 지식을 결합한 Knowledge-infused Neural Network (KiNN)을 제안합니다. 이 모델은 사용자가 이해할 수 있는 설명력을 제공하여 정신 건강 전문가(MHPs)들이 우울증을 감지할 수 있도록 돕습니다.

- **Technical Details**: KiNN 모델은 딥러닝 기술을 활용하여 사용자 수준의 설명 가능한 방식으로 우울증을 감지합니다. 이는 전문가가 이해하는 개념 및 프로세스를 포함하며, 문맥 정보를 고려하여 더 나은 성능을 보여줍니다. 모델은 CLEF e-Risk와 PRIMATE 데이터셋에서 MentalBERT와 비교하여 통계적으로 유의미한 성능 향상을 보여줍니다.

- **Performance Highlights**: 연구 결과, 제안된 KiNN 모델은 CLEF e-Risk에서 25% MCC 증가 및 12% F1 증가, PRIMATE 데이터셋에서는 2.5% MCC 증가 및 19% F1 증가의 성과를 달성하였습니다. KiNN은 기존 모델들에 비해 더 높은 설명 가능성을 제공하며, 또 다른 모델들이 부족한 부분에 대하여 설명할 수 있는 능력을 갖추고 있습니다.



### CoRA: Optimizing Low-Rank Adaptation with Common Subspace of Large Language Models (https://arxiv.org/abs/2409.02119)
- **What's New**: 본 논문에서는 Low-Rank Adaptation (LoRA)의 효율성을 유지하면서 모델 파라미터를 절반으로 줄여주는 새로운 방법론, CoRA를 제안합니다. CoRA는 여러 대규모 모델에서 추출한 공통 기저(subspace)를 활용하여 LoRA의 B 매트를 최적화합니다.

- **Technical Details**: 제안하는 두 가지 접근 방식은 다음과 같습니다: (1) 공통 기저 행렬 B를 고정(freeze)하여 특수 작업에 대한 A 행렬을 학습하는 동시에 LoRA의 전체 파라미터를 50% 줄입니다. (2) 공통 기저 행렬 B를 LoRA의 기본 B 매트에 대한 향상된 초기 상태로 사용하여 더 나은 성능을 달성합니다.

- **Performance Highlights**: 첫 번째 접근 방식은 기존 LoRA 방법과 동일한 효율성을 유지하며, 두 번째 접근 방식은 LoRA의 원래 성능을 초과하여 최대 4%의 성능 개선을 달성합니다. 이는 유창성(fluency), 관련성(relevance), 정확성(accuracy)에서 뛰어난 성능을 보여줍니다.



### TSO: Self-Training with Scaled Preference Optimization (https://arxiv.org/abs/2409.02118)
- **What's New**: 이번 연구에서는 TSO(Self-Training with Scaled Preference Optimization)라는 새로운 프레임워크를 제안하여 Large Language Models(LLMs)의 인간 선호도에 대한 적합성을 높였습니다. 이를 통해 추가적인 보상 모델 없이도 선호 최적화 학습이 가능해지며, 다양한 인간의 반응을 통합해 응답의 다양성을 개선합니다.

- **Technical Details**: TSO 프레임워크는 모델 매트릭스를 활용하여 응답의 다양성을 증대시키며, 인간 및 AI 피드백을 통해 모델 선호 오류를 교정합니다. 이 과정에서 다단계 자기훈련(self-training) 구조를 채택하고, 미니 배치(iterative mini-batches) DPO와 쌍클립 보상 손실(dual clip reward loss)을 도입하여 데이터 효율성을 높이고 최적화 프로세스를 조정합니다.

- **Performance Highlights**: 실험 결과, TSO는 기존의 주요 방법들보다 다양한 정렬 평가 벤치마크에서 우수한 성능을 보였고, 선호 데이터 구축 및 모델 훈련 전략에서 실용적인 통찰력을 제공했습니다.



### Tiny-Toxic-Detector: A compact transformer-based model for toxic content detection (https://arxiv.org/abs/2409.02114)
Comments:
          6 pages

- **What's New**: 새로운 논문에서는 토닉(Toxic) 콘텐츠 탐지를 위한 소형 변환기 기반 모델인 Tiny-toxic-detector를 소개합니다. 210만 개의 매개변수만으로도, ToxiGen 데이터셋에서 90.97%의 정확도와 Jigsaw 데이터셋에서 86.98%의 정확도를 기록하며, 100배 이상의 모델들에 맞먹는 성능을 발휘합니다.

- **Technical Details**: Tiny-toxic-detector는 4개의 transformer encoder 레이어로 구성되며, 각 레이어는 2개의 attention head를 가지고 있습니다. 임베딩 차원은 64로 설정되어 있으며, feedforward 레이어의 차원은 128입니다. 이 모델은 공개 및 비공식 데이터셋을 사용하여 훈련되었으며, 훈련 과정에서 오버피팅을 활용하여 일반화를 개선하였습니다.

- **Performance Highlights**: 이 모델은 환경이 제한된 상황에서도 효과적으로 작동하도록 설계되었으며, 10MB의 RAM과 8MB의 VRAM만 필요합니다. CPU 기반 시스템에서 태이블 2의 결과에서 큰 모델들을 빠르게 능가하는 고속 추론이 가능합니다.



### Conversational Complexity for Assessing Risk in Large Language Models (https://arxiv.org/abs/2409.01247)
Comments:
          14 pages, 6 figures

- **What's New**: 최근 대형 언어 모델(LLM)의 발전은 다양한 응용 프로그램에서 인간과 유사한 텍스트를 생성할 수 있게 했지만, 이 과정에서 유해하거나 비윤리적인 콘텐츠를 생성할 가능성 또한 커졌습니다. 특히 복잡한 대화가 이러한 문제를 초래할 수 있다는 점에서, 최소한의 대화 길이(Conversational Length, CL)와 대화 복잡도(Conversational Complexity, CC)를 새로운 리스크 평가 지표로 제안합니다.

- **Technical Details**: CL은 특정 응답을 얻기 위해 필요한 대화의 길이를 정량화하는 측정 방법이며, CC는 유저의 지시 순서가 해당 응답에 이르는 데 필요한 Kolmogorov 복잡성을 기반으로 정의됩니다. Kolmogorov 복잡성의 비계산 가능성을 해결하기 위해, 우리는 CC를 참조 LLM을 사용하여 유저 지시의 압축성을 추정하는 방법으로 근사화합니다.

- **Performance Highlights**: 대규모 레드팀 데이터셋을 활용한 정량적 분석을 통해, 유해한 대화와 무해한 대화의 길이와 복잡도의 통계 분포를 살펴보았습니다. 연구 결과, 이 분석은 AI 안전성을 이해하는 데 유용한 도구가 될 수 있음을 제시하며, 유해 정보 접근성에 대한 통찰력을 제공합니다.



### CRAFT Your Dataset: Task-Specific Synthetic Dataset Generation Through Corpus Retrieval and Augmentation (https://arxiv.org/abs/2409.02098)
- **What's New**: CRAFT는 사용자가 제공하는 소수의 예시만으로 특정 작업을 위한 고품질 합성 데이터셋을 효율적으로 생성하는 새로운 방법을 제안합니다. 이는 기존의 데이터셋 수집 방법보다 빠르고 자원 소모가 적습니다.

- **Technical Details**: CRAFT( Corpus Retrieval and Augmentation for Fine-Tuning )는 웹 크롤링으로 수집한 대규모 원시 데이터에서 유사도를 기반으로 문서를 검색한 다음, LLMs( Large Language Models )를 활용해 이를 특정 작업 형식으로 변환하여 커스터마이즈된 샘플로 만드는 과정입니다. 이 과정은 정교한 수작업 커스터마이징 없이 자동으로 진행됩니다.

- **Performance Highlights**: CRAFT 활용 모델은 의학, 생물학, 일반 상식 질문 응답(QA) 및 요약 작업에서 데이터를 기반으로 훈련된 경쟁 모델보다 성능이 높거나 동등한 성과를 나타내며, 특히 요약 모델은 인간이 큐레이션한 데이터로 훈련된 모델보다 46 포인트 더 높은 선호도를 보였습니다.



### Political DEBATE: Efficient Zero-shot and Few-shot Classifiers for Political Tex (https://arxiv.org/abs/2409.02078)
Comments:
          26 pages, 5 figures

- **What's New**: 이 논문은 정치 문서의 zero-shot(제로샷) 및 few-shot(퓨샷) 분류를 위한 Political DEBATE 모델을 소개합니다. 이 모델은 기존의 대규모 언어 모델보다 효율적이며 오픈 소스라는 점에서 차별화됩니다.

- **Technical Details**: Political DEBATE 모델은 DeBERTa 알고리즘을 기반으로 하며, 10-25개의 문서로 훈련된 간단한 샘플을 사용하여 수백 또는 수천 개 문서로 훈련된 감독 분류기를 초월할 수 있습니다. PolNLI 데이터셋은 200,000개 이상의 정치 문서로 이루어져 있으며, 800개 이상의 분류 작업에 대해 정밀하게 라벨링 되어 있습니다.

- **Performance Highlights**: 이 모델은 최신 대규모 언어 모델들과 비교할 때 zero-shot 및 few-shot 분류에서 동등하거나 더 좋은 성능을 보이며, 더 높은 효율성을 자랑합니다.



### Spinning the Golden Thread: Benchmarking Long-Form Generation in Language Models (https://arxiv.org/abs/2409.02076)
- **What's New**: 이 논문은 언어 모델의 긴 문맥 처리 능력에서 부족한 점을 지적하며, 긴 텍스트 생성을 평가하는 새로운 기준인 Spinning the Golden Thread(SGT)를 제안합니다. 이 벤치마크는 작성된 긴 텍스트 내 특정 사건이나 조건을 포함하는 능력을 테스트합니다.

- **Technical Details**: SGT 벤치마크는 Diary Writing, Menu Design, Skyscraper Design, Urban Planning의 네 가지 시나리오를 설정하고, 각각의 시나리오에 대해 단일 지시(Single Instruction), 범위 지시(Range Instruction), 주기적 지시(Periodic Instruction) 세 가지 유형의 과제를 정의합니다. 이는 각 모델의 긴 문맥 처리 과정에서의 여러 기준에 대한 적합성을 평가합니다.

- **Performance Highlights**: 10개의 긴 문맥 LM을 평가한 결과, 기존의 Needle-in-a-Haystack(NIAH) 벤치마크에서는 양호한 성능을 보였으나, SGT에서는 만족스러운 결과를 나타내지 못했습니다. 생성된 텍스트의 길이가 증가할수록 모든 모델의 성능이 크게 저하되었습니다.



### OLMoE: Open Mixture-of-Experts Language Models (https://arxiv.org/abs/2409.02060)
Comments:
          61 pages (24 main), 36 figures, 14 tables

- **What's New**: OLMoE는 7억 개의 파라미터를 갖는 완전 개방형 Mixture-of-Experts(MoE) 언어 모델로, 5조 개의 토큰에서 사전 훈련되었습니다. OLMoE-1B-7B-Instruct를 만들기 위해 추가 조정을 했으며, 경쟁 모델을 초월하는 성능을 자랑합니다.

- **Technical Details**: OLMoE는 총 6.9B의 파라미터를 가진 디코더 전용 LM으로, 각 입력 토큰에 대해 활성화되는 파라미터는 1.3B입니다. 이 모델은 64개의 소형 전문가(Experts) 중 8개를 활성화하여 MoE 모듈에서 작동하며, 학습에는 로드 밸런싱 및 라우터 z-손실이 포함됩니다.

- **Performance Highlights**: OLMoE-1B-7B는 모든 공개된 1B 모델을 초월하며, 높은 추론 비용을 요구하는 밀집 모델과 경쟁할 수 있는 성능을 보입니다. MMLU 및 GSM8k와 같은 벤치마크에서 Llama2-13B와 유사한 성능을 기록했습니다.



### Enhancing Code-Switching Speech Recognition with LID-Based Collaborative Mixture of Experts Mod (https://arxiv.org/abs/2409.02050)
Comments:
          Accepted to IEEE SLT 2024

- **What's New**: 이번 연구에서는 코드 스위칭(code-switching) 음성 인식의 어려움을 해결하기 위해 Collaborative-MoE라는 혼합 전문가(MoE) 모델을 제안합니다. 이 모델은 언어 식별(Language Identification, LID)을 통해 전문가 그룹 간의 협업 메커니즘을 활용하여 더 나은 비지도 학습을 구현합니다.

- **Technical Details**: 제안된 모델은 LID 가중치를 기반으로 전문가의 선택을 통해 다양한 언어 도메인에서의 간섭을 최소화합니다. LID 가중치는 전문가 간 협업을 촉진하고, 전문가 그룹 내에서의 게이팅 네트워크를 통해 언어 외 다양한 특성에 대한 협업을 유도합니다. 실험을 통해 이 접근 방식이 기존 방법들보다 성능을 크게 향상시킴을 보여줍니다.

- **Performance Highlights**: 모델은 유사한 MoE 모델들이 필요로 하는 추가적인 미세 조정 없이도 효율적인 추론(inference) 기능을 유지하면서도 성능이 향상되었습니다. 추가 실험을 통해 LID 기반 라우팅 방법이 비지도 학습이 없는 모델보다 더 나은 성능을 달성함을 확인했습니다.



### BEAVER: An Enterprise Benchmark for Text-to-SQL (https://arxiv.org/abs/2409.02038)
- **What's New**: 본 논문에서는 기존의 text-to-SQL 벤치마크가 공용 데이터에서 수집된 테이블을 사용하고, 주로 인간이 생성한 질문과 SQL 쌍으로 구성된 테스트 세트를 기반으로 하고 있다는 점을 강조하고 있습니다. 이에 반해, 기업 데이터 웨어하우스의 데이터를 이용한 새로운 데이터셋 BEAVER를 제안하며, 기존 LLM(large language models)의 성능 저하 문제를 다루고 있습니다.

- **Technical Details**: 이 논문은 BEAVER 데이터셋을 통해 LLM들이 기업 환경에서 text-to-SQL 작업을 수행할 때의 어려움을 보여줍니다. 기업 데이터가 일반적으로 '다크 웹'에 존재하고, 그 스키마가 복잡하기 때문에 기존 LLM들이 교육받지 않은 데이터에 대해 잘 수행하지 못한다는 점을 지적합니다. BEAVER 데이터셋은 실제 기업의 자연어 질문과 그에 대한 올바른 SQL 쿼리 쌍으로 구성되어 있습니다.

- **Performance Highlights**: 최근의 LLM들(GPT-4o 및 Llama3-70B-Instruct 등)을 BEAVER 데이터셋에서 평가한 결과, 이들 모델은 거의 0에 가까운 end-to-end 실행 정확도를 기록하며, 공용 데이터셋에서의 성능과 비교해 상당히 낮은 성능을 보였습니다. 이로 인해, 기존의 프로퍼텐셜(enterprise data)으로부터 학습한 LLM이 실제 데이터 웨어하우스의 작업에 적합하지 않음을 증명하였습니다.



### FuzzCoder: Byte-level Fuzzing Test via Large Language Mod (https://arxiv.org/abs/2409.01944)
Comments:
          11 pages

- **What's New**: 본 논문에서는 FuzzCoder라는 정교하게 조정된 대형 언어 모델을 활용해 소프트웨어의 취약점을 효과적으로 탐지하기 위한 새로운 접근 방식을 제안합니다. 이를 통해 기존의 즐겨 사용하는 방법보다 더 나은 변이 이론과 전략을 적용하여 Fuzzing을 개선합니다.

- **Technical Details**: Fuzzing 과정이 시퀀스-투-시퀀스(sequnce-to-sequence) 모델링으로 구성되며, 대형 언어 모델(LLM)은 바이트 시퀀스를 입력받아 변형된 바이트 시퀀스를 출력합니다. FuzzCoder는 성공적인 공격에서 수집된 패턴을 학습하여 미래의 Fuzzing 탐사를 안내합니다. 이 과정에서 Fuzz-Instruct라는 명령어 데이터셋을 활용하여 LLM을 미세조정합니다.

- **Performance Highlights**: 실험 결과, FuzzCoder는 AFL(American Fuzzy Lop)을 기반으로 다양한 입력 형식(ELF, JPG, MP3, XML)에서 변이의 효율적인 비율(EPM)과 충돌 발생 수(NC)에서 현저한 개선을 보여줍니다. FuzzCoder는 이전 강력한 기준들에 비해 선형 커버리지와 브랜치 커버리지를 유의미하게 향상시켰습니다.



### Towards Leveraging Large Language Models for Automated Medical Q&A Evaluation (https://arxiv.org/abs/2409.01941)
Comments:
          10 pages, 3 figures, 3 tables

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)을 활용하여 의료 질문 및 답변(Q&A) 시스템에서 응답 평가를 자동화할 수 있는 가능성을 탐구합니다.

- **Technical Details**: 기존에는 의료 전문가들이 응답의 질을 평가하는 데 필수적이었으나, LLM을 사용하여 환자 데이터에서 도출된 질문을 통해 신뢰할 수 있는 평가를 재현할 수 있는지 검토하였습니다.

- **Performance Highlights**: 연구 결과는 유망한 성과를 제시하지만, 더욱 구체적이거나 복잡한 질문에 대해선 추가 연구가 필요하다는 점을 강조합니다.



### What are the Essential Factors in Crafting Effective Long Context Multi-Hop Instruction Datasets? Insights and Best Practices (https://arxiv.org/abs/2409.01893)
Comments:
          Work in progress

- **What's New**: 이 논문에서는 최신 대형 언어 모델(LLM)을 위한 새로운 Multi-agent Interactive Multi-hop Generation (MIMG) 프레임워크를 소개하여, 질 높은 멀티 홉(multi-hop) 질문 생성 및 검증을 통해 긴 맥락(long-context) 작업을 개선합니다.

- **Technical Details**: MIMG 프레임워크는 네 가지 주요 구성 요소로 이루어져 있습니다: Quality Verification Agent, Single-hop Question Generation Agent, Multiple Question Sampling 전략, Multi-hop Question Merging Agent. 이 중 Quality Verification Agent는 생성된 질문과 답변의 품질을 자동으로 검증하고, Single-hop Question Generation Agent는 단일 질문을 생성합니다. 여러 질문 샘플링 전략을 통해 질문의 다양성을 높이고, Multi-hop Question Merging Agent는 단일 질문을 통합하여 의미가 있는 멀티 홉 질문을 생성합니다.

- **Performance Highlights**: 실험 결과, 85% 이상의 데이터가 멀티 홉, 고품질 및 비중복적이며, 제안된 방법론은 모델 성능을 평균 7.54% 향상시켜, 대량의 인간 주석 데이터를 기반으로 훈련된 모델보다 뛰어난 성과를 거두었습니다.



### Investigating Expert-in-the-Loop LLM Discourse Patterns for Ancient Intertextual Analysis (https://arxiv.org/abs/2409.01882)
- **What's New**: 이 연구는 대형 언어 모델(LLM)을 활용하여 성경과 코이네 그리스어 텍스트 간의 상호텍스트성(intertextuality) 관계를 식별하고 검토하는 가능성을 탐구합니다. 이를 통해 LLM이 직접 인용, 암시 및 텍스트 간의 메아리를 감지하는 성능을 보여주었습니다.

- **Technical Details**: LLM의 사용을 통해 연구자들은 성경 저자 개인의 사용 패턴을 파악하고, 향후 저자들이 어떻게 이전 텍스트를 이해하고 재구성했는지를 밝힐 수 있습니다. 연구에서는 Claude Opus LLM을 사용하여 특정 신뢰성을 가진 상호텍스트적 관계를 평가하기 위한 전문가 평가 방법론을 제시합니다.

- **Performance Highlights**: LLM은 긴 쿼리 구문에 어려움을 겪고 및 잘못된 상호텍스트적 의존성을 포함할 경우의 한계를 보여주며, 이는 전문가의 평가가 중요함을 강조합니다. 이 접근법은 성경 연구 내에서 복잡한 상호텍스트성의 웹을 탐구할 수 있는 확장 가능한 방법론으로 자리잡을 수 있습니다.



### AgentRE: An Agent-Based Framework for Navigating Complex Information Landscapes in Relation Extraction (https://arxiv.org/abs/2409.01854)
Comments:
          Accepted by CIKM 2024

- **What's New**: 이번 논문에서는 복잡한 상황에서의 관계 추출(Relation Extraction, RE) 문제를 해결하기 위해 에이전트 기반의 프레임워크인 AgentRE를 제안합니다. AgentRE는 대형 언어 모델(Large Language Models, LLMs)의 잠재력을 최대한 활용하여 메모리, 검색, 반영 기능을 포함하여 RE를 수행합니다.

- **Technical Details**: AgentRE는 세 가지 주요 모듈을 포함하여 다양한 유용한 정보를 처리하고 수집하는 도구들로 작용합니다. 기존의 'text-in, text-out' 방식과 달리 AgentRE는 다수의 라운드에서 상호작용과 추론을 통해 정보 추출 작업을 수행하며, 이는 정보 출처의 다양성을 증대시키고 결과적으로 향상된 RE 성능을 이끌어냅니다. 또한, AgentRE는 LLM의 추론 및 메모리 기능을 활용하여 지속적으로 학습하고 지식을 축적합니다.

- **Performance Highlights**: 실험 결과, AgentRE는 영어와 중국어 두 개의 데이터셋에서 낮은 자원 환경에서도 탁월한 성능을 보여주며, 이를 통해 각 모듈의 효율성을 검증하였습니다. AgentRE에서 생성된 추론 경로는 다양한 추론 방식이 통합된 고품질 데이터셋을 구성하는 데 활용되어, 작은 모델의 성능 향상에도 기여합니다.



### Dialogue You Can Trust: Human and AI Perspectives on Generated Conversations (https://arxiv.org/abs/2409.01808)
Comments:
          17 pages, 15 figures, shorter version submitted to 22nd Annual Workshop of the Australasian Language Technology Association (ALTA'24)

- **What's New**: 이 연구는 대화 시스템 및 챗봇의 평가 방법에 대한 새로운 통찰을 제공합니다. AI와 인간의 평가를 비교하여 향상된 대화 평가 지표를 제안합니다.

- **Technical Details**: 본 연구에서는 7가지 주요 성능 지표(KPIs)인 Coherence, Innovation, Concreteness, Goal Contribution, Commonsense Contradiction, Incorrect Fact, Redundancy를 측정하는 실험을 수행했습니다. GPT-4o API를 활용하여 다양한 대화 데이터셋을 생성하고, 두 가지 실험 분석을 통해 인간과 AI의 평가 결과를 비교했습니다.

- **Performance Highlights**: 실험 결과, GPT 모델이 인간의 판단과 밀접하게 일치하며, 특히 사실 accuracy 및 commonsense reasoning에서 우수한 성능을 보여주었습니다. 그러나 Redundancy와 자기 모순 감소에는 여전히 어려움이 있는 것으로 나타났습니다.



### Training on the Benchmark Is Not All You Need (https://arxiv.org/abs/2409.01790)
- **What's New**: 이 논문에서는 Large Language Models (LLMs)의 데이터 누출(data leakage) 문제를 해결하기 위한 새로운 기법을 제안합니다. 다중 선택 형식의 평가 벤치마크를 기반으로 하여, 개별 선택지의 순서를 뒤섞는 방식으로 데이터 세트를 생성하고, 모델의 로그 확률 분포(log probability distribution)를 분석함으로써 데이터 누출을 감지합니다.

- **Technical Details**: 이 방법은 블랙박스(black-box) 환경에서도 작동하며, 모델의 교육 데이터(training data)나 가중치(weights)에 접근하지 않고도 데이터 누출 여부를 판단할 수 있습니다. 주어진 데이터의 로그 확률이 여러 개 고르지 않게 나타나면 데이터 누출을 나타내며, 이는 원래의 평가 데이터가 모델의 사전 훈련 데이터(pre-training data)에서 유출되었음을 시사합니다.

- **Performance Highlights**: 두 개의 LLM을 통한 실험을 통해 제안된 방법의 효과성을 입증하였으며, 31개의 오픈소스 LLM의 데이터 누출 위험을 4개의 벤치마크 데이터 세트에서 평가한 결과, Qwen 계열의 LLM이 여러 벤치마크에서 가장 높은 데이터 누출 위험성을 보였습니다.



### LLM-GAN: Construct Generative Adversarial Network Through Large Language Models For Explainable Fake News Detection (https://arxiv.org/abs/2409.01787)
- **What's New**: LLM-GAN이라는 새로운 프레임워크를 제안하여, LLM이 Generator와 Detector 역할을 하면서 보다 효과적으로 사실적 가짜 뉴스를 생성하고 탐지하도록 함.

- **Technical Details**: LLM-GAN은 두 가지 주요 단계를 통해 설명 가능한 가짜 뉴스 탐지를 가능하게 한다: (i) inter-adversary prompting과 (ii) self-reflection prompting. 이 방법은 LLM Generator와 Detector가 상호작용하여 가짜 뉴스 생성과 탐지를 배우도록 한다.

- **Performance Highlights**: LLM-GAN은 기존의 가짜 뉴스 탐지 방법들과 비교하여 예측 성능과 설명 품질에서 뛰어난 결과를 보였음. 이 모델은 클라우드 네이티브 AI 플랫폼에 통합되어 더욱 향상된 가짜 뉴스 탐지 서비스를 제공한다.



### State-of-the-art Advances of Deep-learning Linguistic Steganalysis Research (https://arxiv.org/abs/2409.01780)
Comments:
          Accepted by 2023 International Conference on Data, Information and Computing Science

- **What's New**: 이번 연구는 깊이 있는 언어적 스테가 분석(deep-learning-based linguistic steganalysis) 기술의 발전을 강조하며, 기존 연구에 대한 포괄적인 리뷰를 제공합니다. 특히, 언어적 스테가 분석의 일반적인 공식과 문서 분류(text classification)와의 차이를 비교하였습니다.

- **Technical Details**: 연구는 주로 두 가지 수준의 기존 작품을 벡터 공간 매핑(vector space mapping) 및 특징 추출(feature extraction) 모델에 따라 분류했습니다. 기존 깊이 학습 모델인 TStega-Net(·)을 통해 다양한 스테가 분석 기능을 추출하며, 수학적인 공식으로 표현되었습니다. 이 연구에서 소개된 특징들은 텍스트의 의미, 문장 구성, 구문 및 문법 관련 기능을 포함합니다.

- **Performance Highlights**: 기존의 통계적 벡터 임베딩(statistical vector embedding) 방법들은 간단한 구현과 짧은 훈련 시간을 제공하여, Word2Vec 및 GloVe와 같은 사전 훈련된 모델을 활용함으로써 성능을 개선하였습니다. 또한, 대규모 언어 모델(large-scale language models)의 도입에 따라, 더욱 높은 차원의 의미 공간으로의 진전을 이끌어냈습니다.



### In Defense of RAG in the Era of Long-Context Language Models (https://arxiv.org/abs/2409.01666)
- **What's New**: 이 논문에서는 긴 컨텍스트 언어 모델 (LLM) 시대에서의 검색 증강 생성(retrieval-augmented generation, RAG)의 유효성을 재조명합니다. 연구팀은 기존 RAG의 단점인 관련 정보에 대한 집중력 저하 문제를 해결하기 위해 순서 보존된 검색 증강 생성(order-preserve retrieval-augmented generation, OP-RAG) 메커니즘을 제안했습니다.

- **Technical Details**: OP-RAG는 긴 텍스트를 N개의 청크(chunks)로 분할하고, 이 중 가장 상위 K개의 청크를 검색하여 원본 텍스트의 순서를 유지합니다. 이렇게 함으로써 답변 품질이 초기에는 증가하다가 일정 시점에서 감소하는 '역 U자형 곡선(inverted U-shaped curve)'을 형성합니다. 이 과정에서 RAG는 구성 요소에 대한 혼란을 줄이고, 관련 정보에 대한 집중도를 높입니다.

- **Performance Highlights**: 실험에 따르면, OP-RAG를 통해 Llama3.1-70B 모델이 16161616K의 검색 토큰만 사용하여 F1 점수 44.43을 기록하였고, 이는 전체 128K 컨텍스트를 사용하는 경우보다 월등한 결과입니다. 반면 RAG 없이 사용할 경우, Llama3.1-70B는 34.32, GPT-4O는 32.36의 F1 점수를 기록하여 OP-RAG의 우수성을 입증하였습니다.



### Interpreting and Improving Large Language Models in Arithmetic Calculation (https://arxiv.org/abs/2409.01659)
Comments:
          Accepted by ICML 2024 (oral)

- **What's New**: 본 논문은 대형 언어 모델(LLMs)의 계산 능력을 심층적으로 분석하며, 이들 모델이 산술 연산을 수행할 때 좌우하는 특정 기제를 밝혀내는 것을 목표로 합니다.

- **Technical Details**: 연구진은 LLMs의 attention heads와 multi-layer perceptrons(MLPs)의 작용을 분석하였고, 계산 과정에서 5% 미만의 attention heads가 핵심적인 역할을 함을 발견했습니다. 이들 heads는 피연산자와 연산자에 집중하며, MLPs는 이 정보를 처리하여 최종 결과로 나아갑니다.

- **Performance Highlights**: 정확한 조정을 통해 32개의 attention heads만으로도 모델의 수학적 역량이 크게 향상되는 것을 발견했습니다. 이 방식은 전체 모델 조정보다 더 나은 성능을 보여주며, 비수학적 작업에 대한 성능 저하 없이 수학적 능력을 개선할 수 있음을 입증했습니다.



### From Yes-Men to Truth-Tellers: Addressing Sycophancy in Large Language Models with Pinpoint Tuning (https://arxiv.org/abs/2409.01658)
Comments:
          Accepted by ICML 2024

- **What's New**: 이 논문에서는 Large Language Models (LLMs)의 sycophancy(아첨) 문제를 해결하기 위해 Supervised Pinpoint Tuning (SPT)이라는 새로운 방법을 제안합니다. 기존의 Supervised Fine Tuning (SFT) 접근법이 LLM의 일반적인 성능 저하를 초래할 수 있는 반면, SPT는 특정 모듈만을 조정하고 나머지를 고정하여 sycophancy 문제를 완화하는 데 초점을 맞춥니다.

- **Technical Details**: SPT는 LLM의 내부 메커니즘을 분석하여 sycophancy에 영향을 미치는 모듈을 식별하고, 이들에 대해서만 미세 조정을 수행합니다. 논문에서는 Llama-2 및 Mistral 시리즈 모델을 활용하여 세 가지 능력(추론, 산술 추론 및 코드 생성)을 평가하는 다양한 실험을 진행하였으며, 특정 attention heads(어텐션 헤드)만 교정하는 방법이 효과적임을 보여주었습니다.

- **Performance Highlights**: 실험 결과, SPT는 sycophancy 문제를 효과적으로 완화하면서 LLM의 일반적인 성능을 거의 손상시키지 않는 것으로 나타났습니다. 기존의 SFT와 비교했을 때, pinpoint tuning은 보다 정확하고 효율적으로 sycophancy를 해결하면서도 다양한 데이터셋에서 우수한 성능을 나타냅니다.



### Booster: Tackling Harmful Fine-tuing for Large Language Models via Attenuating Harmful Perturbation (https://arxiv.org/abs/2409.01586)
- **What's New**: 본 논문에서는 유해한 파라미터 조정(harmful fine-tuning) 문제의 근본 원인이 모델 가중치에 대한 유해한 섭동(harmful perturbation)이라고 제안하며, 이를 해결하기 위한 새로운 접근법인 Booster를 제안합니다.

- **Technical Details**: Booster는 원래의 alignment loss에 손실 정규화기(loss regularizer)를 추가하여 유해한 섭동의 부정적인 영향을 완화합니다. 최적화 과정에서 이러한 정규화기를 포함시켜 손실 감소율(harmful loss reduction rate)을 조절하고, 이를 통해 모델의 안전성을 향상시킵니다.

- **Performance Highlights**: Booster는 기존의 Vaccine과 RepNoise보다 각각 최대 17.26% 및 20.08%의 평균 유해 점수를 줄이면서도 downstream task의 성능을 유지하는 것으로 나타났습니다.



### Towards Cross-Lingual Explanation of Artwork in Large-scale Vision Language Models (https://arxiv.org/abs/2409.01584)
- **What's New**: 이번 연구는 기존 LVLM들이 영어 데이터에 의존하는 문제를 해결하기 위해 다국어 데이터셋을 구축하였고, 머신 번역을 사용하지 않고 10개 언어(중국어, 네덜란드어, 영어, 프랑스어, 독일어, 이탈리아어, 일본어, 러시아어, 스페인어, 스웨덴어)에서 LVLM의 설명 생성 능력을 평가하였습니다.

- **Technical Details**: 연구에서는 LVLM을 구성하는 Vision Encoder 및 LLM을 설명 생성 능력이 포함된 추가 평가를 위해 활용하며, Alignment-10, Alignment-5, Full tasks라는 세 가지 설정으로 다국어 성능을 분석하였습니다. 또한, Instruction-Tuning을 통해 영어로 훈련된 모델이 다른 언어에서 설명 생성 능력을 얼마나 잘 습득할 수 있는지 시도했습니다.

- **Performance Highlights**: LVLM은 영어로 주어졌을 때 최상의 성능을 보였으며, 다른 언어에서는 성능이 떨어지는 경향을 보여 LVLM의 영어 데이터 학습이 다른 언어에 효과적으로 전이되지 않음을 나타냈습니다. 또한, 같은 언어로 주어진 설명 및 응답이 더 나은 성능을 발휘하는 경향이 확인되었습니다.



### AdaComp: Extractive Context Compression with Adaptive Predictor for Retrieval-Augmented Large Language Models (https://arxiv.org/abs/2409.01579)
Comments:
          8 pages, 5 figures, code available at https://anonymous.4open.science/r/AdaComp-8C0C/

- **What's New**: 이번 논문에서는 AdaComp이라는 새로운 저비용의 추출적(context compression) 문서 압축 방법을 제안합니다. 이 방법은 쿼리의 복잡도(query complexity)와 검색의 질(retrieval quality)에 기반하여 압축 비율(compression rate)을 적절히 결정합니다.

- **Technical Details**: AdaComp는 RAG 시스템이 정확한 응답을 하기 위해 필요한 최소 top-k 문서를 주석으로 달아 압축 비율(compression rate)로 설정합니다. 이를 통해 쿼리, 검색된 문서, 압축 비율의 삼중 항(triplet)을 구성하고, 이 데이터셋을 기반으로 압축 비율 예측 기계(compression-rate predictor)를 훈련합니다. 정보 필터링 과정에서는 예측된 압축 비율에 따라 중요한 문서들만을 선택합니다.

- **Performance Highlights**: 세 가지 QA 데이터셋과 하나의 대화형 Multi-doc QA 데이터셋에서 실험한 결과, AdaComp는 성능을 거의 동일하게 유지하면서도 인퍼런스(inference) 비용을 크게 줄이는 데 성공하여 효율성과 성능 간의 균형을 이루었습니다.



### An Implementation of Werewolf Agent That does not Truly Trust LLMs (https://arxiv.org/abs/2409.01575)
- **What's New**: 이번 연구에서는 Werewolf 게임을 위한 컴퓨터 에이전트를 개발하기 위해 대규모 언어 모델(LLM)과 규칙 기반 알고리즘을 결합한 새로운 접근 방식을 제안합니다. 이 에이전트는 대화 기록을 분석하여 상황에 맞는 출력을 선택하고, 특정 상황에서는 반박을 하거나 대화를 종료하는 능력을 가지고 있습니다.

- **Technical Details**: 시스템은 대화 생성 모듈과 대화 분석 모듈로 구성되어 있습니다. 대화 생성 모듈은 게임 상태와 대화 역사에 기초하여 LLM에 입력할 프롬프트를 생성합니다. 대화 분석 모듈은 LLM을 사용하여 투표 및 점검 결과와 관련된 정보를 추출합니다. 이후 규칙 기반 알고리즘이 LLM의 출력을 평가하여 적절한 출력을 선택합니다.

- **Performance Highlights**: 정성적 평가 결과, 본 에이전트는 수정되지 않은 LLM에 비해 더 인간적인 대화 스타일을 가지며, 등장인물(차별화된 개성)을 삽입하여 보다 자연스러운 대화를 할 수 있었습니다. 해당 에이전트는 자유롭게 사용할 수 있으며, 향후 연구에 기여할 수 있도록 소스 코드가 공개되어 있습니다.



### Benchmarking Cognitive Domains for LLMs: Insights from Taiwanese Hakka Cultur (https://arxiv.org/abs/2409.01556)
Comments:
          Submitted to O-COCOSDA 2024

- **What's New**: 본 연구는 대형 언어 모델(LLMs)의 문화 지식 이해 및 처리 성능을 평가하기 위한 포괄적인 벤치마크를 소개하며, 특히 하카 문화에 집중하고 있습니다. Bloom의 분류법(Bloom's Taxonomy)을 활용하여 기억(Remembering), 이해(Understanding), 적용(Applying), 분석(Analyzing), 평가(Evaluating), 창작(Creating) 등 여섯 가지 인지 영역에 걸쳐 LLM을 체계적으로 평가하는 다차원 프레임워크를 개발하였습니다.

- **Technical Details**: 이 연구는 LLM의 성능을 분석하기 위해 다층 질문 세트를 개발하며, 각 질문은 Bloom의 분류법의 여섯 가지 수준에 해당하도록 설계되었습니다. 또한, Retrieval-Augmented Generation (RAG) 기술을 통합하여 LLM이 외부 데이터베이스에서 정보를 검색하고 이를 바탕으로 응답의 정확성을 높입니다. LLM의 문화 지식 적용 및 이해 능력을 분석하기 위해 하카 문화 지식 기반에서 36,522개의 질문이 생성되었습니다.

- **Performance Highlights**: 연구 결과, RAG 기술이 모든 인지 영역에서의 정확성을 향상시키는 데 효과적임을 보여주었으며, 특히 문화 지식의 정확한 검색과 적용이 필요한 작업에서 두드러진 성과를 보였습니다. 그러나 창의적인 작업에서 RAG의 한계가 드러나 향후 최적화의 필요성을 강조했습니다. 이 벤치마크는 문화적으로 다양한 맥락에서 LLM을 평가하고 비교하는 데 유용한 도구로, 미래의 AI 기반 문화 지식 보존 및 전파 연구에 귀중한 통찰력을 제공합니다.



### Self-Instructed Derived Prompt Generation Meets In-Context Learning: Unlocking New Potential of Black-Box LLMs (https://arxiv.org/abs/2409.01552)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 인간 선호와의 정렬을 개선하기 위한 새로운 프레임워크를 제안합니다. 특히, Black-Box LLMs(예: GPT-4)에 쉽게 적용할 수 있는 방법으로, 정보성이 있는 컨텍스트 환경을 자동으로 구성함으로써 더 신뢰할 수 있는 파생 프롬프트를 생성할 수 있게 합니다.

- **Technical Details**: 프레임워크는 self-instructed reinforcement learning (RL) 메커니즘을 포함하여, 파생 프롬프트 생성을 통해 응답 모델과 직접 상호작용합니다. 이 과정에서 원본 프롬프트와의 정렬을 보장하고, 수정된 프롬프트로부터의 불일치를 줄입니다. 또한, LLM의 in-context learning(ICL) 능력을 극대화합니다.

- **Performance Highlights**: 다양한 실험에서 제안한 방법이 기존의 프롬프트 개선 방법보다 응답 품질을 상당히 향상시키며, GPT-4 같은 Black-Box 모델에 대해서도 뛰어난 성능 개선을 보여 주었습니다.



### It is Time to Develop an Auditing Framework to Promote Value Aware Chatbots (https://arxiv.org/abs/2409.01539)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2306.07500

- **What's New**: 이 논문은 ChatGPT를 포함한 생성 AI 도구의 발전이 어떻게 다양한 사회적 가치와 기준에 부합하는지를 모니터링하기 위한 가치 기반 감사(framework)의 필요성을 강조합니다.

- **Technical Details**: 저자들은 AI 시스템의 건강 상태를 체크할 수 있는 감사 템플릿을 제시하며, GPT 3.5와 GPT 4의 응답을 비교하여 성별, 인종 및 장애에 대한 잠재적 편향을 검토했습니다.

- **Performance Highlights**: GPT 모델들은 성별 및 인종 차별 체크를 포함한 직업 관련 질문에 대해 비교적 일관된 응답을 보였으며, 감사 결과는 생성 AI 기술의 안전성과 신뢰성을 확보하는 데 중요한 기초 자료로 작용할 수 있다고 강조합니다.



### S$^3$c-Math: Spontaneous Step-level Self-correction Makes Large Language Models Better Mathematical Reasoners (https://arxiv.org/abs/2409.01524)
- **What's New**: 본 논문에서는 수학적 추론을 위한 새로운 능력인 자발적 단계별 자기 수정(spontaneous step-level self-correction) 기능을 갖춘 S$^3$c-Math라는 수학적 LLM 모델을 소개합니다. 이 기능은 LLM이 진행 중인 추론에서 오류를 감지하고 이를 즉시 수정하여 보다 신뢰할 수 있는 응답을 생성할 수 있게 돕습니다.

- **Technical Details**: 이 연구에서는 단계별 샘플링(step-level sampling) 접근 방식을 사용하여 자기 수정 데이터를 구축하고, 메타 학습(Meta Learning)을 통해 S3c-Math 모델이 자발적으로 자기 수정 기능을 갖추도록 훈련합니다. S3c-MathQA 데이터셋은 532K의 자기 수정 데이터를 포함하고 있으며, 기존의 MetaMathQA 데이터와 혼합하여 927K의 SFT(supervised fine-tuning) 데이터를 생성했습니다.

- **Performance Highlights**: 이 방법은 GSM8K 및 MATH와 같은 여러 수학적 평가에서 LLM 모델의 성능을 일관되게 향상시키는 것으로 나타났습니다. 실험 결과, 자발적 단계별 자기 수정 능력을 갖춘 S3c-Math 모델은 다양한 수학적 벤치마크에서 상당한 발전을 보였습니다.



### DiversityMedQA: Assessing Demographic Biases in Medical Diagnosis using Large Language Models (https://arxiv.org/abs/2409.01497)
- **What's New**: 본 연구는 의료 질의응답 시스템에서 대형 언어 모델(LLMs)의 인종 및 성별 편향을 평가하기 위한 새로운 벤치마크인 DiversityMedQA를 도입하였습니다. 이를 통해 다양한 환자 인구통계를 아우르는 의료 문제에 대한 LLM의 응답을 측정할 수 있는 방법론을 제시합니다.

- **Technical Details**: DiversityMedQA는 MedQA 데이터를 기반으로 질문을 변형하여 성별과 인종에 따른 미세한 진단 차이를 포착하는 것을 목표로 합니다. 연구에서는 GPT-3.5, GPT-4.0, GPT-4o, Llama3-8B, Gemini 모델의 정확도를 비교하였고, 질문 수정 과정에서 Clinical Reasoning을 반영하여 LLM의 불편한 결과를 식별하였습니다.

- **Performance Highlights**: 연구 결과, GPT-3.5에서 GPT-4o로 넘어가면서 성별 및 인종 분류의 정확도가 현저히 향상되었습니다. 성별 질문에 대한 정확도는 61.00%에서 89.82%로 증가하였고, 인종 질문에 대한 정확도는 42.32%에서 86.24%로 상승하였습니다. 모든 GPT 모델은 Llama3-8B를 초과하는 성능을 보였으며, GPT-4와 GPT-4o가 가장 높은 정확도를 달성했습니다.



### The Compressor-Retriever Architecture for Language Model OS (https://arxiv.org/abs/2409.01495)
- **What's New**: 최근 대형 언어 모델(LLM)의 발전은 다양한 모달리티(modality)에서 정보를 집계하고 처리하는 능력을 크게 향상시켰습니다. 이로 인해 LLM이 단순한 챗봇에서부터 실제 세계와 상호 작용할 수 있는 범용 에이전트로 변모할 수 있는 가능성이 열렸습니다. 본 논문에서는 LLM을 운영 체제(OS)의 핵심 구성 요소로 사용하는 개념을 탐구하며, 이는 데이터를 처리하는 CPU 역할을 수행합니다.

- **Technical Details**: 이 논문에서는 context window를 RAM처럼 활용하는 LM OS의 발전을 위해 context의 생애 주기 관리( life-long context management)를 다루고, 이를 위해 compressor-retriever 아키텍처를 제안합니다. 이 아키텍처는 base 모델의 forward function을 사용하여 context를 압축하고 검색하며, end-to-end differentiability를 보장합니다. 이를 통해 모델 간의 간섭을 최소화하면서도 내용의 중복을 피할 수 있습니다.

- **Performance Highlights**: 초기 실험에서는 이 아키텍처가 in-context learning(ICL) 작업에서 유망한 성능을 나타냈으며, 이는 완전한 stateful LLM OS로 발전하는 단계로 나아갈 수 있는 가능성을 보여줍니다.



### Masked Mixers for Language Generation and Retrieva (https://arxiv.org/abs/2409.01482)
Comments:
          23 pages, 15 figures (11 primary, 4 supplementary)

- **What's New**: 이 논문에서는 Attention 메커니즘의 사용으로 인해 입력 데이터의 많은 정보가 손실된다는 주장을 제시합니다. 특히, Self-Attention 대신 Masked Convolutions을 적용한 Masked Mixer 구조를 도입하여 Transformer 모델보다 더 효율적으로 언어 작업을 수행할 수 있음을 보입니다.

- **Technical Details**: 논문에서는 Masked Mixer가 입력 데이터를 더 정확히 표현할 수 있으며, Transformer의 Hidden Layer는 정보가 부족하다는 것을 발견했습니다. Masked Mixer는 데이터셋 TinyStories에서 언어 모델 학습을 더 효과적으로 수행하며, Transformer와의 하이브리드 방식에서도 괜찮은 성능을 보입니다.

- **Performance Highlights**: Masked Mixer는 Transformer보다 언어 검색(task)에서 더 뛰어난 성능을 발휘하였습니다. 또한, Masked Mixer에서 추출한 Embedding이 Transformer에서 얻은 Embedding보다 더욱 뛰어난 요약-이야기 검색 결과를 나타냈습니다.



### PoliPrompt: A High-Performance Cost-Effective LLM-Based Text Classification Framework for Political Scienc (https://arxiv.org/abs/2409.01466)
Comments:
          23 pages, 5 figures

- **What's New**: 최근 대형 언어 모델(LLMs)의 발전으로 정치학에서 텍스트 분류 효율성을 높이는 새로운 방법이 열렸습니다. 본 논문은 LLMs를 활용하여 분류 정확도를 높이는 삼단계(in-context learning) 접근법을 소개하며, 실험 비용을 최소화합니다.

- **Technical Details**: 이 접근법은 자동으로 향상된 프롬프트 생성, 적응형 예시 선택, 두 개의 약한 LLM 간 신뢰성 메커니즘을 포함합니다. 이 방법은 BBC 뉴스 보고서, 카바노 후보자 확정, 2018년 선거 광고 데이터를 사용하여 검증되었습니다. 실험 결과에서 제로샷 분류에서 F1 점수가 0.36 상승하고, 인간 레이블링 대비 경제적 비용이 78% 감소하였습니다.

- **Performance Highlights**: 우리의 접근법은 기존의 전통적인 기계 학습의 한계를 극복하고 정치 과학에서 텍스트 분석을 위한 확장 가능하고 신뢰할 수 있는 솔루션을 제공함을 보여주었습니다.



### GenAgent: Build Collaborative AI Systems with Automated Workflow Generation -- Case Studies on ComfyUI (https://arxiv.org/abs/2409.01392)
- **What's New**: 이 논문에서는 AI 시스템의 새로운 접근 방식인 협업 AI 시스템을 제안합니다. 특히, 복잡한 워크플로우(workflows)를 자동으로 생성할 수 있는 GenAgent라는 LLM 기반 프레임워크를 소개하고 있습니다.

- **Technical Details**: GenAgent는 코드(code)로 워크플로우를 표현하고, 협업 에이전트(collaborative agents)와 함께 단계별로 워크플로우를 구성하는 방법을 독창적으로 구현하였습니다. ComfyUI 플랫폼에서 GenAgent를 구현하고, 새로운 벤치마크인 OpenComfy를 제안합니다.

- **Performance Highlights**: GenAgent는 실행 수준(run-level) 및 작업 수준(task-level) 평가에서 기존 접근 방식에 비해 뛰어난 성능을 보여주며, 복잡한 워크플로우를 효과적이고 안정적으로 생성할 수 있는 능력을 입증하였습니다.



### CV-Probes: Studying the interplay of lexical and world knowledge in visually grounded verb understanding (https://arxiv.org/abs/2409.01389)
Comments:
          13 pages, 1 figure, 11 tables, LIMO Workshop at KONVENS 2024

- **What's New**: 본 연구는 다양한 비전-언어(Vision-Language, VL) 모델들이 맥락 의존적 및 비맥락 의존적 동사구를 구체화하는 능력을 조사합니다. 이를 위해서 맥락 이해를 연구하기 위해 설계된 CV-Probes 데이터셋을 소개합니다. 이 데이터셋은 맥락 의존적 동사(예: 'beg')와 비맥락 의존적 동사(예: 'sit')가 포함된 이미지-캡션 쌍을 포함하고 있습니다. 

- **Technical Details**: 우리는 MM-SHAP 평가를 사용하여 동사 토큰이 모델 예측에 기여하는 정도를 평가합니다. 연구 결과, VL 모델들이 맥락 의존적 동사구를 효과적으로 구체화하는 데 어려움을 겪는다는 것을 보여줍니다. 또한, 비전-언어 모델(VLM)의 동사구 grounding 능력을 분석하고, 다양한 아키텍처를 가진 다섯 가지 최신 모델을 대상으로 실험하였습니다.

- **Performance Highlights**: 모델들은 맥락 의존적 및 비맥락 의존적 동사구를 균등하게 구체화하는데 실패하였으며, 이는 상황 및 행동 설명의 시각적 구체화에 필요한 세계 지식이 증가할수록 과제가 복잡해짐을 시사합니다.



### CHESS: Optimizing LLM Inference via Channel-Wise Thresholding and Selective Sparsification (https://arxiv.org/abs/2409.01366)
- **What's New**: 본 논문은 대형 언어 모델(LLM)을 엣지 디바이스에 배포하는 데 있어 발생하는 계산 오버헤드 및 메모리 요구 사항을 완화하기 위해 새로운 활성화 희소화(activation sparsification) 접근 방식을 제안합니다. 기존의 방법들이 성능 저하를 제대로 모델링하지 못하는 문제를 해결하고자, CHESS라는 새로운 방법론을 소개합니다.

- **Technical Details**: CHESS는 채널 단위(thresholding) 기준의 희소화를 통해 FFN 레이어의 각 활성화 채널에 대해 고유한 임계값(threshold)을 할당하고, 선택적 희소화(selective sparsification)를 통해 주의(attention) 모듈의 특정 레이어에 임계값 기반의 희소화를 적용합니다. 마지막으로, 희소 활성화를 기반으로 한 스파스 커널(sparse kernels)의 구현을 통해 LLM 추론을 가속화합니다.

- **Performance Highlights**: 실험 결과, CHESS는 8개의 다운스트림(downstream) 작업에서 기존 방법들에 비해 더 낮은 성능 저하를 달성하며, 최대 1.27배까지 LLM 추론 속도를 향상시킵니다.



### Know When to Fuse: Investigating Non-English Hybrid Retrieval in the Legal Domain (https://arxiv.org/abs/2409.01357)
Comments:
          Under review

- **What's New**: 이번 연구에서는 프랑스어 법률 도메인에서 하이브리드 검색(hybrid search)의 효과를 평가하며, 다양한 검색 모델을 조합했다. 영어 도메인에서 한정된 검색 방법에 대한 기존 연구와 차별화된다.

- **Technical Details**: 법률 검색을 위한 다양한 도메인 일반 검색 모델을 조합하여 평가하고, 한정된 도메인 특화 학습 데이터를 가정하여 영문 외의 언어 및 분야에서의 하이브리드 검색의 가능성을 탐구한다.  실험 방법으로는 late fusion 기법을 활용하고, 각 모델의 성능은 평균 역순위(MRR@10) 및 평균 r-precision(RP)로 측정한다.

- **Performance Highlights**: 영어 데이터셋 외에도 프랑스어 법률 도메인에서 하이브리드 검색의 일반화 가능성을 보여주었으며, 하이브리드 모델 조합이 단일 모델보다 더 좋은 성능을 생산했다. 그러나 도메인 특화 모델을 사용할 때는 조합이 성능을 저하시킬 수 있음을 관찰했다.



### Language Models Benefit from Preparation with Elicited Knowledg (https://arxiv.org/abs/2409.01345)
- **What's New**: 이 연구에서는 PREP이라는 간단한 일반 프롬프트 기법을 소개했습니다. 이 방법은 두 개의 언어 모델 인스턴스를 사용하여 정보를 생성하고 이를 바탕으로 질문에 답하는 구조로, 사용자의 도메인 지식과 무관하게 다양한 Q&A 작업에 적용할 수 있습니다.

- **Technical Details**: PREP 기법은 첫 번째 인스턴스(LM1)가 관련 정보를 생성하고, 두 번째 인스턴스(LM2)가 이 정보를 기반으로 질문에 답하는 방식으로 작동합니다. 이는 ‘지식 이끌기’와 ‘지식 전이’라는 두 단계를 포함합니다.

- **Performance Highlights**: 실험에서 PREP 방법은 100개의 이진 선택 질문과 세 가지 이미 출판된 상식 추론 데이터셋에서 다른 방법들에 비해 평균적으로 높은 정확도를 보였습니다. 이는 직접 질문하기, 제로샷 CoT, 단일 인스턴스 프롬프트 방법에 비해 일관되게 우수한 성능을 나타냅니다.



### Path-Consistency: Prefix Enhancement for Efficient Inference in LLM (https://arxiv.org/abs/2409.01281)
- **What's New**: 이번 연구에서는 대규모 언어 모델의 추론 효율성을 높이기 위해 경로 일관성(path-consistency)이라는 새로운 방법을 제안합니다. 이는 이전 생성 단계에서 나온 정답의 신뢰도를 활용하여 가장 유망한 경로의 접두사(prefix)를 식별하여 이후 방향의 생성을 유도합니다.

- **Technical Details**: 경로 일관성은 기존의 자기 일관성(self-consistency) 기법과 다르게 동적 추론 방법을 채택하여, 이미 생성된 추론 경로로부터 적절한 추론 단계를 지속적으로 추출하여 ‘접두사’로 사용합니다. 이러한 방식은 불필요한 샘플링을 줄이고 생성되는 토큰 수를 감소시켜 전체 추론 속도를 향상시킵니다.

- **Performance Highlights**: 실험 결과, 경로 일관성 방법은 추론 지연 시간을 7.8%에서 40.5%까지 단축시키면서도 수학적 추론, 상식 추론, 기호 추론 및 코드 생성 작업에서 작업 정확도를 유지하거나 오히려 향상시켰습니다. 수학적 추론에서 평균 28.7%, 상식 추론에서 20.9%, 기호 추론에서 20.3% 의 가속화를 기록했습니다.



### THInC: A Theory-Driven Framework for Computational Humor Detection (https://arxiv.org/abs/2409.01232)
Comments:
          Accepted at CREAI 2024 (International Workshop on Artificial Intelligence and Creativity)

- **What's New**: 본 논문은 여러 유머 이론에 기반한 THInC(Theory-driven Humor Interpretation and Classification)라는 해석 가능한 유머 분류 프레임워크를 제안하여 유머 이론 연구와 컴퓨터 유머 탐지 사이의 격차를 해소하고자 합니다. 이 프레임워크는 다양한 유머 이론을 나타내는 GA2M 분류기를 집합적으로 사용하며, 유머 이론의 다양한 측면을 정량적으로 반영하는 프록시(Proxy) 특성을 생성하는 투명한 흐름을 설계했습니다.

- **Technical Details**: THInC 프레임워크는 여러 유머 이론에 의해 주도되는 GA2M(Generalized Additive Model plus Interactions) 모델을 사용하여 다차원 상호작용을 학습합니다. 이 프레임워크는 F1 스코어 0.85를 달성하였으며, 프록시 특성 분석 및 이론과의 일치성 평가가 가능하다는 점이 특징입니다.

- **Performance Highlights**: 제안된 THInC 프레임워크는 유머 탐지의 기초를 마련한 첫 번째 연구로, 이론 기반 유머 분류의 발전을 위한 기초 시도를 합니다. 구현된 시스템은 유머 이론에 대한 실질적인 통찰을 제공하며, 자동으로 유머 이론의 비교를 가능하게 합니다.



### Prompt Compression with Context-Aware Sentence Encoding for Fast and Improved LLM Inferenc (https://arxiv.org/abs/2409.01227)
- **What's New**: 본 논문에서는 문맥 인식을 통한 프롬프트 압축(Context-aware Prompt Compression, CPC)이라는 새로운 기법을 제안합니다. 이 기법은 질문에 대한 관련성에 따라 문장을 압축하는 방식을 사용하며, 이를 위해 문맥 인식 문장 인코더를 개발하였습니다.

- **Technical Details**: CPC는 문맥 내 각 문장의 질문에 대한 관련성을 평가하는 새로운 문장 인코더를 활용합니다. 이 인코더는 긍정 샘플(질문과 관련된 문장)과 부정 샘플(관련이 없는 문장) 쌍을 학습하여 문장 표현을 구축합니다. 이 기법은 기존의 토큰 기반 압축 방식보다 더 우수한 성능을 보여줍니다.

- **Performance Highlights**: 제안된 방법은 LongBench와 ZeroSCROLLS 벤치마크에서 기존 상태-of-the-art 기법에 비해 평균 1.5%에서 2.1% 더 우수한 성능을 보이며, 인퍼런스 속도는 최대 10.93배 빠릅니다. 또한, 짧은 길이 제한에서도 더 높은 개선 효과를 나타냄으로써 관련 정보의 압축에 효과적임을 보여줍니다.



### A multilingual training strategy for low resource Text to Speech (https://arxiv.org/abs/2409.01217)
Comments:
          12 pages, 2 figures

- **What's New**: 본 논문에서는 저자원이 부족한 언어를 위한 TTS (Text to Speech) 데이터셋 구축을 위해 소셜 미디어에서 수집된 데이터를 활용하는 가능성을 탐구합니다. 또한 다중 언어 모델링과 크로스 링구얼 전이 학습 (transfer learning)의 적용 가능성을 평가합니다.

- **Technical Details**: 이 연구는 멀티링구얼 (multilingual) 모델링을 활용하여, 특정 저자원 언어(모로코 방언인 Darija)용 TTS 모델을 구축하기 위한 데이터 선택 및 훈련 전략을 제안합니다. 특히 1.2시간의 병렬 데이터 만을 사용하여 적응 및 미세 조정(cascade adaptation and fine-tuning) 전략을 탐구합니다.

- **Performance Highlights**: 결과적으로, 다중 언어 사전 훈련이 단일 언어 사전 훈련보다 생성된 음성의 이해 가능성과 자연스러움을 증가시키는 데 더 효과적이라는 것을 확인했습니다.



### Real World Conversational Entity Linking Requires More Than Zeroshots (https://arxiv.org/abs/2409.01152)
- **What's New**: 본 연구는 대화형 시스템에서의 에지 리킹(Entity Linking, EL) 모델의 실제 적용에서 발생하는 어려움에 대해 집중적으로 다루고 있습니다. 기존의 학습 및 평가 방식이 실제의 복잡한 대화 상황을 충분히 반영하지 못하고 있다는 점을 강조하며, 새로운 평가 시나리오와 대화 데이터셋을 제안합니다.

- **Technical Details**: 평가에 있어, 본 연구는 두 개의 지식 베이스(KB)인 Fandom과 Wikipedia를 사용하였으며, Fandom의 엔티티에 기반한 레딧(Reddit) 논의 내용을 토대로 새로운 제로샷(Zero-shot) 대화형 엔티티 링킹 데이터셋을 구축하였습니다. 제안된 데이터셋은 사용자가 Fandom 웹사이트에 하이퍼링크를 포함함으로써 엔티티를 분별하는 예를 포함하며, 뚜렷한 노이즈와 비정형 데이터를 처리할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 현재의 제로샷 EL 모델은 새로운 도메인 특화 KB에 노출될 때 성능이 현저히 저하되는 것을 보여주었으며, 이는 알맞은 학습 없이 대화 상황에 적응하는 데 제한적임을 시사합니다. 제안된 평가 환경은 연구자들이 실질적인 EL 문제를 해결하는 데 도움을 줄 것으로 예상되며, 데이터셋 또한 공개되어 연구에 기여할 것입니다.



### Pre-Trained Language Models for Keyphrase Prediction: A Review (https://arxiv.org/abs/2409.01087)
- **What's New**: 본 논문은 사전 훈련된 언어 모델을 이용한 키프레이즈 예측(PLM-KP)에 대한 포괄적인 분석을 제공합니다. 기존 문헌에서의 키프레이즈 추출 및 생성의 통합 탐색 부족을 다루고, 체계적인 분류체계를 통해 이 두 가지 작업에 대한 이해를 심화시키고자 합니다.

- **Technical Details**: 키프레이즈 예측에는 키프레이즈 추출(Keyphrase Extraction, KPE)과 키프레이즈 생성(Keyphrase Generation, KPG)이라는 두 가지 주요 작업이 포함됩니다. 이 과정에서 사용되는 모델들은 주로 self-supervised learning을 통해 사전 훈련된 언어 모델(Pre-trained Language Models, PLMs)이며, Attention Mechanism, Graph-based Ranking 및 Phrase-Document Similarity와 같은 다양한 방법론을 포함합니다.

- **Performance Highlights**: 본 논문은 PLM-KP 분야의 최신 동향을 다루며, 특히 딥러닝 기술을 활용한 NLP 작업에서의 성과를 강조합니다. 동시에 향후 연구 방향을 제시하여 키프레이즈 예측 연구의 발전을 이끌고자 합니다.



### A Perspective on Literary Metaphor in the Context of Generative AI (https://arxiv.org/abs/2409.01053)
Comments:
          Accepted as oral presentation to Workshop on Artificial Intelligence and Creativity (CREAI) at ECAI 2024

- **What's New**: 이 연구는 문학적 은유(literary metaphor)와 텍스트 생성의 창의성 간의 교차점을 살펴보며, AI를 활용한 새로운 문학적 표현 방식의 가능성을 모색합니다.

- **Technical Details**: 연구는 Afrikaans 언어에서 LSTM 기반 언어 모델을 훈련시켰으며, 이것은 메타포(metaphor), 유사(simile), 의인화(personification)와 같은 독창적인 표현을 생성합니다.

- **Performance Highlights**: 생성된 표현들은 창의적 언어 사용을 통해 독특한 감정적 강도를 전달하며, AI가 새로운 언어 사용 방식을 도입할 수 있는 가능성을 제시합니다.



### NYK-MS: A Well-annotated Multi-modal Metaphor and Sarcasm Understanding Benchmark on Cartoon-Caption Datas (https://arxiv.org/abs/2409.01037)
Comments:
          13 pages, 6 figures

- **What's New**: 새로운 벤치마크 NYK-MS (NewYorKer for Metaphor and Sarcasm)를 통해 1,583개의 은유(metaphor) 샘플과 1,578개의 풍자(sarcasm) 샘플을 포함한 데이터셋을 구축하였습니다. 이 데이터셋은 해당 표현이 포함되어 있는지, 어떤 단어가 은유 또는 풍자를 포함하는지, 무엇을 풍자하는지 등을 이해하기 위한 7가지 작업(tasks)으로 구성되어 있습니다.

- **Technical Details**: NYK-MS 데이터셋은 은유와 풍자 이해를 위해 7개 작업(MC, MW, ME, SC, SW, ST, SE)을 지원하며, 특히 풍자 타겟을 감지하는 ST 작업을 도입했습니다. 데이터셋은 cartoon-caption을 기반으로 하여 전문 아티스트들이 그린 만화에서 캡션을 추출합니다. 또한, GUI와 GPT-4V를 활용하여 주석 작업의 일관성과 품질을 높였습니다.

- **Performance Highlights**: 제로샷(zero-shot) 실험에서 대규모 언어 모델(LLM)과 다중 모달 모델(LMM)이 분류 작업에서 성능이 저조함을 보였으나, 규모가 증가함에 따라 다른 5개 작업의 성능은 향상되었습니다. 전통적인 사전 훈련(pre-training) 모델에 대한 실험 결과, 보강(augmentation) 및 정렬(alignment) 방법을 통해 성능이 개선됨을 보여주었습니다.



### Unleashing the Power of Task-Specific Directions in Parameter Efficient Fine-tuning (https://arxiv.org/abs/2409.01035)
Comments:
          Revisions ongoing. Codes in this https URL

- **What's New**: 이 논문은 대형 언어 모델(Large Language Models, LLMs)의 Parameter Efficient Fine-Tuning(PEFT) 전략과 관련된 새로운 개념인 task-specific directions를 탐구하고, 이를 통해 모델 성능을 향상시키는 LoRA-Dash라는 새로운 접근 방식을 제안합니다.

- **Technical Details**: LoRA-Dash는 task-specific directions을 정의하고 그 속성과 활용 도전 과제를 탐구하는 프레임워크를 제공합니다. 연구는 이러한 방향들이 사전 훈련 상태에서 특정 다운스트림 작업으로 전환하는 데 필수적이라는 점을 강조하며, LoRA 모델의 융통성을 활용하여 저역 랭크 행렬을 통해 파라미터의 효율적인 업데이트를 진행합니다.

- **Performance Highlights**: LoRA-Dash를 활용한 광범위한 실험 결과, 이 접근 방식이 특정 업무에서 모델 성능을 극대화하는 데 효과적임을 입증했으며, LoRA-Dash의 기저 메커니즘에 대한 심도 있는 분석도 수행되었습니다.



### Multi-Modal Multi-Granularity Tokenizer for Chu Bamboo Slip Scripts (https://arxiv.org/abs/2409.01011)
Comments:
          12 pages, 3 figures

- **What's New**: 이번 연구에서는 고대 중국 문자 분석을 위해 설계된 다중 모달리티 및 다중 세분화(tokenizer) 방식의 새로운 방법론을 발표하였습니다. 이 연구는 주로 춘추전국 시대(기원전 771-256년)의 죽간인(Chu bamboo slip, CBS) 문서에 중점을 두고 있습니다. 이전의 연구에서 나타난 한계를 극복하기 위해, 문자 탐지(character detection)와 문자 인식(character recognition)을 접목시킨 새로운 시스템을 구현하였습니다.

- **Technical Details**: 제안된 tokenizer는 이미지 스캔에서 문자 경계를 탐지한 후, 각 문자를 인식하여 정의된 어휘(vocabulary)와 대조합니다. 인식 신뢰도가 낮을 경우, 부문자(sub-character)로 분할하여 추가적인 정보를 추출합니다. 이 연구에서는 100,000개 이상의 주석이 포함된 CBS 문서의 데이터셋 CHUBS를 구축하여 제공하며, 이는 고대 문자 연구에 대한 접근성을 크게 높이는 기초 자료로 활용될 수 있습니다.

- **Performance Highlights**: 제안된 tokenizer는 기존의 서브워드(tokenizer) 방식보다 F1 점수에서 5.5%의 상대적 향상을 이루어냈습니다. 또한, 이 연구는 고대 중국 문자의 특정 분석뿐만 아니라, 기타 고대 문자 연구의 발전 가능성을 제공하는 중요한 기반을 제공합니다.



### DataSculpt: Crafting Data Landscapes for LLM Post-Training through Multi-objective Partitioning (https://arxiv.org/abs/2409.00997)
- **What's New**: 이번 논문에서는 DataSculpt라는 데이터 구성 프레임워크를 소개하며, 이는 긴 컨텍스트 훈련에 적합한 데이터 아키텍처를 전략적으로 보강하는 것을 목표로 합니다.

- **Technical Details**: DataSculpt는 결과적으로 모델의 긴 컨텍스트 활용 성능을 개선하기 위한 다목적 조합 최적화 문제로 훈련 데이터를 재구성하는 새로운 프레임워크입니다. 이 과정에서 관련성, 동질성, 무결성 및 계산 효율성을 포함한 여러 핵심 목표의 균형을 세심하게 조정합니다.

- **Performance Highlights**: DataSculpt는 긴 컨텍스트 훈련에서 다음과 같은 성능 향상을 달성했습니다: 검색 데이터 보강(18.09% 증가), 요약(21.23% 증가), 독해(21.27% 증가), 코드 완성(3.81% 증가), 전반적인 성능 또한 4.88% 향상되었습니다.



### What does it take to get state of the art in simultaneous speech-to-speech translation? (https://arxiv.org/abs/2409.00965)
- **What's New**: 본 논문에서는 음성-음성(speech-to-speech) 모델의 성능에서 나타나는 지연(latency) 특성을 심층적으로 분석하며, 특히 환각(hallucination)으로 인한 지연 스파이크를 중점적으로 다룹니다.

- **Technical Details**: 입력 매개변수인 self.frames_np은 0.35초 간격으로 증가하며, 이 간격은 ASR 시스템의 성능에 영향을 미칠 수 있습니다. 환각은 일반적으로 짧은 입력(≤0.7초)에서 자주 발생하며, 이에 따라 Whisper의 처리 시간이 증가합니다. 또한, avg_log_prob가 환각된 콘텐츠를 일관되게 표시하지 않아 LOG_PROB_THRESHOLD만으로는 환각을 필터링하기 부족함을 나타냅니다. 이를 해결하기 위한 방법으로 MIN_DURATION_THRESHOLD와 MAX_UNCOMMITTED_DURATION을 조정하여 지연을 개선할 수 있습니다.

- **Performance Highlights**: 본 연구는 환각을 최소화하고, 입력 관리와 매개변수 조정을 통해 전체 성능을 향상시킬 수 있는 방법을 제안합니다. 환각 발생을 줄이기 위해 입력을 반복하지 않아야 하며, 평균 지연은 대부분의 생성 작업에서 약 150ms로 유지됩니다.



### Large Language Models for Automatic Detection of Sensitive Topics (https://arxiv.org/abs/2409.00940)
Comments:
          2024 Oz CHI conference

- **What's New**: 본 연구는 정서적 웰빙과 관련된 민감한 정보를 탐지하기 위해 5개의 대형 언어 모델(LLMs)의 성능을 평가했습니다. 이 연구는 이러한 모델이 온라인 데이터셋에서 민감한 메시지를 감지하는 능력을 지님을 보여줍니다.

- **Technical Details**: 연구에서는 정확도(accuracy), 정밀도(precision), 재현율(recall), F1 점수, 일관성(consistency) 등의 성능 지표를 사용하여 LLM의 능력을 평가했습니다. 가장 높은 성능을 보인 모델인 GPT-4o는 평균 99.5%의 정확도와 0.99의 F1 점수를 기록했습니다.

- **Performance Highlights**: LLMs은 콘텐츠 조절 워크플로우에 통합될 가능성이 있으며, 안전하고 정확한 탐지 도구로서의 역할을 할 수 있음을 보여주었습니다. 또한, 향후 연구에서는 LLM 사용의 윤리적 고려 사항을 다룰 필요성이 강조되었습니다.



### Self-Judge: Selective Instruction Following with Alignment Self-Evaluation (https://arxiv.org/abs/2409.00935)
Comments:
          Under review

- **What's New**: 본 논문에서는 선택적 지시 수행(selective instruction following) 연구를 통해 인공지능 모델이 응답 품질이 낮을 경우 지시 이행을 거부할 수 있도록 하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: Self-J라는 새로운 자기 훈련(self-training) 프레임워크를 통해 지시 모델을 개발하며, 이 과정에서 인간 annotator의 품질 점수가 필요하지 않습니다. 모델의 자체 평가(self-evaluation) 기능을 활용하여 레퍼런스(reference) 답변과의 의미적 유사성을 평가하여 품질 점수를 추출합니다.

- **Performance Highlights**: Self-J 프레임워크는 다섯 개의 오픈소스 모델에서 광범위한 실험을 통해 유효성을 입증하였으며, WizardLM-13B-V1.2의 성능을 89.17에서 92.48로, AlpacaEval v1에서 v2로는 12.03에서 15.90으로 향상시켰습니다.



### User-Specific Dialogue Generation with User Profile-Aware Pre-Training Model and Parameter-Efficient Fine-Tuning (https://arxiv.org/abs/2409.00887)
- **What's New**: 이 논문은 사용자 지정 대화(user-specific dialogues)에 대한 새로운 접근 방식을 제안합니다. 기존의 개인화된 대화 연구는 페르소나 설명(persona descriptions)에 기반한 가상 사용자 대화에 초점을 맞추었지만, 본 연구는 실제 사용자 대화를 재현하는 것을 목표로 합니다. 이를 위해 사용자 대화 이력을 활용한 파라미터 효율적인 파인튜닝(parameter-efficient fine-tuning) 방법을 도입하고 있습니다.

- **Technical Details**: 사용자 프로필을 포함한 사전 훈련된 대화 모델(pre-trained dialogue model)과 결합된 학습 방법을 제안합니다. 파라미터 효율적인 파인튜닝을 통해 전체 모델에 소수의 파라미터를 추가하여, 적은 양의 훈련 데이터로도 효율적으로 훈련할 수 있고 모델 파괴에 강한 특징을 가지고 있습니다. 또한 자동으로 추론된 사용자 프로필에 대한 간단한 프롬프트(prompts)를 추가하여 사전 훈련된 모델이 사용자 프로필에 대한 지식을 향상시킨 발화를 생성할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 사용자의 개인 정보를 포함한 프롬프트를 사용한 대규모 언어 모델(large language model) 발화 생성 방식과 비교했을 때, 실제 사용자 발화와 더 높은 재현성을 가지는 발화를 생성할 수 있음을 보여주었습니다. 이는 적은 모델 규모에서도 가능했습니다.



### Self-evolving Agents with reflective and memory-augmented abilities (https://arxiv.org/abs/2409.00872)
- **What's New**: 대형 언어 모델(LLMs)의 성능을 향상시키기 위해 SAGE(Self-evolving Agents with reflective and memory-augmented abilities)라는 새로운 프레임워크를 제안했습니다. 이 프레임워크는 반복적인 피드백과 반사 메커니즘, 그리고 Ebbinghaus 망각 곡선에 기반한 메모리 최적화 메커니즘을 통합하여 다중 작업 처리 및 장기 정보 관리 능력을 크게 향상시킵니다.

- **Technical Details**: SAGE 프레임워크는 세 가지 주요 구성 요소인 반복적인 피드백, 반사, MemorySyntax를 통해 에이전트의 기억 관리 및 반복 개선 능력을 강화합니다. 각 에이전트는 사용자(U), 어시스턴트(A), 검사자(C)로 구성되며, 어시스턴트는 피드백을 기반으로 출력을 최적화합니다. 반사 메커니즘은 에이전트가 과거 경험을 분석하고 저장하여 미래의 작업에 더 잘 대응할 수 있게 합니다. MemorySyntax는 Ebbinghaus의 망각 곡선과 언어적 지식을 결합하여 기억을 최적화합니다.

- **Performance Highlights**: SAGE는 여러 벤치마크에서 강력한 기준선을 초과하는 성과를 달성하며, 특히 작은 모델에서 두드러진 향상을 보여줍니다. GPT-3.5와 GPT-4와 같은 강력한 LLM의 성능이 최대 2.26배 향상되었으며, 오픈 소스 모델에서는 성능이 57.7%에서 100%까지 향상되었습니다. 멀티 소스 질문 응답과 코드 생성 작업에서도 SAGE는 최첨단 결과를 달성했습니다.



### Harnessing the Power of Semi-Structured Knowledge and LLMs with Triplet-Based Prefiltering for Question Answering (https://arxiv.org/abs/2409.00861)
Comments:
          9 pages, published at IJCLR 2024

- **What's New**: 이 논문에서는 LLMs의 응답 품질을 크게 향상시킬 수 있는 4StepFocus라는 파이프라인을 제안합니다. 이 접근법은 모델이 relational context를 포착하고 스스로 기본적인 추론을 수행할 수 있는 능력을 활용하여 외부 지식에 대한 접근을 제공함으로써 이루어집니다.

- **Technical Details**: 4StepFocus는 1) LLM을 통한 triplet 생성을 통해 relational data를 추출하고, 2) 지식 그래프(knowledge graph)를 이용하여 답변 후보를 좁히고, 3) 관련 비구조 데이터와 벡터 유사도 검색(vector similarity search)을 통해 남은 후보를 정렬하며, 4) 제공된 배경 데이터를 통해 LLM이 최상의 후보를 재정렬하는 단계로 구성됩니다.

- **Performance Highlights**: 의학, 제품 추천 및 학술 논문 검색 테스트 세트에서 실험을 수행한 결과, 4StepFocus는 정보 검색에서 관련 traceable 배경 정보를 추가하며, 최신 방법들에 비해 성능을 유의미하게 개선한 것으로 나타났습니다.



### LanguaShrink: Reducing Token Overhead with Psycholinguistics (https://arxiv.org/abs/2409.00855)
- **What's New**: 이번 연구에서는 새로운 프롬프트 압축 프레임워크인 LanguaShrink를 제안합니다. 이 프레임워크는 심리언어학적 원리를 활용하여 입력 프롬프트에서 핵심 정보를 효과적으로 추출하면서도 긴 프롬프트의 길이를 줄이도록 설계되었습니다.

- **Technical Details**: LanguaShrink는 최소 모델을 활용해 압축 목표를 학습하고, KL 정규화 기반의 강화 학습 전략을 통합하여 훈련 과정을 최적화합니다. 이를 통해 각 텍스트 조각의 의미적 및 구조적 중요성을 평가하고, 최적의 조각을 선택하여 압축합니다. 또한, chunk 기반의 압축 알고리즘을 도입하여 조절 가능한 압축 비율을 제공합니다.

- **Performance Highlights**: 실험 결과 LanguaShrink는 기존의 프롬프트 압축 방법에 비해 최대 26배 압축을 달성했으며, 의미적 유사성을 유지하면서도 엔드 투 엔드 지연 시간을 1.43배 감소시켰습니다.



### Comparing Discrete and Continuous Space LLMs for Speech Recognition (https://arxiv.org/abs/2409.00800)
Comments:
          InterSpeech 2024

- **What's New**: 이 논문은 LLM 기반의 자동 음성 인식(ASR)에서 이산(Discrete) 및 연속(Continuous) 음성 표현의 비교를 최초로 종합적으로 수행하였습니다. 음성 특징의 연속성과 훈련 접근 방식을 기준으로 이들을 네 가지 범주로 나누어 분석하였습니다.

- **Technical Details**: 음성 표현은 지도 학습(Supervised)과 비지도 학습(Unsupervised)으로 나뉘며, 이는 이산 및 연속 표현에 대해 각각 적용됩니다. HuBERT와 HuBERT-CTC 모델을 활용하여 연속적인 음성 표현을 추출하고, K-means 군집화를 통해 이산 표현을 생성합니다. 이를 통해 다양한 모델링 접근 방식을 제안하고 평가합니다.

- **Performance Highlights**: LibriSpeech에서 HuBERT 인코더를 사용하여 1.69%의 최신 Word Error Rate (WER)를 기록하며, ASR 및 자연어 처리(NLP) 연구의 발전을 위한 중요한 통찰력을 제공합니다.



### Modeling Text-Label Alignment for Hierarchical Text Classification (https://arxiv.org/abs/2409.00788)
Comments:
          Accepted in ECML-PKDD 2024 Research Track

- **What's New**: 본 논문에서는 텍스트 분류의 새로운 접근 방식인 Hierarchical Text-Label Alignment (HTLA) 모델을 제안합니다. 이는 Text-Label Alignment (TLA) 손실 함수를 활용하여 동적으로 텍스트와 레이블 간의 정렬을 모델링합니다.

- **Technical Details**: HTLA 모델은 BERT를 텍스트 인코더로 사용하고, GPTrans를 그래프 인코더로 활용하여 계층적으로 인식 가능한 표현을 생성합니다. TLA 손실을 사용하여 텍스트와 관련된 레이블 간의 정렬을 증가시키고, 부적합한 레이블과의 거리를 둡니다. 이러한 방식은 텍스트와 레이블의 의미를 정렬하여 모델의 성능을 향상시킵니다.

- **Performance Highlights**: 실험 결과, HTLA 모델은 여러 벤치마크 데이터셋에서 기존 모델과 비교할 때 분류 성능을 개선하는 데 있어 우수성을 나타냈습니다. 특히 다양한 계층 구조를 가진 데이터셋에서도 뛰어난 성능을 보였습니다.



### The Dark Side of Human Feedback: Poisoning Large Language Models via User Inputs (https://arxiv.org/abs/2409.00787)
- **What's New**: 대형 언어 모델(LLMs)의 훈련 과정에서 사용자 피드백을 활용한 정렬 프로세스에 의존하는 시점에 새로운 유형의 사용자 주도 중독 공격이 발생할 수 있는 가능성을 제기합니다.

- **Technical Details**: 이 연구에서는 두 가지 공격 메커니즘을 소개합니다: (1) 선택 기반 메커니즘은 높은 보상 점수를 기록하는 유독한 응답을 유도하고, (2) 생성 기반 메커니즘은 최적화 가능한 프리픽스를 사용하여 모델 출력을 제어합니다. 1%의 악성 프롬프트를 데이터를 통해 주입함으로써 특정 트리거 단어 사용 시 독성 점수가 2배 증가하는 결과를 보여줍니다.

- **Performance Highlights**: 일반적으로 LLaMa-2 및 다양한 크기의 GPT-3 모델에 대한 평가를 통해, 우리가 실시한 사용자 주도 중독 공격은 트리거가 존재할 때 응답의 독성 점수를 26.5-226.9% 증가시켰습니다. 반면, 트리거가 프롬프트에 없을 때는 독성이 낮게 유지됩니다.



### Generating Media Background Checks for Automated Source Critical Reasoning (https://arxiv.org/abs/2409.00781)
- **What's New**: 이 논문은 정보 신뢰성 문제를 해결하기 위한 새로운 NLP 과제를 제안합니다. '미디어 배경 검사(media background checks)'라는 새로운 작업을 통해, 모델이 가져온 정보의 신뢰성과 경향성을 분석하고 요약하도록 합니다.

- **Technical Details**: 6,709개의 미디어 배경 검사 데이터셋을 소개하고, 이를 기반으로 retrieval-augmented 성능을 비교합니다. 강력한 LLM 모델을 사용하여, retrieved 문서의 신뢰성 평가를 지원하는 MBC 생성을 실험하고, 인간 평가를 통해 이들 MBC의 유용성을 검증합니다.

- **Performance Highlights**: MBC를 활용한 LLM 모델은 민감한 주제에 대한 질문에 더욱 나은 답변을 제공하며, 인간 사용자는 MBC가 retrieved 문서의 신뢰성을 판단하는 데 도움이 된다고 보고하였습니다.



### Polyrating: A Cost-Effective and Bias-Aware Rating System for LLM Evaluation (https://arxiv.org/abs/2409.00696)
- **What's New**: 최근의 대형 언어 모델(LLMs)의 효과적인 성능 평가를 위해 미리 평가 기반의 인간 평가 방법이 중요해졌습니다. 그러나 현재의 평가 시스템은 평가 결과에 영향을 미치는 인간의 편향을 고려하지 않으며, 정확한 평가를 위해 대규모의 비싼 데이터셋이 필요합니다. 이를 해결하기 위해, 우리는 Polyrating이라는 새로운 평가 시스템을 도입했습니다.

- **Technical Details**: Polyrating는 최대 사후 확률 추정(maximum a posteriori estimation)에 기반한 표현력 있고 유연한 평가 시스템으로, 자원의 소모를 최소화하며 모델 성능을 더 정교하고 철저하게 분석할 수 있게 합니다. 이 시스템은 인간의 선호에 영향을 미치는 편향을 탐지하고 정량화하여, 더 공정한 모델 비교를 가능하게 합니다.

- **Performance Highlights**: 새로운 모델의 경우 인간 평가 비용을 최대 41%까지 절감할 수 있으며, 새로운 작업에서는 최대 77%까지 비용을 줄일 수 있습니다. Polyrating는 또한 다양한 작업 간 평가를 직접 비교할 수 있게 하여, LLM의 강점, 약점 및 여러 응용 분야에서의 상대적 성능을 포괄적으로 이해하는 데 도움을 줍니다.



### Correcting FLORES Evaluation Dataset for Four African Languages (https://arxiv.org/abs/2409.00626)
- **What's New**: 이 논문은 네 개의 아프리카 언어(하우사, 노던 소토, 시송가 및 이지줄라)에 대한 FLORES 평가 데이터셋의 수정 사항을 설명합니다. 원본 데이터셋은 저자원 언어에 대한 혁신적인 범위를 제공했지만 여러 불일치와 부정확성을 보여 이를 검토한 원주율의 데이터 평가의 무결성을 해칠 가능성이 있었습니다.

- **Technical Details**: 본 연구는 원어민에 의한 철저한 검토 과정을 통해 발견된 오류 수정 사항을 포함하여 네 개의 언어에 대한 오류 요약 및 통계 분석을 제공합니다. 또한 수정된 데이터셋을 향후 평가 작업에 사용할 수 있도록 제공합니다.

- **Performance Highlights**: 수정된 데이터셋은 언어적 정확성과 신뢰성을 향상시켜 네 개 아프리카 언어를 포함한 NLP(Natural Language Processing) 작업의 더욱 효과적인 평가에 기여합니다.



### Entity-Aware Biaffine Attention Model for Improved Constituent Parsing with Reduced Entity Violations (https://arxiv.org/abs/2409.00625)
- **What's New**: 본 논문에서는 entity-aware biaffine attention 모델을 제안하여 constituency parsing에서 entity-violating 문제를 해결하고자 합니다. 기존 모델들이 entity 완전성을 간과하는 반면, 제안된 모델은 entity 정보를 효과적으로 활용합니다.

- **Technical Details**: 제안된 모델은 biaffine attention 메커니즘을 바탕으로 하며, entity role vector를 추가하여 구문 분석의 정확도를 향상시킵니다. 새로운 메트릭인 Entity Violating Rate (EVR)를 도입하여 구문 분석 결과의 entity 위반 정도를 정량화합니다.

- **Performance Highlights**: 세 가지 데이터셋 (ONTONOTES, PTB, CTB)에서 실험한 결과, 제안된 모델은 가장 낮은 EVR을 기록하면서도 기존 모델과 비교하여 높은 precision, recall, F1-score를 유지했습니다. 추가적으로, 문장 감정 분석과 같은 다운스트림 작업에서도 뛰어난 성능을 보여줍니다.



### Does Knowledge Localization Hold True? Surprising Differences Between Entity and Relation Perspectives in Language Models (https://arxiv.org/abs/2409.00617)
Comments:
          CIKM 2024

- **What's New**: 이번 연구에서는 대형 언어 모델(LLM) 내의 지식이 어떻게 저장되고 관리되는지를 조사하였으며, 개체(entity)와 관계(relational) 지식 간의 차이를 밝혀냈습니다. 이 연구는 지식 편집 모델을 통해 개체와 관계를 변형했을 때 그 결과가 일치하지 않음을 발견했습니다.

- **Technical Details**: 연구는 주로 모델 편집(model editing) 기법을 사용하여 개체와 관계 지식, 즉 지식의 쌍을 다루었습니다. 또한, 인과 분석(causal analysis)을 통해 사전 훈련된 모델에서 관계 지식이 어떻게 저장되는지를 조사했습니다. 실험 결과, 관계 지식은 MLP 가중치뿐만 아니라 Attention 모듈에서도 상당히 인코딩되어 있음이 밝혀졌습니다.

- **Performance Highlights**: 연구의 주요 결과는 기존 지식 평가 방법에 대한 의문을 제기하고, 모델 편집에 있어 새로운 기초를 마련했습니다. 개체와 관계 지식 간의 비대칭적인 저장 방식은 LLM의 지식 표현 및 편집 방법에 심도 있는 함의를 지닙니다.



### DAMe: Personalized Federated Social Event Detection with Dual Aggregation Mechanism (https://arxiv.org/abs/2409.00614)
Comments:
          CIKM 2024

- **What's New**: 이 논문은 사회적 이벤트 감지(Social Event Detection, SED)의 연합 학습(Federated Learning, FL) 접근 방식을 개선하기 위해 개인화된 연합 학습 프레임워크인 DAMe(Dual Aggregation Mechanism)를 제안합니다. 기존의 FL paradigms는 데이터의 이질성 문제를 효과적으로 처리하지 못합니다.

- **Technical Details**: DAMe는 두 가지 집계 메커니즘을 채택합니다: 로컬 집계(및 Bayesian optimization을 사용하여 최적의 집계 가중치를 탐색) 및 글로벌 집계(클라이언트 그래프를 최소화하여 각 클라이언트가 최대한의 외부 지식을 확보하도록 합니다). 글로벌-로컬 정렬(global-local alignment)은 로컬 모델이 글로벌 모델과 일치하도록 하는 제약 조건을 도입합니다.

- **Performance Highlights**: 실제 시뮬레이션을 통해 6개 언어와 2개 소셜 미디어 플랫폼에서 다양한 사회적 이벤트 데이터셋을 사용한 실험 결과, DAMe 프레임워크는 효과성과 회복력(robustness)을 입증했습니다. 또한, 이 프레임워크는 연합 공격으로부터 강한 저항성을 보여줍니다.



### TinyAgent: Function Calling at the Edg (https://arxiv.org/abs/2409.00608)
- **What's New**: 최근 대형 언어 모델(LLMs)의 발전으로 다양한 도구 및 API를 통합하여 사용자 질의를 처리할 수 있는 고급 에이전트 시스템이 개발되었습니다. 그러나 이 LLM을 엣지에서 배포하는 것은 이전에 다뤄지지 않았습니다. 본 논문에서는 TinyAgent라는 작업별 소형 언어 모델 에이전트를 훈련하고 배포할 수 있는 엔드-투-엔드(End-to-End) 프레임워크를 제시합니다.

- **Technical Details**: TinyAgent는 오픈 소스 모델에서 정확한 함수 호출을 가능하게 하는 LLMCompiler 프레임워크를 통해 구체화되었고, 함수 호출을 위한 고품질 데이터셋을 체계적으로 구성하여 TinyAgent-1.1B 및 7B 두 가지 소형 언어 모델을 세밀하게 조정하였습니다. 또한, 입력 프롬프트의 길이를 줄이기 위한 새로운 도구 검색 방법과 양자화를 활용하여 추론 속도를 향상시키는 방법이 소개되었습니다.

- **Performance Highlights**: 모델의 성능 실험 결과, TinyAgent는 GPU-4-Turbo와 같은 대형 모델의 함수 호출 기능을 초월할 수 있으며, 맥북에서 텍스트 또는 음성 입력을 통해 사용자 명령을 실행할 수 있는 지역 Siri와 유사한 시스템을 시연하였습니다. 연구 결과에 따르면, 모델은 엣지에서 완전히 배포 가능한 상태에서 더욱 효율적으로 작업을 수행할 수 있음을 입증하였습니다.



### Automatic Pseudo-Harmful Prompt Generation for Evaluating False Refusals in Large Language Models (https://arxiv.org/abs/2409.00598)
- **What's New**: 이 연구에서는 대규모 언어 모델(LLMs)의 '가짜 유해 프롬프트(pseudo-harmful prompts)'에 대한 첫 번째 자동 생성 방법을 제안합니다. 이를 통해 PHTest라는 새로운 데이터셋을 구축하였고, 이 데이터셋은 기존 데이터셋의 10배 크기로 다양한 잘못된 거부 패턴을 포함하고 있습니다.

- **Technical Details**: 제안된 방법은 컨트롤 가능한 텍스트 생성을 활용하여 유창하고 내용이 통제된 가짜 유해 프롬프트를 생성합니다. 20개의 LLM을 평가하여, false refusal rates (FRRs)와 jailbreak 공격에 대한 안전성 간의 상충 관계를 분석했습니다. 우리는 다양한 '유해하지 않은', '논란이 있는', '유해한' 프롬프트의 정의를 새롭게 제시합니다.

- **Performance Highlights**: 평가 결과, Claude 3는 '유해하지 않은' 가짜 유해 프롬프트에 대해 FRR이 더욱 크게 감소하는 경향이 있었으며, 더 큰 모델이 '유해하지 않은' 프롬프트에서 더 낮은 FRR을 보였습니다. 그러나 많은 jailbreak 방어 방법이 FRR을 상당히 증가시키는 것으로 나타났습니다.



### Learning to Ask: When LLMs Meet Unclear Instruction (https://arxiv.org/abs/2409.00557)
- **What's New**: 본 연구는 실제 사용자 명령의 불완전성 하에서 대형 언어 모델(LLMs)이 도구를 사용하는 성능을 평가하기 위한 새로운 방법론을 제안하고, 새로운 벤치마크 Noisy ToolBench를 구축하였습니다. 또한, LLM들이 불분명한 지시를 마주쳤을 때 사용자의 명확한 답변을 요청하도록 유도하는 Ask-when-Needed (AwN) 프레임워크를 소개합니다.

- **Technical Details**: Noisy ToolBench는 명확하지 않은 사용자 명령의 모호함을 감지하고 관련 질문을 제기할 수 있는 LLM의 능력을 측정하기 위한 벤치마크입니다. 또한, ToolEvaluator라는 자동 평가 시스템을 설계하여 LLM의 도구 사용 성능을 효율적으로 평가합니다. 연구는 LLM의 훈련 목표와 명령어 실행 시 발생할 수 있는 주요 문제를 분석하고, 이에 대한 솔루션으로 AwN 메소드를 제안합니다.

- **Performance Highlights**: AwN 프레임워크는 NoisyToolBench에서 기존 방법보다 도구 학습 성능이 크게 향상되었습니다. 이 연구의 주요 기여는 적절한 명확화 질문을 요청하는 LLM의 능력, 올바른 함수 호출을 실행할 수 있는 능력, 그리고 사용자 요구를 충족하는 최종 응답을 제공할 수 있는 성공률을 기반으로 한 새로운 평가 지표를 개발한 것입니다.



### Testing and Evaluation of Large Language Models: Correctness, Non-Toxicity, and Fairness (https://arxiv.org/abs/2409.00551)
Comments:
          PhD Thesis

- **What's New**: 이 논문은 대형 언어 모델(LLMs)인 ChatGPT의 신뢰성, 비독성(non-toxicity), 공정성(fairness)을 자동화 소프트웨어 테스트와 자연어 처리 관점에서 연구한 탐색적인 작업을 소개합니다. 특히 사실성(correctness) 평가를 위한 두 가지 테스트 프레임워크인 FactChecker 및 LogicAsker를 도입하였습니다.

- **Technical Details**: FactChecker는 대규모 지식 기반에서 사실 삼중항(triple)을 활용하여 사실 성능을 평가하고, LogicAsker는 기본 원리에 따라 자연어로 논리 추론 문제를 생성하는 테스트 프레임워크입니다. 비독성을 위해 MTTM이라는 텍스트 내용 감사 소프트웨어의 취약점을 정의하고, 언어의 다양성을 고려한 멀티링구얼 안전 기준 XSafety를 개발하였습니다. 공정성 평가를 위해 BiasAsker 및 XCulturalBench와 같은 새로운 평가 프레임워크를 소개합니다.

- **Performance Highlights**: 이 연구는 최신 LLMs(예: ChatGPT)의 사실 적절성과 논리 추론 능력을 향상시키는 데 기여하였으며, Multilingual 안전성 향상 및 사회적, 문화적 편견을 측정하는 효과적 방법론을 제시하여 AI 모델의 전반적인 신뢰성을 높이는 결과를 도출하였습니다.



### Large Language Models-Enabled Digital Twins for Precision Medicine in Rare Gynecological Tumors (https://arxiv.org/abs/2409.00544)
Comments:
          20 pages, 2 figures, 3 tables, supplements, original article

- **What's New**: 이번 연구에서는 Rare Gynecological Tumors (RGT) 환자를 위한 디지털 트윈 시스템을 구축하였으며, 대규모 언어 모델(LLMs)을 활용하여 개인 맞춤형 치료 전략을 제안하는 새로운 접근 방식을 선보였습니다.

- **Technical Details**: 디지털 트윈 시스템은 21개의 기관 사례와 655개의 문헌원을 통해 얻은 데이터(총 404,265명의 환자)를 통합하여, 전이성 자궁 육종 환자를 위한 맞춤형 치료 계획을 마련하는 데 초점을 맞추었습니다. LLM을 통해 전자 건강 기록(EHR)와 논문 데이터에서 정보 추출 및 구조화를 진행하여, RGT 디지털 트윈 시스템을 구축하였습니다.

- **Performance Highlights**: 이 연구의 결과로 얻어진 맞춤형 치료 옵션은 전통적인 단일 데이터 출처 분석에서는 발견되지 않은 추가적인 치료 가능성을 제시하였으며, 생물학 중심의 종양 정의로의 전환을 통해 RGT 관리를 개선하고 환자 결과를 향상시킬 수 있는 잠재력을 지니고 있습니다.



### Post-OCR Text Correction for Bulgarian Historical Documents (https://arxiv.org/abs/2409.00527)
Comments:
          Accepted for publication in the International Journal on Digital Libraries

- **What's New**: 이 연구는 역사적 불가리아 문서의 OCR (Optical Character Recognition) 텍스트 수정 평가를 위한 첫 번째 벤치마크 데이터셋을 생성하였습니다.

- **Technical Details**: 19세기 Drinov orthography에 기반하여 불가리아 문서에서 OCR 출력을 수정하는 방법을 개발하고, 현대 불가리아 문학 텍스트를 활용하여 합성 데이터를 자동으로 생성하는 방법론을 연구합니다. 또한, 최첨단 LLMs (Large Language Models)와 인코더-디코더 프레임워크를 사용하고, 대각선 주의 손실(diagonal attention loss) 및 복사 및 커버리지 메커니즘을 보강하여 텍스트 수정을 향상시키고 있습니다.

- **Performance Highlights**: 제안된 방법은 인식 과정에서 발생하는 오류를 줄여 문서 품질을 25% 향상시키며, 이는 ICDAR 2019 불가리아 데이터셋의 최신 기술 대비 16% 증가한 결과입니다.



### LongRecipe: Recipe for Efficient Long Context Generalization in Large Language Models (https://arxiv.org/abs/2409.00509)
Comments:
          Work in Progress

- **What's New**: 이 논문은 LLM(대형 언어 모델)의 긴 문맥 처리 능력을 강화하기 위한 새로운 훈련 전략인 LongRecipe를 소개합니다. 기존 방법들은 긴 문맥을 처리하는 데 있어 시간과 자원 소모가 크지만, LongRecipe는 효율적인 훈련 방식을 통해 이를 해결하고 있습니다.

- **Technical Details**: LongRecipe는 효과적인 token 분석, position index 변환 및 훈련 최적화 전략들을 통해 LLM의 문맥 윈도우를 효과적으로 확대합니다. 이 방법은 기존의 긴 문맥 훈련 방식과 비교하여 훈련 효율성을 유지하면서 긴 시퀀스 입력을 시뮬레이션할 수 있습니다.

- **Performance Highlights**: LongRecipe는 세 가지 다양한 LLM에 대한 실험에서 LLM의 효과적인 문맥 윈도우 크기를 8k에서 128k로 확장할 수 있으며, 이 과정에서 GPU 자원을 85% 이상 절감하면서도 GPT-4에 가까운 성능을 달성할 수 있음을 증명했습니다.



### With Good MT There is No Need For End-to-End: A Case for Translate-then-Summarize Cross-lingual Summarization (https://arxiv.org/abs/2409.00414)
- **What's New**: 이번 연구는 크로스링구얼 요약(Cross-lingual Summarization, CLS)을 위한 두 가지 접근 방식인 파이프라인 시스템(pipeline system)과 엔드-투-엔드 시스템(end-to-end system)을 비교했습니다. 연구 결과, 단순한 '번역 후 요약(translate-then-summarize)' 방식이 높은 수준의 데이터에 접근한 엔드-투-엔드 시스템을 지속적으로 초월하는 것으로 나타났습니다.

- **Technical Details**: 본 연구는 39개의 소스 언어(source language)에 대해 테스트를 진행했으며, 파이프라인 시스템에서는 동시에 존재하는 강력한 기계 번역 시스템(competitive MT system)을 이용하였습니다. 이를 통해 공개적으로 배포된 BLEU 점수와의 상관관계를 분석하여 특정 언어 쌍의 가능성을 사전에 평가할 수 있음을 보여주었습니다.

- **Performance Highlights**: 결과적으로, CLS 성능은 주로 단일 언어 요약(monolingual summarization) 및 번역 번역 작업의 개별 발전 조합을 통해 더 좋은 성능을 발휘하며, 엔드-투-엔드 시스템은 신중히 고려해야 할 방법으로 제안됩니다.



### Rethinking Backdoor Detection Evaluation for Language Models (https://arxiv.org/abs/2409.00399)
- **What's New**: 이 논문은 언어 모델에 대한 백도어 공격의 보안 위험을 다루고 있으며, 기존의 백도어 감지 방법들이 실제 환경에서 백도어를 견고하게 식별할 수 있는지를 평가합니다. 특히, 백도어 식별기의 강건성을 테스트하기 위해 훈련 강도를 조작하여 기존 방법의 한계를 드러냅니다.

- **Technical Details**: 연구에서는 백도어 이식 데이터에 대한 훈련 강도를 조작하여 공격자가 기존 백도어 감지 접근 방식을 우회할 수 있는 관계를 조사합니다. 이 과정에서 poisoning rate, learning rate, training epochs를 조정하여 모델의 학습 강도를 다양화하고, 결과적으로 기존 메타 분류기(Meta Classifier)의 탐지 정확도를 100%에서 0%로 떨어뜨리는 방법을 발견했습니다.

- **Performance Highlights**: 연구 결과, 공격자가 보다 공격적이거나 보수적인 훈련으로 백도어를 심을 경우, 이는 기존의 감지 방법보다 더 많은 탐지 실패를 초래하며, 이는 현재 백도어 탐지 기술의 강건성 부족 및 현재 벤치마크의 한계를 강조합니다.



### An Empirical Study on Information Extraction using Large Language Models (https://arxiv.org/abs/2409.00369)
Comments:
          This article has an original arxiv version entitled "Is Information Extraction Solved by ChatGPT? An Analysis of Performance, Evaluation Criteria, Robustness and Errors", whose url link is arXiv/2305.14450

- **What's New**: 본 논문은 OpenAI의 최신 모델인 GPT-4를 통해 정보 추출(Information Extraction, IE) 작업의 성능을 평가하고, 기존의 SOTA(State-of-the-Art) 방법들과의 성능 차이를 확인하며, 이를 개선하기 위한 새로운 프롬프트 기반(proposal-based) 방법들을 제안합니다.

- **Technical Details**: 연구에서는 성능(Performance), 평가 기준(Evaluation Criteria), 견고성(Robustness), 에러 유형(Error Types)의 네 가지 측면에서 GPT-4의 IE 능력을 평가했습니다. 이 과정에서 16개의 데이터셋과 14개의 IE 하위 작업을 적용해 zero-shot, few-shot ICL( In-Context Learning), 그리고 few-shot COT(Chain-of-Thought) 프롬프트 설정에서 성능을 비교했으며, GPT-3.5와의 성능 비교를 통해 향상된 성능을 확인했습니다. 또한, 프롬프트 디자인 방법으로는 'Task-related Knowledge Informing', 'Methodology Specifying', 'Sufficient Extraction Reminder'를 제안했습니다.

- **Performance Highlights**: GPT-4는 대부분의 작업에서 GPT-3.5보다 나은 성능을 보였으나, SOTA 방법들과는 여전히 성능 격차가 존재했으며, 더욱 어려운 작업일수록 격차가 커지는 경향을 보였습니다. 수동 검토를 통해 확인한 바에 따르면, GPT 모델들은 주로 'Missing spans'와 'Unannotated spans'의 에러를 많이 발생시키며, 이는 전체 에러의 60% 이상을 차지했습니다.



### Predicting the Target Word of Game-playing Conversations using a Low-Rank Dialect Adapter for Decoder Models (https://arxiv.org/abs/2409.00358)
Comments:
          6 pages, 3 Figures, 5 Tables

- **What's New**: 이 논문은 디코더 모델에 대한 방언(adapters)을 도입하는 LoRDD(저랭크 방언 강인성 모델) 아키텍처를 제안하여 NLU(Natural Language Understanding) 작업에서 특정 방언의 성능을 향상시킵니다.

- **Technical Details**: LoRDD는 두 가지 LoRA 기반(adapter) 어댭터를 결합하여 작업 어댑터와 방언 어댑터를 구현합니다. 작업 어댑터는 지침 미세 조정을 사용하고, 방언 어댑터는 Pseudo-parallel 대화 코퍼스에서 대조 학습을 수행합니다. 이 연구는 MD-3 데이터셋을 사용하여 en-IN과 en-US 간의 관계를 학습합니다.

- **Performance Highlights**: LoRDD는 Task Word Prediction(TWP) 작업에서 Mistral 및 Gemma 모델에 대해 넷 베이스라인보다 우수한 성능을 보였으며, en-US에 비해 단어 유사성에서 12%, 정확도에서 25%의 성능 격차를 줄였습니다.



### YA-TA: Towards Personalized Question-Answering Teaching Assistants using Instructor-Student Dual Retrieval-augmented Knowledge Fusion (https://arxiv.org/abs/2409.00355)
Comments:
          9 pages, 5 figures

- **What's New**: YA-TA는 강사와 학생 양측의 개인화된 지원을 제공하는 첫 번째 다중 턴 질문-답변(Question-Answering) 에이전트로, Dual Retrieval-augmented Knowledge Fusion(DRAKE) 프레임워크를 통해 두 가지 정보를 통합하여 대답을 생성합니다.

- **Technical Details**: DRAKE 프레임워크는 두 가지 단계를 포함합니다: 1) Dual Retrieval 단계에서 강사와 학생의 지식을 동시 수집하고, 2) Knowledge Fusion 단계에서 집합된 정보를 바탕으로 개인 맞춤형 대답을 생성하는 것입니다. LLMs의 Chain-of-Thought 능력을 활용하여 정보를 통합합니다.

- **Performance Highlights**: 실험 결과, DRAKE 프레임워크는 강사 및 학생의 지식이 잘 일치하는 응답을 생성하는 데 유의미한 향상을 보였으며, Q&A 보드와 자가 학습 도구와 같은 추가 확장을 통해 학생들의 학습 경험을 더욱 풍부하게 만듭니다.



### Does Alignment Tuning Really Break LLMs' Internal Confidence? (https://arxiv.org/abs/2409.00352)
- **What's New**: 이 논문은 대형 언어 모델(Large Language Models, LLMs)의 적용 시 필수적인 신뢰성 교정(calibration)의 중요성을 다루며, LLM을 평가하기 위한 교정 저하(calibration degradation) 분석을 모델, 메트릭, 작업 및 신뢰도 추출 방법 등 네 가지 차원에서 수행하였습니다.

- **Technical Details**: 저자들은 다양한 오픈소스 LLM(Llama2, Llama3, Mistral 등)을 대상으로 Expected Calibration Error (ECE)와 Static Calibration Error (SCE)와 같은 메트릭을 사용하여 교정 오류를 평가했습니다. 그들은 로그잇(logit) 기반의 신뢰도 추출 방법인 continuation-sum, continuation-min, choice 방법을 도입하여 차이를 분석합니다.

- **Performance Highlights**: 결과적으로 alignment 과정이 LLM의 교정에 미치는 영향이 복잡하다는 사실이 발견되었고, 특히 choice 방법이 모든 작업 및 모델 조합에서 긍정적인 SCE 변화를 보였습니다. 이는 교정 메트릭과 신뢰도 추출 방법을 결합할 경우, alignment 과정의 영향을 고려할 때 모든 모델에서 교정이 일관되게 저하된다는 점을 강조합니다.



### Evaluating the Effectiveness of Large Language Models in Representing and Understanding Movement Trajectories (https://arxiv.org/abs/2409.00335)
Comments:
          7 pages, 3 figures

- **What's New**: 이번 연구는 AI 기초 모델의 움직임 궤적(trajectory) 표현 능력을 평가하는 데 초점을 두고 있습니다. 대규모 언어 모델인 GPT-J를 활용하여 궤적의 문자열 형식을 인코딩하고, 이 LLM 기반 표현의 효과를 궤적 데이터 분석에서 평가합니다.

- **Technical Details**: 본 연구는 GPS 로거를 사용하여 수집한 GeoLife 데이터셋을 기반으로 궤적 거리 측정과 위치 예측 작업을 포함하는 두 가지 주요 작업을 설정합니다. LLM인 GPT-J는 궤적의 고차원 표현을 학습하는 인코더 역할을 하며, 효과적인 프롬프트 및 미세 조정 기법을 통해 다양한 작업을 수행할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과는 GPT-J 임베딩에서 파생된 코사인 거리와 하우스도르프(Hausdorff)와 동적 시간 왜곡(DTW) 거리 간의 상관 계수가 0.74를 초과하는 것으로 나타났습니다. 또한, LLM은 궤적 분석에서 위치 예측 작업에 대한 좋은 정확도를 보였습니다.



### WikiCausal: Corpus and Evaluation Framework for Causal Knowledge Graph Construction (https://arxiv.org/abs/2409.00331)
Comments:
          Extended version; poster paper accepted at ISWC 2024

- **What's New**: 최근 일반 도메인 및 도메인 특정 인과 지식 그래프(conceptual knowledge graphs) 구축에 대한 관심이 높아지고 있습니다. 이 지식 그래프는 인과 분석(causal analysis) 및 이벤트 예측(event prediction)을 위한 추론을 가능하게 하며, 여러 분야에 걸쳐 다양한 응용을 제공합니다. 이번 논문에서는 인과 지식 그래프 자동 구축을 위한 데이터셋(corpus), 작업(task), 및 평가 프레임워크(evaluation framework)를 제안합니다.

- **Technical Details**: 제안된 데이터셋은 이벤트 관련 개념이 포함된 위키백과 문서로 구성됩니다. 이 작업의 목표는 코퍼스에서 이벤트 개념 간의 인과 관계(causal relations)를 추출하는 것입니다. 이러한 인과 관계의 품질을 평가하기 위해, 기존의 Wikidata의 인과 관계를 사용하여 회수를 평가하고, Large Language Models(LLMs)를 활용하여 수작업 평가 없이 정밀도를 측정합니다. 평가 프레임워크는 공개적으로 제공되며, 이는 텍스트 코퍼스를 입력으로 받아 인과 관계의 지식 그래프를 생성하는 자동화된 솔루션의 품질을 평가하는 데 최초로 사용됩니다.

- **Performance Highlights**: 모듈식 인과 지식 추출 파이프라인을 사용하여 다양한 조합의 사전 훈련된 신경 모델을 이용해 네 가지 버전의 Wikidata 기반 인과 지식 그래프를 생성하고, 모델 선택이 출력 품질에 미치는 영향을 보여주었습니다. 이 프레임워크를 통해 각 작업에 적절한 모델을 효과적으로 찾을 수 있음을 입증했습니다.



### From Prediction to Application: Language Model-based Code Knowledge Tracing with Domain Adaptive Pre-Training and Automatic Feedback System with Pedagogical Prompting for Comprehensive Programming Education (https://arxiv.org/abs/2409.00323)
Comments:
          9 pages, 2 figures

- **What's New**: 새롭고 혁신적인 접근법인 CodeLKT를 소개합니다. 이 시스템은 기존 Knowledge Tracing (KT) 및 Code KT 모델보다 뛰어난 성능을 보여주며, 프로그래밍 교육을 위한 사용자 맞춤형 피드백 생성을 가능하게 합니다.

- **Technical Details**: CodeLKT는 pre-trained language models를 활용하여 학습 데이터를 처리하며, Domain Adaptive Pre-Training (DAPT)과 Task Adaptive Pre-Training (TAPT)의 효용성을 탐구합니다. 또한, 수학과 코딩 영역 간의 지식 전이 가능성을 조사하고, 이를 통해 모델의 일반화 가능성과 교육적 맥락에서의 적용성을 강조합니다.

- **Performance Highlights**: CodeLKT는 기존 KT 모델에 비해 예측 정확도를 크게 향상시키며, 대규모 언어 모델(LLM)과 결합하여 개인화된 피드백을 제공하는 혁신적인 시스템으로 자리 잡고 있습니다. 이 시스템은 학생의 학습 상태를 모니터링하고, 맞춤형 피드백을 통해 프로그래밍 학습을 지원합니다.



### An Empirical Study on Context Length for Open-Domain Dialog Generation (https://arxiv.org/abs/2409.00315)
Comments:
          6 pages, 2 figures, 2 tables

- **What's New**: 이번 연구에서는 Transformer 기반 오픈 도메인 대화 모델의 문맥 길이에 대한 설정을 다루고 있습니다. 문맥의 적절한 길이를 결정하는 기준이 없으며, 이로 인해 모델 성능에 미치는 영향을 실험하였습니다.

- **Technical Details**: 연구는 다양한 문맥 길이에서의 모델 학습 결과를 바탕으로, 문맥 길이가 길수록 모델 훈련에 도움이 되는지, 다른 문맥 길이의 대화에 따라 훈련 문맥 길이를 변경해야 하는지, 그리고 다양한 대화 샘플이 동일한 문맥 길이에 대한 선호도를 가지는지를 파악하는 세 가지 질문을 설정하였습니다.

- **Performance Highlights**: 실험 결과, Transformer 기반 대화 모델에 있어 문맥 길이는 성능과 효율성을 모두 고려할 때 무조건 길다고 해서 좋은 것은 아니며, 가장 성능이 높은 모델이 다양한 역사 길이를 가진 대화에서 잘 수행되므로, 별도의 모델을 훈련할 필요가 없다는 점이 발견되었습니다. 또한, 각 샘플에 대해 특정 문맥 길이를 고려하는 것이 모델 성능을 더욱 향상시킬 수 있다는 결과가 나왔습니다.



### REFFLY: Melody-Constrained Lyrics Editing Mod (https://arxiv.org/abs/2409.00292)
- **What's New**: 본 논문에서는 REFFLY(REvision Framework For Lyrics)이라는 최초의 개정 프레임워크를 소개합니다. 이 프레임워크는 일반 텍스트 초안을 고품질의 전체 노래 가사로 편집하는 것을 목표로 하며, 멜로디와의 정렬을 보장합니다.

- **Technical Details**: REFFLY는 주어진 멜로디에 맞춰 가사를 수정하기 위해 드래프트 텍스트를 노래 가사로 변환하는 메커니즘을 제공합니다. 이 과정에서 훈련 없는 휴리스틱을 활용하여 중요 단어와 음악 노트를 포착하며, 수작업 레이블링된 데이터셋을 제공하여 모델의 가사 품질을 높입니다.

- **Performance Highlights**: REFFLY는 사용자 지정 입력으로부터 가사를 생성하고, 중국어에서 영어로의 가사 번역 실험을 통해 25%의 음악성과 34%의 텍스트 품질 개선을 달성함으로써 강력한 기준 모델들(예: Lyra, GPT-4)보다 우수한 성능을 보였습니다.



### OnlySportsLM: Optimizing Sports-Domain Language Models with SOTA Performance under Billion Parameters (https://arxiv.org/abs/2409.00286)
Comments:
          13 pages, 4 figures, 4 tables

- **What's New**: 이 논문은 스포츠 관련 데이터로만 훈련된 작은 도메인 특화 언어 모델의 가능성을 탐구합니다. OnlySports라는 새로운 데이터 세트 및 벤치마크를 소개하며, 이를 통해 쿼리의 효율성을 극대화하는 방법을 제시합니다.

- **Technical Details**: 우리는 6000억개의 토큰을 포함하는 OnlySports Dataset을 FineWeb에서 추출하였으며, RWKV 아키텍처를 스포츠 관련 태스크에 최적화하여 196M 파라미터(매개변수)를 가진 모델을 설계했습니다. 모델 구조는 20층, 640차원입니다.

- **Performance Highlights**: OnlySportsLM은 이전의 135M 및 360M 모델 대비 각각 37.62% 및 34.08%의 정확도 개선을 달성했으며, SomlLM 1.7B 및 Qwen 1.5B과 같은 대형 모델과의 성능이 동등합니다.



### Simple stochastic processes behind Menzerath's Law (https://arxiv.org/abs/2409.00279)
Comments:
          The paper was presented at QUALICO 2023, Lausanne. This manuscript has been submitted to the proceedings of this conference. Full scale figures: this http URL

- **What's New**: 이번 논문은 Menzerath의 법칙(Menzerath's Law) 또는 Menzerath-Altmann 법칙을 다시 돌아봅니다. 이 법칙은 언어적 구성체의 길이와 그 구성 요소의 평균 길이 사이의 관계를 모델링합니다.

- **Technical Details**: 이 논문에서는 단어가 음절(syllables)과 음소(phonemes)에서 길이를 변경할 수 있다는 기본 원칙을 채택하여 bivariate log-normal distribution을 도출합니다. 이 원칙을 통해 고전적인 Altmann 모델을 얻고, Gaussian copula를 활용하여 독립적으로 결합 분포(joint distribution)를 모델링함으로써 더 정확한 모델을 구축합니다.

- **Performance Highlights**: 모델은 실증적(empirical) 데이터와 비교되며, 기존 모델들이 현실 세계 데이터에 미치지 못하는 부분과 대안적인 접근법들이 논의됩니다.



### Towards a dynamical model of English vowels. Evidence from diphthongisation (https://arxiv.org/abs/2409.00275)
- **What's New**: 본 연구에서는 이중모음(Diphthong)과 단모음(Monophthong) 간의 음성학적 범주를 조명하며, 뚜렷한 구분 없이 동적인 변화를 가지는 이중모음의 특성을 다룹니다.

- **Technical Details**: 연구에서는 6명의 Northern Anglo-English 화자들을 대상으로 실시한 조음 분석(articulometry)과 음향 데이터(acoustic data)를 통해 이중모음이 단모음과는 명확히 구분되지 않음을 분석합니다. Articulatory Phonology/Task Dynamic 모델을 통해 이중모음과 단모음의 공통된 조음(target)을 제시합니다.

- **Performance Highlights**: 연구 결과, 이중모음은 전통적으로 예상되는 것과는 달리 단모음과의 명확한 구분 없이 두 개의 조음 목표를 가지며, 이는 영국 영어의 역사적 이중모음화와 현재의 동적인 모음 변동을 통해 강력히 뒷받침됩니다.



### Finding frames with BERT: A transformer-based approach to generic news frame detection (https://arxiv.org/abs/2409.00272)
Comments:
          16 pages

- **What's New**: 이 논문은 온라인 커뮤니케이션에서 사회적 현실을 강조하는 방식인 framing 분석을 위한 새로운 해결책을 제시합니다. 특히 transformer 기반의 접근 방식을 통해 영어 콘텐츠에서 뉴스 프레임을 자동으로 탐지하는 방법을 소개합니다.

- **Technical Details**: 논문에서는 학습 및 테스트 데이터셋의 구성, 모델 아키텍처, 접근 방식의 검증 등 기술적인 세부 사항을 다룹니다. 이러한 방법론이 framing 분석의 스케일링 및 새로운 연구 영역에 적응하는 데 어떻게 기여할 수 있는지를 설명합니다.

- **Performance Highlights**: 자동화된 뉴스 프레임 탐지의 가능성과 한계를 반영하여, 인공지능 기반 시스템이 사회적으로 중요한 이슈의 표현에 미치는 영향을 연구하는 데 있어 유용한 도구로 자리매김할 수 있는 잠재력을 가집니다.



### Leveraging a Cognitive Model to Measure Subjective Similarity of Human and GPT-4 Written Conten (https://arxiv.org/abs/2409.00269)
Comments:
          7 Figures, 1 table

- **What's New**: 이번 연구에서는 문서 간 유사성을 측정하는 Instance-Based Individualized Similarity (IBIS) 메트릭을 제안합니다. IBIS 메트릭은 개인의 편견을 반영하여, 주관적인 유사성을 보다 정확하게 평가할 수 있도록 설계되었습니다.

- **Technical Details**: IBIS 메트릭은 Instance-Based Learning (IBL) 모델을 활용하여, 인간의 의사 결정 과정을 디지털 쌍둥이(digital twin)처럼 모사합니다. 이 모델은 사용자 개개인의 경험 이력을 바탕으로 유사성을 계산하며, 참여자들이 이메일을 안전하다고 판단하는지 또는 위험하다고 판단하는지를 평가하는 실험을 기반으로 합니다. 연구에서 수집된 데이터셋은 총 39,230개의 인간 판단과 20487개의 인간-모델 간 대화를 포함합니다.

- **Performance Highlights**: 연구 결과, IBIS 메트릭은 전통적인 유사도 측정 방법보다 교육 환경에서 인간 참여자들의 주관적인 유사성을 보다 잘 반영하는 것으로 나타났습니다. 특히, 피싱 이메일을 구분하는 실험에서 효과적인 성과를 보였습니다.



### DiverseDialogue: A Methodology for Designing Chatbots with Human-Like Diversity (https://arxiv.org/abs/2409.00262)
- **What's New**: 이 논문은 인간 사용자와 유사한 대화 시뮬레이션을 통해 챗봇을 평가하는 새로운 접근법을 제안합니다. 특히, GPT-4o mini를 사용하여 다양성이 풍부한 대화를 자동으로 생성하고, 이를 통해 평가의 정확성을 높이는 방법을 설명합니다.

- **Technical Details**: 연구에서 제안하는 DiverseDialog 방법론은 인구 통계학적 특성(예: 나이, 성별), 정서적 톤, 대화 주제를 고려하여 챗봇의 특성을 설정하고, 이를 통해 다양한 사용자 시뮬레이션을 수행합니다. 이 과정에서 Differential Language Analysis와 Linguistic Inquiry and Word Count (LIWC)와 같은 심리학적 언어 패키지를 활용하여 대화의 언어적 특성을 분석합니다.

- **Performance Highlights**: 실험 결과, 인간과 LLM(large language model) 간 대화의 평균 오류가 54% 감소했음을 보여 줍니다. 이는 LLM 챗봇 대화의 인간 유사성을 높이고, 언어적 다양성을 증가시킨 것으로 평가됩니다.



### Pre-Training Multimodal Hallucination Detectors with Corrupted Grounding Data (https://arxiv.org/abs/2409.00238)
- **What's New**: 본 연구에서는 다중 모달 언어 모델(Multimodal Language Models, MLMs)의 환각(hallucination)을 탐지하는 문제를 시퀀스 레이블링(sequence labeling) 작업으로 설정하였습니다. 이는 기존 접근 방식의 한계를 극복하고 환각된 텍스트 구간을 정확히 식별하는 작업을 수행하는 데 중점을 둡니다.

- **Technical Details**: MLM의 성능을 향상시키기 위해, 우리는 오류가 있는 기초 데이터(corrupted grounding data)를 생성하고 이를 프리트레이닝(pre-training) 데이터로 활용합니다. 구체적으로, 구문 기반 데이터(phrase grounding data)를 이용하여 실제로는 시각적으로 부정확하지만 텍스트 컨텍스트에 논리적으로 적합한 환각 문구를 생성합니다. 이 방법은 모델의 샘플 효율(sample efficiency)을 증가시키는 데 기여합니다.

- **Performance Highlights**: 테스트 결과, 프리트레이닝을 통해 최대 +7 F1 점수 개선을 보였으며, 다양한 모델과 데이터 규모에서 샘플 효율이 눈에 띄게 향상되었습니다. 이는 환각 탐지기의 효율성을 크게 향상시키는 것으로 나타났습니다.



### Can Large Language Models Address Open-Target Stance Detection? (https://arxiv.org/abs/2409.00222)
Comments:
          10 pages, currently under submission

- **What's New**: 이 연구는 Open-Target Stance Detection (OTSD)이라는 새로운 작업을 도입하며, 이는 훈련 중에 타겟을 보지 않거나 입력으로 제공되지 않는 방식으로 수행된다. 기존의 Target-Stance Extraction (TSE) 접근법과 비교하여 LLM(대형 언어 모델)의 성능을 평가한다.

- **Technical Details**: OTSD는 제로-샷 학습(zero-shot learning)을 활용하며, 특정 타겟이나 입력 정보를 이용하지 않고 텍스트에서 직접 타겟을 생성하고 그에 대한 스탠스를 감지하는 작업이다. 연구에서는 GPT-3.5, Llama 3, Mistral 등의 LLM을 사용하여 성능을 평가하였다.

- **Performance Highlights**: 연구 결과, LLM은 명시적으로 언급된 타겟에 대한 스탠스 감지에서는 TSE 접근법보다 우수한 성능을 보였으나, 비명시적(non-explicit) 타겟에 대한 감지에서는 부족한 성능을 보였다. LLM의 경우, 주어진 텍스트에 명시적으로 또는 비명시적으로 언급된 타겟에 대한 생성능력에서도 TSE보다 더 나은 결과를 나타냈다.



### ProGRes: Prompted Generative Rescoring on ASR n-Bes (https://arxiv.org/abs/2409.00217)
Comments:
          IEEE Spoken Language Technology Workshop

- **What's New**: 이 논문은 최근의 생성형 지시 조정된 대규모 언어 모델(LLMs)을 사용하여 음성 인식 가설(n-best hypotheses)을 동적으로 확장하는 새로운 방법인 PROmpted Generative REScoring (ProGRes)을 제안합니다.

- **Technical Details**: ProGRes는 confidence scores, LLM sequence scoring 및 prompt 기반 가설 생성을 결합하여 ASR(n-best) 리스코어링을 수행합니다. Llama-3-Instruct, GPT-3.5 Turbo, GPT-4 Turbo를 프롬프트 기반 생성기 및 Llama-3를 시퀀스 스코어링 LLM으로 비교하였습니다. 원하는 전사를 도출하기 위해서는 기존의 ASR 가설을 매우 낮은 확률로 리스코어링하는 것의 한계를 극복해야 합니다.

- **Performance Highlights**: 제안된 방법은 WER(단어 오류율)에서 5%에서 25%까지의 유의미한 상대적 향상을 달성했습니다. ProGRes는 기존의 방법보다 우수한 성능을 보이며 음성 인식 오류를 효과적으로 줄이는 데 기여합니다.



### Enhancing Document-level Argument Extraction with Definition-augmented Heuristic-driven Prompting for LLMs (https://arxiv.org/abs/2409.00214)
- **What's New**: 이번 논문에서는 문서 수준의 Event Argument Extraction (EAE) 성능을 높이기 위해 새로운 Definition-augmented Heuristic-driven Prompting (DHP) 방법을 제안합니다. 이 방법은 인수 추출과 관련된 정의 및 휴리스틱 규칙을 통합하여 추출 과정을 안내하고, 오류 전파를 줄이며 작업 정확도를 향상시킵니다.

- **Technical Details**: DHP 방법은 Argument extraction 정의와 규칙 기반 지식을 결합하여 이벤트 인수 추출 작업의 성능을 개선합니다. 문서 내용, 작업 정의, argument extraction 규칙, 그리고 식별된 이벤트 유형과 트리거를 포함하여 입력을 처리합니다. 또한 Chain-of-Thought (CoT) 방법을 사용하여 복잡한 문제를 관리 가능한 하위 문제로 나누는 방식으로 인간의 추론을 모사합니다.

- **Performance Highlights**: 실험 결과, DHP 방법이 기존의 prompting 방법과 few-shot supervised learning에 비해 성능 향상을 보여주었으며, LLM의 일반화 능력을 강화하고 대규모 주석 데이터셋에 대한 의존도를 줄이는 데 효과적임을 입증했습니다.



### Enhancing Event Reasoning in Large Language Models through Instruction Fine-Tuning with Semantic Causal Graphs (https://arxiv.org/abs/2409.00209)
- **What's New**: 이 논문은 이벤트 감지(event detection) 분야에서 LLM(대형 언어 모델)의 성능을 향상시키기 위한 새로운 접근법을 제안합니다. 이 방법은 이벤트 트리거와 이벤트 유형 간의 인과 관계를 고려하여 LLM을 지시 조정(instruction fine-tuning)하는 Semantic Causal Graphs(SCGs)를 사용합니다.

- **Technical Details**: 제안된 SCG는 텍스트 내의 인과 관계와 맥락 정보를 동시에 포착하는 방향 그래프입니다. 그리고 SCG Instructions는 이벤트 트리거와 이벤트 유형 간의 관계를 강조하여 인과 서브그래프를 추출함으로써 LLM을 fine-tuning하는 방법을 제공합니다. Low-Rank Adaptation(LoRA) 기법을 사용하여 LLM의 일반적인 언어 이해 능력을 유지하면서 이벤트 감지를 향상시킵니다.

- **Performance Highlights**: SCG Instructions로 훈련된 LLM은 이벤트 트리거 분류(Event Trigger Classification)에서 평균 35.69% 더 나은 성능을 보였습니다. 특히, fine-tuning된 Mistral 7B 모델은 이벤트 감지 주요 메트릭에서 GPT-4보다 평균 31.01%(이벤트 트리거 식별), 37.40%(이벤트 트리거 분류), 16.43%(이벤트 분류) 더 높은 성능을 기록했습니다. 일반적인 능력 유지에서 평균 2.03점의 미미한 성능 저하만 관찰되었습니다.



### The creative psychometric item generator: a framework for item generation and validation using large language models (https://arxiv.org/abs/2409.00202)
Comments:
          CREAI 2024

- **What's New**: 본 논문은 대형 언어 모델(LLMs)을 활용하여 기존의 창의성 평가 방식을 혁신하는 창의적 측정 항목 생성기(CPIG)를 개발했다는 점에서 새롭다. 이 프레임워크는 창의적 문제 해결(CPS) 과제를 위한 새로운 테스트 항목을 자동으로 생성하여 인간과 AI의 창의성을 보다 정확하고 효과적으로 평가할 수 있도록 한다.

- **Technical Details**: CPIG는 LLM 기반의 항목 생성기와 평가자를 혼합하여 사용하여 항목을 반복적으로 개발하는 구조적인 프로세스를 사용한다. 이 과정에서 생성된 항목들은 복잡한 시나리오를 기술하며, 다양하고 창의적인 응답을 유도할 수 있도록 설계된다. CPIG는 창의성 측정의 정확성과 신뢰성을 유지하기 위해 심리측정학적 요약 및 예제 선택에 기반하여 작동한다.

- **Performance Highlights**: 실험 결과, CPIG가 생성한 항목들은 인간이 작성한 항목과 동일한 유효성과 신뢰성을 가지고 있으며, LLM 솔루션은 반복될수록 더 창의적인 결과를 도출하는 것으로 나타났다. 이는 CPIG가 발전할수록 생성 AI의 창의성을 증가시킬 수 있는 방법이 될 수 있다는 것을 보여준다.



### Facilitating phenotyping from clinical texts: the medkit library (https://arxiv.org/abs/2409.00164)
- **What's New**: 이 논문에서는 EHR(전자 건강 기록)에서 특정한 특성이나 상태와 연관된 개인을 식별하기 위해 알고리즘을 적용하는 과정을 보다 용이하게 만드는 오픈 소스 파이썬 라이브러리인 medkit을 소개하고 있습니다. medkit은 데이터 처리 파이프라인을 쉽게 구성할 수 있게 해주는 소프트웨어 블록을 제공합니다.

- **Technical Details**: medkit은 세 가지 기본 클래스—Documents, Annotations, Attributes—로 데이터를 관리합니다. 각 클래스는 데이터와 메타데이터를 표현하는 속성과 방법을 가지고 있으며, 데이터 처리 과정의 주요 클래스는 Operations와 Pipelines입니다. Operations는 데이터를 입력받아 기능을 수행하고 출력을 반환하는 구조를 가지고 있습니다. 이 라이브러리는 비파괴적 처리(non-destructive processing)와 데이터 출처 추적(provenance tracing)을 지원하여 변환 단계를 거친 후에도 정보의 손실을 방지합니다.

- **Performance Highlights**: medkit을 통해 개발된 여러 파이프라인은 약물 치료 정보 추출, 부정 및 가정 감지, 의료 텍스트의 고속 주석 작업 등을 지원합니다. 예를 들어, 화학요법 독성과 관련된 현상 또는 COVID-19의 식별을 위한 파이프라인이 포함되어 있습니다. 이는 다양한 NLP 도구의 재사용 및 체인을 통한 성능 평가를 돕습니다.



### Sequence to Sequence Reward Modeling: Improving RLHF by Language Feedback (https://arxiv.org/abs/2409.00162)
Comments:
          7 pages

- **What's New**: 이 논문에서는 대형 언어 모델 (LLMs)의 행동을 인간의 의도와 가치에 맞추기 위한 새로운 접근 방식인 	extit{sequence-to-sequence (seq2seq) reward modeling} 방법을 제안합니다. 기존의 보상 모델링 방식에서 이진 최대 우도 추정 (MLE)을 시퀀스 MLE로 대체하여 언어 피드백을 학습함으로써 RLHF의 정확성과 세분성을 향상시킵니다.

- **Technical Details**: 제안된 seq2seq 보상 모델링은 두 가지 주요 단계, 즉 보상 모델링과 보상 추출로 구성됩니다. 이 방법은 각 토큰이 반응 점수에 미치는 영향을 직접적으로 반영하며, 각각의 토큰에 대한 긍정 및 부정 피드백을 추출하여 RLHF의 세분성을 개선합니다. 추가적인 데이터 주석, 훈련, 모델 없이 이루어지며, 이는 데이터의 효율성을 높이고 과도한 모델링 복잡성을 줄입니다.

- **Performance Highlights**: 실험 결과, seq2seq RM을 적용했을 때 2B 및 7B 파라미터의 LLM에서 3개의 NLP 작업에 대해 평균 76.9%의 개선 효과를 보였습니다. 또한, seq2seq RM은 분포 밖의 프롬프트에서도 RLHF 성능을 개선할 수 있음을 보여주어, 제안된 방법의 유연성과 효과성을 입증하였습니다.



### LLMs hallucinate graphs too: a structural perspectiv (https://arxiv.org/abs/2409.00159)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 허위 정보 생성, 즉 'hallucination'을 구조화된 형태로 연구할 수 있는 가능성을 제시합니다. 특히, 저자들은 특정 문헌에서 알려진 그래프를 요청했을 때의 허위 응답을 분석하여, 이러한 허위 그래프가 LLM의 출력 특성을 묘사하는 데 어떻게 활용될 수 있는지를 탐구합니다.

- **Technical Details**: 연구의 주요 기여는 두 가지입니다. 첫째, 여러 최신 LLM에서의 토폴로지적 hallucination의 다양성을 관찰합니다. 둘째, 이러한 허위 정보의 크기를 측정할 수 있는 지표인 Graph Atlas Distance를 제안합니다. 이 지표는 여러 그래프에서의 평균 그래프 편집 거리로 정의됩니다. 또한, Hallucination Leaderboard와 비교하여 10,000배 더 많은 프롬프트를 활용한 랭킹을 제공합니다.

- **Performance Highlights**: 연구에서는 미스트랄, Vercel AI SDK, HuggingChat, ChatGPT, together.ai, Google의 Gemini 등 21개의 LLM을 대상으로 비교 분석을 진행했습니다. Zachary의 카라테 클럽 그래프와 Les Misérables 그래프 등에서 허위 응답을 평가하였으며, 각 LLM의 출력 그래프의 품질을 6가지 주요 통계로 검토했습니다.



### Developing an End-to-End Framework for Predicting the Social Communication Severity Scores of Children with Autism Spectrum Disorder (https://arxiv.org/abs/2409.00158)
Comments:
          Accepted for Interspeech 2024

- **What's New**: 이번 연구는 ASD 아동의 사회적 커뮤니케이션 심각도를 자동으로 예측하기 위한 최신 E2E(End-to-End) 프레임워크를 제안합니다. 이 프레임워크는 자동 음성 인식(Automatic Speech Recognition, ASR) 모델과 미세 조정된 사전 훈련 언어 모델(Pre-trained Language Models, PLMs)을 통합하여 ABC 아동의 speech 데이터에서 심각도 예측 점수를 도출합니다.

- **Technical Details**: 프레임워크는 wav2vec2-xls-r-300m과 whisper-large-v2 두 가지 다국어 ASR 모델을 사용하며, ASD 아동의 speech 데이터로 미세 조정됩니다. 이어서 KR-BERT, KLUE/roberta-base, KR-ELECTRA-Discriminator와 같은 세 가지 PLM을 미세 조정하여 예측 점수를 생성합니다. 이 과정에서 전통적인 미세 조정, 수동 프롬프트, p-tuning 접근 방식이 적용됩니다.

- **Performance Highlights**: 평균적으로, 이 시스템은 인간 평가자 점수와의 Pearson 상관 계수 0.6566을 달성하여 기존 진단 도구와 비교할 수 있는 가능성을 보여줍니다. 이 연구는 ASD 진단을 위한 접근 가능하고 객관적인 도구 개발에 기여할 것으로 기대됩니다.



### Speaker Tagging Correction With Non-Autoregressive Language Models (https://arxiv.org/abs/2409.00151)
Comments:
          6 pages, 7 tables

- **What's New**: 이 논문은 자동 음성 인식(ASR) 시스템과 스피커 다이얼리제이션(SD) 시스템의 출력을 결합하여 대화 중 누가 언제 발언했는지를 파악하는 작업의 중요성을 강조합니다. 특히, 스피커 구분이 필요한 자연 대화의 품질을 높이기 위해 비자기적 언어 모델을 기반으로 한 이중 단계 스피커 태깅 수정 시스템을 제안합니다.

- **Technical Details**: 제안된 시스템은 두 개의 데이터세트인 TAL과 Fisher 데이터 세트에서 단어 다이얼리제이션 오류율(WDER)을 감소시키는 성과를 보였으며, 잘못된 발언 경계를 수정하는 데 중점을 두었습니다. 이 시스템은 음성 세그멘테이션과 스피커 태깅의 오류를 분석하고 분류하는 과정을 포함하고 있습니다.

- **Performance Highlights**: 실험에서 제안된 스피커 오류 수정 모듈은 Fisher 테스트 세트와 TAL 데이터 세트에서 기존 방법에 비해 cpWER에서 유의미한 개선을 보여주었습니다, 이는 스피커 다이얼리제이션의 성능을 크게 향상시키는 데 기여합니다.



### MultiMath: Bridging Visual and Mathematical Reasoning for Large Language Models (https://arxiv.org/abs/2409.00147)
- **What's New**: 이 논문은 시각적인 입력과 통합된 수학적 추론의 필요성을 강조하며, 이를 위해 다중 모달 대규모 언어 모델인 MultiMath-7B를 제안합니다. 본 모델은 MultiMath-300K라는 새로운 다중 모달 수학 데이터셋에 기반하여 다양한 수학적 문제가 포함되어 있습니다.

- **Technical Details**: MultiMath-7B는 4단계 훈련 과정을 통해 개발되며, 시각-언어 정렬 (vision-language alignment), 시각 및 수학 지침 조정 (instruction-tuning)과 프로세스 감독 강화 학습 (process-supervised reinforcement learning)에 중점을 둡니다. 이는 DeepSeekMathRL-7B를 기반으로 하여 시각 인코더와 다중 모달 어댑터를 추가한 모델입니다.

- **Performance Highlights**: MultiMath-7B는 기존의 다중 모달 수학 벤치마크에서 SOTA(State-of-the-Art) 성능을 달성하며, 텍스트 전용 수학 벤치마크에서도 뛰어난 성능을 보여줍니다. 특히, 다중 모달 훈련이 특정 텍스트 전용 수학 문제에 대한 성능 향상을 가져온 결과를 보였습니다.



### Dynamic Depth Decoding: Faster Speculative Decoding for LLMs (https://arxiv.org/abs/2409.00142)
- **What's New**: 본 논문에서는 Dynamic Depth Decoding (DDD)를 소개하며, 이는 EAGLE-2의 디코딩 알고리즘을 최적화하여 현재의 최첨단 speculative decoding 방법의 속도를 향상시킵니다.

- **Technical Details**: DDD는 EAGLE-2의 트리 초안 생성 방법을 동적인 깊이로 최적화합니다. 이는 EAGLE-2가 EAGLE에 대해 달성하는 평균 속도 향상을 44% 연장하여 DDD는 평균 3.16배의 속도 향상을 제공합니다. DDD는 드래프트 모델의 신뢰도를 토대로 드래프트 생성을 계속할지 결정하기 위해 확률 합계를 휴리스틱으로 사용합니다.

- **Performance Highlights**: 실험 결과, DDD는 EAGLE-2에 비해 평균 4% 향상을 보여주며, EAGLE-2는 EAGLE에 비해 평균 8% 향상된 성능을 제시합니다. 모든 실험에서 정확도는 측정되지 않았고, 모든 속도 향상 방법은 손실이 없었습니다.



### PrivacyLens: Evaluating Privacy Norm Awareness of Language Models in Action (https://arxiv.org/abs/2409.00138)
Comments:
          Under review

- **What's New**: 이 논문은 언어 모델(LMs)의 프라이버시 인식 수준을 정량화하고, LM이 매개된 커뮤니케이션에서 발생하는 프라이버시 위험을 평가하기 위한 새로운 프레임워크인 PrivacyLens를 제안합니다.

- **Technical Details**: PrivacyLens는 프라이버시 민감한 상황을 표현하는 비네트를 통해 에이전트의 행동에서 프라이버시 유출을 다단계로 평가할 수 있도록 설계되어 있습니다. 이 프레임워크는 프라이버시 문헌에 기반한 프라이버시 규범의 집합과 크라우드소싱된 씨드를 활용해 구체화되었습니다.

- **Performance Highlights**: 최신 LM인 GPT-4와 Llama-3-70B는 프라이버시를 강화하는 지침에도 불구하고 각각 25.68%와 38.69%의 경우에 민감한 정보를 유출하였습니다. 이는 LM의 성능과 실제 사용자 지시를 수행할 때의 행동 간 간극을 드러냅니다.



### HoneyComb: A Flexible LLM-Based Agent System for Materials Scienc (https://arxiv.org/abs/2409.00135)
Comments:
          Under Review on EMNLP 2024

- **What's New**: HoneyComb은 재료 과학을 위해 특별히 설계된 최초의 LLM(large language model) 기반 에이전트 시스템으로, 기존의 LLM이 직면한 문제들을 해결하기 위해 새로운 지식 기반과 도구 허브를 활용합니다.

- **Technical Details**: HoneyComb은 MatSciKB라는 고품질의 재료 과학 지식 기반과 Inductive Tool Construction 방법을 통해 생성된 ToolHub를 통합하여 재료 과학 관련 과제의 정확성과 효율성을 향상시킵니다. 또한, Retriever 모듈을 활용하여 특정 작업에 적합한 지식 출처나 도구를 적응적으로 선택합니다.

- **Performance Highlights**: HoneyComb은 다양한 재료 과학 과제에서 기존 모델보다 현저히 뛰어난 성과를 보이며, LLM의 일반적 능력과 재료 과학의 전문적 요구 사이의 격차를 효과적으로 좁히고 있습니다.



### A Survey for Large Language Models in Biomedicin (https://arxiv.org/abs/2409.00133)
- **What's New**: 현대의 거대 언어 모델(LLMs)이 이루어진 최근의 발전은 전례 없는 자연어 이해 및 생성 능력을 제공하고 있습니다. 본 리뷰는 생명 의학 분야의 LLMs에 대한 종합적인 분석을 제공하며, 특정 응용 프로그램이나 모델 아키텍처에 국한되는 기존 조사와는 차별화됩니다.

- **Technical Details**: 이 리뷰는 484개의 출처를 분석하여 LLM의 현재 상태, 응용 프로그램, 도전 과제 및 향후 전망을 탐구합니다. zero-shot learning 능력을 포함하여 진단 보조, 약물 발견 및 개인 맞춤형 의학과 같은 다양한 생명 의학 작업의 사례를 분석합니다. 또한, uni-modal 및 multi-modal LLM의 fine-tuning 방법과 데이터 보호 및 윤리적 과제를 포함한 생명 의학 분야의 도전 과제를 다룹니다.

- **Performance Highlights**: MedPaLM은 92.9%의 일치를 기록하였으며, scBERT는 단일 세포 유전체 데이터 분석을 향상시키기 위한 임베딩을 생성합니다. LLMs의 적용이 빠르게 증가하고 있으며, 2021년부터 출판물이 급증하고 있습니다. LLM의 신뢰성과 특정성 향상을 위한 특별한 훈련 및 최적화가 필요합니다.



### Logic Contrastive Reasoning with Lightweight Large Language Model for Math Word Problems (https://arxiv.org/abs/2409.00131)
- **What's New**: 이번 연구에서는 경량 대형 언어 모델(Large Language Models, LLMs)의 수학적 추론 과제에서 성능을 향상시키기 위한 새로운 방법을 소개합니다.

- **Technical Details**: 수학적 논리 유사성(mathematical logic similarity)을 측정하는 새로운 방법을 도입하고, 의미적(semantic) 및 논리적(logical) 유사성을 통합한 참조 문제(reference problems) 집합을 구성하기 위한 자동 스크리닝 메커니즘을 설계했습니다. 긍정 및 부정 예시 프롬프트(prompts)를 활용하여 모델이 건전한 추론 로직(sound reasoning logic)을 채택하도록 유도합니다.

- **Performance Highlights**: SVAMP 데이터셋에서 Chain of Thought 접근법 대비 15.8% 향상, GSM8K 데이터셋에서는 21.5% 향상을 기록했습니다. 175억 매개변수(parameter)를 가진 대규모 모델에 이 방법을 적용하면 두 데이터셋에서 최고의 결과와 비교되는 성능을 나타냈습니다. 또한, 추론 과정 중 발생하는 오류에 대한 분석을 통해 미래 연구에 중요한 통찰을 제공했습니다.



### Can AI Replace Human Subjects? A Large-Scale Replication of Psychological Experiments with LLMs (https://arxiv.org/abs/2409.00128)
Comments:
          5 figures, 2 tables

- **What's New**: 이번 연구에서는 최신 Large Language Model인 GPT-4가 사회 과학 분야의 154개의 심리 실험을 복제하여 인간의 반응을 얼마나 잘 모사할 수 있는지를 평가하였습니다.

- **Technical Details**: 연구는 618개의 주요 효과(main effects)와 138개의 상호작용 효과(interaction effects)를 대상으로 하였으며, GPT-4가 이 실험들에서 76.0%의 주요 효과와 47.0%의 상호작용 효과를 효과적으로 재현하였음을 발견하였습니다.

- **Performance Highlights**: 하지만, GPT-4가 재현한 신뢰 구간(confidence intervals) 중 19.44%만이 원본 효과 크기와 일치하였고, 71.6%에서 원래 연구에서 보고된 null findings와 반대되는 유의미한 결과가 나타났습니다. 이는 LLM의 연구 도구로서의 가능성을 보여주지만, AI 기반의 결과 해석 시 주의가 필요함을 강조합니다.



### ConCSE: Unified Contrastive Learning and Augmentation for Code-Switched Embeddings (https://arxiv.org/abs/2409.00120)
Comments:
          ICPR 2024

- **What's New**: 이 논문은 영어와 한국어 간의 Code-Switching (CS) 현상을 연구하며, CS 데이터셋의 필요성을 강조합니다. 특히, 기존의 Equivalence Constraint (EC) 이론이 영어-Korean CS 복잡성을 부분적으로만 설명한다고 주장합니다. 저자들은 이를 해결하기 위해 Koglish 데이터셋을 새롭게 제안합니다.

- **Technical Details**: Koglish 데이터셋은 Koglish-GLUE, Koglish-NLI, Koglish-STS를 포함하며, 영어와 한국어 간의 CS 상황을 다룹니다. 연구는 SimCSE 모델을 기반으로 한 제안된 ConCSE를 통해 CS 문장을 모델링하고, 세 가지 새로운 손실 함수를 도입하여 문장 표현 학습을 진행합니다.

- **Performance Highlights**: 실험 결과, ConCSE 방법은 Koglish-STS 데이터셋에서 SimCSE 대비 평균 1.77% 성능 향상을 보여주며, Koglish 데이터셋의 효과성을 검증했습니다. 여러 NLP 작업을 대상으로 한 비교 실험에서도 Koglish 데이터셋이 다른 다국어 모델에 비해 CS 환경에서의 성능 향상을 이루는 데 기여함을 보여줍니다.



### FedMCP: Parameter-Efficient Federated Learning with Model-Contrastive Personalization (https://arxiv.org/abs/2409.00116)
- **What's New**: 이 논문은 데이터 프라이버시 문제를 해결하기 위해 제안된 FedMCP 방법을 통해 파라미터 효율적인 방법으로 사전 훈련된 언어 모델(PLMs)을 미세 조정하는 연합 학습(federated learning) 기법을 소개하고 있습니다.

- **Technical Details**: FedMCP는 모델 대조 개인화(model-contrastive personalization) 두 개의 경량 어댑터 모듈(global adapter와 private adapter)을 플랜엠의 정지된 모듈에 추가하고, 클라이언트가 서버에 보낼 때는 오직 global adapter만 전송하여 통신 비용을 최소화합니다.

- **Performance Highlights**: FedMCP는 GLUE 벤치마크에서 6개의 데이터셋을 사용한 실험에서 평균 1.5%의 정확도 향상을 보여주었으며, 모든 PLM을 미세 조정하는 것과 비교해 통신 및 계산 비용을 상당히 줄였습니다.



### When All Options Are Wrong: Evaluating Large Language Model Robustness with Incorrect Multiple-Choice Options (https://arxiv.org/abs/2409.00113)
- **What's New**: 이번 연구는 대형 언어 모델(LLM)들이 정답이 없는 객관식 질문을 탐지하는 제로샷(zero-shot) 능력에 대해 탐구합니다. 이는 교육 평가의 질 향상 측면에서 중요한 요소입니다.

- **Technical Details**: 연구에서는 다양한 LLM 및 질문 세트를 활용하여 실험을 수행하였으며, Llama-3.1-405B 모델이 유효한 정답이 없음을 성공적으로 식별하는 데 두드러진 성과를 보였습니다. 모델들은 temperature 집합을 0으로 설정하고 최대 토큰(max_tokens)을 128로 제한하여 명확한 응답을 유도했습니다.

- **Performance Highlights**: 결과적으로, 정답이 있는 질문과 반대로 정답이 없는 질문 간 성능 차이가 상당했으며, LLM들이 교육 환경에서 잘못된 답변을 포함한 질문을 처리할 때 신중하게 접근해야 함을 나타냅니다.



### Toward Large Language Models as a Therapeutic Tool: Comparing Prompting Techniques to Improve GPT-Delivered Problem-Solving Therapy (https://arxiv.org/abs/2409.00112)
Comments:
          Accepted for AMIA 2024 proceedings

- **What's New**: 이번 연구는 대규모 언어 모델(Large Language Models, LLMs)에서 피드백 기술(prompt engineering)의 효과를 탐구하여, 문제 해결 치료(Problem-Solving Therapy, PST) 세션의 증상 식별 및 평가 단계에서 개인화된 목표 설정을 위한 텍스트 제공 능력을 향상시키는 방안을 제시합니다.

- **Technical Details**: 연구에서는 대규모 언어 모델의 성능을 자동 측정 지표 및 경험이 풍부한 의료 전문가들의 평가를 통해 분석합니다. 문제 해결 치료(PST)의 프로토콜을 따르는 능력을 향상시키기 위한 다양한 프롬프트 기법들을 효과적으로 사용함으로써 모델의 성능을 개선할 수 있으며, 이는 무게 조정 없이 이루어집니다.

- **Performance Highlights**: 모델은 PST 대화 상황에서 인간 치료사와의 비교를 통해 상당한 성능을 보였으며, 다양한 프롬프트 기법을 활용한 결과, 전반적인 질, 일관성, 공감 능력이 향상됨을 확인했습니다. 이는 정신 건강 전문가가 부족한 현 상황에서 LLM이 심리 치료 제공에 기여할 가능성을 넓힐 수 있는 연구입니다.



### Zero-Shot Visual Reasoning by Vision-Language Models: Benchmarking and Analysis (https://arxiv.org/abs/2409.00106)
Comments:
          21 pages

- **What's New**: 이번 연구에서는 VLM(vision-language models)의 제로샷(Zero-shot) 시각적 추론 능력을 체계적으로 평가하기 위해 합성 데이터셋(synthetic datasets)을 활용합니다. 이는 기존의 벤치마크가 세계 지식과 혼합되는 점을 명확히 구분하는 시도를 포함합니다.

- **Technical Details**: CLEVR와 PTR 데이터셋을 사용하여 VLM의 순수 시각적 추론 능력을 평가합니다. VLM 내부의 LLM(large language model)에 시각적 임베딩(visual embeddings) 대신 텍스트 기반 장면 설명(textual scene descriptions)을 제공했을 때 성능이 더 좋다는 결과가 나왔습니다. 또한, 체인 오브 사고(chain-of-thought, CoT) 프롬프트가 표준 프롬프트에 비해 더 큰 모델에서만 효과적이라는 것을 발견했습니다.

- **Performance Highlights**: BLIP2-Flan-T5 모델을 활용할 때, 순수 텍스트 정보만으로 18% 더 높은 정확도를 기록하였으며, GPT-4는 GPT-4V보다 CLEVR에서 약 17% 더 높은 정확도를 보였습니다. 그러나 CoT 프롬프트는 더 작은 모델에서는 성능이 떨어지는 것으로 나타나, 모델의 크기가 증가할수록 CoT가 시각적 추론 능력을 개선할 수 있는 잠재력을 지닌다는 것을 시사합니다.



### Negation Blindness in Large Language Models: Unveiling the NO Syndrome in Image Generation (https://arxiv.org/abs/2409.00105)
Comments:
          15 pages, 7 figures

- **What's New**: 이 논문은 최근의 대형 언어 모델(Foundational Large Language Models, LLMs)의 이미지 생성 능력과 관련된 새로운 한계를 제시합니다. 연구팀은 이러한 한계를 'The NO Syndrome'이라고 명명했습니다.

- **Technical Details**: The NO Syndrome은 LLMs가 'NO'와 관련된 자연어 프롬프트를 올바르게 이해하지 못하는 현상으로, 이는 이미지 생성의 정확도에 영향을 미칩니다. 연구자는 다양한 언어(영어, 힌디어, 불어)의 여러 LLM에 대해 시뮬레이션 실험과 엔트로피 기반 통계 분석을 수행했습니다.

- **Performance Highlights**: GPT-4, Gemini, Copilot 등 모든 테스트된 LLM이 이 NO Syndrome의 영향을 받는 것으로 나타났습니다. 이로 인해 생성된 이미지와 텍스트 응답 간의 일관된 불일치가 관찰되었으며, 이는 현재 LLM의 심각한 결함으로 판단됩니다.



### Nuance Matters: Probing Epistemic Consistency in Causal Reasoning (https://arxiv.org/abs/2409.00103)
Comments:
          20 pages

- **What's New**: 이번 연구는 Large Language Models (LLMs)의 인과적 추론에서의 자기 일관성을 평가하기 위해 'causal epistemic consistency'라는 새로운 개념을 제안합니다. 우리는 LLMs의 성능을 평가하기 위한 새로운 지표들을 도입하고, 21개의 고급 LLM을 대상으로 광범위한 실증 연구를 실시하였습니다.

- **Technical Details**: 연구에서는 인과적 에피스템 일관성을 측정하기 위해 세 가지 주요 지표를 제안합니다: (i) intensity ranking concordance, (ii) cross-group position agreement, (iii) intra-group clustering. 이를 통해 LLM이 생성한 중간 결과의 세부적인 차이를 구분할 수 있는 능력을 평가합니다.

- **Performance Highlights**: 실증 연구 결과, 현재 모델들은 인과적 에피스템 일관성을 유지하는 데 어려움을 겪고 있으며, 특히 GPT-4를 포함한 고급 모델조차도 그 성능에 있어 불만족스러운 결과를 보였습니다. 이 연구는 LLMs가 인과 관계의 미세한 차이를 포착하는 데 있어 문제점을 강조합니다.



### Query-by-Example Keyword Spotting Using Spectral-Temporal Graph Attentive Pooling and Multi-Task Learning (https://arxiv.org/abs/2409.00099)
- **What's New**: 이 논문에서는 기존의 키워드 포착(Keyword Spotting, KWS) 시스템의 한계를 극복하기 위해 사용자 정의 키워드를 인식할 수 있는 새로운 Query-by-Example (QbyE) KWS 시스템을 소개합니다. 이 시스템은 spectral-temporal graph attentive pooling (GAP)과 multi-task learning을 적용하여, 사용자가 원하는 키워드를 정의할 수 있고, 사용자 맞춤형 경험을 제공합니다.

- **Technical Details**: 제안된 KWS 시스템은 LiCoNet, Conformer, ECAPA_TDNN 등 세 가지 서로 다른 네트워크 아키텍처를 채택하여 인코더 모델링을 수행합니다. 이 시스템은 하드웨어 효율적인 구조로, GAP을 통해 informative embeddings를 생성하고 하이브리드 손실 함수를 통해 단어와 음소의 구분력을 높이고, 발화자에 따른 변동성을 줄입니다.

- **Performance Highlights**: 실험 결과, LiCoNet이 Conformer 모델과 비교해 13배 더 효율적으로 비슷한 성능을 달성하였으며(각각 1.98%와 1.63%의 FRR), 제안된 QbyE 프레임워크의 효율성을 입증하였습니다. 이는 사용자가 정의한 키워드의 고유성을 보장하는 맞춤형 KWS 시스템의 가능성을 보여줍니다.



### How to Train Text Summarization Model with Weak Supervisions (https://arxiv.org/abs/2409.00098)
- **What's New**: 이 논문에서는 복잡한 목표를 단순한 작업으로 분해하고, 각 작업에 대한 감독 신호(supervision signals)를 생성하는 방법을 제안합니다. 이 방식은 복잡한 라벨이 없는 상황에서도 모델을 학습할 수 있게 합니다.

- **Technical Details**: 제안된 방법은 주제 기반 요약(topic-based summarization) 작업에 적용이 가능합니다. 목표는 문서와 주제를 기반으로 한 추출 요약을 생성하는 것입니다. 각 데이터 샘플은 주제와 문장을 포함하며, 이를 이진 추출 레이블(binary extractive labels)로 모델링합니다. 감독 신호는 일반 요약 레이블을 사용하여 정보성을 촉진하고, 주제와 문장 간의 관련성을 특정 규칙과 의미적 유사성을 통해 강화합니다.

- **Performance Highlights**: CNN 및 DailyMail 데이터셋에서 실험 결과, 제안된 방법은 주제 기반 추출 요약에서 라벨 없이도 뛰어난 정확성을 달성했습니다.



### Large Language Models for Disease Diagnosis: A Scoping Review (https://arxiv.org/abs/2409.00097)
Comments:
          57 pages

- **What's New**: 최근 인공지능(AI) 분야에서 자동 질병 진단의 중요성이 증가하고 있습니다. 대형 언어 모델(LLMs)의 발전으로 이러한 진단 작업에서의 효율성이 입증되었습니다. 하지만 여전히 해결되지 않은 중요한 연구 질문이 많습니다. 본 논문에서는 질병 진단을 위한 LLM 기반 방법을 포괄적으로 분석하였습니다.

- **Technical Details**: 이 연구는 LLM을 사용한 질병 진단에 관한 기존 연구를 검토하고 다양한 질병 유형, 관련 장기 시스템, 임상 데이터, LLM 기술 및 평가 방법을 분석하였습니다. LLM은 텍스트, 이미지, 비디오, 오디오, 표 형식 데이터 및 시계열 데이터를 포함한 다양한 데이터 모드를 처리합니다. 프롬프트 엔지니어링 기술도 조사하였으며, 하드 프롬프트와 소프트 프롬프트 두 가지 유형으로 나뉘어 설명되었습니다.

- **Performance Highlights**: LLMs는 COVID-19 진단에서 높은 정확도를 보여주었으며, PathChat은 인간 병리학에서 최신 성능을 달성했습니다. 또한, GPT-4는 강박장애를 식별할 때 정신 건강 전문가를 초월하는 성능을 보였습니다. 그러나 LLM의 성능을 평가하기 위해 적합한 평가 방법이 여전히 연구되어야 합니다.



### Non-instructional Fine-tuning: Enabling Instruction-Following Capabilities in Pre-trained Language Models without Instruction-Following Data (https://arxiv.org/abs/2409.00096)
Comments:
          16 pages, 2 figures, 15 tables

- **What's New**: 본 논문에서는 비지침적 데이터(non-instructional data)를 이용하여 LLM(대형 언어 모델)의 지침을 따르는 능력을 향상시킬 수 있다는 새로운 접근 방식을 제안합니다. 연구팀은 OpenWebText의 무작위 텍스트 절반을 지침으로 사용하고, GPT-3.5-turbo 또는 GPT-4-turbo를 활용하여 나머지 텍스트를 완료하는 방식을 적용했습니다. 이 연구는 수많은 프리 트레인(pre-trained) LLM을 Fine-tuning하여 이 결과를 확인했습니다.

- **Technical Details**: 연구에서 제안한 방법론은 비지침적 데이터셋을 생성하기 위한 간단한 프레임워크를 제공하며, 조건부 증류(conditional distillation) 및 연속 작성을 통한 지식 증류(knowledge distillation)를 이용하여 데이터를 생성합니다. 이를 통해 LLM의 성능을 높이고, 다양한 벤치마크에서 우수한 결과를 도출합니다.

- **Performance Highlights**: LLaMA-3-70B-Instruct가 Arena Hard 리더보드에서 LLaMA-3.1-70B-Instruct와 동등한 성능을 보였으며, Meta-Llama-3-70b-Instruct 모델은 57.0의 최고 기록을 달성하여, 기존의 SFT(supervised fine-tuning) 데이터셋을 초월했습니다. 이러한 결과는 비지침적 데이터를 통한 Fine-tuning의 효과를 확증합니다.



### Examining Independence in Ensemble Sentiment Analysis: A Study on the Limits of Large Language Models Using the Condorcet Jury Theorem (https://arxiv.org/abs/2409.00094)
- **What's New**: 본 연구는 Condorcet Jury 이론을 감정 분석(sentiment analysis) 분야에 적용하여 다양한 대형 언어 모델(LLM)과 단순 자연어 처리(NLP) 모델의 성능을 비교합니다. 실험 결과, 대형 모델을 사용했음에도 불구하고 성능 향상이 미미하다는 것을 발견했습니다.

- **Technical Details**: 본 연구에서는 Condorcet Jury 이론을 확장하여 다중 클래스 분류에 대한 이론적 프레임워크를 제공합니다. 연구는 LLM과 다른 NLP 모델 간의 독립성이 제한적이라는 가정을 기반으로, 다수결(classifier)을 통한 감정 분석의 정확성을 평가합니다.

- **Performance Highlights**: 대형 언어 모델(예: ChatGPT 4) 사용 시 성능 향상은 미미하며, 간단한 모델들 (예: FinBERT, DistilRoBERTa)보다 큰 개선이 없음을 보여주었습니다. 이 결과는 대형 모델들이 감정 분석 과제에서 충분히 독립적이지 않다는 점을 시사합니다.



### PatentGPT: A Large Language Model for Patent Drafting Using Knowledge-based Fine-tuning Method (https://arxiv.org/abs/2409.00092)
Comments:
          21 pages, 4 figures

- **What's New**: 본 연구는 지식 정밀 조정(Knowledge Fine-Tuning, KFT) 프레임워크를 통해 대형 언어 모델(Large Language Models, LLMs)의 전문 지식과 맥락 인식을 강화하여, AI가 자율적으로 특허 문서를 생성할 수 있도록 하는 혁신적인 접근 방식을 제안합니다.

- **Technical Details**: 제안된 모델인 PatentGPT는 지식 그래프 기반의 사전 학습(pre-training), 도메인 특화된 감독 학습(supervised fine-tuning, SFT), 인간 피드백에서의 강화 학습(Reinforcement Learning From Human Feedback, RLHF)을 결합하여 개발되었습니다.

- **Performance Highlights**: PatentGPT는 특허 관련 기준 벤치마크 테스트에서 최신 모델보다 약 400% 높은 성능을 기록하며, 인간의 창의성과 혁신을 보조하고 증대시키는 능력을 가지고 있음을 입증했습니다.



### Classification of Safety Events at Nuclear Sites using Large Language Models (https://arxiv.org/abs/2409.00091)
- **What's New**: 이 논문은 원자력 발전소의 Station Condition Records (SCRs)를 안전 관련 및 비안전 관련 카테고리로 분류하기 위한 대형 언어 모델 (Large Language Model, LLM) 기반의 기계 학습 분류기 개발을 제안합니다. 기존의 수동 검토 프로세스를 보완하여 안전 분류 프로세스의 효율성과 정확성을 향상시키는 것이 주요 목표입니다.

- **Technical Details**: 이 논문에서는 레이블이 부착된 SCR 데이터셋을 분류하기 위한 실험을 수행하고 분류기의 성능을 평가합니다. 여러 가지 프롬프트 변형 (prompt variations) 구성을 탐구하고 이들이 LLM의 의사결정 프로세스에 미치는 영향을 관찰했습니다. 또한, SCR 안전 분류에 대한 보다 미세하고 유연한 접근 방식을 제공할 수 있는 숫자 점수 메커니즘을 도입했습니다.

- **Performance Highlights**: 이 방법은 원자력 안전 관리에 혁신적인 단계를 나타내며, 안전 이벤트 식별을 위한 확장 가능한 도구를 제공함으로써 안전 분류 프로세스를 개선하는 데 기여합니다.



### Evaluating ChatGPT on Nuclear Domain-Specific Data (https://arxiv.org/abs/2409.00090)
- **What's New**: 이번 논문은 ChatGPT와 같은 대형 언어 모델(LLM)의 핵심 분야인 핵 데이터에서 질문-응답(Q&A) 작업에 대한 적용을 검토합니다.

- **Technical Details**: 논문은 LLM의 응답을 직접적으로 얻는 방식과 RAG(리트리벌 증강 생성) 프레임워크 내에서 LLM의 응답을 비교하여 평가합니다. RAG는 외부 지식 기반과 복잡한 검색 기법을 통합하여 출력의 정확성과 관련성을 향상시키는 방법입니다.

- **Performance Highlights**: 연구 결과, RAG 파이프라인을 포함시킬 경우 LLM의 성능이 향상되며, 특히 핵 분야의 특정 질문에 대해 더 정확하고 적절한 응답을 생성하는 데 기여한다고 합니다.



### On-Device Language Models: A Comprehensive Review (https://arxiv.org/abs/2409.00088)
Comments:
          38 pages, 6 figures

- **What's New**: 이번 논문은 대형 언어 모델(LLMs)의 크기가 커짐에 따라, 이를 엣지 장치에서 배포하는 문제와 그에 대한 혁신적인 해결책을 탐구합니다. 엣지 디바이스에서 LLM을 운영하면 응답 시간이 줄어들고 데이터 로컬라이제이션(data localization) 및 사용자 맞춤화된 경험을 가능하게 합니다.

- **Technical Details**: 엘지테 기계의 제약을 극복하기 위한 다양한 방법들, 즉 효율적인 아키텍처(architectures)와 파라미터 공유(parameter sharing), 모듈형 설계(modular designs), 그리고 양자화(quantization), 프루닝(pruning), 지식 증류(knowledge distillation)와 같은 최신 압축 기술(compression techniques)에 대해 논의합니다.

- **Performance Highlights**: 모바일 제조업체들의 실제 사례 연구를 통해 엣지 장치에서 LLM의 실제 적용 사례를 보여주며, 성능과 자원 활용(resource utilization) 간의 균형을 강조합니다. 또한 적응 학습(adaptive learning), 다중 모달 능력(multi-modal capabilities), 개인화(personalization) 같은 중요 요소들도 다룹니다.



### Genetic Approach to Mitigate Hallucination in Generative IR (https://arxiv.org/abs/2409.00085)
Comments:
          Gen-IR@SIGIR 2024

- **What's New**: 새로운 연구는 Generative language models가 Hallucination(환각)을 발생시키는 문제를 해결하기 위해 Grounded Answer Generation(지속적인 답변 생성) 접근 방식을 개선하여 accuracy(정확도)를 4배 증가시키는 새로운 'balanced fitness function'(균형 잡힌 적합도 함수)을 실제로 적용하고 있습니다.

- **Technical Details**: 이 방법은 'cross-encoder model'(교차 인코더 모델)과 'n-gram overlap metric'(n-그램 중첩 메트릭)을 활용하여 relevance(관련성)와 grounding(기반 데이터의 근거여부)를 유지합니다. GAuGE(Genetic Approach using Grounded Evolution)라는 이름으로 알려진 이 방법론은 첫 단계에서 BM25(정보 검색 알고리즘)를 사용하여 문서를 검색한 후, Electra(교차 인코더 모델)를 활용하여 재정렬합니다.

- **Performance Highlights**: GAuGE는 세 가지 데이터셋을 사용하여 평가하였으며, hallucination을 감소시키면서도 높은 relevance를 유지하며, 여러 개의 seed document(기초 문서)를 사용하여 포괄적인 답변을 생성합니다. GAuGE는 특히 최소한의 hallucination으로 사실 결과를 생성하는 데 성공하였습니다.



### Vision-Language and Large Language Model Performance in Gastroenterology: GPT, Claude, Llama, Phi, Mistral, Gemma, and Quantized Models (https://arxiv.org/abs/2409.00084)
Comments:
          Manuscript Pages: 34, Figures: 7, Tables: 2, Supplementary File Pages: 35, Data Transparency Statement: Code is available at: this https URL . Study data from American College of Gastroenterology (ACG) are restricted and available upon request with ACG permission. Correction: updated abstract considering Llama3.1 results

- **What's New**: 이 연구는 대형 언어 모델(LLMs)과 비전 언어 모델(VLMs)의 위장관학(gastroenterology) 분야에서의 의료 추론 성능을 평가한 최초의 체계적인 분석으로, 다양한 모델 설정과 파라미터, 프롬프트 엔지니어링 전략의 영향을 조사했습니다.

- **Technical Details**: 300개의 위장관학 보드 시험 스타일의 객관식 질문을 사용하였으며, 138개 질문에 이미지를 포함시켰습니다. GPT-3.5를 활용하여 다양한 상업용 및 오픈 소스 LLM(버전), 인터페이스(웹 및 API), 컴퓨팅 환경(클라우드 및 로컬), 모델 정밀도(양자화 유무)에 대해 성능을 평가했습니다.

- **Performance Highlights**: 상업용 모델 중에서는 GPT-4o(73.7%)와 Claude3.5-Sonnet(74.0%)가 최고 정확도를 기록하였고, 오픈 소스 모델에서는 Llama3.1-405b(64%)가 가장 높은 성능을 보였습니다. 특히, 이미지가 포함된 질문에서 VLM의 성능이 저하되는 현상이 관찰되었고, 인간이 제작한 이미지 설명이 포함되었을 때는 10%의 정확도 증가가 있었습니다.



### Towards Human-Level Understanding of Complex Process Engineering Schematics: A Pedagogical, Introspective Multi-Agent Framework for Open-Domain Question Answering (https://arxiv.org/abs/2409.00082)
Comments:
          Our paper is accepted for publication at ML4CCE workshop at ECML PKDD 2024

- **What's New**: 이 논문에서는 프로세스 흐름도(Process Flow Diagrams, PFD)와 배관 및 계기 다이어그램(Piping and Instrumentation Diagrams, P&ID)의 분석을 위한 안전한 기업 솔루션을 제안합니다. 제안된 솔루션은 다계층의 다중 에이전트 Retrieval Augmented Generation (RAG) 프레임워크를 이용하여 개방형 질문 응답(open-domain question answering) 작업을 수행하며 데이터 프라이버시 및 비용 효율성을 증대시킵니다.

- **Technical Details**: 제안된 다중 에이전트 프레임워크는 전문화된 하위 에이전트와 introspective(내성적) 메타 에이전트로 구성됩니다. 각 하위 에이전트는 PFD와 P&ID 분석을 위해 ReAct(Reason + Act) 촉진 기법을 사용하여 정보를 통합합니다. 또한, 사용자 쿼리에 대해 태스크 플래닝을 통해 명령 및 월등한 응답 생성을 위한 외부 API와의 상호작용을 조절합니다.

- **Performance Highlights**: 실험 결과, 제안한 접근법이 이미지 캡셔닝, 시각 질문 응답(VQA), 텍스트 검출(OCR) 작업 등에서 기존의 최신 기술과 동등한 성능을 보여주었으며, 맞춤화, 해석 가능성, 데이터 프라이버시 및 비용 효율성을 제공합니다.



### Are LLM-based methods good enough for detecting unfair terms of service? (https://arxiv.org/abs/2409.00077)
- **What's New**: 이 연구에서는 사용자들이 웹사이트와 앱에서 체결하는 서비스 이용 약관(Terms of Service, ToS) 및 개인 정보 보호 정책을 효율적으로 이해할 수 있도록 대형 언어 모델(LLM)이 어떻게 사용될 수 있는지를 조사했습니다.

- **Technical Details**: 연구진은 220개의 인기 웹사이트에서 수집한 개인 정보 보호 정책에 대한 12개의 질문으로 구성된 'ToS-Busters'라는 데이터셋을 구축했습니다. 그리고 다양한 오픈소스 및 상업용 챗봇에 질의를 하여 그들의 답변을 비교했습니다. 챗봇은 챗GPT, SVM을 포함한 텍스트 생성 방법을 사용했습니다.

- **Performance Highlights**: 결과적으로, 상업적인 챗봇인 ChatGPT4가 가장 높은 성능을 기록했지만, 일부 오픈소스 모델이 ChatGPT3보다 더 높은 정확도를 제공할 수 있었습니다. 전체적으로 모든 모델이 이 작업에서 랜덤 전략보다 약간 더 나은 성능을 보였지만, 실제로 사용되기 위해서는 성능이 크게 향상되어야 합니다.



### Generative-Adversarial Networks for Low-Resource Language Data Augmentation in Machine Translation (https://arxiv.org/abs/2409.00071)
Comments:
          8 pages, 4 figures, 4 tables, presented at ICNLP 2024, to be published in IEEE Explore

- **What's New**: 이번 연구는 제한된 자원으로 된 언어들에서 데이터 증대(data augmentation)을 위한 생성적 적대 신경망(Generative Adversarial Network, GAN)의 가능성을 탐구하였습니다. 저자들은 이전의 연구들과는 달리, 고자원 언어와 저자원 언어 간의 학습 전이 대신, 새로운 문장을 생성하여 저자원 언어 데이터의 양을 늘릴 수 있는 방법을 제시했습니다.

- **Technical Details**: 모델은 인코더-디코더 구조를 채택하며, 첫 번째 단계에서는 인간이 만든 병렬 코퍼스를 통해 초훈련(pre-training)을 진행합니다. 두 번째 단계에서는 GAN을 훈련하여, 생성기(generator)가 무작위 노이즈를 입력받아 이를 잠재 공간 임베딩(latent space embeddings)으로 변환합니다. 마지막으로 생성된 임베딩은 디코더에서 저자원 언어의 문장으로 디코딩되어 새로운 단일언어 데이터 코퍼스를 형성합니다.

- **Performance Highlights**: 이 모델은 20,000 문장 이하의 매우 적은 양의 언어 데이터에서 훈련되었고, 저자원 언어에 대한 데이터 증대에서 잠재적인 가능성을 보여주었습니다. 생성된 문장 예시는 "ask me that healthy lunch im cooking up"과 "my grandfather work harder than your grandfather before"로, 저자원 NMT의 성능 향상에 기여할 수 있음을 제안합니다.



### Learning to Plan Long-Term for Language Modeling (https://arxiv.org/abs/2409.00070)
Comments:
          preprint

- **What's New**: 이 논문에서는 미래 텍스트 예측의 효율성을 높이기 위해 다단계(planner) 접근 방식을 제안합니다. 기존의 단일 단계 계획 기능을 확장하여 더 나은 텍스트 예측을 가능하게 하는 계획 수립 과정을 도입합니다.

- **Technical Details**: 이 방법은 다음 세 단계로 이루어져 있습니다: 1) unlabeled (라벨 없는) 텍스트에서 추론한 행동(action) 시퀀스를 생성, 2) 다단계 (multi-step) planner를 훈련하여 다음 행동을 예측, 3) planner에서 샘플링된 여러 경로를 사용하여 언어 모델을 조건화합니다. 이를 통해 효과적인 예측 코딩이 가능해집니다.

- **Performance Highlights**: 실험 결과, 기존 방법과 비교하여 다단계 계획을 통해 예측 정확도가 향상됨을 확인할 수 있었습니다. 특히, 다단계 예측은 코딩 작업에서 최대 17%의 성능 개선을 보여줍니다.



### An alternative formulation of attention pooling function in translation (https://arxiv.org/abs/2409.00068)
- **What's New**: 이 논문은 번역 과제에서 주의 점수 함수의 대안적 수식을 제시합니다. 언어는 복잡하게 구조화 되어 있으며, 이는 주의 점수 행렬에도 반영됩니다. 이 속성을 활용하여 주의 풀링 함수(attention pooling function)를 정의하였습니다.

- **Technical Details**: 수학적으로, 주의 점수 행렬을 고정 대역폭(bandwidth)을 가진 대역 행렬(band matrices) 공간으로 투영하는 방식으로 수식을 고려합니다. 이 새로운 공간은 대역 행렬과 오차 희소 행렬(error sparse matrices)로 구성된 콤팩트한 하위 공간(comapact subspace)을 증명하였으며, 이 하위 공간은 주의 점수 행렬로부터 최상의 근사값을 보장합니다. 다양한 매개변수(w, num-pos)의 영향을 조사하여 언어 처리 및 번역의 깊은 통찰력을 제공합니다.

- **Performance Highlights**: 새로운 수식이 원래의 주의 점수 수식을 얼마나 잘 근사하는지 계산하여 검증하였고, 이는 문장에서의 단어 관련성과 맥락의 역할에 대한 미묘한 통찰을 제공합니다.



### Phrasing for UX: Enhancing Information Engagement through Computational Linguistics and Creative Analytics (https://arxiv.org/abs/2409.00064)
- **What's New**: 이번 연구는 디지털 플랫폼에서 텍스트 특징과 정보 참여(Information Engagement, IE) 간의 관계를 탐구합니다. 이 연구는 사용자의 상호작용에 대한 계산언어학(computational linguistics)과 분석(analytics)의 영향을 강조합니다.

- **Technical Details**:  연구에서 제안된 READ 모델은 대표성(representativeness), 사용 용이성(ease of use), 정서(affect), 분포(distribution)와 같은 주요 예측 변수들을 정량화하여 참여 수준을 예측합니다. AB 테스트 및 무작위 시험(randomized trials)을 통해 모델의 효과가 검증되었습니다.

- **Performance Highlights**: 참여(participation) 정확도 0.94, 인식(perception) 정확도 0.85, 지속성(perseverance) 정확도 0.81, 그리고 전체 정보 참여(IE) 정확도는 0.97에 이릅니다. 연구 결과에 따르면, READ 모델의 인사이트에 따라 텍스트를 수정하면 대표성과 긍정적 정서가 증가해 선택률이 11% 상승하고, 평균 평가가 3.98에서 4.46으로 증가하며, 유지율도 11% 개선됩니다.



### Enhancing Natural Language Inference Performance with Knowledge Graph for COVID-19 Automated Fact-Checking in Indonesian Languag (https://arxiv.org/abs/2409.00061)
- **What's New**: 이 연구는 COVID-19 관련 허위 정보 검증을 위한 자동화된 사실 확인 시스템의 성능을 높이기 위해 Knowledge Graph (KG)를 활용한 새로운 접근 방식을 제안합니다.

- **Technical Details**: 제안된 모델은 세 가지 모듈로 구성됩니다: fact module (사실 모듈), NLI module (자연어 추론 모듈), classifier module (분류기 모듈). 사실 모듈은 KG에서 정보를 처리하고, NLI 모듈은 주어진 전제와 가설 간의 의미적 관계를 처리합니다. 두 모듈에서 생성된 표현 벡터는 결합되어 분류기 모듈에 입력됩니다.

- **Performance Highlights**: 모델은 인도네시아어로 생성된 COVID-19 사실 확인 데이터셋과 COVID-19 KG Bahasa Indonesia를 사용하여 훈련되었습니다. KG를 통합함으로써 NLI 성능이 크게 향상되며, 최고 정확도 0.8616을 달성하였습니다.



### Understanding Literary Texts by LLMs: A Case Study of Ancient Chinese Poetry (https://arxiv.org/abs/2409.00060)
- **What's New**: 이 연구는 대형 언어 모델(LLM)을 이용하여 고전 중국 시가(詩歌)를 분석하고, 이를 통해 AI 문학 작품의 평가 방식과 품질 향상을 제안하는 새로운 접근 방식을 소개합니다.

- **Technical Details**: 대형 언어 모델을 기반으로 하는 문학 이해 프레임워크를 개발하였으며, 세 가지 주요 단계를 포함합니다: 1) 다양한 고전 시가 작품을 수집하고 전문가의 주석을 추가함. 2) LLM을 미세 조정하고, 이를 통해 다양한 측정 지표를 계산함. 3) 수집된 시가의 패턴을 정리하고, 이를 기반으로 모델 훈련 및 생성 과정을 최적화함.

- **Performance Highlights**: 연구 결과, LLM을 이용해 고전 중국 시가의 정량적 평가가 가능함을 보여주며, 미래의 시가 창작 작업 품질을 향상시킬 수 있는 유용한 패턴을 발견했습니다. 이 과정에서 사용된 적은 양의 전문가 주석 데이터는 평가의 필요성을 해결하는 데 효과적인 방법으로 확인되었습니다.



### Automating Knowledge Discovery from Scientific Literature via LLMs: A Dual-Agent Approach with Progressive Ontology Prompting (https://arxiv.org/abs/2409.00054)
Comments:
          in submission

- **What's New**: 이 논문에서는 대량의 문헌에서의 지식 발견 자동화 문제를 해결하기 위해 Large Language Models (LLMs)를 기반으로 한 혁신적인 프레임워크인 LLM-Duo를 소개합니다. 이는 Progressive Ontology Prompting (POP) 알고리즘과 Dual-Agent System을 결합해 지식 추출 자동화를 향상시키는 것입니다.

- **Technical Details**: POP 알고리즘은 미리 정의된 온톨로지를 기준으로 우선순위가 부여된 breadth-first search (BFS)를 활용하여 구조화된 프롬프트 템플릿 및 행동 순서를 생성합니다. LLM-Duo는 '탐색자'와 '평가자'라는 두 개의 LLM 에이전트를 통해 협력적이고 경쟁적으로 작업하여 발견 및 주석 프로세스의 신뢰성을 높입니다.

- **Performance Highlights**: 본 논문에서 제시한 방법은 64,177개의 연구 논문에서 2,421개의 언어개입(intervention)을 발견하였으며, 이를 바탕으로 스피치-언어 치료 커뮤니티에 기여할 수 있는 공개적인 지식 베이스를 구성했습니다. 실험 결과, 제안하는 방법은 기존의 고급 기준을 초과하는 정확하고 완전한 주석을 가능하게 하였습니다.



### Foundations of Large Language Model Compression -- Part 1: Weight Quantization (https://arxiv.org/abs/2409.02026)
Comments:
          Preprint

- **What's New**: 이번 논문에서는 자원 제약(Device)에 맞춰 대형 언어 모델(LLM)을 압축하는 문제의 중요성을 강조하며, 기존 방법보다 성능이 우수한 정량화(quantization) 방법인 CVXQ를 제안합니다.

- **Technical Details**: LLM 정량화의 기초를 볼록 최적화(convex optimization) 관점에서 제시하며, CVXQ 프레임워크는 수백억 개의 가중치 매개변수를 포함하는 모델에 스케일이 가능하고, 훈련 후 모델을 원하는 크기로 압축할 수 있는 유연성을 제공합니다.

- **Performance Highlights**: CVXQ는 기존의 방법들보다 더 나은 성능을 보여주며, 대형 AI 인프라의 환경적 부담을 줄이는 데 기여할 수 있습니다.



### 3D-LEX v1.0: 3D Lexicons for American Sign Language and Sign Language of the Netherlands (https://arxiv.org/abs/2409.01901)
- **What's New**: 본 연구는 3D에서 수화(SL)를 효율적으로 캡처하기 위한 접근법을 제시하고, 3D-LEX v1.0 데이터세트를 도입하며, 음성 특성을 반자동으로 주석 처리하는 방법을 세부적으로 설명합니다. 이 과정에서는 고해상도의 3D 포즈, 3D 손 모양, 깊이 인지 얼굴 특징을 포함한 세 가지 모션 캡처 기술이 통합되어 있습니다.

- **Technical Details**: 3D-LEX 데이터세트는 미국 수어(ASL)에서 1,000개의 수화와 네덜란드 수어(NGT)에서 1,000개의 수화로 구성되어 있습니다. 이 데이터세트는 수어의 수동 및 비수동 마커를 캡처하기 위한 두 가지 데이터 수집 기술과 비수동 마커를 캡처하는 세 번째 기술로 기록되었습니다. 손 모양 주석을 생성하기 위해 간단한 방법이 제시되었습니다.

- **Performance Highlights**: 주석을 통해 글로스 인식 정확도를 5% 향상시키고, 전문가 주석 대비 1% 향상된 결과를 보였습니다. 여기에 3D 모션 캡처 데이터는 수화 특징에 대한 심층 분석을 지원하고, 모든 시점에서 2D 프로젝션을 생성할 수 있습니다.



### The Role of Large Language Models in Musicology: Are We Ready to Trust the Machines? (https://arxiv.org/abs/2409.01864)
- **What's New**: 이 논문은 음악학(musicology) 분야에서 대형 언어 모델(Large Language Models, LLMs)의 활용 가능성과 신뢰성을 탐구합니다. 연구자들은 LLM의 현황, 우려 사항 및 전문가와의 논의를 통해 이 기술의 aceptación(acceptance)을 평가했습니다. LLM의 신뢰성을 높이기 위해, retrieval-augmented generation 모델을 활용하여 초기 벤치마크를 반자동으로 생성하는 방법을 제안하고, human experts에 의해 검증된 400개의 질문 데이터를 분석하였습니다.

- **Technical Details**: 본 연구는 LLM의 음악학 관련 작업에서의 도메인 전문성을 측정하기 위한 방법론을 제안합니다. 특히, The New Grove Dictionary of Music and Musicians를 기반으로 한 Multiple-Choice Question Generation 접근 방식을 채택하여, 음악 관련 주제에 대한 400개의 질문-답변 쌍을 생성하고 검증하였습니다. 이를 통해 LLM의 음악학 관련 생성 능력을 평가하고, Hallucination(환각) 문제의 정도를 측정합니다.

- **Performance Highlights**: 최종적으로, 생성된 질문의 정확도는 67.4%로 나타났으며, 이는 LLM이 음악학 분야에서 신뢰할 수 있는 텍스트를 생성하는 데 한계가 있음을 보여줍니다. 연구 결과, 대부분의 전문가들은 LLM이 음악학 분야에서 중요한 혁신을 일으킬 가능성이 있다고 보고하였으나, 현재의 LLM 사용 빈도와 신뢰도는 여전히 낮은 상황임을 지적하였습니다.



### Towards Generative Class Prompt Learning for Few-shot Visual Recognition (https://arxiv.org/abs/2409.01835)
Comments:
          Accepted at BMVC 2024

- **What's New**: 본 논문에서는 Generative Class Prompt Learning (GCPL)와 Contrastive Multi-class Prompt Learning (CoMPLe)라는 두 가지 새로운 방법을 제안합니다. 이러한 방법들은 시각-언어적 동기화를 개선하는 데 큰 기여를 하며, 기존의 이미지 인식 방법보다 나은 성능을 보여줍니다.

- **Technical Details**: GCPL은 텍스트-이미지 diffusion 모델을 활용하여 learnable class prompts를 사용하여 few-shot exemplars에 조건화함으로써 클래스 임베딩에서 시각-언어적 시너지를 상당히 개선합니다. CoMPLe는 이 기초 위에서 생성 최적화 과정 중 클래스 간 분리를 독려하는 대조 학습(contrastive learning) 요소를 추가합니다.

- **Performance Highlights**: 실험 결과, 제안된 generative class prompt learning 접근 방식이 기존의 방법보다 현저하게 우수한 성능을 발휘하며, few shot 이미지 인식 문제에 대한 더 나은 대안을 제공합니다.



### LASP: Surveying the State-of-the-Art in Large Language Model-Assisted AI Planning (https://arxiv.org/abs/2409.01806)
- **What's New**: 이 논문은 LLM(대규모 언어 모델)을 활용한 AI 계획의 현재 도전 과제를 살펴보고, 이러한 모델들이 실제 계획 문제를 해결하는 데 어떻게 기여할 수 있는지를 소개합니다.

- **Technical Details**: 기존의 AI 계획 접근 방식은 제한된 도메인에 국한되어 있으며, LLM은 이러한 문제를 해결하기 위한 새로운 프레임워크를 제시합니다. 특히 Planning Domain Definition Language (PDDL)를 활용하여 계획 시스템을 정의하고, 다양한 벤치마크 데이터를 통해 LLM의 계획 능력을 평가합니다.

- **Performance Highlights**: LLM의 계획 능력은 자연어 처리(NLP) 연구자들에게 큰 혜택을 줄 수 있으며, LLM을 계획 프레임워크에 통합한 성공 사례를 정리하여 향후 연구 방향 및 개선 기회를 제시합니다.



### FC-KAN: Function Combinations in Kolmogorov-Arnold Networks (https://arxiv.org/abs/2409.01763)
Comments:
          9 pages, 1 figure

- **What's New**: 이번 논문에서는 FC-KAN(함수 조합을 활용한 Kolmogorov-Arnold 네트워크)을 소개합니다. FC-KAN은 B-splines, wavelets, radial basis functions와 같은 인기 있는 수학적 함수의 조합을 저차원 데이터에서 요소별 연산을 통해 활용하여 성능을 개선합니다.

- **Technical Details**: FC-KAN은 다양한 함수의 출력을 조합하는 여러 방법, 즉 덧셈(sum), 요소별 곱셈(element-wise product), 덧셈과 요소별 곱셈의 조합, 이차 함수 표현(quadratic function representation), 그리고 연결(concatenation)을 탐구합니다. 우리는 메모리 오류를 피하기 위해 3차 함수 사용을 지양하며, 결과적으로 MNIST 및 Fashion-MNIST 데이터셋에서 다른 KAN 네트워크에 비해 더 많은 데이터 특징을 포착하여 성능을 향상시킵니다.

- **Performance Highlights**: FC-KAN의 변형 중 B-splines와 Difference of Gaussians (DoG)의 출력 조합을 이차 함수 형태로 사용하는 것이 5회 독립 훈련 실행의 평균에서 다른 모든 모델보다 우수한 성능을 보였습니다. 이는 FC-KAN의 다양한 함수 조합을 통해 향후 KAN 설계에 대한 기대를 높입니다.



### Empirical evidence of Large Language Model's influence on human spoken communication (https://arxiv.org/abs/2409.01754)
- **What's New**: 이번 연구에서는 ChatGPT와 같은 Large Language Models (LLMs)이 인간의 언어 사용에 미치는 영향을 밝혀냄으로써, 인간 언어의 변화가 AI의 발전에 따라 어떻게 전개되는지를 탐구했습니다. 특히, 280,000개의 영어 발표 및 강연 비디오를 분석하여, ChatGPT의 출시 이후 특정 단어 사용에서 유의미한 변화가 나타났다는 것을 발견했습니다.

- **Technical Details**: 연구는 20,000개의 학술 YouTube 채널에서 약 280,000개의 비디오 전사본을 분석하였으며, ChatGPT의 출시에 따른 단어 사용 추세 변화와 해당 단어 사용의 상관관계를 집중적으로 살펴보았습니다. 주요 연구 결과로는, ChatGPT 출시 후 'delve', 'realm', 'meticulous', 'adept'와 같은 특정 단어의 사용 빈도가 각각 48%, 35%, 40%, 51% 증가한 점이 포함됩니다. 이러한 결과는 LLM이 인간의 언어 패턴에의 영향을 미치고 있다는 것을 증명합니다.

- **Performance Highlights**: 이 연구는 LLM이 인간의 말하기 언어에 영향을 미친 첫 번째 경험적 증거를 제공하며, 이는 사회적, 정책적 우려를 야기합니다. AI가 언어적 다양성을 감소시킬 위험이나 대규모 조작을 위한 악용 가능성 등을 강조하고 있으며, 머신 행동과 인간 문화 간의 피드백 루프에 대한 추가 연구의 필요성을 역설하고 있습니다.



### Taming CLIP for Fine-grained and Structured Visual Understanding of Museum Exhibits (https://arxiv.org/abs/2409.01690)
Comments:
          Accepted to ECCV 2024

- **What's New**: CLIP를 향상시키기 위해 박물관 전시물의 세부적이고 구조화된(structured) 시각 이해를 지향하는 새로운 방법론인 MUZE를 제안합니다.

- **Technical Details**: MUZE는 200K 이상의 이미지-테이블 쌍으로 구성된 데이터셋을 기반으로 하며, 변환기 기반 파싱 네트워크(parseNet)를 통해 CLIP의 이미지 임베딩을 테이블 구조로 매핑합니다. 이 과정에서 입력 이미지에 대한 알려진 속성-값 쌍의 컨텍스트를 통합하여 누락된 속성 값을 예측합니다.

- **Performance Highlights**: MUZE는 박물관 전시물에 대한 세부적이고 구조화된 이해에서 정확성을 크게 향상시켰습니다. 새로운 벤치마크에서 유망한 결과를 달성하며, 전체 실험을 통해 제안된 방법의 효과성을 입증했습니다.



### CTG-KrEW: Generating Synthetic Structured Contextually Correlated Content by Conditional Tabular GAN with K-Means Clustering and Efficient Word Embedding (https://arxiv.org/abs/2409.01628)
- **What's New**: 이번 연구에서는 기존의 Conditional Tabular GAN (CTGAN) 모델의 한계를 극복하기 위해 CTGKrEW(Conditional Tabular GAN with KMeans Clustering and Word Embedding)라는 새로운 프레임워크를 제안했습니다. 이 프레임워크는 합성(tabular) 데이터를 생성할 때, 문맥적으로 연관된 단어들을 세밀히 유지하는 능력을 갖추고 있습니다.

- **Technical Details**: CTGKrEW는 word2vec 방법을 사용하여 개별 단어를 벡터 표현으로 변환한 후, K-Means clustering을 통해 그룹화합니다. 이 프레임워크는 훈련 및 평가에 Upwork의 데이터셋을 사용하며, 기존 방법에 비해 CPU 시간은 99% 줄이고 메모리 사용량은 33% 감소시킵니다.

- **Performance Highlights**: CTGKrEW는 다양한 실험을 통해 생성된 데이터의 변동성, 문맥적 유사성, 빈도 분포 및 연관성을 분석하는 데 성공했습니다. 이러한 결과를 바탕으로 Krew라는 웹 애플리케이션을 개발하여 사용자가 필요에 따라 직무 기술서와 관련된 프로필 정보를 생성할 수 있도록 지원하고 있습니다.



### VoxHakka: A Dialectally Diverse Multi-speaker Text-to-Speech System for Taiwanese Hakka (https://arxiv.org/abs/2409.01548)
Comments:
          Submitted to O-COCOSDA 2024

- **What's New**: VoxHakka는 대만에서 사용되는 자원 부족 언어인 대만 Hakka를 위해 설계된 고품질 텍스트-투-스피치(TTS) 시스템입니다. 이 시스템은 여섯 가지 사투리를 지원하며, YourTTS 프레임워크를 활용하여 발음 정확도 및 자연스러움을 크게 향상시켰습니다.

- **Technical Details**: VoxHakka는 웹 스크래핑과 자동 음성 인식(ASR) 기반 데이터 정제 기법을 통해 다중 화자 및 다중 사투리 데이터셋을 구축하였습니다. 이 모델은 음성 속도 조정이 가능하며 CPU 자원만으로도 효율적으로 동작합니다. VoxHakka는 주로 정부 교육 기관에서 출처를 확보하여 윤리적으로 소싱된 데이터를 사용합니다.

- **Performance Highlights**: 주관적인 청취 테스트 결과 VoxHakka는 기존의 Hakka TTS 시스템과 비교하여 발음 정확도, 톤 정확성, 그리고 전반적인 자연스러움에서 훨씬 뛰어난 성능을 보였습니다. 이 연구는 Hakka 언어 기술의 중대한 발전을 의미하며 언어 보존과 부흥 노력에 중요한 자원을 제공합니다.



### Effective Noise-aware Data Simulation for Domain-adaptive Speech Enhancement Leveraging Dynamic Stochastic Perturbation (https://arxiv.org/abs/2409.01545)
Comments:
          Accepted to IEEE SLT 2024

- **What's New**: 본 논문에서는 기존의 도메인 간 음성 향상 문제를 해결하기 위하여 새로운 데이터 시뮬레이션 방법을 제안합니다. Noise-extractive 기술과 generative adversarial networks (GANs)을 활용하여, 제한된 양의 noisy speech 데이터로부터도 효과적인 음성 향상이 가능하도록 하는 방법론을 개발했습니다.

- **Technical Details**: 제안된 방법, NADA-GAN은 noise encoder를 통해 target-domain 데이터에서 noise embedding을 추출하고, 이를 generator가 이용하여 target domain에 적합한 noisy speech를 합성합니다. 또한, inference 중 dynamic stochastic perturbation을 도입하여 noise embedding에 제어된 변화를 주어, 모델이 미지의 noise 조건에 잘 일반화될 수 있도록 합니다.

- **Performance Highlights**: VoiceBank-DEMAND 벤치마크 데이터셋에서 실험한 결과, 제안된 NADA-GAN 방법이 기존의 강력한 baseline보다 우수한 성능을 보여주었으며, 평균 의견 점수(MOS) 평가에서 뛰어난 성능을 기록하여 음성 향상 외에도 다양한 분야에 적용 가능한 가능성을 제시했습니다.



### Revisiting SMoE Language Models by Evaluating Inefficiencies with Task Specific Expert Pruning (https://arxiv.org/abs/2409.01483)
- **What's New**: 이 연구는 Sparse Mixture of Expert (SMoE) 모델을 활용하여 언어 모델의 성능 향상과 효율성을 위한 새로운 프루닝 기술인 UNCURL을 소개합니다. 이 기술은 사전 훈련 동안 선택된 전문가 수에 따라 모델 구조 설계에 대한 결정을 안내합니다.

- **Technical Details**: SMoE 모델은 조건적으로 활성화된 피드포워드 서브네트워크를 사용하여 모델 파라미터와 예별 계산의 분리를 가능하게 합니다. 연구에서는 모델 프루닝을 통해 사전 훈련 중 전문가 수의 선택을 조절하며, 이를 통해 더 적은 전문가를 가진 SMoE 모델이 더 작은 모델보다 성능 상의 장점을 제공하는지를 평가합니다.

- **Performance Highlights**: UNCURL 기술을 통해 대형 SMoE 모델을 작게 조정하더라도 성능 이점을 유지할 수 있음을 보여줍니다. 더 적은 전문가로 모델을 설계하면 메모리 제약이 큰 상황에서 이익이 발생할 수 있으며, 중요한 벤치마킹 결과를 통해 프루닝 및 전문가 수에 대한 이해를 발전시킵니다.



### Membership Inference Attacks Against In-Context Learning (https://arxiv.org/abs/2409.01380)
Comments:
          To Appear in the ACM Conference on Computer and Communications Security, October 14-18, 2024

- **What's New**: 이 논문은 In-Context Learning (ICL) 해당하는 첫 번째 text-only membership inference attack을 제안합니다. ICL의 개인정보 보호 취약성을 다루며, 공격이 모델의 생성된 텍스트에서만 수행되도록 설계되었습니다.

- **Technical Details**: 논문에서는 네 가지 공격 방법(GAP, Inquiry, Repeat, Brainwash)을 제안하고, 네 가지 인기 있는 대형 언어 모델에 대해 실험을 수행합니다. 새로운 'Brainwash' 공격은 모델이 고정된 응답을 생성하는 시나리오에서 효과적으로 작동합니다.

- **Performance Highlights**: 실험 결과, LLaMA 모델에 대해 95% 이상의 정확도를 기록하며, Hybrid attack은 Brainwash와 Repeat 공격을 결합하여 높은 성능을 보여줍니다. 또한, 세 가지 방어 방법을 통해 개인정보 유출을 효과적으로 감소시킬 수 있음을 입증했습니다.



### Pairing Analogy-Augmented Generation with Procedural Memory for Procedural Q&A (https://arxiv.org/abs/2409.01344)
- **What's New**: 이 연구에서는 복잡한 절차적 질문 응답(task)에서의 성능을 개선하기 위한 'analogy-augmented generation (AAG)'이라는 새로운 시스템을 제안합니다. 이 방법은 기존의 RAG 시스템을 확장하여, 사람의 유사 추론 능력을 활용하여 절차적 지식을 효과적으로 처리합니다.

- **Technical Details**: AAG 시스템은 세 가지 주요 모듈로 구성됩니다: 1) 절차적 메모리 스토어(procedural memory store), 2) 쿼리 재작성(query rewriting) 및 요약, 3) 자기 비판(iterative refinement with self-critic)입니다. 이 시스템은 절차적 지식을 처리하기 위해 특정하게 설계된 메모리 표현을 사용하며, 기존의 정보를 활용하여 새로운 문제를 해결합니다.

- **Performance Highlights**: 제안된 AAG 시스템은 LCStep, RecipeNLG, CHAMP 데이터셋에서 전통적인 RAG 및 few-shot 방법론보다 더 나은 성능을 보여 주었으며, 인류 평가에서도 유의미한 개선이 관찰되었습니다.



### CLIBE: Detecting Dynamic Backdoors in Transformer-based NLP Models (https://arxiv.org/abs/2409.01193)
Comments:
          To appear in the Network and Distributed System Security (NDSS) Symposium, February, 2025

- **What's New**: CLIBE는 Transformer 기반 NLP(natural language processing) 모델에서 동적 백도어(dynamic backdoor)를 탐지하는 최초의 프레임워크입니다.

- **Technical Details**: CLIBE는 의심스러운 Transformer 모델에 'few-shot perturbation'을 주입하여 주의(attention) 레이어에서 최적화된 가중치 변화를 일으킵니다. 이를 통해 변형된 모델이 제한된 참조 샘플을 특정 레이블로 분류하도록 합니다.

- **Performance Highlights**: 세 가지 고급 NLP 동적 백도어 공격과 두 가지 널리 사용되는 Transformer 프레임워크, 네 가지 실제 분류 작업을 통해 CLIBE의 효율성이 강력하게 검증되었습니다. 또한, CLIBE는 다양한 적응형 공격에 대한 강건성을 입증했습니다.



### SCOPE: Sign Language Contextual Processing with Embedding from LLMs (https://arxiv.org/abs/2409.01073)
- **What's New**: 본 연구에서는 대화 맥락을 고려한 새로운 Sign Language Recognition (SLR)과 Sign Language Translation (SLT) 프레임워크인 SCOPE를 소개합니다. 기존의 방법들이 대화 장면에서의 데이터 세트의 다양성 부족과 맥락 정보를 무시하는 문제를 해결하기 위해 제시되었습니다.

- **Technical Details**: SCOPE는 다중 모달 인코더를 활용하여 대화 맥락을 통해 글로스 레벨 인식을 향상시키며, 사전 대화 맥락을 포함하여 대형 언어 모델 (Large Language Model, LLM)을 추가로 미세 조정합니다. 우리는 72시간 분량의 중국 수화 비디오로 구성된 새로운 수화 데이터 세트를 제공하고, 실험을 통해 Phoenix-2014T, CSL-Daily 및 SCOPE 데이터 세트에서 최첨단 성능을 달성하였습니다.

- **Performance Highlights**: SCOPE 프레임워크는 다양한 데이터 세트에서 탁월한 성능을 보였으며, 청각 장애인 커뮤니티의 참가자를 대상으로 실시한 설문조사 결과도 우리의 접근 방식이 실제 응용에서의 강력함과 효과성을 입증하였습니다. 데이터 세트 및 코드는 연구 촉진을 위해 오픈 소스로 제공될 예정입니다.



### VideoLLaMB: Long-context Video Understanding with Recurrent Memory Bridges (https://arxiv.org/abs/2409.01071)
- **What's New**: 최근 대형 비디오-언어 모델의 발전이 실시간 계획 및 세부 상호작용에서 중요한 잠재력을 보여주고 있습니다. 특히, 본 논문에서 소개하는 VideoLLaMB는 전체 비디오 시퀀스와 과거 시각 데이터를 함께 인코딩하여 의미적 연속성을 유지하며 다양한 작업에서 모델 성능을 향상시키는 혁신적인 프레임워크입니다.

- **Technical Details**: VideoLLaMB는 여러 핵심 모듈로 구성되어 있습니다: 1) Semantic Segmentation: 비디오를 의미적으로 구분하는 SceneTilling 알고리즘을 통해 독립적인 의미 단위로 나누어 의미의 흐름을 유지합니다. 2) Recurrent Memory Layer: 재귀 기억 토큰을 이용하여 시각적 정보를 손실없이 인코딩하며, 장기 의존성을 유지합니다. 3) Memory Retriever: 기억 기억소의 주기적 갱신을 통해 성능을 극대화합니다. 이 방법은 16 프레임에서 훈련되었으며, Nvidia A100 GPU에서 최대 320 프레임을 지원합니다.

- **Performance Highlights**: VideoLLaMB는 세 가지 VideoQA 벤치마크에서 5.5 포인트, Egocentric planning에서는 2.06 포인트의 성능 개선을 보여주며, MVBench에서 동급의 7B 모델에 비해 우수한 성과를 기록합니다. 또한, VideoLLaMB는 NIAVH 벤치마크에서 긴 비디오 내 특정 프레임을 정확하게 찾아내는 능력을 입증했습니다.



### Personalized Lip Reading: Adapting to Your Unique Lip Movements with Vision and Languag (https://arxiv.org/abs/2409.00986)
Comments:
          Code available: this https URL

- **What's New**: 본 논문에서는 새로운 화자 적응형 립 리딩(Lip Reading) 방법을 제안하여 시각 및 언어 수준에서 목표 화자에게 적응하도록 하는 모델을 발전시켰습니다. 기존 연구들은 주로 시각 정보에만 초점을 맞췄으나, 이 방법은 화자의 언어적 패턴까지 고려한 혁신적인 접근 방식을 채택하고 있습니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 구성 요소로 이루어져 있습니다: 1) 화자 적응을 위한 시각 수준 적응은 사전 훈련된 립 리딩 모델을 목표 화자의 입 모양, 움직임 및 말하기 속도에 맞춰 조정하며, 2) 언어 수준 적응은 화자의 특정 언어 습관에 따라 사전 훈련된 모델을 수정하여 언어 모델링 확률을 학습하도록 설계되었습니다. 이 과정에서 패딩 프롬프트(padding prompts)와 Low-Rank Adaptation(LoRA)을 효과적으로 이용합니다.

- **Performance Highlights**: 제안된 방법은 새로운 VoxLRS-SA 데이터셋을 사용하여 실제 시나리오에서 검증되었으며, 기존 화자 적응형 접근 방식 대비 목표 화자에 맞춘 성능 개선을 입증하였습니다. 특히, 약 100K 어휘를 포함하고 다양한 자세 변화를 제공하는 이 데이터셋은 실제 환경에서 문장 수준 립 리딩 성능을 평가하는 데 중요한 기여를 하고 있습니다.



### ToolACE: Winning the Points of LLM Function Calling (https://arxiv.org/abs/2409.00920)
Comments:
          21 pages, 22 figures

- **What's New**: 본 논문에서는 ToolACE라는 자동화된 에이전틱 파이프라인을 소개하여, 고품질의 다양한 도구 학습 데이터를 생성하는 방법을 제시합니다. 이는 새로운 자기 진화 합성(Self-Evolution Synthesis) 프로세스를 활용하여 26,507개의 다양한 API 풀을 구성합니다.

- **Technical Details**: ToolACE는 두 가지 주요 모듈인 도구 자기 진화 합성(TSS)과 다중 에이전트 상호작용 대화 생성(MAI)을 포함하며, 이들 구성 요소를 통해 다양한 API를 생성하고, 정교한 대화를 구축하며, 데이터 품질을 엄격하게 검토합니다. 또한 데이터 정확도를 확보하기 위해 규칙 기반과 모델 기반 점검을 결합한 이중 레이어 검증 시스템(DLV)을 구현합니다.

- **Performance Highlights**: ToolACE에서 생성한 데이터로 훈련된 모델은 단 8B 파라미터에도 불구하고 Berkeley Function-Calling Leaderboard에서 최신 GPT-4 모델에 필적하는 성능을 달성하였습니다. 또한 다양한 함수 호출 시나리오에 대한 소개를 통해 LLM의 도구 사용 능력을 향상시킬 수 있는 기반을 마련하였습니다.



### Benchmarking LLM Code Generation for Audio Programming with Visual Dataflow Languages (https://arxiv.org/abs/2409.00856)
- **What's New**: 본 연구에서는 메타프로그래밍 코드 표현을 통한 LLM(대형 언어 모델)의 코드 생성 성능을 시각적 프로그래밍 언어에서 오디오 프로그래밍 작업에 대한 여러 레벨의 표현에서 탐구합니다. 또한, 코드 생성의 새로운 접근 방식을 평가하기 위해 오디오 디지털 신호 처리(DSP)를 위한 벤치마크 세트를 제안합니다.

- **Technical Details**: 연구는 MaxMSP와 MaxPy를 비롯한 두 가지 시각적 언어와 Web Audio API에 대한 코드 생성을 검토하며, 각 언어의 코드 생성은 JSON 형식을 활용합니다. 메타프로그래밍 및 JSON을 통한 직접 노드 생성 방식을 비교하고, LLM이 생성한 코드의 정합성을 측정하는 새로운 메트릭을 정의합니다.

- **Performance Highlights**: 실험 결과, 메타프로그래밍을 통한 코드 생성이 문법적으로 올바른 경우 더 의미적으로 정확한 코드를 생성하는데 기여함을 발견했습니다. 또한, 무작위 함수 및 루프를 활용한 풍부한 메타프로그래밍 요청은 더 복잡한 코드를 생성하는 데 기여했습니다.



### Report Cards: Qualitative Evaluation of Language Models Using Natural Language Summaries (https://arxiv.org/abs/2409.00844)
Comments:
          11 pages, 8 figures

- **What's New**: 대규모 언어 모델(LLMs)의 평가 방식을 개선하기 위해 'Report Cards'라는 새로운 방법론을 제안하였습니다. Report Cards는 모델의 특정 기술이나 주제에 대한 행동을 요약한 인간 친화적인 자연어 형식의 문서입니다.

- **Technical Details**: 이 논문에서는 Report Cards의 품질을 평가하기 위한 세 가지 기준인 specificity(구체성), faithfulness(충실성), interpretability(해석 가능성)를 제시합니다. 또한, LLMs의 출력을 바탕으로 Report Cards를 생성하는 PRESS라는 반복적인 알고리즘도 소개합니다. 실험에서는 다양한 LLMs의 Report Cards가 일반적인 정량적 벤치마크를 넘어서는 인사이트를 제공함을 보여주었습니다.

- **Performance Highlights**: Report Cards는 기존의 벤치마크 지표들이 간과할 수 있는 모델의 고유한 성능을 정확하게 캡처하고, 모델 간의 명확한 구분을 가능하게 하여, LLM의 평가 기준을 더욱 포괄적이고 해석 가능한 방향으로 확장합니다.



### Building FKG.in: a Knowledge Graph for Indian Food (https://arxiv.org/abs/2409.00830)
Comments:
          14 pages, 3 figures, 25 references, Formal Ontology in Information Systems Conference 2024 - Integrated Food Ontology Workshop

- **What's New**: 이번 논문에서는 인도 음식에 대한 정보를 자동으로 수집하기 위한 지식 그래프 구축을 위한 본체 설계와 지식 공학(knowledge engineering) 기법을 제안합니다. 특히 요리, 레시피, 성분, 영양상의 모든 지식을 포괄적으로 수집하는 지능형 방법의 설계를 다루고 있습니다.

- **Technical Details**: 제안된 기법은 공공 도메인에서 레시피 블로그 사이트의 정보를 수집하고 AI(Artificial Intelligence), LLM(Large Language Model), 언어 기술을 활용하여 인도 음식의 지식 그래프를 구축합니다. 이 과정에는 인간의 개입(human-in-the-loop)도 포함되어 신뢰성을 높이고 있습니다.

- **Performance Highlights**: FKG.in이라는 지식 그래프는 인도 요리의 전모를 포괄하여 후속 음식 컴퓨팅 애플리케이션을 구축하는 디지털 자원으로 활용될 것입니다. 이 연구는 다른 도메인에도 적용 가능하며, AI 기반 스마트 분석 및 개인화된 디지털 건강을 위한 추천 시스템 구축에 기여할 것입니다.



### LibriheavyMix: A 20,000-Hour Dataset for Single-Channel Reverberant Multi-Talker Speech Separation, ASR and Speaker Diarization (https://arxiv.org/abs/2409.00819)
Comments:
          InterSpeech 2024

- **What's New**: 이 논문은 다중 발화자와 먼 거리 환경을 테스트할 수 있는 대규모 음성 데이터셋 LibriheavyMix를 소개합니다. 이 데이터셋은 음성 분리(speech separation), 인식(recognition) 및 화자 구분(diarization)을 위한 연구를 발전시키기 위한 중요한 자원으로 개발되었습니다.

- **Technical Details**: LibriheavyMix 데이터셋은 20,000시간 분량의 발화 데이터를 포함하며, 이는 다양한 스피커 턴과 실세계 대화 시나리오를 모사하도록 설계되었습니다. 이 데이터셋은 비교적 큰 용량을 가지고 있으며, 실내 잔향을 도입하여 실제 환경을 더욱 잘 반영합니다.

- **Performance Highlights**: 실험 결과는 WHAMR! 데이터셋에서의 평가를 통해 제안된 데이터셋이 널리 적용 가능함을 입증했습니다. 특히, LibriheavyMix는 기존 데이터셋보다 10배 이상의 데이터를 제공하며, 다양한 음성 인식 기법에 대한 성능 향상을 보여주었습니다.



### ContextCite: Attributing Model Generation to Contex (https://arxiv.org/abs/2409.00729)
- **What's New**: 이 논문에서는 언어 모델이 생성한 응답에 대해 주어진 맥락을 어떻게 활용하는지를 조사합니다. 특히, 'context attribution' 문제를 도입하여 특정 생성 문장이 어떤 맥락에 기반했는지를 파악할 수 있는 방법을 제시합니다. 그리고 ContextCite라는 새롭고 효율적인 방법을 통해 다양한 자연어 처리 작업에서 맥락 기여도를 평가하고 활용할 수 있는 가능성을 보여줍니다.

- **Technical Details**: ContextCite는 언어 모델의 응답이 맥락의 각 부분 포함 여부에 따라 어떻게 영향을 받는지를 모방하는 대체 모델(surrogate model)을 학습하는 방법입니다. 이 방법은 (1) 정확한 언어 모델의 행동을 모델링하고 (2) 적은 수의 추가 추론 패스를 사용하여 효율적으로 추정될 수 있습니다. 이 대체 모델의 가중치는 기여도 점수(attribution scores)로 직접 사용할 수 있습니다.

- **Performance Highlights**: ContextCite의 유용성을 세 가지 응용 프로그램을 통해 보여주었습니다: (1) 생성된 문장의 검증 지원, (2) 맥락을 정리하여 응답 품질 향상, (3) 맥락 오염 공격 감지. 이러한 방법들이 언어 모델의 성능 향상에 기여함을 입증했습니다.



### Hound: Hunting Supervision Signals for Few and Zero Shot Node Classification on Text-attributed Graph (https://arxiv.org/abs/2409.00727)
- **What's New**: 본 연구에서는 텍스트 속성 그래프(Text-attributed graph, TAG)에서의 few-shot 및 zero-shot 노드 분류의 정확성을 향상시키기 위해 Hound라는 새로운 방법론을 제안합니다. 기존 방법들이 대조 손실(contrastive loss)만을 사용했으나, Hound는 더 많은 감독 신호(supervision signals)를 제공하는 것을 목표로 합니다.

- **Technical Details**: Hound는 세 가지 증강 기법(augmentation techniques)을 도입했습니다: 노드 변형(node perturbation), 텍스트 일치(text matching), 의미의 부정(semantics negation). 노드 변형은 그래프에서 엣지를 무작위로 추가/삭제하여 다양한 노드 임베딩을 생성합니다. 텍스트 일치는 유사한 임베딩을 가진 텍스트를 검색하여 노드와 일치시킵니다. 의미의 부정은 원래 텍스트와 반대 의미를 가지는 부정적인 텍스트를 생성하여 본래의 노드 및 텍스트와 대비하는 것입니다.

- **Performance Highlights**: Hound는 5개 데이터셋에 대해 13개의 최신 기법과 비교한 결과, 모든 기준선(baselines)에서 일관되게 더 높은 정확성을 달성했습니다. 특히, few-shot 및 zero-shot 분류에서 각각 평균 4.6% 및 8.8%의 정확도가 향상되었습니다.



### Who Would Chatbots Vote For? Political Preferences of ChatGPT and Gemini in the 2024 European Union Elections (https://arxiv.org/abs/2409.00721)
- **What's New**: 본 연구는 2024 유럽 의회 선거와 관련하여 대규모 언어 모델에 기반한 챗봇(ChatGPT, Gemini)의 정치적 편향을 조사합니다.

- **Technical Details**: 이 연구는 27개 EU 회원국에서 유럽 의회를 대표하는 정치 정당에 대한 평가를 수행하기 위해 표준화된 프롬프트를 사용하여 매일 데이터를 수집했습니다. Gemini는 정치적 질문에 대답하지 않는 반면, ChatGPT는 일관된 평가를 제공했습니다.

- **Performance Highlights**: ChatGPT는 좌파 및 중도 정당을 선호하는 경향을 보였고, 그린/유럽 자유 동맹당에 대해 가장 높은 평가를 주었습니다. 반면, 우파 정당, 특히 정체성과 민주주의 그룹은 가장 낮은 평가를 받았습니다. 정치적 편향성을 비판적으로 접근할 필요성과 투명성과 규제의 필요성이 강조되었습니다.



### Multimodal Multi-turn Conversation Stance Detection: A Challenge Dataset and Effective Mod (https://arxiv.org/abs/2409.00597)
Comments:
          ACM MM2024

- **What's New**: 이 논문에서는 다중 모달 (multimodal) 다회전 대화 기조 감지 데이터셋 (MmMtCSD)를 소개합니다. 기존 연구들이 개별 텍스트-이미지 쌍에서 기조를 모델링하는 데 치중했으나, 이 데이터셋은 실제 소셜 미디어의 다자 대화 맥락을 포착합니다.

- **Technical Details**: 제안된 MLLM-SD (multimodal large language model stance detection) 프레임워크는 텍스트 인코더, 비주얼 인코더 및 다중 모드 융합 모듈로 구성됩니다. 텍스트 인코더는 입력 대화 내역을 인코딩하고, 비주얼 인코더는 ViT (Vision Transformer) 모델을 활용해 이미지의 특징을 추출합니다. 이후, LoRA (low-rank adaptation) 기법을 통해 다중 모드 간 정보를 통합할 수 있습니다.

- **Performance Highlights**: MmMtCSD 데이터셋에서의 실험 결과, 제안한 MLLM-SD 접근법이 다중 모달 기조 감지에서 최신의 성능을 보였습니다. 이 연구는 진정한 대화 맥락과 이미지 정보를 활용하여 기조 감지 연구의 실질적인 적용을 향상시킬 것입니다.



### How Does Diverse Interpretability of Textual Prompts Impact Medical Vision-Language Zero-Shot Tasks? (https://arxiv.org/abs/2409.00543)
- **What's New**: 이번 연구에서는 Medical Vision Language Pre-training (MedVLP) 모델의 문맥 생성 민감도를 체계적으로 평가하여 다양한 텍스트 프롬프트에 대한 안정성을 연구하였습니다. 특히, 15종의 질병에 대한 세 가지 주요 MedVLP 방법의 성능을 검토하였습니다.

- **Technical Details**: 연구에서는 GPT-4o를 활용하여 발전된 6가지 프롬프트 스타일을 생성하였습니다. BioViL, MedKLIP, KAD 모델이 이 프롬프트를 통해 ChestX-ray14, CheXpert, COVIDx CXR-4와 같은 공공 데이터 세트에서 평가되었습니다.

- **Performance Highlights**: 모델의 성능은 원래 프리트레이닝 시 사용된 스타일과 다른 스타일의 프롬프트를 사용할 때 평균 10.17% 감소하였으며, 이는 MedVLP 모델의 강건성 부족을 시사합니다. 프롬프트의 해석 가능성이 높아질수록 복잡한 의료 개념 이해에 어려움을 겪는 것으로 나타났습니다.



### Statistics of punctuation in experimental literature -- the remarkable case of "Finnegans Wake" by James Joyc (https://arxiv.org/abs/2409.00483)
- **What's New**: 이번 연구는 전통 문학 작품에서의 구두점 사용 패턴을 탐구하며, 그 패턴들이 자연언어의 보편적인 특성을 드러내는 방식을 보여줍니다. 특히, 연속적인 구두점 사이의 거리가 이산 Weibull (Weibull) 분포를 따른다는 점에서 중요한 발견을 나타냅니다.

- **Technical Details**: 구두점은 서면 언어의 특정 조직을 유지하는 메커니즘 중 하나로, 텍스트 내 문단을 구분하는 역할을 합니다. 연구에서는 7개 유럽어 문학 텍스트의 구두점 사이의 단어 수를 측정하여 이산 Weibull 분포로 설명된다는 사실을 발견했습니다. 그러나 제임스 조이스의 작품들에서는 이와 다른 패턴이 관찰되었습니다. 'Finnegans Wake'는 특히 두드러진 사례로, 다중 프랙탈성(multifractality)을 드러내며 고유한 통계적 특성을 보입니다.

- **Performance Highlights**: 연구 결과, 모든 분석된 텍스트에서 문장은 구두점 사이의 단어 수와 관련된 패턴에서 더 많은 자유도를 보였으며, 이로 인해 긴 비선형 상관관계(long-range nonlinear correlations)가 유도될 수 있음을 확인했습니다. 'Finnegans Wake'는 다중 프랙탈성 측면에서 특히 주목받는 작품으로, 문장이 긴 텍스트 내에서 자연스러운 흐름을 유지하는 데 기여하고 있습니다.



### Chatting Up Attachment: Using LLMs to Predict Adult Bonds (https://arxiv.org/abs/2409.00347)
- **What's New**: 의료 분야에서의 데이터 수집의 어려움을 극복하기 위해, 대규모 언어 모델(GPT-4, Claude 3 Opus)을 통해 생성한 합성 데이터(synthetic data)를 활용하는 새로운 접근 방식을 소개합니다.

- **Technical Details**: 연구에서는 다양한 프로필, 어린 시절 기억, 애착 스타일을 가진 성인 에이전트를 시뮬레이션하는 과정을 포함하여, 이들이 참여한 성인 애착 인터뷰(Adult Attachment Interviews, AAI)의 응답을 사용합니다. 모델은 9명의 실제 인터뷰 대화록을 이용해 평가되며, 정신 건강 전문가에 의해 분석 및 라벨링되었습니다.

- **Performance Highlights**: 합성 데이터를 사용하여 훈련한 모델이 실제 사람 데이터로 훈련한 모델과 유사한 성과를 낼 수 있음을 발견했습니다. 또한, 합성 응답의 원시 임베딩(raw embeddings)이 실제 응답과는 다른 공간을 차지하지만, 비라벨 인간 데이터와 간단한 표준화를 통해 이들을 더 가깝게 정렬할 수 있음을 보여줍니다. 이 조정은 정성적 분석을 통해 지지되며, 표준화된 임베딩의 예측 정확도가 향상되는 데 기여했습니다.



### MAPWise: Evaluating Vision-Language Models for Advanced Map Queries (https://arxiv.org/abs/2409.00255)
Comments:
          30 Pages, 46 Tables, 6 Figure

- **What's New**: 이번 연구는 Vision-Language Models (VLMs)의 choropleth maps에 대한 질문 응답 능력을 분석합니다. 새로운 데이터셋인 MAPWise를 소개하며, 미국, 인도, 중국을 포함해 1,000개의 질문을 제공합니다.

- **Technical Details**: MAPWise 데이터셋은 다양한 난이도의 질문 템플릿을 포함하며, 지리적 및 공간적 이해를 평가하기 위하여 43개의 질문 유형이 마련되었습니다. VLMs 모델을 통해 실험하였으며, Zero-Shot 및 Explicit Extraction and Reasoning 방식으로 평가하였습니다.

- **Performance Highlights**: VLMs의 성능 평가에서 쟁점과 한계를 밝혀냈으며, 새로운 벤치마크와 데이터셋을 통해 향후 연구 방향을 제시했습니다. 특히, 새롭게 도입한 MAPWise 데이터셋은 choropleth maps와 관련하여 모델의 성능을 비교하는 데 유용합니다.



### Emerging Vulnerabilities in Frontier Models: Multi-Turn Jailbreak Attacks (https://arxiv.org/abs/2409.00137)
- **What's New**: 본 연구에서는 단일 및 다중 턴 입력 형식 모두에서 사용할 수 있는 jailbreak 공격의 데이터셋을 소개합니다. 각 형식이 작성 내용은 동일하지만 성공률이 다르고, 모델의 방어 메커니즘이 입력 구조에 따라 다르게 작동함을 보여줍니다. 또한 최첨단 모델의 취약성은 단일 및 다중 턴 환경 모두에서 심층적으로 연구되어야 함을 강조합니다.

- **Technical Details**: 이 연구는 단일 턴 공격과 다중 턴 공격 간의 차이를 정량화하고, 두 공격 형식의 성공적인 공격률 차이가 최대 50%까지 있다는 것을 보여줍니다. 다양한 데이터셋을 사용하여 공격을 분석하며, 사이퍼 기법을 이용한 공격의 효과도 검토합니다. 데이터셋은 해로운, 완전 무해한, 반 무해한 프롬프트로 구성되어 있으며, 각각의 특성을 바탕으로 실험을 진행합니다.

- **Performance Highlights**: 상위 모델(OpenAI, Anthropic, Meta)에 적용된 공격에서 여전히 안전성 문제들이 드러났습니다. 또한 방어 메커니즘과 공격자의 전략 사이의 복잡한 상호작용으로 인해 방어 조치를 마련하기 어려운 상황임을 확인하였습니다.



### Leveraging Large Language Models for Wireless Symbol Detection via In-Context Learning (https://arxiv.org/abs/2409.00124)
Comments:
          Accepted at IEEE GLOBECOM 2024

- **What's New**: 이 논문은 적은 데이터로 무선 시스템의 문제를 해결하기 위해 대형 언어 모델(LLM)의 인컨텍스트 학습(in-context learning) 능력을 활용하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 무선 기호 변조(symbol demodulation) 작업에 대해 DNN(Deep Neural Network)보다 LLM을 이용한 접근이 더 효과적임을 증명하며, 특히 다양한 프롬프트 템플릿을 사용할 때 LLM의 성능은 크게 달라질 수 있는 점을 강조합니다. 데이터가 제한된 상황에서도 LLM은 학습 없이 추론만으로 활용될 수 있습니다.

- **Performance Highlights**: 실험 결과, ICL(인컨텍스트 학습) 방법을 사용할 경우 전통적인 DNN보다 평균 22%의 성능 향상을 보여주었으며, LLM의 교정(calibration) 기법을 적용하면 높은 신뢰도를 가진 예측 결과를 얻을 수 있음을 시사합니다.



### 3-in-1: 2D Rotary Adaptation for Efficient Finetuning, Efficient Batching and Composability (https://arxiv.org/abs/2409.00119)
Comments:
          24 pages, 6 figures, 13 tables

- **What's New**: 이 논문은 RoAd라는 새로운 기법을 소개하며, 이 기법은 LLMs(대형 언어 모델)을 효율적으로 조정하는 데 필요한 훈련 가능한 매개변수를 최소화합니다. 특히, 여러 작업 또는 사용자에 맞춘 어댑터 필요성이 있는 경우, RoAd 방법은 요청에 따라 효율적인 처리를 가능하게 하고, 해석 가능성도 강화합니다.

- **Technical Details**: RoAd는 2D 회전 기법을 활용하여 LLMs를 조정하며, 이를 통해 다음과 같은 세 가지 주요 문제를 해결합니다: (1) $<0.1\%$의 훈련 가능한 매개변수로 GLUE 벤치마크 및 여러 추론 작업에서 최적의 성능을 제공합니다; (2) 서로 다른 어댑터를 필요로 하는 동시에 요청을 배치 처리하는 데 있어 요소별 곱(element-wise multiplication)을 사용함으로써 계산 부담을 줄입니다; (3) 분산 교환 개입(distributed interchange intervention) 프레임워크 내에서 통합하여 LLM의 해석 가능성을 향상시킵니다.

- **Performance Highlights**: RoAd는 GLUE, 상식 추론 과제 및 산수 과제에서 타 PEFT 방법을 초월하는 성능을 보여주며, LoRA보다 두 배 높은 처리량을 자랑합니다. 또한, RoAd는 다양한 작업에 대해 학습된 가중치를 결합할 수 있는 능력을 보여주어 새로운 기능을 표현할 수 있습니다.



### LCA and energy efficiency in buildings: mapping more than twenty years of research (https://arxiv.org/abs/2409.00065)
- **What's New**: 이 논문은 Life Cycle Assessment (LCA) 분야에서 20년 이상의 연구를 종합적으로 맵핑하고, 기존 문헌 리뷰의 한계를 극복하고자 합니다.

- **Technical Details**: 저자들은 8024개의 과학적 초록을 분석하기 위해 사회적 네트워크 분석(social network analysis)과 텍스트 마이닝(text mining)을 결합한 혁신적인 방법론을 사용하였습니다. 이를 통해 건물 및 지속 가능성 클러스터(BSC)라는 7개의 주요 주제 그룹을 식별하였습니다. 또한, 의미적 브랜드 점수(semantic brand score, SBS) 지표를 적용하여 이들 주제가 더 넓은 건물 및 지속 가능성 담론에서의 중요성을 평가했습니다.

- **Performance Highlights**: 주요 연구 주제는 건축 자재(building materials)와 에너지 효율성(energy efficiency)과 관련이 있으며, 이 논문은 LCA 개념에 초점을 맞추어 진행된 건물 및 지속 가능성의 동향을 추적합니다. 또한, 광범위한 문헌 도메인을 리뷰하는 혁신적인 접근 방법과 함께 신흥 및 개발이 덜 이루어진 주제들을 제시하며, 향후 연구 방향에 대한 중요한 통찰을 제공합니다.



### Urban Mobility Assessment Using LLMs (https://arxiv.org/abs/2409.00063)
Comments:
          13 pages, 10 Figures

- **What's New**: 이 연구는 사용자 추적 또는 여행 설문조사에 의한 이동성 데이터 수집의 어려움을 해결하기 위해 AI 기반 접근 방식을 제안합니다. 대규모 언어 모델(LLMs)을 활용하여 여행 설문조사를 합성하고 다양한 미국 대도시 지역에서 효과성을 평가합니다.

- **Technical Details**: 이 시스템은 Llama-2, Gemini-Pro 및 GPT-4와 같은 LLM을 사용하여 여행 다이어리 항목을 작성함으로써 여행 설문 조사를 생성합니다. 평가는 패턴 수준, 여행 수준, 활동 체인 수준으로 나뉘어 있으며, 2017년 NHTS 데이터와 비교됩니다.

- **Performance Highlights**: 실험 결과, LLM이 실제 여행 설문 데이터와 유사한 합성 데이터를 생성할 수 있으며, 소량의 실제 데이터를 기반으로 fine-tuning을 수행하면 기존 시뮬레이션 기법을 초과하는 성능을 보입니다.



### Evolving Text Data Stream Mining (https://arxiv.org/abs/2409.00010)
Comments:
          134 Pages, 7 Chapters, 38 Figures, 10 Tables

- **What's New**: 최근 온라인 소셜 플랫폼에서 생성되는 방대한 텍스트 데이터를 효과적으로 분석하기 위해 새로운 클러스터링 및 다중 레이블 학습 모델이 제안되었습니다. 이러한 모델은 텍스트 스트림의 고유한 특성을 고려하여 설계되었습니다.

- **Technical Details**: 텍스트 스트림은 시간에 의해 생성된 문서의 정렬된 시퀀스이며, 무한 길이, 데이터 희소성(data sparsity), 진화(evolution)와 같은 특성이 있습니다. 제안된 모델은 고차원 텍스트 데이터로 인한 학습 성능 저하 문제를 해결하고, 문서의 의미적 텍스트 표현을 추출하며, 시간이 지남에 따라 변화하는 주제를 포착하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 제안된 접근 방식은 레이블(label) 부족 문제를 해결하도록 설계되었으며, 기존의 방법들이 필요로 하는 모든 레이블 데이터의 가용성이 부족한 문제를 해결하기 위한 방안을 모색하고 있습니다.



### Measuring Human Contribution in AI-Assisted Content Generation (https://arxiv.org/abs/2408.14792)
- **What's New**: 본 연구는 인공지능(AI) 지원 콘텐츠 생성에서 인간의 기여도를 측정하는 새로운 프레임워크를 제시합니다. 이는 정보 이론에 기초하여 인간 입력과 AI 출력 간의 상호 정보를 이용해 비율적으로 인간의 기여를 정량화합니다.

- **Technical Details**: 제안된 측정 방법은 인간 입력과 AI 출력 간의 상호 정보 비율을 AI 출력의 자기 정보 총량과 비교하여 계산합니다. 다양한 도메인에서 수집된 AI 지원 생성 데이터에서의 실험 결과를 통해 이 방법의 유효성을 확인했습니다.

- **Performance Highlights**: Llama-3 및 Mixtral와 같은 최신 LLM 모델을 이용하여 결과를 도출하였으며, 인간의 기여도가 서로 다른 창작 도메인에서 측정 가능한 일관성을 나타냈습니다. 예를 들어, Llama-3를 사용해 생성한 뉴스 기사에서는 인간의 기여도가 평균 85.37%에서 30.83%까지 변하는 것을 보여주었습니다.



### Squid: Long Context as a New Modality for Energy-Efficient On-Device Language Models (https://arxiv.org/abs/2408.15518)
- **What's New**: 이 논문은 'Dolphin'이라는 새로운 decoder-decoder 아키텍처를 소개하며, 이는 언어 모델에서 긴 문맥을 에너지 효율적으로 처리하는 데 중점을 두고 있습니다. 이 접근법은 온디바이스 모델의 본질적인 에너지 소비 및 지연 시간 문제를 해결합니다.

- **Technical Details**: Dolphin은 0.5B 매개변수를 가진 컴팩트한 decoder를 사용하여 방대한 문맥 정보를 메모리 임베딩으로 증류(distill)하여 입력 길이를 7B 매개변수를 가진 주요 decoder 모델에 의해 크게 줄여줍니다. 이 방법은 비전-언어 모델에서 영감을 얻어 긴 텍스트 문맥을 인코딩하는데 이미지 임베딩 프로젝터를 재사용합니다. 이로 인해 계산적 오버헤드 없이 기존의 긴 문맥을 처리할 수 있습니다.

- **Performance Highlights**: 실제 평가에서는 기존의 전체 길이 문맥 처리 방법에 비해 에너지 효율성이 10배 개선되었고, 지연 시간은 5배 감소되었습니다. 이러한 결과는 응답의 품질 손실 없이 이루어졌습니다.



New uploads on arXiv(cs.IR)

### Building a Scalable, Effective, and Steerable Search and Ranking Platform (https://arxiv.org/abs/2409.02856)
- **What's New**: 본 논문에서는 소비자 행동을 기반으로 직관적인 상품 추천을 제공하기 위한 개인화된 실시간 랭킹 플랫폼을 제안합니다. 이 플랫폼은 대량의 고객 데이터와 상품 정보를 효율적으로 처리하여 사용자 경험을 향상 시킵니다.

- **Technical Details**: 이 시스템은 transformer 기반 모델을 활용하여 다양한 랭킹 레이어에서 고객 행동 시퀀스를 학습하며, 즉각적이고 맥락에 맞는 데이터를 처리할 수 있습니다. 모델은 high-load 환경에서도 쉽게 확장될 수 있도록 설계되어 있으며, 벡터 저장소를 활용하여 인덱싱 및 점수를 효율적으로 계산합니다. 또한, 고객의 행동, 콘텐츠 기반 데이터 및 다양한 맥락 정보를 통합하여 랭킹 품질을 높이고자 합니다.

- **Performance Highlights**: 실험 결과, 제안된 시스템은 기존 솔루션에 비해 10-40% 개선된 오프라인 평가 지표를 보였으며, 온라인 A/B 테스트에서 15%의 고객 참여율 상승과 함께 2.2%의 순수익 증가를 기록했습니다. 이 플랫폼은 유럽의 대규모 전자상거래 환경에서 성공적으로 구축 및 운영되고 있습니다.



### RouterRetriever: Exploring the Benefits of Routing over Multiple Expert Embedding Models (https://arxiv.org/abs/2409.02685)
- **What's New**: 이번 연구에서는 RouterRetriever라는 정보 검색(Information Retrieval) 모델을 소개합니다. 이 모델은 여러 도메인 특화 전문가(Expert)를 활용하고, 라우팅 메커니즘을 통해 각 쿼리에 적합한 전문가를 선택합니다. 이는 도메인 특화 데이터에 대한 성능을 극대화할 수 있는 새로운 접근법입니다.

- **Technical Details**: RouterRetriever는 각 도메인에 대해 훈련된 전문가 게이트를 활용하여 쿼리 임베딩(Embedding)을 처리합니다. 쿼리 입력 시, 라우팅 메커니즘이 작동하여 입력 쿼리와 파일럿 임베딩(Pilot Embeddings) 간의 유사도 점수를 계산하고, 가장 높은 점수를 가진 전문가를 선택합니다. 본 모델은 LoRA 모듈을 사용하여 경량화되며, 전문가의 추가 및 제거 시 추가 훈련이 필요 없습니다.

- **Performance Highlights**: BEIR 벤치마크에서 RouterRetriever는 MSMARCO로 훈련된 모델보다 절대 nDCG@10이 +2.1 향상되었으며, 멀티 태스크 훈련 모델보다 +3.2의 성능 향상을 보였습니다. 다양한 도메인 전문가를 추가할수록 지속적인 성능 향상이 나타나며, 도메인 특화 전문가의 투입이 일반 도메인 전문가보다 더 많은 성능 수익을 제공함을 확인했습니다.



### A Fashion Item Recommendation Model in Hyperbolic Spac (https://arxiv.org/abs/2409.02599)
Comments:
          This work was presented at the CVFAD Workshop at CVPR 2024

- **What's New**: 본 연구에서는 사용자의 구매 기록과 아이템의 시각적 데이터를 기반으로 하여 아이템 간의 암묵적인 계층 구조를 포착하기 위해 하이퍼볼릭 기하학을 통합한 패션 아이템 추천 모델(Hyperbolic Visual Attentive Collaborative Filtering, HVACF)을 제안합니다.

- **Technical Details**: 모델 학습에서 우리는 하이퍼볼릭 거리와 유클리드 거리(Euclidean distance)를 동시에 고려하는 멀티태스크 학습(multi-task learning) 프레임워크를 적용했습니다. 이를 통해 모델이 효과적으로 패션 아이템 간의 관계를 학습할 수 있도록 하였습니다. 하이퍼볼릭 스페이스(hyperbolic space)는 계층적 데이터 모델링에 적합한 특성을 가지고 있으며, Poincaré ball 모델을 채택하였습니다.

- **Performance Highlights**: 우리의 실험 결과, 제안된 모델은 유클리드 공간에서만 훈련된 기존 모델들보다 뛰어난 성능을 보였으며, 멀티태스크 학습이 모델 성능에 핵심적인 역할을 한다는 것을 확인하였습니다. Euclidean loss를 제거할 경우, 모델 성능이 급격히 저하됨을 보였습니다.



### AlignGroup: Learning and Aligning Group Consensus with Member Preferences for Group Recommendation (https://arxiv.org/abs/2409.02580)
Comments:
          10 pages, accepted by CIKM 2024

- **What's New**: 본 논문에서는 AlignGroup이라는 새로운 그룹 추천 방법을 제안합니다. 이 방법은 그룹의 동의(Consensus)와 개인 회원의 선호(Preferences)를 동시에 고려하여 그룹 의사결정을 추론하는 데 중점을 둡니다.

- **Technical Details**: AlignGroup은 하이퍼그래프 신경망(Hypergraph Neural Network)을 활용하여 내부 및 외부 그룹 관계를 학습합니다. 또한, 자기 지도 정렬 태스크(Self-supervised Alignment Task)를 도입하여 그룹의 동의와 회원들의 공통 선호를 정렬함으로써 세밀한 그룹 의사결정을 포착합니다.

- **Performance Highlights**: 실제 데이터셋에서 수행한 실험 결과, AlignGroup은 그룹 추천 작업(Group Recommendation Task) 및 사용자 추천 작업(User Recommendation Task) 모두에서 최신 기술(State-of-the-Art)을 초월하는 성능을 보였습니다.



### Deep Adaptive Interest Network: Personalized Recommendation with Context-Aware Learning (https://arxiv.org/abs/2409.02425)
- **What's New**: 개인화 추천 시스템에서 사용자 관심사의 변화와 맥락 정보를 결합하는 새로운 방법론, Deep Adaptive Interest Network (DAIN)을 제안합니다.

- **Technical Details**: DAIN은 딥 러닝 기술을 활용하여 사용자 관심사를 동적으로 모델링하며, 맥락 인식 학습(context-aware learning) 메커니즘을 통합하여 정확하고 적응적인 추천을 가능하게 합니다. 이 모델은 사용자 관심 변화에 대한 실시간 캡처와 함께 추천 결과를 최적화합니다.

- **Performance Highlights**: 여러 공개 데이터셋을 통한 실험에서 DAIN은 추천 성능과 계산 효율성 모두에서 우수한 결과를 보여주며, 개인화 추천 시스템에 대한 새로운 솔루션과 맥락 인식 학습의 응용에 대한 새로운 통찰력을 제공합니다.



### Bioinformatics Retrieval Augmentation Data (BRAD) Digital Assistan (https://arxiv.org/abs/2409.02864)
- **What's New**: 이번 논문에서는 Bioinformatics Retrieval Augmentation Data (BRAD) 디지털 어시스턴트의 프로토타입을 소개합니다. BRAD는 다양한 생물정보학(Bioinformatics) 작업을 처리할 수 있는 도구 모음을 통합하였습니다.

- **Technical Details**: BRAD의 주요 기능으로는 (1) 검색 강화 생성(retrieval augmented generation, RAG)을 통한 향상된 질문-응답 기능, (2) 복잡한 소프트웨어 파이프라인(pipeline)을 작성하고 실행할 수 있는 능력, (3) 개인 및 팀 에이전트 간의 작업을 조직 및 배포할 수 있는 기능이 있습니다.

- **Performance Highlights**: BRAD는 생물정보학 워크플로우의 자동화를 위해 사용되며, 유전자 풍부화(gene enrichment), 아카이브 검색, 자동 코드 생성 및 바이오마커 식별 파이프라인 실행 등의 작업을 수행합니다. BRAD는 디지털 생물학 실험을 위한 가설 생성 및 테스트를 이끌어내는 자체 포함 루프를 기반으로 한 연구실의 디지털 쌍둥이를 개발하는 궁극적인 목표에 한 걸음 다가가는 도구입니다.



### Pooling And Attention: What Are Effective Designs For LLm-Based Embedding Models? (https://arxiv.org/abs/2409.02727)
Comments:
this https URL

- **What's New**: 대규모 실험을 통해 LLM 기반 임베딩 모델의 풀링(embedding) 및 어텐션(attention) 전략을 비교하고, 새로운 Multi-Layers Trainable Pooling 전략을 제안합니다.

- **Technical Details**: 임베딩 모델의 성능을 비교하기 위해 동일한 훈련 데이터와 LLM(base model)을 사용하여 다양한 풀링 방식 및 어텐션 전략으로 훈련된 여러 모델 간의 실험을 수행했습니다. 제안된 Multi-Layers Trainable Pooling 전략은 모든 은닉 층의 출력을 변환하는 방식으로, cross-attention 네트워크를 활용합니다.

- **Performance Highlights**: Bidirectional attention 및 추가적인 trainable pooling layer가 텍스트 유사도 및 정보 검색에서 뛰어난 성능을 보였으나, 클러스터링 및 분류 작업에서는 간단한 설계 방식에 비해 크게 개선되지 않았습니다. Multi-Layers Trainable Pooling 전략은 기존의 방법보다 통계적으로 유의미한 우수성을 입증했습니다.



### iRangeGraph: Improvising Range-dedicated Graphs for Range-filtering Nearest Neighbor Search (https://arxiv.org/abs/2409.02571)
Comments:
          The paper has been accepted by SIGMOD 2025

- **What's New**: 본 연구에서는 RFANN(query) 검색을 위한 새로운 iRangeGraph 방법을 제안하며, 이를 통해 쿼리 범위에 대한 전용 그래프 인덱스를 동적으로 생성할 수 있다.

- **Technical Details**: iRangeGraph는 고차원 유클리드 공간에서 RFANN 쿼리 처리 시, 사전 생성된 탄력 그래프(elemental graphs)를 사용하여 응답 시간과 정확도 간의 적절한 트레이드오프를 이룬다.

- **Performance Highlights**: 실험 결과, iRangeGraph는 moderate한 메모리 소비를 유지하면서 다양한 쿼리 작업에서도 우수하고 안정적인 쿼리 성능을 보여주었다.



### An Effective Tag Assignment Approach for Billboard Advertisemen (https://arxiv.org/abs/2409.02455)
Comments:
          This Paper has been accepted at The 25th International Web Information Systems Engineering Conference (WISE-2024)

- **What's New**: 이 논문에서는 Billboard Advertisement에서의 Tag Assignment Problem을 다루며, 초기 태그와 슬롯을 올바르게 매핑하여 광고의 영향을 극대화하는 방법을 제안합니다. 특히, One-To-Many Bipartite Matching (OMBM)라는 새로운 모델을 사용하여 효과적인 솔루션을 제안합니다.

- **Technical Details**: 이 문제는 NP-hard로 분류되며, 반복적인 태그 할당 방법을 통해 슬롯에 태그를 단계적으로 배치하는 방식을 사용합니다. 또한, 실제 경로 데이터와 빌보드 데이터 세트에 대한 복잡성 분석이 이루어졌습니다. 기존의 2측 매칭 문제에서 발전된 OMBM에서는 태그가 여러 슬롯에 할당될 수 있지만, 그 반대는 불가능합니다.

- **Performance Highlights**: 실험 결과는 제안한 방법이 실제 환경에서의 효과성과 효율성을 보여주며, 다양한 데이터 세트를 통해 실증적으로 검증되었습니다.



### NUDGE: Lightweight Non-Parametric Fine-Tuning of Embeddings for Retrieva (https://arxiv.org/abs/2409.02343)
- **What's New**: 이 논문에서는 NUDGE라는 새로운 비모수(non-parametric) 임베딩 조정 방법을 소개합니다. 이 방법은 기존의 사전 훈련(pre-trained) 모델과 비교하여 더 높은 정확도와 효율성을 제공합니다.

- **Technical Details**: NUDGE는 k-최근접 이웃 검색(k-NN retrieval)에 최적화된 데이터 레코드의 임베딩을 직접 변경하여 정확도를 극대화합니다. 이 방법은 실제로 NP-Hard 문제로 알려져 있으나, 제한된 변형을 통해 효율적으로 해결할 수 있습니다. 개선된 임베딩 변화는 사전 훈련 과정에서 학습된 의미를 왜곡하지 않도록 제한됩니다.

- **Performance Highlights**: 실험을 통해 NUDGE는 9개의 표준 텍스트 및 이미지 검색 데이터셋에서 기존 조정 방법들보다 NDCG@10에서 평균 10% 더 개선된 성능을 보여줍니다. NUDGE 방법은 시간당 200배 빠르게 실행되며, 평균적으로 정확도는 3.3배에서 4.3배 증가합니다.



### Laser: Parameter-Efficient LLM Bi-Tuning for Sequential Recommendation with Collaborative Information (https://arxiv.org/abs/2409.01605)
Comments:
          11 pages, 4 figures

- **What's New**: 본 논문에서는 협업 정보를 활용한 순차 추천 시스템을 위한 효율적인 매개변수 처리 방법인 Laser를 제안합니다. 이는 LLM의 매개변수를 동결시키고 가변적인 가상 토큰을 입력 시퀀스의 앞과 뒤에 삽입하여 최적화하는 방안입니다.

- **Technical Details**: Bi-Tuning 프레임워크는 두 개의 학습 가능한 가상 토큰을 사용하여 입력 시퀀스의 상단과 하단에 배치합니다. 상단은 사용자-아이템 협업 정보를 포함하고, 하단은 LLM의 출력 임베딩을 추천 공간으로 변환합니다. M-Former를 도입하여 다양한 사용자 특성을 반영한 협업 정보를 효과적으로 통합합니다.

- **Performance Highlights**: 실제 데이터셋에 대한 광범위한 실험 결과, Laser는 기존의 최첨단 방법들보다 확연히 높은 추천 정확도를 달성하며, 매개변수 효율성을 입증합니다.



### Blockchain-based Federated Recommendation with Incentive Mechanism (https://arxiv.org/abs/2409.01563)
Comments:
          This paper has been accepted on 2024 Blockchain and Web3 Technology Innovation and Application Exchange Conference (BWTAC 2024)

- **What's New**: 최근 여러 조직들이 데이터 공유 및 모델 훈련을 통해 사용자 프라이버시를 보호하는 연합 추천 기술이 신속히 발전하고 있습니다. 그러나 연합 추천 시스템은 고객 시스템 비용을 증가시키고, 악의적인 참가자들로부터의 공격에도 취약합니다. 이를 해결하기 위해 블록체인 기반의 인센티브 메커니즘을 지원하는 연합 추천 시스템을 제안합니다.

- **Technical Details**: 본 연구에서는 NeuMF와 FedAvg를 기반으로 한 연합 추천 시스템을 구축하였으며, D3QN 기반의 리버스 경매 메커니즘을 도입하여 최적의 클라이언트를 선택합니다. 블록체인을 사용하여 모델의 온체인 증거 저장을 통해 보안성을 높입니다.

- **Performance Highlights**: 실험 결과, 제안한 인센티브 메커니즘을 통해 우수한 훈련 데이터를 보유한 클라이언트를 보다 낮은 비용으로 유치할 수 있었으며, 연합 추천 경제적 이익을 54.9% 증가시키고 추천 성능을 개선하는 것으로 나타났습니다.



### SSD4Rec: A Structured State Space Duality Model for Efficient Sequential Recommendation (https://arxiv.org/abs/2409.01192)
- **What's New**: 본 논문에서는 기존 추천 시스템의 한계를 극복하기 위해 Mamba 아키텍처를 도입하여, 사용자 행동을 효과적으로 모델링할 수 있는 새로운 추천 프레임워크 SSD4Rec를 제안합니다. SSD4Rec는 가변 길이의 아이템 시퀀스를 처리하는 효율적인 방법을 사용하여 사용자 선호도를 파악합니다.

- **Technical Details**: SSD4Rec는 Structured State Space Duality (SSD)를 기반으로 하여, bidirectional Structured State Space Duality (Bi-SSD) 블록을 사용하여 아이템 표현을 처리하고, 패딩(padding)이나 잘림(truncation) 없이 사용자 상호작용의 가변 길위를 원활하게 관리합니다.

- **Performance Highlights**: SSD4Rec는 200 길이의 사용자 상호작용 데이터에서 대표적인 attention 기반 및 SSM 기반 모델인 SASRec와 Mamba4Rec에 비해 각각 154.62% 및 66.21% 빠른 성능을 보여주며, 사용자 선호도 예측에서 뛰어난 추천 성능을 갖추었습니다.



### LLM-PQA: LLM-enhanced Prediction Query Answering (https://arxiv.org/abs/2409.01140)
Comments:
          This paper is accepted as a demo at CIKM 2024

- **What's New**: 이번 논문에서는 자연어로 표현된 예측 쿼리를 처리하기 위한 혁신적인 도구인 LLM-PQA를 소개합니다. 이 도구는 대규모 언어 모델(LLM)의 기능과 데이터 레이크(data lake), 모델 동물원(model zoo)을 통합하여 예측 쿼리의 요구를 충족시키는 최초의 사례입니다.

- **Technical Details**: LLM-PQA는 사용자 쿼리를 벡터로 인코딩하여 그 유사성을 기준으로 적합한 데이터셋과 모델을 검색하는 벡터 검색(vector search) 전략을 사용합니다. 이 시스템은 모델이 없더라도 필요한 경우 요청에 맞춰서 모델을 훈련할 수 있는 능력을 갖추고 있습니다.

- **Performance Highlights**: LLM-PQA는 자연어로 표현된 복잡한 예측 쿼리를 직관적으로 처리할 수 있으며, 높은 정확성을 보장합니다. 예를 들어, LLM-PQA는 19세 여성 비흡연자의 예상 보험료를 예측하는 쿼리를 제대로 처리할 수 있습니다.



### Smart E-commerce Recommendations with Semantic AI (https://arxiv.org/abs/2409.01137)
Comments:
          8 pages

- **What's New**: 본 논문에서는 사용자 요구를 충족하지 못하는 기존의 웹 마이닝(page recommendations) 한계를 극복하기 위해 semantic web mining과 BP 신경망(BP neural networks)를 결합한 혁신적인 솔루션을 제안합니다.

- **Technical Details**: 사용자의 검색 로그(search logs)를 처리하여 콘텐츠 우선순위(content priority), 소요 시간(time spent), 사용자 피드백(user feedback), 추천의 의미(recommendation semantics), 입력 편차(input deviation)의 다섯 가지 핵심 특성(key features)을 추출합니다. 이러한 특성들은 BP 신경망에 입력되어 웹 페이지를 분류(classify)하고 우선순위를 매깁니다.

- **Performance Highlights**: 책 판매 페이지를 사용한 테스트 결과, 제안된 솔루션은 사용자가 필요로 하는 페이지를 신속하고 정확하게 식별할 수 있음을 보여주었습니다. 이 방법은 더욱 관련성 높은 추천을 제공하여 온라인 쇼핑 경험을 향상시킵니다. 또한, 대규모 데이터셋을 처리하고 실시간(real-time) 추천을 제공할 수 있는 능력 덕분에 현대 전자상거래의 복잡한 문제를 해결할 수 있는 확장 가능하고 효율적인 솔루션입니다.



### Improved Diversity-Promoting Collaborative Metric Learning for Recommendation (https://arxiv.org/abs/2409.01012)
Comments:
          arXiv admin note: text overlap with arXiv:2209.15292

- **What's New**: 이 논문은 사용자의 다양한 관심 범주를 반영하는 새로운 방법인 Diversity-Promoting Collaborative Metric Learning (DPCML)을 제안합니다. 기존의 추천 시스템에서 발생하는 편향 문제를 해결하고, 다수의 사용자 표현을 통해 모든 잠재적인 선호가 포착되도록 설계되었습니다.

- **Technical Details**: DPCML에서는 사용자마다 여러 개의 임베딩을 도입하여 아이템에 대한 사용자 선호를 계산합니다. 여기서 두 가지 할당 전략(Basic Preference Assignment, Adaptive Preference Assignment)을 통해 각 사용자의 임베딩 수를 결정합니다. 또한, Diversity Control Regularization Scheme (DCRS)을 사용하여 다중 벡터 표현의 다양성을 관리합니다. 이론적으로, DPCML은 전통적인 CML보다 작은 일반화 오차를 유도할 수 있음을 보입니다.

- **Performance Highlights**: DPCML의 효과는 여러 벤치마크 데이터셋에서의 포괄적인 실험 결과로 입증되었습니다. 이전의 하드 네거티브 샘플링에 의존하는 한계를 극복하는 대체 샘플링 방법도 제시되었습니다.



### A Counterfactual Explanation Framework for Retrieval Models (https://arxiv.org/abs/2409.00860)
- **What's New**: 이 연구는 정보 검색(Information Retrieval) 모델에 대한 최초의 반사적(counterfactual) 설명 프레임워크를 제안하며, 이는 문서의 비관련성(non-relevance)을 이해하는 데 중점을 두고 있습니다. 구체적으로, 문서의 순위를 높이기 위해 추가해야 할 용어와 비관련성의 원인을 규명합니다.

- **Technical Details**: 이 연구는 제약 최적화 방법(constrained optimization technique)을 사용하여 특정 정보 검색 모델에 대한 문서의 순위를 개선하기 위한 용어를 결정합니다. 실험 결과, 평균 60%의 경우 반사적 설정이 문서의 순위를 개선하는 것으로 나타났습니다.

- **Performance Highlights**: 제안된 방법이 BM25와 같은 전통적인 통계 모델 및 DRMM, DSSM, ColBERT와 같은 딥러닝 모델 모두에서 효과적으로 반사적 예측을 수행했으며, 이는 정보 검색 모델의 성능 개선에 기여할 것으로 보입니다.



### Dissecting Temporal Understanding in Text-to-Audio Retrieva (https://arxiv.org/abs/2409.00851)
Comments:
          9 pages, 5 figures, ACM Multimedia 2024, this https URL

- **What's New**: 이번 연구는 text-to-audio retrieval에서의 소리의 시간적 순서(temporal ordering)에 대한 문제를 다루며 기존 모델의 한계를 분석합니다.

- **Technical Details**: 우리는 HTS-AT라는 최신 transformer 기반 오디오 인코더를 사용하여 CNN 기반 모델과의 시간적 이해 능력을 비교합니다. 또한, AudioCaps 및 Clotho 데이터셋과 함께 새로운 합성된 텍스트-오디오 데이터셋을 도입하여 시간적 정보를 평가하는 데 필요한 통제된 환경을 제공합니다.

- **Performance Highlights**: 우리가 제안한 텍스트 기반의 대조 손실 함수는 모델이 이벤트의 시간적 순서를 더 잘 인식하도록 하여, 합성 데이터셋에서 검색 성능이 향상됨을 보였습니다.



### Fair Reciprocal Recommendation in Matching Markets (https://arxiv.org/abs/2409.00720)
Comments:
          Accepted at RecSys2024

- **What's New**: 이 논문에서는 온라인 매칭 플랫폼의 쌍방향 추천 시스템에서 "envy-freeness"라는 새로운 공정성 개념을 도입하여 추천 기회를 정량화하고 있습니다. 이는 기존의 추천 시스템과의 차별점을 강조합니다.

- **Technical Details**: 제안된 방법은 Nash social welfare function을 활용하여 거의 envy-free하는 정책을 찾는 것입니다. 이를 통해 예상 매치 수를 극대화하는 것과 공정성을 보장하는 것 간의 균형을 찾습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 상대적으로 높은 예상 매치 수와 양측의 기회 공정성을 동시에 달성하는 데 효과적임을 보여주었습니다.



### MARS: Matching Attribute-aware Representations for Text-based Sequential Recommendation (https://arxiv.org/abs/2409.00702)
Comments:
          CIKM 2024

- **What's New**: 이번 연구에서는 텍스트 기반의 순차 추천 시스템을 위한 새로운 모델인 MARS(Matching Attribute-aware Representations for Text-based Sequential Recommendation)를 제안합니다. 이 모델은 사용자와 아이템의 다양한 속성을 고려하여 개인화된 추천을 제공합니다.

- **Technical Details**: MARS는 속성 인식(attribute-aware) 텍스트 인코딩을 통해 세분화된 사용자 및 아이템 표현을 추출하고, 속성 기반 상호작용 매칭을 통해 사용자-아이템 점수를 계산합니다. 이 접근 방식은 미세한 세부정보를 유지하면서 서로 다른 사용자 의도를 캡처할 수 있습니다.

- **Performance Highlights**: MARS는 5개의 벤치마크 데이터셋에서 실험을 통해 기존의 순차 추천 모델에 비해 Recall@10에서 최대 24.43%, NDCG@10에서 최대 29.26%의 성능 향상을 달성했습니다.



### A Learnable Agent Collaboration Network Framework for Personalized Multimodal AI Search Engin (https://arxiv.org/abs/2409.00636)
Comments:
          ACMMM 2024 MMGR WORKSHOP

- **What's New**: 본 논문은 Agent Collaboration Network (ACN)이라는 새로운 AI 검색 엔진 프레임워크를 제안합니다. ACN은 다양한 전문 에이전트들이 협력하여 작동하며, 각 에이전트는 Account Manager, Solution Strategist, Information Manager, Content Creator와 같은 독특한 역할을 수행합니다.

- **Technical Details**: ACN 프레임워크는 사용자 프로필 추적 및 멀티모달 콘텐츠 이해 기제를 통합하여 AI 검색 엔진의 응답 품질과 사용자 맞춤화(interactivity)를 향상시킵니다. Reflective Forward Optimization (RFO) 방법을 도입하여 에이전트 간의 온라인 시너지를 지원하며, 사용자 피드백에 신속하게 적응할 수 있는 강력한 상호 작용 유연성을 제공합니다.

- **Performance Highlights**: ACN은 멀티모달 콘텐츠 생성, 개인화된 응답, 복잡한 논리 요구 사항 응답에서 기존 AI 검색 엔진보다 우수한 성능을 보여줍니다. 실험 결과, ACN은 TianGong 및 Perplexity와 같은 최첨단(AI) 검색 엔진에 비해 더욱 매력적인 정보를 생성하고, 사용자 경험을 개선하는 능력이 우수함을 입증했습니다.



### An Enhanced Batch Query Architecture in Real-time Recommendation (https://arxiv.org/abs/2409.00400)
Comments:
          8 pages, 10 figures, CIKM 2024 Applied Research Paper

- **What's New**: 이번 논문에서는 대규모 산업 추천 시스템을 위한 고성능 배치 쿼리 아키텍처를 설계하고 구현한 내용을 담고 있습니다. 주요 기여는 Cacheline-aware probing 방법을 최적화하여 하이브리드 스토리지 키-값 서비스를 구현한 것입니다.

- **Technical Details**: 이 시스템은 Coalesced Hashing을 기반으로 한 NeighborHash라는 새로운 해시 테이블 구조를 도입하여 캐시라인 접근을 최소화하고, 업데이팅 및 쿼리 프로토콜을 최적화하여 강력한 데이터 일관성을 유지합니다. 또한, NVMe 지원 및 핫 데이터와 쿨 데이터를 위한 이중 스토리지를 통합하여 리소스 소비를 줄이는 데 기여합니다.

- **Performance Highlights**: 이 아키텍처는 bilibili recommendation system에 배치되어 10배의 모델 계산을 지원하면서도 리소스 성장을 최소화하고 실시간 성능을 유지하여 추천 성과를 개선하는 데 큰 효과를 보였습니다.



### Evolving Text Data Stream Mining (https://arxiv.org/abs/2409.00010)
Comments:
          134 Pages, 7 Chapters, 38 Figures, 10 Tables

- **What's New**: 최근 온라인 소셜 플랫폼에서 생성되는 방대한 텍스트 데이터를 효과적으로 분석하기 위해 새로운 클러스터링 및 다중 레이블 학습 모델이 제안되었습니다. 이러한 모델은 텍스트 스트림의 고유한 특성을 고려하여 설계되었습니다.

- **Technical Details**: 텍스트 스트림은 시간에 의해 생성된 문서의 정렬된 시퀀스이며, 무한 길이, 데이터 희소성(data sparsity), 진화(evolution)와 같은 특성이 있습니다. 제안된 모델은 고차원 텍스트 데이터로 인한 학습 성능 저하 문제를 해결하고, 문서의 의미적 텍스트 표현을 추출하며, 시간이 지남에 따라 변화하는 주제를 포착하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 제안된 접근 방식은 레이블(label) 부족 문제를 해결하도록 설계되었으며, 기존의 방법들이 필요로 하는 모든 레이블 데이터의 가용성이 부족한 문제를 해결하기 위한 방안을 모색하고 있습니다.



### Web Retrieval Agents for Evidence-Based Misinformation Detection (https://arxiv.org/abs/2409.00009)
Comments:
          1 main figure, 8 tables, 10 pages, 12 figures in Appendix, 7 tables in Appendix

- **What's New**: 이 논문은 잘못된 정보를 탐지하기 위한 에이전트 기반 자동 사실 확인 접근 방식을 개발하였습니다. 강력한 LLM 에이전트와 온라인 웹 검색 에이전트를 결합하여 독립적으로 사용할 때보다 더 나은 결과를 달성하는 방법을 보여줍니다.

- **Technical Details**: 우리는 LLM을 사용해 쿼리를 생성하고, 웹 검색 엔진에 연결된 또 다른 LLM로 이들을 답하는 방식으로 접근법을 평가했습니다. 이 방법은 Vicuna, Mixtral, Claude, GPT-3.5 및 두 개의 GPT-4 모델을 포함한 다양한 모델에서 성능을 평가하고, 웹 검색 기법이 모든 모델의 성능을 향상시키는 것을 발견했습니다.

- **Performance Highlights**: 잘못된 정보 탐지에 대한 성능이 최대 20% 향상되었습니다. 다양한 소스의 사용과 편향, 필요 증거 유형, 그리고 파이프라인의 다양한 구성 요소에 대한 심층 분석을 통해 시스템의 실용성과 기회를 강조하였습니다.



### Sync from the Sea: Retrieving Alignable Videos from Large-Scale Datasets (https://arxiv.org/abs/2409.01445)
Comments:
          ECCV 2024 Oral

- **What's New**: 이 연구는 Alignable Video Retrieval (AVR)이라는 새로운 작업을 도입하여, 주어진 쿼리 비디오에 대해 잘 정렬 가능한 비디오를 대량의 클립에서 식별하고 이를 시간적으로 동기화하는 방법을 제시합니다.

- **Technical Details**: 주요 기술적 기여는 다음과 같습니다: 1) Dynamic Relative Alignment Quality (DRAQ)라는 비디오 정렬 가능성 지표를 도입하여 후보 중 최상의 정렬 가능한 비디오를 식별하고 재순위화합니다; 2) 매 프레임 비디오 특성을 개선하는 일반화된 특성 설계를 제안하며, 3) 사이클 일관성 메트릭을 사용한 AVR 벤치마크와 평가 프로토콜을 제안합니다.

- **Performance Highlights**: Kinetics700을 포함한 3개의 데이터셋에서의 실험 결과, 제안된 방법이 다양한 데이터셋으로부터 정렬 가능한 비디오 쌍을 식별하는 데 효과적임을 입증했습니다.



### Know When to Fuse: Investigating Non-English Hybrid Retrieval in the Legal Domain (https://arxiv.org/abs/2409.01357)
Comments:
          Under review

- **What's New**: 이번 연구에서는 프랑스어 법률 도메인에서 하이브리드 검색(hybrid search)의 효과를 평가하며, 다양한 검색 모델을 조합했다. 영어 도메인에서 한정된 검색 방법에 대한 기존 연구와 차별화된다.

- **Technical Details**: 법률 검색을 위한 다양한 도메인 일반 검색 모델을 조합하여 평가하고, 한정된 도메인 특화 학습 데이터를 가정하여 영문 외의 언어 및 분야에서의 하이브리드 검색의 가능성을 탐구한다.  실험 방법으로는 late fusion 기법을 활용하고, 각 모델의 성능은 평균 역순위(MRR@10) 및 평균 r-precision(RP)로 측정한다.

- **Performance Highlights**: 영어 데이터셋 외에도 프랑스어 법률 도메인에서 하이브리드 검색의 일반화 가능성을 보여주었으며, 하이브리드 모델 조합이 단일 모델보다 더 좋은 성능을 생산했다. 그러나 도메인 특화 모델을 사용할 때는 조합이 성능을 저하시킬 수 있음을 관찰했다.



### Real World Conversational Entity Linking Requires More Than Zeroshots (https://arxiv.org/abs/2409.01152)
- **What's New**: 본 연구는 대화형 시스템에서의 에지 리킹(Entity Linking, EL) 모델의 실제 적용에서 발생하는 어려움에 대해 집중적으로 다루고 있습니다. 기존의 학습 및 평가 방식이 실제의 복잡한 대화 상황을 충분히 반영하지 못하고 있다는 점을 강조하며, 새로운 평가 시나리오와 대화 데이터셋을 제안합니다.

- **Technical Details**: 평가에 있어, 본 연구는 두 개의 지식 베이스(KB)인 Fandom과 Wikipedia를 사용하였으며, Fandom의 엔티티에 기반한 레딧(Reddit) 논의 내용을 토대로 새로운 제로샷(Zero-shot) 대화형 엔티티 링킹 데이터셋을 구축하였습니다. 제안된 데이터셋은 사용자가 Fandom 웹사이트에 하이퍼링크를 포함함으로써 엔티티를 분별하는 예를 포함하며, 뚜렷한 노이즈와 비정형 데이터를 처리할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 현재의 제로샷 EL 모델은 새로운 도메인 특화 KB에 노출될 때 성능이 현저히 저하되는 것을 보여주었으며, 이는 알맞은 학습 없이 대화 상황에 적응하는 데 제한적임을 시사합니다. 제안된 평가 환경은 연구자들이 실질적인 EL 문제를 해결하는 데 도움을 줄 것으로 예상되며, 데이터셋 또한 공개되어 연구에 기여할 것입니다.



### Evidential Transformers for Improved Image Retrieva (https://arxiv.org/abs/2409.01082)
Comments:
          6 pages, 6 figures, To be presented at the 3rd Workshop on Uncertainty Quantification for Computer Vision, at the ECCV 2024 conference in Milan, Italy

- **What's New**: 본 논문에서는 불확실성 기반의 트랜스포머 모델인 Evidential Transformer를 소개하여 이미지 검색에서 개선된 성능을 보여줍니다. 기존의 멀티 클래스 분류 기반의 딥 메트릭 학습을 능가하는 확률적 방법을 이미지 검색에 통합하여 안정적이고 신뢰할 수 있는 결과를 달성했습니다.

- **Technical Details**: Evidential learning은 이미지 검색 과제에서 불확실성을 정량화하는 효과적인 방법으로, 전통적인 결정론적 신경망과는 달리, 두 번째 차원의 확률 분포를 모델링함으로써 더욱 견고하고 정보가 풍부한 프레임워크를 제공한다. Dirichlet 분포를 사용하여 클래스의 접근성을 표현하며, 이러한 접근 방식을 통해 CNN을 기반으로 하는 새로운 이미지 검색 방법을 제시합니다.

- **Performance Highlights**: CUB-200-2011 및 SOP 데이터셋에서 기존의 이미지 검색 방법들을 능가하는 성능을 입증하였고, 기존의 기준 방법들과 비교하여 불확실성 추정을 기반으로 한 새로운 재순위 방법을 통해서도 뛰어난 결과를 보였습니다.



### Towards Investigating Biases in Spoken Conversational Search (https://arxiv.org/abs/2409.00890)
Comments:
          Accepted Late-Breaking Results at ACM ICMI Companion 2024

- **What's New**: 이 연구는 음성 기반 시스템에서의 사용자 태도 변화 및 편향 연구의 필요성을 강조합니다. 특히 Spoken Conversational Search (SCS) 환경에서의 다양한 관점을 효과적으로 제공하는 방법을 모색하고 있습니다.

- **Technical Details**: 이 논문은 SCS 환경에서 정보 탐색과정에서의 사용자 결정에 영향을 미치는 편향(bias)과 태도 변화(attitude change)에 대한 연구 질문을 설정하고, 음성 기반 검색 시스템의 실험 설계를 제안합니다. 실험에서는 Passage의 순서(order)와 노출(exposure)의 변화가 사용자 태도에 미치는 영향도 분석합니다.

- **Performance Highlights**: 이 연구는 SCS 환경에서 사용자에게 다양한 관점을 공정하게 전달하기 위해 음성 응답의 순서와 발언 비율에 대한 분석을 시도하며, 이는 기존의 스크린 기반 웹 검색에서 관찰된 편향 효과를 탐구합니다.



### The Design of an LLM-powered Unstructured Analytics System (https://arxiv.org/abs/2409.00847)
Comments:
          6 pages, 3 figures, fixed typos

- **What's New**: 이 논문에서는 LLMs (Large Language Models)를 활용한 새로운 비구조적 분석 시스템인 Aryn의 설계와 그 동기 부여가 되는 사용 사례를 설명합니다. Aryn은 자연어로 쿼리를 지정하면, 이를 통해 문서의 의미적 계획을 자동으로 결정하고 실행하여 대형 비구조적 데이터에서 답변을 도출합니다.

- **Technical Details**: Aryn의 핵심은 Sycamore라는 선언적 문서 처리 엔진으로, 이는 Ray를 기반으로 구축되었습니다. Sycamore는 DocSets라는 신뢰할 수 있는 분산 추상화를 제공하며, 사용자는 대규모로 복잡한 문서를 분석하고, 강화하고, 변환할 수 있습니다. Aryn에는 자연어 쿼리를 Sycamore 스크립트로 변환하는 Luna라는 쿼리 플래너와 원시 PDF 및 문서 이미지를 DocSets로 변환하는 Aryn Partitioner가 포함되어 있습니다.

- **Performance Highlights**: Aryn은 실제 응용 사례로 NTSB (National Transportation Safety Board) 사고 보고서를 분석하는 예시를 보여줍니다. 이 전반에서 Aryn 시스템의 작동 방식을 여러 가지 사용 사례에 걸쳐 시연하며, 사용자 인터페이스를 통해 생성된 계획을 검사, 분석 및 디버그하는 과정과 대규모 비구조적 문서를 쉽게 분석할 수 있는 Sycamore 프로그래밍 프레임워크의 단순함을 강조합니다.



### Building FKG.in: a Knowledge Graph for Indian Food (https://arxiv.org/abs/2409.00830)
Comments:
          14 pages, 3 figures, 25 references, Formal Ontology in Information Systems Conference 2024 - Integrated Food Ontology Workshop

- **What's New**: 이번 논문에서는 인도 음식에 대한 정보를 자동으로 수집하기 위한 지식 그래프 구축을 위한 본체 설계와 지식 공학(knowledge engineering) 기법을 제안합니다. 특히 요리, 레시피, 성분, 영양상의 모든 지식을 포괄적으로 수집하는 지능형 방법의 설계를 다루고 있습니다.

- **Technical Details**: 제안된 기법은 공공 도메인에서 레시피 블로그 사이트의 정보를 수집하고 AI(Artificial Intelligence), LLM(Large Language Model), 언어 기술을 활용하여 인도 음식의 지식 그래프를 구축합니다. 이 과정에는 인간의 개입(human-in-the-loop)도 포함되어 신뢰성을 높이고 있습니다.

- **Performance Highlights**: FKG.in이라는 지식 그래프는 인도 요리의 전모를 포괄하여 후속 음식 컴퓨팅 애플리케이션을 구축하는 디지털 자원으로 활용될 것입니다. 이 연구는 다른 도메인에도 적용 가능하며, AI 기반 스마트 분석 및 개인화된 디지털 건강을 위한 추천 시스템 구축에 기여할 것입니다.



### Hound: Hunting Supervision Signals for Few and Zero Shot Node Classification on Text-attributed Graph (https://arxiv.org/abs/2409.00727)
- **What's New**: 본 연구에서는 텍스트 속성 그래프(Text-attributed graph, TAG)에서의 few-shot 및 zero-shot 노드 분류의 정확성을 향상시키기 위해 Hound라는 새로운 방법론을 제안합니다. 기존 방법들이 대조 손실(contrastive loss)만을 사용했으나, Hound는 더 많은 감독 신호(supervision signals)를 제공하는 것을 목표로 합니다.

- **Technical Details**: Hound는 세 가지 증강 기법(augmentation techniques)을 도입했습니다: 노드 변형(node perturbation), 텍스트 일치(text matching), 의미의 부정(semantics negation). 노드 변형은 그래프에서 엣지를 무작위로 추가/삭제하여 다양한 노드 임베딩을 생성합니다. 텍스트 일치는 유사한 임베딩을 가진 텍스트를 검색하여 노드와 일치시킵니다. 의미의 부정은 원래 텍스트와 반대 의미를 가지는 부정적인 텍스트를 생성하여 본래의 노드 및 텍스트와 대비하는 것입니다.

- **Performance Highlights**: Hound는 5개 데이터셋에 대해 13개의 최신 기법과 비교한 결과, 모든 기준선(baselines)에서 일관되게 더 높은 정확성을 달성했습니다. 특히, few-shot 및 zero-shot 분류에서 각각 평균 4.6% 및 8.8%의 정확도가 향상되었습니다.



### Genetic Approach to Mitigate Hallucination in Generative IR (https://arxiv.org/abs/2409.00085)
Comments:
          Gen-IR@SIGIR 2024

- **What's New**: 새로운 연구는 Generative language models가 Hallucination(환각)을 발생시키는 문제를 해결하기 위해 Grounded Answer Generation(지속적인 답변 생성) 접근 방식을 개선하여 accuracy(정확도)를 4배 증가시키는 새로운 'balanced fitness function'(균형 잡힌 적합도 함수)을 실제로 적용하고 있습니다.

- **Technical Details**: 이 방법은 'cross-encoder model'(교차 인코더 모델)과 'n-gram overlap metric'(n-그램 중첩 메트릭)을 활용하여 relevance(관련성)와 grounding(기반 데이터의 근거여부)를 유지합니다. GAuGE(Genetic Approach using Grounded Evolution)라는 이름으로 알려진 이 방법론은 첫 단계에서 BM25(정보 검색 알고리즘)를 사용하여 문서를 검색한 후, Electra(교차 인코더 모델)를 활용하여 재정렬합니다.

- **Performance Highlights**: GAuGE는 세 가지 데이터셋을 사용하여 평가하였으며, hallucination을 감소시키면서도 높은 relevance를 유지하며, 여러 개의 seed document(기초 문서)를 사용하여 포괄적인 답변을 생성합니다. GAuGE는 특히 최소한의 hallucination으로 사실 결과를 생성하는 데 성공하였습니다.



New uploads on arXiv(cs.CV)

### HiPrompt: Tuning-free Higher-Resolution Generation with Hierarchical MLLM Prompts (https://arxiv.org/abs/2409.02919)
- **What's New**: 이 논문에서는 HiPrompt라는 새로운 방법론을 제안합니다. HiPrompt는 고해상도 이미지 생성을 위한 계층적 프롬프트(hierarchical prompts)를 도입하여 개체 반복(object repetition) 및 구조적 아티팩트(structural artifacts) 문제를 해결합니다. 이 방법은 사용자 입력에서 전반적인 내용을 설명하는 글로벌 가이던스(global guidance)와 패치 단위(patch-wise) 묘사를 사용하는 로컬 가이던스(local guidance)를 제공합니다.

- **Technical Details**: HiPrompt는 다단계 프롬프트를 기반으로 하여 이미지 생성 시 각 요소를 제어하는 데 사용됩니다. 생성된 노이즈는 저주파(low-frequency)와 고주파(high-frequency) 공간 성분으로 분해되며, 이 성분들은 세밀한 로컬 및 광범위한 글로벌 프롬프트에 조건이 달려 있습니다. 이 기법은 단계별 노이즈 제거(parallel denoising)를 가능하게 하며, 이미지의 로컬 및 글로벌 측면의 일관성을 높입니다.

- **Performance Highlights**: HiPrompt는 기존 최첨단 방법들과 비교하여 고해상도 이미지 생성에서 성능이 우수하며, 특히 개체 반복을 크게 줄이고 구조적 품질을 향상시킵니다. 실제 실험 결과에 따르면, HiPrompt는 다양한 해상도에서 고품질 이미지를 생성할 수 있으며, 확대 시에도 세부적인 구조가 일관성 있게 유지됩니다.



### UC-NeRF: Uncertainty-aware Conditional Neural Radiance Fields from Endoscopic Sparse Views (https://arxiv.org/abs/2409.02917)
- **What's New**: 본 논문은 외과 수술 장면에서의 새로운 시점 합성을 위해 불확실성 인식 조건부 NeRF(Neural Radiance Field)인 UC-NeRF를 제안합니다. 이는 희소한 외과적 관점에 대한 기하학-광선(geometry-radiance) 모호성을 해결하기 위함입니다.

- **Technical Details**: UC-NeRF의 핵심은 다중 시점 불확실성 추정을 포함하여 외과적 장면의 심각한 광학적 불일치를 조정하는 것입니다. 이를 위해 다중 시점 스테레오 네트워크를 이용해 기하학적 대응관계를 수립하고, 불확실성 추정 및 특징 프라이어를 생성합니다. 또한, 기본 적응형 NeRF 네트워크를 설계하여 불확실성 추정을 활용해 광학적 불일치를 처리하고, 단안 기하학 프라이어의 증류(distillation)를 통해 기하학 학습을 향상시킵니다.

- **Performance Highlights**: SCARED 및 Hamlyn 데이터세트에서 실험을 통해 UC-NeRF가 기존의 최신 기법들보다 우수한 성능을 보이며, 외과적 장면에서의 새로운 시점 합성에서 효과성과 효율성을 일관되게 개선함을 입증했습니다.



### Can LVLMs Obtain a Driver's License? A Benchmark Towards Reliable AGI for Autonomous Driving (https://arxiv.org/abs/2409.02914)
- **What's New**: 본 논문에서는 지능형 운전 지식 기반(Intelligent Driving Knowledge Base, IDKB)이라는 대규모 비전-언어 데이터셋을 처음으로 소개합니다. 이 데이터셋은 이론과 실제 운전 지식을 모두 포함하고 있으며, 주행 안전과 관련된 교통 규칙 및 운전 기술에 대한 명확한 지침을 제공합니다.

- **Technical Details**: IDKB는 세계 여러 나라에서 수집된 100만 개 이상의 데이터 항목으로 구성되어 있으며, 전반적인 운전 이론, 시험 데이터 및 실제 도로 테스트 데이터가 포함되어 있습니다. 이 데이터셋은 주행 핸드북, 이론 테스트 질문 및 시뮬레이션된 도로 시나리오를 통해 운전 지식을 체계적으로 학습할 수 있도록 구성되었습니다. CARLA로 시뮬레이션된 다양한 도로 시나리오를 포함하여 기후, 조명 및 교통 조건의 변화를 처리할 수 있게 합니다.

- **Performance Highlights**: 15개의 기존 LVLM에 대해 IDKB를 사용한 포괄적인 평가를 진행하였고, 이들 모델이 드라이빙 분야 지식의 부족을 보임에 따라, 고품질의 체계적이고 다양한 운전 지식 데이터로 세부 조정할 필요성을 강조했습니다. 본 데이터셋을 사용하여 모델의 성능을 향상시키는 결과를 보여주어, 특화된 도메인 지식의 중요성을 분명히 했습니다.



### SITAR: Semi-supervised Image Transformer for Action Recognition (https://arxiv.org/abs/2409.02910)
Comments:
          Accepted at ICPR 2024

- **What's New**: 이 논문에서는 적은 수의 라벨이 부착된 비디오와 많은 수의 라벨이 없는 비디오를 활용하여 반지도 학습(semi-supervised learning) 환경에서 비디오 액션 인식을 개선하는 새로운 접근 방식을 제안합니다. 이 접근 방식은 정보 제공을 위해 비디오 프레임을 재배열하여 super images를 생성하고, contrastive learning을 사용하여 효율적인 학습을 수행합니다.

- **Technical Details**: 본 연구에서는 비디오를 2D 이미지로 재구성하여 super images를 생성하고, 2개의 경로(pathway)를 통해 시간적으로 증강된 super images의 표현을 생성합니다. 이를 위해 2D 이미지-트랜스포머(image-transformer)를 사용하고, 서로 다른 비디오 간의 표현의 유사성을 최소화하며 동일 비디오의 표현의 유사성을 최대화하는 contrastive loss를 적용합니다.

- **Performance Highlights**: SITAR(Semi-Supervised Image Transformer for Action Recognition)라는 제안된 방법은 여러 벤치마크 데이터셋에서 기존의 최신 방법들과 비교하여 우수한 성능을 보이며, 계산 비용과 파라미터 효율성 또한 개선되었습니다.



### Multi-stream deep learning framework to predict mild cognitive impairment with Rey Complex Figure Tes (https://arxiv.org/abs/2409.02883)
Comments:
          20 pages, 3 figures, 2 tables

- **What's New**: 이 논문에서는 Rey Complex Figure Test (RCFT)를 활용하여 인지 기능을 평가할 수 있는 새로운 다중 스트림(deep learning) 딥러닝 프레임워크를 제안합니다. 이 모델은 두 가지의 프로세싱 스트림(spatial stream과 scoring stream)을 통합하여 작동하며, 1,740명의 한국인 샘플을 기반으로 훈련되었습니다.

- **Technical Details**: 제안된 모델은 두 가지 스트림을 통합하여 사용합니다. 첫 번째 스트림은 원본 RCFT 이미지를 사용하는 multi-head self-attention 기반의 spatial stream이며, 두 번째 스트림은 자동 점수화 시스템을 이용한 scoring stream입니다. 이 모델은 222명의 외부 환자 데이터를 통해 검증되었습니다.

- **Performance Highlights**: 제안된 다중 스트림 모델은 외부 검증에서 기존의 베이스라인 모델들보다 우수한 성능(AUC = 0.872, Accuracy = 0.781)을 보였습니다. 이는 모델이 미세한 인지 결함을 감지하는 능력을 높여주며, 다양한 임상 환경에서도 더 나은 신뢰성을 제공할 수 있도록 합니다.



### Benchmarking Spurious Bias in Few-Shot Image Classifiers (https://arxiv.org/abs/2409.02882)
Comments:
          Accepted to ECCV 2024

- **What's New**: 본 논문에서는 spurious bias에 대한 few-shot 이미지 분류기의 강건성을 평가하는 체계적이고 철저한 벤치마크 프레임워크인 FewSTAB를 제안합니다. 이 프레임워크는 biased attributes를 활용하여 평가 작업을 생성하고, 기존 테스트 데이터를 사용해 spurious bias를 자동으로 벤치마킹할 수 있도록 합니다.

- **Technical Details**: FewSTAB는 attribute 기반 샘플 선택 전략을 통해 지원(set) 및 질의(query) 샘플에 spurious correlations를 통제하며, 이를 통해 few-shot 분류기가 spurious bias에 얼마나 취약한지 평가합니다. 또한, 사전 훈련된 비전-언어 모델(vision-language model)을 활용하여 이미지를 기반으로 다양한 spurious correlations를 시뮬레이션 할 수 있습니다.

- **Performance Highlights**: FewSTAB의 효과는 세 개의 데이터셋에서 열 개의 few-shot 학습 방법을 적용한 실험을 통해 입증되었습니다. 이를 통해 다양한 강건성을 가진 few-shot 분류기 디자인 발전에 기여할 것으로 기대됩니다.



### The Impact of Balancing Real and Synthetic Data on Accuracy and Fairness in Face Recognition (https://arxiv.org/abs/2409.02867)
Comments:
          Accepted at Synthetic Data for Computer Vision Workshop - Side Event at ECCV 2024

- **What's New**: 본 논문은 실제 데이터와 합성 데이터의 인구통계학적으로 균형 잡힌 조합이 얼굴 인식 모델의 정확성과 공정성에 미치는 영향을 조사합니다. 특히, 합성 데이터의 효과성과 공정성 평가에 중점을 두었습니다.

- **Technical Details**: 연구에서는 Diffusion Models (DMs)와 Generative Adversarial Networks (GANs)를 사용하여 합성 데이터셋의 인구통계학적 균형을 맞추기 위해 여러 생성 방법을 도입했습니다. 또한 최신 얼굴 인코더를 훈련하고, 합성 이미지 및 실제 이미지를 사용한 평가를 수행했습니다.

- **Performance Highlights**: 연구 결과, (i) DMs를 통해 생성된 훈련 데이터가 정확성을 향상시키는데 효과적이라는 점과 (ii) 인구통계학적 균형을 고려한 데이터 사용이 공정성에 미치는 영향이 미미하다는 점이 밝혀졌습니다. 대다수의 경우 공정성 점수가 변경되지 않거나 악화되었습니다.



### Hybrid-Segmentor: A Hybrid Approach to Automated Fine-Grained Crack Segmentation in Civil Infrastructur (https://arxiv.org/abs/2409.02866)
Comments:
          25 pages, 6 figures

- **What's New**: 이번 연구에서는 Hybrid-Segmentor라는 새로운 모델을 제안하여 인프라의 균열 탐지 및 분할 성능을 향상시킵니다. 이는 encoder-decoder 기반의 접근 방식으로 다양한 형태와 크기의 균열을 효과적으로 구별할 수 있습니다.

- **Technical Details**: Hybrid-Segmentor는 self-attention 모델을 인코더 수준에 통합하여 계산 성능을 유지하면서도 높은 일반화 능력을 제공합니다. 이 모델은 기존의 벤치마크 모델보다 5개의 정량적 지표에서 더 우수한 성능을 보였습니다. (정확도 0.971, 정밀도 0.804, 재현율 0.744, F1-score 0.770, IoU 점수 0.630)

- **Performance Highlights**: 새롭게 제안된 Hybrid-Segmentor 모델은 기존의 여러 모델들보다 뛰어난 성능을 보여 주었으며, 특히 다양한 표면 타입과 도전적인 이미징 조건에서도 효과적으로 작동합니다.



### Human-VDM: Learning Single-Image 3D Human Gaussian Splatting from Video Diffusion Models (https://arxiv.org/abs/2409.02851)
Comments:
          14 Pages, 8 figures, Project page: this https URL

- **What's New**: 이번 연구에서는 Human-VDM이라는 새로운 방법론을 제안하여 단일 RGB 이미지로부터 일관된 뷰의 3D 인간 모델을 생성할 수 있게 되었습니다. 이는 비디오 디퓨전 모델을 활용하여 지오메트리(geometry)와 리얼리틱한 텍스처를 동시에 향상시킵니다.

- **Technical Details**: Human-VDM은 세 가지 모듈로 구성됩니다: 뷰 일관성 인간 비디오 디퓨전 모듈, 비디오 증강 모듈(super-resolution 및 비디오 프레임 보간 포함)과 3D 인간 가우시안 스플래팅 모듈입니다. 이를 통해 단일 이미지를 입력으로 하여 연속적인 비디오를 생성하고, 이후 품질을 향상시킵니다.

- **Performance Highlights**: 실험 결과, Human-VDM은 단일 이미지를 활용하여 고품질의 3D 인간 모델을 생성하며, 기존의 최첨단 방법들에 비해 양질의 생성 결과를 보여주었습니다.



### MaDis-Stereo: Enhanced Stereo Matching via Distilled Masked Image Modeling (https://arxiv.org/abs/2409.02846)
- **What's New**: 이 논문에서는 Masked Image Modeling (MIM) 기반의 새로운 스테레오 매칭 모델인 MaDis-Stereo를 제안합니다. 이 모델은 Transformer 모델의 지역적 유도 편향(locality inductive bias)을 강화하고, 주어진 랜덤하게 마스킹된 스테레오 이미지를 입력으로 사용하여 이미지 재구성과 깊이 예측 작업을 동시에 수행합니다.

- **Technical Details**: MaDis-Stereo는 MIM을 활용하고, Exponential Moving Average (EMA)를 사용하여 업데이트되는 보조 네트워크(teacher)와 원본 스테레오 모델(student) 간의 지식 전달을 통해 안정적인 학습을 달성합니다. 이 모델은 마스킹 비율을 40%로 설정하여 재구성된 토큰의 두 가지 도전 과제를 해결합니다.

- **Performance Highlights**: 우리의 모델은 ETH3D와 KITTI 2015와 같은 여러 스테레오 매칭 데이터셋에서 기존 방법들과 비교하여 향상된 성능을 보였으며, 지역적 유도 편향을 효과적으로 활용하는 것을 입증했습니다.



### iConFormer: Dynamic Parameter-Efficient Tuning with Input-Conditioned Adaptation (https://arxiv.org/abs/2409.02838)
- **What's New**: 이 논문에서는 입력 인스턴스에 조건화된 동적 어댑터를 활용하는 새로운 Parameter Efficient Fine-Tuning (PEFT) 접근 방식인 iConFormer를 제안합니다. 이 방법은 다양한 다운스트림 작업에서 유연한 학습 가능성을 보장하기 위해 인스턴스 수준의 특징 변환을 가능하게 하는 입력 조건 네트워크(iCoN)를 도입합니다.

- **Technical Details**: iConFormer는 기존의 PEFT 방법들과는 다르게 각 입력 특성에 대한 파라미터를 동적으로 생성하여 더욱 효과적으로 작업 특정(task-specific) 및 세세한 정보를 포착합니다. 실험 결과, Transformer의 백본 파라미터의 1.6%에서 2.8%만 수정하여도 monocular depth estimation, semantic segmentation, image classification, instance segmentation에서 FFT와 유사한 성능을 달성했습니다.

- **Performance Highlights**: 제안된 방법은 CIFAR100 이미지 분류 및 COCO instance segmentation 작업에서 FFT 성능을 초과했으며, NYU-v2 monocular depth estimation 작업에서도 기존 PEFT 방법을 능가하는 결과를 보여주었습니다. 다양한 실험을 통해 iConFormer가 1.6%에서 2.8%의 Transformer 백본 파라미터 조정만으로 뛰어난 성능을 달성함을 입증하였습니다.



### ExpLLM: Towards Chain of Thought for Facial Expression Recognition (https://arxiv.org/abs/2409.02828)
Comments:
          project page: this https URL

- **What's New**: 이번 연구에서는 Facial Expression Recognition (FER) 분야에서 중요한 통찰력을 제공하는 새로운 방법론, ExpLLM을 제안합니다. 이 방법은 대규모 언어 모델을 활용하여 facial expression을 인지하기 위한 Chain of Thought (CoT) 기법을 개발하였습니다.

- **Technical Details**: ExpLLM은 facial action units (AUs) 분석을 통해 emotion의 전체적 해석과 결론을 도출하는 3개의 주요 관점으로 구성된 CoT 메커니즘을 설계하였습니다. 또한, Exp-CoT Engine을 통해 instruction-description pair를 생성하여 모델의 학습 효율성을 개선합니다.

- **Performance Highlights**: RAF-DB와 AffectNet 데이터셋에서 ExpLLM은 기존의 최신 FER 방법들보다 뛰어난 성능을 보였으며, 특히 micro-expressions 인식에서 GPT-4o보다 우월한 정확도를 기록하였습니다.



### Deep Learning Meets Satellite Images -- An Evaluation on Handcrafted and Learning-based Features for Multi-date Satellite Stereo Images (https://arxiv.org/abs/2409.02825)
Comments:
          ECCV2024 Workshop - TradiCV

- **What's New**: 이번 연구는 위성 이미지에서 피처 매칭의 성능을 평가하기 위해 전통적인 방법(SIFT)과 최신 딥러닝 기반 방법(SuperGlue, LightGlue 등)을 비교합니다. 이 연구는 약 500개의 서로 다른 오프트랙 스테레오 이미지 쌍을 사용하여 수행되었습니다.

- **Technical Details**: 연구에서는 SIFT와 7가지 딥러닝 매칭 방법(SuperGlue, LightGlue, LoFTR, ASpanFormer, DKM, GIM-LightGlue, GIM-DKM)의 성능을 비교하였습니다. 특히 위성 이미지에서 나타나는 스펙트럼 왜곡, 긴 베이스라인, 넓은 교차 각도로 인한 도전 과제를 다루며, 유지된 정확성을 바탕으로 평가하였습니다.

- **Performance Highlights**: 전통적인 SIFT 기법도 딥러닝 기반 방법들과 여전히 경쟁력을 가지며, 특정 시나리오에서는 학습 기반 방법이 매우 유망함을 보여주었습니다. 전체 496개의 스테레오 쌍을 기반으로 한 결과는 딥러닝 접근 방식이 성능을 개선할 수 있는 가능성을 나타냅니다.



### MedUnA: Language guided Unsupervised Adaptation of Vision-Language Models for Medical Image Classification (https://arxiv.org/abs/2409.02729)
- **What's New**: 본 논문에서는 제한된 라벨링 데이터 문제를 해결하기 위해 	extit{MedUnA}라는 새로운 언어 안내 의료 비지도 적응 방법을 제안합니다. 이는 기존의 사전 훈련-미세 조정 접근 방식 대신 Vision-Language 모델(VLM)에서 비주얼-텍스추얼 정렬을 활용하여 비지도 학습을 촉진합니다.

- **Technical Details**: 	extit{MedUnA}는 두 단계의 훈련으로 구성됩니다: Adapter Pre-training과 Unsupervised Learning. 첫 번째 단계에서는 Large Language Model(LLM)을 통해 생성된 클래스 레이블에 해당하는 설명을 사용하여 텍스트 임베딩을 생성하고, 이를 통해 가벼운 어댑터를 훈련합니다. 두 번째 단계에서는 훈련된 어댑터를 	extit{MedCLIP}의 비주얼 인코더에 통합하고, 대비 엔트로피 기반 손실과 프로프트 조정을 통해 비주얼 임베딩을 정렬합니다.

- **Performance Highlights**: 	extit{MedUnA}는 유방방사선 사진, 안저 사진, 피부 병변 이미지를 포함한 세 가지 데이터 양식에서 성능을 평가하고, 다양한 데이터 세트에서 기존 모델에 비해 상당한 정확성 향상을 보였습니다.



### GET-UP: GEomeTric-aware Depth Estimation with Radar Points UPsampling (https://arxiv.org/abs/2409.02720)
Comments:
          Accepted by WACV 2025

- **What's New**: 이번 연구에서는 GET-UP이라는 새로운 깊이 추정 프레임워크를 제안합니다. 이 프레임워크는 레이더 데이터를 활용하여 2D와 3D 정보를 상호 교환하고 집계합니다.

- **Technical Details**: GET-UP은 attention-enhanced Graph Neural Networks (GNN)를 사용하여 레이더 포인트 클라우드 내의 귀중한 기하학적 정보를 활용합니다. 또한, point cloud upsampling을 통해 레이더 포인트 클라우드를 밀집하게 만들고 LiDAR 데이터의 지침을 받아 추가적인 3D 기능을 도출합니다.

- **Performance Highlights**: GET-UP은 nuScenes 데이터셋에서 기존 최첨단 모델에 비해 MAE에서 15.3%, RMSE에서 14.7%의 성능 향상을 기록하였습니다.



### LIPIDS: Learning-based Illumination Planning In Discretized (Light) Space for Photometric Stereo (https://arxiv.org/abs/2409.02716)
Comments:
          Accepted in WACV 2025

- **What's New**: 이 논문에서는 LIPIDS (Learning-based Illumination Planning In Discretized light Space)를 소개하여 임의의 조명 분포 아래에서 최적의 조명 구성을 달성하는 방법을 제시합니다. 이 방법은 조명 방향을 최적화하는 Light Sampling Network (LSNet)를 통해서 surface normals를 효과적으로 추정할 수 있게 합니다.

- **Technical Details**: LIPIDS는 정량적이면서 구조화된 조명 샘플링을 가능하게 하며, 조명 설정을 통해 표면 법선을 최소화하는 normal regression network와 결합됩니다. LSNet은 주어진 조명 수에 대해 최적의 조명 구성 (lighting configuration)을 선택하도록 학습합니다. 이 방식은 기존의 photometric stereo 방법들과 통합되어 표면 법선의 추정이 가능하게 합니다.

- **Performance Highlights**: LIPIDS를 통한 조명 구성은 기존의 조명 계획 방법들보다 우수하거나 유사한 성능을 보이며, 합성 및 실제 데이터셋에 대한 광범위한 질적 및 양적 분석을 통해 효과성을 입증하였습니다.



### Recoverable Anonymization for Pose Estimation: A Privacy-Enhancing Approach (https://arxiv.org/abs/2409.02715)
- **What's New**: 본 연구는 개인 정보 보호를 위한 혁신적인 시스템을 제안하며, 이를 통해 높은 성능의 인간 포즈 추정(Human Pose Estimation, HPE)을 유지하면서 개인 정보 보호가 강화된 초상화를 생성합니다.

- **Technical Details**: 제안하는 시스템은 크게 세 가지 모듈로 구성됩니다: (1) 개인 정보 보호 모듈은 cGAN을 사용하여 개인 정보를 익명화하면서 다운스트림 작업에 필요한 특성을 보존합니다. (2) 개인 정보 복구 모듈은 toestemming을 가진 경우 원래의 개인 정보를 복구할 수 있도록 cGAN 쌍을 사용하여 최적화됩니다. (3) 포즈 추정기는 개인 정보 보호와 복구된 이미지를 모두 분석합니다.

- **Performance Highlights**: 실험 결과, 제안된 시스템이 개인 정보 보호, 원본 이미지 복구, HPE에서 강력한 성능을 보인다는 것을 입증하였습니다. 특히, 개인 정보 보호된 이미지에서 평균 정밀도는 약 10% 향상되었고, 복구된 이미지는 정확한 복구와 HPE 관련 정보의 적응적 주입으로 약 3% 품질 향상이 이루어졌습니다.



### MOOSS: Mask-Enhanced Temporal Contrastive Learning for Smooth State Evolution in Visual Reinforcement Learning (https://arxiv.org/abs/2409.02714)
Comments:
          WACV 2025

- **What's New**: MOOSS는 시각적 강화 학습(Visual Reinforcement Learning)에서 픽셀 기반 관측으로부터 상태 진화를 명시적으로 모델링하는 새로운 프레임워크입니다. 이 프레임워크는 그래프 기반의 시공간 마스킹(Spatial-Temporal Masking)과 임시 대조 목표(Temporal Contrastive Objective)를 활용하여 상태 표현의 샘플 효율성을 개선합니다.

- **Technical Details**: MOOSS는 (1) 시공간 마스킹을 위한 픽셀 기반 관측의 그래프 구조 구축, (2) 상태의 시간적 연속성과 변화를 강조하는 다중 수준 대조 학습 메커니즘을 통합하는 자기지도(Self-Supervised) 이중 구성 전략을 제안합니다. 이 프레임워크는 기존 대조 방법의 한계를 극복하고, 더 나아가 시공간 상관관계를 학습할 수 있도록 합니다.

- **Performance Highlights**: 우리의 실험 결과, MOOSS는 DeepMind Control Suite 및 Atari 게임 등 다양한 연속 및 이산 제어 벤치마크에서 기존의 최첨단 시각적 강화 학습 방법보다 뛰어난 샘플 효율성을 보였습니다. 이러한 성과는 MOOSS가 효과적으로 상태 표현을 학습하고 정책 학습을 개선하는 데 기여함을 시사합니다.



### CLDA: Collaborative Learning for Enhanced Unsupervised Domain Adaptation (https://arxiv.org/abs/2409.02699)
- **What's New**: 본 논문은 Unsupervised Domain Adaptation (UDA)에서 Teacher-Student 프레임워크를 활용한 Knowledge Distillation (KD) 접근의 한계를 분석하고, 새로운 Collaborative Learning 방식을 도입하여 Teacher 모델의 비중이 낮은(non-salient) 파라미터를 업데이트하고 Student 모델의 성능을 향상시키는 방법을 제안합니다.

- **Technical Details**: 본 연구에서는 Teacher 모델의 비중 낮은 파라미터(Domain Shift induced Non-salient parameters, DSN) 문제를 해결하기 위해 Layer Saliency Rate (LSR)을 활용한 분석을 실시하였고, Student 모델이 Teacher 모델의 성능을 보완할 수 있다는 것을 발견했습니다. 이후, 교차 정보 교환을 통해 모델 간의 상호작용을 극대화하는 Collaborative Learning (CLDA) 방법을 제안합니다.

- **Performance Highlights**: 실험 결과, CLDA는 다양한 작업과 데이터셋에서 모두 성능 향상을 보여주었으며, GTA에서 Cityscapes로의 Semantic Segmentation에서 Teacher 모델은 +0.7% mIoU, Student 모델은 +1.4% mIoU의 성능 개선을 이루었습니다. Synthia에서 Cityscapes로 이동할 때는 Teacher 모델에서 +0.8% mIoU, Student 모델에서 +2.0% mIoU의 개선을 보였습니다.



### Rethinking HTG Evaluation: Bridging Generation and Recognition (https://arxiv.org/abs/2409.02683)
- **What's New**: 본 연구에서는 Handwriting Text Generation (HTG) 평가를 위해 맞춤형 세 가지 메트릭스 (metrics)를 도입하였으며, 이는 손글씨 이미지 생성 품질 평가에 더 효과적이라고 주장합니다. 새로운 메트릭스는 HTG_HTR, HTG_style 및 HTG_OOV로 명명되었습니다.

- **Technical Details**: HTG 시스템의 성능을 평가하기 위해 Handwriting Text Recognition (HTR) 및 Writer Identification (WI) 모델의 인식 오류/정확성을 기반으로 하는 세 가지 평가 지표를 제안합니다. HTG_HTR는 기존 HTG 방식으로 생성된 합성 손글씨 샘플로 훈련된 HTR 시스템의 성능을 평가하며, HTG_style은 실제 데이터로 훈련된 스타일 분류기의 정확도를 측정합니다. 마지막으로 HTG_OOV는 교육 데이터에 없는 단어 생성을 평가합니다.

- **Performance Highlights**: IAM 손글씨 데이터베이스(IAM handwriting database)를 사용한 포괄적인 실험을 통해, 기존 FID 등 메트릭스가 생성된 손글씨 샘플의 다양성과 실제 유용성을 적절히 정량화하지 못하는 문제를 보여주었습니다. 제안된 메트릭스는 정보가 풍부하며, HTG의 품질 평가를 위한 표준화된 평가 프로토콜의 필요성을 강조합니다.



### Improved Single Camera BEV Perception Using Multi-Camera Training (https://arxiv.org/abs/2409.02676)
Comments:
          This Paper has been accepted to the 27th IEEE International Conference on Intelligent Transportation Systems (ITSC 2024)

- **What's New**: 본 연구는 자율 주행 차량에서 중요하게 여겨지는 Bird's Eye View (BEV) 맵 예측의 성능을 개선하기 위해, 한 대의 카메라로도 충분한 성능을 얻을 수 있는 새로운 방법을 제안합니다. multi-camera 환경에서의 학습에서 단일 카메라 입력으로의 전환 시 성능 저하를 최소화하는 모델을 개발했습니다.

- **Technical Details**: 이 연구는 다중 카메라 서라운드 뷰 모델을 단일 카메라 추론용으로 최적화하는 세 가지 기능을 포함합니다. 첫 번째로, 'inverse block masking' 기법을 적용하여 입력 카메라의 비율을 점진적으로 변화시킵니다. 두 번째로, 'cyclic Learning Rate (LR) schedule'을 도입하여 데이터 분포 변화에 맞추어 모델 학습 속도를 조절합니다. 마지막으로, 'BEV feature reconstruction loss'를 도입하여 잘못 masked된 샘플에서도 올바른 BEV 특성이 복구되도록 지원합니다.

- **Performance Highlights**: 제안된 방법은 단일 카메라로 학습된 모델에 비해 mIoU가 19% 증가하고 mAP는 414% 향상되었습니다. 이는 BEV 맵의 품질 개선과 잘못된 탐지 수의 현저한 감소를 나타냅니다.



### Standing on the Shoulders of Giants: Reprogramming Visual-Language Model for General Deepfake Detection (https://arxiv.org/abs/2409.02664)
- **What's New**: 이번 연구는 Vision-Language Models (VLMs)의 제로샷(zero-shot) 장점을 활용하여 새로운 딥페이크(ddeepfake) 탐지 방법인 RepDFD를 제안합니다. 이는 잘 훈련된 VLM을 대상으로 한 일반적인 딥페이크 탐지에 효과적으로 재프로그래밍하는 접근법입니다.

- **Technical Details**: RepDFD는 입력 조작(input manipulation)을 통해 사전 훈련된 CLIP 모델을 재프로그래밍합니다. 모델의 내부 파라미터를 조정하지 않고도 얼굴 신원(facial identity)을 기반으로 한 페이퍼리(guided) 프롬프트를 생성해 모델의 최적화를 돕습니다. 이 방법은 이미지와 학습 가능한 perturbations를 통합하여 CLIP의 이미지 인코더(image encoder)에 전달합니다.

- **Performance Highlights**: RepDFD는 유명한 벤치마크에서 실험을 수행하여 교차 데이터셋(cross-dataset)과 조작(cross-manipulation) 성능을 88% 이상의 AUC(Area Under the Curve)로 향상시켰습니다. 이 방법은 훈련 가능한 파라미터의 비용을 크게 절감하면서도 효율적인 성능을 보여 미래의 실험적 적용 가능성을 제시합니다.



### PoseTalk: Text-and-Audio-based Pose Control and Motion Refinement for One-Shot Talking Head Generation (https://arxiv.org/abs/2409.02657)
Comments:
          7+5 pages, 15 figures

- **What's New**: 본 연구에서는 음성과 텍스트 프롬프트에 기반하여 자유롭게 생성된 입맞춤 동기화된 토킹 헤드 비디오를 생성할 수 있는 새로운 시스템인 PoseTalk를 제안합니다.

- **Technical Details**: PoseTalk는 두 가지 입력 소스, 즉 오디오와 텍스트 프롬프트를 활용해 헤드 포즈를 생성합니다. 특히, Pose Latent Diffusion (PLD) 모델을 통해 동작 잠재 공간에서 포즈를 예측합니다. 또한, CoarseNet과 RefineNet의 두 단계 네트워크를 통해 자연스러운 비디오 생성을 위한 정제 기반 학습 전략이 사용됩니다.

- **Performance Highlights**: PoseTalk의 포즈 예측 전략은 텍스트 전용 또는 오디오 전용 방법보다 뛰어난 포즈 다양성과 사실성을 달성하였으며, 비디오 생성 모델은 자연스러운 머리 동작의 토킹 비디오 합성에서 최신 기술을 초월하는 성능을 보였습니다.



### Skip-and-Play: Depth-Driven Pose-Preserved Image Generation for Any Objects (https://arxiv.org/abs/2409.02653)
- **What's New**: Diffusion 모델의 발전으로 인해 텍스트만으로도 고품질 이미지를 생성할 수 있게 되었습니다. 기존의 포즈 제어는 특정 객체나 포즈에 한정되어 있었으나, 해당 논문에서는 깊이 맵(depth map)을 이용한 새로운 포즈 제어 방법을 제안하였습니다.

- **Technical Details**: 기존의 카메라 파라미터나 키포인트에서 발생하는 한계를 극복하기 위해 깊이 기반의 포즈 제어를 도입하였습니다. 주목할 점은 Skip-and-Play (SnP)라는 방법을 통해 깊이 맵의 정보를 활용하면서도 생성되는 이미지의 모양(shape) 의존성을 줄이는데 성공하였습니다.

- **Performance Highlights**: 다양한 실험 결과, SnP는 기존 방법들에 비해 뛰어난 성능을 보였으며, 조건(예: 말)과 프롬프트(예: 고슴도치)가 다른 경우에도 다양한 객체와 포즈의 이미지를 생성할 수 있는 능력을 입증하였습니다.



### Learning-Based Error Detection System for Advanced Vehicle Instrument Cluster Rendering (https://arxiv.org/abs/2409.02647)
Comments:
          9 pages

- **What's New**: 자동차 산업에서 디지털 디스플레이 옵션이 확장되고 있으며, 새로운 모델이 출시될 때마다 점점 더 복잡해지고 있습니다. 이에 따라 적절한 모니터링 시스템이 필요하게 되었으며, 고전적인 오류 검출 방법으로는 한계가 있습니다. 이 논문에서는 새로운 학습 기반 오류 감지 시스템을 제안합니다.

- **Technical Details**: 본 시스템은 telltales(신호등)를 예로 들어 렌더링된 콘텐츠의 정확성을 검증합니다. anomaly detection 접근 방식을 사용하여 '좋은' telltales와 '손상된' telltales를 구분하며, 이는 개별 픽셀 오류에 강한 내성을 가지고 있습니다. 또한 alpha blending 및 다른 렌더링 효과를 지원합니다.

- **Performance Highlights**: 모든 '손상된' 테스트 샘플이 정확하게 오류로 분류되었으며, 결함이 없는 시나리오에서는 어떤 허위 경고도 발생하지 않았습니다. 이는 제안된 시스템이 현대 디지털 계기판에서 다채로운 배경과 렌더링 효과에도 적응할 수 있음을 보여줍니다.



### MADiff: Motion-Aware Mamba Diffusion Models for Hand Trajectory Prediction on Egocentric Videos (https://arxiv.org/abs/2409.02638)
- **What's New**: 이 논문에서는 새로운 손 궤적 예측 방법인 MADiff를 제안합니다. 이는 확산 모델(diffusion model)을 통해 미래의 손 위치를 예측하는 방식으로, 카메라 착용자의 egomotion을 고려하여 손 움직임과 시나리오 간의 관계를 파악합니다.

- **Technical Details**: MADiff는 고유한 모션 인식 기법인 Mamba를 활용하여 잠재 공간(latent space)에서의 노이즈 제거 작업을 수행합니다. 이를 통해 우리는 카메라의 egomotion을 사용하여 모션 이끌림 선택적 스캔(motion-driven selective scan, MDSS)을 달성합니다. 이 방법은 2D 비디오의 시맨틱(semanics) 정보와 손 궤적의 역사적인 데이터를 결합하여 예측을 수행합니다.

- **Performance Highlights**: MADiff는 기존의 최첨단 모델들과 비교하여 합리적인 손 궤적을 예측하고, 실시간 성능을 달성하는 데 성공했습니다. 다섯 개의 공개 데이터셋에서 포괄적인 실험을 통해 그 유효성을 입증하였습니다.



### Loopy: Taming Audio-Driven Portrait Avatar with Long-Term Motion Dependency (https://arxiv.org/abs/2409.02634)
- **What's New**: 이번 논문에서는 오디오 조건 기반의 비디오 생성 모델인 Loopy를 제안합니다. 기존 방법들이 사용할 수 있었던 보조적인 공간 신호가 아니라, 오디오 만으로 자연스러운 운동 패턴을 학습하는 방식으로, 이는 무작위성을 증가시키는 대신 더욱 생생하고 안정적인 비디오 결과물을 생성할 수 있게 돕습니다.

- **Technical Details**: Loopy 모델은 클립 간 및 클립 내 시간 모듈을 설계하여, 데이터로부터 오랜 기간의 운동 정보를 활용합니다. 각기 다른 시간 모듈을 통해 운동 프레임을 모델링하고, 음성과 얼굴 운동 관련 특징을 변환하여, 모델에 조건으로 삽입하는 오디오에서 운동 잠재 변수(audio-to-latents) 모듈을 도입하였습니다. 이러한 방식은 25 fps에서 약 5초에 해당하는 100프레임 이상의 수용 분야를 확장하여 자연스러운 모션 패턴을 학습할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, Loopy는 기존의 오디오 기반 인물 확산 모델보다 자연스러운 움직임과 더욱 사실적인 품질의 결과를 보여주었습니다. 다양한 시나리오에 걸쳐, Loopy 모델은 기존 기술에 비해 더 생동감 있고 안정적인 합성 결과를 달성하는 것으로 나타났습니다.



### AdvSecureNet: A Python Toolkit for Adversarial Machine Learning (https://arxiv.org/abs/2409.02629)
- **What's New**: AdvSecureNet는 다중 GPU 지원, CLI 및 API 인터페이스, 외부 YAML 구성 파일을 네이티브로 지원하는 최초의 Adversarial Machine Learning 툴킷입니다. 이 툴킷은 다양한 공격 및 방어 메커니즘, 평가 메트릭스를 포함하여 머신러닝 연구의 유연성과 재현성을 향상시킵니다.

- **Technical Details**: AdvSecureNet는 PyTorch에 기반한 모듈식 툴킷으로, 다중 GPU 환경에서 작동하도록 최적화되어 있습니다. 이 툴킷은 Gradient-based, Decision-based, White-box, Black-box 등 다양한 공격 방법과 Adversarial Training, Ensemble Adversarial Training 등의 방어 메커니즘을 지원합니다. 또한, YAML 구성 파일을 통해 실험 매개변수를 쉽게 조정할 수 있습니다.

- **Performance Highlights**: AdvSecureNet는 다중 GPU 환경에서 다른 툴킷 대비 빠른 실행 시간을 기록합니다. 예를 들어, CIFAR-10에서의 Adversarial Training이 단일 GPU에서 5.07분 걸릴 때 7개의 GPU를 사용할 경우 2.77분으로 단축됩니다. ImageNet에서는 단일 GPU에서 240분 걸리던 것이 7개의 GPU에서 30분으로 줄어들어 8배의 속도 향상을 보여줍니다.



### GoT-CQA: Graph-of-Thought Guided Compositional Reasoning for Chart Question Answering (https://arxiv.org/abs/2409.02611)
- **What's New**: 이번 논문은 Chart Question Answering (CQA)에서의 복잡한 추론을 해결하기 위해 새로운 Graph-of-Thought (GoT) 기반의 구조적 추론 모델인 GoT-CQA를 제안합니다. 이 모델은 시각적 차트 내용을 기반으로 질문에 답변하는 어려운 문제를 해결하는 데 중점을 두고 있습니다.

- **Technical Details**: GoT-CQA는 세 가지 모듈로 구성되어 있습니다: 1) Chart & Question Parsing 모듈, 2) Compositional Reasoning 모듈, 3) Answering 모듈입니다. 이 과정에서는 차트에서의 시각적 특성 시퀀스를 추출하고, 질문에 따른 GoT를 생성한 후, GoT에 의해 유도된 데이터 흐름에 따라 복잡한 지역화, 수치 및 논리적 추론을 수행합니다.

- **Performance Highlights**: GoT-CQA는 ChartQA 및 PlotQA-D 데이터셋에서의 광범위한 실험을 통해 우수한 성능을 입증하였으며, 특히 복잡한 인간 작성 질문과 추론 질문에서 뛰어난 성능을 보였습니다. 이는 최신 기법들과 비교했을 때, 더욱 향상된 결과를 보여주었습니다.



### A Medical Multimodal Large Language Model for Pediatric Pneumonia (https://arxiv.org/abs/2409.02608)
Comments:
          18 pages, 10 figures

- **What's New**: 이 연구에서는 소아 폐렴(Pediatric Pneumonia) 진단 및 치료에 있어 주요 도전 과제를 해결하기 위한 Medical Multimodal Large Language Model(P2Med-MLLM)을 제안했습니다. 이를 통해 다양한 임상 과제를 처리할 수 있으며, 대규모 및 고품질 멀티모달 데이터셋(P2Med-MD)을 통해 실제 임상 정보를 기반으로 훈련되었습니다.

- **Technical Details**: P2Med-MLLM은 순수 텍스트 및 이미지-텍스트 데이터 처리 능력을 갖춘 대형 언어 모델(LLM)로, 2D 흉부 X선 이미지와 3D 흉부 CT 이미지, 해당 방사선 보고서, 외래 및 입원 기록을 포함하는 광범위한 데이터셋을 사용하여 훈련되었습니다. 또한 세 단계의 훈련 전략을 활용하여 의료 지식을 이해하고 다양한 임상 작업을 수행하도록 설계되었습니다.

- **Performance Highlights**: P2Med-MBench라는 벤치마크를 사용하여 P2Med-MLLM의 성능이 평가되었으며, 총 642개의 샘플을 통한 자동 점수 매기기 결과로 기존 오픈 소스 모델보다 뛰어난 성능이 입증되었습니다. 이 모델은 최전선 의사들이 신속하고 정밀한 질병 진단 및 치료 계획 수립을 지원하여 심각한 증상으로 인한 사망률을 감소시키는 데 중요한 역할을 할 것입니다.



### SurgTrack: CAD-Free 3D Tracking of Real-world Surgical Instruments (https://arxiv.org/abs/2409.02598)
- **What's New**: 이 논문에서 제안하는 SurgTrack는 CAD 없이도 3D 수술 기구 추적을 수행할 수 있는 새로운 방법론입니다. 이는 수술 기구의 3D 표현을 모델링하기 위해 Instrument Signed Distance Field (SDF)를 사용하며, 비침습적이고 유연한 3D 천공 기능을 제공합니다.

- **Technical Details**: SurgTrack는 두 가지 단계로 구성되어 있습니다. 첫 번째 단계에서는 Instrument SDF를 이용하여 수술 기구의 3D 등록을 수행하고, 두 번째 단계에서는 자세 메모리 풀과 자세 그래프 최적화 모듈을 통해 6 자유도(DoF) 자세를 추적합니다. 이 과정에서 occlusion과 낮은 텍스처 문제를 해결하기 위해 다양한 최적화 기법을 적용합니다.

- **Performance Highlights**: SurgTrack는 88.82%의 ADD-S 성능과 12.85의 재구성 오류로, 기존 방법들에 비해 뛰어난 성능을 보이며, 실험을 통해 범용성과 확장성도 검증되었습니다.



### BMI Prediction from Handwritten English Characters Using a Convolutional Neural Network (https://arxiv.org/abs/2409.02584)
- **What's New**: 이 연구는 손글씨를 이용하여 BMI(체질량지수)를 예측하는 데 있어 심층 학습 기술과의 명확한 연결고리를 설정하지 않았던 기존의 연구 공백을 해결합니다.

- **Technical Details**: 제안된 방법은 CNN(합성곱 신경망)을 활용하여 손글씨 문자 이미지에서 개인의 BMI 값을 추정합니다. 영어 소문자 스크립트를 포함한 48명의 샘플로 구성된 데이터셋인 EHC(English Handwritten Character) 데이터셋을 사용하여 CNN 모델이 99.92%의 정확도로 성능을 보여줍니다.

- **Performance Highlights**: 제안된 CNN 모델은 AlexNet(99.69%)과 InceptionV3(99.53%)와 비교했을 때 뛰어난 성능을 보이며, 이는 BM 예측을 위한 손글씨 모델로서는 처음으로 SOTA(최첨단 성능)을 초월한 결과입니다.



### Object Gaussian for Monocular 6D Pose Estimation from Sparse Views (https://arxiv.org/abs/2409.02581)
- **What's New**: SGPose는 점보다 적은 10개 뷰에서 물체 자세 추정을 가능하게 해주는 새로운 프레임워크입니다. 이 방법은 기존의 3D Gaussian Splatting 방식에서 벗어나 구조적 기본 모델을 사용하지 않고 임의의 초기화에서 시작하여 2D-3D 대응관계를 회귀합니다.

- **Technical Details**: SGPose는 랜덤 큐보이드 초기화에서 시작하며, 구조 기반의 3DGS 방식 대신 기하학적으로 일관성 있는 깊이 슈퍼비전을 통해 이미지와 재구성된 모델 간의 조밀한 2D-3D 대응관계를 생성합니다. 이를 통해 들어오는 이미지를 기반으로 객체 중심 3D 표현을 효율적으로 생성합니다.

- **Performance Highlights**: 실험 결과, SGPose는 Occlusion LM-O 데이터세트에서 기존 방법들보다 우수한 성능을 발휘하며, 실제 응용에의 가능성을 높입니다. CAD 모델에 의존하지 않고도 효율적이고 강력한 단안 자세 추정을 가능하게 합니다.



### Solving Video Inverse Problems Using Image Diffusion Models (https://arxiv.org/abs/2409.02574)
Comments:
          22 pages, 16 figures

- **What's New**: 최근 디퓨전 모델에 기반한 역문제 해결 방법(DIFFUSION INVERSE SOLVERS, DIS)이 등장하여 이미지 초해상도, 디블러링, 인페인팅 등 다양한 역문제를 해결하는 데에 최첨단 접근법으로 자리잡고 있습니다. 기존의 영상 역문제에 대한 연구는 미비했던 반면, 본 논문에서는 이미지를 기반으로 한 새로운 영상 역문제 해결 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 최근 성공적으로 수행된 분해된 디퓨전 샘플러(Decomposed Diffusion Sampler, DDS)에서 영감을 받아 영상의 시간 차원을 이미지 디퓨전 모델의 배치 차원으로 다루며, 각 이미지 디퓨전 모델에서 파생된 노이즈 제거된 시공간 배치 내에서 시공간 최적화 문제를 해결합니다. 또한 배치 간 일관성을 보장하기 위해 배치 일관성 샘플링 전략을 도입하여 각 이미지 디퓨전 모델의 확률적 노이즈 구성 요소를 동기화합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 영상 역문제에서 다양한 시공간 손상을 효과적으로 처리하며, 최첨단 재구성 성능을 달성함을 확인하였습니다.



### Evaluation Study on SAM 2 for Class-agnostic Instance-level Segmentation (https://arxiv.org/abs/2409.02567)
- **What's New**: 최근에 발매된 Segment Anything Model 2 (SAM2)는 자연 장면에서 제로-샷 (zero-shot) 분할 성능을 향상시켜 연구자들의 기대를 높이고 있습니다. SAM2는 Salient Instance Segmentation (SIS), Camouflaged Instance Segmentation (CIS), Shadow Instance Detection (SID)와 같은 다양한 시나리오에서 클래스 무관 인스턴스 레벨 분할 성능을 평가했습니다.

- **Technical Details**: SAM2는 사용자 맞춤 프롬프트를 수용할 수 있는 강력한 적응성을 가지고 있으며, 다양한 테스트와 벤치마크를 통해 성능을 검증했습니다. 이 연구는 4가지 작업에서 SAM2의 성능을 SAM과 비교하였고, 주로 GT-Bbox 모드와 자동 모드에서 평가했습니다. 각 작업에서 사용된 데이터셋과 평가 지표가 상세히 설명되었습니다.

- **Performance Highlights**: SAM2는 박스 프롬프트를 사용할 경우 CIS 및 SIS 작업에서 특정 작업에 맞춘 기법들보다 뛰어난 성능을 보였습니다. 하지만, 비박스 프롬프트를 사용할 경우 성능이 크게 저하되었고, DIS 작업에서는 세부 구조의 분할이 필요한 경우에도 성능이 다소 떨어지는 결과를 보였습니다.



### How Do You Perceive My Face? Recognizing Facial Expressions in Multi-Modal Context by Modeling Mental Representations (https://arxiv.org/abs/2409.02566)
Comments:
          GCPR 2024

- **What's New**: 이번 논문에서는 인간의 얼굴 표정 인식을 이해하기 위해 기존의 단순 분류 작업을 넘어선 새로운 방법을 제안합니다. 이 모델은 맥락에서 얼굴을 바라볼 때 인간이 지각하는 정신적 표현을 합성하는 기능을 갖추고 있습니다.

- **Technical Details**: 모델은 VAE-GAN (Variational Autoencoder-Generative Adversarial Network) 아키텍처를 사용하여 내용(content)과 맥락(context)의 두 개의 독립적인 표현을 학습합니다. 새롭게 제안된 주의(attention) 메커니즘을 통해 맥락 의존적인 특징 적응을 수행합니다.

- **Performance Highlights**: RAVDESS 데이터셋에서 81.01%의 분류 정확도를 달성하였고, MEAD 데이터셋에서는 79.34%의 정확도를 기록하였습니다. 또한, 해당 모델은 인간의 정신적 표현에 대한 근사치를 효과적으로 생성하는 결과를 보여주었습니다.



### Interacting Multiple Model-based Joint Homography Matrix and Multiple Object State Estimation (https://arxiv.org/abs/2409.02562)
Comments:
          Preprint submitted to Information Fusion

- **What's New**: 새로운 MOT(multiple object tracking) 알고리즘인 IMM Joint Homography State Estimation (IMM-JHSE)가 제안되었습니다. 이 방법은 카메라 프로젝션 매트릭스를 트랙 상태 벡터의 일부로 공동 모델링하여, 카메라 모션 보상 기법이 예측된 트랙 위치 상태에 미치는 영향을 제거했습니다.

- **Technical Details**: IMM-JHSE는 정적 및 동적 카메라 모션 모델을 IMM 필터를 통해 결합합니다. 간단한 바운딩 박스 모션 모델을 사용하여 이미지 평면 정보를 포함한 바운딩 박스 위치를 예측합니다. 또한, 동적 측정 노이즈 추정 기술을 활용하여 특정 트랙의 측정된 바운딩 박스와 각 카메라 모션 모델과 관련된 노이즈를 추정합니다.

- **Performance Highlights**: IMM-JHSE는 DanceTrack 및 KITTI-car 데이터셋에서 각각 HOTA를 2.64 및 2.11 향상시키며, MOT17, MOT20 및 KITTI-pedestrian 데이터셋에서 경쟁력 있는 성과를 보여줍니다.



### Low-Resolution Object Recognition with Cross-Resolution Relational Contrastive Distillation (https://arxiv.org/abs/2409.02555)
Comments:
          This paper is accepted by IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)

- **What's New**: 본 연구에서는 저해상도(低解像度) 객체 인식 문제를 해결하기 위해, 고해상도 모델로부터 지식을 효과적으로 이전하기 위한 새로운 접근 방식인 cross-resolution relational contrastive distillation을 제안합니다.

- **Technical Details**: 이 방법은 로우해상도(low-resolution) 객체의 특징을 잘 학습하도록 구성된 학생 모델(student model)이 고해상도(高解像度) 모델의 행동을 모방하도록 돕습니다. 또한, contrastive relational distillation loss를 통해 다양한 관계 구조에서의 유사성을 보존하여 필요한 지식을 추출합니다.

- **Performance Highlights**: 저해상도 객체 분류 및 얼굴 인식 실험을 통해, 제안한 방법이 뛰어난 성능과 적응성을 보여주었습니다.



### Real-Time Dynamic Scale-Aware Fusion Detection Network: Take Road Damage Detection as an examp (https://arxiv.org/abs/2409.02546)
- **What's New**: 본 연구에서는 Unmanned Aerial Vehicle (UAV)을 이용한 도로 손상 탐지(road damage detection, RDD)를 위해 Dynamic Scale-Aware Fusion Detection Model (RT-DSAFDet)을 제안하였습니다. 이 모델은 다양한 크기와 형태의 도로 손상에 적응하고 배경 간섭을 자동으로 제거하는 특성을 지니고 있습니다.

- **Technical Details**: RT-DSAFDet 모델은 세 가지 주요 모듈로 구성되어 있습니다: 1) Flexible Attention (FA) 모듈은 도로 손상의 형태와 배경 변화에 유연하게 적응하는 기능을 제공합니다. 2) Dynamic Scale-Aware Fusion (DSAF) 모듈은 다양한 크기의 손상 특징을 효율적으로 융합합니다. 3) Spatial Downsampling (SD) 모듈은 모델의 파라미터 수와 계산 복잡도를 줄이면서 실시간 탐지 요구에 적합하도록 설계되었습니다. 이 모델은 UAV-PDD2023와 MS COCO2017 데이터셋에서 실험하여 우수한 성능을 입증했습니다.

- **Performance Highlights**: 모델 RT-DSAFDet는 UAV-PDD2023 데이터셋에서 mAP50이 54.2%로 YOLOv10-m보다 11.1% 향상되었으며, 파라미터 수는 1.8M, FLOPs는 4.6G로 각각 88% 및 93% 감소하였습니다. MS COCO2017 데이터셋에서도 mAP50-95에서 YOLOv9-t와 동등한 성능을 보이며, mAP50이 0.5% 더 높고 파라미터 수는 10%, FLOPs는 40% 적었습니다.



### UniTT-Stereo: Unified Training of Transformer for Enhanced Stereo Matching (https://arxiv.org/abs/2409.02545)
- **What's New**: 본 논문에서는 변환기(Transformer) 기반의 스테레오 깊이 추정을 위한 새로운 방법인 UniTT-Stereo를 제안합니다. 이 방식은 자가 지도 학습(self-supervised learning)과 지도 학습(supervised learning)을 통합하여 스테레오 매칭(stereo matching)의 잠재력을 극대화하는 것을 목표로 하고 있습니다.

- **Technical Details**: UniTT-Stereo는 입력 이미지에서 마스킹된 부분의 피처(features)를 재구성하고 동시에 다른 이미지에서 해당 지점을 예측하는 방식을 탐구합니다. 특히, 모델의 훈련에 제한된 데이터를 효과적으로 활용하기 위해 지역 유도 편향(locality inductive bias)의 관점에서 접근합니다. 또한, 스테레오 맞춤 손실(stereo-tailored losses)을 활용하여 마스킹 비율(masking ratio)을 조절하며 모델이 다양한 정보로부터 학습할 수 있도록 합니다.

- **Performance Highlights**: ETH3D, KITTI 2012 및 KITTI 2015 데이터셋에서 state-of-the-art 결과를 달성하여 UniTT-Stereo의 효율성을 입증하였습니다. 본 모델은 추가 매개변수 없이도 손실 함수의 조합을 통해 향상된 성능을 보여줍니다.



### StyleTokenizer: Defining Image Style by a Single Instance for Controlling Diffusion Models (https://arxiv.org/abs/2409.02543)
Comments:
          Accepted by ECCV2024

- **What's New**: 본 논문에서는 스타일 제어를 위한 새로운 방법인 StyleTokenizer를 소개합니다. 이 방식은 단일 참조 이미지를 사용하여 스타일 표현을 효과적으로 통합하고, 텍스트 표현의 제어 효과를 최소화하여 이미지 생성의 스타일 제어를 지원합니다.

- **Technical Details**: StyleTokenizer는 스타일 표현과 텍스트 표현을 정렬하기 위해 스타일 토크나이저를 사용하여, 두 과정의 상호작용을 줄입니다. 또한 Style30k이라는 잘 라벨링된 스타일 데이터셋을 수집하여 스타일 특성 추출기를 훈련시킵니다. 이 방법은 셀 수 없이 많은 스타일 카테고리를 고려하며, 텍스트 프롬프트와의 일관성을 유지합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 단일 참조 이미지를 통해 스타일 특성을 완벽하게 이해하며, 목표 이미지 스타일과 텍스트 프롬프트에 일치하는 매력적인 이미지를 생성하는 데 성공합니다.



### SG-MIM: Structured Knowledge Guided Efficient Pre-training for Dense Prediction (https://arxiv.org/abs/2409.02513)
- **What's New**: 본 논문에서는 SG-MIM(Structured knowledge Guided Masked Image Modeling)이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 기존의 Masked Image Modeling(MIM) 기법의 한계를 극복하고, 기초 예측(dense prediction) 과제에서 향상된 성능을 제공합니다.

- **Technical Details**: SG-MIM은 이미지와 함께 구조화된 지식을 활용하여 기초 예측 성능을 향상시키기 위해 고안된 경량 관계 안내(relational guidance) 프레임워크를 사용합니다. 이 프레임워크는 피처 레벨(feature level)에서 구조적 지식을 개별적으로 안내하며, 전통적인 멀티모달(pre-training) 방식의 픽셀 레벨에서 단순히 결합하는 방식을 지양합니다. 또한, 선택적 마스킹 전략을 채택하여 일반 표현 학습과 구조 지식-specific 학습 간의 시너지를 극대화합니다.

- **Performance Highlights**: SG-MIM은 KITTI, NYU-v2 및 ADE20k 데이터셋에서 실험 평가를 통해 모노큘러(depth estimation) 및 의미론적(segmentation) 실험에서 우수한 성능을 입증하였습니다. 예를 들어, KITTI 검증 데이터셋에서 RMSE 2.04, NYU-v2에서 δ1 지표 0.91, ADE20K에서 mIoU 47.59를 달성하며, 기존 MIM 모델들과 비교하여 뛰어난 성능을 보였습니다.



### TLD: A Vehicle Tail Light signal Dataset and Benchmark (https://arxiv.org/abs/2409.02508)
- **What's New**: 본 논문에서는 TLD라는 새로운 대규모 차량 후미등 데이터셋을 소개합니다. 이 데이터셋은 주행 중 브레이크 라이트와 방향 지시등을 별도로 주석 처리한 최초의 데이터셋입니다.

- **Technical Details**: TLD 데이터셋은 총 152,690개의 주석 이미지 프레임과 1.5백만 개의 비주석 이미지를 포함합니다. 두 단계 차량 조명 탐지 모델을 개발하였는데, 여기에는 차량 탐지기(YOLOv10과 DeepSORT)와 후미등 분류기가 포함됩니다. 이 모델은 정밀한 후미등 상태 탐지를 가능하게 합니다.

- **Performance Highlights**: 제안된 방법은 TLD 데이터셋에서 탁월한 성능을 보여주며, 다양한 환경(주간/야간, 기상 조건)에서의 안정성을 입증하였습니다. 이 연구는 자율주행 시스템의 주행 의도 인식을 향상시키는데 중요한 기여를 할 것입니다.



### Plane2Depth: Hierarchical Adaptive Plane Guidance for Monocular Depth Estimation (https://arxiv.org/abs/2409.02494)
Comments:
          14 pages, 12 figures, 8 tables

- **What's New**: 본 논문은 Plane2Depth라는 새로운 방법을 제안하여 모노큘러 깊이 추정(Monocular Depth Estimation)의 성능을 향상시킵니다. 이 방법은 장면의 평면 정보(plane information)를 효과적으로 활용하여 깊이 예측을 개선하는 계층적(Hierarchical) 구조를 가지고 있습니다.

- **Technical Details**: Plane2Depth는 두 가지 주요 구성 요소로 이루어져 있습니다: 평면 유도 깊이 생성기(Plane Guided Depth Generator, PGDG)와 적응형 평면 쿼리 집계(Adaptive Plane Query Aggregation, APGA) 모듈입니다. PGDG에서는 평면 쿼리를 프로토타입으로 사용하여 각 픽셀에 대해 평면 계수를 예측하고, APGA에서는 새로운 특징 상호작용 방식을 도입하여 다중 스케일의 평면 특징을 효과적으로 집계합니다.

- **Performance Highlights**: 실험 결과, Plane2Depth 방법이 NYU-Depth-v2 데이터셋에서 최첨단(SOTA) 방법들을 초월한 성능을 보였으며, KITTI 데이터셋에서도 경쟁력 있는 성능을 발휘하였습니다. 또한, SUN RGB-D 데이터셋에 대한 제로샷 테스트에서도 효과적인 일반화 능력을 나타냈습니다.



### Reliable Deep Diffusion Tensor Estimation: Rethinking the Power of Data-Driven Optimization Routin (https://arxiv.org/abs/2409.02492)
- **What's New**: 본 논문에서는 DTI(확산 텐서 이미징) 분야의 기존 문제를 해결하기 위해 DoDTI라는 새로운 데이터 기반 최적화 방법을 제안합니다. 이 방법은 일반적인 모델 기반 방식에서 발생하는 노이즈에 대한 민감성을 극복하는 데 초점을 맞추고 있습니다.

- **Technical Details**: DoDTI는 가중 선형 최소 제곱 적합(Weighted Linear Least Squares Fitting) 알고리즘과 노이즈 제거나 정규화를 위한 딥 러닝 기반의 덴오이저(Denoiser) 기법을 결합합니다. 이 기법은 다양한 획득 환경에서 DW(확산 가중치) 이미지를 확산 텐서 필드(Diffusion Tensor Field)로 맞추고, 기존 DW 이미지 대신 확산 텐서 필드를 정규화합니다. 최적화 문제는 교대 방향 다중 배수 방법(Alternating Direction Method of Multipliers)을 이용하여 해결되고, 이것은 딥 뉴럴 네트워크(DNN)로 구체화됩니다. 마지막으로, 네트워크 파라미터는 데이터 기반 전략을 통해 학습됩니다.

- **Performance Highlights**: 제안된 DoDTI 방법은 내부 시뮬레이션 데이터셋과 외부에서 수집된 생체 데이터셋을 활용한 종합적인 검증 실험을 통해 평가되었습니다. qualitative 및 quantitative 분석 결과, 이 방법은 DTI 파라미터 추정에서 최첨단 성능을 달성하며, 뛰어난 일반화 능력, 정확도 및 효율성을 보여주어 DTI 분야의 널리 응용 가능성을 높였습니다.



### TP-GMOT: Tracking Generic Multiple Object by Textual Prompt with Motion-Appearance Cost (MAC) SOR (https://arxiv.org/abs/2409.02490)
- **What's New**: 이번 논문에서는 Multi-Object Tracking (MOT)과 Generic Multiple Object Tracking (GMOT)의 한계를 극복하고, \\textbf{Refer-GMOT dataset}을 소개하며, 새로운 text prompt 기반의 개방형 어휘 GMOT 프레임워크인 \\textbf{TP-GMOT}를 제안합니다. TP-GMOT는 이전에 훈련된 예시 없이 새로운 물체 카테고리를 추적할 수 있는 기능을 제공합니다.

- **Technical Details**: TP-GMOT 프레임워크는 (i) \\textbf{TP-OD}라는 새로운 텍스트 프롬프트 기반 객체 감지 방법을 포함하여, 특성 기반의 보지 못한 물체를 정확하게 감지하고, (ii) Motion-Appearance Cost SORT (MAC-SORT)라는 새로운 객체 연관 접근법으로, 움직임 및 외관 기반의 매칭 전략을 융합하여 물체를 추적합니다. 이 프레임워크는 Refer-GMOT 데이터셋에서 벤치마킹되며, DanceTrack과 MOT20 데이터셋에 대한 실험도 수행합니다.

- **Performance Highlights**: 실험 결과, TP-GMOT 프레임워크와 MAC-SORT 트래커는 다양한 데이터셋에서 효과성과 일반화 가능성을 입증했습니다. 이 프레임워크는 미리 정의된 카테고리에 대한 훈련 데이터가 필요 없으며, 다양한 물체 추적 시나리오에 적합합니다.



### Boosting Generalizability towards Zero-Shot Cross-Dataset Single-Image Indoor Depth by Meta-Initialization (https://arxiv.org/abs/2409.02486)
Comments:
          IROS 2024. The version supersedes 2305.07269. arXiv admin note: text overlap with arXiv:2305.07269

- **What's New**: 이 논문은 실내 로봇의 깊이 추정 및 메타 학습을 융합하여 한 가지 단일 이미지 깊이 예측 문제에 대한 더 높은 일반화 성능을 달성하는 새로운 접근 방법을 제안합니다. 이를 위해 저자는 제로샷 크로스 데이터셋 추론(zero-shot cross-dataset inference)에 대한 고급 일반화 기술을 도입합니다.

- **Technical Details**: 본 연구는 메타 학습(metalearning)을 활용하여 연속적인 깊이 값을 가지는 복잡한 실내 환경에서의 깊이 추정 문제를 해결하기 위해 각 RGB-D 미니 배치를 작업으로 간주하는 세분화된(task) 작업을 제안합니다. 연구에 따르면, 메타-초기화(meta-initialization)를 통해 더 높은 일반성을 입증하고, 기존 방법 대비 성능 개선을 확인했습니다.

- **Performance Highlights**: 제안된 방법은 제한된 데이터에서 최대 27.8%의 RMSE 개선을 보여주었으며, 메타 학습 초기화 후의 정제(fine-tuning) 과정에서 기본적인 접근 방식보다 일관되게 더 우수한 성능을 나타났습니다. 이 연구는 실내 로봇 및 AR/VR 등의 실제 응용 분야에서의 강력하고 일반화된 깊이 추정을 가능하게 합니다.



### TASAR: Transferable Attack on Skeletal Action Recognition (https://arxiv.org/abs/2409.02483)
Comments:
          arXiv admin note: text overlap with arXiv:2407.08572

- **What's New**: 이번 연구에서는 Skeletal Action Recognition(S-HAR)에 대한 최초의 Transfer-based Attack 방법인 TASAR을 제안합니다. TASAR은 손실 표면의 매끄러움을 개선함으로써 S-HAR의 적대적 전이 가능성을 증가시키고, 이에 따른 공격 방법론의 한계를 극복합니다.

- **Technical Details**: TASAR은 포스트 트레인(Double Bayesian Optimization) 기법을 통해 미리 훈련된 대리 모델의 매끄러운 모델 후행을 탐색합니다. 이 과정에서 각 프레임을 독립적으로 처리하는 기존의 방법과 달리, 시간적 연속성을 고려하여 모션 동역학을 Bayesian 공격 그래디언트에 통합합니다.

- **Performance Highlights**: RobustBenchHAR로 명명된 새로운 대규모 S-HAR 벤치마크를 구축하여, 7개의 S-HAR 모델, 10개의 공격 방법 및 3개의 데이터 세트를 포함하고, TASAR의 우수성과 일반화 가능성을 입증했습니다.



### Volumetric Surfaces: Representing Fuzzy Geometries with Multiple Meshes (https://arxiv.org/abs/2409.02482)
- **What's New**: 본 논문에서는 실시간 뷰 합성(real-time view synthesis)을 위한 새로운 표현 방식을 제안합니다. 이는 표면 배치 방법이 가지는 한계를 극복하여 퍼지한 객체(fuzzy objects)를 효과적으로 표현할 수 있습니다. 새로운 방법은 샘플링 위치의 수를 작고 제한적으로 유지하고, 레스터화(rasterization)를 통해 샘플링 위치를 효율적으로 찾으며, 렌더링 과정에서 정렬(sorting)을 필요로 하지 않는 특징이 있습니다.

- **Technical Details**: 객체는 반투명(multi-layer meshes) 다층 메시로 표현되며, 외부에서 내부로 순서대로 렌더링됩니다. 각 메시 레이어는 최적 간격으로 훈련된 SDF 셸로 모델링됩니다. 이 방법은 기존 방법보다 작은 샘플링 포인트 수(3~9)를 사용하여 더 빠른 렌더링을 가능하게 합니다. 최종적으로 각 레이어는 시각 방향에 따라 투명도가 조정되어, 메쉬에 치명적이지 않은 메모리 용량으로 고품질 텍스쳐를 제공합니다.

- **Performance Highlights**: 저사양 그래픽 하드웨어 및 모바일 기기에서 발생 가능했던 기존 방법보다 높은 프레임 속도(frame rate)를 달성하며, 퍼지한 기하학적 형태를 효과적으로 표현하는 성능을 보입니다. 이 방법은 기존의 볼륨 기반(volume-based) 및 스플래팅 기반(splatting-based) 방법보다 개인용 기기에서 더욱 우수한 렌더링 결과를 보여줍니다.



### Detecting Korean Food Using Image using Hierarchical Mod (https://arxiv.org/abs/2409.02448)
- **What's New**: 이번 연구는 한국 음식의 이미지 인식을 위한 새로운 시스템을 제안하여 특히 식이 요건이 있는 외국인들에게 유용하도록 개발되었습니다. 사용자는 요리의 사진을 업로드하기만 하면 어떤 음식을 먹고 있는지 확인할 수 있습니다. 이미지 처리 기술과 머신 러닝(Machine Learning)을 결합하여 이러한 솔루션이 탄생하게 되었습니다.

- **Technical Details**: 이 시스템은 YOLOv8 모델을 기반으로 하며, 층화된 학습 방식을 통해 데이터셋의 클래스 불균형 문제를 해결하고 다양한 한국 음식을 효과적으로 분류합니다. 연구 과정에서 Convolution Neural Network (CNN)를 활용한 다단계 전이 학습(Multi-stage Transfer Learning)과 시각적으로 관련된 음식 아이템의 클러스터링 기법이 적용되었습니다. 또한, 사진의 품질을 높이기 위해 데이터 증대(Data Augmentation) 기술이 사용되었습니다.

- **Performance Highlights**: 이 연구에서는 150개 클래스의 한국 음식을 분류할 수 있는 모델을 개발하였으며, 이전 연구들보다 높은 정확도와 신속한 인식 시간을 달성하였습니다. 개발된 시스템은 사용자 친화적인 인터페이스를 제공하여, 사용자가 자신이 소모한 음식을 쉽게 기록하고 관리할 수 있도록 돕습니다.



### Non-target Divergence Hypothesis: Toward Understanding Domain Gaps in Cross-Modal Knowledge Distillation (https://arxiv.org/abs/2409.02438)
- **What's New**: 이 논문은 cross-modal knowledge distillation(KD)의 도메인 격차(domain gap)가 성능에 미치는 영향을 심층적으로 분석하고 평가합니다. 특히 Non-Target Divergence Hypothesis (NTDH)를 제안하여 도메인 간의 격차가 cross-modal KD에서 비대상(non-target) 클래스의 분포 차이에 주된 영향을 미친다는 것을 입증합니다.

- **Technical Details**: 이 논문에서 제안하는 NTDH는 cross-modal KD의 유효성을 결정짓는 중요한 요인으로 비대상 클래스의 분포 편차(divergence) 차이를 설명합니다. Vapnik-Chervonenkis(VC) 이론에 기반하여 cross-modal KD의 근사 오차의 상한과 하한을 도출하였으며, 이로써 NTDH의 이론적 타당성을 검토합니다. 또한, 실험을 통해 다섯 개의 cross-modal 데이터셋에서 NTDH의 유효성을 입증합니다.

- **Performance Highlights**: 실험 결과, 비대상 클래스 간의 분포 차이가 작을수록 cross-modal KD의 성능이 향상됨을 보여주었습니다. NTDH를 통해 cross-modal KD의 성능 향상을 위한 새로운 통찰력을 제공하며, 제안된 가중치 조절 방법과 마스킹 방법을 통한 추가적인 검증을 수행하였습니다.



### Training-free Color-Style Disentanglement for Constrained Text-to-Image Synthesis (https://arxiv.org/abs/2409.02429)
Comments:
          16 pages, 17 figures

- **What's New**: 본 연구에서는 사용자가 제공한 참조 이미지의 색상 및 스타일 속성을 독립적으로 제어하는 방법을 제시합니다. 기존의 모든 방법들과는 달리, 본 연구는 훈련이 필요 없는 테스트 시간에만 수행되는 방식으로 텍스트-이미지 확산 모델을 조정할 수 있는 첫 번째 방법을 발표합니다.

- **Technical Details**: 본 연구는 두 가지 주요 혁신을 포함합니다: 첫째, 참조 이미지의 공분산(covariance) 행렬을 따르는 방식으로 현재 생성 결과의 잠재 코드(latent codes)를 변환하는 기능적 변환(feature transformations)을 도입했습니다. 둘째, LAB 이미지 공간에서 색상(color)과 스타일(style) 간의 자연스러운 분리(disentanglement)를 활용하여 생성되는 이미지의 자기 주의(self-attention) 피쳐 맵을 참조 이미지의 L 채널에 따라 변환합니다.

- **Performance Highlights**: 제안된 방법은 동일한 참조 이미지 또는 다른 출처에서 색상 및 스타일 정보를 혼합하여 매우 유연한 이미지를 생성할 수 있습니다. 실험 결과를 통해 생성된 이미지는 참조 이미지의 색상 및 스타일 정보를 잘 반영하고 있습니다.



### MOSMOS: Multi-organ segmentation facilitated by medical report supervision (https://arxiv.org/abs/2409.02418)
Comments:
          14 pages, 7 figures

- **What's New**: 이 논문에서는 Medical repOrt Supervision (MOSMOS)라는 새로운 프레임워크를 제안하여 다중 장기 분할(multi-organ segmentation) 작업에 대한 사전 학습 및 세부 조정(fine-tuning) 방법을 개발하였습니다. 이 접근 방식은 의료 이미지와 보고서의 쌍을 활용하여 지식 전달 문제를 해결하여 임상에서의 자동 주석(annotation) 작업 개선에 초점을 맞추고 있습니다.

- **Technical Details**: MOSMOS 프레임워크는 두 가지 주요 단계로 구성됩니다: (1) 의료 이미지와 보고서 간의 글로벌 대조 학습(global contrastive learning)을 통해 특징을 정렬합니다. (2) 다중 레이블 인식을 통해 이미지 픽셀과 장기 태그 간의 의미론적 대응을 학습합니다. 이를 통해 의료 이미지의 각 픽셀에 장기 태그를 할당합니다.

- **Performance Highlights**: 다양한 질병과 모달리티의 BTCV, AMOS, MMWHS, BRATS 데이터셋을 사용하여 우리의 접근 방식을 광범위하게 평가하였고, 2D U-Net 및 3D UNETR 등 여러 네트워크 설정을 통해 일반화 능력을 검증했습니다. 실험 결과, 우리의 프레임워크가 다중 장기 분할 성능을 상당히 향상시킬 수 있음을 보여주었습니다.



### Local map Construction Methods with SD map: A Novel Survey (https://arxiv.org/abs/2409.02415)
Comments:
          14 pages, 11 figures

- **What's New**: 최근 자율주행차 분야에서 Local maps가 자율주행 기술의 주요 구성 요소로 떠오르고 있습니다. 이 논문은 SD map (Standard Definition Map)을 활용한 Local map 인식 방법의 최신 발전사항을 종합적으로 리뷰합니다. 이 리뷰는 이러한 기술적 발전이 자율주행차의 안전성과 효율성을 어떻게 향상시키는지를 설명합니다.

- **Technical Details**: Local map 인식은 차량 주변 환경을 실시간으로 모델링하고 이해하는 것을 포함합니다. SD map을 활용하면 센서의 불확실성을 완화하고 모델의 강건성을 향상시킬 수 있습니다. 특히, 다양한 센서 데이터(예: 카메라, LiDAR, GPS)를 조합하여 맵을 구성하는 방법들과 멀티모달 데이터 융합 기법에 중점을 둡니다. 핵심 구성 요소에는 이미지 피처 추출을 위한 backbone, 시점 변환을 위한 PV2BEV 모듈, 멀티모달 피쳐 융합 모듈 등이 포함됩니다.

- **Performance Highlights**: 연구에서 제안하는 여러 방법들은 환경의 복잡성과 동적 정보에도 불구하고 높은 정확도로 Local map 인식을 수행할 수 있게 해 줍니다. 특히, SD map과 결합된 기술들은 도로 인식 및 lane detection의 성능을 크게 향상시키고 있습니다. 이 연구는 SD map을 사용한 Local map 생성 방법의 현재의 동향과 획기적인 기법들을 조명하며, 실시간 자율주행 응용 프로그램의 실행 가능성을 탐구합니다.



### Multi-modal Situated Reasoning in 3D Scenes (https://arxiv.org/abs/2409.02389)
Comments:
          Project page: this https URL

- **What's New**: 이번 연구에서는 3D 씬(3D scenes)의 상황 이해와 논리를 위한 대규모 다중 모달 데이터셋인 Multi-modal Situated Question Answering (MSQA)를 제안합니다. 이를 통해 기존 데이터셋의 한계를 극복하고, 다양한 실제 3D 씬을 포괄하는 251K의 질문-답변 쌍을 수집하였습니다.

- **Technical Details**: MSQA는 텍스트, 이미지, 포인트 클라우드(point cloud) 등의 다중 모달 입력을 결합한 새로운 데이터 수집 파이프라인을 통해 생성되었습니다. 이 데이터셋은 9개의 질문 카테고리에 걸쳐, 상황과 질문 설명을 제공하는 다중 모달 입력 설정을 도입하여 복잡한 3D 씬 내에서 상황 인식의 유효성을 평가할 수 있도록 합니다.

- **Performance Highlights**: 종합적인 평가 결과, 기존의 비전-언어 모델들이 다중 모달 입력 처리 및 상황 모델링에서 한계를 보임을 드러냈습니다. 실험 결과 MSR3D라는 강력한 기본 모델이 제안되며, MSQA 및 MSNN에 대한 성능이 개선되었습니다. 데이터 스케일링과 교차 도메인 전이 실험을 통해 MSQA에서의 사전 학습(pre-training)이 모델 개발에 있어 효과적임을 보여주었습니다.



### Unified Framework with Consistency across Modalities for Human Activity Recognition (https://arxiv.org/abs/2409.02385)
Comments:
          Accepted to BMVC 2024

- **What's New**: 본 논문에서는 비디오 기반의 인간 활동 인식을 위한 포괄적인 멀티모달 프레임워크를 제안합니다. 특히, 새로운 구성 쿼리 머신인 COMPUTER($\textbf{COMP}ositional h\textbf{U}man-cen\textbf{T}ric qu\textbf{ER}y$ machine)을 도입하여 인간과 그 주변 환경 간의 상호작용을 모델링합니다.

- **Technical Details**: COMPUTER는 다양한 입력 모달리티의 뚜렷한 표현을 추출하는 데 활용될 수 있는 범용 신경망 아키텍처입니다. 또한, 예측 간의 일치를 강제하는 일관성 손실(consistency loss)을 도입하여 멀티모달 입력의 보완 정보를 활용하여 강력한 인간 움직임 인식을 지원합니다.

- **Performance Highlights**: 행동 로컬라이제이션(action localization) 및 그룹 활동 인식(group activity recognition) 작업에 대한 광범위한 실험을 통해, 제안된 방법은 최첨단(state-of-the-art) 방법들과 비교하여 우수한 성능을 보였습니다.



### GGS: Generalizable Gaussian Splatting for Lane Switching in Autonomous Driving (https://arxiv.org/abs/2409.02382)
- **What's New**: 본 논문에서는 Autonomous Driving을 위한 Generalizable Gaussian Splatting 방법인 GGS를 제안합니다. GGS는 큰 시점 변화에도 사실적인 렌더링을 수행할 수 있는 기능을 갖추고 있습니다. 기존의 일반화된 3D Gaussian Splatting 방법들은 원본 이미지 쌍과 매우 가까운 새로운 뷰를 렌더링하는 데 한계가 있었으며, 이는 Autonomous Driving 시나리오에서 효과적으로 lane switching을 처리할 수 없음을 나타냅니다.

- **Technical Details**: GGS는 virtual lane 생성 모듈을 도입하여 다양한 lane switching을 가능하게 하며, 이는 단일 lane 데이터 세트를 사용하여 높은 품질의 이미지를 생성할 수 있게 합니다. 또한, diffusion loss를 설계하여 virtual lane 이미지 생성을 감독하고, depth refinement 모듈을 통해 depth estimation을 최적화합니다. 이러한 기법을 통해 GGS는 단일 lane 데이터만으로도 다른 lane의 이미지를 생성을 학습할 수 있도록 구성되어 있습니다.

- **Performance Highlights**: GGS는 다양한 시나리오에 대한 광범위한 실험을 통해 기존 접근법과 비교하여 최첨단 성능을 입증했습니다. Laid on the premise of without LiDAR, GGS는 도로의 novel view 특징을 효과적으로 합성할 수 있습니다.



### Coral Model Generation from Single Images for Virtual Reality Applications (https://arxiv.org/abs/2409.02376)
Comments:
          In Proceedings of Explainable AI for the Arts Workshop 2024 (XAIxArts 2024) arXiv:2406.14485

- **What's New**: VR(가상 현실) 기술 발전과 함께 고품질 3D 모델의 수요가 급증하고 있는 상황에서, 본 논문은 단일 이미지로부터 고정밀 3D 산호 모델을 생성하는 딥 러닝 프레임워크를 제안합니다.

- **Technical Details**: 이 프레임워크는 Coral 데이터셋을 활용하여 기하학적 및 텍스처적 특징을 추출하고, 3D 재구성을 수행하며, 디자인 및 재료 혼합을 최적화합니다. 고급 최적화 및 폴리곤 수(control)가 Shape 정확성, Detail 유지 및 다양한 복잡도에 대한 유연한 출력을 보장합니다. 또한, Explainable AI(XAI)를 통합하여 AI 생성 모델을 인터랙티브한 '예술작품'으로 변환합니다.

- **Performance Highlights**: 생성된 모델은 기존 방식보다 Detail, Visual quality 및 Efficiency에서 우수하며, VR(가상 현실) 상호작용에서 실시간 피드백으로 산호 종 및 서식지와 같은 정보를 표시하여 사용자 경험을 개선합니다.



### Exploring Low-Dimensional Subspaces in Diffusion Models for Controllable Image Editing (https://arxiv.org/abs/2409.02374)
- **What's New**: 최근 확산 모델(difussion models)이 강력한 생성을 위한 새로운 접근법으로 자리 잡고 있으며, 이러한 모델의 의미적 공간(semantic space)에 대한 이해를 깊이 있게 다루었습니다. 이를 통해 LOw-rank COntrollable 이미지 편집(LOCO Edit) 방법을 제안하여, 추가 훈련 없이도 이미지 생성에서의 정확한 편집이 가능하다는 점이 주목할 만합니다.

- **Technical Details**: 이 연구에서는 확산 모델 내에서 학습된 후방 평균 예측기(posterior mean predictor, PMP)의 국소 선형성(local linearity)과 자코비안의 특이 벡터(singular vector)가 저차원 의미적 부분공간(low-dimensional semantic subspace)에 놓여 있다는 점을 관찰했습니다. LOCO Edit 방법은 이러한 특성을 활용하여, 훈련이나 텍스트 감독 없이 이미지 내 특정 영역을 조작할 수 있도록 합니다.

- **Performance Highlights**: LOCO Edit는 짧은 시간 내에 매우 정밀한 지역 편집이 가능하며, 다양한 확산 모델에 적용할 수 있습니다. 편집 방향은 선형, 이식 가능(transferable), 조합 가능(composable)하여 이미지의 의미적 특징을 변경하면서도 다른 영역의 일관성을 유지할 수 있는 장점이 있습니다. 대규모 실험을 통해 LOCO Edit의 효과성과 효율성이 입증되었습니다.



### Unfolding Videos Dynamics via Taylor Expansion (https://arxiv.org/abs/2409.02371)
- **What's New**: 비디오 동적 학습을 위한 새로운 접근법인 Video Time-Differentiation for Instance Discrimination (ViDiDi)를 제시합니다. 이 방법은 기존의 자기 지도 비디오 표현 학습 프레임워크에 쉽게 적용할 수 있으며, 다양한 차수의 시계열 미분을 통해 비디오의 다른 측면을 관찰하는 방식입니다.

- **Technical Details**: ViDiDi는 비디오를 개별 프레임의 순서로 취급하는 대신, 연속적이고 동적인 과정으로 간주합니다. 이 과정은 각 프레임에서 시계열 미분을 사용한 Taylor 수열 전개를 통해 표현됩니다. 0차 미분은 각 프레임 자체를 나타내고, 1차 미분은 프레임 간의 즉각적인 움직임을, 2차 미분은 그 움직임의 변화율을 포착합니다. 이러한 다양한 미분의 표현을 통해 비디오의 동적 특성을 더 효과적으로 학습할 수 있습니다.

- **Performance Highlights**: ViDiDi는 UCF101 및 Kinetics 데이터셋으로 사전 훈련된 후, 비디오 검색, 행동 인식 및 행동 탐지와 같은 표준 벤치마크에서 성능이 크게 향상되었습니다. 이 방법은 대규모 모델이나 방대한 데이터셋의 필요 없이도 상당한 성능 개선을 이끌어냈습니다.



### Pluralistic Salient Object Detection (https://arxiv.org/abs/2409.02368)
- **What's New**: 본 논문은 여러 가지 그럴듯한 saliency segmentation 결과를 생성하는 새로운 작업인 pluralistic salient object detection (PSOD)을 제안합니다. 기존의 salient object detection(SOD) 방법들이 단일 segmentation mask를 생성하는 것과는 달리, PSOD는 실제 이미지에서의 복잡성과 사용자 의도의 다양성을 반영하여 여러 개의 saliency mask 후보를 생성합니다.

- **Technical Details**: PSOD는 두 가지 새로운 SOD 데이터셋인 DUTS-MM과 DUTS-MQ를 도입합니다. DUTS-MM은 기존의 DUTS 데이터셋을 기반으로 하여 1) 경계 및 세밀한 구조에 대한 mask 품질 향상, 2) 주석 불일치 문제 완화, 3) saliency ambiguity를 가진 이미지에 대한 다수의 ground-truth masks 제공을 포함하여 새로운 주석을 제공합니다. DUTS-MQ는 사용자 선호 점수가 주석된 약 100K 이미지-마스크 쌍을 포함하고 있어, 마스크 품질을 측정하는 데 있어서 인간의 실제 선호를 학습할 수 있도록 합니다.

- **Performance Highlights**: 제안된 PSOD 프레임워크의 효과성을 입증하기 위해 폭넓은 실험을 실시하였으며, 새로운 데이터셋의 중요성을 강조합니다. 모형은 mask Candidates를 출력하고, 각 mask 후보에 대한 human preference scores를 예측하는 두 가지 작업을 하나의 모델에서 수행할 수 있는 단순하면서도 효과적인 end-to-end baseline을 제시합니다.



### What Do You See in Common? Learning Hierarchical Prototypes over Tree-of-Life to Discover Evolutionary Traits (https://arxiv.org/abs/2409.02335)
Comments:
          34 pages, 27 figures

- **What's New**: 이 연구에서는 Hierarchy aligned Commonality through Prototypical Networks (HComP-Net)라는 새로운 프레임워크를 소개하여, 생물의 진화적 특성을 발견하기 위한 계층적 프로토타입 학습 문제를 해결하고자 하였습니다.

- **Technical Details**: HComP-Net은 흔히 접할 수 있는 프로토타입 학습 방법에서의 문제점을 해결하기 위해, 과도하게 특정화(overspecific)의 프로토타입 학습을 피하고, 다른 조상 집합에 대해 부재함을 보장하는 차별적인 손실(discriminative loss)을 도입하였습니다. 또한 새로운 마스킹 모듈(masking module)을 통해 높은 수준에서 과도하게 특정화된 프로토타입을 제외할 수 있도록 설계되었습니다.

- **Performance Highlights**: HComP-Net은 190종의 새, 38종의 물고기, 30종의 나비 데이터셋에서 기존 방법들과 비교하여 정확하고 의미적으로 일관되며 보지 못한 종에 대해 일반화 가능한 프로토타입을 학습하는 데 성공했습니다. 또한 생물의 진화적 특성에 대한 새로운 가설을 생성할 수 있는 가능성을 보여주었습니다.



### Geometry-aware Feature Matching for Large-Scale Structure from Motion (https://arxiv.org/abs/2409.02310)
- **What's New**: 본 논문에서는 제안된 방법이 기존의 feature matching 방법들을 현저히 향상시킨다고 밝히고 있으며, 특히 색상 정보 외에 기하학적 단서를 통합합니다. 이를 통해 대규모 시나리오에서 적은 중첩으로 인한 간극을 메울 수 있습니다.

- **Technical Details**: 제안된 방법은 Sampson Distance를 사용하여 기하학적 검증을 최적화 문제로 공식화하며, detector-free 방법 내에서 feature matching을 유도하고 detector-based 방법에서의 희소한 대응을 앵커 포인트로 사용합니다. 이 하이브리드 전략은 대응 밀도와 정확도를 크게 개선하고 다중 뷰 일관성을 완화합니다.

- **Performance Highlights**: 제안된 방법은 여러 공개 데이터셋을 통해 평가했으며, camera pose의 추정 및 point cloud 밀도에서 기존의 최첨단 feature matching 방법을 능가했습니다. 또한, UAV에서 거리 뷰 관점으로 매핑하는 어려운 대규모 기반 설정에서도 성능을 개선했습니다.



### Biochemical Prostate Cancer Recurrence Prediction: Thinking Fast & Slow (https://arxiv.org/abs/2409.02284)
Comments:
          8 pages, 3 figures, methodology paper for LEOPRARD Challenge

- **What's New**: 이 논문에서는 전립선암의 생화학적 재발 예측을 위한 새로운 접근법을 제안합니다. 이는 'thinking fast & slow' 전략을 활용한 두 단계의 multiple instance learning (MIL) 방법론을 활용하여 재발 예측 작업을 수행합니다.

- **Technical Details**: 제안된 방법은 먼저 low WSI 해상도에서 가장 중요한 영역을 식별하는 1st 단계('thinking fast')와 고해상도 패치를 활용해 TTR(time to recurrence)을 예측하는 2nd 단계('thinking slow')로 구성됩니다. 이 방식은 전체 이미지의 패치를 기반으로 하여 MC(Mean C-index) 측정에서 0.733을 기록하였습니다.

- **Performance Highlights**: 최종 모델은 LEOPARD 챌린지 검증 세트에서 C-index가 0.603을 기록하며 생화학적 재발 예측에서 기존 방법론에 비해 개선된 성능을 보여주었습니다.



### K-Origins: Better Colour Quantification for Neural Networks (https://arxiv.org/abs/2409.02281)
Comments:
          16 pages, 13 figures, 1 table

- **What's New**: K-Origins라는 새로운 신경망 층이 개발되어 이미지 기반 네트워크 성능을 향상시키는데 기여함. 이는 색상이나 강도를 학습할 때 유리하게 작용한다.

- **Technical Details**: K-Origins는 입력 특징으로부터 출력 특징을 생성하여 의미 분할(semantic segmentation) 정확도를 높이는 데 초점을 맞춥니다. 주요 특징으로는 $	extbf{Y}_k = 	extbf{X}-	extbf{J}ullet w_k$라는 수식을 통해 trainable parameter를 활용하여 입력 특징을 처리하는 방식이 있습니다. 또한, 다양한 receptive fields가 고려되어 최적의 네트워크 깊이를 결정하는 데 기여함.

- **Performance Highlights**: K-Origins를 활용하여 저신호 대 잡음비 환경에서의 객체 탐지와 동일한 형태를 가진 여러 객체를 색상에 따라 분리하는 작업에서 의미 분할의 정확도가 개선됨. 네트워크는 250개 이상의 encoder-decoder convolutional networks로 테스트되었고, 16-bit synthetic 데이터에 대해 최적의 성능을 나타냄.



### Evaluation and Comparison of Visual Language Models for Transportation Engineering Problems (https://arxiv.org/abs/2409.02278)
- **What's New**: 본 연구는 비전 언어 모델(vision language models, VLM)이 교통 공학 분야에 적용되는 최신 기술을 탐구하였습니다. 특히, 이미지 분류(image classification)와 물체 탐지(object detection) 작업을 통해 교통 분야의 필요를 충족합니다. congestion detection 및 crack identification 등 다양한 문제들을 해결하는 데 VLM을 활용하였습니다.

- **Technical Details**: 이 연구에서는 CLIP, BLIP, OWL-ViT, Llava-Next 및 GPT-4o와 같은 오픈 소스 및 폐쇄 소스 VLM 모델을 활용하여 성능을 평가했습니다. 이들 모델은 제로샷 프롬핑(zero-shot prompting) 기법을 통해 훈련 없이도 다양한 작업을 수행할 수 있는 능력을 보여주고, 주석이 달린 데이터 세트나 특정 작업에 대한 세부 조정을 필요로 하지 않습니다.

- **Performance Highlights**: 이미지 분류 작업에서는 기존의 Convolutional Neural Networks (CNN) 모델과 비교했을 때 비슷한 결과를 보였으나, 물체 탐지 과제에서는 개선이 필요함을 보여주었습니다. 본 연구는 VLM 모델의 장단점을 강조하였으며, 향후 개선 사항 및 대규모 구현을 위한 기준점을 제시하였습니다.



### ADHD diagnosis based on action characteristics recorded in videos using machine learning (https://arxiv.org/abs/2409.02274)
Comments:
          Neuroscience Applied

- **What's New**: 이 연구는 ADHD 진단을 위한 새로운 행동 인식 방법을 도입합니다. 이는 원시 비디오 녹화를 식별하고 분석하여 이루어집니다.

- **Technical Details**: 주요 기여는 1) 세 대의 카메라를 통해 참가자들의 주의력과 과잉 행동/충동성을 기록하는 테스트를 설계 및 구현하고, 2) 행동 인식 신경망(based on action recognition neural networks)을 기반으로 한 새로운 머신 러닝 ADHD 진단 시스템을 처음으로 구현하며, 3) ADHD 행동 특성의 진단 결과 및 분석을 제공하기 위한 분류 기준을 제안하는 것입니다.

- **Performance Highlights**: 이 시스템은 기존 서비스의 수요를 효율적으로 충족시키며, ADHD 진단의 정확성을 향상시킬 것으로 기대됩니다.



### Action-Based ADHD Diagnosis in Video (https://arxiv.org/abs/2409.02261)
Comments:
          31st European Symposium on Artificial Neural Networks

- **What's New**: 이번 연구에서는 ADHD 진단 과정에서 비디오 기반의 프레임 수준 행동 인식 네트워크를 처음으로 도입하였으며, 실제 멀티모달 ADHD 데이터셋을 기록하고 비디오 모달리티에서 세 가지 행동 클래스를 추출하였습니다.

- **Technical Details**: 원래의 C3D 구조와는 차별화되어, 입력 데이터 크기에 맞추기 위해 완전 연결 계층이 추가된 3D-CNN 구조를 구현했습니다. 이 시스템은 데이터 처리, 행동 인식, 정지 비율 계산, ADHD 진단의 네 가지 주요 구성 요소를 포함하고 있습니다.

- **Performance Highlights**: 17명의 참가자에 대한 평균 SR (Stationary Ratio) 값은 0.71로 나타났으며, 7명의 ADHD 환자는 0.50, 10명의 일반인(controls)은 0.86의 평균 SR 값을 보였습니다.



### How to Determine the Preferred Image Distribution of a Black-Box Vision-Language Model? (https://arxiv.org/abs/2409.02253)
- **What's New**: 본 연구는 이미지 배급(Distribution) 이해의 중요성을 강조하며, 블랙박스 Vision-Language Models (VLMs)의 출력 일관성을 측정하여 더 나은 성과를 유도하는 데이터 분포를 식별하는 새로운 방법론을 제안합니다.

- **Technical Details**: 이 논문에서는 VLM의 다양한 입력 프롬프트에 대한 출력 일관성을 측정하여 선호하는 이미지 분포를 식별하는 방법을 정립하였습니다. 또한, 전문가 피드백을 통한 인컨텍스트 학습(In-context Learning with Human Feedback, ICL-HF)을 적용해 VLM의 출력 품질을 향상시켰습니다. CAD 관련 비주얼 질문 응답 작업을 평가하기 위한 CAD-VQA라는 새로운 데이터세트를 도입하였습니다.

- **Performance Highlights**: 제안된 방법론을 통해 기존 VLM의 성과를 기준으로 설정할 수 있는 CAD-VQA 데이터세트에서의 출력 일관성을 바탕으로 VLM의 신뢰성을 평가할 수 있는 프레임워크를 제공하며, 이는 다양한 복잡한 비주얼 추론 작업에서 전문가 수준의 해석을 필요로 하는 분야들의 기술적 진보를 가능하게 합니다.



### NoiseAttack: An Evasive Sample-Specific Multi-Targeted Backdoor Attack Through White Gaussian Nois (https://arxiv.org/abs/2409.02251)
- **What's New**: NoiseAttack는 단일 희생자 클래스 대신 여러 목표 클래스를 생성할 수 있는 샘플-특정 다중-targeted backdoor attack을 소개합니다. 이는 White Gaussian Noise (WGN)를 트리거 패턴으로 사용하여 모델의 훈련 단계에서 은밀하게 백도어를 삽입합니다.

- **Technical Details**: NoiseAttack에서는 White Gaussian Noise (WGN)의 다양한 Power Spectral Densities (PSD)를 트리거로 사용합니다. 이 방법은 모델의 훈련 과정에서 모든 입력 샘플에 WGN을 삽입하여, 목표 클래스에 대해 악의적으로 동작하도록 할 수 있습니다.

- **Performance Highlights**: NoiseAttack은 인기 있는 네트워크 아키텍처와 데이터셋에 대해 높은 공격 성공률을 달성하며, 최신 백도어 탐지 방법을 우회할 수 있음을 실험적으로 입증하였습니다.



### A Novel Audio-Visual Information Fusion System for Mental Disorders Detection (https://arxiv.org/abs/2409.02243)
Comments:
          27th International Conference on Information (FUSION)

- **What's New**: 정신 장애 진단을 위한 새로운 다중 모달(multi-modal) 시스템이 소개되었습니다. 이 시스템은 음성 및 얼굴 비디오 입력을 기반으로 하며, ADHD 및 우울증 진단을 최초로 통합하여 수행합니다.

- **Technical Details**: 제안된 시스템은 spatial-temporal attention networks를 기반으로 하며, 계산 집약적이지 않은 pre-train audio recognition network를 사용하여 video recognition 모듈의 성능을 향상시킵니다. 이 시스템은 실제 다중 모달 ADHD 데이터셋에서 80% 이상의 정확도를 달성하였고, AVEC 2014 우울증 데이터셋에서도 최첨단 성과를 기록했습니다.

- **Performance Highlights**: 이 시스템은 ADHD 및 우울증 데이터셋에서 최고 성과를 거리며, 특히 다중 모달 방식이 기존의 단일 진단 방법에 비해 효과적임을 입증했습니다.



### EgoPressure: A Dataset for Hand Pressure and Pose Estimation in Egocentric Vision (https://arxiv.org/abs/2409.02224)
- **What's New**: 여기서는 새로운 EgoPressure 데이터셋을 소개합니다. 이 데이터셋은 손-물체 상호작용에서 접촉 압력(contact pressure) 추정에 필요한 에고센트릭(egocentric) 시점의 데이터로 구성되어 있습니다. 특히 손 자세와 정밀한 압력 강도 데이터를 포함하고 있으며, AR/VR 및 로봇 공학 응용 프로그램에서 손-물체 상호작용을 이해하는 데 기여할 것이 기대됩니다.

- **Technical Details**: EgoPressure는 21명의 참가자로부터 수집된 5.0시간 분량의 터치 접촉 및 압력 상호작용 데이터를 포함하고 있습니다. 데이터 수집은 8대의 RGBD 카메라와 이동하는 에고센트릭 카메라를 사용하여 수행되었습니다. 이 데이터셋에서는 고품질의 손 자세(hand pose) 및 압력(data) 데이터가 제공되며, 다양한 모드로 압력을 추정하는 기준선도 제시합니다. 이를 위해 다중 뷰 시퀀스 기반 최적화 기법이 사용되었습니다.

- **Performance Highlights**: 이 연구는 손 자세와 압력이 상호 보완적임을 보여줍니다. 한편, EgoPressure는 변화하는 카메라 시점에서도 정확한 압력 추정을 가능하게 하며, 최신 헤드 마운트 디스플레이(HMD) 시스템에서의 손 자세와 함께 압력 정보를 시각적으로 결합하여 상호작용의 자유도를 향상시킵니다. 그 결과, 이 데이터셋은 에고센트릭 압력 탐지 및 에고센트릭 상호작용 연구에 새로운 기준을 제시합니다.



### Self-Supervised Learning for Identifying Defects in Sewer Footag (https://arxiv.org/abs/2409.02140)
Comments:
          Poster at the LatinX in AI Workshop @ ICML 2024

- **What's New**: 이 연구는 수도관 검사에 Self-Supervised Learning (SSL)을 처음으로 적용하였습니다. 기존의 수작업 검사에 의존하는 방법 대신, 적은 양의 라벨이 지정된 데이터로도 경쟁력 있는 성능을 달성하는 자동화된 솔루션을 제안합니다.

- **Technical Details**: DINO 방법론을 사용하여 1.3백만 개의 이미지와 17개 결함 유형이 포함된 Sewer-ML 데이터셋에서 평가를 진행하였습니다. 모델은 기존 방법에 비해 최소 5배 작으며, 10%의 데이터로도 강력한 결과(50.05 F2CIW, 87.45 F1Normal)를 나타냅니다. SSL 방법은 라벨링 데이터 수집의 어려움을 해결할 수 있는 혁신적인 접근 방식으로 자리 잡고 있습니다.

- **Performance Highlights**: 기존의 첨단 방법들과 경쟁력 있는 성능을 보임에도 불구하고, 훨씬 더 작은 모델을 성공적으로 훈련하여 작은 장치에서 실시간 탐지에 적합합니다. 이는 자원이 제한된 환경에서도 확장 가능성을 높입니다.



### LongLLaVA: Scaling Multi-modal LLMs to 1000 Images Efficiently via Hybrid Architectur (https://arxiv.org/abs/2409.02889)
Comments:
          19 pages, 7 figures, 6 tables

- **What's New**: 이번 논문에서는 멀티모달 대형 언어 모델(Multi-modal Large Language Models, MLLMs)의 긴 맥락 처리 능력을 확장하는 새로운 슈퍼 모델인 LongLLaVA를 소개합니다. 이는 Mamba와 Transformer 블록의 하이브리드 아키텍처를 채택하고, 여러 이미지 간의 시간적 및 공간적 종속성을 고려한 데이터 구축 및 점진적 훈련 전략을 적용한 것입니다.

- **Technical Details**: LongLLaVA는 멀티모달 아키텍처, 데이터 구성 및 훈련 전략의 세 가지 차원에서 종합적으로 최적화된 모델입니다. Mamba-Transformer 하이브리드 아키텍처는 효율적인 이미지 표현 방식을 적용하고, 고유한 데이터 형식을 설계하여 다중 이미지 처리 시 성능 저하를 방지합니다. 훈련 전략은 단일 이미지 정렬, 단일 이미지 지시 튜닝 및 다중 이미지 지시 튜닝의 세 단계로 되어 있어 점진적으로 모델의 다중 모달 긴 맥락을 다루는 능력을 향상시킵니다.

- **Performance Highlights**: LongLLaVA는 다양한 벤치마크에서 경쟁력 있는 결과를 달성하며, 특히 VNBench에서 정보 검색, 이미지 수 세기 및 정렬 작업에서 선두를 보이고 있습니다. 80GB A100 GPU에서 단일 GPU 환경에서도 1,000개의 이미지를 처리할 수 있어 뛰어난 효율성을 보여줍니다.



### CanvOI, an Oncology Intelligence Foundation Model: Scaling FLOPS Differently (https://arxiv.org/abs/2409.02885)
Comments:
          12 pages, 5 figures

- **What's New**: 디지털 종양병리학(digital oncopathology) 분야에서 발생하는 도전 과제를 해결하기 위해 CanvOI라는 새로운 기반 모델을 제안합니다. 이는 최신 기술을 활용하여 부족한 레이블 데이터 문제를 극복하고 있습니다.

- **Technical Details**: CanvOI는 ViT-g/10 기반의 모델로, 입력 이미지의 특성을 수정하여 디지털 병리학의 기능을 향상시킵니다. 우리는 이미지 타일 크기를 380 x 380 픽셀로, 패치 크기를 10 x 10 픽셀로 설정하여 모델의 성능을 최적화했습니다. 이로 인해 컴퓨팅 자원을 새로운 방향으로 활용하게 되었습니다.

- **Performance Highlights**: CanvOI는 다른 주요 디지털 병리학 기반 모델에 비해 평균 AUC에서 1.5-7.4% 향상을 이루었습니다. 초기 집단의 10%만 사용하여 훈련했을 때에도 성능 차이가 현저히 확대되어, 임상 종양 환자의 결과를 개선할 수 있는 잠재력을 보여줍니다.



### Visually Grounded Speech Models for Low-resource Languages and Cognitive Modelling (https://arxiv.org/abs/2409.02865)
Comments:
          PhD Dissertation

- **What's New**: 이 논문은 라벨이 없는 음성과 이미지 쌍에서 학습하는 시각적으로 기초한 음성(VGS) 모델을 탐구합니다. 특히 자원이 부족한 언어와 인간 언어 습득 이해에 대한 응용에 초점을 맞추고 있습니다.

- **Technical Details**: 우리는 이미지 사용하여 음성에서 키워드를 탐지하고 지역화하는 'Visually Prompted Keyword Localisation' 작업을 소개합니다. VGS 모델이 Yoruba와 같은 자원이 부족한 언어에 대한 Few-Shot Learning 상황에서 얼마나 효과적인지 보여줍니다. 또한 VGS 모델에서 상호 배타성 편향(Mutual Exclusivity Bias)을 조사합니다. 단일 언어 VGS 모델은 이 편향을 나타내지만 다국어 사용이 이 VGS 모델의 편향에 영향을 미치지 않는다는 것을 발견했습니다.

- **Performance Highlights**: 본 연구는 VGS 모델이 자원이 부족한 언어와 관련된 적은 데이터 환경에서도 효과적으로 키워드를 지역화할 수 있음을 입증하며, 어린이의 언어 습득 패턴과 유사한 과정을 보여줍니다.



### Automatic facial axes standardization of 3D fetal ultrasound images (https://arxiv.org/abs/2409.02826)
- **What's New**: 본 연구에서는 3D 초음파(US)를 활용하여 태아의 얼굴 평면을 표준화하고, 임상의가 얼굴 특성을 평가하는 데 도움을 주는 AI 기반 도구를 제안합니다. 이는 진단의 일관성을 향상시키고 임상의의 주관성 영향을 줄이는 데 중점을 두고 있습니다.

- **Technical Details**: 제안된 네트워크는 세 개의 블록으로 구성되어 있습니다: 특징 추출기(feature extractor), 회전 및 전이 회귀(rotation and translation regression), 그리고 공간 변환기(spatial transformer). 이 네트워크는 3개의 직교 2D 슬라이스를 처리하여 3D US의 얼굴 평면을 표준화하기 위한 변환을 추정합니다. 1180개의 태아 얼굴 3D US 이미지로 훈련되어 테스트 세트에서 관찰자 간 회전 변동성을 크게 줄였습니다.

- **Performance Highlights**: 테스트 결과, 평균 기하각 차이는 14.12° ± 18.27°로 나타났으며, 유클리드 각 에러는 7.45° ± 14.88°로 나타나, 본 네트워크가 얼굴 축을 효과적으로 표준화할 수 있음을 보여줍니다. 이는 임상에서 태아 얼굴 평가의 일관성과 정확성을 향상시키는 데 잠재력을 지닙니다.



### MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark (https://arxiv.org/abs/2409.02813)
- **What's New**: 이 논문은 MMMU-Pro를 소개합니다. 이는 Massive Multi-discipline Multimodal Understanding and Reasoning (MMMU) 벤치마크의 강력한 버전으로, 멀티모달 모델의 진정한 이해 및 추론 능력을 평가하기 위해 세 가지 단계로 구성된 평가 방식을 사용합니다.

- **Technical Details**: MMMU-Pro는 1) 텍스트 전용 모델이 답할 수 있는 질문 필터링, 2) 후보 옵션 증강, 3) 질문이 이미지 안에 포함된 비전 전용 입력 설정 도입의 단계를 포함합니다. 이 과정을 통해 MMMU-Pro는 모델이 항목에서 텍스트와 이미지를 동시에 읽고 이해하도록 돕는 도전을 제공합니다.

- **Performance Highlights**: MMMU-Pro에서 모델 성능은 MMMU보다 상당히 낮았으며, 16.8%에서 26.9%의 성능 저하를 보였습니다. OCR 프롬프트는 대부분의 모델에 큰 영향을 미치지 않았지만, 체인 오브 쏘트(Chain of Thought, CoT) 방식은 일반적으로 성능을 향상시켰습니다.



### UnLearning from Experience to Avoid Spurious Correlations (https://arxiv.org/abs/2409.02792)
Comments:
          10 pages

- **What's New**: 이번 연구에서는 딥러닝 모델이 훈련 데이터에서 발생하는 spurious correlation에 민감하다는 점을 지적하고, 이를 해결하기 위한 새로운 접근법인 UnLearning from Experience (ULE)를 제안합니다. 이 방법은 학생 모델과 교사 모델의 병렬 훈련을 통해 spurious correlation을 줄이는 방식입니다.

- **Technical Details**: ULE는 두 개의 분류 모델인 학생 모델(s(x))과 교사 모델(t(x))을 동시에 훈련시킵니다. 학생 모델은 제한 없이 훈련되어 spurious correlation을 학습하고, 교사 모델은 학생의 출력을 기반으로 학생이 저지른 실수를 unlearn합니다. 교사 모델은 학생의 gradient를 추적하여 잘못된 학습을 피할 수 있도록 훈련됩니다.

- **Performance Highlights**: 제안된 ULE 방법은 Waterbirds 및 CelebA 데이터셋에서 SOTA (State-Of-The-Art) 결과를 달성하였으며, Spawrious 데이터셋에서도 유사한 성과를 보였습니다.



### Validation of musculoskeletal segmentation model with uncertainty estimation for bone and muscle assessment in hip-to-knee clinical CT images (https://arxiv.org/abs/2409.02770)
Comments:
          29 pages, 7+10supp figures, 8 tables

- **What's New**: 본 연구는 의료 CT 이미지에서 고관절과 허벅지의 볼륨 기반 MSK(근골격계) 세분화를 위한 개선된 딥러닝 모델을 검증하였습니다. 기존 연구들의 한계를 극복하고 다양한 CT 이미지 데이터베이스를 활용하여 세분화 정확도를 높인 점이 주목할 만합니다.

- **Technical Details**: 개선된 모델은 여러 제조사/스캐너의 CT 이미지를 포함한 데이터베이스를 사용하였으며, 세분화 정확도와 구조의 부피 및 밀도(즉, 평균 HU)를 평가하였습니다. 또한, 예측 불확실성에 기반한 세분화 실패 감지 방법도 탐구하였습니다.

- **Performance Highlights**: 모델은 모든 세분화 정확도 및 구조의 부피/밀도 평가 메트릭에서 전반적으로 향상된 성능을 보여주었습니다. 예측 불확실성은 불완전하거나 실패한 세분화를 감지하는 데 있어 AUROC 곡선에서 큰 면적(AUROCs>=.95)을 기록하며 신뢰성을 나타냈습니다.



### Multi-Head Attention Residual Unfolded Network for Model-Based Pansharpening (https://arxiv.org/abs/2409.02675)
- **What's New**: 이번 연구는 위성 이미지 융합을 위한 모델 기반 딥 언폴딩(deep unfolding) 방법을 제안하며, 이는 전통적인 관측 모델과 고주파수 제약을 결합한 새로운 접근 방식을 포함합니다.

- **Technical Details**: 제안된 방법은 파노라마(PAN) 이미지의 기하학적 정보를 사용하여 업샘플링(upsampling) 및 다운샘플링(downsampling) 레이어를 도입합니다. MARNet이라는 멀티 헤드 어텐션(residual network) 구조를 이용하여 비근접(nonlocal) 연산자에 의한 이미지 자기 유사성(exploitation)을 활용합니다.

- **Performance Highlights**: PRISMA, Quickbird, WorldView2 데이터셋에서의 실험 결과, 제안된 방법은 다양한 센서 구성 및 공간적(spatial)과 스펙트럴(spectral) 해상도가 다른 경우에서도 우수한 성능을 보여 줍니다.



### Creating a Microstructure Latent Space with Rich Material Information for Multiphase Alloy Design (https://arxiv.org/abs/2409.02648)
- **What's New**: 이번 연구에서는 복합상 합금 설계를 위한 새로운 딥러닝 알고리즘을 제안합니다. 이 알고리즘은 진짜 미세구조 정보를 결합하여 보다 정확한 구성/처리-구조-성질(CPSP) 관계를 수립하는 데 중점을 두고 있습니다.

- **Technical Details**: 제안된 알고리즘은 Variational Autoencoder(VAE) 기반의 딥러닝 프레임워크를 활용하여 실제 미세구조 데이터를 잠재 공간(latent space)으로 매핑합니다. 이 잠재 공간 벡터를 통해 구성, 처리 단계 및 물질 성질을 예측할 수 있습니다. 이 모델은 Dual Phase Steel을 설계하는 데 적용되었고, 미세구조 중심의 알고리즘으로 성과를 평가했습니다.

- **Performance Highlights**: 알고리즘은 미세구조와 CPSP 연관성을 통합하여 균일한 Dual Phase (UniDP) 강철을 설계하는 데 성공했습니다. 이 과정에서 실험적으로 검증된 성질들을 통해 새로운 다상 합금의 효과적인 설계를 위한 강력한 기반을 마련했습니다.



### A Fashion Item Recommendation Model in Hyperbolic Spac (https://arxiv.org/abs/2409.02599)
Comments:
          This work was presented at the CVFAD Workshop at CVPR 2024

- **What's New**: 본 연구에서는 사용자의 구매 기록과 아이템의 시각적 데이터를 기반으로 하여 아이템 간의 암묵적인 계층 구조를 포착하기 위해 하이퍼볼릭 기하학을 통합한 패션 아이템 추천 모델(Hyperbolic Visual Attentive Collaborative Filtering, HVACF)을 제안합니다.

- **Technical Details**: 모델 학습에서 우리는 하이퍼볼릭 거리와 유클리드 거리(Euclidean distance)를 동시에 고려하는 멀티태스크 학습(multi-task learning) 프레임워크를 적용했습니다. 이를 통해 모델이 효과적으로 패션 아이템 간의 관계를 학습할 수 있도록 하였습니다. 하이퍼볼릭 스페이스(hyperbolic space)는 계층적 데이터 모델링에 적합한 특성을 가지고 있으며, Poincaré ball 모델을 채택하였습니다.

- **Performance Highlights**: 우리의 실험 결과, 제안된 모델은 유클리드 공간에서만 훈련된 기존 모델들보다 뛰어난 성능을 보였으며, 멀티태스크 학습이 모델 성능에 핵심적인 역할을 한다는 것을 확인하였습니다. Euclidean loss를 제거할 경우, 모델 성능이 급격히 저하됨을 보였습니다.



### Sample what you cant compress (https://arxiv.org/abs/2409.02529)
- **What's New**: 이번 연구는 기존의 GAN 기반 (Generative Adversarial Network) 오토인코더보다 더 뛰어난 재구성 품질을 제공하는 확산 기반 (Diffusion-based) 손실 함수를 사용하여 연속 인코더와 디코더를 공동 학습하는 방법론을 제안합니다. 이 접근법은 'Sample what you can't compress' (SWYCC)라는 이름으로 불리며, 새로운 이미지를 생성할 때 더 다양한 세부 사항을 포착할 수 있습니다.

- **Technical Details**: SWYCC는 확산 손실을 활용하여 오토인코더를 학습하며, 이를 위해 U-net 아키텍처를 사용하여 디코더를 구성하고 있습니다. 이 방법론은 기존의 MSE (Mean Squared Error) 손실과 결합하여 고해상도의 이미지 재구성을 가능하게 하며, 이론적으로도 KL 발산 (Kullback-Leibler Divergence)과 연결됩니다.

- **Performance Highlights**: SWYCC는 모든 압축 수준에서 SOTA (State-of-the-Art) GAN 기반 오토인코더보다 낮은 왜곡을 달성하며, 결과적으로 질적으로 더 나은 잠재 확산 모델 생성을 가능하게 합니다. 추가적으로, 두 개의 부분으로 나누어진 디코더 구조는 학습 동역학을 개선합니다.



### A Learnable Color Correction Matrix for RAW Reconstruction (https://arxiv.org/abs/2409.02497)
Comments:
          Accepted by BMVC2024

- **What's New**: 이 논문에서는 자율주행 알고리즘이 sRGB 이미지를 모델 입력으로 사용하는 것과 관련된 문제점을 다루고 있습니다. 새로운 초경량 RAW 재구성 방법이 제안되어, 실제 세계 데이터 수집의 어려움과 annotation 문제를 해결하고, RAW 이미지 도메인에서의 연구를 지원합니다.

- **Technical Details**: 제안된 모델은 단일 convolutional layer만을 사용하여 복잡한 inverse image signal processor (ISP)를 근사화하는 learnable color correction matrix (CCM)를 도입합니다. 이 접근법은 기존의 복잡한 inverse ISP 방법과 동등한 성능 향상을 실험적으로 입증했습니다.

- **Performance Highlights**: 실제 실험 결과, 저자가 제안한 방법으로 생성된 simulated RAW (simRAW) 이미지는 RAW 도메인 객체 탐지기(pretraining) 훈련에서 성능 개선을 보이며, 이로써 접근법의 효과성과 실용성을 강조합니다.



### FrameCorr: Adaptive, Autoencoder-based Neural Compression for Video Reconstruction in Resource and Timing Constrained Network Settings (https://arxiv.org/abs/2409.02453)
- **What's New**: 이 논문은 데이터가 부분적으로 수신될 때 동영상 프레임을 복원하는 새로운 방법인 FrameCorr을 소개합니다. FrameCorr은 딥러닝 기반의 프레임 간 상관관계를 활용하여 누락된 데이터를 예측하고 복원하는 기능을 제공합니다.

- **Technical Details**: FrameCorr은 이전에 수신된 데이터를 이용해 누락된 프레임의 세그먼트를 예측하여 불완전한 데이터를 사용하여 프레임을 재구성합니다. 이는 H.264 (AVC)와 같은 전통적인 비디오 압축 기법에서의 한계를 극복하기 위한 접근 방식입니다.

- **Performance Highlights**: FrameCorr는 기존의 ABR과 비교하여 처리량과 정확도에서 차별화된 성능을 발휘합니다. 연구 결과, FrameCorr이 포함된 시스템이 보다 효과적으로 누락된 데이터를 처리하는 것으로 나타났습니다.



### Diffusion Models Learn Low-Dimensional Distributions via Subspace Clustering (https://arxiv.org/abs/2409.02426)
Comments:
          39 pages, 9 figures

- **What's New**: 본 연구는 확산 모델(diffusion models)이 고차원의 이미지 데이터를 효율적으로 학습할 수 있는 메커니즘에 대한 이론적 통찰력을 제공합니다. 고립된 저차원 구조를 활용하여 이미지 데이터의 본질적 차원(intrinsic dimensions)을 고려하며, 낮은 차원의 가우시안 혼합모델(mixture of low-rank Gaussians)을 제안합니다.

- **Technical Details**: 저자들은 확산 모델의 학습 손실을 군집화 문제(subspace clustering)와 동일시하여, 낮은 차원에서의 영상 데이터의 구조를 이해하고 최적의 표본 수는 본질적 차원과 선형적으로 비례한다고 증명합니다. 또한, 이미지 데이터의 의미론적 표현과 발견된 저차원 부분공간(subspace) 간의 관계를 연구합니다.

- **Performance Highlights**: 이론적 연구 결과는 확산 모델의 학습 효율성을 높이기 위한 프레임워크를 제공하고, 이미지 생성에서의 의미론적 조작(semantic manipulation)이 가능함을 보여줍니다. 실험 결과, 이 저자들이 제안한 저차원 혼합모델이 실제 이미지 데이터를 학습하는 데 효과적임을 입증하였습니다.



### Hadamard Row-Wise Generation Algorithm (https://arxiv.org/abs/2409.02406)
- **What's New**: 본 논문에서는 Hadamard 행을 효율적으로 생성하는 알고리즘을 소개합니다. 이는 전체 행렬을 사전에 계산해야 하는 메모리 요구 사항을 감소시킵니다.

- **Technical Details**: Sylvester의 재귀 구조를 활용하여, 우리의 방법은 전체 행렬을 생성하지 않고 특정 i-번째 행을 필요에 따라 생성하는 알고리즘입니다. 이 알고리즘은 인덱스의 이진 표현을 바탕으로 곱집합(Kronecker product)을 사용하여 원하는 행을 구성합니다.

- **Performance Highlights**: 이 접근법은 단일 픽셀(single-pixel) 이미징 시스템에서 특히 유용하며, 곱집합을 n-번 수행하는 계산 복잡성은 O(2^n)로 비약적으로 자원을 절감할 수 있음을 보여줍니다.



### Neural Dynamics Model of Visual Decision-Making: Learning from Human Experts (https://arxiv.org/abs/2409.02390)
- **What's New**: 이번 연구에서는 시각 입력에서 행동 출력까지 확장된 종합적인 시각 의사결정 모델을 구현하고, 생물학적 신경망의 구조적 특성에 의존하여 CNN(Convolutional Neural Networks)과 유사한 정확도를 성취하였습니다.

- **Technical Details**: 모델은 비인간 영장류의 배외 시각 경로에 기초하여 설계되었으며, 주요 뇌 영역인 LGN(Lateral Geniculate Nucleus), V1(Primary Visual Cortex), MT(Middle Temporal Area), LIP(Lateral Intraparietal Area)를 포함합니다. 뉴런 활동은 Leaky Integrate-and-Fire (LIF) 모델을 사용하여 시뮬레이션되었으며, 심층 학습 모델 대비 생물학적 신경망의 행동 성능을 보다 정확하게 재현합니다.

- **Performance Highlights**: 모델의 선택 확률은 19.31±0.17로, 평균 인간 연구 참가자의 수준인 15.1±1.9를 통계적으로 초과했습니다. 또한, 모델의 결정 시간 곡선은 인간의 반응 시간과 잘 일치하며, 응집력이 증가함에 따라 결정 시간이 점차 짧아지는 경향을 보입니다.



### Coaching a Robotic Sonographer: Learning Robotic Ultrasound with Sparse Expert's Feedback (https://arxiv.org/abs/2409.02337)
Comments:
          Accepted in IEEE Transactions on Medical Robotics and Bionics (TMRB) 2024

- **What's New**: 이 논문은 로봇 초음파(Robotic Ultrasound, RUS)의 성능을 향상시키기 위해 코칭(coaching) 프레임워크를 제안합니다. 기존의 LfD(learning from demonstrations) 방법을 넘어, 실시간 전문가의 피드백을 통합하여 RUS의 훈련 과정에서 인간 전문가의 적극적인 참여를 이끌어내고자 합니다.

- **Technical Details**: 제안된 방법은 Deep Reinforcement Learning (DRL)과 전문가의 드물게 제공되는 피드백을 결합합니다. DRL은 이미지 품질 평점을 기반으로 한 보상을 사용하며, 전문가의 코칭은 Partially Observable Markov Decision Process (POMDP)로 모델링되어 정책 파라미터를 업데이트합니다.

- **Performance Highlights**: 검증 연구 결과, 코칭을 통해 학습 속도가 25% 증가하고 질 높은 이미지 획득 수는 74.5% 증가하는 것으로 나타났습니다.



### YoloTag: Vision-based Robust UAV Navigation with Fiducial Markers (https://arxiv.org/abs/2409.02334)
- **What's New**: 이번 연구에서 제안된 YoloTag는 실제 환경에서 무인 항공기(UAV)의 실시간 fiducial marker 기반 위치 확인 시스템으로, YOLO v8 객체 탐지기를 사용하여 fiducial markers를 정확하게 감지하고 내비게이션에 필요한 성능 기준을 충족합니다.

- **Technical Details**: YoloTag는 경량 YOLO v8 모델을 통합하여 다양한 실세계 이미지를 처리하면서도 실시간 성능 요구 사항을 만족하여 다중 마커 감지 및 3D 자세 추정의 효율적인 머신러닝 아키텍처를 제공합니다. 경량 CSPDarkNet-53 네트워크 구조를 기반으로 하는 마커 검출 및 각 마커의 상태를 추정합니다.

- **Performance Highlights**: 실내 환경에서 실제 로봇 실험을 통해 YoloTag의 경로 추적 성능을 다양한 거리 메트릭을 사용하여 평가한 결과, 기존 방법보다 뛰어난 정확도와 신뢰성을 보여주었습니다.



### Visual Servoing for Robotic On-Orbit Servicing: A Survey (https://arxiv.org/abs/2409.02324)
Comments:
          Accepted for publication at the 2024 International Conference on Space Robotics (iSpaRo)

- **What's New**: 이 논문은 자율적 궤도 서비스(On-orbit servicing, OOS)에 필요한 비주얼 서보링(Visual Servoing, VS) 기술을 종합적으로 검토합니다. 로봇 OOS 작업에서의 VS 접근 방식을 인식(Recognition), 접근(Approach), 접촉(Contact) 세 단계로 나누어 탐구하며, 현재 기술 동향과 미래 연구 방향을 제시합니다.

- **Technical Details**: 이 논문에서는 로봇 이동 제어 및 작업 수행에 필요한 VS 기술의 세 가지 주요 단계를 정의합니다. 또한 우주 환경에서의 도전 과제를 다루고, 카메라를 사용한 환경 인식 및 실시간 피드백 제어 시스템의 중요성을 강조합니다. 특히, 객체의 상태 추정 및 비주얼 피처 선택에 대한 고급 접근 방법이 설명됩니다.

- **Performance Highlights**: OOS 미션에서 자율 로봇 시스템의 사용은 높은 정밀도와 유연성을 제공하며, 기존의 원거리 조작 방식에 비해 신뢰성을 높입니다. 향후 연구에서는 VS 기술을 통한 자율적 로봇 조작의 효과성을 개선하기 위한 방법 모색이 필요합니다.



### QID$^2$: An Image-Conditioned Diffusion Model for Q-space Up-sampling of DWI Data (https://arxiv.org/abs/2409.02309)
Comments:
          Accepted at MICCAI 2024 International Workshop on Computational Diffusion MRI. Zijian Chen and Jueqi Wang contributed equally to this work

- **What's New**: 본 연구에서는 저각 해상도 확산 가중 이미징(DWI) 데이터로부터 고각 해상도 DWI를 추정하기 위한 이미지 조건부 확산 모델(QID$^2$)을 제안합니다. 이 모델은 대조군 이미지의 위치 정보를 보존하기 위해 U-Net 아키텍처와 교차 주의(cross-attention)를 활용합니다. DWI 데이터는 Human Connectome Project(HCP) 데이터셋을 활용하여 훈련 및 평가됩니다.

- **Technical Details**: QID$^2$ 모델은 저각 해상도 DWI 데이터 집합을 입력으로 받아 특정 그래디언트 방향과 관련된 DWI 데이터를 추정합니다. 이 모델은 가장 가까운 방향을 자동으로 식별하고 해당 이미지를 사전 지식으로 사용하여 타겟 이미지를 생성하며, 이 과정은 U-Net 기반 구조로 수행됩니다. 또한, QID$^2$는 forward noising process와 reverse denoising process를 포함한 위치 민감(diffusion) 모델입니다.

- **Performance Highlights**: QID$^2$는 두 개의 최신 GAN 모델과 비교했을 때 높은 품질의 이미지를 생성할 뿐만 아니라, 여러 가지 지표에서 하위 텐서 추정에서 GAN 모델을 일관되게 초월하는 성능을 보여줍니다. 이 연구는 임상 및 연구 응용을 위한 Q-space 업 샘플링에서 확산 모델, 특히 QID$^2$의 잠재력을 강조합니다.



### Unsupervised Welding Defect Detection Using Audio And Video (https://arxiv.org/abs/2409.02290)
Comments:
          21 pages

- **What's New**: 이 연구에서는 로봇 용접에서 AI의 응용 가능성을 탐구하였습니다. 기존 로봇 용접 기술은 결함 검출 능력이 부족한데, 본 연구는 미세한 결함을 실시간으로 감지할 수 있는 심층 학습 방법을 제안하였습니다.

- **Technical Details**: 딥러닝(deep learning) 방법을 활용하여 마이크와 카메라를 통해 용접 과정 중 발생하는 결함을 실시간으로 감지하는 방법을 설명하고 있습니다. 4000개 이상의 용접 샘플을 수집하여, 무감독(unsupervised) 방식으로 딥러닝 모델을 학습하였습니다. 오디오와 비디오 데이터를 결합한 다중 모달(multi-modal) 접근 방식을 통해 11종의 결함 유형에 대해 0.92의 평균 AUC(Area-under-ROC-Curve)를 달성하였습니다.

- **Performance Highlights**: 본 연구는 카메라와 마이크를 이용하여 실시간 용접 결함 감지 성능을 입증하였으며, 다중 모달 접근 방식을 통해 결함 탐지 성능을 향상시켰습니다. 실제 산업 환경에서의 실험을 통해, 다양한 결함 유형에 걸쳐 높은 정확도를 달성하였습니다.



### Optimal L-Systems for Stochastic L-system Inference Problems (https://arxiv.org/abs/2409.02259)
- **What's New**: 이 논문에서는 주어진 문자열 시퀀스를 생성할 수 있는 최적의 확률적 L-시스템을 구축하기 위한 두 가지 새로운 정리를 제시합니다. 첫 번째 정리는 단일 파생(de derivation)을 통해 주어진 단어 시퀀스를 생산할 확률을 극대화하는 확률적 L-시스템을 만드는 방법을 설명하며, 두 번째 정리는 여러 가능한 파생을 가진 단어 시퀀스의 생산 확률이 가장 높은 확률적 L-시스템을 규명합니다.

- **Technical Details**: 논문에서는 비선형 프로그래밍 솔버(nonlinear programming solvers)를 사용하는 최적화 기법을 통해 주어진 문자열 시퀀스로부터 최적의 확률적 L-시스템을 추론하는 알고리즘을 도입합니다. 이 알고리즘은 긍정적인 데이터만을 사용하여 훈련할 수 있는 L-시스템을 모델링하는 방법을 제안하여, 머신 러닝 기법 사용에 대한 새로운 가능성을 열어줍니다.

- **Performance Highlights**: 확률적 L-시스템은 실제 식물의 성장 패턴을 모방하여 생물학적 프로세스를 모델링하고, 합성 이미지를 생성하여 인공지능 신경망의 훈련 데이터로 사용할 수 있는 잠재력을 지니고 있습니다. 이로 인해 데이터 레이블링의 수고를 줄이고, 새로운 L-시스템을 알고리즘적으로 생성함으로써 생물학적 현상 모델링의 실용성을 크게 향상시킬 수 있습니다.



### What makes a face looks like a hat: Decoupling low-level and high-level Visual Properties with Image Triplets (https://arxiv.org/abs/2409.02241)
Comments:
          Accepted at Workshop on Human-inspired Computer Vision @ ECCV2024

- **What's New**: 이 논문은 저수준(low-level)과 고수준(high-level) 시각 특성을 구분하기 위한 새로운 방법을 제안합니다. 특히, 다른 유형의 자극 세트를 통해 이러한 특성의 영향을 분석할 수 있습니다.

- **Technical Details**: 본 연구에서는 CORnet-S와 VGG-16이라는 두 개의 Convolutional Neural Networks (CNNs)를 사용하여 시각 자극을 생성합니다. Triplet (root, image1, image2)의 구성을 통해 서로 다른 시각적 유사성 수준을 조작합니다. 또한, Neural Predictivity 메트릭을 활용하여 뇌 영역에서의 신경 반응을 예측합니다.

- **Performance Highlights**: 연구 결과, CORnet-S는 고수준 유사성을 설명하는 데 우수한 성능을 보였고, VGG-16은 저수준 유사성을 설명하는 데 더 뛰어난 예측력을 나타냈습니다. 해당 네트워크들의 신경 기록 데이터에 대한 설명 능력은 Brain-Score 평가와 정 qualitatively 대응했습니다.



### Brain-Inspired Online Adaptation for Remote Sensing with Spiking Neural Network (https://arxiv.org/abs/2409.02146)
- **What's New**: 본 연구는 원거리 센싱을 위한 스파이킹 신경망(spiking neural networks, SNN)의 온라인 적응 프레임워크를 제안합니다. 특히, 이는 자원 제약이 있는 엣지(Edge) 장치에서 에너지 효율성을 극대화하고 빠른 환경 적응을 제공하는 방법으로 기존 시스템과 차별화됩니다.

- **Technical Details**: 제안된 프레임워크는 사전 훈련된 SNN 모델을 기반으로 하며, 비지도 온라인 적응 알고리즘을 설계했습니다. BPTT(backpropagation through time) 알고리즘을 근사하여 시간상의 연산을 단순화함으로써 SNN 적응 학습의 계산 복잡성을 현저히 낮춥니다. 또한, 낮은 시간 단계에서도 적응 성능을 향상시키기 위해 적응형 활성화 스케일링 스킴을 도입하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 다양한 날씨 조건 하에서 기존의 도메인 적응(domain adaptation) 및 도메인 일반화(domain generalization) 기법을 크게 능가하는 것을 보여주었습니다. CNN(Convolutional Neural Network) 기반의 기존 적응 방법과 비교했을 때, 제안된 SNN 접근 방식이 훨씬 더 에너지 효율적이고 빠른 온라인 적응을 가능하게 합니다.



### Edge AI: Evaluation of Model Compression Techniques for Convolutional Neural Networks (https://arxiv.org/abs/2409.02134)
- **What's New**: 이 연구는 이미지 분류 작업에서 ConvNeXt 모델의 압축 기술을 평가합니다. CIFAR-10 데이터셋을 사용하여 구조적 가지치기(Structured Pruning), 비구조적 가지치기(Unstructured Pruning), 동적 양자화(Dynamic Quantization) 방법을 통해 모델 크기와 계산 복잡성을 줄이면서 정확도를 유지하는 방법을 제시합니다.

- **Technical Details**: 실험은 클라우드 기반 플랫폼과 엣지 디바이스에서 수행되었으며, 이 압축 기술들의 성능을 평가했습니다. 결과적으로 구조적 가지치기 기술을 통해 최대 75%의 모델 크기 감소가 가능하였고, 동적 양자화는 파라미터 수에서 최대 95%의 감소를 달성하였습니다. 또한, 사전 훈련(Pre-training)과 압축 기법을 결합 시킨 모델들은 더 개선된 압축 성능을 보였습니다.

- **Performance Highlights**: 최종 압축 모델을 엣지 디바이스에 배포한 결과, 92.5%의 높은 정확도와 20 ms의 낮은 추론 시간을 기록하여 실제 엣지 컴퓨팅 애플리케이션에서 압축 기술의 효과성을 검증하였습니다.



### GenAgent: Build Collaborative AI Systems with Automated Workflow Generation -- Case Studies on ComfyUI (https://arxiv.org/abs/2409.01392)
- **What's New**: 이 논문에서는 AI 시스템의 새로운 접근 방식인 협업 AI 시스템을 제안합니다. 특히, 복잡한 워크플로우(workflows)를 자동으로 생성할 수 있는 GenAgent라는 LLM 기반 프레임워크를 소개하고 있습니다.

- **Technical Details**: GenAgent는 코드(code)로 워크플로우를 표현하고, 협업 에이전트(collaborative agents)와 함께 단계별로 워크플로우를 구성하는 방법을 독창적으로 구현하였습니다. ComfyUI 플랫폼에서 GenAgent를 구현하고, 새로운 벤치마크인 OpenComfy를 제안합니다.

- **Performance Highlights**: GenAgent는 실행 수준(run-level) 및 작업 수준(task-level) 평가에서 기존 접근 방식에 비해 뛰어난 성능을 보여주며, 복잡한 워크플로우를 효과적이고 안정적으로 생성할 수 있는 능력을 입증하였습니다.



### Unveiling Deep Shadows: A Survey on Image and Video Shadow Detection, Removal, and Generation in the Era of Deep Learning (https://arxiv.org/abs/2409.02108)
Comments:
          Publicly available results, trained models, and evaluation metrics at this https URL

- **What's New**: 최근 심층 학습(Deep Learning) 기반의 그림자 분석에 관한 포괄적 연구가 진행되었습니다. 이 연구에서는 그림자의 탐지, 제거, 생성에 대한 다양한 기법들을 검토하고, 딥 모델과 데이터셋, 평가 지표를 포함한 여러 측면을 분석합니다.

- **Technical Details**: 이 논문에서는 그림자의 탐지, 제거, 생성 작업을 위한 다양한 딥 모델에 대한 포괄적인 분석을 제공합니다. 실험을 통해 모델 크기, 속도 및 성능의 관계를 탐색하고, 다양한 데이터셋 간의 일반화 능력을 평가합니다. 또한, 그림자 탐지 및 제거를 위한 공개 결과와 모델, 새로운 데이터셋을 제공합니다.

- **Performance Highlights**: 이 연구에서 제안된 방법들은 기존의 그림자 탐지 및 제거 기법에 비해 성능이 크게 향상되었습니다. 특히, 다양한 데이터셋에서의 일반화 능력이 뛰어나고, 향후 인공지능 생성 콘텐츠(AIGC)와 큰 모델에 대한 논의가 포함되어 있어 그림자 분석의 미래 방향성을 제시합니다.



### DynOMo: Online Point Tracking by Dynamic Online Monocular Gaussian Reconstruction (https://arxiv.org/abs/2409.02104)
- **What's New**: 새로운 동적 온라인 단안 경량 재구성 알고리즘인 DynOMo가 소개되었습니다. 이 알고리즘은 단일 RGB 프레임으로 카메라 동작을 추정하면서 동적 장면을 온라인으로 재구성합니다.

- **Technical Details**: DynOMo는 3D Gaussian splatting 기법을 활용하여 동적 장면을 웹상에서 실시간으로 재구성합니다. 이 방법은 강력한 이미지 피처 재구성을 통해 점 궤적(point trajectories)를 구현하고, 새로운 유사성 강화 정규화 항을 도입하여 동시 대응 레벨 감독 없이 동적 장면에서의 점 추적을 가능케 합니다.

- **Performance Highlights**: DynOMo는 실시간 단안 카메라 환경에서 온라인 점 추적에 대한 첫 번째 기준선을 설정하였으며, 기존 오프라인 2D 추적기와 동등한 성능을 달성하였습니다. 이 연구는 동적 장면을 추적하고 재구성하는 분야의 연구가 더욱 발전할 수 있도록 영감을 줄 것입니다.



### Towards Real-World Adverse Weather Image Restoration: Enhancing Clearness and Semantics with Vision-Language Models (https://arxiv.org/abs/2409.02101)
Comments:
          Accepted by ECCV 2024

- **What's New**: 이 논문은 합성 데이터로 훈련된 기존의 악천후 이미지 복원 접근법의 한계를 해결하기 위해 반지도 학습 프레임워크를 제시합니다. 이 프레임워크는 비전-언어 모델(Vision-Language Model, VLM)을 활용하여 실제 환경에서 다양한 악천후 상황에서 복원 성능을 향상시키는 것을 목표로 합니다.

- **Technical Details**: 제안하는 방법은 비전-언어 모델을 사용하여 이미지의 시각적 선명도를 평가하고, 이를 통해 명확한 대체 레이블을 생성하여 이미지 복원 모델을 훈련하는 반지도 학습 방법론을 기반으로 합니다. 이중 단계 전략에서는 VLM이 이미지의 선명도를 선택하고, 날씨 관련 프로프트 학습을 통해 VLM을 조정하여 이미지 복원 프로세스를 수정합니다.

- **Performance Highlights**: 이 방법은 실제 악천후 이미지 복원에서 뛰어난 결과를 달성하여 최신 기술들보다 우수함을 질적 및 양적 비교를 통해 입증하였습니다.



### LinFusion: 1 GPU, 1 Minute, 16K Imag (https://arxiv.org/abs/2409.02097)
Comments:
          Work in Progress. Codes are available at this https URL

- **What's New**: 이 논문에서는 기존의 diffusion 모델의 한계를 극복하기 위해 새로운 linear attention 메커니즘인 LinFusion을 소개합니다. LinFusion은 고해상도 비주얼 생성에서의 성능 저하 문제를 해결하는 혁신적인 접근을 선보입니다.

- **Technical Details**: 논문에서는 Mamba 및 Gated Linear Attention과 같은 최근의 linear complexity 모델을 기반으로 한 generalized linear attention paradigm을 제안합니다. 이 메커니즘은 attention normalization과 비인과적 추론(non-causal inference)을 포함하여, 고해상도 이미지 생성의 성능을 향상시키고, 기존 StableDiffusion 모델의 컨트롤과 호환됩니다.

- **Performance Highlights**: LinFusion 모델은 50k iterations 동안 트레이닝만으로도 원래의 StableDiffusion 모델과 동등하거나 더 나은 성능을 기록하였으며, 16K 해상도의 이미지를 생성할 수 있는 능력을 보여주었습니다. 또한, LinFusion은 SD-v1.5, SD-v2.1, SD-XL과 같은 다양한 모델에서 유효성을 입증받았습니다.



### DepthCrafter: Generating Consistent Long Depth Sequences for Open-world Videos (https://arxiv.org/abs/2409.02095)
Comments:
          Project webpage: this https URL

- **What's New**: DepthCrafter는 오픈 월드 비디오에서 깊이 추정을 위한 새로운 방법으로, 기존의 방법과 달리 카메라 자세나 광학 흐름과 같은 추가 정보 없이도 고정밀의 깊이 시퀀스를 생성한다.

- **Technical Details**: DepthCrafter는 사전 훈련된 이미지-비디오 확산 모델을 기반으로 한 비디오-깊이 모델을 교육하고, 110프레임까지 가변 길이의 깊이 시퀀스를 생성할 수 있도록 설계된 3단계 훈련 전략을 채택한다. 또한 긴 비디오를 세그먼트 별로 처리하고 매끄럽게 합치는 추론 전략을 제안한다.

- **Performance Highlights**: DepthCrafter는 다양한 데이터셋에 대한 평가에서 제로샷 설정 하에 최첨단 성능을 달성했으며, 깊이 기반 비주얼 효과 및 조건부 비디오 생성과 같은 다양한 후속 응용 프로그램을 지원한다.



### Physical Rule-Guided Convolutional Neural Network (https://arxiv.org/abs/2409.02081)
- **What's New**: 이번 논문은 제한된 라벨 데이터 환경에서 CNN(Convolutional Neural Network)의 한계를 극복하기 위한 새로운 물리 기반 CNN(Physics-Guided CNN, PGCNN) 아키텍처를 제안합니다. 이 모델은 과학적 원칙과 실제 지식을 통합하여 해석 가능성과 효율성을 증가시키고자 합니다.

- **Technical Details**: PGCNN 아키텍처는 훈련 가능한 동적인 커스텀 레이어를 포함하여 물리적 규칙을 통합합니다. 이 모델은 Faster R-CNN과 ResNet-50을 기반으로 하여 차량 데이터셋에서 시험 되었으며, 박스를 제거하는 커스텀 레이어와 물리적 맥락을 고려하는 추가 레이어를 포함시킵니다. 또한 OpenAI의 LLM(대형 언어 모델)을 활용하여 객체 크기 간 관계를 동적으로 생성합니다.

- **Performance Highlights**: PGCNN은 여러 데이터셋에서 평가되었으며, baseline CNN 모델에 비해 false positives(허위 검사 양성)를 감소시키고 true detection(정확한 검출)의 confidence score(신뢰도 점수)를 향상시키는 결과를 보였습니다. 이는 PGCNN이 다양한 응용 분야에서 CNN의 신뢰성과 효율성을 높일 잠재력을 지니고 있음을 나타냅니다.



### F2former: When Fractional Fourier Meets Deep Wiener Deconvolution and Selective Frequency Transformer for Image Deblurring (https://arxiv.org/abs/2409.02056)
Comments:
          20 pages, 21 figures

- **What's New**: 본 논문에서는 Fractional Fourier Transform (FRFT)를 기반으로 한 이미지 디블러링 기술을 제안합니다. 이는 기존 Fourier Transform (FT)의 한계를 극복하고 이미지와 같은 비정상 신호의 처리에서 유용한 새로운 접근 방식을 제시합니다.

- **Technical Details**: 제안된 방법은 Fractional Fourier Transformer (F2former)이며, 비네르 디콘볼루션(F2WD)을 기반으로 하여 잔차 이미지의 뚜렷한 표현을 생성합니다. F2former는 새로운 fraction frequency aware transformer block (F2TB)을 포함하여 효율적인 기능 복원 및 디블러링을 수행합니다. 이 구조는 주파수 인식 self-attention (F2SA)과 주파수 분할 다중화(FM-FFN) 네트워크로 구성됩니다.

- **Performance Highlights**: 실험 결과, 제안된 F2former 모델은 모션 디블러링과 디포커스 디블러링 모두에서 다른 최신 기술(SOTA)에 비해 우수한 성능을 보였습니다. PSNR(픽셀 신호 대 잡음비)와 같은 지표에서도 높은 효율성을 기록하여 디블러링 기술의 새로운 이정표를 제시했습니다.



### Low-Resolution Face Recognition via Adaptable Instance-Relation Distillation (https://arxiv.org/abs/2409.02049)
Comments:
          Accepted by IJCNN 2024

- **What's New**: 본 연구에서는 저해상도 얼굴 인식을 용이하게 하기 위해 적응 가능한 인스턴스-관계 증류 방법(Adaptable Instance-Relation Distillation, AIRD)을 제안합니다. 이 방법은 지식 전이 과정을 증류와 적응 단계로 나누어 모델의 적응성을 향상시킵니다.

- **Technical Details**: 우리의 접근법은 인스턴스 수준(instance-level)과 관계 수준(relation-level)에서 고해상도 얼굴의 지식을 학생 모델에 전이하여 저해상도 얼굴 인식을 지원합니다. 교사 모델은 고해상도 얼굴에 대한 풍부한 지식을 가지고 있으며, 학생 모델은 이를 통해 디스크리미네이티브 특성을 생성합니다. 또한, 테스트 시에는 적응형 배치 정규화(Adaptive Batch Normalization, FaceBN)를 통해 실제 저해상도 얼굴 인식을 위한 전이 능력을 향상시킵니다.

- **Performance Highlights**: 광범위한 실험 결과는 제안된 방법이 저해상도 얼굴 인식에서 최첨단 성능을 달성함을 보여줍니다. 이를 통해 모델의 적응성과 효과성을 분명히 입증했습니다.



### ViewCrafter: Taming Video Diffusion Models for High-fidelity Novel View Synthesis (https://arxiv.org/abs/2409.02048)
Comments:
          Project page: this https URL

- **What's New**: 이번 연구에서는 단일 이미지 또는 희소 이미지로부터 고충실도의 새로운 장면 뷰를 합성하기 위한 새로운 방법인 ViewCrafter를 제안합니다. 비디오 확산 모델(video diffusion model)의 사전 지식을 활용하여, 3D 정보와 카메라 자세 제어를 정밀하게 다루며 고품질 비디오 프레임을 생성합니다.

- **Technical Details**: ViewCrafter는 포인트 클라우드(point cloud) 표현과 비디오 확산 모델을 결합하여 높은 충실도의 새로운 뷰를 합성합니다. 이는 6DoF(6 Degrees of Freedom) 카메라 자세를 정밀하게 제어할 수 있도록 하며, 반복적인 뷰 합성 전략과 콘텐츠 적응형 카메라 경로 계획 알고리즘을 통해 새로운 뷰의 범위와 커버리지를 확장합니다. 이 방법은 초기 포인트 클라우드를 기반으로 카메라 경로를 예측하고 그에 따라 포인트 클라우드를 렌더링한 후, ViewCrafter를 통해 새로운 뷰를 합성하는 방식으로 진행됩니다.

- **Performance Highlights**: 다양한 데이터셋에서 실험한 결과, ViewCrafter는 제로샷(Zero-shot) 새로운 뷰 합성과 희소 뷰 3D-GS 재구성에서 강력한 일반화 능력과 뛰어난 성능을 보여주었습니다. 특히, 이미지 품질과 자세 정확도 지표에서 기존 방법들보다 우수한 성능을 나타냈으며, 3D-GS 재구성 분야에서도 이전의 최첨단 성능을 초과했습니다.



### Human-AI Collaborative Multi-modal Multi-rater Learning for Endometriosis Diagnosis (https://arxiv.org/abs/2409.02046)
- **What's New**: 이 논문에서는 3가지 주요 측면(다중 평가자 학습, 다중 모달 학습, 인간-AI 협업)을 탐구하여 새로운 HAICOMM(인간-AI 협업 다중 모달 다중 평가자 학습) 방법론을 제안합니다. 이는 MRI 이미지에서의 POD(더글라스 주머니) 폐쇄를 정확히 분류하는데 도움이 됩니다.

- **Technical Details**: HAICOMM은 '노이즈'가 있는 여러 레이블에서 더 깔끔한 레이블을 추출하는 다중 평가자 학습, T1/T2 MRI 이미지를 활용하는 다중 모달 학습, 그리고 의사와 AI 모델의 예측을 통합하는 인간-AI 협업을 포함하여 세 가지 요소를 동시에 탐구합니다.

- **Performance Highlights**: HAICOMM 모델은 수집된 다중 평가자 T1/T2 MRI 자궁내막증 데이터셋을 사용하여 의사 그룹, 노이즈 레이블 학습 모델, 및 다중 평가자 학습 방법에 비해 더 정확한 POD 분류 결과를 나타냅니다.



### AllWeatherNet:Unified Image enhancement for autonomous driving under adverse weather and lowlight-conditions (https://arxiv.org/abs/2409.02045)
- **What's New**: 이 새로운 연구에서는 눈, 비, 안개, 야간 등 다양한 악천후 조건에서 이미지 퀄리티와 명확성을 향상시키기 위한 AllWeather-Net이라는 방법을 제안합니다. 이 방법은 계층적 아키텍처를 채택하여 세 가지 의미적 수준(장면, 객체, 텍스처)에서 정보를 조합하여 이미지를 개선합니다.

- **Technical Details**: AllWeather-Net은 Scaled Illumination-aware Attention Mechanism (SIAM)을 도입하여 자율주행 인식에 중요한 도로 요소에 대한 학습을 유도합니다. 이 아키텍처는 다층 의미적 패치의 차별화를 통해 다양한 기상 조건에서도 견고하게 작동합니다.

- **Performance Highlights**: AllWeather-Net은 악천후 이미지를 햇살 좋은 날의 장면으로 효과적으로 변환하며, 성능 측면에서도 의미적 분할(semantic segmentation)에서 최대 5.3% mIoU 향상을 보여주고, 재훈련 없이도 이전 도메인에서 최대 3.9% mIoU 개선을 달성합니다.



### Efficient Point Cloud Classification via Offline Distillation Framework and Negative-Weight Self-Distillation Techniqu (https://arxiv.org/abs/2409.02020)
- **What's New**: 본 논문은 오프라인 지식 증류(Offline Knowledge Distillation) 전략을 제시하여, 자원 제약 환경에서도 효율적인 포인트 클라우드(point cloud) 분류를 위한 모델 압축 기술을 개선하고자 합니다. 특히, 기존의 KD(지식 증류)에서 요구되는 교사 모델과 학생 모델의 동시 로딩 문제를 해결하여 하드웨어 요구 사항을 줄이는 방안을 소개합니다.

- **Technical Details**: 제안하는 프레임워크는 데이터를 증강시키는 여러 기법들을 활용하여 교사 모델의 로짓(logit) 출력을 생성하고, 이 출력값과 증강 파라미터를 오프라인으로 기록하는 방식으로 진행됩니다. 여기에는 난수 크기 조정(random scaling) 및 변환(translation) 등의 형태적 증강을 포함하며, 포인트 수준의 증강 작업은 제외됩니다. 이와 함께, 부정 가중치 자기 증류(negative-weight self-distillation) 기술을 통합하여 학생 모델의 일반화 능력을 향상시키려고 합니다.

- **Performance Highlights**: 실험 결과, 제안하는 증류 전략을 통해 학생 모델은 최신 모델(state-of-the-art)과 비교해 유사한 성능을 달성할 수 있었으며, 더 낮은 파라미터 수를 유지하면서도 성능과 복잡성 사이의 최적 균형을 이룰 수 있었습니다. 이 접근 방식은 특히 자원이 제한된 환경에서 포인트 클라우드 분석을 위한 새로운 솔루션을 제공합니다.



### TransDAE: Dual Attention Mechanism in a Hierarchical Transformer for Efficient Medical Image Segmentation (https://arxiv.org/abs/2409.02018)
- **What's New**: TransDAE는 의료 이미지 분할을 위한 새로운 계층형 Transformer 모델로, 공간적 및 채널 간 연관성을 통합하는 이중 주의 메커니즘을 도입합니다. 이 방법은 기존 모델들의 한계를 극복하여 잦은 로컬 정보와 글로벌 의존성을 효과적으로 캡처합니다.

- **Technical Details**: TransDAE는 효율적인 자기 주의(self-attention) 및 향상된 자기 주의 메커니즘을 통합하여 고해상도의 의료 이미지를 효과적으로 모델링하면서 계산 복잡성을 줄입니다. 또한, Inter-Scale Interaction Module(ISIM)을 통해 스킵 연결 경로를 강화하여 특성 재사용을 촉진하고 위치 정확성을 개선합니다.

- **Performance Highlights**: TransDAE는 Synaps 다중 장기 데이터셋에서 기존 최첨단 방법들을 초월하는 성능을 발휘하며, 미리 훈련된 가중치를 사용하지 않고도 뛰어난 결과를 달성합니다.



### Deep learning for objective estimation of Parkinsonian tremor severity (https://arxiv.org/abs/2409.02011)
- **What's New**: 이 논문에서는 파킨슨병(PD)의 자세 떨림을 분석하기 위해 비디오 데이터를 사용한 픽셀 기반 딥 러닝 모델을 소개합니다. 기존의 포즈 추정 기법의 한계를 극복하고, 2,742건의 평가 데이터를 바탕으로 임상 평가와 높은 일치를 나타냈습니다.

- **Technical Details**: 3D Conv-LSTM(3D Convolutional Long Short-Term Memory) 모델이 제안되어, 비디오 프레임에서 떨림의 키포인트를 독립적으로 추출하는 것에 초점을 맞추었습니다. 이 방식은 자세 의존적 방법보다 더 안전하고 신뢰할 수 있는 결과를 도출하였습니다.

- **Performance Highlights**: 모델은 조기 진단과 환자 관리에 있어 76%의 민감도로 떨림의 중증도를 정확히 분류했으며, 특정 치료 조건에서 임상의 평가와 일치하는 예측을 수행했습니다. 또한, 모델은 떨림의 비대칭성을 식별하는 데 있어 77%의 정확성을 보였습니다.



### PMT-MAE: Dual-Branch Self-Supervised Learning with Distillation for Efficient Point Cloud Classification (https://arxiv.org/abs/2409.02007)
- **What's New**: 본 논문에서는 PMT-MAE(Point MLP-Transformer Masked Autoencoder)라는 새로운 자가 지도 학습 프레임워크를 제안합니다. 이 프레임워크는 포인트 클라우드(classification) 분류를 위한 이중 분기 구조를 가지고 있으며, Transformer와 MLP(다층 퍼셉트론) 구성 요소를 통합하여 풍부한 특징을 캡처합니다.

- **Technical Details**: PMT-MAE는 Transformer 분기가 글로벌 self-attention을 활용하여 복잡한 특징 상호작용을 처리하는 동시에, 병렬 MLP 분기는 공유된 fully connected layer를 통해 토큰을 처리하여 상보적인 특징 변환 경로를 제공합니다. 이러한 특징은 모델이 포인트 클라우드 데이터에서 좀 더 다양한 패턴과 구조 정보를 캡처할 수 있도록 돕습니다. 또한, 두 단계의 지식 증류(knowledge distillation) 전략을 통합하여, PMT-MAE는 효율적으로 데이터를 처리하고, 학습 속도를 높일 수 있습니다.

- **Performance Highlights**: PMT-MAE는 ModelNet40 분류 작업에서 93.6%의 정확도를 달성하여, baseline인 Point-MAE(93.2%)와 teacher 모델인 Point-M2AE(93.4%)를 초월합니다. 이 프레임워크는 단 40 epoch로 사전 학습과 미세 조정을 수행할 수 있어 컴퓨팅 자원이 제한된 상황에서도 뛰어난 성능을 발휘합니다.



### Robust Fitting on a Gate Quantum Computer (https://arxiv.org/abs/2409.02006)
Comments:
          Accepted by the European Conference on Computer Vision 2024 (ECCV2024) as Oral. The paper is written for a computer vision audience who generally has minimal quantum physics background

- **What's New**: 이번 논문에서는 양자 컴퓨터(Gate Quantum Computer)에서의 양자 강건 적합(quantum robust fitting) 가능성을 보여주는 새로운 회로를 제안하였다. 이는 기존의 고전적 방식의 한계를 극복하고, 1차원(1D) 문제에 대한 ℓ∞(l-infinity) 적합성 테스트를 통해 실제 양자 컴퓨터인 IonQ Aria에서 강건 적합을 시연하는 최초의 사례로 기록되었다.

- **Technical Details**: 양자 회로는 ℓ∞(l-infinity) 적합성 테스트를 수행하도록 설계되었으며, 이 방법은 1차원 Boolean influence 계산을 가능하게 한다. 이를 활용하여, 더 높은 차원 비선형 모델에 대한 Boolean influence를 축적하고 이를 실험적으로 검증하였다. 또한, 이 연구는 Bernstein-Vazirani 양자 회로(Bernstein-Vazirani Quantum Circuit)와 연계된 계산 방법론을 포함하고 있다.

- **Performance Highlights**: 실험 결과, 제안된 양자 회로는 실제 벤치마크 데이터셋([dataset])에서 강건 적합의 유효성을 성공적으로 입증하였다. 이로 인해, 양자 컴퓨터를 활용한 기하학적 모델 추정(geometry estimation)에서 기존 방법론의 적합성을 확인할 수 있었으며, 같은 범주의 문제를 해결하는 데 있어 혁신적인 가능성을 제시하였다.



### SA-MLP: Enhancing Point Cloud Classification with Efficient Addition and Shift Operations in MLP Architectures (https://arxiv.org/abs/2409.01998)
- **What's New**: 본 연구는 포인트 클라우드 분류의 계산 비효율성을 해결하기 위해 최근 CNN 최적화에서 영감을 받은 새로운 MLP 기반 아키텍처를 도입합니다. 전통적인 신경망은 곱셈 연산에 많이 의존하여 계산 비용이 높아지므로, Add-MLP와 Shift-MLP를 제안하여 곱셈 대신 덧셈 및 쉬프트 연산을 사용하여 효율성을 크게 향상시켰습니다. 이와 함께 SA-MLP라는 하이브리드 모델을 소개하여 번갈아 분포된 쉬프트 및 어더 레이어를 혼합하여 MLP 레이어를 교체합니다.

- **Technical Details**: 본 연구는 Add-MLP 및 Shift-MLP 모델을 통해 각각 곱셈을 덧셈 및 비트 쉬프트 연산으로 대체하여 계산 효율성을 높였습니다. SA-MLP 아키텍처는 번갈아 가며 배치된 어더 및 쉬프트 레이어를 사용하여 학습 시 모든 파라미터가 활성화 되도록 설계되었습니다. 특히, 어더와 쉬프트 레이어에 대해 서로 다른 학습률과 옵티마이저를 설정하여 최적의 학습 효과를 보장합니다.

- **Performance Highlights**: 철저한 실험 결과, Add-MLP 및 Shift-MLP는 경쟁력 있는 성능을 달성하였으나, SA-MLP는 기존의 곱셈 기반 MLP 모델을 크게 초월하며 최신 MLP 기반 모델들과 비교 가능한 성능을 보입니다. 이는 포인트 클라우드 분류에 있어 계산 효율성과 성능을 동시에 만족시키는 보다 효율적이고 효과적인 솔루션을 제공합니다.



### 1DCNNTrans: BISINDO Sign Language Interpreters in Improving the Inclusiveness of Public Services (https://arxiv.org/abs/2409.01975)
Comments:
          6 pages

- **What's New**: 본 연구는 인도네시아의 청각 장애인을 위한 AI 기반 수화 번역 애플리케이션 및 사전 모델 개발에 초점을 맞추고 있습니다. 이를 통해 공공 서비스에 통합이 가능하도록 하여 의사소통 장벽을 해소하고 포괄성을 증대시키고자 합니다.

- **Technical Details**: 연구에서는 LSTM과 1D CNN + Transformer(1DCNNTrans) 모델을 사용하여 수화 인식을 비교하였습니다. LSTM 모델은 94.67%의 정확도를 보였고, 1DCNNTrans 모델은 96.12%의 정확도를 기록하였습니다. LSTM 구조는 128 유닛 LSTM 층을 포함하며, CNN은 1차원 블록 구조를 바탕으로 하여 설계되었습니다. ECA(참여 채널 주의) 기법을 통해 효율적인 특성 강조가 이루어졌습니다.

- **Performance Highlights**: 두 모델 모두 90% 이상의 검증 정확도를 기록하였으며, 50개의 수화 제스처를 빠르게 분류하는 성능을 보여주었습니다. 특히 1DCNNTrans 모델은 복잡도가 다양한 클래스에 대해 더 높은 F1 점수를 기록하며 우수한 안정성을 보였습니다.



### Snapshot: Towards Application-centered Models for Pedestrian Trajectory Prediction in Urban Traffic Environments (https://arxiv.org/abs/2409.01971)
Comments:
          8 Pages, 9 Figures

- **What's New**: 이번 논문에서는 도시 교통 상황에서 보행자 경로 예측을 탐구하고 있으며, 모델의 정확도와 실생활 적용 가능성을 모두 고려하고 있습니다. 새로운 벤치마크와 함께 Snapshot이라는 모듈형 피드포워드 신경망을 소개하여, 기존의 최첨단 기술보다 우수한 성능을 발휘합니다.

- **Technical Details**: Snapshot은 transformer 아키텍처와 합성곱 신경망(CNN) 기술을 결합한 비재귀적 접근 방식으로, 에이전트 중심의 인코딩 스킴을 사용합니다. 이 모델은 다양한 운동 이력을 다루면서도 확장성 및 실시간 성능을 보여줍니다. 또한, Argoverse 2 데이터셋을 기반으로 한 전문 보행자 벤치마크를 제공하여 향후 연구를 위한 개발 플랫폼을 구축하고 있습니다.

- **Performance Highlights**: Snapshot 모델은 100만 개 이상의 훈련, 검증 및 테스트 샘플을 포함하는 Argoverse 2 기반 벤치마크에서 실험되어 우수한 예측 성능을 보였으며, 자율주행 소프트웨어 스택에도 통합되어 실제 세계에서의 적용 가능성을 입증했습니다.



### MetaFood3D: Large 3D Food Object Dataset with Nutrition Values (https://arxiv.org/abs/2409.01966)
Comments:
          Dataset is coming soon

- **What's New**: 이번 연구에서는 3D 음식 모델링을 위한 새로운 데이터셋인 MetaFood3D를 소개합니다. 이 데이터셋은 108개의 카테고리에 걸쳐 637개의 정확히 라벨링된 3D 음식 객체를 포함하고 있으며, 자세한 영양 정보와 음식 코드를 제공합니다.

- **Technical Details**: MetaFood3D 데이터셋은 다양한 텍스처 매시 파일, RGB-D 비디오, 세분화 마스크와 같은 풍부한 모달리티를 강조합니다. 이 데이터셋은 intra-class diversity를 중요시하며, 각 음식 항목에는 세부 영양 정보가 포함됩니다.

- **Performance Highlights**: 실험 결과는 MetaFood3D 데이터셋이 알고리즘 성능 개선에 큰 잠재력을 가지고 있음을 보여주며, 비디오 캡처와 3D 스캔 데이터 간의 도전적인 간극을 강조합니다. 또한 데이터셋은 고품질 데이터 생성, 시뮬레이션 및 증강의 가능성도 보여줍니다.



### Optimizing CLIP Models for Image Retrieval with Maintained Joint-Embedding Alignmen (https://arxiv.org/abs/2409.01936)
- **What's New**: 이 논문은 Contrastive Language and Image Pairing (CLIP) 모델을 최적화하여 텍스트 기반 검색 작업에서의 성능을 유지하면서, 다양한 이미지 기반 유사성 검색 시나리오를 해결하는 새로운 방법을 제안합니다.

- **Technical Details**: 제안한 두 가지 방법은 다음과 같습니다: 첫 번째는 순차적인 fine-tuning 과정으로, 이미지 인코더를 먼저 최적화한 후에 텍스트 인코더를 재조정합니다. 두 번째 방법은 retrieval-optimization 단계에서 pseudo-captions을 통합하여 embedding 공간 내에서의 직접적인 정렬을 촉진합니다.

- **Performance Highlights**: 광범위한 실험을 통해 제안된 방법들이 CLIP 성능을 향상시키고, 이미지 검색, k-NN 분류, 제로샷 텍스트 기반 분류에서의 성능을 개선하였음을 보여주었습니다. 이 모델은 각 이미지를 단일 embedding으로 유지할 수 있도록 하여 대규모 멀티모달 유사성 검색 시스템에 필요한 인프라를 간소화합니다.



### Map-Assisted Remote-Sensing Image Compression at Extremely Low Bitrates (https://arxiv.org/abs/2409.01935)
- **What's New**: 본 논문에서는 RS 이미지의 압축 과정에서 비트 전송률이 매우 낮은 경우에도 고차원적인 이미지 생성(Generative) 모델을 통해 인식 품질을 높은 수준으로 유지할 수 있는 새로운 프레임워크인 Map-Assisted Generative Compression(MAGC)을 제안합니다. MAGC는 사전 학습된 확산 모델(difusion model)을 활용하여 시각적 현실감을 가진 이미지 복원을 수행합니다.

- **Technical Details**: MAGC는 두 단계 파이프라인을 통해 RS 이미지를 압축하고 복원합니다. 첫 번째 단계에서는 이미지를 잠재 표현(latent representation)으로 매핑하고, VAE 아키텍처를 이용해 비트 전송률을 절감합니다. 두 번째 단계에서는 조건부 확산 모델을 활용하여 압축된 잠재 표현을 사용해 의미론적으로 정확한 이미지를 생성합니다. 벡터 맵(vector map)을 사용하여 의미론적 및 구조적 가이드를 제공하여 더욱 높은 이미지 복원 품질을 달성합니다.

- **Performance Highlights**: 실험 결과, MAGC는 표준 코덱 및 다른 학습 기반 방법들과 비교했을 때 인식 품질(perceptual quality)과 의미론적 정확성(semantic accuracy) 모두에서 뛰어난 성능을 보였습니다. 특히, MAGC는 VTM-23.3 대비 90%, MS-ILLM 대비 60% 더 낮은 비트 전송률을 달성하면서도 더 나은 mIoU를 기록했습니다.



### Comprehensive Equity Index (CEI): Definition and Application to Bias Evaluation in Biometrics (https://arxiv.org/abs/2409.01928)
Comments:
          Accepted paper for the 27th International Conference on Pattern Recognition (ICPR) 2024

- **What's New**: 이번 논문에서는 머신 러닝 모델의 편향된 행동을 정량화하기 위해 설계된 새로운 지표를 제안합니다. 이 지표는 스코어 분포간의 유사성을 측정하며, 이를 통해 전반적인 형태와 꼬리 확률을 균형 있게 평가합니다. 특히 인구 통계적 편향을 정량화하는 데 유용한 점에 주목하며, 얼굴 인식 시스템의 운영 평가에 적용했습니다.

- **Technical Details**: 제안된 Comprehensive Equity Index (CEI) 지표는 두 가지 접근 방식을 결합하여, 분포의 꼬리에서 발생하는 오류와 전반적인 분포의 형태를 고려합니다. 기존의 지표들이 제공하는 제한점을 극복하기 위해, 본 논문에서는 NIST FRVT 평가에서 사용되는 고성능 시스템과 현실적인 얼굴 데이터베이스를 포함하여 성능을 측정했습니다.

- **Performance Highlights**: CEI 지표는 두 개의 최첨단 모델과 네 가지 널리 사용되는 데이터베이스에 대해 테스트되었으며, 기존의 편향 지표의 주요 결점을 극복하는 능력을 보여주었습니다. 이를 통해, 고성능 시스템에서 편향 행동을 감지하고 정량화하는 데 있어 CEI의 유용성을 강조했습니다.



### 3D-LEX v1.0: 3D Lexicons for American Sign Language and Sign Language of the Netherlands (https://arxiv.org/abs/2409.01901)
- **What's New**: 본 연구는 3D에서 수화(SL)를 효율적으로 캡처하기 위한 접근법을 제시하고, 3D-LEX v1.0 데이터세트를 도입하며, 음성 특성을 반자동으로 주석 처리하는 방법을 세부적으로 설명합니다. 이 과정에서는 고해상도의 3D 포즈, 3D 손 모양, 깊이 인지 얼굴 특징을 포함한 세 가지 모션 캡처 기술이 통합되어 있습니다.

- **Technical Details**: 3D-LEX 데이터세트는 미국 수어(ASL)에서 1,000개의 수화와 네덜란드 수어(NGT)에서 1,000개의 수화로 구성되어 있습니다. 이 데이터세트는 수어의 수동 및 비수동 마커를 캡처하기 위한 두 가지 데이터 수집 기술과 비수동 마커를 캡처하는 세 번째 기술로 기록되었습니다. 손 모양 주석을 생성하기 위해 간단한 방법이 제시되었습니다.

- **Performance Highlights**: 주석을 통해 글로스 인식 정확도를 5% 향상시키고, 전문가 주석 대비 1% 향상된 결과를 보였습니다. 여기에 3D 모션 캡처 데이터는 수화 특징에 대한 심층 분석을 지원하고, 모든 시점에서 2D 프로젝션을 생성할 수 있습니다.



### Boosting Vision-Language Models for Histopathology Classification: Predict all at onc (https://arxiv.org/abs/2409.01883)
- **What's New**: 이 논문은 비전-언어 모델(Vision-Language Models, VLMs)을 활용하여 히스토 패스홀로지에서의 제로샷(zero-shot) 성능을 향상시키기 위한 전이적 방법(transductive approach)을 소개합니다. 기존의 방법들은 슬라이드를 독립적으로 분해하고 분류하는 데 중점을 두었으나, 새로운 접근법은 패치 간의 유사도와 텍스트 기반 예측을 통해 이러한 성능을 극대화합니다.

- **Technical Details**: 제안된 Histo-TransCLIP 방법은 Gaussian 혼합 모델(Gaussian Mixture Model, GMM)을 기반으로 한 전이적 추론 과정을 사용합니다. 이 방법은 사전 해결된 특징을 바탕으로 대규모 패치 데이터를 몇 초 내에 처리할 수 있으며, 패치 간의 구조를 이용하여 정확도를 향상시킵니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 네 가지 히스토 패스홀로지 데이터 세트와 다섯 가지 VLM에서 인덕티브(inductive) 제로샷 분류에서 상대적으로 크게 개선된 정확도를 나타냈습니다. 이 방식은 별도의 레이블 없이도 효과적인 성능 향상을 달성했습니다.



### SPiKE: 3D Human Pose from Point Cloud Sequences (https://arxiv.org/abs/2409.01879)
Comments:
          16 pages, 4 figures

- **What's New**: SPiKE는 3D Human Pose Estimation (HPE)에서 포인트 클라우드 시퀀스를 활용하여 기존의 개별 프레임 처리 방식을 개선한 새로운 접근 방식입니다. 이 방법은 Temporal (시간적) 정보를 활용하여 보다 정확한 본체 키포인트를 예측합니다.

- **Technical Details**: SPiKE는 Transformer 아키텍처를 사용하여 시퀀스 내 포인트 간의 spatio-temporal (공간-시간적) 관계를 인코딩합니다. 포인트 클라우드를 지역 볼륨으로 분할하고 공간적 특성을 추출하여 Transformer의 효율적인 처리를 보장합니다. 이 방법은 ITOP 벤치마크에서 89.19% mAP를 달성하며, 깊이 맵이나 복셀 모델보다 뛰어난 성능을 기록했습니다.

- **Performance Highlights**: SPiKE는 이전 깊이 기반 HPE 방법에 비해 더 낮은 추론 시간으로 최고 성능을 달성하며, 시퀀스 정보를 활용한 처리의 중요성을 실험을 통해 입증하였습니다.



### CyberHost: Taming Audio-driven Avatar Diffusion Model with Region Codebook Attention (https://arxiv.org/abs/2409.01876)
- **What's New**: 이 논문에서는 오디오 기반의 1단계 인간 애니메이션 프레임워크인 CyberHost를 도입합니다. 이는 손의 완전성(hand integrity)과 신원 일관성(identity consistency), 자연스러운 움직임(natural motion)을 보장하며, 모든 인간 신체 부위의 애니메이션을 생성할 수 있습니다.

- **Technical Details**: CyberHost는 Region Codebook Attention 메커니즘을 채택하여 얼굴과 손의 애니메이션 품질을 향상시키고, 세부적인 지역 특징(local features) 및 학습된 움직임 패턴에 대한 선행 지식을 통합합니다. 또한, 신체 움직임 맵(body movement map), 손 선명도 점수(hand clarity score)와 같은 훈련 전략을 통해 생성 결과를 개선합니다.

- **Performance Highlights**: CyberHost는 정량적 및 정성적 측면 모두에서 기존 방법들보다 우수한 결과를 달성했습니다. 이 모델은 오디오 기반, 비디오 기반 및 다중 모달 드리븐 시나리오에서 뛰어난 성능을 보였으며, 오픈 세트 테스트 이미지에 대한 제로 샷(Zero-Shot) 비디오 생성이 가능합니다.



### Latent Distillation for Continual Object Detection at the Edg (https://arxiv.org/abs/2409.01872)
Comments:
          ECCV workshops, Computational Aspects of Deep Learning (CADL) 2024

- **What's New**: 이 논문에서는 객체 감지(Object Detection)에서의 지속 학습(Continual Learning, CL) 문제를 다루며, 에지 장치(edge devices)에서의 메모리 및 계산 제약을 해결하기 위해 NanoDet이라는 경량 모델을 사용하여 효율적인 업데이트 방법인 Latent Distillation(LD)를 제안합니다.

- **Technical Details**: 제안된 Latent Distillation(LD) 방법은 기존 CL 접근 방식보다 74%의 distillation 파라미터 오버헤드 감소 및 56%의 Floating Points Operations (FLOPs) 감소를 달성하였습니다. 또한 NanoDet는 1.2M의 파라미터로 구성되어 있으며 0.9G FLOPs의 추론을 요구합니다.

- **Performance Highlights**: 실험을 통해 LD 방법이 기존의 방법들보다 더 적은 계산 비용으로 우수한 감지 성능을 유지한다는 것을 밝혔습니다. CLOD 적용에서의 경량 모델의 가능성을 검증하여, 에지 장치에서 실용적인 성능을 demonstrated 하였습니다.



### Real-Time Indoor Object Detection based on hybrid CNN-Transformer Approach (https://arxiv.org/abs/2409.01871)
- **What's New**: 이 연구는 실내 환경에서의 실시간 객체 탐지의 정확도와 속도를 개선하기 위해 새로운 데이터셋과 CNN 기반 탐지 모델을 제안합니다. 이 모델은 내부 장면의 복잡함 속에서 중요한 특징을 구별할 수 있는 주의 메커니즘을 통합하였습니다.

- **Technical Details**: 제안된 CNN 모델은 OpenImages v7에서 파생된 32개의 실내 카테고리를 포함하는 데이터셋을 기반으로 하며, 변형을 통해 효율성을 극대화했습니다. 또한, 하이브리드 아키텍처가 CNN의 강점과 transformer의 공간적 추론 능력을 결합해 실시간 처리에 최적화되어 있습니다.

- **Performance Highlights**: 본 연구는 기존의 최첨단 모델들과 비교해도 정확도와 속도 면에서 경쟁력이 있으며, 실내 환경에서의 객체 탐지 연구와 다양한 애플리케이션의 새로운 가능성을 엽니다.



### Explicit Second-order LiDAR Bundle Adjustment Algorithm Using Mean Squared Group Metric (https://arxiv.org/abs/2409.01856)
- **What's New**: 이번 연구에서는 LiDAR SLAM 시스템의 백엔드에서 고정밀 지도 정제를 달성하기 위한 새로운 BA(번들 조정) 추정기인 RSO-BA를 제안했습니다. 이 연구는 평균 제곱 그룹 메트릭(MSGM)을 도입하여 평면 랜드마크의 측정을 균일하게 처리하도록 개선하였습니다.

- **Technical Details**: 제안된 MSGM은 측정 포인트 수를 고려하여 스케일 해석 가능성을 보장합니다. 이를 통해 강력한 커널 함수와 결합하여 BA 모델을 개선하며, RSO-BA는 분석적 헤시안(Hessian) 매트릭스와 경량 벡터를 사용하여 정확한 솔루션을 보장합니다.

- **Performance Highlights**: 실험 결과, RSO-BA 추정기는 등록 정확성과 강건성 면에서 기존의 암묵적 2차 및 명시적 근사 2차 추정기를 초월하는 성능을 보여주었습니다. 특히 대규모 및 복잡한 비구조 환경에서 뛰어난 성능을 발휘했습니다.



### Towards Generative Class Prompt Learning for Few-shot Visual Recognition (https://arxiv.org/abs/2409.01835)
Comments:
          Accepted at BMVC 2024

- **What's New**: 본 논문에서는 Generative Class Prompt Learning (GCPL)와 Contrastive Multi-class Prompt Learning (CoMPLe)라는 두 가지 새로운 방법을 제안합니다. 이러한 방법들은 시각-언어적 동기화를 개선하는 데 큰 기여를 하며, 기존의 이미지 인식 방법보다 나은 성능을 보여줍니다.

- **Technical Details**: GCPL은 텍스트-이미지 diffusion 모델을 활용하여 learnable class prompts를 사용하여 few-shot exemplars에 조건화함으로써 클래스 임베딩에서 시각-언어적 시너지를 상당히 개선합니다. CoMPLe는 이 기초 위에서 생성 최적화 과정 중 클래스 간 분리를 독려하는 대조 학습(contrastive learning) 요소를 추가합니다.

- **Performance Highlights**: 실험 결과, 제안된 generative class prompt learning 접근 방식이 기존의 방법보다 현저하게 우수한 성능을 발휘하며, few shot 이미지 인식 문제에 대한 더 나은 대안을 제공합니다.



### AstroMAE: Redshift Prediction Using a Masked Autoencoder with a Novel Fine-Tuning Architectur (https://arxiv.org/abs/2409.01825)
Comments:
          This paper has been accepted to 2024 IEEE 20th International Conference on e-Science

- **What's New**: AstroMAE는 천문학적 데이터에 마스크된 자동 인코더를 처음으로 적용하여 레드셔프트 예측 모델을 개발했습니다. 이 연구는 SDSS 이미지에서 사전 훈련된 비전 트랜스포머 인코더를 사용하여 라벨 없이 데이터의 전반적인 패턴을 캡처하며, 이는 기존 방법들보다 더 효율적인 사전 훈련을 가능하게 합니다.

- **Technical Details**: AstroMAE는 비전 트랜스포머(ViT) 아키텍처를 기반으로 하며, 마스크된 자동 인코더(Masked AutoEncoder, MAE) 방법론을 사용하여 사전 훈련을 수행합니다. 이 모델은 레드셔프트 예측에 특화된 구조로 미세 조정(Fine-tuning) 되어 있습니다. 또한, 다양한 CNN 모델과의 성능 비교 실험을 통해 AstroMAE의 우수성을 입증하였습니다.

- **Performance Highlights**: AstroMAE는 여러 비전 트랜스포머 아키텍처 및 CNN 기반 모델과 비교하여 더 높은 성능을 나타냈습니다. 특히, 사전 훈련된 모델과 미세 조정 아키텍처에서 뛰어난 결과를 보였습니다.



### When Does Visual Prompting Outperform Linear Probing for Vision-Language Models? A Likelihood Perspectiv (https://arxiv.org/abs/2409.01821)
- **What's New**: 이 논문에서는 Visual Prompting (VP)과 Linear Probing (LP) 기술의 성능을 비교하기 위해 Log-Likelihood Ratio (LLR) 접근 방식을 제안하고 있습니다. VP는 out-of-distribution (OOD) 작업의 성능을 극적으로 향상시킬 수 있는 최신의 파라미터 효율적인 전이 학습 방법입니다.

- **Technical Details**: 연구에서는 12개의 데이터셋과 5개의 사전 훈련된 모델(ResNet18, ResNext-IG, ViT-B-16, Swin-T, CLIP)을 사용합니다. LLR 점수를 사용하여 VP와 LP의 비교 이점을 분석하고, 리소스 효율적인 시각적 프롬프트 근사를 활용하여 최대 100배의 실행 시간 감소를 달성하며, 91%의 예측 정확도를 기록합니다.

- **Performance Highlights**: VP와 LP의 정확도 차이를 통해 성능 향상을 측정하였으며, 특정 OOD 탐지 기법들과의 비교를 통해 LLR 점수의 효과를 검증하였습니다. 모든 실험은 정의된 지표와 성능 지표를 바탕으로 진행되었으며, VP는 인가된 프롬프트 없이는 0-shot 예측을 가능하게 합니다.



### GeoBEV: Learning Geometric BEV Representation for Multi-view 3D Object Detection (https://arxiv.org/abs/2409.01816)
- **What's New**: 이번 논문에서는 Bird's-Eye-View (BEV) 표현의 기하학적 품질이 간과되고 있음을 지적하고, Radial-Cartesian BEV Sampling (RC-Sampling) 기법을 제안하여 고해상도 밀집 BEV 표현을 효율적으로 생성하는 방법을 보여줍니다. 또한, In-Box Label을 도입하여 전통적인 LiDAR 점에서 생성된 깊이 레이블을 대체하고, Centroid-Aware Inner Loss (CAI Loss)를 통해 물체의 미세한 기하학적 구조를 포착하는 방법을 제안합니다.

- **Technical Details**: RC-Sampling은 고차원 행렬 곱셈을 통해 Radial BEV 특징을 획득하고, bilinear sampling을 사용하여 Cartesian 좌표에서 해당 Radial BEV 특징을 검색하여 BEV 특징을 채웁니다. In-Box Label은 GT 박스 내의 생성된 의사 점을 확인하여 이진 레이블을 생성하고, 이를 통해 물체의 실제 기하학적 구조를 반영할 수 있도록 설계되었습니다. CAI Loss는 물체의 내부 기하학적 구조를 더욱 정교하게 캡처하는 역할을 합니다.

- **Performance Highlights**: nuScenes 데이터셋에 대한 광범위한 실험을 통해 GeoBEV는 기존의 다중 뷰 3D 물체 탐지 방법에 비해 최첨단 성능을 달성하였음을 보여주며, 기하학적 품질과 인식 효율성을 동시에 강조합니다.



### Segmenting Object Affordances: Reproducibility and Sensitivity to Sca (https://arxiv.org/abs/2409.01814)
Comments:
          Paper accepted to Workshop on Assistive Computer Vision and Robotics (ACVR) in European Conference on Computer Vision (ECCV) 2024; 24 pages, 9 figures, 5 tables. Code and trained models are available at this https URL

- **What's New**: 이 논문은 Visual affordance segmentation의 최신 벤치마크를 제공하며, Mask2Former 모델을 재훈련해 여러 테스트 세트에서 최적의 성능을 보여줍니다. 또한, 동일한 구현 체계 하에 기존 방법들을 재구현하여 공정한 비교를 가능하게 했습니다.

- **Technical Details**: 이 연구는 Affordance segmentation을 위한 다양한 모델의 성능을 재훈련하여 평가했습니다. 기존 아키텍처인 Mask2Former를 포함하여, 픽셀 단위에서 각 affordance 클래스를 할당하는 방식의 문제점을 분석했습니다. 또한, latent vectors를 통한 고해상도 계산 방식을 도입하여 특징적인 지역을 예측합니다.

- **Performance Highlights**: Mask2Former는 단일 객체를 대상으로 한 테이블탑 환경과 손으로 가린 객체 환경에서 대부분의 데이터 세트에서 우수한 성능을 나타냈습니다. 연구 결과, 동일한 설정에서 훈련된 모델의 성능이 기존 논문에서 보고된 성능보다 낮았음을 확인했습니다.



### EPRecon: An Efficient Framework for Real-Time Panoptic 3D Reconstruction from Monocular Video (https://arxiv.org/abs/2409.01807)
- **What's New**: EPRecon은 경량 모듈을 도입하여 3D 볼륨 내 장면 깊이 프라이어(scene depth priors)를 직접 추정하고, 이는 비표면 복셀(non-surface voxels)의 대부분을 제거하여 panoptic 3D 재구성 품질을 향상시킵니다.

- **Technical Details**: EPRecon은 다중 시점 이미지 기능을 볼륨에 역투영(back-project)하고, 뷰 간의 기능 유사성을 추출하여 복셀의 점유 확률을 계산합니다. 이 기능을 기반으로 전체 panoptic 재구성 품질을 개선합니다. 복셀 기능과 이미지 기능 두 가지에서 panoptic 특징을 추출하며, 이를 통해 보다 상세하고 포괄적인 panoptic 분할 정보를 얻습니다.

- **Performance Highlights**: EPRecon은 ScanNetV2 데이터셋에서 현재의 최첨단 방법들(SOTA method)보다 panoptic 3D 재구성 품질과 실시간 추론(real-time inference) 모두에서 우수한 성능을 보였습니다.



### UWStereo: A Large Synthetic Dataset for Underwater Stereo Matching (https://arxiv.org/abs/2409.01782)
Comments:
          12pages

- **What's New**: 본 논문에서는 새로운 합성 데이터셋인 UWStereo를 소개합니다. 이 데이터셋은 29,568개의 합성 스테레오 이미지 쌍과 정확한 disparity 주석을 포함하고 있습니다. 이는 복잡한 수중 장면을 위해 설계된 것으로, 기존의 수중 데이터셋보다 규모와 변동성, 주석 및 사실적인 이미지 품질에서 우수합니다.

- **Technical Details**: UWStereo 데이터셋은 Unreal Engine 5를 활용하여 4개의 독특한 수중 장면을 생성했습니다. 이 자료는 다양한 물체로 가득 차 있으며, 카메라 모델, 조명 및 환경 효과에서 추가적인 변화를 유도했습니다. 이 데이터셋은 수중 장면에 적합한 밀도 높은 픽셀 수준 주석을 제공하여, 현재 일반적인 Stereo Matching 모델의 성능을 개선할 수 있도록 합니다.

- **Performance Highlights**: 아홉 개의 최신 스테레오 매칭 알고리즘과의 비교 평가에서, 기존 모델들이 새로운 도메인으로 일반화하는 데 어려움을 겪고 있음을 확인했습니다. 이에 따라, 새로운 전략인 cross view attention enhancement module과 paired masked image reconstruction pretraining이 적용되어 일반화 능력을 향상시키는 데 기여했습니다.



### Dual Advancement of Representation Learning and Clustering for Sparse and Noisy Images (https://arxiv.org/abs/2409.01781)
- **What's New**: 이 논문에서는 Sparse and Noisy Images (SNIs)를 효과적으로 처리하기 위해 Dual Advancement of Representation Learning and Clustering (DARLC)라는 혁신적인 프레임워크를 제안합니다. 이 프레임워크는 masked image modeling에서 파생된 표현을 향상시키기 위해 contrastive learning을 활용하며, 클러스터 할당을 통합하여 end-to-end 접근을 구현합니다.

- **Technical Details**: DARLC는 graph attention network를 이용하여 노이즈가 제거된 이미지를 생성하여 더 신뢰할 수 있는 positive views를 생성합니다. 또한, Student's t mixture model을 사용하여 SNIs의 클러스터링을 더욱 강력하고 유연하게 수행합니다. 데이터 증강 기술로는 GAT 기반 방법을 사용하여 이웃 픽셀의 정보를 통합하여 시각적 패턴을 개선하고, 클러스터링 손실이 CL을 정규화하여 'class collision problem'을 완화합니다.

- **Performance Highlights**: DARLC는 12개의 서로 다른 SNIs 데이터셋에서 수행된 광범위한 실험을 통해 이미지 클러스터링과 이미지 표현 생성에서 기존 최신 기술들(SOTA)보다 우수한 성과를 보여주었습니다. 연구 결과는 DARLC가 생성한 이미지 표현이 클러스터링 성능을 향상시키고, 기능적으로 상호작용하는 유전자 발견과 같은 다른 의미 기반 작업에도 도움이 되는 것을 확인했습니다.



### Gradient events: improved acquisition of visual information in event cameras (https://arxiv.org/abs/2409.01764)
Comments:
          8 pages, 6 figures. This project has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement no 101016734

- **What's New**: 이번 연구에서는 기존의 brightness event를 대체할 수 있는 새로운 종류의 이벤트인 gradient event를 제안합니다. Gradient event는 오실레이팅(oscillating) 광원에 덜 민감하고 더 나은 그레이스케일(frame reconstruction)을 가능하게 합니다.

- **Technical Details**: 제안된 gradient event는 픽셀의 밝기 변화 대신 지역 이미지 기울기를 추정하여 생성되며, 이를 통해 더 나은 시각 정보 획득이 가능합니다. ternary quantization과 위치 의존적인 thresholding 기법을 통해 구현됩니다. 본 연구에서는 Poisson-solver 기반의 방법을 사용하여 gradient events에서 그레이스케일 이미지 복원을 수행했습니다.

- **Performance Highlights**: Gradient event 기반 비디오 복원은 기존의 brightness event 기반 방법들과 비교했을 때, 공공 데이터셋에서 상당한 우수성을 보였습니다. 비디오 복원 과정에서 적은 수의 조정 가능한 매개변수를 사용해 무손실(temporal resolution) 특성을 유지했습니다.



### PRoGS: Progressive Rendering of Gaussian Splats (https://arxiv.org/abs/2409.01761)
- **What's New**: 3D Gaussian Splatting (3DGS)는 3D 장면을 지각적으로 정확하게 표현하는 기술로 주목받고 있습니다. 기존의 렌더링 기법에서는 전체 장면을 메모리에 로드해야 했지만, 이 논문에서는 전체 장면을 로드하지 않고도 가장 중요한 부분을 우선적으로 렌더링하는 혁신적인 방법을 소개합니다.

- **Technical Details**: 이 연구에서는 Gaussian의 최종 장면 기여도를 근사화하여 렌더링 처리에서 포함 순서를 결정하는 방식으로 Progressive Rendering을 진행합니다. 기존의 압축 방법과 결합하여 대역폭 사용을 최적화하고, 사용자가 중요한 스플랫을 중심으로 초기 장면을 빠르게 받을 수 있도록 합니다.

- **Performance Highlights**: 제안된 접근법은 사용된 스플랫 비율에 따른 품질에서 기존 웹 기반 접근법보다 우수한 성능을 보이며, 압축 방법과 결합하여 대역폭 이용을 더욱 줄일 수 있는 가능성을 보여줍니다. 결과적으로, 원거리 호스팅된 3DGS 콘텐츠의 접근 속도가 획기적으로 향상되어 사용자 경험을 개선할 수 있습니다.



### Shuffle Mamba: State Space Models with Random Shuffle for Multi-Modal Image Fusion (https://arxiv.org/abs/2409.01728)
- **What's New**: 본 논문에서는 Multi-modal 이미지 융합을 위한 새로운 Random Shuffle 스캐닝 전략을 제안합니다. 기존 Mamba 기반 기법은 고정된 스캐닝 전략을 사용하여 편향된 정보를 도입할 수 있었습니다. Random Shuffle 기능을 통해 이러한 편향을 제거하고, 정보의 동기화 불변성을 유지하고자 합니다.

- **Technical Details**: Shuffle Mamba Framework에 기반하여, 이 프레임워크는 modality-aware 정보 표현과 cross-modality 정보 상호작용을 공간 및 채널 축을 통해 구현합니다. 또한, Monte-Carlo 평균을 기반으로 한 테스트 방법론을 개발하여 모델의 출력이 예상 결과에 더 가까워지도록 합니다. Random Mamba Block, Random Channel Interactive Mamba Block, Random Modal Interactive Mamba Block와 같은 구성 요소가 포함됩니다.

- **Performance Highlights**: 다양한 multi-modal 이미지 융합 작업에 대한 실험에서, 제안된 방법은 기존 최첨단 기법들보다 우수한 융합 품질을 보여주었습니다. 특히, quantitative 평가 및 시각적 품질 모두에서 탁월한 성능을 달성하였습니다.



### Mahalanobis Distance-based Multi-view Optimal Transport for Multi-view Crowd Localization (https://arxiv.org/abs/2409.01726)
Comments:
          ECCV 2024

- **What's New**: 본 논문에서는 다중 뷰 군중 로컬리제이션(multi-view crowd localization) 작업을 위한 새로운 Mahalanobis 거리 기반의 최적 운송 손실(M-MVOT loss)을 제안합니다. 이 방법은 밀집 지역의 밀도 맵(supervision) 문제를 해결하기 위해 포인트(supervision) 방법을 활용합니다.

- **Technical Details**: M-MVOT 손실은 Mahalanobis 거리를 사용하여 운송 비용을 정의하며, 카메라와의 거리 및 시점 벡터(view-ray direction)에 따라 비용을 조정합니다. 이 손실은 여러 카메라의 정보를 통합하여 최적의 운송 비용을 산출합니다.

- **Performance Highlights**: 실험 결과, 제안한 M-MVOT 손실은 기존의 밀도 맵 기반의 MSE 및 유클리드 거리 기반 최적 운송 손실보다 우수한 군중 로컬리제이션 성능을 보였습니다.



### 4D-CAT: Synthesis of 4D Coronary Artery Trees from Systole and Diasto (https://arxiv.org/abs/2409.01725)
- **What's New**: 본 논문에서는 4D 심장관상동맥 모델을 생성하기 위한 새로운 방법을 제안합니다. 제안된 방법은 비강직(vessel non-rigid) 등록 및 변형 필드(Deformation Field) 예측을 통해 혈관 모양 변화를 시뮬레이션합니다.

- **Technical Details**: 이 연구에서 제안하는 방법은 센터라인(centerline)을 사용하여 혈관을 표현하고, 주변 점들을 바탕으로 변형 필드를 예측합니다. 근본적으로 이 방법은 포인트 클라우드(point cloud)를 두 단계를 통해 처리하고, 샘플링된 심장 주기의 수축기 및 이완기 데이터에서 중간 볼륨을 추론합니다.

- **Performance Highlights**: 실험을 통해 비강직 혈관 포인트 등록 및 4D 관상동맥 모형 생성을 성공적으로 입증하였으며, 이전의 정적 모델들보다 혈관의 동적 변화를 더 효과적으로 나타내는 것을 확인했습니다.



### General OCR Theory: Towards OCR-2.0 via a Unified End-to-end Mod (https://arxiv.org/abs/2409.01704)
- **What's New**: 본 논문은 전통적인 OCR 시스템의 한계를 극복하고, OCR-2.0 시대를 맞이하기 위한 새로운 모델 GOT(General OCR Theory)를 제안합니다.

- **Technical Details**: GOT은 580M 파라미터로 구성된 통합 엔드-투-엔드 모델로, 고효율 인코더와 긴 컨텍스트 길이의 디코더로 이루어져 있습니다. 입력으로는 슬라이스 및 전체 페이지 스타일의 이미지를 지원하며, 출력 결과로는 평문 및 포맷된 결과를 생성할 수 있습니다.

- **Performance Highlights**: GOT는 다양한 OCR 작업에서 우수한 성능을 보이며, 많은 연구자들이 OCR-2.0 연구에 참여하도록 유도할 것으로 기대됩니다.



### On the Vulnerability of Skip Connections to Model Inversion Attacks (https://arxiv.org/abs/2409.01696)
Comments:
          Accepted by ECCV 2024

- **What's New**: 본 논문은 DNN 아키텍처가 모델 역전(Model Inversion, MI) 공격에 미치는 영향을 최초로 연구하며, skip connections(스킵 연결)이 MI 공격을 강화시키고 데이터 프라이버시를 위협한다는 중요한 발견을 하였습니다. 특히 마지막 단계의 skip connections가 MI 공격에 가장 큰 영향을 미친다는 것을 밝혔습니다.

- **Technical Details**: skip connections는 DNN 아키텍처에서 깊은 신경망(Deep Neural Networks) 훈련 중 기울기 소실(vanishing gradient) 문제를 완화하는 데 도움을 주지만, MI 공격에 대한 취약점을 증가시킬 수 있습니다. RepVGG와 같은 기존 기술이 MI 취약성을 완화하지 못함을 분석하며, 새로운 MI 복원력 아키텍처 디자인을 제안합니다. 여기에는 마지막 단계의 skip connection을 제거하는 Removal of Last Stage Skip-Connection(RoLSS)와 같은 접근법이 포함됩니다.

- **Performance Highlights**: 우리의 MI-복원력 아키텍처는 기계 학습 모델의 MI 강인성에서 최신 방어 방법(state-of-the-art, SOTA)을 초과하는 성능을 보여주며, 기존 MI 방어 기술과 상호 보완이 가능합니다. 실험 결과 확장된 실험을 통해 MI 공격에 대한 복원력이 뛰어난 새로운 아키텍처가 입증되었습니다.



### When 3D Partial Points Meets SAM: Tooth Point Cloud Segmentation with Sparse Labels (https://arxiv.org/abs/2409.01691)
Comments:
          To appear at MICCAI24

- **What's New**: 본 논문은 치아 포인트 클라우드 세분화의 새로운 프레임워크인 SAMTooth를 제안합니다. 이는 극도로 희소한 레이블을 보완하기 위해 Segment Anything Model(SAM)의 능력을 활용합니다.

- **Technical Details**: SAMTooth는 두 가지 핵심 방식인 Confidence-aware Prompt Generation(CPG)과 Mask-guided Representation Learning(MRL)을 도입합니다. CPG는 예측된 치아 포인트를 집계하여 SAM에 대한 적절한 프롬프트를 자동 생성하며, MRL은 SAM의 출력을 3D 공간으로 투사하고 3D 특징 학습을 지원합니다.

- **Performance Highlights**: 실험 결과, SAMTooth는 단 0.1%의 주석(치아당 한 점)으로 최신 약한 감독 방법에 비해 큰 성과를 보이며, 최신 완전 감독 방법과도 비교 가능한 성능을 나타냅니다.



### Taming CLIP for Fine-grained and Structured Visual Understanding of Museum Exhibits (https://arxiv.org/abs/2409.01690)
Comments:
          Accepted to ECCV 2024

- **What's New**: CLIP를 향상시키기 위해 박물관 전시물의 세부적이고 구조화된(structured) 시각 이해를 지향하는 새로운 방법론인 MUZE를 제안합니다.

- **Technical Details**: MUZE는 200K 이상의 이미지-테이블 쌍으로 구성된 데이터셋을 기반으로 하며, 변환기 기반 파싱 네트워크(parseNet)를 통해 CLIP의 이미지 임베딩을 테이블 구조로 매핑합니다. 이 과정에서 입력 이미지에 대한 알려진 속성-값 쌍의 컨텍스트를 통합하여 누락된 속성 값을 예측합니다.

- **Performance Highlights**: MUZE는 박물관 전시물에 대한 세부적이고 구조화된 이해에서 정확성을 크게 향상시켰습니다. 새로운 벤치마크에서 유망한 결과를 달성하며, 전체 실험을 통해 제안된 방법의 효과성을 입증했습니다.



### Frequency-Spatial Entanglement Learning for Camouflaged Object Detection (https://arxiv.org/abs/2409.01686)
Comments:
          Accepted at ECCV 2024

- **What's New**: 본 논문에서는 색상에 적응하기 어려운 camouflaged object detection(위장 객체 탐지)의 새로운 접근 방법인 Frequency-Spatial Entanglement Learning(FSEL)을 제안합니다. 이는 주파수 및 공간 도메인 간의 상호 작용을 탐구하여 위장된 객체의 검출 성능을 높입니다.

- **Technical Details**: 제안된 방법은 Entanglement Transformer Block(ETB) 시리즈를 통해 주파수 자기 주의(frequency self-attention)를 활용하여 다양한 주파수 대역 간의 관계를 효과적으로 특징화합니다. 또한 Joint Domain Perception Module과 Dual-domain Reverse Parser를 도입하여 주파수 및 공간 도메인의 기능을 통합합니다.

- **Performance Highlights**: FSEL은 세 가지 널리 사용되는 데이터셋(CAMO, COD10K 및 NC4K)에서 21개의 최신 COD 방법을 초월하는 성능을 보여줍니다. 이 논문에서 제안한 방법은 camouflaged object detection에 있어 중요한 기여를 하고 있습니다.



### Adaptive Explicit Knowledge Transfer for Knowledge Distillation (https://arxiv.org/abs/2409.01679)
Comments:
          19 pages, 5 figures

- **What's New**: 이번 연구에서는 logit 기반 지식 증류(knowledge distillation, KD) 방법의 성능을 향상시키기 위해, teacher 모델의 비대상 클래스에 대한 확률 분포(즉, 'implicit (dark) knowledge')를 student 모델에 효과적으로 전달하는 방법을 제시합니다.

- **Technical Details**: 연구에서는 gradient 분석을 통해 implicit knowledge의 학습 조절이 가능함을 보여주고, explicit knowledge(즉, teacher가 대상 클래스에 대해 가지는 확신)를 함께 학습할 수 있도록 새로운 loss 함수를 제안합니다. 또한, 효과적인 증류 및 클래스 간 관계 모델링을 위해 분류(classification)와 증류(distillation) 작업을 분리할 것을 제안합니다.

- **Performance Highlights**: 제안된 adaptive explicit knowledge transfer (AEKT) 방법은 CIFAR-100 및 ImageNet 데이터셋에서 기존의 최첨단 KD 방법들에 비해 개선된 성능을 달성했습니다.



### Enhancing Fine-Grained Visual Recognition in the Low-Data Regime Through Feature Magnitude Regularization (https://arxiv.org/abs/2409.01672)
- **What's New**: 작지만 강력한 데이터로 훈련된 Fine-Grained Visual Recognition (FGVR) 모델의 성능을 크게 향상시키는 간단한 규제 기술인 Feature Magnitude Regularization (FMR)을 도입했습니다. 이 기법은 특성의 크기 분포를 평등하게 만드는 정규화로, 사전 훈련된 모델에서의 편향을 제거하는 것을 목표로 합니다.

- **Technical Details**: FGVR은 이미지 내의 미세한 차이를 감지하여 대분류를 세부 분류하는 문제로, 데이터가 제한적일 때 품질 높은 특성 표현을 활용하는 것이 중요합니다. 본 연구에서는 정규화의 강도를 다이나믹하게 조정할 수 있는 역량 메커니즘을 개발하여, 학습 과정 중 특성 크기 분포의 균일성을 높이고, 잠재적 편향을 줄이는 방법을 제시합니다.

- **Performance Highlights**: 실험 결과, 제안한 FMR 메커니즘이 기존의 파인 튜닝 기법보다 향상된 성능을 나타내었으며, 제한된 데이터셋에서의 FGVR 성능을 효과적으로 개선했습니다.



### VProChart: Answering Chart Question through Visual Perception Alignment Agent and Programmatic Solution Reasoning (https://arxiv.org/abs/2409.01667)
- **What's New**: 이 논문은 Chart Question Answering (CQA) 분야의 새로운 프레임워크인 VProChart를 소개합니다. VProChart는 Visual Perception Alignment Agent (VPAgent)와 Programmatic Solution Reasoning 접근 방식을 통합하여 복잡한 차트 설명과 논리적 추론 문제를 해결하는 방법입니다.

- **Technical Details**: VProChart는 두 가지 주요 모듈로 구성됩니다: 1) VPAgent는 차트 요소를 인간의 시각적 인식 원리에 맞춰 정렬하고 모델링하여 차트 맥락을 이해합니다. 2) Programmatic Solution Reasoning 접근 방식은 자연어 질문을 구조적인 해결 프로그램으로 변환하여, 정밀한 수치 및 논리적 추론을 가능하게 합니다. 이 시스템은 ChartQA 및 PlotQA와 같은 벤치마크 데이터셋에서 실험을 통해 효과성을 입증하였습니다.

- **Performance Highlights**: VProChart는 기존의 방법들과 비교하여 성능이 크게 향상된 것으로 나타났습니다. 이 연구는 CQA 작업에서 비주얼 요소 간의 관계를 명시적으로 모델링하고, 질문에 기반한 추론을 통해 정확한 답변을 도출하는 와중에, 다양한 데이터셋에서 두드러진 결과를 보여주었습니다.



### Efficiently Expanding Receptive Fields: Local Split Attention and Parallel Aggregation for Enhanced Large-scale Point Cloud Semantic Segmentation (https://arxiv.org/abs/2409.01662)
- **What's New**: 본 연구에서는 Local Split Attention Pooling (LSAP) 메커니즘을 제안하여 대규모 3D 포인트 클라우드 세그멘테이션의 수용 필드(receptive field)를 효과적으로 확장하고, 넓은 맥락적 정보를 확보하려고 하였습니다. 이 메커니즘은 연산의 효율성을 높이면서 과적합의 위험을 줄이는 데 중점을 두었습니다.

- **Technical Details**: LSAP 메커니즘은 일련의 로컬 분할 작업(local split operations)을 통해 수용 필드를 확장하는 기법으로, 연산 부하를 최적화하여 attention-pooling 레이어의 처리 과정을 간소화합니다. 또한, Parallel Aggregation Enhancement (PAE) 모듈을 도입하여 2D 및 3D 이웃 정보를 활용한 병렬 처리를 가능하게 하여 네트워크 내에서의 맥락적 표현을 더욱 강화합니다.

- **Performance Highlights**: LSNet은 S3DIS, Toronto3D, SensatUrban 등 세 가지 벤치마크 데이터셋에서 기존 최첨단 semantic segmentation 네트워크와 비교하여 우수한 성능을 입증하였으며, mean intersection over union (mIoU)에서 최대 11%의 향상을 보였습니다. 또한, 유사한 크기의 수용 필드를 사용하는 경우에 비해 약 38.8%의 속도 개선 효과도 달성하였습니다.



### Unveiling Advanced Frequency Disentanglement Paradigm for Low-Light Image Enhancemen (https://arxiv.org/abs/2409.01641)
Comments:
          Accepted to ECCV 2024, Github \url{this https URL}

- **What's New**: 이 논문에서는 이전의 저조도 이미지 향상(LLIE) 접근법이 복잡한 네트워크 개발에 집중된 반면, 고급 주파수 분리 패러다임을 통해 기존의 LLIE 방법들을 효율적으로 향상시킬 수 있음을 보여줍니다.

- **Technical Details**: 우리는 이미지 라플라스 분해(image Laplace decomposition) 방식을 활용하여 새로운 저주파수 일관성 방법을 제안하였으며, 이를 통해 다양한 모델(CNNs, Transformers, flow-based 및 diffusion models)과 통합할 수 있습니다. 주요 기법으로는 저주파수 조정을 위한 Adaptive Convolutional Composition Aggregation(ACCA) 모듈과 고주파수 향상을 위한 Laplace Decoupled Restoration Model(LDRM)이 포함됩니다.

- **Performance Highlights**: 저희 방법은 PSNR에서 최대 7.68dB의 향상을 달성했으며, 총 88K의 추가 파라미터로 효율성을 유지하는 동시에 다섯 개의 주요 벤치마크에서 눈에 띄는 개선을 보였습니다.



### Dynamic Guidance Adversarial Distillation with Enhanced Teacher Knowledg (https://arxiv.org/abs/2409.01627)
- **What's New**: 본 연구에서는 Dynamic Guidance Adversarial Distillation (DGAD) 프레임워크를 도입하여, 적대적 공격에 대한 학생 모델의 저항력을 향상시키고, 교사 모델의 잘못된 예측을 정정하는 새로운 접근 방식을 제시합니다.

- **Technical Details**: DGAD는 Misclassification-Aware Partitioning (MAP)와 Error-corrective Label Swapping (ELS) 방법을 사용하여 분류 오류를 정정하며, Predictive Consistency Regularization (PCR)을 통해 학생 모델의 일관성을 강화합니다.

- **Performance Highlights**: CIFAR10, CIFAR100, Tiny ImageNet 데이터셋을 활용한 실험 결과, DGAD는 기존 모델들에 비해 정확도와 적대적 방어 능력을 현저히 향상시켰습니다.



### Decompose the model: Mechanistic interpretability in image models with Generalized Integrated Gradients (GIG) (https://arxiv.org/abs/2409.01610)
- **What's New**: 이번 연구에서는 eXplainable AI (XAI) 분야에서 이미지 모델의 메커니즘 해석 가능성(mechanistic interpretability)을 향상시키기 위한 새로운 접근법을 제안합니다. 기존의 클래스별 해석(class-specific interpretations)에서 벗어나 입력(input)부터 최종 출력(output)까지의 모든 중간 레이어를 체계적으로 추적(tracing)하는 방법을 개발하였습니다.

- **Technical Details**: 우리는 모델 임베딩을 해석 가능한 개념 벡터(Concept Vectors)로 분해하기 위해 포인트 단위 특성 벡터(Pointwise Feature Vectors, PFVs)와 효과적인 수용 영역(Effective Receptive Fields, ERFs)을 활용합니다. 일반화된 적분 그래디언트(Generalized Integrated Gradients, GIG)를 통해 개념 벡터 간의 상관성을 계산하고, 데이터셋 전체에 대한 포괄적인 모델 행동 분석을 수행합니다. 이 방법을 통해 ResNet50 모델의 구조를 심층적으로 분석합니다.

- **Performance Highlights**: 제안된 방법은 정성적(qualitative) 및 정량적(quantitative) 평가를 통해 검증되었으며, 이미지 모델의 의미적 중요성(semantic significance)에 대한 깊은 이해를 제공합니다. 이를 통해 우리는 모델의 기능적 메커니즘을 종합적으로 설명할 수 있으며, 기존의 클래스 기반 접근법의 한계를 넘어서 전체 데이터셋에 대한 해석 가능성을 확장합니다.



### EDCSSM: Edge Detection with Convolutional State Space Mod (https://arxiv.org/abs/2409.01609)
- **What's New**: 이 논문에서는 이미지에서 가장자리를 감지하는 새로운 알고리즘을 소개합니다. 이 알고리즘은 기존의 다층 합성곱 및 풀링 아키텍처에서 발생하는 특징 손실 문제를 해결하고 있으며, 작은 객체의 가장자리를 효과적으로 감지할 수 있습니다.

- **Technical Details**: 제안된 알고리즘은 이중 입력 채널을 통해 이미지의 상태 공간 변수를 획득하며, 최소한의 다운샘플링 과정으로 실시간 학습 및 이미지 텍스트 기억화를 가능하게 합니다. 또한, 'Wind Erosion'이라는 후처리 알고리즘을 사용하여 이진 가장자리 맵에서 거짓 가장자리를 제거합니다. 알고리즘의 효율성을 높이기 위해 병렬 컴퓨팅 회로를 설계하였습니다.

- **Performance Highlights**: 실험 결과에 따르면, 제안된 알고리즘은 정밀한 얇은 가장자리 위치를 찾는데 성공했으며 다양한 이미지 유형에서 노이즈 억제 기능을 보여줍니다. 병렬 컴퓨팅 회로 덕분에 이 알고리즘은 5K 이미지에서 30 FPS를 초과하는 처리 속도를 달성합니다.



### DAPONet: A Dual Attention and Partially Overparameterized Network for Real-Time Road Damage Detection (https://arxiv.org/abs/2409.01604)
- **What's New**: DAPONet은 글로벌 및 로컬 정보를 통합한 이중 주의 메커니즘과 다중 스케일 부분 과파라미터화 모듈 및 효율적인 다운샘플링 모듈을 포함한 새로운 접근 방식을 통해 실시간 도로 손상 감지를 개선합니다.

- **Technical Details**: DAPONet은 다음과 같은 주요 모듈을 포함합니다: Global Localization Context Attention (GLCA), Cross Stage Partial Depthwise Over-parameterized Attention (CPDA), Mix Convolutional Downsampling (MCD). 이러한 모듈들은 복잡한 배경의 다중 스케일 대상을 효과적으로 탐지하도록 설계되었습니다.

- **Performance Highlights**: DAPONet은 SVRDD 데이터셋에서 mAP50 70.1%를 달성하며 YOLOv10n보다 10.4% 향상된 성능을 보였으며, MS COCO2017 검증 세트에서는 mAP50 48.3%와 mAP50-95 33.4%로 뛰어난 성과를 나타냈습니다.



### A Time-Intensity Aware Pipeline for Generating Late-Stage Breast DCE-MRI using Generative Adversarial Models (https://arxiv.org/abs/2409.01596)
- **What's New**: 이 논문은 초기 대비 종합적인 파이프라인을 제안하여 조기 대비 정확한 장기(늦은) 대조증강 유방 자기공명영상(MRI)을 생성하는 방법을 다룹니다. 이 전략은 대조제 패턴을 유지하면서 합성된 이미지의 시각적 속성을 보존하는데 초점을 맞추고 있습니다.

- **Technical Details**: 논문에서는 Time-Intensity (TI) 곡선에 기반한 새로운 손실 함수(TI-Loss)를 제안하여 픽셀에 대한 주의 집중(pixel-attention) 기반의 생성 모델을 최적화합니다. 또한, 기존의 표준화 방법과는 달리 여러 시간대의 이미지 시퀀스에서 대조 증강 패턴을 유지하는 새로운 정규화 전략(TI-norm)을 개발했습니다.

- **Performance Highlights**: 실험 결과는 제안된 전략이 대조 증강된 지역의 진단 품질을 유의미하게 향상시킨 이미지를 생성하면서, 전체 이미지의 공간적 특징을 잘 유지하고 있음을 보여줍니다. 이는 임상 시나리오에서 심층 학습을 통해 생성된 합성 Late Enhanced 이미지를 사용할 가능성을 시사합니다.



### DiVE: DiT-based Video Generation with Enhanced Contro (https://arxiv.org/abs/2409.01595)
- **What's New**: 본 논문에서는 자율 주행 시나리오에서 고해상도, 시간적으로 일관된 비디오를 생성하는 새로운 DiT(분산 변환기) 기반 프레임워크를 제안합니다. 이는 주어진 bird's-eye view 레이아웃 제어를 정교하게 일치시키는 것을 목표로 하며, 다중 뷰 비디오 생성 가능성을 탐구합니다.

- **Technical Details**: 제안된 프레임워크는 파라미터 없는 공간적인 view-inflated attention 메커니즘을 활용하여 cross-view 일관성을 보장합니다. 여기에는 joint cross-attention 모듈과 ControlNet-Transformer가 통합되어 있습니다. 모델은 OpenSora 구조를 기반으로 하며, 3D 패치 임베딩을 통해 시공간 정보를 모델링합니다. 또한, classifier-free guidance 기법을 도입하여 제어와 비디오 품질을 향상시킵니다.

- **Performance Highlights**: nuScenes 데이터셋에서 정 qualitatively 비교 실험을 통해, 제안된 방법이 어려운 corner case에서도 긴 시간 동안 제어 가능하고 매우 일관된 비디오를 생성하는 데 효과적임을 입증하였습니다.



### Dynamic Motion Synthesis: Masked Audio-Text Conditioned Spatio-Temporal Transformers (https://arxiv.org/abs/2409.01591)
- **What's New**: 이번 연구는 텍스트와 오디오 입력을 기반으로 여러 모달리티를 동시에 조건으로 하여 전체 신체의 동작 시퀀스를 생성할 수 있는 새로운 동작 생성 프레임워크를 제안합니다. 이 프레임워크는 Vector Quantized Variational Autoencoders (VQVAEs)를 활용하여 동작을 이산화하고, Bidirectional Masked Language Modeling (MLM) 전략을 사용하여 효율적인 토큰 예측을 수행합니다.

- **Technical Details**: 이 연구는 VQVAEs를 사용하여 동작을 토큰으로 이산화하고, Bidirectional Masked Language Modeling (MLM) 전략을 통해 양방향에서 토큰 예측을 실행합니다. 각 Transformer 레이어에 공간적 주의 메커니즘을 통합하여 여러 개의 코드북 간의 공간적 관계를 강화합니다. 또한 텍스트 일관성을 보장하기 위한 텍스트-모션 정렬 모델과 함께 사전 비판(token critic) 메커니즘을 도입하여 시뮬레이션의 일관성을 유지합니다.

- **Performance Highlights**: 제안된 프레임워크는 일반적으로 12-24 스텝의 코드북을 통해 동작을 생성하며, 이는 기존의 오토리그레시브 변환기 및 확산 모델보다 수백 스텝을 대폭 줄인 것입니다. 이 모델은 또한 다양한 형태의 조건을 적용할 수 있는 다중 조건화 방법으로 텍스트 및 오디오 입력에서 동작 생성을 수행할 수 있는 능력을 갖추고 있습니다.



### EvoChart: A Benchmark and a Self-Training Approach Towards Real-World Chart Understanding (https://arxiv.org/abs/2409.01577)
- **What's New**: 이 논문에서는 VLMs(Visual Language Models)의 차트 이해 능력을 향상시키기 위한 새로운 자기 학습(self-training) 방법인 EvoChart를 소개합니다. 또한 실제 세계에서의 모델 차트 이해 능력을 측정하기 위한 새로운 벤치마크인 EvoChart-QA를 제안합니다.

- **Technical Details**: EvoChart는 고품질의 차트 데이터 세트를 생성하기 위한 다단계 자기 학습 접근 방식을 채택하여 차트 이해 모델을 발전시킵니다. EvoChart-QA는 140개의 웹사이트에서 수집한 650개의 고유한 차트를 포함하고, 차트 이해에 중점을 둔 1250개의 전문가가 선별한 질문을 특징으로 합니다.

- **Performance Highlights**: 실험 결과, EvoChart 방법은 54.2%의 정확도를 달성하며, 이는 기존의 공개 및 독점 VLM들에 비해 상당한 성과를 보입니다. GPT-4o와 같은 최상의 독점 모델조차 EvoChart-QA에서 49.8%의 정확도에 그칩니다.



### Improving Apple Object Detection with Occlusion-Enhanced Distillation (https://arxiv.org/abs/2409.01573)
- **What's New**: 이 논문에서는 자연 환경에서 자주 겪는 occlusion(가림현상)이 있는 사과 탐지의 어려움을 해결하기 위해 'Occlusion-Enhanced Distillation'(OED)라는 새로운 기술을 소개합니다. 이 방법은 가림 정보를 활용하여 occluded datasets(가림 데이터셋)의 의미적으로 정렬된 특징 학습을 정규화하고, Exponential Moving Average (EMA)를 통해 학습의 안정성을 높입니다.

- **Technical Details**: OED는 Grounding DINO와 SAM 방법을 활용하여 자연스러운 성장 상태를 반영하는 가림 예제를 만드는 occlusion-enhanced dataset(가림 강화 데이터셋)을 설계합니다. 다중 스케일 지식 증류 전략을 채택하여, 학생 네트워크가 가림 요소가 많아진 이미지를 입력으로 사용하고, 교사 네트워크는 자연적인 가림이 없는 이미지를 사용하여 학습하도록 합니다. 이러한 구조를 통해 학생 네트워크는 서로 다른 스케일 간의 의미적 및 지역적 특징 정렬을 통해 효과적으로 학습합니다.

- **Performance Highlights**: 이 방법은 광범위한 비교 실험을 통해 현재의 최첨단 기술에 비해 현저하게 우수한 성능을 보여줍니다. 특히 extreme(극단적인) 또는 irregular(불규칙한) 데이터의 영향을 최소화하는 EMA 전략을 도입하여 학생 네트워크의 훈련 안정성을 증가시킵니다.



### LSSF-Net: Lightweight Segmentation with Self-Awareness, Spatial Attention, and Focal Modulation (https://arxiv.org/abs/2409.01572)
- **What's New**: 이 연구에서는 모바일 플랫폼에서 피부 병변(segmentation of skin lesions) 분할을 위한 경량 네트워크 구조인 LSSF-Net을 제안합니다. 이 모델은 피부 병변의 복잡한 특성과 불분명한 경계를 효과적으로 캡처하여 최적의 성능을 구현합니다.

- **Technical Details**: LSSF-Net은 인코더-디코더 아키텍처로 구성되어 있으며, conformer-based focal modulation attention, self-aware local and global spatial attention, split channel shuffle 등을 포함합니다. 이 네트워크는 0.8백만 개의 학습 가능한 파라미터를 가지며, 적은 연산 자원으로도 높은 정확도를 달성합니다.

- **Performance Highlights**: LSSF-Net은 ISIC 2016, ISIC 2017, ISIC 2018 및 PH2 등의 네 가지 벤치마크 데이터 세트에서 평가되어 높은 Jaccard index를 기록하며, 기존의 최첨단 성능을 상회하는 결과를 보여줍니다.



### CT-SDM: A Sampling Diffusion Model for Sparse-View CT Reconstruction across All Sampling Rates (https://arxiv.org/abs/2409.01571)
- **What's New**: 이 연구에서는 모든 샘플링 레이트에 대해 높은 성능의 희소 뷰 CT(Sparse View Computed Tomography) 재구성을 달성하는 적응형 재구성 방법을 제안합니다. 이 접근 방식은 단일 모델로 모든 샘플링 레이트에 맞게 조정되며, 진단 도구로서의 CT 이미징의 유연성을 보장합니다.

- **Technical Details**: 제안된 방법은 샘플링 확산 모델을 사용하여 산출 물질의 프로젝션 과정을 시뮬레이션합니다. 새로운 이미지 저하 연산자를 설계하여, 고도로 언샘플링된 측정값에 프로젝션 뷰를 점진적으로 추가하는 방식으로 전체 뷰의 시노그램을 일반화합니다. 훈련 과정에서 그룹화된 랜덤 샘플링 전략을 도입하여 샘플링 레이트와 프로젝션 뷰를 동적으로 조정합니다.

- **Performance Highlights**: 실험 결과, 본 접근법이 다양한 샘플링 레이트에서 고품질 이미지를 재구성하는 데 효과적이고 강건함을 입증하였습니다. 제안된 모델은 하나의 훈련된 모델로 다양한 샘플링 레이트에서 잘 작동하여 실제 임상 환경에서의 실용성을 높입니다.



### ReSpike: Residual Frames-based Hybrid Spiking Neural Networks for Efficient Action Recognition (https://arxiv.org/abs/2409.01564)
- **What's New**: 이번 논문에서는 ReSpike라는 하이브리드 프레임워크를 제안합니다. 이 프레임워크는 인공 신경망(Artificial Neural Networks, ANNs)과 스파이킹 신경망(Spiking Neural Networks, SNNs)의 강점을 결합하여 높은 정확도와 낮은 에너지 비용으로 행동 인식(action recognition) 작업을 수행할 수 있도록 합니다.

- **Technical Details**: ReSpike는 영상 클립을 공간(Spatial)과 시간(Temporal) 구성 요소로 분해하여, RGB 이미지(Key Frames)와 이벤트와 같은 Residual Frames으로 구성됩니다. ANN은 공간 정보를 학습하며 SNN은 시간 정보를 학습합니다. 또한, 효과적인 특징 융합을 위한 다중 스케일 교차 주의 메커니즘을 제안합니다.

- **Performance Highlights**: ReSpike는 HMDB-51, UCF-101, Kinetics-400 데이터셋에서 기존 SNN 기반 모델에 비해 30% 이상의 정확도 향상을 보였으며, 기존 ANN 접근법과 유사한 성능을 달성하면서도 에너지 소비는 최대 6.8배 줄였습니다.



### Blocks as Probes: Dissecting Categorization Ability of Large Multimodal Models (https://arxiv.org/abs/2409.01560)
Comments:
          39 pages, 28 figures, 4 tables. Accepted at The 35th British Machine Vision Conference (BMVC 2024). Project page at this https URL

- **What's New**: 이 논문은 시각 AI 모델의 범주화 성능을 측정하기 위한 새로운 벤치마크인 ComBo를 제안합니다. ComBo는 범주 학습(category learning)부터 적용(category use)까지의 전 과정을 평가할 수 있는 분리된 프레임워크를 제공합니다.

- **Technical Details**: 주요 기술적인 내용은 LMM(large multimodal models)의 범주화 능력을 종합적으로 평가하는 새로운 평가 방식과 이를 위한 데이터셋인 Composite Blocks(ComBo)입니다. ComBo는 본 연구에서 이전 모델들이 겪었던 데이터 누출 문제를 피하며, 인지 과정을 분석하여 범주화 능력을 상세히 평가합니다.

- **Performance Highlights**: 실험 결과 LLMs는 새로운 범주를 학습하는 일반화 능력에서는 acceptable한 성과를 보였지만, 인간에 비해 공간적 관계의 세밀한 인식 및 추상적 범주 이해에서는 여전히 격차가 존재함을 보여주었습니다. 이를 통해 AI의 해석 가능성과 일반화 능력 개선에 기여할 수 있는 통찰을 제시합니다.



### TASL-Net: Tri-Attention Selective Learning Network for Intelligent Diagnosis of Bimodal Ultrasound Video (https://arxiv.org/abs/2409.01557)
- **What's New**: 이번 논문은 bimodal (그레이스케일 및 조영증강) 초음파 비디오의 지능형 진단을 개선하기 위한 새로운 딥 러닝 네트워크인 TASL-Net (Tri-Attention Selective Learning Network)을 제안합니다. TASL-Net은 의료 전문가가 사용하는 세 가지 진단 주의(Temporal, Spatial, Bimodal)를 통합하여 비디오 분석의 효율성을 높이도록 설계되었습니다.

- **Technical Details**: TASL-Net은 비디오 선택을 위한 시간 강도 곡선(time-intensity curve) 기반의 비디오 선택기, 조영증강 비디오 분석을 위한 공간 주의의 통합, 그리고 컨볼루션(convolution) 및 트랜스포머(transformer)를 결합한 상호 인코딩 전략(mutual encoding strategy)을 포함합니다. 이를 통해 TASL-Net은 그레이스케일 비디오의 구조적 특징과 조영증강 비디오의 관류(perfusion) 변화를 동시에 인식할 수 있습니다.

- **Performance Highlights**: TASL-Net은 폐, 유방 및 간의 데이터셋에서 총 791개의 사례에 대해 실험을 수행하였으며, 수동 개입 없이도 우수한 성능을 입증하였습니다. 이 방법은 여러 암의 지능형 진단을 가능하게 하여 의료 전문가의 부담을 덜어주고 관찰자 간 변동성을 줄이는 데 기여할 것으로 기대됩니다.



### EA-RAS: Towards Efficient and Accurate End-to-End Reconstruction of Anatomical Skeleton (https://arxiv.org/abs/2409.01555)
Comments:
          13 pages,15 figures

- **What's New**: EA-RAS는 단일 RGB 이미지를 사용하여 실시간으로 해부학적으로 정확한 인간 골격을 추정할 수 있는 경량/단일 단계 솔루션을 제안합니다. 이는 현재 사용 중인 기존의 골격 모델보다 향상된 기능을 가지고 있습니다.

- **Technical Details**: EA-RAS는 해부학적 스켈레톤 추정의 효율성을 높이기 위해 진화된 최적화 과정과 단계적 훈련 전략을 결합하여, 소량의 데이터를 입력받아 기초 중량을 신속하게 확보할 수 있도록 합니다. 또한, 내외부 스킨 정보를 통합하여 결과의 정확성을 더욱 강화합니다.

- **Performance Highlights**: EA-RAS는 기존 방법보다 800배 빠르며, 후처리 최적화 전략을 통해 재구성 정확도가 50% 이상 향상됩니다. 또한, 실시간 처리 요구 조건을 충족하는 속도 증가를 이룩했습니다.



### Purification-Agnostic Proxy Learning for Agentic Copyright Watermarking against Adversarial Evidence Forgery (https://arxiv.org/abs/2409.01541)
- **What's New**: 이 논문은 AI 모델 보호의 중요성을 강조하며, 특히 모델 워터마킹 기법을 통해 지적 재산권을 보장하는 방법을 제시합니다. 새로운 블랙박스 워터마킹 프로토콜과 적대적 공격에 대한 방어 메커니즘을 도입하였습니다.

- **Technical Details**: 블랙박스 워터마킹 프로토콜에는 해시 기술을 기반으로 한 자가 인증 방식이 포함되어 있습니다. 이 방식은 사전 등록이 필요 없으며, 모델의 파라미터를 직접적인 접근 없이도 연관된 신뢰성을 제공합니다. 또한, 적대적 공격으로 인한 증거 위변조를 방어하기 위한 정화 메커니즘이 제안되었습니다.

- **Performance Highlights**: 실험 결과는 워터마크가 포함된 모델들이 보안성, 신뢰성, 그리고 성능이 향상됨을 보여줍니다. 적대적 공격에 대한 방어 및 모델의 전반적인 성능 개선이 입증되었습니다.



### Long-Range Biometric Identification in Real World Scenarios: A Comprehensive Evaluation Framework Based on Missions (https://arxiv.org/abs/2409.01540)
- **What's New**: 이 논문에서는 다양한 거리와 고도를 고려한 생체 인식 시스템의 연구 솔루션을 평가하고 있습니다. 기존의 문제를 해결하기 위해 얼굴 인식뿐만 아니라 신체의 특징을 결합하여 장거리 인식의 정확성을 높이는 방법을 제안합니다.

- **Technical Details**: 저자는 UAV(무인 항공기)나 건물에 장착된 카메라를 통한 긴 거리 생체 인식의 이점을 논의하며, 장거리, 각도, 해상도와 같은 다양한 환경 조건에 적합한 시스템 구축을 목표로 합니다. 이를 위해 생체 인식 시스템의 성능을 평가하기 위한 데이터 수집 및 준비 방법론을 제시합니다.

- **Performance Highlights**: 초기 결과는 전체 신체 인식에서 유망한 진행 상황을 보여줍니다. 이 연구는 안면 인식 외에도 다양한 생체 신호를 통해 신뢰할 수 있는 식별을 제공하는 시스템 개발의 가능성을 탐구하고 있습니다.



### Think Twice Before Recognizing: Large Multimodal Models for General Fine-grained Traffic Sign Recognition (https://arxiv.org/abs/2409.01534)
- **What's New**: 새로운 전략 'think twice before recognizing'을 제안하여 정교한 교통 표지 인식(TSR)을 개선합니다. 이 방법은 대규모 다중 모달 모델(LMM)의 다차원 사고 능력을 활용하여 복잡한 도로 조건에서 교통 표지를 효과적으로 인식합니다.

- **Technical Details**: 제안된 기법은 세 가지 유형의 설명(컨텍스트, 특성, 차별 설명)을 통해 LMM의 다중 사고 과정을 설계합니다. 첫째, 컨텍스트 설명은 교차로, 차량, 보행자 등 중요한 정보가 포함되어 있으며, 최적화된 중심 좌표 프롬프트를 사용하여 교통 표지를 원본 도로 이미지에서 정확하게 위치를 찾습니다. 둘째, 특성 설명은 템플릿 교통 표지의 몇 장의 샘플을 사용한 컨텍스트 학습을 기반으로 하여 교차 도메인 차이를 줄이고 세밀한 인식 능력을 향상시킵니다. 셋째, 차별 설명을 통해 유사한 교통 표지 간의 미세한 차이를 강조합니다.

- **Performance Highlights**: 세 가지 벤치마크 데이터셋과 두 개의 실제 데이터셋에서 광범위한 실험을 수행하였고, 제안된 방법은 모든 다섯 개 데이터셋에서 최첨단 TSR 결과를 달성하였습니다.



### Lagrangian Motion Fields for Long-term Motion Generation (https://arxiv.org/abs/2409.01522)
Comments:
          13 pages, 9 figures

- **What's New**: 새로운 연구에서 Lagrangian Motion Fields를 도입하여 장기 동작 생성을 위한 신개념을 제안합니다. 이 방법은 각 관절을 Lagrangian 입자로 취급하여 짧은 시간 간격 동안 균일한 속도로 처리하며, "수퍼모션"이라는 압축된 동작 표현으로 통합합니다.

- **Technical Details**: Lagrangian Motion Fields는 동작 생성 네트워크에 통합할 수 있는 두 단계 생성 파이프라인을 제공합니다. 첫 번째 단계에서는 수퍼모션 시퀀스를 생성하고, 두 번째 단계에서 이를 풀 해상도 동작 시퀀스로 압축 해제합니다. 이 방법은 컴퓨테이셔널 오버헤드를 최소화하며 다양한 동작 생성을 지원합니다.

- **Performance Highlights**: Lagrangian Motion Fields는 음악-춤 생성 및 텍스트-동작 생성에서 기존 방법보다 우수한 성능을 보여주며, 긴 동작 시퀀스 생성에서의 효율성, 생성 품질, 다양성을 높입니다. 이 방법은 무한 동작 루프 및 세부 제어 동작 생성과 같은 다양한 응용 프로그램에 활용 가능성이 큽니다.



### From Data to Insights: A Covariate Analysis of the IARPA BRIAR Dataset for Multimodal Biometric Recognition Algorithms at Altitude and Rang (https://arxiv.org/abs/2409.01514)
- **What's New**: 이 논문은 IARPA BRIAR 데이터셋에서 UAV 플랫폼, 고도 위치 및 최대 1000미터 거리에서의 전체 신체 생체 인식(performance in fused whole body biometrics) 성능에 대한 공변량(covariate) 효과를 조사합니다.

- **Technical Details**: 이 연구는 외부 비디오와 내부 이미지 및 제어된 보행 기록(gait recordings)을 비교하고, 예측된 거부율(false accept rates, FAR)과 관련된 정규화된 원시 융합 점수(normalized raw fusion scores)를 제시합니다. 또한 고도 및 거리에서의 정확도에 가장 큰 영향을 미치는 공변량을 분석하기 위해 선형 모델(linear model)을 개발하고, 온도(temperature), 바람 속도(wind speed), 태양 복사(solar loading), 난류(turbulence)와 같은 날씨 요소들을 조사합니다.

- **Performance Highlights**: 해상도(resolution) 및 카메라 거리(camera distance)가 정확도를 가장 잘 예측할 수 있으며, 이 결과들은 장거리 및 UAV 생체 인식 분야에서의 미래 연구 및 개발 노력을 안내하고 국방 및 기타 중요 분야에서 더 신뢰할 수 있고 강력한 시스템 구축에 기여할 수 있습니다.



### Less is more: concatenating videos for Sign Language Translation from a small set of signs (https://arxiv.org/abs/2409.01506)
Comments:
          SIBGRAPI 2024

- **What's New**: 이번 연구에서는 브라질 수화(Libras)와 포르투갈어 번역 모델 훈련을 위한 라벨링된 데이터의 부족 문제를 해결하기 위해, 단기 클립을 연결하여 수화 콘텐츠를 생성하는 방법을 제안합니다. 이를 통해 데이터 수집 및 주석 작업의 비용을 크게 줄일 수 있습니다.

- **Technical Details**: 연구에서는 V-LIBRASIL 데이터셋을 활용하여 1,364개의 수화에 대한 4,089개의 수화 비디오를 바탕으로, 약 170K, 300K 및 500K 비디오를 포함한 다양한 실험을 수행합니다. 이들은 각각의 수화 문장과 그에 해당하는 Libras 번역으로 구성되어 있습니다. 또한, 3D 합성곱 신경망인 i3D를 활용하여 비디오 특성을 추출하고, 다양한 데이터 증강 기법을 적용해 훈련 과정 중 비디오의 다양성을 높였습니다.

- **Performance Highlights**: 실험 결과, 연구에서 생성한 데이터셋을 기반으로 학습한 모델이 BLEU-4 및 METEOR 점수에서 각각 9.2% 및 26.2%를 기록하며 유의미한 성과를 보여주었습니다. 이 방법은 미래 연구 방향에 대한 분명한 이정표를 제공하고 있습니다.



### AMG: Avatar Motion Guided Video Generation (https://arxiv.org/abs/2409.01502)
Comments:
          The project page is at this https URL

- **What's New**: AMG라는 새로운 방법을 제안하여 2D의 현실감과 3D의 제어 가능성을 결합하였습니다. 이 방법은 3D 아바타의 제어된 렌더링에 비디오 확산 모델을 조건으로 적용합니다.

- **Technical Details**: AMG는 3D 아바타의 애니메이션 가능한 인간 모델을 생성하고, 2D 동영상에서 3D 신체 움직임을 추출하여 이 정보를 활용해 비디오를 생성합니다. 이 과정에서 Vision Language Model (VLM)을 사용하여 아바타의 외형을 설명하는 프롬프트를 생성합니다. 또한, 기존 텍스트-비디오 모델에 대해 조건부 매개 변수 고효율 미세 조정 방법을 제안합니다.

- **Performance Highlights**: AMG는 기존 인간 비디오 생성 방법들보다 사실성과 적응성에서 우수한 성능을 보이며, 다중 객체 비디오 생성에서 카메라 위치, 인간 동작 및 배경 스타일에 대한 정밀한 제어를 가능하게 합니다.



### Real-Time Multi-Scene Visibility Enhancement for Promoting Navigational Safety of Vessels Under Complex Weather Conditions (https://arxiv.org/abs/2409.01500)
Comments:
          15 pages, 13 figures

- **What's New**: 이번 논문에서는 다양한 기상 조건에서 촬영된 품질 저하 이미지를 복원하기 위한 일반 목적의 멀티-씬 (multi-scene) 가시성 향상 방법인 ERANet을 제안합니다. 기존의 방법들은 특정 기상 조건에만 국한되어 있었으나, ERANet은 복합 기상 조건에서도 적응적으로 이미지를 복원할 수 있습니다.

- **Technical Details**: ERANet은 채널 주의 (channel attention), 공간 주의 (spatial attention), 그리고 재매개화 (reparameterization) 기술을 활용하여 시각적 품질을 향상시키고, 추가적인 계산 비용을 최소화합니다. Kirsch 연산자를 통해 8 방향의 기울기를 추출하여 의미 있는 엣지 특징을 효과적으로 캡처합니다.

- **Performance Highlights**: 다양한 기상 조건에서의 실험 결과 ERANet은 기존의 여러 가시성 향상 방법에 비해 이미지 품질과 계산 효율성 모두에서 우수한 성능을 보여주었습니다. 특히, 해상도와 장면 분할 성능에서도 두드러진 향상을 보였습니다.



### EarthGen: Generating the World from Top-Down Views (https://arxiv.org/abs/2409.01491)
- **What's New**: 본 연구에서는 고해상도에서 일관된 이미지를 생성하기 위해 다중 해상도로 결합할 수 있는 초해상도 스프레드(diffusion) 모델의 캐스케이드를 기반으로 한 새로운 방법인 EarthGen을 소개합니다. 이 시스템은 수천 제곱킬로미터의 현실적인 지구 표면을 생성할 수 있는 확장 가능한 구조입니다.

- **Technical Details**: EarthGen은 계층적 생성 방식과 구성적 방식의 장점을 결합한 새로운 프레임워크로, 저해상도에서 시작하여 점진적으로 진화하는 방식을 채택합니다. 스프레드 초해상도 모듈을 사용하여 각 스케일에서 그럴듯한 특징을 정밀하게 추가합니다. 이 모듈은 스케일 인식(scale-aware) 기능을 갖추고 있어 지역적부터 킬로미터, 미터, 센티미터까지 다양한 수준의 구조를 실현할 수 있습니다.

- **Performance Highlights**: 본 시스템은 30km x 10km 크기의 지형을 15cm/pixel 해상도로 생성하며, 이는 맨해튼보다 세 배 큰 규모입니다. 1024x 해상도에서 기존의 초해상도 기법보다 우수한 성능을 보여주었고, 다양한 씬을 생성할 수 있는 가능성도 확인했습니다.



### Semantic Segmentation from Image Labels by Reconstruction from Structured Decomposition (https://arxiv.org/abs/2409.01472)
- **What's New**: 본 연구에서는 약한 감독 이미지 분할(Weakly Supervised Image Segmentation, WSSS) 문제를 마스크를 사용한 이미지의 분해 복원 문제로 재구성하여 접근합니다. 기존의 방법들과는 달리, 우리는 다양한 추가적인 정규화를 모델 내에서 암시적으로 포함시키는 구조를 제안합니다.

- **Technical Details**: 이 연구에서는 주어진 이미지와 태그 데이터를 활용하여 약한 분할 문제를 해결하기 위해 두 개의 신경망을 학습합니다: 마스크 네트워크(fm)와 분해 네트워크(fx). 마스크 네트워크는 이미지를 입력받아 각 클래스의 분할 마스크를 출력하고, 분해 네트워크는 입력 이미지를 K개의 이미지로 분해하여 각각의 클래스와 관련된 정보를 더 잘 학습할 수 있도록 합니다.

- **Performance Highlights**: 초기 실험 결과, 제안한 방법이 배경 모호성 문제에 대해 강인성을 보이며, promising results를 보여주었습니다.



### 3D-LSPTM: An Automatic Framework with 3D-Large-Scale Pretrained Model for Laryngeal Cancer Detection Using Laryngoscopic Videos (https://arxiv.org/abs/2409.01459)
- **What's New**: 이 연구는 자동화된 3D-LSPTM 모델을 통해 후두암 탐지를 위한 새로운 접근 방식을 제안합니다. 기존의 수작업 탐지 방식의 한계를 극복하고, 1,109개의 후두경 비디오를 활용하여 딥러닝 기반의 효율적인 진단 도구를 제공합니다.

- **Technical Details**: 연구에서는 C3D, TimeSformer, Video-Swin-Transformer와 같은 3D 대규모 사전 훈련 모델을 이용하여 비디오 데이터를 분석합니다. 이 모델들은 긴 거리 종속성을 캡처하며, 고해상도 비디오 입력의 처리를 최적화하는 다양한 기법을 포함합니다.

- **Performance Highlights**: 3D-LSPTM 모델은 92.4%의 정확도, 95.6%의 민감도, 94.1%의 정밀도, 94.8%의 F1 점수를 기록하여 후두암 탐지에서 뛰어난 성능을 입증했습니다.



### FinePseudo: Improving Pseudo-Labelling through Temporal-Alignablity for Semi-Supervised Fine-Grained Action Recognition (https://arxiv.org/abs/2409.01448)
Comments:
          ECCV 2024

- **What's New**: 이 논문은 반지도 학습(semi-supervised learning)에서 미세한 액션 인식(fine-grained action recognition, FGAR)을 처음으로 정밀하게 탐구합니다.

- **Technical Details**: 제안된 방법은 Alignability-Verification 기반의 메트릭 학습(metric learning) 기법을 통하여 미세한 액션 쌍들을 효과적으로 구별하는 방법론을 제공합니다. 이를 통해 학습된 alignability 점수를 활용하여 주요 비디오 인코더의 의사 라벨(pseudo-labels)을 정제(refine)합니다.

- **Performance Highlights**: FinePseudo라는 협업적 의사 라벨링 프레임워크는 Diving48, FineGym99, FineGym288, FineDiving의 네 가지 미세 액션 인식 데이터셋에서 이전 방법들보다 유의미한 성과를 보였으며, Kinetics400 및 Something-SomethingV2와 같은 기존의 거친 데이터셋에서도 개선된 결과를 기록했습니다.



### Sync from the Sea: Retrieving Alignable Videos from Large-Scale Datasets (https://arxiv.org/abs/2409.01445)
Comments:
          ECCV 2024 Oral

- **What's New**: 이 연구는 Alignable Video Retrieval (AVR)이라는 새로운 작업을 도입하여, 주어진 쿼리 비디오에 대해 잘 정렬 가능한 비디오를 대량의 클립에서 식별하고 이를 시간적으로 동기화하는 방법을 제시합니다.

- **Technical Details**: 주요 기술적 기여는 다음과 같습니다: 1) Dynamic Relative Alignment Quality (DRAQ)라는 비디오 정렬 가능성 지표를 도입하여 후보 중 최상의 정렬 가능한 비디오를 식별하고 재순위화합니다; 2) 매 프레임 비디오 특성을 개선하는 일반화된 특성 설계를 제안하며, 3) 사이클 일관성 메트릭을 사용한 AVR 벤치마크와 평가 프로토콜을 제안합니다.

- **Performance Highlights**: Kinetics700을 포함한 3개의 데이터셋에서의 실험 결과, 제안된 방법이 다양한 데이터셋으로부터 정렬 가능한 비디오 쌍을 식별하는 데 효과적임을 입증했습니다.



### Kvasir-VQA: A Text-Image Pair GI Tract Datas (https://arxiv.org/abs/2409.01437)
Comments:
          to be published in VLM4Bio 2024, part of the ACM Multimedia (ACM MM) conference 2024

- **What's New**: Kvasir-VQA 데이터세트는 HyperKvasir 및 Kvasir-Instrument 데이터세트를 기반으로 질문-답변 주석을 추가하여 구성되었으며, 위장관(Gastrointestinal, GI) 진단을 위한 고급 기계 학습(machine learning, ML) 작업을 지원하는 데 초점을 맞추고 있습니다. 이 데이터세트는 6,500개의 주석이 달린 이미지를 포함하고 있으며, 다양한 질문 유형(y/n, 선택, 위치, 수치 등)을 지원합니다.

- **Technical Details**: Kvasir-VQA 데이터세트는 의학 영상 분석과 진단 도구 간의 격차를 해소하기 위해 질문-답변 주석이 추가된 새로운 형식으로 구성되었습니다. 이 데이터세트는 이미지 캡셔닝(image captioning), Visual Question Answering (VQA), 텍스트 기반 합성 의료 이미지 생성 등을 포함한 다양한 ML 응용 프로그램을 지원하는 데 사용됩니다. 실험은 이 데이터세트의 유용성을 입증하기 위해 image captioning, VQA, 합성 의료 이미지 생성 등의 세 가지 작업을 중점적으로 다룹니다.

- **Performance Highlights**: 실험 결과, Kvasir-VQA 데이터세트는 의료 이미지 분석 및 진단의 기계 학습 모델 훈련에 효과적임을 입증하였습니다. 제시된 평가 메트릭을 통해 각 작업의 성능을 측정하여 데이터세트의 유용성과 다양성을 강조하고 있습니다.



### From Pixels to Objects: A Hierarchical Approach for Part and Object Segmentation Using Local and Global Aggregation (https://arxiv.org/abs/2409.01353)
- **What's New**: 본 연구에서는 정교한 이미지 분할 작업을 위해 설계된 계층적 transformer 기반 모델인 LGFormer를 소개합니다. 이 모델은 부품 분할(part segmentation)과 객체 분할(object segmentation)의 세분화 및 포괄성을 효과적으로 연결합니다.

- **Technical Details**: LGFormer는 픽셀에서 슈퍼픽셀(superpixels), 그리고 최종적으로 cohesive group formations으로 전진하는 다중 레벨 표현 전략을 채택합니다. 이 아키텍처는 지역 집계(local aggregation)와 전역 집계(global aggregation)의 두 가지 주요 집계 전략에 기초하고 있습니다. LGFormer는 비주얼 데이터의 계층적 조직을 통해 다중스케일 정보를 포착하며, 'association-aware upsampling'이라는 혁신적인 업샘플링 기법을 사용하여 세밀한 공간 정보를 보존합니다.

- **Performance Highlights**: PartImageNet 데이터셋에서 LGFormer는 부품 분할 및 객체 분할에서 각각 2.8% 및 0.8%의 mIoU 점수 향상을 기록하며 이전의 최첨단 모델을 초월했습니다. Pascal Part 데이터셋에서도 부품 및 객체 분할에서 각각 1.5% 및 2.0%의 성능 향상을 보였습니다.



### PatternPaint: Generating Layout Patterns Using Generative AI and Inpainting Techniques (https://arxiv.org/abs/2409.01348)
- **What's New**: 이 연구에서는 디자인 규칙에 적합한 금속 레이아웃 패턴을 생성하기 위한 생성 기계 학습 모델의 가능성을 탐구합니다. 제안된 모델은 복잡한 디자인 규칙 설정에서 합법적인 패턴을 생성할 수 있으며 높은 다양성 점수를 달성하는 것으로 나타났습니다.

- **Technical Details**: 이 연구의 핵심은 PatternPaint라는 자동화된 픽셀 기반 레이아웃 생성 프레임워크입니다. 이는 Pre-trained image foundation 모델인 DALL-E, StableDiffusion, Midjourney의 이점을 활용하여 DRC(Design Rule Checking) 적합 패턴을 생성합니다. 기존의 규칙 기반 또는 ML 기반 방법들에 비해 PatternPaint는 훈련 없이 일반적인 Pre-trained 모델을 사용하여 레이아웃 패턴을 생성할 수 있습니다.

- **Performance Highlights**: PatternPaint는 Intel 18A 프로세스 디자인 킷(PDK)에서 검증되었으며, 단 20개의 스타터 패턴으로 4000개 이상의 DRC 적합 패턴 라이브러리를 생성할 수 있는 능력을 가지고 있습니다. 이 프레임워크는 복잡한 산업 표준 디자인에 적응할 수 있으며, 스타터 패턴이 제한된 상황에서도 효과적으로 작동하는 것이 주요 장점입니다.



### Target-Driven Distillation: Consistency Distillation with Target Timestep Selection and Decoupled Guidanc (https://arxiv.org/abs/2409.01347)
- **What's New**: 본 논문에서는 Target-Driven Distillation (TDD)라는 새로운 일관성 증류(consistency distillation) 방법을 도입합니다. TDD는 목표 timesteps의 세밀한 선택 전략을 채택하여 훈련 효율성을 높이고, 분리된 가이던스를 활용하여 추론 기간 동안 사용자 정의가 가능하도록 설계되었습니다.

- **Technical Details**: TDD는 (1) 미리 정의된 일정의 몇 단계 간격을 유지하여 timesteps를 선택하고, (2) 클래스 기반의 가이던스를 도입하여 훈련 중 조건을 조정하며, (3) 비균일 샘플링(non-equidistant sampling) 및 x0 클리핑(x0 clipping)을 통해 이미지 품질을 향상시킵니다. 이러한 방법들을 통해 TDD는 단일 목표 방식을 넘어서는 다중 목표(distillation) 접근 방식을 실현합니다.

- **Performance Highlights**: 실험 결과 TDD는 몇 단계 생성(few-step generation)에서 최첨단 성능을 달성하여 기존의 일관성 증류 모델들 사이에서 더 나은 선택지를 제공합니다.



### Enhancing Test Time Adaptation with Few-shot Guidanc (https://arxiv.org/abs/2409.01341)
Comments:
          8 pages, 7 figures

- **What's New**: 깊은 신경망은 훈련 데이터(소스)와 테스트 데이터(타겟) 간의 도메인 변화에 직면했을 때 성능 저하를 겪는 경향이 있습니다. 이를 해결하기 위해, Few-Shot Test Time Adaptation (FS-TTA)라는 새로운 접근 방식을 제안하여 몇 개의 샷(support set)을 사용하는 이점과 TTA의 특징을 통합했습니다.

- **Technical Details**: FS-TTA는 두 단계 프레임워크로 구성되어 있습니다; (i) few-shot support set을 활용하여 사전 훈련된 소스 모델을 미세 조정하고, 과적합(overfitting)을 방지하기 위해 Feature Diversity Augmentation (FDA) 모듈을 사용하는 것, (ii) 프로토타입 메모리 뱅크(prototype memory bank)를 통해 높은 품질의 의사 라벨(pseudo-label)을 생성하여 모델 적응을 구현하는 것입니다.

- **Performance Highlights**: FS-TTA는 다양한 크로스 도메인 분류 벤치마크에 대한 실험을 통해 기존 TTA 방법들과 비교하여 2.0% (PACS), 7.8% (OfficeHome), 3.9% (DomainNet)의 성능 향상을 달성했습니다.



### Pediatric brain tumor classification using digital histopathology and deep learning: evaluation of SOTA methods on a multi-center Swedish cohor (https://arxiv.org/abs/2409.01330)
- **What's New**: 본 연구는 스웨덴의 다중 센터 데이터셋을 활용하여 소아 뇌종양의 분류를 위해 두 가지 약한 슈퍼바이즈드 다중 인스턴스 학습 접근 방식을 구현하여, 최신 병리학적 컴퓨터 비전 기술이 소아 뇌종양 진단에 기여할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 연구는 540명의 소아 뇌종양 환자의 전체 슬라이드 이미지(WSI)에서 패치 특징을 추출하고, 세 가지 사전 학습된 특징 추출기(ResNet50, UNI, CONCH)를 사용하여, 주의 기반 MIL(ABMIL) 및 클러스터 제약 주의 MIL (CLAM)로 패치 수준의 특징을 환자 수준으로 집계합니다.

- **Performance Highlights**: 종양 범주, 계통, 유형 분류의 경우 각각 Matthew 상관계수 0.86±0.04, 0.63±0.04 및 0.53±0.05를 기록하며, UNI 특징과 ABMIL 집계를 사용했을 때 최고의 분류 성능이 달성되었습니다. 모델 일반화는 UNI와 CONCH 특징을 사용하는 모델이 ResNet50을 사용하는 모델보다 성능이 우수했으며, 테스트 결과는 여러 센터의 데이터에서 0.7의 AUC를 기록했습니다.



### SPDiffusion: Semantic Protection Diffusion for Multi-concept Text-to-image Generation (https://arxiv.org/abs/2409.01327)
- **What's New**: 최근 텍스트-이미지 모델들이 다수의 캐릭터나 객체를 포함하는 다중 개념 이미지 생성에서 특징 혼란(attribute confusion) 문제로 어려움을 겪고 있음을 발견한 연구가 있습니다. 이를 해결하기 위해 Semantic Protection Diffusion (SPDiffusion)이라는 새로운 기법을 제안합니다.

- **Technical Details**: SPDiffusion은 해당 영역이 불필요한 토큰에 주의를 기울이지 않도록 하는 방식으로 작동합니다. 이를 위해 SP-Mask라는 개념을 도입하여 각 영역에서 어떤 토큰이 차단되어야 하는지 표현하고, SP-Attn을 통해 불필요한 토큰의 영향을 차단합니다. 이를 통해 다중 개념 텍스트-이미지 생성 과정에서 발생할 수 있는 속성 혼란을 감소시킬 수 있습니다.

- **Performance Highlights**: SPDiffusion은 CLIP 점수와 FID에서 기존 방법들을 능가하며, 다양한 멀티-개념 벤치마크에서 최첨단 성능을 구현했습니다. 이 방법은 ControlNet, Story Diffusion, PhotoMaker와 같은 기존의 다양한 생성 프레임워크와 연동되어 사용될 수 있으며, 높은 호환성과 확장성을 보여줍니다.



### Guide-and-Rescale: Self-Guidance Mechanism for Effective Tuning-Free Real Image Editing (https://arxiv.org/abs/2409.01322)
- **What's New**: 이 논문은 수정된 확산 샘플링 프로세스를 기반으로 한 기존 이미지 편집 방법들의 한계를 극복할 수 있는 새로운 접근 방식을 제안합니다. 특히, 전체 입력 이미지 구조와 로컬 영역의 외관을 보존하기 위한 자체 안내(self-guidance) 기술을 탐구합니다.

- **Technical Details**: 새롭게 도입한 레이아웃 보존 에너지 함수(layout-preserving energy functions)는 원본 이미지의 로컬 및 글로벌 구조를 유지하도록 설계되었습니다. 본 논문에서 제안된 노이즈 재조정 메커니즘(noise rescaling mechanism)은 생성 중에 클래스화 없는 가이드를 균형 잡아 노이즈 분포를 보존합니다. 이 방법은 이미지를 편집하기 위한 세밀한 튜닝(fine-tuning)을 필요로 하지 않으며, 따라서 효율적입니다.

- **Performance Highlights**: 제안된 방법은 인간 평가 및 정량적 분석을 통해 편집 품질과 원본 이미지 보존 간의 우수한 균형을 달성함을 보여주며, 기존의 방법들보다 더 높은 성능을 발휘합니다. 실험 결과, 표준 이미지-투-이미지 문제에서 본 방법이 다른 기준선보다 나은 성능을 보였습니다.



### LoGex: Improved tail detection of extremely rare histopathology classes via guided diffusion (https://arxiv.org/abs/2409.01317)
- **What's New**: 이 논문은 의료 분야에서의 희귀 질병 데이터의 이상 탐지(OOD detection)를 개선하기 위한 새로운 접근법, LoGex(LoRA + Guidance for improved detection of extremely rare classes)를 소개합니다. 기존의 희귀 클래스 분류보다 이상 탐지에 초점을 맞추어, 극단적인 데이터 부족 상황에서도 성능을 유지할 수 있도록 설계되었습니다.

- **Technical Details**: LoGex 접근법은 low-rank adaptation (LoRA)와 diffusion 모델의 가이드를 활용하여, 헤드 클래스의 분류 성능을 손상시키지 않으면서도 테일 클래스의 탐지 성능을 개선합니다. 본 연구에서는 10개의 샘플만으로 구성된 극단적인 긴 꼬리의 조직병리학적 문제를 제시합니다. 최근 개발된 중요성 있는 diffusion 모델이 이 과정에서 널리 활용됩니다.

- **Performance Highlights**: LoGex를 사용하여 테일 탐지 성능에서 모든 기존 기준 방법들보다 향상된 결과를 보고하였고, 헤드 클래스의 분류 정확도는 손상되지 않았습니다. 이는 의료 이미징 데이터에서 효율적이고 신뢰할 수 있는 희귀 질병 탐지를 위한 중요한 진전을 의미합니다.



### Disentangling Mean Embeddings for Better Diagnostics of Image Generators (https://arxiv.org/abs/2409.01314)
Comments:
          Preprint

- **What's New**: 본 논문에서는 이미지 생성기(Generators) 평가의 어려움을 해결하기 위해 새로운 접근 방식을 제안합니다. 이는 전체 이미지 성능을 클러스터별로 분리하여 평가할 수 있도록 하여 각 클러스터의 기여도를 정량화하는 것입니다.

- **Technical Details**: 기술적으로, 우리는 mean embeddings의 코사인 유사성(cosine similarity)를 분리하여 각 픽셀 클러스터에 대한 코사인 유사성의 곱(product)으로 분해하는 방법을 제시합니다. 이를 위해 중앙 커널 정렬(Central Kernel Alignment)을 이용합니다.

- **Performance Highlights**: DCGAN(Deep Convolutional Generative Adversarial Network) 및 DDPM(Denoising Diffusion Probabilistic Models) 아키텍처를 이용하여 CelebA와 ChestMNIST 데이터셋에서 일반화 성능을 모니터링함으로써 해석 가능성이 향상되는 것을 보여줍니다.



### One-Index Vector Quantization Based Adversarial Attack on Image Classification (https://arxiv.org/abs/2409.01282)
- **What's New**: 본 논문에서는 Vector Quantization (VQ) 도메인에서 새로운 하나의 인덱스 공격(one-index attack) 방법을 제안합니다. 이 방법은 기존의 이미지 분류를 대상으로 한 적대적 공격(adversarial attack) 방법들이 주로 픽셀 영역(pixel domain)에서 진행된다는 문제를 해결합니다.

- **Technical Details**: 하나의 인덱스 공격 방법은 압축된 데이터 스트림에서 단일 인덱스만을 수정하여 디코딩된 이미지가 잘못 분류되도록 합니다. 이 방법은 수정해야 할 인덱스의 수를 제한하여 공격의 효율성을 높입니다. 제안된 방법은 세미 블랙박스 공격(semi-black-box attack)으로 실제 공격 시나리오에 더 적합합니다.

- **Performance Highlights**: 제안된 방법을 통해 Resnet, NIN, VGG16의 세 가지 인기 이미지 분류 모델을 공격한 결과, CIFAR-10과 Fashion MNIST에서 각각 55.9%와 77.4%의 이미지가 성공적으로 공격되었으며, 높은 수준의 잘못 분류되는 신뢰성과 낮은 수준의 이미지 변형이 나타났습니다.



### DAVIDE: Depth-Aware Video Deblurring (https://arxiv.org/abs/2409.01274)
- **What's New**: 본 연구에서는 깊이 정보(Depth Information)를 활용한 비디오 디블러링(Video Deblurring) 기술에 대한 데이터셋인 'Depth-Aware VIdeo DEblurring' (DAVIDE)를 소개합니다. 이 데이터셋은 동기화된 블러(Blurred), 선명한(Sharp), 깊이(Depth) 비디오로 구성되어 있으며, 실제 센서를 사용하여 수집되었습니다.

- **Technical Details**: DAVIDE 데이터셋은 iPhone 13 Pro와 LiDAR 센서를 사용하여 블러, 선명한 및 깊이 맵 영상을 동기화하여 캡처하였습니다. 본 연구에서는 깊이 정보를 기존의 딥 RGB 비디오 디블러링 모델에 어떻게 통합할 수 있는지 탐구하고, 깊이 인젝션 방법(Depth Injection Method)과 Depth-aware Transformer (DaT) 블록을 제안하였습니다.

- **Performance Highlights**: 깊이 정보가 비디오 디블러링 성능을 향상시킴을 보여주며, 이는 가벼운 temporal context 상황에서는 뚜렷한 효과를 보이지만, 긴 temporal context를 제공할 때에는 그 효과가 줄어드는 경향이 있습니다.



### Real-time Accident Anticipation for Autonomous Driving Through Monocular Depth-Enhanced 3D Modeling (https://arxiv.org/abs/2409.01256)
- **What's New**: 본 연구에서는 새로운 프레임워크인 AccNet을 소개하여 기존의 2D 기반 방법들을 넘어서는 예측 능력을 선보입니다. 이는 단안 깊이 정보(monocular depth cues)를 통합하여 복잡한 3D 장면 모델링을 가능하게 합니다. 또한, 사고 데이터셋의 불균형 문제를 해결하기 위해 Binary Adaptive Loss for Early Anticipation (BA-LEA)라는 새로운 손실 함수를 제안합니다.

- **Technical Details**: AccNet은 3D Collision Module을 중심으로 하여, 단안 깊이 정보를 이용해 차량과 보행자와 같은 주요 객체의 3D 좌표를 정확하게 추출합니다. 이를 통해 dashcam 비디오에서의 공간적 관계와 동태를 더 잘 이해할 수 있게 하며, 새로운 그래프 토폴로지를 도입하여 Graph Neural Networks (GNN)의 리스크 평가를 향상시킵니다. 또한, 멀티 태스크 학습 전략(multi-task learning strategy)과 Smooth Module을 도입하여 예측 모델의 결정적인 순간에 대한 초점을 맞추는데 기여합니다.

- **Performance Highlights**: DAD, CCD, A3D, 및 DADA-2000 데이터셋을 통해 검증한 결과, AccNet은 Average Precision (AP) 및 mean Time-To-Accident (mTTA)와 같은 주요 지표에서 기존 방법들을 초월한 뛰어난 예측 성능을 입증하였습니다. 이를 통해 사고 예측 기술에 있어 중요한 진전을 이룩하였습니다.



### Spatial-Aware Conformal Prediction for Trustworthy Hyperspectral Image Classification (https://arxiv.org/abs/2409.01236)
- **What's New**: 본 논문은 Hyperspectral 이미지 분류 (HSI Classification)에서 불확실성을 정량화하는 	extit{Conformal Prediction} (CP) 기법을 제안하며, CP의 적용 가능성을 이론적으로 평가합니다. 이를 바탕으로 	extit{Spatial-Aware Conformal Prediction} (	exttt{SACP})를 도입하여 고차원 HSI의 공간 정보를 활용하는 새로운 방법론을 소개합니다.

- **Technical Details**: 이 연구에서는 HSI 분류의 신뢰할 수 있는 예측 집합을 제공하는 CP 프레임워크를 소개하고, 	exttt{SACP} 기법이 비례 데이터 로드 및 공간 상관성을 고려하여 비일관성 점수를 통합하는 방식을 통해 성능을 향상시킨다고 설명합니다. SACP는 전통적 CP보다 더 효율적인 예측 집합을 생성합니다.

- **Performance Highlights**: 실험 결과, 	exttt{SACP}는 Indian Pines, Pavia University 및 Salinas와 같은 HSI 분류 벤치마크에서 기존의 CP보다 뛰어난 성과를 보였으며, 다양한 비일관성 점수 함수에 대해 평균 예측 집합 크기를 줄이면서도 만족스러운 커버리지 비율을 유지했습니다. 이는 CNN 및 Transformer 기법을 모두 포함한 여러 분류기에 대한 적용 가능성을 보여줍니다.



### A Review of Image Retrieval Techniques: Data Augmentation and Adversarial Learning Approaches (https://arxiv.org/abs/2409.01219)
- **What's New**: 이 논문은 이미지 검색(image retrieval) 분야에서 데이터 증대(data augmentation)와 적대적 학습(adversarial learning) 기술이 검색 성능을 향상시키는 역할에 대한 최신 연구 동향을 종합적으로 요약합니다.

- **Technical Details**: 이미지 검색은 대규모 이미지 데이터베이스에서 쿼리 이미지와 유사한 타겟 이미지를 검색하기 위해 설계된 컴퓨터 비전의 중요한 연구 분야입니다. 깊은 학습(deep learning)의 발전, 특히 합성곱 신경망(Convolutional Neural Networks, CNNs)이 이러한 시스템의 기능 추출 및 매칭에서 표준으로 자리 잡았으나, 여전히 교차 도메인 검색(cross-domain retrieval), 이미지 잡음(image noise) 및 방해 요인(perturbations) 처리에서 도전 과제가 존재합니다.

- **Performance Highlights**: 데이터 증대는 훈련 샘플의 다양성을 증가시켜 모델의 일반화(generalization) 능력을 향상시키며, 적대적 학습은 모델이 잠재적인 공격에 대한 견고성을 높이는 데 기여합니다. 이러한 기술들은 대규모 데이터셋과 실세계 변동성 문제를 해결하기 위해 필수적입니다.



### ESP-PCT: Enhanced VR Semantic Performance through Efficient Compression of Temporal and Spatial Redundancies in Point Cloud Transformers (https://arxiv.org/abs/2409.01216)
- **What's New**: 이번 연구에서는 VR(가상현실) 애플리케이션에서 중요한 의미 인식을 개선하기 위해 새로운 모델인 ESP-PCT를 제안합니다. ESP-PCT는 밀리미터파(mmWave) 신호를 활용하여 점 구름(point cloud)을 생성하고, 인식 과정을 최적화합니다.

- **Technical Details**: ESP-PCT는 두 단계의 의미 인식 프레임워크를 채택하고 있으며, 센서 데이터의 정확성을 활용하여 로컬라이제이션(localization)과 포커스(focus) 단계를 공동으로 훈련시킵니다. 이는 엔드 투 엔드(end-to-end) 방식으로 진행되며, 기존의 Point Transformer 모델에 비해 효율성을 향상시키는 데 중점을 두고 있습니다.

- **Performance Highlights**: ESP-PCT는 인식 효율성에서 엄청난 향상을 보였으며, 정확도는 93.2%에 달합니다. 또한, FLOPs(부동소수점 연산 수)는 76.9%, 메모리 사용량은 78.2% 감소하여 VR 의미 인식에서 높은 정확도와 중복성 감소를 달성할 수 있음을 보여주었습니다.



### MobileIQA: Exploiting Mobile-level Diverse Opinion Network For No-Reference Image Quality Assessment Using Knowledge Distillation (https://arxiv.org/abs/2409.01212)
Comments:
          Accepted by ECCV Workshop 2024

- **What's New**: 이 논문에서는 고해상도(High Resolution) 이미지의 품질 평가를 위한 새로운 방법인 MobileIQA를 제안합니다. MobileIQA는 기존의 NR-IQA 방식이 가진 이미지 세부 정보 손실과 높은 계산 복잡성 문제를 해결하기 위해 가벼운 백본을 사용하여 효율적으로 이미지를 평가합니다. 또한, 다양한 관점을 포착하는 다중 뷰 주의 학습(Multi-view Attention Learning, MAL) 모듈을 도입하여 여러 사람의 주관적인 의견을 시뮬레이션합니다.

- **Technical Details**: MobileIQA는 교사 모델(Teacher Model)과 학생 모델(Student Model)로 구성되어 있으며, 각각 MobileViT-IQA와 MobileNet-IQA를 사용합니다. 이들은 최대 1907×1231 해상도를 지원하여 HR 이미지의 세부 사항을 보존합니다. MAL 모듈을 사용하여 다양한 의견 특징을 집합적으로 수집하고, 지식 증류(Knowledge Distillation) 기법을 통해 MobileViT-IQA에서 MobileNet-IQA로 지식을 전이함으로써 계산 복잡성을 줄입니다.

- **Performance Highlights**: 실험 결과에 따르면 MobileIQA는 최신 IQA 모델에 비해 평가 지표와 계산 효율성 모두에서 우수한 성능을 보여주며, 많은 선진 기법들보다 더 높은 정확도와 효율성을 제공합니다.



### OD-VAE: An Omni-dimensional Video Compressor for Improving Latent Video Diffusion Mod (https://arxiv.org/abs/2409.01199)
Comments:
this https URL

- **What's New**: 최근 논문에서는 Omni-Dimension Compression VAE(OD-VAE)를 제안하여 비디오의 시공간 압축을 동시에 수행할 수 있는 방법을 소개하고 있습니다. 기존의 Variational Autoencoder(VAE)는 비디오 데이터를 주로 공간 차원에서만 압축하였지만, OD-VAE는 시간 차원에서도 압축을 가능하게 하여 더욱 간결한 잠재 표현(latent representations)을 제공합니다.

- **Technical Details**: OD-VAE는 3D-Causal-CNN 아키텍처를 기반으로 하여 비디오의 시공간 정보를 효과적으로 활용합니다. 이 논문에서는 OD-VAE의 네 가지 모델 변형(variants)을 소개하고 각각의 성능을 분석하며, 새로운 tail initialization 및 temporal tiling 기법을 통해 훈련 효율성과 비디오 추론 능력을 향상시키는 방법을 제시합니다.

- **Performance Highlights**: OD-VAE는 비디오 복원 정확성을 높은 수준으로 유지하면서, Latent Video Diffusion Models(LVDMs)의 효율성을 크게 향상시키는 데 기여합니다. 관련 실험 결과들은 OD-VAE가 기존 VAE보다 더 나은 비디오 생성 능력을 가지며, 하드웨어 자원 소모를 줄이는 데 효과적임을 보여줍니다.



### PitVis-2023 Challenge: Workflow Recognition in videos of Endoscopic Pituitary Surgery (https://arxiv.org/abs/2409.01184)
- **What's New**: Pituitary Vision (PitVis) 2023 Challenge는 최소 침습 수술에서의 비디오를 분석하여 수술 단계 및 사용된 의료 기구를 인식하는 새로운 과제를 제안합니다. 특히, 뇌하수체 선종 제거를 위한 내시경 경비인두 접근법에서의 자동화된 수술 워크플로 인식 기능이 강조되었습니다.

- **Technical Details**: PitVis-2023의 도전 과제는 크게 (1) 단계 인식, (2) 기구 인식, (3) 단계 및 기구 인식의 세 가지 작업으로 구성되어 있습니다. 참가자들은 25개의 훈련 비디오를 제공받았으며, 이는 각 초마다 현재 단계와 기구가 주석처리되었습니다. 모델은 8개의 테스트 비디오에서 평가되었고, 심층 학습 모델을 사용한 18개의 제출물이 있었습니다.

- **Performance Highlights**: TOP 모델들은 시각-시간적(spatio-temporal) 및 다중 작업(multitask) 방법을 통합하여, 단일 작업(single-task) 모델에 비해 단계 및 기구 인식에서 각각 50% 및 10% 이상 개선된 매크로 F1 점수를 기록했습니다. 이는 최소 침습 수술에서의 최신의 컴퓨터 비전 모델이 새로운 데이터 세트에서도 뛰어난 성능을 발휘함을 보여줍니다.



### Recoverable Compression: A Multimodal Vision Token Recovery Mechanism Guided by Text Information (https://arxiv.org/abs/2409.01179)
- **What's New**: 이 논문에서는 실질적인 트레이닝 없이 질문의 텍스트 정보에 따라 동적으로 시각 토큰을 회수하는 메커니즘을 제안합니다. 이 방법은 중요 정보가 포함된 시각 토큰을 회복하는 데 도움을 줍니다.

- **Technical Details**: 제안된 메커니즘은 시각 토큰과 질문 텍스트 간의 유사성을 계산하여 질문에 의해 안내되는 토큰 회복 과정을 수행합니다. 초기 필터링을 위해 클래스 토큰과 다른 시각 토큰 간의 유사성을 사용합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 원래 모델에 비해 평균적으로 10%의 시각 토큰 압축을 달성하면서도 경쟁력 있는 성능을 유지합니다.



### Variation of Camera Parameters due to Common Physical Changes in Focal Length and Camera Pos (https://arxiv.org/abs/2409.01171)
Comments:
          8 pages, 15 figures

- **What's New**: 이 논문은 카메라 객체의 초점 거리 변화와 카메라 자세 변화로 인한 주 및 부변화를 식별하는 새로운 보정(calibration) 방법을 제시합니다. 기존의 보정 방법이 물리적 변화에 따른 카메라 매개변수의 일반적인 변화를 파악하는 데 부족함을 보여줍니다.

- **Technical Details**: 보정 방법은 geometry-based camera calibration을 기반으로 하며, 카메라의 초점 거리(focal length)와 위치(pose) 변경에 따른 주 변동(강한 변화)과 부 변동(미세 변화)을 다룹니다. 실험 결과에 따르면, 초점 거리가 바뀔 때 주변화는 카메라마다 다르게 나타나는 경향이 있으며, 카메라 자세 변화는 중력 방향에 따라 일관된 경향을 보입니다.

- **Performance Highlights**: 상업용 줌-렌즈 카메라를 이용한 실험에서는 초점 거리 변화 시 약 70~200픽셀의 주점 이동(raprojection)이 관찰되었고, 자세 변화 시 약 10~20픽셀의 부변동이 확인되었습니다. 다양한 보정 방법을 비교한 결과, Zhang의 방법으로는 적은 보정 패턴으로 유사한 reprojection 오류가 발생했으며, Chuang의 방법이 훨씬 작은 허용 가능한 오류를 달성했습니다.



### Balancing Performance and Efficiency: A Multimodal Large Language Model Pruning Method based Image Text Interaction (https://arxiv.org/abs/2409.01162)
- **What's New**: 최근 멀티모달 대형 언어 모델(MM-LLMs)의 성과가 많아짐에 따라, 고비용 계산 문제로 인해 이 모델의 확장과 응용이 제한되고 있습니다. 본 논문은 MM-LLMs의 시각적 토큰을 연구하고, 동적 가지치기 알고리즘을 설계해 시각적 인코더의 출력량을 줄이고 모델 성능과 효율성을 향상시켰다는 점에서 주목할 만합니다.

- **Technical Details**: MM-LLMs는 도Visual Encoder(예: Vision Transformer, CLIP)와 LLM을 이용하여 텍스트와 시각적 정보를 결합해 멀티모달 작업을 처리합니다. 본 연구에서는 CLS 토큰과의 유사성을 분석하여, 동적 가지치기 방법을 통해 우측의 긴 꼬리(long-tail) 분포를 활용해 시각적 토큰을 최적화합니다. 이 과정에서 시각적-텍스트적 상관관계를 기반으로 토큰의 중요성을 계산하고, 8배 이상의 압축 비율을 달성했습니다.

- **Performance Highlights**: 다양한 데이터셋에서 실험 결과, 제안된 방법으로 평균 22%의 원본 토큰 양을 사용하여 원래 성능과 경쟁하는 성능을 달성했습니다. 이 연구는 메모리 제약이 있는 환경에서도 효율적인 추론을 가능하게 하며, 코드의 공개를 계획 중입니다.



### TempMe: Video Temporal Token Merging for Efficient Text-Video Retrieva (https://arxiv.org/abs/2409.01156)
- **What's New**: 이번 연구는 비디오-텍스트 검색에서 시간적 중복성을 줄이기 위한 Temporal Token Merging(TempMe)이라는 혁신적인 방법을 제안하며, 이를 통해 모델의 복잡성을 감소시키고 성능을 향상시킵니다.

- **Technical Details**: TempMe는 인접한 비디오 클립의 토큰을 단계적으로 병합하는 다단계 프레임워크를 채택하여, 비디오 콘텐츠의 고유한 시간적 중복성을 최소화하고, 이를 통해 컴퓨팅 오버헤드를 줄입니다. 비디오를 개별 이미지의 미세 수준에서 전체 비디오의 거시 수준까지 다양한 시간적 수준으로 볼 수 있도록 합니다.

- **Performance Highlights**: TempMe는 기존의 효율적인 비디오-텍스트 검색 방법들과 비교하여 출력 토큰을 95% 줄이고 GFLOPs를 51% 감소시키며, 1.8배의 속도 향상과 4.4%의 R-Sum 개선을 달성했습니다. 또한, 전체 fine-tuning을 적용할 경우, TempMe는 7.9%의 R-Sum 개선과 함께 1.57배 더 빠른 훈련 속도를 보이고, GPU 메모리 사용량을 75.2% 줄입니다.



### Understanding Multimodal Hallucination with Parameter-Free Representation Alignmen (https://arxiv.org/abs/2409.01151)
- **What's New**: 본 논문은 Multimodal Large Language Models (MLLMs)에서 발생하는 hallucination 문제의 근본 원인을 이해하고자 한다. 이를 위하여, 이미지 표현 간의 유사성을 측정할 수 있는 파라메트릭-프리 representation alignment metric(Pfram)을 제안하였다. 이 메트릭은 이미지 표현과 인간의 표현 시스템 간의 정렬을 평가할 수 있어 특히 오브젝트 hallucination과 관련된 연구에 기여한다.

- **Technical Details**: Pfram은 훈련 파라미터 없이 두 개의 표현 시스템 간의 유사성을 측정할 수 있는 메트릭이다. 본 연구에서는 Pfram을 활용하여 MLLMs의 다양한 구성 요소가 hallucination 문제에 미치는 영향을 독립적으로 분석하고, 주로 오브젝트 hallucination에 초점을 맞추었다. 이 메트릭은 대조제 benchmarks를 통해 이미지 내 오브젝트 정보와의 강한 상관관계를 보여준다.

- **Performance Highlights**: Pfram을 적용한 결과, MLLMs에서 감지된 오브젝트 hallucination은 이미지 표현에 포함된 오브젝트 정보와 강한 상관관계를 나타냈다. 또한, 모델 구조나 크기와 같은 다른 요소들은 이러한 강한 상관관계를 보이지 않았다. 이로써, MLLMs의 이미지 이해 과정이 hallucination에 미치는 주요 영향을 확인하였다.



### FMRFT: Fusion Mamba and DETR for Query Time Sequence Intersection Fish Tracking (https://arxiv.org/abs/2409.01148)
Comments:
          14 pages,14 figures

- **What's New**: 이 논문은 복잡한 수조 환경에서 어류의 다중 추적을 위한 새로운 실시간 모델인 FMRFT를 제안하며, 이를 통해 물고기 군집 내의 유사성과 상호 장애물 문제를 해결할 수 있는 효과적인 방법을 제시합니다.

- **Technical Details**: FMRFT 모델은 Mamba In Mamba (MIM) 아키텍처와 RT-DETR의 기능을 융합하여 다중 물체를 효율적으로 추적합니다. 새로운 QTSI(Query Time Sequence Intersection) 모듈을 도입하여 중복된 추적 프레임을 줄이고 연결된 프레임의 효율성을 개선합니다. 또한, 8,000개의 스턴전 물고기 추적 이미지를 포함한 새로운 데이터셋을 구축하여 연구를 뒷받침합니다.

- **Performance Highlights**: 실험 결과, 제안된 FMRFT 모델은 IDF1 점수 90.3%와 MOTA 정확도 94.3%를 달성하여 어류 추적 분야에서 높은 정확성과 안정성을 입증했습니다.



### Generating Synthetic Satellite Imagery for Rare Objects: An Empirical Comparison of Models and Metrics (https://arxiv.org/abs/2409.01138)
Comments:
          Presented at KI 2024 - 47th German Conference on AI, 2nd Workshop on Public Interest AI, 23 September, 2024, Wuerzburg, DE

- **What's New**: 이 연구에서는 드물게 존재하는 객체(핵 발전소)에 대한 합성 위성 이미지를 생성하는 대규모 실증 연구를 소개합니다. 이는 이전 연구와 달리, 현실 세계의 사례가 제한된 경우의 생성 가능한 이미지를 제어하고 평가하는 방법을 탐구합니다.

- **Technical Details**: 본 연구는 텍스트 입력과 게임 엔진에서 얻은 이미지 입력을 통해 핵 발전소라는 드문 객체 카테고리의 합성 위성 이미지를 생성합니다. 이를 위해 사전 훈련된 텍스트-이미지 모델을 활용하고, 대규모 사용자 연구를 통해 생성된 이미지의 신뢰성을 평가합니다. 또한, 자동 평가 메트릭과 인간 평가 간의 신뢰도를 비교합니다.

- **Performance Highlights**: 핵 발전소와 같은 드문 객체의 합성 위성 이미지를 텍스트나 상세한 건축 레이아웃을 통해 생성할 수 있다는 것을 입증했습니다. 그러나 자동 평가 메트릭과 인간의 인식 간에는 강한 부정 상관관계가 발견되어, 기존의 메트릭이 항상 신뢰할 수 있는 것은 아님을 보여줍니다.



### Large Language Models Can Understanding Depth from Monocular Images (https://arxiv.org/abs/2409.01133)
- **What's New**: 이번 논문은 대형 언어 모델(LLMs)이 최소한의 감독 하에 단일 카메라를 통해 깊이 추정을 효과적으로 수행할 수 있음을 보여줍니다. 특히, LLM-MDE라는 다중 모달 프레임워크를 제안하여 언어 이해를 사용하여 깊이를 해석합니다.

- **Technical Details**: LLM-MDE는 Cross-modal reprogramming과 Adaptive prompt estimation 모듈을 통해 LLM의 깊이 추정 능력을 개선합니다. 이 두 가지 전략은 시각 표현과 텍스트 프로토타입을 정렬하고 단일 이미지 기반으로 자동으로 프롬프트를 생성하는 방식으로 작동합니다. 전반적인 프레임워크는 Vision Transformer (ViT)와 LLM의 조합으로 이루어져 있습니다.

- **Performance Highlights**: 다양한 실제 MDE 데이터셋에 대한 실험을 통해 LLM-MDE의 효과와 우수성을 증명했습니다. 특히, LLM-MDE는 소수 샷(few-shot) 및 제로 샷(zero-shot) 작업에서 뛰어난 성능을 발휘하였으며, 자원 사용을 최소화하면서도 높은 예측 정확도를 달성했습니다.



### KMTalk: Speech-Driven 3D Facial Animation with Key Motion Embedding (https://arxiv.org/abs/2409.01113)
Comments:
          Accepted by ECCV 2024

- **What's New**: 이번 연구에서는 오디오 시퀀스를 기반으로 3D 얼굴 모션을 합성하기 위한 새로운 접근 방식을 제시합니다. 이 방법은 키 모션 임베딩(key motion embeddings)을 사용하여 음성과 3D 얼굴 메쉬 간의 매핑 불확실성을 줄이고 학습 복잡성을 완화합니다.

- **Technical Details**: 제안된 방법은 두 가지 모듈로 구성됩니다: 언어 기반 키 모션 수집 모듈과 교차 모드 모션 완성 모듈입니다. 이 첫 번째 모듈은 음성의 음소 변화에 따라 중요한 모션 스냅샷을 식별하고, 두 번째 모듈은 오디오 피처를 안내로 사용하여 불연속적인 키 모션을 전체 모션 시퀀스로 확장합니다.

- **Performance Highlights**: BIWI와 VOCASET 데이터셋에 대한 광범위한 실험 비교를 통해 제안된 방법이 기존의 최첨단 접근 방식들보다 더 정확하고 사실적인 3D 토킹 얼굴 생성을 달성함을 입증했습니다. 이러한 결과는 기존 방법과 결합했을 때도 일관된 개선을 보여줍니다.



### SOOD-ImageNet: a Large-Scale Dataset for Semantic Out-Of-Distribution Image Classification and Semantic Segmentation (https://arxiv.org/abs/2409.01109)
Comments:
          Accepeted as long paper at "The 3rd Workshop for Out-of-Distribution Generalization in Computer Vision Foundation Models", ECCV 2024

- **What's New**: 이 논문에서는 Out-of-Distribution (OOD) 탐지의 필요성을 강조하고, SOOD-ImageNet이라는 새로운 데이터셋을 제안합니다. 이 데이터셋은 약 160만 장의 이미지로 구성되어 있으며 56개 클래스를 포함하고 있습니다. 이 연구는 특히 semantic shift 문제를 다루며, 기존의 OOD 벤치마크들이 간과해온 점을 보완하고자 합니다.

- **Technical Details**: SOOD-ImageNet은 이미지 분류 및 semantic segmentation과 같은 일반적인 컴퓨터 비전 작업을 위한 대규모 데이터셋으로, 이미지 분포의 semantic shift를 중점적으로 다룹니다. 이 데이터셋은 Vision-Language Models (VLMs)의 기능을 활용하여 데이터 품질을 높이고, 수작업으로 검증하여 높은 품질의 샘플을 제공합니다. SOOD-ImageNet은 IID 훈련용으로 약 100만 장, OOD 테스트용으로 약 60만 장을 포함하고 있습니다.

- **Performance Highlights**: 모델들에 대한 실험 결과, SOOD-ImageNet 데이터셋은 Deep Learning (DL) 모델들이 semantic shift에 대해 제대로 일반화하지 못하는 도전 과제를 제시합니다. 다양한 모델 아키텍처를 비교 및 최신 데이터 증강 기술을 적용했지만, 네트워크는 여전히 SOOD 일반화 문제에서 어려움을 겪고 있다는 결과를 얻었습니다.



### OCMG-Net: Neural Oriented Normal Refinement for Unstructured Point Clouds (https://arxiv.org/abs/2409.01100)
Comments:
          18 pages, 16 figures

- **What's New**: 우리의 새로운 프레임워크는 기존의 매우 계산 집약적이거나 정확도가 낮은 방법들과 차별화되며, 불규칙한 점 구름에서의 방향성 노말 추정의 정확성과 효율성을 모두 높이는 혁신적인 방법을 제시합니다.

- **Technical Details**: 제안된 방법은 Chamfer Normal Distance (CND)라는 새로운 메트릭을 소개하여, 노이즈로 인해 발생하는 방향 불일치를 해결하고, 다중 스케일 지오메트리 강화를 통한 네트워크 아키텍처를 통해 복잡한 기하학적 세부정보를 더욱 효과적으로 캡처합니다.

- **Performance Highlights**: 광범위한 실험 결과는 우리의 방법이 다양한 벤치마크 데이터셋에서 기존 방법들에 비해 특별히 노이즈가 있는 점 구름 상황에서도 뛰어난 성능과 복원을 이룬다는 것을 보여줍니다.



### DS MYOLO: A Reliable Object Detector Based on SSMs for Driving Scenarios (https://arxiv.org/abs/2409.01093)
Comments:
          27th International Conference on Pattern Recognition(ICPR)

- **What's New**: 본 논문에서 제안된 새로운 객체 탐지기 DS MYOLO는 Simplified Volitional Scan Fusion Block (SimVSS Block)과 Efficient Channel Attention Convolution (ECAConv)을 사용하여 효과적인 특징 융합과 채널 간 상호작용을 가능하게 하여, 기존 YOLO 시리즈와 유사한 수준의 모델들 사이에서 경쟁력을 높였습니다.

- **Technical Details**: DS MYOLO는 SimVSS Block을 통하여 깊은 글로벌 특징 융합을 달성하며, ECAConv는 저렴한 계산 복잡도를 유지하면서 채널 간의 의존성을 강화합니다. DS MYOLO는 CCTSDB 2021과 VLD-45 데이터셋에서 성능 배치를 통해 뛰어난 경쟁력을 입증했습니다.

- **Performance Highlights**: DS MYOLO는 기존의 동일 규모 YOLO 시리즈 모델들과 비교하여 CCTSDB 2021과 VLD-45 드라이빙 시나리오 데이터셋에서 경쟁력 있는 성과를 보여줍니다. 이 탐지기는 빠른 속도와 높은 정확도를 동시에 달성하며, 자율주행 시스템의 안전성을 향상시키는 데 기여할 잠재력을 가지고 있습니다.



### DPDEdit: Detail-Preserved Diffusion Models for Multimodal Fashion Image Editing (https://arxiv.org/abs/2409.01086)
Comments:
          13 pages,12 figures

- **What's New**: 본 논문에서는 패션 이미지 편집을 위한 새로운 멀티모달 아키텍처인 Detail-Preserved Diffusion Models (DPDEdit)를 소개합니다. 이 모델은 잠재 확산 모델(latent diffusion models)을 기반으로 하여, 텍스트 프롬프트(text prompts), 영역 마스크(region masks), 인간 포즈 이미지(human pose images), 의류 질감 이미지(garment texture images)를 통합하여 패션 이미지 생성을 안내합니다.

- **Technical Details**: DPDEdit의 주요 구성 요소로는 Grounded-SAM을 사용하여 사용자의 텍스트 설명에 기반해 편집 영역을 예측하고, 입력된 조건들을 결합하여 지역 편집(local editing)을 수행합니다. 또한, 질감 주입(texture injection) 및 정제(refinement) 메커니즘을 통해 주어진 의류 질감의 디테일을 타겟 패션 이미지로 전이하는 방식도 제안합니다. 이를 위해 텍스트 설명과 질감 이미지를 통합하는 분리된 크로스 어텐션 레이어(decoupled cross-attention layer)와 U-Net 구조를 활용하여 생성된 의류 질감의 고주파 세부 사항을 보존합니다.

- **Performance Highlights**: 상세한 실험 결과에 따르면, DPDEdit는 주어진 멀티모달 입력과의 이미지 충실도(image fidelity) 및 일관성(coherence) 측면에서 최신 기법(state-of-the-art methods)을 능가하는 성능을 보여줍니다.



### Evidential Transformers for Improved Image Retrieva (https://arxiv.org/abs/2409.01082)
Comments:
          6 pages, 6 figures, To be presented at the 3rd Workshop on Uncertainty Quantification for Computer Vision, at the ECCV 2024 conference in Milan, Italy

- **What's New**: 본 논문에서는 불확실성 기반의 트랜스포머 모델인 Evidential Transformer를 소개하여 이미지 검색에서 개선된 성능을 보여줍니다. 기존의 멀티 클래스 분류 기반의 딥 메트릭 학습을 능가하는 확률적 방법을 이미지 검색에 통합하여 안정적이고 신뢰할 수 있는 결과를 달성했습니다.

- **Technical Details**: Evidential learning은 이미지 검색 과제에서 불확실성을 정량화하는 효과적인 방법으로, 전통적인 결정론적 신경망과는 달리, 두 번째 차원의 확률 분포를 모델링함으로써 더욱 견고하고 정보가 풍부한 프레임워크를 제공한다. Dirichlet 분포를 사용하여 클래스의 접근성을 표현하며, 이러한 접근 방식을 통해 CNN을 기반으로 하는 새로운 이미지 검색 방법을 제시합니다.

- **Performance Highlights**: CUB-200-2011 및 SOP 데이터셋에서 기존의 이미지 검색 방법들을 능가하는 성능을 입증하였고, 기존의 기준 방법들과 비교하여 불확실성 추정을 기반으로 한 새로운 재순위 방법을 통해서도 뛰어난 결과를 보였습니다.



### SCOPE: Sign Language Contextual Processing with Embedding from LLMs (https://arxiv.org/abs/2409.01073)
- **What's New**: 본 연구에서는 대화 맥락을 고려한 새로운 Sign Language Recognition (SLR)과 Sign Language Translation (SLT) 프레임워크인 SCOPE를 소개합니다. 기존의 방법들이 대화 장면에서의 데이터 세트의 다양성 부족과 맥락 정보를 무시하는 문제를 해결하기 위해 제시되었습니다.

- **Technical Details**: SCOPE는 다중 모달 인코더를 활용하여 대화 맥락을 통해 글로스 레벨 인식을 향상시키며, 사전 대화 맥락을 포함하여 대형 언어 모델 (Large Language Model, LLM)을 추가로 미세 조정합니다. 우리는 72시간 분량의 중국 수화 비디오로 구성된 새로운 수화 데이터 세트를 제공하고, 실험을 통해 Phoenix-2014T, CSL-Daily 및 SCOPE 데이터 세트에서 최첨단 성능을 달성하였습니다.

- **Performance Highlights**: SCOPE 프레임워크는 다양한 데이터 세트에서 탁월한 성능을 보였으며, 청각 장애인 커뮤니티의 참가자를 대상으로 실시한 설문조사 결과도 우리의 접근 방식이 실제 응용에서의 강력함과 효과성을 입증하였습니다. 데이터 세트 및 코드는 연구 촉진을 위해 오픈 소스로 제공될 예정입니다.



### Towards Robust Online Domain Adaptive Semantic Segmentation under Adverse Weather Conditions (https://arxiv.org/abs/2409.01072)
- **What's New**: 이번 논문에서는 RODASS라는 온라인 도메인 적응을 위한 새로운 프레임워크를 제안합니다. RODASS는 도메인 이동을 실시간으로 감지하고 하이퍼파라미터를 조정하여 학습 비용과 오류 전파를 최소화합니다.

- **Technical Details**: RODASS는 Dynamic Ambiguous Patch Mask (DAP Mask) 전략과 Dynamic Source Class Mix (DSC Mix) 방법을 도입하여 각기 다른 환경에서의 도메인 변화를 효과적으로 처리합니다. DAP Mask는 고주파 에너지 분석을 통해 불안정한 영역을 선택하고 마스킹하여 오류 축적을 완화합니다. DSC Mix는 클래스 레벨의 소스 버퍼를 활용하여 타겟 데이터의 불확실성과 노이즈 라벨을 줄입니다.

- **Performance Highlights**: RODASS는 기존 OnDA 벤치마크에서 최첨단 기법들보다 우수한 성능을 보이면서도 약 40 프레임 퍼 세컨드(FPS)를 유지합니다.



### VideoLLaMB: Long-context Video Understanding with Recurrent Memory Bridges (https://arxiv.org/abs/2409.01071)
- **What's New**: 최근 대형 비디오-언어 모델의 발전이 실시간 계획 및 세부 상호작용에서 중요한 잠재력을 보여주고 있습니다. 특히, 본 논문에서 소개하는 VideoLLaMB는 전체 비디오 시퀀스와 과거 시각 데이터를 함께 인코딩하여 의미적 연속성을 유지하며 다양한 작업에서 모델 성능을 향상시키는 혁신적인 프레임워크입니다.

- **Technical Details**: VideoLLaMB는 여러 핵심 모듈로 구성되어 있습니다: 1) Semantic Segmentation: 비디오를 의미적으로 구분하는 SceneTilling 알고리즘을 통해 독립적인 의미 단위로 나누어 의미의 흐름을 유지합니다. 2) Recurrent Memory Layer: 재귀 기억 토큰을 이용하여 시각적 정보를 손실없이 인코딩하며, 장기 의존성을 유지합니다. 3) Memory Retriever: 기억 기억소의 주기적 갱신을 통해 성능을 극대화합니다. 이 방법은 16 프레임에서 훈련되었으며, Nvidia A100 GPU에서 최대 320 프레임을 지원합니다.

- **Performance Highlights**: VideoLLaMB는 세 가지 VideoQA 벤치마크에서 5.5 포인트, Egocentric planning에서는 2.06 포인트의 성능 개선을 보여주며, MVBench에서 동급의 7B 모델에 비해 우수한 성과를 기록합니다. 또한, VideoLLaMB는 NIAVH 벤치마크에서 긴 비디오 내 특정 프레임을 정확하게 찾아내는 능력을 입증했습니다.



### Progressive Retinal Image Registration via Global and Local Deformable Transformations (https://arxiv.org/abs/2409.01068)
Comments:
          Accepted at BIBM 2024

- **What's New**: 본 연구에서는 HybridRetina라는 새로운 하이브리드 레티나 이미지 등록 프레임워크를 제안합니다. 이 프레임워크는 글로벌 및 로컬 변형(Deformable) 변환을 점진적으로 적용하여 이미지 등록의 정확성을 개선합니다.

- **Technical Details**: HybridRetina는 키포인트 감지기(Keypoint Detector)와 GAMorph라는 변형 네트워크를 활용하여 각각 글로벌 및 로컬 변형을 추정합니다. 이 과정에서 다중 레벨 픽셀 관계(Multi-level Pixel Relation) 지식을 통합하여 GAMorph의 훈련을 지원하며, 엣지 주의 모듈(Edge Attention Module)을 적용하여 임상적으로 중요한 혈관 영역에 더 초점을 맞추도록 설계되었습니다.

- **Performance Highlights**: FIRE와 FLoRI21이라는 두 개의 널리 사용되는 데이터세트를 기반으로 실험을 진행했으며, HybridRetina가 여러 최신 기법들과 비교했을 때 상당히 우수한 성능을 보임을 확인했습니다. 그 결과는 HybridRetina의 일반화 및 효과성을 입증합니다.



### Follow-Your-Canvas: Higher-Resolution Video Outpainting with Extensive Content Generation (https://arxiv.org/abs/2409.01055)
Comments:
          Github: this https URL Page: this https URL

- **What's New**: 이 논문은 고해상도 비디오 아웃페인팅(Outpainting)에서의 콘텐츠 생성을 탐구합니다. 기존 방법이 겪는 문제점으로는 저품질 콘텐츠 생성과 GPU 메모리에 의한 제한이 있습니다. 이를 해결하기 위해 'Follow-Your-Canvas'라는 확산 기반 방법을 제안하고, 이 방법은 공간 윈도우를 활용하여 작업을 분산시키고 이들을 원활하게 병합하는 두 가지 주요 설계를 기반으로 합니다.

- **Technical Details**: 'Follow-Your-Canvas'는 비디오의 원본 해상도에서 저차원 아웃페인팅 작업을 수행하여 GPU 메모리 한계에 구애받지 않도록 설계되었습니다. 이 방법은 각 창(window)에서의 상대적 위치 정보를 주입하여 생성 과정에서 원본 비디오와 조화를 이루도록 합니다. 또한, 단계별로 멀티 GPU에서 병렬적으로 아웃페인팅을 수행하여 고해상도의 결과물을 빠르게 생성할 수 있습니다.

- **Performance Highlights**: 'Follow-Your-Canvas'는 비디오의 해상도를 512x512에서 1152x2048까지 향상시키며(9배), 고품질의 시각적 결과를 만들어냅니다. 다양한 해상도 및 크기 설정에서 최고의 정량적 성과를 보여줍니다. 예를 들어, DAVIS 2017 데이터셋에서 512x512에서 2048x1152로 아웃페인팅할 때 FVD 점수가 928.6에서 735.3으로 개선되어 (+193.3) 향상되었습니다.



### Learning to Discover Forgery Cues for Face Forgery Detection (https://arxiv.org/abs/2409.01030)
Comments:
          TIFS 2024

- **What's New**: 본 논문에서는 Forgery Cue Discovery (FoCus)라는 새로운 약한 감독(weakly supervised) 모델을 소개하여, 비매칭(faces unpaired) 얼굴에서 조작 단서를 찾는 방법을 제시합니다. 기존의 방법들은 페어링된(real and fake) 얼굴 이미지에 대한 비교를 요구하여 실제 응용에서의 적합성이 떨어지는 문제를 해결하고자 하였습니다.

- **Technical Details**: FoCus는 분류 과정에서 조작 단서를 찾기 위해 분류 주의(region proposal) 모듈과 보조 학습 모듈을 제안합니다. CARP(Classification Attentive Regions Proposal) 모듈은 특징 맵의 여러 레이어에서 max-pooling을 사용하여 조작된 영역을 정확하게 찾고, Sobel 브랜치를 통해 에지 관련 특성을 강조합니다. 이 두 가지를 결합하여 조작 맵(manipulation maps)을 생성하며, 이는 모델의 학습 과정에서 풍부한 단서로 작용합니다.

- **Performance Highlights**: 다양한 데이터셋과 다중 작업 모델에 대한 실험 결과, FoCus는 기존 방법들보다 더욱 뛰어난 일반화 능력과 해석 가능성, 견고성을 입증하였습니다. FoCus의 조작 맵은 다른 방법들에 비해 더 나은 감독(supervision)을 제공하여 얼굴 위조 탐지 모델의 성능을 향상시키는 데 기여하였습니다.



### SINET: Sparsity-driven Interpretable Neural Network for Underwater Image Enhancemen (https://arxiv.org/abs/2409.01022)
- **What's New**: 새로운 논문에서는 수중 이미지 품질 향상을 위해 스파시티 기반 해석 가능한 신경망(SINET)을 소개합니다. 기존의 심층 학습 방법과 달리, 본 모델은 색상 채널에 특화된 새로운 컨볼루션 희소 코딩(CCSC) 모델을 기반으로 하여 이미지 향상 과정의 해석 가능성을 높였습니다.

- **Technical Details**: SINE는 3개의 색상 채널에서 중요 특징을 추출하기 위해 3개의 스파스 특징 추정 블록(SFEBs)을 사용합니다. SFEB의 구조는 $	ext{l}_1$ 규제화된 컨볼루션 희소 코딩(CSC) 문제를 해결하기 위한 반복 알고리즘을 언롤링(ungroll)하여 설계되었습니다. 이 모델은 또한 각 색상 채널에 대해 별도의 CSC 모델을 사용할 수 있도록 하여 파장 의존적인 광 감쇠 문제를 효과적으로 다룹니다.

- **Performance Highlights**: 실험 결과, SINET는 최신의 PSNR(피크 신호 대 잡음비)을 1.05 dB 초과하며, 계산 복잡도는 3873배 낮습니다.



### CONDA: Condensed Deep Association Learning for Co-Salient Object Detection (https://arxiv.org/abs/2409.01021)
Comments:
          There is an error. In Sec 4.1, the number of images in some dataset is incorrect and needs to be revised

- **What's New**: 이번 연구에서는 co-salient object detection을 위한 효과적인 inter-image association modeling을 제안합니다. 기존 방법들의 한계를 극복하기 위해 raw associations을 변환하여 deep association features로 만드는 새로운 deep learning 전략을 도입했습니다.

- **Technical Details**: 제안된 방법은 먼저 hyperassociations을 생성하여 밀집한 pixel-pair-wise raw associations를 수집합니다. 그런 다음, 이를 기반으로 deep aggregation networks를 배치합니다. 이 과정에서 점진적인 association generation 모듈과 hyperassociation 계산의 향상된 버전이 설계되었습니다. 또한, semantic correspondence estimation을 위한 전제 과제를 도입하여 hyperassociations를 축약하는 association condensation 모듈을 제안하였습니다.

- **Performance Highlights**: 세 가지 벤치마크 데이터세트에서 실시한 실험 결과, 제안된 방법의 효과가 입증되었습니다. 다양한 훈련 설정에서도 뛰어난 성능을 보여주었습니다.



### Fed-MUnet: Multi-modal Federated Unet for Brain Tumor Segmentation (https://arxiv.org/abs/2409.01020)
Comments:
          6 pages, 3 figures, 2 tables. It was accepted by 2024 IEEE International Conference on E-health Networking, Application & Services (HealthCom)

- **What's New**: 본 연구에서는 다중 모달 MRI(실체 자석 영상)에서 뇌종양 분할을 위한 새로운 연합 학습(FL) 프레임워크인 Fed-MUnet을 제안합니다. 기존의 연구는 주로 단일 모달 MRI에 집중해왔으나, 본 논문은 다중 모달 MRI의 적용을 강조합니다.

- **Technical Details**: Fed-MUnet은 클라이언트-서버 아키텍처를 따르며, 클라이언트는 M-Unet을 사용하여 로컬 데이터셋을 훈련합니다. M-Unet은 U-Net의 이점을 활용하는 동시에 다중 모달 기능 융합이 가능합니다. 또한, 경량화된 교차 모달 모듈(CMM)을 설계하여 FL 환경에서의 오버피팅 문제를 완화하고 있습니다.

- **Performance Highlights**: BraTS2022 데이터셋을 평가한 결과, Fed-MUnet은 기존의 최상위 방법(SOTA)보다 향상된 성능을 보였습니다. 특히, 증가하는 종양, 종양 핵 및 전체 종양에 대한 5가지 주요 메트릭의 평균은 각각 87.5%, 90.6% 및 92.2%로 나타났습니다.



### From Bird's-Eye to Street View: Crafting Diverse and Condition-Aligned Images with Latent Diffusion Mod (https://arxiv.org/abs/2409.01014)
Comments:
          Accepted at International Conference on Robotics and Automation(ICRA)

- **What's New**: 본 논문은 Bird's-Eye View (BEV) 맵을 다중 시점 거리 이미지로 변환하는 방법을 탐구합니다. 이는 자율 주행 응용프로그램에서 중대한 역할을 하며, BEV 맵에서 정확한 거리 뷰 이미지를 생성하는 것이 복잡한 교통 시나리오를 보여주고 드라이빙 알고리즘을 향상시키는 데 필수적입니다.

- **Technical Details**: 우리는 Neural View Transformation과 Street Image Generation의 두 가지 주요 구성 요소로 이루어진 실용적인 프레임워크를 소개합니다. Neural View Transformation 단계에서는 BEV 맵을 다중 시점의 세분화 맵으로 변환하고, 이후 Street Image Generation 단계에서 이 세분화를 조건으로 하여 미세 조정된 latent diffusion model을 이용합니다. 이를 통해 다양한 시점과 스타일 일관성을 유지합니다.

- **Performance Highlights**: 우리 모델은 자율주행 관련 다양한 데이터를 통해 미세 조정된 large pretrained diffusion models의 생성 능력을 활용하여, 고품질의 다양하고 조건 일관성이 뛰어난 거리 뷰 이미지를 생성합니다. 실험 결과, 우리의 접근 방식은 기존 방법보다 시각적 품질과 조건 일관성 면에서 뛰어난 성과를 보입니다.



### Free-DyGS: Camera-Pose-Free Scene Reconstruction based on Gaussian Splatting for Dynamic Surgical Videos (https://arxiv.org/abs/2409.01003)
- **What's New**: 이 논문에서는 동적 외과 비디오를 위한 최초의 카메라 자세 자유 장면 재구성 프레임워크인 Free-DyGS를 제안합니다. 이 방법은 3D Gaussian splatting 기술을 활용하여 정밀한 재구성을 가능하게 합니다.

- **Technical Details**: Free-DyGS는 프레임별 재구성 전략을 사용하며, Scene Initialization, Joint Learning, Scene Expansion, Retrospective Learning의 4개 단계로 나뉩니다. Scene Initialization 및 Expansion 단계에서는 Generalizable Gaussians Parameterization 모듈을 도입하여 RGBD 프레임에서 각 픽셀의 Gaussian 속성을 생성합니다. Joint Learning 단계에서는 혁신적인 유연 변형 모듈을 통해 장면 변형 및 카메라 자세를 동시에 유추합니다. Scene Expansion 단계에서 Gaussian 포인트는 카메라가 이동함에 따라 점진적으로 성장합니다. Retrospective Learning 단계는 이전 프레임을 재평가하여 장면 변형의 정확도를 향상시키는 역할을 합니다.

- **Performance Highlights**: 실험 결과 Free-DyGS는 StereoMIS 및 Hamlyn 데이터셋에서 기존의 기준 모델들보다 재구성 품질과 계산 효율성에서 우수한 성능을 보였습니다.



### 3D Priors-Guided Diffusion for Blind Face Restoration (https://arxiv.org/abs/2409.00991)
- **What's New**: 본 논문에서는 3D 얼굴 구조를 디노이징(diffusion) 과정에 통합하여 새로운 디퓨전 기반 얼굴 복원(framework)을 제안합니다. 이를 통해 복원된 이미지의 사실성과 신원 일관성을 향상시키고자 합니다.

- **Technical Details**: 우리는 3D Morphable Model (3DMM)을 활용하여 복원된 초기 얼굴 이미지를 바탕으로 보다 정확한 3D 이전 정보를 재구성하며, 커스텀 다중 레벨 특성 추출 방법(multi-level feature extraction)을 사용하여 구조적 및 신원 정보를 추출합니다. 이 정보는 Time-Aware Fusion Block (TAFB)을 통해 노이즈 추정 과정에 통합됩니다.

- **Performance Highlights**: 제안된 네트워크는 합성 및 실제 데이터셋에서 blind face restoration과 관련하여 최신 알고리즘들과 비교하여 우수한 성능을 보임을 보여줍니다.



### Self-Supervised Multi-Scale Network for Blind Image Deblurring via Alternating Optimization (https://arxiv.org/abs/2409.00988)
Comments:
          21 pages, 17 figures, 94 references

- **What's New**: 이번 논문에서는 블라인드 이미지 디블러링(blind image deblurring) 문제를 해결하기 위한 자가 지도(self-supervised) 다중 스케일(multi-scale) 접근 방식을 제안합니다. 이 방법은 블러 커널(blur kernel)이 알려지지 않은 상황에서 잠재 이미지(latent image)와 블러 커널을 동시에 추정하는 데 중점을 두고 있습니다.

- **Technical Details**: 제안된 Self-MSNet은 이미지 피라미드(image pyramid)를 기반으로 여러 개의 입력 및 출력을 가진 생성기 네트워크(generator network)를 사용하여 다양한 스케일에서 잠재 이미지를 추정합니다. 블러 커널은 정규화된 최소 제곱(regularized least-squares) 방법을 통해 각 스케일에서 독립적으로 추정됩니다. 또한 전통적인 수학적 최적화 방식의 복잡한 연산을 피하는 이점이 있습니다.

- **Performance Highlights**: 자세한 실험 결과에 따르면, 본 방법은 합성(synthetic) 및 실제(real) 데이터셋에서 기존의 수학적 최적화 기반 방법이나 자가 지도 방식의 학습 기반 방법보다 더 우수한 성능을 발휘하며, 특히 큰 블러를 효과적으로 처리하는 데 강점을 보입니다.



### Personalized Lip Reading: Adapting to Your Unique Lip Movements with Vision and Languag (https://arxiv.org/abs/2409.00986)
Comments:
          Code available: this https URL

- **What's New**: 본 논문에서는 새로운 화자 적응형 립 리딩(Lip Reading) 방법을 제안하여 시각 및 언어 수준에서 목표 화자에게 적응하도록 하는 모델을 발전시켰습니다. 기존 연구들은 주로 시각 정보에만 초점을 맞췄으나, 이 방법은 화자의 언어적 패턴까지 고려한 혁신적인 접근 방식을 채택하고 있습니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 구성 요소로 이루어져 있습니다: 1) 화자 적응을 위한 시각 수준 적응은 사전 훈련된 립 리딩 모델을 목표 화자의 입 모양, 움직임 및 말하기 속도에 맞춰 조정하며, 2) 언어 수준 적응은 화자의 특정 언어 습관에 따라 사전 훈련된 모델을 수정하여 언어 모델링 확률을 학습하도록 설계되었습니다. 이 과정에서 패딩 프롬프트(padding prompts)와 Low-Rank Adaptation(LoRA)을 효과적으로 이용합니다.

- **Performance Highlights**: 제안된 방법은 새로운 VoxLRS-SA 데이터셋을 사용하여 실제 시나리오에서 검증되었으며, 기존 화자 적응형 접근 방식 대비 목표 화자에 맞춘 성능 개선을 입증하였습니다. 특히, 약 100K 어휘를 포함하고 다양한 자세 변화를 제공하는 이 데이터셋은 실제 환경에서 문장 수준 립 리딩 성능을 평가하는 데 중요한 기여를 하고 있습니다.



### GCCRR: A Short Sequence Gait Cycle Segmentation Method Based on Ear-Worn IMU (https://arxiv.org/abs/2409.00983)
Comments:
          Accepted by EarComp2024

- **What's New**: 이 논문은 귀에 착용하는 IMU(관성측정장치)를 사용하여 보행 주기(segmentation) 분할의 중요 작업에 다룹니다. 이는 환자의 운동 기능 저하를 위한 비침습적인 홈 모니터링 및 재활 접근 방식을 제공합니다.

- **Technical Details**: 저자들은 Gait Characteristic Curve Regression and Restoration (GCCRR)이라는 새로운 두 단계 접근 방식을 제안합니다. 첫 번째 단계에서는 Gait Characteristic Curve(GCC)라는 주기 정보를 포함한 일차원 특징 시퀀스에 대해 segmentation 작업을 회귀(regression) 작업으로 변환합니다. 두 번째 단계에서는 피크 검출 기법을 사용하여 보행 주기를 복원합니다. 이 방법은 짧은 보행 시퀀스에 대해 신뢰할 수 있는 segmentation을 보장하기 위해 Bi-LSTM 기반의 딥러닝 알고리즘을 사용합니다.

- **Performance Highlights**: HamlynGait 데이터셋에 대한 평가 결과, GCCRR은 80% 이상의 정확도(Accuracy)를 달성하며, 타임스탬프 오류(Timestamp Error)는 한 샘플링 간격 이하로 나타났습니다. 그러나 더 방대한 센서 시스템을 사용하는 방법에 비해 성능이 뒤처지는 점이 강조되었으며, 이는 더 크고 다양한 데이터셋의 필요성을 나타냅니다.



### IVGF: The Fusion-Guided Infrared and Visible General Framework (https://arxiv.org/abs/2409.00973)
Comments:
          11 pages, 8 figures

- **What's New**: 본 논문에서는 적외선(Infrarred) 및 가시광선(Visible) 이미지를 활용한 새로운 프레임워크 IVGF(Infrared and Visible General Framework)를 제안합니다. 이 프레임워크는 여러 고수준 비전(high-level vision) 작업에 쉽게 확장될 수 있으며, 기존의 이미지 퓨전 기반 방법이나 작업 특화 디자인 방법에 비해 일반화(generalization) 능력이 뛰어납니다.

- **Technical Details**: IVGF는 SOTA(State-of-the-Art) 적외선 및 가시광선 기초 모델들(foundation models)을 사용하여 일반 표현(general representations)을 추출합니다. 이는 기능 강화 모듈(feature enhancement module) 및 토큰 강화 모듈(token enhancement module)을 설계하여 고급 비전 작업을 위한 의미론적 정보(semantic information)를 풍부하게 합니다. 또한, 두 가지 모달리티 간의 보완적(complementary) 정보를 효과적으로 융합하기 위해 주의 기반 융합 모듈(attention-guided fusion module)을 제안합니다.

- **Performance Highlights**: IVGF는 세그멘테이션(semantic segmentation) 및 물체 탐지(object detection) 작업에서 최첨단 이중 모달리티 방법에 비해 우수한 성능을 보여줍니다. 실험 결과에 따르면, 각 모듈의 효과도 입증되었으며, 제안된 방법은 이중 모달리티 세그멘테이션 작업에서 결측 모달리티에 대한 강인함을 검토하는 추가 실험을 통해 그 능력을 확인했습니다.



### Interpretable Convolutional SyncN (https://arxiv.org/abs/2409.00971)
Comments:
          8+5 pages

- **What's New**: 이번 연구에서는 기존의 Sync-Net보다 더 해석 가능한 새로운 컨볼루션 기반의 Sync-Net(IC-SyncNet)을 제안합니다. 우리는 Balanced BCE (BBCE) 손실 함수를 사용하여, 대규모 이미지 처리를 지원하며, 출력을 확률적으로 해석할 수 있습니다.

- **Technical Details**: IC-SyncNet은 128x256 크기의 이미지를 5장과 32x80 크기의 Mel 스펙트로그램을 1장 입력으로 받아들이며, 40.6M의 파라미터를 가지고 있습니다. BBCE 손실 함수는 기존의 InfoNCE 손실과 비슷한 성능을 보여주지만, 복잡한 샘플링 기법이 필요하지 않습니다.

- **Performance Highlights**: LRS2 데이터셋에서 96.5%의 정확도를, LRS3 데이터셋에서 93.8%의 정확도를 달성하여 새로운 SOTA(State Of The Art) 성능을 기록했습니다. 또한, 비디오의 싱크 품질을 평가하기 위한 새로운 메트릭으로 오프셋(offset), 오프셋에서의 확률(probability at offset) 및 오프스크린 비율(offscreen ratio)을 제안했습니다.



### PNVC: Towards Practical INR-based Video Compression (https://arxiv.org/abs/2409.00953)
- **What's New**: 이 논문에서는 새로운 구조의 재매개변수화 기반 아키텍처와 적층 품질 제어, 변조 기반 엔트로피 모델링, 스케일-aware 위치 임베딩(scale-aware positional embedding) 등의 혁신적인 설계를 통해 PNVC라는 실용적인 신경 비디오 코덱을 제안합니다.

- **Technical Details**: PNVC는 토대 위에 구조적인 재매개변수화 메소드를 통해 저지연(LD) 및 랜덤 접근(RA) 설정을 지원하며, 기존의 INR 기반 코덱을 크게 능가하는 성능을 보입니다. 또한, 일반적인 오토인코더 기반 방식보다 낮은 복잡성으로 효율적인 인퍼런스를 가능하게 합니다.

- **Performance Highlights**: PNVC는 HEVC HM 18.0 (LD) 대비 35% 이상의 BD-rate 절감을 기록하며, HiNeRV와 비교하여 거의 10% 더 높은 성능을 보였습니다. 1080p 콘텐츠에 대해 20+ FPS의 디코딩 속도를 유지하며, 이는 INR 기반 비디오 코딩 분야에서 실용적 적용을 위한 중요한 진전을 나타냅니다.



### XNet v2: Fewer Limitations, Better Results and Greater Universality (https://arxiv.org/abs/2409.00947)
- **What's New**: XNet v2는 저주파 (Low Frequency, LF) 및 고주파 (High Frequency, HF) 보완 모델로, 이미지 수준의 보완 융합을 수행하며 원본 이미지 입력과 함께 세 개의 서로 다른 서브 네트워크를 사용하여 일관성 손실 (consistency loss)을 구성합니다.

- **Technical Details**: XNet v2는 주 네트워크와 LF 및 HF 네트워크로 구성되어 있으며, 이미지 수준 및 특성 수준에서의 융합 모듈을 도입하여 LF 및 HF 정보를 효과적으로 전달합니다. 주 네트워크는 UNet 구조를 기반으로 하며, LF 및 HF 출력으로 일관성 손실을 구성하여 체크포인트 기반의 학습을 지원합니다.

- **Performance Highlights**: XNet v2는 세미-슈퍼바이즈드 (semi-supervised) 분할에서 최신의 성능을 달성했으며, 풀리-슈퍼바이즈드 (fully-supervised) 학습에서도 경쟁력 있는 결과를 유지합니다. 특히, XNet이 실패하는 시나리오에서도 더욱 뛰어난 성능을 보이며, 3개의 2D 및 2개의 3D 데이터셋에서 효과성을 입증하였습니다.



### VQ-Flow: Taming Normalizing Flows for Multi-Class Anomaly Detection via Hierarchical Vector Quantization (https://arxiv.org/abs/2409.00942)
- **What's New**: 본 연구에서는 Normalizing Flows를 활용하여 다중 클래스 이상 탐지(Multi-Class Anomaly Detection)의 가능성을 모색하고, 이를 위해 벡터 양자화(Vector Quantization)를 통합한 새로운 방법 VQ-Flow를 제안합니다. VQ-Flow는 클래스 레이블 없이도 정상 데이터를 구분할 수 있도록 설계되었습니다.

- **Technical Details**: VQ-Flow는 계층적 벡터 양자화를 기반으로 하는 두 개의 코드북, 즉 개념 프로토타입 코드북(Conceptual Prototype Codebook, CPC)과 개념별 패턴 코드북(Concept-Specific Pattern Codebook, CSPC)을 활용합니다. 이를 통해 서로 다른 개념의 정상 패턴들을 추정하고 모델링하여, 다중 클래스 정상 분포를 혼합 가우시안 분포를 사용해 모사합니다.

- **Performance Highlights**: VQ-Flow는 MVTec AD 데이터셋에서 탐지 및 위치 지정에서 각각 99.5%와 98.3%의 AUROC를 기록하며 최첨단 성능을 자랑합니다. 또한 CIFAR-10 데이터셋에서도 이전 통합 방법들을 초월하는 성과를 거두었습니다.



### Towards Student Actions in Classroom Scenes: New Dataset and Baselin (https://arxiv.org/abs/2409.00926)
- **What's New**: 이 논문은 복잡한 교실 장면에서 학생 행동을 분석하기 위한 새로운 다중 레이블 학생 행동 비디오(SAV) 데이터셋을 소개합니다. 758개 교실에서 수집된 4,324개의 비디오 클립을 포함하며, 각각은 교실에서 학생들이 나타내는 15가지 행동으로 레이블이 붙어 있습니다.

- **Technical Details**: SAV 데이터셋은 실제 교실 장면을 광범위하게 반영하고 있으며, 720P 및 1080P의 고해상도 비디오 데이터를 제공하여 시각적 정보를 상세히 보존합니다. 이 데이터셋의 복잡성은 미세한 행동 변화, 밀집된 객체, 다양한 촬영 각도 및 시각적 장애물 등 여러 도전 과제를 포함합니다. 또한, 수정된 비주얼 트랜스포머(ViT) 구조를 기반으로 한 새로운 기준 방법을 제안하여 학생 행동 감지의 성능을 향상시킵니다.

- **Performance Highlights**: 제안된 방법은 SAV 데이터셋에서 67.9%의 평균 정확도(mAP)를 달성하며, 이는 현재의 최첨단 방법에 필적하는 성능입니다. 이를 통해 학생 행동 분석을 위한 새로운 기회와 도전 과제를 제시합니다.



### MedSAM-U: Uncertainty-Guided Auto Multi-Prompt Adaptation for Reliable MedSAM (https://arxiv.org/abs/2409.00924)
Comments:
          10 pages, 4 figures

- **What's New**: 이 논문에서는 MedSAM의 정확도를 향상시키기 위해 신뢰성 있는 프롬프트 시스템을 개발하는 데 초점을 맞추고 있습니다. 새로운 프레임워크인 MedSAM-U를 소개하여 자동으로 다중 프롬프트 입력을 세밀하게 조정합니다.

- **Technical Details**: MedSAM-U는 Uncertainty-Guided Multi-Prompt (UGMP) 방식과 Uncertainty-Guided Prompt Adaptation (UGPA) 기술을 통해 의료 이미지 분할의 신뢰성을 높입니다. 특히, 다양한 프롬프트에 대한 불확실성을 추정하고 이를 기반으로 신뢰할 만한 프롬프트를 자동으로 생성하는 방법을 사용합니다.

- **Performance Highlights**: MedSAM-U는 다섯 가지 서로 다른 모드의 데이터셋에 대한 실험에서 평균 1.7%에서 20.5%까지 성능 향상을 달성했습니다. 이는 MedSAM에 비해 분명히 개선된 결과입니다.



### Large Scale Unsupervised Brain MRI Image Registration Solution for Learn2Reg 2024 (https://arxiv.org/abs/2409.00917)
Comments:
          MICCAI Learn2Reg 2024 Challenge & WBIR 2024 Workshop on Biomedical Imaging Registration

- **What's New**: 이 논문에서는 learn2reg 2024 챌린지에서 제안된 방법과 실험 결과를 요약합니다. 특히 뇌 MRI 이미지의 해부학적 구조에 대한 비지도(unsupervised) 등록이 주요 과제입니다. 77.34%의 Dice 계수를 달성하여 TransMorph보다 1.4% 높은 결과를 보였으며, 리더보드에서 2위를 차지했습니다.

- **Technical Details**: 효율적인 등록 성능을 위해 강력한 병렬 네트워크(backbone network)를 구축했습니다. 이 네트워크는 Co-Attention Mechanism(CA), Large Kernel Convolution(LK), Bilateral Filtering(BF), Large Convolution Channels(LC)과 같은 다양한 혁신적 요소를 통합하여 정확도를 높였습니다. 손실 함수는 비슷함 손실(NCC)과 정규화 손실(reg)로 구성되어 있습니다.

- **Performance Highlights**: 우리는 OpenBHB 데이터세트를 사용하여 모델을 훈련시켰으며, 비지도 훈련으로 3374개의 샘플을 이용했습니다. 결과적으로, 당사 방법은 Dice 계수에서 1.4% 개선, 비가역적인 볼륨(NDV)에서 0.3508 감소, 최대 표면 거리(HD95)에서 0.1659 개선을 보였습니다. 논문에 제시된 결과는 다양한 네트워크 통합의 성능 향상을 명확히 보여줍니다.



### Merging Multiple Datasets for Improved Appearance-Based Gaze Estimation (https://arxiv.org/abs/2409.00912)
Comments:
          14 pages

- **What's New**: 이 논문에서는 다수의 데이터셋을 활용하여 시선 추정(gaze estimation) 성능을 개선하기 위한 두 가지 혁신적인 방법을 제안합니다. 첫 번째는 Two-stage Transformer 기반의 Gaze-feature Fusion (TTGF) 아키텍처 도입이며, 두 번째는 Gaze Adaptation Module (GAM) 입니다.

- **Technical Details**: 제안된 TTGF 방법은 각 눈과 얼굴의 정보를 별도로 병합한 후 양쪽 눈의 정보를 통합하는 방식을 사용합니다. 이는 머리 자세(head pose)가 변할 때 두 눈 이미지에 미치는 영향을 고려하여 정확한 추정을 가능하게 합니다. GAM은 각 데이터셋의 주석 불일치를 조정하여 단일 추정기를 통해 시선 추정값을 수정합니다.

- **Performance Highlights**: 이러한 혁신적인 접근법들은 기존 SOTA(state-of-the-art) 방법에 비해 10%에서 20%까지 성능 개선을 보였습니다.



### ViRED: Prediction of Visual Relations in Engineering Drawings (https://arxiv.org/abs/2409.00909)
Comments:
          8 pages, 5 figures

- **What's New**: 본 논문에서는 ViRED라는 새로운 비전 기반의 관계 탐지 모델을 제안하여 전기 공학 도면에서 표와 회로 간의 관계를 식별함으로써 기존 방법론보다 높은 정확도를 달성했습니다.

- **Technical Details**: ViRED 모델은 비전 인코더(vision encoder), 객체 인코더(object encoder), 관계 디코더(relation decoder)로 구성됩니다. PyTorch를 사용하여 구현되었으며, 전기 공학 도면 데이터셋을 통해 관계 예측 작업에서 96%의 정확도를 기록했습니다.

- **Performance Highlights**: 시행한 여러 실험 결과, ViRED는 단일 전기 공학 도면에 많은 객체가 포함되어 있는 경우에도 빠른 속도로 추론할 수 있음을 보여줍니다.



### Multi-scale Temporal Fusion Transformer for Incomplete Vehicle Trajectory Prediction (https://arxiv.org/abs/2409.00904)
- **What's New**: 이 논문은 결측값이 있는 차량 궤적 예측을 해결하기 위한 새로운 엔드 투 엔드 프레임워크인 Multi-scale Temporal Fusion Transformer (MTFT)를 제안합니다. 이 모델은 Multi-scale Attention Head (MAH)와 Continuity Representation-guided Multi-scale Fusion (CRMF) 모듈로 구성되어 있습니다.

- **Technical Details**: MTFT 프레임워크는 다중 시간 규모에서 궤적의 모션 표현을 캡처하고 융합하여 예측의 질을 향상시킵니다. MAH는 다중 헤드 어텐션 메커니즘을 활용하여 서로 다른 시간 단위에서 궤적의 모션 표현을 병렬로 캡처하며, CRMF 모듈은 이러한 표현을 융합하여 강력한 시간 특징을 생성합니다. 이 과정에서 궤적의 연속 표현이 추출되어 융합을 안내합니다.

- **Performance Highlights**: 실험 결과, MTFT는 HighD 데이터셋에서 39% 이상의 성능 향상을 보이며 기존의 최신 모델에 비해 뛰어난 성능을 보여줍니다. 또한, 다양한 교통 시나리오에서 네 가지 데이터셋을 평가하여 이 모델의 유효성을 입증했습니다.



### MV-Match: Multi-View Matching for Domain-Adaptive Identification of Plant Nutrient Deficiencies (https://arxiv.org/abs/2409.00903)
Comments:
          BMVC 2024 camera-ready version

- **What's New**: 이번 연구에서는 무관 supervised domain adaptation에 대한 새로운 접근법을 제안했습니다. 특히, 여러 카메라 뷰를 활용하여 라벨이 있는 소스 도메인과 라벨이 없는 타겟 도메인 간의 도메인 적응을 개선하고자 하였습니다.

- **Technical Details**: 제안된 방법인 Multi-View Match (MV-Match)는 다수의 뷰에서 예측의 일관성을 유지하도록 강제합니다. 또한, 유사성 기반 뷰 마이닝(Similarity-guided View Mining, SgVM) 메커니즘을 도입하여 쿼리 이미지에 대해 상호 보완 정보를 포함하는 가장 비슷하지 않은 뷰들을 자동으로 선택합니다.

- **Performance Highlights**: 이 방법은 두 개의 영양 결핍 데이터 세트에서 평가되었으며, 기존의 다른 무관 supervised domain adaptation 방법들과 비교했을 때 최고 성능을 달성하였습니다.



### A Noise and Edge extraction-based dual-branch method for Shallowfake and Deepfake Localization (https://arxiv.org/abs/2409.00896)
- **What's New**: 이 논문에서는 dual-branch 모델을 개발하여 수동으로 설계된 feature noise와 CNN 특성을 결합하여 이미지 조작(Localization)을 위한 새로운 접근 방식을 제안합니다.

- **Technical Details**: 제안된 모델은 두 개의 가지(branch)로 구성되어 있습니다. 하나의 가지는 feature noise의 특성을 통합하고, 다른 가지는 계층적 ConvNext 모듈을 사용하여 RGB 특성을 통합합니다. 또한, edge supervision loss를 활용하여 조작 경계 정보를 정확하게 획득합니다.

- **Performance Highlights**: 모델은 shallowfakes 데이터셋(CASIA, COVERAGE, COLUMBIA, NIST16)과 deepfake 데이터셋(Faceforensics++)에서 테스트되어, 99%의 AUC 점수를 기록하여 기존의 SoTA 모델보다 우수한 성능을 보였습니다.



### Digital Twins in Additive Manufacturing: A Systematic Review (https://arxiv.org/abs/2409.00877)
- **What's New**: 이 논문은 Additive Manufacturing (AM) 분야에서 Digital Twins (DTs)의 응용에 대한 포괄적인 개요를 제공합니다. DTs를 활용하여 생산 과정의 효율성을 향상시키고, 커스터마이징 및 스케일링을 지원할 수 있는 가능성을 탐구합니다.

- **Technical Details**: DTs는 Machine Learning (ML), Augmented Reality (AR), 시뮬레이션 기반 모델 등을 활용해 AM 프로세스에서 지능적이고 적응 가능한 디지털 복제를 생성합니다. DT는 IoT 데이터와 센서를 통해 실시간으로 가상 모델을 업데이트하며, 이는 성능 최적화 및 다운타임 감소에 기여합니다.

- **Performance Highlights**: DT는 AM에서 실시간 모니터링, 시뮬레이션 및 최적화를 통해 제조업체가 잠재적인 문제를 사전에 예측하고 해결할 수 있도록 돕습니다. DT는 가상 테스트 및 검증을 통해 연구 개발을 가속화하며, 예측 유지를 통해 기계의 효율적인 운영을 지원합니다.



### Equitable Skin Disease Prediction Using Transfer Learning and Domain Adaptation (https://arxiv.org/abs/2409.00873)
- **What's New**: 이 연구에서는 피부 상태 진단의 정확성을 향상시키기 위해 이전의 AI 모델들이 가진 편향성을 극복하는 방법을 제안합니다. 특히 다양한 피부 톤을 고려한 데이터셋(Diverse Dermatology Images, DDI)을 활용하여, AI의 진단 성능을 높이는 데 기여합니다. 오히려 기존 모델에서 발생했던 어두운 피부 색조에 대한 성능 저하 문제를 해결하기 위한 접근 방식으로, 이전의 광범위한 이미지 도메인에서의 지식을 전이(transfer learning)를 활용합니다.

- **Technical Details**: 이 연구에서 사용된 몇 가지 모델은 Med-ViT, YOLOv8-Chest, RETFound로, 각 모델은 다양한 의료 이미지로부터 학습된 특징을 활용합니다. 특히 Vision Transformer 기반의 RETFound는 여러 도메인에서 전이된 특성을 활용하여 피부 질환 분류의 성능을 갖추고 있습니다. 모델은 DDI 데이터셋 뿐만 아니라 HAM10000과 같은 추가 피부 이미지 데이터셋으로 도메인 적응(domain adaptation)을 진행하여 성능을 더욱 향상시킵니다.

- **Performance Highlights**: 모델의 성능 평가 결과, Med-ViT가 다른 모델들 중에서 가장 뛰어난 성과를 보였습니다. 연구 결과는 다양한 피부 톤에서의 진단 성능을 높은 수준으로 끌어올리며, AI 도구의 포용성과 정확성을 높이는 데 기여할 것입니다. 이 연구는 특히 과소대표된 피부 톤 시나리오에서 전이 학습과 도메인 적응 기술이 어떻게 성능을 향상시킬 수 있는지를 보여줍니다.



### Detection, Recognition and Pose Estimation of Tabletop Objects (https://arxiv.org/abs/2409.00869)
- **What's New**: 이번 연구는 Deep Neural Networks를 활용하여 혼잡한 테이블을 청소하는 문제를 다룹니다. 일상 생활에서 로봇이 사람과 함께 작업하는 'Social Robotics'의 영역에서 이 기술의 사회적 응용에 중점을 둡니다.

- **Technical Details**: 제안된 모델은 mug, mouse, stapler와 같은 일반적인 테이블 객체를 감지하고 인식하며, 이러한 객체가 테이블에 놓인 각도를 예측합니다. 객체는 고정된 위치에 있어야 하며, 딥러닝 모델의 예측된 방향을 통해 변환 행렬을 계산하여 로봇이 해당 객체를 지정된 위치로 옮길 수 있도록 합니다.

- **Performance Highlights**: 연구에서 사용된 'Tabletop' 데이터셋은 각 객체에 대해 10개 인스턴스와 16개의 다른 포즈로 사진이 촬영되었습니다. 이 데이터는 gray-scaled 형식으로, 객체의 배경을 제거하기 위해 마스크를 사용하여 전처리되었습니다. 실험 결과는 로봇이 혼잡한 테이블 환경에서 객체를 인식하고 배치하는 데 있어 높은 정확도를 보여줍니다.



### Fisher Information guided Purification against Backdoor Attacks (https://arxiv.org/abs/2409.00863)
Comments:
          Accepted to ACM CCS 2024. arXiv admin note: text overlap with arXiv:2306.17441

- **What's New**: 최근 연구에 따르면, 적대자가 소량의 훈련 샘플을 조작하여 딥 뉴럴 네트워크(DNN)의 무결성을 손상시킬 수 있다는 것입니다. 본 논문에서는 네트워크가 나쁜 로컬 미니마(local minima)에 수렴하게 만드는 백도어 공격의 새로운 관점을 소개합니다. 이를 해결하기 위해 Fisher Information에 기반한 새로운 백도어 정화 프레임워크인 FIP를 제안합니다.

- **Technical Details**: 제안된 FIP는 DNN의 Fisher Information Matrix(FIM)를 활용해 백도어의 영향을 억제하고, 청정 데이터 분포의 지식을 보존하며 블랙박스 공격을 제거하는 데 도움을 주는 몇 가지 새로운 규제기를 포함합니다. 이는 혼합(minimization) 과정을 통해 최적화되고, FIP의 변형인 Fast FIP를 통해 파라미터 조정을 대폭 줄이고 약 다섯 배의 성능 향상을 달성합니다.

- **Performance Highlights**: 제안된 방법은 Image Recognition, Object Detection, Video Action Recognition 등 다양한 백도어 방어 벤치마크에서 최첨단 성능(SOTA)을 달성했습니다. 11개의 데이터 세트와 14가지 다른 백도어 공격을 포함한 폭넓은 실험을 통해 입증되었습니다.



### Image-to-Lidar Relational Distillation for Autonomous Driving Data (https://arxiv.org/abs/2409.00845)
Comments:
          Accepted at ECCV 2024

- **What's New**: 본 연구에서는 2D-to-3D distillation (증류) 프레임워크의 구조적 차이로 인해 발생하는 문제를 조사하고, 이를 해결하기 위한 relational distillation framework (관계성 증류 프레임워크)를 제안합니다. 이 프레임워크는 intra-modal (내부 모달) 및 cross-modal (교차 모달) 제약을 적용하여 3D 표현이 2D 표현의 구조를 효과적으로 캐치하도록 합니다.

- **Technical Details**: 연구에서는 contrastive loss (대조 손실) 및 similarity loss (유사도 손실)를 사용한 증류로 인해 발생하는 2D와 3D 표현의 구조적 불일치를 수량화하였습니다. 이를 위해 uniformity (균일성), tolerance (허용 오차), modality gap (모달리티 간의 차이) 등의 지표를 사용합니다. 제안된 relational losses는 2D 학습의 구조적 제약과 잘 일치하며, zero-shot (제로샷) 및 few-shot (소수샷) 세분화 성능을 개선합니다. 

- **Performance Highlights**: 제안된 관계성 손실을 통해, 3D 표현의 품질이 개선되었으며, contrastive distillation을 통해 학습된 모델에 비해 zero-shot 세분화 작업에서 현저히 높은 성능을 보였습니다. 또한, in-distribution (분포 내) 및 out-of-distribution (분포 외) few-shot 세분화 작업에서도 일관된 성능 향상을 달성했습니다.



### Entropy Loss: An Interpretability Amplifier of 3D Object Detection Network for Intelligent Driving (https://arxiv.org/abs/2409.00839)
- **What's New**: 이 논문에서는 인공지능 주행에서의 안전 인식의 중요성이 높아짐에 따라 기존의 딥러닝 방식의 한계를 극복하는 새로운 단위 손실 함수인 ‘Entropy Loss’를 제안합니다. 이 손실 함수는 기능 압축 네트워크의 특징을 바탕으로 설계되었습니다.

- **Technical Details**: Entropy Loss는 정보 전송 프로세스의 정밀한 변화를 모델링하여, 각 레이어의 출력과 정보량을 연속 랜덤 변수로 표현함으로써 정보 엔트로피의 변화를 정량화합니다. 이를 통해 네트워크 파라미터 업데이트가 가능해지며, 해석 가능성을 개선합니다.

- **Performance Highlights**: 실험 결과, Entropy Loss 사용한 3D 객체 탐지 모델의 KITTI 테스트 세트에서의 정확도가 비사용 모델에 비해 최대 4.47% 향상되어 학습 과정이 가속화되었음을 보여줍니다.



### Curvy: A Parametric Cross-section based Surface Reconstruction (https://arxiv.org/abs/2409.00829)
- **What's New**: 새로운 접근법으로, planar sparse cross-sections를 사용하여 생성 모델링(generative modeling)의 도움으로 shape point cloud를 재구성하는 방법을 제안합니다. 기존 문헌의 대부분은 객체 클래스(object class)에 따라 일반화할 수 있는 능력이 부족하며, 신뢰할 수 있는 표면을 재구성하기 위해 복잡한 수학적 기법을 사용합니다. 이 논문에서는 적은 수의 cross-section으로부터 많은 점을 생성하는 간단한 학습 가능 접근법을 소개합니다.

- **Technical Details**: 자세한 기술적 세부 사항으로는, parametric polyline representation을 사용하여 cross-section을 표현하고, Graph Neural Network를 통해 적응 방식으로 기저 형태를 재구성합니다. 이러한 방법은 제공된 cross-section 수에 대한 의존성을 줄이고, cross-section의 희소성 및 비등방성(anisotropic) 특성을 고려하여 포인트 클라우드(point cloud)를 생성합니다.

- **Performance Highlights**: 성능 강조점으로는 두 가지 주목(attention) 메커니즘을 도입하여 cross-section의 지역(local) 및 전역(global) 구조에 집중하며, ablation study를 통해 그 중요성을 실증적으로 보여주는 것이 포함됩니다. 또한 새로운 cross-section의 파라메트릭 표현을 위한 데이터셋(dataset)을 제공하여 연구에 기여합니다.



### Real-Time Weather Image Classification with SVM (https://arxiv.org/abs/2409.00821)
- **What's New**: 이 논문은 영상을 네 가지 날씨 조건(비, 저조도, 안개, 맑음)으로 분류하는 새로운 SVM 기반 날씨 분류 알고리즘을 제안하며, 이를 통해 자동화된 시스템의 신뢰성과 효율성을 향상시키고자 합니다.

- **Technical Details**: 제안된 접근법은 Local Binary Patterns (LBP)을 포함한 다양한 피처(특징)를 사용합니다. 이 피처들은 이미지의 밝기, 포화도, 노이즈 레벨, 블러 측정, 엣지 강도, 그리고 색상 히스토그램 평균 및 분산을 포함합니다. SVM(Support Vector Machine) 알고리즘을 사용하여, 훈련된 모델은 데이터의 특정 피처를 바탕으로 날씨 조건을 분류하는 데에 효과적으로 작동합니다.

- **Performance Highlights**: SVM 기반 방법은 92.8%의 정확도를 달성하였으며, 기존의 전통적 기계 학습 방법의 벤치마크인 80%에서 90%를 초과했습니다. 깊이 학습 방법들과 비교하였을 때, 계산 효율성과 실시간 분류 능력에서 경쟁력을 보여주었습니다.



### Diffusion based multi-domain neuroimaging harmonization method with preservation of anatomical details (https://arxiv.org/abs/2409.00807)
- **What's New**: 이 논문에서는 다중 센터의 신경촬영(neuroimaging) 연구에서 발생하는 기술적 변동성(batch effect)에 대한 새로운 접근 방식을 제시합니다. 기존의 Generative Adversarial Networks (GAN) 기반의 화합(harmonization) 방법에 비해, Denoising Diffusion Probabilistic Model을 활용하여 고온송 취소 효과(generated artifacts)나 해부학적 왜곡(anatomical distortion)을 최소화하는 방법이 소개됩니다.

- **Technical Details**: Multi-domain neuroimaging harmonization을 위한 새로운 접근 방식을 제안하며, 이는 학습된 도메인 불변 조건(domain invariant condition)을 통해 이루어집니다. Denoising diffusion 모델을 이용하여 각 도메인 간의 변동성을 효과적으로 억제하고, 해부학적 세부 사항(anatomical details)을 보존하는 데 중점을 둡니다. 제안된 모델은 각 확산 단계에서 배치 간의 차이를 구분하며, 이를 통해 더 높은 화합 결과를 얻습니다.

- **Performance Highlights**: ADNI1 및 ABIDE II의 두 공공 데이터셋을 활용하여 GAN 기반 방법보다 뛰어난 화합 결과를 도출했습니다. PVS(Perivascular Spaces) 세분화(segmentation) 분석에 있어서도 스캐너 효과의 일관성을 개선하는 결과를 보여주었습니다. 최신 진행된 연구에서 제안하는 모델은 GAN 기반 방법에 비해 휘도(brightness)와 세부 사항에서 더 뛰어난 성능을 보였습니다.



### Zero-Shot Paragraph-level Handwriting Imitation with Latent Diffusion Models (https://arxiv.org/abs/2409.00786)
- **What's New**: 이 연구에서는 커서( cursive ) 손글씨의 모방을 발전시키기 위해 문단 수준에서의 손글씨 생성을 제안합니다. 기존의 방법들에서 벗어나, 한 페이지에 걸쳐 일관성을 유지하도록 한 새로운 접근법을 개발하였습니다.

- **Technical Details**: 본 연구에서는 수정된 latent diffusion model 을 도입하여 인코더-디코더 메커니즘을 보강하고, 스타일(style)과 내용을 명시적으로 유지하는 전문화된 손실 함수(loss functions)를 제공합니다. 또한, 확산 모델의 attention mechanism 을 향상시키기 위해 adaptive 2D positional encoding 과 conditioning mechanism 을 사용하여 스타일 이미지(style image)와 목표 텍스트(target text) 두 가지 모달리티(modality) 간의 작업을 가능하게 합니다.

- **Performance Highlights**: 우리의 접근법은 종합 평가에서 새로운 기준을 설정하였으며, 스타일(style)과 내용(content) 보존을 동시에 고려했을 때, 기존의 모든 손글씨 모방(imitation) 방법을 초월하여 줄(line) 및 문단(paragraph) 수준 모두에서 우수한 성능을 보였습니다.



### Unbalanced Fingerprint Classification for Hybrid Fingerprint Orientation Maps (https://arxiv.org/abs/2409.00779)
Comments:
          10 pages, 18 figures, 4 Tables The work mainly focuses on fingerprint classification and hybrid fingerprint orientation map (HFOM) generation. It highlights the security use cases of HFOM, eg. data encryption

- **What's New**: 이 논문은 다층 퍼지 논리 분류기를 기반으로 한 새로운 지문 분류 기법을 소개합니다. 우리는 드라이, 스탠다드, 웻과 같은 지문을 초기 단계에서 식별하여 탐지 미비의 원인을 목표로 하고 있습니다.

- **Technical Details**: 제안된 방법은 특성 포인트(feature points)와 연관된 선명도를 기반으로 스캔된 이미지를 분류합니다. 또한, 다중 클래스 불균형을 극복하기 위해 고유 벡터 공간(eigenvector space)을 기반으로 새로운 샘플을 생성하는 적응형 알고리즘을 제안합니다. 퍼지 논리(fuzzy logic)를 사용해 지문 이미지를 분류하는 UC-FLEM 기법을 사용하며, HFOM(하이브리드 지문 방향 맵)을 생성하는 새로운 방법론도 포함되어 있습니다.

- **Performance Highlights**: 이 새로운 접근법은 신경망 기반 분류 방법들보다 더 나은 성능을 발휘하는 것으로 나타났습니다. 퍼지 논리와 다층 구조를 통해 지문 분류의 성능을 향상시키고, HFOM은 지문 정보의 보안을 강화하는 새로운 사용 사례를 창출합니다.



### VDPI: Video Deblurring with Pseudo-inverse Modeling (https://arxiv.org/abs/2409.00777)
- **What's New**: 본 논문은 이미지 형성 모델에 대한 지식을 딥러닝 네트워크에 도입하여 비디오 디블러링(video deblurring) 성능을 향상시키는 새로운 방법을 제안합니다.

- **Technical Details**: 이 방법은 블러(blurry)의 의사 역행렬(pseudo-inverse)을 사용하여 딥 네트워크가 블러를 적합하고 의사 역행렬을 추정하게 하는 방식을 채택합니다. 이후, 이 추정 값과 변분적 딥러닝 네트워크(variational deep learning network)를 결합하여 비디오 시퀀스를 디블러링(deblurring)합니다.

- **Performance Highlights**: 실험 결과, 이러한 수정이 딥러닝 모델의 비디오 디블러링 성능을 현저하게 개선함을 보여줍니다. 다양한 데이터셋에서의 실험 또한 탁월한 성능 개선을 이루었으며, 제안된 방법이 다양한 시나리오와 카메라에 일반화될 수 있음을 입증하였습니다.



### SITUATE: Indoor Human Trajectory Prediction through Geometric Features and Self-Supervised Vision Representation (https://arxiv.org/abs/2409.00774)
Comments:
          Accepted at the 27th International Conference on Pattern Recognition (ICPR 2024)

- **What's New**: 이 논문은 SITUATE라는 새로운 모델을 제안합니다. SITUATE는 실내 인간 이동 경로 예측을 위해 동형(homogeneous) 및 불변(invariant) 기하학적 특징 학습 모듈과 자기 지도적(self-supervised) 비전 표현을 활용합니다.

- **Technical Details**: SITUATE는 실내 환경의 고유한 대칭성과 인간의 움직임을 모델링하여, 다양한 스케일 및 방향 전환을 포괄하는 자기 루프를 다루는 기하학 학습 모듈을 포함합니다. 또한, 공간-의미적 정보(spatial-semantic information)를 얻기 위한 비전 표현 모듈을 사용하여 사용자의 향후 위치를 보다 정확하게 예측합니다.

- **Performance Highlights**: 이 방법은 THÖR 및 Supermarket 데이터셋에서 상태-of-the-art(state-of-the-art) 성능을 달성하였으며, 실내 중심 예측 모델이 실외 중심 예측 모델보다 일반화 능력이 더 뛰어남을 보여주었습니다.



### Rethinking Image Super-Resolution from Training Data Perspectives (https://arxiv.org/abs/2409.00768)
Comments:
          Accepted to ECCV2024

- **What's New**: 본 연구에서는 이미지 초해상도(image super-resolution, SR) 모델의 훈련 데이터의 다양성과 품질이 SR 성능에 미치는 영향을 재조명합니다. 특히, 기존의 고해상도 이미지 데이터셋에 대한 의존을 줄이고, 저해상도 이미지 데이터셋인 DiverSeg를 제안하여 조정된 데이터셋이 SR 모델 훈련에 효과적임을 보입니다.

- **Technical Details**: DiverSeg 데이터셋은 고해상도 이미지(HD, 2K, 4K) 대신, 고품질을 지닌 저해상도 이미지로 구성됩니다. 특히, 블록성 아티팩트(blockiness artifacts)에 대한 커널 밀도 추정(kernel density estimation)을 통해 이미지 품질을 추정하여, 특이적으로 과도한 압축 아티팩트가 적은 이미지를 선택합니다. 또한 객체 수(object count)가 많은 이미지에서 SR 모델 성능 향상을 실증적으로 보여줍니다.

- **Performance Highlights**: DiverSeg 데이터셋으로 훈련된 SR 모델은 DF2K 및 LSDIR와 같은 기존의 고해상도 이미지 데이터셋으로 훈련된 모델보다 더 우수한 성능을 기록했습니다. 이는 저해상도 데이터도 적절히 활용하여 경쟁력 있는 SR 모델을 구축할 수 있음을示합니다.



### Trusted Unified Feature-Neighborhood Dynamics for Multi-View Classification (https://arxiv.org/abs/2409.00755)
Comments:
          Ongoing work: 13pages, 13figures, 12 tables

- **What's New**: 새로운 연구에서는 Trusted Unified Feature-NEighborhood Dynamics (TUNED) 모델을 제안하여 다중 시점 분류(Multi-view Classification, MVC)에서의 불확실성과 갈등을 효과적으로 해결합니다. 이 모델은 지역과 글로벌 피처-이웃 구조를 통합하여 보다 강력한 의사결정을 수행합니다.

- **Technical Details**: TUNED 모델은 각 뷰 내에서의 지역 피처-이웃 구조를 추출하고, 선택적 마르코프 랜덤 필드(Selective Markov Random Field, S-MRF)를 통해 뷰 간의 의존성을 동적으로 관리합니다. 이를 통해 다양한 뷰 간의 정보를 효과적으로 결합할 수 있습니다.

- **Performance Highlights**: 실험 결과, TUNED 모델은 기존의 방법들에 비해 정확성과 강인성을 개선하며, 특히 높은 수준의 불확실성과 갈등이 있는 상황에서 우수한 성능을 보입니다.



### Self-Supervised Vision Transformers for Writer Retrieva (https://arxiv.org/abs/2409.00751)
- **What's New**: 이 연구는 작가 검색(writer retrieval) 분야에 Vision Transformer (ViT)를 처음으로 적용한 방법을 제안합니다. 자가 감독(self-supervised) 학습 방식을 통해 레이블 없이도 모델을 훈련할 수 있으며, 제품의 성능 추천을 위해 VLAD 인코딩을 사용합니다.

- **Technical Details**: 제안된 방법은 ViT에서 추출한 지역 특징을 바탕으로 작가 검색 및 식별을 위한 전적으로 자가 감독 방식입니다. 이 연구에서는 Masked Image Modeling과 self-distillation 기법을 결합한 방식으로 ViT를 훈련시키며, 클래스 토큰 대신 전면 패치 토큰을 추출하여 VLAD로 인코딩합니다. 이후 코사인 거리(cosine distance)를 사용하여 페이지 설명자 간의 비교를 수행합니다.

- **Performance Highlights**: 이 연구에서 제안된 방식은 Historical-WI 데이터셋에서 83.1% mAP, HisIR19 데이터셋에서 95.0% mAP라는 새로운 최첨단 성능을 기록하였으며, CVL 데이터베이스에서는 98.6% mAP로 직접 적용될 수 있음을 보여줍니다.



### Assessing UHD Image Quality from Aesthetics, Distortions, and Saliency (https://arxiv.org/abs/2409.00749)
Comments:
          The proposed model won first prize in ECCV AIM 2024 Pushing the Boundaries of Blind Photo Quality Assessment Challenge

- **What's New**: 이 논문은 UHD 이미지 품질 평가(IQA)를 위한 새로운 다중 분기 심층 신경망(DNN) 모델을 제안합니다. 이 모델은 글로벌 미적 특성, 지역 기술 왜곡, 그리고 두드러진 콘텐츠 인식을 통해 UHD 이미지를 평가합니다. 또한, 이 모델은 ECCV AIM 2024 UHD-IQA Challenge에서 1등상을 획득했습니다.

- **Technical Details**: 제안된 모델은 Swin Transformer Tiny를 백본 네트워크로 사용하여 세 가지 관점에서 기능을 추출합니다: 1) 미적 특성 분석을 위해 UHD 이미지를 낮은 해상도로 변환합니다. 2) 기술 왜곡 측정은 미니 패치 샘플링 전략을 사용하여 이루어집니다. 3) 두드러진 콘텐츠는 중심 패치를 크롭하여 평가합니다. 최종적으로, 두 개의 다층 퍼셉트론(MLP) 네트워크를 통해 품질 점수로 회귀합니다.

- **Performance Highlights**: 실험 결과, 제안한 모델은 UHD-IQA 데이터셋에서 가장 우수한 성능을 보여주었으며, 비교적 낮은 계산 복잡도를 유지하면서 실제 시나리오에서도 널리 적용 가능합니다.



### DSLO: Deep Sequence LiDAR Odometry Based on Inconsistent Spatio-temporal Propagation (https://arxiv.org/abs/2409.00744)
Comments:
          6 pages, 5 figures, accepted by IROS 2024

- **What's New**: 이 논문은 LiDAR 오도메트리에 대한 일관되지 않은 시공간 전파(spatio-temporal propagation)를 기반으로 한 3D 포인트 클라우드 시퀀스 학습 모델인 DSLO를 소개합니다. 이 모델은 공간 정보 재사용 전략(spatial information reuse), 순차적 자세 초기화 모듈(sequential pose initialization module), 게이트 계층 자세 정제 모듈(gated hierarchical pose refinement module), 및 시간적 특징 전파 모듈(temporal feature propagation module)로 구성되어 있습니다.

- **Technical Details**: DSLO는 포인트 피처 피라미드(point feature pyramid)를 사용하여 공간 특징을 인코딩하고, 후속 자세 추정을 위해 특징을 재사용하여 계산 오버헤드를 줄입니다. 또한 LiDAR의 고주파 샘플링 특성을 활용하여 LiDAR 자세를 초기화하는 순차적 자세 초기화 방법을 도입합니다. 게이트 계층 자세 정제 메커니즘은 다양한 층의 운동 정보를 게이트 추정에 따라 선택적으로 유지하거나 폐기하여 정제합니다. 마지막으로, 시간적 특징 전파는 포인트 클라우드 시퀀스에서의 역사적 운동 정보를 통합하여 프레임 간의 운동 정보 전달 시 공간 불일치 문제를 해결합니다.

- **Performance Highlights**: KITTI 오도메트리 데이터셋 및 Argoverse 데이터셋에서의 실험 결과, DSLO는 최신 방법들보다 더 우수한 성능을 발휘하며, RTE에서 최소 15.67%, RRE에서 12.64%의 향상을 달성하였습니다. 또한, 기준 방법들과 비교하여 런타임을 34.69% 줄이는 성과를 보였습니다.



### Trust And Balance: Few Trusted Samples Pseudo-Labeling and Temperature Scaled Loss for Effective Source-Free Unsupervised Domain Adaptation (https://arxiv.org/abs/2409.00741)
- **What's New**: 본 논문에서는 Source-Free Unsupervised Domain Adaptation (SF-UDA) 문제를 다루며, Few Trusted Samples Pseudo-labeling (FTSP)와 Temperature Scaled Adaptive Loss (TSAL)라는 두 가지 새로운 방법론을 제안합니다. 이러한 기법들은 라벨이 없는 타겟 도메인에서의 효과적인 분류와 학습을 가능하게 합니다.

- **Technical Details**: FTSP는 타겟 데이터에서 신뢰할 수 있는 소수의 샘플을 활용하여 전체 도메인에 대한 pseudo-label을 생성하는 간단하고 효과적인 방법입니다. TSAL은 쌍온도 스케줄링을 통해 다양성, 분별력 및 pseudo-label의 중요성을 균형있게 조정하는 고급 목표 함수입니다. 이 두 가지 기법은 Trust And Balance (TAB) 적응 방법론으로 통합되었습니다.

- **Performance Highlights**: 제안된 방법은 Office31, Office-Home, ImageCLEF-DA, Adaptiope와 같은 다양한 데이터셋에서 ResNet50 및 ViT-Large 아키텍처를 사용하여 rigorously 평가되었습니다. 결과적으로, 우리의 방법론은 최신 기술들과 비교하여 우수한 성능을 보여 주며, SF-UDA 분야에서의 효율성을 강조합니다.



### MoManifold: Learning to Measure 3D Human Motion via Decoupled Joint Acceleration Manifolds (https://arxiv.org/abs/2409.00736)
Comments:
          Accepted by BMVC 2024. Supplementary material is included at the end of the main paper (12 pages, 11 figures, 5 tables)

- **What's New**: MoManifold는 지속적인 고차원 운동 공간에서 가능성 있는 인간 운동을 모델링하는 새로운 인간 운동 프라이어(Human Motion Prior)를 제안합니다. 이는 기존의 수학적 또는 VAE 기반 방법론과 다르게 신경 거리 필드를 사용하여 인간 역학을 명확하게 정량화합니다.

- **Technical Details**: MoManifold는 연속적인 고차원 운동 공간에서의 인간 운동을 모델링하며, 데코플링된 관절 가속도(manifolds)와 인체 스켈레톤 기하학 기반의 가중치 설계를 통해 인간 역학을 효과적으로 모델링합니다. 이는 운동 관련 작업을 지원하기 위한 고유한 최적화 방법을 도입합니다.

- **Performance Highlights**: MoManifold는 여러 다운스트림 작업에서 기존 SOTA(State Of The Art) 방법론을 능가하는 성능을 보였습니다. 특히, 실제 인간 모션 캡처 데이터의 노이즈 제거, 부분 3D 관찰에서 인간 운동 회복, SMPL 기반 포즈 추정기의 진동 완화 및 운동 사이의 보간 결과 개선에 기여합니다.



### A Critical Analysis on Machine Learning Techniques for Video-based Human Activity Recognition of Surveillance Systems: A Review (https://arxiv.org/abs/2409.00731)
- **What's New**: 본 논문은 인공지능 기반의 비디오 감시 시스템을 통해 공항, 기차역, 쇼핑몰 등 혼잡한 장소에서 비정상적인 활동을 감지하고 대응하는 기술에 대해 다룹니다. 특히, Human Activity Recognition (HAR) 기술의 발전과 이의 효과적인 적용 방법들을 심층적으로 분석합니다.

- **Technical Details**: 본 논문에서는 HAR의 접근 방식을 두 그룹으로 구분합니다: marker-based와 vision-based 시스템. 또한, Convolutional Neural Network (CNN), Recurrent Neural Network (RNN), Hidden Markov Model (HMM), K-means Clustering 등 다양한 machine learning과 deep learning 기술을 비교합니다. 비디오 처리 기술과 관련하여 데이터 전처리, 특징 추출, 객체 분류 및 활동 인식을 포함한 여러 단계가 다룹니다.

- **Performance Highlights**: HAR 시스템은 비정상적인 행동을 정확하게 인식하는 데 필수적입니다. 연구에서는 다양한 머신러닝 기법의 성능 비교뿐만 아니라 도전 과제와 미래 방향에 대해서도 논의하며, HAR 기술 발전을 위한 체계적인 접근 방식을 제공합니다. 실제 사례로는 은행 강도, 심장 마비 및 ATM 강도와 같은 사건을 방지할 수 있는 잠재력을 강조합니다.



### LPUWF-LDM: Enhanced Latent Diffusion Model for Precise Late-phase UWF-FA Generation on Limited Datas (https://arxiv.org/abs/2409.00726)
Comments:
          13 pages, 7 figures

- **What's New**: 본 연구는 Ultra-Wide-Field Scanning Laser Ophthalmoscopy (UWF-SLO)에서 제한된 데이터로부터 고품질 Late-Phase UWF-FA 이미지를 생성할 수 있는 개선된 Latent Diffusion Model을 소개합니다. 이를 통해 기존의 유해한 염료 주입 없이 정확한 안과 질병 진단이 가능해집니다.

- **Technical Details**: 제안된 모델은 Cross-temporal Regional Difference Loss (CTRD Loss)를 사용하여 Early-phase와 Late-phase UWF-FA 간의 차이를 강조합니다. 저주파 정보를 향상시키기 위한 저주파 강화 노이즈 기법을 diffusion 과정에 적용함으로써 의료 이미지의 현실감을 개선하고, Gated Convolutional Encoder를 통해 추가 정보를 추출하여 변동성이 있는 데이터셋에서도 우수한 성능을 발휘합니다.

- **Performance Highlights**: 제안된 Latent Diffusion Model for Ultra-Wide-Field Late-Phase Fluorescein Angiography (LPUWF-LDM)는 기존의 방법들보다 세부정보를 효과적으로 재구성하며, 한정된 데이터셋으로 작업할 시 최첨단 성능을 달성했습니다. 다양한 정량적 및 정성적 지표에서 우수한 결과를 보여주었습니다.



### ReMOVE: A Reference-free Metric for Object Erasur (https://arxiv.org/abs/2409.00707)
Comments:
          Accepted at The First Workshop on the Evaluation of Generative Foundation Models (EvGENFM) at CVPR 2024

- **What's New**: 이번 논문에서는 	exttt{ReMOVE}라는 참조 이미지 없이 객체 제거 효과성을 평가하는 새로운 메트릭을 도입합니다. 기존의 메트릭인 LPIPS와 CLIPScore와 달리, 	exttt{ReMOVE}는 실제 시나리오에서 공통적으로 발생하는 참고 이미지가 없을 때의 평가 문제를 해결합니다.

- **Technical Details**: 	exttt{ReMOVE}는 기존의 메트릭이 필요로 하는 Ground Truth의 의존성을 줄이기 위해, ViT(Visual Transformer)를 활용하여 마스킹된 영역과 마스킹되지 않은 영역의 평균 패치 특성 간의 차이를 측정하여 품질을 평가합니다. 이 메트릭은 객체 제거와 교체를 효과적으로 구분할 수 있게 설계되었습니다.

- **Performance Highlights**: 실험 결과 	exttt{ReMOVE}는 최신의 참조 기반 메트릭들과 높은 상관관계를 보였으며, 인간의 인식과 잘 일치한다는 것을 보여주었습니다. 또한 실제 데이터 세트를 통해 테스트한 결과, 메트릭이 일관된 결과를 나타냈습니다. 이는 	exttt{ReMOVE}가 실제 이미지에 대한 깊이 있는 품질 평가를 가능하게 함을 의미합니다.



### Enhancing Remote Sensing Vision-Language Models for Zero-Shot Scene Classification (https://arxiv.org/abs/2409.00698)
- **What's New**: 본 연구는 RS-TransCLIP이라는 새로운 전이적 추론(transductive inference) 방법을 제안합니다. 이 방법은 기존의 각 패치에 대한 독립적인 예측을 기반으로 한 유도 추론(inductive inference)에서 벗어나, 이미지 인코더의 패치 친화성(patch affinity relationship)을 활용하여 텍스트 프롬프트(text prompting)를 통해 제로샷(zero-shot) 장면 분류를 개선합니다.

- **Technical Details**: RS-TransCLIP은 특징 공간 내 데이터 구조를 가우시안 혼합 모델(Gaussian Mixture Model, GMM)로 모델링합니다. 이 목표 함수는 비지도 GMM 클러스터링, 친화성 기반 라플라시안 정규화(Laplacian regularization), 그리고 Kullback-Leibler 발산(KL divergence) 정규화 항으로 구성됩니다. 각 패치에 대해 개별적인 예측을 만드는 대신, 모든 포인트를 동시에 집합적으로 예측합니다.

- **Performance Highlights**: 실험 결과, RS-TransCLIP은 10개의 원격 탐지 데이터셋에서 기존의 유도 제로샷 분류(inductive zero-shot classification) 방법에 비해 상당한 정확도 향상을 보여주었습니다. 본 알고리즘은 라벨 없이 발생하며, 전체 추론 시간에 미미한 컴퓨팅 비용만 추가됩니다.



### Curriculum Prompting Foundation Models for Medical Image Segmentation (https://arxiv.org/abs/2409.00695)
Comments:
          Accepted by MICCAI 2024

- **What's New**: 본 연구에서는 의료 이미지 분할을 위한 SAM(Segment Anything Model)의 성능을 향상시키기 위해 자동화된 프롬프트 생성 기법을 제안합니다. 기존에 필요한 수작업 프롬프트 생성 과정을 줄이고, 다양한 타입의 프롬프트를 효율적으로 결합하는 방법을 소개합니다.

- **Technical Details**: 연구에서 제안하는 크리큘럼 프롬프팅(Curriculum Prompting) 기법은 코스(Coarse)에서 파인(Fine)으로 진행되는 다양한 유형의 프롬프트를 시스템적으로 통합하여 세분화 문제를 해결합니다. SAM의 기존 구조를 활용하여, 이미지 인코더와 프롬프트 인코더를 통해 세분화 마스크를 생성하는 과정에서, 박스 프롬프트와 포인트 프롬프트를 결합하여 프롬프트의 종류를 다각화합니다.

- **Performance Highlights**: 세 가지 공개 의료 데이터세트에서 실험한 결과, 제안한 기법은 기존 SAM 기반의 의료 이미지 분할 방법보다 정량적, 정성적으로 우수한 성능을 보였습니다. 자동 프롬프트 생성과 크리큘럼 프롬프팅의 결합은 세분화 결과를 크게 향상시켰습니다.



### IAFI-FCOS: Intra- and across-layer feature interaction FCOS model for lesion detection of CT images (https://arxiv.org/abs/2409.00694)
Comments:
          2024 IJCNN

- **What's New**: 본 논문에서는 의료 이미지 내 병변 탐지의 효과성을 높이기 위해, 병변 지역의 특징뿐만 아니라 주변 정보를 충분히 활용할 수 있는 새로운 방법을 제안합니다. 이 방식은 Multi-Scale Feature Fusion Mechanism인 ICAF-FPN과 이를 통해 다양한 레이어의 정보를 효과적으로 이용하는 IAFI-FCOS 모델을 기반으로 하고 있습니다. 이 모델은 intra-layer context augmentation (ICA) block과 across-layer feature weighting (AFW) block을 통해 특징을 풍부하게 개선합니다.

- **Technical Details**: 제안된 IAFI-FCOS 모델은 dilated attention을 활용한 ICA 블록을 통해 병변 지역과 주변 정보 간의 장거리 의존성을 캡처하고, dual-axis attention을 활용한 AFW 블록을 통해 다양한 레이어의 특징을 효율적으로 결합합니다. 이러한 구성은 디테일한 특징의 표현을 강화하고 정보 손실 문제를 완화하는 데 기여합니다.

- **Performance Highlights**: 제안된 모델은 개인 파열 병변 데이터셋과 공개된 DeepLesion 데이터셋에서 광범위한 실험을 진행하였으며, 특히 파열 병변 데이터셋에서 SOTA(최첨단)의 성능을 달성하면서 다양한 데이터셋에 대해 성능과 강인성을 입증했습니다.



### Decoupled and Interactive Regression Modeling for High-performance One-stage 3D Object Detection (https://arxiv.org/abs/2409.00690)
- **What's New**: 본 연구는 3D 물체 탐지의 1단계 모델링에서 경계 상자(단, bounding box) 회귀 작업의 한계를 해결하기 위해 Decoupled and Interactive Regression Modeling (DIRM)을 제안합니다. 이는 두 가지 주요 문제, 즉 제한된 center-offset 예측으로 인한 정확도 저하와 회귀 작업에서 무시된 저품질 샘플의 영향을 다룹니다.

- **Technical Details**: DIRM에는 Decoupled Attribute Regression (DAR)와 Interactive Quality Prediction (IQP)의 두 가지 주요 전략이 포함됩니다. DAR은 중심 속성(단, attribute) 회귀 작업에서 장거리 회귀 모델링을 수행하며, IQP는 클래스에 관계없이 물체 분류 작업을 IoU 예측 작업에 통합하여 저품질 결과의 신뢰성을 향상시킵니다. 이로 인해 서로 다른 품질의 예측 결과를 효과적으로 처리할 수 있습니다.

- **Performance Highlights**: Waymo 및 ONCE 데이터셋에서의 광범위한 실험 결과, DIRM은 여러 최신 기술 대비 2.0~5.0 mAPH 성과 향상을 달성하였으며, 두 데이터셋 모두에서 최신 성능을 이루었습니다.



### Accurate Forgetting for All-in-One Image Restoration Mod (https://arxiv.org/abs/2409.00685)
- **What's New**: 이 연구는 이미지 복원(image restoration) 분야에서 개인 정보 보호 문제를 처음으로 제기하며, AI 모델에서 개인 데이터의 영향을 제거하기 위한 새로운 접근 방식을 제안합니다.

- **Technical Details**: 제안된 방법은 Instance-wise Unlearning 기법과 적대적 예제(adversarial examples)를 활용하여 특정 데이터에서의 영향을 완전히 지우는 것으로, 이는 All-In-One 모델을 통해 다양한 손상 시나리오를 복원할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, 저비용의 GPU를 이용해 다수의 데이터셋에서 지우기를 효과적으로 수행하면서도 남은 데이터에 대한 지식을 보존할 수 있다는 것이 증명되었습니다.



### MERLiN: Single-Shot Material Estimation and Relighting for Photometric Stereo (https://arxiv.org/abs/2409.00674)
Comments:
          Accepted in ECCV 2024

- **What's New**: 본 논문에서는 MERLiN(Material Estimation and ReLighting Network)이라는 주목 기반의 네트워크를 제안하여 단일 이미지에서 역 렌더링(inverse rendering)과 리라이트(relighting)를 통합한 프레임워크를 제공합니다. 이를 통해 복잡한 데이터 수집 작업을 피하면서도 높은 품질의 표면 노멀(surface normals) 추정이 가능함을 보여줍니다.

- **Technical Details**: MERLiN은 물리 기반의 글로벌 조명(global illumination) 인식을 고려한 네트워크로, 공간적으로 변화하는 BRDF(bidirectional reflectance distribution function) 파라미터를 추정하고 리라이트를 수행합니다. 이 과정에서 이미지의 내적 파라미터(intrinsics)를 추정하고, 외관과 조명 간의 복잡한 관계를 학습합니다. 또한, 제안된 모델은 대규모 합성 데이터셋을 기반으로 훈련됩니다.

- **Performance Highlights**: 리라이트된 이미지에 대한 광학 스테레오(photo metric stereo) 벤치마크를 통해 리라이트된 이미지의 물리적 정확성과 노멀 추정 정확도를 검증하여, 실제 이미지에 대한 일반화 성능이 우수함을 입증했습니다. 이러한 접근법은 단일 이미지 기반의 광학 스테레오 문제를 해결하는 데 중요한 이정표가 됩니다.



### Study of Dropout in PointPillars with 3D Object Detection (https://arxiv.org/abs/2409.00673)
- **What's New**: 이번 연구는 PointPillars 모델의 3D 객체 탐지 성능을 향상시키기 위해 다양한 dropout (드롭아웃) 비율을 적용한 분석을 제공합니다.

- **Technical Details**: 드롭아웃은 훈련 중에 뉴런을 무작위로 생략하여 네트워크가 강력하고 다양한 기능(feature)을 학습하도록 하는 정규화(regularization) 기법입니다. 본 연구에서는 모델의 회귀 성능(regression performance)과 정확도(accuracy)를 Average Precision (AP) 및 Average Orientation Similarity (AOS)로 측정하여 다양한 향상 기법의 효과를 체계적으로 비교합니다.

- **Performance Highlights**: 연구 결과는 자율 주행(application)에서 3D 객체 탐지 성능을 향상시키기 위한 최적의 향상 기법에 대한 통찰(insight)을 제공합니다.



### Disparity Estimation Using a Quad-Pixel Sensor (https://arxiv.org/abs/2409.00665)
- **What's New**: 본 논문에서는 quad-pixel (QP) 센서를 활용한 차별화된 disparity 추정 네트워크 QPDNet을 제안합니다. QP 센서는 단일 마이크로 렌즈 아래에 2×2 배치된 포토다이오드를 사용하여 다방향 phase shifting을 생성합니다. 이 정보를 효과적으로 활용하여 깊이 추정을 수행하는 방법을 다루고 있습니다.

- **Technical Details**: QPDNet은 수직 및 수평 stereo-matching 상관관계를 융합하여 QP 센서에서 관찰되는 다방향 disparity를 효과적으로 활용하는 네트워크입니다. 새로운 multi-direction lookup fusion (MDLF) 모듈을 도입하여 다방향 상관성을 추출하고 융합하여 최종 disparity 맵을 예측합니다.

- **Performance Highlights**: 실험 결과, QPDNet은 기존의 stereo 및 DP 방법들에 비해 우수한 성능을 보이며, 동작 검증을 통해 더 효과적인 깊이 추정이 가능함을 입증하였습니다.



### Seed-to-Seed: Image Translation in Diffusion Seed Spac (https://arxiv.org/abs/2409.00654)
- **What's New**: 본 논문에서는 Seed-to-Seed Translation (StS)이라는 새로운 이미지를 변환하는 방식을 도입하였습니다. 이 방법은 diffusion models (DMs)를 사용하여 소스 이미지의 구조를 밀접하게 유지하는 번역을 목표로 하고 있습니다. 기존의 방법과는 달리, 우리는 pretrained DM의 inverted seeds 공간에서 인코딩된 의미론적 정보를 활용하여 구조를 변경하지 않고 다양한 변형을 만들 수 있음을 보여줍니다.

- **Technical Details**: 우리는 sts-GAN이라는 비 독립적 변환 모델을 CycleGAN을 기반으로 훈련하여 사용합니다. 이 과정에서 두 가지 세트의 이미지를 invert하여 얻은 seed 공간을 통해 소스 시드를 목표 시드로 변환하고, 이를 통해 최종 이미지를 생성합니다. 추가적으로 ControlNet을 사용하여 입력 이미지의 구조가 보존되도록 합니다.

- **Performance Highlights**: 자동차 장면의 변환 작업에서 기존 GAN 기반 및 diffusion 기반 방법들에 비해 뛰어난 성능을 보여주었으며, 날씨 변환과 같은 다양한 이미지 변환 작업에서도 효과적임을 입증했습니다. 이 접근 방식은 비매칭 이미지 간 변환 작업에서 소스 의미의 긴밀한 준수를 요구하는 작업에서 그 가능성을 보여줍니다.



### Artificial Intelligence in Gastrointestinal Bleeding Analysis for Video Capsule Endoscopy: Insights, Innovations, and Prospects (2008-2023) (https://arxiv.org/abs/2409.00639)
- **What's New**: 이 논문은 Video Capsule Endoscopy (VCE)에서 위장관(GI) 출혈 탐지를 위한 머신 러닝(ML) 응용의 현재 상태를 종합적으로 검토하고 있습니다.

- **Technical Details**: 총 113개의 논문을 분석하여 ML 방법론, 평가 지표, 오픈 소스 데이터셋 및 기술 분류의 효과성 및 한계를 강조합니다. VCE는 카메라가 내장된 작은 캡슐을 삼켜 소화관을 통해 여행하면서 이미지를 캡처합니다. 이 방법은 진단 정확도를 높이기 위해 ML을 자동화하여 분석하는 것을 목표로 합니다.

- **Performance Highlights**: 기존의 진단 방법에 비해 VCE는 GI 출혈의 빠른 정확한 진단을 가능하게 하고, 인력에 대한 의존도를 줄이며, 진단 및 치료 시간을 단축합니다. ML의 적용은 진단 오류를 최소화하고, GI 출혈 탐지의 미래 연구 방향을 제시합니다.



### IGEV++: Iterative Multi-range Geometry Encoding Volumes for Stereo Matching (https://arxiv.org/abs/2409.00638)
Comments:
          12 pages, 10 figures

- **What's New**: 이 논문에서는 스테레오 매칭을 위한 새로운 심층 네트워크 아키텍처인 IGEV++를 제안합니다. IGEV++는 Multi-range Geometry Encoding Volumes (MGEV)를 구축하여 열악한 조건에서의 매칭 모호성과 큰 차이를 효율적으로 처리하는 방법을 제시합니다.

- **Technical Details**: MGEV는 열악한 지역과 큰 차이 및 디테일과 작은 차이에 대한 세밀한 기하 정보를 인코딩합니다. 이를 위해 적응형 패치 매칭 모듈을 도입하여 뛰어난 성능을 발휘하며, 선택적 기하 특징 융합 모듈을 통해 여러 범위와 세밀도의 기하학적 특징을 통합합니다.

- **Performance Highlights**: IGEV++는 Middlebury, ETH3D, KITTI 2012 및 2015 벤치마크에서 최첨단 정확도를 달성했으며, 768px까지의 모든 차이 범위에서 Scene Flow 테스트 세트에서 최고의 성능을 기록합니다. 또한 실시간 버전인 RT-IGEV++는 KITTI 벤치마크에서 모든 발표된 실시간 방법 중에서 최고의 성과를 나타냅니다.



### Make Your ViT-based Multi-view 3D Detectors Faster via Token Compression (https://arxiv.org/abs/2409.00633)
Comments:
          Accepted by ECCV 2024

- **What's New**: 본 논문에서는 TokenCompression3D(ToC3D)라는 간단하면서도 효과적인 방법을 제안하여, 다중 뷰 3D 감지에서 비전 트랜스포머(ViT)의 백본을 효율적으로 조정함으로써 동적 라우터(.Dynamic Router)를 활용하고, 역사적 객체 쿼리를 foreground priors로 사용하여 정보 밀도를 효과적으로 결정하고, 중요한 foreground 토큰에 더 많은 계산 자원을 할당하는 방법을 모색합니다.

- **Technical Details**: ToC3D는 두 가지 주요 디자인으로 구성되며, 첫째는 모션 쿼리 기반 비트 선택 전략(MQTS)으로, 이는 이미지 토큰과 역사적 객체 쿼리를 입력으로 받아 드는 3D 모션 정보를 모델링하며 이미지 토큰의 중요도 점수를 계산합니다. 둘째, 동적 라우터는 중요한 foreground 프로포절에 더 많은 계산 자원을 할당하여 효율적인 기능 추출을 지원합니다.

- **Performance Highlights**: ToC3D는 nuScenes 데이터셋에서 평가되었으며, StreamPETR 기준선 대비 최대 30%의 추론 속도 향상과 함께 최신 성능을 거의 유지할 수 있음을 입증하였습니다. 이 방법은 ViT 및 입력 해상도를 늘리더라도 일관된 성능 향상을 보여줍니다.



### Roundabout Dilemma Zone Data Mining and Forecasting with Trajectory Prediction and Graph Neural Networks (https://arxiv.org/abs/2409.00622)
- **What's New**: 본 논문은 자율 차량의 안전성을 높이기 위해 교통 원형 교차로에서의 딜레마 존(Dilemma Zone, DZ) 예측을 위한 자동화 시스템을 제시합니다. 이는 측면 데이터를 통합하여 차량과 동적 요소들을 고려한 예측 모델을 기반으로 합니다.

- **Technical Details**: 모듈화된 그래프 구조의 순환 모델을 활용하여 다양한 운전자의 주행 궤적을 예측합니다. 이 모델은 그래프 신경망(Graph Neural Networks, GNN)을 기반으로 하여 DZ 이벤트 예측 및 교통 관리 의사결정을 개선합니다.

- **Performance Highlights**: 현실 세계의 교통 원형 교차로 데이터셋을 사용하여 평가한 결과, 본 시스템은 높은 정확도와 낮은 오탐률(0.1%)을 달성했습니다. 이는 자율 차량 시대의 교차로 안전성을 보장하는데 기여할 것으로 보입니다.



### Enhancing Vectorized Map Perception with Historical Rasterized Maps (https://arxiv.org/abs/2409.00620)
Comments:
          Accepted by ECCV 2024

- **What's New**: 이번 논문에서는 HRMapNet이라는 새로운 프레임워크를 제안하며, 이는 저비용의 역사적 래스터 맵(Historical Rasterized Map)을 활용하여 온라인 벡터화된 맵 인식을 개선하는 방법을 소개합니다.

- **Technical Details**: HRMapNet은 과거 예측된 벡터화된 맵에서 쉽게 생성된 역사적 래스터 맵을 사용하여 온라인 벡터화된 맵 인식을 보강합니다. 두 가지 모듈(특징 집계 모듈 및 쿼리 초기화 모듈)로 구성되어 있어, BEV(Top-Down Bird’s-Eye View) 특징과 맵 요소 쿼리를 향상시킵니다.

- **Performance Highlights**: HRMapNet은 두 개의 최첨단 방법(MapTRv2, StreamMapNet)과 통합되어 nuScenes 및 Argoverse 2 데이터셋에서 성능을 크게 향상시킵니다. 결과적으로 온라인 인식 성능이 개선되었으며, 자율주행 애플리케이션의 실용성을 위한 내구성과 잠재적인 응용을 보여주고 있습니다.



### YOLOO: You Only Learn from Others Onc (https://arxiv.org/abs/2409.00618)
- **What's New**: 본 논문은 YOLOO라는 새로운 다중 모달 3D 객체 추적(MOT) 패러다임을 제시하며, 이 시스템은 훈련 시 다중 모달을 활용하지만 추론 단계에서는 포인트 클라우드 데이터만을 사용하여 효율성 문제를 해결합니다.

- **Technical Details**: YOLOO는 통합 삼중 모달 인코더(UTEnc)와 유연한 기하학적 제약(F-GC) 모듈을 포함하고 있습니다. UTEnc는 포인트 클라우드, 이미지 및 텍스트 인코더를 통합하여 학습하며, CLIP 모델을 기반으로 하여 시각적 및 텍스트 정보를 효과적으로 통합합니다. F-GC는 세밀한 기하학적 정렬 메트릭(F-GAM)을 사용하여 불일치하는 연관성을 걸러내며, 트랙 추적의 정확도를 높입니다.

- **Performance Highlights**: KITTI 및 Waymo 추적 데이터셋에서 20개의 경쟁 모델과 비교한 결과, YOLOO는 모든 경쟁 모델을 초월하는 우수한 성능을 보여주며, 효율성과 견고함을 모두 향상시킵니다.



### Style Transfer: From Stitching to Neural Networks (https://arxiv.org/abs/2409.00606)
- **What's New**: 이번 연구에서는 기존의 전통적인 스타일 전송 방법과 딥러닝 기반의 최신 스타일 전송 방법을 비교하여 각 방법의 장단점을 분석하였습니다. 전통적인 방식은 아트워크의 스타일을 이미지에 더할 수 있지만, 매끄럽게 연결되기 어려운 경우가 많습니다. 반면, 딥러닝 기반 방법은 포그라운드(foreground) 요소의 무결성을 유지하면서 배경에 스타일 전송을 적용함으로써 미적인 품질과 계산 효율성을 향상시킵니다.

- **Technical Details**: 전통적인 스타일 전송 방법들은 주로 패치 기반(patch-based) 방법과 최적화 최적화(optimization) 기법을 사용하며, 통계적 분석을 통해 소스 이미지와 목표 스타일 이미지의 텍스쳐를 일치시킵니다. 반면, 딥러닝 기반 방법은 컨볼루션 신경망(CNNs)을 활용하여 복잡한 패턴과 텍스쳐를 포착하며, 이미지를 스타일과 내용 측면에서 더 유연하게 조합합니다. 이 연구는 Efros의 전통적인 방법과 Ding의 딥러닝 방법을 비교하였습니다.

- **Performance Highlights**: 딥러닝 기반의 방법은 더 매끄럽고 다양한 색상을 전송하는 데 성공적이었고, 전통적인 방법은 주요 구조적 정보를 보존하지만 비자연스러운 선의 문제를 초래했습니다. 이 연구에서는 PASCAL VOC 2012 데이터셋을 통해 두 방법의 성능을 비교하였으며, 딥러닝 방법이 전반적으로 더 나은 결과를 나타냈습니다.



### Uncertainty-oriented Order Learning for Facial Beauty Prediction (https://arxiv.org/abs/2409.00603)
- **What's New**: 이 논문은 얼굴 미모 예측(Facial Beauty Prediction, FBP) 문제에 있어 두 가지 불일치를 고려하여 새로운 불확실성 지향 순서 학습(Uncertainty-oriented Order Learning, UOL) 방법을 제안합니다. 이는 얼굴 이미지 간의 미모 순서 관계를 학습하고, 인간 인지의 불확실성을 모델링하여 FBP 성능을 향상시킵니다.

- **Technical Details**: UOL은 분포 비교 모듈을 통해 불확실한 데이터의 순서 관계를 학습하는 방식입니다. 이 방법은 얼굴 미모 기준의 불일치를 해결하기 위해 얼굴 이미지 간의 상대 관계를 학습하고, 인간 인지의 불일치를 다루기 위해 고차원 심리적 스케일에서의 다차원 가우시안 분포 모델링을 적용합니다. 전통적인 순서 학습 방법은 정확한 레이블을 가진 데이터만 비교할 수 있지만, UOL은 Monte Carlo 샘플링 기반의 분포 비교를 통해 불확실한 데이터의 순서를 처리할 수 있습니다.

- **Performance Highlights**: UOL은 SCUT-FBP5500 및 그 외의 다양한 FBP 데이터셋에 대한 광범위한 실험을 통해 기존의 여섯 가지 경쟁 방법보다 우수한 정확도와 일반화 능력을 보여주었습니다. 이로 인해 UOL은 FBP 분야에서 뛰어난 성능과 더불어 방법론적 혁신을 이끌어내고 있습니다.



### Attention-Guided Multi-scale Interaction Network for Face Super-Resolution (https://arxiv.org/abs/2409.00591)
Comments:
          12 pages, 8 figures, 8 tables

- **What's New**: 본 논문에서는 CNN과 Transformer의 장점을 활용한 Attention-Guided Multi-scale Interaction Network (AMINet)를 제안하여, 다중 스케일 기능의 융합 및 상호작용을 개선하는 방법을 모색합니다.

- **Technical Details**: AMINet은 Local and Global Feature Interaction Module (LGFI)와 Selective Kernel Attention Fusion Module (SKAF)을 포함하여, 인코더-디코더 단계에서의 다중 스케일 기능의 자유로운 흐름과 이를 통한 상호작용을 촉진합니다. LGFI는 다양한 수용 영역의 로컬 기능과 글로벌 기능을 융합하며, SKAF는 적응적으로 다양한 기능을 선택하여 융합합니다.

- **Performance Highlights**: 실험 결과, AMINet은 기존 얼굴 슈퍼 해상도(FSR) 방법보다 더 우수한 성능을 보이며, 적은 계산 비용과 빠른 추론 속도로도 더 나은 결과를 도출합니다.



### COMOGen: A Controllable Text-to-3D Multi-object Generation Framework (https://arxiv.org/abs/2409.00590)
- **What's New**: COMOGen이라는 프레임워크는 텍스트 기반 입력을 통해 여러 개의 3D 개체를 동시에 생성할 수 있는 새로운 방식을 제안한다.

- **Technical Details**: COMOGen은 레이아웃 제어 모듈, 다중 뷰 일관성 제어 모듈, 3D 콘텐츠 향상 모듈의 세 가지 모듈로 구성된다. 레이아웃 다중뷰 스코어 증류(Layout Multi-view Score Distillation, LMSD) 기법을 통해 두 가지 선행 지식을 통합하고 생성된 3D 콘텐츠의 다양성과 품질을 더욱 향상시킨다.

- **Performance Highlights**: COMOGen은 기존의 최첨단 방법들과 비교할 때 효과적인 결과를 보여주며, 텍스트와 경계 상자만으로도 합리적인 3D 콘텐츠를 생성할 수 있게 된다.



### Change-Aware Siamese Network for Surface Defects Segmentation under Complex Background (https://arxiv.org/abs/2409.00589)
- **What's New**: 본 논문에서는 defect appearance에 대한 과도한 의존을 피하고, 정확한 defect segmentation을 달성하기 위한 'change-aware Siamese network'를 제안합니다. 이 모델은 change detection framework 내에서 defect segmentation 문제를 해결하며, 다양한 defect 카테고리를 결합하기 위한 novel multi-class balanced contrastive loss를 도입하여 Transformer 기반 encoder를 지원합니다.

- **Technical Details**: 제안된 change-aware Siamese network에는 change attention mechanism이 포함되어 있으며, 이는 defect-free와 defective 샘플 간의 difference feature를 추출하는 데 사용됩니다. 인코딩 단계에서 multi-class balanced contrastive loss에 의해 제약된 Transformer 기반 Siamese network가 사용되며, 디코딩 단계에서는 feature distance map이 decoder와 skip-connected 되어 pixel-wise defect localization을 도와줍니다.

- **Performance Highlights**: 제안된 모델은 SynLCD, PKU-Market-PCB 및 MvTec-AD 데이터셋에서 SOTA appearance-based segmentation 방법들보다 우수한 성능을 기록하였으며, 다양한 supervision 설정에서 semi-supervised 접근 방식과 비교할 때 새로운 최첨단 성능을 달성했습니다.



### McCaD: Multi-Contrast MRI Conditioned, Adaptive Adversarial Diffusion Model for High-Fidelity MRI Synthesis (https://arxiv.org/abs/2409.00585)
- **What's New**: 이 연구에서는 McCaD(Multi-Contrast MRI Conditioned Adaptive Adversarial Diffusion)라는 새로운 프레임워크를 소개하여, 다양한 MRI 대비를 바탕으로 고품질 이미지를 합성할 수 있는 방법을 제안합니다. 기존 방법들은 단일 이미지 대비에 집중하여 여러 대비 간의 정보를 충분히 활용하지 못했던 점을 보완하였습니다.

- **Technical Details**: McCaD는 다중 대비 MRI 합성을 위해 설계된 적응형 적대적 확산 모델로, 다중 대비를 조건으로 하는 메커니즘을 사용합니다. 이 모델은 다중 스케일의 피처 가이드 메커니즘, 노이즈 제거(denoising) 및 의미론적 인코더를 포함하여 합성 정밀도를 증가시킵니다. 또한, 내재적 특성을 포착하기 위한 적응형 특징 극대화 전략과 공간적 특징 주의 손실 함수를 도입하여, 진단에 중요한 영역을 정확히 형상화합니다.

- **Performance Highlights**: 종양 및 건강한 다중 대비 MRI 데이터셋에 대한 광범위한 실험 결과, McCaD는 최신 기법들과 비교하여 정량적 및 정성적으로 우수한 성능을 보여주었습니다. 개발된 코드와 보조 자료도 제공됩니다.



### Compositional 3D-aware Video Generation with LLM Director (https://arxiv.org/abs/2409.00558)
- **What's New**: 본 논문은 텍스트 프롬프트를 기반으로 각 개념을 3D 표현으로 생성한 후, 이를 조합하는 새로운 텍스트 안내 구Composition 3D-aware Video Generation(이하 C3V) 방식을 제안합니다. 이 방식은 Large Language Models(LLM)과 2D diffusion 모델을 활용하여 개별 개념에 대한 유연한 제어를 가능하게 합니다.

- **Technical Details**: C3V는 다음 세 가지 단계로 구성됩니다: 1) 텍스트 프롬프트를 sub-prompt로 분해하여 각 개념을 설명하고, 이에 따라 pre-trained expert 모델을 사용하여 3D 표현을 생성합니다. 2) 생성된 3D 표현을 조합하기 위해 multi-modal LLM에 질의하여 객체의 스케일과 경로에 대한 큰 가이드를 생성합니다. 3) 최종적으로 Score Distillation Sampling(SDS)을 활용하여 각 객체의 3D 공간 내 위치와 회전을 정교하게 조정합니다.

- **Performance Highlights**: 이 방법은 다양한 모션을 포함한 고화질 비디오를 생성할 수 있으며, 복잡한 쿼리를 처리할 수 있습니다. 또한, 생성된 비디오의 여러 개념을 편집하는 유연성을 보여주며, LLM을 감독자로 활용한 최초의 텍스트 안내 조합 3D 비디오 생성 시도를 특징으로 합니다.



### FADE: Few-shot/zero-shot Anomaly Detection Engine using Large Vision-Language Mod (https://arxiv.org/abs/2409.00556)
Comments:
          13 pages, 2 figures, Accepted for BMVC 2024

- **What's New**: 본 논문에서는 제조 산업의 자동 이미지 이상 탐지를 위해 Few-shot/zero-shot Anomaly Detection Engine (FADE)를 제안합니다. FADE는 CLIP 모델을 활용하여 언어 기반 이상 세분화 및 객체-특화 모델 훈련에서 발생하는 문제를 해결합니다.

- **Technical Details**: FADE는 언어에 더 잘 정렬된 멀티스케일 이미지 패치 임베딩을 사용하여 라벨링 작업을 개선하고, 산업 관련 텍스트 프롬프트를 자동으로 생성하여 이상 탐지 성능을 향상시킵니다. 추가적으로 패치 임베딩을 사용하여 비전 기반 가이드를 강화합니다.

- **Performance Highlights**: FADE는 MVTec-AD 및 VisA 데이터셋에서 zero-shot 상태에서 89.6% (91.5%)의 픽셀-AUROC 성능을 보였고, 1-normal-shot 상태에서는 95.4% (97.5%)로, 기존의 최고 성능 방법을 초월하는 결과를 보였습니다.



### Data Augmentation for Image Classification using Generative AI (https://arxiv.org/abs/2409.00547)
Comments:
          19 pages, 15 figures, 4 tables

- **What's New**: 이 논문에서는 Automated Generative Data Augmentation(AGA) 프레임워크를 제안하여, 대규모 언어 모델(LLMs), 확산 모델(difusion models), 그리고 분할 모델(segmentation models)을 사용해 데이터 증대(data augmentation)를 자동으로 수행하는 방법을 소개합니다. AGA는 전경(foreground)의 진정성(authenticity)을 유지하면서 배경의 다양성을 보장합니다.

- **Technical Details**: AGA의 주요 요소는 다음과 같습니다: 1) 객체 추출을 위한 분할(segment) 및 슈퍼클래스(superclass) 기반 접근법, 2) 프롬프트 다양성을 위한 조합(combinatorial) 복잡성 및 프롬프트 분해(prompt decomposition) 사용, 3) 아핀 변형을 통한 주제(subject) 조작을 포함합니다. AGA는 이미지 세분화를 통해 주제를 분리하고, 사전 학습된 LLM을 사용해 다양한 배경 캡션을 생성합니다. 그 후 안정적 확산(Stable Diffusion)을 통해 다양한 배경을 생성하며 주제와 배경을 무결하게 통합합니다.

- **Performance Highlights**: AGA는 ImageNet, CUB, iWildCam의 세 개 대표 데이터셋에서 최신 기술(state-of-the-art)과 비교하여 실험 평가를 진행하였으며, 극세분류(fine-grained classification) 정확도가 기존 모델 대비 각각 15.6% 및 23.5% 향상되었고, SIC 점수(SIC score)는 64.3% 개선되었습니다.



### How Does Diverse Interpretability of Textual Prompts Impact Medical Vision-Language Zero-Shot Tasks? (https://arxiv.org/abs/2409.00543)
- **What's New**: 이번 연구에서는 Medical Vision Language Pre-training (MedVLP) 모델의 문맥 생성 민감도를 체계적으로 평가하여 다양한 텍스트 프롬프트에 대한 안정성을 연구하였습니다. 특히, 15종의 질병에 대한 세 가지 주요 MedVLP 방법의 성능을 검토하였습니다.

- **Technical Details**: 연구에서는 GPT-4o를 활용하여 발전된 6가지 프롬프트 스타일을 생성하였습니다. BioViL, MedKLIP, KAD 모델이 이 프롬프트를 통해 ChestX-ray14, CheXpert, COVIDx CXR-4와 같은 공공 데이터 세트에서 평가되었습니다.

- **Performance Highlights**: 모델의 성능은 원래 프리트레이닝 시 사용된 스타일과 다른 스타일의 프롬프트를 사용할 때 평균 10.17% 감소하였으며, 이는 MedVLP 모델의 강건성 부족을 시사합니다. 프롬프트의 해석 가능성이 높아질수록 복잡한 의료 개념 이해에 어려움을 겪는 것으로 나타났습니다.



### Incremental Open-set Domain Adaptation (https://arxiv.org/abs/2409.00530)
- **What's New**: 본 논문은 Incremental Open-Set Domain Adaptation (IOSDA) 문제를 도입하여 시각적 인식 시스템의 새로운 도전 과제를 다룹니다. IOSDA를 통해 모델의 지속적인 성능 저하를 방지하고 새로운 도메인에 대한 적응을 가능하게 합니다.

- **Technical Details**: IOSDA에서는 이미지 분류를 위한 새로운 비지도 점진적 개방형 도메인 적응 문제를 제안합니다. 모델은 여러 도메인을 순차적으로 학습하며, 첫 번째 모듈은 생성적 프레임워크를 사용해 랜덤 노이즈에서 이전 도메인을 복제하고, 두 번째 단계에서는 이 가상의 소스 도메인을 현재의 대상 도메인에 적응시킵니다.

- **Performance Highlights**: 모델은 Office-Home, DomainNet, 및 새로 준비된 UPRN-RSDA 데이터셋에서 실험을 진행하였으며, 다양한 도메인에서의 성능 향상을 달성했습니다.



### EraseDraw: Learning to Insert Objects by Erasing Them from Images (https://arxiv.org/abs/2409.00522)
- **What's New**: 이 논문은 EraseDraw라는 새로운 시스템을 제안합니다. EraseDraw는 자연 이미지에서 객체를 언어 프롬프트를 기반으로 효과적으로 삽입하는 작업을 학습하기 위한 스케일 가능한 데이터 생성 파이프라인을 제공합니다.

- **Technical Details**: 제안된 방법은 사진 및 물리적 현실성을 갖춘 객체 제거 기술을 이용하여 언어 조건부 객체 삽입 모델을 학습합니다. 이를 위해 65,000개의 이미지를 생성하고, 이러한 데이터셋으로 대형 사전 학습된 diffusion 모델을 세밀하게 조정합니다.

- **Performance Highlights**: 모델은 in-the-wild 이미지에서 객체 삽입 작업에 대해 최첨단 결과를 달성했으며, 기존 방식에 비해 적은 컴퓨팅 자원으로도 뛰어난 성능을 보였습니다.



### Mapping earth mounds from spac (https://arxiv.org/abs/2409.00518)
Comments:
          6 pages, 4 figures, 3 tables

- **What's New**: 이번 논문에서는 기후 변화와 관련하여 스폿 랜드스케이프의 기원과 이를 식별하기 위한 자동화된 방법을 제안합니다. 특히, 딥러닝 프레임워크를 활용하여 원거리 센싱 원 데이터에서 지구 흙 더미를 자동으로 매핑하는 방법을 탐색합니다.

- **Technical Details**: 연구는 남미와 아프리카의 4가지 유형의 스폿 랜드스케이프를 분석하며, 기후 변화에 대한 적응성과 생태계의 생산성 증대와 관련된 지구 흙 더미의 패턴을 탐구합니다. 딥 러닝 모델을 사용하여 여러 지형과 지역적 특성을 고려하여 연구를 진행합니다. 생태계 엔지니어인 흰개미의 역할과 관련된 다양한 원리가 이들 환경에서 어떻게 작용하는지를 이해하는 것이 중요합니다.

- **Performance Highlights**: 우리는 다양한 최신 딥 네트워크를 평가하여 다양한 특성을 가진 스폿 랜드스케이프에서 성능을 비교합니다. 연구 결과, 자동 화 및 정확한 매핑이 이루어지기 위해서는 추가 연구가 필요함을 확인했습니다. 시스템의 복잡성이 높아 모든 지형에서 일관된 성능을 내기 위해서는 더욱 많은 데이터와 고급 모델링이 요구됩니다.



### Plant detection from ultra high resolution remote sensing images: A Semantic Segmentation approach based on fuzzy loss (https://arxiv.org/abs/2409.00513)
Comments:
          5 pages, 5 figures, 2 tables

- **What's New**: 이 연구에서는 초고해상도 (UHR) 원격 감지 이미지에서 식물 종을 식별하는 도전 과제를 해결하기 위한 접근 방식을 제시합니다. RGB 원격 감지 데이터셋을 도입하였고, 이 데이터셋은 프랑스의 산악 지역에서 여러 필드 원정에 걸쳐 세심하게 수집되었습니다.

- **Technical Details**: 연구에서는 의미론적 분할 문제로 식물 식별 과제를 정의하며, 이 과정에서 기존의 one-hot 인코딩된 실제 라벨 대신, 가우시안 필터를 이용하여 개선된 GT를 적용합니다. 이 모델은 각 픽셀의 클래스 소속 가능성을 모델링하고, 랜덤성을 도입하기 위해 새로운 손실 함수인 fuzzy loss를 제안합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법론의 유효성이 입증되었으며, 향후 개선의 필요성 또한 언급되었습니다. 본 연구는 실질적인 필드 관찰을 통해 획득한 초고해상도 RGB 이미지로 구성된 새로운 데이터셋을 선보이며, 기존 접근법 대비 효율성과 정확성을 올리는 중요한 기초 자료로 자리매김할 것으로 기대됩니다.



### RevCD -- Reversed Conditional Diffusion for Generalized Zero-Shot Learning (https://arxiv.org/abs/2409.00511)
- **What's New**: 본 논문에서는 Reversed Conditional Diffusion (RevCD) 모델을 제안하여, Generalized Zero-Shot Learning (GZSL) 문제를 해결하고자 합니다. RevCD 모델은 시각적 입력에서 합성된 의미적 특징을 생성하여, 시각적 데이터 기반으로 의미 공간을 역으로 생성하는 혁신적인 접근 방식을 제공합니다.

- **Technical Details**: RevCD 모델은 교차 Hadamard-Addition 임베딩과 다중 헤드 비주얼 트랜스포머(Multi-headed visual transformer)로 구성되어 있으며, 시각적 특징과 의미적 정보 간의 관계를 활용해 효율적인 지식 전이를 촉진합니다. 본 연구에서는 학습 비용을 절감하고, 의미적 정보와 일대일 교육을 요구하지 않도록 디자인되었습니다.

- **Performance Highlights**: RevCD 모델은 교차 데이터셋 평가를 통해 검증되었으며, 제안된 접근 방식이 기존의 기술들보다 향상된 일반화 능력을 나타내고, 비선형 데이터 복잡성을 포착하는 데 있어 더 뛰어난 성능을 보여주었습니다.



### Streamlining Forest Wildfire Surveillance: AI-Enhanced UAVs Utilizing the FLAME Aerial Video Dataset for Lightweight and Efficient Monitoring (https://arxiv.org/abs/2409.00510)
Comments:
          accpeted by Proceedings of the International Conference on Intelligent Robots and Systems (2024 IROS)

- **What's New**: 최근 UAV(무인 항공기)가 재난 긴급 대응 지원에서 중요한 역할을 수행한다는 점을 강조하며, 새로운 경량(weighted) 비디오 이해 정책 네트워크를 제안합니다. 이 연구는 향후 정보(station point)를 활용하여 정확성을 높이고, 컴퓨팅 자원을 절감하는 방법을 소개합니다.

- **Technical Details**: 제안하는 AccSampler 모델은 Adaptive Clip-aware Compression and Frame Sampling Network의 약자로, 비디오 클립을 압축하고 중복된 프레임을 제거하여 효율적인 비디오 처리(here, video understanding)를 가능하게 합니다. 이 과정에서 정책 네트워크가 프레임 중요도를 평가하고, 신뢰도 높은 프레임을 선택하여 데이터 세트를 정제(data distillation)합니다. 또한, 실험은 FLAME 데이터셋을 사용하였으며, 13배 이상의 계산 비용 절감과 3%의 정확성 향상을 이루어냈습니다.

- **Performance Highlights**: AccSampler는 Aerial Video Understanding 문제에서 우수한 성능을 보여주며, 경량화된 아키텍처 덕분에 전통적인 딥 러닝 모델에 비해 효율성과 정확성을 유지하면서도 훈련 과정에서의 시간을 대폭 절감할 수 있는 장점을 제공합니다.



### Accurate Compression of Text-to-Image Diffusion Models via Vector Quantization (https://arxiv.org/abs/2409.00492)
Comments:
          project page: this https URL

- **What's New**: 본 논문에서는 텍스트-이미지 확산(diffusion) 모델의 새로운 포스트 트레이닝 양자화(post-training quantization, PTQ) 방법을 제안합니다. 이 방법은 변환기(vector-based) 방법을 활용하여 이전 4비트 양자화 방식보다 우수한 성능을 보여주는 고압축 기법을 목표로 합니다.

- **Technical Details**: 양자화 방식은 스칼라 양자화(scalar quantization)와 벡터 양자화(vector quantization)로 나눌 수 있으며, 벡터 양자화를 통해 모델 성능을 유지하면서 더 높은 압축률을 달성합니다. 연구팀은 2B 이상의 매개변수를 가진 최신 SDXL 및 SDXL-Turbo 모델에 대해 벡터 기반 PTQ 방법을 맞춤 설계하였습니다.

- **Performance Highlights**: 양자화 엘라스틱스의 결과, 3비트로 압축된 모델이 이전의 4비트 양자화 기술과 유사한 이미지 품질 및 텍스트 정렬을 유지함을 입증했습니다. 이는 자원 제한 환경에서도 사용할 수 있는 강력한 솔루션이 될 것입니다.



### Geospatial foundation models for image analysis: evaluating and enhancing NASA-IBM Prithvi's domain adaptability (https://arxiv.org/abs/2409.00489)
- **What's New**: 이번 논문은 NASA-IBM의 GFM Prithvi를 평가하며, 고해상도 원거리 감지 이미지에 대한 오픈 소스 시각 기초 모델의 초기 사례로 주목받고 있습니다.

- **Technical Details**: Prithvi는 고급 이미지 분석 작업에서 예측 성능을 평가하여 다양한 벤치마크 데이터 세트에서 실험이 진행되었습니다. 새로운 전략으로는 band adaptation, multi-scale feature generation, fine-tuning techniques 등이 통합된 이미지 분석 파이프라인이 특징입니다.

- **Performance Highlights**: Prithvi의 성능 분석을 통해 장단점이 도출되었으며, 이는 Prithvi 개선뿐만 아니라 지리공간 작업을 위한 차세대 시각 기초 모델 개발에 유용한 통찰력을 제공합니다.



### TrackSSM: A General Motion Predictor by State-Space Mod (https://arxiv.org/abs/2409.00487)
- **What's New**: TrackSSM을 제안하여 기존의 비효율적이고 비효과적인 모션 모델의 한계를 극복하였습니다. 이 모델은 데이터 기반의 상태 공간 모델(SSM)을 활용하여 향상된 다중 객체 추적(MOT) 성능을 보여줍니다.

- **Technical Details**: TrackSSM은 향상된 인코더-디코더 구조를 바탕으로 한 모션 프레임워크입니다. 특히 Flow-SSM 모듈을 통해 객체 경계 상자의 시간적 상태 전환을 가이드하며, Step-by-Step Linear(S2L) 훈련 전략을 도입하여 경로 흐름 정보를 효과적으로 활용합니다. Mamba-Block을 이용한 간단한 모션 인코더로 다중 시나리오에 적합한 모형을 구축합니다.

- **Performance Highlights**: TrackSSM은 YOLOX 검출기와 결합할 경우 27.5 FPS의 속도에 도달하며, 평균 HOTA 성능에서 기존 Kalman Filter(KF) 모델과 유사한 결과를 보여줍니다. DanceTrack과 SportsMOT 벤치마크에서 각각 57.7 HOTA와 74.4 HOTA의 성과를 기록하며 추적 성능을 크게 향상시켰습니다.



### Multi-scale Multi-instance Visual Sound Localization and Segmentation (https://arxiv.org/abs/2409.00486)
- **What's New**: 새로운 Multi-scale Multi-instance Visual Sound Localization (M2VSL) 프레임워크를 제안하여 다중 스케일의 시각적 특징과 음향 신호를 효과적으로 정렬하고, 소리 발생 객체의 위치를 정확하게 파악할 수 있게 합니다.

- **Technical Details**: M2VSL은 다중 스케일 시각적 특징을 학습하여 관련 이미지의 다양한 레벨에서 오디오-비주얼 표현을 정렬합니다. 또한, Multi-scale Multi-instance Transformer(MMT)를 도입하여 다중 스케일의 크로스 모달 표현을 동적으로 집계합니다. 이 프레임워크는 약한 감독하에 비주얼 사운드 로컬라이제이션을 가능하게 합니다.

- **Performance Highlights**: VGGSound-Instruments, VGG-Sound Sources 및 AVSBench 벤치마크에서 광범위한 실험을 통해 M2VSL이 사운드 객체 로컬라이제이션 및 세분화에서 최고의 성능을 입증했습니다.



### Studying the Effects of Self-Attention on SAR Automatic Target Recognition (https://arxiv.org/abs/2409.00473)
- **What's New**: 이번 논문에서는 합성 개구 레이더(SAR) 자동 목표 인식(ATR) 시스템에서 주목 메커니즘(attention mechanisms)의 중요성을 강조하고, 전통적인 SAR ATR 모델의 한계를 극복하는 방법을 제시합니다. 주목 메커니즘을 통해 차량의 그림자와 같은 중요한 이미지 요소에 초점을 맞춰 목표 분류의 정확성을 향상시키는 방법을 논의합니다.

- **Technical Details**: SAR ATR 시스템은 종종 고차원이고 노이즈가 많은 데이터로 인해 목표를 정확하게 식별하는 데 어려움을 겪습니다. 본 논문에서는 ResNet-18(딥러닝 모델) 기반의 CBAM(Convolutional Block Attention Module) 및 SENet, ECANet의 자가 주목(self-attention) 메커니즘을 활용하여 모델의 성능을 평가합니다. Grad-CAM(Gradient-weighted Class Activation Mapping) 기법을 통해 SAR 이미지 내의 중요한 픽셀을 시각화합니다.

- **Performance Highlights**: 주목 모듈을 적용한 결과, MSTAR 데이터셋에서 top-1 정확도가 증가하고 입력 강건성(input robustness)이 향상되며, 그 결과로 다양한 환경에서 더 실용적이고 견고한 SAR ATR 모델을 만들 수 있음을 보여줍니다. 이는 모델의 결정 과정을 더 명확하게 이해하는 데에도 기여합니다.



### ActionPose: Pretraining 3D Human Pose Estimation with the Dark Knowledge of Action (https://arxiv.org/abs/2409.00449)
- **What's New**: ActionPose는 2D 인간 자세를 3D로 변환하는 과정을 개선하기 위해 동작 지식(action knowledge)을 활용하는 새로운 프레임워크입니다. 이 프레임워크는 동작 임베딩(motion embeddings)과 세밀한 동작 레이블의 텍스트 임베딩(text embeddings)을 정렬하여 깊이 모호성과 가림 현상(occlusion)을 극복하고자 합니다.

- **Technical Details**: ActionPose는 두 단계로 운영됩니다: 사전학습(pretraining)과 미세조정(fine-tuning). 사전학습 단계에서는 모델이 마스크된(masked) 및 노이즈가 포함된 2D 자세로부터 동작을 인식하고 3D 자세를 재구성하는 법을 학습합니다. 미세조정 단계에서는 실제 3D 인간 자세 추정 데이터셋을 사용하여 추가 개선됩니다. 또한, 본 프레임워크는 동작 모델링에서 마스킹된 신체 부위와 마스킹된 시간 창을 포함하여 시간적 및 공간적 경계의 모호성을 줄입니다.

- **Performance Highlights**: ActionPose는 Human3.6M과 MPI-INF-3DHP와 같은 공공 데이터셋에서 최첨단 성능을 달성하였습니다. 구체적으로, Human3.6M에서 동작 검출된 2D 자세를 입력으로 사용하여 36.7mm의 MPJPE(Mean Per Joint Position Error)를, MPI-INF-3DHP에서는 실측 2D 자세를 입력으로 하여 15.5mm를 기록했습니다.



### A Hybrid Transformer-Mamba Network for Single Image Deraining (https://arxiv.org/abs/2409.00410)
Comments:
          12 pages, 9 figures

- **What's New**: 본 연구는 비가 내리는 이미지에서 잔여 물방울을 효과적으로 제거하기 위해 이중 분기 하이브리드 Transformer-Mamba 네트워크(TransMamba)를 제안합니다.

- **Technical Details**: TransMamba 모델은 두 개의 분기로 구성되어 있습니다: Transformer 분기와 Mamba 분기로, 각각 rain streaks(비 줄무늬) 제거 프로세스에서 고유하고 보완적인 역할을 수행합니다. 첫 번째 분기는 spectral-banded Transformer 블록을 사용하여 장거리 зависимости(dependencies)를 모델링합니다. 두 번째 분리는 bi-directional state space model 모듈로 구성되어 있어 지역 및 글로벌 정보를 추가로 캡처합니다. 또한, spectral coherence loss를 통해 깨끗한 이미지 내 신호 수준 관계 복원을 개선합니다.

- **Performance Highlights**: 다양한 데이터셋과 실제 이미지에 대한 광범위한 실험을 통해, 본 방법은 기존의 최첨단 접근법에 비해 우수한 rain streak 제거 성능을 나타냅니다.



### COSMo: CLIP Talks on Open-Set Multi-Target Domain Adaptation (https://arxiv.org/abs/2409.00397)
Comments:
          Accepted in BMVC 2024

- **What's New**: 이 논문은 COSMo라는 새로운 방법을 소개하며, 이는 Multi-Target Domain Adaptation (MTDA) 문제를 제어하는 오픈세트(Open-Set) 환경에서의 도전 과제를 다룬다. 기존의 MTDA 방법들이 주로 시각적 특징에 초점이 맞춰져 있었으나, COSMo는 도메인 독립적인 프롬프트를 학습하여 알려진 및 알려지지 않은 클래스 각각에 대한 기울기를 제공하여 도메인 및 클래스 이동에 효과적으로 적응한다.

- **Technical Details**: COSMo는 소스 도메인 기반의 프롬프트 학습(Source Domain-Guided Prompt Learning)을 활용하여 도메인 불변 정보를 학습하는 새로운 방법을 제시한다. 이 방법은 도메인 특화 편향 네트워크(Domain-Specific Bias Network)를 사용하여 여러 타겟 도메인에서의 클래스 이동을 처리하고, 알려진 클래스와 알려지지 않은 클래스에 대한 개별 프롬프트를 통해 성능을 향상시킨다.

- **Performance Highlights**: COSMo는 Mini-DomainNet, Office-31, Office-Home의 세 가지 도전적인 데이터세트에서 OSMTDA 설정에 적합하게 조정된 다른 DA 방법들과 비교하여 평균 5.1%의 성능 향상을 보였다. 이는 클립(CLIP)과 같은 대규모 비전-언어 모델들이 MTDA의 잠재력 탐구에서 충분히 활용되지 않았음을 보여준다.



### Self-supervised Fusarium Head Blight Detection with Hyperspectral Image and Feature Mining (https://arxiv.org/abs/2409.00395)
Comments:
          Beyond Visible Spectrum: AI for Agriculture Challenge, in conjunted with ICPR 2024

- **What's New**: 이 연구에서는 Fusarium Head Blight (FHB)의 효과적인 탐지를 위한 자가 비지도 학습(self-unsupervised) 분류 방법을 제안합니다. 이 방법은 Hyper-spectral Imaging (HSI)와 함께 endmember extraction 전략과 top-K bands 선택을 활용하여 HSI 데이터에서 구별 가능한 특징 표현을 추출합니다.

- **Technical Details**: 제안된 방법은 HSI의 스펙트럼 정보를 효과적으로 활용하며, 복잡한 데이터 처리 없이도 mild-FHB와 serious-FHB를 분리할 수 있습니다. Normalization 및 spectral averaging 기법을 통해 데이터의 복잡성을 줄이고, K-means 클러스터링을 사용하여 가상의 라벨을 생성하여 top-K 중요한 밴드를 선택합니다. LightGBM을 분류기로 사용하여 고차원 데이터를 효율적으로 처리합니다.

- **Performance Highlights**: 검증 결과, 제안된 방법은 Beyond Visible Spectrum: AI for Agriculture Challenge 2024에서 효과적으로 입증되었습니다. 실험에 사용된 HSI 데이터는 DJI M600 Pro UAV 시스템을 통해 수집되었으며, 정확도를 평가하는 데 사용된 주요 지표는 정확도, 진양성(TP), 진음성(TN), 위양성(FP), 위음성(FN)입니다.



### A method for detecting dead fish on large water surfaces based on improved YOLOv10 (https://arxiv.org/abs/2409.00388)
- **What's New**: 이 논문에서는 물 위에서 죽어 있는 물고기를 신속하고 정확하게 탐지하기 위한 YOLOv10 프레임워크를 기반으로 한 엔드 투 엔드 탐지 모델을 제안합니다. FasterNet을 사용하여 모델의 복잡성을 줄이고, Neck 섹션의 특징 융합 방식을 개선하며, 작은 객체의 탐지 성능을 높이기 위해 컴팩트한 목표 탐지 헤드를 추가했습니다.

- **Technical Details**: 모델의 주요 기술적 개선 사항으로는 (1) YOLOv10의 백본 네트워크를 FasterNet으로 교체하여 높은 탐지 정확도를 유지하면서 모델 복잡성을 감소시킴; (2) Neck 섹션의 특징 융합을 개선하기 위한 향상된 연결 방식과 CSPStage 모듈로 C2f 모듈을 교체; (3) 작은 물체에 대한 탐지 성능을 향상시키기 위한 컴팩트 목표 탐지 헤드를 추가했습니다. 또한, ablation 실험을 통해 각 모델 구성 요소의 전체 시스템 성능에 대한 기여를 체계적으로 분석했습니다.

- **Performance Highlights**: 실험 결과, P(정밀도), R(재현율), AP(평균 정밀도)에서 YOLOv10n 모델 대비 상당한 향상이 있음을 보여주었으며, 모델 크기와 파라미터 수를 대폭 줄이면서도 높은 추론 속도를 유지하고 최적의 AP 성능을 달성했습니다. 이 모델은 대규모 양식 시스템에서의 죽은 물고기 신속하고 정확한 탐지를 가능하게 합니다.



### 3D Gaussian Splatting for Large-scale 3D Surface Reconstruction from Aerial Images (https://arxiv.org/abs/2409.00381)
Comments:
          11 pages

- **What's New**: 이번 연구에서는 3D Gaussian Splatting(3DGS)을 기반으로 하여 대규모 다중 시점 스테레오(Multi-View Stereo, MVS) 공중 이미지를 위한 최초의 대규모 표면 재구성 방법, Aerial Gaussian Splatting(AGS)을 제안합니다.

- **Technical Details**: AGS는 대규모 항공 이미지를 위해 데이터 청크(chunking) 방법을 도입하고, Ray-Gaussian Intersection 방법을 통합하여 깊이(depth) 및 법선(normal) 정보를 얻습니다. 또한, 다중 뷰 기하학적 일관성 제약을 추가하여 전역 기하학적 일관성을 향상하고 재구성 정확도를 개선합니다.

- **Performance Highlights**: 다양한 데이터 세트에서 수행한 실험 결과, AGS는 전통적인 항공 MVS 방법과 기하학적 정확도에서 동등한 성능을 보여주었으며, GS 기반 방법들보다 기하학적 및 렌더링 품질에서 더 우수한 결과를 얻었습니다.



### First Competition on Presentation Attack Detection on ID Card (https://arxiv.org/abs/2409.00372)
- **What's New**: 2024년 국제 합동 생체인식 회의에서 개최된 ID 카드 프레젠테이션 공격 탐지 대회(PAD-IDCard)의 결과가 발표되었습니다. 이 대회는 학계와 산업계에서 10개 팀이 참여하였고, 5개의 유효한 제출물이 평가되었습니다.

- **Technical Details**: 대회는 ID 카드의 프레젠테이션 공격을 탐지하기 위한 최신 알고리즘을 독립적으로 평가하기 위해 다양한 나라의 ID 카드 이미지를 포함한 비공식적인 데이터 세트를 사용했습니다. 이 데이터 세트는 스페인, 칠레, 아르헨티나, 코스타리카에서 수집된 생체 이미지와 합성 공격 이미지로 구성되었습니다.

- **Performance Highlights**: 비공식 팀이 평균 74.80%의 성적을 기록하여 최상의 결과를 올렸으며, 'IDVC' 팀은 77.65%의 성적을 보였습니다. 이는 실제로 더 현실적인 테스트 조건에서 모델의 성능을 평가한 첫 번째 독립적 평가로 여겨집니다.



### UDGS-SLAM : UniDepth Assisted Gaussian Splatting for Monocular SLAM (https://arxiv.org/abs/2409.00362)
- **What's New**: 최근 UniDepth 네트워크를 사용한 단안 신경망 깊이 추정의 발전이 Gaussian splatting 프레임워크와 통합되는 연구를 촉진하였습니다. 본 연구에서는 RGB-D 센서 없이 깊이 추정을 가능하게 하는 UDGS-SLAM이라는 새로운 접근법을 제시합니다.

- **Technical Details**: UDGS-SLAM은 통계적 필터링(statistical filtering)을 사용하여 추정된 깊이의 지역 일관성(local consistency)을 보장하고, 카메라 경로(camera trajectory) 및 Gaussian 장면 표현(Gaussian scene representation) 파라미터를 공동 최적화(joint optimization)합니다. 이 방법은 고충실도(rendered images) 이미지를 생성하며, 카메라 경로의 낮은 ATERMSE를 달성합니다.

- **Performance Highlights**: UDGS-SLAM은 TUM RGB-D 데이터셋을 활용하여 엄격하게 평가되었으며, 여러 기준 방법(baseline methods)과 비교하여 다양한 시나리오에서 우수한 성능을 보여줍니다. 또한, 디자인 선택을 검증하고 다양한 네트워크 백본 인코더(backbone encoders)가 시스템 성능에 미치는 영향을 조사하기 위한 ablation study도 수행되었습니다.



### RI-MAE: Rotation-Invariant Masked AutoEncoders for Self-Supervised Point Cloud Representation Learning (https://arxiv.org/abs/2409.00353)
- **What's New**: 본 논문은 회전 불변성을 갖는 점군(point cloud) 데이터의 자기 지도 학습을 위한 새로운 Masked AutoEncoders인 RI-MAE를 제안합니다. 이는 기존의 방법이 회전에 민감하여 성능이 저하되는 문제를 해결하고자 합니다.

- **Technical Details**: RI-MAE는 회전 불변 latent representation을 달성하기 위해 RI-Transformer를 도입하며, 이 Transformer는 disentangled geometry content, 회전 불변적인 상대 방향 및 위치 임베딩 메커니즘을 갖추고 있습니다. 이를 통해 회전 불변 점군 latent 공간을 구축합니다. 또한, 학생-교사 아키텍처를 통해 masked 패치의 자기 지도 재구성을 가능하게 합니다.

- **Performance Highlights**: RI-MAE는 다양한 다운스트림 작업에서 최첨단 성능을 달성하며, 회전에 강한 내성을 보입니다. 실험 결과는 제안된 방법이 로테이션 변동에 대해 강력하게 작용함을 보여줍니다.



### ToddlerAct: A Toddler Action Recognition Dataset for Gross Motor Development Assessmen (https://arxiv.org/abs/2409.00349)
Comments:
          Accepted by 2024 ECCV ABAW Workshop

- **What's New**: 이 논문에서는 유아의 대근육 운동 발전을 평가하기 위해 ToddlerAct라는 새로운 데이터셋을 제안합니다. 이 데이터셋은 3세 이하 유아의 다양한 대근육 활동을 비디오로 기록한 자료를 포함하고 있습니다.

- **Technical Details**: ToddlerAct 데이터셋은 유아의 운동 인식을 위한 데이터 수집 과정, 주석(annotation) 방법론, 그리고 데이터셋의 특징을 자세히 설명합니다. 또한, 이미지 기반(image-based) 및 골격 기반(skeleton-based) 행동 인식 방법을 포함한 여러 최신 방법들을 우리의 데이터셋에서 벤치마킹(benchmarking) 했습니다.

- **Performance Highlights**: 유아의 대근육 운동 발전을 정확히 평가하기 위해서는 도메인 특별 데이터셋의 중요성을 강조하며, 향후 이 분야의 연구를 위한 기초를 마련합니다. 논문에서 개발한 데이터셋은 특정 URL을 통해 제공될 예정입니다.



### SMAFormer: Synergistic Multi-Attention Transformer for Medical Image Segmentation (https://arxiv.org/abs/2409.00346)
Comments:
          Accepted by BIBM 2024

- **What's New**: SMAFormer는 작은 종양 및 장기 분할을 위한 다채로운 주의 메커니즘을 통합한 고급 Transformer 기반 모델입니다.

- **Technical Details**: SMAFormer는 Pixel Attention, Channel Attention, Spatial Attention을 결합한 Synergistic Multi-Attention (SMA) Transformer 블록과 주의 메커니즘 전환 중 정보 손실을 해결하는 Feature Fusion Modulator로 구성됩니다.

- **Performance Highlights**: SMAFormer는 LiTS2017 및 ISICDM2019 데이터셋에서 Swin UNETR을 초월하는 최신 성능을 달성했습니다.



### PS-StyleGAN: Illustrative Portrait Sketching using Attention-Based Style Adaptation (https://arxiv.org/abs/2409.00345)
- **What's New**: 이 논문에서는 초상화 스케치 생성을 위한 새로운 접근 방식인 Portrait Sketching StyleGAN (PS-StyleGAN)을 소개합니다. 이 방법은 StyleGAN의 시맨틱 W+ 잠재 공간을 활용하여 초상화 스케치를 생성하며, 포즈 및 표정 변경과 같은 의미 있는 수정을 가능하게 합니다.

- **Technical Details**: PS-StyleGAN은 Attentive Affine transform blocks를 포함한 아키텍처를 통해 스타일 잠재 코드를 조정하여 출력 변화를 가능하게 하며, 이 과정에서 모델을 세밀하게 조정할 필요가 없습니다. 이러한 블록들은 내용과 스타일의 잠재적 특징을 고려하여 스타일을 수정하는 방식으로 학습합니다.

- **Performance Highlights**: PS-StyleGAN은 다양한 데이터셋에서 최첨단 방법들과 비교해 질적 및 양적으로 우수성을 입증하였으며, 훈련 시간도 짧고 적은 수의 쌍 예시(약 100개)만으로 스타일을 모델링할 수 있습니다.



### EgoHDM: An Online Egocentric-Inertial Human Motion Capture, Localization, and Dense Mapping System (https://arxiv.org/abs/2409.00343)
- **What's New**: EgoHDM는 6개의 관성 측정 장치(IMUs)와 일반적인 헤드 마운트 RGB 카메라를 사용하는 온라인 에고센트릭-관성 인간 모션 캡처(mocap), 로컬라이제이션 및 밀집 맵핑 시스템입니다. 이 시스템은 거의 실시간으로 밀접한 장면 맵핑을 제공하는 첫 번째 인간 mocap 시스템입니다.

- **Technical Details**: EgoHDM는 카메라 로컬라이제이션 및 맵핑 정보를 관성 인간 모션 캡처와 이중으로 통합하는 시스템입니다. 이 과정에서 로컬 바디 중심의 고도 맵을 활용하여 모션 캡처 인식 밀집 번들 조정(mocap-aware dense bundle adjustment)과 물리 기반 몸 자세 교정 모듈을 설계하였습니다.

- **Performance Highlights**: EgoHDM는 기존 기술에 비해 인간 로컬라이제이션 오류를 41%, 카메라 자세 오류를 71%, 맵핑 정확도를 46% 개선하였습니다. 또한, 다양한 비평면 지형에서의 도전적인 시나리오를 성공적으로 처리할 수 있는 성능을 입증했습니다.



### AdaNAT: Exploring Adaptive Policy for Token-Based Image Generation (https://arxiv.org/abs/2409.00342)
Comments:
          Accepted by ECCV2024

- **What's New**: 최근 연구가 이미지 생성에서의 token 기반 방법의 효과를 입증했습니다. 이 연구에서는 학습 가능한 접근법인 AdaNAT를 제안하며, 각 샘플에 적합한 정책을 자동으로 구성할 수 있는 방법을 제시하고 있습니다.

- **Technical Details**: AdaNAT는 Markov decision process (MDP)로 정책 결정 과정을 공식화하여 강화학습(reinforcement learning)을 통해 경량화된 정책 네트워크를 학습합니다. 이를 위해 기존의 표준 평가 메트릭인 Fréchet Inception Distance (FID)와 사전 학습된 보상 모델을 테스트하였으나, 이러한 방법들이 원하는 품질과 다양성을 확실히 보장하지 못하는 단점을 발견했습니다. 따라서 적대적 보상 모델을 도입하여 정책 네트워크의 훈련을 효과적으로 안내합니다.

- **Performance Highlights**: AdaNAT는 ImageNet, MS-COCO 및 CC3M과 같은 네 가지 벤치마크 데이터셋에서 널리 검증하였으며, NATs의 성능을 40% 상대적으로 향상시키는 결과를 보였습니다. 이는 AdaNAT가 각 샘플에 맞게 생성 정책을 조정하는 능력 덕분입니다.



### Aligning Medical Images with General Knowledge from Large Language Models (https://arxiv.org/abs/2409.00341)
- **What's New**: 본 논문에서는 CLIP와 같은 사전 학습된 대형 비전-언어 모델(VLMs)을 기반으로 한 새로운 의료 이미지 분석 프레임워크인 ViP(Visual Symptom-guided Prompt learning framework)를 제안합니다. ViP는 시각적 증상 생성기(VSG)와 이중 프롬프트 네트워크로 구성되어 있어 CLIP의 일반 지식 전이를 촉진합니다.

- **Technical Details**: ViP는 두 가지 주요 구성 요소, 즉 시각적 증상 생성기(VSG)와 이중 프롬프트 네트워크로 이루어져 있습니다. VSG는 LLMs에게 쿼리하여 의료 이미지 분석에 필요한 설명 가능한 시각적 증상을 생성하며, 이 이중 프롬프트 네트워크는 두 개의 학습 가능한 프롬프트 모듈인 맥락 프롬프트(CoP)와 병합 프롬프트(MeP)를 통해 VLMs에게 의학적 이미지 분석을 효과적으로 적응시킵니다.

- **Performance Highlights**: ViP는 Pneumonia 및 Derm7pt 두 개의 도전적인 데이터셋에서 사전학습된 기존 방법들보다 우수한 성능을 보였습니다. 이는 각 구성 요소가 프레임워크에서 효능을 가지고 있음을 강조합니다.



### Fish Tracking Challenge 2024: A Multi-Object Tracking Competition with Sweetfish Schooling Data (https://arxiv.org/abs/2409.00339)
Comments:
          5 pages, 1 figure

- **What's New**: 2024 Fish Tracking Challenge는 수조에서의 집단 어류 행동을 자동으로 추적하고 분석하는 모델 개발을 촉진하기 위해 설계되었습니다. SweetFish 데이터셋을 활용하여 10마리의 달고기(sweetfish)를 동시에 추적하는 새로운 기술적 도전이 소개됩니다.

- **Technical Details**: 이 대회의 주요 목표는 10마리의 달고기(sweetfish)를 정확하게 추적하는 것입니다. 참가자들은 YOLOv8 객체 탐지기와 다양한 추적 알고리즘을 이용하여 모델을 개발합니다. 데이터셋은 165,150개의 주석(bounding box)을 포함하고 있으며, HOTA (Higher Order Tracking Accuracy) 점수를 사용하여 성능을 평가합니다.

- **Performance Highlights**: 최고의 성능 모델들은 YOLOv8을 활용하고, ByteTrack 및 BoTSORT와 같은 최첨단 추적 알고리즘을 사용하였습니다. 우승한 팀의 접근 방식은 영상 데이터와 바운딩 박스 주석을 기반으로 자동 탐지 및 추적 알고리즘의 혁신을 촉진하여, 집단 동물 행동의 동역학을 이해하는 데 기여할 것입니다.



### GMFL-Net: A Global Multi-geometric Feature Learning Network for Repetitive Action Counting (https://arxiv.org/abs/2409.00330)
- **What's New**: 이번 논문에서는 GMFL-Net(Global Multi-geometric Feature Learning Network)을 제안하며, 다양한 기하학적 정보를 통합하여 반복적인 행동 인식의 정확도를 높이는 데 중점을 두었습니다. 또한 새로운 데이터셋 Countix-Fitness-pose를 수집하였으며, 이를 통해 모델의 성능을 검증했습니다.

- **Technical Details**: GMFL-Net은 MIA-Module(Multi-Geometric Information Aggregation Module)와 GBFL-Module(Global Bilinear Feature Learning Module)을 포함하여 다중 기하학적 특성을 융합하고, 포인트 및 채널 간의 상호 의존성을 증대시킵니다. 이 구조는 비디오에서 카메라 시점의 변화로 인한 왜곡을 안정화하고, 예외에서 실제 행동으로의 전환 시 잘못된 탐지 문제를 줄입니다.

- **Performance Highlights**: GMFL-Net은 RepCount-pose, UCFRep-pose, Countix-Fitness-pose의 세 가지 데이터셋에서 광범위한 실험을 수행하였으며, 그 결과 기존의 방법들보다 뛰어난 성능을 보였습니다. 특히, 새로운 데이터셋Countix-Fitness-pose에서 7,593개의 세밀한 포즈 주석이 제공되어 연구의 풍부함과 도전 수준을 높였습니다.



### FBD-SV-2024: Flying Bird Object Detection Dataset in Surveillance Video (https://arxiv.org/abs/2409.00317)
- **What's New**: 본 논문에서는 감시 비디오를 위한 비행 조류 데이터셋(FBD-SV-2024)을 소개하고, 이를 통해 비행 조류 탐지 알고리즘의 개발 및 성능 평가를 지원하고자 합니다. 이 데이터셋은 483개의 비디오 클립과 총 28,694 프레임을 포함하며, 그중 23,833 프레임에는 28,366개의 비행 조류 인스턴스가 포함되어 있습니다.

- **Technical Details**: FBD-SV-2024 데이터셋은 2023년 12월부터 2024년 5월까지 야외 감시 카메라를 사용하여 수집되었으며, 비디오 클립은 수동으로 스크리닝된 후 비행 조류 객체가 포함된 세그먼트로 나누어졌습니다. 데이터셋 내의 비행 조류 객체는 개별 프레임에서 불규칙한 특징과 작고 다양한 형태로 인해 탐지의 어려움을 초래합니다. 전체 비행 조류 객체 중 36.7%가 단일 프레임에서 표현되지 않거나 인식하기 어려운 것으로 평가되었습니다.

- **Performance Highlights**: 기존의 고급 객체 탐지 알고리즘을 적용하여 FBD-SV-2024에서 실험한 결과, 이 데이터셋은 알고리즘 개발 시 여전히 도전적인 환경을 제공한다는 것이 입증되었습니다. 또한, 훈련 세트와 테스트 세트로 나누어진 고비용 데이터 수집 과정을 통해 비행 조류 탐지 알고리즘의 정확한 성능 평가가 가능해졌습니다.



### Toward a More Complete OMR Solution (https://arxiv.org/abs/2409.00316)
- **What's New**: 본 연구에서는 음표 인식의 새로운 접근 방식을 제시하고 있습니다. 특히, 기존의 완벽한 객체 탐지(output)를 가정하지 않고, 불완전한 객체 탐지를 기반으로 음표 조합(notation assembly) 방식을 개선하는 방법에 중점을 두었습니다.

- **Technical Details**: MUSCIMA++ v2.0 데이터셋을 사용하여 음표를 그래프 형태로 표현하였고, YOLOv8을 기반으로 한 음악 객체 탐지기를 도입하였습니다. 또한, 검출(output)에 기반하여 음표 조합 단계를 완수하는 감독(training) 방법을 제안합니다.

- **Performance Highlights**: 이 모델은 기존의 완벽한 탐지 결과를 훈련에 사용했던 모델보다 우수한 성능을 나타내어, 탐지와 조합 단계를 통합하여 고려하는 것의 유용성을 보여주었습니다. 새로운 평가 지표인 Match+AUC를 통해 탐지 오류 및 조합 오류를 동시에 고려하여 더 완전한 OMR 솔루션을 향해 나아가는 중요한 단계로 평가됩니다.



### Towards Secure and Usable 3D Assets: A Novel Framework for Automatic Visible Watermarking (https://arxiv.org/abs/2409.00314)
Comments:
          Accepted to WACV2025

- **What's New**: 이 논문에서는 자동화된 3D 가시적 워터마크 삽입의 새로운 작업을 정의하며, 이는 워터마크 품질(watermark quality) 및 자산 유용성(asset utility) 두 가지 경쟁 측면에서 접근합니다. 또한, 고품질의 워터마크 및 자산 유용성을 보장하기 위해 3D 자산에 적절한 위치와 방향, 워터마크 수를 자동으로 결정하는 방법을 제안합니다.

- **Technical Details**: 이 방법은 경량화된 변형을 자동으로 학습하여 워터마크의 이상적인 위치를 찾기 위해 역전파(back-propagation) 기반의 경량 바디 최적화(rigid-body optimization) 기법을 사용합니다. 또한, 3D 모델에 워터마크를 융합하기 위한 새로운 곡률 일치(curvature-matching) 방법을 제안합니다. 이 과정을 통해 워터마크의 가독성(readability) 및 보안(security)을 향상시키고 있습니다.

- **Performance Highlights**: 우리는 세 가지 3D 데이터 세트를 통해 제안된 방법의 우수성을 검증하는 상세한 실험 분석을 수행하였으며, 이 연구의 결과로는 자동화된 3D 가시적 워터마킹의 새로운 작업과 이를 위한 엔드 투 엔드 파이프라인을 제시하였습니다. 최종적으로, 다양한 워터마크 및 자산 품질을 측정하기 위한 실용적인 메트릭(metrics)을 제안하고 있습니다.



### Training-Free Sketch-Guided Diffusion with Latent Optimization (https://arxiv.org/abs/2409.00313)
- **What's New**: 본 연구에서는 스케치를 추가 조건으로 포함하는 훈련 없는 파이프라인을 제안하여, 텍스트-이미지(T2I) 생성 모델의 기존 구조를 확장합니다. 이 방법을 통해 사용자는 생성된 이미지의 레이아웃과 구조를 보다 정확하게 제어할 수 있습니다.

- **Technical Details**: 이 연구는 노이즈를 포함한 잠재 공간에서 차별화된 기능을 보유한 확산 모델(diffusion models)을 활용하여, 사용자 제공 스케치를 참조하여 구조적 특징을 추적하는 방법인 cross-attention maps를 사용합니다. 궁극적으로, latent optimization 기법을 도입하여 각 생성 단계에서 고유 노이즈 잠재 변수를 최적화함으로써, 생성 이미지와 스케치 간의 일치를 보장합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 Sketchy 데이터베이스와 ImageNet-Sketch 데이터셋을 사용하여, 훈련없이 다양한 스케치 유형을 효과적으로 처리하여 이미지 생성을 성공적으로 이루어냅니다. 이는 확산 모델이 다양한 도메인에서 스케치의 핵심 객체 특징을 효과적으로 인식하고 추출할 수 있음을 보여줍니다.



### StimuVAR: Spatiotemporal Stimuli-aware Video Affective Reasoning with Multimodal Large Language Models (https://arxiv.org/abs/2409.00304)
- **What's New**: 이 논문은 MLLM을 기반으로 한 StimuVAR을 제안하며, 이는 감정적 반응에 초점을 맞춘 비디오 감정 추론(VAR) 문제에 대한 새로운 접근법입니다.

- **Technical Details**: StimuVAR은 두 가지 수준의 자극 인식 메커니즘(frame-level awareness 및 token-level awareness)을 통합하여 비디오 내의 감정 자극을 식별합니다. 이를 통해 감정 유발 이벤트가 포함된 프레임을 샘플링하고, 토큰 공간에서 감정 유발 구간을 집중적으로 선택하여 MLLM이 감정에 중점을 두게 합니다.

- **Performance Highlights**: StimuVAR은 다양한 기준에서 최첨단 성능을 보여주며, 비디오가 인간의 감정에 미치는 영향을 예측하고 신뢰할 수 있는 설명을 제공합니다.



### ContextVLM: Zero-Shot and Few-Shot Context Understanding for Autonomous Driving using Vision Language Models (https://arxiv.org/abs/2409.00301)
Comments:
          Accepted at the 27th IEEE International Conference on Intelligent Transportation Systems (ITSC) 2024

- **What's New**: 최근 자율주행차(AV) 기술의 발전이 안전한 교통 시스템을 위한 방향으로 주목받고 있습니다. 이 논문에서는 자율주행차가 극복해야 하는 다양한 환경적 도전에 대해 논의하고, 이를 위해 필요한 환경 인식(context recognition) 기술을 중점적으로 다룹니다.

- **Technical Details**: 논문에서는 총 24개의 환경적(contextual) 속성을 정의하며, 이는 기후, 조명, 교통 및 도로 조건을 포함합니다. 새로운 데이터셋인 DrivingContexts는 160만 개 이상의 context-query 쌍을 포함하며, 전통적인 supervised 컴퓨터 비전 방법에 대한 한계를 극복하기 위해 ContextVLM이라는 프레임워크를 제안합니다. 이 프레임워크는 비전-언어 모델을 활용하여 제로 및 소수 샷(zero-shot and few-shot) 방식으로 컨텍스트를 탐지합니다.

- **Performance Highlights**: ContextVLM은 DrivingContexts 데이터셋에서 95% 이상의 정확도로 관련 주행 컨텍스트를 탐지할 수 있으며, 4GB Nvidia GeForce GTX 1050 Ti GPU에서 응답 지연이 10.5 ms로 실시간으로 작동 가능합니다.



### Box2Flow: Instance-based Action Flow Graphs from Videos (https://arxiv.org/abs/2409.00295)
- **What's New**: 이 논문에서는 프로시저 비디오에서 흐름 그래프(flow graph)를 정확하고 세밀하게 생성하는 방법을 제안합니다. 기존의 방법들은 모든 비디오에서 단일 흐름 그래프를 학습하려고 했으나, 이는 세부 단계 설명을 포착하는 데 실패했습니다. 새로운 접근 방식인 Box2Flow를 통해 각 비디오에서 단계 간의 관계를 학습해 보다 효과적으로 흐름 그래프를 추출할 수 있습니다.

- **Technical Details**: Box2Flow는 프로시저 비디오의 개별 단계를 정확하게 예측하기 위해 바운딩 박스(bounding boxes)를 추출하고, 단계 쌍 간의 쌍별(edge) 확률을 예측한 후 스패닝 트리(spanning tree) 알고리즘을 사용하여 흐름 그래프를 만듭니다. 이 과정에서 객체의 상태 변화를 모니터링하여 단계 관계를 더 잘 예측합니다.

- **Performance Highlights**: MM-ReS 및 YouCookII 데이터셋에서의 실험 결과, Box2Flow가 흐름 그래프를 효과적으로 예측할 수 있음을 보여주었습니다. 특히 MM-ReS에서 누락된 프레임을 보간하여 성능을 향상시키고, Maximal Common Subgraph와 같은 구조적 평가를 통해 예측된 흐름 그래프의 정확성을 검증했습니다.



### RealFace -- Pedestrian Face Datas (https://arxiv.org/abs/2409.00283)
- **What's New**: 새로운 Real Face Dataset은 실제 환경에서 보행자 얼굴 탐지(Benchmark dataset for pedestrian face detection) 작업을 위한 귀중한 자료입니다. 11,000개 이상의 이미지와 55,000개 이상의 탐지된 얼굴을 포함하고 있으며, 다양한 환경 조건에서 수집된 실제 얼굴 이미지를 제공합니다.

- **Technical Details**: 이 데이터셋은 조명(Lighting), 크기(Scale), 자세(Pose), 가림(Occlusion) 등 다양한 환경 조건을 반영하여 알고리즘 성능을 평가하는 데 필수적인 다양성을 제공합니다. 실제 시나리오에 중점을 두어 감시(Surveillance) 애플리케이션의 어려움을 해결하는 데 도움을 줍니다.

- **Performance Highlights**: Real Face Dataset은 대규모의 얼굴 탐지 및 인식 방법을 평가할 수 있는 기회를 제공합니다. 실제 환경의 복잡성을 반영함으로써, 실용적인 애플리케이션을 위한 강력하고 효과적인 알고리즘을 개발하고자 하는 연구자와 개발자에게 중요한 자료로 자리매김하고 있습니다.



### AWRaCLe: All-Weather Image Restoration using Visual In-Context Learning (https://arxiv.org/abs/2409.00263)
- **What's New**: 이 논문은 All-Weather Image Restoration (AWIR)을 위한 새로운 접근 방식인 AWRaCLe를 제안합니다. AWRaCLe는 잠재적인 문맥 정보를 활용하여 이미지 복원 과정을 지도합니다. 이는 기존 방법의 한계를 해결하는데 기여합니다.

- **Technical Details**: AWRaCLe는 Degradation Context Extraction (DCE)와 Context Fusion (CF) 블록을 통합하여 생성된 맥락 정보에서 중요한 특성을 추출하고 융합합니다. 이 블록은 CLIP 특징을 활용하며 attention 메커니즘을 적용하여 효과적으로 정보 흐름을 관리합니다.

- **Performance Highlights**: AWRaCLe는 여러 벤치마크 데이터셋에서 AWIR 작업의 최첨단 성능을 달성하였으며, 특히 여러 유형의 악화(예: 안개, 눈, 비)가 있는 이미지를 선택적으로 복원하는 능력을 보여줍니다.



### MAPWise: Evaluating Vision-Language Models for Advanced Map Queries (https://arxiv.org/abs/2409.00255)
Comments:
          30 Pages, 46 Tables, 6 Figure

- **What's New**: 이번 연구는 Vision-Language Models (VLMs)의 choropleth maps에 대한 질문 응답 능력을 분석합니다. 새로운 데이터셋인 MAPWise를 소개하며, 미국, 인도, 중국을 포함해 1,000개의 질문을 제공합니다.

- **Technical Details**: MAPWise 데이터셋은 다양한 난이도의 질문 템플릿을 포함하며, 지리적 및 공간적 이해를 평가하기 위하여 43개의 질문 유형이 마련되었습니다. VLMs 모델을 통해 실험하였으며, Zero-Shot 및 Explicit Extraction and Reasoning 방식으로 평가하였습니다.

- **Performance Highlights**: VLMs의 성능 평가에서 쟁점과 한계를 밝혀냈으며, 새로운 벤치마크와 데이터셋을 통해 향후 연구 방향을 제시했습니다. 특히, 새롭게 도입한 MAPWise 데이터셋은 choropleth maps와 관련하여 모델의 성능을 비교하는 데 유용합니다.



### Medical Report Generation Is A Multi-label Classification Problem (https://arxiv.org/abs/2409.00250)
Comments:
          Accepted to 2024 IEEE International Conference on Medical Artificial Intelligence

- **What's New**: 이 논문에서는 의료 보고서 생성을 다중 레이블 분류 문제(multi-label classification problem)로 재구성하는 새로운 관점을 제시합니다. 기존의 시퀀스 생성 접근 방식에서 벗어나, 의료 이미지에서 중요한 키 개념을 식별하고 분류하는 데 중점을 두었습니다.

- **Technical Details**: 우리는 지식 그래프(knowledge graph)에서 얻은 방사선 노드(radiology nodes)를 활용하여 의료 보고서를 생성하기 위한 새로운 프레임워크를 소개합니다. 이 프레임워크는 BLIP(도구 이름)과 분류된 키 노드를 통합하여 의료 이미지 내의 여러 주요 측면을 효과적으로 분류하고 보고서를 작성할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, 우리는 두 개의 벤치마크 데이터셋(IU X-ray, MIMIC-CXR)에서 기존의 접근 방식보다 뛰어난 성능 지표를 달성하였으며, 제안한 접근 방식이 의료 보고서의 정확성과 명확성을 크게 향상시킬 수 있음을 입증했습니다.



### One-Frame Calibration with Siamese Network in Facial Action Unit Recognition (https://arxiv.org/abs/2409.00240)
- **What's New**: 이 논문에서는 얼굴 표정 분석에서 자주 사용되는 자동 얼굴 액션 유닛(AU) 인식을 위해, 각 얼굴에 대해 중립 표정의 이미지를 보정(reference image)으로 사용하는 단일 프레임 보정(one-frame calibration, OFC) 기법을 제안합니다.

- **Technical Details**: 제안하는 보정 방법은 Calibrating Siamese Network (CSN) 아키텍처를 기반으로 하며, 간단한 iResNet-50 (IR50) 백본을 사용해 AU 인식을 수행합니다. 이 네트워크는 두 개의 동일한 네트워크에 중립 이미지와 목표 이미지를 입력하고, 중간 단계에서 피처 맵의 차이를 계산하여 결합합니다.

- **Performance Highlights**: DISFA, DISFA+, UNBC-McMaster 데이터셋에서 OFC CSN-IR50 모델이 기존 IR50 모델의 성능을 크게 개선하고, naively OFC 방법 및 최첨단 NCG 모델들보다 AU 강도 추정 및 AU 탐지에서 뛰어난 성능을 보였습니다.



### Self-Supervised Learning for Building Robust Pediatric Chest X-ray Classification Models (https://arxiv.org/abs/2409.00231)
Comments:
          15 pages, 6 figures, 4 tables

- **What's New**: 이 연구에서는 아동 흉부 X선(CXR) 이미지를 사용하여 의료 인공지능의 진단 성능을 향상시키기 위한 새로운 접근 방식인 SCC(Self-supervised Contrastive learning with Contrast enhancement)를 제안합니다. 이 방법은 성인 CXR 모델에서의 전이 학습과 자기 지도 대조 학습을 결합한 것입니다.

- **Technical Details**: SCC는 다양한 아동 CXR 이미지를 처리하기 위해 성인 CXR 모델에서 훈련된 전이 학습을 사용하며, 대조 강화(Contrast enhancement)로 폐 부위에 집중합니다. 이 과정은 고품질 임베딩을 생성하며, 기존의 연속적인 전이 학습 세팅에 비해 훨씬 적은 수의 레이블 이미지로 유사한 성능을 달성합니다.

- **Performance Highlights**: SCC는 OOD(out-of-distribution) 상황에서의 성능을 평가한 결과, 일반적인 전이 학습보다 AUC에서 13.6% 및 34.6%의 성능 향상을 보였으며, 레이블된 이미지를 10배 적게 사용하는 few-shot learning 상황에서도 일반 전이 학습과 유사한 성능을 나타냈습니다.



### Structuring Quantitative Image Analysis with Object Prominenc (https://arxiv.org/abs/2409.00216)
Comments:
          Working Paper

- **What's New**: 이 논문은 이미지 내 객체의 당위성(object prominence)을 분석하는 새로운 접근 방식을 제안합니다. 기존의 정량적 이미지 분석 방법이 모든 영역을 동일하게 다루는 데 반해, 이 연구는 이미지 분석 과정에서 객체의 중요성을 분리하여 고려하는 것이 필수적이라고 주장합니다.

- **Technical Details**: 이 연구는 품질(qualitative) 분석과 양적(quantitative) 접근 방법의 결합을 통해 이미지 데이터 분석의 중요성을 강조합니다. 객체를 정의하고 인간이 얼마나 많은 주의를 기울이는지를 조작 가능하게 측정하는 것이 핵심입니다. 여러 명령어(descriptors)와 알고리즘을 활용하여 객체의 당위성을 시각적으로 표현하고 분석합니다.

- **Performance Highlights**: 두 가지 주요 응용 사례를 통해 우리의 접근 방식의 유용성을 보여줍니다. 첫째, 미국 신문 8개의 이미지를 기반으로 이데올로기를 분석하였으며, 둘째, 2016년과 2020년 미국 대선 캠페인 비디오에서 여성의 당위성을 분석하였습니다. 결과적으로 공화당(GOP)은 민주당(Democrats)보다 여성의 이미지를 배경으로 더 많이 배치하는 경향을 보였습니다.



### RING#: PR-by-PE Global Localization with Roto-translation Equivariant Gram Learning (https://arxiv.org/abs/2409.00206)
Comments:
          23 pages, 19 figures

- **What's New**: 본 논문에서는 PR-by-PE(localization 방법)라고 불리는 새로운 패러다임을 제안하며, 이는 포즈 추정(pose estimation)에서 직접 장소 인식(place recognition)을 유도하여 글로벌(localization)을 개선합니다.

- **Technical Details**: 우리의 프레임워크인 RING#는 새롭게 개발된 PR-by-PE localization 네트워크로, bird's-eye view (BEV) 공간에서 작동하며, 비전(vision) 및 LiDAR 센서를 지원합니다. 이 네트워크는 두 가지 동등한 표현(equivariant representations)을 BEV 특징으로부터 학습하는 이론적 기반을 도입하여 글로벌 수렴(convergent) 및 계산적으로 효율적인 포즈 추정을 가능하게 합니다.

- **Performance Highlights**: NCLT 및 Oxford 데이터셋을 통한 포괄적인 실험 결과, 우리의 방법이 최신 기술(state-of-the-art approaches)을 초월하는 성능을 보임을 증명하였습니다. 또한, 우리의 방법의 효과를 확인하는 광범위한 분석도 제공합니다.



### A Generative Adversarial Network-based Method for LiDAR-Assisted Radar Image Enhancemen (https://arxiv.org/abs/2409.00196)
- **What's New**: 본 논문에서는 저해상도 레이더 이미지를 향상시키기 위한 GAN(Generative Adversarial Network) 기반 접근 방법을 제안합니다. 이 연구는 자율주행차량(AVs)의 물체 인식을 개선하기 위해 반사하는 세부사항과 특징을 더 잘 표현하는 것을 목표로 합니다.

- **Technical Details**: 제안된 방법은 고해상도 2D 투영 라이다(LiDAR) 포인트 클라우드를 그라운드 트루스 이미지로 사용하고, 레이더 이미지를 입력으로 활용하여 GAN을 훈련합니다. 이 방법의 추론 과정은 오직 레이더 이미지만을 사용하여 향상된 이미지를 생성합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 저해상도 레이더 이미지에 비해 물체 표현이 더 명확한 향상된 이미지를 생성할 수 있음을 보여주었습니다. 이는 특히 악천후에서도 효과적입니다.



### PolypDB: A Curated Multi-Center Dataset for Development of AI Algorithms in Colonoscopy (https://arxiv.org/abs/2409.00045)
- **What's New**: 이 논문에서는 다기관(multi-center)에서 수집된 다양한 모달리티(modality)를 포함하는 대규모 폴립(segmented polyp) 데이터셋인 PolypDB를 공개하며, 이에 대한 기술적 세부사항과 평가 기준을 제시합니다.

- **Technical Details**: PolypDB는 3934장의 폴립 이미지를 포함하고 있으며, Blue Light Imaging (BLI), Flexible Imaging Color Enhancement (FICE), Linked Color Imaging (LCI), Narrow Band Imaging (NBI), White Light Imaging (WLI)의 다섯 가지 모달리티로 구성됩니다. 각 이미지는 경험이 풍부한 위장병 전문의에 의해 접근성과 정확성이 증명된 데이터를 기반으로 하고 있습니다.

- **Performance Highlights**: 이 연구에서는 8개의 세그멘테이션(segmentation) 방법과 6개의 물체 탐지(object detection) 방법에 대한 평가를 실시하여 각 모달리티에 대해 강력한 기준 벤치마크를 설정하였습니다. 이러한 벤치마크는 연구자들이 다른 딥러닝(d deep learning) 알고리즘의 성능을 비교하는 데 유용합니다.



### Methods based on Radon transform for non-affine deformable image registration of noisy images (https://arxiv.org/abs/2409.00037)
- **What's New**: 이 연구는 비선형 변형(non-affine deformations)을 포착하기 위해 Radon 변환을 기반으로 한 두 가지 새로운 변형 영상 등록(Deformable Image Registration, DIR) 방법을 소개합니다. 이 방법들은 비선형 변형을 효과적으로 처리하며 고전적인 선형 탄성 변형 에너지(linear elastic deformation energy)를 기반으로 한 정규화 기법을 적용합니다.

- **Technical Details**: 제안된 DIR 방법들은 두 가지로 첫 번째는 이미지의 Radon 변환을 비교하여 Radon 공간에서 계산되며, 두 번째는 이미지 도메인에서 Radon 변환의 역 투영(backward projection)을 비교하여 계산됩니다. 두 방법의 해의 존재성과 유일성에 대한 조건이 정립되어 있으며, 다양한 비선형 변형을 캡처하기 위해 영상에 대한 실험 결과가 제시됩니다.

- **Performance Highlights**: 이 방법들은 잡음이 있는 이미지 및 큰 변형을 가진 이미지에서의 성능이 평가되었으며, 특히 폐 영상 등록(lung image registration) 상황에서 효과적인 성능을 보여주었습니다.



### Attack Anything: Blind DNNs via Universal Background Adversarial Attack (https://arxiv.org/abs/2409.00029)
- **What's New**: 이 논문은 DNN(Deep Neural Networks)의 공격 방식을 새롭게 제안하며, 특히 디지털 및 물리적 도메인에서 대상 객체에 직접적인 방해 없이 배경에서 이루어지는 적대적 공격에 주목합니다. 이를 통해 다양한 객체, 모델, 작업을 통한 공격 효과의 일반화 가능성을 강조합니다.

- **Technical Details**: 제안된 방법은 적대적 공격을 반복 최적화 문제(Iterative Optimization Problem)로 개념화하며, 이를 DNN 학습 과정에 비유하고, 특정한 온건하지만 충분한 조건을 바탕으로 수렴(Convergence)을 이론적으로 입증합니다. 또한, 적대적 변형(Adversarial Perturbations)을 위한 새로운 앙상블 전략(Ensemble Strategy)을 제안하여 공격의 효과성과 전이 가능성(Transferability)을 강화합니다.

- **Performance Highlights**: 저자들은 디지털 및 물리적 도메인에서 다양한 객체, 모델과 작업을 포괄하는 광범위하고 엄밀한 실험을 수행하여 제안된 배경 적대적 공격 프레임워크의 뛰어난 효과를 입증합니다. 이 연구는 배경 변형(Background Variations)이 DNN의 신뢰성과 강건성(Robustness)에 대한 재평가를 필요로 하며, 이는 DNN의 취약성을 더욱 부각시킵니다.



### Pupil-Adaptive 3D Holography Beyond Coherent Depth-of-Field (https://arxiv.org/abs/2409.00028)
- **What's New**: 최근 딥 러닝을 활용한 홀로그램 디스플레이 기술이 고충실도의 홀로그램 투사를 가능하게 하였으나, 현실적인 초점 신호를 잘 구현하지 못하고 있다. 본 연구에서는 눈 동공 크기의 변화를 고려하여 3D 홀로그램의 질감을 개선할 수 있는 새로운 프레임워크를 제안하였다.

- **Technical Details**: 제안하는 프레임워크는 정합적(deep learning-based) 학습 기법을 통해 관찰자의 동공 상태에 따라 홀로그램 이미지의 심도(depth-of-field)를 동적으로 조절할 수 있도록 설계되었다. 이 방식은 이미징 모델의 차이를 해결하기 위해 3D 포컬 스택(focal stacks) 데이터셋을 생성하고, 신경망을 통해 예측된 홀로그램의 일관성(coherent) 파면(wavefront) 전파를 사용한다.

- **Performance Highlights**: 제안된 방법은 시뮬레이션 및 실험 프로토타입에서 검증되었으며, 기존 접근법들에 비해 최소 5dB의 피크 신호 대 잡음비(peak signal-to-noise ratio)에서 질적으로나 양적으로 개선된 심도 효과를 보여주었다.



### A Novel Fusion of Optical and Radar Satellite Data for Crop Phenology Estimation using Machine Learning and Cloud Computing (https://arxiv.org/abs/2409.00020)
- **What's New**: 이 논문에서는 독일의 8개 주요 작물에 대한 작물 생장 단계(phenology)를 30m 해상도로 예측하는 새로운 프레임워크를 제안합니다. 이 프레임워크는 Landsat과 Sentinel 2의 데이터를 결합하고, Sentinel 1의 레이더 데이터를 머신러닝 모델(Machine Learning)과 융합하여 생장 단계를 추정합니다.

- **Technical Details**: 연구는 2017년부터 2021년까지 독일 기상청(DWD)의 국가 생태학 네트워크 데이터를 기반으로 하여, 원거리 감지(Remote Sensing) 데이터의 최적 조합을 찾기 위한 철저한 특징 융합 분석(feature fusion analysis)을 수행했습니다. 예측된 작물 생장 단계의 결과는 R² > 0.9 및 평균 절대 오차(Mean Absolute Error) < 2일의 매우 높은 정확도를 보였습니다.

- **Performance Highlights**: 광학(optical) 및 레이더(radar) 데이터의 융합 전략이 높은 성과를 보이며 실용적 응용에 매우 중요한 정확도를 가지고 있습니다. 이는 작물 모델의 보정 및 평가를 지원하고, 지속 가능한 식량 생산에 기여하여 증가하는 글로벌 식량 수요에 대응하는 데 유용할 것입니다.



### DivDiff: A Conditional Diffusion Model for Diverse Human Motion Prediction (https://arxiv.org/abs/2409.00014)
- **What's New**: 이 논문에서는 다양한 인간 운동 예측(Diverse Human Motion Prediction, HMP)에서 보다 다양하고 현실적인 예측을 제공하기 위해 새로운 조건부 확산 모델인 DivDiff를 제안합니다. DivDiff는 Denoising Diffusion Probabilistic Model (DDPM)을 기반으로 하여 이전 HMP 방법들이 가지고 있던 한계를 극복하는 데 중점을 둡니다.

- **Technical Details**: DivDiff 모델은 Discrete Cosine Transform (DCT)와 transformer 메커니즘을 사용하여 관찰된 인간 동작 시퀀스를 인코딩하고, 이를 통해 DDPM의 역전 과정에 대한 조건을 설정합니다. 추가로 Diversified Reinforcement Sampling Function (DRSF)을 통해 인간의 골격 제약을 적용하여 예측된 움직임의 품질을 개선합니다. DRSF는 그래프 합성곱 네트워크를 활용하여 인간 뼈 사이 내부 관계를 효과적으로 포착합니다.

- **Performance Highlights**: 실험 결과는 Human3.6M 및 HumanEva-I 데이터셋에서 DivDiff가 기존의 최첨단 방법들에 비해 다양성과 정확성 모두에서 경쟁력 있는 성능을 달성했음을 보여줍니다. 특히, 이 모델은 보다 현실적이고 다양한 3D 인간 동작 예측을 위한 현대적인 방법으로 자리매김할 Potential을 가지고 있습니다.



### Applying Deep Neural Networks to automate visual verification of manual bracket installations in aerospac (https://arxiv.org/abs/2409.00006)
- **What's New**: 본 연구에서는 Siamese Neural Network 아키텍처를 기반으로 한 자동화된 시각 검사(visual inspection) 및 검증(verification) 알고리즘을 탐구했습니다. 특히, 입력 이미지 쌍이 Siamese Neural Network의 성능에 미치는 영향을 고려하였습니다.

- **Technical Details**: 이 연구에서는 Siamese Neural Network와 Convolutional Neural Networks(CNNs)를 함께 조사했습니다. 또한 모델 성능 개선을 위해 transfer learning과 ensemble methods를 포함한 여러 추가 방법을 탐구했습니다. Siamese Neural Network에 특화된 새로운 투표(voting) 방식도 개발하여, 단일 모델이 여러 참조 이미지에 대해 투표하는 방식으로 차별화되었습니다.

- **Performance Highlights**: 결과적으로 Siamese Neural Network가 훈련 데이터가 부족할 때 자동화된 시각 검사 및 검증 작업에 많은 잠재력을 보여주었습니다. 새로운 유사성 투표(similarity voting) 방법을 포함한 추가 방법들이 모델 성능을 상당히 개선하는 것으로 나타났습니다. 본 연구는 공개된 omniglot 데이터셋을 사용하여 접근 방식을 검증했으며, 항공 우주 분야에서 설치된 브래킷의 자동 검증에 관한 상세한 연구가 처음으로 실시되었습니다.



### Evaluating Explainable AI Methods in Deep Learning Models for Early Detection of Cerebral Palsy (https://arxiv.org/abs/2409.00001)
- **What's New**: 이번 연구는 Deep Learning 기법을 활용하여 아기 움직임에서 추출한 골격 데이터를 분석하여 Cerebral Palsy (CP)를 예측하는 Explainable AI (XAI) 방법의 신뢰성과 적용 가능성을 테스트하였습니다. 특히, Class Activation Mapping (CAM)과 Gradient-weighted Class Activation Mapping (Grad-CAM)의 신뢰도를 정량적으로 평가하기 위한 XAI 평가 메트릭스인 신뢰성(faithfulness)과 안정성(stability)을 사용했습니다.

- **Technical Details**: 이 연구는 CP 예측을 위한 Graph Convolutional Network (GCN) 모델에서 CAM 및 Grad-CAM과 같은 XAI 기법을 적용하고, 이들 기법이 CP 예측에 영향을 미치는 중요한 신체 포인트를 효과적으로 구분할 수 있는지를 탐구합니다. 또한 입력 데이터에 경미한 변동이 있을 때 설명의 안정성도 평가합니다.

- **Performance Highlights**: 결과적으로 Grad-CAM이 RISv 메트릭에서 CAM을 뛰어넘는 성능을 보였으며, RISb 메트릭에서는 CAM이 우수한 성능을 보여 다소 상반된 결과를 보였습니다. 전체 앙상블 접근 방식은 개별 모델의 성과를 종합적으로 보여주었으며, XAI 메트릭 적용을 통해 CP 예측의 안정성을 확보했습니다.



### GraspSplats: Efficient Manipulation with 3D Feature Splatting (https://arxiv.org/abs/2409.02084)
Comments:
          Project webpage: this https URL

- **What's New**: 이 논문은 Vision-Language Models (VLMs)의 발전을 바탕으로, 로봇이 효율적으로 제로샷(zero-shot)으로 물체의 부위를 잡을 수 있는 능력을 제안합니다. 특히, 기존의 NeRFs(Neural Radiance Fields) 방식의 한계를 극복하기 위해 새로운 방법인 GraspSplats를 도입하였습니다.

- **Technical Details**: GraspSplats는 RGBD 프레임을 활용하여, 3D Gaussian Splatting(3DGS) 기법을 통해 가시적이고 고품질의 장면 표현을 생성합니다. 이는 깊이 감독(depth supervision)과 새로운 참조 기능 계산(reference feature computation) 방법을 통해 이루어지며, 60초 이내에 처리됩니다.

- **Performance Highlights**: 연구 결과, GraspSplats는 기존의 NeRF 기반 방법들인 F3RM과 LERF-TOGO보다 다양한 작업 환경에서 현저하게 우수한 성능을 보였습니다. 특히, GraspSplats는 정적 및 동적 장면에서 제로샷 그라스핑(grasping) 작업을 효과적으로 수행할 수 있는 도구로 검증되었습니다.



### Explicit Differentiable Slicing and Global Deformation for Cardiac Mesh Reconstruction (https://arxiv.org/abs/2409.02070)
- **What's New**: 본 연구는 DVS(Deterministic Voxelization and Slicing)라는 새로운 알고리즘을 제안하며, 2D 이미지에서 메쉬 재구성을 위한 차별적 전이(supervision)를 가능하게 하여, 복잡하고 변동성이 큰 심장 구조를 보다 정확히 재구성하는 데 기여합니다.

- **Technical Details**: DVS는 그라디언트 역전파를 허용하는 차별적 voxelization 및 slicing 알고리즘으로, 2D 이미지에서 정의된 손실로부터 직접적으로 메쉬 최적화를 지원합니다. GHD(Graph Harmonic Deformation) 알고리즘은 그래프 푸리에 분석에 기반하여 메쉬의 변형을 서피스 푸리에 웨이브로 분해하여 메쉬의 품질 및 매끄러움을 보존합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 CT와 MRI에서 심장 메쉬 재구성 작업에 대해 SOTA(State-of-the-Art) 성능을 달성하였으며, 여러 데이터셋에서 Dice 점수가 90%에 달해 기존 방식보다 우수한 성능을 보여주었습니다.



### FedMinds: Privacy-Preserving Personalized Brain Visual Decoding (https://arxiv.org/abs/2409.02044)
Comments:
          5 pages, Accepted by JCRAI 2024

- **What's New**: 본 논문에서는 다중 개인의 뇌 시각 디코딩에서 개인 정보 보호를 집중적으로 다룬다. FedMinds라는 새로운 프레임워크를 도입하여 연합 학습(federated learning)을 통해 개인의 프라이버시를 보호하면서 모델 학습을 진행한다.

- **Technical Details**: FedMinds 프레임워크는 각 개인에 맞춘 어댑터(adapters)를 배치하여 개인화된 시각 디코딩을 가능하게 한다. 이를 위해 권위 있는 NSD 데이터셋을 활용하여 실험을 진행하며, 연합 학습을 통해 모든 참여자의 뇌 활동 fMRI 데이터의 중앙 집중식 저장을 피할 수 있다.

- **Performance Highlights**: 제안된 프레임워크는 높은 정밀도의 시각 디코딩을 달성하면서도 개인 정보 보호를 유지하는 성능을 보여준다.



### A Modern Take on Visual Relationship Reasoning for Grasp Planning (https://arxiv.org/abs/2409.02035)
- **What's New**: 이번 논문에서는 로봇이 복잡한 공간적 상관관계를 이해하고 최적의 집게(구집기) 순서를 결정하는 데 필요한 복잡한 의사결정 문제를 해결하기 위한 새로운 접근 방식을 제안합니다. D3GD라는 새로운 테스트베드를 소개하고, 물체의 관계를 동시 탐지하고 인접행렬(adjacency matrix)을 생성하는 D3G라는 새로운 종단 간(transformer-based) 의존성 그래프 생성 모델을 제안합니다.

- **Technical Details**: D3GD는 최대 35개의 물체와 97개의 서로 다른 카테고리로 구성된 bin picking 장면을 포함한 데이터셋입니다. D3G 모델은 전체 이미지를 대상(set prediction problem)으로 하여 물체 탐지와 그래프 관계 특성을 함께 학습하는 방식을 채택합니다. 모델 성능 평가는 표준 지표 대신 처음으로 관계의 평균 정밀도(Average Precision of Relationships)를 사용합니다.

- **Performance Highlights**: 실험 결과, D3G 모델은 도전적인 실험 설정에서 여러 최신 기술들(state-of-the-art competitors)을 능가하는 성능을 보여주었으며, 이 성과는 로봇 조작 분야의 미래 연구에 대한 기반을 마련합니다. 출처가 공개된 코드와 데이터셋은 연구 커뮤니티에 기여할 것입니다.



### AttDiCNN: Attentive Dilated Convolutional Neural Network for Automatic Sleep Staging using Visibility Graph and Force-directed Layou (https://arxiv.org/abs/2409.01962)
Comments:
          In review to IEEEtrans NNLS; 15-pages main paper and 3-pages supplementary material

- **What's New**: 본 연구에서는 Attentive Dilated Convolutional Neural Network (AttDiCNN)이라는 자동화된 수면 단계 분류기를 제시합니다. 이 네트워크는 수면 패턴 인식 및 수면 장애 진단에 있어 데이터 이질성(data heterogeneity), 계산 복잡성(computational complexity), 자동 수면 단계 분류의 신뢰성 문제를 해결합니다.

- **Technical Details**: 제안된 네트워크는 세 가지 컴포지터로 구성되어 있습니다: Localized Spatial Feature Extraction Network (LSFE), Spatio-Temporal-Temporal Long Retention Network (S2TLR), 그리고 Global Averaging Attention Network (G2A). LSFE는 수면 데이터에서 공간 정보(spatial information)를 캡처하고, S2TLR은 장기적 맥락(long-term contexts)에서 중요한 정보를 추출하며, G2A는 LSFE와 S2TLR에서 정보를 집계하여 계산 부담(computational overhead)을 줄입니다.

- **Performance Highlights**: 모델은 EDFX, HMC, NCH의 세 가지 공개 데이터셋에서 평가되었으며, 각각 98.56%, 99.66%, 99.08%의 최첨단 정확도를 기록했습니다. 또한, 1.4M의 파라미터(parameter)로 높은 정확도를 유지하면서도 낮은 계산 복잡성을 자랑합니다. 결과적으로 제안된 아키텍처는 기존 방법론들을 여러 성능 지표(performance metrics)에서 초월하며, 임상 환경에서 자동화 도구로서의 잠재력을 입증합니다.



### $S^2$NeRF: Privacy-preserving Training Framework for NeRF (https://arxiv.org/abs/2409.01661)
Comments:
          To appear in the ACM Conference on Computer and Communications Security (CCS'24), October 14-18, 2024, Salt Lake City, UT, USA

- **What's New**: 이번 논문에서는 NeRF(Neural Radiance Fields) 훈련의 개인 정보 보호 문제를 다루기 위해 Split Learning(SL) 기법을 적용한 SplitNeRF 프레임워크를 제안합니다. 이 프레임워크는 개인 데이터를 서버에 전송하지 않고도 클라이언트와 서버 간의 협업 모델 훈련을 가능하게 합니다.

- **Technical Details**: SplitNeRF는 Sl에 기반한 훈련 프레임워크로, 클라이언트와 서버 간의 협력적인 모델 훈련을 수행하면서 로컬 데이터를 공유하지 않는 방식입니다. 이 프레임워크는 Surrogate Model Attack와 Scene-aided Surrogate Model Attack이라는 두 가지 공격 방법이 있음을 발견하였고, 이를 통해 부족한 정보만으로도 개인 장면 정보를 재구성할 수 있는 취약점을 드러냈습니다. 이를 해결하기 위해 S^2NeRF라는 방어 메커니즘을 갖춘 안전한 SplitNeRF 프레임워크를 개발했습니다.

- **Performance Highlights**: S^2NeRF는 공유된 기울기 정보에 관련된 소음(noise)을 감소시키는 방식을 도입하여 개인 정보를 보호하면서도 NeRF 모델의 높은 유용성을 유지합니다. 다양한 데이터셋에서 실험을 통해, S^2NeRF가 개인 정보 침해에 대해 효과적으로 저항하며, 다양한 응용 프로그램에서 안전한 NeRF 훈련을 위한 실행 가능성을 확인했습니다.



### ReKep: Spatio-Temporal Reasoning of Relational Keypoint Constraints for Robotic Manipulation (https://arxiv.org/abs/2409.01652)
- **What's New**: 본 연구에서는 로봇 조작 작업을 관계형 키포인트 제약(Relational Keypoint Constraints, ReKep)으로 표현하는 새로운 방식을 제안합니다. 이 기법은 로봇과 환경 간의 관계를 기반으로 하여, 다양한 작업에 대해 자동화된 제약을 생성할 수 있습니다.

- **Technical Details**: ReKep는 환경 내 3D 키포인트 집합을 수치적 비용으로 매핑하는 Python 함수로 표현됩니다. 이를 통해 조작 작업을 순차적으로 표현하고, 인식-행동 루프를 통해 로봇 행동을 실시간으로 최적화할 수 있습니다. 또한, 대형 비전 모델(Large Vision Models)과 비전-언어 모델(Vision-Language Models)을 활용하여 새로운 작업의 ReKep를 자동으로 생성합니다.

- **Performance Highlights**: 우리는 로봇 플랫폼에서 제안된 ReKep 기반의 시스템 구현을 시연하였으며, 다단계, 실환경, 양손, 반응형 행동을 포함하는 다양한 조작 작업을 수행할 수 있었습니다. 이는 특정 작업 데이터나 환경 모델 없이도 가능하였으며, 작업당 약 10Hz의 속도로 안정적으로 해결할 수 있었습니다.



### T1-contrast Enhanced MRI Generation from Multi-parametric MRI for Glioma Patients with Latent Tumor Conditioning (https://arxiv.org/abs/2409.01622)
Comments:
          arXiv admin note: text overlap with arXiv:2407.02616

- **What's New**: 이번 연구에서는 Gadolinium 기반 조영제 (GBCA)를 사용하지 않고도 고품질의 T1C MRI 이미지를 생성하는 심층 학습 프레임워크를 개발하였습니다. 이 방법은 종양을 인식할 수 있는 시각 변환기 (Vision Transformer) 모델인 TA-ViT를 활용하여 이루어졌습니다.

- **Technical Details**: TA-ViT 모델은 예상 분할 맵에서 변환기 층을 조정하는 적응형 레이어 노름 제로 메커니즘을 통해 종양 영역의 예측을 크게 향상시킵니다. 또한, MPR-ViT 모델을 사용하여 생성된 분할 맵을 잠재 공간으로 변환하여 압축된 특징적 표현을 형성합니다.

- **Performance Highlights**: TA-ViT 모델의 성능은 기존의 MRP-ViT 모델에 비해 질적 및 양적 모두에서 우수성을 나타냈으며, 일반적인 조직 및 종양 영역에서 각각 NMSE, PSNR 및 NCC 지표가 크게 향상되었습니다. 이 방법으로 생성된 T1C 이미지는 실제 T1C 이미지와 매우 유사하며, 향후 GBCA 독성 위험을 없애고 MRI 스캔 프로토콜을 간소화할 수 있는 가능성을 제공합니다.



### Learning Task-Specific Sampling Strategy for Sparse-View CT Reconstruction (https://arxiv.org/abs/2409.01544)
- **What's New**: 이 논문에서는 Sparse-View Computed Tomography (SVCT)의 품질을 향상시키기 위한 과제별 샘플링 전략을 학습하는 딥러닝 프레임워크를 제안합니다. 이는 각 스캔 작업에 대해 최적의 샘플링 전략을 맞춤 설정할 수 있도록 하여 임상 실험에서의 이미지 품질과 성능을 개선하는 데 기여합니다.

- **Technical Details**: 제안된 프레임워크는 세 가지 구성 요소로 나뉩니다. 첫 번째는 각 스캔 작업에 대한 중요도 분포를 예측하는 샘플링 네트워크입니다. 두 번째는 파라미터가 공유된 재구성 네트워크로, 최적 샘플링 전략을 사용하여 고품질 이미지를 재구성합니다. 마지막으로, 임상 관련 네트워크인 다운스트림-태스크 네트워크가 포함되어 있는데, 이는 재구성과 함께 공동 최적화 됩니다.

- **Performance Highlights**: 다양한 스캔 유형에 대한 실험을 통해 제안된 방법이 기존의 방법보다 우수한 성능을 보이며, 한정된 투사 뷰로 이미지 품질과 다운스트림 태스크 성능이 크게 향상됨을 보여줍니다. 이 프레임워크는 임상 내에서의 방사선 노출을 줄이면서 고품질 이미지를 제공할 가능성이 높습니다.



### Improving Robustness of Spectrogram Classifiers with Neural Stochastic Differential Equations (https://arxiv.org/abs/2409.01532)
- **What's New**: 이번 연구에서는 신호 분석 및 분류에서 발생하는 노이즈 문제를 해결하기 위해 Neural Stochastic Differential Equation (Neural SDE) 기반의 새로운 방법론을 제시합니다. 이는 CNN 모델이 노이즈에 대한 강건성을 높이고, 분류 결과의 신뢰도를 높이기 위한 것입니다.

- **Technical Details**: 신호 처리 및 분류를 위한 2D deep learning 기법의 한계를 극복하기 위한 연구이며, 주요 내용으로는 Convolutional Neural Networks (CNNs)를 사용할 때, 훈련 과정에서 도메인 형태의 노이즈를 주입하여 모델의 강건성과 안정성을 높이는 방법론이 포함됩니다.

- **Performance Highlights**: 실험 결과, ConvNeXt-Base 모델은 시계열 분류에서 우수한 성능을 보였으나, 노이즈가 증가할수록 성능 저하가 심했습니다. 반면, NSDE-ConvNext 변형 모델은 경쟁력 있는 분류 성능을 유지하면서 노이즈가 증가해도 성능 저하가 적었습니다.



### Can Geometric Quantum Machine Learning Lead to Advantage in Barcode Classification? (https://arxiv.org/abs/2409.01496)
Comments:
          5 pages, 5 figures

- **What's New**: 기하학적 양자 기계 학습(GQML) 접근법을 개발하여 벡터의 유사성과 비유사성을 기반으로 분류를 수행하며, 적은 샘플로도 일반화를 가능하게 했습니다.

- **Technical Details**: GQML은 대칭(symmetric) 인식 측정 변화(adaptation)를 통해 단위 파라메트라이제이션(unitary parametrizations) 방법보다 높은 성능을 발휘하며, 이를 classical deep neural networks 및 convolutional neural networks with Siamese architectures와 비교했습니다.

- **Performance Highlights**: GQML 네트워크는 전통적인 네트워크에 비해 월등한 성능을 보이며, 이는 데이터셋 구성을 위한 상관 분포(correlated distributions)를 분석하여 설명할 수 있습니다.



### DiffCSG: Differentiable CSG via Rasterization (https://arxiv.org/abs/2409.01421)
- **What's New**: Differentiable rendering 분야에서 새로운 알고리즘인 DiffCSG를 소개하여, Constructive-Solid-Geometry (CSG) 모델을차별 가능한 방식으로 렌더링할 수 있는 방법을 제시합니다.

- **Technical Details**: DiffCSG 알고리즘은 CSG 래스터화(CSG rasterization)를 기반으로 하며, 이는 원시 객체 간의 불리언 연산의 결과를 명시적으로 메시를 계산하지 않고도 표시하는 방식입니다. Differentiable rendering 파이프라인 내에서 CSG 래스터화를 구현하는 방법을 설명하고, 원시 객체의 교차점에서 안티 에일리어싱을 적용하여 중요한 영역에서 기울기를 얻는 데 주의를 기울입니다.

- **Performance Highlights**: DiffCSG 알고리즘은 간단하고 빠르며 현대 기계 학습 세팅에 쉽게 통합될 수 있습니다. 또한, 컴퓨터 보조 설계에서 CSG 원시 객체의 직접 및 이미지 기반 편집을 포함한 다양한 응용 프로그램을 가능하게 합니다.



### Assessing the Impact of Image Dataset Features on Privacy-Preserving Machine Learning (https://arxiv.org/abs/2409.01329)
- **What's New**: 본 연구는 이미지 데이터셋의 특성이 개인 정보 보호와 유틸리티(utility)에 미치는 영향을 파악하고, Differential Privacy (DP)를 활용한 Privacy-Preserving Machine Learning (PPML)의 효용을 분석했습니다.

- **Technical Details**: 연구에서는 여러 데이터셋과 프라이버시 예산을 분석하여, 불균형 데이터셋이 소수 클래스의 취약성을 증가시키지만, DP가 이 문제를 완화함을 발견했습니다. 클래스 수가 적은 데이터셋은 모델의 유틸리티와 프라이버시 두 가지를 모두 개선하는 반면, 높은 엔트로피(high entropy)나 낮은 Fisher Discriminant Ratio (FDR)를 가진 데이터셋은 유틸리티와 프라이버시 간의 균형을 악화시킨다고 합니다.

- **Performance Highlights**: 이 연구의 통찰력은 연구자와 실무자들이 이미지 데이터셋에서 유틸리티-프라이버시(trade-off)를 예측하고 최적화하는 데 유용한 가이드라인을 제공합니다.



### Adversarial Pruning: A Survey and Benchmark of Pruning Methods for Adversarial Robustness (https://arxiv.org/abs/2409.01249)
- **What's New**: 최근의 연구는 적대적인 예제에 대한 강건성을 유지하면서 신경망의 크기를 줄이기 위한 신경망 가지치기 기법을 제안하였습니다. 이는 복잡하고 세련된 디자인으로 불리어지며, 이러한 방법들 간의 차이를 명확히 하고 공정한 비교를 설정하는 것이 매우 어렵습니다.

- **Technical Details**: 본 연구에서는 현재의 적대적인 가지치기(Adversarial Pruning) 방법을 조사하고, 이를 분류하기 위한 새로운 분류 체계를 제안합니다. 이 분류는 크게 두 가지 주요 차원인 가지치기 파이프라인과 구체적인 가지치기 방법에 기반하여 구성됩니다. 우리는 또한 현재의 경험적 분석의 한계를 강조하고 이를 해결하기 위한 새로운 공정한 평가 벤치마크를 제안하였습니다.

- **Performance Highlights**: 우리는 기존의 적대적인 가지치기 방법을 재평가하고 결과를 논의하며, 상위 성능을 보이는 적대적인 가지치기 방법들의 공통된 특성과 일반적인 문제들을 강조합니다. 이 벤치마크는 각 방법의 모델 강건성에 미치는 영향을 직접 비교할 수 있도록 합니다.



### Ground-truth effects in learning-based fiber orientation distribution estimation in neonatal brains (https://arxiv.org/abs/2409.01195)
Comments:
          11 pages, 4 figures; accepted as an Oral Presentation at the MICCAI 2024 Workshop on Computational Diffusion MRI (CDMRI) in Marrakech, Morocco

- **What's New**: 본 연구에서는 신생아의 뇌 이미징에서 더 나은 FOD 추정(Nascent brain imaging for better FOD estimation) 방법으로 SS3T-CSD(single-shell three-tissue constrained spherical deconvolution)를 제안했습니다. 이는 MSMT-CSD(multi-shell multi-tissue constrained spherical deconvolution)를 사용하는 기존 방법과 비교하여 더 효과적일 수 있다는 가정으로 진행되었습니다.

- **Technical Details**: U-Net 아키텍처를 기반으로 하는 딥러닝 모델을 훈련시키는데, MSMT-CSD와 SS3T-CSD를 모두 사용했습니다. SS3T-CSD는 신생아의 뇌에 적합하며, 입력으로 단일 셸 b1000 이미지를 사용하여 FOD를 추정합니다. 이 방법은 여러 방향의 그라디언트(input gradient directions)를 포함할 때 성능이 크게 개선됩니다.

- **Performance Highlights**: SS3T-CSD는 ηλικιακή 변동성에 관계없이 일관된 성능을 유지하며, 신생아의 뇌 이미징에 있어 더 정확한 결과를 제공합니다. 특히, 샘플 수가 적은 신생아 환경에서도 높은 정확성을 보였습니다.



### Logit Scaling for Out-of-Distribution Detection (https://arxiv.org/abs/2409.01175)
- **What's New**: 이 연구는 OOD(Out-of-Distribution) 데이터 감지를 위한 새로운 접근 방식인 Logit Scaling (LTS)을 제안합니다. LTS는 훈련 데이터 배포에 대한 접근이 필요하지 않으며, 다양한 아키텍처에서 강력한 성능을 유지합니다.

- **Technical Details**: LTS 방법은 후처리(post-hoc) 방식으로 작동하여 훈련 데이터 통계에 의존하지 않습니다. LTS는 penultimate layer의 feature representation을 활용하여 샘플 별 scaling factor를 계산하고, 이를 통해 logits를 조정합니다. 최종 OOD 점수는 energy score 함수로 계산됩니다. 이 방법은 3개의 ID(인-디스트리뷰션) 및 14개의 OOD 데이터셋에서 실험을 통해 검증되었습니다.

- **Performance Highlights**: LTS는 9가지의 다양한 아키텍처에 걸쳐 이전의 최첨단 OOD 탐지 방법인 OptFS보다 우수한 성능을 보였으며, FPR@95를 유의미하게 감소시키면서 AUROC를 유지하였습니다. 다양한 환경에서도 잘 작동하는 OOD 탐지 솔루션을 제시합니다.



### Diffusion-Driven Data Replay: A Novel Approach to Combat Forgetting in Federated Class Continual Learning (https://arxiv.org/abs/2409.01128)
Comments:
          Accepted by ECCV 2024 Oral

- **What's New**: 본 논문에서는 Federated Class Continual Learning (FCCL)에서의 데이터 재생 문제를 다루며, 기존의 생성모델 대신 디퓨전 모델(diffusion model)을 활용한 새로운 방법론인 Data Replay를 제안합니다.

- **Technical Details**: Diffusion-Driven Data Replay (DDDR) 프레임워크를 통해 각 클래스의 조건부 임베딩(condition embedding)을 활용하여 데이터 생성을 역설계하는 방식으로, 이전 태스크의 데이터를 재생산하여 모델이 잊어버리는 것을 방지합니다. 특히, 사전 학습된 conditional diffusion model을 활용하여 효과적이고 자원 소모가 적은 데이터 생성을 가능하게 합니다.

- **Performance Highlights**: 다양한 데이터셋에서 실험을 통해 제안한 방법이 기존 FCCL 기법보다 월등한 성능을 보이며, 새로운 최첨단 기준(SOTA)을 세웠음을 입증합니다.



### Defending against Model Inversion Attacks via Random Erasing (https://arxiv.org/abs/2409.01062)
Comments:
          Under review. The first two authors contributed equally

- **What's New**: 이번 논문에서는 Model Inversion (MI) 공격에 대한 새로운 방어 방법인 Random Erasing (RE)을 적용한 MI Defense via Random Erasing (MIDRE)를 제안합니다. 기존의 방어 방법들이 모델의 유틸리티와 개인 정보 보호 간의 균형을 맞추기 위한 복잡한 정규화를 필요로 하는 반면, 우리의 접근법은 데이터에 집중하여 간단하면서도 효과적인 방어를 제공합니다.

- **Technical Details**: MIDRE는 훈련 이미지에서 랜덤으로 선택된 사각 영역을 삭제하여 모델에 제공되는 개인 정보의 양을 줄입니다. 이를 통해 MI 공격이 고차원 개인 이미지를 재구성하는 데 필요한 정보가 감소하며, 공격의 정확도가 현저히 감소하게 됩니다. 실험 결과, 많은 설정에 대해 선택한 공격 방법과 네트워크 아키텍처에서도 SOTA 성능을 발휘합니다.

- **Performance Highlights**: 23개의 실험 세트업에서 MIDRE 방법이 기존의 방어 방식에 비해 MI 공격 정확도를 크게 저하시켰으나, 모델의 일반적인 정확도는 중간 정도만 영향을 받았으며, 경우에 따라 향상되기도 했습니다. 특히 고해상도 환경에서 우리의 방법은 자연적인 정확도를 희생하지 않으면서도 MI 견고성을 첫 번째로 달성하는 성과를 보였습니다.



### Robust Vehicle Localization and Tracking in Rain using Street Maps (https://arxiv.org/abs/2409.01038)
- **What's New**: 본 논문에서는 비가 오는 날이나 터널을 통과하는 상황에서도 차량의 위치 추정을 향상시키기 위해 Map-Fusion이라는 새로운 접근법을 제안합니다. 이 방법은 간헐적인 GPS 측정값과 드리프팅(drifting) IMU(관성 측정 장치), 그리고 시각 정보(Visual Odometry, VO)를 2D 맵 정보와 융합하여 robust한 차량 로컬라이제이션을 제공합니다.

- **Technical Details**: Map-Fusion은 GPS를 초기화에 사용하고 VO 또는 VIO를 사용하여 드리프트(drift)를 보정하는 센서 퓨전(sensor fusion) 기법입니다. 이 방법은 다양한 센서를 통합하기 위해 factor graph 구조를 사용하며, OpenStreetMap(OSM)에서 얻은 도로 네트워크 정보를 활용하여 차량의 위치를 추정하고 보정합니다.

- **Performance Highlights**: Map-Fusion 알고리즘은 다양한 기후 조건, 즉 맑은 날씨와 비 오는 날씨에서 활용된 실험에서 각각 2.46m와 6.05m의 오차를 보였으며, 이는 최신 VO 및 VIO 방식에 비해 모든 데이터 세트에서 드리프트 오류를 줄여주는 효과를 입증했습니다.



### Unleashing the Power of Task-Specific Directions in Parameter Efficient Fine-tuning (https://arxiv.org/abs/2409.01035)
Comments:
          Revisions ongoing. Codes in this https URL

- **What's New**: 이 논문은 대형 언어 모델(Large Language Models, LLMs)의 Parameter Efficient Fine-Tuning(PEFT) 전략과 관련된 새로운 개념인 task-specific directions를 탐구하고, 이를 통해 모델 성능을 향상시키는 LoRA-Dash라는 새로운 접근 방식을 제안합니다.

- **Technical Details**: LoRA-Dash는 task-specific directions을 정의하고 그 속성과 활용 도전 과제를 탐구하는 프레임워크를 제공합니다. 연구는 이러한 방향들이 사전 훈련 상태에서 특정 다운스트림 작업으로 전환하는 데 필수적이라는 점을 강조하며, LoRA 모델의 융통성을 활용하여 저역 랭크 행렬을 통해 파라미터의 효율적인 업데이트를 진행합니다.

- **Performance Highlights**: LoRA-Dash를 활용한 광범위한 실험 결과, 이 접근 방식이 특정 업무에서 모델 성능을 극대화하는 데 효과적임을 입증했으며, LoRA-Dash의 기저 메커니즘에 대한 심도 있는 분석도 수행되었습니다.



### SeCo-INR: Semantically Conditioned Implicit Neural Representations for Improved Medical Image Super-Resolution (https://arxiv.org/abs/2409.01013)
Comments:
          This paper was accepted for presentation at the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2025

- **What's New**: 이 논문에서는 의학 이미지의 국소적인 정보(Localized Prior)를 활용하는 새로운 프레임워크인 Semantically Conditioned INR (SeCo-INR)을 제안합니다. 이는 기존의 Implicit Neural Representations (INR) 방법을 개선하여 의료 이미지를 보다 정확하게 모델링하고 초해상도를 달성하도록 합니다.

- **Technical Details**: SeCo-INR은 의료 이미지를 기초로 한 의미론적(segmentation) 특징을 연속적으로 학습하며, 각 의미론적 영역에 최적의 INR 매개변수를 도출하는 데이터 기반 프로세스를 적용합니다. 이를 위해 다층 퍼셉트론(Multilayer Perceptrons, MLP)을 사용하여 신호와 좌표 간의 복잡한 관계를 근사합니다.

- **Performance Highlights**: 다양한 의료 이미징 데이터셋에서 실험을 진행한 결과, SeCo-INR은 기존의 최첨단(super-state-of-the-art) 방법들에 비해 더 높은 정량적 성능 점수와 현실감 있는 초해상도 출력(realistic super-resolution outputs)을 만들어냈습니다.



### Multi-Modal Multi-Granularity Tokenizer for Chu Bamboo Slip Scripts (https://arxiv.org/abs/2409.01011)
Comments:
          12 pages, 3 figures

- **What's New**: 이번 연구에서는 고대 중국 문자 분석을 위해 설계된 다중 모달리티 및 다중 세분화(tokenizer) 방식의 새로운 방법론을 발표하였습니다. 이 연구는 주로 춘추전국 시대(기원전 771-256년)의 죽간인(Chu bamboo slip, CBS) 문서에 중점을 두고 있습니다. 이전의 연구에서 나타난 한계를 극복하기 위해, 문자 탐지(character detection)와 문자 인식(character recognition)을 접목시킨 새로운 시스템을 구현하였습니다.

- **Technical Details**: 제안된 tokenizer는 이미지 스캔에서 문자 경계를 탐지한 후, 각 문자를 인식하여 정의된 어휘(vocabulary)와 대조합니다. 인식 신뢰도가 낮을 경우, 부문자(sub-character)로 분할하여 추가적인 정보를 추출합니다. 이 연구에서는 100,000개 이상의 주석이 포함된 CBS 문서의 데이터셋 CHUBS를 구축하여 제공하며, 이는 고대 문자 연구에 대한 접근성을 크게 높이는 기초 자료로 활용될 수 있습니다.

- **Performance Highlights**: 제안된 tokenizer는 기존의 서브워드(tokenizer) 방식보다 F1 점수에서 5.5%의 상대적 향상을 이루어냈습니다. 또한, 이 연구는 고대 중국 문자의 특정 분석뿐만 아니라, 기타 고대 문자 연구의 발전 가능성을 제공하는 중요한 기반을 제공합니다.



### Physics-Informed Neural Network Based Digital Image Correlation Method (https://arxiv.org/abs/2409.00956)
- **What's New**: 이번 논문에서는 Physics-Informed Neural Networks (PINNs)를 기반으로 한 새로운 디지털 이미지 상관관계 방법인 PINN-DIC를 소개합니다. 이 방법은 기존의 복잡한 네트워크 아키텍처에 의존하지 않고 간단한 완전 연결 신경망을 사용하여 변위 필드를 추정합니다.

- **Technical Details**: PINN-DIC는 좌표 영역을 입력으로 받아 변위 필드를 출력하는 신경망 구조를 가지고 있습니다. DIC 지배 방정식을 손실 함수에 통합하여 참조 이미지를 기반으로 변위 필드를 추출해냅니다. 이 모델은 반복 최적화를 통해 수행됩니다.

- **Performance Highlights**: PINN-DIC는 비균일 변형 필드에서도 기존의 심층 학습 기반 DIC와 동일한 정확성을 유지하며, 다음과 같은 세 가지 장점을 제공합니다: 1) 좌표에서 직접 변위 필드를 적합하여 정밀성을 향상, 2) 최소한의 매개변수 조정으로 불규칙 경계 변위 필드를 효과적으로 처리, 3) 다른 신경망 기반 기계 분석 방법과 쉽게 통합되어 종합적인 DIC 결과 분석이 가능.



### Semantically Controllable Augmentations for Generalizable Robot Learning (https://arxiv.org/abs/2409.00951)
Comments:
          Accepted for publication by IJRR. First 3 authors contributed equally. Last 3 authors advised equally

- **What's New**: 이번 연구에서는 로봇 조작을 위한 데이터 검증을 위해 미리 훈련된 이미지-텍스트 생성 모델을 활용하여, 직접적인 경험을 뛰어넘는 다양한 데이터를 발전적 증강(Data Augmentation) 방식으로 제공하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 우리는 데이터의 다양성을 확보하기 위해, 대규모 웹 스크랩 데이터를 기반으로 훈련된 이미지-텍스트 생성 모델을 활용합니다. 이 모델은 로봇의 직접 경험 범위를 초과하는 다양한 실세계 시나리오를 포괄하며, 새로운 합성 경험을 생성하여 로봇 에이전트가 추가적인 세계 이전 지식을 경험하도록 합니다. 이러한 생성을 통해, 로봇 데이터셋의 급속한 증강과 풍부한 변화를 유도하여 실세계 일반화를 가능하게 합니다.

- **Performance Highlights**: 우리는 제안된 생성 증강 프레임워크를 통해 로봇 조작 정책이 시뮬레이션 및 주방과 같은 실세계 환경에서 어떻게 교육 및 배포될 수 있는지를 보여주었습니다. 실험 결과, 생성된 데이터셋으로 훈련된 로봇이 미리 보지 못한 새로운 환경에서도 향상된 일반화 능력을 보임을 발견했습니다.



### A Novel Hybrid Parameter-Efficient Fine-Tuning Approach for Hippocampus Segmentation and Alzheimer's Disease Diagnosis (https://arxiv.org/abs/2409.00884)
- **What's New**: 이번 논문에서는 기존의 데이터 수집과 전문가의 레이블링 의존을 줄이기 위해 HyPS라는 새로운 파라미터 효율적인 파인튜닝 전략을 제안합니다. 이 방법은 하이브리드 병렬 및 직렬 구조를 활용하여 최소한의 모델 파라미터만 업데이트합니다.

- **Technical Details**: HyPS는 SwinUNETR 모델에 적용되며, BraTs2021 데이터셋에서 사전 훈련된 후, 세 가지 다른 해마 데이터셋으로 이전됩니다. 이 접근법은 제한된 훈련 샘플 상황에서 특히 우수한 성능을 보여줍니다.

- **Performance Highlights**: HyPS는 Alzheimer's disease (AD)와 건강한 인지(Normal Cognition, CN) 간의 분류에서 83.78%의 정확도를, 초기 경도 인지 장애(EMCI)와 후기 경도 인지 장애(LMCI) 간의 분류에서 64.29%의 정확도를 달성하였습니다.



### Leveraging SeNet and ResNet Synergy within an Encoder-Decoder Architecture for Glioma Detection (https://arxiv.org/abs/2409.00804)
Comments:
          9 pages, 6 figures, 1 table

- **What's New**: 이 연구는 뇌종양(segmentation) 발견 및 분할을 위해 SeNet과 ResNet 구조의 시너지를 활용하여 새로운 모델을 제안합니다.

- **Technical Details**: 제안된 모델은 SeResNet-152를 백본(backbone)으로 사용하고, 강력한 encoder-decoder 구조에 통합되어 특징(feature) 추출을 향상시키고 분할 정확도를 개선합니다.

- **Performance Highlights**: 모델 평가는 Dice Coefficient에서 87%, 정확도(accuracy) 89.12%, IoU 점수에서 88%, 평균 IoU 점수에서 82%를 달성하여 뇌종양 분할의 복잡한 문제를 효과적으로 해결하는 성능을 보여줍니다.



### Multiscale Color Guided Attention Ensemble Classifier for Age-Related Macular Degeneration using Concurrent Fundus and Optical Coherence Tomography Images (https://arxiv.org/abs/2409.00718)
Comments:
          27th International Conference on Pattern Recognition (ICPR) 2024

- **What's New**: 이 논문은 Age-related Macular Degeneration (AMD) 분류를 위해 주의 메커니즘을 통합한 모달리티별 다중 스케일 색 공간 임베딩 (MCGAEc)을 제안합니다. 이 방식은 서로 다른 이미지 모달리티에서 중요한 정보를 효과적으로 추출할 수 있습니다.

- **Technical Details**: MCGAEc 모델은 YCbCr 및 HSV 색 공간을 사용하여 Fundus 이미지에서 중요한 특성을 캡처합니다. 각 색 공간은 Pre-trained VGG16 모델을 통해 읽고, 자가 주의 메커니즘을 사용하여 특징을 집계한 후 랜덤 포레스트 분류기(RFC)로 전달합니다.

- **Performance Highlights**: MCGAEc 방법은 Project Macula에서 제공하는 공개 다중 모달리티 데이터셋을 사용하여 실험되었으며, 단일 모달리티(단일 Fundus 또는 OCT) 방법과의 성능 비교를 통해 효과를 입증하였습니다.



### DeReStainer: H&E to IHC Pathological Image Translation via Decoupled Staining Channels (https://arxiv.org/abs/2409.00649)
- **What's New**: 이 논문은 H&E 이미지에서 IHC 이미지로의 자동 변환을 위한 DeReStainer라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 H&E와 IHC 이미지 간에 공유되는 Hematoxylin 채널의 특성을 활용하여 효과적인 변환을 수행합니다.

- **Technical Details**: DeReStainer는 먼저 H&E와 IHC 이미지를 처리하여 Hematoxylin 채널을 분리하고, 이를 고차원 특징 공간에서 정렬합니다. 이후, DAB 채널을 기반으로 하는 손실 함수를 사용하여 보다 정확한 HER2 수준을 반영하는 IHC 이미지를 생성합니다. 이 방법은 기존의 SSIM과 PSNR의 한계를 극복하고 새로운 의미 정보 평가 지표를 도입합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 BCI 대회에서 이전의 오픈 소스 방법보다 이미지의 본질적 속성과 의미 정보 측면에서 뛰어난 성능을 보였습니다. 특히, HER2 수준의 정확한 표현과 임상적 중요성을 반영할 수 있었습니다.



### Modifying the U-Net's Encoder-Decoder Architecture for Segmentation of Tumors in Breast Ultrasound Images (https://arxiv.org/abs/2409.00647)
- **What's New**: 이 논문에서는 유방 초음파 이미지(segmentation)에서의 정확도와 효율성을 향상시키기 위한 새로운 방법을 제시합니다.

- **Technical Details**: U-Net을 기반으로 한 신경망(Neural Network)을 제안하며, 인코더-디코더 아키텍처를 활용합니다. U-Net과 다른 Deep Neural Networks(Res-Net 및 MultiResUNet)를 결합하고, 새로운 Co-Block 접근 방식을 도입하여 저수준(low-level) 및 고수준(high-level) 특징을 최대한 보존합니다.

- **Performance Highlights**: CResU-Net이라고 명명된 이 네트워크는 780개의 Breast Ultrasound Images(BUSI) 데이터셋을 사용하여 평가되었으며, Dice similarity coefficients(DSC) 76.88%, Intersection over Union(IoU) 71.5%, Area under curve(AUC) 90.3%, global accuracy(ACC) 97.4%를 기록하여 기존의 최첨단 딥러닝 방법보다 유방 병변을 더 정확하게 분할(segmentation)합니다.



### FLUX that Plays Music (https://arxiv.org/abs/2409.00587)
- **What's New**: 이 연구는 텍스트-음악 생성 분야에서 사용되는 새로운 Transformer 기반 패러다임인 FluxMusic을 소개합니다. 이는 rectified flow 방식을 통해 이전의 diffusion 모델보다 성능 향상을 목표로 합니다.

- **Technical Details**: FluxMusic은 mel-spectrogram의 잠재 VAE 공간 내에서 동작하며, 두 개의 독립적인 attention을 텍스트-음악 스트림에 적용합니다. 또한, coarse textual 정보와 time step embeddings를 이용한 변조 기제를 활용하고, fine-grained textual 세부정보는 음악 패치 시퀀스와 결합하여 입력으로 사용됩니다.

- **Performance Highlights**: FluxMusic은 기존의 diffusion 방법론보다 더 우수한 자동 지표 및 인간 선호도 평가에 기반하여 눈에 띄는 성능 향상을 보여주었습니다. 실험 결과는 FluxMusic이 최근의 다른 모델들과 동등한 생성 성능을 달성했음을 입증했습니다.



### FastBO: Fast HPO and NAS with Adaptive Fidelity Identification (https://arxiv.org/abs/2409.00584)
Comments:
          The 18th European Conference on Computer Vision ECCV 2024 Women in Computer Vision Workshop

- **What's New**: 이 논문에서는 Multi-Fidelity Bayesian Optimization을 위한 새로운 방법인 FastBO를 제안합니다. FastBO는 각 하이퍼파라미터 구성에 대해 적절한 Fidelity(신뢰성을) 동적으로 결정하여 성능을 극대화합니다.

- **Technical Details**: FastBO의 핵심 개념으로는 'efficient point'와 'saturation point'를 소개합니다. Efficient point는 자원(level)을 배가했을 때 성능 향상이 작은 임계값 이하로 떨어지는 지점을 의미하며, saturation point는 자원을 추가하면 성능 변화가 미미해지는 지점을 가리킵니다. 이러한 개념들은 자원의 효율적 사용과 성능 예측에 필요한 중요 요소입니다.

- **Performance Highlights**: FastBO는 각 구성의 적합한 Fidelity를 식별하여 surrogate model을 효과적으로 적용할 수 있게 하며, HPO와 NAS의 효율성 및 성능을 향상시키는 데 기여합니다.



### Two-Stage Hierarchical and Explainable Feature Selection Framework for Dimensionality Reduction in Sleep Staging (https://arxiv.org/abs/2409.00565)
- **What's New**: 이번 연구는 EEG 신호의 고차원 데이터를 시각화 가능한 저차원 표현으로 변환하기 위해 두 단계의 계층적이며 설명 가능한 피쳐 선택 프레임워크를 제안합니다. 이를 통해 EEG 신호의 구조적 정보를 보존하고 수면 패턴 탐색 연구에 기여하고자 합니다.

- **Technical Details**: 제안된 프레임워크는 Recursive Feature Elimination with Cross-Validation (RFECV) 알고리즘을 사용하여 초기 입력 피쳐를 선택하고, 이를 차원 축소 기술과 통합합니다. 또한, Topological Data Analysis (TDA)를 활용하여 EEG 신호에서 구조적 피쳐를 추출합니다. 마지막으로, 차원 축소 알고리즘인 PCA, t-SNE, UMAP을 비교하여 최적의 성능을 제공합니다.

- **Performance Highlights**: t-SNE가 79.8%의 정확도를 보여주었으나, 전반적인 성능과 자원 소모를 고려할 때 UMAP이 최적의 선택으로 평가되었습니다. TDA 피쳐는 기존의 스펙트럼-시간 피쳐를 완전히 대체하지는 않지만, 효과적인 보완 역할을 할 수 있음을 입증했습니다.



### Comparative Analysis of Modality Fusion Approaches for Audio-Visual Person Identification and Verification (https://arxiv.org/abs/2409.00562)
Comments:
          This paper has been submitted to a conference

- **What's New**: 이 논문은 음성과 얼굴 정보를 통합하여 사람을 식별하고 인증하는 멀티모달 학습 전략을 비교합니다. 특히 음성에서 x-vector 추출과 얼굴 모드에서 VGGFace2 네트워크를 활용하여 새로운 방법을 제시합니다.

- **Technical Details**: 시스템은 gammatonegram과 Darknet19을 사용하여 음성 표현을 생성하며, K-겹 교차 검증 기법을 통해 VoxCeleb2 데이터셋의 118 명의 화자를 평가에 사용합니다. 세 가지 멀티모달 전략이 비교되며, 센서 융합, 특성 융합, 점수 융합 방식이 구현됩니다.

- **Performance Highlights**: gammatonegram과 얼굴 특징의 융합 전략이 사람 식별 작업에서 98.37%의 정확도로 최고 성능을 달성하였으나, EER(Equal Error Rate) 검증 작업에서는 0.62%에 불과했습니다.



### DAP: Diffusion-based Affordance Prediction for Multi-modality Storag (https://arxiv.org/abs/2409.00499)
Comments:
          Paper Accepted by IROS2024. Arxiv version is 8 pages

- **What's New**: 저자들은 Diffusion-based Affordance Prediction (DAP)라는 새로운 파이프라인을 제안하며, 이는 다중 모드의 객체 저장 문제를 해결하기 위한 혁신적인 접근 방식을 제공합니다. 기존의 방법들과 달리 DAP는 미세한 6D 조작이 필요한 저장 문제의 복잡성을 효과적으로 처리합니다.

- **Technical Details**: DAP는 두 단계로 구성된 접근 방식을 사용하여 초기 단계에서 컨테이너의 적절한 위치를 식별하고, 다음 단계에서 객체와 해당 위치 간의 상대적인 포즈를 정밀하게 계산합니다. Diffusion 모델을 활용하여 격자 기반의 예측을 수행하며, 여러 가능한 목표 구성 중에서 특정 구성을 도출해냅니다.

- **Performance Highlights**: 실험 결과, DAP는 기존의 RPDiff 방법보다 탁월한 성능과 훈련 효율성을 보여주며, RPDiff 벤치마크에서Remarkable한 성과를달성했습니다. DAP는 실제 로봇 시스템에 배치되어도 효과적인 성능을 보여주며, 소음 관측치와 최소한의 훈련 데이터에도 불구하고 성공적으로 저장 작업을 수행 할 수 있음을 입증했습니다.



### Separation of Body and Background in Radiological Images. A Practical Python Cod (https://arxiv.org/abs/2409.00442)
Comments:
          14 pages, 8 figures

- **What's New**: 본 논문에서는 MRI와 CT 이미지에서 신체 부위와 배경 영역을 분리하는 Python 코드를 소개합니다. 이 코드는 2D 및 3D 방사선 이미지에서 분석을 위한 필요한 작업입니다.

- **Technical Details**: 이 논문에서는 두 가지 주요 기술을 사용합니다: 신체 부위와 배경 분리를 위한 알고리즘과 광도 정규화(intensity normalization) 및 이상치 제한(outlier restriction) 방법입니다. 이 과정은 데이터를 8-bit unsigned integer (UINT8) 형식으로 변환하는 데 조정되었습니다.

- **Performance Highlights**: 알고리즘은 다양한 신체 부위의 MRI와 CT 이미지에서 테스트되었습니다. 결과적으로 신체와 배경의 분리가 효과적으로 이루어졌으며, 이를 통해 이미지 분석의 정확도가 향상되었습니다. Python 코드는 적절한 인용과 함께 사용할 수 있습니다.



### LightPure: Realtime Adversarial Image Purification for Mobile Devices Using Diffusion Models (https://arxiv.org/abs/2409.00340)
- **What's New**: 본 논문은 LightPure라 불리는 새로운 이미지 정화(purification) 방법을 소개합니다. 이 방법은 자율 모바일 시스템에서의 저지연(latency)과 높은 정확성(accuracy) 및 강인성(robustness)을 동시에 향상시키는 것을 목표로 합니다.

- **Technical Details**: LightPure는 두 단계 확산(diffusion) 및 원샷(One-shot) GAN(Generative Adversarial Network) 프레임워크를 사용하여 구현됩니다. 고안된 GAN 구조와 새로운 훈련 기법을 통해 정화 과정의 효율성을 극대화하며, 노이즈 추가와 복구 과정을 단일 단계에서 수행합니다.

- **Performance Highlights**: LightPure는 지연(latency) 측면에서 기존 정화 방법에 비해 최대 10배 향상된 성능을 보이며, 다양한 공격 시나리오에서 높은 정확성과 강인성을 유지합니다. 이 방법은 Jetson Nano 보드에서 구현되었으며, 표준 데이터셋인 CIFAR-10 및 GTSRB를 사용하여 성능을 평가하였습니다.



### Pre-Training Multimodal Hallucination Detectors with Corrupted Grounding Data (https://arxiv.org/abs/2409.00238)
- **What's New**: 본 연구에서는 다중 모달 언어 모델(Multimodal Language Models, MLMs)의 환각(hallucination)을 탐지하는 문제를 시퀀스 레이블링(sequence labeling) 작업으로 설정하였습니다. 이는 기존 접근 방식의 한계를 극복하고 환각된 텍스트 구간을 정확히 식별하는 작업을 수행하는 데 중점을 둡니다.

- **Technical Details**: MLM의 성능을 향상시키기 위해, 우리는 오류가 있는 기초 데이터(corrupted grounding data)를 생성하고 이를 프리트레이닝(pre-training) 데이터로 활용합니다. 구체적으로, 구문 기반 데이터(phrase grounding data)를 이용하여 실제로는 시각적으로 부정확하지만 텍스트 컨텍스트에 논리적으로 적합한 환각 문구를 생성합니다. 이 방법은 모델의 샘플 효율(sample efficiency)을 증가시키는 데 기여합니다.

- **Performance Highlights**: 테스트 결과, 프리트레이닝을 통해 최대 +7 F1 점수 개선을 보였으며, 다양한 모델과 데이터 규모에서 샘플 효율이 눈에 띄게 향상되었습니다. 이는 환각 탐지기의 효율성을 크게 향상시키는 것으로 나타났습니다.



### MedDet: Generative Adversarial Distillation for Efficient Cervical Disc Herniation Detection (https://arxiv.org/abs/2409.00204)
- **What's New**: Cervical disc herniation (CDH)의 자동 탐지를 위한 새로운 프레임워크인 MedDet를 제안하였습니다. 이 프레임워크는 multi-teacher single-student knowledge distillation 방식을 통해 모델 압축과 효율성을 향상시키며, generative adversarial training을 통합하여 성능을 개선합니다.

- **Technical Details**: MedDet는 다수의 teacher 모델과 단일 student 모델을 활용하여 고차원 특성을 추출하고, adaptive feature alignment (AFA) 및 learnable weighted feature fusion (LWFF)를 통한 동적 특성 정렬 및 융합도 시행합니다. 또한, nmODE2를 커스터마이즈하여 MRI 노이즈의 영향을 줄이고, 이미지 표현의 정확도를 저하시키지 않도록 설계되었습니다.

- **Performance Highlights**: CDH-1848 데이터셋에서 이전 방법들에 비해 mAP가 최대 5% 향상되었으며, 파라미터는 약 67.8% 감소하고 FLOPs는 36.9% 감소하였습니다. 이 결과는 운동성 질환 발견의 성능과 효율성을 크게 향상시킵니다.



### Robust Temporal-Invariant Learning in Multimodal Disentanglemen (https://arxiv.org/abs/2409.00143)
Comments:
          5 pages, 2 figures, this is the first version. The code is available at this https URL

- **What's New**: 이번 연구에서는 다중 모달 감정 인식에서 연속적인 시계열에서 발생하는 프레임 수준의 중복성을 제거하기 위해 Temporal-invariant learning 방식을 제안합니다. 이를 통해 보다 부드러운 시계열 패턴을 효과적으로 잡아내어 표현의 질을 향상시키고 모델의 강인성을 높였습니다.

- **Technical Details**: 제안된 모델은 RTIL(Robust Temporal-Invariant Learning)로, 서로 다른 모달리티의 표현을 분리하기 위해 adversarial learning을 활용합니다. 이를 통해 modality-invariant representation과 modality-specific representation을 분리하여 적응형 융합 메커니즘을 제공합니다. 또한, 영상 및 음성 모달리티에는 Transformer Encoders를 사용하고, 텍스트 모달리티에는 RoBERTa를 통해 자연어 표현을 향상시킵니다.

- **Performance Highlights**: 실험 결과, RTIL 모델은 두 개의 공개 데이터셋에서 기존 최첨단 방법들을 초월하는 성능을 발휘했습니다. 이는 모델이 시간의 변동성에도 불구하고 일관된 글로벌 정보를 유지할 수 있다는 것을 증명합니다.



### Statistical Analysis of the Impact of Quaternion Components in Convolutional Neural Networks (https://arxiv.org/abs/2409.00140)
Comments:
          17 pages, 6 figures

- **What's New**: 최근 몇 년 간 사원수(Quaternion)를 이용한 합성곱 신경망(Convolutional Neural Networks)인 QCNN이 다양한 문제에 대해 제안되었습니다. 이 논문에서는 이미지 분류 문제에 대해 QCNN 구성요소들 간의 성능 비교를 위한 통계적 분석을 수행하였으며, 새로운 Fully Quaternion ReLU 활성화 함수를 소개하고 있습니다.

- **Technical Details**: QCNN의 구성 요소로는 활성화 함수(activation function), 완전 연결층(fully connected layer), 초기화 알고리즘(initalization algorithm), 모델의 파라미터 수 등이 포함됩니다. 이 연구에서는 n-way ANOVA 테스트를 통해 이러한 요소들이 분류 정확도(classification accuracy)에 미치는 상호작용 효과를 측정하고 최적의 조합을 찾았습니다.

- **Performance Highlights**: 새롭게 제안된 Fully Quaternion ReLU 활성화 함수는 기존의 Split Quaternion ReLU 함수보다 성능을 향상시켰습니다. 실험 결과, QCNN이 기존의 실수 기반 모델과 비슷하거나 더 나은 결과를 보여주었음을 확인하였습니다.



### Zero-Shot Visual Reasoning by Vision-Language Models: Benchmarking and Analysis (https://arxiv.org/abs/2409.00106)
Comments:
          21 pages

- **What's New**: 이번 연구에서는 VLM(vision-language models)의 제로샷(Zero-shot) 시각적 추론 능력을 체계적으로 평가하기 위해 합성 데이터셋(synthetic datasets)을 활용합니다. 이는 기존의 벤치마크가 세계 지식과 혼합되는 점을 명확히 구분하는 시도를 포함합니다.

- **Technical Details**: CLEVR와 PTR 데이터셋을 사용하여 VLM의 순수 시각적 추론 능력을 평가합니다. VLM 내부의 LLM(large language model)에 시각적 임베딩(visual embeddings) 대신 텍스트 기반 장면 설명(textual scene descriptions)을 제공했을 때 성능이 더 좋다는 결과가 나왔습니다. 또한, 체인 오브 사고(chain-of-thought, CoT) 프롬프트가 표준 프롬프트에 비해 더 큰 모델에서만 효과적이라는 것을 발견했습니다.

- **Performance Highlights**: BLIP2-Flan-T5 모델을 활용할 때, 순수 텍스트 정보만으로 18% 더 높은 정확도를 기록하였으며, GPT-4는 GPT-4V보다 CLEVR에서 약 17% 더 높은 정확도를 보였습니다. 그러나 CoT 프롬프트는 더 작은 모델에서는 성능이 떨어지는 것으로 나타나, 모델의 크기가 증가할수록 CoT가 시각적 추론 능력을 개선할 수 있는 잠재력을 지닌다는 것을 시사합니다.



### Towards Human-Level Understanding of Complex Process Engineering Schematics: A Pedagogical, Introspective Multi-Agent Framework for Open-Domain Question Answering (https://arxiv.org/abs/2409.00082)
Comments:
          Our paper is accepted for publication at ML4CCE workshop at ECML PKDD 2024

- **What's New**: 이 논문에서는 프로세스 흐름도(Process Flow Diagrams, PFD)와 배관 및 계기 다이어그램(Piping and Instrumentation Diagrams, P&ID)의 분석을 위한 안전한 기업 솔루션을 제안합니다. 제안된 솔루션은 다계층의 다중 에이전트 Retrieval Augmented Generation (RAG) 프레임워크를 이용하여 개방형 질문 응답(open-domain question answering) 작업을 수행하며 데이터 프라이버시 및 비용 효율성을 증대시킵니다.

- **Technical Details**: 제안된 다중 에이전트 프레임워크는 전문화된 하위 에이전트와 introspective(내성적) 메타 에이전트로 구성됩니다. 각 하위 에이전트는 PFD와 P&ID 분석을 위해 ReAct(Reason + Act) 촉진 기법을 사용하여 정보를 통합합니다. 또한, 사용자 쿼리에 대해 태스크 플래닝을 통해 명령 및 월등한 응답 생성을 위한 외부 API와의 상호작용을 조절합니다.

- **Performance Highlights**: 실험 결과, 제안한 접근법이 이미지 캡셔닝, 시각 질문 응답(VQA), 텍스트 검출(OCR) 작업 등에서 기존의 최신 기술과 동등한 성능을 보여주었으며, 맞춤화, 해석 가능성, 데이터 프라이버시 및 비용 효율성을 제공합니다.



### Extending Machine Learning Based RF Coverage Predictions to 3D (https://arxiv.org/abs/2409.00050)
Comments:
          2022 IEEE International Symposium on Antennas and Propagation and USNC-URSI Radio Science Meeting (AP-S/URSI)

- **What's New**: 이 논문은 mmWave 통신 환경에서 신호 전력(signal power) 예측의 최신 발전에 대해 다룹니다. 특히, 머신러닝(ML)을 활용하여 좋은 정확도와 실시간 시뮬레이션 속도로 전력 추정치를 제공하는 모델을 훈련할 수 있는 가능성을 제시합니다.

- **Technical Details**: 전처리(pre-processing) 개선이 포함된 훈련 데이터과 임의의 송신기 높이(arbitrary transmitter height)를 고려한 3D 예측 방법에 대해 논의합니다. 이는 효율적인 머신러닝 기반 모델 훈련에 기여합니다.

- **Performance Highlights**: 이 연구는 mmWave 통신에서의 전력 예측 정확도를 크게 향상시키며, 실시간으로 신속하게 실행될 수 있는 방법을 제공합니다.



### No Need to Sacrifice Data Quality for Quantity: Crowd-Informed Machine Annotation for Cost-Effective Understanding of Visual Data (https://arxiv.org/abs/2409.00048)
- **What's New**: 이번 논문에서는 시각적 데이터(Labeling visual data)의 품질을 보장하면서도 대규모로 자동화할 수 있는 새로운 프레임워크를 제시합니다. 특히, 고신뢰성의 결과를 유지하는 동시에 비용 절감이 가능하다는 점에서 주목할 만합니다.

- **Technical Details**: 제안된 방법은 작업별 소프트 라벨(soft labels)에 대한 후방 분포(posterior distributions)를 활용하며, 딥러닝 기반의 컨볼루션 신경망(convolutional neural network)을 사용하여 군중 응답(crowd responses)을 예측합니다. 또한, Dirichlet prior를 이용하여 분석적으로 접근 가능하도록 설계되었습니다.

- **Performance Highlights**: 본 연구는 두 가지 실제 자동차 데이터셋에서 실험하였으며, 제안된 모델이 상당 수의 작업을 완전 자동화할 수 있음을 보여주었습니다. 이는 고비용 절감 효과를 창출하며, 인간 불확실성(human uncertainty)을 신뢰성 있게 예측하는 능력을 통해 더 정확한 검사 및 필터링이 가능해졌습니다.



### Glyph-Based Uncertainty Visualization and Analysis of Time-Varying Vector Fields (https://arxiv.org/abs/2409.00042)
- **What's New**: 이 논문은 3D 벡터 필드의 불확실성을 정확하게 나타내기 위해 기초한 새로운 3D glyph 디자인인 'squid glyph'를 소개합니다. 기존 2D glyph의 한계를 넘어서 기후 변화나 자연 재해와 같은 분야에서 유용한 통찰력을 제공할 수 있습니다.

- **Technical Details**: 'Squid glyph'는 벡터의 크기와 방향 불확실성을 정확히 인코딩하며, 각 glyph 부분이 독립적으로 구분 가능하도록 설계되었습니다. 이 glyph는 superellipse 형태를 사용해 방향 변동성을 더 잘 나타내고 회전 모호성을 제거했습니다. 또한, 주요 성분 분석(Principal Component Analysis, PCA)을 통해 메디안 방향을 추적하고 있습니다.

- **Performance Highlights**: 새로운 squid glyph는 기존의 cone, comet, tailed-disc glyph들과 비교했을 때 불확실성을 더 잘 그려내어, 크기와 방향의 변화를 효과적으로 구분하는 데 있어 개선된 성능을 보여줍니다. 실제로 허리케인과 산불 사례를 통해 해당 glyph의 효능을 입증하였습니다.



### A Novel Approach to Classify Power Quality Signals Using Vision Transformers (https://arxiv.org/abs/2409.00025)
Comments:
          IECON 2024-50th Annual Conference of the IEEE Industrial Electronics Society, Chicago, U.S.A, 2024, pp. 1-6

- **What's New**: 이 논문에서는 스마트 그리드(smart grids)에서 전력 품질 장애(power quality disturbances, PQD) 분류를 위한 새로운 접근 방식을 제안하고 있습니다. 이는 Vision Transformer(ViT) 모델을 기반으로 하며, PQD 신호를 이미지로 변환한 후 미리 훈련된 ViT를 사용해 정확하게 분류합니다.

- **Technical Details**: 제안된 방법은 PQD가 발생할 때 전력 품질 신호를 이미지로 변환하고, 대규모 데이터셋에서 17개의 장애 클래스로 학습 및 테스트하여 분류를 수행합니다. 이는 대부분의 이전 연구와 달리 제한된 클래스나 작은 데이터셋이 아닌 대규모 데이터셋을 활용합니다.

- **Performance Highlights**: 데이터셋에서 제안된 ViT 기반 접근법은 PQD 분류에서 각각 98.28%의 정확도(precision)와 97.98%의 재현율(recall)을 달성하여 최근에 제안된 기법들을 능가하는 성능을 보였습니다.



### Detecting Misinformation in Multimedia Content through Cross-Modal Entity Consistency: A Dual Learning Approach (https://arxiv.org/abs/2409.00022)
Comments:
          Accepted to PACIS 2024. 15 pages, 3 figures

- **What's New**: 이번 연구에서는 Multimodal (다중모달) 형식의 허위 정보 탐지에 대한 새로운 접근 방식을 제시합니다. 특히, 기존 연구들은 주로 단일 모달리티나 텍스트-이미지 조합에 중점을 두었지만, 비디오 콘텐츠를 포함한 다중모달 형식에서의 허위 정보 탐지의 필요성을 강조합니다.

- **Technical Details**: 우리는 Multimedia Misinformation Detection (MultiMD) 프레임워크를 제안합니다. 이 프레임워크는 cross-modal (크로스 모달) entity consistency를 활용하여 다중모달 허위 정보를 탐지합니다. Dual learning (듀얼 러닝) 접근 방식을 사용함으로써, 탐지 성능을 향상시키고 다양한 모달리티에서의 entity consistency 표현 학습을 개선할 수 있습니다.

- **Performance Highlights**: 실험 결과, MultiMD는 최신 기술의 기준 모델들보다 우수한 성능을 보였으며, 각 모달리티가 허위 정보 탐지에서의 중요성을 강조합니다. 이 연구는 다중모달 허위 정보 탐지에 대한 새로운 방법론적 및 기술적 통찰을 제공합니다.



New uploads on arXiv(cs.AI)

### Configurable Foundation Models: Building LLMs from a Modular Perspectiv (https://arxiv.org/abs/2409.02877)
- **What's New**: 최근 LLM(large language model)의 발전은 대규모 매개변수로 인한 컴퓨팅 효율성과 확장성의 새로운 도전과제를 드러냈습니다. 이를 해결하기 위해, 연구진은 LLM을 기능 모듈로 분해하여 이를 조합하는 방식으로 복잡한 과제를 처리하는 모듈식 접근 방식을 제안하고 있습니다. 이 논문에서는 이러한 조합 방식의 효율성과 구성 가능성을 강조하여, 각 기능 모듈을 'brick'으로 정의하고 구성 가능 기반 모델(configurable foundation models)이라는 구조를 제시합니다.

- **Technical Details**: 모듈은 신경망이 훈련되는 과정에서 나타나는 'emergent bricks'와 후속 훈련을 통해 특별히 구축된 'customized bricks'로 구분됩니다. 다양한 기능 브릭에 기반하여 네 가지 브릭 지향 연산: retrieval & routing, merging, updating, growing을 제안하였으며, 이들 연산을 통해 LLM의 동적 구성이 가능해지며 복잡한 작업을 처리할 수 있습니다.

- **Performance Highlights**: 실증적 분석을 통해 FNN 계층이 모듈형 특성을 보이며, 신경망의 기능적 전문화와 신경 분할을 보여주었습니다. 이러한 연구는 기존 LLM 연구에 대해 새로운 모듈식 관점을 제공하며, 더 효율적이고 확장 가능한 기반 모델의 발전에 기여할 것입니다.



### Bioinformatics Retrieval Augmentation Data (BRAD) Digital Assistan (https://arxiv.org/abs/2409.02864)
- **What's New**: 이번 논문에서는 Bioinformatics Retrieval Augmentation Data (BRAD) 디지털 어시스턴트의 프로토타입을 소개합니다. BRAD는 다양한 생물정보학(Bioinformatics) 작업을 처리할 수 있는 도구 모음을 통합하였습니다.

- **Technical Details**: BRAD의 주요 기능으로는 (1) 검색 강화 생성(retrieval augmented generation, RAG)을 통한 향상된 질문-응답 기능, (2) 복잡한 소프트웨어 파이프라인(pipeline)을 작성하고 실행할 수 있는 능력, (3) 개인 및 팀 에이전트 간의 작업을 조직 및 배포할 수 있는 기능이 있습니다.

- **Performance Highlights**: BRAD는 생물정보학 워크플로우의 자동화를 위해 사용되며, 유전자 풍부화(gene enrichment), 아카이브 검색, 자동 코드 생성 및 바이오마커 식별 파이프라인 실행 등의 작업을 수행합니다. BRAD는 디지털 생물학 실험을 위한 가설 생성 및 테스트를 이끌어내는 자체 포함 루프를 기반으로 한 연구실의 디지털 쌍둥이를 개발하는 궁극적인 목표에 한 걸음 다가가는 도구입니다.



### An incremental preference elicitation-based approach to learning potentially non-monotonic preferences in multi-criteria sorting (https://arxiv.org/abs/2409.02760)
Comments:
          37 pages, 22 figures

- **What's New**: 이 논문에서는 다기준 정렬(MCS) 문제에서 잠재적인 비단조(preference) 선호를 학습하기 위한 새로운 점진적인 선호 추출(incremental preference elicitation) 기반 접근 방식을 제안합니다. 의사결정자는 점진적으로 할당 예시 우선순위(preference information)를 제공할 수 있습니다.

- **Technical Details**: 본 논문은 max-margin optimization 기반 모델을 구축하여 비단조 선호 및 비일관적인 할당 예시 우선순위 정보를 처리합니다. 최적 객체 함수 값(optimal objective function value)을 활용하여 정보량 측정 방법(information amount measurement methods) 및 질문 선택 전략(question selection strategies)을 개발합니다.

- **Performance Highlights**: 제안한 방법을 신용 등급 문제(credit rating problem)에 적용하고, 인공 및 실제 데이터 세트를 통한 계산 실험(computational experiments)을 통해 제안된 질문 선택 전략을 여러 벤치마크 전략과 비교하여 효과성을 입증하였습니다.



### Creating a Gen-AI based Track and Trace Assistant MVP (SuperTracy) for PostNL (https://arxiv.org/abs/2409.02711)
- **What's New**: 이번 연구는 PostNL이 고객 서비스와 내부 커뮤니케이션을 개선하기 위해 Generative AI를 활용하는 방법을 탐구했습니다. 특히, 패키지 추적 및 소통을 향상시키기 위해 Multi-Agent LLM 기반 시스템인 SuperTracy의 Minimal Viable Product (MVP)를 개발했습니다.

- **Technical Details**: SuperTracy는 패키지 여정에 대한 이야기를 생성하고, 물류 중단을 식별하는 멀티 에이전트 시스템입니다. 이 시스템은 Retrieval-Augmented Generation (RAG) 기법을 사용해 응답의 정확성을 높이고, 특정 도메인 작업에 최적화된 Large Language Models (LLMs)를 활용합니다.

- **Performance Highlights**: MVP의 성공적인 구현은 PostNL의 패키지 통신을 개선하는 데 있어 기술적 혁신과 실행 가능성을 입증했습니다. SuperTracy는 사용자 문의를 자율적으로 관리하고 내부 지식 처리를 향상시키며, 초기 기대치를 초과하는 성과를 보였습니다.



### Decision Transformer for Enhancing Neural Local Search on the Job Shop Scheduling Problem (https://arxiv.org/abs/2409.02697)
Comments:
          currently under review for IEEE Transactions on Cybernetics

- **What's New**: 최근 연구에서는 머신 러닝(ML)이 작업장 스케줄링 문제(Job Shop Scheduling Problem, JSSP) 해결 방안의 발전에 중요한 역할을 하고 있습니다. 본 논문에서는 Neural Local Search(NLS)라는 최첨단 깊은 강화 학습(deep reinforcement learning, DRL) 에이전트를 기반으로 JSSP의 국소 탐색(local search)을 효과적으로 제어하는 방법을 제시합니다.

- **Technical Details**: 우리는 훈련된 NLS 에이전트가 수행한 탐색 궤적(search trajectories)을 기반으로 결정 변환기(decision transformer, DT) 알고리즘을 훈련시키는 방법을 개발했습니다. 이 프로세스는 배운 의사결정 시퀀스를 향상시키는 데 초점을 맞추고 있습니다. DT는 NLS 에이전트와는 다른 국소 탐색 전략을 학습하며, 종종 더 효과적이기도 합니다.

- **Performance Highlights**: DT는 솔루션 품질(solution quality)과 허용 가능한 계산 시간(compuatational time) 간의 균형에서 특히 우수합니다. 긴 계산 시간이 허용되는 응용 시나리오에서 DT는 더 큰 신경망 아키텍처로 인해 요구되는 단계별 긴 추론 시간(inference times)을 상쇄하고, 더 나은 품질의 의사결정을 제공합니다. 그 결과, ML로 강화된 탐색을 통해 JSSP 해결에 대한 최첨단 결과를 달성했습니다.



### Evaluating Environments Using Exploratory Agents (https://arxiv.org/abs/2409.02632)
Comments:
          9 Pages, 9 figures, 2 tables, work in progress

- **What's New**: 이 논문은 절차적으로 생성된 게임 레벨 디자인에 대한 피드백을 제공할 수 있는 탐색적 에이전트(exploratory agent)의 사용에 대해 논의합니다. 5개의 매력적인(levels)과 5개의 비매력적인(unengaging levels) 레벨을 비교하며, 탐색의 동기를 모델링하는 프레임워크를 확장하고, 탐색 가능성을 평가하기 위한 피트니스 함수(fitness function)를 도입합니다.

- **Technical Details**: 연구는 Wave Function Collapse (WFC) 알고리즘을 사용하여 레벨을 생성하는 2개의 절차적 콘텐츠 생성기(generator A, B)의 탐색 지원 능력을 평가합니다. 에이전트의 탐색 행동은 환경 커버리지(environment coverage), 독특한 개체(inspecting unique objects)의 탐색, 커스텀 노벨티(novelty) 측정, 에이전트 경로의 엔트로피(entropy) 및 에이전트가 경험한 평균 동기 부여(average motivation) 측정 지표로 분석되었습니다. 이러한 지표는 환경의 독창성과 친숙성을 균형 잡아 탐색 품질을 정량화하기 위해 설정되었습니다.

- **Performance Highlights**: 탐색적 에이전트는 매력적인 레벨과 비매력적인 레벨을 명확하게 구분할 수 있으며, 이는 절차적으로 생성된 레벨의 탐색 가능성을 평가하는 효과적인 도구로 자리 잡을 가능성을 제시합니다. 이 연구는 AI 주도 게임 디자인 분야의 발전에 기여하며, 게임 환경을 평가 및 최적화하는 새로운 통찰을 제공합니다.



### Vision-Language Navigation with Continual Learning (https://arxiv.org/abs/2409.02561)
- **What's New**: 이번 연구에서는 Vision-Language Navigation (VLN) 분야에서 에이전트가 자연어 지시를 기반으로 3D 환경을 탐색하는 과정에서 발생하는 성능 격차 문제를 해결하기 위해 Continual Learning (CL) 프레임워크를 적용한 Vision-Language Navigation with Continual Learning (VLNCL) 패러다임을 제안했습니다. 또한 뇌의 메모리 리플레이 메커니즘에서 영감을 받은 이중 루프 시나리오 리플레이 방법(Dual-SR)을 도입하여, 에이전트가 새 환경에 적응하면서도 기존의 지식을 유지할 수 있도록 했습니다.

- **Technical Details**: VLNCL 패러다임은 에이전트가 새로운 환경을 점진적으로 학습하며 과거에 습득한 지식을 유지하는 방식을 사용합니다. 이를 위해 과거의 시나리오 메모리를 통합하고 새로운 작업 학습의 균형을 맞추는 이중 루프 메모리 리플레이 프레임워크를 설계하였습니다. 에이전트는 다중 시나리오 메모리 버퍼를 활용해 다양한 환경에서의 작업 메모리를 효율적으로 저장하고 재생합니다. 성능 평가를 위해 Unseen Transfer (UT)와 Seen Transfer (ST)라는 두 가지 새로운 메트릭을 제안했습니다.

- **Performance Highlights**: VLNCL 패러다임을 적용한 에이전트는 기존의 VLN과 CL 방법들에 비해 16% 향상된 성공률을 보였으며, 지속적 학습 능력에서 최첨단 성능을 달성했습니다. 이 연구는 VLN 분야에서의 지속적 학습을 선도하며, 새로운 환경에 대한 적응 능력을 확보함과 동시에 기존 지식을 유지할 수 있는 가능성을 보여줍니다.



### A Sequential Decision-Making Model for Perimeter Identification (https://arxiv.org/abs/2409.02549)
- **What's New**: 이 연구에서는 교통 흐름 모니터링 및 최적화를 위한 새로운 경계 식별 프레임워크를 제안합니다. 이 프레임워크는 공개 정보만을 사용하여 실시간으로 효율적으로 운영될 수 있도록 설계되어 있습니다.

- **Technical Details**: 제안된 모델은 Markov Decision Process (MDP)를 기반으로 하며, 에이전트는 실시간으로 데이터를 수집하여 최적의 경계를 식별하는 게임을 합니다. 모델은 교통 혼잡도 열 지도 이미지를 활용하여 경계 식별 문제를 순차적 결정 모델로 형성하며, 에이전트는 새로운 교차점을 추가하거나 기존 교차점을 제거하는 행동을 수행합니다.

- **Performance Highlights**: 모델은 실제 사례를 통해 효과성을 입증하며, 적절한 최적 경계를 식별하는 과정에서의 유용성을 강조합니다. 이 접근 방식은 동적인 경계 변화를 추적할 수 있도록 해주며, 다양한 도시 환경에 적용 가능성이 높습니다.



### Cog-GA: A Large Language Models-based Generative Agent for Vision-Language Navigation in Continuous Environments (https://arxiv.org/abs/2409.02522)
- **What's New**: 이번 연구에서는 자연어 지침에 기반하여 무한 3D 공간에서 자유롭게 탐색할 수 있는 에이전트인 Cog-GA(인식 생성 에이전트)를 제안합니다. Cog-GA는 대규모 언어 모델(LLMs)을 활용하여 인지 맵을 구축하고 경로 포인트를 예측하는 두 가지 전략을 사용해 인간과 유사한 인지 프로세스를 모방합니다.

- **Technical Details**: Cog-GA는 시간적, 공간적, 의미적 요소를 통합한 인지 맵을 통해 LLM의 공간 기억을 개발하며, 경로 포인트에 대한 예측 메커니즘을 사용하여 탐색 경로를 최적화합니다. 이 시스템은 '무엇(what)'과 '어디(where)'라는 이중 채널 장면 설명을 통해 환경 정보를 구분하고, 이전 탐색 경험의 피드백을 캡처하여 연속적인 학습을 촉진합니다.

- **Performance Highlights**: VLN-CE 데이터세트에서 Cog-GA는 48%의 성공률을 기록하며 최첨단 성능을 달성했습니다. 이 연구는 전략적이고 해석 가능한 VLN-CE 에이전트 개발에 중요한 기여를 하며, 로봇 내비게이션의 실제 환경에서 인지적 프로세스를 시뮬레이션하는 데 큰 의의를 지닙니다.



### Large Language Models and Cognitive Science: A Comprehensive Review of Similarities, Differences, and Challenges (https://arxiv.org/abs/2409.02387)
Comments:
          10 pages, 1 figure

- **What's New**: 이번 리뷰 논문은 대형 언어 모델(LLMs)과 인지 과학의 교차점을 탐구하며, LLM과 인간의 인지 과정을 비교합니다. LLM의 인지 능력을 평가하는 방법과 인지 모델로서의 잠재력을 분석합니다.

- **Technical Details**: LLM들은 언어 처리, 추론 및 문제 해결에서 인간과 유사한 능력을 보여주지만, 새로운 문제에 대한 추론에서 한계를 보입니다. 이는 LLM의 일반화 능력에 제약을 나타냅니다. CogBench와 MulCogBench와 같은 도구들이 LLM의 인지 능력을 평가하기 위해 개발되었습니다.

- **Performance Highlights**: LLM은 특히 언어 처리 및 감각 판단 작업에서 인간과 유사한 성능을 보여주지만, 새로운 상황에서의 추론이나 기능적 언어 능력에서는 부족함이 있습니다. 향후 연구는 이러한 모델을 개선하고 인간 인지와의 일치를 높이는 데 중점을 두어야 합니다.



### Initial Development and Evaluation of the Creative Artificial Intelligence through Recurring Developments and Determinations (CAIRDD) System (https://arxiv.org/abs/2409.02291)
- **What's New**: 이 논문은 인공지능 일반화(AGI)로 나아가는 과정에서 컴퓨터 시스템의 창의성을 향상시키기 위한 새로운 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 개념 주입(concept injection) 및 정제(refinement)의 반복(iterative) 프로세스를 통해 LLMs의 출력을 창의적으로 개선하는 기술입니다. 연구에서는 CAIRDD(Creative Artificial Intelligence through Recurring Developments and Determinations) 시스템의 초기 개발 작업을 소개하고 주요 시스템 구성 요소의 효능을 평가합니다.

- **Performance Highlights**: 이 기술은 LLM이 생성하는 콘텐츠의 창의성을 개선할 가능성을 제시하고, 사람의 창의성과 차별화된 점을 해결하는 데 기여할 수 있습니다.



### RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins (early version) (https://arxiv.org/abs/2409.02920)
Comments:
          Project page: this https URL

- **What's New**: 이번 논문은 RoboTwin이라는 새로운 벤치마크 데이터세트를 소개합니다. 이는 실제 원격 조작 데이터와 디지털 트윈에서 생성된 합성 데이터의 조합으로, 두 팔 로봇 사용 시나리오를 위해 설계되었습니다.

- **Technical Details**: RoboTwin 데이터세트는 COBOT Magic 플랫폼을 사용하여 수집된 다양한 도구 사용 및 인간-로봇 상호작용 데이터를 포함합니다. AI 생성 콘텐츠(AIGC)을 활용하여 2D 이미지를 3D 모델로 변환하는 새로운 접근 방식이 도입되었습니다. 또한, 대규모 언어 모델(LLMs)을 활용하여 전문가 수준의 훈련 데이터와 작업별 포즈 시퀀스를 생성합니다.

- **Performance Highlights**: 이 연구는 RoboTwin 벤치마크 데이터세트, 효율적인 실제-시뮬레이션 파이프라인, 자동 전문가 데이터 생성을 위한 언어 모델 활용의 세 가지 주요 기여를 통해 로봇 훈련 데이터의 부족 문제를 해결하고, 로봇이 더 높은 기능성을 갖추도록 돕는 데 기여합니다.



### UC-NeRF: Uncertainty-aware Conditional Neural Radiance Fields from Endoscopic Sparse Views (https://arxiv.org/abs/2409.02917)
- **What's New**: 본 논문은 외과 수술 장면에서의 새로운 시점 합성을 위해 불확실성 인식 조건부 NeRF(Neural Radiance Field)인 UC-NeRF를 제안합니다. 이는 희소한 외과적 관점에 대한 기하학-광선(geometry-radiance) 모호성을 해결하기 위함입니다.

- **Technical Details**: UC-NeRF의 핵심은 다중 시점 불확실성 추정을 포함하여 외과적 장면의 심각한 광학적 불일치를 조정하는 것입니다. 이를 위해 다중 시점 스테레오 네트워크를 이용해 기하학적 대응관계를 수립하고, 불확실성 추정 및 특징 프라이어를 생성합니다. 또한, 기본 적응형 NeRF 네트워크를 설계하여 불확실성 추정을 활용해 광학적 불일치를 처리하고, 단안 기하학 프라이어의 증류(distillation)를 통해 기하학 학습을 향상시킵니다.

- **Performance Highlights**: SCARED 및 Hamlyn 데이터세트에서 실험을 통해 UC-NeRF가 기존의 최신 기법들보다 우수한 성능을 보이며, 외과적 장면에서의 새로운 시점 합성에서 효과성과 효율성을 일관되게 개선함을 입증했습니다.



### Masked Diffusion Models are Secretly Time-Agnostic Masked Models and Exploit Inaccurate Categorical Sampling (https://arxiv.org/abs/2409.02908)
Comments:
          40 pages

- **What's New**: 최근 연구에 따르면 Masked Diffusion Models (MDMs)가 시간 변수와 무관하게 작동하며, 이는 그들이 본질적으로 masked models과 수학적으로 동등하다는 것을 보여줍니다. 새로운 first-hitting sampler (FHS) 기법을 통해 MDMs의 샘플링 속도를 20배 향상시킴으로써 이전의 효율성을 크게 개선했습니다.

- **Technical Details**: MDMs는 masked tokens의 개수를 사용하여 continuous-time evidence lower bound (ELBO) 목적함수를 정의하며, 이는 order-agnostic auto-regressive models과 일치합니다. FHS는 시간이 필요 없는 샘플링을 처럼 동작하여, 기존의 categorical sampling을 피하고 어떤 mask token이 처음으로 언마스크될 때를 analytically 샘플링합니다.

- **Performance Highlights**: MDMs는 낮은 generative perplexity를 보여주지만, 토큰의 다양성이 감소함에 따라 생성 품질이 다소 저하되는 문제가 발생합니다. 32-bit floating-point precision을 사용할 때, MDMs의 generative perplexity가 126.11에서 31.24로 크게 개선되지만, sentence entropy는 5.66에서 5.17로 낮아졌습니다.



### LongLLaVA: Scaling Multi-modal LLMs to 1000 Images Efficiently via Hybrid Architectur (https://arxiv.org/abs/2409.02889)
Comments:
          19 pages, 7 figures, 6 tables

- **What's New**: 이번 논문에서는 멀티모달 대형 언어 모델(Multi-modal Large Language Models, MLLMs)의 긴 맥락 처리 능력을 확장하는 새로운 슈퍼 모델인 LongLLaVA를 소개합니다. 이는 Mamba와 Transformer 블록의 하이브리드 아키텍처를 채택하고, 여러 이미지 간의 시간적 및 공간적 종속성을 고려한 데이터 구축 및 점진적 훈련 전략을 적용한 것입니다.

- **Technical Details**: LongLLaVA는 멀티모달 아키텍처, 데이터 구성 및 훈련 전략의 세 가지 차원에서 종합적으로 최적화된 모델입니다. Mamba-Transformer 하이브리드 아키텍처는 효율적인 이미지 표현 방식을 적용하고, 고유한 데이터 형식을 설계하여 다중 이미지 처리 시 성능 저하를 방지합니다. 훈련 전략은 단일 이미지 정렬, 단일 이미지 지시 튜닝 및 다중 이미지 지시 튜닝의 세 단계로 되어 있어 점진적으로 모델의 다중 모달 긴 맥락을 다루는 능력을 향상시킵니다.

- **Performance Highlights**: LongLLaVA는 다양한 벤치마크에서 경쟁력 있는 결과를 달성하며, 특히 VNBench에서 정보 검색, 이미지 수 세기 및 정렬 작업에서 선두를 보이고 있습니다. 80GB A100 GPU에서 단일 GPU 환경에서도 1,000개의 이미지를 처리할 수 있어 뛰어난 효율성을 보여줍니다.



### Multi-stream deep learning framework to predict mild cognitive impairment with Rey Complex Figure Tes (https://arxiv.org/abs/2409.02883)
Comments:
          20 pages, 3 figures, 2 tables

- **What's New**: 이 논문에서는 Rey Complex Figure Test (RCFT)를 활용하여 인지 기능을 평가할 수 있는 새로운 다중 스트림(deep learning) 딥러닝 프레임워크를 제안합니다. 이 모델은 두 가지의 프로세싱 스트림(spatial stream과 scoring stream)을 통합하여 작동하며, 1,740명의 한국인 샘플을 기반으로 훈련되었습니다.

- **Technical Details**: 제안된 모델은 두 가지 스트림을 통합하여 사용합니다. 첫 번째 스트림은 원본 RCFT 이미지를 사용하는 multi-head self-attention 기반의 spatial stream이며, 두 번째 스트림은 자동 점수화 시스템을 이용한 scoring stream입니다. 이 모델은 222명의 외부 환자 데이터를 통해 검증되었습니다.

- **Performance Highlights**: 제안된 다중 스트림 모델은 외부 검증에서 기존의 베이스라인 모델들보다 우수한 성능(AUC = 0.872, Accuracy = 0.781)을 보였습니다. 이는 모델이 미세한 인지 결함을 감지하는 능력을 높여주며, 다양한 임상 환경에서도 더 나은 신뢰성을 제공할 수 있도록 합니다.



### Hybrid Imitation-Learning Motion Planner for Urban Driving (https://arxiv.org/abs/2409.02871)
- **What's New**: 본 논문에서는 기존의 학습 기반(Learning-based) 및 최적화 기반(Optimization-based) 기법을 결합한 새로운 하이브리드 모션 플래너(Hybrid Motion Planner)를 제안합니다. 이 모델은 인류 운전자의 행동을 모방하는 경로를 생성한 후, 이를 최적화하며 충돌 및 동작 불가능한 상황을 피하는 데 중점을 둡니다.

- **Technical Details**: 이 모델은 Multilayer Perceptron (MLP)을 이용해 인간처럼 경로를 생성하고, Model Predictive Trajectory (MPT) 알고리즘을 활용하여 이를 최적화합니다. 시스템은 에고 차량의 상태, 주변 인식 정보, 목표 목적지를 입력받아 샘플 기반 경로를 생성하며, 15개의 서로 다른 경로를 계산하여 가장 높은 점수를 부여받는 경로를 선택합니다.

- **Performance Highlights**: 시뮬레이션 실험을 통해 하이브리드 모델의 효과성을 확인하고, 실제 자율주행 차량에 배포하여 성능을 검증하였습니다. 기존 연구들이 시뮬레이션에만 국한된 점과는 달리, 본 연구는 실제 도시 환경에서의 실용성과 견고성을 시연하고 있습니다.



### Oops, I Sampled it Again: Reinterpreting Confidence Intervals in Few-Shot Learning (https://arxiv.org/abs/2409.02850)
- **What's New**: 본 논문은 Few-Shot Learning (FSL)에서 Confidence Interval (CI)을 계산하는 기존 방법의 문제점을 다루고 있습니다. 특히, 샘플링 시 교체를 허용하는 방식이 데이터의 불확실성을 과소평가하는 경향이 있음을 지적합니다. 이 문제를 해결하기 위한 새로운 접근 방법들을 제안하고 있습니다.

- **Technical Details**: 논문에서는 Closed CIs (CCIs)와 Open CIs (OCIs) 간의 차이를 보다 정량적으로 측정하기 위한 방법론을 제시합니다. 기존 CCIs는 Lindeberg-Lévy Central Limit Theorem (CLT)을 사용하여 통계적으로 유효한 CI를 생성하려는 반면, OCIs는 동일한 데이터 세트에서 실험을 반복했을 때의 결과 범위를 나타냅니다. 이는 데이터를 교체하지 않고 샘플링하는 방식을 통해 달성됩니다.

- **Performance Highlights**: 연구에 따르면, Pair Tests (PT) 방법을 사용함으로써 다양한 방법들을 비교할 때 더 일관된 결과를 얻을 수 있으며, CCIs를 사용한 전통적인 방법들과 OCIs를 비교했을 때 통계적으로 유의미한 다른 결론을 도출하는 경향이 있음을 보여줍니다. 특히, Traffic Signs 데이터셋에서 CLIP 방식이 DINO보다 낮은 성능을 기록했으나, PT에서는 그 결과가 반대로 나타나 통계적 해석의 중요성을 강조합니다.



### R2GQA: Retriever-Reader-Generator Question Answering System to Support Students Understanding Legal Regulations in Higher Education (https://arxiv.org/abs/2409.02840)
- **What's New**: 본 논문에서는 R2GQA 시스템을 제안합니다. 이 시스템은 Retriever-Reader-Generator 기반의 질문-응답 시스템으로, 문서 검색기(Document Retriever), 기계 독해기(Machine Reader), 답변 생성기(Answer Generator)의 세 가지 주요 구성 요소로 이루어져 있습니다. 또한, 베트남 대학 교육 규정에 대한 9,758개의 질문-답변 쌍으로 구성된 ViRHE4QA 데이터셋을 구축하였습니다.

- **Technical Details**: R2GQA 시스템은 문서 검색 모듈이 고급 정보 검색 기술을 활용하여 법적 규정 문서의 컨텍스트를 추출하고, 기계 독해 모듈이 최신 자연어 이해 알고리즘을 이용해 문서를 이해하고 답변을 추출합니다. 마지막으로, 답변 생성기는 추출된 답변을 간결하고 유익한 형태로 합성합니다. 이 시스템은 베트남어로 추상적(answer type: abstractive) 답변을 제공하는 첫 번째 시스템입니다.

- **Performance Highlights**: 우리는 실험을 통해 R2GQA 시스템의 유효성을 입증하였으며, 이는 학생들이 법적 규정을 이해하는 데 있어 큰 도움이 될 것으로 기대합니다. R2GQA 시스템과 ViRHE4QA 데이터셋이 학생들이 복잡한 법적 문서와 규정을 탐색하는 데 크게 기여할 것으로 보이며, 이러한 수단을 통해 학생들이 정보에 입각한 결정을 내리고 제도 정책을 효과적으로 준수할 수 있도록 지원할 것입니다.



### Exploring Sentiment Dynamics and Predictive Behaviors in Cryptocurrency Discussions by Few-Shot Learning with Large Language Models (https://arxiv.org/abs/2409.02836)
- **What's New**: 본 연구는 암호화폐 관련 논의에서 Predictive statements, Hope speech 및 Regret Detection 행동을 분석합니다.

- **Technical Details**: 고급 자연어 처리 기술을 활용하며 'Prediction statements'라는 새로운 분류 체계를 도입하여 댓글을 Predictive Incremental, Predictive Decremental, Predictive Neutral, Non-Predictive로 카테고리화합니다. GPT-4o라는 최첨단 대규모 언어 모델을 사용하여 Cardano, Binance, Matic, Fantom, Ripple 등 5개의 주요 암호화폐에서 감정 역학을 탐색합니다.

- **Performance Highlights**: Matic은 낙관적인 예측을 나타내는 경향이 다른 암호화폐보다 현저히 높다는 분석 결과가 나왔으며, 희망과 후회의 감정 사이에 복잡한 상호작용이 있음을 밝혀냈습니다. 데이터량과 자원 가용성과 관련된 한계를 겪었음에도 투자 행동 및 암호화폐 시장의 감정 트렌드에 대한 귀중한 발견을 보고하였고, 이는 전략적 의사결정 및 미래 연구에 기여할 것입니다.



### A hybrid FEM-PINN method for time-dependent partial differential equations (https://arxiv.org/abs/2409.02810)
Comments:
          25pages

- **What's New**: 이 연구에서는 진화 부분 미분 방정식(evolution PDEs)을 해결하기 위해 시간 유한 요소 방법(time finite element method)과 심층 신경망(deep neural networks)을 융합한 혼합 수치 방법을 제시합니다. 기존의 신경망이 시공간(spatiotemporal) 영역에 정의되어 있는 것과 달리, 우리의 방법론은 신경망의 출력을 공간 의존 계수로 정의하고 시간 방향으로 유한 요소 기초 함수(finite element basis functions)를 활용합니다.

- **Technical Details**: 시간 방향으로 Galerkin 또는 collocation projection을 적용하여 공간 의존 계수에 대한 PDE 시스템을 얻고, 이를 PINN의 틀 안에서 근사합니다. 혼합 형식화의 특장점은 통계적 오차(statistical errors)를 회피하고, 신경망의 출력이 축소된 공간 기초 함수(reduced spatial basis functions) 집합으로 간주될 수 있다는 것입니다. 고차원성과 낮은 정규성(low regularity) 문제를 완화하기 위해 명시적 밀도 모델(explicit density model)을 사용하여 PDE 잔여(residual)로 유도된 분포를 근사하고, 학습된 밀도 모델로부터 새로운 시간 종속 랜덤 샘플을 추가하여 훈련 세트를 보강합니다.

- **Performance Highlights**: 제안하는 방법의 효율성과 효과는 여러 수치 실험을 통해 입증되었습니다. 이 방법은 전통적인 PINN보다 더 나은 성능을 보여주며, 특히 장기 통합(long-term integration) 문제에 강력한 성능을 보입니다.



### Towards Edge-Based Data Lake Architecture for Intelligent Transportation System (https://arxiv.org/abs/2409.02808)
- **What's New**: 본 논문에서는 Intelligent Transportation Systems (ITS)에서 생성되는 복잡한 데이터를 효과적으로 통합하고 분석하기 위해 Edge-based Data Lake Architecture를 제안합니다. 이 아키텍처는 확장성, 결함 허용성(fault tolerance), 성능을 제공하여 의사결정을 개선하고 보다 지능적인 교통 생태계를 위한 혁신적인 서비스를 강화합니다.

- **Technical Details**: Edge-based Data Lake Architecture는 네트워크 에지에서 처리 및 저장 자원을 활용하며, 효율적인 데이터 통합 및 전처리(processing), 추론(inference)을 가능하게 합니다. Multi-access Edge Computing (MEC) 인프라를 기반으로 하는 이 아키텍처는 각각의 계층이 데이터를 다르게 활용하여 분산 아키텍처를 설계합니다.

- **Performance Highlights**: 제안된 아키텍처는 세 가지 사용 사례: (i) 차량 센서 네트워크, (ii) 모바일 네트워크, 및 (iii) 운전자를 식별하는 애플리케이션에 대한 분석을 통해 효과성을 입증했습니다. 이 아키텍처는 원활한 데이터 통합, 저지연 처리 및 고급 데이터 분석에 대한 유연한 지원을 제공합니다.



### Governing dual-use technologies: Case studies of international security agreements and lessons for AI governanc (https://arxiv.org/abs/2409.02779)
- **What's New**: 이 논문은 고급 AI의 글로벌 보안 위험을 줄이기 위한 국제 AI 거버넌스 협정 및 기관의 설계에 관한 사례 연구를 수행한 결과를 담고 있습니다. 여기서는 이중 용도 기술에 초점을 맞춰 핵 안전, 화학 무기, 생물 보안 및 수출 통제와 같은 여러 국제 안전 협정을 개관하고 있습니다.

- **Technical Details**: 논문은 5개의 국제 안전 협정에 대한 사례 연구를 통해 AI 거버넌스를 위한 교훈을 도출하였습니다. 주요 질문은 (a) 목적, (b) 핵심 권한, (c) 거버넌스 구조, (d) 비불이행 사례 등입니다. 이 연구 결과, 강력한 검증 방법(verification methods), 국가 간 권력 균형 조정(core powers and decisions), 기술 전문성(technical expertise) 등을 식별하는 것이 중요하다는 점을 강조합니다.

- **Performance Highlights**: AI 개발 및 배포에 대해 강력하고 적응 가능한 검증 방법을 개발하고, 비불이행을 방지하거나 그에 대응할 전략을 검토해야 한다고 강조합니다. 또한, AI 거버넌스 기관에서는 주요 결정을 내리는 방식과 글로벌 대표성을 유지하는 거버넌스 구조 설계의 필요성을 언급하고 있습니다.



### Tractable Offline Learning of Regular Decision Processes (https://arxiv.org/abs/2409.02747)
Comments:
          To appear in EWRL 2024

- **What's New**: 이 연구에서는 Regular Decision Processes (RDPs)라는 비마르코프 (non-Markovian) 환경에서의 오프라인 강화학습 (Reinforcement Learning, RL) 접근 방식을 제시합니다. 특히, 기존 알고리즘의 두 가지 주요 한계를 극복하며 새로운 기법을 도입하였습니다.

- **Technical Details**: 최신 알고리즘은 формальный 언어(theory of formal languages)를 기반으로 한 새로운 유사 메트릭을 개발하였습니다. 이를 통해 $L_	ext{∞}^p$-구별 가능성 파라미터에 대한 의존성을 없애고, Count-Min-Sketch (CMS)를 사용하여 메모리 요구 사항을 줄였습니다.

- **Performance Highlights**: 제안된 기법은 low complexity 환경에서의 샘플 수를 줄이며, 긴 계획 수립 시간에 대한 메모리 요구 사항을 완화합니다. 실험적으로 이 접근 방식의 유효성을 검증하였습니다.



### GET-UP: GEomeTric-aware Depth Estimation with Radar Points UPsampling (https://arxiv.org/abs/2409.02720)
Comments:
          Accepted by WACV 2025

- **What's New**: 이번 연구에서는 GET-UP이라는 새로운 깊이 추정 프레임워크를 제안합니다. 이 프레임워크는 레이더 데이터를 활용하여 2D와 3D 정보를 상호 교환하고 집계합니다.

- **Technical Details**: GET-UP은 attention-enhanced Graph Neural Networks (GNN)를 사용하여 레이더 포인트 클라우드 내의 귀중한 기하학적 정보를 활용합니다. 또한, point cloud upsampling을 통해 레이더 포인트 클라우드를 밀집하게 만들고 LiDAR 데이터의 지침을 받아 추가적인 3D 기능을 도출합니다.

- **Performance Highlights**: GET-UP은 nuScenes 데이터셋에서 기존 최첨단 모델에 비해 MAE에서 15.3%, RMSE에서 14.7%의 성능 향상을 기록하였습니다.



### Incorporating Like-Minded Peers to Overcome Friend Data Sparsity in Session-Based Social Recommendations (https://arxiv.org/abs/2409.02702)
Comments:
          None

- **What's New**: 본 논문에서는 소셜 네트워크 내에서의 관계를 활용하여 세션 기반 추천(Session-based Recommendation, SR)의 성능을 향상시키는 세션 기반 소셜 추천(Session-based Social Recommendation, SSR)의 개념을 제안합니다. 특히, 'Like-minded Peers' (LMP)라는 새로운 개념을 도입하여 이 문제를 해결하고, 소셜 친구들과의 데이터 희소성 문제를 완화합니다. 

- **Technical Details**: 우리는 Transformer Encoder with Graph Attention Aggregator Recommendation (TEGAARec)이라는 새로운 모델을 제안합니다. 이 모델은 TEGAA 모듈과 GAT 기반 사회 집계 모듈을 포함합니다. TEGAA 모듈은 타겟 사용자와 LMP 사용자의 장기 및 단기 관심을 캡처하고 병합하며, GAT 기반 사회 집계 모듈은 타겟 사용자의 동적 관심과 사회적 영향을 가중치로 집계합니다.

- **Performance Highlights**: 실제 데이터셋 4개에 대한 광범위한 실험을 통해 제안된 모델이 기존의 최신 기술들에 비해 여러 메트릭에서 성능이 크게 개선됨을 보여주었습니다.



### The Role of Artificial Intelligence and Machine Learning in Software Testing (https://arxiv.org/abs/2409.02693)
- **What's New**: AI와 ML(머신 러닝)의 발전이 소프트웨어 개발의 중요한 측면인 소프트웨어 테스트의 방식에 혁신적인 변화를 가져왔습니다. 이 논문은 기존 문헌을 검토하고 AI와 ML의 소프트웨어 테스트에서의 역할을 탐구합니다.

- **Technical Details**: 이 논문에서는 AI와 ML이 테스트 케이스 생성(test case generation), 테스트 실행(test execution), 결과 분석(result analysis)과 같은 복잡한 작업을 자동화하여 소프트웨어 테스트의 효율성과 효과성을 향상시키는 방법을 다룹니다. AI는 역사적 데이터를 분석하여 잠재적인 실패 영역을 예측하고, 목표 지향적이고 효율적인 테스트를 가능하게 합니다.

- **Performance Highlights**: 다양한 AI 기반 테스트 도구(Eggplant AI, Selenium, Appvance, Applitools Eyes, Katalon Studio, Tricentis Tosca)의 사례 연구를 통해 테스트 효율성, 정확성 및 소프트웨어 품질의 상당한 개선을 보여주었습니다.



### LLM-Assisted Visual Analytics: Opportunities and Challenges (https://arxiv.org/abs/2409.02691)
Comments:
          Accepted at EG UK Computer Graphics & Visual Computing 2024

- **What's New**: 대규모 언어 모델(LLMs)을 시각 분석(Visual Analytics, VA) 시스템에 통합함으로써 자연어 처리 능력을 통해 새로운 가능성을 탐색합니다. LLMs는 데이터 관리, 언어 상호작용, 시각화 생성 및 언어 생성 과정에 통합되어 VA의 활용도를 높이는데 기여할 수 있습니다.

- **Technical Details**: 본 논문에서는 LLMs가 VA 파이프라인에서 어떻게 활용되고 있는지를 살펴보고, LLM의 샘플 작업들을 소개합니다. 특히 LLMs의 적용을 통한 데이터 관리, 언어 상호작용, 시각화 생성 및 언어 생성의 최신 연구 동향을 강조합니다. LLMs는 비구조화된 텍스트 데이터를 처리하고 의미 있는 통찰을 도출하는 데 도움을 줄 수 있습니다. 예를 들어, LLMs를 활용하여 다수의 preprocessing 작업을 수행하거나, 기존 데이터를 바탕으로 합성 데이터를 생성하는 사례를 소개합니다.

- **Performance Highlights**: LLMs는 사용자가 요구하는 정보 검색, 비구조화 데이터의 테이블화, 데이터 전처리 및 오류 탐지 과정을 통해 분석 작업의 효율성을 상당히 높일 수 있습니다. 하지만 사용자에게 정보의 신뢰성과 출처를 평가하기 어려운 점, 개인 정보 보호 문제, 그리고 복잡한 데이터 유형을 처리하는 능력의 한계 등 다양한 도전 과제가 존재합니다.



### Deconfounded Causality-aware Parameter-Efficient Fine-Tuning for Problem-Solving Improvement of LLMs (https://arxiv.org/abs/2409.02686)
- **What's New**: 본 논문은 Large Language Models (LLMs)의 추론 능력을 향상시키기 위해 Deconfounded Causal Adaptation (DCA)이라는 새로운 Parameter-Efficient Fine-Tuning (PEFT) 방법을 제안합니다. 이를 통해 모델이 일반 문제 해결 능력을 추출하고 다양한 질문에 적용하게 합니다.

- **Technical Details**: 논문에서는 LLM의 텍스트 생성 프로세스를 주의(attention) 및 표현(representation) 수준에서 시각화하여 모델이 진정한 추론 능력을 가지고 있는지를 조사합니다. 또한, LLM의 추론 과정을 인과적(causal) 프레임워크로 정형화하여 시각화에서 관찰된 문제를 설명하고, DCA를 활용하여 모델의 추론 능력을 향상시키는 방법을 제시합니다.

- **Performance Highlights**: 실험 결과, DCA 방법이 다양한 벤치마크에서 일관되게 기준선(baseline)을 초과하는 성능을 보였으며, 단 1.2M의 조정 가능한 매개변수로 다른 미세 조정 방법과 비교해 더 나은 또는 동등한 결과를 달성하여 모델의 전반적인 정확도와 신뢰성을 향상시키는 것을 입증했습니다.



### RouterRetriever: Exploring the Benefits of Routing over Multiple Expert Embedding Models (https://arxiv.org/abs/2409.02685)
- **What's New**: 이번 연구에서는 RouterRetriever라는 정보 검색(Information Retrieval) 모델을 소개합니다. 이 모델은 여러 도메인 특화 전문가(Expert)를 활용하고, 라우팅 메커니즘을 통해 각 쿼리에 적합한 전문가를 선택합니다. 이는 도메인 특화 데이터에 대한 성능을 극대화할 수 있는 새로운 접근법입니다.

- **Technical Details**: RouterRetriever는 각 도메인에 대해 훈련된 전문가 게이트를 활용하여 쿼리 임베딩(Embedding)을 처리합니다. 쿼리 입력 시, 라우팅 메커니즘이 작동하여 입력 쿼리와 파일럿 임베딩(Pilot Embeddings) 간의 유사도 점수를 계산하고, 가장 높은 점수를 가진 전문가를 선택합니다. 본 모델은 LoRA 모듈을 사용하여 경량화되며, 전문가의 추가 및 제거 시 추가 훈련이 필요 없습니다.

- **Performance Highlights**: BEIR 벤치마크에서 RouterRetriever는 MSMARCO로 훈련된 모델보다 절대 nDCG@10이 +2.1 향상되었으며, 멀티 태스크 훈련 모델보다 +3.2의 성능 향상을 보였습니다. 다양한 도메인 전문가를 추가할수록 지속적인 성능 향상이 나타나며, 도메인 특화 전문가의 투입이 일반 도메인 전문가보다 더 많은 성능 수익을 제공함을 확인했습니다.



### Neural Networks with LSTM and GRU in Modeling Active Fires in the Amazon (https://arxiv.org/abs/2409.02681)
Comments:
          16 pages, in Portuguese language, 24 figures

- **What's New**: 본 연구는 브라질 아마존에서 AQUA_M-T 위성에 의해 감지된 화재 점의 역사적 시계열 모델링 및 예측을 위한 포괄적인 방법론을 제시합니다. 혼합형 순환 신경망(Recurrent Neural Network, RNN) 모델을 사용하여 월별 화재 점 누적값을 예측하며, 이는 장기 메모리(Long Short-Term Memory, LSTM)와 게이티드 순환 유닛(Gated Recurrent Unit, GRU) 아키텍처를 포함합니다.

- **Technical Details**: 제안된 방법론은 데이터 준비, 모델 구성, 훈련 과정에서의 교차 검증을 포함하여 데이터를 잘 일반화할 수 있도록 합니다. 모델의 매개변수 수렴을 확인하며, 이 과정에서 LSTM과 GRU 모델의 혼합을 통해 복잡한 시간 패턴을 포착할 수 있습니다.

- **Performance Highlights**: 혼합 LSTM 및 GRU 모델은 12개월 선행 예측에서 개선된 정확도를 나타내며, 복잡한 시계열 패턴을 효과적으로 캡처할 수 있음을 보여줍니다. 이 연구는 환경 모니터링, 특히 화재 점 예측에 있어 심층 학습 기술의 적용에 중요한 기여를 하며, 향후 다른 시계열 예측 문제에도 적응 가능성을 보여줍니다.



### Independence Constrained Disentangled Representation Learning from Epistemological Perspectiv (https://arxiv.org/abs/2409.02672)
- **What's New**: 본 논문은 분리된 표현 학습(Disentangled Representation Learning)에 대한 새로운 접근 방식을 제안합니다. 특히, 잠재 변수 간의 관계를 이해하기 위해 인식론(Epistemology)에서의 개념을 도입하여, 두 수준(latent space) 잠재 공간 프레임워크를 구축합니다. 이를 통해 잠재 변수 간의 독립성 여부에 대한 논쟁을 종합적으로 해결합니다.

- **Technical Details**: 제안된 방법은 Generative Adversarial Network (GAN) 프레임워크를 채택하여, 두 가지 제약 조건인 상호 정보(mutual information)와 독립성(independence)을 결합하여 최적의 분리된 표현 학습을 실현합니다. 잠재 변수는 두 개의 레벨로 군집화되며, 원자 수준(Atomic Level)에서는 상호 독립적이어야 하며, 복합 수준(Complex Level)에서는 인과적 관계를 통해 원자 변수들 간의 연관성이 있을 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 여러 일반적인 평가 지표에 걸쳐 기준 방법보다 일관되게 높은 성능을 보임을 확인하였습니다. 이는 다양한 의미적 요인을 효과적으로 분리할 수 있는 능력을 보여주며, 알고리즘의 설명 가능성(explainability) 향상에 기여합니다.



### Causality-Aware Transformer Networks for Robotic Navigation (https://arxiv.org/abs/2409.02669)
- **What's New**: 이 논문은 Embodied AI 작업의 독특한 요구사항을 충족시키기 위해 Causality(인과성)를 기반으로 한 새로운 접근 방식을 제안합니다. 특히, 기존의 RNN과 Transformer 모델이 Embodied AI 작업에 적합하지 않다는 점을 드러내고 이를 개선할 방법으로 Causal Understanding Module(인과적 이해 모듈)을 도입하여 성능을 극대화합니다.

- **Technical Details**: Causality-Aware Transformer (CAT) 네트워크는 환경 이해 능력을 향상시키기 위해 설계된 Causal Understanding Module을 포함합니다. 이 방식은 task-specific inductive biases(작업 특화 유도 편향)를 배제하며, End-to-End 방식으로 학습됩니다. 이 논문은 기존의 sequential methods(순차적 방법)가 Embodied AI 작업에서의 한계를 어떻게 드러내는지를 응시합니다.

- **Performance Highlights**: 제안된 방식은 다양한 시뮬레이션 환경 및 작업 세팅에서 벤치마크 성능을 지속적으로 초과하는 결과를 보여줍니다. ablation studies(분리 연구)는 Causal Understanding Module의 효과성과 효율성을 강조하며, Reinforcement Learning 및 Supervised Learning 설정에서도 이 모듈이 전체 성능을 향상시킴을 입증합니다.



### PoseTalk: Text-and-Audio-based Pose Control and Motion Refinement for One-Shot Talking Head Generation (https://arxiv.org/abs/2409.02657)
Comments:
          7+5 pages, 15 figures

- **What's New**: 본 연구에서는 음성과 텍스트 프롬프트에 기반하여 자유롭게 생성된 입맞춤 동기화된 토킹 헤드 비디오를 생성할 수 있는 새로운 시스템인 PoseTalk를 제안합니다.

- **Technical Details**: PoseTalk는 두 가지 입력 소스, 즉 오디오와 텍스트 프롬프트를 활용해 헤드 포즈를 생성합니다. 특히, Pose Latent Diffusion (PLD) 모델을 통해 동작 잠재 공간에서 포즈를 예측합니다. 또한, CoarseNet과 RefineNet의 두 단계 네트워크를 통해 자연스러운 비디오 생성을 위한 정제 기반 학습 전략이 사용됩니다.

- **Performance Highlights**: PoseTalk의 포즈 예측 전략은 텍스트 전용 또는 오디오 전용 방법보다 뛰어난 포즈 다양성과 사실성을 달성하였으며, 비디오 생성 모델은 자연스러운 머리 동작의 토킹 비디오 합성에서 최신 기술을 초월하는 성능을 보였습니다.



### OpenFact at CheckThat! 2024: Combining Multiple Attack Methods for Effective Adversarial Text Generation (https://arxiv.org/abs/2409.02649)
Comments:
          CLEF 2024 - Conference and Labs of the Evaluation Forum

- **What's New**: 이 논문은 CLEF 2024 Task 6에서의 체크 상태! 실험과 결과를 다루며, 신뢰성 평가의 강건성을 테스트하기 위해 적대적 예제를 생성하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 연구팀은 적대적 공격 방법으로 BERT-Attack, Genetic 알고리즘, TextFooler 및 CLARE를 포함하여 다섯 가지 데이터 세트에서 다양한 방법을 테스트 및 수정했습니다. 향상된 공격 효과를 위해 하이브리드 방법을 도입하고, 의미를 유지하면서 공격 성공률을 높이는 방법을 탐구했습니다.

- **Performance Highlights**: 이 연구의 결과는 여러 방법을 수정하고 결합하여 보다 정교하고 효과적인 공격 전략을 만들 수 있는 가능성을 보여줍니다. 이를 통해 다양한 시스템의 내구성과 보안을 강화하는 데 기여할 수 있습니다.



### AdvSecureNet: A Python Toolkit for Adversarial Machine Learning (https://arxiv.org/abs/2409.02629)
- **What's New**: AdvSecureNet는 다중 GPU 지원, CLI 및 API 인터페이스, 외부 YAML 구성 파일을 네이티브로 지원하는 최초의 Adversarial Machine Learning 툴킷입니다. 이 툴킷은 다양한 공격 및 방어 메커니즘, 평가 메트릭스를 포함하여 머신러닝 연구의 유연성과 재현성을 향상시킵니다.

- **Technical Details**: AdvSecureNet는 PyTorch에 기반한 모듈식 툴킷으로, 다중 GPU 환경에서 작동하도록 최적화되어 있습니다. 이 툴킷은 Gradient-based, Decision-based, White-box, Black-box 등 다양한 공격 방법과 Adversarial Training, Ensemble Adversarial Training 등의 방어 메커니즘을 지원합니다. 또한, YAML 구성 파일을 통해 실험 매개변수를 쉽게 조정할 수 있습니다.

- **Performance Highlights**: AdvSecureNet는 다중 GPU 환경에서 다른 툴킷 대비 빠른 실행 시간을 기록합니다. 예를 들어, CIFAR-10에서의 Adversarial Training이 단일 GPU에서 5.07분 걸릴 때 7개의 GPU를 사용할 경우 2.77분으로 단축됩니다. ImageNet에서는 단일 GPU에서 240분 걸리던 것이 7개의 GPU에서 30분으로 줄어들어 8배의 속도 향상을 보여줍니다.



### SurgTrack: CAD-Free 3D Tracking of Real-world Surgical Instruments (https://arxiv.org/abs/2409.02598)
- **What's New**: 이 논문에서 제안하는 SurgTrack는 CAD 없이도 3D 수술 기구 추적을 수행할 수 있는 새로운 방법론입니다. 이는 수술 기구의 3D 표현을 모델링하기 위해 Instrument Signed Distance Field (SDF)를 사용하며, 비침습적이고 유연한 3D 천공 기능을 제공합니다.

- **Technical Details**: SurgTrack는 두 가지 단계로 구성되어 있습니다. 첫 번째 단계에서는 Instrument SDF를 이용하여 수술 기구의 3D 등록을 수행하고, 두 번째 단계에서는 자세 메모리 풀과 자세 그래프 최적화 모듈을 통해 6 자유도(DoF) 자세를 추적합니다. 이 과정에서 occlusion과 낮은 텍스처 문제를 해결하기 위해 다양한 최적화 기법을 적용합니다.

- **Performance Highlights**: SurgTrack는 88.82%의 ADD-S 성능과 12.85의 재구성 오류로, 기존 방법들에 비해 뛰어난 성능을 보이며, 실험을 통해 범용성과 확장성도 검증되었습니다.



### AlignGroup: Learning and Aligning Group Consensus with Member Preferences for Group Recommendation (https://arxiv.org/abs/2409.02580)
Comments:
          10 pages, accepted by CIKM 2024

- **What's New**: 본 논문에서는 AlignGroup이라는 새로운 그룹 추천 방법을 제안합니다. 이 방법은 그룹의 동의(Consensus)와 개인 회원의 선호(Preferences)를 동시에 고려하여 그룹 의사결정을 추론하는 데 중점을 둡니다.

- **Technical Details**: AlignGroup은 하이퍼그래프 신경망(Hypergraph Neural Network)을 활용하여 내부 및 외부 그룹 관계를 학습합니다. 또한, 자기 지도 정렬 태스크(Self-supervised Alignment Task)를 도입하여 그룹의 동의와 회원들의 공통 선호를 정렬함으로써 세밀한 그룹 의사결정을 포착합니다.

- **Performance Highlights**: 실제 데이터셋에서 수행한 실험 결과, AlignGroup은 그룹 추천 작업(Group Recommendation Task) 및 사용자 추천 작업(User Recommendation Task) 모두에서 최신 기술(State-of-the-Art)을 초월하는 성능을 보였습니다.



### Solving Video Inverse Problems Using Image Diffusion Models (https://arxiv.org/abs/2409.02574)
Comments:
          22 pages, 16 figures

- **What's New**: 최근 디퓨전 모델에 기반한 역문제 해결 방법(DIFFUSION INVERSE SOLVERS, DIS)이 등장하여 이미지 초해상도, 디블러링, 인페인팅 등 다양한 역문제를 해결하는 데에 최첨단 접근법으로 자리잡고 있습니다. 기존의 영상 역문제에 대한 연구는 미비했던 반면, 본 논문에서는 이미지를 기반으로 한 새로운 영상 역문제 해결 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 최근 성공적으로 수행된 분해된 디퓨전 샘플러(Decomposed Diffusion Sampler, DDS)에서 영감을 받아 영상의 시간 차원을 이미지 디퓨전 모델의 배치 차원으로 다루며, 각 이미지 디퓨전 모델에서 파생된 노이즈 제거된 시공간 배치 내에서 시공간 최적화 문제를 해결합니다. 또한 배치 간 일관성을 보장하기 위해 배치 일관성 샘플링 전략을 도입하여 각 이미지 디퓨전 모델의 확률적 노이즈 구성 요소를 동기화합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 영상 역문제에서 다양한 시공간 손상을 효과적으로 처리하며, 최첨단 재구성 성능을 달성함을 확인하였습니다.



### Advancing Cyber Incident Timeline Analysis Through Rule Based AI and Large Language Models (https://arxiv.org/abs/2409.02572)
Comments:
          25 pages

- **What's New**: 본 논문에서는 디지털 포렌식(Digital Forensics, DF)에서 타임라인 포렌식(Timeline Forensics, TF)의 핵심인 타임라인 분석(Timeline Analysis, TA)을 혁신적으로 개선한 새로운 프레임워크, GenDFIR을 소개합니다.

- **Technical Details**: GenDFIR는 Rule-Based Artificial Intelligence (R-BAI) 알고리즘과 Large Language Models (LLMs)를 결합하여 TA 과정을 자동화하는 두 단계로 구성됩니다. 첫 번째 단계에서는 R-BAI를 사용하여 사전 정의된 규칙에 따라 이상 디지털 아티팩트를 식별하고 선택합니다. 두 번째 단계에서는 선택된 아티팩트를 임베딩(embeddings)으로 변환하여 Retrieval-Augmented Generation (RAG) 에이전트의 도움으로 LLM에 처리합니다.

- **Performance Highlights**: 우리는 GenDFIR의 성능, 효율성, 신뢰성을 다양한 메트릭을 사용하여 합성 사이버 사건 시뮬레이션 시나리오에서 평가하였습니다. 결과적으로 R-BAI와 LLM을 통합한 TA의 상당한 잠재력을 보여주었으며, 이는 고급 위협 탐지 및 사건 재구성을 위한 새로운 가능성을 제시합니다.



### More is More: Addition Bias in Large Language Models (https://arxiv.org/abs/2409.02569)
Comments:
          25 pages, 8 figures

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)에서 발생하는 additive bias(추가 편향) 복잡성을 탐구하며, 인간의 인지 편향과 유사한 현상을 관찰하여 LLMs의 의사결정 과정에 미치는 영향을 분석합니다.

- **Technical Details**: 본 연구에서는 일련의 실험을 통해 여러 LLM 모델(GPT-3.5 Turbo, Claude 3.5 Sonnet, Mistral 등)의 additive bias를 측정하는 작업을 수행했습니다. 측정된 작업에는 palindrome(회문) 생성, Lego 타워 균형 조정, 레시피 수정 등이 포함되었습니다.

- **Performance Highlights**: 실험 결과는 LLM이 additive 변화를 선호하는 경향을 명확히 보여주었습니다. 예를 들어, Llama 3.1은 회문 생성 작업에서 97.85%의 사건에서 추가하는 방식을 선택했으며, GPT-3.5 Turbo는 Lego 타워 균형 작업에서 76.38%의 사건에서 벽돌을 추가하는 방식으로 응답했습니다. 이러한 결과는 LLM의 디자인과 활용에 있어 환경적인 비용을 증가시킬 수 있는 단점이 있음을 시사합니다.



### Low-Resolution Object Recognition with Cross-Resolution Relational Contrastive Distillation (https://arxiv.org/abs/2409.02555)
Comments:
          This paper is accepted by IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)

- **What's New**: 본 연구에서는 저해상도(低解像度) 객체 인식 문제를 해결하기 위해, 고해상도 모델로부터 지식을 효과적으로 이전하기 위한 새로운 접근 방식인 cross-resolution relational contrastive distillation을 제안합니다.

- **Technical Details**: 이 방법은 로우해상도(low-resolution) 객체의 특징을 잘 학습하도록 구성된 학생 모델(student model)이 고해상도(高解像度) 모델의 행동을 모방하도록 돕습니다. 또한, contrastive relational distillation loss를 통해 다양한 관계 구조에서의 유사성을 보존하여 필요한 지식을 추출합니다.

- **Performance Highlights**: 저해상도 객체 분류 및 얼굴 인식 실험을 통해, 제안한 방법이 뛰어난 성능과 적응성을 보여주었습니다.



### Understanding eGFR Trajectories and Kidney Function Decline via Large Multimodal Models (https://arxiv.org/abs/2409.02530)
Comments:
          This preprint version includes corrections of typographical errors related to numerical values in Table 2, which were present in the version published at the BDH workshop in MIPR 2024. These corrections do not affect the overall conclusions of the study

- **What's New**: 이번 연구는 Large Multimodal Models (LMMs)가 eGFR(estimated Glomerular Filtration Rate) 예측에 효과적일 수 있음을 보여줍니다. 이 모델은 50명의 환자로부터 수집된 실험실 및 임상 데이터를 활용하여 특히 미래의 eGFR 수준 예측에 초점을 맞추고 있습니다.

- **Technical Details**: 연구에서는 다양한 prompting 기술과 LMM의 앙상블을 통합했습니다. 이러한 접근 방식은 정밀한 프롬프트 및 eGFR 궤적의 시각적 표현과 결합될 때 기존의 머신러닝 모델과 유사한 예측 성능을 제공하는 것으로 나타났습니다.

- **Performance Highlights**: LMMs는 정확한 프롬프트와 시각적 데이터로 통합되었을 때, 기존의 머신러닝 모델에 비견되는 예측 성능을 보여주며, 향후 복잡한 의료 예측 문제를 해결하기 위한 기초 모델의 활용 가능성을 제시합니다.



### Continual Diffuser (CoD): Mastering Continual Offline Reinforcement Learning with Experience Rehearsa (https://arxiv.org/abs/2409.02512)
- **What's New**: 본 논문에서는 지속적인 학습을 위한 새로운 접근으로 'Continual Diffuser (CoD)'라는 모델을 제안합니다. 이 모델은 섭동 기반 모델(diffusion-based model)의 장점을 활용하여 환경 변화에 효과적으로 적응할 수 있는 능력(plasticity)과 이전에 습득한 지식을 유지할 수 있는 능력(stability)을 동시에 강화하는 것을 목표로 합니다.

- **Technical Details**: CoD는 지속적 학습을 지원하기 위해 여러 도메인에서 90개의 작업 태스크를 포함하는 오프라인 벤치마크를 구축하였습니다. CoD는 각 태스크에 대해 순차 모델링(sequential modeling)과 조건부 생성(conditional generation)을 통해 의사 결정을 내리며, 이전 데이터셋의 일부를 리허설 버퍼(rehearsal buffer)로 보존하여 반복적으로 재생(replay)함으로써 습득한 지식을 유지합니다.

- **Performance Highlights**: 다양한 태스크에서 수행된 실험에 따르면 CoD는 유망한 플라스틱성과 안정성의 균형을 이루며, 기존의 섭동 기반 방법들과 다른 대표적인 기준선들보다 대부분의 작업에서 우수한 성능을 보였습니다.



### CoAst: Validation-Free Contribution Assessment for Federated Learning based on Cross-Round Valuation (https://arxiv.org/abs/2409.02495)
- **What's New**: CoAst는 검증 데이터 없이도 FL 참가자의 기여도를 평가할 수 있는 새로운 방법이다. 이는 참가자의 기여 평가를 단일 훈련 라운드가 아닌 여러 통신 라운드 간의 유사성을 기반으로 한다.

- **Technical Details**: CoAst는 중요한 모델 파라미터만 세는 가중치 양자화(quantization) 기술을 활용하고, 교차 라운드(valation) 평가 메커니즘을 통해 기여도를 평가한다. CoAst의 핵심 아이디어는 모디픽된 파라미터 업데이트를 바탕으로 하여 여러 후속 라운드의 글로벌 모델 업데이트와 현재 로컬 모델 파라미터의 유사성을 비교하는 것이다.

- **Performance Highlights**: CoAst는 기존의 검증 기반 방법과 비교할 때 유사한 평가 신뢰성을 보이며, 검증이 필요 없는 방법들보다 뛰어난 성능을 보인다.



### NeuroSpex: Neuro-Guided Speaker Extraction with Cross-Modal Attention (https://arxiv.org/abs/2409.02489)
- **What's New**: 본 논문에서는 뇌파 신호(EEG)를 활용하여 연설자 추출(Speaker Extraction)을 위한 새로운 모델인 NeuroSpex를 제안합니다. 이는 청중의 주의 집중 정보를 이용하여 혼합 음성을 효과적으로 분리하는 접근 방식을 소개합니다.

- **Technical Details**: NeuroSpex 모델은 4가지 구성 요소로 이루어져 있으며, 여기에는 음성 인코더(Speech Encoder), EEG 인코더(EEG Encoder), 연설자 추출기(Speaker Extractor), 음성 디코더(Speech Decoder)가 포함됩니다. EEG 인코더는 다채널 EEG 신호의 정보를 활용하여 연설자 추출기에 주의 집중 음성을 안내합니다. 또한, 교차 주의 메커니즘(Cross-Attention, CA)을 도입하여 음성 특징 표현을 강화하며 연설자 추출 마스크(Speaker Extraction Mask)를 생성합니다.

- **Performance Highlights**: 공개 데이터셋을 기반으로 한 실험 결과, 제안된 모델은 다양한 평가 지표에서 두 개의 기준 모델 대비 우수한 성능을 보였습니다.



### Boosting Generalizability towards Zero-Shot Cross-Dataset Single-Image Indoor Depth by Meta-Initialization (https://arxiv.org/abs/2409.02486)
Comments:
          IROS 2024. The version supersedes 2305.07269. arXiv admin note: text overlap with arXiv:2305.07269

- **What's New**: 이 논문은 실내 로봇의 깊이 추정 및 메타 학습을 융합하여 한 가지 단일 이미지 깊이 예측 문제에 대한 더 높은 일반화 성능을 달성하는 새로운 접근 방법을 제안합니다. 이를 위해 저자는 제로샷 크로스 데이터셋 추론(zero-shot cross-dataset inference)에 대한 고급 일반화 기술을 도입합니다.

- **Technical Details**: 본 연구는 메타 학습(metalearning)을 활용하여 연속적인 깊이 값을 가지는 복잡한 실내 환경에서의 깊이 추정 문제를 해결하기 위해 각 RGB-D 미니 배치를 작업으로 간주하는 세분화된(task) 작업을 제안합니다. 연구에 따르면, 메타-초기화(meta-initialization)를 통해 더 높은 일반성을 입증하고, 기존 방법 대비 성능 개선을 확인했습니다.

- **Performance Highlights**: 제안된 방법은 제한된 데이터에서 최대 27.8%의 RMSE 개선을 보여주었으며, 메타 학습 초기화 후의 정제(fine-tuning) 과정에서 기본적인 접근 방식보다 일관되게 더 우수한 성능을 나타났습니다. 이 연구는 실내 로봇 및 AR/VR 등의 실제 응용 분야에서의 강력하고 일반화된 깊이 추정을 가능하게 합니다.



### Adversarial Attacks on Machine Learning-Aided Visualizations (https://arxiv.org/abs/2409.02485)
Comments:
          This is the author's version of the article that has been accepted by the Journal of Visualization

- **What's New**: 본 논문에서는 ML(기계 학습)을 사용하는 시각화(Machine Learning for Visualization, ML4VIS) 분야의 보안 취약점에 대한 탐구를 진행하고 있으며, 특히 적대적 공격(Adversarial Attacks) 관점에서 ML 지원 시각화의 독특한 공격 표면(Attack Surface)을 정의하고 있습니다.

- **Technical Details**: 연구는 ML과 시각화 두 관점에서 접근하여, ML4VIS의 공격 표면을 분석합니다. 공격 예제에서는 다섯 가지 적대적 공격을 제시하며, 이러한 공격이 ML에 의해 생성된 시각화에 미치는 영향을 강조합니다. 특히 Neural Networks(NNs) 기반의 시각화 기법을 포함하여 다양한 공격 가능성을 탐구하고 있습니다.

- **Performance Highlights**: 연구 결과, 적대적 공격이 ML4VIS에 의해 생성된 시각화를 조작하여 사용자의 판단을 오도할 수 있음을 보여주며, 이러한 공격의 탐지 및 방어 메커니즘을 개발할 필요성이 시급함을 강조합니다.



### TASAR: Transferable Attack on Skeletal Action Recognition (https://arxiv.org/abs/2409.02483)
Comments:
          arXiv admin note: text overlap with arXiv:2407.08572

- **What's New**: 이번 연구에서는 Skeletal Action Recognition(S-HAR)에 대한 최초의 Transfer-based Attack 방법인 TASAR을 제안합니다. TASAR은 손실 표면의 매끄러움을 개선함으로써 S-HAR의 적대적 전이 가능성을 증가시키고, 이에 따른 공격 방법론의 한계를 극복합니다.

- **Technical Details**: TASAR은 포스트 트레인(Double Bayesian Optimization) 기법을 통해 미리 훈련된 대리 모델의 매끄러운 모델 후행을 탐색합니다. 이 과정에서 각 프레임을 독립적으로 처리하는 기존의 방법과 달리, 시간적 연속성을 고려하여 모션 동역학을 Bayesian 공격 그래디언트에 통합합니다.

- **Performance Highlights**: RobustBenchHAR로 명명된 새로운 대규모 S-HAR 벤치마크를 구축하여, 7개의 S-HAR 모델, 10개의 공격 방법 및 3개의 데이터 세트를 포함하고, TASAR의 우수성과 일반화 가능성을 입증했습니다.



### Fast, High-Quality and Parameter-Efficient Articulatory Synthesis using Differentiable DSP (https://arxiv.org/abs/2409.02451)
Comments:
          accepted for Spoken Language Technology Workshop 2024

- **What's New**: 본 연구에서는 DDSP(Differentiable Digital Signal Processing)를 활용하여 EMA(전자기 발음 측정) 데이터로부터 고속, 고품질의 말합성(speech synthesis)을 수행할 수 있는 아티큘레이터리 보코더를 제안합니다. 이 모델은 화음(harmonics)과 잡음(noise) 불균형 문제를 해결하고, 다중 해상도 적대적 손실(multi-resolution adversarial loss)을 추가하여 합성 품질을 개선하였습니다.

- **Technical Details**: 제안된 모델은 EMA, F0, 및 loudness 입력을 받아들이는 인코더와 DSP 모듈로 구성됩니다. 인코더는 dilated convolution 네트워크를 사용하여 입력 특성을 처리하며, controlled sine 및 cosine 파형을 생성하는 다층 퍼셉트론(MLP)을 통해 필터 주파수 응답을 제어합니다. H+N(Harmonic-plus-Noise) 모델로 아티큘레이터리 특성을 음성으로 변환합니다. 이 모델은 약 0.4M 파라미터로 SOTA 모델의 9M 파라미터와 동등한 품질을 제공합니다.

- **Performance Highlights**: 제안된 DDSP 보코더는 6.67%의 전사 단어 오류율(Word Error Rate, WER)과 3.74의 평균 의견 점수(Mean Opinion Score, MOS)를 달성하였습니다. 이는 기존 SOTA 모델에 비해 각각 1.63% 및 0.16의 개선입니다. 또한 CPU 추론 시 4.9배 빠른 성능을 보입니다.



### What is lost in Normalization? Exploring Pitfalls in Multilingual ASR Model Evaluations (https://arxiv.org/abs/2409.02449)
Comments:
          Sumbitted to EMNLP 2024

- **What's New**: 이 논문은 다국어 자동 음성 인식 (ASR) 모델 평가의 문제점들을 집중적으로 조사하며, 특히 인디카 언어 스크립트에 중점을 두고 있습니다.

- **Technical Details**: 주요 ASR 모델(OpenAI Whisper, Meta의 MMS, Seamless, Assembly AI의 Conformer)의 텍스트 정규화 절차를 분석하였으며, 철자, 구두점 및 특수 문자와 같은 불일치를 제거하려는 기존의 접근 방식이 인디카 스크립트에 비효율적이라는 점을 강조합니다.

- **Performance Highlights**: 경험적인 분석과 언어학적 검토를 통해, 이러한 결점들이 인디카 언어에 대한 성능 지표를 인위적으로 부풀리게 만든다는 것을 보여주었습니다. 마지막으로, 모국 언어의 전문성을 활용한 새로운 정규화 절차 개발을 제안합니다.



### Detecting Korean Food Using Image using Hierarchical Mod (https://arxiv.org/abs/2409.02448)
- **What's New**: 이번 연구는 한국 음식의 이미지 인식을 위한 새로운 시스템을 제안하여 특히 식이 요건이 있는 외국인들에게 유용하도록 개발되었습니다. 사용자는 요리의 사진을 업로드하기만 하면 어떤 음식을 먹고 있는지 확인할 수 있습니다. 이미지 처리 기술과 머신 러닝(Machine Learning)을 결합하여 이러한 솔루션이 탄생하게 되었습니다.

- **Technical Details**: 이 시스템은 YOLOv8 모델을 기반으로 하며, 층화된 학습 방식을 통해 데이터셋의 클래스 불균형 문제를 해결하고 다양한 한국 음식을 효과적으로 분류합니다. 연구 과정에서 Convolution Neural Network (CNN)를 활용한 다단계 전이 학습(Multi-stage Transfer Learning)과 시각적으로 관련된 음식 아이템의 클러스터링 기법이 적용되었습니다. 또한, 사진의 품질을 높이기 위해 데이터 증대(Data Augmentation) 기술이 사용되었습니다.

- **Performance Highlights**: 이 연구에서는 150개 클래스의 한국 음식을 분류할 수 있는 모델을 개발하였으며, 이전 연구들보다 높은 정확도와 신속한 인식 시간을 달성하였습니다. 개발된 시스템은 사용자 친화적인 인터페이스를 제공하여, 사용자가 자신이 소모한 음식을 쉽게 기록하고 관리할 수 있도록 돕습니다.



### Large Language Models as Efficient Reward Function Searchers for Custom-Environment Multi-Objective Reinforcement Learning (https://arxiv.org/abs/2409.02428)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)을 활용하여 강화 학습(RL)에서의 보상 함수 설계를 가능하게 하였습니다. 이 방법론은 사용자 요구사항에 따른 보상 요소를 생성하고, 각 요소의 가중치를 최적화하는 과정을 통해 RL 알고리즘의 성능을 향상시키는 데 초점을 맞추고 있습니다.

- **Technical Details**: 연구는 LLM을 활용한 화이트 박스 검색(white-box search) 기법을 채택하여 명확한 사용자 요구사항에 따라 보상 코드를 이중 단계로 분리하여 설계합니다. 보상 비평가(reward critic)를 통해 보상 구성 요소를 검증하고, 훈련 로그 분석기를 사용하여 가중치를 최적화하는 데 필요한 데이터를 제공합니다.

- **Performance Highlights**: 이 프레임워크는 인명 감시 없이도 오직 하나의 피드백에 기반하여 보상 코드를 성공적으로 교정하였습니다. 특히, 가중치 초기화 과정을 통해 사용자의 다양한 요구사항을 충족하는 복수의 보상 함수 세트를 확보할 수 있었으며, 100배의 가중치 불일치가 있는 경우라도 평균 4회의 반복만으로 해결책을 도출할 수 있었습니다.



### Accelerating Large Language Model Training with Hybrid GPU-based Compression (https://arxiv.org/abs/2409.02423)
- **What's New**: 이번 연구에서는 Large Language Model (LLM) 훈련을 위해 압축 보조 MPI 집합(PMPI collectives)의 효능을 조사하였습니다. 새로운 하이브리드 압축 스킴인 MZHybrid와 ZHybrid를 제안하였으며, 이를 통해 훈련 속도를 높이면서도 모델 성능을 유지했습니다.

- **Technical Details**: 본 연구는 Data Parallelism (DP), Tensor Parallelism (TP), 그리고 Pipeline Parallelism (PP) 방법론을 활용하여 LLM 훈련의 통신 오버헤드를 감소시키기 위해 GPU 기반 압축 라이브러리와 함께 MPI 라이브러리를 공동 설계하였습니다. 연구진은 192개의 V100 GPU를 사용하는 Lassen 슈퍼컴퓨터에서 실험을 수행했습니다.

- **Performance Highlights**: 실험 결과, 기존의 비압축 집합 통신에 비해 MZHybrid와 ZHybrid를 사용한 경우에 각각 12.7% 및 17.3%의 TFLOPS 증가와 더불어, 초당 처리 샘플 수가 4.4% 증가했습니다.



### Abstractive Text Summarization: State of the Art, Challenges, and Improvements (https://arxiv.org/abs/2409.02413)
Comments:
          9 Tables, 7 Figures

- **What's New**: 이 논문은 추출 요약(extractive summarization) 기법과 대조적으로 추출적 텍스트 요약(abstractive text summarization)의 최신 동향을 심도 있게 조사하고 있으며, 최첨단 기법, 주요 과제 및 연구 방향에 대한 포괄적인 개요를 제공합니다. 또한 모델 복잡성(model complexity)과 확장성(scalability)에 대한 주요 비교 테이블을 제공합니다.

- **Technical Details**: 최신 기법들을 전통적인 Sequence-to-Sequence 모델, Pre-trained Large Language Models(사전 훈련된 대형 언어 모델), Reinforcement Learning(강화 학습), Hierarchical Methods(계층적 방법), Multi-modal Summarization(다중 모달 요약)으로 분류했습니다. 이 논문은 인간처럼 요약을 작성하는 함축적 접근 방식을 탐구하며, 기계 모델의 지식 통합 및 혁신적인 전략들이 주효할 수 있음을 강조했습니다.

- **Performance Highlights**: 이 연구는 의미 표현 부족, 사실 일관성(factual consistency), 제어 가능한 요약 등 여러 도전 과제를 강조하면서, 정보의 정확성과 일관성을 높이기 위한 새로운 접근 방식들을 제안합니다. 또한 이 논문은 다국어 요약(multi-lingual summarization) 및 긴 문서 요약(long-document summarization)과 같은 새로운 연구 분야도 제시하고 있습니다.



### Learning Privacy-Preserving Student Networks via Discriminative-Generative Distillation (https://arxiv.org/abs/2409.02404)
Comments:
          This paper is accepted by IEEE Transactions on Image Processing (TIP)

- **What's New**: 본 논문에서는 개인 정보 유출의 위험을 감수하지 않으면서도 높은 성능 모델을 학습할 수 있는 새로운 접근법인 차별적-생성적 증류(discriminative-generative distillation)를 제안합니다.

- **Technical Details**: 이 연구는 개인 데이터에서 지식을 증류하고, 이를 생성한 합성 데이터로 전송하기 위한 차별적 흐름(discriminative stream)과 생성적 흐름(generative stream)을 사용합니다. 모델은 먼저 개인 정보가 포함된 데이터로 학습한 기본 분류기를 이용하여 나누어진 서브셋에 대한 여러 교사 모델을 구성합니다. 이후, 생성기는 지도 학습 없이 합성 데이터를 생성하고, 이러한 데이터를 변분 오토인코더(Variational Autoencoder, VAE) 학습에 활용합니다.

- **Performance Highlights**: 실험 결과, 제안하는 방법이 개인 정보 보호를 유지하면서도 모델 성능 저하 없이 효과적으로 학습한다는 것을 보여줍니다. 딥 모델의 개인 정보 유출 문제를 해결하면서도 높은 유용성을 확보할 수 있음을 시사합니다.



### Scaling Laws for Economic Productivity: Experimental Evidence in LLM-Assisted Translation (https://arxiv.org/abs/2409.02391)
- **What's New**: 이 논문은 대형 언어 모델(LLM)의 훈련 계산(compute) 양과 성능 간의 경험적 관계인 '스케일링 법칙(scaling laws)'을 경제적 결과에 대해 유도합니다. 300명의 전문 번역가가 1800개의 업무를 수행하는 실험을 통해, 모델의 규모가 생산성을 크게 향상시킬 수 있음을 보여주고 있습니다.

- **Technical Details**: 논문에서는 참가자들이 13개의 서로 다른 LLM을 사용하여 작업을 수행하는 대조군 실험(RCT)을 실시했습니다. 각 번역가는 높은 품질의 작업을 완료할 때마다 보너스를 받는 인센티브 구조가 설계되었습니다. 결과적으로 모델 계산이 10배 증가할 때마다 번역 작업 속도가 12.3% 빨라지고, 품질도 증가하며, 1분당 수익이 16.1% 증가하는 것으로 나타났습니다.

- **Performance Highlights**: 저숙련 작업자는 4배 더 높은 작업 완료 속도 향상을 경험했습니다. 이러한 결과는 향후 LLM의 진화가 임금 불평등에 미칠 수 있는 경제적 의미를 제시하고 있습니다. 현재 LLM의 규모는 매년 4배 증가할 것으로 예상되며, 이는 상당한 경제적 영향을 가져올 것으로 보입니다.



### Neural Dynamics Model of Visual Decision-Making: Learning from Human Experts (https://arxiv.org/abs/2409.02390)
- **What's New**: 이번 연구에서는 시각 입력에서 행동 출력까지 확장된 종합적인 시각 의사결정 모델을 구현하고, 생물학적 신경망의 구조적 특성에 의존하여 CNN(Convolutional Neural Networks)과 유사한 정확도를 성취하였습니다.

- **Technical Details**: 모델은 비인간 영장류의 배외 시각 경로에 기초하여 설계되었으며, 주요 뇌 영역인 LGN(Lateral Geniculate Nucleus), V1(Primary Visual Cortex), MT(Middle Temporal Area), LIP(Lateral Intraparietal Area)를 포함합니다. 뉴런 활동은 Leaky Integrate-and-Fire (LIF) 모델을 사용하여 시뮬레이션되었으며, 심층 학습 모델 대비 생물학적 신경망의 행동 성능을 보다 정확하게 재현합니다.

- **Performance Highlights**: 모델의 선택 확률은 19.31±0.17로, 평균 인간 연구 참가자의 수준인 15.1±1.9를 통계적으로 초과했습니다. 또한, 모델의 결정 시간 곡선은 인간의 반응 시간과 잘 일치하며, 응집력이 증가함에 따라 결정 시간이 점차 짧아지는 경향을 보입니다.



### Multi-modal Situated Reasoning in 3D Scenes (https://arxiv.org/abs/2409.02389)
Comments:
          Project page: this https URL

- **What's New**: 이번 연구에서는 3D 씬(3D scenes)의 상황 이해와 논리를 위한 대규모 다중 모달 데이터셋인 Multi-modal Situated Question Answering (MSQA)를 제안합니다. 이를 통해 기존 데이터셋의 한계를 극복하고, 다양한 실제 3D 씬을 포괄하는 251K의 질문-답변 쌍을 수집하였습니다.

- **Technical Details**: MSQA는 텍스트, 이미지, 포인트 클라우드(point cloud) 등의 다중 모달 입력을 결합한 새로운 데이터 수집 파이프라인을 통해 생성되었습니다. 이 데이터셋은 9개의 질문 카테고리에 걸쳐, 상황과 질문 설명을 제공하는 다중 모달 입력 설정을 도입하여 복잡한 3D 씬 내에서 상황 인식의 유효성을 평가할 수 있도록 합니다.

- **Performance Highlights**: 종합적인 평가 결과, 기존의 비전-언어 모델들이 다중 모달 입력 처리 및 상황 모델링에서 한계를 보임을 드러냈습니다. 실험 결과 MSR3D라는 강력한 기본 모델이 제안되며, MSQA 및 MSNN에 대한 성능이 개선되었습니다. 데이터 스케일링과 교차 도메인 전이 실험을 통해 MSQA에서의 사전 학습(pre-training)이 모델 개발에 있어 효과적임을 보여주었습니다.



### Coral Model Generation from Single Images for Virtual Reality Applications (https://arxiv.org/abs/2409.02376)
Comments:
          In Proceedings of Explainable AI for the Arts Workshop 2024 (XAIxArts 2024) arXiv:2406.14485

- **What's New**: VR(가상 현실) 기술 발전과 함께 고품질 3D 모델의 수요가 급증하고 있는 상황에서, 본 논문은 단일 이미지로부터 고정밀 3D 산호 모델을 생성하는 딥 러닝 프레임워크를 제안합니다.

- **Technical Details**: 이 프레임워크는 Coral 데이터셋을 활용하여 기하학적 및 텍스처적 특징을 추출하고, 3D 재구성을 수행하며, 디자인 및 재료 혼합을 최적화합니다. 고급 최적화 및 폴리곤 수(control)가 Shape 정확성, Detail 유지 및 다양한 복잡도에 대한 유연한 출력을 보장합니다. 또한, Explainable AI(XAI)를 통합하여 AI 생성 모델을 인터랙티브한 '예술작품'으로 변환합니다.

- **Performance Highlights**: 생성된 모델은 기존 방식보다 Detail, Visual quality 및 Efficiency에서 우수하며, VR(가상 현실) 상호작용에서 실시간 피드백으로 산호 종 및 서식지와 같은 정보를 표시하여 사용자 경험을 개선합니다.



### Do Large Language Models Possess Sensitive to Sentiment? (https://arxiv.org/abs/2409.02370)
Comments:
          10 pages, 2 figures

- **What's New**: 최근 대형 언어 모델(LLMs)은 언어 이해에서 뛰어난 능력을 보여주지만, 감정 분석에 대한 포괄적인 평가가 필요합니다. 이 논문은 LLMs의 감정 감지 및 반응 능력을 조사하고, 다양한 응용 프로그램에 통합될 때의 중요성을 강조합니다.

- **Technical Details**: LLMs의 감정 분석 성능을 평가하기 위해 여러 실험을 수행하였고, 긍정적, 부정적, 중립적 감정을 구분하고 적절히 반응하는 능력을 비교했습니다. 'Sentiment Knowledge Workflow'를 개발하고 LLMs의 감정 민감성을 평가하여 훈련 과정 개선이 필요하다는 점을 발견했습니다. 또한, 다양한 아키텍처와 데이터셋에 따라 서로 다른 성능을 나타냈습니다.

- **Performance Highlights**: LLMs는 기본적으로 감정에 대한 민감성을 가지고 있지만, 정확도와 일관성에 있어 상당한 차이가 있었습니다. 예를 들어, 강한 긍정적 감정을 중립으로 잘못 분류하거나 풍자나 아이러니를 인식하지 못하는 사례가 있었습니다. 이러한 발견은 감정 분석의 복잡성을 강조하며, LLMs의 감정 인식을 향상시키기 위한 추가적인 연구가 필요함을 보여줍니다.



### NUDGE: Lightweight Non-Parametric Fine-Tuning of Embeddings for Retrieva (https://arxiv.org/abs/2409.02343)
- **What's New**: 이 논문에서는 NUDGE라는 새로운 비모수(non-parametric) 임베딩 조정 방법을 소개합니다. 이 방법은 기존의 사전 훈련(pre-trained) 모델과 비교하여 더 높은 정확도와 효율성을 제공합니다.

- **Technical Details**: NUDGE는 k-최근접 이웃 검색(k-NN retrieval)에 최적화된 데이터 레코드의 임베딩을 직접 변경하여 정확도를 극대화합니다. 이 방법은 실제로 NP-Hard 문제로 알려져 있으나, 제한된 변형을 통해 효율적으로 해결할 수 있습니다. 개선된 임베딩 변화는 사전 훈련 과정에서 학습된 의미를 왜곡하지 않도록 제한됩니다.

- **Performance Highlights**: 실험을 통해 NUDGE는 9개의 표준 텍스트 및 이미지 검색 데이터셋에서 기존 조정 방법들보다 NDCG@10에서 평균 10% 더 개선된 성능을 보여줍니다. NUDGE 방법은 시간당 200배 빠르게 실행되며, 평균적으로 정확도는 3.3배에서 4.3배 증가합니다.



### Coaching a Robotic Sonographer: Learning Robotic Ultrasound with Sparse Expert's Feedback (https://arxiv.org/abs/2409.02337)
Comments:
          Accepted in IEEE Transactions on Medical Robotics and Bionics (TMRB) 2024

- **What's New**: 이 논문은 로봇 초음파(Robotic Ultrasound, RUS)의 성능을 향상시키기 위해 코칭(coaching) 프레임워크를 제안합니다. 기존의 LfD(learning from demonstrations) 방법을 넘어, 실시간 전문가의 피드백을 통합하여 RUS의 훈련 과정에서 인간 전문가의 적극적인 참여를 이끌어내고자 합니다.

- **Technical Details**: 제안된 방법은 Deep Reinforcement Learning (DRL)과 전문가의 드물게 제공되는 피드백을 결합합니다. DRL은 이미지 품질 평점을 기반으로 한 보상을 사용하며, 전문가의 코칭은 Partially Observable Markov Decision Process (POMDP)로 모델링되어 정책 파라미터를 업데이트합니다.

- **Performance Highlights**: 검증 연구 결과, 코칭을 통해 학습 속도가 25% 증가하고 질 높은 이미지 획득 수는 74.5% 증가하는 것으로 나타났습니다.



### Arctic-SnowCoder: Demystifying High-Quality Data in Code Pretraining (https://arxiv.org/abs/2409.02326)
- **What's New**: 최근 언어 모델의 효과적인 사전 학습에 있어 고품질 데이터의 중요성이 강조되고 있습니다. 이 논문에서는 Arctic-SnowCoder-1.3B라는 새로운 데이터 효율적인 코드 모델을 소개합니다.

- **Technical Details**: Arctic-SnowCoder-1.3B는 555B 토큰을 기반으로 세 단계의 데이터를 통해 사전 학습되었으며, 각 단계에서 품질이 점진적으로 개선됩니다. 첫 번째 단계에서는 500B 표준 품질의 코드 토큰을 일반 사전 학습을 통해 사용하고, 두 번째 단계에서는 BERT 스타일의 품질 주석기를 통해 선택된 50B 고품질 토큰을 사용합니다. 마지막으로, 세 번째 단계에서는 Llama-3.1-70B에 의해 생성된 5B의 합성 데이터를 이용한 향상된 사전 학습을 진행합니다.

- **Performance Highlights**: Arctic-SnowCoder-1.3B는 BigCodeBench에서 최신 성능을 기록했으며, 1T 토큰 이하로 학습된 유사한 규모의 모델들을 능가합니다. 특히, Phi-1.5-1.3B에 비해 36%의 성능 향상을 보여주었으며, HumanEval+ 벤치마크에서도 StarCoder2-3B를 초월하며 경쟁력을 유지하고 있습니다.



### TimeDiT: General-purpose Diffusion Transformers for Time Series Foundation Mod (https://arxiv.org/abs/2409.02322)
Comments:
          23 Pages, 6 Figures, 11 Tables. First present at ICML 2024 Workshop on Foundation Models in the Wild

- **What's New**: 본 논문에서는 Time Diffusion Transformer (TimeDiT)라는 혁신적인 시간 시계열 기초 모델을 소개합니다. 이 모델은 기존의 시간 시계열 모델들이 직면한 문제들을 해결하기 위해 denoising diffusion paradigm을 채택하여, 다양한 도메인의 시계열 데이터를 효과적으로 처리할 수 있도록 설계되었습니다.

- **Technical Details**: TimeDiT는 Transformer 아키텍처를 기반으로 하여 시계열 데이터의 시간적 종속성을 포착하고, novel masking schemes를 통해 대상 분포에 대한 엄격한 가정을 부여하지 않은 채 고품질 샘플을 생성합니다. 또한, 외부 지식을 샘플링 과정에서 통합할 수 있는 finetuning-free model editing 전략을 제안합니다. 이 모델은 다양한 입력 형태에 대해 표준화된 훈련 파이프라인을 통해 운영됩니다.

- **Performance Highlights**: TimeDiT는 20개 이상의 다양한 데이터셋에서 실험을 수행하였으며, 전통적인 선형 모델부터 diffusion 모델, Transformer 기반 모델까지 25개 이상의 오픈 소스 기준 모델에 대해 성능을 평가하였습니다. 이 모델은 몇몇 태스크에서 state-of-the-art 성능을 보였으며, 특히 Electricity와 Traffic 데이터셋에서 확률적 예측을 위한 CRPSsum 점수에서 새로운 최첨단 기록을 달성했습니다.



### On the Benefits of Memory for Modeling Time-Dependent PDEs (https://arxiv.org/abs/2409.02313)
- **What's New**: 이 논문에서는 전통적인 수치 방법의 대안으로 데이터 기반 기술을 사용하는 새로운 접근법을 제안합니다. 특히, 과거 상태를 이용하여 미래를 예측하는 메모리 아키텍처를 활용하여 비선형 방정식(Partial Differential Equations, PDEs)의 모델링을 개선하고자 합니다.

- **Technical Details**: 본 연구에서는 메모리 신경 연산자(Memory Neural Operator, MemNO)를 도입하였으며, 최근의 상태 공간 모델(State Space Model, SSM) 아키텍처와 푸리에 신경 연산자(Fourier Neural Operator, FNO)를 기반으로 합니다. 이 메모리 아키텍처는 과거의 상태 정보를 명시적으로 사용하여 시스템의 진화를 보다 정확하게 예측할 수 있도록 합니다.

- **Performance Highlights**: 여러 PDE 계열에 대한 실험을 통해, MemNO는 메모리가 없는 기존 모델에 비해 6배 이상의 오류 감소를 달성하며, 고주파 푸리에 성분을 포함하는 PDE 해결에 특히 효과적임을 보여주었습니다. 또한, 관측 잡음에 대한 강인성이 증가함을 발견하였습니다.



### Speech Foundation Model Ensembles for the Controlled Singing Voice Deepfake Detection (CtrSVDD) Challenge 2024 (https://arxiv.org/abs/2409.02302)
Comments:
          Accepted to the IEEE Spoken Language Technology Workshop (SLT) 2024. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 이 연구는 Controlled Singing Voice Deepfake Detection (CtrSVDD)의 평가 세트에서 1.79%의 pooled equal error rate (EER)를 달성하기 위한 접근 방법을 상세히 설명합니다. 특히, 음성 기반 모델을 활용한 앙상블 방법과 새로운 Squeeze-and-Excitation Aggregation (SEA) 방법을 소개합니다.

- **Technical Details**: 본 연구에서는 Deepfake 노래 목소리 탐지를 위해 RawBoost 증강 기법을 사용하고, LnL convolutional noise와 ISD additive noise의 병렬 및 순차적 추가 접근 방식을 적용하여 노이즈 특성을 조합합니다. 또한, RawNet2 스타일의 SincConv 레이어와 wav2vec2 및 WavLM 모델을 활용하여 음성 특징을 효과적으로 캡처합니다.

- **Performance Highlights**: 우리의 최적 모델은 CtrSVDD 평가 세트에서 2.70% EER를 달성하며, 앙상블 시스템을 통해 1.79% EER에 도달하는 성과를 입증합니다. 이러한 접근 방식은 Deepfake 노래 음성 탐지의 성능을 크게 향상시킵니다.



### Biochemical Prostate Cancer Recurrence Prediction: Thinking Fast & Slow (https://arxiv.org/abs/2409.02284)
Comments:
          8 pages, 3 figures, methodology paper for LEOPRARD Challenge

- **What's New**: 이 논문에서는 전립선암의 생화학적 재발 예측을 위한 새로운 접근법을 제안합니다. 이는 'thinking fast & slow' 전략을 활용한 두 단계의 multiple instance learning (MIL) 방법론을 활용하여 재발 예측 작업을 수행합니다.

- **Technical Details**: 제안된 방법은 먼저 low WSI 해상도에서 가장 중요한 영역을 식별하는 1st 단계('thinking fast')와 고해상도 패치를 활용해 TTR(time to recurrence)을 예측하는 2nd 단계('thinking slow')로 구성됩니다. 이 방식은 전체 이미지의 패치를 기반으로 하여 MC(Mean C-index) 측정에서 0.733을 기록하였습니다.

- **Performance Highlights**: 최종 모델은 LEOPARD 챌린지 검증 세트에서 C-index가 0.603을 기록하며 생화학적 재발 예측에서 기존 방법론에 비해 개선된 성능을 보여주었습니다.



### Reinforcement Learning-enabled Satellite Constellation Reconfiguration and Retasking for Mission-Critical Applications (https://arxiv.org/abs/2409.02270)
Comments:
          Accepted for publication in the IEEE Military Communications Conference (IEEE MILCOM 2024)

- **What's New**: 본 연구는 위성 군집이 고장나는 경우 재구성과 재과제가 수행되는 방안을 다루며, GPS 위성 군집에 대한 시스템 모델링을 통해 성능 분석을 실시합니다.

- **Technical Details**: 위성 고장 시 시스템 모델링 접근 방식을 통해 성능 역학성과 과제 분배 전략을 연구합니다. 딥러닝 방법론 중 Q-learning, Policy Gradient, DQN, PPO를 적용하여 위성 군집의 재구성과 재과제를 관리하는 기법을 개발했습니다.

- **Performance Highlights**: DQN과 PPO 기법을 사용한 결과 평균 보상, 과제 완료율, 응답 시간 모두에서 효과적인 결과를 나타냈습니다. 이를 통해 위성 장애 발생시 지속적인 군집 운영을 보장하는 데 기여할 수 있습니다.



### Action-Based ADHD Diagnosis in Video (https://arxiv.org/abs/2409.02261)
Comments:
          31st European Symposium on Artificial Neural Networks

- **What's New**: 이번 연구에서는 ADHD 진단 과정에서 비디오 기반의 프레임 수준 행동 인식 네트워크를 처음으로 도입하였으며, 실제 멀티모달 ADHD 데이터셋을 기록하고 비디오 모달리티에서 세 가지 행동 클래스를 추출하였습니다.

- **Technical Details**: 원래의 C3D 구조와는 차별화되어, 입력 데이터 크기에 맞추기 위해 완전 연결 계층이 추가된 3D-CNN 구조를 구현했습니다. 이 시스템은 데이터 처리, 행동 인식, 정지 비율 계산, ADHD 진단의 네 가지 주요 구성 요소를 포함하고 있습니다.

- **Performance Highlights**: 17명의 참가자에 대한 평균 SR (Stationary Ratio) 값은 0.71로 나타났으며, 7명의 ADHD 환자는 0.50, 10명의 일반인(controls)은 0.86의 평균 SR 값을 보였습니다.



### NoiseAttack: An Evasive Sample-Specific Multi-Targeted Backdoor Attack Through White Gaussian Nois (https://arxiv.org/abs/2409.02251)
- **What's New**: NoiseAttack는 단일 희생자 클래스 대신 여러 목표 클래스를 생성할 수 있는 샘플-특정 다중-targeted backdoor attack을 소개합니다. 이는 White Gaussian Noise (WGN)를 트리거 패턴으로 사용하여 모델의 훈련 단계에서 은밀하게 백도어를 삽입합니다.

- **Technical Details**: NoiseAttack에서는 White Gaussian Noise (WGN)의 다양한 Power Spectral Densities (PSD)를 트리거로 사용합니다. 이 방법은 모델의 훈련 과정에서 모든 입력 샘플에 WGN을 삽입하여, 목표 클래스에 대해 악의적으로 동작하도록 할 수 있습니다.

- **Performance Highlights**: NoiseAttack은 인기 있는 네트워크 아키텍처와 데이터셋에 대해 높은 공격 성공률을 달성하며, 최신 백도어 탐지 방법을 우회할 수 있음을 실험적으로 입증하였습니다.



### FastVoiceGrad: One-step Diffusion-Based Voice Conversion with Adversarial Conditional Diffusion Distillation (https://arxiv.org/abs/2409.02245)
Comments:
          Accepted to Interspeech 2024. Project page: this https URL

- **What's New**: FastVoiceGrad는 다단계 확산 기반 음성 변환(VC) 기술의 단점을 극복하기 위해 고안된 참신한 일단계 VC 모델입니다. 이 모델은 수십 번의 반복을 단 한 번으로 줄이면서도 높은 VC 성능을 유지합니다.

- **Technical Details**: FastVoiceGrad는 적대적 조건부 확산 증류(ACDD) 기법을 사용하여 다단계 교사 모델인 VoiceGrad에서 일단계 학생 모델로 지식을 전이합니다. 이를 통해 생성적 적대 신경망(GAN)과 확산 모델의 장점을 모두 활용합니다.

- **Performance Highlights**: FastVoiceGrad는 일회성 모든-대-모든 VC에서 실험적으로 VoiceGrad와 동등하거나 우수한 성능을 보였고, DiffVC와도 비교할 만한 성능을 유지하면서 추론 속도를 개선하였습니다.



### Temporal Order Preserved Optimal Transport-based Cross-modal Knowledge Transfer Learning for ASR (https://arxiv.org/abs/2409.02239)
Comments:
          Accepted to IEEE SLT 2024

- **What's New**: 이 논문에서는 Temporal Order Preserved OT (TOT)를 기반으로 한 Cross-modal Alignment and Knowledge Transfer (CAKT) 모델(TOT-CAKT)을 제안합니다. 이 모델은 음향(sequence)과 언어(sequence) 간의 지식을 효율적으로 전이할 수 있도록 설계되었습니다.

- **Technical Details**: TOT-CAKT 모델은 음향 시퀀스의 이웃 프레임을 언어 시퀀스의 이웃 지역으로 매핑하여 temporal order 관계를 유지합니다. 이 과정에서 pretrained Chinese PLM(BERT 사용)을 통해 Mandarin ASR 실험을 수행하였습니다. 이 과정에서의 특징적인 기법으로는 'Adapter' 모듈과 TOT 기반의 cross-modal matching 모듈이 포함됩니다.

- **Performance Highlights**: TOT-CAKT 모델은 여러 최신 모델들과 비교했을 때 ASR 성능에서 상당한 개선을 보여주었습니다. 특히, OT 기반 방법의 약점을 해결하며, 음향 및 언어 특성 간의 순차적인 정렬을 강화하였습니다.



### A+AI: Threats to Society, Remedies, and Governanc (https://arxiv.org/abs/2409.02219)
Comments:
          25 pages

- **What's New**: 이 문서는 인공지능(AI)이 사회에 가져오는 단기적인 위협에 초점을 맞추고 있습니다. AI뿐만 아니라 알고리즘 프로세스에서 발생할 수 있는 여러 위협을 다루며, 'A+AI'라는 개념을 통해 알고리즘과 인공지능을 통합적으로 이해해야 한다고 강조합니다.

- **Technical Details**: 위협을 완화하기 위한 대책을 논의하며, 이를 위한 표를 포함하고 있습니다. 정부는 모든 소셜 미디어 플랫폼이 사용자 계정 소유를 확인하도록 요구하고, A+AI로 생성되거나 수정된 제품에 대한 명확한 라벨링을 요구해야 합니다.

- **Performance Highlights**: 생각 있는 거버넌스(thoughtful governance)는 사회적 혼란을 줄여 기술 발전을 가속화할 수 있으며, 정부는 위협 완화를 위한 긴급 연구 프로젝트를 지원하고 위협 인식을 높이기 위한 교육 캠페인을 마련해야 합니다.



### Fair Railway Network Design (https://arxiv.org/abs/2409.02152)
Comments:
          32 pages, 18 figures

- **What's New**: 이 논문에서는 국가의 대중 교통 네트워크 설계 시 이동 시간을 최소화하는 방안을 제시합니다. 이는 공정성(fairness) 감안을 배제한 순수한 공리주의(utilitarian) 접근을 다루고 있습니다.

- **Technical Details**: 모델을 정의하고 솔루션 네트워크를 계산하기 위한 알고리즘을 제안했습니다. 실 데이터를 기반으로 한 실험(results based on real data)을 통해 알고리즘의 성능을 평가합니다.

- **Performance Highlights**: 이 연구는 대도시 중심으로 혜택을 주는 기존 모델과는 달리, 주변 도시 간의 이동이 가능하도록 하는 보다 평등한 접근 방식을 탐구합니다.



### Optimal Power Grid Operations with Foundation Models (https://arxiv.org/abs/2409.02148)
- **What's New**: 본 논문은 기후 위기를 극복하기 위한 에너지 전환의 일환으로, 분산된 재생 에너지원들이 기존의 전력망에 통합되어야 한다는 점을 강조하고 있다. 이를 위해 AI Foundation Models (FMs)와 Graph Neural Networks (GNNs)를 활용하여 전력망 운영을 향상시킬 수 있는 방법을 제안하고 있다.

- **Technical Details**: 에너지 전환 과정에서 전력망의 운용과 계획이 복잡성과 불확실성으로 인해 도전받고 있다. 논문에서는 자가 지도 학습(self-supervised learning)을 통해 전력 흐름의 동역학을 학습하는 모델을 구축하여 전력망의 물리학을 캡처하고자 한다. AI FMs는 대규모 데이터셋에 대해 사전 훈련(pre-trained)되어, 소규모 라벨링된 데이터셋에서 미세 조정(fine-tuning)될 수 있다. 이를 통해 다양한 전력망 토폴로지에 효율적으로 적응할 수 있게 된다.

- **Performance Highlights**: FMs를 활용한 전력 흐름 문제의 해결은 현재 전력망 분석 능력과 산업의 필요 사이의 간극을 줄일 수 있으며, 이는 최적 전력망 운영 및 계획에 더 가까워지는 데 기여할 것으로 보인다. 특히, 전력 흐름 해결을 통해 4-5 배의 성능 향상이 가능하다는 점이 강조된다.



### A Multimodal Object-level Contrast Learning Method for Cancer Survival Risk Prediction (https://arxiv.org/abs/2409.02145)
- **What's New**: 본 논문에서는 다중 모달 생존 위험 예측을 위한 새로운 훈련 방법인 multimodal object-level contrast learning을 제안합니다. 이 방법은 생존 위험 관계를 기반으로 학습 쌍을 구성하고, 이를 통해 생존 위험 예측기를 훈련시키는 과정을 포함합니다.

- **Technical Details**: 제안된 방법은 object-level contrast (OC) 학습을 활용하여 각각의 샘플이 다른 샘플의 예측을 관찰함으로써 학습할 수 있도록 합니다. 이를 통해 fine-grained concordance를 모델링하며, cross-modal contrast learning을 통해 다중 모달 시나리오로 확장됩니다. 또한, attention 기반 신경망과 self-normalizing 신경망을 사용하여 병리학적 이미지와 유전체 데이터를 처리하여 다중 모달 생존 위험 예측기를 구성합니다.

- **Performance Highlights**: 제안된 방법으로 훈련된 다중 모달 생존 위험 예측기(MSRP)는 두 개의 공개된 다중 모달 암 생존 예측 데이터셋에서 최신 기술보다 우수한 성능을 보였습니다.



### Efficient and Scalable Estimation of Tool Representations in Vector Spac (https://arxiv.org/abs/2409.02141)
- **What's New**: 최근 대규모 언어 모델(LLM)의 기능 호출 및 도구 사용에서의 발전은 외부 정보 소스와의 상호작용 및 복잡한 작업 실행을 가능하게 하여 모델의 능력을 크게 향상시켰습니다. 그러나 많은 도구가 사용가능할 때 LLM의 제한된 컨텍스트 윈도우는 도전 과제를 제시합니다.

- **Technical Details**: 본 논문에서는 도구 검색 응용 프로그램을 위한 합성 데이터 생성 프레임워크와 작은 인코더 모델을 이용한 데이터 기반의 도구 검색 전략을 제안합니다. 특히 Tool2Vec(사용 기반 도구 임베딩 생성), ToolRefiner(단계적 검색 방법), MLC(다중 레이블 분류 문제로 도구 검색 프레임 설정)와 같은 새로운 접근 방식을 소개합니다.

- **Performance Highlights**: 이 새로운 방법을 통해 ToolBench 데이터셋에서 Recall@K가 27.28까지 향상되었고, ToolBank에서는 30.5의 개선을 달성했습니다. 추가 실험 결과를 통해 우리의 방법의 타당성을 엄격히 검증합니다.



### Self-Supervised Learning for Identifying Defects in Sewer Footag (https://arxiv.org/abs/2409.02140)
Comments:
          Poster at the LatinX in AI Workshop @ ICML 2024

- **What's New**: 이 연구는 수도관 검사에 Self-Supervised Learning (SSL)을 처음으로 적용하였습니다. 기존의 수작업 검사에 의존하는 방법 대신, 적은 양의 라벨이 지정된 데이터로도 경쟁력 있는 성능을 달성하는 자동화된 솔루션을 제안합니다.

- **Technical Details**: DINO 방법론을 사용하여 1.3백만 개의 이미지와 17개 결함 유형이 포함된 Sewer-ML 데이터셋에서 평가를 진행하였습니다. 모델은 기존 방법에 비해 최소 5배 작으며, 10%의 데이터로도 강력한 결과(50.05 F2CIW, 87.45 F1Normal)를 나타냅니다. SSL 방법은 라벨링 데이터 수집의 어려움을 해결할 수 있는 혁신적인 접근 방식으로 자리 잡고 있습니다.

- **Performance Highlights**: 기존의 첨단 방법들과 경쟁력 있는 성능을 보임에도 불구하고, 훨씬 더 작은 모델을 성공적으로 훈련하여 작은 장치에서 실시간 탐지에 적합합니다. 이는 자원이 제한된 환경에서도 확장 가능성을 높입니다.



### The Role of Transformer Models in Advancing Blockchain Technology: A Systematic Review (https://arxiv.org/abs/2409.02139)
- **What's New**: 본 논문은 블록체인 기술의 발전에 따라 Transformer 모델이 블록체인 애플리케이션에 미치는 잠재적인 영향에 대한 포괄적인 문헌 조사를 수행하였습니다. 200개 이상의 관련 논문을 검토하여 각 분야에서의 Transformer의 활용 사례를 정리하였으며, 블록체인 내의 다양한 도전 과제를 해결하기 위한 Transformer의 응용 가능성을 강조합니다.

- **Technical Details**: 논문은 Transformer 모델의 기본 원리, 아키텍처 특징, 이러한 모델이 블록체인 데이터 처리에 효과적인 이유를 설명합니다. 특히, anomaly detection, 스마트 계약의 보안 분석, 암호화폐 예측 및 트렌드 분석, 코드 요약 생성 등의 분야에서 Transformer의 구체적인 응용을 분석합니다. Transformer의 self-attention 메커니즘은 블록체인 데이터의 복잡한 패턴 분석과 문제 해결에 큰 잠재력을 지니고 있습니다.

- **Performance Highlights**: Transformer는 블록체인 기술에서 데이터의 연관성 분석 및 고차원 거래 데이터의 복잡한 패턴을 인식하는 데 매우 유용합니다. 특히, 거래 모니터링 및 이상 탐지 분야에서 기존의 전통적 방법보다 뛰어난 성과를 보입니다. 또한, 스마트 계약의 자동화된 감사 및 보안 분석에서도 promising한 가능성을 보여주며, 이를 통해 블록체인 기술 발전에 기여할 수 있는 방향성을 제시하고 있습니다.



### A Financial Time Series Denoiser Based on Diffusion Mod (https://arxiv.org/abs/2409.02138)
- **What's New**: 본 연구는 금융 시계열 데이터에서 신호 대 잡음 비율(SNR)의 문제를 해결하기 위해 **conditional diffusion model**을 활용하는 새로운 접근 방식을 제안합니다. 이 모델은 노이즈를 점진적으로 추가하고 제거하여 원본 데이터를 복원하는 방법으로, **trading performance** 향상에 기여합니다.

- **Technical Details**: 제안된 방법은 **conditional diffusion model**의 전방향 및 역방향 프로세스를 활용하여 노이즈가 섞인 데이터를 개선합니다. 연구에서는 다양한 **downstream future return classification tasks**에 대한 실험을 통해 노이즈가 제거된 시계열 데이터가 더 나은 성능을 보여주었음을 입증합니다.

- **Performance Highlights**: Denoised 데이터로부터 유도된 **trading signals**는 더 높은 수익과 적은 거래 횟수를 기록하며, 이는 **transaction costs**를 최소화하고 전반적인 **trading efficiency**를 증가시킵니다. 또한, 노이즈 상태를 인식함으로써 초과 수익을 얻을 수 있는 새로운 거래 전략을 제안합니다.



### Large Language Models versus Classical Machine Learning: Performance in COVID-19 Mortality Prediction Using High-Dimensional Tabular Data (https://arxiv.org/abs/2409.02136)
Comments:
          Code is available at: this https URL and this https URL. The datasets are available from the corresponding author on reasonable request (sdamirsa@ymail.com)

- **What's New**: 이 연구는 COVID-19와 관련된 사망률 예측에서 고차원 테이블 데이터셋을 활용해 전통적인 머신러닝 모델(클래식 머신러닝 모델, CML)과 대형 언어 모델(LLM)의 성능을 비교하고 평가했습니다.

- **Technical Details**: 9,134명의 COVID-19 환자 데이터를 활용하였으며, XGBoost와 랜덤 포레스트(RF)를 포함한 7개의 CML 모델이 학습 및 평가되었습니다. 구조화된 데이터는 텍스트로 변환되어 GPT-4와 Mistral-7b를 포함한 8개의 LLM에 의해 제로샷 분류(zero-shot classification)에 사용되었습니다. Mistral-7b는 QLoRA 접근 방식을 통해 파인튜닝(fine-tuning)되었습니다.

- **Performance Highlights**: CML 모델 중 XGBoost와 RF가 내부 검증에서 F1 점수 0.87, 외부 검증에서 0.83으로 가장 높은 정확도를 기록하였습니다. LLM 중에선 GPT-4가 F1 점수 0.43으로 최고 성능을 나타냈습니다. Mistral-7b의 파인튜닝은 리콜(recall)을 1%에서 79%로 크게 향상시켰으며, F1 점수는 0.74로 외부 검증 동안 안정적으로 유지되었습니다.



### Edge AI: Evaluation of Model Compression Techniques for Convolutional Neural Networks (https://arxiv.org/abs/2409.02134)
- **What's New**: 이 연구는 이미지 분류 작업에서 ConvNeXt 모델의 압축 기술을 평가합니다. CIFAR-10 데이터셋을 사용하여 구조적 가지치기(Structured Pruning), 비구조적 가지치기(Unstructured Pruning), 동적 양자화(Dynamic Quantization) 방법을 통해 모델 크기와 계산 복잡성을 줄이면서 정확도를 유지하는 방법을 제시합니다.

- **Technical Details**: 실험은 클라우드 기반 플랫폼과 엣지 디바이스에서 수행되었으며, 이 압축 기술들의 성능을 평가했습니다. 결과적으로 구조적 가지치기 기술을 통해 최대 75%의 모델 크기 감소가 가능하였고, 동적 양자화는 파라미터 수에서 최대 95%의 감소를 달성하였습니다. 또한, 사전 훈련(Pre-training)과 압축 기법을 결합 시킨 모델들은 더 개선된 압축 성능을 보였습니다.

- **Performance Highlights**: 최종 압축 모델을 엣지 디바이스에 배포한 결과, 92.5%의 높은 정확도와 20 ms의 낮은 추론 시간을 기록하여 실제 엣지 컴퓨팅 애플리케이션에서 압축 기술의 효과성을 검증하였습니다.



### From Predictive Importance to Causality: Which Machine Learning Model Reflects Reality? (https://arxiv.org/abs/2409.02130)
- **What's New**: 이 연구는 CatBoost와 LightGBM 모델을 사용하여 Ames Housing Dataset을 분석하고 주택 가격 예측에서의 정확도와 인과 관계를 탐구합니다. 주택 시장 분석에서 예측 모델링과 인과 이해를 정렬하는 것이 복잡하다는 점을 강조합니다.

- **Technical Details**: 연구는 SHAP (SHapley Additive exPlanations) 값과 EconML 예측의 상관관계를 조사하였으며, moderate한 Spearman rank correlation 0.48을 달성하였습니다. CatBoost와 LightGBM을 활용하여 79개의 다양한 특성의 중요성을 평가했습니다.

- **Performance Highlights**: 연구 결과, 특정 특성(예: porches)이 다양한 시나리오에서 주택 가격에 미치는 영향을 탐구하고, 예측 파워와 인과 통찰력이 결합된 통합 접근 방법의 필요성을 강조했습니다.



### The Application of Artificial Neural Network Model to Predicting the Acid Mine Drainage from Long-Term Lab Scale Kinetic Tes (https://arxiv.org/abs/2409.02128)
Comments:
          The 7th Environmental Technology and Management Conference (ETMC 2023)

- **What's New**: 이 연구는 인공신경망(Artificial Neural Network, ANN)을 활용하여 실험실 규모의 동역학 테스트(lab-scale kinetic tests)에서 산 생성의 예측 성능을 향상시키는 새로운 접근 방식을 제시합니다.

- **Technical Details**: 연구에서는 총 83주간의 실험과정을 기반으로 한 ANN 모델을 통해 pH, 산화환원전위(ORP), 전도도(conductivity), 총 용해 고형물(TDS), 황산염(SO4), 중금속(철, 망간)의 변화를 모니터링 합니다. 사용된 주요 ANN 아키텍처는 피드포워드 신경망(feedforward neural network, FNN)과 LSTM(Long Short-Term Memory) 모델입니다. 데이터의 정합성을 평가하기 위해 평균제곱오차(Mean Squared Error, MSE)와 내쉬-서클리프 효율성(Nash-Sutcliffe Efficiency, NSE) 지표가 사용되었습니다.

- **Performance Highlights**: 이 연구에서 얻은 내쉬-서클리프 효율성(NSE)은 0.99로, 실제 실험실 동역학 테스트 데이터와의 강한 상관관계를 보여줍니다. 이는 ANN이 과거 데이터를 통해 패턴과 트렌드를 학습하고 정확한 예측을 가능하게 함을 나타냅니다.



### Enabling Trustworthy Federated Learning in Industrial IoT: Bridging the Gap Between Interpretability and Robustness (https://arxiv.org/abs/2409.02127)
Comments:
          7 pages, 2 figures

- **What's New**: 이 논문에서는 Federated Learning (FL)과 Industrial Internet of Things (IIoT)을 통합하여 해석 가능성과 강인성 간의 균형을 맞추는 방법을 제안합니다. 또한, IIoT 환경에서 신뢰할 수 있는 FL 시스템의 설계 방법론을 제시하고, 실질적인 안전 및 경제적 영향을 고려한 투명하고 신뢰할 수 있는 시스템의 중요성을 강조합니다.

- **Technical Details**: FL은 초기 데이터 저장소 없이 다양한 데이터 출처에서 학습할 수 있는 독창적인 방법을 제공합니다. 이를 통해 데이터 기밀성과 보안이 강화되며, 자원 효율적인 관리가 가능합니다. FL에서 해석 가능성을 높이기 위한 Layer-wise relevance propagation (LRP) 기술과 adversarial training과 같은 방법들이 사용됩니다. 또한, Byzantine-resistant aggregation 알고리즘은 공격에 대한 저항력을 높이기 위해 중요합니다.

- **Performance Highlights**: IIoT 환경에서 FL기술을 활용하여 얻어진 케이스 스터디들이 제시되며, 이 기술을 통해 데이터 전송의 효율성과 모델 정확성을 향상시키는 한편, 사이버 공격에 대한 강인성을 높일 수 있음을 보여주고 있습니다. 논문은 FL의 실행 가능성과 IIoT의 복잡한 요구 사항에 응답하는 실제적이고 신뢰할 수 있는 모델을 제안합니다.



### TrajWeaver: Trajectory Recovery with State Propagation Diffusion Mod (https://arxiv.org/abs/2409.02124)
Comments:
          First submission, extended to 10 pages include ref

- **What's New**: 본 논문에서는 TrajWeaver라는 새로운 경로 복원 프레임워크를 제안합니다. 이 프레임워크는 확률적 확산 모델을 기반으로 하여 다양한 부가 기능에 따라 희소한 원시 경로에서 밀집되고 정제된 경로를 복원할 수 있습니다.

- **Technical Details**: TrajWeaver는 State Propagation Diffusion Model (SPDM)라는 새로운 상태 전파 메커니즘을 도입하여 표준 확산 모델 위에 구축되었습니다. 이 메커니즘은 이전 확산 단계에서 계산된 지식을 나중에 재사용할 수 있게 하여 복원 성능을 개선합니다. 이 구조는 다양한 조건을 효과적으로 집결하고 융합하여 다단계 복원 과정에서의 요소 간의 지식 공유를 가능하게 합니다.

- **Performance Highlights**: TrajWeaver는 다양한 길이, 희소성 수준 및 이질적인 이동 방식을 가진 원시 경로로부터 복원할 수 있는 능력을 검증했습니다. 실험 결과, 기존의 최첨단 방법들과 비교할 때 복원 정확도가 현저하게 개선되었습니다.



### PuYun: Medium-Range Global Weather Forecasting Using Large Kernel Attention Convolutional Networks (https://arxiv.org/abs/2409.02123)
- **What's New**: PuYun 모델은 대규모 커널 주의(convolutional networks) 기반의 자율 회귀(cascade model)로, 날씨 예측 효율성을 개선하고 예측 정확도를 높였습니다. 특히 0-5일과 5-10일 예측을 각각 위한 PuYun-Short와 PuYun-Medium이 도입되었습니다.

- **Technical Details**: PuYun은 LKA (Large Kernel Attention) 기반의 FCN (Fully Convolutional Network) 구조를 사용합니다. 입력 데이터는 탄력적으로 패치 임베딩(patch embedding)을 사용하여 2×69×721×1440 텐서로 변환된 후, LKA-FCN 층을 통해 처리가 이루어집니다.

- **Performance Highlights**: PuYun-Short는 10일 예측에서 GraphCast와 FuXi-Short를 초과하는 성능을 달성하였으며, 10일 차에는 Z500 RMSE를 720 $m^2/s^2$로 줄여 GraphCast의 732 $m^2/s^2$ 및 FuXi-Short의 740 $m^2/s^2$에 비해 개선된 결과를 보였습니다.



### Deep Knowledge-Infusion For Explainable Depression Detection (https://arxiv.org/abs/2409.02122)
Comments:
          13 pages, 2 figures

- **What's New**: 이번 연구는 DepressionFeature ontology (DFO)와 Commonsense Transformer (COMET)로부터 도메인 전문 지식을 결합한 Knowledge-infused Neural Network (KiNN)을 제안합니다. 이 모델은 사용자가 이해할 수 있는 설명력을 제공하여 정신 건강 전문가(MHPs)들이 우울증을 감지할 수 있도록 돕습니다.

- **Technical Details**: KiNN 모델은 딥러닝 기술을 활용하여 사용자 수준의 설명 가능한 방식으로 우울증을 감지합니다. 이는 전문가가 이해하는 개념 및 프로세스를 포함하며, 문맥 정보를 고려하여 더 나은 성능을 보여줍니다. 모델은 CLEF e-Risk와 PRIMATE 데이터셋에서 MentalBERT와 비교하여 통계적으로 유의미한 성능 향상을 보여줍니다.

- **Performance Highlights**: 연구 결과, 제안된 KiNN 모델은 CLEF e-Risk에서 25% MCC 증가 및 12% F1 증가, PRIMATE 데이터셋에서는 2.5% MCC 증가 및 19% F1 증가의 성과를 달성하였습니다. KiNN은 기존 모델들에 비해 더 높은 설명 가능성을 제공하며, 또 다른 모델들이 부족한 부분에 대하여 설명할 수 있는 능력을 갖추고 있습니다.



### CoRA: Optimizing Low-Rank Adaptation with Common Subspace of Large Language Models (https://arxiv.org/abs/2409.02119)
- **What's New**: 본 논문에서는 Low-Rank Adaptation (LoRA)의 효율성을 유지하면서 모델 파라미터를 절반으로 줄여주는 새로운 방법론, CoRA를 제안합니다. CoRA는 여러 대규모 모델에서 추출한 공통 기저(subspace)를 활용하여 LoRA의 B 매트를 최적화합니다.

- **Technical Details**: 제안하는 두 가지 접근 방식은 다음과 같습니다: (1) 공통 기저 행렬 B를 고정(freeze)하여 특수 작업에 대한 A 행렬을 학습하는 동시에 LoRA의 전체 파라미터를 50% 줄입니다. (2) 공통 기저 행렬 B를 LoRA의 기본 B 매트에 대한 향상된 초기 상태로 사용하여 더 나은 성능을 달성합니다.

- **Performance Highlights**: 첫 번째 접근 방식은 기존 LoRA 방법과 동일한 효율성을 유지하며, 두 번째 접근 방식은 LoRA의 원래 성능을 초과하여 최대 4%의 성능 개선을 달성합니다. 이는 유창성(fluency), 관련성(relevance), 정확성(accuracy)에서 뛰어난 성능을 보여줍니다.



### TSO: Self-Training with Scaled Preference Optimization (https://arxiv.org/abs/2409.02118)
- **What's New**: 이번 연구에서는 TSO(Self-Training with Scaled Preference Optimization)라는 새로운 프레임워크를 제안하여 Large Language Models(LLMs)의 인간 선호도에 대한 적합성을 높였습니다. 이를 통해 추가적인 보상 모델 없이도 선호 최적화 학습이 가능해지며, 다양한 인간의 반응을 통합해 응답의 다양성을 개선합니다.

- **Technical Details**: TSO 프레임워크는 모델 매트릭스를 활용하여 응답의 다양성을 증대시키며, 인간 및 AI 피드백을 통해 모델 선호 오류를 교정합니다. 이 과정에서 다단계 자기훈련(self-training) 구조를 채택하고, 미니 배치(iterative mini-batches) DPO와 쌍클립 보상 손실(dual clip reward loss)을 도입하여 데이터 효율성을 높이고 최적화 프로세스를 조정합니다.

- **Performance Highlights**: 실험 결과, TSO는 기존의 주요 방법들보다 다양한 정렬 평가 벤치마크에서 우수한 성능을 보였고, 선호 데이터 구축 및 모델 훈련 전략에서 실용적인 통찰력을 제공했습니다.



### Tiny-Toxic-Detector: A compact transformer-based model for toxic content detection (https://arxiv.org/abs/2409.02114)
Comments:
          6 pages

- **What's New**: 새로운 논문에서는 토닉(Toxic) 콘텐츠 탐지를 위한 소형 변환기 기반 모델인 Tiny-toxic-detector를 소개합니다. 210만 개의 매개변수만으로도, ToxiGen 데이터셋에서 90.97%의 정확도와 Jigsaw 데이터셋에서 86.98%의 정확도를 기록하며, 100배 이상의 모델들에 맞먹는 성능을 발휘합니다.

- **Technical Details**: Tiny-toxic-detector는 4개의 transformer encoder 레이어로 구성되며, 각 레이어는 2개의 attention head를 가지고 있습니다. 임베딩 차원은 64로 설정되어 있으며, feedforward 레이어의 차원은 128입니다. 이 모델은 공개 및 비공식 데이터셋을 사용하여 훈련되었으며, 훈련 과정에서 오버피팅을 활용하여 일반화를 개선하였습니다.

- **Performance Highlights**: 이 모델은 환경이 제한된 상황에서도 효과적으로 작동하도록 설계되었으며, 10MB의 RAM과 8MB의 VRAM만 필요합니다. CPU 기반 시스템에서 태이블 2의 결과에서 큰 모델들을 빠르게 능가하는 고속 추론이 가능합니다.



### Driver Digital Twin for Online Prediction of Personalized Lane Change Behavior (https://arxiv.org/abs/2211.01294)
- **What's New**: 이 연구에서는 Connected and Automated Vehicles (CAVs)가 Human-Driven Vehicles (HDVs)와 혼합된 교통 환경에서 안전한 행동을 취할 수 있도록 도와주는 Driver Digital Twin (DDT) 시스템을 개발했습니다.

- **Technical Details**: DDT는 차량-엣지-클라우드 아키텍처에 배치되어 HDV의 운전 행동 모델을 역사적 자연 주행 데이터에 기반하여 클라우드 서버가 구성합니다. 엣지 서버는 각 운전자의 실시간 데이터를 처리하여 디지털 트윈을 통해 차선 변경 행동을 예측합니다. 이 시스템은 'human-in-the-loop' 공동 시뮬레이션 플랫폼에서 평가되었으며, 이후 4G/LTE 셀룰러 네트워크를 통해 연결된 3대의 승용차로 현장 구현이 이루어졌습니다.

- **Performance Highlights**: 차선 변경 의도가 차량이 차선 분리선(Lane Separation Line)을 넘기 전 평균 6초 이내에 인식되며, 예측 궤적과 GPS 실제 값 간의 평균 유클리드 거리(Mean Euclidean Distance)는 1.03미터입니다. 개인화된 모델을 사용하면 예측 정확도가 27.8% 향상됩니다.



### On a heuristic approach to the description of consciousness as a hypercomplex system state and the possibility of machine consciousness (German edition) (https://arxiv.org/abs/2409.02100)
Comments:
          7 pages, in German language. 1 figure

- **What's New**: 이 연구는 인간의 의식 상태가 물리적인 기반을 가지면서도 상상적인 초복합(hypercomplex) 구조를 가지고 있다는 점에 주목합니다.

- **Technical Details**: 논문은 이론적 분석을 통해 이른바 이중복합(bicomplex) 대수를 이용한 계산을 통해 기계에서 초복합 시스템 상태를 효과적으로 생성하고 활용할 수 있다는 가능성을 제시합니다.

- **Performance Highlights**: 최고 복잡성을 지닌 AI 시스템의 놀라운 성능이 초복합 시스템 상태의 존재 가능성을 지지하고 있지만, 이러한 시스템과 다른 시스템을 구별할 수 있는 실험 데이터는 부족합니다.



### A Deployed Online Reinforcement Learning Algorithm In An Oral Health Clinical Tria (https://arxiv.org/abs/2409.02069)
- **What's New**: 본 논문에서는 구강 질병에 노출된 취약계층을 대상으로 하는 mHealth(mobile Health) 개입 시스템인 Oralytics를 개발하였습니다. Oralytics는 치과 질환 예방을 위한 임상 진료를 지원하기 위해 온라인 강화 학습(reinforcement learning) 알고리즘을 통합하여 개입 프롬프트를 전달하는 최적의 시기를 결정합니다.

- **Technical Details**: Oralytics에는 블루투스(Bluetooth) 지원 전동칫솔과 스마트폰 애플리케이션이 포함되어 있습니다. 각각의 사용자는 이 시스템을 통해 칫솔질의 품질을 감지하고, 해당 품질 데이터를 기반으로 올바른 구강 자기 관리 행동(OSCB)을 유도하는 알림을 받습니다. 알고리즘은 학습 중에 제한된 양의 데이터에서도 최적의 결정을 내릴 수 있도록 설계되었습니다.

- **Performance Highlights**: Oralytics는 2023년 9월부터 2024년 7월까지 UCLA 치과 클리닉에서 진행된 임상 시험에 성공적으로 배포되었으며, 총 79명의 참가자가 등록되었습니다. 알고리즘은 참가자의 아침 및 저녁 칫솔질 시간에 맞춰 참가자들에게 하루에 두 번의 알림을 자동으로 결정하고 전달하였습니다.



### From Grounding to Planning: Benchmarking Bottlenecks in Web Agents (https://arxiv.org/abs/2409.01927)
- **What's New**: 이번 연구는 웹 기반 에이전트를 연구하며, 그 성능 향상을 위해 에이전트의 두 가지 핵심 구성요소인 Planning과 Grounding을 구분하여 분석합니다. 이를 통해 기존의 성능 저하 원인을 둘로 나누어 살펴보며, 실제 웹 환경에서의 과제를 수행하는 데 있어 Grounding은 큰 병목 현상이 아니며, 주로 Planning 부문에서의 개선이 필요하다는 새로운 인사이트를 제공합니다.

- **Technical Details**: 연구는 Mind2Web 데이터셋의 실험을 개선하여 Planning과 Grounding의 영향을 분리하여 평가합니다. 고수준(hight-level)과 저수준(low-level) 두 가지 운영 모드를 도입하여 각 구성 요소의 성능을 명확하게 분석하고, 웹 에이전트가 더 나은 성능을 내도록 하는 다양한 제안사항을 도출했습니다. 이는 웹 기반 태스크의 요소 필터링 및 순위 매기기 메커니즘의 중요성을 강조합니다.

- **Performance Highlights**: WebNaviX 에이전트를 통해 페이지 이해 및 순위 매기기 메커니즘을 향상시킨 결과, 최첨단 기술(SOTA) 대비 13%의 성능 향상을 달성했습니다. 이로 인해 웹 에이전트의 신뢰성을 향상시키고 보다 빈번한 자동화 및 지능적 상호작용이 가능해질 것으로 기대됩니다.



### A randomized simulation trial evaluating ABiMed, a clinical decision support system for medication reviews and polypharmacy managemen (https://arxiv.org/abs/2409.01903)
Comments:
          8 pages, 6 figures

- **What's New**: 이번 연구에서는 약사들이 약물 검토를 수행하는 데 도움을 줄 수 있는 ABiMed라는 임상 의사결정 지원 시스템을 소개합니다.

- **Technical Details**: ABiMed는 약물을 시각적으로 제공하는 기능을 갖춘 STOPP/START v2 가이드라인을 기반으로 한 시스템입니다. 이 시스템은 환자 데이터를 GP의 전자 건강 기록(EHR)에서 추출하고, 약물 지식을 시각적으로 제공하여 약사와 GP 간의 소통을 촉진합니다.

- **Performance Highlights**: ABiMed를 사용한 경우, 약사들은 약물 관련 문제를 1.6배 더 많이 발견하고, 더 나은 중재를 제안할 수 있었으며, 시간 소모는 동일하였습니다. 시스템 유용성 점수는 82.7로 '우수' 등급을 받았습니다.



### Learning State-Dependent Policy Parametrizations for Dynamic Technician Routing with Rework (https://arxiv.org/abs/2409.01815)
- **What's New**: 이번 연구는 기술자의 스킬 이질성과 결재 불능의 불확실성을 고려하여 서비스 라우팅 문제를 모델링합니다. 강화 학습을 활용하여 각 상태에 따라 동적으로 조정되는 최적의 정책을 제안합니다.

- **Technical Details**: 연구에서는 여러 날에 걸쳐 요청된 서비스에 대해 이질적인 기술자들을 고객에게 효율적으로 라우팅하는 과정을 설명합니다. 이를 위해 경로 효율성, 서비스 긴급성, 재작업 위험 등을 종합적으로 고려하는 점수 함수를 통해 기술자와 작업의 비효율적 할당 문제를 해결합니다. 이 과정에서 강화 학습(RL)을 통해 상태 의존적 파라미터화(state-dependent parametrization)를 도입합니다.

- **Performance Highlights**: 이 연구 결과, 고객의 평균 불편을 최소화하면서 서비스 완성률은 95% 이상에 달하며, 정량적 실험을 통해 기존 정책 대비 약 8%의 성능 향상을 달성했습니다. 또한, 적은 수의 전문 기술자를 통해도 유사한 성과를 낼 수 있다는 점이 강조되었습니다.



### LASP: Surveying the State-of-the-Art in Large Language Model-Assisted AI Planning (https://arxiv.org/abs/2409.01806)
- **What's New**: 이 논문은 LLM(대규모 언어 모델)을 활용한 AI 계획의 현재 도전 과제를 살펴보고, 이러한 모델들이 실제 계획 문제를 해결하는 데 어떻게 기여할 수 있는지를 소개합니다.

- **Technical Details**: 기존의 AI 계획 접근 방식은 제한된 도메인에 국한되어 있으며, LLM은 이러한 문제를 해결하기 위한 새로운 프레임워크를 제시합니다. 특히 Planning Domain Definition Language (PDDL)를 활용하여 계획 시스템을 정의하고, 다양한 벤치마크 데이터를 통해 LLM의 계획 능력을 평가합니다.

- **Performance Highlights**: LLM의 계획 능력은 자연어 처리(NLP) 연구자들에게 큰 혜택을 줄 수 있으며, LLM을 계획 프레임워크에 통합한 성공 사례를 정리하여 향후 연구 방향 및 개선 기회를 제시합니다.



### Lexicographic optimization-based approaches to learning a representative model for multi-criteria sorting with non-monotonic criteria (https://arxiv.org/abs/2409.01612)
Comments:
          45 pages, 12 figures

- **What's New**: 이 논문에서는 비모노토닉(non-monotonic) 기준을 대처하기 위한 MCS 문제의 대표 모델 학습을 위해 임계값 기반의 가치 주도 분류(threshold-based value-driven sorting) 절차를 통합하여 새로운 접근 방식을 제안하고 있습니다.

- **Technical Details**: 논문에서는 모형을 UTA(UTA-like) 기능 공간으로 매핑하는 변환 함수(transformations)를 정의하고, 비모노토닉 기준을 모델링하기 위한 제약 조건 세트를 구성합니다. 또한 비모노토닉 기준을 위한 두 가지 레지코그래픽 최적화 접근법(lexicographic optimization-based approaches)을 개발하여 MCS 문제의 대표 모델을 도출합니다.

- **Performance Highlights**: 제안된 접근법의 효용성과 타당성을 보여주기 위해 포괄적인 시뮬레이션 실험을 수행하였고, 비모노토닉 기준 모델링 방법과의 비교 분석을 통해 이 접근법의 성능을 평가하였습니다.



### H-ARC: A Robust Estimate of Human Performance on the Abstraction and Reasoning Corpus Benchmark (https://arxiv.org/abs/2409.01374)
Comments:
          12 pages, 7 figures

- **What's New**: 이 연구는 인기 있는 Abstraction and Reasoning Corpus (ARC) 벤치마크에 대한 인간 성능을 보다 철저히 평가했습니다. 총 1729명이 400개 훈련 및 평가 작업을 수행하여 인간의 평균 성과를 73.3%~77.2%로 추정하였으며, 특히 H-ARC라는 데이터셋을 공개하였습니다.

- **Technical Details**: 이 연구는 400개의 훈련 작업과 400개의 평가 작업을 포함하는 ARC의 전체 데이터셋에서 인간 참가자들의 작업 수행 기록을 수집했습니다. 참가자들은 최대 3번의 시도로 각각의 태스크를 해결하며, 실험은 Amazon Mechanical Turk를 통해 진행되었습니다.

- **Performance Highlights**: 인간 성능은 훈련 세트에서 평균 76.2%의 정확도를 보여주었고, 평가 세트에서는 평균 64.2%의 정확도를 기록했습니다. 특히, 모든 작업 중 790개는 800명 중 한 명 이상이 해결할 수 있어 ARC 작업의 대부분이 일반 크라우드 워커에 의해 해결될 수 있는 가능성을 보여줍니다.



### Pairing Analogy-Augmented Generation with Procedural Memory for Procedural Q&A (https://arxiv.org/abs/2409.01344)
- **What's New**: 이 연구에서는 복잡한 절차적 질문 응답(task)에서의 성능을 개선하기 위한 'analogy-augmented generation (AAG)'이라는 새로운 시스템을 제안합니다. 이 방법은 기존의 RAG 시스템을 확장하여, 사람의 유사 추론 능력을 활용하여 절차적 지식을 효과적으로 처리합니다.

- **Technical Details**: AAG 시스템은 세 가지 주요 모듈로 구성됩니다: 1) 절차적 메모리 스토어(procedural memory store), 2) 쿼리 재작성(query rewriting) 및 요약, 3) 자기 비판(iterative refinement with self-critic)입니다. 이 시스템은 절차적 지식을 처리하기 위해 특정하게 설계된 메모리 표현을 사용하며, 기존의 정보를 활용하여 새로운 문제를 해결합니다.

- **Performance Highlights**: 제안된 AAG 시스템은 LCStep, RecipeNLG, CHAMP 데이터셋에서 전통적인 RAG 및 few-shot 방법론보다 더 나은 성능을 보여 주었으며, 인류 평가에서도 유의미한 개선이 관찰되었습니다.



### Conversational Complexity for Assessing Risk in Large Language Models (https://arxiv.org/abs/2409.01247)
Comments:
          14 pages, 6 figures

- **What's New**: 최근 대형 언어 모델(LLM)의 발전은 다양한 응용 프로그램에서 인간과 유사한 텍스트를 생성할 수 있게 했지만, 이 과정에서 유해하거나 비윤리적인 콘텐츠를 생성할 가능성 또한 커졌습니다. 특히 복잡한 대화가 이러한 문제를 초래할 수 있다는 점에서, 최소한의 대화 길이(Conversational Length, CL)와 대화 복잡도(Conversational Complexity, CC)를 새로운 리스크 평가 지표로 제안합니다.

- **Technical Details**: CL은 특정 응답을 얻기 위해 필요한 대화의 길이를 정량화하는 측정 방법이며, CC는 유저의 지시 순서가 해당 응답에 이르는 데 필요한 Kolmogorov 복잡성을 기반으로 정의됩니다. Kolmogorov 복잡성의 비계산 가능성을 해결하기 위해, 우리는 CC를 참조 LLM을 사용하여 유저 지시의 압축성을 추정하는 방법으로 근사화합니다.

- **Performance Highlights**: 대규모 레드팀 데이터셋을 활용한 정량적 분석을 통해, 유해한 대화와 무해한 대화의 길이와 복잡도의 통계 분포를 살펴보았습니다. 연구 결과, 이 분석은 AI 안전성을 이해하는 데 유용한 도구가 될 수 있음을 제시하며, 유해 정보 접근성에 대한 통찰력을 제공합니다.



### Integrating End-to-End and Modular Driving Approaches for Online Corner Case Detection in Autonomous Driving (https://arxiv.org/abs/2409.01178)
Comments:
          IEEE SMC 2024

- **What's New**: 이 논문은 자율주행 차량의 안전성을 높이기 위해 모듈 시스템과 엔드 투 엔드 접근 방식의 장점을 결합한 온라인 코너 케이스(detection corner case) 감지 방법을 제안합니다. 이 방법은 두 시스템 간의 불일치를 활용하여 코너 케이스를 정의합니다.

- **Technical Details**: 제안된 방법론은 기본 시스템과 이차 시스템으로 나뉘며, 기본 시스템은 모듈식 구조로 차량의 주행 작업을 담당합니다. 이차 시스템은 엔드 투 엔드 네트워크로, 주행 기능에 대한 전체 데이터 최적화를 통해 더 나은 상황 인식을 제공하며, 불일치를 통해 코너 케이스를 감지합니다.

- **Performance Highlights**: 현실 차량에 적용된 결과, 엔드 투 엔드 네트워크가 보조 시스템으로서 코너 케이스 감지에 효과적으로 기여함을 보여주었습니다. 이 접근법은 자율주행 차량의 안전성을 향상시킬 수 있는 가능성을 제시합니다.



### Learning in Hybrid Active Inference Models (https://arxiv.org/abs/2409.01066)
Comments:
          11 pages (+ appendix). Accepted to the International Workshop on Active Inference 2024. arXiv admin note: substantial text overlap with arXiv:2408.10970

- **What's New**: 본 논문에서는 하이브리드 계층적 능동 추론(Active Inference) 에이전트를 새롭게 제안합니다. 이 에이전트는 상위에서 낮은 수준의 연속성을 제어하는 모듈과 결합되어 유연하게 추상적인 목표를 설정하고 효율적인 학습이 가능합니다.

- **Technical Details**: 새로운 계층적 하이브리드 능동 추론 에이전트는 고차원적이고 연속적인 환경에서 유용한 이산적 추상 개념을 학습하는데 중점을 둡니다. 이 시스템은 재귀적 스위칭 선형 동적 시스템(recurrent Switching Linear Dynamical Systems, rSLDS)을 활용하여 복잡한 연속 동역학의 조각별 선형 분해를 통해 의미있는 이산 표현을 학습합니다.

- **Performance Highlights**: 연속 마운틴 카(Continuous Mountain Car) 과제를 통해 제안된 모델을 적용한 결과, 효율적인 시스템 식별과 성공적인 계획을 입증하였습니다. 이 모델은 추상적인 하위 목표를 구체화하여 비재미롭고 복잡한 계획 문제를 해결하는데 강력한 성능을 보였습니다.



### Unlocking the Wisdom of Large Language Models: An Introduction to The Path to Artificial General Intelligenc (https://arxiv.org/abs/2409.01007)
- **What's New**: 이 책자, "대형 언어 모델의 지혜를 여는 법"은 포괄적인 연구 "인공지능 일반화 지능으로 가는 길"의 서론 역할을 합니다.

- **Technical Details**: 본 책자는 아홉 개의 격언(aphorisms)을 통해 AI의 미래를 조명하는 LLM 대화의 원칙과 통찰력을 정리합니다. 이러한 접근법은 인공지능 일반화 지능(AGI)의 실현을 위한 잠재적 경로로 제안됩니다.

- **Performance Highlights**: 책자에는 주요 책의 제목, 초록(abstract), 서론(introduction)뿐만 아니라 첫 두 장을 전문(全文)으로 포함하고 있습니다.



### JaxLife: An Open-Ended Agentic Simulator (https://arxiv.org/abs/2409.00853)
- **What's New**: JaxLife라는 인공 생명 시뮬레이터를 통해, 고급 추론 및 도구 사용이 가능한 진화된 에이전트를 생성하는 새로운 접근 방식을 제시하고 있습니다.

- **Technical Details**: 이 시뮬레이터는 에이전트가 생존하기 위해 행동해야 하는 3가지 주요 요소로 구성되어 있으며, 프로그래머블 로봇과의 상호작용을 통해 중요한 Turing-complete 행동을 표현합니다. 에이전트는 심층 신경망에 의해 파라미터화되어 진화 학습을 수행합니다.

- **Performance Highlights**: 본 연구에서는 초기 농업, 도구 사용 및 기본적인 통신 프로토콜의 출현을 보여주며, 컴퓨팅 자원의 양에 따라 복잡성이 어떻게 스케일링되는지에 대한 초기 추정값도 제공합니다.



### You-Only-Randomize-Once: Shaping Statistical Properties in Constraint-based PCG (https://arxiv.org/abs/2409.00837)
Comments:
          Published in Foundations of Digital Games (FDG) 2024. 10 pages, 6 figures

- **What's New**: 이 논문에서는 You-Only-Randomize-Once (YORO)라는 새로운 사전 롤링(pre-rolling) 기법을 소개하여, 제약 해결기(constraint solver)가 원하는 통계(statistics)를 코드화할 수 있도록 하였다. 이를 통해 출력의 전역 제약(global constraints)을 유지하면서도 타일 그리드(tile-grid) 결과의 통계적 특성을 효과적으로 제어할 수 있다.

- **Technical Details**: YORO 기법은 제약 해결기를 위한 의사결정 변수 순서를 작성하는 방법으로 사용되며, 각 해결기를 실행하기 전에 한 번의 난수 배치를 생성하여 원하는 분포를 따르는 출력 결과를 만들어낸다. WFC(Wave Function Collapse) 알고리즘의 예를 통해, 여러 가지 상용 SAT 솔버(SAT solvers)를 사용하여 이 기법이 얼마나 효과적인지를 보여준다.

- **Performance Highlights**: YORO 기법은 통계적 제어(statistical control)를 통해 여러 결과를 생성하면서도, 각 결과가 원하는 설계 요소 통계 특성을 갖도록 보장한다. 실제 문제를 다루면서, 기존 WFC 알고리즘보다 더 우수한 결과와 다양한 통계적 속성을 재현하는 데 성공하였다.



### Building FKG.in: a Knowledge Graph for Indian Food (https://arxiv.org/abs/2409.00830)
Comments:
          14 pages, 3 figures, 25 references, Formal Ontology in Information Systems Conference 2024 - Integrated Food Ontology Workshop

- **What's New**: 이번 논문에서는 인도 음식에 대한 정보를 자동으로 수집하기 위한 지식 그래프 구축을 위한 본체 설계와 지식 공학(knowledge engineering) 기법을 제안합니다. 특히 요리, 레시피, 성분, 영양상의 모든 지식을 포괄적으로 수집하는 지능형 방법의 설계를 다루고 있습니다.

- **Technical Details**: 제안된 기법은 공공 도메인에서 레시피 블로그 사이트의 정보를 수집하고 AI(Artificial Intelligence), LLM(Large Language Model), 언어 기술을 활용하여 인도 음식의 지식 그래프를 구축합니다. 이 과정에는 인간의 개입(human-in-the-loop)도 포함되어 신뢰성을 높이고 있습니다.

- **Performance Highlights**: FKG.in이라는 지식 그래프는 인도 요리의 전모를 포괄하여 후속 음식 컴퓨팅 애플리케이션을 구축하는 디지털 자원으로 활용될 것입니다. 이 연구는 다른 도메인에도 적용 가능하며, AI 기반 스마트 분석 및 개인화된 디지털 건강을 위한 추천 시스템 구축에 기여할 것입니다.



### Accelerating Hybrid Agent-Based Models and Fuzzy Cognitive Maps: How to Combine Agents who Think Alike? (https://arxiv.org/abs/2409.00824)
Comments:
          To appear at the 2024 Winter Simulation Conference

- **What's New**: 이 논문은 유사한 사고를 가진 에이전트(agents)를 그룹화 하여 에이전트 기반 모델(Agent-Based Models, ABM)의 인구 규모와 계산 시간을 줄이는 새로운 근사를 제안합니다.

- **Technical Details**: 각 에이전트의 행동을 규칙 네트워크(Fuzzy Cognitive Maps, FCM)로 표현하고, 이러한 네트워크 간의 거리 측정을 통해 에이전트 그룹을 형성합니다. 커뮤니티 감지(community detection) 알고리즘을 사용하여 '슈퍼 에이전트'(super agents)로 구성되어, 계산 비용을 절감합니다.

- **Performance Highlights**: 사례 연구를 통해 제안된 단순화가 정확성을 유지함을 보여줍니다. 에이전트의 행동을 비교하는 새로운 메트릭(metrics)을 도입하고 이에 대한 평가를 수행하여 기존의 하이브리드 모델의 계산 비용을 감소시킬 수 있음을 입증했습니다.



### Cooperative Path Planning with Asynchronous Multiagent Reinforcement Learning (https://arxiv.org/abs/2409.00754)
- **What's New**: 본 논문에서는 여러 소스-목적지 쌍(Multiple Source-Destination Pairs, MSD)을 가진 최단 경로 문제(Shortest Path Problem, SPP)를 다루어 모든 최단 경로의 평균 이동 시간을 최소화하는 방법을 제안합니다.

- **Technical Details**: 연구에서 제안하는 asyn-MARL 프레임워크는 로드 네트워크를 여러 개의 서브 그래프(sub-graphs)로 나누고, 두 단계의 경로 계획(inter-region and intra-region route planning) 프로세스를 실행합니다. 글로벌 상태(global state)를 설계하여 다중 에이전트의 공동 관찰 및 행동을 저차원 벡터로 표현하고, 경로의 훈련 궤적에서 중복성을 줄이는 새로운 궤적 수집 메커니즘을 개발하였습니다. 또한, 동일하거나 가까운 목적지를 향한 차량 간 협력을 용이하게 하는 새로운 actor network를 설계했습니다.

- **Performance Highlights**: 합성 및 실제 도로 네트워크에서의 평가 결과, 제안한 접근 방식이 최신 계획 방법(state-of-the-art planning approaches)을 능가함을 입증하였습니다.



### AgGym: An agricultural biotic stress simulation environment for ultra-precision management planning (https://arxiv.org/abs/2409.00735)
- **What's New**: 이번 논문에서는 전통적인 농업 생산에서의 해충 및 질병 관리를 최적화하기 위해 AgGym이라는 머신 러닝 기반 가상 농장 관리 플랫폼을 제안합니다. AgGym은 생물학적 스트레스의 확산을 모델링하고, 화학 처리 유무에 따른 수확량 손실을 추정하는 모듈형 시뮬레이션 프레임워크입니다.

- **Technical Details**: AgGym은 감염 확산 모듈과 화학 처리 하에서의 수확량 추정 모듈로 구성됩니다. 이 프레임워크는 crop models 및 weather data와 통합되어, 날씨 조건, 토양 영양소, 수분 가용성에 따라 기대 가능한 수확량을 제공받고, 이를 기초로 생물학적 스트레스가 유발하는 추가 수확량 손실을 추정합니다. 특히, deep reinforcement learning (RL) 알고리즘을 사용하여 최적의 감염 완화 전략을 설계하고 있습니다.

- **Performance Highlights**: AgGym의 검증 결과, 제한된 데이터로도 다양한 생물학적 스트레스 조건에서 수확량 결과를 시뮬레이션할 수 있음을 보여주며, 이는 보다 정밀한 생물학적 스트레스 관리 전략을 설계하는 데 기여할 잠재력을 지니고 있습니다. AgGym 소프트웨어는 오픈 소스 커뮤니티 리소스로 제공되며, 전문가들의 기여를 환영합니다.



### Hound: Hunting Supervision Signals for Few and Zero Shot Node Classification on Text-attributed Graph (https://arxiv.org/abs/2409.00727)
- **What's New**: 본 연구에서는 텍스트 속성 그래프(Text-attributed graph, TAG)에서의 few-shot 및 zero-shot 노드 분류의 정확성을 향상시키기 위해 Hound라는 새로운 방법론을 제안합니다. 기존 방법들이 대조 손실(contrastive loss)만을 사용했으나, Hound는 더 많은 감독 신호(supervision signals)를 제공하는 것을 목표로 합니다.

- **Technical Details**: Hound는 세 가지 증강 기법(augmentation techniques)을 도입했습니다: 노드 변형(node perturbation), 텍스트 일치(text matching), 의미의 부정(semantics negation). 노드 변형은 그래프에서 엣지를 무작위로 추가/삭제하여 다양한 노드 임베딩을 생성합니다. 텍스트 일치는 유사한 임베딩을 가진 텍스트를 검색하여 노드와 일치시킵니다. 의미의 부정은 원래 텍스트와 반대 의미를 가지는 부정적인 텍스트를 생성하여 본래의 노드 및 텍스트와 대비하는 것입니다.

- **Performance Highlights**: Hound는 5개 데이터셋에 대해 13개의 최신 기법과 비교한 결과, 모든 기준선(baselines)에서 일관되게 더 높은 정확성을 달성했습니다. 특히, few-shot 및 zero-shot 분류에서 각각 평균 4.6% 및 8.8%의 정확도가 향상되었습니다.



### Abstaining Machine Learning -- Philosophical Considerations (https://arxiv.org/abs/2409.00706)
Comments:
          Part of the published PhD Thesis: Daniela Schuster. Suspension of Judgment in Artificial Intelligence-Uncovering Uncertainty in Data-Based and Logic-Based Systems. PhD thesis, University of Konstanz, 2024. this http URL

- **What's New**: 이 논문은 기계 학습(ML)과 철학 분야 간의 연관성을 탐구하며, 중립적인 반응을 제공할 수 있는 특정 유형의 기계 학습 시스템인 abstaining machine learning (AML) 시스템을 소개합니다. 이 시스템은 아직 철학적 관점에서 연구되지 않았으며, 철학의 판단 유보(suspension of judgment) 개념과의 관련성을 논의합니다.

- **Technical Details**: 이 연구는 중립적 행동을 나타내는 AML 시스템의 다양한 유형을 소개하고 이를 두 가지 차원으로 구분합니다. 첫 번째 차원은 abstention을 유발하는 이유(즉, 특정 상황에서의 abstaining output 발생), 두 번째 차원은 시스템에서의 abstention 구현 방법을 설명합니다. AML 시스템을 사용하여 사용자에게 지식의 한계를 명확히 정의하고 전달할 수 있는 방법을 탐구합니다.

- **Performance Highlights**: 논문에서 제안하는 특정 유형의 AML 시스템은 판단 유보 기준을 충족하며, 자율적으로 abstaining 출력을 생성하고 그것에 대해 설명할 수 있는 능력에서 다른 유형들에 비해 월등한 것으로 평가됩니다. 이론적으로, AML 시스템은 결정 과정에서의 불확실성을 해결하는 효과적인 방법으로 사용될 수 있습니다.



### GenAI-powered Multi-Agent Paradigm for Smart Urban Mobility: Opportunities and Challenges for Integrating Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) with Intelligent Transportation Systems (https://arxiv.org/abs/2409.00494)
- **What's New**: 이 논문은 최근의 생성적 AI(Generative AI)와 다중 에이전트 시스템을 활용하여 스마트 시티 응용 프로그램의 기능과 효율성을 향상시키는 방법을 탐구합니다. 특히, 대형 언어 모델(LLM)과 검색 증강 생성(RAG) 기술을 지능형 운송 시스템(ITS)에 통합하여 도시 이동성의 주요 문제를 해결하는 혁신적인 솔루션을 제안합니다.

- **Technical Details**: 이 연구에서는 LLM과 RAG 기술의 통합을 통해 다중 에이전트 시스템을 개발하는 개념적 프레임워크를 제시합니다. 이 시스템은 (a) 과학 기반 자문을 통해 교통 혼잡, 사고 및 탄소 배출을 줄이고, (b) 참여적인 이동성 관리에 대한 공공 교육 및 참여를 촉진하며, (c) 데이터 분석, 지식 표현 및 교통 시뮬레이션과 같은 ITS 플랫폼의 자동화를 지원하는 기능을 포함합니다.

- **Performance Highlights**: 제안된 LLM-RAG 기반의 다중 에이전트 시스템은 고유한 사용자 정의 시나리오에 기반하여 응답을 생성하며, 실시간 및 과거 데이터를 활용하여 도시 이동성의 현재 상황을 반영합니다. 이는 정보가 시의적절하고 맥락을 고려하도록 하여 의사 결정을 향상시키는 데 기여합니다. 또한, 사용자의 특정 요구 및 선호에 맞춘 개인화된 응답 제공을 통해 보다 직관적이고 효율적인 스마트 이동성 서비스를 목표로 합니다.



### The MERIT Dataset: Modelling and Efficiently Rendering Interpretable Transcripts (https://arxiv.org/abs/2409.00447)
- **What's New**: 이 논문은 학교 보고서의 맥락에서 생성된 MERIT Dataset을 소개합니다. 이 데이터셋은 텍스트, 이미지, 레이아웃을 포함한 멀티모달( multimodal) 완전 레이블 데이터셋으로, 400개 이상의 레이블과 33,000개의 샘플로 구성되어 있습니다.

- **Technical Details**: MERIT Dataset은 Visually-rich Document Understanding (VrDU) 작업을 위해 훈련 모델에 유용한 자원입니다. 이 데이터셋은 학생 성적 보고서로 구성되어 있으며, 언어 모델( Language Models)에서 발생할 수 있는 편향(Bias)을 벤치마크(Benchmark) 하는 데 가치가 있습니다. 논문에서는 데이터셋 생성 파이프라인(pipeline)과 텍스트, 시각적, 레이아웃, 편향 도메인에서의 주요 특징을 하이라이트합니다.

- **Performance Highlights**: 실험에서는 MERIT Dataset을 활용하여 토큰 분류 모델(Token Classification Models)과의 벤치마크를 제시하며, SOTA( State of the Art) 모델조차도 이 데이터셋에서 상당한 도전 과제를 직면하고 있다고 보여줍니다. 이러한 모델들은 MERIT Dataset 샘플을 사전 훈련(pretraining) 단계에서 포함함으로써 큰 이득을 보일 것입니다.



### Predicting Femicide in Veracruz: A Fuzzy Logic Approach with the Expanded MFM-FEM-VER-CP-2024 Mod (https://arxiv.org/abs/2409.00359)
Comments:
          24 pages, 2 tables, 3 figures

- **What's New**: 이번 논문은 멕시코 베라크루즈에서의 여성 살해(femicide)라는 긴급한 문제를 다루고 있으며, 여성을 대상으로 한 폭력의 복잡성과 불확실성을 다루기 위해 퍼지 논리(fuzzy logic)를 이용한 MFM_FEM_VER_CP_2024 모델을 개발했습니다.

- **Technical Details**: 이 모델은 강압적 통제(coercive control), 비인간화(dehumanization), 폭력의 순환(cycle of violence)과 같은 위험 요소들을 수학적으로 형식화합니다. 위험 요소들은 다양한 개인 관계와 특정 폭력 행위와 관련된 위험 정도를 평가하는 멤버십 함수(membership functions)를 통해 모델링 되었습니다. 새로운 규칙(rule)을 통합하고 기존 멤버십 함수를 개선함으로써 원래 모델을 강화했습니다.

- **Performance Highlights**: 모델의 예측 정확도(predictive accuracy)가 크게 향상되어 향후 여성을 대상으로 한 폭력 예방 및 개입에 기여할 것으로 기대됩니다.



### Explainable Artificial Intelligence: A Survey of Needs, Techniques, Applications, and Future Direction (https://arxiv.org/abs/2409.00265)
- **What's New**: 이 논문은 Explainable Artificial Intelligence (XAI)에 관한 포괄적인 문헌 리뷰를 제공하며, 기존 문헌의 부족한 부분을 다룹니다. 특히, XAI 모델의 수학적 표현, 설계 방법론 등에 대한 상세한 분석이 отсутств됩니다.

- **Technical Details**: XAI의 개념, 일반 원칙, 다양한 XAI 기법의 분류법을 다루며, 각각의 방법이 어떻게 다양한 응용 분야에서 적용되는지를 설명합니다.

- **Performance Highlights**: XAI는 의료, 금융, 자율 주행 차량과 같은 안전-critical 도메인에서 AI 모델의 투명성, 책임성 및 공정성을 향상시키기 위해 중요합니다. 이 서베이는 XAI 연구자, 실무자, AI 모델 개발자에게 큰 도움이 될 것입니다.



### Estimating the number of reachable positions in Minishog (https://arxiv.org/abs/2409.00129)
Comments:
          This article was submitted to IPSJ (Information Processing Society of Japan) SIG Technical Reports for Game Informatics in September 6, 2024. (a non-reviewed technical report)

- **What's New**: 이 논문에서는 Minishogi(미니쇼기)에서 초기 위치로부터 도달 가능한 포지션의 수를 추정하는 새로운 방법을 제시합니다.

- **Technical Details**: Minishogi는 5x5 그리드에서 진행되는 보드 게임으로, 본 논문에서는 uniform random sampling을 활용해 후보 위치를 생성하고, 이후 일련의 합법적인 이동을 통해 초기 위치로부터 도달 가능한 포지션의 비율을 측정합니다. 이를 통해 도달 가능한 Minishogi 포지션 수는 약 $2.38 \times 10^{18}$로 추정됩니다.

- **Performance Highlights**: 본 연구의 결과는 Minishogi의 도달 가능한 포지션 수에 대한 이해를 높이며, 강력한 해법의 가능성을 탐색하는 데 중요한 기초 자료를 제공합니다.



### CRAFT Your Dataset: Task-Specific Synthetic Dataset Generation Through Corpus Retrieval and Augmentation (https://arxiv.org/abs/2409.02098)
- **What's New**: CRAFT는 사용자가 제공하는 소수의 예시만으로 특정 작업을 위한 고품질 합성 데이터셋을 효율적으로 생성하는 새로운 방법을 제안합니다. 이는 기존의 데이터셋 수집 방법보다 빠르고 자원 소모가 적습니다.

- **Technical Details**: CRAFT( Corpus Retrieval and Augmentation for Fine-Tuning )는 웹 크롤링으로 수집한 대규모 원시 데이터에서 유사도를 기반으로 문서를 검색한 다음, LLMs( Large Language Models )를 활용해 이를 특정 작업 형식으로 변환하여 커스터마이즈된 샘플로 만드는 과정입니다. 이 과정은 정교한 수작업 커스터마이징 없이 자동으로 진행됩니다.

- **Performance Highlights**: CRAFT 활용 모델은 의학, 생물학, 일반 상식 질문 응답(QA) 및 요약 작업에서 데이터를 기반으로 훈련된 경쟁 모델보다 성능이 높거나 동등한 성과를 나타내며, 특히 요약 모델은 인간이 큐레이션한 데이터로 훈련된 모델보다 46 포인트 더 높은 선호도를 보였습니다.



### DepthCrafter: Generating Consistent Long Depth Sequences for Open-world Videos (https://arxiv.org/abs/2409.02095)
Comments:
          Project webpage: this https URL

- **What's New**: DepthCrafter는 오픈 월드 비디오에서 깊이 추정을 위한 새로운 방법으로, 기존의 방법과 달리 카메라 자세나 광학 흐름과 같은 추가 정보 없이도 고정밀의 깊이 시퀀스를 생성한다.

- **Technical Details**: DepthCrafter는 사전 훈련된 이미지-비디오 확산 모델을 기반으로 한 비디오-깊이 모델을 교육하고, 110프레임까지 가변 길이의 깊이 시퀀스를 생성할 수 있도록 설계된 3단계 훈련 전략을 채택한다. 또한 긴 비디오를 세그먼트 별로 처리하고 매끄럽게 합치는 추론 전략을 제안한다.

- **Performance Highlights**: DepthCrafter는 다양한 데이터셋에 대한 평가에서 제로샷 설정 하에 최첨단 성능을 달성했으며, 깊이 기반 비주얼 효과 및 조건부 비디오 생성과 같은 다양한 후속 응용 프로그램을 지원한다.



### OLMoE: Open Mixture-of-Experts Language Models (https://arxiv.org/abs/2409.02060)
Comments:
          61 pages (24 main), 36 figures, 14 tables

- **What's New**: OLMoE는 7억 개의 파라미터를 갖는 완전 개방형 Mixture-of-Experts(MoE) 언어 모델로, 5조 개의 토큰에서 사전 훈련되었습니다. OLMoE-1B-7B-Instruct를 만들기 위해 추가 조정을 했으며, 경쟁 모델을 초월하는 성능을 자랑합니다.

- **Technical Details**: OLMoE는 총 6.9B의 파라미터를 가진 디코더 전용 LM으로, 각 입력 토큰에 대해 활성화되는 파라미터는 1.3B입니다. 이 모델은 64개의 소형 전문가(Experts) 중 8개를 활성화하여 MoE 모듈에서 작동하며, 학습에는 로드 밸런싱 및 라우터 z-손실이 포함됩니다.

- **Performance Highlights**: OLMoE-1B-7B는 모든 공개된 1B 모델을 초월하며, 높은 추론 비용을 요구하는 밀집 모델과 경쟁할 수 있는 성능을 보입니다. MMLU 및 GSM8k와 같은 벤치마크에서 Llama2-13B와 유사한 성능을 기록했습니다.



### Low-Resolution Face Recognition via Adaptable Instance-Relation Distillation (https://arxiv.org/abs/2409.02049)
Comments:
          Accepted by IJCNN 2024

- **What's New**: 본 연구에서는 저해상도 얼굴 인식을 용이하게 하기 위해 적응 가능한 인스턴스-관계 증류 방법(Adaptable Instance-Relation Distillation, AIRD)을 제안합니다. 이 방법은 지식 전이 과정을 증류와 적응 단계로 나누어 모델의 적응성을 향상시킵니다.

- **Technical Details**: 우리의 접근법은 인스턴스 수준(instance-level)과 관계 수준(relation-level)에서 고해상도 얼굴의 지식을 학생 모델에 전이하여 저해상도 얼굴 인식을 지원합니다. 교사 모델은 고해상도 얼굴에 대한 풍부한 지식을 가지고 있으며, 학생 모델은 이를 통해 디스크리미네이티브 특성을 생성합니다. 또한, 테스트 시에는 적응형 배치 정규화(Adaptive Batch Normalization, FaceBN)를 통해 실제 저해상도 얼굴 인식을 위한 전이 능력을 향상시킵니다.

- **Performance Highlights**: 광범위한 실험 결과는 제안된 방법이 저해상도 얼굴 인식에서 최첨단 성능을 달성함을 보여줍니다. 이를 통해 모델의 적응성과 효과성을 분명히 입증했습니다.



### AllWeatherNet:Unified Image enhancement for autonomous driving under adverse weather and lowlight-conditions (https://arxiv.org/abs/2409.02045)
- **What's New**: 이 새로운 연구에서는 눈, 비, 안개, 야간 등 다양한 악천후 조건에서 이미지 퀄리티와 명확성을 향상시키기 위한 AllWeather-Net이라는 방법을 제안합니다. 이 방법은 계층적 아키텍처를 채택하여 세 가지 의미적 수준(장면, 객체, 텍스처)에서 정보를 조합하여 이미지를 개선합니다.

- **Technical Details**: AllWeather-Net은 Scaled Illumination-aware Attention Mechanism (SIAM)을 도입하여 자율주행 인식에 중요한 도로 요소에 대한 학습을 유도합니다. 이 아키텍처는 다층 의미적 패치의 차별화를 통해 다양한 기상 조건에서도 견고하게 작동합니다.

- **Performance Highlights**: AllWeather-Net은 악천후 이미지를 햇살 좋은 날의 장면으로 효과적으로 변환하며, 성능 측면에서도 의미적 분할(semantic segmentation)에서 최대 5.3% mIoU 향상을 보여주고, 재훈련 없이도 이전 도메인에서 최대 3.9% mIoU 개선을 달성합니다.



### BEAVER: An Enterprise Benchmark for Text-to-SQL (https://arxiv.org/abs/2409.02038)
- **What's New**: 본 논문에서는 기존의 text-to-SQL 벤치마크가 공용 데이터에서 수집된 테이블을 사용하고, 주로 인간이 생성한 질문과 SQL 쌍으로 구성된 테스트 세트를 기반으로 하고 있다는 점을 강조하고 있습니다. 이에 반해, 기업 데이터 웨어하우스의 데이터를 이용한 새로운 데이터셋 BEAVER를 제안하며, 기존 LLM(large language models)의 성능 저하 문제를 다루고 있습니다.

- **Technical Details**: 이 논문은 BEAVER 데이터셋을 통해 LLM들이 기업 환경에서 text-to-SQL 작업을 수행할 때의 어려움을 보여줍니다. 기업 데이터가 일반적으로 '다크 웹'에 존재하고, 그 스키마가 복잡하기 때문에 기존 LLM들이 교육받지 않은 데이터에 대해 잘 수행하지 못한다는 점을 지적합니다. BEAVER 데이터셋은 실제 기업의 자연어 질문과 그에 대한 올바른 SQL 쿼리 쌍으로 구성되어 있습니다.

- **Performance Highlights**: 최근의 LLM들(GPT-4o 및 Llama3-70B-Instruct 등)을 BEAVER 데이터셋에서 평가한 결과, 이들 모델은 거의 0에 가까운 end-to-end 실행 정확도를 기록하며, 공용 데이터셋에서의 성능과 비교해 상당히 낮은 성능을 보였습니다. 이로 인해, 기존의 프로퍼텐셜(enterprise data)으로부터 학습한 LLM이 실제 데이터 웨어하우스의 작업에 적합하지 않음을 증명하였습니다.



### TransDAE: Dual Attention Mechanism in a Hierarchical Transformer for Efficient Medical Image Segmentation (https://arxiv.org/abs/2409.02018)
- **What's New**: TransDAE는 의료 이미지 분할을 위한 새로운 계층형 Transformer 모델로, 공간적 및 채널 간 연관성을 통합하는 이중 주의 메커니즘을 도입합니다. 이 방법은 기존 모델들의 한계를 극복하여 잦은 로컬 정보와 글로벌 의존성을 효과적으로 캡처합니다.

- **Technical Details**: TransDAE는 효율적인 자기 주의(self-attention) 및 향상된 자기 주의 메커니즘을 통합하여 고해상도의 의료 이미지를 효과적으로 모델링하면서 계산 복잡성을 줄입니다. 또한, Inter-Scale Interaction Module(ISIM)을 통해 스킵 연결 경로를 강화하여 특성 재사용을 촉진하고 위치 정확성을 개선합니다.

- **Performance Highlights**: TransDAE는 Synaps 다중 장기 데이터셋에서 기존 최첨단 방법들을 초월하는 성능을 발휘하며, 미리 훈련된 가중치를 사용하지 않고도 뛰어난 결과를 달성합니다.



### AI Governance in Higher Education: Case Studies of Guidance at Big Ten Universities (https://arxiv.org/abs/2409.02017)
- **What's New**: 높은 교육 서비스에서 Generative AI의 사용에 대한 새로운 관점을 제공합니다. 이 연구는 14개의 미국 유수 대학의 사례를 통해 AI 거버넌스(govemance) 전략과 그 특성을 분석합니다.

- **Technical Details**: Generative AI의 책임 있는 사용을 위한 거버넌스 전략은 다중 단위 거버넌스(multi-unit governance), 역할별 거버넌스(role-specific governance), 그리고 특이한 학문적 특성을 포함합니다. AI 거버넌스의 이론을 바탕으로 한 분석이 이루어졌습니다.

- **Performance Highlights**: 이 연구의 발견은 HEI 내에서 책임 있는 AI 사용을 위한 실질적인 지침을 제공하며, AI의 긍정적이고 유익한 활용을 위한 효과적인 정책과 가이드라인에 대한 필요성을 강조합니다.



### When Digital Twin Meets 6G: Concepts, Obstacles, and Research Prospects (https://arxiv.org/abs/2409.02008)
Comments:
          7 pages, 6 figures

- **What's New**: 이 논문은 디지털 트윈 기술과 emerging 6G 네트워크의 통합 가능성을 탐구하며, 이 두 기술의 결합에서 발생할 수 있는 다양한 도전과제를 제시하고 기본 원칙을 제안합니다.

- **Technical Details**: 논문에서 제안하는 Wireless Digital Twin Networks (WDTNs) 아키텍처는 Physical layer, Digital twin layer, Application layer의 삼층 구조로 구성됩니다. WDTNs는 6G 네트워크의 고속 데이터 전송과 실시간 동기화를 통해 다양한 응용 서비스를 지원합니다. 인공지능(Artificial Intelligence, AI)과의 통합을 통해 WDTNs는 스스로 학습하고 의사결정을 내릴 수 있는 능력을 가지며, 이를 통해 네트워크의 최적화와 자원 할당을 개선합니다.

- **Performance Highlights**: 제안된 WDTN 아키텍처는 에너지 효율성과 신뢰성을 개선하는 사례 연구를 통해 그 성능을 검증하였습니다. 이 연구는 DT의 배포와 동기화, 마이그레이션을 통합함으로써 장기적인 DT 운영을 효율적으로 달성할 수 있음을 강조합니다.



### vec2wav 2.0: Advancing Voice Conversion via Discrete Token Vocoders (https://arxiv.org/abs/2409.01995)
Comments:
          5 pages, 4 figures

- **What's New**: vec2wav 2.0는 음성 변환(VC)을 위한 새로운 speech discrete token vocoder로, 음성 자기 감독 모델에서 추출한 discrete tokens를 콘텐츠 특징으로 활용합니다. 이 모델은 WavLM의 음색 정보를 사용하여 음성 신호 재구성 과정에서 음색을 효과적으로 통합하고, 샘플링에 대한 새로운 adaptive Snake activation 함수를 제안합니다.

- **Technical Details**: vec2wav 2.0는 전방 생성 모듈을 활용한 prompted discrete token vocoder입니다. 여기서 음성 입력 tokens는 Conformer 기반 모듈로 처리되어 음성 재합성을 위한 timbre 정보를 제공합니다. Cross-attention 메커니즘을 통해 음색 정보가 전방 모듈에 통합되며, adaptive BigVGAN 생성기가 이를 바탕으로 waveforms을 생성합니다.

- **Performance Highlights**: 실험 결과, vec2wav 2.0는 모든 기준선에 비해 오디오 품질과 화자 유사성 측면에서 뛰어난 성능을 보이며, 영어 데이터 학습에도 불구하고 다국어 음성 변환에서도 우수한 성능을 발휘합니다. 이 모델은 음색 조절 능력을 강화하여 VC 및 음성 합성의 새로운 경계를 제시하고 있습니다.



### QueryCheetah: Fast Automated Discovery of Attribute Inference Attacks Against Query-Based Systems (https://arxiv.org/abs/2409.01992)
Comments:
          This is an extended version of the ACM CCS paper which includes appendices

- **What's New**: 이번 논문에서는 QueryCheetah라는 새로운 방법을 소개합니다. 이 방법은 Query-based systems (QBSs)에 대한 개인 정보 공격을 자동으로 빠르게 발견할 수 있습니다. QueryCheetah는 기존 방법보다 18배 빠르고 더욱 강력한 공격을 발견할 수 있습니다.

- **Technical Details**: QueryCheetah는 attribute inference attacks를 대상으로 하며, 기존의 방법들과 달리 쿼리 다중 집합 대신 단일 쿼리 다중 집합을 활용하여 더 빠르게 검색합니다. 각 반복이 이전 방법보다 450배 빠르며, 25배 더 많은 단계를 필요로 하면서도 전체적으로 18배 빨라집니다.

- **Performance Highlights**: QueryCheetah는 반자동 및 완전 자동 공격 발견 방법 모두에 비해 성능이 우수합니다. 또한 공격을 위해 많은 사용자에 대해 합리적인 시간 내에 취약점을 발견할 수 있는 가능성을 보여줍니다. 새로운 쿼리 구문에서의 취약점 발견과 발견된 공격을 저지하기 위해 개발된 방어책에 대한 우회 방법도 찾을 수 있습니다.



### Planning to avoid ambiguous states through Gaussian approximations to non-linear sensors in active inference agents (https://arxiv.org/abs/2409.01974)
Comments:
          13 pages, 3 figures. Accepted to the International Workshop on Active Inference 2024

- **What's New**: 이 논문에서는 비선형 측정 함수의 영향을 고려하여 관측 결과를 추론할 수 있는 방법을 제안합니다. 특히, 두 번째 차수의 Taylor 근사법을 활용하여 상태 의존적인 모호성 항을 도출하고 이를 통해 로봇 내비게이션 실험에서 에이전트가 경로를 계획하는 방식을 분석합니다.

- **Technical Details**: 비선형 관측 함수와 관련된 Gaussian 근사법을 분석하며, 첫 번째 차수 Taylor 근사 및 unscented transform과 같은 방법이 여러 상태에 대해 상수인 모호성 항을 유도하는 반면, 두 번째 차수 Taylor 근사는 비상수인 모호성 항을 유도함을 보여줍니다. 이로 인해 에이전트는 비선형 측정 함수가 강하게 휘어지는 상태를 피하게 됩니다.

- **Performance Highlights**: 로봇 내비게이션 실험을 통해, 에이전트가 주어진 목표 분포로 경로를 계획할 때 모호성 항이 어떻게 작용하는지를 시연하였으며, 이는 상태 추정을 어렵게 만드는 상태를 인지하고 회피하는 경향을 보여주었습니다.



### Exploiting the Vulnerability of Large Language Models via Defense-Aware Architectural Backdoor (https://arxiv.org/abs/2409.01952)
- **What's New**: 이번 논문에서는 백도어 공격(backdoor attack)의 새로운 유형을 도입하여 언어 모델(large language models) 아키텍처 안에 은닉된 구조적 백도어를 설계하는 방법을 제시합니다. 이는 기존의 데이터 포이즈닝(data poisoning) 방식과는 다른 차별화된 접근법입니다.

- **Technical Details**: 제안하는 기술은 입력 트리거 단어를 탐지하고, 특정 레이어에 가우시안 노이즈(Gaussian noise)를 주입하는 두 기능 모듈로 구성된 백도어 모듈을 사용합니다. 이 접근방식은 모델의 정확성을 기존의 데이터 학습 없이도 악화시키고, 기존의 방어 메커니즘도 회피할 수 있는 특징이 있습니다.

- **Performance Highlights**: 이 공격 방식은 다섯 개의 대형 언어 데이터 세트를 사용하여 평가되었으며, 기존 데이터 기반 백도어 공격 방식에 비해 방어 방법을 효과적으로 피할 수 있는 높은 공격 성공률을 기록했습니다. 특히, 정밀 튜닝(fine-tuning) 과정에도 불구하고 공격의 성과가 유지됨을 보였습니다.



### On the design space between molecular mechanics and machine learning force fields (https://arxiv.org/abs/2409.01931)
- **What's New**: 본 논문은 머신러닝 힘장(MLFF)과 분자역학(MM) 힘장 간의 속도-정확도 간의 트레이드오프를 탐색하고, 더욱 빠르고 안정적이며 일반화 가능한 MLFF의 설계에 대한 방향성을 제시합니다.

- **Technical Details**: 머신러닝 힘장(MLFF)은 분자 구조를 기반으로 하여 자동 미분을 통해 에너지와 힘을 예측하는 신경망 모델에서 발전하였습니다. 기존의 분자역학 힘장(MM)은 단순한 기능적 형태로 제한된 표현력을 가지고 있으나, 빠른 계산 속도를 자랑합니다. 알파벳 k로 표시되는 볼츠만 상수와 온도 T를 사용하여 다체 시스템의 에너지 장을 기술합니다. MLFF는 높은 정확도를 달성했지만, 여전히 MM에 비해 계산 속도가 더 느립니다.

- **Performance Highlights**: 최근 머신러닝 힘장 모델들은 1 kcal/mol의 화학적 정확도 기준을 초과하는 성능을 보이지만 여전히 MM 힘장보다 수백 배 느립니다. A100 GPU에서 MLFF는 에너지와 힘 평가에 대해 약 1 밀리초가 소요되고, 반면 MM은 0.005 밀리초로 더 빠릅니다. 이러한 성능 차이는 생물학적으로 중요한 시스템에 대한 이해를 위한 모델링의 실용성을 제한합니다.



### Comprehensive Equity Index (CEI): Definition and Application to Bias Evaluation in Biometrics (https://arxiv.org/abs/2409.01928)
Comments:
          Accepted paper for the 27th International Conference on Pattern Recognition (ICPR) 2024

- **What's New**: 이번 논문에서는 머신 러닝 모델의 편향된 행동을 정량화하기 위해 설계된 새로운 지표를 제안합니다. 이 지표는 스코어 분포간의 유사성을 측정하며, 이를 통해 전반적인 형태와 꼬리 확률을 균형 있게 평가합니다. 특히 인구 통계적 편향을 정량화하는 데 유용한 점에 주목하며, 얼굴 인식 시스템의 운영 평가에 적용했습니다.

- **Technical Details**: 제안된 Comprehensive Equity Index (CEI) 지표는 두 가지 접근 방식을 결합하여, 분포의 꼬리에서 발생하는 오류와 전반적인 분포의 형태를 고려합니다. 기존의 지표들이 제공하는 제한점을 극복하기 위해, 본 논문에서는 NIST FRVT 평가에서 사용되는 고성능 시스템과 현실적인 얼굴 데이터베이스를 포함하여 성능을 측정했습니다.

- **Performance Highlights**: CEI 지표는 두 개의 최첨단 모델과 네 가지 널리 사용되는 데이터베이스에 대해 테스트되었으며, 기존의 편향 지표의 주요 결점을 극복하는 능력을 보여주었습니다. 이를 통해, 고성능 시스템에서 편향 행동을 감지하고 정량화하는 데 있어 CEI의 유용성을 강조했습니다.



### GradINN: Gradient Informed Neural Network (https://arxiv.org/abs/2409.01914)
- **What's New**: 이번 논문에서는 물리 시스템의 근본적인 방정식이 완전히 알려져 있지 않거나 정의될 수 없는 다양한 물리 문제를 효율적으로 근사할 수 있는 Gradient Informed Neural Networks (GradINNs)를 제안합니다. GradINNs는 시스템의 기울기에 대한 사전 신념을 활용하여 예측 함수의 기울기를 모든 입력 차원에서 제한합니다.

- **Technical Details**: GradINNs는 목표 함수 모델링을 위한 기본 신경망과 사전 신념을 표현하는 보조 신경망 두 가지 신경망을 사용합니다. 맞춤형 손실 함수는 두 신경망을 동시에 훈련할 수 있도록 하여, 예측된 솔루션 기울기를 정규화합니다. 이 방식은 물리적 시스템을 더 효과적으로 모델링할 수 있게 합니다.

- **Performance Highlights**: 실험 결과, GradINNs는 다양한 비시간 의존 시스템(Friedman function, Stokes Flow) 및 시간 의존 시스템(Lotka-Volterra, Burger's equation)에서 표준 신경망 및 PINN과 같은 기존 접근 방식에 비해 강력한 성능을 보이며, 특히 낮은 데이터 환경에서도 더 나은 예측 성능을 보여줍니다.



### LUK: Empowering Log Understanding with Expert Knowledge from Large Language Models (https://arxiv.org/abs/2409.01909)
Comments:
          Under review

- **What's New**: 이 논문에서는 LLMs(대형 언어 모델)에서 전문 지식을 확보하여 작은 PLMs(사전 훈련된 언어 모델)의 로그 이해능력을 향상시키기 위한 새로운 지식 강화 프레임워크인 LUK를 소개합니다.

- **Technical Details**: LUK 프레임워크는 LLM에서 전문가 지식을 자동으로 획득하기 위한 다중 전문가 협업 프레임워크를 설계합니다. 이 프레임워크는 Director, Executor, Evaluator 역할로 구성된 전문가 팀을 구축하여, 역할 놀이를 통해 LLM이 태스크를 처리하도록 합니다. 또한 전문 지식을 활용하여 두 가지 신기술 사전 훈련 과제를 제안합니다: 단어 레벨 토큰 예측과 문장 레벨 의미 일치입니다.

- **Performance Highlights**: LUK는 다양한 로그 분석 작업에서 최신 연구 결과를 달성했으며, LLMs에서 얻은 지식을 효과적으로 활용하여 로그 이해도를 개선함을 입증합니다. 특히 낮은 자원 시나리오에서 뛰어난 일반화 및 내구성을 보여줍니다.



### 3D-LEX v1.0: 3D Lexicons for American Sign Language and Sign Language of the Netherlands (https://arxiv.org/abs/2409.01901)
- **What's New**: 본 연구는 3D에서 수화(SL)를 효율적으로 캡처하기 위한 접근법을 제시하고, 3D-LEX v1.0 데이터세트를 도입하며, 음성 특성을 반자동으로 주석 처리하는 방법을 세부적으로 설명합니다. 이 과정에서는 고해상도의 3D 포즈, 3D 손 모양, 깊이 인지 얼굴 특징을 포함한 세 가지 모션 캡처 기술이 통합되어 있습니다.

- **Technical Details**: 3D-LEX 데이터세트는 미국 수어(ASL)에서 1,000개의 수화와 네덜란드 수어(NGT)에서 1,000개의 수화로 구성되어 있습니다. 이 데이터세트는 수어의 수동 및 비수동 마커를 캡처하기 위한 두 가지 데이터 수집 기술과 비수동 마커를 캡처하는 세 번째 기술로 기록되었습니다. 손 모양 주석을 생성하기 위해 간단한 방법이 제시되었습니다.

- **Performance Highlights**: 주석을 통해 글로스 인식 정확도를 5% 향상시키고, 전문가 주석 대비 1% 향상된 결과를 보였습니다. 여기에 3D 모션 캡처 데이터는 수화 특징에 대한 심층 분석을 지원하고, 모든 시점에서 2D 프로젝션을 생성할 수 있습니다.



### What are the Essential Factors in Crafting Effective Long Context Multi-Hop Instruction Datasets? Insights and Best Practices (https://arxiv.org/abs/2409.01893)
Comments:
          Work in progress

- **What's New**: 이 논문에서는 최신 대형 언어 모델(LLM)을 위한 새로운 Multi-agent Interactive Multi-hop Generation (MIMG) 프레임워크를 소개하여, 질 높은 멀티 홉(multi-hop) 질문 생성 및 검증을 통해 긴 맥락(long-context) 작업을 개선합니다.

- **Technical Details**: MIMG 프레임워크는 네 가지 주요 구성 요소로 이루어져 있습니다: Quality Verification Agent, Single-hop Question Generation Agent, Multiple Question Sampling 전략, Multi-hop Question Merging Agent. 이 중 Quality Verification Agent는 생성된 질문과 답변의 품질을 자동으로 검증하고, Single-hop Question Generation Agent는 단일 질문을 생성합니다. 여러 질문 샘플링 전략을 통해 질문의 다양성을 높이고, Multi-hop Question Merging Agent는 단일 질문을 통합하여 의미가 있는 멀티 홉 질문을 생성합니다.

- **Performance Highlights**: 실험 결과, 85% 이상의 데이터가 멀티 홉, 고품질 및 비중복적이며, 제안된 방법론은 모델 성능을 평균 7.54% 향상시켜, 대량의 인간 주석 데이터를 기반으로 훈련된 모델보다 뛰어난 성과를 거두었습니다.



### CyberHost: Taming Audio-driven Avatar Diffusion Model with Region Codebook Attention (https://arxiv.org/abs/2409.01876)
- **What's New**: 이 논문에서는 오디오 기반의 1단계 인간 애니메이션 프레임워크인 CyberHost를 도입합니다. 이는 손의 완전성(hand integrity)과 신원 일관성(identity consistency), 자연스러운 움직임(natural motion)을 보장하며, 모든 인간 신체 부위의 애니메이션을 생성할 수 있습니다.

- **Technical Details**: CyberHost는 Region Codebook Attention 메커니즘을 채택하여 얼굴과 손의 애니메이션 품질을 향상시키고, 세부적인 지역 특징(local features) 및 학습된 움직임 패턴에 대한 선행 지식을 통합합니다. 또한, 신체 움직임 맵(body movement map), 손 선명도 점수(hand clarity score)와 같은 훈련 전략을 통해 생성 결과를 개선합니다.

- **Performance Highlights**: CyberHost는 정량적 및 정성적 측면 모두에서 기존 방법들보다 우수한 결과를 달성했습니다. 이 모델은 오디오 기반, 비디오 기반 및 다중 모달 드리븐 시나리오에서 뛰어난 성능을 보였으며, 오픈 세트 테스트 이미지에 대한 제로 샷(Zero-Shot) 비디오 생성이 가능합니다.



### Latent Distillation for Continual Object Detection at the Edg (https://arxiv.org/abs/2409.01872)
Comments:
          ECCV workshops, Computational Aspects of Deep Learning (CADL) 2024

- **What's New**: 이 논문에서는 객체 감지(Object Detection)에서의 지속 학습(Continual Learning, CL) 문제를 다루며, 에지 장치(edge devices)에서의 메모리 및 계산 제약을 해결하기 위해 NanoDet이라는 경량 모델을 사용하여 효율적인 업데이트 방법인 Latent Distillation(LD)를 제안합니다.

- **Technical Details**: 제안된 Latent Distillation(LD) 방법은 기존 CL 접근 방식보다 74%의 distillation 파라미터 오버헤드 감소 및 56%의 Floating Points Operations (FLOPs) 감소를 달성하였습니다. 또한 NanoDet는 1.2M의 파라미터로 구성되어 있으며 0.9G FLOPs의 추론을 요구합니다.

- **Performance Highlights**: 실험을 통해 LD 방법이 기존의 방법들보다 더 적은 계산 비용으로 우수한 감지 성능을 유지한다는 것을 밝혔습니다. CLOD 적용에서의 경량 모델의 가능성을 검증하여, 에지 장치에서 실용적인 성능을 demonstrated 하였습니다.



### Real-Time Indoor Object Detection based on hybrid CNN-Transformer Approach (https://arxiv.org/abs/2409.01871)
- **What's New**: 이 연구는 실내 환경에서의 실시간 객체 탐지의 정확도와 속도를 개선하기 위해 새로운 데이터셋과 CNN 기반 탐지 모델을 제안합니다. 이 모델은 내부 장면의 복잡함 속에서 중요한 특징을 구별할 수 있는 주의 메커니즘을 통합하였습니다.

- **Technical Details**: 제안된 CNN 모델은 OpenImages v7에서 파생된 32개의 실내 카테고리를 포함하는 데이터셋을 기반으로 하며, 변형을 통해 효율성을 극대화했습니다. 또한, 하이브리드 아키텍처가 CNN의 강점과 transformer의 공간적 추론 능력을 결합해 실시간 처리에 최적화되어 있습니다.

- **Performance Highlights**: 본 연구는 기존의 최첨단 모델들과 비교해도 정확도와 속도 면에서 경쟁력이 있으며, 실내 환경에서의 객체 탐지 연구와 다양한 애플리케이션의 새로운 가능성을 엽니다.



### The Role of Large Language Models in Musicology: Are We Ready to Trust the Machines? (https://arxiv.org/abs/2409.01864)
- **What's New**: 이 논문은 음악학(musicology) 분야에서 대형 언어 모델(Large Language Models, LLMs)의 활용 가능성과 신뢰성을 탐구합니다. 연구자들은 LLM의 현황, 우려 사항 및 전문가와의 논의를 통해 이 기술의 aceptación(acceptance)을 평가했습니다. LLM의 신뢰성을 높이기 위해, retrieval-augmented generation 모델을 활용하여 초기 벤치마크를 반자동으로 생성하는 방법을 제안하고, human experts에 의해 검증된 400개의 질문 데이터를 분석하였습니다.

- **Technical Details**: 본 연구는 LLM의 음악학 관련 작업에서의 도메인 전문성을 측정하기 위한 방법론을 제안합니다. 특히, The New Grove Dictionary of Music and Musicians를 기반으로 한 Multiple-Choice Question Generation 접근 방식을 채택하여, 음악 관련 주제에 대한 400개의 질문-답변 쌍을 생성하고 검증하였습니다. 이를 통해 LLM의 음악학 관련 생성 능력을 평가하고, Hallucination(환각) 문제의 정도를 측정합니다.

- **Performance Highlights**: 최종적으로, 생성된 질문의 정확도는 67.4%로 나타났으며, 이는 LLM이 음악학 분야에서 신뢰할 수 있는 텍스트를 생성하는 데 한계가 있음을 보여줍니다. 연구 결과, 대부분의 전문가들은 LLM이 음악학 분야에서 중요한 혁신을 일으킬 가능성이 있다고 보고하였으나, 현재의 LLM 사용 빈도와 신뢰도는 여전히 낮은 상황임을 지적하였습니다.



### Dialogue You Can Trust: Human and AI Perspectives on Generated Conversations (https://arxiv.org/abs/2409.01808)
Comments:
          17 pages, 15 figures, shorter version submitted to 22nd Annual Workshop of the Australasian Language Technology Association (ALTA'24)

- **What's New**: 이 연구는 대화 시스템 및 챗봇의 평가 방법에 대한 새로운 통찰을 제공합니다. AI와 인간의 평가를 비교하여 향상된 대화 평가 지표를 제안합니다.

- **Technical Details**: 본 연구에서는 7가지 주요 성능 지표(KPIs)인 Coherence, Innovation, Concreteness, Goal Contribution, Commonsense Contradiction, Incorrect Fact, Redundancy를 측정하는 실험을 수행했습니다. GPT-4o API를 활용하여 다양한 대화 데이터셋을 생성하고, 두 가지 실험 분석을 통해 인간과 AI의 평가 결과를 비교했습니다.

- **Performance Highlights**: 실험 결과, GPT 모델이 인간의 판단과 밀접하게 일치하며, 특히 사실 accuracy 및 commonsense reasoning에서 우수한 성능을 보여주었습니다. 그러나 Redundancy와 자기 모순 감소에는 여전히 어려움이 있는 것으로 나타났습니다.



### Training on the Benchmark Is Not All You Need (https://arxiv.org/abs/2409.01790)
- **What's New**: 이 논문에서는 Large Language Models (LLMs)의 데이터 누출(data leakage) 문제를 해결하기 위한 새로운 기법을 제안합니다. 다중 선택 형식의 평가 벤치마크를 기반으로 하여, 개별 선택지의 순서를 뒤섞는 방식으로 데이터 세트를 생성하고, 모델의 로그 확률 분포(log probability distribution)를 분석함으로써 데이터 누출을 감지합니다.

- **Technical Details**: 이 방법은 블랙박스(black-box) 환경에서도 작동하며, 모델의 교육 데이터(training data)나 가중치(weights)에 접근하지 않고도 데이터 누출 여부를 판단할 수 있습니다. 주어진 데이터의 로그 확률이 여러 개 고르지 않게 나타나면 데이터 누출을 나타내며, 이는 원래의 평가 데이터가 모델의 사전 훈련 데이터(pre-training data)에서 유출되었음을 시사합니다.

- **Performance Highlights**: 두 개의 LLM을 통한 실험을 통해 제안된 방법의 효과성을 입증하였으며, 31개의 오픈소스 LLM의 데이터 누출 위험을 4개의 벤치마크 데이터 세트에서 평가한 결과, Qwen 계열의 LLM이 여러 벤치마크에서 가장 높은 데이터 누출 위험성을 보였습니다.



### Empirical evidence of Large Language Model's influence on human spoken communication (https://arxiv.org/abs/2409.01754)
- **What's New**: 이번 연구에서는 ChatGPT와 같은 Large Language Models (LLMs)이 인간의 언어 사용에 미치는 영향을 밝혀냄으로써, 인간 언어의 변화가 AI의 발전에 따라 어떻게 전개되는지를 탐구했습니다. 특히, 280,000개의 영어 발표 및 강연 비디오를 분석하여, ChatGPT의 출시 이후 특정 단어 사용에서 유의미한 변화가 나타났다는 것을 발견했습니다.

- **Technical Details**: 연구는 20,000개의 학술 YouTube 채널에서 약 280,000개의 비디오 전사본을 분석하였으며, ChatGPT의 출시에 따른 단어 사용 추세 변화와 해당 단어 사용의 상관관계를 집중적으로 살펴보았습니다. 주요 연구 결과로는, ChatGPT 출시 후 'delve', 'realm', 'meticulous', 'adept'와 같은 특정 단어의 사용 빈도가 각각 48%, 35%, 40%, 51% 증가한 점이 포함됩니다. 이러한 결과는 LLM이 인간의 언어 패턴에의 영향을 미치고 있다는 것을 증명합니다.

- **Performance Highlights**: 이 연구는 LLM이 인간의 말하기 언어에 영향을 미친 첫 번째 경험적 증거를 제공하며, 이는 사회적, 정책적 우려를 야기합니다. AI가 언어적 다양성을 감소시킬 위험이나 대규모 조작을 위한 악용 가능성 등을 강조하고 있으며, 머신 행동과 인간 문화 간의 피드백 루프에 대한 추가 연구의 필요성을 역설하고 있습니다.



### Interpreting Outliers in Time Series Data through Decoding Autoencoder (https://arxiv.org/abs/2409.01713)
Comments:
          14 pages, 8 figures, accepted at TempXAI @ ECML-PKDD

- **What's New**: 이번 연구는 독일 자동차 공급 산업의 제조 시계열 데이터를 활용하여 이상치 탐지(Outlier Detection) 분야에서 이해 가능한 인공지능(Explainable AI, XAI) 기법을 적용하는 데 중점을 두고 있습니다. 특히, 일반적인 신경망 모델들이 안전-critical 시스템에서 적용될 때 투명성과 신뢰성을 높이기 위한 방법론이 제시되었습니다. 또한, 다수의 XAI 기술들을 통합한 새로운 접근법인 Aggregated Explanatory Ensemble (AEE)을 제안하며, 이를 통해 복잡한 이상치 해석 작업을 보다 효과적으로 수행할 수 있도록 하였습니다.

- **Technical Details**: 연구에서는 변환(autoencoder) 및 흐름 기반의 이상치 탐지 방법론에 집중하며, Convolutional Autoencoder (CAE)를 사용하여 시계열 데이터를 압축하고, 잠재 공간(latent space)에서 이상치를 탐지합니다. 여러 XAI 기법(Grad-CAM, LIME, SHAP, LRP)을 활용하여 인코더의 출력을 기반으로 하는 설명을 생성합니다. AEE 접근법은 이러한 다양한 설명을 하나의 총체적 해석으로 융합하여 보다 명확한 통찰을 제공합니다. 이 연구에서 제안된 품질 측정 기법(QM)을 통해 설명의 질을 정량적으로 평가하였습니다.

- **Performance Highlights**: 기존의 XAI 기법들이 제공하는 다양한 통찰을 결합함으로써, AEE는 이상치 탐지의 해석에 기여하는 효과적인 방법론으로 자리잡았습니다. 연구 결과, AEE 기법은 변수 간 상호작용을 보다 잘 이해할 수 있도록 돕고, 시계열 데이터 내에서의 차별적인 요소를 강조하여 이상치 탐지의 정확성을 높이는 데 기여하게 됩니다.



### USTC-KXDIGIT System Description for ASVspoof5 Challeng (https://arxiv.org/abs/2409.01695)
Comments:
          ASVspoof5 workshop paper

- **What's New**: 이번 연구는 ASVspoof5 챌린지에 제출된 USTC-KXDIGIT 시스템을 다룹니다. 이 시스템은 음성 딥페이크 감지(Track 1)와 스푸핑에 강한 자동 화자 인증(SASV, Track 2) 두 가지 트랙에서 기술적 우수성을 보여줍니다. 특히, 다양한 기술적 요구를 충족시키기 위해서 전방(feature extractor)과 후방(classifier) 구조를 가진 시스템을 제안합니다.

- **Technical Details**: 우리 시스템은 자기 지도 학습(Self-Supervised Learning)을 바탕으로 하 handcrafted feature와 speech representation의 조합으로 설계되었습니다. 여러 적대적 조건에서 스푸핑 공격을 감지하기 위해 증강된 훈련 세트에서 여러 시스템을 훈련했습니다. 또한, 음성 변환(Voice Conversion) 기술을 사용하여 진짜 음성에서 가짜 음성을 합성하여 훈련 알고리즘의 다양성을 높였습니다. 모델 아키텍처의 상호 보완적 정보들을 활용하기 위해 activation ensemble과 다양한 시스템의 점수를 융합하는 방법을 적용했습니다.

- **Performance Highlights**: 제안된 시스템은 closed condition에서 0.3948 minDCF 및 14.33% EER, open condition에서 0.0750 minDCF 및 2.59% EER을 달성하여 적대적 조건에서의 강력한 성능을 입증했습니다. Track 2에서는 CNN 기반 ASV 시스템과의 융합을 통해 closed condition에서 0.2814 min-aDCF, open condition에서 0.0756 min-aDCF의 우수한 성능을 보여주었습니다.



### Differentially Private Kernel Density Estimation (https://arxiv.org/abs/2409.01688)
- **What's New**: 본 논문에서는 개선된 차별적 프라이버시(Differential Privacy, DP) 데이터 구조를 제안하여 커널 밀도 추정(Kernel Density Estimation, KDE)의 프라이버시-유틸리티(utility) 트레이드오프를 향상시키고 이전 결과에 비해 효율성을 높였습니다.

- **Technical Details**: 본 연구는 주어진 유사성 함수 f와 프라이빗 데이터셋 X에 대해, 모든 쿼리 y에 대해 DP 방식으로 \sum_{x \in X} f(x, y)를 근사하도록 프리프로세스(preprocess)하는 알고리즘을 다룹니다. 이전에 최적의 알고리즘은 O(nd) 공간과 시간을 요구했으며, 쿼리 시간은 d \log n 입니다. 본 논문에서는 쿼리 시간을 \alpha^{-1} \log n로 줄이고, 근사 비율을 \alpha에서 1로 개선하며, 오류 의존도를 \alpha^{-0.5} 만큼 줄였습니다.

- **Performance Highlights**: 새롭게 제안된 알고리즘은 쿼리 시간 단축, 향상된 근사 비율, 줄어든 오류 의존성을 통해 차별적 프라이버시를 유지하면서도 이전 알고리즘보다 뛰어난 성능을 보입니다.



### Adaptive Explicit Knowledge Transfer for Knowledge Distillation (https://arxiv.org/abs/2409.01679)
Comments:
          19 pages, 5 figures

- **What's New**: 이번 연구에서는 logit 기반 지식 증류(knowledge distillation, KD) 방법의 성능을 향상시키기 위해, teacher 모델의 비대상 클래스에 대한 확률 분포(즉, 'implicit (dark) knowledge')를 student 모델에 효과적으로 전달하는 방법을 제시합니다.

- **Technical Details**: 연구에서는 gradient 분석을 통해 implicit knowledge의 학습 조절이 가능함을 보여주고, explicit knowledge(즉, teacher가 대상 클래스에 대해 가지는 확신)를 함께 학습할 수 있도록 새로운 loss 함수를 제안합니다. 또한, 효과적인 증류 및 클래스 간 관계 모델링을 위해 분류(classification)와 증류(distillation) 작업을 분리할 것을 제안합니다.

- **Performance Highlights**: 제안된 adaptive explicit knowledge transfer (AEKT) 방법은 CIFAR-100 및 ImageNet 데이터셋에서 기존의 최첨단 KD 방법들에 비해 개선된 성능을 달성했습니다.



### Classifier-Free Diffusion-Based Weakly-Supervised Approach for Health Indicator Derivation in Rotating Machines: Advancing Early Fault Detection and Condition Monitoring (https://arxiv.org/abs/2409.01676)
- **What's New**: 이 논문에서는 회전 기계의 상태 지표(Health Indicator, HI)를 도출하기 위해 전이 기반의 약한 감독 방법을 제안합니다. 이 방법은 건강한 샘플과 몇 가지 이상치를 사용해 훈련된 분류기 없는 확산 모델을 활용하여 조기 결함 감지 및 지속적인 상태 모니터링을 가능하게 합니다.

- **Technical Details**: 제안된 모델은 건강한 샘플을 생성하고, 원본 샘플과 생성된 샘플 간의 차이를 비교하여 이상 맵(anomaly map)을 구축합니다. 이 이상 맵은 결함의 특성 주파수를 드러내어 헬스 모니터링을 위한 HI를 설정하는 데 사용됩니다. 추가로, 이 방법은 샘플 전체 분포가 아닌 결함 특성의 차이만을 기반으로 HI를 구성하므로 설명 가능성을 높이고 노이즈 간섭을 완화합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 다양한 신호 대 잡음 비율(signal-to-noise ratio)에서 노이즈 간섭에 대한 강건성을 보여주며, 회전 기계의 건강 상태 모니터링 및 조기 결함 감지에서 기존 최첨단 방법에 비해 효율성을 발휘했습니다.



### Enhancing Fine-Grained Visual Recognition in the Low-Data Regime Through Feature Magnitude Regularization (https://arxiv.org/abs/2409.01672)
- **What's New**: 작지만 강력한 데이터로 훈련된 Fine-Grained Visual Recognition (FGVR) 모델의 성능을 크게 향상시키는 간단한 규제 기술인 Feature Magnitude Regularization (FMR)을 도입했습니다. 이 기법은 특성의 크기 분포를 평등하게 만드는 정규화로, 사전 훈련된 모델에서의 편향을 제거하는 것을 목표로 합니다.

- **Technical Details**: FGVR은 이미지 내의 미세한 차이를 감지하여 대분류를 세부 분류하는 문제로, 데이터가 제한적일 때 품질 높은 특성 표현을 활용하는 것이 중요합니다. 본 연구에서는 정규화의 강도를 다이나믹하게 조정할 수 있는 역량 메커니즘을 개발하여, 학습 과정 중 특성 크기 분포의 균일성을 높이고, 잠재적 편향을 줄이는 방법을 제시합니다.

- **Performance Highlights**: 실험 결과, 제안한 FMR 메커니즘이 기존의 파인 튜닝 기법보다 향상된 성능을 나타내었으며, 제한된 데이터셋에서의 FGVR 성능을 효과적으로 개선했습니다.



### Pureformer-VC: Non-parallel One-Shot Voice Conversion with Pure Transformer Blocks and Triplet Discriminative Training (https://arxiv.org/abs/2409.01668)
Comments:
          submmited to ICASSP 2025

- **What's New**: 최근에 제안된 Pureformer-VC는 단 하나의 음성 샘플을 사용하여 소스 음성을 목표 화자의 음색에 맞게 변환하는 혁신적인 접근법이다. 이 모델은 Conformer 블록을 활용하여 분리된 인코더를 구축하고, Zipformer 블록을 이용해 스타일 전송 디코더를 구현하였다.

- **Technical Details**: Pureformer-VC 프레임워크는 콘텐츠 인코더, 화자 인코더, 디코더 및 vocoder로 구성된다. 이 모델은 generative VAE 손실과 triplet 손실을 사용하여 구성 요소를 인코딩하고 비지도 학습을 통해 모델을 훈련시킨다. Zipformer에서는 styleformer 메커니즘을 사용하여 생성된 음성에 화자 특성을 효과적으로 통합한다.

- **Performance Highlights**: 실험 결과, Pureformer-VC는 기존 방법에 비해 주관적 점수가 비슷하거나 개선된 객관적 메트릭을 달성하였다. 이는 단일 음성 변환 시나리오에서 매우 경쟁력 있는 성능을 보여준다.



### ReKep: Spatio-Temporal Reasoning of Relational Keypoint Constraints for Robotic Manipulation (https://arxiv.org/abs/2409.01652)
- **What's New**: 본 연구에서는 로봇 조작 작업을 관계형 키포인트 제약(Relational Keypoint Constraints, ReKep)으로 표현하는 새로운 방식을 제안합니다. 이 기법은 로봇과 환경 간의 관계를 기반으로 하여, 다양한 작업에 대해 자동화된 제약을 생성할 수 있습니다.

- **Technical Details**: ReKep는 환경 내 3D 키포인트 집합을 수치적 비용으로 매핑하는 Python 함수로 표현됩니다. 이를 통해 조작 작업을 순차적으로 표현하고, 인식-행동 루프를 통해 로봇 행동을 실시간으로 최적화할 수 있습니다. 또한, 대형 비전 모델(Large Vision Models)과 비전-언어 모델(Vision-Language Models)을 활용하여 새로운 작업의 ReKep를 자동으로 생성합니다.

- **Performance Highlights**: 우리는 로봇 플랫폼에서 제안된 ReKep 기반의 시스템 구현을 시연하였으며, 다단계, 실환경, 양손, 반응형 행동을 포함하는 다양한 조작 작업을 수행할 수 있었습니다. 이는 특정 작업 데이터나 환경 모델 없이도 가능하였으며, 작업당 약 10Hz의 속도로 안정적으로 해결할 수 있었습니다.



### PMLBmini: A Tabular Classification Benchmark Suite for Data-Scarce Applications (https://arxiv.org/abs/2409.01635)
Comments:
          AutoML 2024 Workshop Track

- **What's New**: PMLBmini라는 새로운 벤치마크 스위트를 소개합니다. 이 스위트는 샘플 크기가 500 이하인 44개의 이진 분류 데이터셋으로 구성되어 있으며, 데이터가 부족한 환경에서 기계 학습 방법의 성능을 비교할 수 있도록 설계되었습니다.

- **Technical Details**: 기존의 자동 기계 학습(Automated Machine Learning, AutoML) 프레임워크와 심층 신경망 및 고전적인 선형 모델을 PMLBmini를 사용하여 철저히 평가하였습니다. 분석 결과, 최신 AutoML 및 심층 학습 접근법이 단순한 로지스틱 회귀(lodistic regression) 베이스라인과 상당한 성능 차이를 보이지 않았지만, 특정 조건에서 AutoML과 심층 학습 방법의 적용이 합리적일 수 있음을 확인하였습니다.

- **Performance Highlights**: 55%의 데이터셋에서 로지스틱 회귀가 AutoML 및 심층 학습 접근법과 유사한 성능을 보였으며, 데이터가 부족한 상황에서 최적의 하이퍼파라미터를 제시하였습니다. 연구자들은 PMLBmini를 활용하여 그들의 방법을 평가하고 데이터 효율성을 분석할 수 있습니다.



### Dreaming is All You Need (https://arxiv.org/abs/2409.01633)
- **What's New**: 이 연구에서는 탐색(exploration)과 정확도(precision) 사이의 조화를 이루기 위해 SleepNet과 DreamNet이라는 두 가지 새로운 심층 학습 모델을 소개합니다. SleepNet은 미리 학습된 인코더 모델을 사용하여 감독 학습(supervised learning)과 비감독 "수면(sleep)" 단계가 원활하게 통합되며, DreamNet은 숨겨진 상태를 재구성하여 인간의 꿈을 흉내냅니다.

- **Technical Details**: SleepNet은 비감독 특징을 포함한 전용 뉴런을 갖추고, 탐색 학습을 촉진하는 간헐적인 "수면 블록(sleep blocks)"을 형성합니다. 반면, DreamNet은 전체 인코더-디코더 프레임워크를 사용하여 숨겨진 상태를 재구성하며, 이는 학습된 표현을 탐색하고 다듬는 데 도움을 줍니다. 이 두 모델은 컴퓨터 비전(computer vision)과 자연어 처리(natural language processing) 모두에 적용할 수 있는 일반적인 원칙을 가지고 있습니다.

- **Performance Highlights**: SleepNet과 DreamNet은 다양한 이미지 및 텍스트 데이터셋에서 extensive empirical evaluations를 통해 최신 기술(state-of-the-art) 모델들과 비교하여 우수한 성능을 입증했습니다. 특히 DreamNet은 이미지 및 텍스트 분류 작업에서 최첨단 기준을 일관되게 초과하는 성능을 보여주었습니다.



### SafeEmbodAI: a Safety Framework for Mobile Robots in Embodied AI Systems (https://arxiv.org/abs/2409.01630)
- **What's New**: 이 논문에서는 Embodied AI 시스템에 통합된 모바일 로봇의 안전성을 높이기 위한 새로운 프레임워크 SafeEmbodAI를 제안합니다. 이 프레임워크는 복잡한 언어 명령을 이해하고 다양한 작업을 수행하는 능력을 향상시키는 Large Language Models (LLMs)의 사용에 따른 안전 문제를 해결하는 데 중점을 두고 있습니다.

- **Technical Details**: SafeEmbodAI는 안전한 prompting, 상태 관리(state management), 그리고 안전성 검증(safety validation) 메커니즘을 통합하여 다중 모드 데이터를 처리하고 안전성을 보장합니다. 실제 실험 환경에서 SafeEmbodAI는 LLM이 활용되는 모바일 로봇의 내비게이션 작업에서 악의적인 공격으로부터 효과적으로 방어할 수 있는 능력을 평가하였습니다.

- **Performance Highlights**: 각종 환경 설정에서의 검토 결과, SafeEmbodAI는 악의적인 명령으로부터의 위협을 효과적으로 완화하고 성능을 개선하였습니다. 공격 시나리오에서, 복잡한 장애물이 있는 환경에서 본 방법은 기준 모델에 비해 267%의 성능 증가를 보여 안전한 Embodied AI 시스템의 중요성을 강조합니다.



### T1-contrast Enhanced MRI Generation from Multi-parametric MRI for Glioma Patients with Latent Tumor Conditioning (https://arxiv.org/abs/2409.01622)
Comments:
          arXiv admin note: text overlap with arXiv:2407.02616

- **What's New**: 이번 연구에서는 Gadolinium 기반 조영제 (GBCA)를 사용하지 않고도 고품질의 T1C MRI 이미지를 생성하는 심층 학습 프레임워크를 개발하였습니다. 이 방법은 종양을 인식할 수 있는 시각 변환기 (Vision Transformer) 모델인 TA-ViT를 활용하여 이루어졌습니다.

- **Technical Details**: TA-ViT 모델은 예상 분할 맵에서 변환기 층을 조정하는 적응형 레이어 노름 제로 메커니즘을 통해 종양 영역의 예측을 크게 향상시킵니다. 또한, MPR-ViT 모델을 사용하여 생성된 분할 맵을 잠재 공간으로 변환하여 압축된 특징적 표현을 형성합니다.

- **Performance Highlights**: TA-ViT 모델의 성능은 기존의 MRP-ViT 모델에 비해 질적 및 양적 모두에서 우수성을 나타냈으며, 일반적인 조직 및 종양 영역에서 각각 NMSE, PSNR 및 NCC 지표가 크게 향상되었습니다. 이 방법으로 생성된 T1C 이미지는 실제 T1C 이미지와 매우 유사하며, 향후 GBCA 독성 위험을 없애고 MRI 스캔 프로토콜을 간소화할 수 있는 가능성을 제공합니다.



### Decompose the model: Mechanistic interpretability in image models with Generalized Integrated Gradients (GIG) (https://arxiv.org/abs/2409.01610)
- **What's New**: 이번 연구에서는 eXplainable AI (XAI) 분야에서 이미지 모델의 메커니즘 해석 가능성(mechanistic interpretability)을 향상시키기 위한 새로운 접근법을 제안합니다. 기존의 클래스별 해석(class-specific interpretations)에서 벗어나 입력(input)부터 최종 출력(output)까지의 모든 중간 레이어를 체계적으로 추적(tracing)하는 방법을 개발하였습니다.

- **Technical Details**: 우리는 모델 임베딩을 해석 가능한 개념 벡터(Concept Vectors)로 분해하기 위해 포인트 단위 특성 벡터(Pointwise Feature Vectors, PFVs)와 효과적인 수용 영역(Effective Receptive Fields, ERFs)을 활용합니다. 일반화된 적분 그래디언트(Generalized Integrated Gradients, GIG)를 통해 개념 벡터 간의 상관성을 계산하고, 데이터셋 전체에 대한 포괄적인 모델 행동 분석을 수행합니다. 이 방법을 통해 ResNet50 모델의 구조를 심층적으로 분석합니다.

- **Performance Highlights**: 제안된 방법은 정성적(qualitative) 및 정량적(quantitative) 평가를 통해 검증되었으며, 이미지 모델의 의미적 중요성(semantic significance)에 대한 깊은 이해를 제공합니다. 이를 통해 우리는 모델의 기능적 메커니즘을 종합적으로 설명할 수 있으며, 기존의 클래스 기반 접근법의 한계를 넘어서 전체 데이터셋에 대한 해석 가능성을 확장합니다.



### Laser: Parameter-Efficient LLM Bi-Tuning for Sequential Recommendation with Collaborative Information (https://arxiv.org/abs/2409.01605)
Comments:
          11 pages, 4 figures

- **What's New**: 본 논문에서는 협업 정보를 활용한 순차 추천 시스템을 위한 효율적인 매개변수 처리 방법인 Laser를 제안합니다. 이는 LLM의 매개변수를 동결시키고 가변적인 가상 토큰을 입력 시퀀스의 앞과 뒤에 삽입하여 최적화하는 방안입니다.

- **Technical Details**: Bi-Tuning 프레임워크는 두 개의 학습 가능한 가상 토큰을 사용하여 입력 시퀀스의 상단과 하단에 배치합니다. 상단은 사용자-아이템 협업 정보를 포함하고, 하단은 LLM의 출력 임베딩을 추천 공간으로 변환합니다. M-Former를 도입하여 다양한 사용자 특성을 반영한 협업 정보를 효과적으로 통합합니다.

- **Performance Highlights**: 실제 데이터셋에 대한 광범위한 실험 결과, Laser는 기존의 최첨단 방법들보다 확연히 높은 추천 정확도를 달성하며, 매개변수 효율성을 입증합니다.



### A Time-Intensity Aware Pipeline for Generating Late-Stage Breast DCE-MRI using Generative Adversarial Models (https://arxiv.org/abs/2409.01596)
- **What's New**: 이 논문은 초기 대비 종합적인 파이프라인을 제안하여 조기 대비 정확한 장기(늦은) 대조증강 유방 자기공명영상(MRI)을 생성하는 방법을 다룹니다. 이 전략은 대조제 패턴을 유지하면서 합성된 이미지의 시각적 속성을 보존하는데 초점을 맞추고 있습니다.

- **Technical Details**: 논문에서는 Time-Intensity (TI) 곡선에 기반한 새로운 손실 함수(TI-Loss)를 제안하여 픽셀에 대한 주의 집중(pixel-attention) 기반의 생성 모델을 최적화합니다. 또한, 기존의 표준화 방법과는 달리 여러 시간대의 이미지 시퀀스에서 대조 증강 패턴을 유지하는 새로운 정규화 전략(TI-norm)을 개발했습니다.

- **Performance Highlights**: 실험 결과는 제안된 전략이 대조 증강된 지역의 진단 품질을 유의미하게 향상시킨 이미지를 생성하면서, 전체 이미지의 공간적 특징을 잘 유지하고 있음을 보여줍니다. 이는 임상 시나리오에서 심층 학습을 통해 생성된 합성 Late Enhanced 이미지를 사용할 가능성을 시사합니다.



### Booster: Tackling Harmful Fine-tuing for Large Language Models via Attenuating Harmful Perturbation (https://arxiv.org/abs/2409.01586)
- **What's New**: 본 논문에서는 유해한 파라미터 조정(harmful fine-tuning) 문제의 근본 원인이 모델 가중치에 대한 유해한 섭동(harmful perturbation)이라고 제안하며, 이를 해결하기 위한 새로운 접근법인 Booster를 제안합니다.

- **Technical Details**: Booster는 원래의 alignment loss에 손실 정규화기(loss regularizer)를 추가하여 유해한 섭동의 부정적인 영향을 완화합니다. 최적화 과정에서 이러한 정규화기를 포함시켜 손실 감소율(harmful loss reduction rate)을 조절하고, 이를 통해 모델의 안전성을 향상시킵니다.

- **Performance Highlights**: Booster는 기존의 Vaccine과 RepNoise보다 각각 최대 17.26% 및 20.08%의 평균 유해 점수를 줄이면서도 downstream task의 성능을 유지하는 것으로 나타났습니다.



### GaussianPU: A Hybrid 2D-3D Upsampling Framework for Enhancing Color Point Clouds via 3D Gaussian Splatting (https://arxiv.org/abs/2409.01581)
Comments:
          7 pages, 5 figures

- **What's New**: 본 논문에서는 로봇 인식 분야를 위한 새로운 2D-3D 하이브리드 콜로리드 포인트 클라우드 업샘플링 프레임워크(GaussianPU)를 제안합니다. 이 방법은 3D Gaussian Splatting(3DGS)을 활용하여 3D 포인트 클라우드와 로봇 비전 시스템의 2D 렌더링 이미지 간의 연결을 제공합니다.

- **Technical Details**: 제안된 방법은 듀얼 스케일 렌더링 복원 네트워크를 통해 희소한 포인트 클라우드 렌더링을 밀집 표현으로 변환하며, 이 과정에서 로봇 카메라의 정확한 포즈와 보간된 희소한 포인트 클라우드를 이용하여 밀집 3D 포인트 클라우드를 재구성합니다. 개선된 3DGS는 포인트 수에 대한 정밀한 제어를 가능하게 하여 로봇 장면 이해를 위한 업샘플링된 포인트 클라우드의 품질을 크게 향상시킵니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, 제안된 방식은 색상 포인트 클라우드의 품질을 크게 향상시키고, 대규모 포인트 클라우드와 관련된 자율 로봇 및 인간-로봇 상호작용 시나리오에서의 응용 가능성을 입증합니다. 또한, 이 프레임워크는 단일 소비자 등급 GPU에서 전체 포인트 클라우드를 처리할 수 있어 세분화의 필요성을 제거하고, 수백만 개의 포인트를 포함한 고품질의 밀집 컬러 포인트 클라우드를 제공합니다.



### AdaComp: Extractive Context Compression with Adaptive Predictor for Retrieval-Augmented Large Language Models (https://arxiv.org/abs/2409.01579)
Comments:
          8 pages, 5 figures, code available at https://anonymous.4open.science/r/AdaComp-8C0C/

- **What's New**: 이번 논문에서는 AdaComp이라는 새로운 저비용의 추출적(context compression) 문서 압축 방법을 제안합니다. 이 방법은 쿼리의 복잡도(query complexity)와 검색의 질(retrieval quality)에 기반하여 압축 비율(compression rate)을 적절히 결정합니다.

- **Technical Details**: AdaComp는 RAG 시스템이 정확한 응답을 하기 위해 필요한 최소 top-k 문서를 주석으로 달아 압축 비율(compression rate)로 설정합니다. 이를 통해 쿼리, 검색된 문서, 압축 비율의 삼중 항(triplet)을 구성하고, 이 데이터셋을 기반으로 압축 비율 예측 기계(compression-rate predictor)를 훈련합니다. 정보 필터링 과정에서는 예측된 압축 비율에 따라 중요한 문서들만을 선택합니다.

- **Performance Highlights**: 세 가지 QA 데이터셋과 하나의 대화형 Multi-doc QA 데이터셋에서 실험한 결과, AdaComp는 성능을 거의 동일하게 유지하면서도 인퍼런스(inference) 비용을 크게 줄이는 데 성공하여 효율성과 성능 간의 균형을 이루었습니다.



### Improving Apple Object Detection with Occlusion-Enhanced Distillation (https://arxiv.org/abs/2409.01573)
- **What's New**: 이 논문에서는 자연 환경에서 자주 겪는 occlusion(가림현상)이 있는 사과 탐지의 어려움을 해결하기 위해 'Occlusion-Enhanced Distillation'(OED)라는 새로운 기술을 소개합니다. 이 방법은 가림 정보를 활용하여 occluded datasets(가림 데이터셋)의 의미적으로 정렬된 특징 학습을 정규화하고, Exponential Moving Average (EMA)를 통해 학습의 안정성을 높입니다.

- **Technical Details**: OED는 Grounding DINO와 SAM 방법을 활용하여 자연스러운 성장 상태를 반영하는 가림 예제를 만드는 occlusion-enhanced dataset(가림 강화 데이터셋)을 설계합니다. 다중 스케일 지식 증류 전략을 채택하여, 학생 네트워크가 가림 요소가 많아진 이미지를 입력으로 사용하고, 교사 네트워크는 자연적인 가림이 없는 이미지를 사용하여 학습하도록 합니다. 이러한 구조를 통해 학생 네트워크는 서로 다른 스케일 간의 의미적 및 지역적 특징 정렬을 통해 효과적으로 학습합니다.

- **Performance Highlights**: 이 방법은 광범위한 비교 실험을 통해 현재의 최첨단 기술에 비해 현저하게 우수한 성능을 보여줍니다. 특히 extreme(극단적인) 또는 irregular(불규칙한) 데이터의 영향을 최소화하는 EMA 전략을 도입하여 학생 네트워크의 훈련 안정성을 증가시킵니다.



### LSSF-Net: Lightweight Segmentation with Self-Awareness, Spatial Attention, and Focal Modulation (https://arxiv.org/abs/2409.01572)
- **What's New**: 이 연구에서는 모바일 플랫폼에서 피부 병변(segmentation of skin lesions) 분할을 위한 경량 네트워크 구조인 LSSF-Net을 제안합니다. 이 모델은 피부 병변의 복잡한 특성과 불분명한 경계를 효과적으로 캡처하여 최적의 성능을 구현합니다.

- **Technical Details**: LSSF-Net은 인코더-디코더 아키텍처로 구성되어 있으며, conformer-based focal modulation attention, self-aware local and global spatial attention, split channel shuffle 등을 포함합니다. 이 네트워크는 0.8백만 개의 학습 가능한 파라미터를 가지며, 적은 연산 자원으로도 높은 정확도를 달성합니다.

- **Performance Highlights**: LSSF-Net은 ISIC 2016, ISIC 2017, ISIC 2018 및 PH2 등의 네 가지 벤치마크 데이터 세트에서 평가되어 높은 Jaccard index를 기록하며, 기존의 최첨단 성능을 상회하는 결과를 보여줍니다.



### Blocks as Probes: Dissecting Categorization Ability of Large Multimodal Models (https://arxiv.org/abs/2409.01560)
Comments:
          39 pages, 28 figures, 4 tables. Accepted at The 35th British Machine Vision Conference (BMVC 2024). Project page at this https URL

- **What's New**: 이 논문은 시각 AI 모델의 범주화 성능을 측정하기 위한 새로운 벤치마크인 ComBo를 제안합니다. ComBo는 범주 학습(category learning)부터 적용(category use)까지의 전 과정을 평가할 수 있는 분리된 프레임워크를 제공합니다.

- **Technical Details**: 주요 기술적인 내용은 LMM(large multimodal models)의 범주화 능력을 종합적으로 평가하는 새로운 평가 방식과 이를 위한 데이터셋인 Composite Blocks(ComBo)입니다. ComBo는 본 연구에서 이전 모델들이 겪었던 데이터 누출 문제를 피하며, 인지 과정을 분석하여 범주화 능력을 상세히 평가합니다.

- **Performance Highlights**: 실험 결과 LLMs는 새로운 범주를 학습하는 일반화 능력에서는 acceptable한 성과를 보였지만, 인간에 비해 공간적 관계의 세밀한 인식 및 추상적 범주 이해에서는 여전히 격차가 존재함을 보여주었습니다. 이를 통해 AI의 해석 가능성과 일반화 능력 개선에 기여할 수 있는 통찰을 제시합니다.



### Benchmarking Cognitive Domains for LLMs: Insights from Taiwanese Hakka Cultur (https://arxiv.org/abs/2409.01556)
Comments:
          Submitted to O-COCOSDA 2024

- **What's New**: 본 연구는 대형 언어 모델(LLMs)의 문화 지식 이해 및 처리 성능을 평가하기 위한 포괄적인 벤치마크를 소개하며, 특히 하카 문화에 집중하고 있습니다. Bloom의 분류법(Bloom's Taxonomy)을 활용하여 기억(Remembering), 이해(Understanding), 적용(Applying), 분석(Analyzing), 평가(Evaluating), 창작(Creating) 등 여섯 가지 인지 영역에 걸쳐 LLM을 체계적으로 평가하는 다차원 프레임워크를 개발하였습니다.

- **Technical Details**: 이 연구는 LLM의 성능을 분석하기 위해 다층 질문 세트를 개발하며, 각 질문은 Bloom의 분류법의 여섯 가지 수준에 해당하도록 설계되었습니다. 또한, Retrieval-Augmented Generation (RAG) 기술을 통합하여 LLM이 외부 데이터베이스에서 정보를 검색하고 이를 바탕으로 응답의 정확성을 높입니다. LLM의 문화 지식 적용 및 이해 능력을 분석하기 위해 하카 문화 지식 기반에서 36,522개의 질문이 생성되었습니다.

- **Performance Highlights**: 연구 결과, RAG 기술이 모든 인지 영역에서의 정확성을 향상시키는 데 효과적임을 보여주었으며, 특히 문화 지식의 정확한 검색과 적용이 필요한 작업에서 두드러진 성과를 보였습니다. 그러나 창의적인 작업에서 RAG의 한계가 드러나 향후 최적화의 필요성을 강조했습니다. 이 벤치마크는 문화적으로 다양한 맥락에서 LLM을 평가하고 비교하는 데 유용한 도구로, 미래의 AI 기반 문화 지식 보존 및 전파 연구에 귀중한 통찰력을 제공합니다.



### EA-RAS: Towards Efficient and Accurate End-to-End Reconstruction of Anatomical Skeleton (https://arxiv.org/abs/2409.01555)
Comments:
          13 pages,15 figures

- **What's New**: EA-RAS는 단일 RGB 이미지를 사용하여 실시간으로 해부학적으로 정확한 인간 골격을 추정할 수 있는 경량/단일 단계 솔루션을 제안합니다. 이는 현재 사용 중인 기존의 골격 모델보다 향상된 기능을 가지고 있습니다.

- **Technical Details**: EA-RAS는 해부학적 스켈레톤 추정의 효율성을 높이기 위해 진화된 최적화 과정과 단계적 훈련 전략을 결합하여, 소량의 데이터를 입력받아 기초 중량을 신속하게 확보할 수 있도록 합니다. 또한, 내외부 스킨 정보를 통합하여 결과의 정확성을 더욱 강화합니다.

- **Performance Highlights**: EA-RAS는 기존 방법보다 800배 빠르며, 후처리 최적화 전략을 통해 재구성 정확도가 50% 이상 향상됩니다. 또한, 실시간 처리 요구 조건을 충족하는 속도 증가를 이룩했습니다.



### Self-Instructed Derived Prompt Generation Meets In-Context Learning: Unlocking New Potential of Black-Box LLMs (https://arxiv.org/abs/2409.01552)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 인간 선호와의 정렬을 개선하기 위한 새로운 프레임워크를 제안합니다. 특히, Black-Box LLMs(예: GPT-4)에 쉽게 적용할 수 있는 방법으로, 정보성이 있는 컨텍스트 환경을 자동으로 구성함으로써 더 신뢰할 수 있는 파생 프롬프트를 생성할 수 있게 합니다.

- **Technical Details**: 프레임워크는 self-instructed reinforcement learning (RL) 메커니즘을 포함하여, 파생 프롬프트 생성을 통해 응답 모델과 직접 상호작용합니다. 이 과정에서 원본 프롬프트와의 정렬을 보장하고, 수정된 프롬프트로부터의 불일치를 줄입니다. 또한, LLM의 in-context learning(ICL) 능력을 극대화합니다.

- **Performance Highlights**: 다양한 실험에서 제안한 방법이 기존의 프롬프트 개선 방법보다 응답 품질을 상당히 향상시키며, GPT-4 같은 Black-Box 모델에 대해서도 뛰어난 성능 개선을 보여 주었습니다.



### VoxHakka: A Dialectally Diverse Multi-speaker Text-to-Speech System for Taiwanese Hakka (https://arxiv.org/abs/2409.01548)
Comments:
          Submitted to O-COCOSDA 2024

- **What's New**: VoxHakka는 대만에서 사용되는 자원 부족 언어인 대만 Hakka를 위해 설계된 고품질 텍스트-투-스피치(TTS) 시스템입니다. 이 시스템은 여섯 가지 사투리를 지원하며, YourTTS 프레임워크를 활용하여 발음 정확도 및 자연스러움을 크게 향상시켰습니다.

- **Technical Details**: VoxHakka는 웹 스크래핑과 자동 음성 인식(ASR) 기반 데이터 정제 기법을 통해 다중 화자 및 다중 사투리 데이터셋을 구축하였습니다. 이 모델은 음성 속도 조정이 가능하며 CPU 자원만으로도 효율적으로 동작합니다. VoxHakka는 주로 정부 교육 기관에서 출처를 확보하여 윤리적으로 소싱된 데이터를 사용합니다.

- **Performance Highlights**: 주관적인 청취 테스트 결과 VoxHakka는 기존의 Hakka TTS 시스템과 비교하여 발음 정확도, 톤 정확성, 그리고 전반적인 자연스러움에서 훨씬 뛰어난 성능을 보였습니다. 이 연구는 Hakka 언어 기술의 중대한 발전을 의미하며 언어 보존과 부흥 노력에 중요한 자원을 제공합니다.



### Effective Noise-aware Data Simulation for Domain-adaptive Speech Enhancement Leveraging Dynamic Stochastic Perturbation (https://arxiv.org/abs/2409.01545)
Comments:
          Accepted to IEEE SLT 2024

- **What's New**: 본 논문에서는 기존의 도메인 간 음성 향상 문제를 해결하기 위하여 새로운 데이터 시뮬레이션 방법을 제안합니다. Noise-extractive 기술과 generative adversarial networks (GANs)을 활용하여, 제한된 양의 noisy speech 데이터로부터도 효과적인 음성 향상이 가능하도록 하는 방법론을 개발했습니다.

- **Technical Details**: 제안된 방법, NADA-GAN은 noise encoder를 통해 target-domain 데이터에서 noise embedding을 추출하고, 이를 generator가 이용하여 target domain에 적합한 noisy speech를 합성합니다. 또한, inference 중 dynamic stochastic perturbation을 도입하여 noise embedding에 제어된 변화를 주어, 모델이 미지의 noise 조건에 잘 일반화될 수 있도록 합니다.

- **Performance Highlights**: VoiceBank-DEMAND 벤치마크 데이터셋에서 실험한 결과, 제안된 NADA-GAN 방법이 기존의 강력한 baseline보다 우수한 성능을 보여주었으며, 평균 의견 점수(MOS) 평가에서 뛰어난 성능을 기록하여 음성 향상 외에도 다양한 분야에 적용 가능한 가능성을 제시했습니다.



### Long-Range Biometric Identification in Real World Scenarios: A Comprehensive Evaluation Framework Based on Missions (https://arxiv.org/abs/2409.01540)
- **What's New**: 이 논문에서는 다양한 거리와 고도를 고려한 생체 인식 시스템의 연구 솔루션을 평가하고 있습니다. 기존의 문제를 해결하기 위해 얼굴 인식뿐만 아니라 신체의 특징을 결합하여 장거리 인식의 정확성을 높이는 방법을 제안합니다.

- **Technical Details**: 저자는 UAV(무인 항공기)나 건물에 장착된 카메라를 통한 긴 거리 생체 인식의 이점을 논의하며, 장거리, 각도, 해상도와 같은 다양한 환경 조건에 적합한 시스템 구축을 목표로 합니다. 이를 위해 생체 인식 시스템의 성능을 평가하기 위한 데이터 수집 및 준비 방법론을 제시합니다.

- **Performance Highlights**: 초기 결과는 전체 신체 인식에서 유망한 진행 상황을 보여줍니다. 이 연구는 안면 인식 외에도 다양한 생체 신호를 통해 신뢰할 수 있는 식별을 제공하는 시스템 개발의 가능성을 탐구하고 있습니다.



### Think Twice Before Recognizing: Large Multimodal Models for General Fine-grained Traffic Sign Recognition (https://arxiv.org/abs/2409.01534)
- **What's New**: 새로운 전략 'think twice before recognizing'을 제안하여 정교한 교통 표지 인식(TSR)을 개선합니다. 이 방법은 대규모 다중 모달 모델(LMM)의 다차원 사고 능력을 활용하여 복잡한 도로 조건에서 교통 표지를 효과적으로 인식합니다.

- **Technical Details**: 제안된 기법은 세 가지 유형의 설명(컨텍스트, 특성, 차별 설명)을 통해 LMM의 다중 사고 과정을 설계합니다. 첫째, 컨텍스트 설명은 교차로, 차량, 보행자 등 중요한 정보가 포함되어 있으며, 최적화된 중심 좌표 프롬프트를 사용하여 교통 표지를 원본 도로 이미지에서 정확하게 위치를 찾습니다. 둘째, 특성 설명은 템플릿 교통 표지의 몇 장의 샘플을 사용한 컨텍스트 학습을 기반으로 하여 교차 도메인 차이를 줄이고 세밀한 인식 능력을 향상시킵니다. 셋째, 차별 설명을 통해 유사한 교통 표지 간의 미세한 차이를 강조합니다.

- **Performance Highlights**: 세 가지 벤치마크 데이터셋과 두 개의 실제 데이터셋에서 광범위한 실험을 수행하였고, 제안된 방법은 모든 다섯 개 데이터셋에서 최첨단 TSR 결과를 달성하였습니다.



### Improving Robustness of Spectrogram Classifiers with Neural Stochastic Differential Equations (https://arxiv.org/abs/2409.01532)
- **What's New**: 이번 연구에서는 신호 분석 및 분류에서 발생하는 노이즈 문제를 해결하기 위해 Neural Stochastic Differential Equation (Neural SDE) 기반의 새로운 방법론을 제시합니다. 이는 CNN 모델이 노이즈에 대한 강건성을 높이고, 분류 결과의 신뢰도를 높이기 위한 것입니다.

- **Technical Details**: 신호 처리 및 분류를 위한 2D deep learning 기법의 한계를 극복하기 위한 연구이며, 주요 내용으로는 Convolutional Neural Networks (CNNs)를 사용할 때, 훈련 과정에서 도메인 형태의 노이즈를 주입하여 모델의 강건성과 안정성을 높이는 방법론이 포함됩니다.

- **Performance Highlights**: 실험 결과, ConvNeXt-Base 모델은 시계열 분류에서 우수한 성능을 보였으나, 노이즈가 증가할수록 성능 저하가 심했습니다. 반면, NSDE-ConvNext 변형 모델은 경쟁력 있는 분류 성능을 유지하면서 노이즈가 증가해도 성능 저하가 적었습니다.



### On the Design Space Between Transformers and Recursive Neural Nets (https://arxiv.org/abs/2409.01531)
- **What's New**: 이 논문에서는 Recursive Neural Networks (RvNNs)와 Transformers 간의 밀접한 연결을 탐구합니다. 최근 개발된 Continuous Recursive Neural Networks (CRvNN)와 Neural Data Routers (NDR) 모델을 통해 두 아키텍처 간의 다리를 형성합니다.

- **Technical Details**: CRvNN은 전통적인 RvNN의 경계를 확장하여 Transformer와 유사한 구조로 발전합니다. NDR은 Transformer의 구조적 inductive bias를 개선하기 위해 제약을 둡니다. 두 모델은 알고리즘적 작업과 일반화에서 강력한 성능을 보여줍니다.

- **Performance Highlights**: CRvNN과 NDR은 ListOps와 같은 복잡한 작업에서 뛰어난 성능을 발휘하며, 기존의 RvNN이나 Transformer보다 일반화하는 데 유리합니다. 이 논문에서는 이러한 '브리지 모델'의 설계 공간을 탐색하고 제한 사항을 논의하며 향후 연구를 위한 아이디어를 제안합니다.



### S$^3$c-Math: Spontaneous Step-level Self-correction Makes Large Language Models Better Mathematical Reasoners (https://arxiv.org/abs/2409.01524)
- **What's New**: 본 논문에서는 수학적 추론을 위한 새로운 능력인 자발적 단계별 자기 수정(spontaneous step-level self-correction) 기능을 갖춘 S$^3$c-Math라는 수학적 LLM 모델을 소개합니다. 이 기능은 LLM이 진행 중인 추론에서 오류를 감지하고 이를 즉시 수정하여 보다 신뢰할 수 있는 응답을 생성할 수 있게 돕습니다.

- **Technical Details**: 이 연구에서는 단계별 샘플링(step-level sampling) 접근 방식을 사용하여 자기 수정 데이터를 구축하고, 메타 학습(Meta Learning)을 통해 S3c-Math 모델이 자발적으로 자기 수정 기능을 갖추도록 훈련합니다. S3c-MathQA 데이터셋은 532K의 자기 수정 데이터를 포함하고 있으며, 기존의 MetaMathQA 데이터와 혼합하여 927K의 SFT(supervised fine-tuning) 데이터를 생성했습니다.

- **Performance Highlights**: 이 방법은 GSM8K 및 MATH와 같은 여러 수학적 평가에서 LLM 모델의 성능을 일관되게 향상시키는 것으로 나타났습니다. 실험 결과, 자발적 단계별 자기 수정 능력을 갖춘 S3c-Math 모델은 다양한 수학적 벤치마크에서 상당한 발전을 보였습니다.



### From Data to Insights: A Covariate Analysis of the IARPA BRIAR Dataset for Multimodal Biometric Recognition Algorithms at Altitude and Rang (https://arxiv.org/abs/2409.01514)
- **What's New**: 이 논문은 IARPA BRIAR 데이터셋에서 UAV 플랫폼, 고도 위치 및 최대 1000미터 거리에서의 전체 신체 생체 인식(performance in fused whole body biometrics) 성능에 대한 공변량(covariate) 효과를 조사합니다.

- **Technical Details**: 이 연구는 외부 비디오와 내부 이미지 및 제어된 보행 기록(gait recordings)을 비교하고, 예측된 거부율(false accept rates, FAR)과 관련된 정규화된 원시 융합 점수(normalized raw fusion scores)를 제시합니다. 또한 고도 및 거리에서의 정확도에 가장 큰 영향을 미치는 공변량을 분석하기 위해 선형 모델(linear model)을 개발하고, 온도(temperature), 바람 속도(wind speed), 태양 복사(solar loading), 난류(turbulence)와 같은 날씨 요소들을 조사합니다.

- **Performance Highlights**: 해상도(resolution) 및 카메라 거리(camera distance)가 정확도를 가장 잘 예측할 수 있으며, 이 결과들은 장거리 및 UAV 생체 인식 분야에서의 미래 연구 및 개발 노력을 안내하고 국방 및 기타 중요 분야에서 더 신뢰할 수 있고 강력한 시스템 구축에 기여할 수 있습니다.



### AMG: Avatar Motion Guided Video Generation (https://arxiv.org/abs/2409.01502)
Comments:
          The project page is at this https URL

- **What's New**: AMG라는 새로운 방법을 제안하여 2D의 현실감과 3D의 제어 가능성을 결합하였습니다. 이 방법은 3D 아바타의 제어된 렌더링에 비디오 확산 모델을 조건으로 적용합니다.

- **Technical Details**: AMG는 3D 아바타의 애니메이션 가능한 인간 모델을 생성하고, 2D 동영상에서 3D 신체 움직임을 추출하여 이 정보를 활용해 비디오를 생성합니다. 이 과정에서 Vision Language Model (VLM)을 사용하여 아바타의 외형을 설명하는 프롬프트를 생성합니다. 또한, 기존 텍스트-비디오 모델에 대해 조건부 매개 변수 고효율 미세 조정 방법을 제안합니다.

- **Performance Highlights**: AMG는 기존 인간 비디오 생성 방법들보다 사실성과 적응성에서 우수한 성능을 보이며, 다중 객체 비디오 생성에서 카메라 위치, 인간 동작 및 배경 스타일에 대한 정밀한 제어를 가능하게 합니다.



### EarthGen: Generating the World from Top-Down Views (https://arxiv.org/abs/2409.01491)
- **What's New**: 본 연구에서는 고해상도에서 일관된 이미지를 생성하기 위해 다중 해상도로 결합할 수 있는 초해상도 스프레드(diffusion) 모델의 캐스케이드를 기반으로 한 새로운 방법인 EarthGen을 소개합니다. 이 시스템은 수천 제곱킬로미터의 현실적인 지구 표면을 생성할 수 있는 확장 가능한 구조입니다.

- **Technical Details**: EarthGen은 계층적 생성 방식과 구성적 방식의 장점을 결합한 새로운 프레임워크로, 저해상도에서 시작하여 점진적으로 진화하는 방식을 채택합니다. 스프레드 초해상도 모듈을 사용하여 각 스케일에서 그럴듯한 특징을 정밀하게 추가합니다. 이 모듈은 스케일 인식(scale-aware) 기능을 갖추고 있어 지역적부터 킬로미터, 미터, 센티미터까지 다양한 수준의 구조를 실현할 수 있습니다.

- **Performance Highlights**: 본 시스템은 30km x 10km 크기의 지형을 15cm/pixel 해상도로 생성하며, 이는 맨해튼보다 세 배 큰 규모입니다. 1024x 해상도에서 기존의 초해상도 기법보다 우수한 성능을 보여주었고, 다양한 씬을 생성할 수 있는 가능성도 확인했습니다.



### PoliPrompt: A High-Performance Cost-Effective LLM-Based Text Classification Framework for Political Scienc (https://arxiv.org/abs/2409.01466)
Comments:
          23 pages, 5 figures

- **What's New**: 최근 대형 언어 모델(LLMs)의 발전으로 정치학에서 텍스트 분류 효율성을 높이는 새로운 방법이 열렸습니다. 본 논문은 LLMs를 활용하여 분류 정확도를 높이는 삼단계(in-context learning) 접근법을 소개하며, 실험 비용을 최소화합니다.

- **Technical Details**: 이 접근법은 자동으로 향상된 프롬프트 생성, 적응형 예시 선택, 두 개의 약한 LLM 간 신뢰성 메커니즘을 포함합니다. 이 방법은 BBC 뉴스 보고서, 카바노 후보자 확정, 2018년 선거 광고 데이터를 사용하여 검증되었습니다. 실험 결과에서 제로샷 분류에서 F1 점수가 0.36 상승하고, 인간 레이블링 대비 경제적 비용이 78% 감소하였습니다.

- **Performance Highlights**: 우리의 접근법은 기존의 전통적인 기계 학습의 한계를 극복하고 정치 과학에서 텍스트 분석을 위한 확장 가능하고 신뢰할 수 있는 솔루션을 제공함을 보여주었습니다.



### Kvasir-VQA: A Text-Image Pair GI Tract Datas (https://arxiv.org/abs/2409.01437)
Comments:
          to be published in VLM4Bio 2024, part of the ACM Multimedia (ACM MM) conference 2024

- **What's New**: Kvasir-VQA 데이터세트는 HyperKvasir 및 Kvasir-Instrument 데이터세트를 기반으로 질문-답변 주석을 추가하여 구성되었으며, 위장관(Gastrointestinal, GI) 진단을 위한 고급 기계 학습(machine learning, ML) 작업을 지원하는 데 초점을 맞추고 있습니다. 이 데이터세트는 6,500개의 주석이 달린 이미지를 포함하고 있으며, 다양한 질문 유형(y/n, 선택, 위치, 수치 등)을 지원합니다.

- **Technical Details**: Kvasir-VQA 데이터세트는 의학 영상 분석과 진단 도구 간의 격차를 해소하기 위해 질문-답변 주석이 추가된 새로운 형식으로 구성되었습니다. 이 데이터세트는 이미지 캡셔닝(image captioning), Visual Question Answering (VQA), 텍스트 기반 합성 의료 이미지 생성 등을 포함한 다양한 ML 응용 프로그램을 지원하는 데 사용됩니다. 실험은 이 데이터세트의 유용성을 입증하기 위해 image captioning, VQA, 합성 의료 이미지 생성 등의 세 가지 작업을 중점적으로 다룹니다.

- **Performance Highlights**: 실험 결과, Kvasir-VQA 데이터세트는 의료 이미지 분석 및 진단의 기계 학습 모델 훈련에 효과적임을 입증하였습니다. 제시된 평가 메트릭을 통해 각 작업의 성능을 측정하여 데이터세트의 유용성과 다양성을 강조하고 있습니다.



### Performance-Aware Self-Configurable Multi-Agent Networks: A Distributed Submodular Approach for Simultaneous Coordination and Network Design (https://arxiv.org/abs/2409.01411)
Comments:
          Accepted to CDC 2024

- **What's New**: 많은 분산 에이전트들이 복잡한 작업을 수행하기 위해 자기 통신 토폴로지를 조정할 수 있도록 하는 최초의 철저한 접근 방식을 소개합니다.

- **Technical Details**: 이 연구는 AlterNAting COordination and Network-Design Algorithm (Anaconda)을 제안하며, 에이전트 간의 대역폭 제약을 고려하여 네트워크의 성능을 극대화합니다. 이 알고리즘은 sparse networks에서의 의사 결정 속도를 크게 향상시킵니다.

- **Performance Highlights**: Anaconda는 최적성 보장을 제공하며, 모든 유형의 네트워크에서 평균적으로 빠른 성능을 보입니다. 특히, 대규모 분산 작업에서 기존 알고리즘보다 더 효율적인 의사 결정 속도를 자랑합니다.



### GenAgent: Build Collaborative AI Systems with Automated Workflow Generation -- Case Studies on ComfyUI (https://arxiv.org/abs/2409.01392)
- **What's New**: 이 논문에서는 AI 시스템의 새로운 접근 방식인 협업 AI 시스템을 제안합니다. 특히, 복잡한 워크플로우(workflows)를 자동으로 생성할 수 있는 GenAgent라는 LLM 기반 프레임워크를 소개하고 있습니다.

- **Technical Details**: GenAgent는 코드(code)로 워크플로우를 표현하고, 협업 에이전트(collaborative agents)와 함께 단계별로 워크플로우를 구성하는 방법을 독창적으로 구현하였습니다. ComfyUI 플랫폼에서 GenAgent를 구현하고, 새로운 벤치마크인 OpenComfy를 제안합니다.

- **Performance Highlights**: GenAgent는 실행 수준(run-level) 및 작업 수준(task-level) 평가에서 기존 접근 방식에 비해 뛰어난 성능을 보여주며, 복잡한 워크플로우를 효과적이고 안정적으로 생성할 수 있는 능력을 입증하였습니다.



### VLSI Hypergraph Partitioning with Deep Learning (https://arxiv.org/abs/2409.01387)
- **What's New**: 이 연구에서는 VLSI 설계에서의 그래프 파티셔닝 문제를 해결하기 위해 새로운 합성 벤치마크인 PERRDI(Partitioning Examples with Rent’s Rule Derived Information)를 제안합니다. 이 벤치마크는 실제 네트리스트(netlist) 특성을 모방하며, 솔루션 컷 품질에 대한 알려진 상한을 갖고 있습니다.

- **Technical Details**: 기존의 파티셔닝 알고리즘과 GNN 기반 접근 방식을 비교 분석합니다. 주요 초점은 GNN에서의 풀링 계층(pooling layer)과 VLSI에서의 효과적 적용입니다. GNN은 비균일 그래프를 입력으로 하여 노드 예측, 엣지 예측 및 그래프 분류 작업에 사용됩니다. GAP(Generalizable Approximate Graph Partitioning)는 지도 학습과 비지도 학습을 통해 그래프를 효율적으로 파티셔닝합니다.

- **Performance Highlights**: 새롭게 제안된 PERRDI 벤치마크는 VLSI 디자인의 고유한 특성을 반영하여 기존 GNN 기반 파티셔닝 알고리즘의 성능을 평가합니다. 이 연구는 ML 파티셔닝의 장단점을 분석하고, ML 파티셔닝 방법의 최적 훈련 방법과 런타임 성능을 비교 평가합니다.



### Automatic Detection of LLM-generated Code: A Case Study of Claude 3 Haiku (https://arxiv.org/abs/2409.01382)
Comments:
          Submitted to a journal for potential publication

- **What's New**: 이 논문에서는 Claude 3 Haiku에 의해 생성된 코드의 탐지 방법을 제안합니다. 기존 연구들이 주로 독립 함수에만 중점을 두었던 반면, 이 연구는 실제 소프트웨어 프로젝트에서의 클래스 및 함수 수준 코드 모두를 분석합니다.

- **Technical Details**: 우리는 CodeSearchNet 데이터셋을 사용하여 함수 수준과 클래스 수준의 코드 생성 및 분석을 수행합니다. 22개의 소프트웨어 메트릭(features) 특성을 추출하였으며, LM-generated 코드와 인간 코드 간의 차별성을 분석했습니다. Machine Learning (ML) 모델을 활용하여 Claude 3-generated 코드의 탐지 가능성을 평가하였습니다.

- **Performance Highlights**: 최고 성능의 모델인 CatBoost를 통해 함수 수준 코드의 탐지 정확도가 82%, 클래스 수준 코드는 66%에 달했습니다. 분석 결과, Claude 3이 생성한 코드가 더 긴 함수끼리, 짧은 클래스 구조를 가지는 경향이 있음을 보여주었습니다.



### Imitating Language via Scalable Inverse Reinforcement Learning (https://arxiv.org/abs/2409.01369)
- **What's New**: 본 논문은 언어 모델 훈련에서 imitation learning의 역 강화 학습(inverse reinforcement learning, IRL) 접근 방식을 탐구합니다. 특히, 주어진 토큰의 가능성(likelihood) 대신에 보상을 추출하고 시퀀스를 직접 최적화하는 것에 중점을 둡니다.

- **Technical Details**: 논문에서는 maximum likelihood estimation (MLE)과 inverse reinforcement learning (IRL) 간의 원칙적인 연결을 만드는 새로운 방법으로 inverse soft-Q-learning의 시간 차이 정규화된 확장을 제시합니다. 이를 통해 감독적인 미세 조정(supervised fine-tuning) 설정에서 모델의 성능과 다양성을 균형 있게 개선할 수 있습니다.

- **Performance Highlights**: IRL 기반의 imitation이 언어 모델의 다양성을 유지하면서도 작업 성능을 극대화함에 있어 명확한 이점을 보여 주기 때문에, 고정된 SFT 데이터 세트에서도 강력한 대안이 됩니다. 실험 결과, IRL은 행동 복제(behavior cloning)보다 우수하거나 동등한 작업 성능을 달성하면서 더 높은 생성의 다양성을 나타내는 것을 확인했습니다.



### CHESS: Optimizing LLM Inference via Channel-Wise Thresholding and Selective Sparsification (https://arxiv.org/abs/2409.01366)
- **What's New**: 본 논문은 대형 언어 모델(LLM)을 엣지 디바이스에 배포하는 데 있어 발생하는 계산 오버헤드 및 메모리 요구 사항을 완화하기 위해 새로운 활성화 희소화(activation sparsification) 접근 방식을 제안합니다. 기존의 방법들이 성능 저하를 제대로 모델링하지 못하는 문제를 해결하고자, CHESS라는 새로운 방법론을 소개합니다.

- **Technical Details**: CHESS는 채널 단위(thresholding) 기준의 희소화를 통해 FFN 레이어의 각 활성화 채널에 대해 고유한 임계값(threshold)을 할당하고, 선택적 희소화(selective sparsification)를 통해 주의(attention) 모듈의 특정 레이어에 임계값 기반의 희소화를 적용합니다. 마지막으로, 희소 활성화를 기반으로 한 스파스 커널(sparse kernels)의 구현을 통해 LLM 추론을 가속화합니다.

- **Performance Highlights**: 실험 결과, CHESS는 8개의 다운스트림(downstream) 작업에서 기존 방법들에 비해 더 낮은 성능 저하를 달성하며, 최대 1.27배까지 LLM 추론 속도를 향상시킵니다.



### Correlating Time Series with Interpretable Convolutional Kernels (https://arxiv.org/abs/2409.01362)
Comments:
          11 pages, 7 figures

- **What's New**: 이 연구는 단일, 다중 및 다차원 시계열 데이터에서 컨볼루션(kernel) 학습 문제를 다루며, 이는 시계열의 시간 패턴을 해석하고 다운스트림 머신러닝 작업을 지원하는 데 중요합니다. 단일 시계열에 대한 컨볼루션 커널 학습을 비부정적(non-negative) 제약을 가진 희소 회귀(sparse regression) 문제로 공식화하고, 다차원 데이터에 대해 텐서(tensor) 계산을 활용하여 이 방법을 일반화합니다.

- **Technical Details**: 제안된 방법론에서는 기존의 비부정적 서브스페이스 추구(non-negative subspace pursuit) 방법을 사용하여 최적화 문제를 해결합니다. 이 접근법은 시계열 데이터에서 시간적 상관관계와 패턴을 포착하는 데 초점을 맞춥니다.

- **Performance Highlights**: 뉴욕 및 시카고의 다차원 rideshare 및 택시 데이터에서 테스트하여 컨볼루션 커널들이 해석 가능한 지역 상관관계와 주기적인 패턴(예: 주간 계절성)을 드러내는 것을 보여주었습니다. 또한 유체 흐름 데이터의 경우, 이 커널들이 텐서 분해(tensor factorization)를 강화하여 유체 흐름 재구성 성능을 향상시키는 결과를 나타냈습니다.



### Language Models Benefit from Preparation with Elicited Knowledg (https://arxiv.org/abs/2409.01345)
- **What's New**: 이 연구에서는 PREP이라는 간단한 일반 프롬프트 기법을 소개했습니다. 이 방법은 두 개의 언어 모델 인스턴스를 사용하여 정보를 생성하고 이를 바탕으로 질문에 답하는 구조로, 사용자의 도메인 지식과 무관하게 다양한 Q&A 작업에 적용할 수 있습니다.

- **Technical Details**: PREP 기법은 첫 번째 인스턴스(LM1)가 관련 정보를 생성하고, 두 번째 인스턴스(LM2)가 이 정보를 기반으로 질문에 답하는 방식으로 작동합니다. 이는 ‘지식 이끌기’와 ‘지식 전이’라는 두 단계를 포함합니다.

- **Performance Highlights**: 실험에서 PREP 방법은 100개의 이진 선택 질문과 세 가지 이미 출판된 상식 추론 데이터셋에서 다른 방법들에 비해 평균적으로 높은 정확도를 보였습니다. 이는 직접 질문하기, 제로샷 CoT, 단일 인스턴스 프롬프트 방법에 비해 일관되게 우수한 성능을 나타냅니다.



### Pediatric brain tumor classification using digital histopathology and deep learning: evaluation of SOTA methods on a multi-center Swedish cohor (https://arxiv.org/abs/2409.01330)
- **What's New**: 본 연구는 스웨덴의 다중 센터 데이터셋을 활용하여 소아 뇌종양의 분류를 위해 두 가지 약한 슈퍼바이즈드 다중 인스턴스 학습 접근 방식을 구현하여, 최신 병리학적 컴퓨터 비전 기술이 소아 뇌종양 진단에 기여할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 연구는 540명의 소아 뇌종양 환자의 전체 슬라이드 이미지(WSI)에서 패치 특징을 추출하고, 세 가지 사전 학습된 특징 추출기(ResNet50, UNI, CONCH)를 사용하여, 주의 기반 MIL(ABMIL) 및 클러스터 제약 주의 MIL (CLAM)로 패치 수준의 특징을 환자 수준으로 집계합니다.

- **Performance Highlights**: 종양 범주, 계통, 유형 분류의 경우 각각 Matthew 상관계수 0.86±0.04, 0.63±0.04 및 0.53±0.05를 기록하며, UNI 특징과 ABMIL 집계를 사용했을 때 최고의 분류 성능이 달성되었습니다. 모델 일반화는 UNI와 CONCH 특징을 사용하는 모델이 ResNet50을 사용하는 모델보다 성능이 우수했으며, 테스트 결과는 여러 센터의 데이터에서 0.7의 AUC를 기록했습니다.



### Grounding Language Models in Autonomous Loco-manipulation Tasks (https://arxiv.org/abs/2409.01326)
Comments:
          Summit to ICRA@40. arXiv admin note: substantial text overlap with arXiv:2406.14655

- **What's New**: 이번 연구에서는 다양한 시나리오에 기반한 행동 계획과 선택을 학습하고 실행하는 새로운 프레임워크를 제안합니다. 이는 자율성을 크게 증가시키는 작업으로, 로봇이 인간의 자유 텍스트 명령에 따라 복잡한 loco-manipulation 작업을 수행할 수 있도록 지원합니다.

- **Technical Details**: 우리의 프레임워크는 강화 학습( Reinforcement Learning, RL)과 전체 몸체 최적화( whole-body optimization)를 결합하여 로봇의 동작을 생성합니다. 생성된 동작들은 동작 라이브러리에 저장되며, 대형 언어 모델( Large Language Model, LLM) 기반의 계획 및 추론 기능을 활용하여 계층적 작업 그래프를 구성합니다. 이 그래프는 동작 원시( motion primitives)로 구성되어 하위 수준 실행과 상위 수준 계획을 연결합니다.

- **Performance Highlights**: 실험 결과, LLM 기반의 계획기가 로봇이 다양한 loco-manipulation 작업을 효율적으로 계획할 수 있도록 함을 보여주었습니다. 실제 환경에서 60% 이상의 성공률을 달성하였고, 실패 감지 및 회복( Failure Recovery, FR) 기능을 추가함으로써 작업 실행의 성공률을 더욱 증가시킬 수 있음을 확인했습니다.



### Multi-frequency Neural Born Iterative Method for Solving 2-D Inverse Scattering Problems (https://arxiv.org/abs/2409.01315)
- **What's New**: 이번 연구에서는 다중 주파수 전자기 (EM) 역산란 문제를 해결하기 위해 딥 러닝 기반 이미징 방법을 제안합니다. 단일 주파수 Neural Born Iterative Method (NeuralBIM)의 원리를 바탕으로 다중 주파수 NeuralBIM을 개발하며, 다중 작업 학습 기술과 효율적인 반복 역산법을 결합하여 다중 주파수 Born iterative inversion 모델을 구축합니다.

- **Technical Details**: 본 방법은 동질산불확실성(homoscedastic uncertainty)에 따라 각 주파수 데이터의 가중치를 적응적으로 할당하는 다중 작업 학습 접근 방식을 사용하며, 물리 법칙에 제약된 비지도 학습 방법을 통해 multi-frequency NeuralBIM 모델을 훈련합니다. 이로 인해 대조(constrast) 및 총(field) 데이터 없이도 모델 훈련이 가능합니다.

- **Performance Highlights**: Synthetic 및 experimental 데이터로 검증된 다중 주파수 NeuralBIM은 역산란 문제의 정확도와 계산 효율성을 높이며, 강한 일반화 능력과 소음 저항성을 보여줍니다. 이 방법은 다중 주파수 EM 데이터에 대한 새로운 역산법을 탐구하고, 다중 주파수 데이터의 전자기 역산란 문제를 효과적으로 해결할 수 있는 솔루션을 제공합니다.



### Topological degree as a discrete diagnostic for disentanglement, with applications to the $\Delta$VAE (https://arxiv.org/abs/2409.01303)
- **What's New**: 이 논문에서는 고립된 구형(latent space $	extmath{S}^2$)을 가진 Diffusion Variational Autoencoder ($	extmath{ΔVAE}$)의 토폴로지(topological) 및 기하학적 구조를 포착하고 잠재적 요인을 분리하는 능력을 조사합니다.

- **Technical Details**: 우리는 인코더의 토폴로지 차수(topological degree)를 새로운 분리(disentanglement) 진단 지표로 도입하고, 호모로지 이론(homology theory)에서 도구를 사용하여 이 차수를 계산하는 알고리즘을 구현했습니다. 실험을 통해 $	extmath{ΔVAE}$의 인코더 훈련 후 토폴로지 차수가 -1 또는 +1이 된다는 것을 확인했고, 이는 얻어진 인코더가 최소한 동형사상(homeomorphism)과 호모토픽(homotopic)이라는 것을 암시합니다.

- **Performance Highlights**: $	extmath{ΔVAE}$는 상대적으로 낮은 LSBD 점수를 기록하며, 초기화 후 차수와 관계없이, 훈련이 완료된 후 인코더의 차수가 +1 또는 -1로 변한다는 실험적 결과를 제시하였습니다.



### Path-Consistency: Prefix Enhancement for Efficient Inference in LLM (https://arxiv.org/abs/2409.01281)
- **What's New**: 이번 연구에서는 대규모 언어 모델의 추론 효율성을 높이기 위해 경로 일관성(path-consistency)이라는 새로운 방법을 제안합니다. 이는 이전 생성 단계에서 나온 정답의 신뢰도를 활용하여 가장 유망한 경로의 접두사(prefix)를 식별하여 이후 방향의 생성을 유도합니다.

- **Technical Details**: 경로 일관성은 기존의 자기 일관성(self-consistency) 기법과 다르게 동적 추론 방법을 채택하여, 이미 생성된 추론 경로로부터 적절한 추론 단계를 지속적으로 추출하여 ‘접두사’로 사용합니다. 이러한 방식은 불필요한 샘플링을 줄이고 생성되는 토큰 수를 감소시켜 전체 추론 속도를 향상시킵니다.

- **Performance Highlights**: 실험 결과, 경로 일관성 방법은 추론 지연 시간을 7.8%에서 40.5%까지 단축시키면서도 수학적 추론, 상식 추론, 기호 추론 및 코드 생성 작업에서 작업 정확도를 유지하거나 오히려 향상시켰습니다. 수학적 추론에서 평균 28.7%, 상식 추론에서 20.9%, 기호 추론에서 20.3% 의 가속화를 기록했습니다.



### Real-time Accident Anticipation for Autonomous Driving Through Monocular Depth-Enhanced 3D Modeling (https://arxiv.org/abs/2409.01256)
- **What's New**: 본 연구에서는 새로운 프레임워크인 AccNet을 소개하여 기존의 2D 기반 방법들을 넘어서는 예측 능력을 선보입니다. 이는 단안 깊이 정보(monocular depth cues)를 통합하여 복잡한 3D 장면 모델링을 가능하게 합니다. 또한, 사고 데이터셋의 불균형 문제를 해결하기 위해 Binary Adaptive Loss for Early Anticipation (BA-LEA)라는 새로운 손실 함수를 제안합니다.

- **Technical Details**: AccNet은 3D Collision Module을 중심으로 하여, 단안 깊이 정보를 이용해 차량과 보행자와 같은 주요 객체의 3D 좌표를 정확하게 추출합니다. 이를 통해 dashcam 비디오에서의 공간적 관계와 동태를 더 잘 이해할 수 있게 하며, 새로운 그래프 토폴로지를 도입하여 Graph Neural Networks (GNN)의 리스크 평가를 향상시킵니다. 또한, 멀티 태스크 학습 전략(multi-task learning strategy)과 Smooth Module을 도입하여 예측 모델의 결정적인 순간에 대한 초점을 맞추는데 기여합니다.

- **Performance Highlights**: DAD, CCD, A3D, 및 DADA-2000 데이터셋을 통해 검증한 결과, AccNet은 Average Precision (AP) 및 mean Time-To-Accident (mTTA)와 같은 주요 지표에서 기존 방법들을 초월한 뛰어난 예측 성능을 입증하였습니다. 이를 통해 사고 예측 기술에 있어 중요한 진전을 이룩하였습니다.



### Revisiting Safe Exploration in Safe Reinforcement learning (https://arxiv.org/abs/2409.01245)
- **What's New**: 이번 논문에서는 안전한 강화학습(Safe Reinforcement Learning, SafeRL)의 새로운 지표인 예상 최대 연속 비용 단계(EMCC, Expected Maximum Consecutive Cost Steps)를 소개합니다. 이 지표는 교육 과정에서의 안전성을 평가하는 데 초점을 맞추며, 안전하지 않은 행동의 심각성을 연속 발현 수에 기반해 평가합니다.

- **Technical Details**: EMCC는 각 롤아웃(rollout)에서 비용 단계의 최대 연속 발생을 측정하여 계산됩니다. 이 지표는 훈련 중에 발생할 수 있는 안전 위반의 유형을 구분하는 데 효과적입니다. SafeRL 알고리즘은 CMDP(Constrained Markov Decision Process)를 통해 안전성을 확보하며, 여러 SafeRL 알고리즘의 성능을 평가할 수 있는 새로운 경량 벤치마크 작업인 Circle2D도 제안했습니다.

- **Performance Highlights**: EMCC 지표를 통해 다양한 SafeRL 알고리즘의 안전한 탐색 능력을 종합적으로 벤치마킹했습니다. 이는 안전한 행동을 보장하면서도 효과적인 데이터 수집을 이루기 위한 탐색과 착취(exploitation) 사이의 균형을 맞추는 데 중요한 통찰력을 제공합니다.



### CyberCortex.AI: An AI-based Operating System for Autonomous Robotics and Complex Automation (https://arxiv.org/abs/2409.01241)
- **What's New**: 이 논문에서는 이 http URL 로 명명된 새로운 Robotics Operating System(로봇 운영 체제)을 소개합니다. 이 시스템은 다양한 AI 기반 로봇과 복합 자동화 어플리케이션을 위해 설계되었으며, 로봇들 간의 통신 및 클라우드에 있는 고성능 컴퓨터(HPC)와의 연결을 가능하게 합니다.

- **Technical Details**: CyberCortex.AI 운영 체제는 두 가지 주요 구성 요소로 이루어져 있습니다. 첫 번째는 로봇의 내장 하드웨어에서 실행되는 CyberCortex.AI.inference 시스템이며, 두 번째는 HPC에서 AI 알고리즘을 설계하고 교육하며 배포해주는 CyberCortex.AI.dojo입니다. 로봇의 각각의 기능은 인터넷을 통해 공유되는 DataBlock의 필터 내에서 실행되며, 필터는 로봇 자체에서 또는 다른 로봇 시스템에서 원격으로 계산됩니다.

- **Performance Highlights**: 제안된 방법론의 성능을 평가하기 위해 두 가지 협업 로봇 어플리케이션을 사용하여 정량적 및 정성적 성능 분석을 수행하였습니다. 첫 번째는 Unitree A1 다리 로봇 및 Anafi Parrot 4K 드론을 기반으로 한 산불 예방 시스템이며, 두 번째는 협업 인식 및 모션 제어를 위한 자율 주행 시스템입니다.



### ESP-PCT: Enhanced VR Semantic Performance through Efficient Compression of Temporal and Spatial Redundancies in Point Cloud Transformers (https://arxiv.org/abs/2409.01216)
- **What's New**: 이번 연구에서는 VR(가상현실) 애플리케이션에서 중요한 의미 인식을 개선하기 위해 새로운 모델인 ESP-PCT를 제안합니다. ESP-PCT는 밀리미터파(mmWave) 신호를 활용하여 점 구름(point cloud)을 생성하고, 인식 과정을 최적화합니다.

- **Technical Details**: ESP-PCT는 두 단계의 의미 인식 프레임워크를 채택하고 있으며, 센서 데이터의 정확성을 활용하여 로컬라이제이션(localization)과 포커스(focus) 단계를 공동으로 훈련시킵니다. 이는 엔드 투 엔드(end-to-end) 방식으로 진행되며, 기존의 Point Transformer 모델에 비해 효율성을 향상시키는 데 중점을 두고 있습니다.

- **Performance Highlights**: ESP-PCT는 인식 효율성에서 엄청난 향상을 보였으며, 정확도는 93.2%에 달합니다. 또한, FLOPs(부동소수점 연산 수)는 76.9%, 메모리 사용량은 78.2% 감소하여 VR 의미 인식에서 높은 정확도와 중복성 감소를 달성할 수 있음을 보여주었습니다.



### EnCLAP++: Analyzing the EnCLAP Framework for Optimizing Automated Audio Captioning Performanc (https://arxiv.org/abs/2409.01201)
Comments:
          Accepted to DCASE2024 Workshop

- **What's New**: 이번 연구는 자동 오디오 자막 생성에서 EnCLAP 프레임워크를 분석하고 최적화하여, 기존 모델보다 성능이 크게 향상된 EnCLAP++를 개발했습니다.

- **Technical Details**: EnCLAP++는 신경 오디오 코덱(Neural audio codecs)을 활용하여 입력 오디오 신호를  변환하고, 다양한 기법을 통해 모델 성능을 개선하였습니다. 특히, MCM(masked codec modeling)과 샘플링-재정렬(sampling-and-reranking) 방식을 적용했습니다.

- **Performance Highlights**: EnCLAP++는 DCASE2024 Challenge Task6에서 2위를 차지하며 향상된 자막 생성을 보여주었습니다.



### Logit Scaling for Out-of-Distribution Detection (https://arxiv.org/abs/2409.01175)
- **What's New**: 이 연구는 OOD(Out-of-Distribution) 데이터 감지를 위한 새로운 접근 방식인 Logit Scaling (LTS)을 제안합니다. LTS는 훈련 데이터 배포에 대한 접근이 필요하지 않으며, 다양한 아키텍처에서 강력한 성능을 유지합니다.

- **Technical Details**: LTS 방법은 후처리(post-hoc) 방식으로 작동하여 훈련 데이터 통계에 의존하지 않습니다. LTS는 penultimate layer의 feature representation을 활용하여 샘플 별 scaling factor를 계산하고, 이를 통해 logits를 조정합니다. 최종 OOD 점수는 energy score 함수로 계산됩니다. 이 방법은 3개의 ID(인-디스트리뷰션) 및 14개의 OOD 데이터셋에서 실험을 통해 검증되었습니다.

- **Performance Highlights**: LTS는 9가지의 다양한 아키텍처에 걸쳐 이전의 최첨단 OOD 탐지 방법인 OptFS보다 우수한 성능을 보였으며, FPR@95를 유의미하게 감소시키면서 AUROC를 유지하였습니다. 다양한 환경에서도 잘 작동하는 OOD 탐지 솔루션을 제시합니다.



### Expanding on EnCLAP with Auxiliary Retrieval Model for Automated Audio Captioning (https://arxiv.org/abs/2409.01160)
Comments:
          DCASE2024 Challenge Technical Report. Ranked 2nd in Task 6 Automated Audio Captioning

- **What's New**: 이 기술 보고서에서는 DCASE2024 Challenge Task6(자동 오디오 캡셔닝) 및 Task8(언어 기반 오디오 검색)에 대한 우리의 제출물을 설명합니다. EnCLAP 오디오 캡셔닝 프레임워크를 기반으로 구축된 접근 방식을 개발하고 Task6의 품질을 최적화하였습니다. 기본 모델보다 훨씬 뛰어난 성능을 달성한 것을 강조합니다.

- **Technical Details**: 자동 오디오 캡셔닝(AAC)은 소리 이벤트가 포함된 오디오 신호를 간결하고 의미 있는 자연어 설명으로 변환하는 교차 모달 변환 작업입니다. EnCLAP 프레임워크는 오디오 특성 인코더와 사전 훈련된 BART 모델을 사용하여 타임스탭 수준과 시퀀스 수준의 표현을 생성합니다. 원래 EnCLAP의 EnCodec을 DAC(Dесript Audio Codec)으로 대체하고, 두 단계의 샘플링 및 리랭킹 절차를 사용하여 캡션의 품질을 개선했습니다.

- **Performance Highlights**: Task6에서 FENSE 점수 0.542, Task8에서 mAP@10 점수 0.386을 달성했습니다. 이 결과는 기본 모델보다 상당히 높은 성능을 보일 뿐만 아니라, 제출 모델이 다양한 것으로 나타나 추가적인 점수 향상을 제공했습니다.



### FMRFT: Fusion Mamba and DETR for Query Time Sequence Intersection Fish Tracking (https://arxiv.org/abs/2409.01148)
Comments:
          14 pages,14 figures

- **What's New**: 이 논문은 복잡한 수조 환경에서 어류의 다중 추적을 위한 새로운 실시간 모델인 FMRFT를 제안하며, 이를 통해 물고기 군집 내의 유사성과 상호 장애물 문제를 해결할 수 있는 효과적인 방법을 제시합니다.

- **Technical Details**: FMRFT 모델은 Mamba In Mamba (MIM) 아키텍처와 RT-DETR의 기능을 융합하여 다중 물체를 효율적으로 추적합니다. 새로운 QTSI(Query Time Sequence Intersection) 모듈을 도입하여 중복된 추적 프레임을 줄이고 연결된 프레임의 효율성을 개선합니다. 또한, 8,000개의 스턴전 물고기 추적 이미지를 포함한 새로운 데이터셋을 구축하여 연구를 뒷받침합니다.

- **Performance Highlights**: 실험 결과, 제안된 FMRFT 모델은 IDF1 점수 90.3%와 MOTA 정확도 94.3%를 달성하여 어류 추적 분야에서 높은 정확성과 안정성을 입증했습니다.



### LATEX-GCL: Large Language Models (LLMs)-Based Data Augmentation for Text-Attributed Graph Contrastive Learning (https://arxiv.org/abs/2409.01145)
- **What's New**: 본 논문에서는 Text-Attributed Graphs (TAGs)에서 Graph Contrastive Learning (GCL) 기법을 적용하기 위한 새로운 프레임워크인 LATEX-GCL을 제안합니다. LATEX-GCL은 Large Language Models (LLMs)를 활용하여 텍스트 속성을 증강함으로써 기존 GCL 방법의 한계를 극복하고 TAGs에 효과적으로 적응할 수 있도록 합니다.

- **Technical Details**: LATEX-GCL 프레임워크는 세 가지 주요 모듈로 구성됩니다: I) LLM 기반 텍스트 속성 증강, II) 텍스트 속성 인코딩, III) 그래프 인코딩 및 IV) 그래프 대조 학습. 이를 통해 정보 손실, 의미 손실 및 증강 제한 등의 문제를 해결합니다.

- **Performance Highlights**: 제안된 LATEX-GCL 방법은 네 가지 고품질 TAG 데이터셋에 대한 광범위한 실험을 통해 기존 방법들에 비해 뛰어난 성능을 보여줍니다. 또한, 소스 코드와 데이터셋이 공개되어 재현성을 높이고 사용자들이 효과적으로 활용할 수 있도록 지원합니다.



### Generating Synthetic Satellite Imagery for Rare Objects: An Empirical Comparison of Models and Metrics (https://arxiv.org/abs/2409.01138)
Comments:
          Presented at KI 2024 - 47th German Conference on AI, 2nd Workshop on Public Interest AI, 23 September, 2024, Wuerzburg, DE

- **What's New**: 이 연구에서는 드물게 존재하는 객체(핵 발전소)에 대한 합성 위성 이미지를 생성하는 대규모 실증 연구를 소개합니다. 이는 이전 연구와 달리, 현실 세계의 사례가 제한된 경우의 생성 가능한 이미지를 제어하고 평가하는 방법을 탐구합니다.

- **Technical Details**: 본 연구는 텍스트 입력과 게임 엔진에서 얻은 이미지 입력을 통해 핵 발전소라는 드문 객체 카테고리의 합성 위성 이미지를 생성합니다. 이를 위해 사전 훈련된 텍스트-이미지 모델을 활용하고, 대규모 사용자 연구를 통해 생성된 이미지의 신뢰성을 평가합니다. 또한, 자동 평가 메트릭과 인간 평가 간의 신뢰도를 비교합니다.

- **Performance Highlights**: 핵 발전소와 같은 드문 객체의 합성 위성 이미지를 텍스트나 상세한 건축 레이아웃을 통해 생성할 수 있다는 것을 입증했습니다. 그러나 자동 평가 메트릭과 인간의 인식 간에는 강한 부정 상관관계가 발견되어, 기존의 메트릭이 항상 신뢰할 수 있는 것은 아님을 보여줍니다.



### Smart E-commerce Recommendations with Semantic AI (https://arxiv.org/abs/2409.01137)
Comments:
          8 pages

- **What's New**: 본 논문에서는 사용자 요구를 충족하지 못하는 기존의 웹 마이닝(page recommendations) 한계를 극복하기 위해 semantic web mining과 BP 신경망(BP neural networks)를 결합한 혁신적인 솔루션을 제안합니다.

- **Technical Details**: 사용자의 검색 로그(search logs)를 처리하여 콘텐츠 우선순위(content priority), 소요 시간(time spent), 사용자 피드백(user feedback), 추천의 의미(recommendation semantics), 입력 편차(input deviation)의 다섯 가지 핵심 특성(key features)을 추출합니다. 이러한 특성들은 BP 신경망에 입력되어 웹 페이지를 분류(classify)하고 우선순위를 매깁니다.

- **Performance Highlights**: 책 판매 페이지를 사용한 테스트 결과, 제안된 솔루션은 사용자가 필요로 하는 페이지를 신속하고 정확하게 식별할 수 있음을 보여주었습니다. 이 방법은 더욱 관련성 높은 추천을 제공하여 온라인 쇼핑 경험을 향상시킵니다. 또한, 대규모 데이터셋을 처리하고 실시간(real-time) 추천을 제공할 수 있는 능력 덕분에 현대 전자상거래의 복잡한 문제를 해결할 수 있는 확장 가능하고 효율적인 솔루션입니다.



### Large Language Models Can Understanding Depth from Monocular Images (https://arxiv.org/abs/2409.01133)
- **What's New**: 이번 논문은 대형 언어 모델(LLMs)이 최소한의 감독 하에 단일 카메라를 통해 깊이 추정을 효과적으로 수행할 수 있음을 보여줍니다. 특히, LLM-MDE라는 다중 모달 프레임워크를 제안하여 언어 이해를 사용하여 깊이를 해석합니다.

- **Technical Details**: LLM-MDE는 Cross-modal reprogramming과 Adaptive prompt estimation 모듈을 통해 LLM의 깊이 추정 능력을 개선합니다. 이 두 가지 전략은 시각 표현과 텍스트 프로토타입을 정렬하고 단일 이미지 기반으로 자동으로 프롬프트를 생성하는 방식으로 작동합니다. 전반적인 프레임워크는 Vision Transformer (ViT)와 LLM의 조합으로 이루어져 있습니다.

- **Performance Highlights**: 다양한 실제 MDE 데이터셋에 대한 실험을 통해 LLM-MDE의 효과와 우수성을 증명했습니다. 특히, LLM-MDE는 소수 샷(few-shot) 및 제로 샷(zero-shot) 작업에서 뛰어난 성능을 발휘하였으며, 자원 사용을 최소화하면서도 높은 예측 정확도를 달성했습니다.



### Two-stage initial-value iterative physics-informed neural networks for simulating solitary waves of nonlinear wave equations (https://arxiv.org/abs/2409.01124)
Comments:
          25 pages, 17 figures

- **What's New**: 이 논문은 비선형 파동 방정식의 고유 파동 계산을 위한 새로운 두 단계 초기값 반복 신경망(IINN) 알고리즘을 제안하고 있습니다. IINN은 초기값을 피팅하는 하나의 서브네트워크와 물리적 정보를 통합하는 다른 서브네트워크로 구성되어 있으며, 경계 조건과 같은 추가 데이터 없이도 학습할 수 있습니다.

- **Technical Details**: IINN 알고리즘은 초기값을 설정하고, 첫 번째 서브네트워크를 통해 초기값을 충분히 근사한 후, 두 번째 서브네트워크의 매개변수를 첫 번째 네트워크에서 학습한 가중치와 편향으로 초기화합니다. 이후 손실 함수에서 PDE 잔여(residual)를 고려하여 최적화합니다. 이 과정은 머신러닝 관점에서 전이 학습(transfer learning)으로 알려져 있으며, 초기값이 정확한 솔루션과 합리적으로 가까울 경우 IINN 방법의 효과성과 수렴성이 잘 입증되었습니다.

- **Performance Highlights**: IINN 방법은 다양한 비선형 파동 방정식에서 고유 파동 계산에 적용되며, 전통적인 방법과 비교하여 장점을 보여줍니다. 특히, 고차원 및 복잡한 지역 문제를 효율적으로 해결할 수 있는 능력을 갖추고 있습니다.



### AI Olympics challenge with Evolutionary Soft Actor Critic (https://arxiv.org/abs/2409.01104)
- **What's New**: IROS 2024에서 개최되는 AI 올림픽 대회에 대한 새로운 기법을 제안합니다. 이 기법은 Model-free Deep Reinforcement Learning과 evolutionary strategy를 결합한 접근 방식을 사용합니다.

- **Technical Details**: 이 논문은 대회 시뮬레이션 단계에서의 접근 방식을 설명합니다. Soft Actor-Critic (SAC) 알고리즘을 사용하여 로봇의 스윙업과 안정화를 수행하는 정책을 찾아내며, 이후 진화 방법으로 에이전트를 세부 조정합니다. SAC 알고리즘은 안정적이고 효율적인 훈련을 가능하게 하며, 경쟁의 보상 함수를 최적화하는 데 기여합니다.

- **Performance Highlights**: 제안하는 방법은 기존의 기준선을 초과하는 성능을 나타내며, SAC 알고리즘을 통해 에이전트의 강인성을 보장합니다. 대회에서 요구하는 스윙업과 안정화 작업을 수행하기 위한 최적화된 컨트롤러를 제시하며, 나아가 실제 하드웨어 단계에서도 테스트를 진행합니다.



### DS MYOLO: A Reliable Object Detector Based on SSMs for Driving Scenarios (https://arxiv.org/abs/2409.01093)
Comments:
          27th International Conference on Pattern Recognition(ICPR)

- **What's New**: 본 논문에서 제안된 새로운 객체 탐지기 DS MYOLO는 Simplified Volitional Scan Fusion Block (SimVSS Block)과 Efficient Channel Attention Convolution (ECAConv)을 사용하여 효과적인 특징 융합과 채널 간 상호작용을 가능하게 하여, 기존 YOLO 시리즈와 유사한 수준의 모델들 사이에서 경쟁력을 높였습니다.

- **Technical Details**: DS MYOLO는 SimVSS Block을 통하여 깊은 글로벌 특징 융합을 달성하며, ECAConv는 저렴한 계산 복잡도를 유지하면서 채널 간의 의존성을 강화합니다. DS MYOLO는 CCTSDB 2021과 VLD-45 데이터셋에서 성능 배치를 통해 뛰어난 경쟁력을 입증했습니다.

- **Performance Highlights**: DS MYOLO는 기존의 동일 규모 YOLO 시리즈 모델들과 비교하여 CCTSDB 2021과 VLD-45 드라이빙 시나리오 데이터셋에서 경쟁력 있는 성과를 보여줍니다. 이 탐지기는 빠른 속도와 높은 정확도를 동시에 달성하며, 자율주행 시스템의 안전성을 향상시키는 데 기여할 잠재력을 가지고 있습니다.



### Two-Timescale Synchronization and Migration for Digital Twin Networks: A Multi-Agent Deep Reinforcement Learning Approach (https://arxiv.org/abs/2409.01092)
Comments:
          15 pages, 14 figures

- **What's New**: 이번 논문에서는 이동하는 모바일 사용자(Mobile Users, MUs)와의 디지털 트윈(Digital Twins, DT) 동기화 문제를 다룬 새로운 이론 프레임워크를 제안합니다. 특히, 두 가지 시간 척도(two-timescale)에서 DT 동기화 및 마이그레이션(migration) 문제를 해결하고자 하며, 에너지 소비를 최소화하는 데 중점을 두고 있습니다.

- **Technical Details**: 본 연구에서는 비볼록 확률적 문제(non-convex stochastic problem)를 설정하고, Lyapunov 이론을 활용하여 신뢰성 제약을 변형한 후, 이를 부분 관찰 마르코프 결정 과정(partially observable Markov decision-making process, POMDP)으로 재구성합니다. 추가적으로, 이 문제 해결을 위해 heterogeneous agent proximal policy optimization with Beta distribution (Beta-HAPPO) 방법을 개발하였습니다.

- **Performance Highlights**: Beta-HAPPO 방법은 에너지 효율성 측면에서 다른 벤치마크와 비교하여 현저한 개선 효과를 보였습니다. 이 접근 방식은 DT 동기화 및 마이그레이션 시 발생할 수 있는 동기화 실패를 고려하여, 더욱 신뢰성 높은 네트워크 성능을 보장합니다.



### Pre-Trained Language Models for Keyphrase Prediction: A Review (https://arxiv.org/abs/2409.01087)
- **What's New**: 본 논문은 사전 훈련된 언어 모델을 이용한 키프레이즈 예측(PLM-KP)에 대한 포괄적인 분석을 제공합니다. 기존 문헌에서의 키프레이즈 추출 및 생성의 통합 탐색 부족을 다루고, 체계적인 분류체계를 통해 이 두 가지 작업에 대한 이해를 심화시키고자 합니다.

- **Technical Details**: 키프레이즈 예측에는 키프레이즈 추출(Keyphrase Extraction, KPE)과 키프레이즈 생성(Keyphrase Generation, KPG)이라는 두 가지 주요 작업이 포함됩니다. 이 과정에서 사용되는 모델들은 주로 self-supervised learning을 통해 사전 훈련된 언어 모델(Pre-trained Language Models, PLMs)이며, Attention Mechanism, Graph-based Ranking 및 Phrase-Document Similarity와 같은 다양한 방법론을 포함합니다.

- **Performance Highlights**: 본 논문은 PLM-KP 분야의 최신 동향을 다루며, 특히 딥러닝 기술을 활용한 NLP 작업에서의 성과를 강조합니다. 동시에 향후 연구 방향을 제시하여 키프레이즈 예측 연구의 발전을 이끌고자 합니다.



### DPDEdit: Detail-Preserved Diffusion Models for Multimodal Fashion Image Editing (https://arxiv.org/abs/2409.01086)
Comments:
          13 pages,12 figures

- **What's New**: 본 논문에서는 패션 이미지 편집을 위한 새로운 멀티모달 아키텍처인 Detail-Preserved Diffusion Models (DPDEdit)를 소개합니다. 이 모델은 잠재 확산 모델(latent diffusion models)을 기반으로 하여, 텍스트 프롬프트(text prompts), 영역 마스크(region masks), 인간 포즈 이미지(human pose images), 의류 질감 이미지(garment texture images)를 통합하여 패션 이미지 생성을 안내합니다.

- **Technical Details**: DPDEdit의 주요 구성 요소로는 Grounded-SAM을 사용하여 사용자의 텍스트 설명에 기반해 편집 영역을 예측하고, 입력된 조건들을 결합하여 지역 편집(local editing)을 수행합니다. 또한, 질감 주입(texture injection) 및 정제(refinement) 메커니즘을 통해 주어진 의류 질감의 디테일을 타겟 패션 이미지로 전이하는 방식도 제안합니다. 이를 위해 텍스트 설명과 질감 이미지를 통합하는 분리된 크로스 어텐션 레이어(decoupled cross-attention layer)와 U-Net 구조를 활용하여 생성된 의류 질감의 고주파 세부 사항을 보존합니다.

- **Performance Highlights**: 상세한 실험 결과에 따르면, DPDEdit는 주어진 멀티모달 입력과의 이미지 충실도(image fidelity) 및 일관성(coherence) 측면에서 최신 기법(state-of-the-art methods)을 능가하는 성능을 보여줍니다.



### Affordance-based Robot Manipulation with Flow Matching (https://arxiv.org/abs/2409.01083)
- **What's New**: 이 논문에서는 두 가지 기본적인 문제를 해결하는 조교 로봇 조작을 위한 프레임워크를 제안합니다: 첫째, 대규모 모델을 효율적으로 다운스트림 씬 애포던스 이해 작업에 적응시키는 것; 둘째, 시각적 애포던스 모델을 기반으로 로봇 궤적을 효과적으로 학습하는 것입니다.

- **Technical Details**: 이 연구는 파라미터 효율적인 프롬프트 튜닝 방법을 통해 학습 가능한 텍스트 프롬프트를 고정된 비전 모델에 추가하여 여러 작업 시나리오에서 조작 애포던스를 예측하는 것을 다룹니다. 또한, Flow Matching 방법으로 안내된 로봇 궤적을 연구합니다. Flow Matching은 로봇 비주얼 모터 정책을 다양하고 예측 가능한 궤적을 향해 흐르도록 하는 조건적 프로세스로 모델링합니다.

- **Performance Highlights**: 제안된 프롬프트 튜닝 방법은 데이터 스케일 전반에서 다른 파인튜닝 프로토콜보다 경쟁력 있는 성능을 낸다고 평가되었으며, Flow Matching 방법을 통한 다중 작업 로봇 궤적 학습은 대체 행동 클로닝 방법보다 일관되게 더 나은 성능을 보여줍니다.



### Beyond Efficiency: Molecular Data Pruning for Enhanced Generalization (https://arxiv.org/abs/2409.01081)
Comments:
          20 pages, under review

- **What's New**: 새로운 연구인 MolPeg은 데이터 프루닝(data pruning)을 통해 전이 학습(transfer learning)의 효과를 향상시키는 프레임워크입니다. 특히, 소스-프리(source-free) 데이터 프루닝 시나리오를 위한 새로운 기존 프루닝 방법과의 차별점을 가지고 있습니다.

- **Technical Details**: MolPeg 프레임워크는 두 개의 모델을 유지하며, 서로 다른 업데이트 속도로 훈련하는 방식을 채택합니다. 새로운 스코어링 함수는 손실 불일치(loss discrepancy)를 기반으로 샘플의 유용성을 측정해 주목할 만한 샘플과 도전적인 샘플을 동시에 선택합니다. 이 방법은 성능 최적화를 위한 경량화된 DP를 가능하게 합니다.

- **Performance Highlights**: MolPeg은 HIV와 PCBA 데이터셋에서 데이터의 60-70%를 프룬(prune)하더라도 전체 데이터셋을 사용할 때보다 더 나은 성능을 기록하였습니다. 이는 기존의 데이터 프루닝 전략보다 우수한 일반화 성능을 보여주며, 효율성을 높여줄 수 있는 방법을 제시합니다.



### Bootstrap SGD: Algorithmic Stability and Robustness (https://arxiv.org/abs/2409.01074)
- **What's New**: 본 논문에서는 경험적 부트스트랩(bootstrap) 접근 방식을 활용한 확률적 경량 충전법(stochastic gradient descent, SGD)의 새로운 방법을 제안하고 있습니다. 부트스트랩 SGD는 알고리즘 안정성(algorithmic stability) 및 통계적 강건성(statistical robustness) 관점에서 검토되며, 방법론에 따른 이론적 분석도 포함됩니다.

- **Technical Details**: 주요 접근 방식으로는 세 가지 부트스트랩 방법이 있으며, 각각 Type 1, Type 2, Type 3으로 분류됩니다. Type 1은 가중치 파라미터의 평균을 취하고, Type 2는 각 모델의 예측을 종합하여 최종 예측을 도출합니다. Type 3은 각 모델의 예측을 기반으로 중앙값을 사용하여 점별 신뢰 구간(pointwise confidence intervals)을 생성합니다.

- **Performance Highlights**: 부트스트랩 샘플이 일반화(generalization) 성능을 향상시키는 방법을 분석하며, 특히 Type 2에 대한 안정성 분석을 최초로 수행합니다. 또한, 실험을 통해 제안된 방법의 유용성을 수치적으로 보여 주며, 분포에 무관한 신뢰 구간의 구축 가능성을 입증합니다.



### SCOPE: Sign Language Contextual Processing with Embedding from LLMs (https://arxiv.org/abs/2409.01073)
- **What's New**: 본 연구에서는 대화 맥락을 고려한 새로운 Sign Language Recognition (SLR)과 Sign Language Translation (SLT) 프레임워크인 SCOPE를 소개합니다. 기존의 방법들이 대화 장면에서의 데이터 세트의 다양성 부족과 맥락 정보를 무시하는 문제를 해결하기 위해 제시되었습니다.

- **Technical Details**: SCOPE는 다중 모달 인코더를 활용하여 대화 맥락을 통해 글로스 레벨 인식을 향상시키며, 사전 대화 맥락을 포함하여 대형 언어 모델 (Large Language Model, LLM)을 추가로 미세 조정합니다. 우리는 72시간 분량의 중국 수화 비디오로 구성된 새로운 수화 데이터 세트를 제공하고, 실험을 통해 Phoenix-2014T, CSL-Daily 및 SCOPE 데이터 세트에서 최첨단 성능을 달성하였습니다.

- **Performance Highlights**: SCOPE 프레임워크는 다양한 데이터 세트에서 탁월한 성능을 보였으며, 청각 장애인 커뮤니티의 참가자를 대상으로 실시한 설문조사 결과도 우리의 접근 방식이 실제 응용에서의 강력함과 효과성을 입증하였습니다. 데이터 세트 및 코드는 연구 촉진을 위해 오픈 소스로 제공될 예정입니다.



### A Perspective on Literary Metaphor in the Context of Generative AI (https://arxiv.org/abs/2409.01053)
Comments:
          Accepted as oral presentation to Workshop on Artificial Intelligence and Creativity (CREAI) at ECAI 2024

- **What's New**: 이 연구는 문학적 은유(literary metaphor)와 텍스트 생성의 창의성 간의 교차점을 살펴보며, AI를 활용한 새로운 문학적 표현 방식의 가능성을 모색합니다.

- **Technical Details**: 연구는 Afrikaans 언어에서 LSTM 기반 언어 모델을 훈련시켰으며, 이것은 메타포(metaphor), 유사(simile), 의인화(personification)와 같은 독창적인 표현을 생성합니다.

- **Performance Highlights**: 생성된 표현들은 창의적 언어 사용을 통해 독특한 감정적 강도를 전달하며, AI가 새로운 언어 사용 방식을 도입할 수 있는 가능성을 제시합니다.



### Accelerated Multi-objective Task Learning using Modified Q-learning Algorithm (https://arxiv.org/abs/2409.01046)
Comments:
          9 pages, 9 figures, 7 tables

- **What's New**: 본 논문에서는 Q-learning 알고리즘의 수정 버전인 스케일 거리 메트릭(Q-SD)을 제안합니다. 이 알고리즘은 로봇이 테이블 청소 작업을 수행할 때 움직임 거리를 최소화하면서 작업 학습을 개선합니다.

- **Technical Details**: Q-SD 알고리즘은 테이블을 그리드로 나누어 최적의 청소 경로를 학습합니다. 이 알고리즘은 3x3 및 4x4 그리드 환경에서 실험되어 각각 86% 및 59%의 성공률을 기록하였습니다. 전통적인 Q-learning 알고리즘과 비교했을 때, Q-SD를 활용한 경우 평균 이동 거리가 각각 8.61%와 6.7% 감소하였습니다.

- **Performance Highlights**: Q-SD 알고리즘을 통해 로봇의 이동 거리를 줄이면서 작업 성공률을 높이는 데 기여하였습니다. 이는 자율 청소 로봇의 효율성을 크게 향상시킬 수 있는 가능성을 보여줍니다.



### Robust Vehicle Localization and Tracking in Rain using Street Maps (https://arxiv.org/abs/2409.01038)
- **What's New**: 본 논문에서는 비가 오는 날이나 터널을 통과하는 상황에서도 차량의 위치 추정을 향상시키기 위해 Map-Fusion이라는 새로운 접근법을 제안합니다. 이 방법은 간헐적인 GPS 측정값과 드리프팅(drifting) IMU(관성 측정 장치), 그리고 시각 정보(Visual Odometry, VO)를 2D 맵 정보와 융합하여 robust한 차량 로컬라이제이션을 제공합니다.

- **Technical Details**: Map-Fusion은 GPS를 초기화에 사용하고 VO 또는 VIO를 사용하여 드리프트(drift)를 보정하는 센서 퓨전(sensor fusion) 기법입니다. 이 방법은 다양한 센서를 통합하기 위해 factor graph 구조를 사용하며, OpenStreetMap(OSM)에서 얻은 도로 네트워크 정보를 활용하여 차량의 위치를 추정하고 보정합니다.

- **Performance Highlights**: Map-Fusion 알고리즘은 다양한 기후 조건, 즉 맑은 날씨와 비 오는 날씨에서 활용된 실험에서 각각 2.46m와 6.05m의 오차를 보였으며, 이는 최신 VO 및 VIO 방식에 비해 모든 데이터 세트에서 드리프트 오류를 줄여주는 효과를 입증했습니다.



### From Bird's-Eye to Street View: Crafting Diverse and Condition-Aligned Images with Latent Diffusion Mod (https://arxiv.org/abs/2409.01014)
Comments:
          Accepted at International Conference on Robotics and Automation(ICRA)

- **What's New**: 본 논문은 Bird's-Eye View (BEV) 맵을 다중 시점 거리 이미지로 변환하는 방법을 탐구합니다. 이는 자율 주행 응용프로그램에서 중대한 역할을 하며, BEV 맵에서 정확한 거리 뷰 이미지를 생성하는 것이 복잡한 교통 시나리오를 보여주고 드라이빙 알고리즘을 향상시키는 데 필수적입니다.

- **Technical Details**: 우리는 Neural View Transformation과 Street Image Generation의 두 가지 주요 구성 요소로 이루어진 실용적인 프레임워크를 소개합니다. Neural View Transformation 단계에서는 BEV 맵을 다중 시점의 세분화 맵으로 변환하고, 이후 Street Image Generation 단계에서 이 세분화를 조건으로 하여 미세 조정된 latent diffusion model을 이용합니다. 이를 통해 다양한 시점과 스타일 일관성을 유지합니다.

- **Performance Highlights**: 우리 모델은 자율주행 관련 다양한 데이터를 통해 미세 조정된 large pretrained diffusion models의 생성 능력을 활용하여, 고품질의 다양하고 조건 일관성이 뛰어난 거리 뷰 이미지를 생성합니다. 실험 결과, 우리의 접근 방식은 기존 방법보다 시각적 품질과 조건 일관성 면에서 뛰어난 성과를 보입니다.



### SeCo-INR: Semantically Conditioned Implicit Neural Representations for Improved Medical Image Super-Resolution (https://arxiv.org/abs/2409.01013)
Comments:
          This paper was accepted for presentation at the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2025

- **What's New**: 이 논문에서는 의학 이미지의 국소적인 정보(Localized Prior)를 활용하는 새로운 프레임워크인 Semantically Conditioned INR (SeCo-INR)을 제안합니다. 이는 기존의 Implicit Neural Representations (INR) 방법을 개선하여 의료 이미지를 보다 정확하게 모델링하고 초해상도를 달성하도록 합니다.

- **Technical Details**: SeCo-INR은 의료 이미지를 기초로 한 의미론적(segmentation) 특징을 연속적으로 학습하며, 각 의미론적 영역에 최적의 INR 매개변수를 도출하는 데이터 기반 프로세스를 적용합니다. 이를 위해 다층 퍼셉트론(Multilayer Perceptrons, MLP)을 사용하여 신호와 좌표 간의 복잡한 관계를 근사합니다.

- **Performance Highlights**: 다양한 의료 이미징 데이터셋에서 실험을 진행한 결과, SeCo-INR은 기존의 최첨단(super-state-of-the-art) 방법들에 비해 더 높은 정량적 성능 점수와 현실감 있는 초해상도 출력(realistic super-resolution outputs)을 만들어냈습니다.



### 3D Priors-Guided Diffusion for Blind Face Restoration (https://arxiv.org/abs/2409.00991)
- **What's New**: 본 논문에서는 3D 얼굴 구조를 디노이징(diffusion) 과정에 통합하여 새로운 디퓨전 기반 얼굴 복원(framework)을 제안합니다. 이를 통해 복원된 이미지의 사실성과 신원 일관성을 향상시키고자 합니다.

- **Technical Details**: 우리는 3D Morphable Model (3DMM)을 활용하여 복원된 초기 얼굴 이미지를 바탕으로 보다 정확한 3D 이전 정보를 재구성하며, 커스텀 다중 레벨 특성 추출 방법(multi-level feature extraction)을 사용하여 구조적 및 신원 정보를 추출합니다. 이 정보는 Time-Aware Fusion Block (TAFB)을 통해 노이즈 추정 과정에 통합됩니다.

- **Performance Highlights**: 제안된 네트워크는 합성 및 실제 데이터셋에서 blind face restoration과 관련하여 최신 알고리즘들과 비교하여 우수한 성능을 보임을 보여줍니다.



### Co-Learning: Code Learning for Multi-Agent Reinforcement Collaborative Framework with Conversational Natural Language Interfaces (https://arxiv.org/abs/2409.00985)
Comments:
          12 pages, 8 figures

- **What's New**: 최근 온라인 Q&A 시스템이 대화형 AI 및 코드 수정 지원을 위한 Multi-Agent 프레임워크로 발전하고 있으며, 이를 통해 초보자들이 코드 오류를 독립적으로 수정할 수 있도록 돕고 있습니다.

- **Technical Details**: 이 논문에서는 환경적 강화 학습(Environmental Reinforcement Learning, E-RL)을 활용하여 다수의 대형 언어 모델(Large Language Model, LLM)을 이용한 코드 수정 프레임워크인 Co-Learning을 제안합니다. Co-Learning은 메인 에이전트(Main Agent), 수정 에이전트(Correction Agent), 해석 에이전트(Interpretation Agent), 테스트 에이전트(Test Agent), 주석 에이전트(Annotation Agent)의 5개 에이전트로 구성되어 있습니다. 각 에이전트는 특정 작업을 담당하며 대화 인터페이스를 통해 상호작용합니다.

- **Performance Highlights**: 실험 결과, E-RL 방법을 적용한 경우 Precision 점수가 3% 향상되고, 시간 비용이 15% 감소하는 결과를 보여줍니다.



### DNN-GDITD: Out-of-distribution detection via Deep Neural Network based Gaussian Descriptor for Imbalanced Tabular Data (https://arxiv.org/abs/2409.00980)
Comments:
          17 pages

- **What's New**: 본 연구에서는 불균형(class imbalance) 문제와 OOD(Out-of-Distribution) 표본 탐지 문제를 해결하기 위한 새로운 알고리즘인 Deep Neural Network 기반 Gaussian Descriptor for Imbalanced Tabular Data (DNN-GDITD)를 소개합니다.

- **Technical Details**: DNN-GDITD는 스피어 형태의 결정 경계를 사용하여 불균형 데이터의 분류와 OOD 탐지를 수행합니다. 이 알고리즘은 Push, Score-based 및 focal loss를 결합하여 테스트 데이터 포인트에 신뢰도 점수를 할당하고, 이를 통해 알려진 클래스 또는 OOD 샘플로 분류할 수 있습니다. DNN-GDITD는 기존의 딥러닝 모델 위에 삽입하여 사용할 수 있으며, 다양한 테이블 데이터셋에서 실험을 수행했습니다.

- **Performance Highlights**: 광범위한 실험에서 DNN-GDITD는 세 개의 기존 OOD 탐지 알고리즘과 비교해 우수한 성능을 보였으며, 다양한 불균형 및 균형 시나리오에서도 뛰어난 신뢰도를 나타내었습니다.



### Enhancing Privacy in Federated Learning: Secure Aggregation for Real-World Healthcare Applications (https://arxiv.org/abs/2409.00974)
Comments:
          Accepted at the 5-th MICCAI Workshop on Distributed, Collaborative and Federated Learning in Conjunction with MICCAI 2024

- **What's New**: 본 연구는 의료 분야에서 Federated Learning (FL)의 실제 적용 시 직면하는 문제점인 커뮤니케이션 및 보안 문제를 해결하기 위해 Secure Aggregation (SA) 방법을 탐색합니다. 두 가지 SA 프로토콜(Joye-Libert, Low Overhead Masking)을 Fed-BioMed 프레임워크에 구현하고 비교함으로써 개인 정보 보호를 효과적으로 보장하고자 합니다.

- **Technical Details**: 본 논문에서 연구한 SA 프로토콜은 Joye-Libert(JL)와 Low Overhead Masking(LOM)입니다. 이 프로토콜은 의료 데이터 분석 문제를 다루기 위한 실험적이고 이론적인 평가를 기반으로 세 가지 의료 데이터셋(Fed-IXI, Fed-Heart, REPLACE-BG, FedProstate)에서 수행되었습니다. 훈련 중에는 CPU에서 1% 미만의 계산 오버헤드와 GPU에서 최대 50%의 오버헤드를 보였으며, 보호 단계는 10초 미만에 완료되었습니다.

- **Performance Highlights**: SA를 Fed-BioMed에 통합하는 경우 정확도는 비SA 시나리오에 비해 2%를 넘지 않았으며, 실제 의료 분야 애플리케이션에 SA를 적용하는 것이 가능함을 보여줍니다. 이 결과는 민감한 애플리케이션에서 개인 정보 보호 기술의 채택을 위한 간극을 줄이는 데 기여합니다.



### Solving Integrated Process Planning and Scheduling Problem via Graph Neural Network Based Deep Reinforcement Learning (https://arxiv.org/abs/2409.00968)
Comments:
          24 pages, 13 figures

- **What's New**: 이번 논문에서는 Integrated Process Planning and Scheduling (IPPS) 문제를 해결하기 위해 혁신적인 end-to-end Deep Reinforcement Learning (DRL) 방법을 소개합니다. 이 방법은 IPPS 문제를 Markov Decision Process (MDP)로 모델링하고, Heterogeneous Graph Neural Network (GNN)을 이용해 복잡한 연산, 기계 및 작업 간의 관계를 포착합니다.

- **Technical Details**: 우리는 Proximal Policy Optimization (PPO)을 사용하여 스케줄링 전략을 최적화합니다. 이 모델은 DRL 구조를 통해 작동-기계 쌍을 생성하고, 조기 완료 시간을 기반으로 한 조밀한 보상 함수로 안정성과 일반화를 향상시킵니다. 이를 통해 새로운 스케줄링 환경에서도 효과적으로 일반화할 수 있는 모델을 구축합니다.

- **Performance Highlights**: 실험 결과, 기존의 전통적 방법에 비해 우리의 접근 방식이 대규모 IPPS 인스턴스에서 솔루션의 효율성과 품질을 크게 향상시켰음을 보여주었습니다. 특히 16개 이상의 작업이 포함된 큰 규모 문제에서 OR-Tools와 비교하여 11.35%의 개선을 달성했습니다.



### Semantically Controllable Augmentations for Generalizable Robot Learning (https://arxiv.org/abs/2409.00951)
Comments:
          Accepted for publication by IJRR. First 3 authors contributed equally. Last 3 authors advised equally

- **What's New**: 이번 연구에서는 로봇 조작을 위한 데이터 검증을 위해 미리 훈련된 이미지-텍스트 생성 모델을 활용하여, 직접적인 경험을 뛰어넘는 다양한 데이터를 발전적 증강(Data Augmentation) 방식으로 제공하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 우리는 데이터의 다양성을 확보하기 위해, 대규모 웹 스크랩 데이터를 기반으로 훈련된 이미지-텍스트 생성 모델을 활용합니다. 이 모델은 로봇의 직접 경험 범위를 초과하는 다양한 실세계 시나리오를 포괄하며, 새로운 합성 경험을 생성하여 로봇 에이전트가 추가적인 세계 이전 지식을 경험하도록 합니다. 이러한 생성을 통해, 로봇 데이터셋의 급속한 증강과 풍부한 변화를 유도하여 실세계 일반화를 가능하게 합니다.

- **Performance Highlights**: 우리는 제안된 생성 증강 프레임워크를 통해 로봇 조작 정책이 시뮬레이션 및 주방과 같은 실세계 환경에서 어떻게 교육 및 배포될 수 있는지를 보여주었습니다. 실험 결과, 생성된 데이터셋으로 훈련된 로봇이 미리 보지 못한 새로운 환경에서도 향상된 일반화 능력을 보임을 발견했습니다.



### XNet v2: Fewer Limitations, Better Results and Greater Universality (https://arxiv.org/abs/2409.00947)
- **What's New**: XNet v2는 저주파 (Low Frequency, LF) 및 고주파 (High Frequency, HF) 보완 모델로, 이미지 수준의 보완 융합을 수행하며 원본 이미지 입력과 함께 세 개의 서로 다른 서브 네트워크를 사용하여 일관성 손실 (consistency loss)을 구성합니다.

- **Technical Details**: XNet v2는 주 네트워크와 LF 및 HF 네트워크로 구성되어 있으며, 이미지 수준 및 특성 수준에서의 융합 모듈을 도입하여 LF 및 HF 정보를 효과적으로 전달합니다. 주 네트워크는 UNet 구조를 기반으로 하며, LF 및 HF 출력으로 일관성 손실을 구성하여 체크포인트 기반의 학습을 지원합니다.

- **Performance Highlights**: XNet v2는 세미-슈퍼바이즈드 (semi-supervised) 분할에서 최신의 성능을 달성했으며, 풀리-슈퍼바이즈드 (fully-supervised) 학습에서도 경쟁력 있는 결과를 유지합니다. 특히, XNet이 실패하는 시나리오에서도 더욱 뛰어난 성능을 보이며, 3개의 2D 및 2개의 3D 데이터셋에서 효과성을 입증하였습니다.



### A Framework for Synthetic Audio Conversations Generation using Large Language Models (https://arxiv.org/abs/2409.00946)
Comments:
          This work has been submitted for consideration at the WI-IAT'24 to be held in December 2024

- **What's New**: 본 논문에서는 다양한 페르소나 설정을 사용하는 대형 언어 모델(LLMs)을 활용하여 합성 대화 오디오를 생성하는 ConversaSynth 프레임워크를 소개합니다. ConversaSynth는 여러 주제에 걸쳐 다양한 텍스트 기반 대화를 생성한 후, 이를 텍스트-투-스피치(TTS) 시스템을 이용해 오디오로 변환합니다.

- **Technical Details**: ConversaSynth는 LLMs를 활용하여 여러 화자가 포함된 대화를 생성하고, 각 화자는 개별 특징을 가지고 설정됩니다. 이 과정은 LLM 선택, 대화 페르소나 디자인, 대화 생성, 텍스트를 음성으로 변환, 오디오 대화 결합의 단계를 포함합니다. 실험을 통해 Llama3 모델이 다른 모델보다 빠른 응답 속도와 수용 가능한 오류율을 가지고 있음을 확인하여 선택되었습니다.

- **Performance Highlights**: ConversaSynth로 생성된 합성 데이터셋은 상당한 다양성과 사실성을 지니며, 오디오 태깅, 오디오 분류 및 다중 화자 음성 인식 모델의 훈련 및 평가를 향상시키는 데 유용합니다. 이 연구는 합성 오디오 생성 및 머신러닝, 인공지능 분야의 미래 연구 및 개발을 위한 기반을 마련합니다.



### Large Language Models for Automatic Detection of Sensitive Topics (https://arxiv.org/abs/2409.00940)
Comments:
          2024 Oz CHI conference

- **What's New**: 본 연구는 정서적 웰빙과 관련된 민감한 정보를 탐지하기 위해 5개의 대형 언어 모델(LLMs)의 성능을 평가했습니다. 이 연구는 이러한 모델이 온라인 데이터셋에서 민감한 메시지를 감지하는 능력을 지님을 보여줍니다.

- **Technical Details**: 연구에서는 정확도(accuracy), 정밀도(precision), 재현율(recall), F1 점수, 일관성(consistency) 등의 성능 지표를 사용하여 LLM의 능력을 평가했습니다. 가장 높은 성능을 보인 모델인 GPT-4o는 평균 99.5%의 정확도와 0.99의 F1 점수를 기록했습니다.

- **Performance Highlights**: LLMs은 콘텐츠 조절 워크플로우에 통합될 가능성이 있으며, 안전하고 정확한 탐지 도구로서의 역할을 할 수 있음을 보여주었습니다. 또한, 향후 연구에서는 LLM 사용의 윤리적 고려 사항을 다룰 필요성이 강조되었습니다.



### Development of Occupancy Prediction Algorithm for Underground Parking Lots (https://arxiv.org/abs/2409.00923)
- **What's New**: 이번 연구는 자율주행 기술이 지하와 같은 불리한 환경에서 인식 문제를 해결하기 위해 새로운 접근 방식을 제안합니다. CARLA 시뮬레이션 환경 내에서 지하 주차장 모델을 구축하고, Transformer 기반의 Occupancy Network를 통합하여 점유 그리드 예측 작업을 수행합니다.

- **Technical Details**: 이 연구에서는 Unreal Engine 기반의 CARLA 시뮬레이션 플랫폼을 활용하여 지하 주차장의 현실적인 시나리오를 재현하고, 그 과정에서 수집된 데이터에 대해 다중 프레임 융합 알고리즘을 적용하여 희소 포인트 클라우드 데이터를 조밀하게 처리했습니다. 마지막으로, Transformer 기반의 비전 의미 장면 완성 모델을 통해 점유 그리드 예측을 수행하였습니다.

- **Performance Highlights**: 제안된 솔루션은 SUSTech-COE-ParkingLot이라는 독자적으로 구축된 데이터셋을 사용하여 실험을 진행했으며, 지하 환경에서의 인식 성능을 유효성 있게 검증하고 만족스러운 결과를 도출하였습니다.



### Statically Contextualizing Large Language Models with Typed Holes (https://arxiv.org/abs/2409.00921)
Comments:
          To appear at OOPSLA2024

- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)과 프로그래밍 언어 서버의 결합을 통해 코드 자동 완성의 정확도를 높이는 방법을 제안합니다. 이를 통해 IDE(Integrated Development Environment)와 언어 서비스를 결합하여 AI가 프로그래밍 작업을 보다 효과적으로 지원하게 됩니다.

- **Technical Details**: 이 논문에서는 LLM의 코드를 생성하는 기능을 Hazel live program sketching 환경에 통합합니다. 언어 서버는 다른 파일에서 정의된 타입 및 바인딩 컨텍스트를 식별하여 유의미한 프로그램 스케치를 제공합니다. MVUBench라는 데이터 세트를 도입하여 이를 검증하였으며, TypeScript와의 호환성도 확보하여 고급 언어에서도 적용 가능성을 확인했습니다.

- **Performance Highlights**: 이 연구는 타입 정의와 같은 상황 맥락의 사용이 특히 효과적이라는 것을 발견했습니다. AI 코드 완성 시스템들이 언어 서버 프로토콜을 준수하도록 ChatLSP라는 확장을 제안하여 각종 AI 코드 완성 시스템이 정적 컨텍스트를 활용할 수 있도록 지원합니다.



### ToolACE: Winning the Points of LLM Function Calling (https://arxiv.org/abs/2409.00920)
Comments:
          21 pages, 22 figures

- **What's New**: 본 논문에서는 ToolACE라는 자동화된 에이전틱 파이프라인을 소개하여, 고품질의 다양한 도구 학습 데이터를 생성하는 방법을 제시합니다. 이는 새로운 자기 진화 합성(Self-Evolution Synthesis) 프로세스를 활용하여 26,507개의 다양한 API 풀을 구성합니다.

- **Technical Details**: ToolACE는 두 가지 주요 모듈인 도구 자기 진화 합성(TSS)과 다중 에이전트 상호작용 대화 생성(MAI)을 포함하며, 이들 구성 요소를 통해 다양한 API를 생성하고, 정교한 대화를 구축하며, 데이터 품질을 엄격하게 검토합니다. 또한 데이터 정확도를 확보하기 위해 규칙 기반과 모델 기반 점검을 결합한 이중 레이어 검증 시스템(DLV)을 구현합니다.

- **Performance Highlights**: ToolACE에서 생성한 데이터로 훈련된 모델은 단 8B 파라미터에도 불구하고 Berkeley Function-Calling Leaderboard에서 최신 GPT-4 모델에 필적하는 성능을 달성하였습니다. 또한 다양한 함수 호출 시나리오에 대한 소개를 통해 LLM의 도구 사용 능력을 향상시킬 수 있는 기반을 마련하였습니다.



### MMT-BERT: Chord-aware Symbolic Music Generation Based on Multitrack Music Transformer and MusicBER (https://arxiv.org/abs/2409.00919)
Comments:
          Accepted to the 25th International Society for Music Information Retrieval Conference (ISMIR 2024)

- **What's New**: 이 논문에서는 기호적(multitrack) 음악 생성을 위한 새로운 기호적 음악 표현과 Generative Adversarial Network(GAN) 프레임워크를 제안합니다. 특히, MusicLang 코드 분석 모델을 도입하여 코드 및 스케일 정보를 포함한 새로운 기호적 음악 표현을 통해 음악 생성의 인간다운 특성을 개선합니다.

- **Technical Details**: MMT-BERT 아키텍처를 통해 기호적 음악 표현을 적용하며, 미리 훈련된 MusicBERT 모델을 판별기로 활용하고, Relativistic Standard Loss를 도입하여 훈련의 안정성과 일관성을 최적화합니다. 이를 통해 코드 및 스케일 정보를 효과적으로 통합하고, GAN의 성능을 향상시킵니다.

- **Performance Highlights**: 이 방법은 최신 기술을 따르며, 실험 결과에서는 더 높은 품질의 음악을 자동 생성할 수 있는 효과가 입증되었습니다. 또한, 코드 분석 모델을 통해 생성된 음악은 더 조화롭고 구조적인 특성을 가지며, 인간 작곡에 가까운 음악 생성이 가능해졌습니다.



### ViRED: Prediction of Visual Relations in Engineering Drawings (https://arxiv.org/abs/2409.00909)
Comments:
          8 pages, 5 figures

- **What's New**: 본 논문에서는 ViRED라는 새로운 비전 기반의 관계 탐지 모델을 제안하여 전기 공학 도면에서 표와 회로 간의 관계를 식별함으로써 기존 방법론보다 높은 정확도를 달성했습니다.

- **Technical Details**: ViRED 모델은 비전 인코더(vision encoder), 객체 인코더(object encoder), 관계 디코더(relation decoder)로 구성됩니다. PyTorch를 사용하여 구현되었으며, 전기 공학 도면 데이터셋을 통해 관계 예측 작업에서 96%의 정확도를 기록했습니다.

- **Performance Highlights**: 시행한 여러 실험 결과, ViRED는 단일 전기 공학 도면에 많은 객체가 포함되어 있는 경우에도 빠른 속도로 추론할 수 있음을 보여줍니다.



### Multi-scale Temporal Fusion Transformer for Incomplete Vehicle Trajectory Prediction (https://arxiv.org/abs/2409.00904)
- **What's New**: 이 논문은 결측값이 있는 차량 궤적 예측을 해결하기 위한 새로운 엔드 투 엔드 프레임워크인 Multi-scale Temporal Fusion Transformer (MTFT)를 제안합니다. 이 모델은 Multi-scale Attention Head (MAH)와 Continuity Representation-guided Multi-scale Fusion (CRMF) 모듈로 구성되어 있습니다.

- **Technical Details**: MTFT 프레임워크는 다중 시간 규모에서 궤적의 모션 표현을 캡처하고 융합하여 예측의 질을 향상시킵니다. MAH는 다중 헤드 어텐션 메커니즘을 활용하여 서로 다른 시간 단위에서 궤적의 모션 표현을 병렬로 캡처하며, CRMF 모듈은 이러한 표현을 융합하여 강력한 시간 특징을 생성합니다. 이 과정에서 궤적의 연속 표현이 추출되어 융합을 안내합니다.

- **Performance Highlights**: 실험 결과, MTFT는 HighD 데이터셋에서 39% 이상의 성능 향상을 보이며 기존의 최신 모델에 비해 뛰어난 성능을 보여줍니다. 또한, 다양한 교통 시나리오에서 네 가지 데이터셋을 평가하여 이 모델의 유효성을 입증했습니다.



### MarsCode Agent: AI-native Automated Bug Fixing (https://arxiv.org/abs/2409.00899)
Comments:
          Yizhou Liu and Pengfei Gao contributed equally and the order is determined by rolling the dice. Chao Peng is the corresponding author

- **What's New**: 최근 연구에서 대형 언어 모델(Large Language Models, LLMs)을 활용하여 코드 완성, 테스트 생성 및 버그 수정과 같은 다양한 소프트웨어 개발 작업을 자동화할 수 있는 가능성이 커졌습니다. 본 논문에서는 소프트웨어 코드에서 버그를 자동으로 인식하고 수정하는 혁신적인 프레임워크인 MarsCode Agent를 소개합니다.

- **Technical Details**: MarsCode Agent는 LLM과 고급 코드 분석 기술을 결합하여 결함을 정확하게 식별하고 패치를 생성합니다. 이 시스템은 계획, 버그 재현, 결함 식별, 후보 패치 생성 및 검증의 시스템적 과정을 따릅니다. MarsCode Agent는 다중 에이전트 협업 프레임워크를 개발하여 문제의 특성에 따라 정적 또는 동적 해결 파이프라인을 할당합니다. 또한 코드 편집 시, MarsCode Agent는 충돌 기반 코드 수정 설명과 정적 구문 검사를 사용하여 잘 형식화된 코드 패치를 생성합니다.

- **Performance Highlights**: SWE-bench라는 종합적인 실제 소프트웨어 프로젝트 벤치마크에서 MarsCode Agent를 평가한 결과, 기존의 자동 버그 수정 도구에 비해 높은 성공률을 기록하며 효과적인 버그 수정을 보여주었습니다. LLM의 능력을 체계적이고 구조적으로 활용하여 MarsCode Agent는 완전한 자율 소프트웨어 유지 관리의 길을 열어주고 있습니다.



### User-Specific Dialogue Generation with User Profile-Aware Pre-Training Model and Parameter-Efficient Fine-Tuning (https://arxiv.org/abs/2409.00887)
- **What's New**: 이 논문은 사용자 지정 대화(user-specific dialogues)에 대한 새로운 접근 방식을 제안합니다. 기존의 개인화된 대화 연구는 페르소나 설명(persona descriptions)에 기반한 가상 사용자 대화에 초점을 맞추었지만, 본 연구는 실제 사용자 대화를 재현하는 것을 목표로 합니다. 이를 위해 사용자 대화 이력을 활용한 파라미터 효율적인 파인튜닝(parameter-efficient fine-tuning) 방법을 도입하고 있습니다.

- **Technical Details**: 사용자 프로필을 포함한 사전 훈련된 대화 모델(pre-trained dialogue model)과 결합된 학습 방법을 제안합니다. 파라미터 효율적인 파인튜닝을 통해 전체 모델에 소수의 파라미터를 추가하여, 적은 양의 훈련 데이터로도 효율적으로 훈련할 수 있고 모델 파괴에 강한 특징을 가지고 있습니다. 또한 자동으로 추론된 사용자 프로필에 대한 간단한 프롬프트(prompts)를 추가하여 사전 훈련된 모델이 사용자 프로필에 대한 지식을 향상시킨 발화를 생성할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 사용자의 개인 정보를 포함한 프롬프트를 사용한 대규모 언어 모델(large language model) 발화 생성 방식과 비교했을 때, 실제 사용자 발화와 더 높은 재현성을 가지는 발화를 생성할 수 있음을 보여주었습니다. 이는 적은 모델 규모에서도 가능했습니다.



### Beyond Parameter Count: Implicit Bias in Soft Mixture of Experts (https://arxiv.org/abs/2409.00879)
Comments:
          21 pages, 5 figures, 13 tables

- **What's New**: 최근 도입된 Soft MoE는 Sparse MoE의 이산적 라우팅 메커니즘을 대체하는 미분 가능한 게이팅 함수(differentiable gating function)를 사용합니다. 이는 토큰을 부드럽게 혼합하여 다양한 훈련 불안정을 완화시키지만, Soft MoE의 표현력이나 전문가 전문화 가능성에 영향을 미치는 편향이 있는지는 확실하지 않았습니다.

- **Technical Details**: Soft MoE는 단일 전문가와 임의의 강력한 전문가로 구성되어도 단순한 볼록 함수(convex functions)를 표현할 수 없다는 것을 입증했습니다. 이는 전통적인 관점이 다수의 작은 전문가들이 같은 총 매개변수 수를 가지고 있는 단일 대형 전문가의 표현력을 모방할 수 있다고 주장하는 것을 반박하며, 실제로는 여러 전문가가 필요하다는 것을 보여줍니다.

- **Performance Highlights**: 전문가 수를 늘리면서 총 매개변수 수를 고정하면 아키텍처가 특정 입력 레이블을 예측하기 위해 전문화된 전문가 집합(expert subset)을 효율적으로 근사할 수 있도록 편향된다는 것을 경험적으로 증명했습니다. 이 방법은 추론(inference) 시 계산량을 줄이는 데 쉽게 구현될 수 있습니다.



### Equitable Skin Disease Prediction Using Transfer Learning and Domain Adaptation (https://arxiv.org/abs/2409.00873)
- **What's New**: 이 연구에서는 피부 상태 진단의 정확성을 향상시키기 위해 이전의 AI 모델들이 가진 편향성을 극복하는 방법을 제안합니다. 특히 다양한 피부 톤을 고려한 데이터셋(Diverse Dermatology Images, DDI)을 활용하여, AI의 진단 성능을 높이는 데 기여합니다. 오히려 기존 모델에서 발생했던 어두운 피부 색조에 대한 성능 저하 문제를 해결하기 위한 접근 방식으로, 이전의 광범위한 이미지 도메인에서의 지식을 전이(transfer learning)를 활용합니다.

- **Technical Details**: 이 연구에서 사용된 몇 가지 모델은 Med-ViT, YOLOv8-Chest, RETFound로, 각 모델은 다양한 의료 이미지로부터 학습된 특징을 활용합니다. 특히 Vision Transformer 기반의 RETFound는 여러 도메인에서 전이된 특성을 활용하여 피부 질환 분류의 성능을 갖추고 있습니다. 모델은 DDI 데이터셋 뿐만 아니라 HAM10000과 같은 추가 피부 이미지 데이터셋으로 도메인 적응(domain adaptation)을 진행하여 성능을 더욱 향상시킵니다.

- **Performance Highlights**: 모델의 성능 평가 결과, Med-ViT가 다른 모델들 중에서 가장 뛰어난 성과를 보였습니다. 연구 결과는 다양한 피부 톤에서의 진단 성능을 높은 수준으로 끌어올리며, AI 도구의 포용성과 정확성을 높이는 데 기여할 것입니다. 이 연구는 특히 과소대표된 피부 톤 시나리오에서 전이 학습과 도메인 적응 기술이 어떻게 성능을 향상시킬 수 있는지를 보여줍니다.



### Harnessing the Power of Semi-Structured Knowledge and LLMs with Triplet-Based Prefiltering for Question Answering (https://arxiv.org/abs/2409.00861)
Comments:
          9 pages, published at IJCLR 2024

- **What's New**: 이 논문에서는 LLMs의 응답 품질을 크게 향상시킬 수 있는 4StepFocus라는 파이프라인을 제안합니다. 이 접근법은 모델이 relational context를 포착하고 스스로 기본적인 추론을 수행할 수 있는 능력을 활용하여 외부 지식에 대한 접근을 제공함으로써 이루어집니다.

- **Technical Details**: 4StepFocus는 1) LLM을 통한 triplet 생성을 통해 relational data를 추출하고, 2) 지식 그래프(knowledge graph)를 이용하여 답변 후보를 좁히고, 3) 관련 비구조 데이터와 벡터 유사도 검색(vector similarity search)을 통해 남은 후보를 정렬하며, 4) 제공된 배경 데이터를 통해 LLM이 최상의 후보를 재정렬하는 단계로 구성됩니다.

- **Performance Highlights**: 의학, 제품 추천 및 학술 논문 검색 테스트 세트에서 실험을 수행한 결과, 4StepFocus는 정보 검색에서 관련 traceable 배경 정보를 추가하며, 최신 방법들에 비해 성능을 유의미하게 개선한 것으로 나타났습니다.



### Trustworthy Human-AI Collaboration: Reinforcement Learning with Human Feedback and Physics Knowledge for Safe Autonomous Driving (https://arxiv.org/abs/2409.00858)
Comments:
          33 pages, 20 figures

- **What's New**: PE-RLHF(Physics-enhanced Reinforcement Learning with Human Feedback)는 인간의 피드백과 물리학적 지식을 통합하여 자율주행 정책의 안전성과 신뢰성을 높이는 혁신적인 접근 방식을 제공합니다.

- **Technical Details**: 이 프레임워크는 강화 학습(Reinforcement Learning) 훈련 루프에서 인간의 피드백(예: 인간 개입, 시연)과 물리학적 지식(예: 교통 흐름 모델)을 통합합니다. PE-RLHF는 학습된 정책이 주어진 물리 기반 정책보다 성능이 떨어지지 않도록 보장하여, 신뢰할 수 있는 안전성을 제공합니다.

- **Performance Highlights**: PE-RLHF는 다양한 운전 시나리오에서 기존 방법보다 우수한 성능을 보이며, 새로운 기술로서 안전성, 효율성 및 일반화 측면에서 SOTA(state-of-the-art) 성능을 달성합니다.



### Benchmarking LLM Code Generation for Audio Programming with Visual Dataflow Languages (https://arxiv.org/abs/2409.00856)
- **What's New**: 본 연구에서는 메타프로그래밍 코드 표현을 통한 LLM(대형 언어 모델)의 코드 생성 성능을 시각적 프로그래밍 언어에서 오디오 프로그래밍 작업에 대한 여러 레벨의 표현에서 탐구합니다. 또한, 코드 생성의 새로운 접근 방식을 평가하기 위해 오디오 디지털 신호 처리(DSP)를 위한 벤치마크 세트를 제안합니다.

- **Technical Details**: 연구는 MaxMSP와 MaxPy를 비롯한 두 가지 시각적 언어와 Web Audio API에 대한 코드 생성을 검토하며, 각 언어의 코드 생성은 JSON 형식을 활용합니다. 메타프로그래밍 및 JSON을 통한 직접 노드 생성 방식을 비교하고, LLM이 생성한 코드의 정합성을 측정하는 새로운 메트릭을 정의합니다.

- **Performance Highlights**: 실험 결과, 메타프로그래밍을 통한 코드 생성이 문법적으로 올바른 경우 더 의미적으로 정확한 코드를 생성하는데 기여함을 발견했습니다. 또한, 무작위 함수 및 루프를 활용한 풍부한 메타프로그래밍 요청은 더 복잡한 코드를 생성하는 데 기여했습니다.



### The Design of an LLM-powered Unstructured Analytics System (https://arxiv.org/abs/2409.00847)
Comments:
          6 pages, 3 figures, fixed typos

- **What's New**: 이 논문에서는 LLMs (Large Language Models)를 활용한 새로운 비구조적 분석 시스템인 Aryn의 설계와 그 동기 부여가 되는 사용 사례를 설명합니다. Aryn은 자연어로 쿼리를 지정하면, 이를 통해 문서의 의미적 계획을 자동으로 결정하고 실행하여 대형 비구조적 데이터에서 답변을 도출합니다.

- **Technical Details**: Aryn의 핵심은 Sycamore라는 선언적 문서 처리 엔진으로, 이는 Ray를 기반으로 구축되었습니다. Sycamore는 DocSets라는 신뢰할 수 있는 분산 추상화를 제공하며, 사용자는 대규모로 복잡한 문서를 분석하고, 강화하고, 변환할 수 있습니다. Aryn에는 자연어 쿼리를 Sycamore 스크립트로 변환하는 Luna라는 쿼리 플래너와 원시 PDF 및 문서 이미지를 DocSets로 변환하는 Aryn Partitioner가 포함되어 있습니다.

- **Performance Highlights**: Aryn은 실제 응용 사례로 NTSB (National Transportation Safety Board) 사고 보고서를 분석하는 예시를 보여줍니다. 이 전반에서 Aryn 시스템의 작동 방식을 여러 가지 사용 사례에 걸쳐 시연하며, 사용자 인터페이스를 통해 생성된 계획을 검사, 분석 및 디버그하는 과정과 대규모 비구조적 문서를 쉽게 분석할 수 있는 Sycamore 프로그래밍 프레임워크의 단순함을 강조합니다.



### Report Cards: Qualitative Evaluation of Language Models Using Natural Language Summaries (https://arxiv.org/abs/2409.00844)
Comments:
          11 pages, 8 figures

- **What's New**: 대규모 언어 모델(LLMs)의 평가 방식을 개선하기 위해 'Report Cards'라는 새로운 방법론을 제안하였습니다. Report Cards는 모델의 특정 기술이나 주제에 대한 행동을 요약한 인간 친화적인 자연어 형식의 문서입니다.

- **Technical Details**: 이 논문에서는 Report Cards의 품질을 평가하기 위한 세 가지 기준인 specificity(구체성), faithfulness(충실성), interpretability(해석 가능성)를 제시합니다. 또한, LLMs의 출력을 바탕으로 Report Cards를 생성하는 PRESS라는 반복적인 알고리즘도 소개합니다. 실험에서는 다양한 LLMs의 Report Cards가 일반적인 정량적 벤치마크를 넘어서는 인사이트를 제공함을 보여주었습니다.

- **Performance Highlights**: Report Cards는 기존의 벤치마크 지표들이 간과할 수 있는 모델의 고유한 성능을 정확하게 캡처하고, 모델 간의 명확한 구분을 가능하게 하여, LLM의 평가 기준을 더욱 포괄적이고 해석 가능한 방향으로 확장합니다.



### Entropy Loss: An Interpretability Amplifier of 3D Object Detection Network for Intelligent Driving (https://arxiv.org/abs/2409.00839)
- **What's New**: 이 논문에서는 인공지능 주행에서의 안전 인식의 중요성이 높아짐에 따라 기존의 딥러닝 방식의 한계를 극복하는 새로운 단위 손실 함수인 ‘Entropy Loss’를 제안합니다. 이 손실 함수는 기능 압축 네트워크의 특징을 바탕으로 설계되었습니다.

- **Technical Details**: Entropy Loss는 정보 전송 프로세스의 정밀한 변화를 모델링하여, 각 레이어의 출력과 정보량을 연속 랜덤 변수로 표현함으로써 정보 엔트로피의 변화를 정량화합니다. 이를 통해 네트워크 파라미터 업데이트가 가능해지며, 해석 가능성을 개선합니다.

- **Performance Highlights**: 실험 결과, Entropy Loss 사용한 3D 객체 탐지 모델의 KITTI 테스트 세트에서의 정확도가 비사용 모델에 비해 최대 4.47% 향상되어 학습 과정이 가속화되었음을 보여줍니다.



### Serialized Speech Information Guidance with Overlapped Encoding Separation for Multi-Speaker Automatic Speech Recognition (https://arxiv.org/abs/2409.00815)
- **What's New**: 이 논문에서는 multiple speakers 음성 인식을 위한 새로운 접근 방식을 제안합니다. 특히, Overlapped Encoding Separation (EncSep) 기법을 사용하여 CTC(연결주의 시간 분류)와 attention hybrid loss를 결합하여 성능을 더욱 향상시키는 방법을 소개합니다. 또한, 이를 통해 single-speaker 정보를 강조하여 decoding 과정에서의 attention을 향상시킵니다.

- **Technical Details**: EncSep 방식은 encoder 뒤에 추가적인 separator를 삽입하여 여러 화자의 정보를 추출하는 방식을 사용합니다. separated encodings는 speaking time에 따라 정렬되며, CTC 손실이 계산됩니다. 추가적으로 GEncSep 방식은 분리된 인코딩을 결합하여 decoding 과정에서 single-speaker 정보를 가이드합니다.

- **Performance Highlights**: 실험 결과, LibriMix 데이터셋에서 단일 화자 인코딩이 overlapping 인코딩으로부터 효과적으로 분리될 수 있음을 보여주었으며, CTC 손실이 복잡한 시나리오에서도 encoder 표현을 향상시키는 데 도움이 되었습니다. GEncSep는 기존 성능을 더욱 높이는 데 기여하였습니다.



### A Novel Self-Attention-Enabled Weighted Ensemble-Based Convolutional Neural Network Framework for Distributed Denial of Service Attack Classification (https://arxiv.org/abs/2409.00810)
Comments:
          19 pages, 3 tables, 9 figures

- **What's New**: 본 논문에서는 분산 서비스 거부(DDoS) 공격을 정확하게 탐지하기 위한 새로운 접근 방식을 제안합니다. 전통적인 방법들이 다양한 특징을 추출하는 데 어려움을 겪는 반면, 이 연구는 세 가지 독특한 CNN 아키텍처를 결합한 방법을 사용합니다.

- **Technical Details**: 제안된 방법은 SA-Enabled CNN과 XGBoost, SA-Enabled CNN과 LSTM, SA-Enabled CNN과 Random Forest를 결합하여 사용합니다. 각 모델은 여러 스케일에서 특징을 추출하며, self-attention 메커니즘을 통해 특징 통합 및 관련성을 강화합니다.

- **Performance Highlights**: 제안된 방법은 98.71%의 precision, 98.66%의 F1-score, 98.63%의 recall 및 98.69%의 accuracy를 달성하여 전통적인 방법들을 능가하며 DDoS 공격 탐지 분야에서 새로운 벤치마크를 설정했습니다.



### Diffusion based multi-domain neuroimaging harmonization method with preservation of anatomical details (https://arxiv.org/abs/2409.00807)
- **What's New**: 이 논문에서는 다중 센터의 신경촬영(neuroimaging) 연구에서 발생하는 기술적 변동성(batch effect)에 대한 새로운 접근 방식을 제시합니다. 기존의 Generative Adversarial Networks (GAN) 기반의 화합(harmonization) 방법에 비해, Denoising Diffusion Probabilistic Model을 활용하여 고온송 취소 효과(generated artifacts)나 해부학적 왜곡(anatomical distortion)을 최소화하는 방법이 소개됩니다.

- **Technical Details**: Multi-domain neuroimaging harmonization을 위한 새로운 접근 방식을 제안하며, 이는 학습된 도메인 불변 조건(domain invariant condition)을 통해 이루어집니다. Denoising diffusion 모델을 이용하여 각 도메인 간의 변동성을 효과적으로 억제하고, 해부학적 세부 사항(anatomical details)을 보존하는 데 중점을 둡니다. 제안된 모델은 각 확산 단계에서 배치 간의 차이를 구분하며, 이를 통해 더 높은 화합 결과를 얻습니다.

- **Performance Highlights**: ADNI1 및 ABIDE II의 두 공공 데이터셋을 활용하여 GAN 기반 방법보다 뛰어난 화합 결과를 도출했습니다. PVS(Perivascular Spaces) 세분화(segmentation) 분석에 있어서도 스캐너 효과의 일관성을 개선하는 결과를 보여주었습니다. 최신 진행된 연구에서 제안하는 모델은 GAN 기반 방법에 비해 휘도(brightness)와 세부 사항에서 더 뛰어난 성능을 보였습니다.



### The Dark Side of Human Feedback: Poisoning Large Language Models via User Inputs (https://arxiv.org/abs/2409.00787)
- **What's New**: 대형 언어 모델(LLMs)의 훈련 과정에서 사용자 피드백을 활용한 정렬 프로세스에 의존하는 시점에 새로운 유형의 사용자 주도 중독 공격이 발생할 수 있는 가능성을 제기합니다.

- **Technical Details**: 이 연구에서는 두 가지 공격 메커니즘을 소개합니다: (1) 선택 기반 메커니즘은 높은 보상 점수를 기록하는 유독한 응답을 유도하고, (2) 생성 기반 메커니즘은 최적화 가능한 프리픽스를 사용하여 모델 출력을 제어합니다. 1%의 악성 프롬프트를 데이터를 통해 주입함으로써 특정 트리거 단어 사용 시 독성 점수가 2배 증가하는 결과를 보여줍니다.

- **Performance Highlights**: 일반적으로 LLaMa-2 및 다양한 크기의 GPT-3 모델에 대한 평가를 통해, 우리가 실시한 사용자 주도 중독 공격은 트리거가 존재할 때 응답의 독성 점수를 26.5-226.9% 증가시켰습니다. 반면, 트리거가 프롬프트에 없을 때는 독성이 낮게 유지됩니다.



### Trusted Unified Feature-Neighborhood Dynamics for Multi-View Classification (https://arxiv.org/abs/2409.00755)
Comments:
          Ongoing work: 13pages, 13figures, 12 tables

- **What's New**: 새로운 연구에서는 Trusted Unified Feature-NEighborhood Dynamics (TUNED) 모델을 제안하여 다중 시점 분류(Multi-view Classification, MVC)에서의 불확실성과 갈등을 효과적으로 해결합니다. 이 모델은 지역과 글로벌 피처-이웃 구조를 통합하여 보다 강력한 의사결정을 수행합니다.

- **Technical Details**: TUNED 모델은 각 뷰 내에서의 지역 피처-이웃 구조를 추출하고, 선택적 마르코프 랜덤 필드(Selective Markov Random Field, S-MRF)를 통해 뷰 간의 의존성을 동적으로 관리합니다. 이를 통해 다양한 뷰 간의 정보를 효과적으로 결합할 수 있습니다.

- **Performance Highlights**: 실험 결과, TUNED 모델은 기존의 방법들에 비해 정확성과 강인성을 개선하며, 특히 높은 수준의 불확실성과 갈등이 있는 상황에서 우수한 성능을 보입니다.



### MaskGCT: Zero-Shot Text-to-Speech with Masked Generative Codec Transformer (https://arxiv.org/abs/2409.00750)
- **What's New**: 이번 연구에서는 Masked Generative Codec Transformer(MaskGCT)라는 새로운 비자기회귀(non-autoregressive) 모델을 소개합니다. 이 모델은 텍스트와 음성 간의 정밀한 정렬 정보를 요구하지 않으며, 두 단계로 구성되어 있습니다. 첫 번째 단계에서는 텍스트를 사용하여 음성 자기지도 학습(self-supervised learning) 모델에서 추출된 의미 토큰을 예측하고, 두 번째 단계에서는 이러한 의미 토큰을 기반으로 음향 토큰을 예측합니다.

- **Technical Details**: MaskGCT는 'mask-and-predict' 학습 패러다임을 따르며, 훈련 과정에서 마스킹된 의미 또는 음향 토큰을 주어진 조건과 프롬프트에 따라 예측하는 방법을 학습합니다. 추론 시, 모델은 지정된 길이의 토큰을 병렬로 생성합니다. 또한, VQVAE(Van Den Oord et al., 2017)를 학습하여 자기지도 음성 학습 기능을 정량화하고, 이로 인해 의미 특징의 정보 손실을 최소화합니다.

- **Performance Highlights**: MaskGCT는 품질, 유사성 및 이해 가능성 면에서 최신 제로샷 TTS 시스템과 비교할 때 우수하거나 경쟁력 있는 성능을 보여주었습니다. 특히, LibriSpeech, SeedTTS test-en 및 SeedTTS test-zh 데이터셋에서 생성된 음성은 공통의 지표에서 높은 자연스러움과 유사성을 달성했습니다. MaskGCT는 생성된 음성의 총 길이를 제어할 수 있는 기능도 보여줍니다.



### Interpretable Clustering: A Survey (https://arxiv.org/abs/2409.00743)
Comments:
          11 pages, 2 figures

- **What's New**: 이 논문은 설명 가능한 클러스터링 알고리즘에 대한 종합적인 리뷰를 제공하며, 클러스터링의 해석 가능성을 높이기 위한 방법들을 제안합니다. 특히, 의료 및 금융 분야와 같은 중요한 분야에서의 필요성을 강조하고, 기존 클러스터링 방법들과의 차별성을 알아봅니다.

- **Technical Details**: 이 연구에서는 설명 가능한 클러스터링 방법에 대한 새로운 분류법을 제안하고, 클러스터링 과정을 세 가지 단계로 구분합니다: (1) 특징 선택 단계(pre-clustering), (2) 모델 구축 단계(in-clustering), (3) 모델 설명 단계(post-clustering). 각 단계에서의 해석 가능성을 고려하여 다양한 방법들을 체계적으로 정리합니다.

- **Performance Highlights**: 이 논문은 다양한 클러스터링 방법들이 사용자 요구와 맥락에 맞춰 선택될 수 있도록 돕고, 해석 가능한 결과를 제공하여 신뢰성을 확보하는 데 기여할 수 있는 시스템적인 틀을 제시합니다.



### Simulation of Social Media-Driven Bubble Formation in Financial Markets using an Agent-Based Model with Hierarchical Influence Network (https://arxiv.org/abs/2409.00742)
Comments:
          11 pages, 7 figures, To appear in Proceedings of 36th European Modeling and Simulation Symposium (EMSS), 21st International Multidisciplinary Modelling and Simulation Multiconference (I3M), Tenerife, Spain, Sep. 2024

- **What's New**: 이 연구는 금융 시장의 사회적 미디어 영향과 투자자 행동 간의 교차점을 강조하는 새로운 agent-based 모델을 제안하고 있습니다. 이 모델은 계층적 구조를 통해 시장의 emergent behaviour(출현 행동)을 효과적으로 모델링할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 새로운 모델은 Lux-Marchesi 모델을 확장하여 거래 에이전트를 계층적 네트워크 내의 커뮤니티로 임베드함으로써 사회적 미디어의 영향을 반영합니다. 이 계층적 구조는 트레이더의 전략과 의견을 형성하는데 중요한 역할을 합니다. 모델은 echo chambers와 pump-and-dump 스킴의 효과를 현실적으로 시뮬레이트할 수 있습니다.

- **Performance Highlights**: 모델의 분석 결과는 실제 금융 시장에서 관찰되는 여러 가지 스타일화된 사실(stylized facts)을 일관되게 따르며, 소셜 미디어 영향에 따른 시장의 행동 변화를 효과적으로 설명합니다. 이로써 금융시장 참여자들의 새로운 행동 패턴과 그로 인한 시장 역학을 이해하는 데 기여합니다.



### LPUWF-LDM: Enhanced Latent Diffusion Model for Precise Late-phase UWF-FA Generation on Limited Datas (https://arxiv.org/abs/2409.00726)
Comments:
          13 pages, 7 figures

- **What's New**: 본 연구는 Ultra-Wide-Field Scanning Laser Ophthalmoscopy (UWF-SLO)에서 제한된 데이터로부터 고품질 Late-Phase UWF-FA 이미지를 생성할 수 있는 개선된 Latent Diffusion Model을 소개합니다. 이를 통해 기존의 유해한 염료 주입 없이 정확한 안과 질병 진단이 가능해집니다.

- **Technical Details**: 제안된 모델은 Cross-temporal Regional Difference Loss (CTRD Loss)를 사용하여 Early-phase와 Late-phase UWF-FA 간의 차이를 강조합니다. 저주파 정보를 향상시키기 위한 저주파 강화 노이즈 기법을 diffusion 과정에 적용함으로써 의료 이미지의 현실감을 개선하고, Gated Convolutional Encoder를 통해 추가 정보를 추출하여 변동성이 있는 데이터셋에서도 우수한 성능을 발휘합니다.

- **Performance Highlights**: 제안된 Latent Diffusion Model for Ultra-Wide-Field Late-Phase Fluorescein Angiography (LPUWF-LDM)는 기존의 방법들보다 세부정보를 효과적으로 재구성하며, 한정된 데이터셋으로 작업할 시 최첨단 성능을 달성했습니다. 다양한 정량적 및 정성적 지표에서 우수한 결과를 보여주었습니다.



### BUET Multi-disease Heart Sound Dataset: A Comprehensive Auscultation Dataset for Developing Computer-Aided Diagnostic Systems (https://arxiv.org/abs/2409.00724)
Comments:
          14 pages, 13 figures

- **What's New**: 이번 논문에서는 심혈관 질환(CVD) 진단을 위한 주목할 만한 데이터셋인 BUET Multi-disease Heart Sound (BMD-HS) 데이터셋을 소개합니다. 이 데이터셋은 864개의 심장 소리 녹음을 포함하며, 전반적인 심장 질환 스펙트럼을 다룸으로써 스스로 진단을 자동화 할 수 있는 기계 학습 모델 개선의 기초 자료로 사용될 전망입니다.

- **Technical Details**: BMD-HS 데이터셋은 5개의 서로 다른 심장 소리 클래스를 포함하고 있으며, 혁신적인 multi-label annotation 시스템을 통해 다양한 질병 및 고유한 질병 상태를 포착합니다. 데이터는 표준화된 수집 방법을 통해 생성되며, 데이터의 신뢰성을 높이기 위해 모든 녹음은 동일한 청진기를 사용하여 수집됩니다. 또한, 데이터셋은 심장 초음파를 통한 진단 확인을 포함하고 있으며, 이에 따라 임상적 관련성을 강화합니다.

- **Performance Highlights**: BMD-HS 데이터셋은 심장 소리 분석 및 심혈관 질환의 조기 발견에 유망한 도구로, 데이터의 일관성과 신뢰성을 높이는 데 기여합니다. 특히, 다중 이상 상태를 동시에 반영할 수 있는 가능성으로 인해 기계 학습 모델의 성능을 향상시키는 데 중요한 자산이 될 것으로 기대됩니다. 데이터셋은 공개되어 있어 연구자들이 자유롭게 접근할 수 있습니다.



### Who Would Chatbots Vote For? Political Preferences of ChatGPT and Gemini in the 2024 European Union Elections (https://arxiv.org/abs/2409.00721)
- **What's New**: 본 연구는 2024 유럽 의회 선거와 관련하여 대규모 언어 모델에 기반한 챗봇(ChatGPT, Gemini)의 정치적 편향을 조사합니다.

- **Technical Details**: 이 연구는 27개 EU 회원국에서 유럽 의회를 대표하는 정치 정당에 대한 평가를 수행하기 위해 표준화된 프롬프트를 사용하여 매일 데이터를 수집했습니다. Gemini는 정치적 질문에 대답하지 않는 반면, ChatGPT는 일관된 평가를 제공했습니다.

- **Performance Highlights**: ChatGPT는 좌파 및 중도 정당을 선호하는 경향을 보였고, 그린/유럽 자유 동맹당에 대해 가장 높은 평가를 주었습니다. 반면, 우파 정당, 특히 정체성과 민주주의 그룹은 가장 낮은 평가를 받았습니다. 정치적 편향성을 비판적으로 접근할 필요성과 투명성과 규제의 필요성이 강조되었습니다.



### Multiscale Color Guided Attention Ensemble Classifier for Age-Related Macular Degeneration using Concurrent Fundus and Optical Coherence Tomography Images (https://arxiv.org/abs/2409.00718)
Comments:
          27th International Conference on Pattern Recognition (ICPR) 2024

- **What's New**: 이 논문은 Age-related Macular Degeneration (AMD) 분류를 위해 주의 메커니즘을 통합한 모달리티별 다중 스케일 색 공간 임베딩 (MCGAEc)을 제안합니다. 이 방식은 서로 다른 이미지 모달리티에서 중요한 정보를 효과적으로 추출할 수 있습니다.

- **Technical Details**: MCGAEc 모델은 YCbCr 및 HSV 색 공간을 사용하여 Fundus 이미지에서 중요한 특성을 캡처합니다. 각 색 공간은 Pre-trained VGG16 모델을 통해 읽고, 자가 주의 메커니즘을 사용하여 특징을 집계한 후 랜덤 포레스트 분류기(RFC)로 전달합니다.

- **Performance Highlights**: MCGAEc 방법은 Project Macula에서 제공하는 공개 다중 모달리티 데이터셋을 사용하여 실험되었으며, 단일 모달리티(단일 Fundus 또는 OCT) 방법과의 성능 비교를 통해 효과를 입증하였습니다.



### Multi-Agent Reinforcement Learning from Human Feedback: Data Coverage and Algorithmic Techniques (https://arxiv.org/abs/2409.00717)
- **What's New**: 이 논문에서는 Multi-Agent Reinforcement Learning from Human Feedback (MARLHF)에 대한 이론적 기초와 실증적 검증을 탐구합니다. 특히, 일반합 게임에서의 선호 기반 오프라인 데이터셋을 통해 Nash equilibrium을 식별하는 작업을 정의하고, 데이터셋 커버리지 조건 및 알고리즘 기법을 제안합니다.

- **Technical Details**: 이론적으로, 본 연구는 효과적인 MARLHF에서 Nash equilibrium을 학습하기 위한 데이터셋 커버리지 조건을 특징짓습니다. 실험적으로, 우리는 보상 정규화와 모방 학습을 포함한 두 가지 알고리즘 기법을 소개하고, 이를 통해 효과적인 보상 분포를 달성하고 학습 결과를 개선합니다.

- **Performance Highlights**: 종합적인 실험을 통해 unilateral coverage의 이론적 필요성을 확인하였으며, 다양한 Multi-Agent Particle Environment (MPE) 시나리오에서 실험 결과를 통해 보상 정규화 계수 및 데이터셋 다양성이 성능 향상에 기여함을 입증하였습니다.



### ReMOVE: A Reference-free Metric for Object Erasur (https://arxiv.org/abs/2409.00707)
Comments:
          Accepted at The First Workshop on the Evaluation of Generative Foundation Models (EvGENFM) at CVPR 2024

- **What's New**: 이번 논문에서는 	exttt{ReMOVE}라는 참조 이미지 없이 객체 제거 효과성을 평가하는 새로운 메트릭을 도입합니다. 기존의 메트릭인 LPIPS와 CLIPScore와 달리, 	exttt{ReMOVE}는 실제 시나리오에서 공통적으로 발생하는 참고 이미지가 없을 때의 평가 문제를 해결합니다.

- **Technical Details**: 	exttt{ReMOVE}는 기존의 메트릭이 필요로 하는 Ground Truth의 의존성을 줄이기 위해, ViT(Visual Transformer)를 활용하여 마스킹된 영역과 마스킹되지 않은 영역의 평균 패치 특성 간의 차이를 측정하여 품질을 평가합니다. 이 메트릭은 객체 제거와 교체를 효과적으로 구분할 수 있게 설계되었습니다.

- **Performance Highlights**: 실험 결과 	exttt{ReMOVE}는 최신의 참조 기반 메트릭들과 높은 상관관계를 보였으며, 인간의 인식과 잘 일치한다는 것을 보여주었습니다. 또한 실제 데이터 세트를 통해 테스트한 결과, 메트릭이 일관된 결과를 나타냈습니다. 이는 	exttt{ReMOVE}가 실제 이미지에 대한 깊이 있는 품질 평가를 가능하게 함을 의미합니다.



### Seeing Your Speech Style: A Novel Zero-Shot Identity-Disentanglement Face-based Voice Conversion (https://arxiv.org/abs/2409.00700)
- **What's New**: 본 논문에서는 얼굴 이미지를 활용하여 목표 화자의 목소리 스타일을 생성하는 새로운 작업인 Face-based Voice Conversion (FVC)에 대해 다룹니다. 특히, 기존 방법의 두 가지 단점을 해결한 Identity-Disentanglement Face-based Voice Conversion (ID-FaceVC) 방법을 제안합니다.

- **Technical Details**: ID-FaceVC는 두 가지 주요 모듈을 포함합니다: (1) Identity-Aware Query-based Contrastive Learning (IAQ-CL) 모듈로, 얼굴 특징을 추출하고 (2) Mutual Information-based Dual Decoupling (MIDD) 모듈로 음성 특징을 정제합니다. 이 밖에도, 오디오 혹은 텍스트 입력을 모두 받으며 감정 톤과 속도 조정이 가능합니다.

- **Performance Highlights**: ID-FaceVC는 여러 성능 지표에서 SOTA 성능을 보였으며, 자연스러움, 유사성 및 다양성 측면에서 효과적임을 사용자 연구를 통해 확인하였습니다.



### Polyrating: A Cost-Effective and Bias-Aware Rating System for LLM Evaluation (https://arxiv.org/abs/2409.00696)
- **What's New**: 최근의 대형 언어 모델(LLMs)의 효과적인 성능 평가를 위해 미리 평가 기반의 인간 평가 방법이 중요해졌습니다. 그러나 현재의 평가 시스템은 평가 결과에 영향을 미치는 인간의 편향을 고려하지 않으며, 정확한 평가를 위해 대규모의 비싼 데이터셋이 필요합니다. 이를 해결하기 위해, 우리는 Polyrating이라는 새로운 평가 시스템을 도입했습니다.

- **Technical Details**: Polyrating는 최대 사후 확률 추정(maximum a posteriori estimation)에 기반한 표현력 있고 유연한 평가 시스템으로, 자원의 소모를 최소화하며 모델 성능을 더 정교하고 철저하게 분석할 수 있게 합니다. 이 시스템은 인간의 선호에 영향을 미치는 편향을 탐지하고 정량화하여, 더 공정한 모델 비교를 가능하게 합니다.

- **Performance Highlights**: 새로운 모델의 경우 인간 평가 비용을 최대 41%까지 절감할 수 있으며, 새로운 작업에서는 최대 77%까지 비용을 줄일 수 있습니다. Polyrating는 또한 다양한 작업 간 평가를 직접 비교할 수 있게 하여, LLM의 강점, 약점 및 여러 응용 분야에서의 상대적 성능을 포괄적으로 이해하는 데 도움을 줍니다.



### Curriculum Prompting Foundation Models for Medical Image Segmentation (https://arxiv.org/abs/2409.00695)
Comments:
          Accepted by MICCAI 2024

- **What's New**: 본 연구에서는 의료 이미지 분할을 위한 SAM(Segment Anything Model)의 성능을 향상시키기 위해 자동화된 프롬프트 생성 기법을 제안합니다. 기존에 필요한 수작업 프롬프트 생성 과정을 줄이고, 다양한 타입의 프롬프트를 효율적으로 결합하는 방법을 소개합니다.

- **Technical Details**: 연구에서 제안하는 크리큘럼 프롬프팅(Curriculum Prompting) 기법은 코스(Coarse)에서 파인(Fine)으로 진행되는 다양한 유형의 프롬프트를 시스템적으로 통합하여 세분화 문제를 해결합니다. SAM의 기존 구조를 활용하여, 이미지 인코더와 프롬프트 인코더를 통해 세분화 마스크를 생성하는 과정에서, 박스 프롬프트와 포인트 프롬프트를 결합하여 프롬프트의 종류를 다각화합니다.

- **Performance Highlights**: 세 가지 공개 의료 데이터세트에서 실험한 결과, 제안한 기법은 기존 SAM 기반의 의료 이미지 분할 방법보다 정량적, 정성적으로 우수한 성능을 보였습니다. 자동 프롬프트 생성과 크리큘럼 프롬프팅의 결합은 세분화 결과를 크게 향상시켰습니다.



### When Heterophily Meets Heterogeneous Graphs: Latent Graphs Guided Unsupervised Representation Learning (https://arxiv.org/abs/2409.00687)
Comments:
          14 pages

- **What's New**: 본 논문에서는 비지도 이질 그래프 표현 학습(Unsupservised Heterogeneous Graph Representation Learning, UHGRL) 분야에서 의미론적 이질성(semantic heterophily)의 중요성을 강조하며 새로운 프레임워크인 Latent Graphs Guided Unsupervised Representation Learning (LatGRL)을 제안합니다. 이 연구는 특히, 동종 메타 경로에 연결된 동일 유형의 노드들 간의 속성 차이를 다룹니다.

- **Technical Details**: LatGRL 프레임워크는 전역 구조(global structures)와 노드 속성(node attributes)을 결합한 유사성 탐색(similarity mining) 방법을 통해 세분화된 동종(homophilic) 및 이종(heterophilic) 잠재 그래프(latent graphs)를 구축하여 표현 학습을 안내합니다. 게다가 적응형 이중 주파수 의미 융합 메커니즘(adaptive dual-frequency semantic fusion mechanism)을 도입하여 노드 수준의 의미론적 이질성을 처리하고 대규모 실제 데이터를 다루기 위한 확장 가능한 구현(scalable implementation)을 설계합니다.

- **Performance Highlights**: 제안된 LatGRL 프레임워크는 평가 데이터셋에서 널리 실험되었으며, 노드 분류(node classification) 및 군집화(clustering) 작업에서 효과성과 효율성을 입증했습니다. 이를 통해 LatGRL은 기존의 UHGRL 방법들보다 의미론적 이질성을 효과적으로 다루고, 실제 애플리케이션에서 그 가능성을 높였습니다.



### Comprehensive Botnet Detection by Mitigating Adversarial Attacks, Navigating the Subtleties of Perturbation Distances and Fortifying Predictions with Conformal Layers (https://arxiv.org/abs/2409.00667)
Comments:
          46 pages

- **What's New**: 최신 연구는 Botnet 공격에 대응하기 위한 머신 러닝 기반 탐지 시스템의 취약성을 다루며, 이를 통해 adversarial attacks(적대적 공격)의 영향과 회피 전략을 분석합니다.

- **Technical Details**: 이 연구에서는 Genetic Algorithm(GA) 및 Particle Swarm Optimization(PSO) 등을 활용하여 머신 러닝 모델의 최적 하이퍼파라미터를 조정하며, C&W attack과 GAN(Generative Adversarial Network) 공격을 통해 특징 조작을 진행합니다. 여러 모델의 취약성을 분석하여 adversarial examples의 전이 가능성을 탐구합니다.

- **Performance Highlights**: 최종적으로, 통계 기반의 conformal prediction을 도입하여 잘못된 예측을 58.20 % (ISCX 데이터셋) 및 98.94 % (ISOT 데이터셋) 비율로 거부하여 예측 정확도를 크게 향상시켰습니다.



### Nasdaq-100 Companies' Hiring Insights: A Topic-based Classification Approach to the Labor Mark (https://arxiv.org/abs/2409.00658)
Comments:
          17 pages, 4 figures, 1 table. Presented at the International Conference on Optimization and Data Science in Industrial Engineering (ODSIE 2023)

- **What's New**: 본 연구는 데이터 분석에 기반한 노동 시장 인사이트 기법을 활용하여 온라인 노동 시장의 직업 분류를 제안합니다.

- **Technical Details**: 구조적 주제 모델링(Structural Topic Modeling)을 방법론으로 사용하여 NASDAQ-100에 등록된 기업의 LinkedIn 온라인 직업 공고 데이터를 분석합니다.

- **Performance Highlights**: 13개의 직업 카테고리 중에서 마케팅, 브랜딩 및 판매; 소프트웨어 엔지니어링; 하드웨어 엔지니어링; 산업 엔지니어링; 그리고 프로젝트 관리가 가장 자주 게시되는 직업 분류로 나타났습니다.



### Artificial Intelligence in Gastrointestinal Bleeding Analysis for Video Capsule Endoscopy: Insights, Innovations, and Prospects (2008-2023) (https://arxiv.org/abs/2409.00639)
- **What's New**: 이 논문은 Video Capsule Endoscopy (VCE)에서 위장관(GI) 출혈 탐지를 위한 머신 러닝(ML) 응용의 현재 상태를 종합적으로 검토하고 있습니다.

- **Technical Details**: 총 113개의 논문을 분석하여 ML 방법론, 평가 지표, 오픈 소스 데이터셋 및 기술 분류의 효과성 및 한계를 강조합니다. VCE는 카메라가 내장된 작은 캡슐을 삼켜 소화관을 통해 여행하면서 이미지를 캡처합니다. 이 방법은 진단 정확도를 높이기 위해 ML을 자동화하여 분석하는 것을 목표로 합니다.

- **Performance Highlights**: 기존의 진단 방법에 비해 VCE는 GI 출혈의 빠른 정확한 진단을 가능하게 하고, 인력에 대한 의존도를 줄이며, 진단 및 치료 시간을 단축합니다. ML의 적용은 진단 오류를 최소화하고, GI 출혈 탐지의 미래 연구 방향을 제시합니다.



### Entity-Aware Biaffine Attention Model for Improved Constituent Parsing with Reduced Entity Violations (https://arxiv.org/abs/2409.00625)
- **What's New**: 본 논문에서는 entity-aware biaffine attention 모델을 제안하여 constituency parsing에서 entity-violating 문제를 해결하고자 합니다. 기존 모델들이 entity 완전성을 간과하는 반면, 제안된 모델은 entity 정보를 효과적으로 활용합니다.

- **Technical Details**: 제안된 모델은 biaffine attention 메커니즘을 바탕으로 하며, entity role vector를 추가하여 구문 분석의 정확도를 향상시킵니다. 새로운 메트릭인 Entity Violating Rate (EVR)를 도입하여 구문 분석 결과의 entity 위반 정도를 정량화합니다.

- **Performance Highlights**: 세 가지 데이터셋 (ONTONOTES, PTB, CTB)에서 실험한 결과, 제안된 모델은 가장 낮은 EVR을 기록하면서도 기존 모델과 비교하여 높은 precision, recall, F1-score를 유지했습니다. 추가적으로, 문장 감정 분석과 같은 다운스트림 작업에서도 뛰어난 성능을 보여줍니다.



### Enhancing Vectorized Map Perception with Historical Rasterized Maps (https://arxiv.org/abs/2409.00620)
Comments:
          Accepted by ECCV 2024

- **What's New**: 이번 논문에서는 HRMapNet이라는 새로운 프레임워크를 제안하며, 이는 저비용의 역사적 래스터 맵(Historical Rasterized Map)을 활용하여 온라인 벡터화된 맵 인식을 개선하는 방법을 소개합니다.

- **Technical Details**: HRMapNet은 과거 예측된 벡터화된 맵에서 쉽게 생성된 역사적 래스터 맵을 사용하여 온라인 벡터화된 맵 인식을 보강합니다. 두 가지 모듈(특징 집계 모듈 및 쿼리 초기화 모듈)로 구성되어 있어, BEV(Top-Down Bird’s-Eye View) 특징과 맵 요소 쿼리를 향상시킵니다.

- **Performance Highlights**: HRMapNet은 두 개의 최첨단 방법(MapTRv2, StreamMapNet)과 통합되어 nuScenes 및 Argoverse 2 데이터셋에서 성능을 크게 향상시킵니다. 결과적으로 온라인 인식 성능이 개선되었으며, 자율주행 애플리케이션의 실용성을 위한 내구성과 잠재적인 응용을 보여주고 있습니다.



### Does Knowledge Localization Hold True? Surprising Differences Between Entity and Relation Perspectives in Language Models (https://arxiv.org/abs/2409.00617)
Comments:
          CIKM 2024

- **What's New**: 이번 연구에서는 대형 언어 모델(LLM) 내의 지식이 어떻게 저장되고 관리되는지를 조사하였으며, 개체(entity)와 관계(relational) 지식 간의 차이를 밝혀냈습니다. 이 연구는 지식 편집 모델을 통해 개체와 관계를 변형했을 때 그 결과가 일치하지 않음을 발견했습니다.

- **Technical Details**: 연구는 주로 모델 편집(model editing) 기법을 사용하여 개체와 관계 지식, 즉 지식의 쌍을 다루었습니다. 또한, 인과 분석(causal analysis)을 통해 사전 훈련된 모델에서 관계 지식이 어떻게 저장되는지를 조사했습니다. 실험 결과, 관계 지식은 MLP 가중치뿐만 아니라 Attention 모듈에서도 상당히 인코딩되어 있음이 밝혀졌습니다.

- **Performance Highlights**: 연구의 주요 결과는 기존 지식 평가 방법에 대한 의문을 제기하고, 모델 편집에 있어 새로운 기초를 마련했습니다. 개체와 관계 지식 간의 비대칭적인 저장 방식은 LLM의 지식 표현 및 편집 방법에 심도 있는 함의를 지닙니다.



### DAMe: Personalized Federated Social Event Detection with Dual Aggregation Mechanism (https://arxiv.org/abs/2409.00614)
Comments:
          CIKM 2024

- **What's New**: 이 논문은 사회적 이벤트 감지(Social Event Detection, SED)의 연합 학습(Federated Learning, FL) 접근 방식을 개선하기 위해 개인화된 연합 학습 프레임워크인 DAMe(Dual Aggregation Mechanism)를 제안합니다. 기존의 FL paradigms는 데이터의 이질성 문제를 효과적으로 처리하지 못합니다.

- **Technical Details**: DAMe는 두 가지 집계 메커니즘을 채택합니다: 로컬 집계(및 Bayesian optimization을 사용하여 최적의 집계 가중치를 탐색) 및 글로벌 집계(클라이언트 그래프를 최소화하여 각 클라이언트가 최대한의 외부 지식을 확보하도록 합니다). 글로벌-로컬 정렬(global-local alignment)은 로컬 모델이 글로벌 모델과 일치하도록 하는 제약 조건을 도입합니다.

- **Performance Highlights**: 실제 시뮬레이션을 통해 6개 언어와 2개 소셜 미디어 플랫폼에서 다양한 사회적 이벤트 데이터셋을 사용한 실험 결과, DAMe 프레임워크는 효과성과 회복력(robustness)을 입증했습니다. 또한, 이 프레임워크는 연합 공격으로부터 강한 저항성을 보여줍니다.



### Hyper-Compression: Model Compression via Hyperfunction (https://arxiv.org/abs/2409.00592)
- **What's New**: 논문에서는 대규모 모델의 메모리 요구 사항이 급증함에 따라 GPU 메모리의 한계를 해결하기 위해 'hyper-compression'이라는 새로운 모델 압축 접근 방식을 제안합니다. 이는 작은 매개변수 함수가 큰 네트워크의 가중치를 생성할 수 있도록 합니다.

- **Technical Details**: 제안된 hyper-compression 기법은 ergodic theory에 기반하여 저차원 동적 시스템이 고차원 공간을 채울 수 있는 가능성을 연구합니다. 이를 통해 모델 파라미터를 효율적으로 인코딩하는 hyperfunction 개념을 개발하고, LLaMA2-7B 모델을 1시간 이내에 압축하며, 성능 하락은 1% 미만으로 유지됩니다.

- **Performance Highlights**: 압축 비율은 평균 2.60배로 높으며, UNet 모델에서는 7.93배의 압축을 이루어냈고 그 성능 하락은 1%로 촉촉합니다. 이 방법은 기존의 훈련이 필요 없는 성격 덕분에 실제 적용의 유연성을 높이며, 다양한 다운스트림 작업에서 우수한 성능을 나타냅니다.



### FastBO: Fast HPO and NAS with Adaptive Fidelity Identification (https://arxiv.org/abs/2409.00584)
Comments:
          The 18th European Conference on Computer Vision ECCV 2024 Women in Computer Vision Workshop

- **What's New**: 이 논문에서는 Multi-Fidelity Bayesian Optimization을 위한 새로운 방법인 FastBO를 제안합니다. FastBO는 각 하이퍼파라미터 구성에 대해 적절한 Fidelity(신뢰성을) 동적으로 결정하여 성능을 극대화합니다.

- **Technical Details**: FastBO의 핵심 개념으로는 'efficient point'와 'saturation point'를 소개합니다. Efficient point는 자원(level)을 배가했을 때 성능 향상이 작은 임계값 이하로 떨어지는 지점을 의미하며, saturation point는 자원을 추가하면 성능 변화가 미미해지는 지점을 가리킵니다. 이러한 개념들은 자원의 효율적 사용과 성능 예측에 필요한 중요 요소입니다.

- **Performance Highlights**: FastBO는 각 구성의 적합한 Fidelity를 식별하여 surrogate model을 효과적으로 적용할 수 있게 하며, HPO와 NAS의 효율성 및 성능을 향상시키는 데 기여합니다.



### Enhancing Source Code Security with LLMs: Demystifying The Challenges and Generating Reliable Repairs (https://arxiv.org/abs/2409.00571)
- **What's New**: 최근 인공지능(AI) 컴퓨팅의 급격한 발전으로 인해 대형 언어 모델(LLMs)의 발전이 빠르게 진행되고 있으며, 특히 보안 분야에서 명확한 지침을 설정하는 데 도전이 되고 있습니다. 이 연구에서는 LLM 워크플로우 전반에 걸친 세 가지 주요 기술적 도전을 식별하고 설명합니다: 데이터 수집 및 라벨링, 시스템 설계 및 학습, 성능 평가. 이러한 도전을 바탕으로, 취약한 소스 코드를 신뢰성 있게 식별하고 설명하며 자동으로 수정할 수 있는 지침 기반 LLM 시스템인 SecRepair을 소개합니다.

- **Technical Details**: SecRepair은 강화 학습(Reinforcement Learning) 기반의 미세 조정(fine-tuning)을 사용하며, 생성된 코드의 기능성과 보안성을 충족하기 위한 의미론적 보상(semantic reward)을 적용합니다. 이 시스템은 데이터 준비 및 증강 기법, 최신 LLM 모델 선택 및 조정, 평가 절차에 대한 실행 가능한 가이드를 제공합니다.

- **Performance Highlights**: SecRepair은 강화 학습을 통해 훈련된 다른 LLM과 비교하여 보안 코드 수리에서 12% 개선되었음을 보여줍니다. 더욱이 실제 테스트 케이스를 사용하여 신뢰할 수 있고 기능적이며 컴파일 가능한 보안 코드 수리를 자동화된 평가 메트릭을 통해 생성할 수 있는 능력을 입증하였습니다.



### Using Deep Learning to Design High Aspect Ratio Fusion Devices (https://arxiv.org/abs/2409.00564)
- **What's New**: 이 연구는 직접 설계(deign) 문제를 풀어 최적화된 stellarator 구성(configuration)을 효율적으로 생성하기 위해 머신러닝 모델을 활용하는 방법을 제안합니다. 특히, Mixture Density Networks(MDNs)를 통한 확률적 접근이 강조됩니다.

- **Technical Details**: Stellarator는 자기 제어 자기 융합 장치로서, 비축 방향 비대칭(non-axisymmetric)의 토로이드(toroidal) 기하학을 가지고 있습니다. 이 연구에서는 near-axis expansion 방법을 두 번째 차수로 사용하여 유한 플라스마(β)를 포함하는 안정적인 구성(configuration)을 설계합니다. 이를 위해 neural network(신경망)를 사용하여 구성의 특성을 기반으로 하는 입력 매개변수를 찾습니다.

- **Performance Highlights**: 연구 결과는 최적화된 구성(configuration)을 신뢰성 있게 생성할 수 있음을 보여주었습니다. 이 방법은 기존의 계산 비용을 절감하며, quasisymmetry 및 MHD 안정성을 향상시키는 유리한 특성을 가진 구성(Configuration)을 제시합니다.



### Learning to Ask: When LLMs Meet Unclear Instruction (https://arxiv.org/abs/2409.00557)
- **What's New**: 본 연구는 실제 사용자 명령의 불완전성 하에서 대형 언어 모델(LLMs)이 도구를 사용하는 성능을 평가하기 위한 새로운 방법론을 제안하고, 새로운 벤치마크 Noisy ToolBench를 구축하였습니다. 또한, LLM들이 불분명한 지시를 마주쳤을 때 사용자의 명확한 답변을 요청하도록 유도하는 Ask-when-Needed (AwN) 프레임워크를 소개합니다.

- **Technical Details**: Noisy ToolBench는 명확하지 않은 사용자 명령의 모호함을 감지하고 관련 질문을 제기할 수 있는 LLM의 능력을 측정하기 위한 벤치마크입니다. 또한, ToolEvaluator라는 자동 평가 시스템을 설계하여 LLM의 도구 사용 성능을 효율적으로 평가합니다. 연구는 LLM의 훈련 목표와 명령어 실행 시 발생할 수 있는 주요 문제를 분석하고, 이에 대한 솔루션으로 AwN 메소드를 제안합니다.

- **Performance Highlights**: AwN 프레임워크는 NoisyToolBench에서 기존 방법보다 도구 학습 성능이 크게 향상되었습니다. 이 연구의 주요 기여는 적절한 명확화 질문을 요청하는 LLM의 능력, 올바른 함수 호출을 실행할 수 있는 능력, 그리고 사용자 요구를 충족하는 최종 응답을 제공할 수 있는 성공률을 기반으로 한 새로운 평가 지표를 개발한 것입니다.



### Multi-Output Distributional Fairness via Post-Processing (https://arxiv.org/abs/2409.00553)
Comments:
          17 pages, 4 figures

- **What's New**: 본 논문에서는 다중 출력 모델에 대한 후처리(post-processing) 방법을 제안하여 머신러닝 모델의 공정성을 높이는 방식에 관해 설명하고 있습니다. 기존의 후처리 방법들은 주로 단일 출력 모델에 맞춰져 있었으며, 다중 클래스 및 다중 태스크 구분의 경우는 충분히 탐구되지 않았습니다.

- **Technical Details**: 제안된 방법은 (optimal) transport mapping을 사용하여 모델의 출력을 서로 다른 그룹의 경험적 Wasserstein barycenter로 이동하는 방식으로 구성됩니다. 이 과정에서 복잡성을 줄이기 위한 근사 기법이 적용되며, 커널 회귀(kernel regression) 방법을 통해 샘플 외 데이터에 대한 확장성을 제공하고 있습니다.

- **Performance Highlights**: 제안된 방법은 다중 라벨 및 다중 클래스 분류, 표현학습 등의 과제에서 현재의 후처리 기준과 비교했을 때, 공정성을 높이는 데 효과적임을 실증적으로 입증하였습니다.



### Testing and Evaluation of Large Language Models: Correctness, Non-Toxicity, and Fairness (https://arxiv.org/abs/2409.00551)
Comments:
          PhD Thesis

- **What's New**: 이 논문은 대형 언어 모델(LLMs)인 ChatGPT의 신뢰성, 비독성(non-toxicity), 공정성(fairness)을 자동화 소프트웨어 테스트와 자연어 처리 관점에서 연구한 탐색적인 작업을 소개합니다. 특히 사실성(correctness) 평가를 위한 두 가지 테스트 프레임워크인 FactChecker 및 LogicAsker를 도입하였습니다.

- **Technical Details**: FactChecker는 대규모 지식 기반에서 사실 삼중항(triple)을 활용하여 사실 성능을 평가하고, LogicAsker는 기본 원리에 따라 자연어로 논리 추론 문제를 생성하는 테스트 프레임워크입니다. 비독성을 위해 MTTM이라는 텍스트 내용 감사 소프트웨어의 취약점을 정의하고, 언어의 다양성을 고려한 멀티링구얼 안전 기준 XSafety를 개발하였습니다. 공정성 평가를 위해 BiasAsker 및 XCulturalBench와 같은 새로운 평가 프레임워크를 소개합니다.

- **Performance Highlights**: 이 연구는 최신 LLMs(예: ChatGPT)의 사실 적절성과 논리 추론 능력을 향상시키는 데 기여하였으며, Multilingual 안전성 향상 및 사회적, 문화적 편견을 측정하는 효과적 방법론을 제시하여 AI 모델의 전반적인 신뢰성을 높이는 결과를 도출하였습니다.



### Data Augmentation for Image Classification using Generative AI (https://arxiv.org/abs/2409.00547)
Comments:
          19 pages, 15 figures, 4 tables

- **What's New**: 이 논문에서는 Automated Generative Data Augmentation(AGA) 프레임워크를 제안하여, 대규모 언어 모델(LLMs), 확산 모델(difusion models), 그리고 분할 모델(segmentation models)을 사용해 데이터 증대(data augmentation)를 자동으로 수행하는 방법을 소개합니다. AGA는 전경(foreground)의 진정성(authenticity)을 유지하면서 배경의 다양성을 보장합니다.

- **Technical Details**: AGA의 주요 요소는 다음과 같습니다: 1) 객체 추출을 위한 분할(segment) 및 슈퍼클래스(superclass) 기반 접근법, 2) 프롬프트 다양성을 위한 조합(combinatorial) 복잡성 및 프롬프트 분해(prompt decomposition) 사용, 3) 아핀 변형을 통한 주제(subject) 조작을 포함합니다. AGA는 이미지 세분화를 통해 주제를 분리하고, 사전 학습된 LLM을 사용해 다양한 배경 캡션을 생성합니다. 그 후 안정적 확산(Stable Diffusion)을 통해 다양한 배경을 생성하며 주제와 배경을 무결하게 통합합니다.

- **Performance Highlights**: AGA는 ImageNet, CUB, iWildCam의 세 개 대표 데이터셋에서 최신 기술(state-of-the-art)과 비교하여 실험 평가를 진행하였으며, 극세분류(fine-grained classification) 정확도가 기존 모델 대비 각각 15.6% 및 23.5% 향상되었고, SIC 점수(SIC score)는 64.3% 개선되었습니다.



### Large Language Models-Enabled Digital Twins for Precision Medicine in Rare Gynecological Tumors (https://arxiv.org/abs/2409.00544)
Comments:
          20 pages, 2 figures, 3 tables, supplements, original article

- **What's New**: 이번 연구에서는 Rare Gynecological Tumors (RGT) 환자를 위한 디지털 트윈 시스템을 구축하였으며, 대규모 언어 모델(LLMs)을 활용하여 개인 맞춤형 치료 전략을 제안하는 새로운 접근 방식을 선보였습니다.

- **Technical Details**: 디지털 트윈 시스템은 21개의 기관 사례와 655개의 문헌원을 통해 얻은 데이터(총 404,265명의 환자)를 통합하여, 전이성 자궁 육종 환자를 위한 맞춤형 치료 계획을 마련하는 데 초점을 맞추었습니다. LLM을 통해 전자 건강 기록(EHR)와 논문 데이터에서 정보 추출 및 구조화를 진행하여, RGT 디지털 트윈 시스템을 구축하였습니다.

- **Performance Highlights**: 이 연구의 결과로 얻어진 맞춤형 치료 옵션은 전통적인 단일 데이터 출처 분석에서는 발견되지 않은 추가적인 치료 가능성을 제시하였으며, 생물학 중심의 종양 정의로의 전환을 통해 RGT 관리를 개선하고 환자 결과를 향상시킬 수 있는 잠재력을 지니고 있습니다.



### Mapping earth mounds from spac (https://arxiv.org/abs/2409.00518)
Comments:
          6 pages, 4 figures, 3 tables

- **What's New**: 이번 논문에서는 기후 변화와 관련하여 스폿 랜드스케이프의 기원과 이를 식별하기 위한 자동화된 방법을 제안합니다. 특히, 딥러닝 프레임워크를 활용하여 원거리 센싱 원 데이터에서 지구 흙 더미를 자동으로 매핑하는 방법을 탐색합니다.

- **Technical Details**: 연구는 남미와 아프리카의 4가지 유형의 스폿 랜드스케이프를 분석하며, 기후 변화에 대한 적응성과 생태계의 생산성 증대와 관련된 지구 흙 더미의 패턴을 탐구합니다. 딥 러닝 모델을 사용하여 여러 지형과 지역적 특성을 고려하여 연구를 진행합니다. 생태계 엔지니어인 흰개미의 역할과 관련된 다양한 원리가 이들 환경에서 어떻게 작용하는지를 이해하는 것이 중요합니다.

- **Performance Highlights**: 우리는 다양한 최신 딥 네트워크를 평가하여 다양한 특성을 가진 스폿 랜드스케이프에서 성능을 비교합니다. 연구 결과, 자동 화 및 정확한 매핑이 이루어지기 위해서는 추가 연구가 필요함을 확인했습니다. 시스템의 복잡성이 높아 모든 지형에서 일관된 성능을 내기 위해서는 더욱 많은 데이터와 고급 모델링이 요구됩니다.



### Plant detection from ultra high resolution remote sensing images: A Semantic Segmentation approach based on fuzzy loss (https://arxiv.org/abs/2409.00513)
Comments:
          5 pages, 5 figures, 2 tables

- **What's New**: 이 연구에서는 초고해상도 (UHR) 원격 감지 이미지에서 식물 종을 식별하는 도전 과제를 해결하기 위한 접근 방식을 제시합니다. RGB 원격 감지 데이터셋을 도입하였고, 이 데이터셋은 프랑스의 산악 지역에서 여러 필드 원정에 걸쳐 세심하게 수집되었습니다.

- **Technical Details**: 연구에서는 의미론적 분할 문제로 식물 식별 과제를 정의하며, 이 과정에서 기존의 one-hot 인코딩된 실제 라벨 대신, 가우시안 필터를 이용하여 개선된 GT를 적용합니다. 이 모델은 각 픽셀의 클래스 소속 가능성을 모델링하고, 랜덤성을 도입하기 위해 새로운 손실 함수인 fuzzy loss를 제안합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법론의 유효성이 입증되었으며, 향후 개선의 필요성 또한 언급되었습니다. 본 연구는 실질적인 필드 관찰을 통해 획득한 초고해상도 RGB 이미지로 구성된 새로운 데이터셋을 선보이며, 기존 접근법 대비 효율성과 정확성을 올리는 중요한 기초 자료로 자리매김할 것으로 기대됩니다.



### Streamlining Forest Wildfire Surveillance: AI-Enhanced UAVs Utilizing the FLAME Aerial Video Dataset for Lightweight and Efficient Monitoring (https://arxiv.org/abs/2409.00510)
Comments:
          accpeted by Proceedings of the International Conference on Intelligent Robots and Systems (2024 IROS)

- **What's New**: 최근 UAV(무인 항공기)가 재난 긴급 대응 지원에서 중요한 역할을 수행한다는 점을 강조하며, 새로운 경량(weighted) 비디오 이해 정책 네트워크를 제안합니다. 이 연구는 향후 정보(station point)를 활용하여 정확성을 높이고, 컴퓨팅 자원을 절감하는 방법을 소개합니다.

- **Technical Details**: 제안하는 AccSampler 모델은 Adaptive Clip-aware Compression and Frame Sampling Network의 약자로, 비디오 클립을 압축하고 중복된 프레임을 제거하여 효율적인 비디오 처리(here, video understanding)를 가능하게 합니다. 이 과정에서 정책 네트워크가 프레임 중요도를 평가하고, 신뢰도 높은 프레임을 선택하여 데이터 세트를 정제(data distillation)합니다. 또한, 실험은 FLAME 데이터셋을 사용하였으며, 13배 이상의 계산 비용 절감과 3%의 정확성 향상을 이루어냈습니다.

- **Performance Highlights**: AccSampler는 Aerial Video Understanding 문제에서 우수한 성능을 보여주며, 경량화된 아키텍처 덕분에 전통적인 딥 러닝 모델에 비해 효율성과 정확성을 유지하면서도 훈련 과정에서의 시간을 대폭 절감할 수 있는 장점을 제공합니다.



### Geospatial foundation models for image analysis: evaluating and enhancing NASA-IBM Prithvi's domain adaptability (https://arxiv.org/abs/2409.00489)
- **What's New**: 이번 논문은 NASA-IBM의 GFM Prithvi를 평가하며, 고해상도 원거리 감지 이미지에 대한 오픈 소스 시각 기초 모델의 초기 사례로 주목받고 있습니다.

- **Technical Details**: Prithvi는 고급 이미지 분석 작업에서 예측 성능을 평가하여 다양한 벤치마크 데이터 세트에서 실험이 진행되었습니다. 새로운 전략으로는 band adaptation, multi-scale feature generation, fine-tuning techniques 등이 통합된 이미지 분석 파이프라인이 특징입니다.

- **Performance Highlights**: Prithvi의 성능 분석을 통해 장단점이 도출되었으며, 이는 Prithvi 개선뿐만 아니라 지리공간 작업을 위한 차세대 시각 기초 모델 개발에 유용한 통찰력을 제공합니다.



### Rapid Gyroscope Calibration: A Deep Learning Approach (https://arxiv.org/abs/2409.00488)
Comments:
          10 Pages, 14 Figures,

- **What's New**: 본 연구에서는 저비용 자이로스코프의 보정을 보다 신속하게 수행하기 위해 딥러닝(Deep Learning) 기술을 활용하는 새로운 접근법을 제안합니다. 특히, 하나의 자이로스코프의 보정을 개선하기 위해 다수의 실제 및 가상 자이로스코프 데이터를 활용하는 방법을 탐구하였습니다.

- **Technical Details**: 우리는 저비용 자이로스코프를 정적 조건에서 보정하는 데 집중하였으며, 자이로스코프 오차 모델을 사용하여 보정 프로세스를 설계했습니다. 24개 자이로스코프에서 수집한 169시간의 데이터셋을 기반으로 하여, 가상 자이로스코프 데이터를 포함한 딥러닝 프레임워크를 구성했습니다. 이 프레임워크는 보정 시간 단축을 목표로 하며, 기존의 모델 기반 방법론에 비해 89%까지 보정 시간을 줄일 수 있었습니다.

- **Performance Highlights**: 우리의 제안된 방법은 정확도와 보정 속도 모두에서 모델 기반 베이스라인 방법과 비교하여 개선된 결과를 보였고, 특히 시간에 민감한 응용 분야에서 유용성을 입증했습니다. 이 기술은 인명 구조 활동, 항공기 및 로봇 공학 등에서 신속한 보정이 절실한 환경에서 매우 중요한 역할을 할 것으로 기대됩니다.



### PSLF: A PID Controller-incorporated Second-order Latent Factor Analysis Model for Recommender System (https://arxiv.org/abs/2409.00448)
- **What's New**: 본 논문에서는 고차원 불완전 상호작용 데이터(HDI)를 위한 새로운 그래프 표현 학습 모델인 PID 제어기가 포함된 SLF(PSLF) 모델을 제안합니다. 이 모델은 손실 경관(curvature information) 정보를 활용하여 기존 SLF 모델의 낮은 수렴 속도 문제를 해결합니다.

- **Technical Details**: PSLF 모델은 두 가지 주요 전략을 통해 발전합니다: a) PID 제어기 원리를 통한 학습 오류 추정 개선, b) Hessian-벡터 곱(Hessian-vector products)을 통한 2차 정보 통찰력 확보. 이 결과로 비선형(non-convex) 문제를 효과적으로 해결할 수 있습니다.

- **Performance Highlights**: 여러 HDI 데이터셋에 대한 실험 결과, 제안된 PSLF 모델은 수렴 속도 및 일반화 성능 면에서 4개의 최신 잠재 인자(latent factor) 모델을 능가하는 것으로 나타났습니다.



### Breaking Down Financial News Impact: A Novel AI Approach with Geometric Hypergraphs (https://arxiv.org/abs/2409.00438)
Comments:
          16 pages, conference

- **What's New**: 최근 금융 시장에서의 주식 가격 예측의 중요성이 강조되고 있습니다. 본 논문에서는 Explainable Artificial Intelligence (XAI)를 활용하여 Geometric Hypergraph Attention Network (GHAN)라는 새로운 접근 방식을 제시합니다. 이 모델은 재무 뉴스가 시장 행동에 미치는 영향을 분석할 수 있습니다.

- **Technical Details**: GHAN은 전통적인 그래프 구조를 확장하는 기하적 하이퍼그래프 구조를 통해 다중 노드를 연결하여 고차원 관계를 효과적으로 모델링합니다. 또한 Attention 메커니즘을 포함하여 가장 관련성이 높은 정보를 집중적으로 학습하고, BERT 기반 임베딩을 통해 재무 뉴스 텍스트의 의미적 풍부성을 캡쳐합니다. 하이퍼그래프에서 작동하는 Attention 메커니즘과 SHAP(Shapley Additive Explanations) 값을 통합하여 모델의 투명성을 보장하고 주요 영향을 미치는 요소를 강조합니다.

- **Performance Highlights**: 실험 결과, GHAN 모델은 전통적인 감성 분석 및 시계열 모델보다 우수한 성능을 나타냈으며, 복잡하고 빠르게 변화하는 주식 거래 환경에서 투자자에게 유용한 도구로 자리잡을 것으로 기대됩니다.



### Robust off-policy Reinforcement Learning via Soft Constrained Adversary (https://arxiv.org/abs/2409.00418)
Comments:
          33 pages, 12 figures, 2 tables

- **What's New**: 이번 연구는 f-divergence를 활용하여 강력한 적대적 공격에 대한 강화 학습의 새로운 접근 방식을 제안합니다. 이는 기존의 L_p-norm 제약을 넘어 실제 환경에서의 분포적 정보를 고려하여 강인한 강화 학습 알고리즘을 개발하는 것을 목표로 하고 있습니다.

- **Technical Details**: 연구는 Soft Worst-Case Attack (SofA)과 Epsilon Worst-Case Attack (EpsA)이라는 두 가지 전형적인 공격 및 해당 공격에 대한 강인한 학습 프레임워크를 제안합니다. 또한, f-divergence 제약을 이용한 최적의 적대자 검색 문제로서의 새로운 관점을 도입합니다.

- **Performance Highlights**: 제안된 methods는 샘플 효율성이 높은 off-policy 강화 학습에서 탁월한 성과를 달성하며, 강화 학습의 기존 제한 사항을 극복할 수 있는 가능성을 보여줍니다.



### Density Adaptive Attention-based Speech Network: Enhancing Feature Understanding for Mental Health Disorders (https://arxiv.org/abs/2409.00391)
- **What's New**: 본 논문은 DAAMAudioCNNLSTM 및 DAAMAudioTransformer라는 두 가지 파라미터 효율적이고 설명 가능한 모델을 소개하여, 음성 신호를 기반으로 한 우울증 탐지에서의 도전 과제를 해결하고 있습니다. 이는 기존 접근 방식과 달리 추가적인 정보 없이도 높은 성능을 달성하였습니다.

- **Technical Details**: DAAMAudioCNNLSTM 모델은 CNN-LSTM 구조와 다중 헤드 밀도 적응 주의 메커니즘(DAAM)을 활용하여 음성 데이터의 중요 부분에 집중합니다. DAAM은 가우시안 분포를 통해 가장 정보가 많은 음성 데이터 부분을 동적으로 타겟팅해서 모델의 특징 우선순위를 정하고 있습니다. DAAMAudioTransformer는 CNN-LSTM 구조 대신 Transformer 인코더를 사용하여 같은 DAAM 모듈을 통합하여 더 나은 해석 가능성을 제공합니다.

- **Performance Highlights**: DAAMAudioCNNLSTM은 DAIC-WOZ 데이터셋에서 0.702의 F1 매크로 점수를 달성하였고, DAAMAudioTransformer는 0.72의 F1 매크로 점수를 기록하여 기존의 최첨단 결과를 초월하였습니다. 두 모델 모두 실시간 분석에 적합하고, 기존의 데이터 의존성을 줄이며 보다 신뢰할 수 있는 임상 진단 도구로서의 가능성을 보여줍니다.



### Predicting the Target Word of Game-playing Conversations using a Low-Rank Dialect Adapter for Decoder Models (https://arxiv.org/abs/2409.00358)
Comments:
          6 pages, 3 Figures, 5 Tables

- **What's New**: 이 논문은 디코더 모델에 대한 방언(adapters)을 도입하는 LoRDD(저랭크 방언 강인성 모델) 아키텍처를 제안하여 NLU(Natural Language Understanding) 작업에서 특정 방언의 성능을 향상시킵니다.

- **Technical Details**: LoRDD는 두 가지 LoRA 기반(adapter) 어댭터를 결합하여 작업 어댑터와 방언 어댑터를 구현합니다. 작업 어댑터는 지침 미세 조정을 사용하고, 방언 어댑터는 Pseudo-parallel 대화 코퍼스에서 대조 학습을 수행합니다. 이 연구는 MD-3 데이터셋을 사용하여 en-IN과 en-US 간의 관계를 학습합니다.

- **Performance Highlights**: LoRDD는 Task Word Prediction(TWP) 작업에서 Mistral 및 Gemma 모델에 대해 넷 베이스라인보다 우수한 성능을 보였으며, en-US에 비해 단어 유사성에서 12%, 정확도에서 25%의 성능 격차를 줄였습니다.



### Contrastive Augmentation: An Unsupervised Learning Approach for Keyword Spotting in Speech Technology (https://arxiv.org/abs/2409.00356)
Comments:
          This paper has been accepted by the ICPR2024

- **What's New**: 이 논문은 Keyword Spotting (KWS) 분야에서 레이블이 있는 데이터의 부족 문제를 해결하기 위해 비지도 대조 학습(unsupervised contrastive learning)과 독특한 증강 방법(augmentation-based technique)을 결합한 새로운 접근 방식을 제시합니다.

- **Technical Details**: 제안된 방법은 비지도 학습의 가능성을 활용하여 레이블이 없는 데이터 세트에서 신경망을 훈련할 수 있도록 하며, 이는 제한된 레이블 데이터 세트가 있는 다운스트림 작업의 성능을 향상시킬 수 있습니다. 또한, 같은 키워드를 포함한 음성 발화는 속도나 볼륨의 변화에도 불구하고 유사한 고수준 특성 표현을 가져야 한다고 주장합니다. 이를 달성하기 위해, 음성 증강 기반의 비지도 학습 방법을 제안합니다.

- **Performance Highlights**: 이 방법은 Google Speech Commands V2 데이터 세트에서 강력한 성능을 보였으며, KWS의 강력한 성능을 위한 압축된 합성곱 아키텍처(compressed convolutional architecture)를 제안했습니다. 또한, 비지도 손실(unsupervised loss) 및 대조 손실(contrastive loss)을 개발하여 원본 및 증강된 음성 간의 유사성을 평가합니다.



### GSpect: Spectral Filtering for Cross-Scale Graph Classification (https://arxiv.org/abs/2409.00338)
- **What's New**: 본 연구에서는 크로스 스케일 그래프 분류 작업을 위한 고급 스펙트럴 그래프 필터링 모델인 GSpect를 제안합니다. 기존의 그래프 분류 방법의 한계를 극복하기 위해 그래프 웨이블릿 신경망을 사용하여 다중 스케일 메시지를 집계하고, 스펙트럴 풀링 레이어를 설계하여 크로스 스케일 그래프를 동일한 크기로 축소합니다.

- **Technical Details**: GSpect는 그래프 웨이블릿 이론을 활용하여 그래프 분류 작업에 적용하고, 그래프의 인접 행렬 및 노드 속성에 대해 푸리에 변환을 수행하여 주파수 도메인 표현을 얻습니다. 주파수 정보 필터링을 통해 고주파 정보를 재조정하고, 불필요한 메시지를 필터링히는 저차 그래프를 생성합니다.

- **Performance Highlights**: GSpect는 공개 데이터 세트에서 평균 1.62%의 분류 정확도가 개선되었고, PROTEINS 데이터 세트에서는 최대 3.33% 향상을 보였습니다. MSG 데이터 세트에서는 평균 15.55%의 분류 정확도 개선이 나타났습니다.



### Evaluating the Effectiveness of Large Language Models in Representing and Understanding Movement Trajectories (https://arxiv.org/abs/2409.00335)
Comments:
          7 pages, 3 figures

- **What's New**: 이번 연구는 AI 기초 모델의 움직임 궤적(trajectory) 표현 능력을 평가하는 데 초점을 두고 있습니다. 대규모 언어 모델인 GPT-J를 활용하여 궤적의 문자열 형식을 인코딩하고, 이 LLM 기반 표현의 효과를 궤적 데이터 분석에서 평가합니다.

- **Technical Details**: 본 연구는 GPS 로거를 사용하여 수집한 GeoLife 데이터셋을 기반으로 궤적 거리 측정과 위치 예측 작업을 포함하는 두 가지 주요 작업을 설정합니다. LLM인 GPT-J는 궤적의 고차원 표현을 학습하는 인코더 역할을 하며, 효과적인 프롬프트 및 미세 조정 기법을 통해 다양한 작업을 수행할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과는 GPT-J 임베딩에서 파생된 코사인 거리와 하우스도르프(Hausdorff)와 동적 시간 왜곡(DTW) 거리 간의 상관 계수가 0.74를 초과하는 것으로 나타났습니다. 또한, LLM은 궤적 분석에서 위치 예측 작업에 대한 좋은 정확도를 보였습니다.



### WikiCausal: Corpus and Evaluation Framework for Causal Knowledge Graph Construction (https://arxiv.org/abs/2409.00331)
Comments:
          Extended version; poster paper accepted at ISWC 2024

- **What's New**: 최근 일반 도메인 및 도메인 특정 인과 지식 그래프(conceptual knowledge graphs) 구축에 대한 관심이 높아지고 있습니다. 이 지식 그래프는 인과 분석(causal analysis) 및 이벤트 예측(event prediction)을 위한 추론을 가능하게 하며, 여러 분야에 걸쳐 다양한 응용을 제공합니다. 이번 논문에서는 인과 지식 그래프 자동 구축을 위한 데이터셋(corpus), 작업(task), 및 평가 프레임워크(evaluation framework)를 제안합니다.

- **Technical Details**: 제안된 데이터셋은 이벤트 관련 개념이 포함된 위키백과 문서로 구성됩니다. 이 작업의 목표는 코퍼스에서 이벤트 개념 간의 인과 관계(causal relations)를 추출하는 것입니다. 이러한 인과 관계의 품질을 평가하기 위해, 기존의 Wikidata의 인과 관계를 사용하여 회수를 평가하고, Large Language Models(LLMs)를 활용하여 수작업 평가 없이 정밀도를 측정합니다. 평가 프레임워크는 공개적으로 제공되며, 이는 텍스트 코퍼스를 입력으로 받아 인과 관계의 지식 그래프를 생성하는 자동화된 솔루션의 품질을 평가하는 데 최초로 사용됩니다.

- **Performance Highlights**: 모듈식 인과 지식 추출 파이프라인을 사용하여 다양한 조합의 사전 훈련된 신경 모델을 이용해 네 가지 버전의 Wikidata 기반 인과 지식 그래프를 생성하고, 모델 선택이 출력 품질에 미치는 영향을 보여주었습니다. 이 프레임워크를 통해 각 작업에 적절한 모델을 효과적으로 찾을 수 있음을 입증했습니다.



### Demo: FedCampus: A Real-world Privacy-preserving Mobile Application for Smart Campus via Federated Learning & Analytics (https://arxiv.org/abs/2409.00327)
Comments:
          2 pages, 3 figures, accepted for publication in ACM Mobihoc 2024

- **What's New**: FedCampus는 분산 학습(federated learning, FL)과 분산 분석(federated analytics, FA)을 지원하는 개인 정보 보호 모바일 애플리케이션으로, 스마트 캠퍼스를 위한 솔루션으로 소개됩니다. iOS와 Android 모두에서 작동하며, 지속적으로 모델과 알고리즘을 배포할 수 있는 MLOps를 지원합니다.

- **Technical Details**: FedCampus는 사용자의 스마트 기기에서 사생활 보호를 위한 FL과 FA를 구현합니다. 모델 변환, 통합된 학습 API를 통해 Android (TFLite) 및 iOS (Core ML)에서 호환 가능한 형식으로 모델을 변환하고, 코드 구현 시 GPU 및 NPU 가속을 활용하였습니다. 또한, 다중 세션 FL 서버를 지원하여 모델 배포 및 학습을 효율적으로 관리합니다.

- **Performance Highlights**: Duke Kunshan University에서 100명의 자원봉사자를 통해 FedCampus가 성공적으로 구현되었으며, 수면 추적, 신체 활동 모니터링, 맞춤형 추천 및 주요 활동 분석을 포함한 다양한 스마트 캠퍼스 작업을 수행하였습니다. 이 프로젝트는 오픈 소스로 제공되며, 실제 환경에서의 적용이 성공적으로 이루어졌습니다.



### Toward a More Complete OMR Solution (https://arxiv.org/abs/2409.00316)
- **What's New**: 본 연구에서는 음표 인식의 새로운 접근 방식을 제시하고 있습니다. 특히, 기존의 완벽한 객체 탐지(output)를 가정하지 않고, 불완전한 객체 탐지를 기반으로 음표 조합(notation assembly) 방식을 개선하는 방법에 중점을 두었습니다.

- **Technical Details**: MUSCIMA++ v2.0 데이터셋을 사용하여 음표를 그래프 형태로 표현하였고, YOLOv8을 기반으로 한 음악 객체 탐지기를 도입하였습니다. 또한, 검출(output)에 기반하여 음표 조합 단계를 완수하는 감독(training) 방법을 제안합니다.

- **Performance Highlights**: 이 모델은 기존의 완벽한 탐지 결과를 훈련에 사용했던 모델보다 우수한 성능을 나타내어, 탐지와 조합 단계를 통합하여 고려하는 것의 유용성을 보여주었습니다. 새로운 평가 지표인 Match+AUC를 통해 탐지 오류 및 조합 오류를 동시에 고려하여 더 완전한 OMR 솔루션을 향해 나아가는 중요한 단계로 평가됩니다.



### An Empirical Study on Context Length for Open-Domain Dialog Generation (https://arxiv.org/abs/2409.00315)
Comments:
          6 pages, 2 figures, 2 tables

- **What's New**: 이번 연구에서는 Transformer 기반 오픈 도메인 대화 모델의 문맥 길이에 대한 설정을 다루고 있습니다. 문맥의 적절한 길이를 결정하는 기준이 없으며, 이로 인해 모델 성능에 미치는 영향을 실험하였습니다.

- **Technical Details**: 연구는 다양한 문맥 길이에서의 모델 학습 결과를 바탕으로, 문맥 길이가 길수록 모델 훈련에 도움이 되는지, 다른 문맥 길이의 대화에 따라 훈련 문맥 길이를 변경해야 하는지, 그리고 다양한 대화 샘플이 동일한 문맥 길이에 대한 선호도를 가지는지를 파악하는 세 가지 질문을 설정하였습니다.

- **Performance Highlights**: 실험 결과, Transformer 기반 대화 모델에 있어 문맥 길이는 성능과 효율성을 모두 고려할 때 무조건 길다고 해서 좋은 것은 아니며, 가장 성능이 높은 모델이 다양한 역사 길이를 가진 대화에서 잘 수행되므로, 별도의 모델을 훈련할 필요가 없다는 점이 발견되었습니다. 또한, 각 샘플에 대해 특정 문맥 길이를 고려하는 것이 모델 성능을 더욱 향상시킬 수 있다는 결과가 나왔습니다.



### Objective Features Extracted from Motor Activity Time Series for Food Addiction Analysis Using Machine Learning (https://arxiv.org/abs/2409.00310)
Comments:
          16 pages, 3 figures, 14 tables

- **What's New**: 이 연구는 식품 중독(FA) 진단 및 증상(SC) 평가를 위한 객관적인 특성을 식별하기 위해 머신러닝 알고리즘을 조사했습니다.

- **Technical Details**: 81명의 참가자(평균 나이: 21.5세)를 대상으로 Yale Food Addiction Scale(YFAS)을 사용하여 FA와 SC를 측정했습니다. 참가자들은 인구통계학적 데이터, YFAS, Zung Self-Rating Depression Scale, Dutch Eating Behavior Questionnaire를 완료하고, 비지배 손목에 액티미터(actimeter)를 착용하여 일주일 동안 운동 활동을 기록했습니다. 액티미터 데이터 분석 결과, FA와 SC를 정확하게 예측하는 중요한 통계적 및 엔트로피 기반 특징이 발견되었습니다. 매튜스 상관계수(MCC)를 주요 지표로 사용했습니다.

- **Performance Highlights**: 활동 관련 특징들은 FA 예측에서 더 효과적이었으며(MCC=0.88), 휴식 관련 특징의 경우 MCC는 0.68에 불과했습니다. SC의 경우, 활동 세그먼트는 MCC=0.47, 휴식 세그먼트는 MCC=0.38을 기록하였고, 이들의 조합은 MCC=0.51을 보였습니다. 액티미터 특징이 FA, 정서적 및 제한된 식습관과의 значение관계를 보여주며 모델의 유효성을 지지했습니다.



### Quantum Machine Learning for Anomaly Detection in Consumer Electronics (https://arxiv.org/abs/2409.00294)
Comments:
          7 pages, 2 figures, 1 table, under ISVLSI 2024 proceedings

- **What's New**: 이 논문은 소비자 전자기기에서 이상 탐지를 위한 양자 기계 학습(Quantum Machine Learning, QML) 알고리즘의 응용을 소개하고 있습니다. 특히, QML의 일반적인 프레임워크를 제시하며, 최근 연구 사례를 통해 실제 사례를 논의합니다.

- **Technical Details**: 양자 기계 학습은 데이터의 차원 축소(Dimensionality Reduction), 양자 회로(QML Circuit)를 통한 데이터 처리, 그리고 훈련된 모델을 사용하여 새로운 데이터를 예측하는 세 가지 주요 단계를 포함합니다. 또한, variational quantum circuits 및 kernel-based circuits와 같은 다양한 QML 회로 모델을 다루고 있습니다.

- **Performance Highlights**: 연구 결과, QML을 사용한 이상 탐지 시스템은 기존의 머신 러닝 모델보다 효율적이며, 신속하게 새로운 이상에 적응할 수 있는 잠재력을 갖추고 있습니다. 다섯 가지 사례 연구를 통해 QML의 실제 응용 가능성을 확인했습니다.



### OnlySportsLM: Optimizing Sports-Domain Language Models with SOTA Performance under Billion Parameters (https://arxiv.org/abs/2409.00286)
Comments:
          13 pages, 4 figures, 4 tables

- **What's New**: 이 논문은 스포츠 관련 데이터로만 훈련된 작은 도메인 특화 언어 모델의 가능성을 탐구합니다. OnlySports라는 새로운 데이터 세트 및 벤치마크를 소개하며, 이를 통해 쿼리의 효율성을 극대화하는 방법을 제시합니다.

- **Technical Details**: 우리는 6000억개의 토큰을 포함하는 OnlySports Dataset을 FineWeb에서 추출하였으며, RWKV 아키텍처를 스포츠 관련 태스크에 최적화하여 196M 파라미터(매개변수)를 가진 모델을 설계했습니다. 모델 구조는 20층, 640차원입니다.

- **Performance Highlights**: OnlySportsLM은 이전의 135M 및 360M 모델 대비 각각 37.62% 및 34.08%의 정확도 개선을 달성했으며, SomlLM 1.7B 및 Qwen 1.5B과 같은 대형 모델과의 성능이 동등합니다.



### Reframing Data Value for Large Language Models Through the Lens of Plausability (https://arxiv.org/abs/2409.00284)
- **What's New**: 본 논문에서는 데이터 가치 평가(data valuation)의 중요한 질문인 '이 데이터는 얼마나 가치가 있는가?'에 대한 새로운 관점을 제안합니다. 기존의 방법들은 주로 discriminative 모델에 집중되어 있었으나, 우리는 generative 모델의 가능성을 기반으로 데이터의 유용성뿐만 아니라 plausibility(그럴듯성)에 초점을 맞추고 있습니다.

- **Technical Details**: 우리는 데이터의 가치를 통계적 방식으로 정량화하는 새로운 방법론을 제안합니다. 이 방법론은 data가 모델에 의해 plausibly 생성될 수 있는지를 분석하여 데이터 가치 평가를 다루며, Rosenblatt의 변환을 통해 continuous random variables(연속 확률 변수)로 변환하여 유사성 테스트를 수행합니다. 데이터가 모델로부터 생성되었다면, 데이터가 힘들거나 불가능하게 생성될수록 그 데이터의 가치가 높다고 가정합니다.

- **Performance Highlights**: 이 논문에서 제안하는 Uniform-Marginal and Independence (UMI) value function은 높은 성능을 보여줍니다. 데이터의 분포와 모델 간의 통계적 불일치를 정량적으로 분석함으로써, 우리는 데이터 세트의 가치를 효율적으로 평가할 수 있는 방법을 개발했습니다.



### The Artificial Intelligence Act: critical overview (https://arxiv.org/abs/2409.00264)
- **What's New**: 이번 연구는 최근 통과된 인공지능 법령(Artificial Intelligence Act)에 대한 비판적 개요를 제공합니다. 주로 Regulation (EU) 2024/1689의 주요 구조, 목표 및 접근 방식에 대해 논의합니다.

- **Technical Details**: 법령은 명시적으로 원칙을 설정하지 않지만, 공정성(fairness), 책임(accountability), 투명성(transparency), 평등성(equity) 등의 핵심 개념을 중심으로 규칙을 형성합니다. 금지된 인공지능 관행에는 조작(manipulation), 취약성(vulnerabilities)의 착취(exploitation), 사회적 점수(social scoring), 생체 인식(biometric identification) 및 예측 경찰(predictive policing) 등이 포함됩니다.

- **Performance Highlights**: 법령은 고위험(high-risk) 인공지능 시스템의 규제 및 투명성 의무를 논의하며, 일반 목적 모델에 대한 규제, 인증(certification), 감독(supervision) 및 제재(sanctions) 규칙을 포함합니다. 전체 프레임워크는 적절하고 균형 잡힌 것으로 평가되지만, 복잡한 접근 방식이 유럽 연합 내에서 책임 있는 혁신(promoting responsible innovation)을 촉진하려는 목적을 저해할 위험이 있습니다.



### MAPWise: Evaluating Vision-Language Models for Advanced Map Queries (https://arxiv.org/abs/2409.00255)
Comments:
          30 Pages, 46 Tables, 6 Figure

- **What's New**: 이번 연구는 Vision-Language Models (VLMs)의 choropleth maps에 대한 질문 응답 능력을 분석합니다. 새로운 데이터셋인 MAPWise를 소개하며, 미국, 인도, 중국을 포함해 1,000개의 질문을 제공합니다.

- **Technical Details**: MAPWise 데이터셋은 다양한 난이도의 질문 템플릿을 포함하며, 지리적 및 공간적 이해를 평가하기 위하여 43개의 질문 유형이 마련되었습니다. VLMs 모델을 통해 실험하였으며, Zero-Shot 및 Explicit Extraction and Reasoning 방식으로 평가하였습니다.

- **Performance Highlights**: VLMs의 성능 평가에서 쟁점과 한계를 밝혀냈으며, 새로운 벤치마크와 데이터셋을 통해 향후 연구 방향을 제시했습니다. 특히, 새롭게 도입한 MAPWise 데이터셋은 choropleth maps와 관련하여 모델의 성능을 비교하는 데 유용합니다.



### One-Frame Calibration with Siamese Network in Facial Action Unit Recognition (https://arxiv.org/abs/2409.00240)
- **What's New**: 이 논문에서는 얼굴 표정 분석에서 자주 사용되는 자동 얼굴 액션 유닛(AU) 인식을 위해, 각 얼굴에 대해 중립 표정의 이미지를 보정(reference image)으로 사용하는 단일 프레임 보정(one-frame calibration, OFC) 기법을 제안합니다.

- **Technical Details**: 제안하는 보정 방법은 Calibrating Siamese Network (CSN) 아키텍처를 기반으로 하며, 간단한 iResNet-50 (IR50) 백본을 사용해 AU 인식을 수행합니다. 이 네트워크는 두 개의 동일한 네트워크에 중립 이미지와 목표 이미지를 입력하고, 중간 단계에서 피처 맵의 차이를 계산하여 결합합니다.

- **Performance Highlights**: DISFA, DISFA+, UNBC-McMaster 데이터셋에서 OFC CSN-IR50 모델이 기존 IR50 모델의 성능을 크게 개선하고, naively OFC 방법 및 최첨단 NCG 모델들보다 AU 강도 추정 및 AU 탐지에서 뛰어난 성능을 보였습니다.



### Deep learning surrogate models of JULES-INFERNO for wildfire prediction on a global sca (https://arxiv.org/abs/2409.00237)
- **What's New**: 이 논문은 JULES-INFERNO 모델을 대체할 수 있는 두 가지 데이터 기반 모델을 제안하여 전 세계 산불 예측의 속도를 높이는 방법을 제시합니다. 이 모델들은 기후 변화에 대한 적응력을 갖춘 도구로 활용될 수 있습니다.

- **Technical Details**: 제안된 모델은 Convolutional Auto-encoder (CAE)와 Long Short-Term Memory (LSTM) 기반의 CAE-LSTM 및 Convolutional LSTM (ConvLSTM) 모델입니다. 이들은 기후 데이터(온도, 식생 밀도, 토양 수분)를 기반으로 반복적으로 훼손된 면적을 예측합니다. Average Error per Pixel (AEP) 및 Structural Similarity Index Measure (SSIM)이 성능 평가에 사용됩니다.

- **Performance Highlights**: 모델은 30년 예측 시 예상 평균 오류(AEP)가 0.3% 이하, SSIM이 98% 이상으로, JULES-INFERNO의 속도는 약 5시간 걸리는 반면, 제안된 모델은 약 10초로 계산 효율성을 확보했습니다.



### Spatially-Aware Diffusion Models with Cross-Attention for Global Field Reconstruction with Sparse Observations (https://arxiv.org/abs/2409.00230)
- **What's New**: Diffusion 모델의 발전을 통해 제한된 관측으로부터 완전한 공간 필드를 추정하는 새로운 방법을 제안합니다. 개선된 condition encoding 접근법을 도입하여 관찰 영역과 비관찰 영역 간의 효율적인 매핑을 구성합니다.

- **Technical Details**: 이 연구는 score-based diffusion 모델을 활용한 분야 재구성을 다루며, 학습 가능한 sparse observations 및 보간된 필드를 통합하여 inductive bias를 형성합니다. 또한, 조건화 방법에 따라 guided sampling, classifier-free guidance (CFG), 및 cross-attention의 효과를 비교 분석합니다.

- **Performance Highlights**: 제안된 diffusion 모델은 noisy 조건에서 cross-attention을 적용할 경우 다른 방법들보다 뛰어난 성능을 보이며, deterministic 방법보다 정밀하고 효율적인 결과를 제공합니다. 특히, steady 문제에서는 두 방법 모두 수치적 접근법보다 높은 정확도를 나타냅니다.



### A Generative Adversarial Network-based Method for LiDAR-Assisted Radar Image Enhancemen (https://arxiv.org/abs/2409.00196)
- **What's New**: 본 논문에서는 저해상도 레이더 이미지를 향상시키기 위한 GAN(Generative Adversarial Network) 기반 접근 방법을 제안합니다. 이 연구는 자율주행차량(AVs)의 물체 인식을 개선하기 위해 반사하는 세부사항과 특징을 더 잘 표현하는 것을 목표로 합니다.

- **Technical Details**: 제안된 방법은 고해상도 2D 투영 라이다(LiDAR) 포인트 클라우드를 그라운드 트루스 이미지로 사용하고, 레이더 이미지를 입력으로 활용하여 GAN을 훈련합니다. 이 방법의 추론 과정은 오직 레이더 이미지만을 사용하여 향상된 이미지를 생성합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 저해상도 레이더 이미지에 비해 물체 표현이 더 명확한 향상된 이미지를 생성할 수 있음을 보여주었습니다. 이는 특히 악천후에서도 효과적입니다.



### Deep Neural Networks for Predicting Recurrence and Survival in Patients with Esophageal Cancer After Surgery (https://arxiv.org/abs/2409.00163)
Comments:
          14 pages, 3 figures, 4 tables. To appear in CaPTion: MICCAI Workshop on Cancer Prevention, detection, and intervenTion, Sharib Ali et al., MICCAI 2024, Lecture Notes in Computer Science, Springer

- **What's New**: 이 연구에서는 식도암 환자의 질병 무진행 생존 (Disease-Free Survival, DFS) 및 전체 생존 (Overall Survival, OS) 예측을 위한 Cox 비례 위험 모델(Cox Proportional Hazards, CoxPH)과 두 가지 심층 신경망 모델(DeepSurv 및 DeepHit)을 비교하였습니다. ENURE 연구에서 수집된 다기관 국제 데이터셋을 이용하여 중요한 예후 인자를 식별하고 예측 정확성을 향상시키기 위해 심층 신경망을 활용했습니다.

- **Technical Details**: 연구는 ENSURE 연구에서 수집된 4972명의 환자와 170개 이상의 변수를 포함한 대규모 데이터셋을 기반으로 하였습니다. CoxPH 모델을 통해 각 특성이 결과에 미치는 영향을 평가한 후, DeepSurv와 DeepHit를 사용하여 DFS 및 OS를 예측했습니다. 여러 변수를 추출하고 성능을 비교하는 실험을 수행했습니다.

- **Performance Highlights**: DeepSurv와 DeepHit 모델은 CoxPH 모델과 유사한 구분 정확도를 보였으며, DeepSurv은 DFS 및 OS 예측 과제에서 약간의 우위를 보여 C-인덱스(C-index)가 각각 0.735 및 0.74에 도달했습니다. 이로 인해 심층 신경망이 예후 예측 도구로서 개인화된 치료 접근 방식을 제공할 가능성이 나타났습니다.



### Sequence to Sequence Reward Modeling: Improving RLHF by Language Feedback (https://arxiv.org/abs/2409.00162)
Comments:
          7 pages

- **What's New**: 이 논문에서는 대형 언어 모델 (LLMs)의 행동을 인간의 의도와 가치에 맞추기 위한 새로운 접근 방식인 	extit{sequence-to-sequence (seq2seq) reward modeling} 방법을 제안합니다. 기존의 보상 모델링 방식에서 이진 최대 우도 추정 (MLE)을 시퀀스 MLE로 대체하여 언어 피드백을 학습함으로써 RLHF의 정확성과 세분성을 향상시킵니다.

- **Technical Details**: 제안된 seq2seq 보상 모델링은 두 가지 주요 단계, 즉 보상 모델링과 보상 추출로 구성됩니다. 이 방법은 각 토큰이 반응 점수에 미치는 영향을 직접적으로 반영하며, 각각의 토큰에 대한 긍정 및 부정 피드백을 추출하여 RLHF의 세분성을 개선합니다. 추가적인 데이터 주석, 훈련, 모델 없이 이루어지며, 이는 데이터의 효율성을 높이고 과도한 모델링 복잡성을 줄입니다.

- **Performance Highlights**: 실험 결과, seq2seq RM을 적용했을 때 2B 및 7B 파라미터의 LLM에서 3개의 NLP 작업에 대해 평균 76.9%의 개선 효과를 보였습니다. 또한, seq2seq RM은 분포 밖의 프롬프트에서도 RLHF 성능을 개선할 수 있음을 보여주어, 제안된 방법의 유연성과 효과성을 입증하였습니다.



### Learning-Based Finite Element Methods Modeling for Complex Mechanical Systems (https://arxiv.org/abs/2409.00160)
- **What's New**: 본 연구에서는 복잡한 기계 시스템 시뮬레이션을 효과적으로 학습하기 위해 새로운 두 단계 메시 그래프 네트워크를 제안합니다. 이 네트워크는 Graph Block(GBK)과 Attention Block(ABK)을 결합하여 장거리 공간 종속성이 있는 기계 상호작용을 학습할 수 있도록 설계되었습니다.

- **Technical Details**: 제안된 두 단계 메시 그래프 네트워크는 Encoder-Processor-Decoder 패러다임을 따릅니다. 미세 메시 노드 그래프 수준에서는 GBK를 사용하여 로컬 표현을 학습하고, 더 큰 메시 노드에서는 ABK를 사용하여 글로벌 표현을 학습합니다. 프로세서 모듈은 각 GBK 후에 ABK가 포함된 M𝑀 층의 시퀀스를 사용하여 로컬 및 글로벌 표현을 함께 학습합니다.

- **Performance Highlights**: 세 가지 합성 데이터셋과 하나의 실제 데이터셋에 대한 평가 결과, 제안된 방법은 54.3% 더 낮은 예측 오차와 9.87% 적은 학습 가능한 네트워크 매개변수를 보여줘 기존 최첨단 기술에 비해 우수성을 입증합니다.



### LLMs hallucinate graphs too: a structural perspectiv (https://arxiv.org/abs/2409.00159)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 허위 정보 생성, 즉 'hallucination'을 구조화된 형태로 연구할 수 있는 가능성을 제시합니다. 특히, 저자들은 특정 문헌에서 알려진 그래프를 요청했을 때의 허위 응답을 분석하여, 이러한 허위 그래프가 LLM의 출력 특성을 묘사하는 데 어떻게 활용될 수 있는지를 탐구합니다.

- **Technical Details**: 연구의 주요 기여는 두 가지입니다. 첫째, 여러 최신 LLM에서의 토폴로지적 hallucination의 다양성을 관찰합니다. 둘째, 이러한 허위 정보의 크기를 측정할 수 있는 지표인 Graph Atlas Distance를 제안합니다. 이 지표는 여러 그래프에서의 평균 그래프 편집 거리로 정의됩니다. 또한, Hallucination Leaderboard와 비교하여 10,000배 더 많은 프롬프트를 활용한 랭킹을 제공합니다.

- **Performance Highlights**: 연구에서는 미스트랄, Vercel AI SDK, HuggingChat, ChatGPT, together.ai, Google의 Gemini 등 21개의 LLM을 대상으로 비교 분석을 진행했습니다. Zachary의 카라테 클럽 그래프와 Les Misérables 그래프 등에서 허위 응답을 평가하였으며, 각 LLM의 출력 그래프의 품질을 6가지 주요 통계로 검토했습니다.



### Developing an End-to-End Framework for Predicting the Social Communication Severity Scores of Children with Autism Spectrum Disorder (https://arxiv.org/abs/2409.00158)
Comments:
          Accepted for Interspeech 2024

- **What's New**: 이번 연구는 ASD 아동의 사회적 커뮤니케이션 심각도를 자동으로 예측하기 위한 최신 E2E(End-to-End) 프레임워크를 제안합니다. 이 프레임워크는 자동 음성 인식(Automatic Speech Recognition, ASR) 모델과 미세 조정된 사전 훈련 언어 모델(Pre-trained Language Models, PLMs)을 통합하여 ABC 아동의 speech 데이터에서 심각도 예측 점수를 도출합니다.

- **Technical Details**: 프레임워크는 wav2vec2-xls-r-300m과 whisper-large-v2 두 가지 다국어 ASR 모델을 사용하며, ASD 아동의 speech 데이터로 미세 조정됩니다. 이어서 KR-BERT, KLUE/roberta-base, KR-ELECTRA-Discriminator와 같은 세 가지 PLM을 미세 조정하여 예측 점수를 생성합니다. 이 과정에서 전통적인 미세 조정, 수동 프롬프트, p-tuning 접근 방식이 적용됩니다.

- **Performance Highlights**: 평균적으로, 이 시스템은 인간 평가자 점수와의 Pearson 상관 계수 0.6566을 달성하여 기존 진단 도구와 비교할 수 있는 가능성을 보여줍니다. 이 연구는 ASD 진단을 위한 접근 가능하고 객관적인 도구 개발에 기여할 것으로 기대됩니다.



### Speaker Tagging Correction With Non-Autoregressive Language Models (https://arxiv.org/abs/2409.00151)
Comments:
          6 pages, 7 tables

- **What's New**: 이 논문은 자동 음성 인식(ASR) 시스템과 스피커 다이얼리제이션(SD) 시스템의 출력을 결합하여 대화 중 누가 언제 발언했는지를 파악하는 작업의 중요성을 강조합니다. 특히, 스피커 구분이 필요한 자연 대화의 품질을 높이기 위해 비자기적 언어 모델을 기반으로 한 이중 단계 스피커 태깅 수정 시스템을 제안합니다.

- **Technical Details**: 제안된 시스템은 두 개의 데이터세트인 TAL과 Fisher 데이터 세트에서 단어 다이얼리제이션 오류율(WDER)을 감소시키는 성과를 보였으며, 잘못된 발언 경계를 수정하는 데 중점을 두었습니다. 이 시스템은 음성 세그멘테이션과 스피커 태깅의 오류를 분석하고 분류하는 과정을 포함하고 있습니다.

- **Performance Highlights**: 실험에서 제안된 스피커 오류 수정 모듈은 Fisher 테스트 세트와 TAL 데이터 세트에서 기존 방법에 비해 cpWER에서 유의미한 개선을 보여주었습니다, 이는 스피커 다이얼리제이션의 성능을 크게 향상시키는 데 기여합니다.



### From Semantics to Hierarchy: A Hybrid Euclidean-Tangent-Hyperbolic Space Model for Temporal Knowledge Graph Reasoning (https://arxiv.org/abs/2409.00149)
- **What's New**: 본 연구에서는 유클리드(Euclidean) 모델과 쌍곡선(hyperbolic) 모델의 장점을 결합한 새로운 혼합 기하학적 공간 접근 방식을 제안합니다. 이 접근 방식은 단일 공간에서 다중 공간으로의 매개변수 모델링을 전환하여 의미론(semantic) 및 계층적 정보(hierarchical information)를 효과적으로 포착합니다.

- **Technical Details**: 먼저 복잡한 의미 정보를 Euclidean 공간에서 사실 동시 발생 및 자기 회귀 방법을 통해 캡처한 후, 스케일링 메커니즘을 이용하여 Tangent 공간으로 변환합니다. 이후 쿼리-후보 분리 모델링 접근 방식을 통해 계층 구조를 재학습하고, 마지막으로 하이브리드 유도 편향(hybrid inductive bias)을 통해 하이퍼볼릭(hyperbolic)과 유클리드 점수 함수(Euclidean scoring functions)를 결합합니다.

- **Performance Highlights**: 실험 결과 YAGO 데이터셋에서 평균 역rank(MRR) 기준으로 기존 단일 공간 모델에 비해 최대 15.0%의 상대적 오차 감소를 보였습니다. 시각화 분석을 통해 다양한 의미적 및 계층적 복잡성을 갖춘 데이터셋에 대한 적응 능력이 확인되었습니다.



### MultiMath: Bridging Visual and Mathematical Reasoning for Large Language Models (https://arxiv.org/abs/2409.00147)
- **What's New**: 이 논문은 시각적인 입력과 통합된 수학적 추론의 필요성을 강조하며, 이를 위해 다중 모달 대규모 언어 모델인 MultiMath-7B를 제안합니다. 본 모델은 MultiMath-300K라는 새로운 다중 모달 수학 데이터셋에 기반하여 다양한 수학적 문제가 포함되어 있습니다.

- **Technical Details**: MultiMath-7B는 4단계 훈련 과정을 통해 개발되며, 시각-언어 정렬 (vision-language alignment), 시각 및 수학 지침 조정 (instruction-tuning)과 프로세스 감독 강화 학습 (process-supervised reinforcement learning)에 중점을 둡니다. 이는 DeepSeekMathRL-7B를 기반으로 하여 시각 인코더와 다중 모달 어댑터를 추가한 모델입니다.

- **Performance Highlights**: MultiMath-7B는 기존의 다중 모달 수학 벤치마크에서 SOTA(State-of-the-Art) 성능을 달성하며, 텍스트 전용 수학 벤치마크에서도 뛰어난 성능을 보여줍니다. 특히, 다중 모달 훈련이 특정 텍스트 전용 수학 문제에 대한 성능 향상을 가져온 결과를 보였습니다.



### Robust Temporal-Invariant Learning in Multimodal Disentanglemen (https://arxiv.org/abs/2409.00143)
Comments:
          5 pages, 2 figures, this is the first version. The code is available at this https URL

- **What's New**: 이번 연구에서는 다중 모달 감정 인식에서 연속적인 시계열에서 발생하는 프레임 수준의 중복성을 제거하기 위해 Temporal-invariant learning 방식을 제안합니다. 이를 통해 보다 부드러운 시계열 패턴을 효과적으로 잡아내어 표현의 질을 향상시키고 모델의 강인성을 높였습니다.

- **Technical Details**: 제안된 모델은 RTIL(Robust Temporal-Invariant Learning)로, 서로 다른 모달리티의 표현을 분리하기 위해 adversarial learning을 활용합니다. 이를 통해 modality-invariant representation과 modality-specific representation을 분리하여 적응형 융합 메커니즘을 제공합니다. 또한, 영상 및 음성 모달리티에는 Transformer Encoders를 사용하고, 텍스트 모달리티에는 RoBERTa를 통해 자연어 표현을 향상시킵니다.

- **Performance Highlights**: 실험 결과, RTIL 모델은 두 개의 공개 데이터셋에서 기존 최첨단 방법들을 초월하는 성능을 발휘했습니다. 이는 모델이 시간의 변동성에도 불구하고 일관된 글로벌 정보를 유지할 수 있다는 것을 증명합니다.



### Dynamic Depth Decoding: Faster Speculative Decoding for LLMs (https://arxiv.org/abs/2409.00142)
- **What's New**: 본 논문에서는 Dynamic Depth Decoding (DDD)를 소개하며, 이는 EAGLE-2의 디코딩 알고리즘을 최적화하여 현재의 최첨단 speculative decoding 방법의 속도를 향상시킵니다.

- **Technical Details**: DDD는 EAGLE-2의 트리 초안 생성 방법을 동적인 깊이로 최적화합니다. 이는 EAGLE-2가 EAGLE에 대해 달성하는 평균 속도 향상을 44% 연장하여 DDD는 평균 3.16배의 속도 향상을 제공합니다. DDD는 드래프트 모델의 신뢰도를 토대로 드래프트 생성을 계속할지 결정하기 위해 확률 합계를 휴리스틱으로 사용합니다.

- **Performance Highlights**: 실험 결과, DDD는 EAGLE-2에 비해 평균 4% 향상을 보여주며, EAGLE-2는 EAGLE에 비해 평균 8% 향상된 성능을 제시합니다. 모든 실험에서 정확도는 측정되지 않았고, 모든 속도 향상 방법은 손실이 없었습니다.



### Statistical Analysis of the Impact of Quaternion Components in Convolutional Neural Networks (https://arxiv.org/abs/2409.00140)
Comments:
          17 pages, 6 figures

- **What's New**: 최근 몇 년 간 사원수(Quaternion)를 이용한 합성곱 신경망(Convolutional Neural Networks)인 QCNN이 다양한 문제에 대해 제안되었습니다. 이 논문에서는 이미지 분류 문제에 대해 QCNN 구성요소들 간의 성능 비교를 위한 통계적 분석을 수행하였으며, 새로운 Fully Quaternion ReLU 활성화 함수를 소개하고 있습니다.

- **Technical Details**: QCNN의 구성 요소로는 활성화 함수(activation function), 완전 연결층(fully connected layer), 초기화 알고리즘(initalization algorithm), 모델의 파라미터 수 등이 포함됩니다. 이 연구에서는 n-way ANOVA 테스트를 통해 이러한 요소들이 분류 정확도(classification accuracy)에 미치는 상호작용 효과를 측정하고 최적의 조합을 찾았습니다.

- **Performance Highlights**: 새롭게 제안된 Fully Quaternion ReLU 활성화 함수는 기존의 Split Quaternion ReLU 함수보다 성능을 향상시켰습니다. 실험 결과, QCNN이 기존의 실수 기반 모델과 비슷하거나 더 나은 결과를 보여주었음을 확인하였습니다.



### PrivacyLens: Evaluating Privacy Norm Awareness of Language Models in Action (https://arxiv.org/abs/2409.00138)
Comments:
          Under review

- **What's New**: 이 논문은 언어 모델(LMs)의 프라이버시 인식 수준을 정량화하고, LM이 매개된 커뮤니케이션에서 발생하는 프라이버시 위험을 평가하기 위한 새로운 프레임워크인 PrivacyLens를 제안합니다.

- **Technical Details**: PrivacyLens는 프라이버시 민감한 상황을 표현하는 비네트를 통해 에이전트의 행동에서 프라이버시 유출을 다단계로 평가할 수 있도록 설계되어 있습니다. 이 프레임워크는 프라이버시 문헌에 기반한 프라이버시 규범의 집합과 크라우드소싱된 씨드를 활용해 구체화되었습니다.

- **Performance Highlights**: 최신 LM인 GPT-4와 Llama-3-70B는 프라이버시를 강화하는 지침에도 불구하고 각각 25.68%와 38.69%의 경우에 민감한 정보를 유출하였습니다. 이는 LM의 성능과 실제 사용자 지시를 수행할 때의 행동 간 간극을 드러냅니다.



### Emerging Vulnerabilities in Frontier Models: Multi-Turn Jailbreak Attacks (https://arxiv.org/abs/2409.00137)
- **What's New**: 본 연구에서는 단일 및 다중 턴 입력 형식 모두에서 사용할 수 있는 jailbreak 공격의 데이터셋을 소개합니다. 각 형식이 작성 내용은 동일하지만 성공률이 다르고, 모델의 방어 메커니즘이 입력 구조에 따라 다르게 작동함을 보여줍니다. 또한 최첨단 모델의 취약성은 단일 및 다중 턴 환경 모두에서 심층적으로 연구되어야 함을 강조합니다.

- **Technical Details**: 이 연구는 단일 턴 공격과 다중 턴 공격 간의 차이를 정량화하고, 두 공격 형식의 성공적인 공격률 차이가 최대 50%까지 있다는 것을 보여줍니다. 다양한 데이터셋을 사용하여 공격을 분석하며, 사이퍼 기법을 이용한 공격의 효과도 검토합니다. 데이터셋은 해로운, 완전 무해한, 반 무해한 프롬프트로 구성되어 있으며, 각각의 특성을 바탕으로 실험을 진행합니다.

- **Performance Highlights**: 상위 모델(OpenAI, Anthropic, Meta)에 적용된 공격에서 여전히 안전성 문제들이 드러났습니다. 또한 방어 메커니즘과 공격자의 전략 사이의 복잡한 상호작용으로 인해 방어 조치를 마련하기 어려운 상황임을 확인하였습니다.



### HoneyComb: A Flexible LLM-Based Agent System for Materials Scienc (https://arxiv.org/abs/2409.00135)
Comments:
          Under Review on EMNLP 2024

- **What's New**: HoneyComb은 재료 과학을 위해 특별히 설계된 최초의 LLM(large language model) 기반 에이전트 시스템으로, 기존의 LLM이 직면한 문제들을 해결하기 위해 새로운 지식 기반과 도구 허브를 활용합니다.

- **Technical Details**: HoneyComb은 MatSciKB라는 고품질의 재료 과학 지식 기반과 Inductive Tool Construction 방법을 통해 생성된 ToolHub를 통합하여 재료 과학 관련 과제의 정확성과 효율성을 향상시킵니다. 또한, Retriever 모듈을 활용하여 특정 작업에 적합한 지식 출처나 도구를 적응적으로 선택합니다.

- **Performance Highlights**: HoneyComb은 다양한 재료 과학 과제에서 기존 모델보다 현저히 뛰어난 성과를 보이며, LLM의 일반적 능력과 재료 과학의 전문적 요구 사이의 격차를 효과적으로 좁히고 있습니다.



### MAPF-GPT: Imitation Learning for Multi-Agent Pathfinding at Sca (https://arxiv.org/abs/2409.00134)
- **What's New**: 이 논문에서는 다중 에이전트 경로 찾기 문제(Multi-agent pathfinding, MAPF)를 해결하기 위한 새로운 학습 기반 솔버를 제안합니다. 특히 'MAPF-GPT'라는 이름의 기초 모델을 만들어, 부분 관찰 상태에서 추가적인 휴리스틱이나 보상 함수, 에이전트 간의 통신 없이 경로를 제시할 수 있도록 훈련했습니다.

- **Technical Details**: MAPF 문제 해결을 위해, 먼저 관찰(observation)과 행동(action)을 설명하는 데 사용할 수 있는 토큰(token)이라는 용어의 어휘를 설계했습니다. 그런 다음, 성공적인 서브 최적의 MAPF 솔루션으로 구성된 대규모 및 다양한 전문가 데이터셋을 수집하고 이를 (관찰, 행동) 튜플의 시퀀스로 변환한 후, 트랜스포머 기반의 비자율 신경망을 사용하여 관찰에 기반하여 올바른 행동을 예측하도록 학습했습니다.

- **Performance Highlights**: MAPF-GPT 모델은 훈련 데이터셋에 없었던 MAPF 문제 인스턴스를 해결할 때 제로샷 학습(zero-shot learning) 능력을 보여줍니다. 이 모델은 현재의 최상급 학습 기반 MAPF 솔버와 비교했을 때 다양한 문제 인스턴스에서 뛰어난 성능을 발휘하며, 특히 분포 외 평가(out-of-distribution evaluation)에서 우수한 성능을 보였습니다.



### A Survey for Large Language Models in Biomedicin (https://arxiv.org/abs/2409.00133)
- **What's New**: 현대의 거대 언어 모델(LLMs)이 이루어진 최근의 발전은 전례 없는 자연어 이해 및 생성 능력을 제공하고 있습니다. 본 리뷰는 생명 의학 분야의 LLMs에 대한 종합적인 분석을 제공하며, 특정 응용 프로그램이나 모델 아키텍처에 국한되는 기존 조사와는 차별화됩니다.

- **Technical Details**: 이 리뷰는 484개의 출처를 분석하여 LLM의 현재 상태, 응용 프로그램, 도전 과제 및 향후 전망을 탐구합니다. zero-shot learning 능력을 포함하여 진단 보조, 약물 발견 및 개인 맞춤형 의학과 같은 다양한 생명 의학 작업의 사례를 분석합니다. 또한, uni-modal 및 multi-modal LLM의 fine-tuning 방법과 데이터 보호 및 윤리적 과제를 포함한 생명 의학 분야의 도전 과제를 다룹니다.

- **Performance Highlights**: MedPaLM은 92.9%의 일치를 기록하였으며, scBERT는 단일 세포 유전체 데이터 분석을 향상시키기 위한 임베딩을 생성합니다. LLMs의 적용이 빠르게 증가하고 있으며, 2021년부터 출판물이 급증하고 있습니다. LLM의 신뢰성과 특정성 향상을 위한 특별한 훈련 및 최적화가 필요합니다.



### Logic Contrastive Reasoning with Lightweight Large Language Model for Math Word Problems (https://arxiv.org/abs/2409.00131)
- **What's New**: 이번 연구에서는 경량 대형 언어 모델(Large Language Models, LLMs)의 수학적 추론 과제에서 성능을 향상시키기 위한 새로운 방법을 소개합니다.

- **Technical Details**: 수학적 논리 유사성(mathematical logic similarity)을 측정하는 새로운 방법을 도입하고, 의미적(semantic) 및 논리적(logical) 유사성을 통합한 참조 문제(reference problems) 집합을 구성하기 위한 자동 스크리닝 메커니즘을 설계했습니다. 긍정 및 부정 예시 프롬프트(prompts)를 활용하여 모델이 건전한 추론 로직(sound reasoning logic)을 채택하도록 유도합니다.

- **Performance Highlights**: SVAMP 데이터셋에서 Chain of Thought 접근법 대비 15.8% 향상, GSM8K 데이터셋에서는 21.5% 향상을 기록했습니다. 175억 매개변수(parameter)를 가진 대규모 모델에 이 방법을 적용하면 두 데이터셋에서 최고의 결과와 비교되는 성능을 나타냈습니다. 또한, 추론 과정 중 발생하는 오류에 대한 분석을 통해 미래 연구에 중요한 통찰을 제공했습니다.



### Mirror contrastive loss based sliding window transformer for subject-independent motor imagery based EEG signal recognition (https://arxiv.org/abs/2409.00130)
Comments:
          This paper has been accepted by the Fourth International Workshop on Human Brain and Artificial Intelligence, joint workshop of the 33rd International Joint Conference on Artificial Intelligence, Jeju Island, South Korea, from August 3rd to August 9th, 2024

- **What's New**: 본 연구에서는 motor imagery (운동 상상) 기반의 EEG (뇌파) 신호 인식에서 기존의 딥러닝 모델이 '블랙 박스(black box)'로 작동하는 문제를 해결하고자 합니다. 왼손 또는 오른손의 운동 상상이 뇌의 반대편 운동 감각 영역에서 event-related desynchronization (ERD)을 유도한다는 신경학적 발견에 착안하여 Mirror Contrastive Loss 기반 Sliding Window Transformer (MCL-SWT)를 제안하였습니다.

- **Technical Details**: MCL-SWT에서는 original EEG 신호와 mirror EEG 신호 간의 대조를 통해 ERD의 공간적 위치에 대한 민감도를 향상시키는 mirror contrastive loss를 활용합니다. Mirror EEG 신호는 EEG 신호의 왼쪽과 오른쪽 반구의 채널을 교환하여 생성됩니다. 또한, 고시간 해상도(feature) 데이터에서 self-attention score를 계산하는 temporal sliding window transformer을 도입하여 모델 성능을 개선하고 계산 복잡성을 관리할 수 있습니다.

- **Performance Highlights**: MCL-SWT의 성능은 subject-independent motor imagery EEG 신호 인식 작업에서 평가되었으며, 66.48%와 75.62%의 정확도를 달성하여 최첨단(SOTA) 모델을 각각 2.82%와 2.17% 초과하였습니다. 추가적으로, ablation 실험을 통해 제안된 mirror contrastive loss의 효과성을 검증하였습니다.



### Can AI Replace Human Subjects? A Large-Scale Replication of Psychological Experiments with LLMs (https://arxiv.org/abs/2409.00128)
Comments:
          5 figures, 2 tables

- **What's New**: 이번 연구에서는 최신 Large Language Model인 GPT-4가 사회 과학 분야의 154개의 심리 실험을 복제하여 인간의 반응을 얼마나 잘 모사할 수 있는지를 평가하였습니다.

- **Technical Details**: 연구는 618개의 주요 효과(main effects)와 138개의 상호작용 효과(interaction effects)를 대상으로 하였으며, GPT-4가 이 실험들에서 76.0%의 주요 효과와 47.0%의 상호작용 효과를 효과적으로 재현하였음을 발견하였습니다.

- **Performance Highlights**: 하지만, GPT-4가 재현한 신뢰 구간(confidence intervals) 중 19.44%만이 원본 효과 크기와 일치하였고, 71.6%에서 원래 연구에서 보고된 null findings와 반대되는 유의미한 결과가 나타났습니다. 이는 LLM의 연구 도구로서의 가능성을 보여주지만, AI 기반의 결과 해석 시 주의가 필요함을 강조합니다.



### Latent-EnSF: A Latent Ensemble Score Filter for High-Dimensional Data Assimilation with Sparse Observation Data (https://arxiv.org/abs/2409.00127)
Comments:
          13 pages, 10 figures, 1 table

- **What's New**: 본 논문에서는 Latent-EnSF라는 새로운 데이터 동화(data assimilation) 방법을 제안합니다. 이 방법은 EnSF(Ensemble Score Filters)를 활용하여 비선형 베이지안 필터링(nonlinear Bayesian filtering)의 고차원성과 관측 데이터의 희소성(sparsity) 문제를 해결하는 데 중점을 두고 있습니다.

- **Technical Details**: Latent-EnSF는 커플드 변분 오토인코더(coupled Variational Autoencoder, VAE)를 사용하여 전체 상태와 희소한 관측을 일관된 방식으로 인코딩하고, 최종적으로는 상태 재구성을 통해 데이터 동화를 시행합니다. 이 과정에서 EnSF의 고차원 비선형 시스템에 대한 샘플링을 효율적으로 수행합니다.

- **Performance Highlights**: Latent-EnSF는 복잡한 모델에 대한 두 가지 도전적 응용 사례에서 기존 방법들과 비교할 때 정확도가 높고, 수렴 속도가 빠르며, 보다 높은 효율성을 보여주었습니다. 이 방법은 얕은 물결 전파(shallow water wave propagation)와 중거리 기상 예측(medium-range weather forecasting) 문제에 적용되었으며, 공간과 시간 모두에서 매우 희소한 관측 조건에서도 잘 작동함을 입증하였습니다.



### A Hybrid Framework for Spatial Interpolation: Merging Data-driven with Domain Knowledg (https://arxiv.org/abs/2409.00125)
Comments:
          21 pages, 13 figures; typos corrected, references updated

- **What's New**: 이 논문에서는 흩어진 관측 데이터셋을 통해 공간적으로 분포된 정보를 추정하는 새로운 하이브리드 프레임워크를 제안합니다. 이는 데이터 기반의 공간 의존성(feature extraction)과 규칙 보조 공간 의존성 함수(mapping)를 통합하여 도메인 지식(domain knowledge)을 증강하는 데 중점을 두고 있습니다.

- **Technical Details**: 제안된 프레임워크는 관측 데이터와 규칙 보조 지식을 결합하여 공간 정보 추정을 혁신적인 방법으로 접근합니다. 이 프레임워크는 거리 기반의 비선형 추정(nonlinear estimation)을 강화하기 위해 변환된 퍼지 규칙(transformed fuzzy rules)을 적용하며, 관측 데이터셋과 관련된 내재적 불확실성(uncertainties)을 정량화하는 기능을 포함합니다.

- **Performance Highlights**: 비교 응용 시나리오에서 우리의 프레임워크는 더 국소화된 공간적 특징을 포착하는 뛰어난 성능을 보여주며, 공간 분포 필드(reconstructed distribution fields)의 품질을 크게 향상시키는 것에 성공하였습니다.



### Leveraging Large Language Models for Wireless Symbol Detection via In-Context Learning (https://arxiv.org/abs/2409.00124)
Comments:
          Accepted at IEEE GLOBECOM 2024

- **What's New**: 이 논문은 적은 데이터로 무선 시스템의 문제를 해결하기 위해 대형 언어 모델(LLM)의 인컨텍스트 학습(in-context learning) 능력을 활용하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 무선 기호 변조(symbol demodulation) 작업에 대해 DNN(Deep Neural Network)보다 LLM을 이용한 접근이 더 효과적임을 증명하며, 특히 다양한 프롬프트 템플릿을 사용할 때 LLM의 성능은 크게 달라질 수 있는 점을 강조합니다. 데이터가 제한된 상황에서도 LLM은 학습 없이 추론만으로 활용될 수 있습니다.

- **Performance Highlights**: 실험 결과, ICL(인컨텍스트 학습) 방법을 사용할 경우 전통적인 DNN보다 평균 22%의 성능 향상을 보여주었으며, LLM의 교정(calibration) 기법을 적용하면 높은 신뢰도를 가진 예측 결과를 얻을 수 있음을 시사합니다.



### Brant-X: A Unified Physiological Signal Alignment Framework (https://arxiv.org/abs/2409.00122)
Comments:
          Accepted by SIGKDD 2024

- **What's New**: 이 논문은 EEG (electroencephalogram)와 다른 생리 신호 간의 상관관계를 모델링하기 위한 통합 생리 신호 정렬 프레임워크인 Brant-X를 제안합니다. 기존 연구에서는 주로 단일 생리 신호에 집중하였으나, 이 연구는 신호 간 상호 연관성을 활용하여 더 좋은 성능을 확보하고자 합니다.

- **Technical Details**: Brant-X는 EEG 기반 모델을 사용하여 다른 생리 신호로 지식을 효율적으로 전달하며, 두 수준의 정렬을 도입하여 EEG와 다른 신호의 의미를 완전히 정렬합니다. 각 신호의 샘플링 속도 차이를 극복하기 위해 샘플링 증강(sampling augmentation)을 도입하였습니다.

- **Performance Highlights**: Brant-X는 수면 단계 분류(sleep stage classification), 감정 인식(emotion recognition), 동결 보행 탐지(freezing of gait detection), 안구 운동 통신(eye movement communication) 등 다양한 다운스트림 작업에서 최첨단 성능을 기록하였습니다. 또한, 심장 부정맥 탐지(arrhythmia detection) 작업에 대한 분석에서 Brant-X의 EEG에서 다른 생리 신호로의 지식 이전의 효과성을 입증하였습니다.



### BELT-2: Bootstrapping EEG-to-Language representation alignment for multi-task brain decoding (https://arxiv.org/abs/2409.00121)
- **What's New**: BELT-2는 EEG(뇌전도) 신호로부터 자연어 Decoding을 위한 최초의 멀티태스크 모델로, BPE(byte-pair encoding) 레벨 EEG-언어 정렬 및 멀티태스크 훈련을 통합하여 혁신을 이끌어냈습니다.

- **Technical Details**: BELT-2는 Q-Conformer라는 새로운 EEG 인코더를 도입하여 멀티태스크 처리가 가능하고, BPE 수준의 대조 학습(BPE-CL)을 통해 EEG와 언어 표현 간의 강력한 정렬을 구축합니다.

- **Performance Highlights**: BELT-2는 ZuCo 데이터셋에서 52.2% BLEU-1 점수를 기록하며, 다른 번역 벤치마크에서도 31%에서 162%의 성능 향상을 보였습니다. EEG 감정 분석에서 74.62%의 정확도를 달성하였으며, 최초로 EEG 요약 작업에서도 SOTA 31.17 BLEU-1 점수를 기록했습니다.



### ConCSE: Unified Contrastive Learning and Augmentation for Code-Switched Embeddings (https://arxiv.org/abs/2409.00120)
Comments:
          ICPR 2024

- **What's New**: 이 논문은 영어와 한국어 간의 Code-Switching (CS) 현상을 연구하며, CS 데이터셋의 필요성을 강조합니다. 특히, 기존의 Equivalence Constraint (EC) 이론이 영어-Korean CS 복잡성을 부분적으로만 설명한다고 주장합니다. 저자들은 이를 해결하기 위해 Koglish 데이터셋을 새롭게 제안합니다.

- **Technical Details**: Koglish 데이터셋은 Koglish-GLUE, Koglish-NLI, Koglish-STS를 포함하며, 영어와 한국어 간의 CS 상황을 다룹니다. 연구는 SimCSE 모델을 기반으로 한 제안된 ConCSE를 통해 CS 문장을 모델링하고, 세 가지 새로운 손실 함수를 도입하여 문장 표현 학습을 진행합니다.

- **Performance Highlights**: 실험 결과, ConCSE 방법은 Koglish-STS 데이터셋에서 SimCSE 대비 평균 1.77% 성능 향상을 보여주며, Koglish 데이터셋의 효과성을 검증했습니다. 여러 NLP 작업을 대상으로 한 비교 실험에서도 Koglish 데이터셋이 다른 다국어 모델에 비해 CS 환경에서의 성능 향상을 이루는 데 기여함을 보여줍니다.



### 3-in-1: 2D Rotary Adaptation for Efficient Finetuning, Efficient Batching and Composability (https://arxiv.org/abs/2409.00119)
Comments:
          24 pages, 6 figures, 13 tables

- **What's New**: 이 논문은 RoAd라는 새로운 기법을 소개하며, 이 기법은 LLMs(대형 언어 모델)을 효율적으로 조정하는 데 필요한 훈련 가능한 매개변수를 최소화합니다. 특히, 여러 작업 또는 사용자에 맞춘 어댑터 필요성이 있는 경우, RoAd 방법은 요청에 따라 효율적인 처리를 가능하게 하고, 해석 가능성도 강화합니다.

- **Technical Details**: RoAd는 2D 회전 기법을 활용하여 LLMs를 조정하며, 이를 통해 다음과 같은 세 가지 주요 문제를 해결합니다: (1) $<0.1\%$의 훈련 가능한 매개변수로 GLUE 벤치마크 및 여러 추론 작업에서 최적의 성능을 제공합니다; (2) 서로 다른 어댑터를 필요로 하는 동시에 요청을 배치 처리하는 데 있어 요소별 곱(element-wise multiplication)을 사용함으로써 계산 부담을 줄입니다; (3) 분산 교환 개입(distributed interchange intervention) 프레임워크 내에서 통합하여 LLM의 해석 가능성을 향상시킵니다.

- **Performance Highlights**: RoAd는 GLUE, 상식 추론 과제 및 산수 과제에서 타 PEFT 방법을 초월하는 성능을 보여주며, LoRA보다 두 배 높은 처리량을 자랑합니다. 또한, RoAd는 다양한 작업에 대해 학습된 가중치를 결합할 수 있는 능력을 보여주어 새로운 기능을 표현할 수 있습니다.



### Quantum Kernel Principal Components Analysis for Compact Readout of Chemiresistive Sensor Arrays (https://arxiv.org/abs/2409.00115)
- **What's New**: 이번 연구에서는 기존의 cPCA (classical principal component analysis) 대신 qPCA (quantum principal component analysis)를 사용하여 IoT 기기에서 발생하는 대량의 데이터를 효율적으로 압축할 수 있는 방법을 제시합니다.

- **Technical Details**: CSAs (chemiresistive sensor arrays)는 IoT 시스템의 중요한 구성 요소로, 다수의 센서가 동시에 작동함에 따라 대량의 데이터를 생성합니다. cPCA는 데이터 압축을 위한 전통적인 방법이지만, 차원 축소 시 중요한 정보를 보존하는 데 한계가 있습니다. 반면 qPCA는 이 정보를 더 효과적으로 유지할 수 있습니다.

- **Performance Highlights**: 연구 결과, qPCA는 다양한 머신 러닝 모델링 작업에서 cPCA보다 더 나은 성능을 보였습니다. 특히 제한된 qubit (양자 비트)에 접근할 수 있는 저차원 시나리오에서 탁월한 결과를 나타내며, NISQ (noisy intermediate-scale quantum) 컴퓨터의 가능성을 보여줍니다.



### When All Options Are Wrong: Evaluating Large Language Model Robustness with Incorrect Multiple-Choice Options (https://arxiv.org/abs/2409.00113)
- **What's New**: 이번 연구는 대형 언어 모델(LLM)들이 정답이 없는 객관식 질문을 탐지하는 제로샷(zero-shot) 능력에 대해 탐구합니다. 이는 교육 평가의 질 향상 측면에서 중요한 요소입니다.

- **Technical Details**: 연구에서는 다양한 LLM 및 질문 세트를 활용하여 실험을 수행하였으며, Llama-3.1-405B 모델이 유효한 정답이 없음을 성공적으로 식별하는 데 두드러진 성과를 보였습니다. 모델들은 temperature 집합을 0으로 설정하고 최대 토큰(max_tokens)을 128로 제한하여 명확한 응답을 유도했습니다.

- **Performance Highlights**: 결과적으로, 정답이 있는 질문과 반대로 정답이 없는 질문 간 성능 차이가 상당했으며, LLM들이 교육 환경에서 잘못된 답변을 포함한 질문을 처리할 때 신중하게 접근해야 함을 나타냅니다.



### Toward Large Language Models as a Therapeutic Tool: Comparing Prompting Techniques to Improve GPT-Delivered Problem-Solving Therapy (https://arxiv.org/abs/2409.00112)
Comments:
          Accepted for AMIA 2024 proceedings

- **What's New**: 이번 연구는 대규모 언어 모델(Large Language Models, LLMs)에서 피드백 기술(prompt engineering)의 효과를 탐구하여, 문제 해결 치료(Problem-Solving Therapy, PST) 세션의 증상 식별 및 평가 단계에서 개인화된 목표 설정을 위한 텍스트 제공 능력을 향상시키는 방안을 제시합니다.

- **Technical Details**: 연구에서는 대규모 언어 모델의 성능을 자동 측정 지표 및 경험이 풍부한 의료 전문가들의 평가를 통해 분석합니다. 문제 해결 치료(PST)의 프로토콜을 따르는 능력을 향상시키기 위한 다양한 프롬프트 기법들을 효과적으로 사용함으로써 모델의 성능을 개선할 수 있으며, 이는 무게 조정 없이 이루어집니다.

- **Performance Highlights**: 모델은 PST 대화 상황에서 인간 치료사와의 비교를 통해 상당한 성능을 보였으며, 다양한 프롬프트 기법을 활용한 결과, 전반적인 질, 일관성, 공감 능력이 향상됨을 확인했습니다. 이는 정신 건강 전문가가 부족한 현 상황에서 LLM이 심리 치료 제공에 기여할 가능성을 넓힐 수 있는 연구입니다.



### Evaluating the Impact of Multiple DER Aggregators on Wholesale Energy Markets: A Hybrid Mean Field Approach (https://arxiv.org/abs/2409.00107)
- **What's New**: 이 연구는 분산 에너지 자원(DER)이 도매 에너지 시장에 통합될 때의 이점을 다루고 있습니다. 특히 여러 DER 집합체가 시장에서 상호작용하면서 가격에 영향력을 문화할 수 있는 방식을 설명합니다.

- **Technical Details**: 본 논문에서는 다수의 DER 집합체가 포함된 도매 시장 모델을 제안합니다. 우리는 Mean-Field Game (MFG)과 Mean-Field Control (MFC)를 결합하여 각 집합체가 시장에서의 장기적인 LMP(trends)를 예측하고 최적의 전략을 학습할 수 있도록 합니다. 또한 강화 학습(RL) 기반 방법을 제안하여 각 에이전트가 시장 조건에 적응할 수 있도록 합니다.

- **Performance Highlights**: 수치적 시뮬레이션 결과, LMP는 하이브리드 Mean-Field 접근 방식에서 빠르게 안정 상태에 도달했습니다. 에너지 저장과 Mean-Field 학습의 조합은 가격 변동성을 현저히 감소시켜, 저장장치가 없는 시나리오와 비교할 때 더 나은 시장 성능을 보여줍니다.



### Zero-Shot Visual Reasoning by Vision-Language Models: Benchmarking and Analysis (https://arxiv.org/abs/2409.00106)
Comments:
          21 pages

- **What's New**: 이번 연구에서는 VLM(vision-language models)의 제로샷(Zero-shot) 시각적 추론 능력을 체계적으로 평가하기 위해 합성 데이터셋(synthetic datasets)을 활용합니다. 이는 기존의 벤치마크가 세계 지식과 혼합되는 점을 명확히 구분하는 시도를 포함합니다.

- **Technical Details**: CLEVR와 PTR 데이터셋을 사용하여 VLM의 순수 시각적 추론 능력을 평가합니다. VLM 내부의 LLM(large language model)에 시각적 임베딩(visual embeddings) 대신 텍스트 기반 장면 설명(textual scene descriptions)을 제공했을 때 성능이 더 좋다는 결과가 나왔습니다. 또한, 체인 오브 사고(chain-of-thought, CoT) 프롬프트가 표준 프롬프트에 비해 더 큰 모델에서만 효과적이라는 것을 발견했습니다.

- **Performance Highlights**: BLIP2-Flan-T5 모델을 활용할 때, 순수 텍스트 정보만으로 18% 더 높은 정확도를 기록하였으며, GPT-4는 GPT-4V보다 CLEVR에서 약 17% 더 높은 정확도를 보였습니다. 그러나 CoT 프롬프트는 더 작은 모델에서는 성능이 떨어지는 것으로 나타나, 모델의 크기가 증가할수록 CoT가 시각적 추론 능력을 개선할 수 있는 잠재력을 지닌다는 것을 시사합니다.



### Negation Blindness in Large Language Models: Unveiling the NO Syndrome in Image Generation (https://arxiv.org/abs/2409.00105)
Comments:
          15 pages, 7 figures

- **What's New**: 이 논문은 최근의 대형 언어 모델(Foundational Large Language Models, LLMs)의 이미지 생성 능력과 관련된 새로운 한계를 제시합니다. 연구팀은 이러한 한계를 'The NO Syndrome'이라고 명명했습니다.

- **Technical Details**: The NO Syndrome은 LLMs가 'NO'와 관련된 자연어 프롬프트를 올바르게 이해하지 못하는 현상으로, 이는 이미지 생성의 정확도에 영향을 미칩니다. 연구자는 다양한 언어(영어, 힌디어, 불어)의 여러 LLM에 대해 시뮬레이션 실험과 엔트로피 기반 통계 분석을 수행했습니다.

- **Performance Highlights**: GPT-4, Gemini, Copilot 등 모든 테스트된 LLM이 이 NO Syndrome의 영향을 받는 것으로 나타났습니다. 이로 인해 생성된 이미지와 텍스트 응답 간의 일관된 불일치가 관찰되었으며, 이는 현재 LLM의 심각한 결함으로 판단됩니다.



### Nuance Matters: Probing Epistemic Consistency in Causal Reasoning (https://arxiv.org/abs/2409.00103)
Comments:
          20 pages

- **What's New**: 이번 연구는 Large Language Models (LLMs)의 인과적 추론에서의 자기 일관성을 평가하기 위해 'causal epistemic consistency'라는 새로운 개념을 제안합니다. 우리는 LLMs의 성능을 평가하기 위한 새로운 지표들을 도입하고, 21개의 고급 LLM을 대상으로 광범위한 실증 연구를 실시하였습니다.

- **Technical Details**: 연구에서는 인과적 에피스템 일관성을 측정하기 위해 세 가지 주요 지표를 제안합니다: (i) intensity ranking concordance, (ii) cross-group position agreement, (iii) intra-group clustering. 이를 통해 LLM이 생성한 중간 결과의 세부적인 차이를 구분할 수 있는 능력을 평가합니다.

- **Performance Highlights**: 실증 연구 결과, 현재 모델들은 인과적 에피스템 일관성을 유지하는 데 어려움을 겪고 있으며, 특히 GPT-4를 포함한 고급 모델조차도 그 성능에 있어 불만족스러운 결과를 보였습니다. 이 연구는 LLMs가 인과 관계의 미세한 차이를 포착하는 데 있어 문제점을 강조합니다.



### Query-by-Example Keyword Spotting Using Spectral-Temporal Graph Attentive Pooling and Multi-Task Learning (https://arxiv.org/abs/2409.00099)
- **What's New**: 이 논문에서는 기존의 키워드 포착(Keyword Spotting, KWS) 시스템의 한계를 극복하기 위해 사용자 정의 키워드를 인식할 수 있는 새로운 Query-by-Example (QbyE) KWS 시스템을 소개합니다. 이 시스템은 spectral-temporal graph attentive pooling (GAP)과 multi-task learning을 적용하여, 사용자가 원하는 키워드를 정의할 수 있고, 사용자 맞춤형 경험을 제공합니다.

- **Technical Details**: 제안된 KWS 시스템은 LiCoNet, Conformer, ECAPA_TDNN 등 세 가지 서로 다른 네트워크 아키텍처를 채택하여 인코더 모델링을 수행합니다. 이 시스템은 하드웨어 효율적인 구조로, GAP을 통해 informative embeddings를 생성하고 하이브리드 손실 함수를 통해 단어와 음소의 구분력을 높이고, 발화자에 따른 변동성을 줄입니다.

- **Performance Highlights**: 실험 결과, LiCoNet이 Conformer 모델과 비교해 13배 더 효율적으로 비슷한 성능을 달성하였으며(각각 1.98%와 1.63%의 FRR), 제안된 QbyE 프레임워크의 효율성을 입증하였습니다. 이는 사용자가 정의한 키워드의 고유성을 보장하는 맞춤형 KWS 시스템의 가능성을 보여줍니다.



### Large Language Models for Disease Diagnosis: A Scoping Review (https://arxiv.org/abs/2409.00097)
Comments:
          57 pages

- **What's New**: 최근 인공지능(AI) 분야에서 자동 질병 진단의 중요성이 증가하고 있습니다. 대형 언어 모델(LLMs)의 발전으로 이러한 진단 작업에서의 효율성이 입증되었습니다. 하지만 여전히 해결되지 않은 중요한 연구 질문이 많습니다. 본 논문에서는 질병 진단을 위한 LLM 기반 방법을 포괄적으로 분석하였습니다.

- **Technical Details**: 이 연구는 LLM을 사용한 질병 진단에 관한 기존 연구를 검토하고 다양한 질병 유형, 관련 장기 시스템, 임상 데이터, LLM 기술 및 평가 방법을 분석하였습니다. LLM은 텍스트, 이미지, 비디오, 오디오, 표 형식 데이터 및 시계열 데이터를 포함한 다양한 데이터 모드를 처리합니다. 프롬프트 엔지니어링 기술도 조사하였으며, 하드 프롬프트와 소프트 프롬프트 두 가지 유형으로 나뉘어 설명되었습니다.

- **Performance Highlights**: LLMs는 COVID-19 진단에서 높은 정확도를 보여주었으며, PathChat은 인간 병리학에서 최신 성능을 달성했습니다. 또한, GPT-4는 강박장애를 식별할 때 정신 건강 전문가를 초월하는 성능을 보였습니다. 그러나 LLM의 성능을 평가하기 위해 적합한 평가 방법이 여전히 연구되어야 합니다.



### Non-instructional Fine-tuning: Enabling Instruction-Following Capabilities in Pre-trained Language Models without Instruction-Following Data (https://arxiv.org/abs/2409.00096)
Comments:
          16 pages, 2 figures, 15 tables

- **What's New**: 본 논문에서는 비지침적 데이터(non-instructional data)를 이용하여 LLM(대형 언어 모델)의 지침을 따르는 능력을 향상시킬 수 있다는 새로운 접근 방식을 제안합니다. 연구팀은 OpenWebText의 무작위 텍스트 절반을 지침으로 사용하고, GPT-3.5-turbo 또는 GPT-4-turbo를 활용하여 나머지 텍스트를 완료하는 방식을 적용했습니다. 이 연구는 수많은 프리 트레인(pre-trained) LLM을 Fine-tuning하여 이 결과를 확인했습니다.

- **Technical Details**: 연구에서 제안한 방법론은 비지침적 데이터셋을 생성하기 위한 간단한 프레임워크를 제공하며, 조건부 증류(conditional distillation) 및 연속 작성을 통한 지식 증류(knowledge distillation)를 이용하여 데이터를 생성합니다. 이를 통해 LLM의 성능을 높이고, 다양한 벤치마크에서 우수한 결과를 도출합니다.

- **Performance Highlights**: LLaMA-3-70B-Instruct가 Arena Hard 리더보드에서 LLaMA-3.1-70B-Instruct와 동등한 성능을 보였으며, Meta-Llama-3-70b-Instruct 모델은 57.0의 최고 기록을 달성하여, 기존의 SFT(supervised fine-tuning) 데이터셋을 초월했습니다. 이러한 결과는 비지침적 데이터를 통한 Fine-tuning의 효과를 확증합니다.



### Examining Independence in Ensemble Sentiment Analysis: A Study on the Limits of Large Language Models Using the Condorcet Jury Theorem (https://arxiv.org/abs/2409.00094)
- **What's New**: 본 연구는 Condorcet Jury 이론을 감정 분석(sentiment analysis) 분야에 적용하여 다양한 대형 언어 모델(LLM)과 단순 자연어 처리(NLP) 모델의 성능을 비교합니다. 실험 결과, 대형 모델을 사용했음에도 불구하고 성능 향상이 미미하다는 것을 발견했습니다.

- **Technical Details**: 본 연구에서는 Condorcet Jury 이론을 확장하여 다중 클래스 분류에 대한 이론적 프레임워크를 제공합니다. 연구는 LLM과 다른 NLP 모델 간의 독립성이 제한적이라는 가정을 기반으로, 다수결(classifier)을 통한 감정 분석의 정확성을 평가합니다.

- **Performance Highlights**: 대형 언어 모델(예: ChatGPT 4) 사용 시 성능 향상은 미미하며, 간단한 모델들 (예: FinBERT, DistilRoBERTa)보다 큰 개선이 없음을 보여주었습니다. 이 결과는 대형 모델들이 감정 분석 과제에서 충분히 독립적이지 않다는 점을 시사합니다.



### PatentGPT: A Large Language Model for Patent Drafting Using Knowledge-based Fine-tuning Method (https://arxiv.org/abs/2409.00092)
Comments:
          21 pages, 4 figures

- **What's New**: 본 연구는 지식 정밀 조정(Knowledge Fine-Tuning, KFT) 프레임워크를 통해 대형 언어 모델(Large Language Models, LLMs)의 전문 지식과 맥락 인식을 강화하여, AI가 자율적으로 특허 문서를 생성할 수 있도록 하는 혁신적인 접근 방식을 제안합니다.

- **Technical Details**: 제안된 모델인 PatentGPT는 지식 그래프 기반의 사전 학습(pre-training), 도메인 특화된 감독 학습(supervised fine-tuning, SFT), 인간 피드백에서의 강화 학습(Reinforcement Learning From Human Feedback, RLHF)을 결합하여 개발되었습니다.

- **Performance Highlights**: PatentGPT는 특허 관련 기준 벤치마크 테스트에서 최신 모델보다 약 400% 높은 성능을 기록하며, 인간의 창의성과 혁신을 보조하고 증대시키는 능력을 가지고 있음을 입증했습니다.



### Classification of Safety Events at Nuclear Sites using Large Language Models (https://arxiv.org/abs/2409.00091)
- **What's New**: 이 논문은 원자력 발전소의 Station Condition Records (SCRs)를 안전 관련 및 비안전 관련 카테고리로 분류하기 위한 대형 언어 모델 (Large Language Model, LLM) 기반의 기계 학습 분류기 개발을 제안합니다. 기존의 수동 검토 프로세스를 보완하여 안전 분류 프로세스의 효율성과 정확성을 향상시키는 것이 주요 목표입니다.

- **Technical Details**: 이 논문에서는 레이블이 부착된 SCR 데이터셋을 분류하기 위한 실험을 수행하고 분류기의 성능을 평가합니다. 여러 가지 프롬프트 변형 (prompt variations) 구성을 탐구하고 이들이 LLM의 의사결정 프로세스에 미치는 영향을 관찰했습니다. 또한, SCR 안전 분류에 대한 보다 미세하고 유연한 접근 방식을 제공할 수 있는 숫자 점수 메커니즘을 도입했습니다.

- **Performance Highlights**: 이 방법은 원자력 안전 관리에 혁신적인 단계를 나타내며, 안전 이벤트 식별을 위한 확장 가능한 도구를 제공함으로써 안전 분류 프로세스를 개선하는 데 기여합니다.



### Evaluating ChatGPT on Nuclear Domain-Specific Data (https://arxiv.org/abs/2409.00090)
- **What's New**: 이번 논문은 ChatGPT와 같은 대형 언어 모델(LLM)의 핵심 분야인 핵 데이터에서 질문-응답(Q&A) 작업에 대한 적용을 검토합니다.

- **Technical Details**: 논문은 LLM의 응답을 직접적으로 얻는 방식과 RAG(리트리벌 증강 생성) 프레임워크 내에서 LLM의 응답을 비교하여 평가합니다. RAG는 외부 지식 기반과 복잡한 검색 기법을 통합하여 출력의 정확성과 관련성을 향상시키는 방법입니다.

- **Performance Highlights**: 연구 결과, RAG 파이프라인을 포함시킬 경우 LLM의 성능이 향상되며, 특히 핵 분야의 특정 질문에 대해 더 정확하고 적절한 응답을 생성하는 데 기여한다고 합니다.



### Watermarking Techniques for Large Language Models: A Survey (https://arxiv.org/abs/2409.00089)
Comments:
          Preprint. 19 figures, 7 tables

- **What's New**: 본 논문은 LLM(대형 언어 모델) 워터마킹 기술에 대한 최초의 포괄적인 리뷰를 제공하며, 기존 워터마킹 기술의 역사와 현재 LLM 워터마킹 연구 상태를 자세히 분석합니다.

- **Technical Details**: LLM 워터마킹은 데이터 추적 기능, 저작권 보호 기능, LLM이 생성한 콘텐츠 식별 기능으로 나눌 수 있습니다. 또한 텍스트, 이미지, 비디오, 오디오의 다양한 형태로 구분할 수 있으며, 각 형태에서 발생하는 도전 과제도 다룹니다.

- **Performance Highlights**: 현재 LLM 워터마킹의 주요 문제는 워터마크의 강건성, 의미 불변성, 보안 취약점 및 시스템 소비 문제입니다. 연구가 진행 중이며, 앞으로 LLM 워터마킹의 효과적인 연구와 응용 가능성을 제시합니다.



### A Lightweight Human Pose Estimation Approach for Edge Computing-Enabled Metaverse with Compressive Sensing (https://arxiv.org/abs/2409.00087)
- **What's New**: 이 논문은 5G/6G 네트워크를 활용하여 사용자의 3D 움직임을 추정하는 모델을 제안합니다. IMU (Inertial Measurement Unit) 센서에서 수집한 데이터를 통해 생성된 노이즈가 있는 신호를 효율적으로 처리하는 방법에 대해 다루고 있습니다.

- **Technical Details**: 제안된 방법은 랜덤 가우시안 매트릭스를 사용하여 원 신호를 저차원 공간으로 변환하고, 압축 센싱(compressive sensing) 이론을 활용하여 신호를 압축합니다. 수신측에서는 심층 생성 모델(deep generative model)을 개발하여 노이즈가 있는 압축 데이터를 통해 원래의 IMU 신호를 복원합니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 원 신호의 82%만을 사용하여 사용자 3D 포즈를 매우 정확하게 추정할 수 있으며, 이는 Lasso와 같은 최적화 기반 접근 방식과 비교할 때 속도 면에서도 훨씬 우수합니다.



### Vision-Language and Large Language Model Performance in Gastroenterology: GPT, Claude, Llama, Phi, Mistral, Gemma, and Quantized Models (https://arxiv.org/abs/2409.00084)
Comments:
          Manuscript Pages: 34, Figures: 7, Tables: 2, Supplementary File Pages: 35, Data Transparency Statement: Code is available at: this https URL . Study data from American College of Gastroenterology (ACG) are restricted and available upon request with ACG permission. Correction: updated abstract considering Llama3.1 results

- **What's New**: 이 연구는 대형 언어 모델(LLMs)과 비전 언어 모델(VLMs)의 위장관학(gastroenterology) 분야에서의 의료 추론 성능을 평가한 최초의 체계적인 분석으로, 다양한 모델 설정과 파라미터, 프롬프트 엔지니어링 전략의 영향을 조사했습니다.

- **Technical Details**: 300개의 위장관학 보드 시험 스타일의 객관식 질문을 사용하였으며, 138개 질문에 이미지를 포함시켰습니다. GPT-3.5를 활용하여 다양한 상업용 및 오픈 소스 LLM(버전), 인터페이스(웹 및 API), 컴퓨팅 환경(클라우드 및 로컬), 모델 정밀도(양자화 유무)에 대해 성능을 평가했습니다.

- **Performance Highlights**: 상업용 모델 중에서는 GPT-4o(73.7%)와 Claude3.5-Sonnet(74.0%)가 최고 정확도를 기록하였고, 오픈 소스 모델에서는 Llama3.1-405b(64%)가 가장 높은 성능을 보였습니다. 특히, 이미지가 포함된 질문에서 VLM의 성능이 저하되는 현상이 관찰되었고, 인간이 제작한 이미지 설명이 포함되었을 때는 10%의 정확도 증가가 있었습니다.



### On-device Learning of EEGNet-based Network For Wearable Motor Imagery Brain-Computer Interfac (https://arxiv.org/abs/2409.00083)
- **What's New**: 본 논문에서는 Electroencephalogram (EEG) 기반의 Motor Imagery 인식에 대한 새로운 경량화 기술을 제안하며, 사용자가 등록되지 않았더라도 실시간으로 EEG 신호에 적응할 수 있는 on-device learning engine을 구현하였습니다. 이는 다양한 사용자 집단 간의 feature distribution drift 문제를 해결하는 데 중점을 두고 있습니다.

- **Technical Details**: 제안된 방법은 EEGNet 아키텍처에 적용되며, Greeenwaves의 GAP9 프로세서를 활용하여 낮은 전력의 RISC-V 기반 병렬 프로세서에서 작동합니다. 15.6 KByte의 메모리 풋프린트에서 최대 7.31%의 정확도 향상이 이루어졌습니다. 온라인 학습 동안 단일 추론 시 14.9 ms의 추론 시간과 0.76 mJ의 전력 소비이며, 단일 업데이트 시 20 us와 0.83 uJ의 전력 소비를 기록하였습니다.

- **Performance Highlights**: 본 연구에서 제안한 접근 방식은 등록되지 않은 사용자에 대한 EEG 모션 이미저리의 정확도를 크게 향상시킬 수 있음을 보여주었으며, 사전 훈련된 모델의 전송 학습을 기반으로 하여 on-site에서 지속적으로 학습할 수 있는 가능성을 제시했습니다. 이는 베터리 구동 웨어러블 AI 시스템에도 효과적으로 적용될 수 있습니다.



### Towards Human-Level Understanding of Complex Process Engineering Schematics: A Pedagogical, Introspective Multi-Agent Framework for Open-Domain Question Answering (https://arxiv.org/abs/2409.00082)
Comments:
          Our paper is accepted for publication at ML4CCE workshop at ECML PKDD 2024

- **What's New**: 이 논문에서는 프로세스 흐름도(Process Flow Diagrams, PFD)와 배관 및 계기 다이어그램(Piping and Instrumentation Diagrams, P&ID)의 분석을 위한 안전한 기업 솔루션을 제안합니다. 제안된 솔루션은 다계층의 다중 에이전트 Retrieval Augmented Generation (RAG) 프레임워크를 이용하여 개방형 질문 응답(open-domain question answering) 작업을 수행하며 데이터 프라이버시 및 비용 효율성을 증대시킵니다.

- **Technical Details**: 제안된 다중 에이전트 프레임워크는 전문화된 하위 에이전트와 introspective(내성적) 메타 에이전트로 구성됩니다. 각 하위 에이전트는 PFD와 P&ID 분석을 위해 ReAct(Reason + Act) 촉진 기법을 사용하여 정보를 통합합니다. 또한, 사용자 쿼리에 대해 태스크 플래닝을 통해 명령 및 월등한 응답 생성을 위한 외부 API와의 상호작용을 조절합니다.

- **Performance Highlights**: 실험 결과, 제안한 접근법이 이미지 캡셔닝, 시각 질문 응답(VQA), 텍스트 검출(OCR) 작업 등에서 기존의 최신 기술과 동등한 성능을 보여주었으며, 맞춤화, 해석 가능성, 데이터 프라이버시 및 비용 효율성을 제공합니다.



### Learning to Plan Long-Term for Language Modeling (https://arxiv.org/abs/2409.00070)
Comments:
          preprint

- **What's New**: 이 논문에서는 미래 텍스트 예측의 효율성을 높이기 위해 다단계(planner) 접근 방식을 제안합니다. 기존의 단일 단계 계획 기능을 확장하여 더 나은 텍스트 예측을 가능하게 하는 계획 수립 과정을 도입합니다.

- **Technical Details**: 이 방법은 다음 세 단계로 이루어져 있습니다: 1) unlabeled (라벨 없는) 텍스트에서 추론한 행동(action) 시퀀스를 생성, 2) 다단계 (multi-step) planner를 훈련하여 다음 행동을 예측, 3) planner에서 샘플링된 여러 경로를 사용하여 언어 모델을 조건화합니다. 이를 통해 효과적인 예측 코딩이 가능해집니다.

- **Performance Highlights**: 실험 결과, 기존 방법과 비교하여 다단계 계획을 통해 예측 정확도가 향상됨을 확인할 수 있었습니다. 특히, 다단계 예측은 코딩 작업에서 최대 17%의 성능 개선을 보여줍니다.



### How to Measure Human-AI Prediction Accuracy in Explainable AI Systems (https://arxiv.org/abs/2409.00069)
- **What's New**: 이 연구는 인공지능(AI) 시스템의 투명성을 수학적 측정을 통해 평가하는 새로운 방법론을 제시합니다. 특히, 인간의 예측 성능을 측정하는 기존의 이진(binary) 프레임이 가지고 있는 한계점을 논의하고, '부분적 오차(partial wrongness)'를 측정하는 수학적 기초를 제안합니다.

- **Technical Details**: 연구는 36개의 선택지와 4개의 선택지를 가진 행동 공간에서의 두 가지 실험을 통해 진행되었습니다. 첫 번째 실험은 86명의 참가자를 대상으로 한 in-lab 연구로, MNK 게임에서 에이전트의 다음 행동을 예측하게 하였습니다. 두 번째 실험은 기존의 연구 데이터를 재 분석하여 4개의 선택지 중 에이전트의 행동을 예측하도록 하였습니다. 해당 결과를 바탕으로 부분적 오차를 평가할 수 있는 새로운 방법론을 제시합니다.

- **Performance Highlights**: 연구는 기존의 이진 예측 방법이 충분한 정보를 제공하지 못함을 강조하며, AI 행동의 예측성이 높아짐에 따라, 보다 세분화된 예측 평가 방법이 필요하다는 것을 보여줍니다. 특히, 넓은 행동 공간을 가진 경우 이 한계가 더욱 두드러집니다.



### Phrasing for UX: Enhancing Information Engagement through Computational Linguistics and Creative Analytics (https://arxiv.org/abs/2409.00064)
- **What's New**: 이번 연구는 디지털 플랫폼에서 텍스트 특징과 정보 참여(Information Engagement, IE) 간의 관계를 탐구합니다. 이 연구는 사용자의 상호작용에 대한 계산언어학(computational linguistics)과 분석(analytics)의 영향을 강조합니다.

- **Technical Details**:  연구에서 제안된 READ 모델은 대표성(representativeness), 사용 용이성(ease of use), 정서(affect), 분포(distribution)와 같은 주요 예측 변수들을 정량화하여 참여 수준을 예측합니다. AB 테스트 및 무작위 시험(randomized trials)을 통해 모델의 효과가 검증되었습니다.

- **Performance Highlights**: 참여(participation) 정확도 0.94, 인식(perception) 정확도 0.85, 지속성(perseverance) 정확도 0.81, 그리고 전체 정보 참여(IE) 정확도는 0.97에 이릅니다. 연구 결과에 따르면, READ 모델의 인사이트에 따라 텍스트를 수정하면 대표성과 긍정적 정서가 증가해 선택률이 11% 상승하고, 평균 평가가 3.98에서 4.46으로 증가하며, 유지율도 11% 개선됩니다.



### Enhancing Natural Language Inference Performance with Knowledge Graph for COVID-19 Automated Fact-Checking in Indonesian Languag (https://arxiv.org/abs/2409.00061)
- **What's New**: 이 연구는 COVID-19 관련 허위 정보 검증을 위한 자동화된 사실 확인 시스템의 성능을 높이기 위해 Knowledge Graph (KG)를 활용한 새로운 접근 방식을 제안합니다.

- **Technical Details**: 제안된 모델은 세 가지 모듈로 구성됩니다: fact module (사실 모듈), NLI module (자연어 추론 모듈), classifier module (분류기 모듈). 사실 모듈은 KG에서 정보를 처리하고, NLI 모듈은 주어진 전제와 가설 간의 의미적 관계를 처리합니다. 두 모듈에서 생성된 표현 벡터는 결합되어 분류기 모듈에 입력됩니다.

- **Performance Highlights**: 모델은 인도네시아어로 생성된 COVID-19 사실 확인 데이터셋과 COVID-19 KG Bahasa Indonesia를 사용하여 훈련되었습니다. KG를 통합함으로써 NLI 성능이 크게 향상되며, 최고 정확도 0.8616을 달성하였습니다.



### Automating Knowledge Discovery from Scientific Literature via LLMs: A Dual-Agent Approach with Progressive Ontology Prompting (https://arxiv.org/abs/2409.00054)
Comments:
          in submission

- **What's New**: 이 논문에서는 대량의 문헌에서의 지식 발견 자동화 문제를 해결하기 위해 Large Language Models (LLMs)를 기반으로 한 혁신적인 프레임워크인 LLM-Duo를 소개합니다. 이는 Progressive Ontology Prompting (POP) 알고리즘과 Dual-Agent System을 결합해 지식 추출 자동화를 향상시키는 것입니다.

- **Technical Details**: POP 알고리즘은 미리 정의된 온톨로지를 기준으로 우선순위가 부여된 breadth-first search (BFS)를 활용하여 구조화된 프롬프트 템플릿 및 행동 순서를 생성합니다. LLM-Duo는 '탐색자'와 '평가자'라는 두 개의 LLM 에이전트를 통해 협력적이고 경쟁적으로 작업하여 발견 및 주석 프로세스의 신뢰성을 높입니다.

- **Performance Highlights**: 본 논문에서 제시한 방법은 64,177개의 연구 논문에서 2,421개의 언어개입(intervention)을 발견하였으며, 이를 바탕으로 스피치-언어 치료 커뮤니티에 기여할 수 있는 공개적인 지식 베이스를 구성했습니다. 실험 결과, 제안하는 방법은 기존의 고급 기준을 초과하는 정확하고 완전한 주석을 가능하게 하였습니다.



### Needles in Needle Stacks: Meaningful Clinical Information Buried in Noisy Waveform Data (https://arxiv.org/abs/2409.00041)
Comments:
          Machine Learning For Health Care 2024 (MLHC)

- **What's New**: 이 논문은 중앙 정맥선(C-Lines)과 동맥선(A-Lines)에서 발생하는 고주파 혈압 파형의 잡음 아티팩트를 실시간으로 탐지하는 머신 러닝 모델을 개발하고 이를 통한 문서화 자동화의 가능성을 제시하였습니다. 특히, 라인 접근 이벤트를 효과적으로 감지하여 환자 안전과 임상 효율성을 높이는 데 중점을 두었습니다.

- **Technical Details**: 제안된 시스템은 고주파 데이터에서 클리니션이 잘 알고 있는 특정 아티팩트를 인식하는 ML(Classifier)을 사용하며, 전통적인 신호 처리 방법은 아티팩트를 제거하는 데 중점을 둡니다. 논문에서는 이러한 아티팩트를 활용해 실시간으로 라인 접근 이벤트를 식별할 수 있는 알고리즘을 설계하였습니다.

- **Performance Highlights**: ML 모델은 실제 아동 병원에서 평가되었으며, 문서화 부담을 줄이고 클리니션에게 더 많은 정보를 제공하며, 환자 안전 개선을 위한 여러 이니셔티브에 기여할 가능성을 입증했습니다.



### Quality Assessment in the Era of Large Models: A Survey (https://arxiv.org/abs/2409.00031)
- **What's New**: 이 논문은 대형 모델(large models)의 등장으로 인한 품질 평가(quality assessment) 분야의 변화를 다루고 있으며, 최신 품질 평가 방식에 대한 종합적인 리뷰를 제공합니다.

- **Technical Details**: 이 논문은 품질 평가를 주관적(subjective) 및 객관적(objective) 방법으로 나누어 설명하며, 대형 모델 시대의 품질 평가 방법들과 관련한 최신 연구 동향 및 기술을 다룹니다. 기존의 작은 전문가 모델(expert models)과 대형 언어 모델(large language models, LLMs) 및 다중 모드 모델(large multi-modal models, LMMs)을 활용한 평가 방법이 비교됩니다.

- **Performance Highlights**: 대형 모델을 활용한 품질 평가의 발전을 통해 설명 가능성(explainability)이 향상되었으며, AI 생성 콘텐츠(AIGC)의 품질 감시와 개선을 위한 새로운 벤치마크와 방법이 개발되고 있습니다. 이는 다양한 플랫폼에서도 멀티미디어 콘텐츠 최적화에 기여하고 있습니다.



### TimeSense: Multi-Person Device-free Indoor Localization via R (https://arxiv.org/abs/2409.00030)
- **What's New**: 본 논문에서는 TimeSense라는 심층 학습 기반의 다중인원 장치 비대면 실내 위치추적 시스템을 제안합니다. 이 시스템은 인간 존재에 의해 유도되는 환경의 동적 변화를 반영하는 신호 왕복 시간을 측정하여 문제를 해결합니다.

- **Technical Details**: TimeSense는 IEEE 802.11-2016 표준의 정밀 시간 측정 프로토콜을 활용하여 인간의 존재에 의해 변동되는 왕복 시간(round trip time, RTT) 정보를 이용합니다. 이 시스템은 스택형 잡음 제거 오토인코더(stacked denoising auto-encoder) 모델을 사용하여 비정상 행동을 효과적으로 감지하고 사용자 위치를 추정합니다.

- **Performance Highlights**: TimeSense는 두 개의 실제 환경에서 평가된 결과, 각각 1.57m 및 2.65m의 중위 위치 추적 정확도를 달성했습니다. 이는 기존 기술에 비해 각각 49%와 103%의 성능 향상을 보여주었습니다.



### Detecting Misinformation in Multimedia Content through Cross-Modal Entity Consistency: A Dual Learning Approach (https://arxiv.org/abs/2409.00022)
Comments:
          Accepted to PACIS 2024. 15 pages, 3 figures

- **What's New**: 이번 연구에서는 Multimodal (다중모달) 형식의 허위 정보 탐지에 대한 새로운 접근 방식을 제시합니다. 특히, 기존 연구들은 주로 단일 모달리티나 텍스트-이미지 조합에 중점을 두었지만, 비디오 콘텐츠를 포함한 다중모달 형식에서의 허위 정보 탐지의 필요성을 강조합니다.

- **Technical Details**: 우리는 Multimedia Misinformation Detection (MultiMD) 프레임워크를 제안합니다. 이 프레임워크는 cross-modal (크로스 모달) entity consistency를 활용하여 다중모달 허위 정보를 탐지합니다. Dual learning (듀얼 러닝) 접근 방식을 사용함으로써, 탐지 성능을 향상시키고 다양한 모달리티에서의 entity consistency 표현 학습을 개선할 수 있습니다.

- **Performance Highlights**: 실험 결과, MultiMD는 최신 기술의 기준 모델들보다 우수한 성능을 보였으며, 각 모달리티가 허위 정보 탐지에서의 중요성을 강조합니다. 이 연구는 다중모달 허위 정보 탐지에 대한 새로운 방법론적 및 기술적 통찰을 제공합니다.



### TACOS: Task Agnostic Continual Learning in Spiking Neural Networks (https://arxiv.org/abs/2409.00021)
- **What's New**: 이번 연구에서는 TACOS라는 새로운 spiking neural network (SNN) 모델을 소개하여, 지속적인 학습 중에 발생할 수 있는 catastrophic interference 문제를 해결합니다. 이전의 생물학적 영감을 받은 접근 방식과 달리 TACOS는 특정 작업 인식 없이도 작동하여, 메모리 크기를 고정한 상태에서 효율적으로 새로운 정보를 학습할 수 있습니다.

- **Technical Details**: TACOS는 시냅스 지역 정보만을 활용하여 시냅스 통합(synaptic consolidation)과 메타플라스틱성(metaplasticity)을 포함하는 복잡한 시냅스 모델을 통해 설계되었습니다. 이는 인공 신경망이 새로운 작업을 학습하면서도 이전의 정보를 보호할 수 있게 해줍니다.

- **Performance Highlights**: TACOS는 연속적인 이미지 인식 작업에서 기존의 정규화 기법들보다 더 효과적으로 catastrophic interference를 줄이는 것으로 나타났습니다. 또한 모델의 성능은 추가된 파라미터 수가 적고 고정된 메모리 크기로 새로운 작업을 학습할 때 변하지 않는 특성을 보여주었습니다.



### Navigating the sociotechnical labyrinth: Dynamic certification for responsible embodied AI (https://arxiv.org/abs/2409.00015)
- **What's New**: 본 논문은 AI 시스템 발전에 따른 새로운 사회기술적 규제 접근 방식인 '다이나믹 인증(dynamic certification)'을 제안합니다. 이는 변화하는 AI 기술에 적응할 수 있는 규제 체계를 만들어, 기술의 진보와 규제의 간극을 해소하고자 합니다.

- **Technical Details**: 다이나믹 인증은 특정 사용 및 맥락에 적합한 규제나 테스트의 반복적인 개발 과정을 포함합니다. 이는 시스템 성능에 대한 정보를 수집하고, 안전성이 보장될 경우 점진적으로 허용되는 사용이나 맥락을 확장해 나가는 방식입니다. 이러한 방법은 특히 투명성과 일관성이 결여된 AI 시스템에 효과적입니다.

- **Performance Highlights**: 제안된 접근 방식은 AI 시스템의 안전하고 윤리적인 배치를 보장하며, 실제 운영하는 맥락과 양방향 연결을 유지하는 데 중점을 둡니다. 이 연구는 기존의 규제 모델에 비해 더 유연하고, 다양한 분야의 요구를 수용할 수 있는 솔루션을 제공하여, AI 기술의 신뢰성을 높이는 기여를 할 것으로 기대됩니다.



### DivDiff: A Conditional Diffusion Model for Diverse Human Motion Prediction (https://arxiv.org/abs/2409.00014)
- **What's New**: 이 논문에서는 다양한 인간 운동 예측(Diverse Human Motion Prediction, HMP)에서 보다 다양하고 현실적인 예측을 제공하기 위해 새로운 조건부 확산 모델인 DivDiff를 제안합니다. DivDiff는 Denoising Diffusion Probabilistic Model (DDPM)을 기반으로 하여 이전 HMP 방법들이 가지고 있던 한계를 극복하는 데 중점을 둡니다.

- **Technical Details**: DivDiff 모델은 Discrete Cosine Transform (DCT)와 transformer 메커니즘을 사용하여 관찰된 인간 동작 시퀀스를 인코딩하고, 이를 통해 DDPM의 역전 과정에 대한 조건을 설정합니다. 추가로 Diversified Reinforcement Sampling Function (DRSF)을 통해 인간의 골격 제약을 적용하여 예측된 움직임의 품질을 개선합니다. DRSF는 그래프 합성곱 네트워크를 활용하여 인간 뼈 사이 내부 관계를 효과적으로 포착합니다.

- **Performance Highlights**: 실험 결과는 Human3.6M 및 HumanEva-I 데이터셋에서 DivDiff가 기존의 최첨단 방법들에 비해 다양성과 정확성 모두에서 경쟁력 있는 성능을 달성했음을 보여줍니다. 특히, 이 모델은 보다 현실적이고 다양한 3D 인간 동작 예측을 위한 현대적인 방법으로 자리매김할 Potential을 가지고 있습니다.



### AVIN-Chat: An Audio-Visual Interactive Chatbot System with Emotional State Tuning (https://arxiv.org/abs/2409.00012)
- **What's New**: 이번 연구는 사용자가 3D 아바타와 실시간으로 대화할 수 있는 오디오-비주얼 인터랙티브 챗봇 시스템인 AVIN-Chat을 제안합니다. 기존 챗봇 서비스는 텍스트 또는 음성만을 제공했지만, AVIN-Chat은 오디오-비주얼 커뮤니케이션을 통해 더 나은 사용자 경험을 제공합니다.

- **Technical Details**: AVIN-Chat 시스템은 세 가지 주요 서브 모듈로 구성됩니다: (1) 변형 가능한 현실적인 얼굴 생성, (2) 실시간 음성 인식 및 응답, (3) 응답에 맞춘 아바타의 얼굴 표정 및 입 모양 생성을 포함합니다. 이 시스템에서는 추가적으로 사용자 감정 상태에 따라 대화 중 감정 표현을 조정하는 기능이 포함되어 있습니다.

- **Performance Highlights**: 사용자 주관적 테스트 결과, AVIN-Chat 시스템은 기존의 텍스트 또는 음성 기반 AI 챗봇 시스템에 비해 높은 몰입감을 제공하는 것으로 나타났습니다. 이는 사용자와 챗봇 간의 강한 유대감을 형성하여 더욱 몰입한 경험을 가능하게 합니다.



### Web Retrieval Agents for Evidence-Based Misinformation Detection (https://arxiv.org/abs/2409.00009)
Comments:
          1 main figure, 8 tables, 10 pages, 12 figures in Appendix, 7 tables in Appendix

- **What's New**: 이 논문은 잘못된 정보를 탐지하기 위한 에이전트 기반 자동 사실 확인 접근 방식을 개발하였습니다. 강력한 LLM 에이전트와 온라인 웹 검색 에이전트를 결합하여 독립적으로 사용할 때보다 더 나은 결과를 달성하는 방법을 보여줍니다.

- **Technical Details**: 우리는 LLM을 사용해 쿼리를 생성하고, 웹 검색 엔진에 연결된 또 다른 LLM로 이들을 답하는 방식으로 접근법을 평가했습니다. 이 방법은 Vicuna, Mixtral, Claude, GPT-3.5 및 두 개의 GPT-4 모델을 포함한 다양한 모델에서 성능을 평가하고, 웹 검색 기법이 모든 모델의 성능을 향상시키는 것을 발견했습니다.

- **Performance Highlights**: 잘못된 정보 탐지에 대한 성능이 최대 20% 향상되었습니다. 다양한 소스의 사용과 편향, 필요 증거 유형, 그리고 파이프라인의 다양한 구성 요소에 대한 심층 분석을 통해 시스템의 실용성과 기회를 강조하였습니다.



### Federated Sequence-to-Sequence Learning for Load Disaggregation from Unbalanced Low-Resolution Smart Meter Data (https://arxiv.org/abs/2409.00007)
- **What's New**: 본 논문에서는 저해상도(기본 제공하는 시간당) 데이터를 활용하여 비침입 전력 모니터링(Non-Intrusive Load Monitoring, NILM)을 통해 12개의 다양한 전력 소비 기기의 부하 분해를 수행하는 새로운 접근 방식을 제안합니다. 특히, 날씨 데이터(temperature와 humidity)를 사용하여 부하 분해 성능을 획기적으로 향상시키고 있습니다.

- **Technical Details**: 제안된 모델은 seq2seq(sequence-to-sequence) 구조를 기반으로 하며, 입력으로 날씨 데이터와 총 부하 데이터를 받아 내부 인코더(encoder)를 통해 전 세계적인 정보(global information)를 추출합니다. 이 정보는 디코더(decoder)로 전달되어, 데이터 공유 없이 기기별 전력 소비를 시간 단위로 분해합니다. 본 연구에서는 L2GD(federated learning framework)를 통해 통신 부담을 줄이는 동시에 데이터의 이질성(data heterogeneity) 문제를 해결합니다.

- **Performance Highlights**: 실험 결과에 따르면, 제안한 L2GD 모델은 동질적 및 이질적 데이터 시나리오에서 FedAvg 및 FedProx에 비해 약 절반의 통신 라운드(communication rounds)를 통해 유사한 성능을 달성하였으며, 데이터 이질적 조건에서도 우수한 성능을 나타내었습니다.



### Csi-LLM: A Novel Downlink Channel Prediction Method Aligned with LLM Pre-Training (https://arxiv.org/abs/2409.00005)
- **What's New**: 본 연구에서는 Csi-LLM이라는 신규 하향 채널 예측 기법을 소개합니다. 이 기법은 가변 길이의 과거 시퀀스를 모델링하며, 자연어 처리에서 사용되는 모달리티와 무선 통신 데이터를 효과적으로 정렬하여 LLM의 패턴 인식 및 추론 능력을 활용할 수 있도록 합니다.

- **Technical Details**: Csi-LLM은 대규모 다중 입력 다중 출력(massive MIMO) 시스템의 하향 링크 채널 예측에서 LLM의 다음 토큰 생성 기능을 활용합니다. 모델은 고정 길이의 역사적 시퀀스에 의존하지 않고, 데이터 처리 및 최적화 목표를 LLM의 목적과 일치시켜 설계되었습니다. 이를 통해 다양한 이동 상황에 적합한 유연한 모델을 제시합니다.

- **Performance Highlights**: 시뮬레이션 결과, Csi-LLM은 다양한 시나리오에서 안정적인 성능 개선을 지속적으로 보여주었으며, 연속적 다단계 예측에서 중요한 잠재력을 입증했습니다.



### Evaluating Explainable AI Methods in Deep Learning Models for Early Detection of Cerebral Palsy (https://arxiv.org/abs/2409.00001)
- **What's New**: 이번 연구는 Deep Learning 기법을 활용하여 아기 움직임에서 추출한 골격 데이터를 분석하여 Cerebral Palsy (CP)를 예측하는 Explainable AI (XAI) 방법의 신뢰성과 적용 가능성을 테스트하였습니다. 특히, Class Activation Mapping (CAM)과 Gradient-weighted Class Activation Mapping (Grad-CAM)의 신뢰도를 정량적으로 평가하기 위한 XAI 평가 메트릭스인 신뢰성(faithfulness)과 안정성(stability)을 사용했습니다.

- **Technical Details**: 이 연구는 CP 예측을 위한 Graph Convolutional Network (GCN) 모델에서 CAM 및 Grad-CAM과 같은 XAI 기법을 적용하고, 이들 기법이 CP 예측에 영향을 미치는 중요한 신체 포인트를 효과적으로 구분할 수 있는지를 탐구합니다. 또한 입력 데이터에 경미한 변동이 있을 때 설명의 안정성도 평가합니다.

- **Performance Highlights**: 결과적으로 Grad-CAM이 RISv 메트릭에서 CAM을 뛰어넘는 성능을 보였으며, RISb 메트릭에서는 CAM이 우수한 성능을 보여 다소 상반된 결과를 보였습니다. 전체 앙상블 접근 방식은 개별 모델의 성과를 종합적으로 보여주었으며, XAI 메트릭 적용을 통해 CP 예측의 안정성을 확보했습니다.



### Measuring Human Contribution in AI-Assisted Content Generation (https://arxiv.org/abs/2408.14792)
- **What's New**: 본 연구는 인공지능(AI) 지원 콘텐츠 생성에서 인간의 기여도를 측정하는 새로운 프레임워크를 제시합니다. 이는 정보 이론에 기초하여 인간 입력과 AI 출력 간의 상호 정보를 이용해 비율적으로 인간의 기여를 정량화합니다.

- **Technical Details**: 제안된 측정 방법은 인간 입력과 AI 출력 간의 상호 정보 비율을 AI 출력의 자기 정보 총량과 비교하여 계산합니다. 다양한 도메인에서 수집된 AI 지원 생성 데이터에서의 실험 결과를 통해 이 방법의 유효성을 확인했습니다.

- **Performance Highlights**: Llama-3 및 Mixtral와 같은 최신 LLM 모델을 이용하여 결과를 도출하였으며, 인간의 기여도가 서로 다른 창작 도메인에서 측정 가능한 일관성을 나타냈습니다. 예를 들어, Llama-3를 사용해 생성한 뉴스 기사에서는 인간의 기여도가 평균 85.37%에서 30.83%까지 변하는 것을 보여주었습니다.




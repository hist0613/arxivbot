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




New uploads on arXiv(cs.CL)

### Comparable Corpora: Opportunities for New Research Directions (https://arxiv.org/abs/2501.14721)
Comments:
          Keynote in this https URL, workshop associated with Coling-2025

- **What's New**: 이번 논문은 기존의 연구 결과를 제시하는 것보다, 청중이 자발적으로 기여할 수 있는 기회를 제공하는 데 집중하고 있습니다. 특히, comparable corpora (CC) 사용의 새로운 방향을 제안하며, 다양한 언어의 chat bots 성공 사례를 통해 CC의 중요성을 강조합니다. 또한, CC 사용의 더 깊은 탐구와 다양한 문화적 관점의 필요성을 제기합니다.

- **Technical Details**: 논문에서는 CC의 역사적 배경을 검토한 후, 기계 번역, BLI (bilingual lexicon induction), WSD (word-sense disambiguation)와 같은 기술적 개념들을 설명합니다. 특히, 기존의 병렬 말뭉치에서 생기는 왜곡 문제와, CC가 이를 대체할 수 있는 가능성을 강조합니다. 또한, 신경망(neural network) 기법이 기계 번역에서 표준으로 자리잡게 된 과정을 이야기하며, 전통적인 Hidden Markov Models(HMMs)와의 비교를 통해 CC의 유용성을 설명합니다.

- **Performance Highlights**: CC를 활용하여 다양한 언어에서의 어휘 의미의 심층 분석 및 언어 간의 대조 자료를 생성할 수 있는 가능성을 제시합니다. 실질적으로, CC는 번역 뿐만 아니라, 다국어 설정에서도 효과적으로 사용될 수 있음을 강조합니다. 새로운 기준의 벤치마크를 통해 CC가 기존의 기계 번역 시스템과 비교하여 얼마나 효과적인지를 평가할 필요성도 언급됩니다.



### Do LLMs Provide Consistent Answers to Health-Related Questions across Languages? (https://arxiv.org/abs/2501.14719)
Comments:
          9 pages. Short paper appeared at 47th European Conference on Information Retrieval (ECIR 2025)

- **What's New**: 이번 연구는 다양한 언어에서 건강관련 질문에 대한 LLM(대형 언어 모델)의 응답 일관성을 분석하였습니다. 영어, 독일어, 터키어, 중국어의 다국적 맥락에서 LLM의 답변을 검토하고, 언어에 따라 다양한 질병 유형별로 분류된 건강 관련 질문 데이터를 확장했습니다. 이 연구의 주요 기여는 다국어 건강 질의 데이터셋과 이질적 언어 간의 비교를 가능하게 하는 새로운 프롬프트 기반 평가 작업 흐름을 도입한 것입니다.

- **Technical Details**: 연구에서는 NER(명명 개체 인식)을 사용하여 샘플을 질병별로 분류하고, 개편된 HealthFC 데이터셋을 사용해 LLM에 질병 분류에 따라 프롬프트를 제공하였습니다. 또한, 원래 영어와 독일어로 제공되던 데이터를 터키어와 중국어로 번역하고, 이를 통해 언어 간에 응답 일관성을 평가하기 위한 프롬프트 기반 평가 프레임워크가 개발되었습니다. 이러한 방법론은 LLM의 여러 언어에서의 응답을 심층적으로 분석할 수 있도록 하였습니다.

- **Performance Highlights**: 이 연구에서 발생한 주요 결과 중 하나는 LLM의 다양한 언어에서의 일관성 부족으로 인해 정확하지 않은 의료 정보를 퍼뜨릴 위험이 있다는 것입니다. 파싱(Parsing)된 응답 간의 비교를 통해 일관성의 측정을 시도했으며, Kappa Scores를 통해 응답의 일관성 정도를 평가하였습니다. 특히, 터키어(TR)와 중국어(ZH)에서 LLM과 인간 평가자 간의 상당한 일치도가 발견되었고, 독일어(DE)에서도 중간 정도의 일치가 관찰되었습니다.



### Towards Better Understanding Table Instruction Tuning: Decoupling the Effects from Data versus Models (https://arxiv.org/abs/2501.14717)
- **What's New**: 이 연구는 자연어 처리의 최근 발전을 통해 테이블 관련 작업을 위한 Large Language Models (LLMs)의 성능을 향상시키기 위해 instruction tuning을 활용하는 방법에 대해 논의하고 있습니다. 기존 연구들이 다양한 base model과 데이터셋을 사용하면서 실질적인 비교가 부족했던 문제를 해결하기 위해, Mistral, OLMo, Phi 모델을 동일한 공개 데이터셋으로 fine-tuning하여 새로운 최첨단 성능을 달성했습니다. 특히, HiTab이라는 테이블 질문-답변 데이터셋에서 SOTA (state-of-the-art) 기록을 세우고, 훈련 데이터와 base model의 개별 영향을 분석했습니다.

- **Technical Details**: 이 연구에서는 Mistral, OLMo, Phi 모델 계열의 같은 base 모델을 사용하여 기존 연구의 훈련 데이터에서 fine-tuning을 수행했습니다. 실험 결과는 우리 모델들이 기존 테이블 LLM보다 동등하거나 더 나은 성능을 발휘하는 것을 보여 주었습니다. 다양한 테이블 작업에 대한 체계적인 평가를 통해 테이블 질문-답변, 테이블-텍스트 생성, 테이블 사실 검증 등을 포함한 결과를 제시하며, 테이블 작업에 대한 specialization이 일반적인 능력에 미치는 영향을 분석하였습니다.

- **Performance Highlights**: 결과적으로, 연구자는 fine-tuning된 모델들이 기존 연구에서 보고된 벤치마크에서 동등하거나 우수한 성능을 발휘하며, HiTab 데이터셋에서 새로운 SOTA 결과를 세웠습니다. 또한, LLMs가 테이블 작업에서 갖는 전반적 성능을 평가하며 비목표 도메인 작업에서의 훈련 데이터의 효과를 분석하여, 훈련 데이터와 base 모델의 선택이 모델 성능에 미치는 상관관계를 입증하였습니다. 이 연구는 향후 효과적인 테이블 LLM을 구축하는 데 있어 모델 선택 및 데이터셋 구축에서 중요한 통찰력을 제공할 것을 기대합니다.



### FlexiGPT: Pruning and Extending Large Language Models with Low-Rank Weight Sharing (https://arxiv.org/abs/2501.14713)
Comments:
          Accepted to NAACL 2025 - Main Conference

- **What's New**: 최근 LLM의 사용이 늘어나면서 메모리 제약이 있는 기기에서도 효율적인 모델 배치를 위한 방법이 필요해졌습니다. 본 연구에서는 모델 블록을 선택적으로 제거하는 pruning 기법을 제안하며, 이를 통해 메모리가 한정된 환경에서 LLM의 성능을 유지할 수 있습니다. 특히, 기계 학습에서的重要한 블록의 중요도를 기반으로 하는 측정 방식을 도입하여 최적의 대체 전략을 구현합니다.

- **Technical Details**: 이 논문에서는 ShortGPT의 Block Influence (BI) 점수를 활용하여 모델 블록을 제거하고, 각각의 제거된 블록을 기존의 특정 블록에서 공유된 가중치로 교체하는 방법을 제안합니다. Low-Rank Adapters (LoRA)와 같은 기술을 도입하여 가중치 공유를 통한 파라미터 효율성을 극대화하고, 더 나은 특성 정규화(output feature normalization)를 통해 출력 안정성을 높입니다. 이러한 기법들은 대체 블록의 초기화와 그 후의 모델 적응을 촉진하는 역할을 합니다.

- **Performance Highlights**: Empirical evaluations of this method, named FlexiGPT, show significant improvements in performance, achieving state-of-the-art results on 5 out of 6 benchmarks at a 30% compression rate and on all benchmarks at a 40% rate. 이 연구는 또한 TinyLLaMA와 같은 소형 모델을 반복하여 효율적으로 확장할 수 있는 가능성을 보여주며, 0.3% 토큰만으로도 6/6 벤치마크에서 놀라운 성능 상승을 달성합니다. 이러한 결과를 통해 LLM을 메모리 제약이 있는 기기에서도 효과적으로 사용할 수 있도록 하는 새로운 접근 방식을 제안합니다.



### NLP-based assessment of prescription appropriateness from Italian referrals (https://arxiv.org/abs/2501.14701)
- **What's New**: 이 연구는 이탈리아의 처방 적합성을 평가하기 위해 자연어 처리(Natural Language Processing, NLP) 파이프라인을 제안합니다. 첫 번째로, 각 처방의 이유를 종합적으로 요약하고 적합성을 정량화하는 접근 방식을 개발하였습니다. 특히 이 연구는 2019년부터 2021년까지 로마냐 지역의 496,971개의 처방 데이터를 분석하는 케이스 스터디를 통해 검증을 수행하였습니다.

- **Technical Details**: 이 파이프라인은 변환기(transformer) 기반 모델의 임베딩을 활용하여 처방 텍스트를 클러스터링하고, 이러한 클러스터를 적절한 라벨에 매핑합니다. BERT 모델을 통해 생성된 임베딩을 활용한 클러스터링 과정을 통해 기존 가이드라인과의 일치를 평가합니다. 1,000개의 처방을 수동으로 주석하여 결과를 검증하였으며, 이로써 파이프라인의 정확성을 확보하고자 하였습니다.

- **Performance Highlights**: 제안된 파이프라인은 고도의 성능을 보였으며, 수동으로 주석된 데이터 셋에서 처방 이유의 정밀도(Precision)는 92.43%, 재현율(Recall)은 83.28% 그리고 적합성에서는 93.58%의 정밀도와 91.52%의 재현율을 기록하였습니다. 전체 데이터셋 분석 결과, 34.32%의 처방이 적합하며, 34.07%는 부적합하다고 평가되었습니다. 이러한 발견은 로마냐 지역의 건강 정책 강화와 부적절한 처방 감소에 기여할 수 있습니다.



### Rethinking Table Instruction Tuning (https://arxiv.org/abs/2501.14693)
- **What's New**: 최근 테이블 이해(table understanding) 분야에서는 테이블 관련 작업을 위해 대규모 언어 모델(LLM)을 instruction-tuning하는 발전이 있었습니다. 그러나 기존 연구에서는 하이퍼파라미터 선택의 영향을 간과하고, 도메인 외(out-of-domain) 테이블 이해 능력에 대한 종합적인 평가가 부족했습니다. 본 연구에서는 이러한 능력을 평가하고, 하이퍼파라미터가 테이블 LLM의 성능에 미치는 영향을 체계적으로 분석하여, 새로운 모델 TAMA를 도입했습니다.

- **Technical Details**: 기존의 테이블 LLM이 도메인 외 테이블 이해 능력 및 일반적인 능력에서 큰 성능 하락을 보이는 것을 발견했습니다. 특히, 학습률(learning rate)과 훈련 인스턴스(instance)의 수와 같은 하이퍼파라미터 선택이 모델의 성능에 미치는 효과는 중요한 요소입니다. 본 연구의 분석에 따르면, 작은 학습률이 테이블 이해 능력을 향상시키면서도 일반 능력을 유지할 수 있음을 확인했습니다.

- **Performance Highlights**: TAMA 모델은 LLaMA 3.1 8B Instruct에서 instruction-tuning되어, 테이블 관련 작업에서 GPT-3.5 및 GPT-4의 성능에 필적하거나 이를 초과했습니다. 또한, TAMA는 강력한 도메인 외 테이블 이해 및 일반 능력을 유지하며, 데이터 주석 비용을 절감할 잠재력을 가지고 있음을 강조하였습니다.



### State Space Models for Extractive Summarization in Low Resource Scenarios (https://arxiv.org/abs/2501.14673)
- **What's New**: 이번 논문에서는 MPoincareSum 방법을 제안합니다. 이는 최근의 연구 동향을 반영하여 저자원이 부족한 환경에서의 추출적 요약(extractive summarization) 성과를 향상시키기 위해 개발되었습니다. 기존 연구들에 비해 문장 관련성을 예측하고 요약을 최적화하는 데 중점을 두었습니다.

- **Technical Details**: MPoincareSum 방법은 Mamba 상태 공간 모델(state space model)을 활용하여 리뷰와 문장의 의미(semantics)를 생성합니다. 이후 Poincare 압축(Poincare compression)을 통해 가장 의미 있는 특징(features)을 선택하고, 선형 레이어(linear layer)를 적용하여 문장의 관련성을 예측합니다. 최종적으로, 관련 문장을 패러프레이즈(paraphrase)하여 요약을 생성합니다.

- **Performance Highlights**: 아마존 리뷰 데이터셋을 사용하여 MPoincareSum의 효과를 평가하였습니다. 실험 결과, 이 방법이 기존의 여러 접근 방식보다 성능이 우수하다는 것을 ROUGE 점수(ROUGE scores)를 통해 입증하였습니다. MPoincareSum은 저자원 환경에서도 뛰어난 요약 결과를 제공합니다.



### Investigating the (De)Composition Capabilities of Large Language Models in Natural-to-Formal Language Conversion (https://arxiv.org/abs/2501.14649)
Comments:
          Accepted at NAACL 2025 main conference

- **What's New**: 이 연구는 자연어에서 형식 언어로의 변환(N2F) 과정에서 대형 언어 모델(LLMs)의 기본 능력을 평가하기 위한 새로운 DEDC 프레임워크를 제안합니다. 이 프레임워크는 샘플 및 과제를 반자동 방식으로 생성하여 LLM의 분해(decomposition)와 조합(composition) 능력을 독립적으로 평가할 수 있게 합니다. 연구 결과에 따르면, LLM은 분해와 조합 모두에서 부족함을 보이며, 이로 인해 자연어 이해에 대한 결함과 기호 체계의 학습 및 사용에서 여러 종류의 오류가 발생합니다.

- **Technical Details**: DEDC 프레임워크는 표 형식 추론(Tabular Reasoning) 맥락에서 10개의 기본 원시(primitives)를 가진 형식 언어를 정의하고, 이를 토대로 방향 비순환 그래프(directed acyclic graph)로 표현되는 표현식의 조합 구조를 분석합니다. 각 표현식은 함수와 출력 타입에 따라 자연어 질문으로 변환되며, 그 결과로 323개의 샘플이 생성됩니다. 각 LLM의 작업은 세 개의 데모 샘플을 참고하여 주어진 질문을 처리하여 표현식으로 변환하는 것입니다.

- **Performance Highlights**: 연구에서는 LLM이 N2F 작업에서 수행하는 정확도를 평가하며, 분해 및 조합 능력의 영향을 효과적으로 분리하여 측정합니다. LLM의 성능은 다양한 원시를 포함한 샘플을 기반으로 하여 평가되며, 조합 능력만 필요한 경우에도 성능 손실이 발생하는 정보를 제공합니다. 이 연구는 LLM 개선을 위한 결함 분석과 기초 능력에 대한 통찰을 제공하며, 기존의 언어 모델이 겪는 문제점을 다루고 있습니다.



### Funzac at CoMeDi Shared Task: Modeling Annotator Disagreement from Word-In-Context Perspectives (https://arxiv.org/abs/2501.14617)
Comments:
          Accepted to CoMeDi Shared Task at COLING 2025

- **What's New**: 이번 연구는 Word-in-Context (WiC) 작업에서 주석자 간 불일치를 평가하는데 중점을 두었습니다. 기존의 연구들은 단일 문장을 분석하여 주석자의 속성과 불일치를 모델링했으나, 본 연구는 WiC 개념을 도입하여 문장 수준 의미 표현과 주석자 판단 변동의 간극을 줄이고자 했습니다. 우리는 세 가지 방법을 개발했으며, 이들 방법은 enriched feature와 task-specific 특성을 포함하여 성과를 개선했습니다.

- **Technical Details**: 우리는 XLM-RoBERTa 모델을 기반으로 상대 단어의 문맥 내 임베딩을 사용하여 주석 결과를 예측하는 여러 방법을 개발했습니다. 각 방법은 ordinality를 고려한 분류와 회귀 작업을 포함하며, Adapter 블록을 통해 문맥 임베딩의 task-specific 변환을 적용했습니다. 우리의 시스템은 OGWiC와 DisWiC 작업에서 서로 다른 복잡성을 지닌 분류기를 사용하여 불일치 예측을 개선하였습니다.

- **Performance Highlights**: 이 연구의 결과는 OGWiC 작업에서 가장 좋은 성능을 보인 다른 시스템과 비교했을 때, 우리의 방법이 부족했음을 나타냅니다. 그러나 DisWiC 작업에서는 우리의 접근 방식이 공식 평가 결과와 경쟁력을 유지했습니다. 또한, 우리는 시스템을 단순한 linear 모델과 dropout을 사용하여 구성했으며, 각 서브태스크에 대해 공개적으로 구현 결과를 제공합니다.



### Idiom Detection in Sorani Kurdish Texts (https://arxiv.org/abs/2501.14528)
Comments:
          22 pages, 8 figures, 7 tables

- **What's New**: 이 연구에서는 Sorani Kurdish에서의 관용구 탐지를 다루며, 이를 텍스트 분류 작업으로 접근했습니다. 연구진은 다양한 맥락에서 101개의 Sorani Kurdish 관용구를 포함한 10,580개의 문장으로 구성된 데이터셋을 개발했습니다. 기존의 연구와 달리, 세 가지 딥러닝 모델과 함께 관용구 탐지에 필요한 데이터를 체계적으로 제공하고자 했습니다.

- **Technical Details**: 관용구 탐지의 비효율성을 극복하기 위해 KuBERT 기반의 transformer 순서 분류, 재귀 합성곱 신경망(RCNN), 주의 메커니즘을 갖춘 BiLSTM 모델을 개발했습니다. BERT(Transformer 기반의 Bidirectional Encoder Representations from Transformers) 모델을 사용하여 두 방향에서 맥락을 포착하고, 구어체 및 문맥에 맞는 관용구를 인식할 수 있도록 세밀하게 조정했습니다. 이 연구에서는 특히 언어의 복잡한 형태를 처리하기 위해 KuBERT 모델을 도입했습니다.

- **Performance Highlights**: 세 모델의 평가 결과, 세밀하게 조정된 BERT 모델이 99%에 가까운 정확도로 다른 모델들보다 뛰어난 성능을 나타냈습니다. RCNN은 96.5%의 정확도를 기록했으며, BiLSTM은 80%의 정확도를 보였습니다. 이러한 결과는 쿠르드어와 같은 저자원 언어에서 Transformer 기반 아키텍처의 효과성을 강조할 수 있게 해줍니다.



### WanJuanSiLu: A High-Quality Open-Source Webtext Dataset for Low-Resource Languages (https://arxiv.org/abs/2501.14506)
- **What's New**: 이번 연구에서는 저자원 언어를 위한 고품질 학습 데이터 세트인 WanJuanSiLu를 소개합니다. 저자원 언어를 대상으로 한 다국어 모델의 연구 및 개발을 촉진하기 위해 체계적인 데이터 처리 프레임워크를 개발하였습니다. 이 프레임워크는 데이터 추출, 데이터 클리닝, 내용 중복 제거, 보안 필터링, 품질 평가 및 주제 분류와 같은 주요 단계를 포함합니다.

- **Technical Details**: WanJuanSiLu는 5개 언어, 3억 5천만 개의 문서, 1.2TB의 데이터, 3천억 개 이상의 토큰을 포함하는 대규모 다국어 텍스트 말뭉치입니다. 데이터 분류 시스템은 7개의 주요 범주와 34개의 하위 범주로 구성되어 있으며, 언어의 지리적 특성에 따라 다양한 내용을 포함합니다. 이 데이터 세트는 다국어 모델 학습을 위한 신뢰할 수 있는 데이터 지원을 제공하기 위해 향상된 데이터 라벨링 및 보안 기능을 갖추고 있습니다.

- **Performance Highlights**: 파일럿 데이터에서 데이터 품질을 평가한 결과, 품질 좋은 데이터가 44.53%에 불과하며, 관련성 문제(33.49%)가 가장 두드러진 문제로 나타났습니다. 퍼플렉시티(PPL) 기준을 사용하여 처음 필터링을 통과한 데이터를 정밀하게 평가할 수 있는 다국어-BERT 모델을 개발하였으며, 이를 통해 한국어 데이터의 경우 데이터의 양을 37% 줄이는 데 성공했습니다. 전체적으로 이 연구는 저자원 언어의 효율성을 높이는 데 기여하고 있습니다.



### Evaluating and Improving Graph to Text Generation with Large Language Models (https://arxiv.org/abs/2501.14497)
Comments:
          NAACL 2025

- **What's New**: 이번 연구에서는 그래프-텍스트 생성 작업에 대한 최신 대형 언어 모델(LLMs)의 가능성을 평가하였습니다. 특히, LLM이 그래프를 자연어로 표현하는 능력을 향상시키기 위한 새로운 데이터셋인 PlanGTG를 소개하며, reorder와 attribution이라는 두 가지 하위 작업을 포함시켰습니다. 연구 결과, LLM의 성능을 개선하기 위한 최적의 프롬프트 전략을 탐색하고, 그래프의 복잡성이 LLM의 계획 수립에 미치는 영향을 강조하였습니다.

- **Technical Details**: PlanGTG 데이터셋은 약 30,000개 데이터 쌍으로 구성되어 있으며, LLM을 세밀하게 조정하여 생성 텍스트의 품질을 높이는 데 기여합니다. 연구진은 각 그래프의 트리플 수와 다이아미터에 따라 LLM이 성능에 힘들어하는 경향을 발견하였습니다. 연구의 주요 목표는 LLM의 구조를 그래프 데이터와 결합하여 보다 효과적으로 텍스트를 생성하는 것입니다.

- **Performance Highlights**: 자동 평가 및 인간 평가를 통해 PlanGTG 데이터셋으로 조정된 LLM이 Kaggle 등의 기존 데이터셋에서 훈련된 모델보다 우수한 성능을 보임을 확인하였습니다. 특히, LLM이 KG 트리플의 순서를 조정하고 생성된 텍스트에서 순차적 번호를 정확히 부여하는 능력이 향상되었습니다. 이러한 결과는 LLM의 명확성과 성능을 높이는 기회로 작용하여, 그래프-텍스트 생성 분야의 새로운 연구 방향을 제시합니다.



### RealCritic: Towards Effectiveness-Driven Evaluation of Language Model Critiques (https://arxiv.org/abs/2501.14492)
- **What's New**: 본 연구는 기존의 평가 방식을 개선하기 위해 LLM의 비판 능력을 평가하는 새로운 기준을 도입합니다. 기존 벤치마크가 오픈 루프(open-loop) 방식이었다면, 우리는 피드백을 통해 수정되는 폐쇄 루프(closed-loop) 방식을 채택하여 LLM의 비판 품질을 더 신뢰성 있게 평가합니다. 이 새로운 기준은 자기 비판(self-critique), 상호 비판(cross-critique), 반복 비판(iterative critique) 기능을 포함하여 고급 추론 모델의 능력을 구분하는 데 필수적입니다.

- **Technical Details**: 평가 프레임워크는 여러 주요 개념으로 구성되어 있습니다. 폐쇄 루프 방법론을 통해 LLM의 비판 능력을 테스트하게 되며, 비판이 적용된 후 생성된 솔루션의 정확도를 통해 비판의 품질을 평가합니다. 또한 두 가지 주요 축을 통해 자기 비판과 상호 비판의 차이를 구분하고, 단일 라운드뿐만 아니라 반복적인 비판 과정을 고려하여 모델의 장기적인 추론 능력을 평가합니다.

- **Performance Highlights**: 실험 결과, 고전적인 LLM은 고급 추론 기반 모델인 o1-mini에 비해 모든 비판 시나리오에서 눈에 띄게 뒤처지는 것으로 나타났습니다. 또한 자기 비판 및 반복 비판 설정에서 고전 LLM이 기준 성능에 비해 더 낮은 성능을 보이는 경우도 발견되었습니다. 이러한 결과는 우리의 벤치마크가 LLM의 개선 방향을 제시할 수 있는 중요한 자원이 될 것이라는 점을 강조합니다.



### Analyzing the Effect of Linguistic Similarity on Cross-Lingual Transfer: Tasks and Experimental Setups Matter (https://arxiv.org/abs/2501.14491)
- **What's New**: 이 연구는 266개의 다양한 언어에서의 크로스링구얼(다국어) 전이(Cross-lingual transfer)를 분석하고, POS 태깅(POS tagging), 의존 구문 분석(dependency parsing), 주제 분류(topic classification)라는 세 가지 NLP 작업에 대한 결과를 제시합니다. 이전의 연구에서 제기된 언어적 유사성이 전이 성능에 미치는 영향이 다양한 요인에 따라 다르다는 점을 발견했습니다. 언어 모델과 데이터셋의 구조나 특성에 따라 가장 적합한 언어를 선택하는 방법에 대한 가이드를 제공합니다.

- **Technical Details**: 연구는 266개 언어와 33개 언어계통을 포함하여, 각각 POS 태깅, 의존 구문 분석, 주제 분류와 같은 세 가지 작업을 조사합니다. 연구에서는 미리 훈련된 모델을 사용하여 특정(source) 언어에서 작업을 학습한 후 타겟(target) 언어에서 추가 훈련 없이 평가하는 제로샷(zero-shot) 접근 방식을 채택했습니다. 다양한 유사성 측정치를 사용해 각각의 작업에 대한 최적의 성능 예측 방법을 구체적으로 설명합니다.

- **Performance Highlights**: 결과적으로, 서로 다른 유사성 측정치는 작업과 입력 표현에 따라 중요도가 달라지며, 예를 들어 구문적 유사성은 POS 태깅과 의존 구문 분석에서 가장 예측력이 높습니다. 반면, 삼중그램(trigram)의 중첩은 주제 분류에서 중요한 예측 지표로 나타났습니다. 크로스링구얼 전이 성능에 대한 정보가 없을 경우, 비슷한 작업을 기반으로 전이 언어를 선택하는 것이 비교적 안전한 선택이라는 것을 발견했습니다.



### Understanding and Mitigating Gender Bias in LLMs via Interpretable Neuron Editing (https://arxiv.org/abs/2501.14457)
Comments:
          preprint

- **What's New**: 이번 연구에서는 기존 대형 언어 모델(LLM)에서 발생하는 성 편향(gender bias) 문제를 해결하기 위한 새로운 데이터셋인 CommonWords를 제안합니다. 이 데이터셋은 성 편향을 체계적으로 평가할 수 있는 5가지 범주의 일반 단어로 구성되어 있습니다. 연구자들은 LLM에서 성 편향이 광범위하게 나타나며, 특정 뉴런 회로가 이 행동에 책임이 있다는 사실을 밝혀냈습니다.

- **Technical Details**: LLM에서 성 편향은 표층 레이어의 'gender neurons'와 일반적인 'general neurons'의 상호 작용으로 인해 발생합니다. 연구팀은 이러한 뉴런에 대한 이해를 바탕으로, 성 편향을 선택적으로 목표로 하는 해석 가능한 뉴런 편집(interpretable neuron editing) 방법을 제안했습니다. 이 방법은 logro-based 및 causal-based 접근 방식을 결합하여, 성 편향을 효과적으로 줄이면서 모델의 원래 기능을 유지합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존의 미세 조정(fine-tuning) 및 편집 방법들보다 성 편향을 더욱 효과적으로 줄이는 동시에 모델의 원래 능력을 유지하는 것으로 나타났습니다. 이러한 접근은 LLM의 성 편향 기제를 깊이 이해하고, 향후 연구에 유용한 CommonWords 데이터셋과 편리한 솔루션을 제공하는 데 기여합니다.



### Domaino1s: Guiding LLM Reasoning for Explainable Answers in High-Stakes Domains (https://arxiv.org/abs/2501.14431)
- **What's New**: 이 연구에서는 Domaino1s라는 새로운 모델을 소개하여 고위험 도메인 작업에서 대형 언어 모델의 (LLMs) 사고(procesing) 기능을 크게 향상시켰습니다. 이 모델은 감독된 미세 조정(supervised fine-tuning)과 트리 탐색(tree search)을 통해 설명가능한 답변을 생성하는 데 중점을 두었습니다. CoT-stock-2k 및 CoT-legal-2k 데이터셋을 구성하여 도메인 특화 사고 단계를 활성화하는 모델을 미세 조정하였으며, 설명 및 결정의 신뢰성을 높이는 새로운 평가 지표인 PROOF-Score를 제안했습니다.

- **Technical Details**: Domaino1s는 두 가지 모델 변형인 Domaino1s-finance와 Domaino1s-legal로 구성되어 있습니다. 이 모델은 GPT-4o를 사용하여 CoT 데이터를 생성하고, 26개의 특수 토큰을 활용하여 사고 과정의 각 단계를 명확히 구분할 수 있도록 데이터셋을 구성합니다. 새로운 Selective Tree Exploration 방식을 도입하여 최적의 사고 경로를 탐색하고, 각 단계의 평균 perplexity를 사용하여 새로운 경로를 탐색할지를 결정합니다.

- **Performance Highlights**: Domaino1s의 실험 결과는 주식 투자 추천 및 법적 질문 응답 과제에서 두드러진 성능을 나타내며, 설명 가능성과 사고의 정확성을 동시에 향상시켰습니다. 기존의 정확성 기준 외에도, DOMAINO1S는 모델의 설명 가능성을 평가하는 새로운 시각을 제시하며, 이는 결정적인 법적 또는 윤리적 위험을 줄이는 데 기여할 수 있습니다. 이 연구는 고위험 도메인 작업을 해결하기 위한 LLM의 실행 가능성과 효율성을 입증했습니다.



### DRESSing Up LLM: Efficient Stylized Question-Answering via Style Subspace Editing (https://arxiv.org/abs/2501.14371)
Comments:
          ICLR 2025 Accepted

- **What's New**: DRESS는 LLM의 반응을 스타일화하기 위한 새롭고 효과적인 접근법으로, 표현 편집을 통해 스타일 관련 하위공간(style subspace)을 분리합니다. 기존의 방법인 프롬프트(prompting)나 미세 조정(fine-tuning)은 복잡한 스타일 적응을 위해 충분하지 않거나 계산적으로 비용이 많이 듭니다. DRESS는 LLM의 오버 파라미터화된 특성을 활용하여 표현 공간에서 의미(original semantics)에 최소한의 영향을 미치면서 스타일을 조정할 수 있는 방법을 제안합니다.

- **Technical Details**: DRESS는 스타일 관련 하위공간을 단계적으로 식별하고 그들 안에서 의미적으로 격리된 스타일 조정을 수행하는 세 가지 전략을 포함합니다. 첫 번째는 주의 헤드 여과(attention head filtering)로, 스타일과 관련된 주의 헤드를 식별합니다. 두 번째는 스타일 하위공간 여과로(style subspace filtering), 선택된 주의 헤드 내에서 스타일과 관련없는 컴포넌트를 제거합니다. 마지막으로 각 하위공간 기초와 생성된 토큰에 대해 적응형 편집 강도(adaptive editing strength)를 적용하여 유연성을 제공합니다.

- **Performance Highlights**: DRESS는 스타일 강도(style intensity), 의미 보존(semantic preservation), 유창성과 같은 객관적인 평가 지표를 포함하는 두 개의 스타일화된 QA 벤치마크 데이터셋을 사용하여 그 효과를 입증합니다. 실험 결과, DRESS는 SFT, 프롬프트 및 다른 표현 편집 방법들과 비교하여 성능이 크게 향상된 것으로 나타났습니다. DRESS는 또한 대화형 에이전트를 개발하는 데 특히 유용하여, 경량화되고 교육이 필요 없는 솔루션을 제공합니다.



### Clear Minds Think Alike: What Makes LLM Fine-tuning Robust? A Study of Token Perplexity (https://arxiv.org/abs/2501.14315)
- **What's New**: 이번 연구에서는 LLM(대규모 언어 모델) 생성 데이터를 사용한 파인튜닝이 타겟 작업의 성과를 향상시킬 뿐만 아니라, 실제 데이터로 파인튜닝할 때보다 OOD(out-of-domain) 성능 저하를 줄이는 데 도움을 준다는 사실을 발견했습니다. 이는 LLM이 생성한 데이터에서 고 퍼플렉시티(high perplexity) 토큰의 비율이 낮아지는 데 기인합니다.

- **Technical Details**: 고 퍼플렉시티 토큰 마스킹(masking)을 통해 실제 데이터에서도 OOD 성능 보존을 달성할 수 있음을 보여 주었습니다. 다양한 도메인에서의 데이터 시퀀스를 분석하는 것을 통해 이러한 OOD 내구성의 강화를 문서화하였습니다. 또한 Gemma2-2B, Mistral-7B, Llama3-8B 모델 등 여러 아키텍처를 통해 포괄적인 실험을 수행하였습니다.

- **Performance Highlights**: 이번 연구의 결과들은 LLM 생성 훈련 데이터가 뛰어난 OOD 내구성을 부여함을 기계적인 설명을 통해 첫 번째로 제시하였습니다. 이로 인해 더 강력한 파인튜닝 전략을 개발하는 데 중요한 통찰력을 제공하고 있습니다.



### Examining Alignment of Large Language Models through Representative Heuristics: The Case of Political Stereotypes (https://arxiv.org/abs/2501.14294)
Comments:
          ICLR 2025

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)을 인간의 의도와 가치에 맞추는 데 있어 정치적 성향의 중요성을 강조하고 있습니다. 이전 연구들이 LLM의 정치적 경향성을 보여주었으나, LLM이 실제 위치에서 벗어나는 정도와 그 조건은 충분히 조사되지 않았습니다. 본 연구는 LLM의 정치적 문제에 대한 편향을 수량화하고, 이로 인해 발생하는 편차의 조건을 규명하고자 하였습니다.

- **Technical Details**: 이 연구는 인지 과학의 자료를 바탕으로 대표성 휴리스틱(representativeness heuristics)을 통해 LLM의 응답을 분석합니다. 대표성 휴리스틱은 특정 집단의 특징을 과대평가하는 인지 현상으로, 정치적 편향이 특정 정당의 입장을 과장하는 방식으로 나타나는지 실험을 통해 조사하였습니다. 연구 결과, LLM은 특정 정치 정당의 입장을 모방할 수 있으나, 인간 응답자에 비해 이러한 입장을 더 과장하게 나타냅니다.

- **Performance Highlights**: LLM의 응답은 일반적으로 실제의 진실성(kernel-of-truth)을 반영하였으나, 인간 응답자보다 더 극단적으로 나타났습니다. 이 연구는 LLM이 정치적 편향에 취약할 수 있는 가능성을 확인하였으며, 대표성을 줄이기 위한 프롬프트 기반의 방법들이 효과적임을 보여주었습니다. 궁극적으로, LLM에 대한 새로운 관점을 제시하고, 그들의 편향을 완화하기 위한 전략을 제안하는 것을 목표로 했습니다.



### A Comprehensive Framework for Semantic Similarity Detection Using Transformer Architectures and Enhanced Ensemble Techniques (https://arxiv.org/abs/2501.14288)
- **What's New**: 이 논문은 AI 생성 텍스트, 특히 짧은 맥락 문서에서의 탐지 문제를 해결하기 위해 새로운 teacher-student 모델을 제안합니다. teacher 모델은 DeBERTa-v3-large와 Mamba-790m을 결합하여 도메인 특화 미세 조정을 통해 의미론적 지식을 학습합니다. student 모델은 짧은 맥락 텍스트를 더 효율적으로 처리하며, 손실 함수로 Mean Squared Error (MSE)를 사용하여 학습을 안내받습니다.

- **Technical Details**: 이 모델은 d학습에 있어 도메인 적응과 데이터 증강을 통합하고, 다양한 텍스트 변경 방법(예: 오타 수정, 오류 삽입)을 통해 강건성을 강화할 수 있습니다. 특히, teacher 모델은 세부적인 의미적 특성을 포착하고 AI 생성 콘텐츠를 보다 잘 탐지하도록 도와줍니다. 최종적으로, 이 시스템은 짧은 텍스트 문서 분류에서 높은 성능을 발휘하며, 다른 모델들이 어려움을 겪는 영역에서 효과적으로 작동합니다.

- **Performance Highlights**: 실험 결과는 이 접근 방식이 기존의 기준 방법들보다 더 우수한 성능을 보여준다고 밝혀졌습니다. 특히, 이 모델은 짧은 맥락 문서에서도 AI 생성 텍스트 탐지 및 텍스트 분류 작업에 유용성을 증명합니다. 이러한 결과는 실시간 AI 생성 텍스트 탐지와 같은 다양한 응용 프로그램에 적용할 수 있습니다.



### Leveraging Online Olympiad-Level Math Problems for LLMs Training and Contamination-Resistant Evaluation (https://arxiv.org/abs/2501.14275)
- **What's New**: 이번 논문에서는 올림피아드(olympiad) 수준의 수학 문제 해결을 위한 고품질 데이터셋 구축과 LLM(대형 언어 모델)의 성능 향상을 위한 자동화된 파이프라인을 소개합니다. 특히, AoPS(Art of Problem Solving) 포럼의 자원을 활용하여 60만 개 이상의 QA(pair) 쌍을 포함하는 AoPS-Instruct 데이터셋을 개발했습니다. 이 외에도, 새로운 평가 세트인 LiveAoPSBench를 통해 LLM의 수학 추론 능력을 평가할 수 있는 오염 저항적인 벤치마크를 구축했습니다.

- **Technical Details**: 이 연구에서는 LLM의 미세 조정을 위한 AoPS-Instruct 데이터셋과 더불어 LiveAoPSBench라는 자동화된 평가 세트를 소개합니다. AoPS-Instruct는 AoPS 포럼의 게시글에서 QA 쌍을 추출하여 생성되었으며, 이는 고급 수학 문제에 적합한 대규모 데이터셋입니다. 자동화된 파이프라인을 통해 지속적으로 최신 문제를 포함한 평가 세트를 운영함으로써 오염 가능성을 줄였습니다.

- **Performance Highlights**: 실험 결과, LLM을 AoPS-Instruct로 미세 조정한 후, OlympiadBench, Omni-Math, LiveAoPSBench와 같은 다양한 벤치마크에서 성능이 향상되었습니다. 또한, 시간 경과에 따른 성능 저하가 관찰되어, 오래된 문제에 대한 성공이 실제 수학적 추론 능력이 아니라 사전 훈련 피드백에 기인할 수 있음을 시사합니다. 따라서 본 연구는 LLM의 수학적 추론 능력에 대한 통찰력을 제공하며, 이를 통해 데이터셋의 생성과 유지 관리 방법론의 중요성을 강조하고 있습니다.



### Siren: A Learning-Based Multi-Turn Attack Framework for Simulating Real-World Human Jailbreak Behaviors (https://arxiv.org/abs/2501.14250)
- **What's New**: 이 논문에서는 LLM(대형 언어 모델)의 안전성과 신뢰성 문제를 해결하기 위해 Siren이라는 학습 기반의 멀티 턴 공격 프레임워크를 제안합니다. 기존의 단일 턴 공격 방법과는 달리, Siren은 공격자가 여러 턴을 통해 동적으로 공격 쿼리를 생성하고 롤 플레이를 통해 실제 인간의 jailbreak 행동을 시뮬레이션합니다. 이를 통해 Siren은 다른 멀티 턴 모델에 비해 공격 성공률을 획기적으로 향상시킵니다.

- **Technical Details**: Siren은 세 가지 단계로 구성됩니다: (1) Turn-Level LLM 피드백을 활용한 훈련 세트 구축, (2) 감독된 세부 조정(Supervised Fine-Tuning, SFT) 및 직접 선호 최적화(Direct Preference Optimization, DPO)를 통한 사후 훈련 공격자, (3) 공격자와 목표 LLM 간의 상호작용. 이 프레임워크는 학습 기반 접근 방식을 사용하여 쿼리 생성을 자동화하며, 공격의 복잡성에 효과적으로 적응합니다.

- **Performance Highlights**: 실험 결과에 따르면, Siren은 LLaMA-3-8B를 사용하여 Gemini-1.5-Pro 모델에 대한 공격 성공률(ASR)을 90% 달성하였으며, Mistral-7B를 이용해 GPT-4o에 대해서는 70%의 성공률을 기록했습니다. 또한, Siren은 7B 규모 모델로써 멀티 턴 공격에서 강력한 효과를 발휘하며, 효과적인 공격과 함께 인과적 의미 연관성을 유지하는 전략을 사용하여 적은 턴 수를 요구합니다.



### Multi-agent KTO: Reinforcing Strategic Interactions of Large Language Model in Language Gam (https://arxiv.org/abs/2501.14225)
Comments:
          Preprint. Code and data will be available at this https URL

- **What's New**: 이번 연구는 인공지능 일반 지능(AGI)의 실현을 위해 AI 에이전트가 전략적 의사결정을 내리고 유의미한 소통을 할 수 있도록 하는 새로운 접근법을 제안합니다. 특히, 비트겐슈타인의 언어 게임 이론에 영감을 받아, 전통적인 다단계 프로세스를 탈피하여 함수적 상호작용을 통한 학습을 강조합니다. 연구에서는 사회적 추론 게임인 'Werewolf'를 통해 새로운 다중 에이전트 최적화 모델인 MaKTO를 개발하여, AI의 의사결정 및 자연어 생성 능력을 강화하였습니다.

- **Technical Details**: MaKTO의 핵심은 세 가지 주요 혁신 요소에 있습니다. 첫째, 행동 클로닝(Behavior Cloning) 기법을 통해 게임의 전문 용어 및 전략 자료를 기반으로 모델의 학습을 지원합니다. 둘째, 여러 모델 간의 상호작용을 통해 전략 고정화를 방지하고 일반화 능력을 향상시키며, 셋째, 단계적 피드백을 통해 더 정교한 최적화를 위한 선택 방법을 도입하여 승패 이상의 복잡한 게임 작업에 대응하도록 설계되었습니다.

- **Performance Highlights**: 구현된 MaKTO는 9인 플레이어가 참가하는 'Seer-Witch-Guard' 게임에서 61%의 평균 승률을 기록하며, 기존의 GPT-4o 및 RL 기반 모델보다 23.0% 및 10.9% 향상된 성능을 보였습니다. 또한, 인간 전문가와의 대결에서 60%의 승률을 달성하여 인간과 거의 구별할 수 없는 수준의 대화 스타일을 나타내었으며, 판별 테스트에서 48.9% 정확도로 AI 식별이 어려웠습니다. 이러한 결과는 MaKTO가 보다 폭넓은 사회적 추론 게임 및 다자간 협상 시나리오로 확장될 가능성을 내포하고 있습니다.



### Test-Time Code-Switching for Cross-lingual Aspect Sentiment Triplet Extraction (https://arxiv.org/abs/2501.14144)
- **What's New**: 이번 연구에서는 Aspect Sentiment Triplet Extraction (ASTE) 작업에 있어 교차 언어 전이(cross-lingual transfer)의 적용을 탐구하였습니다. 이번 연구는 Test-Time Code-SWitching (TT-CSW) 프레임워크를 도입하여, 이중 언어 훈련 단계와 단일 언어 테스트 예측 간의 간극을 메우고자 하였습니다. 이는 기존의 코드 스위칭 방법에서 나타나는 용어 경계 감지(term boundary detection)와 사전(dictionary) 문제를 개선하기 위한 새로운 접근입니다.

- **Technical Details**: 우리의 프레임워크는 이중 언어 코드 스위칭 훈련 데이터를 기반으로 하는 생성 모델(generative model)을 개발하여 이중 언어 입력에 대해 이중 언어 ASTE triplet을 생성할 수 있도록 합니다. 테스트 단계에서는 정렬 기반(code-switching technique) 코드 스위칭 기법을 활용하여 테스트 시간에서의 향상을 꾀하였습니다. 이 연구는 여러 교차 언어 ASTE 데이터 세트를 바탕으로 하여 방대한 실험을 통해 제안된 방법의 효과성을 검증하였습니다.

- **Performance Highlights**: 총 4개의 서로 다른 데이터 세트에서 평균 3.7%의 가중 평균 F1 향상을 달성하였으며, ChatGPT와 GPT-4를 사용하여 벤치마크를 설정하였습니다. 그 결과, 우리의 TT-CSW 프레임워크로 미세 조정된 소형 생성 모델이 ChatGPT와 GPT-4를 각각 14.2%와 5.0% 초과하는 성능을 보였습니다.



### Autonomous Structural Memory Manipulation for Large Language Models Using Hierarchical Embedding Augmentation (https://arxiv.org/abs/2501.14119)
- **What's New**: 최근 대형 언어 모델(LLMs) 분야에서는 다계층 의미 구조를 통해 토큰의 표현을 재정의하는 계층적 임베딩 증강이 도입되었습니다. 이는 복잡한 언어 입력에 대해 더 나은 적응성을 제공합니다. 또한, 자율 구조 기억 조작은 중요한 맥락적 특징을 우선시하며 덜 관련된 정보를 억제하는 역동적인 기억 재배치 메커니즘을 통해 이러한 패러다임을 더욱 발전시킵니다. 실험 결과, 기억 재조직화 전략을 통해 긴 입력 시퀀스에 대한 처리 오버헤드를 크게 줄이는 등의 계산 효율성이 크게 향상되었습니다.

- **Technical Details**: 제안된 방법론은 계층적 임베딩 증강과 자율 구조 기억 조작을 결합한 새로운 접근 방식을 소개합니다. 이 방법은 각 임베딩 레이어를 가중치 조합을 통해 다계층 구조로 인코딩하여 토큰 표현을 변형합니다. 이러한 구성을 통해 모델은 중요한 맥락 요소를 우선적으로 분석하고, 덜 중요한 정보를 효율적으로 버릴 수 있어 태스크 일반화와 투명성 모두를 개선합니다. 이 방식은 고유한 구조적 향상을 통해 모델의 의사결정 경로를 한층 더 이해하기 쉽게 만듭니다.

- **Performance Highlights**: 제안된 방법은 기존 LLM 아키텍처의 정적 토큰 표현의 한계를 해결하여 다양한 태스크에서의 적용 가능성을 높였습니다. 비교 분석 결과, 정확도, 효율성 및 해석 가능성 측면에서 특정 이점을 보여주었으며, 특히 복잡한 맥락적 이해가 요구되는 업무에서 두드러진 성과를 나타냈습니다. 이 연구는 자율적 메모리 재구성을 통한 실시간 메모리 관리를 통해 다영역 일반화 및 실시간 의사결정 시스템 등 여러 응용 분야에서의 강력함을 인증합니다.



### LeCoPCR: Legal Concept-guided Prior Case Retrieval for European Court of Human Rights cases (https://arxiv.org/abs/2501.14114)
Comments:
          Accepted to NAACL 2025

- **What's New**: 이번 연구는 법률 사례를 검색하는 Prior Case Retrieval (PCR) 분야에서 새로운 접근법인 LeCoPCR를 제안합니다. 이 방법은 쿼리 사례의 사실을 기반으로 법적 개념을 생성하고, 이를 통해 모델이 쿼리와 관련된 사례의 의미적 의도를 더 잘 이해하도록 돕습니다. 기존 방법들은 종종 쿼리의 의미적 의도를 간과했으며, LeCoPCR는 이를 해결하기 위해 핵심 법적 개념을 명시적으로 생성합니다.

- **Technical Details**: LeCoPCR는 쿼리 사례의 사실 설명에서 법적 개념을 생성하며, 이 개념들을 사용하여 관련된 이전 사례를 검색합니다. 데이터 주석이 부족하기 때문에 약한 감독(weak supervision) 방법을 채택하여, 추론 섹션에서 법적 개념을 추출합니다. 중요한 법적 개념을 선택하기 위해 Determinantal Point Process (DPP)를 사용하여 질과 다양성을 균형 있게 유지합니다.

- **Performance Highlights**: ECtHR-PCR 데이터셋에서 LeCoPCR의 효과성을 실험적으로 검증하였으며, 법적 개념을 명시적으로 활용하여 모델의 관련성을 향상시켰습니다. DPP 기반의 개념 추출은 다른 접근 방식보다 우수한 검색 성능을 보여주며, 이는 대표적인 다양한 개념의 선택 덕분입니다.



### RELexED: Retrieval-Enhanced Legal Summarization with Exemplar Diversity (https://arxiv.org/abs/2501.14113)
Comments:
          Accepted to NAACL 2025

- **What's New**: 이번 논문에서는 법적 요약(legal summarization) 작업을 위한 RELexED라는 새로운 프레임워크를 제안합니다. RELexED는 법적 문서를 명확하고 간결한 요약으로 변환하는 데 도움을 주며, 예시(summary exemplars)를 활용하여 모델의 작성을 안내합니다. 기존 접근 방식에서 발생하는 내용 일탈(content theme deviation)과 불일치 문제를 해결하기 위해, 우리는 정보를 효과적으로 추출할 수 있는 두 단계의 예시 선택 전략을 도입했습니다.

- **Technical Details**: RELexED는 두 가지 모듈로 구성되어 있으며, 우선 훈련 데이터베이스에서 관련 예시를 검색하는 retriever 모듈과, 검색된 예시와 소스 문서를 조합하여 요약을 생성하는 summarizer 모듈입니다. RELexED는 결정론적 점 프로세스(Determinantal Point Process, DPP)를 활용하여 선택된 예시의 질과 다양성을 균형 있게 유지합니다. 이 과정에서 각 예시 간의 상호 관련성을 고려하여 유사성이 높은 예시들을 선택하는 전통적인 방법을 극복하고자 합니다.

- **Performance Highlights**: 실험 결과는 RELexED가 예시를 사용하지 않는 모델과 유사성 기반 예시 선택 방식에 의존하는 모델보다 월등히 우수한 성능을 보인다는 것을 입증합니다. 특히, SuperSCOTUS와 CivilSum의 두 법적 요약 데이터셋에서 RELexED의 효과가 입증되었습니다. 이는 고급의 결과를 얻기 위해 모델의 크기를 늘리는 것보다, 적당한 크기의 모델에 관련된 예시 데이터로 성능을 높일 수 있다는 것을 시사합니다.



### CoPERLex: Content Planning with Event-based Representations for Legal Case Summarization (https://arxiv.org/abs/2501.14112)
Comments:
          Accepted to NAACL 2025

- **What's New**: 본 논문에서는 법적 사례 요약을 위한 CoPERLex라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 이벤트 중심의 표현(event-centric representations)을 사용하여 법적 문서의 서사를 반영하는 구조적 계획을 도입합니다. CoPERLex는 콘텐츠 선택(content selection), 콘텐츠 계획(content planning), 요약 실현(summary realization)이라는 세 가지 단계로 구성됩니다.

- **Technical Details**: CoPERLex는 '주어-동사-목적어'(Subject-Verb-Object) 튜플을 통해 중요한 사건을 캡처하는 간단한 이벤트 표현을 사용합니다. 콘텐츠 선택 단계에서는 MemSum이라는 추출 요약 시스템을 사용하여 가장 관련성이 높은 정보를 식별하고, 그 정보를 기반으로 중간 요약 표현을 생성합니다. 마지막 단계인 요약 실현에서는 Longformer Encoder-Decoder 모델을 활용하여 생성된 중간 계획을 바탕으로 유창하고 응집력 있는 요약을 생성합니다.

- **Performance Highlights**: 실험 결과, CoPERLex는 기존 방법들보다 신뢰성과 일관성에서 현저한 개선을 보여주었습니다. 특히, 이벤트 중심의 계획은 전통적인 엔티티 중심 표현(entity-centric representations)을 초월하는 성과를 이뤘습니다. 이 연구는 법적 요약 분야에서의 새로운 접근 방식을 제시하며, 법적 문서의 복잡한 맥락을 효과적으로 캡처할 수 있는 가능성을 보여줍니다.



### MedSlice: Fine-Tuned Large Language Models for Secure Clinical Note Sectioning (https://arxiv.org/abs/2501.14105)
Comments:
          Our code is publicly available on github ( this https URL )

- **What's New**: 이 연구는 공개 소스의 큰 언어 모델(Large Language Models, LLMs)을 활용하여 임상 노트를 자동으로 구분하는 방법론을 제시합니다. 본 연구는 특히 현재 질병의 역사, 구간 역사 및 평가 및 계획의 세 가지 섹션에 중점을 두어, 487개의 진행 노트에 대한 세심한 데이터셋을 사용하여 모델을 개선하고 평가했습니다. 연구 결과, 미세 조정된 Llama 3.1 8B 모델이 GPT-4o를 초과하는 성능(F1=0.92)을 나타내며, 이는 병원과 임상 분야에서의 접근성과 비용 효율성을 크게 향상시킬 수 있는 가능성을 보여줍니다.

- **Technical Details**: 임상 노트는 전문의의 직간접적인 관찰과 평가를 기록하는 반면, 이러한 노트를 효과적으로 분류하는 것은 어려움이 많습니다. 연구진은 Clinical-Longformer와 같은 다양한 기계 학습 모델을 사용하여 모델 성능을 평가했으며, 임상 노트에서 특정 구간을 추출하기 위한 미세 조정된 LLM을 훈련했습니다. 최종 데이터셋은 1,147개의 노트로 구성되며, 미세 조정 과정에서는 rank-stabilized LoRA 방법이 사용되었습니다.

- **Performance Highlights**: 최적화된 Llama 3.1 8B 모델은 F1 점수 0.92를 기록하며, 낮은 비용의 소스 모델로서 현대의 의료 시스템에서 발전된 성과를 달성했습니다. 외적 유효성 테스트 세트에서도 F1 점수 0.85를 유지하며 높은 성능을 보여주었습니다. 이러한 결과는 임상 노트를 구조화하여 후속 분석을 수행하는 데 있어 큰 장점을 제공할 것입니다.



### Communicating Activations Between Language Model Agents (https://arxiv.org/abs/2501.14082)
- **What's New**: 본 연구는 다중 언어 모델 간의 효율적인 통신을 위한 새로운 접근 방식을 제안합니다. 기존의 자연어 대신 활성화(activations)를 통해 모델 간 통신을 함으로써, 추론 비용을 대폭 절감하고 높은 정보 밀도를 유지할 수 있습니다. 이러한 대안적 언어를 통해 LLM(대형 언어 모델) 간의 재생성 능력을 극대화할 수 있음을 보입니다.

- **Technical Details**: 구체적으로, 모델 B의 계산을 중간 레이어에서 일시 정지하고, B의 활성화와 다른 모델 A의 활성화를 함수 f를 통해 결합합니다. 이후, 이 출력을 B의 다음 레이어로 전달하며 계산을 계속 진행합니다. 이 방식은 새로운 작업에서 제로 추가 파라미터로 LLM을 확장할 수 있도록 하며, 다양한 도메인과 환경에 적용할 수 있습니다.

- **Performance Highlights**: 실험을 통해 이 방법은 전통적인 자연어 통신 대비 최대 27.0%의 성능 향상을 이루었으며, 계산 비용은 1/4 미만으로 감소했습니다. 또한, 본 연구는 LLM의 다양한 크기와 스위트에 걸쳐 일반화 가능성을 보여주어, 소형 LLM도 통신의 이점을 활용할 수 있음을 입증했습니다.



### Enhancing Biomedical Relation Extraction with Directionality (https://arxiv.org/abs/2501.14079)
- **What's New**: 이번 연구는 BioRED 데이터셋에서 생물학적 관계의 엔티티 역할을 구체적으로 주석(annotation)하여 네트워크의 방향성과 복잡한 관계를 분석하는 데 도움을 주고자 합니다. 이는 생물학적 메커니즘을 연구하기 위해 필수적인 정보입니다. 또한, 방향성을 포함한 10,864개의 주석을 포함하는 개선된 BioRED 데이터셋을 발표했습니다.

- **Technical Details**: 연구팀은 BioRED 데이터셋을 기반으로 하여 새로운 multi-task language model을 제안했습니다. 이 모델은 soft-prompt learning 기술을 활용하여 관계 및 엔티티 역할을 동시에 식별할 수 있습니다. 이를 통해 기존의 대형 언어 모델들에 비해 더 나은 성능을 발휘합니다.

- **Performance Highlights**: 제안된 방법은 최신 모델인 GPT-4 및 Llama-3를 포함한 기존의 대형 언어 모델보다 두 가지 벤치마크 작업에서 우수한 성과를 보였습니다. 이는 생물학적 관계 네트워크의 이해를 한층 더 깊이 있게 할 수 있는 가능성을 보여줍니다.



### LLMs are Vulnerable to Malicious Prompts Disguised as Scientific Languag (https://arxiv.org/abs/2501.14073)
Comments:
          15 pages

- **What's New**: 이번 연구는 많은 최신 대형 언어 모델들이 과학적 언어 뒤에 숨겨진 악의적인 요청에 취약하다는 점을 드러냅니다. 특히, 특정한 요청이 기존 사회과학 및 심리학 연구를 왜곡하여 편향성을 부각시키는 결과를 초래한다는 사실이 밝혀졌습니다. 연구진은 이러한 모델들이 왜곡된 과학 논거를 만들어내는 문제를 다루며, 이는 유해한 사용자가 모델을 해킹할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 연구에서는 StereoSet 데이터를 사용하여 다양한 LLM(GPT4o, Llama3, Gemini 등)과 실험을 진행하였습니다. 실험 결과, 편향적 요청을 받았을 때 모델의 편향 및 독성이 상당히 증가하는 것으로 나타났습니다. 또한, 과학 논문의 제목과 초록을 생성하여 편향된 반응을 유도하는 실험을 설계하였으며, 이는 기존의 강화 학습이나 안전 장치(strategies like guardrails)와 같은 기존 접근법의 한계를 드러냅니다.

- **Performance Highlights**: 연구 결과, 과학 논문을 기반으로 한 설득 전략이 LLM들에게 편향된 결과를 효과적으로 유도할 수 있음을 보여주었습니다. 특히, 다회 대화에서 특정 메타데이터(저자 이름과 발표 장소 등)가 LLM의 설득력에 영향을 미치는 것으로 관찰되었으며, 대화가 진행될수록 편향 점수가 증가하는 경향이 있었습니다. 이로 인해 현재의 LLM이 과학적 텍스트에 취약하다는 점과, 향후 훈련 과정에서의 과학적 데이터 사용에 대한 재검토가 필요함을 강조합니다.



### Leveraging Large Language Models to Analyze Emotional and Contextual Drivers of Teen Substance Use in Online Discussions (https://arxiv.org/abs/2501.14037)
Comments:
          28 pages, 9 figures with an appendix

- **What's New**: 이 연구는 청소년의 사회적 미디어 게시물 분석에 큰 언어 모델(Large Language Models, LLMs)을 적용하여 감정 패턴과 맥락적 요인을 밝혀냈습니다. 이를 통해 약물 사용과 관련된 정서(예: 슬픔, 죄책감, 두려움, 기쁨)와 환경(예: 가족, 또래, 학교)을 투명하게 드러냈습니다. 또한, 이러한 분석 기법을 통해 약물 사용과 관련된 게시물의 주요 예측 인자를 파악했습니다.

- **Technical Details**: 연구는 머신 러닝(Manchine Learning) 기술과 히트맵(Heatmap) 분석을 결합하여 게시물에서 나타난 정서적 양상과 사회적 맥락을 조사했습니다. 결과적으로 부정적인 감정인 슬픔과 죄책감이 약물 사용 맥락에서 더 빈번하게 나타났으며, 죄책감은 보호 요인으로 작용했습니다. 반면, 수치심과 또래 영향은 약물 사용 위험을 증가시켰습니다.

- **Performance Highlights**: 연구 결과, 또래의 영향은 슬픔, 두려움, 혐오와 강한 상관관계를 보였고, 가족과 학교 환경은 비약물 사용 관련성과 일치했습니다. 이 발견은 청소년의 감정적 취약성과 맥락적 영향을 다루는 것의 중요성을 강조하며, 가족, 학교 및 지역 사회가 협력하여 위험 요인을 줄이고 더 건강한 청소년 발달을 지원할 수 있는 방법을 제안합니다.



### Advancing Math Reasoning in Language Models: The Impact of Problem-Solving Data, Data Synthesis Methods, and Training Stages (https://arxiv.org/abs/2501.14002)
Comments:
          ICLR 2025

- **What's New**: 이번 연구는 수학적 추론(Mathematical Reasoning) 능력을 크게 향상시키기 위한 새로운 접근 방식을 제안합니다. 기존의 수학 특정 LLMs는 일반적인 수학 문서와 문제 데이터셋을 활용하여 두 단계로 학습하지만, 연구 결과 일반 수학 문서보다 문제 해결 데이터의 사용이 더 효과적임을 발견했습니다. 이를 통해 JiuZhang-8B라는 강력한 수학 기반 모델이 개발되었습니다.

- **Technical Details**: 연구는 세 가지 주요 질문을 탐구합니다: (1) 문제 해결 데이터가 일반 수학 문서보다 CPT(Continue Pre-training) 과정에서 더 효과적으로 수학적 추론 능력을 향상시킬 수 있는가? (2) 동일한 출처의 합성 데이터(Synthetic Data)의 효과성을 비교하며, 어떤 합성 방법이 가장 효율적인가? (3) 동일한 문제 해결 데이터를 기반으로 개발된 능력이 CPT와 SFT(Supervised Fine-Tuning) 단계에서 어떻게 다르고, 이러한 차이에 기여하는 요인은 무엇인가? 여러 실험을 통해 문제 해결 데이터가 수학적 능력을 크게 향상시킨다는 것을 보여주었습니다.

- **Performance Highlights**: 문제 해결 데이터를 기반으로 한 CPT가 SFT보다 더 나은 성능을 발휘하였으며, 특히 튜터쉽 증폭 합성 방법(Tutorship Amplification Synthesis Method)이 최상의 결과를 보였습니다. SFT는 지시 사항을 따르는 능력을 향상시키지만, 복잡한 다단계 문제 해결 데이터에 대한 학습 능력 저하로 인해 CPT에 비해 성능이 떨어지는 경향이 있습니다. 이러한 연구 결과는 LLMs의 수학적 추론 능력 최적화를 위한 귀중한 가이드를 제공합니다.



### Framework for Progressive Knowledge Fusion in Large Language Models Through Structured Conceptual Redundancy Analysis (https://arxiv.org/abs/2501.13999)
- **What's New**: 본 연구는 대규모 모델 내에서 잠재 지식의 구조적 중복성을 분석하고 재구성하는 새로운 방법론을 제안하였다. 기존의 중복 문제를 해결하기 위한 알고리즘적 혁신과 이론적 기초를 바탕으로 하여, 지식 통합의 효율성을 높이면서 성능을 개선할 수 있는 방법을 모색하였다. 제안된 프레임워크는 중요한 의미적 관계를 보존하면서 불필요한 겹침을 제거하는 데 중점을 두었다.

- **Technical Details**: 구체적으로, 저자들은 잠재 공간 내에서 개념적 클러스터를 식별하기 위해 계층적 클러스터링 기법을 활용하였다. 이는 서로 관련성이 있는 개념들이 통합될 수 있도록 돕고, 맥락의 풍부함을 유지하면서 중복을 해결할 수 있게 한다. 또한, 동적 가중치 재조정 기법을 통해 높은 유용성을 지닌 표현들을 우선시하여 중요하지 않은 중복을 제거했다.

- **Performance Highlights**: 실험 결과는 메모리 효율성이 개선되고 추론 속도가 빨라지며, 오류율 감소 및 적대적 강건성이 증가하는 등 모델의 성능이 크게 향상되었음을 보여주었다. 자원 소비의 감소와 번역 및 요약 작업에서의 성과 향상이 두드러지며, 실제 배포 환경에서의 실용성을 추가적인 에너지 메트릭을 통해 입증하였다.



### CAPRAG: A Large Language Model Solution for Customer Service and Automatic Reporting using Vector and Graph Retrieval-Augmented Generation (https://arxiv.org/abs/2501.13993)
Comments:
          14 pages, 5 Figures, 3 Tables

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)을 기반으로 한 금융 챗봇을 통한 고객 경험 개선을 위한 AI 에이전트를 도입했습니다. 고객은 은행 서비스 및 연례 보고서에 대한 관련 정보를 손쉽게 얻을 수 있으며, 하이브리드 고객 분석 파이프라인인 CAPRAG를 통해 관계 기반 및 맥락 기반 질문 모두를 효과적으로 처리합니다. 이러한 접근 방식은 벡터 RAG와 그래프 RAG를 결합하여 더 나은 고객 참여를 촉진합니다.

- **Technical Details**: 하이브리드 접근 방식을 통해 CAPRAG는 고객의 질문을 확장한 후, 검색된 결과를 그래프 데이터베이스와 결합하여 사용자에게 정확한 답변을 제공합니다. Cypher 쿼리 언어를 사용하여 그래프 데이터베이스를 효율적으로 질의할 수 있으며, 이를 LLM을 통해 응답 생성에 활용합니다. 저자들은 이러한 방식을 통해 디지털 환경에서의 모호함을 줄이고 정보 접근성을 높이려고 노력하고 있습니다.

- **Performance Highlights**: 연구팀은 LLMs를 평가 기준으로 삼아 결과의 정확성과 일관성을 평가합니다. 주요 평가 지표로는 답변의 관련성, 맥락의 관련성, 사실 기반의 신뢰성을 포함하여, 모델의 출력 품질을 지속적으로 관리합니다. 이를 통해 사용자 경험을 개선하고 시스템에 대한 신뢰를 구축할 수 있었습니다.



### Comprehensive Modeling and Question Answering of Cancer Clinical Practice Guidelines using LLMs (https://arxiv.org/abs/2501.13984)
- **What's New**: 이번 연구에서는 의료 전문가가 치료 결정을 내리는 데 도움이 되기 위해 Clinical Practice Guidelines (CPGs)의 규정 지식을 효과적으로 포착하는 방법을 제안합니다. 특히 National Comprehensive Cancer Network (NCCN) Cancer CPGs의 맥락을 풍부하게 한 디지털 표현을 그래프 형태로 구축하는 접근 방식을 소개하고 있습니다. 이를 통해 CPGs에 포함된 의료 지식을 보다 충실하게 캡처할 수 있습니다.

- **Technical Details**: 연구진은 자동 추출 및 노드(node)와 관계(relationship)의 분류를 통해 보다 풍부한 디지털 표현을 생성하였습니다. 또한, 대형 언어 모델(Large Language Models, LLMs)을 활용하여 노드를 분류함으로써 80.86%와 88.47%의 정확도를 달성하였습니다. 이 과정에서 zero-shot learning과 few-shot learning을 적용하였습니다.

- **Performance Highlights**: 이 연구에서 개발한 방법론은 자연어 질문에 대한 답변을 제공하는 데 활용됩니다. LLMs를 활용하여 가이드라인 지식 기반에서 관련 서브그래프(subgraph)를 추출하고, 이를 통해 정확한 답변을 생성함으로써 의료 분야의 질문 응답(Question Answering)에서 발생할 수 있는 오류와 환각(hallucination) 문제를 완화합니다. 이러한 접근 방식은 의료 도메인에서 사실 정확성을 보장하는 데 기여합니다.



### AdEval: Alignment-based Dynamic Evaluation to Mitigate Data Contamination in Large Language Models (https://arxiv.org/abs/2501.13983)
- **What's New**: 本 연구에서는 AdEval(Alignment-based Dynamic Evaluation)이라는 새로운 동적 데이터 평가 방법을 제안한다. 이 방법은 데이터 오염(data contamination)의 영향을 줄이고 평가 신뢰성을 높이기 위해 설계되었다. AdEval은 정적(static) 데이터의 핵심 개념과 일치하는 질문을 생성하며, 온라인 검색을 통해 관련 지식 포인트에 대한 자세한 설명을 제공하여 평가 샘플을 생성한다.

- **Technical Details**: AdEval의 주요 혁신은 동적으로 정렬된 평가 데이터를 생성하는 것이다. 이를 위해 정적 데이터에서 핵심 지식 포인트와 주요 아이디어를 추출하고, 온라인 검색 결과를 활용하여 관련 내용을 상세히 확장한다. 또한, Bloom의 분류법을 기반으로 LLM의 성과를 기억, 이해, 적용, 분석, 평가, 생성의 여섯 인지 수준에서 다차원적 평가를 수행한다.

- **Performance Highlights**: 다양한 데이터셋에 대한 실험 결과, AdEval은 데이터 오염이 평가 결과에 미치는 영향을 효과적으로 줄이며 신뢰성과 공정성을 향상시킨다. 이 방법은 LLM 평가에서 신뢰할 수 있는 결과를 도출하고, 다양한 복잡성 수준을 지원하여 평가의 질을 높이는 데 기여한다.



### Chain of Grounded Objectives: Bridging Process and Goal-oriented Prompting for Code Generation (https://arxiv.org/abs/2501.13978)
- **What's New**: 최근 대규모 언어 모델(LLM)의 코드 생성 활용도가 급증하면서 기존 방법들의 한계를 해결하기 위한 새로운 접근 방식인 Chain of Grounded Objectives (CGO)가 제안되었습니다. CGO는 입력 프롬프트에 기능적 목표를 포함하여 코드 생성을 향상시키며, 그 과정에서 명시적인 순차적 절차를 피하면서 프로그래밍 작업의 구조적 특성에 잘 적응합니다. 이를 통해 기존의 접근 방법들의 제약을 극복하고 코드 생성의 질을 높이는 데 기여하고자 합니다.

- **Technical Details**: CGO는 기능적 목표를 자연어 또는 다른 구조화된 형태로 입력 프롬프트에 직접 포함시키는 방식으로, 목표 정의의 명확성과 프로세스 지향적 추론의 암묵적인 논리 구조를 통합합니다. 이 과정에서 LLM이 훈련받은 코드 주석 형식의 구조를 활용하여, 문제의 목표와 생성된 코드 간의 정렬을 보장합니다. CGO는 ‘목표 생성 단계’와 ‘코드 생성 단계’의 두 가지 단계로 구성되어 있으며, 앞서 언급된 목표들이 LLM에 의해 생성된 후 이를 바탕으로 코드를 생성하는 방식입니다.

- **Performance Highlights**: 실험 결과에 따르면, CGO는 기존의 방법들에 비해 코드 생성 성능이 현저히 향상된 것으로 나타났습니다. 특히, pass@1 및 pass-ratio@10과 같은 정확성 메트릭에서 더 높은 정확도를 기록하며, LLM의 코드 주석과 문서화 패턴에 대한 익숙함을 활용하여 최종 코드와 기능적 목표 간의 정렬을 강화했습니다. 이러한 성과는 CGO가 코드 생성 벤치마크에서 기준 모델들을 초월하는 데 기여했음을 입증합니다.



### Re-ranking Using Large Language Models for Mitigating Exposure to Harmful Content on Social Media Platforms (https://arxiv.org/abs/2501.13977)
Comments:
          This paper is under peer review

- **What's New**: 이 논문에서는 소셜 미디어 플랫폼이 사용자의 참여를 극대화하기 위해 사용하는 기계 학습(ML) 및 인공지능(AI) 기반 추천 알고리즘이 부정적인 콘텐츠에 노출되는 문제를 해결하기 위한 새로운 접근법을 제안합니다. 제안된 방법은 대규모 데이터를 필요로 하지 않고, 대형 언어 모델(LLMs)을 활용하여 콘텐츠 시퀀스를 동적으로 평가하고 재정렬하여 유해한 콘텐츠 노출을 줄이는 것입니다. 이 방법은 기존의 콘텐츠 조절 방식과 비교하여 더 높은 성능을 발휘하는 것을 보여줍니다.

- **Technical Details**: 이 논문에서는 대형 언어 모델(LLMs)이 제로샷(zero-shot) 및 퓨샷(few-shot) 학습 설정에서 우수한 추론 능력을 보여준다는 점을 강조하며, 이를 통해 유해한 콘텐츠 노출을 줄이기 위한 방법을 제시합니다. LLMs는 페어와이즈 비교(pairwise comparison)와 재정렬(re-ranking)을 통해 콘텐츠 시퀀스를 검토하여 권장할 콘텐츠의 질을 높이는 역할을 합니다. 또한 유해한 콘텐츠에 대한 노출을 분석하기 위한 두 가지 새로운 메트릭(metrics)을 도입하여, 제안하는 방법의 유용성을 평가합니다.

- **Performance Highlights**: 실험 결과, 제안된 LLM 기반 접근법은 세 가지 유해 데이터셋과 세 가지 LLM 아키텍처를 통해 기존의 상업적 조절 API인 Perspective API 및 OpenAI Moderation API에 비해 우수한 성능을 보였습니다. 제로샷 환경에서도 산업 표준 분류기를 초월하는 성능을 나타내어, 다양한 유형의 유해에 일반화 가능하다는 점을 강조합니다. 이를 통해 플랫폼에서 유해 콘텐츠 노출을 효과적으로 줄일 수 있는 확장 가능하고 적응 가능한 솔루션을 제공한다고 결론짓습니다.



### Towards Safer Social Media Platforms: Scalable and Performant Few-Shot Harmful Content Moderation Using Large Language Models (https://arxiv.org/abs/2501.13976)
Comments:
          This paper is in submission and under peer review

- **What's New**: 본 논문에서는 소셜 미디어에서 유해 콘텐츠를 효과적으로 모니터링하기 위해 Large Language Models (LLMs)를 활용한 새로운 접근 방식을 소개합니다. 기존의 자동화된 방법들과 비교하여, LLMs는 소량의 예제만으로도 유해 콘텐츠를 식별하는 성능이 뛰어난 것으로 나타났습니다. 또한, 비주얼 정보(영상 썸네일)를 통합함으로써 모델의 성능을 더욱 향상시킬 수 있다는 점을 밝힙니다.

- **Technical Details**: 연구에서는 Llama2-12B, Mistral-7B, GPT-4o-Mini, GPT-3.5-Turbo 등 여러 LLM을 활용하여 유해 콘텐츠를 분류하는 실험을 진행했습니다. 결과적으로, LLMs는 zero-shot 설정에서도 기존의 상업적 기준인 Google의 Perspective API 및 OpenAI의 Moderation API를 초과하는 성능을 보여주었습니다. 또한, few-shot learning을 통해 지침 프롬프트에서 유해 및 비유해 콘텐츠의 예제를 제공함으로써 분류 성능을 크게 향상시켰습니다.

- **Performance Highlights**: 이번 연구를 통해 LLMs가 유해 콘텐츠 식별에서 기존의 딥러닝 모델을 초과함을 입증했습니다. 또한, multimodal LLMs를 이용해 시각적 입력을 포함할 경우 유해 콘텐츠를 식별하는 정확도가 향상됨을 밝혔습니다. 특히, 오픈소스 multimodal 모델보다 비공식 소스 모델에서 더 나은 성능이 나타났습니다.



### Assisting Mathematical Formalization with A Learning-based Premise Retriever (https://arxiv.org/abs/2501.13959)
- **What's New**: 이번 연구에서는 수학의 정형화를 지원하기 위한 프레미스 리트리버(premise retriever) 훈련을 위한 혁신적인 방법을 소개합니다. 이 방법은 BERT 모델을 사용하여 증명 상태와 프레미스를 공유 잠재 공간(latent space)에 임베딩합니다. 실험 결과, 우리의 모델은 기존 기준선보다 성능이 뛰어나며, 컴퓨팅 자원이 적게 소모되어 효율적임을 보여주었습니다.

- **Technical Details**: 프레미스 리트리버 모델은 컨텍스트 프리 리트리벌(CFR) 모듈과 컨텍스트 인지 재정렬(CAR) 모듈로 구성되어 있습니다. 각 모듈은 BERT를 기반으로 하며, CFR 모듈은 임베딩을 생성하여 입력된 증명 상태와 유사한 프레미스를 검색합니다. CAR 모듈은 검색된 프레미스를 재정렬하여 최상위 k-k recall을 개선하여 결과의 정확성을 높이는 데 기여합니다.

- **Performance Highlights**: 우리 모델은 성능에서 최첨단 모델을 초월하는 결과를 보였으며, 여전히 계산 효율성을 유지하고 있습니다. 또한 우리는 이 모델을 독립적으로 훈련된 전술 생성기(tactic generator)와 통합하여 검색 보강 자동 정리 증명기(retrieval-augmented automated theorem prover)를 평가하였습니다. 이 시스템은 수학적 증명 과정의 효율성을 높이는 데 기여하며, 사용자들에게 편리한 웹 검색 엔진을 제공할 예정입니다.



### A Survey of Graph Retrieval-Augmented Generation for Customized Large Language Models (https://arxiv.org/abs/2501.13958)
- **What's New**: 최근의 연구는 Graph-based Retrieval-Augmented Generation (GraphRAG) 모델이 기존 RAG의 한계를 극복하는 새롭고 효율적인 방법임을 보여줍니다. GraphRAG는 전문 도메인에 맞는 LLM의 적용을 통해 복잡한 쿼리를 이해하고, 다양한 지식을 통합하며, 효율성을 향상시킵니다. 이러한 혁신적인 접근 방식은 지식 구조를 그래프 형태로 표현하여 도메인 내의 관계와 계층을 명확하게 설명합니다.

- **Technical Details**: GraphRAG는 도메인 지식의 명확한 표현을 위해 그래프 구조를 활용하며, 이를 통해 다단계 추론을 지원하는 효율적인 검색 기법을 도입합니다. 이 모델은 쿼리에 대한 맥락을 유지하면서 지식을 검색할 수 있도록 설계되었으며, 얻어진 지식을 활용해 LLM의 생성 품질을 높이는 구조 인지 알고리즘을 포함합니다. 또한, 다양한 전문 분야에서 GraphRAG의 현재 구현 사례를 분석하여 중요한 기술적 도전 과제를 확인합니다.

- **Performance Highlights**: GraphRAG의 통합적인 지식 검색 방법은 사용자 쿼리에 대한 정확하고 논리적으로 일관된 응답을 생성할 수 있게 해 줍니다. 기존 RAG 시스템의 성능 한계를 극복하여 복잡한 쿼리를 보다 효과적으로 처리하며, 시스템의 효율성을 높이는 동시에 실시간으로 도메인 특화된 지식에 접근할 수 있도록 합니다. 이를 통해 LLM을 전문화된 환경에서 더욱 효과적으로 활용할 수 있는 가능성을 제시합니다.



### Benchmarking Generative AI for Scoring Medical Student Interviews in Objective Structured Clinical Examinations (OSCEs) (https://arxiv.org/abs/2501.13957)
Comments:
          11 pages, 4 figures (+3 figures in supplementary appendix)

- **What's New**: 이번 연구는 Objective Structured Clinical Examination (OSCE)에서 의료 학생들의 의사소통 능력을 평가하기 위해 대규모 언어 모델(LLMs)을 자동화 가능성을 조사했습니다. 특히, Master Interview Rating Scale (MIRS)을 사용하여 OSCE 평가의 효율성을 높일 수 있는 방법을 모색하였습니다.

- **Technical Details**: 연구에서는 제로샷(zero-shot), 체인 오브 사고(chain-of-thought, CoT), 몇 번의 샷(few-shot), 다단계(multi-step) 프롬프트 기법을 사용하여 최신 LLM 포트폴리오인 GPT-4o, Claude 3.5, Llama 3.1, Gemini 1.5 Pro의 성능을 비교했습니다. 174명의 전문가 합의 점수가 제공되는 10개의 OSCE 사례 데이터셋을 기준으로 모델 성능이 측정되었으며, 정확성 메트릭스는 exact, off-by-one, thresholded로 평가되었습니다.

- **Performance Highlights**: LLMs는 모든 MIRS 항목과 OSCE 사례에 대해 평균 정확도가 낮았지만, off-by-one과 thresholded 정확도는 중간에서 높은 범위에 있었습니다. 특히, GPT-4o는 높은 intra-rater 신뢰성을 보였으며(α = 0.98), CoT, few-shot, multi-step 기법은 특정 평가 항목에 맞추어 조정되었을 때 효과적이었습니다. 연구 결과는 AI를 활용한 OSCE 평가의 가능성을 보여주며, 향후 임상 의사소통 기술 자동화 평가 연구의 기초 성능 평가로 활용될 수 있습니다.



### Zep: A Temporal Knowledge Graph Architecture for Agent Memory (https://arxiv.org/abs/2501.13956)
Comments:
          12 pages, 3 tables

- **What's New**: Zep은 AI 에이전트를 위한 새로운 메모리 레이어 서비스로, Deep Memory Retrieval (DMR) 벤치마크에서 기존의 MemGPT를 능가하는 성능을 보여줍니다. Zep은 DMR보다 더 포괄적이면서도 어려운 평가에서도 뛰어난 성과를 내어 실제 기업 사용 사례를 보다 잘 반영합니다. Graphiti라는 핵심 구성 요소를 통해 다양한 데이터 소스로부터 동적인 지식 통합을 수행하여 기업의 요구를 충족합니다.

- **Technical Details**: Zep의 정보 저장 방식은 동적인 지식 그래프(knowledge graph)를 기반으로 하며, 이 그래프는 입력 메시지 데이터를 포함하는 에피소드 서브그래프, 에피소드에서 추출된 엔티티를 나타내는 의미 엔티티 서브그래프, 그리고 강하게 연결된 엔티티들의 클러스터를 나타내는 커뮤니티 서브그래프의 세 가지 계층으로 구성되어 있습니다. Graphiti는 비손실 방식으로 지식 그래프를 업데이트하며, 각 사실과 관계의 타임라인을 유지하여 복잡하고 진화하는 세상을 모델링합니다. 이 기술적 접근은 LLM 기반 에이전트의 기억을 혁신적으로 확장하는 데 기여합니다.

- **Performance Highlights**: Zep은 DMR 벤치마크에서 94.8%의 성능을 기록하여 MemGPT의 93.4%를 초과하였습니다. LongMemEval 벤치마크에서도 Zep은 최대 18.5%의 정확도 향상과 함께 응답 지연 시간을 90% 단축하는 성과를 기록했습니다. 이러한 결과는 특히 세션 간 정보 통합 및 장기적 컨텍스트 유지와 같은 기업 필수 작업에서 두드러지며, 실제 애플리케이션에서의 효과성을 입증합니다.



### Guided Persona-based AI Surveys: Can we replicate personal mobility preferences at scale using LLMs? (https://arxiv.org/abs/2501.13955)
- **What's New**: 본 연구에서는 독일의 개인 이동 선호도를 중심으로 인공지능 언어 모델(Large Language Models, LLMs)의 잠재력을 탐구합니다. 전통적인 설문조사 방법의 한계를 해결하기 위해 LLMs를 활용하여 합성 데이터를 생성하는 방식으로 진행됩니다. 특히, 인구통계학적 및 행동적 특성을 조합한 '페르소나(Personas)'를 도입하여 다섯 가지 다른 합성 설문조사 방법과 비교했습니다.

- **Technical Details**: 연구에서 사용한 MiD 2017 데이터셋은 독일의 이동 선호도에 대한 자세한 정보를 제공합니다. AI 설문조사 생성에 있어 GPT-4o API를 사용하였고, 10,000명의 인구를 생성하여 15,840개의 독특한 페르소나를 정의했습니다. 각각의 방법에 대해 구체적으로 설계된 프롬프트를 사용하여 LLM의 응답 생성을 안내했습니다.

- **Performance Highlights**: 결과적으로 제안된 가이드된 페르소나 기반 설문조사가 기존의 방법들에 비해 정확도와 일관성에서 큰 개선을 보였습니다. 이 접근 방식은 교통 계획 및 사회 과학 연구에서 높은 신뢰도를 가진 합성 데이터 세트를 생성할 수 있는 기회를 제공합니다. LLMs를 활용한 합성 데이터는 비용 효율적이면서도 개인 정보 보호가 가능한 데이터 생성을 가능하게 합니다.



### Chat3GPP: An Open-Source Retrieval-Augmented Generation Framework for 3GPP Documents (https://arxiv.org/abs/2501.13954)
- **What's New**: Chat3GPP는 3GPP 문서에 특화된 오픈 소스 RAG(확장된 정보 검색 생성) 프레임워크로 제안되며, 이는 도메인 특화된 파인 튜닝 없이 사용자 쿼리에 대한 관련 정보 검색 및 정확한 응답 생성을 가능하게 합니다. 이러한 유연성 및 확장성 덕분에 Chat3GPP는 3GPP 외의 다른 기술 표준에도 쉽게 적용될 수 있는 잠재력을 지니고 있습니다.

- **Technical Details**: Chat3GPP는 3GPP의 기술 문서에서 정보를 처리하고 검색하기 위해 데이터 전처리, 인덱싱, 검색 및 생성의 여러 주요 단계를 포함하는 아키텍처를 가지고 있습니다. 이를 위해, 3GPP FTP 사이트에서 Release 17과 Release 18 기술 사양 문서를 크롤링하고, 문서의 형식을 변환한 후, 내용을 필터링하여 텍스트만을 추출하여 임베딩 데이터베이스로 변환합니다.

- **Performance Highlights**: Chat3GPP는 통신 관련 데이터 세트에서 평가되었으며, 기존 방법들에 비해 우수한 성능을 보여줍니다. 이는 프로토콜 생성 및 코드 자동화와 같은 다운스트림 태스크에 대한 잠재력을 강조하며, LLM의 발전을 통해 통신 산업의 다양한 측면에서 혁신을 기대할 수 있습니다.



### Redundancy Principles for MLLMs Benchmarks (https://arxiv.org/abs/2501.13953)
- **What's New**: 최근 Multi-modality Large Language Models (MLLMs)의 급속한 발전과 함께 매년 수백 개의 벤치마크가 생산되고 있습니다. 이로 인해 기반 성능 평가에서 중복된 부분이 발생하고 있으며, 이 논문은 이러한 중복 문제를 체계적으로 분석하여 효과적인 MLLM 벤치마크 설계를 위한 원칙들을 제시합니다. 특히, 벤치마크의 능력 차원, 테스트 질문 수의 중복성, 도메인 간의 중복성을 강조하고 있습니다.

- **Technical Details**: MLLM 평가의원칙으로는 독립된 차원, 최적의 인스턴스 수, 도메인 대표성이 제안됩니다. 중복성의 정량화는 Performance Correlation Redundancy Framework를 통해 이루어지며, 수행 순위의 상관관계를 측정하여 얼마나 중복성이 존재하는지 파악합니다. 100개 이상의 MLLM을 포함한 VLMEvalKit 데이터를 활용하여 체계적이고 포괄적인 분석을 수행하였습니다.

- **Performance Highlights**: 본 논문은 MLLM 벤치마크의 중복성을 평가함으로써 벤치마크 설계의 최적화를 도모하고, 효율성을 높여 모델 평가 시스템을 개선할 수 있는 방안을 제시합니다. 또한, 중복성을 시스템적으로 해결함으로써 MLLM 평가의 자원 요구를 줄이고 보다 효과적인 평가 생태계를 구축하는 데 기여할 수 있을 것으로 보입니다.



### The Dual-use Dilemma in LLMs: Do Empowering Ethical Capacities Make a Degraded Utility? (https://arxiv.org/abs/2501.13952)
- **What's New**: 이 논문은 Direct Preference Optimization (DPO) 기반의 정렬 프레임워크인 LibraAlign를 제안하여 안전성과 유용성 간의 균형을 맞추는 방법을 제시합니다. DPO 프레임워크는 화학 분야의 데이터셋인 LibraChemQA를 사용하여 안전한 정보 차단과 유효한 요청 수용을 동시에 고려합니다. 이 연구는 기존 LLM의 성능을 향상시키기 위한 다양한 방법론을 통해 이 분야의 윤리적 문제를 탐구합니다. 특히, 화학 분야에서의 응용을 증명 사례로 제시하며, 다른 도메인에도 적용 가능성을 강조합니다.

- **Technical Details**: LibraAlign 프레임워크는 강화 학습에서의 인체 피드백(RLHF) 및 DPO 모델을 기반으로 하여 안전성과 유용성을 동시에 고려하는 메커니즘을 갖추고 있습니다. 이 모델은 31,600개의 트리플 인스턴스를 포함한 화학 질문-답변 데이터셋인 LibraChemQA를 생성하여, 긍정적 요청과 부정적 요청을 모두 아우르는 균형 잡힌 데이터 생성을 목표로 합니다. 또한, LLM의 도메인 특화 이해도를 높이기 위해 질문 재구성(rephrasing) 기법을 포함하는 데이터 증강(data augmentation) 방식이 혁신적으로 도입되었습니다.

- **Performance Highlights**: LibraChem 모델은 Claude-3, GPT-4o, LLaMA-3과 같은 기존 LLM보다 각각 13.44%, 7.16%, 7.10%의 성능 향상을 기록하며 기존 모델을 능가하는 것으로 나타났습니다. 실험 결과는 안전성과 유용성 모두를 고려했을 때의 모델의 전반적인 성능 향상을 보여줍니다. 이 연구는 도메인 전문 지식을 유지하면서도 윤리적인 역량을 강화하는 이중 사용(dilemma) 문제를 해결하는 데 있어 중요한 기여를 하고 있습니다.



### A Layered Multi-Expert Framework for Long-Context Mental Health Assessments (https://arxiv.org/abs/2501.13951)
- **What's New**: 이번 연구에서 제안된 Stacked Multi-Model Reasoning (SMMR) 프레임워크는 대규모 언어 모델(LLMs)의 한계를 극복하기 위해 여러 개의 모델을 활용합니다. 이 프레임워크는 각 LLM을 독립적인 '전문가'로 간주하여, 정확성과 신뢰성을 높이는 데 기여합니다. SMMR은 우선 짧고 개별적인 작업을 처리하고, 이후에 긴 문맥을 다룰 수 있는 모델을 사용하여 결과를 통합하고 정교화합니다. 이러한 접근은 정신 건강 평가에 있어 보다 균형 잡힌 다양한 관점을 제공합니다.

- **Technical Details**: SMMR 프레임워크는 깊은 레이어로 구성되어 있으며, 초기 레이어는 작은 LLM이나 전문화된 모델을 활용해 초기 평가를 생성합니다. 이후 레이어는 긴 문맥 모델을 통해 각 단계에서 측정된 출력을 합치고 세분화하여 최종 결과를 도출합니다. 최종적으로는 가장 안정적인 성능을 보이는 장기 문맥 모델을 선택하여 복잡한 정신 건강 평가를 위한 최종 결과를 생성합니다. SMMR은 성능 지표를 활용하여 각 레이어에서 개선이 이루어질 때만 다음 레이어로 진행하는 동적 중지 메커니즘을 포함합니다.

- **Performance Highlights**: SMMR은 DAIC-WOZ 우울증 선별 데이터를 포함한 다양한 사례 연구에 대한 평가를 통해, 단일 모델 기준에 비해 일관된 개선이 있음을 입증했습니다. 정확도, F1 점수, 그리고 PHQ-8 오류 감소 면에서 성과를 향상시켰습니다. 또한, 여러 전문가의 의견을 활용함으로써 환각(hallucinations) 및 미세한 임상 세부 사항을 포착할 수 있는 능력이 향상되었습니다. 이 연구 결과는 AI 기반 선별의 신뢰성을 높이는 데 기여할 것으로 기대됩니다.



### Can OpenAI o1 Reason Well in Ophthalmology? A 6,990-Question Head-to-Head Evaluation Study (https://arxiv.org/abs/2501.13949)
Comments:
          44 pages

- **What's New**: 이번 연구에서는 OpenAI의 o1 모델과 다른 대형 언어 모델(LLMs)의 성능과 추론 능력을 검토하였습니다. 특히, 안과 분야에 특화된 6,990개의 질문을 통해 비교 분석을 수행했습니다. o1은 인식 정확도(accuracy)와 macro-F1 점수에서 최고 기록을 세웠지만, 텍스트 생성 메트릭스에서는 추론 능력 면에서 세 번째로 평가받았습니다.

- **Technical Details**: 연구에서는 o1이 "Lens"와 "Glaucoma" 주제에서 1위를 차지했지만, "Corneal and External Diseases", "Vitreous and Retina", 및 "Oculoplastic and Orbital Diseases"에서는 GPT-4o에 이어 두 번째로 위치했습니다. 또한, 하위 주제에 대한 분석을 통해 o1이 긴 정답 설명이 포함된 질문에 대해 더 나은 성과를 보였음을 확인했습니다.

- **Performance Highlights**: 연구 결과에 따르면, o1의 추론 능력이 안과 분야에 완전히 적용되지 않을 수 있다는 점이 강조되었습니다. 이는 안과와 같은 전문 분야에서 최적의 성능을 위해 도메인 구체적인 개선이 필요함을 시사합니다. 따라서, o1의 발전이 특정 분야에 어떻게 영향을 미치는지 이해하는 것이 중요합니다.



### Longitudinal Abuse and Sentiment Analysis of Hollywood Movie Dialogues using LLMs (https://arxiv.org/abs/2501.13948)
- **What's New**: 이번 연구는 1950년부터 2024년까지 헐리우드 영화의 대사(Dialogue)에 대한 장기적인 학대(Abuse) 및 정서(Sentiment) 분석을 수행하였습니다. Large Language Models (LLMs)를 활용하여 1,000개 이상의 영화 자막(Subtitles)을 분석하고, 감정과 폭력적 콘텐츠의 변화를 살펴보았습니다. 연구 결과, 최근 몇십 년 동안 영화 대사에서의 학대 내용은 지속적으로 증가하였고, 이는 사회적 규범과 정책의 변화反映를 나타냅니다.

- **Technical Details**: 본 연구에 사용된 LLM의 세부 사항은 BERT(Bidirectional Encoder Representation from Transformers) 기반 모델을 미세 조정(Fine-tuning)하여 감정 분석 및 폭력 및 학대 콘텐츠 탐지를 수행하는 것입니다. 정서 분석(Sentiment analysis)은 SenWave 데이터셋을 통해 수행되며, 장기적 연구(Longitudinal study)의 구성을 통해 영화 장르별로 데이터 패턴을 비교 분석합니다. 본 연구는 특정 시점의 측정이 아니라 시간에 따른 변동을 직접적(Direct)가로 살펴보는 방법론(Metodology)을 적용합니다.

- **Performance Highlights**: 연구 결과, 최근 20년 동안 헐리우드 영화에서의 학대 콘텐츠는 상당히 증가하고 있으며, 이는 오스카 수상 영화가 상위 블록버스터 영화의 내용을 초과하는 추세를 보였습니다. 스릴러(Thriller) 장르는 여전히 높은 빈도의 학대 콘텐츠를 포함하고 있으며, 긍정적인 감정을 포함한 유머와 낙관론이 대다수 영화에서 지속적으로 나타난다고 보고되었습니다. 본 연구는 다양한 장르에서 학대 및 감정 분석이 어떻게 상호작용하는지를 보여줍니다.



### A Comprehensive Survey on Integrating Large Language Models with Knowledge-Based Methods (https://arxiv.org/abs/2501.13947)
- **What's New**: 본 논문은 Large Language Models (LLMs)와 구조화된 지식 기반 시스템의 통합을 통해 인공지능(AI) 능력을 향상시키려는 접근 방식을 탐구합니다. LLMs의 생성적 언어 이해와 구조화된 시스템의 정확한 지식 표현을 결합함으로써 AI의 실제 응용 프로그램에서 발생할 수 있는 기술적, 운영적, 윤리적 도전 과제가 논의됩니다. 이 연구는 LLMs와 지식 기반 시스템 간의 시너지를 분석하여 데이터 맥락화, 모델 정확도 향상 및 지식 자원의 효율적 활용의 이점을 강조합니다.

- **Technical Details**: LLMs는 여러 자연어 처리(NLP) 작업에서 깊이 있는 이해와 생성을 가능하게 하는 거대한 파라미터 격자에 기반한 모델입니다. 이 모델들은 Transformer 아키텍처에서 개발되어, self-attention 메커니즘을 이용하여 텍스트 시퀀스를 효과적으로 처리하며, 과거의 통계적 언어 모델 및 신경망(NN) 모델에서 발전한 형태입니다. 더불어, Retrieval-Augmented Generation (RAG), Knowledge Graph 및 Prompt Engineering과 같은 통합 방법론을 통해 LLMs의 한계가 극복될 수 있는 가능성을 조사합니다.

- **Performance Highlights**: 이 논문은 LLMs에서의 주요 성과로서, 생성적 AI와 외부 지식 기반의 통합을 통해 모델의 정확성과 데이터 맥락화 및 계산 비용 절감 등을 달성할 수 있는 기회를 제공합니다. 또한, 최근 연구 동향 및 통합을 통한 LLMs의 응용 가능성을 식별하며, 이러한 접근 방법이 다양한 산업에서 AI 기술의 실질적 배치를 지원할 수 있는 방안을 제안합니다. 연구 결과는 LLMs의 현재 연구 상태를 개괄적으로 제공하며, 향후 발전에 필요한 실제적인 지침을 제시합니다.



### Hallucination Mitigation using Agentic AI Natural Language-Based Frameworks (https://arxiv.org/abs/2501.13946)
Comments:
          18 pages, 6 figures

- **What's New**: 본 연구는 여러 전문 AI 에이전트를 조율하여 Generative AI 모델에서 발생하는 hallucination 문제를 완화할 수 있는 방법을 제안합니다. 한 가지 주요 목표는 Natural Language Processing(NLP) 시스템을 통해 이러한 에이전트 간의 원활한 상호작용을 지원하는 것입니다. 이를 통해 신뢰할 수 있는 AI 시스템 구축이 가능하다는 점을 강조하고 있으며, 새로운 Key Performance Indicators(KPIs)를 활용해 hallucination의 수준을 평가합니다.

- **Technical Details**: 이 논문에서는 310개의 prompt를 설계하여 hallucination을 유도하고, 각 에이전트가 distinct한 large language 모델을 사용하여 이러한 출력물의 검토 및 수정 프로세스를 수행하는 multi-agent pipeline을 제안합니다. Open Voice Network(OVON) 프레임워크를 통해 에이전트 간의 오류 검출과 수정 과정을 JSON 메시지를 통해 원활하게 수행할 수 있도록 설계되었습니다. 또한, 다양한 KPIs는 hallucination 완화의 효과를 정량적으로 평가할 수 있는 구조적 프레임워크를 제공합니다.

- **Performance Highlights**: 결과적으로 다수의 전문 에이전트를 통해 hallucination 확률을 체계적으로 감소시킬 수 있으며, 이러한 시스템은 AI 커뮤니티 내에서의 신뢰성을 강화하는 데 기여할 것으로 기대됩니다. 특히, 실험 결과는 multi-agent pipeline이 프로세스를 거치면서 hallucination 점수가 향상되는 것을 시각적으로 나타내며, 이러한 개선이 AI 모델의 신뢰성 향상으로 이어질 수 있음을 보여줍니다.



### Self-Explanation in Social AI Agents (https://arxiv.org/abs/2501.13945)
Comments:
          Extended version of the paper published in International Conference on Intelligent Tutoring Systems, pages 351-360, 2024, Springer. Images corrected, and live deployment, ablation, and precision study results added

- **What's New**: 본 연구에서는 사회적 AI 에이전트가 커뮤니티 멤버와 상호 작용하여 그들의 행동을 변화시키는 방식에 대해 다룹니다. 특히, 온라인 학습 환경에서 AI 사회 보조자가 학습자 간의 상호 작용을 증진할 수 있는 가능성에 주목합니다. AI 보조자가 학습자에게 자신의 기능을 설명함으로써 투명성과 신뢰성을 높일 수 있는 방법을 제시합니다.

- **Technical Details**: 연구에서는 AI 사회 보조자의 자기 모델(self-model)을 통해 반성(introspection)을 활용한 자기 설명(self-explanation) 기법을 개발합니다. 이 자기 모델은 AI 에이전트가 작업을 수행하기 위해 지식을 어떻게 사용하는지를 정의하는 기능적 모델로 포착됩니다. 자기 설명 생성 과정에서는 Chain of Thought 기법을 통해 자신을 반성하고, ChatGPT를 이용해 그 기능에 대한 설명을 제공합니다.

- **Performance Highlights**: AI 사회 보조자의 자기 설명을 완전성과 정확성 측면에서 평가하고, 실제 수업에서 이 기술의 배포에 대해 보고합니다. 연구 결과, 이러한 자기 설명 기능이 학습자들의 신뢰를 증진시키는데 긍정적인 영향을 미쳤음을 확인했습니다.



### Fanar: An Arabic-Centric Multimodal Generative AI Platform (https://arxiv.org/abs/2501.13944)
- **What's New**: Fanar는 아랍어 중심의 다중 모드 생성 AI 시스템을 위한 플랫폼으로, 언어, 음성 및 이미지 생성 작업을 지원합니다. Fanar의 핵심에는 두 가지 강력한 아랍어 대형 언어 모델인 Fanar Star와 Fanar Prime이 있으며, 이들은 유사 모델군에서 검증된 벤치마크에서 최고의 성능을 자랑합니다.

- **Technical Details**: Fanar Star 모델은 70억 개의 매개변수(parameter)를 가지고 있으며, 거의 1조 개의 깨끗하고 중복되지 않은 아랍어, 영어 및 코드 토큰으로 처음부터 훈련되었습니다. 반면 Fanar Prime 모델은 90억 개의 매개변수를 가지고 있으며, 동일한 1조 개의 토큰 세트를 기반으로 Gemma-2 9B 모델을 통해 지속적으로 훈련되었습니다.

- **Performance Highlights**: Fanar 플랫폼은 종교 관련 프롬프트를 처리하기 위한 맞춤형 이슬람 검색 증강 생성(RAG) 시스템과 최근 정보 요약을 위한 최신 RAG 기능을 제공합니다. 추가적으로, 플랫폼은 다양한 아랍어 방언을 지원하는 이중 언어 음성 인식, 지역 특성을 잘 반영한 음성 및 이미지 생성을 제공합니다.



### Language Representation Favored Zero-Shot Cross-Domain Cognitive Diagnosis (https://arxiv.org/abs/2501.13943)
- **What's New**: 이 논문은 언어 표현 기반의 제로 샷 크로스 도메인 인지 진단(Language Representation Favored Zero-shot Cross-domain Cognitive Diagnosis, LRCD) 접근법을 제안합니다. 기존의 인지 진단 모델들은 특정 도메인에 맞춰 모델을 훈련해야 하며, 이는 다양한 과목이나 교육 플랫폼에서의 직접적인 응용을 어렵게 만듭니다. LRCD는 텍스트 설명을 사용하여 학생, 과제 및 개념의 프로필을 만들고, 이를 통일된 언어 공간에 벡터화하여 인지 진단 모델과 통합하는 방법을 제안합니다.

- **Technical Details**: LRCD는 학생과 과제의 행위 패턴을 분석하고, 각 도메인에서 이를 텍스트로 설명합니다. 이렇게 생성된 프로필은 최신 텍스트 임베딩 모듈을 통해 통합 언어 공간의 벡터로 변환됩니다. 그러나 언어 공간과 인지 진단 공간 간의 불일치를 해결하기 위해, LRCD는 두 공간 간의 매핑을 학습하는 언어-인지 맵퍼를 제안합니다. 이를 통해 기존의 인지 진단 모델과 통합하여 효율적으로 훈련할 수 있습니다.

- **Performance Highlights**: LRCD는 다양한 실제 데이터셋에 대해 제로 샷 성능을 commendable하게 달성할 수 있으며, 특정 경우에는 기존의 인지 진단 모델과 경쟁할 만한 성능을 보여줍니다. 학계 및 교육 현장에서 학생들의 과목 간 차이를 분석하는 데에도 유용한 인사이트를 제공합니다. 흥미롭게도, 과학 과목에서 생성된 데이터는 다양한 목표 도메인에서 더 나은 전이 성능을 보이며, 고등 교육 수준의 학생 데이터를 기반으로 할 경우 저학년 학생들에게도 더 큰 전이 가능성을 보입니다.



### Mitigating GenAI-powered Evidence Pollution for Out-of-Context Multimodal Misinformation Detection (https://arxiv.org/abs/2501.14728)
Comments:
          12 pages, 11 figures

- **What's New**: 이번 연구는 대형 Generative Artificial Intelligence (GenAI) 모델의 오용이 온라인 정보 보안에 미치는 영향을 다룹니다. 특히, Out-of-Context (OOC) 멀티모달 허위 정보 탐지에서 GenAI로 오염된 증거(evidence)를 처리하는 방법을 제안합니다. 기존 연구들은 주로 스타일 재작성(stylistic rewriting)을 통해 언어적 단서를 숨기는 방식으로 GenAI 오염을 모사했지만, 실질적인 증거 수준의 오염을 간과했습니다.

- **Technical Details**: 연구에서는 오염된 증거가 기존 OOC 탐지기의 성능에 미치는 영향을 분석했고, 그 결과 성능이 9% 이상 저하되는 것을 발견했습니다. 이에 대응하기 위해 cross-modal evidence reranking 및 cross-modal claim-evidence reasoning라는 두 가지 전략을 제안했습니다. 이 방법들은 오염된 증거와 함께 존재하는 기존 탐지기의 강건성을 강화하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 두 개의 벤치마크 데이터셋에서 수행된 광범위한 실험을 통해 제안된 전략들이 오염된 증거에 맞서 기존의 OOC 탐지기를 효과적으로 향상시킨다는 점을 입증했습니다. 이로 인해 정보 탐색(application) 분야에서 GenAI의 부정적 영향을 줄이는 데 기여할 수 있습니다.



### The Karp Datas (https://arxiv.org/abs/2501.14705)
Comments:
          Accepted to the 4th workshop on mathematical reasoning and AI at NeurIPS 2024

- **What's New**: 이 논문은 Large Language Models (LLMs)의 수학적 추론 능력을 이해하는 데 중점을 두고 있으며, 이를 위한 새로운 Karp 데이터셋을 소개합니다. Karp 데이터셋은 NP-완전성 증명의 구체적인 사례들로 구성되어 있으며, 다양한 난이도로 이루어져 있어 LLM의 성능을 평가하는 데 사용될 수 있습니다. 이는 기존 데이터셋이 단순한 숫자나 기호의 결과에 집중하는 것에 비해, 더 복잡한 논리적 사고를 요구하는 연구입니다.

- **Technical Details**: Karp 데이터셋은 NP-난해성 증명을 위한 다양한 변환 과정을 문서화한 자연어 설명들로 구성되어 있습니다. 데이터셋은 명확한 정의와 함께 정확성을 증명하는 구조화된 템플릿을 따릅니다. 모든 변환 예시는 저명한 문헌에서 출처를 두고 있으며, 각 변환의 길이는 평균 약 2000자가 넘는 긴 설명을 포함하고 있습니다. 이러한 구조적이고 교육적인 접근 방식은 LLM이 증명을 이해하고 처리하는데 유용하도록 돕습니다.

- **Performance Highlights**: 현재 LLM은 MATH, GSM8K, MGSM 등의 기존 데이터셋에서 뛰어난 성능을 보이고 있지만, NP-완전성 증명의 변환을 다루는 새로운 Karp 데이터셋에 대한 평가가 이루어질 때 더 높은 임계값이 설정될 것으로 기대됩니다. LLM의 학습과 성능 평가에 있어 Karp 데이터셋이 새로운 기준을 제공할 것이며, 특히 높은 수준의 수학적 문제에 대한 이해도를 높이는 데 기여할 수 있습니다.



### Chain-of-Retrieval Augmented Generation (https://arxiv.org/abs/2501.14342)
Comments:
          18 pages

- **What's New**: 이 논문은 CoRAG(Chain-of-Retrieval Augmented Generation)라는 새로운 접근 방식을 소개하여, 관련 정보를 단계적으로 검색하고 추론하는 방법을 통해 최종 답변을 생성합니다. 기존의 RAG 방법들은 단일 검색 단계를 수행하기 때문에 복잡한 쿼리에 대한 대응력이 제한적이었으나, CoRAG는 모델이 현재 상태에 따라 동적으로 쿼리를 재구성할 수 있도록 합니다.

- **Technical Details**: CoRAG는 자동으로 중간 검색 체인을 생성하기 위해 거부 샘플링(rejection sampling)을 활용하여 기존 RAG 데이터셋을 증강하고, 테스트 시간에 계산을 조절하는 다양한 디코딩 전략을 제안합니다. 이 모델은 단계별로 검색하고 추론하는 과정을 명시적으로 교육하여, 복잡한 질문을 해결하기 위한 정보를 반복적으로 검색하는 인간의 문제 해결 방식과 유사한 형태로 동작합니다.

- **Performance Highlights**: 여러 벤치마크 테스트 결과, CoRAG는 특히 다단계 질문 응답(Task)에서 강력한 기준선보다 10포인트 이상의 개선을 보였습니다. KILT 벤치마크에서 CoRAG는 지식 집약적인 다양한 작업에서 새로운 최첨단 성능을 확립하였으며, CoRAG의 스케일링 행동에 대한 포괄적인 분석을 통해 향후 사실 기반 및 근거 모델 개발을 위한 기초를 다진 것으로 평가됩니다.



### Fast Think-on-Graph: Wider, Deeper and Faster Reasoning of Large Language Model on Knowledge Graph (https://arxiv.org/abs/2501.14300)
- **What's New**: 이번 논문에서는 Fast Think-on-Graph (FastToG)라는 혁신적인 패러다임을 제안합니다. FastToG는 그래프 정보를 활용하여 LLM이 '커뮤니티 단위'로 사고하도록 고안되었습니다. 이 접근법은 Community Detection 기법을 사용하여 더 깊은 상관관계를 포착하고, 코스와 파인의 두 단계의 커뮤니티 프루닝을 통해 빠른 정보 검색을 가능하게 합니다.

- **Technical Details**: FastToG는 그래프 내에서 커뮤니티 감지를 통해 추론 체인을 구축하고 Community-to-Text 기법을 활용하여 커뮤니티 구조를 텍스트 형식으로 변환합니다. Local Community Search (LCS)라는 새로운 기법을 도입하여 그래프의 지역 범위에서 커뮤니티를 탐지하고, 효율성을 높이기 위해 모듈성 기반의 코스 프루닝과 LLM 기반의 파인 프루닝을 적용합니다.

- **Performance Highlights**: 실험 결과, FastToG는 이전 방법들에 비해 더 높은 정확도, 더 빠른 추론, 그리고 더 높은 설명 가능성을 보여주었습니다. 커뮤니티 기반의 추론이 추론 체인을 단축시켜 LLM 호출 횟수를 줄일 수 있음을 확인했습니다. 또한 FastToG는 LLM에 대한 검색 과정을 단순화하고 사용자에게 더 나은 설명 가능성을 제공합니다.



### Humanity's Last Exam (https://arxiv.org/abs/2501.14249)
Comments:
          25 pages, 6 figures

- **What's New**: 이번 논문에서는 Humanity's Last Exam (HLE)라는 새로운 다중 모드 벤치마크를 소개합니다. 기존의 벤치마크가 충분한 도전을 제공하지 못하는 상황에서, HLE는 3,000개의 질문을 포함하여 인간 지식의 최전선에서 설계되었습니다. 본 연구는 과거의 전통적인 벤치마크에서 LLM의 성능을 측정하는 데 있어 격차를 해소하고자 합니다.

- **Technical Details**: HLE는 수학, 인문학, 자연 과학 등 다양한 주제를 포함한 질문들로 구성되어 있으며, 선택형 및 단답형 질문 형식을 제공합니다. 질문은 명확하고 검증 가능하며, 단순한 인터넷 검색으로는 빠르게 답변할 수 없습니다. 데이터셋은 공신력 있는 주제 전문가들에 의해 개발되었으며, 정확한 답변과 함께 상세한 해설도 포함되어 있습니다.

- **Performance Highlights**: 최신 LLM은 HLE에서 10% 미만의 정확도를 보이며, 이는 현재의 LLM 능력과 전문가 인간의 성능 사이에 존재하는 큰 격차를 보여줍니다. AI 시스템이 여러 영역에서 인간 전문가의 성능에 접근함에 따라, 이러한 정밀한 측정은 연구 및 정책 결정을 위한 중요한 토대가 됩니다. HLE의 높은 성과는 클로즈드 엑센들 테스트에서 전문가 수준의 능력을 나타낼 것임을 시사합니다.



### QuanTaxo: A Quantum Approach to Self-Supervised Taxonomy Expansion (https://arxiv.org/abs/2501.14011)
- **What's New**: 이번 논문에서는 전통적인 taxonomy 확장 접근 방식을 역전시키는 QuanTaxo라는 혁신적인 양자 기반 프레임워크를 소개합니다. QuanTaxo는 실체를 양자 공간에 인코딩하여 계층적 다의성을 효과적으로 모델링하며, 기존의 고전적인 단어 임베딩의 한계를 초월합니다. 이 프레임워크는 Hilbert 공간의 원리를 활용하여 실체 간의 간섭 효과를 포착하며, 더 풍부하고 미묘한 표현을 제공합니다.

- **Technical Details**: QuanTaxo는 자가 감독형 학습 프레임워크를 통해 세부 taxonomy에서 학습한 양자 표현을 사용하여 확장 과정을 자동화합니다. 양자 얽힘 기반 접근 방식과 정밀한 수학적 기법을 적용하여 부모-자식 엔티티 간의 관계를 명확히 정량화합니다. 두 가지 변형 모델인 Quant-Sup와 Quant-Mix를 제안하여, 각각 잠재 개념의 선형 조합 및 단어 상태의 가중 조합으로 실체의 양자 표현을 구성합니다.

- **Performance Highlights**: QuanTaxo는 4개의 실제 기준 데이터세트에서 고전적인 임베딩 모델을 크게 능가하는 성능을 보여주었으며, 정확도에서 18.45%, Mean Reciprocal Rank에서 20.5%, Wu & Palmer 지표에서 17.87%의 개선을 달성했습니다. 제안된 프레임워크의 우수성은 포괄적인 ablation 연구와 사례 연구를 통해 강조되었습니다.



### GaussMark: A Practical Approach for Structural Watermarking of Language Models (https://arxiv.org/abs/2501.13941)
- **What's New**: 최근 Large Language Models (LLMs)의 발전은 자연어 처리(NLP) 작업에 큰 개선을 가져왔지만, 사람과 같은 품질의 텍스트 생성을 가능하게 하는 이 기술은 윤리적 및 운영상의 우려를 야기하고 있습니다. 이에 따라, LLM으로 생성된 텍스트를 확인할 수 있는 수단인 워터마킹(watermarking) 기법 개발에 대한 연구가 이루어졌습니다. 현재의 기존 워터마킹 기법은 생성 지연(generation latency)과 탐지 시간(detection time), 텍스트 품질 저하 등 여러 측면에서 실용적이지 않은 경우가 많았으며, 이 문제들을 해결하기 위한 새로운 접근 방식이 필요합니다.

- **Technical Details**: 이 연구에서는 GaussMark라는 새로운 워터마킹 기법을 소개합니다. 이 방법은 구현하기 간단하고 효율적이며, 형태학적인 워터마크(structural watermark)를 모델의 가중치(weights) 자체에 내장합니다. Gaussian 독립성 테스트를 기반으로 하고 있으며, LLM의 가중치에 소량의 Gaussian 노이즈를 추가하여 워터마킹을 수행합니다. 이 과정은 통계적으로 검출 가능하도록 설계되었으며, 비밀 키를 가진 제공자가 이 텍스트가 자신의 모델에 의해 생성되었다는 것을 확인할 수 있게 합니다.

- **Performance Highlights**: GaussMark는 광범위한 실험을 통해 신뢰성과 효율성을 입증하였으며, 삽입, 삭제, 치환, 왕복 번역(roundtrip translation)과 같은 복원력 있는 다양한 왜곡에 대해 비교적 강력함을 보여줍니다. 이 방법은 모델 품질 손실 없이 거의 무제한의 성능 범위를 제공합니다. 또한, 우리의 접근 방식은 생성 과정에서 추가적인 지연을 발생시키지 않아 실제 적용에 적합합니다.



### Evaluating Computational Accuracy of Large Language Models in Numerical Reasoning Tasks for Healthcare Applications (https://arxiv.org/abs/2501.13936)
Comments:
          13 pages, 1 figure, 2 tables

- **What's New**: 이번 연구는 의료 부문에서 사용되는 대규모 언어 모델(Large Language Models, LLMs)의 숫자 추론 능력에 대한 최초의 평가를 수행했습니다. LLMs는 자연어 이해 및 생성에서의 놀라운 성능을 보여주지만, 높아진 절차적 요구에 맞춘 숫자 추론은 여전히 미흡하다는 점에 주목합니다. 이 연구가 다룬 1,000개의 숫자 문제를 통해 실제 의료 환경에서의 수치적 추론 능력을 탐색하게 되었습니다.

- **Technical Details**: 연구에서는 GPT-3 아키텍처를 기반으로 한 정제된 LLM의 성능을 평가하기 위해 신중하게 수집된 데이터셋을 사용했습니다. 방법론에는 프롬프트 엔지니어링(prompt engineering), 사실 확인 파이프라인(fact-checking pipelines) 통합, 정규화 기법(regularization techniques) 적용이 포함되어 모델의 정확도 및 일반화 능력을 향상시켰습니다. 모델의 효율성을 평가하기 위해 precision, recall, F1-score와 같은 주요 메트릭(metrics)을 사용했습니다.

- **Performance Highlights**: 결과적으로 LLM은 84.10%의 전체 정확도를 기록했으며, 간단한 숫자 작업에서는 향상된 성능을 보였지만, 다단계 추론(multi-step reasoning)에서는 어려움이 있었습니다. 사실 확인 파이프라인의 통합이 정확도를 11% 향상시켜 검증 메커니즘의 중요성을 강조했습니다. 이 연구는 의료 분야에서 숫자 추론에 대한 LLM의 잠재력을 부각시키며, 임상 환경에서의 중요한 의사 결정 지원을 위한 LLM의 추가 개선 방향 또한 제시합니다.



### ExLM: Rethinking the Impact of [MASK] Tokens in Masked Language Models (https://arxiv.org/abs/2501.13397)
Comments:
          29 pages, 12 figures

- **What's New**: 이번 연구에서는 Masked Language Models (MLMs)의 [MASK] 토큰 사용이 모델 성능에 미치는 영향을 분석합니다. 기존 연구들은 주로 [MASK]의 비현실적인 문제에 집중했으나, 이번 논문에서는 맥락 왜곡 문제와 함께 이를 다룹니다. 제안된 개선된 맥락 MLM은 ExLM으로, [MASK] 토큰을 활용해 더 풍부한 의미 정보를 캡처하도록 설계되었습니다.

- **Technical Details**: ExLM 모델은 입력 맥락 내에서 [MASK] 토큰을 확대하여 고유한 상태 간의 의존성을 모델링합니다. 이를 통해 문맥 용량이 증가하고, 전이 학습(transfer learning) 시 부정확한 의미를 줄이는 데 도움이 됩니다. 모델 실험은 BERT 기초 모델과 유사한 설정으로 수행되었으며, 14세트의 MLM에 대해 다양한 매개변수를 사용해 훈련합니다.

- **Performance Highlights**: ExLM은 텍스트 모델링 및 SMILES 모델링 작업에서 성능 향상을 크게 보여주었습니다. 추가 분석을 통해 ExLM이 맥락 향상을 통해 의미 표현을 강화시키고, MLM에서 자주 발생하는 다중 의미 문제를 효과적으로 줄임을 확인했습니다. 따라서 ExLM은 MLM의 성능 향상에 기여할 수 있는 가능성이 높은 모델입니다.



New uploads on arXiv(cs.IR)

### Knowledge Graphs Construction from Criminal Court Appeals: Insights from the French Cassation Cour (https://arxiv.org/abs/2501.14579)
- **What's New**: 이 논문은 법원 판결과 같은 비구조적 데이터(unstructured data)를 구조화된 형태(structured form)로 정확하고 신뢰성 있게 표현하는 것을 목표로 합니다. 최근 생성적 인공지능(generative AI)의 발전으로, 언어 모델링(language modeling)을 통해 텍스트를 지식 그래프(knowledge graphs)로 변환할 수 있는 기회가 열렸습니다. 연구의 주제는 프랑스 카시션 법원(Cassation Court)의 항소 사례를 대상으로 지식 그래프를 구축하는 프레임워크(framework)를 제시하는 것입니다.

- **Technical Details**: 제시된 프레임워크는 특정 도메인에 맞춘 온톨로지(ontology)와 파생된 데이터셋(derived dataset)을 포함하여 구조화된 법률 데이터representation을 위한 기초를 제공합니다. 이 연구는 비구조적 법률 텍스트를 효과적으로 구조화하여 분석할 수 있는 방법을 제안하고 있습니다. 이러한 접근은 법률 분야에서 데이터의 활용성을 높이는 데 기여할 수 있습니다.

- **Performance Highlights**: 이 프레임워크를 통해 법률 데이터의 분석 및 모델링(modeling)의 새로운 가능성이 열리며, 이는 법률 전문가들이 보다 향상된 의사 결정을 내릴 수 있도록 지원합니다. 또한, 생성적 모델을 통해 다양한 법률 사례를 효과적으로 시각화할 수 있는 기회를 제공합니다. 이러한 성과는 특히 법률 기술(legal tech) 분야에서 큰 영향력을 미칠 전망입니다.



### On Correlating Factors for Domain Adaptation Performanc (https://arxiv.org/abs/2501.14466)
- **What's New**: 이 논문은 Dense retrievers의 도메인 적응(doman adaptation) 성능을 향상시킬 수 있는 변수들을 분석합니다. 기존 연구에서 언급된 변수들을 확장하여, 생성된 쿼리의 종류 분포와 도메인 유사성을 조사합니다. 특히, 테스트 문서와 유사한 도메인의 쿼리를 생성할 경우 성능 개선을 가져온다는 점을 강조합니다.

- **Technical Details**: Dense retrieval (DR) 방식은 신속하고 쉽게 인덱싱할 수 있으나, 입력을 의미 있는 벡터 공간으로 맵핑하는 것이 도전적입니다. 논문에서는 GPL 및 InPars 방법을 사용하여 도메인 적응 연구를 진행하며, 생성된 쿼리의 어휘와 유형 배포의 중복성을 평가합니다. 실험에서는 BEIR 및 LoTTE 데이터셋을 활용하여 다양한 도메인에서 성능을 비교합니다.

- **Performance Highlights**: 실험 결과, GPL 방법이 LoTTE 데이터셋에서 NDCG@10에서 4.6 포인트 개선된 성능을 보였습니다. 생성된 쿼리의 유형 분포 엔트로피가 도메인 적응 성능과 양의 상관관계를 보이며, 유사한 어휘를 가진 쿼리가 중요한 역할을 한다는 점이 밝혀졌습니다. 그러나 InPars 방법에서는 생성된 쿼리의 중복성이 오히려 두 프레임워크 모두에 부정적인 영향을 미친다는 점이 흥미롭습니다.



### Interpretability Analysis of Domain Adapted Dense Retrievers (https://arxiv.org/abs/2501.14459)
- **What's New**: 본 논문에서는 dense retriever의 도메인 적응 후 모델의 동작 변화를 분석하기 위해 Integrated Gradients (IG) 프레임워크를 활용한 설명 방법을 제안합니다. 이를 통해 FIQA와 TREC-COVID 데이터셋에서 쿼리 및 문서 토큰에 대한 입력 기여도를 시각화하여 도메인 적응의 영향을 평가합니다. 특히, 도메인 적응된 모델이 더 많은 인도메인 용어에 집중한다는 점이 발견되었습니다.

- **Technical Details**: 이 연구는 Dense Retriever 모델을 다루며, 문서의 사전 인덱싱을 이용하여 쿼리와 문서 임베딩 간의 내적 유사도를 기반으로 상위 K개의 문서를 검색하는 방식을 채택합니다. DistilBERT를 사용하여 쿼리와 문서의 임베딩을 생성하고, Integrated Gradients를 통해 쿼리 및 문서 기여도를 계산합니다. 연구에서 사용된 데이터셋은 FIQA와 TREC-COVID이며, 두 데이터셋에서 도메인 적응의 효과를 분석하기 위한 방법론을 설계했습니다.

- **Performance Highlights**: 연구 결과, 도메인 적응된 모델은 인도메인 용어에 더 많은 집중을 보이며, 이는 기존 비적응 모델이 간과하는 언어적 요소를 반영합니다. Integrated Gradients를 통해 얻은 인사이트는 dense retriever의 내부 메커니즘을 분석하는 데 유용하며, 도메인 적응이 모델의 성능과 해석 가능성에 미치는 영향을 강조합니다. 이러한 분석은 향후 정보 검색(Information Retrieval) 연구에 있어 중요한 기여를 할 것으로 보입니다.



### Remining Hard Negatives for Generative Pseudo Labeled Domain Adaptation (https://arxiv.org/abs/2501.14434)
- **What's New**: 이 논문은 Dense retrievers가 다소 제한적인 도메인 적응 방법을 개선하는 새로운 방법을 제안합니다. Generative Pseudo Labeling (GPL) 기법을 사용하여 모델의 지식을 전이하고, 하드 네거티브를 갱신함으로써 더 나은 하드 네거티브의 채굴을 가능하게 합니다. 이로 인해, 본 연구에서는 LoTTE 및 BEIR 데이터세트에서 성능이 저조한 기존 접근법을 개선할 수 있음을 입증합니다.

- **Technical Details**: Dense passage retrieval은 현대 NLP 파이프라인에서 중요한 역할을 하는 기법으로, 쿼리와 문서를 저차원 벡터(embeddings)로 표현하여 유사성 비교를 통해 관련성을 측정합니다. 논문에서는 하드 네거티브 마이닝(hard-negative mining)을 통해 생성된 쿼리와 긍정적인 쌍을형성하여 성능을 향상시키기 위한 방법론적 근거를 제시합니다. 이 연구에서는 R-GPL 접근법을 통해 도메인 적응 모델이 훈련하는 동안 더 관련성 높은 하드 네거티브를 사용할 수 있음을 입증합니다.

- **Performance Highlights**: 저자들은 14개의 BEIR 데이터셋 중 13개와 12개의 LoTTe 데이터셋 중 9개에서 도메인 적응 모델의 랭킹 성능이 향상된 것을 보여줍니다. 이 연구는 처음으로 LoTTE 벤치마크에서 GPL을 평가하고, 도메인 적응 중 하드 네거티브를 지속적으로 갱신함으로써 성능의 개선을 이루었습니다. 이러한 결과는 도메인 적응이 훈련 데이터를 보다 효율적으로 활용할 수 있는 가능성을 보여줍니다.



### Handling Heterophily in Recommender Systems with Wavelet Hypergraph Diffusion (https://arxiv.org/abs/2501.14399)
- **What's New**: 본 논문에서는 하이퍼그래프 기반 추천 시스템의 진화를 위한 새로운 FWHDNN(퓨전 기반 웨이브릿 하이퍼그래프 확산 신경망) 프레임워크를 소개합니다. 이 모델은 이질적(hheterophily) 패턴과 다차원 사용자-아이템 상호작용을 포착하는 데 필요한 주요 세 가지 구성 요소를 포함합니다. 제안하는 방법은 다양한 클래스 레이블에 적응할 수 있는 메시지 패싱을 지원하는 이질성 인식 하이퍼그래프 확산을 활용합니다.

- **Technical Details**: FWHDNN은 세 가지 주요 구성 요소로 이루어져 있습니다. 첫 번째는 Cross-Difference Relation Encoder로, 다양한 클래스 레이블에 적응하는 메시지 패싱을 통해 이질적 패턴을 모델링합니다. 두 번째는 Multi-level Cluster-wise Encoder로, 웨이브릿 변환을 활용하여 다중 스케일의 구조적 관계를 포착합니다. 마지막으로, 통합된 다중 모달 융합 메커니즘을 통해 구조적 및 텍스트 정보를 결합합니다.

- **Performance Highlights**: FWHDNN은 실제 데이터셋에서 광범위한 실험을 통해 기존의 최첨단 방법들을 초월하는 정확성과 강인성 및 확장성을 보여줍니다. 특히, 사용자와 아이템 간의 고차원 연결 관계를 효과적으로 포착하는 데 강점을 지니고 있습니다. 이러한 성능은 FWHDNN이 복잡한 추천 환경을 효과적으로 처리할 수 있는 가능성을 시사합니다.



### Chain-of-Retrieval Augmented Generation (https://arxiv.org/abs/2501.14342)
Comments:
          18 pages

- **What's New**: 이 논문은 CoRAG(Chain-of-Retrieval Augmented Generation)라는 새로운 접근 방식을 소개하여, 관련 정보를 단계적으로 검색하고 추론하는 방법을 통해 최종 답변을 생성합니다. 기존의 RAG 방법들은 단일 검색 단계를 수행하기 때문에 복잡한 쿼리에 대한 대응력이 제한적이었으나, CoRAG는 모델이 현재 상태에 따라 동적으로 쿼리를 재구성할 수 있도록 합니다.

- **Technical Details**: CoRAG는 자동으로 중간 검색 체인을 생성하기 위해 거부 샘플링(rejection sampling)을 활용하여 기존 RAG 데이터셋을 증강하고, 테스트 시간에 계산을 조절하는 다양한 디코딩 전략을 제안합니다. 이 모델은 단계별로 검색하고 추론하는 과정을 명시적으로 교육하여, 복잡한 질문을 해결하기 위한 정보를 반복적으로 검색하는 인간의 문제 해결 방식과 유사한 형태로 동작합니다.

- **Performance Highlights**: 여러 벤치마크 테스트 결과, CoRAG는 특히 다단계 질문 응답(Task)에서 강력한 기준선보다 10포인트 이상의 개선을 보였습니다. KILT 벤치마크에서 CoRAG는 지식 집약적인 다양한 작업에서 새로운 최첨단 성능을 확립하였으며, CoRAG의 스케일링 행동에 대한 포괄적인 분석을 통해 향후 사실 기반 및 근거 모델 개발을 위한 기초를 다진 것으로 평가됩니다.



### Multi-stage Large Language Model Pipelines Can Outperform GPT-4o in Relevance Assessmen (https://arxiv.org/abs/2501.14296)
Comments:
          WebConf'25, WWW'25

- **What's New**: 이번 연구에서는 LLM(대형 언어 모델)를 기반으로 한 모듈형 분류 파이프라인을 제안합니다. 이 접근 방식은 관련성 평가 작업을 여러 단계로 나누어 각기 다른 프롬프트와 모델을 활용합니다. TREC Deep Learning에 적용한 결과, GPT-4o mini보다 18.4% 더 높은 Krippendorff의 α 정확도를 보이며, 비용은 입력 토큰 100만 개당 약 0.2달러로 더 효율적이고 확장 가능한 솔루션을 제공합니다.

- **Technical Details**: 제안된 파이프라인은 이진 분류와 세 가지 수준의 상세한 관련성 라벨링으로 구성됩니다. 처음에는 LLM이 이진 분류를 수행하고, 그 다음에는 좀 더 세분화된 관련성 라벨링을 진행하여 불필요한 분류를 줄입니다. 이 과정에서 GPT-4o와 GPT-4o mini를 분석에 사용하고, Llama와 Claude 같은 다양한 모델을 비교하여 정확도와 비용의 차이를 확인합니다.

- **Performance Highlights**: 결과적으로, 제안된 파이프라인이 최신 LLM과 유사한 라벨링 정확도를 제공하는 동시에 비용은 현저히 낮춰 고품질 데이터 주석을 위한 확장 가능한 솔루션을 제공합니다. 파이프라인 방법론을 통해, GPT-4o의 정확도는 9.7% 개선되었으며, 여러 모델을 사용하는 다단계 접근 방식이 특히 성과를 거두었습니다.



### Hierarchical Time-Aware Mixture of Experts for Multi-Modal Sequential Recommendation (https://arxiv.org/abs/2501.14269)
Comments:
          Accepted to WWW 2025

- **What's New**: 이 논문에서는 Multi-modal Sequential Recommendation (SR)에서의 한계를 극복하기 위해 Hierarchical time-aware Mixture of Experts (HM4SR)라는 새로운 방법을 제안합니다. 기존 방법들은 주로 사용자 선호도의 변화와 관련한 다채로운 모달리티 데이터를 활용하지 않고 있으나, HM4SR은 두 가지 레벨의 Mixture of Experts (MoE)를 도입하여 이러한 모달의 중요 정보를 추출합니다. 또한, 시간 관련 임베딩을 통해 사용자 관심의 변화를 더 효과적으로 모델링하는 접근을 취하고 있습니다.

- **Technical Details**: HM4SR의 구조는 두 개의 MoE로 구성되어 있으며, 첫 번째 MoE인 Interactive MoE는 다중 모달 데이터에서 사용자 관심과 관련된 핵심 정보를 추출합니다. 두 번째 MoE인 Temporal MoE는 명시적인 시간 정보를 사용하여 동적인 사용자 관심을 캡처하고, 이를 통해 멀티모달 학습에서의 시간 관련 정보를 통합하는 방식으로 설계되었습니다. 또한, 세 가지 보조 감독 작업(CP, IDCL, PCL)을 통해 더욱 풍부한 학습 신호를 제공합니다.

- **Performance Highlights**: HM4SR은 네 개의 공개 데이터셋에 대한 광범위한 실험을 통해 기존의 여러 최첨단 방법들과 비교하여 그 효과성을 입증하였습니다. 이 방법은 다중 모달 정보의 풍부함을 활용하여 사용자 관심의 진정한 모델링을 지원하고, 시계열 정보를 통해 동적인 관심 변화를 효과적으로 캡처하는 데 성공했습니다. 전반적으로 HM4SR은 다중 모달 SR 분야에서 중요한 기여를 할 것으로 보입니다.



### Pre-train and Fine-tune: Recommenders as Large Models (https://arxiv.org/abs/2501.14268)
Comments:
          Accepted by WWW2025

- **What's New**: 이 논문에서는 다중 도메인 학습의 한계를 극복하기 위해 추천 시스템을 대규모 사전 훈련 모델로 간주하고 이를 세밀하게 조정(fine-tune)하는 방법을 제안합니다. 기존의 다중 도메인 학습에서는 새로운 작업을 추가할 때 모델 전체를 재훈련해야 하지만, 본 연구에서는 각 도메인에 대해 사전 훈련된 모델의 소규모 조정을 통해 효율적으로 모델을 개선할 수 있습니다. 또한, 정보 병목(information bottleneck)의 관점에서 추천 시스템의 조정 기법에 대한 이론적 설명을 제공합니다.

- **Technical Details**: 제안된 방법론은 정보 인식 적응 커널(Information-Aware Adaptive Kernel, IAK)을 활용하여 사전 훈련된 추천 모델을 미세 조정하는데 중점을 둡니다. IAK는 일반적인 비즈니스 지식을 압축하고 주어진 하위 작업에 대한 특정 지식을 학습하며, 관련 없는 지식은 잊어버리는 구조를 취합니다. 이 과정은 사용자 맞춤형 추천 결과를 보다 정확하게 도출할 수 있도록 하며, 다양한 도메인이나 조건에서 유연한 추천을 가능하게 합니다.

- **Performance Highlights**: 다양한 오프라인 및 온라인 실험에서 제안된 접근 방식의 우수성이 입증되었습니다. 특히, 이 연구에서 제안한 IAK 기법은 수억 명의 사용자를 대상으로 한 대규모 온라인 음식 플랫폼에서 실제로 배포되어 상당한 비즈니스 이익도 창출하였습니다. 또한, 연구팀은 추천 시스템에서 발견된 두 가지 잠재적 문제점에 대해 논의하며, 이 문제들을 해결하기 위한 탐색을 제공합니다.



### Do LLMs Provide Consistent Answers to Health-Related Questions across Languages? (https://arxiv.org/abs/2501.14719)
Comments:
          9 pages. Short paper appeared at 47th European Conference on Information Retrieval (ECIR 2025)

- **What's New**: 이번 연구는 다양한 언어에서 건강관련 질문에 대한 LLM(대형 언어 모델)의 응답 일관성을 분석하였습니다. 영어, 독일어, 터키어, 중국어의 다국적 맥락에서 LLM의 답변을 검토하고, 언어에 따라 다양한 질병 유형별로 분류된 건강 관련 질문 데이터를 확장했습니다. 이 연구의 주요 기여는 다국어 건강 질의 데이터셋과 이질적 언어 간의 비교를 가능하게 하는 새로운 프롬프트 기반 평가 작업 흐름을 도입한 것입니다.

- **Technical Details**: 연구에서는 NER(명명 개체 인식)을 사용하여 샘플을 질병별로 분류하고, 개편된 HealthFC 데이터셋을 사용해 LLM에 질병 분류에 따라 프롬프트를 제공하였습니다. 또한, 원래 영어와 독일어로 제공되던 데이터를 터키어와 중국어로 번역하고, 이를 통해 언어 간에 응답 일관성을 평가하기 위한 프롬프트 기반 평가 프레임워크가 개발되었습니다. 이러한 방법론은 LLM의 여러 언어에서의 응답을 심층적으로 분석할 수 있도록 하였습니다.

- **Performance Highlights**: 이 연구에서 발생한 주요 결과 중 하나는 LLM의 다양한 언어에서의 일관성 부족으로 인해 정확하지 않은 의료 정보를 퍼뜨릴 위험이 있다는 것입니다. 파싱(Parsing)된 응답 간의 비교를 통해 일관성의 측정을 시도했으며, Kappa Scores를 통해 응답의 일관성 정도를 평가하였습니다. 특히, 터키어(TR)와 중국어(ZH)에서 LLM과 인간 평가자 간의 상당한 일치도가 발견되었고, 독일어(DE)에서도 중간 정도의 일치가 관찰되었습니다.



### CAMEO: Autocorrelation-Preserving Line Simplification for Lossy Time Series Compression (https://arxiv.org/abs/2501.14432)
Comments:
          14 pages, 13 figures

- **What's New**: 이번 논문에서는 통계적 특성에서 보장을 제공하는 새로운 손실 압축 방법인 CAMEO를 제안합니다. 이 방법은 시계열의 자기상관 함수(ACF)와 부분 자기상관 함수(PACF)를 보존하면서 효율적으로 데이터 압축을 수행합니다. CAMEO는 압축 데이터가 주어진 최대 편차 내에서 ACF와 PACF를 유지하도록 설계되었으며, 기존에는 없었던 미세한 오류 범위를 설정할 수 있는 가능성을 제공합니다.

- **Technical Details**: CAMEO 압축기는 이터레이티브 그리디 접근법을 사용하여 ACF 및 PACF에 미치는 영향이 큰 점들을 제거합니다. 또한, ACF와 PACF 통계치를 점진적으로 업데이트하기 위해 적은 수의 집계를 유지합니다. 성능 향상을 위해 블로킹(blocking) 및 병렬화(parallelization) 전략을 활용하며, 이러한 방식은 시간 복잡성을 줄이는 데 기여합니다.

- **Performance Highlights**: CAMEO는 평균적으로 2배의 압축 비율을 향상시키며, 일부 데이터 세트에서는 최대 54배까지 증가합니다. 이 방법은 ACF의 편차를 유지하면서 예측 정확도를 개선하거나 최소한 유지하는 데 기여합니다. 또한, 이상 탐지(anomaly detection)의 정확도 또한 향상시키는 promising한 결과를 보여주었습니다.



### Revisiting Applicable and Comprehensive Knowledge Tracing in Large-Scale Data (https://arxiv.org/abs/2501.14256)
- **What's New**: 이 논문에서는 기존 Deep Knowledge Tracing (DKT) 모델의 한계를 극복하고자 새로운 모델, DKT2를 제안합니다. DKT2는 최근 개발된 xLSTM 아키텍처를 기반으로 하여 더 나은 지식 상태를 생성할 수 있도록 설계되었습니다. 또한, Rasch 모델을 활용해 입력 표현을 강화하고, Item Response Theory (IRT)를 통해 학습한 지식을 해석합니다. 이로써, DKT2는 지식의 친숙한 부분과 낯선 부분을 구분할 수 있습니다.

- **Technical Details**: DKT2는 교육 심리학에서 유래된 Rasch 모델을 사용하여 과거 상호 작용을 처리하고, xLSTM을 통해 지식을 학습합니다. xLSTM은 기억 용량을 증가시키고, 정보를 더 효율적으로 검색할 수 있도록 하며, 이를 통해 DKT2의 성능을 극대화합니다. DKT2는 학생의 지식 상태와 예측된 문제를 통합하여 포괄적인 지식 상태를 생성합니다. 이 방법은 수집한 데이터를 기반으로 다양한 예측 작업에서 높은 예측 성능을 나타냅니다.

- **Performance Highlights**: 세 개의 대규모 데이터셋에서 DKT2는 17개의 기준 모델을 지속적으로 초월하는 성능을 보였습니다. 특히, 다양한 예측 작업에서 DKT2는 기존 모델의 한계를 극복하며, 실제 교육 환경에 유용할 수 있는 가능성을 보여줍니다. DKT2는 교육적 응용 분야에서 이론과 실제 적용 간의 간극을 해소하는 데 기여할 것으로 기대됩니다.



### MedSlice: Fine-Tuned Large Language Models for Secure Clinical Note Sectioning (https://arxiv.org/abs/2501.14105)
Comments:
          Our code is publicly available on github ( this https URL )

- **What's New**: 이 연구는 공개 소스의 큰 언어 모델(Large Language Models, LLMs)을 활용하여 임상 노트를 자동으로 구분하는 방법론을 제시합니다. 본 연구는 특히 현재 질병의 역사, 구간 역사 및 평가 및 계획의 세 가지 섹션에 중점을 두어, 487개의 진행 노트에 대한 세심한 데이터셋을 사용하여 모델을 개선하고 평가했습니다. 연구 결과, 미세 조정된 Llama 3.1 8B 모델이 GPT-4o를 초과하는 성능(F1=0.92)을 나타내며, 이는 병원과 임상 분야에서의 접근성과 비용 효율성을 크게 향상시킬 수 있는 가능성을 보여줍니다.

- **Technical Details**: 임상 노트는 전문의의 직간접적인 관찰과 평가를 기록하는 반면, 이러한 노트를 효과적으로 분류하는 것은 어려움이 많습니다. 연구진은 Clinical-Longformer와 같은 다양한 기계 학습 모델을 사용하여 모델 성능을 평가했으며, 임상 노트에서 특정 구간을 추출하기 위한 미세 조정된 LLM을 훈련했습니다. 최종 데이터셋은 1,147개의 노트로 구성되며, 미세 조정 과정에서는 rank-stabilized LoRA 방법이 사용되었습니다.

- **Performance Highlights**: 최적화된 Llama 3.1 8B 모델은 F1 점수 0.92를 기록하며, 낮은 비용의 소스 모델로서 현대의 의료 시스템에서 발전된 성과를 달성했습니다. 외적 유효성 테스트 세트에서도 F1 점수 0.85를 유지하며 높은 성능을 보여주었습니다. 이러한 결과는 임상 노트를 구조화하여 후속 분석을 수행하는 데 있어 큰 장점을 제공할 것입니다.



### CAPRAG: A Large Language Model Solution for Customer Service and Automatic Reporting using Vector and Graph Retrieval-Augmented Generation (https://arxiv.org/abs/2501.13993)
Comments:
          14 pages, 5 Figures, 3 Tables

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)을 기반으로 한 금융 챗봇을 통한 고객 경험 개선을 위한 AI 에이전트를 도입했습니다. 고객은 은행 서비스 및 연례 보고서에 대한 관련 정보를 손쉽게 얻을 수 있으며, 하이브리드 고객 분석 파이프라인인 CAPRAG를 통해 관계 기반 및 맥락 기반 질문 모두를 효과적으로 처리합니다. 이러한 접근 방식은 벡터 RAG와 그래프 RAG를 결합하여 더 나은 고객 참여를 촉진합니다.

- **Technical Details**: 하이브리드 접근 방식을 통해 CAPRAG는 고객의 질문을 확장한 후, 검색된 결과를 그래프 데이터베이스와 결합하여 사용자에게 정확한 답변을 제공합니다. Cypher 쿼리 언어를 사용하여 그래프 데이터베이스를 효율적으로 질의할 수 있으며, 이를 LLM을 통해 응답 생성에 활용합니다. 저자들은 이러한 방식을 통해 디지털 환경에서의 모호함을 줄이고 정보 접근성을 높이려고 노력하고 있습니다.

- **Performance Highlights**: 연구팀은 LLMs를 평가 기준으로 삼아 결과의 정확성과 일관성을 평가합니다. 주요 평가 지표로는 답변의 관련성, 맥락의 관련성, 사실 기반의 신뢰성을 포함하여, 모델의 출력 품질을 지속적으로 관리합니다. 이를 통해 사용자 경험을 개선하고 시스템에 대한 신뢰를 구축할 수 있었습니다.



### Assisting Mathematical Formalization with A Learning-based Premise Retriever (https://arxiv.org/abs/2501.13959)
- **What's New**: 이번 연구에서는 수학의 정형화를 지원하기 위한 프레미스 리트리버(premise retriever) 훈련을 위한 혁신적인 방법을 소개합니다. 이 방법은 BERT 모델을 사용하여 증명 상태와 프레미스를 공유 잠재 공간(latent space)에 임베딩합니다. 실험 결과, 우리의 모델은 기존 기준선보다 성능이 뛰어나며, 컴퓨팅 자원이 적게 소모되어 효율적임을 보여주었습니다.

- **Technical Details**: 프레미스 리트리버 모델은 컨텍스트 프리 리트리벌(CFR) 모듈과 컨텍스트 인지 재정렬(CAR) 모듈로 구성되어 있습니다. 각 모듈은 BERT를 기반으로 하며, CFR 모듈은 임베딩을 생성하여 입력된 증명 상태와 유사한 프레미스를 검색합니다. CAR 모듈은 검색된 프레미스를 재정렬하여 최상위 k-k recall을 개선하여 결과의 정확성을 높이는 데 기여합니다.

- **Performance Highlights**: 우리 모델은 성능에서 최첨단 모델을 초월하는 결과를 보였으며, 여전히 계산 효율성을 유지하고 있습니다. 또한 우리는 이 모델을 독립적으로 훈련된 전술 생성기(tactic generator)와 통합하여 검색 보강 자동 정리 증명기(retrieval-augmented automated theorem prover)를 평가하였습니다. 이 시스템은 수학적 증명 과정의 효율성을 높이는 데 기여하며, 사용자들에게 편리한 웹 검색 엔진을 제공할 예정입니다.



### A Survey of Graph Retrieval-Augmented Generation for Customized Large Language Models (https://arxiv.org/abs/2501.13958)
- **What's New**: 최근의 연구는 Graph-based Retrieval-Augmented Generation (GraphRAG) 모델이 기존 RAG의 한계를 극복하는 새롭고 효율적인 방법임을 보여줍니다. GraphRAG는 전문 도메인에 맞는 LLM의 적용을 통해 복잡한 쿼리를 이해하고, 다양한 지식을 통합하며, 효율성을 향상시킵니다. 이러한 혁신적인 접근 방식은 지식 구조를 그래프 형태로 표현하여 도메인 내의 관계와 계층을 명확하게 설명합니다.

- **Technical Details**: GraphRAG는 도메인 지식의 명확한 표현을 위해 그래프 구조를 활용하며, 이를 통해 다단계 추론을 지원하는 효율적인 검색 기법을 도입합니다. 이 모델은 쿼리에 대한 맥락을 유지하면서 지식을 검색할 수 있도록 설계되었으며, 얻어진 지식을 활용해 LLM의 생성 품질을 높이는 구조 인지 알고리즘을 포함합니다. 또한, 다양한 전문 분야에서 GraphRAG의 현재 구현 사례를 분석하여 중요한 기술적 도전 과제를 확인합니다.

- **Performance Highlights**: GraphRAG의 통합적인 지식 검색 방법은 사용자 쿼리에 대한 정확하고 논리적으로 일관된 응답을 생성할 수 있게 해 줍니다. 기존 RAG 시스템의 성능 한계를 극복하여 복잡한 쿼리를 보다 효과적으로 처리하며, 시스템의 효율성을 높이는 동시에 실시간으로 도메인 특화된 지식에 접근할 수 있도록 합니다. 이를 통해 LLM을 전문화된 환경에서 더욱 효과적으로 활용할 수 있는 가능성을 제시합니다.



### Zep: A Temporal Knowledge Graph Architecture for Agent Memory (https://arxiv.org/abs/2501.13956)
Comments:
          12 pages, 3 tables

- **What's New**: Zep은 AI 에이전트를 위한 새로운 메모리 레이어 서비스로, Deep Memory Retrieval (DMR) 벤치마크에서 기존의 MemGPT를 능가하는 성능을 보여줍니다. Zep은 DMR보다 더 포괄적이면서도 어려운 평가에서도 뛰어난 성과를 내어 실제 기업 사용 사례를 보다 잘 반영합니다. Graphiti라는 핵심 구성 요소를 통해 다양한 데이터 소스로부터 동적인 지식 통합을 수행하여 기업의 요구를 충족합니다.

- **Technical Details**: Zep의 정보 저장 방식은 동적인 지식 그래프(knowledge graph)를 기반으로 하며, 이 그래프는 입력 메시지 데이터를 포함하는 에피소드 서브그래프, 에피소드에서 추출된 엔티티를 나타내는 의미 엔티티 서브그래프, 그리고 강하게 연결된 엔티티들의 클러스터를 나타내는 커뮤니티 서브그래프의 세 가지 계층으로 구성되어 있습니다. Graphiti는 비손실 방식으로 지식 그래프를 업데이트하며, 각 사실과 관계의 타임라인을 유지하여 복잡하고 진화하는 세상을 모델링합니다. 이 기술적 접근은 LLM 기반 에이전트의 기억을 혁신적으로 확장하는 데 기여합니다.

- **Performance Highlights**: Zep은 DMR 벤치마크에서 94.8%의 성능을 기록하여 MemGPT의 93.4%를 초과하였습니다. LongMemEval 벤치마크에서도 Zep은 최대 18.5%의 정확도 향상과 함께 응답 지연 시간을 90% 단축하는 성과를 기록했습니다. 이러한 결과는 특히 세션 간 정보 통합 및 장기적 컨텍스트 유지와 같은 기업 필수 작업에서 두드러지며, 실제 애플리케이션에서의 효과성을 입증합니다.



### Chat3GPP: An Open-Source Retrieval-Augmented Generation Framework for 3GPP Documents (https://arxiv.org/abs/2501.13954)
- **What's New**: Chat3GPP는 3GPP 문서에 특화된 오픈 소스 RAG(확장된 정보 검색 생성) 프레임워크로 제안되며, 이는 도메인 특화된 파인 튜닝 없이 사용자 쿼리에 대한 관련 정보 검색 및 정확한 응답 생성을 가능하게 합니다. 이러한 유연성 및 확장성 덕분에 Chat3GPP는 3GPP 외의 다른 기술 표준에도 쉽게 적용될 수 있는 잠재력을 지니고 있습니다.

- **Technical Details**: Chat3GPP는 3GPP의 기술 문서에서 정보를 처리하고 검색하기 위해 데이터 전처리, 인덱싱, 검색 및 생성의 여러 주요 단계를 포함하는 아키텍처를 가지고 있습니다. 이를 위해, 3GPP FTP 사이트에서 Release 17과 Release 18 기술 사양 문서를 크롤링하고, 문서의 형식을 변환한 후, 내용을 필터링하여 텍스트만을 추출하여 임베딩 데이터베이스로 변환합니다.

- **Performance Highlights**: Chat3GPP는 통신 관련 데이터 세트에서 평가되었으며, 기존 방법들에 비해 우수한 성능을 보여줍니다. 이는 프로토콜 생성 및 코드 자동화와 같은 다운스트림 태스크에 대한 잠재력을 강조하며, LLM의 발전을 통해 통신 산업의 다양한 측면에서 혁신을 기대할 수 있습니다.



New uploads on arXiv(cs.CV)

### HERMES: A Unified Self-Driving World Model for Simultaneous 3D Scene Understanding and Generation (https://arxiv.org/abs/2501.14729)
Comments:
          Work in progress. The code will be available at this https URL

- **What's New**: 본 논문에서는 Driving World Models (DWMs) 분야에 새로운 모델인 HERMES를 제안합니다. HERMES는 3D scene understanding과 미래 장면 예측(generation)을 통합하여 자율주행 시나리오에서 더 나은 성능을 제공합니다. 이 모델은 Bird's-Eye View (BEV) 표현을 활용하여 다각적 공간 정보를 통합하고 기하학적 관계와 상호작용을 효과적으로 보존합니다.

- **Technical Details**: HERMES는 세계 지식을 BEV 기능에 causal attention을 통해 통합하는 world queries를 도입합니다. 이를 통해 HERMES는 복잡한 자율주행 환경에 대한 예측 및 이해를 동시에 가능하게 합니다. 연구 결과에 따르면, HERMES는 nuScenes 및 OmniDrive-nuScenes와 같은 데이터셋에서 32.4%의 생성 오류 감소 및 CIDEr 성능 지표에서 8.0% 향상을 이뤘습니다.

- **Performance Highlights**: HERMES는 공공 데이터셋에서 기존의 최신 기법들과 비교했을 때 눈에 띄는 성능 향상을 보여주었습니다. 특히, HERMES는 3초 후의 미래 포인트 클라우드를 예측하는 데 있어 뛰어난 성능을 보이고 있습니다. HERMES의 코드 및 모델은 공개될 예정입니다.



### Relightable Full-Body Gaussian Codec Avatars (https://arxiv.org/abs/2501.14726)
Comments:
          14 pages, 9 figures. Project page: this https URL

- **What's New**: 새로운 Relightable Full-Body Gaussian Codec Avatars 기술을 제안합니다. 이 접근법은 인체의 얼굴과 손을 포함한 다양한 세부 사항을 정밀하게 모델링하여 조명을 조정할 수 있는 전체 신체 아바타를 생성하는 데 중점을 둡니다. 특히, 본 연구는 조명 전달의 지역적 및 비지역적 효과를 분리하여 모델링할 수 있는 방법을 제시합니다. 이는 기존 방법에서의 한계를 극복하고, 더 나은 일반화 능력을 갖춘 조명 조건에서 작동할 수 있는 모델을 가능하게 합니다.

- **Technical Details**: 본 연구에서는 zonal harmonics를 사용하여 지역적 확산 방사선 전달을 모델링합니다. 이는 전통적인 구형 조화 함수(Spherical Harmonics)보다 효율적으로 본체의 관절 구동에 대해 회전할 수 있습니다. 또한, 그림자 네트워크를 도입하여 기반 메시 위에 들어오는 조도를 정규화된 방식으로 예측하여 아바타의 각 신체 부위 간의 상호 차폐 효과를 고려합니다. 마지막으로, deferred shading 접근법을 통하여 고충실도의 스펙큘러 반사 모델링을 적용합니다.

- **Performance Highlights**: 본 연구는 전체 신체 아바타의 지역적 및 비지역적 조명 전달을 성공적으로 모델링하여, 새로운 조명 조건과 미지의 포즈에서도 우수한 일반화 능력을 보입니다. 제안된 모델은 고충실도의 리라이트 가능 성능을 제공하며, 뚜렷한 얼굴 세부 묘사를 유지하면서도 필요한 Gaussian 개수의 증가를 최소화합니다. 이러한 성능은 고충실도의 리라이트 가능한 인간 아바타 모델링을 가능하게 합니다.



### Approach to Designing CV Systems for Medical Applications: Data, Architecture and AI (https://arxiv.org/abs/2501.14689)
Comments:
          9 pages, 3 figures

- **What's New**: 이 논문은 fundus 이미지 분석을 위한 혁신적인 소프트웨어 시스템을 소개합니다. 전통적인 진단 예측 접근법과는 달리, 이 시스템은 fundus 구조의 정상 및 병리적 특징을 철저히 분석하여 의료 전문가에게 최종 결정권을 위임합니다. 이 연구는 객관적인 임상 분석의 필요성을 해결하고 임상 작업 흐름을 자동화 및 향상시키고자 하며, AI 모델을 통해 모듈식 분석 디자인을 제공합니다.

- **Technical Details**: EYAS는 AI 기반의 fundus 이미지 분석 시스템으로, 데이터 부족 문제를 해결하기 위해 전문 의료진과의 협업을 강조합니다. 시스템은 특정 작업을 위한 전문 모듈을 통합하여 생성된 중간 결과물을 임상의가 이해하고 검증할 수 있도록 설계되었습니다. 또한, 최신 딥러닝 기법과 기존의 컴퓨터 비전 알고리즘을 결합하여 fundus 구조에 대한 포괄적이고 세분화된 분석을 제공합니다.

- **Performance Highlights**: 연구 결과, EYAS의 접근 방식은 fundus 이미지 분석 혁신에 효과적이며, 다양한 의료 분야에서의 잠재적 응용성을 보여줍니다. 이러한 디자인 접근법은 의료 전문가의 실질적인 요구에 맞춰 AI 기술과의 간극을 줄이며, 안과 및 다른 의료 분야에서 AI의 의미 있는 채택을 촉진하는 것을 목표로 합니다.



### Surface Vision Mamba: Leveraging Bidirectional State Space Model for Efficient Spherical Manifold Representation (https://arxiv.org/abs/2501.14679)
- **What's New**: 이 연구에서는 Attention 기반 방법들이 구형 피질 표면에서의 장기 의존성 모델링에 매우 우수한 성능을 보였음을 강조합니다. 그러나 긴 추론 시간과 높은 메모리 요구사항이 대규모 데이터셋에의 적용을 어렵게 만들고 있습니다. 이를 해결하기 위해, 우리는 Attention 없이 작동하는 'Vision Mamba (Vim)'를 제안하며, 이는 구형 다양체(spherical manifold)에서 데이터 분석을 위한 도메인 독립적 아키텍처입니다.

- **Technical Details**: 제안된 방법은 구형 데이터를 잘게 나눈 삼각형 패치(triangular patches)로 표현하여 표면 패치(surface patching)를 수행합니다. 'Surface Vision Mamba (SiM)'는 신생아 뇌의 피질 표면 메트릭을 이용한 여러 신경 발달 표현 회귀(task)에서 평가되었습니다. 이 방법은 Ico-4 그리드(partitioning) 하에서 Surface Vision Transformer (SiT)와 비교했을 때, 4.8배 빠른 추론 속도와 91.7% 낮은 메모리 소비를 달성했습니다.

- **Performance Highlights**: 실험적 결과는 SiM이 Attention 기반 및 GDL 모델을 모두 초월하는 성능을 보여준다고 강조합니다. 또한 민감도 분석(sensitivity analysis)을 통해 SiM이 미세한 인지 발달 패턴을 식별할 수 있는 잠재력을 가지고 있음을 확인했습니다. 이 코드는 해당 URL에서 제공됩니다.



### MatAnyone: Stable Video Matting with Consistent Memory Propagation (https://arxiv.org/abs/2501.14677)
Comments:
          Project page: this https URL

- **What's New**: MatAnyone은 비디오 매팅 분야에서 새로운 연구로, 복잡한 배경에서도 오브젝트 트래킹을 안정적으로 수행할 수 있는 강화된 프레임워크입니다. 특히, region-adaptive memory fusion이라는 혁신적 메커니즘을 도입하여 메모리의 전이와 통합을 적응적으로 수행합니다. 이를 통해 핵심 영역의 의미적 안정성을 보장하면서도 객체 경계의 세부 사항을 유지하는 것이 가능합니다. 또한 새로운 대규모 데이터셋인 VM800과 YoutubeMatte를 활용하여 비디오 매팅의 안정성과 신뢰성을 개선합니다.

- **Technical Details**: 본 연구에서는 이전 프레임으로부터 메모리를 통합하는 일관된 메모리 전파 메커니즘을 제안하였습니다. 메모리 은행에서의 정보 쿼리를 통해 'large-change' 지역은 현재 프레임의 정보를 중시하며, 'small-change' 지역은 이전 프레임의 메모리를 유지하도록 설정됩니다. 이러한 구조는 비디오 전반에 걸쳐 메모리 전파의 안정성을 높여주며, 경계 세부 사항을 유지하는 데 크게 기여합니다. 태스크의 정확성과 세부 사항을 높이기 위해 동일한 네트워크 헤드에서 세그멘테이션 데이터를 매팅과 함께 사용하는 새로운 훈련 전략도 도입되었습니다.

- **Performance Highlights**: MatAnyone은 기존 메소드들보다 실질적으로 향상된 비디오 매팅 성능을 보여줍니다. 새로운 메모리 전파 기법과 대규모 데이터세트를 통해 핵심 영역에서는 의미적 안정성을 극대화하고, 경계 영역에서는 세부 사항이 더욱 뚜렷하게 표현됩니다. 다양한 실제 시나리오에서의 성능 비교에서, MatAnyone은 안정적이고 정확한 결과를 제공하며 비디오 매팅 분야에서 중요한 진전을 이루었습니다.



### Towards Unified Structured Light Optimization (https://arxiv.org/abs/2501.14659)
- **What's New**: 이 논문의 핵심은 Structured Light (SL) 3D 재구성을 위한 통합 최적화 프레임워크의 제안입니다. 이 프레임워크는 다양한 조명 조건, 물체 유형 및 SL 유형에 적응할 수 있으며, 오직 하나의 투사된 이미지만으로 최적의 프로젝션 패턴을 신속하게 결정할 수 있도록 도와줍니다.

- **Technical Details**: 제안된 방법은 프로젝터-카메라 정합을 위한 새로운 글로벌 매칭 방식과 포토메트릭 조정 모듈을 포함하는 프로젝션 보정 모델을 포함하여, SL 패턴 최적화 과정에서 발생하는 아티팩트를 줄이는데 중점을 두고 있습니다. 핵심은 2D 평면 삼각 측량 텍스처 맵핑 기법을 사용하는 것으로, 이는 고속 정렬을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 다양한 물체, SL 패턴 및 조명 조건에서 기존 방법보다 우수한 디코딩 정확도를 보여주었습니다. 이러한 성과는 SL 3D 재구성의 신뢰성을 높이며, 기존의 연구에서 다루지 못한 다양한 응용 분야에서의 활용 가능성을 넓혀줍니다.



### SyncAnimation: A Real-Time End-to-End Framework for Audio-Driven Human Pose and Talking Head Animation (https://arxiv.org/abs/2501.14646)
Comments:
          11 pages, 7 figures

- **What's New**: 이 논문에서는 SyncAnimation이라는 새로운 NeRF 기반 방법을 소개합니다. SyncAnimation은 오디오에 기반한 안정적이고 실시간으로 구동되는 화상 아바타 생성을 위해 오디오-포즈 매칭과 오디오-표정 동기화를 결합합니다. 이 방법은 AudioPose Syncer와 AudioEmotion Syncer를 통합하여 탐지된 오디오에 대해 높은 정확도의 포즈와 표정을 생성할 수 있게 합니다.

- **Technical Details**: SyncAnimation은 세 가지 핵심 모듈로 구성됩니다: AudioPose Syncer, AudioEmotion Syncer, 그리고 High-Synchronization Human Renderer입니다. AudioPose Syncer는 오디오를 동적인 헤드 포즈로 정확하게 매핑하고, AudioEmotion Syncer는 오디오에 의해 제어 가능한 얼굴 표정을 촉진합니다. 마지막으로, High-Synchronization Human Renderer는 얼굴과 상체의 통합을 보장하며, 음성과 동기화된 리얼한 상체 생성을 가능하게 합니다.

- **Performance Highlights**: SyncAnimation은 NVIDIA RTX 4090 GPU에서 41 FPS의 인퍼런스를 달성했으며, 이는 실시간 오디오 기반 아바타 생성의 첫 번째 사례로, 상체 움직임과 헤드 모션을 오디오와 동기화하여 생성할 수 있습니다. 실험 결과, SyncAnimation은 원본 비디오와 같은 스케일의 리얼한 아바타를 성공적으로 생성하며, 정량적 및 정성적 평가 모두에서 기존 최첨단 방법들을 크게 능가하는 성능을 보였습니다.



### ReferDINO: Referring Video Object Segmentation with Visual Grounding Foundations (https://arxiv.org/abs/2501.14607)
Comments:
          Project page: this https URL

- **What's New**: ReferDINO는 설명된 텍스트를 기반으로 비디오에서 목표 객체를 분할하는 최신 모델로, 뛰어난 vision-language 이해 및 효과적인 temporal understanding 능력을 갖추고 있습니다. 기존 RVOS 모델들이 복잡한 객체 설명을 처리하는 데 어려움을 겪는 가운데, ReferDINO는 pretrained visual grounding foundation models의 이점을 활용하여 이러한 문제를 극복합니다. 이 모델은 object-consistent temporal enhancer, grounding-guided deformable mask decoder 및 confidence-aware query pruning strategy 등 세 가지 주요 혁신을 도입하여 성능을 향상시킵니다.

- **Technical Details**: ReferDINO는 GroundingDINO의 구조를 기반으로 하여 비디오 클립과 텍스트 설명을 입력으로 사용합니다. GroundingDINO의 cross-modal encoder를 통해 이미지와 텍스트의 깊이 있는 특성을 추출하고, 시간에 따른 객체의 일관성을 촉진하는 object-consistent temporal enhancer를 활용합니다. 이와 함께 grounding-guided deformable mask decoder를 통해 텍스트와 grounding 조건을 통합하여 보다 정확한 객체 분할을 달성하고, computational efficiency를 위한 confidence-aware query pruning 전략을 설계하여 성능을 유지합니다.

- **Performance Highlights**: ReferDINO는 5개의 공공 RVOS 벤치마크에서 SOTA 방법들보다 현저한 성능 향상을 이루어냈습니다. 예를 들어, Ref-DAVIS17 데이터셋에서 ReferDINO 모델은 SOTA 방법 대비 4.0% 향상된 성능을 보였으며, Ref-Youtube-VOS 데이터셋에서도 12.0% 이상의 성능 개선을 기록했습니다. 이러한 결과는 ReferDINO의 설계가 pretrained foundation models의 공간 grounding 지식을 최대한 활용하고 있음을 보여줍니다.



### 3DLabelProp: Geometric-Driven Domain Generalization for LiDAR Semantic Segmentation in Autonomous Driving (https://arxiv.org/abs/2501.14605)
- **What's New**: 이번 논문에서는 LiDAR 데이터에 대한 영역 일반화(domain generalization) 문제를 해결하기 위한 새로운 기하학 기반 접근법인 3DLabelProp을 제안합니다. 이 방법은 LiDAR 센서의 순차적 구조를 활용하여 기존의 학습 기반 방법들과 차별화됩니다. 3DLabelProp은 LiDAR Semantic Segmentation(LSS) 작업에서 사용되며, 다양한 데이터셋에 대한 광범위한 실험을 통해 최첨단 성능을 입증하였습니다.

- **Technical Details**: 기존의 학습 기반 방법들이 아닌 기하학적 전략을 통해 영역 일반화를 증가시킬 수 있다는 아이디어를 바탕으로 3DLabelProp을 설계했습니다. 이 방법은 새로운 절반 조밀한 포인트 클라우드(pseudo-dense point clouds) 개념을 통해 LiDAR 데이터에 대한 이해와 평가를 돕고자 합니다. 또한, 실험 설정을 formalize하고, 이를 통해 3DLabelProp의 효율적인 처리 방법도 제시합니다.

- **Performance Highlights**: 3DLabelProp은 기존의 나이브(naïve) 접근법과 다른 영역 일반화 방법들보다 더 우수한 성능을 보였습니다. 실험 결과, 3DLabelProp은 총 7개의 LiDAR 데이터셋에서 광범위한 결론을 도출할 수 있게 하며, LiDAR Semantic Segmentation의 현 상태에서 뛰어난 성과를 거두었습니다. 이 연구는 LiDAR 데이터 인식의 향상된 일반화 능력을 위한 새로운 방법론을 제안합니다.



### Geometric Mean Improves Loss For Few-Shot Learning (https://arxiv.org/abs/2501.14593)
- **What's New**: 이 논문에서는 기하 평균(geometric mean)을 기반으로 한 새로운 few-shot learning (FSL) 손실 함수를 제안합니다. 기존의 손실 함수들은 일반적으로 산술 평균(arithmetic mean)을 사용하여 구성이 되었으나, 제안된 방법은 샘플 간의 쌍(pair-wise) 관계를 집계하여 여러 클래스 영역에서의 판별력을 향상시키는 특징이 있습니다. 이 새로운 손실 함수는 간단한 형태로 수식화되어 있으며, FSL의 특성에 잘 맞도록 이론적으로 철저하게 분석되었습니다.

- **Technical Details**: 제안된 손실 함수는 softmax 기반의 주의 가중치(attention weight)를 통한 쌍(pair-wise) 관계의 인코딩을 통해 효과적인 feature metric을 학습하게 해줍니다. 이는 더 넓은 샘플 분포의 구조를 고려하여, 제한된 샘플의 구조와 양에만 집중하는 기존의 FSL 손실 함수와는 대조적인 특징을 가집니다. 손실 함수의 설계는 훈련 샘플 전체에 대한 관점을 반영하며, 딥 러닝 모델을 통해 효과적으로 훈련될 수 있도록 구성됩니다.

- **Performance Highlights**: 실험 결과, 제안한 손실 함수는 기존의 FSL 손실 함수들과 비교하여 경쟁력 있는 성능을 보여줍니다. 특히, FSL 이미지 분류 작업에서 뛰어난 효율을 입증하였으며, 이론적 분석을 통해 다양한 관점에서 손실 함수의 유리한 특성을 밝혀냈습니다. 이러한 모든 요소들은 FSL을 위한 딥 특징(metric) 학습의 가능성을 높여주는 결과를 보여줍니다.



### Visual Localization via Semantic Structures in Autonomous Photovoltaic Power Plant Inspection (https://arxiv.org/abs/2501.14587)
Comments:
          47 pages, 22 figures

- **What's New**: 이 논문에서는 열화상 카메라로 장착된 무인 항공기(UAV)를 활용한 태양광 발전소(PV) 점검 시스템의 새로운 로컬라이제이션 파이프라인을 제안합니다. 이 시스템은 PV 모듈 탐지를 UAV 탐색과 직접 통합하여 점검 시 정밀한 위치 지정을 가능하게 합니다.

- **Technical Details**: 탐지된 데이터는 이미지에서 발전소 구조를 식별하고 이를 발전소 모델과 연관시키는 데 사용됩니다. 초기 연관을 위해 시각적으로 인식 가능한 앵커 포인트를 정의하고, 객체 추적(object tracking)을 통해 글로벌 연관을 식별합니다. 논문에서는 전통적인 컴퓨터 비전, 딥 러닝 및 두 가지를 융합한 방법을 포함한 세 가지 다양한 PV 모듈의 시각적 분할 방법을 제시합니다.

- **Performance Highlights**: 제시된 방법들은 맞춤형 공중 점검 데이터 세트를 사용하여 검증되었으며, 실시간 탐색을 위한 강력성과 적용 가능성을 보여주었습니다. 또한, 발전소 모델의 정밀도가 로컬라이제이션 방법에 미치는 영향을 평가하여 전반적인 성능을 분석합니다.



### Large-scale and Fine-grained Vision-language Pre-training for Enhanced CT Image Understanding (https://arxiv.org/abs/2501.14548)
Comments:
          Accepted by ICLR 2025

- **What's New**: 이 논문에서는 의료 영상 해석을 위한 정밀한 비전-언어 모델(fVLM)을 제안합니다. 기존의 글로벌 대비 학습(global contrastive learning) 방식에서 벗어나 CT 이미지와 방사선 보고서 간의 해부학적 레벨의 세밀한 정렬을 수행합니다. 이는 의료 보고서가 CT 이미지에서 임상적으로 중요한 이상 소견을 상세히 기록하기 때문에, 텍스트로 설명된 발견과 이미지 위치 간의 본질적인 정렬을 확립하는 데 기여합니다.

- **Technical Details**: fVLM은 CT 이미지와 보고서 모두에 대해 해부학적 레벨의 분해 및 일치를 수행하고, 동일한 해부학에 대한 시각적 임베딩과 보고서 임베딩 간의 세밀한 정렬을 통해 글로벌 대비 학습에서 발생하는 잘못된 정렬 문제를 완화합니다. 이러한 접근법은 과도한 정상 샘플과 유사한 이상 소견으로 인한 오탐(False negative) 문제를 해결하기 위한 이중 오탐 감지 모듈을 도입하여, 환자 수준 pairing에서 질병 인지형(pairing) 접근법으로 전환합니다.

- **Performance Highlights**: 우리의 fVLM은 69,086명의 환자 데이터로 구성된 가장 큰 CT 데이터셋인 MedVL-CT69K에서 실험되어, 54개 진단 작업에서 평균 AUC 81.3%를 달성하며 CLIP 및 감독 방법을 각각 12.9%와 8.0% 초과했습니다. 또한, 공개 CT-RATE 및 Rad-ChestCT 데이터셋에서도 각기 7.4% 및 4.8%의 성능 개선을 보이며, 진단 작업 외에도 보고서 생성 작업에서도 뛰어난 성능을 보였습니다.



### Leveraging ChatGPT's Multimodal Vision Capabilities to Rank Satellite Images by Poverty Level: Advancing Tools for Social Science Research (https://arxiv.org/abs/2501.14546)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)이 위성 이미지를 사용해 마을 수준의 빈곤을 예측하는 새로운 방법을 탐색합니다. LLMs는 본래 자연어 이해를 위해 개발되었으나, 다양한 도메인에서의 적응력이 뛰어나므로 지리적 분석과 같은 다중 모드 작업에 혁신적인 활용이 가능해졌습니다. 연구 결과, ChatGPT는 전문가와 유사한 정확도로 빈곤 수준에 따라 위성 이미지를 분류할 수 있음을 보였습니다.

- **Technical Details**: 본 연구는 2015/2016 탄자니아 인구 및 건강 조사(DHS) 데이터를 기반으로 하여, 각 지역의 가구 수를 수집하고 이를 빈곤의 지표로 사용했습니다. Wealth index(재산 지수)는 가구의 자산 소유 및 기타 사회 경제적 요소를 고려하여 계산되었으며, 이를 통해 빈곤 상태를 평가하는 데 기여했습니다. 연구는 OpenAI의 GPT-4o 모델을 활용하여 위성 이미지의 쌍을 비교하고, 인프라의 질과 가시적인 주변 환경을 평가하여 부유한 위치를 결정했습니다.

- **Performance Highlights**: 이 연구에서는 ChatGPT가 빈곤을 기반으로 한 이미지 순위를 전문가와 비슷한 수준으로 수행할 수 있음을 보여주었습니다. 비교 결과는 LLMs를 사용하여 빈곤 평가에 있어 전통적인 방법과 유사한 신뢰성을 가지며, 대규모 및 비용 효율적인 빈곤 모니터링의 가능성을 제시합니다. LLMs의 사용은 데이터의 복잡성을 해소하고 해석 가능한 통찰력을 제공하는 데 기여할 수 있어 사회 경제적 연구에 중요한 기초를 마련했습니다.



### Rethinking Encoder-Decoder Flow Through Shared Structures (https://arxiv.org/abs/2501.14535)
- **What's New**: 본 연구는 기존의 디코더 아키텍처에 새로운 은행(bank) 구조를 도입하여 깊이 추정(dense prediction) 성능을 향상시키려는 목적을 가지고 있습니다. 각 디코딩 블록이 개별 중간 특성 맵만을 처리하는 대신, 이전 및 후속 블록의 정보도 활용할 수 있는 공유 텐서(shared tensor)를 운영하여, 더 많은 맥락(context)을 제공할 수 있도록 설계되었습니다. 이러한 시스템은 대규모 데이터셋에서 훈련하여 자연 및 합성 이미지에서의 깊이 추정 성능을 크게 개선하였습니다.

- **Technical Details**: 본 연구는 이미지 인코더 𝐄와 이미지 디코더 𝐃로 구성된 구조를 가지고 있습니다. 여기서 ViT(Vision Transformer) 인코더는 4개의 중간 특성 맵을 출력하고, 디코더는 각 블록이 해당 특성 맵과 은행 구조를 활용하여 깊이 맵 𝒪d를 생성하는 방식을 채택합니다. 특별히, 은행 구조는 모든 디코딩 블록에서 사용될 수 있는 공통의 정보를 제공하는데, 이를 통해 블록 디자인의 성능이 향상됩니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 GFLOPS나 파라미터 수의 미미한 증가에 비해 깊이 추정 정확도를 비례적으로 더 높일 수 있는 것으로 나타났습니다. 이 연구가 제시한 은행 구조는 특정 디코더나 깊이 추정 작업에 한정되지 않으며, 다양한 밀집 예측 작업(dense prediction tasks)에 적용 가능하다는 장점도 있습니다. 향후 연구는 이러한 구조를 다양한 아키텍처에 적용하여 그 효용성을 검증할 필요가 있습니다.



### Trick-GS: A Balanced Bag of Tricks for Efficient Gaussian Splatting (https://arxiv.org/abs/2501.14534)
Comments:
          Accepted at ICASSP'25

- **What's New**: 이번 논문에서는 Gaussian splatting (GS) 기반의 3D 재구성에서 발생하는 문제를 해결하기 위해 Trick-GS라는 새로운 접근법을 제안합니다. Trick-GS는 여러가지 전략을 조합하여 최적화함으로써, 특히 리소스가 제한된 장치에서의 효율성을 향상시키고자 합니다. 이는 보다 빠른 훈련 시간, 작은 디스크 크기 및 향상된 렌더링 속도를 통해 GS의 한계를 극복합니다.

- **Technical Details**: Trick-GS는 three key strategies을 사용하여 구현됩니다. 첫 번째는 점진적 훈련(progessive training)으로, 해상도, 노이즈 및 Gaussian 스케일을 학습하는 것이고, 두 번째는 중요도에 따라 primitives 및 SH 밴드를 자르고(mask) 마스킹하는 것입니다. 마지막으로, 가속화된 GS 훈련 프레임워크를 통해 전반적인 성능을 개선하였습니다.

- **Performance Highlights**: 세 가지 데이터셋에 대한 결과는 Trick-GS가 Vanilla GS와 비교하여 훈련 시간이 최대 2배 단축되며, 디스크 크기가 40배 작아지고 렌더링 속도가 2배 빨라짐을 보여줍니다. 이것은 GS 방식이 빠른 훈련과 높은 정확성을 유지하면서도, 리소스가 제한된 환경을 효과적으로 지원할 수 있다는 것을 뜻합니다.



### CheapNVS: Real-Time On-Device Narrow-Baseline Novel View Synthesis (https://arxiv.org/abs/2501.14533)
Comments:
          Accepted to ICASSP 2025

- **What's New**: 이 논문에서는 저비용으로 단일 뷰에서 새로운 뷰를 합성하는 CheapNVS라는 이름의 새로운 방법을 제안합니다. CheapNVS는 효율적인 멀티 인코더/디코더 구조를 기반으로 하여, 카메라 포즈 임베딩을 활용하여 경량의 학습 가능한 모듈로 3D 이미지 왜곡을 근사합니다. 이 방법은 성능 개선을 위해 차단된 영역에서 인페인팅을 병렬로 수행하며, 기존의 최첨단 기법보다 10배 빨라지고 메모리 사용량도 6% 감소합니다.

- **Technical Details**: CheapNVS는 내부적으로 공유 RGBD 인코더, 외부 인코더 및 세 개의 디코더를 구성하는 새로운 아키텍처를 적용합니다. 이 구조는 NVS 파이프라인을 더욱 가속화하는 동시에, 실시간으로 모바일 장치에서 작동할 수 있도록 설계되었습니다. 학습 과정은 Open Images 데이터셋의 하위 집합을 기반으로 하였으며, 이로 인해 최첨단 기법과 비교해 동등하거나 더 나은 정확도를 제공하면서도 실행 속도와 메모리 사용에 있어 큰 개선을 이루었습니다.

- **Performance Highlights**: 평가 결과, CheapNVS는 삼성 Tab 9+와 같은 모바일 기기에서 30 FPS 이상의 속도로 실시간으로 실행될 수 있습니다. 이는 사용자의 요구에 부합하는 성능으로, 다양한 응용 분야에서 혁신적인 가능성을 제시합니다. 결론적으로 CheapNVS는 기계학습 알고리즘의 효율성을 높이고, 전통적인 방법의 한계를 극복하는 실질적인 솔루션으로 자리잡을 것으로 기대됩니다.



### Training-Free Style and Content Transfer by Leveraging U-Net Skip Connections in Stable Diffusion 2.* (https://arxiv.org/abs/2501.14524)
- **What's New**: 최근의 확산 모델(diffusion models) 발전에도 불구하고, 이들 내부의 잠재 표현(latent representations)은 여전히 잘 이해되지 않고 있습니다. 본 연구에서는 U-Net의 skip connection을 활용한 모델인 SkipInject를 제안하며, 이미지 재구성과정에서 공간 정보를 어떻게 표현하는지 분석했습니다. 이 연구를 통해 텍스트 기반의 수정, 정밀한 변경 사항 및 스타일 전이에 대한 정보를 제공합니다.

- **Technical Details**: 우리는 U-Net의 세 번째 인코더 블록에서 전달된 잔여 연결이 재구성된 이미지의 대부분의 공간 정보를 포함하고 있다는 사실을 발견했습니다. 모델은 이미지 A에서 B로의 공간 구성 전환을 가능하게 하며, 이를 위해 이미지 B의 스타일을 A로 전송하는 방법을 제시합니다. 또한 주입된 임베딩의 혼합(modulation)의 변화를 통해 출력 강도를 조절할 수 있음을 보여줍니다.

- **Performance Highlights**: SkipInject는 텍스트 기반 이미지 편집 및 스타일 전송에서 뛰어난 성능을 발휘하며, 기존의 최첨단 방법들과 비교했을 때 최상의 콘텐츠 정렬(content alignment)과 구조 보존(tradeoff)을 입력하였습니다. 또한 다양한 작업에서의 우수한 성능을 달성할 수 있는 효율적이고 통제 가능한 이미지 편집 방법을 제안합니다.



### PARASIDE: An Automatic Paranasal Sinus Segmentation and Structure Analysis Tool for MRI (https://arxiv.org/abs/2501.14514)
- **What's New**: 본 연구에서는 chronic rhinosinusitis (CRS)에 대한 자동 분할 도구인 PARASIDE를 소개합니다. PARASIDE는 T1 MRI에서 상악, 전두, 접형, 비갑개 뼈 구조의 공기 및 연부 조직의 볼륨을 자동으로 분할함으로써, 이전에 수동적으로만 관찰되었던 특성의 관계를 정량화 가능하게 합니다. 이 시스템은 총 16개의 구조체에 대한 자동 전두 비강 완전 분할을 최초로 수행하며, Lund-Mackay 점수와 같은 의학적으로 관련된 특징을 계산할 수 있도록 설계되었습니다.

- **Technical Details**: 연구 데이터는 독일 동북부의 SHIP-START 및 SHIP-TREND 코호트에서 수집되었습니다. 총 8,728명의 참가자 중, 성별, 연령, 병리의 균형을 맞춘 273명의 피험자를 랜덤으로 선별하여 분석하였습니다. 세분화 모델 훈련을 위해 nnUNet 알고리즘을 사용하였고, 총 5,000 에폭(epoch) 동안 훈련하였습니다. 이 과정에서 100개의 교육 샘플에 대해 전문가의 수작업 주석을 기반으로 세분화 마스크를 수정했습니다.

- **Performance Highlights**: 자동으로 세분화된 공기 볼륨의 주석은 우수한 성능을 보여주며, 연부 조직에 비해 일관되게 낮은 평균 강도를 나타냅니다. 설계된 시스템은 60명의 피험자를 대상으로 한 테스트 데이터에서 Dice similarity coefficient (DSC)와 average symmetric surface distance (ASSD) 등의 다양한 성능 평가 지표를 사용하여 성능을 검증하였습니다. 연구 결과는 CHRONIC RHINOSINUSITIS 진단 결과와의 비교를 통한 정량적 평가를 가능하게 하여, 미래의 개인 맞춤형 의료에 기여할 것으로 기대됩니다.



### Deep-BrownConrady: Prediction of Camera Calibration and Distortion Parameters Using Deep Learning and Synthetic Data (https://arxiv.org/abs/2501.14510)
- **What's New**: 이번 연구는 심층 학습 모델을 사용하여 단일 이미지에서 카메라 보정 및 왜곡 매개변수 예측의 과제를 해결합니다. 주요 기여로는 실제 및 합성 이미지 혼합으로 학습된 심층 학습 모델이 단일 이미지로부터 카메라 및 렌즈 매개변수를 정확히 예측할 수 있다는 점과 AILiveSim 시뮬레이션 플랫폼을 이용해 포괄적인 합성 데이터셋을 개발한 것입니다.

- **Technical Details**: 본 연구는 여러 방향의 보정 물체 이미지가 필요했던 전통적인 방법에 비해 단일 이미지에서 카메라 보정 및 렌즈 왜곡 매개변수를 예측하는 심층 학습 접근 방식을 도입합니다. AILiveSim 플랫폼을 통해 생성된 합성 데이터셋을 활용하여, 수평 시야각(H-FOV), 주점(ci,cysubscript𝑐𝑥subscript𝑐𝑦c_{x},c_{y}italic_c start_POSTSUBSCRIPT italic_x end_POSTSUBSCRIPT , italic_c start_POSTSUBSCRIPT italic_y end_POSTSUBSCRIPT), 및 왜곡 계수(k1,k2,k3,p1,p2subscript𝑘1subscript𝑘2subscript𝑘3subscript𝑝1subscript𝑝2k_{1},k_{2},k_{3},p_{1},p_{2}italic_k start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_k start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , italic_k start_POSTSUBSCRIPT 3 end_POSTSUBSCRIPT , italic_p start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_p start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT)의 예측을 가능케 합니다.

- **Performance Highlights**: 제안된 방법은 실험적으로 검증되어 다양한 센서 크기 및 타입에 걸쳐 일반화 능력을 보여줍니다. KITTI 데이터셋을 이용한 검증 결과, 이 모델은 합성 데이터에서 실제 시나리오로의 일반화 성능을 성공적으로 입증하였으며, 카메라 보정 매개변수 예측의 신뢰성을 입증했습니다. 이러한 연구는 카메라 매개변수를 추정하는 데 있어 합성 데이터 활용을 확대할 수 있는 미래 연구 방향을 제시합니다.



### BILLNET: A Binarized Conv3D-LSTM Network with Logic-gated residual architecture for hardware-efficient video inferenc (https://arxiv.org/abs/2501.14495)
Comments:
          Published at IEEE SiPS 2022

- **What's New**: 본 논문에서는 효율적인 비디오 추론을 위해 훨씬 더 작고 컴팩트한 Binarized Conv3D-LSTM 모델 아키텍처인 BILLNET을 제안합니다. 이 모델은 메모리 제한이 있는 하드웨어와의 호환성을 목표로 하며, Conv3D의 복잡성을 점수 내어 비용을 감소시킵니다. 또한, 효과적인 LSTM 레이어의 전량화(quantization) 훈련을 위한 다단계 훈련 방식을 제안하여 기존 모델에 비해 메모리와 계산 자원 소모를 최소화했습니다.

- **Technical Details**: BILLNET은 16 프레임의 시퀀스를 입력으로 받아들이며, 공간 해상도는 96×128입니다. 이 모델은 주변 프레임 간의 시공간 특징을 추출하는 Conv3D 부분과 긴 시간 의존성을 추적하는 LSTM 부분으로 구성되어 있습니다. 핵심 구성 요소로는 두 개의 포인트 와이즈(convolution) 레이어와 그룹화된 컨볼루션을 사용하는 3D 컨볼루션 팩토리제이션이 포함되며, 이 사이에 액티베이션이 없습니다.

- **Performance Highlights**: Jester 데이터셋에서의 실험 결과, BILLNET은 기존의 Conv3D 기반의 자원 효율적 모델들과 비교하여 매우 낮은 메모리와 계산 비용으로 높은 정확도를 달성할 수 있었음을 보여주었습니다. 이는 binarization(이진화) 기법과 효율적인 훈련 전략이 결합된 결과로, 비디오 기반 추론에서 상당한 성과를 나타냅니다. 또한, 본 연구는 LSTM을 통합한 전량화 모델이 비디오 추론에 유용하게 활용될 수 있음을 시사합니다.



### Triple Path Enhanced Neural Architecture Search for Multimodal Fake News Detection (https://arxiv.org/abs/2501.14455)
Comments:
          This paper has been accepted into the IEEE International Conference on Acoustics, Speech, and Signal Processing(ICASSP 2024)

- **What's New**: 이 논문에서는 MUSE라는 새로운 멀티모달(fake news) 탐지 모델을 제안합니다. MUSE는 동적 경로 두 개와 정적 경로 하나를 포함하여보다 적합한 멀티모달 특징 추출을 위한 유연성을 제공합니다. 이러한 모델은 Neural Architecture Search(NAS) 기술을 활용하여 일반화 능력을 향상시킵니다. 실험 결과, MUSE는 기존 모델들보다 더 안정적인 성능 향상을 달성하는 것으로 나타났습니다.

- **Technical Details**: MUSE 모델은 텍스트와 이미지 모달리티를 포함한 가짜 뉴스 탐지에 초점을 맞추고 있습니다. 이 모델은 특징 추출, 특징 융합, 뉴스 예측의 세 가지 단계로 구성됩니다. 또한, MUSE는 Global-Local Guidance를 사용하여 멀티모달 특징을 추가로 정제합니다. 전체적으로 MUSE는 결측 데이터가 존재하는 실제 시나리오를 시뮬레이션하여 다양한 유형의 가짜 뉴스를 탐지할 수 있도록 설계되었습니다.

- **Performance Highlights**: MUSE 모델의 실험 결과는 훈련되지 않은 모달리티를 포함한 데이터에서도 일관되게 우수한 성능을 보여주었습니다. 이 모델은 다양한 가짜 뉴스 유형에 적응할 수 있도록 설계되어, 여러 모달리티를 가진 뉴스의 복잡한 상호작용을 효과적으로 처리합니다. MUSE는 특히 부분 모달리티가 포함된 가짜 뉴스에 대해 향상된 탐지 성능을 제공하였습니다.



### Optimizing Human Pose Estimation Through Focused Human and Joint Regions (https://arxiv.org/abs/2501.14439)
- **What's New**: 이 논문에서는 기존 방법이 인간 몸체의 특정 정보에 집중하지 않고 모든 픽셀에서 동작 정보를 학습하기 때문에 발생하는 문제를 지적합니다. 이에 따라 우리는 인간 관절 위치를 더 정확하게 추정하기 위해 인간-키포인트 마스크 모듈을 제안하고, 변형 가능한 크로스 어텐션 메커니즘을 도입하였습니다. 이러한 접근은 인간 몸체에만 집중하여 비필수적인 정보로부터 모델을 방지하는 데 기여합니다.

- **Technical Details**: 제안된 방법은 이중 스트림 구조를 채택하여 시각적 표현 강화(Visual Representation Enhancement)와 동작 분리(Motion Disentanglement)를 통해 작업을 수행합니다. 첫 번째 단계에서 우리는 사람의 몸체와 키포인트에서 추출한 표현을 점진적으로 정제하는 과정을 포함합니다. 두 번째 단계에서는 비필요한 시각적 요소를 제거하고 타겟 인물의 동작 정보를 적응적으로 분리하는 모듈을 사용하여 세밀한 포즈 추정을 목표로 합니다.

- **Performance Highlights**: 제안된 방법은 세 가지 대규모 벤치마크 데이터셋에서 state-of-the-art 성능을 기록하였습니다. 특히, PoseTrack2017 데이터셋에서 wrist joint의 평균 정밀도(mean Average Precision, mAP) 84.8을 달성하였으며, 이는 현재 최첨단 방법이 기록한 81.5를 넘어서는 성과입니다. 이러한 결과는 변형 가능한 크로스 어텐션 메커니즘이 인간의 동작 추정에서 탁월한 효율성을 보여줌을 입증합니다.



### Context-CrackNet: A Context-Aware Framework for Precise Segmentation of Tiny Cracks in Pavement images (https://arxiv.org/abs/2501.14413)
- **What's New**: Context-CrackNet은 최신 인코더-디코더 아키텍처로, Region-Focused Enhancement Module (RFEM)와 Context-Aware Global Module (CAGM)을 특징으로 합니다. 이러한 혁신은 모델이 미세한 로컬 세부 사항과 전반적인 맥락 의존성을 효과적으로 포착할 수 있도록 합니다. 모델은 10개의 공개 크랙 세그멘테이션 데이터셋에서 평가되었으며, 9개의 최신 세그멘테이션 모델보다 꾸준히 우수한 성능을 보였습니다.

- **Technical Details**: Context-CrackNet의 RFEM은 작은 균열에 대한 정밀 세분화를 가능하게 하여 조기 발견을 지원합니다. CAGM은 고해상도 이미지의 전역 맥락을 분석하고, 긴 균열과 악어 균열 같은 큰 결함을 효과적으로 감지합니다. 이 두 모듈 간의 상호작용은 mIoU와 Dice 점수의 notable한 향상을 가져왔습니다.

- **Performance Highlights**: Context-CrackNet은 우수한 성능 지표인 mIoU와 Dice 점수를 달성하며, 경쟁력 있는 추론 효율성을 유지하여 대규모 도로 감시 시스템에서 실시간 배치가 가능하다는 잠재력을 보여줍니다. 기존의 다양한 최신 모델들과 비교했을 때, Context-CrackNet은 데이터셋 전반에서 뛰어난 성능을 발휘했습니다.



### Kolmogorov Arnold Neural Interpolator for Downscaling and Correcting Meteorological Fields from In-Situ Observations (https://arxiv.org/abs/2501.14404)
- **What's New**: 본 논문에서는 기존의 격자 기반 메테오로지컬(field) 데이터에 대한 한계를 극복하기 위해 Kolmogorov Arnold Neural Interpolator (KANI)라는 새로운 프레임워크를 제안합니다. KANI는 메테오로지컬 필드 표현을 연속적인 신경 기능으로 재정의하여 격자 기반 모델링의 비효율성을 해결하려고 합니다. 이론적으로 지원되는 KANI는 희소한 현장 관측 데이터를 활용해 시스템적인 편향을 수정하고, 제로샷(Zero-shot) 다운스케일링 규범을 도입하여 고해상도 지형 정보에 의해 안내되는 혁신적인 방법을 제공합니다.

- **Technical Details**: KANI는 Kolmogorov–Arnold 정리를 기반으로 하는 아키텍처를 채택하여 다변량 함수에 대한 연속적인 보편 근사기(existence of continuous universal approximators)를 보장합니다. 이 구조는 메테오로지컬 필드 컨볼루션 인코더, MLP 기반 가중치 생성기 및 KAN 기반 신경 복원기로 구성되어 있습니다. KANI는 과거 시스템 편향의 분포를 학습하여 다양한 지리적 위치와 공간 해상도에서 메테오로지컬 상태를 정확하게 복원 및 수정할 수 있습니다.

- **Performance Highlights**: 실험 결과, KANI는 미 대륙의 세 개 연구 지역에서 온도 정확도를 40.28%, 풍속 예측에서 67.41%까지 개선하며 기존의 대립(interpolation) 방법에 비해 현저한 성과를 거두었습니다. KANI는 정밀한 이론적 기초를 통해 훈련되고 다양한 격자 해상도에서 높은 정확도를 유지할 수 있는 강력한 일반화 능력을 보여 주었습니다. 이는 전통적인 격자 기반 표현의 한계를 초월하는 지속적인 신경 표현의 가능성을 제시합니다.



### CVOCSemRPL: Class-Variance Optimized Clustering, Semantic Information Injection and Restricted Pseudo Labeling based Improved Semi-Supervised Few-Shot Learning (https://arxiv.org/abs/2501.14401)
- **What's New**: 이 논문은 클래스 변동 최적화 클러스터링(class-variance optimized clustering)을 사용하는 새로운 반지도 소수 샘플 학습(semi-supervised few-shot learning) 접근 방식을 제안합니다. 전통적인 클러스터링 방법의 한계를 극복하여 레이블이 없는 샘플의 클러스터링 효율성을 개선할 수 있습니다. 또한, 제한된 가짜 레이블링(restricted pseudo-labeling) 방식을 통해 클러스터의 중심을 사전 학습된 데이터로 정제하여 모델의 성능을 향상시킵니다.

- **Technical Details**: 제안된 방법은 클래스 변동 최적화 클러스터링(cvoc)을 통해 레이블이 있는 샘플과 레이블이 없는 샘플을 효과적으로 클러스터링합니다. 의미 정보 주입(semantic information injection)을 통해 클러스터의 중심을 정제하며, 이 과정을 통해 생성된 제한된 가짜 레이블이 모델의 학습에 활용됩니다. 다양한 실험을 통해 여러 반지도 소수 샘플 학습 벤치마크 데이터셋에서 성능이 향상됨을 입증하였습니다.

- **Performance Highlights**: 제안된 접근 방식은 최신의 최첨단 방법들보다 성능이 크게 향상되었음을 실험적으로 보여주었습니다. 각 구성 요소의 기여도를 평가하기 위한 분리 실험(ablation experiments) 또한 수행하여 방법의 효과성을 증명하였습니다. 모델 성능의 향상은 클래스의 의미적 특징과 클래스 내 및 클래스 간 거리 손실(distances losses)을 관리함으로써 더욱 두드러집니다.



### Low-rank Prompt Interaction for Continual Vision-Language Retrieva (https://arxiv.org/abs/2501.14369)
- **What's New**: 이번 연구는 다중 모달(멀티모달) 작업에서 지속적인 학습을 개선하기 위한 새로운 접근법인 Low-rank Prompt Interaction (LPI)를 제안합니다. 기존 연구들에서는 교차 모달 및 교차 작업 상호 작용을 명시적으로 고려하지 않았습니다. LPI는 교차 모달 상호 작용과 교차 작업 상호 작용을 동시에 반영하여 효율적으로 학습할 수 있는 방법을 제공합니다.

- **Technical Details**: 저자는 LPI를 통해 Transformer 아키텍처에 기반한 멀티모달 이해를 달성하고자 하였습니다. 이 방법은 낮은 계급의 인터랙션 증강 분해(low-rank interaction-augmented decomposition)를 사용하여 메모리 사용을 최적화하며, 실질적인 파라미터 수의 증가 없이 교차 모달 연관성을 강화합니다. 또한, 계층적 낮은 계급 대비 학습(hierarchical low-rank contrastive learning)을 통해 학습의 견고성을 보장하였습니다.

- **Performance Highlights**: 두 가지 검색 작업인 이미지-텍스트 검색과 참조 표현 이해에서의 실험 결과는 LPI 방법이 다른 접근 방법들보다 뛰어난 성능을 보인다는 것을 입증하였습니다. 연구 결과는 적은 수의 추가 파라미터로도 상태 최적(state-of-the-art) 접근 방식들을 초월할 수 있음을 보여줍니다.



### Causal-Inspired Multitask Learning for Video-Based Human Pose Estimation (https://arxiv.org/abs/2501.14356)
Comments:
          9 pages, 3 figures

- **What's New**: 이 논문에서는 causal 관점에서 비디오 기반 인체 자세 추정(video-based human pose estimation) 문제에 접근하여, 신뢰할 수 있는 결과를 달성하기 위한 주요 요구사항인 원인 추론(causal reasoning) 능력과 모델의 해석 가능성(interpretability)을 강조합니다. 이 접근법을 통해 우리는 두 단계로 구성된 causal-inspired multitask learning framework를 제안합니다. 첫 번째 단계에서는 자가 지도(self-supervision) 보조 작업을 통해 네트워크에 원인 시간적 모델링(causal spatio-temporal modeling) 능력을 부여하고, 두 번째 단계에서는 Token Causal Importance Selection 모듈을 사용하여 인체 자세 추정에 필수적인 인과적 토큰을 선별합니다.

- **Technical Details**: 첫 번째 단계에서 우리는 두 개의 자기 지도 보조 작업을 도입하여 네트워크가 도전적인 키포인트를 추론할 수 있도록 합니다. 이러한 자가 감독 보조 작업에서는 특성 토큰을 무작위로 마스킹(masking)하거나 가우시안 노이즈(Gaussian noise)를 추가하여 모델의 causal spatio-temporal modeling 능력을 증진시킵니다. 두 번째 단계에서는 Token Causal Importance Selection 모듈과 비인과적(non-causal) 토큰 군집화(clustering) 모듈을 제안하여, 인과적 특성과 덜 중복된 비인과적 특성을 식별하고 결합합니다.

- **Performance Highlights**: 우리는 세 가지 대형 벤치마크 데이터셋인 PoseTrack2017, PoseTrack2018, PoseTrack2021에서 제안한 방법이 최신 방법들보다 우수한 성능을 달성했음을 보여주었습니다. 이 연구는 비디오 기반 인체 자세 추정 분야에서 인과적 관점을 통해 더 나은 모델 해석 가능성과 강인성을 제공하며, 미래 연구에 대한 새로운 방향성을 제시하는 데 기여할 것으로 판단됩니다.



### Correlation-Based Band Selection for Hyperspectral Image Classification (https://arxiv.org/abs/2501.14338)
Comments:
          5 pages, 1 figure

- **What's New**: 본 연구에서는 hyperspectral image classification을 위한 새로운 band selection 방법을 제안합니다. 이 방법은 correlation 기반의 방식으로, 인접한 bands 간의 상관관계를 분석하여 서로 다른 bands 간의 관계를 파악합니다. 이를 통해 상관관계가 낮은 bands를 선택하여 다양한 정보와 비중복 정보를 보장합니다.

- **Technical Details**: 제안된 방법은 Correlation Coefficient (CC)를 활용하여 bands 간의 관계를 측정하고, Average Band Correlation (ABC)을 통해 각 band의 상관관계를 정의합니다. ABC가 설정한 임계값 0.65보다 낮은 bands를 선택하며, 이를 통해 낮은 상관관계를 가진 bands의 조합을 확보하여 정보의 중복성을 줄입니다. 이 과정은 데이터의 차원을 줄이고 처리 속도를 향상시키는 데 기여합니다.

- **Performance Highlights**: 제안한 방법은 두 개의 표준 benchmark 데이터셋, Pavia University와 Salinas Valley에서 평가되었습니다. 실험 결과는 제안된 방법이 다른 band selection 접근법들과 비교하여 경쟁력 있는 성능을 발휘함을 보여주었습니다. 이는 hyperspectral 데이터 분석에서 유용한 방법이 될 것으로 기대됩니다.



### Scalable Benchmarking and Robust Learning for Noise-Free Ego-Motion and 3D Reconstruction from Noisy Video (https://arxiv.org/abs/2501.14319)
Comments:
          Accepted by ICLR 2025; 92 Pages; Project Repo: this https URL. arXiv admin note: substantial text overlap with arXiv:2406.16850

- **What's New**: 이 논문에서는 기존 모델들이 소음이 없는 데이터에 의존하는 한계점을 극복하기 위해 강력한 자아 모션 추정(ego-motion estimation)과 사진 현실적인 3D 재구성을 재정의하고자 합니다. 실제 환경에서 발생하는 예측 불가능한 소음의 복잡성을 포착하지 못하는 문제를 다루며, 동적 모션, 센서의 불완전성(sensor imperfections), 동기화의 혼란(synchronization perturbations) 등의 요소가 이러한 모델의 성능 감소로 이어짐을 강조합니다.

- **Technical Details**: 연구팀은 세 가지 주요 도전과제에 대응하기 위해 우선 다양한 데이터를 생성하는 스케일러블(noisy data synthesis) 프로세스를 소개합니다. 이 프로세스는 복잡한 모션, 센서의 불완전성, 동기화 오류를 시뮬레이션하여 복원력을 갖춘 Robust-Ego3D라는 새로운 벤치마크를 개발합니다. 마지막으로, 새로운 테스트 시간 적응(method of test-time adaptation) 방식인 Correspondence-guided Gaussian Splatting (CorrGS)를 제안하여 노이즈가 포함된 정보와 클린 3D 지도를 정렬하여 성능을 향상시킵니다.

- **Performance Highlights**: CorrGS는 신선한 실험을 통해 이전의 최첨단(state-of-the-art) 방법들보다 일관되게 뛰어난 성능을 발휘하는 것으로 나타났습니다. 특히 빠른 동작과 동적 조명(dynamics illumination) 시나리오에서 특히 뛰어난 성능 차이를 보이며, 이로 인해 자아 모션 정확성(ego-motion accuracy)과 3D 재구성 품질의 한계를 명확히 드러냅니다.



### Nautilus: Locality-aware Autoencoder for Scalable Mesh Generation (https://arxiv.org/abs/2501.14317)
Comments:
          14 pages

- **What's New**: Nautilus는 아티스트처럼 3D 메시(메쉬)를 생성하기 위한 지역 인식(autoencoder) 오토인코더로, 지역적 속성을 활용해 구조적 충실도(structural fidelity)와 효율적인 표현을 달성합니다. 이를 위해 인접한 면의 관계를 보존하고 지역적으로 공유된 정점(vertex)과 모서리(edge)를 통해 시퀀스 길이를 압축하는 새로운 토큰화(tokenization) 알고리즘을 도입했습니다. Nautilus는 최대 5,000개의 면을 가진 메시를 생성할 수 있는 전례 없는 성능을 자랑합니다.

- **Technical Details**: 이 연구에서는 Nautilus 스타일 메시 토큰화 알고리즘을 설계하여 인접한 면의 근접성을 유지하고, 세밀한 지역 기하학을 캡처하는 다중 스케일 기하학적 가이드를 제공하는 듀얼 스트림 포인트 컨디셔너(Dual-stream Point Conditioner)를 도입했습니다. 이 방식은 데이터를 생성하는 동안 지역 의존성을 모델링할 수 있는 토대를 마련하고, 최대 5,000개의 면을 가진 정교한 메시를 생성할 수 있게 해줍니다. Nautilus는 기존의 직접 조건부 생성 작업을 넘어서는 성능을 차지하며, 메시의 복잡한 주제를 다루는 데 강력한 기반이 됩니다.

- **Performance Highlights**: Nautilus의 실험 결과는 기존의 최첨단 방법보다 훨씬 높은 충실도와 확장성을 제공하는 것을 보여주었으며, 정교한 메시 생성을 통해 복잡한 토폴로지를 풍부하게 표현할 수 있음을 입증했습니다. Nautilus는 고급 데이터셋에서 광범위한 테스트를 통해 그 성능을 확인하였고, 아티스트들이 생성한 모델에 필적하는 품질을 달성하였습니다. 이로써 Nautilus는 현대 3D 애플리케이션의 요구를 충족시키는 중요한 기술로 자리 잡을 것으로 기대됩니다.



### PAID: A Framework of Product-Centric Advertising Image Design (https://arxiv.org/abs/2501.14316)
- **What's New**: 이 논문에서는 Product-Centric Advertising Image Design (PAID)이라는 새로운 자동 광고 이미지 디자인 프레임워크를 제안합니다. PAID는 제품 전경 이미지, 마케팅 태그라인 및 목표 크기를 입력받아 광고 이미지를 자동으로 생성하는 시스템으로, 기존 방법의 한계를 극복하기 위해 각 단계의 조정 및 작업 정의를 최적화했습니다. PAID는 'prompt generation', 'layout generation', 'background image generation', 'graphics rendering'의 네 가지 단계로 구성되어 있으며, 각 단계에서 전문 모델이 훈련됩니다.

- **Technical Details**: PAID의 주요 기술은 Visual Language Model (VLM)을 활용하여 제품 전경 및 배경에 어울리는 prompt를 생성하는 것입니다. 레이아웃 생성 모델은 제품 전경 및 마케팅 태그라인에 따라 텍스트와 이미지의 배치를 예측하여 조화를 이룹니다. 각 단계에서 리소스 사용을 최적화하고, SDXL 기반의 레이아웃 제어 인페인팅 모델을 훈련하여 미적 배경 이미지를 생성하며, 디자인 시 요소 간의 관계를 고려합니다.

- **Performance Highlights**: PAID는 기존 방법들보다 더 시각적으로 매력적인 광고 이미지를 생성하는 데 성공했습니다. 신뢰할 수 있는 실험 결과에 따르면, PAID의 생성된 이미지는 광고 클릭률을 높이고, 사용자 경험을 개선하는 데 기여할 것으로 기대됩니다. 또한, 새롭게 구축된 PITA와 PIL 데이터셋은 고품질 이미지를 훈련하고 검증하는 데 사용됩니다.



### BrainGuard: Privacy-Preserving Multisubject Image Reconstructions from Brain Activities (https://arxiv.org/abs/2501.14309)
Comments:
          AAAI 2025 oral

- **What's New**: BrainGuard는 다수의 피실험자의 fMRI 데이터를 활용하여 이미지 재구성을 향상시키기 위한 개인 정보 보호를 고려한 협업 교육 프레임워크입니다. 기존의 접근법이 피실험자 간의 데이터를 집계해야 한다는 우려가 있었던 반면, BrainGuard는 각 피실험자가 자신의 로컬 데이터를 활용하여 개별 모델을 훈련하고 이를 공유하는 글로벌 모델과 연결하여 개인 데이터를 보호합니다. 이 연구는 개인의 고유성과 다수의 피실험자 간의 공통점을 모두 고려할 수 있는 새로운 방법론을 제시합니다.

- **Technical Details**: BrainGuard는 분산 및 협동 학습을 촉진하는 글로벌-로컬 아키텍처를採用합니다. 여기서 각 피실험자에 대한 개별 모델은 자신의 fMRI 데이터를 기반으로 훈련되며, 업데이트된 매개변수는 글로벌 모델로 통합되어 다른 피실험자 간의 공통 패턴을 식별합니다. 또한, BrainGuard는 fMRI 데이터의 복잡성을 해결하기 위해 하이브리드 동기화 전략을 통합하여 각 모델이 글로벌 모델의 매개변수를 동적으로 통합할 수 있도록 설계되었습니다.

- **Performance Highlights**: BrainGuard의 실험 결과는 새로운 고수준 및 저수준 메트릭에서 기준을 설정했습니다. 이 프레임워크는 뇌 디코딩 분야의 최첨단 기술을 발전시키며, 이미지 재구성의 정확성을 향상시킵니다. 생체 신호의 복잡성이 분석 방법의 효과성을 저하시키는 문제를 해결하면서, 개인 데이터의 보안을 강화하는 동시에 연구 결과의 신뢰성을 높이고 있습니다.



### Learning Primitive Relations for Compositional Zero-Shot Learning (https://arxiv.org/abs/2501.14308)
Comments:
          Accepted to ICASSP 2025

- **What's New**: 본 논문에서는 Compositional Zero-Shot Learning (CZSL) 분야에서 새로운 프레임워크인 Learning Primitive Relations (LPR)을 제안합니다. CZSL은 이전에 본 조합에서 얻은 지식을 활용하여 보지 못한 상태-객체 조합을 식별하는 것을 목표로 합니다. 기존 접근 방법들은 상태와 객체를 독립적으로 예측하는 경향이 있었으나, LPR은 이들의 관계를 확률적으로 포착합니다.

- **Technical Details**: LPR은 cross-attention mechanism을 활용하여 상태와 객체 간의 의존성을 고려합니다. 이 연구는 상태-객체 관계를 기반으로 보지 못한 조합의 가능성을 추론할 수 있도록 모델을 설계하였습니다. 기술적으로, 이 프레임워크는 상태와 객체 간의 복잡한 상호작용을 효과적으로 학습할 수 있게 해줍니다.

- **Performance Highlights**: 실험 결과, LPR은 닫힌 세계(closed-world)와 열린 세계(open-world) 설정에서 모두 세 가지 CZSL 벤치마크 데이터 세트에서 최신 기술(state-of-the-art) 방법보다 우수한 성능을 보였습니다. 정성적 분석을 통해 LPR이 보지 않은 조합 예측을 위해 상태-객체 관계를 효과적으로 활용함을 보여주었습니다.



### Additive Manufacturing Processes Protocol Prediction by Artificial Intelligence using X-ray Computed Tomography data (https://arxiv.org/abs/2501.14306)
Comments:
          21 pages, 21 figures, 5 tables

- **What's New**: 이 연구에서는 Additive Manufacturing (AM) 프로세스의 품질을 높이기 위한 새로운 비대칭(Non-iterative) 방법론을 제안합니다. 이 방법론은 인간 개입 없이 인공지능(Artificial Intelligence, AI)을 활용하여 프로세스 파라미터를 최적화하며, 자율적으로 적합한 AI 모델을 훈련시킬 수 있는 능력을 갖추고 있습니다.

- **Technical Details**: 연구는 재료 압출(Material Extrusion, MEX) AM 프로세스를 기반으로 한 세 가지 상용 3D 프린터를 사용하여 진행되었습니다. 레이어 높이와 노즐 속도를 변화시켜 여섯 가지 AM 프로세스 파라미터로 샘플이 3D 프린트되었습니다. 품질 검사를 통해 얻은 학습 데이터와 비파괴 검사(Non-Destructive Testing, NDT) 방법을 활용하여 AI 기반의 이미지 세분화(Image Segmentation) 단계를 도입한 점이 이 방법론의 새로움입니다.

- **Performance Highlights**: 훈련된 AI 모델은 전통적인 임계값(Thresholding) 기반 소프트웨어 툴과 비교하였을 때 99.3%의 정확도를 기록했습니다. 반면, 가장 우수한 상용 전통 이미지 방법의 정확도는 83.44%였습니다. 또한, ANN(Artificial Neural Network) 훈련을 위한 R의 최적 값은 0.82로 측정되었으며, MEX 프로세스는 설계 대비 22.06%의 기공 오류를 나타냈습니다. 전체 프로세스 최적화를 위한 두 개의 AI 모델이 전통적인 최적화 및 기계적 테스트 방법으로 검증되었습니다.



### TD-RD: A Top-Down Benchmark with Real-Time Framework for Road Damage Detection (https://arxiv.org/abs/2501.14302)
- **What's New**: 최근 10년간 Object Detection 분야는 딥러닝(deep learning)과 대규모 데이터셋의 발전에 힘입어 놀라운 성장을 이루었습니다. 그러나 도로 손상 탐지 분야는 비교적 덜 탐구되어 왔습니다. 본 논문에서는 기존 데이터셋에 대한 보완적인 관점을 제공하는 Top Down Road Damage Detection Dataset (TDRD)를 소개하며, 도로 손상 탐지 전용의 새로운 벤치마크를 제시합니다.

- **Technical Details**: TDRD 데이터셋은 7,088장의 고해상도 이미지로 구성되어 있으며, 도로 손상의 세 가지 주요 카테고리인 갈라짐(cracks), 함몰(potholes), 수리(patches)를 포함합니다. 본 연구는 TD-YOLOV10이라는 실시간 객체 탐지(framework) 모델을 도입하여 TDRD 데이터셋의 독특한 도전 과제를 다룹니다. TD-YOLOV10은 self-attention 프레임워크와 Multi-Scale Attention with Positional Squeeze-and-Excitation (MAPSE) 메커니즘을 통해 도로 면 이미지의 탐지 및 인식을 향상시킵니다.

- **Performance Highlights**: TD-YOLOV10은 TD-RD 및 두 개의 다른 공개 도로 포장 데이터셋인 CNRDD와 CRDDC'22와 비교하여 경쟁력 있는 성능을 보여주었습니다. 특히 TD-YOLOV10의 구조는 다양한 크기의 객체에 대해 정확한 탐지를 가능하게 하여 실제 탐지 작업에서 모델의 신뢰도와 정확성을 향상시킵니다. 본 논문을 통해 TDRD가 도로 손상 탐지 분야의 연구를 가속화할 것으로 기대합니다.



### Dense-SfM: Structure from Motion with Dense Consistent Matching (https://arxiv.org/abs/2501.14277)
- **What's New**: 본 연구에서 제안하는 Dense-SfM은 다중 뷰 이미지로부터 밀도 높은 3D 재구성을 위한 새로운 Structure from Motion (SfM) 프레임워크입니다. 이 프레임워크는 기존의 희소 매칭(sparse matching) 한계를 극복하기 위해 Gaussian Splatting(GS) 기반의 트랙 확장(track extension) 기법을 통합하여 일관성과 더 긴 특징 트랙을 제공합니다. Dense-SfM은 또한 다중 뷰 커널라이즈드 매칭(matching) 모듈을 갖추고 있어 다양한 뷰에서의 강력한 트랙 정제를 가능하게 합니다.

- **Technical Details**: Dense-SfM은 밀도 매칭(dense matching) 및 Gaussian Splatting(GS)를 활용하여 3D 점 구름(point cloud)을 생성합니다. 이 과정은 초기 SfM 모델을 구축한 후 소규모 3D 가우시안(Gaussians)을 활용해 색상 및 위치와 같은 속성을 통합하여 이루어집니다. 미리 정의된 카메라 포즈를 통해 각 초기 SfM 포인트의 가시성을 추론하여 이미지 트랙을 연장함으로써, 짧은 트랙 문제를 해결하고, 이후 다중 뷰 커널라이즈드 매칭 및 기하학적 번들 조정(bundle adjustment)을 통해 최적화를 수행합니다.

- **Performance Highlights**: Dense-SfM은 ETH3D 및 Texture-Poor SfM 데이터셋에서 평가된 결과, 기존의 최첨단 SfM 시스템 대비 정확도와 밀도에서 유의미한 개선을 보여주었습니다. 밀집 매칭과 Gaussian Splatting을 통해 트랙 확장을 구현하는 덕분에, 카메라 포즈 추정 정확도와 밀도 높은 3D 재구성을 달성하였습니다. 이러한 성과는 다양한 메트릭을 통해 입증되었으며, Dense-SfM은 텍스처가 부족한 영역에서도 탁월한 성능을 발휘합니다.



### Global Semantic-Guided Sub-image Feature Weight Allocation in High-Resolution Large Vision-Language Models (https://arxiv.org/abs/2501.14276)
Comments:
          10 pages, 10 figures and tables

- **What's New**: 본 논문에서는 고해상도 이미지 처리의 수요가 증가함에 따라, 기존의 균등한 sub-image partitioning 방법이 모델의 시각적 이해 능력을 저하시킬 수 있음을 밝히고, Global Semantic-guided Weight Allocator (GSWA) 모듈을 제안합니다. GSWA 모듈은 정보 밀도에 따라 동적으로 sub-image의 가중치를 할당하여 모델이 더 유익한 영역에 집중할 수 있도록 합니다. 이를 통해 SleighVL이라는 경량 모델을 개발하였으며, 이는 기존 모델들과의 성능 비교에서 우수한 결과를 얻었습니다.

- **Technical Details**: 연구에서는 Vision Transformer (ViT)를 활용한 인코더를 기반으로 하여, sub-image partitioning을 통해 고해상도 이미지를 더 잘 처리하는 방안을 제시합니다. GSWA 모듈은 각 sub-image의 정보 밀도를 고려하여 가중치를 조정하며, 이는 인간의 시각적 주의 메커니즘을 모사합니다. 따라서 모델은 중요한 정보가 포함된 지역에 더 집중할 수 있게 됩니다. 이러한 방식으로 SleighVL은 여러 벤치마크에서 유의미한 성능 향상을 보여 줍니다.

- **Performance Highlights**: SleighVL 모델은 기존 LVLM들 및 최신 SOTA 모델들과 비교했을 때, 경량화에도 불구하고 높은 성능을 유지합니다. 다양한 평가 기준에서 기본 모델에 비해 현저한 성능 개선을 달성하였으며, 고해상도 시각적 정보를 효과적으로 통합할 수 있는 능력을 입증하였습니다. 특히, GSWA 모듈은 복잡한 이미지 처리 시 더 나은 효과를 보여줍니다.



### Bayesian Neural Networks for One-to-Many Mapping in Image Enhancemen (https://arxiv.org/abs/2501.14265)
- **What's New**: 이번 연구에서는 저조도 및 수중 이미지 향상과 같은 작업에서 단일 입력 이미지가 여러 유사한 목표 이미지로 매핑되는 문제를 해결하기 위해 베이지안 신경망(Bayesian Neural Networks, BNNs)을 활용한 베이지안 향상 모델(Bayesian Enhancement Model, BEM)을 제안합니다. 이를 통해 데이터 불확실성을 효과적으로 캡처하고 다양한 출력을 생성하는 접근법을 구현했습니다. 특히, BEM은 두 단계의 접근 방식을 도입하여 실시간 추론이 가능하도록 하였으며, 이를 통해 복잡한 고차원 공간에서 더 빠른 수렴을 이끌어내는 동적 모멘텀 프라이어(Momentum Prior)를 제안했습니다.

- **Technical Details**: 베이지안 신경망(BNN)은 네트워크 가중치의 분포를 학습하여 불확실성을 정량화하며, 특히 이미지 향상에서 단일 입력을 여러 출력으로 매핑하는 가능성을 탐색합니다. 연구에서는 변분 추론(Variational Inference, VI)과 드롭아웃(dropout)을 통해 BNN을 효율적으로 구현하는 방법을 논의하였으며, 최종 층에 불확실성을 추가하는 방식으로 전체 BNN을 근사할 수 있음을 보여주었습니다. 또한, BNN이 고차원 가중치 공간에서 언더피팅(underfitting)의 문제를 해결하기 위한 모멘텀 프라이어를 도입하여 더 나은 로컬 최적(Local Optimum)으로 수렴하도록 유도하였습니다.

- **Performance Highlights**: 다양한 저조도 및 수중 이미지 향상보다 BEM이 결정론적 모델보다 우수한 성능을 보이는 것을 실험을 통해 입증했습니다. 연구 결과, BEM은 저조도 이미지 향상과 수중 이미지 향상 작업에서 단일 입력과 출력 간의 복잡한 매핑 문제를 효과적으로 해결하는 것으로 나타났습니다. 향상된 유연성과 견고성을 제공하여 기존의 결정론적 모델 대비 더 나은 결과를 도출하는데 기여하였습니다.



### Point-LN: A Lightweight Framework for Efficient Point Cloud Classification Using Non-Parametric Positional Encoding (https://arxiv.org/abs/2501.14238)
Comments:
          This paper has been accepted for presentation at the 29th International Computer Conference, Computer Society of Iran (CSICC) 2025

- **What's New**: 이번 논문에서는 3D 포인트 클라우드 분류를 위해 설계된 경량화된 새로운 프레임워크인 Point-LN을 소개합니다. Point-LN은 Farthest Point Sampling (FPS), k-Nearest Neighbors (k-NN)와 같은 비모수적(non-parametric) 컴포넌트를 통합하여 학습 가능한 분류기와 연결하여 분류 정확도를 높였으며, 최소한의 파라미터로 컴퓨팅 비용을 줄입니다. 이 하이브리드 아키텍처는 실시간 및 리소스 제약이 있는 애플리케이션에 최적화되어 있습니다.

- **Technical Details**: Point-LN은 특징 인코더(feature encoder)와 분류기(classifier)라는 두 가지 주요 구성 요소로 이루어져 있습니다. 특징 인코더는 원시 입력 포인트 클라우드를 고차원 특징 표현으로 변환하며, 분류기는 이러한 인코딩된 특징을 목표 레이블 공간에 매핑하여 분류 로짓을 산출합니다. 논문에서는 일반적으로 사용되는 비모수적 포지셔널 인코딩을 경량화된 방법에서의 응용도 다룹니다.

- **Performance Highlights**: Point-LN은 ModelNet40 및 ScanObjectNN과 같은 벤치마크 데이터셋에서 경쟁력 있는 성능을 보여줍니다. 비모수적 방법의 강점을 활용하면서 학습 가능한 분류기를 통합하여 복잡한 시나리오에서도 높은 정확도를 달성할 수 있도록 설계되었습니다. 이 연구는 다양한 포인트 클라우드 분류 작업에 적합하고, 광범위한 컴퓨터 비전 애플리케이션에서의 채택 가능성을 강조합니다.



### Micro-macro Wavelet-based Gaussian Splatting for 3D Reconstruction from Unconstrained Images (https://arxiv.org/abs/2501.14231)
Comments:
          11 pages, 6 figures,accepted by AAAI 2025

- **What's New**: 이번 연구에서는 Micro-macro Wavelet 기반의 Gaussian Splatting (MW-GS)이라는 새로운 3D 복원 기법을 제안합니다. 이 방법은 장면 표현을 글로벌, 정제 및 내재적 구성 요소로 분리하여 동시에 다양한 동적 장면을 효과적으로 모델링합니다. 주요 혁신으로는 Gaussian 포인트가 여러 스케일에서 세부 사항을 포착할 수 있도록 도와주는 Micro-macro Projection과 주파수 영역 정보를 활용하여 특징 표현을 정제하는 Wavelet 기반 Sampling이 있습니다. 이들은 협력적으로 작업의 품질을 극대화합니다.

- **Technical Details**: MW-GS는 Gaussian 특징을 세 가지 고유한 구성 요소로 분해합니다: 글로벌 외관, 정제된 외관, 그리고 내재적 특징. 이 기법은 Adaptive Sampling을 통해 좁고 넓은 원추형 프러스트럼에서 세밀한 텍스처와 장면 충실도에 영향을 미치는 장거리 특성을 모두 포착하도록 최적화합니다. 또한, Wavelet 기반 Sampling을 통합하여 다중 해상도 샘플링을 가능하게 하고, 복원의 정확도를 향상시키며, Hierarchical Residual Fusion Network를 통해 여러 레벨에서 효과적으로 특징을 통합하여 3D 재구성을 최적화합니다.

- **Performance Highlights**: 다양한 실험을 통해 MW-GS는 기존 방법들을 능가하는 뛰어난 재구성 결과와 렌더링 품질을 달성하였습니다. 이 기술은 복잡한 동적 환경에서도 더 나은 성능을 보여 주며, 이미지의 미세한 변화를 효과적으로 처리할 수 있는 능력을 입증했습니다. 연구 결과는 MW-GS가 3D 복원 작업에서 최신 기술임을 입증하며, 다양한 응용 분야에 활용될 수 있는 가능성을 제공합니다.



### GreedyPixel: Fine-Grained Black-Box Adversarial Attack Via Greedy Algorithm (https://arxiv.org/abs/2501.14230)
- **What's New**: GreedyPixel은 타겟 모델의 쿼리 기반 피드백만을 사용하여 고품질의 적대적 예제를 생성하는 새로운 픽셀 기반 탐욕 알고리즘입니다. 이 알고리즘은 그라디언트 정보 없이도 백박스 환경에서 성공적인 공격을 수행할 수 있도록 설계되었습니다. GreedyPixel은 기존의 접근법들과 비교해 공격의 성공률, 시간 효율성 및 시각적 품질에서 우수한 성능을 발휘합니다.

- **Technical Details**: GreedyPixel 알고리즘은 서라겟 모델을 통해 얻은 그라디언트를 기반으로 픽셀의 우선순위를 정하고, 각 픽셀을 순차적으로 개별적으로 변형하여 공격을 최적화합니다. 이 과정은 좀 더 눈에 띄지 않는 변형을 초래하고, 기존 방법론보다 더 적은 계산 비용으로 공격을 성공적으로 수행합니다. 또한 이 알고리즘은 다양한 이미지 해상도에서 평가되어, 낮은 해상도에서 특히 효과적입니다.

- **Performance Highlights**: GreedyPixel은 CIFAR-10 및 ImageNet 데이터셋에서 기존의 백박스 및 화이트박스 공격 방법들과 비교한 결과, 공격 성공률(Attack Success Rate, ASR)이 동등한 수준에 이르렀습니다. 또한, 기존의 적대적 방어에 대한 평가에서도 더 높은 ASR을 기록하는 등 공격 효과성이 뛰어난 것으로 나타났습니다. 최종적으로, GreedyPixel은 적대적 공격 방법론의 새로운 경지를 열어가는 기여를 하였습니다.



### Detection and Classification of Acute Lymphoblastic Leukemia Utilizing Deep Transfer Learning (https://arxiv.org/abs/2501.14228)
Comments:
          4 pages, 4 figures, Submitted to UCICS

- **What's New**: 이 연구는 백혈병 진단을 위한 새로운 접근 방식을 제안합니다. 기존의 복잡한 진단 과정이 아닌, 딥 러닝(Deep Learning)을 활용하여 초기 단계에서 질병을 식별할 수 있는 방법을 모색했습니다. 이 방법은 총 네 가지 단계인 Benign, Early, Pre, Pro를 포함합니다.

- **Technical Details**: 연구에서는 두 개의 Convolutional Neural Network (CNN) 모델을 사용했습니다. 첫 번째는 MobileNetV2 모델로, 헤드를 수정하여 활용하였고, 두 번째는 여러 convolutional layers로 구성된 커스텀 모델입니다. 커스텀 모델은 각 층에 최대 풀링(max pooling)을 결합하여 설계되었습니다.

- **Performance Highlights**: 커스텀 모델은 98.6%의 정확도를 달성했으며, MobileNetV2는 99.69%로 더 높은 정확도를 기록했습니다. 사전 훈련된(pretrained) 모델 또한 유망한 결과를 보여 실제 임상 적용 가능성을 높이는 것으로 판단됩니다.



### PuzzleGPT: Emulating Human Puzzle-Solving Ability for Time and Location Prediction (https://arxiv.org/abs/2501.14210)
Comments:
          NAACL 2025 Findings

- **What's New**: 이 논문에서는 이미지로부터 시간과 장소를 예측하는 복잡한 과제를 인간과 같은 퍼즐 해결 능력으로 형식화하고, 이것을 다양한 모듈들로 구현한 전문가 파이프라인인 PuzzleGPT를 제안합니다. PuzzleGPT는 시각적 단서를 식별하는 perceiver, 예측 후보를 추론하는 reasoner, 다양한 단서를 결합하는 combiner, 외부 지식을 검색하는 web retriever, 및 강건성을 위한 noise filter로 구성됩니다. 이는 최첨단 성능을 기록하며, 기존의 대형 VLM(Visual-Language Models)와 자동 생성된 추론 파이프라인에 비해 최소 32%에서 38%의 성능 향상을 보여줍니다.

- **Technical Details**: PuzzleGPT는 다섯 가지 핵심 기술인 perceiver, reasoner, combiner, noise filter, knowledge retriever로 구성됩니다. 이 중 perceiver는 시각 신호를 처리하고 개체를 식별하며, reasoner는 위치 및 시간 후보와 그들의 관계를 추론합니다. 여러 개체에서 얻은 단서를 효율적으로 결합하기 위해 신뢰도 기반의 계층적 조합 방법을 제안하여, 단순히 모든 단서를 결합하는 것이 아니라 각각의 단서를 점진적으로 분석하게 됩니다. 이러한 설계는 추론 과정을 신뢰할 수 있도록 하고 강건성을 추가합니다.

- **Performance Highlights**: PuzzleGPT는 TARA 데이터셋에서 기존의 VLM 모델들, 박사리 BLIP-2 및 GPT-4V를 포함하여 최소 32%의 성능 향상을 보이며, 이는 다양한 기술들을 동시에 활용할 수 없는 기존 모델들의 한계를 드러냅니다. 또한 PuzzleGPT는 WikiTilo 데이터셋에서도 최첨단 성능을 자랑하며, 이를 통해 복잡한 문제 해결을 위한 전문가 설계의 중요성을 강조합니다. 이 방법은 이미지에서 시간과 장소를 예측하는 인간의 퍼즐 해결 능력을 시뮬레이션할 수 있는 기초를 제공합니다.



### Dynamic Token Reduction during Generation for Vision Language Models (https://arxiv.org/abs/2501.14204)
- **What's New**: 이번 논문에서는 시각-언어 모델(VLM)을 위한 동적 프루닝 전략인 Dynamic Rate(DyRate)를 소개합니다. 이 접근 방식은 생성 과정에서 압축 비율을 점진적으로 조정하여 계산 복잡성을 줄이는 동시에 응답의 품질을 유지하는 데 중점을 두고 있습니다. 기존 연구들이 단일 전방 패스에서 시각적 토큰의 중복성을 제거하는 데 집중했던 반면, DyRate는 전체 생성 과정에서 시각적 토큰의 중요성을 면밀히 분석하여 동적으로 조정되는 압축 비율을 적용했습니다.

- **Technical Details**: DyRate는 응답 생성을 위한 토큰 축소 비율을 주의 분포(attention distribution)와 연결하여 적응형 동적 축소를 구현한 첫 번째 방법입니다. 이 모델은 경량 분류기를 사용하여 각 생성 단계에서 시각적 토큰들의 주의 분포를 수집하고, 이를 통해 최적의 축소 비율을 예측합니다. 또한, Gumbel-Softmax 기법을 사용하여 예측된 압축 비율을 미분 가능하게 만들고, 모델의 순전파 과정에 통합하여 더 나은 학습 가능성을 제공합니다.

- **Performance Highlights**: 실험 결과, DyRate는 계산 요구 사항을 크게 줄이면서도 응답의 정확성을 유지하는 데 성공했습니다. 분석 결과, 생성 과정이 진행될수록 시각적 토큰의 중요성이 감소함을 확인하고, 이러한 중요성의 변화를 반영하여 동적 압축 비율을 사용하는 것이 효과적임을 입증했습니다. 이는 VLM의 실제 적용 범위를 확장하는 데 기여할 것으로 기대됩니다.



### VideoShield: Regulating Diffusion-based Video Generation Models via Watermarking (https://arxiv.org/abs/2501.14195)
Comments:
          International Conference on Learning Representations (ICLR) 2025

- **What's New**: 이번 논문에서는 VideoShield라는 새로운 워터마킹 프레임워크를 제안합니다. 이 프레임워크는 비디오 생성 과정에서 직접 워터마크를 삽입하여 고품질 비디오 생성을 방해하지 않고도 저작권 보호를 가능하게 합니다. 기존의 전처리 방식이 아닌, 생성 능력 자체를 활용하는 혁신적인 접근 방식입니다.

- **Technical Details**: VideoShield는 수신 화면과 템플릿 비트를 사용하여 데이터에서 복잡한 이상 징후를 탐지합니다. 웨이트 비트는 가우시안 노이즈에 매핑되며, 이는 노이즈 제거 과정에서 워터마크를 생성하는 데 사용됩니다. Denoising Diffusion Implicit Model (DDIM) Inversion을 사용하여 원래의 워터마크가 포함된 노이즈로 비디오를 되돌린 후, 두 가지 모듈을 통해 시간적 및 공간적 변조를 로컬라이즈합니다.

- **Performance Highlights**: 실험 결과, VideoShield는 다양한 비디오 생성 모델에서 효과적으로 워터마크를 추출하고 변조를 탐지하였습니다. 비디오 품질을 저하시키지 않으면서도 변조를 정확하게 표시하는 성능을 보여줍니다. 뿐만 아니라, 이 방법은 이미지 생성 모델에도 적용 가능하여 동일하게 변조 탐지를 수행할 수 있습니다.



### ENTER: Event Based Interpretable Reasoning for VideoQA (https://arxiv.org/abs/2501.14194)
- **What's New**: 이번 논문에서는 ENTER라는 해석 가능한 Video Question Answering (VideoQA) 시스템을 제안합니다. 이 시스템은 이벤트 그래프(event graphs)를 기반으로 하여 비디오를 그래픽 표현으로 변환하고, 이 과정에서 이벤트 간의 관계를 명확히 모델링합니다. ENTER는 기존 시스템들이 간과한 저수준의 시각 정보(contextual visual information)를 활용하여 더욱 신뢰할 수 있는 해답을 제공합니다. 실험 결과, ENTER는 기존의 해석 가능한 VideoQA 접근법보다 우수한 성능을 보여줍니다.

- **Technical Details**: ENTER는 이벤트 그래프라는 구조화된 표현을 사용하여 비디오 이벤트를 노드로, 이벤트 간 관계를 엣지로 구성합니다. 이를 통해 모델은 사건 간의 관계를 명확히 이해할 수 있으며, 대규모 언어 모델(LLM)을 통해 생성된 코드를 실행하여 질문에 대한 응답을 얻습니다. 이 과정에서는 초기 그래프의 불완전함을 보완하기 위해 계층적인 반복 업데이트(hierarchical iterative update)를 적용하여 추가 정보를 통합하고 그래프의 완전성을 향상시킵니다.

- **Performance Highlights**: ENTER는 NeXT-QA, IntentQA, EgoSchema와 같은 비디오 QA 데이터셋에서 최첨단 성능을 상회하는 결과를 달성했습니다. 실험을 통해 ENTER의 구조화된 이벤트 그래프가 비디오 내에서의 복잡한 관계를 명확히 포착하고 더 정확하며 해석 가능한 질문 응답을 가능하게 한다는 것을 입증했습니다. 이 접근 방식은 오류를 디버깅할 때 더욱 집중할 수 있게 해주며, 오류 원인을 쉽게 식별할 수 있도록 돕습니다.



### High-Precision Fabric Defect Detection via Adaptive Shape Convolutions and Large Kernel Spatial Modeling (https://arxiv.org/abs/2501.14190)
Comments:
          8 pages, 9 figures

- **What's New**: 본 연구에서는 섬유 결함 탐지를 위한 혁신적인 모델인 Fab-ASLKS를 제안합니다. Fab-ASLKS는 YOLOv8s 아키텍처를 기반으로 하여 두 가지 주요 모듈인 Adaptive Shape Convolution Module (ASCM)과 Large Kernel Shift Convolution Module (LKSCM)을 통합하여 복합 섬유 결함의 정확한 탐지를 가능하게 합니다. 이러한 모듈은 특징 추출과 정보 통합을 최적화하여 실시간 탐지를 위한 높은 정확도와 효율성을 제공합니다.

- **Technical Details**: Fab-ASLKS는 Neck에 Adaptive Shape Convolution Module (ASCM)을 통합하여 표준 C2f 구조의 기능을 확장합니다. 이는 동적으로 조정 가능한 컨볼루션 커널을 통해 공간 변환을 캡처할 수 있는 능력을 향상시키며, Backbone에 통합된 Large Kernel Shift Convolution Module (LKSCM)은 대형 커널 효과를 모방하여 복잡한 결함 정보를 효과적으로 추출합니다. 이로 인해 모델은 5% 향상된 mAP@50 성능을 달성하며, 효율적인 실시간 탐지 요구에 부응합니다.

- **Performance Highlights**: Tianchi 섬유 결함 탐지 데이터셋에서의 실험 결과, Fab-ASLKS는 기존 모델 대비 5% 향상된 mAP@50을 기록하여 뛰어난 정확성과 효율성을 입증하였습니다. 본 연구는 기존의 전통적인 방법에 비해 더 정교하고 복잡한 결함 탐지에서 성능을 발휘합니다. 이를 통해 실시간 산업 응용에 적합한 자동화된 탐지 시스템의 필요성이 충족됩니다.



### Post-hoc Spurious Correlation Neutralization with Single-Weight Fictitious Class Unlearning (https://arxiv.org/abs/2501.14182)
- **What's New**: 이번 연구에서는 인공신경망(Artificial Neural Network, ANN)의 스퍼리어스 특징(spurious features)이 목표 레이블(target labels)과의 잘못된 상관관계로 모델의 예측 정확도를 떨어뜨리는 문제를 다룹니다. 기존의 방식은 주로 훈련 이전(ante-hoc) 방법으로 모델의 재훈련이나 강력한 훈련을 필요로 하는 대신, 후속 수정(post-hoc) 접근법을 통해 이러한 특징의 영향을 제거하려고 합니다. 이를 통해 최소한의 성능 저하로도 모델의 예측 정확도를 개선하려는 새로운 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 고수준 특징(high-level features)을 원래 클래스 내의 허구적 서브 클래스(fictitious sub-classes)로 개념화하여, 특정 서브 클래스에 대한 모델의 주의를 차단함으로써 예측 과정을 조정합니다. 단일 가중치 수정(single-weight modification) 기법을 활용하여 목표 클래스에 대한 예측 의존도를 줄이는 방식으로 설계되었으며, 이는 모델의 다른 클래스 성능에 거의 위협을 주지 않습니다. 이러한 기술적 기초는 클래스 활성화(class activations)와 모델 그래디언트(model gradients) 분석을 통해 지원됩니다.

- **Performance Highlights**: 광범위한 실험을 통해 제안된 방법은 기존의 최첨단 방법들에 비교하여 매우 경쟁력 있는 성능을 보여줍니다. 오직 단일 가중치를 수정하는 방식으로 후속 수정(post-hoc) 가능성을 제공하며, 유사한 방식의 기존 기법들보다 더욱 효과적으로 스퍼리어스 상관관계를 완화할 수 있습니다. 특히, 그룹 라벨이 필요 없는 점이 우리의 접근법의 큰 장점으로 작용합니다.



### Dreamweaver: Learning Compositional World Representations from Pixels (https://arxiv.org/abs/2501.14174)
- **What's New**: 이번 연구에서는 Dreamweaver라는 신경망 아키텍처를 제안하여 원시 비디오에서 계층적이고 조합적인 표현을 발견하고 조합된 미래 시뮬레이션을 생성합니다. 이는 전통적으로 언어와 같은 보조 데이터 없이 비디오를 모델링하는 데 어려움이 있었던 AI 시스템의 한계를 극복하는 데 기여합니다. 특히, Recurrent Block-Slot Unit (RBSU)를 활용하여 비디오를 구성 요소 객체 및 속성으로 분해하고, 동적 개념을 효과적으로 캡처하기 위한 다중 미래 프레임 예측 목표를 사용합니다.

- **Technical Details**: Dreamweaver 모델은 T 개의 과거 이미지를 인코딩하는 RBSU를 포함합니다. 이 RBSU는 독립적으로 업데이트되는 슬롯 상태 집합으로 상태를 표현하며, 이는 나중에 혼합된 몬올리식 슬롯 상태를 독립적인 블록 조합으로 매핑하는 블록- 슬롯 병목을 거칩니다. 각 단계에서 개념 프로토타입 라이브러리에 대한 주의를 통해 블록 벡터가 값을 취득하도록 하여 동적 개념 추상화를 생성하는 데 중요한 예측 재구성 목표를 설정합니다.

- **Performance Highlights**: 실험 결과, Dreamweaver는 여러 데이터 세트를 통한 DCI 프레임워크에서 최신 오브젝트 중심 방법보다 우수한 성능을 보여주었습니다. 또한, RBSU의 모듈화된 개념 표현이 조합적 상상력을 가능하게 하여 서로 다른 객체에서 속성을 재조합하여 새로운 비디오를 생성하는 능력을 입증했습니다. 이 연구는 언어에 의존하지 않고 독창적인 비디오를 생성하는 길로 나아가는 중요한 첫걸음을 제시합니다.



### Enhancing Multimodal Entity Linking with Jaccard Distance-based Conditional Contrastive Learning and Contextual Visual Augmentation (https://arxiv.org/abs/2501.14166)
- **What's New**: 이번 연구에서는 Multimodal Entity Linking (MEL)에 대한 기존의 접근 방식과 달리, JD-CCL(Jaccard Distance-based Conditional Contrastive Learning)이라는 새로운 기법을 제안합니다. JD-CCL은 메타 정보를 활용하여 네거티브 샘플을 더욱 정교하게 선택함으로써, 모델의 엔티티 매칭 능력을 강화합니다. 또한, CVaCPT(Contextual Visual-aid Controllable Patch Transform) 모듈을 도입하여 다양한 시각적 표현을 개선하고, 보다 복잡한 특징을 고려할 수 있도록 합니다.

- **Technical Details**: JD-CCL은 메타 속성을 활용하여 유사한 속성을 가진 네거티브 샘플을 선택합니다. 이는 모델이 단순한 속성이 아닌 보다 복잡한 속성을 기반으로 올바른 엔티티를 연결하도록 요구합니다. CVaCPT 모듈은 입력 이미지의 중요한 특징을 개별 화하여, 각각의 엔티티에 맞춘 다양한 시각적 표현을 생성합니다.

- **Performance Highlights**: 실험 결과, WikiDiverse, RichpediaMEL 및 WikiMEL이라는 세 가지 벤치마크 데이터셋에서 제안한 방법이 이전의 최신 기술들에 비해 뛰어난 성능을 보였습니다. 특히, JD-CCL과 CVaCPT는 멀티모달 데이터셋에서 엔티티에 대한 정확한 연결을 강화하는 데 기여했습니다.



### Advancing MRI Reconstruction: A Systematic Review of Deep Learning and Compressed Sensing Integration (https://arxiv.org/abs/2501.14158)
- **What's New**: 이 논문에서는 자기공명영상(MRI) 재구성에서의 딥러닝(DL) 기술의 발전을 다룬다. MRI는 비침습적(Non-invasive) 이미징 모달리티로, 해부학적 및 기능적 통찰을 제공하지만 길어진 데이터 수집 시간으로 인한 문제들이 존재한다. 본 리뷰는 이러한 문제를 해결하기 위한 DL 기반의 재구성 기법들을 종합적으로 분석하고, 그 잠재적 이점을 강조한다. 특히, DL이 병렬 이미징(Parallel Imaging) 및 압축 센싱(Compressed Sensing)과 결합될 때 더 빠르고 정확한 MRI 재구성이 가능함을 설명한다.

- **Technical Details**: MRI의 전통적인 재구성 방식은 데이터 수집의 비선형성으로 인해 상당한 시간과 비용이 소모된다. 이에 따라, 압축 센싱(CS) 기법은 스파스(sparse) 데이터에서 이미지를 재구성하고, DL 기법은 이런 방식을 보완하여 더욱 정교한 재구성을 가능하게 한다. 특히, DL 기법은 훈련 데이터를 통해 스파스 이미지를 전혀 다른 응축(maps)으로 변환하여, 스캔 시간을 단축하고 다양한 아티팩트를 줄이는 데 기여한다. 여러 구조의 DL 접근 방식이 있으며, 이 과정에서 병렬 이미징을 통한 SNR 개선과 그에 따른 가속화 메커니즘을 확인할 수 있다.

- **Performance Highlights**: DL 기반의 MRI 재구성 기법들은 빠른 재구성 시간을 자랑하며, 다양한 아티팩트와 노이즈를 효과적으로 완화하는 성능을 발휘한다. 객관적인 성과 지표 및 측정 기준을 토대로 한 여러 데이터셋에 대한 연구 결과와 관심이 증가하고 있으며, 이는 MRI 재구성 기술의 발전을 이끄는 주요 동력이 되고 있다. 또한, 미래 연구 방향을 논의하여 DL 기반 MRI 재구성의 중요성을 피력하며, 이를 통해 의료 이미징의 발전을 도모할 수 있는 방안을 제시한다.



### Effective Defect Detection Using Instance Segmentation for NDI (https://arxiv.org/abs/2501.14149)
Comments:
          6 pages, 2 figures, 2 tables. Published at AI2ASE 2025 workshop at AAAI2025. Accepted publication is available at this https URL

- **What's New**: 본 연구는 항공우주 제조업에서 사용되는 복합 패널의 초음파 스캔 이미지에서 결함을 식별하기 위해 인스턴스 세분화(instance segmentation) 기법을 적용하였다. Mask-RCNN과 YOLO 모델을 사용하여 결함을 탐지하고, 맞춤형 전처리 기법의 필요성을 줄이는 간단한 통계적 전처리 방법을 구현하였다. 이번 연구는 NDI(비파괴 검사) 프로세스에서 인스턴스 세분화 사용의 가능성과 효율성을 보여주며 데이터 전처리 시간과 검사 시간을 크게 단축시켰다.

- **Technical Details**: 본 연구에서는 초음파 검사가 사용되는 복합재료의 결함을 찾아내기 위해 Mask-RCNN(Detectron 2)과 YOLO 11을 기반으로 한 두 가지 모델을 활용하였다. 인스턴스 세분화 기법은 객체 경계 박스를 감지하고 이미지의 각 픽셀을 분류하여 결함을 탐지한다. 또한, 자동화된 NDI 프로세스를 지원하기 위해 초음파 스캔 이미지에 대한 간단한 전처리 방법을 사용하여 데이터의 복잡성을 완화하였다.

- **Performance Highlights**: 모델의 성능 비교 결과, 인스턴스 세분화를 통한 결함 탐지 방식이 전처리 시간과 검사 시간을 현저히 줄일 수 있음을 보여주었다. 이 연구를 통해 초음파 스캔 이미지에서의 결함 탐지에서 인스턴스 세분화 기법의 유용성이 입증되었으며, NDI 프로세스의 효율성을 높일 수 있는 가능성을 제시하였다. 연구 결과는 실제 항공우주 부품의 안전성을 확보하는 데 기여할 것으로 기대된다.



### SelfPrompt: Confidence-Aware Semi-Supervised Tuning for Robust Vision-Language Model Adaptation (https://arxiv.org/abs/2501.14148)
- **What's New**: SelfPrompt는 Semi-supervised learning(반지도학습) 환경에서 Vision-Language Models(VLMs)를 조정하기 위한 새로운 접근법을 제안합니다. 기존 방법들은 miscalibration(잘못된 조정)으로 인한 부정확한 pseudo-labels(가짜 라벨)의 누적 문제를 가지고 있습니다. SelfPrompt는 클러스터 기반의 pseudo-labelling과 자신감에 따른 반지도학습 모듈을 도입하여 이러한 문제를 해결하고, 라벨이 적은 상황에서도 학습을 최적화합니다.

- **Technical Details**: SelfPrompt는 두 가지 주요 구성 요소로 이루어져 있습니다. 첫 번째는 cluster-guided pseudo-labelling으로, 라벨링된 샘플을 중심으로 모든 샘플을 클러스터링한 후 각 샘플에 해당 클래스 라벨을 부여하여 pseudo-label 세트를 생성합니다. 두 번째는 confidence-aware semi-supervised learning으로, 높은 신뢰도를 가진 pseudo-label을 활용하여 학습하고, 낮은 신뢰도 샘플에 대해서는 약한 감독 방식으로 학습합니다.

- **Performance Highlights**: SelfPrompt는 13개의 데이터셋에 대한 종합 평가 결과, 기존 방법보다 평균 6.23%의 성능 향상을 달성했습니다. 또한, 한 샘플당 라벨을 줄였음에도 불구하고 평균 11.78%의 성능 개선을 보여, 강력한 일반화 능력을 입증했습니다. 게다가, SelfPrompt는 pseudo-labeling의 정확도를 높이며, 데이터의 배경 분포를 보다 대표적으로 반영할 수 있도록 개선된 수업 전략을 제시합니다.



### StreamingRAG: Real-time Contextual Retrieval and Generation Framework (https://arxiv.org/abs/2501.14101)
Comments:
          Accepted and Presented at AI4Sys, HPDC 2024

- **What's New**: 이 논문에서는 StreamingRAG라는 새로운 Retrieval-Augmented Generation (RAG) 프레임워크를 소개합니다. 이 프레임워크는 스트리밍 데이터의 실시간 분석을 위해 설계되었으며, 다중 모달 데이터를 제공하는 여러 도메인에서 실시간 통찰력을 추출하는 도전을 해결하고자 합니다. StreamingRAG는 장면-객체-엔티티 관계를 포착하는 진화하는 지식 그래프를 구축하고, MM-LLMs(Multi-Modal Large Language Models)를 활용해 컨텍스트에 민감한 장면 표현을 달성합니다.

- **Technical Details**: StreamingRAG는 사용자 제약, 진화하는 사건 및 현재 상황에 따라 특정 에터(actor)와 관계에 대한 정보를 우선시하는 동적 우선순위 기반 접근 방식을 사용합니다. 이 시스템은 지속적인 쿼리와 대화형 쿼리를 처리하며, 각 프레임의 내용을 분석하고 복잡성을 고려하여 최적의 입력 데이터 비율을 선택하도록 설계된 스케줄러를 통합합니다. 이로 인해, 시스템은 자원 소모는 적으면서도 실시간 데이터를 효과적으로 추출하고 관리할 수 있습니다.

- **Performance Highlights**: StreamingRAG는 기존 RAG 시스템과 비교해 5-6배 더 빠른 처리량을 자랑하며, 실시간 분석을 보장합니다. 또한, 시간적 지식 그래프를 활용하여 진화하는 시나리오에서 컨텍스트의 정확성을 유지하고, 경량 모델을 사용함으로써 자원 사용량을 2-3배 줄이는 데 성공했습니다. 이 모든 요소는 스마트 시티와 같은 응용 프로그램에서 실시간으로 데이터 흐름을 모니터링하는 데 크게 기여합니다.



### Expanding on the BRIAR Dataset: A Comprehensive Whole Body Biometric Recognition Resource at Extreme Distances and Real-World Scenarios (Collections 1-4) (https://arxiv.org/abs/2501.14070)
Comments:
          10 pages, 11 figures, 2 tables, submitted to CVPR

- **What's New**: 최근 생체 인식 알고리즘과 시스템의 발전은 많은 진전을 이루어냈지만, 극단적인 거리와 고도에서의 비전통적인 환경에서의 적용은 여전히 도전 과제입니다. 이 논문은 이러한 운영상의 도전에 대응하는 대규모 데이터세트의 확장을 요약하며, 그 구성과 데이터 수집, 관리 및 주석 달기 방법론을 설명합니다.

- **Technical Details**: BRIAR 프로그램은 어려운 조건에서 생체 인식 기술을 발전시키기 위해 고안된 미국 정부 지원의 프로그램으로, 기후와 환경이 다양한 3지역에서 1,760명의 피험자를 대상으로 475,000장의 이미지와 3,450시간의 영상을 포함하는 데이터세트를 구축했습니다. 이 데이터세트는 외부 및 실내 환경에서 다양한 활동을 수행하는 참가자들을 포함하며, 고해상도 이미지와 영상 데이터를 제공합니다.

- **Performance Highlights**: BRIAR 데이터세트는 생체 인식 및 열기 완화 관련 100편 이상의 논문에서 중요한 기준으로 사용되었으며, 연구자들이 해당 데이터에 접근할 수 있는 경로를 제공합니다. 이 연구는 향후 데이터 추가 및 품질 향상을 위한 지속적인 노력을 강조하고 있으며, 인종, 성별 등의 다양성을 고려한 데이터 확장을 통해 인식 모델의 공정성과 견고성을 보장하는 데 중점을 두고 있습니다.



### Prior Knowledge Injection into Deep Learning Models Predicting Gene Expression from Whole Slide Images (https://arxiv.org/abs/2501.14056)
- **What's New**: 이번 논문은 전체 슬라이드 이미지(Whole Slide Images, WSI)에서 형태학적 특징을 사용하여 유전자 발현(gene expression) 예측의 정확도를 향상시키기 위한 새로운 접근 방식을 제안합니다. 특히, 유전자-유전자 상호작용에 대한 사전 지식을 통합하는 모델-비독립적(model-agnostic) 프레임워크를 도입하여 기존 방법의 강건성을 개선하고자 합니다. 연구 결과, 제안된 방법은 유방암 데이터셋에서 25,761개 유전자 중 983개의 유전자의 발현 예측 성능을 평균적으로 향상시켰습니다.

- **Technical Details**: 제안된 프레임워크는 사전 지식을 유전자 임베딩(gene embeddings) 형식으로 변환하여 딥러닝 모델에 주입하는 방식으로 작동합니다. 이를 통해 유전자들 간의 관계를 보다 현실적으로 반영한 예측을 가능하게 하여 예측의 강건성을 증가시킵니다. 여러 모델 아키텍처를 적용하여 개발 데이터셋(TCGA-BRCA)과 독립 데이터셋(CPTAC-BRCA)에서 예측 정확도 향상을 평가하였습니다.

- **Performance Highlights**: 이번 연구에서, 제안된 방법은 18개의 실험에서 평균적으로 983개의 유전자의 예측 성능 향상을 달성했습니다. 또한, 14개의 실험은 독립적인 데이터셋에서도 일반화된 성능 향상을 보였습니다. 이러한 결과는 사전 지식을 주입함으로써 WSI에서의 유전자 발현 예측 성능을 증가시킬 수 있는 높은 잠재력을 보여줍니다.



### Revisiting CLIP: Efficient Alignment of 3D MRI and Tabular Data using Domain-Specific Foundation Models (https://arxiv.org/abs/2501.14051)
Comments:
          10 pages, 2 figures. To be published in ISBI 2025

- **What's New**: 이 논문에서는 CLIP 훈련을 3D 뇌 MRI에 처음으로 응용하고, 도메인 специф적인 3D 기반 모델 이미지 인코더를 훈련하여 모드 간 정렬이 소규모 데이터셋에서도 가능함을 보여줍니다. 또한, 3D에서의 훈련 시 배치 간의 임베딩을 누적하여 대비 손실을 안정화하는 방법을 제안했습니다. 이를 통해 3D MRI와 표 형식 데이터 간의 의미 있는 정렬이 가능함을 입증하였습니다.

- **Technical Details**: 주요 기술적 요소로는 제한된 크기의 데이터셋을 다룰 때 각 모드의 의미 있는 표현을 동시에 학습할 수 없다는 관찰이 있습니다. 이 문제를 해결하기 위해 우선 각 모드의 구체적인 인코더를 학습하고 이후 이를 일관된 임베딩 공간에 정렬하는 접근법을 사용합니다. 3D 뇌 MRI를 위한 인코더는 공공 MRI 데이터셋과 AMAES 프레임워크를 이용하여 대규모 프리트레이닝을 통해 학습되었습니다.

- **Performance Highlights**: 본 연구에서는 제안된 방법론이 제로샷 분류(zero-shot classification)와 이미지 검색(image-retrieval) 과제에서 평가되며, 특히 제로샷 분류에서는 3D MRI와 표 형식 데이터 간의 의미 있는 정렬을 보여주는 결과를 얻었습니다. 그러나 제로샷 이미지 검색은 여전히 도전적임을 발견하였습니다.



### LLM-guided Instance-level Image Manipulation with Diffusion U-Net Cross-Attention Maps (https://arxiv.org/abs/2501.14046)
Comments:
          Presented at BMVC 2024

- **What's New**: 이 논문은 텍스트에서 이미지로의 변환을 위한 새로운 파이프라인을 제안합니다. 기존 방법들은 세밀한 조정이 필요하거나 보조 정보에 의존했지만, 이 방법은 Large Language Models (LLMs)과 open-vocabulary detectors를 활용해 인스턴스 수준에서 이미지를 조작할 수 있습니다. 이 과정에서는 크로스 어텐션 맵(Cross-Attention Maps)과 확산 U-Net의 중간 활성화(Intermediate Activations)를 사용해 더욱 정밀한 이미지 수정을 가능하게 합니다.

- **Technical Details**: 제안된 방법은 LLM을 이용해 프롬프트에서 언급된 물체를 인식하고, 생성된 이미지 내에서 이를 직접적으로 조작합니다. 특히, 프롬프트와 이미지의 일치성을 높이는 동시에 물체의 위치 조정이 용이합니다. 이 접근법은 별도의 데이터 조정이나 보조 정보 없이도 원본 이미지의 세부 사항을 보존하며 조작할 수 있는 특징을 가지고 있습니다.

- **Performance Highlights**: 이 연구에서 제안된 방법은 기존의 이미지 편집 기술과 비교하여 더 나은 유연성과 정밀성을 제공합니다. 특히, 인스턴스 수준의 조작을 가능하게 함으로써 사용자가 보다 직관적으로 원하는 이미지를 생성할 수 있도록 합니다. 연구진은 코드도 공개하여 다른 연구자들이 이 기술을 쉽게 활용할 수 있도록 지원하고 있습니다.



### Implicit Neural Surface Deformation with Explicit Velocity Fields (https://arxiv.org/abs/2501.14038)
Comments:
          ICLR 2025, 10 pages

- **What's New**: 이 연구에서는 첫 번째로 비지도 방식(unsupervised method)을 사용하여 시간이 변하는 신경 임플리시트 표면(neural implicit surfaces)과 점 구름(point clouds) 쌍 간의 변형을 동시에 예측하는 방법을 제안합니다. 본 방법은 시간에 따라 변하는 임플리시트 필드를 직접 변형시키기 위해 수정된 레벨셋(level-set) 방정식을 사용하여 점의 이동을 모델링합니다. 이를 통해 물리적으로 그럴듯한 중간 형상을 복구할 수 있으며, 고정된 중간 형상에 대한 감독 없이 강직(rigid) 및 비강직(non-rigid) 변형을 처리할 수 있습니다.

- **Technical Details**: 본 방법은 점 구름 간의 표면 점 변형을 예측하기 위해 매끄럽고 발산 없는(divergence-free) 속도 필드를 기반으로 훈련된 속도 네트워크를 사용하여 자연스러운 변형을 보장합니다. 레벨셋 방정식에 따라 임플리시트 필드를 직접 변형시키고, Eikonal 제약조건을 결합하여 서명 거리 필드(signed distance field)가 손상되지 않도록 하는 수정된 레벨셋 방정식을 제안합니다. 이 과정에서 메쉬 렌더링을 요구하지 않는 end-to-end의 완전 미분 가능 훈련 프로세스를 제공합니다.

- **Performance Highlights**: 실험 결과, 본 방법은 기존의 방법들보다 뛰어난 성능을 보이며, 품질 및 효율성 양 측면에서 우수한 성과를 보여줍니다. 다양한 데이터셋을 통해 물리적으로 그럴듯한 고품질 보간(interpolation)을 이루어 내었음을 확인했습니다. 또한, 비정상적(topology)가 변화하는 형상 및 부분적(shapes with partiality) 형상에서도 우수한 변형 능력을 발휘합니다.



### INDIGO+: A Unified INN-Guided Probabilistic Diffusion Algorithm for Blind and Non-Blind Image Restoration (https://arxiv.org/abs/2501.14014)
Comments:
          Accepted by IEEE Journal of Selected Topics in Signal Processing (JSTSP)

- **What's New**: 이번 논문에서는 간헐적 이미지 복원(Non-blind Image Restoration, IR)과 일반화되지 않은 이미지 복원(Blind Image Restoration)을 위한 새로운 INN(Invertible Neural Network) 기반 확률적 확산 모델(Probabilistic Diffusion Model)을 제안합니다. 이 모델은 재구성 프로세스에서 INN의 완벽한 재구성 속성과 사전 훈련된 확산 모델의 강력한 생성 능력을 결합하여 더 높은 유연성과 성능을 제공합니다. 이러한 접근법은 복원 성능을 향상시키고 실제 복원 문제를 해결하는 데 기여할 수 있습니다.

- **Technical Details**: 제안하는 방법은 INN의 전방 프로세스를 훈련하여 임의의 열화(degradation) 프로세스를 시뮬레이션하고, 이를 통해 얻은 중간 이미지를 역 확산 샘플링 프로세스를 안내하는 데 사용합니다. INN의 고유한 데이터 일관성 단계를 통해 복원 결과가 입력 데이터와 일치하도록 강제함으로써, 세부 사항을 유지하면서 동작합니다. 이는 특히 복잡한 열화 과정을 처리할 수 있게 해 주며, 기존의 방법들이 직면했던 문제를 해결합니다.

- **Performance Highlights**: 실험 결과, 제안된 INDIGO 및 BlindINDIGO 알고리즘은 기존의 최첨단 방법들과 비교하여 정량적 및 시각적으로 경쟁력 있는 성과를 달성했습니다. 두 모델 모두 합성 및 실제 저품질 이미지에 대해 우수한 성능을 보이며, 실험을 통해 우수한 세부 복원 능력을 확인할 수 있었습니다. 이러한 결과는 INN과 확산 모델 간의 협력이 실제 문제 해결에 유리한 점을 보여줍니다.



### Device-aware Optical Adversarial Attack for a Portable Projector-camera System (https://arxiv.org/abs/2501.14005)
- **What's New**: 이 논문에서는 기존의 아드바이셔리 라이트 공격 방식의 한계를 극복하여 실제 얼굴 인식 시스템에서의 효율성을 높이는 방법을 제시합니다. 디지털 공격 알고리즘에 기기 인식(adaptation) 요소를 포함시키고 해상도와 색상 조정 같은 요소를 통합하여 디지털에서 물리적 도메인으로의 전환 중 손실을 최소화합니다. 실험 결과, 제안한 알고리즘이 실제 및 스푸프 공격에 대해 탁월한 성능을 발휘함을 보여줍니다.

- **Technical Details**: 본 연구는 모바일 프로젝터-카메라 장비를 사용하여 얼굴 인식 시스템을 공격하기 위한 기기 인식(adaptation) 알고리즘을 제안합니다. 이를 위해 프로젝터와 카메라 간의 색상 및 해상도 불일치 문제를 해결하는 방법을 도입하였고, 이 과정에서 모스웨어(noise)와 같은 요소를 다루기 위해 회색조 모드로 공격 최적화를 비롯한 기술적 접근이 포함됩니다. 이러한 기법은 모든 디지털 공격 알고리즘에 통합될 수 있어, 기존 방법론의 단점을 보완합니다.

- **Performance Highlights**: 실험을 통해 제안된 알고리즘이 다양한 재료로 만들어진 실제 및 스푸프 공격에 대해 높은 물리적 코사인 유사도 점수를 지속적으로 기록함을 확인하였습니다. 평균적으로 디지털 공격에서 물리적 공격으로 전환 시 점수의 감소는 14%에 불과하며, 이는 공격 성공률의 높은 수치를 나타냅니다. 본 연구는 얼굴 인식 시스템에 대한 아드바이셔리 공격의 실용성을 높이는 데 기여할 것으로 기대됩니다.



### ME-CPT: Multi-Task Enhanced Cross-Temporal Point Transformer for Urban 3D Change Detection (https://arxiv.org/abs/2501.14004)
- **What's New**: 이 논문에서는 도시 지역의 3D 변화를 감지하기 위한 새로운 Multi-task Enhanced Cross-temporal Point Transformer (ME-CPT) 네트워크를 제안합니다. 기존의 3D 변화 탐지 방법들이 가지고 있던 문제점들과 더불어, Multi-temporal ALS (Airborne Laser Scanning) 포인트 클라우드를 이용하여 도시 계획 및 비상 관리에서의 필요성을 강조합니다. 또한, 필요한 다양한 데이터셋의 부족이라는 문제를 해결하기 위해 22.5km² 크기의 3D 의미 변화 탐지 데이터셋을 출시했습니다.

- **Technical Details**: ME-CPT는 서로 다른 시간대의 포인트 클라우드 간 공간적 대응관계를 설정하여 변화 특징을 효과적으로 추출합니다. 이 네트워크는 다중 작업 훈련 전략을 통합하여 의미적 특징의 구별력을 향상시키며, 클래스 불균형 문제를 완화합니다. 특히, 크로스-템포럴(point cloud을 이용한 시간 간격) 주의 메커니즘을 활용하여 다중 시간대 포인트 간의 정보 교환을 촉진합니다.

- **Performance Highlights**: 제안된 ME-CPT는 여러 데이터셋에서 기존의 최첨단 방법들과 비교하여 우수한 성능을 보였습니다. 실험 결과, 이 네트워크가 복잡한 도시 환경에서의 3D 변화 탐지 알고리즘 성능을 제고함을 확인할 수 있었습니다. 연구에서는 다양한 시나리오에서의 적용 가능성을 논의하며, 향후 연구 방향에 있어서의 기초 자료로 활용될 수 있습니다.



### Enhancing kelp forest detection in remote sensing images using crowdsourced labels with Mixed Vision Transformers and ConvNeXt segmentation models (https://arxiv.org/abs/2501.14001)
- **What's New**: 이번 연구는 군중 소싱(crowdsourced) 라벨과 첨단 인공지능 모델을 통합하여 Landsat 이미지를 활용한 해조류 캐노피(kelp canopy) 탐지 파이프라인을 빠르고 정확하게 개발하는 것을 목적으로 하고 있습니다. 이 방법은 기계 학습 대회에서 3위를 기록하였으며, 지역 검증 및 공개/비공식 리더보드에서 일관되게 좋은 성과를 보였습니다. Mixed Vision Transformers (MIT)와 ConvNeXt 모델의 조합이 효과적임을 강조하고 있습니다.

- **Technical Details**: 해조류 탐지를 위한 모델 학습은 다양한 이미지 크기로 진행되어 앙상블(ensemble) 결과의 정확도를 크게 향상시켰습니다. 특히, U-Net이 최고의 세분화(segmentation) 아키텍처로 선정되었고, UpperNet 또한 최종 앙상블에 기여하였습니다. Landsat의 ShortWave InfraRed (SWIR1) 및 Near-InfraRed (NIR)와 같은 주요 밴드가 중요한 역할을 했으며, 고도 데이터를 사용하여 거짓 긍정(false positives)을 제거하는 후처리 작업이 이루어졌습니다.

- **Performance Highlights**: 이 방법론은 높은 탐지율을 기록하였으며, 해조류 캐노피를 포함한 픽셀의 약 75%를 정확히 식별하면서 거짓 긍정은 낮은 수준으로 유지하였습니다. Landsat 위성의 중해상도에도 불구하고, 광범위한 역사적 데이터는 해조류 숲 연구에 효과적임을 입증하고 있습니다. 또한, 기계 학습 모델과 군중 소싱 데이터를 결합한 방법의 환경 모니터링에서의 가능성을 강조하고 있습니다.



### Integrating Persian Lip Reading in Surena-V Humanoid Robot for Human-Robot Interaction (https://arxiv.org/abs/2501.13996)
- **What's New**: 이번 논문에서는 페르시아어 Lip reading(립 리딩) 데이터셋을 생성하여 Surena-V 휴머노이드 로봇에 통합함으로써 로봇의 음성 인식 능력을 향상시키고 있습니다. 이 연구는 간접적인 얼굴 랜드마크 추적 방법과 직접적인 CNN(Convolutional Neural Networks) 및 LSTM(Long Short-Term Memory) 네트워크를 활용하는 두 가지 보완적인 접근 방식을 탐색합니다. 연구 결과, 가장 성능이 좋은 모델인 LSTM이 89%의 정확도를 달성했으며, 이는 실제 인간-로봇 상호작용에서 효과적으로 구현되었습니다.

- **Technical Details**: 페르시아어로 사용되는 7개의 일반적인 단어를 인식하도록 훈련된 모델은 RGB-D 카메라를 사용하여 데이터를 수집했습니다. 영상은 20fps로 촬영되어 얼굴의 모든 특징을 강조하고, 영상의 해상도를 300×300으로 줄여 Lips(입술) 움직임을 보다 명확하게 추적할 수 있도록 했습니다. 두 가지 방법, 즉 간접 방법은 키 포인트를 추적하고, 직접 방법은 CNN 및 LSTM을 활용하여 데이터 추출 및 처리를 간소화하고 있습니다.

- **Performance Highlights**: 연구에서 최종적으로 적용된 LSTM 모델은 휴머노이드 로봇 Surena-V에 통합되어 실시간으로 89%의 높은 인식 정확성을 보였습니다. 이러한 성과는 언어 커뮤니케이션이 제한된 환경에서도 로봇이 시각적 신호를 통해 효과적으로 소통할 수 있도록 돕습니다. 다양한 데이터 전처리 및 모델링 기법을 통해 비즈니스와 헬스케어와 같은 복잡한 환경에서도 로봇이 원활하게 작동할 수 있는 가능성을 보여주고 있습니다.



### CSAOT: Cooperative Multi-Agent System for Active Object Tracking (https://arxiv.org/abs/2501.13994)
- **What's New**: 이번 연구에서는 복수의 에이전트가 단일 장치에서 협력하여 능동적인 물체 추적(Active Object Tracking, AOT) 작업을 수행하는 새로운 협업 시스템을 제안합니다. 기존의 AOT 솔루션과 달리, Collaborative System for Active Object Tracking (CSAOT)는 추가적인 장비 없이도 다수의 에이전트를 활용하여 비용을 절감하고 성능을 향상시킵니다. 이 시스템은 각 에이전트가 특정 역할을 맡아 협력함으로써 동적 환경에서의 학습 및 강인성을 개선하고 있습니다.

- **Technical Details**: CSAOT는 다중 에이전트 심층 강화 학습(Multi-Agent Deep Reinforcement Learning, MADRL)와 전문 집합(Mixture of Experts, MoE) 프레임워크를 이용하여, 각 에이전트가 특정 시나리오에 최적화된 정책을 학습하도록 설계되었습니다. MoE는 각 에이전트가 현재 상황에 기반하여 향상된 결정 능력을 발휘할 수 있도록 하는 게이팅 메커니즘을 포함하고 있습니다. 이 접근법은 의사 결정 속도를 높이고, 추가 하드웨어의 필요성을 줄이며, 통신 비용을 최소화합니다.

- **Performance Highlights**: CSAOT의 성능은 동적 및 정적 장애물이 있는 다양한 상호작용 맵에서 실험하여 검증되었습니다. 이 시스템은 빠른 움직임에 대한 강인성과 가려짐에 대한 저항력을 개선했으며, 카메라 동작 최적화를 통해 더 긴 추적 시간을 달성했습니다. 이러한 성능은 복잡한 환경 내에서 효과적인 물체 추적이 가능함을 보여줍니다.



### CGI: Identifying Conditional Generative Models with Example Images (https://arxiv.org/abs/2501.13991)
- **What's New**: 최근 생성 모델의 성과가 주목받고 있으며, 여러 모델 허브가 등장했습니다. 이러한 허브들은 데이터 검색에 있어 기본적인 텍스트 일치를 가정하지만, 많은 수의 모델과 다양한 추상화로 인해 사용자 경험이 제한됩니다. 따라서 사용자 요구에 맞는 모델을 효과적으로 찾을 수 있는 새로운 방법론인 Conditional Generative Model Identification (CGI)를 제안합니다.

- **Technical Details**: CGI는 사용자 요구사항과 모델의 기능을 정량적으로 매칭할 수 있는 혁신적인 방법론입니다. 이를 통해 사용자가 제공하는 예시 이미지를 기반으로 적합한 모델을 식별하고, 모델 기능을 보다 정확하게 설명하기 위한 Prompt-Based Model Identification (PMI) 접근방식을 제안합니다. 실험을 통해 PMI가 65개의 모델과 9100개의 식별 과제를 포함하는 벤치마크에서 효과적인 성능을 보인다는 것을 입증하였습니다.

- **Performance Highlights**: PMI 방법을 통해 제공된 4개의 예시 이미지를 사용하여 92%의 모델을 정확하게 식별하는 성과를 달성했습니다. 이는 기존의 모델 탐색 방식보다 우수한 성능을 나타냅니다. 방대한 데이터와 실험적 검증을 통해 PMI의 유효성과 효율성을 입증하였으며, 이는 향후 연구 방향에도 기여할 수 있을 것입니다.



### Attribute-based Visual Reprogramming for Image Classification with CLIP (https://arxiv.org/abs/2501.13982)
- **What's New**: 이 논문에서는 CLIP을 위한 속성 기반 Visual Reprogramming(AttrVR)을 제안합니다. 기존의 VR 접근 방법이 고정된 텍스트 템플릿을 활용할 때 발생하는 단점을 극복하기 위해, 다양한 속성(DesAttrs 및 DistAttrs)을 이용하여 동적인 패턴 최적화를 추구합니다. AttrVR은 각 이미지 샘플에 대해 반복적으로 속성을 수정하며 학습하는 방식으로, 더 나은 분류 성능을 보여줍니다.

- **Technical Details**: AttrVR은 Descriptive attributes(DesAttrs)와 Distinctive attributes(DistAttrs)를 활용하여 다양한 클래스를 설명합니다. DesAttrs는 공통 특성을, DistAttrs는 클래스 간의 차별화된 특성을 나타냅니다. 각 샘플에 대해 가장 근접한 k개의 속성을 반복적으로 조회하여 VR 패턴을 개선하며, 이는 고정 템플릿 기반 레이블보다 더 맥락에 적합한 이미지-속성 정렬을 가능하게 합니다.

- **Performance Highlights**: Empirical 측면에서, AttrVR은 ViT 기반 및 ResNet 기반 CLIP 솔루션 모두에서 12개의 다운스트림 작업에서 우수한 성능을 달성하며, 기존 VR 방법들보다 일관되게 더 나은 결과를 나타냅니다. AttrVR의 성능은 이론적 분석과 실험적 결과 모두에서 증명되며, 기존의 레이블 기반 방법에 비해 분류 작업에서 명확한 장점을 제공합니다.



### Enhanced PEC-YOLO for Detecting Improper Safety Gear Wearing Among Power Line Workers (https://arxiv.org/abs/2501.13981)
- **What's New**: 본 논문에서는 복잡한 전력선 환경에서 안전 장비의 잘못된 사용으로 인한 위험을 줄이기 위해 PEC-YOLO 객체 탐지 알고리즘을 제안합니다. 이 알고리즘은 PConv와 EMA 주의 메커니즘을 통합하여 특성 추출 효율성을 개선하고 모델 복잡성을 최소화합니다. 또한 SPPF 모듈에 CPCA 주의 메커니즘을 추가하여 중요한 정보를 강조하고 탐지 정확도를 향상시킵니다.

- **Technical Details**: PEC-YOLO는 저수준 및 고수준의 특성을 최적화하기 위해 BiFPN 목 구조를 도입하여 특성 표현을 향상시킵니다. 이 방법은 포인트 컨볼루션과 효율적인 특성 집합 기술을 활용하여 모델의 파라미터 수를 줄이고 계산 복잡성을 낮춥니다. CPCA 주의 메커니즘은 중요 정보를 목표로 하여 탐지 정확도를 높인다.

- **Performance Highlights**: 실험 결과, PEC-YOLO는 YOLOv8s에 비해 2.7% 향상된 탐지 정확도를 달성하고 모델 파라미터를 42.58% 감소시켰습니다. 또한 동일한 조건에서 PEC-YOLO는 다른 모델들보다 더 뛰어난 탐지 속도를 보여 주었으며, 이는 건설현장 안전 장비 탐지의 다각적인 정확도 요건을 충족합니다.



### 3DGS$^2$: Near Second-order Converging 3D Gaussian Splatting (https://arxiv.org/abs/2501.13975)
Comments:
          11 pages, submit on SIGGRAPH 2025

- **What's New**: 이 논문은 3D Gaussian Splatting (3DGS) 훈련을 위한 새로운 (near) second-order 수렴 훈련 알고리즘을 소개합니다. 기존의 표준 stochastic gradient descent (SGD) 방법에 비해 훈련 속도를 10배 이상 가속화하여 훈련 시간을 분에서 초로 줄이는 독창적인 접근 방식을 제시합니다. 우리 접근법은 개별 Gaussian 커널의 최적화 및 공간적 정보를 활용하는 두 가지 주요 관찰에 기반합니다.

- **Technical Details**: 3DGS 훈련은 각 Gaussian 커널의 속성이 이미지 공간 손실에 독립적으로 기여한다는 사실에서 영감을 받았습니다. 이를 통해 개별 커널 속성 수준에서 최적화를 분리하고 각 매개변수 그룹에 대해 작고 효율적인 Newton 시스템을 구성합니다. 각 GS 커널에서 병렬로 실행되는 지역 업데이트 덕분에 우리는 강력한 2차 수렴을 관찰하게 됩니다.

- **Performance Highlights**: 이 알고리즘은 3DGS 데이터셋에서 SGD 기반의 GPU 훈련보다 항상 10배 더 빠르며, 품질은 유지되거나 초과하는 결과를 보입니다. 따라서 3DGS 훈련에서 간섭을 최소화하고, 입력 이미지의 오버슈트를 효과적으로 완화하는 새로운 샘플링 전략도 도입하였습니다.



### A Spatio-temporal Graph Network Allowing Incomplete Trajectory Input for Pedestrian Trajectory Prediction (https://arxiv.org/abs/2501.13973)
- **What's New**: 이번 연구에서는 과거의 불완전한 경로를 고려하여 보행자의 미래 경로를 예측할 수 있는 새로운 spatio-temporal graph network인 STGN-IT를 제안합니다. 기존의 알고리즘들은 불완전한 정보를 처리하지 못하고, 이로 인해 로봇 내비게이션에 어려움을 겪어왔습니다. STGN-IT는 정적 장애물 정보를 통합하여 예측 정확도를 높이는데 기여합니다.

- **Technical Details**: STGN-IT 알고리즘은 spatio-temporal graph를 사용하여 보행자와 장애물 정보를 표현하고, 특별한 인코딩 방법을 통해 불완전한 과거 경로를 처리합니다. 이 알고리즘은 기존의 semantic map 대신 자동으로 생성된 occupancy grid map을 활용하여 더욱 유연한 접근을 제공합니다. 데이터셋 STCrowd에서 STGN-IT의 성능을 비교 평가한 결과, 기존 최첨단 알고리즘을 능가하는 성과를 보였습니다.

- **Performance Highlights**: STGN-IT는 기존 알고리즘과 비교했을 때 ADE와 FDE에서 개선된 성과를 나타내며, 특히 egocentric view 데이터셋에서 불완전한 경로에 대한 저항성을 발휘합니다. 실험 결과, STGN-IT는 로봇 내비게이션 환경에서 보다 안전하고 효율적인 경로 예측을 가능하게 하며, 이는 로봇이 보행자와의 충돌 위험을 줄이는데 도움이 됩니다.



### GS-LiDAR: Generating Realistic LiDAR Point Clouds with Panoramic Gaussian Splatting (https://arxiv.org/abs/2501.13971)
- **What's New**: 본 연구는 GS-LiDAR라는 새로운 프레임워크를 제안하며, 이를 통해 현실감 있는 LiDAR 포인트 클라우드를 패노라믹 가우시안 스플래팅(panoramic Gaussian splatting)을 사용하여 생성합니다. 이 방법은 2D 가우시안 원형을 이용하여 정적 및 동적 요소를 정밀하게 재구성할 수 있습니다. 또한, 패노라믹 LiDAR 감독에 의해 가이드되는 새로운 렌더링 기법을 도입하여 LiDAR 데이터를 보다 효율적으로 시뮬레이션합니다.

- **Technical Details**: GS-LiDAR는 주기적 진동 특성을 가진 2D 가우시안 원형을 활용하여 주행 시나리오에서 다양한 객체를 균일하게 표현합니다. 새로운 패노라마 렌더링 프로세스를 통해 2D 가우시안 원형을 사용하여 패노라마 깊이 맵을 신속하고 효율적으로 렌더링하며, 정확한 기하학적 특성을 유지합니다. 이 시스템은 LiDAR 특정 속성(예: 뷰 의존 강도 및 레이 드롭 확률)을 포함하여 실제 라이다 데이터로부터 슈퍼비전(supervision)을 받습니다.

- **Performance Highlights**: GS-LiDAR는 KITTI-360 및 nuScenes 데이터셋에서 다양한 성능 비교 실험을 통해 이전 방법인 LiDAR4D에 비해 약 10.7%의 RMSE 개선을 달성했습니다. 동적 장면에서는 11.5%의 RMSE 감소를 보였으며, 이와 함께 훈련 속도와 렌더링 속도 또한 각각 1.67배, 31배 향상되었습니다. 이러한 결과는 GS-LiDAR가 LiDAR 뷰 합성에서 최고의 성능을 발휘함을 보여줍니다.



### InsTex: Indoor Scenes Stylized Texture Synthesis (https://arxiv.org/abs/2501.13969)
- **What's New**: 본 논문에서는 3D 실내 장면을 위한 고품질 및 스타일 일관성을 갖춘 텍스처를 생성하는 새로운 아키텍처인 InsTex를 제안합니다. InsTex는 깊이-이미지 확산(Depth-to-Image Diffusion) 프라이어를 사용하여 초기 다중 뷰 이미지를 생성하고, 이후 품질과 일관성을 높이기 위한 리파인먼트 단계를 포함합니다. 이 방법은 텍스트와 비주얼 프롬프트를 모두 지원하며, 기존의 콘텐츠 생성 방식보다 개선된 결과를 보여줍니다.

- **Technical Details**: 제안하는 InsTex는 두 단계를 거치는 파이프라인을 사용하여 3D 실내 장면의 개별 객체를 텍스처링합니다. 첫 번째 단계에서는 깊이 인식을 통해 다중 뷰 이미지를 생성하고, 두 번째 단계에서는 이 이미지를 통해 텍스처를 정제하여 일관성을 유지합니다. 또한, 동적 뷰 파티셔닝 전략을 채택하여 각 뷰포인트에 따라 다른 영역에서 생성 목표를 설정하고, 깊이 인식을 통해 텍스처를 보강합니다.

- **Performance Highlights**: InsTex는 다양한 프롬프트를 기반으로 고품질의 텍스처를 생성하며, 시각적 품질 및 정량적 지표에서 최신 기법들을 초월하는 성능을 보여줍니다. 제안된 방법이 여러 3D 텍스처링 응용 프로그램에 효과적으로 적용될 수 있음을 입증하며, 특히 리얼타임 AR/VR 환경에서도 유용할 것으로 기대됩니다.



### Triplet Synthesis For Enhancing Composed Image Retrieval via Counterfactual Image Generation (https://arxiv.org/abs/2501.13968)
Comments:
          4 pages, 4 figures

- **What's New**: 이 논문은 Composed Image Retrieval (CIR)에서의 훈련 데이터 생성을 개선하기 위한 새로운 triplet 합성 방법을 제안합니다. 이 방법은 counterfactual image generation을 활용하여 수작업 없이 다양한 training triplets를 자동으로 생성함으로써 CIR 모델의 성능을 높입니다. 이러한 접근법은 특히 데이터가 제한적인 상황에서 유용하게 적용될 수 있습니다.

- **Technical Details**: 제안된 방법은 두 단계를 포함합니다: counterfactual caption 생성과 그에 따른 이미지 생성입니다. 초기 reference 이미지에 수정 텍스트를 적용하여 target 이미지를 생성하며, 이 과정에서 Language-guided Counterfactual Image (LANCE) 모형이 사용됩니다. LANCE는 간접적인 변경 사항을 반영하여 이미지의 특정 속성을 변화시켜, 고유한 triplet을 합성할 수 있도록 도와줍니다.

- **Performance Highlights**: 제안된 방법은 고품질의 triplet 생성을 통해 CIR 모델을 개선할 수 있는 잠재력을 증명합니다. 제한된 데이터셋에서도 다양하고 질 높은 triplet 생성을 지원하며, 이러한 triplet들은 모델의 훈련에 있어 더 효과적으로 작용합니다. 결과적으로, 이 방법은 기존의 자동 triplet 합성에서 발생할 수 있는 문제들을 극복하고 실용적인 대규모 CIR 데이터셋 생성을 지원할 수 있습니다.



### FedDAG: Federated Domain Adversarial Generation Towards Generalizable Medical Image Analysis (https://arxiv.org/abs/2501.13967)
- **What's New**: 이 논문에서는 Federated Domain Adversarial Generation (FedDAG)이라는 새로운 프레임워크를 제안하여 연합 학습(federated learning) 환경에서 모델의 일반화 능력을 향상시키는 방법을 다룹니다. FedDAG는 기존의 연합 도메인 일반화(federated domain generalization) 접근이 갖고 있는 제한을 극복하고, 지역 도메인 특성과 전역 도메인 특성을 반영하는 새로운 도메인을 적대적으로 생성하여 모델의 일반화 성능을 개선하는 데 초점을 맞추고 있습니다. 특히, FedDAG는 현업에서의 의료 시나리오에 맞춰 세밀하게 설계되었습니다.

- **Technical Details**: FedDAG는 각 클라이언트에서 독립적으로 적대적 생성을 수행하면서 의미 일관성을 유지하는 2단계 구속을 적용하여 새로운 스타일의 이미지를 생성합니다. 이는 기존의 지역 특성과는 다른 특성을 지닌 이미지를 생성하는 데 중점을 두며, 이를 통해 의료 데이터의 중요한 의미를 극대화합니다. 또한, 모델의 일반화 기여를 평가하기 위해 모델 집합체를 계층적으로 집계하여 각 클라이언트의 기여를 균형 있게 조정하는 방식을 취합니다.

- **Performance Highlights**: 논문에서 언급된 대규모 실험은 FedDAG가 기존 방식보다 더 나은 일반화 능력을 갖추고 있음을 보여줍니다. 네 가지 의학적 기준 벤치마크에 대한 실험 결과는 FedDAG가 의료 데이터의 도메인 변화에 적절히 대응할 수 있는 능력을 입증하였습니다. 이러한 결과들은 전반적으로 FedDAG가 의학적 시나리오에서 모델의 성능을 더욱 향상시킬 수 있음을 나타냅니다.



### Advancing the Understanding and Evaluation of AR-Generated Scenes: When Vision-Language Models Shine and Stumb (https://arxiv.org/abs/2501.13964)
Comments:
          6 pages

- **What's New**: 이번 연구에서는 증강 현실(AR) 경험의 품질을 평가하기 위해 세 가지 최첨단 Vision-Language Models(VLMs)인 GPT, Gemini, Claude의 능력을 분석했습니다. DiverseAR이라는 새로운 데이터셋을 사용하여 VLM이 AR 장면을 인식하고 설명하는 능력을 평가했습니다. 연구 결과, VLM은 AR 장면을 인식하는 데에서 최대 93%의 True Positive Rate(TPR)을 달성했으나, 매우 통합된 콘텐츠에 대한 인식에서는 어려움을 겪었습니다.

- **Technical Details**: DiverseAR 데이터셋은 다양한 AR 시나리오를 수집한 298개의 이미지를 포함하고 있으며, 이 데이터셋은 VLMs의 AR 장면 이해 능력을 평가하기 위해 설계되었습니다. 연구에서는 다양한 난이도의 AR 장면을 정의하고, VLMs의 성능을 평가하기 위해 일반 이미지 캡셔닝 프롬프트와 테스크 인식 이미지 캡셔닝 프롬프트를 사용했습니다. 각 AR 샘플은 정확한 인식, 부분 인식, 놓친 인식, 잘못된 인식, 정확한 비-AR 인식 등 다섯 가지 범주로 나누어 분석되었습니다.

- **Performance Highlights**: VLM은 눈에 띄는 가상 객체는 잘 인식하지만, 물리적 법칙을 충족하는 콘텐츠에서는 어려움을 겪고 있습니다. 다양한 AR 장면을 분석한 결과, VLM의 인식 및 설명 성능의 한계가 명확히 드러났습니다. 이러한 연구는 AR 경험의 품질을 평가하는 데 있어 VLM의 가능성을 강조하며, 실제와 가상 콘텐츠 간의 효과적인 구분법을 제시하고 있습니다.



### Procedural Generation of 3D Maize Plant Architecture from LIDAR Data (https://arxiv.org/abs/2501.13963)
- **What's New**: 이 연구는 LiDAR 포인트 클라우드 데이터를 기반으로 옥수수 식물의 절차적 3D 모델을 생성하는 강력한 프레임워크를 소개합니다. 이 프레임워크는 Non-Uniform Rational B-Spline (NURBS) 표면을 사용하여 옥수수 잎을 모델링하며, Particle Swarm Optimization (PSO)를 초기 근사화에 활용합니다. 이후 미분 가능 프로그래밍 프레임워크를 통해 표면을 정교하게 다듬어 LiDAR 데이터에 맞춥니다.

- **Technical Details**: 최초 최적화 단계에서 PSO는 제어점을 최적화하여 NURBS 표면을 생성하고, LiDAR 데이터와 정렬하여 신뢰할 수 있는 시작점을 제공합니다. 두 번째 단계에서는 NURBS-Diff라는 미분 가능 프로그래밍 프레임워크를 사용하여 표면 기하학의 정확성을 향상시키고 복잡한 잎 세부 정보를 포착합니다. 이것은 다양한 유전자형에서 정확한 3D 재구성을 가능하게 하여 복잡한 특성 추출로 이어집니다.

- **Performance Highlights**: 연구 결과는 PSO가 강력한 초기 피팅을 확립하는 반면, 미분 가능 NURBS의 통합이 재구성된 표면의 전반적인 품질과 충실도를 크게 향상시킨다는 것을 보여줍니다. 이 계층적 최적화 전략은 다양한 유전자형의 옥수수 잎의 정확한 3D 재구성을 지원하며, 이후 복잡한 특성의 추출을 용이하게 합니다. 모든 코드는 오픈 소스 형태로 제공되어 이러한 표현형 분석 접근 방식을 민주화하는 데 기여합니다.



### A Fast, Scalable, and Robust Deep Learning-based Iterative Reconstruction Framework for Accelerated Industrial Cone-beam X-ray Computed Tomography (https://arxiv.org/abs/2501.13961)
- **What's New**: 본 논문에서는 대규모 산업용 원뿔형 X선 컴퓨터 단층촬영(cone-beam XCT) 데이터를 위한 새로운 딥 뉴럴 네트워크 기반의 반복 알고리즘을 제안합니다. 이 알고리즘은 아티팩트 제거 훈련된 CNN을 사전 모델로 통합하고 자동화된 정규화 파라미터 선택 기능을 포함하여, 극도로 밀집한 금속 부품에 대한 고품질 3D 재구성을 몇 번의 반복만으로도 수행할 수 있습니다. 또한, 다양한 스캐닝 조건에서 얻어진 분포 외(scan out-of-distribution) 스캔에 대한 일반화 가능성도 입증됩니다.

- **Technical Details**: 제안하는 알고리즘은 Half-Quadratic Splitting(HQS) 구성에서 영감을 받아 CNN-정규화된 물리 기반 역전파 알고리즘을 사용합니다. 프로세스에서 CNN과 공액 경량(CG) 알고리즘 간의 교차 횟수를 재구성 품질에 따라 조절하며, CNN은 저품질 희소 보기 FDK 재구성의 아티팩트를 줄이는 훈련을 받습니다. 또한, 자동으로 정규화 강도를 조정하여 CT 재구성 알고리즘에서 중요한 도전과제를 해결하는 정규화 파라미터 선택 전략이 포함됩니다.

- **Performance Highlights**: 제안된 알고리즘은 X선 소스 전압, 총 통합 시간 및 희소성 등 다양한 조건에서 얻은 XCT 스캔에 대해 기존 방법들과 비교하여 인상적인 성능을 보입니다. 정규화 선택 전략은 고정된 정규화 파라미터를 사용하는 것보다 더 나은 성능을 발휘하며, 반복 수가 적어 MBIR보다 계산 복잡도가 크게 줄어듭니다. 이 알고리즘은 원뿔형 XCT 응용에 적합하도록 설계되었지만, 다른 대규모 이미지 재구성 응용으로의 확장 가능성도 가지고 있습니다.



### DEFEND: A Large-scale 1M Dataset and Foundation Model for Tobacco Addiction Prevention (https://arxiv.org/abs/2501.13950)
Comments:
          11 pages, 5 figures, 5 tables

- **What's New**: 본 논문에서는 Tobacco-1M이라는 대규모 담배 제품 이미지 데이터셋을 소개하며, 75개의 제품 카테고리 전반에 걸쳐 100만 개의 이미지를 포함하고 있습니다. 또한 DEFEND라는 새로운 기초 모델을 제안하여 담배 제품을 이해하는 데 초점을 맞추었습니다. 이 연구는 담배 제품 모니터링을 위한 혁신적인 방법론을 제공하며, 기존 모델과 비교하여 뛰어난 성능을 입증하였습니다.

- **Technical Details**: Tobacco-1M 데이터셋은 효율적인 분류와 건강 영향 평가를 위한 상세한 기능 설명이 포함된 주석을 통하여 다양한 카테고리와 하위 카테고리를 정의합니다. DEFEND 모델은 Feature Enhancement Module을 사용하여 시각적 및 텍스트 기반 데이터 간의 복잡한 관계를 학습하며, Local-Global Visual Coherence 메커니즘을 통해 섬세한 특징 구분을 수행합니다. 이 모델의 실험 결과는 제품 분류에서 83.1%의 정확 종과 시각적 질문 응답 과제에서 73.8%의 성능을 보여줍니다.

- **Performance Highlights**: DEFEND 모델은 새로운 제품 카테고리에 대해 45.6%의 정확도로 제로샷 학습 능력을 보여주며, 이는 기존의 방법들보다 훨씬 높은 성능을 의미합니다. Tobacco-1M 데이터셋은 선형 회귀 모델 및 기존 담배 연구에서의 한계를 극복하고, AI 기반의 담배 모니터링 연구를 가속화하는 중요한 자료로 자리잡았습니다. 이 연구는 규제 기관과 공공 건강 연구자들에게 담배 제품 및 마케팅 전략을 모니터링할 수 있는 강력한 도구를 제공합니다.



### Mitigating GenAI-powered Evidence Pollution for Out-of-Context Multimodal Misinformation Detection (https://arxiv.org/abs/2501.14728)
Comments:
          12 pages, 11 figures

- **What's New**: 이번 연구는 대형 Generative Artificial Intelligence (GenAI) 모델의 오용이 온라인 정보 보안에 미치는 영향을 다룹니다. 특히, Out-of-Context (OOC) 멀티모달 허위 정보 탐지에서 GenAI로 오염된 증거(evidence)를 처리하는 방법을 제안합니다. 기존 연구들은 주로 스타일 재작성(stylistic rewriting)을 통해 언어적 단서를 숨기는 방식으로 GenAI 오염을 모사했지만, 실질적인 증거 수준의 오염을 간과했습니다.

- **Technical Details**: 연구에서는 오염된 증거가 기존 OOC 탐지기의 성능에 미치는 영향을 분석했고, 그 결과 성능이 9% 이상 저하되는 것을 발견했습니다. 이에 대응하기 위해 cross-modal evidence reranking 및 cross-modal claim-evidence reasoning라는 두 가지 전략을 제안했습니다. 이 방법들은 오염된 증거와 함께 존재하는 기존 탐지기의 강건성을 강화하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 두 개의 벤치마크 데이터셋에서 수행된 광범위한 실험을 통해 제안된 전략들이 오염된 증거에 맞서 기존의 OOC 탐지기를 효과적으로 향상시킨다는 점을 입증했습니다. 이로 인해 정보 탐색(application) 분야에서 GenAI의 부정적 영향을 줄이는 데 기여할 수 있습니다.



### Enhanced Confocal Laser Scanning Microscopy with Adaptive Physics Informed Deep Autoencoders (https://arxiv.org/abs/2501.14709)
- **What's New**: 이 논문에서는 Confocal Laser Scanning Microscopy (CLSM)의 한계를 극복하기 위한 물리 기반(physics-informed) 딥러닝 프레임워크를 제안합니다. 기존의 CLSM이 직면한 문제인 회절 한계 해상도(diffraction limited resolution), 잡음(noise), 낮은 레이저 파워 조건으로 인한 언더샘플링(undersampling)을 다룹니다. 이 프레임워크는 모델 아키텍처에 광학 시스템의 포인트 스프레드 함수(point spread function, PSF)와 여러 이미지 저하 메커니즘을 통합하였습니다.

- **Technical Details**: 모델은 합성곱(convolutional) 및 전치 합성곱(transposed convolutional) 레이어를 사용하여 심한 잡음이 있는 입력에서 고충실도(high fidelity) 이미지를 복원합니다. 또한, 압축 감지(compressed sensing) 기술을 기반으로 하여 데이터 수집 요구 사항을 크게 줄이면서도 이미지 해상도는 유지합니다. 다양한 구조의 시뮬레이션 CLSM 이미지에서 이 방법을 광범위하게 검증하였으며, 리차드슨-루시(Richardson-Lucy, RL) 및 기타 전통적인 디콘볼루션(deconvolution) 알고리즘과 비교하여 우수성을 입증하였습니다.

- **Performance Highlights**: AdaptivePhysicsAutoencoder는 다양한 CLSM 조건에서 안정적인 이미지 향상을 달성하였으며, 구조적 유사성 지수(Structural Similarity Index, SSIM) 및 피크 신호 대 잡음비(Peak Signal to Noise Ratio, PSNR)와 같은 평가 지표에서 긍정적인 결과를 보였습니다. 이를 통해 이미지 획득 속도를 높이고, 광 손상을 줄이며, 저조도(low light) 및 희소 샘플링(sparse sampling) 상황에서도 신뢰할 수 있는 성능을 제공합니다. 본 연구는 생체 세포 이미지 생성, 동적 생물학적 연구, 고처리량 재료 특성화(high throughput material characterization) 등의 응용 가능성을 보여줍니다.



### Stroke classification using Virtual Hybrid Edge Detection from in silico electrical impedance tomography data (https://arxiv.org/abs/2501.14704)
Comments:
          21 pages, 5 figures

- **What's New**: 이 논문에서는 전기 임피던스 단층 촬영(EIT)과 새로운 VHED(Virtual Hybrid Edge Detection) 기능을 결합하여 뇌졸중 분류 문제를 다루고 있습니다. 기존 연구들이 주로 원시 EIT 전압 데이터를 사용했던 반면, 본 연구는 보다 정교하고 수학적으로 현실적인 모델을 통해 VHED 기능을 입력으로 사용하는 방법을 검증하고 있습니다. 이 연구는 실제 환자를 대상으로 한 EIT 요법의 성공적인 적용 가능성을 보여줍니다. 또한, VHED 함수의 도입이 노이즈에 강점이 있음을 강조하여 뇌졸중 분류에서의 유효성을 확립하고자 하였습니다.

- **Technical Details**: EIT는 외부 전극에서 전기를 측정하여 내부 전도도 분포를 이미지화합니다. 본 연구는 가상 환자를 위해 물리적으로 상세한 2D 두개 모델을 사용하여 전도도 값은 통계적으로 현실적인 분포에서 추출하였습니다. VHED 함수는 완전 전극 모델(CEM)을 사용하여 생성된 시뮬레이션된 EIT 데이터로부터 처리됩니다. 이를 통해 EIT 전압 데이터와 VHED 함수의 사용을 비교하여, 보다 정교한 모델에서 뇌졸중 분류의 정확성을 높였습니다.

- **Performance Highlights**: 실험 결과, 물리적으로 상세하고 수학적으로 현실적인 모델에서 2D EIT 데이터를 기반으로 뇌졸중을 높은 정확도로 분류할 수 있음을 보였습니다. 또한, 소음이 있는 환경에서 VHED 함수는 원시 EIT 데이터에 비해 더욱 우수한 성능을 나타내었습니다. 이는 VHED 기능이 뇌졸중 분류 문제에서 중요한 입력 특성으로 작용할 수 있음을 의미합니다. 결국, VHED 함수의 활용이 EIT 기술을 통한 뇌졸중 진단의 향상에 기여할 수 있음을 제안합니다.



### Rethinking Foundation Models for Medical Image Classification through a Benchmark Study on MedMNIS (https://arxiv.org/abs/2501.14685)
Comments:
          submitted to MIDL2025

- **What's New**: 본 연구에서는 MedMNIST 데이터셋을 통해 의료 이미지 분류 작업에서 사용되는 다양한 foundation model의 성능을 비교합니다. 이 연구는 다양한 convolutional 및 Transformer 기반 모델을 포함하여 이전 연구에서 제시되지 않은 모델 선택을 통해 더 깊이 있는 통찰력을 제공합니다. 또한, 이미지 크기와 리사이징 전략의 효과에 대한 탐구도 포함되어 있어, 의료 이미지 분류 작업으로의 전이 가능성에 대한 새로운 인사이트를 제공합니다.

- **Technical Details**: 의료 이미지 분류 연구에서 사용된 foundation model은 CNN 기반 및 ViT 기반 아키텍처를 통해 수행되었습니다. CNN 모델은 VGG16, DenseNet-121 등 다양한 백본을 포함하며, ViT 모델은 최적화 전략에 따라 구분하여 4개의 모델이 활용되었습니다. 모델 평가를 위한 학습 과정에서는 15,000회의 반복 훈련이 진행되며, AdamW 옵티마이저가 사용되었습니다.

- **Performance Highlights**: 연구 결과, 다양한 기법으로 훈련된 foundation model들은 의료 이미지 분류 작업에서 뛰어난 잠재력을 보여주었습니다. 각 모델의 성능은 평균 ± 표준편차 방식으로 평가되어, 서로 다른 이미지 해상도에서 일관된 결과가 나타났습니다. 또한, 실험을 통해 제안된 다양한 리사이징 전략이 전반적인 분류 성능에 긍정적인 영향을 미쳤음을 확인하였습니다.



### Improved Vessel Segmentation with Symmetric Rotation-Equivariant U-N (https://arxiv.org/abs/2501.14592)
Comments:
          Accepted by IEEE ISBI 2025

- **What's New**: 본 논문에서는 효율적인 대칭 회전 불변(symmetric rotation-equivariant) 컨볼루션(SRE-Conv) 커널을 U-Net 아키텍처에 적용하여 의료 이미지 분할 성능을 극대화하는 방법을 제안합니다. 기존의 CNN 기반 분할 기법은 회전이나 반사를 고려하지 않음으로써 성능 저하를 초래해 왔습니다. 제안하는 SRE U-Net 모델은 특히 회전된 이미지를 처리할 때 우수한 결과를 보여주며, 이는 기존의 불변 학습 방법보다 더 적은 학습 파라미터와 메모리 비용을 요구합니다.

- **Technical Details**: 이 연구에서는 SRE-Conv 커널을 통해 회전 및 반사 불변 기능을 학습합니다. SRE-Conv는 커널을 중앙 대칭 형태로 매개화하여 많은 중복 값을 갖도록 설계되었습니다. 최종적으로 SRE-Conv 커널은 파라미터를 효율적으로 분할하여 정의된 규칙에 따라 생성되며, 이는 표준 컨볼루션보다 더 낮은 계산 복잡도와 부동 소수점 연산 수(FLOPs)를 달성하게 합니다.

- **Performance Highlights**: DRIVE 망막 혈관 분할 데이터셋을 통해 제안된 SRE U-Net의 성능을 평가한 결과, 회전된 이미지 입력에 대해 정확한 혈관 세분화를 달성했습니다. 제안된 방법은 표준 U-Net과 비교하여 성능이 현저히 향상되었고, 기존의 불변 학습 방식 및 최신 기술(State-of-the-art)과 비교하여도우수한 결과를 보였습니다. 검증된 수치 성과는 다양한 회전 조건에서의 일관된 결과를 제공하여 실제 진단 애플리케이션에서도 유용함을 입증합니다.



### Scene Understanding Enabled Semantic Communication with Open Channel Coding (https://arxiv.org/abs/2501.14520)
- **What's New**: 이번 연구에서는 6G 네트워크의 요구에 부응하는 의미 기반(semantic) 통신 시스템을 제안합니다. 기존의 시스템이 가지는 정적인 지식 베이스의 한계를 극복하기 위해, 대규모 언어 모델(Large Language Models, LLMs)과 개방형 채널 코딩(open channel coding)을 결합한 새로운 시스템인 OpenSC를 개발했습니다. OpenSC는 다양한 환경과 과제에 유연하게 적응할 수 있도록 설계되었습니다.

- **Technical Details**: OpenSC 시스템은 장면 이해(scene understanding)과 구조화된 장면 그래프(scene graphs)를 활용하여 시각적 요소 간의 관계를 효율적으로 인코딩합니다. 또한, 동적(open) 채널 코딩 기법을 도입함으로써 실시간으로 채널 조건에 적응하여 데이터 전송의 효율성을 높입니다. 이러한 구조적인 의미 인코딩 방식은 중요한 객체와 관계를 선택적으로 인코딩하여 데이터의 중복성을 최소화합니다.

- **Performance Highlights**: 실험 결과, OpenSC는 의미 이해(semantic understanding)와 전송 효율성(transmission efficiency)에서 유의미한 개선을 보여주었습니다. 이 시스템은 Visual Question Answering(VQA)과 같은 복잡한 다중 모달 응용 분야에서 더욱 향상된 성능을 발휘하며, 중앙 집중식 지식 베이스에서 분산형 지식 모델로 전환할 수 있는 가능성을 제공합니다.



### LiDAR-Based Vehicle Detection and Tracking for Autonomous Racing (https://arxiv.org/abs/2501.14502)
Comments:
          13 pages

- **What's New**: 이번 논문은 Team PoliMOVE의 자율 주행 레이스카에 배포된 LiDAR 기반 인식 알고리즘을 소개합니다. 이 기술은 Indy Autonomous Challenge 시리즈에서 여러 차례 우승한 경험이 있습니다. 특히, 빠른 Point Cloud 분할 기술과 특정 차량 자세 추정 방법론이 도입되었고, 변동 단계의 다중 목표 추적 알고리즘이 통합되었습니다. 실험 결과는 이 알고리즘이 자율 주행 응용에 적합함을 입증하며, 275 km/h 이상의 속도로 완전 자율 추월이 가능하다는 점을 강조합니다.

- **Technical Details**: 자율 주행 차의 Point Cloud 데이터에서 신뢰할 수 있는 차량 탐지를 위한 온라인 알고리즘을 개발하였습니다. 이 알고리즘은 LiDAR 센서를 통해 수집한 비구조적 Point Cloud에서 작동하며, 라벨 데이터 없이 2D 자세 추정과 모션 추적을 수행합니다. 차량의 과속과 회전 비율을 예측할 수 있는 다중 목표 추적(Multi-Target Tracking) 모듈을 통합하여, 최종적으로 상대 차량의 궤적 예측 및 자율 주행 차의 궤적 계획 모듈에 연결합니다.

- **Performance Highlights**: 제안된 알고리즘은 빠르고 효율적인 Point Cloud 분할 알고리즘을 사용하여 정보 손실 없이 작동하며, 자율 주행 경주 애플리케이션에서 높은 신뢰성을 입증합니다. 이 논문은 다중 차량 경쟁에서 실제 환경에서 완전 자동 추월을 성공적으로 수행한 사례를 중심으로 성능을 정량적으로 평가하였습니다. 고속 주행 조건에서도 안전한 사고 회피 및 차선 변경을 지원하는 데 필요한 신뢰성을 갖춘 시스템임을 보여줍니다.



### A Note on Implementation Errors in Recent Adaptive Attacks Against Multi-Resolution Self-Ensembles (https://arxiv.org/abs/2501.14496)
Comments:
          4 pages, 2 figures, technical note addressing an issue in arXiv:2411.14834v1

- **What's New**: 이 논문은 최근 Zhang 외 (2024)가 다룬 multi-resolution self-ensemble defense (Fort 및 Lakshminarayanan, 2024)에 대한 adaptive attack의 구현 문제를 문서화하고 있습니다. 구현 과정에서 적대적 작용(perturbation)이 L₁₀ = 8/255의 표준 경계를 20배 이상 넘는 L₁₀ = 160/255에 도달하여, 방어 메커니즘의 성과를 근본적으로 왜곡했습니다. 올바른 경계 내에서 적대적 공격을 제어했을 때, 방어는 의미 있는 강건성을 유지했습니다.

- **Technical Details**: 근본적으로 문제는 반복(iteration) 간 perturbation의 누적(accumulation) 과정에서 발생했습니다. 각 반복(iteration)에서 개별 perturbation을 L₁₀ = 8/255로 제대로 제한했음에도 불구하고, 알고리즘은 수정된 이미지를 새로운 기준으로 삼아 추가적인 perturbation을 쌓아 올려 경계를 초과하게 되었습니다. 이로 인해, 최대 20회의 반복에서도 예상보다 뛰어난 L₁₀ 값이 발생했습니다.

- **Performance Highlights**: 적절한 경계 내에서 adaptive attacks를 수행한 결과, CIFAR-100에서 L₁₀ 공격 하에 20% 이상으로 초기 적대적 정확도를 기록했습니다. 본 연구는 L₁₀ = 8/255 perturbations가 이미지의 인지된 클래스를 변화시킬 수 있음을 보여주며, 기존의 연구 가정을 도전했습니다. 또한, 인간 인지와 모델의 예측이 유사하게 영향을 미칠 수 있다는 흥미로운 사실도 발견했습니다.



### Registration of Longitudinal Liver Examinations for Tumor Progress Assessmen (https://arxiv.org/abs/2501.14483)
- **What's New**: 이번 연구는 간 CT 검사에서 암 진행 상황을 평가하는 새로운 방법을 제안합니다. 기존의 등록 방법들이 주로 구조적 특성에 의존하여 종양 영역을 왜곡하는 문제점을 해결하기 위해, 본 논문은 해부학적 및 기하학적 정보를 기반으로 하는 새로운 등록 프레임워크를 제시합니다. 이는 단순히 시각적 특징 대신 간 세분화에서의 정보를 활용하여, 종양의 외관을 보다 정확하게 보존할 수 있도록 합니다.

- **Technical Details**: 제안된 방법론은 동일한 환자의 각기 다른 시점에서 촬영된 이미지를 정렬하는 데 중점을 둡니다. 간 세분화 도구를 사용하여, 이미지 A와 B의 세분화 마스크를 정의하고 이를 기반으로 변환 함수를 도출합니다. 제안된 프레임워크는 세분화 맵에서 변위 필드(Displacement Field)를 생성하여, 정렬된 세분화 마스크를 얻기 위해 공간 변환 네트워크(Spatial Transformer Network)를 활용합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 기존의 등록 기술보다 더 부드러운 변형을 제공하면서도 종양의 부하를 잘 보존함을 보여줍니다. 본 연구의 방법은 317명의 환자에게서 훈련하고 53명의 환자에서 테스트하여, 모든 실험 결과에서 높은 정확성과 효과성을 입증했습니다. 이러한 성과는 간 CT 스캔의 종양 모니터링 과정을 개선하는 데 기여할 것입니다.



### ECTIL: Label-efficient Computational Tumour Infiltrating Lymphocyte (TIL) assessment in breast cancer: Multicentre validation in 2,340 patients with breast cancer (https://arxiv.org/abs/2501.14379)
Comments:
          Under review. 54 pages including supplementary materials, 2 main tables, 3 main figures, 14 supplementary figures, 4 supplementary tables

- **What's New**: 이 논문에서는 삼중 음성 유방암 환자를 위한 종양 침윤 림프구(TIL)의 평가를 위한 새로운 접근 방식을 제안합니다. 기존의 복잡한 Computation TIL Assessment (CTA) 모델과 달리, 연구팀은 병리학적 주석이 적고도 훈련 시간이 단축된 딥 러닝 기반의 ECTIL 모델을 개발했습니다. 이 모델은 소수의 주석으로도 유방암 환자들의 TIL 점수를 예측할 수 있도록 만들어졌습니다.

- **Technical Details**: ECTIL 모델은 병리학 기초 모델을 사용하여 전체 슬라이드 이미지(WSI)에서 형태학적 특징을 추출합니다. 모델을 훈련하는 데 소수의 샘플만 필요하며, 다양한 외부 코호트에서 병리학자와의 일치도(r=0.54-0.74, AUROC=0.80-0.94)가 입증되었습니다. 또한, ECTIL이 TIL 점수를 직접 회귀 분석하는 방법으로, 수십 배 적은 주석으로 훈련될 수 있는 장점을 가지고 있습니다.

- **Performance Highlights**: 모델의 성능은 ECTIL이 훈련된 데이터셋에서 병리학자 점수와 유사한 생존율 비율(hazard ratio, HR) 0.86을 도출하면서 입증되었습니다. ECTIL은 기존 방법들보다 단순한 구조를 가지고 있으며, 임상 치료 시험에 환자를 선별하는 데 사용될 수 있는 가능성을 가지고 있습니다. 오픈 소스 라이센스 하에 제공되며, 이는 더 많은 연구자들이 사용할 수 있도록 합니다.



### Automatic detection and prediction of nAMD activity change in retinal OCT using Siamese networks and Wasserstein Distance for ordinality (https://arxiv.org/abs/2501.14323)
Comments:
          Solution to the MICCAI 2024 MARIO Challange. First 3 authors contributed equally. Models can be found at this https URL

- **What's New**: 이번 연구는 딥러닝 모델을 사용하여 신생혈관성 노화 관련 황반변성(nAMD)의 진행을 예측하는 새로운 접근법을 제안합니다. 특히, 제안된 모델은 양안(Siamese) 네트워크와 Wasserstein Distance 기반의 손실 함수를 활용하여 증세 변화 예측에서 더 나은 성능을 발휘할 수 있도록 설계되었습니다. 이 연구는 OCT(Optical Coherence Tomography) 이미지를 활용하여 환자의 상태 변화를 탐지하고 예측하는 최적의 솔루션으로 간주됩니다.

- **Technical Details**: 제안된 모델은 SiamRETFound라는 이름의 시아미즈(Siamese) 신경망으로, 두 방문 간의 B-scan 이미지를 비교하여 변화를 감지합니다. 이를 위해 Visio Transformer(ViT) 기반의 구조를 사용하며, 조정된 분류 헤드를 통해 임상적으로 중요 변화의 평가를 수행합니다. 또한, 훈련 데이터는 MARIO 챌린지의 두 가지 작업을 기반으로 구성되며, 전처리 단계에서는 Kermany 데이터셋을 통해 자체적으로 모델을 사전 훈련했습니다.

- **Performance Highlights**: 제안된 모델은 MARIO 챌린지의 초기 리더보드에서 높은 순위를 기록했습니다. 향후 nAMD 치료 관리에 도움을 줄 수 있는 예측 능력을 보여주었으며, 특히 시간에 따른 변화 예측의 중요성이 강조되었습니다. 이 연구는 OCT 이미지를 기반으로 한 임상 결정 지원 시스템의 자동화를 위한 중요한 기반을 제공할 것으로 기대됩니다.



### Snapshot multi-spectral imaging through defocusing and a Fourier imager network (https://arxiv.org/abs/2501.14287)
Comments:
          22 Pages, 7 Figures

- **What's New**: 이번 연구에서는 추가적인 스펙트럼 필터나 맞춤형 부품 없이 표준 흑백 이미지 센서를 이용한 스냅샷 다채널 이미징 방법을 소개합니다. 이 방법은 장의 고유한 색수차(chromatic aberration)를 활용하여 다채널 정보의 물리적 인코딩을 자연스럽게 생성합니다. 이를 통해 기존의 복잡한 시스템 없이도 고품질의 다채널 이미지를 획득할 수 있습니다.

- **Technical Details**: 이 시스템은 딥 러닝 기반의 다채널 푸리에 이미저 네트워크(multi-spectral Fourier Imager Network, mFIN)를 통해 인코딩된 이미지 정보를 신속하게 디코드합니다. 실험에서는 여섯 개의 조명 채널을 사용하여 입력 조명 채널 예측에 92.98%의 정확도를 달성했으며, 다양한 테스트 객체에서 강력한 다채널 이미지 재구성을 보여주었습니다.

- **Performance Highlights**: 딥 러닝으로 강화된 이 프레임워크는 흑백 이미지 센서를 사용한 스냅샷 이미지 캡처를 통해 고품질 다채널 이미지 재구성을 가능하게 합니다. 이를 통해 생명과학(biomedicine), 산업 품질 관리(industrial quality control), 농업 등 다양한 분야에서 응용 가능성을 크게 넓혔습니다.



### Deep Learning-Powered Classification of Thoracic Diseases in Chest X-Rays (https://arxiv.org/abs/2501.14279)
- **What's New**: 이번 연구는 폐 X선 사진을 분석하여 호흡기 질환을 진단하는 데 중점을 두고 있습니다. 최근 COVID-19의 확산에 따라 빠르고 정확한 진단 방법의 필요성이 더욱 커진 상황입니다. 이 연구는 전이 학습(transfer learning)과 딥러닝 기법을 통해 질병 탐지 및 분류의 효율성을 높이고자 했습니다.

- **Technical Details**: 연구에서는 AlexNet, ResNet, InceptionNet과 같은 사전 훈련된 모델을 활용했고, 모델 미세 조정(fine-tuning) 및 Focal Loss를 적용하여 클래스 불균형 문제를 해결했습니다. Grad-CAM 시각화를 통해 모델의 해석 가능성을 향상시키고, 임상적으로 중요한 예측에 영향을 미치는 영역을 확인할 수 있었습니다.

- **Performance Highlights**: InceptionV3 모델은 AUC가 28% 개선되고 F1-Score가 15% 증가하는 성과를 올렸습니다. 이러한 결과는 딥러닝 기술이 진단 과정의 효율성을 높이고 임상 의사 결정 지원에 기여할 수 있는 잠재력을 보여줍니다. 연구 결과는 호흡기 질환의 조기 진단에 중요한 역할을 할 것으로 기대됩니다.



### CDI: Blind Image Restoration Fidelity Evaluation based on Consistency with Degraded Imag (https://arxiv.org/abs/2501.14264)
- **What's New**: 최근 Blind Image Restoration (BIR) 기술이 Generative Adversarial Networks (GAN)와 Diffusion Models (DMs)를 기반으로 시각적 품질을 크게 향상시켰습니다. 그러나 기존의 Full-Reference Image Quality Assessment (IQA) 방법들은 이러한 이미지들을 부정확하게 평가하는 문제를 야기하고 있습니다. 본 논문에서는 BIR의 Solution Non-Uniqueness와 Degradation Indeterminacy 문제를 재검토하고, 이를 해결하기 위한 특화된 BIR IQA 시스템의 구축을 제안합니다.

- **Technical Details**: BIR IQA는 복원된 이미지와 참조 이미지의 직접 비교 대신, 일관성(Consistency with Degraded Image, CDI)을 계산하여 충실도를 평가합니다. 특히, 웨이브렛(domain wavelet) 기반의 Reference Guided CDI 알고리즘을 제안하여 다양한 유형의 열화를 알지 못한 상태에서도 평가할 수 있도록 하였습니다. 또한, 참조 이미지 없이 BIR 충실도를 평가할 수 있는 Reference Agnostic CDI (RACDI) 접근법도 개발하였습니다.

- **Performance Highlights**: DISDCD라는 새로운 데이터세트를 통해 BIR 충실도의 주관적 평가를 위한 실험이 수행되었으며, CDI가 기존의 FR-IQA 방법들에 비해 월등히 우수하다는 것이 검증되었습니다. 이는 BIR IQA 평가 방법의 필요성과 효용성을 강조하며, 설정한 알고리즘과 데이터세트가 산업계와 연구 커뮤니티에 널리 활용될 수 있을 것임을示합니다.



### You Only Teach Once: Learn One-Shot Bimanual Robotic Manipulation from Video Demonstrations (https://arxiv.org/abs/2501.14208)
Comments:
          under review

- **What's New**: 이번 연구에서는 YOTO(You Only Teach Once)라는 새로운 프레임워크를 제안하여, 단 하나의 쌍안경 관찰(binocular observation)로부터 이중 팔 조작(bimanual manipulation) 패턴을 추출하고 주입하여 로봇 팔에게 복잡한 작업을 가르치는 혁신적인 접근 방식을 보여줍니다. 기존 연구들이 사전 정의된 행동 분류나 원격 조작(teleoperation)에 의존했던 것과는 달리, YOTO는 인간 시연 비디오를 활용하여 공간-시간 위치, 동적 자세, 상호작용 상태 등 다양한 요소를 학습할 수 있도록 합니다. 이러한 접근은 단순성, 다양성 및 확장성 측면에서 이점을 제공합니다.

- **Technical Details**: YOTO는 손 움직임을 관찰하여 이중 팔 조작 패턴을 자동으로 학습하는 방법론을 사용합니다. 이를 위해 영상에서 손의 위치, 형태, 접촉 상태 등을 인식하고, 실시간으로 수집된 데이터를 기반으로 키프레임(keyframe) 모션을 생성합니다. 또한, 3차원 포인트 클라우드(point cloud) 데이터를 사용하여 조작 객체의 기하학적 변형을 지원함으로써, 다양한 훈련 시연(demonstration)을 신속하게 생성할 수 있도록 합니다. 이 방법은 기존의 원격 조작 방식보다 빠르고 효율적인 데이터 생성 방법을 제공합니다.

- **Performance Highlights**: 실험 결과, YOTO는 5가지 복잡한 이중 조작 작업을 성공적으로 모방하는 성과를 보였으며, 다양한 시각적 및 공간적 조건에서 강력한 일반화 성능을 보였습니다. BiDP(bimanual diffusion policy)는 복잡한 기술을 모방하는 데 있어 높은 효율성과 효과성을 입증하였으며, 기존의 시각 운동 모방 학습 방법 대비 정확도와 효율성에서 우수한 성능을 기록하였습니다. 이러한 연구 결과를 통해 YOTO 프레임워크는 다양한 이중 조작 작업에 호환이 가능함을 확인하였습니다.



### Sparse Mixture-of-Experts for Non-Uniform Noise Reduction in MRI Images (https://arxiv.org/abs/2501.14198)
Comments:
          Accepted to the WACV Workshop on Image Quality

- **What's New**: 이 논문은 MRI 영상에서 존재하는 비균일 잡음을 제거하기 위한 새로운 희소 전문가 혼합(sparse mixture-of-experts, MoE) 접근 방식을 제안합니다. 각 전문가는 특정 영상 영역에 관련된 잡음 특성을 타겟으로 하여 조정된 특수한 denoising convolutional neural network(CNN)입니다. 이 방법은 합성 및 현실 세계의 뇌 MRI 데이터셋에서 최신 denoising 기법보다 뛰어난 성능을 보여주며, 이전 데이터셋에서 학습된 모델이 보지 못한 데이터셋에서도 효과적으로 일반화됩니다.

- **Technical Details**: 우리는 잔여 학습(residual learning) 접근 방식을 기반으로 하여 잡음 추가(non-uniform noise)가 있는 이미지를 인식합니다. 잡음이 신호에 따라 비균일하게 분포되기 때문에, 편안한 denoising 성능을 보장하기 위해 각 영역에 맞춤화된 모델이 필요합니다. 구체적으로, 우리는 이미지를 세분화하여 지역별로 정확한 잡음 제거를 수행하고, 관련 구조를 독립적으로 denoise하여 고품질 이미지를 재구성합니다.

- **Performance Highlights**: 우리의 제안은 기존의 HydraNet과 같은 지역 기반 접근법보다 더 세분화된 구조를 활용하여 잡음 제거 성능을 개선합니다. 각 세분화된 구조에 할당된 전용 전문가가 있어, 요구되는 고유의 잡음 특성을 잘 처리할 수 있습니다. 실험 결과, 우리는 다양한 구조적 세분화 수준을 평가하였고, 세분화가 높을수록 결과가 개선된다는 것을 확인했습니다.



### UltraLightSqueezeNet: A Deep Learning Architecture for Malaria Classification with up to 54x fewer trainable parameters for resource constrained devices (https://arxiv.org/abs/2501.14172)
- **What's New**: 말라리아 진단을 위한 경량 딥러닝 접근 방식이 주목받고 있습니다. 본 연구에서는 저자원 환경에서 진단 개선 가능성을 가진 SqueezeNet1.1을 선택했습니다. 이는 SqueezeNet1.0의 개선된 버전으로, 원래 모델보다 2.4배 더 효율적입니다.

- **Technical Details**: 세 가지 초경량 아키텍처 변형을 SqueezeNet1.1에 제안하였습니다. 변형 1(모듈 1개), 변형 2(모듈 2개), 변형 3(모듈 4개)으로, 이들은 SqueezeNetV1.1(모듈 8개)보다 더 컴팩트합니다. NIH Malaria 데이터셋을 활용하여 각 모델의 성능을 정확도, 재현율, 정밀도, F1-score 및 AUC(곡선 아래 면적)로 평가하였습니다.

- **Performance Highlights**: SqueezeNet1.1 모델은 모든 메트릭에서 97.12%의 분류 정확도로 최고의 성능을 보였습니다. 변형 3(모듈 4개)은 96.55%의 정확도로 거의 동일한 결과를 제공하면서도 계산 오버헤드를 6배 줄였습니다. 변형 2는 28배, 변형 1은 54배의 훈련 가능 파라미터 감소를 보여주었습니다.



### Fully Guided Neural Schr\"odinger bridge for Brain MR image synthesis (https://arxiv.org/abs/2501.14171)
Comments:
          9 pages,4 figures

- **What's New**: 본 논문에서는 다중 모드 뇌 MRI의 이미지 손실 문제를 해결하기 위해, 최신의 Neural Schrödinger Bridges를 기반으로 한 Fully Guided Schrödinger Bridges (FGSB)라는 새로운 프레임워크를 제안합니다. 이 모델은 최소한의 쌍 데이터만으로도 안정적이고 고품질의 모드 생성을 가능하게 하며, 특정 영역에 대한 ground truth 또는 segmentation network를 제공할 경우 중요 이미지 특징인 병변을 보존하면서 시각화를 수행합니다. FGSB는 현재의 기술적 한계로 인해 발생하는 손실 문제를 완화하고, 임상 데이터 수집의 복잡한 과정을 단순화하고자 합니다.

- **Technical Details**: FGSB는 두 개의 연속적인 단계로 구성됩니다. 첫 번째는 Generation Phase로, 생성된 이미지, 쌍 참조 이미지, 그리고 Gaussian noise를 융합하여 iteratively refinement를 사용하여 문제를 완화하는 단계입니다. 두 번째는 Training Phase로, 생성된 이미지와 대상 모드 간의 매핑을 학습하는 단계입니다. 이 과정에서 상호 정보 손실(mutual information loss)을 활용하여 생성 과정에서의 일관성을 유지하며, 자율 감독(discriminative learning)을 통해 적은 수의 데이터에서도 학습할 수 있도록 지원합니다.

- **Performance Highlights**: 실험 결과, FGSB는 두 명의 피험자에서 생성된 데이터로 대규모 데이터셋에서 훈련된 방법들과 비슷한 생성 성능을 달성했습니다. 또한, 병변 정보의 활용은 FGSB가 중점적으로 보존해야 하는 병변 특징의 보존 능력을 크게 향상시켰음을 시사합니다. 이와 같이 FGSB는 한정된 데이터에서도 우수한 이미지 품질과 임상적으로 중요한 특징을 유지하는 것으로 나타났습니다.



### Reinforcement Learning Platform for Adversarial Black-box Attacks with Custom Distortion Filters (https://arxiv.org/abs/2501.14122)
Comments:
          Under Review for 2025 AAAI Conference on Artificial Intelligence Proceedings

- **What's New**: 이 논문에서는 RLAB라는 새로운 강화 학습 플랫폼을 소개합니다. 이 플랫폼은 사용자가 서로 다른 왜곡 필터(distortion filters)를 선택하여 적대적 사례(adversarial examples)를 생성할 수 있도록 설계되었습니다. RLAB는 입력 이미지에 최소한의 왜곡을 추가하면서도 목표 모델을 오분류시키기 위해 강화 학습 에이전트를 활용합니다.

- **Technical Details**: RLAB는 입력 이미지를 각 단계에서 탐색하여 왜곡을 추가해야 할 민감한 영역을 식별합니다. 이를 위해 새로운 이중 행동(dual-action) 방법을 사용하여 목표 모델에 미치는 영향이 적은 노이즈를 제거합니다. 이러한 방식은 공격의 수렴(convergence)을 더 빠르고 효율적으로 만들어줍니다.

- **Performance Highlights**: 또한 RLAB는 특정 왜곡 유형에 대한 이미지 분류 모델의 견고성을 측정하는 데에도 사용될 수 있습니다. 적대적 샘플을 사용하여 모델을 재훈련(retraining)함으로써 벤치마크 데이터셋에서 평가할 때 견고성이 눈에 띄게 향상되었습니다. 제안된 플랫폼은 오분류를 유발하는 데 필요한 쿼리 수에서 최신 기술(state-of-the-art) 방법들을 초월했습니다.



### Efficient 2D CT Foundation Model for Contrast Phase Classification (https://arxiv.org/abs/2501.14066)
- **What's New**: 이 연구는 2D Foundation model을 활용하여 도메인 변화에 강한 강력한 기법인 CT 대비 단계 분류기를 개발하는 것을 목표로 한다. 기존의 3D CNN 모델보다 빠른 훈련 속도와 더 낮은 메모리 요구 사항을 제공하며, 자원 소비를 줄이면서도 성능을 유지하는 점이 혁신적이다. 또한, 이 연구는 CT 검사에서 소요되는 데이터를 자동화하여 방사선사의 피로를 줄일 수 있는 가능성을 보여준다.

- **Technical Details**: 이 연구는 DeepLesion, VinDr Multiphase, WAW-TACE의 세 가지 공개 데이터 세트를 활용하여 진행되었다. 이들 데이터 세트는 2D DICOM 이미지에서 피처 벡터(1024-길이) 생성을 위해 Masked Autoencoder를 사용하여 훈련된 2D Foundation model을 활용하며, 해당 모델은 self-supervision을 통해 학습되었다. 2D CT 단면 이미지를 3D CT 볼륨으로 처리하기 위해 BERT 모델을 변형하여 피처 임베딩을 처리하는 방식으로 구현하였다.

- **Performance Highlights**: VinDr 데이터 세트에서, 모델은 비대비(non-contrast), 동맥(arterial), 정맥(venous) 단계를 각각 99.2%, 94.2%, 93.1% F1 점수로 매우 높은 성능을 보였다. WAW-TACE 데이터 세트에서도 AUROC 점수 91.0% 및 85.6%를 기록하며, F1 점수는 비대비 및 동맥 단계에서 각각 87.3%와 74.1%로 강력한 성능을 나타냈다. 또한, 3D 감독 모델과 비교하여 훈련 시간이 짧고, 성능이 같거나 더 뛰어나며, 도메인 변화에 대한 강건성을 보여주었다.



### SIDDA: SInkhorn Dynamic Domain Adaptation for Image Classification with Equivariant Neural Networks (https://arxiv.org/abs/2501.14048)
Comments:
          25 pages, 5 figures, 4 tables. code available at: this https URL

- **What's New**: 이 논문에서는 Sinkhorn divergence를 기반으로 하는 새로운 도메인 적응 훈련 알고리즘인 SIDDA(SInkhorn Dynamic Domain Adaptation)를 소개합니다. SIDDA는 효율적인 도메인 정렬을 가능하게 하며, 최소한의 하이퍼파라미터 조정과 컴퓨팅 비용으로 사용할 수 있습니다. 이 방법은 다양한 복잡성의 시뮬레이션 및 실제 데이터 세트에 대해 효과성을 입증하였으며, 특히 동등 변환 신경망(equivariant neural networks, ENNs)과 함께 사용할 때 분류 정확도와 모델 보정 효과를 크게 향상시킵니다.

- **Technical Details**: SIDDA는 옵티멀 전송(optimal transport) 이론을 기반으로 하며, 엔트로픽 정규화(entropic regularization)와 분류 손실(classification loss) 및 DA 손실(domain adaptation loss) 토큰의 가중치를 활성화하여 훈련 과정에서 하이퍼파라미터 조정을 최소화합니다. 이 알고리즘은 다양한 NN 아키텍처와 호환되며 복잡성에 따라 조정할 수 있는 우수한 방법을 제공합니다. 연구에서는 ENNs의 강 robustness를 함께 공부하며, SIDDA를 통한 개선 효과도 조사합니다.

- **Performance Highlights**: SIDDA는 레이블이 없는 타겟 데이터에 대해 약 40%의 분류 정확도 향상을 달성하여 NNs의 일반화 능력을 향상하도록 설계되었습니다. 또한, ECE(Expectation Calibration Error)와 Brier 점수(Brier score)에서 10배 이상의 개선 효과를 보이며, 모델 보정에서도 유의미한 결과를 나타냅니다. SIDDA의 자동화된 접근 방식은 다중 데이터 세트 연구 및 높은 일반화 가능한 모델 개발에 기여할 수 있는 잠재력을 지니고 있습니다.



### Leveraging Multiphase CT for Quality Enhancement of Portal Venous CT: Utility for Pancreas Segmentation (https://arxiv.org/abs/2501.14013)
Comments:
          ISBI 2025

- **What's New**: 이번 연구는 복수의 CT 단계를 활용하여 스캔 품질 향상을 시도한 최초의 연구로, 특히 문맥 정맥 단계를 강화하는 데 중점을 두었습니다. 연구진은 3D Progressive Fusion and Non-local (PFNL) 네트워크를 통해 세 가지 저품질 CT 단계(비대조, 동맥, 문맥 정맥)를 입력으로 사용했습니다. 이 방식을 통해 문맥 정맥 CT의 품질을 향상시켜 췌장 분할(segmentation)의 성능을 3% 개선했습니다.

- **Technical Details**: 본 연구는 VinDr-Multiphase 데이터세트를 사용하여 2015년부터 2020년까지 265명의 환자로부터 총 168개의 CT 연구를 분석했습니다. 비대조, 동맥, 문맥 정맥 단계를 포함하는 CT 스캔을 선택하여 이들을 세 가지 저품질 CT 단계로 변환한 후, 3D PFNL 모델을 통한 품질 향상을 시도했습니다. 손실 함수는 L1 재구성 손실과 3D Sobel 엣지 기반 손실의 조합으로 설정되어 구조의 경계 강화를 목표로 했습니다.

- **Performance Highlights**: 최종 결과는 3D PFNL 모델이 문맥 정맥 단계를 개선하여 췌장 분할 정확도를 3% 향상시켰음을 보여줍니다. 이 연구는 CT 스캔 품질 향상 및 췌장 분할에 있어 유의미한 진전을 이루었으며, 다양한 질병의 진단 및 관리에 있어 중요한 기여를 할 것으로 기대됩니다.



### MCRL4OR: Multimodal Contrastive Representation Learning for Off-Road Environmental Perception (https://arxiv.org/abs/2501.13988)
Comments:
          Github repository: this https URL

- **What's New**: 자율주행차에 대한 연구는 주로 도시 환경에 집중되어 왔으나, 이 논문은 오프로드(Off-Road) 환경 인식을 위한 새로운 접근법인 MCRL4OR을 제안합니다. MCRL4OR은 시각적 이미지, 이동 상태, 제어 동작을 처리하기 위한 세 가지 인코더를 공동으로 학습하며, 이들 간의 연관성을 대조 학습(Contrastive Learning) 프레임워크 내에서 조화롭게 학습합니다. 이 연구는 비 구조적 환경에서 자율주행차가 안전하고 효과적으로 작동하도록 하는 중요한 기여를 하고 있습니다.

- **Technical Details**: MCRL4OR은 (visual observation encoder), (control action encoder), (locomotion state encoder)의 세 가지 분기로 구성되어 있으며, 여러 모달리티 간의 올바른 대응 관계를 예측하는 것을 목표로 합니다. 이 모델은 TartanDrive 데이터셋을 사용하여 사전 훈련(Pre-training)되며, 이후에는 크로스 모달 검색(Cross-modal retrieval), 동역학 예측(Dynamics prediction), 장면 분할(Scene segmentation)과 같은 다양한 하위 인식 작업에 적용됩니다. 이 논문에서 제안한 정렬 전략은 현재 지형에 따라 주어진 제어 동작이 어떻게 운동 상태에 영향을 미치는지를 기반으로 합니다.

- **Performance Highlights**: 실험 결과, MCRL4OR의 사전 훈련된 표현은 다양한 하위 작업에서 향상된 성능을 보여줍니다. 특히, 이 연구는 오프로드 환경에서의 대조 학습 모델에 대한 성능 향상이 입증되었으며, 이를 통해 논문은 MCRL4OR이 대조 학습 전략의 효과성을 강조하고 있습니다. 또한, 사전 학습된 모델이 다양한 작업에서 일관되게 성능을 개선한다는 점에서 다중 모달 표현의 일반화 가능성을 확인했습니다.



### Pilot: Building the Federated Multimodal Instruction Tuning Framework (https://arxiv.org/abs/2501.13985)
- **What's New**: 본 논문에서는 분산 장치에서 다양한 모달리티의 지시 데이터로 MLLMs를 협력적으로 세밀하게 조정하는 데 중요한 새로운 작업인 Federated Multimodal Instruction Tuning(FedMIT)을 탐구합니다. 이를 위해 'Pilot'이라는 연합 다중 모달 지시 조정 프레임워크를 제안합니다. 이 프레임워크는 비전 인코더와 LLM을 연결하는 부분에 'adapter on adapter'의 두 가지 단계로 통합되어 있습니다.

- **Technical Details**: 1단계에서는 비주얼 정보를 기반으로 작업 특화 기능과 클라이언트 특화 기능을 추출합니다. 2단계에서는 Cross-task Mixture-of-Adapters(CT-MoA) 모듈을 구축하여 작업 간 상호작용을 수행합니다. 각 클라이언트는 지역 데이터의 개인화 정보를 캡처할 뿐만 아니라, 다른 작업에서의 일반적인 지식을 학습하여 모델 성능을 향상시킬 수 있도록 합니다.

- **Performance Highlights**: 이 프레임워크는 다양한 로컬 클라이언트의 분산 데이터를 협력적으로 활용하여 지시 조정 중 작업 이질성에 영향을 받지 않고 교차 작업 지식을 학습할 수 있습니다. 실험을 통해 Pilot 방법이 최신 LLaVA 모델에서 두 가지 다른 교차 작업 시나리오에서 효과적임을 확인했습니다.



### Synthetic CT image generation from CBCT: A Systematic Review (https://arxiv.org/abs/2501.13972)
Comments:
          21 pages, 14 Figures, Accepted in the IEEE Transactions on Radiation and Plasma Medical Sciences

- **What's New**: 이번 연구는 딥러닝 방법론을 사용하여 콘빔 컴퓨터 단층 촬영(CT) 데이터로부터 합성 CT(sCT) 이미지를 생성하는 새로운 접근법을 제시하고 있습니다. 이 체계적인 리뷰는 2014년부터 2024년까지의 문헌을 평가하여 sCT 생성에 대한 기존 연구를 종합적으로 다루고 있습니다. 특히 합성 CT 생성에 있어 딥러닝의 활용 비중이 커지며, 자주 사용되는 아키텍처로는 CNN, GAN, Transformer 및 확산 모델 등이 포함됩니다.

- **Technical Details**: 연구에서는 이미지 품질의 개선을 위한 여러 가지 평가 지표인 평균 절대 오차(MAE), 제곱근 평균 제곱 오차(RMSE), 피크 신호 대 잡음 비율(PSNR), 구조적 유사성 지수(SSIM)를 활용하여 sCT 이미지와 표준 CT(pCT) 간의 비교 가능성을 확인하였습니다. 이 과정에서, 비균일한 시야(FOV)와 임상적 절차와의 통합 등 여러 도전 과제가 논의되며, 향후 연구와 표준화의 필요성이 강조됩니다. 또한, 강화학습을 통해 확산 수준을 조절하는 등 다양한 딥러닝 프레임워크가 개발되고 있습니다.

- **Performance Highlights**: 연구 결과는 sCT 기반 접근법이 개인 맞춤형 치료 계획 및 적응형 방사선 치료에서의 중요한 역할을 수행할 가능성을 강조합니다. 딥러닝 기술이 고해상도의 CBCT 이미지를 효과적으로 생성하여 실시간 치료 조정에 기여할 수 있는 잠재력을 보여줍니다. 향후 이와 같은 기술이 방사선 치료의 정확성과 환자 치료의 품질 향상에 기여할 것으로 기대됩니다.



### Patch-Based and Non-Patch-Based inputs Comparison into Deep Neural Models: Application for the Segmentation of Retinal Diseases on Optical Coherence Tomography Volumes (https://arxiv.org/abs/2501.13970)
Comments:
          6 pages, 1 figures, 2 tables, submitted to 15th IEEE Symposium on Computer Applications & Industrial Electronics

- **What's New**: 이 논문은 나이 관련 황반 변성(AMD)의 치유를 위한 자동화된 분석 기법을 소개합니다. 특히, 깊은 학습 모델의 입력 크기 설정이 성능에 미치는 영향을 분석하여, 패치 기반 입력이 전체 이미지를 사용하는 경우보다 성능을 개선하는 방법을 제시합니다. 이 연구는 2D, 2.5D, 3D 데이터를 사용하여 서로 다른 입력 크기에서 깊은 신경망의 성능을 비교합니다.

- **Technical Details**: 정밀한 계측 기법인 Optical Coherence Tomography (OCT)를 활용하여 망막의 다양한 층과 액체를 이미지화하고 연구합니다. 이 논문에서는 Intraretinal fluid (IRF), Subretinal fluid (SRF), Pigment epithelial detachment (PED)와 같은 망막 질환에 대한 자동 분할 성능을 조사합니다. 그리고 다수의 연구에서 발표된 다양한 딥러닝 모델의 비교를 통해 그 효율성을 분석합니다.

- **Performance Highlights**: 연구에서는 딥러닝 모델이 인간의 성능을 초월하는 결과를 보여주며, Dice Similarity Coefficient (DSC) 점수가 0.88로 나타났습니다. 이는 패치 기반 모델의 성능 향상이 입증된 결과로, 기존 인간의 평균 성능인 0.71보다 현저히 높은 수치입니다. 이 모델은 각기 다른 크기의 입력을 활용하여 세밀한 위험 요소들을 효과적으로 구별하는 데 성공적인 결과를 나타냈습니다.



### LiCAR: pseudo-RGB LiDAR image for CAR segmentation (https://arxiv.org/abs/2501.13960)
Comments:
          This is a preprint version of the work accepted at 5th International Conference on Robotics, Computer Vision and Intelligent Systems (ROBOVIS 2025)

- **What's New**: 이 연구에서는 LiDAR 센서를 통해 생기는 가상의 RGB 이미지에서 자동차를 세분화하는 새로운 데이터셋이 생성되었습니다. 이 데이터셋은 LiDAR 센서의 반사율, 근적외선 및 신호 강도 2D 이미지를 결합하여 만든 구형 범위 이미지(Spherical Range Image, SRI)를 포함하고 있습니다. 또한, You Only Look Once (YOLO)-v8 모델을 사용하여 88%의 경계 상자(Bounding Box) 탐지 정확도와 81.5%의 마스크 세분화 정확도를 달성하였습니다.

- **Technical Details**: LiDAR 센서를 통해 생성된 데이터를 기반으로 두 단계의 이미지를 생성합니다. 첫 번째 단계에서, LiDAR 데이터를 이용하여 가상의 RGB 포인트 클라우드를 생성하고, 두 번째 단계에서 이를 SRI로 변환합니다. YOLO-v5, YOLO-v7, YOLO-v8의 여러 단일 단계 인스턴스 세분화 방법이 적용되었으며, 이전 모델들보다 새롭고 강력한 방식인 focal loss와 mixup 데이터 증강 방법이 도입되었습니다.

- **Performance Highlights**: 실험 결과, YOLO 모델은 각각 400장의 이미지를 기반으로 훈련되었으며, 14개 자동차 인스턴스가 포함된 다양한 환경의 이미지를 세분화하였습니다. 최종적으로 YOLO 모델을 사용하여 높은 인식률과 정확도로 자동차를 추적하였으며 현실 세계의 실험에서도 뛰어난 성능을 발휘하였습니다. 데이터셋은 온라인에서 제공되며, 훈련/검증/테스트 세트를 85/10/5 비율로 나누어 사용되었습니다.



New uploads on arXiv(cs.AI)

### Recommending Actionable Strategies: A Semantic Approach to Integrating Analytical Frameworks with Decision Heuristics (https://arxiv.org/abs/2501.14634)
- **What's New**: 이 논문은 전략 프레임워크와 의사결정 휴리스틱을 의미 분석(semantic analysis)을 통해 통합하여 실행 가능한 전략적 추천을 제공하는 새로운 접근법을 제시합니다. 기존의 분석 프레임워크는 체계적인 평가 및 계획 모델을 제공하며, 의사결정 휴리스틱은 경험적 지식을 담고 있습니다. 하지만 이 두 전통은 역사적으로 독립적이었으므로, 이 연구를 통해 두 가지 방법론을 연결하여 더 효과적인 의사결정을 가능하게 합니다.

- **Technical Details**: 연구는 6C 모델과 서른 여섯 가지 계략(The Thirty-Six Stratagems)을 통합하여, 벡터 스페이스 표현(vector space representations)과 의미 유사성 계산(semantic similarity calculations)을 활용합니다. 이 과정을 통해 프레임워크 파라미터를 휴리스틱 패턴에 매핑하고, 심층 의미 처리와 제한된 대형 언어 모델(LLMs)을 결합한 계산 아키텍처가 구성됩니다. 이를 통해 다양한 분석 프레임워크와 휴리스틱 집합에 일반화 가능하며, 전략적 추천 시스템을 생성하는 플러그 앤 플레이 아키텍처를 개발합니다.

- **Performance Highlights**: 이 연구는 기업 전략 케이스 스터디를 통해 개발된 접근법의 효과성을 입증합니다. 이러한 방법론은 의사결정자가 자연어로 시나리오를 표현할 수 있게 하여, 관련 점수를 계산하고 적절한 추천을 반환하는 상호작용적 시뮬레이션 환경을 제공합니다. 이로써, 포괄적인 전략 분석과 입증된 휴리스틱 통찰력을 통합하여 실행 가능한 가이드를 제공함으로써 복잡한 환경에서도 유효한 의사결정을 지원하는 시스템을 제안합니다.



### Extracting Problem Structure with LLMs for Optimized SAT Local Search (https://arxiv.org/abs/2501.14630)
- **What's New**: 이 논문은 Conflict-Driven Clause Learning (CDCL) 솔버의 성능을 향상시키는 새로운 접근법을 제시합니다. 이는 대형 언어 모델(LLMs)을 활용하여 Python 기반의 인코딩 코드를 분석함으로써 문제 구조에서 hidden structural patterns를 발견합니다. 이 방법은 문제 인스턴스에 관계없이 강력한 초기 할당을 자동으로 생성하는 특수화된 local search алгоритмы를 생성하는 데 중점을 둡니다.

- **Technical Details**: 제안된 방법은 PySAT 코드를 읽고 해석하여 SAT로 변환되는 문제 구조를 이해하는 알고리즘을 만듭니다. LLM은 그래프 연결, 경로 제약 및 카운팅 한계와 같은 높은 수준의 구조를 발견한 후, 이를 이용하여 특수화된 local search 절차를 구축합니다. 이 절차는 특정 인스턴스가 아닌 인코딩에 초점을 맞추어, 동일한 방식으로 인코딩된 모든 인스턴스에 적용 가능합니다.

- **Performance Highlights**: 논문은 Directed Feedback Vertex Set, Bounded Depth Decision Trees 및 Treewidth와 같은 여러 구조적 문제에서 방법의 유효성을 평가하였습니다. LLM이 생성한 알고리즘은 인코딩 구조를 효과적으로 활용하여 성능을 높이며, 기존 SAT 솔버보다 12개의 추가 인스턴스를 해결하는 데 성공했습니다. 이러한 결과는 LLM의 사용이 SAT 문제 해결을 위한 자동화된, 문제 인식 전처리의 가능성을 보여줍니다.



### Hybrid Quantum-Classical Multi-Agent Pathfinding (https://arxiv.org/abs/2501.14568)
- **What's New**: 본 연구는 Branch-and-Cut-and-Price (BCP) 기반으로 최적의 하이브리드 양-고전적 (quantum-classical) 다중 에이전트 경로 찾기 (MAPF) 알고리즘을 제시합니다. 이 알고리즘은 충돌 그래프를 기반으로 하여 QUBO 문제를 반복적으로 해결하는 방식으로 양자 컴퓨팅을 통합합니다. 실제 양자 하드웨어에서의 실험 결과는 본 접근법이 이전 QUBO 형식 및 기존 MAPF 솔버에 비해 우위를 점함을 보여줍니다.

- **Technical Details**: MAPF는 동시에 다수의 에이전트에 대해 충돌하지 않는 경로를 계산하는 문제로, 엄청난 계산적 도전 과제가 존재합니다. 기존 기술로는 Conflict-based Search (CBS)와 Branch-and-Cut-and-Price (BCP) 알고리즘이 있으며, BCP는 단일 에이전트 경로 찾기 문제를 해결하고, ILP (Integer Linear Programming)를 통해 에이전트에 경로를 할당하는 방식으로 작동합니다. 양자 컴퓨팅은 QUBO 문제를 더 빠르게 해결할 수 있는 가능성을 제공하며, 이는 고차원 변수 간의 관계를 효과적으로 탐색할 수 있는 장점이 있습니다.

- **Performance Highlights**: 양자 하드웨어에서 수행된 실험들은 본 하이브리드 알고리즘의 성능이 기존의 MAPF 솔버와 QUBO 문제의 이전 공식화를 능가함을 시사합니다. 이 알고리즘은 대규모 에이전트 경로 찾기 문제에 있어 기초적인 솔루션 품질을 높이는 데 기여할 수 있습니다. 또한, 본 연구는 기계적 접근과 양자 기법의 시너지를 활용하여 실용적인 응용에서의 효과성을 입증합니다.



### VERUS-LM: a Versatile Framework for Combining LLMs with Symbolic Reasoning (https://arxiv.org/abs/2501.14540)
- **What's New**: 이 논문에서는 VERUS-LM이라는 새로운 프레임워크를 소개하고 있습니다. 이 프레임워크는 대규모 언어 모델(LLMs)과 기호 추론 시스템의 장점을 결합하여 복잡한 추론 작업을 효과적으로 해결하고자 합니다. VERUS-LM은 비전문적인 프롬프트를 사용하여 다양한 논리적 추론 작업을 지원하며, 지식과 쿼리를 명확히 분리하여 시스템의 적응성을 향상시킵니다.

- **Technical Details**: VERUS-LM은 일반적인 프롬프트 메커니즘을 사용한 새로운 신경기호 프레임워크입니다. 이 프레임워크는 선언적 지식과 질문 간의 명확한 구분을 통해 지식 기반 효율성을 높이며, 복잡한 의사 결정을 요구하는 다양한 논리적 추론 작업을 지원합니다. 이러한 접근은 최적화와 제약 만족과 같은 풍부한 추론 형태를 가능하게 합니다.

- **Performance Highlights**: 새로운 데이터셋을 통해 검증한 결과, VERUS-LM은 다양한 추론에서 이전의 LLM을 능가하는 뛰어난 성능을 보였습니다. 또한, 공통 추론 벤치마크에서도 우수한 결과를 달성하여 다른 최신 기술들과 경쟁할 수 있는 성과를 인증했습니다. 특히, AR-LSAT 데이터셋에서의 성능은 현저히 뛰어나며, 이 프레임워크는 신경기호 인공지능 시스템의 가능성을 한 단계 끌어올리는 중요한 발전을 보여줍니다.



### In System Alignments we Trust! Explainable Alignments via Projections (https://arxiv.org/abs/2501.14360)
- **What's New**: 본 논문에서는 시스템 로그(Logs)와 정규 프로세스 모델(Models) 사이의 정렬 과정을 다루며, "relaxations" 개념을 도입하여 불완전한 모델과 로그의 문제를 해결하는 방법을 제안합니다. 이러한 relaxed alignments는 신뢰할 수 있는 정보와 신뢰할 수 없는 정보를 구별하여 프로세스의 근본적인 이해를 높이고 데이터 품질 문제를 드러내는데 도움을 줍니다. 이 방식은 복잡한 상호작용을 가진 여러 엔티티가 포함된 프로세스의 효과적인 분석을 가능하게 합니다.

- **Technical Details**: 이 논문은 우선적으로 멀티셋(Multiset)과 부분 순서 집합(Partially Ordered Sets)에 대한 정의를 제공하고, 다음으로 Petri Nets와 관련된 기본 개념들을 소개합니다. 주목할 점은 대부분의 정의가 이전 연구를 반복하여 제시되며, 이를 통해 독립적인 논문으로서의 완전성을 목표로 하고 있다는 것입니다. 여기서는 어려운 영어 용어를 최소화하고, 수학적 표현을 통해 이론적 기초를 마련하고 있습니다.

- **Performance Highlights**: 연구에서는 패키지 배송 프로세스와 같은 실제 사례를 통해 제안한 방법을 시연합니다. 각 프로세스의 로그와 모델은 서로 상충하는 정보를 포함할 수 있으며, relaxed alignments를 통해 이러한 부분적 준수 여부를 드러낼 수 있습니다. 이러한 기술은 프로세스의 보다 정확한 표현을 가능하게 하고, 시스템의 진정한 행동을 이해하고 평가하는 데 기여할 수 있습니다.



### Exploring the sustainable scaling of AI dilemma: A projective study of corporations' AI environmental impacts (https://arxiv.org/abs/2501.14334)
- **What's New**: 본 논문에서는 인공지능(AI)과 특히 대형 언어 모델(LLM)의 환경적 영향을 평가하기 위한 새로운 방법론을 제안합니다. AI의 기후 변화에 미치는 영향을 이해하려는 비즈니스에 유용한 통찰력을 제공하면서도, AI와 생애 주기 평가(LCA)에 대한 전문 지식이 필요하지 않습니다.

- **Technical Details**: 연구 결과, 대규모 생성형 AI 모델이 기존 모델에 비해 최대 4600배 더 많은 에너지를 소비한다는 것이 확인되었습니다. 우리의 모델링 접근법은 AI 사용의 증가, 하드웨어 컴퓨팅 효율, 그리고 IPCC 시나리오에 따른 전력 혼합의 변화를 반영하여 2030년까지의 AI 전력 사용을 예측합니다.

- **Performance Highlights**: 높은 채택 시나리오 하에서, 복잡한 생성형 AI 및 에이전트 채택 덕분에 AI 전력 사용은 기하급수적으로 증가할 것으로 예상됩니다. 2030년까지 생성형 AI의 환경적 영향을 줄이기 위해서는 AI 가치 사슬 전반에 걸친 협력이 필수적이며, 단독의 하드웨어 효율성 개선이나 모델 효율성만으로는 부족함을 강조합니다.



### MASTER: A Multi-Agent System with LLM Specialized MCTS (https://arxiv.org/abs/2501.14304)
Comments:
          Accepted by main NAACL 2025

- **What's New**: 이번 논문에서는 LLM(Large Language Models)의 계획 능력을 개선하기 위해 새로운 다중 에이전트 시스템인 MASTER를 소개합니다. MASTER는 LLM 전문 MCTS(Monte Carlo Tree Search) 알고리즘을 기반으로 에이전트를 모집하고 상호작용하는 구조로, 에이전트 수를 작업의 복잡성에 따라 자동으로 조정합니다. 이 시스템은 기존의 MCTS의 한계점을 극복하여, 시뮬레이션을 통한 보상 획득 대신 LLM의 자기 평가 기능을 활용해 보상을 할당합니다.

- **Technical Details**: MASTER 프레임워크는 MCTS의 시뮬레이션 절차를 제거하고, LLM이 주어진 작업에 대해 더 많은 맥락을 제공하여 자기 평가를 수행하도록 돕습니다. 보상의 객관성을 높이기 위해 LLM의 신뢰도를 보상의 가중치로 사용하고, 잘못 배분된 보상을 업데이트할 수 있는 역전파 메커니즘을 포함합니다. 이 시스템은 다양한 작업(예: HotpotQA, WebShop)에서 실험을 통해 76% 및 80%의 정확도를 달성하며 최신 성능을 기록합니다.

- **Performance Highlights**: MASTER는 다양한 작업을 통해 당시 상태에서 최고의 성능을 발휘하며, 해당 데이터셋에서 새로운 최첨단(SOTA) 성능을 기록했습니다. HotpotQA에서는 76%의 정확도를 기록하고, WebShop에서는 80%의 정확도로 다른 접근 방식들을 능가했습니다. 이러한 결과는 LLM의 계획 능력과 효율성을 한층 더 강화하였음을 시사합니다.



### Fast Think-on-Graph: Wider, Deeper and Faster Reasoning of Large Language Model on Knowledge Graph (https://arxiv.org/abs/2501.14300)
- **What's New**: 이번 논문에서는 Fast Think-on-Graph (FastToG)라는 혁신적인 패러다임을 제안합니다. FastToG는 그래프 정보를 활용하여 LLM이 '커뮤니티 단위'로 사고하도록 고안되었습니다. 이 접근법은 Community Detection 기법을 사용하여 더 깊은 상관관계를 포착하고, 코스와 파인의 두 단계의 커뮤니티 프루닝을 통해 빠른 정보 검색을 가능하게 합니다.

- **Technical Details**: FastToG는 그래프 내에서 커뮤니티 감지를 통해 추론 체인을 구축하고 Community-to-Text 기법을 활용하여 커뮤니티 구조를 텍스트 형식으로 변환합니다. Local Community Search (LCS)라는 새로운 기법을 도입하여 그래프의 지역 범위에서 커뮤니티를 탐지하고, 효율성을 높이기 위해 모듈성 기반의 코스 프루닝과 LLM 기반의 파인 프루닝을 적용합니다.

- **Performance Highlights**: 실험 결과, FastToG는 이전 방법들에 비해 더 높은 정확도, 더 빠른 추론, 그리고 더 높은 설명 가능성을 보여주었습니다. 커뮤니티 기반의 추론이 추론 체인을 단축시켜 LLM 호출 횟수를 줄일 수 있음을 확인했습니다. 또한 FastToG는 LLM에 대한 검색 과정을 단순화하고 사용자에게 더 나은 설명 가능성을 제공합니다.



### Top Ten Challenges Towards Agentic Neural Graph Databases (https://arxiv.org/abs/2501.14224)
Comments:
          12 Pages

- **What's New**: 이 논문은 기존 Neural Graph Databases (NGDBs)의 한계를 극복하기 위해 Agentic Neural Graph Databases (Agentic NGDBs)라는 새로운 개념을 제안합니다. Agentic NGDBs는 자율적인 쿼리 구성, 신경 쿼리 실행, 지속적인 학습과 같은 세 가지 핵심 기능을 통해 인공지능(AI) 시스템의 자율성과 적응력을 높입니다. 이러한 접근은 특히 데이터가 불완전하거나 노이즈가 많은 경우를 다룰 때 더욱 유용합니다.

- **Technical Details**: 논문에서는 Agentic NGDBs를 구현하기 위해 해결해야 할 열 가지 주요 도전 과제를 제시합니다. 이들은 의미 단위 표현(semantic unit representation), 유추적 추론(abductive reasoning), 확장 가능한 쿼리 실행(scalable query execution), 대규모 언어 모델(LLMs)과의 통합(integration with foundation models) 등을 포함합니다. 이러한 기술적 도전 과제를 극복함으로써, Agentic NGDB들은 지능적이고 스스로 개선하는 데이터 관리를 가능하게 합니다.

- **Performance Highlights**: Agentic NGDBs는 기존 NGDBs의 기능을 넘어 자율성과 능동적 학습의 특성을 도입하여 데이터 관리 분야에서 진일보한 성과를 보여줍니다. 특히 데이터의 복잡성과 대규모 처리를 다루는 데 있어 뛰어난 성능을 발휘하고, 사용자가 정의한 쿼리에 의존하지 않고도 의미 있는 분석을 할 수 있습니다. 이들은 고급 추론 및 의사결정 지원을 통해 현대 데이터 중심 응용 프로그램들에서 필수적인 역할을 할 것으로 기대됩니다.



### Distributed Multi-Agent Coordination Using Multi-Modal Foundation Models (https://arxiv.org/abs/2501.14189)
- **What's New**: 이번 연구에서는 시각 언어 지침에 기반한 배포 제약 최적화 문제(VL-DCOPs)를 제안합니다. 이는 자연어와 비주얼 지침을 통해 자동으로 제약 조건을 생성할 수 있는 대규모 다중 모달 시스템을 활용하여, 인간과 에이전트 간의 상호작용을 모델링합니다. 이러한 접근 방식은 기존 DCOPs의 한계를 극복하고 동적인 문제 해결이 가능하게 합니다.

- **Technical Details**: VL-DCOPs는 A1, A2, A3의 세 가지 에이전트 유형으로 구성됩니다. A1은 전통적인 알고리즘을 사용하면서도 LFM을 통해 인간 지침을 파싱하는 신경 상징 에이전트이며, A2는 불확실성을 해결하는 대화형 솔루션을 제공하는 보다 정교한 신경 상징 에이전트입니다. 마지막으로 A3는 지침에 따라 다양한 최적화 알고리즘을 시뮬레이션하는 완전 신경 에이전트입니다.

- **Performance Highlights**: VL-DCOPs를 평가하기 위한 세 가지 새로운 벤치마크가 설계되었습니다. 이들 벤치마크는 각각 고전적인 분산 그래프 색칠 문제와 실제 회의 일정 문제를 기반으로 하며, 최신 LLM 및 VLM을 사용하여 성능을 비교 분석합니다. 이 연구는 향후 DCOP literatures를 확장할 수 있는 흥미로운 연구 방향을 제시하고 있습니다.



### Human-Alignment Influences the Utility of AI-assisted Decision Making (https://arxiv.org/abs/2501.14035)
- **What's New**: 이번 연구에서는 AI 모델의 신뢰성(신뢰도)과 사용자의 신뢰(신뢰감) 간의 정렬 정도가 AI 지원 의사결정의 효용에 미치는 영향을 조사했습니다. 이전 연구에서는 AI 신뢰도 값의 명확한 해석이 의사결정자에게의 신뢰형성에서 어려움을 초래한다고 지적했으며, 이 작업은 이를 실증적으로 규명하고자 한 것입니다.

- **Technical Details**: 연구에서는 703명의 참가자를 대상으로 단순한 결정 과제인 카드 게임을 설계하고 실행하였습니다. 게임은 AI 모델을 사용하여 참가자들이 무작위로 선택된 카드의 색상을 맞추는 것으로, AI 신뢰도에 따라 제공되는 카드의 색상 비율이 조정되었습니다. 각 참가자는 AI 신뢰도와 자신의 신뢰도 간의 정렬을 조절하여 네 가지 그룹 중 하나에 무작위 배정되었습니다.

- **Performance Highlights**: 결과에 따르면 AI 신뢰도와 사용자 신뢰도 간의 정렬 정도가 높을수록 AI 지원 의사결정의 효용이 증가하는 긍정적인 상관관계가 발견되었습니다. 또한, AI 신뢰도를 후처리하여 참가자의 신뢰도와 다중 보정(multicalibration)을 시도함으로써, 정렬의 정도와 AI 지원 의사결정의 효용을 동시에 증가시킬 수 있다는 것이 확인되었습니다.



### Prompt-Based Monte Carlo Tree Search for Mitigating Hallucinations in Large Models (https://arxiv.org/abs/2501.13942)
- **What's New**: 이 연구는 인공지능에서 대형 모델의 응용 능력을 복잡한 과학 연구 문제를 처리하는 데에 강화하기 위한 개선된 Monte Carlo Tree Search (MCTS) 방법을 제안합니다. 제안된 방법은 탐색 매개변수의 동적 조정 및 적응형 선택 전략을 도입해 탐색과 활용의 균형을 더욱 잘 맞추고, 허구 현상을 줄입니다. 연구 결과, 개선된 MCTS 방법이 기존 모델들과 비교하여 더 나은 성능을 보임을 확인했습니다.

- **Technical Details**: 제안된 개선된 MCTS 알고리즘은 선택, 확장, 시뮬레이션, 백프로파게이션의 네 단계로 이루어져 있습니다. 동적 탐색 매개변수 조정 기능은 노드 방문 수에 따라 조정되며, 탐색 초기에는 더 많은 경로를 시험하여 지역 최적해에 빠지지 않도록 합니다. 또한, 문제 복잡성을 평가하여 적합한 시뮬레이션 전략(탐욕 전략 또는 무작위 전략)을 선택하는 적응형 기능도 포함되어 있습니다.

- **Performance Highlights**: SciEval 데이터셋의 네 개 하위 집합에 대한 테스트에서 Glm-4-flash+Improved MCTS 방법이 평균 65.6점으로 가장 우수한 성능을 기록했습니다. 비교된 모델들(GPT-3.5-Turbo+CoT, GPT-3.5-Turbo+ToT, GPT-3.5-Turbo+ReST-MCTS)의 평균 점수는 61.06에서 62.31까지였으며, 개선된 MCTS 방법이 각 모델에 비해 뛰어난 효율성을 보였습니다.



### Evaluating Computational Accuracy of Large Language Models in Numerical Reasoning Tasks for Healthcare Applications (https://arxiv.org/abs/2501.13936)
Comments:
          13 pages, 1 figure, 2 tables

- **What's New**: 이번 연구는 의료 부문에서 사용되는 대규모 언어 모델(Large Language Models, LLMs)의 숫자 추론 능력에 대한 최초의 평가를 수행했습니다. LLMs는 자연어 이해 및 생성에서의 놀라운 성능을 보여주지만, 높아진 절차적 요구에 맞춘 숫자 추론은 여전히 미흡하다는 점에 주목합니다. 이 연구가 다룬 1,000개의 숫자 문제를 통해 실제 의료 환경에서의 수치적 추론 능력을 탐색하게 되었습니다.

- **Technical Details**: 연구에서는 GPT-3 아키텍처를 기반으로 한 정제된 LLM의 성능을 평가하기 위해 신중하게 수집된 데이터셋을 사용했습니다. 방법론에는 프롬프트 엔지니어링(prompt engineering), 사실 확인 파이프라인(fact-checking pipelines) 통합, 정규화 기법(regularization techniques) 적용이 포함되어 모델의 정확도 및 일반화 능력을 향상시켰습니다. 모델의 효율성을 평가하기 위해 precision, recall, F1-score와 같은 주요 메트릭(metrics)을 사용했습니다.

- **Performance Highlights**: 결과적으로 LLM은 84.10%의 전체 정확도를 기록했으며, 간단한 숫자 작업에서는 향상된 성능을 보였지만, 다단계 추론(multi-step reasoning)에서는 어려움이 있었습니다. 사실 확인 파이프라인의 통합이 정확도를 11% 향상시켜 검증 메커니즘의 중요성을 강조했습니다. 이 연구는 의료 분야에서 숫자 추론에 대한 LLM의 잠재력을 부각시키며, 임상 환경에서의 중요한 의사 결정 지원을 위한 LLM의 추가 개선 방향 또한 제시합니다.



### Do LLMs Provide Consistent Answers to Health-Related Questions across Languages? (https://arxiv.org/abs/2501.14719)
Comments:
          9 pages. Short paper appeared at 47th European Conference on Information Retrieval (ECIR 2025)

- **What's New**: 이번 연구는 다양한 언어에서 건강관련 질문에 대한 LLM(대형 언어 모델)의 응답 일관성을 분석하였습니다. 영어, 독일어, 터키어, 중국어의 다국적 맥락에서 LLM의 답변을 검토하고, 언어에 따라 다양한 질병 유형별로 분류된 건강 관련 질문 데이터를 확장했습니다. 이 연구의 주요 기여는 다국어 건강 질의 데이터셋과 이질적 언어 간의 비교를 가능하게 하는 새로운 프롬프트 기반 평가 작업 흐름을 도입한 것입니다.

- **Technical Details**: 연구에서는 NER(명명 개체 인식)을 사용하여 샘플을 질병별로 분류하고, 개편된 HealthFC 데이터셋을 사용해 LLM에 질병 분류에 따라 프롬프트를 제공하였습니다. 또한, 원래 영어와 독일어로 제공되던 데이터를 터키어와 중국어로 번역하고, 이를 통해 언어 간에 응답 일관성을 평가하기 위한 프롬프트 기반 평가 프레임워크가 개발되었습니다. 이러한 방법론은 LLM의 여러 언어에서의 응답을 심층적으로 분석할 수 있도록 하였습니다.

- **Performance Highlights**: 이 연구에서 발생한 주요 결과 중 하나는 LLM의 다양한 언어에서의 일관성 부족으로 인해 정확하지 않은 의료 정보를 퍼뜨릴 위험이 있다는 것입니다. 파싱(Parsing)된 응답 간의 비교를 통해 일관성의 측정을 시도했으며, Kappa Scores를 통해 응답의 일관성 정도를 평가하였습니다. 특히, 터키어(TR)와 중국어(ZH)에서 LLM과 인간 평가자 간의 상당한 일치도가 발견되었고, 독일어(DE)에서도 중간 정도의 일치가 관찰되었습니다.



### An Attentive Graph Agent for Topology-Adaptive Cyber Defenc (https://arxiv.org/abs/2501.14700)
- **What's New**: 사이버 공격이 더 복잡해지고 예측할 수 없게 되면서, 적응형 자율 방어 시스템의 개발이 사이버 보안 연구의 주요 초점이 되고 있습니다. 이 논문에서는 Graph Neural Networks (GNNs)와 Graph Attention Networks (GATs)를 활용하여 컴퓨터 네트워크의 고유 그래프 구조를 통해 자율 방어 정책의 효과성을 높이는 새로운 접근법을 제시합니다. 이를 통해 트레이닝 중 본 적이 없는 다양한 네트워크 구성에서도 잘 작동할 수 있도록 일반화 능력을 향상시킬 수 있습니다.

- **Technical Details**: CybORG 시뮬레이터를 기반으로 한 새로운 환경에서, 네트워크 상태를 방향 그래프로 인코딩하여 실시간 변화하는 네트워크 토폴로지에 적응할 수 있는 정책을 개발했습니다. GAT 아키텍처를 적용하여 노드, 엣지, 글로벌 기능을 처리하고, 이 출력을 강화 학습의 정책 기울기 방법에 맞게 조정했습니다. 이 접근법은 기존의 평탄한 상태 관찰 방식에 비해 여러 가지 장점이 있으며, 저수준 그래프 관찰을 통해 조작 가능한 방어 정책을 구축할 수 있게 해줍니다.

- **Performance Highlights**: 우리의 훈련된 정책은 다양한 크기의 네트워크에서 평가되었으며, 동일한 서브네트워크 구조를 가진 네트워크에서도 잘 동작함을 보여주었습니다. 특히, 기존의 정책과 비교했을 때, GAT 기반 정책은 다이나믹 연결 변화에 대한 적응력과 해석 가능성을 높임으로써 더 강력하고 유연한 사이버 방어 시스템을 개발하는데 기여합니다. 이 연구는 실제 네트워크 보안 문제에 대한 방어 능력을 크게 개선할 수 있는 가능성을 보여 줍니다.



### Towards Automated Self-Supervised Learning for Truly Unsupervised Graph Anomaly Detection (https://arxiv.org/abs/2501.14694)
Comments:
          Manuscript submitted to Data Mining and Knowledge Discovery in May 2024 for possible publication. This is the revised version submitted in January 2025

- **What's New**: 이 논문은 Self-supervised Learning (SSL) 기법이 그래프 이상 탐지에 효과적이나, 성능에 큰 영향을 미치는 여러 요인들(SSL 전략, 하이퍼파라미터 튜닝, 전략 조합 가중치 할당)을 보완하기 위한 방법을 제시합니다. 특히, Label Information Leakage 문제를 지적하며, 이를 해결하기 위해 내부 평가 전략을 활용하여 하이퍼파라미터를 선택하는 방안을 제안합니다. 이는 기존 방법들과는 달리, 레이블 정보를 사용하지 않고도 SSL 최적화를 이룰 수 있는 방안으로, 실용적인 그래프 이상 탐지를 가능하게 합니다.

- **Technical Details**: 이 논문은 정적인 속성 그래프에서의 노드 이상 탐지에 초점을 맞춥니다. 연구에서는 최근 발표된 SSL 기반 그래프 이상 탐지 알고리즘 10개를 다양한 벤치마크 데이터셋에서 실험하여 하이퍼파라미터 선택과 우리의 제안한 전략의 효과를 입증합니다. 또한, 모듈 간 상호작용이 강조되며, 데이터 증강 기법, 하이퍼파라미터 선택 및 다양한 SSL 전략들을 결합하는 방법에 대한 문제를 다룹니다.

- **Performance Highlights**: 연구 결과, AutoGAD 접근방식이 기존의 수동적인 방법들보다 월등한 성능을 보이며, 서로 다른 데이터셋에 특화된 최적화가 필요한 것을 보여주었습니다. 이 논문에서는 SSL 기반의 그래프 이상 탐지 방법들이 자주 겪는 성능의 과대 추정 문제를 해결하는 중요한 단서를 제공하며, 자동화된 과정이 실제 환경에서 유용하게 활용될 수 있음을 입증합니다. 실험을 통해 제안한 방법이 다양한 데이터셋에서 기존의 높은 성능을 가진 알고리즘에 필적하거나 이를 초월하는 결과를 나타냈습니다.



### Rethinking Table Instruction Tuning (https://arxiv.org/abs/2501.14693)
- **What's New**: 최근 테이블 이해(table understanding) 분야에서는 테이블 관련 작업을 위해 대규모 언어 모델(LLM)을 instruction-tuning하는 발전이 있었습니다. 그러나 기존 연구에서는 하이퍼파라미터 선택의 영향을 간과하고, 도메인 외(out-of-domain) 테이블 이해 능력에 대한 종합적인 평가가 부족했습니다. 본 연구에서는 이러한 능력을 평가하고, 하이퍼파라미터가 테이블 LLM의 성능에 미치는 영향을 체계적으로 분석하여, 새로운 모델 TAMA를 도입했습니다.

- **Technical Details**: 기존의 테이블 LLM이 도메인 외 테이블 이해 능력 및 일반적인 능력에서 큰 성능 하락을 보이는 것을 발견했습니다. 특히, 학습률(learning rate)과 훈련 인스턴스(instance)의 수와 같은 하이퍼파라미터 선택이 모델의 성능에 미치는 효과는 중요한 요소입니다. 본 연구의 분석에 따르면, 작은 학습률이 테이블 이해 능력을 향상시키면서도 일반 능력을 유지할 수 있음을 확인했습니다.

- **Performance Highlights**: TAMA 모델은 LLaMA 3.1 8B Instruct에서 instruction-tuning되어, 테이블 관련 작업에서 GPT-3.5 및 GPT-4의 성능에 필적하거나 이를 초과했습니다. 또한, TAMA는 강력한 도메인 외 테이블 이해 및 일반 능력을 유지하며, 데이터 주석 비용을 절감할 잠재력을 가지고 있음을 강조하였습니다.



### Approach to Designing CV Systems for Medical Applications: Data, Architecture and AI (https://arxiv.org/abs/2501.14689)
Comments:
          9 pages, 3 figures

- **What's New**: 이 논문은 fundus 이미지 분석을 위한 혁신적인 소프트웨어 시스템을 소개합니다. 전통적인 진단 예측 접근법과는 달리, 이 시스템은 fundus 구조의 정상 및 병리적 특징을 철저히 분석하여 의료 전문가에게 최종 결정권을 위임합니다. 이 연구는 객관적인 임상 분석의 필요성을 해결하고 임상 작업 흐름을 자동화 및 향상시키고자 하며, AI 모델을 통해 모듈식 분석 디자인을 제공합니다.

- **Technical Details**: EYAS는 AI 기반의 fundus 이미지 분석 시스템으로, 데이터 부족 문제를 해결하기 위해 전문 의료진과의 협업을 강조합니다. 시스템은 특정 작업을 위한 전문 모듈을 통합하여 생성된 중간 결과물을 임상의가 이해하고 검증할 수 있도록 설계되었습니다. 또한, 최신 딥러닝 기법과 기존의 컴퓨터 비전 알고리즘을 결합하여 fundus 구조에 대한 포괄적이고 세분화된 분석을 제공합니다.

- **Performance Highlights**: 연구 결과, EYAS의 접근 방식은 fundus 이미지 분석 혁신에 효과적이며, 다양한 의료 분야에서의 잠재적 응용성을 보여줍니다. 이러한 디자인 접근법은 의료 전문가의 실질적인 요구에 맞춰 AI 기술과의 간극을 줄이며, 안과 및 다른 의료 분야에서 AI의 의미 있는 채택을 촉진하는 것을 목표로 합니다.



### Decoding Generalization from Memorization in Deep Neural Networks (https://arxiv.org/abs/2501.14687)
- **What's New**: 이 논문은 Deep Learning의 overparameterized 모델이 어떻게 잘 일반화되는지를 탐구하며, 이러한 모델들이 기억(memorization) 능력을 갖고 있지만 여전히 잘 일반화될 수 있다는 새로운 증거를 제시합니다. 특히, 특정 층의 출력에서 정보가 추출 가능하다는 점을 강조하여, 메모리 효과가 있는 모델임에도 불구하고 좋은 일반화 성능을 나타낼 수 있음을 보여줍니다.

- **Technical Details**: 저자들은 여러 Deep Neural Networks의 층별 출력에서 클래스 조건 하의 서브스페이스 조직을 연구하고, Principal Components Analysis (PCA)를 사용하여 이들 서브스페이스를 추정합니다. 제안된 분류기(classifier)는 입력 데이터 포인트의 출력 벡터와 해당 클래스 서브스페이스 간의 각도를 측정하여 클래스를 결정하며, 이를 통해 네트워크가 좋은 일반화 가능성을 가지는지 확인합니다. 이 접근 방식은 corrupted training data에 대해서도 적합합니다.

- **Performance Highlights**: 실험 결과, label noise로 손상된 데이터로 훈련된 모델임에도 불구하고, 단순한 분류기를 통해 높은 테스트 정확도를 달성할 수 있음을 발견했습니다. 또한, 훈련된 클래스 레이블이 사후에 알려지면 일관되게 더 나은 일반화 성능을 발휘하는 분류기를 구축할 수 있음을 보여주었습니다. 이러한 발견은 메모리화와 일반화가 동시에 존재할 수 있다는 가능성을 시사합니다.



### Rethinking Foundation Models for Medical Image Classification through a Benchmark Study on MedMNIS (https://arxiv.org/abs/2501.14685)
Comments:
          submitted to MIDL2025

- **What's New**: 본 연구에서는 MedMNIST 데이터셋을 통해 의료 이미지 분류 작업에서 사용되는 다양한 foundation model의 성능을 비교합니다. 이 연구는 다양한 convolutional 및 Transformer 기반 모델을 포함하여 이전 연구에서 제시되지 않은 모델 선택을 통해 더 깊이 있는 통찰력을 제공합니다. 또한, 이미지 크기와 리사이징 전략의 효과에 대한 탐구도 포함되어 있어, 의료 이미지 분류 작업으로의 전이 가능성에 대한 새로운 인사이트를 제공합니다.

- **Technical Details**: 의료 이미지 분류 연구에서 사용된 foundation model은 CNN 기반 및 ViT 기반 아키텍처를 통해 수행되었습니다. CNN 모델은 VGG16, DenseNet-121 등 다양한 백본을 포함하며, ViT 모델은 최적화 전략에 따라 구분하여 4개의 모델이 활용되었습니다. 모델 평가를 위한 학습 과정에서는 15,000회의 반복 훈련이 진행되며, AdamW 옵티마이저가 사용되었습니다.

- **Performance Highlights**: 연구 결과, 다양한 기법으로 훈련된 foundation model들은 의료 이미지 분류 작업에서 뛰어난 잠재력을 보여주었습니다. 각 모델의 성능은 평균 ± 표준편차 방식으로 평가되어, 서로 다른 이미지 해상도에서 일관된 결과가 나타났습니다. 또한, 실험을 통해 제안된 다양한 리사이징 전략이 전반적인 분류 성능에 긍정적인 영향을 미쳤음을 확인하였습니다.



### Surface Vision Mamba: Leveraging Bidirectional State Space Model for Efficient Spherical Manifold Representation (https://arxiv.org/abs/2501.14679)
- **What's New**: 이 연구에서는 Attention 기반 방법들이 구형 피질 표면에서의 장기 의존성 모델링에 매우 우수한 성능을 보였음을 강조합니다. 그러나 긴 추론 시간과 높은 메모리 요구사항이 대규모 데이터셋에의 적용을 어렵게 만들고 있습니다. 이를 해결하기 위해, 우리는 Attention 없이 작동하는 'Vision Mamba (Vim)'를 제안하며, 이는 구형 다양체(spherical manifold)에서 데이터 분석을 위한 도메인 독립적 아키텍처입니다.

- **Technical Details**: 제안된 방법은 구형 데이터를 잘게 나눈 삼각형 패치(triangular patches)로 표현하여 표면 패치(surface patching)를 수행합니다. 'Surface Vision Mamba (SiM)'는 신생아 뇌의 피질 표면 메트릭을 이용한 여러 신경 발달 표현 회귀(task)에서 평가되었습니다. 이 방법은 Ico-4 그리드(partitioning) 하에서 Surface Vision Transformer (SiT)와 비교했을 때, 4.8배 빠른 추론 속도와 91.7% 낮은 메모리 소비를 달성했습니다.

- **Performance Highlights**: 실험적 결과는 SiM이 Attention 기반 및 GDL 모델을 모두 초월하는 성능을 보여준다고 강조합니다. 또한 민감도 분석(sensitivity analysis)을 통해 SiM이 미세한 인지 발달 패턴을 식별할 수 있는 잠재력을 가지고 있음을 확인했습니다. 이 코드는 해당 URL에서 제공됩니다.



### A Predictive Approach for Enhancing Accuracy in Remote Robotic Surgery Using Informer Mod (https://arxiv.org/abs/2501.14678)
- **What's New**: 이 연구에서는 Tactile Internet (TI) 환경에서 원거리 로봇 수술의 성공을 위해 로봇 팔의 정확하고 실시간 위치 추정을 위한 새로운 예측 모델을 제안합니다. Transformer 기반의 Informer 프레임워크를 활용하여 효율적인 위치 추정을 구현하며, 4-State Hidden Markov Model (4-State HMM)을 결합하여 현실적인 패킷 손실 시나리오를 시뮬레이션합니다. 이 접근법은 네트워크 지연, 지터(jitter), 패킷 손실 등의 문제를 해결하여 원거리 수술 응용 프로그램에서 안정적인 작업을 가능하게 합니다.

- **Technical Details**: 제안된 방법은 Informer 모델에 다양한 제약 조건을 포함한 최적화 문제를 통합하여 에너지 효율성, 부드러움(smoothness), 강인성(robustness)을 고려한 교육 과정을 통해 훈련합니다. Informer 모델은 ProbSparse attention, attention distilling, 및 생성 스타일 디코더(generative-style decoder)의 특징을 활용하여 위치 중요 기능에 집중하며, O(L log L)의 낮은 계산 복잡도를 유지합니다. 이 방법은 JIGSAWS 데이터셋을 사용하여 평가되었으며, 다양한 네트워크 시나리오에서 90% 이상의 예측 정확도를 달성합니다.

- **Performance Highlights**: 색다른 심층 학습 모델인 TCN, RNN, LSTM과 비교 시, Informer 프레임워크가 위치 예측을 처리하고 실시간 요건을 충족하는 데 뛰어난 성능을 보임을 강조합니다. 제안된 모델은 네트워크의 다양한 상황에서도 PSM의 정확하고 효율적인 위치 추정을 구현할 수 있어 원거리 로봇 수술에 적합합니다. 이를 통해 수술의 효율성과 안정성을 높이며, 민감한 원거리 수술의 실현 가능성을 높이는 중요한 기반을 제공합니다.



### State Space Models for Extractive Summarization in Low Resource Scenarios (https://arxiv.org/abs/2501.14673)
- **What's New**: 이번 논문에서는 MPoincareSum 방법을 제안합니다. 이는 최근의 연구 동향을 반영하여 저자원이 부족한 환경에서의 추출적 요약(extractive summarization) 성과를 향상시키기 위해 개발되었습니다. 기존 연구들에 비해 문장 관련성을 예측하고 요약을 최적화하는 데 중점을 두었습니다.

- **Technical Details**: MPoincareSum 방법은 Mamba 상태 공간 모델(state space model)을 활용하여 리뷰와 문장의 의미(semantics)를 생성합니다. 이후 Poincare 압축(Poincare compression)을 통해 가장 의미 있는 특징(features)을 선택하고, 선형 레이어(linear layer)를 적용하여 문장의 관련성을 예측합니다. 최종적으로, 관련 문장을 패러프레이즈(paraphrase)하여 요약을 생성합니다.

- **Performance Highlights**: 아마존 리뷰 데이터셋을 사용하여 MPoincareSum의 효과를 평가하였습니다. 실험 결과, 이 방법이 기존의 여러 접근 방식보다 성능이 우수하다는 것을 ROUGE 점수(ROUGE scores)를 통해 입증하였습니다. MPoincareSum은 저자원 환경에서도 뛰어난 요약 결과를 제공합니다.



### Neural-Symbolic Message Passing with Dynamic Pruning (https://arxiv.org/abs/2501.14661)
Comments:
          19 pages, 5 figures, 16 tables

- **What's New**: 본 논문에서는 Neural-Symbolic Message Passing (NSMP) 프레임워크를 제안하며, 그것은 사전 훈련된 신경 링크 예측기를 기반으로 합니다. NSMP는 1단계 추론을 수행하기 위해 상징적 추론과 퍼지 논리를 통합하여 복잡한 쿼리에 대해 훈련 없이도 일반화할 수 있습니다. 또한, NSMP는 변동 노드 간의 노이즈 메시지를 필터링하는 동적 가지치기 전략을 도입하여 해석 가능한 답변을 제공합니다.

- **Technical Details**: NSMP 프레임워크는 메시지 전달 방식으로 작동하며, 변동 노드의 중간 상태를 계산하는 데 신경 및 상징적 추론을 통합합니다. 이를 통해 변동 노드 간의 메시지를 효과적으로 집계하고 업데이트하여 퍼지 벡터로 표현된 중간 상태가 해석될 수 있도록 합니다. 특히, NSMP는 필요 없는 노이즈 메시지를 동적으로 제거하여 성능을 개선하는 가지치기 전략을 채택합니다.

- **Performance Highlights**: 실험 결과에 따르면, NSMP는 다양한 쿼리 유형에 대해 이전의 최첨단(neural-symbolic) 모델보다 2배에서 150배 이상 빠른 추론 시간을 보이며, 특히 부정 쿼리에서 유의미한 성능 향상을 이룹니다. 이러한 결과는 NSMP가 복잡한 쿼리 데이터에 대한 훈련 없이도 강력한 성과를 달성할 수 있음을 보여줍니다.



### MedAgentBench: Dataset for Benchmarking LLMs as Agents in Medical Applications (https://arxiv.org/abs/2501.14654)
- **What's New**: 최근 대형 언어 모델(LLMs)은 챗봇의 전통적인 역할을 넘어 에이전트로서의 기능에서 유의미한 발전을 보여주었습니다. 이러한 에이전트는 고수준의 작업을 수행하기 위해 계획 및 도구 활용 능력을 적극적으로 활용할 수 있습니다. 그러나 의료 분야에서 LLM의 에이전트 기능을 평가할 수 있는 표준화된 데이터셋은 부족한 상황입니다.

- **Technical Details**: 이 논문에서는 MedAgentBench라는 평가 도구를 도입했습니다. MedAgentBench는 10개 카테고리에서 임상의에 의해 작성된 100개의 환자 맞춤 임상 과제를 포함하고 있으며, 700,000개 이상의 데이터 요소가 포함된 100명의 현실적인 환자 프로필을 제공합니다. 이 환경은 현대 전자 의료 기록 시스템에서 사용되는 표준 API와 통신 인프라를 바탕으로 구축, 실시간 EMR 시스템으로 쉽게 이식할 수 있는 기능을 갖추고 있습니다.

- **Performance Highlights**: 현재 가장 발전된 모델인 GPT-4o는 72%의 성공률을 기록했으나, 에이전트 기능 향상을 위한 많은 개선 여지가 남아 있습니다. 또한, 작업 카테고리에 따라 성능의 변동이 크며, MedAgentBench는 이러한 변화를 명확히 하고 있어 모델 개발자들에게 유용한 프레임워크를 제공합니다. 이 프로젝트는 공개적으로 제공되어 의료 도메인 내 LLM의 에이전트 기능 상승을 위한 중요한 초석이 될 것입니다.



### Federated Domain Generalization with Data-free On-server Gradient Matching (https://arxiv.org/abs/2501.14653)
Comments:
          26 pages, 15 figures, ICLR

- **What's New**: 이번 논문에서는 분산된 도메인에서 도메인 불변 모델을 학습할 수 있는 새로운 접근 방식인 FedOMG (Federated Learning via On-server Matching Gradient)를 제안합니다. 기존의 Federated Domain Generalization (FDG) 문제를 해결하며, 기존의 통신 비용 없이도 중앙 서버에서 분산 모델의 특성을 집계할 수 있는 방법을 제공합니다. FedOMG는 로컬 그래디언트를 활용하여 도메인 불변 그래디언트 방향을 찾는 방식을 채택합니다.

- **Technical Details**: FedOMG는 Gradient Inner Product (GIP) 최적화 문제에서 영감을 받아 설계되었습니다. 이는 각 도메인에서 유도된 그래디언트의 높은 상관관계를 기반으로 하며, 두 가지 주요 한계를 극복합니다: 1) GIP의 직접적인 최소화는 모델 파라미터의 2차 미분을 필요로 하여 계산 비용이 많이 드는 점과, 2) 클라이언트 모델의 지속적 전송이 필요하여 통신 오버헤드가 과도해지는 문제입니다. FedOMG는 이러한 문제를 간접 최적화 방식을 통해 해결하며, 효율적인 볼록 최적화 수식을 도입하여 서버 측 최적화를 간소화합니다.

- **Performance Highlights**: 실험 결과, FedOMG는 MNIST, EMNIST, CIFAR-10, CIFAR-100과 같은 네 가지 FL 벤치마크 데이터셋과 PACS, VLCS, OfficeHome의 세 가지 FDG 벤치마크 데이터셋에서 최신 성능 기준(SOTA)을 능가했습니다. FedOMG는 IID 및 non-IID 데이터셋과 OOD 일반화 상황에서 기존 FL 방법보다 우수한 성능을 보였습니다. 이러한 우수함은 FedOMG가 통신 효율성과 개인 정보 보호를 동시에 유지할 수 있도록 합니다.



### Whisper D-SGD: Correlated Noise Across Agents for Differentially Private Decentralized Learning (https://arxiv.org/abs/2501.14644)
Comments:
          6 pages, 3 figures, preprint

- **What's New**: 이번 논문은 Whisper D-SGD라는 새로운 covariance 기반 접근 방식을 소개하며, 이는 분산된 에이전트 간에 상관관계가 있는 프라이버시 노이즈를 생성하여 기존 기법들을 통합합니다. 이 기법은 에이전트들이 로컬 모델을 업데이트하고 직접 이웃들과 혼합할 수 있도록 하여, 분산 학습에서의 개인 정보 보호를 강화합니다. Whisper D-SGD는 네트워크 토폴로지와 혼합 가중치를 활용하여 노이즈를 최적화함으로써, 기존의 방식과 비교해 더욱 효과적인 노이즈 캔슬링을 달성합니다.

- **Technical Details**: 논문에서는 n개의 에이전트가 각자의 로컬 데이터셋을 보유하고 있으며, 공동의 기계 학습 모델 파라미터를 학습하기 위해 글로벌 목적을 최소화한다는 내용을 다룹니다. 이 과정에서 자주 사용되는 알고리즘인 Decentralized Stochastic Gradient Descent (D-SGD)를 활용하여 에이전트들이 이웃과 파라미터를 교환하고 가중 평균을 계산하는 두 단계의 과정을 제시합니다. 이 과정에서 사용되는 혼합 매트릭스 W는 에이전트 간의 통신을 수립하는 데 핵심적인 역할을 합니다.

- **Performance Highlights**: 실험 결과 Whisper D-SGD는 기존의 pairwise-correlation 방식보다 더 많은 노이즈를 캔슬하며, CDP-LDP 간의 격차를 줄이고 동일한 프라이버시 보장 하에서도 모델 성능을 향상시킵니다. 특히 네트워크 연결성이 약한 환경에서도 우수한 프라이버시-효용(trade-off)을 달성하는 것으로 나타났습니다. 이러한 결과는 Whisper D-SGD가 프라이버시 보호와 모델 성능 간의 균형을 잘 맞출 수 있는 가능성을 보여줍니다.



### ACT-JEPA: Joint-Embedding Predictive Architecture Improves Policy Representation Learning (https://arxiv.org/abs/2501.14622)
- **What's New**: 이번 연구에서는 모방 학습(imitation learning)과 자기 지도 학습(self-supervised learning)의 통합을 통해 정책 표현(policy representations)을 향상시키는 새로운 아키텍처인 ACT-JEPA를 제안합니다. 기존의 모방 학습은 전문가의 시연에 의존하여 고비용이 소요되며, 세계 모델(world model)이 잘 발전되지 않는 한계가 있습니다. ACT-JEPA는 다양한 비표시 데이터로부터 학습할 수 있는 가능성을 제공하여 이러한 문제를 해결하고자 하였습니다.

- **Technical Details**: ACT-JEPA는 정책(policy)을 학습하기 위해 두 가지 주요 목표를 설정합니다: (1) 행동 시퀀스(action sequences)의 예측과 (2) 추상 관찰 시퀀스(abstract observation sequences)의 예측입니다. 첫 번째 목표는 행동 청킹(action chunking)을 활용하여 행동 예측을 개선하고 오류를 줄입니다. 두 번째 목표는 추상 관찰 시퀀스를 예측하여 청킹의 개념을 확장함으로써 모델이 불필요한 세부 사항을 필터링할 수 있게 합니다.

- **Performance Highlights**: 실험 결과, ACT-JEPA는 시간적 환경 역학을 학습함으로써 표현의 질을 개선하는 것으로 나타났습니다. 또한, 추상 관찰 시퀀스를 예측하는 능력 덕분에 행동 시퀀스 예측에 효과적으로 일반화되는 표현을 생성합니다. ACT-JEPA는 여러 의사결정 태스크에서 기존 기준 모델들과 동등한 성능을 보였습니다.



### Leveraging Spatial Cues from Cochlear Implant Microphones to Efficiently Enhance Speech Separation in Real-World Listening Scenes (https://arxiv.org/abs/2501.14610)
Comments:
          10 pages, 5 figures

- **What's New**: 이 논문은 단일 채널 건조 음성 혼합물에 대한 음성 분리 기술의 발전을 다루고 있습니다. 특히, 실제 환경에서의 음성이 혼합된 상황에서 어음 분리의 성능을 향상시키기 위한 공간 신호(spatial cues)의 중요성을 강조하고 있습니다. 저자는 자연환경에서의 소음과 잔향이 어음 분리의 성능에 미치는 영향을 정량적으로 분석하고 있습니다.

- **Technical Details**: 이 연구에서는 두 명의 화자가 있는 음성 혼합물의 파형을 추정하는 음성 분리 작업을 수행했습니다. WSJ0-2mix 데이터셋을 사용하여 잔향이 포함된 공간적 현실감 있는 음성 혼합을 시뮬레이션했습니다. 연구에서 제안된 모델은 시간이 도메인(time-domain)에서 작동하며, 효율적인 음성 분리를 위한 고성능의 구조를 갖추고 있습니다.

- **Performance Highlights**: 실험 결과는 실제 환경이 음성 분리 성능에 미치는 부정적인 영향을 보여주며, 공간 신호(implicit 또는 explicit)가 어음 분리를 상당히 개선할 수 있음을 입증합니다. 특히, 단일 청각 임플란트를 사용하는 경우에도 명시적 공간 신호가 중요한 이점을 제공함을 발견했습니다. 이 연구는 CI와 같은 보조 청각 장치에서의 음성 처리 성능을 향상시키기 위한 개발 방향을 제시합니다.



### Age and Power Minimization via Meta-Deep Reinforcement Learning in UAV Networks (https://arxiv.org/abs/2501.14603)
Comments:
          10 pages, 8 figures

- **What's New**: 이 연구는 비행 드론(UAV)을 활용한 IoT 네트워크에서의 데이터 수집을 지원하는 전력 제한 환경을 다룹니다. 특히 변동하는 Age-of-Information (AoI)와 전송 전력을 최소화하기 위한 UAV의 비행 경로 및 스케줄링 정책을 최적화하는 데 집중하고 있습니다. 제안된 메타 딥 강화 학습(meta-deep reinforcement learning) 접근법은 DQNs와 MAML을 통합하여, 빠른 수렴 속도와 변화하는 목표에 대한 강력한 적응성을 제공합니다.

- **Technical Details**: 이 논문에서는 DQN(Deep Q-Network)과 MAML(Model-Agnostic Meta-Learning)을 결합하여 UAV의 비행 경로 및 스케줄링 정책을 최적화합니다. DQN은 상태에서 최적의 결정을 내리는 반면, MAML은 다양한 목표 함수들에 대한 확장성을 제공합니다. 이 접근법은 하이퍼파라미터 조정을 통해 UAV의 동적 환경에서의 최적화 문제에 빠르게 적응할 수 있게 합니다.

- **Performance Highlights**: 제안된 메타 딥 강화 학습 알고리즘은 MAML이 없는 기존 강화 학습 모델보다 새로운 목표에 빠르게 적응하며, 경험 샷 수가 적고, AoI와 전송 전력을 전반적으로 최적화하는 성능을 보였습니다. 특히, 실험 결과는 UAV 지원 IoT 네트워크에서 실시간 데이터 수집과 전력 효율성을 모두 만족할 수 있는 가능성을 보여주고 있습니다.



### ZETA: Leveraging Z-order Curves for Efficient Top-k Attention (https://arxiv.org/abs/2501.14577)
Comments:
          25 pages, 4 figures, accepted in International Conference on Learning Representations (ICLR) 2025

- **What's New**: 최근 Transformer는 시퀀스 모델링 아키텍처의 기본 구성 요소로 자리 잡았습니다. 하지만 self-attention의 메모리 및 계산 비용이 시퀀스 길이 $N$에 따라 제곱으로 증가하여 긴 시퀀스의 처리에 제한이 있습니다. 본 연구에서는 ZETA라는 새로운 접근 방식을 제안하여, 과거 토큰을 효율적으로 병렬 쿼리할 수 있는 방법을 제시합니다. ZETA는 $k$개의 가장 관련 있는 토큰을 선택하되 훈련 효율성을 극대화합니다.

- **Technical Details**: ZETA는 	extbf{Z}-Order 곡선을 활용하여 효율적인 top-$k$ attention을 구현합니다. 핵심적으로 키(key)와 쿼리(query)의 차원을 줄여서 상대 거리 정보를 보존할 수 있도록 합니다. 이를 통해 저차원 공간에서 키와 쿼리를 일차원으로 매핑하며, 이로 인해 병렬 정렬이 가능해져 top-$k$ 토큰 선택의 효율성을 크게 향상시킵니다. 이론적으로도 차원 저하에 따른 Trade-off를 명확히 보여줍니다.

- **Performance Highlights**: 실험 결과, ZETA는 synthetic 	extsc{Multi-Query Associative Recall} 작업에서 표준 attention과 유사한 성능을 보여주며, 	extsc{Long Range Arena}와 	extsc{WikiText-103} 언어 모델링 작업에서는 기존의 attention 및 변형 모델보다 월등한 성능을 방출합니다. 이러한 성능의 향상은 ZETA의 병렬 처리 능력과 효율적인 토큰 검색 방법에 기인합니다.



### Leveraging ChatGPT's Multimodal Vision Capabilities to Rank Satellite Images by Poverty Level: Advancing Tools for Social Science Research (https://arxiv.org/abs/2501.14546)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)이 위성 이미지를 사용해 마을 수준의 빈곤을 예측하는 새로운 방법을 탐색합니다. LLMs는 본래 자연어 이해를 위해 개발되었으나, 다양한 도메인에서의 적응력이 뛰어나므로 지리적 분석과 같은 다중 모드 작업에 혁신적인 활용이 가능해졌습니다. 연구 결과, ChatGPT는 전문가와 유사한 정확도로 빈곤 수준에 따라 위성 이미지를 분류할 수 있음을 보였습니다.

- **Technical Details**: 본 연구는 2015/2016 탄자니아 인구 및 건강 조사(DHS) 데이터를 기반으로 하여, 각 지역의 가구 수를 수집하고 이를 빈곤의 지표로 사용했습니다. Wealth index(재산 지수)는 가구의 자산 소유 및 기타 사회 경제적 요소를 고려하여 계산되었으며, 이를 통해 빈곤 상태를 평가하는 데 기여했습니다. 연구는 OpenAI의 GPT-4o 모델을 활용하여 위성 이미지의 쌍을 비교하고, 인프라의 질과 가시적인 주변 환경을 평가하여 부유한 위치를 결정했습니다.

- **Performance Highlights**: 이 연구에서는 ChatGPT가 빈곤을 기반으로 한 이미지 순위를 전문가와 비슷한 수준으로 수행할 수 있음을 보여주었습니다. 비교 결과는 LLMs를 사용하여 빈곤 평가에 있어 전통적인 방법과 유사한 신뢰성을 가지며, 대규모 및 비용 효율적인 빈곤 모니터링의 가능성을 제시합니다. LLMs의 사용은 데이터의 복잡성을 해소하고 해석 가능한 통찰력을 제공하는 데 기여할 수 있어 사회 경제적 연구에 중요한 기초를 마련했습니다.



### Distributed Conformal Prediction via Message Passing (https://arxiv.org/abs/2501.14544)
Comments:
          16 pages, 11 figures, submitted for posssible publication

- **What's New**: 이 논문에서는 안전이 중요한 분야에서 신뢰할 수 있는 추론을 보장하기 위한 전이 학습 모델의 사후 보정(post-hoc calibration) 문제를 다룹니다. 특히, Conformal Prediction (CP)을 기반으로 한 두 가지 분산 접근 방식인 Q-DCP(quantile-based distributed conformal prediction)와 H-DCP(histogram-based distributed conformal prediction)를 제안합니다. 이러한 접근 방식은 서로 다른 그래프 토폴로지에서 제한된 보정 데이터를 사용하여 효과적인 예측 세트를 생성할 수 있도록 합니다.

- **Technical Details**: Q-DCP는 분산 quantile 회귀(distributed quantile regression)에 맞춤형 평활화(smoothing) 및 정규화(regularization) 항을 통합하여 수렴(convergence)을 가속화합니다. H-DCP는 합의 기반(histogram estimation) 방법을 사용하여 다수의 장치에서 수집된 로컬 보정 데이터로부터 전 세계적으로 histogram을 추정합니다. 논문에서는 통신 오버헤드 및 하이퍼파라미터 조정 요구사항과 같은 여러 트레이드오프를 실험적으로 평가합니다.

- **Performance Highlights**: Q-DCP와 H-DCP는 각각 고유한 장단점을 가지며, H-DCP는 하이퍼파라미터에 의존하지 않고 안정적인 커버리지(coverage) 보장을 제공합니다. 그러나 H-DCP는 Q-DCP보다 더 많은 통신 부하를 요구합니다. 이 연구 결과는 분산 환경에서 신뢰할 수 있는 예측을 위한 새로운 방향과 가능성을 제시합니다.



### ABPT: Amended Backpropagation through Time with Partially Differentiable Rewards (https://arxiv.org/abs/2501.14513)
- **What's New**: 이 논문에서는 기존의 gradient bias 문제를 해결하기 위해 Amended Backpropagation-through-Time (ABPT)라는 새로운 접근 방식을 제안합니다. ABPT는 0-step 반환과 N-step 반환을 결합하여 reward의 비가산성에 의해 발생하는 gradient bias를 완화함으로써 정책 학습의 효율성을 향상시킵니다. 또한, entropy 정규화 및 상태 초기화 메커니즘을 도입하여 훈련 중 탐색을 장려합니다.

- **Technical Details**: 강화학습의 목표는 주어진 상태에서 누적 보상을 극대화하는 확률적 정책을 찾는 것입니다. ABPT에서는 기존의 actor-critic 방법론을 사용하며, actor와 critic 모두 신경망에 의해 근사됩니다. 정책 경량화 과정에서 ABPT는 0-step 반환을 활용하여 가치 그래디언트를 균형 있게 조정하고, replay buffer를 통해 상태 경험을 저장하여 샘플링 효율성을 높입니다.

- **Performance Highlights**: 실험 결과는 ABPT가 기존의 학습 알고리즘보다 빠른 수렴 속도와 더 높은 최종 보상을 달성함을 보여줍니다. 특히, 비가산성 보상을 포함한 quadrotor 작업에서 ABPT의 우수성이 입증되었습니다. 또한, ABPT는 다양한 학습 속도와 보상 구조에서도 강인성을 보이며, 실제 quadrotor 작업에서 효과적인 학습 성과를 달성했습니다.



### RealCritic: Towards Effectiveness-Driven Evaluation of Language Model Critiques (https://arxiv.org/abs/2501.14492)
- **What's New**: 본 연구는 기존의 평가 방식을 개선하기 위해 LLM의 비판 능력을 평가하는 새로운 기준을 도입합니다. 기존 벤치마크가 오픈 루프(open-loop) 방식이었다면, 우리는 피드백을 통해 수정되는 폐쇄 루프(closed-loop) 방식을 채택하여 LLM의 비판 품질을 더 신뢰성 있게 평가합니다. 이 새로운 기준은 자기 비판(self-critique), 상호 비판(cross-critique), 반복 비판(iterative critique) 기능을 포함하여 고급 추론 모델의 능력을 구분하는 데 필수적입니다.

- **Technical Details**: 평가 프레임워크는 여러 주요 개념으로 구성되어 있습니다. 폐쇄 루프 방법론을 통해 LLM의 비판 능력을 테스트하게 되며, 비판이 적용된 후 생성된 솔루션의 정확도를 통해 비판의 품질을 평가합니다. 또한 두 가지 주요 축을 통해 자기 비판과 상호 비판의 차이를 구분하고, 단일 라운드뿐만 아니라 반복적인 비판 과정을 고려하여 모델의 장기적인 추론 능력을 평가합니다.

- **Performance Highlights**: 실험 결과, 고전적인 LLM은 고급 추론 기반 모델인 o1-mini에 비해 모든 비판 시나리오에서 눈에 띄게 뒤처지는 것으로 나타났습니다. 또한 자기 비판 및 반복 비판 설정에서 고전 LLM이 기준 성능에 비해 더 낮은 성능을 보이는 경우도 발견되었습니다. 이러한 결과는 우리의 벤치마크가 LLM의 개선 방향을 제시할 수 있는 중요한 자원이 될 것이라는 점을 강조합니다.



### Registration of Longitudinal Liver Examinations for Tumor Progress Assessmen (https://arxiv.org/abs/2501.14483)
- **What's New**: 이번 연구는 간 CT 검사에서 암 진행 상황을 평가하는 새로운 방법을 제안합니다. 기존의 등록 방법들이 주로 구조적 특성에 의존하여 종양 영역을 왜곡하는 문제점을 해결하기 위해, 본 논문은 해부학적 및 기하학적 정보를 기반으로 하는 새로운 등록 프레임워크를 제시합니다. 이는 단순히 시각적 특징 대신 간 세분화에서의 정보를 활용하여, 종양의 외관을 보다 정확하게 보존할 수 있도록 합니다.

- **Technical Details**: 제안된 방법론은 동일한 환자의 각기 다른 시점에서 촬영된 이미지를 정렬하는 데 중점을 둡니다. 간 세분화 도구를 사용하여, 이미지 A와 B의 세분화 마스크를 정의하고 이를 기반으로 변환 함수를 도출합니다. 제안된 프레임워크는 세분화 맵에서 변위 필드(Displacement Field)를 생성하여, 정렬된 세분화 마스크를 얻기 위해 공간 변환 네트워크(Spatial Transformer Network)를 활용합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 기존의 등록 기술보다 더 부드러운 변형을 제공하면서도 종양의 부하를 잘 보존함을 보여줍니다. 본 연구의 방법은 317명의 환자에게서 훈련하고 53명의 환자에서 테스트하여, 모든 실험 결과에서 높은 정확성과 효과성을 입증했습니다. 이러한 성과는 간 CT 스캔의 종양 모니터링 과정을 개선하는 데 기여할 것입니다.



### The Pseudo-Dimension of Contracts (https://arxiv.org/abs/2501.14474)
- **What's New**: 본 논문은 알고리즘적 계약 설계를 다루며, 에이전트의 유형이 알려지지 않은 분포에서 샘플링되는 설정을 중점적으로 분석합니다. 저자들은 샘플 에이전트 유형으로부터 근사 최적 계약(near-optimal contract)을 학습하기 위한 오프라인 학습 프레임워크를 제안합니다. 특히, 통계적 학습 이론의 'pseudo-dimension' 개념을 도구로 활용하여 계약 설계에서의 단순성(simplicity)과 최적성(optimality) 간의 균형을 새로운 시각으로 접근합니다. 이를 통해 저자들은 다양한 계약 유형에 대한 샘플 복잡성을 보장하는 알고리즘을 제시합니다.

- **Technical Details**: 이 논문은 선형 계약(linear contract), 유계 계약(bounded contract), 비유계 계약(unbounded contract) 세 가지 계약 유형을 분석합니다. 각 계약 클래스에 대해 pseudo-dimension과 표현 오류(representation error) 사이의 상충 관계를 규명하고 샘플 복잡성(sample complexity) 보장을 제공합니다. 특히, 선형 계약의 pseudo-dimension은 Θ(log n)으로 나타내며, 계약 유형으로부터의 에이전트 샘플을 통한 학습 방법을 제안합니다. 또한, 제안된 알고리즘은 오프라인 학습 설정에서 효율성을 보장하는 데 주력합니다.

- **Performance Highlights**: 제안된 방법론은 높은 확률로 근사 최적 계약을 학습하는 데 필요한 샘플 수를 최소화합니다. 특히 선형 및 유계 계약에 대해, 제안된 모델은 이론적으로 최적의 거래를 제공합니다. 반면, 비유계 계약에 대해서는 어떤 학습 알고리즘도 존재하지 않음을 보여주는 불가능 결과(impossibility result)를 제공합니다. 마지막으로, 제안된 알고리즘은 온라인 학습 셋팅으로 확장 가능하며, 이는 데이터 기반의 알고리즘 설계에 대한 새로운 통찰을 제공합니다.



### Pesti-Gen: Unleashing a Generative Molecule Approach for Toxicity Aware Pesticide Design (https://arxiv.org/abs/2501.14469)
Comments:
          9 pages, 2 figures, 5 tables

- **What's New**: 이 논문에서는 기존의 기계 학습 방법들이 농업에서 새로운 분자 구조를 생성하는 데 한계가 있음을 지적하고, 최초로 새로운 농약 후보를 생성할 수 있는 생성 모델인 Pesti-Gen을 제안합니다. 이 모델은 변량 자동 인코더(Variational Autoencoder, VAE)를 기반으로 하여 농약의 최적화된 특성을 가진 후보 분자를 생성하는 것을 목표로 합니다. Pesti-Gen은 초기의 일반 화학 구조 표현을 포착하는 사전 교육 및 특정 독성 정보를 반영하는 미세 조정 단계를 포함한 두 단계의 학습 과정을 통해 작동합니다.

- **Technical Details**: Pesti-Gen은 가축 독성(LD50) 및 수생 생태 독성(LC50)과 같은 다양한 독성 지표를 최적화하여 농업 환경에 적합한 친환경 농약 후보를 생성합니다. 이를 위해 연구진들은 WHO 독성 분류 기준에 기반하여 세심하게 선별된 독성 메트릭스를 사용하고, 다양한 농약 후보의 SMILES 표현과 독성 점수를 포함한 맞춤형 데이터 세트를 구축하였습니다. Pesti-Gen은 화학 유효성을 평가하고, 물리화학적 특성을 비교하며, SMILES 유효성 검사를 통해 모델의 신뢰성을 검증하였습니다.

- **Performance Highlights**: Pesti-Gen은 새로운 분자 구조를 생성하는 데 있어 약 68%의 화학 유효성을 달성하였습니다. 이 모델은 또한 합성 접근성과 독성 감소 간의 균형을 잘 유지하며 실질적인 응용 가능성을 강조합니다. 농약 설계의 생태독성 제약들을 통합함으로써, Pesti-Gen은 혁신적이고 환경 친화적인 농약 솔루션 개발의 초석을 다지고 있습니다.



### Interpretability Analysis of Domain Adapted Dense Retrievers (https://arxiv.org/abs/2501.14459)
- **What's New**: 본 논문에서는 dense retriever의 도메인 적응 후 모델의 동작 변화를 분석하기 위해 Integrated Gradients (IG) 프레임워크를 활용한 설명 방법을 제안합니다. 이를 통해 FIQA와 TREC-COVID 데이터셋에서 쿼리 및 문서 토큰에 대한 입력 기여도를 시각화하여 도메인 적응의 영향을 평가합니다. 특히, 도메인 적응된 모델이 더 많은 인도메인 용어에 집중한다는 점이 발견되었습니다.

- **Technical Details**: 이 연구는 Dense Retriever 모델을 다루며, 문서의 사전 인덱싱을 이용하여 쿼리와 문서 임베딩 간의 내적 유사도를 기반으로 상위 K개의 문서를 검색하는 방식을 채택합니다. DistilBERT를 사용하여 쿼리와 문서의 임베딩을 생성하고, Integrated Gradients를 통해 쿼리 및 문서 기여도를 계산합니다. 연구에서 사용된 데이터셋은 FIQA와 TREC-COVID이며, 두 데이터셋에서 도메인 적응의 효과를 분석하기 위한 방법론을 설계했습니다.

- **Performance Highlights**: 연구 결과, 도메인 적응된 모델은 인도메인 용어에 더 많은 집중을 보이며, 이는 기존 비적응 모델이 간과하는 언어적 요소를 반영합니다. Integrated Gradients를 통해 얻은 인사이트는 dense retriever의 내부 메커니즘을 분석하는 데 유용하며, 도메인 적응이 모델의 성능과 해석 가능성에 미치는 영향을 강조합니다. 이러한 분석은 향후 정보 검색(Information Retrieval) 연구에 있어 중요한 기여를 할 것으로 보입니다.



### Learning more with the same effort: how randomization improves the robustness of a robotic deep reinforcement learning agen (https://arxiv.org/abs/2501.14443)
Comments:
          This article was accepted and published in Applied Intelligence (https://doi.org/10.1007/s10489-022-04227-3)

- **What's New**: 이 연구는 Deep Reinforcement Learning (DRL)의 산업적 응용을 가로막는 데이터 수집의 불리함을 해결하기 위해, 인공 경험을 효과적으로 실제 세계로 전이하는 방법에 대해 논의합니다. 특히, Progressive Neural Networks (PNNs)라는 최신 기술의 강건성을 분석하고, 인공 경험에 다양성을 추가하는 것이 이 문제를 어떻게 보완할 수 있는지 연구합니다. 최종 목표는 가상 훈련 단계 후 에이전트의 강건성을 측정하여 실제 환경에서의 높은 성능을 보장하는 것입니다.

- **Technical Details**: 이 연구에서는 RL 문제의 일반적인 공식화와 에이전트의 정책을 정의하며, 모델 프리(model-free) 방법과 모델 기반(model-based) 방법을 비교합니다. RL 에이전트는 자체적인 경험을 통해 상태 전이 및 보상을 예측하는 기능을 학습하지만, 실제 환경의 진실 모델이 항상 존재하지 않기 때문에 이로 인한 도전이 발생합니다. 이 연구는 Domain Randomization (DR) 기법을 통해, 특히 가상 모델 훈련 중 특정 변수를 무작위화하는 방법이 에이전트의 강건성을 어떻게 향상시키는지를 다룹니다.

- **Performance Highlights**: 연구 결과, PNN을 기반으로 한 에이전트는 실제 훈련 단계 초기에 강건성이 크게 저하되는 경향을 보였습니다. 그러나 훈련 과정에 다양성을 도입할 경우 모델의 정확도가 평균 25% 증가하며, 이는 동일한 강건성 성능을 위해 필요한 실제 경험의 양을 줄이는 효과를 가져옵니다. 이로써 인공 경험의 품질에 관계없이, 실제 경험의 추가가 여전히 유리하다는 점을 강조합니다.



### Adaptive Rank Allocation for Federated Parameter-Efficient Fine-Tuning of Language Models (https://arxiv.org/abs/2501.14406)
- **What's New**: 이 논문은 Federated Parameter-Efficient Fine-Tuning (FedPEFT)의 한계를 극복하기 위해 FedARA라는 새로운 방법을 제안합니다. FedARA는 비독립 및 동일 분포(Non-IID) 데이터로 인한 성능 저하와 고정 파라미터 구성으로 인한 통신 비효율성을 개선하기 위해 설계되었습니다. 이 방법은 트렁케이티드 싱귤러 벨류 분해(Truncated SVD), 동적 랭크 할당(Dynamic Rank Allocation), 랭크 기반 모듈 프루닝(Rank-based Module Pruning)을 결합하여 언어 모델의 파라미터 효율성 있는 미세조정을 지원합니다.

- **Technical Details**: FedARA는 비독립적이고 비동일 분포의 데이터가 미치는 영향을 줄이기 위해 고안되었습니다. 이를 위해, FedARA는 트렁케이티드 싱귤러 벨류 분해 적응 방식을 도입하여 다양한 클라이언트 간 공통 특성을 학습하도록 지원합니다. 또한, 동적 랭크 할당을 통해 각 클라이언트에서 중요한 랭크를 동적으로 식별하며, 비효율적인 통신을 줄이기 위해 랭크 변화에 따라 비활성 모듈을 안전하게 제거합니다.

- **Performance Highlights**: FedARA는 여러 데이터셋에서 평균 8.49%의 정확도 향상을 보여주며, 비독립적 데이터 환경에서도 기존의 약한 기준선보다 평균 6.95% 우수함을 입증하였습니다. 실험 결과, AGX Orin, Orin Nano, Raspberry Pi 5와 같은 모바일 장치에서 총 훈련 시간이 최대 48.90%, 에너지 소비는 46.95%까지 감소하는 성과를 거두었습니다. 이로 인해 FedARA는 모바일 장치에서의 학습 효율성과 에너지 효율성을 크게 개선함을 확인하였습니다.



### SKIL: Semantic Keypoint Imitation Learning for Generalizable Data-efficient Manipulation (https://arxiv.org/abs/2501.14400)
Comments:
          22 pages, 22 figures

- **What's New**: 이번 논문에서는 로봇들이 다양한 작업을 수행하도록 하는 새로운 방법, Semantic Keypoint Imitation Learning (SKIL)을 제안합니다. SKIL은 비전 모델을 사용하여 자동으로 의미론적 키포인트(semantic keypoints)를 얻고, 이러한 키포인트를 기반으로 효율적인 모방 학습(imitation learning)을 수행합니다. 이 방법은 복잡한 로봇 작업을 수행하는 데 필요한 샘플 복잡성을 기존 방법보다 획기적으로 낮출 수 있습니다.

- **Technical Details**: SKIL 프레임워크는 비전 모델을 활용하여 의미론적 키포인트를 관찰(observations)로 식별합니다. 이 스파스 간소화된 표현은 문제의 차원을 줄이는 데 도움이 되며, 훈련과 테스트 객체 간에 일관된 키포인트를 매칭함으로써 로봇의 동작을 출력할 수 있는 diffusion-based action head에 조건 입력으로 제공합니다. SKIL은 이러한 의미론적 키포인트 추상을 통해 크로스-엠바디먼트 학습(cross-embodiment learning)을 자연스럽게 지원합니다.

- **Performance Highlights**: 실험 결과, SKIL은 컵이나 마우스를 집는 작업과 같은 6개의 실제 작업에서 이전 방법들에 비해 146% 향상된 성공률 72.8%를 달성했습니다. 특히 수건을 걸치는와 같은 장기 작업(long-horizon tasks)에서도 30회의 시연으로 평균 70%의 성공률을 보이며, 기존 방법이 완전히 실패했던 작업에서도 놀라운 성과를 냈습니다. 이러한 결과들은 SKIL이 데이터 효율적이고 일반화 가능한 로봇 학습을 성공적으로 달성했음을 보여줍니다.



### Handling Heterophily in Recommender Systems with Wavelet Hypergraph Diffusion (https://arxiv.org/abs/2501.14399)
- **What's New**: 본 논문에서는 하이퍼그래프 기반 추천 시스템의 진화를 위한 새로운 FWHDNN(퓨전 기반 웨이브릿 하이퍼그래프 확산 신경망) 프레임워크를 소개합니다. 이 모델은 이질적(hheterophily) 패턴과 다차원 사용자-아이템 상호작용을 포착하는 데 필요한 주요 세 가지 구성 요소를 포함합니다. 제안하는 방법은 다양한 클래스 레이블에 적응할 수 있는 메시지 패싱을 지원하는 이질성 인식 하이퍼그래프 확산을 활용합니다.

- **Technical Details**: FWHDNN은 세 가지 주요 구성 요소로 이루어져 있습니다. 첫 번째는 Cross-Difference Relation Encoder로, 다양한 클래스 레이블에 적응하는 메시지 패싱을 통해 이질적 패턴을 모델링합니다. 두 번째는 Multi-level Cluster-wise Encoder로, 웨이브릿 변환을 활용하여 다중 스케일의 구조적 관계를 포착합니다. 마지막으로, 통합된 다중 모달 융합 메커니즘을 통해 구조적 및 텍스트 정보를 결합합니다.

- **Performance Highlights**: FWHDNN은 실제 데이터셋에서 광범위한 실험을 통해 기존의 최첨단 방법들을 초월하는 정확성과 강인성 및 확장성을 보여줍니다. 특히, 사용자와 아이템 간의 고차원 연결 관계를 효과적으로 포착하는 데 강점을 지니고 있습니다. 이러한 성능은 FWHDNN이 복잡한 추천 환경을 효과적으로 처리할 수 있는 가능성을 시사합니다.



### ECTIL: Label-efficient Computational Tumour Infiltrating Lymphocyte (TIL) assessment in breast cancer: Multicentre validation in 2,340 patients with breast cancer (https://arxiv.org/abs/2501.14379)
Comments:
          Under review. 54 pages including supplementary materials, 2 main tables, 3 main figures, 14 supplementary figures, 4 supplementary tables

- **What's New**: 이 논문에서는 삼중 음성 유방암 환자를 위한 종양 침윤 림프구(TIL)의 평가를 위한 새로운 접근 방식을 제안합니다. 기존의 복잡한 Computation TIL Assessment (CTA) 모델과 달리, 연구팀은 병리학적 주석이 적고도 훈련 시간이 단축된 딥 러닝 기반의 ECTIL 모델을 개발했습니다. 이 모델은 소수의 주석으로도 유방암 환자들의 TIL 점수를 예측할 수 있도록 만들어졌습니다.

- **Technical Details**: ECTIL 모델은 병리학 기초 모델을 사용하여 전체 슬라이드 이미지(WSI)에서 형태학적 특징을 추출합니다. 모델을 훈련하는 데 소수의 샘플만 필요하며, 다양한 외부 코호트에서 병리학자와의 일치도(r=0.54-0.74, AUROC=0.80-0.94)가 입증되었습니다. 또한, ECTIL이 TIL 점수를 직접 회귀 분석하는 방법으로, 수십 배 적은 주석으로 훈련될 수 있는 장점을 가지고 있습니다.

- **Performance Highlights**: 모델의 성능은 ECTIL이 훈련된 데이터셋에서 병리학자 점수와 유사한 생존율 비율(hazard ratio, HR) 0.86을 도출하면서 입증되었습니다. ECTIL은 기존 방법들보다 단순한 구조를 가지고 있으며, 임상 치료 시험에 환자를 선별하는 데 사용될 수 있는 가능성을 가지고 있습니다. 오픈 소스 라이센스 하에 제공되며, 이는 더 많은 연구자들이 사용할 수 있도록 합니다.



### DRESSing Up LLM: Efficient Stylized Question-Answering via Style Subspace Editing (https://arxiv.org/abs/2501.14371)
Comments:
          ICLR 2025 Accepted

- **What's New**: DRESS는 LLM의 반응을 스타일화하기 위한 새롭고 효과적인 접근법으로, 표현 편집을 통해 스타일 관련 하위공간(style subspace)을 분리합니다. 기존의 방법인 프롬프트(prompting)나 미세 조정(fine-tuning)은 복잡한 스타일 적응을 위해 충분하지 않거나 계산적으로 비용이 많이 듭니다. DRESS는 LLM의 오버 파라미터화된 특성을 활용하여 표현 공간에서 의미(original semantics)에 최소한의 영향을 미치면서 스타일을 조정할 수 있는 방법을 제안합니다.

- **Technical Details**: DRESS는 스타일 관련 하위공간을 단계적으로 식별하고 그들 안에서 의미적으로 격리된 스타일 조정을 수행하는 세 가지 전략을 포함합니다. 첫 번째는 주의 헤드 여과(attention head filtering)로, 스타일과 관련된 주의 헤드를 식별합니다. 두 번째는 스타일 하위공간 여과로(style subspace filtering), 선택된 주의 헤드 내에서 스타일과 관련없는 컴포넌트를 제거합니다. 마지막으로 각 하위공간 기초와 생성된 토큰에 대해 적응형 편집 강도(adaptive editing strength)를 적용하여 유연성을 제공합니다.

- **Performance Highlights**: DRESS는 스타일 강도(style intensity), 의미 보존(semantic preservation), 유창성과 같은 객관적인 평가 지표를 포함하는 두 개의 스타일화된 QA 벤치마크 데이터셋을 사용하여 그 효과를 입증합니다. 실험 결과, DRESS는 SFT, 프롬프트 및 다른 표현 편집 방법들과 비교하여 성능이 크게 향상된 것으로 나타났습니다. DRESS는 또한 대화형 에이전트를 개발하는 데 특히 유용하여, 경량화되고 교육이 필요 없는 솔루션을 제공합니다.



### HorNets: Learning from Discrete and Continuous Signals with Routing Neural Networks (https://arxiv.org/abs/2501.14346)
Comments:
          Accepted to the ACML conference journal track with the Machine Learning journal. The first and the last authors share an equal contribution

- **What's New**: 본 논문에서는 HorNets (Horn Networks)라는 신경망 아키텍처를 제안합니다. HorNets는 연속적이고 불연속적인 표형 데이터에서 효율적으로 학습할 수 있도록 설계되었으며, 이는 적은 수의 데이터로도 높은 성능을 발휘합니다. 이 아키텍처는 클리핑된 다항식 유사 활성화 함수를 기반으로 하며, 입력의 카드inality에 따라 최적화할 신경망의 부분을 결정하는 사용자 지정된 라우팅 메커니즘이 포함되어 있습니다. HorNets는 특히 생물 의학적 고차원 데이터 세트에서 최첨단 분류 성능을 달성합니다.

- **Technical Details**: HorNets는 폴리클립 활성화 함수(polyClip activation function)와 사용자 정의 라우팅 메커니즘을 사용하여 입력 데이터의 특성 조합 공간을 명시적으로 모델링합니다. 이러한 접근 방식은 기계 학습 모델이 특정 데이터 조각에 대해 가장 적합한 작동 방식을 결정할 수 있도록 합니다. HorNets는 논리 게이트를 모델링하는 능력을 평가하기 위해 광범위한 벤치마크를 실시했으며, 일반적인 논리 게이트를 모델링하는 데 뛰어난 재현율(recall)을 보여주었습니다. 이는 고차원 생물 의학 데이터 세트에서 경쟁력 있는 성능을 입증하였습니다.

- **Performance Highlights**: HorNets는 14개의 실제 생물 의학적 데이터 세트에서 최고의 성능을 달성했으며, 이는 그래디언트 부스팅 트리(gradient-boosted trees) 및 AutoML 기반 분류기와 비교하여 우수합니다. 연구 결과, HorNets는 노이즈가 포함된 XNOR와 같은 논리 절을 안정적으로 검색할 수 있는 몇 안 되는 접근 방식 중 하나입니다. 이 아키텍처는 10개 이상의 실제 데이터 세트에서 경이로운 성능을 보이며, 기존 방법들보다 효과적인 솔루션으로 자리 잡았습니다.



### Relative Layer-Wise Relevance Propagation: a more Robust Neural Networks eXplaination (https://arxiv.org/abs/2501.14322)
Comments:
          arXiv admin note: text overlap with arXiv:2012.14501, arXiv:1605.01713 by other authors

- **What's New**: 이번 연구에서는 새로운 정의인 Relative LRP(R-LRP)를 소개하여, 기존의 Layer-Wise Relevance Propagation(LRP) 방법의 한계점을 해결하고자 했습니다. 특히, 작은 값으로 나누는 문제를 최소화하여 다층 신경망의 출력에 기여하는 정도를 보다 정확하게 시각화할 수 있게 되었습니다. 이러한 접근 방식은 이미지 분류 작업에 특히 유용하며, 픽셀 단위로 기여도를 분석함으로써 예측에 영향을 미치는 중요 영역을 찾는데 도움을 줍니다.

- **Technical Details**: R-LRP는 기존의 LRP 방법과는 달리 하이퍼파라미터 튜닝이 필요하지 않아 사용이 용이합니다. 이 방법은 일반적인 CNN, VGG16, VGG19, 및 Resnet50 네트워크 등 다양한 네트워크에서 적용 가능하며, 다층 신경망의 입력 도메인에 대한 출력 기여도를 정량화합니다. 연구에서는 특히 Resnet의 스킵 연결에 대해서만 작은 값으로 나누는 방식이 남아있음을 강조하였습니다.

- **Performance Highlights**: R-LRP 방법은 다양한 데이터셋에서 간단한 CNN 아키텍처와 VGG, Resnet 네트워크와 비교하여 효과적임을 보여주었습니다. 이 방법은 모델의 결정 과정을 보다 투명하게 하여 사용자가 신뢰할 수 있는 예측을 제공하는 데 기여합니다. 픽셀 기여도의 시각화는 예측의 중요 부분에 대한 깊이 있는 분석을 가능하게 합니다.



### Permutation-based multi-objective evolutionary feature selection for high-dimensional data (https://arxiv.org/abs/2501.14310)
- **What's New**: 이 논문에서는 고차원 데이터의 특성 선택(feature selection) 방법을 새롭게 제안합니다. 기존의 permutation feature importance (PFI) 방법을 확장하여 개별 특성 대신 속성 하위 집합을 평가하는 방식입니다. 이 새로운 접근법은 모델 성능의 상호작용을 더 효과적으로 포착하며, 복잡한 특성 상호작용을 고려한 선택을 가능하게 합니다.

- **Technical Details**: 제안된 방법은 다목적 진화 알고리즘(multi-objective evolutionary algorithm)을 사용하여 후보 특성 하위 집합을 탐색합니다. 주요 목표는 선택된 특성을 셔플(shuffle)할 때 모델 성능의 감소를 극대화하고, 특성 하위 집합의 크기를 최소화하는 것입니다. PSEFS-MOEA(Permutation-based Subset Evaluation Feature Selection with MOEA)는 이러한 방식을 적용하여 하이퍼파라미터의 영향을 최소화하며, 더욱 정확하고 전체적인 특성 선택을 가능하게 합니다.

- **Performance Highlights**: PSEFS-MOEA는 24개의 고차원 데이터셋에서 실험을 통해 기존의 다른 9가지 유명한 특성 선택 방법보다 우수한 성능을 보여주었습니다. 이 방법은 모델의 성능 지표 측면에서도 꾸준히 높은 평가를 받았으며, 과적합(overfitting) 없이 강력한 일반화 능력을 지니고 있습니다. 논문에서는 이 방법이 고차원 데이터의 복잡한 특성 상호작용을 효과적으로 처리할 수 있음을 입증하고 있습니다.



### Learning Primitive Relations for Compositional Zero-Shot Learning (https://arxiv.org/abs/2501.14308)
Comments:
          Accepted to ICASSP 2025

- **What's New**: 본 논문에서는 Compositional Zero-Shot Learning (CZSL) 분야에서 새로운 프레임워크인 Learning Primitive Relations (LPR)을 제안합니다. CZSL은 이전에 본 조합에서 얻은 지식을 활용하여 보지 못한 상태-객체 조합을 식별하는 것을 목표로 합니다. 기존 접근 방법들은 상태와 객체를 독립적으로 예측하는 경향이 있었으나, LPR은 이들의 관계를 확률적으로 포착합니다.

- **Technical Details**: LPR은 cross-attention mechanism을 활용하여 상태와 객체 간의 의존성을 고려합니다. 이 연구는 상태-객체 관계를 기반으로 보지 못한 조합의 가능성을 추론할 수 있도록 모델을 설계하였습니다. 기술적으로, 이 프레임워크는 상태와 객체 간의 복잡한 상호작용을 효과적으로 학습할 수 있게 해줍니다.

- **Performance Highlights**: 실험 결과, LPR은 닫힌 세계(closed-world)와 열린 세계(open-world) 설정에서 모두 세 가지 CZSL 벤치마크 데이터 세트에서 최신 기술(state-of-the-art) 방법보다 우수한 성능을 보였습니다. 정성적 분석을 통해 LPR이 보지 않은 조합 예측을 위해 상태-객체 관계를 효과적으로 활용함을 보여주었습니다.



### A Zero-Shot LLM Framework for Automatic Assignment Grading in Higher Education (https://arxiv.org/abs/2501.14305)
- **What's New**: 이번 연구에서는 Zero-Shot Large Language Model (LLM)-기반 자동 과제 채점 시스템(AAG 시스템)을 제안합니다. 이는 추가적인 훈련이나 파인튜닝 없이 학생의 답변을 평가할 수 있으며, 개인 맞춤형 피드백을 제공합니다. 기존 자동 채점 시스템이 직면한 문제점인 데이터셋 요구, 개인화된 피드백 부족, 벤치마크 성능 우선 경향을 해결합니다.

- **Technical Details**: AAG 시스템은 학생의 과제 제출물, 과제 질문, 그리고 선택적으로 참조 솔루션 또는 마킹 기준을 입력으로 사용합니다. 이 시스템은 특히 STAT1011 통계 입문 과목에서 활용될 수 있으며, 다양한 과제 문제를 처리하는 데 유연성을 제공합니다. 이 시스템은 GPT-4를 사용하여 학생 제출물 평가 및 문제 요약을 수행하며, 비교 평가 결과 GPT-4가 가장 효과적인 모델로 선정되었습니다.

- **Performance Highlights**: 종합 평가 결과, AAG 시스템이 기존의 전통적 채점 방법에 비해 학생들의 동기 부여, 이해도 및 준비성을 크게 향상시켰습니다. 이는 설문조사를 통해 입증된 결과로, 시스템이 교육 평가의 질과 학생 경험을 혁신할 수 있는 가능성을 보여줍니다. 또한, 개인 맞춤형 피드백을 통해 학습 성과를 더욱 향상시킵니다.



### Examining Alignment of Large Language Models through Representative Heuristics: The Case of Political Stereotypes (https://arxiv.org/abs/2501.14294)
Comments:
          ICLR 2025

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)을 인간의 의도와 가치에 맞추는 데 있어 정치적 성향의 중요성을 강조하고 있습니다. 이전 연구들이 LLM의 정치적 경향성을 보여주었으나, LLM이 실제 위치에서 벗어나는 정도와 그 조건은 충분히 조사되지 않았습니다. 본 연구는 LLM의 정치적 문제에 대한 편향을 수량화하고, 이로 인해 발생하는 편차의 조건을 규명하고자 하였습니다.

- **Technical Details**: 이 연구는 인지 과학의 자료를 바탕으로 대표성 휴리스틱(representativeness heuristics)을 통해 LLM의 응답을 분석합니다. 대표성 휴리스틱은 특정 집단의 특징을 과대평가하는 인지 현상으로, 정치적 편향이 특정 정당의 입장을 과장하는 방식으로 나타나는지 실험을 통해 조사하였습니다. 연구 결과, LLM은 특정 정치 정당의 입장을 모방할 수 있으나, 인간 응답자에 비해 이러한 입장을 더 과장하게 나타냅니다.

- **Performance Highlights**: LLM의 응답은 일반적으로 실제의 진실성(kernel-of-truth)을 반영하였으나, 인간 응답자보다 더 극단적으로 나타났습니다. 이 연구는 LLM이 정치적 편향에 취약할 수 있는 가능성을 확인하였으며, 대표성을 줄이기 위한 프롬프트 기반의 방법들이 효과적임을 보여주었습니다. 궁극적으로, LLM에 대한 새로운 관점을 제시하고, 그들의 편향을 완화하기 위한 전략을 제안하는 것을 목표로 했습니다.



### A Comprehensive Framework for Semantic Similarity Detection Using Transformer Architectures and Enhanced Ensemble Techniques (https://arxiv.org/abs/2501.14288)
- **What's New**: 이 논문은 AI 생성 텍스트, 특히 짧은 맥락 문서에서의 탐지 문제를 해결하기 위해 새로운 teacher-student 모델을 제안합니다. teacher 모델은 DeBERTa-v3-large와 Mamba-790m을 결합하여 도메인 특화 미세 조정을 통해 의미론적 지식을 학습합니다. student 모델은 짧은 맥락 텍스트를 더 효율적으로 처리하며, 손실 함수로 Mean Squared Error (MSE)를 사용하여 학습을 안내받습니다.

- **Technical Details**: 이 모델은 d학습에 있어 도메인 적응과 데이터 증강을 통합하고, 다양한 텍스트 변경 방법(예: 오타 수정, 오류 삽입)을 통해 강건성을 강화할 수 있습니다. 특히, teacher 모델은 세부적인 의미적 특성을 포착하고 AI 생성 콘텐츠를 보다 잘 탐지하도록 도와줍니다. 최종적으로, 이 시스템은 짧은 텍스트 문서 분류에서 높은 성능을 발휘하며, 다른 모델들이 어려움을 겪는 영역에서 효과적으로 작동합니다.

- **Performance Highlights**: 실험 결과는 이 접근 방식이 기존의 기준 방법들보다 더 우수한 성능을 보여준다고 밝혀졌습니다. 특히, 이 모델은 짧은 맥락 문서에서도 AI 생성 텍스트 탐지 및 텍스트 분류 작업에 유용성을 증명합니다. 이러한 결과는 실시간 AI 생성 텍스트 탐지와 같은 다양한 응용 프로그램에 적용할 수 있습니다.



### Active Learning for Continual Learning: Keeping the Past Alive in the Presen (https://arxiv.org/abs/2501.14278)
- **What's New**: 논문에서는 AccuACL(Accumulated informativeness-based Active Continual Learning)이라는 새로운 방법을 제안합니다. 이는 Fisher 정보 행렬을 샘플 선택 기준으로 사용하여 과거 지식을 보존하면서 새로운 작업을 신속하게 배울 수 있도록 돕습니다. 이 접근 방식은 기존의 Active Learning(AL) 전략과는 달리 기계 학습 모델의 성능을 극대화하면서 레이블 비용을 줄이는 데 초점을 맞추고 있습니다. 따라서 ACL(Active Continual Learning) 시나리오에서 복잡하고 연속적인 데이터 분포에 효과적으로 대응할 수 있는 가능성을 제시합니다.

- **Technical Details**: AccuACL은 Fisher 정보 행렬을 통해 누적 정보를 모델링하여, 과거 특정 작업과 새로운 작업 간의 정보를 균형 있게 평가합니다. 이 알고리즘은 레이블이 붙지 않은 데이터의 풀에서 최적의 예제를 선택하는 조합 최적화 문제로 모델링되었습니다. 논문의 핵심 기술은 새로운 작업에서 과거 지식을 보존하는 데 기여할 수 있는 예제를 식별하는 데 중점을 두며, 이를 통해 악성 망각(catastrophic forgetting)을 방지하는 것을 목표로 합니다.

- **Performance Highlights**: AccuACL은 SplitCIFAR10, SplitCIFAR100, SplitTinyImageNet의 세 가지 CL 벤치마크에서 기존 AL 기준을 23.8% 및 잊어버림 측면에서 17.0% 향상시켰습니다. 실험 결과, AccuACL은 과거 작업과 관련된 예제에 대한 높은 성과를 보여주어 전체적인 성능에서 우위를 점하였습니다. 이러한 결과는 AccuACL이 과거 지식과 새로운 작업 학습 사이의 균형을 잘 유지할 수 있음을 시사합니다.



### Global Semantic-Guided Sub-image Feature Weight Allocation in High-Resolution Large Vision-Language Models (https://arxiv.org/abs/2501.14276)
Comments:
          10 pages, 10 figures and tables

- **What's New**: 본 논문에서는 고해상도 이미지 처리의 수요가 증가함에 따라, 기존의 균등한 sub-image partitioning 방법이 모델의 시각적 이해 능력을 저하시킬 수 있음을 밝히고, Global Semantic-guided Weight Allocator (GSWA) 모듈을 제안합니다. GSWA 모듈은 정보 밀도에 따라 동적으로 sub-image의 가중치를 할당하여 모델이 더 유익한 영역에 집중할 수 있도록 합니다. 이를 통해 SleighVL이라는 경량 모델을 개발하였으며, 이는 기존 모델들과의 성능 비교에서 우수한 결과를 얻었습니다.

- **Technical Details**: 연구에서는 Vision Transformer (ViT)를 활용한 인코더를 기반으로 하여, sub-image partitioning을 통해 고해상도 이미지를 더 잘 처리하는 방안을 제시합니다. GSWA 모듈은 각 sub-image의 정보 밀도를 고려하여 가중치를 조정하며, 이는 인간의 시각적 주의 메커니즘을 모사합니다. 따라서 모델은 중요한 정보가 포함된 지역에 더 집중할 수 있게 됩니다. 이러한 방식으로 SleighVL은 여러 벤치마크에서 유의미한 성능 향상을 보여 줍니다.

- **Performance Highlights**: SleighVL 모델은 기존 LVLM들 및 최신 SOTA 모델들과 비교했을 때, 경량화에도 불구하고 높은 성능을 유지합니다. 다양한 평가 기준에서 기본 모델에 비해 현저한 성능 개선을 달성하였으며, 고해상도 시각적 정보를 효과적으로 통합할 수 있는 능력을 입증하였습니다. 특히, GSWA 모듈은 복잡한 이미지 처리 시 더 나은 효과를 보여줍니다.



### Leveraging Online Olympiad-Level Math Problems for LLMs Training and Contamination-Resistant Evaluation (https://arxiv.org/abs/2501.14275)
- **What's New**: 이번 논문에서는 올림피아드(olympiad) 수준의 수학 문제 해결을 위한 고품질 데이터셋 구축과 LLM(대형 언어 모델)의 성능 향상을 위한 자동화된 파이프라인을 소개합니다. 특히, AoPS(Art of Problem Solving) 포럼의 자원을 활용하여 60만 개 이상의 QA(pair) 쌍을 포함하는 AoPS-Instruct 데이터셋을 개발했습니다. 이 외에도, 새로운 평가 세트인 LiveAoPSBench를 통해 LLM의 수학 추론 능력을 평가할 수 있는 오염 저항적인 벤치마크를 구축했습니다.

- **Technical Details**: 이 연구에서는 LLM의 미세 조정을 위한 AoPS-Instruct 데이터셋과 더불어 LiveAoPSBench라는 자동화된 평가 세트를 소개합니다. AoPS-Instruct는 AoPS 포럼의 게시글에서 QA 쌍을 추출하여 생성되었으며, 이는 고급 수학 문제에 적합한 대규모 데이터셋입니다. 자동화된 파이프라인을 통해 지속적으로 최신 문제를 포함한 평가 세트를 운영함으로써 오염 가능성을 줄였습니다.

- **Performance Highlights**: 실험 결과, LLM을 AoPS-Instruct로 미세 조정한 후, OlympiadBench, Omni-Math, LiveAoPSBench와 같은 다양한 벤치마크에서 성능이 향상되었습니다. 또한, 시간 경과에 따른 성능 저하가 관찰되어, 오래된 문제에 대한 성공이 실제 수학적 추론 능력이 아니라 사전 훈련 피드백에 기인할 수 있음을 시사합니다. 따라서 본 연구는 LLM의 수학적 추론 능력에 대한 통찰력을 제공하며, 이를 통해 데이터셋의 생성과 유지 관리 방법론의 중요성을 강조하고 있습니다.



### Hierarchical Time-Aware Mixture of Experts for Multi-Modal Sequential Recommendation (https://arxiv.org/abs/2501.14269)
Comments:
          Accepted to WWW 2025

- **What's New**: 이 논문에서는 Multi-modal Sequential Recommendation (SR)에서의 한계를 극복하기 위해 Hierarchical time-aware Mixture of Experts (HM4SR)라는 새로운 방법을 제안합니다. 기존 방법들은 주로 사용자 선호도의 변화와 관련한 다채로운 모달리티 데이터를 활용하지 않고 있으나, HM4SR은 두 가지 레벨의 Mixture of Experts (MoE)를 도입하여 이러한 모달의 중요 정보를 추출합니다. 또한, 시간 관련 임베딩을 통해 사용자 관심의 변화를 더 효과적으로 모델링하는 접근을 취하고 있습니다.

- **Technical Details**: HM4SR의 구조는 두 개의 MoE로 구성되어 있으며, 첫 번째 MoE인 Interactive MoE는 다중 모달 데이터에서 사용자 관심과 관련된 핵심 정보를 추출합니다. 두 번째 MoE인 Temporal MoE는 명시적인 시간 정보를 사용하여 동적인 사용자 관심을 캡처하고, 이를 통해 멀티모달 학습에서의 시간 관련 정보를 통합하는 방식으로 설계되었습니다. 또한, 세 가지 보조 감독 작업(CP, IDCL, PCL)을 통해 더욱 풍부한 학습 신호를 제공합니다.

- **Performance Highlights**: HM4SR은 네 개의 공개 데이터셋에 대한 광범위한 실험을 통해 기존의 여러 최첨단 방법들과 비교하여 그 효과성을 입증하였습니다. 이 방법은 다중 모달 정보의 풍부함을 활용하여 사용자 관심의 진정한 모델링을 지원하고, 시계열 정보를 통해 동적인 관심 변화를 효과적으로 캡처하는 데 성공했습니다. 전반적으로 HM4SR은 다중 모달 SR 분야에서 중요한 기여를 할 것으로 보입니다.



### Pre-train and Fine-tune: Recommenders as Large Models (https://arxiv.org/abs/2501.14268)
Comments:
          Accepted by WWW2025

- **What's New**: 이 논문에서는 다중 도메인 학습의 한계를 극복하기 위해 추천 시스템을 대규모 사전 훈련 모델로 간주하고 이를 세밀하게 조정(fine-tune)하는 방법을 제안합니다. 기존의 다중 도메인 학습에서는 새로운 작업을 추가할 때 모델 전체를 재훈련해야 하지만, 본 연구에서는 각 도메인에 대해 사전 훈련된 모델의 소규모 조정을 통해 효율적으로 모델을 개선할 수 있습니다. 또한, 정보 병목(information bottleneck)의 관점에서 추천 시스템의 조정 기법에 대한 이론적 설명을 제공합니다.

- **Technical Details**: 제안된 방법론은 정보 인식 적응 커널(Information-Aware Adaptive Kernel, IAK)을 활용하여 사전 훈련된 추천 모델을 미세 조정하는데 중점을 둡니다. IAK는 일반적인 비즈니스 지식을 압축하고 주어진 하위 작업에 대한 특정 지식을 학습하며, 관련 없는 지식은 잊어버리는 구조를 취합니다. 이 과정은 사용자 맞춤형 추천 결과를 보다 정확하게 도출할 수 있도록 하며, 다양한 도메인이나 조건에서 유연한 추천을 가능하게 합니다.

- **Performance Highlights**: 다양한 오프라인 및 온라인 실험에서 제안된 접근 방식의 우수성이 입증되었습니다. 특히, 이 연구에서 제안한 IAK 기법은 수억 명의 사용자를 대상으로 한 대규모 온라인 음식 플랫폼에서 실제로 배포되어 상당한 비즈니스 이익도 창출하였습니다. 또한, 연구팀은 추천 시스템에서 발견된 두 가지 잠재적 문제점에 대해 논의하며, 이 문제들을 해결하기 위한 탐색을 제공합니다.



### Siren: A Learning-Based Multi-Turn Attack Framework for Simulating Real-World Human Jailbreak Behaviors (https://arxiv.org/abs/2501.14250)
- **What's New**: 이 논문에서는 LLM(대형 언어 모델)의 안전성과 신뢰성 문제를 해결하기 위해 Siren이라는 학습 기반의 멀티 턴 공격 프레임워크를 제안합니다. 기존의 단일 턴 공격 방법과는 달리, Siren은 공격자가 여러 턴을 통해 동적으로 공격 쿼리를 생성하고 롤 플레이를 통해 실제 인간의 jailbreak 행동을 시뮬레이션합니다. 이를 통해 Siren은 다른 멀티 턴 모델에 비해 공격 성공률을 획기적으로 향상시킵니다.

- **Technical Details**: Siren은 세 가지 단계로 구성됩니다: (1) Turn-Level LLM 피드백을 활용한 훈련 세트 구축, (2) 감독된 세부 조정(Supervised Fine-Tuning, SFT) 및 직접 선호 최적화(Direct Preference Optimization, DPO)를 통한 사후 훈련 공격자, (3) 공격자와 목표 LLM 간의 상호작용. 이 프레임워크는 학습 기반 접근 방식을 사용하여 쿼리 생성을 자동화하며, 공격의 복잡성에 효과적으로 적응합니다.

- **Performance Highlights**: 실험 결과에 따르면, Siren은 LLaMA-3-8B를 사용하여 Gemini-1.5-Pro 모델에 대한 공격 성공률(ASR)을 90% 달성하였으며, Mistral-7B를 이용해 GPT-4o에 대해서는 70%의 성공률을 기록했습니다. 또한, Siren은 7B 규모 모델로써 멀티 턴 공격에서 강력한 효과를 발휘하며, 효과적인 공격과 함께 인과적 의미 연관성을 유지하는 전략을 사용하여 적은 턴 수를 요구합니다.



### Humanity's Last Exam (https://arxiv.org/abs/2501.14249)
Comments:
          25 pages, 6 figures

- **What's New**: 이번 논문에서는 Humanity's Last Exam (HLE)라는 새로운 다중 모드 벤치마크를 소개합니다. 기존의 벤치마크가 충분한 도전을 제공하지 못하는 상황에서, HLE는 3,000개의 질문을 포함하여 인간 지식의 최전선에서 설계되었습니다. 본 연구는 과거의 전통적인 벤치마크에서 LLM의 성능을 측정하는 데 있어 격차를 해소하고자 합니다.

- **Technical Details**: HLE는 수학, 인문학, 자연 과학 등 다양한 주제를 포함한 질문들로 구성되어 있으며, 선택형 및 단답형 질문 형식을 제공합니다. 질문은 명확하고 검증 가능하며, 단순한 인터넷 검색으로는 빠르게 답변할 수 없습니다. 데이터셋은 공신력 있는 주제 전문가들에 의해 개발되었으며, 정확한 답변과 함께 상세한 해설도 포함되어 있습니다.

- **Performance Highlights**: 최신 LLM은 HLE에서 10% 미만의 정확도를 보이며, 이는 현재의 LLM 능력과 전문가 인간의 성능 사이에 존재하는 큰 격차를 보여줍니다. AI 시스템이 여러 영역에서 인간 전문가의 성능에 접근함에 따라, 이러한 정밀한 측정은 연구 및 정책 결정을 위한 중요한 토대가 됩니다. HLE의 높은 성과는 클로즈드 엑센들 테스트에서 전문가 수준의 능력을 나타낼 것임을 시사합니다.



### Point-LN: A Lightweight Framework for Efficient Point Cloud Classification Using Non-Parametric Positional Encoding (https://arxiv.org/abs/2501.14238)
Comments:
          This paper has been accepted for presentation at the 29th International Computer Conference, Computer Society of Iran (CSICC) 2025

- **What's New**: 이번 논문에서는 3D 포인트 클라우드 분류를 위해 설계된 경량화된 새로운 프레임워크인 Point-LN을 소개합니다. Point-LN은 Farthest Point Sampling (FPS), k-Nearest Neighbors (k-NN)와 같은 비모수적(non-parametric) 컴포넌트를 통합하여 학습 가능한 분류기와 연결하여 분류 정확도를 높였으며, 최소한의 파라미터로 컴퓨팅 비용을 줄입니다. 이 하이브리드 아키텍처는 실시간 및 리소스 제약이 있는 애플리케이션에 최적화되어 있습니다.

- **Technical Details**: Point-LN은 특징 인코더(feature encoder)와 분류기(classifier)라는 두 가지 주요 구성 요소로 이루어져 있습니다. 특징 인코더는 원시 입력 포인트 클라우드를 고차원 특징 표현으로 변환하며, 분류기는 이러한 인코딩된 특징을 목표 레이블 공간에 매핑하여 분류 로짓을 산출합니다. 논문에서는 일반적으로 사용되는 비모수적 포지셔널 인코딩을 경량화된 방법에서의 응용도 다룹니다.

- **Performance Highlights**: Point-LN은 ModelNet40 및 ScanObjectNN과 같은 벤치마크 데이터셋에서 경쟁력 있는 성능을 보여줍니다. 비모수적 방법의 강점을 활용하면서 학습 가능한 분류기를 통합하여 복잡한 시나리오에서도 높은 정확도를 달성할 수 있도록 설계되었습니다. 이 연구는 다양한 포인트 클라우드 분류 작업에 적합하고, 광범위한 컴퓨터 비전 애플리케이션에서의 채택 가능성을 강조합니다.



### Detection and Classification of Acute Lymphoblastic Leukemia Utilizing Deep Transfer Learning (https://arxiv.org/abs/2501.14228)
Comments:
          4 pages, 4 figures, Submitted to UCICS

- **What's New**: 이 연구는 백혈병 진단을 위한 새로운 접근 방식을 제안합니다. 기존의 복잡한 진단 과정이 아닌, 딥 러닝(Deep Learning)을 활용하여 초기 단계에서 질병을 식별할 수 있는 방법을 모색했습니다. 이 방법은 총 네 가지 단계인 Benign, Early, Pre, Pro를 포함합니다.

- **Technical Details**: 연구에서는 두 개의 Convolutional Neural Network (CNN) 모델을 사용했습니다. 첫 번째는 MobileNetV2 모델로, 헤드를 수정하여 활용하였고, 두 번째는 여러 convolutional layers로 구성된 커스텀 모델입니다. 커스텀 모델은 각 층에 최대 풀링(max pooling)을 결합하여 설계되었습니다.

- **Performance Highlights**: 커스텀 모델은 98.6%의 정확도를 달성했으며, MobileNetV2는 99.69%로 더 높은 정확도를 기록했습니다. 사전 훈련된(pretrained) 모델 또한 유망한 결과를 보여 실제 임상 적용 가능성을 높이는 것으로 판단됩니다.



### Multi-agent KTO: Reinforcing Strategic Interactions of Large Language Model in Language Gam (https://arxiv.org/abs/2501.14225)
Comments:
          Preprint. Code and data will be available at this https URL

- **What's New**: 이번 연구는 인공지능 일반 지능(AGI)의 실현을 위해 AI 에이전트가 전략적 의사결정을 내리고 유의미한 소통을 할 수 있도록 하는 새로운 접근법을 제안합니다. 특히, 비트겐슈타인의 언어 게임 이론에 영감을 받아, 전통적인 다단계 프로세스를 탈피하여 함수적 상호작용을 통한 학습을 강조합니다. 연구에서는 사회적 추론 게임인 'Werewolf'를 통해 새로운 다중 에이전트 최적화 모델인 MaKTO를 개발하여, AI의 의사결정 및 자연어 생성 능력을 강화하였습니다.

- **Technical Details**: MaKTO의 핵심은 세 가지 주요 혁신 요소에 있습니다. 첫째, 행동 클로닝(Behavior Cloning) 기법을 통해 게임의 전문 용어 및 전략 자료를 기반으로 모델의 학습을 지원합니다. 둘째, 여러 모델 간의 상호작용을 통해 전략 고정화를 방지하고 일반화 능력을 향상시키며, 셋째, 단계적 피드백을 통해 더 정교한 최적화를 위한 선택 방법을 도입하여 승패 이상의 복잡한 게임 작업에 대응하도록 설계되었습니다.

- **Performance Highlights**: 구현된 MaKTO는 9인 플레이어가 참가하는 'Seer-Witch-Guard' 게임에서 61%의 평균 승률을 기록하며, 기존의 GPT-4o 및 RL 기반 모델보다 23.0% 및 10.9% 향상된 성능을 보였습니다. 또한, 인간 전문가와의 대결에서 60%의 승률을 달성하여 인간과 거의 구별할 수 없는 수준의 대화 스타일을 나타내었으며, 판별 테스트에서 48.9% 정확도로 AI 식별이 어려웠습니다. 이러한 결과는 MaKTO가 보다 폭넓은 사회적 추론 게임 및 다자간 협상 시나리오로 확장될 가능성을 내포하고 있습니다.



### TFG-Flow: Training-free Guidance in Multimodal Generative Flow (https://arxiv.org/abs/2501.14216)
- **What's New**: 본 논문에서는 TFG-Flow라는 새로운 training-free guidance 방법을 소개합니다. TFG-Flow는 멀티모달 (multimodal) 생성 흐름을 위한 방법으로, 기존의 방법들이 연속적인 데이터만을 처리하는 한계를 극복합니다. 이 방법은 생성 모델을 다양한 결과로 이끌 수 있는 효율적인 기술로 주목받고 있으며, 선택적으로 분리된 연속 및 이산 변수를 안내할 수 있습니다.

- **Technical Details**: TFG-Flow는 확률 공간에서 분자의 표현을 정의하고, 예측 대상 속성 (target property)에 대해 유효한 분포를 생성하는 수학적 기초를 제공합니다. 본 연구에서는 TFG-Flow가 데이터를 생성하는 과정에서 발생할 수 있는 차원의 저주 (curse-of-dimensionality)를 해결하며, 이산 변수(주로 화합물의 속성)에서의 편향 없는 샘플링 (unbiased sampling) 속성을 유지합니다. 또한, 흐름 마진 (flow marginals) 보존, 목표 예측기와의 정렬, 그리고 이력과 목표의 조건부 독립을 보장하는 이론적인 결과를 제시합니다.

- **Performance Highlights**: TFG-Flow의 성능은 4개의 분자 설계 과제를 통해 검증되었습니다. 실험 결과, TFG-Flow는 원하는 특성을 가진 분자를 효과적으로 생성할 수 있는 잠재력을 보여주었으며, 이는 약물 설계 (drug design) 분야에서도 유망한 성과로 평가됩니다. 기존의 방법들과 비교했을 때, TFG-Flow는 더욱 유연하고 강력한 생성 모델을 가능하게 합니다.



### PuzzleGPT: Emulating Human Puzzle-Solving Ability for Time and Location Prediction (https://arxiv.org/abs/2501.14210)
Comments:
          NAACL 2025 Findings

- **What's New**: 이 논문에서는 이미지로부터 시간과 장소를 예측하는 복잡한 과제를 인간과 같은 퍼즐 해결 능력으로 형식화하고, 이것을 다양한 모듈들로 구현한 전문가 파이프라인인 PuzzleGPT를 제안합니다. PuzzleGPT는 시각적 단서를 식별하는 perceiver, 예측 후보를 추론하는 reasoner, 다양한 단서를 결합하는 combiner, 외부 지식을 검색하는 web retriever, 및 강건성을 위한 noise filter로 구성됩니다. 이는 최첨단 성능을 기록하며, 기존의 대형 VLM(Visual-Language Models)와 자동 생성된 추론 파이프라인에 비해 최소 32%에서 38%의 성능 향상을 보여줍니다.

- **Technical Details**: PuzzleGPT는 다섯 가지 핵심 기술인 perceiver, reasoner, combiner, noise filter, knowledge retriever로 구성됩니다. 이 중 perceiver는 시각 신호를 처리하고 개체를 식별하며, reasoner는 위치 및 시간 후보와 그들의 관계를 추론합니다. 여러 개체에서 얻은 단서를 효율적으로 결합하기 위해 신뢰도 기반의 계층적 조합 방법을 제안하여, 단순히 모든 단서를 결합하는 것이 아니라 각각의 단서를 점진적으로 분석하게 됩니다. 이러한 설계는 추론 과정을 신뢰할 수 있도록 하고 강건성을 추가합니다.

- **Performance Highlights**: PuzzleGPT는 TARA 데이터셋에서 기존의 VLM 모델들, 박사리 BLIP-2 및 GPT-4V를 포함하여 최소 32%의 성능 향상을 보이며, 이는 다양한 기술들을 동시에 활용할 수 없는 기존 모델들의 한계를 드러냅니다. 또한 PuzzleGPT는 WikiTilo 데이터셋에서도 최첨단 성능을 자랑하며, 이를 통해 복잡한 문제 해결을 위한 전문가 설계의 중요성을 강조합니다. 이 방법은 이미지에서 시간과 장소를 예측하는 인간의 퍼즐 해결 능력을 시뮬레이션할 수 있는 기초를 제공합니다.



### Dynamic Token Reduction during Generation for Vision Language Models (https://arxiv.org/abs/2501.14204)
- **What's New**: 이번 논문에서는 시각-언어 모델(VLM)을 위한 동적 프루닝 전략인 Dynamic Rate(DyRate)를 소개합니다. 이 접근 방식은 생성 과정에서 압축 비율을 점진적으로 조정하여 계산 복잡성을 줄이는 동시에 응답의 품질을 유지하는 데 중점을 두고 있습니다. 기존 연구들이 단일 전방 패스에서 시각적 토큰의 중복성을 제거하는 데 집중했던 반면, DyRate는 전체 생성 과정에서 시각적 토큰의 중요성을 면밀히 분석하여 동적으로 조정되는 압축 비율을 적용했습니다.

- **Technical Details**: DyRate는 응답 생성을 위한 토큰 축소 비율을 주의 분포(attention distribution)와 연결하여 적응형 동적 축소를 구현한 첫 번째 방법입니다. 이 모델은 경량 분류기를 사용하여 각 생성 단계에서 시각적 토큰들의 주의 분포를 수집하고, 이를 통해 최적의 축소 비율을 예측합니다. 또한, Gumbel-Softmax 기법을 사용하여 예측된 압축 비율을 미분 가능하게 만들고, 모델의 순전파 과정에 통합하여 더 나은 학습 가능성을 제공합니다.

- **Performance Highlights**: 실험 결과, DyRate는 계산 요구 사항을 크게 줄이면서도 응답의 정확성을 유지하는 데 성공했습니다. 분석 결과, 생성 과정이 진행될수록 시각적 토큰의 중요성이 감소함을 확인하고, 이러한 중요성의 변화를 반영하여 동적 압축 비율을 사용하는 것이 효과적임을 입증했습니다. 이는 VLM의 실제 적용 범위를 확장하는 데 기여할 것으로 기대됩니다.



### Coordinating Ride-Pooling with Public Transit using Reward-Guided Conservative Q-Learning: An Offline Training and Online Fine-Tuning Reinforcement Learning Framework (https://arxiv.org/abs/2501.14199)
- **What's New**: 이 논문은 다중 모달 교통 네트워크 내에서 라이드 풀링(ride-pooling)과 대중 교통(public transit) 간의 조정을 개선하기 위한 새로운 강화 학습(RL) 프레임워크인 Reward-Guided Conservative Q-learning(RG-CQL)을 소개합니다. 연구진은 각 라이드 풀링 차량을 마르코프 의사결정 과정(Markov Decision Process, MDP)으로 모델링하고, 최적의 운영 결정을 학습하기 위해 오프라인 훈련(offline training) 및 온라인 미세 조정(online fine-tuning) RL 프레임워크를 제안합니다.

- **Technical Details**: 오프라인 훈련 단계에서는 Conservative Double Deep Q Network(CDDQN)를 액션 실행기(action executor)로 활용하고, 보상 추정기를 위한 지도 학습 기반의 가이더 네트워크(Guider Network)를 개발하여 데이터 배치에서 액션-보상 관계에 대한 유용한 통찰을 추출합니다. 온라인 미세 조정 단계에서는 가이더 네트워크가 탐색 가이드(exploration guide) 역할을 하여 CDDQN이 알려지지 않은 상태-액션 쌍을 효과적이고 보수적으로 탐험하도록 돕습니다.

- **Performance Highlights**: 현실적 사례 연구를 통해 맨해튼의 실제 데이터를 사용하여 알고리즘의 효능이 증명되었습니다. 라이드 풀링을 대중 교통과 통합한 결과, 단독 라이드와 대중 교통 조정, 라이드 풀링과 대중 교통 없이 조정된 두 가지 벤치마크 사례보다 각각 17% 및 22% 더 나은 시스템 보상을 달성했습니다. 또한, 혁신적인 오프라인 훈련 및 온라인 미세 조정 프레임워크는 기존 온라인 RL 방법에 비해 데이터 효율성을 81.3% 개선하여 총 보상을 4.3% 증가시키고 과대 추정 오류를 5.6% 줄였습니다.



### ENTER: Event Based Interpretable Reasoning for VideoQA (https://arxiv.org/abs/2501.14194)
- **What's New**: 이번 논문에서는 ENTER라는 해석 가능한 Video Question Answering (VideoQA) 시스템을 제안합니다. 이 시스템은 이벤트 그래프(event graphs)를 기반으로 하여 비디오를 그래픽 표현으로 변환하고, 이 과정에서 이벤트 간의 관계를 명확히 모델링합니다. ENTER는 기존 시스템들이 간과한 저수준의 시각 정보(contextual visual information)를 활용하여 더욱 신뢰할 수 있는 해답을 제공합니다. 실험 결과, ENTER는 기존의 해석 가능한 VideoQA 접근법보다 우수한 성능을 보여줍니다.

- **Technical Details**: ENTER는 이벤트 그래프라는 구조화된 표현을 사용하여 비디오 이벤트를 노드로, 이벤트 간 관계를 엣지로 구성합니다. 이를 통해 모델은 사건 간의 관계를 명확히 이해할 수 있으며, 대규모 언어 모델(LLM)을 통해 생성된 코드를 실행하여 질문에 대한 응답을 얻습니다. 이 과정에서는 초기 그래프의 불완전함을 보완하기 위해 계층적인 반복 업데이트(hierarchical iterative update)를 적용하여 추가 정보를 통합하고 그래프의 완전성을 향상시킵니다.

- **Performance Highlights**: ENTER는 NeXT-QA, IntentQA, EgoSchema와 같은 비디오 QA 데이터셋에서 최첨단 성능을 상회하는 결과를 달성했습니다. 실험을 통해 ENTER의 구조화된 이벤트 그래프가 비디오 내에서의 복잡한 관계를 명확히 포착하고 더 정확하며 해석 가능한 질문 응답을 가능하게 한다는 것을 입증했습니다. 이 접근 방식은 오류를 디버깅할 때 더욱 집중할 수 있게 해주며, 오류 원인을 쉽게 식별할 수 있도록 돕습니다.



### VarDrop: Enhancing Training Efficiency by Reducing Variate Redundancy in Periodic Time Series Forecasting (https://arxiv.org/abs/2501.14183)
- **What's New**: 이번 논문에서는 multivariate time series forecasting의 효율성을 높이기 위해 VarDrop이라는 새로운 전략을 제안합니다. VarDrop은 훈련 중 중복된 variate tokens를 생략하여 계산 비용을 줄입니다. 특히, k-dominant frequency hashing(k-DFH)를 활용하여 각 variate의 주파수 도메인에서 상관관계가 높은 토큰을 그룹화합니다. 이를 통해 기존 방법론보다 우수한 성능을 보여주는 것이 목표입니다.

- **Technical Details**: VarDrop 전략은 주어진 multivariate time series에 대해 fast Fourier transform을 수행하여 k-dominant 주파수를 식별합니다. 각 variate는 amplitude가 높은 k개의 주파수에 대한 해시값으로 그룹화되고, 계층화된 샘플링을 통해 대표적인 variate token이 선택됩니다. 선택된 token으로 sparse attention을 수행하여 계산 비용을 크게 줄일 수 있으며, 이 과정에서 중요한 정보를 보존합니다.

- **Performance Highlights**: 공식 벤치마크 데이터셋에서 실시한 실험 결과, VarDrop은 기존의 효율적인 기준 모델들을 초월하는 성능을 보였습니다. 다양한 데이터셋에서 실험을 통해 VarDrop의 적용 가능성과 실효성을 입증하였으며, 이를 통해 시간 시계열 예측의 정확성을 높이는 데 기여할 것으로 기대됩니다.



### RL + Transformer = A General-Purpose Problem Solver (https://arxiv.org/abs/2501.14176)
- **What's New**: 이 논문에서는 기존의 모델이 훈련된 내용을 초월하여 새로운 문제를 해결하는 능력을 갖추는 메타 학습(meta-learning) 방법으로, In-Context Reinforcement Learning (ICRL)이라는 emergent ability를 소개합니다. 특히, Llama 3.1 8B 모델이 RL 목표를 통해 스스로 학습하며, 훈련 데이터의 다양성에 강건함을 보여줍니다. 이 연구는 대규모 언어 모델이 비정상적인 문제를 해결할 수 있는 가능성을 제시하며, AI 시스템의 범용문제 해결 능력을 향상시킬 수 있음을 강조합니다.

- **Technical Details**: 연구에서는 LLaMA 3.1 8B Instruct라는 오픈 소스 대규모 언어 모델을 활용하여 ICRL의 기능을 탐구합니다. 모델은 여러 에피소드를 통해 강화 학습(reinforcement learning)으로 미세 조정(fine-tuning)되며, 이를 통해 새로운 문제를 해결하는 능력을 개발합니다. 이러한 접근 방식은 RL이 직면하는 샘플 효율성(sample efficiency) 문제를 해결하는 데 중점을 두며, 모델이 복잡한 작업을 수행하면서 학습한 기술을 조합할 수 있도록 합니다.

- **Performance Highlights**: ICRL로 훈련된 변환기(transformer)는 유사한 환경에서 높은 샘플 효율성을 나타내며, 비정상적(non-stationary) 환경에서도 강력한 성능을 유지합니다. 실험 결과, 모델은 높은 훈련 데이터 품질에 대한 민감도를 감소시키며, 최근 상호작용을 우선시하여 신속하게 적응합니다. ICRL의 도입으로 AI 시스템의 인간과 유사한 적응력을 지닌 발전이 가능하다는 것을 입증하며, 이는 AI 분야의 패러다임 전환을 의미합니다.



### Dreamweaver: Learning Compositional World Representations from Pixels (https://arxiv.org/abs/2501.14174)
- **What's New**: 이번 연구에서는 Dreamweaver라는 신경망 아키텍처를 제안하여 원시 비디오에서 계층적이고 조합적인 표현을 발견하고 조합된 미래 시뮬레이션을 생성합니다. 이는 전통적으로 언어와 같은 보조 데이터 없이 비디오를 모델링하는 데 어려움이 있었던 AI 시스템의 한계를 극복하는 데 기여합니다. 특히, Recurrent Block-Slot Unit (RBSU)를 활용하여 비디오를 구성 요소 객체 및 속성으로 분해하고, 동적 개념을 효과적으로 캡처하기 위한 다중 미래 프레임 예측 목표를 사용합니다.

- **Technical Details**: Dreamweaver 모델은 T 개의 과거 이미지를 인코딩하는 RBSU를 포함합니다. 이 RBSU는 독립적으로 업데이트되는 슬롯 상태 집합으로 상태를 표현하며, 이는 나중에 혼합된 몬올리식 슬롯 상태를 독립적인 블록 조합으로 매핑하는 블록- 슬롯 병목을 거칩니다. 각 단계에서 개념 프로토타입 라이브러리에 대한 주의를 통해 블록 벡터가 값을 취득하도록 하여 동적 개념 추상화를 생성하는 데 중요한 예측 재구성 목표를 설정합니다.

- **Performance Highlights**: 실험 결과, Dreamweaver는 여러 데이터 세트를 통한 DCI 프레임워크에서 최신 오브젝트 중심 방법보다 우수한 성능을 보여주었습니다. 또한, RBSU의 모듈화된 개념 표현이 조합적 상상력을 가능하게 하여 서로 다른 객체에서 속성을 재조합하여 새로운 비디오를 생성하는 능력을 입증했습니다. 이 연구는 언어에 의존하지 않고 독창적인 비디오를 생성하는 길로 나아가는 중요한 첫걸음을 제시합니다.



### UltraLightSqueezeNet: A Deep Learning Architecture for Malaria Classification with up to 54x fewer trainable parameters for resource constrained devices (https://arxiv.org/abs/2501.14172)
- **What's New**: 말라리아 진단을 위한 경량 딥러닝 접근 방식이 주목받고 있습니다. 본 연구에서는 저자원 환경에서 진단 개선 가능성을 가진 SqueezeNet1.1을 선택했습니다. 이는 SqueezeNet1.0의 개선된 버전으로, 원래 모델보다 2.4배 더 효율적입니다.

- **Technical Details**: 세 가지 초경량 아키텍처 변형을 SqueezeNet1.1에 제안하였습니다. 변형 1(모듈 1개), 변형 2(모듈 2개), 변형 3(모듈 4개)으로, 이들은 SqueezeNetV1.1(모듈 8개)보다 더 컴팩트합니다. NIH Malaria 데이터셋을 활용하여 각 모델의 성능을 정확도, 재현율, 정밀도, F1-score 및 AUC(곡선 아래 면적)로 평가하였습니다.

- **Performance Highlights**: SqueezeNet1.1 모델은 모든 메트릭에서 97.12%의 분류 정확도로 최고의 성능을 보였습니다. 변형 3(모듈 4개)은 96.55%의 정확도로 거의 동일한 결과를 제공하면서도 계산 오버헤드를 6배 줄였습니다. 변형 2는 28배, 변형 1은 54배의 훈련 가능 파라미터 감소를 보여주었습니다.



### Enhancing Multimodal Entity Linking with Jaccard Distance-based Conditional Contrastive Learning and Contextual Visual Augmentation (https://arxiv.org/abs/2501.14166)
- **What's New**: 이번 연구에서는 Multimodal Entity Linking (MEL)에 대한 기존의 접근 방식과 달리, JD-CCL(Jaccard Distance-based Conditional Contrastive Learning)이라는 새로운 기법을 제안합니다. JD-CCL은 메타 정보를 활용하여 네거티브 샘플을 더욱 정교하게 선택함으로써, 모델의 엔티티 매칭 능력을 강화합니다. 또한, CVaCPT(Contextual Visual-aid Controllable Patch Transform) 모듈을 도입하여 다양한 시각적 표현을 개선하고, 보다 복잡한 특징을 고려할 수 있도록 합니다.

- **Technical Details**: JD-CCL은 메타 속성을 활용하여 유사한 속성을 가진 네거티브 샘플을 선택합니다. 이는 모델이 단순한 속성이 아닌 보다 복잡한 속성을 기반으로 올바른 엔티티를 연결하도록 요구합니다. CVaCPT 모듈은 입력 이미지의 중요한 특징을 개별 화하여, 각각의 엔티티에 맞춘 다양한 시각적 표현을 생성합니다.

- **Performance Highlights**: 실험 결과, WikiDiverse, RichpediaMEL 및 WikiMEL이라는 세 가지 벤치마크 데이터셋에서 제안한 방법이 이전의 최신 기술들에 비해 뛰어난 성능을 보였습니다. 특히, JD-CCL과 CVaCPT는 멀티모달 데이터셋에서 엔티티에 대한 정확한 연결을 강화하는 데 기여했습니다.



### LoCoML: A Framework for Real-World ML Inference Pipelines (https://arxiv.org/abs/2501.14165)
Comments:
          The paper has been accepted for presentation at the 4th International Conference on AI Engineering (CAIN) 2025 co-located with 47th IEEE/ACM International Conference on Software Engineering (ICSE) 2025

- **What's New**: 기계 학습(ML)의 일반적인 채택으로 인해 다양한 아키텍처와 데이터 요구 사항을 가진 모델들이 등장하였고, 이러한 시스템들을 실제 응용 프로그램에 통합하는 데 새로운 도전 과제가 발생했습니다. 전통적인 솔루션은 이질적인 모델을 연결하는 복잡성을 관리하는 데 어려움을 겪고 있으며, 특히 다양한 기술 사양을 가진 모델들이 협력하는 대규모 프로젝트에서 이러한 한계가 더욱 부각됩니다. 이에 대응하기 위해, LoCoML이라는 저코드(low-code) 프레임워크가 개발되어 AI 기반 언어 기술의 통합을 지원하며, 여러 ML 모델을 간소화하여 통합할 수 있는 솔루션을 제공합니다.

- **Technical Details**: LoCoML은 Model Hub와 Pipeline Orchestrator라는 두 가지 주요 구성 요소로 구성되어 있습니다. Model Hub는 시스템에 필요한 모든 ML 모델을 공급하며, Pipeline Orchestrator는 모든 추론 관련 프로세스를 관리합니다. 사용자 역할로는 모델을 개발하는 Model Developer, 파이프라인을 설계하는 Pipeline Designer, 및 파이프라인을 사용하는 Pipeline User가 있으며, 각 역할은 모델을 적절하게 구성하고 통합하는 데 필수적입니다.

- **Performance Highlights**: LoCoML은 초기 평가를 통해 큰 계산적 부담 없이 효율적이고 효과적으로 대규모 ML 통합을 가능하게 한다고 합니다. 이러한 저코드 접근 방식을 통해 다양한 기술 배경을 가진 사용자들이 ML 워크플로우에 기여할 수 있으며, 시스템 개발에 필요한 시간과 전문 지식을 줄일 수 있습니다. Bhashini 프로젝트의 요구 사항과 복잡성을 해결하는 데 있어 LoCoML은 유용한 전략을 제공하며, 다양한 협력자가 모델을 효과적으로 개선할 수 있도록 강조하고 있습니다.



### Advancing MRI Reconstruction: A Systematic Review of Deep Learning and Compressed Sensing Integration (https://arxiv.org/abs/2501.14158)
- **What's New**: 이 논문에서는 자기공명영상(MRI) 재구성에서의 딥러닝(DL) 기술의 발전을 다룬다. MRI는 비침습적(Non-invasive) 이미징 모달리티로, 해부학적 및 기능적 통찰을 제공하지만 길어진 데이터 수집 시간으로 인한 문제들이 존재한다. 본 리뷰는 이러한 문제를 해결하기 위한 DL 기반의 재구성 기법들을 종합적으로 분석하고, 그 잠재적 이점을 강조한다. 특히, DL이 병렬 이미징(Parallel Imaging) 및 압축 센싱(Compressed Sensing)과 결합될 때 더 빠르고 정확한 MRI 재구성이 가능함을 설명한다.

- **Technical Details**: MRI의 전통적인 재구성 방식은 데이터 수집의 비선형성으로 인해 상당한 시간과 비용이 소모된다. 이에 따라, 압축 센싱(CS) 기법은 스파스(sparse) 데이터에서 이미지를 재구성하고, DL 기법은 이런 방식을 보완하여 더욱 정교한 재구성을 가능하게 한다. 특히, DL 기법은 훈련 데이터를 통해 스파스 이미지를 전혀 다른 응축(maps)으로 변환하여, 스캔 시간을 단축하고 다양한 아티팩트를 줄이는 데 기여한다. 여러 구조의 DL 접근 방식이 있으며, 이 과정에서 병렬 이미징을 통한 SNR 개선과 그에 따른 가속화 메커니즘을 확인할 수 있다.

- **Performance Highlights**: DL 기반의 MRI 재구성 기법들은 빠른 재구성 시간을 자랑하며, 다양한 아티팩트와 노이즈를 효과적으로 완화하는 성능을 발휘한다. 객관적인 성과 지표 및 측정 기준을 토대로 한 여러 데이터셋에 대한 연구 결과와 관심이 증가하고 있으며, 이는 MRI 재구성 기술의 발전을 이끄는 주요 동력이 되고 있다. 또한, 미래 연구 방향을 논의하여 DL 기반 MRI 재구성의 중요성을 피력하며, 이를 통해 의료 이미징의 발전을 도모할 수 있는 방안을 제시한다.



### Reinforcement Learning Platform for Adversarial Black-box Attacks with Custom Distortion Filters (https://arxiv.org/abs/2501.14122)
Comments:
          Under Review for 2025 AAAI Conference on Artificial Intelligence Proceedings

- **What's New**: 이 논문에서는 RLAB라는 새로운 강화 학습 플랫폼을 소개합니다. 이 플랫폼은 사용자가 서로 다른 왜곡 필터(distortion filters)를 선택하여 적대적 사례(adversarial examples)를 생성할 수 있도록 설계되었습니다. RLAB는 입력 이미지에 최소한의 왜곡을 추가하면서도 목표 모델을 오분류시키기 위해 강화 학습 에이전트를 활용합니다.

- **Technical Details**: RLAB는 입력 이미지를 각 단계에서 탐색하여 왜곡을 추가해야 할 민감한 영역을 식별합니다. 이를 위해 새로운 이중 행동(dual-action) 방법을 사용하여 목표 모델에 미치는 영향이 적은 노이즈를 제거합니다. 이러한 방식은 공격의 수렴(convergence)을 더 빠르고 효율적으로 만들어줍니다.

- **Performance Highlights**: 또한 RLAB는 특정 왜곡 유형에 대한 이미지 분류 모델의 견고성을 측정하는 데에도 사용될 수 있습니다. 적대적 샘플을 사용하여 모델을 재훈련(retraining)함으로써 벤치마크 데이터셋에서 평가할 때 견고성이 눈에 띄게 향상되었습니다. 제안된 플랫폼은 오분류를 유발하는 데 필요한 쿼리 수에서 최신 기술(state-of-the-art) 방법들을 초월했습니다.



### On the Transfer of Knowledge in Quantum Algorithms (https://arxiv.org/abs/2501.14120)
Comments:
          12 pages, 8 figures, 4 tables. Paper submitted for its review in Expert Systems journal

- **What's New**: 본 논문은 기존의 인공지능에서 사용되던 transfer of knowledge 기법을 양자 컴퓨팅(quantum computing)에 통합하는 방법을 탐구합니다. 이렇게 함으로써 양자 알고리즘의 효율성과 효과성을 증가시킬 수 있는 가능성을 제시합니다.

- **Technical Details**: Transfer Learning 및 Transfer Optimization에 중점을 두고 transfer models의 종합적인 분류를 제공합니다. 이러한 모델들은 양자 컴퓨팅에서 지식 공유(knowledge sharing)의 혜택을 받을 수 있는 관련 스킴(schemes)과 이론적인 통찰력을 기반으로 한 초기 실험 결과를 포함합니다.

- **Performance Highlights**: 지식 전달(transfer of knowledge)을 활용함으로써 하이브리드 솔버(hybrid solvers)와 관련된 양자 알고리즘의 최적화를 가속화하고 양자 프로세서의 계산 부담(computational burden)을 줄이는 효과를 기대할 수 있습니다. 이러한 접근 방식은 양자 컴퓨팅 기술을 발전시키는데 기여할 수 있는 유용한 도구입니다.



### Autonomous Structural Memory Manipulation for Large Language Models Using Hierarchical Embedding Augmentation (https://arxiv.org/abs/2501.14119)
- **What's New**: 최근 대형 언어 모델(LLMs) 분야에서는 다계층 의미 구조를 통해 토큰의 표현을 재정의하는 계층적 임베딩 증강이 도입되었습니다. 이는 복잡한 언어 입력에 대해 더 나은 적응성을 제공합니다. 또한, 자율 구조 기억 조작은 중요한 맥락적 특징을 우선시하며 덜 관련된 정보를 억제하는 역동적인 기억 재배치 메커니즘을 통해 이러한 패러다임을 더욱 발전시킵니다. 실험 결과, 기억 재조직화 전략을 통해 긴 입력 시퀀스에 대한 처리 오버헤드를 크게 줄이는 등의 계산 효율성이 크게 향상되었습니다.

- **Technical Details**: 제안된 방법론은 계층적 임베딩 증강과 자율 구조 기억 조작을 결합한 새로운 접근 방식을 소개합니다. 이 방법은 각 임베딩 레이어를 가중치 조합을 통해 다계층 구조로 인코딩하여 토큰 표현을 변형합니다. 이러한 구성을 통해 모델은 중요한 맥락 요소를 우선적으로 분석하고, 덜 중요한 정보를 효율적으로 버릴 수 있어 태스크 일반화와 투명성 모두를 개선합니다. 이 방식은 고유한 구조적 향상을 통해 모델의 의사결정 경로를 한층 더 이해하기 쉽게 만듭니다.

- **Performance Highlights**: 제안된 방법은 기존 LLM 아키텍처의 정적 토큰 표현의 한계를 해결하여 다양한 태스크에서의 적용 가능성을 높였습니다. 비교 분석 결과, 정확도, 효율성 및 해석 가능성 측면에서 특정 이점을 보여주었으며, 특히 복잡한 맥락적 이해가 요구되는 업무에서 두드러진 성과를 나타냈습니다. 이 연구는 자율적 메모리 재구성을 통한 실시간 메모리 관리를 통해 다영역 일반화 및 실시간 의사결정 시스템 등 여러 응용 분야에서의 강력함을 인증합니다.



### MedSlice: Fine-Tuned Large Language Models for Secure Clinical Note Sectioning (https://arxiv.org/abs/2501.14105)
Comments:
          Our code is publicly available on github ( this https URL )

- **What's New**: 이 연구는 공개 소스의 큰 언어 모델(Large Language Models, LLMs)을 활용하여 임상 노트를 자동으로 구분하는 방법론을 제시합니다. 본 연구는 특히 현재 질병의 역사, 구간 역사 및 평가 및 계획의 세 가지 섹션에 중점을 두어, 487개의 진행 노트에 대한 세심한 데이터셋을 사용하여 모델을 개선하고 평가했습니다. 연구 결과, 미세 조정된 Llama 3.1 8B 모델이 GPT-4o를 초과하는 성능(F1=0.92)을 나타내며, 이는 병원과 임상 분야에서의 접근성과 비용 효율성을 크게 향상시킬 수 있는 가능성을 보여줍니다.

- **Technical Details**: 임상 노트는 전문의의 직간접적인 관찰과 평가를 기록하는 반면, 이러한 노트를 효과적으로 분류하는 것은 어려움이 많습니다. 연구진은 Clinical-Longformer와 같은 다양한 기계 학습 모델을 사용하여 모델 성능을 평가했으며, 임상 노트에서 특정 구간을 추출하기 위한 미세 조정된 LLM을 훈련했습니다. 최종 데이터셋은 1,147개의 노트로 구성되며, 미세 조정 과정에서는 rank-stabilized LoRA 방법이 사용되었습니다.

- **Performance Highlights**: 최적화된 Llama 3.1 8B 모델은 F1 점수 0.92를 기록하며, 낮은 비용의 소스 모델로서 현대의 의료 시스템에서 발전된 성과를 달성했습니다. 외적 유효성 테스트 세트에서도 F1 점수 0.85를 유지하며 높은 성능을 보여주었습니다. 이러한 결과는 임상 노트를 구조화하여 후속 분석을 수행하는 데 있어 큰 장점을 제공할 것입니다.



### The Role of Generative AI in Software Student CollaborAItion (https://arxiv.org/abs/2501.14084)
Comments:
          7 pages, 1 figure

- **What's New**: 최근 AI의 발전은 컴퓨터 교육에서 협력의 방식에 큰 변화를 가져올 것으로 예상됩니다. 이 연구에서는 AI 에이전트가 협력 과정에서 다양한 역할을 맡을 수 있는 시나리오를 제시합니다. 특히 소프트웨어 개발에서의 협력의 중요성에 주목하며 AI가 이러한 역할을 어떻게 지원할 수 있을지에 대해 논의합니다. 또한 AI 시대의 컴퓨터 교육에 대한 새로운 가능성과 도전을 설계 허구(design fiction)를 통해 보여줍니다.

- **Technical Details**: AI 에이전트는 팀원, 멘토, 보조자 등의 역할을 맡아 소프트웨어 개발 과정에서 인간과 협력할 수 있는 역량을 갖출 것으로 예상됩니다. 협력의 여러 분류는 각 참여자가 특정 역할을 맡고, 다양한 매체와 모드가 활용되는 구조를 가지고 있습니다. AI 에이전트가 다양한 역할을 잘 수행할 수 있도록 훈련시키는 것이 그들의 효과적인 협업을 위해 필요합니다. 특히, 교육적 환경에서 AI 에이전트의 역할 설정에 관한 논의가 중요합니다.

- **Performance Highlights**: AI 에이전트가 소프트웨어 개발 팀에서 중요한 역할을 맡으면서 학생들의 협업 능력이 더욱 중요해질 것입니다. 현재 여러 연구에서 AI를 활용한 프로그래밍 협업 및 동료 교육 등의 모델이 탐색되고 있습니다. 하지만 AI와의 협력이 감정적인 요소가 결여될 수 있으며, 인간과의 상호작용에서 느껴지는 정서적 긴장감은 결여될 것이라는 점도 고려해야 합니다. 이러한 점들은 교육 현장에서 AI를 효과적으로 활용하는 데 필수적인 요소가 될 것입니다.



### Communicating Activations Between Language Model Agents (https://arxiv.org/abs/2501.14082)
- **What's New**: 본 연구는 다중 언어 모델 간의 효율적인 통신을 위한 새로운 접근 방식을 제안합니다. 기존의 자연어 대신 활성화(activations)를 통해 모델 간 통신을 함으로써, 추론 비용을 대폭 절감하고 높은 정보 밀도를 유지할 수 있습니다. 이러한 대안적 언어를 통해 LLM(대형 언어 모델) 간의 재생성 능력을 극대화할 수 있음을 보입니다.

- **Technical Details**: 구체적으로, 모델 B의 계산을 중간 레이어에서 일시 정지하고, B의 활성화와 다른 모델 A의 활성화를 함수 f를 통해 결합합니다. 이후, 이 출력을 B의 다음 레이어로 전달하며 계산을 계속 진행합니다. 이 방식은 새로운 작업에서 제로 추가 파라미터로 LLM을 확장할 수 있도록 하며, 다양한 도메인과 환경에 적용할 수 있습니다.

- **Performance Highlights**: 실험을 통해 이 방법은 전통적인 자연어 통신 대비 최대 27.0%의 성능 향상을 이루었으며, 계산 비용은 1/4 미만으로 감소했습니다. 또한, 본 연구는 LLM의 다양한 크기와 스위트에 걸쳐 일반화 가능성을 보여주어, 소형 LLM도 통신의 이점을 활용할 수 있음을 입증했습니다.



### Expanding on the BRIAR Dataset: A Comprehensive Whole Body Biometric Recognition Resource at Extreme Distances and Real-World Scenarios (Collections 1-4) (https://arxiv.org/abs/2501.14070)
Comments:
          10 pages, 11 figures, 2 tables, submitted to CVPR

- **What's New**: 최근 생체 인식 알고리즘과 시스템의 발전은 많은 진전을 이루어냈지만, 극단적인 거리와 고도에서의 비전통적인 환경에서의 적용은 여전히 도전 과제입니다. 이 논문은 이러한 운영상의 도전에 대응하는 대규모 데이터세트의 확장을 요약하며, 그 구성과 데이터 수집, 관리 및 주석 달기 방법론을 설명합니다.

- **Technical Details**: BRIAR 프로그램은 어려운 조건에서 생체 인식 기술을 발전시키기 위해 고안된 미국 정부 지원의 프로그램으로, 기후와 환경이 다양한 3지역에서 1,760명의 피험자를 대상으로 475,000장의 이미지와 3,450시간의 영상을 포함하는 데이터세트를 구축했습니다. 이 데이터세트는 외부 및 실내 환경에서 다양한 활동을 수행하는 참가자들을 포함하며, 고해상도 이미지와 영상 데이터를 제공합니다.

- **Performance Highlights**: BRIAR 데이터세트는 생체 인식 및 열기 완화 관련 100편 이상의 논문에서 중요한 기준으로 사용되었으며, 연구자들이 해당 데이터에 접근할 수 있는 경로를 제공합니다. 이 연구는 향후 데이터 추가 및 품질 향상을 위한 지속적인 노력을 강조하고 있으며, 인종, 성별 등의 다양성을 고려한 데이터 확장을 통해 인식 모델의 공정성과 견고성을 보장하는 데 중점을 두고 있습니다.



### Revisiting CLIP: Efficient Alignment of 3D MRI and Tabular Data using Domain-Specific Foundation Models (https://arxiv.org/abs/2501.14051)
Comments:
          10 pages, 2 figures. To be published in ISBI 2025

- **What's New**: 이 논문에서는 CLIP 훈련을 3D 뇌 MRI에 처음으로 응용하고, 도메인 специф적인 3D 기반 모델 이미지 인코더를 훈련하여 모드 간 정렬이 소규모 데이터셋에서도 가능함을 보여줍니다. 또한, 3D에서의 훈련 시 배치 간의 임베딩을 누적하여 대비 손실을 안정화하는 방법을 제안했습니다. 이를 통해 3D MRI와 표 형식 데이터 간의 의미 있는 정렬이 가능함을 입증하였습니다.

- **Technical Details**: 주요 기술적 요소로는 제한된 크기의 데이터셋을 다룰 때 각 모드의 의미 있는 표현을 동시에 학습할 수 없다는 관찰이 있습니다. 이 문제를 해결하기 위해 우선 각 모드의 구체적인 인코더를 학습하고 이후 이를 일관된 임베딩 공간에 정렬하는 접근법을 사용합니다. 3D 뇌 MRI를 위한 인코더는 공공 MRI 데이터셋과 AMAES 프레임워크를 이용하여 대규모 프리트레이닝을 통해 학습되었습니다.

- **Performance Highlights**: 본 연구에서는 제안된 방법론이 제로샷 분류(zero-shot classification)와 이미지 검색(image-retrieval) 과제에서 평가되며, 특히 제로샷 분류에서는 3D MRI와 표 형식 데이터 간의 의미 있는 정렬을 보여주는 결과를 얻었습니다. 그러나 제로샷 이미지 검색은 여전히 도전적임을 발견하였습니다.



### GraphRAG under Fir (https://arxiv.org/abs/2501.14050)
Comments:
          13 pages

- **What's New**: 이 연구에서는 그래프 기반의 RAG(검색 증강 생성)인 GraphRAG의 독특한 공격 취약성을 탐구합니다. GraphRAG는 외부 지식을 다중 스케일의 지식 그래프로 변환하여 언어 모델이 더욱 세부적이고 폭넓은 맥락을 통합할 수 있도록 지원합니다. 기존 RAG 모델보다 GraphRAG가 단순한 공격에 더 높은 내성을 보이는 것을 발견했습니다. 그러나 이러한 기능들이 새로운 공격 표면을 창출한다는 점도 주목할 만합니다.

- **Technical Details**: GraphRAG은 지식 그래프를 사용하여 외부 지식을 조직하며, 질의마다 관련 정보를 검색하여 이를 프롬프트에 추가하고 응답을 생성합니다. 이 과정에서 관계 주입, 관계 증대 및 내러티브 생성의 세 가지 주요 전략을 사용하여 악의적인 콘텐츠가 포함된 오염 텍스트를 생성합니다. 특히, GragPoison이라는 새로운 공격 기법이 제안되며, 이는 관계를 기반으로 여러 질의를 동시에 타겟팅하는 독특한 접근 방식을 채택합니다.

- **Performance Highlights**: GragPoison의 실증적 평가 결과, 기존의 공격 방법들에 비해 98%의 성공률을 기록하며 공격의 효과성과 확장성을 크게 향상시켰습니다. 여러 GraphRAG 모델과 다양한 데이터셋을 통해 이러한 성능이 입증되었으며, GragPoison은 기존 방어에 대해 강한 저항력을 보입니다. 또한, 향후 연구 방향으로 새로운 방어 대책의 필요성을 강조합니다.



### SIDDA: SInkhorn Dynamic Domain Adaptation for Image Classification with Equivariant Neural Networks (https://arxiv.org/abs/2501.14048)
Comments:
          25 pages, 5 figures, 4 tables. code available at: this https URL

- **What's New**: 이 논문에서는 Sinkhorn divergence를 기반으로 하는 새로운 도메인 적응 훈련 알고리즘인 SIDDA(SInkhorn Dynamic Domain Adaptation)를 소개합니다. SIDDA는 효율적인 도메인 정렬을 가능하게 하며, 최소한의 하이퍼파라미터 조정과 컴퓨팅 비용으로 사용할 수 있습니다. 이 방법은 다양한 복잡성의 시뮬레이션 및 실제 데이터 세트에 대해 효과성을 입증하였으며, 특히 동등 변환 신경망(equivariant neural networks, ENNs)과 함께 사용할 때 분류 정확도와 모델 보정 효과를 크게 향상시킵니다.

- **Technical Details**: SIDDA는 옵티멀 전송(optimal transport) 이론을 기반으로 하며, 엔트로픽 정규화(entropic regularization)와 분류 손실(classification loss) 및 DA 손실(domain adaptation loss) 토큰의 가중치를 활성화하여 훈련 과정에서 하이퍼파라미터 조정을 최소화합니다. 이 알고리즘은 다양한 NN 아키텍처와 호환되며 복잡성에 따라 조정할 수 있는 우수한 방법을 제공합니다. 연구에서는 ENNs의 강 robustness를 함께 공부하며, SIDDA를 통한 개선 효과도 조사합니다.

- **Performance Highlights**: SIDDA는 레이블이 없는 타겟 데이터에 대해 약 40%의 분류 정확도 향상을 달성하여 NNs의 일반화 능력을 향상하도록 설계되었습니다. 또한, ECE(Expectation Calibration Error)와 Brier 점수(Brier score)에서 10배 이상의 개선 효과를 보이며, 모델 보정에서도 유의미한 결과를 나타냅니다. SIDDA의 자동화된 접근 방식은 다중 데이터 세트 연구 및 높은 일반화 가능한 모델 개발에 기여할 수 있는 잠재력을 지니고 있습니다.



### Leveraging Multiphase CT for Quality Enhancement of Portal Venous CT: Utility for Pancreas Segmentation (https://arxiv.org/abs/2501.14013)
Comments:
          ISBI 2025

- **What's New**: 이번 연구는 복수의 CT 단계를 활용하여 스캔 품질 향상을 시도한 최초의 연구로, 특히 문맥 정맥 단계를 강화하는 데 중점을 두었습니다. 연구진은 3D Progressive Fusion and Non-local (PFNL) 네트워크를 통해 세 가지 저품질 CT 단계(비대조, 동맥, 문맥 정맥)를 입력으로 사용했습니다. 이 방식을 통해 문맥 정맥 CT의 품질을 향상시켜 췌장 분할(segmentation)의 성능을 3% 개선했습니다.

- **Technical Details**: 본 연구는 VinDr-Multiphase 데이터세트를 사용하여 2015년부터 2020년까지 265명의 환자로부터 총 168개의 CT 연구를 분석했습니다. 비대조, 동맥, 문맥 정맥 단계를 포함하는 CT 스캔을 선택하여 이들을 세 가지 저품질 CT 단계로 변환한 후, 3D PFNL 모델을 통한 품질 향상을 시도했습니다. 손실 함수는 L1 재구성 손실과 3D Sobel 엣지 기반 손실의 조합으로 설정되어 구조의 경계 강화를 목표로 했습니다.

- **Performance Highlights**: 최종 결과는 3D PFNL 모델이 문맥 정맥 단계를 개선하여 췌장 분할 정확도를 3% 향상시켰음을 보여줍니다. 이 연구는 CT 스캔 품질 향상 및 췌장 분할에 있어 유의미한 진전을 이루었으며, 다양한 질병의 진단 및 관리에 있어 중요한 기여를 할 것으로 기대됩니다.



### Transfer Learning of Surrogate Models via Domain Affine Transformation Across Synthetic and Real-World Benchmarks (https://arxiv.org/abs/2501.14012)
- **What's New**: 이번 연구에서는 비미분 가능 서러게이트 모델(surrogate model)인 랜덤 포레스트(random forest)를 사용하여, 특정 도메인 간의 아핀 변환(affine transformation)을 가정하고 모델을 전이하는 방법을 제안합니다. 이는 이전 연구에서 다뤘던 미분 가능 모델에 대한 전이 학습(transfer learning) 기법을 확장한 것으로, 적은 양의 데이터를 통해 새로운 작업에 대한 서러게이트 모델을 구축할 수 있게 합니다.

- **Technical Details**: 연구의 핵심은 도메인 시프트(domain shift) 문제를 해결하고자 하는 것으로, 입력 변수의 분포가 두 작업 간에 이동할 때, 예측 분포는 동일하다는 가정 하에 모델을 전이하는 방법에 집중합니다. 특히 아핀 전이의 경우, 새로운 작업의 입력 분포가 기존 작업의 아핀 변환으로 설명될 수 있음을 바탕으로, 전이된 모델의 성능을 향상시키는 접근 방식이 개발되었습니다. 연구진은 실험을 통해 이 방법이 실제 복잡한 문제에 효과적임을 입증했습니다.

- **Performance Highlights**: 제안된 방법은 Black-Box Optimization Benchmark (BBOB) 테스트베드와 네 개의 실제 전이 학습 문제에 대해 평가되었으며, 모델 훈련에서 필요한 데이터의 양과 계산 비용을 획기적으로 줄이는 성과를 나타냈습니다. 연구 결과, 전이된 모델은 처음부터 훈련한 모델에 비해 훨씬 낮은 대칭 평균 절대 백분율 오류(symmetric mean absolute percentage error, SMAPE)를 달성하였으며, 이는 비미분 가능 서러게이트 모델로서의 장점을 입증합니다.



### Scalable and Explainable Verification of Image-based Neural Network Controllers for Autonomous Vehicles (https://arxiv.org/abs/2501.14009)
Comments:
          11 pages, 5 figures

- **What's New**: 본 논문에서는 고차원 입력으로 인한 안전성 및 신뢰성과 관련된 문제들을 해결하기 위해, \

- **Technical Details**: SEVIN(Scalable and Explainable Verification of Image-Based Neural Network Controllers)라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 변분 오토인코더(Variational Autoencoder, VAE)를 활용하여 고차원 이미지를 낮은 차원의 설명 가능한 잠재 공간으로 인코딩합니다. 잠재 변수와 해당 제어 작업을 주석 처리하여 볼록 다면체(convex polytopes)를 생성하여 검증을 위한 구조적인 입력 공간을 정의함으로써 계산 복잡성을 크게 줄이고 확장성을 향상시킵니다.

- **Performance Highlights**: 실험 결과, SEVIN은 이미지 기반 신경망 컨트롤러에 대한 효율적이고 확장 가능한 검증을 달성하고 컨트롤러 행동에 대한 설명 가능한 통찰력을 제공함을 보여줍니다. 이 접근 방식은 안전이 중요한 시스템에서의 형식 검증 기술과 실제 응용 간의 격차를 해소합니다.



### Adaptive Genetic Algorithms for Pulse-Level Quantum Error Mitigation (https://arxiv.org/abs/2501.14007)
Comments:
          21 pages, 11 figures

- **What's New**: 본 논문은 양자 컴퓨팅에서의 노이즈 문제를 해결하기 위한 새로운 적응형 알고리즘을 소개합니다. 이 알고리즘은 회로 게이트를 수정하지 않고도 동적으로 노이즈 조건에 반응하여 펄스 신뢰도를 향상시킵니다. 본 연구는 Grover와 Deutsch-Jozsa 알고리즘에 적용되어 실험적 결과를 보여줍니다.

- **Technical Details**: 제안된 알고리즘은 펄스 레벨( pulse-level )에서의 양자 오류 완화(quantum error mitigation)를 목표로 하며, 다양한 노이즈 소스의 영향을 줄이기 위해 펄스 매개변수(pulse parameters)를 직접적으로 조정합니다. 이 방법은 양자 회로에서 알고리즘의 저항성을 개선하고, 실험 데이터를 통해 효율성을 입증합니다.

- **Performance Highlights**: 실험 결과, 이 펄스 레벨 전략이 양자 회로의 노이즈가 많은 실행 중에도 신뢰도를 증가시키는 유연하고 효율적인 솔루션임을 보여줍니다. 이는 견고한 양자 컴퓨팅을 위한 오류 완화 기법의 발전에 기여하는 중요한 연구로 평가됩니다.



### Asymmetrical Latent Representation for Individual Treatment Effect Modeling (https://arxiv.org/abs/2501.14006)
- **What's New**: 이 논문은 Asymmetrical Latent Representation for Individual Treatment Effect (ALRITE)라는 새로운 CATE(조건부 평균 치료 효과) 추정 방법을 제안합니다. 이 방법은 두 개의 잠재 공간(latent space)을 비대칭적으로 탐색하여 각각의 공간이 제어와 치료 집단에 대한 효과적인 예측을 최적화하도록 설계되었습니다. ALRITE는 또한 이질적 효과(heterogeneous effects)의 추정 정확도를 높일 수 있는 여지를 제공합니다.

- **Technical Details**: ALRITE는 치료 집단(treated population)과 제어 집단(control population) 각각에 대해 잠재 공간을 최적화하는 기법입니다. 이 연구는 인과 추론(causal reasoning) 분야에서 응용할 수 있는 CATE 추정의 이론적 및 실용적 측면을 다룹니다. 적당한 가정 하에, ALRITE는 이질적 효과 추정의 정밀도(Precision of Estimation of Heterogeneous Effects)에 대한 상한선(upper bound)을 인정합니다.

- **Performance Highlights**: ALRITE 접근법은 기존의 최첨단(state-of-the-art) 기법들과 비교하여 실증적으로 성공적으로 검증되었습니다. 이러한 결과는 ALRITE가 치료 효과를 예측하는 데 매우 효과적임을 입증하며, 특히 의료, 사회학, 광고와 같은 다양한 분야에서의 활용 가능성을 제시합니다.



### Device-aware Optical Adversarial Attack for a Portable Projector-camera System (https://arxiv.org/abs/2501.14005)
- **What's New**: 이 논문에서는 기존의 아드바이셔리 라이트 공격 방식의 한계를 극복하여 실제 얼굴 인식 시스템에서의 효율성을 높이는 방법을 제시합니다. 디지털 공격 알고리즘에 기기 인식(adaptation) 요소를 포함시키고 해상도와 색상 조정 같은 요소를 통합하여 디지털에서 물리적 도메인으로의 전환 중 손실을 최소화합니다. 실험 결과, 제안한 알고리즘이 실제 및 스푸프 공격에 대해 탁월한 성능을 발휘함을 보여줍니다.

- **Technical Details**: 본 연구는 모바일 프로젝터-카메라 장비를 사용하여 얼굴 인식 시스템을 공격하기 위한 기기 인식(adaptation) 알고리즘을 제안합니다. 이를 위해 프로젝터와 카메라 간의 색상 및 해상도 불일치 문제를 해결하는 방법을 도입하였고, 이 과정에서 모스웨어(noise)와 같은 요소를 다루기 위해 회색조 모드로 공격 최적화를 비롯한 기술적 접근이 포함됩니다. 이러한 기법은 모든 디지털 공격 알고리즘에 통합될 수 있어, 기존 방법론의 단점을 보완합니다.

- **Performance Highlights**: 실험을 통해 제안된 알고리즘이 다양한 재료로 만들어진 실제 및 스푸프 공격에 대해 높은 물리적 코사인 유사도 점수를 지속적으로 기록함을 확인하였습니다. 평균적으로 디지털 공격에서 물리적 공격으로 전환 시 점수의 감소는 14%에 불과하며, 이는 공격 성공률의 높은 수치를 나타냅니다. 본 연구는 얼굴 인식 시스템에 대한 아드바이셔리 공격의 실용성을 높이는 데 기여할 것으로 기대됩니다.



### ME-CPT: Multi-Task Enhanced Cross-Temporal Point Transformer for Urban 3D Change Detection (https://arxiv.org/abs/2501.14004)
- **What's New**: 이 논문에서는 도시 지역의 3D 변화를 감지하기 위한 새로운 Multi-task Enhanced Cross-temporal Point Transformer (ME-CPT) 네트워크를 제안합니다. 기존의 3D 변화 탐지 방법들이 가지고 있던 문제점들과 더불어, Multi-temporal ALS (Airborne Laser Scanning) 포인트 클라우드를 이용하여 도시 계획 및 비상 관리에서의 필요성을 강조합니다. 또한, 필요한 다양한 데이터셋의 부족이라는 문제를 해결하기 위해 22.5km² 크기의 3D 의미 변화 탐지 데이터셋을 출시했습니다.

- **Technical Details**: ME-CPT는 서로 다른 시간대의 포인트 클라우드 간 공간적 대응관계를 설정하여 변화 특징을 효과적으로 추출합니다. 이 네트워크는 다중 작업 훈련 전략을 통합하여 의미적 특징의 구별력을 향상시키며, 클래스 불균형 문제를 완화합니다. 특히, 크로스-템포럴(point cloud을 이용한 시간 간격) 주의 메커니즘을 활용하여 다중 시간대 포인트 간의 정보 교환을 촉진합니다.

- **Performance Highlights**: 제안된 ME-CPT는 여러 데이터셋에서 기존의 최첨단 방법들과 비교하여 우수한 성능을 보였습니다. 실험 결과, 이 네트워크가 복잡한 도시 환경에서의 3D 변화 탐지 알고리즘 성능을 제고함을 확인할 수 있었습니다. 연구에서는 다양한 시나리오에서의 적용 가능성을 논의하며, 향후 연구 방향에 있어서의 기초 자료로 활용될 수 있습니다.



### PaMMA-Net: Plasmas magnetic measurement evolution based on data-driven incremental accumulative prediction (https://arxiv.org/abs/2501.14003)
Comments:
          20 pages, 8 figures

- **What's New**: 이 논문에서는 딥 러닝 기반의 새로운 자기 측정 진화 방법인 PaMMA-Net(Plasma Magnetic Measurements Incremental Accumulative Prediction Network)을 도입하였습니다. 이 네트워크는 토카막(discharge experiments)에서 자기 측정치를 장기간 발전시키는 능력을 가지고 있으며, 평형 재구성 알고리즘과 결합하여 플라즈마 형태와 같은 거시 매개변수도 발전시킬 수 있습니다. PaMMA-Net은 데이터 증강(data augmentation) 기법을 활용하여 기존 연구보다 우수한 진화 결과를 달성하며, 실제 실험 데이터에 대한 테스트도 수행하였습니다.

- **Technical Details**: PaMMA-Net은 자기 측정 진화(origination of magnetic measurements)를 위해 데이터 기반의 접근 방식을 채택합니다. 이 모델은 긴 시퀀스의 자기 측정을 예측하기 위해 단계적 누적 예측(incremental accumulative prediction) 접근법을 사용하며, 자기 측정 신호의 자상관(autocorrelation) 및 교차상관(cross-correlation)을 모델링합니다. 이 방법은 물리적으로 일관된 데이터 증강 기법을 채택해, 높은 정확도를 달성하며, 플라즈마의 실제 상태 정보를 포괄하기 위한 접근방식을 기반으로 하고 있습니다.

- **Performance Highlights**: 제안된 PaMMA-Net은 기존의 물리 모델에 비해 우수한 일반화 능력을 갖춘 예측 결과를 제공합니다. 실험 결과, 장기 예측 시점에서 제안된 방법의 예측 정확도가 유사한 계산 비용을 가진 기존 데이터 기반 모델보다 뛰어난 것으로 확인되었습니다. 이로 인해 PaMMA-Net은 플라즈마 자기 측정 진화 작업에서 높은 성능을 입증하였습니다.



### Advancing Math Reasoning in Language Models: The Impact of Problem-Solving Data, Data Synthesis Methods, and Training Stages (https://arxiv.org/abs/2501.14002)
Comments:
          ICLR 2025

- **What's New**: 이번 연구는 수학적 추론(Mathematical Reasoning) 능력을 크게 향상시키기 위한 새로운 접근 방식을 제안합니다. 기존의 수학 특정 LLMs는 일반적인 수학 문서와 문제 데이터셋을 활용하여 두 단계로 학습하지만, 연구 결과 일반 수학 문서보다 문제 해결 데이터의 사용이 더 효과적임을 발견했습니다. 이를 통해 JiuZhang-8B라는 강력한 수학 기반 모델이 개발되었습니다.

- **Technical Details**: 연구는 세 가지 주요 질문을 탐구합니다: (1) 문제 해결 데이터가 일반 수학 문서보다 CPT(Continue Pre-training) 과정에서 더 효과적으로 수학적 추론 능력을 향상시킬 수 있는가? (2) 동일한 출처의 합성 데이터(Synthetic Data)의 효과성을 비교하며, 어떤 합성 방법이 가장 효율적인가? (3) 동일한 문제 해결 데이터를 기반으로 개발된 능력이 CPT와 SFT(Supervised Fine-Tuning) 단계에서 어떻게 다르고, 이러한 차이에 기여하는 요인은 무엇인가? 여러 실험을 통해 문제 해결 데이터가 수학적 능력을 크게 향상시킨다는 것을 보여주었습니다.

- **Performance Highlights**: 문제 해결 데이터를 기반으로 한 CPT가 SFT보다 더 나은 성능을 발휘하였으며, 특히 튜터쉽 증폭 합성 방법(Tutorship Amplification Synthesis Method)이 최상의 결과를 보였습니다. SFT는 지시 사항을 따르는 능력을 향상시키지만, 복잡한 다단계 문제 해결 데이터에 대한 학습 능력 저하로 인해 CPT에 비해 성능이 떨어지는 경향이 있습니다. 이러한 연구 결과는 LLMs의 수학적 추론 능력 최적화를 위한 귀중한 가이드를 제공합니다.



### Enhancing kelp forest detection in remote sensing images using crowdsourced labels with Mixed Vision Transformers and ConvNeXt segmentation models (https://arxiv.org/abs/2501.14001)
- **What's New**: 이번 연구는 군중 소싱(crowdsourced) 라벨과 첨단 인공지능 모델을 통합하여 Landsat 이미지를 활용한 해조류 캐노피(kelp canopy) 탐지 파이프라인을 빠르고 정확하게 개발하는 것을 목적으로 하고 있습니다. 이 방법은 기계 학습 대회에서 3위를 기록하였으며, 지역 검증 및 공개/비공식 리더보드에서 일관되게 좋은 성과를 보였습니다. Mixed Vision Transformers (MIT)와 ConvNeXt 모델의 조합이 효과적임을 강조하고 있습니다.

- **Technical Details**: 해조류 탐지를 위한 모델 학습은 다양한 이미지 크기로 진행되어 앙상블(ensemble) 결과의 정확도를 크게 향상시켰습니다. 특히, U-Net이 최고의 세분화(segmentation) 아키텍처로 선정되었고, UpperNet 또한 최종 앙상블에 기여하였습니다. Landsat의 ShortWave InfraRed (SWIR1) 및 Near-InfraRed (NIR)와 같은 주요 밴드가 중요한 역할을 했으며, 고도 데이터를 사용하여 거짓 긍정(false positives)을 제거하는 후처리 작업이 이루어졌습니다.

- **Performance Highlights**: 이 방법론은 높은 탐지율을 기록하였으며, 해조류 캐노피를 포함한 픽셀의 약 75%를 정확히 식별하면서 거짓 긍정은 낮은 수준으로 유지하였습니다. Landsat 위성의 중해상도에도 불구하고, 광범위한 역사적 데이터는 해조류 숲 연구에 효과적임을 입증하고 있습니다. 또한, 기계 학습 모델과 군중 소싱 데이터를 결합한 방법의 환경 모니터링에서의 가능성을 강조하고 있습니다.



### Local Control Networks (LCNs): Optimizing Flexibility in Neural Network Data Pattern Captur (https://arxiv.org/abs/2501.14000)
- **What's New**: 본 논문에서는 Multi-layer perceptrons (MLPs)의 한계점을 지적하고, 각 노드에서 서로 다른 activation function을 사용하는 Local Control Networks (LCNs)를 제안합니다. 기존의 MLP는 고정된 활성화 함수를 사용하여 데이터 패턴을 포착하는 데 있어 유연성이 떨어지는데, LCN은 B-spline 함수를 통해 다채로운 활성화 곡선을 지원합니다. 이를 통해 LCN은 MLP보다 향상된 성능과 효율성을 제공하며, Kolmogorov-Arnold Networks (KANs)와 비교했을 때 더욱 접근성이 좋습니다.

- **Technical Details**: LCNs는 각 노드에서 독립적으로 학습 가능한 B-spline 함수의 매개변수를 사용하여 지역적으로 적응 가능한 활성화를 구현합니다. 해당 접근법은 활성화 함수의 다양성을 통해 신경망이 복잡한 데이터 패턴을 효과적으로 포착할 수 있도록 해줍니다. B-spline 곡선은 각 뉴런이 처리하는 데이터 특징에 맞춰 조정될 수 있어, 뉴런의 해석 가능성을 높여줍니다. 신경망 내에서 이 활성화 함수들은 각 뉴런이 특정 특징을 감지하는 데 도움을 주며, 결과적으로 네트워크의 안정성과 수렴 속도를 향상시킵니다.

- **Performance Highlights**: 실험 결과, LCNs는 컴퓨터 비전 작업에서 MLP보다 미세한 성능 개선을 보여주며 KANs보다 약 5% 높은 성능을 기록했습니다. 기본 머신러닝 작업에서는 MLP보다 1%, KANs보다 0.6% 개선된 성능을 나타냈습니다. 특히 기호 수식 표현 작업에서는 LCN과 KAN 모두 MLP를 초과하는 성능을 보였습니다. 이러한 결과는 각 노드 수준에서 다양한 활성화를 사용하는 것이 성능과 효율성을 개선하는 데 기여할 수 있음을 시사합니다.



### Framework for Progressive Knowledge Fusion in Large Language Models Through Structured Conceptual Redundancy Analysis (https://arxiv.org/abs/2501.13999)
- **What's New**: 본 연구는 대규모 모델 내에서 잠재 지식의 구조적 중복성을 분석하고 재구성하는 새로운 방법론을 제안하였다. 기존의 중복 문제를 해결하기 위한 알고리즘적 혁신과 이론적 기초를 바탕으로 하여, 지식 통합의 효율성을 높이면서 성능을 개선할 수 있는 방법을 모색하였다. 제안된 프레임워크는 중요한 의미적 관계를 보존하면서 불필요한 겹침을 제거하는 데 중점을 두었다.

- **Technical Details**: 구체적으로, 저자들은 잠재 공간 내에서 개념적 클러스터를 식별하기 위해 계층적 클러스터링 기법을 활용하였다. 이는 서로 관련성이 있는 개념들이 통합될 수 있도록 돕고, 맥락의 풍부함을 유지하면서 중복을 해결할 수 있게 한다. 또한, 동적 가중치 재조정 기법을 통해 높은 유용성을 지닌 표현들을 우선시하여 중요하지 않은 중복을 제거했다.

- **Performance Highlights**: 실험 결과는 메모리 효율성이 개선되고 추론 속도가 빨라지며, 오류율 감소 및 적대적 강건성이 증가하는 등 모델의 성능이 크게 향상되었음을 보여주었다. 자원 소비의 감소와 번역 및 요약 작업에서의 성과 향상이 두드러지며, 실제 배포 환경에서의 실용성을 추가적인 에너지 메트릭을 통해 입증하였다.



### Predictive Learning in Energy-based Models with Attractor Structures (https://arxiv.org/abs/2501.13997)
- **What's New**: 이 논문에서는 에너지 기반 모델(EBM)을 활용하여 뇌의 예측 과정, 학습 및 추론을 포함하는 생물학적으로 그럴듯한 모델을 제안합니다. 특히, 이 모델은 행동 후 관찰을 예측하는 과정을 포착하기 위해 계층 구조와 지속적 매력자 신경망(CANN)을 통합합니다. 연구 결과, 제안된 모델은 훈련 환경에서뿐만 아니라 보지 않은 환경에서도 정확한 예측을 생성하는 성능을 보입니다.

- **Technical Details**: 모델은 마르코프 체인(Markov chain) 구조를 갖고 있으며, 가우시안 분포를 따르는 조건부 확률을 사용하여 계산 효율성을 높입니다. 또한, 에러 뉴런을 도입하여 학습 과정을 국소화하고, CANN을 활용해 과거 사건을 기억하는 방식을 개선합니다. 기존의 변별적 학습 알고리즘이나 이론에 의존하지 않고, 에너지 기반 원칙에 근거한 새로운 접근 방식을 제시합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 다양한 행동을 포함한 시나리오에서 효과적인 예측을 보여 주며, 머신 러닝 방법과 동등한 성능을 기록했습니다. 모델의 강점은 복잡한 확률 분포를 학습할 수 있는 능력과 새로운 환경에서도 합리적인 예측을 제공할 수 있는 점입니다. 이러한 성과는 뇌의 예측 메커니즘에 대한 깊은 이해를 돕는 데 기여할 것으로 기대됩니다.



### CSAOT: Cooperative Multi-Agent System for Active Object Tracking (https://arxiv.org/abs/2501.13994)
- **What's New**: 이번 연구에서는 복수의 에이전트가 단일 장치에서 협력하여 능동적인 물체 추적(Active Object Tracking, AOT) 작업을 수행하는 새로운 협업 시스템을 제안합니다. 기존의 AOT 솔루션과 달리, Collaborative System for Active Object Tracking (CSAOT)는 추가적인 장비 없이도 다수의 에이전트를 활용하여 비용을 절감하고 성능을 향상시킵니다. 이 시스템은 각 에이전트가 특정 역할을 맡아 협력함으로써 동적 환경에서의 학습 및 강인성을 개선하고 있습니다.

- **Technical Details**: CSAOT는 다중 에이전트 심층 강화 학습(Multi-Agent Deep Reinforcement Learning, MADRL)와 전문 집합(Mixture of Experts, MoE) 프레임워크를 이용하여, 각 에이전트가 특정 시나리오에 최적화된 정책을 학습하도록 설계되었습니다. MoE는 각 에이전트가 현재 상황에 기반하여 향상된 결정 능력을 발휘할 수 있도록 하는 게이팅 메커니즘을 포함하고 있습니다. 이 접근법은 의사 결정 속도를 높이고, 추가 하드웨어의 필요성을 줄이며, 통신 비용을 최소화합니다.

- **Performance Highlights**: CSAOT의 성능은 동적 및 정적 장애물이 있는 다양한 상호작용 맵에서 실험하여 검증되었습니다. 이 시스템은 빠른 움직임에 대한 강인성과 가려짐에 대한 저항력을 개선했으며, 카메라 동작 최적화를 통해 더 긴 추적 시간을 달성했습니다. 이러한 성능은 복잡한 환경 내에서 효과적인 물체 추적이 가능함을 보여줍니다.



### CAPRAG: A Large Language Model Solution for Customer Service and Automatic Reporting using Vector and Graph Retrieval-Augmented Generation (https://arxiv.org/abs/2501.13993)
Comments:
          14 pages, 5 Figures, 3 Tables

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)을 기반으로 한 금융 챗봇을 통한 고객 경험 개선을 위한 AI 에이전트를 도입했습니다. 고객은 은행 서비스 및 연례 보고서에 대한 관련 정보를 손쉽게 얻을 수 있으며, 하이브리드 고객 분석 파이프라인인 CAPRAG를 통해 관계 기반 및 맥락 기반 질문 모두를 효과적으로 처리합니다. 이러한 접근 방식은 벡터 RAG와 그래프 RAG를 결합하여 더 나은 고객 참여를 촉진합니다.

- **Technical Details**: 하이브리드 접근 방식을 통해 CAPRAG는 고객의 질문을 확장한 후, 검색된 결과를 그래프 데이터베이스와 결합하여 사용자에게 정확한 답변을 제공합니다. Cypher 쿼리 언어를 사용하여 그래프 데이터베이스를 효율적으로 질의할 수 있으며, 이를 LLM을 통해 응답 생성에 활용합니다. 저자들은 이러한 방식을 통해 디지털 환경에서의 모호함을 줄이고 정보 접근성을 높이려고 노력하고 있습니다.

- **Performance Highlights**: 연구팀은 LLMs를 평가 기준으로 삼아 결과의 정확성과 일관성을 평가합니다. 주요 평가 지표로는 답변의 관련성, 맥락의 관련성, 사실 기반의 신뢰성을 포함하여, 모델의 출력 품질을 지속적으로 관리합니다. 이를 통해 사용자 경험을 개선하고 시스템에 대한 신뢰를 구축할 수 있었습니다.



### Dual-Branch HNSW Approach with Skip Bridges and LID-Driven Optimization (https://arxiv.org/abs/2501.13992)
- **What's New**: HNSW 알고리즘의 한계를 극복하기 위해 새로운 알고리즘을 제안합니다. 이 알고리즘은 local optima 문제와 cluster disconnections를 완화하며, 구성 속도를 향상시키고 추론 속도를 유지합니다. 주요 구성 요소로는 LID 기반의 삽입 메커니즘을 활용한 이중 분기 구조와 중간 레이어를 건너뛰는_bridge-building_ 기술이 포함됩니다.

- **Technical Details**: 이 연구에서는 각 노드를 LID(Local Intrinsic Dimensionality) 값을 기반으로 삽입하는 방법을 제안하여, 클러스터 내부의 이상치 노드를 더 잘 포착합니다. 이중 분기 구조를 채택하여 탐색 경로의 다양성을 높이고, 레이어를 건너뛰는 과정을 통해 추론 속도를 유지하면서 복잡도를 개선합니다. 실험을 통해 이 접근법의 정확도와 속도가 원래 HNSW보다 우수함을 보여주었습니다.

- **Performance Highlights**: 다양한 벤치마크와 데이터셋에서 실험을 수행한 결과, NLP(자연어 처리) 작업에서 18% 향상된 recall과 CV(컴퓨터 비전) 작업에서 최대 30% 향상된 성능을 달성했습니다. 또한 구성 시간은 최대 20% 감소했으며 추론 속도는 유지되었습니다. ablation 연구 결과 LID 기반 삽입이 성능에 가장 큰 영향을 미친 것으로 나타났습니다.



### CGI: Identifying Conditional Generative Models with Example Images (https://arxiv.org/abs/2501.13991)
- **What's New**: 최근 생성 모델의 성과가 주목받고 있으며, 여러 모델 허브가 등장했습니다. 이러한 허브들은 데이터 검색에 있어 기본적인 텍스트 일치를 가정하지만, 많은 수의 모델과 다양한 추상화로 인해 사용자 경험이 제한됩니다. 따라서 사용자 요구에 맞는 모델을 효과적으로 찾을 수 있는 새로운 방법론인 Conditional Generative Model Identification (CGI)를 제안합니다.

- **Technical Details**: CGI는 사용자 요구사항과 모델의 기능을 정량적으로 매칭할 수 있는 혁신적인 방법론입니다. 이를 통해 사용자가 제공하는 예시 이미지를 기반으로 적합한 모델을 식별하고, 모델 기능을 보다 정확하게 설명하기 위한 Prompt-Based Model Identification (PMI) 접근방식을 제안합니다. 실험을 통해 PMI가 65개의 모델과 9100개의 식별 과제를 포함하는 벤치마크에서 효과적인 성능을 보인다는 것을 입증하였습니다.

- **Performance Highlights**: PMI 방법을 통해 제공된 4개의 예시 이미지를 사용하여 92%의 모델을 정확하게 식별하는 성과를 달성했습니다. 이는 기존의 모델 탐색 방식보다 우수한 성능을 나타냅니다. 방대한 데이터와 실험적 검증을 통해 PMI의 유효성과 효율성을 입증하였으며, 이는 향후 연구 방향에도 기여할 수 있을 것입니다.



### FreEformer: Frequency Enhanced Transformer for Multivariate Time Series Forecasting (https://arxiv.org/abs/2501.13989)
- **What's New**: 이 논문에서는 다변량 시계열 예측을 위한 새로운 모델인 FreEformer를 소개합니다. 이 모델은 주파수 스펙트럼을 활용하여, 다양한 주파수에서의 시계열 구성에 대한 글로벌 관점을 제공합니다. 저자들은 주파수 도메인에서의 레빛 손실을 해결하기 위해 고안된 향상된 주의(attention) 메커니즘을 통해 듀얼 주의 행렬을 추가하여 예측 성능을 향상시킵니다.

- **Technical Details**: FreEformer는 주기적 패턴을 포착하기 위해 Discrete Fourier Transform (DFT)을 사용하여 시계열 데이터를 복잡한 주파수 도메인으로 변환합니다. 변환기 구조는 주파수 스펙트럼에 적용되어 상호 변량 의존성을 포착하며, 실제 및 허수 부분은 독립적으로 처리됩니다. 향상된 주의 메커니즘은 수정한 소프트맥스 주의 행렬에 학습 가능한 행렬을 추가하고, 행별 L1 정규화를 통해 구현됩니다.

- **Performance Highlights**: 다양한 전 세계의 데이터셋을 포함하는 18개 벤치마크 실험에서 FreEformer는 기존의 최첨단 모델들보다 일관되게 우수한 성능을 보여주었습니다. 특히, 향상된 주의 메커니즘이 기존 Transformer 기반 예측기의 성능을 지속적으로 향상시킴을 확인하였습니다. 본 논문은 날씨, 에너지 및 금융 등 다양한 분야에서의 강력한 예측 성능을 강조합니다.



### MCRL4OR: Multimodal Contrastive Representation Learning for Off-Road Environmental Perception (https://arxiv.org/abs/2501.13988)
Comments:
          Github repository: this https URL

- **What's New**: 자율주행차에 대한 연구는 주로 도시 환경에 집중되어 왔으나, 이 논문은 오프로드(Off-Road) 환경 인식을 위한 새로운 접근법인 MCRL4OR을 제안합니다. MCRL4OR은 시각적 이미지, 이동 상태, 제어 동작을 처리하기 위한 세 가지 인코더를 공동으로 학습하며, 이들 간의 연관성을 대조 학습(Contrastive Learning) 프레임워크 내에서 조화롭게 학습합니다. 이 연구는 비 구조적 환경에서 자율주행차가 안전하고 효과적으로 작동하도록 하는 중요한 기여를 하고 있습니다.

- **Technical Details**: MCRL4OR은 (visual observation encoder), (control action encoder), (locomotion state encoder)의 세 가지 분기로 구성되어 있으며, 여러 모달리티 간의 올바른 대응 관계를 예측하는 것을 목표로 합니다. 이 모델은 TartanDrive 데이터셋을 사용하여 사전 훈련(Pre-training)되며, 이후에는 크로스 모달 검색(Cross-modal retrieval), 동역학 예측(Dynamics prediction), 장면 분할(Scene segmentation)과 같은 다양한 하위 인식 작업에 적용됩니다. 이 논문에서 제안한 정렬 전략은 현재 지형에 따라 주어진 제어 동작이 어떻게 운동 상태에 영향을 미치는지를 기반으로 합니다.

- **Performance Highlights**: 실험 결과, MCRL4OR의 사전 훈련된 표현은 다양한 하위 작업에서 향상된 성능을 보여줍니다. 특히, 이 연구는 오프로드 환경에서의 대조 학습 모델에 대한 성능 향상이 입증되었으며, 이를 통해 논문은 MCRL4OR이 대조 학습 전략의 효과성을 강조하고 있습니다. 또한, 사전 학습된 모델이 다양한 작업에서 일관되게 성능을 개선한다는 점에서 다중 모달 표현의 일반화 가능성을 확인했습니다.



### OstQuant: Refining Large Language Model Quantization with Orthogonal and Scaling Transformations for Better Distribution Fitting (https://arxiv.org/abs/2501.13987)
Comments:
          10 Pages

- **What's New**: 이 논문에서는 대형 언어 모델(Large Language Models, LLMs)의 압축 및 가속화에 널리 채택된 기법인 사후 학습 정량화(Post-training Quantization, PTQ)에 대해 다룹니다. 기존의 기법들이 데이터 분포를 최적화하지 못하는 한계를 극복하기 위해, 저자들은 정량화 공간 활용률(Quantization Space Utilization Rate, QSUR)이라는 새로운 메트릭을 도입하였습니다. QSUR는 변환된 데이터의 정량화 공간 내의 활용도를 측정함으로써 데이터의 정량화 가능성을 평가합니다.

- **Technical Details**: 논문에서는 QSUR와 함께 다양한 변환의 영향 및 한계를 논의하는 수학적 유도도 제공하여, 직교 및 스케일링 변환 기반의 정량화(Orthogonal and Scaling Transformation-based Quantization, OSTQuant)의 개발을 위한 기초 자료를 제공합니다. OSTQuant는 가중치와 활성화의 분포를 전체 정량화 공간에 걸쳐 최적화하기 위해 학습 가능한 동등 변환을 사용합니다. 또한, 노이즈를 줄이면서 PTQ에 의해 제한된 데이터 내에서도 풍부한 의미 정보를 유지할 수 있도록 설계된 KL-Top 손실 함수도 제안합니다.

- **Performance Highlights**: OSTQuant는 여러 LLM 및 벤치마크에서 기존 연구들을 초월하는 성능을 발휘합니다. 특히, W4만 사용한 설정에서, 99.5%의 플로팅 포인트 정확도를 유지하였으며, 더 도전적인 W4A4KV4 환경을 위해 LLaMA-3-8B 모델에서 성능 격차를 32% 줄였습니다. 이러한 결과는 OSTQuant가 최신 방법들과 비교했을 때 효율적인 성능 개선을 이루었다는 것을 보여줍니다.



### An Efficient Sparse Kernel Generator for O(3)-Equivariant Deep Networks (https://arxiv.org/abs/2501.13986)
Comments:
          12 pages, 9 figures, 3 tables

- **What's New**: 이번 연구는 회전 불변(Equivariant) 그래프 신경망에 대한 GPU 희소 커널 생성기를 소개합니다. 이는 화학 분야의 깊은 학습 과제에서 더욱 향상된 성능을 제공합니다. Clebsch-Gordon 텐서 곱을 활용하여 복잡한 계산을 효율적으로 수행하는 방법을 제안하며, 기존의 다양한 오픈소스 및 프라이빗 구현보다 현저하게 빠른 속도를 보여줍니다.

- **Technical Details**: 연구에서는 CG 텐서 제품을 통해 두 개의 밀집 특성 벡터를 희소 텐서와 결합하여 새로운 밀집 벡터를 생성합니다. 이 연산은 대규모 원자 데이터셋에 대해 수백만 번 수행되어 성능 저하의 주요 원인이 됩니다. GPU 메모리 관리를 통해 작은 연산을 극대화하고, CG 텐서 제품과 그래프 컨볼루션의 통합을 통해 메모리 트래픽을 줄이는 방법이 소개됩니다.

- **Performance Highlights**: 제안한 커널은 NVIDIA cuEquivariance보다 최대 4.5배 속도가 향상되며, e3nn 패키지보다 10배 이상 향상된 성능을 보여줍니다. MACE 화학 기초 모델에 대해서도 원본 최적화 버전보다 5.3배의 추론 시간 단축效果를 나타냅니다. 이러한 성과는 회전 불변 아키텍처의 성능을 극대화하는 데 중요한 역할을 합니다.



### Pilot: Building the Federated Multimodal Instruction Tuning Framework (https://arxiv.org/abs/2501.13985)
- **What's New**: 본 논문에서는 분산 장치에서 다양한 모달리티의 지시 데이터로 MLLMs를 협력적으로 세밀하게 조정하는 데 중요한 새로운 작업인 Federated Multimodal Instruction Tuning(FedMIT)을 탐구합니다. 이를 위해 'Pilot'이라는 연합 다중 모달 지시 조정 프레임워크를 제안합니다. 이 프레임워크는 비전 인코더와 LLM을 연결하는 부분에 'adapter on adapter'의 두 가지 단계로 통합되어 있습니다.

- **Technical Details**: 1단계에서는 비주얼 정보를 기반으로 작업 특화 기능과 클라이언트 특화 기능을 추출합니다. 2단계에서는 Cross-task Mixture-of-Adapters(CT-MoA) 모듈을 구축하여 작업 간 상호작용을 수행합니다. 각 클라이언트는 지역 데이터의 개인화 정보를 캡처할 뿐만 아니라, 다른 작업에서의 일반적인 지식을 학습하여 모델 성능을 향상시킬 수 있도록 합니다.

- **Performance Highlights**: 이 프레임워크는 다양한 로컬 클라이언트의 분산 데이터를 협력적으로 활용하여 지시 조정 중 작업 이질성에 영향을 받지 않고 교차 작업 지식을 학습할 수 있습니다. 실험을 통해 Pilot 방법이 최신 LLaVA 모델에서 두 가지 다른 교차 작업 시나리오에서 효과적임을 확인했습니다.



### Comprehensive Modeling and Question Answering of Cancer Clinical Practice Guidelines using LLMs (https://arxiv.org/abs/2501.13984)
- **What's New**: 이번 연구에서는 의료 전문가가 치료 결정을 내리는 데 도움이 되기 위해 Clinical Practice Guidelines (CPGs)의 규정 지식을 효과적으로 포착하는 방법을 제안합니다. 특히 National Comprehensive Cancer Network (NCCN) Cancer CPGs의 맥락을 풍부하게 한 디지털 표현을 그래프 형태로 구축하는 접근 방식을 소개하고 있습니다. 이를 통해 CPGs에 포함된 의료 지식을 보다 충실하게 캡처할 수 있습니다.

- **Technical Details**: 연구진은 자동 추출 및 노드(node)와 관계(relationship)의 분류를 통해 보다 풍부한 디지털 표현을 생성하였습니다. 또한, 대형 언어 모델(Large Language Models, LLMs)을 활용하여 노드를 분류함으로써 80.86%와 88.47%의 정확도를 달성하였습니다. 이 과정에서 zero-shot learning과 few-shot learning을 적용하였습니다.

- **Performance Highlights**: 이 연구에서 개발한 방법론은 자연어 질문에 대한 답변을 제공하는 데 활용됩니다. LLMs를 활용하여 가이드라인 지식 기반에서 관련 서브그래프(subgraph)를 추출하고, 이를 통해 정확한 답변을 생성함으로써 의료 분야의 질문 응답(Question Answering)에서 발생할 수 있는 오류와 환각(hallucination) 문제를 완화합니다. 이러한 접근 방식은 의료 도메인에서 사실 정확성을 보장하는 데 기여합니다.



### AdEval: Alignment-based Dynamic Evaluation to Mitigate Data Contamination in Large Language Models (https://arxiv.org/abs/2501.13983)
- **What's New**: 本 연구에서는 AdEval(Alignment-based Dynamic Evaluation)이라는 새로운 동적 데이터 평가 방법을 제안한다. 이 방법은 데이터 오염(data contamination)의 영향을 줄이고 평가 신뢰성을 높이기 위해 설계되었다. AdEval은 정적(static) 데이터의 핵심 개념과 일치하는 질문을 생성하며, 온라인 검색을 통해 관련 지식 포인트에 대한 자세한 설명을 제공하여 평가 샘플을 생성한다.

- **Technical Details**: AdEval의 주요 혁신은 동적으로 정렬된 평가 데이터를 생성하는 것이다. 이를 위해 정적 데이터에서 핵심 지식 포인트와 주요 아이디어를 추출하고, 온라인 검색 결과를 활용하여 관련 내용을 상세히 확장한다. 또한, Bloom의 분류법을 기반으로 LLM의 성과를 기억, 이해, 적용, 분석, 평가, 생성의 여섯 인지 수준에서 다차원적 평가를 수행한다.

- **Performance Highlights**: 다양한 데이터셋에 대한 실험 결과, AdEval은 데이터 오염이 평가 결과에 미치는 영향을 효과적으로 줄이며 신뢰성과 공정성을 향상시킨다. 이 방법은 LLM 평가에서 신뢰할 수 있는 결과를 도출하고, 다양한 복잡성 수준을 지원하여 평가의 질을 높이는 데 기여한다.



### Chain of Grounded Objectives: Bridging Process and Goal-oriented Prompting for Code Generation (https://arxiv.org/abs/2501.13978)
- **What's New**: 최근 대규모 언어 모델(LLM)의 코드 생성 활용도가 급증하면서 기존 방법들의 한계를 해결하기 위한 새로운 접근 방식인 Chain of Grounded Objectives (CGO)가 제안되었습니다. CGO는 입력 프롬프트에 기능적 목표를 포함하여 코드 생성을 향상시키며, 그 과정에서 명시적인 순차적 절차를 피하면서 프로그래밍 작업의 구조적 특성에 잘 적응합니다. 이를 통해 기존의 접근 방법들의 제약을 극복하고 코드 생성의 질을 높이는 데 기여하고자 합니다.

- **Technical Details**: CGO는 기능적 목표를 자연어 또는 다른 구조화된 형태로 입력 프롬프트에 직접 포함시키는 방식으로, 목표 정의의 명확성과 프로세스 지향적 추론의 암묵적인 논리 구조를 통합합니다. 이 과정에서 LLM이 훈련받은 코드 주석 형식의 구조를 활용하여, 문제의 목표와 생성된 코드 간의 정렬을 보장합니다. CGO는 ‘목표 생성 단계’와 ‘코드 생성 단계’의 두 가지 단계로 구성되어 있으며, 앞서 언급된 목표들이 LLM에 의해 생성된 후 이를 바탕으로 코드를 생성하는 방식입니다.

- **Performance Highlights**: 실험 결과에 따르면, CGO는 기존의 방법들에 비해 코드 생성 성능이 현저히 향상된 것으로 나타났습니다. 특히, pass@1 및 pass-ratio@10과 같은 정확성 메트릭에서 더 높은 정확도를 기록하며, LLM의 코드 주석과 문서화 패턴에 대한 익숙함을 활용하여 최종 코드와 기능적 목표 간의 정렬을 강화했습니다. 이러한 성과는 CGO가 코드 생성 벤치마크에서 기준 모델들을 초월하는 데 기여했음을 입증합니다.



### Re-ranking Using Large Language Models for Mitigating Exposure to Harmful Content on Social Media Platforms (https://arxiv.org/abs/2501.13977)
Comments:
          This paper is under peer review

- **What's New**: 이 논문에서는 소셜 미디어 플랫폼이 사용자의 참여를 극대화하기 위해 사용하는 기계 학습(ML) 및 인공지능(AI) 기반 추천 알고리즘이 부정적인 콘텐츠에 노출되는 문제를 해결하기 위한 새로운 접근법을 제안합니다. 제안된 방법은 대규모 데이터를 필요로 하지 않고, 대형 언어 모델(LLMs)을 활용하여 콘텐츠 시퀀스를 동적으로 평가하고 재정렬하여 유해한 콘텐츠 노출을 줄이는 것입니다. 이 방법은 기존의 콘텐츠 조절 방식과 비교하여 더 높은 성능을 발휘하는 것을 보여줍니다.

- **Technical Details**: 이 논문에서는 대형 언어 모델(LLMs)이 제로샷(zero-shot) 및 퓨샷(few-shot) 학습 설정에서 우수한 추론 능력을 보여준다는 점을 강조하며, 이를 통해 유해한 콘텐츠 노출을 줄이기 위한 방법을 제시합니다. LLMs는 페어와이즈 비교(pairwise comparison)와 재정렬(re-ranking)을 통해 콘텐츠 시퀀스를 검토하여 권장할 콘텐츠의 질을 높이는 역할을 합니다. 또한 유해한 콘텐츠에 대한 노출을 분석하기 위한 두 가지 새로운 메트릭(metrics)을 도입하여, 제안하는 방법의 유용성을 평가합니다.

- **Performance Highlights**: 실험 결과, 제안된 LLM 기반 접근법은 세 가지 유해 데이터셋과 세 가지 LLM 아키텍처를 통해 기존의 상업적 조절 API인 Perspective API 및 OpenAI Moderation API에 비해 우수한 성능을 보였습니다. 제로샷 환경에서도 산업 표준 분류기를 초월하는 성능을 나타내어, 다양한 유형의 유해에 일반화 가능하다는 점을 강조합니다. 이를 통해 플랫폼에서 유해 콘텐츠 노출을 효과적으로 줄일 수 있는 확장 가능하고 적응 가능한 솔루션을 제공한다고 결론짓습니다.



### Towards Safer Social Media Platforms: Scalable and Performant Few-Shot Harmful Content Moderation Using Large Language Models (https://arxiv.org/abs/2501.13976)
Comments:
          This paper is in submission and under peer review

- **What's New**: 본 논문에서는 소셜 미디어에서 유해 콘텐츠를 효과적으로 모니터링하기 위해 Large Language Models (LLMs)를 활용한 새로운 접근 방식을 소개합니다. 기존의 자동화된 방법들과 비교하여, LLMs는 소량의 예제만으로도 유해 콘텐츠를 식별하는 성능이 뛰어난 것으로 나타났습니다. 또한, 비주얼 정보(영상 썸네일)를 통합함으로써 모델의 성능을 더욱 향상시킬 수 있다는 점을 밝힙니다.

- **Technical Details**: 연구에서는 Llama2-12B, Mistral-7B, GPT-4o-Mini, GPT-3.5-Turbo 등 여러 LLM을 활용하여 유해 콘텐츠를 분류하는 실험을 진행했습니다. 결과적으로, LLMs는 zero-shot 설정에서도 기존의 상업적 기준인 Google의 Perspective API 및 OpenAI의 Moderation API를 초과하는 성능을 보여주었습니다. 또한, few-shot learning을 통해 지침 프롬프트에서 유해 및 비유해 콘텐츠의 예제를 제공함으로써 분류 성능을 크게 향상시켰습니다.

- **Performance Highlights**: 이번 연구를 통해 LLMs가 유해 콘텐츠 식별에서 기존의 딥러닝 모델을 초과함을 입증했습니다. 또한, multimodal LLMs를 이용해 시각적 입력을 포함할 경우 유해 콘텐츠를 식별하는 정확도가 향상됨을 밝혔습니다. 특히, 오픈소스 multimodal 모델보다 비공식 소스 모델에서 더 나은 성능이 나타났습니다.



### A Spatio-temporal Graph Network Allowing Incomplete Trajectory Input for Pedestrian Trajectory Prediction (https://arxiv.org/abs/2501.13973)
- **What's New**: 이번 연구에서는 과거의 불완전한 경로를 고려하여 보행자의 미래 경로를 예측할 수 있는 새로운 spatio-temporal graph network인 STGN-IT를 제안합니다. 기존의 알고리즘들은 불완전한 정보를 처리하지 못하고, 이로 인해 로봇 내비게이션에 어려움을 겪어왔습니다. STGN-IT는 정적 장애물 정보를 통합하여 예측 정확도를 높이는데 기여합니다.

- **Technical Details**: STGN-IT 알고리즘은 spatio-temporal graph를 사용하여 보행자와 장애물 정보를 표현하고, 특별한 인코딩 방법을 통해 불완전한 과거 경로를 처리합니다. 이 알고리즘은 기존의 semantic map 대신 자동으로 생성된 occupancy grid map을 활용하여 더욱 유연한 접근을 제공합니다. 데이터셋 STCrowd에서 STGN-IT의 성능을 비교 평가한 결과, 기존 최첨단 알고리즘을 능가하는 성과를 보였습니다.

- **Performance Highlights**: STGN-IT는 기존 알고리즘과 비교했을 때 ADE와 FDE에서 개선된 성과를 나타내며, 특히 egocentric view 데이터셋에서 불완전한 경로에 대한 저항성을 발휘합니다. 실험 결과, STGN-IT는 로봇 내비게이션 환경에서 보다 안전하고 효율적인 경로 예측을 가능하게 하며, 이는 로봇이 보행자와의 충돌 위험을 줄이는데 도움이 됩니다.



### FedDAG: Federated Domain Adversarial Generation Towards Generalizable Medical Image Analysis (https://arxiv.org/abs/2501.13967)
- **What's New**: 이 논문에서는 Federated Domain Adversarial Generation (FedDAG)이라는 새로운 프레임워크를 제안하여 연합 학습(federated learning) 환경에서 모델의 일반화 능력을 향상시키는 방법을 다룹니다. FedDAG는 기존의 연합 도메인 일반화(federated domain generalization) 접근이 갖고 있는 제한을 극복하고, 지역 도메인 특성과 전역 도메인 특성을 반영하는 새로운 도메인을 적대적으로 생성하여 모델의 일반화 성능을 개선하는 데 초점을 맞추고 있습니다. 특히, FedDAG는 현업에서의 의료 시나리오에 맞춰 세밀하게 설계되었습니다.

- **Technical Details**: FedDAG는 각 클라이언트에서 독립적으로 적대적 생성을 수행하면서 의미 일관성을 유지하는 2단계 구속을 적용하여 새로운 스타일의 이미지를 생성합니다. 이는 기존의 지역 특성과는 다른 특성을 지닌 이미지를 생성하는 데 중점을 두며, 이를 통해 의료 데이터의 중요한 의미를 극대화합니다. 또한, 모델의 일반화 기여를 평가하기 위해 모델 집합체를 계층적으로 집계하여 각 클라이언트의 기여를 균형 있게 조정하는 방식을 취합니다.

- **Performance Highlights**: 논문에서 언급된 대규모 실험은 FedDAG가 기존 방식보다 더 나은 일반화 능력을 갖추고 있음을 보여줍니다. 네 가지 의학적 기준 벤치마크에 대한 실험 결과는 FedDAG가 의료 데이터의 도메인 변화에 적절히 대응할 수 있는 능력을 입증하였습니다. 이러한 결과들은 전반적으로 FedDAG가 의학적 시나리오에서 모델의 성능을 더욱 향상시킬 수 있음을 나타냅니다.



### ZKLoRA: Efficient Zero-Knowledge Proofs for LoRA Verification (https://arxiv.org/abs/2501.13965)
Comments:
          7 pages, 3 figures

- **What's New**: ZKLoRA는 Large Language Models (LLMs)을 위한 Low-Rank Adaptation (LoRA) 방법을 사용하여 분산 훈련 환경에서의 신뢰성 문제를 해결하는 제로 지식 검증 프로토콜을 제시합니다. 이 프로토콜은 LoRA 가중치의 기밀성을 유지하면서도 base 모델과의 호환성을 인증할 수 있도록 합니다. ZKLoRA는 검증 시간을 1-2초로 단축 시키고, 안전한 협업을 가능하게 하여, 기여자의 지적 재산을 보호합니다.

- **Technical Details**: ZKLoRA는 LoRA 모듈의 정확성을 검증하기 위해 간결한 증명을 제공하며, Multi-Party Inference 절차를 기반으로 합니다. 이렇게 설계된 ZKLoRA는 분산 환경에서도 효율적으로 작동하여, LoRA 기여자가 자신의 가중치를 공개하지 않고도 그 가중치가 실제 base 모델과 잘 작동하는지를 보장합니다. 이 방법은 저지연 접근성을 통해 실시간 검증을 촉진합니다.

- **Performance Highlights**: ZKLoRA는 다양한 LLM과 소규모 모델에서 벤치마킹 테스트를 수행하였으며, LoRA 모듈 수에 따라 검증 시간과 설정, 증명 생성 시간을 분석했습니다. 모델의 LoRA 모듈 수가 많을수록 검증 시간이 증가하지만, ZKLoRA의 설계 덕분에 80개의 모듈을 검증하는 데 몇 분밖에 걸리지 않습니다. 이로써 ZKLoRA는 대규모 LLM 파이프라인에서의 LoRA 사용의 가능성을 높이고, 점진적 검증 절차를 통해 코드 기밀성을 보장합니다.



### Advancing the Understanding and Evaluation of AR-Generated Scenes: When Vision-Language Models Shine and Stumb (https://arxiv.org/abs/2501.13964)
Comments:
          6 pages

- **What's New**: 이번 연구에서는 증강 현실(AR) 경험의 품질을 평가하기 위해 세 가지 최첨단 Vision-Language Models(VLMs)인 GPT, Gemini, Claude의 능력을 분석했습니다. DiverseAR이라는 새로운 데이터셋을 사용하여 VLM이 AR 장면을 인식하고 설명하는 능력을 평가했습니다. 연구 결과, VLM은 AR 장면을 인식하는 데에서 최대 93%의 True Positive Rate(TPR)을 달성했으나, 매우 통합된 콘텐츠에 대한 인식에서는 어려움을 겪었습니다.

- **Technical Details**: DiverseAR 데이터셋은 다양한 AR 시나리오를 수집한 298개의 이미지를 포함하고 있으며, 이 데이터셋은 VLMs의 AR 장면 이해 능력을 평가하기 위해 설계되었습니다. 연구에서는 다양한 난이도의 AR 장면을 정의하고, VLMs의 성능을 평가하기 위해 일반 이미지 캡셔닝 프롬프트와 테스크 인식 이미지 캡셔닝 프롬프트를 사용했습니다. 각 AR 샘플은 정확한 인식, 부분 인식, 놓친 인식, 잘못된 인식, 정확한 비-AR 인식 등 다섯 가지 범주로 나누어 분석되었습니다.

- **Performance Highlights**: VLM은 눈에 띄는 가상 객체는 잘 인식하지만, 물리적 법칙을 충족하는 콘텐츠에서는 어려움을 겪고 있습니다. 다양한 AR 장면을 분석한 결과, VLM의 인식 및 설명 성능의 한계가 명확히 드러났습니다. 이러한 연구는 AR 경험의 품질을 평가하는 데 있어 VLM의 가능성을 강조하며, 실제와 가상 콘텐츠 간의 효과적인 구분법을 제시하고 있습니다.



### Adaptive Cyber-Attack Detection in IIoT Using Attention-Based LSTM-CNN Models (https://arxiv.org/abs/2501.13962)
- **What's New**: 이 논문은 산업 인터넷의 사물(IIoT) 환경에서 사이버 공격을 탐지하고 분류하기 위한 하이브리드 LSTM-Convolution Neural Network(CNN)-Attention 아키텍처에 기반한 진보된 침입 탐지 시스템(IDS)의 개발 및 평가를 제시합니다. 이 연구는 이진 및 다중 클래스 분류의 두 가지 주요 분류 작업에 중점을 두고 있으며, Edge-IIoTset 데이터셋을 사용하여 제안된 모델이 엄격히 테스트되었습니다. 또한, SMOTE(특정 소수 샘플 오버 샘플링 기법)를 활용하여 불균형한 데이터 세트를 보정함으로써 모델의 학습 효과를 극대화했습니다.

- **Technical Details**: 제안된 LSTM-CNN-Attention 모델은 LSTM의 시간적 모델링 강점, CNN의 공간적 특성 추출 능력, 그리고 Attention 메커니즘에 의한 동적 초점을 결합하여 구축되었습니다. 이 연구에서는 60개의 특성과 15가지 공격 유형을 포함하는 데이터셋을 이차원 형태로 구조화하였으며, 비율에 따라 80%는 훈련용, 20%는 테스트용으로 분할했습니다. SMOTE 알고리즘을 사용하여 소수 클래스에 대한 합성 샘플을 생성하고, 학습과 검증 세트를 다시 나누어 균형 분포를 확인했습니다.

- **Performance Highlights**: 실험 결과, LSTM-CNN-Attention 모델은 이진 분류에서 거의 완벽한 정확도를 달성하였고, 다중 클래스 분류에서는 99.04%의 높은 정확도를 유지하며 다양한 공격 유형을 효과적으로 분류했습니다. 손실 값은 0.0220%으로 매우 낮아, 모델이 다수의 메트릭스에서 다른 모델들보다 우수함을 입증했습니다. 이러한 성과는 IIoT 환경의 복잡한 사이버 보안 문제를 해결하는 데 기여할 것으로 기대됩니다.



### Assisting Mathematical Formalization with A Learning-based Premise Retriever (https://arxiv.org/abs/2501.13959)
- **What's New**: 이번 연구에서는 수학의 정형화를 지원하기 위한 프레미스 리트리버(premise retriever) 훈련을 위한 혁신적인 방법을 소개합니다. 이 방법은 BERT 모델을 사용하여 증명 상태와 프레미스를 공유 잠재 공간(latent space)에 임베딩합니다. 실험 결과, 우리의 모델은 기존 기준선보다 성능이 뛰어나며, 컴퓨팅 자원이 적게 소모되어 효율적임을 보여주었습니다.

- **Technical Details**: 프레미스 리트리버 모델은 컨텍스트 프리 리트리벌(CFR) 모듈과 컨텍스트 인지 재정렬(CAR) 모듈로 구성되어 있습니다. 각 모듈은 BERT를 기반으로 하며, CFR 모듈은 임베딩을 생성하여 입력된 증명 상태와 유사한 프레미스를 검색합니다. CAR 모듈은 검색된 프레미스를 재정렬하여 최상위 k-k recall을 개선하여 결과의 정확성을 높이는 데 기여합니다.

- **Performance Highlights**: 우리 모델은 성능에서 최첨단 모델을 초월하는 결과를 보였으며, 여전히 계산 효율성을 유지하고 있습니다. 또한 우리는 이 모델을 독립적으로 훈련된 전술 생성기(tactic generator)와 통합하여 검색 보강 자동 정리 증명기(retrieval-augmented automated theorem prover)를 평가하였습니다. 이 시스템은 수학적 증명 과정의 효율성을 높이는 데 기여하며, 사용자들에게 편리한 웹 검색 엔진을 제공할 예정입니다.



### A Survey of Graph Retrieval-Augmented Generation for Customized Large Language Models (https://arxiv.org/abs/2501.13958)
- **What's New**: 최근의 연구는 Graph-based Retrieval-Augmented Generation (GraphRAG) 모델이 기존 RAG의 한계를 극복하는 새롭고 효율적인 방법임을 보여줍니다. GraphRAG는 전문 도메인에 맞는 LLM의 적용을 통해 복잡한 쿼리를 이해하고, 다양한 지식을 통합하며, 효율성을 향상시킵니다. 이러한 혁신적인 접근 방식은 지식 구조를 그래프 형태로 표현하여 도메인 내의 관계와 계층을 명확하게 설명합니다.

- **Technical Details**: GraphRAG는 도메인 지식의 명확한 표현을 위해 그래프 구조를 활용하며, 이를 통해 다단계 추론을 지원하는 효율적인 검색 기법을 도입합니다. 이 모델은 쿼리에 대한 맥락을 유지하면서 지식을 검색할 수 있도록 설계되었으며, 얻어진 지식을 활용해 LLM의 생성 품질을 높이는 구조 인지 알고리즘을 포함합니다. 또한, 다양한 전문 분야에서 GraphRAG의 현재 구현 사례를 분석하여 중요한 기술적 도전 과제를 확인합니다.

- **Performance Highlights**: GraphRAG의 통합적인 지식 검색 방법은 사용자 쿼리에 대한 정확하고 논리적으로 일관된 응답을 생성할 수 있게 해 줍니다. 기존 RAG 시스템의 성능 한계를 극복하여 복잡한 쿼리를 보다 효과적으로 처리하며, 시스템의 효율성을 높이는 동시에 실시간으로 도메인 특화된 지식에 접근할 수 있도록 합니다. 이를 통해 LLM을 전문화된 환경에서 더욱 효과적으로 활용할 수 있는 가능성을 제시합니다.



### Benchmarking Generative AI for Scoring Medical Student Interviews in Objective Structured Clinical Examinations (OSCEs) (https://arxiv.org/abs/2501.13957)
Comments:
          11 pages, 4 figures (+3 figures in supplementary appendix)

- **What's New**: 이번 연구는 Objective Structured Clinical Examination (OSCE)에서 의료 학생들의 의사소통 능력을 평가하기 위해 대규모 언어 모델(LLMs)을 자동화 가능성을 조사했습니다. 특히, Master Interview Rating Scale (MIRS)을 사용하여 OSCE 평가의 효율성을 높일 수 있는 방법을 모색하였습니다.

- **Technical Details**: 연구에서는 제로샷(zero-shot), 체인 오브 사고(chain-of-thought, CoT), 몇 번의 샷(few-shot), 다단계(multi-step) 프롬프트 기법을 사용하여 최신 LLM 포트폴리오인 GPT-4o, Claude 3.5, Llama 3.1, Gemini 1.5 Pro의 성능을 비교했습니다. 174명의 전문가 합의 점수가 제공되는 10개의 OSCE 사례 데이터셋을 기준으로 모델 성능이 측정되었으며, 정확성 메트릭스는 exact, off-by-one, thresholded로 평가되었습니다.

- **Performance Highlights**: LLMs는 모든 MIRS 항목과 OSCE 사례에 대해 평균 정확도가 낮았지만, off-by-one과 thresholded 정확도는 중간에서 높은 범위에 있었습니다. 특히, GPT-4o는 높은 intra-rater 신뢰성을 보였으며(α = 0.98), CoT, few-shot, multi-step 기법은 특정 평가 항목에 맞추어 조정되었을 때 효과적이었습니다. 연구 결과는 AI를 활용한 OSCE 평가의 가능성을 보여주며, 향후 임상 의사소통 기술 자동화 평가 연구의 기초 성능 평가로 활용될 수 있습니다.



### Zep: A Temporal Knowledge Graph Architecture for Agent Memory (https://arxiv.org/abs/2501.13956)
Comments:
          12 pages, 3 tables

- **What's New**: Zep은 AI 에이전트를 위한 새로운 메모리 레이어 서비스로, Deep Memory Retrieval (DMR) 벤치마크에서 기존의 MemGPT를 능가하는 성능을 보여줍니다. Zep은 DMR보다 더 포괄적이면서도 어려운 평가에서도 뛰어난 성과를 내어 실제 기업 사용 사례를 보다 잘 반영합니다. Graphiti라는 핵심 구성 요소를 통해 다양한 데이터 소스로부터 동적인 지식 통합을 수행하여 기업의 요구를 충족합니다.

- **Technical Details**: Zep의 정보 저장 방식은 동적인 지식 그래프(knowledge graph)를 기반으로 하며, 이 그래프는 입력 메시지 데이터를 포함하는 에피소드 서브그래프, 에피소드에서 추출된 엔티티를 나타내는 의미 엔티티 서브그래프, 그리고 강하게 연결된 엔티티들의 클러스터를 나타내는 커뮤니티 서브그래프의 세 가지 계층으로 구성되어 있습니다. Graphiti는 비손실 방식으로 지식 그래프를 업데이트하며, 각 사실과 관계의 타임라인을 유지하여 복잡하고 진화하는 세상을 모델링합니다. 이 기술적 접근은 LLM 기반 에이전트의 기억을 혁신적으로 확장하는 데 기여합니다.

- **Performance Highlights**: Zep은 DMR 벤치마크에서 94.8%의 성능을 기록하여 MemGPT의 93.4%를 초과하였습니다. LongMemEval 벤치마크에서도 Zep은 최대 18.5%의 정확도 향상과 함께 응답 지연 시간을 90% 단축하는 성과를 기록했습니다. 이러한 결과는 특히 세션 간 정보 통합 및 장기적 컨텍스트 유지와 같은 기업 필수 작업에서 두드러지며, 실제 애플리케이션에서의 효과성을 입증합니다.



### Guided Persona-based AI Surveys: Can we replicate personal mobility preferences at scale using LLMs? (https://arxiv.org/abs/2501.13955)
- **What's New**: 본 연구에서는 독일의 개인 이동 선호도를 중심으로 인공지능 언어 모델(Large Language Models, LLMs)의 잠재력을 탐구합니다. 전통적인 설문조사 방법의 한계를 해결하기 위해 LLMs를 활용하여 합성 데이터를 생성하는 방식으로 진행됩니다. 특히, 인구통계학적 및 행동적 특성을 조합한 '페르소나(Personas)'를 도입하여 다섯 가지 다른 합성 설문조사 방법과 비교했습니다.

- **Technical Details**: 연구에서 사용한 MiD 2017 데이터셋은 독일의 이동 선호도에 대한 자세한 정보를 제공합니다. AI 설문조사 생성에 있어 GPT-4o API를 사용하였고, 10,000명의 인구를 생성하여 15,840개의 독특한 페르소나를 정의했습니다. 각각의 방법에 대해 구체적으로 설계된 프롬프트를 사용하여 LLM의 응답 생성을 안내했습니다.

- **Performance Highlights**: 결과적으로 제안된 가이드된 페르소나 기반 설문조사가 기존의 방법들에 비해 정확도와 일관성에서 큰 개선을 보였습니다. 이 접근 방식은 교통 계획 및 사회 과학 연구에서 높은 신뢰도를 가진 합성 데이터 세트를 생성할 수 있는 기회를 제공합니다. LLMs를 활용한 합성 데이터는 비용 효율적이면서도 개인 정보 보호가 가능한 데이터 생성을 가능하게 합니다.



### Chat3GPP: An Open-Source Retrieval-Augmented Generation Framework for 3GPP Documents (https://arxiv.org/abs/2501.13954)
- **What's New**: Chat3GPP는 3GPP 문서에 특화된 오픈 소스 RAG(확장된 정보 검색 생성) 프레임워크로 제안되며, 이는 도메인 특화된 파인 튜닝 없이 사용자 쿼리에 대한 관련 정보 검색 및 정확한 응답 생성을 가능하게 합니다. 이러한 유연성 및 확장성 덕분에 Chat3GPP는 3GPP 외의 다른 기술 표준에도 쉽게 적용될 수 있는 잠재력을 지니고 있습니다.

- **Technical Details**: Chat3GPP는 3GPP의 기술 문서에서 정보를 처리하고 검색하기 위해 데이터 전처리, 인덱싱, 검색 및 생성의 여러 주요 단계를 포함하는 아키텍처를 가지고 있습니다. 이를 위해, 3GPP FTP 사이트에서 Release 17과 Release 18 기술 사양 문서를 크롤링하고, 문서의 형식을 변환한 후, 내용을 필터링하여 텍스트만을 추출하여 임베딩 데이터베이스로 변환합니다.

- **Performance Highlights**: Chat3GPP는 통신 관련 데이터 세트에서 평가되었으며, 기존 방법들에 비해 우수한 성능을 보여줍니다. 이는 프로토콜 생성 및 코드 자동화와 같은 다운스트림 태스크에 대한 잠재력을 강조하며, LLM의 발전을 통해 통신 산업의 다양한 측면에서 혁신을 기대할 수 있습니다.



### Redundancy Principles for MLLMs Benchmarks (https://arxiv.org/abs/2501.13953)
- **What's New**: 최근 Multi-modality Large Language Models (MLLMs)의 급속한 발전과 함께 매년 수백 개의 벤치마크가 생산되고 있습니다. 이로 인해 기반 성능 평가에서 중복된 부분이 발생하고 있으며, 이 논문은 이러한 중복 문제를 체계적으로 분석하여 효과적인 MLLM 벤치마크 설계를 위한 원칙들을 제시합니다. 특히, 벤치마크의 능력 차원, 테스트 질문 수의 중복성, 도메인 간의 중복성을 강조하고 있습니다.

- **Technical Details**: MLLM 평가의원칙으로는 독립된 차원, 최적의 인스턴스 수, 도메인 대표성이 제안됩니다. 중복성의 정량화는 Performance Correlation Redundancy Framework를 통해 이루어지며, 수행 순위의 상관관계를 측정하여 얼마나 중복성이 존재하는지 파악합니다. 100개 이상의 MLLM을 포함한 VLMEvalKit 데이터를 활용하여 체계적이고 포괄적인 분석을 수행하였습니다.

- **Performance Highlights**: 본 논문은 MLLM 벤치마크의 중복성을 평가함으로써 벤치마크 설계의 최적화를 도모하고, 효율성을 높여 모델 평가 시스템을 개선할 수 있는 방안을 제시합니다. 또한, 중복성을 시스템적으로 해결함으로써 MLLM 평가의 자원 요구를 줄이고 보다 효과적인 평가 생태계를 구축하는 데 기여할 수 있을 것으로 보입니다.



### The Dual-use Dilemma in LLMs: Do Empowering Ethical Capacities Make a Degraded Utility? (https://arxiv.org/abs/2501.13952)
- **What's New**: 이 논문은 Direct Preference Optimization (DPO) 기반의 정렬 프레임워크인 LibraAlign를 제안하여 안전성과 유용성 간의 균형을 맞추는 방법을 제시합니다. DPO 프레임워크는 화학 분야의 데이터셋인 LibraChemQA를 사용하여 안전한 정보 차단과 유효한 요청 수용을 동시에 고려합니다. 이 연구는 기존 LLM의 성능을 향상시키기 위한 다양한 방법론을 통해 이 분야의 윤리적 문제를 탐구합니다. 특히, 화학 분야에서의 응용을 증명 사례로 제시하며, 다른 도메인에도 적용 가능성을 강조합니다.

- **Technical Details**: LibraAlign 프레임워크는 강화 학습에서의 인체 피드백(RLHF) 및 DPO 모델을 기반으로 하여 안전성과 유용성을 동시에 고려하는 메커니즘을 갖추고 있습니다. 이 모델은 31,600개의 트리플 인스턴스를 포함한 화학 질문-답변 데이터셋인 LibraChemQA를 생성하여, 긍정적 요청과 부정적 요청을 모두 아우르는 균형 잡힌 데이터 생성을 목표로 합니다. 또한, LLM의 도메인 특화 이해도를 높이기 위해 질문 재구성(rephrasing) 기법을 포함하는 데이터 증강(data augmentation) 방식이 혁신적으로 도입되었습니다.

- **Performance Highlights**: LibraChem 모델은 Claude-3, GPT-4o, LLaMA-3과 같은 기존 LLM보다 각각 13.44%, 7.16%, 7.10%의 성능 향상을 기록하며 기존 모델을 능가하는 것으로 나타났습니다. 실험 결과는 안전성과 유용성 모두를 고려했을 때의 모델의 전반적인 성능 향상을 보여줍니다. 이 연구는 도메인 전문 지식을 유지하면서도 윤리적인 역량을 강화하는 이중 사용(dilemma) 문제를 해결하는 데 있어 중요한 기여를 하고 있습니다.



### A Layered Multi-Expert Framework for Long-Context Mental Health Assessments (https://arxiv.org/abs/2501.13951)
- **What's New**: 이번 연구에서 제안된 Stacked Multi-Model Reasoning (SMMR) 프레임워크는 대규모 언어 모델(LLMs)의 한계를 극복하기 위해 여러 개의 모델을 활용합니다. 이 프레임워크는 각 LLM을 독립적인 '전문가'로 간주하여, 정확성과 신뢰성을 높이는 데 기여합니다. SMMR은 우선 짧고 개별적인 작업을 처리하고, 이후에 긴 문맥을 다룰 수 있는 모델을 사용하여 결과를 통합하고 정교화합니다. 이러한 접근은 정신 건강 평가에 있어 보다 균형 잡힌 다양한 관점을 제공합니다.

- **Technical Details**: SMMR 프레임워크는 깊은 레이어로 구성되어 있으며, 초기 레이어는 작은 LLM이나 전문화된 모델을 활용해 초기 평가를 생성합니다. 이후 레이어는 긴 문맥 모델을 통해 각 단계에서 측정된 출력을 합치고 세분화하여 최종 결과를 도출합니다. 최종적으로는 가장 안정적인 성능을 보이는 장기 문맥 모델을 선택하여 복잡한 정신 건강 평가를 위한 최종 결과를 생성합니다. SMMR은 성능 지표를 활용하여 각 레이어에서 개선이 이루어질 때만 다음 레이어로 진행하는 동적 중지 메커니즘을 포함합니다.

- **Performance Highlights**: SMMR은 DAIC-WOZ 우울증 선별 데이터를 포함한 다양한 사례 연구에 대한 평가를 통해, 단일 모델 기준에 비해 일관된 개선이 있음을 입증했습니다. 정확도, F1 점수, 그리고 PHQ-8 오류 감소 면에서 성과를 향상시켰습니다. 또한, 여러 전문가의 의견을 활용함으로써 환각(hallucinations) 및 미세한 임상 세부 사항을 포착할 수 있는 능력이 향상되었습니다. 이 연구 결과는 AI 기반 선별의 신뢰성을 높이는 데 기여할 것으로 기대됩니다.



### Can OpenAI o1 Reason Well in Ophthalmology? A 6,990-Question Head-to-Head Evaluation Study (https://arxiv.org/abs/2501.13949)
Comments:
          44 pages

- **What's New**: 이번 연구에서는 OpenAI의 o1 모델과 다른 대형 언어 모델(LLMs)의 성능과 추론 능력을 검토하였습니다. 특히, 안과 분야에 특화된 6,990개의 질문을 통해 비교 분석을 수행했습니다. o1은 인식 정확도(accuracy)와 macro-F1 점수에서 최고 기록을 세웠지만, 텍스트 생성 메트릭스에서는 추론 능력 면에서 세 번째로 평가받았습니다.

- **Technical Details**: 연구에서는 o1이 "Lens"와 "Glaucoma" 주제에서 1위를 차지했지만, "Corneal and External Diseases", "Vitreous and Retina", 및 "Oculoplastic and Orbital Diseases"에서는 GPT-4o에 이어 두 번째로 위치했습니다. 또한, 하위 주제에 대한 분석을 통해 o1이 긴 정답 설명이 포함된 질문에 대해 더 나은 성과를 보였음을 확인했습니다.

- **Performance Highlights**: 연구 결과에 따르면, o1의 추론 능력이 안과 분야에 완전히 적용되지 않을 수 있다는 점이 강조되었습니다. 이는 안과와 같은 전문 분야에서 최적의 성능을 위해 도메인 구체적인 개선이 필요함을 시사합니다. 따라서, o1의 발전이 특정 분야에 어떻게 영향을 미치는지 이해하는 것이 중요합니다.



### Longitudinal Abuse and Sentiment Analysis of Hollywood Movie Dialogues using LLMs (https://arxiv.org/abs/2501.13948)
- **What's New**: 이번 연구는 1950년부터 2024년까지 헐리우드 영화의 대사(Dialogue)에 대한 장기적인 학대(Abuse) 및 정서(Sentiment) 분석을 수행하였습니다. Large Language Models (LLMs)를 활용하여 1,000개 이상의 영화 자막(Subtitles)을 분석하고, 감정과 폭력적 콘텐츠의 변화를 살펴보았습니다. 연구 결과, 최근 몇십 년 동안 영화 대사에서의 학대 내용은 지속적으로 증가하였고, 이는 사회적 규범과 정책의 변화反映를 나타냅니다.

- **Technical Details**: 본 연구에 사용된 LLM의 세부 사항은 BERT(Bidirectional Encoder Representation from Transformers) 기반 모델을 미세 조정(Fine-tuning)하여 감정 분석 및 폭력 및 학대 콘텐츠 탐지를 수행하는 것입니다. 정서 분석(Sentiment analysis)은 SenWave 데이터셋을 통해 수행되며, 장기적 연구(Longitudinal study)의 구성을 통해 영화 장르별로 데이터 패턴을 비교 분석합니다. 본 연구는 특정 시점의 측정이 아니라 시간에 따른 변동을 직접적(Direct)가로 살펴보는 방법론(Metodology)을 적용합니다.

- **Performance Highlights**: 연구 결과, 최근 20년 동안 헐리우드 영화에서의 학대 콘텐츠는 상당히 증가하고 있으며, 이는 오스카 수상 영화가 상위 블록버스터 영화의 내용을 초과하는 추세를 보였습니다. 스릴러(Thriller) 장르는 여전히 높은 빈도의 학대 콘텐츠를 포함하고 있으며, 긍정적인 감정을 포함한 유머와 낙관론이 대다수 영화에서 지속적으로 나타난다고 보고되었습니다. 본 연구는 다양한 장르에서 학대 및 감정 분석이 어떻게 상호작용하는지를 보여줍니다.



### A Comprehensive Survey on Integrating Large Language Models with Knowledge-Based Methods (https://arxiv.org/abs/2501.13947)
- **What's New**: 본 논문은 Large Language Models (LLMs)와 구조화된 지식 기반 시스템의 통합을 통해 인공지능(AI) 능력을 향상시키려는 접근 방식을 탐구합니다. LLMs의 생성적 언어 이해와 구조화된 시스템의 정확한 지식 표현을 결합함으로써 AI의 실제 응용 프로그램에서 발생할 수 있는 기술적, 운영적, 윤리적 도전 과제가 논의됩니다. 이 연구는 LLMs와 지식 기반 시스템 간의 시너지를 분석하여 데이터 맥락화, 모델 정확도 향상 및 지식 자원의 효율적 활용의 이점을 강조합니다.

- **Technical Details**: LLMs는 여러 자연어 처리(NLP) 작업에서 깊이 있는 이해와 생성을 가능하게 하는 거대한 파라미터 격자에 기반한 모델입니다. 이 모델들은 Transformer 아키텍처에서 개발되어, self-attention 메커니즘을 이용하여 텍스트 시퀀스를 효과적으로 처리하며, 과거의 통계적 언어 모델 및 신경망(NN) 모델에서 발전한 형태입니다. 더불어, Retrieval-Augmented Generation (RAG), Knowledge Graph 및 Prompt Engineering과 같은 통합 방법론을 통해 LLMs의 한계가 극복될 수 있는 가능성을 조사합니다.

- **Performance Highlights**: 이 논문은 LLMs에서의 주요 성과로서, 생성적 AI와 외부 지식 기반의 통합을 통해 모델의 정확성과 데이터 맥락화 및 계산 비용 절감 등을 달성할 수 있는 기회를 제공합니다. 또한, 최근 연구 동향 및 통합을 통한 LLMs의 응용 가능성을 식별하며, 이러한 접근 방법이 다양한 산업에서 AI 기술의 실질적 배치를 지원할 수 있는 방안을 제안합니다. 연구 결과는 LLMs의 현재 연구 상태를 개괄적으로 제공하며, 향후 발전에 필요한 실제적인 지침을 제시합니다.



### Hallucination Mitigation using Agentic AI Natural Language-Based Frameworks (https://arxiv.org/abs/2501.13946)
Comments:
          18 pages, 6 figures

- **What's New**: 본 연구는 여러 전문 AI 에이전트를 조율하여 Generative AI 모델에서 발생하는 hallucination 문제를 완화할 수 있는 방법을 제안합니다. 한 가지 주요 목표는 Natural Language Processing(NLP) 시스템을 통해 이러한 에이전트 간의 원활한 상호작용을 지원하는 것입니다. 이를 통해 신뢰할 수 있는 AI 시스템 구축이 가능하다는 점을 강조하고 있으며, 새로운 Key Performance Indicators(KPIs)를 활용해 hallucination의 수준을 평가합니다.

- **Technical Details**: 이 논문에서는 310개의 prompt를 설계하여 hallucination을 유도하고, 각 에이전트가 distinct한 large language 모델을 사용하여 이러한 출력물의 검토 및 수정 프로세스를 수행하는 multi-agent pipeline을 제안합니다. Open Voice Network(OVON) 프레임워크를 통해 에이전트 간의 오류 검출과 수정 과정을 JSON 메시지를 통해 원활하게 수행할 수 있도록 설계되었습니다. 또한, 다양한 KPIs는 hallucination 완화의 효과를 정량적으로 평가할 수 있는 구조적 프레임워크를 제공합니다.

- **Performance Highlights**: 결과적으로 다수의 전문 에이전트를 통해 hallucination 확률을 체계적으로 감소시킬 수 있으며, 이러한 시스템은 AI 커뮤니티 내에서의 신뢰성을 강화하는 데 기여할 것으로 기대됩니다. 특히, 실험 결과는 multi-agent pipeline이 프로세스를 거치면서 hallucination 점수가 향상되는 것을 시각적으로 나타내며, 이러한 개선이 AI 모델의 신뢰성 향상으로 이어질 수 있음을 보여줍니다.



### Self-Explanation in Social AI Agents (https://arxiv.org/abs/2501.13945)
Comments:
          Extended version of the paper published in International Conference on Intelligent Tutoring Systems, pages 351-360, 2024, Springer. Images corrected, and live deployment, ablation, and precision study results added

- **What's New**: 본 연구에서는 사회적 AI 에이전트가 커뮤니티 멤버와 상호 작용하여 그들의 행동을 변화시키는 방식에 대해 다룹니다. 특히, 온라인 학습 환경에서 AI 사회 보조자가 학습자 간의 상호 작용을 증진할 수 있는 가능성에 주목합니다. AI 보조자가 학습자에게 자신의 기능을 설명함으로써 투명성과 신뢰성을 높일 수 있는 방법을 제시합니다.

- **Technical Details**: 연구에서는 AI 사회 보조자의 자기 모델(self-model)을 통해 반성(introspection)을 활용한 자기 설명(self-explanation) 기법을 개발합니다. 이 자기 모델은 AI 에이전트가 작업을 수행하기 위해 지식을 어떻게 사용하는지를 정의하는 기능적 모델로 포착됩니다. 자기 설명 생성 과정에서는 Chain of Thought 기법을 통해 자신을 반성하고, ChatGPT를 이용해 그 기능에 대한 설명을 제공합니다.

- **Performance Highlights**: AI 사회 보조자의 자기 설명을 완전성과 정확성 측면에서 평가하고, 실제 수업에서 이 기술의 배포에 대해 보고합니다. 연구 결과, 이러한 자기 설명 기능이 학습자들의 신뢰를 증진시키는데 긍정적인 영향을 미쳤음을 확인했습니다.



### Fanar: An Arabic-Centric Multimodal Generative AI Platform (https://arxiv.org/abs/2501.13944)
- **What's New**: Fanar는 아랍어 중심의 다중 모드 생성 AI 시스템을 위한 플랫폼으로, 언어, 음성 및 이미지 생성 작업을 지원합니다. Fanar의 핵심에는 두 가지 강력한 아랍어 대형 언어 모델인 Fanar Star와 Fanar Prime이 있으며, 이들은 유사 모델군에서 검증된 벤치마크에서 최고의 성능을 자랑합니다.

- **Technical Details**: Fanar Star 모델은 70억 개의 매개변수(parameter)를 가지고 있으며, 거의 1조 개의 깨끗하고 중복되지 않은 아랍어, 영어 및 코드 토큰으로 처음부터 훈련되었습니다. 반면 Fanar Prime 모델은 90억 개의 매개변수를 가지고 있으며, 동일한 1조 개의 토큰 세트를 기반으로 Gemma-2 9B 모델을 통해 지속적으로 훈련되었습니다.

- **Performance Highlights**: Fanar 플랫폼은 종교 관련 프롬프트를 처리하기 위한 맞춤형 이슬람 검색 증강 생성(RAG) 시스템과 최근 정보 요약을 위한 최신 RAG 기능을 제공합니다. 추가적으로, 플랫폼은 다양한 아랍어 방언을 지원하는 이중 언어 음성 인식, 지역 특성을 잘 반영한 음성 및 이미지 생성을 제공합니다.



### Language Representation Favored Zero-Shot Cross-Domain Cognitive Diagnosis (https://arxiv.org/abs/2501.13943)
- **What's New**: 이 논문은 언어 표현 기반의 제로 샷 크로스 도메인 인지 진단(Language Representation Favored Zero-shot Cross-domain Cognitive Diagnosis, LRCD) 접근법을 제안합니다. 기존의 인지 진단 모델들은 특정 도메인에 맞춰 모델을 훈련해야 하며, 이는 다양한 과목이나 교육 플랫폼에서의 직접적인 응용을 어렵게 만듭니다. LRCD는 텍스트 설명을 사용하여 학생, 과제 및 개념의 프로필을 만들고, 이를 통일된 언어 공간에 벡터화하여 인지 진단 모델과 통합하는 방법을 제안합니다.

- **Technical Details**: LRCD는 학생과 과제의 행위 패턴을 분석하고, 각 도메인에서 이를 텍스트로 설명합니다. 이렇게 생성된 프로필은 최신 텍스트 임베딩 모듈을 통해 통합 언어 공간의 벡터로 변환됩니다. 그러나 언어 공간과 인지 진단 공간 간의 불일치를 해결하기 위해, LRCD는 두 공간 간의 매핑을 학습하는 언어-인지 맵퍼를 제안합니다. 이를 통해 기존의 인지 진단 모델과 통합하여 효율적으로 훈련할 수 있습니다.

- **Performance Highlights**: LRCD는 다양한 실제 데이터셋에 대해 제로 샷 성능을 commendable하게 달성할 수 있으며, 특정 경우에는 기존의 인지 진단 모델과 경쟁할 만한 성능을 보여줍니다. 학계 및 교육 현장에서 학생들의 과목 간 차이를 분석하는 데에도 유용한 인사이트를 제공합니다. 흥미롭게도, 과학 과목에서 생성된 데이터는 다양한 목표 도메인에서 더 나은 전이 성능을 보이며, 고등 교육 수준의 학생 데이터를 기반으로 할 경우 저학년 학생들에게도 더 큰 전이 가능성을 보입니다.



### GaussMark: A Practical Approach for Structural Watermarking of Language Models (https://arxiv.org/abs/2501.13941)
- **What's New**: 최근 Large Language Models (LLMs)의 발전은 자연어 처리(NLP) 작업에 큰 개선을 가져왔지만, 사람과 같은 품질의 텍스트 생성을 가능하게 하는 이 기술은 윤리적 및 운영상의 우려를 야기하고 있습니다. 이에 따라, LLM으로 생성된 텍스트를 확인할 수 있는 수단인 워터마킹(watermarking) 기법 개발에 대한 연구가 이루어졌습니다. 현재의 기존 워터마킹 기법은 생성 지연(generation latency)과 탐지 시간(detection time), 텍스트 품질 저하 등 여러 측면에서 실용적이지 않은 경우가 많았으며, 이 문제들을 해결하기 위한 새로운 접근 방식이 필요합니다.

- **Technical Details**: 이 연구에서는 GaussMark라는 새로운 워터마킹 기법을 소개합니다. 이 방법은 구현하기 간단하고 효율적이며, 형태학적인 워터마크(structural watermark)를 모델의 가중치(weights) 자체에 내장합니다. Gaussian 독립성 테스트를 기반으로 하고 있으며, LLM의 가중치에 소량의 Gaussian 노이즈를 추가하여 워터마킹을 수행합니다. 이 과정은 통계적으로 검출 가능하도록 설계되었으며, 비밀 키를 가진 제공자가 이 텍스트가 자신의 모델에 의해 생성되었다는 것을 확인할 수 있게 합니다.

- **Performance Highlights**: GaussMark는 광범위한 실험을 통해 신뢰성과 효율성을 입증하였으며, 삽입, 삭제, 치환, 왕복 번역(roundtrip translation)과 같은 복원력 있는 다양한 왜곡에 대해 비교적 강력함을 보여줍니다. 이 방법은 모델 품질 손실 없이 거의 무제한의 성능 범위를 제공합니다. 또한, 우리의 접근 방식은 생성 과정에서 추가적인 지연을 발생시키지 않아 실제 적용에 적합합니다.



New uploads on arXiv(cs.LG)

### MLPs at the EOC: Concentration of the NTK (https://arxiv.org/abs/2501.14724)
Comments:
          36 pages, 1 figure

- **What's New**: 이번 연구에서는 $l$-레이어 다층 퍼셉트론(MLP)의 신경 탄젠트 커널(Neural Tangent Kernel, NTK)의 집중(concentration) 현상을 다루고 있습니다. 특히 Edge Of Chaos (EOC)에서 초기화된 파라미터를 갖고 있는 활성화 함수 $	ext{phi}(s) = a s + b |s|$에 대해 다룹니다.

- **Technical Details**: 연구는 최대 부등식(maximal inequalities)을 통해 데이터셋 $\\{x_1,\cdots,x_n\}$에 대해 NTK의 항들이 동시에 집중함을 보여줍니다. 결과적으로 NTK 행렬 $K(\theta)$는 여유 파라미터에 대한 필요 없이 무한대 너비 한계($\overset{\scriptscriptstyle\infty}{K}$)로 집중하는 것을 입증합니다.

- **Performance Highlights**: MLP의 숨겨진 층 너비는 집중을 위해 $m_k = k^2 m$과 같은 제곱적으로 증가해야 합니다. 연구 결과는 특정한 조건에서 NTK의 집중도를 비교하면서 절댓값 경우가 ReLU보다 더 집중된다는 점을 보여줍니다.



### CodeMonkeys: Scaling Test-Time Compute for Software Engineering (https://arxiv.org/abs/2501.14723)
- **What's New**: 이번 논문에서는 LLM(대형 언어 모델)의 테스트 시간 컴퓨팅을 확장하는 방법을 탐구합니다. 이를 실현하기 위해, CodeMonkeys라는 시스템을 개발하여 GitHub의 실제 문제를 해결하는 데 초점을 맞추었습니다. 이 시스템은 모델이 코드베이스를 반복적으로 수정하고, 테스트 스크립트를 생성 및 실행할 수 있도록 하는 멀티 턴(multi-turn) 프로세스를 사용하여 다양한 후보 수정을 생성합니다.

- **Technical Details**: SWE-bench 데이터셋을 활용하여 코드베이스 수정 문제를 해결하는 세 가지 주요 단계가 정의됩니다: 1) 관련 코드베이스 컨텍스트 식별하기, 2) 문제 해결을 위한 후보 코드베이스 수정 생성하기, 3) 후보 수정 중 선택하기. 이 과정에서 Serial과 Parallel 테스트 시간 컴퓨팅을 결합하여 진행하며, 코드 수정 시 테스트 스크립트를 작성하도록 하여 반복적으로 수정 및 실험 결과에 응답할 수 있도록 합니다.

- **Performance Highlights**: CodeMonkeys 시스템은 총 비용 약 2300 달러를 들여 SWE-bench Verified 문제에서 57.4%의 해결률을 달성했습니다. 또한, 기존 SWE-bench Verified 제출물 중 상위 4개를 조합하여 생성한 편집 후보군을 선택하여 66.2%의 성과를 기록, 가장 좋은 개별 성과보다도 높은 성과를 보여주었습니다. 논문에서 제시하는 메커니즘은 코드 수정 후보군의 결합과 선택에 유용하게 활용될 수 있음을 입증했습니다.



### The Karp Datas (https://arxiv.org/abs/2501.14705)
Comments:
          Accepted to the 4th workshop on mathematical reasoning and AI at NeurIPS 2024

- **What's New**: 이 논문은 Large Language Models (LLMs)의 수학적 추론 능력을 이해하는 데 중점을 두고 있으며, 이를 위한 새로운 Karp 데이터셋을 소개합니다. Karp 데이터셋은 NP-완전성 증명의 구체적인 사례들로 구성되어 있으며, 다양한 난이도로 이루어져 있어 LLM의 성능을 평가하는 데 사용될 수 있습니다. 이는 기존 데이터셋이 단순한 숫자나 기호의 결과에 집중하는 것에 비해, 더 복잡한 논리적 사고를 요구하는 연구입니다.

- **Technical Details**: Karp 데이터셋은 NP-난해성 증명을 위한 다양한 변환 과정을 문서화한 자연어 설명들로 구성되어 있습니다. 데이터셋은 명확한 정의와 함께 정확성을 증명하는 구조화된 템플릿을 따릅니다. 모든 변환 예시는 저명한 문헌에서 출처를 두고 있으며, 각 변환의 길이는 평균 약 2000자가 넘는 긴 설명을 포함하고 있습니다. 이러한 구조적이고 교육적인 접근 방식은 LLM이 증명을 이해하고 처리하는데 유용하도록 돕습니다.

- **Performance Highlights**: 현재 LLM은 MATH, GSM8K, MGSM 등의 기존 데이터셋에서 뛰어난 성능을 보이고 있지만, NP-완전성 증명의 변환을 다루는 새로운 Karp 데이터셋에 대한 평가가 이루어질 때 더 높은 임계값이 설정될 것으로 기대됩니다. LLM의 학습과 성능 평가에 있어 Karp 데이터셋이 새로운 기준을 제공할 것이며, 특히 높은 수준의 수학적 문제에 대한 이해도를 높이는 데 기여할 수 있습니다.



### An Attentive Graph Agent for Topology-Adaptive Cyber Defenc (https://arxiv.org/abs/2501.14700)
- **What's New**: 사이버 공격이 더 복잡해지고 예측할 수 없게 되면서, 적응형 자율 방어 시스템의 개발이 사이버 보안 연구의 주요 초점이 되고 있습니다. 이 논문에서는 Graph Neural Networks (GNNs)와 Graph Attention Networks (GATs)를 활용하여 컴퓨터 네트워크의 고유 그래프 구조를 통해 자율 방어 정책의 효과성을 높이는 새로운 접근법을 제시합니다. 이를 통해 트레이닝 중 본 적이 없는 다양한 네트워크 구성에서도 잘 작동할 수 있도록 일반화 능력을 향상시킬 수 있습니다.

- **Technical Details**: CybORG 시뮬레이터를 기반으로 한 새로운 환경에서, 네트워크 상태를 방향 그래프로 인코딩하여 실시간 변화하는 네트워크 토폴로지에 적응할 수 있는 정책을 개발했습니다. GAT 아키텍처를 적용하여 노드, 엣지, 글로벌 기능을 처리하고, 이 출력을 강화 학습의 정책 기울기 방법에 맞게 조정했습니다. 이 접근법은 기존의 평탄한 상태 관찰 방식에 비해 여러 가지 장점이 있으며, 저수준 그래프 관찰을 통해 조작 가능한 방어 정책을 구축할 수 있게 해줍니다.

- **Performance Highlights**: 우리의 훈련된 정책은 다양한 크기의 네트워크에서 평가되었으며, 동일한 서브네트워크 구조를 가진 네트워크에서도 잘 동작함을 보여주었습니다. 특히, 기존의 정책과 비교했을 때, GAT 기반 정책은 다이나믹 연결 변화에 대한 적응력과 해석 가능성을 높임으로써 더 강력하고 유연한 사이버 방어 시스템을 개발하는데 기여합니다. 이 연구는 실제 네트워크 보안 문제에 대한 방어 능력을 크게 개선할 수 있는 가능성을 보여 줍니다.



### Towards Automated Self-Supervised Learning for Truly Unsupervised Graph Anomaly Detection (https://arxiv.org/abs/2501.14694)
Comments:
          Manuscript submitted to Data Mining and Knowledge Discovery in May 2024 for possible publication. This is the revised version submitted in January 2025

- **What's New**: 이 논문은 Self-supervised Learning (SSL) 기법이 그래프 이상 탐지에 효과적이나, 성능에 큰 영향을 미치는 여러 요인들(SSL 전략, 하이퍼파라미터 튜닝, 전략 조합 가중치 할당)을 보완하기 위한 방법을 제시합니다. 특히, Label Information Leakage 문제를 지적하며, 이를 해결하기 위해 내부 평가 전략을 활용하여 하이퍼파라미터를 선택하는 방안을 제안합니다. 이는 기존 방법들과는 달리, 레이블 정보를 사용하지 않고도 SSL 최적화를 이룰 수 있는 방안으로, 실용적인 그래프 이상 탐지를 가능하게 합니다.

- **Technical Details**: 이 논문은 정적인 속성 그래프에서의 노드 이상 탐지에 초점을 맞춥니다. 연구에서는 최근 발표된 SSL 기반 그래프 이상 탐지 알고리즘 10개를 다양한 벤치마크 데이터셋에서 실험하여 하이퍼파라미터 선택과 우리의 제안한 전략의 효과를 입증합니다. 또한, 모듈 간 상호작용이 강조되며, 데이터 증강 기법, 하이퍼파라미터 선택 및 다양한 SSL 전략들을 결합하는 방법에 대한 문제를 다룹니다.

- **Performance Highlights**: 연구 결과, AutoGAD 접근방식이 기존의 수동적인 방법들보다 월등한 성능을 보이며, 서로 다른 데이터셋에 특화된 최적화가 필요한 것을 보여주었습니다. 이 논문에서는 SSL 기반의 그래프 이상 탐지 방법들이 자주 겪는 성능의 과대 추정 문제를 해결하는 중요한 단서를 제공하며, 자동화된 과정이 실제 환경에서 유용하게 활용될 수 있음을 입증합니다. 실험을 통해 제안한 방법이 다양한 데이터셋에서 기존의 높은 성능을 가진 알고리즘에 필적하거나 이를 초월하는 결과를 나타냈습니다.



### Decoding Generalization from Memorization in Deep Neural Networks (https://arxiv.org/abs/2501.14687)
- **What's New**: 이 논문은 Deep Learning의 overparameterized 모델이 어떻게 잘 일반화되는지를 탐구하며, 이러한 모델들이 기억(memorization) 능력을 갖고 있지만 여전히 잘 일반화될 수 있다는 새로운 증거를 제시합니다. 특히, 특정 층의 출력에서 정보가 추출 가능하다는 점을 강조하여, 메모리 효과가 있는 모델임에도 불구하고 좋은 일반화 성능을 나타낼 수 있음을 보여줍니다.

- **Technical Details**: 저자들은 여러 Deep Neural Networks의 층별 출력에서 클래스 조건 하의 서브스페이스 조직을 연구하고, Principal Components Analysis (PCA)를 사용하여 이들 서브스페이스를 추정합니다. 제안된 분류기(classifier)는 입력 데이터 포인트의 출력 벡터와 해당 클래스 서브스페이스 간의 각도를 측정하여 클래스를 결정하며, 이를 통해 네트워크가 좋은 일반화 가능성을 가지는지 확인합니다. 이 접근 방식은 corrupted training data에 대해서도 적합합니다.

- **Performance Highlights**: 실험 결과, label noise로 손상된 데이터로 훈련된 모델임에도 불구하고, 단순한 분류기를 통해 높은 테스트 정확도를 달성할 수 있음을 발견했습니다. 또한, 훈련된 클래스 레이블이 사후에 알려지면 일관되게 더 나은 일반화 성능을 발휘하는 분류기를 구축할 수 있음을 보여주었습니다. 이러한 발견은 메모리화와 일반화가 동시에 존재할 수 있다는 가능성을 시사합니다.



### Neural-Symbolic Message Passing with Dynamic Pruning (https://arxiv.org/abs/2501.14661)
Comments:
          19 pages, 5 figures, 16 tables

- **What's New**: 본 논문에서는 Neural-Symbolic Message Passing (NSMP) 프레임워크를 제안하며, 그것은 사전 훈련된 신경 링크 예측기를 기반으로 합니다. NSMP는 1단계 추론을 수행하기 위해 상징적 추론과 퍼지 논리를 통합하여 복잡한 쿼리에 대해 훈련 없이도 일반화할 수 있습니다. 또한, NSMP는 변동 노드 간의 노이즈 메시지를 필터링하는 동적 가지치기 전략을 도입하여 해석 가능한 답변을 제공합니다.

- **Technical Details**: NSMP 프레임워크는 메시지 전달 방식으로 작동하며, 변동 노드의 중간 상태를 계산하는 데 신경 및 상징적 추론을 통합합니다. 이를 통해 변동 노드 간의 메시지를 효과적으로 집계하고 업데이트하여 퍼지 벡터로 표현된 중간 상태가 해석될 수 있도록 합니다. 특히, NSMP는 필요 없는 노이즈 메시지를 동적으로 제거하여 성능을 개선하는 가지치기 전략을 채택합니다.

- **Performance Highlights**: 실험 결과에 따르면, NSMP는 다양한 쿼리 유형에 대해 이전의 최첨단(neural-symbolic) 모델보다 2배에서 150배 이상 빠른 추론 시간을 보이며, 특히 부정 쿼리에서 유의미한 성능 향상을 이룹니다. 이러한 결과는 NSMP가 복잡한 쿼리 데이터에 대한 훈련 없이도 강력한 성과를 달성할 수 있음을 보여줍니다.



### MedAgentBench: Dataset for Benchmarking LLMs as Agents in Medical Applications (https://arxiv.org/abs/2501.14654)
- **What's New**: 최근 대형 언어 모델(LLMs)은 챗봇의 전통적인 역할을 넘어 에이전트로서의 기능에서 유의미한 발전을 보여주었습니다. 이러한 에이전트는 고수준의 작업을 수행하기 위해 계획 및 도구 활용 능력을 적극적으로 활용할 수 있습니다. 그러나 의료 분야에서 LLM의 에이전트 기능을 평가할 수 있는 표준화된 데이터셋은 부족한 상황입니다.

- **Technical Details**: 이 논문에서는 MedAgentBench라는 평가 도구를 도입했습니다. MedAgentBench는 10개 카테고리에서 임상의에 의해 작성된 100개의 환자 맞춤 임상 과제를 포함하고 있으며, 700,000개 이상의 데이터 요소가 포함된 100명의 현실적인 환자 프로필을 제공합니다. 이 환경은 현대 전자 의료 기록 시스템에서 사용되는 표준 API와 통신 인프라를 바탕으로 구축, 실시간 EMR 시스템으로 쉽게 이식할 수 있는 기능을 갖추고 있습니다.

- **Performance Highlights**: 현재 가장 발전된 모델인 GPT-4o는 72%의 성공률을 기록했으나, 에이전트 기능 향상을 위한 많은 개선 여지가 남아 있습니다. 또한, 작업 카테고리에 따라 성능의 변동이 크며, MedAgentBench는 이러한 변화를 명확히 하고 있어 모델 개발자들에게 유용한 프레임워크를 제공합니다. 이 프로젝트는 공개적으로 제공되어 의료 도메인 내 LLM의 에이전트 기능 상승을 위한 중요한 초석이 될 것입니다.



### Federated Domain Generalization with Data-free On-server Gradient Matching (https://arxiv.org/abs/2501.14653)
Comments:
          26 pages, 15 figures, ICLR

- **What's New**: 이번 논문에서는 분산된 도메인에서 도메인 불변 모델을 학습할 수 있는 새로운 접근 방식인 FedOMG (Federated Learning via On-server Matching Gradient)를 제안합니다. 기존의 Federated Domain Generalization (FDG) 문제를 해결하며, 기존의 통신 비용 없이도 중앙 서버에서 분산 모델의 특성을 집계할 수 있는 방법을 제공합니다. FedOMG는 로컬 그래디언트를 활용하여 도메인 불변 그래디언트 방향을 찾는 방식을 채택합니다.

- **Technical Details**: FedOMG는 Gradient Inner Product (GIP) 최적화 문제에서 영감을 받아 설계되었습니다. 이는 각 도메인에서 유도된 그래디언트의 높은 상관관계를 기반으로 하며, 두 가지 주요 한계를 극복합니다: 1) GIP의 직접적인 최소화는 모델 파라미터의 2차 미분을 필요로 하여 계산 비용이 많이 드는 점과, 2) 클라이언트 모델의 지속적 전송이 필요하여 통신 오버헤드가 과도해지는 문제입니다. FedOMG는 이러한 문제를 간접 최적화 방식을 통해 해결하며, 효율적인 볼록 최적화 수식을 도입하여 서버 측 최적화를 간소화합니다.

- **Performance Highlights**: 실험 결과, FedOMG는 MNIST, EMNIST, CIFAR-10, CIFAR-100과 같은 네 가지 FL 벤치마크 데이터셋과 PACS, VLCS, OfficeHome의 세 가지 FDG 벤치마크 데이터셋에서 최신 성능 기준(SOTA)을 능가했습니다. FedOMG는 IID 및 non-IID 데이터셋과 OOD 일반화 상황에서 기존 FL 방법보다 우수한 성능을 보였습니다. 이러한 우수함은 FedOMG가 통신 효율성과 개인 정보 보호를 동시에 유지할 수 있도록 합니다.



### Decoupled SGDA for Games with Intermittent Strategy Communication (https://arxiv.org/abs/2501.14652)
- **What's New**: 본 논문은 멀티플레이어 게임에서 전략을 통해 발생하는 통신 오버헤드를 줄일 수 있는 방법을 모색합니다. 새로운 방법인 Decoupled SGDA를 소개하며, 이를 통해 플레이어들은 상대 플레이어의 전략이 오래된 상태이더라도 독립적으로 자신의 전략을 업데이트할 수 있습니다. 이 방식은 강력하고 볼록-오목(SCSC) 게임에서도 최적에 가까운 통신 복잡도를 달성할 수 있음을 보여줍니다.

- **Technical Details**: Decoupled SGDA는 Stochastic Gradient Descent Ascent(SGDA)를 기반으로 하며, 각 플레이어는 상대의 오래된 전략에 따라 K번 업데이트를 한 후 동기화 단계에서 전략을 교환합니다. 이러한 접근 방식은 통신이 낮거나 간헐적인 상황에서도 효과적으로 작동하게 설계되었습니다. 추가적으로 매우 약한 결합 게임(Weakly Coupled Games)에서 통신 비용이 크게 줄어드는 것을 확인하였습니다.

- **Performance Highlights**: 본 연구는 Decoupled SGDA가 비선형 박스(Minimax) 최적화와 비볼록 GAN 설정에서 효과적임을 수치적 실험을 통해 입증하였습니다. 또한 플레이어 간의 노이즈가 불균형한 환경에서도 기존의 연합(minimax) 기법에 비해 성능이 크게 향상됨을 보여주었습니다. 이러한 성과는 알고리즘의 수렴 행동에 대한 심층 분석과 함께 제시되었습니다.



### Whisper D-SGD: Correlated Noise Across Agents for Differentially Private Decentralized Learning (https://arxiv.org/abs/2501.14644)
Comments:
          6 pages, 3 figures, preprint

- **What's New**: 이번 논문은 Whisper D-SGD라는 새로운 covariance 기반 접근 방식을 소개하며, 이는 분산된 에이전트 간에 상관관계가 있는 프라이버시 노이즈를 생성하여 기존 기법들을 통합합니다. 이 기법은 에이전트들이 로컬 모델을 업데이트하고 직접 이웃들과 혼합할 수 있도록 하여, 분산 학습에서의 개인 정보 보호를 강화합니다. Whisper D-SGD는 네트워크 토폴로지와 혼합 가중치를 활용하여 노이즈를 최적화함으로써, 기존의 방식과 비교해 더욱 효과적인 노이즈 캔슬링을 달성합니다.

- **Technical Details**: 논문에서는 n개의 에이전트가 각자의 로컬 데이터셋을 보유하고 있으며, 공동의 기계 학습 모델 파라미터를 학습하기 위해 글로벌 목적을 최소화한다는 내용을 다룹니다. 이 과정에서 자주 사용되는 알고리즘인 Decentralized Stochastic Gradient Descent (D-SGD)를 활용하여 에이전트들이 이웃과 파라미터를 교환하고 가중 평균을 계산하는 두 단계의 과정을 제시합니다. 이 과정에서 사용되는 혼합 매트릭스 W는 에이전트 간의 통신을 수립하는 데 핵심적인 역할을 합니다.

- **Performance Highlights**: 실험 결과 Whisper D-SGD는 기존의 pairwise-correlation 방식보다 더 많은 노이즈를 캔슬하며, CDP-LDP 간의 격차를 줄이고 동일한 프라이버시 보장 하에서도 모델 성능을 향상시킵니다. 특히 네트워크 연결성이 약한 환경에서도 우수한 프라이버시-효용(trade-off)을 달성하는 것으로 나타났습니다. 이러한 결과는 Whisper D-SGD가 프라이버시 보호와 모델 성능 간의 균형을 잘 맞출 수 있는 가능성을 보여줍니다.



### Towards Scalable Topological Regularizers (https://arxiv.org/abs/2501.14641)
Comments:
          31 pages, accepted to ICLR 2025

- **What's New**: 이번 연구에서는 latent space matching에서 persistent homology를 활용한 새로운 topological regularizer인 principal persistence measures(PPM)을 제안합니다. 이는 topology의 특성을 고려하면서도 계산 효율성을 높이는 방법으로, adversarial tasks, 도메인 적응, generative modeling에서 활용될 수 있습니다. PPM은 많은 작은 서브샘플의 persistent homology를 계산하여 얻어지며, 이를 통해 안정적인 훈련 결과를 기대할 수 있습니다.

- **Technical Details**: 본 연구의 핵심 방법은 두 가지 구성 요소, 즉 topological summary와 metric을 수정하는 것입니다. principal persistence measures(PPM)의 경우, 작은 배치에서 persistent homology를 병렬로 계산하여 computational cost를 줄입니다. 또한, PPM을 비교하기 위한 metric으로 Wasserstein distance 대신 maximum mean discrepancy(MMD)를 사용하여 계산 비용을 줄이고, gradient의 연속성을 보장합니다.

- **Performance Highlights**: 실험 결과, PPM-Reg는 GAN 프레임워크에서 효과적으로 동작하며, shape matching, 이미지 생성, 반감독 학습 등의 다양한 태스크에서 topological feature를 안정적으로 통합할 수 있음을 보여주었습니다. 이 연구는 규모가 큰 머신러닝 작업에서 topological feature를 성공적으로 포함할 수 있는 가능성을 열어줍니다.



### A Paired Autoencoder Framework for Inverse Problems via Bayes Risk Minimization (https://arxiv.org/abs/2501.14636)
Comments:
          22 pages, 9 figures

- **What's New**: 본 연구에서는 머신 러닝, 특히 autoencoder 네트워크 구조를 활용한 새로운 데이터 기반 접근 방식을 설명합니다. 두 개의 autoencoder를 사용하여 입력 공간과 목표 공간을 효율적으로 표현하고, 이들 사이의 최적 매핑을 학습하여 전방 및 역방향 surrogate 매핑을 가능하게 합니다. 또한 Bayes risk와 empirical Bayes risk 최소화를 통해 이론적인 결과를 제시하며, 기존의 저랭크 행렬 근사화 기법과의 연결성을 제공합니다.

- **Technical Details**: 이 연구에서는 관측값을 매개변수로 매핑하는 역방향 맵을 학습하기 위해 autoencoder를 훈련시킵니다. 이로써, 입력과 목표 간의 매핑을 학습하고, 이를 통해 개별 latent 공간의 매핑을 발견합니다. 제안된 프레임워크는 Bayes risk와 empirical Bayes risk 최소화 이론을 기반으로 하며, 이는 특정 전제 조건하에 다양한 문제를 접근할 수 있도록 합니다.

- **Performance Highlights**: PAIR 프레임워크는 훈련 샘플이 풍부하지만 입력-목표 쌍이 부족한 문제에서 기존 접근 방식보다 우수한 성능을 보입니다. 또한 모델 및 차원 축소 프로세스를 분리하여 서로 다른 차원의 latent 공간을 활용할 수 있어, 데이터의 불확실성을 효과적으로 전달할 수 있습니다. PAIR 접근법은 역문제에 대한 새로운 통찰력을 제공하며, 수학적 최적화 및 과학 컴퓨팅에 널리 적용될 수 있습니다.



### ACT-JEPA: Joint-Embedding Predictive Architecture Improves Policy Representation Learning (https://arxiv.org/abs/2501.14622)
- **What's New**: 이번 연구에서는 모방 학습(imitation learning)과 자기 지도 학습(self-supervised learning)의 통합을 통해 정책 표현(policy representations)을 향상시키는 새로운 아키텍처인 ACT-JEPA를 제안합니다. 기존의 모방 학습은 전문가의 시연에 의존하여 고비용이 소요되며, 세계 모델(world model)이 잘 발전되지 않는 한계가 있습니다. ACT-JEPA는 다양한 비표시 데이터로부터 학습할 수 있는 가능성을 제공하여 이러한 문제를 해결하고자 하였습니다.

- **Technical Details**: ACT-JEPA는 정책(policy)을 학습하기 위해 두 가지 주요 목표를 설정합니다: (1) 행동 시퀀스(action sequences)의 예측과 (2) 추상 관찰 시퀀스(abstract observation sequences)의 예측입니다. 첫 번째 목표는 행동 청킹(action chunking)을 활용하여 행동 예측을 개선하고 오류를 줄입니다. 두 번째 목표는 추상 관찰 시퀀스를 예측하여 청킹의 개념을 확장함으로써 모델이 불필요한 세부 사항을 필터링할 수 있게 합니다.

- **Performance Highlights**: 실험 결과, ACT-JEPA는 시간적 환경 역학을 학습함으로써 표현의 질을 개선하는 것으로 나타났습니다. 또한, 추상 관찰 시퀀스를 예측하는 능력 덕분에 행동 시퀀스 예측에 효과적으로 일반화되는 표현을 생성합니다. ACT-JEPA는 여러 의사결정 태스크에서 기존 기준 모델들과 동등한 성능을 보였습니다.



### Inverse Evolution Data Augmentation for Neural PDE Solvers (https://arxiv.org/abs/2501.14604)
- **What's New**: 본 연구에서는 진화 방정식(evolution equations)을 위한 신경 연산자(neural operator) 훈련에 최적화된 새로운 데이터 증강(data augmentation) 방법을 제안합니다. 이 방법은 역 과정(inverse processes)의 통찰력을 활용하여 무작위 초기화(random initialization)에서 데이터를 효율적으로 생성하고, 원본 데이터와 결합하는 방식을 적용합니다. 또한, 고차 역 진화 방법(high-order inverse evolution schemes)을 도입하여 생성된 데이터의 정확성을 강화합니다.

- **Technical Details**: 데이터 생성은 해당 진화 방정식의 역 과정을 이용하여 이루어지며, 몇 번의 명시적(exPLICIT) 계산 단계만으로도 결과 데이터 쌍을 생성할 수 있습니다. 이 과정에서 생성된 데이터는 해당하는 암시적(implicit) 수치 기법을 만족함을 증명할 수 있습니다. 전통적인 수치 해법이 요구하는 작은 시간 간격(time steps)에 비해, 우리의 방법은 상대적으로 큰 시간 간격을 허용하여 계산 비용(computational costs)을 크게 줄입니다.

- **Performance Highlights**: 포리에 신경 연산자(Fourier Neural Operator)와 UNet을 이용한 실험 결과, 제안한 새로운 데이터 증강 방법이 성능과 강인성을 크게 향상시킨다는 것을 확인하였습니다. 특히, 생성된 데이터는 원본 데이터에 비해 더 높은 주파수 성분을 포함하고 있어 솔루션 공간을 확장하는데 기여하고 있습니다. 이러한 성과는 진화 방정식 해결을 위한 신경 연산자의 활용성을 높여줄 것입니다.



### Age and Power Minimization via Meta-Deep Reinforcement Learning in UAV Networks (https://arxiv.org/abs/2501.14603)
Comments:
          10 pages, 8 figures

- **What's New**: 이 연구는 비행 드론(UAV)을 활용한 IoT 네트워크에서의 데이터 수집을 지원하는 전력 제한 환경을 다룹니다. 특히 변동하는 Age-of-Information (AoI)와 전송 전력을 최소화하기 위한 UAV의 비행 경로 및 스케줄링 정책을 최적화하는 데 집중하고 있습니다. 제안된 메타 딥 강화 학습(meta-deep reinforcement learning) 접근법은 DQNs와 MAML을 통합하여, 빠른 수렴 속도와 변화하는 목표에 대한 강력한 적응성을 제공합니다.

- **Technical Details**: 이 논문에서는 DQN(Deep Q-Network)과 MAML(Model-Agnostic Meta-Learning)을 결합하여 UAV의 비행 경로 및 스케줄링 정책을 최적화합니다. DQN은 상태에서 최적의 결정을 내리는 반면, MAML은 다양한 목표 함수들에 대한 확장성을 제공합니다. 이 접근법은 하이퍼파라미터 조정을 통해 UAV의 동적 환경에서의 최적화 문제에 빠르게 적응할 수 있게 합니다.

- **Performance Highlights**: 제안된 메타 딥 강화 학습 알고리즘은 MAML이 없는 기존 강화 학습 모델보다 새로운 목표에 빠르게 적응하며, 경험 샷 수가 적고, AoI와 전송 전력을 전반적으로 최적화하는 성능을 보였습니다. 특히, 실험 결과는 UAV 지원 IoT 네트워크에서 실시간 데이터 수집과 전력 효율성을 모두 만족할 수 있는 가능성을 보여주고 있습니다.



### Data Assetization via Resources-decoupled Federated Learning (https://arxiv.org/abs/2501.14588)
- **What's New**: 이번 연구에서는 자원 분리 (resource-decoupled) 연합 학습 (Federated Learning, FL) 환경을 규명하고, 모델 소유자, 데이터 소유자, 컴퓨팅 센터 간의 최적화 방법을 제시합니다. 개인 정보 보호 문제를 고려하여 데이터를 직접 전송하는 대신 정보의 흐름을 통해 데이터 가치를 극대화하고자 하는 방안을 탐구했습니다. 특히, 품질을 인식하는 동적 자원 분리 FL 알고리즘(QD-RDFL)을 제안하고, 이를 통해 데이터 소유자의 기여도를 평가하여 글로벌 모델의 성능을 향상시키려는 노력이 돋보입니다.

- **Technical Details**: 이 연구는 데이터 자원의 이질성 문제를 다루며, 세 가지 역할(모델 소유자, 데이터 소유자, 컴퓨팅 센터) 간의 상호작용을 위해 삼자 스탈켈버그 모델(Tripartite Stackelberg Model)을 설계했습니다. 스탈켈버그-내쉬 균형(SNE)을 이론적으로 분석하여 모든 당사자가 글로벌 유틸리티를 최적화할 수 있도록 하는 전략을 도출하였으며, 후진 귀납법(backward induction) 및 동적 최적화 메커니즘을 활용했습니다. 이를 통해 데이터 품질 평가를 통한 기여도 평가를 가능케 했습니다.

- **Performance Highlights**: 우리의 실험 결과에 따르면, 제안된 QD-RDFL 알고리즘이 세 당사자 간의 협력을 효율적으로 촉진하여 글로벌 유틸리티와 데이터 자산 가치를 극대화하는 데 기여한 것으로 나타났습니다. 성능 지표로는 모델 훈련 시 데이터 품질 기여도가 고려되며, 이들의 상호작용을 통해 FL의 효과성을 극대화했습니다. 이러한 접근 방식은 금융, 보험, 헬스케어 산업 등 여러 분야에서의 FL 적용에 있어 중요한 통찰력을 제공합니다.



### ZETA: Leveraging Z-order Curves for Efficient Top-k Attention (https://arxiv.org/abs/2501.14577)
Comments:
          25 pages, 4 figures, accepted in International Conference on Learning Representations (ICLR) 2025

- **What's New**: 최근 Transformer는 시퀀스 모델링 아키텍처의 기본 구성 요소로 자리 잡았습니다. 하지만 self-attention의 메모리 및 계산 비용이 시퀀스 길이 $N$에 따라 제곱으로 증가하여 긴 시퀀스의 처리에 제한이 있습니다. 본 연구에서는 ZETA라는 새로운 접근 방식을 제안하여, 과거 토큰을 효율적으로 병렬 쿼리할 수 있는 방법을 제시합니다. ZETA는 $k$개의 가장 관련 있는 토큰을 선택하되 훈련 효율성을 극대화합니다.

- **Technical Details**: ZETA는 	extbf{Z}-Order 곡선을 활용하여 효율적인 top-$k$ attention을 구현합니다. 핵심적으로 키(key)와 쿼리(query)의 차원을 줄여서 상대 거리 정보를 보존할 수 있도록 합니다. 이를 통해 저차원 공간에서 키와 쿼리를 일차원으로 매핑하며, 이로 인해 병렬 정렬이 가능해져 top-$k$ 토큰 선택의 효율성을 크게 향상시킵니다. 이론적으로도 차원 저하에 따른 Trade-off를 명확히 보여줍니다.

- **Performance Highlights**: 실험 결과, ZETA는 synthetic 	extsc{Multi-Query Associative Recall} 작업에서 표준 attention과 유사한 성능을 보여주며, 	extsc{Long Range Arena}와 	extsc{WikiText-103} 언어 모델링 작업에서는 기존의 attention 및 변형 모델보다 월등한 성능을 방출합니다. 이러한 성능의 향상은 ZETA의 병렬 처리 능력과 효율적인 토큰 검색 방법에 기인합니다.



### Fairness of Deep Ensembles: On the interplay between per-group task difficulty and under-representation (https://arxiv.org/abs/2501.14551)
Comments:
          12 pages, 6 figures

- **What's New**: 이번 논문에서는 동질적 앙상블 기법이 알고리즘의 편향을 완화하는 데 효과적임을 입증합니다. 특히 성별, 연령 또는 인종과 같은 보호 속성에 따라 정의된 하위 집단에 대해 성능 향상을 보여주었습니다. 저자들은 균형 잡힌 데이터셋이 오히려 불리한 그룹에 손해를 줄 수 있음을 강조했습니다.

- **Technical Details**: 연구에서는 두 개의 합성 시나리오를 구축하여 각 그룹의 하위 표현과 작업 난이도의 상호작용을 분석했습니다. 이 시나리오는 두 개의 인구 통계적 하위 그룹(남성과 여성)과 두 개의 목표 클래스(건강과 질병) 간의 성능 차이를 실험적으로 고려합니다. 저자들은 동질적 앙상블이 이러한 성능 격차를 줄일 수 있음을 보여주었습니다.

- **Performance Highlights**: 실험 결과, 동질적 앙상블을 사용했을 때 성능 향상폭이 1.3%에서 4%에 달하며, 그룹 간 격차를 무의미한 수준까지 줄일 수 있었습니다. 이러한 결과는 동질적 앙상블 기법이 알고리즘의 공정성을 높이는 데 기여할 수 있음을 시사합니다. 이는 의료와 같은 중요한 분야에서 더욱 더 중요한 성과가 될 수 있습니다.



### Distributed Conformal Prediction via Message Passing (https://arxiv.org/abs/2501.14544)
Comments:
          16 pages, 11 figures, submitted for posssible publication

- **What's New**: 이 논문에서는 안전이 중요한 분야에서 신뢰할 수 있는 추론을 보장하기 위한 전이 학습 모델의 사후 보정(post-hoc calibration) 문제를 다룹니다. 특히, Conformal Prediction (CP)을 기반으로 한 두 가지 분산 접근 방식인 Q-DCP(quantile-based distributed conformal prediction)와 H-DCP(histogram-based distributed conformal prediction)를 제안합니다. 이러한 접근 방식은 서로 다른 그래프 토폴로지에서 제한된 보정 데이터를 사용하여 효과적인 예측 세트를 생성할 수 있도록 합니다.

- **Technical Details**: Q-DCP는 분산 quantile 회귀(distributed quantile regression)에 맞춤형 평활화(smoothing) 및 정규화(regularization) 항을 통합하여 수렴(convergence)을 가속화합니다. H-DCP는 합의 기반(histogram estimation) 방법을 사용하여 다수의 장치에서 수집된 로컬 보정 데이터로부터 전 세계적으로 histogram을 추정합니다. 논문에서는 통신 오버헤드 및 하이퍼파라미터 조정 요구사항과 같은 여러 트레이드오프를 실험적으로 평가합니다.

- **Performance Highlights**: Q-DCP와 H-DCP는 각각 고유한 장단점을 가지며, H-DCP는 하이퍼파라미터에 의존하지 않고 안정적인 커버리지(coverage) 보장을 제공합니다. 그러나 H-DCP는 Q-DCP보다 더 많은 통신 부하를 요구합니다. 이 연구 결과는 분산 환경에서 신뢰할 수 있는 예측을 위한 새로운 방향과 가능성을 제시합니다.



### Reducing Action Space for Deep Reinforcement Learning via Causal Effect Estimation (https://arxiv.org/abs/2501.14543)
- **What's New**: 이 논문은 딥 강화 학습(DRL)에서 큰 동작 공간 내의 중복 행동을 해결하기 위한 새로운 접근 방식을 제시합니다. 기존 연구들은 중복 행동을 줄이기 위해 노력했지만, 이러한 방법들은 실질적인 정량적 증거를 제공하지 못했습니다. 본 연구에서는 행동의 인과적 영향을 평가하여 탐색 효율성을 개선하는 새로운 방법, 인과 효과 추정(causal effect estimation, CEE)을 소개합니다.

- **Technical Details**: 본 논문에서는 역동력 모델(inverse dynamics model)을 사전 훈련하여 환경에 대한 지식으로 사용하고, 전체 행동 공간에서 행동을 분류한 후 각 행동의 인과적 영향을 추정합니다. 이를 통해 탐색 중 불필요한 행동을 억제할 수 있습니다. 연구자들은 Kullback-Leibler (KL) 발산을 사용하여 다음 상태 분포에 대한 행동의 중복성을 정의하고, 이 방법을 통해 동작을 효과적으로 필터링합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 다양한 환경에서 중복 행동을 필터링하며 탐색 효율을 크게 높이는 것으로 확인되었습니다. 연구진은 이론적 분석과 함께 실험적 결과를 통해 제안된 방법의 유효성을 입증했습니다. 결과적으로, CEE와 행동 분류를 결합함으로써 전반적인 성능이 개선된 것으로 나타났습니다.



### On Hardening DNNs against Noisy Computations (https://arxiv.org/abs/2501.14531)
Comments:
          Presented at AccML workshop co-located HiPEAC 2025

- **What's New**: 본 논문에서는 심층 신경망(Deep Neural Networks, DNNs)의 안정성을 높이기 위한 양자화(quantization)와 소음 주입(noise injection) 기법을 탐구합니다. 특히, 양자화 인식 훈련(Quantization-Aware Training, QAT)을 통해 아날로그 컴퓨팅에서의 소음에 대한 강인성을 향상시킬 수 있음을 관찰했습니다. 두 가지 기법인 QAT와 소음 주입 훈련을 평가하여, DNN의 복잡한 아키텍처에서도 우수한 성능을 나타내는 것을 보여주었습니다.

- **Technical Details**: DNN의 훈련 및 평가에 필요한 강력한 프로세서(GPUs 등)의 필요성에도 불구하고, 점점 더 큰 DNN의 용량은 메모리 공간과 전력 소모 문제를 야기합니다. 아날로그 컴퓨팅과 같은 대안 기술을 도입함으로써 에너지 효율성을 개선할 수 있습니다. 그러나 아날로그 연산으론 본질적으로 소음이 동반되며, 이로 인해 높은 예측 정확도를 유지하기 어려운 문제점이 있습니다. 본 연구에서는 이러한 소음을 극복하기 위해 양자화와 소음 주입을 통한 훈련 기법을 사용했습니다.

- **Performance Highlights**: 양자화와 소음 주입 훈련 방법은 CIFAR-10 데이터셋을 통한 이미지 분류 작업에서 다양한 DNN 아키텍처에 대한 평가에서 강력한 성능을 발휘했습니다. 소음 주입 훈련이 복잡한 신경망 아키텍처에서 특히 뛰어난 강인성 향상을 제공하면서 두 기법 모두 소음에 대한 내성을 높임을 입증했습니다. 결과적으로, 양자화 인식 훈련(QAT) 방법이 아날로그 연산의 내재된 소음을 극복하는 데 효과적임을 보여주었습니다.



### Automated Assignment Grading with Large Language Models: Insights From a Bioinformatics Cours (https://arxiv.org/abs/2501.14499)
- **What's New**: 최근의 대규모 언어 모델(LLMs)의 발전은 이들 기술이 교육 분야와 관련된 다양한 적용 가능성을 높이고 있습니다. 본 연구에서는 LLM을 사용하여 대규모 학생의 서면 과제를 자동으로 채점하는 새로운 접근법을 제안합니다. LLM은 과제 피드백을 보다 빠르고 효율적으로 제공하면서도 교육 효과성을 향상시킬 수 있는 잠재력을 지니고 있습니다.

- **Technical Details**: 본 연구는 슬로베니아 리유블라나 대학교에서 진행된 생물정보학 입문 과정에서, 119명의 학생들이 5개의 과제를 통해 LLM을 활용하여 서면 응답을 평가받는 구조로 설계되었습니다. LLM은 8명의 채점자 중 하나로 무작위로 배정되어 학생의 서면 제출물을 채점했습니다. 이 과정에서 OpenAI의 ChatGPT와 Facebook의 Llama 3 모델을 포함한 다양한 LLM 아키텍처가 사용되어 그들의 성능과 피드백을 비교 분석했습니다.

- **Performance Highlights**: 연구 결과, 잘 설계된 프롬프트(prompts)와 함께 사용할 경우 LLM은 인간 채점자에 버금가는 채점 정확도와 피드백 품질을 달성할 수 있는 것으로 나타났습니다. 또한, 오픈 소스 LLM의 성능이 상용 LLM에 준하는 것으로 평가되어, 학교에서 데이터를 보호하며 독자적인 채점 시스템을 구현할 수 있는 가능성이 확인되었습니다. 이러한 결과는 대규모 학생 교육에서 LLM의 활용 가능성을 제시하는 중요한 사례로 여겨집니다.



### Pesti-Gen: Unleashing a Generative Molecule Approach for Toxicity Aware Pesticide Design (https://arxiv.org/abs/2501.14469)
Comments:
          9 pages, 2 figures, 5 tables

- **What's New**: 이 논문에서는 기존의 기계 학습 방법들이 농업에서 새로운 분자 구조를 생성하는 데 한계가 있음을 지적하고, 최초로 새로운 농약 후보를 생성할 수 있는 생성 모델인 Pesti-Gen을 제안합니다. 이 모델은 변량 자동 인코더(Variational Autoencoder, VAE)를 기반으로 하여 농약의 최적화된 특성을 가진 후보 분자를 생성하는 것을 목표로 합니다. Pesti-Gen은 초기의 일반 화학 구조 표현을 포착하는 사전 교육 및 특정 독성 정보를 반영하는 미세 조정 단계를 포함한 두 단계의 학습 과정을 통해 작동합니다.

- **Technical Details**: Pesti-Gen은 가축 독성(LD50) 및 수생 생태 독성(LC50)과 같은 다양한 독성 지표를 최적화하여 농업 환경에 적합한 친환경 농약 후보를 생성합니다. 이를 위해 연구진들은 WHO 독성 분류 기준에 기반하여 세심하게 선별된 독성 메트릭스를 사용하고, 다양한 농약 후보의 SMILES 표현과 독성 점수를 포함한 맞춤형 데이터 세트를 구축하였습니다. Pesti-Gen은 화학 유효성을 평가하고, 물리화학적 특성을 비교하며, SMILES 유효성 검사를 통해 모델의 신뢰성을 검증하였습니다.

- **Performance Highlights**: Pesti-Gen은 새로운 분자 구조를 생성하는 데 있어 약 68%의 화학 유효성을 달성하였습니다. 이 모델은 또한 합성 접근성과 독성 감소 간의 균형을 잘 유지하며 실질적인 응용 가능성을 강조합니다. 농약 설계의 생태독성 제약들을 통합함으로써, Pesti-Gen은 혁신적이고 환경 친화적인 농약 솔루션 개발의 초석을 다지고 있습니다.



### MLMC: Interactive multi-label multi-classifier evaluation without confusion matrices (https://arxiv.org/abs/2501.14460)
Comments:
          12 pages

- **What's New**: 본 논문에서는 다중 라벨(classifier) 분류기를 평가하기 위한 시각적 탐색 도구인 MLMC(Multi-Label Multi-Class) 를 소개합니다. MLMC는 전통적인 confusion matrix의 확장성을 능가하여 사용자가 인스턴스, 라벨 및 분류기 관점에서 성능을 평가할 수 있도록 도와줍니다. 사용자 연구 결과 MLMC는 기존의 평가 방식보다 사용자 친화적인 다중 라벨 분류기 평가를 제공하는 것으로 나타났습니다.

- **Technical Details**: MLMC는 세 가지 분류 문제 유형인 이진 분류(binary classification), 다중 클래스(multi-class), 다중 라벨(multi-label)으로 정리됩니다. 기존 접근법은 이진 분류 문제에 효과적이나, 다중 라벨 문제에서는 많은 클래스를 동시에 처리하기 어려웠습니다. 사용자 맞춤형 인터페이스와 직관적인 색상 사용을 통해 MLMC는 다중 라벨 분류기의 평가를 간소화하며, 다양한 데이터 타입(text, images, audio)에 적용할 수 있도록 설계되었습니다.

- **Performance Highlights**: MLMC의 유용성을 평가하기 위해 두 개의 사용자 연구가 수행되었습니다. 첫 번째 연구에서는 사용성을 평가하였고, 두 번째 연구에서는 전통적인 confusion matrix 접근법과의 성능 비교가 이루어졌습니다. 결과적으로, 사용자들은 MLMC를 사용했을 때 시간, 신뢰도 및 정확성의 향상을 경험하였으며, 이는 기존의 평가 도구와 비교하여 더 효과적인 성능 분석을 가능하게 합니다.



### A Survey of Optimization Methods for Training DL Models: Theoretical Perspective on Convergence and Generalization (https://arxiv.org/abs/2501.14458)
- **What's New**: 이 논문은 딥러닝(Deep Learning) 최적화 및 일반화의 이론적 기초에 대한 포괄적인 요약을 제공합니다. 다양한 최적화 방법론, 수렴(convergence) 분석 및 일반화(generalization) 능력에 대한 논의가 포함되어 있습니다. 또한, 딥러닝 손실 경관(loss landscape)의 특성에 적응하는 최적화 기법 분석도 다루고 있어, 이 분야의 연구자들에게 중요한 통찰을 제공합니다.

- **Technical Details**: 이 논문에서는 일반적으로 사용되는 기울기 기반(gradient-based) 1차 및 2차 최적화 방법의 이론적 분석을 포함합니다. 특히, 딥러닝 손실 경관에 적합하도록 조정된 최적화 기술을 다루어, 최적의 일반화 포인트를 발견하는 것을 장려합니다. 분산 최적화(distributed optimization) 방법론도 탐구하여 중앙 집중식(centralized) 및 분산(decentralized) 접근 방식을 포괄적으로 논의합니다.

- **Performance Highlights**: 조사한 최적화 알고리즘에 대해 볼록(convex) 및 비볼록(non-convex) 분석을 제공하며, 최신 ML 연구에서 간과된 이론적 분석을 보강합니다. 이 논문은 딥러닝 최적화 방법에 대한 포괄적인 이론 핸드북으로, 초보자와 숙련된 연구자 모두에게 유용한 자료로 구성되어 있습니다.



### Optimal Strategies for Federated Learning Maintaining Client Privacy (https://arxiv.org/abs/2501.14453)
- **What's New**: 본 논문은 Federated Learning (FL)에서 모델 성능과 통신 복잡성 간의 균형을 이론적으로 분석합니다. 특히, 하나의 로컬 에폭을 훈련하는 것이 최적의 성능을 달성할 수 있음을 증명했습니다. 또한, 클라이언트 수가 증가할 때 FL 모델의 유용성이 어떻게 변화하는지를 탐구하며, 이를 통해 개인정보 보호 예산이 동일할 때 더 많은 클라이언트가 참여하면 유용성이 향상된다고 주장합니다.

- **Technical Details**: 이 연구에서는 Differential Privacy (DP)를 보장하기 위해 DP-SGD를 도입하고, 클라이언트가 로컬 모델을 서버에 업데이트하는 빈도를 증가시킬 때 성능이 향상된다는 것을 공식적으로 증명합니다. 또한, 개인 정보 보호를 유지하면서 FL 성능을 향상시키기 위한 'PFL' 프레임워크를 제안합니다. 이 방법은 클라이언트 수가 많을수록 비공식 모델에 가까운 성능을 발휘한다는 것을 암시합니다.

- **Performance Highlights**: 실험 결과는 MNIST, FashionMNIST, CIFAR10 데이터셋을 사용하여 이론적 주장인 Theorem 2와 Theorem 3에 대한 유효성을 입증했습니다. 논문에서 제안하는 방법은 해당 데이터셋에서 만족스러운 정확도를 달성하면서도 개인 정보 보호를 고려한 훈련 방법입니다. 따라서, 이 연구는 개인 정보 보호를 유지하면서 FL 시스템의 성능을 극대화할 수 있는 길잡이가 됩니다.



### Impact of Batch Normalization on Convolutional Network Representations (https://arxiv.org/abs/2501.14441)
- **What's New**: 이번 연구에서는 일반적으로 사용되는 설계 기법인 배치 정규화(Batch Normalization, BatchNorm)가 심층 신경망의 숨겨진 표현(hidden representations)에 미치는 영향을 분석합니다. 연구자들은 CNN의 활성화 값의 희소성(sparsity) 및 클러스터링(clustering) 특성을 주의 깊게 살펴보았으며, 배치 정규화의 적용 여부에 따른 비교 결과를 통해 일반화(generalization) 성능을 이해하는 데 중요한 통찰을 제공합니다. 이러한 비교 분석은 BatchNorm이 숨겨진 표현의 특성에 어떤 영향을 미치는지를 드러내는 데 목적을 두고 있습니다.

- **Technical Details**: 배치 정규화는 심층 신경망(DNN) 훈련 시 활성화 값을 정규화하여 훈련 속도와 성능을 향상시키는 데 사용됩니다. 발표된 방법론에서는 CNN의 활성화 값이 4D 텐서로 표현되며, 배치의 각 채널에서 평균(mean) 및 표준편차(standard deviation)를 계산하여 정규화합니다. 이 연구에서는 DNN의 숨겨진 표현의 희소성 및 클러스터링 특성을 분석하여, 이러한 특성이 모델의 일반화 능력에 미치는 영향을 조사하고 있습니다.

- **Performance Highlights**: 배치 정규화를 통해 훈련된 모델은 일반적으로 더 빠른 수렴(convergence)과 더 나은 일반화 능력을 보여줍니다. 연구 결과, 배치 정규화의 영향이 대표적인 희소성에 크게 영향을 미치지 않지만, 클러스터링 측면에서는 보다 유리한 특성을 보임을 명확히 하였습니다. 이로 인해 모델의 성능 향상은 배치 정규화에 기인한 clustering 특성에서 비롯되는 것으로 해석됩니다.



### Convergence of gradient based training for linear Graph Neural Networks (https://arxiv.org/abs/2501.14440)
Comments:
          27 pages, 8 figures

- **What's New**: 이번 논문에서는 선형 그래프 신경망(linear Graph Neural Networks, GNNs)의 학습 과정에서 기울기 동역학의 수렴성을 분석합니다. 특히, 평균 제곱 오차(mean squared loss)를 사용하는 기울기 흐름 훈련이 전역 최소값(global minimum)으로 지수적으로 수렴하는 것을 증명하였습니다. 이러한 수렴 속도는 초기 가중치(initial weights) 및 그래프 이동 연산자(graph shift operator)에 따라 달라지며, 이를 유명한 그래프 모델과 실제 데이터셋에서 검증하였습니다.

- **Technical Details**: 그래프는 노드 집합 V와 엣지 집합 E로 구성된 순서쌍 G=(V,E)로 정의됩니다. 이 논문에서는 선형 GNN 모델을 구축하고, 그와 관련된 평균 제곱 손실을 기반으로 한 기초 매트릭스 이론(preliminary results from matrix theory)을 회고합니다. 후속 섹션에서는 기울기 흐름 훈련의 수렴성을 분석하고, 손실 표면에서 전역 최소값의 가중치 최적화에 대해 논의합니다.

- **Performance Highlights**: 선형 GNNs는 이 논문에서 확인된 바와 같이 평균 제곱 손실에서 기울기 흐름 훈련을 통해 전역 최소값으로 지수적으로 수렴합니다. 또한, 초기 가중치를 긍정적인 특이값을 가진 경우, 기울기 하강법(gradient descent) 훈련이 전역 최소값에 수렴하는 방법을 설명합니다. 이 연구는 그래프 구조와 관련된 데이터에 대한 깊은 이해를 제공하며, 실제 데이터셋과 합성 데이터셋에서 이 그레이디언트 동역학을 검증하였습니다.



### GraphBC: Improving LLMs for Better Graph Data Processing (https://arxiv.org/abs/2501.14427)
- **What's New**: 이번 연구에서는 GraphBC라는 새로운 모델 프레임워크를 소개합니다. GraphBC는 그래프의 올바른 직렬화 순서를 보장하기 위한 Order Selector Module과 더 나은 구조의 서브그래프를 샘플링하기 위한 Subgraph Sampling Module을 특징으로 합니다. 이는 LLM의 추론 및 제로샷 학습 능력을 개선하고자 하는 목적을 가지고 있습니다.

- **Technical Details**: GraphBC는 LLM을 예측기로 활용하며, 그래프 데이터를 텍스트로 변환하여 입력하는 방식으로 작동합니다. 그래프는 노드 집합 
 𝒱와 엣지 집합  ℰ를 포함하는 구조로 표현되며, 각 노드의 특징은 피쳐 매트릭스 𝐗에 나타납니다. 연구는 강화학습 개념을 사용하여 서브그래프 샘플링 모듈을 훈련하며, 이 과정을 통해 그래프 데이터를 향상시키기 위한 방법들도 제안합니다.

- **Performance Highlights**: 여러 데이터셋을 대상으로 한 실험 결과, GraphBC는 노드 분류 및 그래프 질문-응답 작업에서 LLM의 성능을 개선하며 일반화 능력을 향상시키는 것으로 나타났습니다. 성능 테스트에서 서로 다른 아키텍처와 규모를 가진 LLM을 비교했으며, 다양한 노드 및 엣지의 순서에 따라 성능 변동을 분석하였습니다. 결과적으로, GraphBC는 감독 및 제로샷 그래프 학습 환경에서 뛰어난 성능을 보여주었습니다.



### CENTS: Generating synthetic electricity consumption time series for rare and unseen scenarios (https://arxiv.org/abs/2501.14426)
- **What's New**: 최근 대규모 생성 모델의 발전으로 자연어 처리, 컴퓨터 비전, 단백질 구조 예측 등 다양한 분야에서 foundation models의 잠재력이 입증되었습니다. 하지만 에너지 및 스마트 그리드 분야에서는 고품질 데이터의 부족과 이질성으로 인해 그 적용이 제한적입니다. 본 연구에서는 드문 컨텍스트 변수에 대한 고충실도 전력 소비 시계열 데이터 생성을 위한 방법(CENTS)을 제안하며, 이는 데이터 부족 문제를 해결하기 위해 설계되었습니다.

- **Technical Details**: CENTS는 세 가지 핵심 혁신을 포함합니다. 첫째, 관찰되지 않은 컨텍스트 변수의 역변환을 가능하게 하는 컨텍스트 정규화 접근법을 통해 시계열 데이터의 품질을 높입니다. 둘째, 컨텍스트 인코더를 통해 임의의 수와 조합의 컨텍스트 변수에 조건화된 최신 시계열 생성기를 지원합니다. 셋째, 컨텍스트 재구성 손실(auxiliary context classification loss)을 사용하여 모델 성능을 높이는 데 도움을 줍니다.

- **Performance Highlights**: CENTS는 가정 수준 전력 소비 데이터를 현실적으로 생성할 수 있는 효율성을 강조합니다. 이 방법을 통해 일반적인 시나리오뿐만 아니라 드문 시나리오와 신규 조합의 전력 소비 시계열을 생성할 수 있어 실제 데이터와 합성 데이터를 기반으로 한 더 큰 foundation models 훈련의 길을 열어줍니다. 또한, 다양한 평가 지표를 통해 생성된 시계열 모델의 효율성을 종합적으로 검토하였습니다.



### SoK: What Makes Private Learning Unfair? (https://arxiv.org/abs/2501.14414)
Comments:
          Systemization of Knowledge (SoK) paper. This work has been accepted for publication in the 3rd IEEE Conference on Secure and Trustworthy Machine Learning (SaTML'25). The final version will be available on IEEE Xplore

- **What's New**: 이 논문은 Differential Privacy(차등 프라이버시) 기술이 머신 러닝에서 공정성과 성능 차이를 어떻게 심화시키는지를 연구한 최초의 포괄적인 리뷰를 제공합니다. 기존의 다양한 연구들은 차등 프라이버시 기법들과 공정성 개념을 정의하는 방식에서 차이를 보였고, 이는 연구 결과에 혼란을 초래했습니다. 이번 연구는 이러한 요인들을 머신 러닝 파이프라인 내의 위치에 따라 분류하여, 그것들의 상호작용과 완화 전략의 가능성을 분석합니다.

- **Technical Details**: 연구는 감독 학습(supervised learning)에서의 분류에 중점을 두는 방식으로 진행되며, 학습 데이터 세트와 기초 분포가 불균형할 때 나타나는 불균형한 영향(disparate impact)에 대한 여러 요인을 분석합니다. Differentially Private Stochastic Gradient Descent(DP-SGD) 기술이 이러한 불균형의 주요 원인으로 지목되었으며, 그에 따른 완화 전략을 제안하고 있습니다. 이 논문의 중요한 기여 중 하나는 차등 프라이버시로 인한 불균형 심화에 기여하는 요인을 체계적으로 정리한 것입니다.

- **Performance Highlights**: 연구에서는 불균형한 훈련 데이터셋과 의사 결정 경계의 거리 차이가 불균형 심화의 주요 조건임을 발견했습니다. 차등 프라이버시 기술의 사용이 특히 고용, 교육 및 대출과 같은 분야에서 결정 편향을 강화할 수 있다는 점에서 큰 우려가 제기됩니다. 이 논문은 차등 프라이버시가 머신 러닝 분야에 미치는 영향과 그에 대한 연구의 방향성을 제시하여, 향후 연구자들에게 유용한 기본 자료를 제공합니다.



### Reinforcement Learning for Efficient Returns Managemen (https://arxiv.org/abs/2501.14394)
- **What's New**: 본 논문에서는 리테일 창고에서 반품 상품의 효율적인 처리를 위해 새로운 온라인 재할당 접근법을 제안합니다. 기존의 오프라인 접근 방식과 비교할 때, 제안된 방식은 반품 상품의 평균 저장 시간을 96%나 줄이면서도 성능 격차는 3%에 불과한 것으로 나타났습니다. 새로운 강화 학습( reinforcement learning ) 알고리즘인 PostAlloc은 제품의 도착 즉시 재할당 결정을 신속하게 내릴 수 있도록 설계되었습니다.

- **Technical Details**: 여기서 다루는 모델은 전통적인 다중 배낭 문제( multiple knapsack problem )를 확장된 형식으로 표현합니다. 각 매장( store )의 용량(C)와 반품 상품의 가치(v) 및 무게(w) 정보를 통해 최적의 재배치를 목표로 합니다. PostAlloc 알고리즘은 도착한 상품이 임시 저장 저장소에 보관되도록 하고, 최종 순서의 끝에서 결정을 연기할 수 있는 기능을 추가하여 의사결정의 질을 높이는 데 중점을 둡니다.

- **Performance Highlights**: 제안된 접근법의 실험 결과는 기존 오프라인 수학적 해결책과 비교하여 양호한 성능을 보였습니다. 특히, 오프라인 접근법에 비해 저장 시간 측면에서 96%의 큰 개선을 이룬 것으로 평가되었습니다. 이는 리테일 운영의 비용 절감과 더불어 보다 지속 가능한 공급망 관리를 가능하게 합니다.



### Distinguishing Parkinson's Patients Using Voice-Based Feature Extraction and Classification (https://arxiv.org/abs/2501.14390)
Comments:
          Presented at the 13th International Marmara Science Congress (IMASCON 2024)

- **What's New**: 이 연구는 파킨슨병(Parkinson's disease, PD) 환자와 건강한 대조군 간의 음성 특징을 분석하여 차별화하는 데 초점을 맞추고 있습니다. 특히 약물 요법의 영향을 살펴보기 위해 두 개의 그룹으로 나뉘어진 환자 데이터셋이 사용되었습니다. 이 연구는 비침습적인 음성 분석이 PD의 조기 발견 및 모니터링에 어떻게 기여할 수 있는지를 보여줍니다.

- **Technical Details**: 연구팀은 H1N Zoom 마이크를 사용하여 Fırat 대학 신경과에서 설정된 환경에서 환자들이 미리 정해진 텍스트를 읽는 음성 기록을 수집했습니다. 이 과정에서 19개의 핵심 특징이 추출되었으며, 이는 jitter, luminance, zero-crossing rate (ZCR), root mean square (RMS) energy, entropy, skewness 등을 포함합니다. MATLAB의 Classification Learner 툴박스를 이용해 여러 머신러닝(classification algorithm) 모델이 적용되어 그룹 분류가 이루어졌습니다.

- **Performance Highlights**: 3층 인공신경망(artificial neural network) 아키텍처의 분류 정확도는 전통적인 머신러닝 알고리즘과 비교되었습니다. 연구 결과는 비침습적인 음성 분석과 머신러닝이 PD 환자의 조기 진단과 모니터링에서 큰 잠재력을 가진다는 것을 강조합니다. 향후 연구는 특징 선택(feature selection)의 최적화와 더 발전된 분류 기술을 탐구함으로써 진단 정확성을 더욱 향상시킬 수 있습니다.



### Fat-to-Thin Policy Optimization: Offline RL with Sparse Policies (https://arxiv.org/abs/2501.14373)
Comments:
          accepted by ICLR 2025; code available at this https URL

- **What's New**: 이 논문은 안전성 중심의 희소 정책(sparse policies)을 학습하기 위해 오프라인 강화 학습(offline reinforcement learning)과 결합된 새로운 패러다임을 제안합니다. 제안된 방법론인 Fat-to-Thin Policy Optimization (FtTPO)은 기존의 알고리즘들이 가진 한계를 극복하고, 안전-critical한 환경에서 효율적으로 작동할 수 있는 방법을 제공합니다. 기존의 후버지에 대한 문제를 해결하며, 데이터셋에서 효과적으로 학습된 두 가지 정책을 유지합니다.

- **Technical Details**: 이 연구는 할인 마르코프 결정 과정(discounted Markov Decision Processes)의 기초 위에서 희소 정책의 학습에 필요한 수학적 접근 방식을 제시합니다. 특히, q-Gaussian 분포의 개념을 도입하여 주요 작업을 처리하며, 제안된 알고리즘은 다양한 상황에서 동작할 수 있도록 설계되었습니다. 상태-행동 가치 함수(state-action value function)와 정책의 기대치를 정의하고, 기대 보상(expected return)을 극대화하는 것이 핵심 목표입니다.

- **Performance Highlights**: 제안된 FtTPO 알고리즘은 안전-critical한 치료 시뮬레이션과 표준 MuJoCo(Multi-Joint dynamics with Contact) 환경에서 유리한 성과를 나타냈습니다. 이를 통해 제안하는 방법의 효과성과 안정성을 입증하였으며, 실제 응용 프로그램에 대한 뛰어난 적응 능력을 보여주었습니다. 또한 코드도 공개되어 있어, 후속 연구자들이 쉽게 접근하고 활용할 수 있습니다.



### Facies Classification with Copula Entropy (https://arxiv.org/abs/2501.14351)
Comments:
          12 pages, 5 figures, 3 tables. arXiv admin note: text overlap with arXiv:2310.16633

- **What's New**: 본 논문은 facies 분류에 copula entropy (CE)를 적용하는 방법을 제안합니다. CE를 이용해 지질 변수와 facies 클래스 간의 상관관계를 측정하고, 큰 음의 CE 값과 연관된 변수를 선택함으로써 더 적은 수의 변수를 이용해도 분류 성능을 유지할 수 있음을 검증했습니다. 이러한 변수들은 지질학적 의미를 담고 있어 지질학자에게 해석 가능한 결과를 제공합니다.

- **Technical Details**: copula 이론은 multivariate 의존성을 copula 함수로 표현하는 것에 관한 이론으로, Sklar's 정리에 따라 다변량 확률 밀도 함수는 주변 밀도와 copula 밀도 함수의 곱으로 표현될 수 있습니다. 본 논문에서는 copula를 이용한 통계적 독립성 측정 방법을 정의하며, CE를 통해 다양한 데이터 세트에서 효과적으로 변수를 선택할 수 있는 방법을 제안하고 있습니다. CE는 비모수적 방법으로 추정할 수 있어 널리 활용될 수 있습니다.

- **Performance Highlights**: 본 연구는 Kansas의 Council Grove 가스 저장소에서 얻은 데이터를 사용하여 실험을 진행하였습니다. 3232개의 샘플에서 7개의 변수를 사용했으며, GE를 통해 선택된 변수들이 RF(Random Forest) 분류기를 훈련하는 데 사용되었습니다. 실험 결과는 기존의 블랙박스 방법보다 해석 가능하고 더 적은 지질 변수를 사용하여도 성능을 희생하지 않음을 보여주었습니다.



### Online Inverse Linear Optimization: Improved Regret Bound, Robustness to Suboptimality, and Toward Tight Regret Analysis (https://arxiv.org/abs/2501.14349)
- **What's New**: 이번 연구에서는 온라인 학습 문제를 다루며, 시간에 따라 변화하는 가능한 행동 집합과 최적 행동을 관찰하는 학습자의 접근 방식을 제시합니다. 기존의 연구에서는 $O(n^4\ln T)$의 후회 경계(regret bound)를 달성했지만, 본 논문에서는 이를 $O(n\ln T)$로 개선하여 $n^3$의 비율로 향상시켰습니다. 더불어, 학습자가 가능성이 낮은 행동을 고려할 때, 후회 경계를 $O(n\ln T+\sqrt{\Delta_T n\ln T})$로 설정하였습니다.

- **Technical Details**: 본 논문은 온라인 뉴턴 단계(online Newton step, ONS) 기법을 적절한 exp-concave 손실 함수에 적용하여 후회 경계를 달성합니다. MetaGrad를 사용하여 $\Theta(\ln T)$ 개의 서로 다른 학습률을 병렬로 운용함으로써, 학습자가 최적 행동이 아닌 경우에도 효과적으로 후회 경계를 설정합니다. 또한, 저자들은 $O(n\ln T)$ 경계가 $O(\ln T)$ 요인까지 긴밀하게 밀접하다는 것을 보여주는 하한(a lower bound) 사례도 제공합니다.

- **Performance Highlights**: 연구 결과, 본 방법은 공간 차원(dimension) $n=2$의 특수한 경우에 대해 $O(1)$ 후회 경계를 달성할 수 있음을 보여주었습니다. 일반적인 경우와 차원 확장에 대한 도전 과제도 규명되었습니다. 본 논문은 온라인 학습 분야에서 효율적인 알고리즘 개발을 위한 중요한 기초가 될 것입니다.



### HorNets: Learning from Discrete and Continuous Signals with Routing Neural Networks (https://arxiv.org/abs/2501.14346)
Comments:
          Accepted to the ACML conference journal track with the Machine Learning journal. The first and the last authors share an equal contribution

- **What's New**: 본 논문에서는 HorNets (Horn Networks)라는 신경망 아키텍처를 제안합니다. HorNets는 연속적이고 불연속적인 표형 데이터에서 효율적으로 학습할 수 있도록 설계되었으며, 이는 적은 수의 데이터로도 높은 성능을 발휘합니다. 이 아키텍처는 클리핑된 다항식 유사 활성화 함수를 기반으로 하며, 입력의 카드inality에 따라 최적화할 신경망의 부분을 결정하는 사용자 지정된 라우팅 메커니즘이 포함되어 있습니다. HorNets는 특히 생물 의학적 고차원 데이터 세트에서 최첨단 분류 성능을 달성합니다.

- **Technical Details**: HorNets는 폴리클립 활성화 함수(polyClip activation function)와 사용자 정의 라우팅 메커니즘을 사용하여 입력 데이터의 특성 조합 공간을 명시적으로 모델링합니다. 이러한 접근 방식은 기계 학습 모델이 특정 데이터 조각에 대해 가장 적합한 작동 방식을 결정할 수 있도록 합니다. HorNets는 논리 게이트를 모델링하는 능력을 평가하기 위해 광범위한 벤치마크를 실시했으며, 일반적인 논리 게이트를 모델링하는 데 뛰어난 재현율(recall)을 보여주었습니다. 이는 고차원 생물 의학 데이터 세트에서 경쟁력 있는 성능을 입증하였습니다.

- **Performance Highlights**: HorNets는 14개의 실제 생물 의학적 데이터 세트에서 최고의 성능을 달성했으며, 이는 그래디언트 부스팅 트리(gradient-boosted trees) 및 AutoML 기반 분류기와 비교하여 우수합니다. 연구 결과, HorNets는 노이즈가 포함된 XNOR와 같은 논리 절을 안정적으로 검색할 수 있는 몇 안 되는 접근 방식 중 하나입니다. 이 아키텍처는 10개 이상의 실제 데이터 세트에서 경이로운 성능을 보이며, 기존 방법들보다 효과적인 솔루션으로 자리 잡았습니다.



### Relative Layer-Wise Relevance Propagation: a more Robust Neural Networks eXplaination (https://arxiv.org/abs/2501.14322)
Comments:
          arXiv admin note: text overlap with arXiv:2012.14501, arXiv:1605.01713 by other authors

- **What's New**: 이번 연구에서는 새로운 정의인 Relative LRP(R-LRP)를 소개하여, 기존의 Layer-Wise Relevance Propagation(LRP) 방법의 한계점을 해결하고자 했습니다. 특히, 작은 값으로 나누는 문제를 최소화하여 다층 신경망의 출력에 기여하는 정도를 보다 정확하게 시각화할 수 있게 되었습니다. 이러한 접근 방식은 이미지 분류 작업에 특히 유용하며, 픽셀 단위로 기여도를 분석함으로써 예측에 영향을 미치는 중요 영역을 찾는데 도움을 줍니다.

- **Technical Details**: R-LRP는 기존의 LRP 방법과는 달리 하이퍼파라미터 튜닝이 필요하지 않아 사용이 용이합니다. 이 방법은 일반적인 CNN, VGG16, VGG19, 및 Resnet50 네트워크 등 다양한 네트워크에서 적용 가능하며, 다층 신경망의 입력 도메인에 대한 출력 기여도를 정량화합니다. 연구에서는 특히 Resnet의 스킵 연결에 대해서만 작은 값으로 나누는 방식이 남아있음을 강조하였습니다.

- **Performance Highlights**: R-LRP 방법은 다양한 데이터셋에서 간단한 CNN 아키텍처와 VGG, Resnet 네트워크와 비교하여 효과적임을 보여주었습니다. 이 방법은 모델의 결정 과정을 보다 투명하게 하여 사용자가 신뢰할 수 있는 예측을 제공하는 데 기여합니다. 픽셀 기여도의 시각화는 예측의 중요 부분에 대한 깊이 있는 분석을 가능하게 합니다.



### Domain Expansion: Parameter-Efficient Modules as Building Blocks for Composite Domains (https://arxiv.org/abs/2501.14321)
Comments:
          6 pages, 3 figures, 2 tables

- **What's New**: 본 연구에서는 Parameter-Efficient Fine-Tuning (PEFT) 방법을 활용하여 개인화된 MBTI 성격을 표현하는 새로운 기법을 제안합니다. PEFT를 통해 학습된 Parameter-Efficient Modules (PEM)을 조합하여 16가지 MBTI 성격을 생성하는 방식을 설명하고 있으며, 이는 추가적인 파인튜닝 없이도 가능하다는 점이 특징입니다. 연구는 언어 모델 학습을 기반으로 하여, 다양한 도메인에 대한 일반화를 목표로 합니다.

- **Technical Details**: PEFT는 사전 학습된 언어 모델 (PLM)에서 대부분의 파라미터를 고정하고, 매우 적은 부분만을 세밀하게 조정하여 파라미터 효율적인 모듈인 PEM을 생성하는 방법입니다. 본 연구에서는 LoRA와 IA3 같은 최신 PEFT 방식을 사용하여 8개의 개별 특성 PEM을 학습한 후, 이들을 단순 조합 함수를 통해 최종 16가지 MBTI 성격 PEM으로 통합합니다. 이 과정에서 추가적인 파인튜닝 없이 개별 PEM의 가중치 공간에서 작동하는 조합 기능이 사용됩니다.

- **Performance Highlights**: 16가지 MBTI 성격을 생성하기 위한 온라인 성격 테스트를 통해 PEM의 효과를 검증하였습니다. 연구 결과, 제안된 방법이 개인 특성 PEM 및 조합된 성격 PEM 모두에서 뛰어난 성능을 보임을 확인하였으며, 이는 PEFT의 효율성을 잘 나타냅니다. 해당 방법은 크기가 작은 모델을 통해 메모리 사용량을 줄이면서도 강력한 성능을 나타내는 장점이 있습니다.



### Graph Feedback Bandits on Similar Arms: With and Without Graph Structures (https://arxiv.org/abs/2501.14314)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2405.11171

- **What's New**: 이 논문에서는 그래프 피드백을 가진 확률적 다중 무장 밴디트 문제를 연구합니다. 특히, 임상 시험 및 추천 시스템의 응용프로그램을 모토로 하여, 두 개의 팔(arms)이 유사할 경우 연결된다고 가정합니다. 새로운 피드백 구조 하에서 유감(lower bound)와 함께 문제 독립적인 Upper Confidence Bound(UCB) 알고리즘인 Double-UCB 및 문제 의존적인 Conservative-UCB 알고리즘을 소개합니다.

- **Technical Details**: 이 연구는 두 가지 밴디트 모델을 고려합니다: 표준 그래프 피드백 밴디트 모델과 팔의 수가 증가하는 문제입니다. 특히, 임상 시험에서 각 팔은 서로 다른 치료 계획을 나타내며, 선택한 계획의 보상과 유사한 다른 계획의 효과를 동시에 확인할 수 있습니다. 이러한 새로운 피드백 모델 하에서 다양한 알고리즘의 설계와 분석이 이루어지며, 그래프 구조 정보 없이도 작동할 수 있는 알고리즘도 제안됩니다.

- **Performance Highlights**: 실험 결과를 통해 이론적 결과를 검증하였으며, 제안된 알고리즘은 더 실용적인 환경에서도 효과적으로 작동할 수 있음을 보여줍니다. 연구에서 얻은 결과들은 무작위 기준에 따라 두 알고리즘의 상한과 하한을 설정하였고, 특히 Conservative-UCB는 문제에 따라 선형적이지 않은 유감 값을 달성할 수 있는 가능성을 보여줍니다.



### An Efficient Real Time DDoS Detection Model Using Machine Learning Algorithms (https://arxiv.org/abs/2501.14311)
Comments:
          7 pages, 14 figures

- **What's New**: 이 연구는 인터넷 서비스에 대한 의존도가 증가함에 따라 DDoS (Distributed Denial of Service) 공격 방지를 위한 효율적인 실시간 탐지 시스템을 개발하였습니다. 기존의 다양한 탐지 기술이 존재하지만, 시간 효율성과 정확도 간의 트레이드오프 때문에 효과적인 방법을 선택하기가 어렵습니다. 연구에서는 UNB CICDDoS2019 데이터셋을 활용하여 다양한 ML (Machine Learning) 분류기를 사용해 DDoS와 비-DDoS 트래픽을 분류합니다.

- **Technical Details**: 주요 기술적 접근 방식은 데이터 전처리와 다양한 기계 학습 알고리즘 사용입니다. 데이터 정제(data cleaning), 표준화(standardization) 및 주성분 분석(Principal Component Analysis, PCA) 기법을 사용하여 데이터셋을 세밀하게 준비하였습니다. 연구에서는 로지스틱 회귀(Logistic Regression), KNN (K-Nearest Neighbors), 랜덤 포레스트(Random Forest), 서포트 벡터 머신(Support Vector Machine), 나이브 베이즈(Naive Bayes)와 같은 여러 분류기를 통해 DDoS 탐지를 수행하였습니다.

- **Performance Highlights**: 연구 결과, 랜덤 포레스트, 아다부스트(AdaBoost), XGBoost가 정확도와 효율성 면에서 다른 알고리즘을 능가하여 실시간 응용 프로그램에 적합한 모델로 확인되었습니다. 또한, 각 분류기의 정밀도(precision), 재현율(recall), F1 점수(F1-score), 그리고 시간 복잡성을 평가하여 신뢰할 수 있는 DDoS 탐지 시스템을 구현하는 데 중점을 두었습니다.



### Permutation-based multi-objective evolutionary feature selection for high-dimensional data (https://arxiv.org/abs/2501.14310)
- **What's New**: 이 논문에서는 고차원 데이터의 특성 선택(feature selection) 방법을 새롭게 제안합니다. 기존의 permutation feature importance (PFI) 방법을 확장하여 개별 특성 대신 속성 하위 집합을 평가하는 방식입니다. 이 새로운 접근법은 모델 성능의 상호작용을 더 효과적으로 포착하며, 복잡한 특성 상호작용을 고려한 선택을 가능하게 합니다.

- **Technical Details**: 제안된 방법은 다목적 진화 알고리즘(multi-objective evolutionary algorithm)을 사용하여 후보 특성 하위 집합을 탐색합니다. 주요 목표는 선택된 특성을 셔플(shuffle)할 때 모델 성능의 감소를 극대화하고, 특성 하위 집합의 크기를 최소화하는 것입니다. PSEFS-MOEA(Permutation-based Subset Evaluation Feature Selection with MOEA)는 이러한 방식을 적용하여 하이퍼파라미터의 영향을 최소화하며, 더욱 정확하고 전체적인 특성 선택을 가능하게 합니다.

- **Performance Highlights**: PSEFS-MOEA는 24개의 고차원 데이터셋에서 실험을 통해 기존의 다른 9가지 유명한 특성 선택 방법보다 우수한 성능을 보여주었습니다. 이 방법은 모델의 성능 지표 측면에서도 꾸준히 높은 평가를 받았으며, 과적합(overfitting) 없이 강력한 일반화 능력을 지니고 있습니다. 논문에서는 이 방법이 고차원 데이터의 복잡한 특성 상호작용을 효과적으로 처리할 수 있음을 입증하고 있습니다.



### Advances in Temporal Point Processes: Bayesian, Deep, and LLM Approaches (https://arxiv.org/abs/2501.14291)
- **What's New**: 이 논문은 최근의 Temporal Point Processes (TPPs) 연구를 포괄적으로 리뷰합니다. 특히 Bayesian 방법론, 딥러닝(deep learning), 그리고 대형 언어 모델(LLM) 접근 방식에 중점을 두고 있습니다. 기존의 TPP 모델들과의 차별성을 두기 위해, 특히 Bayesian 비모수(nonparametric) TPP에 대한 최근 동향을 강조하며, LLM 기반의 TPP 연구가 recent years에 크게 주목받고 있음을 보여줍니다.

- **Technical Details**: TPPs는 사건 발생을 시간 창 [0,T] 내에서 모델링하는 확률적 과정입니다. 이들은 비동기 사건들로 구성되어 있으며 서로 영향을 미치고 복잡한 동적 특성을 보입니다. 모델 설계에서는 사건 간 시간 간격의 분포를 명시하는 다양한 접근 방식이 존재하며, 과거의 사건 이력을 기반으로 한 조건부 밀도 함수(conditional density function)가 주요하게 사용됩니다.

- **Performance Highlights**: 본 논문은 TPP 모델에서 기존 통계적 방법과 현재 머신러닝의 발전을 융합한 신경망 기반 방식의 성과를 강조합니다. 특히 통계적 비모수 TPP와 관련된 최근의 발전을 새롭게 조명하고, 2020년 이후 신경 TPP의 발전을 다룹니다. 또한 LLM의 맥락 이해를 활용하여 이벤트 시퀀스 모델링의 새로운 가능성을 제시하며, 앞으로의 연구 방향과 과제를 논의합니다.



### Active Learning for Continual Learning: Keeping the Past Alive in the Presen (https://arxiv.org/abs/2501.14278)
- **What's New**: 논문에서는 AccuACL(Accumulated informativeness-based Active Continual Learning)이라는 새로운 방법을 제안합니다. 이는 Fisher 정보 행렬을 샘플 선택 기준으로 사용하여 과거 지식을 보존하면서 새로운 작업을 신속하게 배울 수 있도록 돕습니다. 이 접근 방식은 기존의 Active Learning(AL) 전략과는 달리 기계 학습 모델의 성능을 극대화하면서 레이블 비용을 줄이는 데 초점을 맞추고 있습니다. 따라서 ACL(Active Continual Learning) 시나리오에서 복잡하고 연속적인 데이터 분포에 효과적으로 대응할 수 있는 가능성을 제시합니다.

- **Technical Details**: AccuACL은 Fisher 정보 행렬을 통해 누적 정보를 모델링하여, 과거 특정 작업과 새로운 작업 간의 정보를 균형 있게 평가합니다. 이 알고리즘은 레이블이 붙지 않은 데이터의 풀에서 최적의 예제를 선택하는 조합 최적화 문제로 모델링되었습니다. 논문의 핵심 기술은 새로운 작업에서 과거 지식을 보존하는 데 기여할 수 있는 예제를 식별하는 데 중점을 두며, 이를 통해 악성 망각(catastrophic forgetting)을 방지하는 것을 목표로 합니다.

- **Performance Highlights**: AccuACL은 SplitCIFAR10, SplitCIFAR100, SplitTinyImageNet의 세 가지 CL 벤치마크에서 기존 AL 기준을 23.8% 및 잊어버림 측면에서 17.0% 향상시켰습니다. 실험 결과, AccuACL은 과거 작업과 관련된 예제에 대한 높은 성과를 보여주어 전체적인 성능에서 우위를 점하였습니다. 이러한 결과는 AccuACL이 과거 지식과 새로운 작업 학습 사이의 균형을 잘 유지할 수 있음을 시사합니다.



### TLXML: Task-Level Explanation of Meta-Learning via Influence Functions (https://arxiv.org/abs/2501.14271)
Comments:
          22 pages

- **What's New**: 이 논문에서는 메타-러닝에서의 영향 함수(influence functions)를 통해 과거 학습 작업이 모델의 인퍼런스에 미치는 영향을 설명하는 새로운 방법, Task-Level eXplanation of Meta-Learning (TLXML)을 제안합니다. 기존의 메타-러닝 기법은 안전한 적응에 대한 고려가 부족하여 부적절한 모델 업데이트를 초래할 수 있는 위험이 존재했습니다. TLXML은 이러한 문제를 해결하기 위해 훈련 샘플의 영향을 정량화하고 이를 사용자에게 일관된 방식으로 설명합니다.

- **Technical Details**: TLXML은 과거 학습 작업의 영향을 정량화하기 위해 영향 함수를 사용하며, 메타-파라미터와 적응 네트워크 가중치, 인퍼런스를 분석합니다. Hessian 행렬의 근사를 위한 Gauss-Newton 행렬을 도입하여 계산 비용을 현저히 줄이며, TLXML의 계산 비용을 O(pq)로 최적화합니다. 이 방법은 메타-러닝 현장에서의 모델 설명 가능성을 높이기 위해 설계되었습니다.

- **Performance Highlights**: 실험 결과 TLXML이 이미지 분류 작업에서 MAML 및 Prototypical Network를 사용하여 작업 구분과 분포 구분의 적합성을 입증하였습니다. 사용자 수준에 맞춰, 보다 간결하고 명확한 작업 기반 설명을 제공하는 TLXML은 메타-러닝 과정의 해석 가능성을 높이는 데 기여합니다. 이 연구는 기계 학습 모델의 적응성을 평가하는 데 있어 과거 학습 작업의 중요성을 강조합니다.



### TrajFlow: A Generative Framework for Occupancy Density Estimation Using Normalizing Flows (https://arxiv.org/abs/2501.14266)
Comments:
          10 pages 6 figures 3 tables

- **What's New**: 이 논문에서는 TrajFlow라는 새로운 생성적 프레임워크를 소개하여 교통 참여자의 점유 밀도를 추정하는 방법을 제안합니다. TrajFlow는 causal encoder를 사용하여 관측된 경로의 의미 있는 임베딩을 추출하고 normalizing flow를 통해 이 임베딩을 디코딩하여 미래의 위치를 예측합니다. 또한, 이 프레임워크는 마가리널 분포를 모델링하여 기존의 접근 방식과 차별화됩니다.

- **Technical Details**: TrajFlow는 관측된 경로에서 마가리널 밀도를 모델링하며, 이를 통해 지속적인 미래 위치 샘플링을 가능하게 합니다. 이 구조는 neural differential equations를 기본으로 하여 설계되었으며, 경량화를 위한 실험도 포함되어 있습니다. 이 방식은 주어진 관찰 데이터를 기반으로 미래의 위치를 예측할 수 있도록 합니다.

- **Performance Highlights**: TrajFlow는 도전적인 경로 예측 벤치마크에서 뛰어난 성능을 보여주며 기존 모델보다 높은 정확도를 기록했습니다. 마가리널 밀도 포뮬레이션을 통해 TrajFlow는 개별 에이전트의 동작 경로와 점유 그리드 같은 다운스트림 작업을 수행하는 데 적합합니다. 이는 자율주행차와 교통 시스템의 복잡한 도전 과제를 해결하는 데 중요한 역할을 할 수 있습니다.



### Revisiting Applicable and Comprehensive Knowledge Tracing in Large-Scale Data (https://arxiv.org/abs/2501.14256)
- **What's New**: 이 논문에서는 기존 Deep Knowledge Tracing (DKT) 모델의 한계를 극복하고자 새로운 모델, DKT2를 제안합니다. DKT2는 최근 개발된 xLSTM 아키텍처를 기반으로 하여 더 나은 지식 상태를 생성할 수 있도록 설계되었습니다. 또한, Rasch 모델을 활용해 입력 표현을 강화하고, Item Response Theory (IRT)를 통해 학습한 지식을 해석합니다. 이로써, DKT2는 지식의 친숙한 부분과 낯선 부분을 구분할 수 있습니다.

- **Technical Details**: DKT2는 교육 심리학에서 유래된 Rasch 모델을 사용하여 과거 상호 작용을 처리하고, xLSTM을 통해 지식을 학습합니다. xLSTM은 기억 용량을 증가시키고, 정보를 더 효율적으로 검색할 수 있도록 하며, 이를 통해 DKT2의 성능을 극대화합니다. DKT2는 학생의 지식 상태와 예측된 문제를 통합하여 포괄적인 지식 상태를 생성합니다. 이 방법은 수집한 데이터를 기반으로 다양한 예측 작업에서 높은 예측 성능을 나타냅니다.

- **Performance Highlights**: 세 개의 대규모 데이터셋에서 DKT2는 17개의 기준 모델을 지속적으로 초월하는 성능을 보였습니다. 특히, 다양한 예측 작업에서 DKT2는 기존 모델의 한계를 극복하며, 실제 교육 환경에 유용할 수 있는 가능성을 보여줍니다. DKT2는 교육적 응용 분야에서 이론과 실제 적용 간의 간극을 해소하는 데 기여할 것으로 기대됩니다.



### Humanity's Last Exam (https://arxiv.org/abs/2501.14249)
Comments:
          25 pages, 6 figures

- **What's New**: 이번 논문에서는 Humanity's Last Exam (HLE)라는 새로운 다중 모드 벤치마크를 소개합니다. 기존의 벤치마크가 충분한 도전을 제공하지 못하는 상황에서, HLE는 3,000개의 질문을 포함하여 인간 지식의 최전선에서 설계되었습니다. 본 연구는 과거의 전통적인 벤치마크에서 LLM의 성능을 측정하는 데 있어 격차를 해소하고자 합니다.

- **Technical Details**: HLE는 수학, 인문학, 자연 과학 등 다양한 주제를 포함한 질문들로 구성되어 있으며, 선택형 및 단답형 질문 형식을 제공합니다. 질문은 명확하고 검증 가능하며, 단순한 인터넷 검색으로는 빠르게 답변할 수 없습니다. 데이터셋은 공신력 있는 주제 전문가들에 의해 개발되었으며, 정확한 답변과 함께 상세한 해설도 포함되어 있습니다.

- **Performance Highlights**: 최신 LLM은 HLE에서 10% 미만의 정확도를 보이며, 이는 현재의 LLM 능력과 전문가 인간의 성능 사이에 존재하는 큰 격차를 보여줍니다. AI 시스템이 여러 영역에서 인간 전문가의 성능에 접근함에 따라, 이러한 정밀한 측정은 연구 및 정책 결정을 위한 중요한 토대가 됩니다. HLE의 높은 성과는 클로즈드 엑센들 테스트에서 전문가 수준의 능력을 나타낼 것임을 시사합니다.



### A Data-driven Dynamic Temporal Correlation Modeling Framework for Renewable Energy Scenario Generation (https://arxiv.org/abs/2501.14233)
- **What's New**: 이 논문에서는 대기 시스템이 비선형 및 시간적으로 변하는 특성을 가진다는 점을 고려하여 재생 가능 에너지의 시나리오 생성을 위한 동적 시계열 상관 관계 모델링 프레임워크를 제안합니다. 새로운 분리형 매핑 경로(decoupled mapping path)를 통해 결합 확률 분포 모델링을 진행하며, 관계의 합리성을 보장하기 위해 적절한 점수 규칙(scoring rules)을 사용하여 회귀 작업을 수립합니다.

- **Technical Details**: 시나리오 생성 과정은 두 단계로 나뉘며, 첫 번째 단계에서는 동적 공분산 행렬(dynamic covariance matrix)을 기반으로 한 동적 상관 네트워크가 시간 상관 관계를 모델링합니다. 이는 재생 가능 에너지의 시간 변화 특성을 포착하고, 블랙박스 모델의 해석 가능성을 향상시킵니다. 두 번째 단계에서는 암묵적 분위수 네트워크(implicit quantile network)가 비모수적(nonparametric)이며 연속적인 방식으로 분배의 분위수 함수(marginal quantile function)를 모델링합니다.

- **Performance Highlights**: 실험 결과, 제안된 동적 상관 분위수 네트워크(dynamic correlation quantile network)가 최신 방법(state-of-the-art methods)보다 단기 재생 가능 에너지 시나리오 생성을 위한 불확실성 quantifying 및 동적 상관 관계 포착에서 뛰어난 성능을 보임을 확인했습니다.



### TFG-Flow: Training-free Guidance in Multimodal Generative Flow (https://arxiv.org/abs/2501.14216)
- **What's New**: 본 논문에서는 TFG-Flow라는 새로운 training-free guidance 방법을 소개합니다. TFG-Flow는 멀티모달 (multimodal) 생성 흐름을 위한 방법으로, 기존의 방법들이 연속적인 데이터만을 처리하는 한계를 극복합니다. 이 방법은 생성 모델을 다양한 결과로 이끌 수 있는 효율적인 기술로 주목받고 있으며, 선택적으로 분리된 연속 및 이산 변수를 안내할 수 있습니다.

- **Technical Details**: TFG-Flow는 확률 공간에서 분자의 표현을 정의하고, 예측 대상 속성 (target property)에 대해 유효한 분포를 생성하는 수학적 기초를 제공합니다. 본 연구에서는 TFG-Flow가 데이터를 생성하는 과정에서 발생할 수 있는 차원의 저주 (curse-of-dimensionality)를 해결하며, 이산 변수(주로 화합물의 속성)에서의 편향 없는 샘플링 (unbiased sampling) 속성을 유지합니다. 또한, 흐름 마진 (flow marginals) 보존, 목표 예측기와의 정렬, 그리고 이력과 목표의 조건부 독립을 보장하는 이론적인 결과를 제시합니다.

- **Performance Highlights**: TFG-Flow의 성능은 4개의 분자 설계 과제를 통해 검증되었습니다. 실험 결과, TFG-Flow는 원하는 특성을 가진 분자를 효과적으로 생성할 수 있는 잠재력을 보여주었으며, 이는 약물 설계 (drug design) 분야에서도 유망한 성과로 평가됩니다. 기존의 방법들과 비교했을 때, TFG-Flow는 더욱 유연하고 강력한 생성 모델을 가능하게 합니다.



### When GNNs meet symmetry in ILPs: an orbit-based feature augmentation approach (https://arxiv.org/abs/2501.14211)
- **What's New**: 이번 연구에서는 정수 선형 프로그램(ILP)의 대칭성 문제를 해결하기 위해 GNN(Graph Neural Networks)의 새로운 특성을 탐구합니다. 구체적으로, GNN이 대칭 변수를 구별하지 못하는 문제점을 다루고, 이를 개선하기 위한 특징 증강(feature augmentation) 방안을 제안합니다. 여기에 따라 우리는 대칭 변수를 그룹화하고 구성하는 오르빗 기반(orbit-based) 증강 방식도 개발하였습니다.

- **Technical Details**: 선형 목적 함수와 제약 조건을 가진 ILP의 대칭성 문제는 효율적인 해법 개발에 큰 도전 과제가 됩니다. 연구에서는 GNN의 순열 불변성(permutation invariance)과 불변성(permutation equivariance)을 통해 대칭 변수를 구별하는 어려움이 발생하는 이유를 분석합니다. 이러한 분석을 바탕으로 세 가지 중요한 원칙, 즉 구별 가능성(distinguishability), 증강 단순성(augmentation parsimony), 동형 일관성(isomorphic consistency)에 기반한 특징 증강 방법을 제안합니다.

- **Performance Highlights**: 우리의 실험 결과에 따르면 제안한 오르빗 기반 증강 방식은 ILP 문제의 훈련 효율성과 예측 성능을 획기적으로 향상시키는 효과가 있습니다. 대칭적 특성이 두드러지는 고전적인 ILP 문제에서 성능 개선을 확인할 수 있었으며, 기존 방법들과 비교 시 우수한 결과를 보였습니다. 이러한 접근법은 ILP 솔버의 대칭 인스턴스에 대한 처리 능력을 극대화할 수 있는 가능성을 보여줍니다.



### Coordinating Ride-Pooling with Public Transit using Reward-Guided Conservative Q-Learning: An Offline Training and Online Fine-Tuning Reinforcement Learning Framework (https://arxiv.org/abs/2501.14199)
- **What's New**: 이 논문은 다중 모달 교통 네트워크 내에서 라이드 풀링(ride-pooling)과 대중 교통(public transit) 간의 조정을 개선하기 위한 새로운 강화 학습(RL) 프레임워크인 Reward-Guided Conservative Q-learning(RG-CQL)을 소개합니다. 연구진은 각 라이드 풀링 차량을 마르코프 의사결정 과정(Markov Decision Process, MDP)으로 모델링하고, 최적의 운영 결정을 학습하기 위해 오프라인 훈련(offline training) 및 온라인 미세 조정(online fine-tuning) RL 프레임워크를 제안합니다.

- **Technical Details**: 오프라인 훈련 단계에서는 Conservative Double Deep Q Network(CDDQN)를 액션 실행기(action executor)로 활용하고, 보상 추정기를 위한 지도 학습 기반의 가이더 네트워크(Guider Network)를 개발하여 데이터 배치에서 액션-보상 관계에 대한 유용한 통찰을 추출합니다. 온라인 미세 조정 단계에서는 가이더 네트워크가 탐색 가이드(exploration guide) 역할을 하여 CDDQN이 알려지지 않은 상태-액션 쌍을 효과적이고 보수적으로 탐험하도록 돕습니다.

- **Performance Highlights**: 현실적 사례 연구를 통해 맨해튼의 실제 데이터를 사용하여 알고리즘의 효능이 증명되었습니다. 라이드 풀링을 대중 교통과 통합한 결과, 단독 라이드와 대중 교통 조정, 라이드 풀링과 대중 교통 없이 조정된 두 가지 벤치마크 사례보다 각각 17% 및 22% 더 나은 시스템 보상을 달성했습니다. 또한, 혁신적인 오프라인 훈련 및 온라인 미세 조정 프레임워크는 기존 온라인 RL 방법에 비해 데이터 효율성을 81.3% 개선하여 총 보상을 4.3% 증가시키고 과대 추정 오류를 5.6% 줄였습니다.



### Bi-directional Curriculum Learning for Graph Anomaly Detection: Dual Focus on Homogeneity and Heterogeneity (https://arxiv.org/abs/2501.14197)
Comments:
          8pages, 5 figures

- **What's New**: 최근의 Graph Anomaly Detection (GAD) 연구는 기존의 모델 기반 접근 방식을 넘어, 데이터 기반 접근 방식을 강조합니다. 이는 다양한 노드가 학습 과정에 기여하는 방식의 차이를 고려하지 않는 기존의 연구를 비판하며, 실행 가능하고 효과적인 모듈인 Graph Curriculum Learning (GCL)을 도입하여 기존 탐지 방법들을 최적화합니다. 특히 이 연구는 Bi-directional Curriculum Learning (BCL) 전략을 제안하여, 고유 모양의 노드 기능을 통해 GAD 모델의 성능을 향상시키고자 합니다.

- **Technical Details**: 연구에서는 GAD 모델의 훈련 전략을 개선하기 위해 새로운 난이도 점수 계산 방법을 도입했습니다. 이 방법은 노드의 유사성을 기반으로 노드를 그룹화하여, 동질성을 중시하는 방향과 비동질성을 중시하는 방향으로 각각 쉽게 훈련할 수 있는 샘플을 우선적으로 학습하도록 합니다. 이를 통해 GAD 모델은 훈련 과정에서 더 효과적으로 적절한 샘플을 선택할 수 있으며, 상대적으로 어려운 샘플은 점진적으로 통합됩니다.

- **Performance Highlights**: 제안된 BCL 방법은 기존의 탐지 프로세스에 신속히 통합할 수 있으며, 널리 사용되는 7개의 데이터셋에서 10개의 GAD 모델의 성능을 유의미하게 개선하는 효과를 보였습니다. 광범위한 실험 결과는 BCL이 GAD 모델에 대한 탐지 정확성을 크게 향상시킴을 증명합니다. 이 연구는 GAD의 학습 방식에서 혁신적인 변화를 가져오며, 다양한 응용 분야에서의 활용 가능성을 제시합니다.



### VarDrop: Enhancing Training Efficiency by Reducing Variate Redundancy in Periodic Time Series Forecasting (https://arxiv.org/abs/2501.14183)
- **What's New**: 이번 논문에서는 multivariate time series forecasting의 효율성을 높이기 위해 VarDrop이라는 새로운 전략을 제안합니다. VarDrop은 훈련 중 중복된 variate tokens를 생략하여 계산 비용을 줄입니다. 특히, k-dominant frequency hashing(k-DFH)를 활용하여 각 variate의 주파수 도메인에서 상관관계가 높은 토큰을 그룹화합니다. 이를 통해 기존 방법론보다 우수한 성능을 보여주는 것이 목표입니다.

- **Technical Details**: VarDrop 전략은 주어진 multivariate time series에 대해 fast Fourier transform을 수행하여 k-dominant 주파수를 식별합니다. 각 variate는 amplitude가 높은 k개의 주파수에 대한 해시값으로 그룹화되고, 계층화된 샘플링을 통해 대표적인 variate token이 선택됩니다. 선택된 token으로 sparse attention을 수행하여 계산 비용을 크게 줄일 수 있으며, 이 과정에서 중요한 정보를 보존합니다.

- **Performance Highlights**: 공식 벤치마크 데이터셋에서 실시한 실험 결과, VarDrop은 기존의 효율적인 기준 모델들을 초월하는 성능을 보였습니다. 다양한 데이터셋에서 실험을 통해 VarDrop의 적용 가능성과 실효성을 입증하였으며, 이를 통해 시간 시계열 예측의 정확성을 높이는 데 기여할 것으로 기대됩니다.



### RL + Transformer = A General-Purpose Problem Solver (https://arxiv.org/abs/2501.14176)
- **What's New**: 이 논문에서는 기존의 모델이 훈련된 내용을 초월하여 새로운 문제를 해결하는 능력을 갖추는 메타 학습(meta-learning) 방법으로, In-Context Reinforcement Learning (ICRL)이라는 emergent ability를 소개합니다. 특히, Llama 3.1 8B 모델이 RL 목표를 통해 스스로 학습하며, 훈련 데이터의 다양성에 강건함을 보여줍니다. 이 연구는 대규모 언어 모델이 비정상적인 문제를 해결할 수 있는 가능성을 제시하며, AI 시스템의 범용문제 해결 능력을 향상시킬 수 있음을 강조합니다.

- **Technical Details**: 연구에서는 LLaMA 3.1 8B Instruct라는 오픈 소스 대규모 언어 모델을 활용하여 ICRL의 기능을 탐구합니다. 모델은 여러 에피소드를 통해 강화 학습(reinforcement learning)으로 미세 조정(fine-tuning)되며, 이를 통해 새로운 문제를 해결하는 능력을 개발합니다. 이러한 접근 방식은 RL이 직면하는 샘플 효율성(sample efficiency) 문제를 해결하는 데 중점을 두며, 모델이 복잡한 작업을 수행하면서 학습한 기술을 조합할 수 있도록 합니다.

- **Performance Highlights**: ICRL로 훈련된 변환기(transformer)는 유사한 환경에서 높은 샘플 효율성을 나타내며, 비정상적(non-stationary) 환경에서도 강력한 성능을 유지합니다. 실험 결과, 모델은 높은 훈련 데이터 품질에 대한 민감도를 감소시키며, 최근 상호작용을 우선시하여 신속하게 적응합니다. ICRL의 도입으로 AI 시스템의 인간과 유사한 적응력을 지닌 발전이 가능하다는 것을 입증하며, 이는 AI 분야의 패러다임 전환을 의미합니다.



### Cybersecurity Assessment of Smart Grid Exposure Using a Machine Learning Based Approach (https://arxiv.org/abs/2501.14175)
- **What's New**: 파워 시스템의 안정적인 운영에 대한 위협이 급증함에 따라, 사이버 공격의 실시간 대응을 위한 머신러닝 기반의 평가 솔루션이 중요해졌습니다. 논문에서는 Mississippi State University와 Oak Ridge National Laboratory의 데이터셋을 활용해 XGB Classifier 모델링 접근법을 통해 다양한 파워 시스템 사건을 진단합니다.

- **Technical Details**: 이 연구는 892개의 특성과 하나의 레이블로 구성된 데이터셋을 사용하여, SHAP(Shapley Additive Explanations)를 활용한 설명 가능한 인공지능(XAI) 기법을 적용했습니다. XGB Classifier는 Attack Events, Natural Events 그리고 No-Events를 분류하며, 테스트 결과는 모든 서브 데이터셋에서 일관되게 좋은 성능을 보였습니다.

- **Performance Highlights**: 모델은 다양한 파워 시스템 사건을 정확히 식별하고 분류하는 데 있어서 높은 성능을 보여줍니다. 머신러닝 기법이 사이버 공격 탐지 및 평가에서 한층 더 발전할 수 있는 방향성을 제시하고 있습니다.



### UltraLightSqueezeNet: A Deep Learning Architecture for Malaria Classification with up to 54x fewer trainable parameters for resource constrained devices (https://arxiv.org/abs/2501.14172)
- **What's New**: 말라리아 진단을 위한 경량 딥러닝 접근 방식이 주목받고 있습니다. 본 연구에서는 저자원 환경에서 진단 개선 가능성을 가진 SqueezeNet1.1을 선택했습니다. 이는 SqueezeNet1.0의 개선된 버전으로, 원래 모델보다 2.4배 더 효율적입니다.

- **Technical Details**: 세 가지 초경량 아키텍처 변형을 SqueezeNet1.1에 제안하였습니다. 변형 1(모듈 1개), 변형 2(모듈 2개), 변형 3(모듈 4개)으로, 이들은 SqueezeNetV1.1(모듈 8개)보다 더 컴팩트합니다. NIH Malaria 데이터셋을 활용하여 각 모델의 성능을 정확도, 재현율, 정밀도, F1-score 및 AUC(곡선 아래 면적)로 평가하였습니다.

- **Performance Highlights**: SqueezeNet1.1 모델은 모든 메트릭에서 97.12%의 분류 정확도로 최고의 성능을 보였습니다. 변형 3(모듈 4개)은 96.55%의 정확도로 거의 동일한 결과를 제공하면서도 계산 오버헤드를 6배 줄였습니다. 변형 2는 28배, 변형 1은 54배의 훈련 가능 파라미터 감소를 보여주었습니다.



### Argos: Agentic Time-Series Anomaly Detection with Autonomous Rule Generation via Large Language Models (https://arxiv.org/abs/2501.14170)
- **What's New**: 본 논문에서는 클라우드 인프라에서 시계열 이상을 감지하기 위해 대규모 언어 모델(LLMs)을 활용하는 Argos라는 시스템을 소개합니다. Argos는 설명 가능하고 재현 가능한 이상 규칙을 중간 표현으로 사용하며, LLMs를 통해 이러한 규칙을 자율적으로 생성하는 것이 특징입니다. 이를 통해 여러 협업 에이전트를 통해 오류 없는 이상 규칙을 효율적으로 훈련하고, 저비용의 온라인 이상 감지를 위해 배포할 수 있습니다.

- **Technical Details**: Argos는 에이전트 기반 파이프라인을 사용하여 이상 탐지 규칙을 반복적으로 수정하고 정확성을 향상시키는 피드백 루프를 이메일링합니다. 각 반복에서 여러 에이전트가 규칙을 제안하고 검증하며 수정하여 문법 오류를 줄이고 정확성을 높이는 방식으로 작동합니다. 또한 Argos는 기존의 이상 탐지 시스템의 예측과 규칙의 예측을 집계하여 보다 정확한 결과를 도출합니다.

- **Performance Highlights**: Argos를 KPI와 Yahoo와 같은 공개 시계열 이상 탐지 데이터셋 및 Microsoft에서 수집한 내부 데이터셋을 통해 평가한 결과, 기존의 최첨단 방법 대비 F1 점수를 각각 9.5% 및 28.3% 향상시켰습니다. 또한, KPI, Yahoo 및 내부 데이터셋에서 각각 3.0배, 34.3배 및 1.5배 더 빠른 추론 속도를 달성하였습니다. 이러한 결과는 Argos의 효율성과 효과성을 입증합니다.



### Multimodal Prescriptive Deep Learning (https://arxiv.org/abs/2501.14152)
- **What's New**: 이번 논문에서는 Optimization(최적화)과 Machine Learning(기계 학습)의 아이디어를 결합하여 Prescriptive Neural Networks (PNNs)라는 멀티모달 딥 러닝 프레임워크를 소개합니다. 이 PNN은 Outcome-optimizing prescription(결과 최적화 처방)을 출력하기 위해 임베딩을 기반으로 훈련된 Feedforward Neural Network(전방향 신경망)입니다. 우리는 두 개의 실제 멀티모달 데이터셋을 통해 PNN이 TAVR 시술에서 예상되는 수술 후 합병증 비율을 32% 줄이고, 간 외상에서 사망률을 40% 이상 낮추는 치료를 추천할 수 있음을 보여줍니다.

- **Technical Details**: PNN 모델은 복잡한 데이터 구조를 처리하며, 멀티모달 및 유니모달 실제 응용 프로그램 모두에서 효과적입니다. 이 모델은 다양한 치료 시나리오에 유연하게 대처하며, 기존의 처방 방법들과 비교할 때 안정적이고 현실적인 결과를 제공합니다. 특히, 표 형식의 데이터셋에서는 Knowledge Distillation(지식 증류)를 적용하여 해석 가능성을 회복하고, PNN의 처방을 분류 목표로 하여 해석 가능한 Optimal Classification Tree(최적 분류 트리) 모델을 적합함으로써, 성능에 큰 손실 없이 해석 가능성을 확보할 수 있음을 보여줍니다.

- **Performance Highlights**: 우리의 멀티모달 PNN 모델은 무작위 데이터 분할에서 안정성을 보여주며, 기존의 처방 방법과 유사한 안정성을 보입니다. 또한, 다양한 데이터셋에서 현실적인 처방을 생성할 수 있습니다. PNN은 실제 응용 프로그램에서의 더 많은 제어를 사용자에게 제공하며, 결과적으로 예측 문제를 넘어서는 처방적 문제 해결 가능성을 제시합니다.



### An Extensive and Methodical Review of Smart Grids for Sustainable Energy Management-Addressing Challenges with AI, Renewable Energy Integration and Leading-edge Technologies (https://arxiv.org/abs/2501.14143)
- **What's New**: 이 논문에서는 스마트 그리드(Smart Grid)를 활용한 에너지 관리의 최신 동향을 다루고 있습니다. 디지털 통신 기술을 사용하여 전기의 생산과 분배의 지속 가능성을 높이는 스마트 그리드의 장점과 구성 요소를 설명합니다. 최근 몇 년간 많은 연구가 진행되었으며, 인공지능(AI), 데이터 분석(Data Analytics), 사이버 보안(Cybersecurity) 및 개인 정보 보호(Privacy)에 대한 통합도 강조하고 있습니다.

- **Technical Details**: 스마트 그리드 시스템은 에너지 자원의 관리와 통합을 위한 고급 기술과 방법론을 결합하여 운영 효율성을 극대화합니다. 에너지 관리 시스템(EMS) 및 에너지 저장 시스템(ESS)은 변동성 있는 재생 가능 에너지원(RES)의 통합과 안정성을 보장하는 데 필수적입니다. 이 시스템은 국제 전기기술 위원회(IEC) 61970 표준에 따라 정의되며, 지속적인 부하 공급을 통해 전력망의 안정성을 증대시킵니다.

- **Performance Highlights**: 이 논문은 지속 가능성 및 효율성을 달성하기 위한 데이터 분석과 지능형 기술의 통합이 에너지 관리 분야에 미치는 영향을 탐구합니다. 에너지 감사(Energy Auditing), 모니터링(Monitoring), 그리고 최적화(Optimization) 기술은 에너지 사용의 효율을 향상시키고 비용을 절감하는 데 기여합니다. 또한, 재생 가능 에너지원의 통합과 사이버 공격으로부터의 보안 확보는 스마트 그리드의 필수적인 현안으로 지목됩니다.



### Saliency Maps are Ambiguous: Analysis of Logical Relations on First and Second Order Attributions (https://arxiv.org/abs/2501.14136)
Comments:
          20 pages for the main article including references, 14 main article figures, 5 tables, 7 appendix figures

- **What's New**: 이 논문에서는 eXplainable Artificial Intelligence(XAI)에서의 saliency 방법의 한계를 탐구합니다. 이전 연구에서 발견된 bar 구조적 결함을 기반으로 하여 다양한 자료와 방식으로 saliency 방법의 오류를 재조명합니다. 특히, Global Coherence Representation(GCR) 프레임워크를 도입하여 입력을 생략한 효과적인 평가를 시도합니다.

- **Technical Details**: 제안된 GCR 프레임워크는 saliency 점수를 클래스 관련성 가중치로 해석하며, 각 입력 간의 상호작용을 탐구하여 실행 가능한 정보의 더 높은 순서의 상관관계를 명확히 합니다. 이 논문은 다양한 논리 데이터 세트와 saliency 방법을 사용하여 실험을 확장하고, saliency 점수의 해석에 세 가지 접근 방식을 제안합니다. 이는 saliency 방법의 성능을 상황에 맞게 분류하고 해석하는 데 도움을 줍니다.

- **Performance Highlights**: 본 연구에서는 saliency 방법의 성능을 다양한 논리적 시나리오에 따라 평가하고, 각 방법이 특정 상황에서 어떻게 작동하는지를 검토합니다. 이 과정을 통해 saliency 방법의 신뢰성과 유용성을 높이기 위한 새로운 방향을 제시하고, 이론적으로 근거 있는 해석을 위한 기초를 제공합니다. 특히, 두 번째 순위 attribution 점수를 적용하여 saliency 방법의 실제적인 이점을 강조합니다.



### Reinforcement Learning Platform for Adversarial Black-box Attacks with Custom Distortion Filters (https://arxiv.org/abs/2501.14122)
Comments:
          Under Review for 2025 AAAI Conference on Artificial Intelligence Proceedings

- **What's New**: 이 논문에서는 RLAB라는 새로운 강화 학습 플랫폼을 소개합니다. 이 플랫폼은 사용자가 서로 다른 왜곡 필터(distortion filters)를 선택하여 적대적 사례(adversarial examples)를 생성할 수 있도록 설계되었습니다. RLAB는 입력 이미지에 최소한의 왜곡을 추가하면서도 목표 모델을 오분류시키기 위해 강화 학습 에이전트를 활용합니다.

- **Technical Details**: RLAB는 입력 이미지를 각 단계에서 탐색하여 왜곡을 추가해야 할 민감한 영역을 식별합니다. 이를 위해 새로운 이중 행동(dual-action) 방법을 사용하여 목표 모델에 미치는 영향이 적은 노이즈를 제거합니다. 이러한 방식은 공격의 수렴(convergence)을 더 빠르고 효율적으로 만들어줍니다.

- **Performance Highlights**: 또한 RLAB는 특정 왜곡 유형에 대한 이미지 분류 모델의 견고성을 측정하는 데에도 사용될 수 있습니다. 적대적 샘플을 사용하여 모델을 재훈련(retraining)함으로써 벤치마크 데이터셋에서 평가할 때 견고성이 눈에 띄게 향상되었습니다. 제안된 플랫폼은 오분류를 유발하는 데 필요한 쿼리 수에서 최신 기술(state-of-the-art) 방법들을 초월했습니다.



### Selecting Critical Scenarios of DER Adoption in Distribution Grids Using Bayesian Optimization (https://arxiv.org/abs/2501.14118)
Comments:
          10 pages, 2 tables, 11 figures

- **What's New**: 이 논문에서는 배전망의 DER(분산 에너지 자원) 채택 시나리오 선정에 대한 새로운 방법론을 제안합니다. 기존의 결정론적 방법이나 임의의 시나리오 선택 방식에 의존하는 대신, 다목적 베이지안 최적화(multi-objective Bayesian Optimization)에 기반한 효율적인 검색 프레임워크를 도입하였습니다. 이 접근 방식은 전력 흐름 및 전압 위반과 관련된 다양한 목표를 고려하여, 특정 노드에서 DER이 채택될 경우 발생할 수 있는 전력망의 전반적인 스트레스를 평가합니다.

- **Technical Details**: 저자들은 복잡한 전력망 스트레스 메트릭을 블랙 박스 함수로 취급하고, 가우시안 프로세스(Gaussian Process) 대리모델로 이를 근사화합니다. 이 경우, 파레토-크리티컬(Pareto-critical) 시나리오의 확률을 기반으로 하는 획득 함수(acquisition function)를 설계합니다. 이 논문은 전통적인 철저한 검색 방식에 비해 수치적 보장을 제공하며, 속도 또한 대폭 향상되는 방법을 제시합니다.

- **Performance Highlights**: 실제 발전소를 기반으로 한 사례 연구에서, 제안된 방법론은 DER 채택 시나리오가 반드시 극단적인 집계 패턴과 관련이 없음을 체계적으로 보여줍니다. 두 개의 현실적인 피더(case study)에서의 결과는 본 방법론이 효과적이고 정확하다는 것을 증명하며, 배전망 계획에서의 중요성과 필요성을 강조합니다. 이는 전반적인 계획 개선을 위한 시나리오 선택의 새로운 기준을 제시하고 있습니다.



### Personalized Interpolation: An Efficient Method to Tame Flexible Optimization Window Estimation (https://arxiv.org/abs/2501.14103)
- **What's New**: 본 논문에서는 온라인 광고 시스템의 전환 최적화를 위한 개인화된 보간법(Personalized Interpolation)이라는 새로운 접근 방식을 제안하고 있습니다. 이 방법은 고정된 전환 시간 창 모델을 발전시켜 다양한 지연 범위의 전환을 정확하게 추정할 수 있도록 합니다. 또한 이 방법은 기존 모델과 통합하여 시스템 복잡성을 증가시키지 않으면서도 효율적으로 작동할 수 있습니다.

- **Technical Details**: 연구에서는 유연한 최적화 창(Flexible Optimization Window, FOW) 추정을 위해 누적 분포 함수(Cumulative Distribution Function, CDF)를 근사하는 작업으로 개념화하였습니다. 개인화된 보간법은 이 CDF의 단조 증가 및 오목 속성을 활용하여 효율적으로 FOW를 추정합니다. 또한 이 방법은 기존의 블랙박스 모델과 통합되어 추가적인 계산 비용 없이 활용될 수 있습니다.

- **Performance Highlights**: 제안된 보간법은 실험을 통해 높은 예측 정확도를 달성하며 이론적 성능 상한에 근접한 결과를 보여주었습니다. 이러한 성과는 추가적인 훈련 및 인프라 자원 없이도 효율적으로 FOW 추정을 가능하게 하는 점에서 매우 의미가 있습니다. 논문에 제안된 접근법은 온라인 광고 시스템의 전환 최적화를 크게 향상시킬 잠재력을 갖추고 있습니다.



### 5G LDPC Linear Transformer for Channel Decoding (https://arxiv.org/abs/2501.14102)
Comments:
          8 pages, 9 figures

- **What's New**: 이 논문은 새로운 형태의 완전 미분 가능(linearly differentiable) linear 시간 복잡도를 가진 transformer decoder를 소개하며, 5G New Radio (NR) LDPC를 복호화하는데 사용됩니다. 제안된 방법은 O(n^2) 복잡도를 가진 기존 transformer보다 O(n) 복잡도로 선형 블록 코드를 복호화할 수 있는 확장 가능한 접근 방식을 제공합니다. 이 연구는 5G NR LDPC 코드에 적용된 최초의 transformer 기반 복호화기임을 강조합니다.

- **Technical Details**: 제안된 시스템은 self-attention 메커니즘에 코드의 패리티 체크 매트릭스(parity-check matrix)를 통합하여 학습 과정을 가이드하면서 계산 효율성을 유지합니다. 핵심 혁신점은 계산 복잡도를 O(n^2)에서 O(n)으로 줄인 linear transformer decoder를 도입한 것입니다. 이 방법은 대형 블록 크기를 효율적으로 복호화할 수 있게 해주며, 차원의 저주(curse of dimensionality)에 제약받지 않습니다.

- **Performance Highlights**: 제안된 decoder는 비트 오류율(bit error rate) 성능에서 일반 transformer decoder와 동등한 성능을 달성하며, 단일 반복의 Belief Propagation(BP)에 비해 우수한 성능을 보여줍니다. 더불어, 큰 블록 코드에 대해서도 BP와 경쟁력 있는 시간 성능을 달성하였습니다. 연구 결과는 Nvidia의 Sionna 5G 및 6G 물리층 연구 소프트웨어를 통해 재현 가능한 결과를 보장합니다.



### Datasheets for AI and medical datasets (DAIMS): a data validation and documentation framework before machine learning analysis in medical research (https://arxiv.org/abs/2501.14094)
Comments:
          10 pages, 1 figure, 2 tables

- **What's New**: 본 논문은 데이터 엔지니어링이 발전하고 있음에도 불구하고, 머신러닝 관련 연구에서 데이터 검증과 문서화 절차 간의 일관성이 부족한 문제를 다루고 있습니다. 기존의 "Datasheets for Datasets" 프레임워크를 확장하여 "AI 및 의료 데이터셋을 위한 데이터시트 - DAIMS"를 제안합니다. DAIMS는 데이터 준비를 위한 체크리스트, 소프트웨어 도구, 데이터 문서화 양식, 데이터 사전 및 ML 분석 흐름도를 포함하고 있습니다.

- **Technical Details**: DAIMS는 데이터 표준화 요구 사항 24개를 포함하는 체크리스트를 제공하며, 이들 중 일부를 검사하고 검증하는 도구를 소프트웨어 도구로 제공합니다. 연구 질문을 제안된 ML 방법에 매핑하는 흐름도를 통해 연구자들이 보다 효과적인 ML 기술을 적용할 수 있도록 돕습니다. 체크리스트와 도구는 데이터셋을 표준화하고 연구 목적에 맞게 준비하는 데 중요한 역할을 합니다.

- **Performance Highlights**: DAIMS는 GitHub와 온라인 앱으로 제공되어 데이터셋 평가는 자동화할 수 있으며, ML 연구를 위한 데이터셋 준비를 효율적으로 진행할 수 있도록 돕습니다. 이는 의료 연구를 수행하는 연구자들에게 데이터셋 표준화의 참조 역할을 하고, 효과적인 머신러닝 기법 적용에 대한 로드맵을 제공합니다.



### Making Reliable and Flexible Decisions in Long-tailed Classification (https://arxiv.org/abs/2501.14090)
- **What's New**: 이 논문에서는 long-tailed classification의 문제점을 해결하기 위해 기존의 방법들이 간과하는 중요한 결정 리스크를 고려한 RF-DLC(이 Reliable and Flexible Decisions in Long-tailed Classification)라는 새로운 프레임워크를 제안합니다. 이 방법은 Bayesian Decision Theory를 활용하여, long-tailed 데이터의 분포와 의사결정 절차를 통합적으로 다루는 새로운 방법론을 구현합니다. RF-DLC는 다양한 위험을 반영할 수 있는 유틸리티 매트릭스(utility matrix)의 활용으로 유연한 적용이 가능하다는 장점 또한 가지고 있습니다.

- **Technical Details**: RF-DLC는 Bayesian Decision Theory의 통합 이득(integrated gain)을 도입하여, 특정 클래스 간의 오류에 따라 결정 리스크를 관리할 수 있도록 설계되었습니다. 이 방법은 효율적인 변분 최적화(variational optimization) 전략을 제안하여 의사 결정 목표의 리스크를 극대화합니다. 논문에서는 False Head Rate(FHR)라는 새로운 지표를 도입하여 tail 클래스에서 head 클래스로의 잘못된 예측을 수량화하고, 다양한 실험을 통해 RF-DLC의 신뢰성과 유연성을 실증합니다.

- **Performance Highlights**: 금번 연구는 RF-DLC가 기존의 long-tailed classification 방법과 비교하여 의사결정의 신뢰성을 크게 향상시키며, 정확도와 보정(calibration)과 같은 전통적인 지표들을 유지하거나 개선할 수 있음을 보여줍니다. 다양한 실제 과제에서 RF-DLC의 성능을 평가한 결과, tail 클래스에 대한 민감성 리스크를 효과적으로 관리할 수 있는 방법으로 자리매김할 가능성이 높습니다. 특히, 질병 탐지나 자율주행과 같은 응용 분야에 적용함으로써 더욱 중요한 의사결정에 실질적인 도움을 줄 수 있음을 강조합니다.



### GraphRAG under Fir (https://arxiv.org/abs/2501.14050)
Comments:
          13 pages

- **What's New**: 이 연구에서는 그래프 기반의 RAG(검색 증강 생성)인 GraphRAG의 독특한 공격 취약성을 탐구합니다. GraphRAG는 외부 지식을 다중 스케일의 지식 그래프로 변환하여 언어 모델이 더욱 세부적이고 폭넓은 맥락을 통합할 수 있도록 지원합니다. 기존 RAG 모델보다 GraphRAG가 단순한 공격에 더 높은 내성을 보이는 것을 발견했습니다. 그러나 이러한 기능들이 새로운 공격 표면을 창출한다는 점도 주목할 만합니다.

- **Technical Details**: GraphRAG은 지식 그래프를 사용하여 외부 지식을 조직하며, 질의마다 관련 정보를 검색하여 이를 프롬프트에 추가하고 응답을 생성합니다. 이 과정에서 관계 주입, 관계 증대 및 내러티브 생성의 세 가지 주요 전략을 사용하여 악의적인 콘텐츠가 포함된 오염 텍스트를 생성합니다. 특히, GragPoison이라는 새로운 공격 기법이 제안되며, 이는 관계를 기반으로 여러 질의를 동시에 타겟팅하는 독특한 접근 방식을 채택합니다.

- **Performance Highlights**: GragPoison의 실증적 평가 결과, 기존의 공격 방법들에 비해 98%의 성공률을 기록하며 공격의 효과성과 확장성을 크게 향상시켰습니다. 여러 GraphRAG 모델과 다양한 데이터셋을 통해 이러한 성능이 입증되었으며, GragPoison은 기존 방어에 대해 강한 저항력을 보입니다. 또한, 향후 연구 방향으로 새로운 방어 대책의 필요성을 강조합니다.



### SIDDA: SInkhorn Dynamic Domain Adaptation for Image Classification with Equivariant Neural Networks (https://arxiv.org/abs/2501.14048)
Comments:
          25 pages, 5 figures, 4 tables. code available at: this https URL

- **What's New**: 이 논문에서는 Sinkhorn divergence를 기반으로 하는 새로운 도메인 적응 훈련 알고리즘인 SIDDA(SInkhorn Dynamic Domain Adaptation)를 소개합니다. SIDDA는 효율적인 도메인 정렬을 가능하게 하며, 최소한의 하이퍼파라미터 조정과 컴퓨팅 비용으로 사용할 수 있습니다. 이 방법은 다양한 복잡성의 시뮬레이션 및 실제 데이터 세트에 대해 효과성을 입증하였으며, 특히 동등 변환 신경망(equivariant neural networks, ENNs)과 함께 사용할 때 분류 정확도와 모델 보정 효과를 크게 향상시킵니다.

- **Technical Details**: SIDDA는 옵티멀 전송(optimal transport) 이론을 기반으로 하며, 엔트로픽 정규화(entropic regularization)와 분류 손실(classification loss) 및 DA 손실(domain adaptation loss) 토큰의 가중치를 활성화하여 훈련 과정에서 하이퍼파라미터 조정을 최소화합니다. 이 알고리즘은 다양한 NN 아키텍처와 호환되며 복잡성에 따라 조정할 수 있는 우수한 방법을 제공합니다. 연구에서는 ENNs의 강 robustness를 함께 공부하며, SIDDA를 통한 개선 효과도 조사합니다.

- **Performance Highlights**: SIDDA는 레이블이 없는 타겟 데이터에 대해 약 40%의 분류 정확도 향상을 달성하여 NNs의 일반화 능력을 향상하도록 설계되었습니다. 또한, ECE(Expectation Calibration Error)와 Brier 점수(Brier score)에서 10배 이상의 개선 효과를 보이며, 모델 보정에서도 유의미한 결과를 나타냅니다. SIDDA의 자동화된 접근 방식은 다중 데이터 세트 연구 및 높은 일반화 가능한 모델 개발에 기여할 수 있는 잠재력을 지니고 있습니다.



### Efficient Precision Control in Object Detection Models for Enhanced and Reliable Ovarian Follicle Counting (https://arxiv.org/abs/2501.14036)
Comments:
          11 pages, 3 figures

- **What's New**: 이 논문은 고해상도 가상 슬라이드 스캐너를 활용하여 생리적 메커니즘을 정밀하게 분석하는 방법을 제시하고 있습니다. 특히, 마우스의 원시 난포(PMF) 수를 평가하는 데 기여하여 난소 예비능을 이해하는 데 도움을 줄 수 있습니다. 또한, 머신러닝의 주요 문제인 정밀도와 재현율 간의 균형을 효과적으로 다루는 새로운 다중 테스트 절차를 도입합니다.

- **Technical Details**: 본 연구에서 제안하는 알고리즘은 고유한 오브젝트 탐지(Object Detection) 기술을 통해 난포 계수를 신속하고 신뢰성을 높이며 결과의 재현성을 보장합니다. 연구진은 고급 생물학적 정보를 기반으로 하고, 이를 통해 정밀도를 보장하는 새로운 맥락 인식(object detection procedure)을 개발했습니다. 본 방법은 모델에 구애받지 않으며, 기존 모델의 성능(F1-score)을 향상시킬 수 있는 가능성을 제시합니다.

- **Performance Highlights**: 결과적으로, 제안된 모델은 난소 내 PMF의 정량적 분석을 개선하고, 검증 시간을 감소시킵니다. 또한, 새로운 고품질 이미지 데이터세트를 공개하여 생식 기능 관련 연구를 촉진할 것입니다. 모델에 대한 프로그래밍 코드 및 데이터 세트는 공개되어 연구자들이 쉽게 접근하고 활용할 수 있도록 제공됩니다.



### Transfer Learning of Surrogate Models via Domain Affine Transformation Across Synthetic and Real-World Benchmarks (https://arxiv.org/abs/2501.14012)
- **What's New**: 이번 연구에서는 비미분 가능 서러게이트 모델(surrogate model)인 랜덤 포레스트(random forest)를 사용하여, 특정 도메인 간의 아핀 변환(affine transformation)을 가정하고 모델을 전이하는 방법을 제안합니다. 이는 이전 연구에서 다뤘던 미분 가능 모델에 대한 전이 학습(transfer learning) 기법을 확장한 것으로, 적은 양의 데이터를 통해 새로운 작업에 대한 서러게이트 모델을 구축할 수 있게 합니다.

- **Technical Details**: 연구의 핵심은 도메인 시프트(domain shift) 문제를 해결하고자 하는 것으로, 입력 변수의 분포가 두 작업 간에 이동할 때, 예측 분포는 동일하다는 가정 하에 모델을 전이하는 방법에 집중합니다. 특히 아핀 전이의 경우, 새로운 작업의 입력 분포가 기존 작업의 아핀 변환으로 설명될 수 있음을 바탕으로, 전이된 모델의 성능을 향상시키는 접근 방식이 개발되었습니다. 연구진은 실험을 통해 이 방법이 실제 복잡한 문제에 효과적임을 입증했습니다.

- **Performance Highlights**: 제안된 방법은 Black-Box Optimization Benchmark (BBOB) 테스트베드와 네 개의 실제 전이 학습 문제에 대해 평가되었으며, 모델 훈련에서 필요한 데이터의 양과 계산 비용을 획기적으로 줄이는 성과를 나타냈습니다. 연구 결과, 전이된 모델은 처음부터 훈련한 모델에 비해 훨씬 낮은 대칭 평균 절대 백분율 오류(symmetric mean absolute percentage error, SMAPE)를 달성하였으며, 이는 비미분 가능 서러게이트 모델로서의 장점을 입증합니다.



### Scalable and Explainable Verification of Image-based Neural Network Controllers for Autonomous Vehicles (https://arxiv.org/abs/2501.14009)
Comments:
          11 pages, 5 figures

- **What's New**: 본 논문에서는 고차원 입력으로 인한 안전성 및 신뢰성과 관련된 문제들을 해결하기 위해, \

- **Technical Details**: SEVIN(Scalable and Explainable Verification of Image-Based Neural Network Controllers)라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 변분 오토인코더(Variational Autoencoder, VAE)를 활용하여 고차원 이미지를 낮은 차원의 설명 가능한 잠재 공간으로 인코딩합니다. 잠재 변수와 해당 제어 작업을 주석 처리하여 볼록 다면체(convex polytopes)를 생성하여 검증을 위한 구조적인 입력 공간을 정의함으로써 계산 복잡성을 크게 줄이고 확장성을 향상시킵니다.

- **Performance Highlights**: 실험 결과, SEVIN은 이미지 기반 신경망 컨트롤러에 대한 효율적이고 확장 가능한 검증을 달성하고 컨트롤러 행동에 대한 설명 가능한 통찰력을 제공함을 보여줍니다. 이 접근 방식은 안전이 중요한 시스템에서의 형식 검증 기술과 실제 응용 간의 격차를 해소합니다.



### Asymmetrical Latent Representation for Individual Treatment Effect Modeling (https://arxiv.org/abs/2501.14006)
- **What's New**: 이 논문은 Asymmetrical Latent Representation for Individual Treatment Effect (ALRITE)라는 새로운 CATE(조건부 평균 치료 효과) 추정 방법을 제안합니다. 이 방법은 두 개의 잠재 공간(latent space)을 비대칭적으로 탐색하여 각각의 공간이 제어와 치료 집단에 대한 효과적인 예측을 최적화하도록 설계되었습니다. ALRITE는 또한 이질적 효과(heterogeneous effects)의 추정 정확도를 높일 수 있는 여지를 제공합니다.

- **Technical Details**: ALRITE는 치료 집단(treated population)과 제어 집단(control population) 각각에 대해 잠재 공간을 최적화하는 기법입니다. 이 연구는 인과 추론(causal reasoning) 분야에서 응용할 수 있는 CATE 추정의 이론적 및 실용적 측면을 다룹니다. 적당한 가정 하에, ALRITE는 이질적 효과 추정의 정밀도(Precision of Estimation of Heterogeneous Effects)에 대한 상한선(upper bound)을 인정합니다.

- **Performance Highlights**: ALRITE 접근법은 기존의 최첨단(state-of-the-art) 기법들과 비교하여 실증적으로 성공적으로 검증되었습니다. 이러한 결과는 ALRITE가 치료 효과를 예측하는 데 매우 효과적임을 입증하며, 특히 의료, 사회학, 광고와 같은 다양한 분야에서의 활용 가능성을 제시합니다.



### Local Control Networks (LCNs): Optimizing Flexibility in Neural Network Data Pattern Captur (https://arxiv.org/abs/2501.14000)
- **What's New**: 본 논문에서는 Multi-layer perceptrons (MLPs)의 한계점을 지적하고, 각 노드에서 서로 다른 activation function을 사용하는 Local Control Networks (LCNs)를 제안합니다. 기존의 MLP는 고정된 활성화 함수를 사용하여 데이터 패턴을 포착하는 데 있어 유연성이 떨어지는데, LCN은 B-spline 함수를 통해 다채로운 활성화 곡선을 지원합니다. 이를 통해 LCN은 MLP보다 향상된 성능과 효율성을 제공하며, Kolmogorov-Arnold Networks (KANs)와 비교했을 때 더욱 접근성이 좋습니다.

- **Technical Details**: LCNs는 각 노드에서 독립적으로 학습 가능한 B-spline 함수의 매개변수를 사용하여 지역적으로 적응 가능한 활성화를 구현합니다. 해당 접근법은 활성화 함수의 다양성을 통해 신경망이 복잡한 데이터 패턴을 효과적으로 포착할 수 있도록 해줍니다. B-spline 곡선은 각 뉴런이 처리하는 데이터 특징에 맞춰 조정될 수 있어, 뉴런의 해석 가능성을 높여줍니다. 신경망 내에서 이 활성화 함수들은 각 뉴런이 특정 특징을 감지하는 데 도움을 주며, 결과적으로 네트워크의 안정성과 수렴 속도를 향상시킵니다.

- **Performance Highlights**: 실험 결과, LCNs는 컴퓨터 비전 작업에서 MLP보다 미세한 성능 개선을 보여주며 KANs보다 약 5% 높은 성능을 기록했습니다. 기본 머신러닝 작업에서는 MLP보다 1%, KANs보다 0.6% 개선된 성능을 나타냈습니다. 특히 기호 수식 표현 작업에서는 LCN과 KAN 모두 MLP를 초과하는 성능을 보였습니다. 이러한 결과는 각 노드 수준에서 다양한 활성화를 사용하는 것이 성능과 효율성을 개선하는 데 기여할 수 있음을 시사합니다.



### Predictive Learning in Energy-based Models with Attractor Structures (https://arxiv.org/abs/2501.13997)
- **What's New**: 이 논문에서는 에너지 기반 모델(EBM)을 활용하여 뇌의 예측 과정, 학습 및 추론을 포함하는 생물학적으로 그럴듯한 모델을 제안합니다. 특히, 이 모델은 행동 후 관찰을 예측하는 과정을 포착하기 위해 계층 구조와 지속적 매력자 신경망(CANN)을 통합합니다. 연구 결과, 제안된 모델은 훈련 환경에서뿐만 아니라 보지 않은 환경에서도 정확한 예측을 생성하는 성능을 보입니다.

- **Technical Details**: 모델은 마르코프 체인(Markov chain) 구조를 갖고 있으며, 가우시안 분포를 따르는 조건부 확률을 사용하여 계산 효율성을 높입니다. 또한, 에러 뉴런을 도입하여 학습 과정을 국소화하고, CANN을 활용해 과거 사건을 기억하는 방식을 개선합니다. 기존의 변별적 학습 알고리즘이나 이론에 의존하지 않고, 에너지 기반 원칙에 근거한 새로운 접근 방식을 제시합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 다양한 행동을 포함한 시나리오에서 효과적인 예측을 보여 주며, 머신 러닝 방법과 동등한 성능을 기록했습니다. 모델의 강점은 복잡한 확률 분포를 학습할 수 있는 능력과 새로운 환경에서도 합리적인 예측을 제공할 수 있는 점입니다. 이러한 성과는 뇌의 예측 메커니즘에 대한 깊은 이해를 돕는 데 기여할 것으로 기대됩니다.



### Dual-Branch HNSW Approach with Skip Bridges and LID-Driven Optimization (https://arxiv.org/abs/2501.13992)
- **What's New**: HNSW 알고리즘의 한계를 극복하기 위해 새로운 알고리즘을 제안합니다. 이 알고리즘은 local optima 문제와 cluster disconnections를 완화하며, 구성 속도를 향상시키고 추론 속도를 유지합니다. 주요 구성 요소로는 LID 기반의 삽입 메커니즘을 활용한 이중 분기 구조와 중간 레이어를 건너뛰는_bridge-building_ 기술이 포함됩니다.

- **Technical Details**: 이 연구에서는 각 노드를 LID(Local Intrinsic Dimensionality) 값을 기반으로 삽입하는 방법을 제안하여, 클러스터 내부의 이상치 노드를 더 잘 포착합니다. 이중 분기 구조를 채택하여 탐색 경로의 다양성을 높이고, 레이어를 건너뛰는 과정을 통해 추론 속도를 유지하면서 복잡도를 개선합니다. 실험을 통해 이 접근법의 정확도와 속도가 원래 HNSW보다 우수함을 보여주었습니다.

- **Performance Highlights**: 다양한 벤치마크와 데이터셋에서 실험을 수행한 결과, NLP(자연어 처리) 작업에서 18% 향상된 recall과 CV(컴퓨터 비전) 작업에서 최대 30% 향상된 성능을 달성했습니다. 또한 구성 시간은 최대 20% 감소했으며 추론 속도는 유지되었습니다. ablation 연구 결과 LID 기반 삽입이 성능에 가장 큰 영향을 미친 것으로 나타났습니다.



### FreEformer: Frequency Enhanced Transformer for Multivariate Time Series Forecasting (https://arxiv.org/abs/2501.13989)
- **What's New**: 이 논문에서는 다변량 시계열 예측을 위한 새로운 모델인 FreEformer를 소개합니다. 이 모델은 주파수 스펙트럼을 활용하여, 다양한 주파수에서의 시계열 구성에 대한 글로벌 관점을 제공합니다. 저자들은 주파수 도메인에서의 레빛 손실을 해결하기 위해 고안된 향상된 주의(attention) 메커니즘을 통해 듀얼 주의 행렬을 추가하여 예측 성능을 향상시킵니다.

- **Technical Details**: FreEformer는 주기적 패턴을 포착하기 위해 Discrete Fourier Transform (DFT)을 사용하여 시계열 데이터를 복잡한 주파수 도메인으로 변환합니다. 변환기 구조는 주파수 스펙트럼에 적용되어 상호 변량 의존성을 포착하며, 실제 및 허수 부분은 독립적으로 처리됩니다. 향상된 주의 메커니즘은 수정한 소프트맥스 주의 행렬에 학습 가능한 행렬을 추가하고, 행별 L1 정규화를 통해 구현됩니다.

- **Performance Highlights**: 다양한 전 세계의 데이터셋을 포함하는 18개 벤치마크 실험에서 FreEformer는 기존의 최첨단 모델들보다 일관되게 우수한 성능을 보여주었습니다. 특히, 향상된 주의 메커니즘이 기존 Transformer 기반 예측기의 성능을 지속적으로 향상시킴을 확인하였습니다. 본 논문은 날씨, 에너지 및 금융 등 다양한 분야에서의 강력한 예측 성능을 강조합니다.



### OstQuant: Refining Large Language Model Quantization with Orthogonal and Scaling Transformations for Better Distribution Fitting (https://arxiv.org/abs/2501.13987)
Comments:
          10 Pages

- **What's New**: 이 논문에서는 대형 언어 모델(Large Language Models, LLMs)의 압축 및 가속화에 널리 채택된 기법인 사후 학습 정량화(Post-training Quantization, PTQ)에 대해 다룹니다. 기존의 기법들이 데이터 분포를 최적화하지 못하는 한계를 극복하기 위해, 저자들은 정량화 공간 활용률(Quantization Space Utilization Rate, QSUR)이라는 새로운 메트릭을 도입하였습니다. QSUR는 변환된 데이터의 정량화 공간 내의 활용도를 측정함으로써 데이터의 정량화 가능성을 평가합니다.

- **Technical Details**: 논문에서는 QSUR와 함께 다양한 변환의 영향 및 한계를 논의하는 수학적 유도도 제공하여, 직교 및 스케일링 변환 기반의 정량화(Orthogonal and Scaling Transformation-based Quantization, OSTQuant)의 개발을 위한 기초 자료를 제공합니다. OSTQuant는 가중치와 활성화의 분포를 전체 정량화 공간에 걸쳐 최적화하기 위해 학습 가능한 동등 변환을 사용합니다. 또한, 노이즈를 줄이면서 PTQ에 의해 제한된 데이터 내에서도 풍부한 의미 정보를 유지할 수 있도록 설계된 KL-Top 손실 함수도 제안합니다.

- **Performance Highlights**: OSTQuant는 여러 LLM 및 벤치마크에서 기존 연구들을 초월하는 성능을 발휘합니다. 특히, W4만 사용한 설정에서, 99.5%의 플로팅 포인트 정확도를 유지하였으며, 더 도전적인 W4A4KV4 환경을 위해 LLaMA-3-8B 모델에서 성능 격차를 32% 줄였습니다. 이러한 결과는 OSTQuant가 최신 방법들과 비교했을 때 효율적인 성능 개선을 이루었다는 것을 보여줍니다.



### An Efficient Sparse Kernel Generator for O(3)-Equivariant Deep Networks (https://arxiv.org/abs/2501.13986)
Comments:
          12 pages, 9 figures, 3 tables

- **What's New**: 이번 연구는 회전 불변(Equivariant) 그래프 신경망에 대한 GPU 희소 커널 생성기를 소개합니다. 이는 화학 분야의 깊은 학습 과제에서 더욱 향상된 성능을 제공합니다. Clebsch-Gordon 텐서 곱을 활용하여 복잡한 계산을 효율적으로 수행하는 방법을 제안하며, 기존의 다양한 오픈소스 및 프라이빗 구현보다 현저하게 빠른 속도를 보여줍니다.

- **Technical Details**: 연구에서는 CG 텐서 제품을 통해 두 개의 밀집 특성 벡터를 희소 텐서와 결합하여 새로운 밀집 벡터를 생성합니다. 이 연산은 대규모 원자 데이터셋에 대해 수백만 번 수행되어 성능 저하의 주요 원인이 됩니다. GPU 메모리 관리를 통해 작은 연산을 극대화하고, CG 텐서 제품과 그래프 컨볼루션의 통합을 통해 메모리 트래픽을 줄이는 방법이 소개됩니다.

- **Performance Highlights**: 제안한 커널은 NVIDIA cuEquivariance보다 최대 4.5배 속도가 향상되며, e3nn 패키지보다 10배 이상 향상된 성능을 보여줍니다. MACE 화학 기초 모델에 대해서도 원본 최적화 버전보다 5.3배의 추론 시간 단축效果를 나타냅니다. 이러한 성과는 회전 불변 아키텍처의 성능을 극대화하는 데 중요한 역할을 합니다.



### Pilot: Building the Federated Multimodal Instruction Tuning Framework (https://arxiv.org/abs/2501.13985)
- **What's New**: 본 논문에서는 분산 장치에서 다양한 모달리티의 지시 데이터로 MLLMs를 협력적으로 세밀하게 조정하는 데 중요한 새로운 작업인 Federated Multimodal Instruction Tuning(FedMIT)을 탐구합니다. 이를 위해 'Pilot'이라는 연합 다중 모달 지시 조정 프레임워크를 제안합니다. 이 프레임워크는 비전 인코더와 LLM을 연결하는 부분에 'adapter on adapter'의 두 가지 단계로 통합되어 있습니다.

- **Technical Details**: 1단계에서는 비주얼 정보를 기반으로 작업 특화 기능과 클라이언트 특화 기능을 추출합니다. 2단계에서는 Cross-task Mixture-of-Adapters(CT-MoA) 모듈을 구축하여 작업 간 상호작용을 수행합니다. 각 클라이언트는 지역 데이터의 개인화 정보를 캡처할 뿐만 아니라, 다른 작업에서의 일반적인 지식을 학습하여 모델 성능을 향상시킬 수 있도록 합니다.

- **Performance Highlights**: 이 프레임워크는 다양한 로컬 클라이언트의 분산 데이터를 협력적으로 활용하여 지시 조정 중 작업 이질성에 영향을 받지 않고 교차 작업 지식을 학습할 수 있습니다. 실험을 통해 Pilot 방법이 최신 LLaVA 모델에서 두 가지 다른 교차 작업 시나리오에서 효과적임을 확인했습니다.



### FlexiGPT: Pruning and Extending Large Language Models with Low-Rank Weight Sharing (https://arxiv.org/abs/2501.14713)
Comments:
          Accepted to NAACL 2025 - Main Conference

- **What's New**: 최근 LLM의 사용이 늘어나면서 메모리 제약이 있는 기기에서도 효율적인 모델 배치를 위한 방법이 필요해졌습니다. 본 연구에서는 모델 블록을 선택적으로 제거하는 pruning 기법을 제안하며, 이를 통해 메모리가 한정된 환경에서 LLM의 성능을 유지할 수 있습니다. 특히, 기계 학습에서的重要한 블록의 중요도를 기반으로 하는 측정 방식을 도입하여 최적의 대체 전략을 구현합니다.

- **Technical Details**: 이 논문에서는 ShortGPT의 Block Influence (BI) 점수를 활용하여 모델 블록을 제거하고, 각각의 제거된 블록을 기존의 특정 블록에서 공유된 가중치로 교체하는 방법을 제안합니다. Low-Rank Adapters (LoRA)와 같은 기술을 도입하여 가중치 공유를 통한 파라미터 효율성을 극대화하고, 더 나은 특성 정규화(output feature normalization)를 통해 출력 안정성을 높입니다. 이러한 기법들은 대체 블록의 초기화와 그 후의 모델 적응을 촉진하는 역할을 합니다.

- **Performance Highlights**: Empirical evaluations of this method, named FlexiGPT, show significant improvements in performance, achieving state-of-the-art results on 5 out of 6 benchmarks at a 30% compression rate and on all benchmarks at a 40% rate. 이 연구는 또한 TinyLLaMA와 같은 소형 모델을 반복하여 효율적으로 확장할 수 있는 가능성을 보여주며, 0.3% 토큰만으로도 6/6 벤치마크에서 놀라운 성능 상승을 달성합니다. 이러한 결과를 통해 LLM을 메모리 제약이 있는 기기에서도 효과적으로 사용할 수 있도록 하는 새로운 접근 방식을 제안합니다.



### Overcoming Fairness Trade-offs via Pre-processing: A Causal Perspectiv (https://arxiv.org/abs/2501.14710)
- **What's New**: 이번 논문에서는 머신 러닝 모델의 공정성(fairness)과 정확도(accuracy) 간의 상충 관계를 해결하는 새로운 접근법을 제시합니다. FiND(허구적이고 규범적으로 바람직한) 세계라는 개념을 통해, 공정 특성이 예측 변수에 미치는 영향을 제거하고 데이터를 조정하는 방법을 제안합니다. 이 연구는 기존의 공정성 메트릭들과 정확도 간의 무모호한 상관관계가 실제로 존재할 수 있음을 보여주며, 이를 통해 공정성과 높은 예측 성과를 동시에 달성할 수 있는 방법을 제공합니다.

- **Technical Details**: 논문에서는 FiND 세계에서의 공정한 그리고 비편향된 데이터 품질이 결정적인 역할을 한다고 주장합니다. 이는 전통적인 공정성 메트릭이 서로 상충하지 않고, 동시에 만족될 수 있도록 하는 구조적 기반을 제공합니다. 저자들은 FairAdapt와 Residual-Based Warping이라는 두 가지 인과적 사전 처리 방법을 조사하고, 이들이 FiND 세계를 근사하는 데 성공적인지를 평가하여 실제 사용 사례에서의 유용성을 검증합니다.

- **Performance Highlights**: 시뮬레이션과 실험 연구를 통해 이 사전 처리 방법들이 FiND 세계를 성공적으로 근사함을 보여주며, 공정성을 유지하면서도 예측 성과를 손상시키지 않음을 입증했습니다. 논문에서 제안된 방법론은 공정성과 성능 간의 무너질 듯한 상충 관계를 실제 환경에서도 해결할 수 있는 실행 가능한 솔루션을 제공합니다. 이를 통해 다양한 테스트 환경에서의 공정성 메트릭을 동시에 만족시킬 수 있는 가능성을 열어줍니다.



### Decision-Focused Learning for Complex System Identification: HVAC Management System Application (https://arxiv.org/abs/2501.14708)
Comments:
          12 pages, 9 figures, submitted to ACM e-energy 2025

- **What's New**: 이 논문은 기존의 통계적 메트릭 최소화나 태스크에 중립적인 손실을 목표로 하는 전통적인 훈련 방법과 달리, Decision-Focused Learning (DFL)을 통해 머신 러닝 모델을 하위 의사결정 도구의 최적 성능을 위해 훈련할 수 있음을 주장합니다. DFL을 활용하면 시스템 동역학의 매개변수를 학습할 수 있으며, 이는 볼록 최적화 제어 정책의 제약 조건으로 표현됩니다. 또한, 제어 정책이 적용되면 시스템의 동작이 변화하기 때문에, 과거 데이터의 적용 가능성이 낮아지는 문제를 해결합니다.

- **Technical Details**: DFL은 시스템 식별과 제어를 동시에 수행하며, 모델의 정확도가 제어와 가장 관련이 깊은 영역에 집중되도록 보장합니다. 특히, 블랙 박스 시스템이 비미분 가능하므로, 시스템 응답 측정만으로 구성된 손실 함수가 설계되었습니다. 이 논문에서는 역사적 데이터를 기반으로 사전 훈련하고 DFL의 안정성을 높이기 위해 제약 완화를 제안합니다.

- **Performance Highlights**: 제안된 방법은 미국 덴버에 위치한 15개의 구역을 가진 실제 건물의 HVAC 관리 시스템에 적용하여 그 유용성을 입증했습니다. 기존의 RC 모델은 역사적 데이터를 기반으로 한 파라미터로 HVAC 전력 소비를 과소 평가한 반면, DFL을 통해 얻어진 파라미터를 사용한 경우에는 예상 비용을 단지 3%만 과소 평가했습니다. 이러한 결과는 DFL이 효율적인 제어 정책에 기여할 수 있음을 보여줍니다.



### NLP-based assessment of prescription appropriateness from Italian referrals (https://arxiv.org/abs/2501.14701)
- **What's New**: 이 연구는 이탈리아의 처방 적합성을 평가하기 위해 자연어 처리(Natural Language Processing, NLP) 파이프라인을 제안합니다. 첫 번째로, 각 처방의 이유를 종합적으로 요약하고 적합성을 정량화하는 접근 방식을 개발하였습니다. 특히 이 연구는 2019년부터 2021년까지 로마냐 지역의 496,971개의 처방 데이터를 분석하는 케이스 스터디를 통해 검증을 수행하였습니다.

- **Technical Details**: 이 파이프라인은 변환기(transformer) 기반 모델의 임베딩을 활용하여 처방 텍스트를 클러스터링하고, 이러한 클러스터를 적절한 라벨에 매핑합니다. BERT 모델을 통해 생성된 임베딩을 활용한 클러스터링 과정을 통해 기존 가이드라인과의 일치를 평가합니다. 1,000개의 처방을 수동으로 주석하여 결과를 검증하였으며, 이로써 파이프라인의 정확성을 확보하고자 하였습니다.

- **Performance Highlights**: 제안된 파이프라인은 고도의 성능을 보였으며, 수동으로 주석된 데이터 셋에서 처방 이유의 정밀도(Precision)는 92.43%, 재현율(Recall)은 83.28% 그리고 적합성에서는 93.58%의 정밀도와 91.52%의 재현율을 기록하였습니다. 전체 데이터셋 분석 결과, 34.32%의 처방이 적합하며, 34.07%는 부적합하다고 평가되었습니다. 이러한 발견은 로마냐 지역의 건강 정책 강화와 부적절한 처방 감소에 기여할 수 있습니다.



### Rethinking Foundation Models for Medical Image Classification through a Benchmark Study on MedMNIS (https://arxiv.org/abs/2501.14685)
Comments:
          submitted to MIDL2025

- **What's New**: 본 연구에서는 MedMNIST 데이터셋을 통해 의료 이미지 분류 작업에서 사용되는 다양한 foundation model의 성능을 비교합니다. 이 연구는 다양한 convolutional 및 Transformer 기반 모델을 포함하여 이전 연구에서 제시되지 않은 모델 선택을 통해 더 깊이 있는 통찰력을 제공합니다. 또한, 이미지 크기와 리사이징 전략의 효과에 대한 탐구도 포함되어 있어, 의료 이미지 분류 작업으로의 전이 가능성에 대한 새로운 인사이트를 제공합니다.

- **Technical Details**: 의료 이미지 분류 연구에서 사용된 foundation model은 CNN 기반 및 ViT 기반 아키텍처를 통해 수행되었습니다. CNN 모델은 VGG16, DenseNet-121 등 다양한 백본을 포함하며, ViT 모델은 최적화 전략에 따라 구분하여 4개의 모델이 활용되었습니다. 모델 평가를 위한 학습 과정에서는 15,000회의 반복 훈련이 진행되며, AdamW 옵티마이저가 사용되었습니다.

- **Performance Highlights**: 연구 결과, 다양한 기법으로 훈련된 foundation model들은 의료 이미지 분류 작업에서 뛰어난 잠재력을 보여주었습니다. 각 모델의 성능은 평균 ± 표준편차 방식으로 평가되어, 서로 다른 이미지 해상도에서 일관된 결과가 나타났습니다. 또한, 실험을 통해 제안된 다양한 리사이징 전략이 전반적인 분류 성능에 긍정적인 영향을 미쳤음을 확인하였습니다.



### End-to-end workflow for machine learning-based qubit readout with QICK and hls4m (https://arxiv.org/abs/2501.14663)
- **What's New**: 본 논문에서는 QICK 플랫폼에 공기된 신경망(Neural Networks, NNs)을 이용하여 초전도 큐비트(quantum bit)의 판독 작업을 위한 최적화된 비즈니스 프로세스를 제시합니다. Xilinx RFSoC FPGA에서 구현된 맞춤형 펌웨어와 소프트웨어를 활용하여 기계 학습(machine learning, ML)을 통해 큐비트 판독의 정확성 및 확장성 문제를 해결하고자 합니다. 특히, hls4ml 패키지를 활용하여 하드웨어에 효율적으로 ML 모델을 변환하는 방법을 제시합니다.

- **Technical Details**: 우리는 ML 모델을 FPGA 구현으로 변환하기 위한 end-to-end 워크플로우를 개발하고, 이를 통해 단일 Transmon 큐비트의 판독에서 96%의 단일 샷 충실도를 달성하였습니다. 이 워크플로우는 Python API를 이용하여 기존 기술자들이 소프트웨어와 하드웨어를 통합하는 데 필요한 노력을 최소화하도록 설계되었습니다. 논문에서는 hls4ml 패키지를 기반으로 하는 신경망 디자인 최적화 기법도 설명합니다.

- **Performance Highlights**: 실험적으로 우리는 단일 초전도 Transmon 큐비트의 판독에서 32ns의 지연 시간과 16% 미만의 FPGA Look-Up Table 자원 사용률을 기록했습니다. 이러한 결과는 기계 학습 기반의 큐비트 판독 및 적응 제어를 발전시키기 위한 접근 가능한 워크플로우를 제공합니다. 이 연구는 향후 양자 정보 처리 응용에 기여할 것으로 기대됩니다.



### Mean-field limit from general mixtures of experts to quantum neural networks (https://arxiv.org/abs/2501.14660)
- **What's New**: 이번 연구에서는 Mixture of Experts (MoE) 모델이 supervised learning 문제를 통해 gradient flow에 의해 훈련될 때의 점근적 행동(asymptotic behavior)을 살펴봅니다. 주요 결과로는 전문가(expert)의 수가 증가함에 따라 chaos가 전파(propagation of chaos)된다는 것을 입증하였습니다. 또한, 파라미터의 경험적 분포(empirical measure)는 비선형 연속 방정식을 가능하게 하는 확률적 분포에 가까워진다는 것을 보여줍니다.

- **Technical Details**: 연구에서는 quantum neural network로 생성된 MoE에 대해 다룹니다. 각 전문가의 파라미터는 훈련 세트를 기반으로 최적화되며, 연구는 gradient-based 방법을 통한 파라미터 최적화에 중점을 두고 있습니다. 여기서 MoE는 이전의 연구와 달리 quantum feature space를 통해 performance를 개선하고, QML(Quantum Machine Learning) 알고리즘의 속성을 활용합니다.

- **Performance Highlights**: 우리는 MoE를 통해 각 전문가는 파라미터가 설정된 quantum 회로로, 이 회로의 결과는 quantum observable의 기대값(expectation value)으로 나타납니다. 다양한 분야에서 MoE의 응용 가능성이 논의되고 있으며, 그 성능 개선(PERFORMANCE IMPROVEMENT)에 대한 수학적 근거를 제시합니다. 앞으로의 연구는 MoE의 서로 다른 전문가들이 독립적으로 행동하는 현상인 'chaos 전파'를 통해 이론적 및 실험적 결과를 검증할 예정입니다.



### Optimal Transport Barycenter via Nonconvex-Concave Minimax Optimization (https://arxiv.org/abs/2501.14635)
- **What's New**: WDHA (Wasserstein-Descent \dot{\mathbb{H}}^1-Ascent) 알고리즘은 복잡도를 줄이고 정확한 Wasserstein barycenter를 계산할 수 있는 새로운 방법을 제안합니다. 이 알고리즘은 두 가지 Wasserstein 및 Sobolev 최적화 기하학을 번갈아 사용하여 효율적인 계산을 가능하게 합니다. 특히 이 알고리즘은 1024x1024 해상도의 이미지와 같은 대규모 데이터에서 우수한 성능을 보입니다.

- **Technical Details**: WDHA 알고리즘은 최적 수송(Wasserstein) barycenter 문제를 비볼록-오목(Minimax) 최적화 문제로 재구성합니다. 알고리즘의 혁신은 바리센터 업데이트를 위한 Wasserstein 그래디언트와 Kantorovich 쌍대 형태 업데이트를 결합하는 것입니다. 이 과정에서 각 반복의 시간 복잡도가 O(m log m)으로 줄어들며, 이는 기존의 OT 맵 계산에서 요구되는 O(m^3)의 시간 비용과 비교됩니다.

- **Performance Highlights**: WDHA 알고리즘은 고해상도 데이터에서 기존의 Sinkhorn 유형 알고리즘에 비해 뛰어난 계산 효율성과 정확성을 보여줍니다. 실험을 통해 다양한 테스트 환경에서의 우수한 수렴 속도와 성능을 입증하였으며, 고차원 배열에서도 효과적으로 작동하는 것이 평가되었습니다.



### Accelerated Preference Elicitation with LLM-Based Proxies (https://arxiv.org/abs/2501.14625)
- **What's New**: 이번 논문에서는 조합 경매에서 입찰자의 선호를 자연어로 제시하는 새로운 접근 방식을 제안합니다. 기존의 선호 끌어내기(preference elicitation) 기술들은 주로 질의(query)-기반의 방법론에 의존했으나, 새로운 접근은 LLM(대형 언어 모델)을 활용하여 더 효율적이고 직관적인 의사소통을 목표로 합니다. 이로 인해 인지적 부담을 줄이고 통신 오버헤드를 최소화할 수 있습니다.

- **Technical Details**: 우리는 LLM 기반의 프록시를 사용하여 입찰자의 선호를 끌어내는 새로운 메커니즘을 개발했습니다. 이 메커니즘은 LLM 파이프라인과 DNF-올바른 학습(DNF-proper learning) 기법을 결합하여, 의사소통이 제한된 상황에서도 빠르게 선호를 추정할 수 있도록 설계되었습니다. 실험을 통해 다양한 LLM 프록시 디자인을 테스트하고, 입찰자의 선호를 효과적으로 파악하는 방법을 탐구하였습니다.

- **Performance Highlights**: 우리의 실험 결과, 제안한 LLM 프록시 디자인은 기존의 전통적인 방법보다 약 다섯 배 적은 질의 수로 효율적인 결과에 도달할 수 있음을 보여주었습니다. 이는 조합 경매의 실행 가능성을 높이고, LLM을 활용한 새로운 접근 방식의 잠재력을 입증합니다. 이 연구는 기존의 기계 학습 기법들과 비교할 때 자연어 통합의 결여를 극복하는 방향으로 나아가는 중요한 단계를 제시합니다.



### Single-neuron deep generative model uncovers underlying physics of neuronal activity in Ca imaging data (https://arxiv.org/abs/2501.14615)
Comments:
          12 pages, 5 figures, ECCB 2025

- **What's New**: 이 논문에서는 단일 신경 세포 (single-neuron) 분석을 위한 새로운 프레임워크를 제안합니다. 기존의 spike inference 알고리즘에 의존하지 않고, 각 개별 신경 세포의 시공간 신호를 저차원 공간으로 임베딩하는 autoregressive variational autoencoders (AVAE)를 사용합니다. 이러한 접근법은 신경 세포 활동을 시각화하고 클러스터링하는 데 유리하며, 단일 신경 세포의 명확한 표현 학습을 가능하게 합니다.

- **Technical Details**: 제안된 AVAE는 비선형 매핑을 학습하여 신경 세포의 시공간 신호를 저차원 공간에 임베딩합니다. 이를 통해 신경 세포의 하위 집단을 식별하고, 활동 및 연결성 변화를 포착할 수 있습니다. 네트워크는 Izhikevich의 통합 및 화재 모델을 따라 구성되며, 시뮬레이션된 데이터는 이 모델의 성능을 평가하는 데 사용됩니다.

- **Performance Highlights**: AVAE는 기존의 선형 방법인 주성분 분석 (PCA)과 비교하여 우수한 재구성 성능을 입증하였습니다. 제안된 모델은 신경 세포 간의 연결성과 물리적 구조를 효과적으로 학습하며, 시각화 및 클러스터링과 같은 후속 작업에서 보다 적합한 잠재 표현을 제공합니다. 이 연구는 신경 과학 분야의 단일 세포 데이터 분석을 위한 강력한 도구로서 AVAE의 잠재력을 강조합니다.



### Improved Vessel Segmentation with Symmetric Rotation-Equivariant U-N (https://arxiv.org/abs/2501.14592)
Comments:
          Accepted by IEEE ISBI 2025

- **What's New**: 본 논문에서는 효율적인 대칭 회전 불변(symmetric rotation-equivariant) 컨볼루션(SRE-Conv) 커널을 U-Net 아키텍처에 적용하여 의료 이미지 분할 성능을 극대화하는 방법을 제안합니다. 기존의 CNN 기반 분할 기법은 회전이나 반사를 고려하지 않음으로써 성능 저하를 초래해 왔습니다. 제안하는 SRE U-Net 모델은 특히 회전된 이미지를 처리할 때 우수한 결과를 보여주며, 이는 기존의 불변 학습 방법보다 더 적은 학습 파라미터와 메모리 비용을 요구합니다.

- **Technical Details**: 이 연구에서는 SRE-Conv 커널을 통해 회전 및 반사 불변 기능을 학습합니다. SRE-Conv는 커널을 중앙 대칭 형태로 매개화하여 많은 중복 값을 갖도록 설계되었습니다. 최종적으로 SRE-Conv 커널은 파라미터를 효율적으로 분할하여 정의된 규칙에 따라 생성되며, 이는 표준 컨볼루션보다 더 낮은 계산 복잡도와 부동 소수점 연산 수(FLOPs)를 달성하게 합니다.

- **Performance Highlights**: DRIVE 망막 혈관 분할 데이터셋을 통해 제안된 SRE U-Net의 성능을 평가한 결과, 회전된 이미지 입력에 대해 정확한 혈관 세분화를 달성했습니다. 제안된 방법은 표준 U-Net과 비교하여 성능이 현저히 향상되었고, 기존의 불변 학습 방식 및 최신 기술(State-of-the-art)과 비교하여도우수한 결과를 보였습니다. 검증된 수치 성과는 다양한 회전 조건에서의 일관된 결과를 제공하여 실제 진단 애플리케이션에서도 유용함을 입증합니다.



### coverforest: Conformal Predictions with Random Forest in Python (https://arxiv.org/abs/2501.14570)
Comments:
          In peer review

- **What's New**: 이 논문에서는 새로운 conformal prediction 방법인 coverforest를 소개합니다. coverforest는 Random Forest 모델을 최적화하여 효율적인 cross-conformal 예측을 가능하게 하며, 이는 기존의 split conformal 방법보다 뛰어난 데이터 효율성을 가지고 있습니다. 이를 통해 회귀 및 분류 작업에서 사용 가능한 여러 방법론을 통합하여 예측을 수행합니다.

- **Technical Details**: coverforest는 Python 패키지로, conformal prediction 테크닉을 Random Forest에 최적화한 것입니다. 이 패키지는 split conformal, CV+, Jackknife+-after-bootstrap, 적응형 예측 집합(APS)과 같은 다양한 conformal prediction 기법을 지원합니다. 또한, parallel computing과 Cython 최적화를 활용하여 out-of-bag 계산 속도를 향상시킵니다.

- **Performance Highlights**: 실험 결과, coverforest는 원하는 커버리지 수준을 달성하면서도 다른 기존 구현보다 트레이닝 및 예측 시간이 2-9배 더 빠른 것으로 나타났습니다. 이러한 성과 덕분에 실용적인 머신러닝 어플리케이션에서의 유용성이 크게 증가됩니다. 실제로 GitHub에 소스 코드가 공개되어 있어, 연구자 및 개발자들이 쉽게 활용할 수 있도록 하고 있습니다.



### A Recurrent Spiking Network with Hierarchical Intrinsic Excitability Modulation for Schema Learning (https://arxiv.org/abs/2501.14539)
Comments:
          31 pages, 9 figures

- **What's New**: 이 연구는 RSNN(Recurrent Spiking Neural Network)을 이용한 스키마 학습 모델링의 첫 시도로, 기존 RNN의 한계를 극복하기 위해 고안된 고급 RSNN 모델인 HM-RSNN(Hierarchical Modulation of Intrinsic Excitability with Recurrent Spiking Neural Networks)을 제안합니다. HM-RSNN은 계층적 내재 흥분성 조절 메커니즘으로 구성되어 있으며, 이는 신경세포의 생화학적 상태를 조정하여 작업별 요구를 최적화하도록 돕습니다. 또한, 스키마 학습에 대한 포괄적인 연구를 가능하게 하는 일반화된 스키마 학습 프레임워크와 세 가지 새로운 인지 과제를 도입했습니다.

- **Technical Details**: 소개에서 제안하는 HM-RSNN 모델은 두 가지 수준에서 조절됩니다. 상위 레벨은 작업별 요구에 맞춰 흥분성 특성을 선택하고, 하위 레벨은 이러한 선택된 특성을 더욱 세부적으로 조정합니다. 이 연구는 다양한 자연적 시뮬레이션 및 고유 상태의 진화에 대한 시각적 분석을 통해 HM-RSNN의 계산적 이점을 강조합니다. HM-RSNN의 작동을 위해 세 가지 흥미로운 내재 흥분성 특성, 즉 수상돌기 시간 상수, 체세포 시간 상수 및 발화 임계값을 활용합니다.

- **Performance Highlights**: 실험 결과, HM-RSNN은 기존 RSNN 및 RNN 모델에 비해 모든 과제에서 유의미한 성능 향상을 보였습니다. 특히 HM-RSNN은 세 가지 새로운 인지 과제에서 더욱 우수한 결과를 보여주며, 작업의 갑작스러운 변화에 더 잘 적응합니다. 해당 연구는 스키마 학습을 모델링할 수 있는 새로운 경로를 제시하며, HM-RSNN이 전통적인 RNN에 대한 유망한 대안임을 입증합니다.



### Rethinking Encoder-Decoder Flow Through Shared Structures (https://arxiv.org/abs/2501.14535)
- **What's New**: 본 연구는 기존의 디코더 아키텍처에 새로운 은행(bank) 구조를 도입하여 깊이 추정(dense prediction) 성능을 향상시키려는 목적을 가지고 있습니다. 각 디코딩 블록이 개별 중간 특성 맵만을 처리하는 대신, 이전 및 후속 블록의 정보도 활용할 수 있는 공유 텐서(shared tensor)를 운영하여, 더 많은 맥락(context)을 제공할 수 있도록 설계되었습니다. 이러한 시스템은 대규모 데이터셋에서 훈련하여 자연 및 합성 이미지에서의 깊이 추정 성능을 크게 개선하였습니다.

- **Technical Details**: 본 연구는 이미지 인코더 𝐄와 이미지 디코더 𝐃로 구성된 구조를 가지고 있습니다. 여기서 ViT(Vision Transformer) 인코더는 4개의 중간 특성 맵을 출력하고, 디코더는 각 블록이 해당 특성 맵과 은행 구조를 활용하여 깊이 맵 𝒪d를 생성하는 방식을 채택합니다. 특별히, 은행 구조는 모든 디코딩 블록에서 사용될 수 있는 공통의 정보를 제공하는데, 이를 통해 블록 디자인의 성능이 향상됩니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 GFLOPS나 파라미터 수의 미미한 증가에 비해 깊이 추정 정확도를 비례적으로 더 높일 수 있는 것으로 나타났습니다. 이 연구가 제시한 은행 구조는 특정 디코더나 깊이 추정 작업에 한정되지 않으며, 다양한 밀집 예측 작업(dense prediction tasks)에 적용 가능하다는 장점도 있습니다. 향후 연구는 이러한 구조를 다양한 아키텍처에 적용하여 그 효용성을 검증할 필요가 있습니다.



### PARASIDE: An Automatic Paranasal Sinus Segmentation and Structure Analysis Tool for MRI (https://arxiv.org/abs/2501.14514)
- **What's New**: 본 연구에서는 chronic rhinosinusitis (CRS)에 대한 자동 분할 도구인 PARASIDE를 소개합니다. PARASIDE는 T1 MRI에서 상악, 전두, 접형, 비갑개 뼈 구조의 공기 및 연부 조직의 볼륨을 자동으로 분할함으로써, 이전에 수동적으로만 관찰되었던 특성의 관계를 정량화 가능하게 합니다. 이 시스템은 총 16개의 구조체에 대한 자동 전두 비강 완전 분할을 최초로 수행하며, Lund-Mackay 점수와 같은 의학적으로 관련된 특징을 계산할 수 있도록 설계되었습니다.

- **Technical Details**: 연구 데이터는 독일 동북부의 SHIP-START 및 SHIP-TREND 코호트에서 수집되었습니다. 총 8,728명의 참가자 중, 성별, 연령, 병리의 균형을 맞춘 273명의 피험자를 랜덤으로 선별하여 분석하였습니다. 세분화 모델 훈련을 위해 nnUNet 알고리즘을 사용하였고, 총 5,000 에폭(epoch) 동안 훈련하였습니다. 이 과정에서 100개의 교육 샘플에 대해 전문가의 수작업 주석을 기반으로 세분화 마스크를 수정했습니다.

- **Performance Highlights**: 자동으로 세분화된 공기 볼륨의 주석은 우수한 성능을 보여주며, 연부 조직에 비해 일관되게 낮은 평균 강도를 나타냅니다. 설계된 시스템은 60명의 피험자를 대상으로 한 테스트 데이터에서 Dice similarity coefficient (DSC)와 average symmetric surface distance (ASSD) 등의 다양한 성능 평가 지표를 사용하여 성능을 검증하였습니다. 연구 결과는 CHRONIC RHINOSINUSITIS 진단 결과와의 비교를 통한 정량적 평가를 가능하게 하여, 미래의 개인 맞춤형 의료에 기여할 것으로 기대됩니다.



### ABPT: Amended Backpropagation through Time with Partially Differentiable Rewards (https://arxiv.org/abs/2501.14513)
- **What's New**: 이 논문에서는 기존의 gradient bias 문제를 해결하기 위해 Amended Backpropagation-through-Time (ABPT)라는 새로운 접근 방식을 제안합니다. ABPT는 0-step 반환과 N-step 반환을 결합하여 reward의 비가산성에 의해 발생하는 gradient bias를 완화함으로써 정책 학습의 효율성을 향상시킵니다. 또한, entropy 정규화 및 상태 초기화 메커니즘을 도입하여 훈련 중 탐색을 장려합니다.

- **Technical Details**: 강화학습의 목표는 주어진 상태에서 누적 보상을 극대화하는 확률적 정책을 찾는 것입니다. ABPT에서는 기존의 actor-critic 방법론을 사용하며, actor와 critic 모두 신경망에 의해 근사됩니다. 정책 경량화 과정에서 ABPT는 0-step 반환을 활용하여 가치 그래디언트를 균형 있게 조정하고, replay buffer를 통해 상태 경험을 저장하여 샘플링 효율성을 높입니다.

- **Performance Highlights**: 실험 결과는 ABPT가 기존의 학습 알고리즘보다 빠른 수렴 속도와 더 높은 최종 보상을 달성함을 보여줍니다. 특히, 비가산성 보상을 포함한 quadrotor 작업에서 ABPT의 우수성이 입증되었습니다. 또한, ABPT는 다양한 학습 속도와 보상 구조에서도 강인성을 보이며, 실제 quadrotor 작업에서 효과적인 학습 성과를 달성했습니다.



### Deep-BrownConrady: Prediction of Camera Calibration and Distortion Parameters Using Deep Learning and Synthetic Data (https://arxiv.org/abs/2501.14510)
- **What's New**: 이번 연구는 심층 학습 모델을 사용하여 단일 이미지에서 카메라 보정 및 왜곡 매개변수 예측의 과제를 해결합니다. 주요 기여로는 실제 및 합성 이미지 혼합으로 학습된 심층 학습 모델이 단일 이미지로부터 카메라 및 렌즈 매개변수를 정확히 예측할 수 있다는 점과 AILiveSim 시뮬레이션 플랫폼을 이용해 포괄적인 합성 데이터셋을 개발한 것입니다.

- **Technical Details**: 본 연구는 여러 방향의 보정 물체 이미지가 필요했던 전통적인 방법에 비해 단일 이미지에서 카메라 보정 및 렌즈 왜곡 매개변수를 예측하는 심층 학습 접근 방식을 도입합니다. AILiveSim 플랫폼을 통해 생성된 합성 데이터셋을 활용하여, 수평 시야각(H-FOV), 주점(ci,cysubscript𝑐𝑥subscript𝑐𝑦c_{x},c_{y}italic_c start_POSTSUBSCRIPT italic_x end_POSTSUBSCRIPT , italic_c start_POSTSUBSCRIPT italic_y end_POSTSUBSCRIPT), 및 왜곡 계수(k1,k2,k3,p1,p2subscript𝑘1subscript𝑘2subscript𝑘3subscript𝑝1subscript𝑝2k_{1},k_{2},k_{3},p_{1},p_{2}italic_k start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_k start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , italic_k start_POSTSUBSCRIPT 3 end_POSTSUBSCRIPT , italic_p start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_p start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT)의 예측을 가능케 합니다.

- **Performance Highlights**: 제안된 방법은 실험적으로 검증되어 다양한 센서 크기 및 타입에 걸쳐 일반화 능력을 보여줍니다. KITTI 데이터셋을 이용한 검증 결과, 이 모델은 합성 데이터에서 실제 시나리오로의 일반화 성능을 성공적으로 입증하였으며, 카메라 보정 매개변수 예측의 신뢰성을 입증했습니다. 이러한 연구는 카메라 매개변수를 추정하는 데 있어 합성 데이터 활용을 확대할 수 있는 미래 연구 방향을 제시합니다.



### A Note on Implementation Errors in Recent Adaptive Attacks Against Multi-Resolution Self-Ensembles (https://arxiv.org/abs/2501.14496)
Comments:
          4 pages, 2 figures, technical note addressing an issue in arXiv:2411.14834v1

- **What's New**: 이 논문은 최근 Zhang 외 (2024)가 다룬 multi-resolution self-ensemble defense (Fort 및 Lakshminarayanan, 2024)에 대한 adaptive attack의 구현 문제를 문서화하고 있습니다. 구현 과정에서 적대적 작용(perturbation)이 L₁₀ = 8/255의 표준 경계를 20배 이상 넘는 L₁₀ = 160/255에 도달하여, 방어 메커니즘의 성과를 근본적으로 왜곡했습니다. 올바른 경계 내에서 적대적 공격을 제어했을 때, 방어는 의미 있는 강건성을 유지했습니다.

- **Technical Details**: 근본적으로 문제는 반복(iteration) 간 perturbation의 누적(accumulation) 과정에서 발생했습니다. 각 반복(iteration)에서 개별 perturbation을 L₁₀ = 8/255로 제대로 제한했음에도 불구하고, 알고리즘은 수정된 이미지를 새로운 기준으로 삼아 추가적인 perturbation을 쌓아 올려 경계를 초과하게 되었습니다. 이로 인해, 최대 20회의 반복에서도 예상보다 뛰어난 L₁₀ 값이 발생했습니다.

- **Performance Highlights**: 적절한 경계 내에서 adaptive attacks를 수행한 결과, CIFAR-100에서 L₁₀ 공격 하에 20% 이상으로 초기 적대적 정확도를 기록했습니다. 본 연구는 L₁₀ = 8/255 perturbations가 이미지의 인지된 클래스를 변화시킬 수 있음을 보여주며, 기존의 연구 가정을 도전했습니다. 또한, 인간 인지와 모델의 예측이 유사하게 영향을 미칠 수 있다는 흥미로운 사실도 발견했습니다.



### RealCritic: Towards Effectiveness-Driven Evaluation of Language Model Critiques (https://arxiv.org/abs/2501.14492)
- **What's New**: 본 연구는 기존의 평가 방식을 개선하기 위해 LLM의 비판 능력을 평가하는 새로운 기준을 도입합니다. 기존 벤치마크가 오픈 루프(open-loop) 방식이었다면, 우리는 피드백을 통해 수정되는 폐쇄 루프(closed-loop) 방식을 채택하여 LLM의 비판 품질을 더 신뢰성 있게 평가합니다. 이 새로운 기준은 자기 비판(self-critique), 상호 비판(cross-critique), 반복 비판(iterative critique) 기능을 포함하여 고급 추론 모델의 능력을 구분하는 데 필수적입니다.

- **Technical Details**: 평가 프레임워크는 여러 주요 개념으로 구성되어 있습니다. 폐쇄 루프 방법론을 통해 LLM의 비판 능력을 테스트하게 되며, 비판이 적용된 후 생성된 솔루션의 정확도를 통해 비판의 품질을 평가합니다. 또한 두 가지 주요 축을 통해 자기 비판과 상호 비판의 차이를 구분하고, 단일 라운드뿐만 아니라 반복적인 비판 과정을 고려하여 모델의 장기적인 추론 능력을 평가합니다.

- **Performance Highlights**: 실험 결과, 고전적인 LLM은 고급 추론 기반 모델인 o1-mini에 비해 모든 비판 시나리오에서 눈에 띄게 뒤처지는 것으로 나타났습니다. 또한 자기 비판 및 반복 비판 설정에서 고전 LLM이 기준 성능에 비해 더 낮은 성능을 보이는 경우도 발견되었습니다. 이러한 결과는 우리의 벤치마크가 LLM의 개선 방향을 제시할 수 있는 중요한 자원이 될 것이라는 점을 강조합니다.



### The Pseudo-Dimension of Contracts (https://arxiv.org/abs/2501.14474)
- **What's New**: 본 논문은 알고리즘적 계약 설계를 다루며, 에이전트의 유형이 알려지지 않은 분포에서 샘플링되는 설정을 중점적으로 분석합니다. 저자들은 샘플 에이전트 유형으로부터 근사 최적 계약(near-optimal contract)을 학습하기 위한 오프라인 학습 프레임워크를 제안합니다. 특히, 통계적 학습 이론의 'pseudo-dimension' 개념을 도구로 활용하여 계약 설계에서의 단순성(simplicity)과 최적성(optimality) 간의 균형을 새로운 시각으로 접근합니다. 이를 통해 저자들은 다양한 계약 유형에 대한 샘플 복잡성을 보장하는 알고리즘을 제시합니다.

- **Technical Details**: 이 논문은 선형 계약(linear contract), 유계 계약(bounded contract), 비유계 계약(unbounded contract) 세 가지 계약 유형을 분석합니다. 각 계약 클래스에 대해 pseudo-dimension과 표현 오류(representation error) 사이의 상충 관계를 규명하고 샘플 복잡성(sample complexity) 보장을 제공합니다. 특히, 선형 계약의 pseudo-dimension은 Θ(log n)으로 나타내며, 계약 유형으로부터의 에이전트 샘플을 통한 학습 방법을 제안합니다. 또한, 제안된 알고리즘은 오프라인 학습 설정에서 효율성을 보장하는 데 주력합니다.

- **Performance Highlights**: 제안된 방법론은 높은 확률로 근사 최적 계약을 학습하는 데 필요한 샘플 수를 최소화합니다. 특히 선형 및 유계 계약에 대해, 제안된 모델은 이론적으로 최적의 거래를 제공합니다. 반면, 비유계 계약에 대해서는 어떤 학습 알고리즘도 존재하지 않음을 보여주는 불가능 결과(impossibility result)를 제공합니다. 마지막으로, 제안된 알고리즘은 온라인 학습 셋팅으로 확장 가능하며, 이는 데이터 기반의 알고리즘 설계에 대한 새로운 통찰을 제공합니다.



### Data-efficient Performance Modeling via Pre-training (https://arxiv.org/abs/2501.14438)
- **What's New**: 이 논문은 코드 최적화를 위한 자동화와 관련된 최신 연구를 소개하며, autoencoders를 이용한 자기 지도(pre-training) 방식을 제안합니다. 이 방식은 레이블이 있는 데이터에 대한 필요성을 줄이며, 성능 모델의 정확도를 개선합니다. 데이터를 다룰 때 과거에는 약 1800만 데이터 포인트가 필요했지만, proposed method를 사용함으로써 이 숫자를 360만으로 줄일 수 있었습니다.

- **Technical Details**: 제안된 방식은 autoencoder를 이용하여 프로그램과 변환의 표현을 학습하고, 이 표현을 성능 모델에 임베딩하여 훈련 데이터의 필요성을 크게 줄입니다. 데이터의 양이 감소하더라도, Tiramisu 모델은 높은 정확도를 유지할 수 있음을 보여줍니다. 이 과정에서 MAPE(Mean Absolute Percentage Error) 수치가 개선되며, 모델의 학습 속도도 향상됩니다.

- **Performance Highlights**: 논문에서 제시한 결과에 따르면, autoencoder 기반의 사전 훈련 방식을 사용했을 때, Tiramisu 모델은 360만 데이터 포인트로도 유사한 성능을 보이는 것이 확인되었습니다. 가장 놀라운 점은 45만 데이터 포인트로 MAPE가 29.69%에 달할 수 있음을 보여주었습니다. 이러한 결과는 기존 방식보다 데이터 요구량을 5배 줄일 수 있는 가능성을 제시합니다.



### Remining Hard Negatives for Generative Pseudo Labeled Domain Adaptation (https://arxiv.org/abs/2501.14434)
- **What's New**: 이 논문은 Dense retrievers가 다소 제한적인 도메인 적응 방법을 개선하는 새로운 방법을 제안합니다. Generative Pseudo Labeling (GPL) 기법을 사용하여 모델의 지식을 전이하고, 하드 네거티브를 갱신함으로써 더 나은 하드 네거티브의 채굴을 가능하게 합니다. 이로 인해, 본 연구에서는 LoTTE 및 BEIR 데이터세트에서 성능이 저조한 기존 접근법을 개선할 수 있음을 입증합니다.

- **Technical Details**: Dense passage retrieval은 현대 NLP 파이프라인에서 중요한 역할을 하는 기법으로, 쿼리와 문서를 저차원 벡터(embeddings)로 표현하여 유사성 비교를 통해 관련성을 측정합니다. 논문에서는 하드 네거티브 마이닝(hard-negative mining)을 통해 생성된 쿼리와 긍정적인 쌍을형성하여 성능을 향상시키기 위한 방법론적 근거를 제시합니다. 이 연구에서는 R-GPL 접근법을 통해 도메인 적응 모델이 훈련하는 동안 더 관련성 높은 하드 네거티브를 사용할 수 있음을 입증합니다.

- **Performance Highlights**: 저자들은 14개의 BEIR 데이터셋 중 13개와 12개의 LoTTe 데이터셋 중 9개에서 도메인 적응 모델의 랭킹 성능이 향상된 것을 보여줍니다. 이 연구는 처음으로 LoTTE 벤치마크에서 GPL을 평가하고, 도메인 적응 중 하드 네거티브를 지속적으로 갱신함으로써 성능의 개선을 이루었습니다. 이러한 결과는 도메인 적응이 훈련 데이터를 보다 효율적으로 활용할 수 있는 가능성을 보여줍니다.



### Domaino1s: Guiding LLM Reasoning for Explainable Answers in High-Stakes Domains (https://arxiv.org/abs/2501.14431)
- **What's New**: 이 연구에서는 Domaino1s라는 새로운 모델을 소개하여 고위험 도메인 작업에서 대형 언어 모델의 (LLMs) 사고(procesing) 기능을 크게 향상시켰습니다. 이 모델은 감독된 미세 조정(supervised fine-tuning)과 트리 탐색(tree search)을 통해 설명가능한 답변을 생성하는 데 중점을 두었습니다. CoT-stock-2k 및 CoT-legal-2k 데이터셋을 구성하여 도메인 특화 사고 단계를 활성화하는 모델을 미세 조정하였으며, 설명 및 결정의 신뢰성을 높이는 새로운 평가 지표인 PROOF-Score를 제안했습니다.

- **Technical Details**: Domaino1s는 두 가지 모델 변형인 Domaino1s-finance와 Domaino1s-legal로 구성되어 있습니다. 이 모델은 GPT-4o를 사용하여 CoT 데이터를 생성하고, 26개의 특수 토큰을 활용하여 사고 과정의 각 단계를 명확히 구분할 수 있도록 데이터셋을 구성합니다. 새로운 Selective Tree Exploration 방식을 도입하여 최적의 사고 경로를 탐색하고, 각 단계의 평균 perplexity를 사용하여 새로운 경로를 탐색할지를 결정합니다.

- **Performance Highlights**: Domaino1s의 실험 결과는 주식 투자 추천 및 법적 질문 응답 과제에서 두드러진 성능을 나타내며, 설명 가능성과 사고의 정확성을 동시에 향상시켰습니다. 기존의 정확성 기준 외에도, DOMAINO1S는 모델의 설명 가능성을 평가하는 새로운 시각을 제시하며, 이는 결정적인 법적 또는 윤리적 위험을 줄이는 데 기여할 수 있습니다. 이 연구는 고위험 도메인 작업을 해결하기 위한 LLM의 실행 가능성과 효율성을 입증했습니다.



### Statistical Verification of Linear Classifiers (https://arxiv.org/abs/2501.14430)
Comments:
          16 pages, 3 figures

- **What's New**: 이 논문에서는 두 샘플 간의 선형 분리 가능성과 관련된 동질성 검정을 제안합니다. 이 검정을 통해 선형 분류기가 단순히 '무작위'인지, 아니면 두 클래스 간의 차이를 효과적으로 포착하는지를 판단할 수 있습니다. 특히, 실험을 통해 정규 분포 샘플에 대한 테스트의 p-값에 대한 상한이 매우 정확함을 입증하며, 유전자 쌍 발현 기반으로 ER-양성 유방암 재발을 탐지하는 분류기를 평가합니다.

- **Technical Details**: 이 연구에서는 두 차원 샘플에 적용할 때 동질성 검정의 p-값에 대한 상한을 설정하는 데 중점을 둡니다. 유전자 발현 데이터의 분석을 통한 분류기 구성 과정에서, 각각의 유전자 쌍이 분류의 정확도를 어떻게 향상시키는지에 대한 논의도 포함됩니다. 또한, 여러 번의 테스트 문제에도 불구하고 실험적으로 확인된 동질성 검정의 통계적 유의성을 보여줍니다.

- **Performance Highlights**: IGFBP6 및 ELOVL5 유전자가 ER-양성 유방암 재발 탐지 과정에서 중요한 역할을 함을 확인하였습니다. 이 연구는 실질적으로 작은 테스트 세트에서도 통계적 유의성을 확인할 수 있음을 보여주며, 이는 의료 데이터 분석에서 매우 중요한 결과입니다. 마지막으로, 실제 시나리오에서 통계적 가설 검정을 통해 분류기의 성능을 평가할 수 있는 방법론을 제안합니다.



### Adaptive Rank Allocation for Federated Parameter-Efficient Fine-Tuning of Language Models (https://arxiv.org/abs/2501.14406)
- **What's New**: 이 논문은 Federated Parameter-Efficient Fine-Tuning (FedPEFT)의 한계를 극복하기 위해 FedARA라는 새로운 방법을 제안합니다. FedARA는 비독립 및 동일 분포(Non-IID) 데이터로 인한 성능 저하와 고정 파라미터 구성으로 인한 통신 비효율성을 개선하기 위해 설계되었습니다. 이 방법은 트렁케이티드 싱귤러 벨류 분해(Truncated SVD), 동적 랭크 할당(Dynamic Rank Allocation), 랭크 기반 모듈 프루닝(Rank-based Module Pruning)을 결합하여 언어 모델의 파라미터 효율성 있는 미세조정을 지원합니다.

- **Technical Details**: FedARA는 비독립적이고 비동일 분포의 데이터가 미치는 영향을 줄이기 위해 고안되었습니다. 이를 위해, FedARA는 트렁케이티드 싱귤러 벨류 분해 적응 방식을 도입하여 다양한 클라이언트 간 공통 특성을 학습하도록 지원합니다. 또한, 동적 랭크 할당을 통해 각 클라이언트에서 중요한 랭크를 동적으로 식별하며, 비효율적인 통신을 줄이기 위해 랭크 변화에 따라 비활성 모듈을 안전하게 제거합니다.

- **Performance Highlights**: FedARA는 여러 데이터셋에서 평균 8.49%의 정확도 향상을 보여주며, 비독립적 데이터 환경에서도 기존의 약한 기준선보다 평균 6.95% 우수함을 입증하였습니다. 실험 결과, AGX Orin, Orin Nano, Raspberry Pi 5와 같은 모바일 장치에서 총 훈련 시간이 최대 48.90%, 에너지 소비는 46.95%까지 감소하는 성과를 거두었습니다. 이로 인해 FedARA는 모바일 장치에서의 학습 효율성과 에너지 효율성을 크게 개선함을 확인하였습니다.



### Handling Heterophily in Recommender Systems with Wavelet Hypergraph Diffusion (https://arxiv.org/abs/2501.14399)
- **What's New**: 본 논문에서는 하이퍼그래프 기반 추천 시스템의 진화를 위한 새로운 FWHDNN(퓨전 기반 웨이브릿 하이퍼그래프 확산 신경망) 프레임워크를 소개합니다. 이 모델은 이질적(hheterophily) 패턴과 다차원 사용자-아이템 상호작용을 포착하는 데 필요한 주요 세 가지 구성 요소를 포함합니다. 제안하는 방법은 다양한 클래스 레이블에 적응할 수 있는 메시지 패싱을 지원하는 이질성 인식 하이퍼그래프 확산을 활용합니다.

- **Technical Details**: FWHDNN은 세 가지 주요 구성 요소로 이루어져 있습니다. 첫 번째는 Cross-Difference Relation Encoder로, 다양한 클래스 레이블에 적응하는 메시지 패싱을 통해 이질적 패턴을 모델링합니다. 두 번째는 Multi-level Cluster-wise Encoder로, 웨이브릿 변환을 활용하여 다중 스케일의 구조적 관계를 포착합니다. 마지막으로, 통합된 다중 모달 융합 메커니즘을 통해 구조적 및 텍스트 정보를 결합합니다.

- **Performance Highlights**: FWHDNN은 실제 데이터셋에서 광범위한 실험을 통해 기존의 최첨단 방법들을 초월하는 정확성과 강인성 및 확장성을 보여줍니다. 특히, 사용자와 아이템 간의 고차원 연결 관계를 효과적으로 포착하는 데 강점을 지니고 있습니다. 이러한 성능은 FWHDNN이 복잡한 추천 환경을 효과적으로 처리할 수 있는 가능성을 시사합니다.



### DRESSing Up LLM: Efficient Stylized Question-Answering via Style Subspace Editing (https://arxiv.org/abs/2501.14371)
Comments:
          ICLR 2025 Accepted

- **What's New**: DRESS는 LLM의 반응을 스타일화하기 위한 새롭고 효과적인 접근법으로, 표현 편집을 통해 스타일 관련 하위공간(style subspace)을 분리합니다. 기존의 방법인 프롬프트(prompting)나 미세 조정(fine-tuning)은 복잡한 스타일 적응을 위해 충분하지 않거나 계산적으로 비용이 많이 듭니다. DRESS는 LLM의 오버 파라미터화된 특성을 활용하여 표현 공간에서 의미(original semantics)에 최소한의 영향을 미치면서 스타일을 조정할 수 있는 방법을 제안합니다.

- **Technical Details**: DRESS는 스타일 관련 하위공간을 단계적으로 식별하고 그들 안에서 의미적으로 격리된 스타일 조정을 수행하는 세 가지 전략을 포함합니다. 첫 번째는 주의 헤드 여과(attention head filtering)로, 스타일과 관련된 주의 헤드를 식별합니다. 두 번째는 스타일 하위공간 여과로(style subspace filtering), 선택된 주의 헤드 내에서 스타일과 관련없는 컴포넌트를 제거합니다. 마지막으로 각 하위공간 기초와 생성된 토큰에 대해 적응형 편집 강도(adaptive editing strength)를 적용하여 유연성을 제공합니다.

- **Performance Highlights**: DRESS는 스타일 강도(style intensity), 의미 보존(semantic preservation), 유창성과 같은 객관적인 평가 지표를 포함하는 두 개의 스타일화된 QA 벤치마크 데이터셋을 사용하여 그 효과를 입증합니다. 실험 결과, DRESS는 SFT, 프롬프트 및 다른 표현 편집 방법들과 비교하여 성능이 크게 향상된 것으로 나타났습니다. DRESS는 또한 대화형 에이전트를 개발하는 데 특히 유용하여, 경량화되고 교육이 필요 없는 솔루션을 제공합니다.



### Exploring the sustainable scaling of AI dilemma: A projective study of corporations' AI environmental impacts (https://arxiv.org/abs/2501.14334)
- **What's New**: 본 논문에서는 인공지능(AI)과 특히 대형 언어 모델(LLM)의 환경적 영향을 평가하기 위한 새로운 방법론을 제안합니다. AI의 기후 변화에 미치는 영향을 이해하려는 비즈니스에 유용한 통찰력을 제공하면서도, AI와 생애 주기 평가(LCA)에 대한 전문 지식이 필요하지 않습니다.

- **Technical Details**: 연구 결과, 대규모 생성형 AI 모델이 기존 모델에 비해 최대 4600배 더 많은 에너지를 소비한다는 것이 확인되었습니다. 우리의 모델링 접근법은 AI 사용의 증가, 하드웨어 컴퓨팅 효율, 그리고 IPCC 시나리오에 따른 전력 혼합의 변화를 반영하여 2030년까지의 AI 전력 사용을 예측합니다.

- **Performance Highlights**: 높은 채택 시나리오 하에서, 복잡한 생성형 AI 및 에이전트 채택 덕분에 AI 전력 사용은 기하급수적으로 증가할 것으로 예상됩니다. 2030년까지 생성형 AI의 환경적 영향을 줄이기 위해서는 AI 가치 사슬 전반에 걸친 협력이 필수적이며, 단독의 하드웨어 효율성 개선이나 모델 효율성만으로는 부족함을 강조합니다.



### Automatic detection and prediction of nAMD activity change in retinal OCT using Siamese networks and Wasserstein Distance for ordinality (https://arxiv.org/abs/2501.14323)
Comments:
          Solution to the MICCAI 2024 MARIO Challange. First 3 authors contributed equally. Models can be found at this https URL

- **What's New**: 이번 연구는 딥러닝 모델을 사용하여 신생혈관성 노화 관련 황반변성(nAMD)의 진행을 예측하는 새로운 접근법을 제안합니다. 특히, 제안된 모델은 양안(Siamese) 네트워크와 Wasserstein Distance 기반의 손실 함수를 활용하여 증세 변화 예측에서 더 나은 성능을 발휘할 수 있도록 설계되었습니다. 이 연구는 OCT(Optical Coherence Tomography) 이미지를 활용하여 환자의 상태 변화를 탐지하고 예측하는 최적의 솔루션으로 간주됩니다.

- **Technical Details**: 제안된 모델은 SiamRETFound라는 이름의 시아미즈(Siamese) 신경망으로, 두 방문 간의 B-scan 이미지를 비교하여 변화를 감지합니다. 이를 위해 Visio Transformer(ViT) 기반의 구조를 사용하며, 조정된 분류 헤드를 통해 임상적으로 중요 변화의 평가를 수행합니다. 또한, 훈련 데이터는 MARIO 챌린지의 두 가지 작업을 기반으로 구성되며, 전처리 단계에서는 Kermany 데이터셋을 통해 자체적으로 모델을 사전 훈련했습니다.

- **Performance Highlights**: 제안된 모델은 MARIO 챌린지의 초기 리더보드에서 높은 순위를 기록했습니다. 향후 nAMD 치료 관리에 도움을 줄 수 있는 예측 능력을 보여주었으며, 특히 시간에 따른 변화 예측의 중요성이 강조되었습니다. 이 연구는 OCT 이미지를 기반으로 한 임상 결정 지원 시스템의 자동화를 위한 중요한 기반을 제공할 것으로 기대됩니다.



### Locality-aware Fair Scheduling in LLM Serving (https://arxiv.org/abs/2501.14312)
- **What's New**: 이 논문은 대형 언어 모델(LLM) 서비스에 대해 최초의 지역 인식 공정 스케줄링 알고리즘인 Deficit Longest Prefix Match (DLPM)를 제안합니다. DLPM은 접두사 지역성을 보장하면서도 공정성을 유지하는 특징을 갖고 있으며, 이를 통해 기존의 Virtual Token Counter (VTC)보다 성능을 크게 향상시킬 수 있습니다. 또한, DLPM을 기반으로 한 새로운 분산 스케줄링 알고리즘인 Double Deficit LPM (D$^2$LPM)도 소개되어 공정성, 지역성, 부하 분산 간의 균형을 찾는 데 기여합니다.

- **Technical Details**: 이 논문에서는 LLM 추론의 효율성을 높이기 위해 접두사를 공유하는 방법을 탐구합니다. DLPM 알고리즘은 요청의 순서를 완화하여 접두사 지역성을 최대한 보존하는 동시에 공정성을 보장합니다. 또한, D$^2$LPM 알고리즘은 분산 환경에서 높은 접두사 지역성을 유지하도록 설계되었습니다. 이들 알고리즘은 요청 처리의 효율성을 향상시키고 GPU 메모리에서의 중복 계산을 줄이는 데 목표를 두고 있습니다.

- **Performance Highlights**: 저자들의 실험 결과 DLPM과 D$^2$LPM은 기존의 VTC보다 최대 2.87배 높은 처리량과 7.18배 낮은 클라이언트당 지연 시간을 기록하여 성능이 우수함을 입증했습니다. 이러한 결과는 공정성을 유지하며 높은 시스템 처리량을 달성하는 데 기여합니다. 따라서 이 논문은 공정성과 효율성을 동시에 달성하기 위한 새로운 접근 방식을 제시하고 있습니다.



### Fast Think-on-Graph: Wider, Deeper and Faster Reasoning of Large Language Model on Knowledge Graph (https://arxiv.org/abs/2501.14300)
- **What's New**: 이번 논문에서는 Fast Think-on-Graph (FastToG)라는 혁신적인 패러다임을 제안합니다. FastToG는 그래프 정보를 활용하여 LLM이 '커뮤니티 단위'로 사고하도록 고안되었습니다. 이 접근법은 Community Detection 기법을 사용하여 더 깊은 상관관계를 포착하고, 코스와 파인의 두 단계의 커뮤니티 프루닝을 통해 빠른 정보 검색을 가능하게 합니다.

- **Technical Details**: FastToG는 그래프 내에서 커뮤니티 감지를 통해 추론 체인을 구축하고 Community-to-Text 기법을 활용하여 커뮤니티 구조를 텍스트 형식으로 변환합니다. Local Community Search (LCS)라는 새로운 기법을 도입하여 그래프의 지역 범위에서 커뮤니티를 탐지하고, 효율성을 높이기 위해 모듈성 기반의 코스 프루닝과 LLM 기반의 파인 프루닝을 적용합니다.

- **Performance Highlights**: 실험 결과, FastToG는 이전 방법들에 비해 더 높은 정확도, 더 빠른 추론, 그리고 더 높은 설명 가능성을 보여주었습니다. 커뮤니티 기반의 추론이 추론 체인을 단축시켜 LLM 호출 횟수를 줄일 수 있음을 확인했습니다. 또한 FastToG는 LLM에 대한 검색 과정을 단순화하고 사용자에게 더 나은 설명 가능성을 제공합니다.



### Snapshot multi-spectral imaging through defocusing and a Fourier imager network (https://arxiv.org/abs/2501.14287)
Comments:
          22 Pages, 7 Figures

- **What's New**: 이번 연구에서는 추가적인 스펙트럼 필터나 맞춤형 부품 없이 표준 흑백 이미지 센서를 이용한 스냅샷 다채널 이미징 방법을 소개합니다. 이 방법은 장의 고유한 색수차(chromatic aberration)를 활용하여 다채널 정보의 물리적 인코딩을 자연스럽게 생성합니다. 이를 통해 기존의 복잡한 시스템 없이도 고품질의 다채널 이미지를 획득할 수 있습니다.

- **Technical Details**: 이 시스템은 딥 러닝 기반의 다채널 푸리에 이미저 네트워크(multi-spectral Fourier Imager Network, mFIN)를 통해 인코딩된 이미지 정보를 신속하게 디코드합니다. 실험에서는 여섯 개의 조명 채널을 사용하여 입력 조명 채널 예측에 92.98%의 정확도를 달성했으며, 다양한 테스트 객체에서 강력한 다채널 이미지 재구성을 보여주었습니다.

- **Performance Highlights**: 딥 러닝으로 강화된 이 프레임워크는 흑백 이미지 센서를 사용한 스냅샷 이미지 캡처를 통해 고품질 다채널 이미지 재구성을 가능하게 합니다. 이를 통해 생명과학(biomedicine), 산업 품질 관리(industrial quality control), 농업 등 다양한 분야에서 응용 가능성을 크게 넓혔습니다.



### Leveraging Online Olympiad-Level Math Problems for LLMs Training and Contamination-Resistant Evaluation (https://arxiv.org/abs/2501.14275)
- **What's New**: 이번 논문에서는 올림피아드(olympiad) 수준의 수학 문제 해결을 위한 고품질 데이터셋 구축과 LLM(대형 언어 모델)의 성능 향상을 위한 자동화된 파이프라인을 소개합니다. 특히, AoPS(Art of Problem Solving) 포럼의 자원을 활용하여 60만 개 이상의 QA(pair) 쌍을 포함하는 AoPS-Instruct 데이터셋을 개발했습니다. 이 외에도, 새로운 평가 세트인 LiveAoPSBench를 통해 LLM의 수학 추론 능력을 평가할 수 있는 오염 저항적인 벤치마크를 구축했습니다.

- **Technical Details**: 이 연구에서는 LLM의 미세 조정을 위한 AoPS-Instruct 데이터셋과 더불어 LiveAoPSBench라는 자동화된 평가 세트를 소개합니다. AoPS-Instruct는 AoPS 포럼의 게시글에서 QA 쌍을 추출하여 생성되었으며, 이는 고급 수학 문제에 적합한 대규모 데이터셋입니다. 자동화된 파이프라인을 통해 지속적으로 최신 문제를 포함한 평가 세트를 운영함으로써 오염 가능성을 줄였습니다.

- **Performance Highlights**: 실험 결과, LLM을 AoPS-Instruct로 미세 조정한 후, OlympiadBench, Omni-Math, LiveAoPSBench와 같은 다양한 벤치마크에서 성능이 향상되었습니다. 또한, 시간 경과에 따른 성능 저하가 관찰되어, 오래된 문제에 대한 성공이 실제 수학적 추론 능력이 아니라 사전 훈련 피드백에 기인할 수 있음을 시사합니다. 따라서 본 연구는 LLM의 수학적 추론 능력에 대한 통찰력을 제공하며, 이를 통해 데이터셋의 생성과 유지 관리 방법론의 중요성을 강조하고 있습니다.



### Distributionally Robust Coreset Selection under Covariate Shif (https://arxiv.org/abs/2501.14253)
- **What's New**: 이번 연구에서는 Distributionally Robust Coreset Selection (DRCS) 방법을 제안합니다. DRCS는 미래의 공변량 분포(covariate distribution)가 훈련 분포와 정의된 범위 내에서 벗어날 수 있다고 가정하여 최악의 테스트 오류(worst-case test error) 상한을 이론적으로 유도합니다. 이를 통해 DRCS는 배포 환경의 불확실성과 변동성에도 불구하고 효율적인 훈련 데이터 선택을 가능하게 합니다.

- **Technical Details**: 이 연구에서는 covariate shift 조건 하에서의 분포적으로 강건한 설정(distributionally robust setting)에 집중합니다. DRCS 방법은 가장 악화된 테스트 분포 하에서의 테스트 오류를 최소화하는 코어셋 선택을 목표로 합니다. 또한, 이론적으로 유도된 오류 상한을 기반으로 효율적인 알고리즘을 제안하여, 선택된 데이터 서브셋이 향후 테스트 분포 변화에 잘 적응할 수 있도록 합니다.

- **Performance Highlights**: DRCS 방법의 효과는 실험을 통해 검증됩니다. 특히, 이 방법은 대규모 데이터셋이나 자원 제약 환경에서도 예측 모델의 성능을 유지할 수 있도록 설계되었습니다. 연구 결과, DRCS는 기존의 방법들에 비해 불확실한 미래 환경에서도 높은 성능을 발휘함을 보여줍니다.



### Adaptive Progressive Attention Graph Neural Network for EEG Emotion Recognition (https://arxiv.org/abs/2501.14246)
- **What's New**: 이 논문에서는 인간의 감정을 효과적으로 인식하기 위해 Adaptive Progressive Attention Graph Neural Network (APAGNN)라는 새로운 방법을 제안합니다. APAGNN은 세 개의 전문 전문가를 통해 뇌 영역 간의 공간적 관계를 동적으로 분석합니다. 이러한 계층적 접근 방식은 정밀한 신경 활동 분석을 가능하게 하여 EEG 기반 감정 인식의 성능을 크게 향상시킵니다.

- **Technical Details**: APAGNN은 첫 번째 전문가가 전반적인 뇌 패턴을 캡처하고, 두 번째 전문가가 특정 영역의 특성을 분석하며, 세 번째 전문가가 감정 관련 채널을 조사하는 구조로 구성됩니다.(weights generator)가 모든 전문가의 출력을 통합하여 최종 예측 레이블을 생성하는 방식으로 구성되어 있습니다. 이론적 기초로 Graph Neural Networks (GNNs)와 주의 메커니즘이 포함되며, 비유클리드 공간을 처리하는 데 강점을 가지고 있습니다.

- **Performance Highlights**: 제안된 APAGNN 모델은 SEED, SEED-IV 및 MPED의 세 가지 공개 데이터셋에서 광범위한 실험을 수행하여 이전 방법에 비해 우수한 성과를 보여주었습니다. 이 모델은 다양한 개인마다 변동성이 큰 감정 관련 뇌 활동을 효과적으로 포착하는 데 뛰어난 결과를 나타내어, 감정 분류의 정확도를 높였습니다. 특히, 전문가 수와 주의 메커니즘이 성능에 미치는 영향을 분석하여 성능 향상을 달성했습니다.



### Point-LN: A Lightweight Framework for Efficient Point Cloud Classification Using Non-Parametric Positional Encoding (https://arxiv.org/abs/2501.14238)
Comments:
          This paper has been accepted for presentation at the 29th International Computer Conference, Computer Society of Iran (CSICC) 2025

- **What's New**: 이번 논문에서는 3D 포인트 클라우드 분류를 위해 설계된 경량화된 새로운 프레임워크인 Point-LN을 소개합니다. Point-LN은 Farthest Point Sampling (FPS), k-Nearest Neighbors (k-NN)와 같은 비모수적(non-parametric) 컴포넌트를 통합하여 학습 가능한 분류기와 연결하여 분류 정확도를 높였으며, 최소한의 파라미터로 컴퓨팅 비용을 줄입니다. 이 하이브리드 아키텍처는 실시간 및 리소스 제약이 있는 애플리케이션에 최적화되어 있습니다.

- **Technical Details**: Point-LN은 특징 인코더(feature encoder)와 분류기(classifier)라는 두 가지 주요 구성 요소로 이루어져 있습니다. 특징 인코더는 원시 입력 포인트 클라우드를 고차원 특징 표현으로 변환하며, 분류기는 이러한 인코딩된 특징을 목표 레이블 공간에 매핑하여 분류 로짓을 산출합니다. 논문에서는 일반적으로 사용되는 비모수적 포지셔널 인코딩을 경량화된 방법에서의 응용도 다룹니다.

- **Performance Highlights**: Point-LN은 ModelNet40 및 ScanObjectNN과 같은 벤치마크 데이터셋에서 경쟁력 있는 성능을 보여줍니다. 비모수적 방법의 강점을 활용하면서 학습 가능한 분류기를 통합하여 복잡한 시나리오에서도 높은 정확도를 달성할 수 있도록 설계되었습니다. 이 연구는 다양한 포인트 클라우드 분류 작업에 적합하고, 광범위한 컴퓨터 비전 애플리케이션에서의 채택 가능성을 강조합니다.



### GreedyPixel: Fine-Grained Black-Box Adversarial Attack Via Greedy Algorithm (https://arxiv.org/abs/2501.14230)
- **What's New**: GreedyPixel은 타겟 모델의 쿼리 기반 피드백만을 사용하여 고품질의 적대적 예제를 생성하는 새로운 픽셀 기반 탐욕 알고리즘입니다. 이 알고리즘은 그라디언트 정보 없이도 백박스 환경에서 성공적인 공격을 수행할 수 있도록 설계되었습니다. GreedyPixel은 기존의 접근법들과 비교해 공격의 성공률, 시간 효율성 및 시각적 품질에서 우수한 성능을 발휘합니다.

- **Technical Details**: GreedyPixel 알고리즘은 서라겟 모델을 통해 얻은 그라디언트를 기반으로 픽셀의 우선순위를 정하고, 각 픽셀을 순차적으로 개별적으로 변형하여 공격을 최적화합니다. 이 과정은 좀 더 눈에 띄지 않는 변형을 초래하고, 기존 방법론보다 더 적은 계산 비용으로 공격을 성공적으로 수행합니다. 또한 이 알고리즘은 다양한 이미지 해상도에서 평가되어, 낮은 해상도에서 특히 효과적입니다.

- **Performance Highlights**: GreedyPixel은 CIFAR-10 및 ImageNet 데이터셋에서 기존의 백박스 및 화이트박스 공격 방법들과 비교한 결과, 공격 성공률(Attack Success Rate, ASR)이 동등한 수준에 이르렀습니다. 또한, 기존의 적대적 방어에 대한 평가에서도 더 높은 ASR을 기록하는 등 공격 효과성이 뛰어난 것으로 나타났습니다. 최종적으로, GreedyPixel은 적대적 공격 방법론의 새로운 경지를 열어가는 기여를 하였습니다.



### Top Ten Challenges Towards Agentic Neural Graph Databases (https://arxiv.org/abs/2501.14224)
Comments:
          12 Pages

- **What's New**: 이 논문은 기존 Neural Graph Databases (NGDBs)의 한계를 극복하기 위해 Agentic Neural Graph Databases (Agentic NGDBs)라는 새로운 개념을 제안합니다. Agentic NGDBs는 자율적인 쿼리 구성, 신경 쿼리 실행, 지속적인 학습과 같은 세 가지 핵심 기능을 통해 인공지능(AI) 시스템의 자율성과 적응력을 높입니다. 이러한 접근은 특히 데이터가 불완전하거나 노이즈가 많은 경우를 다룰 때 더욱 유용합니다.

- **Technical Details**: 논문에서는 Agentic NGDBs를 구현하기 위해 해결해야 할 열 가지 주요 도전 과제를 제시합니다. 이들은 의미 단위 표현(semantic unit representation), 유추적 추론(abductive reasoning), 확장 가능한 쿼리 실행(scalable query execution), 대규모 언어 모델(LLMs)과의 통합(integration with foundation models) 등을 포함합니다. 이러한 기술적 도전 과제를 극복함으로써, Agentic NGDB들은 지능적이고 스스로 개선하는 데이터 관리를 가능하게 합니다.

- **Performance Highlights**: Agentic NGDBs는 기존 NGDBs의 기능을 넘어 자율성과 능동적 학습의 특성을 도입하여 데이터 관리 분야에서 진일보한 성과를 보여줍니다. 특히 데이터의 복잡성과 대규모 처리를 다루는 데 있어 뛰어난 성능을 발휘하고, 사용자가 정의한 쿼리에 의존하지 않고도 의미 있는 분석을 할 수 있습니다. 이들은 고급 추론 및 의사결정 지원을 통해 현대 데이터 중심 응용 프로그램들에서 필수적인 역할을 할 것으로 기대됩니다.



### PuzzleGPT: Emulating Human Puzzle-Solving Ability for Time and Location Prediction (https://arxiv.org/abs/2501.14210)
Comments:
          NAACL 2025 Findings

- **What's New**: 이 논문에서는 이미지로부터 시간과 장소를 예측하는 복잡한 과제를 인간과 같은 퍼즐 해결 능력으로 형식화하고, 이것을 다양한 모듈들로 구현한 전문가 파이프라인인 PuzzleGPT를 제안합니다. PuzzleGPT는 시각적 단서를 식별하는 perceiver, 예측 후보를 추론하는 reasoner, 다양한 단서를 결합하는 combiner, 외부 지식을 검색하는 web retriever, 및 강건성을 위한 noise filter로 구성됩니다. 이는 최첨단 성능을 기록하며, 기존의 대형 VLM(Visual-Language Models)와 자동 생성된 추론 파이프라인에 비해 최소 32%에서 38%의 성능 향상을 보여줍니다.

- **Technical Details**: PuzzleGPT는 다섯 가지 핵심 기술인 perceiver, reasoner, combiner, noise filter, knowledge retriever로 구성됩니다. 이 중 perceiver는 시각 신호를 처리하고 개체를 식별하며, reasoner는 위치 및 시간 후보와 그들의 관계를 추론합니다. 여러 개체에서 얻은 단서를 효율적으로 결합하기 위해 신뢰도 기반의 계층적 조합 방법을 제안하여, 단순히 모든 단서를 결합하는 것이 아니라 각각의 단서를 점진적으로 분석하게 됩니다. 이러한 설계는 추론 과정을 신뢰할 수 있도록 하고 강건성을 추가합니다.

- **Performance Highlights**: PuzzleGPT는 TARA 데이터셋에서 기존의 VLM 모델들, 박사리 BLIP-2 및 GPT-4V를 포함하여 최소 32%의 성능 향상을 보이며, 이는 다양한 기술들을 동시에 활용할 수 없는 기존 모델들의 한계를 드러냅니다. 또한 PuzzleGPT는 WikiTilo 데이터셋에서도 최첨단 성능을 자랑하며, 이를 통해 복잡한 문제 해결을 위한 전문가 설계의 중요성을 강조합니다. 이 방법은 이미지에서 시간과 장소를 예측하는 인간의 퍼즐 해결 능력을 시뮬레이션할 수 있는 기초를 제공합니다.



### Distributed Multi-Agent Coordination Using Multi-Modal Foundation Models (https://arxiv.org/abs/2501.14189)
- **What's New**: 이번 연구에서는 시각 언어 지침에 기반한 배포 제약 최적화 문제(VL-DCOPs)를 제안합니다. 이는 자연어와 비주얼 지침을 통해 자동으로 제약 조건을 생성할 수 있는 대규모 다중 모달 시스템을 활용하여, 인간과 에이전트 간의 상호작용을 모델링합니다. 이러한 접근 방식은 기존 DCOPs의 한계를 극복하고 동적인 문제 해결이 가능하게 합니다.

- **Technical Details**: VL-DCOPs는 A1, A2, A3의 세 가지 에이전트 유형으로 구성됩니다. A1은 전통적인 알고리즘을 사용하면서도 LFM을 통해 인간 지침을 파싱하는 신경 상징 에이전트이며, A2는 불확실성을 해결하는 대화형 솔루션을 제공하는 보다 정교한 신경 상징 에이전트입니다. 마지막으로 A3는 지침에 따라 다양한 최적화 알고리즘을 시뮬레이션하는 완전 신경 에이전트입니다.

- **Performance Highlights**: VL-DCOPs를 평가하기 위한 세 가지 새로운 벤치마크가 설계되었습니다. 이들 벤치마크는 각각 고전적인 분산 그래프 색칠 문제와 실제 회의 일정 문제를 기반으로 하며, 최신 LLM 및 VLM을 사용하여 성능을 비교 분석합니다. 이 연구는 향후 DCOP literatures를 확장할 수 있는 흥미로운 연구 방향을 제시하고 있습니다.



### Dreamweaver: Learning Compositional World Representations from Pixels (https://arxiv.org/abs/2501.14174)
- **What's New**: 이번 연구에서는 Dreamweaver라는 신경망 아키텍처를 제안하여 원시 비디오에서 계층적이고 조합적인 표현을 발견하고 조합된 미래 시뮬레이션을 생성합니다. 이는 전통적으로 언어와 같은 보조 데이터 없이 비디오를 모델링하는 데 어려움이 있었던 AI 시스템의 한계를 극복하는 데 기여합니다. 특히, Recurrent Block-Slot Unit (RBSU)를 활용하여 비디오를 구성 요소 객체 및 속성으로 분해하고, 동적 개념을 효과적으로 캡처하기 위한 다중 미래 프레임 예측 목표를 사용합니다.

- **Technical Details**: Dreamweaver 모델은 T 개의 과거 이미지를 인코딩하는 RBSU를 포함합니다. 이 RBSU는 독립적으로 업데이트되는 슬롯 상태 집합으로 상태를 표현하며, 이는 나중에 혼합된 몬올리식 슬롯 상태를 독립적인 블록 조합으로 매핑하는 블록- 슬롯 병목을 거칩니다. 각 단계에서 개념 프로토타입 라이브러리에 대한 주의를 통해 블록 벡터가 값을 취득하도록 하여 동적 개념 추상화를 생성하는 데 중요한 예측 재구성 목표를 설정합니다.

- **Performance Highlights**: 실험 결과, Dreamweaver는 여러 데이터 세트를 통한 DCI 프레임워크에서 최신 오브젝트 중심 방법보다 우수한 성능을 보여주었습니다. 또한, RBSU의 모듈화된 개념 표현이 조합적 상상력을 가능하게 하여 서로 다른 객체에서 속성을 재조합하여 새로운 비디오를 생성하는 능력을 입증했습니다. 이 연구는 언어에 의존하지 않고 독창적인 비디오를 생성하는 길로 나아가는 중요한 첫걸음을 제시합니다.



### Learning to Price with Resource Constraints: From Full Information to Machine-Learned Prices (https://arxiv.org/abs/2501.14155)
Comments:
          28 pages, 4 figures

- **What's New**: 이 논문은 리소스 제약(resource constraints) 하에서 탐색(exploration)과 활용(exploitation) 간의 균형을 맞추는 동적 가격 책정 문제(dynamic pricing problem)를 연구합니다. 연구자들은 전량 정보를 가진 상황을 위한 Boundary Attracted Re-solve Method, 이전 정보가 없는 경우를 위한 온라인 학습 알고리즘, 그리고 알고리즘에서 추정 오차의 상한을 이용한 estimate-then-select 재구성 알고리즘의 세 가지 접근법을 소개합니다. 이 방법들은 각각 다른 정보 환경에서 동작하며, 리소스 제약이 있는 경우에도 극대화된 성능을 발휘합니다.

- **Technical Details**: 저자들은 n개의 제품과 m개의 자원, 그리고 T라는 한정된 시간을 고려하여 온라인 학습 문제를 구성합니다. 매 시간 단계 t에서, 자원의 남은 용량을 나타나는 𝒄t와 각 제품에 대한 가격을 결정하는 𝒑t와 같은 변수를 정의합니다. 결과적으로 이 논문에서 제안한 알고리즘은 가격과 수요 간의 미지의 관계를 학습하는 동시에 자원 소비를 고려하여 최적의 결정을 내리도록 설계되었습니다.

- **Performance Highlights**: 수치 실험(numerical experiments)은 다양한 시나리오에서 제안된 알고리즘의 효과성과 견고성을 검증합니다. Boundary Attracted Re-solve Method는 비선형 조건 없이 로그적 후회(logarithmic regret)를 달성하며, 온라인 학습 알고리즘은 최적의 O(√T) 후회를 달성합니다. 또한 estimate-then-select 방법은 신뢰할 수 있는 오프라인 데이터가 있을 때 향상된 후회 경계를 제공합니다.



### Effective Defect Detection Using Instance Segmentation for NDI (https://arxiv.org/abs/2501.14149)
Comments:
          6 pages, 2 figures, 2 tables. Published at AI2ASE 2025 workshop at AAAI2025. Accepted publication is available at this https URL

- **What's New**: 본 연구는 항공우주 제조업에서 사용되는 복합 패널의 초음파 스캔 이미지에서 결함을 식별하기 위해 인스턴스 세분화(instance segmentation) 기법을 적용하였다. Mask-RCNN과 YOLO 모델을 사용하여 결함을 탐지하고, 맞춤형 전처리 기법의 필요성을 줄이는 간단한 통계적 전처리 방법을 구현하였다. 이번 연구는 NDI(비파괴 검사) 프로세스에서 인스턴스 세분화 사용의 가능성과 효율성을 보여주며 데이터 전처리 시간과 검사 시간을 크게 단축시켰다.

- **Technical Details**: 본 연구에서는 초음파 검사가 사용되는 복합재료의 결함을 찾아내기 위해 Mask-RCNN(Detectron 2)과 YOLO 11을 기반으로 한 두 가지 모델을 활용하였다. 인스턴스 세분화 기법은 객체 경계 박스를 감지하고 이미지의 각 픽셀을 분류하여 결함을 탐지한다. 또한, 자동화된 NDI 프로세스를 지원하기 위해 초음파 스캔 이미지에 대한 간단한 전처리 방법을 사용하여 데이터의 복잡성을 완화하였다.

- **Performance Highlights**: 모델의 성능 비교 결과, 인스턴스 세분화를 통한 결함 탐지 방식이 전처리 시간과 검사 시간을 현저히 줄일 수 있음을 보여주었다. 이 연구를 통해 초음파 스캔 이미지에서의 결함 탐지에서 인스턴스 세분화 기법의 유용성이 입증되었으며, NDI 프로세스의 효율성을 높일 수 있는 가능성을 제시하였다. 연구 결과는 실제 항공우주 부품의 안전성을 확보하는 데 기여할 것으로 기대된다.



### EFiGP: Eigen-Fourier Physics-Informed Gaussian Process for Inference of Dynamic Systems (https://arxiv.org/abs/2501.14107)
- **What's New**: 본 논문은 데이터 기반의 동적 시스템에서 매개변수 추정(parameter estimation) 및 경로 재구성(trajectory reconstruction)을 위한 새로운 알고리즘인 Eigen-Fourier Physics-Informed Gaussian Process (EFiGP)를 제안합니다. 이 알고리즘은 Fourier transformation 및 eigen-decomposition을 통합하여 고전적인 수치적 통합(numerical integration)의 필요성을 없애면서 계산 효율성(computational efficiency) 및 정확성을 크게 향상시킵니다.

- **Technical Details**: EFiGP는 베이지안 프레임워크(bayesian framework) 위에 구축되며, ODE 시스템을 확률적 조건(probabilistic conditioning)을 통해 통합합니다. 이 과정에서 고주파 성분(high-frequency terms)을 절단하여 잡음 제거(denoising) 및 계산 비용 절감(computational savings)을 달성합니다. 또한, eigen-decomposition을 사용하여 Gaussian Process covariance 연산을 단순화함으로써, 조밀한 격자(dense-grid) 환경에서도 효율적으로 경로와 매개변수를 회복할 수 있도록 합니다.

- **Performance Highlights**: EFiGP는 세 가지 벤치마크 예제에 대한 실용적 유효성을 검증하였으며, 복잡한 동적 시스템에 대한 신뢰할 수 있고 해석 가능한 모델링을 위해 주요 도전 과제를 해결할 잠재력을 보여줍니다. 이 알고리즘은 매개변수 추정과 경로 복구 분야에서의 장점을 종합하여 기존 방법들이 직면한 한계를 극복하고 있습니다.



### MedSlice: Fine-Tuned Large Language Models for Secure Clinical Note Sectioning (https://arxiv.org/abs/2501.14105)
Comments:
          Our code is publicly available on github ( this https URL )

- **What's New**: 이 연구는 공개 소스의 큰 언어 모델(Large Language Models, LLMs)을 활용하여 임상 노트를 자동으로 구분하는 방법론을 제시합니다. 본 연구는 특히 현재 질병의 역사, 구간 역사 및 평가 및 계획의 세 가지 섹션에 중점을 두어, 487개의 진행 노트에 대한 세심한 데이터셋을 사용하여 모델을 개선하고 평가했습니다. 연구 결과, 미세 조정된 Llama 3.1 8B 모델이 GPT-4o를 초과하는 성능(F1=0.92)을 나타내며, 이는 병원과 임상 분야에서의 접근성과 비용 효율성을 크게 향상시킬 수 있는 가능성을 보여줍니다.

- **Technical Details**: 임상 노트는 전문의의 직간접적인 관찰과 평가를 기록하는 반면, 이러한 노트를 효과적으로 분류하는 것은 어려움이 많습니다. 연구진은 Clinical-Longformer와 같은 다양한 기계 학습 모델을 사용하여 모델 성능을 평가했으며, 임상 노트에서 특정 구간을 추출하기 위한 미세 조정된 LLM을 훈련했습니다. 최종 데이터셋은 1,147개의 노트로 구성되며, 미세 조정 과정에서는 rank-stabilized LoRA 방법이 사용되었습니다.

- **Performance Highlights**: 최적화된 Llama 3.1 8B 모델은 F1 점수 0.92를 기록하며, 낮은 비용의 소스 모델로서 현대의 의료 시스템에서 발전된 성과를 달성했습니다. 외적 유효성 테스트 세트에서도 F1 점수 0.85를 유지하며 높은 성능을 보여주었습니다. 이러한 결과는 임상 노트를 구조화하여 후속 분석을 수행하는 데 있어 큰 장점을 제공할 것입니다.



### Improved subsample-and-aggregate via the private modified winsorized mean (https://arxiv.org/abs/2501.14095)
Comments:
          40 pages, 2 figures

- **What's New**: 이번 논문에서는 univariate한 차별적으로 프라이빗(mean private) 평균 추정기인 private modified winsorized mean을 개발하였다. 이 추정기는 subsample-and-aggregate 알고리즘에서 사용하기 위해 설계되었으며, 기존 추정기들이 가지고 있는 여러 단점을 피할 수 있다. 연구에 따르면, 기존의 다양한 차별적으로 프라이빗한 다변량(mean multivariate) 평균 추정기는 대량의 관측치가 있는 dataset에서도 효과적이지 않음을 보여준다.

- **Technical Details**: private modified winsorized mean은 통계 추정에서의 bias와 sample size로 인한 문제를 해결하고, adversarial contamination 하에서도 성능이 우수함을 증명하였다. 이 추정기는 Lugosi와 Mendelson의 winsorized mean과 Durfee의 프라이빗 quantile estimation 방법을 결합한 것으로, 일반적인 분포의 평균을 추정할 수 있으며, 두 가지 모멘트만 존재하면 된다. 이를 통해 finite sample deviation bound와 함께 새로운 집계기의 유용성을 입증하였다.

- **Performance Highlights**: 이 연구는 다양한 시뮬레이션을 통해 private modified winsorized mean이 여러 경쟁 추정기에 비해 우수한 성능을 보임을 보여주었다. 연구 결과에 따르면, 다변량 평균 추정기들이 8000개의 관찰 데이터에서도 효과적으로 작동하지 않음을 알 수 있으며, 이는 본 논문의 주장을 뒷받침한다. 이러한 결과는 새로운 집계기가 subsample-and-aggregate에 적합하다는 중요한 통찰력을 제공한다.



### Communicating Activations Between Language Model Agents (https://arxiv.org/abs/2501.14082)
- **What's New**: 본 연구는 다중 언어 모델 간의 효율적인 통신을 위한 새로운 접근 방식을 제안합니다. 기존의 자연어 대신 활성화(activations)를 통해 모델 간 통신을 함으로써, 추론 비용을 대폭 절감하고 높은 정보 밀도를 유지할 수 있습니다. 이러한 대안적 언어를 통해 LLM(대형 언어 모델) 간의 재생성 능력을 극대화할 수 있음을 보입니다.

- **Technical Details**: 구체적으로, 모델 B의 계산을 중간 레이어에서 일시 정지하고, B의 활성화와 다른 모델 A의 활성화를 함수 f를 통해 결합합니다. 이후, 이 출력을 B의 다음 레이어로 전달하며 계산을 계속 진행합니다. 이 방식은 새로운 작업에서 제로 추가 파라미터로 LLM을 확장할 수 있도록 하며, 다양한 도메인과 환경에 적용할 수 있습니다.

- **Performance Highlights**: 실험을 통해 이 방법은 전통적인 자연어 통신 대비 최대 27.0%의 성능 향상을 이루었으며, 계산 비용은 1/4 미만으로 감소했습니다. 또한, 본 연구는 LLM의 다양한 크기와 스위트에 걸쳐 일반화 가능성을 보여주어, 소형 LLM도 통신의 이점을 활용할 수 있음을 입증했습니다.



### Expanding on the BRIAR Dataset: A Comprehensive Whole Body Biometric Recognition Resource at Extreme Distances and Real-World Scenarios (Collections 1-4) (https://arxiv.org/abs/2501.14070)
Comments:
          10 pages, 11 figures, 2 tables, submitted to CVPR

- **What's New**: 최근 생체 인식 알고리즘과 시스템의 발전은 많은 진전을 이루어냈지만, 극단적인 거리와 고도에서의 비전통적인 환경에서의 적용은 여전히 도전 과제입니다. 이 논문은 이러한 운영상의 도전에 대응하는 대규모 데이터세트의 확장을 요약하며, 그 구성과 데이터 수집, 관리 및 주석 달기 방법론을 설명합니다.

- **Technical Details**: BRIAR 프로그램은 어려운 조건에서 생체 인식 기술을 발전시키기 위해 고안된 미국 정부 지원의 프로그램으로, 기후와 환경이 다양한 3지역에서 1,760명의 피험자를 대상으로 475,000장의 이미지와 3,450시간의 영상을 포함하는 데이터세트를 구축했습니다. 이 데이터세트는 외부 및 실내 환경에서 다양한 활동을 수행하는 참가자들을 포함하며, 고해상도 이미지와 영상 데이터를 제공합니다.

- **Performance Highlights**: BRIAR 데이터세트는 생체 인식 및 열기 완화 관련 100편 이상의 논문에서 중요한 기준으로 사용되었으며, 연구자들이 해당 데이터에 접근할 수 있는 경로를 제공합니다. 이 연구는 향후 데이터 추가 및 품질 향상을 위한 지속적인 노력을 강조하고 있으며, 인종, 성별 등의 다양성을 고려한 데이터 확장을 통해 인식 모델의 공정성과 견고성을 보장하는 데 중점을 두고 있습니다.



### Revisiting CLIP: Efficient Alignment of 3D MRI and Tabular Data using Domain-Specific Foundation Models (https://arxiv.org/abs/2501.14051)
Comments:
          10 pages, 2 figures. To be published in ISBI 2025

- **What's New**: 이 논문에서는 CLIP 훈련을 3D 뇌 MRI에 처음으로 응용하고, 도메인 специф적인 3D 기반 모델 이미지 인코더를 훈련하여 모드 간 정렬이 소규모 데이터셋에서도 가능함을 보여줍니다. 또한, 3D에서의 훈련 시 배치 간의 임베딩을 누적하여 대비 손실을 안정화하는 방법을 제안했습니다. 이를 통해 3D MRI와 표 형식 데이터 간의 의미 있는 정렬이 가능함을 입증하였습니다.

- **Technical Details**: 주요 기술적 요소로는 제한된 크기의 데이터셋을 다룰 때 각 모드의 의미 있는 표현을 동시에 학습할 수 없다는 관찰이 있습니다. 이 문제를 해결하기 위해 우선 각 모드의 구체적인 인코더를 학습하고 이후 이를 일관된 임베딩 공간에 정렬하는 접근법을 사용합니다. 3D 뇌 MRI를 위한 인코더는 공공 MRI 데이터셋과 AMAES 프레임워크를 이용하여 대규모 프리트레이닝을 통해 학습되었습니다.

- **Performance Highlights**: 본 연구에서는 제안된 방법론이 제로샷 분류(zero-shot classification)와 이미지 검색(image-retrieval) 과제에서 평가되며, 특히 제로샷 분류에서는 3D MRI와 표 형식 데이터 간의 의미 있는 정렬을 보여주는 결과를 얻었습니다. 그러나 제로샷 이미지 검색은 여전히 도전적임을 발견하였습니다.



### Enhancing kelp forest detection in remote sensing images using crowdsourced labels with Mixed Vision Transformers and ConvNeXt segmentation models (https://arxiv.org/abs/2501.14001)
- **What's New**: 이번 연구는 군중 소싱(crowdsourced) 라벨과 첨단 인공지능 모델을 통합하여 Landsat 이미지를 활용한 해조류 캐노피(kelp canopy) 탐지 파이프라인을 빠르고 정확하게 개발하는 것을 목적으로 하고 있습니다. 이 방법은 기계 학습 대회에서 3위를 기록하였으며, 지역 검증 및 공개/비공식 리더보드에서 일관되게 좋은 성과를 보였습니다. Mixed Vision Transformers (MIT)와 ConvNeXt 모델의 조합이 효과적임을 강조하고 있습니다.

- **Technical Details**: 해조류 탐지를 위한 모델 학습은 다양한 이미지 크기로 진행되어 앙상블(ensemble) 결과의 정확도를 크게 향상시켰습니다. 특히, U-Net이 최고의 세분화(segmentation) 아키텍처로 선정되었고, UpperNet 또한 최종 앙상블에 기여하였습니다. Landsat의 ShortWave InfraRed (SWIR1) 및 Near-InfraRed (NIR)와 같은 주요 밴드가 중요한 역할을 했으며, 고도 데이터를 사용하여 거짓 긍정(false positives)을 제거하는 후처리 작업이 이루어졌습니다.

- **Performance Highlights**: 이 방법론은 높은 탐지율을 기록하였으며, 해조류 캐노피를 포함한 픽셀의 약 75%를 정확히 식별하면서 거짓 긍정은 낮은 수준으로 유지하였습니다. Landsat 위성의 중해상도에도 불구하고, 광범위한 역사적 데이터는 해조류 숲 연구에 효과적임을 입증하고 있습니다. 또한, 기계 학습 모델과 군중 소싱 데이터를 결합한 방법의 환경 모니터링에서의 가능성을 강조하고 있습니다.



### Comprehensive Modeling and Question Answering of Cancer Clinical Practice Guidelines using LLMs (https://arxiv.org/abs/2501.13984)
- **What's New**: 이번 연구에서는 의료 전문가가 치료 결정을 내리는 데 도움이 되기 위해 Clinical Practice Guidelines (CPGs)의 규정 지식을 효과적으로 포착하는 방법을 제안합니다. 특히 National Comprehensive Cancer Network (NCCN) Cancer CPGs의 맥락을 풍부하게 한 디지털 표현을 그래프 형태로 구축하는 접근 방식을 소개하고 있습니다. 이를 통해 CPGs에 포함된 의료 지식을 보다 충실하게 캡처할 수 있습니다.

- **Technical Details**: 연구진은 자동 추출 및 노드(node)와 관계(relationship)의 분류를 통해 보다 풍부한 디지털 표현을 생성하였습니다. 또한, 대형 언어 모델(Large Language Models, LLMs)을 활용하여 노드를 분류함으로써 80.86%와 88.47%의 정확도를 달성하였습니다. 이 과정에서 zero-shot learning과 few-shot learning을 적용하였습니다.

- **Performance Highlights**: 이 연구에서 개발한 방법론은 자연어 질문에 대한 답변을 제공하는 데 활용됩니다. LLMs를 활용하여 가이드라인 지식 기반에서 관련 서브그래프(subgraph)를 추출하고, 이를 통해 정확한 답변을 생성함으로써 의료 분야의 질문 응답(Question Answering)에서 발생할 수 있는 오류와 환각(hallucination) 문제를 완화합니다. 이러한 접근 방식은 의료 도메인에서 사실 정확성을 보장하는 데 기여합니다.



### Attribute-based Visual Reprogramming for Image Classification with CLIP (https://arxiv.org/abs/2501.13982)
- **What's New**: 이 논문에서는 CLIP을 위한 속성 기반 Visual Reprogramming(AttrVR)을 제안합니다. 기존의 VR 접근 방법이 고정된 텍스트 템플릿을 활용할 때 발생하는 단점을 극복하기 위해, 다양한 속성(DesAttrs 및 DistAttrs)을 이용하여 동적인 패턴 최적화를 추구합니다. AttrVR은 각 이미지 샘플에 대해 반복적으로 속성을 수정하며 학습하는 방식으로, 더 나은 분류 성능을 보여줍니다.

- **Technical Details**: AttrVR은 Descriptive attributes(DesAttrs)와 Distinctive attributes(DistAttrs)를 활용하여 다양한 클래스를 설명합니다. DesAttrs는 공통 특성을, DistAttrs는 클래스 간의 차별화된 특성을 나타냅니다. 각 샘플에 대해 가장 근접한 k개의 속성을 반복적으로 조회하여 VR 패턴을 개선하며, 이는 고정 템플릿 기반 레이블보다 더 맥락에 적합한 이미지-속성 정렬을 가능하게 합니다.

- **Performance Highlights**: Empirical 측면에서, AttrVR은 ViT 기반 및 ResNet 기반 CLIP 솔루션 모두에서 12개의 다운스트림 작업에서 우수한 성능을 달성하며, 기존 VR 방법들보다 일관되게 더 나은 결과를 나타냅니다. AttrVR의 성능은 이론적 분석과 실험적 결과 모두에서 증명되며, 기존의 레이블 기반 방법에 비해 분류 작업에서 명확한 장점을 제공합니다.



### A Spatio-temporal Graph Network Allowing Incomplete Trajectory Input for Pedestrian Trajectory Prediction (https://arxiv.org/abs/2501.13973)
- **What's New**: 이번 연구에서는 과거의 불완전한 경로를 고려하여 보행자의 미래 경로를 예측할 수 있는 새로운 spatio-temporal graph network인 STGN-IT를 제안합니다. 기존의 알고리즘들은 불완전한 정보를 처리하지 못하고, 이로 인해 로봇 내비게이션에 어려움을 겪어왔습니다. STGN-IT는 정적 장애물 정보를 통합하여 예측 정확도를 높이는데 기여합니다.

- **Technical Details**: STGN-IT 알고리즘은 spatio-temporal graph를 사용하여 보행자와 장애물 정보를 표현하고, 특별한 인코딩 방법을 통해 불완전한 과거 경로를 처리합니다. 이 알고리즘은 기존의 semantic map 대신 자동으로 생성된 occupancy grid map을 활용하여 더욱 유연한 접근을 제공합니다. 데이터셋 STCrowd에서 STGN-IT의 성능을 비교 평가한 결과, 기존 최첨단 알고리즘을 능가하는 성과를 보였습니다.

- **Performance Highlights**: STGN-IT는 기존 알고리즘과 비교했을 때 ADE와 FDE에서 개선된 성과를 나타내며, 특히 egocentric view 데이터셋에서 불완전한 경로에 대한 저항성을 발휘합니다. 실험 결과, STGN-IT는 로봇 내비게이션 환경에서 보다 안전하고 효율적인 경로 예측을 가능하게 하며, 이는 로봇이 보행자와의 충돌 위험을 줄이는데 도움이 됩니다.



### Patch-Based and Non-Patch-Based inputs Comparison into Deep Neural Models: Application for the Segmentation of Retinal Diseases on Optical Coherence Tomography Volumes (https://arxiv.org/abs/2501.13970)
Comments:
          6 pages, 1 figures, 2 tables, submitted to 15th IEEE Symposium on Computer Applications & Industrial Electronics

- **What's New**: 이 논문은 나이 관련 황반 변성(AMD)의 치유를 위한 자동화된 분석 기법을 소개합니다. 특히, 깊은 학습 모델의 입력 크기 설정이 성능에 미치는 영향을 분석하여, 패치 기반 입력이 전체 이미지를 사용하는 경우보다 성능을 개선하는 방법을 제시합니다. 이 연구는 2D, 2.5D, 3D 데이터를 사용하여 서로 다른 입력 크기에서 깊은 신경망의 성능을 비교합니다.

- **Technical Details**: 정밀한 계측 기법인 Optical Coherence Tomography (OCT)를 활용하여 망막의 다양한 층과 액체를 이미지화하고 연구합니다. 이 논문에서는 Intraretinal fluid (IRF), Subretinal fluid (SRF), Pigment epithelial detachment (PED)와 같은 망막 질환에 대한 자동 분할 성능을 조사합니다. 그리고 다수의 연구에서 발표된 다양한 딥러닝 모델의 비교를 통해 그 효율성을 분석합니다.

- **Performance Highlights**: 연구에서는 딥러닝 모델이 인간의 성능을 초월하는 결과를 보여주며, Dice Similarity Coefficient (DSC) 점수가 0.88로 나타났습니다. 이는 패치 기반 모델의 성능 향상이 입증된 결과로, 기존 인간의 평균 성능인 0.71보다 현저히 높은 수치입니다. 이 모델은 각기 다른 크기의 입력을 활용하여 세밀한 위험 요소들을 효과적으로 구별하는 데 성공적인 결과를 나타냈습니다.



### InsTex: Indoor Scenes Stylized Texture Synthesis (https://arxiv.org/abs/2501.13969)
- **What's New**: 본 논문에서는 3D 실내 장면을 위한 고품질 및 스타일 일관성을 갖춘 텍스처를 생성하는 새로운 아키텍처인 InsTex를 제안합니다. InsTex는 깊이-이미지 확산(Depth-to-Image Diffusion) 프라이어를 사용하여 초기 다중 뷰 이미지를 생성하고, 이후 품질과 일관성을 높이기 위한 리파인먼트 단계를 포함합니다. 이 방법은 텍스트와 비주얼 프롬프트를 모두 지원하며, 기존의 콘텐츠 생성 방식보다 개선된 결과를 보여줍니다.

- **Technical Details**: 제안하는 InsTex는 두 단계를 거치는 파이프라인을 사용하여 3D 실내 장면의 개별 객체를 텍스처링합니다. 첫 번째 단계에서는 깊이 인식을 통해 다중 뷰 이미지를 생성하고, 두 번째 단계에서는 이 이미지를 통해 텍스처를 정제하여 일관성을 유지합니다. 또한, 동적 뷰 파티셔닝 전략을 채택하여 각 뷰포인트에 따라 다른 영역에서 생성 목표를 설정하고, 깊이 인식을 통해 텍스처를 보강합니다.

- **Performance Highlights**: InsTex는 다양한 프롬프트를 기반으로 고품질의 텍스처를 생성하며, 시각적 품질 및 정량적 지표에서 최신 기법들을 초월하는 성능을 보여줍니다. 제안된 방법이 여러 3D 텍스처링 응용 프로그램에 효과적으로 적용될 수 있음을 입증하며, 특히 리얼타임 AR/VR 환경에서도 유용할 것으로 기대됩니다.



### Triplet Synthesis For Enhancing Composed Image Retrieval via Counterfactual Image Generation (https://arxiv.org/abs/2501.13968)
Comments:
          4 pages, 4 figures

- **What's New**: 이 논문은 Composed Image Retrieval (CIR)에서의 훈련 데이터 생성을 개선하기 위한 새로운 triplet 합성 방법을 제안합니다. 이 방법은 counterfactual image generation을 활용하여 수작업 없이 다양한 training triplets를 자동으로 생성함으로써 CIR 모델의 성능을 높입니다. 이러한 접근법은 특히 데이터가 제한적인 상황에서 유용하게 적용될 수 있습니다.

- **Technical Details**: 제안된 방법은 두 단계를 포함합니다: counterfactual caption 생성과 그에 따른 이미지 생성입니다. 초기 reference 이미지에 수정 텍스트를 적용하여 target 이미지를 생성하며, 이 과정에서 Language-guided Counterfactual Image (LANCE) 모형이 사용됩니다. LANCE는 간접적인 변경 사항을 반영하여 이미지의 특정 속성을 변화시켜, 고유한 triplet을 합성할 수 있도록 도와줍니다.

- **Performance Highlights**: 제안된 방법은 고품질의 triplet 생성을 통해 CIR 모델을 개선할 수 있는 잠재력을 증명합니다. 제한된 데이터셋에서도 다양하고 질 높은 triplet 생성을 지원하며, 이러한 triplet들은 모델의 훈련에 있어 더 효과적으로 작용합니다. 결과적으로, 이 방법은 기존의 자동 triplet 합성에서 발생할 수 있는 문제들을 극복하고 실용적인 대규모 CIR 데이터셋 생성을 지원할 수 있습니다.



### ZKLoRA: Efficient Zero-Knowledge Proofs for LoRA Verification (https://arxiv.org/abs/2501.13965)
Comments:
          7 pages, 3 figures

- **What's New**: ZKLoRA는 Large Language Models (LLMs)을 위한 Low-Rank Adaptation (LoRA) 방법을 사용하여 분산 훈련 환경에서의 신뢰성 문제를 해결하는 제로 지식 검증 프로토콜을 제시합니다. 이 프로토콜은 LoRA 가중치의 기밀성을 유지하면서도 base 모델과의 호환성을 인증할 수 있도록 합니다. ZKLoRA는 검증 시간을 1-2초로 단축 시키고, 안전한 협업을 가능하게 하여, 기여자의 지적 재산을 보호합니다.

- **Technical Details**: ZKLoRA는 LoRA 모듈의 정확성을 검증하기 위해 간결한 증명을 제공하며, Multi-Party Inference 절차를 기반으로 합니다. 이렇게 설계된 ZKLoRA는 분산 환경에서도 효율적으로 작동하여, LoRA 기여자가 자신의 가중치를 공개하지 않고도 그 가중치가 실제 base 모델과 잘 작동하는지를 보장합니다. 이 방법은 저지연 접근성을 통해 실시간 검증을 촉진합니다.

- **Performance Highlights**: ZKLoRA는 다양한 LLM과 소규모 모델에서 벤치마킹 테스트를 수행하였으며, LoRA 모듈 수에 따라 검증 시간과 설정, 증명 생성 시간을 분석했습니다. 모델의 LoRA 모듈 수가 많을수록 검증 시간이 증가하지만, ZKLoRA의 설계 덕분에 80개의 모듈을 검증하는 데 몇 분밖에 걸리지 않습니다. 이로써 ZKLoRA는 대규모 LLM 파이프라인에서의 LoRA 사용의 가능성을 높이고, 점진적 검증 절차를 통해 코드 기밀성을 보장합니다.



### Procedural Generation of 3D Maize Plant Architecture from LIDAR Data (https://arxiv.org/abs/2501.13963)
- **What's New**: 이 연구는 LiDAR 포인트 클라우드 데이터를 기반으로 옥수수 식물의 절차적 3D 모델을 생성하는 강력한 프레임워크를 소개합니다. 이 프레임워크는 Non-Uniform Rational B-Spline (NURBS) 표면을 사용하여 옥수수 잎을 모델링하며, Particle Swarm Optimization (PSO)를 초기 근사화에 활용합니다. 이후 미분 가능 프로그래밍 프레임워크를 통해 표면을 정교하게 다듬어 LiDAR 데이터에 맞춥니다.

- **Technical Details**: 최초 최적화 단계에서 PSO는 제어점을 최적화하여 NURBS 표면을 생성하고, LiDAR 데이터와 정렬하여 신뢰할 수 있는 시작점을 제공합니다. 두 번째 단계에서는 NURBS-Diff라는 미분 가능 프로그래밍 프레임워크를 사용하여 표면 기하학의 정확성을 향상시키고 복잡한 잎 세부 정보를 포착합니다. 이것은 다양한 유전자형에서 정확한 3D 재구성을 가능하게 하여 복잡한 특성 추출로 이어집니다.

- **Performance Highlights**: 연구 결과는 PSO가 강력한 초기 피팅을 확립하는 반면, 미분 가능 NURBS의 통합이 재구성된 표면의 전반적인 품질과 충실도를 크게 향상시킨다는 것을 보여줍니다. 이 계층적 최적화 전략은 다양한 유전자형의 옥수수 잎의 정확한 3D 재구성을 지원하며, 이후 복잡한 특성의 추출을 용이하게 합니다. 모든 코드는 오픈 소스 형태로 제공되어 이러한 표현형 분석 접근 방식을 민주화하는 데 기여합니다.



### Adaptive Cyber-Attack Detection in IIoT Using Attention-Based LSTM-CNN Models (https://arxiv.org/abs/2501.13962)
- **What's New**: 이 논문은 산업 인터넷의 사물(IIoT) 환경에서 사이버 공격을 탐지하고 분류하기 위한 하이브리드 LSTM-Convolution Neural Network(CNN)-Attention 아키텍처에 기반한 진보된 침입 탐지 시스템(IDS)의 개발 및 평가를 제시합니다. 이 연구는 이진 및 다중 클래스 분류의 두 가지 주요 분류 작업에 중점을 두고 있으며, Edge-IIoTset 데이터셋을 사용하여 제안된 모델이 엄격히 테스트되었습니다. 또한, SMOTE(특정 소수 샘플 오버 샘플링 기법)를 활용하여 불균형한 데이터 세트를 보정함으로써 모델의 학습 효과를 극대화했습니다.

- **Technical Details**: 제안된 LSTM-CNN-Attention 모델은 LSTM의 시간적 모델링 강점, CNN의 공간적 특성 추출 능력, 그리고 Attention 메커니즘에 의한 동적 초점을 결합하여 구축되었습니다. 이 연구에서는 60개의 특성과 15가지 공격 유형을 포함하는 데이터셋을 이차원 형태로 구조화하였으며, 비율에 따라 80%는 훈련용, 20%는 테스트용으로 분할했습니다. SMOTE 알고리즘을 사용하여 소수 클래스에 대한 합성 샘플을 생성하고, 학습과 검증 세트를 다시 나누어 균형 분포를 확인했습니다.

- **Performance Highlights**: 실험 결과, LSTM-CNN-Attention 모델은 이진 분류에서 거의 완벽한 정확도를 달성하였고, 다중 클래스 분류에서는 99.04%의 높은 정확도를 유지하며 다양한 공격 유형을 효과적으로 분류했습니다. 손실 값은 0.0220%으로 매우 낮아, 모델이 다수의 메트릭스에서 다른 모델들보다 우수함을 입증했습니다. 이러한 성과는 IIoT 환경의 복잡한 사이버 보안 문제를 해결하는 데 기여할 것으로 기대됩니다.



### A Fast, Scalable, and Robust Deep Learning-based Iterative Reconstruction Framework for Accelerated Industrial Cone-beam X-ray Computed Tomography (https://arxiv.org/abs/2501.13961)
- **What's New**: 본 논문에서는 대규모 산업용 원뿔형 X선 컴퓨터 단층촬영(cone-beam XCT) 데이터를 위한 새로운 딥 뉴럴 네트워크 기반의 반복 알고리즘을 제안합니다. 이 알고리즘은 아티팩트 제거 훈련된 CNN을 사전 모델로 통합하고 자동화된 정규화 파라미터 선택 기능을 포함하여, 극도로 밀집한 금속 부품에 대한 고품질 3D 재구성을 몇 번의 반복만으로도 수행할 수 있습니다. 또한, 다양한 스캐닝 조건에서 얻어진 분포 외(scan out-of-distribution) 스캔에 대한 일반화 가능성도 입증됩니다.

- **Technical Details**: 제안하는 알고리즘은 Half-Quadratic Splitting(HQS) 구성에서 영감을 받아 CNN-정규화된 물리 기반 역전파 알고리즘을 사용합니다. 프로세스에서 CNN과 공액 경량(CG) 알고리즘 간의 교차 횟수를 재구성 품질에 따라 조절하며, CNN은 저품질 희소 보기 FDK 재구성의 아티팩트를 줄이는 훈련을 받습니다. 또한, 자동으로 정규화 강도를 조정하여 CT 재구성 알고리즘에서 중요한 도전과제를 해결하는 정규화 파라미터 선택 전략이 포함됩니다.

- **Performance Highlights**: 제안된 알고리즘은 X선 소스 전압, 총 통합 시간 및 희소성 등 다양한 조건에서 얻은 XCT 스캔에 대해 기존 방법들과 비교하여 인상적인 성능을 보입니다. 정규화 선택 전략은 고정된 정규화 파라미터를 사용하는 것보다 더 나은 성능을 발휘하며, 반복 수가 적어 MBIR보다 계산 복잡도가 크게 줄어듭니다. 이 알고리즘은 원뿔형 XCT 응용에 적합하도록 설계되었지만, 다른 대규모 이미지 재구성 응용으로의 확장 가능성도 가지고 있습니다.



### Language Representation Favored Zero-Shot Cross-Domain Cognitive Diagnosis (https://arxiv.org/abs/2501.13943)
- **What's New**: 이 논문은 언어 표현 기반의 제로 샷 크로스 도메인 인지 진단(Language Representation Favored Zero-shot Cross-domain Cognitive Diagnosis, LRCD) 접근법을 제안합니다. 기존의 인지 진단 모델들은 특정 도메인에 맞춰 모델을 훈련해야 하며, 이는 다양한 과목이나 교육 플랫폼에서의 직접적인 응용을 어렵게 만듭니다. LRCD는 텍스트 설명을 사용하여 학생, 과제 및 개념의 프로필을 만들고, 이를 통일된 언어 공간에 벡터화하여 인지 진단 모델과 통합하는 방법을 제안합니다.

- **Technical Details**: LRCD는 학생과 과제의 행위 패턴을 분석하고, 각 도메인에서 이를 텍스트로 설명합니다. 이렇게 생성된 프로필은 최신 텍스트 임베딩 모듈을 통해 통합 언어 공간의 벡터로 변환됩니다. 그러나 언어 공간과 인지 진단 공간 간의 불일치를 해결하기 위해, LRCD는 두 공간 간의 매핑을 학습하는 언어-인지 맵퍼를 제안합니다. 이를 통해 기존의 인지 진단 모델과 통합하여 효율적으로 훈련할 수 있습니다.

- **Performance Highlights**: LRCD는 다양한 실제 데이터셋에 대해 제로 샷 성능을 commendable하게 달성할 수 있으며, 특정 경우에는 기존의 인지 진단 모델과 경쟁할 만한 성능을 보여줍니다. 학계 및 교육 현장에서 학생들의 과목 간 차이를 분석하는 데에도 유용한 인사이트를 제공합니다. 흥미롭게도, 과학 과목에서 생성된 데이터는 다양한 목표 도메인에서 더 나은 전이 성능을 보이며, 고등 교육 수준의 학생 데이터를 기반으로 할 경우 저학년 학생들에게도 더 큰 전이 가능성을 보입니다.



### GaussMark: A Practical Approach for Structural Watermarking of Language Models (https://arxiv.org/abs/2501.13941)
- **What's New**: 최근 Large Language Models (LLMs)의 발전은 자연어 처리(NLP) 작업에 큰 개선을 가져왔지만, 사람과 같은 품질의 텍스트 생성을 가능하게 하는 이 기술은 윤리적 및 운영상의 우려를 야기하고 있습니다. 이에 따라, LLM으로 생성된 텍스트를 확인할 수 있는 수단인 워터마킹(watermarking) 기법 개발에 대한 연구가 이루어졌습니다. 현재의 기존 워터마킹 기법은 생성 지연(generation latency)과 탐지 시간(detection time), 텍스트 품질 저하 등 여러 측면에서 실용적이지 않은 경우가 많았으며, 이 문제들을 해결하기 위한 새로운 접근 방식이 필요합니다.

- **Technical Details**: 이 연구에서는 GaussMark라는 새로운 워터마킹 기법을 소개합니다. 이 방법은 구현하기 간단하고 효율적이며, 형태학적인 워터마크(structural watermark)를 모델의 가중치(weights) 자체에 내장합니다. Gaussian 독립성 테스트를 기반으로 하고 있으며, LLM의 가중치에 소량의 Gaussian 노이즈를 추가하여 워터마킹을 수행합니다. 이 과정은 통계적으로 검출 가능하도록 설계되었으며, 비밀 키를 가진 제공자가 이 텍스트가 자신의 모델에 의해 생성되었다는 것을 확인할 수 있게 합니다.

- **Performance Highlights**: GaussMark는 광범위한 실험을 통해 신뢰성과 효율성을 입증하였으며, 삽입, 삭제, 치환, 왕복 번역(roundtrip translation)과 같은 복원력 있는 다양한 왜곡에 대해 비교적 강력함을 보여줍니다. 이 방법은 모델 품질 손실 없이 거의 무제한의 성능 범위를 제공합니다. 또한, 우리의 접근 방식은 생성 과정에서 추가적인 지연을 발생시키지 않아 실제 적용에 적합합니다.



### Evaluating Computational Accuracy of Large Language Models in Numerical Reasoning Tasks for Healthcare Applications (https://arxiv.org/abs/2501.13936)
Comments:
          13 pages, 1 figure, 2 tables

- **What's New**: 이번 연구는 의료 부문에서 사용되는 대규모 언어 모델(Large Language Models, LLMs)의 숫자 추론 능력에 대한 최초의 평가를 수행했습니다. LLMs는 자연어 이해 및 생성에서의 놀라운 성능을 보여주지만, 높아진 절차적 요구에 맞춘 숫자 추론은 여전히 미흡하다는 점에 주목합니다. 이 연구가 다룬 1,000개의 숫자 문제를 통해 실제 의료 환경에서의 수치적 추론 능력을 탐색하게 되었습니다.

- **Technical Details**: 연구에서는 GPT-3 아키텍처를 기반으로 한 정제된 LLM의 성능을 평가하기 위해 신중하게 수집된 데이터셋을 사용했습니다. 방법론에는 프롬프트 엔지니어링(prompt engineering), 사실 확인 파이프라인(fact-checking pipelines) 통합, 정규화 기법(regularization techniques) 적용이 포함되어 모델의 정확도 및 일반화 능력을 향상시켰습니다. 모델의 효율성을 평가하기 위해 precision, recall, F1-score와 같은 주요 메트릭(metrics)을 사용했습니다.

- **Performance Highlights**: 결과적으로 LLM은 84.10%의 전체 정확도를 기록했으며, 간단한 숫자 작업에서는 향상된 성능을 보였지만, 다단계 추론(multi-step reasoning)에서는 어려움이 있었습니다. 사실 확인 파이프라인의 통합이 정확도를 11% 향상시켜 검증 메커니즘의 중요성을 강조했습니다. 이 연구는 의료 분야에서 숫자 추론에 대한 LLM의 잠재력을 부각시키며, 임상 환경에서의 중요한 의사 결정 지원을 위한 LLM의 추가 개선 방향 또한 제시합니다.



### Low rank matrix completion and realization of graphs: results and problems (https://arxiv.org/abs/2501.13935)
Comments:
          21 pages, 6 figures

- **What's New**: 이 논문에서는 넷플릭스 문제(Netflix problem)와 관련된 매트릭스 완성을 다룹니다. 기존의 특정 매트릭스 요소를 아는 대신, 선형 관계를 통해 매트릭스의 최적 추천을 위한 나머지 항목 예측을 수행하는 방법을 제안합니다. 이 연구는 매트릭스의 랭크를 최소화하며, 토폴로지적 응용에서 그래프의 서피스(embeddings of graphs in surfaces)에 대한 중요한 결과를 다룹니다.

- **Technical Details**: 이 논문에서는 ℤ2={0,1}의 원소를 갖는 매트릭스의 초기 결과를 다루며, 특히 대각선의 미지의 요소에 대한 최소 랭크를 추정하기 위한 알고리즘을 제시합니다. 또한, 구체적인 매트릭스 요소 대신 선형 관계를 활용하여 더 일반적인 문제를 연구합니다. 이를 통해 두 가지 정리에 따라 이러한 선형 관계를 가진 매트릭스의 최소 랭크를 계산할 수 있습니다.

- **Performance Highlights**: 논문에서는 비대칭 매트릭스에 대한 O(n²)의 복잡도를 갖는 결정 알고리즘을 소개합니다. 이 알고리즘은 대각선에서 일부 항목을 변경하여 결과적으로 모든 비영(非零) 행이 같아지는지를 결정합니다. 이러한 결과는 매트릭스의 특성과 관련하여 다양한 토폴로지적 응용 가능성을 열어줍니다.



### Revisiting Online Learning Approach to Inverse Linear Optimization: A Fenchel$-$Young Loss Perspective and Gap-Dependent Regret Analysis (https://arxiv.org/abs/2501.13648)
- **What's New**: 이번 논문은 Bärmann et al. (2017)에서 연구된 역선형최적화(inverse linear optimization)의 온라인 학습(online learning) 접근 방식을 재조명합니다. 연구의 주요 목표는 에이전트의 입력-출력 쌍에 대한 순차적 관찰로부터 알려지지 않은 선형 목표 함수(linear objective function)를 추론하는 것입니다. 또한, 본 연구는 Fenchel-Young losses와의 연결을 통해 온라인 학습 방법을 간단히 설명합니다.

- **Technical Details**: 논문은 목표 함수의 하위 최적 손실(suboptimality loss)에 대한 오프라인 보장을 제공합니다. 이는 예측된 목표가 에이전트의 선택을 얼마나 잘 설명하는지를 측정합니다. 또한, 에이전트의 의사결정 문제에서 최적 및 비최적 목표 값 사이의 갭이 존재한다고 가정할 때, 시간 지평(T)와 무관하게 하위 최적 손실과 추정 손실(estimate losses)의 합에 대한 상한을 도출합니다.

- **Performance Highlights**: 흥미롭게도, 갭에 의존하는 분석은 역선형 최적화에 고유한 구조를 활용함으로써 표준 $O(\sqrt{T})$ 후회(bound)보다 빠른 속도를 기록합니다. 이는 손실 함수와 그 도메인이 강한 볼록성(strong convexity) 같은 바람직한 특성을 갖지 않더라도 달성한 결과입니다.



### A Comprehensive Survey on Spectral Clustering with Graph Structure Learning (https://arxiv.org/abs/2501.13597)
- **What's New**: 본 논문은 스펙트럴 클러스터링의 방법론을 포괄적으로 살펴보며, 그래프 구조 학습(Graph Structure Learning, GSL)의 중요성을 강조합니다. 다양한 그래프 구성 기법(techniques)인 pairwise, anchor, hypergraph 기반 방법을 고정 방식과 적응 방식에서 탐구하며, 단일 및 다중 뷰 프레임워크(single-view and multi-view frameworks)로 분류합니다. 또한, 여러 정보 융합 기술을 검토하여 복잡한 데이터 클러스터링의 향상에 기여하는 중요한 통찰을 제공합니다.

- **Technical Details**: 스펙트럴 클러스터링은 고차원 데이터를 처리하기 위해 그래프 기반 접근법을 사용하여 비볼록(non-convex) 클러스터와 비선형 구조를 잡아내는 기법입니다. 특히, 유사성 그래프의 건설은 전체 클러스터링 프로세스의 기초로, 클러스터링 성능 개선에 중점을 두어 GSL의 역할을 강조합니다. 본 논문에서는 고정형 및 적응형 그래프 구조를 통해 클러스터링 기법을 분류하고 다양한 그래프 구성 방식이 클러스터링 결과에 미치는 영향을 논의합니다.

- **Performance Highlights**: 스펙트럴 클러스터링은 복잡하고 고차원 데이터에 특히 적합합니다. 최적의 그래프 구축 기법을 활용함으로써 클러스터링 성능이 향상되며, 특히 대규모 데이터셋의 경우 anchor graph 기반의 기법이 계산 자원을 절약하면서도 높은 클러스터링 품질을 유지할 수 있다는 장점이 있습니다. 따라서 본 논문에서는 GSL이 스펙트럴 클러스터링에서의 대규모 및 고차원 데이터 클러스터링 작업에 있어 중요한 역할을 한다고 강조합니다.



### ExLM: Rethinking the Impact of [MASK] Tokens in Masked Language Models (https://arxiv.org/abs/2501.13397)
Comments:
          29 pages, 12 figures

- **What's New**: 이번 연구에서는 Masked Language Models (MLMs)의 [MASK] 토큰 사용이 모델 성능에 미치는 영향을 분석합니다. 기존 연구들은 주로 [MASK]의 비현실적인 문제에 집중했으나, 이번 논문에서는 맥락 왜곡 문제와 함께 이를 다룹니다. 제안된 개선된 맥락 MLM은 ExLM으로, [MASK] 토큰을 활용해 더 풍부한 의미 정보를 캡처하도록 설계되었습니다.

- **Technical Details**: ExLM 모델은 입력 맥락 내에서 [MASK] 토큰을 확대하여 고유한 상태 간의 의존성을 모델링합니다. 이를 통해 문맥 용량이 증가하고, 전이 학습(transfer learning) 시 부정확한 의미를 줄이는 데 도움이 됩니다. 모델 실험은 BERT 기초 모델과 유사한 설정으로 수행되었으며, 14세트의 MLM에 대해 다양한 매개변수를 사용해 훈련합니다.

- **Performance Highlights**: ExLM은 텍스트 모델링 및 SMILES 모델링 작업에서 성능 향상을 크게 보여주었습니다. 추가 분석을 통해 ExLM이 맥락 향상을 통해 의미 표현을 강화시키고, MLM에서 자주 발생하는 다중 의미 문제를 효과적으로 줄임을 확인했습니다. 따라서 ExLM은 MLM의 성능 향상에 기여할 수 있는 가능성이 높은 모델입니다.




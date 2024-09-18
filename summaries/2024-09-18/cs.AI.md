New uploads on arXiv(cs.CL)

### AraDiCE: Benchmarks for Dialectal and Cultural Capabilities in LLMs (https://arxiv.org/abs/2409.11404)
Comments:
          Benchmarking, Culturally Informed, Large Language Models, Arabic NLP, LLMs

- **What's New**: 이번 연구는 아랍어 방언(dialect)에 대한 데이터셋 부족 문제를 해결하기 위해, 현대 표준 아랍어(Modern Standard Arabic, MSA) 및 아랍어 방언의 합성 데이터셋을 7개 소개하고, AraDiCE라는 새로운 벤치마크를 발표합니다. 이 벤치마크는 방언 이해(dialect comprehension) 및 생성(generation) 평가에 초점을 맞추고 있습니다.

- **Technical Details**: 연구에서는 기계 번역(Machine Translation, MT)과 인간의 후편집을 결합하여 방언을 포함한 7개의 합성 데이터셋을 구축하였습니다. 이 데이터셋은 레반트(Levantine)와 이집트(Egypt) 방언에 중점을 두고 있으며, 지역 문화 인식을 평가하기 위한 AraDiCE-Culture라는 새 벤치마크를 포함하고 있습니다. 이는 지역의 문화적 특성을 이해하는 데 집중하고 있습니다.

- **Performance Highlights**: 아랍어 중심 모델(Jais, AceGPT)은 방언 작업에서 다국어 모델보다 우수한 성능을 보였으나, 여전히 방언 식별(dialect identification), 생성, 번역에서 상당한 어려움이 있는 것으로 나타났습니다. LLM은 특정 방언을 이해하고 생성하는 데 제한이 있으며, 문화적 뉘앙스를 이해하는 데 있어 아랍어 중심 모델이 다국어 모델보다 우수합니다. 이 연구는 약 45,000개의 후편집 샘플과 문화 벤치마크를 제공하여 LLM의 성능을 향상시키는 맞춤형 훈련의 중요성을 강조합니다.



### NVLM: Open Frontier-Class Multimodal LLMs (https://arxiv.org/abs/2409.11402)
- **What's New**: NVLM 1.0은 비전-언어(vision-language) 작업에서 최첨단 결과를 달성하는 다중 모드 대형 언어 모델(multimodal large language models, LLMs)입니다. 특히, NVLM 1.0은 다중 모드 훈련 후 텍스트 전용 성능이 LLM 백본보다 개선되었습니다.

- **Technical Details**: NVLM 1.0은 세 가지 아키텍처(NVLM-D, NVLM-X, NVLM-H)로 구성되어 있습니다. NVLM-D는 디코더 전용 아키텍처로 OCR 관련 작업에서 높은 정확도를 제공하며, NVLM-X는 교차 주의(cross-attention) 기반 아키텍처로 고해상도 이미지를 처리하는 데 효율적입니다. NVLM-H는 두 가지 접근 방식을 통합하여 성능과 효율성을 동시에 향상시킵니다.

- **Performance Highlights**: 다양한 비전-언어 작업 및 텍스트 전용 작업에서 NVLM 모델들은 최고의 성능을 달성했습니다. 특히, 멀티모달 수학 및 코딩 데이터 세트를 통합하여 모든 NVLM 모델에서 텍스트 전용 성능의 개선이 달성되었습니다.



### Says Who? Effective Zero-Shot Annotation of Focalization (https://arxiv.org/abs/2409.11390)
- **What's New**: 이번 논문에서는 현대의 Large Language Models(LLMs)가 문학 텍스트의 초점(focalization) 모드를 주석(annotation)하는 능력을 실험을 통해 평가합니다. LLMs는 훈련된 인간 주석자와 유사한 성능을 보여주며, 이 접근 방식이 컴퓨터 문학 연구에 유용함을 강조합니다.

- **Technical Details**: 연구에서는 Stephen King의 16개 소설로 구성된 데이터셋을 사용하여 LLM이 초점 주석을 수행할 수 있는지를 평가했습니다. Naive Bayes 분류기, DistilBERT 모델, quantized Llama 3 모델, GPT 모델들을 포함한 다양한 모델이 분석되었으며, gpt-4o는 인간 합의 레이블과 86%의 높은 F1 점수를 달성하였습니다.

- **Performance Highlights**: gpt-4o는 다양한 프롬프트에도 일관된 라벨링을 제공하였고, Stephen King의 작품에서 감각적 정보 측정과 비교하여 초점 모드에 따라 다채로운 독창성을 보여주었습니다.



### Diversify and Conquer: Diversity-Centric Data Selection with Iterative Refinemen (https://arxiv.org/abs/2409.11378)
Comments:
          21 pages, 6 figures

- **What's New**: 이 연구에서는 대형 언어 모델(LLM)의 단계적 훈련을 위해 최적의 데이터 서브셋을 선택하는 방법에 대한 새로운 접근 방식을 제안합니다. 기존 연구는 특정 인스턴스 품질에 중점을 두었으나, 우리는 전 세계적인 데이터 다양성을 강조합니다. k-means 클러스터링을 사용하여 전체 데이터 집합을 효과적으로 대표하는 서브셋을 선택하는 방법을 소개합니다.

- **Technical Details**: 제안하는 방법은 반복적인 세분화 기법을 활용하여 인스턴스를 클러스터로부터 재샘플링하고, 각 클러스터의 중요성과 샘플링 가중치를 훈련(iteration)마다 재평가합니다. 이 과정은 이상치(outlier)의 영향을 줄이고 저품질 데이터가 포함된 클러스터를 자동으로 필터링하는데 효과적입니다.

- **Performance Highlights**: 자연어 추론, 일반 세계 지식, 코드 및 수학 추론 작업 등의 평가에서 모델 성능이 일관되게 향상되었습니다. 무작위 선택 대비 7% 증가, 최신 샘플링 방법 대비 3.8% 개선된 성능을 달성했습니다. 또한, 대부분의 다운스트림(task)에서 모든 이전 방법보다 우수한 결과를 보여주었습니다.



### CoCA: Regaining Safety-awareness of Multimodal Large Language Models with Constitutional Calibration (https://arxiv.org/abs/2409.11365)
Comments:
          10 pages, COLM-2024

- **What's New**: 이번 연구에서는 Multimodal Large Language Models (MLLMs)이 악의적인 이미지 입력에 대한 안전 인식을 보유하고 있는지를 검토하였으며, 이를 통해 모델의 안전 인식을 향상시키기 위한 간단한 방법인 Constitutional Calibration (CoCA)을 제안했습니다.

- **Technical Details**: MLLM은 Large Language Models (LLMs) 기반으로 시각 인코더를 통합하여 시각적 입력을 처리합니다. 연구에 따르면, MLLMs는 악의적인 의도를 가진 이미지 입력에 대해 유해한 응답을 생성하는 경향이 있으며, 이는 모달리티 격차(modality gap)로 인해 약화됩니다. CoCA 기법은 안전 요구사항을 명시하는 추가 프롬프트를 입력에 추가하여 MLLM의 안전 인식을 강화합니다. LoGits 차이를 계산하고 이를 조정하여 안전 프롬프트의 효과를 극대화합니다. 이 과정에서 모델의 시각적 이해력과 추론 능력도 유지됩니다.

- **Performance Highlights**: 제안된 CoCA 기법은 MLLM의 안전 인식을 강화하면서도 원래의 기능을 잃지 않도록 도와줍니다. 실험 결과, 이 접근법은 MLLMs의 안전 인식을 증대시키고 악의적인 이미지 입력에 대한 경계선을 높였음을 보여주었습니다.



### CORE-Bench: Fostering the Credibility of Published Research Through a Computational Reproducibility Agent Benchmark (https://arxiv.org/abs/2409.11363)
Comments:
          Benchmark harness and code available at this http URL

- **What's New**: CORE-Bench (Computational Reproducibility Agent Benchmark)가 도입되었습니다. 이 벤치마크는 90편의 과학 논문을 기반으로 270개의 작업으로 구성되어 있으며, 컴퓨터 과학, 사회 과학, 의학의 세 가지 분야를 포함합니다. 이 작업은 AI 에이전트의 실험 차원에서 중요한 컴퓨테이셔널 재현성(computational reproducibility)을 평가하는 데 중점을 두고 있습니다.

- **Technical Details**: CORE-Bench는 세 가지 난이도 수준의 작업으로 구성되며, language-only 및 vision-language 작업을 포함합니다. 두 개의 기본 에이전트인 일반 목적의 AutoGPT와 특정 작업에 맞춘 CORE-Agent를 평가하였으며, 각각 GPT-4o와 GPT-4o-mini 기반으로 실험을 진행했습니다. CORE-Bench는 특정 AI 에이전트의 성능을 신속하고 병렬적으로 측정할 수 있는 평가 시스템을 제공하여, 평가 시간이 단축되었습니다.

- **Performance Highlights**: 가장 높은 성능을 보인 에이전트는 가장 어려운 작업에서 21%의 정확도를 기록하였습니다. 이는 과학적 연구의 자동화에 대한 잠재력이 크지만, 미흡한 부분이 많다는 것을 나타냅니다. 특정 작업에 적응할 수 있는 일반 에이전트의 성능 향상 가능성도 확인되었습니다.



### THaMES: An End-to-End Tool for Hallucination Mitigation and Evaluation in Large Language Models (https://arxiv.org/abs/2409.11353)
Comments:
          Submitted to NeurIPS 2024 SoLaR (Socially Responsible Language Modelling Research ) Workshop

- **What's New**: 본 논문은 LLM(대형 언어 모델)에서 발생하는 환각(hallucination) 문제를 해결하기 위해 THaMES(도구 툴 for Hallucination Mitigations and EvaluationS)를 소개합니다. THaMES는 환각의 평가 및 완화를 위한 통합된 프레임워크로서, 자동화된 테스트 세트 생성, 다양한 벤치마킹, 적응형 완화 전략을 제공합니다.

- **Technical Details**: THaMES는 (1) 사용자 제공 코퍼스에서 테스트 세트 생성, (2) 테스트 세트를 기반으로 한 기준 메트릭 평가, (3) 기준 메트릭에 따라 완화 전략 평가의 세 가지 주요 요소로 구성되어 있습니다. 이 프레임워크는 배치 처리(batch processing), 가중 샘플링(weighted sampling) 및 다양한 질문 유형을 포함하여 환각을 식별하고 생성하는 능력을 평가합니다.

- **Performance Highlights**: THaMES를 통해 OpenAI의 GPT-4o와 Meta의 Llama-3.1-8B-Instruct 모델을 평가했습니다. 결과적으로 GPT-4o는 RAG(검색 증강 생성)를 통해 성능이 크게 향상되었으며, Llama-3.1은 ICL(문맥 내 학습)로 더 좋은 성과를 보였습니다. PEFT(매개변수 효율적 미세 조정)가 Llama-3.1의 성능을 개선하는 결과를 나타냈습니다.



### SpMis: An Investigation of Synthetic Spoken Misinformation Detection (https://arxiv.org/abs/2409.11308)
Comments:
          Accepted in SLT 2024

- **What's New**: 이번 논문에서는 합성 음성 정보의 오용 문제를 해결하기 위한 최초의 데이터셋인 SpMis를 소개합니다. 이 데이터셋은 1,000명 이상의 화자를 통해 생성된 합성 음성을 포함하고 있으며, 합성 음성이 미치는 잠재적 영향 분석에 중점을 두고 있습니다.

- **Technical Details**: SpMis 데이터셋은 금융, 의학, 정치, 법률 및 교육의 5개 주제에 대해 생성된 합성 음성을 포함하고 있으며, 이는 최신 text-to-speech 시스템을 사용하여 생성되었습니다. 이 연구는 스피커의 신원, 주제 및 합성 여부를 고려하여 합성 음성 정보 감지를 위한 기초 탐지 시스템을 제안합니다.

- **Performance Highlights**: 기초 감지 시스템은 유망한 결과를 보였으나, 실용적인 구현에는 상당한 도전이 있다는 점이 강조되었습니다. 합성 음성의 오용을 막기 위한 기존 연구는 주로 기계 생성 음성과 인간 음성을 이분법적으로 분류하는 데 중점을 두고 있었으나, 본 연구에서는 특정 주제를 다룰 수 있는 단골 화자에 한하여 합성 음성 괴롭힘을 탐지하는 것을 목표로 하고 있습니다.



### Zero-resource Hallucination Detection for Text Generation via Graph-based Contextual Knowledge Triples Modeling (https://arxiv.org/abs/2409.11283)
- **What's New**: 이번 연구에서는 그래프 기반의 맥락 인식(GCA, Graph-Based Context-Aware) hallucination 탐지 방법을 제안하여, 긴 텍스트 생성에 있어서 여러 사실 간의 종속성을 고려하고 사실 정렬을 개선하여 일관성 비교를 수행합니다.

- **Technical Details**: 제안된 방법은 삼중항(response segmentation) 기반으로 여러 지식 삼중(triples)을 추출하고, RGCN(Relational Graph Convolutional Network)을 통해 사실 간의 종속성을 모델링합니다. 이를 통해 연결된 노드의 특성을 집계하여 정보 전송을 촉진시킵니다.

- **Performance Highlights**: 실험 결과, 본 방법은 외부 자원 없이 검정 모델에서 생성된 긴 텍스트 응답의 hallucination 탐지 정확도를 효과적으로 향상시켰으며, 모든 기본선(line)보다 우수한 성능을 보였습니다.



### Leveraging Distillation Techniques for Document Understanding: A Case Study with FLAN-T5 (https://arxiv.org/abs/2409.11282)
Comments:
          Presented at AI@WORK-Workshop / Informatik-Festival (GI-Jahrestagung) (Wiesbaden, Germany, 2024)

- **What's New**: 이번 논문에서는 비즈니스 보고서 및 환경 평가와 같은 비표준 문서의 이해에 대한 중요성이 증가함에 따라, 대형 언어 모델(LLMs)의 문서 이해에 대한 직접적인 응용의 어려움을 다룹니다. 본 연구는 ChatGPT에서 FLAN-T5로 문서 이해 지식을 증류하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 연구 방법론은 레이블링(labeling) 및 커리큘럼 학습(curriculum-learning) 메커니즘을 통합하여 효율적인 지식 이전(knowledge transfer)을 촉진합니다. 이 증류(distillation) 방법은 대형 LLM의 계산적 한계를 수용하면서 효과적으로 활용할 수 있는 방법을 제시합니다.

- **Performance Highlights**: 이 연구는 자원 집약적인 LLM과 실제 응용 프로그램 간의 간극을 좁히는 확장 가능한 솔루션을 제공하여 자연어 처리(natural language processing) 및 문서 이해(document comprehension) 분야의 발전을 촉진합니다. 또한, 고급 언어 모델을 실제 시나리오에서 배포할 수 있는 잠재력을 강조합니다.



### Task Arithmetic for Language Expansion in Speech Translation (https://arxiv.org/abs/2409.11274)
- **What's New**: 최근 대규모 언어 모델(LLMs)의 발전으로 인해 음성-텍스트(multimodal) 기초 모델이 주목받고 있으며, 지침 기반의 음성 번역(speech translation, ST)에서 뛰어난 성능을 보여주고 있습니다. 그러나 기존 ST 시스템에서 언어 쌍을 확장하는 작업은 비싼 재훈련을 요구합니다. 본 연구에서는 언어 제어 모델(language control model)을 추가하여 언어 혼동을 제거하고 새로운 언어 쌍을 확장하는 방법을 제안합니다.

- **Technical Details**: 본 연구에서는 기존 ST 모델과 새로운 언어 쌍을 훈련받은 모델을 결합하는 모델 병합(Model Merging) 접근법을 사용합니다. 기본적으로 작업 산술(task arithmetic)을 통해 모델의 파라미터를 조작하여 새로운 모델을 생성하고, 추가된 언어 제어 모델을 통해 정확한 목표 언어 토큰을 생성하도록 유도합니다. 실험에서는 MuST-C 및 CoVoST-2 데이터셋을 사용하여 성능을 평가하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 언어 제어 모델을 통해 MuST-C 데이터셋에서 최대 4.66, CoVoST-2 데이터셋에서 최대 4.92 의 BLEU 점수 향상을 달성했습니다. 또한, ST 훈련 데이터가 없거나 사전 훈련된 ST 모델이 존재하지 않는 경우에도 새로운 언어 쌍에 대해 시스템을 합성하고 기존 모델에 병합하여 성능을 향상시킬 수 있음을 보여주었습니다.



### LOLA -- An Open-Source Massively Multilingual Large Language Mod (https://arxiv.org/abs/2409.11272)
- **What's New**: 이번 논문에서는 160개 이상의 언어로 훈련된 대규모 멀티링구얼 모델 LOLA를 소개합니다. LOLA는 sparse Mixture-of-Experts Transformer 아키텍처를 기반으로 하여 언어적 다양성을 활용하는 동시에 효율성을 유지하고 멀티링구얼리티(multilinguality)의 일반적인 문제들을 피하도록 설계되었습니다.

- **Technical Details**: LOLA는 GPT 스타일의 decoder-only 아키텍처를 따르며, sparse Mixture-of-Experts (MoE) 층을 활용하여 모델 용량을 늘립니다. MoE는 특정 언어 군에 대한 은닉 구조를 학습하고, 이를 통해 더 나은 언어 모델 성능을 달성할 수 있도록 돕습니다. 또한, LOLA는 정확한 데이터셋 분석과 훈련 과정에 대한 심층적인 탐구를 제공합니다.

- **Performance Highlights**: LOLA는 4가지 작업 유형(질문 응답(Q&A), 추론, 자연어 추론(NLI), 독해)을 평가하였고, 13개의 멀티링구얼 작업에서 17개의 다른 모델들과 비교하여 경쟁력 있는 성능을 보였습니다. 특히, LOLA는 낮은 자원 언어를 위한 교차 언어 전이 학습 향상에 긍정적인 영향을 미쳤습니다.



### The Art of Storytelling: Multi-Agent Generative AI for Dynamic Multimodal Narratives (https://arxiv.org/abs/2409.11261)
- **What's New**: 이 논문은 Generative Artificial Intelligence (GenAI)를 활용하여 어린이들을 위한 스토리텔링을 향상시키기 위한 교육 도구의 개념을 소개합니다. 이 시스템은 GenAI 기반의 내러티브 공동 창작(narrative co-creation), 텍스트-음성 변환(text-to-speech) 및 텍스트-비디오 생성(text-to-video)을 결합하여 학습자에게 매력적인 경험을 제공합니다.

- **Technical Details**: 시스템은 Freytag의 피라미드 및 Propp의 서사적 기능을 기반으로 구성되어 있으며, 사용자의 입력에 따라 내러티브를 생성하기 위해 Large Language Models (LLMs), TTS 모델, TTV 모델, 그리고 TTM 모델을 활용합니다. 이 시스템은 다중 에이전트 구조를 통해 스토리생성, 리뷰 및 음성과 비디오의 변환 과정을 체계화하였습니다.

- **Performance Highlights**: 시스템 평가는 생성된 스토리의 언어학적 분석, TTS 변환 품질 및 생성된 비주얼의 정확성을 포함합니다. 사용자 평가 데이터를 공개하여 LLMs-as-evaluators 및 LLMs-as-content-reviewers에 대한 벤치마크를 제공합니다. 최종 출력은 텍스트 스토리, 음성 내레이터 및 배경 음악을 포함한 애니메이션 비디오로, 어린이를 위한 몰입형 스토리텔링 경험을 제공합니다.



### Norm of Mean Contextualized Embeddings Determines their Varianc (https://arxiv.org/abs/2409.11253)
- **What's New**: 이 연구에서는 Transformer 모델의 중간 계층에서 추출한 contextualized embeddings의 분포 분석을 통해, 평균 embedding의 노름(nor)과 분산(variance) 간의 강한 trade-off 관계를 발견했습니다. 이는 layer normalization 메커니즘의 영향을 받을 수 있습니다.

- **Technical Details**: 연구는 벡터 공간에서 token의 contextualized embeddings의 평균 제곱 노름(mean squared norm) Q(Xt), 평균 embedding의 제곱 노름(squared norm of the mean embedding) M(Xt), 그리고 각 컴포넌트의 분산의 합(sum of the variances) V(Xt)을 중심으로 진행되었습니다. 특히, M(Xt)이 의미의 강도를 나타내고 V(Xt)는 분포의 확산(spread)을 나타냄을 보여줍니다.

- **Performance Highlights**: 실험 결과, Transformer 모델의 깊이가 깊어질수록 embeddings가 원점(origin)에서 멀어지고, between-cluster variance는 상대적으로 감소하며, within-cluster variance는 상대적으로 증가하는 경향을 보였습니다.



### WER We Stand: Benchmarking Urdu ASR Models (https://arxiv.org/abs/2409.11252)
- **What's New**: 이 논문은 우르두어 자동 음성 인식(ASR) 모델의 포괄적인 평가를 제공합니다. Whisper, MMS, Seamless-M4T의 세 가지 ASR 모델 계열을 성능 분석하고, 최초의 우르두어 대화 음성 데이터셋을 통해 ASR 모델을 벤치마킹하는 작업을 포함합니다.

- **Technical Details**: 읽기 음성과 대화 음성을 기준으로 두 가지 유형의 데이터셋을 사용하여 ASR 모델의 성능을 분석하였습니다. 모델의 성능은 Word Error Rate (WER)로 측정되며, 오류 유형은 삽입, 삭제 및 대체를 포함합니다. 새로운 대화 음성 데이터셋은 실제 대화 환경을 재현하기 위해 인터넷 전화를 통해 녹음되었습니다.

- **Performance Highlights**: Seamless-large 모델은 읽기 음성 데이터셋에서 최고의 성능을 보였고, whisper-large 모델은 대화 음성 데이터셋에서 최상의 성능을 기록했습니다. 본 연구는 저자원 언어인 우르두어를 위한 ASR 모델 평가의 복잡성을 강조하며, 강력한 텍스트 정규화 시스템의 필요성을 강조합니다.



### Linear Recency Bias During Training Improves Transformers' Fit to Reading Times (https://arxiv.org/abs/2409.11250)
- **What's New**: 최근 심리언어학 연구에서는 인간의 읽기 시간과 언어 모델의 surprisal 추정치를 비교하여 인간의 문장 처리 난이도를 결정하는 요인을 연구했습니다. 이 연구는 Transformer 모델의 수정판을 평가하여 ALiBi (Press et al., 2022)를 사용함으로써 성과를 얻었습니다.

- **Technical Details**: ALiBi는 주의 점수를 조정하기 위해 최근성 편향을 추가하는 방법이며, 이로 인해 Transformer의 surprisal 추정치가 향상되었습니다. 실험에서는 ALiBi의 기울기 혼합이 각 주의 헤드에서의 기억 감소율을 결정하는 역할을 하여 언어적 의존성을 추적하는 데 도움을 주는 것을 보여주었습니다.

- **Performance Highlights**: ALiBi를 사용한 Transformer는 표준 Transformer에 비해 인간의 읽기 시간과 더 잘 맞는 surprisal 추정치를 제공했습니다. 이러한 결과는 인간 언어 이해 모델에서 기억 감소를 구현하는 데 흥미로운 의미를 가질 수 있습니다.



### Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to Refus (https://arxiv.org/abs/2409.11242)
- **What's New**: 이 연구에서는 LLM(대규모 언어 모델)이 RAG(검색 보강 생성) 시스템에서의 적합성을 평가하기 위한 새로운 신뢰성 지표인 Trust-Score를 도입합니다. Trust-Align 프레임워크를 통해 LLM의 신뢰성을 높이는 방법을 제안하며, 이를 통해 LLaMA-3-8b 모델이 다양한 오픈소스 LLM에 비해 성능 개선을 보였습니다.

- **Technical Details**: Trust-Score는 LLM의 신뢰성을 여러 차원에서 평가합니다: Grounded Refusals(거부 가능성 평가), Exact Match Recall(정확한 응답 비율), Citation Recall(인용 지원 비율), Citation Precision(인용의 적절성). Trust-Align 프레임워크를 통해 19,000개의 질문과 문서, 긍정 및 부정 응답으로 구성된 정렬 데이터셋을 생성하였습니다. 이 데이터셋은 5가지 환각 유형을 해결하기 위해 사용됩니다.

- **Performance Highlights**: Trust-Align이 적용된 모델은 ASQA, QAMPARI, ELI5 벤치마크 데이터셋에서 각각 10.73%, 29.24%, 14.88%의 Trust-Score 향상을 보여주었습니다. 또한 모델의 거부 반응 정확성을 9.87%, 22.53%, 5.32% 향상시켜 신뢰성과 인용 품질을 높였습니다. 이 연구는 RAG 설정에서 LLM의 환각을 조사한 첫 번째 연구입니다.



### Spontaneous Informal Speech Dataset for Punctuation Restoration (https://arxiv.org/abs/2409.11241)
Comments:
          8 pages, 7 tables, 1 figure, Recognition Technologies, Inc. Technical Report

- **What's New**: SponSpeech는 비격식적인 대화에서 파생된 구두점 복원 데이터세트로, 다양한 말씨와 구두점 정보를 포함하며, 마지막 평가 세트인 test-amb는 보다 구두점 애매성을 포함한 문장을 포함한다.

- **Technical Details**: SponSpeech 데이터세트는 YouTube에서 제공되는 창의적 커먼즈(creative commons) 라이센스를 가진 비디오에서 수집하였으며, 데이터의 품질을 보장하기 위해 5단계 필터링 파이프라인을 사용하여 입후보 비디오를 평가하였다. 데이터셋은 훈련 세트(train), 검증 세트(dev), 일반 테스트 세트(test), 그리고 구두점 애매성 테스트 세트(test-amb)로 나뉜다. 각 세트의 비율은 전체 데이터셋의 약 70%, 12%, 9%, 9%를 차지한다.

- **Performance Highlights**: SponSpeech 데이터세트는 비격식적이고 자연스러운 대화를 반영하여 모델들이 구두점 애매성을 해결하는 데 더 높은 도전을 제공, 따라서 모델의 전반적인 성능 향상을 기대할 수 있다.



### LLM-as-a-Judge & Reward Model: What They Can and Cannot Do (https://arxiv.org/abs/2409.11239)
Comments:
          preprint

- **What's New**: 이 논문에서는 LLM(as-a-Judge) 및 보상 모델이 다국어 환경에서 어떻게 작동하는지를 조사하였고, 특히 한국어에 대한 첫 번째 비영어 메타 평가 데이터셋인 Kudge를 공개했습니다.

- **Technical Details**: 본 연구는 영어 이외의 언어에서 자동 평가자의 작동 방식을 분석하며, LLM의 평가 능력이 특정 언어의 능력보다도 영어 평가 능력에 크게 의존한다는 사실을 발견했습니다. 또한, LLM이 정보의 정확성, 문화적 오해, 불필요한 언어 사용을 감지하고 처벌하는 데 실패하는 주요 단점을 확인했습니다.

- **Performance Highlights**: 이 연구는 LLM의 비영어 환경에서의 한계와 강점을 밝히고, Kudge 데이터셋을 통해 한국어 LLM 평가의 기반을 마련함으로써 차세대 LLM의 발전에 기여할 것입니다.



### Evaluating the Impact of Compression Techniques on Task-Specific Performance of Large Language Models (https://arxiv.org/abs/2409.11233)
- **What's New**: 이번 연구에서는 대형 언어 모델(Large Language Models, LLMs)의 압축 기법들이 효율성과 성능 간의 균형을 어떻게 효과적으로 관리할 수 있는지를 분석하였습니다. 특히, Magnitude Pruning, SparseGPT, 및 Wanda와 같은 압축 방법들이 LLaMA-2-7B 모델에 미치는 영향을 평가했습니다.

- **Technical Details**: 압축 기법의 효과를 평가하기 위해 Jenson-Shannon (JS) Divergence를 새로운 지표로 도입하였습니다. 이 지표는 모델 압축 후의 세부 변화들을 더 잘 설명하며, 모델 성능에 대한 임무 특화된 교정 데이터가 중요한 역할을 한다는 것을 시사합니다. 이 연구에서는 128개의 무작위 샘플을 C4 데이터셋으로 이용하여 SparseGPT와 Wanda의 성능을 조절하고, 5,000개의 무작위 샘플에서 손실(Loss)과 Perplexity를 측정했습니다.

- **Performance Highlights**: SparseGPT와 Wanda는 각각 50% 희소성(sparsity)에서도 perplexity를 유지했지만, 하위 과제(downstream task) 성능에서는 큰 손실을 보였습니다. 새로운 평가 지표인 JS Divergence는 압축 후 모델의 성능 유지를 평가하는 데 더욱 적합함을 보여주었으며, 특수한 교정 데이터가 압축된 모델의 성능을 큰 폭으로 향상시켰다는 점을 강조했습니다.



### Fast Analysis of the OpenAI O1-Preview Model in Solving Random K-SAT Problem: Does the LLM Solve the Problem Itself or Call an External SAT Solver? (https://arxiv.org/abs/2409.11232)
- **What's New**: OpenAI의 O1-preview 모델이 K-SAT 문제 해결에 대한 새로운 성능 분석이 진행되었으며, 모델이 외부 SAT solver를 호출했음을 발견했다는 점이 주목할 만하다.

- **Technical Details**: 모델 성능 분석은 K-SAT 문제의 채택도에 따라 달라지며, α = M/N의 함수로 작용한다. 모델은 문제를 직접 해결하기보다는 외부 SAT solver에 의존하며, 이는 성능 평가에 한계를 두고 있다. K는 2, 3, 4로 설정되어 있으며, 복잡한 문제의 경우 SAT solver와의 호출이 이루어졌다. 이 과정에서 모델의 학습 수준과 작동 원리에 대한 불확실성이 제기되었다.

- **Performance Highlights**: OpenAI O1-preview 모델은 초기 간단한 문제를 잘 해결했지만, 복잡성이 증가함에 따라 외부 도구를 사용하는 경향을 보였다. 분석 과정에서 모델이 K-SAT 문제를 스스로 해결할 수 있는지에 대한 의문이 제기되었고, 모델의 진정한 지능 여부에 대한 논의가 필요함을 알렸다.



### Exploring ChatGPT-based Augmentation Strategies for Contrastive Aspect-based Sentiment Analysis (https://arxiv.org/abs/2409.11218)
Comments:
          8 pages, 3 figures

- **What's New**: 이 연구에서는 ChatGPT를 활용한 데이터 증강(data augmentation) 기법을 통해 특정 측면 용어에 대한 감정 분류 성능을 향상시키는 방법을 제시합니다. 이를 통해 라벨링된 데이터 부족 문제를 해결하고, 다각적인 감정 이해를 가능하게 합니다.

- **Technical Details**: 제안된 세 가지 데이터 증강 기법은 context-focused, aspect-focused, 그리고 context-aspect 증강 방식으로 구성됩니다. 특히, BERT를 백본 모델로 사용해 데이터를 변환하고, ChatGPT를 활용하여 데이터의 다양성과 의미적 풍부성을 증가시키는 것이 주요한 요소입니다.

- **Performance Highlights**: 모든 데이터 증강 기법이 성능 향상에 기여하였으며, 특히 context-aspect 증강 전략이 가장 뛰어난 성능을 보였습니다. 실험 결과, 제안된 방법은 기존 모델들의 성능을 상회하는 것으로 나타났습니다.



### Self-Evolutionary Large Language Models through Uncertainty-Enhanced Preference Optimization (https://arxiv.org/abs/2409.11212)
Comments:
          17 pages

- **What's New**: 본 논문에서는 Iterative Preference Optimization을 위한 새로운 UPO (Uncertainty-enhanced Preference Optimization) 프레임워크를 제안합니다. 이 프레임워크는 LLM이 신뢰할 수 있는 피드백을 통해 스스로 발전할 수 있도록 설계되었습니다.

- **Technical Details**: UPO 프레임워크는 현재 정책 및 보상 모델로부터 파생된 noisy preference data를 완화하기 위해 Pair-wise uncertainty estimation 및 reliable feedback sampling을 수행합니다. Monte Carlo (MC) dropout을 사용하는 추정기 모델이 도입되어 Bayesian neural network (BNN)를 통해 uncertainty estimation을 수행합니다.

- **Performance Highlights**: 다양한 NLP 벤치마크 및 수학적 추론 작업에 대한 실험 결과, UPO 프레임워크가 iterative preference optimization의 효과성을 크게 향상시키고 자동 평가에서 최상의 성능을 달성하는 것으로 나타났습니다.



### SAGED: A Holistic Bias-Benchmarking Pipeline for Language Models with Customisable Fairness Calibration (https://arxiv.org/abs/2409.11149)
Comments:
          Submitted to COLING 2025 Main Conference

- **What's New**: SAGED(-Bias)는 편향을 평가하기 위한 최초의 포괄적 메트릭 파이프라인으로, 기존의 한계점을 극복하고 대규모 정량적 평가를 가능하게 합니다.

- **Technical Details**: SAGED는 다섯 개의 핵심 단계로 구성됩니다: Scraping, Assembling, Generating, Extracting, Diagnosing. 이 과정에서 최대 불균형(Max Disparity) 및 편향 집중(Bias Concentration) 지표를 도입하여 편향 평가를 진행합니다.

- **Performance Highlights**: Mistral과 Qwen2는 Gemma2 및 Llama3.1보다 낮은 최대 불균형과 높은 편향 집중도를 보였으나, 모든 모델은 러시아와 중국에 대해 상당한 편향을 나타냈습니다. 역할 수행 실험에서는 Llama3.1과 Gemma2가 트럼프를 도와주는 성향이 더 두드러졌습니다.



### Improving the Efficiency of Visually Augmented Language Models (https://arxiv.org/abs/2409.11148)
- **What's New**: 이 논문은 시각적 지식이 부족한 기존의 Language Model (LM)들을 개선하는 새로운 접근 방식을 제시합니다. 새로운 모델 BLIND-VALM은 이미지 검색 시스템이나 생성 시스템 없이 CLIP 모델에서 얻은 시각적 기반 텍스트 표현을 사용하여 LM을 시각적으로 보강합니다.

- **Technical Details**: BLIND-VALM은 기존의 VALM 구조를 수정하여 CLIP 모델의 텍스트 인코더 표현을 활용합니다. 이는 이미지 검색 및 표현을 피하면서도 LM에 시각적 지식을 통합할 수 있게 해줍니다. BLIND-VALM은 VLU, NLU 및 LM 작업에서 VALM과 동등한 성능을 보입니다.

- **Performance Highlights**: BLIND-VALM은 훈련 및 추론 속도에서 VALM보다 극적으로 빠르며, 동일한 계산 예산 내에서 VALM을 초과하는 성능 개선을 보여줍니다.



### Reasoning Graph Enhanced Exemplars Retrieval for In-Context Learning (https://arxiv.org/abs/2409.11147)
- **What's New**: 이 논문에서는 Reasoning Graph-enhanced Exemplar Retrieval(RGER)라는 새로운 방법을 제안합니다. RGER는 LLM의 성능을 개선하기 위해 문제 해결 과정의 중간 단계 간 관계를 그래프 구조로 표현합니다.

- **Technical Details**: RGER는 크게 두 부분으로 구성됩니다. 첫 번째 부분에서는 LLM에 두 번 쿼리를 수행하여 초기 응답을 생성하고, 그 결과를 통해 그래프 구조의 정보를 생성합니다. 두 번째 부분에서는 이 구조적 정보를 활용하여 최종 예제를 선정하는 retrieval 과정을 진행합니다.

- **Performance Highlights**: RGER는 수학 및 논리 추론 과제에서 기존의 최첨단 접근 방식들에 비해 우수한 성능을 나타냄을 보였으며, 다양한 시나리오에서의 강건성을 입증하였습니다.



### Semformer: Transformer Language Models with Semantic Planning (https://arxiv.org/abs/2409.11143)
- **What's New**: 이번 논문에서는 Semformer라는 새로운 방법론을 제안합니다. 이는 Transformer 언어 모델을 훈련시키는 새로운 방법으로, 응답의 의미적 계획(semantic planning)을 명시적으로 모델링합니다. 특히, 계획 토큰 시퀀스를 접두사(prefix)에 포함시켜, 이 계획 토큰 표현들이 후속 응답의 잠재적 의미 표현을 예측하도록 유도합니다.

- **Technical Details**: Semformer는 오직 훈련 중에만 사용되는 언어 모델과 오토인코더(autoencoder)로 구성됩니다. 언어 모델 부분에서, 입력의 접두사 뒤에 의미적 계획 토큰 시퀀스를 도입하며, 이 시퀀스는 일반적인 다음 토큰 예측 손실(next-token prediction loss)을 무시하고 이후 토큰의 잠재적 표현(latent representations)을 예측하는 데 사용됩니다. 오토인코더는 후속 토큰을 저차원 공간으로 압축하는 잠재적 표현의 시퀀스를 생성하는 방법을 학습합니다.

- **Performance Highlights**: Semformer는 그래프 경로 탐색(graph path-finding) 문제에서 거의 100%의 정확도를 달성하며, 이는 기존의 기초 모델들보다 우수한 성능을 보여줍니다. 또한, Baselines보다 훨씬 빠른 속도로 문제를 해결하는 데 성공했으며, OpenWebText에서 125M 파라미터로 처음부터 훈련하여, perplexity 평가, in-context learning, 추상 요약(fine-tuning on summarization tasks)에서 향상된 성능을 보여주었습니다.



### Diversity-grounded Channel Prototypical Learning for Out-of-Distribution Intent Detection (https://arxiv.org/abs/2409.11114)
Comments:
          work in progress

- **What's New**: 이 연구는 대규모 언어 모델(large language models, LLMs)을 위한 새로운 파인튜닝 프레임워크를 제안하여 실세계 시나리오에서 발생하는 형식이 잘못된 발화를 효과적으로 처리할 수 있는 의도 감지 메커니즘을 향상시킵니다. 특히, 기존의 식별내부(in-distribution, ID) 의도 분류와 비식별내부(out-of-distribution, OOD) 의도 감지를 개선하는데 중점을 두고 있습니다.

- **Technical Details**: 이 프레임워크는 ID 클래스 이름에서 파생된 프로토타입과 의미적 매칭(semantic matching)을 사용하여, 각 ID 클래스를 위한 의미 프로토타입(semantic prototypes)을 구축합니다. 다양한 예제를 기반으로 한 프롬프트 튜닝(diversity-grounded prompt tuning) 접근 방식을 통해 LLM의 고유한 표현을 활용하여, LLM은 의미 프로토타입을 분류하는 데 사용됩니다. 본 연구는 ID와 OOD 클래스가 의미적으로 유사한 상황, 즉 'near' OOD 감지 문제를 다룹니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존의 파인튜닝 접근 방식과 비교하여 제한된 수의 샘플을 이용한 ID 의도 분류 및 near-OOD 의도 탐지 작업에서 우수한 성능을 보여줍니다. 이는 실질적인 응용에서의 효율성을 향상시킵니다.



### Strategic Insights in Human and Large Language Model Tactics at Word Guessing Games (https://arxiv.org/abs/2409.11112)
Comments:
          Published in the 4th Wordplay: When Language Meets Games Workshop @ ACL 2024

- **What's New**: 2022년 초부터 시작된 간단한 단어 추측 게임이 전 세계적으로 인기를 끌며 다양한 언어로 확장되었습니다. 이 연구는 2년 이상 동안 발전한 일일 단어 추측 게임 플레이어의 전략을 분석합니다.

- **Technical Details**: 조사 결과, 25%의 자주 플레이하는 사용자들의 전략과 지속적인 플레이의 동기를 파악했습니다. 또한, 여러 공개된 대형 언어 모델 시스템과 오픈 소스 모델이 두 가지 언어로 게임을 이해하고 플레이하는 능력을 조사했습니다.

- **Performance Highlights**: 결과는 특정 모델들이 올바른 추측 길이를 유지하는 데 어려움을 겪고, 반복을 생성하는 데 힘든 점, 존재하지 않는 단어와 굴절형을 만들어내는 환각(hallucinations) 문제를 강조합니다.



### RoMath: A Mathematical Reasoning Benchmark in Romanian (https://arxiv.org/abs/2409.11074)
Comments:
          4 Figures, 12 Tables

- **What's New**: RoMath라는 새로운 Romanian 수학적 추론 벤치마크 스위트를 소개하며, 이는 세 가지 데이터셋(RoMath-Baccalaureate, RoMath-Competitions, RoMath-Synthetic)을 포함하여 다양한 수학 영역 및 난이도를 커버하고 있습니다. 이 연구는 비영어 언어 모델을 개선하고 다국어 AI 개발을 촉진하는 것을 목표로 합니다.

- **Technical Details**: RoMath는 약 76,910개의 문제 진술로 구성된 새로운 수학적 추론 벤치마크 스위트입니다. 이 데이터세트는 세 가지 하위 집합으로 나뉘며, 각각의 고유한 특성과 난이도 수준(예: RoMath-Baccalaureate와 같은 고등학교 수준 문제)을 가지고 있습니다. 연구팀은 70,000개의 문제를 포함한 데이터를 수집하고 정리하는 데 반자동 워크플로우를 사용하여 기초 LLM(large language models)을 활용했습니다.

- **Performance Highlights**: RoMath는 여러 영어 및 Romanian 개방형 가중치 LLM을 사용하는 포괄적인 벤치마크를 제공하며, LLM-as-a-judge 패러다임을 사용하여 평가 절차를 시행합니다. 이 연구에서는 단순한 문제 진술의 번역이 성능을 상당히 저하시킨다는 것을 강조하고, 영어와 다른 언어에서 전념할 필요가 있음을 논의합니다.



### KVPruner: Structural Pruning for Faster and Memory-Efficient Large Language Models (https://arxiv.org/abs/2409.11057)
- **What's New**: KVPruner는 대형 언어 모델의 추론 프로세스를 위한 KV 캐시 최적화를 위한 새로운 구조적 가지치기 방법을 제안합니다. 특히, 글로벌 당혹도 기반 분석을 통해 각 블록의 중요 비율을 결정하고 비필수 KV 채널 가지치기를 위한 다양한 전략을 제공합니다.

- **Technical Details**: KVPruner는 두 단계로 구성되어 있습니다: Phase A는 주어진 가지치기 목표에 대한 최적의 글로벌 가지치기 비율을 평가하고, Phase B는 Q, K, V 및 O 채널의 블록 내 중요성을 처리하여 가지치기 작업을 실행합니다. 이 방법은 런타임 중 추가 오버헤드 없이 원샷 가지치기를 수행합니다.

- **Performance Highlights**: KVPruner는 원래 모델에 비해 런타임 메모리 사용량을 50% 줄이고, 처리량(throughput)을 35% 이상 증가시킵니다. 또한, 소규모 데이터셋에서 단 2시간의 LoRA 미세 조정만으로 대부분의 성능을 회복할 수 있습니다.



### Large Language Models are Good Multi-lingual Learners : When LLMs Meet Cross-lingual Prompts (https://arxiv.org/abs/2409.11056)
- **What's New**: 이 논문에서는 LLM(대규모 언어 모델)의 복잡한 규칙을 이해하고 추론하는 능력을 향상시키기 위해 MLPrompt라는 새로운 프롬프트 기법을 제안합니다. 이 방법은 LLM이 따르기 어려운 규칙을 자동으로 다른 언어로 번역하여 이를 강조하게 만듭니다.

- **Technical Details**: MLPrompt는 LLM이 긴 문맥에서 자주 발생하는 규칙 생략 문제를 해결하기 위한 기법으로, 다양한 데이터 세트에서 Chain of Thought, Tree of Thought, Self-Consistency와 같은 기존의 최첨단 프롬프트 기법들을 능가하는 성능을 보였습니다.

- **Performance Highlights**: 실험 결과, MLPrompt는 여러 공공 데이터 세트 및 태스크에서 탁월한 성능을 보여주었으며, 텍스트를 MIP(최소 인터페이스 표현)과 SQL(구조적 쿼리 언어)로 변환하는 사례 연구를 통해 구조화된 데이터 생성을 위한 능력을 입증했습니다.



### A Comprehensive Evaluation of Quantized Instruction-Tuned Large Language Models: An Experimental Analysis up to 405B (https://arxiv.org/abs/2409.11055)
Comments:
          11 pages, 1 figure

- **What's New**: 이번 연구에서는 다양한 양자화 방법(GPTQ, AWQ, SmoothQuant, FP8)을 사용하여 7B에서 405B에 이르는 instruction-tuned LLM의 성능을 평가하였습니다. 13개 벤치마크를 활용하여 발생중인 LLM의 정확성과 성능 변화를 종합적으로 분석하였습니다.

- **Technical Details**: 연구에서는 양자화 방법론으로 Quantization Aware Training (QAT)과 Post-Training Quantization (PTQ)을 분류하였습니다.  경량화된 모델을 위한 4개의 양자화 방법이 사용되었으며, 각 방법은 7B에서 405B에 이르는 다양한 크기의 모델에 적용되었습니다. 평가에서는 commonsense Q&A, 지식 및 언어 이해, instruction following, hallucination detection, 수학, 대화 등 총 6개 작업 유형이 사용되었습니다.

- **Performance Highlights**: 주요 발견 사항으로는 (1) 더 큰 LLM을 유사한 크기(예: 13B)로 양자화했을 때, 전반적으로 우수한 성능을 보였으며, (2) 양자화 방법, 모델 크기, 비트 폭에 따라 성능이 크게 변동했고, (3) 작업 난이도는 양자화로 인한 정확도 저하에 큰 영향을 미치지 않았고, (4) MT-Bench 평가 방법이 최근 고성능 LLM들 간의 차별화 능력이 제한적임을 확인하였습니다.



### Towards No-Code Programming of Cobots: Experiments with Code Synthesis by Large Code Models for Conversational Programming (https://arxiv.org/abs/2409.11041)
- **What's New**: 이 연구는 Collaborative Robots (cobots)가 산업 환경에서 자연어를 통해 프로그래밍할 수 있는 새로운 접근 방식을 제안합니다. 기존의 복잡한 프로그래밍 기술을 요구하는 것이 아니라, Large Language Models (LLMs)를 활용하여 자연어로 지시를 주고 이에 따른 코드를 생성하는 방식을 모색합니다.

- **Technical Details**: 이 연구에서는 'Repetitive Assembly Task' (RATS)를 정의하여 시뮬레이션 산업 조립 시나리오의 기초를 마련합니다. 데이터셋은 목표 구조와 다양한 예제 지시(인간 작성, 템플릿 기반, 모델 생성) 및 예제 코드를 쌍으로 이루어집니다. 연구 결과, LLMs는 정확한 '1차 코드'를 생성할 수 있으나, '고차 코드' 생성에서는 어려움을 겪는 것으로 나타났습니다.

- **Performance Highlights**: 상황 시뮬레이션 환경에서 LLM의 코드를 생성하는 능력을 체계적으로 평가한 결과, LLM이 1차 코드 생성에는 성공적이지만, 함수나 루프와 같은 고차 코드 생성에는 제한적이라는 점을 발견하였습니다.



### Hierarchical Narrative Analysis: Unraveling Perceptions of Generative AI (https://arxiv.org/abs/2409.11032)
- **What's New**: 이 논문에서는 기존의 텍스트 마이닝 기법이 갖는 한계를 극복하기 위해 대규모 언어 모델(large language models, LLMs)을 이용하여 문헌의 구조를 계층적으로 분석하는 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 일반적인 감정 분석(sentiment analysis) 및 주제 모델링(topic modeling) 기술이 조망할 수 없는 복잡한 서사적 구조(hierarchical narrative structures)를 추출하고 조직화하는 것에 중점을 둡니다. 일본 문화청에서 수집된 생성 AI에 대한 공적 의견을 분석하여 지지자와 비판자의 내러티브를 비교합니다.

- **Performance Highlights**: 분석 결과, 생성 AI에 대한 상이한 의견을 형성하는 요인들을 명확히 시각화할 수 있었으며, 동의 및 반대의 구조에 대한 깊은 통찰력을 제공합니다.



### GEIC: Universal and Multilingual Named Entity Recognition with Large Language Models (https://arxiv.org/abs/2409.11022)
- **What's New**: 이번 논문에서는 Named Entity Recognition (NER) 작업을 위한 새로운 접근법인 Generation-based Extraction and In-context Classification (GEIC)을 제안합니다. GEIC는 LLMs의 주의 메커니즘을 활용하여 기존의 NER 방법보다 더 효율적으로 작동하도록 설계되었습니다.

- **Technical Details**: CascadeNER라는 프레임워크를 통해 GEIC를 구현하며, 이는 두 개의 소규모 LLM(Small Language Models, SLMs)을 이용하여 추출 및 분류 작업을 수행합니다. 이 방식은 자원 소비를 줄이고 정확성을 증대시킵니다. 또한, AnythingNER이라는 첫 번째 NER 데이터셋을 소개하며, 8개 언어와 155개의 개체 유형을 포함하고 있습니다.

- **Performance Highlights**: CascadeNER는 CrossNER와 FewNERD를 포함한 낮은 자원 및 세밀한 시나리오에서 SOTA(state-of-the-art) 성능을 달성했습니다. 또한, 이 연구는 LLM 기반 NER 방법에 대한 포괄적인 비교 및 분석을 제공하여 향후 방향성을 제시합니다.



### CAST: Cross-modal Alignment Similarity Test for Vision Language Models (https://arxiv.org/abs/2409.11007)
- **What's New**: 이 논문에서는 Cross-modal Alignment Similarity Test (CAST)를 제안하여 Vision Language Models (VLMs)의 확장성과 일관성을 평가하고, 전통적인 Visual Question Answering (VQA)와는 다른 새로운 접근 방식을 탐색합니다.

- **Technical Details**: CAST는 두 장면 간의 유사성을 찾도록 모델에 요구함으로써 VLM의 멀티모달(self-consistency) 기능을 평가하는 자동화된 2단계 방법입니다. 이 방법은 장면의 이미지와 고품질 설명 간의 유사성을 평가하며, 텍스트 전용, 이미지 전용 또는 두 가지 모두를 활용하여 모달리티(modality) 간의 일관성을 분석합니다.

- **Performance Highlights**: 테스트 결과, 다수의 VLM들이 다양한 다운스트림 작업에서 뛰어난 성능을 보였음에도 불구하고 내부의 자기 일관성(self-consistency)과 모달리티 정렬(modiaty alignment)에서 결핍 현상을 보였습니다. 이는 CAST가 VLM의 추론 능력과 잠재적인 편향(bias)을 이해하는 데 중요한 역할을 한다는 것을 보여줍니다.



### Enhancing Low-Resource Language and Instruction Following Capabilities of Audio Language Models (https://arxiv.org/abs/2409.10999)
Comments:
          5 pages. Preprint under review

- **What's New**: 이번 논문은 오디오 언어 모델이 태국어와 같은 저자원 언어에서 영어 데이터와의 공통성을 통해 훈련받은 기존 모델의 한계를 지적합니다. 특히, 저자원 언어에 대한 이해능력이 부족하다는 점을 강조하며, Typhoon-Audio라는 새로운 모델이 기존 오픈소스 모델들보다 우수한 성능을 보여줌을 입증했습니다.

- **Technical Details**: Typhoon-Audio 모델은 SALMONN 아키텍처를 기반으로 하며, 타겟 언어로 태국어와 영어를 설정하였습니다. Whisper-large-v3와 BEATS가 오디오 인코더 역할을 하며, Q-Former가 어댑터로 사용됩니다. 모델 훈련은 두 단계로 나누어 진행되며, 첫 단계에서는 오디오와 텍스트 표현을 정렬하기 위해 어댑터만 훈련되고, 두 번째 단계에서는 다양한 태스크를 통해 모델의 명령 수행 능력을 향상시킵니다.

- **Performance Highlights**: Typhoon-Audio 모델은 영어 및 태국어 모두에서 최신 기술의 Gemini-1.5-Pro와 비교해도 유사한 성능을 보이며, 기존 오픈소스 오디오 언어 모델에 비해 상당히 개선된 성능을 기록했습니다.



### Contextual Breach: Assessing the Robustness of Transformer-based QA Models (https://arxiv.org/abs/2409.10997)
- **What's New**: 적대적 (adversarial) 노이즈에 대한 질문-응답 모델의 취약성을 다룬 새로운 데이터셋을 소개합니다. 이 데이터셋은 SQuAD 데이터셋에 대해 7가지 유형의 적대적 노이즈를 각기 다른 5개 강도로 적용하였습니다.

- **Technical Details**: 우리는 robustness metrics를 사용하여 다양한 노이즈 유형과 강도에 따른 모델 성능 평가를 위한 표준화된 측정을 제공하였습니다. 실험은 transformer 기반 질문-응답 모델을 통해 이루어졌습니다.

- **Performance Highlights**: 실험 결과, 모델은 실제 텍스트 입력에서의 적대적 노이즈에 대해 다수의 취약점을 드러냈으며, 이는 모델 성능에 중요한 통찰을 제공합니다.



### Less is More: A Simple yet Effective Token Reduction Method for Efficient Multi-modal LLMs (https://arxiv.org/abs/2409.10994)
Comments:
          9 pages, 3 figures, 6 tables

- **What's New**: 이 연구는 Multimodal Large Language Models (MLLMs)의 자원 소모 문제를 해결하기 위해 새로운 접근 방식을 제안합니다. 이 방법은 CLIP metric을 활용하여 이미지 토큰을 선택하고 줄이는 Token Reduction using CLIP Metric (TRIM)입니다.

- **Technical Details**: TRIM은 Visual Question Answering (VQA) 작업에서의 인간 주의 패턴에 영감을 받아 이미지 토큰의 선택 및 감축을 위한 새로운 관점을 제시합니다. 이 방식은 12개의 데이터셋에서 광범위하게 테스트되었으며, 성능을 유지하면서 계산 오버헤드를 크게 줄이는 결과를 보여주었습니다. 또한, TRIM은 Interquartile Range (IQR) 스코어 기능을 사용하여 질문 응답에 중요한 이미지 토큰을 적응적으로 선택합니다.

- **Performance Highlights**: TRIM 방법은 약 79%의 이미지 토큰 수 감소, 처리 시간 67% 단축 및 메모리 사용량 30% 절감을 달성했습니다. 이 효율성은 원래 모델과 유사한 성능을 유지하면서 이루어졌습니다.



### GOSt-MT: A Knowledge Graph for Occupation-related Gender Biases in Machine Translation (https://arxiv.org/abs/2409.10989)
Comments:
          Accepted at the KG-STAR'24: Workshop on Knowledge Graphs for Responsible AI co-located with the 33rd ACM CIKM Conference, October 25, 2024, Boise, Idaho

- **What's New**: 이 논문은 직업과 관련된 성별 편향(gender bias)을 연구하기 위한 새로운 접근 방식인 GOSt-MT(Gender and Occupation Statistics for Machine Translation) 지식 그래프(Knowledge Graph)를 소개합니다.

- **Technical Details**: GOSt-MT는 실제 노동 data의 포괄적인 성별 통계와 MT 훈련(training)에 사용된 텍스트 코퍼스를 통합하여 생성되었습니다. 이 지식 그래프는 영어, 프랑스어, 그리스어에서 성별 편향을 상세히 분석할 수 있는 도구를 제공합니다.

- **Performance Highlights**: GOSt-MT는 노동 시장과 MT 시스템에서 직업이 성별화되는 방식을 이해하는 구조화된 프레임워크를 제공함으로써 MT 시스템을 더욱 공평하게 만들고 자동 번역에서 성별 편향을 줄이기 위한 노력을 지원합니다.



### Cross-lingual transfer of multilingual models on low resource African Languages (https://arxiv.org/abs/2409.10965)
- **What's New**: 이 연구는 높은 자원을 가진 언어에서 자원이 부족한 언어로의 Cross-lingual transfer(교차 언어 이전) 능력을 평가합니다. Kinyarwanda와 Kirundi라는 두 Bantu 언어를 중심으로 다중 언어 모델과 단일 언어 모델의 성능 비교를 다룹니다.

- **Technical Details**: 모델은 Multilingual BERT (mBERT), AfriBERT, BantuBERTa와 같은 transformer 기반 아키텍처와 BiGRU, CNN, char-CNN과 같은 신경망 기반 아키텍처가 포함됩니다. 모델은 Kinyarwanda에서 학습되고 Kirundi에서 테스트되며, fine-tuning(미세 조정)을 통해 성능 향상 정도와 catastrophic forgetting(재앙적 망각)을 평가합니다.

- **Performance Highlights**: AfriBERT는 fine-tuning 이후 88.3%의 교차 언어 정확도를 달성했으며, BiGRU는 83.3%의 정확도로 신경망 모델 중 가장 높은 성과를 보였습니다. 단일 언어 모델이 경쟁력을 유지하는 반면, 다중 언어 모델이 자원이 제한된 환경에서도 강력한 교차 언어 이전 능력을 제공함을 보여주었습니다.



### Investigating Context-Faithfulness in Large Language Models: The Roles of Memory Strength and Evidence Sty (https://arxiv.org/abs/2409.10955)
- **What's New**: 이번 연구는 Retrieval-Augmented Generation (RAG) 과정에서 Large Language Models (LLMs)의 문맥 충실성(context-faithfulness)을 탐구하고, 기억 강도(memory strength) 및 증거 제출 방식의 영향을 조사하였습니다.

- **Technical Details**: 연구에서는 LLM의 기억 강도를 양적으로 측정하기 위해 같은 질문의 다양한 패러프레이즈에 대한 LLM 응답의 이탈(divergence)을 평가하는 방법을 도입했습니다. 이에 따라 직접적( direct ) 및 간접적( indirect )으로 제공되는 증거의 다양한 스타일을 생성하여 그 효과를 평가했습니다.

- **Performance Highlights**: 연구 결과, LLM들은 강한 기억력을 가진 질문에 대해서는 내부 기억에 의존할 가능성이 높으며, 특히 GPT-4와 같은 대형 모델에서 두드러졌습니다. 또한 패러프레이즈된 증거를 제시할 경우 LLM의 외부 정보 수용성(receptiveness)이 크게 증가하여 단순 반복보다 효과적이라는 사실이 밝혀졌습니다.



### Propulsion: Steering LLM with Tiny Fine-Tuning (https://arxiv.org/abs/2409.10927)
Comments:
          26 pages, 11 figures

- **What's New**: 이 논문에서는 Propulsion이라는 새로운 파라미터 효율적인 미세 조정 (PEFT) 방법을 제안합니다. 이 방법은 특정 작업에 대한 성능을 최적화하면서 계산 오버헤드를 획기적으로 줄이는 것을 목표로 합니다. Propulsion은 모델의 파라미터를 변경하지 않고, 사전 훈련된 모델의 특정 차원을 선택적으로 다시 스케일링하여 작업 목표에 맞게 출력 예측을 유도합니다.

- **Technical Details**: Propulsion은 네트워크의 출력 벡터에서 소규모의 목표 지향적인 수정이 모델의 전체 행동에 큰 영향을 미칠 수 있다는 관찰에 기반합니다. 이 방법은 사전 훈련된 레이어에 경량의 학습 가능한 Propulsion 파라미터를 도입하여 미세 조정 중 업데이트되는 파라미터의 수를 최소화합니다. 이론적인 분석은 Neural Tangent Kernel (NTK) 이론을 통해 Propulsion이 전체 미세 조정의 성능을 훨씬 적은 수의 학습 가능한 파라미터로 근사할 수 있음을 보여줍니다.

- **Performance Highlights**: Propulsion은 355.3백만에서 단 0.086백만 파라미터로 파라미터 수를 10배 이상 줄이며, LoRA와 같은 표준 접근 방식과 비교하여 경쟁력 있는 성능을 유지합니다.



### Attention-Seeker: Dynamic Self-Attention Scoring for Unsupervised Keyphrase Extraction (https://arxiv.org/abs/2409.10907)
- **What's New**: Attention-Seeker라는 새로운 비지도 키프레이즈 추출 방법이 제안되었습니다. 이 방법은 대규모 언어 모델의 self-attention 맵을 이용하여 후보 구문들의 중요도를 평가합니다. 이전 모델들과는 달리 매개변수 수동 조정이 필요하지 않아 유용성이 향상되었습니다.

- **Technical Details**: Attention-Seeker는 self-attention 맵의 특정 구성요소(레이어, 헤드, 어텐션 벡터)를 통해 텍스트의 주요 주제에 집중합니다. 이 방식은 ALICE(The Attention Based Language Interpretation Comprehension) 모델을 기반으로 하여 입력 텍스트의 특성에 자동으로 적응하도록 설계되었습니다. 길이에 따라 단문은 개별 attention 벡터를 평가하고, 장문은 문서 세그먼트를 고려합니다.

- **Performance Highlights**: Attention-Seeker는 Inspec, SemEval2010, SemEval2017, Krapivin의 네 개 공적 데이터셋에서 실험되었으며, 매개변수 조정 없이도 대부분의 기준 모델을 초월하고 세 개의 데이터셋에서 최첨단 성능을 달성했습니다. 특히 긴 문서에서 키프레이즈 추출에 탁월했습니다.



### CREAM: Comparison-Based Reference-Free ELO-Ranked Automatic Evaluation for Meeting Summarization (https://arxiv.org/abs/2409.10883)
- **What's New**: 이 논문에서는 긴 맥락과 대화 기반의 회의 요약을 위한 새로운 평가 프레임워크 CREAM(Comparison-based Reference-Free Elo-Ranked Automatic Evaluation for Meeting Summarization)을 제안합니다. 이는 회의 요약 평가의 독특한 도전 과제를 해결하기 위해 설계되었습니다.

- **Technical Details**: CREAM은 체인 오브 띵크(Chain-of-Thought) 추론 및 키 사실 정렬을 결합하여 참조(reference) 없이 모델이 생성한 요약의 간결성(conciseness) 및 완전성(completeness)을 평가합니다. ELO 랭킹 시스템을 사용하여 다양한 모델 간의 품질을 비교하는 강력한 메커니즘을 제공합니다.

- **Performance Highlights**: 실험 결과, 기존 LLM 기반 평가기가 회의 요약에 효과적이지 않음을 보여주고, CREAM 프레임워크가 요약 평가에서 탁월한 성능을 발휘함을 입증합니다. 다양한 GPT 모델을 벤치마킹한 결과 GPT-4o가 완전성에서 뛰어난 성능을 보였고, GPT-4는 간결성에서 우수했으나, 모든 모델이 간결성과 완전성 간의 균형을 찾는 데 어려움을 겪었습니다.



### American Sign Language to Text Translation using Transformer and Seq2Seq with LSTM (https://arxiv.org/abs/2409.10874)
Comments:
          Submit on ICTIIA 2024

- **What's New**: 수화 번역(Sign language translation)에 관한 연구가 진행되어, Transformer 모델과 Sequence-to-Sequence(Seq2Seq) 모델의 성능 비교가 이루어졌습니다.

- **Technical Details**: 본 연구에서는 미국 수화(American Sign Language)를 텍스트로 번역하는 과정에서 Transformer와 Seq2Seq 모델의 성능을 비교하였습니다. 또한, Transformer에 Residual Long Short-Term Memory(Residual LSTM)를 추가하여 실험을 진행하였습니다.

- **Performance Highlights**: Transformer 모델은 Seq2Seq 모델에 비하여 BLEU Score 값이 28.14 증가하였으나, Residual LSTM을 추가한 경우 성능이 23.37% 저하되었습니다.



### Adaptive Large Language Models By Layerwise Attention Shortcuts (https://arxiv.org/abs/2409.10870)
Comments:
          6 pages, 3 figures

- **What's New**: 본 논문은 Transformer 아키텍처의 효율성을 높이기 위해, 최종 레이어가 중간 레이어에 관여하도록 하여 계산을 조정할 수 있는 Adaptive Computations 모델을 제안합니다. 이는 기존의 단순한 레이어 스택 방식 대신, Attention Mechanism을 통해 각 입력 토큰에 맞춰 계산을 최적화하는 방식을 도입합니다.

- **Technical Details**: 제안된 모델은 각 입력의 특징에 따라 깊이와 맥락에 적응하게 설계되었으며, 이를 통해 복잡한 self-attention 블록을 더 어려운 입력예측에 사용하고 더 간단한 예측은 표면적인 레이어에서 직접 이루어지도록 합니다. 연구에서는 LibriSpeech, text-8, Wiki-103 및 MAESTRO 데이터셋을 사용하였습니다.

- **Performance Highlights**: 제안된 아키텍처는 GPT 유사 모델에 비해, 다양한 데이터셋에서 뛰어난 성능을 보였으며, Attention Map을 통해 모델이 입력 토큰의 복잡성에 따라 적절한 깊이의 레이어에 관여하여 학습한다는 것을 입증했습니다.



### BAD: Bidirectional Auto-regressive Diffusion for Text-to-Motion Generation (https://arxiv.org/abs/2409.10847)
- **What's New**: 본 논문에서는 Autoregressive 모델과 mask 기반 generative 모델의 장점을 통합한 Bidirectional Autoregressive Diffusion (BAD) 프레임워크를 제안합니다. BAD는 자연스러운 시퀀스 구조를 유지하면서도 인과적 종속성을 강제하는 순열 기반의 손상 기술을 사용합니다.

- **Technical Details**: BAD 프레임워크는 두 단계로 구성됩니다. 첫 번째 단계에서는 Vector-Quantized Variational Autoencoders (VQ-VAEs)를 기반으로 하는 모션 토크나이저를 훈련하여 연속 모션 데이터를 이산 모션 토큰으로 변환하고, 두 번째 단계에서는 이러한 모션 토큰을 사용하여 Transformer 아키텍처를 훈련합니다. 이 과정에서 하이브리드 attention mask를 구축하여 각 토큰 간의 의존성을 결정합니다.

- **Performance Highlights**: BAD는 text-to-motion 생성에서 기존의 autoregressive 및 mask 기반 모델보다 우수한 성능을 보이며, HumanML3D 및 KIT-ML 데이터셋에서 Frechet Inception Distance (FID)를 개선하였습니다. 특히, BAD는 고급 모션 토크나이저를 사용하는 방법들과 비슷한 성능을 보여줍니다.



### ReXErr: Synthesizing Clinically Meaningful Errors in Diagnostic Radiology Reports (https://arxiv.org/abs/2409.10829)
- **What's New**: 본 논문에서는 의료 영상 해석 및 방사선 보고서 작성을 위한 새로운 방법론인 ReXErr을 소개하고 있습니다. ReXErr은 대형 언어 모델(Large Language Models)을 활용하여 흉부 X선 보고서 내에서 대표적인 오류를 생성합니다.

- **Technical Details**: ReXErr은 임상 타당성을 유지하면서 다양한 오류를 주입하기 위한 새로운 샘플링 기법을 사용합니다. 이 방법은 오류 카테고리를 정의하고, 인간 및 AI 생성 보고서에서 일반적으로 발생하는 실수를 포괄하는 데이터셋을 생성할 수 있도록 합니다.

- **Performance Highlights**: ReXErr는 오류 카테고리 전반에서 일관성을 보이며, 실제 상황에서 발견되는 오류와 유사한 오류를 생성할 수 있는 잠재력을 보여줍니다. 이 과정은 방사선 보고서의 품질과 신뢰성을 개선하는 데 기여할 수 있습니다.



### Model Tells Itself Where to Attend: Faithfulness Meets Automatic Attention Steering (https://arxiv.org/abs/2409.10790)
Comments:
          12 pages, 4 figures

- **What's New**: 이 논문에서는 AutoPASTA라는 새로운 방법을 제안하여 LLM의 주의 집중을 강화하고 중요한 맥락 정보를 효과적으로 활용하도록 합니다. 이는 주의 점수 조정을 통해 LLM이 중요한 정보에 더 잘 집중할 수 있도록 돕습니다.

- **Technical Details**: AutoPASTA는 (1) 주요 맥락 정보를 자동으로 식별하고, (2) 이 정보를 강조하여 성능을 향상시키는 접근 방식입니다. 이 방법은 기존의 prompting 방법 대신, 문맥과 질문에서 의미론적 임베딩을 사용하여 원래 문장을 매핑함으로써 입력의 길이를 줄이고 오류 전파를 완화하는 방식입니다.

- **Performance Highlights**: 실험 결과에 따르면, AutoPASTA는 LLAMA3-70B-Instruct 모델에서 평균 8.99%의 성능 향상을 달성했으며, 다양한 작업에서의 주의 헤드 집합이 우수한 일반화 능력을 보여주었습니다.



### Predicting Punctuation in Ancient Chinese Texts: A Multi-Layered LSTM and Attention-Based Approach (https://arxiv.org/abs/2409.10783)
- **What's New**: 본 논문에서는 고대 중국 텍스트에서 문장 부호를 예측하기 위한 새로운 접근 방식을 제안합니다. 이 방식은 Oh et al (2017)의 연구를 확장하여 다층의 양방향 LSTM과 멀티 헤드 주의 메커니즘을 활용합니다.

- **Technical Details**: 제안된 모델은 멀티 레이어 (multi-layered) LSTM을 기반으로 하며, 멀티 헤드 (multi-head) 주의 메커니즘을 통합하여 문장 부호의 위치와 유형을 예측합니다. 이 모델은 고대 중국 텍스트의 의미를 이해하는 데 도움을 줄 수 있도록 설계되었습니다.

- **Performance Highlights**: 제안된 접근 방식은 문장 부호가 없는 고대 중국 텍스트를 평가할 때 통상적인 RNN보다 훨씬 우수한 성능을 보였습니다. 다층 LSTM과 멀티 헤드 주의를 사용할 경우 RNN에서 발생하는 여러 한계를 극복할 수 있음을 확인했습니다.



### Semantics Preserving Emoji Recommendation with Large Language Models (https://arxiv.org/abs/2409.10760)
- **What's New**: 본 연구는 사용자 텍스트와 의미적 일관성을 유지하는 이모지 추천의 능력을 평가하는 새로운 평가 프레임워크를 제안합니다. 기존의 이모지 추천 방법과는 달리, 정확한 일치를 넘어 사용자의 감정, 인구통계적 프로필 및 태도가 변하지 않도록 추천된 이모지가 그 의미를 잘 유지하는지를 평가합니다.

- **Technical Details**: 새로운 의미 유지 평가 프레임워크는 다섯 개의 하위 분류 작업(감정 분석, 감정 분류, 태도 탐지, 연령 예측, 성별 예측)을 통해 모델의 성능을 평가합니다. 대형 언어 모델(LLMs)인 GPT-4o는 79.23%의 의미 유지 점수를 기록하며 다른 LLMs보다 성능이 우수한 것으로 나타났습니다. 이 연구에서는 다양한 프롬프트 기법을 사용하여 여섯 개의 LLM의 성능을 체계적으로 평가하였습니다.

- **Performance Highlights**: GPT-4o 모델은 데이터셋의 의미 유지 점수에서 79.23%를 달성하며, 이모지 추천 작업에서 다른 모델에 비해 우수한 성능을 보였습니다. 또한, 모델의 바이어스 분석 및 추천된 이모지의 다양성 평가를 위한 사례 연구도 수행되었습니다.



### Generalized Measures of Anticipation and Responsivity in Online Language Processing (https://arxiv.org/abs/2409.10728)
- **What's New**: 이 논문에서는 온라인 언어 처리에서 예측 불확실성을 측정하기 위한 고전적인 정보 이론적 방법을 일반화하였습니다. 새로운 예상 연속성의 시뮬레이션을 기반으로 하는 이 접근법은 반응적(responsive) 및 예측적(anticipatory) 측정을 정의하며, 기존의 next-symbol entropy 및 surprisal보다 더 표현력 있는 측정을 정의하는 도구를 제공합니다.

- **Technical Details**: 이 연구의 프레임워크는 대칭화된 변수에 대한 일반화된 surprisal을 도출하며, 이는 언어적 맥락의 연속에 대한 기대와 일치합니다. 이 프레임워크는 기존 정보 이론적 측정을 특별한 경우로 포함하며, sequence-level entropy와 next-symbol information value와 같은 새로운 특별한 경우를 제안합니다. 몬테 카를로 시뮬레이션을 사용하여 다양한 반응적 및 예측적 측정을 추정하며, 이 과정에서 런타임(runtime)과 변동성(variance) 간의 트레이드오프를 분석합니다.

- **Performance Highlights**: 연구 결과, (1) 문맥 확률이 인간의 cloze completion을 surprisal보다 더 잘 예측하며, (2) 정보 값이 N400을 더 잘 예측하고, (3) 본 논문에서 제안한 sequence-level entropy가 ELAN의 유일한 유의미한 예측자라는 사실이 밝혀졌습니다. 또한, (4) 다양한 반응적 측정이 자극 시작 후 여러 시간대의 ERP 진폭을 예측하는 데 차별화된 성능을 보였습니다.



### Self-Attention Limits Working Memory Capacity of Transformer-Based Models (https://arxiv.org/abs/2409.10715)
Comments:
          8 pages, 12 figures

- **What's New**: 최근 연구를 통해 Transformer 기반의 대규모 언어 모델(LLMs)의 작업 기억 용량에 한계가 있음을 발견했습니다. 이는 N-back 작업의 성과가 N이 증가함에 따라 유의미하게 감소한다는 점에서 인간의 기억 한계와 유사합니다. 하지만 이러한 현상의 기계적 해석이 부족합니다.

- **Technical Details**: 본 연구에서는 Transformer의 self-attention 메커니즘이 작업 기억 용량 한계의 원인일 수 있다는 가설을 세우고 이를 검증하기 위해 vanilla decoder-only transformers를 훈련시켰습니다. 실험 결과, attention score가 훈련 과정에서 점차 N-back 위치로 집계되는 현상이 관찰되었습니다. 이는 모델이 현재 위치와 N-back 위치 간의 관계에 주의하기 위한 전략을 학습하고 있다는 것을 시사합니다.

- **Performance Highlights**: 모델의 N-back 작업 수행 능력은 N이 증가함에 따라 감소하였으며, 각 위치의 예측 정확도는 attention score와 긍정적 상관관계를 보였습니다. 특히, attention score 행렬의 총 엔트로피는 N이 증가함에 따라 증가했으며, 이는 attention score의 분산이 N-back 작업에서의 용량 한계의 원인일 수 있음을 나타냅니다.



### Visualizing Temporal Topic Embeddings with a Compass (https://arxiv.org/abs/2409.10649)
Comments:
          11 pages, 9 figures, conference paper

- **What's New**: 이 논문은 기존의 Dynamic Topic Modeling(DTM) 방식의 한계를 극복하기 위해 새로운 방법론인 Temporal Topic Embeddings with a Compass(TTEC)를 제안합니다. 이 방법은 단어 사용의 변화와 문서의 컨텍스트를 통합하여 시계열 분석을 가능하게 하는 시각화 기법을 제공합니다.

- **Technical Details**: TTEC는 compass-aligned temporal Word2Vec 방식을 확장하여 동적 주제 모델링에 적용하며, 전체 단어 및 문서의 집합에서 글로벌 임베딩 공간과 각 시간 구간별 로컬 임베딩 공간을 생성합니다. 이로 인해 단어와 문서의 임베딩을 동일한 주제 임베딩 공간 내에서 비교할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 다양한 크기의 시계열 데이터셋에서 주제의 관련성과 다양성에 대해 경쟁력 있는 성능을 발휘하며, 또한 글로벌 주제의 진화와 동시적인 단어 임베딩의 변화를 시각적으로 분석할 수 있는 통찰력 있는 시각화를 제공합니다.



### Improving Multi-candidate Speculative Decoding (https://arxiv.org/abs/2409.10644)
- **What's New**: 이번 논문에서는 Multi-Candidate Speculative Decoding (MCSD)의 효율성을 개선하기 위한 새로운 접근 방식을 제안합니다. 이 방법은 대상 모델(target model)을 초기화한 멀티-후보 프로세스, 동적 슬라이스 토폴로지 인지 인과 마스크(dynamic sliced topology-aware causal mask)을 추가하여 동적 길이 조정을 가능하게 하며, 조기 중단을 최적화하기 위한 의사 결정 모델을 포함합니다.

- **Technical Details**: 제안된 방법은 대상 모델 초기화 멀티-후보 토큰 트리(target model initialized multi-candidate token tree), 동적 길이 조정을 위한 동적 슬라이스 토폴로지 인지 인과 마스크(dynamic sliced topology-aware causal mask), 그리고 각 드래프트 생성 단계에서의 동적인 조기 중단을 결정하는 의사 결정 모델(Decision Model)을 활용합니다.

- **Performance Highlights**: 이 프레임워크는 승인율(acceptance rate)을 최대 164% 증가시키고 MCSD 기반 대비 최대 75%의 속도 개선을 보이며, 평균적으로는 승인율을 40% 향상시키고 23%의 생성 시간 단축을 달성했습니다.



### Exploring Fine-tuned Generative Models for Keyphrase Selection: A Case Study for Russian (https://arxiv.org/abs/2409.10640)
- **What's New**: 이 연구는 러시아 학술 텍스트 내의 키프레이즈 선택(task of keyphrase selection)을 위해 세밀하게 조정된 생성적 변환 모델(generative transformer-based models)의 적용 방안을 탐구하였습니다. 이를 통해 다양한 분야에서의 성과를 평가하였으며, 특히 mBART 모델이 러시아어 키프레이즈 추출 기준선(baselines) 대비 성과 향상을 보여주었습니다.

- **Technical Details**: 연구에서는 ruT5, ruGPT, mT5, mBART의 네 가지 생성적 모델을 사용했습니다. 이들 모델은 한국 수학 및 컴퓨터 과학, 역사, 의학, 언어학 분야의 러시아어 과학 초록(texts of Russian scientific abstracts)을 분석하는 데 적용되었습니다. 키프레이즈 선택 영역에서 딥 신경망 모델(deep neural models)을 활용하여, 전통적인 비지도 학습 방식의 한계를 극복하고 싶은 목표를 가지고 있습니다.

- **Performance Highlights**: mBART 모델을 사용한 결과, 키프레이즈 선택의 in-domain 성과가 BERTScore에서 최대 4.9%, ROUGE-1에서 9.0%, F1-score에서 12.2% 향상되었으며, cross-domain에서도 몇 가지 경우에 대해 기준 성과를 초과하는 긍정적인 결과를 보였습니다. 이는 러시아어 키프레이즈 선택 기술의 추가 탐색 및 개선 가능성을 부각시킵니다.



### Language Models and Retrieval Augmented Generation for Automated Structured Data Extraction from Diagnostic Reports (https://arxiv.org/abs/2409.10576)
- **What's New**: 이 논문은 비구조화된 방사선 및 병리학 보고서에서 구조화된 임상 정보를 자동으로 추출하기 위한 시스템을 개발하고 평가한 내용을 담고 있습니다. 특히 open-weights large language models (LMs)와 retrieval augmented generation (RAG) 기술을 활용하였습니다.

- **Technical Details**: 이 연구는 두 개의 데이터셋을 사용하였습니다: 7,294개의 방사선 보고서(BT-RADS 점수 주석 포함)와 2,154개의 병리학 보고서(IDH 변이 상태 주석 포함). 다양한 LMs와 RAG 구성을 평가하기 위한 자동화된 파이프라인이 개발되었으며, 모델 크기, 양자화(quantization), 프롬프트 전략(prompting strategies), 출력 형식(output formatting), 추론 매개변수(inference parameters) 등의 변수들이 성능에 미치는 영향을 체계적으로 분석하였습니다.

- **Performance Highlights**: 최고 성능 모델은 방사선 보고서에서 BT-RADS 점수를 98% 이상 정확도로 추출하였으며, 병리학 보고서에서 IDH 변이 상태 추출에 대해 90% 이상의 정확도를 기록하였습니다. 성능이 가장 뛰어난 모델은 medical fine-tuned llama3로, 크고 최신의 도메인 맞춤형 모델들이 이전 모델들보다 일관되게 성능이 우수했습니다. 양자화는 성능에 미치는 영향이 미미했으며, few-shot prompting이 정확도를 크게 향상시켰습니다. RAG는 복잡한 병리학 보고서의 성능을 개선했지만 짧은 방사선 보고서에는 효과가 없었습니다.



### EIA: Environmental Injection Attack on Generalist Web Agents for Privacy Leakag (https://arxiv.org/abs/2409.11295)
Comments:
          24 pages

- **What's New**: 이번 연구에서는 일반화된 웹 에이전트(Generalist web agents)의 프라이버시 리스크를 탐구하며, 이를 위한 최초의 연구를 진행했습니다. 특히, 적대적 환경에서의 에이전트의 안전성을 고찰합니다.

- **Technical Details**: 우리는 두 가지 유형의 적대적 목표를 고려했습니다: 사용자의 특정 개인 식별 정보(PII)를 탈취하거나, 전체 사용자 요청을 도용하는 것입니다. 이를 위해 Environmental Injection Attack (EIA)이라는 새로운 공격 방법을 제안하며, 이는 에이전트가 의도치 않은 행동을 하도록 유도하는 악성 콘텐츠를 주입합니다. 이 공격은 정보 유출을 유도하기 위해 설계된 악성 웹 요소를 삽입합니다.

- **Performance Highlights**: EIA는 최대 70%의 ASR(Attack Success Rate)으로 사용자의 특정 PII를 탈취할 수 있었으며, 전체 사용자 요청을 도용하는 것은 더 도전적이지만 완화된 버전의 EIA는 여전히 16% ASR을 달성했습니다. 이러한 결과는 높은 자율성과 보안 간의 상충관계를 강조합니다.



### P-RAG: Progressive Retrieval Augmented Generation For Planning on Embodied Everyday Task (https://arxiv.org/abs/2409.11279)
- **What's New**: 이 논문에서는 Embodied Everyday Task를 위한 새로운 접근법인 Progressive Retrieval Augmented Generation (P-RAG)을 제안합니다. 기존의 Large Language Model (LLM) 기반 접근법의 한계를 극복하고, 자연어 지시를 기반으로 하여 임베디드 AI 환경에서의 작업 수행 능력을 개선하는 데 초점을 맞추었습니다.

- **Technical Details**: P-RAG는 자연어 지시와 시각 관찰을 기반으로 하는 작업에서, ground-truth 데이터 없이도 작업에 대한 특정 지식을 점진적으로 축적하는 데 사용됩니다. 기존 RAG 방법과는 달리 P-RAG는 반복적 접근법을 도입하여 데이터베이스를 점진적으로 업데이트하고, 각 반복에서 최신 데이터베이스를 조회하여 경험적으로 참고할 수 있는 역사적 정보를 얻습니다.

- **Performance Highlights**: P-RAG는 기존 방법들과 비교하여 몇 가지 성과를 이룬 것으로 나타났습니다. 특히 몇 번의 샘플로 구성된 학습 환경에서도 우수한 성능을 보였으며, 자기 반복을 통해 성능을 further 향상시키는 기능을 입증하였습니다.



### Bio-Inspired Mamba: Temporal Locality and Bioplausible Learning in Selective State Space Models (https://arxiv.org/abs/2409.11263)
Comments:
          17 pages, 1 figure, 2 tables

- **What's New**: 이 논문은 생물학적 학습 원리를 통합하여 선택적 상태 공간 모델을 위한 온라인 학습 프레임워크인 Bio-Inspired Mamba (BIM)를 소개합니다. BIM은 실시간 반복 학습(Real-Time Recurrent Learning, RTRL)과 스파이크 타이밍 의존 가소성(Spike-Timing-Dependent Plasticity, STDP)과 유사한 지역 학습 규칙을 결합하여 스파이킹 신경망의 학습에서 발생하는 시간적 지역성 및 생물학적 타당성 문제를 해결합니다.

- **Technical Details**: BIM은 선택적 상태 공간 모델을 기반으로 하며, 입력 데이터에 따라 정보를 동적으로 전파하거나 잊을 수 있는 메커니즘을 제공합니다. RTRL은 네트워크의 현재 상태와 입력 데이터에 따라 각 시간 단계에서 업데이트를 수행하며, STDP는 시냅스 강도를 조정하여 생물학적 신경망의 학습 및 기억 형성에 중요한 역할을 합니다.

- **Performance Highlights**: BIM은 언어 모델링, 음성 인식 및 생물 의학 신호 분석 작업에서 평가되었으며, 전통적인 방법에 비해 경쟁력 있는 성능을 보여주었습니다. 결과는 에너지 효율성의 개선과 신경모방 하드웨어(Neuromorphic hardware) 구현의 잠재력을 보여줍니다.



### Capturing Differences in Character Representations Between Communities: An Initial Study with Fandom (https://arxiv.org/abs/2409.11170)
Comments:
          Accepted and presented as a working paper in SBP-BRiMS 2024

- **What's New**: 이 연구는 온라인 커뮤니티 내에서 서사(narrative) 서사 세계의 중요한 부분인 캐릭터(character) 재해석에 대한 새로운 방법론을 제시합니다.

- **Technical Details**: 해리 포터(Harry Potter) 소설, r/HarryPotter subreddit 및 Archive of Our Own의 팬픽션을 사용하여 캐릭터 멘션 및 공존 네트워크(co-occurrence networks)에서 중심성(centrality) 측정을 분석하였습니다. 이러한 분석을 통해 캐릭터의 언급 변화 및 의미적 연관성(semantic associations)에 대한 변화를 연구하였습니다.

- **Performance Highlights**: 팬픽션의 남성 캐릭터 분석 시, 남남 로맨스에서 여성성(femininity) 특성이 더 높게 평가된 결과가 과거의 질적 연구를 뒷받침하였습니다. 이러한 발견은 온라인 팬덤(fandom) 내의 캐릭터 재개념화(re-conceptualization) 및 질적 연구를 지원하는 데 있어 컴퓨터 방법의 잠재력을 강조합니다.



### ISO: Overlap of Computation and Communication within Seqenence For LLM Inferenc (https://arxiv.org/abs/2409.11155)
- **What's New**: 본 논문에서는 대형 언어 모델(LLM) 추론의 효율성을 높이기 위해 새로운 계산-통신 겹침(overlap) 전략을 제안합니다. 이 전략은 시퀀스(sequence) 수준에서 작동하며, 계산과 통신 과정을 최적화하여 기존 방법보다 뛰어난 성능을 보여줍니다.

- **Technical Details**: 제안된 ISO(인트라 시퀀스 오버랩) 방법은 시퀀스를 두 부분으로 나누어 순서 의존적인 계산(어텐션 부분)을 우선 처리한 후 비순서 의존적인 부분에 접근합니다. 이 과정에서 두 마이크로 배치(micro-batch) 간의 겹침을 촉진하여 효율성을 개선합니다.

- **Performance Highlights**: 실험 결과, 30B 및 70B 모델을 사용하여 4090 GPU에서는 약 35%, A800 GPU에서는 약 15%의 시간 소모 감소를 확인했습니다. 이는 LLM 추론의 사전 채우기(prefill) 단계에서의 효율성을 크게 향상시키는 결과입니다.



### Promptriever: Instruction-Trained Retrievers Can Be Prompted Like Language Models (https://arxiv.org/abs/2409.11136)
- **What's New**: 이 논문에서는 언어 모델(LM)처럼 프롬프트(prompts)를 사용할 수 있는 최초의 검색 모델인 Promptriever를 소개합니다. Promptriever는 50만 개의 인스턴스를 포함하는 새로운 인스턴스 수준 지침(training set)을 기반으로 훈련되었습니다.

- **Technical Details**: Promptretriever는 Bi-encoder 구조를 사용하며, LLaMA-2 7B와 같은 대형 언어 모델을 백본으로 활용합니다. 이는 자연어 프롬프트를 통해 쿼리의 연관성을 동적으로 조정하여 정보 검색을 가능하게 합니다.

- **Performance Highlights**: Promptriever는 다음과 같은 성능 향상을 보입니다: (1) 상세한 연관성 지침을 따랐을 때 14.3의 p-MRR과 3.1 nDCG 증가, (2) 쿼리와 지침의 어휘 선택에 대한 강건성(robustness) 증가, (3) 프롬프트를 통해 하이퍼파라미터 검색이 가능하여 검색 성능을 1.4 포인트 향상시킴.



### Improving Speech Emotion Recognition in Under-Resourced Languages via Speech-to-Speech Translation with Bootstrapping Data Selection (https://arxiv.org/abs/2409.10985)
Comments:
          5 pages, 2 figures, Submitted to ICASSP 2025

- **What's New**: 이번 논문에서는 고급 리소스를 가진 언어의 데이터를 활용하여 저자원 언어에서의 Speech Emotion Recognition (SER) 성능을 향상시키기 위한 새로운 접근 방식을 제안합니다. 이 방법은 Speech-to-Speech Translation (S2ST)을 사용하여 타겟 언어 데이터의 레이블이 부여된 데이터를 생성합니다.

- **Technical Details**: 논문에서는 고급 리소스를 가진 언어에서 생성된 합성 데이터를 사용하여 저자원 언어에서 SER 성능을 높이는 방법을 제안합니다. 방법론적으로는 두 단계의 파이프라인을 사용: 데이터 생성과 부트스트래핑 데이터 선택. S2ST 모델을 이용해 타겟 언어 데이터를 생성하고, 부트스트래핑 방식을 통해 이전 모델의 예측을 바탕으로 유용한 데이터를 반복적으로 선택합니다.

- **Performance Highlights**: 대규모 실험 결과, 제안된 방법은 다양한 모델과 언어에서 일관되게 성능을 개선하는 것으로 나타났습니다. 이 연구는 저자원 언어에서 SER의 성능을 향상시킬 수 있는 가능성을 제시하며, 더 확장 가능하고 견고한 다국어 SER 시스템의 개발을 촉진할 수 있음을 보여줍니다.



### Enhancing Multilingual Speech Generation and Recognition Abilities in LLMs with Constructed Code-switched Data (https://arxiv.org/abs/2409.10969)
Comments:
          Submitted to ICASSP 2025

- **What's New**: 본 논문에서는 기존의 단일 언어 모델의 한계를 극복하고 다국어 및 코드 스위칭 (code-switched) 상황에서의 음성 생성 및 인식 작업을 통합한 MultiLingual MultiTask (MLMT) 모델을 제안합니다. 이 모델은 여러 언어의 음성을 동시에 처리하는 멀티링궐 토크나이저를 활용하여, 고품질의 CS 데이터 없이도 코드 스위칭 합성 능력을 제공합니다.

- **Technical Details**: MLMT 모델은 대규모 라벨 없는 음성을 활용하여 Self-Supervised Learning (SSL) 기법을 통해 추출된 표현을 기반으로 하며, ASR(Automatic Speech Recognition)과 TTS(Text-to-Speech) 작업을 통합하였습니다. 모델은 LLaMA 3 8B를 백본으로 사용하여 30개 이상의 언어에서 우수한 성능을 발휘하도록 설계되었습니다. 데이터 구성 방법론으로는 다국어 말하기 데이터를 분리 및 연결하는 접근 방식을 사용하며, 이로 인해 고품질 CS 데이터에 대한 의존성을 줄입니다.

- **Performance Highlights**: 실험 결과, MLMT 모델은 기존의 모델들과 비교했을 때 음성 생성 및 인식 작업에서 더 나은 성능을 기록했으며, 사용자 수 및 객관적 실험 모두에서 우수한 결과를 보였습니다. 특히, 멀티링궐 음성 생성 및 인식 작업에 있어 LLM의 성능을 개선하였고, CS TTS 작업의 일관된 화자 유사성을 유지하면서 CS 합성 능력을 강화했습니다.



### GenCRF: Generative Clustering and Reformulation Framework for Enhanced Intent-Driven Information Retrieva (https://arxiv.org/abs/2409.10909)
- **What's New**: GenCRF(Generative Clustering and Reformulation Framework)가 정보 검색 분야에서 사용자 쿼리의 다양한 의도를 포착하기 위해 처음으로 도입되었습니다. 이 프레임워크는 Large Language Models(LLMs)를 활용하여 다수의 맞춤형 프롬프트를 사용해 여러 쿼리를 생성하고 클러스터링하여 쿼리 성공률을 향상시키는 목표를 가지고 있습니다.

- **Technical Details**: GenCRF는 LLM을 통해 초기 쿼리에서 파생된 다양한 쿼리를 생성하고, 이들을 동적으로 클러스터링하여 정보 중복을 최소화합니다. 쿼리 통합을 위해 유사성 기반 동적 가중치 및 점수 기반 동적 가중치를 포함한 여러 가중 집계 전략을 적용하며, Query Evaluation Rewarding Model(QERM)을 통해 쿼리 개선 프로세스를 피드백 루프를 통해 정교화합니다.

- **Performance Highlights**: BEIR 데이터세트에서 실시한 실험을 통해 GenCRF는 기존 쿼리 재구성 기술 대비 최대 12% 향상된 nDCG@10 성능을 기록하며, 다양한 도메인과 쿼리 유형에 대해 지속적으로 우수한 성능을 입증하였습니다.



### NaviQAte: Functionality-Guided Web Application Navigation (https://arxiv.org/abs/2409.10741)
- **What's New**: NaviQAte는 웹 애플리케이션 탐색을 질문-응답(task) 작업으로 재구성하여 세부 파라미터 없이 기능에 대한 동작 시퀀스를 생성합니다. 이는 현재의 웹 자동화 테스트 방법의 한계점을 극복하기 위한 접근 방식입니다.

- **Technical Details**: NaviQAte는 세 가지 주요 단계인 행동 계획, 선택 추출 및 의사 결정을 사용하여 기능 중심의 웹 탐색을 실현합니다. 이를 위해 대형 언어 모델(GPT-4o 등)과 비용 효율적인 모델(GPT-4o mini 등)을 통합하여 복잡한 결정 및 간단한 작업을 처리합니다. 다양한 텍스트 및 이미지 입력을 통합하여 문맥 이해를 향상시킵니다.

- **Performance Highlights**: NaviQAte는 Mind2Web-Live 및 Mind2Web-Live-Abstracted 데이터셋에 대한 평가에서 각각 44.23%와 38.46%의 사용자 작업 탐색 및 기능 탐색 성공률을 기록하였으며, WebCanvas에 비해 각각 15% 및 33%의 성능 개선을 보였습니다.



### Self-supervised Speech Models for Word-Level Stuttered Speech Detection (https://arxiv.org/abs/2409.10704)
Comments:
          Accepted by IEEE SLT 2024

- **What's New**: 본 연구에서는 단어 수준의 더듬이(스퍼트) 탐지를 위한 모델을 제안하며, 이 과정에서 자가 지도(Self-Supervised) 음성 모델을 활용하였습니다. 이는 더듬이 장애를 가진 환자들의 자동화된 스크리닝을 가능하게 합니다.

- **Technical Details**: 연구팀은 단어 수준의 더듬이 탐지 모델을 위해 WavLM(자기지도 학습 기반 음성 모델)을 이용하고, LibriSpeech 데이터를 합성 불유창(Disfluency) 증강법으로 사전 훈련한 후 SEP-28K 데이터셋으로 미세 조정(fine-tune) 하였습니다. 모델 구조는 계층적 합성곱 인터페이스(Hierarchical Convolution Interface)를 활용하여 여러 층에서 정보를 집계합니다.

- **Performance Highlights**: 제안된 모델은 단어 수준의 더듬이 탐지에서 이전 연구를 크게 초과하는 성능을 보였습니다. 또한, 더듬이 장애 탐지에 있어 자가 지도 음성 모델의 효과에 대한 포괄적인 실험 분석을 수행하였습니다.



### Model-in-the-Loop (MILO): Accelerating Multimodal AI Data Annotation with LLMs (https://arxiv.org/abs/2409.10702)
- **What's New**: AI 훈련 데이터에 대한 수요 증가로 인해 데이터 주석(annotation) 산업이 글로벌 산업으로 성장하고 있습니다. 기존의 인간 주석자에 의존한 접근 방식은 시간 소모가 크고 노동 집약적이며 일관된 품질을 보장하기 어렵습니다. 이를 해결하기 위해 제안된 MILO(Model-in-the-Loop) 프레임워크는 AI/ML 모델을 주석 과정에 통합합니다.

- **Technical Details**: MILO 프레임워크는 전문 인간 주석자와 대규모 언어 모델(LLM)의 강점을 활용하는 협업 패러다임을 도입합니다. LLM을 사전 주석(pre-annotation) 및 실시간 어시스턴트(real-time assistants)로 사용하고, 주석자 응답에 대한 심사위원(judges) 역할을 수행함으로써, 인간 주석자와 LLM 간의 효과적인 상호작용 패턴을 가능하게 합니다. 세 가지 실증 연구(empirical studies)를 통해 다중 모달(multimodal) 데이터 주석에서 MILO의 효과를 입증하였습니다.

- **Performance Highlights**: MILO는 처리 시간을 줄이고(data handling time), 데이터 품질을 향상시키며(quality), 주석자 경험을 개선하는 성과를 보여주었습니다. 또한 유연한 평가(quality rubrics) 및 열린 주석(open-ended annotations)에 대한 세분화된 피드백(fine-grained feedback)을 제공합니다. MILO 프레임워크는 AI/ML 개발을 가속화하고, 인간 주석에 대한 의존도를 줄이며, 인간과 기계 가치 간의 더 나은 정렬(alignment)을 촉진하는 데 기여할 수 있습니다.



### A Bayesian Interpretation of Adaptive Low-Rank Adaptation (https://arxiv.org/abs/2409.10673)
- **What's New**: 이번 연구는 민감도 기반의 중요도 점수를 사용하는 AdaLoRA의 한계를 극복하기 위해 이론적으로 지지받는 신호 대 잡음비(Signal-to-Noise Ratio, SNR)와 Improved Variational Online Newton (IVON) 옵티마이저를 활용하여 파라미터 예산 할당을 수행하는 Bayesian 접근을 제안합니다. 이 방법은 기존의 중요도 점수를 대체함으로써 더욱 향상된 성능과 빠른 계산 속도를 자랑합니다.

- **Technical Details**: 이 연구에서는 Bayesian Neural Networks (BNNs)에서 유래된 이론적으로 강력한 중요도 메트릭인 SNR과 IVON 옵티마이저를 결합하여 AdaLoRA의 Bayesian 버전을 개발했습니다. SNR은 각 파라미터의 중요도를 측정하기 위해 사용되며, SNR이 낮은 경우는 조정 중에 프루닝(pruning)하여 동적으로 랭크를 조절합니다. 또한, 민감도 점수와 SNR 간의 강한 연결고리를 이론적으로 제시하고 있습니다.

- **Performance Highlights**: 연구 결과, 제안된 Bayesian 접근 방식은 GLUE 벤치마크에서 민감도 기반 중요도 메트릭과 비교하여 동등하거나 우수한 성능을 달성했으며, Adam과 함께 사용하는 기존 AdaLoRA에 비해 10% 빠른 속도를 보였습니다. 이 연구는 파라미터의 중요성을 정량화하는 데 있어 진폭(magnitude)이 분산(variance)보다 더 중요한 지표임을 제안합니다.



### CSKV: Training-Efficient Channel Shrinking for KV Cache in Long-Context Scenarios (https://arxiv.org/abs/2409.10593)
- **What's New**: 이 논문에서는 KV 캐시의 메모리 오버헤드를 줄이기 위한 새로운 기법인 CSKV(채널 축소 기법)를 소개합니다. 기존의 KV 캐시 압축 방법이 가지고 있는 부족한 성능과 높은 훈련 비용 문제를 해결하려는 접근법입니다.

- **Technical Details**: CSKV는 (1) 키와 값 레이어의 특이값 분포를 분석하여 채널 차원에서 큰 중복성과 압축 가능성을 발견합니다. (2) 이 구조에서는 윈도우 기반의 전정밀 KV 캐시와 저정밀 압축 KV 캐시를 결합한 이중 분기 KV 캐시를 도입합니다. (3) 전체 LLM 재훈련 없이, 층별 재구성 손실을 최소화하여 훈련 비용을 절감합니다.

- **Performance Highlights**: CSKV는 KV 캐시의 메모리 오버헤드를 80%까지 줄이면서 모델의 장기 컨텍스트 능력을 유지할 수 있습니다. 또한 4비트 양자화와 결합하여 최대 95%의 압축 비율을 달성하는 향상된 성능을 보여줍니다.



### Eureka: Evaluating and Understanding Large Foundation Models (https://arxiv.org/abs/2409.10566)
- **What's New**: 이 논문에서는 AI의 평가 프로세스를 개선하기 위한 새로운 프레임워크인 Eureka를 소개합니다. 이 프레임워크는 대형 기초 모델(Large Foundation Models)에 대한 더 나은 평가를 가능하게 하며, 단일 점수 보고와 순위 매김을 넘어서 다양한 모델의 능력을 비교하는 데 초점을 둡니다.

- **Technical Details**: Eureka는 모델 평가를 위한 유연한 라이브러리를 제공하여 데이터 전처리, 프롬프트 템플릿, 모델 추론, 데이터 사후 처리, 메트릭 계산 및 보고 작업을 사용자 맞춤형으로 조합 가능하게 합니다. 또한, Eureka-Bench는 사전 정의된 벤치마크 모음을 갖추고 있어 기존의 평가 방법이 간과하는 언어 및 다중 모드 능력을 테스트할 수 있도록 설계되었습니다.

- **Performance Highlights**: Eureka-Bench의 평가 결과, Claude 3.5 Sonnet, GPT-4o 2024-05-13, Llama 3.1 405B와 같은 특정 모델들이 여러 능력에서 반복적으로 다른 모델들보다 우수한 성능을 보였으나, 전반적으로는 어떤 단일 모델이 모든 작업에서 최선의 성능을 내지 않음을 보여줍니다. 특히, 현대 모델들은 이미지에 대한 세부 이해와 같은 기본 능력에서 여전히 한계를 보이고 있습니다.



### Unveiling Induction Heads: Provable Training Dynamics and Feature Learning in Transformers (https://arxiv.org/abs/2409.10559)
Comments:
          100 pages, 10 figures

- **What's New**: 이 연구는 인-context 학습(In-context Learning, ICL)의 이론적 기초를 탐구하며, 기존의 연구들이 주로 주의(attention) 메커니즘만을 다루는 반면, 이 논문은 transformer 아키텍처의 모든 구성 요소가 ICL에 어떻게 기여하는지를 분석합니다.

- **Technical Details**: 이 논문에서는 2개 주의 층(attention layer)을 가진 transformer 모델을 사용하여 n-gram Markov chain 데이터에 대한 ICL을 수행하는 방법을 연구합니다. 상대적 위치 임베딩(relative positional embedding), 다중 헤드 소프트맥스 주의(multi-head softmax attention), 정규화가 적용된 피드포워드 층(feed-forward layer)에 대한 논의가 포함되어 있습니다. 또한 교차 엔트로피 ICL 손실(cross-entropy ICL loss)과 관련된 그래디언트 흐름(gradient flow)이 수렴하는 방식도 증명합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델이 모든 구성 요소의 조화로운 기여로 인해 일반화된 induction head 메커니즘(generalized version of induction head mechanism)을 효율적으로 수행함을 입증하였습니다. 첫 번째 주의 층은 과거 토큰을 복사하는 역할을 하고(feed-forward 네트워크는 중요한 부모로부터 feature vector를 생성하는 선택자 역할을 하며), 두 번째 주의 층은 생성된 feature와 출력 위치의 feature를 비교하여 원하는 출력을 생성하는 분류기(classifier) 역할을 합니다.



### Agentic Society: Merging skeleton from real world and texture from Large Language Mod (https://arxiv.org/abs/2409.10550)
Comments:
          16 pages, 5 figures and 4 tables

- **What's New**: 최근 대규모 언어 모델(LLMs)과 에이전트 기술의 발전이 사회 과학 실험의 시뮬레이션을 위한 유망한 솔루션을 제공하고 있지만, 많은 모델들이 요구하는 실제 인구 데이터의 가용성이 여전히 주요 도전 과제가 되고 있습니다. 이 논문에서는 인구 조사 데이터를 활용하여 가상 인구를 생성하는 새로운 프레임워크를 탐구합니다.

- **Technical Details**: 이 접근법은 먼저 인구의 인구 통계적 특성을 반영한 페르소나(persona)를 생성합니다. 이어서 LLMs를 사용하여 이러한 페르소나에 세밀한 정보를 추가하며, 이는 이미지 생성 모델에서의 기술을 텍스트 데이터에 적용한 것입니다. 또한, 빅파이브(Big Five) 성격 특성 테스트를 기반으로 LLMs의 능력에 대한 방법의 실행 가능성을 평가하는 프레임워크를 제안합니다.

- **Performance Highlights**: 예비 실험과 분석을 통해 이 방법이 사회 과학 실험에서 다양한 인간 행동을 시뮬레이션하는 데 필수적인 변동성을 가진 페르소나를 생성함을 보여주지만, 현행 LLMs의 제한된 능력으로 인해 통계적 진실성이 약하게만 나타났습니다. 연구의 통찰력은 LLMs가 인간의 가치와 현실 세계의 복잡성을 반영하는 것 사이의 긴장감을 강조합니다.



### SAM4MLLM: Enhance Multi-Modal Large Language Model for Referring Expression Segmentation (https://arxiv.org/abs/2409.10542)
Comments:
          ECCV 2024

- **What's New**: SAM4MLLM은 Segment Anything Model(SAM)과 Multi-Modal Large Language Models(MLLMs)를 통합하여 픽셀 수준의 정보 인식을 향상시키는 혁신적인 접근 방식을 소개합니다. 이 방법은 기존 모델 아키텍처의 과도한 수정이나 전문 토큰의 추가 없이 MLLM이 피펠 수준의 위치 정보를 학습할 수 있도록 합니다.

- **Technical Details**: SAM4MLLM은 사진의 픽셀 위치에 대한 참조 표현(segmentation)을 활용하여 MLLM과 SAM 간의 통합을 구현합니다. 기존 MLLM 아키텍처를 수정하지 않으면서 픽셀 수준의 정보를 학습할 수 있으며, 이를 통해 이미지에서 객체를 정확하게 인식합니다. MLLM이 SAM을 위해 효과적인 프롬프트 포인트를 얻기 위해 적극적으로 질문하는 새로운 방법을 소개합니다.

- **Performance Highlights**: 다양한 RES 벤치마크(RES 데이터셋, GRES, ReasonSeg)에 대한 실험 결과는 SAM4MLLM의 유효성을 입증하며, 복잡한 픽셀 인식 작업을 처리하는 데 있어 우수한 성능을 보임을 보여줍니다.



### "Is This It?": Towards Ecologically Valid Benchmarks for Situated Collaboration (https://arxiv.org/abs/2409.10525)
- **What's New**: 이번 연구는 대형 멀티모달 모델(Large Multimodal Models, LMMs)의 실제 협업에서의 능력을 평가하기 위한 생태학적으로 유효한 벤치마크를 구축하는 초기 작업을 보고합니다. 기존 벤치마크와 달리, 사용자가 인터랙션 도중 직접 생성한 질문을 바탕으로 질문-답변 쌍을 생성하는 시스템 중심 접근 방식을 제안합니다.

- **Technical Details**: 이 연구는 Sigma라는 혼합 현실 작업 지원 시스템을 활용하여 대규모 데이터 수집을 진행하였습니다. Sigma는 HoloLens 2를 기반으로 하며, 음성 인식과 생성, LLMs를 이용하여 사용자에게 절차적 작업을 단계적으로 안내합니다. 이를 통해 수집한 데이터는 전통적인 벤치마크와는 다른 질문 유형과 도전과제를 드러내는 것을 목표로 합니다.

- **Performance Highlights**: 파일럿 실험에서 26개의 상호작용 세션을 통해 수집된 데이터는 사용자가 자연스럽게 생성한 질문의 형태와 내용을 기존의 질문 응답 벤치마크와 비교할 때 상당한 차이를 보였습니다. 이는 LMM의 실제 사용자 경험과의 일치 여부를 보다 정확히 반영할 수 있음을 시사합니다.



### Large Language Model Based Generative Error Correction: A Challenge and Baselines for Speech Recognition, Speaker Tagging, and Emotion Recognition (https://arxiv.org/abs/2409.09785)
Comments:
          IEEE SLT 2024. The initial draft version has been done in December 2023. Post-ASR Text Processing and Understanding Community: this https URL

- **What's New**: 최근 생성 AI 기술의 발전에 따라, LLM(대규모 언어 모델)이 동결된 사전 훈련된 자동 음성 인식(ASR) 모델의 텍스트 디코딩 결과를 활용하여 음향 모델링 작업을 향상시킬 수 있는 방법에 대한 질문이 제기되었습니다. 이 논문에서는 새로운 언어 모델링 기능을 탐구하기 위해 생성 음성 필기 오류 수정(GenSEC) 챌린지를 도입했습니다.

- **Technical Details**: GenSEC 챌린지는 다음 세 가지 post-ASR 언어 모델링 작업으로 구성됩니다: (i) post-ASR 전사 수정, (ii) 화자 태깅, (iii) 감정 인식. 이 작업들은 음성 기반 인터페이스를 처리하는 LLM 기반 에이전트의 미래를 모방하는 데 목표를 두고 있으며, 개방형 사전 훈련된 언어 모델이나 에이전트 기반 API를 사용하여 폭넓은 청중이 접근할 수 있도록 디자인되었습니다.

- **Performance Highlights**: 기초 평가에서 얻은 통찰력을 논의하며, 향후 평가 설계에 대한 교훈을 제공합니다. 참가자들은 ASR-LLM의 제한을 감안하여 텍스트 전용 모달리티로 음성을 처리하는 한계를 탐색하고, ASR 모델의 N-best 가설을 활용하여 발화 및 비언어적 정보를 복구하는 등의 연구를 촉진할 수 있습니다.



New uploads on arXiv(cs.IR)

### Beyond Relevance: Improving User Engagement by Personalization for Short-Video Search (https://arxiv.org/abs/2409.11281)
- **What's New**: 이 논문에서는 짧은 동영상 검색의 개인화에 대한 새로운 접근법인 PR²(Personalized Retrieval and Ranking augmented search system)를 제안합니다. 이 시스템은 대규모 비디오 코퍼스에서 개인화된 콘텐츠를 추출하고, 사용자 장기 선호와 실시간 행동을 효과적으로 활용할 수 있는 QIN(Query-Dominate User Interest Network) 모델을 활용합니다.

- **Technical Details**: PR²는 쿼리 관련 협업 필터링(query-relevant collaborative filtering)과 개인화된 밀집 검색(personalized dense retrieval)을 결합하여 사용자 맞춤형 콘텐츠를 제공합니다. 또한, 다중 작업 학습(multi-task learning) 프레임워크를 사용하여 사용자 피드백을 학습하고 이를 통해 쿼리와 관련된 사용자 관심을 반영합니다.

- **Performance Highlights**: PR²를 실제 시스템에 배포한 결과, 지난 몇 년간 가장 두드러진 사용자 참여 개선이 달성되었습니다: CTR@10이 10.2% 증가하였고, 동영상 시청 시간이 20% 증가했으며, 검색 DAU가 1.6% 상승했습니다.



### Promptriever: Instruction-Trained Retrievers Can Be Prompted Like Language Models (https://arxiv.org/abs/2409.11136)
- **What's New**: 이 논문에서는 언어 모델(LM)처럼 프롬프트(prompts)를 사용할 수 있는 최초의 검색 모델인 Promptriever를 소개합니다. Promptriever는 50만 개의 인스턴스를 포함하는 새로운 인스턴스 수준 지침(training set)을 기반으로 훈련되었습니다.

- **Technical Details**: Promptretriever는 Bi-encoder 구조를 사용하며, LLaMA-2 7B와 같은 대형 언어 모델을 백본으로 활용합니다. 이는 자연어 프롬프트를 통해 쿼리의 연관성을 동적으로 조정하여 정보 검색을 가능하게 합니다.

- **Performance Highlights**: Promptriever는 다음과 같은 성능 향상을 보입니다: (1) 상세한 연관성 지침을 따랐을 때 14.3의 p-MRR과 3.1 nDCG 증가, (2) 쿼리와 지침의 어휘 선택에 대한 강건성(robustness) 증가, (3) 프롬프트를 통해 하이퍼파라미터 검색이 가능하여 검색 성능을 1.4 포인트 향상시킴.



### Multi-modal Generative Models in Recommendation System (https://arxiv.org/abs/2409.10993)
Comments:
          32 pages 5 figures

- **What's New**: 이 논문에서는 사용자 입력을 텍스트 문자열로 한정하던 기존의 추천 시스템의 한계를 지적하고, 생성 AI(Generative AI)의 발전으로 인해 사용자들이 기대하는 더 풍부한 상호작용 수준에 대해 설명합니다.

- **Technical Details**: 추천 시스템은 텍스트와 이미지 등 다양한 모달리티(modality)를 동시에 활용하여 제품에 대한 정보의 공유 및 보완 관계를 발견해야 합니다. 기존 시스템은 종종 텍스트 검색과 이미지 검색을 별개로 처리하여 상호작용이 제한적입니다.

- **Performance Highlights**: 향후 추천 시스템은 고객과 제품에 대한 풍부한 정보를 활용하여 더 나은 추천 결과를 제공할 수 있을 것으로 기대됩니다.



### A Best-of-Both Approach to Improve Match Predictions and Reciprocal Recommendations for Job Search (https://arxiv.org/abs/2409.10992)
- **What's New**: 이 논문은 상호 추천 시스템에서의 매칭 사용자와의 선호를 효과적으로 개선하기 위한 새로운 방법을 제안합니다. 특히, ‘pseudo-match scores’를 활용하여 실제 일치 레이블과 밀접하지만 부정확한 예측을 융합한 새로운 매칭 점수를 생성합니다. 이를 통해 사용자 맞춤형 개인화된 추천의 정확성을 높입니다.

- **Technical Details**: 논문에서는 기존의 직접 매칭 예측 방법과 각 방향의 별도 모델을 결합한 ‘best-of-both (BoB)’ 접근방식을 제안합니다. 이는 정확하지만 드문 매칭 레이블과 밀접하지만 부정확한 예측의 장점을 조합하여 pseudo-match scores를 생성하며, 이를 통해 메타 모델을 교육하여 매칭 실패 확률을 최소화합니다.

- **Performance Highlights**: 오프라인 실험에서 제안된 BoB 방법은 개인화된 pseudo-match scores를 통해 기존 방법보다 매칭 성능이 우수함을 입증하였습니다. 이 방법은 특히 적은 활동 사용자 세그먼트에서 더 나은 결과를 보여주며, 상호 추천의 효율성을 극대화합니다.



### GenCRF: Generative Clustering and Reformulation Framework for Enhanced Intent-Driven Information Retrieva (https://arxiv.org/abs/2409.10909)
- **What's New**: GenCRF(Generative Clustering and Reformulation Framework)가 정보 검색 분야에서 사용자 쿼리의 다양한 의도를 포착하기 위해 처음으로 도입되었습니다. 이 프레임워크는 Large Language Models(LLMs)를 활용하여 다수의 맞춤형 프롬프트를 사용해 여러 쿼리를 생성하고 클러스터링하여 쿼리 성공률을 향상시키는 목표를 가지고 있습니다.

- **Technical Details**: GenCRF는 LLM을 통해 초기 쿼리에서 파생된 다양한 쿼리를 생성하고, 이들을 동적으로 클러스터링하여 정보 중복을 최소화합니다. 쿼리 통합을 위해 유사성 기반 동적 가중치 및 점수 기반 동적 가중치를 포함한 여러 가중 집계 전략을 적용하며, Query Evaluation Rewarding Model(QERM)을 통해 쿼리 개선 프로세스를 피드백 루프를 통해 정교화합니다.

- **Performance Highlights**: BEIR 데이터세트에서 실시한 실험을 통해 GenCRF는 기존 쿼리 재구성 기술 대비 최대 12% 향상된 nDCG@10 성능을 기록하며, 다양한 도메인과 쿼리 유형에 대해 지속적으로 우수한 성능을 입증하였습니다.



### Challenging Fairness: A Comprehensive Exploration of Bias in LLM-Based Recommendations (https://arxiv.org/abs/2409.10825)
- **What's New**: 본 연구는 Large Language Model (LLM) 기반 추천 시스템에서 발생하는 편향(bias)과 그 영향을 심층적으로 분석합니다. 특히, 음악, 노래, 도서 추천의 맥락에서 다양한 인구 통계 및 문화 그룹을 살펴봅니다.

- **Technical Details**: LLM 기반 추천 시스템은 사용자의 행동 및 콘텐츠를 깊이 분석하여 추천의 질을 높입니다. 하지만 이 시스템은 종종 훈련 데이터의 불균형으로 인해 주류 콘텐츠를 선호하고 비주류 옵션을 소외시키는 경향이 있습니다. 다양한 LLM 모델을 사용하여 bias의 영향을 평가합니다.

- **Performance Highlights**: 연구 결과, bias는 시스템에 깊게 내재되어 있으며, prompt engineering과 같은 간단한 개입만으로도 추천 결과에서 bias를 크게 줄일 수 있습니다. 교차 정체성(intersecting identities) 및 사회경제적 지위(socioeconomic status)와 같은 맥락 정보가 그러한 편향을 더욱 악화시킬 수 있다는 점도 강조됩니다.



### Googling the Big Lie: Search Engines, News Media, and the US 2020 Election Conspiracy (https://arxiv.org/abs/2409.10531)
Comments:
          40 pages

- **What's New**: 2020년 미국 대선에서의 사기 주장을 다룬 'Big Lie' 이론에 대해 검색 엔진들이 어떻게 뉴스를 제공했는지를 분석한 연구 결과가 발표되었습니다.

- **Technical Details**: 이 연구는 Google, DuckDuckGo, Bing의 세 가지 검색 엔진과 오하이오, 캘리포니아, 영국의 세 가지 위치에서 총 열한 개의 검색 쿼리를 활용하여 대규모 알고리즘 감사(algorithm audit)를 수행하였습니다. 결과적으로 세 검색 엔진 모두에서 공통적으로 사기 주장을 단순히 부정하는 방식이 가장 큰 반박 전략임을 발견했습니다.

- **Performance Highlights**: Google은 'Big Lie'에 초점을 맞춘 기사들에 대한 강력한 주류화(mainstreaming) 효과를 보여줍니다. 반면 DuckDuckGo와 Bing은 각 지역에 따라 음모론을 지지하거나 반박하지 않는 기사가 많이 나타났습니다. 또한, 특정 이데올로기에 기반한 검색 쿼리는 음모론을 지지하는 자료와의 연관성을 보이지 않았고, 사실 이러한 내용은 주로 이데올로기와 무관한 일반 검색 쿼리에서 발생했습니다.



### Towards Empathetic Conversational Recommender Systems (https://arxiv.org/abs/2409.10527)
- **What's New**: 이 논문은 Empathetic Conversational Recommender (ECR) 시스템을 제안하여 사용자의 감정을 이해하고 반영하는 새로운 접근법을 소개합니다. 일반적인 대화형 추천 시스템(CRS)에서는 사용자 요구와 상관없이 표준 아이템과 응답을 기준으로 삼는 경향이 있는데, ECR은 이와 같은 오차를 수정하기 위해 감성을 도입합니다.

- **Technical Details**: ECR은 두 가지 주요 모듈로 구성됩니다: 감정 인식 아이템 추천(emotion-aware item recommendation)과 감정 정렬 응답 생성(emotion-aligned response generation). 사용자 감정을 기반으로 해 섬세한 추천 모델링을 수행하며, 감정에 맞춰 사전 훈련된 언어 모델(pre-trained language model)을 미세 조정하여 사람처럼 감정적인 응답을 생성합니다. 이때 대규모 언어 모델(large language models)을 사용하여 감정 라벨을 추가하고, 외부 소스에서 수집된 감정 리뷰로 데이터를 확장합니다.

- **Performance Highlights**: ReDial 데이터셋을 통해 ECR 프레임워크는 추천 정확도(recommendation accuracy)를 향상시키고 사용자 만족도(user satisfaction)를 크게 개선하여 기존 모델들을 능가하는 성능을 보였습니다. 새로운 평가 지표는 감정 강도(emotional intensity), 감정 설득력(emotional persuasiveness), 논리적 설득력(logic persuasiveness), 정보 제공(informativeness), 생동감(lifelikeness) 등을 포함하여 실세계 CRS 시나리오에서의 사용자 만족도를 더 잘 반영합니다.



### Bridging User Dynamics: Transforming Sequential Recommendations with Schr\"odinger Bridge and Diffusion Models (https://arxiv.org/abs/2409.10522)
Comments:
          CIKM '24

- **What's New**: 본 논문에서는 확산모델(difussion model)을 이용한 순차 추천 시스템(sequential recommendation)에서, 가우시안 분포(Gaussian distribution)의 한계를 극복하고 사용자의 현재 상태에 기반한 새로운 추천 모델(SdifRec)을 제안합니다.

- **Technical Details**: SdifRec 모델은 사용자의 현재 상태를 고려하여 기존의 가우시안 사전 분포를 대체합니다. 또한 사용자 클러스터링 정보를 조건으로 활용하여 posterior distribution을 개선하는 con-SdifRec라는 확장 모델도 개발했습니다. 연구에서 제안된 슈뢰딩거 다리(Schrödinger Bridge) 기법은 두 분포 간의 최소 비용 경로를 찾는 방법론으로, 이를 통해 모델의 유연성을 증가시킵니다.

- **Performance Highlights**: 다양한 공개 벤치마크 데이터셋에서 실시한 비교 실험 결과, SdifRec 및 con-SdifRec 모델이 여러 최신 방법론에 비해 상당한 성능 향상을 보이며 효율성과 안정성이 검증되었습니다.



### LSTM Recurrent Neural Networks for Cybersecurity Named Entity Recognition (https://arxiv.org/abs/2409.10521)
- **What's New**: 이 논문에서는 사이버 보안 분야의 정보 추출에 필요한 Named Entity Recognition (NER) 모델을 자동화한 새로운 접근 방식을 제시합니다. 이 모델은 도메인 독립적이며, 사이버 보안 엔티티의 특수 기능에 의존하지 않기 때문에 전문가의 지식 없이도 사용할 수 있습니다.

- **Technical Details**: 모델은 Long Short-Term Memory (LSTM)와 Conditional Random Fields (CRFs) 기법을 활용하여 구현되었습니다. 이 방법은 극복해야 할 사이버 보안의 복잡한 엔티티 구조를 다루기 위해 설계되었습니다.

- **Performance Highlights**: 제안된 방법은 적당한 크기의 주석이 있는 말뭉치를 사용했을 때 최첨단 방법들보다 우수한 성능을 보였습니다.



### TISIS : Trajectory Indexing for SImilarity Search (https://arxiv.org/abs/2409.11301)
- **What's New**: 이번 연구에서는 사용자 이동 경로를 reconstruction할 수 있는 geo-location data를 활용하여, 다양한 활용 사례에서 포인트(POIs) 간 유사성을 측정하는 두 가지 새로운 방법 TISIS 및 TISIS*를 제안합니다.

- **Technical Details**: TISIS는 trajectory indexing을 이용한 효율적인 방법으로, POIs의 순서가 동일한 유사한 경로를 신속히 찾을 수 있도록 도와줍니다. TISIS*는 POI 임베딩을 통합하여, contextoally 비슷한 POIs 간의 유사성을 고려하여 더 포괄적인 결과를 제공합니다. 기존의 LCSS(최장 공통 부분 수열) 알고리즘 기반의 방법보다 성능이 크게 향상되었습니다.

- **Performance Highlights**: 제안된 TISIS와 TISIS* 방법은 다양한 실제 데이터셋에서 LCSS 알고리즘 기반의 기존 방법에 비해 상당한 처리 속도 개선과 성능 향상을 보였습니다.



### P-RAG: Progressive Retrieval Augmented Generation For Planning on Embodied Everyday Task (https://arxiv.org/abs/2409.11279)
- **What's New**: 이 논문에서는 Embodied Everyday Task를 위한 새로운 접근법인 Progressive Retrieval Augmented Generation (P-RAG)을 제안합니다. 기존의 Large Language Model (LLM) 기반 접근법의 한계를 극복하고, 자연어 지시를 기반으로 하여 임베디드 AI 환경에서의 작업 수행 능력을 개선하는 데 초점을 맞추었습니다.

- **Technical Details**: P-RAG는 자연어 지시와 시각 관찰을 기반으로 하는 작업에서, ground-truth 데이터 없이도 작업에 대한 특정 지식을 점진적으로 축적하는 데 사용됩니다. 기존 RAG 방법과는 달리 P-RAG는 반복적 접근법을 도입하여 데이터베이스를 점진적으로 업데이트하고, 각 반복에서 최신 데이터베이스를 조회하여 경험적으로 참고할 수 있는 역사적 정보를 얻습니다.

- **Performance Highlights**: P-RAG는 기존 방법들과 비교하여 몇 가지 성과를 이룬 것으로 나타났습니다. 특히 몇 번의 샘플로 구성된 학습 환경에서도 우수한 성능을 보였으며, 자기 반복을 통해 성능을 further 향상시키는 기능을 입증하였습니다.



### Inside Alameda Research: A Multi-Token Network Analysis (https://arxiv.org/abs/2409.10949)
- **What's New**: 이번 논문은 Ethereum의 토큰 전송 네트워크를 분석하고, Alameda Research 관련 계정의 활동을 검토합니다. 네트워크의 노드 중심성과 백본을 조사하여, 중요한 계정, 토큰 및 활동 그룹을 식별하는 데 중점을 둡니다.

- **Technical Details**: 다중 토큰 네트워크 표현을 사용하여 계정과 토큰 간의 상호작용을 분석하며, 시간에 따른 Alameda 계정의 변화도 살펴봅니다. 이는 이 회사가 2022년 11월 파산에 이르기까지의 토큰 축적 및 분배 패턴을 연구합니다.

- **Performance Highlights**: 네트워크 분석을 통해 DeFi 생태계를 형성하는 활동 및 역학에 대한 통찰을 제공합니다. 특히 Alameda Research와 그 이면에 있는 복잡한 관계가 어떻게 형성되었는지에 대한 이해를 돕습니다.



### Attention-Seeker: Dynamic Self-Attention Scoring for Unsupervised Keyphrase Extraction (https://arxiv.org/abs/2409.10907)
- **What's New**: Attention-Seeker라는 새로운 비지도 키프레이즈 추출 방법이 제안되었습니다. 이 방법은 대규모 언어 모델의 self-attention 맵을 이용하여 후보 구문들의 중요도를 평가합니다. 이전 모델들과는 달리 매개변수 수동 조정이 필요하지 않아 유용성이 향상되었습니다.

- **Technical Details**: Attention-Seeker는 self-attention 맵의 특정 구성요소(레이어, 헤드, 어텐션 벡터)를 통해 텍스트의 주요 주제에 집중합니다. 이 방식은 ALICE(The Attention Based Language Interpretation Comprehension) 모델을 기반으로 하여 입력 텍스트의 특성에 자동으로 적응하도록 설계되었습니다. 길이에 따라 단문은 개별 attention 벡터를 평가하고, 장문은 문서 세그먼트를 고려합니다.

- **Performance Highlights**: Attention-Seeker는 Inspec, SemEval2010, SemEval2017, Krapivin의 네 개 공적 데이터셋에서 실험되었으며, 매개변수 조정 없이도 대부분의 기준 모델을 초월하고 세 개의 데이터셋에서 최첨단 성능을 달성했습니다. 특히 긴 문서에서 키프레이즈 추출에 탁월했습니다.



### Online Learning via Memory: Retrieval-Augmented Detector Adaptation (https://arxiv.org/abs/2409.10716)
Comments:
          Accepted at ECCV 2024, Human-Inspired Computer Vision (HCV) workshop

- **What's New**: 본 논문은 기존의 객체 탐지 모델을 신규 도메인에 온라인으로 적응시키는 새로운 방법을 제시합니다. 이 과정에서 모델 재훈련 없이 메모리 내 유사 객체 개념을 검색하여 활용합니다. 이를 통해 아무리 적은 데이터(예: 카테고리 당 10개의 이미지)로도 객체 탐지를 효과적으로 수행할 수 있습니다.

- **Technical Details**: 제안된 프레임워크는 네 가지 주요 모듈로 구성됩니다: (i) 온라인 업데이트 가능한 메모리 뱅크, (ii) 객체 제안 모델, (iii) 컨텍스트 검색 모듈, (iv) 인스턴스 검색 모듈. 이 구조를 통해 객체 설정을 수행하고 새로운 개념을 분류합니다.

- **Performance Highlights**: 실험 결과, 기존의 클로즈세트(close-set) 및 오픈세트(open-set) 모델에 비해 제안한 온라인 학습 방법이 큰 성능 향상을 보여주었으며, 소량의 레이블링 노력으로 지난해 DOTA 데이터셋에서 mAP 20 이상을 달성했습니다.



### Language Models and Retrieval Augmented Generation for Automated Structured Data Extraction from Diagnostic Reports (https://arxiv.org/abs/2409.10576)
- **What's New**: 이 논문은 비구조화된 방사선 및 병리학 보고서에서 구조화된 임상 정보를 자동으로 추출하기 위한 시스템을 개발하고 평가한 내용을 담고 있습니다. 특히 open-weights large language models (LMs)와 retrieval augmented generation (RAG) 기술을 활용하였습니다.

- **Technical Details**: 이 연구는 두 개의 데이터셋을 사용하였습니다: 7,294개의 방사선 보고서(BT-RADS 점수 주석 포함)와 2,154개의 병리학 보고서(IDH 변이 상태 주석 포함). 다양한 LMs와 RAG 구성을 평가하기 위한 자동화된 파이프라인이 개발되었으며, 모델 크기, 양자화(quantization), 프롬프트 전략(prompting strategies), 출력 형식(output formatting), 추론 매개변수(inference parameters) 등의 변수들이 성능에 미치는 영향을 체계적으로 분석하였습니다.

- **Performance Highlights**: 최고 성능 모델은 방사선 보고서에서 BT-RADS 점수를 98% 이상 정확도로 추출하였으며, 병리학 보고서에서 IDH 변이 상태 추출에 대해 90% 이상의 정확도를 기록하였습니다. 성능이 가장 뛰어난 모델은 medical fine-tuned llama3로, 크고 최신의 도메인 맞춤형 모델들이 이전 모델들보다 일관되게 성능이 우수했습니다. 양자화는 성능에 미치는 영향이 미미했으며, few-shot prompting이 정확도를 크게 향상시켰습니다. RAG는 복잡한 병리학 보고서의 성능을 개선했지만 짧은 방사선 보고서에는 효과가 없었습니다.



### MUSE: Flexible Voiceprint Receptive Fields and Multi-Path Fusion Enhanced Taylor Transformer for U-Net-based Speech Enhancemen (https://arxiv.org/abs/2406.04589)
Comments:
          This paper was accepted by Interspeech 2024

- **What's New**: 이번 논문에서는 경량화된 음성 향상 네트워크인 MUSE를 소개합니다. MUSE는 U-Net 아키텍처를 기반으로 하여 Multi-path Enhanced Taylor (MET) Transformer 블록을 통합하여 음성 인식 과정의 유연성을 높였습니다.

- **Technical Details**: MUSE는 Deformable Embedding (DE) 기능을 포함하여 음성의 복잡한 특징을 효과적으로 학습합니다. MET Transformer는 Channel and Spatial Attention (CSA) 브랜치를 융합하여 채널 정보의 교환을 촉진하고, 공간 주의력 부족 문제를 해결합니다. 최종적으로, amplitude와 phase 스펙트럼을 독립적으로 디코딩하여 개선된 음성 신호를 Inverse Short-Time Fourier Transform (ISTFT)를 통해 재구성하였습니다.

- **Performance Highlights**: MUSE는 VoiceBank+DEMAND 데이터셋에서 실험을 진행하였으며, 0.51M의 적은 파라미터 수로도 경쟁력 있는 성능을 달성했습니다. 또한, 메모리 사용을 최소화하여 8GB GPU로도 충분히 훈련이 가능하다는 점이 특징입니다.



New uploads on arXiv(cs.CV)

### Phidias: A Generative Model for Creating 3D Content from Text, Image, and 3D Conditions with Reference-Augmented Diffusion (https://arxiv.org/abs/2409.11406)
Comments:
          Project page: this https URL

- **What's New**: Phidias는 3D 모델 생성에서 참고 모델을 활용하여 품질과 일반화 능력, 제어 가능성을 향상시키는 새로운 생성 모델입니다.

- **Technical Details**: 이 모델은 세 가지 주요 구성 요소로 이루어집니다: 1) meta-ControlNet, 2) dynamic reference routing, 3) self-reference augmentations. 메타-ControlNet은 조건 강도를 동적으로 조절하며, 동적 참조 라우팅은 입력 이미지와 3D 참조 간의 불일치를 완화하고, 자가 참조 보강은 자기 지도 학습을 가능하게 합니다.

- **Performance Highlights**: Phidias는 텍스트, 이미지 및 3D 조건을 사용하여 3D 세팅의 통합 프레임워크를 제공합니다. 실험 결과 기존 방법들에 비해 질적으로나 양적으로 우수한 성능을 보여주었습니다.



### Training Datasets Generation for Machine Learning: Application to Vision Based Navigation (https://arxiv.org/abs/2409.11383)
Comments:
          6 pages, 4 figures, preprint of the proceedings of ESA SPAICE conference 2024

- **What's New**: 본 논문은 머신 러닝을 활용한 우주 응용 프로그램을 위해 적절한 훈련 데이터셋을 생성하는 방법론을 제시합니다. 두 가지 사용 사례(인오르빗 레인더부와 달 착륙 시나리오)를 통해 고품질 이미지와 메타데이터를 포함하는 데이터셋을 생성하였습니다.

- **Technical Details**: 데이터셋은 SurRender 소프트웨어 및 Generative Adversarial Networks (GAN)을 이용해 생성되었으며, AI 기반 자세 추정 알고리즘과 밀집 광학 흐름 알고리즘을 벤치마크로 선정하였습니다. Chang’e 3의 아카이브 데이터셋, DLR TRON 시설 실험실 데이터 및 Airbus Robotic 실험실 데이터가 활용되었습니다.

- **Performance Highlights**: 실험 결과, SurRender 및 실험실 데이터셋으로 생성된 데이터가 머신 러닝 알고리즘 훈련에 적합하다는 것이 입증되었습니다. CNN 기반의 키포인트 감지 알고리즘과 기존 PnP 알고리즘을 결합하여 높은 정확도의 자세 추정 결과를 도출하였습니다.



### Ultrasound Image Enhancement with the Variance of Diffusion Models (https://arxiv.org/abs/2409.11380)
Comments:
          Accepted by the IEEE International Ultrasonics Symposium (IUS) 2024

- **What's New**: 이 논문은 초음파(Ultrasound) 이미징의 노이즈와 아티팩트를 줄이기 위한 새로운 접근을 제안합니다. Adaptive beamforming과 denoising diffusion-based variance imaging을 결합하여 더 나은 이미지를 생성하는 방법을 소개합니다.

- **Technical Details**: 제안된 방법은 Eigenspace-Based Minimum Variance (EBMV) beamforming을 사용하여 시간 도메인에서 수신된 RF 신호를 공간 도메인으로 변환합니다. 이후, conditional diffusion generative process를 여러 번 실행하고 생성된 샘플의 분산을 계산하여 개선된 이미지를 생성합니다. 노이즈 제거 과정은 기존의 역 문제 모델 대신 denoising 모델을 사용합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 단일 plane-wave 출처에서 우수한 이미지 재구성을 달성하여 속도와 이미지 품질 모두에서 경쟁력을 보여줍니다. 이러한 성능 향상은 효율적인 샘플링과 더욱 정확한 신호 처리 덕분입니다.



### Multi-OCT-SelfNet: Integrating Self-Supervised Learning with Multi-Source Data Fusion for Enhanced Multi-Class Retinal Disease Classification (https://arxiv.org/abs/2409.11375)
Comments:
          25 pages, 9 tables, 10 figures

- **What's New**: 본 연구에서는 다양한 데이터 출처를 결합하여 좋은 성능과 새로운 데이터로의 일반화를 개선하기 위해 대규모 언어 모델(LLMs)과 SwinV2 기반의 자가 감독 학습 프레임워크를 개발하였다. 이 작업은 광학 단층 촬영(OCT) 이미지를 사용한 눈 질환 탐지 능력을 향상시키는 것을 목표로 한다.

- **Technical Details**: 두 단계 훈련 방법론인 자가 감독 프리트레이닝과 다운스트림 감독 분류기에 대한 미세 조정을 채택하였다. 여러 인코더 백본을 사용한 세 개의 데이터세트를 통해 수행된 절제 연구는 데이터 융합 없이, 데이터 가용성이 낮은 설정에서 자가 감독 프리트레이닝 없는 시나리오를 비교하여 우리의 방법론의 견고함을 강조하였다.

- **Performance Highlights**: 세 가지 다양한 조건에서 일관된 성능을 보여주며, ResNet-50의 기준 모델과 비교할 때 우수한 일반화 능력을 갖추었다. 본 연구는 제한된 데이터로도 효과적인 학습이 가능하도록 설계되었다.



### Uncertainty and Prediction Quality Estimation for Semantic Segmentation via Graph Neural Networks (https://arxiv.org/abs/2409.11373)
Comments:
          11 pages, 3 figures, submitted to BMVC "Workshop on Robust Recognition in the Open World" (this https URL)

- **What's New**: 본 논문에서는 시맨틱 세분화(Semantic Segmentation)에서 예측 품질(Prediction Quality)과 불확실성(Uncertainty) 추정을 인접 세그먼트의 정보를 활용하여 개선하는 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 그래프 신경망(Graph Neural Networks, GNNs)을 이용하여 주어진 세그먼트의 품질과 인접 세그먼트의 메트릭스 간의 관계를 모델링합니다. 이를 통해 세그먼트 레벨에서 불확실성 지표를 기반으로 예측 품질을 추정합니다.

- **Performance Highlights**: GNN 아키텍처의 다양한 구현을 평가한 결과, 인접 세그먼트를 고려하지 않은 기초 모델 대비 AUROC에서 최대 1.78 포인트 향상이 있음을 보였습니다.



### OSV: One Step is Enough for High-Quality Image to Video Generation (https://arxiv.org/abs/2409.11367)
- **What's New**: 본 논문에서는 영상 생성에서의 비디오 확산 모델의 한계를 극복하기 위한 새로운 두 단계 교육 프레임워크를 제안합니다. 이 프레임워크는 일관성 증류(consistency distillation)와 GAN 교육을 결합하여 성능을 향상시키고, 디코딩 단계 없이 고품질 비디오를 됩니다.

- **Technical Details**: 제안된 OSV(One Step Video Generation) 모델은 낮은 순위 적응(Low-Rank Adaptation, LoRA) 및 GAN 교육을 통합하여 빠른 수렴을 달성하는 두 단계 비디오 확산 가속 교육 전략을 사용합니다. 첫 번째 단계에서는 실제 데이터를 GAN의 조건으로 사용하여 모델 수렴을 가속화하고, 두 번째 단계에서는 LCM 훈련 기능을 도입하여 GAN 교육의 불안정성을 해결합니다. 또한, VAE 디코더를 제거하고 단순 업샘플링 연산을 사용하는 새로운 비디오 판별기 설계를 제안합니다.

- **Performance Highlights**: OSV 모델은 단 1단계로 고품질 비디오를 생성할 수 있으며, OpenWebVid-1M 벤치마크에서 기존의 방법들과 비교했을 때 현저히 우수한 성능을 보여줍니다. 1단계 성능(FVD 171.15)은 8단계의 AnimateLCM 성능(FVD 184.79)을 초과하고, 25단계 고급 안정적인 비디오 확산 모델(Stable Video Diffusion, FVD 156.94) 성능에 근접합니다.



### RenderWorld: World Model with Self-Supervised 3D Lab (https://arxiv.org/abs/2409.11356)
- **What's New**: RenderWorld는 LiDAR-vision 융합 방식보다 비용 효율적이고 신뢰성이 높은 순수 비전을 기반으로 한 자율 주행 프레임워크입니다.

- **Technical Details**: 이 시스템은 self-supervised gaussian 기반의 Img2Occ 모듈을 사용하여 3D occupancy 레이블을 생성하고, AM-VAE(Adversarial Variational Autoencoder)를 통해 레이블을 인코딩한 후, 세계 모델을 이용하여 예측 및 계획을 수행합니다. Gaussian Splatting 기법을 적용하여 3D 장면을 표현하고 2D 이미지를 렌더링함으로써 NeRF 기반 방법에 비해 세분화 정확도를 크게 향상시키고 GPU 메모리 소비를 줄입니다.

- **Performance Highlights**: RenderWorld는 공기와 비공기를 별도로 인코딩하는 AM-VAE의 적용을 통해 보다 세밀한 장면 요소 표현을 달성하며, 4D occupancy 예측과 자가 회귀형 세계 모델에 의한 모션 계획에서 최첨단 성능을 보여줍니다.



### Fine-Tuning Image-Conditional Diffusion Models is Easier than You Think (https://arxiv.org/abs/2409.11355)
Comments:
          Project page: this https URL

- **What's New**: 본 논문에서는 대형 diffusion 모델을 단안(depth) 추정(task)에서 재사용할 수 있는 방법을 제시합니다. 이 모델은 과거에 보고된 최선의 구성과 유사한 성능을 보이면서도 속도가 200배 이상 빨라졌습니다.

- **Technical Details**: 모델은 단일 단계(single-step) 모델을 기반으로 하여 작업에 특화된 손실(task-specific losses)로 엔드 투 엔드(fine-tuning)를 수행합니다. 이를 통해 일반적인 제로샷(zero-shot) 벤치마크에서 모든 다른 diffusion 기반 모델들을 초월하는 결정론적(deterministic) 모델을 획득하게 됩니다.

- **Performance Highlights**: 이 방법은 또한 Stable Diffusion에 대한 직접적인 미세 조정(fine-tuning)에서도 효과가 있으며, 기존의 state-of-the-art diffusion 모델들과 유사한 성능을 보여줍니다. 이는 이전 연구의 결론에 의문을 제기합니다.



### OmniGen: Unified Image Generation (https://arxiv.org/abs/2409.11340)
- **What's New**: OmniGen은 통합된 이미지 생성을 위한 새로운 diffusion 모델로, 기존의 ControlNet이나 IP-Adapter 같은 추가 모듈 없이 다양한 제어 조건을 처리할 수 있습니다. 이 모델은 텍스트-이미지 생성뿐만 아니라 이미지 편집 및 주제 기반 생성과 같은 다운스트림 작업을 지원합니다.

- **Technical Details**: OmniGen의 구조는 Variational Autoencoder (VAE)와 transformer 모델로 구성되어 있으며, 추가 텍스트 인코더가 필요하지 않습니다. 이는 사용자가 복잡한 작업을 지시를 통해 수행할 수 있도록 도와줍니다. 모델은 텍스트와 이미지를 동시에 입력받아 처리할 수 있습니다.

- **Performance Highlights**: OmniGen은 기존의 이미지 생성 모델보다 경쟁력 있는 텍스트-이미지 생성 능력을 보이며, 이미지 편집, 시각적 조건 생성을 포함한 다양한 이미지 생성 작업을 효과적으로 지원합니다. 실험 결과, OmniGen은 이전에 본 적 없는 작업과 도메인에서도 혁신적인 능력을 나타냅니다.



### CLIP Adaptation by Intra-modal Overlap Reduction (https://arxiv.org/abs/2409.11338)
Comments:
          BMVC 2024, Oral

- **What's New**: 이 논문에서는 사전 훈련된 CLIP 모델의 few-shot classification을 위해 가벼운 adapter를 훈련하는 새로운 방법을 제안합니다. 이 방법은 이미지 공간 내의 intra-modal overlap (IMO)을 줄임으로써, few-shot classification의 성능을 향상시키고자 합니다.

- **Technical Details**: 저자들은 Google Open Images 데이터셋에서 일반 샘플 세트에 대해 단일 epoch 동안 경량 adapter를 훈련하는 방식을 사용합니다. 이 과정에서 저자들은 cosine similarity분포의 차이를 통해 paired 이미지와 unpaired 이미지의 유사성을 평가하여 IMO를 해결합니다.

- **Performance Highlights**: 이 방법은 다양한 테스트 데이터셋에서 성능을 개선하며, 예를 들어 EuroSAT 데이터셋의 경우, 1개의 예제를 사용하여 분류 성능이 48.38%에서 68% 이상으로 향상되었습니다. 이는 기존의 approach와 결합하여 전체 정확도를 일관되게 개선하는 성질을 보여줍니다.



### Reducing Catastrophic Forgetting in Online Class Incremental Learning Using Self-Distillation (https://arxiv.org/abs/2409.11329)
Comments:
          10 pages, 2 figures

- **What's New**: 이번 논문에서는 continuous learning(연속 학습) 분야에서 발생하는 catastrophic forgetting(재앙적 망각) 문제를 해결하기 위해, self-distillation(자기 증류) 기법을 활용한 새로운 접근 방식을 제안했습니다. 이 방법은 얕은 층에서 얻은 일반화 가능한 출력을 교사로 이용하여, 모델의 일반화 능력을 향상시키는 것을 목표로 합니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 요소로 구성됩니다. 첫째로, self-distillation loss(자기 증류 손실)를 도입하여 얕은 층의 일반화된 특징을 활용합니다. 둘째로, 쉽게 잘못 분류된 샘플의 저장을 우선시하는 새로운 메모리 업데이트 방식을 통하여 더 효율적인 학습과 충분한 훈련을 보장합니다.

- **Performance Highlights**: 실험 결과, 우리 방법은 CIFAR10, CIFAR100, MiniImageNet 데이터셋에서 기존 온라인 연속 학습 방법들에 비해 재앙적 망각을 감소시키는 데 성공하였습니다. 특히, 가장 작은 메모리 버퍼 크기인 M=100, 500, 500에서 각각 5.9%, 3.2%, 4.0%의 분류 정확도 향상을 보여주었습니다.



### TopoMaskV2: Enhanced Instance-Mask-Based Formulation for the Road Topology Problem (https://arxiv.org/abs/2409.11325)
Comments:
          Accepted to ECCV 2024 2nd Workshop on Vision-Centric Autonomous Driving (VCAD). TopoMaskV2 includes significant architectural improvements and extensive ablation studies over the original TopoMask, which received an innovation award in the OpenLane Topology Challenge 2023

- **What's New**: TopoMask는 기존의 keypoint 기반 또는 매개변수(parametric) 방법 대신 인스턴스 마스크(instance-mask) 기반의 접근 방식을 도입하여 중심선(centerline) 예측 성능을 향상시키는 혁신적인 방법입니다.

- **Technical Details**: TopoMask는 BEV(Bird’s Eye View) 도메인에서 흐름 정보를 포함한 4방향 레이블 표현을 통해 인스턴스 마스크의 성능을 개선하며, 마스크 예측과 베지어 회귀(Bezier regression) 결과를 융합하여 중심선 예측을 진행합니다. 또한, multi-height bin 구성을 통해 Lift Splat 기술의 한계를 극복하고 있습니다.

- **Performance Highlights**: TopoMask는 OpenLane-V2 데이터셋에서 Subset-A에서 44.1에서 49.4로, Subset-B에서 44.7에서 51.8로 성능이 향상되어 최첨단 성능을 달성했습니다. 이 연구는 OpenLane Topology Challenge 2023에서 혁신상을 수상하기도 했습니다.



### LPT++: Efficient Training on Mixture of Long-tailed Experts (https://arxiv.org/abs/2409.11323)
Comments:
          Extended version of arXiv:2210.01033

- **What's New**: 이번 연구에서는 LPT++라는 새로운 프레임워크를 소개합니다. LPT++는 파라미터 효율적인 파인튜닝(PEFT)과 학습 가능한 모델 앙상블을 결합한 길이가 긴 분류( long-tailed classification) 시스템으로, 세 가지 주요 구성요소를 통합하여 기존의 Vision Transformers (ViTs)를 향상시킵니다.

- **Technical Details**: LPT++는 보편적인 길이 긴 적응 모듈, 혼합 길이 긴 전문가 프레임워크(Mixture of Long-tailed Experts Framework)와 미분화(Mixture-of-Experts, MoE) 스코어러, 세 단계 학습 프레임워크를 포함하여 빠른 적응성과 뛰어난 성능을 제공합니다. LPT++는 공유 프롬프트와 그룹 고유 프롬프트를 학습하여 다양한 패턴을 인식합니다.

- **Performance Highlights**: LPT++는 약 1%의 추가 훈련 가능한 파라미터로, 전체적으로 동일한 사전 훈련 모델을 조정하는 기존 방법에 비해 더 높은 정확도를 달성하였고, 특히 Place-LT 및 iNaturalist 2018 데이터 세트에서 각각 1.2% 및 1.4% 더 높은 정확도를 보였습니다.



### MSDNet: Multi-Scale Decoder for Few-Shot Semantic Segmentation via Transformer-Guided Prototyping (https://arxiv.org/abs/2409.11316)
- **What's New**: 본 연구에서는 Few-shot Semantic Segmentation (FSS) 문제를 해결하기 위해 transformer 아키텍처 기반의 새로운 프레임워크를 제안합니다. 이 프레임워크는 spatial transformer decoder와 contextual mask generation 모듈을 도입하여 support 이미지와 query 이미지 간의 관계적 이해를 개선하고, 글로벌 피처를 통합하여 컨텍스트 이해를 향상시킵니다.

- **Technical Details**: 제안된 방법은 multi-scale decoder를 사용하여 다양한 해상도의 피처를 계층적으로 통합하여 segmentation mask를 세밀하게 개선합니다. 또한, encoder의 중간 단계에서 글로벌 피처를 통합하여 복잡도를 줄이고 경량 구조를 유지합니다.

- **Performance Highlights**: 이 접근 방식은 1-shot 및 5-shot 설정 모두에서 $PASCAL-5^i$ 및 $COCO-20^i$와 같은 벤치마크 데이터셋에서 최신 기술을 능가하는 성능을 달성하였습니다. 특히, 150만 개의 매개변수만을 사용하여 기존의 방법론의 한계를 극복하면서 경쟁력 있는 성능을 보여주었습니다.



### fMRI-3D: A Comprehensive Dataset for Enhancing fMRI-based 3D Reconstruction (https://arxiv.org/abs/2409.11315)
Comments:
          Extended version of "MinD-3D: Reconstruct High-quality 3D objects in Human Brain", ECCV 2024 (arXiv: 2312.07485)

- **What's New**: 본 논문에서는 기존 2D 시각 정보 재구성 방법을 넘어 3D 시각 정보를 디코딩하고 재구성하기 위한 새로운 작업인 Recon3DMind를 제안합니다. 이를 위해 fMRI-3D 데이터셋(fMRI-3D dataset)인 fMRI-Shape 및 fMRI-Objaverse를 도입하여 3D 객체와 관련된 다양한 세부 정보를 담고 있습니다.

- **Technical Details**: MinD-3D라는 새로운 프레임워크는 fMRI 신호로부터 3D 시각 정보를 데이터 코드화하는 데 초점을 맞추고 있습니다. 이 프레임워크는 세 가지 주요 단계로 이루어져 있습니다: (1) neuro-fusion encoder를 사용하여 fMRI 데이터에서 특징을 추출 및 집계, (2) feature-bridge diffusion 모델을 사용하여 시각 특징 생성, (3) generative transformer decoder를 활용하여 3D 객체 재구성.

- **Performance Highlights**: 실험 결과 MinD-3D는 높은 의미적(semantic) 및 공간적(spatial) 정확도로 3D 객체를 재구성하는 데 성공했으며, 새로운 벤치마크를 설정했습니다. Out-of-Distribution 설정에서도 모델의 효과성을 평가하였고, 뇌의 3D 시각 정보 처리 방식에 대한 이해를 심화시켰습니다.



### GS-Net: Generalizable Plug-and-Play 3D Gaussian Splatting Modu (https://arxiv.org/abs/2409.11307)
- **What's New**: 이 논문에서 소개한 GS-Net은 3D Gaussian Splatting(3DGS) 기술에 대해 새로운 접근 방식을 제안하며, 서로 다른 장면에서의 일반화를 지원하는 최초의 플러그 앤 플레이(plug-and-play) 모듈입니다. GS-Net은 Sparse SfM(point clouds)로부터 더 밀집된 Gaussian ellipsoids를 생성하여 기하학적 구조 표현을 강화합니다.

- **Technical Details**: GS-Net은 Sparse SfM(point clouds)을 입력으로 사용하여, 이러한 점들을 강화하여 밀도가 높은 Gaussian ellipsoids를 생성하고, 이는 서로 다른 장면에서의 훈련 및을 가능하게 하여 기존의 3DGS가 가진 장면 경계의 한계를 극복합니다. 본 연구에서 우리는 CARLA-NVS 데이터 세트를 생성하여, 12 개의 카메라 시점에서의 훈련 및 평가를 지원하여 자율주행에 대한 성능을 평가합니다.

- **Performance Highlights**: 실험 결과 GS-Net을 3DGS에 적용 시, 전통적인 시점에서 PSNR(peak signal-to-noise ratio)이 2.08 dB, 새로운 시점에서는 1.86 dB 개선된 성과를 보였습니다. 이는 GS-Net의 방법이 실효성과 견고함을 입증하고 있음을 보여줍니다.



### Temporal As a Plugin: Unsupervised Video Denoising with Pre-Trained Image Denoisers (https://arxiv.org/abs/2409.11256)
- **What's New**: 본 논문에서는 'Temporal As a Plugin'(TAP)이라는 새로운 비지도 비디오 노이징 프레임워크를 제안합니다. 이 프레임워크는 사전 훈련된 이미지 노이저를 기반으로 하여 이미지 노이징 기술의 강점을 비디오 노이징에 통합합니다.

- **Technical Details**: TAP 프레임워크는 인코더-디코더 기반의 이미지 노이저에 조정 가능한 시간 모듈을 삽입하여 노이프레임 간의 시간 정보를 활용할 수 있도록 합니다. 진보적인 미세 조정 전략을 통해 각 시간 모듈을 생성된 의사 청정 비디오 프레임으로 정제하여 네트워크의 노이징 성능을 향상시킵니다.

- **Performance Highlights**: 제안된 프레임워크는 sRGB 및 원시 비디오 노이징 데이터셋에서 기존의 비지도 비디오 노이징 방법보다 우수한 성능을 보여줌으로써 최첨단의 비디오 노이징 성능을 달성합니다.



### SLAck: Semantic, Location, and Appearance Aware Open-Vocabulary Tracking (https://arxiv.org/abs/2409.11235)
Comments:
          ECCV2024

- **What's New**: 본 논문에서는 새로운 Open-vocabulary Multiple Object Tracking (MOT) 프레임워크 SLAck를 제안합니다. SLAck는 객체의 의미론적 정보(semantics), 위치(location), 외관(appearance) 정보를 초기 단계에서 통합하여 사전 훈련된 검출기를 통해 다양한 객체를 효과적으로 추적할 수 있도록 합니다. 이 접근 방식은 기존 히스테리틱(heuristics) 기반의 복잡한 후처리 단계를 제거함으로써 대규모 open-vocabulary 추적에서의 성능을 크게 향상시킵니다.

- **Technical Details**: SLAck는 다중 헤드 어텐션(multi-head attention)을 사용하여 공간 및 시간 객체 그래프(spatial and temporal object graph)를 구성합니다. 이 모델은 데이터로부터 학습한 암묵적(implicit) 모션 프라이어(motion priors)를 유지 고정하지 않고 스페이셜과 템포럴 인식(temporal perception)을 통해 객체 간의 관계를 모델링합니다. 기존의 Kalman 필터(Kalman Filter)에 의존하지 않고 객체의 위치와 형태 정보를 특성 공간(feature space)으로 매핑하여 모션 정보를 강화합니다.

- **Performance Highlights**: 본 연구는 open-vocabulary MOT와 TAO TETA 기준에서 새로운 클래스 추적의 성능을 크게 향상시켰으며, SLAck는 과거의 최첨단 방법들을 초월한 결과를 보여줍니다. 실험 결과는 암묵적으로 학습된 모션 패턴과 의미론적 정보의 통합이 추적 성능을 현저히 높일 수 있음을 밝혀냈습니다. 또한, 외관 정보의 통합은 연관 정확도를 크게 향상시키며, 이를 통해 더욱 강력한 잡음 제어 및 객체 추적을 실현했습니다.



### STCMOT: Spatio-Temporal Cohesion Learning for UAV-Based Multiple Object Tracking (https://arxiv.org/abs/2409.11234)
- **What's New**: 이번 연구에서는 UAV(무인 항공기) 비디오에서의 다중 물체 추적(Multiple Object Tracking, MOT) 문제를 해결하기 위해 새로운 Spatio-Temporal Cohesion Multiple Object Tracking(STCMOT) 프레임워크를 제안합니다. STCMOT는 순차적으로 객체 탐지 특징과 재식별(ReID) 특징을 모델링하기 위해 과거의 임베딩(embedding) 특징을 활용합니다.

- **Technical Details**: STCMOT는 크게 세 가지 모듈로 구성됩니다: 인접 프레임 특징 추출기, temporal embedding boosting module (TEBM), temporal detection refinement module (TDRM)입니다. TEBM은 인접 프레임의 ReID 특징 맵을 통합하여 개별 임베딩의 차별성을 높입니다. TDRM은 비디오 시퀀스 내에서 추적된 궤적의 연속성을 활용하여 기사의 위치를 세밀하게 개선합니다.

- **Performance Highlights**: STCMOT는 VisDrone2019 및 UAVDT 데이터셋에서 기존 최첨단 성능을 초과하는 새로운 성과를 달성했으며, MOTA(Multiple Object Tracking Accuracy) 및 IDF1 메트릭에서 탁월한 결과를 보였습니다.



### Generalized Few-Shot Semantic Segmentation in Remote Sensing: Challenge and Benchmark (https://arxiv.org/abs/2409.11227)
Comments:
          7 pages, 3 figures, and 2 tables

- **What's New**: 이 연구는 원격 탐지(remote sensing) 분야에서의 일반화된 few-shot semantic segmentation 벤치마크를 최초로 제안합니다. 기존의 데이터셋들과 기준들을 넘어, 이 벤치마크는 훈련된 기본 클래스(base classes)에서의 성능 유지를 요구합니다.

- **Technical Details**: OpenEarthMap 데이터셋을 기반으로 한 OEM-GFSS 과정을 통해, 8개의 클래스에서 15개의 세부 클래스까지 확장된 지구 통계의 정확성을 높였습니다. 이는 submeter-level 토지 피복(mapping) 및 일반화된 few-shot 평가 설정을 위한 데이터입니다.

- **Performance Highlights**: 이 연구에서는 OEM-GFSS 데이터의 두 개의 단계에서 벤치마크 결과를 제공하며, 연구자들이 낮은 학습 자원 환경에서도 성능을 발휘할 수 있는 기본선을 설정했습니다.



### A Human-Centered Risk Evaluation of Biometric Systems Using Conjoint Analysis (https://arxiv.org/abs/2409.11224)
- **What's New**: 이 논문에서는 생체 인식 시스템의 공격자 동기에 미치는 위험 요소의 영향을 정량화하기 위해 신규 인간 중심 위험 평가 프레임워크를 제안합니다.

- **Technical Details**: 이 프레임워크는 다양한 위험 요인이 공격자의 동기 발생 확률에 미치는 영향을 평가하기 위해 conjoint analysis를 적용하였습니다. 이를 통해 위험 요인에 따라 공격 발생 확률을 정량화하고, 이를 바탕으로 False Acceptance Rate (FAR)와 공격 발생 확률을 반영하여 위험 가치를 계산합니다.

- **Performance Highlights**: 600명의 일본 참가자를 대상으로 실시한 조사 결과, 제안된 방법의 효과성을 입증하였습니다. 보안 조치가 공격자의 동기에 미치는 영향을 수치적으로 제시하며, 사용자맞춤형 생체 인식 시스템 구성이 가능하음을 보여줍니다.



### Multimodal Attention-Enhanced Feature Fusion-based Weekly Supervised Anomaly Violence Detection (https://arxiv.org/abs/2409.11223)
- **What's New**: 이번 연구는 약한 감독 하의 비디오 이상 탐지(Weakly Supervised Video Anomaly Detection, WS-VAD) 분야에서 RGB 비디오, optical flow, 오디오 신호 등 세 가지 특징 스트림을 이용하여 뛰어난 정확도와 강건성을 가지는 지능형 감시 시스템을 개발하였습니다.

- **Technical Details**: 첫 번째 스트림에서는 RGB 비디오에서 공간적(spatial) 및 시간적(temporal) 특징을 개선하기 위해 주의 기반(attention-based) 다단계(feature enhancement) 접근 방식을 사용하였습니다. 이 과정에서 ViT 기반 CLIP 모듈과 I3D 및 Temporal Contextual Aggregation (TCA)이 결합된 특징을 이용하여 풍부한 spatiotemporal 특징을 추출합니다. 두 번째 스트림은 흐름 데이터(modality-based feature)에서 주의(attention) 모듈을 활용하여 향상된 spatiotemporal 특징을 추출하였으며, 세 번째 스트림은 VGGish 모델을 통합하여 오디오에서 청각 신호를 탐지합니다.

- **Performance Highlights**: 세 가지 데이터셋을 통해 제안된 시스템이 기존의 최첨단 시스템 대비 상당히 향상된 성능을 보여주었으며, 모달리티(concatenation of multimodal fusion)의 장점을 활용하여 각 모달리티의 특징을 통합함으로써 이상 탐지의 정확도와 강건성을 크게 향상시켰습니다.



### Score Forgetting Distillation: A Swift, Data-Free Method for Machine Unlearning in Diffusion Models (https://arxiv.org/abs/2409.11219)
- **What's New**: 이 논문에서는 안전하고 신뢰할 수 있는 생성형 인공지능(Generative AI, GenAI) 모델 개발을 위한 기초로 '기계 비학습(Machine Unlearning, MU)'의 중요성을 강조하며, 전통적인 MU 방법의 한계를 극복하는 새로운 방법론인 '점수 망각 증류(Score Forgetting Distillation, SFD)'를 제안합니다.

- **Technical Details**: SFD는 사전 학습된 확산 모델의 점수 증류 목표에 점수 기반 MU 손실을 포함하여, '안전하지 않은' 클래스의 조건부 점수를 '안전한' 클래스의 점수와 정렬하도록 허용합니다. 이 방식은 생성 능력을 유지하고, 1단계 생성기를 통해 합성 데이터를 생성할 수 있도록 합니다. 실험 결과, SFD는 특정 클래스나 개념의 망각을 가속화하면서도, 다른 클래스의 품질을 보존하는 효과를 보여주었습니다.

- **Performance Highlights**: 우리의 방법은 CIFAR-10과 STL-10 데이터셋에서 사전 학습된 클래스 조건부 확산 모델에 대한 실험을 통해 목표 클래스를 효과적으로 지우며, 다른 클래스의 이미지 생성 품질을 유지합니다. Inception Score(IS), Fréchet Inception Distance(FID), Unlearning Accuracy(UA) 등의 성능 지표가 유의미하게 향상되었으며, 안정적인 확산 모델에 대한 실험에서도 특정 텍스트 입력과 관련된 개념을 성공적으로 제거하는 성과를 보였습니다.



### SplatFields: Neural Gaussian Splats for Sparse 3D and 4D Reconstruction (https://arxiv.org/abs/2409.11211)
Comments:
          ECCV 2024 paper. The project page and code are available at this https URL

- **What's New**: 본 연구에서는 3D Gaussian Splatting (3DGS) 방법이 지닌 한계를 극복하기 위한 새로운 최적화 전략을 제안합니다.

- **Technical Details**: 우리의 접근법은 splat features의 공간 자기상관(spatial autocorrelation) 부족을 해결하기 위해, 이를 상응하는 implicit neural field의 출력으로 모델링하여 정규화합니다. 이로 인해 다양한 시나리오에서 재구성 품질이 향상됩니다.

- **Performance Highlights**: 정적(static) 및 동적(dynamic) 장면 모두에서 이 접근법의 성능을 확인하였으며, 복잡한 장면 구성에서도 일관된 재구성 품질 개선이 이루어졌습니다.



### High-Order Evolving Graphs for Enhanced Representation of Traffic Dynamics (https://arxiv.org/abs/2409.11206)
- **What's New**: 이 논문은 자율주행 차량의 동적 교통 장면 해석을 위한 고차원 진화 그래프(High-Order Evolving Graphs)를 활용한 혁신적인 프레임워크를 제안합니다. 이 방식은 실시간으로 교통 환경 내 복잡한 상호작용을 모델링하는 데 효과적인 양방향 이분 그래프(temporal bidirectional bipartite graphs)를 생성합니다.

- **Technical Details**: 이 접근법은 그래프 신경망(Graph Neural Networks, GNNs)과 고차원 다중 집계 전략(high-order multi-aggregation strategies)을 통합하여 교통 장면의 동적 모델링을 향상시킵니다. 또한, GraphSAGE 프레임워크에서 영감을 받은 유도 학습(inductive learning) 기법을 포함하여, 새로운 교통 시나리오에 적응 가능하도록 설계되었습니다.

- **Performance Highlights**: ROAD 및 ROAD Waymo 데이터셋에 대한 광범위한 실험을 통해 교통 행동을 정확하게 포착할 수 있는 우리의 방법론의 가능성을 입증했습니다. 고차원 통계적 모멘트와 특징 게이팅 주의 메커니즘(feature-gated attention mechanism)이 교통 행동 분석을 개선하는 데 중요한 역할을 하였으며, 자율주행 기술의 발전 기반을 마련하였습니다.



### HS3-Bench: A Benchmark and Strong Baseline for Hyperspectral Semantic Segmentation in Driving Scenarios (https://arxiv.org/abs/2409.11205)
Comments:
          Accepted for publication at 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2024)

- **What's New**: 이 논문에서는 HS3-Bench라는 하이퍼스펙트럴(HSI) 시맨틱 세분화(semantic segmentation) 벤치마크를 소개합니다. 이 벤치마크는 운전 시나리오에 초점을 맞추고 있으며, 세 가지 데이터셋의 주석이 포함된 하이퍼스펙트럴 이미지를 결합하고 표준화된 메트릭과 평가 프로토콜을 제공합니다.

- **Technical Details**: HS3-Bench는 HyKo2, HSI-Drive, HCV2라는 세 가지 기존 데이터셋을 기반으로 구축되었습니다. 이 연구에서는 U-Net 아키텍처를 기반으로 한 모델과 추가 RGB 학습 데이터를 활용한 DeeplabV3+ 모델을 포함한 두 가지 강력한 베이스라인 모델을 제안합니다. 벤치마크의 구현 및 평가와 관련된 다양한 메트릭(Accuracy, F1-Score, Jaccard Index 등)을 수록하고 있습니다.

- **Performance Highlights**: 두 개의 제안된 베이스라인 모델은 이전의 최첨단 성능을 능가했으며, 기존 학습 기반 방법은 추가 HSI 채널을 활용하기보다는 추가 RGB 학습 데이터에서 더 많은 이점을 얻는 것으로 나타났습니다. 이는 하이퍼스펙트럴 이미징 분야에서의 향후 연구에 대한 중요한 질문을 제기합니다.



### Deep Learning tools to support deforestation monitoring in the Ivory Coast using SAR and Optical satellite imagery (https://arxiv.org/abs/2409.11186)
- **What's New**: 이번 연구에서는 코코아 생산이 활발한 코트디부아르에서의 산림 파괴를 모니터링하기 위해 Sentinel 위성 이미지를 이용한 Forest-Non-Forest map (FNF) 작성 방법이 제안되었습니다.

- **Technical Details**: U-Net, Attention U-Net, Segnet, FCN32와 같은 최신 모델들이 Sentinel-1 및 Sentinel-2 이미지를 결합하여 삼림/비삼림(segmentation) 분류에 사용되었으며, 구름이 자주 낀 지역에서도 효과적인 분석이 가능한 SAR (Synthetic Aperture Radar) 이미지를 활용하였습니다.

- **Performance Highlights**: 연구를 통해 2019년과 2020년 사이에 제거된 삼림 면적을 추정할 수 있었으며, 개방형 데이터셋을 통한 산림 및 비산림 픽셀 분류 모델의 가능성이 입증되었습니다.



### Video Token Sparsification for Efficient Multimodal LLMs in Autonomous Driving (https://arxiv.org/abs/2409.11182)
Comments:
          10 pages, 3 figures, 4 tables

- **What's New**: 이번 연구에서는 Video Token Sparsification (VTS)라는 새로운 접근 방식을 제안하여 자율주행 시스템에서 MLLMs의 시각적 토큰 수를 효과적으로 감소시키는 방법을 다룹니다. 이 방법은 비디오 프레임 간의 중복성을 활용하여 많은 양의 시각적 정보를 유지하면서 불필요한 토큰을 제거합니다.

- **Technical Details**: VTS는 경량의 CNN 기반 제안 모델을 사용하여 키 프레임을 선택하고 상대적으로 덜 유용한 토큰을 제거합니다. 이를 통해 불필요한 계산 부하를 최소화하면서도 MLLMs의 성능을 유지합니다. 또한, 이 연구에서는 DRAMA와 LingoQA 벤치마크를 활용하여 VTS의 효과를 입증하였습니다.

- **Performance Highlights**: VTS는 최대 33%의 추론 처리량 향상과 28%의 메모리 사용량 감소를 보여주며, 성능을 저하시키지 않고 자율주행 시나리오에서 다양한 비디오 질의 응답 작업을 수행할 수 있음을 입증하였습니다.



### Synthetic data augmentation for robotic mobility aids to support blind and low vision peop (https://arxiv.org/abs/2409.11164)
- **What's New**: 이 연구에서는 블라인드 및 저시력(BLV) 개인을 위한 로봇 이동 보조 기기의 신뢰성을 높이기 위한 합성 데이터의 효과를 조사했습니다. Unreal Engine 4를 활용하여 합성 데이터를 생성하고, 이를 통해 로봇 시스템의 인식 모델 성능을 향상시킬 수 있음을 보여줍니다.

- **Technical Details**: 연구진은 NVIDIA Deep Learning Dataset Synthesizer(NDDS) 플러그인을 사용하여 Unreal Engine 4를 기반으로 한 합성 데이터 생성 파이프라인을 개발했습니다. 이 데이터셋은 촉각 포장 탐지(tactile paving detection) 및 장면 설명(scene description)의 주요 작업에 맞춰져 있으며, 다양한 조명 및 날씨 조건을 시뮬레이션하여 현실적인 데이터를 생성합니다.

- **Performance Highlights**: 합성 데이터를 활용한 모델이 촉각 포장 탐지 및 장면 설명 작업에서 향상된 성능을 보였으며, 이를 통해 BLV 개인을 위한 이동 보조 기술의 발전에 기여할 수 있는 귀중한 통찰력을 제공합니다. 연구진은 합성 데이터셋을 공개하여 추가 연구를 지원합니다.



### UltimateDO: An Efficient Framework to Marry Occupancy Prediction with 3D Object Detection via Channel2heigh (https://arxiv.org/abs/2409.11160)
- **What's New**: 본 연구에서는 3D 객체 탐지(3D object detection)와 점유 예측(occupancy prediction) 작업을 통합하려는 새로운 프레임워크인 UltimateDO를 제안하고 있습니다. 이는 2D convolution을 활용해 모델을 재구성하며, 두 작업이 상호 연관성을 가지도록 작업을 우선시하여 보다 높은 효율성을 얻고자 합니다.

- **Technical Details**: UltimateDO는 FlashOcc를 통해 경량의 점유 예측 헤드를 3D 객체 탐지 네트워크에 결합하며, 추가적인 시간 소모는 1.1ms에 불과합니다. 입력으로 주어진 주변 이미지(surround-view images)는 2D 이미지 인코더를 통해 고급 피처(high-level features)로 변환됩니다. 이 후, BEV(Bird's Eye View) 표현으로 변환되어 BEV 인코더를 거쳐 세분화된 BEV 특성을 생성합니다.

- **Performance Highlights**: 최신의 nuScenes 데이터셋에서 UltimateDO는 SOTA(state-of-the-art) 성능을 달성하며, 점유 예측과 객체 탐지 작업에서 높은 정확도를 보입니다. 또한, 경량화된 점유 예측 헤드를 통합했음에도 불구하고 연산 시간은 최소화되었습니다.



### Scale generalisation properties of extended scale-covariant and scale-invariant Gaussian derivative networks on image datasets with spatial scaling variations (https://arxiv.org/abs/2409.11140)
Comments:
          50 pages, 23 figures, 16 tables

- **What's New**: 본 논문은 scale-covariant 및 scale-invariant Gaussian derivative networks의 스케일 일반화(scale generalisation) 특성에 대한 심층 분석을 제공합니다. 실험적으로 새로운 데이터셋인 Fashion-MNIST와 CIFAR-10에서의 평가와 함께, 기존의 STIR 데이터셋에서의 결과를 비교하여 이전의 딥 네트워크보다 성과를 개선했다고 보고합니다.

- **Technical Details**: Gaussian derivative networks는 다양한 스케일에서의 객체를 처리하는 데 효과적이며, 평균 풀링(average pooling) 접근 방식이 때때로 이전의 최대 풀링(max pooling) 접근 방식보다 더 나은 결과를 도출할 수 있음을 입증합니다. 또한, 최종 레이어 후에 공간 최대 풀링(spatial max pooling) 메커니즘을 사용하여 비중심(non-centred) 객체의 위치를 확인할 수 있으며, 정규화(regulation)로서 scale channel dropout을 적용하면 성능과 스케일 일반화가 개선됩니다.

- **Performance Highlights**: 이 모델은 새로운 데이터셋에서 스케일 일반화 특성을 잘 수행하며, Gaussian derivative 네트워크는 설명 가능성(explainability) 속성이 뛰어남을 시각화하여 보여줍니다. 실험적으로 기존의 다양한 방법과 비교하여 여전히 높은 성능을 유지함을 확인하였으며, 일반화 개념의 실질적인 구현 가능성을 제시합니다.



### Genetic Information Analysis of Age-Related Macular Degeneration Fellow Eye Using Multi-Modal Selective V (https://arxiv.org/abs/2409.11128)
- **What's New**: 이번 연구에서는 나이 관련 황반 변성(Age-related Macular Degeneration, AMD)과 관련된 다수의 유전자 감수성(gene susceptibility)을 예측하기 위해 다양한 의료 이미지를 활용하는 방법을 제시하고 있습니다. 이 방법은 fundus 이미지, Optical Coherence Tomography (OCT) 이미지 및 의료 기록을 통합하여 80% 이상의 정확도로 유전자 감수성을 예측하는 능력을 입증하였습니다.

- **Technical Details**: 제안된 방법인 MSViT는 다중 모달 임베딩(Multi-Modal Embedding, MME)과 선택적 변환기(Selective Transformer, ST)로 구성되어 있습니다. 이 모델은 의료 기록 정보를 테이블 토큰으로 변환하고 이미지 데이터를 각각의 패치 임베딩을 통해 처리하여 이들 간의 상관관계를 분석합니다. 선택적 주의(selective attention)와 CNN을 통해 정보 추출을 강화하고, 비대칭적 정보(content) 문제를 해결하기 위해 동시 학습 방식으로 의료 기록을 재구성합니다.

- **Performance Highlights**: 실험 결과, 1,192세트의 fundus 이미지, OCT 이미지 및 의료 기록을 사용하여 ARMS2와 CFH 유전자의 감수성 유전자(risk alleles)를 80% 이상의 정확도로 예측할 수 있음을 보여주었습니다.



### Quantitative Evaluation of MILs' Reliability For WSIs Classification (https://arxiv.org/abs/2409.11110)
- **What's New**: 이 논문은 의료 분야의 신뢰성이 중요한 Multi Instance Learning (MIL) 모델의 신뢰성을 최초로 정량적으로 평가하는 방법을 제시합니다. 세 가지 지표인 Mutual Information, Spearman의 상관관계, Precision Recall Curve의 면적을 사용하여 MIL 모델의 비교를 수행하였습니다.

- **Technical Details**: Whole Slide Image (WSI) 분류를 위한 MIL 모델을 평가하기 위해, 데이터셋으로는 Camelyon16, CATCH, TCGA BRCA를 사용했습니다. MIL 방법론을 통해 슬라이드를 N개의 작은 패치로 분할하고, 각 패치를 기반으로 주어진 슬라이드의 레이블을 예측합니다. 신뢰성을 평가하기 위해 세 가지 지표를 사용했습니다.

- **Performance Highlights**: Mean pooling instance (MEAN-POOL-INS) 모델이 다른 네트워크보다 더 신뢰성이 높다고 평가했습니다. 이를 통해 기존의 모델들이 직면한 여러 문제들을 해결하는 데 기여하며, 의료 분야에서의 신뢰성 높은 MIL 모델 개발에 기여할 것으로 기대됩니다.



### Depth-based Privileged Information for Boosting 3D Human Pose Estimation on RGB (https://arxiv.org/abs/2409.11104)
Comments:
          ECCV 2024 Workshop T-CAP: TOWARDS A COMPLETE ANALYSIS OF PEOPLE: FINE-GRAINED UNDERSTANDING FOR REAL-WORLD APPLICATIONS

- **What's New**: 이 연구는 RGB 이미지에서 3D 인간 포즈를 추정하는 데 Depth 정보의 이점을 획기적으로 활용하며, 주목할 만한 것은 훈련 시에만 Depth 정보를 사용하는 Privileged Information의 패러다임을 적용한 첫 번째 사례라는 점입니다.

- **Technical Details**: 연구에서는 heatmap 기반의 3D 포즈 추정기를 제안하며, RGB 이미지로부터 Hallucinated(환각된) Depth 정보를 학습합니다. 이 방법은 Semi-Perspective Decoupled Heatmaps (SPDH)를 사용하여 3D 인간 포즈를 직접 예측합니다. 훈련 시 RGB와 함께 Depth 데이터를 사용하고, 추론 시에는 RGB만을 사용하여 포즈를 추정합니다.

- **Performance Highlights**: 실험 결과, Privileged Information 패러다임이 모델의 성능을 유의미하게 향상시키며, RGB 이미지만으로 Depth 정보를 효율적으로 추출할 수 있다는 점을 증명하였습니다. 이 방법은 작은 데이터셋에서도 뛰어난 효과를 보이며, 다양한 응용 가능성을 확보합니다.



### ShapeAug++: More Realistic Shape Augmentation for Event Data (https://arxiv.org/abs/2409.11075)
Comments:
          accepted in Lecture Notes in Computer Science (LNCS)

- **What's New**: ShapeAug++라는 새로운 데이터 증강 방법이 제안되었습니다. 이 방법은 임의로 생성된 다각형과 곡선 운동을 포함하여 기존의 ShapeAug보다 복잡성을 크게 높였습니다.

- **Technical Details**: ShapeAug++는 곡선 움직임과 여러 가지 복잡한 다각형을 생성하여 선형 경로를 따라 움직이는 객체를 시뮬레이션하는 방법론을 가지고 있습니다. 이 과정에서 Spiking Neural Network (SNN)를 활용하여 시간 단계를 신경망에 순차적으로 입력합니다.

- **Performance Highlights**: ShapeAug++는 여러 DVS 분류 데이터셋에서 ShapeAug와 비교하여 최대 3.7%의 top-1 정확도를 향상시켰습니다.



### OneEncoder: A Lightweight Framework for Progressive Alignment of Modalities (https://arxiv.org/abs/2409.11059)
- **What's New**: 이번 논문에서는 다양한 모달리티(모드) 간의 정렬을 점진적으로 수행할 수 있는 새로운 경량 프레임워크인 OneEncoder를 제안합니다. 이 프레임워크는 이미지, 텍스트, 오디오, 비디오를 포함한 네 가지 모달리티를 통합하여 부하가 적고 비용 효율적인 방식으로 작동합니다.

- **Technical Details**: OneEncoder는 우선 이미지와 텍스트 모달리티를 정렬하기 위해 Lightweight Universal Projection 모듈(UP)을 학습하며, 이후 그 모듈을 고정하여 새로운 모달리티를 점진적으로 정렬합니다. 이 과정에서 Alignment Layer(AL)은 피처 추출을 위한 전용 변환 공정을 단순화합니다. 사용된 인코더로는 ViT(Visual Transformer), BERT, Wav2Vec2, VideoMAE가 포함됩니다.

- **Performance Highlights**: OneEncoder는 적은 양의 쌍 데이터로 훈련되면서도 다양한 다운스트림 작업(예: 분류, 쿼리, 시각적 질문 응답 등)에서 기존 방법론들을 초월하는 성능을 발휘하고 있습니다. 이는 대규모 데이터셋에 의존하는 모델들보다 구조적으로 더 효율적입니다.



### Down-Sampling Inter-Layer Adapter for Parameter and Computation Efficient Ultra-Fine-Grained Image Recognition (https://arxiv.org/abs/2409.11051)
Comments:
          Accepted to ECCV 2024 Workshop on Efficient Deep Learning for Foundation Models (EFM). Main: 13 pages, 3 figures, 2 tables. Appendix: 3 pages, 1 table. Total: 16 pages, 3 figures, 4 tables

- **What's New**: 이번 논문에서는 초미세 분류(UFGIR) 작업을 위한 새로운 접근 방식을 제안합니다. 이 방법은 파라미터 효율적인 설정에서 하위 샘플링 인터 레이어 어댑터(down-sampling inter-layer adapters)를 사용하여 Backbone 매개변수를 동결하고 추가 모듈만 미세 조정합니다.

- **Technical Details**: 제안된 방법은 중간 레이어 어댑터(Intermediate Layer Adapter, ILA)를 사용하여 두 가지 공간적 다운 샘플링 가지를 통합합니다. 이 구조는 비파라메트릭(feature)을 효과적으로 집계하고, 파라미터 수와 부동 소수점 연산(Floating-Point Operations, FLOPs)의 수를 상당히 줄여줍니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 다른 방법들보다 평균 정확도를 최소 6.8% 향상시키며, 현재의 가장 앞선 UFGIR 방법에 비해 적어도 123배 적은 학습 가능한 매개변수를 요구하고, FLOPs는 평균적으로 30% 감소합니다.



### Estimating the distribution of numerosity and non-numerical visual magnitudes in natural scenes using computer vision (https://arxiv.org/abs/2409.11028)
- **What's New**: 이번 연구에서는 자연적 환경에서의 수량 인식(numerosity perception)을 자동으로 처리 및 측정할 수 있는 혁신적인 컴퓨터 비전 파이프라인을 제안합니다. 이 파이프라인은 대규모 데이터셋에서 수량과 비수적 크기(non-numerical magnitudes)의 분포를 추정할 수 있도록 설계되었습니다.

- **Technical Details**: 제안된 파이프라인은 여러 최첨단 AI 기술의 결합에 기반하고 있으며, 첫 번째 단계에서 다중 모드 대형 언어 모델(multimodal large language model)을 사용해 원시 픽셀에서 고수준의 의미 정보를 추출합니다. 이후, 개방형 객체 탐지기(open-set object detector)가 각 객체의 정확한 위치(바운딩 박스)를 반환하며, 마지막으로 분할 모델(segmentation model)을 통해 객체의 실루엣을 추출합니다.

- **Performance Highlights**: MSCOCO 및 PASCAL 데이터셋을 이용한 검증을 통해 파이프라인이 사람의 주석과 유사한 정확도로 객체를 탐지하고 분할할 수 있음을 입증하였습니다. 이 연구는 인간의 수량 인식 연구에 새로운 방향을 제시하며, 자연적 개발 환경의 통계적 특성을 분석하는 기초가 될 것입니다.



### Unleashing the Potential of Mamba: Boosting a LiDAR 3D Sparse Detector by Using Cross-Model Knowledge Distillation (https://arxiv.org/abs/2409.11018)
- **What's New**: 본 논문에서는 Faster LiDAR 3D 객체 탐지 프레임워크인 FASD를 제안합니다. 이 프레임워크는 비균질 모델 증류(heterogeneous model distillation)를 통해 시공간 스팸(Spatial sparseness)을 최적화하여 속도와 정확도 사이의 균형을 맞추고 있습니다.

- **Technical Details**: FASD는 동적 voxel 그룹화(Dynamic Voxel Group) 및 적응형 주의 전략(Adaptive Attention)을 활용하여 효율적이고 정확한 LiDAR 기반 3D 객체 탐지를 가능하게 합니다. 교사 모델은 글로벌 맥락 모델링(global context modeling)을 위한 스케일 적응형 주의 메커니즘(scale-adaptive attention)을 통합하여 기능을 효과적으로 학습합니다. 학생 모델은 Mamba 모델을 기반으로 하여 [FLOPs]를 적게 소모하며 성능을 향상합니다.

- **Performance Highlights**: Waymo 및 nuScenes 데이터셋에서 4배의 자원 소비 감소와 함께 현재의 최첨단 방법에 비해 1-2%의 성능향상을 달성했습니다. 이러한 결과는 실시간 인식의 효율성을 높이고 LiDAR 3D 객체 탐지의 실제 적용 가능성을 증가시키는 데 기여합니다.



### MM2Latent: Text-to-facial image generation and editing in GANs with multimodal assistanc (https://arxiv.org/abs/2409.11010)
Comments:
          Accepted at ECCV 2024 AIM workshop

- **What's New**: 이 논문에서는 MM2Latent라는 새로운 멀티모달 이미지 생성 및 편집 프레임워크를 제안합니다. MM2Latent는 StyleGAN2를 이미지 생성기로 사용하고, 텍스트 인코딩을 위해 FaRL을 활용하며, 마스크, 스케치, 3DMM과 같은 공간적 모달리티를 위한 오토인코더를 훈련합니다.

- **Technical Details**: 제안된 접근 방식은 멀티모달 입력을 StyleGAN의 w latent space로 매핑하기 위해 매핑 네트워크를 훈련하는 전략을 포함합니다. 이 네트워크는 이미지 임베딩에 대해 훈련되지만 추론 단계에서는 텍스트 임베딩을 수용할 수 있습니다. MM2Latent는 하이퍼파라미터 조정과 수동 작업을 제거하고, 빠른 추론 속도와 실제 이미지 편집 기능을 보장합니다.

- **Performance Highlights**: 광범위한 실험을 통해 MM2Latent가 현재의 최첨단 방법들에 비해 멀티모달 일관성, 이미지 품질 및 추론 속도에서 우수한 성능을 보였음을 입증합니다. MM2Latent는 텍스트와 마스크, 스케치, 또는 3DMM을 결합하여 제어 가능한 얼굴 이미지 생성을 가능하게 하고, 실제 이미지의 대화형 편집을 위한 여러 제어 기능을 제공합니다.



### Versatile Incremental Learning: Towards Class and Domain-Agnostic Incremental Learning (https://arxiv.org/abs/2409.10956)
Comments:
          17 pages, 6 figures, 6 tables, ECCV 2024 Poster

- **What's New**: 본 연구에서는 기존의 Incremental Learning (IL) 방법론에 비해 더 현실적이고 도전적인 상황인 Versatile Incremental Learning (VIL)을 제안합니다. VIL은 다음 작업에서 어떤 클래스나 도메인이 증가할지에 대한 사전 정보가 없으며, 주로 intra-class domain confusion과 inter-domain class confusion 문제를 다룹니다.

- **Technical Details**: VIL 스cenario에 대응하기 위해 Incremental Classifier with Adaptation Shift cONtrol (ICON)라는 새로운 IL 프레임워크를 제안하며, Cluster-based Adaptation Shift conTrol (CAST)라는 정규화 방법을 통해 모델이 이전에 학습한 지식과의 혼동을 피하고 새로운 지식을 효과적으로 습득할 수 있도록 합니다. 또한, Incremental Classifier (IC)를 도입하여 다양한 도메인에서 학습 시 과거 지식을 유지하면서 출력 노드를 확장합니다.

- **Performance Highlights**: ICON은 3개의 벤치마크에서 수행된 실험을 통해 모든 시나리오에서의 성능이 입증되었으며, 특히 다음 작업이 랜덤하게 변경될 수 있는 경우에 특징적으로 뛰어난 성능을 보였습니다. 본 연구의 결과는 기존의 IL 방법들에 비해 월등한 성능 향상을 보여줍니다.



### HGSLoc: 3DGS-based Heuristic Camera Pose Refinemen (https://arxiv.org/abs/2409.10925)
- **What's New**: HGSLoc라는 새로운 경량의 플러그 앤 플레이(pose optimization) 프레임워크를 제안하여 3D 재구성과 휴리스틱 정제 전략을 통합함으로써 포즈 추정 정확성을 향상시킵니다.

- **Technical Details**: HGSLoc는 명시적 기하학적 맵(geometric map)을 통한 3D 표현과 고충실도 렌더링(high-fidelity rendering)을 제공하여 고품질의 합성 뷰(synthesized views)를 생성합니다. 또한, 3D Gaussian Splatting(3DGS)을 활용하여 장면 포인트(scene points)를 가우시안 분포(Gaussian distributions)로 표현함으로써 데이터 처리 부하를 줄입니다. 그리고, 효율적인 경로 탐색(pathfinding) 기능을 갖춘 헤uristic 정제 알고리즘을 통해 추정된 포즈의 정확성을 최적화합니다.

- **Performance Highlights**: 제안된 HGSLoc 방법은 NeRF 기반(neural rendering localization approaches) 모형에 비해 렌더링 속도와 로컬라이제이션(localization) 정확성을 개선하였으며, 다양한 벤치마크 데이터 세트(7Scenes, DB 등)에서 강력한 성능을 입증했습니다.



### KALE: An Artwork Image Captioning System Augmented with Heterogeneous Graph (https://arxiv.org/abs/2409.10921)
Comments:
          Accepted at IJCAI 2024

- **What's New**: KALE(Knowledge-Augmented vision-Language model for artwork Elaborations)는 기존의 vision-language 모델을 강화하여 미술 작품의 메타데이터를 통합함으로써 작품의 의미에 대한 심층 해석을 가능하게 하는 새로운 접근 방식을 제시합니다.

- **Technical Details**: KALE는 두 가지 방식으로 메타데이터를 통합합니다. 첫 번째는 텍스트 입력으로 직접 포함하는 것이고, 두 번째는 다중 모드 이질적 지식 그래프(multimodal heterogeneous knowledge graph)를 통해서입니다. 새로운 cross-modal alignment loss를 도입하여 이미지와 그에 해당하는 메타데이터 간의 유사성을 극대화합니다. 이 시스템은 선행 학습된 비전-언어 모델을 미술 영역으로 확대하여 이미지 설명을 생성합니다.

- **Performance Highlights**: KALE는 여러 미술 작품 데이터셋에서 CIDEr로 평가했을 때 현존하는 최첨단 기술을 능가하는 성능을 보여줍니다. 실험 결과, KALE는 기존의 미술 작품 이미지 캡셔닝 모델을 상당히 초월하는 성능을 입증하며, 미술 작품의 내러티브에 대한 이해를 더욱 촉진할 수 있음을 나타냅니다.



### AMEGO: Active Memory from long EGOcentric videos (https://arxiv.org/abs/2409.10917)
Comments:
          Accepted to ECCV 2024. Project webpage: this https URL

- **What's New**: 본 논문에서는 egocentric 비디오의 이해를 향상시키기 위한 새로운 접근법인 AMEGO를 소개합니다. AMEGO는 한 번의 시청만으로도 정보를 저장하는 인간의 능력에서 영감을 받아, 키 위치 및 객체 상호작용을 포착한 자기 포함적 representation을 구축합니다.

- **Technical Details**: AMEGO는 'hand-object interaction (HOI) tracklets'과 'location segments'로 구성되어 있습니다. HOI tracklets는 카메라 착용자와 객체 간의 일관된 상호작용을 포함하고, location segments는 특정 위치에서 활동을 수행하는 동안의 시간 간격을 나타냅니다. 이 접근법은 'visual perception models'을 사용하여 활동과 움직임의 정보를 캡처합니다.

- **Performance Highlights**: AMEGO는 새로운 Active Memories Benchmark (AMB)에서 다른 비디오 QA 기준선을 12.7% 초과하는 성능을 보이며, 활성 객체, 위치 및 그 상호작용을 포함한 20K 이상의 비주얼 질문-답변 쌍을 다룹니다.



### TrajSSL: Trajectory-Enhanced Semi-Supervised 3D Object Detection (https://arxiv.org/abs/2409.10901)
- **What's New**: 이 연구는 자율주행 인식 데이터셋에서 수동 레이블링의 필요성을 줄이기 위해 세미-슈퍼바이즈드 3D object detection을 개선하는 방법을 제시합니다. 이 방법은 모션 예측 모델을 활용하여 pseudo-label의 품질을 향상시킵니다.

- **Technical Details**: 제안된 방법인 TrajSSL은 teacher-student 아키텍처를 바탕으로 하며, labeled 데이터와 unlabeled 데이터 간의 pseudo-labels의 품질을 높이기 위해 여러 프레임 간의 일관성을 유지하고, 예측된 객체 경로를 사용하여 false negative를 보완합니다.

- **Performance Highlights**: nuScenes 데이터셋에서의 실험을 통해, 제안된 접근 방식이 기존의 세미-슈퍼바이즈드 3D object detection 방법보다 mAP의 절대적인 향상을 보여줍니다.



### Shaking the Fake: Detecting Deepfake Videos in Real Time via Active Probes (https://arxiv.org/abs/2409.10889)
- **What's New**: SFake는 스마트폰에서 실시간으로 딥페이크를 탐지하는 새로운 방법을 제안하며, 물리적 간섭을 이용한 능동적인 특징 검색을 통해 탐지 정확도를 향상시킵니다.

- **Technical Details**: SFake는 스마트폰에 진동을 유도하는 프로브를 보내며, 촬영된 비디오의 얼굴 영역과 프로브 패턴의 일관성을 통해 딥페이크 여부를 판단합니다. 1920x1080 픽셀 이상의 해상과 2배 줌 이상의 카메라를 요구합니다.

- **Performance Highlights**: SFake의 탐지 정확도는 95.2% 이상이며, 처리 속도는 5초 이하, 메모리 소비는 450MB 미만으로 다른 6개 방법보다 우수한 성능을 보였습니다.



### 3DFacePolicy: Speech-Driven 3D Facial Animation with Diffusion Policy (https://arxiv.org/abs/2409.10848)
- **What's New**: 본 논문은 3DFacePolicy라는 새로운 모델을 제안하여 오디오 기반의 3D 얼굴 애니메이션 생성에서 감정 표현과 사실감을 개선하고, 다양한 얼굴 움직임을 생성하는 데 중점을 두고 있습니다.

- **Technical Details**: 3DFacePolicy는 diffusion policy 모델을 활용하여 3D 얼굴 애니메이션을 예측하는 방법론입니다. 이 모델은 오디오 입력과 3D vertex 상태를 관찰하여 vertex trajectory를 예측하고, 실제 인간의 감정을 모방하여 자연스러운 흐름을 유지합니다. 모델은 복잡한 데이터 분포를 처리하기 위한 시퀀스 샘플러를 사용하여 동작을 매끄럽게 생성합니다.

- **Performance Highlights**: 3DFacePolicy는 VOCASET 벤치마크를 기준으로 다른 최신 방법들에 비해 주목할 만한 장점을 보이며, 유연하고 동적인 얼굴 움직임을 생성하는 데 있어 효과적임을 입증하였습니다.



### Single-Layer Learnable Activation for Implicit Neural Representation (SL$^{2}$A-INR) (https://arxiv.org/abs/2409.10836)
- **What's New**: SL$^{2}$A-INR은 learnable activation function을 도입하여 Implicit Neural Representation (INR)에서 고주파 세부 정보를 포착하는 능력을 enhance합니다. 이는 기존 MLP 아키텍처의 ReLU 기반 접근방식에 비해 개선된 성능을 보여줍니다.

- **Technical Details**: SL$^{2}$A-INR은 학습 가능한 activation function을 가진 hybrid network를 제안합니다. 이 네트워크는 Chebyshev approximation 방법을 사용하여 초기 레이어에서의 activation function의 정밀한 다항식 근사화를 통해 고주파 신호와 세부 정보를 포착합니다. 추가적으로 Fusion Block을 통해 subsequent ReLU기반 MLP에 skip connections을 추가해 다양한 주파수의 요소를 통합합니다.

- **Performance Highlights**: SL$^{2}$A-INR은 이미지 표현, 3D 형태 재구성, inpainting, 단일 이미지 super-resolution, CT 재구성 및 새로운 보기 합성 등 다양한 작업에서 뛰어난 성능을 나타냅니다. 실험을 통해 정확도, 품질, 수렴 속도에서 새로운 기준을 설정했습니다.



### Are Deep Learning Models Robust to Partial Object Occlusion in Visual Recognition Tasks? (https://arxiv.org/abs/2409.10775)
- **What's New**: 본 논문에서는 부분 가림(occlusion) 조건에서의 이미지 인식 성능을 검증하기 위해 Real-world 및 인공적으로 가려진 이미지로 구성된 Image Recognition Under Occlusion (IRUO) 데이터셋을 개발하였습니다. 이 데이터셋은 실제 환경에서의 가림 현상을 다루기 위한 대규모 공공 벤치마크로 기능하며, 현대의 CNN 모델과 Vision Transformer 모델의 성능을 검토합니다.

- **Technical Details**: IRUO 데이터셋은 Occluded Video Instance Segmentation (OVIS) 데이터셋을 기반으로 하며, 88,000장의 이미지와 23개의 객체 클래스를 포함합니다. 이 연구에서는 ResNeXt와 Swin 같은 최첨단 CNN 모델과 Vision Transformer들을 비교하여, 모델의 정확도를 평가하고 가림에 대한 상대적 강인성을 분석합니다. 또한, 인간 관찰자와 모델의 인식 성능을 비교하여, 인식 정확도가 어느 정도 차이가 나는지 확인합니다.

- **Performance Highlights**: 최신 CNN 기반 모델은 이전 모델보다 가려진 이미지에 대한 인식 정확도가 개선되었으며, Vision Transformer 모델은 CNN 기반 모델보다 높은 정확도를 보이며, 인간의 정확도와 비슷한 수준을 나타냅니다. 또한, 'diffuse occlusion' 같은 특정 가림 종류는 딥러닝 모델의 정확도를 크게 저하시킬 수 있음을 발견하였습니다.



### Depth from Coupled Optical Differentiation (https://arxiv.org/abs/2409.10725)
- **What's New**: 이번 연구에서는 depth from coupled optical differentiation을 제안하며, 이는 계산량이 적은 수동 조명 3D 센싱 메커니즘입니다. 이 메커니즘은 초점이 흐릿한 이미지의 광학 파생물 쌍을 이용하여 픽셀 단위의 객체 거리를 정확하게 결정할 수 있다는 발견에 기반하고 있습니다. 이 방법은 이전의 depth-from-defocus (DfD) 기법보다 노이즈에 대한 강인성이 크게 향상되었습니다.

- **Technical Details**: 제안된 센서는 변형 가능한 렌즈와 모터 구동 조리개를 포함하여 광학 파워와 조리개 크기를 동적으로 조정할 수 있으며, 두 쌍의 이미지를 캡처합니다. 각 이미지 쌍은 각각 광학 파워와 조리개 크기에 대한 차이를 갖습니다. 네 개의 이미지를 통해 깊이 및 신뢰도 맵을 생성할 수 있으며, 이 과정에서 단 36 floating point operations per output pixel (FLOPOP)의 계산량을 요구합니다.

- **Performance Highlights**: 제안된 센서는 이전 DfD 방법보다 배 이상의 작업 범위를 실현하며, 계산량은 이전의 가장 효율적인 수동 조명 깊이 센서의 10배 이상 낮은 성능을 보입니다. 또한, 제안된 이론은 장면의 모양에 무관하게 깊이를 정확히 추정할 수 있는 능력을 보여줍니다.



### A Missing Data Imputation GAN for Character Sprite Generation (https://arxiv.org/abs/2409.10721)
Comments:
          Published in SBGames 2024

- **What's New**: 본 연구에서는 다양한 애니메이션과 포즈를 포함하는 픽셀 아트 캐릭터 스프라이트(pixel art character sprites)를 생성하고 업데이트하는 과정에서 반복적이 되는 작업을 부분적으로 자동화할 수 있는 방법을 제안합니다. 특히, 다른 세 방향에서 캐릭터 이미지가 주어질 때 특정 포즈(target pose)로의 이미지 생성을 처리하는 새로운 접근 방식을 소개합니다.

- **Technical Details**: 이 연구에서는 generative adversarial networks (GANs)를 사용하여 캐릭터의 다양한 포즈를 위한 이미지를 생성하는 모델을 제안합니다. 제안된 모델은 CollaGAN 아키텍처에 기반해 있으며, 생성기(generator) 구조와 훈련 절차에 변경을 가했습니다. 대상 포즈의 이미지를 생성하기 위해 다른 포즈의 이미지를 입력으로 사용합니다. 주어진 입력 이미지들이 많을수록 생성된 이미지의 품질이 향상된다는 것을 실험적으로 증명하였습니다. 평가 지표로는 Frechét Inception Distance (FID)와 L1 distance를 사용했습니다.

- **Performance Highlights**: 단일 생성기/구분기를 사용하는 GAN 모델이 여러 캐릭터 포즈를 대상으로 작업할 수 있음을 보였으며, 입력으로 제공되는 이미지 수가 많을수록 생성된 스프라이트의 품질이 향상됨을 보여주었습니다. 제안된 변경 사항들이 기존의 CollaGAN에 비해 보다 나은 결과를 도출하는 데 기여하였습니다.



### Benchmarking VLMs' Reasoning About Persuasive Atypical Images (https://arxiv.org/abs/2409.10719)
- **What's New**: 이 논문에서는 광고와 같은 수사적이고 설득력 있는 시각 매체의 이해력을 평가하기 위해 비전 언어 모델(VLMs)에 대한 새로운 과제를 소개합니다. 특별히, 비정상적인 이미지(이상성)를 이해하는 능력을 테스트하는 세 가지 새로운 과제, 즉 Multi-label Atypicality Classification, Atypicality Statement Retrieval, Atypical Object Recognition을 제안합니다.

- **Technical Details**: VLMs는 광고의 메시지를 추론하는 데 필요한 비정상적인 요소를 이해하기 위해 고급 추론 능력을 요구합니다. 논문에서는 VLMs의 이해도를 평가하기 위해 심리적이며 의미적으로 도전적인 부정적인 예제들을 사용하는 action-reason retrieval (ARR) 작업을 소개합니다.

- **Performance Highlights**: 연구 결과, VLMs는 복잡한 비정상적 이미지 해석에서 LLMs보다 부족한 고급 추론 능력을 보였으며, VLM이 제안한 비정상적 언어화를 사용하더라도 ARR 작업에서 성능이 하락하는 결과를 보였습니다. 이는 VLMs가 수사적 광고의 메시지를 이해하는 데 있어 한계를 가지며, 효과적인 광고를 생성하기 위한 통찰력을 제공합니다.



### Online Learning via Memory: Retrieval-Augmented Detector Adaptation (https://arxiv.org/abs/2409.10716)
Comments:
          Accepted at ECCV 2024, Human-Inspired Computer Vision (HCV) workshop

- **What's New**: 본 논문은 기존의 객체 탐지 모델을 신규 도메인에 온라인으로 적응시키는 새로운 방법을 제시합니다. 이 과정에서 모델 재훈련 없이 메모리 내 유사 객체 개념을 검색하여 활용합니다. 이를 통해 아무리 적은 데이터(예: 카테고리 당 10개의 이미지)로도 객체 탐지를 효과적으로 수행할 수 있습니다.

- **Technical Details**: 제안된 프레임워크는 네 가지 주요 모듈로 구성됩니다: (i) 온라인 업데이트 가능한 메모리 뱅크, (ii) 객체 제안 모델, (iii) 컨텍스트 검색 모듈, (iv) 인스턴스 검색 모듈. 이 구조를 통해 객체 설정을 수행하고 새로운 개념을 분류합니다.

- **Performance Highlights**: 실험 결과, 기존의 클로즈세트(close-set) 및 오픈세트(open-set) 모델에 비해 제안한 온라인 학습 방법이 큰 성능 향상을 보여주었으며, 소량의 레이블링 노력으로 지난해 DOTA 데이터셋에서 mAP 20 이상을 달성했습니다.



### CoMamba: Real-time Cooperative Perception Unlocked with State Space Models (https://arxiv.org/abs/2409.10699)
- **What's New**: CoMamba라는 새로운 협력적 3D 탐지 프레임워크를 소개하며, 이는 실시간 차량 인식을 위한 상태 공간 모델(state-space models)을 활용한다.

- **Technical Details**: CoMamba는 양방향 상태 공간 모델을 사용하여 주의 메커니즘의 이차 복잡성을 우회한 보다 확장 가능한 3D 모델이다. 이 모델은 Cooperative 2D-Selective-Scan Module과 Global-wise Pooling Module의 두 가지 주요 모듈로 구성된다.

- **Performance Highlights**: CoMamba는 V2X/V2V 데이터셋 기반으로 한 실험에서 기존 방법보다 우수한 성능을 기록했으며, 37.1ms의 낮은 지연 시간으로 실시간 처리 능력을 유지한다. 또한, 0.64GB의 GPU 메모리로 26.9 FPS의 추론 속도를 달성하여 이전의 최첨단 모델보다 19.4% 더 빠르다.



### Playground v3: Improving Text-to-Image Alignment with Deep-Fusion Large Language Models (https://arxiv.org/abs/2409.10695)
- **What's New**: Playground v3 (PGv3)는 최신 텍스트-이미지 생성 모델로, 기존의 모델들과는 달리 Large Language Models (LLMs)를 기반으로 하여 픽셀 단위의 세밀한 이미지 생성 능력을 보여줍니다. 주목할 점은 PGv3가 독창적인 구조를 도입하여 이미지 캡셔닝 성능을 향상시키기 위한 자체 개발된 captioner를 포함하고 있으며, 새로운 벤치마크 CapsBench 또한 소개한 것입니다.

- **Technical Details**: PGv3는 Latent Diffusion Model (LDM)을 기반으로 하며, 기존의 텍스트 인코더를 사용하지 않고 Llama3-8B LLM을 통합하여 텍스트 조건을 제공합니다. 모델 아키텍처는 Deep-Fusion 구조를 채택하여 LLM의 지식을 활용하고, 복잡한 이미지-텍스트 정렬 및 추론을 지원합니다. 이미지와 텍스트의 결합 관리에서 joint attention 구조를 사용해 계산 비용을 줄입니다.

- **Performance Highlights**: PGv3는 이미지 캡셔닝, 복잡한 추론 및 정확한 텍스트 렌더링에서 뛰어난 성능을 보이며, 실제 인간 디자이너를 초월하는 그래픽 디자인 능력을 갖추고 있습니다. RGB 색상 제어 및 강력한 다국어 이해도 제공하며, CapsBench 벤치마크에서의 성과는 PGv3의 우수성을 잘 보여줍니다.



### HAVANA: Hierarchical stochastic neighbor embedding for Accelerated Video ANnotAtions (https://arxiv.org/abs/2409.10641)
- **What's New**: 이 논문은 비디오 주석(Annotation) 프로세스를 가속화하기 위한 새로운 주석 파이프라인을 제안합니다. 기존의 선형 방법들과 비교했을 때, 전통적인 방식에 비해 클릭 수가 10배 이상 감소하는 효과를 보여줍니다.

- **Technical Details**: 제안된 주석 파이프라인은 Hierarchical Stochastic Neighbor Embedding (HSNE)을 사용하여 비디오 기능의 다중 스케일 표현을 생성합니다. 이 방법은 전처리된 특징을 입력으로 사용하며, HSNE는 고차원 데이터 포인트를 2D로 삽입할 수 있어 비슷한 행동에 해당하는 특징들을 함께 배치할 수 있습니다.

- **Performance Highlights**: 12시간 이상의 비디오 주석을 수행하는 데 필요한 클릭 수를 10배 이상 줄이며, 다양한 데이터셋에서 전반적인 신뢰성과 사용성을 검증했습니다. 이 연구는 비디오 이해 시대의 주석 작업을 확대할 수 있는 유망한 방향을 제공합니다.



### Optimizing Resource Consumption in Diffusion Models through Hallucination Early Detection (https://arxiv.org/abs/2409.10597)
Comments:
          Accepted at ECCV Workshop 2024

- **What's New**: 본 논문에서는 HEaD (Hallucination Early Detection)라는 새로운 패러다임을 소개하여 생성 과정 초기에 오류 생성을 신속하게 검출할 수 있는 방법을 제시합니다.

- **Technical Details**: HEaD는 cross-attention 맵과 Predicted Final Image (PFI)를 결합하여 초기 생성 단계에서의 정보를 활용하여 최종 이미지를 예측합니다. 이를 통해 오류를 조기에 식별하고 자원을 절약합니다.

- **Performance Highlights**: HEaD를 사용하면 두 개체 조합의 경우 평균 12%의 생성 시간을 절약하는 것으로 나타났습니다. 이를 통해 생성 모델의 효율성과 정확성을 향상시키는 데 기여합니다.



### SoccerNet 2024 Challenges Results (https://arxiv.org/abs/2409.10587)
Comments:
          7 pages, 1 figure

- **What's New**: 2024년 SoccerNet 챌린지가 네 번째 연례 비디오 이해 챌린지로, 축구 관련 방송 비디오, 필드 이해 및 선수 이해의 여러 테마에서 연구를 촉진하는 것을 목표로 합니다. 올해는 볼 액션 스포팅, 밀집 비디오 캡션, 다중 시점 파울 인식 및 게임 상태 재구성을 포함한 네 가지 비전 기반 작업이 포함되었습니다.

- **Technical Details**: 볼 액션 스포팅, 밀집 비디오 캡션, 다중 시점 파울 인식과 게임 상태 재구성 작업이 소개되었습니다. 각 작업은 데이터 세트를 통해 더욱 향상되었으며, 특히 볼 액션 스포팅은 12개의 축구 볼 관련 행동 클래스를 포함합니다. 또한 mAP(Mean Average Precision) 지표를 사용하여 정확도 평가를 수행합니다.

- **Performance Highlights**: 8888팀이 볼 액션 스포팅 챌린지에 참가하여 61616161 제출이 이뤄졌고, 기준값에서 73.39 mAP@1의 상승이 있었습니다. 우승자는 Artur Xarles, Sergio Escalera, Thomas B. Moeslund, Albert Clapés 팀으로, 그들의 접근법인 T-DEED를 사용하여 성과를 거두었습니다.



### Are Existing Road Design Guidelines Suitable for Autonomous Vehicles? (https://arxiv.org/abs/2409.10562)
Comments:
          Currently under review by IEEE Transactions on Software Engineering (TSE)

- **What's New**: 이번 연구에서는 자율주행차량(AV)의 인식 시스템에 대한 새로운 공격 기법인 TrashFuzz를 제안합니다. 이는 도로 디자인 지침을 따르는 일상적인 가로변 대상의 위치를 조작하여 자율주행차량이 잘못 판단하도록 유도하는 고유한 방법입니다.

- **Technical Details**: TrashFuzz는 가로변에 배치된 쓰레기통, 벤치, 나무, 소화전과 같은 객체의 위치를 조정하여 자율주행 시스템(ADS)이 교통 법규를 위반하게 만드는 시나리오를 생성합니다. 이 기술은 객체 배치를 최적화하기 위해 탐욕적 검색 알고리즘(greedy search algorithm)을 활용하며, 하이피델리티 시뮬레이터를 통해 생성된 시나리오가 안전 규칙과 교통 법규를 위반하는지 평가합니다.

- **Performance Highlights**: TrashFuzz를 통해 Baidu Apollo(버전 7.0)가 총 24개의 교통 법규 중 15개를 위반하는 것을 확인했습니다. 이는 자율주행 시스템이 일상적인 객체 배치에서 예상치 못한 인식 오류를 겪을 수 있음을 시사합니다.



### Convolutional Networks as Extremely Small Foundation Models: Visual Prompting and Theoretical Perspectiv (https://arxiv.org/abs/2409.10555)
- **What's New**: 이 논문은 ImageNet 클래시피케이션과 같은 일반 데이터셋에서 학습된 기본적인 딥 뉴럴 네트워크를 새로운 작업으로 적응시키기 위한 프롬프팅 모듈을 설계합니다. 특히, Semi-parametric Deep Forest (SDForest)라는 구체적인 모듈을 도입하여 비디오 객체 분할 작업에서 높은 성능을 보이고 있습니다.

- **Technical Details**: 제안된 SDForest는 비모수적 방법들(예: correlation filter, random forest, image-guided filter)과 ImageNet 분류 작업을 위해 학습된 딥 네트워크를 결합합니다. 이 연구는 비디오 객체 분할(VOS) 문제에서 제안된 간단한 프롬프팅 모듈이 높은 일반화 성능을 갖도록 설계되었습니다.

- **Performance Highlights**: SDForest는 CPU에서도 실시간 처리 속도를 달성하며, DAVIS2016 및 DAVIS2017의 비디오 객체 분할 작업에서 순수한 딥 러닝 접근 방식과 경쟁할 만한 성능을 보였습니다.



### An Examination of Offline-Trained Encoders in Vision-Based Deep Reinforcement Learning for Autonomous Driving (https://arxiv.org/abs/2409.10554)
- **What's New**: 이 연구는 복잡한 부분 관찰 마르코프 의사결정 프로세스(POMDP)에서의 심층 강화 학습(Deep Reinforcement Learning, DRL)의 도전을 조사하고, 이러한 환경에서의 비전 기반 내비게이션 솔루션을 제안합니다.

- **Technical Details**: 이 연구는 BDD100K 드라이빙 비디오를 사용하여 사전 훈련된 인코더를 통해 일반화 가능한 표현을 학습하고, 이를 통해 CARLA 자율주행 시뮬레이터에서 차량 제어를 수행하는 드릴 네트워크를 훈련합니다. 다양한 자기지도 학습(self-supervised learning) 방법을 비교하고 대한 실험을 통해 DRL 에이전트의 성능을 평가합니다. 또한, 이 연구는 인코더의 아키텍처 설계 및 하이퍼파라미터 조정에 대한 세부 사항을 다룹니다.

- **Performance Highlights**: 사전 훈련된 인코더로부터 학습한 표현은 DRL 에이전트의 성능을 개선하며, CARLA 시뮬레이터에서의 차선 추적 및 충돌 회피 작업에서 제로샷 학습(zero-shot learning) 방식으로 직접 전이 가능한 것으로 나타났습니다.



### ResEmoteNet: Bridging Accuracy and Loss Reduction in Facial Emotion Recognition (https://arxiv.org/abs/2409.10545)
Comments:
          5 pages, 3 figures, 3 tables

- **What's New**: 최근 딥러닝 기술의 발전으로 인해, 얼굴 표정 인식(Facial Emotion Recognition, FER) 기술이 혁신적인 발전을 이루고 있습니다. 이 연구에서는 Convolutional Networks, Squeeze-Excitation (SE) 및 Residual Networks의 조합으로 구성된 새로운 딥러닝 아키텍처인 ResEmoteNet을 제안합니다.

- **Technical Details**: ResEmoteNet은 Convolutional Neural Network (CNN) 블록, Squeeze and Excitation (SE) 블록, 그리고 여러 Residual 블록으로 구성되어 있습니다. SE 블록은 중요 특성에 선택적으로 집중하여 특징 표현을 향상시키고 덜 관련된 것들은 억제합니다. 이 네트워크는 FER2013, RAF-DB 및 AffectNet 데이터셋에서 평가되었고, 각 데이터셋에서 79.79%, 94.76%, 72.39%의 정확도를 기록했습니다.

- **Performance Highlights**: 이 논문에서 제안한 ResEmoteNet은 세 개의 공개 데이터베이스에 대한 평가에서 기존의 최첨단 모델들을 초월하는 성능을 보여주었습니다. 특히, RAF-DB 데이터베이스에서는 94.76%의 높은 정확도를 성취하였습니다.



### OxML Challenge 2023: Carcinoma classification using data augmentation (https://arxiv.org/abs/2409.10544)
Comments:
          Paper has been accepted at IMVIP 2024

- **What's New**: 이번 연구에서는 제한적이고 불균형한 데이터셋을 경량화된 Convolutional Neural Network (CNN) 모델들과 함께 패딩 데이터 증강을 활용하여 carcinoma 분류 문제를 해결했습니다. 이를 통해 OXML 2023 챌린지에서 상위 3위에 들어 승리하였습니다.

- **Technical Details**: 본 연구에서는 다양한 크기의 이미지로 인한 특성 손실을 피하기 위해 패딩(data padding) 기법을 사용해 이미지를 공통 크기로 조정했습니다. 또한, 쇄신 데이터 증강(jitter data augmentation)을 사용하여 작은 샘플 수를 보완하고 클래스 간의 불균형 문제를 해결했습니다. 시행한 Ensembles (앙상블) 기법은 ResNet34, ResNet50, VGG16, EfficientNet, MobileNetV의 다섯 가지 모델을 결합하여 예측의 정확성을 높였습니다.

- **Performance Highlights**: OXML 2023 챌린지에서 39개 팀이 참가하였으며, 본 연구팀은 공개 및 비공식 점수에서 상위 3위로 발표되었습니다. 제안한 방법은 carcinoma 분류 성능을 향상시키며, 향후 암 진단과 의료 이미징 분석의 발전에 기여할 것으로 기대됩니다.



### Learning Co-Speech Gesture Representations in Dialogue through Contrastive Learning: An Intrinsic Evaluation (https://arxiv.org/abs/2409.10535)
- **What's New**: 이 논문은 코-스피치 제스처의 표현 학습에 대한 새로운 접근 방식을 제시합니다. 기존의 접근법에서는 제스처의 변동성과 말과의 관계를 효과적으로 고려하지 못했으나, 본 연구에서는 자기 감독 기반의 대조 학습(self-supervised contrastive learning)을 활용하여 이 문제를 해결합니다.

- **Technical Details**: 제안된 방법론은 단일 모달(unimodal)과 다중 모달(multimodal) 프리 트레이닝(pre-training) 방법을 포함하여 제스처 표현을 동시 발생하는 말에 기반하여 조정합니다. 이 연구는 풍부한 제스처 데이터셋에서 학습한 결과, 사람의 평가와 높은 상관관계를 나타내며, 동적 대화 상호 작용 관련 패턴과 잘 일치하는 제스처 표현의 유의미한 회복 가능성을 보여줍니다.

- **Performance Highlights**: 모델은 200 에폭(epoch) 동안 Pytorch 기반의 구현으로 훈련되었으며, 다중 모달 대조 학습을 통해 좋은 성능을 보였습니다. 연구 결과는 다양한 제스처 쌍의 유사성을 평가한 결과, 고유한 음성 및 말의 특징을 강조할 수 있는 잠재적 가능성을 나타내며, 텍스트와 제스처 간의 관계를 연구하는 데 큰 가능성을 열었습니다.



### Ethical Challenges in Computer Vision: Ensuring Privacy and Mitigating Bias in Publicly Available Datasets (https://arxiv.org/abs/2409.10533)
- **What's New**: 이 논문은 컴퓨터 비전 기술의 개발과 배치에서 윤리적 문제를 다루고 있으며, 특히 공개 데이터셋 사용 시 발생하는 개인정보 보호 및 편향의 문제를 심층적으로 분석합니다.

- **Technical Details**: 본 연구는 다양한 공개 데이터셋(COCO, LFW, ImageNet, CelebA, PASCAL VOC 등)의 윤리적 문제를 탐구하며, 데이터 수집하였을 때 접근해야 할 법적(legal) 및 윤리적 가이드라인을 제시합니다. 이 가이드라인은 개인의 권리 보호와 편향 최소화를 포함합니다.

- **Performance Highlights**: 이 논문은 컴퓨터 비전 모델의 학습에 윤리적 기준을 통합하여 AI 개발이 사회적 가치 및 윤리적 기준을 수용할 수 있도록 하는 것을 목표로 하며, 이를 통해 공공의 해를 방지하는 데 기여하고자 합니다.



### From Latent to Engine Manifolds: Analyzing ImageBind's Multimodal Embedding Spac (https://arxiv.org/abs/2409.10528)
Comments:
          The 26th International Conference on Artificial Intelligence (ICAI'24)

- **What's New**: 이 연구에서는 ImageBind의 능력을 조사하여 온라인 자동차 부품 목록에 대한 의미 있는 융합 다중 모달 임베딩을 생성할 수 있음을 밝혔습니다. 또한 이미지/텍스트 쌍의 중복 정보를 캡처하고 포스트의 의미를 결합하여 공동 임베딩으로 통합하는 단순한 임베딩 융합 워크플로를 제안합니다.

- **Technical Details**: 이 연구에서는 ImageBind를 활용하여 이미지와 텍스트 임베딩을 론칭하여 중복 정보를 포착하는 데 초점을 맞추었습니다. ImageBind는 이미지, 텍스트, 오디오를 포함한 여섯 가지 데이터 유형을 단일 임베딩 공간으로 결합하여 의미를 표현하는 방법을 제공합니다. 최종적으로, 혼합된 다중 모달 임베딩은 임베딩 벡터의 평균을 통해 생성되며, 주 성능은 PCA(주성분 분석)를 통해 차원 축소 후 클러스터링을 통해 분석됩니다.

- **Performance Highlights**: ImageBind의 초기 결과는 순수 오디오 임베딩이 의미적으로 유사한 시장 목록과 상관관계를 가질 수 있음을 나타내며, 이는 향후 연구의 잠재적인 경로를 제시합니다. 클러스터 중심에 가장 가까운 게시물을 조사하여 공동 임베딩의 의미적 품질을 전달하는 경험적 증거를 제공합니다.



### Harnessing Artificial Intelligence for Wildlife Conservation (https://arxiv.org/abs/2409.10523)
Comments:
          13 pages, 13 figures

- **What's New**: 이 논문은 인공지능(AI)을 활용한 야생동물 보존을 위한 혁신적인 전략을 탐구하며, Conservation AI 플랫폼에 대한 초점을 맞추고 있습니다. 이 플랫폼은 생물다양성(Biodiversity) 감시를 위한 최신 기술을 통합하여 야생동물의 모니터링을 향상시키고 있습니다.

- **Technical Details**: Conservation AI는 머신러닝(Machine Learning)과 컴퓨터 비전(Computer Vision)을 활용하여 시각 스펙트럼과 열화상 카메라를 통해 동물, 인간, 밀렵 관련 객체를 감지하고 분류합니다. 이 데이터는 합성곱 신경망(Convolutional Neural Networks, CNNs) 및 트랜스포머(Transformer) 구조를 이용하여 처리됩니다. 실시간 감지는 밀렵과 같은 시간에 민감한 상황을 지원하고, 비실시간 분석은 장기적인 야생동물 모니터링 및 서식지 건강 평가를 가능하게 합니다.

- **Performance Highlights**: 유럽, 북미, 아프리카, 동남아시아의 사례 연구를 통해 species identification, biodiversity monitoring, poaching prevention에서 플랫폼의 성공을 보여줍니다. 데이터 품질, 모델 정확성 및 물류적 제약과 같은 도전 과제에 대해 논의하며, 기술 발전, 새로운 지역으로의 확장 및 지역 사회 및 정책 입안자와의 깊은 협력을 포함한 미래 방향을 제시합니다.



### NVLM: Open Frontier-Class Multimodal LLMs (https://arxiv.org/abs/2409.11402)
- **What's New**: NVLM 1.0은 비전-언어(vision-language) 작업에서 최첨단 결과를 달성하는 다중 모드 대형 언어 모델(multimodal large language models, LLMs)입니다. 특히, NVLM 1.0은 다중 모드 훈련 후 텍스트 전용 성능이 LLM 백본보다 개선되었습니다.

- **Technical Details**: NVLM 1.0은 세 가지 아키텍처(NVLM-D, NVLM-X, NVLM-H)로 구성되어 있습니다. NVLM-D는 디코더 전용 아키텍처로 OCR 관련 작업에서 높은 정확도를 제공하며, NVLM-X는 교차 주의(cross-attention) 기반 아키텍처로 고해상도 이미지를 처리하는 데 효율적입니다. NVLM-H는 두 가지 접근 방식을 통합하여 성능과 효율성을 동시에 향상시킵니다.

- **Performance Highlights**: 다양한 비전-언어 작업 및 텍스트 전용 작업에서 NVLM 모델들은 최고의 성능을 달성했습니다. 특히, 멀티모달 수학 및 코딩 데이터 세트를 통합하여 모든 NVLM 모델에서 텍스트 전용 성능의 개선이 달성되었습니다.



### Compact Implicit Neural Representations for Plane Wave Images (https://arxiv.org/abs/2409.11370)
Comments:
          Accepted by the IEEE International Ultrasonics Symposium (IUS) 2024

- **What's New**: 본 연구에서는 Ultrafast Plane-Wave (PW) 이미징의 각도 보간을 위한 Implicit Neural Representations (INRs)의 첫 번째 적용을 제안합니다. 이 접근 방식은 중요한 방향 의존 정보를 보존하면서 다중 평면 시퀀스를 압축하여 인코딩합니다.

- **Technical Details**: Multi-Layer Perceptron (MLP) 기반의 모델을 사용하여 PW 이미지를 여러 각도에서 표현합니다. 모델은 위치 및 각도 정보를 픽셀 강도로 변환하는 연속 함수를 학습하게 됩니다. Positional Encoding (PE)을 사용하여 입력 데이터를 고차원 임베딩으로 변환하고, 출력은 2D 블러링 커널과 합성하여 최종 강도를 예측합니다.

- **Performance Highlights**: 모델의 저장 효율성은 뛰어나며, 모델 가중치는 530 KB에 불과하고, PW 이미지를 직접 저장하기 위한 8 MB에 비해 약 15:1의 압축 비율을 달성합니다. SSIM, PSNR 및 표준 초음파 메트릭을 사용하여 방법의 효과성이 검증되었습니다.



### TTT-Unet: Enhancing U-Net with Test-Time Training Layers for biomedical image segmentation (https://arxiv.org/abs/2409.11299)
- **What's New**: TTT-Unet는 기존 U-Net 구조에 Test-Time Training(TTT) 레이어를 통합하여 생체의학적 이미지 세분화에서의 장기 의존성을 모델링하는 한계를 극복합니다. 이 새로운 프레임워크는 테스트 시 모델 파라미터를 동적으로 조정하여 지역 및 장거리 피처를 보다 효과적으로 포착할 수 있도록 합니다.

- **Technical Details**: TTT-Unet는 기존 U-Net 구조에 Mamba 블록 내에서 TTT 레이어를 통합하여 설계되었습니다. 이 구조는 테스트 데이터에 기반하여 지속적으로 모델의 파라미터를 업데이트할 수 있도록 하여 특징 추출 능력을 향상시키고 장거리 의존성을 적응적으로 학습할 수 있게 합니다.

- **Performance Highlights**: TTT-Unet는 CT 및 MRI 스캔에서의 3D 복부 장기 세분화, 내시경 이미지에서의 기구 세분화, 현미경 이미지에서의 세포 세분화와 같은 다양한 의료 이미징 데이터세트에서 평가되었으며, 기존의 CNN 및 Transformer 기반 세그멘테이션 모델보다 모든 작업에서 일관되게 우수한 성능을 보였습니다.



### LASERS: LAtent Space Encoding for Representations with Sparsity for Generative Modeling (https://arxiv.org/abs/2409.11184)
Comments:
          Preprint, under review. Submitted to 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)

- **What's New**: 이 논문에서는 VQ (Vector Quantization)의 구조적 가정을 완화하여 잠재 공간(latent space)의 대안 표현을 제안하고 있습니다. 특히, 잠재 공간이 희소성 제약(sparsity constraint)에 따른 사전 기반(dictionary-based) 표현의 부분 공간(subspaces)의 합으로 근사될 수 있다고 가정합니다.

- **Technical Details**: 제안된 방법은 추출된 잠재 공간을 구성하는 코드북의 희소한 학습(dictionary learning)을 통해 표현합니다. DL-VAE (Dictionary Learning Variational Autocoders) 및 DL-GAN (DL-VAEs with Generative Adversarial Networks) 모델로 실험을 진행하여 구성 품질을 개선하는 데 중점을 두고, VQ 접근 방식과 비교하여 잠재 공간의 표현 능력이 향상되었음을 확인하였습니다.

- **Performance Highlights**: 우리의 실험 결과, 작성한 모델은 MNIST, CIFAR10, Oxford Flowers 및 FFHQ 등의 일반적인 이미지 데이터셋에서 재구성 품질(reconstruction quality)을 개선하였습니다. 또한, VQ 계열 모델에서 일반적으로 발생하는 코드북 붕괴(codebook collapse) 문제를 해결하는 데 효과적이라는 것을 보여주었습니다.



### Annealed Winner-Takes-All for Motion Forecasting (https://arxiv.org/abs/2409.11172)
Comments:
          7 pages, 8 figures

- **What's New**: 이번 연구는 자율주행에서 모션 예측(Motion Prediction)의 성능을 향상시키기 위해 새로운 방식인 annealed Winner-Takes-All (aWTA) 손실 함수를 제안합니다. 이 방법은 현재의 WTA 방식에서 발생하는 초기화 민감성과 훈련 불안정을 완화하는 데 중점을 둡니다.

- **Technical Details**: aWTA 손실은 기존의 여러 제안 방식을 통해 도출된 것으로, 수많은 제안 수를 사용하는 대신 소수의 가설만으로도 효과적으로 작동합니다. 이는 예측 모델이 훈련 중 오직 최상의 예측 방향으로만 업데이트되던 기존의 MCL 구조에서 벗어나, 다수의 가능성 있는 미래 궤적을 생성하는 데 기여합니다.

- **Performance Highlights**: aWTA 손실을 적용함으로써, 다양한 최신 모션 예측 모델에서 성능이 크게 향상되었으며, 특히 두 개의 대규모 실제 데이터 세트에서 성능 개선이 입증되었습니다. 기존의 선택 과정 없이 적은 수의 가설로도 우수한 예측 결과를 얻을 수 있습니다.



### MAISI: Medical AI for Synthetic Imaging (https://arxiv.org/abs/2409.11169)
- **What's New**: 의료 이미징 분석에 대한 새로운 접근법인 Medical AI for Synthetic Imaging (MAISI)를 소개합니다. MAISI는 합성 3D CT 이미지를 생성하기 위해 diffusion model을 활용하여 데이터 부족, 높은 주석 비용, 그리고 개인정보 보호 문제를 해결합니다.

- **Technical Details**: MAISI는 volume compression network와 latent diffusion model을 활용하여 512 x 512 x 768 크기의 고해상도 CT 이미지를 생성합니다. ControlNet이 추가되어, 127개의 해부학적 구조에 대한 organ segmentation을 처리합니다. 이를 통해 다양한 다운스트림 작업에 활용할 수 있는 정확하게 주석이 달린 합성 이미지를 생성합니다.

- **Performance Highlights**: 실험 결과, MAISI는 현실적이고 해부학적으로 정확한 이미지를 생성할 수 있는 능력을 보여주며, 합성 데이터를 사용하여 여러 도전 과제를 완화하는 데 도움이 될 가능성을 제시합니다.



### Gradient-free Post-hoc Explainability Using Distillation Aided Learnable Approach (https://arxiv.org/abs/2409.11123)
Comments:
          12 pages, 10 figures, Accepted in IEEE Journal of Selected Topics in Signal Processing (JSTSP), 2024

- **What's New**: 이번 논문에서는 깊은 신경망 모델의 설명 가능성을 높이기 위해 새로운 프레임워크인 distillation aided explainability (DAX)를 제안합니다. DAX는 모델에 구애받지 않는 gradient-free 접근 방식으로, saliency 기반 설명을 생성합니다.

- **Technical Details**: DAX 프레임워크는 두 개의 네트워크인 mask generation network 및 student distillation network을 사용합니다. mask generation network는 입력의 salient 부분을 찾는 multiplier mask를 생성하고, student distillation network는 블랙 박스 모델의 지역적 행동을 근사하는 목표로 합니다. 두 네트워크는 locally perturbed input samples를 활용하여 공동 최적화를 수행합니다.

- **Performance Highlights**: DAX는 이미지 및 오디오 분류 작업에서 9개 다른 방법과 비교하였을 때 모든 모달리티와 평가 지표에서 기존 접근 방식을 상당히 초월하는 성능을 보였습니다. IoU metric, deletion area under the curve (AUC), 그리고 주관적 평가에서도 개선된 성능을 나타냈습니다.



### Multi-Cohort Framework with Cohort-Aware Attention and Adversarial Mutual-Information Minimization for Whole Slide Image Classification (https://arxiv.org/abs/2409.11119)
Comments:
          11 pages, 5 figures

- **What's New**: 본 논문에서는 다양한 암 유형의 차이를 활용한 다중 집단 Whole Slide Images (WSIs) 분석을 위한 새로운 접근 방식을 제안합니다. 기존의 단일 암 유형에 집중한 접근 방식의 한계를 극복하고, 모델의 일반화와 확장성을 향상시키기 위한 Cohort-Aware Attention 모듈과 상반 조정 메커니즘을 도입했습니다.

- **Technical Details**: 우리는 Cohort-Aware Attention Encoder를 설계하여 암 유형별 고유한 특징과 일반적인 병리학적 패턴을 동시 추출할 수 있게 하였고, Adversarial cohort regularization을 통해 특정 집단의 편향을 최소화하였습니다. 또한, 각 집단 간의 불균형을 해결하기 위한 계층적 샘플 균형 전략을 개발하였습니다.

- **Performance Highlights**: 다양한 암 유형을 포함하는 독특하게 구성된 다중 암 데이터셋에서 실시한 실험 결과, 기존의 최신 기법에 비해 일반화 능력이 크게 개선되었으며, WSI 분류를 위한 확장 가능한 솔루션을 제공함을 보여주었습니다.



### Few-Shot Domain Adaptation for Learned Image Compression (https://arxiv.org/abs/2409.11111)
- **What's New**: 이 논문에서는 사전 훈련된 Learned Image Compression (LIC) 모델의 도메인 적응을 위한 새로운 방법인 few-shot domain adaptation을 제안합니다. 이 방법은 plug-and-play adapters를 통해 모델의 일반화 능력을 향상시킵니다.

- **Technical Details**: 이 방법은 convolution-based adapters (Conv-Adapters)와 low-rank adapters (LoRA-Adapters)를 사용하여 채널 단위의 재배치를 수행합니다. 이 어댑터들은 필요한 수의 타겟 샘플로 훈련되며, 다양한 도메인과 주류 LIC 스킴에 적용 가능합니다.

- **Performance Highlights**: 우리의 실험 결과, 제안한 방법은 VTM에 비해 모든 도메인에서 사전 훈련된 모델의 성능을 크게 향상시키며, 단 25개의 타겟 도메인 샘플로 H.266/VVC의 성능에 필적하는 결과를 보였습니다. 또한, 전체 모델 fine-tuning과 유사한 성능을 제공하면서 매개변수의 2% 미만만 전송합니다.



### Enhanced segmentation of femoral bone metastasis in CT scans of patients using synthetic data generation with 3D diffusion models (https://arxiv.org/abs/2409.11011)
Comments:
          14 pages, 5 figures 3 tables

- **What's New**: 본 연구는 CT 스캔에서 대퇴골 전이의 세분화를 개선하기 위해 3D Denoising Diffusion Probabilistic Models (DDPM)을 활용한 자동 데이터 합성 파이프라인을 제안합니다.

- **Technical Details**: 29개의 기존 병변 및 26개의 건강한 대퇴골 이미지를 사용하여 새로운 합성 전이 이미지를 생성하고, 이를 통해 생성된 합성 데이터와 실제 데이터를 사용하여 3D U-Net 모델을 훈련시킵니다. 각 모델의 성능은 훈련에 사용된 합성 데이터의 양에 따라 평가됩니다.

- **Performance Highlights**: 합성 데이터로 훈련된 세분화 모델이 실제 데이터에 대해서만 훈련된 모델을 능가하였으며, 특히 운영자 변동성을 고려할 때 더욱 우수한 성능을 보였습니다.



### CAST: Cross-modal Alignment Similarity Test for Vision Language Models (https://arxiv.org/abs/2409.11007)
- **What's New**: 이 논문에서는 Cross-modal Alignment Similarity Test (CAST)를 제안하여 Vision Language Models (VLMs)의 확장성과 일관성을 평가하고, 전통적인 Visual Question Answering (VQA)와는 다른 새로운 접근 방식을 탐색합니다.

- **Technical Details**: CAST는 두 장면 간의 유사성을 찾도록 모델에 요구함으로써 VLM의 멀티모달(self-consistency) 기능을 평가하는 자동화된 2단계 방법입니다. 이 방법은 장면의 이미지와 고품질 설명 간의 유사성을 평가하며, 텍스트 전용, 이미지 전용 또는 두 가지 모두를 활용하여 모달리티(modality) 간의 일관성을 분석합니다.

- **Performance Highlights**: 테스트 결과, 다수의 VLM들이 다양한 다운스트림 작업에서 뛰어난 성능을 보였음에도 불구하고 내부의 자기 일관성(self-consistency)과 모달리티 정렬(modiaty alignment)에서 결핍 현상을 보였습니다. 이는 CAST가 VLM의 추론 능력과 잠재적인 편향(bias)을 이해하는 데 중요한 역할을 한다는 것을 보여줍니다.



### PSFHS Challenge Report: Pubic Symphysis and Fetal Head Segmentation from Intrapartum Ultrasound Images (https://arxiv.org/abs/2409.10980)
- **What's New**: 이 연구는 국제 산부인과 초음파 학회(ISUOG)에서 권장하는 출산 중 초음파 이미징을 통해 태아 및 모체 구조의 자동 분할(segmentation) 알고리즘의 발전을 목표로 하고 있습니다. 특히, 출산 경과 모니터링에 필수적인 작업으로, 이 연구는 5,101개의 출산 중 초음파 이미지를 포함하는 대규모 데이터셋을 제공함으로써 자동 분할 기술의 최적화를 위한 기회를 제공합니다.

- **Technical Details**: 이 연구에서는 26회 의료 영상 컴퓨팅 및 컴퓨터 지원 중재 국제 회의(MICCAI 2023)와 함께 열린 Grand Challenge on Pubic Symphysis-Fetal Head Segmentation (PSFHS)의 결과를 다룹니다. 데이터셋은 두 개의 초음파 기계에서 세 개의 병원에서 수집되었고, 179개의 참가자 중 상위 8개의 참가작이 선택되었습니다. 이 알고리즘들은 출산 중 초음파 이미지에서 자동 PSFHS의 최신 기술 수준을 향상시켰습니다.

- **Performance Highlights**: 무려 193명의 참가자의 초기 등록 후, 상위 8개 참가작이 선발되어 경쟁의 두 번째 단계로 나아갔습니다. 이 연구의 결과들은 현장에서 여전히 해결되지 않은 문제들을 발견하고, 향후 연구를 위한 권장 사항을 정리했습니다. 최상의 솔루션과 전체 데이터셋은 공개되어 있으며, 출산 중 초음파 이미징을 위한 자동 분할 및 생체 측정(biometry)의 발전을 촉진하고 있습니다.



### Edge-based Denoising Image Compression (https://arxiv.org/abs/2409.10978)
- **What's New**: 최근 딥러닝 기반 이미지 압축 연구에서 생성 모델을 활용한 이미지 압축이 중요한 연구 분야로 부각되고 있습니다. 이 논문에서는 이미지의 복원 품질을 높이기 위해 새로운 압축 모델을 제안합니다. 특히, 디퓨전 모델과 노이즈 제거 단계를 결합하여 기존 모델의 한계를 극복합니다.

- **Technical Details**: 제안된 모델은 Variational Autoencoder (VAE)에서 지식을 얻은 Edge Estimation Network를 사용하여 이미지의 경계 정보를 추출합니다. 이러한 경계 정보는 디퓨전 모델의 노이즈 제거 과정에 활용되어 이미지의 흐릿함을 개선합니다. 전체 구조는 단일 입력 이미지 (I)에서 잠재 표현 (z)로 인코딩한 후, 디퓨전 모델을 통해 노이즈를 제거하고 최종적으로 복원된 이미지 (I′)를 제공합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 기존 모델과 비교하여 이미지 품질 및 압축 효율성에서 우수하거나 동등한 성능을 보였습니다. 특히, 부분 이미지 손실 또는 과도한 노이즈 환경에서도 뛰어난 성능을 발휘하여 실질적인 이미지 압축의 한계를 극복하는 견고한 솔루션을 제공합니다.



### CUNSB-RFIE: Context-aware Unpaired Neural Schr"{o}dinger Bridge in Retinal Fundus Image Enhancemen (https://arxiv.org/abs/2409.10966)
- **What's New**: 이 논문에서는 망막 이미지 개선을 위한 새로운 접근 방식인 Context-aware Unpaired Neural Schrödinger Bridge (CUNSB-RFIE)를 제안합니다. 이 방법은 Optimal Transport (OT) 이론을 활용하며, 이전의 Generative Adversarial Networks (GANs) 방법들이 갖고 있는 훈련 안정성과 출력 다양성 사이의 트레이드오프 한계를 극복합니다.

- **Technical Details**: CUNSB-RFIE 프레임워크는 Dynamic Snake Convolution(DSC)을 도입하여 구조적 세부사항을 보존합니다. DSC는 곡선 형태의 수용 필드를 사용하여 혈관과 같은 미세 구조를 더욱 잘 보존합니다. 훈련 중에는 PatchNCE 및 SSIM 정규화를 적용하여 저품질 이미지와 개선된 이미지 간의 일관성을 유지합니다.

- **Performance Highlights**: 대규모 데이터셋을 사용한 실험 결과, 제안한 방법이 기존의 여러 최신 감독 및 비감독 방법들에 비해 이미지 품질과 다운스트림 작업에서 성능이 우수함을 보여주었습니다.



### Towards Effective User Attribution for Latent Diffusion Models via Watermark-Informed Blending (https://arxiv.org/abs/2409.10958)
Comments:
          9 pages, 7 figures

- **What's New**: 이번 연구는 텍스트 기반 설명에서 초현실적인 이미지를 생성하는 멀티모달 대형 언어 모델에 대한 최신 발전을 다루며, 사용자 전용 워터마크를 자동으로 블렌딩하여 이미지 품질을 저하시키지 않고 효과적인 사용자 추적을 가능하게 하는 새로운 프레임워크인 TEAWIB를 소개합니다.

- **Technical Details**: TEAWIB(Towards Effective user Attribution for latent diffusion models via Watermark-Informed Blending) 프레임워크는 두 가지 주요 모듈인 Dynamic Watermark Blending (DWB)와 Image Quality Preservation (IQP)를 포함합니다. DWB는 워터마크 전용 가중치를 포함하여 모델이 유연한 워터마킹을 지원하도록 하며, IQP는 노이즈 및 증강 작업을 픽셀 수준에서 통합해 사용자의 특정 정보를 섬세하게 삽입하면서 이미지 품질을 유지합니다.

- **Performance Highlights**: TEAWIB는 MS-COCO 검증 데이터셋에 대한 포괄적인 실험을 통해 기존 통합 워터마킹 방법들보다 강화된 시각적 품질과 저작권 추적 정확성을 입증하였으며, 실제 세계에서의 적용 가능성을 높였습니다.



### Lite-FBCN: Lightweight Fast Bilinear Convolutional Network for Brain Disease Classification from MRI Imag (https://arxiv.org/abs/2409.10952)
- **What's New**: 본 연구에서는 MRI (Magnetic Resonance Imaging) 스캐너를 이용한 뇌 질환 분류에서 높은 정확도와 계산 효율성을 동시에 달성하기 위한 새로운 솔루션인 Lite-FBCN을 제안합니다.

- **Technical Details**: Lite-FBCN은 전통적인 이중 네트워크 모델과는 달리 단일 네트워크 아키텍처(single-network architecture)를 활용합니다. 이 구조는 계산 부하를 크게 줄여 주며, 경량(pre-trained) CNN을 사용하여 관련 피처(feature)를 추출하고, bilinear pooling 전에 채널 리듀서(channel reducer) 레이어를 포함시켜 기능 맵의 차원(dimensions)을 최소화합니다.

- **Performance Highlights**: Lite-FBCN은 MobileNetV1을 활용하여 교차 검증(cross-validation)에서 98.10%의 정확도를 달성하였고, 홀드아웃(hold-out) 데이터에서는 69.37%의 정확도를 기록하였습니다. 이는 기존 baseline CNN 대비 3% 향상된 결과입니다. UMAP 시각화를 통해 유사한 뇌 질환 클래스 간의 구분 능력이 더욱 확고히 입증되었습니다.



### RoadRunner M&M -- Learning Multi-range Multi-resolution Traversability Maps for Autonomous Off-road Navigation (https://arxiv.org/abs/2409.10940)
Comments:
          Under review for IEEE RA-L

- **What's New**: 이번 연구는 로드러너(RoadRunner)를 기반으로, 거리에 따라 100m까지의 terrain traversability(주행 가능성) 예측 문제를 해결하기 위한 학습 기반 프레임워크인 로드러너 M&M(RoadRunner M&M)을 제안합니다. 이 프레임워크는 여러 이미지와 LiDAR voxel map을 입력으로 받아 고속 주행 중에도 정확한 경량의 예측을 수행할 수 있습니다.

- **Technical Details**: 로드러너 M&M은 여러 범위(50m, 100m)와 해상도(0.2m, 0.8m)에서 terrain traversability 및 elevation maps(고도 맵)을 직접 예측하는 end-to-end(끝에서 끝까지) 학습 모델입니다. 모델은 self-supervised 방식으로 기존의 traversability 추정 스택(X-Racer)과 위성 Digital Elevation Maps(디지털 고도 지도)로부터 생성된 밀집 지도 신호를 활용하여 훈련되었습니다.

- **Performance Highlights**: 로드러너 M&M은 고도 맵에서 최대 50% 그리고 주행 가능성 예측에서 30%의 성능 향상을 달성하였으며, X-Racer보다 30% 더 넓은 지역을 예측할 수 있습니다. 다양한 분포 외 데이터셋에서 테스트한 결과, 데이터 기반 접근 방식이 새로운 비구조적 환경에서도 일반화되기 시작함을 나타냈습니다.



### Anti-ESIA: Analyzing and Mitigating Impacts of Electromagnetic Signal Injection Attacks (https://arxiv.org/abs/2409.10922)
Comments:
          2 pages, 2 figures

- **What's New**: 이 논문은 Electromagnetic Signal Injection Attacks (ESIA)의 영향을 조사하여 지능형 시스템의 성능 저하에 기여하는 두 가지 주요 요인을 식별하고, ESIA의 영향을 완화하기 위한 효과적이고 실용적인 방법을 소개합니다.

- **Technical Details**: 본 연구는 이미지 분류 작업에서 ESIA의 두 가지 주요 요인인 픽셀 손실(pixel loss)과 색상 스트립(color strips)을 분석합니다. ESIA는 하드웨어 결함을 이용하여 카메라 회로에 전자기 신호를 원격으로 주입함으로써 발생하며, 이로 인해 캡처된 이미지의 신뢰성이 손상됩니다. 위 연구에서는 median interpolation 기법을 사용하여 픽셀 손실과 색상 왜곡을 완화하는 방법을 탐구하였습니다.

- **Performance Highlights**: 실험 결과, 색상 스트립이 개별 픽셀 손실보다 모델 성능에 더 큰 영향을 미친다는 것을 보여주었습니다. 또한, median interpolation 방법은 중간 수준의 픽셀 손실 및 색상 왜곡에 대해 성능 회복을 부분적으로 가능하게 하였으나, 손실된 픽셀이 많아질수록 그 효과가 감소하는 한계를 가지고 있습니다.



### SkinMamba: A Precision Skin Lesion Segmentation Architecture with Cross-Scale Global State Modeling and Frequency Boundary Guidanc (https://arxiv.org/abs/2409.10890)
Comments:
          Submitted to ACCV2024 workshop

- **What's New**: 본 연구에서는 SkinMamba라는 하이브리드 아키텍처를 제안하였습니다. Mamba와 CNN을 결합하여 피부 병변 segmentation의 효율성을 높이고, 다양한 병변 크기 및 경계 모호성 문제를 해결하였습니다.

- **Technical Details**: SkinMamba는 Scale Residual State Space Block (SRSSB)와 Frequency Boundary Guided Module (FBGM)을 핵심 요소로 사용합니다. SRSSB는 전역 컨텍스트 관계를 파악하며, FBGM은 경계 정보를 보존하고 정확한 경계 segmentation을 지원합니다. 전체 모델은 5레벨 인코더-디코더 구조로 구성되어 있습니다.

- **Performance Highlights**: ISIC2017 및 ISIC2018 두 개의 공개 데이터셋에서 SkinMamba는 mIoU, DSC, Acc, Spe, Sen 측면에서 최신 기술들을 능가하는 성능을 보여주었습니다.



### Neural Fields for Adaptive Photoacoustic Computed Tomography (https://arxiv.org/abs/2409.10876)
- **What's New**: NF-APACT는 Neural Fields를 활용하여 2D 속도 분포(SOS)와 초기 압력 이미지를 동시에 복원하는 새로운 자기 지도 학습(Self-Supervised Learning) 기반 프레임워크입니다.

- **Technical Details**: 본 연구에서는 PA 신호만을 사용하여 SOS를 추정하는 NF-APACT 프레임워크를 제안합니다. 이 방법은 복원 시간을 30배 이상 단축하고, 물리 기반의 포워드 모듈을 통해 SOS에 대한 해석이 가능합니다. 또한, 사용자 정의 정규화(User-defined regularization)를 통해 도메인 지식을 반영할 수 있는 장점을 지니고 있습니다.

- **Performance Highlights**: 시뮬레이션 및 실험적 데이터를 기반으로 NF-APACT는 기존의 APACT 방법에 비해 이미지 품질과 연산 시간을 포함한 여러 성능 지표에서 우수한 결과를 나타냈습니다. 이 방법은 비정상적인 효과(예: 변환기 EIR 및 기하학적 오프셋)에 강인성을 보여주며, 실제 PACT 시스템에 문제 없이 적용되었습니다.



### BAD: Bidirectional Auto-regressive Diffusion for Text-to-Motion Generation (https://arxiv.org/abs/2409.10847)
- **What's New**: 본 논문에서는 Autoregressive 모델과 mask 기반 generative 모델의 장점을 통합한 Bidirectional Autoregressive Diffusion (BAD) 프레임워크를 제안합니다. BAD는 자연스러운 시퀀스 구조를 유지하면서도 인과적 종속성을 강제하는 순열 기반의 손상 기술을 사용합니다.

- **Technical Details**: BAD 프레임워크는 두 단계로 구성됩니다. 첫 번째 단계에서는 Vector-Quantized Variational Autoencoders (VQ-VAEs)를 기반으로 하는 모션 토크나이저를 훈련하여 연속 모션 데이터를 이산 모션 토큰으로 변환하고, 두 번째 단계에서는 이러한 모션 토큰을 사용하여 Transformer 아키텍처를 훈련합니다. 이 과정에서 하이브리드 attention mask를 구축하여 각 토큰 간의 의존성을 결정합니다.

- **Performance Highlights**: BAD는 text-to-motion 생성에서 기존의 autoregressive 및 mask 기반 모델보다 우수한 성능을 보이며, HumanML3D 및 KIT-ML 데이터셋에서 Frechet Inception Distance (FID)를 개선하였습니다. 특히, BAD는 고급 모션 토크나이저를 사용하는 방법들과 비슷한 성능을 보여줍니다.



### Multi-frequency Electrical Impedance Tomography Reconstruction with Multi-Branch Attention Image Prior (https://arxiv.org/abs/2409.10794)
Comments:
          10 pages, 10 figures, journal

- **What's New**: 이 논문은 Multi-frequency Electrical Impedance Tomography (mfEIT) 재구성을 위한 최초의 비지도 학습 방법을 제안합니다. 제안된 방법은 Multi-Branch Attention Image Prior (MAIP)를 기반으로 하여, 다양한 주파수 의존 전도도 이미지를 효과적으로 표현하고 동시에 mfEIT 이미지를 재구성합니다.

- **Technical Details**: 이 연구에서는 Multi-Branch Attention Network (MBA-Net)를 사용하여 mfEIT 이미지를 재구성합니다. MBA-Net은 서로 다른 주파수 측정값에서 다중 브랜치 기능을 캡처하며, 이 후에 Fusion Unit (FU)와 Branch Attention (BA) 모듈을 통해 inter- 및 intra-frequency 상관관계를 향상시킵니다. 이 방법은 데이터 학습 없이도 강력한 재구성을 가능하게 합니다.

- **Performance Highlights**: 시뮬레이션 및 실제 실험을 통해 제안된 접근 방식은 기존의 SOTA(SOTA) 알고리즘과 동등하거나 더 나은 성능을 보여주었습니다. MAIP 기반 방법은 mfEIT의 신뢰성과 적용성을 다양한 상황에서 개선할 수 있는 가능성을 가지고 있습니다.



### MotIF: Motion Instruction Fine-tuning (https://arxiv.org/abs/2409.10683)
- **What's New**: 본 논문에서는 로봇의 동작 이해를 위한 새로운 접근법인 Motion Instruction Fine-tuning (MotIF)을 제안합니다. MotIF는 로봇의 동작을 평가할 때 전체 궤적에 주목하여, vision-language models (VLMs) 의 성공 판별력을 향상시키는데 초점을 맞추고 있습니다.

- **Technical Details**: MotIF는 로봇의 궤적을 최종 이미지 위에 겹쳐 시각화를 통해 궤적 기반의 시각적 동작 표현을 만듭니다. 이를 통해 로봇 동작과 환경의 의미적 연결을 포착하여 VLM의 성능을 향상시킵니다. 또한, MotIF-1K 데이터세트(653개의 인간 예시 및 369개의 로봇 예시 포함)는 13개 작업 범주에서 다양한 동작을 포함하고 있으며, VLM을 세밀하게 조정하는 데 유용한 자료를 제공합니다.

- **Performance Highlights**: MotIF는 최신 VLMs에 비해 2배 이상의 정밀도 향상과 56.1%의 재현율 향상을 보여주었습니다. MotIF는 다양한 작업, 동작 및 환경에 대한 일반화 능력을 갖추고 있으며, 로봇 계획의 개선 및 종료, 동작에 대한 점수 매기기 등의 실제 응용을 시장 가치에 적합하게 보여주고 있습니다.



### Kolmogorov-Arnold Transformer (https://arxiv.org/abs/2409.10594)
Comments:
          Code: this https URL

- **What's New**: Kolmogorov-Arnold Transformer (KAT)는 전통적인 MLP 레이어를 Kolmogorov-Arnold Network (KAN) 레이어로 대체하여 모델의 표현력과 성능을 향상시키는 새로운 아키텍처입니다. KAN을 Transformers에 통합하는 과정에서 발견된 3가지 주요 도전 과제를 해결하기 위해 혁신적인 접근 방식을 제안합니다.

- **Technical Details**: KAT는 KAN의 기본 함수를 B-spline 대신 유리 함수로 변경하고, 그룹 KAN을 통해 활성화 가중치를 공유하여 계산 부담을 줄이며, 분산 보존 초기화를 사용하여 층 간의 활성화 분산을 유지합니다. 이러한 디자인은 KAT가 효과적으로 확장될 수 있도록 하고, 전통적인 MLP 기반 Transformers를 능가하는 성능을 제공합니다.

- **Performance Highlights**: KAT는 이미지 인식, 객체 탐지, 의미 분할을 포함한 다양한 비전 작업에서 평가됐으며, ImageNet-1K에서 82.3%의 정확도를 달성하여 동일한 크기의 ViT 모델보다 3.1% 향상된 성능을 보였습니다. 또한, ViT의 사전 훈련된 가중치로 초기화했을 때 성능이 82.7%로 더욱 개선되었습니다.



### WaveMixSR-V2: Enhancing Super-resolution with Higher Efficiency (https://arxiv.org/abs/2409.10582)
Comments:
          10 pages. arXiv admin note: text overlap with arXiv:2307.00430

- **What's New**: WaveMixSR의 향상된 버전인 WaveMixSR-V2가 새로운 다단계 설계를 통해 4×4 초해상도(Super-Resolution) 작업을 개선하고 있습니다. 또한 기존의 전치 합성곱 층을 Pixel Shuffle 연산으로 대체하여 자원 효율성을 높였습니다.

- **Technical Details**: WaveMixSR-V2는 2차원 이산 웨이브렛 변환(2D-DWT)을 이용한 스페셜 토큰 믹싱으로, 다단계 설계를 통해 해상도를 점진적으로 doubling 하여 더욱 정교한 디테일 복원을 가능케 합니다. Pixel Shuffle은 전치 합성곱의 매개변수 수를 줄이고, 계산 비용을 감소시킵니다.

- **Performance Highlights**: WaveMixSR-V2는 BSD100 데이터셋에서 이전 SOTA보다 50% 이하의 매개변수로 성능을 달성하였으며, 자원 소모가 적고, 지연 시간과 처리량이 향상되었습니다.



### GLEAN: Generative Learning for Eliminating Adversarial Nois (https://arxiv.org/abs/2409.10578)
- **What's New**: 디지털 아트 커뮤니티에서 DALL-E와 Stable Diffusion과 같은 강력한 diffusion 모델로 인해 스타일 모방 공격(style mimicry attacks)이 증가하고 있습니다. 이를 방지하기 위해 Glaze라는 도구가 제안되었고, 본 논문에서는 GLEAN이라는 새로운 접근 방식을 제안합니다.

- **Technical Details**: GLEAN은 I2I (Image-to-Image) generative networks를 사용하여 Glazed 이미지에서 perturbations를 제거하는 방법입니다. 이 방법은 Glaze의 결과물에 대한 스타일 모방 공격의 성능을 평가합니다.

- **Performance Highlights**: GLEAN은 Glaze의 한계를 부각시키고 향후 개발을 촉진할 수 있도록 도와줍니다. Glaze는 스타일 모방 공격을 예방하는 데 중요한 성과를 보였지만, 일부 품질 저하나 인식할 수 없는 노이즈 같은 아티팩트(artifacts)가 발생합니다.



### Eureka: Evaluating and Understanding Large Foundation Models (https://arxiv.org/abs/2409.10566)
- **What's New**: 이 논문에서는 AI의 평가 프로세스를 개선하기 위한 새로운 프레임워크인 Eureka를 소개합니다. 이 프레임워크는 대형 기초 모델(Large Foundation Models)에 대한 더 나은 평가를 가능하게 하며, 단일 점수 보고와 순위 매김을 넘어서 다양한 모델의 능력을 비교하는 데 초점을 둡니다.

- **Technical Details**: Eureka는 모델 평가를 위한 유연한 라이브러리를 제공하여 데이터 전처리, 프롬프트 템플릿, 모델 추론, 데이터 사후 처리, 메트릭 계산 및 보고 작업을 사용자 맞춤형으로 조합 가능하게 합니다. 또한, Eureka-Bench는 사전 정의된 벤치마크 모음을 갖추고 있어 기존의 평가 방법이 간과하는 언어 및 다중 모드 능력을 테스트할 수 있도록 설계되었습니다.

- **Performance Highlights**: Eureka-Bench의 평가 결과, Claude 3.5 Sonnet, GPT-4o 2024-05-13, Llama 3.1 405B와 같은 특정 모델들이 여러 능력에서 반복적으로 다른 모델들보다 우수한 성능을 보였으나, 전반적으로는 어떤 단일 모델이 모든 작업에서 최선의 성능을 내지 않음을 보여줍니다. 특히, 현대 모델들은 이미지에 대한 세부 이해와 같은 기본 능력에서 여전히 한계를 보이고 있습니다.



### SAM4MLLM: Enhance Multi-Modal Large Language Model for Referring Expression Segmentation (https://arxiv.org/abs/2409.10542)
Comments:
          ECCV 2024

- **What's New**: SAM4MLLM은 Segment Anything Model(SAM)과 Multi-Modal Large Language Models(MLLMs)를 통합하여 픽셀 수준의 정보 인식을 향상시키는 혁신적인 접근 방식을 소개합니다. 이 방법은 기존 모델 아키텍처의 과도한 수정이나 전문 토큰의 추가 없이 MLLM이 피펠 수준의 위치 정보를 학습할 수 있도록 합니다.

- **Technical Details**: SAM4MLLM은 사진의 픽셀 위치에 대한 참조 표현(segmentation)을 활용하여 MLLM과 SAM 간의 통합을 구현합니다. 기존 MLLM 아키텍처를 수정하지 않으면서 픽셀 수준의 정보를 학습할 수 있으며, 이를 통해 이미지에서 객체를 정확하게 인식합니다. MLLM이 SAM을 위해 효과적인 프롬프트 포인트를 얻기 위해 적극적으로 질문하는 새로운 방법을 소개합니다.

- **Performance Highlights**: 다양한 RES 벤치마크(RES 데이터셋, GRES, ReasonSeg)에 대한 실험 결과는 SAM4MLLM의 유효성을 입증하며, 복잡한 픽셀 인식 작업을 처리하는 데 있어 우수한 성능을 보임을 보여줍니다.



New uploads on arXiv(cs.AI)

### Navigating Process Mining: A Case study using pm4py (https://arxiv.org/abs/2409.11294)
- **What's New**: 이번 연구에서는 Python의 pm4py 라이브러리를 사용하여 도로 교통 벌금 관리 프로세스를 종합적으로 분석합니다. 이는 신규 이벤트 데이터 분석에 대한 접근 방식을 보여줍니다.

- **Technical Details**: 이 연구는 이벤트 로그 데이터셋을 가져오고, 활동의 분포와 프로세스 변형을 포함한 특성을 탐색합니다. Alpha Miner, Inductive Miner, Heuristic Miner와 같은 다양한 프로세스 마이닝 알고리즘을 적용하여 이벤트 로그 데이터에서 프로세스 모델을 발견합니다.

- **Performance Highlights**: 연구 결과는 도로 교통 벌금 관리 프로세스의 효율성과 효과성에 대한 중요한 통찰을 제공합니다. 또한, 프로세스 최적화 및 의사 결정에 유용한 정보를 제공합니다.



### Neural Networks for Vehicle Routing Problem (https://arxiv.org/abs/2409.11290)
- **What's New**: 본 논문에서는 Vehicle Routing Problem (VRP)을 해결하기 위한 새로운 그래픽 신경망 모델을 제안하고, 해당 모델의 성능 분석을 통해 신경망이 경로 최적화에 효과적으로 적용될 수 있음을 보여줍니다.

- **Technical Details**: 이 연구는 기존의 최적화 방법론인 genetic algorithm, simulated annealing, tabu search, ant colony optimization, firefly algorithm과 차별화된 접근 방식을 가지고 있으며, 신경망(neural networks)을 활용한 경로 최적화에 대한 가능성을 탐구합니다. 특히, 제안된 NN 아키텍처의 세부사항과 적용 가능성을 분석합니다.

- **Performance Highlights**: 실험 결과, 제안된 신경망 모델이 기존의 방법론들에 비해 우수한 성능을 나타내었으며, 복잡한 문제 해결을 위한 신경망의 효율성을 입증하였습니다.



### Machine Learning and Theory Ladenness -- A Phenomenological Accoun (https://arxiv.org/abs/2409.11277)
Comments:
          29 pages with reference

- **What's New**: 최근 머신 러닝(ML) 방법론이 과학 연구에 퍼지면서 이론의 의존성(theory ladenness)에 대한 논의가 다시 일어났습니다. 이 연구는 ML 모델(MLM)과 ML 모델링 전략이 과학 영역의 이론에 어떻게 영향을 받는지를 검사합니다.

- **Technical Details**: 이 논문에서는 ML 연구의 이론 의존성에 대한 분석을 제공합니다. MLM의 구성은 특정 이론에 비교적 독립적일 수 있지만, 특정 도메인 내에서 이러한 모델을 실제로 구현하고 해석하는 것은 여전히 기본 이론적 가정 및 배경 지식에 의존한다고 주장합니다.

- **Performance Highlights**: 전통적인 과학과 ML 기반 과학의 차이에 대한 논의는 단순한 양측론으로 요약할 수 없으며, ML 방법과 도메인 이론 간의 상호작용에 대한 이해를 진전시키지 못합니다.



### Gradient-free Post-hoc Explainability Using Distillation Aided Learnable Approach (https://arxiv.org/abs/2409.11123)
Comments:
          12 pages, 10 figures, Accepted in IEEE Journal of Selected Topics in Signal Processing (JSTSP), 2024

- **What's New**: 이번 논문에서는 깊은 신경망 모델의 설명 가능성을 높이기 위해 새로운 프레임워크인 distillation aided explainability (DAX)를 제안합니다. DAX는 모델에 구애받지 않는 gradient-free 접근 방식으로, saliency 기반 설명을 생성합니다.

- **Technical Details**: DAX 프레임워크는 두 개의 네트워크인 mask generation network 및 student distillation network을 사용합니다. mask generation network는 입력의 salient 부분을 찾는 multiplier mask를 생성하고, student distillation network는 블랙 박스 모델의 지역적 행동을 근사하는 목표로 합니다. 두 네트워크는 locally perturbed input samples를 활용하여 공동 최적화를 수행합니다.

- **Performance Highlights**: DAX는 이미지 및 오디오 분류 작업에서 9개 다른 방법과 비교하였을 때 모든 모달리티와 평가 지표에서 기존 접근 방식을 상당히 초월하는 성능을 보였습니다. IoU metric, deletion area under the curve (AUC), 그리고 주관적 평가에서도 개선된 성능을 나타냈습니다.



### Logic Synthesis Optimization with Predictive Self-Supervision via Causal Transformers (https://arxiv.org/abs/2409.10653)
- **What's New**: 본 연구는 Logic Synthesis Optimization (LSO) 분야에서 Autoregressive transformer 모델과 predictive SSL을 활용하여 Quality of Results (QoR)의 예측 경로를 예측하는 새로운 접근 방식인 LSOformer를 제안합니다.

- **Technical Details**: LSOformer는 cross-attention 모듈을 통합하여 회로 그래프와 최적화 시퀀스의 통찰을 결합함으로써 QoR 메트릭스에 대한 예측 정확성을 향상시킵니다. 또한, 중간 QoR를 예측하는 보조 과제를 정의하여 학습 과정에서 더 강력한 감독 신호를 제공합니다.

- **Performance Highlights**: 실험 결과, LSOformer는 EPFL, OABCD 및 독점 회로 데이터셋에서 각각 5.74%, 4.35% 및 17.06%의 QoR 예측 개선을 보여주는 것으로 확인되었습니다.



### SAM4MLLM: Enhance Multi-Modal Large Language Model for Referring Expression Segmentation (https://arxiv.org/abs/2409.10542)
Comments:
          ECCV 2024

- **What's New**: SAM4MLLM은 Segment Anything Model(SAM)과 Multi-Modal Large Language Models(MLLMs)를 통합하여 픽셀 수준의 정보 인식을 향상시키는 혁신적인 접근 방식을 소개합니다. 이 방법은 기존 모델 아키텍처의 과도한 수정이나 전문 토큰의 추가 없이 MLLM이 피펠 수준의 위치 정보를 학습할 수 있도록 합니다.

- **Technical Details**: SAM4MLLM은 사진의 픽셀 위치에 대한 참조 표현(segmentation)을 활용하여 MLLM과 SAM 간의 통합을 구현합니다. 기존 MLLM 아키텍처를 수정하지 않으면서 픽셀 수준의 정보를 학습할 수 있으며, 이를 통해 이미지에서 객체를 정확하게 인식합니다. MLLM이 SAM을 위해 효과적인 프롬프트 포인트를 얻기 위해 적극적으로 질문하는 새로운 방법을 소개합니다.

- **Performance Highlights**: 다양한 RES 벤치마크(RES 데이터셋, GRES, ReasonSeg)에 대한 실험 결과는 SAM4MLLM의 유효성을 입증하며, 복잡한 픽셀 인식 작업을 처리하는 데 있어 우수한 성능을 보임을 보여줍니다.



### AraDiCE: Benchmarks for Dialectal and Cultural Capabilities in LLMs (https://arxiv.org/abs/2409.11404)
Comments:
          Benchmarking, Culturally Informed, Large Language Models, Arabic NLP, LLMs

- **What's New**: 이번 연구는 아랍어 방언(dialect)에 대한 데이터셋 부족 문제를 해결하기 위해, 현대 표준 아랍어(Modern Standard Arabic, MSA) 및 아랍어 방언의 합성 데이터셋을 7개 소개하고, AraDiCE라는 새로운 벤치마크를 발표합니다. 이 벤치마크는 방언 이해(dialect comprehension) 및 생성(generation) 평가에 초점을 맞추고 있습니다.

- **Technical Details**: 연구에서는 기계 번역(Machine Translation, MT)과 인간의 후편집을 결합하여 방언을 포함한 7개의 합성 데이터셋을 구축하였습니다. 이 데이터셋은 레반트(Levantine)와 이집트(Egypt) 방언에 중점을 두고 있으며, 지역 문화 인식을 평가하기 위한 AraDiCE-Culture라는 새 벤치마크를 포함하고 있습니다. 이는 지역의 문화적 특성을 이해하는 데 집중하고 있습니다.

- **Performance Highlights**: 아랍어 중심 모델(Jais, AceGPT)은 방언 작업에서 다국어 모델보다 우수한 성능을 보였으나, 여전히 방언 식별(dialect identification), 생성, 번역에서 상당한 어려움이 있는 것으로 나타났습니다. LLM은 특정 방언을 이해하고 생성하는 데 제한이 있으며, 문화적 뉘앙스를 이해하는 데 있어 아랍어 중심 모델이 다국어 모델보다 우수합니다. 이 연구는 약 45,000개의 후편집 샘플과 문화 벤치마크를 제공하여 LLM의 성능을 향상시키는 맞춤형 훈련의 중요성을 강조합니다.



### NVLM: Open Frontier-Class Multimodal LLMs (https://arxiv.org/abs/2409.11402)
- **What's New**: NVLM 1.0은 비전-언어(vision-language) 작업에서 최첨단 결과를 달성하는 다중 모드 대형 언어 모델(multimodal large language models, LLMs)입니다. 특히, NVLM 1.0은 다중 모드 훈련 후 텍스트 전용 성능이 LLM 백본보다 개선되었습니다.

- **Technical Details**: NVLM 1.0은 세 가지 아키텍처(NVLM-D, NVLM-X, NVLM-H)로 구성되어 있습니다. NVLM-D는 디코더 전용 아키텍처로 OCR 관련 작업에서 높은 정확도를 제공하며, NVLM-X는 교차 주의(cross-attention) 기반 아키텍처로 고해상도 이미지를 처리하는 데 효율적입니다. NVLM-H는 두 가지 접근 방식을 통합하여 성능과 효율성을 동시에 향상시킵니다.

- **Performance Highlights**: 다양한 비전-언어 작업 및 텍스트 전용 작업에서 NVLM 모델들은 최고의 성능을 달성했습니다. 특히, 멀티모달 수학 및 코딩 데이터 세트를 통합하여 모든 NVLM 모델에서 텍스트 전용 성능의 개선이 달성되었습니다.



### LLM-Agent-UMF: LLM-based Agent Unified Modeling Framework for Seamless Integration of Multi Active/Passive Core-Agents (https://arxiv.org/abs/2409.11393)
Comments:
          35 pages, 14 figures, 3 tables

- **What's New**: 본 논문은 기존 LLM 기반 에이전트의 문제점을 해결하기 위해 통합된 소프트웨어 아키텍처를 제안하며, 이를 통해 보다 명확한 구조를 정의하고 모듈성을 향상시키고 있습니다.

- **Technical Details**: LLM-Agent-UMF(LLM 기반 에이전트 통합 모델링 프레임워크)는 에이전트의 구성 요소를 명확하게 구분하여 각 모듈 간의 상호작용을 정의합니다. 이 프레임워크는 계획, 기억, 프로필, 행동 및 보안이라는 5개의 모듈로 구성되어 있으며, 보안 모듈을 추가하여 에이전트의 신뢰성을 강화했습니다.

- **Performance Highlights**: 제안된 다양한 멀티 코어 에이전트 아키텍처가 실제 최첨단 에이전트에 적용되어 그 기능과 아키텍처적 측면을 명확히 분석하였으며, 여러 개별 에이전트를 통합한 하이브리드 활성/수동 코어 에이전트 시스템의 시스템 효율성을 평가하여 잠재적인 개선 방안을 제시했습니다.



### Diversify and Conquer: Diversity-Centric Data Selection with Iterative Refinemen (https://arxiv.org/abs/2409.11378)
Comments:
          21 pages, 6 figures

- **What's New**: 이 연구에서는 대형 언어 모델(LLM)의 단계적 훈련을 위해 최적의 데이터 서브셋을 선택하는 방법에 대한 새로운 접근 방식을 제안합니다. 기존 연구는 특정 인스턴스 품질에 중점을 두었으나, 우리는 전 세계적인 데이터 다양성을 강조합니다. k-means 클러스터링을 사용하여 전체 데이터 집합을 효과적으로 대표하는 서브셋을 선택하는 방법을 소개합니다.

- **Technical Details**: 제안하는 방법은 반복적인 세분화 기법을 활용하여 인스턴스를 클러스터로부터 재샘플링하고, 각 클러스터의 중요성과 샘플링 가중치를 훈련(iteration)마다 재평가합니다. 이 과정은 이상치(outlier)의 영향을 줄이고 저품질 데이터가 포함된 클러스터를 자동으로 필터링하는데 효과적입니다.

- **Performance Highlights**: 자연어 추론, 일반 세계 지식, 코드 및 수학 추론 작업 등의 평가에서 모델 성능이 일관되게 향상되었습니다. 무작위 선택 대비 7% 증가, 최신 샘플링 방법 대비 3.8% 개선된 성능을 달성했습니다. 또한, 대부분의 다운스트림(task)에서 모든 이전 방법보다 우수한 결과를 보여주었습니다.



### Multi-OCT-SelfNet: Integrating Self-Supervised Learning with Multi-Source Data Fusion for Enhanced Multi-Class Retinal Disease Classification (https://arxiv.org/abs/2409.11375)
Comments:
          25 pages, 9 tables, 10 figures

- **What's New**: 본 연구에서는 다양한 데이터 출처를 결합하여 좋은 성능과 새로운 데이터로의 일반화를 개선하기 위해 대규모 언어 모델(LLMs)과 SwinV2 기반의 자가 감독 학습 프레임워크를 개발하였다. 이 작업은 광학 단층 촬영(OCT) 이미지를 사용한 눈 질환 탐지 능력을 향상시키는 것을 목표로 한다.

- **Technical Details**: 두 단계 훈련 방법론인 자가 감독 프리트레이닝과 다운스트림 감독 분류기에 대한 미세 조정을 채택하였다. 여러 인코더 백본을 사용한 세 개의 데이터세트를 통해 수행된 절제 연구는 데이터 융합 없이, 데이터 가용성이 낮은 설정에서 자가 감독 프리트레이닝 없는 시나리오를 비교하여 우리의 방법론의 견고함을 강조하였다.

- **Performance Highlights**: 세 가지 다양한 조건에서 일관된 성능을 보여주며, ResNet-50의 기준 모델과 비교할 때 우수한 일반화 능력을 갖추었다. 본 연구는 제한된 데이터로도 효과적인 학습이 가능하도록 설계되었다.



### CORE-Bench: Fostering the Credibility of Published Research Through a Computational Reproducibility Agent Benchmark (https://arxiv.org/abs/2409.11363)
Comments:
          Benchmark harness and code available at this http URL

- **What's New**: CORE-Bench (Computational Reproducibility Agent Benchmark)가 도입되었습니다. 이 벤치마크는 90편의 과학 논문을 기반으로 270개의 작업으로 구성되어 있으며, 컴퓨터 과학, 사회 과학, 의학의 세 가지 분야를 포함합니다. 이 작업은 AI 에이전트의 실험 차원에서 중요한 컴퓨테이셔널 재현성(computational reproducibility)을 평가하는 데 중점을 두고 있습니다.

- **Technical Details**: CORE-Bench는 세 가지 난이도 수준의 작업으로 구성되며, language-only 및 vision-language 작업을 포함합니다. 두 개의 기본 에이전트인 일반 목적의 AutoGPT와 특정 작업에 맞춘 CORE-Agent를 평가하였으며, 각각 GPT-4o와 GPT-4o-mini 기반으로 실험을 진행했습니다. CORE-Bench는 특정 AI 에이전트의 성능을 신속하고 병렬적으로 측정할 수 있는 평가 시스템을 제공하여, 평가 시간이 단축되었습니다.

- **Performance Highlights**: 가장 높은 성능을 보인 에이전트는 가장 어려운 작업에서 21%의 정확도를 기록하였습니다. 이는 과학적 연구의 자동화에 대한 잠재력이 크지만, 미흡한 부분이 많다는 것을 나타냅니다. 특정 작업에 적응할 수 있는 일반 에이전트의 성능 향상 가능성도 확인되었습니다.



### AI Suggestions Homogenize Writing Toward Western Styles and Diminish Cultural Nuances (https://arxiv.org/abs/2409.11360)
- **What's New**: 본 논문에서는 서구 중심의 AI 모델이 사용자에게 특히 비서구 문화 배경을 가진 사용자에게 어떤 영향을 미치는지를 다룹니다. 이를 위해 인도와 미국의 참여자를 대상으로 한 문화 간 실험을 진행하여 AI가 작성 제안하는 방식이 서로 다른 문화적 배경을 가진 사용자에게 어떻게 작용하는지를 분석했습니다.

- **Technical Details**: 연구팀은 118명의 참여자를 모집하여, AI 제안 유무에 따라 문화적으로 기반한 작성 과제를 수행하도록 했습니다. 연구는 Hofstede의 문화 양파 모델을 활용해 문화적 관행을 이끌어 냈습니다. AI 조건과 No AI 조건에서 작성된 에세이를 비교 분석하여, AI가 서구 중심의 작문 스타일로 비서구 사용자의 작문 스타일을 동질화한다는 것을 발견했습니다.

- **Performance Highlights**: AI는 미국 참여자들에게 더 높은 생산성 향상을 보여 주었으며, 인도 참여자들은 AI의 작문 제안 때문에 서구 스타일을 더 많이 채택하게 되어 문화적 표현의 뉘앙스가 사라지는 위험이 있음을 보여주었습니다. 이 연구는 LLM(대형 언어 모델)의 문화적 편향과 이로 인한 잠재적 해악을 논의하고, 문화적 제국주의와 언어 단일성을 해결하기 위한 전략을 제안합니다.



### RenderWorld: World Model with Self-Supervised 3D Lab (https://arxiv.org/abs/2409.11356)
- **What's New**: RenderWorld는 LiDAR-vision 융합 방식보다 비용 효율적이고 신뢰성이 높은 순수 비전을 기반으로 한 자율 주행 프레임워크입니다.

- **Technical Details**: 이 시스템은 self-supervised gaussian 기반의 Img2Occ 모듈을 사용하여 3D occupancy 레이블을 생성하고, AM-VAE(Adversarial Variational Autoencoder)를 통해 레이블을 인코딩한 후, 세계 모델을 이용하여 예측 및 계획을 수행합니다. Gaussian Splatting 기법을 적용하여 3D 장면을 표현하고 2D 이미지를 렌더링함으로써 NeRF 기반 방법에 비해 세분화 정확도를 크게 향상시키고 GPU 메모리 소비를 줄입니다.

- **Performance Highlights**: RenderWorld는 공기와 비공기를 별도로 인코딩하는 AM-VAE의 적용을 통해 보다 세밀한 장면 요소 표현을 달성하며, 4D occupancy 예측과 자가 회귀형 세계 모델에 의한 모션 계획에서 최첨단 성능을 보여줍니다.



### Clinical Validation of a Real-Time Machine Learning-based System for the Detection of Acute Myeloid Leukemia by Flow Cytometry (https://arxiv.org/abs/2409.11350)
- **What's New**: 본 논문에서는 Flow Cytometry(유세포분석기)에서 Acute Myeloid Leukemia(급성 골수성 백혈병) 감지를 위한 머신러닝(ML) 모델과 이를 지원하는 임상 적용 인프라에 대해 설명합니다.

- **Technical Details**: 제안된 인프라는 클라우드(compute cloud)를 활용하여 모델 추론(model inference)을 수행하고, Kubernetes 기반의 워크플로우 시스템을 통해 모델 재현성(model reproducibility)과 자원 관리를 제공합니다. 또한, 전체 텍스트 보고서에서 구조화된 진단 정보를 추출하는 시스템을 포함합니다.

- **Performance Highlights**: 배포 후 분석을 통해 회전 시간(turn-around time)에 미친 영향을 평가하고, 생산 정확도(production accuracy)를 원래 검증 통계(original validation statistics)와 비교하였습니다. 이 분석은 ML 모델의 임상적 효용성을 잘 보여줍니다.



### OmniGen: Unified Image Generation (https://arxiv.org/abs/2409.11340)
- **What's New**: OmniGen은 통합된 이미지 생성을 위한 새로운 diffusion 모델로, 기존의 ControlNet이나 IP-Adapter 같은 추가 모듈 없이 다양한 제어 조건을 처리할 수 있습니다. 이 모델은 텍스트-이미지 생성뿐만 아니라 이미지 편집 및 주제 기반 생성과 같은 다운스트림 작업을 지원합니다.

- **Technical Details**: OmniGen의 구조는 Variational Autoencoder (VAE)와 transformer 모델로 구성되어 있으며, 추가 텍스트 인코더가 필요하지 않습니다. 이는 사용자가 복잡한 작업을 지시를 통해 수행할 수 있도록 도와줍니다. 모델은 텍스트와 이미지를 동시에 입력받아 처리할 수 있습니다.

- **Performance Highlights**: OmniGen은 기존의 이미지 생성 모델보다 경쟁력 있는 텍스트-이미지 생성 능력을 보이며, 이미지 편집, 시각적 조건 생성을 포함한 다양한 이미지 생성 작업을 효과적으로 지원합니다. 실험 결과, OmniGen은 이전에 본 적 없는 작업과 도메인에서도 혁신적인 능력을 나타냅니다.



### SOAP: Improving and Stabilizing Shampoo using Adam (https://arxiv.org/abs/2409.11321)
- **What's New**: Shampoo와 Adafactor 사이의 공식적인 연결고리를 형성하였고, Shampoo의 고유 기저(eigenbasis)에서 Adam을 실행하는 SOAP 알고리즘을 설계하였다.

- **Technical Details**: SOAP(SHampoO with Adam in the Preconditioner’s eigenbasis)는 Shampoo와 Adafactor 사이의 관계를 활용하여 설계된 알고리즘이다. Shampoo의 고유 기저에서 AdamW를 실행하고, 추가적인 하이퍼파라미터(hyperparameter)로는 정규화 주기(preconditioning frequency)를 도입한다. Adam 방식으로 두 번째 모멘트의 이동 평균을 지속적으로 업데이트하여 성능을 개선한다.

- **Performance Highlights**: SOAP은 360m 및 660m 크기의 언어 모델에 대한 사전 훈련(pre-training) 작업에서 AdamW와 Shampoo보다 각각 40% 및 35% 이상의 반복 횟수 및 실행 시간을 줄여주는 성능 향상을 보여주었다. 또한, 대규모 배치에서 SOAP은 Shampoo에 비해 약 20% 정도의 성능 향상을 달성하였다.



### MSDNet: Multi-Scale Decoder for Few-Shot Semantic Segmentation via Transformer-Guided Prototyping (https://arxiv.org/abs/2409.11316)
- **What's New**: 본 연구에서는 Few-shot Semantic Segmentation (FSS) 문제를 해결하기 위해 transformer 아키텍처 기반의 새로운 프레임워크를 제안합니다. 이 프레임워크는 spatial transformer decoder와 contextual mask generation 모듈을 도입하여 support 이미지와 query 이미지 간의 관계적 이해를 개선하고, 글로벌 피처를 통합하여 컨텍스트 이해를 향상시킵니다.

- **Technical Details**: 제안된 방법은 multi-scale decoder를 사용하여 다양한 해상도의 피처를 계층적으로 통합하여 segmentation mask를 세밀하게 개선합니다. 또한, encoder의 중간 단계에서 글로벌 피처를 통합하여 복잡도를 줄이고 경량 구조를 유지합니다.

- **Performance Highlights**: 이 접근 방식은 1-shot 및 5-shot 설정 모두에서 $PASCAL-5^i$ 및 $COCO-20^i$와 같은 벤치마크 데이터셋에서 최신 기술을 능가하는 성능을 달성하였습니다. 특히, 150만 개의 매개변수만을 사용하여 기존의 방법론의 한계를 극복하면서 경쟁력 있는 성능을 보여주었습니다.



### TTT-Unet: Enhancing U-Net with Test-Time Training Layers for biomedical image segmentation (https://arxiv.org/abs/2409.11299)
- **What's New**: TTT-Unet는 기존 U-Net 구조에 Test-Time Training(TTT) 레이어를 통합하여 생체의학적 이미지 세분화에서의 장기 의존성을 모델링하는 한계를 극복합니다. 이 새로운 프레임워크는 테스트 시 모델 파라미터를 동적으로 조정하여 지역 및 장거리 피처를 보다 효과적으로 포착할 수 있도록 합니다.

- **Technical Details**: TTT-Unet는 기존 U-Net 구조에 Mamba 블록 내에서 TTT 레이어를 통합하여 설계되었습니다. 이 구조는 테스트 데이터에 기반하여 지속적으로 모델의 파라미터를 업데이트할 수 있도록 하여 특징 추출 능력을 향상시키고 장거리 의존성을 적응적으로 학습할 수 있게 합니다.

- **Performance Highlights**: TTT-Unet는 CT 및 MRI 스캔에서의 3D 복부 장기 세분화, 내시경 이미지에서의 기구 세분화, 현미경 이미지에서의 세포 세분화와 같은 다양한 의료 이미징 데이터세트에서 평가되었으며, 기존의 CNN 및 Transformer 기반 세그멘테이션 모델보다 모든 작업에서 일관되게 우수한 성능을 보였습니다.



### EIA: Environmental Injection Attack on Generalist Web Agents for Privacy Leakag (https://arxiv.org/abs/2409.11295)
Comments:
          24 pages

- **What's New**: 이번 연구에서는 일반화된 웹 에이전트(Generalist web agents)의 프라이버시 리스크를 탐구하며, 이를 위한 최초의 연구를 진행했습니다. 특히, 적대적 환경에서의 에이전트의 안전성을 고찰합니다.

- **Technical Details**: 우리는 두 가지 유형의 적대적 목표를 고려했습니다: 사용자의 특정 개인 식별 정보(PII)를 탈취하거나, 전체 사용자 요청을 도용하는 것입니다. 이를 위해 Environmental Injection Attack (EIA)이라는 새로운 공격 방법을 제안하며, 이는 에이전트가 의도치 않은 행동을 하도록 유도하는 악성 콘텐츠를 주입합니다. 이 공격은 정보 유출을 유도하기 위해 설계된 악성 웹 요소를 삽입합니다.

- **Performance Highlights**: EIA는 최대 70%의 ASR(Attack Success Rate)으로 사용자의 특정 PII를 탈취할 수 있었으며, 전체 사용자 요청을 도용하는 것은 더 도전적이지만 완화된 버전의 EIA는 여전히 16% ASR을 달성했습니다. 이러한 결과는 높은 자율성과 보안 간의 상충관계를 강조합니다.



### Zero-resource Hallucination Detection for Text Generation via Graph-based Contextual Knowledge Triples Modeling (https://arxiv.org/abs/2409.11283)
- **What's New**: 이번 연구에서는 그래프 기반의 맥락 인식(GCA, Graph-Based Context-Aware) hallucination 탐지 방법을 제안하여, 긴 텍스트 생성에 있어서 여러 사실 간의 종속성을 고려하고 사실 정렬을 개선하여 일관성 비교를 수행합니다.

- **Technical Details**: 제안된 방법은 삼중항(response segmentation) 기반으로 여러 지식 삼중(triples)을 추출하고, RGCN(Relational Graph Convolutional Network)을 통해 사실 간의 종속성을 모델링합니다. 이를 통해 연결된 노드의 특성을 집계하여 정보 전송을 촉진시킵니다.

- **Performance Highlights**: 실험 결과, 본 방법은 외부 자원 없이 검정 모델에서 생성된 긴 텍스트 응답의 hallucination 탐지 정확도를 효과적으로 향상시켰으며, 모든 기본선(line)보다 우수한 성능을 보였습니다.



### Task Arithmetic for Language Expansion in Speech Translation (https://arxiv.org/abs/2409.11274)
- **What's New**: 최근 대규모 언어 모델(LLMs)의 발전으로 인해 음성-텍스트(multimodal) 기초 모델이 주목받고 있으며, 지침 기반의 음성 번역(speech translation, ST)에서 뛰어난 성능을 보여주고 있습니다. 그러나 기존 ST 시스템에서 언어 쌍을 확장하는 작업은 비싼 재훈련을 요구합니다. 본 연구에서는 언어 제어 모델(language control model)을 추가하여 언어 혼동을 제거하고 새로운 언어 쌍을 확장하는 방법을 제안합니다.

- **Technical Details**: 본 연구에서는 기존 ST 모델과 새로운 언어 쌍을 훈련받은 모델을 결합하는 모델 병합(Model Merging) 접근법을 사용합니다. 기본적으로 작업 산술(task arithmetic)을 통해 모델의 파라미터를 조작하여 새로운 모델을 생성하고, 추가된 언어 제어 모델을 통해 정확한 목표 언어 토큰을 생성하도록 유도합니다. 실험에서는 MuST-C 및 CoVoST-2 데이터셋을 사용하여 성능을 평가하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 언어 제어 모델을 통해 MuST-C 데이터셋에서 최대 4.66, CoVoST-2 데이터셋에서 최대 4.92 의 BLEU 점수 향상을 달성했습니다. 또한, ST 훈련 데이터가 없거나 사전 훈련된 ST 모델이 존재하지 않는 경우에도 새로운 언어 쌍에 대해 시스템을 합성하고 기존 모델에 병합하여 성능을 향상시킬 수 있음을 보여주었습니다.



### LOLA -- An Open-Source Massively Multilingual Large Language Mod (https://arxiv.org/abs/2409.11272)
- **What's New**: 이번 논문에서는 160개 이상의 언어로 훈련된 대규모 멀티링구얼 모델 LOLA를 소개합니다. LOLA는 sparse Mixture-of-Experts Transformer 아키텍처를 기반으로 하여 언어적 다양성을 활용하는 동시에 효율성을 유지하고 멀티링구얼리티(multilinguality)의 일반적인 문제들을 피하도록 설계되었습니다.

- **Technical Details**: LOLA는 GPT 스타일의 decoder-only 아키텍처를 따르며, sparse Mixture-of-Experts (MoE) 층을 활용하여 모델 용량을 늘립니다. MoE는 특정 언어 군에 대한 은닉 구조를 학습하고, 이를 통해 더 나은 언어 모델 성능을 달성할 수 있도록 돕습니다. 또한, LOLA는 정확한 데이터셋 분석과 훈련 과정에 대한 심층적인 탐구를 제공합니다.

- **Performance Highlights**: LOLA는 4가지 작업 유형(질문 응답(Q&A), 추론, 자연어 추론(NLI), 독해)을 평가하였고, 13개의 멀티링구얼 작업에서 17개의 다른 모델들과 비교하여 경쟁력 있는 성능을 보였습니다. 특히, LOLA는 낮은 자원 언어를 위한 교차 언어 전이 학습 향상에 긍정적인 영향을 미쳤습니다.



### Integrating Reinforcement Learning and Model Predictive Control with Applications to Microgrids (https://arxiv.org/abs/2409.11267)
- **What's New**: 이 논문은 강화 학습(reinforcement learning)과 모델 예측 제어(MPC)를 통합하여 혼합 논리 동적 시스템(mixed-logical dynamical systems)에서 유한한 시간 범위 최적 제어 문제를 효과적으로 해결하는 접근 방식을 제안합니다.

- **Technical Details**: 이 연구는 이산(discrete) 및 연속(continuous) 의사결정 변수에 대한 결정을 분리하여 조합적(combinatorial) 문제의 복잡성을 완화합니다. 제안된 방법은 예측 지평선(prediction horizon)에서 Q-function을 분리하는 것을 통해 학습 문제를 단순화하고 반복 신경망(recurrent neural network)를 통해 과거 정보를 저장하여 Q-funciton을 계산합니다.

- **Performance Highlights**: 마이크로그리드(microgrid) 시스템에 대한 시뮬레이션 결과는 제안된 접근 방식이 MPC 제어기의 온라인 계산 시간을 크게 줄이고 최적성_gap과 높은 실행 가능성을 유지함을 보여줍니다.



### The Sounds of Home: A Speech-Removed Residential Audio Dataset for Sound Event Detection (https://arxiv.org/abs/2409.11262)
- **What's New**: 이 논문은 노인 복지를 증진하기 위한 스마트 홈 응용에 대한 사운드 이벤트 감지 연구를 지원하기 위해 주거 오디오 데이터셋을 제시합니다. 55세에서 80세 사이의 8명 참가자의 집에서 7일 동안 오디오 녹음 시스템을 배치하여 아카운적 특성을 문서화하였습니다.

- **Technical Details**: 이 연구는 'Sound Wellbeing in Later Life'라는 광범위한 연구의 일환으로 수행되었으며, AI 모델 배치를 위한 녹음 환경의 복제를 가능하게 하기 위해 구조 재료 정보와 자세한 바닥 계획을 포함한 데이터 수집 방법론을 사용하였습니다. 특히, 음성과 다른 사운드 이벤트를 구분하기 위해 사전 훈련된 오디오 신경망을 이용한 자동화 음성 제거 파이프라인을 개발했습니다.

- **Performance Highlights**: 결과적으로 이 데이터셋에는 주거 공간 내에서의 일상 활동 및 사운드스케이프를 정확하게 캡처한 비공식 오디오 녹음이 포함되어 있으며, 이는 가정 내 응용을 위해 특별히 맞춤화된 사운드 이벤트 감지 모델 개발 및 벤치마킹을 가능케 합니다.



### Attacking Slicing Network via Side-channel Reinforcement Learning Attack (https://arxiv.org/abs/2409.11258)
Comments:
          9 pages

- **What's New**: 이 논문은 5G 및 미래 6G 네트워크에서 네트워크 슬라이싱의 보안 취약점을 다루고 있습니다. 구체적으로 강화 학습(Reinforcement Learning)을 기반으로 한 사이드 채널 캐시 공격 프레임워크를 소개하여, 슬라이스 환경에서의 캐시 위치를 동적으로 식별하고 활용하는 방법을 제안합니다.

- **Technical Details**: 이 연구에서는 강화 학습을 통해 캐시 타이밍 공격을 모델링하며, 공격 슬라이스와 피해 슬라이스 사이의 추측 게임으로 케이스를 설정합니다. 실험을 통해 다양한 캐시 설계와 방어 수단에 자동으로 적응하여 캐시 타이밍 공격을 효과적으로 식별할 수 있는 능력을 강조합니다.

- **Performance Highlights**: 실험 결과, 제안한 접근법이 민감한 데이터 저장 위치를 정확히 식별하는 데 95%~98%의 성공률을 달성하며, 이는 공유 네트워크 슬라이싱 환경의 잠재적 위험성을 강조하고 강력한 보안 조치의 필요성을 부각시킵니다.



### Fast Analysis of the OpenAI O1-Preview Model in Solving Random K-SAT Problem: Does the LLM Solve the Problem Itself or Call an External SAT Solver? (https://arxiv.org/abs/2409.11232)
- **What's New**: OpenAI의 O1-preview 모델이 K-SAT 문제 해결에 대한 새로운 성능 분석이 진행되었으며, 모델이 외부 SAT solver를 호출했음을 발견했다는 점이 주목할 만하다.

- **Technical Details**: 모델 성능 분석은 K-SAT 문제의 채택도에 따라 달라지며, α = M/N의 함수로 작용한다. 모델은 문제를 직접 해결하기보다는 외부 SAT solver에 의존하며, 이는 성능 평가에 한계를 두고 있다. K는 2, 3, 4로 설정되어 있으며, 복잡한 문제의 경우 SAT solver와의 호출이 이루어졌다. 이 과정에서 모델의 학습 수준과 작동 원리에 대한 불확실성이 제기되었다.

- **Performance Highlights**: OpenAI O1-preview 모델은 초기 간단한 문제를 잘 해결했지만, 복잡성이 증가함에 따라 외부 도구를 사용하는 경향을 보였다. 분석 과정에서 모델이 K-SAT 문제를 스스로 해결할 수 있는지에 대한 의문이 제기되었고, 모델의 진정한 지능 여부에 대한 논의가 필요함을 알렸다.



### Learning Source Disentanglement in Neural Audio Codec (https://arxiv.org/abs/2409.11228)
Comments:
          project page: this https URL

- **What's New**: 이번 논문에서는 Source-Disentangled Neural Audio Codec (SD-Codec)라는 새로운 접근 방식을 소개합니다. 이 모델은 오디오 코딩과 소스 분리를 결합하여 서로 다른 소스 도메인을 구분하는 코드를 제공합니다. 이러한 접근은 오디오 신호의 세부사항을 보다 잘 이해하고 제어할 수 있도록 돕습니다.

- **Technical Details**: SD-Codec은 여러 도메인별 양자화기의 사용을 통해 다양한 오디오 소스(예: 음성, 음악, 음향 효과)를 분리합니다. 모델은 입력 오디오에 대해 여러 개의 양자화기를 훈련시켜, 각각의 도메인에 맞는 고유한 코드북을 할당합니다. 이 구조는 오디오 재합성을 위한 디코더와 결합되어 각 소스의 재구성이 가능합니다.

- **Performance Highlights**: 실험 결과에 따르면, SD-Codec은 경쟁력 있는 재합성 품질을 유지하고, 라텐트 공간(latent space)에서의 서로 다른 소스의 분리를 성공적으로 보여줍니다. 이를 통해 오디오 코드의 해석 가능성이 향상되며, 오디오 생성 프로세스에 대한 더 세밀한 제어가 가능해집니다.



### SDP: Spiking Diffusion Policy for Robotic Manipulation with Learnable Channel-Wise Membrane Thresholds (https://arxiv.org/abs/2409.11195)
- **What's New**: 본 논문은 Spiking Neurons와 Learnable Channel-wise Membrane Thresholds (LCMT)를 통합한 Spiking Diffusion Policy (SDP) 학습 방법을 소개하여 로봇 조작의 계산 효율성을 높이고 높은 성능을 달성합니다.

- **Technical Details**: SDP 모델은 U-Net 아키텍처를 기반으로 하여 Spiking Neural Network (SNN) 내에서의 확산 학습을 수행합니다. 스파이크 합성곱 연산과 Leaky Integrate-and-Fire (LIF) 노드 간의 잔여 연결을 전략적으로 배치하여 스파이크 상태에 대한 방해를 방지합니다. 또한, 시간 인코딩 블록과 시간 디코딩 블록을 도입하여 정적 및 동적 데이터를 서로 변환합니다. LCMT를 통해 채널 간의 막 전위 임계값을 적응적으로 획득할 수 있습니다.

- **Performance Highlights**: SNN 타임스텝 $T_S=4$에서 SDP 모델을 평가한 결과, ANN과 유사한 결과를 달성했으며, SNN 방법보다 빠른 수렴 속도를 보였습니다. 이 개선은 45nm 하드웨어에서 추정한 94.3%의 동적 에너지 소비 절감과 함께 이루어졌습니다.



### Towards Ethical Personal AI Applications: Practical Considerations for AI Assistants with Long-Term Memory (https://arxiv.org/abs/2409.11192)
- **What's New**: 이 논문은 개인 AI 동반자 및 어시스턴트에서 장기 기억(Short-term Memory, LTM) 기능을 활용하는 발전의 의미와 함께 이러한 시스템의 배포에 관한 중요 고려사항을 조사합니다.

- **Technical Details**: 장기 기억(LTM) 메커니즘은 초기 심볼릭 시스템에서 현대 대규모 언어 모델(LLM)로 발전해왔습니다. 이를 통해 AI는 사용자 선호를 학습하고, 인터랙션의 맥락을 유지하고 사용자에게 맞춤화된 정보를 효과적으로 제공할 수 있습니다. LSTM(Long Short-Term Memory)과 Attention 메커니즘은 이러한 과제 해결을 도와줍니다.

- **Performance Highlights**: 다양한 개인 AI 애플리케이션이 장기 기억 기능을 통해 개인화된 상호작용을 향상시키며, Charlie Mnemonic과 Google Gemini와 같은 시스템은 개인의 작업을 지원하고, 높은 개인화된 경험을 제공합니다.



### SuperCoder2.0: Technical Report on Exploring the feasibility of LLMs as Autonomous Programmer (https://arxiv.org/abs/2409.11190)
- **What's New**: SuperCoder2.0는 소프트웨어 개발을 위한 최첨단 자율 시스템으로, 완전 자율 코딩 기능을 제공하며, 새로운 리트라이 메커니즘 및 오류 추적 기능을 포함하고 있습니다.

- **Technical Details**: SuperCoder2.0는 Retrieval Augmented Generation (RAG) 방식을 활용하여 코드 베이스 내에서의 탐색 및 오류 현황 작성에 대한 3단계 계층적 검색 공간 축소 접근 방식을 적용합니다. 이 과정에서 파일 수준 도면을 사용하여 후보 파일을 식별하고, 가장 관련 있는 파일로 좁히며, 해당 파일 내 '관련 위치'를 추출합니다.

- **Performance Highlights**: SWE-bench Lite 데이터셋에서 SuperCoder2.0은 84.33%의 경우에서 상위 5개 후보 중 올바른 파일을 로컬라이즈하며 34%의 테스트 인스턴스를 성공적으로 해결하는 성과를 보였습니다. 이는 SWE-bench 리더보드에서 SuperCoder2.0을 세계 4위에 올리는 성과입니다.



### Deep Learning tools to support deforestation monitoring in the Ivory Coast using SAR and Optical satellite imagery (https://arxiv.org/abs/2409.11186)
- **What's New**: 이번 연구에서는 코코아 생산이 활발한 코트디부아르에서의 산림 파괴를 모니터링하기 위해 Sentinel 위성 이미지를 이용한 Forest-Non-Forest map (FNF) 작성 방법이 제안되었습니다.

- **Technical Details**: U-Net, Attention U-Net, Segnet, FCN32와 같은 최신 모델들이 Sentinel-1 및 Sentinel-2 이미지를 결합하여 삼림/비삼림(segmentation) 분류에 사용되었으며, 구름이 자주 낀 지역에서도 효과적인 분석이 가능한 SAR (Synthetic Aperture Radar) 이미지를 활용하였습니다.

- **Performance Highlights**: 연구를 통해 2019년과 2020년 사이에 제거된 삼림 면적을 추정할 수 있었으며, 개방형 데이터셋을 통한 산림 및 비산림 픽셀 분류 모델의 가능성이 입증되었습니다.



### Identifying Influential nodes in Brain Networks via Self-Supervised Graph-Transformer (https://arxiv.org/abs/2409.11174)
- **What's New**: 본 논문은 자기지도 학습(self-supervised learning) 기반의 그래프 복원 프레임워크(SSGR-GT)를 통해 뇌 네트워크에서 영향력 있는 노드(I-nodes)를 식별하는 방법을 제안합니다. 기존 연구와 달리, 이 접근 방식은 그래프 이론에 의존하지 않고 데이터에서 의미 있는 특징을 학습합니다.

- **Technical Details**: SSGR-GT는 뇌를 그래프로 표현하며, 노드는 국소적 및 전반적 특징을 추출하기 위해 Graph-Transformer 구조를 사용합니다. 이 모델은 뇌의 기능적 연결성과 구조적 연결성을结合하여 노드의 중요도를 예측하고, 최종적으로 높은 점수를 가진 노드를 I-nodes로 식별합니다.

- **Performance Highlights**: 연구를 통해 식별된 I-nodes는 주요 뇌 영역에 분포하고 있으며, 이들은 더 많은 뇌 네트워크에 관여하고 긴 섬유 연결을 가지며, 구조적 연결성에서도 중심적인 위치를 차지합니다. 이들은 기능적 및 구조적 네트워크에서도 강한 연결성과 높은 노드 효율성을 보여줍니다.



### MAISI: Medical AI for Synthetic Imaging (https://arxiv.org/abs/2409.11169)
- **What's New**: 의료 이미징 분석에 대한 새로운 접근법인 Medical AI for Synthetic Imaging (MAISI)를 소개합니다. MAISI는 합성 3D CT 이미지를 생성하기 위해 diffusion model을 활용하여 데이터 부족, 높은 주석 비용, 그리고 개인정보 보호 문제를 해결합니다.

- **Technical Details**: MAISI는 volume compression network와 latent diffusion model을 활용하여 512 x 512 x 768 크기의 고해상도 CT 이미지를 생성합니다. ControlNet이 추가되어, 127개의 해부학적 구조에 대한 organ segmentation을 처리합니다. 이를 통해 다양한 다운스트림 작업에 활용할 수 있는 정확하게 주석이 달린 합성 이미지를 생성합니다.

- **Performance Highlights**: 실험 결과, MAISI는 현실적이고 해부학적으로 정확한 이미지를 생성할 수 있는 능력을 보여주며, 합성 데이터를 사용하여 여러 도전 과제를 완화하는 데 도움이 될 가능성을 제시합니다.



### Improving the Efficiency of Visually Augmented Language Models (https://arxiv.org/abs/2409.11148)
- **What's New**: 이 논문은 시각적 지식이 부족한 기존의 Language Model (LM)들을 개선하는 새로운 접근 방식을 제시합니다. 새로운 모델 BLIND-VALM은 이미지 검색 시스템이나 생성 시스템 없이 CLIP 모델에서 얻은 시각적 기반 텍스트 표현을 사용하여 LM을 시각적으로 보강합니다.

- **Technical Details**: BLIND-VALM은 기존의 VALM 구조를 수정하여 CLIP 모델의 텍스트 인코더 표현을 활용합니다. 이는 이미지 검색 및 표현을 피하면서도 LM에 시각적 지식을 통합할 수 있게 해줍니다. BLIND-VALM은 VLU, NLU 및 LM 작업에서 VALM과 동등한 성능을 보입니다.

- **Performance Highlights**: BLIND-VALM은 훈련 및 추론 속도에서 VALM보다 극적으로 빠르며, 동일한 계산 예산 내에서 VALM을 초과하는 성능 개선을 보여줍니다.



### High-Resolution Speech Restoration with Latent Diffusion Mod (https://arxiv.org/abs/2409.11145)
- **What's New**: 이 논문에서는 Hi-ResLDM이라는 새로운 생성 모델을 제안합니다. 이 모델은 여러 종류의 왜곡을 처리할 수 있도록 설계되었으며, 스튜디오 품질의 음성 복원을 목표로 하고 있습니다.

- **Technical Details**: Hi-ResLDM은 잠재적 확산(latent diffusion) 기반의 프레임워크로, 두 단계의 프로세스를 통해 동작합니다. 첫 번째 단계에서는 왜곡된 입력을 깨끗한 중간 추정값으로 변환하고, 두 번째 단계에서는 이 중간 결과를 최종 음성 추정값으로 복원합니다. 또한, SNR(신호 대 잡음 비율)을 개선하기 위해 왜곡 제거 및 강도(normalization) 조정을 분리하여 진행합니다.

- **Performance Highlights**: Hi-ResLDM은 기존의 GAN 및 Conditional Flow Matching(CFM) 기반 방법들과 비교하여 높은 성능을 보였으며, 사람 평가에서도 선호되었습니다. 특히 고주파 세부사항의 재생성에 있어서 우수하며, 고해상도 음성 복원에 적합합니다.



### Learning Generalized Hamiltonians using fully Symplectic Mappings (https://arxiv.org/abs/2409.11138)
Comments:
          Submitted to The 39th Annual AAAI Conference on Artificial Intelligence

- **What's New**: 이 논문은 일반화된 비분리 해밀토니안(GHH)을 사용하는 새로운 해밀토니안 신경망(HNN) 모델의 설계를 제안하며, 이는 물리적 보존 법칙을 준수하는 수용적 시뮬레이션을 위한 보다 효율적인 방법을 제공합니다.

- **Technical Details**: 해밀토니안 신경망(HNN)은 신경망(Neural Network)으로 파라미터화된 해밀토니안을 학습하기 위해 일반화된 비분리 해밀토니안을 채택하고, 시뮬레이션과 통합하기 위해 심플렉틱 적분기(symplectic integrator)를 사용합니다. 이를 통해 ODE 솔버(ODE solver)를 통한 역전파(backpropagation)의 계산 부담을 덜 수 있습니다. 또한, 예측자-수정자 방법(predictor-corrector method)을 사용하여 비분리 해밀토니안에 필요한 암시적 적분기(implicit integrator)의 계산 부담을 줄입니다.

- **Performance Highlights**: 본 연구에서 제안한 방법은 잡음이 있는 관찰로부터 상태 변수 샘플링 시 해밀토니안 시스템의 복원 및 보존에서 우수한 성능을 보여주었습니다. 특히, 일반화된 비분리 해밀토니안 시스템에서의 성능이 기존 방법보다 현저히 뛰어나다는 것을 확인하였습니다.



### Diversity-grounded Channel Prototypical Learning for Out-of-Distribution Intent Detection (https://arxiv.org/abs/2409.11114)
Comments:
          work in progress

- **What's New**: 이 연구는 대규모 언어 모델(large language models, LLMs)을 위한 새로운 파인튜닝 프레임워크를 제안하여 실세계 시나리오에서 발생하는 형식이 잘못된 발화를 효과적으로 처리할 수 있는 의도 감지 메커니즘을 향상시킵니다. 특히, 기존의 식별내부(in-distribution, ID) 의도 분류와 비식별내부(out-of-distribution, OOD) 의도 감지를 개선하는데 중점을 두고 있습니다.

- **Technical Details**: 이 프레임워크는 ID 클래스 이름에서 파생된 프로토타입과 의미적 매칭(semantic matching)을 사용하여, 각 ID 클래스를 위한 의미 프로토타입(semantic prototypes)을 구축합니다. 다양한 예제를 기반으로 한 프롬프트 튜닝(diversity-grounded prompt tuning) 접근 방식을 통해 LLM의 고유한 표현을 활용하여, LLM은 의미 프로토타입을 분류하는 데 사용됩니다. 본 연구는 ID와 OOD 클래스가 의미적으로 유사한 상황, 즉 'near' OOD 감지 문제를 다룹니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존의 파인튜닝 접근 방식과 비교하여 제한된 수의 샘플을 이용한 ID 의도 분류 및 near-OOD 의도 탐지 작업에서 우수한 성능을 보여줍니다. 이는 실질적인 응용에서의 효율성을 향상시킵니다.



### MonoKAN: Certified Monotonic Kolmogorov-Arnold Network (https://arxiv.org/abs/2409.11078)
Comments:
          10 pages, 2 figures

- **What's New**: 이번 연구에서는 최근 제안된 Kolmogorov-Arnold Network (KAN) 아키텍처를 기반으로 MonoKAN이라는 새로운 인공신경망(ANN) 아키텍처를 소개합니다. MonoKAN은 해석 가능성과 인증된 부분 단조성(partial monotonicity)을 동시에 달성합니다.

- **Technical Details**: MonoKAN은 KAN 아키텍처를 활용하여 cubic Hermite splines를 사용함으로써 단조성을 보장합니다. 이를 통해 입력-output 간의 단조 관계를 유지할 수 있으며, 모델의 해석 가능성도 증대합니다. 기존의 다층 퍼셉트론(MLP)에서는 해석 가능성과 인증된 부분 단조성 모두를 만족시키기 어려웠으나, MonoKAN은 이러한 문제를 해결합니다.

- **Performance Highlights**: 실험 결과, MonoKAN은 대부분의 벤치마크에서 예측 성능을 향상시켰으며, 기존의 단조 MLP 접근 방식보다 뛰어난 성능을 보였습니다.



### RoMath: A Mathematical Reasoning Benchmark in Romanian (https://arxiv.org/abs/2409.11074)
Comments:
          4 Figures, 12 Tables

- **What's New**: RoMath라는 새로운 Romanian 수학적 추론 벤치마크 스위트를 소개하며, 이는 세 가지 데이터셋(RoMath-Baccalaureate, RoMath-Competitions, RoMath-Synthetic)을 포함하여 다양한 수학 영역 및 난이도를 커버하고 있습니다. 이 연구는 비영어 언어 모델을 개선하고 다국어 AI 개발을 촉진하는 것을 목표로 합니다.

- **Technical Details**: RoMath는 약 76,910개의 문제 진술로 구성된 새로운 수학적 추론 벤치마크 스위트입니다. 이 데이터세트는 세 가지 하위 집합으로 나뉘며, 각각의 고유한 특성과 난이도 수준(예: RoMath-Baccalaureate와 같은 고등학교 수준 문제)을 가지고 있습니다. 연구팀은 70,000개의 문제를 포함한 데이터를 수집하고 정리하는 데 반자동 워크플로우를 사용하여 기초 LLM(large language models)을 활용했습니다.

- **Performance Highlights**: RoMath는 여러 영어 및 Romanian 개방형 가중치 LLM을 사용하는 포괄적인 벤치마크를 제공하며, LLM-as-a-judge 패러다임을 사용하여 평가 절차를 시행합니다. 이 연구에서는 단순한 문제 진술의 번역이 성능을 상당히 저하시킨다는 것을 강조하고, 영어와 다른 언어에서 전념할 필요가 있음을 논의합니다.



### Improve Machine Learning carbon footprint using Parquet dataset format and Mixed Precision training for regression algorithms (https://arxiv.org/abs/2409.11071)
Comments:
          35 pages, 16 tables, 19 figures. arXiv admin note: substantial text overlap with arXiv:2409.07853

- **What's New**: 이번 연구는 석사 학위 논문의 두 번째 부분으로, 회귀 ML (Machine Learning) 모델을 훈련하면서 Comma-Separated Values (CSV) 형식과 Parquet 데이터셋 형식의 전력 소비를 비교했습니다.

- **Technical Details**: 연구에서는 기본 부동 소수점 (32비트)과 Nvidia 혼합 정밀도 (16비트 및 32비트)를 사용하였으며, 다양한 ML 하이퍼파라미터 (hyper-parameters)로 Deep Neural Networks (DNN)을 구축했습니다. 실험에서는 Excel에 결과를 기록하였고, 그룹 간의 평균을 계산하기 위해 기술 통계 (descriptive statistics)를 사용했습니다. ANOVA와 T-test를 활용하여 평균 간의 관계를 비교했습니다.

- **Performance Highlights**: 혼합 정밀도를 특정 하이퍼파라미터와 결합하여 사용할 경우, 전력 소비가 7 ~ 11 Watts 감소했습니다. 그러나 하이 배치 크기와 많은 뉴런이 전력 소비에 부정적인 영향을 미칠 수 있으며, 회귀 테스트에서 평균 간의 통계적 유의성은 없었습니다. GPU 클러스터의 더 넓은 구현은 샘플 크기를 크게 증가시킬 수 있습니다.



### A Comprehensive Evaluation of Quantized Instruction-Tuned Large Language Models: An Experimental Analysis up to 405B (https://arxiv.org/abs/2409.11055)
Comments:
          11 pages, 1 figure

- **What's New**: 이번 연구에서는 다양한 양자화 방법(GPTQ, AWQ, SmoothQuant, FP8)을 사용하여 7B에서 405B에 이르는 instruction-tuned LLM의 성능을 평가하였습니다. 13개 벤치마크를 활용하여 발생중인 LLM의 정확성과 성능 변화를 종합적으로 분석하였습니다.

- **Technical Details**: 연구에서는 양자화 방법론으로 Quantization Aware Training (QAT)과 Post-Training Quantization (PTQ)을 분류하였습니다.  경량화된 모델을 위한 4개의 양자화 방법이 사용되었으며, 각 방법은 7B에서 405B에 이르는 다양한 크기의 모델에 적용되었습니다. 평가에서는 commonsense Q&A, 지식 및 언어 이해, instruction following, hallucination detection, 수학, 대화 등 총 6개 작업 유형이 사용되었습니다.

- **Performance Highlights**: 주요 발견 사항으로는 (1) 더 큰 LLM을 유사한 크기(예: 13B)로 양자화했을 때, 전반적으로 우수한 성능을 보였으며, (2) 양자화 방법, 모델 크기, 비트 폭에 따라 성능이 크게 변동했고, (3) 작업 난이도는 양자화로 인한 정확도 저하에 큰 영향을 미치지 않았고, (4) MT-Bench 평가 방법이 최근 고성능 LLM들 간의 차별화 능력이 제한적임을 확인하였습니다.



### A logical alarm for misaligned binary classifiers (https://arxiv.org/abs/2409.11052)
Comments:
          17 pages, 7 figures, under review

- **What's New**: 이 논문에서는 두 개의 에이전트가 이진 분류 작업에서 동의하지 않을 때, 그들이 모두 옳지 않을 수 있음을 직관적으로 이해하고 이를 형식화합니다. 에이전트의 의사결정 정합성을 평가하기 위한 공리 기반 시스템을 개발하였으며, 이를 통해 레이블이 라벨링되지 않은 데이터를 사용하여 적어도 한 구성원에 장애가 있음을 입증할 수 있는 완전한 논리적 경고 시스템을 구축합니다.

- **Technical Details**: 이 논문에서는 에이전트 평가의 형식을 정립하기 위해, 동의 및 불일치 데이터에 대한 집합의 공리를 설정하여 이진 응답자 평가를 수행합니다. N=1 및 N=2의 경우 각각에 대해 공리 집합을 유도하고, 이를 기반으로 오작동 감지를 위한 논리적 경고 시스템을 구축합니다. 이 접근 방식은 하는데 소프트웨어 검증 및 AI 시스템 안전성의 최근 요구와 유사성이 있으며, 형식적 소프트웨어 검증의 세 가지 측면인 세계 모델, 안전 사양, 검증자를 사용합니다.

- **Performance Highlights**: 구성원이 단일 분류기에서 법칙을 설정하고 이를 통해 시스템의 정합성 및 일관성을 입증함으로써, 안전 및 신뢰성을 높이고 재조정 필요성을 극복할 수 있습니다. 이 방법은 슈퍼 인텔리전트 에이전트와의 관계에서도 잠재적으로 유용할 수 있습니다.



### D2Vformer: A Flexible Time Series Prediction Model Based on Time Position Embedding (https://arxiv.org/abs/2409.11024)
- **What's New**: D2Vformer 모델이 새롭게 제안되어, 기존의 예측 모델에서 직면했던 시간 위치 정보의 복잡한 패턴을 효과적으로 포착하며, 동적 길이 변화에 대응할 수 있도록 설계되었습니다.

- **Technical Details**: D2Vformer의 핵심 기술은 Date2Vec 모듈로, 시간 스탬프 정보와 특징 시퀀스를 활용하여 시간 위치 임베딩을 생성합니다. 이를 통해 모델은 입력 시퀀스와 예측 시퀀스 간의 유사성을 탐색하여 예측을 생성하는 새로운 Fusion Block을 도입합니다.

- **Performance Highlights**: D2Vformer는 다양한 데이터 세트에 대한 실험에서 Date2Vec이 다른 시간 위치 임베딩 방법보다 우수함을 보여주었으며, 고정 길이와 가변 길이 예측 작업 모두에서 최첨단 방법들을 초월하는 성능을 발휘했습니다.



### GEIC: Universal and Multilingual Named Entity Recognition with Large Language Models (https://arxiv.org/abs/2409.11022)
- **What's New**: 이번 논문에서는 Named Entity Recognition (NER) 작업을 위한 새로운 접근법인 Generation-based Extraction and In-context Classification (GEIC)을 제안합니다. GEIC는 LLMs의 주의 메커니즘을 활용하여 기존의 NER 방법보다 더 효율적으로 작동하도록 설계되었습니다.

- **Technical Details**: CascadeNER라는 프레임워크를 통해 GEIC를 구현하며, 이는 두 개의 소규모 LLM(Small Language Models, SLMs)을 이용하여 추출 및 분류 작업을 수행합니다. 이 방식은 자원 소비를 줄이고 정확성을 증대시킵니다. 또한, AnythingNER이라는 첫 번째 NER 데이터셋을 소개하며, 8개 언어와 155개의 개체 유형을 포함하고 있습니다.

- **Performance Highlights**: CascadeNER는 CrossNER와 FewNERD를 포함한 낮은 자원 및 세밀한 시나리오에서 SOTA(state-of-the-art) 성능을 달성했습니다. 또한, 이 연구는 LLM 기반 NER 방법에 대한 포괄적인 비교 및 분석을 제공하여 향후 방향성을 제시합니다.



### Enhanced segmentation of femoral bone metastasis in CT scans of patients using synthetic data generation with 3D diffusion models (https://arxiv.org/abs/2409.11011)
Comments:
          14 pages, 5 figures 3 tables

- **What's New**: 본 연구는 CT 스캔에서 대퇴골 전이의 세분화를 개선하기 위해 3D Denoising Diffusion Probabilistic Models (DDPM)을 활용한 자동 데이터 합성 파이프라인을 제안합니다.

- **Technical Details**: 29개의 기존 병변 및 26개의 건강한 대퇴골 이미지를 사용하여 새로운 합성 전이 이미지를 생성하고, 이를 통해 생성된 합성 데이터와 실제 데이터를 사용하여 3D U-Net 모델을 훈련시킵니다. 각 모델의 성능은 훈련에 사용된 합성 데이터의 양에 따라 평가됩니다.

- **Performance Highlights**: 합성 데이터로 훈련된 세분화 모델이 실제 데이터에 대해서만 훈련된 모델을 능가하였으며, 특히 운영자 변동성을 고려할 때 더욱 우수한 성능을 보였습니다.



### Single-stage TTS with Masked Audio Token Modeling and Semantic Knowledge Distillation (https://arxiv.org/abs/2409.11003)
Comments:
          Demo page: see this https URL

- **What's New**: 이번 연구에서는 음성 합성을 위한 오디오 토큰 모델링에서 단일 단계(single-stage) 접근 방식을 통해 고품질 음성을 생성하는 새로운 방법인 의미 지식 증류(semantic knowledge distillation, SKD)를 도입하여 기존의 두 단계(two-stage) 접근 방식을 간소화하였습니다.

- **Technical Details**: 제안된 음성 합성 모델인 NARSiS(NAR Single Stage TTS)는 의미와 음향 모델링을 병행하여 단일 단계 디자인을 유지합니다. 훈련 과정에서 자가 지도 학습(self-supervised learning)을 통해 음성 인코더에서 고수준의 지식을 증류함으로써 효율을 높이고 있습니다.

- **Performance Highlights**: NARSiS 모델은 단일 단계 기준선 모델과 비교할 때 음성 품질과 이해도(intelligibility), 화자 유사성에서 개선된 결과를 보였으며, 두 단계 NAR TTS 모델과의 성능 격차를 크게 줄였습니다.



### Enhancing Low-Resource Language and Instruction Following Capabilities of Audio Language Models (https://arxiv.org/abs/2409.10999)
Comments:
          5 pages. Preprint under review

- **What's New**: 이번 논문은 오디오 언어 모델이 태국어와 같은 저자원 언어에서 영어 데이터와의 공통성을 통해 훈련받은 기존 모델의 한계를 지적합니다. 특히, 저자원 언어에 대한 이해능력이 부족하다는 점을 강조하며, Typhoon-Audio라는 새로운 모델이 기존 오픈소스 모델들보다 우수한 성능을 보여줌을 입증했습니다.

- **Technical Details**: Typhoon-Audio 모델은 SALMONN 아키텍처를 기반으로 하며, 타겟 언어로 태국어와 영어를 설정하였습니다. Whisper-large-v3와 BEATS가 오디오 인코더 역할을 하며, Q-Former가 어댑터로 사용됩니다. 모델 훈련은 두 단계로 나누어 진행되며, 첫 단계에서는 오디오와 텍스트 표현을 정렬하기 위해 어댑터만 훈련되고, 두 번째 단계에서는 다양한 태스크를 통해 모델의 명령 수행 능력을 향상시킵니다.

- **Performance Highlights**: Typhoon-Audio 모델은 영어 및 태국어 모두에서 최신 기술의 Gemini-1.5-Pro와 비교해도 유사한 성능을 보이며, 기존 오픈소스 오디오 언어 모델에 비해 상당히 개선된 성능을 기록했습니다.



### Less is More: A Simple yet Effective Token Reduction Method for Efficient Multi-modal LLMs (https://arxiv.org/abs/2409.10994)
Comments:
          9 pages, 3 figures, 6 tables

- **What's New**: 이 연구는 Multimodal Large Language Models (MLLMs)의 자원 소모 문제를 해결하기 위해 새로운 접근 방식을 제안합니다. 이 방법은 CLIP metric을 활용하여 이미지 토큰을 선택하고 줄이는 Token Reduction using CLIP Metric (TRIM)입니다.

- **Technical Details**: TRIM은 Visual Question Answering (VQA) 작업에서의 인간 주의 패턴에 영감을 받아 이미지 토큰의 선택 및 감축을 위한 새로운 관점을 제시합니다. 이 방식은 12개의 데이터셋에서 광범위하게 테스트되었으며, 성능을 유지하면서 계산 오버헤드를 크게 줄이는 결과를 보여주었습니다. 또한, TRIM은 Interquartile Range (IQR) 스코어 기능을 사용하여 질문 응답에 중요한 이미지 토큰을 적응적으로 선택합니다.

- **Performance Highlights**: TRIM 방법은 약 79%의 이미지 토큰 수 감소, 처리 시간 67% 단축 및 메모리 사용량 30% 절감을 달성했습니다. 이 효율성은 원래 모델과 유사한 성능을 유지하면서 이루어졌습니다.



### GOSt-MT: A Knowledge Graph for Occupation-related Gender Biases in Machine Translation (https://arxiv.org/abs/2409.10989)
Comments:
          Accepted at the KG-STAR'24: Workshop on Knowledge Graphs for Responsible AI co-located with the 33rd ACM CIKM Conference, October 25, 2024, Boise, Idaho

- **What's New**: 이 논문은 직업과 관련된 성별 편향(gender bias)을 연구하기 위한 새로운 접근 방식인 GOSt-MT(Gender and Occupation Statistics for Machine Translation) 지식 그래프(Knowledge Graph)를 소개합니다.

- **Technical Details**: GOSt-MT는 실제 노동 data의 포괄적인 성별 통계와 MT 훈련(training)에 사용된 텍스트 코퍼스를 통합하여 생성되었습니다. 이 지식 그래프는 영어, 프랑스어, 그리스어에서 성별 편향을 상세히 분석할 수 있는 도구를 제공합니다.

- **Performance Highlights**: GOSt-MT는 노동 시장과 MT 시스템에서 직업이 성별화되는 방식을 이해하는 구조화된 프레임워크를 제공함으로써 MT 시스템을 더욱 공평하게 만들고 자동 번역에서 성별 편향을 줄이기 위한 노력을 지원합니다.



### Control-flow Reconstruction Attacks on Business Process Models (https://arxiv.org/abs/2409.10986)
- **What's New**: 이 논문은 프로세스 모델을 기반으로 한 재구성이 기업 프로세스의 비밀 정보를 노출할 수 있는 위험성을 실험적으로 조사한 첫 번째 연구입니다.

- **Technical Details**: 각 프로세스 모델은 이벤트 로그로부터 자동으로 생성되며, 여기에는 실행 빈도와 같은 행동 통계가 주석으로 포함될 수 있습니다. 연구는 프로세스 트리 모델을 기반으로 제어 흐름을 재구성하는 다양한 play-out 전략을 제안합니다.

- **Performance Highlights**: 실험 결과, 구조화된 프로세스의 빈도 주석이 있는 모델들이 특히 취약하다는 것이 밝혀졌습니다. 이는 비즈니스 프로세스를 외부에 공개하는 경우의 위험성을 강조합니다.



### Active learning for energy-based antibody optimization and enhanced screening (https://arxiv.org/abs/2409.10964)
Comments:
          8 pages

- **What's New**: 이 연구는 특정 단백질 표적에 대한 에너지 함수를 효율적으로 학습하도록 깊은 학습 모델을 훈련시키는 능동 학습 워크플로우를 제안합니다. Trastuzumab을 최적화하여 보다 나은 결합 친화도를 갖는 변종을 탐지하는 데 성공하였습니다.

- **Technical Details**: 이 방법은 RDE-Network 깊이 학습 모델과 Rosetta의 에너지 함수 기반 Flex ddG를 통합하여 변이를 효율적으로 탐색합니다. 모델은 다중 작업 학습(multitask learning)을 통해 실험적 ΔΔG 값과 계산된 결합 정보 예측을 통합하여 작동합니다.

- **Performance Highlights**: HER2 결합 Trastuzumab 변종을 대상으로 한 사례 연구에서, 제안된 방법은 무작위 선택 대비 스크리닝 성능을 유의미하게 개선하여 실험적 ΔΔG 데이터 없이도 더 나은 결합 특성을 가진 변종을 식별하는 능력을 입증했습니다.



### Versatile Incremental Learning: Towards Class and Domain-Agnostic Incremental Learning (https://arxiv.org/abs/2409.10956)
Comments:
          17 pages, 6 figures, 6 tables, ECCV 2024 Poster

- **What's New**: 본 연구에서는 기존의 Incremental Learning (IL) 방법론에 비해 더 현실적이고 도전적인 상황인 Versatile Incremental Learning (VIL)을 제안합니다. VIL은 다음 작업에서 어떤 클래스나 도메인이 증가할지에 대한 사전 정보가 없으며, 주로 intra-class domain confusion과 inter-domain class confusion 문제를 다룹니다.

- **Technical Details**: VIL 스cenario에 대응하기 위해 Incremental Classifier with Adaptation Shift cONtrol (ICON)라는 새로운 IL 프레임워크를 제안하며, Cluster-based Adaptation Shift conTrol (CAST)라는 정규화 방법을 통해 모델이 이전에 학습한 지식과의 혼동을 피하고 새로운 지식을 효과적으로 습득할 수 있도록 합니다. 또한, Incremental Classifier (IC)를 도입하여 다양한 도메인에서 학습 시 과거 지식을 유지하면서 출력 노드를 확장합니다.

- **Performance Highlights**: ICON은 3개의 벤치마크에서 수행된 실험을 통해 모든 시나리오에서의 성능이 입증되었으며, 특히 다음 작업이 랜덤하게 변경될 수 있는 경우에 특징적으로 뛰어난 성능을 보였습니다. 본 연구의 결과는 기존의 IL 방법들에 비해 월등한 성능 향상을 보여줍니다.



### Investigating Context-Faithfulness in Large Language Models: The Roles of Memory Strength and Evidence Sty (https://arxiv.org/abs/2409.10955)
- **What's New**: 이번 연구는 Retrieval-Augmented Generation (RAG) 과정에서 Large Language Models (LLMs)의 문맥 충실성(context-faithfulness)을 탐구하고, 기억 강도(memory strength) 및 증거 제출 방식의 영향을 조사하였습니다.

- **Technical Details**: 연구에서는 LLM의 기억 강도를 양적으로 측정하기 위해 같은 질문의 다양한 패러프레이즈에 대한 LLM 응답의 이탈(divergence)을 평가하는 방법을 도입했습니다. 이에 따라 직접적( direct ) 및 간접적( indirect )으로 제공되는 증거의 다양한 스타일을 생성하여 그 효과를 평가했습니다.

- **Performance Highlights**: 연구 결과, LLM들은 강한 기억력을 가진 질문에 대해서는 내부 기억에 의존할 가능성이 높으며, 특히 GPT-4와 같은 대형 모델에서 두드러졌습니다. 또한 패러프레이즈된 증거를 제시할 경우 LLM의 외부 정보 수용성(receptiveness)이 크게 증가하여 단순 반복보다 효과적이라는 사실이 밝혀졌습니다.



### Contrasformer: A Brain Network Contrastive Transformer for Neurodegenerative Condition Identification (https://arxiv.org/abs/2409.10944)
- **What's New**: 본 논문에서는 Contrasformer라는 새로운 대조적 뇌 네트워크 Transformer를 제안합니다. 이는 뇌 네트워크의 특성을 활용하여 Transformer 기반의 모델을 극대화합니다.

- **Technical Details**: Contrasformer는 두 개의 스트림 주의(attention) 메커니즘을 통해 대조 그래프를 생성하여 아형에 따른 분포 변화(SPS noise)를 해결합니다. 이 과정에서 노드의 정체성을 강조하는 identity embedding을 활용하며, 그룹 간 일관성을 보장하기 위한 보조 손실(loss)을 도입합니다.

- **Performance Highlights**: Contrasformer는 4개의 기능적 뇌 네트워크 데이터 세트에서 평가되었으며, 13개의 최신 방법보다 최대 10.8% 향상된 정확도를 달성했습니다. 또한, 사례 연구를 통해 신경 과학과 일치하는 명확하고 해석 가능한 패턴을 추출했습니다.



### Early Detection of Coronary Heart Disease Using Hybrid Quantum Machine Learning Approach (https://arxiv.org/abs/2409.10932)
- **What's New**: 이 연구에서는 양자 기계 학습(Quantum Machine Learning, QML) 분류기를 기반으로 한 하이브리드 기계 학습 모델을 활용하여 관상동맥 심장병(Coronary Heart Disease, CHD)의 위험 예측 방법을 제안합니다. 이 방법은 다양한 차원의 의료 데이터에 효과적으로 대응할 수 있는 능력 덕분에 강력한 성능을 발휘합니다.

- **Technical Details**: 제안된 접근법은 다단계 추론 프레임워크에서 양자 알고리즘과 고전적인 기계 학습 알고리즘을 융합하여 복잡한 문제를 해결하는 데 도움을 줍니다. 연구는 Raspberry Pi 5의 GPU 플랫폼에서 개발되어 CHD 환자와 건강한 대조군의 임상 및 이미징 데이터를 통합한 광범위한 데이터셋에서 테스트되었습니다.

- **Performance Highlights**: 제안된 하이브리드 QML 모델은 고전적인 기계 학습 모델과 비교하여 정확도(accuracy), 민감도(sensitivity), F1 점수(F1 score), 및 특이도(specificity)에서 현저히 높은 성능을 기록하였습니다. 이 연구는 심장 질환 및 사망률 증가에 대처하기 위한 조기 발견의 필요성을 강조합니다.



### KALE: An Artwork Image Captioning System Augmented with Heterogeneous Graph (https://arxiv.org/abs/2409.10921)
Comments:
          Accepted at IJCAI 2024

- **What's New**: KALE(Knowledge-Augmented vision-Language model for artwork Elaborations)는 기존의 vision-language 모델을 강화하여 미술 작품의 메타데이터를 통합함으로써 작품의 의미에 대한 심층 해석을 가능하게 하는 새로운 접근 방식을 제시합니다.

- **Technical Details**: KALE는 두 가지 방식으로 메타데이터를 통합합니다. 첫 번째는 텍스트 입력으로 직접 포함하는 것이고, 두 번째는 다중 모드 이질적 지식 그래프(multimodal heterogeneous knowledge graph)를 통해서입니다. 새로운 cross-modal alignment loss를 도입하여 이미지와 그에 해당하는 메타데이터 간의 유사성을 극대화합니다. 이 시스템은 선행 학습된 비전-언어 모델을 미술 영역으로 확대하여 이미지 설명을 생성합니다.

- **Performance Highlights**: KALE는 여러 미술 작품 데이터셋에서 CIDEr로 평가했을 때 현존하는 최첨단 기술을 능가하는 성능을 보여줍니다. 실험 결과, KALE는 기존의 미술 작품 이미지 캡셔닝 모델을 상당히 초월하는 성능을 입증하며, 미술 작품의 내러티브에 대한 이해를 더욱 촉진할 수 있음을 나타냅니다.



### GenCRF: Generative Clustering and Reformulation Framework for Enhanced Intent-Driven Information Retrieva (https://arxiv.org/abs/2409.10909)
- **What's New**: GenCRF(Generative Clustering and Reformulation Framework)가 정보 검색 분야에서 사용자 쿼리의 다양한 의도를 포착하기 위해 처음으로 도입되었습니다. 이 프레임워크는 Large Language Models(LLMs)를 활용하여 다수의 맞춤형 프롬프트를 사용해 여러 쿼리를 생성하고 클러스터링하여 쿼리 성공률을 향상시키는 목표를 가지고 있습니다.

- **Technical Details**: GenCRF는 LLM을 통해 초기 쿼리에서 파생된 다양한 쿼리를 생성하고, 이들을 동적으로 클러스터링하여 정보 중복을 최소화합니다. 쿼리 통합을 위해 유사성 기반 동적 가중치 및 점수 기반 동적 가중치를 포함한 여러 가중 집계 전략을 적용하며, Query Evaluation Rewarding Model(QERM)을 통해 쿼리 개선 프로세스를 피드백 루프를 통해 정교화합니다.

- **Performance Highlights**: BEIR 데이터세트에서 실시한 실험을 통해 GenCRF는 기존 쿼리 재구성 기술 대비 최대 12% 향상된 nDCG@10 성능을 기록하며, 다양한 도메인과 쿼리 유형에 대해 지속적으로 우수한 성능을 입증하였습니다.



### WaterQualityNeT: Prediction of Seasonal Water Quality of Nepal Using Hybrid Deep Learning Models (https://arxiv.org/abs/2409.10898)
- **What's New**: 이번 연구에서는 네팔의 수질 (water quality) 모니터링을 위한 하이브리드 딥러닝 모델을 제안합니다. 이 모델은 계절별 수질 예측을 위해 작은 데이터셋을 사용하여 여러 수질 매개변수를 활용합니다.

- **Technical Details**: 제안된 모델은 시간적 (temporal) 및 공간적 (spatial) 패턴을 활용하기 위해 합성곱 신경망 (CNN)과 순환 신경망 (RNN)을 통합합니다. 이러한 네트워크 구조는 수질 예측에서 전통적인 방법보다 더 나은 성능을 보여줍니다.

- **Performance Highlights**: 모델은 WQI (Water Quality Index) 매개변수를 사용하여 사람들을 좋은, 나쁜, 평균 그룹으로 분류하는 데 92%의 정확도를 보였습니다. 회귀 분석을 통해 WQI 값을 예측할 때 R2 점수는 0.97, 평균 제곱근 오차 (root mean square error)는 2.87로 나타났습니다. 또한 회귀 및 분류 접근법을 모두 사용하는 다기능 애플리케이션이 개발되었습니다.



### Shaking the Fake: Detecting Deepfake Videos in Real Time via Active Probes (https://arxiv.org/abs/2409.10889)
- **What's New**: SFake는 스마트폰에서 실시간으로 딥페이크를 탐지하는 새로운 방법을 제안하며, 물리적 간섭을 이용한 능동적인 특징 검색을 통해 탐지 정확도를 향상시킵니다.

- **Technical Details**: SFake는 스마트폰에 진동을 유도하는 프로브를 보내며, 촬영된 비디오의 얼굴 영역과 프로브 패턴의 일관성을 통해 딥페이크 여부를 판단합니다. 1920x1080 픽셀 이상의 해상과 2배 줌 이상의 카메라를 요구합니다.

- **Performance Highlights**: SFake의 탐지 정확도는 95.2% 이상이며, 처리 속도는 5초 이하, 메모리 소비는 450MB 미만으로 다른 6개 방법보다 우수한 성능을 보였습니다.



### SIFToM: Robust Spoken Instruction Following through Theory of Mind (https://arxiv.org/abs/2409.10849)
Comments:
          7 pages, 4 figures

- **What's New**: 이번 연구에서는 인지과학에서의 이론적 접근성을 바탕으로 한 새로운 모델, Speech Instruction Following through Theory of Mind (SIFToM)을 제시합니다. 이 모델은 다양한 음성 조건에서 로봇이 인간의 지시를 효과적으로 따를 수 있는 방법을 제공합니다.

- **Technical Details**: SIFToM은 인간의 목표와 공동 계획을 추론하여 언어 이해를 향상시키는 mixed-observability Markov decision process (MOMDP) 구조를 사용합니다. 이 구조는 로봇이 음성 지시에서 전체 팀 목표와 하위 목표를 유추해야 함을 나타냅니다.

- **Performance Highlights**: SIFToM 모델은 VirtualHome 2에서 실행된 실험을 통해 최신 음성 및 언어 모델을 능가하는 성능을 보였으며, 더욱이 인간 수준의 정확도에 근접했습니다. 로봇은 아침 준비 작업에서 인간 사용자를 지원하는 데 가장 짧은 시간을 기록했습니다.



### 3DFacePolicy: Speech-Driven 3D Facial Animation with Diffusion Policy (https://arxiv.org/abs/2409.10848)
- **What's New**: 본 논문은 3DFacePolicy라는 새로운 모델을 제안하여 오디오 기반의 3D 얼굴 애니메이션 생성에서 감정 표현과 사실감을 개선하고, 다양한 얼굴 움직임을 생성하는 데 중점을 두고 있습니다.

- **Technical Details**: 3DFacePolicy는 diffusion policy 모델을 활용하여 3D 얼굴 애니메이션을 예측하는 방법론입니다. 이 모델은 오디오 입력과 3D vertex 상태를 관찰하여 vertex trajectory를 예측하고, 실제 인간의 감정을 모방하여 자연스러운 흐름을 유지합니다. 모델은 복잡한 데이터 분포를 처리하기 위한 시퀀스 샘플러를 사용하여 동작을 매끄럽게 생성합니다.

- **Performance Highlights**: 3DFacePolicy는 VOCASET 벤치마크를 기준으로 다른 최신 방법들에 비해 주목할 만한 장점을 보이며, 유연하고 동적인 얼굴 움직임을 생성하는 데 있어 효과적임을 입증하였습니다.



### PDMX: A Large-Scale Public Domain MusicXML Dataset for Symbolic Music Processing (https://arxiv.org/abs/2409.10831)
- **What's New**: PDMX는 250K 이상의 Public Domain MusicXML 파일을 수집한 대규모 오픈 소스 데이터셋으로, 기존의 상업적 라이센스 문제를 해결하고자 합니다. 이는 현재로서는 가장 큰 저작권 없는 상징 음악 데이터셋입니다.

- **Technical Details**: PDMX 데이터셋은 MuseScore 웹사이트에서 수집된 MusicXML 파일로 구성되며, 각 파일에는 장르, 태그, 설명 및 인기도와 같은 메타데이터가 포함되어 있습니다. MusicRender라는 확장을 통해 MusicXML 파일을 파싱하고, 음악의 성능 지시사항을 지원하는 기능을 제공합니다.

- **Performance Highlights**: PDMX의 사용자 평가 자료를 활용하여, 모델의 데이터 품질 평가에 효과적으로 기여할 수 있으며, 다양한 하위 집합을 통해 생성 모델의 훈련 및 미세 조정에 필요한 고품질 데이터를 선택할 수 있음을 보여주었습니다.



### Challenging Fairness: A Comprehensive Exploration of Bias in LLM-Based Recommendations (https://arxiv.org/abs/2409.10825)
- **What's New**: 본 연구는 Large Language Model (LLM) 기반 추천 시스템에서 발생하는 편향(bias)과 그 영향을 심층적으로 분석합니다. 특히, 음악, 노래, 도서 추천의 맥락에서 다양한 인구 통계 및 문화 그룹을 살펴봅니다.

- **Technical Details**: LLM 기반 추천 시스템은 사용자의 행동 및 콘텐츠를 깊이 분석하여 추천의 질을 높입니다. 하지만 이 시스템은 종종 훈련 데이터의 불균형으로 인해 주류 콘텐츠를 선호하고 비주류 옵션을 소외시키는 경향이 있습니다. 다양한 LLM 모델을 사용하여 bias의 영향을 평가합니다.

- **Performance Highlights**: 연구 결과, bias는 시스템에 깊게 내재되어 있으며, prompt engineering과 같은 간단한 개입만으로도 추천 결과에서 bias를 크게 줄일 수 있습니다. 교차 정체성(intersecting identities) 및 사회경제적 지위(socioeconomic status)와 같은 맥락 정보가 그러한 편향을 더욱 악화시킬 수 있다는 점도 강조됩니다.



### PReLU: Yet Another Single-Layer Solution to the XOR Problem (https://arxiv.org/abs/2409.10821)
- **What's New**: 이번 연구는 PReLU (Parametric Rectified Linear Unit) 활성화 함수를 사용하는 단일 레이어 신경망이 XOR 문제를 해결할 수 있음을 보여줍니다. 이는 지금까지 간과되어 온 사실로, 이 연구에서는 PReLU의 사용이 어떻게 이러한 해결책을 가능하게 하는지 설명합니다.

- **Technical Details**: PReLU는 학습 가능한 매개변수 a를 포함하여 다양한 활성화 함수를 일반화할 수 있는 특징이 있습니다. 이 연구에서는 이 기능을 통해 XOR 문제를 추가적인 레이어 없이 단일 레이어에서 해결하는 방법을 다룹니다. 또한 단일 레이어 PReLU 네트워크는 100% 성공률을 달성할 수 있으며, 학습률이 다양한 범위에서 안정적으로 작동합니다.

- **Performance Highlights**: PReLU 기반의 단일 레이어 네트워크는 3개의 학습 가능한 파라미터만으로 XOR 문제를 완벽하게 해결할 수 있음을 보여줍니다. 이는 기존의 다층 퍼셉트론(MLP) 및 GCU (Growing Cosine Unit) 활성화 함수와 비교할 때 더 간단한 구조로 복잡한 문제를 해결할 수 있는 가능성을 제시합니다.



### Model Tells Itself Where to Attend: Faithfulness Meets Automatic Attention Steering (https://arxiv.org/abs/2409.10790)
Comments:
          12 pages, 4 figures

- **What's New**: 이 논문에서는 AutoPASTA라는 새로운 방법을 제안하여 LLM의 주의 집중을 강화하고 중요한 맥락 정보를 효과적으로 활용하도록 합니다. 이는 주의 점수 조정을 통해 LLM이 중요한 정보에 더 잘 집중할 수 있도록 돕습니다.

- **Technical Details**: AutoPASTA는 (1) 주요 맥락 정보를 자동으로 식별하고, (2) 이 정보를 강조하여 성능을 향상시키는 접근 방식입니다. 이 방법은 기존의 prompting 방법 대신, 문맥과 질문에서 의미론적 임베딩을 사용하여 원래 문장을 매핑함으로써 입력의 길이를 줄이고 오류 전파를 완화하는 방식입니다.

- **Performance Highlights**: 실험 결과에 따르면, AutoPASTA는 LLAMA3-70B-Instruct 모델에서 평균 8.99%의 성능 향상을 달성했으며, 다양한 작업에서의 주의 헤드 집합이 우수한 일반화 능력을 보여주었습니다.



### Are Deep Learning Models Robust to Partial Object Occlusion in Visual Recognition Tasks? (https://arxiv.org/abs/2409.10775)
- **What's New**: 본 논문에서는 부분 가림(occlusion) 조건에서의 이미지 인식 성능을 검증하기 위해 Real-world 및 인공적으로 가려진 이미지로 구성된 Image Recognition Under Occlusion (IRUO) 데이터셋을 개발하였습니다. 이 데이터셋은 실제 환경에서의 가림 현상을 다루기 위한 대규모 공공 벤치마크로 기능하며, 현대의 CNN 모델과 Vision Transformer 모델의 성능을 검토합니다.

- **Technical Details**: IRUO 데이터셋은 Occluded Video Instance Segmentation (OVIS) 데이터셋을 기반으로 하며, 88,000장의 이미지와 23개의 객체 클래스를 포함합니다. 이 연구에서는 ResNeXt와 Swin 같은 최첨단 CNN 모델과 Vision Transformer들을 비교하여, 모델의 정확도를 평가하고 가림에 대한 상대적 강인성을 분석합니다. 또한, 인간 관찰자와 모델의 인식 성능을 비교하여, 인식 정확도가 어느 정도 차이가 나는지 확인합니다.

- **Performance Highlights**: 최신 CNN 기반 모델은 이전 모델보다 가려진 이미지에 대한 인식 정확도가 개선되었으며, Vision Transformer 모델은 CNN 기반 모델보다 높은 정확도를 보이며, 인간의 정확도와 비슷한 수준을 나타냅니다. 또한, 'diffuse occlusion' 같은 특정 가림 종류는 딥러닝 모델의 정확도를 크게 저하시킬 수 있음을 발견하였습니다.



### VulnLLMEval: A Framework for Evaluating Large Language Models in Software Vulnerability Detection and Patching (https://arxiv.org/abs/2409.10756)
- **What's New**: VulnLLMEval 프레임워크를 통해 대규모 언어 모델(LLMs)의 소프트웨어 취약점 감지(SVD) 및 패칭(SVP) 성능을 평가하는 새로운 벤치마크를 제시합니다.

- **Technical Details**: 307개의 실제 취약점 데이터를 포함하며, 이는 Linux 커널에서 추출된 C 코드의 취약점 및 패치 코드를 포함합니다. 새로운 자동 데이터 수집 방법을 통해 취약 코드 및 패치 코드를 수집하였으며, 모델 평가를 위한 다양한 메트릭을 사용합니다.

- **Performance Highlights**: LLMs는 취약 코드와 패치 코드를 구별하는 데 어려움을 겪는 것으로 나타났습니다. SVP 작업에서는 코드 단순화로 인한 비사용 가능 솔루션을 생성하는 경향이 있음을 확인했습니다.



### AutoSafeCoder: A Multi-Agent Framework for Securing LLM Code Generation through Static Analysis and Fuzz Testing (https://arxiv.org/abs/2409.10737)
- **What's New**: 최근 자동 코드 생성 및 보안 소프트웨어 개발의 발전이 큰 주목을 받고 있습니다. 본 논문에서는 LLM(대형 언어 모델)을 기반으로 다중 에이전트 프레임워크인 AutoSafeCoder를 제안합니다. 이 시스템은 코드 생성, 취약점 분석, 보안 강화를 위해 협력하는 여러 에이전트를 활용하여 보안이 강화된 코드를 생성합니다.

- **Technical Details**: AutoSafeCoder는 세 가지 주요 에이전트로 구성됩니다: 1) Coding Agent (코드 생성 에이전트) - 코드 요구 사항에 따라 코드를 생성합니다. 2) Static Analyzer Agent (정적 분석 에이전트) - 코드의 취약점을 식별합니다. 3) Fuzzing Agent (퍼징 에이전트) - 동적 테스트를 수행하여 런타임 오류를 탐지합니다. 이 시스템은 GPT-4를 활용하여 코드 생성을 수행하며, 정적 및 동적 테스트가 통합된 반복적인 프로세스를 통해 보안을 강화합니다.

- **Performance Highlights**: SecurityEval 데이터셋을 사용한 실험 결과, AutoSafeCoder는 코드의 취약점을 기존 LLM 대비 13% 감소시키는 성과를 보였습니다. 이 과정에서 기능성에는 타격을 주지 않았습니다.



### Generalized Measures of Anticipation and Responsivity in Online Language Processing (https://arxiv.org/abs/2409.10728)
- **What's New**: 이 논문에서는 온라인 언어 처리에서 예측 불확실성을 측정하기 위한 고전적인 정보 이론적 방법을 일반화하였습니다. 새로운 예상 연속성의 시뮬레이션을 기반으로 하는 이 접근법은 반응적(responsive) 및 예측적(anticipatory) 측정을 정의하며, 기존의 next-symbol entropy 및 surprisal보다 더 표현력 있는 측정을 정의하는 도구를 제공합니다.

- **Technical Details**: 이 연구의 프레임워크는 대칭화된 변수에 대한 일반화된 surprisal을 도출하며, 이는 언어적 맥락의 연속에 대한 기대와 일치합니다. 이 프레임워크는 기존 정보 이론적 측정을 특별한 경우로 포함하며, sequence-level entropy와 next-symbol information value와 같은 새로운 특별한 경우를 제안합니다. 몬테 카를로 시뮬레이션을 사용하여 다양한 반응적 및 예측적 측정을 추정하며, 이 과정에서 런타임(runtime)과 변동성(variance) 간의 트레이드오프를 분석합니다.

- **Performance Highlights**: 연구 결과, (1) 문맥 확률이 인간의 cloze completion을 surprisal보다 더 잘 예측하며, (2) 정보 값이 N400을 더 잘 예측하고, (3) 본 논문에서 제안한 sequence-level entropy가 ELAN의 유일한 유의미한 예측자라는 사실이 밝혀졌습니다. 또한, (4) 다양한 반응적 측정이 자극 시작 후 여러 시간대의 ERP 진폭을 예측하는 데 차별화된 성능을 보였습니다.



### A Missing Data Imputation GAN for Character Sprite Generation (https://arxiv.org/abs/2409.10721)
Comments:
          Published in SBGames 2024

- **What's New**: 본 연구에서는 다양한 애니메이션과 포즈를 포함하는 픽셀 아트 캐릭터 스프라이트(pixel art character sprites)를 생성하고 업데이트하는 과정에서 반복적이 되는 작업을 부분적으로 자동화할 수 있는 방법을 제안합니다. 특히, 다른 세 방향에서 캐릭터 이미지가 주어질 때 특정 포즈(target pose)로의 이미지 생성을 처리하는 새로운 접근 방식을 소개합니다.

- **Technical Details**: 이 연구에서는 generative adversarial networks (GANs)를 사용하여 캐릭터의 다양한 포즈를 위한 이미지를 생성하는 모델을 제안합니다. 제안된 모델은 CollaGAN 아키텍처에 기반해 있으며, 생성기(generator) 구조와 훈련 절차에 변경을 가했습니다. 대상 포즈의 이미지를 생성하기 위해 다른 포즈의 이미지를 입력으로 사용합니다. 주어진 입력 이미지들이 많을수록 생성된 이미지의 품질이 향상된다는 것을 실험적으로 증명하였습니다. 평가 지표로는 Frechét Inception Distance (FID)와 L1 distance를 사용했습니다.

- **Performance Highlights**: 단일 생성기/구분기를 사용하는 GAN 모델이 여러 캐릭터 포즈를 대상으로 작업할 수 있음을 보였으며, 입력으로 제공되는 이미지 수가 많을수록 생성된 스프라이트의 품질이 향상됨을 보여주었습니다. 제안된 변경 사항들이 기존의 CollaGAN에 비해 보다 나은 결과를 도출하는 데 기여하였습니다.



### Self-Attention Limits Working Memory Capacity of Transformer-Based Models (https://arxiv.org/abs/2409.10715)
Comments:
          8 pages, 12 figures

- **What's New**: 최근 연구를 통해 Transformer 기반의 대규모 언어 모델(LLMs)의 작업 기억 용량에 한계가 있음을 발견했습니다. 이는 N-back 작업의 성과가 N이 증가함에 따라 유의미하게 감소한다는 점에서 인간의 기억 한계와 유사합니다. 하지만 이러한 현상의 기계적 해석이 부족합니다.

- **Technical Details**: 본 연구에서는 Transformer의 self-attention 메커니즘이 작업 기억 용량 한계의 원인일 수 있다는 가설을 세우고 이를 검증하기 위해 vanilla decoder-only transformers를 훈련시켰습니다. 실험 결과, attention score가 훈련 과정에서 점차 N-back 위치로 집계되는 현상이 관찰되었습니다. 이는 모델이 현재 위치와 N-back 위치 간의 관계에 주의하기 위한 전략을 학습하고 있다는 것을 시사합니다.

- **Performance Highlights**: 모델의 N-back 작업 수행 능력은 N이 증가함에 따라 감소하였으며, 각 위치의 예측 정확도는 attention score와 긍정적 상관관계를 보였습니다. 특히, attention score 행렬의 총 엔트로피는 N이 증가함에 따라 증가했으며, 이는 attention score의 분산이 N-back 작업에서의 용량 한계의 원인일 수 있음을 나타냅니다.



### Self-supervised Speech Models for Word-Level Stuttered Speech Detection (https://arxiv.org/abs/2409.10704)
Comments:
          Accepted by IEEE SLT 2024

- **What's New**: 본 연구에서는 단어 수준의 더듬이(스퍼트) 탐지를 위한 모델을 제안하며, 이 과정에서 자가 지도(Self-Supervised) 음성 모델을 활용하였습니다. 이는 더듬이 장애를 가진 환자들의 자동화된 스크리닝을 가능하게 합니다.

- **Technical Details**: 연구팀은 단어 수준의 더듬이 탐지 모델을 위해 WavLM(자기지도 학습 기반 음성 모델)을 이용하고, LibriSpeech 데이터를 합성 불유창(Disfluency) 증강법으로 사전 훈련한 후 SEP-28K 데이터셋으로 미세 조정(fine-tune) 하였습니다. 모델 구조는 계층적 합성곱 인터페이스(Hierarchical Convolution Interface)를 활용하여 여러 층에서 정보를 집계합니다.

- **Performance Highlights**: 제안된 모델은 단어 수준의 더듬이 탐지에서 이전 연구를 크게 초과하는 성능을 보였습니다. 또한, 더듬이 장애 탐지에 있어 자가 지도 음성 모델의 효과에 대한 포괄적인 실험 분석을 수행하였습니다.



### Model-in-the-Loop (MILO): Accelerating Multimodal AI Data Annotation with LLMs (https://arxiv.org/abs/2409.10702)
- **What's New**: AI 훈련 데이터에 대한 수요 증가로 인해 데이터 주석(annotation) 산업이 글로벌 산업으로 성장하고 있습니다. 기존의 인간 주석자에 의존한 접근 방식은 시간 소모가 크고 노동 집약적이며 일관된 품질을 보장하기 어렵습니다. 이를 해결하기 위해 제안된 MILO(Model-in-the-Loop) 프레임워크는 AI/ML 모델을 주석 과정에 통합합니다.

- **Technical Details**: MILO 프레임워크는 전문 인간 주석자와 대규모 언어 모델(LLM)의 강점을 활용하는 협업 패러다임을 도입합니다. LLM을 사전 주석(pre-annotation) 및 실시간 어시스턴트(real-time assistants)로 사용하고, 주석자 응답에 대한 심사위원(judges) 역할을 수행함으로써, 인간 주석자와 LLM 간의 효과적인 상호작용 패턴을 가능하게 합니다. 세 가지 실증 연구(empirical studies)를 통해 다중 모달(multimodal) 데이터 주석에서 MILO의 효과를 입증하였습니다.

- **Performance Highlights**: MILO는 처리 시간을 줄이고(data handling time), 데이터 품질을 향상시키며(quality), 주석자 경험을 개선하는 성과를 보여주었습니다. 또한 유연한 평가(quality rubrics) 및 열린 주석(open-ended annotations)에 대한 세분화된 피드백(fine-grained feedback)을 제공합니다. MILO 프레임워크는 AI/ML 개발을 가속화하고, 인간 주석에 대한 의존도를 줄이며, 인간과 기계 가치 간의 더 나은 정렬(alignment)을 촉진하는 데 기여할 수 있습니다.



### Playground v3: Improving Text-to-Image Alignment with Deep-Fusion Large Language Models (https://arxiv.org/abs/2409.10695)
- **What's New**: Playground v3 (PGv3)는 최신 텍스트-이미지 생성 모델로, 기존의 모델들과는 달리 Large Language Models (LLMs)를 기반으로 하여 픽셀 단위의 세밀한 이미지 생성 능력을 보여줍니다. 주목할 점은 PGv3가 독창적인 구조를 도입하여 이미지 캡셔닝 성능을 향상시키기 위한 자체 개발된 captioner를 포함하고 있으며, 새로운 벤치마크 CapsBench 또한 소개한 것입니다.

- **Technical Details**: PGv3는 Latent Diffusion Model (LDM)을 기반으로 하며, 기존의 텍스트 인코더를 사용하지 않고 Llama3-8B LLM을 통합하여 텍스트 조건을 제공합니다. 모델 아키텍처는 Deep-Fusion 구조를 채택하여 LLM의 지식을 활용하고, 복잡한 이미지-텍스트 정렬 및 추론을 지원합니다. 이미지와 텍스트의 결합 관리에서 joint attention 구조를 사용해 계산 비용을 줄입니다.

- **Performance Highlights**: PGv3는 이미지 캡셔닝, 복잡한 추론 및 정확한 텍스트 렌더링에서 뛰어난 성능을 보이며, 실제 인간 디자이너를 초월하는 그래픽 디자인 능력을 갖추고 있습니다. RGB 색상 제어 및 강력한 다국어 이해도 제공하며, CapsBench 벤치마크에서의 성과는 PGv3의 우수성을 잘 보여줍니다.



### Encoding Reusable Multi-Robot Planning Strategies as Abstract Hypergraphs (https://arxiv.org/abs/2409.10692)
- **What's New**: 이 논문에서는 Multi-Robot Task Planning (MR-TP) 문제를 위한 새로운 접근법을 제안합니다. 이 접근법은 Decomposable State Space Hypergraph (DaSH)와 learning-by-abstraction 기술을 결합하여 다중 로봇 계획을 개선하고자 합니다.

- **Technical Details**: 이 연구는 MR-TP 문제를 해결하기 위해 hypergraph 기반의 접근 방식을 사용합니다. 개별 계획 경험에서 일반화 가능한 계획 전략을 자동으로 추출 및 재사용하는 방법을 통해, 하이퍼그래프를 사용한 MR-TP의 효과성을 향상시키려고 합니다.

- **Performance Highlights**: 이 접근법은 기존의 단일 로봇 계획에서 효과적이었던 전략을 MR-TP에 적용하여, 로봇의 수나 작업 복잡성에 따라 초래되는 대규모 검색 공간 문제를 완화할 수 있는 가능성을 보여줍니다.



### MotIF: Motion Instruction Fine-tuning (https://arxiv.org/abs/2409.10683)
- **What's New**: 본 논문에서는 로봇의 동작 이해를 위한 새로운 접근법인 Motion Instruction Fine-tuning (MotIF)을 제안합니다. MotIF는 로봇의 동작을 평가할 때 전체 궤적에 주목하여, vision-language models (VLMs) 의 성공 판별력을 향상시키는데 초점을 맞추고 있습니다.

- **Technical Details**: MotIF는 로봇의 궤적을 최종 이미지 위에 겹쳐 시각화를 통해 궤적 기반의 시각적 동작 표현을 만듭니다. 이를 통해 로봇 동작과 환경의 의미적 연결을 포착하여 VLM의 성능을 향상시킵니다. 또한, MotIF-1K 데이터세트(653개의 인간 예시 및 369개의 로봇 예시 포함)는 13개 작업 범주에서 다양한 동작을 포함하고 있으며, VLM을 세밀하게 조정하는 데 유용한 자료를 제공합니다.

- **Performance Highlights**: MotIF는 최신 VLMs에 비해 2배 이상의 정밀도 향상과 56.1%의 재현율 향상을 보여주었습니다. MotIF는 다양한 작업, 동작 및 환경에 대한 일반화 능력을 갖추고 있으며, 로봇 계획의 개선 및 종료, 동작에 대한 점수 매기기 등의 실제 응용을 시장 가치에 적합하게 보여주고 있습니다.



### Multi-agent Path Finding in Continuous Environmen (https://arxiv.org/abs/2409.10680)
Comments:
          The 36th IEEE International Conference on Tools with Artificial Intelligence (ICTAI). 2024, In press

- **What's New**: 본 논문에서는 연속 환경에서의 다중 에이전트 경로 찾기 문제(CE-MAPF)의 변형을 다루고 있으며, 에이전트들이 부드러운 곡선을 따라 이동하는 방안을 제안합니다. 본 연구에서는 연속적 충돌 기반 탐색(CE-CBS) 알고리즘을 새롭게 도입했습니다.

- **Technical Details**: CE-CBS는 높은 수준의 탐색 프레임워크를 위한 충돌 기반 탐색(CBS)과 낮은 수준의 경로 계획을 위한 RRT* 알고리즘을 결합한 것입니다. 이 알고리즘은 연속 시간 속에서 동작하며, 에이전트의 경로는 B-스플라인 곡선으로 매끄럽게 조정됩니다.

- **Performance Highlights**: 실험 결과, CE-CBS는 MAPF의 연속적 측면을 고려한 다른 알고리즘들과 비교했을 때 경쟁력 있는 성능을 보였습니다.



### Disentangling Uncertainty for Safe Social Navigation using Deep Reinforcement Learning (https://arxiv.org/abs/2409.10655)
Comments:
          Submitted to the IEEE for possible publication, 8 pages, 6 figures

- **What's New**: 이 연구는 심층 강화 학습(Deep Reinforcement Learning, DRL) 기반 내비게이션 프레임워크에 불확실성 추정(uncertainty estimation)을 통합한 새로운 접근 방식을 도입합니다. 특히, 관측 의존적 분산(Observation-Dependent Variance, ODV)과 드롭아웃(dropout)을 Proximal Policy Optimization (PPO) 알고리즘에 통합하여 DRL의 의사결정 불확실성 문제를 해결하고자 합니다.

- **Technical Details**: 이 연구는 aleatoric, epistemic, 그리고 predictive uncertainty라는 세 가지 타입의 불확실성을 추정하고 구분합니다. ODV 및 MC-Dropout 방법을 통해 정책의 불확실성을 정량화하고, 다양한 perturbations에 대한 비교를 통해 정책의 불확실성을 개선합니다. DRL 정책을 사용할 때, robot은 불확실한 상호작용 상황에서 보수적인 충돌 회피 행동으로 변경해야 합니다.

- **Performance Highlights**: ODV-PPO 알고리즘은 더 빠르게 수렴하고, 일반화 성능이 향상되며, aleatoric과 epistemic 불확실성을 잘 분리합니다. MC-Dropout 접근법은 perturbation에 더 민감하며 불확실성의 종류와 perturbation의 종류 간의 상관관계를 잘 나타냅니다. 제안된 안전한 행동 선택 방식을 통해 로봇은 perturbated environment에서 충돌을 줄이며 내비게이션 할 수 있습니다.



### Exploring Fine-tuned Generative Models for Keyphrase Selection: A Case Study for Russian (https://arxiv.org/abs/2409.10640)
- **What's New**: 이 연구는 러시아 학술 텍스트 내의 키프레이즈 선택(task of keyphrase selection)을 위해 세밀하게 조정된 생성적 변환 모델(generative transformer-based models)의 적용 방안을 탐구하였습니다. 이를 통해 다양한 분야에서의 성과를 평가하였으며, 특히 mBART 모델이 러시아어 키프레이즈 추출 기준선(baselines) 대비 성과 향상을 보여주었습니다.

- **Technical Details**: 연구에서는 ruT5, ruGPT, mT5, mBART의 네 가지 생성적 모델을 사용했습니다. 이들 모델은 한국 수학 및 컴퓨터 과학, 역사, 의학, 언어학 분야의 러시아어 과학 초록(texts of Russian scientific abstracts)을 분석하는 데 적용되었습니다. 키프레이즈 선택 영역에서 딥 신경망 모델(deep neural models)을 활용하여, 전통적인 비지도 학습 방식의 한계를 극복하고 싶은 목표를 가지고 있습니다.

- **Performance Highlights**: mBART 모델을 사용한 결과, 키프레이즈 선택의 in-domain 성과가 BERTScore에서 최대 4.9%, ROUGE-1에서 9.0%, F1-score에서 12.2% 향상되었으며, cross-domain에서도 몇 가지 경우에 대해 기준 성과를 초과하는 긍정적인 결과를 보였습니다. 이는 러시아어 키프레이즈 선택 기술의 추가 탐색 및 개선 가능성을 부각시킵니다.



### Kolmogorov-Arnold Transformer (https://arxiv.org/abs/2409.10594)
Comments:
          Code: this https URL

- **What's New**: Kolmogorov-Arnold Transformer (KAT)는 전통적인 MLP 레이어를 Kolmogorov-Arnold Network (KAN) 레이어로 대체하여 모델의 표현력과 성능을 향상시키는 새로운 아키텍처입니다. KAN을 Transformers에 통합하는 과정에서 발견된 3가지 주요 도전 과제를 해결하기 위해 혁신적인 접근 방식을 제안합니다.

- **Technical Details**: KAT는 KAN의 기본 함수를 B-spline 대신 유리 함수로 변경하고, 그룹 KAN을 통해 활성화 가중치를 공유하여 계산 부담을 줄이며, 분산 보존 초기화를 사용하여 층 간의 활성화 분산을 유지합니다. 이러한 디자인은 KAT가 효과적으로 확장될 수 있도록 하고, 전통적인 MLP 기반 Transformers를 능가하는 성능을 제공합니다.

- **Performance Highlights**: KAT는 이미지 인식, 객체 탐지, 의미 분할을 포함한 다양한 비전 작업에서 평가됐으며, ImageNet-1K에서 82.3%의 정확도를 달성하여 동일한 크기의 ViT 모델보다 3.1% 향상된 성능을 보였습니다. 또한, ViT의 사전 훈련된 가중치로 초기화했을 때 성능이 82.7%로 더욱 개선되었습니다.



### CSKV: Training-Efficient Channel Shrinking for KV Cache in Long-Context Scenarios (https://arxiv.org/abs/2409.10593)
- **What's New**: 이 논문에서는 KV 캐시의 메모리 오버헤드를 줄이기 위한 새로운 기법인 CSKV(채널 축소 기법)를 소개합니다. 기존의 KV 캐시 압축 방법이 가지고 있는 부족한 성능과 높은 훈련 비용 문제를 해결하려는 접근법입니다.

- **Technical Details**: CSKV는 (1) 키와 값 레이어의 특이값 분포를 분석하여 채널 차원에서 큰 중복성과 압축 가능성을 발견합니다. (2) 이 구조에서는 윈도우 기반의 전정밀 KV 캐시와 저정밀 압축 KV 캐시를 결합한 이중 분기 KV 캐시를 도입합니다. (3) 전체 LLM 재훈련 없이, 층별 재구성 손실을 최소화하여 훈련 비용을 절감합니다.

- **Performance Highlights**: CSKV는 KV 캐시의 메모리 오버헤드를 80%까지 줄이면서 모델의 장기 컨텍스트 능력을 유지할 수 있습니다. 또한 4비트 양자화와 결합하여 최대 95%의 압축 비율을 달성하는 향상된 성능을 보여줍니다.



### Offline Reinforcement Learning for Learning to Dispatch for Job Shop Scheduling (https://arxiv.org/abs/2409.10589)
Comments:
          9 pages, 3 figures, 2 tables

- **What's New**: 본 논문에서는 Job Shop Scheduling Problem (JSSP)을 위한 Offline Reinforcement Learning (RL) 방법론인 Offline-LD를 제안합니다. 이 새로운 접근법은 기존의 온라인 RL 방식의 한계를 극복합니다.

- **Technical Details**: Offline-LD는 maskable action spaces를 위한 두 가지 CQL 기반 Q-learning 방법(mQRDQN, discrete mSAC)을 조정하고, discrete SAC를 위한 새로운 entropy bonus 수정을 도입하며, 사전처리를 통한 보상 정규화를 활용합니다.

- **Performance Highlights**: 실험 결과, Offline-LD는 생성된 인스턴스와 벤치마크 인스턴스 모두에서 온라인 RL보다 우수한 성능을 나타내며, '노이즈'가 포함된 데이터셋을 활용해 더 다양한 훈련 세트를 만드는 것이 바람직함을 보여주었습니다.



### Opponent Shaping for Antibody Developmen (https://arxiv.org/abs/2409.10588)
Comments:
          Preprint

- **What's New**: 본 연구는 전통적인 항바이러스 치료의 한계를 극복하기 위해, 현재의 바이러스 변종 뿐만 아니라 미래의 변종을 예측하여 대응할 수 있는 항체 설계에 중점을 두고 있습니다. 이를 통해 'shape'와 같은 새로운 개념의 최적화된 항체가 제안됩니다.

- **Technical Details**: 항체와 바이러스 사이의 상호작용을 두 플레이어의 제로섬 게임으로 모델링하고, 이를 통해 항체 최적화를 위한 알고리즘을 개발했습니다. 이 과정에서 Absolut! 프레임워크를 적용하여 단백질 간의 상호작용을 시뮬레이션하며, 'shaper' 항체가 진화적 압박을 매개하여 바이러스의 발전 경로를 조정하도록 합니다.

- **Performance Highlights**: shaper 항체는 단기적 효능을 최대화하는 전통적인 방식의 항체보다 바이러스 변종의 진화 경로에 긍정적인 영향을 미치며, 항체 결합에서의 바이러스의 탈출 능력을 최소화하는 효과를 입증했습니다.



### Motion Forecasting via Model-Based Risk Minimization (https://arxiv.org/abs/2409.10585)
Comments:
          6 pages, 2 figures, to be published in IEEE International Conference on Robotics & Automation (2025)

- **What's New**: 이 논문에서는 자율주행 차량의 안전하고 효율적인 경로 계획을 위해 주변 차량의 미래 궤적을 예측하는 데 새로운 샘플링 방법을 제안합니다. 기존의 예측 방법에서 발생할 수 있는 성능 저하를 해결하기 위해, 여러 모델의 예측을 기반으로 한 최적의 궤적 생성 방법을 제시하고 있습니다.

- **Technical Details**: 우리의 접근법은 Neural Networks의 앙상블을 사용하여 최적의 궤적을 생성하는 방법으로, 이것을 리스크 최소화 문제로 프레임화하여 진행합니다. 이 과정에서 PGP, LAformer, LaPred와 같은 최첨단 모델을 기본 학습 모델로 사용하여, 궤적 예측의 다양성과 정확성을 극대화합니다.

- **Performance Highlights**: nuScenes 예측 데이터 세트를 사용한 실험에서, 제안하는 방법이 최신 기술들을 초월하여 우수한 성과를 발휘하고, 예측 정확성을 크게 향상시켰음을 보여줍니다. 또한, 우리 방법은 nuScenes 리더보드에서 상위 순위를 기록하는 성과를 달성했습니다.



### Manifold-Constrained Nucleus-Level Denoising Diffusion Model for Structure-Based Drug Design (https://arxiv.org/abs/2409.10584)
- **What's New**: 이번 연구에서는 구조 기반 약물 설계(Structure-based drug design)를 위한 새로운 모델인 NucleusDiff를 제안합니다. 이 모델은 원자 간의 최소 거리를 유지하는 물리적 제약을 통합하여, 분리 위반(separation violation)을 방지합니다.

- **Technical Details**: NucleusDiff는 원자 핵(atomic nuclei)과 그 주위의 전자 구름(electron clouds) 간의 상호작용을 모델링하며, manifold를 활용하여 원자 간 거리 제약을 강화합니다. 이를 통해 분리 위반을 줄이고, 분자 상호작용에서의 매력적인 힘(attractive forces)과 반발력(repulsive forces)의 원리를 유지합니다.

- **Performance Highlights**: NucleusDiff는 100K 단백질-리간드 결합 복합체(CrossDocked2020)에서 22.16%의 결합 친화도(binding affinity) 향상과 100.00%의 분리 위반 감소를 나타냅니다. COVID-19 치료제 타겟에 대한 연구에서도 21.37%의 결합 친화도 향상과 66.67%의 분리 위반 감소를 기록했습니다.



### Reinforcement Learning with Quasi-Hyperbolic Discounting (https://arxiv.org/abs/2409.10583)
- **What's New**: 이 논문에서는 Quasi-Hyperbolic (QH) discounting이라는 새로운 프레임워크를 도입하여 강화 학습(Reinforcement Learning)에서 인간의 즉각적인 보상 선호를 모델링하는 방법을 제안합니다. 기존의 지수 할인(exponential discounting) 및 평균 보상(average reward) 방식의 한계를 극복하고, Markov Perfect Equilibrium (MPE)을 활용하는 정책을 제안합니다.

- **Technical Details**: QH discounting은 복잡한 하이퍼볼릭 할인(hyperbolic discounting)과 비교하여 더 단순하고 수학적으로 다루기 쉬운 대안입니다. 이 연구에서는 QH discounting 기반의 정책이 최적(opand, optimal)으로 최종 도달점에 접근할 수 있도록 하는 방법을 탐색합니다. 이를 위해 두 가지 속도의 분석(two-timescale analysis)을 활용하여 알고리즘의 수렴성을 검증합니다.

- **Performance Highlights**: 논문에서는 스토캐스틱 수요(stochastic demands)를 가진 표준 재고 시스템(inventory system)에 대한 수치적 검증을 통해 제안된 알고리즘이 MPE에 수렴함을 증명합니다. 이러한 결과는 강화 학습의 실제 응용 가능성을 크게 향상시키는 데 기여합니다.



### WaveMixSR-V2: Enhancing Super-resolution with Higher Efficiency (https://arxiv.org/abs/2409.10582)
Comments:
          10 pages. arXiv admin note: text overlap with arXiv:2307.00430

- **What's New**: WaveMixSR의 향상된 버전인 WaveMixSR-V2가 새로운 다단계 설계를 통해 4×4 초해상도(Super-Resolution) 작업을 개선하고 있습니다. 또한 기존의 전치 합성곱 층을 Pixel Shuffle 연산으로 대체하여 자원 효율성을 높였습니다.

- **Technical Details**: WaveMixSR-V2는 2차원 이산 웨이브렛 변환(2D-DWT)을 이용한 스페셜 토큰 믹싱으로, 다단계 설계를 통해 해상도를 점진적으로 doubling 하여 더욱 정교한 디테일 복원을 가능케 합니다. Pixel Shuffle은 전치 합성곱의 매개변수 수를 줄이고, 계산 비용을 감소시킵니다.

- **Performance Highlights**: WaveMixSR-V2는 BSD100 데이터셋에서 이전 SOTA보다 50% 이하의 매개변수로 성능을 달성하였으며, 자원 소모가 적고, 지연 시간과 처리량이 향상되었습니다.



### Veridical Data Science for Medical Foundation Models (https://arxiv.org/abs/2409.10580)
- **What's New**: 이 논문은 최신의 Foundation Models (FMs), 특히 대형 언어 모델(Large Language Models, LLMs)의 발전이 데이터 과학, 특히 의료 분야에 끼친 영향을 다루고 있습니다. 의료 데이터 과학의 표준 워크플로우가 FMs을 통해 어떻게 변화했는지를 설명하며, 새로운 Foundation Model Lifecycle (FMLC)의 개념을 제안합니다.

- **Technical Details**: FMLC는 고유의 상류(upstream) 및 하류(downstream) 프로세스를 포함하고 있으며, 컴퓨팅 자원, 모델 및 데이터 접근, 의사 결정 권한이 다양한 이해관계자들 간에 분산되어 있습니다. FMs은 기본적으로 통계적 모델로, Veridical Data Science (VDS)의 원칙을 도전하며, 예를 들어 예측 가능성(predictability), 계산 가능성(computability), 안정성(stability) 등을 새롭게 조명합니다.

- **Performance Highlights**: 의료 FMLC의 원칙이 VDS의 기본 원칙과 어떻게 일치하지 않는지를 비판적으로 분석하고, 컴퓨터 효율성과 접근성을 고려한 새로운 FMLC의 수립을 위한 권장 사항을 제시합니다. 이는 투명하고 과학적으로 재현 가능한 데이터 과학 관행을 위한 강력한 통계 분석을 촉진하는 데 기여할 것입니다.



### Recent advances in deep learning and language models for studying the microbiom (https://arxiv.org/abs/2409.10579)
- **What's New**: 이번 논문에서는 대규모 언어 모델(LLMs)이 미생물군집 및 메타게놈(data) 연구에 미치는 영향을 조사합니다. 특히, 미생물 단백질 및 유전체 시퀀스가 자연어와 유사한 방식으로 ‘생명의 언어’를 형성한다는 점을 강조합니다. 이를 통해 LLMs가 복잡한 미생물 생태계에서 유용한 통찰력을 추출하는 데 어떻게 응용될 수 있는지에 대해 리뷰합니다.

- **Technical Details**: 미생물 메타게놈 연구를 위해, 단백질 언어 모델(protein language models) 및 DNA 언어 모델(DNA language models) 두 가지 주요 언어 모델 유형을 논의합니다. 이들 모델은 단백질 및 유전체 시퀀스의 복잡한 종속 구조를 캡처하기 위해 트랜스포머(transformer) 아키텍처를 기반으로 하며, 입력 시퀀스를 인코딩하기 위해 자체 주의(attention) 메커니즘을 활용합니다.

- **Performance Highlights**: LLMs에 기반한 언어 모델을 통해 생성된 단백질은 자연 단백질과 유사한 특성을 가지며, 잠재적으로 새로운 단백질 생성 및 기능 예측에서 우수한 성능을 보입니다. 예를 들어 ProGen 및 ProtGPT2 모델이 사용되었으며, 이들은 수백만 개의 단백질 시퀀스를 학습하여 인위적 단백질이 자연 단백질의 기능적 특성을 가질 수 있도록 생성합니다.



### GLEAN: Generative Learning for Eliminating Adversarial Nois (https://arxiv.org/abs/2409.10578)
- **What's New**: 디지털 아트 커뮤니티에서 DALL-E와 Stable Diffusion과 같은 강력한 diffusion 모델로 인해 스타일 모방 공격(style mimicry attacks)이 증가하고 있습니다. 이를 방지하기 위해 Glaze라는 도구가 제안되었고, 본 논문에서는 GLEAN이라는 새로운 접근 방식을 제안합니다.

- **Technical Details**: GLEAN은 I2I (Image-to-Image) generative networks를 사용하여 Glazed 이미지에서 perturbations를 제거하는 방법입니다. 이 방법은 Glaze의 결과물에 대한 스타일 모방 공격의 성능을 평가합니다.

- **Performance Highlights**: GLEAN은 Glaze의 한계를 부각시키고 향후 개발을 촉진할 수 있도록 도와줍니다. Glaze는 스타일 모방 공격을 예방하는 데 중요한 성과를 보였지만, 일부 품질 저하나 인식할 수 없는 노이즈 같은 아티팩트(artifacts)가 발생합니다.



### A Tie-breaking based Local Search Algorithm for Stable Matching Problems (https://arxiv.org/abs/2409.10575)
Comments:
          Submitted to Journal of Heuristics

- **What's New**: 본 논문은 불완전한 선호 목록과 동점이 있는 안정적인 결혼 문제(SMTI)와 병원/거주자 문제(HRT)에 대해 최대 크기의 약한 안정적 매칭을 달성하기 위한 동점 해소 기반 지역 검색 알고리즘(TBLS)을 소개합니다.

- **Technical Details**: TBLS는 초기 단계에서 모든 동점을 무작위로 해결하고, 이후에는 선호 순위와 현재 안정적 매칭에 따라 동점 해소 전략을 점진적으로 개선합니다. 여기에 TBLS-E라는 공정성 중심의 변형을 추가로 제안하여 SMTI 문제를 해결할 때 공정성을 강조합니다.

- **Performance Highlights**: TBLS는 10개의 다른 근사 및 지역 검색 알고리즘과 비교하여 가장 큰 매칭 크기를 달성하며, TBLS-E는 성별 평등 비용이 가장 낮은 성과를 보였습니다. 두 알고리즘 모두 대형 인스턴스를 해결하는 데 있어 다른 지역 검색 알고리즘보다 더 빠른 계산 속도를 자랑합니다.



### Detection Made Easy: Potentials of Large Language Models for Solidity Vulnerabilities (https://arxiv.org/abs/2409.10574)
- **What's New**: 최근 수년간 Ethereum 메인넷에 대한 대규모 Solidity 스마트 계약 배포가 재정적 동기를 가진 공격자들을 유인하고 있다. 본 논문은 OWASP Top Ten 취약점을 감지하기 위해 대형 언어 모델(LLMs)의 사용을 포괄적으로 조사하고, VulSmart라는 새로운 데이터셋을 소개하여 개방형 LLM과 폐쇄형 모델의 성능을 비교한다.

- **Technical Details**: 우리는 SmartVD라는 새로운 프레임워크를 제안하고, 이를 BLEU 및 ROUGE 지표를 활용하여 스마트 계약에서 취약점 감지 효과를 평가한다. 다양한 프롬프트 전략(Zero-shot, Few-shot, Chain-of-thought)을 사용하여 다중 클래스 분류 및 생성 능력을 평가하고, LLM 모델의 성능을 정량적으로 분석한다.

- **Performance Highlights**: SmartVD는 오픈소스 모델들을 초월하는 성능을 보여주며, GPT-3.5 Turbo 및 GPT-4 Mini와 같은 폐쇄형 모델들이 fine-tuning 후 99% 정확도로 취약점을 감지하는 데 성공하였다. 특히, Chain-of-thought 프롬프트 기법을 사용할 경우 SmartVD가 가장 높은 성능을 발휘한다.



### ASFT: Aligned Supervised Fine-Tuning through Absolute Likelihood (https://arxiv.org/abs/2409.10571)
- **What's New**: 본 연구에서는 Aligned Supervised Fine-Tuning (ASFT)이라는 새로운 접근법을 제안하여, Direct Preference Optimization (DPO)의 한계점을 해결하고 Large Language Model (LLM)을 인간의 선호와 더욱 잘 맞추도록 합니다.

- **Technical Details**: ASFT는 Bradley-Terry (BT) 모델을 사용하지 않고 절대 가능성을 최적화하여 쌍대 데이터셋에 대해 LLM을 정렬합니다. 이 방법은 추가적인 참조 모델 없이도 SFT 단계에서만 인간의 선호를 학습할 수 있게 합니다.

- **Performance Highlights**: ASFT는 MT-Bench 및 Arena-Hard 벤치마크에서 SFT 대비 48% 향상된 성능을 보였으며, DPO 및 그 변형보다도 월등한 결과를 나타냈습니다.



### Protecting Copyright of Medical Pre-trained Language Models: Training-Free Backdoor Watermarking (https://arxiv.org/abs/2409.10570)
Comments:
          9 pages

- **What's New**: 이번 연구에서는 의료 도메인에 특화된 첫 번째 훈련 없는 백도어 워터마킹 방법을 제안합니다. 이는 Med-PLMs(의료 사전 훈련 언어 모델)의 저작권 보호를 위해 개발되었습니다.

- **Technical Details**: 제안된 방법은 세 가지 단계로 구성됩니다: (1) 트리거 단어 선택: 원래 모델 소유자의 신원을 반영하는 특수 기호 선택. (2) 워터마크 삽입: 트리거 단어의 임베딩을 특정 의료 용어의 임베딩으로 대체. (3) 워터마크 추출: 트리거 단어가 입력될 때 모델의 반응을 관찰하여 워터마크를 확인합니다.

- **Performance Highlights**: 본 방법은 다양한 의료 하위 작업에서 높은 정확도를 유지하면서도 워터마크를 효과적으로 추출합니다. 또한, 워터마크 삽입 시간을 10시간에서 10초로 단축하여 효율성을 크게 향상했습니다.



### On the limits of agency in agent-based models (https://arxiv.org/abs/2409.10568)
Comments:
          19 pages, 5 appendices, 5 figures

- **What's New**: AgentTorch라는 새로운 프레임워크를 소개하며, 이는 대규모 에이전트 기반 모델(ABM)을 수백만 개의 에이전트로 확장하여 고해상도 에이전트 행동을 포착할 수 있도록 설계되었습니다.

- **Technical Details**: AgentTorch는 LLM(large language models)을 ABM 에이전트로 활용하여, 대규모 인구에서 에이전트의 의사결정 및 상호작용을 시뮬레이션하는 방법을 제시합니다. 이를 통해 시뮬레이션의 정확성과 에이전트의 적응성 사이의 균형을 분석하고, 에이전트 아키텍처에 대한 비교를 통해 역사적 데이터의 한계를 극복하는 방안을 제안합니다.

- **Performance Highlights**: COVID-19 팬데믹을 사례로 하여 뉴욕시를 대표하는 840만 개의 에이전트를 시뮬레이션한 결과, 사회적 고립과 고용 행동이 건강 및 경제적 결과에 미치는 영향을 효과적으로 포착하였습니다. 또한, 다른 에이전트 아키텍처와 LLM 에이전트의 성능을 비교하여 질병의 발생 및 실업률 예측에서의 정확성을 입증했습니다.



### Eureka: Evaluating and Understanding Large Foundation Models (https://arxiv.org/abs/2409.10566)
- **What's New**: 이 논문에서는 AI의 평가 프로세스를 개선하기 위한 새로운 프레임워크인 Eureka를 소개합니다. 이 프레임워크는 대형 기초 모델(Large Foundation Models)에 대한 더 나은 평가를 가능하게 하며, 단일 점수 보고와 순위 매김을 넘어서 다양한 모델의 능력을 비교하는 데 초점을 둡니다.

- **Technical Details**: Eureka는 모델 평가를 위한 유연한 라이브러리를 제공하여 데이터 전처리, 프롬프트 템플릿, 모델 추론, 데이터 사후 처리, 메트릭 계산 및 보고 작업을 사용자 맞춤형으로 조합 가능하게 합니다. 또한, Eureka-Bench는 사전 정의된 벤치마크 모음을 갖추고 있어 기존의 평가 방법이 간과하는 언어 및 다중 모드 능력을 테스트할 수 있도록 설계되었습니다.

- **Performance Highlights**: Eureka-Bench의 평가 결과, Claude 3.5 Sonnet, GPT-4o 2024-05-13, Llama 3.1 405B와 같은 특정 모델들이 여러 능력에서 반복적으로 다른 모델들보다 우수한 성능을 보였으나, 전반적으로는 어떤 단일 모델이 모든 작업에서 최선의 성능을 내지 않음을 보여줍니다. 특히, 현대 모델들은 이미지에 대한 세부 이해와 같은 기본 능력에서 여전히 한계를 보이고 있습니다.



### DrLLM: Prompt-Enhanced Distributed Denial-of-Service Resistance Method with Large Language Models (https://arxiv.org/abs/2409.10561)
- **What's New**: 본 연구는 최신 DDoS 공격에 대한 효과적인 대처를 목표로 하는 DrLLM(Distributed Denial of Service Resistance Large Language Model)을 제안합니다. DrLLM은 로컬 및 글로벌 트래픽 데이터를 Reasoning 패러다임에 통합하여 데이터를 효율적으로 분류하고 해석 가능성을 높입니다.

- **Technical Details**: DrLLM은 Knowledge Embedding, Token Embedding, Progressive Role Reasoning의 세 가지 모듈로 구성되어 있으며, 이들 모듈은 LLM(대형 언어 모델)의 전이학습(transfer learning)과 데이터 처리 기술을 통해 네트워크 흐름 데이터를 분석합니다. Knowledge Embedding 모듈은 데이터의 글로벌 정보를 추출하여 Knowledge Prompt에 삽입하며, Token Embedding 모듈은 네트워크 흐름 데이터를 코드화하여 텍스트 모달리티(text modality)와 정렬합니다.

- **Performance Highlights**: 공공 데이터셋 CICDDoS2019를 활용한 평가에서 DrLLM은 제로샷(zero-shot) 시나리오에서 뛰어난 성능을 입증하였으며, 다양한 실험을 통해 LLM이 네트워크 보안 분야에서 효과적으로 활용될 수 있음을 보여주었습니다.



### Unveiling Induction Heads: Provable Training Dynamics and Feature Learning in Transformers (https://arxiv.org/abs/2409.10559)
Comments:
          100 pages, 10 figures

- **What's New**: 이 연구는 인-context 학습(In-context Learning, ICL)의 이론적 기초를 탐구하며, 기존의 연구들이 주로 주의(attention) 메커니즘만을 다루는 반면, 이 논문은 transformer 아키텍처의 모든 구성 요소가 ICL에 어떻게 기여하는지를 분석합니다.

- **Technical Details**: 이 논문에서는 2개 주의 층(attention layer)을 가진 transformer 모델을 사용하여 n-gram Markov chain 데이터에 대한 ICL을 수행하는 방법을 연구합니다. 상대적 위치 임베딩(relative positional embedding), 다중 헤드 소프트맥스 주의(multi-head softmax attention), 정규화가 적용된 피드포워드 층(feed-forward layer)에 대한 논의가 포함되어 있습니다. 또한 교차 엔트로피 ICL 손실(cross-entropy ICL loss)과 관련된 그래디언트 흐름(gradient flow)이 수렴하는 방식도 증명합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델이 모든 구성 요소의 조화로운 기여로 인해 일반화된 induction head 메커니즘(generalized version of induction head mechanism)을 효율적으로 수행함을 입증하였습니다. 첫 번째 주의 층은 과거 토큰을 복사하는 역할을 하고(feed-forward 네트워크는 중요한 부모로부터 feature vector를 생성하는 선택자 역할을 하며), 두 번째 주의 층은 생성된 feature와 출력 위치의 feature를 비교하여 원하는 출력을 생성하는 분류기(classifier) 역할을 합니다.



### An Examination of Offline-Trained Encoders in Vision-Based Deep Reinforcement Learning for Autonomous Driving (https://arxiv.org/abs/2409.10554)
- **What's New**: 이 연구는 복잡한 부분 관찰 마르코프 의사결정 프로세스(POMDP)에서의 심층 강화 학습(Deep Reinforcement Learning, DRL)의 도전을 조사하고, 이러한 환경에서의 비전 기반 내비게이션 솔루션을 제안합니다.

- **Technical Details**: 이 연구는 BDD100K 드라이빙 비디오를 사용하여 사전 훈련된 인코더를 통해 일반화 가능한 표현을 학습하고, 이를 통해 CARLA 자율주행 시뮬레이터에서 차량 제어를 수행하는 드릴 네트워크를 훈련합니다. 다양한 자기지도 학습(self-supervised learning) 방법을 비교하고 대한 실험을 통해 DRL 에이전트의 성능을 평가합니다. 또한, 이 연구는 인코더의 아키텍처 설계 및 하이퍼파라미터 조정에 대한 세부 사항을 다룹니다.

- **Performance Highlights**: 사전 훈련된 인코더로부터 학습한 표현은 DRL 에이전트의 성능을 개선하며, CARLA 시뮬레이터에서의 차선 추적 및 충돌 회피 작업에서 제로샷 학습(zero-shot learning) 방식으로 직접 전이 가능한 것으로 나타났습니다.



### AI Literacy for All: Adjustable Interdisciplinary Socio-technical Curriculum (https://arxiv.org/abs/2409.10552)
Comments:
          Published at 2024 IEEE Frontiers in Education Conference

- **What's New**: 이 논문은 "모두를 위한 AI 리터러시(AI Literacy for All)"라는 교육과정을 제안하며, AI에 대한 다학제적 이해와 사회기술적 함의, 실제 적용에 초점을 맞추고 있습니다. 빠르게 발전하는 AI 기술에 대한 이해도가 필요한 시점에서, 이러한 새로운 접근법은 전통적인 AI 교육과정을 넘어 포괄적인 AI 리터러시 교육의 필요성을 강조합니다.

- **Technical Details**: AI 리터러시는 네 가지 주요 기둥으로 구성됩니다: AI의 범위 및 기술적 차원 이해, Gen-AI와의 지식적이고 책임 있는 상호작용, 윤리적이고 책임 있는 AI 관련 사회기술적 문제, AI의 사회적 및 미래적 함의. 이는 기술적 및 비기술적 학습 결과를 포함하는 균형 잡힌 교육과정을 강조합니다.

- **Performance Highlights**: AI 리터러시는 다양한 교육 환경에 적합하게 조정할 수 있으며, CS 전공 외의 전공자나 고등학교 여름 캠프 참석자, 성인 근로자, 대중을 대상으로도 적용될 수 있습니다. 이러한 교육과정은 기술 기반의 AI 교육을 넘어, AI와 관련된 윤리적, 사회적 논의의 주체로서 시민을 준비시키는 데 중점을 둡니다.



### NoPhish: Efficient Chrome Extension for Phishing Detection Using Machine Learning Techniques (https://arxiv.org/abs/2409.10547)
Comments:
          21 pages, 13 figures, 5 listings, 1 table

- **What's New**: 본 논문에서는 크롬 웹 브라우저를 위한 "NoPhish"라는 확장 프로그램을 개발하였습니다. 이 확장 프로그램은 사용자가 방문하는 웹 페이지의 진위를 확인하고 피싱 웹사이트 경고 기능을 제공하는 미들웨어로 기능합니다.

- **Technical Details**: 이 확장 프로그램은 여러 머신 러닝 기법을 사용하여 피싱 웹페이지를 식별합니다. PhishTank 데이터셋을 사용하여 알렉사 데이터베이스에서 평가된 22개의 주요 특성을 추출하였으며, 랜덤 포레스트(Random Forest), 서포트 벡터 머신(Support Vector Machine), k-최근접 이웃(k-Nearest Neighbor) 알고리즘을 적용했습니다. 실험 결과 랜덤 포레스트가 가장 높은 정확도를 보였습니다.

- **Performance Highlights**: 랜덤 포레스트 알고리즘이 최고의 성능을 기록하였으며, 이는 피싱 사이트를 식별하는 데에 있어서 유용한 결과를 나타냅니다. 이 확장 프로그램은 웹 보안과 사용자 경험의 통합을 통해 피싱 공격에 대한 방어 강화를 목표로 하고 있습니다.



### Adapting to the AI Disruption: Reshaping the IT Landscape and Educational Paradigms (https://arxiv.org/abs/2409.10541)
Comments:
          Submitted and accepted for CSCE'24: July 22-25, 2024

- **What's New**: 인공지능(AI)의 발전이 경제와 고용 구조를 크게 변화시키고 있으며, 이에 대한 기회와 도전을 탐색하는 논의가 이루어지고 있습니다. AI는 단순한 자동화 이상의 의미를 지니며, 산업 구조와 근무 형태를 재구성하는 커다란 변화를 가져옵니다.

- **Technical Details**: AI 시스템은 스스로 결정하고 분석할 수 있는 뛰어난 자율성을 가지고 있으며, 이는 전통적인 IT 일자리와 기술 세트를 위협합니다. 기업은 AI의 발전에 대응하기 위해 인력 관리 전략과 재교육 프로그램을 재검토해야 합니다.

- **Performance Highlights**: AI는 생산성 향상과 더불어 근무 시간 단축 가능성을 열어주고, 더 나아가 더 나은 근무 환경과 직원 복지를 위한 기회를 제공합니다. 4일 근무 주를 도입하는 기업은 근무의 질을 높이고 경쟁력을 갖추게 됩니다.



### The potential functions of an international institution for AI safety. Insights from adjacent policy areas and recent trends (https://arxiv.org/abs/2409.10536)
- **What's New**: 본 논문은 AI 기술의 원활한 관리와 위험 완화를 위해 국제 AI 안전 연구소의 필요성과 기능을 고찰합니다.

- **Technical Details**: 국제 AI 안전 연구소는 기술 연구 및 협력(technical research and cooperation), 안전 장치 및 평가(safeguards and evaluations), 정책 수립 및 관리 지원(policing and governance support) 등 세 가지 기능적 영역으로 나눌 수 있습니다. 또한, 기존의 국제 거버넌스 모델과 영국 및 미국의 국가 AI 안전 연구소로부터 배울 수 있는 점들을 분석합니다.

- **Performance Highlights**: 이 연구는 AI 기술의 잠재적 위험을 규명하고 평가하기 위한 국제적인 프로세스의 필요성을 강조하며, 새로운 국제 기구의 설계가 단순한 대안이 아닌, 그 구조를 모듈화된 관점에서 이해함으로써 사용할 수 있는 도구들을 찾을 수 있음을 제안합니다.



### Learning Co-Speech Gesture Representations in Dialogue through Contrastive Learning: An Intrinsic Evaluation (https://arxiv.org/abs/2409.10535)
- **What's New**: 이 논문은 코-스피치 제스처의 표현 학습에 대한 새로운 접근 방식을 제시합니다. 기존의 접근법에서는 제스처의 변동성과 말과의 관계를 효과적으로 고려하지 못했으나, 본 연구에서는 자기 감독 기반의 대조 학습(self-supervised contrastive learning)을 활용하여 이 문제를 해결합니다.

- **Technical Details**: 제안된 방법론은 단일 모달(unimodal)과 다중 모달(multimodal) 프리 트레이닝(pre-training) 방법을 포함하여 제스처 표현을 동시 발생하는 말에 기반하여 조정합니다. 이 연구는 풍부한 제스처 데이터셋에서 학습한 결과, 사람의 평가와 높은 상관관계를 나타내며, 동적 대화 상호 작용 관련 패턴과 잘 일치하는 제스처 표현의 유의미한 회복 가능성을 보여줍니다.

- **Performance Highlights**: 모델은 200 에폭(epoch) 동안 Pytorch 기반의 구현으로 훈련되었으며, 다중 모달 대조 학습을 통해 좋은 성능을 보였습니다. 연구 결과는 다양한 제스처 쌍의 유사성을 평가한 결과, 고유한 음성 및 말의 특징을 강조할 수 있는 잠재적 가능성을 나타내며, 텍스트와 제스처 간의 관계를 연구하는 데 큰 가능성을 열었습니다.



### From Latent to Engine Manifolds: Analyzing ImageBind's Multimodal Embedding Spac (https://arxiv.org/abs/2409.10528)
Comments:
          The 26th International Conference on Artificial Intelligence (ICAI'24)

- **What's New**: 이 연구에서는 ImageBind의 능력을 조사하여 온라인 자동차 부품 목록에 대한 의미 있는 융합 다중 모달 임베딩을 생성할 수 있음을 밝혔습니다. 또한 이미지/텍스트 쌍의 중복 정보를 캡처하고 포스트의 의미를 결합하여 공동 임베딩으로 통합하는 단순한 임베딩 융합 워크플로를 제안합니다.

- **Technical Details**: 이 연구에서는 ImageBind를 활용하여 이미지와 텍스트 임베딩을 론칭하여 중복 정보를 포착하는 데 초점을 맞추었습니다. ImageBind는 이미지, 텍스트, 오디오를 포함한 여섯 가지 데이터 유형을 단일 임베딩 공간으로 결합하여 의미를 표현하는 방법을 제공합니다. 최종적으로, 혼합된 다중 모달 임베딩은 임베딩 벡터의 평균을 통해 생성되며, 주 성능은 PCA(주성분 분석)를 통해 차원 축소 후 클러스터링을 통해 분석됩니다.

- **Performance Highlights**: ImageBind의 초기 결과는 순수 오디오 임베딩이 의미적으로 유사한 시장 목록과 상관관계를 가질 수 있음을 나타내며, 이는 향후 연구의 잠재적인 경로를 제시합니다. 클러스터 중심에 가장 가까운 게시물을 조사하여 공동 임베딩의 의미적 품질을 전달하는 경험적 증거를 제공합니다.



### Towards Empathetic Conversational Recommender Systems (https://arxiv.org/abs/2409.10527)
- **What's New**: 이 논문은 Empathetic Conversational Recommender (ECR) 시스템을 제안하여 사용자의 감정을 이해하고 반영하는 새로운 접근법을 소개합니다. 일반적인 대화형 추천 시스템(CRS)에서는 사용자 요구와 상관없이 표준 아이템과 응답을 기준으로 삼는 경향이 있는데, ECR은 이와 같은 오차를 수정하기 위해 감성을 도입합니다.

- **Technical Details**: ECR은 두 가지 주요 모듈로 구성됩니다: 감정 인식 아이템 추천(emotion-aware item recommendation)과 감정 정렬 응답 생성(emotion-aligned response generation). 사용자 감정을 기반으로 해 섬세한 추천 모델링을 수행하며, 감정에 맞춰 사전 훈련된 언어 모델(pre-trained language model)을 미세 조정하여 사람처럼 감정적인 응답을 생성합니다. 이때 대규모 언어 모델(large language models)을 사용하여 감정 라벨을 추가하고, 외부 소스에서 수집된 감정 리뷰로 데이터를 확장합니다.

- **Performance Highlights**: ReDial 데이터셋을 통해 ECR 프레임워크는 추천 정확도(recommendation accuracy)를 향상시키고 사용자 만족도(user satisfaction)를 크게 개선하여 기존 모델들을 능가하는 성능을 보였습니다. 새로운 평가 지표는 감정 강도(emotional intensity), 감정 설득력(emotional persuasiveness), 논리적 설득력(logic persuasiveness), 정보 제공(informativeness), 생동감(lifelikeness) 등을 포함하여 실세계 CRS 시나리오에서의 사용자 만족도를 더 잘 반영합니다.



### Effective Monitoring of Online Decision-Making Algorithms in Digital Intervention Implementation (https://arxiv.org/abs/2409.10526)
- **What's New**: 디지털 개입 분야에서 온라인 의사결정 알고리즘을 모니터링하기 위한 가이드라인을 제시하며, 이를 통해 개인의 치료를 동적으로 개인화하는 알고리즘의 효과적인 모니터링 방법을 설명합니다.

- **Technical Details**: 이 논문은 온라인 의사결정 알고리즘을 모니터링하기 위한 두 가지 주요 가이드라인을 소개합니다. 첫째, 문제가 발생할 경우 실행할 사전 정의된 절차인 fallback methods를 개발하고, 둘째, 잠재적 문제를 심각도에 따라 분류하여 식별합니다. 또한, 사례 연구로 Oralytics 및 MiWaves 두 가지 임상시험에서 실시간 문제를 감지한 모니터링 시스템에 대한 경험을 논의합니다.

- **Performance Highlights**: 모니터링 시스템은 메모리 부족 문제, 데이터베이스 시간 초과, 외부 출처와의 통신 실패 등의 실시간 문제를 감지했습니다. 이러한 방법은 참가자들이 치료를 누락당하지 않도록 하였고, 통계 분석에서 잘못된 데이터를 사용하는 것을 방지하였습니다. 이 사례 연구들은 건강 과학자들이 디지털 개입을 위한 모니터링 시스템을 구축할 수 있는 방법을 보여줍니다.



### "Is This It?": Towards Ecologically Valid Benchmarks for Situated Collaboration (https://arxiv.org/abs/2409.10525)
- **What's New**: 이번 연구는 대형 멀티모달 모델(Large Multimodal Models, LMMs)의 실제 협업에서의 능력을 평가하기 위한 생태학적으로 유효한 벤치마크를 구축하는 초기 작업을 보고합니다. 기존 벤치마크와 달리, 사용자가 인터랙션 도중 직접 생성한 질문을 바탕으로 질문-답변 쌍을 생성하는 시스템 중심 접근 방식을 제안합니다.

- **Technical Details**: 이 연구는 Sigma라는 혼합 현실 작업 지원 시스템을 활용하여 대규모 데이터 수집을 진행하였습니다. Sigma는 HoloLens 2를 기반으로 하며, 음성 인식과 생성, LLMs를 이용하여 사용자에게 절차적 작업을 단계적으로 안내합니다. 이를 통해 수집한 데이터는 전통적인 벤치마크와는 다른 질문 유형과 도전과제를 드러내는 것을 목표로 합니다.

- **Performance Highlights**: 파일럿 실험에서 26개의 상호작용 세션을 통해 수집된 데이터는 사용자가 자연스럽게 생성한 질문의 형태와 내용을 기존의 질문 응답 벤치마크와 비교할 때 상당한 차이를 보였습니다. 이는 LMM의 실제 사용자 경험과의 일치 여부를 보다 정확히 반영할 수 있음을 시사합니다.



### 3CSim: CARLA Corner Case Simulation for Control Assessment in Autonomous Driving (https://arxiv.org/abs/2409.10524)
- **What's New**: 본 논문에서는 CARLA 시뮬레이터를 기반으로 자율주행 시스템을 평가하기 위한 CARLA corner case simulation (3CSim) 프레임워크를 제안합니다. 3CSim은 비표준이고 드문 시나리오에 중점을 두어 자율주행 모델의 제어 능력을 평가하며, 32개의 고유 corner case를 구현하여 데이터셋 생성을 용이하게 합니다.

- **Technical Details**: 3CSim은 상태 이상(state anomalies), 행동 이상(behavior anomalies), 증거 기반 이상(evidence-based anomalies)으로 분류된 corner case를 제공합니다. 이 프레임워크는 날씨, 타이밍, 교통 밀도 등 9개의 사전 정의된 날씨 조건을 포함한 조절 가능한 매개변수를 포함하고 있습니다.

- **Performance Highlights**: 3CSim은 반복 가능하고 수정 가능한 시나리오 평가를 가능하게 하여 자율주행 시스템의 복잡한 환경에서의 성능을 분석합니다. 이를 통해 이 시스템의 안전성과 신뢰성을 향상시키는 데 기여할 수 있습니다.



### Harnessing Artificial Intelligence for Wildlife Conservation (https://arxiv.org/abs/2409.10523)
Comments:
          13 pages, 13 figures

- **What's New**: 이 논문은 인공지능(AI)을 활용한 야생동물 보존을 위한 혁신적인 전략을 탐구하며, Conservation AI 플랫폼에 대한 초점을 맞추고 있습니다. 이 플랫폼은 생물다양성(Biodiversity) 감시를 위한 최신 기술을 통합하여 야생동물의 모니터링을 향상시키고 있습니다.

- **Technical Details**: Conservation AI는 머신러닝(Machine Learning)과 컴퓨터 비전(Computer Vision)을 활용하여 시각 스펙트럼과 열화상 카메라를 통해 동물, 인간, 밀렵 관련 객체를 감지하고 분류합니다. 이 데이터는 합성곱 신경망(Convolutional Neural Networks, CNNs) 및 트랜스포머(Transformer) 구조를 이용하여 처리됩니다. 실시간 감지는 밀렵과 같은 시간에 민감한 상황을 지원하고, 비실시간 분석은 장기적인 야생동물 모니터링 및 서식지 건강 평가를 가능하게 합니다.

- **Performance Highlights**: 유럽, 북미, 아프리카, 동남아시아의 사례 연구를 통해 species identification, biodiversity monitoring, poaching prevention에서 플랫폼의 성공을 보여줍니다. 데이터 품질, 모델 정확성 및 물류적 제약과 같은 도전 과제에 대해 논의하며, 기술 발전, 새로운 지역으로의 확장 및 지역 사회 및 정책 입안자와의 깊은 협력을 포함한 미래 방향을 제시합니다.



### Bridging User Dynamics: Transforming Sequential Recommendations with Schr\"odinger Bridge and Diffusion Models (https://arxiv.org/abs/2409.10522)
Comments:
          CIKM '24

- **What's New**: 본 논문에서는 확산모델(difussion model)을 이용한 순차 추천 시스템(sequential recommendation)에서, 가우시안 분포(Gaussian distribution)의 한계를 극복하고 사용자의 현재 상태에 기반한 새로운 추천 모델(SdifRec)을 제안합니다.

- **Technical Details**: SdifRec 모델은 사용자의 현재 상태를 고려하여 기존의 가우시안 사전 분포를 대체합니다. 또한 사용자 클러스터링 정보를 조건으로 활용하여 posterior distribution을 개선하는 con-SdifRec라는 확장 모델도 개발했습니다. 연구에서 제안된 슈뢰딩거 다리(Schrödinger Bridge) 기법은 두 분포 간의 최소 비용 경로를 찾는 방법론으로, 이를 통해 모델의 유연성을 증가시킵니다.

- **Performance Highlights**: 다양한 공개 벤치마크 데이터셋에서 실시한 비교 실험 결과, SdifRec 및 con-SdifRec 모델이 여러 최신 방법론에 비해 상당한 성능 향상을 보이며 효율성과 안정성이 검증되었습니다.



### LSTM Recurrent Neural Networks for Cybersecurity Named Entity Recognition (https://arxiv.org/abs/2409.10521)
- **What's New**: 이 논문에서는 사이버 보안 분야의 정보 추출에 필요한 Named Entity Recognition (NER) 모델을 자동화한 새로운 접근 방식을 제시합니다. 이 모델은 도메인 독립적이며, 사이버 보안 엔티티의 특수 기능에 의존하지 않기 때문에 전문가의 지식 없이도 사용할 수 있습니다.

- **Technical Details**: 모델은 Long Short-Term Memory (LSTM)와 Conditional Random Fields (CRFs) 기법을 활용하여 구현되었습니다. 이 방법은 극복해야 할 사이버 보안의 복잡한 엔티티 구조를 다루기 위해 설계되었습니다.

- **Performance Highlights**: 제안된 방법은 적당한 크기의 주석이 있는 말뭉치를 사용했을 때 최첨단 방법들보다 우수한 성능을 보였습니다.



### Achieving Responsible AI through ESG: Insights and Recommendations from Industry Engagemen (https://arxiv.org/abs/2409.10520)
Comments:
          10 pages, 1 table, 1 figure

- **What's New**: 이번 연구는 인공지능(AI)의 책임 있는 배치(Responsible AI, RAI)를 환경, 사회 및 거버넌스(ESG) 프레임워크와 통합하는 것이 필수적임을 강조합니다. 특히, 28개의 산업 리더 인터뷰를 통해 RAI와 ESG 간의 강력한 연결 고리를 발견했습니다.

- **Technical Details**: 연구 방법론으로는 28개 기업의 인터뷰 및 데스크 리서치를 기반으로 하여 AI 정책, 실행 전략, 거버넌스 및 이해관계자 참여를 심층적으로 조사했습니다. ESG-AI 프레임워크를 통해 RAI 구현을 평가했으며, 회사들이 RAI를 ESG 목표와 어떻게 조화시킬 수 있는지를 논의했습니다.

- **Performance Highlights**: 연구 결과, 많은 기업들은 내부 RAI 정책과 공개된 정보 간의 격차가 존재하며, 이는 투명성 및 책임감 개선이 필요함을 나타냅니다. RAI 전략을 강화하기 위한 투명성, 교차 기능 협업, 기존 ESG 프레임워크와의 원활한 통합 방안을 제시하였습니다.



### ASMA: An Adaptive Safety Margin Algorithm for Vision-Language Drone Navigation via Scene-Aware Control Barrier Functions (https://arxiv.org/abs/2409.10283)
- **What's New**: 이번 연구에서는 비전-언어 내비게이션(VLN) 환경에서 원격 조종 드론의 안전성을 높이기 위해 진정한 장면 인식 제어 장벽 함수(CBF)를 제안하였습니다. 또한, 적응형 안전 여유 알고리즘(ASMA)을 통해 동적인 환경에 실시간으로 적응할 수 있는 방법을 소개합니다.

- **Technical Details**: 이 논문에서는 CLIP 모델과 YOLO 객체 탐지기를 결합하여 비전-언어 이해 모듈(VLUM)을 구현하였으며, RGB-D 센서를 통해 얻은 자아 중심의 관찰을 사용하여 장면 인식 제어 장벽 함수를 구성합니다. ASMA는 드론의 깊이 맵을 잘라내어 동적 장애물 추적에 적응할 수 있도록 설계되었습니다.

- **Performance Highlights**: ASMA를 적용한 결과, 기존 CBF 없는 VLN 방법에 비해 성공률이 59.4% - 61.8% 증가하였고, 경로 길이는 5.4% - 8.2% 증가하는 것으로 나타났습니다. 이는 불안전한 상황에서 회복하는 데에도 효과적입니다.



### Deception Detection from Linguistic and Physiological Data Streams Using Bimodal Convolutional Neural Networks (https://arxiv.org/abs/2311.10944)
Comments:
          Accepted by 2024 5th International Conference on Information Science, Parallel and Distributed Systems

- **What's New**: 이 논문은 다중 모달 (multimodal) 속임수 탐지(deception detection)를 위한 컨볼루션 신경망 (convolutional neural networks, CNNs)의 적용을 탐구합니다. 저자들은 104명의 피험자와의 인터뷰에서 수집된 데이터를 사용하여 진실한 응답과 허위 응답을 비교하고, 이를 바탕으로 신경망 모델을 훈련시킵니다.

- **Technical Details**: 논문에서는 언어적 특성과 생리적 특성을 추출하여 두 가지 모달리티(linguistic and physiological)에서 훈련한 CNN 모델을 제안합니다. 또한 두 모달리티를 융합한 CNN 모델을 개발하였으며, 작은 훈련 데이터셋을 사용할 때 과적합(overfitting) 문제를 해결하기 위한 단순한 접근법으로 다수결(voting)을 적용하였습니다.

- **Performance Highlights**: 제안된 시스템은 일반적인 분류(classification) 방법보다 성능이 우수하며, 제한된 데이터에서도 신경망을 활용한 속임수 탐지의 가능성을 보여주었습니다. 이는 다중 모달 데이터에 대한 심화 학습 방식을 적용함으로써 달성되었습니다.



### Diverse Neural Audio Embeddings -- Bringing Features back ! (https://arxiv.org/abs/2309.08751)
Comments:
          6 pages, 1 figure, 2 table, Under Review for 50th IEEE ICASSP 2025, Hyderabad, India

- **What's New**: 최근 인공지능 아키텍처의 발전으로 인해, 도메인 특화 지식 없이 최적화된 end-to-end 아키텍처가 주목받고 있습니다. 본 연구에서는 다양한 오디오 특성에 대한 로버스트한 임베딩을 학습하여 오디오 분류 성능을 향상시키는 방법을 탐구합니다.

- **Technical Details**: 연구에서는 pitch, timbre 와 같은 전통적인 수동 특성 임베딩을 end-to-end 아키텍처와 결합하여 성능을 개선하는 방법을 제시합니다. Transformer 아키텍처를 활용하여 특징 기반의 정보를 처리하고, 각 슬롯에 대해 64차원 임베딩을 생성하는 방법이 포함되어 있습니다.

- **Performance Highlights**: 본 연구의 결과로 feature-based 임베딩이 end-to-end 아키텍처보다 우수한 성능을 발휘함을 보여주었습니다. 특히, 도메인 지식을 활용한 임베딩은 전통적인 방식의 딥러닝 모델보다 더 강력한 분류 정확도를 자랑합니다.



### Audio Transformers:Transformer Architectures For Large Scale Audio Understanding. Adieu Convolutions (https://arxiv.org/abs/2105.00335)
Comments:
          5 pages, 4 figures; Under review WASPAA 2021

- **What's New**: 본 논문에서는 CNN (Convolutional Neural Networks) 아키텍처의 대안으로, 변형된 (Transformer) 아키텍처를 원시 오디오 신호에 적용하여 뛰어난 성과를 보여주고 있습니다. 특히, 자연어 처리와 컴퓨터 비전과 달리 비지도 전이 학습 없이도 우수한 성능을 발휘하는 것이 주목할 만합니다.

- **Technical Details**: 오디오 분류 작업에 있어 200개 범주로 구성된 Free Sound 50K 데이터셋을 사용하였고, 논문의 모델은 CNN 기반 모델을 초월하여 최고 성능을 기록했습니다. 또한, CNN에서 영감을 받은 풀링 기법과 웨이브릿 (wavelet)에서 기인한 다중 비율 신호 처리 아이디어를 Transformer embedding에 적용하여 성능을 한층 개선하였습니다.

- **Performance Highlights**: 우리 모델은 평균 정밀도(mean average precision) 기준으로 상당한 성능 향상을 보여주었으며, 비선형 비상수 대역폭 필터 뱅크를 학습하여 오디오 이해 작업을 위한 적응 가능한 시간 주파수 표현을 생성하는 방법도 제시합니다.



### Can GPT-O1 Kill All Bugs? An Evaluation of GPT-Family LLMs on QuixBugs (https://arxiv.org/abs/2409.10033)
- **What's New**: 이번 연구에서는 최신 GPT-o1 모델의 자동 프로그램 수리(Automated Program Repair, APR) 성능을 기존 GPT 계열 모델과 비교하여 최초로 검토했습니다. 여러 개의 평가 관점에서 4개의 GPT 계열 모델을 평가하여 O1이 모든 40개의 버그를 성공적으로 수정하는 결과를 도출했습니다.

- **Technical Details**: 연구에서는 QuixBugs 벤치마크를 사용하여 GPT 모델들이 버그를 수정하는 두 단계의 프로세스를 설계했습니다. 첫 번째 단계에서는 기본 수리 템플릿을 사용하여 모델이 버그를 탐지하도록 하고, 두 번째 단계에서는 테스트 케이스 오류 정보를 제공하여 추가 수리를 수행하도록 하였습니다. GPT-o1은 강화 학습(Reinforcement Learning, RL)과 사고의 연쇄(Chain of Thought, COT) 기법을 활용합니다.

- **Performance Highlights**: O1 모델은 ChatGPT(31/40)와 GPT-4o(38/40)보다 더 높은 성능을 보였고, 40개의 버그를 모두 수정하는 능력을 보여주었습니다. O1의 사고의 연쇄(pattern)는 복잡한 논리를 이해하고 올바른 수리 코드를 제공하는 데 특히 효과적임을 입증했습니다.



### Large Language Model Based Generative Error Correction: A Challenge and Baselines for Speech Recognition, Speaker Tagging, and Emotion Recognition (https://arxiv.org/abs/2409.09785)
Comments:
          IEEE SLT 2024. The initial draft version has been done in December 2023. Post-ASR Text Processing and Understanding Community: this https URL

- **What's New**: 최근 생성 AI 기술의 발전에 따라, LLM(대규모 언어 모델)이 동결된 사전 훈련된 자동 음성 인식(ASR) 모델의 텍스트 디코딩 결과를 활용하여 음향 모델링 작업을 향상시킬 수 있는 방법에 대한 질문이 제기되었습니다. 이 논문에서는 새로운 언어 모델링 기능을 탐구하기 위해 생성 음성 필기 오류 수정(GenSEC) 챌린지를 도입했습니다.

- **Technical Details**: GenSEC 챌린지는 다음 세 가지 post-ASR 언어 모델링 작업으로 구성됩니다: (i) post-ASR 전사 수정, (ii) 화자 태깅, (iii) 감정 인식. 이 작업들은 음성 기반 인터페이스를 처리하는 LLM 기반 에이전트의 미래를 모방하는 데 목표를 두고 있으며, 개방형 사전 훈련된 언어 모델이나 에이전트 기반 API를 사용하여 폭넓은 청중이 접근할 수 있도록 디자인되었습니다.

- **Performance Highlights**: 기초 평가에서 얻은 통찰력을 논의하며, 향후 평가 설계에 대한 교훈을 제공합니다. 참가자들은 ASR-LLM의 제한을 감안하여 텍스트 전용 모달리티로 음성을 처리하는 한계를 탐색하고, ASR 모델의 N-best 가설을 활용하여 발화 및 비언어적 정보를 복구하는 등의 연구를 촉진할 수 있습니다.




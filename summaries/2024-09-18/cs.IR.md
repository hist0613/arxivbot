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




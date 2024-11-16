New uploads on arXiv(cs.CL)

### A Bayesian Optimization Approach to Machine Translation Reranking (https://arxiv.org/abs/2411.09694)
Comments:
          v1: Preprint version

- **What's New**: 이 논문에서는 기계 번역 시스템의 후보 리스트를 외부 평가 모델로 리랭킹하여 최상위 점수를 반환하는 기존 접근 방식을 Bayesian 최적화(Bayesian optimization, BayesOpt) 문제로 설정합니다. 이는 탐색(exploration)과 활용(exploitation)의 균형을 통해 후보를 전략적으로 선택함으로써 이루어집니다.

- **Technical Details**: BayesOpt와 Gaussian process (GP)를 조합하여, 후보 점수의 일부만 평가하고도 최대 점수를 근접하게 달성하는 일반적인 리랭킹 알고리즘을 제시합니다. 본 연구에서는 저비용의 노이즈가 있는 프록시 스코어링 모델을 사용한 다중 신뢰도(multi-fidelity) 환경을 도입하여 비용-성능 트레이드오프를 더욱 개선합니다.

- **Performance Highlights**: 연구 방법은 200개의 후보에서 0.8216의 CometKiwi 점수를 얻을 수 있는 경우, 70개의 스코어 평가로 0.8210을 달성하며, 이는 기존 방식이 180번의 평가를 요구하는 것에 비해 효율적임을 보여줍니다. 노이즈가 있는 프록시 모델을 활용하여 후보 리스트를 선별하는 접근법이 성능 향상에 기여하며, 이는 이전 연구보다 더 향상된 성과를 나타냅니다.



### Squeezed Attention: Accelerating Long Context Length LLM Inferenc (https://arxiv.org/abs/2411.09688)
- **What's New**: 최근의 LLM(대형 언어 모델) 응용 프로그램은 문서 분석과 코드 생성과 같은 복잡한 하위 작업을 수행하기 위해 긴 입력 프롬프트(long input prompts)를 요구합니다. 본 연구에서는 입력 프롬프트의 큰 부분이 고정되어 있음을 활용하여 오프라인 최적화(offline optimization)를 수행하는 'Squeezed Attention' 메커니즘을 제안합니다.

- **Technical Details**: Squeezed Attention은 K-means 클러스터링을 사용하여 고정 컨텍스트의 키들을 의미적 유사성에 따라 그룹화한 후 각 클러스터를 단일 중심 값(centroid value)으로 표현합니다. 추론(inference) 중에는 사용자 입력의 쿼리 토큰(query tokens)과 중심값을 비교하여 의미적으로 관련된 키들을 예측하고, 이러한 중요한 키들만을 사용하여 정확한 attention을 계산하여 대역폭(bandwidth) 및 계산 비용(computational costs)을 줄입니다. 또한, 계층적 중심 조회(hierarchical centroid lookup)을 통해 중요 키를 식별하는 방법으로 attention 복잡성을 선형에서 로그(logarithmic)로 줄입니다.

- **Performance Highlights**: 최적화된 Triton 커널을 사용하여 중심값 비교 및 중요 키를 사용하는 희소 FlashAttention(sparse FlashAttention)을 구현하였으며, 긴 컨텍스트의 추론(pre-fill 및 생성 단계) 과정에서 4배 이상의 속도 향상을 달성했습니다. 또한, LongBench를 포함한 다양한 긴 컨텍스트 벤치마크에서 평가를 진행하여 KV 캐시 예산을 3배 감소시키고, 다양한 모델에서 0.5 포인트 정확도 차이로 최대 8배 감소를 이뤘습니다.



### Adaptive Decoding via Latent Preference Optimization (https://arxiv.org/abs/2411.09661)
- **What's New**: 이 논문에서는 기존의 고정된 온도 설정을 벗어나, Adaptive Decoding이라는 새로운 방식을 도입하여 언어 모델의 디코딩 과정에서 더 높은 성능을 발휘할 수 있도록 합니다. 이는 모델이 각 작업의 맥락에 따라 적절한 온도를 동적으로 선택할 수 있게 해 줍니다.

- **Technical Details**: Adaptive Decoding은 AdaptiveDecoder라는 새로운 레이어를 추가하여 모델이 다음 토큰을 디코딩하기 위한 이상적인 온도를 선택할 수 있게 합니다. 온도는 모델이 생성하는 출력의 창의성과 정확성을 조절하는 데 필수적인 요소입니다. Latent Preference Optimization (LPO) 기법을 통해, 모델은 다양한 온도를 학습하여, 수학 문제와 같은 정확성이 요구되는 작업에 대해 낮은 온도를, 창의적 글쓰기에는 높은 온도를 선택할 수 있도록 훈련됩니다.

- **Performance Highlights**: 실험 결과, AdaptiveDecoder는 다양한 작업에서 기존의 고정 온도 디코딩 방식을 초월하여 성능을 향상시킵니다. 예를 들어, UltraFeedback, Creative Story Writing, GSM8K와 같은 다양한 작업에서 더 높은 성능을 보이며, 특정 작업에 따라 최적의 온도로 적응하여 다양한 아웃풋을 생성할 수 있습니다.



### PTR: Precision-Driven Tool Recommendation for Large Language Models (https://arxiv.org/abs/2411.09613)
- **What's New**: 이번 연구에서는 LLMs(대형 언어 모델)에 대한 도구 추천 문제를 새롭게 정의하고, Precision-driven Tool Recommendation (PTR) 접근 방식을 제안합니다. 이 방식은 역사적 도구 묶음 사용 데이터를 활용하여 적절한 도구 세트를 추천하며, RecTools라는 새로운 데이터셋과 TRACC라는 평가 지표를 도입하여 LLMs의 도구 추천 효과성을 평가합니다.

- **Technical Details**: PTR 접근 방식은 도구 묶음 확보, 기능 범위 매핑, 다중 뷰 기반 재정렬의 세 가지 주요 단계를 포함합니다. 이 과정에서 LLMs가 쿼리를 해결하는 데 필요한 도구를 정확히 찾고, 분류하여 추천할 수 있도록 합니다. 새로운 평가 지표 TRACC는 추천된 도구의 정확성과 품질을 모두 고려하여 도구 추천 성과를 측정합니다.

- **Performance Highlights**: 이 연구를 통해 제안된 PTR 접근 방식은 두 개의 공개 벤치마크 및 RecTools 데이터셋에서 높은 정확도를 입증하였습니다. 전반적으로 권장된 도구 세트는 효율적이고 적절하여 LLMs의 작업 처리 성능을 향상시키는 데 기여합니다.



### The Moral Foundations Weibo Corpus (https://arxiv.org/abs/2411.09612)
- **What's New**: 이번 논문에서는 도덕적 감정을 측정하기 위한 새로운 데이터셋인 Moral Foundation Weibo Corpus를 소개합니다. 이 데이터셋은 25,671개의 중국어 댓글로 구성되어 있으며, six diverse topic areas를 포함합니다.

- **Technical Details**: 해당 댓글들은 도덕 이론에 기초하여 정의된 ten moral categories에 따라 최소 세 명의 체계적으로 훈련된 주석자들에 의해 수동으로 주석이 달렸습니다. 주석자 신뢰성을 평가하기 위해 kappa test 결과가 제시됩니다.

- **Performance Highlights**: 최근의 여러 대형 언어 모델을 활용하여 수동 주석을 보완하고, 도덕적 감정 분류를 위한 성능 비교 실험을 수행하여 baseline 결과를 보고하였습니다.



### BabyLM Challenge: Exploring the Effect of Variation Sets on Language Model Training Efficiency (https://arxiv.org/abs/2411.09587)
Comments:
          This paper accepted BabyLM challenge 2024 at CONLL 2024

- **What's New**: 이번 연구는 아동지향 언어(CDS)가 현대 언어 모델의 데이터 효율성을 개선하는 데 미치는 영향 중 Variation Sets (VSs)의 역할을 탐구합니다. 연구에서는 VSs를 다양한 비율로 포함한 CDS 데이터를 사용하여 GPT-2 모델을 훈련시켰습니다.

- **Technical Details**: Variation Sets (VSs)은 유사한 의도를 가진 일련의 발화로, 약간의 단어와 구조의 변화가 있으며, CDS에서 흔히 나타납니다. 연구진은 VSs 비율을 0%, 20%, 40%, 60%, 80%, 100%로 조절하여 신경망 모델의 훈련 데이터를 강화했습니다. 모델의 성능은 BLiMP, EWOK, GLUE 벤치마크를 통해 평가되었습니다.

- **Performance Highlights**: 연구 결과, BLiMP 및 GLUE 점수는 VSs의 존재에서 이익을 보았으나, EWOK 점수는 그러하지 않았습니다. 또한, 실험 결과는 에포크 수와 발화 제시 순서에 따라 달라지는 것으로 나타났습니다.



### Piecing It All Together: Verifying Multi-Hop Multimodal Claims (https://arxiv.org/abs/2411.09547)
- **What's New**: 새로운 멀티호프 멀티모달 주장 검증 작업을 도입하여 텍스트, 이미지, 표 등 다양한 출처의 여러 증거를 통해 주장의 정확성을 평가합니다.

- **Technical Details**: MMCV는 16,000개의 멀티호프 주장을 담은 대규모 데이터셋으로, 대규모 언어 모델(LLMs)을 활용하여 데이터 주석을 생성하고 인간의 피드백으로 품질을 보장합니다.

- **Performance Highlights**: 최신 멀티모달 대형 언어 모델(MLLMs)조차 복잡한 멀티모달 주장을 검증하는 데 어려움을 겪으며, 특히 추론 단계가 증가할수록 성능이 제한됩니다.



### A Practical Guide to Fine-tuning Language Models with Limited Data (https://arxiv.org/abs/2411.09539)
- **What's New**: 이 논문은 제한된 데이터로 LLMs(대규모 언어 모델)를 학습하는 최신 전이 학습(transfer learning) 접근 방식을 조사하였습니다. 데이터가 부족한 상황에서 최적의 모델 성능을 달성하기 위한 실용적인 지침을 제공합니다.

- **Technical Details**: 저자들은 초기 및 지속적인 사전 학습(pre-training) 전략을 다루며, 제한된 데이터 동안의 파인 튜닝(fine-tuning) 및 몇 가지 샷 학습(few-shot learning) 중 유용성 극대화 방법을 제시합니다. 또한, 특정 작업 관점에서 데이터 부족 문제를 해결하기 위한 모델과 방법들을 검토합니다.

- **Performance Highlights**: 이 연구는 NLP 분야의 연구자와 실무자에게 데이터가 제한된 시나리오에서 모델 성능을 최적화하기 위한 현재 최첨단 방법에 대한 종합적인 개요를 제공합니다. 주요 전략으로는 사전 학습 및 후속 파인 튜닝 과정에서의 데이터 효율적인 활용에 중점을 두어 고안되었습니다.



### The Use of Readability Metrics in Legal Text: A Systematic Literature Review (https://arxiv.org/abs/2411.09497)
- **What's New**: 본 연구는 법적 문서의 가독성을 평가하는 데 사용되는 다양한 언어적 메트릭스를 체계적으로 검토하였습니다. 3566개의 논문을 선별하여 34개의 관련 연구를 발견하고 분석한 결과, 특히 Flesch-Kincaid Grade Level 방식이 가장 많이 사용되는 메트릭스로 나타났습니다.

- **Technical Details**: 연구는 PRISMA(Preferred Reporting Items for Systematic Reviews and Meta-Analyses) 기준을 사용하여 문헌 검색 과정을 구조화하였으며, Scopus, Web of Science, IEEEXplore를 통한 데이터베이스 검색을 수행했습니다. 사용된 주요 키워드는 (complex* OR metric* OR measur*) AND (readability OR linguistic) AND (regulat* OR law OR legislation)입니다. 가독성 또는 언어 관련 측정 또는 방법론을 포함한 영어로 작성된 문서만 포함되었습니다.

- **Performance Highlights**: 법적 문서의 가독성을 높이는 것이 중요하지만, 법적 언어의 복잡성을 다루기 위한 표준화된 접근 방법이 부족하다는 점이 강조되었습니다. 현재 법적 문서의 가독성을 평가하기 위해 활용되는 메트릭스에 대한 체계적인 연구가 없다 보니, 향후 연구에서는 특정 법적 분야에서의 가독성 개선이 제한적일 수 있습니다.



### MM-Eval: A Hierarchical Benchmark for Modern Mongolian Evaluation in LLMs (https://arxiv.org/abs/2411.09492)
- **What's New**: 이 논문은 현대 몽골어에 대한 대형 언어 모델(LLM)의 능력을 평가하기 위한 MM-Eval이라는 전문화된 데이터셋을 소개합니다. 이 데이터셋은 LM의 언어 능력(구문 및 의미)과 인지 능력(지식 및 추론)을 분리해 평가함으로써 몽골어와 같은 저자원 언어 지원의 격차를 해소하는 것을 목표로 합니다.

- **Technical Details**: MM-Eval 데이터셋은 569개의 구문, 677개의 의미, 344개의 지식, 250개의 추론 문제를 포함합니다. 이는 LLM의 성능을 분석하기 위해 구문(Syntax), 의미(Semantics), 지식(Knowledge), 추론(Reasoning) 네 가지 구성요소로 나뉘어져 있습니다.

- **Performance Highlights**: 예비 실험 결과에 따르면, 모든 모델이 구문 작업에서 의미 작업보다 더 우수한 성능을 보였으며, 이는 깊은 언어 이해에서의 격차를 강조합니다. 또한 지식 작업에서도 중간 수준의 성능 저하가 관찰되어, 모델들이 고자원 언어에서 저자원 언어로 일반 지식을 이전할 수 있음을 시사합니다.



### Everyone deserves their voice to be heard: Analyzing Predictive Gender Bias in ASR Models Applied to Dutch Speech Data (https://arxiv.org/abs/2411.09431)
Comments:
          Accepted at ECML PKDD 2024, 4th Workshop on Bias and Fairness in AI (BIAS)

- **What's New**: 본 연구는 Whisper 모델이 네덜란드어 음성 데이터에서 성별 그룹 간 성과 차이를 보이는 것을 분석하여 ASR(Automatic Speech Recognition) 기술의 예측 편향을 조사했습니다. 특히, ASR 시스템의 공정성과 품질 서비스 저해 요소에 대한 도덕적 관점을 포함하여 깊이 있는 논의를 제공합니다.

- **Technical Details**: Whisper 모델은 680k 시간 분량의 인간 생성 콘텐츠로 훈련된 제로샷(Zero-shot) 기초 모델입니다. 이 모델은 다양한 오디오 품질을 처리할 수 있는 Transformer 아키텍처를 사용하며, 여러 가지 크기(모델 복잡성)를 가진 Whisper 모델들을 평가했습니다. 각 모델의 성능은 단어 오류율(Word Error Rate)과 동일 의미 기준에 따라 비교했습니다.

- **Performance Highlights**: Whisper 모델의 성능 분석 결과, 조사된 모든 모델 크기에서 성별 그룹 간 단어 오류율(WER)에서 상당한 차이가 나타났으며, 통계적 검정을 통해 이 편향이 확인되었습니다. 이 연구는 자동 자막 생성을 위한 ASR 시스템의 공정성을 향상시키기 위한 잠재적 개선 영역을 식별하는데 기여합니다.



### DriveThru: a Document Extraction Platform and Benchmark Datasets for Indonesian Local Language Archives (https://arxiv.org/abs/2411.09318)
Comments:
          12 pages, 3 figures, 6 tables

- **What's New**: 이 논문은 인도네시아어 언어 처리 자원을 수집하는 데 있어 새로운 접근 방식을 제안하며, 기존의 수작업 외에도 문서의 디지털화와 OCR 기술을 활용하여 더 많은 언어 자원을 구축할 수 있는 방법을 모색하고 있습니다. 이를 위해 DriveThru 플랫폼을 개발하였습니다.

- **Technical Details**: DriveThru 플랫폼은 사용자가 이미지 형식의 문서를 업로드하면 자동으로 텍스트를 추출하는 시스템입니다. TesseractOCR과 같이 상용 OCR 엔진을 사용하고, 이후 LLM(대형 언어 모델)을 활용한 후처리를 통해 문자 정확도와 단어 정확도를 높이는 연구를 포함합니다. 이 시스템은 특히 자원이 부족한 4가지 인도네시아어(local languages) 언어인 자바어(Javanese), 순다어(Sundanese), 미낭카바우어(Minangkabau), 발리어(Balinese)로 테스트 되었습니다.

- **Performance Highlights**: 본 연구에서는 OCR의 문자 정확도(CAR)와 단어 정확도(WAR)를 향상시키기 위해 LLM을 활용한 결과, 기존 상용 OCR보다 개선된 성과를 보였습니다. 이는 기존 자원을 디지털화하고 후처리하는 접근 방식을 통해 인도네시아어 자원의 양과 질을 동시에 향상시킬 수 있음을 보여줍니다.



### DTELS: Towards Dynamic Granularity of Timeline Summarization (https://arxiv.org/abs/2411.09297)
Comments:
          Under review

- **What's New**: 본 논문에서는 기존의 Timeline Summarization (TLS)을 넘어섭니다. 새로운 접근법인 Dynamic-granularity TimELine Summarization (DTELS)을 제안하며, 사용자의 요구에 따라 동적으로 변화하는 타임라인을 생성하는 것을 목표로 합니다.

- **Technical Details**: DTELS는 정보성(Informativeness), 세분성 일관성(Granular Consistency), 사실성(Factuality), 일관성(Coherence) 등 네 가지 측면에서 타임라인 품질을 평가하는 포괄적인 벤치마크를 구축합니다. 대규모 다원 출처 데이터셋인 DTELS-Bench를 구축하여 세 가지 사전 정의된 세분성으로 주석을 다는 방식을 사용합니다.

- **Performance Highlights**: 실험 결과, LLM 기반의 솔루션들이 기존의 최첨단 TLS 기법에 비해 뛰어난 성능을 보이지만, 정보의 품질과 요구되는 세분성에 대한 일관성을 충족하는 데 여전히 한계가 있음을 보여줍니다.



### StreamAdapter: Efficient Test Time Adaptation from Contextual Streams (https://arxiv.org/abs/2411.09289)
Comments:
          22 Pages, 9 Figures

- **What's New**: 본 논문에서는 기존의 in-context learning (ICL) 방식의 한계를 극복하기 위해 StreamAdapter라는 새로운 방법을 제안합니다. StreamAdapter는 모델 파라미터를 테스트 시간에 직접 업데이트하여 명시적인 in-context demonstration이 필요하지 않도록 합니다.

- **Technical Details**: StreamAdapter는 context mapping 및 weight absorption 메커니즘을 사용하여 ICL demonstration을 파라미터 업데이트로 동적으로 변환합니다. 이 접근법은 추가적인 파라미터 없이 ICL의 장현을 활용하여 모델이 새로운 작업에 적응할 수 있게 지원합니다.

- **Performance Highlights**: 다양한 작업과 모델 아키텍처를 통해 수행된 실험에서 StreamAdapter는 ICL과 유사하거나 우수한 적응 능력을 보여주었으며, 더 적은 demonstration으로도 효율적인 추론을 가능하게 했습니다. 또한, StreamAdapter는 상수 시간 복잡도로 추론 비용을 크게 줄이고, 더 나은 강인성과 성능을 입증했습니다.



### Cross-Modal Consistency in Multimodal Large Language Models (https://arxiv.org/abs/2411.09273)
- **What's New**: 이 연구에서는 시각(vision) 및 언어(language) 모달리티 간의 상호작용을 탐구하고 이를 비교하는 새로운 개념인 cross-modal consistency를 도입합니다. 또한 이 개념에 기반한 정량적 평가 프레임워크를 제안합니다.

- **Technical Details**: 연구에서는 Vision Large Language Models (VLLMs)에 대해 전방위적으로 평가하는 새로운 접근 방식을 제시하며, GPT-4V 모델의 비전과 언어 간의 일관성(notably, inconsistency)도 드러냅니다. 이를 위해 다양한 비전-언어 병렬 데이터세트를 구성하고 실험을 수행했습니다.

- **Performance Highlights**: GPT-4V의 성능은 동일한 태스크 인스턴스에 대해 서로 다른 모달리티에서 다르게 나타나는 경향이 있으며, 이로 인해 Vision-Depicting-Prompting (VDP) 방법론을 도입하여 개선 가능성을 모색합니다. 이러한 발견은 향후 멀티모달 모델 사용에 있어 더 효과적인 활용 방안을 제시합니다.



### DAHL: Domain-specific Automated Hallucination Evaluation of Long-Form Text through a Benchmark Dataset in Biomedicin (https://arxiv.org/abs/2411.09255)
Comments:
          EMNLP2024/FEVER

- **What's New**: DAHL은 생물의학 영역에서 긴 형식의 텍스트 생성에서 환각(hallucination)의 평가를 위해 설계된 벤치마크 데이터셋과 자동 평가 시스템을 소개합니다. 8,573개의 질문이 포함된 데이터셋은 생물의학 연구 논문을 기반으로 하여 세심하게 구성되었습니다.

- **Technical Details**: DAHL은 각 응답을 원자 단위(atomic units)로 분해하여, 각 응답이 구성하는 정보 조각의 정확성을 평균하여 DAHL Score를 생성합니다. 이는 이전의 다중 선택 과제가 아닌 긴 형식 텍스트 생성의 사실적 정확성을 평가하는 새로운 방법론입니다.

- **Performance Highlights**: 8개의 다양한 모델을 통한 실험 결과, 더 큰 모델일수록 환각이 적은 경향을 보였으나 7~8억 매개변수를 초과하는 모델의 크기 증가는 사실 정확도에 유의미한 차이를 보이지 않았습니다. DAHL Score는 인간 주석(preference labels) 대비 효율적인 대안으로, 다른 전문 분야로의 확장이 가능합니다.



### Enhancing Financial Domain Adaptation of Language Models via Model Augmentation (https://arxiv.org/abs/2411.09249)
- **What's New**: 이 연구는 Composition to Augment Language Models (CALM)라는 모델을 통해 대형 언어 모델(LLMs)의 금융 도메인(domain)에 대한 적응 가능성을 검증하였습니다.

- **Technical Details**: CALM은 서로 다른 기능을 가진 두 개의 LLM 간의 cross-attention을 도입하여 기존 모델의 능력을 확장하는 모델입니다. 연구팀은 금융 특화 LLM을 활용하여 응답 능력이 강한 기존 LLM의 금융 성능을 향상시키는 방안을 모색했습니다. CALM은 금융 특화 LLM의 훈련에 사용된 데이터셋과는 다른 금융 데이터셋을 사용하여 훈련되었습니다.

- **Performance Highlights**: CALM은 정량적인 일본 금융 벤치마크와 정성적인 응답 비교를 통해 원래 모델들과 기준선(baselines)보다 더 높은 점수를 기록하며 우수한 응답을 제공할 수 있음을 보여주었습니다. 또한 모델의 중간 레이어를 연결하는 것이 금융 도메인 적응에 가장 효과적이라는 결과를 확인했습니다.



### HateGPT: Unleashing GPT-3.5 Turbo to Combat Hate Speech on X (https://arxiv.org/abs/2411.09214)
Comments:
          Accepted at FIRE 2024 (Track: Hate Speech and Offensive Content Identification in English and Indo-Aryan Languages (HASOC)). arXiv admin note: text overlap with arXiv:2411.05039, arXiv:2411.06946

- **What's New**: 이번 연구에서는 소셜 미디어 플랫폼에서의 혐오 발언 및 공격적인 콘텐츠 탐지를 위해 최첨단 대형 언어 모델인 GPT-3.5 Turbo를 활용했습니다. 특히, 영어 트윗을 두 가지 범주인 혐오 및 공격적인 콘텐츠와 비혐오 비공격적인 콘텐츠로 분류하는 작업을 진행하였고, 이 과정을 통해 코드-믹스드 언어 처리의 필요성을 강조하였습니다.

- **Technical Details**: 연구에서는 Macro-F1 점수를 주요 평가 지표로 사용하여 모델의 성능을 평가하였습니다. 세 가지 실험에서 Run 1은 0.756의 Macro-F1 점수를 기록하여 가장 높은 성능을 보였고, Run 2는 0.751, Run 3은 0.754의 점수를 기록하였습니다. 이러한 결과는 모델이 정밀도(precision)와 재현율(recall) 측면에서 일관된 성능을 보여줌을 시사합니다.

- **Performance Highlights**: 모델은 세 가지 실험에서 모두 우수한 성능을 보였으며, 특히 Run 1에서 가장 높은 Macro-F1 점수인 0.756을 달성했습니다. 이는 모델이 다양한 클래스 간의 균형을 잘 유지하고 있다는 것을 의미하며, 전반적으로 모델의 신뢰성과 견고성을 강조합니다.



### Comprehensive and Practical Evaluation of Retrieval-Augmented Generation Systems for Medical Question Answering (https://arxiv.org/abs/2411.09213)
- **What's New**: 이 연구는 의료 분야에서 질문-답변(QA) 시스템의 신뢰성을 평가하기 위한 새로운 간행물인 Medical Retrieval-Augmented Generation Benchmark (MedRGB)를 소개합니다. MedRGB는 LLMs의 여러 가지 상황에서의 성능을 평가하기 위해 4개의 테스트 시나리오를 포함하고 있습니다.

- **Technical Details**: MedRGB는 3480개의 인스턴스로 구성되어 있으며, 이를 통해 모델의 강도와 취약점을 평가하기 위해 7개의 LLMs를 테스트합니다. 테스트 시나리오는 Standard-RAG, Sufficiency, Integration, Robustness로 나뉘며, 각 시나리오는 LLMs의 정보 통합과 노이즈 처리 능력을 평가합니다.

- **Performance Highlights**: 실험 결과, 현재 모델들은 검색된 문서에서의 노이즈와 잘못된 정보 처리에 한계를 보였습니다. 이 연구는 RAG 시스템의 신뢰성을 높이기 위한 향후 방향에 대한 통찰을 제공합니다.



### Unstructured Text Enhanced Open-domain Dialogue System: A Systematic Survey (https://arxiv.org/abs/2411.09166)
Comments:
          45 pages, 3 Figures, 11 Tables

- **What's New**: 본 논문에서는 비구조 텍스트를 외부 지식 소스로 사용하는 오픈 도메인 대화 시스템(UTEDS)에 대해 연구하였습니다. UTEDS의 전통적인 데이터 기반 대화 시스템과의 차별점을 분석하고 그에 따른 정의를 제공하며, 최근 발표된 데이터셋과 모델을 요약합니다.

- **Technical Details**: UTEDS 모델은 검색 기반(Retrieval)과 생성 기반(Generative)으로 나뉘며, 각각의 모델은 다수의 모듈로 구성됩니다. 검색 모델은 Fusion, Matching, Ranking 모듈로 구성되며, 생성 모델은 대화 인코딩(Dialogue Encoding), 지식 선택(Knowledge Selection), 반응 생성(Response Generation) 모듈을 포함합니다.

- **Performance Highlights**: UTEDS의 성능 평가 방법을 요약하고 현재 모델의 성능을 분석하였습니다. 논문에서는 UTEDS의 발전 방향을 논의하며, 향후 연구에 새로운 영감을 주기를 희망합니다.



### DROJ: A Prompt-Driven Attack against Large Language Models (https://arxiv.org/abs/2411.09125)
- **What's New**: 이번 연구는 Directed Representation Optimization Jailbreak (DROJ)라는 새로운 기법을 소개하여, LLM의 프롬프트에서 악의적인 쿼리의 숨겨진 표현을 변경하여 모델이 긍정적인 반응을 유도하는 방향으로 최적화하는 방법을 제안합니다.

- **Technical Details**: DROJ는 주어진 쿼리의 숨겨진 상태를 분석하고, 주성분 분석 (PCA)을 사용하여 낮은 차원 공간으로 프로젝션 한 후, 각 쿼리의 거부 확률을 예측하기 위해 로지스틱 회귀 모델을 적합합니다. 이 과정을 통해, harmful query와 harmless query 모두 모델의 긍정적인 응답을 유도하기 쉬운 방향으로 전환될 수 있습니다.

- **Performance Highlights**: DROJ는 LLaMA-2-7b-chat 모델에서 100%의 키워드 기반 공격 성공률 (Attack Success Rate, ASR)을 달성했으며, 직접적인 거부를 효과적으로 방지합니다. 다만, 모델의 응답이 때때로 반복적이고 비정보적일 수 있으며, 이를 해결하기 위해 유용성 시스템 프롬프트를 도입하여 응답의 품질을 향상시켰습니다.



### P-MMEval: A Parallel Multilingual Multitask Benchmark for Consistent Evaluation of LLMs (https://arxiv.org/abs/2411.09116)
- **What's New**: 이번 연구는 LLMs(대형 언어 모델)의 다국어 처리 능력을 종합적으로 평가하기 위한 새로운 벤치마크 P-MMEval을 제안합니다. 이는 기존 연구에서 간과했던 벤치마크의 유용성을 고려하여 다양한 데이터셋을Selecting하는 새로운 파이프라인을 도입하여 다국어 멀티태스킹을 위한 포괄적인 기준을 제공합니다.

- **Technical Details**: P-MMEval 벤치마크는 세 가지 기본 NLP(자연어 처리) 데이터셋과 다섯 가지 전문적 능력에 관한 데이터셋으로 구성되어 있으며 10개 언어에서 언어 커버리지를 유지하도록 설계되었습니다. 분석에는 T-test와 같은 통계적 방법을 활용하여 각 모델의 성능을 정확하게 평가합니다.

- **Performance Highlights**: P-MMEval을 활용하여 다양한 LLMs의 성능을 비교 분석하며, 다양한 프롬프트, 모델, 언어, 작업에 따른 멀티링구얼 성능을 확장적으로 실험합니다. 이로 인해 향후 연구에 유용한 통찰력을 제공합니다.



### Personalized Help for Optimizing Low-Skilled Users' Strategy (https://arxiv.org/abs/2411.09109)
Comments:
          9 pages, 3 figures

- **What's New**: 이 연구에서는 Diplomacy 게임에서 AIs와 인간의 협업을 살펴보았습니다. 특히, 자연어 에이전트인 CICERO를 개선하여 플레이어의 의도에 기반한 이동 및 메시지 조언을 생성합니다.

- **Technical Details**: 이번 연구는 Personalized Help for Optimizing Low-Skilled Users’ Strategy (pholus)라는 자연어 에이전트를 소개했습니다. pholus는 실시간으로 Diplomacy 플레이어에게 이동 및 메시지 조언을 제공하며, 12개의 게임에서 1070개의 플레이어 턴과 117시간의 데이터를 수집했습니다.

- **Performance Highlights**: pholus는 초보 플레이어가 경험 많은 플레이어와 경쟁할 수 있도록 도와주며, 때로는 초보자가 경험 많은 플레이어를 초과하는 경우도 있었습니다. 연구에 따르면, 조언을 받은 초보자는 조언을 받지 않은 경우보다 더욱 나은 성과를 보였습니다.



### Code-mixed LLM: Improve Large Language Models' Capability to Handle Code-Mixing through Reinforcement Learning from AI Feedback (https://arxiv.org/abs/2411.09073)
Comments:
          initial version: 5 pages, 2 figures

- **What's New**: 본 논문은 코드-믹싱(code-mixing) 및 코드-스위칭(code-switching) 캠에 대한 다국어 대형 언어 모델(LLM)의 성능을 평가하고, AI 피드백을 통한 강화 학습(Reinforcement Learning from AI Feedback, RLAIF)을 통해 모델의 이러한 혼합 언어 이해 능력을 개선하는 방법을 제안합니다.

- **Technical Details**: 코드-믹싱은 두 개 이상의 언어가 혼합된 언어 사용을 의미하며, 이를 효과적으로 처리하기 위한 다국어 LLM의 성능을 벤치마킹합니다. 성능 향상을 위해, 기존의 LLM에 RLAIF 방법론을 적용하여 코드-믹싱 기계 번역 작업을 개선합니다.

- **Performance Highlights**: 실험 결과는 RLAIF 방법이 코드-믹싱을 처리하는 LLM의 성능을 향상시킬 가능성이 있음을 보여주고 있습니다. RLAIF와 같이 AI 피드백을 활용하는 접근 방식이 향후 더 유연하고 인간 중심의 AI 시스템 개발에 기여할 수 있을 것입니다.



### CoCoP: Enhancing Text Classification with LLM through Code Completion Promp (https://arxiv.org/abs/2411.08979)
- **What's New**: 코드 완성 프롬프트(Code Completion Prompt, CoCoP) 방법을 제안하여 대형 언어 모델(LLMs)의 텍스트 분류 성능을 개선했습니다. 이 방법은 텍스트 분류 문제를 코드 완성 작업으로 변환하여 LLMs의 코드 완성 기능을 활용합니다.

- **Technical Details**: CoCoP는 LLMs의 코드 완성 능력을 활용하고, in-context learning 기법을 사용합니다. 이 방법은 단계별로 불완전한 코드를 생성하여 LLM이 사용자 쿼리에 대한 적절한 레이블을 결정할 수 있도록 합니다. CoCoP는 LLaMA2 및 CodeLLaMA 모델을 기반으로 다양한 분류 데이터셋에서 검증되었습니다.

- **Performance Highlights**: CoCoP는 SST2 데이터셋의 정확도를 20% 이상 향상시켰습니다. 코드 관련 작업을 위해 설계된 LLMs(코드 모델)와 통합할 경우, CoCoP는 소규모 모델(7B 및 13B)로도 70B 모델과 비슷하거나 우수한 성능을 보였습니다.



### LLM Hallucination Reasoning with Zero-shot Knowledge Tes (https://arxiv.org/abs/2411.09689)
Comments:
          12 pages, 2 figures

- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 환각 문제를 해결하기 위한 새로운 접근법을 제안합니다. 환각이란 LLM이 잘못된 텍스트를 생성하는 현상으로, 다양한 환각 유형 간의 차이를 이해하고 이를 분류하는 'Hallucination Reasoning' 작업을 도입했습니다. 이 작업은 텍스트를 aligned, misaligned, fabricated 세 가지 유형으로 분류합니다.

- **Technical Details**: 논문에서 제안한 새로운 제로샷(zero-shot) 방법인 MKT는 LLM이 주어진 텍스트와 프롬프트에 대한 충분한 지식을 보유하고 있는지를 평가합니다. 이 과정에서 외부 지식이나 레이블이 있는 데이터셋, LLM의 파인튜닝(fine-tuning) 없이도 환각 유형을 구별할 수 있습니다. 두 단계의 워크플로우(workflow)를 통해 환각 reasoning을 수행하며, 이는 LLM의 특정 지식 부족을 식별하는 데 중점을 둡니다.

- **Performance Highlights**: 실험 결과, 제안된 MKT 방법이 기존 QA와 자유 형식 텍스트 생성 작업에서 우수한 성능을 나타냄을 보여주었습니다. 또한, 우리의 방법을 기존의 환각 탐지 알고리즘에 통합했을 때 성능이 크게 향상되어, 환각 reasoning의 중요성이 강조되었습니다.



### On the Limits of Language Generation: Trade-Offs Between Hallucination and Mode Collaps (https://arxiv.org/abs/2411.09642)
Comments:
          Abstract shortened to fit arXiv limit

- **What's New**: 본 연구는 언어 모델의 생성에서 '일관성(consistency)'과 '폭(breadth)'을 동시에 충족할 수 있는지에 대한 문제를 다루고 있습니다. 이 문제에 대한 기존 연구와 반대의 결과를 제시하며, 언어 모델의 난제 중 하나인 생성의 폭과 일관성 간의 긴장 관계를 수학적으로 설명합니다.

- **Technical Details**: 알고리즘은 무한 집합으로부터 랜덤 샘플을 받아, 이를 바탕으로 모든 보이지 않는 문자열을 생성하는 것을 목표로 합니다. 이 과정에서 잘 정의된 언어 K 내에서 유의미한 데이터 분포를 학습해야 하며, 이는 기존 GAN(Generative Adversarial Networks)에서의 mode collapse 문제와 관련이 있습니다. 더불어, 샘플 수가 증가할 때 출력이 K의 모든 보이지 않는 문자열에 수렴하는지를 분석합니다.

- **Performance Highlights**: 연구 결과에 따르면, 많은 언어 모델에서는 '일관성'과 '폭'을 동시에 달성하는 것이 불가능하다는 것이 밝혀졌습니다. 그러나 긍정적인 예제와 부정적인 예제(정답 외의 문자열)가 함께 제공될 때 카운팅 가능한 모든 언어 집합에 대해 일관성 있는 폭 있는 생성을 달성할 수 있는 가능성을 제시합니다. 이는 히스토그램 피드백이 환각(hallucination)을 줄이는 데 중요한 역할을 할 수 있음을 시사합니다.



### Initial Nugget Evaluation Results for the TREC 2024 RAG Track with the AutoNuggetizer Framework (https://arxiv.org/abs/2411.09607)
- **What's New**: 이번 보고서는 TREC 2024 Retrieval-Augmented Generation (RAG) Track의 부분 결과를 처음으로 살펴봅니다. RAG 평가가 정보 접근 및 인공지능 분야에서 지속적인 발전에 장애가 되고 있다고 판단하며, 이를 해결하기 위한 다양한 도전에 대한 기여를 희망합니다. 주요 가설은 2003년 TREC Question Answering Track을 위해 개발된 nugget 평가 방법론이 RAG 시스템 평가에 적합하다는 것입니다.

- **Technical Details**: RAG 시스템 평가를 위해 AutoNuggetizer 프레임워크를 구현했습니다. 이 프레임워크는 대형 언어 모델(LLMs)을 활용하여 자동으로 nugget을 생성하고 시스템 응답에 nugget을 자동으로 할당합니다. TREC 설정 내에서 이 완전 자동화된 프로세스를 준수하여 인간 평가자에 의해 반수동으로 생성된 nugget과 비교하고 조정합니다. 본 보고서는 45개의 실행에서 21개의 주제에 걸친 초기 결과를 바탕으로 자동화된 nugget 평가 결과가 수동 평가와 높은 상관관계를 가지는 것으로 나타났습니다.

- **Performance Highlights**: 초기 결과에 따르면, 완전 자동화된 nugget 평가와 대부분 수동으로 수행된 nist assessors의 평가 점수 사이에 강한 상관관계를 발견했습니다. 이는 우리의 완전 자동화된 평가 프로세스가 향후 RAG 시스템 반복을 안내하는 데 사용될 수 있음을 시사합니다.



### LLaMA-Mesh: Unifying 3D Mesh Generation with Language Models (https://arxiv.org/abs/2411.09595)
Comments:
          See the project website at this https URL

- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)을 활용하여 텍스트 기반 3D 메쉬를 생성하는 새로운 방법인 LLaMA-Mesh를 제안합니다. 이 접근법은 LLMs의 기존 공간적 지식을 활용하고, 대화형 3D 생성 및 메쉬 이해를 가능하게 합니다.

- **Technical Details**: LLaMA-Mesh는 3D 메쉬의 정점 좌표와 면 정의를 일반 텍스트로 표현하여 LLM과의 통합을 용이하게 합니다. 이를 위해, OBJ 파일 형식을 사용하며, 기존의 토크나이저를 확장하지 않고 새로운 데이터 형식을 처리합니다. 또한, 감독형 미세 조정(SFT) 데이터셋을 만들어 LLM이 3D 메쉬를 이해하고 생성할 수 있도록 학습시킵니다.

- **Performance Highlights**: LLaMA-Mesh는 사전 훈련된 LLM이 3D 메쉬를 텍스트 프롬프트로부터 생성하고, 텍스트와 3D 메쉬를 혼합하여 출력할 수 있는 능력을 보여줍니다. 제안된 방법은 모델을 처음부터 훈련한 모델과 동등한 품질의 메쉬 생성을 달성하면서도 강력한 텍스트 생성 성능을 유지합니다.



### Communication Compression for Tensor Parallel LLM Inferenc (https://arxiv.org/abs/2411.09510)
- **What's New**: 대규모 언어 모델(LLMs)의 속도 향상을 위해, 본 연구에서는 Tensor Parallel 전략을 이용한 다수의 하드웨어 가속기에서의 모델 배치를 다루며, 활발한 협업을 통한 지연 시간을 줄이는 방법을 제안합니다.

- **Technical Details**: 최적화된 inter-accelerator communication을 위해 활동량을 3.5 - 4.5배 압축하는 세밀한 양자화 기법을 활용했습니다. 이 방법은 타임 투 퍼스트 토큰(TTFT)을 최대 2배 줄이는 결과를 가져오면서도 모델 성능 저하는 거의 없는 수준입니다.

- **Performance Highlights**: 기존 하드웨어 세팅에서 실행할 때, 느린 inter-accelerator 대역폭 환경에서의 TTFT가 3.5 - 4.5배 향상될 수 있음을 발견했습니다.



### Robot Tasks with Fuzzy Time Requirements from Natural Language Instructions (https://arxiv.org/abs/2411.09436)
Comments:
          9 pages, 8 figures, to be published in 2024 IEEE International Conference on Robotic Computing (IRC)

- **What's New**: 자연어를 활용한 로봇 프로그래밍의 접근성을 높이는 작업을 진행합니다. 특히, '몇 분 내에 시작'과 같은 불확실한 시간 요구 사항을 가진 지시를 다루고, 이러한 요구 사항을 충족시키기 위해 사용자의 만족도를 기반으로 한 만족 함수(satisfaction function)를 도입합니다.

- **Technical Details**: 우리는 로봇이 특정 작업을 수행할 때 발생하는 다양한 시간 요건을 모델링 하는 방법론을 제시합니다. 사용자의 만족도를 수치화하고, 이를 기반으로 로봇의 행동 일정을 최적화하는 도구인 만족 함수를 설계했습니다. 이를 통해 여러 개의 불확실한 기술(fuzzy skills)이 동시에 요청될 때, 최적의 수행 시점을 결정할 수 있습니다.

- **Performance Highlights**: 사용자 연구 결과, 잡음이 있는 시간 요구 사항을 효과적으로 처리할 수 있는 만족도 함수의 모델링을 통해 로봇의 작업 수행 시 사용자 만족도가 증가하는 경향이 확인되었습니다. 특히, 시간적 여유를 두고 실행될 수 있는 경우 사용자 만족도가 더 높아지는 경향을 보였습니다.



### Less is More: Unseen Domain Fake News Detection via Causal Propagation Substructures (https://arxiv.org/abs/2411.09389)
Comments:
          9 pages, 2 figures, 5 tables

- **What's New**: 본 논문에서는 인-distribution(내적 분포) 데이터에서 인과 서브그래프(causal subgraphs)를 추출하여 zero-shot(제로샷) 가짜 뉴스 탐지를 개선하기 위한 Causal Subgraph-oriented Domain Adaptive Fake News Detection (CSDA) 모델을 제안합니다.

- **Technical Details**: CSDA 모델은 그래프 신경망(graph neural network)을 기반으로 한 마스크 생성 과정을 통해 전파 그래프(propagation graph) 내의 주요 노드와 엣지를 식별하고, 이를 가짜 뉴스 탐지에 활용합니다. 모델은 이진 마스크를 사용하여 각 노드와 엣지를 인과적 요소(causal elements) 또는 편향 요소(biased elements)로 분류합니다. 또한, 제한된 OOD 데이터에서의 few-shot(소수 샷) 상황에서 대조 학습(contrastive learning)을 통해 성능을 개선합니다.

- **Performance Highlights**: CSDA는 공개 소셜 미디어 데이터셋에서 OOD 가짜 뉴스 탐지를 수행하며, 다른 최신 모델들 대비 7%에서 16%의 정확도 향상을 달성하였습니다.



### Re-Parameterization of Lightweight Transformer for On-Device Speech Emotion Recognition (https://arxiv.org/abs/2411.09339)
- **What's New**: 이번 연구에서는 Transformer 모델의 lightweight 버전의 성능을 향상시키기 위해 새로운 재파라미터화( re-parameterization) 방법을 제안합니다. 이 방법은 모델 크기와 계산 작업을 늘리지 않으면서도 성능을 크게 개선할 수 있는 특징을 가지고 있습니다.

- **Technical Details**: 제안된 Transformer re-parameterization 접근 방식은 두 가지 과정으로 구성됩니다. 첫 번째는 High-Rank Factorization (HRF) 과정으로, 훈련 단계에서 가벼운 Transformer의 Feed-Forward Network (FFN) 앞에 추가 선형 레이어를 삽입합니다. 두 번째는 de-High-Rank Factorization (deHRF) 과정으로, 추론 단계에서 추가된 HRF 레이어와 FFN 레이어를 병합하여 원래 경량 모델의 구조를 복원합니다.

- **Performance Highlights**: 제안된 방법은 ConvTransformer, Conformer, SpeechFormer 네트워크와 같은 세 가지 널리 사용되는 Transformer 변형에서 평가되었으며, IEMOCAP, M3ED, DAIC-WOZ 데이터셋에서 수행된 음성 감정 인식(SER) 작업에서 일정하게 성능을 향상시켰습니다. 심지어 제안된 경량 Transformer는 대형 모델과 비교할 수 있을 정도의 성능을 달성했습니다.



### Jailbreak Attacks and Defenses against Multimodal Generative Models: A Survey (https://arxiv.org/abs/2411.09259)
Comments:
          ongoing work

- **What's New**: 최근 다중 모달 기초 모델(Multimodal Foundation Models)의 급속한 발전으로 인해 텍스트, 이미지, 오디오 및 비디오를 아우르는 다양한 모달리티에서의 교차 모달 이해 및 생성(CG)을 위한 큰 발전이 있었습니다. 그러나 이러한 모델은 내장된 안전 메커니즘을 우회할 수 있는 jailbreak 공격에 취약한 문제가 있습니다. 본 논문은 이러한 문제를 다루며, 다중 모달 생성 모델에서의 jailbreak 공격 및 방어 메커니즘을 포괄적으로 리뷰합니다.

- **Technical Details**: 이 논문은 데이터의 유형(텍스트, 이미지, 오디오, 비디오)을 통합 및 처리하여 복잡한 상호작용 공간을 만들어내는 다중 모달 생성 모델의 공격 및 방어 전략을 입력(input), 인코더(encoder), 생성기(generator), 출력(output) 네 가지 수준에서 체계적으로 탐구합니다. 이를 통해 각 모델이 갖고 있는 독특한 아키텍처 내의 공통적인 취약성을 공유하는 네 가지 주요 단계에 초점을 맞춥니다.

- **Performance Highlights**: 이 논문은 기존 공격 방법론과 방어 전략의 포괄적 리뷰를 통해 다중 모달 생성 모델의 jailbreak 공격 및 방어를 위한 일반적인 범주화를 제공합니다. 또한, 다양한 입력-출력 모달리티 및 모델 구조에 걸친 공격, 방어 및 평가 전략에 대한 체계적인 리뷰를 제시하고, 실세계 응용에 대한 한계존, 도전 과제 및 미래 방향에 대해 심도 있게 논의합니다.



### Bridging the Visual Gap: Fine-Tuning Multimodal Models with Knowledge-Adapted Captions (https://arxiv.org/abs/2411.09018)
- **What's New**: 본 연구에서는 비전-언어 모델(vision-language models, VLMs)이 긴 상세한 이미지 캡션에 어떻게 적응하는지를 탐구합니다. 우리는 Decomposed NLI (DNLI)라는 새로운 평가 프레임워크를 제안하여 생성된 캡션을 개별 제안으로 분해하고 각 제안을 독립적으로 평가합니다. 또한, KnowAda라는 데이터 중심의 파인튜닝 기법을 소개하여 VLM이 고유한 지식과 시각적 이해를 활용하여 캡션을 자동으로 조정하도록 합니다.

- **Technical Details**: KnowAda는 세 가지 단계로 구성된 캡션 변환 방법으로, 각 단계에서 VLM의 시각적 지식을 평가하고, 모델이 이해하지 못하는 이미지를 기반으로 한 질문을 생성하여 미지의 내용을 반영하도록 캡션을 수정합니다. DNLI 프레임워크는 캡션의 설명력 및 오류율을 보다 신뢰할 수 있는 방식으로 평가하며, 복잡한 캡션이 모델의 성능에 미치는 영향을 분석합니다.

- **Performance Highlights**: KnowAda는 2억에서 70억 개의 파라미터를 가진 여러 소규모 VLM에서 검증되어, 원본 캡션으로 훈련했을 때보다 환각(hallucinations)을 일관되게 줄이며, 자동 및 인간 평가 모두에서 캡션의 설명력(descriptiveness)과 사실 정확도(factual accuracy) 간의 균형을 잘 제공함을 보여주었습니다.



### Cut Your Losses in Large-Vocabulary Language Models (https://arxiv.org/abs/2411.09009)
Comments:
          Code is available at this https URL

- **What's New**: 이 논문에서는 Cut Cross-Entropy (CCE)라는 새로운 방법론을 제안합니다. 이는 대형 언어 모델(LLM)의 학습 시 기하급수적으로 증가하는 메모리 사용량을 감소시키기 위해 개발되었습니다. CCE는 모든 토큰에 대한 logits를 메모리에 저장하지 않고도 cross-entropy 손실을 계산할 수 있게 해줍니다.

- **Technical Details**: CCE는 정답 토큰에 대한 logit만 계산하고, 모든 logits에 대한 log-sum-exp를 즉석에서 계산하여 메모리 사용량을 줄입니다. 이를 통해 단순한 단일 ground-truth label에 대한 index matrix multiplication과 log-sum-exp 연산으로 cross-entropy 손실을 분해하게 됩니다. CCE 구현을 위해 사용자 정의 CUDA 커널을 활용하여 SRAM에서 매트릭스 곱셈 및 log-sum-exp 축소를 수행합니다.

- **Performance Highlights**: Gemma 2(2B)의 경우, CCE는 손실 계산의 메모리 사용량을 24GB에서 1MB로 줄이고, 분류기 헤드의 총 훈련 시간 메모리 사용량은 28GB에서 1GB로 감소시켰습니다. CCE의 구현은 빠르기를 해치지 않으면서도 메모리 소모를 획기적으로 줄이는 성과를 보여주었습니다.



### Refusal in LLMs is an Affine Function (https://arxiv.org/abs/2411.09003)
- **What's New**: 본 논문에서는 언어 모델의 행동을 조정하는 새로운 접근법인 affine concept editing (ACE)을 제안합니다. 이 방법은 모델의 activation을 직접 조작하여 원하는 출력된 결과를 얻을 수 있도록 합니다. 특히, ACE는 기존의 방향성 절제(directional ablation) 및 activation 추가(activation addition)의 일반화를 통해 모델의 거부 응답을 정밀하게 제어할 수 있습니다.

- **Technical Details**: ACE는 affine decomposition을 기반으로 하여 모델의 activation 벡터를 분석합니다. 이 기법은 고차원 공간에서 개념을 선형적으로 표현하는 'linear representation hypothesis'에 기초하고 있으며, 기존의 기법들보다 보다 일반적인 방법으로 모델의 행동을 조정합니다. 구체적으로, ACE는 기존의 'zero-point' 개념을 확장하여 다양한 reference points에 따라 변동하는 선형 및 상수 항을 포함하는 affine 함수를 사용합니다.

- **Performance Highlights**: 연구 결과에 따르면 ACE는 다양한 프롬프트 유형에 대해 모델의 거부 응답을 보다 일관되게 제어하는 성과를 보여주었습니다. 특히, ACE는 기존 기술이 비일관한 출력을 유발하는 특정 모델에서도 안정적인 결과를 도출하는 것으로 나타났습니다. 이러한 결과는 다른 LLM 기반 스코어링 방법을 통해 평가되었습니다.



### Sparse Upcycling: Inference Inefficient Finetuning (https://arxiv.org/abs/2411.08968)
Comments:
          12 pages, 4 figures, To appear in the 4th NeurIPS Workshop on Efficient Natural Language and Speech Processing (ENLSP), 2024

- **What's New**: 이 연구에서는 사전 훈련된 밀집 모델을 Mixture-of-Experts (MoE) 아키텍처로 변환하는 Sparse Upcycling 방법이 Continued Pretraining (CPT) 보다 어떤지에 대한 비교를 진행했습니다. 실험 결과, 특정 상황에서 Sparse Upcycling이 CPT에 비해 20% 이상의 향상을 이끌어낼 수 있음을 발견하였습니다. 그러나 이 방법은 큰 모델의 경우 40% 느려짐을 초래하는 탐색 비용이 큽니다.

- **Technical Details**: Sparse Upcycling은 밀집 모델의 파라미터 수를 늘리고 질을 향상시키는 방법으로, MoE 아키텍처를 사용하여 모델의 효율성을 높입니다. 각 모델은 436M 및 1.4B 크기로 훈련되었으며, Dense Pretraining과 CPT/Upcycling 두 단계로 나누어 훈련되었습니다. 실험 결과, 업사이클링된 모델이 CPT 보다 낮은 손실(loss)을 달성하여 품질이 개선됨을 보여줍니다.

- **Performance Highlights**: 두 가지 모델 크기에 대해 CPT와 Sparse Upcycling 모델 간의 지연(latency) 및 처리량(throughput)을 비교했습니다. 436M 및 1.4B 크기의 모델에서 업사이클링된 모델은 CPT보다 더 나은 품질을 보여주었으나, 처리 속도 측면에서는 상당한 비용이 발생했음을 알 수 있습니다. 이는 특히 대규모 실세계 적용에서의 유용성에 한계를 줄 수 있습니다.



### Quantifying Risk Propensities of Large Language Models: Ethical Focus and Bias Detection through Role-Play (https://arxiv.org/abs/2411.08884)
- **What's New**: 이번 연구는 Large Language Models (LLMs)의 안전성, 윤리 및 편향에 대한 우려가 커지는 가운데, LLM의 윤리적 리스크 태도를 평가하기 위해 Domain-Specific Risk-Taking (DOSPERT) 스케일을 혁신적으로 적용하고, 새로운 Ethical Decision-Making Risk Attitude Scale (EDRAS)을 제안합니다.

- **Technical Details**: 이 연구에서는 LLM의 위험 성향을 평가하기 위해 DOSPERT 스케일과 역할 놀이를 통합한 혁신적인 접근 방식을 사용하였습니다. 특히, 여러 주요 LLM의 윤리적 영역에서 위험 성향을 평가하고, 시스템적 편향을 정량적으로 분석했습니다.

- **Performance Highlights**: 연구 결과, 각 LLM은 차별화된 위험 성향을 보였으며, LLM의 윤리적 리스크 성향 평가에 EDRAS가 효과적으로 적용될 수 있다는 것을 보여주었습니다. 또한, 역할 가설을 이용한 위험 태도 탐색을 통해 다양한 사회 집단에 대한 시스템적 편향을 탐지할 가능성을 제시했습니다.



New uploads on arXiv(cs.IR)

### Initial Nugget Evaluation Results for the TREC 2024 RAG Track with the AutoNuggetizer Framework (https://arxiv.org/abs/2411.09607)
- **What's New**: 이번 보고서는 TREC 2024 Retrieval-Augmented Generation (RAG) Track의 부분 결과를 처음으로 살펴봅니다. RAG 평가가 정보 접근 및 인공지능 분야에서 지속적인 발전에 장애가 되고 있다고 판단하며, 이를 해결하기 위한 다양한 도전에 대한 기여를 희망합니다. 주요 가설은 2003년 TREC Question Answering Track을 위해 개발된 nugget 평가 방법론이 RAG 시스템 평가에 적합하다는 것입니다.

- **Technical Details**: RAG 시스템 평가를 위해 AutoNuggetizer 프레임워크를 구현했습니다. 이 프레임워크는 대형 언어 모델(LLMs)을 활용하여 자동으로 nugget을 생성하고 시스템 응답에 nugget을 자동으로 할당합니다. TREC 설정 내에서 이 완전 자동화된 프로세스를 준수하여 인간 평가자에 의해 반수동으로 생성된 nugget과 비교하고 조정합니다. 본 보고서는 45개의 실행에서 21개의 주제에 걸친 초기 결과를 바탕으로 자동화된 nugget 평가 결과가 수동 평가와 높은 상관관계를 가지는 것으로 나타났습니다.

- **Performance Highlights**: 초기 결과에 따르면, 완전 자동화된 nugget 평가와 대부분 수동으로 수행된 nist assessors의 평가 점수 사이에 강한 상관관계를 발견했습니다. 이는 우리의 완전 자동화된 평가 프로세스가 향후 RAG 시스템 반복을 안내하는 데 사용될 수 있음을 시사합니다.



### MARM: Unlocking the Future of Recommendation Systems through Memory Augmentation and Scalable Complexity (https://arxiv.org/abs/2411.09425)
Comments:
          Work in progress

- **What's New**: 이 논문은 RecSys(Recommendation System)의 스케일링 법칙(scaling-law)을 제시하며, 기존 NLP(자연어처리) 모델 설정과는 다른 점을 명확히 하였습니다.

- **Technical Details**: MARM(Memory Augmented Recommendation Model)은 RecSys의 성능을 향상시키기 위해 새로운 캐시 스케일링 법칙을 탐구합니다. 연속적으로 트레이닝된 데이터 샘플의 양은 50억 개가 넘으며, 수백억 개의 파라미터를 가진 시스템으로 설계되었습니다. MARM은 복잡한 마스크된 자기 주의 모듈의 계산 결과를 캐시하여 시간 복잡성을 줄이고, 다층 주의 모델링을 단순하게 변형했습니다.

- **Performance Highlights**: 우리의 MARM 프레임워크는 기존의 모델보다 평균 0.43% GAUC 개선과 온라인 사용자 당 2.079% 플레이 타임 향상을 보여주었습니다. MARM은 효율성을 크게 높이며 기존의 고성능 Transformer 기반 모델에 쉽게 통합될 수 있습니다.



### LLM-assisted Explicit and Implicit Multi-interest Learning Framework for Sequential Recommendation (https://arxiv.org/abs/2411.09410)
Comments:
          10 pages

- **What's New**: 이번 연구는 사용자 행동 데이터와 텍스트의 의미적 정보를 조합하여 더 정확한 사용자 관심 모델링을 위한 새로운 프레임워크(EIMF)를 제안합니다. 이 프레임워크는 사용자 행동과 의미적 두 가지 층에서 사용자의 관심을 모델링합니다.

- **Technical Details**: EIMF는 두 가지 모듈로 구성됩니다: Implicit Behavioral Interest Module (IBIM)과 Explicit Semantic Interest Module (ESIM). IBIM은 사용자 아이템 간의 상호작용을 통해 암묵적인 행동 관심을 학습하며, ESIM은 클러스터링 알고리즘을 사용하여 대표 샘플을 선택하고 LLM을 활용해 명시적 의미 관심을 추출합니다. 훈련 단계에서는 다중 작업 학습을 통해 의미 정보를 행동 관심 표현과 결합합니다.

- **Performance Highlights**: 실제 데이터셋에 대한 광범위한 실험을 통해 EIMF가 추천 성능을 크게 향상시킬 수 있음을 보여주었고, 일반화 능력이 뛰어난 결과를 보였습니다.



### Harnessing multiple LLMs for Information Retrieval: A case study on Deep Learning methodologies in Biodiversity publications (https://arxiv.org/abs/2411.09269)
- **What's New**: 본 연구에서는 5개의 오픈 소스 Large Language Models (LLMs)를 사용하여 과학적 출판물에서 Deep Learning (DL) 방법론 정보를 자동으로 추출하는 새로운 접근 방식을 제안합니다. 이 방식을 통해 DL 연구의 투명성과 재현성을 높이는 것을 목표로 합니다.

- **Technical Details**: 연구에서는 Llama-3, Mixtral, Gemma와 같은 다양한 LLM을 활용하며, Retrieval-Augmented Generation (RAG) 방식을 통해 DL 관련 정보를 자동으로 추출합니다. 이 과정에서 다수의 LLM 출력을 조합한 투표 분류기를 개발하여 정확한 정보 전달을 도모합니다.

- **Performance Highlights**: 우리의 접근 방식을 통해 DL 방법론 정보를 텍스트 내용만으로 대조했을 때 69.5%의 정확도를 달성했습니다. 이 결과는 DL 방법론 보고의 투명성을 높이고, 과학적 연구에서의 정보 재현성을 향상시키는 데 기여할 것으로 기대됩니다.



### DeBaTeR: Denoising Bipartite Temporal Graph for Recommendation (https://arxiv.org/abs/2411.09181)
- **What's New**: 이 논문에서는 명시적 사용자 피드백의 부족으로 인해 대체 데이터 출처로서의 암시적 피드백의 문제를 다루고, 시간 정보를 활용하여 추천 시스템의 노이즈를 제거하고 예측 성능을 향상시키는 두 가지 방법인 DeBaTeR-A와 DeBaTeR-L을 제안합니다.

- **Technical Details**: 사용자-아이템 상호작용은 이분 그래프(bipartite graph)로 모델링되며, 시간 정보를 활용하여 사용자의 특성과 아이템의 특성을 고려한 시간을 인식하는 사용자/아이템 임베딩을 생성합니다. DeBaTeR는 인접 행렬의 가중치를 재조정하거나 손실 함수의 가중치를 재조정하여 노이즈 상호작용을 식별하고 제거합니다.

- **Performance Highlights**: 광범위한 실험을 통해 제안된 방법들이 최신 모델에 비해 우수한 성능과 견고성을 보여주었으며, 실제 데이터셋에서 높은 정확도를 기록하였습니다.



### Language-Model Prior Overcomes Cold-Start Items (https://arxiv.org/abs/2411.09065)
Comments:
          This paper is dedicated to cold-start item recommendation using language-model priors

- **What's New**: 이 논문은 추천 시스템에서 아이템의 콜드 스타트 문제를 해결하기 위해 언어 모델(Language Model, LM)을 활용하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 제안된 프레임워크는 Bayesian 정규화를 RecSys의 훈련 과정에 통합하여 아이템 간의 의미적 유사성을 고려합니다. LM이 인코딩한 사전 지식을 활용하여 연속 잠재 공간에서 세밀한 아이템 임베딩을 학습합니다.

- **Performance Highlights**: 실험을 통해 SASRec 및 BPRMF와 같은 추천 시스템에 통합하여 두 개의 실제 데이터 세트에서 평가한 결과, 제안된 접근 방식이 SASRec의 정규화 할인 누적 이익(Normalized Discounted Cumulative Gain)을 17.78% 향상시켰습니다.



### Information Need in Metaverse Recordings - A Field Study (https://arxiv.org/abs/2411.09053)
Comments:
          12 pages, 3 Figures, 8 Tables

- **What's New**: 본 논문은 Metaverse Recordings (MVRs)라는 새로운 미디어 유형을 다루며, 사용자 정보 요구 및 MVR 검색 행동을 파악하기 위해 실시한 현장 연구의 결과를 제시합니다. MVR의 기존 애플리케이션 시나리오 및 검색 문제를 강조하고, MVR 검색 시스템 개발을 위한 기초를 제공합니다.

- **Technical Details**: MVR은 Multimedia Information Retrieval (MMIR) 영역 내에서 높은 사용성을 보이며, 그래픽 렌더링 프로세스에서 수집된 시계열 데이터와 관련된 입력-출력 장치와의 관계를 통해 사용자 요구와의 연결성을 확보합니다. 연구는 사용자 고유의 검색 행동과 정보 탐색 동기를 이해하기 위한 다양한 저자 인터뷰를 기반으로 진행되었습니다.

- **Performance Highlights**: 연구 결과 MVR의 다양한 응용 가능성을 확인하였으며, MVR 검색 시스템의 설계에 있어 필요한 사용자 스테레오타입과 구체적 요구 사항을 정의했습니다. 또한, 기존 기술과 사용자 관심을 반영한 정보 검색 사용자 행동에 대한 이해를 높였으며, 향후 연구 및 시스템 설계를 위한 발판을 마련하였습니다.



### Calibrated Decision-Making through LLM-Assisted Retrieva (https://arxiv.org/abs/2411.08891)
- **What's New**: 이번 논문에서는 Calibrated Retrieval-Augmented Generation (CalibRAG)이라는 새로운 검색 방법을 제안하여, LLM이 제공하는 정보의 신뢰성을 높이고 잘 조율된 결정-making을 지원합니다.

- **Technical Details**: CalibRAG는 외부 문서에서 정보를 검색하는 전통적인 Retrieval Augmented Generation (RAG) 방법보다 한 단계 발전하여, 사용자의 결정이 잘 조율되도록 보장합니다. 또한 예측 기능(forecasting function)을 활용하여, RAG에서 제공하는 정보에 대한 신뢰도를 적절하게 반영합니다.

- **Performance Highlights**: 실험 결과, CalibRAG는 다양한 데이터셋에서 기존의 불확실성 조정 기준선과 비교하여 정확도와 조율(performance calibration) 성능 향상이 입증되었습니다.



### KisanQRS: A Deep Learning-based Automated Query-Response System for Agricultural Decision-Making (https://arxiv.org/abs/2411.08883)
- **What's New**: 이 논문에서는 Kisan Query Response System (KisanQRS)라는 농업 분야를 위한 딥러닝 기반의 견고한 쿼리-응답 프레임워크를 소개합니다. 농부들이 신속하게 정보와 지침을 받을 수 있도록 도와줍니다.

- **Technical Details**: KisanQRS는 농부의 쿼리에 대한 의미적 및 어휘적 유사성을 통합하고 빠른 임계값 기반 군집화(clustering) 방법을 사용합니다. 군집화 알고리즘은 쿼리를 반복적으로 돌아보는 선형 검색(linear search) 기법을 기반으로 하며, LSTM(Long Short-Term Memory) 모델이 쿼리 매핑에 최적의 방법으로 발견되었습니다. 제안된 답변 검색 방법은 작물에 대한 후보 답변을 군집화하고, 군집 내의 답변 수에 따라 답변 군집을 순위매기며 각 군집의 리더를 선택합니다.

- **Performance Highlights**: KisanQRS는 인도 정부가 운영하는 Kisan Call Centre (KCC)의 3,400만 통화를 포함한 데이터 셋을 기반으로 하며, 30만 샘플에 대한 쿼리 매핑 모듈의 성능 평가에서 주(state)별로 96.58%의 높은 F1-score를 달성했습니다. 답변 검색 모듈은 10,000 샘플에서 평가되며 96.20%의 경쟁력 있는 NDCG 점수를 기록했습니다. 이를 통해, KisanQRS는 농부들이 그들의 농업 관행에 대해 정보에 기반한 결정을 내릴 수 있도록 신속하고 관련성 있는 응답을 제공합니다.



### Comprehensive and Practical Evaluation of Retrieval-Augmented Generation Systems for Medical Question Answering (https://arxiv.org/abs/2411.09213)
- **What's New**: 이 연구는 의료 분야에서 질문-답변(QA) 시스템의 신뢰성을 평가하기 위한 새로운 간행물인 Medical Retrieval-Augmented Generation Benchmark (MedRGB)를 소개합니다. MedRGB는 LLMs의 여러 가지 상황에서의 성능을 평가하기 위해 4개의 테스트 시나리오를 포함하고 있습니다.

- **Technical Details**: MedRGB는 3480개의 인스턴스로 구성되어 있으며, 이를 통해 모델의 강도와 취약점을 평가하기 위해 7개의 LLMs를 테스트합니다. 테스트 시나리오는 Standard-RAG, Sufficiency, Integration, Robustness로 나뉘며, 각 시나리오는 LLMs의 정보 통합과 노이즈 처리 능력을 평가합니다.

- **Performance Highlights**: 실험 결과, 현재 모델들은 검색된 문서에서의 노이즈와 잘못된 정보 처리에 한계를 보였습니다. 이 연구는 RAG 시스템의 신뢰성을 높이기 위한 향후 방향에 대한 통찰을 제공합니다.



New uploads on arXiv(cs.CV)

### MagicQuill: An Intelligent Interactive Image Editing System (https://arxiv.org/abs/2411.09703)
Comments:
          Code and demo available at this https URL

- **What's New**: 새로운 이미지 편집 시스템인 MagicQuill을 소개합니다. 이 시스템은 사용자 인터페이스를 간소화하고, 다중 모달 대형 언어 모델(MLLM)을 통해 실시간으로 편집 의도를 미리 예측합니다.

- **Technical Details**: MagicQuill은 세 가지 핵심 모듈인 Editing Processor, Painting Assistor, Idea Collector로 구성되어 있습니다. Editing Processor는 사용자의 의도에 맞게 색상 및 가장자리 조정으로 편집을 구현하며, Painting Assistor는 사용자의 브러시 스트로크를 이해하고 편집 의도를 예측하여 Draw&Guess 작업을 수행합니다. Idea Collector는 직관적인 인터페이스를 제공하여 사용자가 쉽게 아이디어를 입력할 수 있게 합니다.

- **Performance Highlights**: 실험 결과, MagicQuill은 기존 방법들보다 편집 정확도와 효율성을 크게 향상시키며, 사용자 연구에 따르면 Idea Collector는 시스템 사용성 전체에서 다른 인터페이스를 능가했습니다.



### CropCraft: Inverse Procedural Modeling for 3D Reconstruction of Crop Plants (https://arxiv.org/abs/2411.09693)
Comments:
          Preprint

- **What's New**: 본 연구에서는 이미지를 기반으로 농작물의 3D 형태를 재구성하는 혁신적인 방법을 제시합니다. 이 방법은 식물 형태 모델의 매개변수를 최적화하여 완전한 3D 디지털 트윈을 자동으로 생성합니다.

- **Technical Details**: 우리는 먼저 신경 방사장(Neural Radiance Field, NeRF) 기법을 사용하여 장면의 기하학을 추정한 후, Bayesian 최적화를 통해 식물 형태 모델의 매개변수를 추정합니다. 이 과정에서, 잎의 개별 위치는 불필요하게 여겨지고, 대신 작물 캐노피의 집합적 형태 특성을 유지하는 방향으로 최적화됩니다.

- **Performance Highlights**: 이 방법은 실제 농작물 필드를 포함한 다중 뷰 데이터셋에서 검증되었으며, 다양한 성장 단계에서 사실적인 농작물 캐노피를 성공적으로 재구성하고, 농작물 생산성에 대한 직접적인 모니터링 가능성을 보여주었습니다.



### Advancing Fine-Grained Visual Understanding with Multi-Scale Alignment in Multi-Modal Models (https://arxiv.org/abs/2411.09691)
- **What's New**: 본 연구에서는 객체 텍스트, 좌표 및 이미지를 포함하는 다중 스케일(Multi-scale) 지식을 효과적으로 정렬하고 통합하는 새로운 방법론인 세밀한 시각 지식 정렬(Fine-grained Visual Knowledge Alignment) 기법을 제안합니다. 이는 다중 스케일을 통해 모델의 세밀한 시각 이해를 증진시키기 위한 데이터 합성 파이프라인인 Multi-scale Fine-grained Enhancement Data Synthesis Pipeline에 기반을 두고 있습니다.

- **Technical Details**: 이 모델은 세 가지 단계로 훈련됩니다: (1) 객체 및 관계 인식 사전 훈련(Object and Relation Perception Pretraining), (2) 다중 스케일 세밀한 지역적 지식 정렬(Multi-scale Fine-grained Local Knowledge Alignment), (3) 세부적인 글로벌 지식 정렬(Detailed Global Knowledge Alignment). 특히, TinyGroundingGPT는 Compact model을 기반으로 하여 약 3B의 파라미터로 구성되며, 기존의 대형 MLLM과의 성능 비교를 통해 우수성을 입증하고 있습니다.

- **Performance Highlights**: TinyGroundingGPT는 여러 기준에서 탁월한 성능을 발휘하며, 특히 인지적 평가(hallucination evaluation) 및 grounding 작업에서 뛰어난 결과를 보여줍니다. 과거의 대형 모델과 유사한 성능을 유지하면서 저장 공간을 덜 요구하여 실용적입니다.



### Local-Global Attention: An Adaptive Mechanism for Multi-Scale Feature Integration (https://arxiv.org/abs/2411.09604)
- **What's New**: 최근 몇 년 동안 객체 감지(object detection) 분야에서 중요한 진전이 있었으며, 단순한 기술 개선으로도 성능 지표가 상당히 향상되었습니다. 본 논문에서는 Local-Global Attention이라는 새로운 주의 메커니즘(attention mechanism)을 제안하며, 이는 기존의 로컬(local) 및 글로벌(global) 주의의 한계를 극복하기 위해 다중 스케일 컨볼루션(multi-scale convolution)과 위치 인코딩(positional encoding)을 통합하여 최적의 특징 표현(feature representation)을 제공합니다.

- **Technical Details**: Local-Global Attention 메커니즘은 입력 데이터의 세부 정보와 전반적인 맥락을 모두 캡처할 수 있도록 설계되었습니다. 이 메커니즘은 다중 스케일 컨볼루션을 활용하여 세밀한 특징(local features)을 추출하고, 동시에 더 큰 커널을 사용하여 글로벌 특징(global features)을 추출하여 이 둘의 정보를 융합합니다. 또한, 각 주의의 상대적 중요도를 동적으로 조절할 수 있는 학습 가능한 파라미터(learnable parameters)를 도입하여, 특정 작업의 요구사항에 따라 주의의 비율을 최적화합니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋(VOC2007, VOC2012, COCO2017 등)에서 Local-Global Attention의 성능을 평가한 결과, 이 메커니즘은 여러 키 지표에서 기존의 주의 메커니즘보다 지속적으로 뛰어난 성능을 보였으며, 특히 소규모 객체 감지(multi-class and small object detection) 작업에서 강력한 성능을 발휘했습니다. 성능 향상은 계산 효율성을 유지하면서 이루어졌습니다.



### Dynamic Reconstruction of Hand-Object Interaction with Distributed Force-aware Contact Representation (https://arxiv.org/abs/2411.09572)
- **What's New**: ViTaM-D는 분산 촉각 센서를 통합하여 손-객체 상호작용 재구성을 위한 새로운 시각-촉각 프레임워크를 제공합니다. 기존 방식의 한계를 해결하기 위해 DF-Field라는 분산 힘 인식 접촉 표현을 도입하여 동적 접촉 모델링을 향상시킵니다.

- **Technical Details**: ViTaM-D는 두 가지 주요 요소로 구성됩니다: 1) VDT-Net이라는 시각적 동적 추적 네트워크를 사용하여 손-객체 상호작용을 시각 데이터만으로 재구성하고, 2) 힘 인식 최적화 과정(FO)을 통해 DF-Field를 적용하여 접촉 세부 정보를 개선합니다. HOT 데이터셋을 통해 600개의 손-객체 상호작용 시퀀스를 포함하여 정확한 촉각 정보를 제공합니다.

- **Performance Highlights**: ViTaM-D는 DexYCB와 HOT 데이터셋을 기반으로 gSDF 및 HOTrack과 같은 최신 방법들에 비해 상당한 정확도 향상을 보여주었습니다. 특히, 비틀림이나 나쁜 접촉 상태에서의 손 포즈 정제를 효과적으로 수행하여 강체 및 변형체 재구성에서 우수한 성능을 발휘합니다.



### VPBSD:Vessel-Pattern-Based Semi-Supervised Distillation for Efficient 3D Microscopic Cerebrovascular Segmentation (https://arxiv.org/abs/2411.09567)
- **What's New**: 본 논문에서는 고해상도 3D 미세 혈관 이미지의 세분화 문제를 해결하기 위한 새로운 Vessel-Pattern-Based Semi-Supervised Distillation (VpbSD) 파이프라인을 제안합니다. 이 파이프라인은 교사 모델의 사전 훈련 단계에서 다양한 혈관 구조를 캡처하는 vessel-pattern 코드북을 구축하여 노른자위 데이터를 활용합니다.

- **Technical Details**: VpbSD는 교사 모델의 사전 훈련 단계와 이후 지식 전이 및 증류 단계로 구성됩니다. 자가 지도 학습(self-supervised learning)을 사용하여 레이블이 없는 데이터를 통해 다양한 혈관 구조를 캡처한 후, 반 감독 학습(semi-supervised learning)을 접목하여 학생 모델의 정보 획득을 효과적으로 풍부하게 합니다. 또한, 코드북 기반의 지식 증류(feature knowledge distillation) 기술을 통해 모델 크기와 계산 복잡성을 줄이며, 전체 성능을 향상시킵니다.

- **Performance Highlights**: 실제 데이터에 대한 실험 결과, VpbSD는 최신 기법들과 비교했을 때 미세 혈관 세분화에서 뛰어난 효과를 보이며, 다양한 학습 샘플에 대한 노출을 통해 세분화 정확성을 향상시키는 데 성공했습니다.



### Adaptive Deviation Learning for Visual Anomaly Detection with Data Contamination (https://arxiv.org/abs/2411.09558)
Comments:
          Accepted to IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 2025)

- **What's New**: 시각적 이상 감지(Visual Anomaly Detection)를 위해 오염된 데이터에서도 효과적으로 작동할 수 있도록 설계된 새로운 적응형 편차 학습(Adaptive Deviation Learning) 방법을 소개합니다. 이 방법은 각 데이터 인스턴스의 중요도를 동적으로 조정하여 이상 점수(anomaly score)를 학습합니다.

- **Technical Details**: 이 방법은 편차 학습(deviation learning)을 기반으로 하며, 각 인스턴스에 대해 상대적 중요도를 부여하여 이상 점수를 계산하는 체계적인 접근 방식을 채택합니다. 또한 제약 최적화(constrained optimization) 문제를 내포하여 매 미니 배치(mini-batch)마다 인스턴스의 가중치를 업데이트 합니다.

- **Performance Highlights**: MVTec와 VisA 벤치마크 데이터셋에서 수행한 포괄적인 실험 결과, 제안된 방법이 다른 경쟁 기술들을 초과 달성하며 데이터 오염이 있는 환경에서도 안정성과 강 robustness을 보여주었습니다.



### Image Processing for Motion Magnification (https://arxiv.org/abs/2411.09555)
- **What's New**: 본 논문은 Phase-Based Motion Magnification (MM) 기술을 제안하며, 영상의 변화를 증폭하여 육안으로는 보기 힘든 미세한 움직임을 가시화합니다. 이 기술은 Fourier Domain에서 비디오 시퀀스를 분석하고 Fourier Shift Property에 의존하여 구현됩니다.

- **Technical Details**: Phase-Based Motion Magnification은 풀뎃 비디오의 각 프레임 간의 미세한 이동을 추출하며, 이를 통해 영상 속의 불가시적인 움직임이나 색상 변화가 확장되어 나타납니다. 이 알고리즘은 MATLAB에서 구현되어 있으며 Discrete Fourier Transform (DFT)을 통해 수치 처리 방법을 설명하고 있습니다.

- **Performance Highlights**: 초기 실험에서는 합성 이미지(synthetic images)를 이용하여 기본 테스트를 수행하였으며, 이 방법으로 얻은 결과는 비디오에서의 미세한 이동을 강조하여 보여줍니다. 이러한 기술은 의료분야의 호흡 모니터링과 구조 진단 등 다양한 응용 분야에서 활용될 수 있습니다.



### OOD-SEG: Out-Of-Distribution detection for image SEGmentation with sparse multi-class positive-only annotations (https://arxiv.org/abs/2411.09553)
- **What's New**: 이 연구에서는 의료 이미지 세분화(segmentation)에서의 새로운 접근 방식을 제안합니다. 주요 혁신점은 sparse annotated data(희소 주석 데이터)를 사용하여 multi-class 이미지 세분화에서 OOD(detection) 픽셀을 학습할 수 있다는 것입니다. 기존의 background class 없이 positive-only 클래스를 통해 효과적으로 OOD 검출을 수행할 수 있는 프레임워크를 제공합니다.

- **Technical Details**: 제안된 프레임워크는 positive-only learning(양수 전용 학습)을 기반으로 하며, sparse annotations(희소 주석)를 활용하여 픽셀 레벨에서 OOD 픽셀을 탐지합니다. 여기서는 기존 OOD 검출 기법을 세분화 과제에 통합하여 OOD 픽셀을 신뢰성 있게 탐지할 수 있습니다. 크로스-밸리데이션(cross-validation) 전략을 통해 labeled classes를 OOD 데이터로 처리하여 검증합니다.

- **Performance Highlights**: 다양한 multi-class hyperspectral 및 RGB 외과 이미지 데이터셋을 통해 수행된 광범위한 실험에서 제안한 프레임워크의 강건성과 일반화 능력이 입증되었습니다. 특히, 제안된 프레임워크는 기존 OOD 검출 방법과 결합하여 세분화 성능을 향상시키는 데 성공하였습니다.



### MFTIQ: Multi-Flow Tracker with Independent Matching Quality Estimation (https://arxiv.org/abs/2411.09551)
Comments:
          accepted to WACV 2025

- **What's New**: 이번 연구에서 우리는 MFTIQ라는 새로운 밀집 장기 추적 모델을 제안합니다. 이는 Multi-Flow Tracker (MFT) 프레임워크를 발전시켜 비디오 시퀀스의 포인트 수준 시각 추적에서의 과제를 해결합니다. MFTIQ는 독립 품질(Quality, IQ) 모듈을 통합하여 광학 흐름(optical flow) 계산과 일치 품질 추정을 분리함으로써 추적 프로세스의 정확성과 유연성을 크게 향상시킵니다.

- **Technical Details**: MFTIQ는 '플러그 앤 플레이' 방식으로 설계되어, 사용자가 별도의 미세 조정이나 아키텍처 수정 없이 기존의 광학 흐름 메서드를 통합할 수 있도록 합니다. 이 방법은 다양한 광학 흐름 메서드에 일반화 가능하며, 훈련 당시 보지 못했던 광학 흐름 메서드에 대해서도 효과적으로 작동할 수 있습니다. IQ 모듈은 이 occlusion 및 correspondence 품질 추정을 광학 흐름 계산에서 분리하여, 오차누적 없이 직접적으로 추정할 수 있게 합니다.

- **Performance Highlights**: MFTIQ는 TAP-Vid Davis 데이터셋에서 RoMa 광학 흐름을 사용하여 MFT보다 우수할 뿐만 아니라 최신 추적기(state-of-the-art trackers)와 비교할 수 있는 성능을 보여줍니다. 또한 MFTIQ는 가장 느린 광학 흐름 메서드로도 다른 최신 추적기들보다 상당히 빠른 처리 속도를 자랑합니다. 이는 긴 occlusion 상황에서도 신뢰성 있는 궤적 예측을 유지할 수 있도록 해줍니다.



### Prompting the Unseen: Detecting Hidden Backdoors in Black-Box Models (https://arxiv.org/abs/2411.09540)
- **What's New**: 이 연구에서는 black-box 모델 수준에서 백도어(Backdoor) 탐지를 위한 새로운 방법론인 BProm을 제안합니다. 이 방법은 시각적 프롬프트(Visual Prompting, VP)를 활용하여 원본 도메인의 훈련된 모델을 사용하여 타겟 도메인 작업에 적응시키는 기술을 기반으로 합니다.

- **Technical Details**: BProm은 VP를 사용하여 의심스러운 모델에 클린(cleam) 데이터셋을 적용합니다. 모델의 저조한 분류 정확도를 사용하여 백도어가 존재함을 식별합니다. 이 연구에서는 class subspace inconsistency라는 개념을 명확히 하여 감염된 모델과 클린 데이터셋 간의 불일치를 확인합니다.

- **Performance Highlights**: BProm의 성능 검증을 위한 광범위한 실험이 수행되었으며, 이는 의심스러운 모델에서의 백도어 탐지 효과를 입증합니다. 실험 결과, BProm은 백도어가 있는 모델에서 낮은 분류 정확도를 보였음을 나타냈습니다.



### Marker-free Human Gait Analysis using a Smart Edge Sensor System (https://arxiv.org/abs/2411.09538)
Comments:
          accepted for SII 2025

- **What's New**: 이 논문에서는 특정 마커 없이도 걸음 분석(gait analysis)을 수행할 수 있는 새롭고 혁신적인 방법을 제안합니다. 다중 카메라 시스템과 스마트 엣지 센서를 이용하여 3D 신체 자세를 추정하며, 이는 훈련된 딥러닝 네트워크를 통해 개인별 걸음 패턴을 효과적으로 식별합니다.

- **Technical Details**: 논문에서 제안하는 시스템은 25개의 스마트 엣지 센서와 RGB-D 카메라를 배치하여 2D 자세를 추정하고, 이를 융합하여 3D 스켈레톤 모델을 생성합니다. 해당 시스템은 저사양의 모델 아키텍처를 채택하고, 로컬 상에서 동작하여 데이터를 처리합니다. 개인의 걸음 패턴을 식별하기 위해 Triplet Loss를 활용한 시암 네트워크(Siamese network)를 설계하였습니다.

- **Performance Highlights**: 제안된 시스템을 사용하여 다양한 실제 환경에서 고속으로 걸음 분석을 수행할 수 있으며, 병원 임상에서의 환자 및 치료 선택을 돕는 데 있어 유용성을 보여줍니다. 실험을 통해 얻은 결과는 인간 움직임의 수집 및 분석을 자동화하는 데 기여할 가능성이 큽니다.



### Image Matching Filtering and Refinement by Planes and Beyond (https://arxiv.org/abs/2411.09484)
Comments:
          project page: this https URL

- **What's New**: 이 논문은 이미지 매칭(image matching)에서 희소(correspondences)를 필터링 및 정제하는 모듈형(modular) 비딥러닝(non-deep learning) 방법을 소개합니다. 이 방법은 장면 내에서의 모션 흐름이 지역 호모그래피(local homography) 변환으로 근사될 수 있다는 가정을 바탕으로 합니다.

- **Technical Details**: 이 방법은 반복적인 RANSAC(RAndom SAmpling Consensus) 기반 접근 방식을 통해 가상 평면에 대응하는 겹치는 클러스터로 매칭을 집계하며, 비일치하는 대응 관계를 버립니다. 또한 저자들은 패치 재투영 이후 키포인트 위치를 정제할 수 있도록 매칭과 관련된 지역 패치들 간의 명시적 맵을 제공합니다.

- **Performance Highlights**: 제안된 방법은 표준 데이터셋과 이미지 매칭 파이프라인에서 광범위하게 평가되었으며, 최신 딥러닝 방법들과 비교하여 뛰어난 성능을 보였습니다. 특히 이 방법은 카메라 내부 정보(camera intrinsics)가 없는 보다 일반적이고 실제적인 경우를 고려하여 설계되었습니다.



### Renal Cell Carcinoma subtyping: learning from multi-resolution localization (https://arxiv.org/abs/2411.09471)
- **What's New**: 이 연구에서는 Renal Cell Carcinoma의 하위 유형 분류를 위한 새로운 self-supervised training 전략을 제시합니다. 이 방법은 주석이 달린 데이터셋의 필요성을 줄이고, 정확도를 크게 저하하지 않으며, 병리학자의 의사결정 과정을 모방하여 다양한 배율에서 학습된 특성을 통합합니다.

- **Technical Details**: 이 연구의 접근법은 Whole Slide Images (WSIs)를 기반으로 하며, self-supervised learning (SSL) 기법을 활용하여 암 유형을 분류합니다. SSL은 데이터 자체에서 얻은 라벨을 사용하여 semi-automatic 과정으로 supervised pretext task를 수행하며, 복잡한 주석을 요구하지 않습니다.

- **Performance Highlights**: 본 연구에서는 제안된 방법이 ccRCC, pRCC, chRCC 및 ONCO 등 4개의 주요 RCC 하위 유형을 분류하는 데 성공적이며, 기존의 fully supervised 방법들과 유사한 성능을 달성합니다. 이는 대규모 주석 데이터 필요성을 줄이는 동시에 정확도를 유지할 수 있음을 보여줍니다.



### SINETRA: a Versatile Framework for Evaluating Single Neuron Tracking in Behaving Animals (https://arxiv.org/abs/2411.09462)
Comments:
          5 pages, 3 figures, submitted at 2025 IEEE International Symposium on Biomedical Imaging (ISBI)

- **What's New**: 이번 연구에서는 SINETRA라는 시뮬레이터를 개발하여 효율적으로 동물 행동 중의 신경 활동을 추적할 수 있는 합성 데이터 세트를 생성합니다. 이 시뮬레이터는 지형적 배경을 가진 2D 및 3D 비디오를 생성하며, 실제 동물 녹화에서 나타나는 복잡한 움직임을 모방합니다.

- **Technical Details**: SINETRA는 변형 가능한 배경 위에서 입자를 시뮬레이트하여 동작을 모델링합니다. 신경세포의 복잡한 움직임을 재현하기 위해, 시스템 기반의 감쇠 진동자 또는 광학 흐름(Optical Flow) 추정을 사용하여 동물 행동 관련 조직의 변형을 처리합니다. 이를 통해 실시간 플루오레센스 이미징에서 신경세포의 동작을 추적하기 위한 데이터 생성이 가능해집니다.

- **Performance Highlights**: 네 가지 최첨단 추적 알고리즘(eMHT, u-track, KOFT, ZephIR)을 사용하여 SINETRA에서 생성된 합성 데이터로 벤치마크를 수행하였습니다. 결과적으로, 신경세포의 복잡한 움직임을 처리하는 데 있어 기존 방법의 한계를 드러내며, 생물학적 역동계에서의 신경 추적 기술 개발을 위한 새로운 가능성을 제시합니다.



### Long-Tailed Object Detection Pre-training: Dynamic Rebalancing Contrastive Learning with Dual Reconstruction (https://arxiv.org/abs/2411.09453)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 이 논문에서는 기존의 long-tailed distributions에 대한 문제를 해결하기 위한 새로운 pre-training (사전 훈련) 프레임워크인 Dynamic Rebalancing Contrastive Learning with Dual Reconstruction (2DRCL)을 제안합니다. 이 방법은 object detection (객체 탐지)과 일치하는 pre-training을 위한 Holistic-Local Contrastive Learning 메커니즘을 기반으로 합니다.

- **Technical Details**: 2DRCL은 global contextual semantics (전반적인 문맥 의미)와 detailed local patterns (세부적인 지역 패턴)을 포착하며, underrepresented instances (저대표 인스턴스)의 샘플링을 조정하는 동적 재균형 전략을 설계하여 long-tailed data의 불균형 문제를 해결합니다. Dual Reconstruction은 self-consistency principle (자기 일관성 원리)에 따라 재구성 작업을 강제하여 simplicity bias (단순성 편향) 문제를 해결합니다.

- **Performance Highlights**: COCO와 LVIS v1.0 데이터셋에서 실험을 통해 제안된 2DRCL 방법이 tail classes (꼬리 클래스)의 mAP/AP 점수를 특히 향상시키는 데 효과적임을 입증하였습니다. 2DRCL은 기존의 Focal Loss 및 ECM Loss 방법보다 높은 전반적인 평균 정확도(APb)인 26.4%를 달성하였고 rare classes (희귀 클래스)에서도 최고의 성능을 보였습니다.



### Image Regeneration: Evaluating Text-to-Image Model via Generating Identical Image with Multimodal Large Language Models (https://arxiv.org/abs/2411.09449)
- **What's New**: 이 연구에서는 새로운 Image Regeneration 작업을 도입하여 T2I(Text-to-Image) 모델의 성능을 평가하는 방법을 제안합니다. 이전의 평가 방법들은 텍스트 입력과 생성된 이미지 간의 불일치로 인해 신뢰할 수 없는 결과를 야기하였습니다. 이를 해결하기 위해, 제안된 방법은 참조 이미지를 기반으로 이미지를 생성하도록 T2I 모델에 요구하여 결과를 비교합니다.

- **Technical Details**: ImageRepainter 프레임워크를 통해 MLLM(Multi-Modal Large Models)의 도움을 받아 참조 이미지와 텍스트 입력 간의 격차를 해소하고, 이미지의 콘텐츠를 이해할 수 있도록 합니다. 이 프레임워크는 크게 두 단계로 나뉘며: (1) 이미지 이해: MLLM을 사용하여 이미지 정보를 정리하고, 이를 기반으로 텍스트 프롬프트를 생성합니다. (2) 반복 생성: 텍스트-이미지 생성 과정에서 발생하는 반복적인 탐색 과정을 포함합니다.

- **Performance Highlights**: 포괄적인 실험 결과는 ImageRepainter 프레임워크가 T2I 모델의 생성 능력을 효과적으로 평가할 수 있음을 보여주었으며, 강력한 T2M(Task-to-Model) 모델이 참조 이미지와 유사한 이미지를 생성할 수 있음을 입증하였습니다.



### Spider: Any-to-Many Multimodal LLM (https://arxiv.org/abs/2411.09439)
- **What's New**: 본 연구에서는 Spider라는 혁신적인 Any-to-Many Modalities Generation (AMMG) 프레임워크를 소개합니다. 기존의 MLLM(다중 모달 대형 언어 모델)은 'Text + X' 형식의 페어와 모달리티 생성에 한정되었으나, Spider는 'Text + Xs'의 임의 조합을 생성할 수 있습니다.

- **Technical Details**: Spider 프레임워크는 세 가지 핵심 구성 요소로 이루어져 있습니다: 1) Base Model은 기본적인 X-to-X 모달리티 처리를 지원하며, 2) Efficient Decoders-Controller는 여러 모달리티 생성기를 효율적으로 제어하고, 3) Any-to-Many Instruction Template는 다양한 모달리티 신호 프롬프트를 생성합니다.

- **Performance Highlights**: Spider는 새로운 Text-formatted Many-Modal (TMM) 데이터셋을 활용하여 학습되었으며, 이를 통해 최초의 X-to-Xs 다중 모달 데이터셋을 생성하게 됩니다. 이 연구는 AMMG 작업을 발전시키기 위한 풍부한 데이터 지원을 제공합니다.



### ReMP: Reusable Motion Prior for Multi-domain 3D Human Pose Estimation and Motion Inbetweening (https://arxiv.org/abs/2411.09435)
Comments:
          8 main pages, WACV 2025

- **What's New**: 이 논문에서는 Reusable Motion Prior (ReMP)를 제안합니다. ReMP는 다양한 다운스트림 작업에서 움직임의 시간적 변화를 정확하게 추적할 수 있는 효과적인 모션 프라이어이며, 3D 동역학을 캡처하여 여러 센서 모달리티에 적용할 수 있습니다.

- **Technical Details**: ReMP는 모션 시퀀스에서 파라메트릭 모델을 통해 학습된 강력한 모션 prior를 기반으로 하여, 누락된 프레임이나 잡음이 있는 측정을 통해 포즈를 추정하는 데 용이합니다. 이를 위해 temporal attention mechanism을 사용하며, 변형 자동 인코더(VAE) 아키텍처와 temporal transformer를 결합하여 세밀한 모션 변화를 캡처합니다.

- **Performance Highlights**: ReMP는 depth point clouds, LiDAR 스캔, IMU 센서 데이터 등 다양한 3D 모션 데이터에서 기본 방법보다 우수한 성능을 보여줍니다. 또한, ReMP는 불완전하고 도전적인 입력 측정에서도 비현실적인 움직임을 효과적으로 예측하여 훈련 효율성을 크게 향상시킵니다.



### Mediffusion: Joint Diffusion for Self-Explainable Semi-Supervised Classification and Medical Image Generation (https://arxiv.org/abs/2411.09434)
- **What's New**: Mediffusion은 반지도 학습(semi-supervised learning) 및 설명 가능한 분류를 위한 새로운 방법론으로, 조인트 확산 모델(joint diffusion model)을 기반으로 합니다. 의료 이미징 도메인에서의 데이터 라벨링 부족 문제를 해결하고, 높은 성능과 신뢰성을 요구하는 응용 프로그램에 적합한 솔루션을 제공합니다.

- **Technical Details**: Mediffusion은 Denoising Diffusion Probabilistic Models (DDPM)를 사용하여, UNet 구조 내에서 생성(generative) 및 분류(discriminative) 작업을 위해 공유되는 매개변수화(parametrization)를 사용합니다. 이 모델은 라벨된 데이터와 라벨이 없는 데이터를 모두 통해 효과적으로 학습하며, 카운터팩추얼 예제를 이용하여 정확한 설명을 제공합니다.

- **Performance Highlights**: Mediffusion은 최근의 반지도 학습 기술들과 비교할 만한 성능을 달성하면서도, 보다 신뢰할 수 있는 정확한 설명을 제공합니다. 실험 결과, Mediffusion은 설명 가능성과 합성 데이터 샘플링 가능성을 함께 제공하는 최첨단 방법임을 입증했습니다.



### SAG-ViT: A Scale-Aware, High-Fidelity Patching Approach with Graph Attention for Vision Transformers (https://arxiv.org/abs/2411.09420)
Comments:
          10 pages, 4 figures, 3 tables

- **What's New**: 본 논문에서 소개하는 Scale-Aware Graph Attention Vision Transformer (SAG-ViT)는 멀티스케일 특징 표현을 효율적으로 통합하여 image classification을 향상시키는 새로운 프레임워크입니다. EfficientNet을 백본으로 사용하고, 그래프 구조를 통해 특징 맵을 처리함으로써 고급 의미 정보를 유지하며 고유한 관계를 모델링합니다.

- **Technical Details**: SAG-ViT는 EfficientNet을 기반으로 한 멀티스케일 특징 맵을 추출하여 이를 패치로 나누고, 공간적 및 특징 유사성에 따라 그래프를 구성합니다. 그래프 내의 각 노드는 패치를 나타내며, Graph Attention Network (GAT)를 통해 최신의 패치 정보를 동적으로 강조합니다. 이후 Transformer 인코더가 장기 종속성과 복잡한 상호작용을 포착합니다.

- **Performance Highlights**: SAG-ViT는 여러 벤치마크 데이터셋에서 평가되었으며, 기존의 Transformer 기반 접근 방법과 비교하여 이미지 분류 성능을 높이는 데 성공하였습니다.



### Script-centric behavior understanding for assisted autism spectrum disorder diagnosis (https://arxiv.org/abs/2411.09413)
Comments:
          5 pages, 4 figures, submitted to ICASSP 2025

- **What's New**: 이 논문은 비지도 학습을 기반으로 한 새로운 접근 방식을 도입하여 ASD(자폐 스펙트럼 장애)을 자동으로 감지합니다. 기존의 지도 학습 방법에 비해 ASD 진단을 더욱 효과적으로 다룰 수 있는 기술적 진전을 보여줍니다.

- **Technical Details**: 이 방법은 비디오 내용을 스크립트로 변환하는 'Behavioral Transcription Module(BTM)', 스크립트의 행동 데이터를 처리하여 LLMs(Large Language Models)와의 연결을 다리 역할을 하는 'Script Transcription Module(STM)', 그리고 도메인 지식을 결합하는 'Domain Prompts Module(DPM)'의 세 가지 모듈로 구성됩니다.

- **Performance Highlights**: 이 연구에서 제안한 방법은 24개월 된 아동에서 ASD 진단 정확도가 92.00%로, 기존의 지도 학습 방법보다 3.58% 높은 성과를 보여 주었습니다. 이러한 결과는 LLMs의 비디오 데이터에 대한 이해력이 ASD 연구에 중요한 기여를 할 수 있음을 나타냅니다.



### Building Height Estimation Using Shadow Length in Satellite Imagery (https://arxiv.org/abs/2411.09411)
Comments:
          6 pages, 5 figures, 2 tables

- **What's New**: 이번 연구에서는 위성 이미지를 이용한 건물 높이 추정에 있어, 단일 시점 이미지에서 3D 정보를 잃는 문제를 해결하기 위해 그림자의 길이를 추가적으로 이용하는 새로운 방법을 제안했습니다.

- **Technical Details**: 제안된 방법은 먼저 YOLOv7 모델을 통해 건물과 그 그림자를 지역화한 다음, 천체 기하학(geometric) 원리를 활용하여 그림자 길이를 추정합니다. 이 과정을 통해 신뢰할 수 있는 건물 높이를 추정하여, ResNet18을 백본 아키텍처로 사용한 회귀 모델을 통해 최종 높이를 계산합니다.

- **Performance Highlights**: 제안된 프레임워크는 42개의 도시에서 평가되었으며, 기존의 최첨단 방법에 비해 우수한 성능을 보였습니다. 이 연구는 건물 높이 추정 분야의 발전에 기여할 것으로 기대됩니다.



### Instruction-Driven Fusion of Infrared-Visible Images: Tailoring for Diverse Downstream Tasks (https://arxiv.org/abs/2411.09387)
Comments:
          10 pages, 7 figures

- **What's New**: 이번 논문에서는 다중 다운스트림 태스크를 동시에 처리할 때 발생하는 문제를 해결하기 위해 Task-Oriented Adaptive Regulation (T-OAR)이라는 새로운 적응 메커니즘을 제안합니다. 추가로, 사용자 입력 텍스트 지침으로부터 태스크-specific 동적 프롬프트를 생성하는 Task-related Dynamic Prompt Injection (T-DPI) 모듈을 도입하였습니다.

- **Technical Details**: T-OAR 프레임워크는 다중 태스크 환경을 위한 적응형 규제를 제공하며, T-DPI 모듈은 목표 표현에 설치된 태스크 관련 동적 프롬프트를 통합합니다. 이는 feature extraction 모듈이 다운스트림 태스크의 특정 요구사항에 더 잘 맞춘 표현을 생성하도록 유도합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 객체 탐지(object detection), 의미론적 분할(semantic segmentation), 두드러진 객체 탐지(salient object detection)에서 우수한 성능을 보였으며, 다양한 태스크에서 높은 적응성, 유연성 및 태스크 특이성을 입증하였습니다.



### DSCformer: A Dual-Branch Network Integrating Enhanced Dynamic Snake Convolution and SegFormer for Crack Segmentation (https://arxiv.org/abs/2411.09371)
- **What's New**: 이번 연구에서는 새로운 하이브리드 모델인 DSCformer를 제안합니다. 이 모델은 개선된 Dynamic Snake Convolution (DSConv)과 Transformer 아키텍처를 통합하여 콘크리트 구조물의 균열 분할을 보다 정확하게 수행할 수 있도록 합니다.

- **Technical Details**: DSCformer는 피라미드 커널(pyramid kernel)을 통한 적응형 오프셋 계산과 양방향으로 학습 가능한 오프셋 반복을 도입하여 효과적으로 균열 구조를 포착할 수 있습니다. 또한, Weighted Convolutional Attention Module (WCAM)을 통해 채널 주의를 정제하여 더욱 정밀하고 적응적인 특징 주의를 가능하게 합니다.

- **Performance Highlights**: DSCformer는 Crack3238 및 FIND 데이터셋에서 각각 59.22% 및 87.24%의 IoU(Intersection over Union) 성능을 달성했습니다. 이 결과는 기존의 최첨단 방법들을 초월하는 성능을 보여줍니다.



### Time-to-Event Pretraining for 3D Medical Imaging (https://arxiv.org/abs/2411.09361)
Comments:
          34 pages, 19 figures

- **What's New**: 본 연구는 3D 의료 이미징 모델을 위한 새로운 사전 훈련 접근법인 'time-to-event pretraining'을 제안하며, 대규모 전자 건강 기록(EHR) 데이터에서 시간에 따라 발생하는 이벤트 정보를 활용하여 기존의 잃어버린 맥락 문제를 해결합니다.

- **Technical Details**: 이 방법은 18,945개의 흉부 CT 스캔을 포함하는 데이터셋을 사용하여 EHR 데이터로부터 유도된 TTE 분포를 통해 사전 훈련을 수행합니다. 이 과정을 통해 8개의 벤치마크 작업에서 AUROC가 평균 23.7% 증가하고 Harrell의 C-index가 29.4% 향상되었습니다.

- **Performance Highlights**: 기존 진단 분류 성능을 저하시키지 않고도 발생 예측 성능을 크게 향상시키며, 모델 교차 범위(Integrated Brier Score)의 정확도를 평균 54% 개선했습니다. 모든 실험은 공개된 의료 데이터셋을 사용하여 재현 가능성을 보장하였고, 관련 코드를 GitHub에 공개했습니다.



### Adaptively Augmented Consistency Learning: A Semi-supervised Segmentation Framework for Remote Sensing (https://arxiv.org/abs/2411.09344)
- **What's New**: 본 논문은 Adaptively Augmented Consistency Learning (AACL)이라는 새로운 반지도 세분화(segmentation) 프레임워크를 제안합니다. 이 프레임워크는 주석이 적은 데이터 환경에서 원거리 탐지(remote sensing, RS) 세분화 정확도를 높이는 데 초점을 두고 있습니다. AACL은 레이블이 없는 이미지에서 추가 정보를 추출하고 이를 통해 성능을 개선합니다.

- **Technical Details**: AACL은 Uniform Strength Augmentation (USAug)와 Adaptive CutMix (AdaCM)이라는 두 가지 첨단 기법을 활용합니다. USAug는 레이블이 없는 이미지에 일관되지만 다양한 데이터 증강(augmentation)을 적용하여 내재된 정보를 풍부하게 만들고, AdaCM은 모델의 성숙도에 따라 CutMix를 동적으로 적용하여 레이블이 없는 이미지를 보강합니다. 이러한 기법을 통해 AACL은 레이블이 부족한 상황에서도 효과적인 학습이 이루어집니다.

- **Performance Highlights**: AACL은 다양한 RS 데이터셋에서 평가되었으며, 기존의 최첨단 프레임워크에 비해 특정 카테고리에서 최대 20% 향상된 성능을 보였고, 전체 성능에서도 2% 증가를 달성하여 경쟁력을 입증했습니다.



### Exploring Zero-Shot Anomaly Detection with CLIP in Medical Imaging: Are We There Yet? (https://arxiv.org/abs/2411.09310)
Comments:
          accepted at 3rd AIxIA Workshop on Artificial Intelligence for Healthcare and 5th Data4SmartHealth

- **What's New**: 본 논문에서는 CLIP 기반 모델을 이용하여 Zero-shot anomaly detection (ZSAD)을 통해 브레인 종양 탐지 작업을 수행합니다. 이는 의료 영상에서 최소한의 감독으로 이상을 탐지할 수 있는 가능성을 제시하며, 데이터 주석의 한계로 인해 발생하는 여러 도전 과제를 다룹니다.

- **Technical Details**: 연구에서는 CLIP (Contrastive Language-Image Pretraining) 모델을 사용하여, BraTS dataset에서 뇌 전이 종양을 탐지하는 작업을 진행했습니다. CLIP의 투과적 학습 방법은 정형화되지 않은 데이터에서도 일반화 성능을 발휘하지만, 의료 분야 특유의 섬세한 이상을 탐지하는 데에는 한계가 있는 것으로 나타났습니다. 또한, ZSAD는 훈련 데이터 없이 정상 또는 비정상 상태를 설명하는 텍스트 프롬프트를 사용하여 이상 패턴을 식별합니다.

- **Performance Highlights**: CLIP 기반 모델은 산업적 AD 작업에서는 우수한 성능을 보였으나, 의료 영상에서는 아직 임상 적용에 필요한 정밀도에 미치지 못했습니다. 따라서, CLIP 기반 모델이 의료 이상 탐지에 신뢰성 있게 적용될 수 있도록 추가적인 조정이 필요하다는 것을 강조했습니다.



### LHRS-Bot-Nova: Improved Multimodal Large Language Model for Remote Sensing Vision-Language Interpretation (https://arxiv.org/abs/2411.09301)
- **What's New**: 이 연구에서는 LHRS-Bot-Nova라는 원격 탐사(RS) 이미지 이해에 특화된 다중모드 대형 언어 모델(MLLM)을 소개합니다. LHRS-Bot-Nova는 향상된 비전 인코더와 새로운 브릿지 레이어를 갖추고 있어, 시각 정보를 효과적으로 압축하고 언어-비전 정렬을 개선합니다.

- **Technical Details**: LHRS-Bot-Nova는 고해상도 입력을 수용하기 위해 비전 인코더를 확장하고 MoE 아키텍처 기반의 크로스 도메인 맵핑을 위한 브릿지 레이어를 설계하였습니다. 이 구조는 비전 정보를 손실 없이 압축하는 데 도움을 줍니다. 또한, LHRS-Align-Recap이라는 대규모 RS 이미지-캡션 데이터셋을 구축하여 사용자의 명령을 효과적으로 해석할 수 있도록 개선하였습니다.

- **Performance Highlights**: LHRS-Bot-Nova는 다양한 RS 이미지 이해 작업에서 우수한 성능을 보였으며, 다중 선택 질문 평가 기준인 LHRS-Bench를 통해 RS 분야의 다양한 능력을 포괄적으로 평가하는 실험을 수행했습니다. 이를 통해 MLLM의 효율성과 신뢰성을 증진시켰습니다.



### LLV-FSR: Exploiting Large Language-Vision Prior for Face Super-resolution (https://arxiv.org/abs/2411.09293)
- **What's New**: 본 논문은 기존의 얼굴 초해상도(Face Super-Resolution, FSR) 방법의 한계를 극복하기 위해 언어-비전(vision) 복합 표현을 도입하여 FSR 성능을 향상시키는 새로운 프레임워크 LLV-FSR을 제안합니다. LLV-FSR은 고급 언어-비전 모델과 시각적 정보를 조합하여 얼굴 이미지의 품질을 획기적으로 개선합니다.

- **Technical Details**: 이 프레임워크는 기존 입력에서 직접 지식을 흡수하는 것 외에도, 사전 훈련된 언어-비전 모델을 활용하여 이미지 캡션(caption), 설명(description), 얼굴 의미 마스크(semantic mask), 깊이(depth) 정보를 생성합니다. 이러한 정보를 통해 더 중요한 특징 표현을 유도하고, 현실적이고 고품질의 얼굴 초해상도를 달성합니다. 또한, 언어-비전 Priors의 통합 블록을 설계하여 언어와 비전의 상호 보완성을 최대한 활용합니다.

- **Performance Highlights**: 실험 결과 LLV-FSR은 MMCelebA-HQ 데이터셋에서 PSNR(Peak Signal-to-Noise Ratio) 성능을 0.43 dB 향상시켜 SOTA(State-of-the-Art)를 초월했음을 보여줍니다.



### LES-Talker: Fine-Grained Emotion Editing for Talking Head Generation in Linear Emotion Spac (https://arxiv.org/abs/2411.09268)
- **What's New**: 본 논문에서는 기존의 원샷(One-shot) 말하는 얼굴 생성 모델이 감정 수정에서 발전을 이루었으나, 해석 가능성이 높은 세밀한 감정 수정 모델이 부족하다는 점을 강조합니다. 이에 LES-Talker라는 새로운 모델을 제안하며, 이는 감정 유형, 감정 수준 및 얼굴 단위에 걸쳐 세밀한 감정 수정을 가능하게 합니다.

- **Technical Details**: Linear Emotion Space (LES)라는 개념을 도입하여 감정 변화를 벡터 변환으로 정의합니다. Cross-Dimension Attention Net (CDAN)을 설계하여 LES 표현과 3D 모델 표현 간의 상관관계를 깊이 탐구합니다. 이 구조는 다양한 특징과 구조 차원 간의 관계를 발굴하여 LES 표현이 3D 모델의 변형을 제어할 수 있도록 합니다.

- **Performance Highlights**: LES-Talker는 여러 차원에서 세밀하고 변별 가능한 감정 수정을 가능하게 하며, 주류 방법보다 높은 시각 품질을 제공합니다. 실험 결과는 우리 방법이 수많은 감정 유형과 수준에 걸쳐 세밀한 감정 수정을 성공적으로 수행함을 보여줍니다.



### How Good is ChatGPT at Audiovisual Deepfake Detection: A Comparative Study of ChatGPT, AI Models and Human Perception (https://arxiv.org/abs/2411.09266)
- **What's New**: 이 연구에서는 대규모 언어 모델인 ChatGPT의 오디오 및 비디오 콘텐츠에서 딥페이크를 탐지하는 능력을 평가합니다. 기존의 비디오 조작 탐지 방법들과 ChatGPT의 탐지 성능을 비교하고, Prompt Engineering(프롬프트 엔지니어링)의 역할을 강조합니다.

- **Technical Details**: 이 연구는 딥페이크 탐지를 위해 LLMs(대규모 언어 모델)를 활용하는 방법을 제안합니다. 실험은 벤치마크 멀티모달 딥페이크 데이터셋의 비디오를 사용하여 수행하였으며, ChatGPT는 다양한 오디오 및 비주얼 아티팩트(visual and auditory artifacts)를 분석하는 데 적용되었습니다. 제안된 방법은 입력 비디오에 대한 시각적, 청각적 분석을 포함하여 깊이 있는 분석을 제공합니다.

- **Performance Highlights**: 실험 결과 ChatGPT가 멀티모달 딥페이크 탐지에서 인간 및 최신 AI 모델들과 비교하여 경쟁력 있는 성능을 보였으며, 특히 프롬프트 설정에 따라 탐지 성능이 달라지는 것으로 나타났습니다. 그러나 ChatGPT는 탐지 과정의 해석 가능성(interpretability) 부족과 특정 조작에 대한 일반화의 한계를 갖고 있습니다.



### BEARD: Benchmarking the Adversarial Robustness for Dataset Distillation (https://arxiv.org/abs/2411.09265)
Comments:
          15 pages, 6 figures

- **What's New**: BEARD를 소개하여 Dataset Distillation (DD) 방법의 적대적 안전성을 체계적으로 평가하기 위한 통합 벤치마크를 제공한다.

- **Technical Details**: BEARD는 다양한 적대적 공격(FGSM, PGD, C&W 등)과 디스틸 데이터셋(CIFAR-10/100, TinyImageNet 등)에 대한 평가를 포함하며, Robustness Ratio (RR), Attack Efficiency Ratio (AE), Comprehensive Robustness-Efficiency Index (CREI)라는 세 가지 주요 메트릭을 도입한다.

- **Performance Highlights**: BEARD 리더보드에서 얻은 결과는 다양한 상황에서의 DD 기법의 적대적 안전성 평가를 가능하게 하며, 모델 및 데이터셋 풀을 제공하여 재현 가능한 연구를 지원한다.



### Jailbreak Attacks and Defenses against Multimodal Generative Models: A Survey (https://arxiv.org/abs/2411.09259)
Comments:
          ongoing work

- **What's New**: 최근 다중 모달 기초 모델(Multimodal Foundation Models)의 급속한 발전으로 인해 텍스트, 이미지, 오디오 및 비디오를 아우르는 다양한 모달리티에서의 교차 모달 이해 및 생성(CG)을 위한 큰 발전이 있었습니다. 그러나 이러한 모델은 내장된 안전 메커니즘을 우회할 수 있는 jailbreak 공격에 취약한 문제가 있습니다. 본 논문은 이러한 문제를 다루며, 다중 모달 생성 모델에서의 jailbreak 공격 및 방어 메커니즘을 포괄적으로 리뷰합니다.

- **Technical Details**: 이 논문은 데이터의 유형(텍스트, 이미지, 오디오, 비디오)을 통합 및 처리하여 복잡한 상호작용 공간을 만들어내는 다중 모달 생성 모델의 공격 및 방어 전략을 입력(input), 인코더(encoder), 생성기(generator), 출력(output) 네 가지 수준에서 체계적으로 탐구합니다. 이를 통해 각 모델이 갖고 있는 독특한 아키텍처 내의 공통적인 취약성을 공유하는 네 가지 주요 단계에 초점을 맞춥니다.

- **Performance Highlights**: 이 논문은 기존 공격 방법론과 방어 전략의 포괄적 리뷰를 통해 다중 모달 생성 모델의 jailbreak 공격 및 방어를 위한 일반적인 범주화를 제공합니다. 또한, 다양한 입력-출력 모달리티 및 모델 구조에 걸친 공격, 방어 및 평가 전략에 대한 체계적인 리뷰를 제시하고, 실세계 응용에 대한 한계존, 도전 과제 및 미래 방향에 대해 심도 있게 논의합니다.



### Embedding Space Allocation with Angle-Norm Joint Classifiers for Few-Shot Class-Incremental Learning (https://arxiv.org/abs/2411.09250)
- **What's New**: 본 논문은 Few-Shot Class-Incremental Learning(FSCIL)에서의 샘플 부족 문제를 해결하기 위해 Class-Center Guided Embedding Space Allocation with Angle-Norm Joint Classifiers (SAAN) 학습 프레임워크를 제안합니다. 이 프레임워크는 새로운 클래스 학습을 위한 공간을 효과적으로 할당하며, 기존 메서드의 문제점을 해결합니다.

- **Technical Details**: SAAN은 두 가지 주요 구성 요소인 Class-Center Guided Embedding Space Allocation (CCSA)와 Angle-Norm Joint Classifiers (ANJ)로 구성됩니다. CCSA는 특성 공간을 여러 하위 공간으로 나누고, 각 세션을 위한 전용 하위 공간을 샘플의 카테고리 중심으로 안내합니다. ANJ는 각 클래스의 norm 분포를 설정하여 각 클래스의 norm logit을 추정하고, NCM과 결합하여 각도-logit을 생성합니다.

- **Performance Highlights**: 실험 결과, SAAN는 SOTA 성능에 근접한 성과를 달성했으며, 기존 SOTA 방법의 성능을 더욱 향상시킬 수 있다는 것을 확인했습니다. 특히, SAAN은 세 가지 데이터 세트와 두 가지 방법에서 마지막 단계의 정확도를 3% 이상 향상시켰습니다.



### Harnessing Vision Foundation Models for High-Performance, Training-Free Open Vocabulary Segmentation (https://arxiv.org/abs/2411.09219)
Comments:
          12 pages, 5 figures

- **What's New**: 본 논문은 CLIP의 한계를 극복하기 위한 새로운 접근 방식을 제안합니다. 기존의 Segment-then-Splice 방식과는 달리, Splice-then-Segment 패러다임을 도입하여 고해상도 이미지에서 의미론적 세분화를 개선하는 데 중점을 두고 있습니다.

- **Technical Details**: Trident라는 새로운 프레임워크를 소개하며, CLIP과 DINO로부터 추출된 특성 맵을 분할하고, SAM의 인코더를 활용하여 상관 행렬(correlation matrix)을 생성합니다. 이를 통해 전역 집합(global aggregation)을 가능하게 하여 효과적인 세분화를 달성합니다. 또한, CLIP의 조잡한 세분화 출력을 향상시키기 위한 정제 전략(refinement strategy)을 제안하여 SAM의 프롬프트로 변환합니다.

- **Performance Highlights**: Trident는 8개의 기준에서 기존 SOTA(State Of The Art) 결과를 크게 초월하며, mIoU(Mean Intersection over Union)가 44.4에서 더 높은 성과를 달성합니다. 이는 학습이 필요 없는 방법들 중에서도 SOTA 성능을 기록하고 있음을 보여줍니다.



### JoyVASA: Portrait and Animal Image Animation with Diffusion-Based Audio-Driven Facial Dynamics and Head Motion Generation (https://arxiv.org/abs/2411.09209)
- **What's New**: 본 논문에서는 JoyVASA라는 새로운 방법을 제안합니다. JoyVASA는 오디오 기반의 얼굴 애니메이션에서 얼굴 다이나믹스와 머리 움직임을 생성하는 diffusion-based 방법입니다. 이 방법은 정적 3D 얼굴 표현을 동적 표정과 분리하여 장시간의 영상을 생성할 수 있게 합니다.

- **Technical Details**: JoyVASA는 두 단계로 나뉘어 있습니다. 첫 번째 단계에서는 정적 3D 얼굴 표현과 동적 얼굴 표현을 분리하여, 이를 통해 제너레이터가 긴 비디오를 생성할 수 있도록 합니다. 두 번째 단계에서는 diffusion transformer가 오디오 신호에 기반하여 직접적으로 동작 시퀀스를 생성합니다. 이 과정은 캐릭터의 정체성과 독립적으로 이루어집니다.

- **Performance Highlights**: JoyVASA는 인간의 초상화뿐만 아니라 동물의 얼굴까지도 애니메이션 처리할 수 있는 확장성을 가지고 있으며, 실험 결과에서 그 효과성을 입증하였습니다. 향후 연구에서는 실시간 성능 향상과 표정 조정을 더욱 개선할 계획입니다.



### LEAP:D -- A Novel Prompt-based Approach for Domain-Generalized Aerial Object Detection (https://arxiv.org/abs/2411.09180)
Comments:
          ICIP 2024 Workshop accepted paper

- **What's New**: 이번 연구에서는 드론이 촬영한 이미지를 대상으로 한 객체 탐지의 성능 향상을 위한 혁신적인 비전-언어(vision-language) 접근법인 LEAP:D를 제안합니다. LEAP:D는 학습 가능한 프롬프트(learnable prompts)를 활용하여 기존 수동 프롬프트 방식의 한계를 극복하고, 더 넓은 도메인-특화 지식(domain-specific knowledge)을 활용할 수 있도록 합니다.

- **Technical Details**: 제안된 LEAP:D 접근법은 객체 탐지 모델인 Faster R-CNN과 Feature Pyramid Network를 기반으로 하며, CLIP 모델을 이용해 도메인-특화 특징(domain-specific features)을 필터링합니다. 훈련 과정에서 LEAP:D는 시각적 임베딩(visual embedding)과 중간 특징(intermediate features) 간의 유사성을 통해 도메인-불변 특징(domain-invariant features)을 학습하게 하며, 이는 훈련 후 객체 탐지 모델의 성능을 향상시킵니다.

- **Performance Highlights**: LEAP:D는 전통적인 두 단계 훈련 방식에 비해 훈련 과정을 단일 단계로 간소화하여, 훈련의 효율성과 성능을 높였습니다. 이는 다양한 환경에서도 드론 이미지에서 객체 탐지의 정확성을 향상시킵니다.



### Advancing Diffusion Models: Alias-Free Resampling and Enhanced Rotational Equivarianc (https://arxiv.org/abs/2411.09174)
Comments:
          13 pages, 7 figures

- **What's New**: 최근의 이미지 생성 기술, 특히 diffusion models의 발전은 이미지 합성 품질의 인상적인 개선을 가져왔습니다. 본 논문에서는 올바르지 않은 재샘플링 작업이 aliasing을 유발하여 이미지 품질을 저하시킨다고 가정하며, 이 문제를 해결하기 위해 alias-free resampling을 UNet 아키텍처에 통합하는 방법을 제안합니다.

- **Technical Details**: 우리는 alias-free resampling을 통해 diffusion models의 UNet 구조에 새로운 학습 가능한 매개변수를 추가하지 않고 모델 효율성을 유지하며 성능을 향상시키는 것을 목표로 하고 있습니다. 이를 통해 높은 주파수 아티팩트를 방지하며 이미지의 회전 동등성을 향상시킬 수 있습니다. 또한, 사용자 제어에 의한 이미지 회전을 가능하게 하는 수정된 diffusion 프로세스를 제안합니다.

- **Performance Highlights**: 실험 결과, CIFAR-10, MNIST 및 MNIST-M과 같은 벤치마크 데이터셋에서 이미지 품질이 일관되게 향상되었음을 확인했습니다. 특히 FID 및 KID 점수에서 향상이 두드러지며, 이는 더 나은 결과를 위해 이미지 처리 원리에 의해 안내된 적절한 alias-free resampling 레이어의 사용 가능성을 강조합니다.



### DyGASR: Dynamic Generalized Exponential Splatting with Surface Alignment for Accelerated 3D Mesh Reconstruction (https://arxiv.org/abs/2411.09156)
- **What's New**: 최근 3D Gaussian Splatting (3DGS)의 발전으로 고품질의 새로운 시점 합성과 빠른 렌더링이 가능해졌지만, 많은 수의 미세한 3D Gaussian 점들에서 메쉬를 추출하는 것은 여전히 큰 도전 과제입니다. 이러한 문제를 해결하기 위해 DyGASR을 제안하며, 전통적인 3D Gaussian 대신 일반화된 지수 함수를 사용하여 입자의 수를 줄이고 잡음 신호의 표현을 동적으로 최적화합니다.

- **Technical Details**: DyGASR에서 제안한 Generalized Exponential Splatting (GES)은 기존의 Gaussian보다 적은 수의 입자로 신호 포인트를 더 정확히 표현합니다. 또한 Generalized Surface Regularization (GSR)을 도입하여 점 구름을 표면과 정렬함으로써, Poisson surface mesh 복원을 촉진합니다. 훈련 단계에서 저해상도에서 고해상도로 점진적으로 이미지 해상도를 증가시키는 코사인 스케줄을 활용한 동적 해상도 조정 전략도 포함됩니다.

- **Performance Highlights**: 우리의 방법은 기존 3DGS 기반 메쉬 복원 방법을 초월하며, 다양한 장면 데이터셋에 대한 광범위한 평가에서 25%의 속도 증가와 30%의 메모리 사용량 감소를 달성했습니다. 전체적인 PSNR에서도 0.29 dB 개선을 보여주며 뛰어난 성능을 입증합니다.



### VidMan: Exploiting Implicit Dynamics from Video Diffusion Model for Effective Robot Manipulation (https://arxiv.org/abs/2411.09153)
Comments:
          Accepted to NeurIPS 2024

- **What's New**: 본 논문에서는 VidMan (Video Diffusion for Robot Manipulation)이라는 새로운 프레임워크를 제안하여 로봇 조작의 정확성과 안정성을 향상시킬 수 있는 두 단계의 훈련 메커니즘을 소개합니다. 이 연구는 신경과학의 이중 과정 이론에 영감을 받아 로봇이 환경의 동적 특성을 더 잘 이해할 수 있도록 하였습니다.

- **Technical Details**: VidMan 프레임워크는 Open X-Embodiment 데이터셋을 사용하여 첫 번째 단계에서 비디오 노이즈 제거(diffusion) 방식으로 미래의 시각적 궤적을 예측하는 사전 훈련을 수행합니다. 두 번째 단계에서는 레이어별 자기 주의 어댑터(self-attention adapter)를 도입하여 동적 지식에 기반한 액션 예측을 가능하게 합니다.

- **Performance Highlights**: VidMan은 CALVIN 벤치마크에서 SOTA 모델인 GR-1보다 11.7%의 상대적 향상을 보였으며, OXE 소규모 데이터셋에서는 9% 이상의 정확도 향상을 나타냈습니다. 이러한 결과는 VidMan이 로봇 행동 예측의 정밀성을 크게 향상시킬 수 있음을 보여줍니다.



### Mono2Stereo: Monocular Knowledge Transfer for Enhanced Stereo Matching (https://arxiv.org/abs/2411.09151)
Comments:
          8 pages, 6 figures

- **What's New**: 이 논문에서는 모노큘러 깊이 추정(mono depth estimation)을 활용하여 스테레오 매칭(stereo matching) 성능을 향상시키는 Mono2Stereo 프레임워크를 제안합니다. 이는 두 단계의 훈련 과정을 포함하며, 합성 데이터 전처리와 실세계 데이터 미세 조정(fine-tuning)을 통해 이루어집니다.

- **Technical Details**: Mono2Stereo 프레임워크는 합성 데이터 생성을 위한 파이프라인을 설계하고 모노큘러 깊이를 사용하는 워핑(warping) 및 새 시점 합성을 사용합니다. 또한 에지 인식(Edge-Aware, EA) 인페인팅 모듈을 포함하여 생성된 이미지에서 누락된 내용을 보완합니다. 이후 S2DKD(Sparse-to-Dense Knowledge Distillation) 전략을 도입하여 스파스(sparse) 라벨에서 밀접하게 관련된 분포를 조정합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 강력한 제로샷(zero-shot) 일반화 능력을 가지고 있으며, 도메인 특이적인 미세 조정을 통한 성능 향상이 두드러집니다. 본 방법은 다양한 데이터셋에서 최첨단(results) 성능을 기록했습니다.



### UniHOI: Learning Fast, Dense and Generalizable 4D Reconstruction for Egocentric Hand Object Interaction Videos (https://arxiv.org/abs/2411.09145)
- **What's New**: 이 논문에서는 Dense 4D Reconstruction을 위한 혁신적인 모델인 UniHOI를 제안합니다. UniHOI는 에고센트릭 손-물체 상호작용(HOI) 장면을 감안하여, 카메라의 내부적 특성, 포즈 및 비디오 깊이 데이터를 통합적으로 예측할 수 있는 빠른 추론 방식을 제공합니다.

- **Technical Details**: UniHOI는 모든 변수를 동시에 추정하여 4D 재구성을 위한 빠르고 밀집된 과정을 가능하게 합니다. 모델은 대규모 단안 비디오 데이터셋에서만 훈련되며, 여러 단계를 거치는 기존 방법들의 누적 오류를 줄이는 End-to-End 최적화를 사용합니다. 이를 통해 3D 공간에서의 일관성을 향상시키고, 모든 관련 변수를 최적화합니다.

- **Performance Highlights**: UniHOI는 포인트 클라우드 시퀀스 재구성과 장기 3D 장면 흐름 복구 작업에서 모든 기준선 모델을 초월하는 성능을 보여줍니다. 이 모델은 빠르고 밀집된 재구성을 통해 에고센트릭 HOI 장면의 복잡한 동작을 효과적으로 처리합니다.



### Adversarial Vessel-Unveiling Semi-Supervised Segmentation for Retinopathy of Prematurity Diagnosis (https://arxiv.org/abs/2411.09140)
Comments:
          10 pages, 5 figures

- **What's New**: 소아 안과에서의 주석이 없는 데이터를 활용한 반감독(segmentation) 접근법을 제안합니다. 이는 ROP(미숙아 망막병증) 연구를 발전시키기 위해 설계된 프레임워크입니다.

- **Technical Details**: 본 연구에서는 교사-학생 네트워크를 사용하는 반감독 학습(SSL) 기법을 사용하여, 불확실성 가중치 망막 혈관 드러내기 모듈과 도메인 적대적 학습을 결합하였습니다. 이를 통해 다양한 도메인에서의 특성 표현을 정렬하고, 가려진 혈관 구조를 효과적으로 드러냅니다.

- **Performance Highlights**: 공식적인 평가 지표에서 우수한 성능을 입증하였으며, 특히 ROP의 단계별 분류 작업에서_segmentation_된 혈관 마스크를 활용하여 진단 정확성을 향상시킨 점이 강조되었습니다.



### SCAN: Bootstrapping Contrastive Pre-training for Data Efficiency (https://arxiv.org/abs/2411.09126)
- **What's New**: 본 논문은 contrastive pre-training에서 데이터 효율성 문제를 해결하기 위해 새로운 동적 부트스트래핑 데이터 세트 프루닝(dynamic bootstrapping dataset pruning) 방법을 제안합니다. 기존 정적 코어셋 선택(static coreset selection) 메소드의 한계를 극복하고 데이터 유용성을 동적으로 추적할 수 있는 혁신적인 접근입니다.

- **Technical Details**: 제안된 SCAN 알고리즘은 학습 중 일관된 프루닝 비율 대신 덜 중요한 데이터 하위 집합을 식별 및 제거하는 부트스트래핑 방식으로 작동합니다. 이 과정은 반복적이고 동적으로 업데이트되며, CLIP과 MoCo와 같은 두 가지 대표적인 contrastive pre-training 프레임워크에 적용되어 성능을 검증했습니다.

- **Performance Highlights**: 16개의 사전 훈련 모델에서 30-35%의 데이터 프루닝 비율을 적용하였음에도 평균 1% 미만의 성능 저하로, 전체 데이터 세트에서 훈련한 모델과 비교해도 우수한 성능을 보였습니다. SCAN 방법은 여러 기초선과 비교할 때도 큰 성능 차이를 나타내며, 원본 데이터 세트 이후 발생한 코어셋도 다른 정적 코어셋 선택 방법보다 우수한 성능을 보여줍니다.



### VCBench: A Controllable Benchmark for Symbolic and Abstract Challenges in Video Cognition (https://arxiv.org/abs/2411.09105)
- **What's New**: 이 논문에서는 VCBench라는 새로운 비디오 인지 벤치마크를 소개하여, 상징적 및 추상적 개념을 포함하는 LVLM(대규모 비디오-언어 모델)의 인지 능력을 평가할 수 있도록 합니다.

- **Technical Details**: VCBench는 Python 기반 엔진을 사용하여 동적이고 과제 지향적인 비디오를 생성하며, 복잡한 장면과 추상 개념을 포함합니다. 각 작업은 특정 인지 과제를 대상으로 하는 맞춤형 질문 템플릿과 결합되어 있습니다. 주요 평가 차원으로는 Object Perception, Action Perception, Spatial Reasoning, Temporal Reasoning 등이 있습니다.

- **Performance Highlights**: 이번 평가에서 Qwen2-VL-72B와 같은 최신 모델조차 상징적 요소를 포함한 간단한 비디오 인지 작업에서 어려움을 겪었으며, 비디오 복잡성이 증가함에 따라 성능이 19% 급감했습니다. 이는 LVLM의 고급 인지 작업에 대한 한계를 명확히 드러냅니다.



### Heuristical Comparison of Vision Transformers Against Convolutional Neural Networks for Semantic Segmentation on Remote Sensing Imagery (https://arxiv.org/abs/2411.09101)
- **What's New**: 본 논문에서는 Remote Sensing 항공 이미지의 semantic segmentation에 있어 ViT(비전 트랜스포머)의 사용을 비교하고, 새로운 결합 가중 손실 함수(combined weighted loss function)를 제안하여 UNet CNN 모델의 성능을 향상시키는 연구 결과를 발표합니다.

- **Technical Details**: 본 연구에서는 UNet에 새로운 결합 가중 손실 함수를 도입하여 mIoU(평균 교차분할 비율) 점수를 극대화하고, 불확실성을 보존하면서 강력한 마스크 예측을 가능하게 합니다. 또한, Meta의 MaskFormer와 일반적인 UNet CNN 간의 전이 학습을 비교하여 학습 효율성과 추론 시간에서의 성능 차이를 분석합니다.

- **Performance Highlights**: 실험 결과, 잡음 제거 및 배경 클래스 관리와 같은 다양한 기법들이 CNN 모델 성능을 크게 개선하는 데 도움이 되는 것으로 나타났으며, 특정 상황에서 ViT 모델보다 뛰어난 결과를 보여주었습니다. 새로운 손실 함수는 데이터셋에서 더 나은 일반화를 가능하게 하며, ViT 기반 모델에 대한 비교에서 더 높은 성능을 달성했습니다.



### Drone Detection using Deep Neural Networks Trained on Pure Synthetic Data (https://arxiv.org/abs/2411.09077)
Comments:
          12 pages, 8 figures

- **What's New**: 이 논문은 순수한 합성 데이터셋으로 훈련된 드론 탐지 Faster-RCNN 모델을 제시하며, 이는 실제 데이터로의 전이 가능성을 입증하였습니다. 실험 결과, MAV-Vid 데이터셋에서 97.0%의 AP_50 성능을 달성하였으며, 이는 실제 데이터로 훈련된 모델의 97.8%와 유사합니다.

- **Technical Details**: 모델 훈련에 사용된 합성 데이터셋은 Structured Domain Randomization (SDR) 기법을 이용하여 생성되었습니다. 드론 탐지의 정확도 향상을 위해, 다양한 조명, 텍스쳐 및 자세를 무작위로 변경하여 실제 환경과 유사한 시나리오를 구현하였습니다. 이를 통해 모델이 실제 데이터에서도 성능을 발휘할 수 있도록 하였습니다.

- **Performance Highlights**: 드론 탐지 분야에서 합성 데이터를 활용함으로써, 데이터 수집 비용을 줄이고 레이블 품질을 향상시킬 수 있는 가능성을 보여주었습니다. 이 연구 결과는 앞으로 더 정교한 합성 드론 데이터셋 개발의 초석이 될 수 있습니다. 또한, 안전이 중요한 애플리케이션인 공항 드론 탐지의 데이터셋 생성을 리스크를 줄일 수 있습니다.



### Multimodal Object Detection using Depth and Image Data for Manufacturing Parts (https://arxiv.org/abs/2411.09062)
- **What's New**: 이번 연구는 RGB 카메라와 3D 포인트 클라우드 센서를 결합한 다중 센서 시스템을 제안하며, 기존의 단일 센서 방식의 한계를 극복합니다.

- **Technical Details**: 제안하는 방법은 RGB 이미지를 처리하기 위해 설계된 Faster R-CNN 기반의 다중 모달 객체 탐지 방법입니다. RGB 영상과 깊이 데이터를 동시에 처리하여 더 정확하고 안정적인 객체 탐지를 구현하였습니다.

- **Performance Highlights**: 이 모델은 RGB 전용 기준 대비 mAP(Mean Average Precision)에서 13% 개선, Depth 전용 기준 대비 mAP에서 78% 개선되었습니다. 이는 스마트 제조 애플리케이션에서의 객체 탐지 신뢰성을 크게 향상시킵니다.



### A Transformer-Based Visual Piano Transcription Algorithm (https://arxiv.org/abs/2411.09037)
Comments:
          9 pages, 2 figures

- **What's New**: 본 연구는 Transformer 기반의 비주얼 피아노 전사(Visual Piano Transcription, VPT) 시스템을 제안하며, 이는 오디오 기반 자동 음악 전사(Automatic Music Transcription, AMT)와 비교하여 새로운 접근 방식을 제공합니다.

- **Technical Details**: 제안된 VPT 시스템은 피아노 바운딩 박스 검출 모델(YOLOv8)과 발생 및 음높이 검출 모델(VideoMAE)을 결합하여 구성됩니다. 이를 통해 시스템은 불완전한 이미지 크롭 및 약간 기울어진 이미지에서도 안정적인 성능을 발휘합니다.

- **Performance Highlights**: 본 모델은 기존의 CNN 기반 모델과 비교할 때 더 나은 성능을 보이며, 피아노의 음표 발음(onset) 및 음높이를 효과적으로 인식하는 능력을 입증하였습니다.



### CoMiX: Cross-Modal Fusion with Deformable Convolutions for HSI-X Semantic Segmentation (https://arxiv.org/abs/2411.09023)
- **What's New**: 이 연구에서는 CoMiX라는 새로운 비대칭 인코더-디코더 아키텍처를 제안하여 하이퍼스펙트럴 이미지(Hyperspectral Image, HSI)와 X-모달리티의 정보를 효과적으로 융합합니다. CoMiX는 변형 가능한 컨볼루션(Deformable Convolutions, DCNs)을 사용하여 두 데이터 소스에서 정보를 추출하고 통합합니다.

- **Technical Details**: CoMiX는 두 개의 병렬 및 상호작용하는 백본을 갖춘 인코더와 경량의 모든 다층 퍼셉트론(All-Multilayer Perceptron, ALL-MLP) 디코더로 구성됩니다. 인코더는 공간-스펙트럼 특성을 적응적으로 집계하기 위해 HSI에 대한 3D DCN 블록과 X 모델에 대한 2D DCN 블록을 포함하는 네 단계로 구성되어 있습니다. 또한 각 단계에는 다양한 모달리티에서 공간-스펙트럼 상관관계를 활용하는 CMFeX 모듈과 정보를 융합하는 FFM 모듈이 포함되어 있습니다.

- **Performance Highlights**: 광범위한 실험을 통해 CoMiX가 다양한 다중모달 인식 작업에서 우수한 성능을 달성하고 잘 일반화됨을 입증하였습니다. 연구 결과는 하이퍼스펙트럴 이미지와 기타 모달리티의 융합을 통한 시맨틱 세분화에서 CoMiX의 효과성을 강조합니다.



### Bridging the Visual Gap: Fine-Tuning Multimodal Models with Knowledge-Adapted Captions (https://arxiv.org/abs/2411.09018)
- **What's New**: 본 연구에서는 비전-언어 모델(vision-language models, VLMs)이 긴 상세한 이미지 캡션에 어떻게 적응하는지를 탐구합니다. 우리는 Decomposed NLI (DNLI)라는 새로운 평가 프레임워크를 제안하여 생성된 캡션을 개별 제안으로 분해하고 각 제안을 독립적으로 평가합니다. 또한, KnowAda라는 데이터 중심의 파인튜닝 기법을 소개하여 VLM이 고유한 지식과 시각적 이해를 활용하여 캡션을 자동으로 조정하도록 합니다.

- **Technical Details**: KnowAda는 세 가지 단계로 구성된 캡션 변환 방법으로, 각 단계에서 VLM의 시각적 지식을 평가하고, 모델이 이해하지 못하는 이미지를 기반으로 한 질문을 생성하여 미지의 내용을 반영하도록 캡션을 수정합니다. DNLI 프레임워크는 캡션의 설명력 및 오류율을 보다 신뢰할 수 있는 방식으로 평가하며, 복잡한 캡션이 모델의 성능에 미치는 영향을 분석합니다.

- **Performance Highlights**: KnowAda는 2억에서 70억 개의 파라미터를 가진 여러 소규모 VLM에서 검증되어, 원본 캡션으로 훈련했을 때보다 환각(hallucinations)을 일관되게 줄이며, 자동 및 인간 평가 모두에서 캡션의 설명력(descriptiveness)과 사실 정확도(factual accuracy) 간의 균형을 잘 제공함을 보여주었습니다.



### Scale Contrastive Learning with Selective Attentions for Blind Image Quality Assessmen (https://arxiv.org/abs/2411.09007)
- **What's New**: 본 논문에서는 새로운 다중 스케일 이미지 품질 평가(BIQA) 프레임워크인 Contrast-Constrained Scale-Focused IQA Framework (CSFIQA)를 제안합니다. 이 프레임워크는 정보 중복성을 최소화하고 품질 관련 정보를 강조하기 위한 선택적 주의 메커니즘을 포함하며, 서로 다른 스케일에서 품질 차이를 식별하는 차별적 학습 모듈을 도입했습니다.

- **Technical Details**: CSFIQA는 선택적 주의(attention) 메커니즘을 활용하여 다중 스케일 정보의 중복성을 줄이고, 정보 필터링과 기능 증폭을 통해 중요한 품질 관련 피처를 강조합니다. 또한, 스케일 간 및 스케일 내 품질 인식을 위한 차별적 학습 전략이 포함되어 있으며, 이로 인해 서로 다른 스케일에서 품질 정보의 모순을 효율적으로 구분합니다.

- **Performance Highlights**: CSFIQA는 8개의 기준 데이터셋에서 뛰어난 성능을 보여주었으며, 특히 SRCC 값 0.967을 달성하여 기존 CSIQ의 0.947 및 LIVEC의 0.876보다 높은 결과를 기록했습니다. 이러한 결과는 CSFIQA가 기존의 BIQA 방법보다 월등한 성능을 가지고 있음을 보여줍니다.



### Computed tomography using meta-optics (https://arxiv.org/abs/2411.08995)
- **What's New**: 본 논문에서는 훈련 데이터에 의존하지 않고, 라돈 변환(Radon transform)을 구현하는 메타 옵틱 이미저(metaoptic imager)에 대해 소개합니다. 이 시스템은 높은 품질의 이미지 재구성을 제공하며, 압축 비율이 0.6%인 이미지를 생성합니다.

- **Technical Details**: 라돈 변환은 이미지와 신호 처리에서 중요한 요소로, 메타 옵틱을 통해 비선형 빛을 사용하여 구현됩니다. 메탈렌스(metalens) 디자인과 원주형 렌즈를 사용하여 이미지를 기록하는 동시에, 90%의 분류 정확도로 실험적으로 측정된 라돈 데이터 세트를 분류할 수 있도록 인공 신경망(artificial neural network)이 활용됩니다.

- **Performance Highlights**: 실험 결과, 재구성된 이미지의 픽셀 수는 약 0.6%에 불과하지만, 높은 품질의 이미지가 생성되었습니다. 또한 디지털 변환된 이미지로 훈련된 신경망을 사용하여 라돈 변환된 이미지에서 90%의 분류 정확도를 달성하였습니다.



### Dual-Head Knowledge Distillation: Enhancing Logits Utilization with an Auxiliary Head (https://arxiv.org/abs/2411.08937)
Comments:
          Preprint

- **What's New**: 전통적인 Knowledge Distillation(KD) 기법이 확률 예측과 ground-truth label 간의 정렬에 중점을 두는 반면, 이 연구에서는 logit 수준의 손실 함수(logit-level loss function)를 추가하여 logit의 잠재적 정보를 활용하는 새로운 방법인 Dual-Head Knowledge Distillation(DHKD)을 제안합니다.

- **Technical Details**: DHKD에서는 주선형 분류기(linear classifier)를 두 개의 분류 헤드로 분할하여 서로 다른 손실을 담당하도록 구성함으로써, BinaryKL 손실을 이용해 logit의 정보를 효과적으로 활용합니다. 이 과정에서 BinaryKL 손실은 backbone 학습에 긍정적인 영향을 미치지만, linear classifier 헤드의 수렴을 방해하는 문제를 해결할 수 있습니다.

- **Performance Highlights**: 광범위한 실험 결과, DHKD 방법이 기존의 최첨단 기법보다 우수한 성능을 보여주며, logit 내부의 정보를 효과적으로 활용하며 student 모델의 성능 저하 문제를 해결함을 입증하였습니다.



### Classification of Keratitis from Eye Corneal Photographs using Deep Learning (https://arxiv.org/abs/2411.08935)
Comments:
          6 pages; Accepted at IEEE's International Conference on Bioinformatics and Biomedicine (2024)

- **What's New**: 본 연구에서는 각기 다른 딥러닝 접근법을 통해 각 감염원의 진단을 비교하고, 새로운 브라질 코라네 데이터셋을 활용하여 '각막염'의 다양한 원인을 분류하기 위한 알고리즘을 제안합니다.

- **Technical Details**: 연구는 세 가지의 딥러닝 모델을 사용합니다: 1) 각각의 감염 유형을 예측하기 위한 세 개의 바이너리 모델; 2) 공유된 백본과 세 개의 병렬 분류 레이어를 가진 멀티태스크 모델(Multitask V1); 3) 공유된 백본과 다중 헤드 분류 레이어를 가진 멀티태스크 모델(Multitask V2). 최종적으로 Multitask V2가 가장 유리한 성능을 보였습니다. 데이터셋 전처리 과정에서, 4,767개의 확인된 각막염 샘플 중에서 2,064개의 고유 사례로 정리하였습니다.

- **Performance Highlights**: Multitask V2 모델의 성능은 세 가지 감염 유형의 AUROC (Receiver Operating Characteristic curve 아래 면적)에서 각각 0.7413-0.7740 (세균), 0.8395-0.8725 (곰팡이), 0.9448-0.9616 (아메바)를 기록하였습니다. 또한, 통계 분석 결과에 따르면 성별이 아메바 감염 예측에 유의미한 영향을 미치며, 나이는 곰팡이와 세균 감염 예측에 영향을 주는 것으로 나타났습니다.



### Predicting household socioeconomic position in Mozambique using satellite and household imagery (https://arxiv.org/abs/2411.08934)
- **What's New**: 남부 모잠비크의 준 농촌 지역에서 975가구의 데이터셋을 구성하여 세대 수준에서 사회경제적 지위(SocioEconomic Position, SEP)를 예측하는 새로운 접근 방식이 제시되었습니다. 이를 통해 위성 이미지와 가정 사진 설문조사를 결합한 다중 모달 데이터셋을 활용하였습니다.

- **Technical Details**: CNN(Convolutional Neural Network)을 미세 조정하여 이미지에서 특징 벡터를 추출하고, 이를 회귀 분석(regression analysis)에 사용하여 가정의 SEP를 모델링 하였습니다. 자산 기반 SEP를 예측할 때 랜덤 포레스트(Random Forest) 모델을 활용하여 가장 좋은 성능을 보였습니다. SHAP(SHapley Additive exPlanations)를 사용하여 이미지의 긍정 및 부정적 영향 차이를 분석하였습니다.

- **Performance Highlights**: 모든 이미지 유형을 사용할 때 자산 기반 SEP의 예측 성능이 가장 뛰어난 것으로 나타났으며, 홈 사진을 사용함으로써 개별 가정 예측으로 zoom-in 하여 데이터 수집 노력을 최소화 할 수 있음을 보여주었습니다. 또한, 중요한 가정 요소만을 사용한 축소 모델은 전체 이미지 모델에 비해 성능이 약간 낮았으나 합리적인 결과를 도출하였습니다.



### Confidence-aware Denoised Fine-tuning of Off-the-shelf Models for Certified Robustness (https://arxiv.org/abs/2411.08933)
Comments:
          26 pages; TMLR 2024; Code is available at this https URL

- **What's New**: 최근 발표된 논문에서는 일반적인 사전 훈련된 분류기의 적대적 강건성을 높이기 위한 새로운 방법인 FT-CADIS를 제안합니다. 이 방법은 특히 역압축된 이미지를 구분하기 위해 신뢰도를 활용합니다.

- **Technical Details**: FT-CADIS는 신뢰도를 고려하여 회복된 이미지를 선택하는 세부 조정 구조를 가지고 있습니다. 이 구조는 두 가지 손실(Confidence-aware selective cross-entropy loss, Confidence-aware masked adversarial loss)을 사용하여 비환각화된 이미지만을 사용하여 분류기를 조정합니다. 이로 인해 분류기는 적대적 강건성을 향상시킬 수 있습니다.

- **Performance Highlights**: FT-CADIS는 CIFAR-10 및 ImageNet 벤치마크에서 기존의 최첨단 방법들과 비교하여 적대적 강건성 면에서 상당한 성능 향상을 보여주었습니다. 특히, FT-CADIS는 높은 가우시안 분산 조건에서 29.5%에서 39.4%의 성능 향상을 기록하였습니다.



### Structured Pattern Expansion with Diffusion Models (https://arxiv.org/abs/2411.08930)
- **What's New**: 본 논문에서는 diffusion 모델을 사용하여 정형화된 패턴을 합성하는 새로운 접근 방식을 제안합니다. 사용자는 스타일과 구조를 조절하면서 부분적으로 손으로 그린 패턴을 더 큰 디자인으로 확장할 수 있는 방법이 소개되었습니다.

- **Technical Details**: 이 연구는 Low-Rank Adaptation (LoRA) 기법을 적용하여 이미지 프리트레인 모델을 세밀하게 조정하고, 노이즈 롤링 기법을 적용하여 타일 가능성을 보장하며 패치 기반 접근 방식을 활용하여 대규모 자산 생성을 용이하게 합니다. 기본적으로 Latent Diffusion Models을 기반으로 하여 학습된 모델을 특정 패턴 도메인에 적합하게 조정했습니다.

- **Performance Highlights**: 엄청난 실험 세트를 통해 제안된 방법이 기존 모델들과 비교하여 사용자 입력에 직접 반응하며 다양한 일관된 패턴을 생성하는 데 성공적임을 보여줍니다. 사용자 만족도를 평가하기 위한 연구도 진행되어 생성 품질에 대한 선호와 인식된 충실도가 수집되었습니다.



### Retrieval of sun-induced plant fluorescence in the O$_2$-A absorption band from DESIS imagery (https://arxiv.org/abs/2411.08925)
Comments:
          submitted to ECCV CVPPA 2024, 14 pages, 8 figures

- **What's New**: 이 논문에서는 공간 기반의 SIF (Sun-Induced Fluorescence) 맵을 30m의 지상 해상도로 추출할 수 있는 최초의 방법을 제시합니다. 이 방법은 고품질의 공중 기반 SIF 추정과 강한 상관관계(r^2=0.6)를 가지고 있습니다. SIF 데이터는 농업 관리 및 생리학적 연구와 관련된 다양한 작업에 대한 설명 정보를 제공할 수 있습니다.

- **Technical Details**: 본 연구는 ESA의 FLEX 미션에 대비하여 O$_2$-A 반대역의 하이퍼스펙트럼 DESIS 이미지를 활용하여 SIF의 추출 방법을 개발했습니다. SFMNN(Spectral Fitting Method Neural Network)이라는 시뮬레이션 기반의 자기 감독 네트워크를 훈련시키고, 대기 변수 예측을 위한 추가적인 감독 규제를 통해 성능 향상을 테스트했습니다. 검증 연구 결과, 모델은 740 nm에서 HyPlant로부터 유도된 SIF 추정치와의 평균 절대 차이 0.78 mW / nm / sr / m²를 기록했습니다.

- **Performance Highlights**: DESIS 이미지를 통해 얻은 SIF 제품은 전례 없는 30m의 공간 해상도를 제공하여, FLEX 미션을 위한 보조 유효성 검증 데이터를 높은 해상도로 수집할 수 있게 합니다. 기존의 전통적인 SIF 추출 방법으로는 DESIS의 SR과 SNR이 부족하지만, 제안된 방법은 이러한 제약을 극복하고 일관된 SIF 추출을 가능하게 합니다.



### Aligning Visual Contrastive learning models via Preference Optimization (https://arxiv.org/abs/2411.08923)
- **What's New**: 이 논문은 디자인된 Preference Optimization (PO) 방법을 통해 대비학습(contrastive learning) 모델의 훈련 방식을 개선하고, 특히 모델의 내구성을 높이며 성별 이해를 분리하고 성차별을 완화하는 데 중점을 둡니다.

- **Technical Details**: 대비학습(contrastive learning) 모델은 표현을 임베딩 공간(embedding space)에서 정렬하여 의미적 유사성을 포착하는 인상적인 능력을 보여주지만, 훈련 데이터의 품질 및 내재된 편견에 의해 성능이 제한될 수 있습니다. 이 연구는 PO 방법을 통해 모델의 행동을 원하는 선호도와 체계적으로 정렬시키며, 일반적인 대비 모델인 CLIP에 대해 치명적인 공격(typographic attacks)에 대한 내성을 강화합니다.

- **Performance Highlights**: PO를 사용하여 훈련된 모델은 표준 대비학습 기술보다 뛰어난 성과를 보이며, 적대적 도전(adversarial challenges)을 처리하고 다른 하위 작업에서 정확성을 유지하는 능력을 보여줍니다. 이 방법은 공정성(fairness), 내구성(robustness), 특정 선호도(preference)와의 정렬이 필요한 작업에 적합합니다.



### On the Surprising Effectiveness of Attention Transfer for Vision Transformers (https://arxiv.org/abs/2411.09702)
Comments:
          NeurIPS 2024. Code: this https URL

- **What's New**: 이번 연구는 전통적인 ViT(Vision Transformers) 프리트레이닝(pre-training)이 다운스트림 작업에서의 성능 향상에 얼마나 기여하는지를 탐구합니다. 연구 결과, 프리트레이닝 중 학습된 특징(feature)과 표현(representation)은 반드시 필수적이지 않다는 것을 발견했습니다. 놀랍게도, 단순히 프리트레이닝에서 얻은 attention 패턴만으로도 높은 품질의 특징을 새롭게 학습하고, 비교 가능한 성능을 달성할 수 있음을 보여줍니다.

- **Technical Details**: 연구에서는 'attention transfer'라는 간단한 방법을 제안합니다. 이를 통해 프리트레이닝된 teacher ViT로부터 attention 패턴만을 학생 모델에게 전달하여, 학생은 이러한 attention 패턴에 의해 특징을 독립적으로 학습합니다. 두 가지 방식인 Attention Copy와 Attention Distillation이 제안되며, 후자는 teacher의 attention 패턴을 학생에게 증류하여 전달합니다. 이는 프리트레이닝된 attention 맵을 활용한 학습 방법으로, 그 효과를 구분할 수 있습니다.

- **Performance Highlights**: 이 방법들은 놀라운 효과를 보였으며, Attention Copy는 대부분의 성능 차이를 해소했고, Attention Distillation은 ImageNet-1K 분류에서 훌륭한 정확도를 달성했습니다. attention transfer를 통해 학생이 스스로 특징을 학습하게 되는 점에서 기존의 파인튜닝(fine-tuning) 방식과 차별화된 성과를 보여주며, 이러한 접근법이 프리트레이닝된 ViT 활용에 있어 효과적인 대안이 될 수 있음을 제안합니다.



### One-Shot Manipulation Strategy Learning by Making Contact Analogies (https://arxiv.org/abs/2411.09627)
Comments:
          CoRL LEAP Workshop, 2024

- **What's New**: 본 논문에서는 새로운 접근 방식인 MAGIC(Manipulation Analogies for Generalizable Intelligent Contacts)를 제안하여 단일 시연(One-shot demonstration)을 통해 조작 전략을 신속하게 학습하고, 새로운 물체에 대한 빠르고 광범위한 일반화를 가능하게 합니다.

- **Technical Details**: MAGIC는 사전 훈련된 신경망 특성과 로컬 곡률 분석을 결합하여 정확하고 물리적으로 그럴듯한 접촉 점을 보장하는 두 단계의 접촉 점 매칭 프로세스를 기반으로 합니다. 이 방법은 참조 행동 궤적(Reference action trajectory)을 활용하여 새로운 물체에서 유사한 접촉 점과 액션 시퀀스를 효과적으로 식별합니다.

- **Performance Highlights**: MAGIC는 스쿱(scooping), 걸이(hanging), 후킹(hooking)과 같은 세 가지 작업에서 뛰어난 성능을 보여주며 기존 방법들에 비해 실행 속도와 다양한 물체 카테고리에 대한 일반화에서 상당한 개선을 달성했습니다.



### Vision-based Manipulation of Transparent Plastic Bags in Industrial Setups (https://arxiv.org/abs/2411.09623)
- **What's New**: 이 논문은 비전 기반 (vision-based) 조작 기술을 활용하여 산업 환경에서 투명 플라스틱 봉지를 자율적으로 자르고 개봉하는 과정의 도전 과제를 다룹니다. 4차 산업혁명 (Industry 4.0) 패러다임에 부합하며, 로봇 공학 및 머신 러닝 (Machine Learning) 알고리즘을 통해 향상된 접근성과 지속 가능성을 제공하는 시스템을 제안합니다.

- **Technical Details**: 제안하는 시스템은 Convolutional Neural Networks (CNNs)를 활용하여 다양한 조명 및 배경 조건에서 투명 플라스틱 봉지를 식별합니다. 추적 알고리즘과 깊이 감지 (depth sensing) 기술을 이용하여 3D 공간 인식을 구현하며, 진공 그립 기술을 통한 최적의 안전한 조작 포인트를 고려하여 효율적인 상호 작용을 가능하게 합니다.

- **Performance Highlights**: FRANKA 로봇팔을 사용하여 실험실에서 시스템을 성공적으로 테스트하였으며, 플라스틱 봉지의 개봉 및 절단 자동화에 대한 효과성을 입증했습니다. 이 시스템은 산업 전반에 걸쳐 광범위한 응용 가능성을 보여줍니다.



### Assessing the Performance of the DINOv2 Self-supervised Learning Vision Transformer Model for the Segmentation of the Left Atrium from MRI Images (https://arxiv.org/abs/2411.09598)
Comments:
          6 pages, 3 figures, SPIE Medical Imaging, 2025

- **What's New**: 본 연구에서는 DINOv2라는 최신 self-supervised learning (SSL) 비전 트랜스포머를 활용하여 심장 MRI 이미지에서 좌심방(Left Atrium, LA) 세분화를 수행하는 방법을 탐구합니다. DINOv2는 자연 이미지에서 훈련되어 의료 이미지 분석에 효과적으로 전이학습(transfer learning)할 수 있는 잠재력을 보여주고 있습니다.

- **Technical Details**: DINOv2는 142M 개의 큐레이션된 자연 이미지로 훈련된 비전 트랜스포머 아키텍처(ViT) 기반의 프레임워크입니다. 본 연구에서는 얇은 경계와 제한된 주석 데이터로 인해 LA 세분화의 도전을 극복하기 위해 DINOv2의 힘을 활용하며, MRI 이미지를 448×448 픽셀 RGB 이미지로 변환한 후 세분화 작업을 수행합니다. DINOv2 ViT-g/14 아키텍처를 사용하여 입력 이미지를 1024개의 비중복 패치로 나누고, 각 패치를 1536차원 임베딩으로 변환하여 최종 세분화 맵을 생성합니다.

- **Performance Highlights**: DINOv2는 평균 Dice 점수 0.871 및 Jaccard 지수 0.792를 기록하며, 다양한 데이터 크기와 환자 수에 대해 few-shot learning을 통해 기초 모델보다 지속적으로 뛰어난 성능을 보여주었습니다. 이는 제한된 데이터 환경에서도 DINOv2가 MRI에 효과적으로 적응할 수 있음을 강조하며, 의료 영상 분야에서의 사용 가능성을 높여줍니다.



### LLaMA-Mesh: Unifying 3D Mesh Generation with Language Models (https://arxiv.org/abs/2411.09595)
Comments:
          See the project website at this https URL

- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)을 활용하여 텍스트 기반 3D 메쉬를 생성하는 새로운 방법인 LLaMA-Mesh를 제안합니다. 이 접근법은 LLMs의 기존 공간적 지식을 활용하고, 대화형 3D 생성 및 메쉬 이해를 가능하게 합니다.

- **Technical Details**: LLaMA-Mesh는 3D 메쉬의 정점 좌표와 면 정의를 일반 텍스트로 표현하여 LLM과의 통합을 용이하게 합니다. 이를 위해, OBJ 파일 형식을 사용하며, 기존의 토크나이저를 확장하지 않고 새로운 데이터 형식을 처리합니다. 또한, 감독형 미세 조정(SFT) 데이터셋을 만들어 LLM이 3D 메쉬를 이해하고 생성할 수 있도록 학습시킵니다.

- **Performance Highlights**: LLaMA-Mesh는 사전 훈련된 LLM이 3D 메쉬를 텍스트 프롬프트로부터 생성하고, 텍스트와 3D 메쉬를 혼합하여 출력할 수 있는 능력을 보여줍니다. 제안된 방법은 모델을 처음부터 훈련한 모델과 동등한 품질의 메쉬 생성을 달성하면서도 강력한 텍스트 생성 성능을 유지합니다.



### SMILE-UHURA Challenge -- Small Vessel Segmentation at Mesoscopic Scale from Ultra-High Resolution 7T Magnetic Resonance Angiograms (https://arxiv.org/abs/2411.09593)
- **What's New**: SMILE-UHURA 챌린지는 7T MRI를 사용하여 획득된 Time-of-Flight MRA 데이터에 대해 주석이 달린 데이터셋을 제공하여 뇌의 미세 혈관 세분화에 대한 연구를 촉진하는 것이 목적입니다. 이는 자동 세분화 알고리즘 개발에 필요한 공개 데이터셋의 부족 문제를 해결하고자 합니다.

- **Technical Details**: 이 챌린지에서는 7T MRI를 사용한 Time-of-Flight Angiography의 주석이 달린 데이터셋을 제공합니다. 이 데이터셋은 자동 사전 세분화와 포괄적인 수동 세분화의 조합을 통해 생성되었습니다. 16개의 제출된 방법과 2개의 기준 방법이 두 가지 다른 데이터셋에서 비교되었습니다: 같은 데이터셋의 테스트 MRA와 별도의 7T ToF MRA 데이터셋에서 검증되었습니다.

- **Performance Highlights**: 제출된 대부분의 딥러닝 방법들은 제공된 훈련 데이터셋으로 훈련되어 신뢰할 수 있는 세분화 성능을 달성했습니다. 해당 데이터를 기반으로 Dice 점수는 각각 0.838 ± 0.066 및 0.716 ± 0.125에 도달했으며, 평균 성능은 최대 0.804 ± 0.15에 달했습니다.



### GAN-Based Architecture for Low-dose Computed Tomography Imaging Denoising (https://arxiv.org/abs/2411.09512)
- **What's New**: 이 리뷰 논문은 저선량 컴퓨터 단층 촬영(LDCT) 이미징에서 생성적 적대 신경망(GANs)의 발전을 다루고 있습니다. LDCT의 복잡한 이미지 품질과 방사선 노출 문제를 해결하기 위한 최신 GAN 기반 노이즈 제거 기법의 발전을 종합적으로 분석합니다.

- **Technical Details**: 리뷰에서는 기본 아키텍처부터 최신 모델에 이르기까지 GAN 기반 LDCT 노이즈 제거 기법의 진화를 다룹니다. 특히 해부학적 프라이어(anatomical priors), 지각 손실 함수(perceptual loss functions), 혁신적인 규제 전략(regularization strategies)등이 포함된 최첨단 모델에 대해 설명합니다. 각 GAN 아키텍처(cGANs, CycleGANs, SRGANs)의 강점과 한계를 분석합니다.

- **Performance Highlights**: 논문은 PSNR, SSIM, LPIPS와 같은 메트릭을 이용하여 벤치마크 및 임상 데이터셋에서 성능 향상을 정성적 및 정량적으로 보여줍니다. GAN 모델들이 중대한 성과를 보였음에도 불구하고, 해석 가능성, 합성 아티팩트(synthetic artifacts), 임상 관련 메트릭의 필요성 등이 널리 사용되지 않는 도전 과제가 논의됩니다.



### Golden Noise for Diffusion Models: A Learning Framework (https://arxiv.org/abs/2411.09502)
- **What's New**: 이번 논문에서는 텍스트 프롬프트와 무작위 가우시안 노이스를 이용하여 개인화된 이미지를 생성하기 위한 새로운 개념인 '노이즈 프롬프트(noise prompt)'를 제안합니다. 이 개념은 무작위 노이즈를 텍스트 프롬프트에서 유도된 작은 변화를 추가하여 '골든 노이즈(golden noise)'로 변환하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 우리는 노이즈 프롬프트 학습 프레임워크(noise prompt learning framework)를 수립하여 텍스트 프롬프트와 연관된 '프롬프트된(golden) 노이즈'를 체계적으로 학습합니다. 또한, 10만 쌍의 무작위 노이즈와 골든 노이즈가 포함된 대규모 노이즈 프롬프트 데이터셋(noise prompt dataset, NPD)을 수집하였습니다. 이를 통해 훈련된 NPNet은 랜덤 노이즈를 골든 노이즈로 직접 변환할 수 있는 소형 네트워크입니다.

- **Performance Highlights**: NPNet을 사용하여 SDXL, DreamShaper-xl-v2-turbo 및 Hunyuan-DiT와 같은 여러 주류 확산 모델에서 실험한 결과, 생성된 이미지의 질과 심미성이 크게 향상되었습니다. NPNet은 표준 파이프라인에 비해 추가적인 3%의 추론 시간만 소모하며, 메모리 요구량도 약 3%로 낮추었으며, 이는 실세계 적용 가능성을 높입니다.



### Automated Segmentation of Ischemic Stroke Lesions in Non-Contrast Computed Tomography Images for Enhanced Treatment and Prognosis (https://arxiv.org/abs/2411.09402)
Comments:
          7 pages, 3 figures, MICCAI Meets Africa Workshop

- **What's New**: 이 연구에서는 비대조 컴퓨터 단층 촬영(NCCT) 이미지를 사용하여 허혈성 뇌졸중 병변 세그멘테이션(segmentation)의 자동화 방법을 제안하였습니다. nnU-Net 프레임워크를 기반으로 하여 초기 치료를 개선하고 허혈성 뇌졸중 환자의 예후를 향상시키는 것을 목표로 했습니다.

- **Technical Details**: 본 연구는 Acute Ischemic stroke Dataset (AISD)를 사용하였으며, NCCT 스캔은 허혈 증상이 시작된 지 24시간 이내에 수집되었습니다. Residual Encoder U-Net 아키텍처를 채택하여 50 epochs 동안 훈련이 진행되었습니다. 데이터 정규화, 슬라이스 처리 및 패치 크기 조정 등 사전 처리 과정이 포함되었습니다.

- **Performance Highlights**: 초기 평가에서 Dice 점수는 0.596, IoU 점수는 0.510으로 나타났습니다. 아웃라이어를 조정한 후, Dice 점수는 0.752로 향상되었고 IoU 점수는 0.643으로 개선되었습니다. 이 결과는 NCCT에서 허혈성 병변을 자동으로 세그먼트화하는 데 있어 효과적인 잠재력을 보여줍니다.



### Are nuclear masks all you need for improved out-of-domain generalisation? A closer look at cancer classification in histopathology (https://arxiv.org/abs/2411.09373)
Comments:
          Poster at NeurIPS 2024

- **What's New**: 이 논문은 암 검출을 위한 도메인 일반화(domain generalisation)에서의 혁신적인 접근법을 제시하고 있습니다. 특히, 세포 핵(nuclei)에 초점을 맞추어 OOD(out-of-domain) 일반화를 개선하는 방법을 모색하고 있습니다.

- **Technical Details**: 본 연구에서는 CNN(Convolutional Neural Networks)을 활용하여 핵의 형태와 조직에 중점을 두고 학습을 진행합니다. 핵 분할(segmentation) 마스크를 원본 이미지와 통합하여 모델이 세포 핵의 공간적 배열을 우선시하도록 유도합니다. 추가적인 손실 항을 도입하여 형태 기반 특징을 강조하는 방식으로 진행됩니다.

- **Performance Highlights**: 세 가지 데이터셋에 대한 실험 결과, 제안된 방법은 기존 방법들보다 OOD 일반화를 현저히 개선하였으며, 이미지 손상 및 적대적 공격에 대한 강건성도 증가했습니다. 모델의 성능 향상은 모든 비교 기반 방법에 비해 우수한 결과를 보여주었습니다.



### DT-JRD: Deep Transformer based Just Recognizable Difference Prediction Model for Video Coding for Machines (https://arxiv.org/abs/2411.09308)
Comments:
          Submitted to IEEE Transactions on Multimedia

- **What's New**: 이 논문에서는 기계 비전(vision)에 최적화된 Visual Signal Processing을 위한 Just Recognizable Difference (JRD) 예측 모델인 Deep Transformer 기반의 JRD (DT-JRD)를 제안합니다. 이를 통해 정확하게 예측된 JRD가 코딩 비트 전송률을 줄이는 데 사용될 수 있습니다.

- **Technical Details**: JRD 예측을 다중 클래스 분류(multi-class classification) 문제로 모델링하고, 개선된 임베딩(embedding), 내용 및 왜곡(feature extraction) 특성 추출, 다중 클래스 분류, 새로운 학습 전략을 통합한 DT-JRD 프레임워크를 개발했습니다. 또한, Gaussian Distribution 기반의 Soft Labels (GDSL)를 사용하여 비점근적 JRD 손실(asymptotic JRD loss)을 제안하여 훈련 레이블 수를 늘리고 분류 경계를 완화했습니다.

- **Performance Highlights**: DT-JRD에 의해 예측된 JRD의 평균 절대 오차(mean absolute error)는 5.574로, 기존의 최고 성능 JRD 예측 모델보다 13.1% 뛰어난 결과를 보였습니다. DT-JRD 기반 VCM은 VVC와 비교하여 평균 29.58%의 비트 전송률을 줄이면서 물체 감지(object detection) 정확도를 유지하는 것으로 나타났습니다.



### Leveraging Auxiliary Classification for Rib Fracture Segmentation (https://arxiv.org/abs/2411.09283)
Comments:
          Accepted at ICVGIP'24

- **What's New**: 이 연구에서는 흉부 외상에서의 갈비뼈 골절을 효과적으로 분리하고 분석하기 위해 보조 분류 작업을 포함한 혁신적인 딥러닝 모델을 제안합니다. 이 모델은 CT 스캔에서 얻은 패치들로부터 갈비뼈가 골절된 영역과 비골절된 영역을 식별하는 데 도움을 주며, 이를 통해 세부적인 특징 표현을 개선합니다.

- **Technical Details**: 이 모델은 UNet 구조를 기반으로 하여 보조 클래스 작업을 통합하여 깊이 있는 세분화를 개선합니다. 주로 RibFrac 데이터셋을 사용하여 실험을 진행하였으며, 패치-레벨 주석을 활용하여 골절 발생 가능성이 높은 영역을 파악합니다. 이를 통해 특징 표현을 개선하고, 더 나은 세분화 성능을 제공합니다.

- **Performance Highlights**: RibFrac 데이터셋에서의 실험 결과, 제안된 모델은 기존의 방법에 비해 세분화 성능에서 상당한 개선을 보였습니다. 특히 다양한 크기와 형태의 골절을 분리하는 데 유의미한 성과를 나타냈습니다.



### Rethinking Weight-Averaged Model-merging (https://arxiv.org/abs/2411.09263)
- **What's New**: 최근 연구에 따르면, weight-averaged model-merging 기법이 딥러닝에서 효과적인 성능 향상을 가져온다는 사실이 확인되었습니다. 이 연구에서는 모델 가중치를 시각화하여 인지된 패턴이 구조화되고 해석 가능함을 보여줍니다.

- **Technical Details**: 이 논문에서는 weight-averaged model-merging의 세 가지 핵심 관점을 통해 접근합니다: (1) 모델 가중치의 내재적 패턴 분석, (2) 가중치 평균화와 특성 평균화 전략 비교, (3) 다양한 매개변수 크기 변화에 따른 예측 안정성 탐구.

- **Performance Highlights**: 모델 병합 후, 다양한 학습 템플릿을 보존함으로써 상호작용을 통한 성능 향상이 이루어집니다. 연구 결과는 모델 병합의 근본적인 원리를 밝혀내어 향후 연구에 대한 유용한 제안과 공개 소스를 제공합니다.



### Cross Space and Time: A Spatio-Temporal Unitized Model for Traffic Flow Forecasting (https://arxiv.org/abs/2411.09251)
- **What's New**: 이번 연구에서는 공간적(spatial) 및 시간적(temporal) 의존성과 이들 간의 복잡한 상호 작용을 포착하기 위해 Spatio-Temporal Unitized Model (STUM)을 도입합니다. STUM은 기존 접근 방식의 한계를 극복하며, 데이터의 이질성(spatio-temporal heterogeneity)을 해결하기 위한 분포 정렬(distribution alignment)과 특징 융합(feature fusion) 기술을 포함하고 있습니다.

- **Technical Details**: STUM의 핵심은 Adaptive Spatio-temporal Unitized Cell (ASTUC)으로, 이는 저랭크(low-rank) 행렬을 사용하여 공간과 시간, 그리고 이들의 상관관계를 원활하게 저장하고 업데이트합니다. 이 프레임워크는 다양한 spatio-temporal graph neural networks와 결합할 수 있는 모듈형(modular) 설계를 가지고 있으며, 예측 모듈을 통해 예측 정확도와 계산 효율성을 동시에 확보하고 있습니다.

- **Performance Highlights**: 여러 실제 데이터셋에 대한 실험 결과에서 STUM은 기존의 데이터 기반 모델들에 비해 예측 성능을 일관되게 향상시킴을 입증하였으며, 최소한의 계산 비용으로도 우수한 성과를 보여주었습니다. 이 결과는 하이퍼파라미터 최적화(hyperparameter optimization), 사전 훈련(pre-training) 분석 및 결과 시각화(result visualization)를 통해 더욱 뒷받침되었습니다.



### Gazing at Rewards: Eye Movements as a Lens into Human and AI Decision-Making in Hybrid Visual Foraging (https://arxiv.org/abs/2411.09176)
- **What's New**: 본 연구에서는 인간의 시선 추적 데이터를 활용하여 하이브리드 비주얼 포리징(hybrid visual foraging) 작업을 수행하고, 목표의 가치와 유병률(prevalence)이 채집 및 시선 행동에 미치는 영향을 분석하였습니다. 이를 통해 AI 모델인 Visual Forager (VF)를 개발하였습니다.

- **Technical Details**: Visual Forager (VF) 모델은 변형자(transformer) 기반의 구조로, 강화 학습(reinforcement learning) 기법을 통해 훈련되었습니다. 이 모델은 다양한 타겟 객체와 그에 해당하는 가치, 검색 이미지를 입력으로 받아들이고, 포비드(foveated) 비전을 활용하여 이미지 처리 후, 일련의 시선 이동과 각 항목을 수집할지 결정하는 과정까지 시뮬레이션합니다.

- **Performance Highlights**: VF 모델은 모든 기본 모델을 능가하여 누적 보상이 인간과 유사하게 나타났으며, 시간 제약 환경 내에서의 인간의 포리징 행동을 잘 근사하였습니다. 무작위적 조건에서도 유효한 일반화 능력을 보여 주었으며, 이는 추가한 데이터 증강(data augmentation) 기법 덕분입니다.



### Fast probabilistic snake algorithm (https://arxiv.org/abs/2411.09137)
- **What's New**: 본 논문에서는 확률 이론을 기반으로 한 새로운 능동적인 윤곽(Active Contour) 알고리즘을 제안하고 있습니다. A. Blake의 연구와 P. Réfrégier 팀의 연구를 바탕으로 하여 이번 알고리즘은 빠르고 정확한 윤곽 설명을 제공합니다.

- **Technical Details**: 제안된 알고리즘은 Bayes' theorem을 활용하여 이미지에서 윤곽선을 찾고, 이를 통해 'snake' 모델을 efficiantly 개선했습니다. 주요 수식으로는 p(C|B)와 p(B|C)가 있으며, 각각은 곡선이 가장자리에 있을 확률과 현재 곡선 위치에서 가장자리가 있을 확률을 나타냅니다.

- **Performance Highlights**: 우리의 모델은 기존의 snake 모델 및 CASP 모델과 비교했을 때, 폐곡선 및 개곡선 케이스에서 더 나은 성능을 보였습니다. 실험 결과로는 고해상도 윤곽을 위한 재샘플링 및 최적화 과정이 포함되어 있습니다.



### Computational metaoptics for imaging (https://arxiv.org/abs/2411.09133)
- **What's New**: 이 논문은 메타서피스(metasurfaces)와 계산 이미징(computational imaging)의 통합인 컴퓨테이셔널 메타옵틱스(computational metaoptics)에 대해 탐구합니다. 이 접근법은 메타서피스의 물리적 파면 형성 능력과 고급 알고리즘을 결합하여 기존 이미징 성능을 뛰어넘는 가능성을 제시합니다.

- **Technical Details**: 메타서피스는 서브웨이브렝스(subwavelength) 구조로 구성된 초박형(optical) 기구로, 전자기파의 진폭(amplitude), 위상(phase), 편광(polarization) 및 스펙트럼(spectral) 속성을 정교하게 제어할 수 있습니다. 이러한 기술은 이미지 재구성 알고리즘과 결합하여 전통적인 이미징 시스템의 한계를 극복합니다. 컴퓨터에서의 최적화 기법이 메타서피스의 설계와 함께 사용됨으로써, 자동화된 최적의 설계 발견이 가능해집니다.

- **Performance Highlights**: 이 논문은 계산 메타옵틱스가 다양한 첨단 응용 프로그램, 예를 들어 위상 이미징(phase imaging) 및 양자 상태(measurement of quantum states) 측정 등에서 메타서피스의 복잡한 빛 제어 능력과 계산 알고리즘의 높은 차원 정보 재구성 능력을 어떻게 활용하는지를 강조합니다. 이러한 접근법은 이미징 기술 및 과학의 미래에 중심 역할을 할 가능성을 높입니다.



### A multidimensional measurement of photorealistic avatar quality of experienc (https://arxiv.org/abs/2411.09066)
Comments:
          arXiv admin note: text overlap with arXiv:2204.06784

- **What's New**: 이 논문에서는 포토리얼리스틱 아바타의 성능을 주관적으로 평가할 수 있는 오픈 소스 테스트 프레임워크를 제공합니다. 기존의 객관적인 지표(PSNR, SSIM, LPIPS 등)와는 달리, 인간 사용성 요인을 여러 차원에서 평가할 수 있습니다.

- **Technical Details**: 테스트 프레임워크는 리얼리즘(realism), 신뢰(trust), 편안함(comfortableness), 적절성(appropriateness), 기묘함(creepiness), 형식(formality), 친밀감(affinity), 유사성(resemblance), 감정 정확도(emotion accuracy) 등 10개 차원에서 아바타의 성능을 측정합니다. 결과적으로 주관적 측정 결과와 객관적 메트릭 간의 상관관계는 약한 것으로 나타났으며, 특히 감정 정확도와의 관계는 중간 정도였습니다.

- **Performance Highlights**: 연구 결과, 포토리얼리스틱 아바타는 신뢰성을 높이고 비디오 피로를 줄일 수 있는 가능성을 보여주었습니다. 리얼리즘이 특정 수준을 넘어서면, 아바타 친밀감과 리얼리즘 간에는 강한 상관관계가 존재하며, 이는 통신 시나리오에서 기묘한 골짜기 효과(uncanny valley effect)가 없음을 시사합니다.



### IDCIA: Immunocytochemistry Dataset for Cellular Image Analysis (https://arxiv.org/abs/2411.08992)
- **What's New**: 이 연구에서는 기계 학습 방법의 효율성을 향상시키기 위해 새로운 주석이 달린 미세 세포 이미지 데이터셋을 제시합니다. 이 데이터셋은 세포 수 카운팅을 자동화할 수 있는 가능성을 제시하며, 다양한 항체로 염색된 세포 이미지가 포함되어 있습니다.

- **Technical Details**: 이 데이터셋은 전기 자극 과정 후 촬영된 Adult Hippocampal Progenitor Cells (AHPCs)의 이미지를 포함하고 있습니다. 이미지는 ICC(면역세포화학)를 통해 수집되었으며, 각 이미지는 세포의 위치와 수량을 주석 처리하였습니다. DNN(Deep Neural Networks) 모델의 성능 비교 결과, 기존 모델들이 수동 카운트를 대체할 만큼의 정확도를 달성하지 못했습니다.

- **Performance Highlights**: 제안된 데이터셋은 다양한 항체 염색 방법을 포함하여, 공개된 다른 데이터셋들보다 더 많은 이미지를 제공합니다. 연구팀의 실험 결과, 현재 DNN 기반 세포 카운팅 방법들은 정확도 면에서 한계가 있으며, 이 데이터셋은 향후 연구 및 개선의 기준이 될 수 있습니다.



### Fluoroformer: Scaling multiple instance learning to multiplexed images via attention-based channel fusion (https://arxiv.org/abs/2411.08975)
Comments:
          Findings paper presented at Machine Learning for Health (ML4H) symposium 2024, December 15-16, 2024, Vancouver, Canada, 14 pages

- **What's New**: 이번 논문에서는 Multiplexed Whole Slide Images (WSIs) 처리를 위한 새로운 다중 샘플 학습(Multiple Instance Learning, MIL) 전략인 Fluoroformer 모듈을 제안합니다. 이 방법은 세포 이미지와 생물학적 구조에 대한 중요한 정보를 융합하여 해석 가능한 출력을 제공합니다.

- **Technical Details**: Fluoroformer는 Scaled Dot-Product Attention (SDPA) 기술을 활용하여 서로 다른 채널의 정보를 하나의 요약 벡터로 융합합니다. 이로 인해 각 패치를 위한 주의(attention) 행렬을 생성할 수 있으며, 이는 세포 간 상호작용 및 생물학적 구조에 대한 통찰력을 제공합니다.

- **Performance Highlights**: 434개의 비소세포 폐암(non-small cell lung cancer, NSCLC) 샘플에서 Fluoroformer는 예측 성능이 뛰어나며, 면역 종양 상호작용에 대한 중요한 관찰 결과를 도출하였습니다. 이는 공간 생물학적 기술과 최신 인공지능 모델을 결합하여 이 emerging 분야의 잠재력을 극대화합니다.



### Clustered Patch Embeddings for Permutation-Invariant Classification of Whole Slide Images (https://arxiv.org/abs/2411.08936)
Comments:
          arXiv admin note: text overlap with arXiv:2411.08530

- **What's New**: 이 논문에서는 Whole Slide Imaging (WSI)의 분석 효율성을 높이기 위해 새로운 방법론을 제안합니다. 기존의 복잡한 WSIs를 단일 벡터로 압축하여 필요한 특징을 효과적으로 추출하는 기법이 핵심 혁신점입니다.

- **Technical Details**: 제안된 방법론은 다양한 인코더를 활용하여 WSI를 사전 처리하고 중요한 특징을 추출합니다. Deep Learning 모델과 적응형 필터링 기술을 사용하여 WSI의 품질을 개선하고, 512x512 크기로 패치를 나누어 ResNet50, EfficientNet 등 여러 모델을 적용하여 고차원 특징 공간으로부터 클러스터링 기반의 compact한 표현을 생성합니다.

- **Performance Highlights**: 본 연구의 실험 결과, 기존 방법에 비해 분류 정확도가 유의미하게 향상되었으며, WSI의 대규모 분석에 대한 새로운 가능성을 제시합니다. 이 연구는 병리학적 데이터의 효과적이고 견고한 분석을 위한 프레임워크를 제공하며, 의료 진단 및 연구에 있어 중요한 기여를 합니다.



### DG-PPU: Dynamical Graphs based Post-processing of Point Clouds extracted from Knee Ultrasounds (https://arxiv.org/abs/2411.08926)
Comments:
          This paper was submitted to the IEEE International Symposium on Biomedical Imaging (ISBI). This is a preprint version and may be subject to copyright

- **What's New**: 총무릎관절 치환술(TKA) 후 비특이적 전방 무릎 통증을 경험하는 환자들을 위해, 이 연구는 초음파 스캔으로부터 추출된 포인트 클라우드(point cloud)를 사용하여 patellofemoral joint (PFJ) 운동을 정확히 시각화하는 새로운 알고리즘인 Dynamical Graph-based Post-processing for Ultrasounds (DG-PPU)를 제안합니다.

- **Technical Details**: DG-PPU는 Dynamical Graph Convolutional Neural Networks (DGCNN)를 활용하여 초음파 포인트 클라우드 데이터를 후처리합니다. 이 알고리즘은 잘못 레이블된 소프트 티슈를 제거하고, PFJ의 기하학적 정보를 이해하여 3D 포인트 클라우드를 부드럽고 정확하게 개선합니다. 훈련 과정에서 k-regular graphs를 생성하며, 전체 500개의 포인트 클라우드를 생성하여 효율성 및 정확성을 높입니다.

- **Performance Highlights**: DG-PPU는 세 가지 다른 무릎 각도에서 건강 관리 기사에 의해 수행된 수동 데이터 클리닝과 비교하여 98.2%의 정밀도를 달성하며, 인간 뼈에서 PFJ 운동을 시각화하는 혁신적인 시스템 개발을 위해 기여합니다.



### Enhancing Lie Detection Accuracy: A Comparative Study of Classic ML, CNN, and GCN Models using Audio-Visual Features (https://arxiv.org/abs/2411.08885)
Comments:
          11 pages, 18 figures

- **What's New**: 이번 연구는 오디오 입력, 시각적 얼굴 미세 표현(micro-expressions), 그리고 수작업으로 기록된 제스처 주석을 사용하는 독특한 다중 모달 트랜스포머 아키텍처를 개발하여 기존의 방법을 개선합니다. 이를 통해 비침습적인 거짓말 탐지 모델에 더 가까워집니다.

- **Technical Details**: 비주얼 및 오디오 특징은 각각 Vision Transformer(ViT) 및 OpenSmile 모델을 사용하여 추출되었으며, 참가자의 미세 표현 및 제스처 형태 기록과 함께 연결(concatenate) 되었습니다. 최종적으로 CNN Conv1D 다중 모달 모델이 평균 정확도 95.4%를 달성하였습니다. 그러나 더 높은 품질의 데이터셋과 더욱 일반화된 모델을 위해 추가 연구가 필요합니다.

- **Performance Highlights**: 이 연구는 거짓말 탐지에서 기존 방법들과 최근 AI 모델들에 비해 제안된 AI 모델의 효율성을 평가하고, 예측에서 가장 큰 비중을 가지는 특징(feature)을 분석합니다. 특히, 다중 모달 데이터 처리의 잠재력을 보여주며, 법 집행, 법정 증언에 대한 신뢰성 평가, 금융 서비스에서의 사기 탐지와 같은 여러 분야에 혁신을 가져올 수 있을 것으로 기대됩니다.



### A Novel Multimodal System to Predict Agitation in People with Dementia Within Clinical Settings: A Proof of Concep (https://arxiv.org/abs/2411.08882)
- **What's New**: 이번 연구는 중증 치매 환자에서의 동요 및 공격성(Agitation and Aggression, AA) 사건을 실시간으로 예측하기 위한 다중 모드 접근법을 도입한 5년간의 연구 결과를 발표합니다. EmbracePlus 손목 밴드와 비디오 감지 시스템을 통합하여 AA의 선행 패턴을 식별하고 관련 사건을 예측할 수 있는 새로운 시스템을 개발하였습니다.

- **Technical Details**: 이 시스템은 EmbracePlus 손목 밴드에서 수집된 생체 데이터와 CCTV 카메라에서 얻은 비디오 데이터를 활용하여 AA를 예측합니다. 심리적 신호를 포함한 데이터는 실시간으로 깊은 학습 딥러닝 모델을 통해 분석되며, 전통적인 방법보다 개인 맞춤형 조기 개입을 가능하게 합니다. 연구 결과, 시스템은 AA 사건 발생 최소 6분 전에 동요 패턴을 탐지할 수 있었습니다.

- **Performance Highlights**: 파일럿 연구에서 3명의 참가자를 대상으로 손목밴드와 비디오 시스템을 동시에 활용하여 AA 사건을 높은 정확도로 탐지하였습니다. 리커시브 뉴럴 네트워크(Recall Neural Network, RNN) 기반의 LSTM과 GRU 모델을 통해 실시간 처리가 최적화되어 AA 사건을 신속하게 분석할 수 있었습니다. 전반적으로 이 시스템의 초기 데이터 분석 결과는 AA 사건을 효과적으로 예측할 수 있는 능력을 보였습니다.



### Machine learning-enabled velocity model building with uncertainty quantification (https://arxiv.org/abs/2411.06651)
- **What's New**: 본 논문에서는 전통적인 FWI(Full-Waveform Inversion) 방법의 한계를 극복하기 위해, Diffusion networks를 활용하여 물리적 요약 통계(physics-informed summary statistics)와 결합한 확장 가능한 방법론을 제안합니다. 이 접근법은 복잡한 이미징 문제에 적합하며, 기존 속도 모델이 부족한 경우에도 효율적으로 베이esian posterior 샘플을 생성할 수 있도록 합니다.

- **Technical Details**: 이 연구는 주로 베이esian 추론(Bayesian inference) 프레임워크를 활용하여, 관측 데이터(지진 촬영 데이터)와 프라이어 정보(훈련 샘플)를 결합하여 다수의 지구 모델을 도출합니다. 이 방법은 수치적으로 효율적이며, 큰 데이터 세트에서도 적용이 가능합니다. 주요 기술적 요소로는 조건부 Diffusion networks와 물리적 요약 통계가 포함됩니다.

- **Performance Highlights**: 실험을 통해 기존 방법들과 비교하여 Common-Image Gathers(CIGs)의 사용으로 인한 개선을 확인하였으며, 복잡한 소금 구조를 다루기 위한 새로운 반복적 워크플로우인 ASPIRE를 제안합니다. 마지막으로, 이 방법은 분야 데이터셋에서도 잘 작동하여 산업 규모 문제에 적합하다는 것을 입증하였습니다.



New uploads on arXiv(cs.AI)

### LLM Hallucination Reasoning with Zero-shot Knowledge Tes (https://arxiv.org/abs/2411.09689)
Comments:
          12 pages, 2 figures

- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 환각 문제를 해결하기 위한 새로운 접근법을 제안합니다. 환각이란 LLM이 잘못된 텍스트를 생성하는 현상으로, 다양한 환각 유형 간의 차이를 이해하고 이를 분류하는 'Hallucination Reasoning' 작업을 도입했습니다. 이 작업은 텍스트를 aligned, misaligned, fabricated 세 가지 유형으로 분류합니다.

- **Technical Details**: 논문에서 제안한 새로운 제로샷(zero-shot) 방법인 MKT는 LLM이 주어진 텍스트와 프롬프트에 대한 충분한 지식을 보유하고 있는지를 평가합니다. 이 과정에서 외부 지식이나 레이블이 있는 데이터셋, LLM의 파인튜닝(fine-tuning) 없이도 환각 유형을 구별할 수 있습니다. 두 단계의 워크플로우(workflow)를 통해 환각 reasoning을 수행하며, 이는 LLM의 특정 지식 부족을 식별하는 데 중점을 둡니다.

- **Performance Highlights**: 실험 결과, 제안된 MKT 방법이 기존 QA와 자유 형식 텍스트 생성 작업에서 우수한 성능을 나타냄을 보여주었습니다. 또한, 우리의 방법을 기존의 환각 탐지 알고리즘에 통합했을 때 성능이 크게 향상되어, 환각 reasoning의 중요성이 강조되었습니다.



### Med-Bot: An AI-Powered Assistant to Provide Accurate and Reliable Medical Information (https://arxiv.org/abs/2411.09648)
Comments:
          3 figures, 5 pages Keywords-LLM, AI-powered healthcare, Medical chatbot, Context-based interaction, Llama-assisted data processing, AutoGPT-Q, PyTorch, TensorFlow, Reliable medical information, Machine learning in healthcare, Conversational AI

- **What's New**: 이번 연구에서는 Med-Bot이라는 AI 기반의 챗봇을 소개합니다. Med-Bot은 의료 정보를 제공하기 위해 설계되었으며, 자연어 이해(Natural Language Understanding)의 복잡성을 처리할 수 있도록 여러 고급 라이브러리와 프레임워크를 활용합니다.

- **Technical Details**: Med-Bot은 PyTorch, Chromadb, Langchain, AutoGPT-Q와 같은 기술을 사용하여 구축되었습니다. 이러한 기술들은 자연어 처리의 효율성을 높이고, 의료 문헌의 PDF 기반 쿼리에 대한 응답 성능을 향상합니다. Llama-assisted 데이터 처리 기법과 AutoGPT-Q 통합이 특징적입니다.

- **Performance Highlights**: Med-Bot은 사용자에게 신뢰성 있는 의료 정보를 제공하는데 중점을 두며, 시뮬레이션된 의료 제공자와의 상호작용을 통해 개인화된 응답을 생성합니다. 다양한 최신 기법을 통해 의료 챗봇의 기능을 강화하고, 사용자 경험을 향상시키는데 기여하고 있습니다.



### Accelerating Knowledge Graph and Ontology Engineering with Large Language Models (https://arxiv.org/abs/2411.09601)
- **What's New**: 본 논문은 LLM(대규모 언어 모델)에 기반한 지식 그래프(Knowledge Graph) 및 온톨로지 공학(Ontology Engineering)의 새로운 연구 분야를 제시하며, 모듈화(modularity) 접근 방식의 중요성을 강조합니다.

- **Technical Details**: 지식 그래프 및 온톨로지 공학(KGOE)은 지식 그래프의 수명 주기와 관련된 여러 작업을 포함합니다. 이 작업은 온톨로지 모델링, 확장, 수정, 인구화, 정렬 및 엔티티의 불확실성 해결 등을 포함하며, 수십 년의 연구에도 불구하고 자동화하기가 어렵습니다. LLM은 자연어 지식 기반으로 사용되며, KGOE 문제의 초안을 생성하는 데 유용할 수 있습니다.

- **Performance Highlights**: 세부 사례 연구에서 LLM이 모듈화된 온톨로지를 활용하여 높은 정밀도와 재현율로 규칙을 생성하는 데 성공함을 보여줍니다. 이 연구는 향후 LLM 기반 KGOE의 발전을 위한 구체적인 연구 과제를 제안합니다.



### Automating Reformulation of Essence Specifications via Graph Rewriting (https://arxiv.org/abs/2411.09576)
Comments:
          Presented at the PTHG 2024 workshop

- **What's New**: 이 논문은 그래프 변환을 이용하여 파라미터화된 문제 클래스의 입력 모델을 자동으로 재형성하여 성능을 개선하는 시스템을 소개합니다. 기존 연구와의 차별점은 문제 인스턴스가 아닌 전체 문제 클래스에 대한 모델을 재형성하는 점입니다.

- **Technical Details**: Essence 추상 제약 조건 사양 언어를 기반으로 하여, 입력 사양의 추상 구문 트리(abstract syntax tree)를 이용하여 그래프 규칙을 적용합니다. GP2 언어를 통해 재형성 규칙을 표현하고, 이를 사용하여 고급 고수준 타입을 변환하게 됩니다. 특히, 관계 선언의 AST 부분과 일치하도록 설계된 rewrite rule에 대해 논의합니다.

- **Performance Highlights**: k-fold 그래프 색칠 문제에 대한 사례 연구를 통해 시스템의 효과성을 입증합니다. 이를 통해 재형성이 해결 시간을 단축시킬 뿐만 아니라 메모리 사용량을 줄이고, 원본 형태로는 해결 불가능했던 문제를 해결할 수 있는 가능성을 보여줍니다.



### Navigating the Risks: A Survey of Security, Privacy, and Ethics Threats in LLM-Based Agents (https://arxiv.org/abs/2411.09523)
- **What's New**: 본 논문에서는 LLM (Large Language Model) 기반 에이전트의 보안 및 개인 정보 보호 위협을 포괄적으로 분석하고, 다양한 위험들을 분류하는 새로운 세분화 체계를 제안합니다. 이 연구는 기존의 분류 체계가 모듈 간 및 단계 간 위협을 효과적으로 다루지 못하는 문제를 해결하기 위해 새로운 접근 방식을 취하고 있습니다.

- **Technical Details**: 연구는 LLM의 내부 및 외부 공격에 대한 위협을 분석하며, 위험의 출처와 유형에 따라 이들을 이진 테이블로 분류합니다. 새로운 분류 체계는 입력 문제(Problematic Inputs), 모델 결함(Model Flaws), 복합 위협(Combined Threats)으로 나누어 LLM 기반 에이전트의 위험성을 체계적으로 평가합니다.

- **Performance Highlights**: 논문은 LLM 기술의 발전에도 불구하고 이러한 에이전트가 겪는 보안 문제와 개인 정보 유출 등의 위협들을 강조합니다. 이를 통해 연구자들이 향후 연구 방향을 설정하는 데 도움을 줄 수 있는 기초 자료를 제공합니다.



### An Adaptive Open-Source Dataset Generation Framework for Machine Learning Tasks in Logic Synthesis (https://arxiv.org/abs/2411.09422)
Comments:
          14 pages

- **What's New**: 이 논문은 논리 합성(logic synthesis) 과정 내에서 머신러닝(machine learning) 응용 프로그램의 향상을 위해 설계된 적응형(logic synthesis) 데이터셋 생성 프레임워크를 소개합니다. 이 프레임워크는 기존의 특정 작업에 맞춘 데이터셋 생성 흐름과 달리, 논리 합성의 세 가지 기본 단계를 포괄하여 다양한 머신러닝 작업을 지원합니다.

- **Technical Details**: 제안된 프레임워크는 부울 표현(Boolean representation), 논리 최적화(logic optimization), 기술 매핑(technology mapping)의 세 단계를 포함합니다. 중간 파일에서 원본 정보를 유지할 수 있으며, 이 파일은 Verilog 및 GraphML 형식으로 저장됩니다. Verilog 파일은 반자동(customizable) 옵션을 제공하여 연구자들이 단계 추가 및 데이터셋 개선을 점진적으로 할 수 있도록 합니다. 또한, 적응형 회로 엔진(adaptive circuit engine)을 포함하여 GraphML 파일 로딩을 지원합니다.

- **Performance Highlights**: 생성된 OpenLS-D 데이터셋은 46개의 조합(combinational) 디자인으로 구성되어 있으며, 966,000개 이상의 부울 회로(Boolean circuits)를 포함합니다. 각 디자인은 1000개 합성 레시피(synthesis recipes)에서 생성된 21,000회로를 포함하고, 7000개의 부울 네트워크, 7000개의 ASIC 넷리스트(netlists), 7000개의 FPGA 넷리스트를 포함합니다. OpenLS-D는 새롭게 원하는 데이터 특징을 통합할 수 있어 다양한 도전에 더욱 유용합니다. 이 데이터셋은 회로 분류(circuit classification), 회로 순위 매김(circuit ranking), 결과 품질(prediction of quality of results, QoR), 확률 예측(probability prediction) 같은 네 가지 다운스트림(downstream) 작업을 통해 활용됩니다.



### Imagined Speech and Visual Imagery as Intuitive Paradigms for Brain-Computer Interfaces (https://arxiv.org/abs/2411.09400)
Comments:
          4 pages

- **What's New**: 최근 뇌-컴퓨터 인터페이스(BCI) 기술의 발전은 상상된 말(imagined speech)과 시각적 이미지(visual imagery)가 직관적인 커뮤니케이션을 위한 효과적인 패러다임으로서의 가능성을 강조하고 있습니다. 이번 연구는 이러한 패러다임과 관련된 분류 성능(classification performance)과 뇌 연결성(brain connectivity patterns)을 조사했습니다.

- **Technical Details**: 16명의 참가자가 13개의 상상된 말 및 시각적 이미지 클래스를 포함하는 작업에 참여하였으며, 두 패러다임 모두에서 기회 수준을 넘는 분류 정확도(classification accuracy)를 보였습니다. 개별 클래스(class) 간 분류 정확도의 변동성은 상상된 말에서의 감각(Sensory) 및 운동(Motor) 연관, 시각적 이미지에서는 생생한 시각적 연결(vivid visual associations)의 영향을 강조합니다. 연결성 분석(connectivity analysis)에서는 상상된 말이 언어 관련 및 감각 영역에서 기능적 연결(functional connectivity)이 증가하였고, 시각적 이미지는 공간적(spatial) 및 시각적 처리(visual processing) 네트워크를 활성화 함을 보여주었습니다.

- **Performance Highlights**: 이 연구 결과는 상상된 말 및 시각적 이미지가 BCI 커뮤니케이션을 위한 직관적이고 확장 가능한 패러다임으로서의 잠재력을 시사합니다. 최적의 단어 클래스(word classes)를 선택할 때 이러한 두 패러다임의 디코딩 결과(decoding outcomes)에 대한 추가 탐색이 실질적인 BCI 커뮤니케이션을 위한 통찰을 제공할 수 있습니다.



### Multi-scale Generative Modeling for Fast Sampling (https://arxiv.org/abs/2411.09356)
- **What's New**: 새로운 제안은 파워-로우(포물선) 감소로 인해 발생하는 문제를 해결하기 위해, 웨이브렛(wavelet) 도메인에서 다중 스케일(멀티 스케일) 생서 모델링을 포함한 것입니다. 이 연구는 저 주파수 대역과 고주파수 대역을 처리하는 데 각각 다른 전략을 제안합니다.

- **Technical Details**: 이 연구에서는 저주파수 대역에 대해서는 점수 기반 생성 모델링(score-based generative modeling)을 적용하고, 고주파수 대역에 대해서는 다중 스케일 적대적 학습(multi-scale generative adversarial learning)을 활용합니다. 이러한 방법은 웨이브렛 도메인에서 고품질 데이터 생성을 지원하며, 이론적 분석과 실험 결과로 성과를 입증하였습니다.

- **Performance Highlights**: 제안된 모델은 훈련 가능한 매개변수 수, 샘플링 단계, 시간 모두를 줄이면서 성능을 크게 향상시켰습니다.



### Cross Space and Time: A Spatio-Temporal Unitized Model for Traffic Flow Forecasting (https://arxiv.org/abs/2411.09251)
- **What's New**: 이번 연구에서는 공간적(spatial) 및 시간적(temporal) 의존성과 이들 간의 복잡한 상호 작용을 포착하기 위해 Spatio-Temporal Unitized Model (STUM)을 도입합니다. STUM은 기존 접근 방식의 한계를 극복하며, 데이터의 이질성(spatio-temporal heterogeneity)을 해결하기 위한 분포 정렬(distribution alignment)과 특징 융합(feature fusion) 기술을 포함하고 있습니다.

- **Technical Details**: STUM의 핵심은 Adaptive Spatio-temporal Unitized Cell (ASTUC)으로, 이는 저랭크(low-rank) 행렬을 사용하여 공간과 시간, 그리고 이들의 상관관계를 원활하게 저장하고 업데이트합니다. 이 프레임워크는 다양한 spatio-temporal graph neural networks와 결합할 수 있는 모듈형(modular) 설계를 가지고 있으며, 예측 모듈을 통해 예측 정확도와 계산 효율성을 동시에 확보하고 있습니다.

- **Performance Highlights**: 여러 실제 데이터셋에 대한 실험 결과에서 STUM은 기존의 데이터 기반 모델들에 비해 예측 성능을 일관되게 향상시킴을 입증하였으며, 최소한의 계산 비용으로도 우수한 성과를 보여주었습니다. 이 결과는 하이퍼파라미터 최적화(hyperparameter optimization), 사전 훈련(pre-training) 분석 및 결과 시각화(result visualization)를 통해 더욱 뒷받침되었습니다.



### Towards Unified Neural Decoding of Perceived, Spoken and Imagined Speech from EEG Signals (https://arxiv.org/abs/2411.09243)
- **What's New**: 이 연구는 비침습적 방법으로 뇌신호를 해독하는 데 있어서 심층 학습 모델을 활용하여, 다양한 연설 패러다임(그릇된, 대화식, 속삭임, 가상 연설)을 구분하는 데 초점을 맞췄습니다. 특히, 감마 주파수 대역에서 공간적 컨볼루션 신경망(spatial convolutional neural network)을 사용한 모델이 다른 모델에 비해 향상된 성능을 보여주었습니다.

- **Technical Details**: 연구는 10명의 건강한 참가자로부터 수집된 128채널 EEG 신호를 기반으로 진행되었습니다. 실험에서는 '지각된', '대화식', '속삭임', '상상된' 연설 상태를 분석했으며, 각 상태에 따라 20개의 단어 클래스에 대해 데이터가 수집되었습니다. FFT(Short-Time Fourier Transform)와 같은 방법을 사용하여 주파수 밴드에서의 파워를 효과적으로 추출했습니다. 또한, PLV와 PLI를 통해 뇌신호 간의 연결성을 평가했습니다.

- **Performance Highlights**: 제안된 모델은 특히 감마 대역에서 다른 방법들보다 유의미하게 우수한 성능을 발휘하였으며, 테타 주파수 대역에서의 상상된 연설 또한 통계적으로 유의미한 차이를 보였습니다. 또한, 추정된 신경 동역학이 각각의 연설 상태와 단어 범주에 대해 강력한 결과를 얻었습니다.



### Dynamic Neural Communication: Convergence of Computer Vision and Brain-Computer Interfac (https://arxiv.org/abs/2411.09211)
Comments:
          4 pages, 2 figures, 1 table, Name of Conference: International Conference on Brain-Computer Interface

- **What's New**: 이번 연구에서는 인간의 신경 신호를 해석하여 정적(Static) 및 동적(Dynamic) 발화 의도를 디코딩하는 혁신적인 방법을 제안합니다. 특히, 뇌-컴퓨터 인터페이스(Brain-Computer Interface)와 컴퓨터 비전(Computer Vision) 기술을 활용하여 다이나믹한 신경 커뮤니케이션(Dynamic Neural Communication)을 가능하게 하였습니다.

- **Technical Details**: 이 연구의 방법론은 사용자의 신경 신호로부터 의도를 캡처하고, 짧은 시간 단위로 비젬(Viseme)을 디코딩하여 동적인 시각 출력을 생성하는 것입니다. 다양한 특징을 가진 신경 신호를 효과적으로 활용하고, 얼굴 움직임을 신속하게 복원하는 데 초점을 맞추고 있습니다.

- **Performance Highlights**: 결과적으로, 제안된 방법은 자연스러운 언어 발화 시도 중 인간의 신경 신호로부터 입술 움직임을 빠르게 캡처하고 재구성할 수 있는 가능성을 보여주었으며, 이는 컴퓨터 비전과 뇌-컴퓨터 인터페이스의 융합을 통한 동적 신경 커뮤니케이션의 실현에 기여합니다.



### Improvement and Implementation of a Speech Emotion Recognition Model Based on Dual-Layer LSTM (https://arxiv.org/abs/2411.09189)
- **What's New**: 본 논문은 기존의 음성 감정 인식 모델에 추가적인 LSTM 레이어를 추가하여 오디오 데이터에서 감정 인식의 정확성과 처리 효율성을 개선합니다.

- **Technical Details**: 이 연구에서 제안한 더블 레이어 LSTM 구조는 오디오 시퀀스의 장기 의존성을 효과적으로 포착하여 복잡한 감정 패턴을 보다 정확하게 인식하고 분류할 수 있습니다. 주요 기술적 요소로는 Dual-layer LSTM, Mel-Frequency Cepstral Coefficients (MFCC), 그리고 Softmax 출력 층이 포함됩니다.

- **Performance Highlights**: RAVDESS 데이터셋에서 수행된 실험 결과, 수정된 더블 레이어 LSTM 모델은 단일 레이어 LSTM보다 2% 더 높은 정확도를 보였으며, 인식 지연을 현저히 줄여 실시간 성능을 향상시켰습니다.



### Gazing at Rewards: Eye Movements as a Lens into Human and AI Decision-Making in Hybrid Visual Foraging (https://arxiv.org/abs/2411.09176)
- **What's New**: 본 연구에서는 인간의 시선 추적 데이터를 활용하여 하이브리드 비주얼 포리징(hybrid visual foraging) 작업을 수행하고, 목표의 가치와 유병률(prevalence)이 채집 및 시선 행동에 미치는 영향을 분석하였습니다. 이를 통해 AI 모델인 Visual Forager (VF)를 개발하였습니다.

- **Technical Details**: Visual Forager (VF) 모델은 변형자(transformer) 기반의 구조로, 강화 학습(reinforcement learning) 기법을 통해 훈련되었습니다. 이 모델은 다양한 타겟 객체와 그에 해당하는 가치, 검색 이미지를 입력으로 받아들이고, 포비드(foveated) 비전을 활용하여 이미지 처리 후, 일련의 시선 이동과 각 항목을 수집할지 결정하는 과정까지 시뮬레이션합니다.

- **Performance Highlights**: VF 모델은 모든 기본 모델을 능가하여 누적 보상이 인간과 유사하게 나타났으며, 시간 제약 환경 내에서의 인간의 포리징 행동을 잘 근사하였습니다. 무작위적 조건에서도 유효한 일반화 능력을 보여 주었으며, 이는 추가한 데이터 증강(data augmentation) 기법 덕분입니다.



### Rationality based Innate-Values-driven Reinforcement Learning (https://arxiv.org/abs/2411.09160)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2401.05572

- **What's New**: AI 에이전트의 내재적 동기를 설명하는 탈내재 가치 기반 강화 학습(IVRL) 모델을 제안하고, 이를 통해 인간 사회와 안전하고 조화롭게 통합될 수 있는 AI를 지원하는 방법을 탐구합니다.

- **Technical Details**: 제안된 IVRL 모델은 DQN과 A2C 알고리즘을 기반으로 하며, 역할 수행 게임(RPG) 강화 학습 테스트 플랫폼인 VIZDoom을 사용하여 성능을 검증하고 다른 벤치마크 알고리즘과 비교합니다.

- **Performance Highlights**: IVRL 모델은 다양한 개별적 요구를 합리적으로 조직함으로써 더 나은 성과를 달성할 수 있음을 보여주었습니다. 이 모델은 복잡하고 도전적인 작업에 효율적으로 적응할 수 있는 수렴성을 가집니다.



### The \emph{Optimist}: Towards Fully Automated Graph Theory Research (https://arxiv.org/abs/2411.09158)
- **What's New**: 이번 연구는 	extit{Optimist}라는 자율 시스템을 소개하며, 그래프 이론에서 자동 추측 생성 기술을 진전시키기 위해 개발되었습니다. 이 시스템은 혼합 정수 프로그래밍(mixed-integer programming, MIP)과 휴리스틱(heuristic) 방법을 활용하여 기존 정리를 재발견하고 새로운 부등식을 제안합니다.

- **Technical Details**: 	extit{Optimist}는 메모리 기반 계산과 에이전트와 같은 적응성을 결합하여, 새로운 데이터를 통합함으로써 추측을 반복적으로 정제합니다. 시스템은 파이썬으로 구현되었으며 오픈 소스이며, Jupyter 노트북과 함께 제공되어 접근성과 재현성을 지원합니다. 주요 특징은 동적 메모리 구조를 통해 그래프 불변량과 정리를 저장하며, 이로 인해 새로운 그래프가 도입될 때 신속한 검색과 점진적 업데이트가 가능해집니다.

- **Performance Highlights**: 초기 실험 결과, 	extit{Optimist}는 그래프 이론의 기초 결과를 발견하고, 미래 탐사를 위한 흥미로운 추측을 생성할 수 있는 잠재력을 보여줍니다. 이 연구는 	extit{Pessimist}라는 대응 에이전트와의 통합을 설명하며, 두 시스템이 경쟁하는 구조인 GraphMind를 형성하여 완전 자동화된 그래프 이론 연구를 추진할 것입니다.



### Set-Based Retrograde Analysis: Precomputing the Solution to 24-card Bridge Double Dummy Deals (https://arxiv.org/abs/2411.09089)
- **What's New**: 본 논문은 게임 끝의 상태에서 출발하여 이전 상태를 해결하는 기존의 retrograde analysis 방식을 개선한 setrograde analysis 알고리즘을 소개합니다. 이 알고리즘은 동일한 게임 가치를 가진 상태의 집합을 활용하여 탐색 효율성을 크게 높입니다.

- **Technical Details**: setrograde analysis는 전통적인 retrograde algorithm에 비해 10^3배 적은 탐색 작업을 수행하고, 10^4배 적은 데이터베이스 항목을 생성합니다. 이 방법은 24카드 Bridge double dummy 카드 플레이 문제에서 10^27 상태를 효율적으로 처리하고, 800조(兆) 상태 공간을 해결하는데 필요한 Computing resource를 3 OOM (order of magnitude) 줄였습니다.

- **Performance Highlights**: 제안된 알고리즘은 특정 하드웨어에서 50GiB의 저장소를 사용하여 단 주 만에 24카드 Bridge double-dummy 끝 게임을 완전히 해결할 수 있음을 보여줍니다. 이러한 성과는 retrograde searching 기술이 새로운 수준으로 확장될 수 있는 가능성을 제시합니다.



### Liner Shipping Network Design with Reinforcement Learning (https://arxiv.org/abs/2411.09068)
- **What's New**: 본 논문은 Liner Shipping Network Design Problem (LSNDP)을 해결하기 위한 새로운 강화학습( Reinforcement Learning ) 프레임워크를 제안합니다. 이 프레임워크는 전통적인 문제 해결 방법과는 다른 접근 방식을 사용합니다.

- **Technical Details**: 전통적인 LSNDP 해결 방법은 문제를 네트워크 설계(network design) 및 다중 상품 흐름(multi-commodity flow)과 같은 서브 문제로 분해합니다. 그러나 제안된 방법은 모델 없는 강화학습 알고리즘을 네트워크 설계에 적용하고, 휴리스틱 기반의 다중 상품 흐름 솔버와 통합하여 경쟁력 있는 결과를 도출합니다.

- **Performance Highlights**: 제안한 방법은 공개적으로 제공되는 LINERLIB 벤치마크에서 경쟁력 있는 성능을 보이며, 학습 후 변형된 인스턴스에서도 일반화(generalization) 능력을 발휘합니다.



### The Systems Engineering Approach in Times of Large Language Models (https://arxiv.org/abs/2411.09050)
Comments:
          This paper has been accepted for the upcoming 58th Hawaii International Conference on System Sciences (HICSS-58)

- **What's New**: 이 논문은 Large Language Models (LLMs)와 시스템 엔지니어링 접근 방식의 관계를 다루며, LLMs를 사회 기술적 시스템에 통합하는 과정의 도전 과제를 제시합니다. 특히, 시스템 엔지니어링 원칙이 이러한 문제를 해결하는 데 어떻게 기여할 수 있는지를 탐구하고, LLMs의 효과적인 사용을 위한 방향성을 제안합니다.

- **Technical Details**: LLMs는 대량의 데이터로 훈련된 신경망 아키텍처를 활용하여 언어 패턴을 학습하며, 구조적 비결정성과 오류 가능성을 가진 결과(예: hallucinations)를 산출합니다. 시스템 엔지니어링은 문제와 그 맥락을 우선시하여 이러한 AI 기술의 도입을 용이하게 합니다. 이 논문은 특히 2017년 이후의 시스템 엔지니어링 접근 방식을 통해 LLMs와 유사한 도전 과제를 해결하려는 연구를 조사합니다.

- **Performance Highlights**: 이 연구는 LLMs 적용에서 발생하는 주요 문제들을 요약하며, 이들 문제가 시스템의 신뢰성과 책임성에 미치는 영향을 강조합니다. 특히, 데이터 프라이버시 및 보안 요건의 중요성을 인식하고 LLMs의 확산이 추가적인 보안 위협을 초래할 수 있음을 경고합니다.



### Reliability, Resilience and Human Factors Engineering for Trustworthy AI Systems (https://arxiv.org/abs/2411.08981)
- **What's New**: AI 시스템의 신뢰성 및 안전성을 보장하기 위해 기존의 신뢰성 및 회복력 엔지니어링 원칙을 통합한 새로운 프레임워크를 제안합니다.

- **Technical Details**: 전통적인 메트릭인 실패율(failure rate) 및 평균 고장 간격(Mean Time Between Failures, MTBF)을 사용하여 AI 시스템 성능을 관리하고, 고장을 예방하거나 효율적으로 복구하는 방법론을 제시합니다. 이 방식은 고전적인 공학 기법을 AI 시스템에 적용하며, 향후 기술 연구를 위한 연구 계획을 개요합니다.

- **Performance Highlights**: 실제 AI 시스템에 우리의 프레임워크를 적용하여 openAI와 같은 플랫폼의 시스템 상태 데이터를 사용해 실용성을 입증합니다. 이 프레임워크는 새로운 세계적 표준 및 규제 프레임워크와 일치하여 AI 시스템의 신뢰성을 높이는 방법론을 제공합니다.



### Comment on Is Complexity an Illusion? (https://arxiv.org/abs/2411.08897)
Comments:
          Comment on arXiv:2404.07227

- **What's New**: 이 논문은 '복잡성(Complexity)은 환상인가?'라는 제목으로 복잡성과 학습, 추론(Inference), 일반화(Generalization) 등을 위한 형식적 정의를 제공합니다. 또한 '정책(Policy)'에 대한 공식 정의를 소개합니다.

- **Technical Details**: 저자는 수학적 증명과 포괄적 탐색(Exhaustive Search)을 통해 감독된 다중 클래스 분류(Supervised Multi-Class Classification)의 단순한 작업에 대해 올바른 정책이 존재하지 않음을 보여줍니다.

- **Performance Highlights**: 이 결과는 복잡성과 정책 이론에 대한 중요한 함의를 가지며, 이론에 대한 가능성 있는 반응 및 수정 방안이 논의됩니다.



### On the Surprising Effectiveness of Attention Transfer for Vision Transformers (https://arxiv.org/abs/2411.09702)
Comments:
          NeurIPS 2024. Code: this https URL

- **What's New**: 이번 연구는 전통적인 ViT(Vision Transformers) 프리트레이닝(pre-training)이 다운스트림 작업에서의 성능 향상에 얼마나 기여하는지를 탐구합니다. 연구 결과, 프리트레이닝 중 학습된 특징(feature)과 표현(representation)은 반드시 필수적이지 않다는 것을 발견했습니다. 놀랍게도, 단순히 프리트레이닝에서 얻은 attention 패턴만으로도 높은 품질의 특징을 새롭게 학습하고, 비교 가능한 성능을 달성할 수 있음을 보여줍니다.

- **Technical Details**: 연구에서는 'attention transfer'라는 간단한 방법을 제안합니다. 이를 통해 프리트레이닝된 teacher ViT로부터 attention 패턴만을 학생 모델에게 전달하여, 학생은 이러한 attention 패턴에 의해 특징을 독립적으로 학습합니다. 두 가지 방식인 Attention Copy와 Attention Distillation이 제안되며, 후자는 teacher의 attention 패턴을 학생에게 증류하여 전달합니다. 이는 프리트레이닝된 attention 맵을 활용한 학습 방법으로, 그 효과를 구분할 수 있습니다.

- **Performance Highlights**: 이 방법들은 놀라운 효과를 보였으며, Attention Copy는 대부분의 성능 차이를 해소했고, Attention Distillation은 ImageNet-1K 분류에서 훌륭한 정확도를 달성했습니다. attention transfer를 통해 학생이 스스로 특징을 학습하게 되는 점에서 기존의 파인튜닝(fine-tuning) 방식과 차별화된 성과를 보여주며, 이러한 접근법이 프리트레이닝된 ViT 활용에 있어 효과적인 대안이 될 수 있음을 제안합니다.



### Towards a Classification of Open-Source ML Models and Datasets for Software Engineering (https://arxiv.org/abs/2411.09683)
Comments:
          5 pages, 8 figures

- **What's New**: 본 연구는 오픈소스 플랫폼 Hugging Face (HF)에서 소프트웨어 공학 (Software Engineering, SE) 필요에 맞춘 Pre-Trained Models (PTMs) 및 데이터셋의 분류를 시도하며, PTMs의 시간에 따른 진화를 분석합니다.

- **Technical Details**: HF API를 사용하여 1,060,419개의 PTMs와 229,767개의 데이터셋을 수집한 후, SE 관련 메타 데이터를 분석하여 PTMs와 데이터셋을 10,077개 PTM과 1,836개 데이터셋으로 정제했습니다. 또한, Gemini 1.5 Pro를 사용하여 SE 관련성을 확인했습니다.

- **Performance Highlights**: 코드 생성을 위한 PTMs가 가장 일반적이며, 2023 Q2 이후 SE를 위한 PTMs의 수가 현저히 증가하였습니다. 그러나 현재 PTMs의 33%는 모델 카드가 없고, 65%는 SE 작업을 언급하지 않고 있으며, 데이터셋 내에서도 SE 작업 언급 비율은 0.80%에 불과합니다.



### NeuralDEM - Real-time Simulation of Industrial Particulate Flows (https://arxiv.org/abs/2411.09678)
Comments:
          Project page: this https URL

- **What's New**: 이번 연구에서는 NeuralDEM이라는 새로운 접근 방식을 통해 느린 수치 DEM(Discrete Element Method) 방정식을 빠르고 적응 가능한 딥러닝 대체 모델로 바꾸었습니다.

- **Technical Details**: NeuralDEM은 DEM의 라그랑지안(Lagrangian) 이산화(discretization)를 기본 연속 필드로 간주하며, 동시에 고맥락(macroscopic) 행동을 추가 보조 필드로 모델링합니다. 또한, 여러 분기(multi-branch) 신경 연산자(neural operators)를 도입하여 산업 규모의 시나리오를 실시간으로 모델링할 수 있도록 하였습니다.

- **Performance Highlights**: NeuralDEM은 160k CFD 셀과 500k DEM 입자가 결합된 유체화된 침대 리액터를 정확히 모델링하였으며, 28초 동안의 궤적을 추적할 수 있습니다. 이로 인해 고급 엔지니어링 및 더 빠른 공정 주기에 많은 새로운 가능성이 열릴 것입니다.



### On the Limits of Language Generation: Trade-Offs Between Hallucination and Mode Collaps (https://arxiv.org/abs/2411.09642)
Comments:
          Abstract shortened to fit arXiv limit

- **What's New**: 본 연구는 언어 모델의 생성에서 '일관성(consistency)'과 '폭(breadth)'을 동시에 충족할 수 있는지에 대한 문제를 다루고 있습니다. 이 문제에 대한 기존 연구와 반대의 결과를 제시하며, 언어 모델의 난제 중 하나인 생성의 폭과 일관성 간의 긴장 관계를 수학적으로 설명합니다.

- **Technical Details**: 알고리즘은 무한 집합으로부터 랜덤 샘플을 받아, 이를 바탕으로 모든 보이지 않는 문자열을 생성하는 것을 목표로 합니다. 이 과정에서 잘 정의된 언어 K 내에서 유의미한 데이터 분포를 학습해야 하며, 이는 기존 GAN(Generative Adversarial Networks)에서의 mode collapse 문제와 관련이 있습니다. 더불어, 샘플 수가 증가할 때 출력이 K의 모든 보이지 않는 문자열에 수렴하는지를 분석합니다.

- **Performance Highlights**: 연구 결과에 따르면, 많은 언어 모델에서는 '일관성'과 '폭'을 동시에 달성하는 것이 불가능하다는 것이 밝혀졌습니다. 그러나 긍정적인 예제와 부정적인 예제(정답 외의 문자열)가 함께 제공될 때 카운팅 가능한 모든 언어 집합에 대해 일관성 있는 폭 있는 생성을 달성할 수 있는 가능성을 제시합니다. 이는 히스토그램 피드백이 환각(hallucination)을 줄이는 데 중요한 역할을 할 수 있음을 시사합니다.



### One-Shot Manipulation Strategy Learning by Making Contact Analogies (https://arxiv.org/abs/2411.09627)
Comments:
          CoRL LEAP Workshop, 2024

- **What's New**: 본 논문에서는 새로운 접근 방식인 MAGIC(Manipulation Analogies for Generalizable Intelligent Contacts)를 제안하여 단일 시연(One-shot demonstration)을 통해 조작 전략을 신속하게 학습하고, 새로운 물체에 대한 빠르고 광범위한 일반화를 가능하게 합니다.

- **Technical Details**: MAGIC는 사전 훈련된 신경망 특성과 로컬 곡률 분석을 결합하여 정확하고 물리적으로 그럴듯한 접촉 점을 보장하는 두 단계의 접촉 점 매칭 프로세스를 기반으로 합니다. 이 방법은 참조 행동 궤적(Reference action trajectory)을 활용하여 새로운 물체에서 유사한 접촉 점과 액션 시퀀스를 효과적으로 식별합니다.

- **Performance Highlights**: MAGIC는 스쿱(scooping), 걸이(hanging), 후킹(hooking)과 같은 세 가지 작업에서 뛰어난 성능을 보여주며 기존 방법들에 비해 실행 속도와 다양한 물체 카테고리에 대한 일반화에서 상당한 개선을 달성했습니다.



### Vision-based Manipulation of Transparent Plastic Bags in Industrial Setups (https://arxiv.org/abs/2411.09623)
- **What's New**: 이 논문은 비전 기반 (vision-based) 조작 기술을 활용하여 산업 환경에서 투명 플라스틱 봉지를 자율적으로 자르고 개봉하는 과정의 도전 과제를 다룹니다. 4차 산업혁명 (Industry 4.0) 패러다임에 부합하며, 로봇 공학 및 머신 러닝 (Machine Learning) 알고리즘을 통해 향상된 접근성과 지속 가능성을 제공하는 시스템을 제안합니다.

- **Technical Details**: 제안하는 시스템은 Convolutional Neural Networks (CNNs)를 활용하여 다양한 조명 및 배경 조건에서 투명 플라스틱 봉지를 식별합니다. 추적 알고리즘과 깊이 감지 (depth sensing) 기술을 이용하여 3D 공간 인식을 구현하며, 진공 그립 기술을 통한 최적의 안전한 조작 포인트를 고려하여 효율적인 상호 작용을 가능하게 합니다.

- **Performance Highlights**: FRANKA 로봇팔을 사용하여 실험실에서 시스템을 성공적으로 테스트하였으며, 플라스틱 봉지의 개봉 및 절단 자동화에 대한 효과성을 입증했습니다. 이 시스템은 산업 전반에 걸쳐 광범위한 응용 가능성을 보여줍니다.



### PTR: Precision-Driven Tool Recommendation for Large Language Models (https://arxiv.org/abs/2411.09613)
- **What's New**: 이번 연구에서는 LLMs(대형 언어 모델)에 대한 도구 추천 문제를 새롭게 정의하고, Precision-driven Tool Recommendation (PTR) 접근 방식을 제안합니다. 이 방식은 역사적 도구 묶음 사용 데이터를 활용하여 적절한 도구 세트를 추천하며, RecTools라는 새로운 데이터셋과 TRACC라는 평가 지표를 도입하여 LLMs의 도구 추천 효과성을 평가합니다.

- **Technical Details**: PTR 접근 방식은 도구 묶음 확보, 기능 범위 매핑, 다중 뷰 기반 재정렬의 세 가지 주요 단계를 포함합니다. 이 과정에서 LLMs가 쿼리를 해결하는 데 필요한 도구를 정확히 찾고, 분류하여 추천할 수 있도록 합니다. 새로운 평가 지표 TRACC는 추천된 도구의 정확성과 품질을 모두 고려하여 도구 추천 성과를 측정합니다.

- **Performance Highlights**: 이 연구를 통해 제안된 PTR 접근 방식은 두 개의 공개 벤치마크 및 RecTools 데이터셋에서 높은 정확도를 입증하였습니다. 전반적으로 권장된 도구 세트는 효율적이고 적절하여 LLMs의 작업 처리 성능을 향상시키는 데 기여합니다.



### Local-Global Attention: An Adaptive Mechanism for Multi-Scale Feature Integration (https://arxiv.org/abs/2411.09604)
- **What's New**: 최근 몇 년 동안 객체 감지(object detection) 분야에서 중요한 진전이 있었으며, 단순한 기술 개선으로도 성능 지표가 상당히 향상되었습니다. 본 논문에서는 Local-Global Attention이라는 새로운 주의 메커니즘(attention mechanism)을 제안하며, 이는 기존의 로컬(local) 및 글로벌(global) 주의의 한계를 극복하기 위해 다중 스케일 컨볼루션(multi-scale convolution)과 위치 인코딩(positional encoding)을 통합하여 최적의 특징 표현(feature representation)을 제공합니다.

- **Technical Details**: Local-Global Attention 메커니즘은 입력 데이터의 세부 정보와 전반적인 맥락을 모두 캡처할 수 있도록 설계되었습니다. 이 메커니즘은 다중 스케일 컨볼루션을 활용하여 세밀한 특징(local features)을 추출하고, 동시에 더 큰 커널을 사용하여 글로벌 특징(global features)을 추출하여 이 둘의 정보를 융합합니다. 또한, 각 주의의 상대적 중요도를 동적으로 조절할 수 있는 학습 가능한 파라미터(learnable parameters)를 도입하여, 특정 작업의 요구사항에 따라 주의의 비율을 최적화합니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋(VOC2007, VOC2012, COCO2017 등)에서 Local-Global Attention의 성능을 평가한 결과, 이 메커니즘은 여러 키 지표에서 기존의 주의 메커니즘보다 지속적으로 뛰어난 성능을 보였으며, 특히 소규모 객체 감지(multi-class and small object detection) 작업에서 강력한 성능을 발휘했습니다. 성능 향상은 계산 효율성을 유지하면서 이루어졌습니다.



### LLaMA-Mesh: Unifying 3D Mesh Generation with Language Models (https://arxiv.org/abs/2411.09595)
Comments:
          See the project website at this https URL

- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)을 활용하여 텍스트 기반 3D 메쉬를 생성하는 새로운 방법인 LLaMA-Mesh를 제안합니다. 이 접근법은 LLMs의 기존 공간적 지식을 활용하고, 대화형 3D 생성 및 메쉬 이해를 가능하게 합니다.

- **Technical Details**: LLaMA-Mesh는 3D 메쉬의 정점 좌표와 면 정의를 일반 텍스트로 표현하여 LLM과의 통합을 용이하게 합니다. 이를 위해, OBJ 파일 형식을 사용하며, 기존의 토크나이저를 확장하지 않고 새로운 데이터 형식을 처리합니다. 또한, 감독형 미세 조정(SFT) 데이터셋을 만들어 LLM이 3D 메쉬를 이해하고 생성할 수 있도록 학습시킵니다.

- **Performance Highlights**: LLaMA-Mesh는 사전 훈련된 LLM이 3D 메쉬를 텍스트 프롬프트로부터 생성하고, 텍스트와 3D 메쉬를 혼합하여 출력할 수 있는 능력을 보여줍니다. 제안된 방법은 모델을 처음부터 훈련한 모델과 동등한 품질의 메쉬 생성을 달성하면서도 강력한 텍스트 생성 성능을 유지합니다.



### SMILE-UHURA Challenge -- Small Vessel Segmentation at Mesoscopic Scale from Ultra-High Resolution 7T Magnetic Resonance Angiograms (https://arxiv.org/abs/2411.09593)
- **What's New**: SMILE-UHURA 챌린지는 7T MRI를 사용하여 획득된 Time-of-Flight MRA 데이터에 대해 주석이 달린 데이터셋을 제공하여 뇌의 미세 혈관 세분화에 대한 연구를 촉진하는 것이 목적입니다. 이는 자동 세분화 알고리즘 개발에 필요한 공개 데이터셋의 부족 문제를 해결하고자 합니다.

- **Technical Details**: 이 챌린지에서는 7T MRI를 사용한 Time-of-Flight Angiography의 주석이 달린 데이터셋을 제공합니다. 이 데이터셋은 자동 사전 세분화와 포괄적인 수동 세분화의 조합을 통해 생성되었습니다. 16개의 제출된 방법과 2개의 기준 방법이 두 가지 다른 데이터셋에서 비교되었습니다: 같은 데이터셋의 테스트 MRA와 별도의 7T ToF MRA 데이터셋에서 검증되었습니다.

- **Performance Highlights**: 제출된 대부분의 딥러닝 방법들은 제공된 훈련 데이터셋으로 훈련되어 신뢰할 수 있는 세분화 성능을 달성했습니다. 해당 데이터를 기반으로 Dice 점수는 각각 0.838 ± 0.066 및 0.716 ± 0.125에 도달했으며, 평균 성능은 최대 0.804 ± 0.15에 달했습니다.



### Adopting RAG for LLM-Aided Future Vehicle Design (https://arxiv.org/abs/2411.09590)
Comments:
          Conference paper accepted in IEEE FLLM 2024

- **What's New**: 이 논문은 대형 언어 모델(Large Language Models, LLMs)과 Retrieval-Augmented Generation (RAG)의 통합을 자동차 산업의 자동 설계 및 소프트웨어 개발에 어떻게 활용하는지 탐구합니다.

- **Technical Details**: 연구에서는 RAG를 활용한 두 가지 사례 연구를 제시합니다: 표준화 준수 챗봇과 설계 보조 도구입니다. 또한, GPT-4o, LLAMA3, Mistral, Mixtral이라는 네 가지 LLM의 응답 정확도와 실행 시간을 비교 평가합니다.

- **Performance Highlights**: 결과적으로 GPT-4가 우수한 성능을 보였지만, LLAMA3와 Mistral도 자동차 응용 프로그램에서 데이터 프라이버시 문제를 해결하며 유망한 능력을 보여주었습니다. 이 연구는 RAG로 증강된 LLM이 자동차 공학의 설계 워크플로우 및 준수를 개선할 수 있는 잠재력을 강조합니다.



### Software Performance Engineering for Foundation Model-Powered Software (FMware) (https://arxiv.org/abs/2411.09580)
- **What's New**: 이 논문에서는 상용화 가능한 FMware(Foundation Models 소프트웨어)를 개발하는 과정에서 성능 공학의 중요성을 강조합니다. FMware의 성능 목표인 처리량(througput) 및 대기 시간(latency)을 충족시켜 사용자 불만 및 금전적 손실을 방지하기 위한 필요성이 제기되었습니다.

- **Technical Details**: FMware 개발과정에서 네 가지 주요 성능 공학(SPE) 도전 과제가 확인되었습니다: 인지 아키텍처 설계(cognitive architecture design), 통신 프로토콜(communication protocols), 조정 및 최적화(tuning and optimization), 배포(deployment). 이 논문은 산업 이해관계자 및 학계와의 깊이 있는 토론과 분석을 통해 다양한 기술적 과제를 제시합니다.

- **Performance Highlights**: FM은 새로운 소프트웨어의 축으로 자리 잡았으며, 특히 Large Language Models(LLMs)의 출현은 소프트웨어 개발의 패러다임을 변화시키고 있습니다. 효율적인 하드웨어 활용 및 지속적인 성능 조정이 FMware의 성공적인 상용화에 필수적입니다.



### Piecing It All Together: Verifying Multi-Hop Multimodal Claims (https://arxiv.org/abs/2411.09547)
- **What's New**: 새로운 멀티호프 멀티모달 주장 검증 작업을 도입하여 텍스트, 이미지, 표 등 다양한 출처의 여러 증거를 통해 주장의 정확성을 평가합니다.

- **Technical Details**: MMCV는 16,000개의 멀티호프 주장을 담은 대규모 데이터셋으로, 대규모 언어 모델(LLMs)을 활용하여 데이터 주석을 생성하고 인간의 피드백으로 품질을 보장합니다.

- **Performance Highlights**: 최신 멀티모달 대형 언어 모델(MLLMs)조차 복잡한 멀티모달 주장을 검증하는 데 어려움을 겪으며, 특히 추론 단계가 증가할수록 성능이 제한됩니다.



### OpenGeMM: A High-Utilization GeMM Accelerator Generator with Lightweight RISC-V Control and Tight Memory Coupling (https://arxiv.org/abs/2411.09543)
- **What's New**: 이번 연구에서는 자원 제약이 있는 엣지(Edge) 장치에서 딥 뉴럴 네트워크의 효율성을 높이기 위해 OpenGeMM이라는 오픈 소스 가속화 플랫폼을 제안했습니다. 이 플랫폼은 높은 효율성과 활용도, 구성의 용이성을 동시에 달성합니다.

- **Technical Details**: OpenGeMM은 매개변수화된 Chisel 코드로 작성된 GeMM 가속기, 경량 RISC-V 프로세서, 그리고 밀접하게 결합된 멀티 뱅크 스크래치패드( scratchpad) 메모리를 포함하고 있습니다. 시스템의 효율성은 구성 사전 로딩(configuration pre-loading), 입력 사전 페칭(input pre-fetching) 및 출력 버퍼링(output buffering), 프로그래머블 스트라이드 메모리 접근(programmable strided memory access)을 통해 향상됩니다.

- **Performance Highlights**: OpenGeMM은 다양한 CNN 및 Transformer 작업 부하에서 하드웨어 활용도가 81.89%에서 99.34%에 이르는 것으로 나타났습니다. 기존의 Gemmini 가속기와 비교할 때, OpenGeMM은 GeMM 작업 부하에서 정규화된 처리량에서 3.58배에서 16.40배의 속도 향상을 달성하였으며, 4.68 TOPS/W의 시스템 효율성을 기록했습니다.



### Prompting the Unseen: Detecting Hidden Backdoors in Black-Box Models (https://arxiv.org/abs/2411.09540)
- **What's New**: 이 연구에서는 black-box 모델 수준에서 백도어(Backdoor) 탐지를 위한 새로운 방법론인 BProm을 제안합니다. 이 방법은 시각적 프롬프트(Visual Prompting, VP)를 활용하여 원본 도메인의 훈련된 모델을 사용하여 타겟 도메인 작업에 적응시키는 기술을 기반으로 합니다.

- **Technical Details**: BProm은 VP를 사용하여 의심스러운 모델에 클린(cleam) 데이터셋을 적용합니다. 모델의 저조한 분류 정확도를 사용하여 백도어가 존재함을 식별합니다. 이 연구에서는 class subspace inconsistency라는 개념을 명확히 하여 감염된 모델과 클린 데이터셋 간의 불일치를 확인합니다.

- **Performance Highlights**: BProm의 성능 검증을 위한 광범위한 실험이 수행되었으며, 이는 의심스러운 모델에서의 백도어 탐지 효과를 입증합니다. 실험 결과, BProm은 백도어가 있는 모델에서 낮은 분류 정확도를 보였음을 나타냈습니다.



### Communication Compression for Tensor Parallel LLM Inferenc (https://arxiv.org/abs/2411.09510)
- **What's New**: 대규모 언어 모델(LLMs)의 속도 향상을 위해, 본 연구에서는 Tensor Parallel 전략을 이용한 다수의 하드웨어 가속기에서의 모델 배치를 다루며, 활발한 협업을 통한 지연 시간을 줄이는 방법을 제안합니다.

- **Technical Details**: 최적화된 inter-accelerator communication을 위해 활동량을 3.5 - 4.5배 압축하는 세밀한 양자화 기법을 활용했습니다. 이 방법은 타임 투 퍼스트 토큰(TTFT)을 최대 2배 줄이는 결과를 가져오면서도 모델 성능 저하는 거의 없는 수준입니다.

- **Performance Highlights**: 기존 하드웨어 세팅에서 실행할 때, 느린 inter-accelerator 대역폭 환경에서의 TTFT가 3.5 - 4.5배 향상될 수 있음을 발견했습니다.



### Toward a Cohesive AI and Simulation Software Ecosystem for Scientific Innovation (https://arxiv.org/abs/2411.09507)
Comments:
          5 pages

- **What's New**: 이 논문에서는 인공지능(AI)과 모델링 및 시뮬레이션(ModSim) 도구를 통합한 소프트웨어 스택의 필요성을 논의합니다. 통일된 AI/ModSim 소프트웨어 생태계를 통해 다양한 고성능 컴퓨팅 시스템에서의 호환성을 보장하고 배포 및 버전 관리를 용이하게 하자는 주장을 하고 있습니다.

- **Technical Details**: 주요 과제는 AI와 ModSim의 독특한 요구사항(특히 소프트웨어 빌드 관행, 의존성 관리 및 호환성)의 균형을 맞추는 것입니다. 지속적인 통합(Continuous Integration), 커뮤니티 기반 관리 및 에너지부(DOE)와의 협업을 강조하여 휴대 가능하고 응집력 있는 과학 소프트웨어 생태계를 개발해야 함을 강조합니다.

- **Performance Highlights**: 표준화된 환경을 지원하기 위한 이니셔티브로 Extreme-scale Scientific Software Stack(E4S)와 Spack을 활용하여 학제 간 혁신을 촉진하고 새로운 과학적 발전을 가능하게 하려는 제안이 포함되어 있습니다.



### MM-Eval: A Hierarchical Benchmark for Modern Mongolian Evaluation in LLMs (https://arxiv.org/abs/2411.09492)
- **What's New**: 이 논문은 현대 몽골어에 대한 대형 언어 모델(LLM)의 능력을 평가하기 위한 MM-Eval이라는 전문화된 데이터셋을 소개합니다. 이 데이터셋은 LM의 언어 능력(구문 및 의미)과 인지 능력(지식 및 추론)을 분리해 평가함으로써 몽골어와 같은 저자원 언어 지원의 격차를 해소하는 것을 목표로 합니다.

- **Technical Details**: MM-Eval 데이터셋은 569개의 구문, 677개의 의미, 344개의 지식, 250개의 추론 문제를 포함합니다. 이는 LLM의 성능을 분석하기 위해 구문(Syntax), 의미(Semantics), 지식(Knowledge), 추론(Reasoning) 네 가지 구성요소로 나뉘어져 있습니다.

- **Performance Highlights**: 예비 실험 결과에 따르면, 모든 모델이 구문 작업에서 의미 작업보다 더 우수한 성능을 보였으며, 이는 깊은 언어 이해에서의 격차를 강조합니다. 또한 지식 작업에서도 중간 수준의 성능 저하가 관찰되어, 모델들이 고자원 언어에서 저자원 언어로 일반 지식을 이전할 수 있음을 시사합니다.



### ResidualDroppath: Enhancing Feature Reuse over Residual Connections (https://arxiv.org/abs/2411.09475)
- **What's New**: 본 논문에서는 기본적인 residual connection의 한계를 분석하고, 이를 보완하기 위한 ResidualDroppath 알고리즘을 제안합니다. 이는 딥러닝 네트워크에서 특징 재사용(feature reuse)을 더 효과적으로 학습할 수 있도록 돕는 새로운 훈련 방법론입니다.

- **Technical Details**: ResidualDroppath 알고리즘은 두 가지 유형의 반복(iteration)을 포함합니다. 첫 번째 반복에서는 droppath를 사용하여 랜덤하게 몇 개의 레이어를 드롭하여 특징 재사용을 강제합니다. 두 번째 반복은 드롭된 모델 부분을 학습하는 동안 남은 부분을 고정(freeze)하여 특징 재사용을 장려하도록 합니다. 이를 통해 모델은 남은 부분을 참고하여 더욱 활용적으로 학습할 수 있습니다.

- **Performance Highlights**: 이 알고리즘은 ResNet50 및 ResNet50d 모델에 적용되어 CIFAR10과 MNIST 데이터셋에서 Top-1 및 Top-5 정확도에서 상당한 성능 향상을 보였으며, ImageNet1K 데이터셋에서도 ResNet50d 모델의 성능 개선이 관찰되었습니다.



### Renal Cell Carcinoma subtyping: learning from multi-resolution localization (https://arxiv.org/abs/2411.09471)
- **What's New**: 이 연구에서는 Renal Cell Carcinoma의 하위 유형 분류를 위한 새로운 self-supervised training 전략을 제시합니다. 이 방법은 주석이 달린 데이터셋의 필요성을 줄이고, 정확도를 크게 저하하지 않으며, 병리학자의 의사결정 과정을 모방하여 다양한 배율에서 학습된 특성을 통합합니다.

- **Technical Details**: 이 연구의 접근법은 Whole Slide Images (WSIs)를 기반으로 하며, self-supervised learning (SSL) 기법을 활용하여 암 유형을 분류합니다. SSL은 데이터 자체에서 얻은 라벨을 사용하여 semi-automatic 과정으로 supervised pretext task를 수행하며, 복잡한 주석을 요구하지 않습니다.

- **Performance Highlights**: 본 연구에서는 제안된 방법이 ccRCC, pRCC, chRCC 및 ONCO 등 4개의 주요 RCC 하위 유형을 분류하는 데 성공적이며, 기존의 fully supervised 방법들과 유사한 성능을 달성합니다. 이는 대규모 주석 데이터 필요성을 줄이는 동시에 정확도를 유지할 수 있음을 보여줍니다.



### An Explainable Attention Model for Cervical Precancer Risk Classification using Colposcopic Images (https://arxiv.org/abs/2411.09469)
Comments:
          19 pages, 9 figure, and 7 tables

- **What's New**: 이번 연구에서는 Cervix-AID-Net 모델을 통해 자궁경부 전암 위험 분류를 위한 혁신적인 방법을 제안하였습니다. 이 모델은 환자 콜포스코피 이미지를 기반으로 설계되어, 고위험군과 저위험군을 구별할 수 있는 해석 가능하고 대표적인 특징을 추출합니다.

- **Technical Details**: Cervix-AID-Net 모델은 Convolutional Block Attention Module (CBAM)과 여러 convolutional layer로 구성되어 있으며, 콜포스코픽 이미지의 특징을 추출하여 전암 위험을 분류합니다. 또한 이 모델은 gradient class activation maps, Local Interpretable Model-agnostic Explanations, CartoonX, pixel rate distortion explanation과 같은 네 가지 설명 가능한 기술을 통합하고 있습니다.

- **Performance Highlights**: Cervix-AID-Net 모델은 holdout 및 10-겹 교차 검증을 통해 각각 99.33% 및 99.81%의 분류 정확도를 달성했으며, Gaussian noise와 blur의 영향을 받지 않아 성능이 일정 수준 유지되는 것으로 나타났습니다. 모델은 다른 Deep Learning 접근법보다 우수한 성능을 보이며 자궁경부 전암 위험 평가의 보조 도구로서 가능성을 보여줍니다.



### DiffRoad: Realistic and Diverse Road Scenario Generation for Autonomous Vehicle Testing (https://arxiv.org/abs/2411.09451)
Comments:
          14 pages, 9 figures

- **What's New**: DiffRoad라는 새로운 diffusion model을 통해 고품질의 3D 도로 시나리오를 생성하는 방법을 제안합니다. 이 모델은 도로 환경의 복잡성과 다양성을 포착하여 자율 주행 차량의 테스트에 적합한 현실적이고 다양한 도로 시나리오를 자동으로 생성할 수 있습니다.

- **Technical Details**: DiffRoad는 흰 잡음으로부터 도로 레이아웃을 합성하는 역 노이즈 제거 과정을 통해 도로 시나리오를 생성하고, Road-UNet 아키텍처를 사용하여 생성된 도로의 질을 최적화합니다. 또한, 도로의 연속성과 합리성을 평가하기 위한 도로 시나리오 평가 모듈을 도입하여, OpenDRIVE 형식으로 자동 변환하여 자율 주행 시뮬레이션 테스트에 활용할 수 있습니다.

- **Performance Highlights**: DiffRoad는 실제 데이터셋에 대한 실험을 통해 고해상도 도로 시나리오를 생성하며, 원래 분포를 유지하면서 원활한 도로 구조를 구현하는 능력을 입증하였습니다. 이로 인해 자율 주행 차량 테스트를 위한 풍부하고 다양한 시나리오 라이브러리를 제공하며, 미래 인프라 설계에 대한 통찰력을 제시합니다.



### AI-driven inverse design of materials: Past, present and futur (https://arxiv.org/abs/2411.09429)
Comments:
          43 pages, 5 figures, 2 tables

- **What's New**: 이 논문은 AI-driven 재료의 역설계(inverse design) 분야에서 최신 진전을 소개하고, 기능성 재료의 효율적인 설계를 가능하게 하는 인공지능 기술의 발전을 조망합니다.

- **Technical Details**: 금속물질, 전자구조(electronic structure), 밀도기능이론(density functional theory), 고속 컴퓨팅 방법론(high-throughput computational methods) 등을 통해 물질의 다차원, 비선형 매핑(high-dimensional, nonlinear mapping)을 구축합니다. 특히, AI 모델이 물질의 특성과 결정 구조(crystal structure) 간의 복잡한 연관성을 파악하는 방법을 강조합니다.

- **Performance Highlights**: AI를 활용한 재료 발견이 기하급수적으로 증가하고 있으며, 효율적인 예측을 통해 재료 개발 사이클을 단축시키는 성과가 있음을 보여줍니다. GNoME와 OMat24와 같은 플랫폼의 기여로, 물질 발견 과정이 빠르게 진행되고 있습니다.



### SAG-ViT: A Scale-Aware, High-Fidelity Patching Approach with Graph Attention for Vision Transformers (https://arxiv.org/abs/2411.09420)
Comments:
          10 pages, 4 figures, 3 tables

- **What's New**: 본 논문에서 소개하는 Scale-Aware Graph Attention Vision Transformer (SAG-ViT)는 멀티스케일 특징 표현을 효율적으로 통합하여 image classification을 향상시키는 새로운 프레임워크입니다. EfficientNet을 백본으로 사용하고, 그래프 구조를 통해 특징 맵을 처리함으로써 고급 의미 정보를 유지하며 고유한 관계를 모델링합니다.

- **Technical Details**: SAG-ViT는 EfficientNet을 기반으로 한 멀티스케일 특징 맵을 추출하여 이를 패치로 나누고, 공간적 및 특징 유사성에 따라 그래프를 구성합니다. 그래프 내의 각 노드는 패치를 나타내며, Graph Attention Network (GAT)를 통해 최신의 패치 정보를 동적으로 강조합니다. 이후 Transformer 인코더가 장기 종속성과 복잡한 상호작용을 포착합니다.

- **Performance Highlights**: SAG-ViT는 여러 벤치마크 데이터셋에서 평가되었으며, 기존의 Transformer 기반 접근 방법과 비교하여 이미지 분류 성능을 높이는 데 성공하였습니다.



### Script-centric behavior understanding for assisted autism spectrum disorder diagnosis (https://arxiv.org/abs/2411.09413)
Comments:
          5 pages, 4 figures, submitted to ICASSP 2025

- **What's New**: 이 논문은 비지도 학습을 기반으로 한 새로운 접근 방식을 도입하여 ASD(자폐 스펙트럼 장애)을 자동으로 감지합니다. 기존의 지도 학습 방법에 비해 ASD 진단을 더욱 효과적으로 다룰 수 있는 기술적 진전을 보여줍니다.

- **Technical Details**: 이 방법은 비디오 내용을 스크립트로 변환하는 'Behavioral Transcription Module(BTM)', 스크립트의 행동 데이터를 처리하여 LLMs(Large Language Models)와의 연결을 다리 역할을 하는 'Script Transcription Module(STM)', 그리고 도메인 지식을 결합하는 'Domain Prompts Module(DPM)'의 세 가지 모듈로 구성됩니다.

- **Performance Highlights**: 이 연구에서 제안한 방법은 24개월 된 아동에서 ASD 진단 정확도가 92.00%로, 기존의 지도 학습 방법보다 3.58% 높은 성과를 보여 주었습니다. 이러한 결과는 LLMs의 비디오 데이터에 대한 이해력이 ASD 연구에 중요한 기여를 할 수 있음을 나타냅니다.



### Quantum Machine Learning: An Interplay Between Quantum Computing and Machine Learning (https://arxiv.org/abs/2411.09403)
Comments:
          In submission

- **What's New**: 이번 논문은 양자 컴퓨팅(quantum computing)과 기계 학습(machine learning) 간의 융합을 통해 양자 기계 학습(quantum machine learning, QML) 아키텍처를 개발하는 방법을 소개합니다. 특히, 다양한 산업에 대한 QML 연구의 잠재적 영향을 탐구합니다.

- **Technical Details**: 이 논문에서는 변분 양자 회로(variational quantum circuits, VQC)를 이용하여 잡음 중간 규모 양자(NISQ) 장치에서 QML 아키텍처를 개발하는 방법을 다룹니다. 강력한 최적화 기법을 통해 VQC의 파라미터 조정을 통해 데이터에 적합한 회로 구조를 생성하며, 고전적 기계 학습(classical machine learning)과의 하이브리드 아키텍처를 통해 자원을 최대한 활용하는 방안도 제시됩니다.

- **Performance Highlights**: 양자 컴퓨터는 전통적인 방법에 비해 기계 학습 성능을 크게 향상시킬 수 있는 잠재력을 지니고 있으며, 변분 양자 회로(VQC)는 실제 응용에서 양자 잡음에 강한 저항성을 보입니다. PCA 및 SVM의 경우, 양자 알고리즘을 통해 기하급수적인 성능 개선을 기대할 수 있습니다.



### Automated Segmentation of Ischemic Stroke Lesions in Non-Contrast Computed Tomography Images for Enhanced Treatment and Prognosis (https://arxiv.org/abs/2411.09402)
Comments:
          7 pages, 3 figures, MICCAI Meets Africa Workshop

- **What's New**: 이 연구에서는 비대조 컴퓨터 단층 촬영(NCCT) 이미지를 사용하여 허혈성 뇌졸중 병변 세그멘테이션(segmentation)의 자동화 방법을 제안하였습니다. nnU-Net 프레임워크를 기반으로 하여 초기 치료를 개선하고 허혈성 뇌졸중 환자의 예후를 향상시키는 것을 목표로 했습니다.

- **Technical Details**: 본 연구는 Acute Ischemic stroke Dataset (AISD)를 사용하였으며, NCCT 스캔은 허혈 증상이 시작된 지 24시간 이내에 수집되었습니다. Residual Encoder U-Net 아키텍처를 채택하여 50 epochs 동안 훈련이 진행되었습니다. 데이터 정규화, 슬라이스 처리 및 패치 크기 조정 등 사전 처리 과정이 포함되었습니다.

- **Performance Highlights**: 초기 평가에서 Dice 점수는 0.596, IoU 점수는 0.510으로 나타났습니다. 아웃라이어를 조정한 후, Dice 점수는 0.752로 향상되었고 IoU 점수는 0.643으로 개선되었습니다. 이 결과는 NCCT에서 허혈성 병변을 자동으로 세그먼트화하는 데 있어 효과적인 잠재력을 보여줍니다.



### Less is More: Unseen Domain Fake News Detection via Causal Propagation Substructures (https://arxiv.org/abs/2411.09389)
Comments:
          9 pages, 2 figures, 5 tables

- **What's New**: 본 논문에서는 인-distribution(내적 분포) 데이터에서 인과 서브그래프(causal subgraphs)를 추출하여 zero-shot(제로샷) 가짜 뉴스 탐지를 개선하기 위한 Causal Subgraph-oriented Domain Adaptive Fake News Detection (CSDA) 모델을 제안합니다.

- **Technical Details**: CSDA 모델은 그래프 신경망(graph neural network)을 기반으로 한 마스크 생성 과정을 통해 전파 그래프(propagation graph) 내의 주요 노드와 엣지를 식별하고, 이를 가짜 뉴스 탐지에 활용합니다. 모델은 이진 마스크를 사용하여 각 노드와 엣지를 인과적 요소(causal elements) 또는 편향 요소(biased elements)로 분류합니다. 또한, 제한된 OOD 데이터에서의 few-shot(소수 샷) 상황에서 대조 학습(contrastive learning)을 통해 성능을 개선합니다.

- **Performance Highlights**: CSDA는 공개 소셜 미디어 데이터셋에서 OOD 가짜 뉴스 탐지를 수행하며, 다른 최신 모델들 대비 7%에서 16%의 정확도 향상을 달성하였습니다.



### LTLf+ and PPLTL+: Extending LTLf and PPLTL to Infinite Traces (https://arxiv.org/abs/2411.09366)
- **What's New**: 이번 논문에서는 LTLf+(LTLf plus)와 PPLTL+(PPLTL plus)이라는 새로운 논리를 소개합니다. 이들 논리는 기존의 LTLf 및 PPLTL을 기반으로 하여 무한 상태 추적에 대한 속성을 표현할 수 있게 해줍니다. 기존의 LTL처럼 안전(progress) 및 진행(safety) 속성의 계층구조를 활용하며, 결정론적 유한 오토마타(DFA)로부터 전략 추출을 위한 게임 아레나를 도출할 수 있는 특성을 가집니다.

- **Technical Details**: LTLf+/PPLTL+의 정합성 및 결정 가능성은 각각 2EXPTIME-완전과 EXPTIME-완전입니다. LTLf+의 경우는 LTLf와 비교했을 때 같은 급수에 해당하며, PPLTL+는 LTL의 모든 표현 능력을 보존하지만, 반응 합성 문제는 EXPTIME-완전으로 줄어듭니다. 이들 기술은 만족 가능성(satisfiability), 유효성(validity), 모델 확인(model-checking)을 최적화하는 데에도 적용되었습니다.

- **Performance Highlights**: LTLf+에 대한 만족 가능성 문제는 EXPSPACE-완전으로 확장되며, PPLTL+는 PSPACE-완전성을 가집니다. 이 연구결과는 LTLf와 PPLTL의 성공적인 합성 기법을 LTL로 확대하는 중요한 발판이 됩니다.



### Your Fixed Watermark is Fragile: Towards Semantic-Aware Watermark for EaaS Copyright Protection (https://arxiv.org/abs/2411.09359)
- **What's New**: 이번 논문에서는 Embedding-as-a-Service (EaaS)의 저작권 보호를 위한 이전 워터마킹 기법들이 의미적으로 독립적이라는 점을 지적하고, Semantic Perturbation Attack (SPA)이라는 새로운 공격 방법을 제안합니다. 이 공격은 현재의 워터마킹 시스템을 우회할 수 있는 능력을 가지고 있습니다.

- **Technical Details**: 기존의 워터마킹 기법들은 입력 텍스트의 의미와 관계없이 고정된 신호를 주입합니다. 이로 인해 공격자가 워터마크 신호를 탐지하고 제거할 수 있는 Semantic Perturbation Attack (SPA)가 가능해집니다. 이를 방어하기 위해 Semantic Aware Watermarking (SAW) 기법을 제안하여, 입력 텍스트의 의미에 따라 워터마크 신호를 동적으로 주입하도록 설계했습니다.

- **Performance Highlights**: SAW 기법은 SPA에 대해 95% 이상의 True Positive Rate (TPR)를 보여줍니다. 이는 기존 방식보다 우수한 성능을 보이며, 기존의 워터마킹 스킴들이 무력화되는 것을 방지할 수 있습니다.



### EEG-Based Speech Decoding: A Novel Approach Using Multi-Kernel Ensemble Diffusion Models (https://arxiv.org/abs/2411.09302)
- **What's New**: 이 연구에서는 전극(EEG) 기반의 발화(over speech) 분류를 위한 앙상블 학습 프레임워크를 제안합니다. 다양한 컨볼루션 커널 크기를 가진 Denoising Diffusion Probabilistic Models(DDPMs)을 활용하여 신호에 내재된 다중 규모의 시간적 특성을 효과적으로 포착하는 데 중점을 둡니다.

- **Technical Details**: 이 앙상블은 51, 101, 201의 커널 크기를 가진 세 모델로 구성되어 있으며, 다중 스케일 시간적 특성을 포착합니다. 이 접근 방식은 신경 신호의 복잡한 시간적 변동성을 수용하여 음성 디코딩의 강건성 및 정확성을 향상시킵니다. 또한, Conditional Autoencoders(CAEs)와 함께 작업하여 재구성된 신호를 정제하고 다운스트림(classification) 작업을 위한 유용한 정보를 극대화합니다.

- **Performance Highlights**: 제안된 앙상블 기반 접근 방식은 개별 모델과 기존의 최신 기술에 비해 상당히 뛰어난 성능을 보였습니다. 이 결과는 뇌 신호 디코딩의 발전에서 앙상블 방법의 잠재력을 입증하며, 특히 언어 장애인 개인을 지원하는 뇌-컴퓨터 인터페이스 시스템에 대한 비언어적 통신 응용 프로그램에서의 새로운 가능성을 제시합니다.



### Learning Hand State Estimation for a Light Exoskeleton (https://arxiv.org/abs/2411.09294)
- **What's New**: 본 논문에서는 가벼운 외골격을 활용한 재활용 손 상태(Hand State)의 기계 학습 기반 추정기를 제안합니다. 이 기술은 손의 개방 정도와 순응 수준을 재구성하여 치료 진행 상황을 평가하고 적응형 제어 행동을 개발하는 데 유용합니다.

- **Technical Details**: 제안된 접근 방식은 팔뚝의 근육 활동을 측정하는 Electromyography (EMG) 센서와 외골격의 운동 정보를 사용하여 손의 상태를 추정합니다. 이 시스템은 감독 학습(Supervised Learning)에 기반하여 환자의 손의 실제 동작을 학습합니다.

- **Performance Highlights**: 실험 결과, 동일 사용자의 데이터로 훈련하고 동일한 사용자에게 테스트한 경우 예측 성능이 우수하며, 다양한 세션에서도 안정적인 일반화 능력을 보여줍니다. 이러한 점에서 제안된 시스템은 실제 재활에 유망하게 사용될 수 있습니다.



### StreamAdapter: Efficient Test Time Adaptation from Contextual Streams (https://arxiv.org/abs/2411.09289)
Comments:
          22 Pages, 9 Figures

- **What's New**: 본 논문에서는 기존의 in-context learning (ICL) 방식의 한계를 극복하기 위해 StreamAdapter라는 새로운 방법을 제안합니다. StreamAdapter는 모델 파라미터를 테스트 시간에 직접 업데이트하여 명시적인 in-context demonstration이 필요하지 않도록 합니다.

- **Technical Details**: StreamAdapter는 context mapping 및 weight absorption 메커니즘을 사용하여 ICL demonstration을 파라미터 업데이트로 동적으로 변환합니다. 이 접근법은 추가적인 파라미터 없이 ICL의 장현을 활용하여 모델이 새로운 작업에 적응할 수 있게 지원합니다.

- **Performance Highlights**: 다양한 작업과 모델 아키텍처를 통해 수행된 실험에서 StreamAdapter는 ICL과 유사하거나 우수한 적응 능력을 보여주었으며, 더 적은 demonstration으로도 효율적인 추론을 가능하게 했습니다. 또한, StreamAdapter는 상수 시간 복잡도로 추론 비용을 크게 줄이고, 더 나은 강인성과 성능을 입증했습니다.



### Cross-Modal Consistency in Multimodal Large Language Models (https://arxiv.org/abs/2411.09273)
- **What's New**: 이 연구에서는 시각(vision) 및 언어(language) 모달리티 간의 상호작용을 탐구하고 이를 비교하는 새로운 개념인 cross-modal consistency를 도입합니다. 또한 이 개념에 기반한 정량적 평가 프레임워크를 제안합니다.

- **Technical Details**: 연구에서는 Vision Large Language Models (VLLMs)에 대해 전방위적으로 평가하는 새로운 접근 방식을 제시하며, GPT-4V 모델의 비전과 언어 간의 일관성(notably, inconsistency)도 드러냅니다. 이를 위해 다양한 비전-언어 병렬 데이터세트를 구성하고 실험을 수행했습니다.

- **Performance Highlights**: GPT-4V의 성능은 동일한 태스크 인스턴스에 대해 서로 다른 모달리티에서 다르게 나타나는 경향이 있으며, 이로 인해 Vision-Depicting-Prompting (VDP) 방법론을 도입하여 개선 가능성을 모색합니다. 이러한 발견은 향후 멀티모달 모델 사용에 있어 더 효과적인 활용 방안을 제시합니다.



### Harnessing multiple LLMs for Information Retrieval: A case study on Deep Learning methodologies in Biodiversity publications (https://arxiv.org/abs/2411.09269)
- **What's New**: 본 연구에서는 5개의 오픈 소스 Large Language Models (LLMs)를 사용하여 과학적 출판물에서 Deep Learning (DL) 방법론 정보를 자동으로 추출하는 새로운 접근 방식을 제안합니다. 이 방식을 통해 DL 연구의 투명성과 재현성을 높이는 것을 목표로 합니다.

- **Technical Details**: 연구에서는 Llama-3, Mixtral, Gemma와 같은 다양한 LLM을 활용하며, Retrieval-Augmented Generation (RAG) 방식을 통해 DL 관련 정보를 자동으로 추출합니다. 이 과정에서 다수의 LLM 출력을 조합한 투표 분류기를 개발하여 정확한 정보 전달을 도모합니다.

- **Performance Highlights**: 우리의 접근 방식을 통해 DL 방법론 정보를 텍스트 내용만으로 대조했을 때 69.5%의 정확도를 달성했습니다. 이 결과는 DL 방법론 보고의 투명성을 높이고, 과학적 연구에서의 정보 재현성을 향상시키는 데 기여할 것으로 기대됩니다.



### How Good is ChatGPT at Audiovisual Deepfake Detection: A Comparative Study of ChatGPT, AI Models and Human Perception (https://arxiv.org/abs/2411.09266)
- **What's New**: 이 연구에서는 대규모 언어 모델인 ChatGPT의 오디오 및 비디오 콘텐츠에서 딥페이크를 탐지하는 능력을 평가합니다. 기존의 비디오 조작 탐지 방법들과 ChatGPT의 탐지 성능을 비교하고, Prompt Engineering(프롬프트 엔지니어링)의 역할을 강조합니다.

- **Technical Details**: 이 연구는 딥페이크 탐지를 위해 LLMs(대규모 언어 모델)를 활용하는 방법을 제안합니다. 실험은 벤치마크 멀티모달 딥페이크 데이터셋의 비디오를 사용하여 수행하였으며, ChatGPT는 다양한 오디오 및 비주얼 아티팩트(visual and auditory artifacts)를 분석하는 데 적용되었습니다. 제안된 방법은 입력 비디오에 대한 시각적, 청각적 분석을 포함하여 깊이 있는 분석을 제공합니다.

- **Performance Highlights**: 실험 결과 ChatGPT가 멀티모달 딥페이크 탐지에서 인간 및 최신 AI 모델들과 비교하여 경쟁력 있는 성능을 보였으며, 특히 프롬프트 설정에 따라 탐지 성능이 달라지는 것으로 나타났습니다. 그러나 ChatGPT는 탐지 과정의 해석 가능성(interpretability) 부족과 특정 조작에 대한 일반화의 한계를 갖고 있습니다.



### Automating Autograding: Large Language Models as Test Suite Generators for Introductory Programming (https://arxiv.org/abs/2411.09261)
Comments:
          Submitted to Journal of Computer Assisted Learning

- **What's New**: 이 연구는 Large Language Models (LLMs)를 활용하여 CS1 수준의 프로그래밍 문제에 대한 테스트 슈트를 자동 생성하는 방법을 제시했습니다. 이를 통해 수업 시간에서 프로그래밍 과제를 자동 채점하는 시스템의 효율성을 높이고자 합니다.

- **Technical Details**: 이 연구에서는 26개의 프로그래밍 문제 샘플과 25,000개 이상의 학생의 시도된 해결책을 사용하여, LLM이 생성한 테스트 슈트의 성능을 교사가 생성한 테스트 슈트와 비교했습니다. 특히 GPT-4가 문제의 진술과 참조 솔루션을 바탕으로 테스트 슈트를 생성합니다.

- **Performance Highlights**: 연구 결과, LLM이 생성한 테스트 슈트는 대부분의 유효한 솔루션을 적절히 식별할 수 있었으며, 문제의 대다수에서 교사가 생성한 테스트 슈트와 동등하거나 그 이상으로 포괄적이었습니다. 더불어 LLM이 생성한 테스트 슈트를 통해 일부 문제 진술의 모호성을 발견할 수 있었고, 이는 자율 채점 및 교육 설계를 개선하는 데 도움이 될 잠재력을 보여줍니다.



### Enhancing Financial Domain Adaptation of Language Models via Model Augmentation (https://arxiv.org/abs/2411.09249)
- **What's New**: 이 연구는 Composition to Augment Language Models (CALM)라는 모델을 통해 대형 언어 모델(LLMs)의 금융 도메인(domain)에 대한 적응 가능성을 검증하였습니다.

- **Technical Details**: CALM은 서로 다른 기능을 가진 두 개의 LLM 간의 cross-attention을 도입하여 기존 모델의 능력을 확장하는 모델입니다. 연구팀은 금융 특화 LLM을 활용하여 응답 능력이 강한 기존 LLM의 금융 성능을 향상시키는 방안을 모색했습니다. CALM은 금융 특화 LLM의 훈련에 사용된 데이터셋과는 다른 금융 데이터셋을 사용하여 훈련되었습니다.

- **Performance Highlights**: CALM은 정량적인 일본 금융 벤치마크와 정성적인 응답 비교를 통해 원래 모델들과 기준선(baselines)보다 더 높은 점수를 기록하며 우수한 응답을 제공할 수 있음을 보여주었습니다. 또한 모델의 중간 레이어를 연결하는 것이 금융 도메인 적응에 가장 효과적이라는 결과를 확인했습니다.



### Programming with AI: Evaluating ChatGPT, Gemini, AlphaCode, and GitHub Copilot for Programmers (https://arxiv.org/abs/2411.09224)
Comments:
          8 pages

- **What's New**: 이 연구는 ChatGPT, Gemini(Bard AI), AlphaCode, GitHub Copilot과 같은 최신 프로그래밍 보조 도구에 대한 포괄적인 평가를 제공합니다. 이 모델들이 소프트웨어 개발에서 수행하는 역할과 그 중요성에 대해 강조하였습니다.

- **Technical Details**: 이 논문은 자연어 처리(NLP) 및 코드 생성 정확성을 기준으로 Java, Python, C++와 같은 다양한 프로그래밍 언어에서의 성능을 평가하였고, Transformer 아키텍처를 기반으로 한 모델들을 비교하였습니다. 이들 각 모델은 고유한 훈련 데이터셋과 함께 Transformer 아키텍처를 사용하고 있습니다.

- **Performance Highlights**: 모델들은 높은 언어 이해 및 코드 생성 능력을 보였으나, 결과의 신뢰성과 정확성을 높이기 위한 개선이 필요함을 강조하였습니다. 또한, 이러한 AI 모델의 윤리적 개발 관행을 위한 필요성도 제기되었습니다.



### Transferable Adversarial Attacks against ASR (https://arxiv.org/abs/2411.09220)
Comments:
          IEEE SPL

- **What's New**: 본 논문은 자동 음성 인식(ASR) 모델의 견고성을 평가하기 위해 흑상자(black-box) 공격의 취약성을 심도 있게 연구하였으며, 고급 타임 도메인 기반 전이 가능 공격 방법을 제안합니다. 또한 음성 인식에 대한 그래디언트 최적화 접근법인 SAGO(Speech-Aware Gradient Optimization)를 도입하여, 사람의 인지와 유사한 방식으로 공격할 수 있는 새로운 방법론을 제공합니다.

- **Technical Details**: 이 연구에서는 MI-FGSM(Momentum Iterative Fast Gradient Sign Method)와 VMI-FGSM(Variance Tuning Momentum Iterative Fast Gradient Sign Method)을 사용하여 시간 도메인 오디오 신호에서 차별화된 특징 추출 디자인을 통해 직접적으로 작동합니다. 특히 음성 활동 감지(VAD: Voice Activity Detection)와 그래디언트 최적화를 결합하여, 음성 신호에 대한 공격을 강화합니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, 제안된 방법은 두 가지 데이터베이스에서 다섯 개 모델에 대한 기초 접근 방식 대비 성능 향상을 이룬 것으로 나타났습니다.



### Comprehensive and Practical Evaluation of Retrieval-Augmented Generation Systems for Medical Question Answering (https://arxiv.org/abs/2411.09213)
- **What's New**: 이 연구는 의료 분야에서 질문-답변(QA) 시스템의 신뢰성을 평가하기 위한 새로운 간행물인 Medical Retrieval-Augmented Generation Benchmark (MedRGB)를 소개합니다. MedRGB는 LLMs의 여러 가지 상황에서의 성능을 평가하기 위해 4개의 테스트 시나리오를 포함하고 있습니다.

- **Technical Details**: MedRGB는 3480개의 인스턴스로 구성되어 있으며, 이를 통해 모델의 강도와 취약점을 평가하기 위해 7개의 LLMs를 테스트합니다. 테스트 시나리오는 Standard-RAG, Sufficiency, Integration, Robustness로 나뉘며, 각 시나리오는 LLMs의 정보 통합과 노이즈 처리 능력을 평가합니다.

- **Performance Highlights**: 실험 결과, 현재 모델들은 검색된 문서에서의 노이즈와 잘못된 정보 처리에 한계를 보였습니다. 이 연구는 RAG 시스템의 신뢰성을 높이기 위한 향후 방향에 대한 통찰을 제공합니다.



### RibCageImp: A Deep Learning Framework for 3D Ribcage Implant Generation (https://arxiv.org/abs/2411.09204)
- **What's New**: 이 연구는 인공지능 기반의 자동화된 갈비뼈 임플란트 생성 프로세스를 제안합니다. 이는 기존의 전통적인 CAD 방식에서 벗어나 환자 맞춤형 임플란트 디자인을 위한 새로운 패러다임을 제시합니다.

- **Technical Details**: 3D U-Net 아키텍처를 활용하여 CT 스캔을 처리하고 환자 개별의 갈비뼈 임플란트를 자동으로 생성합니다. 데이터 준비와 모델 학습을 위해 RibFrac 데이터셋을 사용하여 420개의 CT 샘플로 훈련하고, 160개를 테스트하며, 80개를 검증합니다.

- **Performance Highlights**: 초기 결과는 보통 수준이지만, 자동화된 갈비뼈 재구성을 위한 상당한 가능성과 도전을 강조합니다. 이 연구는 향후 연구를 위한 기초를 마련하고, 임플란트 디자인의 시간 단축과 품질 향상에 기여할 수 있습니다.



### Dynamic technology impact analysis: A multi-task learning approach to patent citation prediction (https://arxiv.org/abs/2411.09184)
- **What's New**: 본 연구는 특허 인용 정보를 사용하여 기술의 영향을 분석하는 데 있어 머신 러닝(ML) 모델의 한계를 극복하기 위해 다중 작업 학습(Multi-task Learning, MTL) 접근법을 제안합니다.

- **Technical Details**: ML 기반의 기존 방법들은 시간이 지남에 따라 기술 영향의 동적 특성과 서로 다른 기간의 이러한 영향 간의 상호 의존성을 충분히 고려하지 못하는 경향이 있습니다. 본 연구에서는 인용 분석을 통해 기술 영향의 패턴을 식별하고, 여러 특허 지표를 사용하여 반복적으로 인용 수를 예측하는 MTL 모델을 개발합니다. 또한, SHapley Additive exPlanation 방법을 사용하여 주요 입력 지표의 변화와 패턴을 분석합니다.

- **Performance Highlights**: 전지 기술에 대한 사례 연구를 통해 본 접근법이 기술 영향 이해를 심화시키고 예측 정확도를 개선함으로써 학계와 산업계 모두에 유용한 통찰을 제공함을 보여줍니다.



### DeBaTeR: Denoising Bipartite Temporal Graph for Recommendation (https://arxiv.org/abs/2411.09181)
- **What's New**: 이 논문에서는 명시적 사용자 피드백의 부족으로 인해 대체 데이터 출처로서의 암시적 피드백의 문제를 다루고, 시간 정보를 활용하여 추천 시스템의 노이즈를 제거하고 예측 성능을 향상시키는 두 가지 방법인 DeBaTeR-A와 DeBaTeR-L을 제안합니다.

- **Technical Details**: 사용자-아이템 상호작용은 이분 그래프(bipartite graph)로 모델링되며, 시간 정보를 활용하여 사용자의 특성과 아이템의 특성을 고려한 시간을 인식하는 사용자/아이템 임베딩을 생성합니다. DeBaTeR는 인접 행렬의 가중치를 재조정하거나 손실 함수의 가중치를 재조정하여 노이즈 상호작용을 식별하고 제거합니다.

- **Performance Highlights**: 광범위한 실험을 통해 제안된 방법들이 최신 모델에 비해 우수한 성능과 견고성을 보여주었으며, 실제 데이터셋에서 높은 정확도를 기록하였습니다.



### LEAP:D - A Novel Prompt-based Approach for Domain-Generalized Aerial Object Detection (https://arxiv.org/abs/2411.09180)
Comments:
          ICIP 2024 Workshop accepted paper

- **What's New**: 이 논문에서는 드론으로 촬영한 이미지에서의 객체 탐지를 향상시키기 위해 학습 가능한 프롬프트(learnable prompts)를 활용한 혁신적인 비전-언어 접근법을 제안합니다.

- **Technical Details**: 제안된 LEAP:D 접근법은 대규모 비전-언어 모델을 기반으로 하여 수동 프롬프트를 제거하고 간소화된 하나의 단계의 훈련 프로세스를 촉진합니다. 이를 통해 객체 탐지 모델이 도메인 불변(features) 특성을 보존하고 도메인 특정 특성의 영향을 최소화하는 방식으로 훈련됩니다.

- **Performance Highlights**: LEAP:D 방식은 다양한 환경에서도 효과적으로 작동할 수 있는 강력하고 적응력 있는 모델을 생성하여 항공 객체 탐지의 성능을 향상시킵니다. 기존의 방식에 비해 효율적인 훈련이 가능해졌습니다.



### Advancing Diffusion Models: Alias-Free Resampling and Enhanced Rotational Equivarianc (https://arxiv.org/abs/2411.09174)
Comments:
          13 pages, 7 figures

- **What's New**: 최근의 이미지 생성 기술, 특히 diffusion models의 발전은 이미지 합성 품질의 인상적인 개선을 가져왔습니다. 본 논문에서는 올바르지 않은 재샘플링 작업이 aliasing을 유발하여 이미지 품질을 저하시킨다고 가정하며, 이 문제를 해결하기 위해 alias-free resampling을 UNet 아키텍처에 통합하는 방법을 제안합니다.

- **Technical Details**: 우리는 alias-free resampling을 통해 diffusion models의 UNet 구조에 새로운 학습 가능한 매개변수를 추가하지 않고 모델 효율성을 유지하며 성능을 향상시키는 것을 목표로 하고 있습니다. 이를 통해 높은 주파수 아티팩트를 방지하며 이미지의 회전 동등성을 향상시킬 수 있습니다. 또한, 사용자 제어에 의한 이미지 회전을 가능하게 하는 수정된 diffusion 프로세스를 제안합니다.

- **Performance Highlights**: 실험 결과, CIFAR-10, MNIST 및 MNIST-M과 같은 벤치마크 데이터셋에서 이미지 품질이 일관되게 향상되었음을 확인했습니다. 특히 FID 및 KID 점수에서 향상이 두드러지며, 이는 더 나은 결과를 위해 이미지 처리 원리에 의해 안내된 적절한 alias-free resampling 레이어의 사용 가능성을 강조합니다.



### Towards Scalable Handwriting Communication via EEG Decoding and Latent Embedding Integration (https://arxiv.org/abs/2411.09170)
Comments:
          4 pages, 2 figures, 1 table, Name of Conference: International Conference on Brain-Computer Interface

- **What's New**: 이 연구에서는 EEG 신호와 함께 필기 동작에 대한 카테고리를 분류하는 새로운 접근 방식인 CEBRA(consistent embeddings from high-dimensional neural recordings using auxiliary variables)를 사용하여 더욱 세밀한 필기 인식을 달성했습니다.

- **Technical Details**: CEBRA는 행동 데이터와 신경 데이터를 함께 활용하여 해석 가능한 일관된 임베딩을 생성합니다. 이 방법은 EEG 신호와 손의 운동학 데이터를 통합하여 CNN(Convolutional Neural Network)을 통해 처리하여 고차원 신경 기록에서 의미 있는 저차원 임베딩을 추출합니다.

- **Performance Highlights**: 모델은 9개의 다양한 손글씨 문자를 분류하여 91%의 정확도를 달성하였으며, 이 연구의 접근 방식은 기존 CNN, EEGNet 및 DeepConvNet보다 우수한 성능을 보였습니다.



### Artificial Theory of Mind and Self-Guided Social Organisation (https://arxiv.org/abs/2411.09169)
Comments:
          4 pages

- **What's New**: 이 논문에서는 인공지능(AI) 분야의 주요 도전 과제 중 하나로, 여러 에이전트의 협력이 필요하다는 점을 강조합니다. 기존의 연구와 함께 AI-AI 경계에서도 인지 과정을 존중해야 한다는 주장을 합니다.

- **Technical Details**: 논문에서는 집단 지능(collective intelligence)의 일반적인 설정을 제시하며, 신경망에서의 단일 신경 세포 복잡성과 개미 집단의 유연성을 다루고 있습니다. 생태적 네트워크에서 종 간의 관계를 설명하기 위해 니치 선택(niche selection), 니치 선택(niche choice), 그리고 니치 복종(niche conformity) 개념을 도입하고, 이를 통해 인간 사회적 네트워크의 발전과 유사성을 이끌어냅니다.

- **Performance Highlights**: 사회 구조는 개인의 신경 생리학(neuro-physiology), 심리학(psychology), 언어(linguistics)의 영향을 받으며, 이로 인해 사회 네트워크 내에서의 개인의 역할이 복잡한 과제 수행에 미치는 영향을 강조합니다. 또한, 이와 같은 집단 인공지능은 스스로 사회적 구조를 이끌어낼 수 있는 잠재성을 지니고 있다고 결론지었습니다.



### Theory of Mind Enhances Collective Intelligenc (https://arxiv.org/abs/2411.09168)
Comments:
          20 pages, 2 figures, 1 table

- **What's New**: 이 논문은 인간 집단 지능의 심리적 과정이 사회적 수준에서의 자기 조직화 구조의 출현과 어떻게 연결되는지에 대해 깊이 있는 통찰을 제공합니다.

- **Technical Details**: 집단 지능(Collective Intelligence, CI)은 개인이 아닌 집단이 달성할 수 있는 복잡한 행동을 다루며, 이 과정에서 개별 에이전트(예: 신경 세포, 개미 등)의 상호작용 네트워크의 토폴로지가 중요합니다. 저자들은 정보 이론(Information Theory)을 적용하여 CI의 계산적 성격을 정량화하려 합니다.

- **Performance Highlights**: 인간 사회 집단 지능이 심리적 요인에 의해 개선되는 여러 예를 제시하며, 이론적 사고(Theory of Mind)의 발달이 사회적 집단 지능과 일반 집단 지능을 구별짓는 핵심 요인임을 강조합니다.



### ABCI 3.0: Evolution of the leading AI infrastructure in Japan (https://arxiv.org/abs/2411.09134)
Comments:
          4 pages, 2 figures

- **What's New**: ABCI 3.0는 AIST가 운영하는 대규모 오픈 AI 인프라의 최신 버전으로, 2018년 8월 이래로 운영되어 왔으며, 2025년 1월에 완전 가동 예정입니다. 이 시스템은 6128개의 NVIDIA H200 GPU와 올플래시 스토리지 시스템을 갖추고 있으며, 이전 버전인 ABCI 2.0보다 7배에서 13배 빠른 성능을 자랑합니다.

- **Technical Details**: ABCI 3.0는 766개의 Compute Node (H)와 75PB의 공유 파일 시스템 및 클라우드 스토리지를 포함합니다. 이 네트워크는 InfiniBand NDR/HDR을 통해 연결되며, 모든 Compute Node는 HPE Cray XD670로 구성되어 있습니다. GPU, 메모리 및 SSD의 조합으로 성능 최적화를 이루어냅니다. 또한, Open OnDemand 지원 및 Altair PBS Professional job scheduler를 통해 자원의 효율적 관리가 가능합니다.

- **Performance Highlights**: ABCI 3.0는 반 정밀도에서 6.22 exaflops, 단 정밀도에서 3.0 exaflops의 최대 성능을 제공하며, 데이터 저장 용량과 이론적 읽기/쓰기 성능이 두 배 이상 증가했습니다. 특히 생성형 AI 연구 및 개발 가속화를 위한 기반을 마련하고 있습니다.



### DROJ: A Prompt-Driven Attack against Large Language Models (https://arxiv.org/abs/2411.09125)
- **What's New**: 이번 연구는 Directed Representation Optimization Jailbreak (DROJ)라는 새로운 기법을 소개하여, LLM의 프롬프트에서 악의적인 쿼리의 숨겨진 표현을 변경하여 모델이 긍정적인 반응을 유도하는 방향으로 최적화하는 방법을 제안합니다.

- **Technical Details**: DROJ는 주어진 쿼리의 숨겨진 상태를 분석하고, 주성분 분석 (PCA)을 사용하여 낮은 차원 공간으로 프로젝션 한 후, 각 쿼리의 거부 확률을 예측하기 위해 로지스틱 회귀 모델을 적합합니다. 이 과정을 통해, harmful query와 harmless query 모두 모델의 긍정적인 응답을 유도하기 쉬운 방향으로 전환될 수 있습니다.

- **Performance Highlights**: DROJ는 LLaMA-2-7b-chat 모델에서 100%의 키워드 기반 공격 성공률 (Attack Success Rate, ASR)을 달성했으며, 직접적인 거부를 효과적으로 방지합니다. 다만, 모델의 응답이 때때로 반복적이고 비정보적일 수 있으며, 이를 해결하기 위해 유용성 시스템 프롬프트를 도입하여 응답의 품질을 향상시켰습니다.



### VCBench: A Controllable Benchmark for Symbolic and Abstract Challenges in Video Cognition (https://arxiv.org/abs/2411.09105)
- **What's New**: 이 논문에서는 VCBench라는 새로운 비디오 인지 벤치마크를 소개하여, 상징적 및 추상적 개념을 포함하는 LVLM(대규모 비디오-언어 모델)의 인지 능력을 평가할 수 있도록 합니다.

- **Technical Details**: VCBench는 Python 기반 엔진을 사용하여 동적이고 과제 지향적인 비디오를 생성하며, 복잡한 장면과 추상 개념을 포함합니다. 각 작업은 특정 인지 과제를 대상으로 하는 맞춤형 질문 템플릿과 결합되어 있습니다. 주요 평가 차원으로는 Object Perception, Action Perception, Spatial Reasoning, Temporal Reasoning 등이 있습니다.

- **Performance Highlights**: 이번 평가에서 Qwen2-VL-72B와 같은 최신 모델조차 상징적 요소를 포함한 간단한 비디오 인지 작업에서 어려움을 겪었으며, 비디오 복잡성이 증가함에 따라 성능이 19% 급감했습니다. 이는 LVLM의 고급 인지 작업에 대한 한계를 명확히 드러냅니다.



### Provocation: Who benefits from "inclusion" in Generative AI? (https://arxiv.org/abs/2411.09102)
Comments:
          3 pages, 1 figure. Published as a Short Paper in the NeurIPS 2024 Workshop on Evaluating Evaluations: Examining Best Practices for Measuring Broader Impacts of Generative AI

- **What's New**: 이번 논문에서는 생성 AI (Generative AI)의 개발과 평가에서 사회적 소외 집단을 포괄하는 참여적 구조의 중요성을 강조하고, 이러한 구조가 미치는 혜택과 해로움을 명확히 규명해야 한다고 주장합니다. 또한, 저자는 자신들의 경험을 통해 논의된 사례 연구를 바탕으로 참여적 구조가 어떻게 변할 수 있는지에 대한 통찰을 제공합니다.

- **Technical Details**: 저자들은 AI 개발에서 사회적 소외 집단의 참여가 필수적이며, 이를 통해 기대할 수 있는 혜택과 해로움을 체계적으로 분석하였습니다. 이들은 'data leverage'라는 개념을 통해 기여자들이 데이터셋에서 그들의 권리를 어떻게 행사할 수 있는지를 다루고 있습니다.

- **Performance Highlights**: 논문은 참여적 구조의 한계를 드러내고 있으며, 더 나아가 소외된 집단이 AI 개발과 평가에서 어떤 방식으로 더 효과적으로 참여할 수 있을지를 탐색합니다. 특히, 커뮤니티 중심의 데이터 사용 모델을 개발하는 사례들이 제시되어 실질적인 방향성을 제안합니다.



### Heuristical Comparison of Vision Transformers Against Convolutional Neural Networks for Semantic Segmentation on Remote Sensing Imagery (https://arxiv.org/abs/2411.09101)
- **What's New**: 본 논문에서는 Remote Sensing 항공 이미지의 semantic segmentation에 있어 ViT(비전 트랜스포머)의 사용을 비교하고, 새로운 결합 가중 손실 함수(combined weighted loss function)를 제안하여 UNet CNN 모델의 성능을 향상시키는 연구 결과를 발표합니다.

- **Technical Details**: 본 연구에서는 UNet에 새로운 결합 가중 손실 함수를 도입하여 mIoU(평균 교차분할 비율) 점수를 극대화하고, 불확실성을 보존하면서 강력한 마스크 예측을 가능하게 합니다. 또한, Meta의 MaskFormer와 일반적인 UNet CNN 간의 전이 학습을 비교하여 학습 효율성과 추론 시간에서의 성능 차이를 분석합니다.

- **Performance Highlights**: 실험 결과, 잡음 제거 및 배경 클래스 관리와 같은 다양한 기법들이 CNN 모델 성능을 크게 개선하는 데 도움이 되는 것으로 나타났으며, 특정 상황에서 ViT 모델보다 뛰어난 결과를 보여주었습니다. 새로운 손실 함수는 데이터셋에서 더 나은 일반화를 가능하게 하며, ViT 기반 모델에 대한 비교에서 더 높은 성능을 달성했습니다.



### Drone Detection using Deep Neural Networks Trained on Pure Synthetic Data (https://arxiv.org/abs/2411.09077)
Comments:
          12 pages, 8 figures

- **What's New**: 이 논문은 순수한 합성 데이터셋으로 훈련된 드론 탐지 Faster-RCNN 모델을 제시하며, 이는 실제 데이터로의 전이 가능성을 입증하였습니다. 실험 결과, MAV-Vid 데이터셋에서 97.0%의 AP_50 성능을 달성하였으며, 이는 실제 데이터로 훈련된 모델의 97.8%와 유사합니다.

- **Technical Details**: 모델 훈련에 사용된 합성 데이터셋은 Structured Domain Randomization (SDR) 기법을 이용하여 생성되었습니다. 드론 탐지의 정확도 향상을 위해, 다양한 조명, 텍스쳐 및 자세를 무작위로 변경하여 실제 환경과 유사한 시나리오를 구현하였습니다. 이를 통해 모델이 실제 데이터에서도 성능을 발휘할 수 있도록 하였습니다.

- **Performance Highlights**: 드론 탐지 분야에서 합성 데이터를 활용함으로써, 데이터 수집 비용을 줄이고 레이블 품질을 향상시킬 수 있는 가능성을 보여주었습니다. 이 연구 결과는 앞으로 더 정교한 합성 드론 데이터셋 개발의 초석이 될 수 있습니다. 또한, 안전이 중요한 애플리케이션인 공항 드론 탐지의 데이터셋 생성을 리스크를 줄일 수 있습니다.



### Code-mixed LLM: Improve Large Language Models' Capability to Handle Code-Mixing through Reinforcement Learning from AI Feedback (https://arxiv.org/abs/2411.09073)
Comments:
          initial version: 5 pages, 2 figures

- **What's New**: 본 논문은 코드-믹싱(code-mixing) 및 코드-스위칭(code-switching) 캠에 대한 다국어 대형 언어 모델(LLM)의 성능을 평가하고, AI 피드백을 통한 강화 학습(Reinforcement Learning from AI Feedback, RLAIF)을 통해 모델의 이러한 혼합 언어 이해 능력을 개선하는 방법을 제안합니다.

- **Technical Details**: 코드-믹싱은 두 개 이상의 언어가 혼합된 언어 사용을 의미하며, 이를 효과적으로 처리하기 위한 다국어 LLM의 성능을 벤치마킹합니다. 성능 향상을 위해, 기존의 LLM에 RLAIF 방법론을 적용하여 코드-믹싱 기계 번역 작업을 개선합니다.

- **Performance Highlights**: 실험 결과는 RLAIF 방법이 코드-믹싱을 처리하는 LLM의 성능을 향상시킬 가능성이 있음을 보여주고 있습니다. RLAIF와 같이 AI 피드백을 활용하는 접근 방식이 향후 더 유연하고 인간 중심의 AI 시스템 개발에 기여할 수 있을 것입니다.



### Language-Model Prior Overcomes Cold-Start Items (https://arxiv.org/abs/2411.09065)
Comments:
          This paper is dedicated to cold-start item recommendation using language-model priors

- **What's New**: 이 논문은 추천 시스템에서 아이템의 콜드 스타트 문제를 해결하기 위해 언어 모델(Language Model, LM)을 활용하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 제안된 프레임워크는 Bayesian 정규화를 RecSys의 훈련 과정에 통합하여 아이템 간의 의미적 유사성을 고려합니다. LM이 인코딩한 사전 지식을 활용하여 연속 잠재 공간에서 세밀한 아이템 임베딩을 학습합니다.

- **Performance Highlights**: 실험을 통해 SASRec 및 BPRMF와 같은 추천 시스템에 통합하여 두 개의 실제 데이터 세트에서 평가한 결과, 제안된 접근 방식이 SASRec의 정규화 할인 누적 이익(Normalized Discounted Cumulative Gain)을 17.78% 향상시켰습니다.



### Multimodal Object Detection using Depth and Image Data for Manufacturing Parts (https://arxiv.org/abs/2411.09062)
- **What's New**: 이번 연구는 RGB 카메라와 3D 포인트 클라우드 센서를 결합한 다중 센서 시스템을 제안하며, 기존의 단일 센서 방식의 한계를 극복합니다.

- **Technical Details**: 제안하는 방법은 RGB 이미지를 처리하기 위해 설계된 Faster R-CNN 기반의 다중 모달 객체 탐지 방법입니다. RGB 영상과 깊이 데이터를 동시에 처리하여 더 정확하고 안정적인 객체 탐지를 구현하였습니다.

- **Performance Highlights**: 이 모델은 RGB 전용 기준 대비 mAP(Mean Average Precision)에서 13% 개선, Depth 전용 기준 대비 mAP에서 78% 개선되었습니다. 이는 스마트 제조 애플리케이션에서의 객체 탐지 신뢰성을 크게 향상시킵니다.



### SAFELOC: Overcoming Data Poisoning Attacks in Heterogeneous Federated Machine Learning for Indoor Localization (https://arxiv.org/abs/2411.09055)
- **What's New**: 이 논문에서는 모바일 기기에서의 indoor localization(실내 위치 추적) 문제를 해결하기 위한 새로운 프레임워크인 SAFELOC을 제안합니다. SAFELOC은 기기 이질성(device heterogeneity) 및 ML 데이터 중독 공격(data poisoning attack)과 같은 도전 과제를 극복하는 데 중점을 두고 있습니다.

- **Technical Details**: SAFELOC은 분산 및 협력적 학습 환경을 목표로 하며, federated learning (FL)을 사용하여 사용자 데이터의 프라이버시를 보호합니다. heterogenous(이질적) 모바일 기기를 고려하여 설계된 새로운 융합 신경망 아키텍처(fused neural network architecture)로 데이터 중독 탐지와 위치 추적을 수행합니다. 또한, 탐지된 데이터 중독 상황의 심각도에 따라 적응하는 동적 주목도 맵(dynamic saliency map)을 기반으로 한 집계 전략(aggregation strategy)을 도입합니다.

- **Performance Highlights**: SAFELOC은 다양한 빌딩 평면도, 모바일 기기, ML 데이터 중독 공격 시나리오에 따라 기존의 indoor localization 프레임워크에 비해 평균 위치 추적 오류(mean localization error)를 최대 5.9배 감소시키고, 최악의 경우 위치 추적 오류(worst-case localization error)를 7.8배까지 줄이며, 모델 추론 지연(latency)을 2.1배 줄이는 성능 향상을 보여줍니다.



### IDCIA: Immunocytochemistry Dataset for Cellular Image Analysis (https://arxiv.org/abs/2411.08992)
- **What's New**: 이 연구에서는 기계 학습 방법의 효율성을 향상시키기 위해 새로운 주석이 달린 미세 세포 이미지 데이터셋을 제시합니다. 이 데이터셋은 세포 수 카운팅을 자동화할 수 있는 가능성을 제시하며, 다양한 항체로 염색된 세포 이미지가 포함되어 있습니다.

- **Technical Details**: 이 데이터셋은 전기 자극 과정 후 촬영된 Adult Hippocampal Progenitor Cells (AHPCs)의 이미지를 포함하고 있습니다. 이미지는 ICC(면역세포화학)를 통해 수집되었으며, 각 이미지는 세포의 위치와 수량을 주석 처리하였습니다. DNN(Deep Neural Networks) 모델의 성능 비교 결과, 기존 모델들이 수동 카운트를 대체할 만큼의 정확도를 달성하지 못했습니다.

- **Performance Highlights**: 제안된 데이터셋은 다양한 항체 염색 방법을 포함하여, 공개된 다른 데이터셋들보다 더 많은 이미지를 제공합니다. 연구팀의 실험 결과, 현재 DNN 기반 세포 카운팅 방법들은 정확도 면에서 한계가 있으며, 이 데이터셋은 향후 연구 및 개선의 기준이 될 수 있습니다.



### CoCoP: Enhancing Text Classification with LLM through Code Completion Promp (https://arxiv.org/abs/2411.08979)
- **What's New**: 코드 완성 프롬프트(Code Completion Prompt, CoCoP) 방법을 제안하여 대형 언어 모델(LLMs)의 텍스트 분류 성능을 개선했습니다. 이 방법은 텍스트 분류 문제를 코드 완성 작업으로 변환하여 LLMs의 코드 완성 기능을 활용합니다.

- **Technical Details**: CoCoP는 LLMs의 코드 완성 능력을 활용하고, in-context learning 기법을 사용합니다. 이 방법은 단계별로 불완전한 코드를 생성하여 LLM이 사용자 쿼리에 대한 적절한 레이블을 결정할 수 있도록 합니다. CoCoP는 LLaMA2 및 CodeLLaMA 모델을 기반으로 다양한 분류 데이터셋에서 검증되었습니다.

- **Performance Highlights**: CoCoP는 SST2 데이터셋의 정확도를 20% 이상 향상시켰습니다. 코드 관련 작업을 위해 설계된 LLMs(코드 모델)와 통합할 경우, CoCoP는 소규모 모델(7B 및 13B)로도 70B 모델과 비슷하거나 우수한 성능을 보였습니다.



### Fluoroformer: Scaling multiple instance learning to multiplexed images via attention-based channel fusion (https://arxiv.org/abs/2411.08975)
Comments:
          Findings paper presented at Machine Learning for Health (ML4H) symposium 2024, December 15-16, 2024, Vancouver, Canada, 14 pages

- **What's New**: 이번 논문에서는 Multiplexed Whole Slide Images (WSIs) 처리를 위한 새로운 다중 샘플 학습(Multiple Instance Learning, MIL) 전략인 Fluoroformer 모듈을 제안합니다. 이 방법은 세포 이미지와 생물학적 구조에 대한 중요한 정보를 융합하여 해석 가능한 출력을 제공합니다.

- **Technical Details**: Fluoroformer는 Scaled Dot-Product Attention (SDPA) 기술을 활용하여 서로 다른 채널의 정보를 하나의 요약 벡터로 융합합니다. 이로 인해 각 패치를 위한 주의(attention) 행렬을 생성할 수 있으며, 이는 세포 간 상호작용 및 생물학적 구조에 대한 통찰력을 제공합니다.

- **Performance Highlights**: 434개의 비소세포 폐암(non-small cell lung cancer, NSCLC) 샘플에서 Fluoroformer는 예측 성능이 뛰어나며, 면역 종양 상호작용에 대한 중요한 관찰 결과를 도출하였습니다. 이는 공간 생물학적 기술과 최신 인공지능 모델을 결합하여 이 emerging 분야의 잠재력을 극대화합니다.



### Inconsistencies In Consistency Models: Better ODE Solving Does Not Imply Better Samples (https://arxiv.org/abs/2411.08954)
Comments:
          NeurIPS 2024 ATTRIB Workshop

- **What's New**: 최근 Consistency Models (CMs)이 새로운 확산 모델 증류 방법으로 주목받고 있으며, 이들은 몇 번의 반복만으로도 고품질 샘플을 생성할 수 있는 가능성을 보여줍니다. 본 논문에서는 기존 확산 모델의 확률 흐름 ODE(ordinary differential equation)를 직접 최소화하는 Direct CMs를 소개합니다.

- **Technical Details**: Direct CMs는 CMs와 비교하여 ODE 해결 오류를 줄이는데 더 효과적이지만, 샘플 품질이 현저히 저하되는 반응을 보였습니다. 이는 ODE 해결이 꼭 샘플 품질 향상으로 이어지지 않는다는 새로운 시사점을 제공합니다.

- **Performance Highlights**: 실험 결과, Direct CMs는 PF ODE를 보다 잘 해결하지만 예기치 않게 샘플 품질이 나빠지는 결과를 보여, CMs와 그 변형들에서 암묵적으로 가정된 ‘좋은 ODE 해결이 좋은 샘플 품질을 보장한다’는 개념에 의문을 제기합니다.



### Confidence-aware Denoised Fine-tuning of Off-the-shelf Models for Certified Robustness (https://arxiv.org/abs/2411.08933)
Comments:
          26 pages; TMLR 2024; Code is available at this https URL

- **What's New**: 최근 발표된 논문에서는 일반적인 사전 훈련된 분류기의 적대적 강건성을 높이기 위한 새로운 방법인 FT-CADIS를 제안합니다. 이 방법은 특히 역압축된 이미지를 구분하기 위해 신뢰도를 활용합니다.

- **Technical Details**: FT-CADIS는 신뢰도를 고려하여 회복된 이미지를 선택하는 세부 조정 구조를 가지고 있습니다. 이 구조는 두 가지 손실(Confidence-aware selective cross-entropy loss, Confidence-aware masked adversarial loss)을 사용하여 비환각화된 이미지만을 사용하여 분류기를 조정합니다. 이로 인해 분류기는 적대적 강건성을 향상시킬 수 있습니다.

- **Performance Highlights**: FT-CADIS는 CIFAR-10 및 ImageNet 벤치마크에서 기존의 최첨단 방법들과 비교하여 적대적 강건성 면에서 상당한 성능 향상을 보여주었습니다. 특히, FT-CADIS는 높은 가우시안 분산 조건에서 29.5%에서 39.4%의 성능 향상을 기록하였습니다.



### PyGen: A Collaborative Human-AI Approach to Python Package Creation (https://arxiv.org/abs/2411.08932)
Comments:
          33 pages, 13 figures

- **What's New**: Pygen은 연구자, 기술자, 그리고 취미 활동가들이 추상적인 아이디어를 구체적이고 사용 가능한 소프트웨어 도구로 전환할 수 있도록 지원하는 자동화 플랫폼입니다. 이 플랫폼은 autoregressive large language models의 힘을 활용하여 인간의 창의성을 증장시키고, Python 패키지를 자동 생성하여 도구 개발의 수작업 부담을 상당히 줄여줍니다.

- **Technical Details**: Pygen은 사용자 입력에 기반하여 Python 패키지를 자동으로 생성하며, 전체 작업 흐름을 포괄하는 패키지 생성 및 문서화를 포함합니다. 이 시스템은 사용자 요청을 개선하여 더욱 구체적이고 실행 가능한 패키지 설명으로 발전시키며, LLM 기반 평가 및 Human Evaluation을 통해 생성된 패키지와 문서의 품질을 평가했습니다.

- **Performance Highlights**: Pygen은 연구자의 생산성을 크게 향상시키며, 탄력적이고 모듈화된 패키지를 효율적으로 생성할 수 있도록 지원합니다. 사용자는 필요한 패키지 유형과 기능을 지정할 수 있으며, Pygen은 이를 바탕으로 최적의 구현 전략을 수립합니다. 또한, 생성된 패키지와 문서가 한 곳에 모두 제공되어 사용자들이 즉시 활용할 수 있습니다.



### Retrieval of sun-induced plant fluorescence in the O$_2$-A absorption band from DESIS imagery (https://arxiv.org/abs/2411.08925)
Comments:
          submitted to ECCV CVPPA 2024, 14 pages, 8 figures

- **What's New**: 이 논문에서는 공간 기반의 SIF (Sun-Induced Fluorescence) 맵을 30m의 지상 해상도로 추출할 수 있는 최초의 방법을 제시합니다. 이 방법은 고품질의 공중 기반 SIF 추정과 강한 상관관계(r^2=0.6)를 가지고 있습니다. SIF 데이터는 농업 관리 및 생리학적 연구와 관련된 다양한 작업에 대한 설명 정보를 제공할 수 있습니다.

- **Technical Details**: 본 연구는 ESA의 FLEX 미션에 대비하여 O$_2$-A 반대역의 하이퍼스펙트럼 DESIS 이미지를 활용하여 SIF의 추출 방법을 개발했습니다. SFMNN(Spectral Fitting Method Neural Network)이라는 시뮬레이션 기반의 자기 감독 네트워크를 훈련시키고, 대기 변수 예측을 위한 추가적인 감독 규제를 통해 성능 향상을 테스트했습니다. 검증 연구 결과, 모델은 740 nm에서 HyPlant로부터 유도된 SIF 추정치와의 평균 절대 차이 0.78 mW / nm / sr / m²를 기록했습니다.

- **Performance Highlights**: DESIS 이미지를 통해 얻은 SIF 제품은 전례 없는 30m의 공간 해상도를 제공하여, FLEX 미션을 위한 보조 유효성 검증 데이터를 높은 해상도로 수집할 수 있게 합니다. 기존의 전통적인 SIF 추출 방법으로는 DESIS의 SR과 SNR이 부족하지만, 제안된 방법은 이러한 제약을 극복하고 일관된 SIF 추출을 가능하게 합니다.



### A Machine Learning based Hybrid Receiver for 5G NR PRACH (https://arxiv.org/abs/2411.08919)
Comments:
          6 pages, 9 figures

- **What's New**: 이 논문에서는 사용자가 기지국(BS)에 자신을 식별하는 랜덤 액세스 절차에서의 하이브리드 수신기 설계를 소개합니다. 하이브리드 수신기는 AI/ML 모델을 기반으로 한 프리앰블 감지와 전통적인 피크 검출을 결합하여 타이밍 어드벤스(TA)를 추정합니다.

- **Technical Details**: 하이브리드 수신기는 여러 안테나의 상관 창에서의 전력 지연 프로파일(Power Delay Profile, PDP)을 결합하여 신경망(Neural Network) 모델의 입력으로 사용합니다. 이 모델은 특정 프리앰블 창에서 사용자의 존재 여부를 예측하며, 이후 피크 검출을 통해 타이밍 어드벤스를 추정합니다.

- **Performance Highlights**: 실험 결과에 따르면, 이 하이브리드 수신기는 전통적인 상관 기반 수신기와 다른 AI/ML 기반 접근 방식들보다 우수한 성능을 보이며, 시뮬레이션 데이터와 실제 하드웨어 수집 데이터에서 모두 성능이 향상되었습니다.



### Wireless Federated Learning over UAV-enabled Integrated Sensing and Communication (https://arxiv.org/abs/2411.08918)
Comments:
          Accepted to IEEE Conference on Standards for Communications and Networking (CSCN), 6 pages

- **What's New**: 이번 논문은 자율 비행체(UAV)가 지원하는 연합 학습(Federated Learning)에서 통합 감지 및 통신(Integrated Sensing and Communication)과 관련된 새로운 지연(latency) 최적화 문제를 다룹니다. 이 환경에서 분산된 UAV는 감지된 데이터를 사용하여 모델 훈련에 참여하고, FL 집계기(base station)와 협력하여 글로벌 모델을 구축합니다.

- **Technical Details**: 논문에서는 UAV 간의 자원 배분과 UAV 및 기본 스테이션(BS)의 자원 배분을 함께 최적화하여 FL 시스템 지연을 최소화하는 문제를 공식화하였습니다. 이 최적화 문제의 비볼록성(non-convexity) 때문에 해결이 어려운 이 문제를 해결하기 위해 블록 좌표 하강법(block coordinate descent)과 연속 볼록 근사(successive convex approximation) 기법을 활용한 효율적인 반복 알고리즘을 개발하였습니다.

- **Performance Highlights**: 시뮬레이션 결과, 제안한 공동 최적화 전략이 벤치마크(baseline) 방식과 비교하여 최대 68.54%까지 시스템 지연을 절감하는 효과를 증명하였습니다.



### Automated Feedback in Math Education: A Comparative Analysis of LLMs for Open-Ended Responses (https://arxiv.org/abs/2411.08910)
Comments:
          12 pages including references, 4 figures, 9 tables

- **What's New**: 이 연구는 대형 언어 모델(Large Language Models, LLMs), 특히 Mistral 기반의 Llama 변형 모델을 활용하여 수학 교육에서 자동화된 피드백을 제공하는 동향을 탐구합니다. 구체적으로, Mistral 모델을 수학 문제에 맞게 세부 조정하여 학생의 개방형 응답을 평가하는 성능을 비교합니다.

- **Technical Details**: 이 연구에서는 LLM을 사용하여 학생의 수학 문제 응답을 평가하고, Llama, SBERT-Canberra, GPT-4 모델의 성능을 비교합니다. Mistral 모델은 교육 데이터를 활용해 세부 조정 되었으며, SBERT 모델도 유사한 접근법을 사용하였습니다. GPT-4는 제로샷 학습(zero-shot learning) 접근 방식을 통해 개방형 질문과 관련된 특정 루브릭을 기반으로 평가됩니다.

- **Performance Highlights**: 세 가지 모델(SBERT-Canberra, Mistral-Llama, GPT-4)의 성능을 교사 두 명의 평가를 통해 정량적 및 정성적으로 분석하였습니다. 연구 결과, LLM이 생성하는 피드백의 정확성과 관련성을 기반으로 한 교사 평가에서 가치있는 통찰력을 제공하며, 향후 자동화된 피드백 시스템의 발전 방향을 제시합니다.



### Assessing the Auditability of AI-integrating Systems: A Framework and Learning Analytics Case Study (https://arxiv.org/abs/2411.08906)
- **What's New**: 본 논문에서는 인공지능(AI)이 통합된 학습 분석 시스템의 감사 가능성을 평가하기 위한 프레임워크를 제안합니다. 이 프레임워크는 시스템의 유효성, 유용성 및 윤리에 대한 검증 가능한 주장, 다양한 유형의 증거, 그리고 감사인이 접근할 수 있는 증거의 세 가지 요소로 구성되어 있습니다. 실제로 이 프레임워크를 Moodle의 중퇴 예측 시스템에 적용하였고, 제한된 문서화와 부족한 모니터링 기능, 이용 가능한 테스트 데이터의 결여로 인해 Moodle의 감사 가능성이 제한적임을 발견했습니다.

- **Technical Details**: 제안된 감사 가능성 평가 프레임워크는 세 부분으로 나뉩니다: (1) 시스템의 유효성, 유용성 및 윤리에 대한 검증 가능한 주장, (2) 데이터, 모델 또는 시스템에 대한 다양한 유형의 증거(문서화, 원천 자료, 로그)로 주장을 뒷받침하거나 반박, (3) 감사인이 기술적인 수단(API, 모니터링 도구, 설명 가능한 AI 등)을 통해 접근할 수 있는 증거. 이를 통해 AI 기반 학습 분석 시스템의 감사 가능성을 평가하고, 감사 가능한 시스템의 설계를 개선하는 데 기여할 수 있습니다.

- **Performance Highlights**: Moodle의 감사 가능성을 평가해본 결과, 불완전한 문서, 부족한 모니터링 기능, 사용할 수 있는 테스트 데이터의 결여가 주요 제한 요인으로 작용함을 확인했습니다. 이 연구는 AI 기반 학습 분석 시스템의 감사 가능성을 개선하고, 이러한 시스템의 설계를 개선하는 데 중요한 통찰을 제공합니다.



### RNA-GPT: Multimodal Generative System for RNA Sequence Understanding (https://arxiv.org/abs/2411.08900)
Comments:
          Machine Learning for Structural Biology Workshop, NeurIPS 2024

- **What's New**: 이번 논문에서는 RNA 발견을 간소화하기 위해 RNA-GPT라는 멀티모달 RNA 챗 모델을 소개합니다. 이 모델은 방대한 RNA 문헌을 활용하여 RNA 연구를 지원합니다.

- **Technical Details**: RNA-GPT는 RNA 서열 인코더(RNA sequence encoders)와 선형 프로젝션 레이어(linear projection layers), 최신 대형 언어 모델(large language models, LLMs)을 통합하여 정밀한 표현 정렬(representation alignment)을 가능하게 합니다. 이 모델은 사용자 업로드 RNA 서열을 처리하고 간결하고 정확한 응답을 제공합니다.

- **Performance Highlights**: 실험 결과, RNA-GPT는 복잡한 RNA 쿼리를 효과적으로 해결함으로써 RNA 연구를 촉진시킵니다. 또한, 407,616개의 RNA 샘플을 포함하는 RNA-QA 데이터셋을 제공하여 모달리티 정렬(modality alignment) 및 instruction tuning을 통해 RNA 연구 도구의 가능성을 더욱 발전시킵니다.



### FinVision: A Multi-Agent Framework for Stock Market Prediction (https://arxiv.org/abs/2411.08899)
Comments:
          Accepted at ICAIF 2024

- **What's New**: 본 연구는 금융 거래를 위한 다중 모달 다중 에이전트 시스템을 도입하였으며, 각각 다양한 금융 데이터를 처리하고 해석하는 데 능숙한 LLM 기반 에이전트 팀을 활용하고 있습니다. 또한, 과거 거래 신호와 그 결과를 분석하는 반영 모듈을 통합하여 의사결정 능력을 향상시킵니다.

- **Technical Details**: 이 프레임워크는 Summarize Module, Technical Analyst Module, Prediction Module 및 Reflection Module의 네 가지 주요 구성 요소로 이루어져 있습니다. Summarize Module은 뉴스 데이터를 요약하고, Technical Analyst Agent는 캔들스틱 차트 분석을 수행하며, Prediction Module은 거래 행동을 예측하고, Reflection Module은 이전 거래의 성과를 분석합니다.

- **Performance Highlights**: 본 연구의 프레임워크는 Apple, Amazon 및 Microsoft를 대상으로 한 7개월간의 성과 평가에서 기존의 규칙 기반 및 RL 기반 모델을 초월하며, FinAgent 프레임워크의 기준에는 미치지 못하지만 다양한 데이터 소스의 통합을 통한 종합적인 접근 방식을 보여줍니다.



### Temporal Patterns of Multiple Long-Term Conditions in Welsh Individuals with Intellectual Disabilities: An Unsupervised Clustering Approach to Disease Trajectories (https://arxiv.org/abs/2411.08894)
- **What's New**: 이번 연구는 지적 장애(individuals with intellectual disabilities) 환자에서 다중 장기 조건(multiple long-term conditions, MLTC)의 동시 발생을 분석하기 위해 전자 건강 기록(electronic health records, EHRs)을 활용하여 진행된 최초의 대규모 연구입니다.

- **Technical Details**: 이 연구는 2000년부터 2021년까지 웨일스의 지적 장애 환자 13,069名의 EHR을 기반으로 하고 있으며, 40개의 장기 조건(long-term conditions, LTCs)을 분석하였습니다. 무감독 기계 학습(unsupervised machine learning) 기법을 사용하여 질병의 연대적 경향성과 클러스터링을 평가하였습니다.

- **Performance Highlights**: 연구 결과, 45세 이하 남성 그룹은 신경학적 조건(neurological conditions)이 32.4%로 다수를 차지하는 단일 클러스터를 형성했습니다. 반면, 45세 이상의 남성은 순환기 조건(circulatory conditions), 소화기 조건(digestive conditions), 근골격계 문제(musculoskeletal issues)에 따른 세 개의 클러스터를 형성했습니다. 이 클러스터는 지적 장애 환자의 질병 진행 패턴을 이해하는 데 중요한 통찰을 제공합니다.



### Auto-assessment of assessment: A conceptual framework towards fulfilling the policy gaps in academic assessment practices (https://arxiv.org/abs/2411.08892)
Comments:
          20 Pages, 5 Figures, submitted for journal peer-review

- **What's New**: 이 연구는 Generative Artificial Intelligence (GAI) 기술이 교육에 미치는 영향과 적절한 사용 정책의 필요성을 강조하고 있습니다. 또한, 학문적 평가를 위한 새로운 AI 프레임워크를 제안합니다.

- **Technical Details**: 조사는 영국, UAE, 이라크의 117명의 학자들을 대상으로 이루어졌으며, 응답자의 71.79%가 AI가 자율 평가에 유용하다고 응답했습니다. 이 연구는 교육 표준을 유지하면서 기술 발전의 이점을 활용하기 위한 새로운 정책 개발의 필요성을 제기합니다.

- **Performance Highlights**: 대부분의 경험 있는 학자들은 AI에 대한 긍정적인 의견을 가지고 있으며, 완전 금지보다는 유용한 교육 도구로 보는 경향이 있음을 확인했습니다. 또한, 현대 AI 기반 도구(예: ChatGPT)에 대한 인식 부족이 발견되었습니다.



### Calibrated Decision-Making through LLM-Assisted Retrieva (https://arxiv.org/abs/2411.08891)
- **What's New**: 이번 논문에서는 Calibrated Retrieval-Augmented Generation (CalibRAG)이라는 새로운 검색 방법을 제안하여, LLM이 제공하는 정보의 신뢰성을 높이고 잘 조율된 결정-making을 지원합니다.

- **Technical Details**: CalibRAG는 외부 문서에서 정보를 검색하는 전통적인 Retrieval Augmented Generation (RAG) 방법보다 한 단계 발전하여, 사용자의 결정이 잘 조율되도록 보장합니다. 또한 예측 기능(forecasting function)을 활용하여, RAG에서 제공하는 정보에 대한 신뢰도를 적절하게 반영합니다.

- **Performance Highlights**: 실험 결과, CalibRAG는 다양한 데이터셋에서 기존의 불확실성 조정 기준선과 비교하여 정확도와 조율(performance calibration) 성능 향상이 입증되었습니다.



### Spotlight Session on Autonomous Weapons Systems at ICRC 34th International Conferenc (https://arxiv.org/abs/2411.08890)
Comments:
          8 pages, 2415 words, 1 figure. Panelist notes for the Spotlight Session on Autonomous Weapons Systems at the ICRC 34th International Conference 28-31 Oct 2024

- **What's New**: 이번 논문은 자율 무기 시스템(Autonomous Weapons Systems, AWS)의 발전이 인간의 의사결정 방식과 그 결정의 결과, 그리고 결정에 대한 책임소재에 미치는 영향을 다루고 있습니다. 우리는 AWS의 개발, 사용 및 정당화에 대한 논의에서 경계심을 갖고 인간 중심의 접근 방식을 유지해야 합니다.

- **Technical Details**: 논문에서는 국제 인도법(International Humanitarian Law, IHL) 준수를 향상시키기 위한 여러 가지 방법을 제시하고 있습니다. 이를 위해 무기 의사 결정자를 IHL 교육, 무기 검토에서의 모범 사례 개발, 인간 중심의 시험 및 평가 방법 개발, 디지털 인프라 투자, 민간인 피해에 대한 연구 등이 필요하다고 강조합니다.

- **Performance Highlights**: 정부는 무기 시스템의 요구 사항 설정 책임이 있으며, 윤리성과 치명성을 모두 고려해야 합니다. 유엔(UN)은 IHL 준수와 인권, 인간 중심의 무기 시스템 사용 및 군사적 의사결정 모니터링 기제 개선을 옹호할 수 있습니다.



### Multilingual Standalone Trustworthy Voice-Based Social Network for Disaster Situations (https://arxiv.org/abs/2411.08889)
Comments:
          Accepted for publication in IEEE UEMCON 2024, to appear in December 2024. 7 pages, 3 figures

- **What's New**: 이 논문은 재난 상황에서의 다국어 음성 기반 소셜 네트워크를 설계하여 언어 장벽 문제를 해결하는 새로운 접근 방식을 제시하고 있습니다. 이 시스템은 AI와 블록체인 기술을 통합하여 오프라인에서 다국어 음성 커뮤니케이션을 가능하게 합니다.

- **Technical Details**: 제안된 애플리케이션은 실시간 음성 메시지 번역, 블록체인 기반의 안전한 저장장치, 그리고 다양한 기기에서의 일관된 사용자 경험을 제공합니다. 이 애플리케이션은 모든 AI 프로세싱과 블록체인 관리가 로컬에서 이루어져 외부 서버에 의존하지 않으며, 보안성과 신뢰성을 보장합니다.

- **Performance Highlights**: 이 시스템은 음성 인식 및 번역의 높은 정확도, 낮은 대기 시간, 그리고 사용자 만족도를 통해 효과성을 입증했습니다. 이를 통해 재난 상황에서의 언어 격차를 줄이고 응급 대응을 더욱 효율적으로 지원할 수 있습니다.



### Exploring Capabilities of Time Series Foundation Models in Building Analytics (https://arxiv.org/abs/2411.08888)
Comments:
          7 pages, 1 figures, and 4 tables

- **What's New**: 본 연구는 사물인터넷(IoT) 데이터와 건물 에너지 분석을 위한 시간 시계열 모델의 성능을 평가하는 종합적인 비교 연구를 수행했습니다. 특히, 다양한 IoT 데이터셋을 활용하여 빌딩 관리와 에너지 효율성 향상을 목표로 합니다.

- **Technical Details**: 연구는 BLDG59와 BTS 데이터셋을 사용하며, 시계열 모델은 머신 러닝 기법인 RandomForest와 XGBoost, 딥 러닝 모델 DLinear, PatchTST, Informer, iTransformer, 그리고 시간 시계열 기초 모델 LLaTA, One-Fits-All, TimeLLM으로 구성됩니다. 이 연구는 주어진 알맞은 지표(MAE, MSE, SMAPE)를 통해 모델의 효율성을 평가합니다.

- **Performance Highlights**: One-Fits-All 모델은 모든 기차/테스트 설정에서 가장 낮은 SMAPE 점수를 기록하였으며, 다수의 메트릭에서 1위를 차지하였습니다. LLaTA 모델은 MSE와 SMAPE에서 상대적으로 저조한 성과를 보였으며, 눈에 띄게 높은 오류율을 기록했습니다.



### Deep Learning-Based CKM Construction with Image Super-Resolution (https://arxiv.org/abs/2411.08887)
- **What's New**: 본 논문에서는 Channel Knowledge Map (CKM)의 구성을 위한 새로운 접근 방식을 제안합니다. 이 방법은 Sparse Data 기반으로 좋은 통신 및 감지 성능을 실현합니다.

- **Technical Details**: CKM은 환경 인식 향상 및 무선 시스템의 성능 개선을 위한 기술로, 이번 연구에서는 Super-Resolution (SR) 네트워크 SRResNet을 활용하여 CKM을 구축하는 심층 학습 기반의 방법론을 제시합니다. 주요 데이터셋으로 CKMImageNet을 사용하여 경로 손실 맵과 채널 각도 맵(CAM)을 구축할 수 있습니다.

- **Performance Highlights**: 제안하는 방법은 기존의 보간(bicubic, nearest neighbour) 및 SRGAN 방법을 초월하며, 경로 손실에서 평균 제곱근 오차(RMSE) 1.1 dB 달성을 위해 단 1/16의 측정 위치만 필요하다는 사실이 수치적으로 입증되었습니다.



### Enhancing Lie Detection Accuracy: A Comparative Study of Classic ML, CNN, and GCN Models using Audio-Visual Features (https://arxiv.org/abs/2411.08885)
Comments:
          11 pages, 18 figures

- **What's New**: 이번 연구는 오디오 입력, 시각적 얼굴 미세 표현(micro-expressions), 그리고 수작업으로 기록된 제스처 주석을 사용하는 독특한 다중 모달 트랜스포머 아키텍처를 개발하여 기존의 방법을 개선합니다. 이를 통해 비침습적인 거짓말 탐지 모델에 더 가까워집니다.

- **Technical Details**: 비주얼 및 오디오 특징은 각각 Vision Transformer(ViT) 및 OpenSmile 모델을 사용하여 추출되었으며, 참가자의 미세 표현 및 제스처 형태 기록과 함께 연결(concatenate) 되었습니다. 최종적으로 CNN Conv1D 다중 모달 모델이 평균 정확도 95.4%를 달성하였습니다. 그러나 더 높은 품질의 데이터셋과 더욱 일반화된 모델을 위해 추가 연구가 필요합니다.

- **Performance Highlights**: 이 연구는 거짓말 탐지에서 기존 방법들과 최근 AI 모델들에 비해 제안된 AI 모델의 효율성을 평가하고, 예측에서 가장 큰 비중을 가지는 특징(feature)을 분석합니다. 특히, 다중 모달 데이터 처리의 잠재력을 보여주며, 법 집행, 법정 증언에 대한 신뢰성 평가, 금융 서비스에서의 사기 탐지와 같은 여러 분야에 혁신을 가져올 수 있을 것으로 기대됩니다.



### Quantifying Risk Propensities of Large Language Models: Ethical Focus and Bias Detection through Role-Play (https://arxiv.org/abs/2411.08884)
- **What's New**: 이번 연구는 Large Language Models (LLMs)의 안전성, 윤리 및 편향에 대한 우려가 커지는 가운데, LLM의 윤리적 리스크 태도를 평가하기 위해 Domain-Specific Risk-Taking (DOSPERT) 스케일을 혁신적으로 적용하고, 새로운 Ethical Decision-Making Risk Attitude Scale (EDRAS)을 제안합니다.

- **Technical Details**: 이 연구에서는 LLM의 위험 성향을 평가하기 위해 DOSPERT 스케일과 역할 놀이를 통합한 혁신적인 접근 방식을 사용하였습니다. 특히, 여러 주요 LLM의 윤리적 영역에서 위험 성향을 평가하고, 시스템적 편향을 정량적으로 분석했습니다.

- **Performance Highlights**: 연구 결과, 각 LLM은 차별화된 위험 성향을 보였으며, LLM의 윤리적 리스크 성향 평가에 EDRAS가 효과적으로 적용될 수 있다는 것을 보여주었습니다. 또한, 역할 가설을 이용한 위험 태도 탐색을 통해 다양한 사회 집단에 대한 시스템적 편향을 탐지할 가능성을 제시했습니다.



### KisanQRS: A Deep Learning-based Automated Query-Response System for Agricultural Decision-Making (https://arxiv.org/abs/2411.08883)
- **What's New**: 이 논문에서는 Kisan Query Response System (KisanQRS)라는 농업 분야를 위한 딥러닝 기반의 견고한 쿼리-응답 프레임워크를 소개합니다. 농부들이 신속하게 정보와 지침을 받을 수 있도록 도와줍니다.

- **Technical Details**: KisanQRS는 농부의 쿼리에 대한 의미적 및 어휘적 유사성을 통합하고 빠른 임계값 기반 군집화(clustering) 방법을 사용합니다. 군집화 알고리즘은 쿼리를 반복적으로 돌아보는 선형 검색(linear search) 기법을 기반으로 하며, LSTM(Long Short-Term Memory) 모델이 쿼리 매핑에 최적의 방법으로 발견되었습니다. 제안된 답변 검색 방법은 작물에 대한 후보 답변을 군집화하고, 군집 내의 답변 수에 따라 답변 군집을 순위매기며 각 군집의 리더를 선택합니다.

- **Performance Highlights**: KisanQRS는 인도 정부가 운영하는 Kisan Call Centre (KCC)의 3,400만 통화를 포함한 데이터 셋을 기반으로 하며, 30만 샘플에 대한 쿼리 매핑 모듈의 성능 평가에서 주(state)별로 96.58%의 높은 F1-score를 달성했습니다. 답변 검색 모듈은 10,000 샘플에서 평가되며 96.20%의 경쟁력 있는 NDCG 점수를 기록했습니다. 이를 통해, KisanQRS는 농부들이 그들의 농업 관행에 대해 정보에 기반한 결정을 내릴 수 있도록 신속하고 관련성 있는 응답을 제공합니다.



### A Novel Multimodal System to Predict Agitation in People with Dementia Within Clinical Settings: A Proof of Concep (https://arxiv.org/abs/2411.08882)
- **What's New**: 이번 연구는 중증 치매 환자에서의 동요 및 공격성(Agitation and Aggression, AA) 사건을 실시간으로 예측하기 위한 다중 모드 접근법을 도입한 5년간의 연구 결과를 발표합니다. EmbracePlus 손목 밴드와 비디오 감지 시스템을 통합하여 AA의 선행 패턴을 식별하고 관련 사건을 예측할 수 있는 새로운 시스템을 개발하였습니다.

- **Technical Details**: 이 시스템은 EmbracePlus 손목 밴드에서 수집된 생체 데이터와 CCTV 카메라에서 얻은 비디오 데이터를 활용하여 AA를 예측합니다. 심리적 신호를 포함한 데이터는 실시간으로 깊은 학습 딥러닝 모델을 통해 분석되며, 전통적인 방법보다 개인 맞춤형 조기 개입을 가능하게 합니다. 연구 결과, 시스템은 AA 사건 발생 최소 6분 전에 동요 패턴을 탐지할 수 있었습니다.

- **Performance Highlights**: 파일럿 연구에서 3명의 참가자를 대상으로 손목밴드와 비디오 시스템을 동시에 활용하여 AA 사건을 높은 정확도로 탐지하였습니다. 리커시브 뉴럴 네트워크(Recall Neural Network, RNN) 기반의 LSTM과 GRU 모델을 통해 실시간 처리가 최적화되어 AA 사건을 신속하게 분석할 수 있었습니다. 전반적으로 이 시스템의 초기 데이터 분석 결과는 AA 사건을 효과적으로 예측할 수 있는 능력을 보였습니다.



### Can We Trust AI Agents? An Experimental Study Towards Trustworthy LLM-Based Multi-Agent Systems for AI Ethics (https://arxiv.org/abs/2411.08881)
- **What's New**: 본 연구는 LLM(Large Language Model)을 활용하여 윤리적 AI 시스템을 개발할 때 신뢰성을 높이는 기술을 탐구합니다. 이를 통해 윤리적 AI 시스템의 구현에 필요한 실용적 가이드를 제시합니다.

- **Technical Details**: 이 연구에서는 Design Science Research(DSR) 방법론을 사용하여 여러 신뢰성 향상 기술을 식별합니다: 다중 에이전트, 명확한 역할, 구조화된 커뮤니케이션 및 논의의 반복적 진행. LLM-BMAS 프로토타입을 설계하고 AI Incident Database에서 실제 윤리적 AI 문제에 대해 구조화된 논의를 진행합니다. 성능 평가는 주제 분석, 계층적 군집화, 제거 연구(ablation study), 소스 코드 실행을 통해 진행됩니다.

- **Performance Highlights**: 프로토타입은 한 번의 실행으로 약 2,000라인의 코드를 생성하며, 이는 제거 연구에서의 80라인과 비교됩니다. LLM-BMAS는 편향 탐지, 투명성, 책임성, 사용자 동의, GDPR 준수, 공정성 평가, EU AI 법 준수와 같은 복잡한 윤리적 AI 문제들을 다루는 각각의 코드와 문서를 생성하는 능력을 보여줍니다. 하지만 소스 코드 통합 및 의존성 관리와 관련해 실용적인 문제들이 있어 실제 운영자들이 시스템을 원활하게 채택하는 데 제약이 될 수 있습니다.



### NLIP_Lab-IITH Low-Resource MT System for WMT24 Indic MT Shared Task (https://arxiv.org/abs/2410.03215)
Comments:
          WMT2024 INDICMT Shared Task

- **What's New**: 이번 연구에서는 WMT 2024 저자원 인도어 번역의 공유 작업을 위한 시스템을 소개합니다. 이 작업에서는 영어(eng)와 아삼어(as), 카시어(kha), 미조어(lus), 마니푸르어(mni) 등의 언어 쌍을 고려합니다. 저자원 언어 번역의 문제를 해결하기 위해 모델의 미세 조정(fine-tuning)과 다국어 훈련(multilingual training)을 탐구하였습니다.

- **Technical Details**: 연구의 핵심 시스템은 사전 학습(pre-trained) 모델에 대한 언어별 미세 조정으로 구성됩니다. 22개의 인도 언어에 대한 embedding을 정렬하는 목표로 사전 학습된 IndicRASP 모델을 기반으로 하였으며, 이 모델은 두 개의 사전 학습 모델인 IndicRASP와 IndicRASP Seed에 대한 미세 조정을 통해 다양한 실험을 수행했습니다. 실험 결과, eng→as, eng→kha, eng→lus, eng→mni 쌍에서 각각 chrF2 점수 50.6, 42.3, 54.9, 66.3을 달성하였습니다.

- **Performance Highlights**: BLEU 점수 또한 공개 테스트 세트에서 eng→as 20.1, eng→kha 19.1, eng→lus 30.0, eng→mni 35.6을 기록하며, 언어별 미세 조정이 번역 품질 향상에 기여하였음을 보여주었습니다. 연구 결과에 따르면, 사전 학습된 alignment-augmented 모델을 활용하는 것이 저자원 환경에서도 높은 번역 품질을 개선할 수 있는 잠재력을 지닌 것으로 나타났습니다.



### Learning Multi-Agent Loco-Manipulation for Long-Horizon Quadrupedal Pushing (https://arxiv.org/abs/2411.07104)
- **What's New**: 이 논문에서는 다수의 4족 로봇이 장애물을 인식하며, 장기적인 푸시 작업을 수행할 수 있는 새로운 계층적 다중 에이전트 강화 학습(MARL) 프레임워크를 제안합니다. 이 프레임워크는 세 가지 제어 수준으로 구성되어 장애물을 피하면서 물체를 효과적으로 밀 수 있도록 돕습니다.

- **Technical Details**: 제안된 방법은 높은 수준의 컨트롤러가 RRT(Rapidly-exploring Random Tree) 계획자와 중앙 집중식 적응 정책을 통합하여 서브 목표를 생성하고, 중간 수준의 컨트롤러가 분산 목표 조건 정책을 사용하여 로봇이 해당 서브 목표로 안내하도록 합니다. 저수준 컨트롤러는 사전 훈련된 보행 정책을 통해 이동 명령을 실행합니다.

- **Performance Highlights**: 시뮬레이션에서 제안된 방법은 최고의 기준선 방법보다 성공률이 36.0% 높고, 완료 시간은 24.5% 단축되는 성능을 보였습니다. 이 방법은 실제 로봇인 Go1에서 Push-Cuboid 및 Push-T와 같은 장애물 인식 및 장기 푸시 작업을 성공적으로 수행할 수 있도록 합니다.



New uploads on arXiv(cs.LG)

### On the Surprising Effectiveness of Attention Transfer for Vision Transformers (https://arxiv.org/abs/2411.09702)
Comments:
          NeurIPS 2024. Code: this https URL

- **What's New**: 이번 연구는 전통적인 ViT(Vision Transformers) 프리트레이닝(pre-training)이 다운스트림 작업에서의 성능 향상에 얼마나 기여하는지를 탐구합니다. 연구 결과, 프리트레이닝 중 학습된 특징(feature)과 표현(representation)은 반드시 필수적이지 않다는 것을 발견했습니다. 놀랍게도, 단순히 프리트레이닝에서 얻은 attention 패턴만으로도 높은 품질의 특징을 새롭게 학습하고, 비교 가능한 성능을 달성할 수 있음을 보여줍니다.

- **Technical Details**: 연구에서는 'attention transfer'라는 간단한 방법을 제안합니다. 이를 통해 프리트레이닝된 teacher ViT로부터 attention 패턴만을 학생 모델에게 전달하여, 학생은 이러한 attention 패턴에 의해 특징을 독립적으로 학습합니다. 두 가지 방식인 Attention Copy와 Attention Distillation이 제안되며, 후자는 teacher의 attention 패턴을 학생에게 증류하여 전달합니다. 이는 프리트레이닝된 attention 맵을 활용한 학습 방법으로, 그 효과를 구분할 수 있습니다.

- **Performance Highlights**: 이 방법들은 놀라운 효과를 보였으며, Attention Copy는 대부분의 성능 차이를 해소했고, Attention Distillation은 ImageNet-1K 분류에서 훌륭한 정확도를 달성했습니다. attention transfer를 통해 학생이 스스로 특징을 학습하게 되는 점에서 기존의 파인튜닝(fine-tuning) 방식과 차별화된 성과를 보여주며, 이러한 접근법이 프리트레이닝된 ViT 활용에 있어 효과적인 대안이 될 수 있음을 제안합니다.



### NeuralDEM - Real-time Simulation of Industrial Particulate Flows (https://arxiv.org/abs/2411.09678)
Comments:
          Project page: this https URL

- **What's New**: 이번 연구에서는 NeuralDEM이라는 새로운 접근 방식을 통해 느린 수치 DEM(Discrete Element Method) 방정식을 빠르고 적응 가능한 딥러닝 대체 모델로 바꾸었습니다.

- **Technical Details**: NeuralDEM은 DEM의 라그랑지안(Lagrangian) 이산화(discretization)를 기본 연속 필드로 간주하며, 동시에 고맥락(macroscopic) 행동을 추가 보조 필드로 모델링합니다. 또한, 여러 분기(multi-branch) 신경 연산자(neural operators)를 도입하여 산업 규모의 시나리오를 실시간으로 모델링할 수 있도록 하였습니다.

- **Performance Highlights**: NeuralDEM은 160k CFD 셀과 500k DEM 입자가 결합된 유체화된 침대 리액터를 정확히 모델링하였으며, 28초 동안의 궤적을 추적할 수 있습니다. 이로 인해 고급 엔지니어링 및 더 빠른 공정 주기에 많은 새로운 가능성이 열릴 것입니다.



### On the Limits of Language Generation: Trade-Offs Between Hallucination and Mode Collaps (https://arxiv.org/abs/2411.09642)
Comments:
          Abstract shortened to fit arXiv limit

- **What's New**: 본 연구는 언어 모델의 생성에서 '일관성(consistency)'과 '폭(breadth)'을 동시에 충족할 수 있는지에 대한 문제를 다루고 있습니다. 이 문제에 대한 기존 연구와 반대의 결과를 제시하며, 언어 모델의 난제 중 하나인 생성의 폭과 일관성 간의 긴장 관계를 수학적으로 설명합니다.

- **Technical Details**: 알고리즘은 무한 집합으로부터 랜덤 샘플을 받아, 이를 바탕으로 모든 보이지 않는 문자열을 생성하는 것을 목표로 합니다. 이 과정에서 잘 정의된 언어 K 내에서 유의미한 데이터 분포를 학습해야 하며, 이는 기존 GAN(Generative Adversarial Networks)에서의 mode collapse 문제와 관련이 있습니다. 더불어, 샘플 수가 증가할 때 출력이 K의 모든 보이지 않는 문자열에 수렴하는지를 분석합니다.

- **Performance Highlights**: 연구 결과에 따르면, 많은 언어 모델에서는 '일관성'과 '폭'을 동시에 달성하는 것이 불가능하다는 것이 밝혀졌습니다. 그러나 긍정적인 예제와 부정적인 예제(정답 외의 문자열)가 함께 제공될 때 카운팅 가능한 모든 언어 집합에 대해 일관성 있는 폭 있는 생성을 달성할 수 있는 가능성을 제시합니다. 이는 히스토그램 피드백이 환각(hallucination)을 줄이는 데 중요한 역할을 할 수 있음을 시사합니다.



### MCCE: Missingness-aware Causal Concept Explainer (https://arxiv.org/abs/2411.09639)
- **What's New**: 본 논문에서는 관측되지 않은 개념이 관측된 개념의 인과 효과 추정을 편향시킬 수 있음을 이론적으로 입증하고, 이를 해결하기 위한 'Missingness-aware Causal Concept Explainer (MCCE)'라는 새로운 프레임워크를 제안합니다. MCCE는 관측되지 않은 개념의 영향을 파악하고, 선형 예측기를 통해 블랙박스 모델 출력과 개념 간의 관계를 모델링합니다.

- **Technical Details**: MCCE는 관측된 개념에 대해 관측되지 않은 개념의 영향을 고려하여 인과 개념 효과를 추정합니다. 이 프레임워크는 관측된 개념과 직교하는 의사 개념을 구성하여 미비한 개념으로 인한 잔여 편향을 보정합니다. 또한, MCCE는 모델의 출력을 설명하기 위해 개별 샘플에 대한 로컬 설명뿐만 아니라 모델이 의사 결정을 내리는 데 사용되는 일반적인 규칙을 설명하는 글로벌 설명도 제공합니다.

- **Performance Highlights**: 실제 데이터셋을 사용한 검증 결과, MCCE는 기존의 최신 설명 방법들에 비해 인과 개념 효과 추정에서 유망한 성능을 보였으며, 개별 개념 인과 효과 오류 (ICaCE-Error)를 효과적으로 측정할 수 있음을 입증했습니다. 또한 MCCE는 해석 가능한 예측 모델로 기능할 수 있습니다.



### LLaMA-Mesh: Unifying 3D Mesh Generation with Language Models (https://arxiv.org/abs/2411.09595)
Comments:
          See the project website at this https URL

- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)을 활용하여 텍스트 기반 3D 메쉬를 생성하는 새로운 방법인 LLaMA-Mesh를 제안합니다. 이 접근법은 LLMs의 기존 공간적 지식을 활용하고, 대화형 3D 생성 및 메쉬 이해를 가능하게 합니다.

- **Technical Details**: LLaMA-Mesh는 3D 메쉬의 정점 좌표와 면 정의를 일반 텍스트로 표현하여 LLM과의 통합을 용이하게 합니다. 이를 위해, OBJ 파일 형식을 사용하며, 기존의 토크나이저를 확장하지 않고 새로운 데이터 형식을 처리합니다. 또한, 감독형 미세 조정(SFT) 데이터셋을 만들어 LLM이 3D 메쉬를 이해하고 생성할 수 있도록 학습시킵니다.

- **Performance Highlights**: LLaMA-Mesh는 사전 훈련된 LLM이 3D 메쉬를 텍스트 프롬프트로부터 생성하고, 텍스트와 3D 메쉬를 혼합하여 출력할 수 있는 능력을 보여줍니다. 제안된 방법은 모델을 처음부터 훈련한 모델과 동등한 품질의 메쉬 생성을 달성하면서도 강력한 텍스트 생성 성능을 유지합니다.



### Expert Study on Interpretable Machine Learning Models with Missing Data (https://arxiv.org/abs/2411.09591)
Comments:
          Findings paper presented at Machine Learning for Health (ML4H) symposium 2024, December 15-16, 2024, Vancouver, Canada, 13 pages

- **What's New**: 이 연구에서는 임상 결정을 위한 해석 가능한 머신 러닝(Interpretable Machine Learning, IML) 모델이 결측값(missing values) 문제에 어떻게 대처하는지에 대한 71명의 임상의에서의 조사 결과를 다룹니다. 연구 결과에 따르면, 전통적인 결측값 처리 기법이 임상의의 직관과 불일치할 수 있으며, native하게 결측값을 처리하는 모델이 선호되는 것으로 나타났습니다.

- **Technical Details**: 이 연구는 29개의 외상 센터에서 71명의 임상을 대상으로 진행된 온라인 설문조사입니다. 설문조사는 임상의의 IML 모델에 대한 이해와 결측값 처리 효율성을 평가하기 위해 개발되었습니다. 데이터 분석에는 Descriptive Statistics와 Factor Analysis for Mixed Data(FAMD)가 사용되었습니다. 다양한 결측값 처리 방법의 영향을 포함한 예측 모델에 대한 임상의의 반응을 조사하였습니다.

- **Performance Highlights**: 임상의의 36.4%가 설문에 완전하게 응답하였고, 이들은 결측값을 다루는 데 있어 임상의적 직관을 기반으로 다양한 전략을 사용하고 있는 것으로 보입니다. 설문 조사 결과, 평균값이나 제로 임퓨테이션(zero imputation)과 같은 전통적인 기법의 경우, 임상의가 이를 의사 결정에 적용할 수 없는 것으로 나타났습니다. 모델에 대한 해석 가능성의 중요성이 강조되었으며, MICE와 같은 복잡한 방식은 해석에 대한 우려로 인해 낮은 선호도를 보였습니다.



### Communication Compression for Tensor Parallel LLM Inferenc (https://arxiv.org/abs/2411.09510)
- **What's New**: 대규모 언어 모델(LLMs)의 속도 향상을 위해, 본 연구에서는 Tensor Parallel 전략을 이용한 다수의 하드웨어 가속기에서의 모델 배치를 다루며, 활발한 협업을 통한 지연 시간을 줄이는 방법을 제안합니다.

- **Technical Details**: 최적화된 inter-accelerator communication을 위해 활동량을 3.5 - 4.5배 압축하는 세밀한 양자화 기법을 활용했습니다. 이 방법은 타임 투 퍼스트 토큰(TTFT)을 최대 2배 줄이는 결과를 가져오면서도 모델 성능 저하는 거의 없는 수준입니다.

- **Performance Highlights**: 기존 하드웨어 세팅에서 실행할 때, 느린 inter-accelerator 대역폭 환경에서의 TTFT가 3.5 - 4.5배 향상될 수 있음을 발견했습니다.



### Golden Noise for Diffusion Models: A Learning Framework (https://arxiv.org/abs/2411.09502)
- **What's New**: 이번 논문에서는 텍스트 프롬프트와 무작위 가우시안 노이스를 이용하여 개인화된 이미지를 생성하기 위한 새로운 개념인 '노이즈 프롬프트(noise prompt)'를 제안합니다. 이 개념은 무작위 노이즈를 텍스트 프롬프트에서 유도된 작은 변화를 추가하여 '골든 노이즈(golden noise)'로 변환하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 우리는 노이즈 프롬프트 학습 프레임워크(noise prompt learning framework)를 수립하여 텍스트 프롬프트와 연관된 '프롬프트된(golden) 노이즈'를 체계적으로 학습합니다. 또한, 10만 쌍의 무작위 노이즈와 골든 노이즈가 포함된 대규모 노이즈 프롬프트 데이터셋(noise prompt dataset, NPD)을 수집하였습니다. 이를 통해 훈련된 NPNet은 랜덤 노이즈를 골든 노이즈로 직접 변환할 수 있는 소형 네트워크입니다.

- **Performance Highlights**: NPNet을 사용하여 SDXL, DreamShaper-xl-v2-turbo 및 Hunyuan-DiT와 같은 여러 주류 확산 모델에서 실험한 결과, 생성된 이미지의 질과 심미성이 크게 향상되었습니다. NPNet은 표준 파이프라인에 비해 추가적인 3%의 추론 시간만 소모하며, 메모리 요구량도 약 3%로 낮추었으며, 이는 실세계 적용 가능성을 높입니다.



### Developement of Reinforcement Learning based Optimisation Method for Side-Sill Design (https://arxiv.org/abs/2411.09499)
- **What's New**: 이 논문은 차량의 내충격성(crashworthiness) 디자인 최적화를 위한 머신러닝 기반의 새로운 방법론을 제안합니다. 특히 다중세포 사이드 실(multi-cell side sill)의 설계를 통해 내충격성 성능을 향상시키는 데 중점을 두고 있습니다.

- **Technical Details**: 기존의 최적화 과정에서는 비선형 동적 해석을 위한 Finite Element (FE) 시뮬레이션이 필수적입니다. 이 시뮬레이션은 계산 자원이 많이 소모되므로, 서브리메서리(surrogate modeling)와 머신러닝(ML) 기법을 활용하여 효율성을 높입니다. 특히, 강화 학습(reinforcement learning, RL)을 통해 최적 설계 파라미터를 탐색하며, FE 솔버와 결합하여 결과를 개선합니다.

- **Performance Highlights**: 제안된 방법을 통해 여러 설계 변수(예: 벽 두께)를 최적화하여 에너지 흡수량을 최대화하고, 동시에 사이드 실의 전체 질량을 최소화할 수 있습니다. 멀티 오브젝트 최적화(multi-objective optimization) 문제를 해결함으로써, 내충격성 성능을 향상시키는 동시에 경량화도 달성하게 됩니다.



### What makes a good BIM design: quantitative linking between design behavior and quality (https://arxiv.org/abs/2411.09481)
- **What's New**: 이 연구는 Architecture Engineering & Construction (AEC) 산업에서의 디자인 행동이 디자인 품질에 미치는 영향을 처음으로 정량적으로 설명하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 이 연구에서는 Building Information Modeling (BIM)을 기반으로 하여 디자인 행동과 디자인 품질 간의 관계를 규명합니다. 실시간 데이터 수집과 로그 마이닝(log mining)을 통해 디자인 행동의 원시 데이터를 수집하고, 특성 공학(feature engineering) 및 다양한 기계 학습(machine learning) 모델을 활용하여 정량적 모델링 및 해석을 수행합니다.

- **Performance Highlights**: Extremely Random Trees를 사용한 최고의 성능 모델이 테스트 세트에서 R2 값 0.88을 달성하였습니다. 디자이너의 기술 수준과 디자인 의도의 변화와 관련된 행동 특성이 디자인 품질에 중요한 영향을 미친다는 사실이 확인되었습니다.



### ResidualDroppath: Enhancing Feature Reuse over Residual Connections (https://arxiv.org/abs/2411.09475)
- **What's New**: 본 논문에서는 기본적인 residual connection의 한계를 분석하고, 이를 보완하기 위한 ResidualDroppath 알고리즘을 제안합니다. 이는 딥러닝 네트워크에서 특징 재사용(feature reuse)을 더 효과적으로 학습할 수 있도록 돕는 새로운 훈련 방법론입니다.

- **Technical Details**: ResidualDroppath 알고리즘은 두 가지 유형의 반복(iteration)을 포함합니다. 첫 번째 반복에서는 droppath를 사용하여 랜덤하게 몇 개의 레이어를 드롭하여 특징 재사용을 강제합니다. 두 번째 반복은 드롭된 모델 부분을 학습하는 동안 남은 부분을 고정(freeze)하여 특징 재사용을 장려하도록 합니다. 이를 통해 모델은 남은 부분을 참고하여 더욱 활용적으로 학습할 수 있습니다.

- **Performance Highlights**: 이 알고리즘은 ResNet50 및 ResNet50d 모델에 적용되어 CIFAR10과 MNIST 데이터셋에서 Top-1 및 Top-5 정확도에서 상당한 성능 향상을 보였으며, ImageNet1K 데이터셋에서도 ResNet50d 모델의 성능 개선이 관찰되었습니다.



### Harnessing Machine Learning for Single-Shot Measurement of Free Electron Laser Pulse Power (https://arxiv.org/abs/2411.09468)
Comments:
          10 pages, 4 figures, Machine Learning and the Physical Sciences Workshop, NeurIPS 2024 this https URL

- **What's New**: 본 논문은 electron beam accelerators의 전통적인 진단기술의 한계를 극복하기 위해 기계학습(ML) 모델을 새롭게 개발하였습니다. 특히, free-electron lasers(FELs) 환경에서 lasing-off 상태의 electron bunch의 전력 프로파일을 예측하는 방법을 제시합니다.

- **Technical Details**: 연구에서는 multi-layer perceptron(MLP) 모델을 활용하여 2826개의 electron bunch에서 수집된 기계 매개변수를 입력으로 사용하고, lasing-off 상태에서의 전력 프로파일을 예측했습니다. 이 모델은 MSE(mean squared error) 손실 함수를 사용하며, 학습 및 검증 과정에서 Dropout을 통해 과적합(overfitting)을 방지했습니다.

- **Performance Highlights**: MLP 모델의 예측 결과는 실제 측정값과 매우 잘 일치하며, 평균 제곱 오차(mean squared error)가 가장 낮아 기존의 평균 기반 접근 방식 보다 우수한 성능을 보였습니다. 이러한 결과는 FEL의 photon pulse 프로파일 재구성 진단 능력을 크게 향상시킬 것으로 기대됩니다.



### Caravan MultiMet: Extending Caravan with Multiple Weather Nowcasts and Forecasts (https://arxiv.org/abs/2411.09459)
- **What's New**: Caravan 대규모 수문학 데이터셋에 기상 예측 데이터가 추가되었습니다. 이 확장에서 세 가지 강수 예측 제품(CPC, IMERG v07 Early, CHIRPS)과 세 가지 기상 예측 제품(ECMWF IFS HRES, GraphCast, CHIRPS-GEFS)을 포함하여 hydrological 모델의 평가와 벤치마킹을 강화합니다.

- **Technical Details**: 추가된 강수 및 기상 예측 데이터는 ERA5-Land 데이터에 통합되어, 연구자들이 다양한 기상 강제요소에 대해 모델 성능을 rigorously하게 평가할 수 있는 플랫폼을 제공합니다. 특히, 모든 데이터는 UTC+0 시간대에 유지되어 다른 기상 제품과의 비교를 용이하게 합니다.

- **Performance Highlights**: 이러한 데이터 확장은 Caravan을 세계 최초의 기상 예측 데이터를 포함하는 대규모 수문학 데이터셋으로 만들며, 이는 실시간 기상 예측 시나리오에서의 수문 모델 성능을 향상시키는 데 기여합니다.



### Inherently Interpretable and Uncertainty-Aware Models for Online Learning in Cyber-Security Problems (https://arxiv.org/abs/2411.09393)
- **What's New**: 이 논문에서는 사이버 보안 분야의 온라인 학습 문제에 대한 해석 가능하고 불확실성을 인식하는 머신 러닝 모델의 필요성을 다루고 있습니다. 특히, Additive Gaussian Processes (AGPs) 모델을 활용하여 예측 성능과 투명성을 모두 해결하려는 새로운 방법론을 제안합니다.

- **Technical Details**: 이 연구에서는 머신 러닝(Machine Learning) 모델의 해석 가능성 (interpretability)과 불확실성 인식 (uncertainty awareness)을 중요하게 강조합니다. 또한, AGPs를 사용하는 온라인 슈퍼바이즈드 학습 (online supervised learning) 파이프라인을 구성하고, 이 모델이 큰 데이터세트를 처리하는 데 있어 스케일링 문제 (scaling issue)를 해결하는 방법을 논의합니다.

- **Performance Highlights**: 실험 결과 AGPs가 사이버 보안 문제에서 신뢰할 수 있는 예측 결과를 제공함으로써 위협 탐지(validation), 오류 감소(reduce false positives) 및 정보에 기반한 의사 결정을 지원할 수 있음을 보여줍니다. AGPs 모델은 해석 가능성 높은 예측 및 분류 결과를 제공하는 특성으로 인해 사이버 보안 분야에서 유망한 모델로 자리매김할 수 있을 것으로 기대됩니다.



### A survey of probabilistic generative frameworks for molecular simulations (https://arxiv.org/abs/2411.09388)
- **What's New**: 이번 연구에서는 분자 과학에서의 생성적 인공지능 사용을 강조하며, 확률적 생성 모델을 사용한 성능 벤치마킹이 부족하다는 점을 지적합니다. 새롭게 제안된 여러 종류의 생성 모델을 설명하고, 이를 흐름 기반 모델(flow-based models)과 확산 모델(diffusion models)로 분류합니다.

- **Technical Details**: 연구에서는 Neural Spline Flows, Conditional Flow Matching, Denoising Diffusion Probabilistic Models의 세 가지 대표 모델을 선택하여 데이터의 차원, 복잡성, 모드 비대칭성에 따라 이들의 정확도, 계산 비용, 생성 속도를 평가합니다. 각 모델은 저차원 데이터의 모드 비대칭성을 잘 포착하거나, 저복잡도 고차원 데이터에서 우수한 성능을 나타내며, 높은 복잡도의 저차원 데이터에서 가장 효과적으로 작동하는 것으로 나타났습니다.

- **Performance Highlights**: Neural Spline Flows(NS)는 저차원 데이터의 모드 비대칭을 포착하는 데 가장 뛰어난 성능을 보였고, Conditional Flow Matching(CFM)은 저복잡도 고차원 데이터에서 우수한 성능을 보였습니다. Denoising Diffusion Probabilistic Models(DDPM)은 복잡한 다중 모드를 잘 모델링하지만, 고차원 데이터에서 정확도가 떨어지는 경향을 보였습니다.



### Stability and Generalization for Distributed SGDA (https://arxiv.org/abs/2411.09365)
- **What's New**: 이 논문에서는 분산(minimax) 최적화(optimization) 알고리즘에 대한 새로운 안정성 기반 일반화 분석 프레임워크를 제안합니다. 이 프레임워크는 Local-SGDA와 Local-DSGDA 두 가지 인기 있는 알고리즘을 통합합니다.

- **Technical Details**: 연구는 분산 SGDA에 대한 안정성 오류(stability error), 일반화 격차(generalization gap), 그리고 인구 위험(population risk)을 서로 다른 설정(SC-(S)C, PL-SC, NC-NC) 하에서 포괄적으로 분석합니다.

- **Performance Highlights**: 이론적 결과는 일반화 격차(generalization gap)와 최적화 오류(optimization error) 간의 상충(trade-off) 관계를 보여주며, 최적 인구 위험을 얻기 위해 하이퍼파라미터(hyperparameters) 선택을 제안합니다. 수치 실험(numerical experiments)은 Local-SGDA와 Local-DSGDA의 이론적 결과를 검증합니다.



### Approximated Variational Bayesian Inverse Reinforcement Learning for Large Language Model Alignmen (https://arxiv.org/abs/2411.09341)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 정렬을 베이지안 역 강화 학습(Bayesian Inverse Reinforcement Learning, BIRL) 문제로 공식화하고, 새로운 학습 목표인 Approximated Variational Alignment (AVA)를 제안합니다. AVA는 각 시연에 대해 직접적인 보상 모델링을 가능하게 하여, 피드백 데이터의 훈련 신호 활용을 극대화합니다.

- **Technical Details**: 모델은 각 시연의 임시 보상을 모델링하며, Approximated Variational Reward Imitation Learning (AVRIL)을 사용하여 보상 분포를 유도합니다. AVA는 선택한 시연과 거부된 시연 간의 보상 차이를 모델링하는 기존 방법과 달리, 모든 시연의 진정한 보상을 직접 모델링하는 방식을 취합니다.

- **Performance Highlights**: AVA는 기존의 LLM 정렬 접근 방식보다 보상 모델링, RL 미세 조정 및 직접 최적화에서 성능이 우수하다는 것을 실험적으로 입증했으며, 보상 해킹 문제를 개선하고 모델의 표현력과 일반화 능력을 향상시켰습니다.



### Pie: Pooling CPU Memory for LLM Inferenc (https://arxiv.org/abs/2411.09317)
- **What's New**: 이 논문은 LLM(대형 언어 모델)의 추론을 위한 새로운 프레임워크인 Pie를 소개합니다. Pie는 CPU 메모리와 GPU 메모리 간의 성능 투명한 스와핑(performance-transparent swapping) 및 적응형 확장(adaptive expansion) 기능을 통해 메모리 요구사항 문제를 해결하고 있습니다.

- **Technical Details**: Pie 프레임워크는 NVIDIA GH200 Grace Hopper Superchip의 높은 대역폭을 활용하여, 데이터 스와핑과 전경 계산(foreground computation)을 동시 진행할 수 있도록 설계되었습니다. 이는 CPU 메모리를 동적으로 조정하여 성능 저하 없이 메모리 사용량을 최적화합니다. 성능 투명한 스와핑은 메모리 용량 증가가 계산 대기 시간에 영향을 주지 않도록 하여, 실제로 GPU가 더 많은 메모리를 가진 것처럼 보이도록 합니다.

- **Performance Highlights**: 실험 결과, Pie는 vLLM보다 최대 1.9배 높은 처리량과 2배 낮은 지연 시간을 달성하며, FlexGen에 비해 지연 시간을 최대 60배 줄이고 처리량을 9.4배 증가시키는 성능을 나타냈습니다. Pie는 또한 GPU 메모리 사용량을 최대 1.67배 줄이면서도 동일한 성능을 유지합니다.



### Approximate Probabilistic Inference forTime-Series Data A Robust Latent Gaussian Model With Temporal Awareness (https://arxiv.org/abs/2411.09312)
- **What's New**: 비정상(non-stationary) 시계열 데이터에 대한 강력한 생성 모델의 개발에 관한 중요한 연구 결과를 발표했습니다.

- **Technical Details**: 제안된 모델은 Time Deep Latent Gaussian Model (tDLGM)으로, Deep Latent Gaussian Model (DLGM)에서 영감을 받아 설계되었습니다. 이 모델은 음의 로그 손실(negative log loss)을 기반으로 한 손실 함수(loss function)를 최소화하도록 훈련됩니다. 또한, 데이터 추세(data trends)를 고려한 정규화기(regularizer)를 통해 안정성을 높였습니다.

- **Performance Highlights**: 실험 결과, tDLGM은 복잡한 시계열 데이터를 재구성하고 생성할 수 있으며, noise 및 결함이 있는 데이터에 대해서도 강건성을 유지하는 것으로 나타났습니다.



### Compression Method for Solar Polarization Spectra Collected from Hinode SOT/SP Observations (https://arxiv.org/abs/2411.09311)
- **What's New**: 이 연구에서는 깊은 오토인코더(deep autoencoder, DAE)와 1D-컨볼루션 오토인코더(1D-convolutional autoencoder, CAE)를 활용하여 태양의 분광 데이터를 압축하는 새로운 딥러닝 기반 기법을 제안합니다. 특히 조용한 태양과 활발한 지역에서의 Stokes I 및 V 편광 스펙트럼을 압축함으로써 극한의 자기장과 관련된 스펙트럼 분석을 수행하였습니다.

- **Technical Details**: 제안된 모델은 두 가지 서로 다른 아키텍처인 DAE와 CAE로 구성됩니다. 두 모델은 스펙트럼 데이터의 복잡성을 줄여주는 특성 추출기(feature extractor)의 역할을 하며, 고차원의 데이터에서 중요한 특성을 압축하여 저차원 표현을 생성합니다. DAE는 완전 연결 층을 사용하고, CAE는 1D-컨볼루션 레이어를 도입하여 데이터를 처리합니다. 이들의 구조는 입력 데이터를 압축하고 다시 원래의 크기로 되살리는 인코더-디코더(encoder-decoder) 방식으로 작동합니다.

- **Performance Highlights**: 실험 결과 CAE 모델이 DAE 모델보다 Stokes 프로필의 재구성에서 우수한 성능을 보였으며, 관측 노이즈 수준의 재구성 오류를 기록하였습니다. 태양 분광 분석에 있어 효과적인 압축 방법을 제안하여, 이상 스펙트럼 신호 탐지와 같은 실질적인 응용 가능성을 강조하고 있습니다.



### A Centralized-Distributed Transfer Model for Cross-Domain Recommendation Based on Multi-Source Heterogeneous Transfer Learning (https://arxiv.org/abs/2411.09286)
Comments:
          Published in: 2022 IEEE International Conference on Data Mining (ICDM) (The authors were affiliated Hangzhou NetEase Cloud Music Technology Co., Ltd.)

- **What's New**: 이 논문에서는 여러 소스 도메인에서 학습한 지식을 활용하여 크로스 도메인 추천(Cross-Domain Recommendation, CDR)의 성능을 향상시키는 중앙 집중식 - 분산 전송 모델(centralized-distributed transfer model, CDTM)을 제안합니다.

- **Technical Details**: CDTM은 두 가지 임베딩 구조를 사용하여 도메인 특화 임베딩(Domain Specific Embedding, DSE)과 글로벌 공유 임베딩(Global Shared Embedding, GSE)을 통해 데이터의 이질성을 다룹니다. GSE는 모든 도메인에서 학습된 공통 특성을 모델링하고, DSE는 각 도메인에 특화된 독특한 특성을 모델링합니다. 이 두 임베딩은 전송 매트릭스와 어텐션 메커니즘을 통해 결합됩니다.

- **Performance Highlights**: 실제 상업 데이터에 대한 오프라인 및 온라인 실험을 통해 제안된 모델이 효과적이고 강건함을 입증했습니다. CDTM은 여러 도메인에서 정보를 동시에 전송하여 추천 정확도를 향상시킬 수 있는 가능성을 보여줍니다.



### Towards efficient compression and communication for prototype-based decentralized learning (https://arxiv.org/abs/2411.09267)
Comments:
          15 pages, 2 tables, 7 figures, 6 algorithms

- **What's New**: 이 논문은 프로토타입 기반의 연합 학습에서 중앙 집계 서버 없이 완전 분산형 구현을 다룹니다. 네트워크 장애에 대한 저항력이 크고 데이터의 통계적 분포 변화에 빠르게 대응하는 장점이 있습니다.

- **Technical Details**: 프로토타입을 기반으로 하는 분산 학습 시스템을 설계하기 위해 데이터 압축 기법을 두 가지 활용합니다. 첫째, 정보 이론적으로 유용한 프로토타입만 전송 (제이슨-샤넌 거리 활용)하하고, 둘째로 군집화를 통해 업데이트 메시지를 압축합니다. 비순차적 대화(Gossip) 방식으로 병렬 통신을 채택하여 정보의 전달 속도를 개선합니다.

- **Performance Highlights**: 개선된 통신 방식 덕분에 학습 알고리즘의 수렴 속도를 저하시킴 없이 통신 부하를 크게 줄일 수 있음을 실험을 통해 보여주었습니다.



### Rethinking Weight-Averaged Model-merging (https://arxiv.org/abs/2411.09263)
- **What's New**: 최근 연구에 따르면, weight-averaged model-merging 기법이 딥러닝에서 효과적인 성능 향상을 가져온다는 사실이 확인되었습니다. 이 연구에서는 모델 가중치를 시각화하여 인지된 패턴이 구조화되고 해석 가능함을 보여줍니다.

- **Technical Details**: 이 논문에서는 weight-averaged model-merging의 세 가지 핵심 관점을 통해 접근합니다: (1) 모델 가중치의 내재적 패턴 분석, (2) 가중치 평균화와 특성 평균화 전략 비교, (3) 다양한 매개변수 크기 변화에 따른 예측 안정성 탐구.

- **Performance Highlights**: 모델 병합 후, 다양한 학습 템플릿을 보존함으로써 상호작용을 통한 성능 향상이 이루어집니다. 연구 결과는 모델 병합의 근본적인 원리를 밝혀내어 향후 연구에 대한 유용한 제안과 공개 소스를 제공합니다.



### FluidML: Fast and Memory Efficient Inference Optimization (https://arxiv.org/abs/2411.09242)
- **What's New**: FluidML은 기계 학습(ML) 모델의 실행을 보다 빠르고 메모리 효율적으로 최적화할 수 있는 일반적인 메모리 관리 및 최적화 프레임워크입니다. 이 프레임워크는 전체 그래프에 걸쳐 최적의 연산자 메모리 레이아웃을 공동 최적화하고, 메모리 접근에 친화적인 실행 청사진을 생성하여 연산자 및 끝에서 끝까지의 그래프 실행 속도를 높입니다.

- **Technical Details**: FluidML은 Open Neural Network Exchange (ONNX)를 프론트 엔드 형식으로 사용하여 수십 개의 널리 사용되는 연산자를 지원하며, LLVM과 Multi-Level Intermediate Representation (MLIR) 위에 구축된 FluidML 컴파일러를 포함합니다. FluidML은 Intel, AMD, Aarch64의 세 가지 플랫폼에서 Just-in-Time (JIT) 실행 엔진으로 평가하여 인기 있는 언어 모델에서 최대 25.38%의 지연 시간을 줄이고, 피크 메모리 사용량을 최대 41.47%까지 줄일 수 있음을 입증했습니다.

- **Performance Highlights**: FluidML은 모델 실행 청사진을 변환하여 얻은 성과로, 관측된 지연 시간 감소가 언어 모델(BERT 등)에서 최대 25.38%에 달하며, 널리 사용되는 연산자(MatMul)에서 최대 17.87%의 성능 향상을 이루었습니다. 또한, FluidML은 피크 메모리 사용량을 41.47% 줄일 수 있어 메모리 최적화 측면에서도 큰 효과를 보여줍니다.



### Rethinking the "Heatmap + Monte Carlo Tree Search" Paradigm for Solving Large Scale TSP (https://arxiv.org/abs/2411.09238)
- **What's New**: 이 연구는 Travelling Salesman Problem (TSP)에 대한 heatmap + Monte Carlo Tree Search (MCTS) 접근 방식을 재조명하였으며, MCTS 전략의 조정이 최적화 문제 해결에 있어 도드라진 영향을 미친다는 점을 강조합니다.

- **Technical Details**: 연구는 MCTS 전략이 문제 해결의 질에 미치는 영향을 분석하며, 특정 MCTS 구성 요소를 조정했을 때 그 성능이 크게 향상될 수 있음을 보여줍니다. 또한, 복잡한 heatmap 대신 간단한 k-nearest 기반의 파라미터 없는 heatmap이 또한 우수한 결과를 낼 수 있음을 발견하였습니다.

- **Performance Highlights**: 본 연구의 접근 방식은 다양한 TSP 문제 크기에서 경쟁력 있는 성능을 달성하며, 기존의 heatmap 기반 방식의 세부화보다 MCTS의 조정과 일반적인 방법론의 재고가 필요함을 제안합니다.



### Ghost-Connect Net: A Generalization-Enhanced Guidance For Sparse Deep Networks Under Distribution Shifts (https://arxiv.org/abs/2411.09199)
Comments:
          21 pages, 4 figures, 3 subfigures, 42 tables

- **What's New**: 이번 연구에서는 Ghost Connect-Net (GC-Net)이라는 동반 네트워크를 소개하여 딥 뉴럴 네트워크(DNN)의 성능을 향상시키고자 하였습니다. GC-Net은 원래 네트워크의 연속적인 층 간 연결성을 모니터링하며, 분포 변화(distribution shifts)에 대한 일반화(generalization) 이점을 제공합니다.

- **Technical Details**: GC-Net의 가중치(weights)는 원래 네트워크의 연속적인 층 간 연결 측정을 나타냅니다. GC-Net을 가지치기(pruning)한 후, 가지치기된 위치를 원래 네트워크에 맵핑하여 연결성을 기반으로 한 가지치기 방법과 결합합니다. 이 방법은 Magnitude 기반의 가지치기와 연결성을 기반으로 한 가지치기를 혼합합니다.

- **Performance Highlights**: CIFAR-10, Fashion MNIST, Tiny ImageNet과 같은 일반적인 DNN 벤치마크를 사용한 실험 결과, GC-Net 가이드를 활용하여 네트워크의 후반부 층에 대한 가지치기를 수행하고, 초기 층에 대해서는 직접적인 가지치기를 적용했을 때 유망한 성과를 보였습니다.



### Dynamic technology impact analysis: A multi-task learning approach to patent citation prediction (https://arxiv.org/abs/2411.09184)
- **What's New**: 본 연구는 특허 인용 정보를 사용하여 기술의 영향을 분석하는 데 있어 머신 러닝(ML) 모델의 한계를 극복하기 위해 다중 작업 학습(Multi-task Learning, MTL) 접근법을 제안합니다.

- **Technical Details**: ML 기반의 기존 방법들은 시간이 지남에 따라 기술 영향의 동적 특성과 서로 다른 기간의 이러한 영향 간의 상호 의존성을 충분히 고려하지 못하는 경향이 있습니다. 본 연구에서는 인용 분석을 통해 기술 영향의 패턴을 식별하고, 여러 특허 지표를 사용하여 반복적으로 인용 수를 예측하는 MTL 모델을 개발합니다. 또한, SHapley Additive exPlanation 방법을 사용하여 주요 입력 지표의 변화와 패턴을 분석합니다.

- **Performance Highlights**: 전지 기술에 대한 사례 연구를 통해 본 접근법이 기술 영향 이해를 심화시키고 예측 정확도를 개선함으로써 학계와 산업계 모두에 유용한 통찰을 제공함을 보여줍니다.



### SAFES: Sequential Privacy and Fairness Enhancing Data Synthesis for Responsible AI (https://arxiv.org/abs/2411.09178)
- **What's New**: 이번 논문에서는 데이터 프라이버시와 공정성을 동시에 고려하는 SAFES라는 새로운 절차를 소개합니다. SAFES는 Differential Privacy (DP) 데이터 합성과 공정성을 염두에 둔 데이터 변환을 결합하여 민감한 개인 정보를 보호하고 공정성 메트릭을 향상시키는 방안을 제시합니다.

- **Technical Details**: SAFES는 Sequential PrivAcy and Fairness Enhancing data Synthesis의 약자로, DP 데이터 합성과 공정성을 고려한 사전 처리 변환을 전략적으로 결합하여 데이터를 생성합니다. 이는 개인 프라이버시와 공정성을 조정 가능한 매개변수를 통해 완벽하게 조절할 수 있는 장점을 가지고 있습니다.

- **Performance Highlights**: 성능 평가 결과, SAFES로 생성된 합성 데이터는 합리적인 프라이버시 손실을 기준으로 공정성 메트릭을 크게 향상시키면서도 상대적으로 낮은 유틸리티 손실을 보였습니다. 이는 SAFES가 다양한 DP 보장과 공정성 메트릭을 충족하는 변환 방법을 수용할 수 있는 일반적인 프레임워크임을 보여줍니다.



### Towards Scalable Handwriting Communication via EEG Decoding and Latent Embedding Integration (https://arxiv.org/abs/2411.09170)
Comments:
          4 pages, 2 figures, 1 table, Name of Conference: International Conference on Brain-Computer Interface

- **What's New**: 이 연구에서는 EEG 신호와 함께 필기 동작에 대한 카테고리를 분류하는 새로운 접근 방식인 CEBRA(consistent embeddings from high-dimensional neural recordings using auxiliary variables)를 사용하여 더욱 세밀한 필기 인식을 달성했습니다.

- **Technical Details**: CEBRA는 행동 데이터와 신경 데이터를 함께 활용하여 해석 가능한 일관된 임베딩을 생성합니다. 이 방법은 EEG 신호와 손의 운동학 데이터를 통합하여 CNN(Convolutional Neural Network)을 통해 처리하여 고차원 신경 기록에서 의미 있는 저차원 임베딩을 추출합니다.

- **Performance Highlights**: 모델은 9개의 다양한 손글씨 문자를 분류하여 91%의 정확도를 달성하였으며, 이 연구의 접근 방식은 기존 CNN, EEGNet 및 DeepConvNet보다 우수한 성능을 보였습니다.



### GRAINRec: Graph and Attention Integrated Approach for Real-Time Session-Based Item Recommendations (https://arxiv.org/abs/2411.09152)
Comments:
          Accepted to the 2024 IEEE International Conference on Big Data (IEEE BigData 2024)

- **What's New**: 최근 딥러닝 기법을 활용한 세션 기반 추천 모델이 성능 향상을 보여주고 있습니다. GRAINRec이라는 새로운 모델을 제안하며, 이 모델은 그래프(Graph)와 주의(attention) 메커니즘을 통합하여 실시간으로 추천을 생성할 수 있도록 설계되었습니다.

- **Technical Details**: GRAINRec는 LESSR 모델의 기초 위에 구축되며, 세션 내 모든 항목의 중요성을 고려하여 동적으로 관련 추천을 예측합니다. 세션은 보통 몇 분 간의 시간적 근접성으로 정의된 항목의 정렬된 시퀀스로, 이 모델은 실시간 추론을 구현하기 위해 근접 이웃 매트릭스(nearest neighbor matrix) 방식을 채택하고 있습니다.

- **Performance Highlights**: 모델의 평가 결과, 오프라인 평가 지표 전반에서 평균 1.5%의 개선을 보였으며, 2주 간의 A/B 테스트에서 클릭률(click through rate)이 10%, 수요(attributable demand)가 9% 증가했습니다.



### Laplace Transform Interpretation of Differential Privacy (https://arxiv.org/abs/2411.09142)
- **What's New**: 이 논문에서는 Differential Privacy (DP) 개념을 privacy loss distribution의 Laplace transform을 활용하여 새로운 방식으로 해석하는 방법을 제시합니다. 이 해석 방법은 DP의 시간 및 주파수 영역의 이중성을 활용하여 DP 속성을 분석하고 연결합니다.

- **Technical Details**: 우리는 $(q, ho(q))$-Rényi DP 곡선과 $(	ext{ε}, 	ext{δ(ε)})$-DP 곡선이 서로의 Laplace 변환과 역 Laplace 변환으로 연결될 수 있음을 보여줍니다. 또한, $f$-DP의 대칭성에 대한 문제를 해결하여 모든 DP 개념 간의 동등성을 가져올 수 있도록 합니다.

- **Performance Highlights**: Laplace transform 기반의 분석을 통해 $(	ext{ε}, 	ext{δ})$-DP 보장에 대한 적응형 조합 정리를 증명하였습니다. 이 정리는 모든 ε 값에 대해 정확히 타이트합니다.



### Complexity-Aware Training of Deep Neural Networks for Optimal Structure Discovery (https://arxiv.org/abs/2411.09127)
Comments:
          28 pages, 4 figures, 5 tables

- **What's New**: 본 논문에서는 사전 훈련된 네트워크 없이 훈련 중에 동작하는 심층 신경망의 단위/필터 및 레이어 가지치기를 위한 새로운 알고리즘을 제안합니다. 이 알고리즘은 예측 정확도와 가지치기 수준을 최적의 trade-off 하며, 세 가지 사용자 정의 매개변수를 사용해 레이어와 단위/필터 가지치기를 균형 있게 조절합니다.

- **Technical Details**: 모델 구조 최적화는 네트워크 가중치와 랜덤 변수의 변량 Bernoulli 배포 매개변수에 대한 확률적 최적화 문제의 솔루션으로 찾습니다. 가지치기는 변량 매개변수가 0으로 수렴할 때 발생하여 해당 구조를 비활성화합니다. 이 접근은 예측 정확도와 네트워크 가지치기 목표를 결합한 비용 함수를 정의합니다.

- **Performance Highlights**: CIFAR-10/100 및 ImageNet 데이터셋에서 ResNet 아키텍처를 사용하여 성능 평가를 했으며, 제안된 방법은 레이어 또는 단위 가지치기보다 개선된 성능을 보이며 사전 훈련된 네트워크를 필요로 하는 다른 알고리즘과 비교해도 유리한 결과를 보여줍니다.



### Neural Graph Simulator for Complex Systems (https://arxiv.org/abs/2411.09120)
- **What's New**: 이번 연구에서는 Neural Graph Simulator (NGS)를 도입하여 시간 불변 자율 시스템을 그래프에서 시뮬레이션하는 새로운 방법을 제안합니다. NGS는 그래프 신경망 (Graph Neural Network, GNN)을 활용하여 다양한 동적 시스템을 효율적으로 시뮬레이션할 수 있는 통일된 프레임워크를 제공합니다.

- **Technical Details**: NGS는 크게 세 가지 유형의 계수: 노드 계수 (node coefficients), 엣지 계수 (edge coefficients), 그리고 글로벌 계수 (global coefficient)를 사용해서 시스템의 동적 상태를 업데이트합니다. 이러한 계수들은 각 노드 및 엣지에 벡터로 할당되며, 그래프의 인접 행렬 (adjacency matrix) 및 샘플링된 시간 간격 Δtm을 입력으로 받아 작업을 수행합니다.

- **Performance Highlights**: NGS는 전통적인 수치 해법에 비해 10^5배 이상의 성능 향상을 보여주며, 실제 교통 데이터를 활용한 예측에서도 최첨단 정확도를 달성했습니다. 이 연구는 NGS의 유연성이 다양한 시스템에 적용될 수 있음을 보여줍니다.



### Efficiently learning and sampling multimodal distributions with data-based initialization (https://arxiv.org/abs/2411.09117)
- **What's New**: 이번 연구에서는 Markov 체인을 이용해 소수의 샘플로부터 다중 모드(multi-modal) 분포를 샘플링하는 문제를 다룹니다. 기존의 느린 mixing 문제를 극복하기 위해 고차원 스펙트럴 갭(higher-order spectral gap) 개념을 적용하여, $	ilde O(k/	ext{ε}^2)$ 샘플만으로도 효율적인 샘플 생성을 가능하게 했습니다. 또한 이 연구는 Koehler와 Vuong(2023)의 결과를 일반화하여, $k$에 선형적으로 의존하도록 개선하였습니다.

- **Technical Details**: 이론적으로, 고차원 스펙트럴 갭은 Markov 과정의 전이 행렬(transition matrix)이나 생성자(generator)의 중요한 수학적 객체로서 정의됩니다. 연구에서 제시된 일반적인 도구는 MCMC 체인의 다중 모드 상황을 분석하기 위한 것으로, 해당 이론에서는 Poincaré 부등식(Poincaré inequality)을 만족하는 $k$분포의 혼합에 적용될 수 있으며, log-Sobolev 부등식을 만족할 경우 더 빠른 수렴을 보입니다. 또한 Langevin 확산(Langevin diffusion)과 Glauber 동역학(Glauber dynamics)에 대해서도 적용 가능합니다.

- **Performance Highlights**: 이 연구의 성과는 여러 데이터 기반 초기화(data-based initialization) 방법의 성공을 정당화하는 데 기여하며, 저복잡도의 Ising 측정(low-complexity Ising measures) 클래스가 샘플에서 효율적으로 학습될 수 있음을 처음으로 보여줍니다. 이론적 개선을 통해, MCMC는 데이터 분포에 대한 느린 mixing에도 불구하고 효과적으로 작동할 수 있습니다.



### Reducing Reasoning Costs - The Path of Optimization for Chain of Thought via Sparse Attention Mechanism (https://arxiv.org/abs/2411.09111)
Comments:
          The main text is 9 pages, totaling 13 pages; 5 figures, 3 tables; preprints have been submitted to NeurIPS 2024 Workshop MusIML and OpenReview

- **What's New**: 이 연구는 대형 언어 모델의 추론 비용 증가 문제를 해결하기 위해 몇 가지 관련 토큰에만 초점을 맞춘 sparse attention(희소 주의) 메커니즘을 제안합니다.

- **Technical Details**: 연구자는 새로운 attention(주의) 메커니즘을 구성하였고, 사용자 정의 GPT로 훈련된 GiantRabbit을 실험 도구로 사용하였습니다. 실험은 MIT OpenCourseWare의 선형대수학 테스트 질문들을 해결하기 위해 GiantRabbit과 o1 Preview 모델의 추론 시간, 정확성 점수 및 사고의 연쇄 길이를 비교하였습니다.

- **Performance Highlights**: 실험 결과, GiantRabbit의 추론 시간과 사고의 연쇄 길이가 o1 Preview보다 significantly lower(상당히 낮은) 것으로 나타났으며, 이는 sparse attention 메커니즘이 사고의 연쇄 추론을 줄이는 데 있어 가능성을 확인시켜 줍니다.



### Continuous GNN-based Anomaly Detection on Edge using Efficient Adaptive Knowledge Graph Learning (https://arxiv.org/abs/2411.09072)
Comments:
          Accepted to DATE 2025

- **What's New**: 본 연구는 클라우드 종속성을 줄이고, 엣지 디바이스에서 지속적인 지식 그래프(KG) 적응을 가능하게 하는 새로운 프레임워크를 제안합니다. 이를 통해 다양한 환경에서 비디오 이상 탐지 모델의 성능과 견고성을 향상시킵니다.

- **Technical Details**: 제안된 프레임워크는 세 가지 단계로 구성됩니다: 미션 특화 KG 생성, 경량 GNN 기반 의사 결정 모델 훈련, 배포 및 지속적인 KG 적응 학습. 이 과정에서 KG는 프루닝(pruning), 변경(altering), 노드 생성(creating nodes)의 세 가지 단계로 동적으로 수정됩니다.

- **Performance Highlights**: 이 연구는 엣지 컴퓨팅 환경에서 새로운 데이터 트렌드에 실시간으로 적응할 수 있는 이상 탐지 모델을 제공하며, 클라우드 기반 계산에 의존하지 않고도 모델을 업데이트할 수 있는 아이디어를 제시합니다.



### Optimisation Strategies for Ensuring Fairness in Machine Learning: With and Without Demographics (https://arxiv.org/abs/2411.09056)
Comments:
          PhD thesis. arXiv admin note: text overlap with arXiv:2310.11407

- **What's New**: 이 논문은 AI 관련 알고리즘에서의 공정성을 확보하는 중요성을 강조합니다. 머신러닝 공정성 분야의 최신 동향을 제시하고, 두 개의 공식적인 프레임워크를 소개하여 머신러닝 공정성 관련 문제를 해결하려고 합니다.

- **Technical Details**: 첫 번째 프레임워크는 operator-valued optimisation과 min-max objectives를 사용하여 시계열 문제에서의 불공정성을 다룹니다. 두 번째 프레임워크는 성별이나 인종과 같은 민감한 속성이 부족한 데이터셋에서의 문제를 해결하기 위해 개발된 group-blind bias-repair 접근법을 도입합니다. 각 프레임워크에 대한 자세한 알고리즘 분석과 수렴 보장(convergence guarantees)을 제공합니다.

- **Performance Highlights**: 첫 번째 프레임워크는 COMPAS 벤치마크 데이터셋에서 최첨단 성능을 보여주며, 두 번째 프레임워크는 Adult Census Income 데이터셋에 대한 분석을 통해 그 효과성을 입증하였습니다.



### SAFELOC: Overcoming Data Poisoning Attacks in Heterogeneous Federated Machine Learning for Indoor Localization (https://arxiv.org/abs/2411.09055)
- **What's New**: 이 논문에서는 모바일 기기에서의 indoor localization(실내 위치 추적) 문제를 해결하기 위한 새로운 프레임워크인 SAFELOC을 제안합니다. SAFELOC은 기기 이질성(device heterogeneity) 및 ML 데이터 중독 공격(data poisoning attack)과 같은 도전 과제를 극복하는 데 중점을 두고 있습니다.

- **Technical Details**: SAFELOC은 분산 및 협력적 학습 환경을 목표로 하며, federated learning (FL)을 사용하여 사용자 데이터의 프라이버시를 보호합니다. heterogenous(이질적) 모바일 기기를 고려하여 설계된 새로운 융합 신경망 아키텍처(fused neural network architecture)로 데이터 중독 탐지와 위치 추적을 수행합니다. 또한, 탐지된 데이터 중독 상황의 심각도에 따라 적응하는 동적 주목도 맵(dynamic saliency map)을 기반으로 한 집계 전략(aggregation strategy)을 도입합니다.

- **Performance Highlights**: SAFELOC은 다양한 빌딩 평면도, 모바일 기기, ML 데이터 중독 공격 시나리오에 따라 기존의 indoor localization 프레임워크에 비해 평균 위치 추적 오류(mean localization error)를 최대 5.9배 감소시키고, 최악의 경우 위치 추적 오류(worst-case localization error)를 7.8배까지 줄이며, 모델 추론 지연(latency)을 2.1배 줄이는 성능 향상을 보여줍니다.



### Anomaly Detection in Large-Scale Cloud Systems: An Industry Case and Datas (https://arxiv.org/abs/2411.09047)
- **What's New**: 본 논문은 IBM Cloud에서 수집된 새로운 고차원 데이터 세트를 소개하며, 이 데이터 세트는 4.5개월 동안 39,365개의 행(row)과 117,448개의 열(column)로 구성된 텔레메트리(telemetry) 데이터를 포함합니다. 이 데이터 세트는 클라우드 시스템 모니터링을 위한 리소스를 제공하며, 실제 데이터에서의 이상 탐지(anomaly detection) 방법을 보다 효율적으로 테스트할 수 있도록 돕습니다.

- **Technical Details**: 이 논문은 클라우드 시스템에서의 이상 탐지를 위한 새로운 데이터 세트를 도입하고, 기계 학습(machine learning) 모델을 활용하여 이상 탐지의 적용을 보여줍니다. 또한, 'curse of dimensionality'라는 문제와 함께 높은 차원의 데이터 처리에서의 도전 과제에 대해 논의합니다. 기존의 연구들이 소규모 데이터 세트에 중점을 두었다면, 본 연구는 실제 IBM Cloud 시스템에서의 대규모 데이터 세트를 공동체에 공유함으로써 이상 탐지 방법의 평가를 보다 포괄적으로 수행할 수 있게 합니다.

- **Performance Highlights**: 제안된 모델은 이전 모니터링 시스템에 비해 최대 20분 전에 이상을 탐지할 수 있으며, 잘못된 경고(false alerts)를 크게 감소시켰습니다. 이러한 성능은 공개적으로 이용 가능한 벤치마크 데이터 세트와 IBM Cloud 플랫폼에서 수집된 실제 데이터를 통해 검증되었습니다.



### Transformer-based Time-Series Biomarker Discovery for COPD Diagnosis (https://arxiv.org/abs/2411.09027)
Comments:
          Accepted as a workshop paper to NeurIPS 2024

- **What's New**: 이번 연구에서는 기존의 분산측정치(summary measures) 대신 고차원 원시 스피로그램(raw spirogram) 데이터를 사용하여 만성 폐쇄성 폐질환(COPD)의 임상적 관련 지표를 예측할 수 있는 새로운 방법을 제안합니다. 이 방법은 transformer 기반 딥러닝 기술을 활용하며, 기존 연구보다 우수한 성능을 보이면서도 계산 효율성도 향상되었습니다.

- **Technical Details**: UK Biobank 데이터셋을 기반으로 하여, 10밀리초 간격으로 기록된 폐활량수치(spirometry data)를 사용하였습니다. 우리는 스피로그램의 시간-부피(curves)를 흐름-부피(Flow-Volume) 곡선으로 변환하여 모델에 입력합니다. 또, 인구학적 정보(age, sex, smoking status, height)를 포함하여 주어진 데이터를 patchify하고 transformer 인코더를 통해 분석합니다. 성능 예측은 COPD 위험, 악화 및 사망률과 같은 세 가지 임상 관련 지표로 수행됩니다.

- **Performance Highlights**: 제안된 모델은 대부분의 평가 지표에서 이전 연구보다 약 4.5배 더 계산적으로 효율적이고, 중요한 스피로그램 부분을 잘 식별해낼 수 있음을 보여주었습니다. 또한, 전문가의 의견과 일치하는 예측 결과를 제공하여 모델의 해석 가능성을 높였습니다.



### Cut Your Losses in Large-Vocabulary Language Models (https://arxiv.org/abs/2411.09009)
Comments:
          Code is available at this https URL

- **What's New**: 이 논문에서는 Cut Cross-Entropy (CCE)라는 새로운 방법론을 제안합니다. 이는 대형 언어 모델(LLM)의 학습 시 기하급수적으로 증가하는 메모리 사용량을 감소시키기 위해 개발되었습니다. CCE는 모든 토큰에 대한 logits를 메모리에 저장하지 않고도 cross-entropy 손실을 계산할 수 있게 해줍니다.

- **Technical Details**: CCE는 정답 토큰에 대한 logit만 계산하고, 모든 logits에 대한 log-sum-exp를 즉석에서 계산하여 메모리 사용량을 줄입니다. 이를 통해 단순한 단일 ground-truth label에 대한 index matrix multiplication과 log-sum-exp 연산으로 cross-entropy 손실을 분해하게 됩니다. CCE 구현을 위해 사용자 정의 CUDA 커널을 활용하여 SRAM에서 매트릭스 곱셈 및 log-sum-exp 축소를 수행합니다.

- **Performance Highlights**: Gemma 2(2B)의 경우, CCE는 손실 계산의 메모리 사용량을 24GB에서 1MB로 줄이고, 분류기 헤드의 총 훈련 시간 메모리 사용량은 28GB에서 1GB로 감소시켰습니다. CCE의 구현은 빠르기를 해치지 않으면서도 메모리 소모를 획기적으로 줄이는 성과를 보여주었습니다.



### Refusal in LLMs is an Affine Function (https://arxiv.org/abs/2411.09003)
- **What's New**: 본 논문에서는 언어 모델의 행동을 조정하는 새로운 접근법인 affine concept editing (ACE)을 제안합니다. 이 방법은 모델의 activation을 직접 조작하여 원하는 출력된 결과를 얻을 수 있도록 합니다. 특히, ACE는 기존의 방향성 절제(directional ablation) 및 activation 추가(activation addition)의 일반화를 통해 모델의 거부 응답을 정밀하게 제어할 수 있습니다.

- **Technical Details**: ACE는 affine decomposition을 기반으로 하여 모델의 activation 벡터를 분석합니다. 이 기법은 고차원 공간에서 개념을 선형적으로 표현하는 'linear representation hypothesis'에 기초하고 있으며, 기존의 기법들보다 보다 일반적인 방법으로 모델의 행동을 조정합니다. 구체적으로, ACE는 기존의 'zero-point' 개념을 확장하여 다양한 reference points에 따라 변동하는 선형 및 상수 항을 포함하는 affine 함수를 사용합니다.

- **Performance Highlights**: 연구 결과에 따르면 ACE는 다양한 프롬프트 유형에 대해 모델의 거부 응답을 보다 일관되게 제어하는 성과를 보여주었습니다. 특히, ACE는 기존 기술이 비일관한 출력을 유발하는 특정 모델에서도 안정적인 결과를 도출하는 것으로 나타났습니다. 이러한 결과는 다른 LLM 기반 스코어링 방법을 통해 평가되었습니다.



### Lynx: Enabling Efficient MoE Inference through Dynamic Batch-Aware Expert Selection (https://arxiv.org/abs/2411.08982)
- **What's New**: 이 논문에서는 Lynx라는 시스템을 소개하여 Mixture-of-Experts (MoE) 아키텍처의 추론 지연(latency)을 효과적으로 줄이는 방법을 제시합니다. Lynx는 동적이고 배치 인지(batch-aware) 전문가 선택을 통해 효율적인 MoE 추론을 가능하게 합니다.

- **Technical Details**: Lynx는 전문가의 중요성을 토큰 및 추론 단계에 따라 분석하여 실시간 최적화를 가능하게 합니다. 이 시스템은 모형 수정이나 작업별 조정 없이 기존 MoE 배포에 즉시 적용될 수 있습니다. Lynx는 전문가 선택에서의 계층 구조를 활용하여 주요 전문가 선택을 우선하고, 디코드 단계에서 더 공격적인 최적화를 수행합니다.

- **Performance Highlights**: Lynx는 복잡한 코드 생성 및 수학적 추론 작업에서 baseline 모델에 비해 1.55배 지연(latency) 감소를 이루는 동시에 모델 정확도의 손실을 거의 없이 유지하는 성능을 보여줍니다.



### Sparse Upcycling: Inference Inefficient Finetuning (https://arxiv.org/abs/2411.08968)
Comments:
          12 pages, 4 figures, To appear in the 4th NeurIPS Workshop on Efficient Natural Language and Speech Processing (ENLSP), 2024

- **What's New**: 이 연구에서는 사전 훈련된 밀집 모델을 Mixture-of-Experts (MoE) 아키텍처로 변환하는 Sparse Upcycling 방법이 Continued Pretraining (CPT) 보다 어떤지에 대한 비교를 진행했습니다. 실험 결과, 특정 상황에서 Sparse Upcycling이 CPT에 비해 20% 이상의 향상을 이끌어낼 수 있음을 발견하였습니다. 그러나 이 방법은 큰 모델의 경우 40% 느려짐을 초래하는 탐색 비용이 큽니다.

- **Technical Details**: Sparse Upcycling은 밀집 모델의 파라미터 수를 늘리고 질을 향상시키는 방법으로, MoE 아키텍처를 사용하여 모델의 효율성을 높입니다. 각 모델은 436M 및 1.4B 크기로 훈련되었으며, Dense Pretraining과 CPT/Upcycling 두 단계로 나누어 훈련되었습니다. 실험 결과, 업사이클링된 모델이 CPT 보다 낮은 손실(loss)을 달성하여 품질이 개선됨을 보여줍니다.

- **Performance Highlights**: 두 가지 모델 크기에 대해 CPT와 Sparse Upcycling 모델 간의 지연(latency) 및 처리량(throughput)을 비교했습니다. 436M 및 1.4B 크기의 모델에서 업사이클링된 모델은 CPT보다 더 나은 품질을 보여주었으나, 처리 속도 측면에서는 상당한 비용이 발생했음을 알 수 있습니다. 이는 특히 대규모 실세계 적용에서의 유용성에 한계를 줄 수 있습니다.



### Inconsistencies In Consistency Models: Better ODE Solving Does Not Imply Better Samples (https://arxiv.org/abs/2411.08954)
Comments:
          NeurIPS 2024 ATTRIB Workshop

- **What's New**: 최근 Consistency Models (CMs)이 새로운 확산 모델 증류 방법으로 주목받고 있으며, 이들은 몇 번의 반복만으로도 고품질 샘플을 생성할 수 있는 가능성을 보여줍니다. 본 논문에서는 기존 확산 모델의 확률 흐름 ODE(ordinary differential equation)를 직접 최소화하는 Direct CMs를 소개합니다.

- **Technical Details**: Direct CMs는 CMs와 비교하여 ODE 해결 오류를 줄이는데 더 효과적이지만, 샘플 품질이 현저히 저하되는 반응을 보였습니다. 이는 ODE 해결이 꼭 샘플 품질 향상으로 이어지지 않는다는 새로운 시사점을 제공합니다.

- **Performance Highlights**: 실험 결과, Direct CMs는 PF ODE를 보다 잘 해결하지만 예기치 않게 샘플 품질이 나빠지는 결과를 보여, CMs와 그 변형들에서 암묵적으로 가정된 ‘좋은 ODE 해결이 좋은 샘플 품질을 보장한다’는 개념에 의문을 제기합니다.



### Conditional regression for the Nonlinear Single-Variable Mod (https://arxiv.org/abs/2411.09686)
Comments:
          55 pages, 10 figures

- **What's New**: 이 논문은 비선형 성격을 가진 회귀 문제에 대한 새로운 모델을 제시하며, 데이터의 차원 수의 저주(curse of dimensionality)를 피할 수 있는 가능성을 탐구합니다. 특히, 특정 조건 하에 비모수적(nonparametric) 추정기를 제안하고, 이 추정기가 관측 오차 수준까지 성능을 끌어올릴 수 있음을 입증합니다.

- **Technical Details**: 논문의 주요 모델은 함수 F가 F(X) = f(Π_γ X) 형태로 정의되고, 여기서 Π_γ는 매개변수 γ에 대한 최단 거리 투영(projection)입니다. f는 특정 조건, 예를 들어 대략적인 단조(monotonicity)를 만족하는 경우에 대해 nonparametric 회귀에서 최적의 min-max(최소 최대) 학습 속도에 도달할 수 있음을 보여줍니다.

- **Performance Highlights**: 이 모델은 O(d²n log n) 시간 안에 구축될 수 있으며, 학습 경계의 모든 상수들은 차원 d에 대한 저차 다항식의 최대값을 가집니다. 이는 비모수 회귀의 성능을 상당히 개선할 수 있는 잠재력이 있음을 의미합니다.



### Towards a Classification of Open-Source ML Models and Datasets for Software Engineering (https://arxiv.org/abs/2411.09683)
Comments:
          5 pages, 8 figures

- **What's New**: 본 연구는 오픈소스 플랫폼 Hugging Face (HF)에서 소프트웨어 공학 (Software Engineering, SE) 필요에 맞춘 Pre-Trained Models (PTMs) 및 데이터셋의 분류를 시도하며, PTMs의 시간에 따른 진화를 분석합니다.

- **Technical Details**: HF API를 사용하여 1,060,419개의 PTMs와 229,767개의 데이터셋을 수집한 후, SE 관련 메타 데이터를 분석하여 PTMs와 데이터셋을 10,077개 PTM과 1,836개 데이터셋으로 정제했습니다. 또한, Gemini 1.5 Pro를 사용하여 SE 관련성을 확인했습니다.

- **Performance Highlights**: 코드 생성을 위한 PTMs가 가장 일반적이며, 2023 Q2 이후 SE를 위한 PTMs의 수가 현저히 증가하였습니다. 그러나 현재 PTMs의 33%는 모델 카드가 없고, 65%는 SE 작업을 언급하지 않고 있으며, 데이터셋 내에서도 SE 작업 언급 비율은 0.80%에 불과합니다.



### Med-Bot: An AI-Powered Assistant to Provide Accurate and Reliable Medical Information (https://arxiv.org/abs/2411.09648)
Comments:
          3 figures, 5 pages Keywords-LLM, AI-powered healthcare, Medical chatbot, Context-based interaction, Llama-assisted data processing, AutoGPT-Q, PyTorch, TensorFlow, Reliable medical information, Machine learning in healthcare, Conversational AI

- **What's New**: 이번 연구에서는 Med-Bot이라는 AI 기반의 챗봇을 소개합니다. Med-Bot은 의료 정보를 제공하기 위해 설계되었으며, 자연어 이해(Natural Language Understanding)의 복잡성을 처리할 수 있도록 여러 고급 라이브러리와 프레임워크를 활용합니다.

- **Technical Details**: Med-Bot은 PyTorch, Chromadb, Langchain, AutoGPT-Q와 같은 기술을 사용하여 구축되었습니다. 이러한 기술들은 자연어 처리의 효율성을 높이고, 의료 문헌의 PDF 기반 쿼리에 대한 응답 성능을 향상합니다. Llama-assisted 데이터 처리 기법과 AutoGPT-Q 통합이 특징적입니다.

- **Performance Highlights**: Med-Bot은 사용자에게 신뢰성 있는 의료 정보를 제공하는데 중점을 두며, 시뮬레이션된 의료 제공자와의 상호작용을 통해 개인화된 응답을 생성합니다. 다양한 최신 기법을 통해 의료 챗봇의 기능을 강화하고, 사용자 경험을 향상시키는데 기여하고 있습니다.



### How do Machine Learning Models Change? (https://arxiv.org/abs/2411.09645)
- **What's New**: 이번 연구는 Hugging Face (HF) 플랫폼에서 호스팅되는 기계 학습(ML) 모델의 변경 패턴을 포괄적으로 조사한 최초의 장기 연구입니다.

- **Technical Details**: 본 연구에서는 50,000개 이상의 모델에서의 200,000건의 커밋(commit)과 1,200개의 릴리스(release)를 분석하기 위해 repository mining 기법과 장기적 분석 방법론을 사용했습니다. 또한, 커밋 분류를 위해 ML 변경 분류 체계를 복제 및 확장하고, Bayesian networks를 활용하여 시간에 따른 커밋 및 릴리스 활동의 패턴을 밝혀냈습니다.

- **Performance Highlights**: 연구 결과, 커밋 활동은 CRISP-DM과 같은 데이터 과학 방법론과 일치하며, 주요 업데이트는 문서화 중심으로 이루어지는 경향이 있습니다. 인기 있는 프로젝트는 초기 단계에서 인프라 개선에 중점을 두며, 협업이 활발한 프로젝트는 문서화 기준이 향상되는 경향을 보였습니다. 이러한 통찰은 커뮤니티 플랫폼에서의 모델 변화 이해를 높이고, 모델 유지 관리의 모범 사례에 대한 귀중한 지침을 제공합니다.



### Neural Operators Can Play Dynamic Stackelberg Games (https://arxiv.org/abs/2411.09644)
- **What's New**: 이 논문은 동적 스택엘버그 게임(Dynamic Stackelberg games)에서 후행자의 최적 반응(operator) 문제가 해결되는 방법을 제시합니다. 특히 후행자의 최적 반응이 일반적으로 분석적으로 다루기 어려운 상황에서, 	extit{attention-based neural operator}를 이용한 근사 구현을 보여줍니다.

- **Technical Details**: 후행자의 최적 반응 연산자는 리더의 제어에 대한 함수로서 보통 해석적으로 다루기 어렵지만, 이 논문에서는 이를 	extit{attention-based neural operator}로 근사할 수 있다는 점을 제시합니다. 또한, 논문에서는 스택엘버그 게임의 가치가 이 근사된 반응 연산자를 사용할 때 원본 게임의 가치와 근접함을 나타냅니다. 이 주된 결과는 제곱 적분 가능(adapted stochastic processes)한 프로세스 공간 사이의 	extit{universal approximation theorem}을 이용하여 도출되었습니다.

- **Performance Highlights**: 논문에서 제안하는 방법은 스택엘버그 게임의 후행자가 근사된 최적 반응 연산자를 사용할 경우, 원래 스택엘버그 게임의 가치를 잘 근사함을 보여줍니다. 이는 후행자의 최적 반응 계산이 효과적으로 이루어질 수 있는 가능성을 제공하여, 게임 이론과 기계 학습(ML) 분야의 융합 가능성을 제시합니다.



### Counterfactual Uncertainty Quantification of Factual Estimand of Efficacy from Before-and-After Treatment Repeated Measures Randomized Controlled Trials (https://arxiv.org/abs/2411.09635)
Comments:
          39 pages, 7 figures

- **What's New**: 이 논문에서는 새로운 치료법 Rx를 통제군 C와 비교하기 위한 이상적인 추정량이 counterfactual efficacy Rx:C로 정의되며, 이는 모든 환자가 두 가지 치료를 받았을 때의 기대 효과를 나타낸다고 설명합니다. 또한, counterfactual uncertainty quantification (CUQ)을 통해 사실적 Randomized Controlled Trials (RCTs)에서의 포인트 추정에 대한 불확실성을 정량화하는 것이 가능하다는 점을 강조합니다.

- **Technical Details**: 이 연구는 ETZ라는 새로운 통계적 모델링 원칙을 제안하며, 이를 통해 환자의 결과를 추적하고 변화 측정을 위한 변동성 구성 요소를 정량화합니다. BAtRM (Before-and-After treatment Repeated Measures) RCTs를 사용하는 경우, 이 모델링 원칙은 포인트 추정의 불확실성을 줄이고, 평균 치료 효과에 대한 bias-free한 counterfactual point estimation을 보장합니다.

- **Performance Highlights**: 전통 의학에서는 counterfactual point estimation이 항상 unbiased하다는 것이 입증되었으며, 대상 치료법에서도 평균적인 치료 효과에 대해 unbiased한 추정값을 제공합니다. 하지만, 생체 표지자 측정에 오류가 있는 경우 서브그룹 내 치료 효과 예측에 bias가 생길 수 있음을 경고합니다.



### Local deployment of large-scale music AI models on commodity hardwar (https://arxiv.org/abs/2411.09625)
Comments:
          2 pages

- **What's New**: MIDInfinite라는 웹 애플리케이션을 소개합니다. 이 애플리케이션은 일반 하드웨어에서 대규모 생성 AI 모델을 사용하여 상징적 음악을 생성할 수 있습니다. 또한, Anticipatory Music Transformer를 MLC 프레임워크로 포팅하여 다양한 런타임에서 추론할 수 있도록 하고 있습니다.

- **Technical Details**: 이 연구에서는 Lakh MIDI 데이터셋으로 사전 훈련된 Anticipatory Music Transformer 모델을 선택하여 MLC-LLM 프레임워크로 포팅합니다. 모델 구조는 GPT-2의 아키텍처를 기반으로 하며, MIDI 노트는 시작 시간, 노트 지속 시간 및 악기 및 음높이를 나타내는 세 개의 토큰으로 인코딩됩니다. MLC는 WebGPU와 Metal을 활용하여 브라우저 및 Apple 기기에서 효율적으로 실행됩니다.

- **Performance Highlights**: 프로토타입 웹 애플리케이션에서 M3 Macbook Pro를 사용하여 초당 51노트를 생성할 수 있으며, 이는 실시간 재생 속도보다 빠르며, 2초의 버퍼링을 추가하면 86.3%로 증가합니다. 이 프로토타입은 무한 스트림의 다중 악기 MIDI를 생성할 수 있습니다.



### MICCAI-CDMRI 2023 QuantConn Challenge Findings on Achieving Robust Quantitative Connectivity through Harmonized Preprocessing of Diffusion MRI (https://arxiv.org/abs/2411.09618)
Comments:
          Accepted for publication at the Journal of Machine Learning for Biomedical Imaging (MELBA) this https URL

- **What's New**: 이번 연구는 DW-MRI 데이터의 전처리를 통해 수집 방식의 차이를 최소화하고 생물학적 변동은 유지하는 방법을 제시하고자 하는 QuantConn 챌린지를 다루고 있습니다. 또한, MUSHAC 및 SuperMUDI 챌린지보다 10배 많은 샘플을 제공하여 이전 연구들에 비해 더 포괄적인 접근을 시도합니다.

- **Technical Details**: 채널의 유사성을 높이기 위해 연구자는 두 가지 서로 다른 프로토콜, A와 B로 스캔한 103명의 환자 데이터를 사용했습니다. 참가자들은 각자의 하모나이제이션(harmonization) 방법을 적용하여 데이터의 일관성을 높였습니다. 평가에서는 bundle-wise 마이크로스트럭처(microstructure) 측정, bundle 형상(feature), connectomics 등을 포함한 다양한 특징들이 검토되었습니다.

- **Performance Highlights**: 할당된 미세구조 분석 및 연결망(connectomelike) 분석에서의 정확도를 기반으로, 연구 결과는 DW-MRI 처리에서의 acquisition 차이에 따른 편향이 bundle surface area, fractional anisotropy, connectome assortativity 등 여러 지표에서 나타남을 발견했습니다. 특히, 기계 학습(voxel-wise correction), RISH 매핑, NeSH 방법이 이러한 편향을 효과적으로 줄이는 것으로 나타났습니다.



### The Moral Foundations Weibo Corpus (https://arxiv.org/abs/2411.09612)
- **What's New**: 이번 논문에서는 도덕적 감정을 측정하기 위한 새로운 데이터셋인 Moral Foundation Weibo Corpus를 소개합니다. 이 데이터셋은 25,671개의 중국어 댓글로 구성되어 있으며, six diverse topic areas를 포함합니다.

- **Technical Details**: 해당 댓글들은 도덕 이론에 기초하여 정의된 ten moral categories에 따라 최소 세 명의 체계적으로 훈련된 주석자들에 의해 수동으로 주석이 달렸습니다. 주석자 신뢰성을 평가하기 위해 kappa test 결과가 제시됩니다.

- **Performance Highlights**: 최근의 여러 대형 언어 모델을 활용하여 수동 주석을 보완하고, 도덕적 감정 분류를 위한 성능 비교 실험을 수행하여 baseline 결과를 보고하였습니다.



### Latency Optimization in LEO Satellite Communications with Hybrid Beam Pattern and Interference Contro (https://arxiv.org/abs/2411.09600)
- **What's New**: 본 논문은 빠르게 발전하는 저궤도(LEO) 위성 통신 시스템의 비행 일정(beam scheduling) 및 자원 할당(resource allocation)을 최적화하기 위한 새로운 프레임워크를 제안합니다. 이를 통해 지상의 비대칭 트래픽 수요를 충족하고, 다운링크(downlink) 서비스의 품질을 개선하며 전송 지연을 최소화합니다.

- **Technical Details**: 하이브리드 빔 패턴(hybrid beam pattern)을 활용하여 다중 빔 LEO 시스템의 성능을 최적화하며, 동적 동주 채널 간섭 제어(dynamic co-channel interference control) 메커니즘을 개발했습니다. 이 연구는 사용자-빔-주파수(user-beam-frequency) 할당 문제를 혼합 정수 동적 프로그래밍(mixed-integer dynamic programming) 모델로 수립하고, 낮은 복잡도를 가진 신경망 기반 그래프 생성 알고리즘을 사용해 해결합니다.

- **Performance Highlights**: 제안된 방법은 전체 주파수 재사용(full frequency reuse) 및 단일 채널 전송(single-channel transmission)과 같은 기존 방법들에 비해 성능이 우수함을 시뮬레이션 결과를 통해 입증했습니다. 또한 다중 사용자 전송(multi-user transmissions)에서 성능을 더욱 개선할 가능성을 강조합니다.



### Adaptive Deviation Learning for Visual Anomaly Detection with Data Contamination (https://arxiv.org/abs/2411.09558)
Comments:
          Accepted to IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 2025)

- **What's New**: 시각적 이상 감지(Visual Anomaly Detection)를 위해 오염된 데이터에서도 효과적으로 작동할 수 있도록 설계된 새로운 적응형 편차 학습(Adaptive Deviation Learning) 방법을 소개합니다. 이 방법은 각 데이터 인스턴스의 중요도를 동적으로 조정하여 이상 점수(anomaly score)를 학습합니다.

- **Technical Details**: 이 방법은 편차 학습(deviation learning)을 기반으로 하며, 각 인스턴스에 대해 상대적 중요도를 부여하여 이상 점수를 계산하는 체계적인 접근 방식을 채택합니다. 또한 제약 최적화(constrained optimization) 문제를 내포하여 매 미니 배치(mini-batch)마다 인스턴스의 가중치를 업데이트 합니다.

- **Performance Highlights**: MVTec와 VisA 벤치마크 데이터셋에서 수행한 포괄적인 실험 결과, 제안된 방법이 다른 경쟁 기술들을 초과 달성하며 데이터 오염이 있는 환경에서도 안정성과 강 robustness을 보여주었습니다.



### Equation-informed data-driven identification of flow budgets and dynamics (https://arxiv.org/abs/2411.09545)
- **What's New**: 본 연구에서는 컴퓨팅 유체역학(Computational Fluid Dynamics, CFD)에서 플로우 클러스터링(flow clustering)을 위한 새로운 하이브리드 접근 방식을 제안합니다. 이 방법은 각 샘플 포인트의 지역 역학을 나타내는 방정식 기반의 특징을 사용하여 플로우의 동역학을 자동으로 식별하고 클러스터링합니다.

- **Technical Details**: 제안된 접근 방식은 Sparse Identification of Nonlinear Dynamical systems (SINDy) 방법을 사용하여 시간 진화를 기반으로 한 데이터에 대해 지점별로 방정식 기반 클러스터링(Girvan-Newman 알고리즘)을 수행합니다. 이 알고리즘은 Eulerian 및 Lagrangian 프레임워크에서 구현되어 각 샘플 포인트의 궤적에 대한 동적 클러스터링을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, 원통 주변 유동에 대한 연구에서 클러스터의 동적 진화가 성공적으로 나타났으며, 불규칙 플로우 데이터에서도 두 개의 뚜렷한 클러스터가 식별되었습니다. 이는 유동의 상이한 동적 특성을 효과적으로 탐지할 수 있음을 보여줍니다.



### Prompting the Unseen: Detecting Hidden Backdoors in Black-Box Models (https://arxiv.org/abs/2411.09540)
- **What's New**: 이 연구에서는 black-box 모델 수준에서 백도어(Backdoor) 탐지를 위한 새로운 방법론인 BProm을 제안합니다. 이 방법은 시각적 프롬프트(Visual Prompting, VP)를 활용하여 원본 도메인의 훈련된 모델을 사용하여 타겟 도메인 작업에 적응시키는 기술을 기반으로 합니다.

- **Technical Details**: BProm은 VP를 사용하여 의심스러운 모델에 클린(cleam) 데이터셋을 적용합니다. 모델의 저조한 분류 정확도를 사용하여 백도어가 존재함을 식별합니다. 이 연구에서는 class subspace inconsistency라는 개념을 명확히 하여 감염된 모델과 클린 데이터셋 간의 불일치를 확인합니다.

- **Performance Highlights**: BProm의 성능 검증을 위한 광범위한 실험이 수행되었으며, 이는 의심스러운 모델에서의 백도어 탐지 효과를 입증합니다. 실험 결과, BProm은 백도어가 있는 모델에서 낮은 분류 정확도를 보였음을 나타냈습니다.



### A Practical Guide to Fine-tuning Language Models with Limited Data (https://arxiv.org/abs/2411.09539)
- **What's New**: 이 논문은 제한된 데이터로 LLMs(대규모 언어 모델)를 학습하는 최신 전이 학습(transfer learning) 접근 방식을 조사하였습니다. 데이터가 부족한 상황에서 최적의 모델 성능을 달성하기 위한 실용적인 지침을 제공합니다.

- **Technical Details**: 저자들은 초기 및 지속적인 사전 학습(pre-training) 전략을 다루며, 제한된 데이터 동안의 파인 튜닝(fine-tuning) 및 몇 가지 샷 학습(few-shot learning) 중 유용성 극대화 방법을 제시합니다. 또한, 특정 작업 관점에서 데이터 부족 문제를 해결하기 위한 모델과 방법들을 검토합니다.

- **Performance Highlights**: 이 연구는 NLP 분야의 연구자와 실무자에게 데이터가 제한된 시나리오에서 모델 성능을 최적화하기 위한 현재 최첨단 방법에 대한 종합적인 개요를 제공합니다. 주요 전략으로는 사전 학습 및 후속 파인 튜닝 과정에서의 데이터 효율적인 활용에 중점을 두어 고안되었습니다.



### Randomized Truthful Auctions with Learning Agents (https://arxiv.org/abs/2411.09517)
- **What's New**: 이번 논문에서는 반복 경매에서 요령 없는 학습(learning) 알고리즘을 사용하는 에이전트들의 행동을 연구합니다. 이전 연구에서 드러난 바와 같이(노래 투표 경매의 매커니즘과 관련이 있음), 경매의 참가자들이 반가격 경매(second-price auctions)에 참여할 때, 그들의 학습이 진정한 가치에 수렴하지 않을 수 있음을 보여줍니다.

- **Technical Details**: 연구에서는 두 명의 참가자가 그들의 가치에 대한 추정을 학습하는 반복적인 환경을 고려합니다. 에이전트들은 평균 기반의 경매 최적화 알고리즘을 사용하며, 경매 진행 중에 진정한 가치를 반영할 수 있는 여러 메커니즘을 분석합니다. 또한 반드시 진정한 가치에 수렴하는 Strictly-IC(완전 진실성) 경매와 결정적인 구조가 적은 일반 경매 간의 차이를 설명합니다.

- **Performance Highlights**: 연구 결과, 랜덤 경매가 수익을 극대화하는 데 있어 우수한 성능을 발휘할 수 있음을 발견했습니다. 반복적이며 안정적인 가치 분포와 랜덤화된 경매 메커니즘을 활용하여, 참가자들이 진정한 가치에 수렴할 수 있도록 하는 메커니즘의 효과를 확인하였습니다. 이 논문은 AIS(Auction in Society)와 같은 분야에서 랜덤화된 메커니즘의 중요성을 강조하고 있습니다.



### GAN-Based Architecture for Low-dose Computed Tomography Imaging Denoising (https://arxiv.org/abs/2411.09512)
- **What's New**: 이 리뷰 논문은 저선량 컴퓨터 단층 촬영(LDCT) 이미징에서 생성적 적대 신경망(GANs)의 발전을 다루고 있습니다. LDCT의 복잡한 이미지 품질과 방사선 노출 문제를 해결하기 위한 최신 GAN 기반 노이즈 제거 기법의 발전을 종합적으로 분석합니다.

- **Technical Details**: 리뷰에서는 기본 아키텍처부터 최신 모델에 이르기까지 GAN 기반 LDCT 노이즈 제거 기법의 진화를 다룹니다. 특히 해부학적 프라이어(anatomical priors), 지각 손실 함수(perceptual loss functions), 혁신적인 규제 전략(regularization strategies)등이 포함된 최첨단 모델에 대해 설명합니다. 각 GAN 아키텍처(cGANs, CycleGANs, SRGANs)의 강점과 한계를 분석합니다.

- **Performance Highlights**: 논문은 PSNR, SSIM, LPIPS와 같은 메트릭을 이용하여 벤치마크 및 임상 데이터셋에서 성능 향상을 정성적 및 정량적으로 보여줍니다. GAN 모델들이 중대한 성과를 보였음에도 불구하고, 해석 가능성, 합성 아티팩트(synthetic artifacts), 임상 관련 메트릭의 필요성 등이 널리 사용되지 않는 도전 과제가 논의됩니다.



### Sparse Bayesian Generative Modeling for Compressive Sensing (https://arxiv.org/abs/2411.09483)
- **What's New**: 본 연구에서는 압축 센싱(compressive sensing, CS)의 기본적인 선형 역문제를 다루며, 새 유형의 정규화 생성 사전(generative prior)을 도입합니다. 이 방법은 고전적인 사전 기반 CS와 희소 베이즈 학습(sparse Bayesian learning, SBL)의 아이디어를 활용하여 희소 솔루션을 위한 강력한 정규화를 통합합니다. 또한 조건부 가우스성을 활용하여 생성 모델의 적응성을 훈련 데이터에 포함시킵니다.

- **Technical Details**: 제안된 모델은 훈련 데이터 없이도 몇 개의 압축되고 noisy 데이터 샘플만으로 학습할 수 있습니다. 이는 전통적인 CS 알고리즘과 현대의 생성 모델에 비해 데이터 학습 방식에서 차별화됩니다. 또한, 압축 가능한 모든 유형의 신호에 적용할 수 있는 특성을 가지고 있으며, 불확실성 추정을 위한 공액 사전을 매개변수화합니다. 변분 추론(variational inference)의 개념을 통해 이론적으로 지지되며, 다양한 압축 가능한 신호를 사용하여 실증적으로 검증됩니다.

- **Performance Highlights**: 제안된 모델은 전통적인 CS 알고리즘과 현대 신경망 기반 접근 방식에 비해 데이터로부터 학습하는 능력을 보유하고 있습니다. 특히, 손상된 데이터를 통해 학습할 수 있는 강력한 잠재력을 가지며, 다양한 실제 응용 분야에서의 활용 가능성을 갖고 있습니다.



### Graph Neural Networks and Differential Equations: A hybrid approach for data assimilation of fluid flows (https://arxiv.org/abs/2411.09476)
- **What's New**: 본 연구는 유체 역학 응용 프로그램 전반에 걸쳐 평균 흐름 재구성을 향상시키기 위해 Graph Neural Networks (GNNs)와 Reynolds-averaged Navier Stokes (RANS) 방정식을 결합한 새로운 하이브리드 접근 방식을 제안합니다.

- **Technical Details**: 제안된 방법은 GNN 학습 과정에서 RANS로부터 유도된 기울기를 최적화 항으로 사용하기 위해 adjoint method를 활용하여 물리적으로 일관성 있는 모델을 유지하고 있습니다. GNN은 복잡한 기하학을 자연스럽게 처리할 수 있으며, 비정형 데이터를 다루는 데 특화되어 있습니다. 이 연구는 다양한 CFD 시나리오에서 검증되었습니다.

- **Performance Highlights**: 결과적으로, 제한된 양의 데이터만을 사용했음에도 불구하고 재구성한 평균 흐름의 정확성이 순수 데이터 기반 모델에 비해 현저히 개선되었습니다. 이 접근법은 데이터가 부족한 유체 역학 응용에 특히 유용합니다.



### Renal Cell Carcinoma subtyping: learning from multi-resolution localization (https://arxiv.org/abs/2411.09471)
- **What's New**: 이 연구에서는 Renal Cell Carcinoma의 하위 유형 분류를 위한 새로운 self-supervised training 전략을 제시합니다. 이 방법은 주석이 달린 데이터셋의 필요성을 줄이고, 정확도를 크게 저하하지 않으며, 병리학자의 의사결정 과정을 모방하여 다양한 배율에서 학습된 특성을 통합합니다.

- **Technical Details**: 이 연구의 접근법은 Whole Slide Images (WSIs)를 기반으로 하며, self-supervised learning (SSL) 기법을 활용하여 암 유형을 분류합니다. SSL은 데이터 자체에서 얻은 라벨을 사용하여 semi-automatic 과정으로 supervised pretext task를 수행하며, 복잡한 주석을 요구하지 않습니다.

- **Performance Highlights**: 본 연구에서는 제안된 방법이 ccRCC, pRCC, chRCC 및 ONCO 등 4개의 주요 RCC 하위 유형을 분류하는 데 성공적이며, 기존의 fully supervised 방법들과 유사한 성능을 달성합니다. 이는 대규모 주석 데이터 필요성을 줄이는 동시에 정확도를 유지할 수 있음을 보여줍니다.



### Long-Tailed Object Detection Pre-training: Dynamic Rebalancing Contrastive Learning with Dual Reconstruction (https://arxiv.org/abs/2411.09453)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 이 논문에서는 기존의 long-tailed distributions에 대한 문제를 해결하기 위한 새로운 pre-training (사전 훈련) 프레임워크인 Dynamic Rebalancing Contrastive Learning with Dual Reconstruction (2DRCL)을 제안합니다. 이 방법은 object detection (객체 탐지)과 일치하는 pre-training을 위한 Holistic-Local Contrastive Learning 메커니즘을 기반으로 합니다.

- **Technical Details**: 2DRCL은 global contextual semantics (전반적인 문맥 의미)와 detailed local patterns (세부적인 지역 패턴)을 포착하며, underrepresented instances (저대표 인스턴스)의 샘플링을 조정하는 동적 재균형 전략을 설계하여 long-tailed data의 불균형 문제를 해결합니다. Dual Reconstruction은 self-consistency principle (자기 일관성 원리)에 따라 재구성 작업을 강제하여 simplicity bias (단순성 편향) 문제를 해결합니다.

- **Performance Highlights**: COCO와 LVIS v1.0 데이터셋에서 실험을 통해 제안된 2DRCL 방법이 tail classes (꼬리 클래스)의 mAP/AP 점수를 특히 향상시키는 데 효과적임을 입증하였습니다. 2DRCL은 기존의 Focal Loss 및 ECM Loss 방법보다 높은 전반적인 평균 정확도(APb)인 26.4%를 달성하였고 rare classes (희귀 클래스)에서도 최고의 성능을 보였습니다.



### DiffRoad: Realistic and Diverse Road Scenario Generation for Autonomous Vehicle Testing (https://arxiv.org/abs/2411.09451)
Comments:
          14 pages, 9 figures

- **What's New**: DiffRoad라는 새로운 diffusion model을 통해 고품질의 3D 도로 시나리오를 생성하는 방법을 제안합니다. 이 모델은 도로 환경의 복잡성과 다양성을 포착하여 자율 주행 차량의 테스트에 적합한 현실적이고 다양한 도로 시나리오를 자동으로 생성할 수 있습니다.

- **Technical Details**: DiffRoad는 흰 잡음으로부터 도로 레이아웃을 합성하는 역 노이즈 제거 과정을 통해 도로 시나리오를 생성하고, Road-UNet 아키텍처를 사용하여 생성된 도로의 질을 최적화합니다. 또한, 도로의 연속성과 합리성을 평가하기 위한 도로 시나리오 평가 모듈을 도입하여, OpenDRIVE 형식으로 자동 변환하여 자율 주행 시뮬레이션 테스트에 활용할 수 있습니다.

- **Performance Highlights**: DiffRoad는 실제 데이터셋에 대한 실험을 통해 고해상도 도로 시나리오를 생성하며, 원래 분포를 유지하면서 원활한 도로 구조를 구현하는 능력을 입증하였습니다. 이로 인해 자율 주행 차량 테스트를 위한 풍부하고 다양한 시나리오 라이브러리를 제공하며, 미래 인프라 설계에 대한 통찰력을 제시합니다.



### Learning efficient and provably convergent splitting methods (https://arxiv.org/abs/2411.09444)
- **What's New**: 이 논문에서는 초기값 문제(initial value problem, IVP)를 해결하기 위한 기계 학습(machine learning) 기반 분할 방법(splitting methods)을 제안합니다. 이는 전통적인 수치 방법의 한계를 완화하고, 대규모 타임스탬프에서 최적의 정확성을 달성할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 첫째, 이 방법은 넓은 범위의 물리적 응용에서 발생하는 1차 초기값 문제에 중점을 둡니다. 분할 및 조합(composition) 방법을 통해 더 간단한 IVP로 분리하고 해결할 수 있습니다. 고전적인 방법은 안정성을 보장하였지만 큰 타임스탬프에서는 최적의 성능을 발휘하지 못하는 한계가 있습니다. 제안된 기계 학습 방법은 상대적으로 큰 타임스탬프에서 효율적으로 적합하며 수렴성과 보존 보장을 제공합니다.

- **Performance Highlights**: 제안된 학습된 방법은 Schrodinger 방정식에 대해 제공되는 제한된 계산 예산 내에서 전통적인 방법보다 훨씬 더 효율적임을 수치적으로 입증했습니다. 이는 대규모 타임스탬프에서 우수한 성능을 나타내며, 특정한 계산 예산 아래에서 전통적인 방법보다 적은 오류로 결과를 얻을 수 있는 잠재력이 있음을 시사합니다.



### Mediffusion: Joint Diffusion for Self-Explainable Semi-Supervised Classification and Medical Image Generation (https://arxiv.org/abs/2411.09434)
- **What's New**: Mediffusion은 반지도 학습(semi-supervised learning) 및 설명 가능한 분류를 위한 새로운 방법론으로, 조인트 확산 모델(joint diffusion model)을 기반으로 합니다. 의료 이미징 도메인에서의 데이터 라벨링 부족 문제를 해결하고, 높은 성능과 신뢰성을 요구하는 응용 프로그램에 적합한 솔루션을 제공합니다.

- **Technical Details**: Mediffusion은 Denoising Diffusion Probabilistic Models (DDPM)를 사용하여, UNet 구조 내에서 생성(generative) 및 분류(discriminative) 작업을 위해 공유되는 매개변수화(parametrization)를 사용합니다. 이 모델은 라벨된 데이터와 라벨이 없는 데이터를 모두 통해 효과적으로 학습하며, 카운터팩추얼 예제를 이용하여 정확한 설명을 제공합니다.

- **Performance Highlights**: Mediffusion은 최근의 반지도 학습 기술들과 비교할 만한 성능을 달성하면서도, 보다 신뢰할 수 있는 정확한 설명을 제공합니다. 실험 결과, Mediffusion은 설명 가능성과 합성 데이터 샘플링 가능성을 함께 제공하는 최첨단 방법임을 입증했습니다.



### SAG-ViT: A Scale-Aware, High-Fidelity Patching Approach with Graph Attention for Vision Transformers (https://arxiv.org/abs/2411.09420)
Comments:
          10 pages, 4 figures, 3 tables

- **What's New**: 본 논문에서 소개하는 Scale-Aware Graph Attention Vision Transformer (SAG-ViT)는 멀티스케일 특징 표현을 효율적으로 통합하여 image classification을 향상시키는 새로운 프레임워크입니다. EfficientNet을 백본으로 사용하고, 그래프 구조를 통해 특징 맵을 처리함으로써 고급 의미 정보를 유지하며 고유한 관계를 모델링합니다.

- **Technical Details**: SAG-ViT는 EfficientNet을 기반으로 한 멀티스케일 특징 맵을 추출하여 이를 패치로 나누고, 공간적 및 특징 유사성에 따라 그래프를 구성합니다. 그래프 내의 각 노드는 패치를 나타내며, Graph Attention Network (GAT)를 통해 최신의 패치 정보를 동적으로 강조합니다. 이후 Transformer 인코더가 장기 종속성과 복잡한 상호작용을 포착합니다.

- **Performance Highlights**: SAG-ViT는 여러 벤치마크 데이터셋에서 평가되었으며, 기존의 Transformer 기반 접근 방법과 비교하여 이미지 분류 성능을 높이는 데 성공하였습니다.



### Less is More: Unseen Domain Fake News Detection via Causal Propagation Substructures (https://arxiv.org/abs/2411.09389)
Comments:
          9 pages, 2 figures, 5 tables

- **What's New**: 본 논문에서는 인-distribution(내적 분포) 데이터에서 인과 서브그래프(causal subgraphs)를 추출하여 zero-shot(제로샷) 가짜 뉴스 탐지를 개선하기 위한 Causal Subgraph-oriented Domain Adaptive Fake News Detection (CSDA) 모델을 제안합니다.

- **Technical Details**: CSDA 모델은 그래프 신경망(graph neural network)을 기반으로 한 마스크 생성 과정을 통해 전파 그래프(propagation graph) 내의 주요 노드와 엣지를 식별하고, 이를 가짜 뉴스 탐지에 활용합니다. 모델은 이진 마스크를 사용하여 각 노드와 엣지를 인과적 요소(causal elements) 또는 편향 요소(biased elements)로 분류합니다. 또한, 제한된 OOD 데이터에서의 few-shot(소수 샷) 상황에서 대조 학습(contrastive learning)을 통해 성능을 개선합니다.

- **Performance Highlights**: CSDA는 공개 소셜 미디어 데이터셋에서 OOD 가짜 뉴스 탐지를 수행하며, 다른 최신 모델들 대비 7%에서 16%의 정확도 향상을 달성하였습니다.



### Are nuclear masks all you need for improved out-of-domain generalisation? A closer look at cancer classification in histopathology (https://arxiv.org/abs/2411.09373)
Comments:
          Poster at NeurIPS 2024

- **What's New**: 이 논문은 암 검출을 위한 도메인 일반화(domain generalisation)에서의 혁신적인 접근법을 제시하고 있습니다. 특히, 세포 핵(nuclei)에 초점을 맞추어 OOD(out-of-domain) 일반화를 개선하는 방법을 모색하고 있습니다.

- **Technical Details**: 본 연구에서는 CNN(Convolutional Neural Networks)을 활용하여 핵의 형태와 조직에 중점을 두고 학습을 진행합니다. 핵 분할(segmentation) 마스크를 원본 이미지와 통합하여 모델이 세포 핵의 공간적 배열을 우선시하도록 유도합니다. 추가적인 손실 항을 도입하여 형태 기반 특징을 강조하는 방식으로 진행됩니다.

- **Performance Highlights**: 세 가지 데이터셋에 대한 실험 결과, 제안된 방법은 기존 방법들보다 OOD 일반화를 현저히 개선하였으며, 이미지 손상 및 적대적 공격에 대한 강건성도 증가했습니다. 모델의 성능 향상은 모든 비교 기반 방법에 비해 우수한 결과를 보여주었습니다.



### Time-to-Event Pretraining for 3D Medical Imaging (https://arxiv.org/abs/2411.09361)
Comments:
          34 pages, 19 figures

- **What's New**: 본 연구는 3D 의료 이미징 모델을 위한 새로운 사전 훈련 접근법인 'time-to-event pretraining'을 제안하며, 대규모 전자 건강 기록(EHR) 데이터에서 시간에 따라 발생하는 이벤트 정보를 활용하여 기존의 잃어버린 맥락 문제를 해결합니다.

- **Technical Details**: 이 방법은 18,945개의 흉부 CT 스캔을 포함하는 데이터셋을 사용하여 EHR 데이터로부터 유도된 TTE 분포를 통해 사전 훈련을 수행합니다. 이 과정을 통해 8개의 벤치마크 작업에서 AUROC가 평균 23.7% 증가하고 Harrell의 C-index가 29.4% 향상되었습니다.

- **Performance Highlights**: 기존 진단 분류 성능을 저하시키지 않고도 발생 예측 성능을 크게 향상시키며, 모델 교차 범위(Integrated Brier Score)의 정확도를 평균 54% 개선했습니다. 모든 실험은 공개된 의료 데이터셋을 사용하여 재현 가능성을 보장하였고, 관련 코드를 GitHub에 공개했습니다.



### Improving hp-Variational Physics-Informed Neural Networks for Steady-State Convection-Dominated Problems (https://arxiv.org/abs/2411.09329)
Comments:
          25 pages, 11 figures, 8 tables

- **What's New**: 이 논문은 FastVPINNs 프레임워크를 이용하여 대류 우세 대류-확산-반응(convection-diffusion-reaction, CDR) 문제에 대한 두 가지 확장을 제안하고 연구합니다. 첫째, 손실 함수(loss functional)에 SUPG 안정화(stabilization)와 유사한 항을 포함시키고 공간적으로 변동하는 안정화 매개변수를 예측하는 네트워크 아키텍처를 제안합니다. 둘째, 강제 제약 Dirichlet 경계 조건에서 지시 함수(indicator function)의 선택이 계산된 솔루션의 정확도에 미치는 영향을 관찰한 결과, 좋은 매개변수를 학습하는 네트워크 아키텍처를 제안합니다.

- **Technical Details**: 제안된 FastVPINNs 프레임워크는 대류-diffusion-reaction 문제를 위해 두 가지 방법론을 통합합니다. 첫째, SUPG 안정화를 포함한 손실 함수를 사용하여 모델의 정확도를 높입니다. 둘째, 다양한 지시 함수에 대한 적절한 매개변수를 학습하는 네트워크 아키텍처를 통해 계산의 정확도를 개선하고 있습니다. 이 연구는 수치 연구를 통해 두 가지 제안이 문헌에서 찾아볼 수 있는 방법들보다 더 정확한 결과를 이끌어냄을 보여주었습니다.

- **Performance Highlights**: 수치 연구 결과, 제안된 두 가지 방법론이 기존의 접근 방식들에 비해 눈에 띄게 더 정확한 결과를 제공함을 확인했습니다. 이는 특히 매개변수의 선택과 손실 함수의 설계가 모델 성능에 미치는 중요성을 강조합니다.



### Enhancing generalization in high energy physics using white-box adversarial attacks (https://arxiv.org/abs/2411.09296)
Comments:
          10 pages, 4 figures, 8 tables, 3 algorithms, to be published in Physical Review D (PRD), presented at the ML4Jets 2024 conference

- **What's New**: 본 연구는 입자 물리학에서 파생된 신경망 기반의 감독 학습(supervised learning) 모델의 일반화(generalization) 성능 향상을 위해 adversarial 공격(adversarial attacks) 기법을 도입하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 이 논문에서는 전문가적 공격을 활용하여 Higgs 보존(decay signal) 분류의 성능을 평가하고, 이 과정에서 weight space 공격과 feature space 공격으로 구분된 4가지 유형의 white-box adversarial 공격을 다룹니다. 또한, gradient ascent 및 reduced Hessian eigenvalue 분석을 통해 다양한 지역 최소(local minima)의 sharpness를 정량화합니다.

- **Performance Highlights**: 연구 결과, white-box adversarial 공격이 모델의 일반화 성능을 유의미하게 개선하는 것으로 나타났지만, 계산 복잡성이 증가했습니다. 이는 adversarial 훈련(adversarial training)이 자연 정확도(natural accuracy)를 감소시키면서도 모델의 강건성(robustness)을 높이는 점을 강조합니다.



### How Good is ChatGPT at Audiovisual Deepfake Detection: A Comparative Study of ChatGPT, AI Models and Human Perception (https://arxiv.org/abs/2411.09266)
- **What's New**: 이 연구에서는 대규모 언어 모델인 ChatGPT의 오디오 및 비디오 콘텐츠에서 딥페이크를 탐지하는 능력을 평가합니다. 기존의 비디오 조작 탐지 방법들과 ChatGPT의 탐지 성능을 비교하고, Prompt Engineering(프롬프트 엔지니어링)의 역할을 강조합니다.

- **Technical Details**: 이 연구는 딥페이크 탐지를 위해 LLMs(대규모 언어 모델)를 활용하는 방법을 제안합니다. 실험은 벤치마크 멀티모달 딥페이크 데이터셋의 비디오를 사용하여 수행하였으며, ChatGPT는 다양한 오디오 및 비주얼 아티팩트(visual and auditory artifacts)를 분석하는 데 적용되었습니다. 제안된 방법은 입력 비디오에 대한 시각적, 청각적 분석을 포함하여 깊이 있는 분석을 제공합니다.

- **Performance Highlights**: 실험 결과 ChatGPT가 멀티모달 딥페이크 탐지에서 인간 및 최신 AI 모델들과 비교하여 경쟁력 있는 성능을 보였으며, 특히 프롬프트 설정에 따라 탐지 성능이 달라지는 것으로 나타났습니다. 그러나 ChatGPT는 탐지 과정의 해석 가능성(interpretability) 부족과 특정 조작에 대한 일반화의 한계를 갖고 있습니다.



### Classical Verification of Quantum Learning Advantages with Noises (https://arxiv.org/abs/2411.09210)
Comments:
          13 pages 1 figure

- **What's New**: 본 논문에서는 기존의 양자(quantum) 학습 검증(classical verification of quantum learning) 방법들이 노이즈(noise)가 많아진 현재의 양자 장비에 적용될 수 있는지를 탐구합니다. 저자들은 상수 수준의 노이즈가 있는 양자 푸리에 샘플링 회로를 기반으로 노이즈가 없는 결과를 재구축할 수 있는 효율적인 고전적 오차 수정 알고리즘(classical error rectification algorithm)을 제안합니다.

- **Technical Details**: 이 알고리즘은 문제 규모에 따라 로그(logarithmically)로 스케일되는 소수의 노이즈 샘플을 사용하여 강력한 푸리에 계수(heavy Fourier coefficients)를 복원할 수 있음을 증명합니다. 이 알고리즘은 균등 입력 마진(uniform input marginal)으로 구성된 무관심 패리티 학습(agnostic parity learning) 작업에 적용하여 노이즈가 가득한 양자 장비에서도 효율적으로 작업을 수행할 수 있음을 입증합니다.

- **Performance Highlights**: 우리는 고전 클라이언트(classical client)가 무작위 예제 오라클(random example oracle)에 접근하여 노이즈가 있는 양자 개인자(quantum prover)로부터 무관심 패리티 학습 결과를 효율적으로 검증할 수 있음을 증명했습니다. 이 결과는 노이즈가 있는 현재의 중간 규모 양자 장비를 사용한 이론적 연구와 실제 응용에 대해 귀중한 지침을 제공하며, 노이즈에도 불구하고 양자 학습의 이점을 고전적으로 검증할 수 있는 가능성을 보여줍니다.



### DeBaTeR: Denoising Bipartite Temporal Graph for Recommendation (https://arxiv.org/abs/2411.09181)
- **What's New**: 이 논문에서는 명시적 사용자 피드백의 부족으로 인해 대체 데이터 출처로서의 암시적 피드백의 문제를 다루고, 시간 정보를 활용하여 추천 시스템의 노이즈를 제거하고 예측 성능을 향상시키는 두 가지 방법인 DeBaTeR-A와 DeBaTeR-L을 제안합니다.

- **Technical Details**: 사용자-아이템 상호작용은 이분 그래프(bipartite graph)로 모델링되며, 시간 정보를 활용하여 사용자의 특성과 아이템의 특성을 고려한 시간을 인식하는 사용자/아이템 임베딩을 생성합니다. DeBaTeR는 인접 행렬의 가중치를 재조정하거나 손실 함수의 가중치를 재조정하여 노이즈 상호작용을 식별하고 제거합니다.

- **Performance Highlights**: 광범위한 실험을 통해 제안된 방법들이 최신 모델에 비해 우수한 성능과 견고성을 보여주었으며, 실제 데이터셋에서 높은 정확도를 기록하였습니다.



### Hybrid deep additive neural networks (https://arxiv.org/abs/2411.09175)
Comments:
          29 pages, 13 figures

- **What's New**: 본 논문에서는 기존의 다층 퍼셉트론(MLP) 신경망의 한계를 극복하기 위해 새로운 형태의 딥 신경망 구조인 DANN(Deep Additive Neural Network)을 제안합니다. 이 신경망은 비선형 관계를 효과적으로 캡쳐할 수 있도록 설계되었으며, 복잡한 수학적 가정 없이도 다양한 예측 문제에서 성능을 개선할 수 있습니다.

- **Technical Details**: 제안된 DANN은 고정된 활성화 함수와 단순하면서도 유연한 기초 함수(basis function)를 사용하여 계산과 구현이 용이합니다. 이 네트워크는 기존의 전통적인 신경망 구조와의 하이브리드(hybrid) 방식으로 조합되어 오버피팅(overfitting)을 효과적으로 피할 수 있습니다. 또한, 이 네트워크는 모든 연속 함수에 대해 보편적 근사 속성을 갖습니다.

- **Performance Highlights**: 시뮬레이션 연구와 실제 데이터 분석 결과, DANN은 예측 오차(prediction error)와 파라미터 수 측면에서 기존의 전통적인 신경망보다 일반적으로 더 나은 성능을 보였습니다. 이 연구는 신경망의 파라미터 수를 줄이면서도 높은 predictive accuracy를 달성하는 가능성을 제시하고 있습니다.



### Advancing Diffusion Models: Alias-Free Resampling and Enhanced Rotational Equivarianc (https://arxiv.org/abs/2411.09174)
Comments:
          13 pages, 7 figures

- **What's New**: 최근의 이미지 생성 기술, 특히 diffusion models의 발전은 이미지 합성 품질의 인상적인 개선을 가져왔습니다. 본 논문에서는 올바르지 않은 재샘플링 작업이 aliasing을 유발하여 이미지 품질을 저하시킨다고 가정하며, 이 문제를 해결하기 위해 alias-free resampling을 UNet 아키텍처에 통합하는 방법을 제안합니다.

- **Technical Details**: 우리는 alias-free resampling을 통해 diffusion models의 UNet 구조에 새로운 학습 가능한 매개변수를 추가하지 않고 모델 효율성을 유지하며 성능을 향상시키는 것을 목표로 하고 있습니다. 이를 통해 높은 주파수 아티팩트를 방지하며 이미지의 회전 동등성을 향상시킬 수 있습니다. 또한, 사용자 제어에 의한 이미지 회전을 가능하게 하는 수정된 diffusion 프로세스를 제안합니다.

- **Performance Highlights**: 실험 결과, CIFAR-10, MNIST 및 MNIST-M과 같은 벤치마크 데이터셋에서 이미지 품질이 일관되게 향상되었음을 확인했습니다. 특히 FID 및 KID 점수에서 향상이 두드러지며, 이는 더 나은 결과를 위해 이미지 처리 원리에 의해 안내된 적절한 alias-free resampling 레이어의 사용 가능성을 강조합니다.



### Rationality based Innate-Values-driven Reinforcement Learning (https://arxiv.org/abs/2411.09160)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2401.05572

- **What's New**: AI 에이전트의 내재적 동기를 설명하는 탈내재 가치 기반 강화 학습(IVRL) 모델을 제안하고, 이를 통해 인간 사회와 안전하고 조화롭게 통합될 수 있는 AI를 지원하는 방법을 탐구합니다.

- **Technical Details**: 제안된 IVRL 모델은 DQN과 A2C 알고리즘을 기반으로 하며, 역할 수행 게임(RPG) 강화 학습 테스트 플랫폼인 VIZDoom을 사용하여 성능을 검증하고 다른 벤치마크 알고리즘과 비교합니다.

- **Performance Highlights**: IVRL 모델은 다양한 개별적 요구를 합리적으로 조직함으로써 더 나은 성과를 달성할 수 있음을 보여주었습니다. 이 모델은 복잡하고 도전적인 작업에 효율적으로 적응할 수 있는 수렴성을 가집니다.



### FxTS-Net: Fixed-Time Stable Learning Framework for Neural ODEs (https://arxiv.org/abs/2411.09118)
- **What's New**: 이 논문에서는 Neural Ordinary Differential Equations(Neural ODEs)의 안정성을 보장하기 위한 새로운 방법인 FxTS-Net을 제안합니다. FxTS-Net은 사용자 정의된 고정 시간 내에 정확한 예측을 달성하기 위해 설계된 FxTS 손실(FxTS-Loss)을 사용합니다.

- **Technical Details**: FxTS-Net은 Lyapunov 함수에 기반한 FxTS 손실(FxTS-Loss)을 이용하여 고정 시간 안정성(Fixed-time stability)을 보장합니다. 또한, 네트워크 구조와 다양한 작업에 맞춘 Lyapunov 함수를 구성하는 혁신적인 접근 방식을 제시하며 supervised 정보를 활용합니다. 최적화를 위해, perturbation sampling 방법을 통해 FxTS-Loss를 근사화하는 학습 알고리즘을 개발했습니다.

- **Performance Highlights**: 실험 결과, FxTS-Net은 예측 성능이 향상되고 입력 perturbation에 대한 강건성을 개선하였습니다. 이론적 통찰과 일치하게, 고정 시간 안정성이 FxTS-Net의 전반적인 성능에 긍정적인 영향을 미침을 입증하였습니다.



### Drone Detection using Deep Neural Networks Trained on Pure Synthetic Data (https://arxiv.org/abs/2411.09077)
Comments:
          12 pages, 8 figures

- **What's New**: 이 논문은 순수한 합성 데이터셋으로 훈련된 드론 탐지 Faster-RCNN 모델을 제시하며, 이는 실제 데이터로의 전이 가능성을 입증하였습니다. 실험 결과, MAV-Vid 데이터셋에서 97.0%의 AP_50 성능을 달성하였으며, 이는 실제 데이터로 훈련된 모델의 97.8%와 유사합니다.

- **Technical Details**: 모델 훈련에 사용된 합성 데이터셋은 Structured Domain Randomization (SDR) 기법을 이용하여 생성되었습니다. 드론 탐지의 정확도 향상을 위해, 다양한 조명, 텍스쳐 및 자세를 무작위로 변경하여 실제 환경과 유사한 시나리오를 구현하였습니다. 이를 통해 모델이 실제 데이터에서도 성능을 발휘할 수 있도록 하였습니다.

- **Performance Highlights**: 드론 탐지 분야에서 합성 데이터를 활용함으로써, 데이터 수집 비용을 줄이고 레이블 품질을 향상시킬 수 있는 가능성을 보여주었습니다. 이 연구 결과는 앞으로 더 정교한 합성 드론 데이터셋 개발의 초석이 될 수 있습니다. 또한, 안전이 중요한 애플리케이션인 공항 드론 탐지의 데이터셋 생성을 리스크를 줄일 수 있습니다.



### Code-mixed LLM: Improve Large Language Models' Capability to Handle Code-Mixing through Reinforcement Learning from AI Feedback (https://arxiv.org/abs/2411.09073)
Comments:
          initial version: 5 pages, 2 figures

- **What's New**: 본 논문은 코드-믹싱(code-mixing) 및 코드-스위칭(code-switching) 캠에 대한 다국어 대형 언어 모델(LLM)의 성능을 평가하고, AI 피드백을 통한 강화 학습(Reinforcement Learning from AI Feedback, RLAIF)을 통해 모델의 이러한 혼합 언어 이해 능력을 개선하는 방법을 제안합니다.

- **Technical Details**: 코드-믹싱은 두 개 이상의 언어가 혼합된 언어 사용을 의미하며, 이를 효과적으로 처리하기 위한 다국어 LLM의 성능을 벤치마킹합니다. 성능 향상을 위해, 기존의 LLM에 RLAIF 방법론을 적용하여 코드-믹싱 기계 번역 작업을 개선합니다.

- **Performance Highlights**: 실험 결과는 RLAIF 방법이 코드-믹싱을 처리하는 LLM의 성능을 향상시킬 가능성이 있음을 보여주고 있습니다. RLAIF와 같이 AI 피드백을 활용하는 접근 방식이 향후 더 유연하고 인간 중심의 AI 시스템 개발에 기여할 수 있을 것입니다.



### Language-Model Prior Overcomes Cold-Start Items (https://arxiv.org/abs/2411.09065)
Comments:
          This paper is dedicated to cold-start item recommendation using language-model priors

- **What's New**: 이 논문은 추천 시스템에서 아이템의 콜드 스타트 문제를 해결하기 위해 언어 모델(Language Model, LM)을 활용하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 제안된 프레임워크는 Bayesian 정규화를 RecSys의 훈련 과정에 통합하여 아이템 간의 의미적 유사성을 고려합니다. LM이 인코딩한 사전 지식을 활용하여 연속 잠재 공간에서 세밀한 아이템 임베딩을 학습합니다.

- **Performance Highlights**: 실험을 통해 SASRec 및 BPRMF와 같은 추천 시스템에 통합하여 두 개의 실제 데이터 세트에서 평가한 결과, 제안된 접근 방식이 SASRec의 정규화 할인 누적 이익(Normalized Discounted Cumulative Gain)을 17.78% 향상시켰습니다.



### Minimax Optimal Two-Sample Testing under Local Differential Privacy (https://arxiv.org/abs/2411.09064)
Comments:
          59 pages, 5 figures

- **What's New**: 이 논문은 Local Differential Privacy (LDP) 하에서 두 샘플 테스트의 통계적 유용성과 개인 정보 보호 간의 균형을 탐구하는 새로운 방법론을 제안합니다.

- **Technical Details**: 다양한 프라이버시 메커니즘(Laplace, discrete Laplace, Google의 RAPPOR)을 사용하여 사전 처리된 순열 테스트를 통해 다항 분포 및 연속 데이터를 다루며, LDP 하에서의 균일 분리 속도를 분석합니다. 특히, Bonferroni 유형의 적응형 테스트를 제안하여 밀도 테스트의 매끄러움 매개변수가 알려져 있지 않은 경우에도 강력한 성능을 보장합니다.

- **Performance Highlights**: 제안된 테스트는 모든 유한 샘플 크기에 대해 Type I 오류를 엄격히 통제하며, LDP 제약 조건을 준수하고 LDP 하에서의 minimax 분리 속도를 달성하여 개인 정보 보호와 통계적 유용성 간의 내재적인 균형을 성공적으로 증명합니다.



### ClevrSkills: Compositional Language and Visual Reasoning in Robotics (https://arxiv.org/abs/2411.09052)
Comments:
          To appear at NeurIPS 2024 (D&B track)

- **What's New**: ClevrSkills는 로봇 공학에서의 조합적 추론을 위한 새로운 벤치마크 세트로, ManiSkill2 시뮬레이터를 기반으로 하여 33개의 다양한 작업과 데이터 세트를 포함합니다. 이 환경은 로봇의 저수준 능력을 평가하고 이를 통해 복잡한 고수준 작업을 수행하는 능력을 테스트합니다.

- **Technical Details**: ClevrSkills는 기본적인 운동 기술이 필요한 간단한 작업부터 시작하여 점진적으로 복잡한 작업으로 진행되는 커리큘럼을 포함합니다. 환경 내에서 저수준 능력의 조합을 통해 고수준 작업을 수행할 수 있는지를 평가합니다. 이 작업들은 로봇이 물체를 집고, 놓고, 던지고, 누르고 만지는 등 다양한 조작 작업을 포함합니다.

- **Performance Highlights**: 최신 비전 언어 모델(VLM)이 다수의 작업에 대해 사전 훈련된 후에도 로봇 공학 작업에서 조합적 추론에 실패하는 결과를 보여주었습니다. 이는 이 모델들이 고수준의 인간 같은 추론을 수행하는 데 있어 한계를 가짐을 나타냅니다.



### Bridging the Visual Gap: Fine-Tuning Multimodal Models with Knowledge-Adapted Captions (https://arxiv.org/abs/2411.09018)
- **What's New**: 본 연구에서는 비전-언어 모델(vision-language models, VLMs)이 긴 상세한 이미지 캡션에 어떻게 적응하는지를 탐구합니다. 우리는 Decomposed NLI (DNLI)라는 새로운 평가 프레임워크를 제안하여 생성된 캡션을 개별 제안으로 분해하고 각 제안을 독립적으로 평가합니다. 또한, KnowAda라는 데이터 중심의 파인튜닝 기법을 소개하여 VLM이 고유한 지식과 시각적 이해를 활용하여 캡션을 자동으로 조정하도록 합니다.

- **Technical Details**: KnowAda는 세 가지 단계로 구성된 캡션 변환 방법으로, 각 단계에서 VLM의 시각적 지식을 평가하고, 모델이 이해하지 못하는 이미지를 기반으로 한 질문을 생성하여 미지의 내용을 반영하도록 캡션을 수정합니다. DNLI 프레임워크는 캡션의 설명력 및 오류율을 보다 신뢰할 수 있는 방식으로 평가하며, 복잡한 캡션이 모델의 성능에 미치는 영향을 분석합니다.

- **Performance Highlights**: KnowAda는 2억에서 70억 개의 파라미터를 가진 여러 소규모 VLM에서 검증되어, 원본 캡션으로 훈련했을 때보다 환각(hallucinations)을 일관되게 줄이며, 자동 및 인간 평가 모두에서 캡션의 설명력(descriptiveness)과 사실 정확도(factual accuracy) 간의 균형을 잘 제공함을 보여주었습니다.



### Microfoundation Inference for Strategic Prediction (https://arxiv.org/abs/2411.08998)
- **What's New**: 이 논문에서는 performative prediction(퍼포머티브 예측)이라는 현상을 다루며, 예측 모델이 타겟 분포에 미치는 장기적 영향을 학습하는 방법론을 제안합니다. 특히, 비용 조정 유틸리티 극대화 문제로 에이전트의 반응을 모델링하고, 사전-모델 노출(pre-model exposure)과 사후-모델 노출(post-model exposure) 분포를 정렬하는 방법을 다루고 있습니다.

- **Technical Details**: 본 연구는 에이전트 반응 모델의 잘못된 사양 문제를 해결하기 위해, 에이전트 반응 데이터에서 미세 구조(microfoundation) 모델을 학습하는 방식으로 접근합니다. 이러한 방식을 통해 퍼포머티브 리스크를 최소화하는 빠른 최적화 알고리즘을 사용할 수 있습니다. 우리는 임의의 유틸리티 극대화 모델의 비용 함수를 추정하는 방법을 제안하며, 최적 수송(optimal transport)을 활용해 비용을 추정합니다.

- **Performance Highlights**: 본 연구에서는 방법론의 성능을 다양한 하류(multiple downstream) 작업을 통해 입증하며, 비용 추정의 성능을 수치 실험으로 처리합니다. 또한, 제안된 방법론이 이익 함수의 잘못된 사양에 대해 특정 강건성 속성을 만족함을 실증적으로 입증합니다.



### Parameter Inference via Differentiable Diffusion Bridge Importance Sampling (https://arxiv.org/abs/2411.08993)
- **What's New**: 본 연구에서는 고차원 비선형 확산 프로세스에서 매개변수 추론(parameter inference)을 수행하기 위한 새로운 방법론을 소개합니다. 이 방법은 생물 종의 진화 및 종 간의 관계를 이해하는 데 유용하게 적용되어 조상 상태 재구성을 포함합니다.

- **Technical Details**: 이론적으로 스코어 매칭(score matching)을 활용하여 확산 브릿지(diffusion bridges)를 근사하고, 이를 중요 샘플러(importance sampler)에 사용하여 로그 가능도(log-likelihood)를 추정합니다. 이 과정은 전적으로 미분 가능하여, 근사된 로그 가능도에 대해 기울기 상승법(gradient ascent)을 사용할 수 있습니다. 제안된 방법론은 시뮬레이션된 확산 브릿지를 제안으로 사용하여 매개변수 추론과 확산 평균 추정을 허용합니다.

- **Performance Highlights**: 이 프레임워크는 생물학적 2차원 및 3차원 형태 계측 데이터에 성공적으로 적용되었으며, 이론적 안정성과 효율성을 제공합니다. 또한 제안된 방법론은 다변량 가우시안 근사를 통해 수치적 불안정성을 방지하는 여러 기법을 포함하여, 보다 안정적인 매개변수 추론을 가능하게 합니다.



### Non-Euclidean High-Order Smooth Convex Optimization (https://arxiv.org/abs/2411.08987)
- **What's New**: 이번 연구에서는 Hölder 연속적인 q차 미분을 가지는 볼록 함수의 최적화를 위한 알고리즘을 개발하였으며, 이는 q차 오라클 (oracle)을 통해 이루어졌습니다. 새로운 비유클리드 (non-Euclidean) 이차 근사적 접근법을 사용하여 최적화를 진행하였으며, 막연한 균일 볼록 정규화기 (uniformly convex regularizer)를 활용했습니다.

- **Technical Details**: 우리는 p-norm에 대해 (L, ν)-Hölder 연속성을 가지는 q차 미분 가능한 볼록 함수 f의 최적화를 연구했습니다. 여기서 p와 q는 각각 1 이상입니다. 오라클은 주어진 포인트에서 함수의 q차까지의 모든 미분 결과를 반환합니다. 이 방법은 이론적으로 각 p,q 조합에 대해 최적화 문제를 해결할 수 있으며, 각 반복마다 이진 검색이 필요하지 않습니다.

- **Performance Highlights**: 제안된 알고리즘은 기존에 알려진 하한에서 향상된 성능을 보여줍니다. 또한, 우리는 결정론적 알고리즘의 하한을 제시하였고, 이 알고리즘은 로컬 오라클을 통해 최적화를 수행합니다. 성능은 상수 및 로그 계수까지 일치합니다.



### Dual-Head Knowledge Distillation: Enhancing Logits Utilization with an Auxiliary Head (https://arxiv.org/abs/2411.08937)
Comments:
          Preprint

- **What's New**: 전통적인 Knowledge Distillation(KD) 기법이 확률 예측과 ground-truth label 간의 정렬에 중점을 두는 반면, 이 연구에서는 logit 수준의 손실 함수(logit-level loss function)를 추가하여 logit의 잠재적 정보를 활용하는 새로운 방법인 Dual-Head Knowledge Distillation(DHKD)을 제안합니다.

- **Technical Details**: DHKD에서는 주선형 분류기(linear classifier)를 두 개의 분류 헤드로 분할하여 서로 다른 손실을 담당하도록 구성함으로써, BinaryKL 손실을 이용해 logit의 정보를 효과적으로 활용합니다. 이 과정에서 BinaryKL 손실은 backbone 학습에 긍정적인 영향을 미치지만, linear classifier 헤드의 수렴을 방해하는 문제를 해결할 수 있습니다.

- **Performance Highlights**: 광범위한 실험 결과, DHKD 방법이 기존의 최첨단 기법보다 우수한 성능을 보여주며, logit 내부의 정보를 효과적으로 활용하며 student 모델의 성능 저하 문제를 해결함을 입증하였습니다.



### Clustered Patch Embeddings for Permutation-Invariant Classification of Whole Slide Images (https://arxiv.org/abs/2411.08936)
Comments:
          arXiv admin note: text overlap with arXiv:2411.08530

- **What's New**: 이 논문에서는 Whole Slide Imaging (WSI)의 분석 효율성을 높이기 위해 새로운 방법론을 제안합니다. 기존의 복잡한 WSIs를 단일 벡터로 압축하여 필요한 특징을 효과적으로 추출하는 기법이 핵심 혁신점입니다.

- **Technical Details**: 제안된 방법론은 다양한 인코더를 활용하여 WSI를 사전 처리하고 중요한 특징을 추출합니다. Deep Learning 모델과 적응형 필터링 기술을 사용하여 WSI의 품질을 개선하고, 512x512 크기로 패치를 나누어 ResNet50, EfficientNet 등 여러 모델을 적용하여 고차원 특징 공간으로부터 클러스터링 기반의 compact한 표현을 생성합니다.

- **Performance Highlights**: 본 연구의 실험 결과, 기존 방법에 비해 분류 정확도가 유의미하게 향상되었으며, WSI의 대규모 분석에 대한 새로운 가능성을 제시합니다. 이 연구는 병리학적 데이터의 효과적이고 견고한 분석을 위한 프레임워크를 제공하며, 의료 진단 및 연구에 있어 중요한 기여를 합니다.



### Classification of Keratitis from Eye Corneal Photographs using Deep Learning (https://arxiv.org/abs/2411.08935)
Comments:
          6 pages; Accepted at IEEE's International Conference on Bioinformatics and Biomedicine (2024)

- **What's New**: 본 연구에서는 각기 다른 딥러닝 접근법을 통해 각 감염원의 진단을 비교하고, 새로운 브라질 코라네 데이터셋을 활용하여 '각막염'의 다양한 원인을 분류하기 위한 알고리즘을 제안합니다.

- **Technical Details**: 연구는 세 가지의 딥러닝 모델을 사용합니다: 1) 각각의 감염 유형을 예측하기 위한 세 개의 바이너리 모델; 2) 공유된 백본과 세 개의 병렬 분류 레이어를 가진 멀티태스크 모델(Multitask V1); 3) 공유된 백본과 다중 헤드 분류 레이어를 가진 멀티태스크 모델(Multitask V2). 최종적으로 Multitask V2가 가장 유리한 성능을 보였습니다. 데이터셋 전처리 과정에서, 4,767개의 확인된 각막염 샘플 중에서 2,064개의 고유 사례로 정리하였습니다.

- **Performance Highlights**: Multitask V2 모델의 성능은 세 가지 감염 유형의 AUROC (Receiver Operating Characteristic curve 아래 면적)에서 각각 0.7413-0.7740 (세균), 0.8395-0.8725 (곰팡이), 0.9448-0.9616 (아메바)를 기록하였습니다. 또한, 통계 분석 결과에 따르면 성별이 아메바 감염 예측에 유의미한 영향을 미치며, 나이는 곰팡이와 세균 감염 예측에 영향을 주는 것으로 나타났습니다.



### Predicting household socioeconomic position in Mozambique using satellite and household imagery (https://arxiv.org/abs/2411.08934)
- **What's New**: 남부 모잠비크의 준 농촌 지역에서 975가구의 데이터셋을 구성하여 세대 수준에서 사회경제적 지위(SocioEconomic Position, SEP)를 예측하는 새로운 접근 방식이 제시되었습니다. 이를 통해 위성 이미지와 가정 사진 설문조사를 결합한 다중 모달 데이터셋을 활용하였습니다.

- **Technical Details**: CNN(Convolutional Neural Network)을 미세 조정하여 이미지에서 특징 벡터를 추출하고, 이를 회귀 분석(regression analysis)에 사용하여 가정의 SEP를 모델링 하였습니다. 자산 기반 SEP를 예측할 때 랜덤 포레스트(Random Forest) 모델을 활용하여 가장 좋은 성능을 보였습니다. SHAP(SHapley Additive exPlanations)를 사용하여 이미지의 긍정 및 부정적 영향 차이를 분석하였습니다.

- **Performance Highlights**: 모든 이미지 유형을 사용할 때 자산 기반 SEP의 예측 성능이 가장 뛰어난 것으로 나타났으며, 홈 사진을 사용함으로써 개별 가정 예측으로 zoom-in 하여 데이터 수집 노력을 최소화 할 수 있음을 보여주었습니다. 또한, 중요한 가정 요소만을 사용한 축소 모델은 전체 이미지 모델에 비해 성능이 약간 낮았으나 합리적인 결과를 도출하였습니다.



### Confidence-aware Denoised Fine-tuning of Off-the-shelf Models for Certified Robustness (https://arxiv.org/abs/2411.08933)
Comments:
          26 pages; TMLR 2024; Code is available at this https URL

- **What's New**: 최근 발표된 논문에서는 일반적인 사전 훈련된 분류기의 적대적 강건성을 높이기 위한 새로운 방법인 FT-CADIS를 제안합니다. 이 방법은 특히 역압축된 이미지를 구분하기 위해 신뢰도를 활용합니다.

- **Technical Details**: FT-CADIS는 신뢰도를 고려하여 회복된 이미지를 선택하는 세부 조정 구조를 가지고 있습니다. 이 구조는 두 가지 손실(Confidence-aware selective cross-entropy loss, Confidence-aware masked adversarial loss)을 사용하여 비환각화된 이미지만을 사용하여 분류기를 조정합니다. 이로 인해 분류기는 적대적 강건성을 향상시킬 수 있습니다.

- **Performance Highlights**: FT-CADIS는 CIFAR-10 및 ImageNet 벤치마크에서 기존의 최첨단 방법들과 비교하여 적대적 강건성 면에서 상당한 성능 향상을 보여주었습니다. 특히, FT-CADIS는 높은 가우시안 분산 조건에서 29.5%에서 39.4%의 성능 향상을 기록하였습니다.



### Aligning Visual Contrastive learning models via Preference Optimization (https://arxiv.org/abs/2411.08923)
- **What's New**: 이 논문은 디자인된 Preference Optimization (PO) 방법을 통해 대비학습(contrastive learning) 모델의 훈련 방식을 개선하고, 특히 모델의 내구성을 높이며 성별 이해를 분리하고 성차별을 완화하는 데 중점을 둡니다.

- **Technical Details**: 대비학습(contrastive learning) 모델은 표현을 임베딩 공간(embedding space)에서 정렬하여 의미적 유사성을 포착하는 인상적인 능력을 보여주지만, 훈련 데이터의 품질 및 내재된 편견에 의해 성능이 제한될 수 있습니다. 이 연구는 PO 방법을 통해 모델의 행동을 원하는 선호도와 체계적으로 정렬시키며, 일반적인 대비 모델인 CLIP에 대해 치명적인 공격(typographic attacks)에 대한 내성을 강화합니다.

- **Performance Highlights**: PO를 사용하여 훈련된 모델은 표준 대비학습 기술보다 뛰어난 성과를 보이며, 적대적 도전(adversarial challenges)을 처리하고 다른 하위 작업에서 정확성을 유지하는 능력을 보여줍니다. 이 방법은 공정성(fairness), 내구성(robustness), 특정 선호도(preference)와의 정렬이 필요한 작업에 적합합니다.



### A Machine Learning based Hybrid Receiver for 5G NR PRACH (https://arxiv.org/abs/2411.08919)
Comments:
          6 pages, 9 figures

- **What's New**: 이 논문에서는 사용자가 기지국(BS)에 자신을 식별하는 랜덤 액세스 절차에서의 하이브리드 수신기 설계를 소개합니다. 하이브리드 수신기는 AI/ML 모델을 기반으로 한 프리앰블 감지와 전통적인 피크 검출을 결합하여 타이밍 어드벤스(TA)를 추정합니다.

- **Technical Details**: 하이브리드 수신기는 여러 안테나의 상관 창에서의 전력 지연 프로파일(Power Delay Profile, PDP)을 결합하여 신경망(Neural Network) 모델의 입력으로 사용합니다. 이 모델은 특정 프리앰블 창에서 사용자의 존재 여부를 예측하며, 이후 피크 검출을 통해 타이밍 어드벤스를 추정합니다.

- **Performance Highlights**: 실험 결과에 따르면, 이 하이브리드 수신기는 전통적인 상관 기반 수신기와 다른 AI/ML 기반 접근 방식들보다 우수한 성능을 보이며, 시뮬레이션 데이터와 실제 하드웨어 수집 데이터에서 모두 성능이 향상되었습니다.



### A Message Passing Neural Network Surrogate Model for Bond-Associated Peridynamic Material Correspondence Formulation (https://arxiv.org/abs/2411.08911)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2410.00934

- **What's New**: 이 논문에서는 파형(Wave)과 물질 간의 연결을 다루는 새로운 접근 방식으로, MPNN(Message-Passing Neural Network)을 기반으로 한 서그레이트 모델을 제안하여 기존의 bond-associated peridynamic material correspondence formulation의 계산 비용 문제를 해결합니다.

- **Technical Details**: Peridynamics는 비국소적(Non-local) 연속체 역학 이론으로, 물질 점(Material point) 간의 비국소적 상호작용을 통한 힘을 설명합니다. 본 연구에서는 MPNN을 사용하여, 이러한 연속체 모형의 본질적인 특성을 그대로 살리면서, GPU 가속을 통해 계산 시간을 단축시키는 방법을 제안합니다. 모델은 고정된 이웃 연결성이 필요하지 않아 유연하고 다양한 구성에서 활용 가능합니다.

- **Performance Highlights**: MPNN 기반 서그레이트 모델은 기존 bond-associated formulation 보다 빠른 계산 속도를 제공하며, 물리적 객관성을 유지하고, 다양한 복잡한 시스템에도 적응 가능합니다. 이 접근법은 새로운 물질 반응을 발견하는 데도 기여할 수 있을 것으로 기대됩니다.



### Automated Feedback in Math Education: A Comparative Analysis of LLMs for Open-Ended Responses (https://arxiv.org/abs/2411.08910)
Comments:
          12 pages including references, 4 figures, 9 tables

- **What's New**: 이 연구는 대형 언어 모델(Large Language Models, LLMs), 특히 Mistral 기반의 Llama 변형 모델을 활용하여 수학 교육에서 자동화된 피드백을 제공하는 동향을 탐구합니다. 구체적으로, Mistral 모델을 수학 문제에 맞게 세부 조정하여 학생의 개방형 응답을 평가하는 성능을 비교합니다.

- **Technical Details**: 이 연구에서는 LLM을 사용하여 학생의 수학 문제 응답을 평가하고, Llama, SBERT-Canberra, GPT-4 모델의 성능을 비교합니다. Mistral 모델은 교육 데이터를 활용해 세부 조정 되었으며, SBERT 모델도 유사한 접근법을 사용하였습니다. GPT-4는 제로샷 학습(zero-shot learning) 접근 방식을 통해 개방형 질문과 관련된 특정 루브릭을 기반으로 평가됩니다.

- **Performance Highlights**: 세 가지 모델(SBERT-Canberra, Mistral-Llama, GPT-4)의 성능을 교사 두 명의 평가를 통해 정량적 및 정성적으로 분석하였습니다. 연구 결과, LLM이 생성하는 피드백의 정확성과 관련성을 기반으로 한 교사 평가에서 가치있는 통찰력을 제공하며, 향후 자동화된 피드백 시스템의 발전 방향을 제시합니다.



### Long-context Protein Language Mod (https://arxiv.org/abs/2411.08909)
Comments:
          32 pages, 17 figures, 11 tables

- **What's New**: 이 연구에서는 LC-PLM이라는 새로운 단백질 언어 모델을 제안하며, 이는 BiMamba-S 아키텍처를 기반으로 하여 단백질 시퀀스를 학습합니다. 기존의 Transformer 기반 모델들이 가진 한계를 극복하고, 단백질 상호작용 정보가 포함된 그래프를 연계한 후속 학습을 통해 성능을 높입니다.

- **Technical Details**: LC-PLM 모델은 선택적 구조 상태 공간 모델(SSM)인 BiMamba-S를 기반으로 하며, 마스킹된 언어 모델링(Masked Language Modeling) 기법을 사용하여 단백질의 아미노산 토큰 수준에서 고품질의 보편적 단백질 표현을 학습합니다. 또한, LC-PLM-G라는 그래프 문맥을 활용한 변형 모델을 도입하여 단백질-단백질 상호작용(PPI) 그래프를 컨텍스트로 삼아 두 번째 단계의 훈련을 진행합니다.

- **Performance Highlights**: LC-PLM은 Transformer 기반의 ESM-2 모델에 비해 7%에서 34%의 성능 향상을 보였으며, 특히 긴 단백질과 단백질 복합체에 대한 구조 예측 작업에서 두드러진 성과를 나타냈습니다. 추가적으로, PPI 그래프 문맥을 고려한 LC-PLM-G 모델은 원거리 상동성 예측 및 단백질 기능 예측에서 유망한 결과를 보였습니다.



### Turkey's Earthquakes: Damage Prediction and Feature Significance Using A Multivariate Analysis (https://arxiv.org/abs/2411.08903)
- **What's New**: 이번 연구는 터키에서의 지진 피해 예측을 위해 다양한 머신러닝 모델을 테스트한 결과, 랜덤 포레스트(Random Forest) 모델이 가장 신뢰할 수 있는 예측을 제공함을 확인했습니다. 이 모델은 지진의 규모와 건물 안정성을 주요 요인으로 강조하며, 향후 지진 재난 대비 및 대응 전략 수립에 기여하고자 합니다.

- **Technical Details**: 연구에서는 1950년 이전 발생한 지진에 대한 네 가지 데이터셋을 분석하였고, 각 데이터셋은 지진의 크기, 깊이, 사망자 수 및 진원지 좌표에 대한 정보를 포함합니다. 또한, 터키의 각 주에 대한 인구 밀도, 구조적 안정성(Building Condition Index, BCI) 및 건물의 지진 피해 취약성(Structural Vulnerability Index, SVI) 데이터를 통합하였습니다. 모델 학습에서는 평가지표로 MAPE(Mean Absolute Percent Error) 및 MAE(Mean Absolute Error)를 사용하여 성능을 평가하였습니다.

- **Performance Highlights**: 모델 성능 평가 결과, 결정 트리(Decision Tree)와 랜덤 포레스트 모델이 가장 낮은 MAPE 및 MAE 값을 기록하며 우수한 성능을 보였습니다. 특히 랜덤 포레스트 모델이 여러 결정 트리를 결합하여 보다 정교한 예측을 가능하게 하여 최상의 피해 예측 모델로 선정되었습니다. 인구 밀도는 사망자 수 예측에서 가장 중요한 요소로 나타났으며, 지진 규모는 전체 사망자 수 예측에서 가장 중요한 요소로 판별되었습니다.



### SoccerGuard: Investigating Injury Risk Factors for Professional Soccer Players with Machine Learning (https://arxiv.org/abs/2411.08901)
- **What's New**: 이번 논문에서는 여성 축구에서 부상을 예측하기 위한 새로운 머신러닝 프레임워크인 SoccerGuard를 소개합니다. 이 프레임워크는 선수의 주관적인 웰빙 및 훈련 부하 보고서, 객관적인 GPS 센서 측정값, 제3자 선수 통계 및 의료진에 의해 확인된 부상 보고서를 포함한 다양한 출처의 데이터를 수집할 수 있습니다.

- **Technical Details**: SoccerGuard는 세 가지 주요 파이프라인으로 구성됩니다: 전처리 블록(Preprocessing Block), 자동화된 머신러닝 블록(Automated Machine Learning Block) 및 축구 대시보드(Soccer Dashboard). 이 논문에서는 부상 사건 예측을 위한 90개의 고유한 머신러닝 실험을 수행하였으며, 최적의 결과는 입력 창(input window)이 축소되고, 더 큰 출력 창(output window)이 정의되었을 때 나타났습니다. 또한 대시보드에는 사용자 친화적인 그래픽 사용자 인터페이스(GUI)가 포함되어 있어 상호작용적인 분석 및 시각화를 지원합니다.

- **Performance Highlights**: 적절한 구성 및 특징 조합이 제공될 경우, 부상 사건 예측은 상당한 정확도로 수행될 수 있습니다. SoccerGuard의 데이터 세트와 파이프라인은 부상 예측 정확도를 향상시키는 데 기여하고, 선수 데이터 및 잠재적인 부상 위험에 대한 포괄적인 이해를 제공합니다.



### RNA-GPT: Multimodal Generative System for RNA Sequence Understanding (https://arxiv.org/abs/2411.08900)
Comments:
          Machine Learning for Structural Biology Workshop, NeurIPS 2024

- **What's New**: 이번 논문에서는 RNA 발견을 간소화하기 위해 RNA-GPT라는 멀티모달 RNA 챗 모델을 소개합니다. 이 모델은 방대한 RNA 문헌을 활용하여 RNA 연구를 지원합니다.

- **Technical Details**: RNA-GPT는 RNA 서열 인코더(RNA sequence encoders)와 선형 프로젝션 레이어(linear projection layers), 최신 대형 언어 모델(large language models, LLMs)을 통합하여 정밀한 표현 정렬(representation alignment)을 가능하게 합니다. 이 모델은 사용자 업로드 RNA 서열을 처리하고 간결하고 정확한 응답을 제공합니다.

- **Performance Highlights**: 실험 결과, RNA-GPT는 복잡한 RNA 쿼리를 효과적으로 해결함으로써 RNA 연구를 촉진시킵니다. 또한, 407,616개의 RNA 샘플을 포함하는 RNA-QA 데이터셋을 제공하여 모달리티 정렬(modality alignment) 및 instruction tuning을 통해 RNA 연구 도구의 가능성을 더욱 발전시킵니다.



### Demand-Aware Beam Hopping and Power Allocation for Load Balancing in Digital Twin empowered LEO Satellite Networks (https://arxiv.org/abs/2411.08896)
- **What's New**: 저널 논문에서는 Low-Earth Orbit (LEO) 위성을 위한 Digital Twin (DT) 기반의 협업 자원 할당 네트워크를 탐구하고 있습니다. 이 네트워크는 서로 겹치는 커버리지 영역을 가진 다수의 LEO 위성에서 beam hopping (BH) 기술을 적용하여 효율적인 자원 사용을 목표로 합니다.

- **Technical Details**: 저자는 두 단계 최적화 문제를 제안하여 로드 밸런스와 셀 서비스 공정성에 집중합니다. DT 레이어에서는 각 위성을 위한 BH 패턴을 설계하고, LEO 레이어에서는 선택된 서비스 셀에 대한 전력 할당을 최적화합니다. DT 레이어에서는 Actor-Critic 네트워크를 사용하고, LEO 레이어에서는 Multi-Agent Reinforcement Learning 알고리즘을 적용하여 최적화를 수행합니다.

- **Performance Highlights**: 시뮬레이션 결과는 이 방법이 위성 로드 불균형을 72.5% 감소시키고 평균 지연 시간을 12ms로 줄이는 성과를 보였으며, 전송량(throughput)에서도 다른 벤치마크를 초월하는 결과를 나타냈습니다.



### Network scaling and scale-driven loss balancing for intelligent poroelastography (https://arxiv.org/abs/2411.08886)
- **What's New**: 이번 연구는 poroelastography를 위한 심층 학습 프레임워크를 개발하여 다중 스케일에서 poroelastic 매체를 특성화하는 새로운 방법을 제시합니다. 특히 이 방법은 다양한 스케일에서 급격히 변화하는 다상적 속성을 지닌 이질 환경에 중점을 두고 있으며, 실시간 데이터로부터 생체 조직의 기계적 특성을 복원하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 이 연구는 포화 고체 변위와 기공 압력 필드를 사용하여 Biot 방정식과 관련된 6개의 수리 기계적 속성을 회복하는 것을 목표로 합니다. 두 가지 주요 도전 과제가 있으며: (i) 서로 매우 다른 불확실한 스케일에 걸쳐 속성이 존재하며, (ii) 손실 함수가 다목적(multi-objective)이며 다중 스케일(multi-scale)입니다. 이를 해결하기 위해 네트워크 스케일링(Network Scaling) 개념을 도입하였고, 동적 스케일링(Dynamic Scaling) 방법을 통해 손실 균형 조절을 위한 적응형 접근 방식을 제안합니다.

- **Performance Highlights**: 이 모델의 성능은 GradNorm(그래디언트 정규화) 및 Softmax 기반의 적응형 가중치(SoftAdapt)를 통해 손실 균형을 맞추며 입증되었습니다. 포로엘라스토그래피에 대한 수치 실험을 통해 제안된 방법과 전통적인 방법간의 비교 분석이 수행되어 더 나은 성능을 보여주었습니다.



### Machine learning-enabled velocity model building with uncertainty quantification (https://arxiv.org/abs/2411.06651)
- **What's New**: 본 논문에서는 전통적인 FWI(Full-Waveform Inversion) 방법의 한계를 극복하기 위해, Diffusion networks를 활용하여 물리적 요약 통계(physics-informed summary statistics)와 결합한 확장 가능한 방법론을 제안합니다. 이 접근법은 복잡한 이미징 문제에 적합하며, 기존 속도 모델이 부족한 경우에도 효율적으로 베이esian posterior 샘플을 생성할 수 있도록 합니다.

- **Technical Details**: 이 연구는 주로 베이esian 추론(Bayesian inference) 프레임워크를 활용하여, 관측 데이터(지진 촬영 데이터)와 프라이어 정보(훈련 샘플)를 결합하여 다수의 지구 모델을 도출합니다. 이 방법은 수치적으로 효율적이며, 큰 데이터 세트에서도 적용이 가능합니다. 주요 기술적 요소로는 조건부 Diffusion networks와 물리적 요약 통계가 포함됩니다.

- **Performance Highlights**: 실험을 통해 기존 방법들과 비교하여 Common-Image Gathers(CIGs)의 사용으로 인한 개선을 확인하였으며, 복잡한 소금 구조를 다루기 위한 새로운 반복적 워크플로우인 ASPIRE를 제안합니다. 마지막으로, 이 방법은 분야 데이터셋에서도 잘 작동하여 산업 규모 문제에 적합하다는 것을 입증하였습니다.



### Mixed Effects Deep Learning for the interpretable analysis of single cell RNA sequencing data by quantifying and visualizing batch effects (https://arxiv.org/abs/2411.06635)
Comments:
          Main manuscript: 29 pages, including 10 figures and 8 tables. Supplemental material: 17 pages

- **What's New**: 이번 연구에서는 배치 효과(batch effects)를 구분하여 처리할 수 있는 Mixed Effects Deep Learning (MEDL) 오토인코더(autoencoder) 프레임워크를 제안합니다. 기존의 딥러닝 모델들이 배치 특유 정보(batch-specific information)를 버리면서 생물학적 통찰력을 손실할 수 있는 반면, 우리의 접근법은 이를 해결합니다.

- **Technical Details**: MEDL 프레임워크는 고정 효과(fixed effects)와 랜덤 효과(random effects)를 별도로 모델링하여 배치 불변 생물학적 상태와 배치 변동을 분리합니다. 이를 통해 예측 모델에 두 가지를 통합하고, 동일한 세포가 배치마다 어떻게 나타나는지를 2D 비주얼화(visualization)하여 해석 가능성을 높입니다.

- **Performance Highlights**: 이 프레임워크는 세 가지 데이터셋(심혈관계(Healthy Heart), 자폐 스펙트럼 장애(ASD), 급성 골수 백혈병(AML))에 적용되었습니다. 특히, Healthy Heart 데이터셋에서는 147개의 배치를 처리하여 모델의 배치 처리 능력을 시험했으며, ASD 데이터셋에서는 자폐 및 건강한 개인 간 기여자의 이질성을 포착했습니다. AML 데이터셋에서는 결측 세포 유형에도 불구하고 기여자 이질성을 제대로 구분할 수 있었습니다. 이러한 결과는 고정 및 랜덤 효과의 표시, 배치 효과의 시각화 개선, 그리고 다양한 데이터셋에서의 예측 정확도를 향상시키는 데 기여함을 나타냅니다.



### Learning Multi-Agent Loco-Manipulation for Long-Horizon Quadrupedal Pushing (https://arxiv.org/abs/2411.07104)
- **What's New**: 이 논문에서는 다수의 4족 로봇이 장애물을 인식하며, 장기적인 푸시 작업을 수행할 수 있는 새로운 계층적 다중 에이전트 강화 학습(MARL) 프레임워크를 제안합니다. 이 프레임워크는 세 가지 제어 수준으로 구성되어 장애물을 피하면서 물체를 효과적으로 밀 수 있도록 돕습니다.

- **Technical Details**: 제안된 방법은 높은 수준의 컨트롤러가 RRT(Rapidly-exploring Random Tree) 계획자와 중앙 집중식 적응 정책을 통합하여 서브 목표를 생성하고, 중간 수준의 컨트롤러가 분산 목표 조건 정책을 사용하여 로봇이 해당 서브 목표로 안내하도록 합니다. 저수준 컨트롤러는 사전 훈련된 보행 정책을 통해 이동 명령을 실행합니다.

- **Performance Highlights**: 시뮬레이션에서 제안된 방법은 최고의 기준선 방법보다 성공률이 36.0% 높고, 완료 시간은 24.5% 단축되는 성능을 보였습니다. 이 방법은 실제 로봇인 Go1에서 Push-Cuboid 및 Push-T와 같은 장애물 인식 및 장기 푸시 작업을 성공적으로 수행할 수 있도록 합니다.




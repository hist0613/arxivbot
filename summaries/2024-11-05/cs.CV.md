New uploads on arXiv(cs.CL)

### Mitigating Tail Narrowing in LLM Self-Improvement via Socratic-Guided Sampling (https://arxiv.org/abs/2411.00750)
Comments:
          Codes are publicly available at this https URL

- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 자체 개선(self-improvement) 방법의 성능 한계를 극복하기 위해 'Guided Self-Improvement (GSI)'라는 새로운 전략을 도입합니다. GSI는 Socratic-style guidance signals를 활용하여 복잡한 쿼리에 대한 모델의 추론을 돕고, 고차원 데이터 샘플링의 효율성을 향상시키는 것을 목표로 합니다.

- **Technical Details**: GSI는 샘플링 단계 후 복잡한 쿼리 샘플링을 위한 추가 재샘플링 단계인 distribution re-balancing을 도입하여 샘플링 공간을 축소하고 모델의 허위 추론(hallucinations)을 줄입니다. 이를 통해 GSI는 도전적인 쿼리에 대한 솔루션 범위를 확장하고 성능 향상을 도모합니다.

- **Performance Highlights**: GSI는 네 개의 모델 및 여섯 개의 수학적 추론 작업에 대한 실험에서 성능 병목 현상을 완화하고, 계산 효율성을 유지하면서 균형 잡힌 솔루션 분포와 개선된 모델 일반화 성능을 보여줍니다.



### MolCap-Arena: A Comprehensive Captioning Benchmark on Language-Enhanced Molecular Property Prediction (https://arxiv.org/abs/2411.00737)
- **What's New**: 이 논문은 생물 분자 모델링(biomolecular modeling)과 자연어 정보(natural language information)를 연결하는 최신 연구의 일환으로, 대규모 언어 모델(large language models, LLMs)을 사용하는 방법을 제시합니다.

- **Technical Details**: 이 연구에서는 Molecule Caption Arena라는 새로운 벤치마크를 도입하여 LLM으로 증강된 분자 속성 예측(molecular property prediction)을 평가합니다. 이를 위해 일반 용도 및 특정 도메인에 맞춘 분자 설명 생성기(molecule captioners) 등 20개 이상의 LLM을 다양한 예측 작업을 통해 비교했습니다. 논문은 또한 새로운 전투 기반 평가 시스템(battle-based rating system)을 소개합니다.

- **Performance Highlights**: LLM에서 추출한 지식이 최신의 분자 표현(molecular representations)을 향상시키는 능력을 확인했으며, 모델, 프롬프트(prompt), 데이터셋에 따라 다양한 성과 차이를 보였습니다.



### SPRING Lab IITM's submission to Low Resource Indic Language Translation Shared Task (https://arxiv.org/abs/2411.00727)
Comments:
          To be published in WMT 2024. Low-Resource Indic Language Translation Shared Task

- **What's New**: 본 연구에서는 Khasi, Mizo, Manipuri, Assamese 등 네 가지 저자원 언어에 대한 강력한 번역 모델을 개발하였으며, 데이터 수집 및 전처리에서 교육 및 평가에 이르는 포괄적인 파이프라인을 포함하고 있습니다.

- **Technical Details**: 우리는 WMT, BPCC, PMIndia, OpenLanguageData에서 데이터를 활용하여 모델을 훈련시켰으며, 단일 언어 데이터셋에 대한 역번역(back-translation) 기법을 통해 Mizo와 Khasi의 이중 언어 데이터를 획기적으로 확장하였습니다. 또한, NLLB 3.3B 모델을 세 가지 언어에 대해 파인튜닝(fine-tuning) 하여 성능을 개선했습니다.

- **Performance Highlights**: 모델 평가 결과, Assamese의 경우 BLEU 점수가 27.26, Khasi는 NLLB 모델 지원 부족에도 불구하고 특수 토큰을 도입하여 훈련된 결과가 긍정적이었습니다. 전체적으로 Mizo와 Manipuri 번역 방향의 두 점수는 역번역 데이터 품질 저하로 인해 낮았습니다.



### A graph-based approach to extracting narrative signals from public discours (https://arxiv.org/abs/2411.00702)
Comments:
          23 pages, 4 figures

- **What's New**: 이 논문은 정치적 내러티브(narrative)의 분석을 위한 새로운 그래프 기반(formalism) 방법을 제안합니다. 이러한 방법은 디지털 텍스트로부터 내러티브 신호를 효과적으로 추출하고 분석하는 데 중점을 두며, 특히 정치적인 상황에서 유용하게 활용될 수 있습니다.

- **Technical Details**: 우리는 Abtract Meaning Representation (AMR)에 기반하여 텍스트 집합에서 각 문장의 의미를 그래프와 같은 표현으로 추출합니다. 이후, 서사학(narratology)의 개념을 활용하여 1) 행위자(actors), 2) 이들이 포함된 사건(events), 3) 사건의 관점(perspectivization)을 필터링하는 휴리스틱(heuristics)을 적용합니다. 이러한 요소들은 정치적 내러티브를 형성하는 핵심 신호로 간주됩니다.

- **Performance Highlights**: 유럽연합의 연설을 사례 연구(case study)로 사용하여 제안된 방법이 공적 담론에서 정치적 내러티브의 신호를 효과적으로 표출하는 데 어떻게 활용되는지를 보여줍니다.



### Leveraging Large Language Models for Code-Mixed Data Augmentation in Sentiment Analysis (https://arxiv.org/abs/2411.00691)
Comments:
          17 pages, 4 figures, 11 tables, To be published in the Proceedings of the Second Workshop on Social Influence in Conversations (SICon 2024), co-located with EMNLP 2024

- **What's New**: 다국어 사회에서 흔히 나타나는 코드 혼합(CM) 언어 사용의 복잡성과 제한된 데이터 때문에 자연어 처리가 어려운 상황에서, 대형 언어 모델(LLM)을 활용하여 합성 CM 데이터를 생성하고 이를 통해 감정 분석(Sentiment Analysis) 모델의 성능을 향상시키는 방법을 제안합니다.

- **Technical Details**: 이 연구에서는 스페인어-영어와 말라얄람어-영어 샘플을 대상으로 LLMS를 통해 생성한 합성 데이터를 사용하여 감정 분석 모델을 fine-tune 합니다. 이 방식은 기존 변환 기술보다 효과적이며, 특히 낮은 성능 기준을 가진 경우에 자연스러운 CM 문장을 효과적으로 생성할 수 있습니다.

- **Performance Highlights**: 스페인어-영어 경우, 합성 데이터는 F1 점수를 9.32% 향상시켜 기존 성능 기준을 초과했으며, 말라얄람어-영어 데이터셋에서는 합성 데이터가 낮은 기준에서만 도움을 주었습니다. 이 연구는 감정 분석 시스템 개발에 중요한 영향을 미칠 수 있는 CM 데이터 증강을 위한 유망한 방법을 제시합니다.



### Towards Multi-Source Retrieval-Augmented Generation via Synergizing Reasoning and Preference-Driven Retrieva (https://arxiv.org/abs/2411.00689)
Comments:
          5 pages, 1 figure

- **What's New**: 새로운 MSPR 프레임워크는 다양한 Retrieval Sources를 효과적으로 탐색하기 위한 기법으로, 정보 수집 시 '언제'와 '무엇을' 검색할지를 결정하는 방법을 제안합니다.

- **Technical Details**: MSPR은 세 가지 주요 구성요소로 이루어져 있습니다: 1) Adaptive Reasoning-and-Retrieval Agent (ARA)는 최적의 검색 행동을 결정하며, 2) Preference-Driven Retrieval Strategy Selector (PRS)는 고품질 소스를 우선적으로 탐색하게 하고, 3) Corrective Answer Reviewer (CAR)는 답변 품질에 대한 피드백을 제공합니다.

- **Performance Highlights**: MSPR은 세 개의 데이터셋에서 다양한 실험을 통해 기존 MS-ARAG 대비 EM 메트릭에서 14.4% 향상된 성능을 보여주었습니다.



### Latent Paraphrasing: Perturbation on Layers Improves Knowledge Injection in Language Models (https://arxiv.org/abs/2411.00686)
Comments:
          NeurIPS 2024

- **What's New**: 이 논문에서는 LaPael이라는 새로운 잠재 레벨( latent-level) 패러프레이징 방법을 소개하여 모델의 초기 레이어에 입력 의존성 노이즈(input-dependent noise)를 적용함으로써 지식 주입(knowledge injection)을 혁신적으로 개선했습니다.

- **Technical Details**: LaPael은 LLMs에서 패러프레이징(paraphrasing) 데이터를 활용해 지식을 주입하는 기존 접근법의 두 가지 주요 문제인 높은 계산 비용과 제한된 샘플 다양성을 해결합니다. 이 방법은 모델 내부에서 직접적으로 다양한 의미적으로 일관된 증강(augments)을 생성할 수 있게 합니다.

- **Performance Highlights**: 질문-응답(Question-Answering) 벤치마크에서의 광범위한 실험 결과, LaPael은 기존의 표준 파인튜닝(standard fine-tuning) 및 노이즈 기반 접근법보다 지식 주입의 성능을 개선하며, 데이터 수준의 패러프레이징과 결합할 경우 성능을 더욱 강화하는 것으로 나타났습니다.



### Zipfian Whitening (https://arxiv.org/abs/2411.00680)
Comments:
          NeurIPS 2024

- **What's New**: 본 논문은 신경망 모델의 단어 임베딩 공간이 왜곡되어 있으며, 이를 수정하면 작업 성능이 개선된다는 점을 지적합니다. 특히, 단어 빈도가 균일하다는 기존 가정과는 달리, Zipf의 법칙(counter) 따라 매우 비균등 분포를 따른다는 사실을 강조하였습니다.

- **Technical Details**: 단어 임베딩의 교정 및 대칭성 측정은 주로 균일한 단어 빈도가 전제되어 왔습니다. 하지만 본 연구에서는 Zipf 법칙을 따르는 경험적 단어 빈도로 가중된 PCA whitening을 수행하는 것만으로도 성능이 크게 향상되는 것으로 나타났습니다. 이론적으로는, 단어 표현이 균일하거나 Zipfian 기준 측정에 따라 배포되는 지 확인하였습니다. 후자의 방법을 채택함으로써 우리는 로우 빈도(low-frequency) 단어를 강조할 수 있습니다.

- **Performance Highlights**: 실험 결과, Zipf 법칙에 따른 경험적 단어 빈도로 가중치가 부여된 PCA whitening 방법이 기존의 기법들을 초월하는 성능 개선을 보여주었습니다. 또한 염려와 대비하여 효과적인 임베딩이 제공되는 자연어 처리 방법들(skipping-gram negative sampling, WhiteningBERT 등)의 이론적 근거를 제공하였습니다.



### Phase Diagram of Vision Large Language Models Inference: A Perspective from Interaction across Image and Instruction (https://arxiv.org/abs/2411.00646)
Comments:
          6 pages, 5 figures

- **What's New**: 이 논문에서는 Vision Large Language Models (VLLMs)의 내부 동작 방식에 대한 심층적인 조사를 통해 서로 다른 모달리티(모드) 간의 상호작용을 다룹니다.

- **Technical Details**: 4단계의 추론 동역학(혹은 dynamics)을 제시합니다: (I) Alignment, (II) Intra-modal Encoding, (III) Inter-modal Encoding, (IV) Output Preparation. 각 단계에서의 contextualization(문맥화) 변화를 분석하여 Transformer 기반의 LMs에서 모달리티 간의 상호작용이 어떻게 진행되는지를 설명합니다.

- **Performance Highlights**: VLLMs의 여러 레이어에서 모달리티 간의 유사성이 증가하는 일반적인 경향을 확인했습니다. 특히, 중간 레이어와 깊은 레이어에서 집중적인 주의(Attention)가 점진적으로 나타나지만, 마지막 레이어에서는 현저히 약해지는 경향을 발견했습니다.



### ConvCounsel: A Conversational Dataset for Student Counseling (https://arxiv.org/abs/2411.00604)
Comments:
          Accepted at O-COCOSDA 2024, Won Best Student Paper Award

- **What's New**: 이 논문은 대학생 상담을 위한 ConvCounsel 데이터세트를 소개하며, 대화에서의 적극적인 경청 전략을 강조합니다. 이는 일반적인 정신 건강 데이터세트의 한계를 보완하고자 하는 시도입니다.

- **Technical Details**: ConvCounsel 데이터세트는 대학생과 상담자 간의 40개 상담 세션으로 구성되어 있으며, 발화 데이터와 텍스트 데이터가 포함되어 있습니다. 이 데이터는 자연어 처리(NLP) 및 대화형 AI 시스템에 활용될 수 있습니다.

- **Performance Highlights**: NYCUKA 시스템은 ConvCounsel 데이터세트를 기반으로 하여 중문 입력에 empathetic responses를 제공하며, 사용자 경험을 향상시키기 위해 애니메이션 캐릭터를 통해 상담 시뮬레이션을 구현하였습니다.



### Adapting Language Models via Token Translation (https://arxiv.org/abs/2411.00593)
- **What's New**: Sparse Sinkhorn Token Translation (S2T2) 알고리즘을 도입하여 특정 도메인에서 효과적인 압축을 위한 맞춤형 토크나이저를 훈련하고, 원천과 목표 토큰 간의 변환을 학습합니다.

- **Technical Details**: S2T2는 높은 자원 소모 없이도 새로운 목표 도메인에서 훈련된 토큰의 분포를 학습하며, 사전 훈련된 모델을 활용하여 더 나은 다음 토큰 예측을 가능하게 합니다. 이를 통해 기존 모델에 비해 더 나은 perplexity 및 압축 성능을 발휘합니다.

- **Performance Highlights**: 실험 결과, S2T2는 아웃 오브 도메인(in-domain) 단백질 서열에 대해 perplexity와 압축 모두에서 개선된 결과를 보여줍니다. 또한 작은 모델에서 학습된 토큰 변환은 더 큰 모델에도 직접 이전 가능해 비용을 낮추면서도 효과를 누릴 수 있습니다.



### ReverseNER: A Self-Generated Example-Driven Framework for Zero-Shot Named Entity Recognition with Large Language Models (https://arxiv.org/abs/2411.00533)
- **What's New**: ReverseNER는 대규모 언어 모델(LLMs)이 제로샷(NER, Named Entity Recognition) 과제에서 갖는 한계를 극복하기 위해 제안된 새로운 프레임워크입니다. 이 방법은 NER 프로세스를 반대로 진행하여 신뢰할 수 있는 예시 라이브러리를 구축합니다.

- **Technical Details**: ReverseNER는 사전 훈련된 BERT 모델을 사용하여 작업 문장 간의 유사성을 계산한 후 클러스터링을 통해 주요 문장을 추출하고, 이를 기반으로 LLM이 관련 예시 문장 및 엔티티를 생성하도록 유도합니다. 이 과정은 문장 생성 시 특정 '특징 문장'의 구조를 복제하도록 LLM을 안내하여 세밀하게 주석이 달린 문장을 생성하게 합니다.

- **Performance Highlights**: 실험 결과, ReverseNER는 전통적인 제로샷 NER 방법보다 상당히 우수한 성능을 보이며, 데이터가 제한된 도메인에서의 NER 성능이 크게 향상되었음을 보여줍니다.



### Multi-expert Prompting Improves Reliability, Safety, and Usefulness of Large Language Models (https://arxiv.org/abs/2411.00492)
Comments:
          EMNLP 2024 Main Conference

- **What's New**: 저희는 다중 전문가 프롬프트(Multi-expert Prompting)라는 새로운 기술을 소개합니다. 이 기술은 ExpertPrompting(Xu et al., 2023)을 발전시켜 대형 언어 모델(LLM)의 생성 품질을 향상시키는 것을 목표로 합니다.

- **Technical Details**: 이 기술은 입력 지시사항을 수행하기 위해 여러 전문가를 시뮬레이션하고, 그들의 응답을 집계하여 최선의 응답을 선택하는 방법을 사용합니다. 이 과정은 1974년 Ven과 Delbecq가 개발한 의사결정 프레임워크인 Nominal Group Technique에서 파생된 일곱 개의 세부 작업을 통해 단일 사고 체인으로 수행됩니다.

- **Performance Highlights**: Multi-expert Prompting은 응답의 진실성(truthfulness), 사실성(factuality), 정보성(informativeness) 및 유용성(usefulness)을 크게 향상시키며, 독성(toxicity) 및 상처를 줄이는 데도 효과적입니다. 또한 ChatGPT와 함께 사용할 경우, 기존 최선 기준보다 8.69% 더 높은 진실성을 달성하며, 최고 성능을 기록하고 있습니다.



### GDTB: Genre Diverse Data for English Shallow Discourse Parsing across Modalities, Text Types, and Domains (https://arxiv.org/abs/2411.00491)
Comments:
          Accepted to EMNLP 2024 (main, long); camera-ready version

- **What's New**: 이번 논문에서는 PDTB 스타일의 얕은 담화 구문 분석을 위한 새로운 공개형 다장르 기준 데이터셋을 제안하고 평가합니다. 기존의 UD English GUM 코퍼스를 기반으로 하여 담화 관계 주석이 다른 프레임워크에서 이미 존재하는 데이터입니다.

- **Technical Details**: 이 연구에서는 Penn Discourse Treebank (PDTB)와 호환되는 새로운 데이터셋을 구축하였습니다. GUM 코퍼스를 활용하여 RST 및 eRST와 같은 계층적 담화 구문 분석 프레임워크로부터 세미 자동 변환 프로세스를 통해 고품질의 데이터를 생성합니다. 실험을 통해 데이터의 호환성 및 도메인 간의 성능 저하를 평가합니다.

- **Performance Highlights**: 교차 도메인 관계 분류 실험에서, 연구 결과는 우리의 데이터셋이 PDTB와 호환되지만, 상당한 도메인 외 성능 저하가 관찰되었으며, 두 데이터셋에 대한 공동 훈련(joint training)을 통해 이러한 문제를 일부 완화할 수 있음을 보여줍니다.



### E2E-AFG: An End-to-End Model with Adaptive Filtering for Retrieval-Augmented Generation (https://arxiv.org/abs/2411.00437)
Comments:
          13 pages, 3 figures, 5 tables

- **What's New**: 이 논문에서는 외부 지식 기반에서 얻은 정보의 질을 개선하기 위해 답변 존재 판단과 텍스트 생성을 하나의 종합적인 end-to-end 프레임워크로 통합한 적응형 필터링 (adaptive filtering) 기법인 E2E-AFG를 제안합니다.

- **Technical Details**: E2E-AFG는 retrieval-augmented generation (RAG) 기법을 활용하며, 모델이 관련 콘텐츠에 보다 효과적으로 집중할 수 있도록 지원하여, 불필요한 정보의 영향을 줄여 정확한 답변을 생성합니다.

- **Performance Highlights**: E2E-AFG는 6개의 대표적인 지식 집약적 언어 데이터셋에서 평가되었으며, 모든 태스크에서 기본 모델들에 비해 일관되게 우수한 성과를 보여주며 제안된 접근 방식의 효과성과 강건성을 입증했습니다.



### DARD: A Multi-Agent Approach for Task-Oriented Dialog Systems (https://arxiv.org/abs/2411.00427)
- **What's New**: 본 논문에서는 DARD (Domain Assigned Response Delegation)라는 다중 에이전트 대화 시스템을 제안하여, 여러 도메인에 걸쳐 효과적으로 다이얼로그를 처리할 수 있음을 보여줍니다.

- **Technical Details**: DARD는 중앙 대화 관리자 에이전트에 의해 조정되는 도메인별 에이전트를 활용하여, 대화 맥락에 따라 사용자 메시지에 대한 응답을 생성합니다. 실험에는 Flan-T5-large, Mistral-7B, Claude Sonnet 3.0 모델이 포함되었습니다. MultiWOZ 2.2 데이터셋을 사용하여 성능을 평가하였으며, Joint State Accuracy (JSA), Inform, Success 및 BLEU 점수를 기준으로 성과를 측정합니다.

- **Performance Highlights**: DARD는 기존 방법보다 다이얼로그 정보율을 6.6%, 성공률을 4.1% 향상시키며, MultiWOZ 벤치마크에서 최신 성과를 달성하였습니다. 또한, MultiWOZ 데이터셋의 주석자 간 불일치 문제를 분석하였습니다.



### Self-Evolved Reward Learning for LLMs (https://arxiv.org/abs/2411.00418)
Comments:
          19 pages,6 figures

- **What's New**: 이 논문은 Self-Evolved Reward Learning (SER)이라는 새로운 접근 방식을 제안하고 있습니다. SER은 보상 모델(Reward Model, RM)이 자체적으로 추가 학습 데이터를 생성하여 스스로를 반복적으로 개선하는 방법입니다. 이로 인해 인간이 주석을 단 데이터에 대한 의존성을 줄이고도 언어 모델의 성능을 향상시킬 수 있습니다.

- **Technical Details**: SER 접근 방식에서는 RM이 스스로 높은 신뢰도의 예측을 학습하며, 초기에는 소량의 인간 주석 데이터를 사용하여 보상 모델을 훈련합니다. 그 후 RM은 자체 레이블링(self-labeling)과 반복 훈련을 통해 진화하며, RL 접근 방식에 따라 언어 모델 훈련에 사용됩니다. 이 과정을 통해 RM은 자신의 피드백으로부터 학습하면서 성능을 향상시킵니다.

- **Performance Highlights**: 여러 데이터셋과 LLM(대형 언어 모델)에서 실험을 수행한 결과, 인간 주석 데이터의 15%만 사용하더라도, 기존의 전량 데이터를 사용한 모델과 유사한 성능을 달성하는 것으로 나타났습니다. 최종적으로는 평균 7.88%의 성능 향상을 기록하며, 사람의 주석이 포함된 전체 데이터셋을 사용하는 모델의 성능을 초과할 수 있는 잠재력을 보여주었습니다.



### Enhancing Authorship Attribution through Embedding Fusion: A Novel Approach with Masked and Encoder-Decoder Language Models (https://arxiv.org/abs/2411.00411)
- **What's New**: AI 생성 콘텐츠와 사람 작성 텍스트의 증가로 인해 신뢰할 수 있는 구별 방법의 필요성이 대두되고 있습니다. 본 연구에서는 사전 훈련된 언어 모델(Pre-trained Language Models, PLMs)에서 텍스트 임베딩을 이용하여 AI 생성 텍스트와 인간 저자 텍스트를 구별하는 새로운 프레임워크를 제안합니다.

- **Technical Details**: 본 연구는 Embedding Fusion 기술을 활용하여 여러 언어 모델의 의미 정보를 통합하고, 이를 통해 성능을 강화합니다. PLMs로부터의 텍스트 표현을 이미지로 변환하여 2D 형태로 재구성함으로써 상호 임베딩 관계를 효과적으로 캡처합니다. 실험 결과, 우리의 프레임워크는 AI 생성 텍스트와 인간 저자 텍스트를 구별하는 과제에서 96% 이상의 분류 정확도와 0.93 이상의 Matthews Correlation Coefficient (MCC)를 달성하였습니다.

- **Performance Highlights**: 우리의 새로운 접근법은 다섯 가지 유명한 대형 언어 모델(Large Language Models, LLMs)에서 생성된 균형 잡힌 텍스트 데이터셋을 사용하여 강력한 성능을 보여주었습니다. 이러한 성과는 NLG(자연어 생성) 기술의 전체 잠재력을 실현하고 그 오 남용 가능성을 줄이기 위한 정확한 탐지 메커니즘의 필요성을 강조합니다.



### MetaMetrics-MT: Tuning Meta-Metrics for Machine Translation via Human Preference Calibration (https://arxiv.org/abs/2411.00390)
Comments:
          Preprint

- **What's New**: MetaMetrics-MT는 기계 번역(Machine Translation, MT) 작업을 평가하기 위해 인간 선호도와 밀접하게 연관되도록 설계된 혁신적인 메트릭입니다. Bayesian optimization과 Gaussian Processes를 활용하여 기존 MT 메트릭의 인간 평가와의 상관관계를 최적화합니다.

- **Technical Details**: MetaMetrics-MT는 여러 개의 메트릭을 활용하여 MT 작업을 평가하며, 각 메트릭에 특정 가중치를 부여하여 성능을 최적화합니다. 이 메트릭은 독립적이지 않고 서로 다른 메트릭의 점수를 조합합니다. 그 과정에서 Gaussian Processes를 사용하여 메타 메트릭 기능을 모델링하고, 인간 평가 점수와의 상관관계를 최대화하는 가중치를 결정합니다.

- **Performance Highlights**: WMT24 metric shared task에서 MetaMetrics-MT는 기존 모든 기준선을 초과하는 성능을 보여주며, 참조(reference)-기반 설정에서 새 벤치마크를 수립했습니다. 또한, 참조-free 설정에서도 유사한 결과를 달성하여 효율성을 높였습니다. 이 메트릭은 40GB 메모리의 상업적 GPU에서 작동할 수 있는 반면, XCOMET-Ensemble과 같은 비교 메트릭은 최소 80GB의 높은 메모리를 요구합니다.



### STEM-POM: Evaluating Language Models Math-Symbol Reasoning in Document Parsing (https://arxiv.org/abs/2411.00387)
Comments:
          Accepted to NeurIPS Math-AI 2024

- **What's New**: 이 논문에서는 액기스 있는 수학 기호의 이해와 해석을 평가하기 위한 종합적인 벤치마크 데이터셋인 STEM-PoM을 소개합니다. 이 데이터셋은 실제 ArXiv 문서에서 추출된 2000개 이상의 수학 기호로 구성되어 있으며, LLM(large language models)의 수학적 추론 능력을 평가한다는 점에서 혁신적입니다.

- **Technical Details**: STEM-PoM 데이터셋은 변수, 상수, 연산자 및 단위 설명자로 분류된 2109개의 수학 기호를 포함하며, 각 기호에는 관련된 맥락의 텍스트 또는 표현이 포함되어 있습니다. 이러한 기호들은 첫 번째 및 두 번째 수준의 속성으로 분류되며, 각 기호는 변수, 상수, 연산자 또는 단위 설명자로 구분됩니다. 이 데이터셋은 여러 모델의 수학 기호 분류 작업을 평가하기 위해 고안되었습니다.

- **Performance Highlights**: 최신 LLM은 in-context learning에서 평균 20-60%의 정확도를 보이며, fine-tuning을 통해 50-60%의 정확도를 달성했습니다. 이는 이들 모델이 수학적 추론 능력에서 상당한 격차가 있음을 시사합니다. Mistral-8x7B와 같은 대형 모델은 문서의 맥락 길이가 증가함에 따라 성능이 크게 향상되며, Claude3.5-Sonnet은 모든 맥락 길이에서 GPT-4o와 유사한 성능을 유지합니다.



### GRS-QA -- Graph Reasoning-Structured Question Answering Datas (https://arxiv.org/abs/2411.00369)
Comments:
          15 pages, 24 figures, 10 tables

- **What's New**: 새롭게 소개된 GRS-QA 데이터셋은 질문-답변 쌍에 대한 명시적인 추론 구조를 제공하여 LLM의 다중 단계 추론 능력을 평가하는 데 필요한 명확한 경로를 제공합니다. 이 데이터셋은 다양한 추론 구조를 포함하고 있어 LLM의 성능을 세밀하게 분석할 수 있게 합니다.

- **Technical Details**: GRS-QA 데이터셋은 논리적 관계에 따라 문장을 노드로 처리하고, 그 연결을 나타내는 엣지를 추가하여 추론 그래프를 구축합니다. 그래프는 긍정 및 부정 추론 그래프로 나뉘며, 긍정 그래프는 올바른 논리적 단계에 따른 반면, 부정 그래프는 구조적 변형을 포함하여 추론 구조의 중요성을 조사합니다.

- **Performance Highlights**: 연구 결과, LLM은 서로 다른 추론 구조를 가진 질문을 처리할 때 성능 차이를 보였습니다. 이러한 발견은 추론 구조와 의미를 비교하는 탐색을 용이하게 하며, LLM의 성능을 세밀하게 분석하는 데 기여합니다.



### Learning to Rank Salient Content for Query-focused Summarization (https://arxiv.org/abs/2411.00324)
Comments:
          Long paper accepted at EMNLP 2024 (Main)

- **What's New**: 본 연구는 Learning-to-Rank (LTR) 기법과 Query-focused Summarization (QFS)을 통합하여 요약의 관련성을 향상시키는 가능성을 탐구합니다. 이를 통해 콘텐츠 우선 순위를 조정하여 더 나은 요약 결과를 도출하고 있습니다.

- **Technical Details**: 이 연구에서는 공유 디코더를 활용하여 요약 디코더와 함께 세그먼트 수준에서 LTR 작업을 수행합니다. SEGEnc 접근법을 기반으로 하여, LTR 원칙을 통합한 새로운 QFS 확장을 소개하며, 이를 통해 모델의 정보 중요도를 효과적으로 평가하고 우선 순위를 매길 수 있습니다.

- **Performance Highlights**: 우리의 시스템은 QMSum 벤치마크에서 모든 메트릭에서 우수한 성능을 보여주며, Rouge-L(+0.42) 및 BertScore(+0.34)에서 개선된 결과를 나타냅니다. SQuALITY 벤치마크에서는 Rouge-1 및 Rouge-2 점수에서 약간의 도전에 직면했지만, Rouge-L에서 +1.47의 현저한 성과를 보였습니다. 인간 평가 결과는 생성된 요약의 관련성과 신뢰성을 강조하고 있으며, 유창성을 희생하지 않고도 원활한 요약을 가능하게 하고 있습니다.



### Rationale-Guided Retrieval Augmented Generation for Medical Question Answering (https://arxiv.org/abs/2411.00300)
- **What's New**: 이번 연구에서는 RAG$^2$ (Rationale-Guided RAG)라는 새로운 프레임워크를 소개하여 생물 의학 분야에서 retrieval-augmented generation (RAG)의 신뢰성을 향상시킬 예정이다.

- **Technical Details**: RAG$^2$는 세 가지 주요 혁신을 포함: 1) perplexity 기반 라벨로 훈련된 소규모 필터 모델이 정보성 스니펫을 선택적으로 증강시키고 방해 요소를 걸러냄, 2) LLM이 생성한 이유(rationales)를 질의로 사용하여 검색된 스니펫의 유용성을 향상시킴, 3) 네 가지 생물 의학 코퍼스에서 스니펫을 균형 있게 검색하여 retriever의 편향을 줄이는 구조를 채택한다.

- **Performance Highlights**: RAG$^2$는 다양한 크기의 최신 LLM의 평균 정확도를 최대 6.1% 향상시켰으며, 세 가지 의료 질문-답변 벤치마크에서 이전의 최고 의료 RAG 모델을 최대 5.6% 초과하였다.



### LLM-Ref: Enhancing Reference Handling in Technical Writing with Large Language Models (https://arxiv.org/abs/2411.00294)
Comments:
          20 pages, 7 figures, submitted to ARR October 2024

- **What's New**: 이 논문에서는 LLM-Ref라는 새로운 글쓰기 보조 도구를 소개합니다. 이 도구는 사용자 제공 데이터에서 직접 정보를 검색하고 생성하는 방식으로, 전통적인 RAG 시스템이 가진 문제점을 개선합니다. 이를 통해 사용자는 연구 문서를 보다 효과적으로 작성할 수 있게 됩니다.

- **Technical Details**: LLM-Ref는 기존의 RAG 시스템과 달리 텍스트 조각(chunking)이나 인덱싱을 사용하지 않고, 연구 기사의 단락(paragraph)에서 직접 콘텐츠를 검색하고 생성합니다. 이 방법은 LLM의 제약을 관리하면서도 긴 문맥(context)을 처리하는 데 효과적이며, 계층적 섹션 구조를 보존합니다. 이 도구는 주 참고 문서 및 보조 참고 문서를 모두 제공할 수 있습니다.

- **Performance Highlights**: LLM-Ref는 기존 RAG 시스템에 비해 Ragas Score에서 3.25배에서 6.26배 향상된 성능을 보여줍니다. 또한, 단일 및 다중 출처 문서의 경우 각각 4.7배, 5.5배 더 높은 Context Relevancy 점수를 기록하였습니다. 이 도구는 더 정확하고 관련성 있으며 문맥적으로 적절한 출력을 제공하여 글쓰기의 유용성과 신뢰성을 높입니다.



### A Demonstration of Adaptive Collaboration of Large Language Models for Medical Decision-Making (https://arxiv.org/abs/2411.00248)
- **What's New**: MDAgents는 의료 결정 과정에서 LLM(대형 언어 모델)의 협업적 문제 해결 능력을 향상시키기 위해 복잡성과 요구 사항에 따라 협업 구조를 동적으로 할당하는 새로운 프레임워크입니다.

- **Technical Details**: MDAgents는 네 단계로 작동하며, 각 의료 쿼리에 맞춰 복잡성을 평가한 후 적절한 팀을 구성합니다. 분석 및 종합 단계에서는 Chain-of-Thought (CoT) 기법을 활용하고, 최종 결정 단계에서는 다양한 에이전트의 통찰을 활용하여 최종 답변을 종합합니다. 이 시스템은 MedRAG를 포함하여 최신 생의학 데이터를 통해 정확성을 향상시킵니다.

- **Performance Highlights**: MDAgents는 10개의 벤치마크 중 7개에서 최고의 정확도를 달성하였으며, 단일 LLM 및 정적 다중 에이전트 방법을 초월했습니다. 평균 추론 시간은 저복잡성(14.7초), 중복잡성(95.5초), 고복잡성(226초)으로, 3-에이전트 구성이 최적의 성능을 보였습니다. 또한, API 호출 횟수가 적어 계산 효율성이 높다는 특징이 있습니다.



### RESTOR: Knowledge Recovery through Machine Unlearning (https://arxiv.org/abs/2411.00204)
- **What's New**: 본 논문에서는 RESTOR 프레임워크를 제안합니다. RESTOR는 기계적 비학습(machine unlearning)의 새로운 관점을 제공하며, 문제의 데이터 포인트를 잊을 뿐만 아니라 모델이 원래의 상태로 복원되는 것을 강조합니다.

- **Technical Details**: RESTOR 프레임워크는 (1) 실제 세계의 사실적 지식에 초점을 맞춘 작업 설정, (2) 다양한 유형의 복원 시나리오, (3) 원래의 상태 복구를 평가하는 메트릭으로 구성됩니다. 논문에서는 다양한 해석이 가능한 사이클과 함께 데이터 손상 및 비학습 알고리즘의 효과를 평가하기 위해 여러 실험을 실시했습니다.

- **Performance Highlights**: 연구에 따르면 기존의 여러 비학습 방법들은 잊는 데는 뛰어난 성능을 보였지만, 원래의 지식 상태를 회복하는 데는 어려움을 겪고 있음이 드러났습니다. 특히, 특정 알고리즘은 손상된 데이터의 특정 부분에만 비학습을 적용하면 성능이 향상되는 결과를 보여주었습니다.



### Beyond Label Attention: Transparency in Language Models for Automated Medical Coding via Dictionary Learning (https://arxiv.org/abs/2411.00173)
- **What's New**: 이 논문은 의료 언어 모델에서 의료 코딩의 해석 가능성을 향상시키기 위해 딕셔너리 학습(Dictionary Learning) 기법을 활용하고 있습니다. 기존의 주의 메커니즘이 ICD 코드와 관련 없는 불필요한 토큰을 강조하는 문제를 해결하고, 의료 관련 개념을 효율적으로 캡처할 수 있는 해석 가능한 딕셔너리를 구축합니다.

- **Technical Details**: 저자는 두 가지 희소 오토인코더(sparse autoencoder) 접근 방식 중 하나인 L1 최소화와 SPINE 손실 함수의 사용을 통해 해석 가능한 표현을 생성하는 방법을 제시합니다. AutoCodeDL이라는 새로운 해석 가능성 프레임워크를 통해 학습된 딕셔너리 특성을 결합하여 다운스트림 ICD 예측의 설명 가능성을 향상시킵니다.

- **Performance Highlights**: 이 기술을 통해 저자는 학습된 딕셔너리의 특성이 모델 행동을 안내하고 90% 이상의 의료 관련 없는 토큰의 숨겨진 의미를 설명할 수 있음을 보여주며, 이는 사람들에게 이해할 수 있는 방식으로 이루어집니다.



### Scaling Up Membership Inference: When and How Attacks Succeed on Large Language Models (https://arxiv.org/abs/2411.00154)
Comments:
          Our code is available at this https URL

- **What's New**: 본 논문에서는 Membership Inference Attack (MIA)를 대규모 언어 모델(LLM)에 적용했을 때의 효과를 밝히고자 합니다. 이전 연구들에서는 MIA가 LLM에 대한 효과가 없다고 결론 내렸지만, 본 연구에서는 여러 문서를 동시에 테스트할 때만 MIA가 유효하다고 주장합니다.

- **Technical Details**: 우리는 MIA의 새로운 평가 기준을 도입하며, 데이터 샘플의 연속적 스케일을 측정하는 새로운 벤치마크를 구축했습니다. MIA 방법은 문장(n-grams)부터 문서(다수의 토큰의 조각)에 이르기까지 다양한 스케일에서 평가됩니다. 또한, 기존의 Dataset Inference (DI) 방법을 MIA에 맞게 조정하여 문서와 문서 집합 수준에서 성능을 평가할 수 있도록 하였습니다.

- **Performance Highlights**: MIA 방법은 문서 집합에 대해 80% 이상의 AUROC 점수를 달성하였습니다. 또한, 미세 조정 데이터에 대한 MIA 성능은 88% 이상의 AUROC를 기록하여 Continual Learning MIA가 효과적임을 입증하였습니다. 논문은 다양한 LLM 미세 조정 시나리오에서 성능 향상을 보여주며, 우리는 LLM에서 MIA 성능의 포괄적 분석을 수행하는 데 필요한 기반을 마련했습니다.



### Schema Augmentation for Zero-Shot Domain Adaptation in Dialogue State Tracking (https://arxiv.org/abs/2411.00150)
- **What's New**: 이 연구에서는 대화 상태 추적(Dialogue State Tracking, DST)에 대한 제로샷(domain adaptation) 접근 방식에서 애드혹한 대안으로 'Schema Augmentation' 기법을 도입한다. 이 기법은 슬롯 이름에 대한 변형을 도입하여 언어 모델의 제로샷 도메인 적응력을 크게 향상시킨다.

- **Technical Details**: Schema Augmentation은 기존 슬롯 이름에서 유의어(synonyms)를 사용하거나 비의미적 코드(non-semantic codes)로 교체하여 데이터 증강(data augmentation)을 수행한다. 연구에서는 Synonym Schema Augmentation(SSA)와 Encoding Schema Augmentation(ESA)의 두 가지 변형을 제안하며, 각 변형에는 단일(single) 및 다중(multi) 방식이 포함된다.

- **Performance Highlights**: MultiWOZ와 SpokenWOZ 데이터셋에서 실시한 실험 결과, 제안된 접근 방식은 기존 기준선 대비 최대 두 배의 정확도를 달성하였다. 특히, 보지 못한 도메인에 대해 정확도가 크게 향상되었으며, 모든 도메인에 대해 동등하거나 우수한 성능을 유지하였다.



### JudgeRank: Leveraging Large Language Models for Reasoning-Intensive Reranking (https://arxiv.org/abs/2411.00142)
- **What's New**: JudgeRank는 문서 관련성을 평가하기 위한 새로운 에이전틱 리랭커로, 인간의 인지 과정을 모방하여 문서 평가의 한계를 극복합니다. 이 접근법은 쿼리 분석, 문서 분석, 그리고 관련성 판단의 세 가지 주요 단계로 구성됩니다.

- **Technical Details**: 문서 요구 정보를 확보하기 위해 JudgeRank는 크게 세 가지 단계로 운영됩니다: (1) 쿼리 분석을 통해 핵심 문제를 파악하고, (2) 문서 분석에서 쿼리 인식을 반영한 요약을 추출하며, (3) 최종적으로 문서의 관련성을 간결하게 평가합니다. 이러한 방식은 Chain-of-Thought와 LLM-as-a-Judge 접근법에서 영감을 받았습니다.

- **Performance Highlights**: JudgeRank는 BRIGHT(Reasoning-Intensive Generative Retrieval Tasks) 벤치마크에서 첫 단계 검색 방법들보다 현저한 성능 향상을 보여주었으며, BEIR 벤치마크에서도 최신 리랭커와 동등한 성능을 발휘했습니다. 다양한 크기의 LLM에서 JudgeRank의 일반화 성능이 우수하게 나타났으며, 여러 모델을 앙상블 하였을 때 더욱 향상된 성능을 보였습니다.



### RSL-SQL: Robust Schema Linking in Text-to-SQL Generation (https://arxiv.org/abs/2411.00073)
- **What's New**: 이 논문에서는 Text-to-SQL 생성의 새로운 프레임워크인 RSL-SQL을 제안합니다. 이 프레임워크는 양방향 스키마 연결(bidirectional schema linking), 맥락 정보 보강(contextual information augmentation), 이진 선택 전략(binary selection strategy), 다회전 자기 수정(multi-turn self-correction)을 결합하여 성능을 향상시킵니다.

- **Technical Details**: RSL-SQL는 먼저 완전한 데이터베이스 스키마를 사용해 초기 SQL을 생성하고, 양방향 스키마 연결을 통해 높은 리콜(recall) 비율을 달성합니다. 그 다음, 데이터베이스 스키마를 단순화한 후 풍부한 맥락 정보로 강화하여 또 다른 SQL을 생성합니다. 이어서 이진 선택 전략을 통해 완전 스키마와 단순화된 스키마 중 더 나은 SQL을 선택합니다. 마지막으로 SQL 실행 결과의 피드백을 통합하여 잘못된 SQL 문을 반복적으로 수정합니다.

- **Performance Highlights**: BIRD와 Spider 데이터셋을 대상으로 한 실험에서 RSL-SQL은 각각 67.2% 및 87.9%의 실행 정확도를 기록하며, 기존 오픈소스 솔루션 중에서 최상의 성능을 달성했습니다. 또한 DeepSeek를 이용할 경우, RSL-SQL이 많은 GPT-4 기반 방법들보다 더 탁월한 성능을 보이면서 비용 효율성을 입증했습니다.



### Interpretable Language Modeling via Induction-head Ngram Models (https://arxiv.org/abs/2411.00066)
- **What's New**: 이 논문에서는 Induction-head ngram 모델(Induction-Gram)을 제안하여 고비용의 계산 환경에서도 효율적이고 해석 가능한 언어 모델을 구축합니다. 이를 통해 각 생성된 토큰에 대한 ngram 수준의 기반을 제공하며, 기존 모델보다 다음 단어 예측 성능을 크게 향상시킵니다.

- **Technical Details**: Induction-Gram은 현대 ngram 모델에 'induction head'라는 손수 설계한 요소를 추가하여 구현됩니다. 이 induction head는 사용자 지정된 신경 유사성 메트릭을 사용하여 모델 입력 컨텍스트에서 다음 단어 완료를 위한 잠재적인 제안을 효율적으로 검색합니다. 이 방법은 fMRI 반응 예측과 같은 자연어 뇌과학 설정에서도 차별화된 성능을 보입니다.

- **Performance Highlights**: Induction-Gram은 기준 해석 가능 모델에 비해 최대 26%p의 향상된 다음 단어 예측 성능을 보여주며, 자연어 fMRI 응답 예측에서도 20% 상대적 향상을 이끌어냅니다. 이를 통해 LLM의 추론 과정을 가속화할 수 있는 가능성을 엿봅니다.



### Evolving Alignment via Asymmetric Self-Play (https://arxiv.org/abs/2411.00062)
- **What's New**: 본 논문에서는 기존의 RLHF(frameworks for aligning large language models) 프레임워크의 한계를 극복하기 위해, 비대칭 게임(asymmetric game)으로 양자 간의 상호작용을 통해 프롬프트 분포(prompt distribution)를 점진적으로 발전시키는 새로운 RLHF 프레임워크인 eva(Evolving Alignment via Asymmetric Self-Play)를 제안합니다.

- **Technical Details**: eva는 두 플레이어(creator와 solver)가 상호작용하는 구조로, creator는 보상 모델(reward model)을 사용하여 더 정보가 많은 프롬프트를 생성하고, solver는 creator가 생성한 프롬프트에 대해 더 선호되는 응답(responses)을 생성하는 방식으로 진행됩니다. 이 구조는 기존의 RLHF 알고리즘을 효과적으로 활용할 수 있도록 설계되었습니다.

- **Performance Highlights**: eva는 Arena-Hard 벤치마크에서 Gemma-2-9B-it의 승률을 DPO로 51.6%에서 60.1%로, SPPO로 55.7%에서 58.9%로, SimPO로 52.3%에서 60.7%로, ORPO로 54.8%에서 60.3%로 향상시키는 등, 최신 기술과 비교할 때도 우수한 성능을 보였습니다.



### Generating Diverse Negations from Affirmative Sentences (https://arxiv.org/abs/2411.00056)
Comments:
          Accepted at "Adaptive Foundation Models: Evolving AI for Personalized and Efficient Learning" workshop at NeurIPS 2024

- **What's New**: NegVerse라는 새로운 방법을 제안하여 부정(Negation) 데이터 부족 문제를 해결하고 다양한 부정 유형을 생성합니다. 이 방법은 긍정적인 문장에서 부정을 생성하며, 구문 구조에 따라 부정이 가장 잘 발생할 부분을 마스킹하는 새로운 규칙을 제공합니다.

- **Technical Details**: 이 연구는 긍정적 문장으로부터 부정을 생성하는 NegVerse 방법을 도입합니다. 이 과정에서는 대체 문장 생성을 위해 GPT-2 기반의 모델을 활용하며, 문장을 유창하게 유지하도록 선택형 마스킹 전략을 사용합니다. 또한, 부정 신호를 판별하고 부적절한 예시를 제거하는 필터링 메커니즘을 제안합니다.

- **Performance Highlights**: 실험 결과, NegVerse는 기존 방법들보다 개선된 성능을 보이며 생성된 부정 문장이 원래 문장과 높은 어휘적 유사성을 유지하고, 구문적 보존 및 부정 다양성 측면에서 우수한 결과를 보여줍니다.



### ACC-Debate: An Actor-Critic Approach to Multi-Agent Deba (https://arxiv.org/abs/2411.00053)
- **What's New**: 이번 논문에서는 액터-비평가 기반의 학습 프레임워크인 ACC-Debate를 제안하여 두 에이전트 팀이 반복적인 대화를 통해 문제를 협력적으로 해결할 수 있도록 훈련하는 새로운 패러다임을 소개합니다.

- **Technical Details**: ACC-Debate는 두 개의 에이전트로 구성되어 있으며, 하나는 답변을 제공하는 액터(agent)이고, 다른 하나는 피드백을 제공하는 비평가(critic)입니다. 이 프레임워크는 ‘유도 대화(guided-debate)’라는 새로운 오프-정책 학습 방식을 도입하여, 고품질의 다중턴 훈련 데이터를 생성하고, 액터와 비평가의 성능을 향상시킵니다.

- **Performance Highlights**: ACC-Debate는 기존의 최첨단(debate techniques) 기술들보다 다양한 벤치마크에서 성능이 우수하다는 것을 입증하였습니다.



### Larger models yield better results? Streamlined severity classification of ADHD-related concerns using BERT-based knowledge distillation (https://arxiv.org/abs/2411.00052)
Comments:
          20 figures, 31 pages, review 1 from plos one journal

- **What's New**: 이번 연구는 지식 증류(knowledge distillation) 방법을 통해 경량화되었지만 성능이 뛰어난 BERT 기반 모델인 LastBERT를 개발하였습니다. 또한, 소셜 미디어 텍스트 데이터에서 주의력 결핍 과다행동 장애(ADHD) 관련 우려 사항의 심각도 수준을 분류하는 실제 세계 작업에 모델을 적용하였습니다.

- **Technical Details**: LastBERT 모델은 기본 BERT 모델의 매개변수를 1억 1천만에서 2천9백만으로 줄여 약 73.64% 더 작은 모델입니다. GLUE 벤치마크에서 LastBERT는 다양한 자연어 처리(NLP) 작업에서 뛰어난 성능을 유지하였고, ADHD 데이터셋에서 85%의 정확도 및 F1 점수를 달성하였습니다.

- **Performance Highlights**: LastBERT는 DistilBERT(6천6백만 매개변수) 및 ClinicalBERT(1억 1천만 매개변수)와 비교해 유사한 성능을 보였으나, DistilBERT가 87%로 약간 더 나은 성능을 기록하였습니다. 본 연구 결과는 LastBERT가 소셜 미디어에서 생성된 사용자의 콘텐츠를 이해하고 평가하는 데 유용한 도구로 작용할 수 있음을 보여줍니다.



### Rule by Rule: Learning with Confidence through Vocabulary Expansion (https://arxiv.org/abs/2411.00049)
Comments:
          29 pages, 8 figures

- **What's New**: 이 논문에서는 텍스트 기반 데이터에 특화된 혁신적인 iterative 접근 방식을 통해 규칙 학습을 제안합니다. 각 반복에서 사용하는 어휘를 점진적으로 확장하여 메모리 소비를 크게 줄이는 방법을 소개합니다. 또한, 생성된 규칙의 신뢰성을 나타내는 신뢰 수준(Value of Confidence)을 도입하여 가장 강력하고 신뢰할 수 있는 규칙만을 유지함으로써 규칙 학습 과정의 전반적인 품질을 향상시킵니다.

- **Technical Details**: 이 방법은 FOIL과 RIPPER와 같은 기존의 규칙 학습 알고리즘을 사용하며, 단순한 사전(dictionary)에서 시작하여 점진적으로 어휘를 확장함으로써 복잡한 텍스트 데이터를 처리할 수 있게 합니다. 초기에는 적은 수의 예를 고려하여 일반적인 규칙을 학습하고, 규칙이 학습될 때마다 긍정적인 예시는 감소하며, 이후 품질 기준에 따라 어휘를 확장합니다.

- **Performance Highlights**: 다양한 텍스트 및 비텍스트 데이터 세트에 대한 광범위한 실험을 통해 이 방법의 효과를 입증하였으며, 특히 보험 산업에 대한 사례 연구를 통해 실제 적용 가능성을 시연하였습니다. 결과적으로, 이 접근 방식은 매우 큰 데이터 세트에서도 실행 가능하며, 해석 가능성과 정확성 간의 균형을 유지하는 데 기여합니다.



### CurateGPT: A flexible language-model assisted biocuration too (https://arxiv.org/abs/2411.00046)
- **What's New**: 이 논문은 데이터 기반의 생물 의학 발견을 위한 효과적인 데이터 관리(data curation)의 중요성을 강조하고, 새로운 Generative AI 기술인 CurateGPT를 소개합니다.

- **Technical Details**: CurateGPT는 instruction-tuned large language models (LLMs)의 능력을 활용하여, 전문가들이 수작업으로 수행하던 과정을 자동화하고, 외부 정보 소스와 지식을 통합하는 데 도움을 줍니다. 이 시스템은 reasoning, ontology 검색 및 지식 통합과 같은 작업을 통해 효율성을 극대화합니다.

- **Performance Highlights**: CurateGPT는 LLM과의 직접 상호작용보다 더 나은 정보 접근성을 제공하며, 각 주장을 뒷받침하는 데이터에 직접 연결되는 링크를 제공합니다. 이러한 방식으로 연구자와 엔지니어는 방대한 과학 데이터의 증가 속도에 맞춰 더욱 효율적으로 데이터 관리를 확대할 수 있습니다.



### A Novel Psychometrics-Based Approach to Developing Professional Competency Benchmark for Large Language Models (https://arxiv.org/abs/2411.00045)
Comments:
          36 pages, 2 figures

- **What's New**: 본 논문은 Evidence-centered design (ECD) 방법론을 도입하여 교육 및 교수법 분야에서 새로운 벤치마크를 개발하는 접근법을 제안합니다. 현재의 벤치마크 개발 방법의 한계를 지적하고 LLM(대형 언어 모델)의 발전을 고려합니다.

- **Technical Details**: Bloom's taxonomy에 따라 구성된 새로운 벤치마크는 교육 전문가들이 rigorously 디자인하여 LLM에 맞춘 평가 도구를 제공합니다. 이 벤치마크는 현재 GPT 모델을 러시아어로 테스트하여 다양한 과제 복잡성에 따라 모델 성능을 평가합니다.

- **Performance Highlights**: 결과적으로, 생성 AI 도구는 교육에서 개인화된 튜터링, 실시간 피드백, 다국어 학습 등과 같은 과제를 지원할 수 있는 잠재력을 가지고 있지만, 깊은 인지적 참여가 필요한 과제와 같은 다양한 분야에서 자율적으로 교사를 보조하는 데에는 한계가 있음을 보여줍니다.



### MIMIC-IV-Ext-PE: Using a large language model to predict pulmonary embolism phenotype in the MIMIC-IV datas (https://arxiv.org/abs/2411.00044)
- **What's New**: 이 연구는 Pulmonary embolism (PE) 진단을 위한 새로운 접근 방식을 제시합니다. MIMIC-IV 데이터베이스를 활용하여 방사선 보고서에 PE 레이블을 추가하고, Bio_ClinicalBERT 언어 모델인 VTE-BERT를 적용하여 자동으로 레이블을 추출했습니다.

- **Technical Details**: 연구에서는 19,942명의 응급실 및 병원입원 환자의 computed tomography pulmonary angiography (CTPA) 방사선 보고서를 분석했습니다. VTE-BERT 모델을 통해 PE positive (급성 PE) 또는 PE negative로 레이블을 매겼으며, 이를 의료 전문가의 수작업 레이블링과 비교하여 모델의 신뢰성을 검증했습니다.

- **Performance Highlights**: VTE-BERT의 민감도(sensitivity)는 92.4%, 양성 예측 값(positive predictive value, PPV)은 87.8%로 나타났습니다. 반면, 진단 코드의 경우 민감도는 95.4%이고 PPV는 83.8%였습니다. 연구 결과, VTE-BERT가 방사선 보고서 분석에서 효과적인 도구임을 입증하였습니다.



### Problem Categorization Can Help Large Language Models Solve Math Problems (https://arxiv.org/abs/2411.00042)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)을 최적화하여 수학 문제를 빠르고 정확하게 해결하는 방법을 탐구합니다. 문제를 서로 다른 카테고리로 분류하는 접근 방식이 효과적임을 보여주며, 정확한 데이터 세트를 생성하여 분류 작업을 최적화합니다. 이는 LLM의 환각(hallucination) 문제를 완화하여 수학 문제 해결 능력을 향상시키는 데 기여합니다.

- **Technical Details**: 저자들은 LLM에 입력하기 전에 수학 문제를 대수(Algebra), 조합론(Combinatorics), 기하학(Geometry), 수론(Number Theory)의 네 가지 카테고리로 분류하는 방법을 개발합니다. 이 분류는 문제 해결 전략을 결정하는 데 사용되며, 두 가지 접근 방식인 체인 오브 생각(Chain of Thought, CT)과 프로그램 오브 생각(Program of Thought, PT)을 제안합니다. CT는 문제에 대한 단계별 해결 과정을 요구하고, PT는 Sympy 코드 기반으로 문제를 해결합니다. 이 방법을 통해 LLM의 문제 해결 정확도가 80% 이상의 정확도로 향상되었습니다.

- **Performance Highlights**: 제안된 접근 방식을 사용하면 문제 해결의 정확도가 무작위로 선택된 전략보다 67% 더 개선되었으며, 실제 카테고리를 기반으로 한 전략 선택보다 29% 낮지만 여전히 우수한 성능을 보였습니다. 이러한 결과는 문제를 정확하게 분류하고 최적의 전략을 결정하는 것이 문제 해결의 정확도를 크게 향상시킨다는 것을 나타냅니다.



### NeuroSym-BioCAT: Leveraging Neuro-Symbolic Methods for Biomedical Scholarly Document Categorization and Question Answering (https://arxiv.org/abs/2411.00041)
- **What's New**: 본 연구에서는 OVB-LDA라는 최적화된 주제 모델링 프레임워크와 BI-POP CMA-ES 최적화 기법을 통합한 새로운 접근 방식을 제안하여 생물 의학 논문 초록 분류의 정확성을 향상시키고 있습니다.

- **Technical Details**: 이 연구는 세 가지 구성으로 평가되며, 각 구성은 학술 문서 초록 검색, 금관 표준 학술 문서 초록, 그리고 금관 조각을 포함합니다. MiniLM 모델을 정상적으로 사용하여 주제 모델링과 고급 기계 학습 기술을 결합한 신경 상징적(answer extraction) 접근 방식을 활용하여 데이터를 정교하게 조정하면서도 높은 정확도로回答(answer extraction)를 수행할 수 있습니다.

- **Performance Highlights**: 기존의 RYGH 및 bio-answer finder와 같은 방법들을 뛰어넘는 성능을 보여주며, MiniLM이 작은 모델임에도 불구하고 복잡한 작업을 처리할 수 있는 경쟁력 있는 성능을 나타냅니다. 이 연구의 결과는 생물 의학 질문 응답 분야에서 초록의 유용성을 강조하며, 효율성과 정확성을 개선하기 위한 향후 연구 방향을 제시합니다.



### Linear Chain Transformation: Expanding Optimization Dynamics for Fine-Tuning Large Language Models (https://arxiv.org/abs/2411.00039)
Comments:
          9 pages, 2 figures, 4 tables

- **What's New**: 본 연구에서는 Linear Chain Transformation (LinChain)이라는 새로운 접근 방식을 제안하여, 사전 훈련된 대형 언어 모델(LLMs)을 특정 다운스트림 작업에 잘 적응하게 만들기 위해 최적화 동역학을 풍부하게 하는 일련의 선형 변환을 도입했습니다.

- **Technical Details**: LinChain은 파라미터 업데이트 과정에 여러 개의 선형 변환을 통합하여 업데이트의 효과적인 랭크를 확장하고, 복잡한 작업 별 표현을 학습하는 모델의 능력을 향상시킵니다. 이 방법은 고정된 저랭크(LoRA) 접근법의 한계를 극복하면서도 효율성을 유지합니다.

- **Performance Highlights**: LinChain은 다양한 벤치마크 작업에서 최첨단 방법에 비해 성능을 크게 향상시키며, 더 적은 학습 가능한 파라미터로 더 나은 일반화와 작업 적응을 유도하였습니다.



### Topic-Conversation Relevance (TCR) Dataset and Benchmarks (https://arxiv.org/abs/2411.00038)
Comments:
          To be published in 38th Conference on Neural Information Processing Systems (NeurIPS 2024) Track on Datasets and Benchmarks

- **What's New**: 이번 연구는 효과적인 회의 운영을 위해 대화를 주제에 맞추는 중요성을 강조하며, 1,500개의 회의와 2,200만 개 단어의 전사문이 포함된 포괄적인 Topic-Conversation Relevance (TCR) 데이터셋을 생성했습니다.

- **Technical Details**: TCR 데이터셋은 15,000개 이상의 회의 주제를 포함하고 있으며, GPT-4를 사용하여 긴 회의록을 사전 회의 아젠다 주제 스타일로 재작성합니다. 또한 다양한 회의 변형을 만들 수 있는 확장 가능한 스키마를 제공합니다.

- **Performance Highlights**: GPT-4를 활용하여 주제-대화 관련성을 이해하는 데 있어 모델의 정확도를 평가했습니다. 결과는 회의의 효과성을 높이는 데 기여할 수 있는 인사이트를 제공합니다.



### Is Our Chatbot Telling Lies? Assessing Correctness of an LLM-based Dutch Support Chatbo (https://arxiv.org/abs/2411.00034)
Comments:
          10 pages + 2 pages references, 4 figures

- **What's New**: AFAS는 고객 지원 팀의 개입 최소화 및 스스로 고객 문의에 대한 정확한 답변을 제공할 수 있는 AI 기반 챗봇을 개발하기 위해 노력하고 있습니다. 본 연구는 정확성(정확한 답변이란 무엇인지)을 정의하고, 이를 바탕으로 LLM 사용을 통한 고객 지원의 자동화를 목표로 하고 있습니다.

- **Technical Details**: 연구는 자연어 생성(Natural Language Generation) 및 자동 답변 평가 시스템(Automated Answer Grading)을 활용하여 AFAS 지원 팀이 의사 결정을 하는 방식을 모델링합니다. 연구 과정에서 수집된 데이터는 79개의 훈련 사례와 154개의 테스트 사례로 구성되어 있으며, 응답의 진실성을 평가하기 위한 휴리스틱(heuristics)과 맞춤형 메트릭(metrics)을 도출합니다.

- **Performance Highlights**: 제안된 모델은 55%의 정확도로 잘못된 응답을 식별할 수 있으며, 터키어와 영어의 번역된 텍스트에서 인간 평가와 비교했을 때 높은 상관 관계를 보였습니다. 이는 AFAS의 고객 서비스 품질 개선에 기여할 수 있는 가능성을 보여줍니다.



### WikiNER-fr-gold: A Gold-Standard NER Corpus (https://arxiv.org/abs/2411.00030)
- **What's New**: 본 논문에서는 다국어 Named Entity Recognition (NER) 코퍼스인 WikiNER의 품질을 논의하고, 개선된 버전인 WikiNER-fr-gold를 제공합니다. 이는 프랑스어 부분의 수정된 버전으로, 원본 프랑스어 하위 코퍼스의 20%를 무작위 샘플링하여 생성되었습니다.

- **Technical Details**: WikiNER는 반감독(semi-supervised) 방식으로 주석(annotation)이 생성되었으며, 사후 수동 검증이 이루어지지 않아 'silver-standard'(실버 스탠다드) 코퍼스로 분류됩니다. WikiNER-fr-gold는 26,818 문장과 700,000 토큰을 포함하는 원본 하위 코퍼스에서 선택된 무작위 샘플로 구성됩니다. 본 논문에서는 각 범주에 포함된 엔티티 유형을 요약하여 주석 가이드를 정의하고, 코퍼스를 수정하는 과정을 설명합니다.

- **Performance Highlights**: WikiNER-fr 코퍼스에서 관찰된 오류 및 일관성 부족에 대한 분석을 제공하며, 향후 연구 방향에 대한 논의도 포함됩니다. 이러한 분석을 통해 NER 시스템의 품질 향상 가능성을 제시합니다.



### Preserving Pre-trained Representation Space: On Effectiveness of Prefix-tuning for Large Multi-modal Models (https://arxiv.org/abs/2411.00029)
Comments:
          Findings of EMNLP 2024

- **What's New**: 최근 대규모 다중 모달 모델(Large Multi-modal Models, LMMs)의 발전을 통해 기계와 세계 간의 상호작용 방식이 혁신적으로 변화하고 있습니다. 이러한 모델을 하위 작업에 적합하도록 조정하기 위해 파라미터 효율적인 미세 조정(Parameter-efficient fine-tuning, PEFT) 기술이 인기를 얻고 있으며, 본 논문에서는 PEFT의 작동 방식에 대한 심층 분석을 제공합니다.

- **Technical Details**: 이 연구에서는 두 단계로 구성된 PT-PEFT(Prefix-Tuned PEFT) 방법을 제안합니다. PT-PEFT는 먼저 prefix-tuning을 수행한 후, 이후에 PEFT 방법(예: Adapter, LoRA)을 통해 모델 파라미터를 조정합니다. 이를 통해 사전 훈련된 지식의 활용을 극대화합니다. 특히 본 연구에서는 singular value decomposition (SVD)을 사용하여 feature representation matrices의 변화를 분석했습니다.

- **Performance Highlights**: PT-PEFT는 이미지 캡셔닝(Image Captioning, IC) 및 시각적 질문 응답(Visual Question Answering, VQA) 작업에서 기존 PEFT 방법에 비해 성능을 개선하는 것으로 나타났습니다. 본 논문에서 관련된 네 가지 사전 훈련 모델을 대상으로 실험한 결과, PT-PEFT가 representation space를 보존하면서 전반적인 성능을 향상시키는 데 기여함을 확인하였습니다.



### Synergizing LLM Agents and Knowledge Graph for Socioeconomic Prediction in LBSN (https://arxiv.org/abs/2411.00028)
- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)과 지식 그래프(Knowledge Graph)를 결합해 위치 기반 소셜 네트워크(LBSN) 데이터를 활용한 사회경제 예측(socioeconomic prediction)을 수행할 수 있는 새로운 접근 방식을 제안합니다. 이러한 접근법은 서로 다른 Task 간의 지식 공유를 통해 예측 성능을 향상시킵니다.

- **Technical Details**: 연구자는 위치 기반 지식 그래프(LBKG)를 구축하여 다양한 출처의 LBSN 데이터를 통합하고, LLM 에이전트를 활용해 각 사회경제 예측 태스크에 적합한 메타 경로(meta-path)를 자동으로 추출합니다. 이 과정에서 의미론적 안내(attention)가 포함된 융합 모듈을 설계하여 메타 경로를 기반으로 한 지식을 융합합니다. 또한 교차 태스크 통신 메커니즘을 도입하여 태스크 간 지식을 공유하고, LLM 에이전트와 KG 수준에서 성능을 개선합니다.

- **Performance Highlights**: 두 개의 도시 규모 데이터셋에서 실험을 진행한 결과, 제안된 모델은 기존 방법보다 2.9%에서 74.2%까지 R^2 성능이 향상되는 것을 확인했습니다. 이는 LLM과 KG 간의 협력적 모델 설계의 효과성을 나타냅니다.



### Personalization of Large Language Models: A Survey (https://arxiv.org/abs/2411.00027)
- **What's New**: 이 연구에서는 개인화된 대형 언어 모델(LLMs)의 사용을 위한 새로운 분류 체계를 제안하고, 개인화된 LLM의 여러 측면과 이와 관련된 주요 과제들을 종합적으로 정리합니다. 이를 통해 개인화된 LLM 연구의 두 가지 주요 방향을 연결하고, 사용자 요구에 맞춘 응답 생성을 가능하게 하는 접근법을 논의합니다.

- **Technical Details**: 개인화된 LLM의 활용을 위해 제안된 세 가지 개인화 세분화 수준은 사용자 수준 개인화(user-level personalization), 페르소나 수준 개인화(persona-level personalization), 글로벌 선호 정렬(global preference alignment)로 나뉘며, 각 수준의 장단점과 필요한 데이터 양에 대해 논의합니다. 또한, 개인화에 대한 다양한 기술(예: retrieval-augmented generation, prompt engineering, reinforcement learning from human feedback)과 평가 지표를 체계적으로 분류합니다.

- **Performance Highlights**: 이 논문은 개인화된 LLM의 다양한 사용 사례를 다루며, 특히 교육, 헬스케어, 금융 및 추천 시스템 분야에서 개인화된 LLM이 사용자 경험을 향상시키고 작업 수행 성과를 개선할 수 있는 잠재력을 강조합니다. 향후 연구에 대한 중요한 과제와 문제들을 제시하며, 개인화된 LLM의 효과적인 평가를 위한 기준과 지표 개선의 필요성을 강조합니다.



### A Perspective for Adapting Generalist AI to Specialized Medical AI Applications and Their Challenges (https://arxiv.org/abs/2411.00024)
- **What's New**: 의료 애플리케이션에 LLM(대규모 언어 모델)의 통합이 활발해지고 있으며, 그 중에서도 약물 발견, 임상 의사결정 지원 및 원격의료 지원에 대한 최신 동향이 소개되었습니다.

- **Technical Details**: 의료 LLM 연구 활동을 위한 3단계 프레임워크가 제안되었습니다: 1) Modeling: 복잡한 의료 작업을 관리 가능한 단계로 나누어 의학 전문 모델 개발; 2) Optimization: 맞춤형 프롬프트를 통한 모델 성능 최적화 및 외부 지식 통합; 3) System engineering: 복잡한 작업을 하위 작업으로 나누고 인간 전문 지식 활용.

- **Performance Highlights**: LLM 기반 의료 AI 애플리케이션, 임상 시험 설계 최적화, 임상 의사결정 지원 강화 및 의료 영상 분석 향상과 같은 다양한 사용 사례를 통해 LLM의 성능이 입증되었습니다.



### Freeze-Omni: A Smart and Low Latency Speech-to-speech Dialogue Model with Frozen LLM (https://arxiv.org/abs/2411.00774)
Comments:
          Project Page: this https URL

- **What's New**: 최근 대화형 인공지능 모델인 Freeze-Omni가 제안되었습니다. 이 모델은 LLM(대규모 언어 모델)의 파라미터를 고정(frozen)한 상태에서 음성 입력과 출력을 연결하는 멀티모달 아키텍처를 갖추고 있습니다. 이를 통해 음성 대화 능력을 획득하면서도 기존 LLM의 지능을 유지합니다.

- **Technical Details**: Freeze-Omni는 음성 인코더와 디코더로 구성되어 있습니다. 모델의 훈련 과정에서 첫 단계는 ASR(data for automatic speech recognition) 데이터를 사용해 음성을 텍스트로 변환하는 것입니다. 이후, 텍스트-음성이 결합된 데이터를 활용해 출력 음성을 생성합니다. 이 모델은 음성-음성 대화 구조를 통해 사용자의 입력을 처리하고 유연한 대화가 가능하도록 설계되어 있습니다.

- **Performance Highlights**: Freeze-Omni는 훈련에 필요한 데이터 양이 적고, 계산 리소스를 절약합니다. 본 모델은 8개의 GPU에서 60,000개의 다중 회전 Q&A 데이터만으로 효과적인 음성 대화 기능을 달성했습니다. 낮은 지연(latency) 시간을 유지하며, 텍스트 모드에서의 지능도를 손상시키지 않고 음성 모드에서도 유사한 성능을 발휘합니다.



### CORAG: A Cost-Constrained Retrieval Optimization System for Retrieval-Augmented Generation (https://arxiv.org/abs/2411.00744)
- **What's New**: 이 논문에서는 승수 최적화를 위한 RAG(Retrieval-Augmented Generation) 시스템을 제안하며, 특히 여러 청크(chunk) 간의 상관관계 및 청크 유틸리티의 비단조성(non-monotonicity)을 완벽하게 고려한 방법을 설명합니다.

- **Technical Details**: 제안된 CORAG 시스템은 몬테카를로 트리 검색(Monte Carlo Tree Search, MCTS) 기반의 정책 프레임워크를 활용하여 청크 조합을 최적화합니다. 이 방법은 청크 간의 상관관계를 고려하여 청크 조합의 순서를 정할 수 있도록 합니다. 또한, 단순히 예산 소진을 종료 조건으로 보는 대신, 예산 제약을 청크 조합의 최적화에 통합하여 비단조성 문제를 해결합니다.

- **Performance Highlights**: 실험 결과, CORAG는 기존의 RAG 시스템보다 청크 선택의 중복성을 현저히 줄이며, 다양한 사용자 쿼리 유형에 더 적응력 있는 성능을 보여주었습니다. 이는 LLM의 응답 품질을 향상시키는 데 기여합니다.



### Decoding Dark Matter: Specialized Sparse Autoencoders for Interpreting Rare Concepts in Foundation Models (https://arxiv.org/abs/2411.00743)
- **What's New**: 본 논문에서는 Specialized Sparse Autoencoders (SSAEs)를 제안하여, Foundation Models (FMs)에서 특정 하위 도메인에 관련된 희귀한 특징들을 효율적으로 추출하는 새로운 방법을 소개합니다. 이러한 접근 방식은 기존의 Sparse Autoencoders (SAEs)보다 더 많은 tail concepts를 캡처하는데 집중합니다.

- **Technical Details**: SSAEs는 unsupervised targeted 방법으로, 특정 하위 도메인에 맞춰 훈련된 스파스 오토인코더입니다. SSAEs 훈련 시 Dense retrieval 기법을 활용하여 관련 훈련 데이터를 선택하고, Tilted Empirical Risk Minimization (TERM)를 학습 목표로 설정하여 tail 개념의 표현을 개선합니다.

- **Performance Highlights**: SSAEs는 Bias in Bios 데이터셋에서 12.5%의 worst-group classification accuracy 향상을 보이며, 전반적인 성능에서도 일반적인 SAEs를 초월한 결과를 나타냅니다. 이는 SSAEs가 하위 도메인에서의 FMs의 내부 작용을 보다 깊이 있게 탐구할 수 있는 강력한 도구가 됨을 의미합니다.



### TaxaBind: A Unified Embedding Space for Ecological Applications (https://arxiv.org/abs/2411.00683)
Comments:
          Accepted to WACV 2025

- **What's New**: TaxaBind는 모든 관심 있는 종을 특성화하기 위한 통합 임베딩 공간을 제시합니다. 이는 종의 지상 이미지, 지리적 위치, 위성 이미지, 텍스트, 오디오, 환경적 특징을 포함한 6가지 모달리티를 활용하는 다중모달(Multimodal) 임베딩 공간입니다.

- **Technical Details**: TaxaBind는 다양한 모달리티의 지식을 효과적으로 증류하기 위한 기술로서 multimodal patching을 제안합니다. 이는 각 모달리티의 고유한 정보 손실을 방지하며, 다양한 생태적 작업을 위한 학습을 가능하게 합니다. 또한, iSatNat과 iSoundNat이라는 두 가지 대형 데이터셋을 구축하고, TaxaBench-8k라는 평가를 위한 진정한 다중모달 데이터셋을 소개합니다.

- **Performance Highlights**: TaxaBind는 다음과 같은 작업에서 강력한 제로샷(zero-shot) 및 돌발(emergent) 능력을 보였습니다: 종 분류(Species Classification), 교차 모델 검색(Cross-model Retrieval), 오디오 분류(Audio Classification). 추가적으로, 여러 벤치마크 및 제로샷 작업에서 모델의 효과성과 성장 가능성을 입증하였습니다.



### Optimizing Contextual Speech Recognition Using Vector Quantization for Efficient Retrieva (https://arxiv.org/abs/2411.00664)
Comments:
          14 pages, 7 figures, submitted to IEEE/ACM Transactions on Audio, Speech, and Language Processing

- **What's New**: 본 연구는 음성 인식 모델이 맥락 정보를 효과적으로 활용할 수 있도록 하는 신경 맥락 바이어스(Neural Contextual Biasing) 기술을 개선합니다. 일반적으로 사용되는 크로스 어텐션(cross-attention) 모듈의 계산 복잡도가 바이어스 카탈로그(biasing catalogue)의 크기에 제한을 두는 문제를 해결하는 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 벡터 양자화(vector quantization)를 기반으로 한 크로스 어텐션 스코어링의 근사치를 도입하며, 대규모 바이어스 카탈로그의 연산 및 메모리 사용을 효율적으로 수행할 수 있게 합니다. 우리는 효율적인 양자화된 검색 모듈을 사용하여 오디오에 기반하여 바이어스 항목을 선별(shortlist)하고, 이를 사용하여 바이어싱합니다. 이 접근법은 바이어스 방법에 독립적이며, 전체 크로스 어텐션(full cross-attention)과 LLM 프롬팅(LLM prompting) 방법을 결합하여 사용할 수 있습니다.

- **Performance Highlights**: 제안된 검색 기반 선별 방법은 시스템이 수 천 개의 바이어스 카탈로그를 효율적으로 활용하도록 하여 개인 엔티티 인식에서 상대 오류율(relative error rate)을 최대 71% 감소시키는 성과를 나타냅니다. 또한, 제안된 근사 알고리즘은 표준 점곱 크로스 어텐션과 비교할 때 최대 100만 개 항목에 대해 계산 시간을 20% 단축시키고 메모리 사용량을 85-95%까지 줄입니다.



### Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations (https://arxiv.org/abs/2411.00640)
Comments:
          14 pages

- **What's New**: 이 논문은 대형 언어 모델(LLM)에 대한 평가를 위한 새로운 통계적 접근 방식을 제안합니다. 특히, 연구자들이 언어 모델 평가에서 신뢰 구간(confidence intervals)을 계산하고 결과를 보고하는 방법에 대한 구체적인 권고사항을 제시합니다.

- **Technical Details**: 저자들은 평가의 질문이 보이지 않는 슈퍼 모집단에서 뽑혔다고 가정합니다. 이를 바탕으로 평가 데이터 분석을 위한 공식을 제시하고, 두 모델 간의 차이를 측정하며 평가 실험의 계획을 세우는 방법을 기술합니다. 또한, 평균의 표준 오차(standard errors)를 중앙극한정리(Central Limit Theorem)를 통해 계산하고, 질문 수준에서 통계적 추론을 수행할 것을 권장합니다.

- **Performance Highlights**: 논문에서는 Galleon과 Dreadnought라는 두 모델을 비교하는 가상의 사례를 통해 통계적 해석의 중요성을 강조합니다. Galleon은 수학적 추론 평가에서 더 높은 점수를 기록했으나, Dreadnought는 프로그래밍 평가에서 더 잘 수행했습니다. 이러한 상반된 결과를 바탕으로 통계적 신뢰성을 갖춘 평가 방식의 필요성을 제기합니다.



### Adapting While Learning: Grounding LLMs for Scientific Problems with Intelligent Tool Usage Adaptation (https://arxiv.org/abs/2411.00412)
Comments:
          26 pages, 15 figures

- **What's New**: 이 연구는 기존의 Large Language Models (LLMs)의 한계를 극복하기 위한 새로운 접근 방식을 제안합니다. 특히, 문제의 복잡성을 분석한 후 적절한 솔루션 접근 방식을 선택하는 인간 전문가의 방식에서 영감을 받아, LLM을 위한 두 가지 구성 요소로 이루어진 새로운 fine-tuning 방법을 제시합니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 구성 요소로 나뉩니다. 첫 번째는 World Knowledge Distillation (WKD)로, LLM이 도구 정보를 활용해 생성된 솔루션으로부터 직접 학습하여 도메인 지식을 내재화하는 과정입니다. 두 번째는 Tool Usage Adaptation (TUA)로, 모델의 직접 응답 정확성을 기준으로 문제를 쉬운 문제와 어려운 문제로 나누어, 쉬운 문제에서는 WKD와 동일한 정렬 목표를 유지하면서도 어려운 문제에 대해서는 도구 사용으로 전환할 수 있도록 훈련합니다.

- **Performance Highlights**: 이 방법은 수학, 기후과학, 역학을 포함한 여섯 가지 과학 벤치마크 데이터셋에서 검증되었습니다. 평균적으로, 제안된 모델은 모든 데이터셋에서 정답 정확성이 28.18%, 도구 사용의 정밀도가 13.89% 향상되었으며, GPT-4o 및 Claude-3.5와 같은 최첨단 모델을 초월하는 성과를 보였습니다.



### Compositional Automata Embeddings for Goal-Conditioned Reinforcement Learning (https://arxiv.org/abs/2411.00205)
- **What's New**: 본 논문에서는 goal-conditioned reinforcement learning (RL)에서 cDFA(compositional Deterministic Finite Automata)를 활용하여 시간적 목표 표현을 제안합니다. 기존의 목표 표현이 지닌 한계를 극복하기 위해, cDFA를 사용하여 RL 에이전트를 유도함으로써 명확한 시간적 의미론을 제공하며 해석을 용이하게 합니다.

- **Technical Details**: cDFA는 DFAs의 조합으로 정의되어 있으며, 목표 조건 RL 에서 시간적 작업을 처리하는 데 적합합니다. 또한, 본 연구에서는 cDFA의 인코딩을 위해 그래프 주의망(GATv2)을 사용하는 방법을 제안하고, reach-avoid derived (RAD) DFA에 대한 사전 학습을 통해 RL 에이전트가 다양한 cDFA 작업 클래스를 제로샷 제너럴리제이션(zero-shot generalization)할 수 있도록 합니다.

- **Performance Highlights**: 실험을 통해 제안된 사전 학습 방법이 다양한 cDFA 작업 클래스에 대해 제로샷 제너럴리제이션을 가능하게 하고, 계층적 방법의 단점인 단기 최적성(myopic suboptimality)을 극복하는 정책 전문화(policy specialization)를 가속화함을 입증하였습니다.



### LLM4Mat-Bench: Benchmarking Large Language Models for Materials Property Prediction (https://arxiv.org/abs/2411.00177)
Comments:
          Accepted at NeurIPS 2024-AI4Mat Workshop. The Benchmark and code can be found at: this https URL

- **What's New**: 이번 논문에서는 LLM4Mat-Bench를 소개하여 결정성 물질의 특성을 예측하는 대규모 리서치 벤치마크 데이터셋을 제공하고 있습니다. LLM4Mat-Bench는 약 190만 개의 결정 구조를 포함하고 있으며, 10개의 공개된 재료 데이터 소스에서 수집되었습니다.

- **Technical Details**: LLM4Mat-Bench는 다양한 입력 모달리티를 활용하여 4.7M, 615.5M, 및 3.1B 개의 토큰을 제공합니다. 연구진은 LLM-Prop, MatBERT와 같은 다양한 크기의 모델을 미세 조정하고, Llama, Gemma, Mistral과 같은 모델의 속성을 평가하기 위해 zero-shot 및 few-shot 프롬프트를 제공했습니다.

- **Performance Highlights**: 결과는 일반 목적으로 설계된 LLM이 재료 과학에서 겪는 도전 과제와 재료 속성 예측에서 작업 특화 모델 및 지침 조정된 LLM의 필요성을 강조합니다.



### Device-Directed Speech Detection for Follow-up Conversations Using Large Language Models (https://arxiv.org/abs/2411.00023)
- **What's New**: 이 연구에서는 Virtual Assistant (VA)와의 대화에서 후속 쿼리의 정확한 Device-directed Speech Detection (DDSD)에 대한 필요성을 강조합니다. 특히, 기존의 단일 쿼리 감지 접근법 대신, Large Language Models (LLMs)를 활용하여 후속 쿼리와 초기 쿼리의 문맥을 결합하여 정확도를 높이는 방법을 제안합니다.

- **Technical Details**: 기존 DDSD 시스템의 한계를 보완하기 위해, 초점은 두 가지 접근법에 있습니다: (i) 프롬프트 방식으로 미리 훈련된 LLM을 텍스트 기반으로 직접 사용하는 방법, (ii) LLM의 위에 이진 분류기를 추가하여 확률적 결정을 내리는 방법입니다. ASR (Automatic Speech Recognition) 불확실성을 활용하여 쿼리 문맥을 강화하고, n-best ASR 히포세시스를 통해 활용하여 더 많은 정보를 제공합니다.

- **Performance Highlights**: 실험 결과, 공동 모델링을 통해 DDSD 정확도가 약 20-40% 향상된 것을 보여줍니다. 전통적인 방식 대비 후속 쿼리만을 사용했을 때보다 훨씬 높은 성능을 나타냅니다. 이 연구는 실제 데이터 세트를 통해 그 효과를 입증하였습니다.



### SFM-Protein: Integrative Co-evolutionary Pre-training for Advanced Protein Sequence Representation (https://arxiv.org/abs/2410.24022)
- **What's New**: 본 연구에서는 아미노산 잔기 간의 상호작용을 강조하는 새로운 단백질 기초 모델의 사전 훈련 전략을 제안합니다. 이를 통해 단백질 서열 데이터로부터 단기 및 장기 공진화(co-evolutionary) 특징을 더욱 효과적으로 추출할 수 있습니다.

- **Technical Details**: 제안된 모델은 대규모 단백질 서열 데이터셋에서 훈련되었으며, 복잡한 상호작용을 반영하는 통합 손실 함수(integrative loss function)를 사용하여 단기 및 장기 공진화 정보를 효과적으로 포착합니다. 이러한 접근법은 진화 정보를 보다 잘 모델링하여 단일 서열 모델과 MSA 기반 방법의 성능 차이를 줄이고자 합니다.

- **Performance Highlights**: 우리의 모델은 다양한 하위 작업에서 기존 모델들보다 우수한 성능을 보였으며, 특히 ESM 모델 같은 유사한 크기의 기준 모델들과 비교하여 뛰어난 일반화 능력을 입증했습니다. 실험 결과를 통해 공진화 정보 통합의 효과성을 확인했습니다.



New uploads on arXiv(cs.IR)

### Making Sense of Metadata Mess: Alignment & Risk Assessment for Diatom Data Use Cas (https://arxiv.org/abs/2411.00677)
Comments:
          13 pages, 2 figures, 1 table, to be published in MTSR 2024 conference proceedings

- **What's New**: 이 논문은 Drexel University의 Academy of Natural Sciences의 Diatom Herbarium의 디지털 컬렉션 접근성을 높이기 위해 진행된 메타데이터 연구 결과를 보고합니다. 이 연구는 Diatoms(규조류) 보존을 위한 기존 방법을 디지털화하면서 나타나는 메타데이터의 도전과 기회를 다룹니다.

- **Technical Details**: 논문은 세 가지 주요 연구 결과를 포함합니다: 1) Hammer 외(2018)가 제안한 microscopy 메타데이터 프레임워크 및 관련 표준 검토, 2) 현재 규조류 메타데이터 속성을 표준 메타데이터 유형에 매핑한 기초 메타데이터 정렬, 3) 표준 데이터 큐레이션 관행과 관련된 메타데이터 위험 분석. 이 연구는 DataFed 시스템을 통해 디지털 슬라이드의 전환과 연계되었습니다.

- **Performance Highlights**: 시스템의 독특한 조합 덕분에 메타데이터 생성 및 처리 작업이 내부에서 이루어졌습니다. 새로운 연구 프로젝트는 ANS Diatom Herbarium 접근성을 높이고 데이터 공유를 개선하기 위한 국제 협력의 일환으로 진행되고 있습니다.



### Enhancing Semantic Interoperability Across Materials Science With HIVE4MA (https://arxiv.org/abs/2411.00676)
Comments:
          11 pages, 1 figures, 3 tables, to be published in SeMatS 2024 workshop proceedings

- **What's New**: HIVE4MAT는 재료 과학에 유용한 온톨로지를 탐색하기 위한 연결된 데이터 기반의 인터랙티브 애플리케이션입니다. 이 시스템은 표준화된 용어로 텍스트 자원을 자동으로 색인할 수 있는 기능을 제공합니다.

- **Technical Details**: HIVE4MAT는 RDF 기반 포맷을 SKOS 스키마로 변환하여 사용자가 쉽게 온톨로지를 탐색할 수 있도록 설계되었습니다. 몬티의 SKOS는 고수준 개념에서 하위 개념까지 계층적으로 탐색할 수 있는 기능을 제공합니다. 또한 HIVE4MAT는 메타데이터 표현을 위한 표준화된 용어를 제공하고 자동 색인 기능을 지원합니다.

- **Performance Highlights**: HIVE4MAT는 탐색, 검색, 색인화 기능이 통합되어 있어 사용자들이 다양한 온톨로지를 효과적으로 비교하고 탐색할 수 있도록 돕습니다. 특히, 사용자는 자연어 처리(NLP)를 활용하여 텍스트를 자동으로 색인하고 해당 지역에 맞는 표준 온톨로지 용어를 선택할 수 있습니다.



### LLM-KT: A Versatile Framework for Knowledge Transfer from Large Language Models to Collaborative Filtering (https://arxiv.org/abs/2411.00556)
Comments:
          accepted at ICDM 2024 (demo track)

- **What's New**: 본 연구에서는 LLM-KT라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 Collaborative Filtering (CF) 모델에 LLM (Large Language Model)에서 생성된 특징을 통합하여 모델의 성능을 향상시키도록 설계되었습니다. 기존 방법들과 달리, LLM-KT는 LLM에서 생성된 특징을 직접 입력으로 사용하는 것이 아니라 CF 모델의 중간 레이어에 주입하여 모델이 내부적으로 이를 재구성하고 활용하도록 합니다.

- **Technical Details**: LLM-KT 프레임워크는 모델 아키텍처를 변경하지 않고 CF 모델의 특정 내부 레이어에서 사용자 선호도를 재구성하는 데 중점을 둡니다. 이 과정은 LLM을 사용하여 각 사용자에 대한 짧은 선호 요약인 '프로필'을 생성하여, 텍스트 임베딩 모델을 사용해 이를 밀집 임베딩으로 변환한 후, 재구성 손실과 모델 손실의 가중합을 사용하여 훈련하는 방식으로 이루어집니다.

- **Performance Highlights**: MovieLens 및 Amazon 데이터셋을 기반으로 한 실험 결과, LLM-KT는 기존 CF 모델보다 최대 21%까지 성능을 개선함을 보여줍니다. 또, 이 모델은 context-aware 설정에서도 최첨단 방법들과 경쟁력이 있으며, 기존 방법들과 비교하여 더욱 다양한 CF 모델에 적용될 수 있는 가능성을 보입니다.



### DivNet: Diversity-Aware Self-Correcting Sequential Recommendation Networks (https://arxiv.org/abs/2411.00395)
Comments:
          Published at CIKM

- **What's New**: 이 논문에서는 복합 추천 시스템의 최종 단계에서 전체 페이지의 다양성과 관련성을 최적화하기 위한 새로운 접근법인 	extit{Diversity-aware Self-Correcting Sequential Recommendation Networks} (DivNet)를 제안합니다. DivNet는 항목 간의 복잡한 상호작용을 캡처하여 공통의 추천 세트를 다변화할 수 있는 기능을 가지고 있습니다.

- **Technical Details**: DivNet는 아이템과 사용자 특성을 컨텍스트 아이템과 함께 숨겨진 공간으로 투영하는 자기 주의 네트워크(self-attention networks)를 활용합니다. 이어서 선택된 아이템의 영향을 고려하여 다음 아이템을 선택하고, 이 과정에서 항목의 효용을 추정하며 후보 항목과 기존 항목 간의 유사성을 계산합니다. 자기 수정 모듈은 총 효용을 극대화하고 선택된 항목 간의 다양성을 향상시키는 것을 목표로 합니다.

- **Performance Highlights**: 산업 응용에서의 방대한 배제 테스트 결과와 상업 플랫폼에서의 배포 실험에 따르면, DivNet는 집합 추천이 포함되거나 포함되지 않은 경우 모두 기존 기법들보다 우수한 성능을 보여줍니다. 또한, 이 방법은 실제 추천 시스템에서 대규모 쿼리를 지원할 수 있는 실용성을 가지고 있습니다.



### A Survey on Bundle Recommendation: Methods, Applications, and Challenges (https://arxiv.org/abs/2411.00341)
- **What's New**: 본 논문은 번들 추천 시스템(Bundle Recommendation System)의 최근 발전을 종합적으로 검토하고, 제품 번들링(Product Bundling)의 분류 체계 및 전략을 제시합니다. 번들 추천을 구별(Discriminative)과 생성(Generative)하는 두 가지 카테고리로 나누고, 각 카테고리의 방법론을 체계적으로 분석합니다.

- **Technical Details**: 번들 추천은 사용자에게 관련성 높은 아이템 집합을 추천하여 사용자 경험을 개선하고 판매를 증가시키는 기능을 합니다. 두 가지 접근 방식으로는 1) 아이템 수준(Item Level)과 번들 수준(Bundle Level)의 표현 학습(Representation Learning) 및 상호작용 모델링(Interaction Modeling)과 2) 아이템 수준에서의 표현 학습과 번들 생성(Bundle Generation)이 있습니다.

- **Performance Highlights**: 번들 추천 시스템은 사용자 맞춤형 아이템 세트를 제안하여 정보 과부하를 줄이고 사용자 만족도를 높입니다. 다양한 도메인에서 활용되며, 특히 전자상거래에서의 판매 증대에 기여하고 있습니다. 연구자들은 성과 향상을 위해 많은 연구를 진행하고 있으며, 관련 데이터셋과 코드는 공개되어 있습니다.



### Beyond Utility: Evaluating LLM as Recommender (https://arxiv.org/abs/2411.00331)
- **What's New**: 이번 논문은 LLM(대형 언어 모델) 기반 추천 시스템의 새로운 평가 차원을 제안하며, 추천 모델의 성능 평가에 있어 정보 서비스의 정확성을 넘어서 다양한 측면을 고려함.

- **Technical Details**: LLM 기반 추천 시스템의 성능을 평가하기 위해 1) history length sensitivity, 2) candidate position bias, 3) generation-involved performance, 4) hallucinations의 네 가지 새로운 평가 차원을 포함하는 다차원 평가 프레임워크를 제안함. 이 프레임워크는 전통적인 평가 측면과 함께 사용되어 LLM의 성능을 전통적인 모델들과 비교함.

- **Performance Highlights**: LLM은 짧은 입력 히스토리를 가진 사용자에게 더욱 우수한 성능을 발휘하였으며, 랭킹 설정에서는 전통 모델들을 여러 차원에서 초월하였지만, 후보 위치 편향과 비존재 아이템의 환각 현상이 문제로 나타남.



### CORAG: A Cost-Constrained Retrieval Optimization System for Retrieval-Augmented Generation (https://arxiv.org/abs/2411.00744)
- **What's New**: 이 논문에서는 승수 최적화를 위한 RAG(Retrieval-Augmented Generation) 시스템을 제안하며, 특히 여러 청크(chunk) 간의 상관관계 및 청크 유틸리티의 비단조성(non-monotonicity)을 완벽하게 고려한 방법을 설명합니다.

- **Technical Details**: 제안된 CORAG 시스템은 몬테카를로 트리 검색(Monte Carlo Tree Search, MCTS) 기반의 정책 프레임워크를 활용하여 청크 조합을 최적화합니다. 이 방법은 청크 간의 상관관계를 고려하여 청크 조합의 순서를 정할 수 있도록 합니다. 또한, 단순히 예산 소진을 종료 조건으로 보는 대신, 예산 제약을 청크 조합의 최적화에 통합하여 비단조성 문제를 해결합니다.

- **Performance Highlights**: 실험 결과, CORAG는 기존의 RAG 시스템보다 청크 선택의 중복성을 현저히 줄이며, 다양한 사용자 쿼리 유형에 더 적응력 있는 성능을 보여주었습니다. 이는 LLM의 응답 품질을 향상시키는 데 기여합니다.



### A graph-based approach to extracting narrative signals from public discours (https://arxiv.org/abs/2411.00702)
Comments:
          23 pages, 4 figures

- **What's New**: 이 논문은 정치적 내러티브(narrative)의 분석을 위한 새로운 그래프 기반(formalism) 방법을 제안합니다. 이러한 방법은 디지털 텍스트로부터 내러티브 신호를 효과적으로 추출하고 분석하는 데 중점을 두며, 특히 정치적인 상황에서 유용하게 활용될 수 있습니다.

- **Technical Details**: 우리는 Abtract Meaning Representation (AMR)에 기반하여 텍스트 집합에서 각 문장의 의미를 그래프와 같은 표현으로 추출합니다. 이후, 서사학(narratology)의 개념을 활용하여 1) 행위자(actors), 2) 이들이 포함된 사건(events), 3) 사건의 관점(perspectivization)을 필터링하는 휴리스틱(heuristics)을 적용합니다. 이러한 요소들은 정치적 내러티브를 형성하는 핵심 신호로 간주됩니다.

- **Performance Highlights**: 유럽연합의 연설을 사례 연구(case study)로 사용하여 제안된 방법이 공적 담론에서 정치적 내러티브의 신호를 효과적으로 표출하는 데 어떻게 활용되는지를 보여줍니다.



### MIRFLEX: Music Information Retrieval Feature Library for Extraction (https://arxiv.org/abs/2411.00469)
Comments:
          2 pages, 4 tables, submitted to Extended Abstracts for the Late-Breaking Demo Session of the 25th Int. Society for Music Information Retrieval Conf., San Francisco, United States, 2024

- **What's New**: 본 논문에서는 음악 정보 검색(Music Information Retrieval, MIR) 연구를 지원하기 위해 다양한 음악 특징 추출 모델을 통합한 확장 가능한 모듈형 시스템인 MIRFLEX를 소개합니다.

- **Technical Details**: MIRFLEX는 키(key), 다운비트(downbeat), 장르(genre)와 같은 음악적 요소 및 악기 인식(instrument recognition), 보컬/악기 분류(vocals/instrumental classification), 성별 인식(vocals gender detection)과 같은 오디오 특성을 포함하는 다양한 특징 추출기를 제공합니다. 이 시스템은 최신 오픈 소스 모델 및 최첨단 기술을 사용하여, 연구자들이 음악 애플리케이션에 통합할 수 있는 잠재적인 라벨(latent or post-processed labels)을 추출할 수 있도록 합니다.

- **Performance Highlights**: MIRFLEX는 음악 데이터의 다양한 측면을 포괄하는 포괄적인 기능 세트를 제공하여 연구자들이 다양한 음악 관련 애플리케이션을 탐색할 수 있도록 합니다. 또한, 새로운 시스템의 손쉬운 통합을 지원하여 벤치마킹 및 비교 도구로서의 역할을 합니다.



### Improving Few-Shot Cross-Domain Named Entity Recognition by Instruction Tuning a Word-Embedding based Retrieval Augmented Large Language Mod (https://arxiv.org/abs/2411.00451)
- **What's New**: 본 논문에서는 IF-WRANER(Instruction Finetuned Word-embedding based Retrieval Augmented large language model for Named Entity Recognition)라는 새로운 접근 방식을 제안합니다. 이는 LLM(large language model) 기반의 NER(named entity recognition)에서 이전의 최신 기술에 비해 2% 이상의 F1 점수 향상을 보여 주었습니다.

- **Technical Details**: IF-WRANER는 LLM의 파인튜닝(finetuning) 중 사용되는 정규화(regularization) 기법과, 문장 수준의 임베딩(embedding) 대신 단어 수준의 임베딩을 이용해 예제를 검색하는 방식을 활용하여 성능을 강화합니다. 이 모델은 데이터 부족(target domain) 문제를 해결하기 위해 소스 도메인(source domain)에서 태그가 있는 데이터만으로 파인튜닝됩니다.

- **Performance Highlights**: IF-WRANER는 CrossNER 데이터셋에서 이전의 최신 모델에 비해 F1 점수를 2% 이상 향상시켰으며, 사용자는 이 모델을 통해 고객 서비스 도메인에서 자동화된 워크플로우로 고객을 안내함으로써 사람의 개입을 약 15% 줄이고 연간 수백만 달러의 비용 절감 효과를 얻었습니다.



### Improving Musical Instrument Classification with Advanced Machine Learning Techniques (https://arxiv.org/abs/2411.00275)
Comments:
          43 pages, 35 figures, 14 tables

- **What's New**: 음악 정보 검색(Music Information Retrieval)의 주요 분야인 악기 분류에 대한 관심이 증가하고 있습니다. 본 연구에서는 오디오 신호에서 악기를 식별하고 분류하는 능력을 향상시키기 위해 다양한 기계 학습(machine learning) 방법을 적용하였습니다.

- **Technical Details**: 이 연구에서는 Naive Bayes, Support Vector Machines, Random Forests, AdaBoost 및 XGBoost 등의 부스팅 기법, 그리고 Convolutional Neural Networks(CNN) 및 Artificial Neural Networks(ANN)와 같은 딥러닝(deep learning) 모델을 포함한 여러 기계 학습 방법을 사용하였습니다. NSynth 데이터셋을 활용하여 이러한 방법들의 효과를 평가하였습니다.

- **Performance Highlights**: 각 방법의 장점과 한계를 비교하여 보다 정확하고 효율적인 분류 시스템 개발을 위한 지침을 제공합니다. 또한 하이브리드 모델 테스트 및 논의도 포함되어 있어 향후 연구 방향에 대한 제안도 이루어졌습니다.



### Content Aware Analysis of Scholarly Networks: A Case Study on CORD19 Datas (https://arxiv.org/abs/2411.00262)
- **What's New**: 이 논문은 과학 연구 네트워크의 주요 요소인 논문, 연구자 및 저널 간의 관계를 조사합니다. 우리는 HITS 알고리즘을 기반으로 주제 정보의 전파에 새로운 접근방식을 도입하고, Named Entity Recognition 및 Entity Linkage를 통해 파생된 주제 정보를 사용하여 COVID-19 도메인에 초점을 맞췄습니다.

- **Technical Details**: MedCAT(Medical Concept Annotation Tool)를 사용하여 CORD19 데이터셋에서 COVID-19 관련 주제를 추출하였으며, 하이브리드 HITS 알고리즘을 적용하여 주제 데이터를 통합함으로써 논문 순위에 상당한 영향을 미치는 것을 보여주었습니다.

- **Performance Highlights**: 우리의 접근 방식은 인용 프레임워크 내에서 주제 관련 정보를 통합하는 효과성을 증명하며, 이를 통해 학문 공동체의 구조에 대한 깊은 통찰을 제공합니다.



### Building Multi-Agent Copilot towards Autonomous Agricultural Data Management and Analysis (https://arxiv.org/abs/2411.00188)
- **What's New**: 이 논문에서는 전통적인 농업 데이터 관리 방식의 한계를 극복하기 위해 대형 언어 모델(LLM)을 기반으로 한 자율 농업 데이터 관리와 분석을 위한 ADMA(농업 데이터 관리 및 분석) 항공 파일럿 시스템을 제안합니다.

- **Technical Details**: ADMA Copilot은 사용자 의도를 이해하고 데이터 처리 파이프라인을 계획하여 자동으로 작업을 수행하는 다중 에이전트 시스템입니다. 이 시스템은 LLM 기반의 컨트롤러, 입력 포맷터 및 출력 포맷터의 세 에이전트가 협력하여 동작하며, 메타 프로그램 그래프를 정의하여 제어 흐름과 데이터 흐름을 분리하여 예측 가능성을 높였습니다.

- **Performance Highlights**: 시스템의 실험 결과, ADMA Copilot은 지능적이고 자율적이며 효율적이고 확장 가능하며 유연하고 개인정보 보호를 강화한 시스템으로 평가되었습니다. 기존 시스템들과 비교하여 우수성과 잠재력을 강조하였으며, 농업 데이터 관리의 현대적 도전 과제를 해결하는 데 기여할 것으로 기대됩니다.



### PSL: Rethinking and Improving Softmax Loss from Pairwise Perspective for Recommendation (https://arxiv.org/abs/2411.00163)
- **What's New**: 본 논문에서는 기존의 Softmax Loss (SL)의 두 가지 주요 한계를 분석하고 이를 개선하기 위해 Pairwise Softmax Loss (PSL)이라는 새로운 손실 함수 범주를 제안합니다. PSL은 SL에서 사용된 지수 함수를 다른 적절한 activation function으로 대체하면서 성능을 향상시킵니다.

- **Technical Details**: PSL은 SL을 쌍(pairwise) 방식으로 재구성하여 양수-음수 쌍 간의 점수 차이에 손실을 적용합니다. 이를 통해 PSL은 DCG와 같은 전통적인 ranking metrics에 대한 이론적 연결을 확립하며, ReLU 또는 Tanh와 같은 적합한 surrogate activation을 선택함으로써 더 나은 성능을 발휘합니다.

- **Performance Highlights**: PSL은 추천 정확도, OOD(Out-of-Distribution) 강인성, 및 노이즈 저항성에 있어 기존 손실 함수들보다 우수한 성능을 보입니다. 본 연구에서 수행한 다양한 실험을 통해 PSL의 효과성과 강인성을 검증했습니다.



### Cost-Aware Query Policies in Active Learning for Efficient Autonomous Robotic Exploration (https://arxiv.org/abs/2411.00137)
- **What's New**: 이 논문에서는 유한 자원에 의해 제한된 임무에서 데이터 수집의 효율성을 중요시하며, Gaussian Process 회귀를 위한 액티브 학습(AL) 알고리즘을 소개합니다. 이 알고리즘은 액션 비용(action cost)을 통합하여 기존의 회귀 문제에 대한 액티브 학습 구현에서의 한계를 극복하고자 합니다.

- **Technical Details**: 논문에서 제안된 AL 알고리즘은 다양한 회귀 문제에 대한 성능을 분석합니다. terrain mapping을 포함하여 다채로운 시뮬레이션된 표면에서의 성능이 root mean square error(RMSE), 샘플 및 수렴(distance until convergence), 수렴 시 모델 분산(model variance) 등의 지표로 비교됩니다. 이 방법은 거리(distance) 제약이 없는 전통적인 불확실성 메트릭을 활용하여 경로 거리에서 RMSE를 최소화합니다.

- **Performance Highlights**: 비용 의존적 획득 정책(cost-dependent acquisition policy)이 정보 이득을 거리(distance) 측면에서 최적화하지 않음에도 불구하고, 전통적인 불확실성 메트릭은 경로 거리에서 RMSE를 최소화하는 데 가장 효과적임을 입증하였습니다. 이는 현실적인 임무 제약 조건 하에 탐사를 최적화하는 방법에 대한 통찰력을 제공합니다.



### NeuroSym-BioCAT: Leveraging Neuro-Symbolic Methods for Biomedical Scholarly Document Categorization and Question Answering (https://arxiv.org/abs/2411.00041)
- **What's New**: 본 연구에서는 OVB-LDA라는 최적화된 주제 모델링 프레임워크와 BI-POP CMA-ES 최적화 기법을 통합한 새로운 접근 방식을 제안하여 생물 의학 논문 초록 분류의 정확성을 향상시키고 있습니다.

- **Technical Details**: 이 연구는 세 가지 구성으로 평가되며, 각 구성은 학술 문서 초록 검색, 금관 표준 학술 문서 초록, 그리고 금관 조각을 포함합니다. MiniLM 모델을 정상적으로 사용하여 주제 모델링과 고급 기계 학습 기술을 결합한 신경 상징적(answer extraction) 접근 방식을 활용하여 데이터를 정교하게 조정하면서도 높은 정확도로回答(answer extraction)를 수행할 수 있습니다.

- **Performance Highlights**: 기존의 RYGH 및 bio-answer finder와 같은 방법들을 뛰어넘는 성능을 보여주며, MiniLM이 작은 모델임에도 불구하고 복잡한 작업을 처리할 수 있는 경쟁력 있는 성능을 나타냅니다. 이 연구의 결과는 생물 의학 질문 응답 분야에서 초록의 유용성을 강조하며, 효율성과 정확성을 개선하기 위한 향후 연구 방향을 제시합니다.



New uploads on arXiv(cs.CV)

### Randomized Autoregressive Visual Generation (https://arxiv.org/abs/2411.00776)
Comments:
          simple method improving autoregressive image generator to SOTA performance; Project page at this https URL

- **What's New**: 이 논문은 이미지 생성 작업에서 새로운 최첨단 성능을 달성하면서 언어 모델링 프레임워크와의 완전한 호환성을 유지하는 랜덤화 자동회귀 모델링(Randomized AutoRegressive modeling, RAR)을 제안합니다. RAR은 표준 자동회귀 훈련 과정 동안 입력 시퀀스를 무작위로 순서를 변경하여 이 모델이 양방향 맥락을 효과적으로 모델링할 수 있게 합니다.

- **Technical Details**: RAR은 입력 시퀀스를 무작위로 재구성하는 확률인 r을 도입하여 훈련 동안 r이 1에서 0으로 선형적으로 감소하도록 설정합니다. 이 과정을 통해 모델이 모든 가능한 재구성 순서에 대해 기대 가능성을 극대화하도록 학습합니다. 결과적으로 각 토큰은 양방향 맥락 아래에서 훈련되고 예측됩니다. 또한 RAR은 기존의 자동회귀 모델 아키텍처를 유지하면서도 성능을 향상시킵니다.

- **Performance Highlights**: ImageNet-256 벤치마크에서 RAR은 FID 점수 1.48을 달성하여 기존의 최첨단 자동회귀 이미지 생성기를 능가할 뿐만 아니라, 확산 기반 및 마스크 변환기 기반 방법을 초월하는 성능을 보였습니다.



### CityGaussianV2: Efficient and Geometrically Accurate Reconstruction for Large-Scale Scenes (https://arxiv.org/abs/2411.00771)
Comments:
          Project Page: this https URL

- **What's New**: 본 논문에서는 대규모 장면 재구성을 위한 새로운 접근 방식인 CityGaussianV2를 소개합니다. 이는 3D Gaussian Splatting (3DGS)의 단점을 해결하고 기하학적 정확성과 효율성을 높이는 데 중점을 두고 있습니다.

- **Technical Details**: CityGaussianV2는 Decomposed-Gradient-based Densification (DGD) 기법을 활용하여 모호한 아티팩트를 제거하고 수렴 속도를 가속화합니다. 또한, 엘로게이션 필터를 도입하여 Gaussian 수의 폭발 문제를 완화하며, 병렬 학습을 위해 CityGaussian 파이프라인을 최적화하여 최대 10배의 저장 용량 절감과 25%의 학습 시간 단축을 달성합니다.

- **Performance Highlights**: 실험 결과, CityGaussianV2는 시각적 품질, 기하학적 정확성, 저장 공간 및 학습 비용 측면에서 우수한 성능을 보이며, 기존 기법들에 비해 훨씬 효율적입니다.



### GameGen-X: Interactive Open-world Game Video Generation (https://arxiv.org/abs/2411.00769)
Comments:
          Project Page: this https URL

- **What's New**: GameGen-X는 오픈 월드 게임 비디오를 생성하고 상호작용적으로 제어할 수 있도록 설계된 최초의 diffusion transformer 모델입니다. 이 모델은 혁신적인 캐릭터, 동적 환경, 복잡한 행동, 다양한 이벤트 등을 포함한 고품질의 콘텐츠 생성을 지원합니다.

- **Technical Details**: GameGen-X는 Open-World Video Game Dataset (OGameData)을 기반으로 훈련되었습니다. 이 데이터셋은 150개 이상의 게임에서 수집된 100만 개의 다양한 플레이 비디오 클립으로 구성되며, 두 단계 훈련 과정(foundation model pre-training 및 instruction tuning)을 거칩니다. InstructNet을 통해 게임과 관련된 다중 모달 제어 신호 전문가를 통합하고, 사용자 입력에 따라 잠재 표현을 조정할 수 있게 합니다.

- **Performance Highlights**: 실험 결과 GameGen-X는 다양한 게임 내용을 고품질로 생성하고, 사용자 입력에 따라 동적으로 반응하는 비디오 클립을 생성할 수 있는 뛰어난 성능을 보여주었습니다. 이 모델은 전통적인 게임 디자인 방식에 대해 확장 가능하고 효율적인 보조 도구로서의 잠재력을 가지고 있습니다.



### Face Anonymization Made Simp (https://arxiv.org/abs/2411.00762)
- **What's New**: 이 논문은 기존의 얼굴 익명화 방법들과는 다르게, 얼굴 인식 모델에 의존하지 않고, 오로지 재구성 손실(reconstruction loss)만을 사용하여 얼굴 익명화를 수행합니다. 이를 통해 얼굴 랜드마크(facial landmarks)나 마스크(mask)가 필요 없으며, 이미지의 세밀한 디테일을 보존하면서도 사실적인 익명화된 얼굴을 생성할 수 있습니다.

- **Technical Details**: 제안하는 방법은 확산 모델(diffusion model)을 기반으로 하며, denoising UNet 구조를 사용합니다. 이 구조는 텍스트-이미지 생성에서사용되는 것과 유사합니다. 또한, 두 가지 상황에서 학습하는 듀얼 설정(dual setting)을 사용하여 소스 이미지가 없는 경우에도 얼굴을 교체할 수 있는 기능을 제공합니다.

- **Performance Highlights**: 이 모델은 신원 익명화(identity anonymization), 얼굴 속성 보존(facial attribute preservation), 이미지 품질(image quality)에서 최신 기술(state-of-the-art) 성능을 달성했습니다. 또한, 추가적인 얼굴 이미지를 입력으로 사용하여 얼굴 교체(face swapping) 작업도 수행할 수 있는 다재다능함을 보여주었습니다.



### Autobiasing Event Cameras (https://arxiv.org/abs/2411.00729)
Comments:
          ECCV 2024 NeVi Workshop

- **What's New**: 이 논문은 event camera를 사용한 머신 비전 어플리케이션에서 발생하는 조명 조건의 문제를 해결하기 위한 자율적인 방법을 제시합니다. 기존보다 더 효율적인 bias 설정을 통해 다양한 조명 환경에서도 쉽게 적용할 수 있는 방식을 탐구하며, 이를 통해 driver monitoring system의 성능을 지속적으로 모니터링하고 자동으로 조정할 수 있게 되었습니다.

- **Technical Details**: 이 연구에서는 event camera의 bias 설정을 조정하여 다양한 조명 조건에 적응하는 DMS 응용 프로그램의 능력을 향상시키는 것을 목표로 합니다. 논문은 자동으로 LCD 내의 모든 bias를 조정하여 (1) 성능을 최적화하고 (2) 사람 얼굴 인식 성능을 개선하는 방식을 언급합니다. Nelder-Mead simplex algorithm을 통해 성능 지표가 허용 수준 이하로 떨어지면 bias 값을 자동으로 조정하여 최적 성능을 유지합니다.

- **Performance Highlights**: 이 연구는 검토된 모든 조건에서 사람 얼굴 탐지를 개선하였으며, autobiasing을 통해 YOLO의 객체 탐지 신뢰도 지표가 33% 이상, 얼굴 탐지 신뢰도 지표가 37% 증가하여 DMS의 효율성을 입증하였습니다. 실험 결과, 낮은 조명 및 다양한 깜박임 주파수에서 기본 bias 값으로는 인간 얼굴을 탐지할 수 없던 조건에서도 성능이 크게 향상되었습니다.



### B-cosification: Transforming Deep Neural Networks to be Inherently Interpretab (https://arxiv.org/abs/2411.00715)
Comments:
          31 pages, 9 figures, 12 tables, Neural Information Processing Systems (NeurIPS) 2024

- **What's New**: B-cos Networks의 새로운 접근인 'B-cosification'을 통해 기존의 사전 학습된 모델을 효율적으로 해석 가능하게 만드는 방법을 제안합니다. 이 방법은 CNN과 ViT와 같은 모델을 작은 변화로 변환하여 보다 우수한 해석성을 제공하며, 훈련 비용을 대폭 절감합니다.

- **Technical Details**: B-cosification은 기존의 DNN을 아키텍처적 변경 없이도 해석 가능하도록 조정하는 기술입니다. 이 과정에서 модели가 계산하는 방식을 인간이 이해할 수 있는 형태로 줄이는 데 초점을 맞춥니다. B-cos 변환의 '정렬 압력(alignment pressure)'을 조정하여 모델의 해석성을 개선하며, 기존의 사전 훈련된 CLIP 모델도 이 방식을 통해 B-cosified CLIP으로 변환됩니다.

- **Performance Highlights**: B-cosified 모델은 기존 DNN 대비 비슷한 해석성을 유지하면서도 분류 성능에서 우수한 결과를 나타내고, 특히 훈련 비용은 훨씬 낮습니다. 다양한 데이터 세트에서 제로 샷(zero-shot) 성능 테스트를 통해 B-cosified CLIP 모델이 경쟁력을 보입니다.



### Debiasify: Self-Distillation for Unsupervised Bias Mitigation (https://arxiv.org/abs/2411.00711)
Comments:
          Accepted at the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV2025)

- **What's New**: Debiasify는 이전의 방법들과는 달리 사전 지식 없이도 편향을 줄이는 새로운 자기 증류(self-distillation) 접근 방식을 제안합니다.

- **Technical Details**: 이 방법은 심층층에서 간단한 속성에 조건화된 특성으로 지식 전이를 수행하는 새로운 증류 손실(distillation loss)을 활용하여, 네트워크의 깊은 층에서 복잡하고 예측력이 높은 특징을 얕은 층으로 전달합니다.

- **Performance Highlights**: CELEBA 데이터셋에서 Wavy Hair 분류에서 worst-group 정확도가 10.13% 향상되었으며, Debiasify는 이전의 비지도 학습(unsupervised) 방법들에 비해 매우 우수한 성능을 보여줍니다.



### ReMatching Dynamic Reconstruction Flow (https://arxiv.org/abs/2411.00705)
Comments:
          Our project website is at this https URL

- **What's New**: 이 연구는 동적인 장면을 이미지 입력으로부터 재구성하는 핵심 컴퓨터 비전 작업을 다루고 있습니다. ReMatching 프레임워크를 도입하여 변형 사전 (deformation priors)을 동적 재구성 모델에 통합함으로써 일반화 품질을 향상시키는 방안을 제시합니다.

- **Technical Details**: ReMatching 프레임워크는 속도 장(field-based velocity fields) 기반의 사전을 채택하여 기존 동적 재구성 파이프라인을 보완할 수 있는 매칭 절차를 제안합니다. 이를 통해 다양한 동적 표현에 적용 가능하며, 여러 유형의 모델 사전 통합을 지원합니다. 프레임워크는 원활한 통합을 위한 최적화 목표와 유연한 설계를 제공합니다.

- **Performance Highlights**: 이 연구는 다양한 벤치마크에서 기존 최첨단 모델들의 재구성 정확성을 명확히 개선했음을 입증하였습니다. 동적인 장면의 실제 및 합성 장면을 포함한 평가에서 이러한 향상이 드러났습니다.



### TaxaBind: A Unified Embedding Space for Ecological Applications (https://arxiv.org/abs/2411.00683)
Comments:
          Accepted to WACV 2025

- **What's New**: TaxaBind는 모든 관심 있는 종을 특성화하기 위한 통합 임베딩 공간을 제시합니다. 이는 종의 지상 이미지, 지리적 위치, 위성 이미지, 텍스트, 오디오, 환경적 특징을 포함한 6가지 모달리티를 활용하는 다중모달(Multimodal) 임베딩 공간입니다.

- **Technical Details**: TaxaBind는 다양한 모달리티의 지식을 효과적으로 증류하기 위한 기술로서 multimodal patching을 제안합니다. 이는 각 모달리티의 고유한 정보 손실을 방지하며, 다양한 생태적 작업을 위한 학습을 가능하게 합니다. 또한, iSatNat과 iSoundNat이라는 두 가지 대형 데이터셋을 구축하고, TaxaBench-8k라는 평가를 위한 진정한 다중모달 데이터셋을 소개합니다.

- **Performance Highlights**: TaxaBind는 다음과 같은 작업에서 강력한 제로샷(zero-shot) 및 돌발(emergent) 능력을 보였습니다: 종 분류(Species Classification), 교차 모델 검색(Cross-model Retrieval), 오디오 분류(Audio Classification). 추가적으로, 여러 벤치마크 및 제로샷 작업에서 모델의 효과성과 성장 가능성을 입증하였습니다.



### Towards High-fidelity Head Blending with Chroma Keying for Industrial Applications (https://arxiv.org/abs/2411.00652)
Comments:
          Accepted by WACV 2025. Project page: this https URL

- **What's New**: CHANGER라는 새로운 산업용 Head Blending 파이프라인을 소개하며, 이는 배우의 머리를 대상 몸에 매끄럽게 통합하는 작업을 위해 고안되었습니다. 이 파이프라인은 기존의 방법이 전경(foreground)과 배경(background)을 단일 작업으로 처리하는 데서 발생하는 질 낮은 블렌딩 문제를 해결합니다.

- **Technical Details**: CHANGER는 배경 통합과 전경 블렌딩을 분리하여 다루는 두 가지 하위 작업으로 문제를 분해합니다. 이를 위해 크로마 키(chroma keying) 기술을 사용하여 아티팩트 없는 배경 생성을 가능하게 하고, H2 증대 기법(H2 augmentation)을 도입하여 다양한 머리 모양과 헤어 스타일을 시뮬레이션합니다. 또한, Foreground Predictive Attention Transformer(FPAT) 모듈을 통해 전경 블렌딩 품질을 향상시킵니다.

- **Performance Highlights**: CHANGER는 벤치마크 데이터셋에서 기존 최첨단 방법 대비 정량적 및 정성적 평가에서 모두 우수한 성능을 보이며, 고충실도의 산업-grade 결과를 제공합니다.



### Event-guided Low-light Video Semantic Segmentation (https://arxiv.org/abs/2411.00639)
Comments:
          12 pages, 5 figures, Accepted to IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2025

- **What's New**: 이번 논문에서는 EVSNet이라는 경량 프레임워크를 제안하여 저조도(低照度) 비디오 의미 분할(video semantic segmentation) 문제를 해결하고자 한다. 기존 연구에서 다루지 않았던 이벤트 모달리티(event modality)를 활용하여, 비디오 프레임에서 조명에 불변한 통합 표현을 학습하는 새로운 접근 방식을 제시하였다.

- **Technical Details**: EVSNet은 이미지 피쳐(image features)와 모션 피쳐(motion features)를 결합하기 위해 세 가지 주요 모듈인 이미지 인코더(Image Encoder), 모션 추출 모듈(Motion Extraction Module), 모션 융합 모듈(Motion Fusion Module)을 사용한다. Motion Extraction Module은 단기 및 장기 모션을 추출하고, 모션 융합 모듈은 이미지 피쳐와 모션 피쳐를 적응적으로 통합한다. 최종적으로 Temporal Decoder를 통해 비디오 컨텍스트를 활용하여 세분화(segmentation) 예측을 생성한다.

- **Performance Highlights**: EVSNet은 세 개의 대규모 저조도 비디오 의미 분할 데이터셋에서 검증한 결과, 최신 기술(State-of-the-art) 방법보다 최대 11배 높은 파라미터 효율성을 보이며, 더 나은 의미 분할 결과를 나타냈다. EVSNet은 경량 아키텍처에도 불구하고 SOTA 성능을 달성하는 데 성공하였다.



### PCoTTA: Continual Test-Time Adaptation for Multi-Task Point Cloud Understanding (https://arxiv.org/abs/2411.00632)
Comments:
          Accepted to NeurIPS 2024

- **What's New**: 이번 논문에서는 멀티태스크 포인트 클라우드 이해를 위한 지속적 테스트 시간 적응(PCoTTA) 프레임워크를 소개합니다. 이 프레임워크는 변화하는 타겟 도메인으로의 모델 전이 가능성을 극대화합니다.

- **Technical Details**: PCoTTA는 자동 프로토타입 혼합(Automatic Prototype Mixture, APM), 가우시안 스플랫 특성 이동(Gaussian Splatted Feature Shifting, GSFS), 대조적 프로토타입 반발(Contrastive Prototype Repulsion, CPR)의 세 가지 주요 구성 요소로 이루어져 있습니다. APM은 원본 프로토타입과 학습 가능한 프로토타입을 혼합하여 재해를 방지하고, GSFS는 테스트 샘플을 원본 도메인으로 동적으로 이동시켜 오류 누적을 완화하며, CPR은 가장 가까운 학습 가능한 프로토타입을 테스트 특성과 가깝게 하고 다른 프로토타입과는 멀리 작업하여 적응 중 각 프로토타입을 구분 가능하게 만듭니다.

- **Performance Highlights**: 본 연구는 총 30,954개의 포인트 클라우드 샘플을 포함한 새로운 벤치마크를 수립하였고, PCoTTA는 현재의 최신 방법들보다 성능이 크게 우수함을 실험적으로 입증하였습니다.



### STAA: Spatio-Temporal Attention Attribution for Real-Time Interpreting Transformer-based Video Models (https://arxiv.org/abs/2411.00630)
- **What's New**: 이 논문에서는 STAA(Spatio-Temporal Attention Attribution)라는 새로운 XAI(Explainable AI) 방법을 소개합니다. 이 방법은 비디오 Transformer 모델을 해석하는 데 있어 공간적 및 시간적 정보를 동시에 제공하는 혁신적인 접근 방식을 제시하여 기존의 한 차원적 설명 방식의 한계를 극복합니다.

- **Technical Details**: STAA는 Transformer 모델의 주의(attention) 값을 기반으로 spatial(공간적) 및 temporal(시간적) 정보를 동시에 제공하는 독창적인 방법입니다. 이 방법은 Kinetics-400 데이터셋을 활용하여 실험을 진행하며, 설명 품질을 평가하기 위한 메트릭스도 도입되었습니다. 또한, 동적 임계값 설정과 주의 집중 메커니즘을 적용하여 신호 대 잡음 비율(signal-to-noise ratio)을 개선했습니다.

- **Performance Highlights**: STAA는 전통적인 XAI 방법의 3% 이하의 계산 자원만으로 작동하여 실시간 비디오 XAI 분석 애플리케이션에 적합합니다. Kinetics-400 데이터셋을 활용한 평가 결과, 설명 품질과 계산 효율성 모두에서 유의미한 개선을 나타냈습니다.



### Investigating the Gestalt Principle of Closure in Deep Convolutional Neural Networks (https://arxiv.org/abs/2411.00627)
Comments:
          Published at the ESANN 2024 proceedings, European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning. Bruges (Belgium) and online event, 9-11 October 2024

- **What's New**: 이번 연구는 컨볼루션 신경망(CNN)에서 '폐쇄의 게슈탈트 원리(Gestalt principle of closure)'를 탐구합니다. 우리는 개념을 모두 흡수하여 빈 공간을 채우는 인간의 지각 방식과 비교하여 CNN의 성능을 평가하는 프로토콜을 제안하였습니다.

- **Technical Details**: 연구에서 고안된 데이터셋은 320개의 완전한 다각형으로 구성되어 있으며, 경계가 제거된 비율에 따라 폴리곤을 테스트합니다. CNN 모델은 AlexNet, VGG16, ResNet, DenseNet 등 다양한 아키텍처를 포함하며, 마지막 레이어는 10개의 노드로 교체하여 10개의 서로 다른 폴리곤에 대한 분류를 학습합니다. 실험은 Adam optimizer를 사용하여 25 epochs 동안 진행됩니다.

- **Performance Highlights**: 완전한 폴리곤의 경우 VGG16과 SqueezeNet V1.1이 90% 이상의 정확도로 우수한 성능을 보였으며, AlexNet과 ResNet50은 70% 이상의 정확도를 기록했습니다. 반면, EfficientNet B0 및 MobileNetV3는 40%에서 50% 사이의 정확도를 보였으며, 이는 이들 모델의 낮은 복잡성에 기인합니다.



### ZIM: Zero-Shot Image Matting for Anything (https://arxiv.org/abs/2411.00626)
Comments:
          preprint (21 pages, 16 figures, and 8 tables)

- **What's New**: 이번 논문은 기존의 Segment Anything Model (SAM)의 제로샷(segmentation zero-shot) 능력을 활용하여 이미지 매팅(image matting) 문제를 해결하기 위한 새로운 모델 ZIM을 제안합니다. ZIM은 세밀한 매트(matte) 마스크 생성을 개선하며, SA1B-Matte라는 새로운 데이터셋을 구축하여 저비용으로 맞춤형 레이블을 생성합니다.

- **Technical Details**: ZIM은 두 가지 핵심 구성요소로 이루어집니다. 첫째, 레이블 변환기(label converter)를 통해 세분화(segmentation) 레이블을 상세한 매트 레이블로 변환하여 SA1B-Matte 데이터셋을 생성합니다. 둘째, 계층적 픽셀 디코더(hierarchical pixel decoder)를 장착하여 마스크 표현력을 향상시키고, 비주얼 프롬프트에 기반한 주목(attention) 메커니즘을 설계하여 성능을 개선합니다.

- **Performance Highlights**: ZIM은 MicroMat-3K 테스트 세트를 사용하여 실험하였으며, 기존의 매팅 모델들보다 더 높은 정확도의 세밀한 마스크를 생성하였습니다. 실험 결과, ZIM은 제로샷 기능을 유지하면서도 마스크 생성을 위해 우수한 정밀도를 제공합니다. 또한, ZIM은 이미지 인페인팅(image inpainting) 및 3D NeRF와 같은 다양한 다운스트림 작업에서도 유용성을 입증했습니다.



### Dual Low-Rank Adaptation for Continual Learning with Pre-Trained Models (https://arxiv.org/abs/2411.00623)
- **What's New**: 이 논문에서는 지속적인 학습(continual learning, CL)과 파라미터 효율적 세부 조정(parameter-efficient fine-tuning, PEFT) 기술을 결합한 새로운 방법론을 제안합니다. Dual Low-Rank Adaptation(DualLoRA)은 각 레이어에 대해 직교 LoRA 어댑터와 잔여 LoRA 어댑터를 도입하여 안정성과 플라스틱성의 균형을 맞춥니다.

- **Technical Details**: DualLoRA는 두 가지 어댑터를 사용합니다. 첫 번째는 직교 LoRA(adapter)로, 이전 과업의 직교 서브스페이스에서 파라미터를 업데이트하여 치명적인 망각(catatropic forgetting)을 줄이고, 두 번째는 잔여 LoRA로, 과업 특정 기반에 의해 형성된 잔여 서브스페이스에서 파라미터를 업데이트합니다. 이를 통해 새로운 과업에 대한 세부 조정이 가능합니다.

- **Performance Highlights**: ViT 기반 모델에 대한 실험 결과, DualLoRA는 기존 CL 방법들에 비해 정확도, 추론 속도 및 메모리 효율성이 크게 향상됨을 보여주었습니다.



### HopTrack: A Real-time Multi-Object Tracking System for Embedded Devices (https://arxiv.org/abs/2411.00608)
- **What's New**: HopTrack는 임베디드 장치에 최적화된 실시간 다중 객체 추적 시스템입니다. 이 시스템은 동적 샘플링 및 새롭게 개선된 매칭 기법을 통해 추적 정확도를 향상시키고 있으며, 기존의 MOT 시스템보다 우수한 성능을 보여줍니다.

- **Technical Details**: HopTrack는 디스크리타이즈된 정적 및 동적 매칭 접근방식을 사용하여 추적 정확성을 높이며, 콘텐츠 인식 동적 샘플링 기법을 통해 프레임에서 물체를 추출합니다. 두 가지 데이터 연관 전략인 Hop Fuse 및 Hop Update를 활용하여 추적 결과를 융합하고 추적 오류를 수정합니다.

- **Performance Highlights**: HopTrack은 NVIDIA AGX Xavier에서 최대 39.29 fps로 처리 속도를 기록하고, MOT16 벤치마크에서 최대 63.12%의 multi-object tracking accuracy (MOTA)를 달성합니다. 에너지 소비는 20.8% 감소하고, 전력 소비는 5%, 메모리 사용량은 8% 줄어드는 효율성을 보였습니다.



### On Deep Learning for Geometric and Semantic Scene Understanding Using On-Vehicle 3D LiDAR (https://arxiv.org/abs/2411.00600)
Comments:
          PhD thesis (Durham University, Computer Science), 149 pages (the 2024 BMVA Sullivan Doctoral Thesis Prize runner-up). Includes published content from arXiv:2407.10159 (ECCV 2024 ORAL), arXiv:2303.11203 (CVPR 2023), and arXiv:2406.10068 (3DV 2021), with minor revisions to the examined version: this https URL

- **What's New**: 본 연구에서는 LiDAR(측距 센서) 기반의 작업에서의 정확성과 효율성을 개선하기 위해, 최초의 고충실도 128채널 3D LiDAR 데이터세트인 DurLAR를 제시합니다. 이 데이터세트는 파노라마 주변(Near Infrared) 및 반사 이미지를 포함하고 있습니다.

- **Technical Details**: DurLAR 데이터세트는 3D 포인트 클라우드와 관련된 기하학적 및 의미론적(scene understanding) 이해에 필수적입니다. 새로운 파이프라인은 더 작은 아키텍처를 적용하여 더 적은 Ground-truth 주석(annotation)으로도 경쟁력 있는 세분화(segmentation) 정확도를 달성합니다. 또한, Range-Aware Pointwise Distance Distribution (RAPiD) 기능과 RAPiD-Seg 아키텍처를 도입하여 기존 접근 방안보다 세분화 정확도를 높였습니다.

- **Performance Highlights**: 이 연구의 모든 기여는 동료 심사를 거친 학회에서 인정받았으며, 자율 주행 기술에서의 3D LiDAR 응용 프로그램의 정확도와 효율성이 모두 향상되었음을 강조합니다.



### Federated Voxel Scene Graph for Intracranial Hemorrhag (https://arxiv.org/abs/2411.00578)
- **What's New**: 이번 연구에서는 Intracranial Hemorrhage (ICH) 문제를 해결하기 위해 Federated Scene Graph Generation을 최초로 적용하였습니다. 이를 통해 다양한 데이터가 더욱 효율적으로 활용될 수 있음을 입증하였습니다.

- **Technical Details**: Federated Learning (FedL) 방법론을 사용하여 다양한 데이터 소스로부터 모델을 훈련시킴으로써, 기존의 중앙집중식 데이터셋 기반 모델보다 최대 20% 더 많은 임상적으로 관련 있는 관계를 회수할 수 있었습니다. 또한, Fed-MOTIF와 Fed-IMP 방법론을 통해 클라이언트 간의 공통 관계 분포를 학습할 수 있었습니다.

- **Performance Highlights**: 이 연구의 모델들은 4개 서로 다른 데이터셋에서 훈련되었으며, 각 데이터셋에서 ICH 탐지를 개선하는 결과를 보여주었습니다. FedL을 통해 훈련된 모델은 단일 중앙집중식 데이터셋으로 훈련된 모델에 비해 더 많은 관계를 회수하여, 전체적인 일반화 능력을 향상시켰습니다.



### Handheld Video Document Scanning: A Robust On-Device Model for Multi-Page Document Scanning (https://arxiv.org/abs/2411.00576)
- **What's New**: 스마트폰을 사용하여 다중 페이지 문서를 자동으로 스캔하는 새로운 접근법이 제안되었습니다. 이 방법은 사용자가 문서를 넘기는 동안 비디오 스트림에서 페이지를 인식하고 실시간으로 피드백을 제공합니다.

- **Technical Details**: 이 시스템은 사용자가 핸드헬드 스캔을 하면서 발생하는 흔들림 및 불안정성에 견딜 수 있도록 설계되었으며, 효율적인 On-device deep learning 모델을 사용하는 것이 특징입니다. 이 모델은 비디오 캡처 중 실시간으로 분류를 수행하여 사용자에게 신속한 피드백을 제공합니다.

- **Performance Highlights**: PUCIT 페이지 전환 데이터셋에서 최첨단 결과를 달성했으며, 특히 사용자가 문서를 직접 넘기는 동시에 비디오를 촬영할 수 있는 능력이 강조되었습니다.



### Automated Classification of Cell Shapes: A Comparative Evaluation of Shape Descriptors (https://arxiv.org/abs/2411.00561)
- **What's New**: 이 연구는 세포 형태를 구분하기 위한 다양한 특징들을 평가하고, 노이즈가 포함된 윤곽을 통해 세포 형태를 자동으로 분류하는 데 필요한 종합적이고 정량적인 비교를 제시합니다.

- **Technical Details**: 본 연구는 Elliptical Fourier Descriptors와 같은 2D 윤곽 묘사 방법을 사용하여, 축 비율이나 솔리디티와 같은 스칼라 특징들을 평가합니다. 세포 윤곽을 정규화하기 위해 Procrustes 등록 방식을 적용하였으며, 이후 PCA와 같은 차원 축소 기법을 활용하여 세포 형태를 분석했습니다.

- **Performance Highlights**: 100,000개의 노이즈가 포함된 윤곽이 포함된 합성 데이터셋에서 가장 효과적인 특징 집합을 확인하고, CytoDArk0 데이터셋을 통해 실제 데이터에서도 성능이 검증되었습니다.



### Topology and Intersection-Union Constrained Loss Function for Multi-Region Anatomical Segmentation in Ocular Images (https://arxiv.org/abs/2411.00560)
Comments:
          5 pages, 4 figures, International Symposium on Biomedical Imaging 2025

- **What's New**: 본 논문에서는 Ocular Myasthenia Gravis (OMG) 조기 진단을 위한 새로운 접근법을 제안합니다. 기존에 공개된 데이터셋과 도구가 없었던 문제를 해결하기 위해, 작은 훈련 데이터셋에서도 성능을 향상시킬 수 있는 새로운 topology 및 intersection-union constrained loss function (TIU loss)를 개발했습니다.

- **Technical Details**: 제안된 방법은 MaxPooling을 활용한 다중 스케일 픽셀 표현과 ReLU를 적용하여 intersection-union 제약을 enforce하는 두 가지 주요 작업으로 구성됩니다. 이 방법은 cross-entropy 또는 Dice loss와 같은 픽셀 수준의 loss function과 통합되어 Deep Learning 모델에서 평가되었습니다. 제안된 TIU loss function은 sclera, iris, pupil 간의 해부학적 관계를 기반으로 동작합니다.

- **Performance Highlights**: 55명의 대상과 2,197개의 이미지를 포함하는 공개 데이터셋에서 실험한 결과, 우리의 방법은 83.12%의 평균 Dice 점수를 기록하였으며, 10%의 제한된 훈련 데이터로 baseline보다 8.32% 향상된 성능을 보였습니다. 47명의 환자와 501개의 이미지를 대상으로 한 임상 환경에서도 64.44%의 Dice 점수를 기록하였습니다.



### Is Multiple Object Tracking a Matter of Specialization? (https://arxiv.org/abs/2411.00553)
Comments:
          NeurIPS 2024

- **What's New**: 본 논문에서는 Parameter-efficient Scenario-specific Tracking Architecture (PASTA)를 제안하여, 다양한 시나리오에서의 Multiple Object Tracking (MOT) 성능을 개선하고자 합니다. 이를 통해 비슷한 문제점을 겪고 있는 기존 transformer 기반 트래커들을 보다 효과적으로 지원할 수 있습니다.

- **Technical Details**: PASTA는 Parameter Efficient Fine-Tuning (PEFT) 기술과 Modular Deep Learning (MDL)을 결합하여, 카메라 시점, 조명 조건 등 주요 시나리오 속성을 정의하고 이에 맞춘 PEFT 모듈을 각 속성별로 학습합니다. 이들 모듈은 파라미터 공간에서 결합되어, 새로운 도메인에 대한 일반화를 용이하게 하며 추론 시간의 증가 없이 구성됩니다.

- **Performance Highlights**: MOTSynth 데이터셋에서의 실험과 MOT17 및 PersonPath22에 대한 제로샷 평가 결과, PASTA로 구성된 신경망 추적기는 단일 기능으로 구성된 기존 모델보다 우수한 성능을 나타냅니다. 이는 PASTA가 모듈 간의 부정적 간섭을 방지하고, 도메인 전이 능력을 향상시킨 덕분입니다.



### Tracking one-in-a-million: Large-scale benchmark for microbial single-cell tracking with experiment-aware robustness metrics (https://arxiv.org/abs/2411.00552)
Comments:
          17 pages, 4 figures, 3 tables, BioImage Computing @ ECCV 2024

- **What's New**: 본 논문에서는 미생물 생세포 이미징(Microbial Live-Cell Imaging, MLCI) 분야에서 가장 큰 공개 데이터셋을 소개하며, 140만 개 이상의 세포 인스턴스와 29,000개의 셀 트랙, 14,000개의 셀 분열을 포함하고 있습니다. 또한, 실험 매개변수에 따른 추적 성능의 영향을 분석하기 위한 새로운 벤치마크를 제시합니다.

- **Technical Details**: MLCI는 고속 스크리닝 기술로, 마이크로 유체 칩 장치에 생미생물 세포를 도입하여 1회 실험에서 수십 개의 시간 경과 이미지를 기록합니다. 본 연구에서는 세포 분열과 관련된 메트릭을 기존의 세포 추적 메트릭에 추가하여 실험 매개변수를 고려한 평가 방법을 제안합니다. 이 연구는 낮은 이미징 간격과 높은 세포 수에서 추적 성능이 저하되는 것을 정량화하였습니다.

- **Performance Highlights**: 새로운 벤치마크를 통해 실험 매개변수가 추적 품질에 미치는 영향을 정량적으로 분석하였으며, 데이터 기반 방법 개발의 기회를 제공합니다. 결과적으로 현재의 세포 추적 알고리즘이 실험 매개변수에 얼마나 의존적인지를 밝혔습니다.



### Generative AI-based Pipeline Architecture for Increasing Training Efficiency in Intelligent Weed Control Systems (https://arxiv.org/abs/2411.00548)
- **What's New**: 본 연구는 합성 이미지 생성을 위한 새로운 접근 방식을 제안하여, 지능형 잡초 제어를 위한 딥 러닝(Object Detection) 모델의 성능을 향상시킵니다. 이를 위해 Segment Anything Model(SAM)과 Stable Diffusion Model을 통합하여 실제 환경의 다양한 조건을 반영한 합성 이미지를 생성합니다.

- **Technical Details**: 연구진은 YOLO 모델을 통해 10%의 합성 이미지와 90%의 실제 이미지를 조합하여 훈련시킨 결과, mAP50과 mAP50-95 점수에서 뛰어난 성능을 보여주었습니다. 이 방법은 특히 이미지 품질과 데이터 효율성을 높이며, 자동 주석 달기 과정을 구현하여 더욱 향상된 데이터 처리 파이프라인을 제공합니다.

- **Performance Highlights**: YOLO 모델은 합성 이미지 데이터셋에서 더욱 높은 정확도를 보였으며, 이는 실제 이미지 데이터셋만으로 훈련된 모델보다 우수한 결과를 나타냅니다. 본 연구는 지능형 시스템의 자가 개선 기능을 통해 지속적인 학습 능력을 향상시킬 수 있는 가능성을 제시합니다.



### 3D Equivariant Pose Regression via Direct Wigner-D Harmonics Prediction (https://arxiv.org/abs/2411.00543)
Comments:
          Accepted to NeurIPS 2024, Project webpage at this http URL

- **What's New**: 본 논문은 주어진 이미지에서 물체의 3D 방향을 추정하는 single-image pose estimation의 새로운 접근 방식을 소개합니다. 기존의 방법들은 3D 회전을 Euler 각도나 quaternion으로 매개변수화하여 학습하지만, 이들은 단절성과 특이성을 야기할 수 있습니다. 본 연구에서는 frequency-domain을 활용하여 Wigner-D 계수를 직접 예측하는 새로운 방법론을 제안하여 이러한 문제를 해결했습니다.

- **Technical Details**: 이 연구에서 제안한 SO(3)-equivariant pose harmonics predictor는 frequency domain에서 Wigner-D 계수를 예측함으로써 3D 회전 회귀를 수행합니다. 이는 spherical CNNs의 작업 방식과 일치하여, spatial parameterizations에서 발생하는 단절성 및 특이성을 극복하는 데 도움을 줍니다. 또한, frequency-domain MSE loss를 도입하여 3D 회전의 연속적인 학습을 가능하게 합니다.

- **Performance Highlights**: 모델은 ModelNet10-SO(3) 및 PASCAL3D+와 같은 표준 벤치마크에서 최첨단 성능을 달성하였으며, 정확도, 강인성 및 데이터 효율성에서 유의미한 개선을 보여 주었습니다.



### Cross-modal semantic segmentation for indoor environmental perception using single-chip millimeter-wave radar raw data (https://arxiv.org/abs/2411.00499)
Comments:
          5291 words, 17 pages, 11 figures

- **What's New**: 이 논문에서는 실내 환경 인식을 위한 단일 칩 밀리미터파(mmWave) 레이더를 기반으로 한 크로스 모달(Cross-modal) 의미 분할(Semantic Segmentation) 모델을 제안하였습니다. 새로운 자동 라벨 생성 방법도 소개되어 LiDAR 포인트 클라우드와 점유 그리드 맵을 활용하여 고품질 라벨을 효율적으로 획득합니다.

- **Technical Details**: 제안된 의미 분할 모델은 U-Net을 기반으로 하며, 공간 주의 모듈(Spatial Attention Module)이 통합되어 모델의 성능을 향상시킵니다. 이 모델은 전통적인 방법과 달리 방위각(Azimuth)에 의해 최소한의 영향을 받으며, 상대적으로 멀어질수록 성능이 저하되지만 잘 설계된 모델을 통해 이를 완화할 수 있습니다. 또한, 원시 ADC 데이터를 입력으로 사용하는 것은 비효율적이며 RA 텐서(RA tensors)보다 RD 텐서(RD tensors)가 제안된 모델에 더 적합하다는 점도 확인되었습니다.

- **Performance Highlights**: 크로스 모달 의미 분할 기술이 실내 환경에 대해 더 직관적이고 정확한 표현을 제공한다는 결과가 나타났습니다. 이 방법이 실내 소방 및 구조 작업에서 구조원의 안전을 향상시키고 구조 효율성을 개선하는 데 도움을 줄 것으로 기대됩니다.



### LAM-YOLO: Drones-based Small Object Detection on Lighting-Occlusion Attention Mechanism YOLO (https://arxiv.org/abs/2411.00485)
- **What's New**: 이번 연구에서는 드론 기반 타겟 탐지를 위한 LAM-YOLO 모델을 소개합니다. 이 모델은 다양한 조명 조건에서 소형 타겟의 가시성을 높이기 위해 광 차단 주의 메커니즘을 도입하였고, 피쳐 레이어 간 상호작용을 개선하기 위해 Involution 모듈을 통합했습니다.

- **Technical Details**: LAM-YOLO 모델은 여러 가지 향상된 기능을 포함하고 있습니다. 첫째, 조명 차단 주의 메커니즘(Lighting-Occlusion Attention Module)으로 소형 타겟의 가시성을 개선하며, 둘째, 개선된 SIB-IoU(Soft Intersection Bounding Box IoU) 회귀 손실 함수를 사용하여 모델 수렴 속도를 증가시킵니다. 셋째, 두 개의 보조 탐지 헤드를 추가하여 소형 타겟 탐지의 정확성을 높입니다.

- **Performance Highlights**: VisDrone2019 공개 데이터 세트에서 LAM-YOLO는 기존 YOLOv8에 비해 평균 정확도가 7.1% 향상되었고, Faster R-CNN, YOLOv9, YOLOv10과 비교하여 mAP@0.5 및 mAP@0.5:0.95 성능 등이 뛰어났습니다.



### MV-Adapter: Enhancing Underwater Instance Segmentation via Adaptive Channel Attention (https://arxiv.org/abs/2411.00472)
- **What's New**: 이 연구에서는 수중 비전 작업에서의 분할 성능을 향상시키기 위한 MarineVision Adapter (MV-Adapter)를 제안합니다. 이 모듈은 적응형 채널 주의 메커니즘을 통합하여 모델이 수중 이미지의 특성에 따라 각 채널의 특징 가중치를 동적으로 조정할 수 있게 합니다.

- **Technical Details**: MV-Adapter는 수중 환경에서의 광학적 특성과 다양한 장면 특징에 따라 채널 특성 가중치를 동적으로 조절하는 기능을 갖추고 있습니다. 이를 통해 빛 감쇠 및 색 왜곡의 조건에서도 안정적인 세분화 성능을 유지할 수 있습니다.

- **Performance Highlights**: MV-Adapter 모듈은 USIS10K 데이터셋의 여러 핵심 메트릭에서 기존 모델을 초과 성능을 보였으며, 특히 높은 정밀도의 세분화 작업에서 뛰어난 성과를 보여주었습니다.



### Target-Guided Adversarial Point Cloud Transformer Towards Recognition Against Real-world Corruptions (https://arxiv.org/abs/2411.00462)
Comments:
          Accepted by NeurIPS 2024; code: this https URL

- **What's New**: 이번 연구에서는 Target-Guided Adversarial Point Cloud Transformer(APCT)를 제안하여 3D 포인트 클라우드 인식의 견고성을 향상시키는 새로운 방법을 소개합니다. APCT는 학습 과정에서 매 단계마다 발견된 패턴을 기반으로 적대적 특성 제거 메커니즘을 통해 전반적인 구조를 포착합니다.

- **Technical Details**: APCT는 두 가지 핵심 모듈인 Adversarial Significance Identifier와 Target-guided Promptor를 통합하여 동작합니다. Adversarial Significance Identifier는 전역 맥락 분석을 통해 토큰의 중요성을 판별하며, Target-guided Promptor는 self-attention 메커니즘 내에서 토큰 탈락 가능성을 강조합니다. 이러한 구조는 모델이 다양한 객체 패턴을 더 효과적으로 인식하도록 돕습니다.

- **Performance Highlights**: APCT는 ModelNet-C 및 ScanObjectNN-C 벤치마크를 포함한 여러 성능 평가에서 최첨단 결과를 달성했습니다. 또한 ShapeNet-C의 형태 분할 작업에서 강력한 일반화 능력을 보여주며, 제안된 방법의 효과성을 명확히 입증하였습니다.



### ConceptFactory: Facilitate 3D Object Knowledge Annotation with Object Conceptualization (https://arxiv.org/abs/2411.00448)
Comments:
          NeurIPS 2024 Track on Datasets and Benchmarks

- **What's New**: ConceptFactory는 3D 객체 지식의 효율적인 주석을 지원하는 새로운 접근 방식으로, 일반화된 개념을 통해 3D 객체를 인식하는 기술을 도입합니다.

- **Technical Details**: 이 도구는 두 가지 주요 구성 요소(ConceptFactory Suite와 ConceptFactory Asset)로 구성됩니다. ConceptFactory Suite는 객체 개념화를 위한 웹 기반 플랫폼을 제공하며, Standard Concept Template Library (STL-C)에서 제공하는 263개의 개념 템플릿을 사용하여 객체의 형상을 설명합니다. ConceptFactory Asset은 39개 범주에서 수집된 4380개의 개념화된 객체를 포함합니다.

- **Performance Highlights**: 타당성 검증을 위해 각종 기준 작업에서 최첨단 알고리즘을 통해 실험을 수행하며, 기존 주석 방식보다 더 우수한 품질과 다양성을 보여줍니다.



### PLATYPUS: Progressive Local Surface Estimator for Arbitrary-Scale Point Cloud Upsampling (https://arxiv.org/abs/2411.00432)
- **What's New**: 이번 논문에서는 3D 포인트 클라우드의 업샘플링 문제를 해결하기 위해 Progressive Local Surface Estimator (PLSE)라는 새로운 접근 방식을 소개합니다. PLSE는 곡률 기반 샘플링 기법을 통해 복잡한 지역의 로컬 피처를 보다 효과적으로 포착하며, 이는 특히 높은 곡률 영역에 중점을 두고 있습니다.

- **Technical Details**: PLSE는 포인트 클라우드의 곡률 분포를 활용하여 학습 진행에 있어 쉽고 어려운 샘플을 정의하는 커리큘럼 학습 전략을 통합합니다. 이를 통해 모델이 복잡한 구조가 있는 지역에서 특징을 추출하는 데 집중할 수 있도록 돕습니다.

- **Performance Highlights**: 실험 결과, PLSE는 기존 방법들에 비해 훨씬 뛰어난 성능을 보이며, 높은 품질과 밀도를 가진 포인트 클라우드를 생성하는 능력을 입증했습니다. 확실한 정확도와 디테일로 업샘플링의 새로운 기준을 세웠습니다.



### Cityscape-Adverse: Benchmarking Robustness of Semantic Segmentation with Realistic Scene Modifications via Diffusion-Based Image Editing (https://arxiv.org/abs/2411.00425)
Comments:
          19 pages, under review, code and dataset will be available at this https URL

- **What's New**: 본 논문은 Cityscape-Adverse라는 새로운 벤치마크를 소개하며, 이는 확산 기반 이미지 편집을 활용하여 다양한 악조건에서의 이미지를 시뮬레이션하여 시맨틱 세분화 모델의 강건성을 평가하는 데 중점을 둡니다.

- **Technical Details**: Cityscape-Adverse 벤치마크는 8가지의 다양한 환경 조건을 포함하여, 계절 변화, 날씨 변화, 조명 변화를 시뮬레이션합니다. 또한, CNN과 Transformer 기반의 최첨단 시맨틱 세분화 모델들이 이러한 조건에서 어떻게 성능을 발휘하는지를 평가하였습니다.

- **Performance Highlights**: 모든 모델은 극단적인 조건에서 성능 저하를 경험하였으며, 특히 CNN 기반 아키텍처의 성능 저하가 두드러졌습니다. 반면, Transformer 기반 모델은 상대적으로 높은 강건성을 보였습니다. Cityscape-Adverse에서 훈련된 모델은 이전에 보지 못한 도메인에 적용할 때에도 현저한 강건성이 향상됨을 확인하였습니다.



### Improving Viewpoint-Independent Object-Centric Representations through Active Viewpoint Selection (https://arxiv.org/abs/2411.00402)
- **What's New**: 이 논문에서는 기존의 다중 시점 객체 중심 학습 방법들이 랜덤하거나 순차적인 시점 선택 전략을 사용하는 한계를 지적하고, 정보 격차가 가장 큰 시점을 선택하는 새로운 능동적 시점 선택 전략을 제안합니다.

- **Technical Details**: 제안된 AVS(Active Viewpoint Selection)는 관찰 세트에서 학습된 객체 중심 표현을 통해 미지의 시점 이미지를 예측하고, 이들 간의 정보를 비교하여 최대 정보 이득을 가져오는 미지의 시점을 선택합니다. 이를 통해 객관적이고 정보 중심적인 학습을 지원합니다.

- **Performance Highlights**: 다양한 데이터셋 실험을 통해 AVS는 랜덤 시점 선택 전략에 비해 세분화 및 재구성 성능을 유의미하게 향상시키며, 미지의 시점에 대한 이미지를 정확하게 예측할 수 있는 능력을 입증했습니다.



### StyleTex: Style Image-Guided Texture Generation for 3D Models (https://arxiv.org/abs/2411.00399)
Comments:
          Accepted to Siggraph Asia 2024

- **What's New**: 이번 연구에서는 3D 모델을 위한 스타일 유도 텍스처 생성을 다루는 새로운 방법인 StyleTex를 제안합니다. 이 방법은 참조 이미지에서 스타일 정보를 분리하여, 텍스처 생성 과정에서 사용합니다. 또한, 텍스트 프롬프트와의 정렬을 통해 다양한 스타일 텍스처를 자동으로 생성합니다.

- **Technical Details**: StyleTex는 참조 이미지의 CLIP 임베딩에서 스타일 특징을 분해하는 방법을 사용합니다. 이를 통해 스타일 정보를 추출하고, 교차 주의 메커니즘(cross-attention mechanism)을 통해 생성 과정에 통합합니다. 내용 정보는 부정적 프롬프트로 사용하여 추가적인 분리를 달성합니다. 또한, Interval Score Matching (ISM) 기법을 사용하여 과도한 부드러움(over-smoothness) 문제를 해결하고, 기하학적으로 일관성을 유지하기 위해 ControlNet을 상용합니다.

- **Performance Highlights**: StyleTex는 기존의 기준 방법들과 비교하여 향상된 성능을 보여주며, 생성된 텍스처는 참조 이미지의 스타일을 유지하면서 제공된 3D 메시의 고유 세부정보와 텍스트 프롬프트와도 잘 정렬됩니다. 정량적 및 정성적 실험 결과에서 StyleTex의 뛰어난 특징이 입증되었습니다.



### Right this way: Can VLMs Guide Us to See More to Answer Questions? (https://arxiv.org/abs/2411.00394)
Comments:
          NeurIPS 2024

- **What's New**: 이번 연구에서는 비전 언어 모델(Vision Language Models, VLMs)의 정보 부족 상황에서의 방향성 안내(Directional Guidance) 제공 능력을 평가하는 새로운 비주얼 질문 응답(Visual Question Answering, VQA) 과제를 정의합니다. 이 과제는 특히 시각 장애인에게 더 나은 지원을 제공하기 위한 것입니다.

- **Technical Details**: 연구팀은 현재의 VLMs가 정보를 충분히 평가하지 못하는 문제를 해결하기 위해, 정보 부족 상황에서 이미지를 조정하는 방법을 안내하는 능력을 평가합니다. 새로운 방향성 안내 데이터셋과 자동화된 VQA 데이터 증가 프레임워크를 통해, 시뮬레이션된 정보 부족 시나리오에서 VLM을 학습시킵니다.

- **Performance Highlights**: 세 가지 오픈 소스 VLM을 대상으로 한 실험 결과, 제안된 방향성 안내 작업에서 성능이 유의미하게 향상되었습니다. 또한, 가장 성능이 좋은 모델은 GPT-4o (CoT)보다 3% 높은 정확도를 기록하였습니다.



### TextDestroyer: A Training- and Annotation-Free Diffusion Method for Destroying Anomal Text from Images (https://arxiv.org/abs/2411.00355)
- **What's New**: 이 논문에서 제안하는 TextDestroyer는 최초의 트레이닝 및 주석 없이 장면 텍스트를 제거할 수 있는 방법입니다. 기존의 텍스트 제거 방법은 복잡한 주석 작업과 재교육이 필요했으며, 희미하지만 인식 가능한 텍스트 정보가 남아 개인 정보 보호와 콘텐츠 숨기기를 저해하는 문제가 있었습니다.

- **Technical Details**: TextDestroyer는 세 단계의 계층적 프로세스를 통해 정확한 텍스트 마스크를 생성합니다. 먼저, 가우시안 분포를 사용하여 잠재 시작 코드에서 텍스트 영역을 혼란스럽게 조작한 후, 복원 과정에서는 원본 잠재값에서의 자기 주의 (self-attention) 키와 값을 참조하여 손상된 배경을 복원합니다. 각 역전 단계에서 저장된 잠재 코드는 복원 과정에서 대체에 사용됩니다.

- **Performance Highlights**: TextDestroyer의 장점은 (1) 수작업 데이터 주석과 자원 소모적인 훈련을 제거; (2) 인식 가능한 흔적 없이 철저한 텍스트 파괴; (3) 현실 세계의 장면과 생성된 이미지 모두에서 잘 작동하는 우수한 일반화 능력을 보여줍니다.



### GAFusion: Adaptive Fusing LiDAR and Camera with Multiple Guidance for 3D Object Detection (https://arxiv.org/abs/2411.00340)
- **What's New**: 최근 LiDAR와 카메라 간의 상호작용을 강조한 새로운 다중 모달리티 3D 객체 탐지 방법인 GAFusion을 제안합니다. LiDAR의 도움으로 깊이 정보를 보완하고, 다양한 모달 BEV(Bird's-Eye-View) 특징 간의 상호작용을 향상시킵니다.

- **Technical Details**: GAFusion은 sparse depth guidance (SDG)와 LiDAR occupancy guidance (LOG)를 사용하여 3D 특징을 생성합니다. LGAFT(LiDAR-guided adaptive fusion transformer) 모듈을 도입하여 LiDAR와 카메라 데이터의 생생한 교차작용을 개선하고, MSDPT(multi-scale dual-path transformer)를 통해 수신 영역을 확장하며, 마지막으로 과거 프레임들의 특징을 집계하는 temporal fusion 모듈을 설계합니다.

- **Performance Highlights**: GAFusion은 nuScenes 테스트 세트에서 73.6% mAP 및 74.9% NDS로 최첨단 3D 객체 탐지 성능을 달성했습니다.



### NCST: Neural-based Color Style Transfer for Video Retouching (https://arxiv.org/abs/2411.00335)
Comments:
          10 pages, 8 figures

- **What's New**: 이 연구에서는 색상 스타일 전송(video color style transfer) 방식의 투명성을 높이고 사용자 조정 기능을 제공하는 새로운 방법을 제안합니다. 기존의 신경망(neural network) 기반 방법은 전달 과정이 불투명하고 결과물에 대한 사용자 제어가 제한적이었습니다.

- **Technical Details**: 제안된 방법은 두 이미지를 사용하여 색상 스타일 전송 시 필요한 특정 파라미터를 예측하는 신경망을 활용합니다. 사용자는 이 파라미터를 통해 컬러 스타일 전송 과정을 이해하고 세밀하게 조정할 수 있습니다. 알고리즘은 고프레임(key frames)을 사용하여 영상에 스타일을 적용하며, 3D LUT를 생성하여 초고속 변환 속도를 유지합니다.

- **Performance Highlights**: 실험 결과, 제안된 알고리즘은 기존 방법들보다 색상 스타일 전송 품질이 우수하며, 동영상 간 일관성을 높여줍니다. 또한 사용자는 제공된 파라미터를 바탕으로 최적의 결과를 위해 추가 조정을 할 수 있습니다.



### Multiple Information Prompt Learning for Cloth-Changing Person Re-Identification (https://arxiv.org/abs/2411.00330)
- **What's New**: 본 논문에서는 의류 변경 인물 재식별(cloth-changing person re-identification, CC-ReID) 문제를 해결하기 위한 새로운 다중 정보 프롬프트 학습(multiple information prompt learning, MIPL) 방법을 제안합니다. 이 방법은 여러 메시지의 공통 프롬프트 안내를 통해 강인한 신원 특징을 학습합니다.

- **Technical Details**: MIPL 프레임워크는 1) 의류 정보 분리 모듈(clothing information stripping, CIS)을 통해 이미지 특징에서 의류 정보를 효과적으로 분리하고, 2) 생물학적 정보 안내(bio-guided attention, BGA) 모듈을 통해 신원과 강한 상관관계를 가진 생물학적 주요 특징을 학습하며, 3) 이중 길이 하이브리드 패치(dual-length hybrid patch, DHP) 모듈을 통해 특징 편향의 영향을 최소화합니다.

- **Performance Highlights**: 제안된 MIPL 방법은 LTCC, Celeb-reID, Celeb-reID-light 및 CSCC 데이터셋에서 각각 74.8%, 73.3%, 66.0%, 88.1%의 rank-1 점수를 달성하며, AIM, ACID, SCNet과 비교 시 11.3%, 13.8%, 7.9%의 rank-1 향상을 보였습니다.



### Unified Generative and Discriminative Training for Multi-modal Large Language Models (https://arxiv.org/abs/2411.00304)
- **What's New**: 본 논문에서는 Vision-Language Models (VLMs)의 두 가지 주요 훈련 패러다임인 generative training과 discriminative training의 장점을 통합하는 새로운 접근법을 제안합니다. 특히, 연속적으로 배치된 이미지-텍스트 시퀀스를 입력 샘플의 일반 형태로 간주하고, 구조 유도 훈련 전략을 도입하여 입력 샘플과 MLLM의 hidden state 간의 의미적 관계를 명확히 합니다.

- **Technical Details**: Sugar라고 불리는 우리 방법론은 Dynamic Time Warping 프레임워크를 활용하여 동적인 시퀀스 정렬 문제를 해결하고, 새로운 분산(discriminative) 커널을 통합함으로써 입력 샘플의 의미를 더 잘 구별할 수 있도록 지원합니다. 이 과정에서 MLLM이 전역 의미와 미세한 세부 사항을 효과적으로 포착할 수 있도록 합니다.

- **Performance Highlights**: 우리는 광범위한 실험을 통해 우리의 접근법이 여러 생성 작업에서 최첨단 성능을 달성했음을 입증했습니다. 특히, 복잡한 다중모달 이해 작업 및 미세한 의미 구분에서 새로운 최첨단 결과를 달성하였으며, interleaved retrieval 및 fine-grained retrieval 작업에서 CLIP을 크게 초월하는 성능을 보였습니다.



### RadFlag: A Black-Box Hallucination Detection Method for Medical Vision Language Models (https://arxiv.org/abs/2411.00299)
Comments:
          15 pages, 8 figures

- **What's New**: RadFlag는 의료 영상에서 생성된 방사선 보고서의 정확성을 높이기 위한 새로운 방법으로, VLMs(비전 언어 모델)에서 발생하는 환각(hallucination)을 탐지하여 제거하는 기술입니다.

- **Technical Details**: 이 방법은 샘플링 기반의 플래깅 기법을 사용하여 여러 보고서를 샘플링하고, LLM(대형 언어 모델)을 통해 일관성이 낮은 주장을 식별합니다. 이로써 환각으로 의심되는 주장을 플래그(flag)하여 추가 검토를 진행하거나 자동 거부할 수 있습니다. 일반적으로 RadFlag는 온도 매개변수만 필요한 블랙박스 시스템으로, 다양한 방사선 보고서 생성 모델에서 호환됩니다.

- **Performance Highlights**: RadFlag는 보고서 생성 모델 Medversa를 활용하여 28%의 환각 문장을 정확하게 플래그 하며, 플래그 정확도는 73%를 기록했습니다. 보고서 레벨에서는 208개의 생성된 보고서를 분석하여, 플래그된 세트에서는 평균 4.2개의 환각이 발견된 반면, 수용된 세트에서는 평균 1.9개의 환각만 존재했습니다.



### Detection and tracking of gas plumes in LWIR hyperspectral video sequence data (https://arxiv.org/abs/2411.00281)
- **What's New**: 이 연구는 화학 가스 플룸의 감지를 위한 하이퍼스펙트럴(hyperspectral) 비디오 시퀀스를 효과적으로 시각화하고, 전처리된 비디오에서의 분할(segmentation) 기법의 효과를 조사하는 새로운 방법을 제시합니다.

- **Technical Details**: 본 연구에서는 PCA(Principal Components Analysis)를 사용하여 하이퍼스펙트럴 비디오의 차원을 축소하고, 그 후 Midway histogram equalization 방법을 사용하여 프레임 간 플리커(flicker) 문제를 완화합니다. 이후 K-means, spectral clustering, 그리고 Ginzburg-Landau 기능을 최소화하는 수정된 MBO 방법을 사용하여 화학 플룸의 분할을 비교합니다.

- **Performance Highlights**: 이 연구의 제안된 방법은 전통적인 RGB 영상보다 더 정확하고 효과적으로 화학 가스 플룸을 감지할 수 있는 가능성을 보여주었으며, 다양한 클러스터링 기법 간의 성능 비교를 통해 각 방법의 장단점을 분석했습니다.



### Adaptive Residual Transformation for Enhanced Feature-Based OOD Detection in SAR Imagery (https://arxiv.org/abs/2411.00274)
- **What's New**: 이 연구는 Synthetic Aperture Radar (SAR) 이미지에서 Unknown Target Detection을 위한 새로운 접근 방식을 제안합니다. 기존 feature-based OOD (Out-of-Distribution) 감지 방식을 class-localized feature-residual 기반 방법으로 변형하여 다양한 Unknown Target의 분포 조건에서 안정성을 향상시킵니다.

- **Technical Details**: 제안된 방법은 feature 정보를 localized feature-residual로 변환하여 각 클래스별로 feature 벡터의 클래스 내 및 클래스 간 차이를 계산합니다. 이 방식은 SAR 이미지의 특수한 특성을 고려하여 in-distribution (ID) 데이터와 OOD 데이터를 좀 더 견고하게 구분할 수 있는 기준 공간을 제공합니다.

- **Performance Highlights**: 이 접근 방식은 실제 세계의 SAR 시나리오에서 잠재적인 성과를 보여줍니다. 높은 노이즈와 혼잡한 환경에서도 효과적으로 적응하여 Unknown Target을 탐지할 수 있는 가능성을 확인했습니다.



### IO Transformer: Evaluating SwinV2-Based Reward Models for Computer Vision (https://arxiv.org/abs/2411.00252)
Comments:
          15 pages, 3 figures, 2 tables

- **What's New**: 이 논문은 SwinV2 기반의 보상 모델인 Input-Output Transformer (IO Transformer)와 Output Transformer를 소개하며, 이 모델들이 다른 모델의 출력 품질을 평가하는 데 어떻게 활용될 수 있는지를 다룹니다.

- **Technical Details**: IO Transformer는 입력과 출력 간의 관계를 평가하여 모델의 예측 품질을 평가할 수 있는 아키텍처로, 특히 이진 이미지 분할(task)에서 뛰어난 성과를 보입니다. SwinV2 아키텍처를 수정하며, IO Transformer는 입력에 따라 의존성이 높은 출력 모델 평가에 최적화 되어 있습니다.

- **Performance Highlights**: 입력에 의존적인 작업에서 IO Transformer는 Change Dataset 25 (CD25)에서 완벽한 평가 정확도를 달성하였으며, IO Segmentation Dataset에서 95.41%의 점수로 Swin V2가 IO Transformer를 초과합니다.



### ResiDual Transformer Alignment with Spectral Decomposition (https://arxiv.org/abs/2411.00246)
- **What's New**: 이 논문은 비전 변환기(vision transformers)의 residual streams를 통한 분석을 통해, attention heads가 특정 작업 또는 입력 속성에 특화되는 놀라운 현상을 다룹니다. 연구자는 ResiDual이라는 새로운 기법을 도입하여, residual stream의 스펙트럼 정렬을 통해 특정 작업에 맞는 신호를 강화할 수 있음을 보여줍니다.

- **Technical Details**: 기술적으로 이 논문은 다양한 비전 변환기 모델을 대상으로 residual 단위의 기하학적 구조를 조사하고, 여러 데이터 분포에서 attention heads의 특화된 역할을 정량적으로 분석합니다. 주요 컴포넌트를 기반으로 한 스펙트럼 분석 방법을 도입하여, residual 단위의 유사성을 측정하고, CLIP과 같은 비전-언어 모델에서 특정 직원의 중요성을 강조합니다.

- **Performance Highlights**: 제안된 ResiDual 기법은 전체 파인튜닝 없이도 모델 성능을 크게 향상시키는 데 기여하며, 최소한의 매개변수 오버헤드로 경쟁력 있는 성능을 달성할 수 있음을 보여줍니다. 이 연구는 50개 이상의 사전 훈련된 네트워크와 데이터셋 조합에서 확장 가능한 결과를 보였습니다.



### Aquatic-GS: A Hybrid 3D Representation for Underwater Scenes (https://arxiv.org/abs/2411.00239)
Comments:
          13 pages, 7 figures

- **What's New**: 이 논문에서는 Aquatic-GS라는 새로운 하이브리드 3D 표현 방법을 제안하여 수중 장면에서 물체와 수중 매체를 동시에 효과적으로 나타내려고 합니다. 이 방법은 Neural Water Field(NWF)를 사용하여 수중의 파라미터를 암묵적으로 모델링하며, 최신 3D Gaussian Splatting(3DGS)을 확장하여 물체를 명시적으로 모델링합니다.

- **Technical Details**: Aquatic-GS는 물의 비선형 속성을 고려하여 Neural Water Field(NWF)를 설계하고, 3DGS의 효율성을 활용하여 물체의 실제 외관과 기하학을 포착합니다. 또한 Depth-Guided Optimization(DGO) 메커니즘을 도입하여 장면의 정밀한 기하학적 표현을 위한 보조 지침으로 의사 깊이 맵을 사용합니다. 물리 기반의 수중 이미지 형성 모델을 통해 이 두 요소가 통합됩니다.

- **Performance Highlights**: Aquatic-GS는 세 가지 실제 수중 데이터 세트와 시뮬레이션 데이터 세트에서 수중 새로운 뷰 합성(NVS) 및 수중 이미지 복원(UIR) 작업에서 성능을 평가한 결과, 410배의 렌더링 속도 향상과 함께 최고의 렌더링 품질을 달성했습니다. 또한 색상 보정, 세부 복구 및 안정성에 있어 대표적인 탈수 방법을 초월하는 성능을 보여줍니다.



### Fashion-VDM: Video Diffusion Model for Virtual Try-On (https://arxiv.org/abs/2411.00225)
Comments:
          Accepted to SIGGRAPH Asia 2025

- **What's New**: 이 논문은 Fashion-VDM이라는 비디오 확산 모델(VDM)을 제안하여 가상 착용 비디오를 생성하는 방법을 소개합니다. 기존 비디오 가상 착용 방법의 문제점을 해결하며, 높은 품질의 착용 비디오를 생성할 수 있습니다.

- **Technical Details**: Fashion-VDM은 단일 네트워크로 작동하는 확산 기반 접근 방식입니다. 3D-convolution 및 temporal attention 블록을 활용하여 M&M VTO 아키텍처를 확장하고, 64프레임의 비디오에서 시간적 일관성을 유지하기 위해 점진적인 훈련 방식을 취합니다. split classifier-free guidance (split-CFG)를 도입하여 입력 신호에 대한 제어력을 강화합니다.

- **Performance Highlights**: 실험 결과, Fashion-VDM은 기존 방법들보다 월등히 성능이 우수하며, 높은 품질의 착용 비디오를 생성합니다. 우리는 Fashion-VDM이 비디오 가상 착용 분야에서 새로운 최첨단 모델임을 입증했습니다.



### Scale-Aware Recognition in Satellite Images under Resource Constrain (https://arxiv.org/abs/2411.00210)
Comments:
          15,4

- **What's New**: 이 논문에서는 위성 이미징에서의 개념 인식을 위한 새로운 접근 방식을 제시하고 있으며, 고해상도(High Resolution, HR) 및 저해상도(Low Resolution, LR) 이미지를 효율적으로 사용하기 위한 세 가지 주요 구성 요소를 소개합니다: 1) HR 이미지를 통해 훈련된 모델의 지식을 LR 모델로 증류하는 기법, 2) 모델 불일치에 기반한 HR 이미지 샘플링 전략, 3) 개념의 '스케일'을 추리하기 위한 LLM 기반 접근 방식.

- **Technical Details**: 제안된 시스템은 개념 인식의 스케일을 인식하고, 정확성을 높이며, 예산 제약을 준수하는 고유한 프레임워크를 기반으로 합니다. 제안된 방법론은 다양한 인식 모델(감독 및 개방형 어휘)과 여러 위성 모드에서 평가되었으며, HR 이미지를 항상 사용하는 것보다 정확도를 13포인트 향상시키면서 HR 이미지 사용량을 5배 줄였습니다. 이러한 접근 방식은 비용과 정확성 간의 균형을 자동으로 최적화하여 수행됩니다.

- **Performance Highlights**: 제안된 방법은 순수 HR 기준선보다 최대 26.3%의 정확도 향상을 가지며, 비용을 76.3% 줄이면서도 순간적으로 뛰어난 성능을 보였습니다. 또한, 본 연구는 다른 기존 연구보다 25포인트 이상 개선된 결과를 보여주었습니다.



### Semantic Knowledge Distillation for Onboard Satellite Earth Observation Image Classification (https://arxiv.org/abs/2411.00209)
Comments:
          Under revisions

- **What's New**: 본 연구는 자원 제약 환경에서 효율적인 지구 관측(Earth Observation, EO) 이미지 분류에 맞춰진 혁신적인 동적 가중치 지식 증류(Knowledge Distillation, KD) 프레임워크를 제시합니다.

- **Technical Details**: EfficientViT 및 MobileViT를 교사 모델로 사용하여, ResNet8과 ResNet16 같은 경량 학생 모델이 90% 이상의 정확도, 정밀도 및 재현율을 달성할 수 있도록 지원합니다. 우리의 적응형 가중치 메커니즘은 각 교사 모델의 신뢰도에 따라 동적으로 반응하여, 학생 모델이 보다 신뢰할 수 있는 지식 출처를 우선시할 수 있게 합니다.

- **Performance Highlights**: ResNet8은 파라미터를 97.5% 줄이고, FLOPs를 96.7% 감소시키며, 전력 소모를 86.2% 줄이고, 추론 속도를 MobileViT보다 63.5% 향상시키는 등 상당한 효율성 개선을 달성하였습니다.



### Evaluating the Evolution of YOLO (You Only Look Once) Models: A Comprehensive Benchmark Study of YOLO11 and Its Predecessors (https://arxiv.org/abs/2411.00201)
Comments:
          20 pages

- **What's New**: 본 연구는 YOLO(You Only Look Once) 알고리즘의 다양한 버전, 특히 YOLO11의 성능을 종합적으로 평가한 최초의 연구로, 여러 데이터셋에서의 성능 비교를 통해 YOLO 알고리즘의 진화를 분석합니다.

- **Technical Details**: 연구는 Traffic Signs, African Wildlife, Ships와 같은 세 가지 데이터셋에서의 YOLO 알고리즘 성능을 평가하며, Precision, Recall, Mean Average Precision (mAP), 처리 시간, GFLOPs 수 및 모델 크기와 같은 포괄적인 메트릭을 사용합니다. YOLO11의 mAP50-95 점수는 Traffic Signs에서 0.795, African Wildlife에서 0.81, Ships 데이터셋에서 0.325에 도달하며, 평균 추론 시간은 2.4ms입니다.

- **Performance Highlights**: YOLO11은 높은 정확도와 효율성을 보여줍니다. 특히, YOLO11m 모델은 38.8Mb의 모델 크기로 우수한 성능을 발휘하며, YOLOv9, YOLOv10보다 속도와 계산 효율성에서 뛰어난 성능을 보입니다. 이러한 결과는 산업 및 학계에 중요한 통찰력을 제공하여 다양한 애플리케이션에 적합한 YOLO 알고리즘 선택에 도움을 줍니다.



### Whole-Herd Elephant Pose Estimation from Drone Data for Collective Behavior Analysis (https://arxiv.org/abs/2411.00196)
Comments:
          Accepted to CV4Animals: Computer Vision for Animal Behavior Tracking and Modeling Workshop in conjunction with Computer Vision and Pattern Recognition 2024

- **What's New**: 이 연구는 드론에서 수집된 데이터를 기반으로 야생에서 코끼리 행동을 연구하는 자동 포즈 추정(automated pose estimation)의 혁신적인 응용을 보여줍니다.  케냐의 샘부루 국립 보호 구역에서 촬영한 비디오 영상을 활용하였고, DeepLabCut과 YOLO-NAS-Pose라는 두 가지 포즈 추정 워크플로우를 평가하였습니다.

- **Technical Details**: Drone 기술을 통해 다수의 코끼리를 한 프레임에서 관찰하였으며, 촬영 중 드론은 정지 상태에서 특정 높이를 유지했습니다. 본 연구에서는 총 23개의 비디오와 133개의 프레임을 통해 1308마리의 코끼리를 분석하였습니다. YOLO-NAS-Pose와 DeepLabCut 모델을 사용하여 코끼리의 주요 포인트(머리, 척추, 귀 등)를 추정하였습니다.

- **Performance Highlights**: YOLO-NAS-Pose 모델이 DeepLabCut보다 RMSE, PCK, OKS와 같은 지표에서 우수한 성능을 보였으며, 객체 탐지 평가에서도 DeepLabCut을 초과하는 성과를 나타냈습니다. 이를 통해 야생 행동 연구와 드론 모니터링의 새로운 접근 방식을 제시하며, 동물 보존에 대한 중요한 시사점을 제공합니다.



### Optical Lens Attack on Monocular Depth Estimation for Autonomous Driving (https://arxiv.org/abs/2411.00192)
Comments:
          28 pages. arXiv admin note: substantial text overlap with arXiv:2409.17376

- **What's New**: 본 논문에서는 모노큘러 깊이 추정(MDE) 알고리즘의 취약점을 활용한 새로운 물리적 공격인 LensAttack을 소개합니다. LensAttack은 자율 주행(Autonomous Driving) 차량의 카메라에 광학 렌즈를 배치하여 인식된 물체의 깊이를 조작합니다.

- **Technical Details**: LensAttack은 볼록 렌즈 공격과 오목 렌즈 공격 두 가지 형식을 포함하며, 각기 다른 광학 렌즈를 사용하여 잘못된 깊이 인식을 유발합니다. 공격의 수학적 모델을 개발하고 시뮬레이션 및 실제 평가를 통해 최신 MDE 모델에 대한 공격의 효과를 평가합니다. CARLA 플랫폼을 이용한 종합 시뮬레이션을 통해 공격의 영향을 평가합니다.

- **Performance Highlights**: 연구 결과, LensAttack은 자율 주행 시스템의 깊이 추정 과정을 심각하게 방해할 수 있으며, 이로 인해 시스템의 신뢰성과 안전성에 심각한 위협을 가합니다. 최적화된 공격 절차를 통해 다양한 광학 렌즈 매개변수에서 공격의 정확도가 향상됨을 보여주었습니다. 실험 결과 볼록 렌즈 공격은 평균 오류율 11.48%를 기록하였습니다.



### Clinical Evaluation of Medical Image Synthesis: A Case Study in Wireless Capsule Endoscopy (https://arxiv.org/abs/2411.00178)
- **What's New**: 이 논문은 임상 연구 및 훈련을 위한 데이터 공유의 필요성을 강조하며, 인공지능(AI) 모델을 활용한 합성 데이터 생성(SDG)이 개인정보 보호 장벽을 극복할 수 있음을 보여줍니다. 특히, 무선 캡슐 내시경(WCE) 이미지를 사용하여 염증성 장질환(IBD)을 진단하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 연구는 합성 이미지를 체계적으로 평가하기 위한 프로토콜을 제시하고, TIDE-II라는 새로운 변별 오토인코더 기반 모델을 적용하여 고해상도 WCE 이미지를 합성하였습니다. 최종적으로 10명의 국제 WCE 전문가에 의해 이미지 품질, 다양성, 사실성 및 임상 의사결정에 대한 종합적인 정성적 평가가 이루어졌습니다.

- **Performance Highlights**: TIDE-II 모델은 임상적으로 중요한 WCE 이미지를 생성하여 데이터 부족 문제를 해결하고 진단 도구를 향상시키는 데 기여하는 것으로 나타났습니다. 제안된 프로토콜은 의학 이미지 생성 기술에 대한 향후 연구에 대한 참고 자료로 활용될 수 있습니다.



### Pedestrian Trajectory Prediction with Missing Data: Datasets, Imputation, and Benchmarking (https://arxiv.org/abs/2411.00174)
Comments:
          Accepted at NeurIPS 2024

- **What's New**: Pedestrian trajectory prediction을 위한 새로운 데이터셋인 TrajImpute를 소개하며, 이는 관측된 궤적에서 누락된 좌표를 시뮬레이션하는 기능을 제공합니다. 이는 실제 환경에서의 데이터 부족 문제를 해결하는 데 중점을 두고 있습니다.

- **Technical Details**: TrajImpute는 'easy'와 'hard' 모드 두 가지 데이터 생성 전략을 통해 누락된 좌표를 시뮬레이션합니다. 연구자들은 여러 기존의 imputation 방법을 평가하여 누락된 좌표를 복원하고, 이를 바탕으로 경로 예측 모델의 성능을 평가했습니다. 데이터셋은 Creative Commons CC BY-SA 4.0 라이센스 하에 공개됩니다.

- **Performance Highlights**: 이 연구는 기존의 여러 imputation 방법들을 벤치마킹하고, 이들이 TrajImpute 데이터셋에 대해 어떻게 작동하는지를 분석했습니다. 연구 결과는 미래의 pedestrian trajectory prediction 연구와 알고리즘 개선에 기여할 중요한 통찰력을 제공합니다.



### SeafloorAI: A Large-scale Vision-Language Dataset for Seafloor Geological Survey (https://arxiv.org/abs/2411.00172)
- **What's New**: 이 논문에서는 해양 과학 분야에서의 머신 러닝 모델 발전을 저해하는 주요 장애물인 AI 준비 데이터셋의 부족을 해결하기 위해 SeafloorAI를 소개합니다.

- **Technical Details**: SeafloorAI는 5개의 지질층에 걸친 해저 지도 작성을 위한 최초의 광범위한 AI 준비 데이터셋으로, 해양 과학자들과 협력하여 큐레이팅되었습니다. 이 데이터셋은 62개의 지오-분포 조사가 포함되어 있으며, 총 17,300 제곱킬로미터에 걸쳐 696K개의 소나 이미지, 827K개의 주석이 달린 세그멘테이션 마스크, 696K개의 자세한 언어 설명 및 약 7M개의 질문-답변 쌍을 포함하고 있습니다.

- **Performance Highlights**: 또한, 데이터 처리 소스 코드를 공개하여 해양 과학 커뮤니티가 데이터 풀을 풍부하게 하고 머신 러닝 커뮤니티가 보다 강력한 모델을 개발하도록 영감을 주는 것을 목표로 하고 있습니다.



### Aerial Flood Scene Classification Using Fine-Tuned Attention-based Architecture for Flood-Prone Countries in South Asia (https://arxiv.org/abs/2411.00169)
- **What's New**: 본 논문에서는 남아시아의 홍수 구역을 식별하기 위해 새로운 항공 이미지 데이터셋을 만들고, 이를 기반으로 한 이미지 분류 기법을 제안합니다. Compact Convolutional Transformer (CCT)를 활용하여 빠르고 정확하게 구역을 분류함으로써 인명 구조 작전의 효율성을 높이려는 목표를 가지고 있습니다.

- **Technical Details**: 제안된 데이터셋은 '홍수', '주택이 있는 홍수', '인간이 있는 홍수', '홍수 없음'의 4개 클래스 로 구분됩니다. 실험 결과, fine-tuned CCT 모델은 98.62%의 정확도와 98.50%의 macro average precision을 달성했습니다. YOLOv8 모델을 사용하여 주택과 인간을 탐지하고, 여러 개의 Transformer 아키텍처 (Vision Transformer, Swin Transformer 등)와 CNN 모델을 비교 분석했습니다.

- **Performance Highlights**: CCT와 DCECNN (Deep Custom Ensembled Convolutional Neural Network)을 통해 각각 98.62%와 98.78%라는 높은 정확도를 기록했습니다. ViT, Swin Transformer, EANet 등 다른 모델의 정확도는 각각 88.66%, 84.74%, 66.56%로 저조했습니다. 이를 통해 제안된 CCT의 우수를 입증했습니다.



### A Recipe for Geometry-Aware 3D Mesh Transformers (https://arxiv.org/abs/2411.00164)
- **What's New**: 이 연구에서는 3D mesh 데이터 처리를 위한 새로운 기법인 GeoTransformer를 소개합니다. 기존의 기법들이 JSON으로 불필요한 변환 및 재샘플링에 의존하는 경향이 있는 반면, GeoTransformer는 구조적 인코딩(Structural Embedding)과 스펙트럼 보존(tokenization) 접근 방식을 통합하여 이러한 문제를 해결하고 있습니다.

- **Technical Details**: GeoTransformer는 다음과 같은 요소들을 포함합니다: 1) 스펙트럼 보존을 위한 패치 기반 토크나이제이션, 2) 각 패치의 노드 수에 맞춘 구조적 임베딩, 3) Heat Kernel Signature(HKS)를 활용한 상대적 위치 정보를 위한 임베딩 방식, 4) U-Net 스타일의 네트워크 아키텍처로 패치 크기 임베딩 통합. 또한, 지오데식 주의(masking)를 통해 특정 토큰에 대한 집중을 향상시킵니다.

- **Performance Highlights**: 이 모델은 기존의 단순 MLP 모델에 비해 향상된 성능을 보여주며, 특히 세분화(segmentation) 및 분류(classification) 작업에서 뛰어난 효율성을 입증했습니다. 연구의 결과는 특히 구조적 및 위치 임베딩을 중요하게 다루며, 향후 메쉬 및 포인트 클라우드 트랜스포머에 대한 연구를 위한 기초 자료를 제공합니다.



### Using Deep Neural Networks to Quantify Parking Dwell Tim (https://arxiv.org/abs/2411.00158)
Comments:
          Paper accepted to the 2024 International Conference on Machine Learning and Applications

- **What's New**: 이 논문에서는 주차 공간의 차량 체류 시간을 자동으로 측정하기 위해 두 개의 심층 신경망(Deep Neural Networks)을 결합한 새로운 방법을 제안합니다. 특히 직면하는 주요 문제는 저해상도 카메라, 조명 변화 및 날씨 영향으로 인한 이미지 수집의 어려움입니다.

- **Technical Details**: 제안된 시스템은 먼저 심층 분류 네트워크를 사용하여 주차 공간이 점유되었는지 비어있는지를 판별합니다. 이어서, 시암 네트워크(Siamese Network)를 통해 주차된 차량이 이전 이미지와 동일한 지를 확인합니다. 이 전체 파이프라인은 Convolutional Neural Network(CNN) 기반의 시암 네트워크를 포함하며, 차량의 체류 시간을 업데이트하는 알고리즘 또한 포함합니다.

- **Performance Highlights**: 실험 결과, 완벽한 분류기를 사용할 경우 주차된 차량의 체류 시간을 75% 정확도로 예측하는 것이 가능하다고 밝혔습니다. 그러나 실제 환경에서 분류기를 사용할 경우 예측 품질이 감소하여 49%의 정확도를 기록했습니다. 이 연구는 시암 네트워크의 가능성을 보여주지만 초기 파이프라인에서 사용되는 분류기의 품질에 영향을 받는다는 점을 강조합니다.



### NIMBA: Towards Robust and Principled Processing of Point Clouds With SSMs (https://arxiv.org/abs/2411.00151)
- **What's New**: 최근 논문에서는 점군(point cloud) 데이터를 효과적으로 처리하기 위한 새로운 방법인 NIMBA를 소개하였습니다. NIMBA는 점군을 1D 시퀀스로 변환하면서 3D 공간 구조를 유지하도록 설계되었습니다. 이는 데이터 복제를 필요로 하지 않으며, 기존 Mamba 모델의 순차 처리 방식에서 발생할 수 있는 문제를 해결합니다.

- **Technical Details**: NIMBA는 기존 SSM(State Space Model)와 유사한 구조로, 양방향 주문 의존 처리 방식 대신 점군 데이터의 비순차 구조를 통합하여 효율적인 순차 처리를 가능하게 합니다. 기존의 위치 임베딩(positional embeddings)을 사용할 필요 없이, 입력 데이터를 실행할 수 있으며, 일관된 성능 개선을 보입니다.

- **Performance Highlights**: NIMBA는 ModelNet40 및 ScanObjectNN 데이터셋에서 최신 기술 수준의 성능을 달성했으며, Transformer 기반 모델보다 정확도와 효율성을 뛰어넘었습니다. 이는 Mamba와 같은 SSM이 높은 해상도의 3D 데이터 처리에서도 효과적임을 입증합니다.



### Self-Ensembling Gaussian Splatting for Few-shot Novel View Synthesis (https://arxiv.org/abs/2411.00144)
- **What's New**: 이 논문에서는 자가 앙상블(self-ensembling) 기법을 도입하여 3D Gaussian Splatting (3DGS) 모델의 과적합(overfitting) 문제를 해결하고, 소수의 훈련 이미지에서도 향상된 성능을 보여줍니다.

- **Technical Details**: 이 논문에서 제안하는 새로운 접근법인 자기 앙상블 Gaussian Splatting (SE-GS)은 두 가지 모델인 𝚺-모델(Σ-model)과 𝚫-모델(Δ-model)로 구성됩니다. 𝚺-모델은 새로운 비주얼 이미지를 생성하며, 𝚫-모델은 Gaussian 매개변수 공간에서 시간적 샘플을 나타내고 불확실성을 기반으로 모델을 동적으로 섞어 다양성을 제공합니다.

- **Performance Highlights**: LLFF, Mip-NeRF360, DTU, MVImgNet 데이터셋에서 몇 장의 훈련 이미지를 사용했음에도 불구하고, SE-GS가 기존 최첨단 기술들보다 더 나은 성능을 발휘하며, 노이즈를 줄이고 안정성을 높여 줍니다.



### Muscles in Time: Learning to Understand Human Motion by Simulating Muscle Activations (https://arxiv.org/abs/2411.00128)
- **What's New**: 이번 연구에서는 Muscles in Time (MinT)라는 대규모 합성 근육 활성화 데이터셋을 개발하였습니다. 이 데이터셋은 생체역학적 모델을 이용하여 기존의 모션 캡처 데이터셋에 근육 활성화 시뮬레이션을 추가하여 생성되었습니다.

- **Technical Details**: MinT 데이터셋은 227명의 피험자와 402개의 시뮬레이션된 근육 섬유를 포함하여 9시간 이상의 시뮬레이션 데이터를 제공합니다. OpenSim 플랫폼을 활용하여 생체역학적 인간 모델에서 유도된 근육 활성화 데이터를 포함하고 있으며, 이는 인체 동작과 관련된 깊이 있는 통찰력을 제공합니다.

- **Performance Highlights**: 이 데이터셋을 이용하여 인체 자세 시퀀스를 기반으로 한 신경망 기반 근육 활성화 추정 결과를 보여주며, 두 가지 Sequence-to-Sequence 아키텍처를 적용하여 좋은 성능을 보였습니다. 이는 인체 동작 이해를 위한 효과적인 데이터 세트로 기능할 수 있음을 의미합니다.



### How Good Are We? Evaluating Cell AI Foundation Models in Kidney Pathology with Human-in-the-Loop Enrichmen (https://arxiv.org/abs/2411.00078)
- **What's New**: 이번 연구는 디지털 병리학 분야에서의 AI 기반 세포 모델의 성능을 최초로 다각적으로 평가하고, 인간의 개입을 통해 데이터 보강 전략을 개발하여 모델 성능을 향상시키려는 시도를 했습니다.

- **Technical Details**: 이 연구에서는 2,542개의 신장 전체 슬라이드 이미지(Whole Slide Images, WSIs)를 포함한 다센터, 다질병, 다종의 데이터셋을 구축하였습니다. 세 가지 최신(cell foundation) 모델인 Cellpose, StarDist, CellViT를 평가하고, 데이터 보강 전략으로는 사람의 개입을 통한 잘못된(predictions) 예측 패치를 수정하고 여러 모델의 예측을 결합하여 성능을 향상시키는 방법을 사용하였습니다.

- **Performance Highlights**: 세 가지 모델 모두 데이터 보강 후 성능이 개선되었고, StarDist 모델이 최고 F1 점수 0.82를 기록했습니다. 하지만 F1 점수가 가장 높은 기본 모델(CellViT)은 미세 조정 후 최상의 세분화 결과를 내지 못했습니다. 이는 ‘좋은’ 및 ‘나쁜’ 이미지 패치를 결합하는 전략이 가장 효과적임을 보여줍니다.



### PathoGen-X: A Cross-Modal Genomic Feature Trans-Align Network for Enhanced Survival Prediction from Histopathology Images (https://arxiv.org/abs/2411.00749)
- **What's New**: 이번 연구에서는 암 치료를 위한 개인 맞춤형 생존 예측을 개선하기 위해 PathoGen-X라는 크로스 모달 유전체 특성 변환 및 정렬 네트워크를 제안하였습니다.

- **Technical Details**: PathoGen-X는 트랜스포머 기반의 네트워크를 사용하여 이미지 특성을 유전체 특성 공간으로 정렬하고 변환합니다. 이 방법은 두 가지 모달리티를 결합하여 이미징 데이터에 대해 더 강한 신호를 제공하며, 적은 수의 짝지어진 샘플로도 효과적입니다.

- **Performance Highlights**: TCGA-BRCA, TCGA-LUAD, TCGA-GBM 데이터셋에서 평가한 결과, PathoGen-X는 생존 예측 성능이 뛰어난 것으로 나타났으며, 이미징 데이터만을 사용하여도 유전체 데이터로 훈련된 모델에 필적하는 성능을 발휘했습니다.



### Cross-Fundus Transformer for Multi-modal Diabetic Retinopathy Grading with Catarac (https://arxiv.org/abs/2411.00726)
Comments:
          10 pages, 4 figures

- **What's New**: 본 연구는 색깔 망막 사진(CFP)과 적외선 망막 사진(IFP) 정보를 융합하여 더욱 정확한 당뇨병성 망막병증(DR) 등급을 위한 다중 모달 딥러닝 프레임워크, Cross-Fundus Transformer (CFT)를 제안합니다.

- **Technical Details**: CFT는 ViT 기반의 이중 스트림 아키텍처로 CFP와 IFP 이미지를 융합하며, Cross-Fundus Attention (CFA) 모듈을 도입해 두 이미지 간의 대응 관계를 포착합니다. 자동으로 CFA 모듈을 통해 두 모달리티의 정보를 융합하여 DR을 진단합니다.

- **Performance Highlights**: 1,713쌍의 다중 모달 fundus 이미지로 구성된 임상 데이터셋을 기반으로 한 실험에서, 제안된 방법은 기존의 방법들보다 우월한 성능을 보여줍니다. 본 연구는 CFP와 IFP 이미지를 동시에 이용하여 DR을 자동으로 진단한 최초의 시도로, 최첨단 성능을 입증합니다.



### Why do we regularise in every iteration for imaging inverse problems? (https://arxiv.org/abs/2411.00688)
- **What's New**: ProxSkip 알고리즘을 활용하여 다양한 영상 역 문제(imaging inverse problems)에서 계산 시간 절약과 고품질 복원을 동시에 달성할 수 있는 가능성을 처음으로 탐구하였습니다. 새로운 PDHGSkip 버전도 제안하였습니다.

- **Technical Details**: ProxSkip 알고리즘은 매 반복 단계에서 정규화(proximal operator) 단계를 무작위로 건너뛰는 방식으로 작동하여 계산 시간(computational time)을 줄입니다. 이 알고리즘은 정규화 항의 평가에 대한 계산 오버헤드를 줄이면서 수렴(convergence)에 영향을 주지 않습니다. 논문에서는 다양한 영상 역 문제에 대해 ProxSkip의 효과성을 입증하기 위해 광범위한 수치 실험(numerical experiments)을 수행하였습니다.

- **Performance Highlights**: ProxSkip가 FISTA의 가속화된 비-스킵(non-skip) 버전보다 더 나은 성능을 보이는 것을 보여주었으며, 특히 실제 세계의 토모그래픽(tomographic) 애플리케이션에서도 잠재적인 계산 이점을 강조하였습니다.



### A Graph Attention-Guided Diffusion Model for Liver Vessel Segmentation (https://arxiv.org/abs/2411.00617)
Comments:
          This work has been submitted to the IEEE for possible publication

- **What's New**: 본 논문에서는 간 혈관 세분화를 위한 새로운 방법으로, 다중 스케일 그래프 주의 메커니즘을 활용한 확산 모델 기반 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 2D 확산 모델을 사용하고, 그래프 주의 계층을 통합하여 혈관 연속성을 향상시키며, 지역 집합 모듈을 통해 그래프 노드의 인접 특징을 통합하여 부드러운 전환을 유지합니다. 또한, 작은 혈관을 세분화하기 위해 다양한 스케일에서 특징을 추출합니다.

- **Performance Highlights**: 제안된 방법은 3D-ircadb-01 및 LiVS의 두 개의 공공 데이터셋에서 기존 방법에 비해 Dice 계수 및 민감도가 각각 11.67% 및 24.21%, 그리고 3.21% 및 9.11% 향상되었습니다. 또한, 연결성 면에서도 최상의 성능을 보였습니다. 이 방법은 작은 간 혈관 세분화에 있어 신뢰할 수 있는 성능을 보입니다.



### Tumor Location-weighted MRI-Report Contrastive Learning: A Framework for Improving the Explainability of Pediatric Brain Tumor Diagnosis (https://arxiv.org/abs/2411.00609)
- **What's New**: 이번 연구에서는 MRI 이미지와 방사선 보고서를 결합한 새로운 컨트라스티브 러닝(Contrastive Learning, CL) 프레임워크를 도입하였습니다. 이 프레임워크는 CNN의 설명 가능성을 향상시키는 것을 목표로 하며, 뇌 종양 진단의 정확성을 높이기 위해 종양 위치 정보를 통합하였습니다. 특히, 소아 저급 신경교종(pediatric Low-grade Glioma, pLGG)의 유전자 마커 분류에 초점을 맞추어 설명 가능성과 성능을 향상시켰습니다.

- **Technical Details**: 연구의 CL 아키텍처는 3D 뇌 MRI 스캔과 방사선 보고서를 사용하여 유용한 MRI 표현을 학습합니다. 종양 위치 정보를 통합하여 다양한 뇌 종양 분석 작업의 일반성을 향상시키는 데 중점을 두었습니다. 이 연구에서는 Dice 점수 31.1%와 테스트 분류 성능 87.7%를 달성하였으며, 이는 기존 기준 대비 유의미한 개선을 보여줍니다.

- **Performance Highlights**: 모델의 주의 맵과 수동 종양 분할 간의 Dice 점수는 31.1%로, 이를 통해 모델의 설명 가능성을 측정하였으며, 테스트 분류 성능은 87.7%로 확인되었습니다. 이러한 개선 사항은 방사선 전문의들 사이에서 모델에 대한 신뢰를 구축하여 임상 진료에의 통합을 촉진할 수 있음을 시사합니다.



### pcaGAN: Improving Posterior-Sampling cGANs via Principal Component Regularization (https://arxiv.org/abs/2411.00605)
Comments:
          To appear at NeurIPS 2024

- **What's New**: 본 연구에서는 이미지 복원 문제에서 포스터리어 샘플링을 위한 빠르고 정확한 조건부 생성 적대 신경망(conditional Generative Adversarial Network, cGAN)을 제안합니다. 이 모델은 이전 지식과 관측된 측정을 바탕으로 올바른 포스터리어 평균 및 공분산 행렬의 주성분을 추정할 수 있도록 하는 새로운 정규화를 통해 기존 방법론의 한계를 극복합니다.

- **Technical Details**: 제안된 cGAN은 이미지 복원 과정에서 생성된 여러 가설을 탐색하고, 이를 통해 불확실성을 정량화하며 인식/왜곡 간 균형을 맞추는 것을 목표로 합니다. 또한, 포스터리어 공분산 행렬의 흔적(trace) 및 주성분(K principal components) 분석을 중시하여 결과의 정확성을 높였습니다.

- **Performance Highlights**: 수치 실험 결과, 우리의 방법은 현대적인 cGAN 및 확산 모델(diffusion models)보다 이미지 복원 문제인 노이즈 제거(denoising), 대규모 인페인팅(large-scale inpainting), 가속 MRI 복원에서 우수한 성능을 보였습니다.



### Deep learning-based auto-contouring of organs/structures-at-risk for pediatric upper abdominal radiotherapy (https://arxiv.org/abs/2411.00594)
Comments:
          23 pages, 5 figures, 1 table. Submitted to Radiotherapy and Oncology (2024-11-01)

- **What's New**: 이번 연구에서는 소아 복부 상부 종양에서 위험 장기(organs-at-risk, OARs)를 delineate하기 위한 CT(computed tomography) 기반의 다기관 분할(segmentation) 모델을 개발했습니다. 이 모델은 다양한 데이터셋에서의 견고성을 평가하였습니다.

- **Technical Details**: 사용된 데이터셋은 189명의 소아 신장 종양 및 신경모세포종 환자의 수술 후 CT 이미지를 포함한 사내 데이터셋과 흉부-복부(thoracoabdominal) 영역을 포함하는 공공 데이터셋으로, 총 189개의 CT 이미지를 사용했습니다. 17개의 OARs가 delineated 되며, 9개는 전문의에 의해(Type 1), 나머지 8개는 TotalSegmentator를 통해 자동으로 분할(Type 2)되었습니다. 두 가지 모델(Model-PMC-UMCU와 Model-Combined)이 학습되었습니다.

- **Performance Highlights**: Model-PMC-UMCU는 9개의 OAR 중 5개에서 평균 DSC(Dice Similarity Coefficient) 값이 0.95 이상에 도달했습니다. 비장과 심장은 0.90에서 0.95 사이의 값을 보였고, 위-장 및 췌장은 0.90 이하의 값을 보였습니다. Model-Combined는 두 데이터셋에서 견고성이 향상된 결과를 보여주었습니다. 임상 평가에서는 전문의가 9개 Type 1 OAR 중 6개를 4점 이상, 8개 Type 2 OAR 중 6개를 3점 이상으로 평가하여 사용 가능성을 확인하였습니다.



### MAROON: A Framework for the Joint Characterization of Near-Field High-Resolution Radar and Optical Depth Imaging Techniques (https://arxiv.org/abs/2411.00527)
- **What's New**: 본 연구는 광학(depth) 및 레이더(radar) 센서의 융합 및 동시 특성을 비교 평가하는 새로운 접근 방식을 제시합니다. 특히, 여러 개의 깊이 센서를 사용하여 다중 모드(multimodal) 데이터셋(MAROON)을 생성하여 서로 다른 센서의 성능을 체계적으로 분석하였습니다.

- **Technical Details**: 본 논문에서는 네 개의 깊이 이미저(depth imager)로부터 데이터 수집이 이루어졌습니다. 이들은 활성 및 수동 스테레오, 근적외선(Near-Infrared, NIR) ToF(Time-of-Flight), 밀리미터파 범위의 RF ToF를 포함합니다. 본 연구는 다양한 물체 재질 및 기하학적 형태를 기준으로 깊이 측정의 포괄적인 평가를 수행하였습니다.

- **Performance Highlights**: 우리의 평가 결과, RF ToF 센서는 광학 센서에 비해 완전도가 낮고, 부분 전송 재료와의 상호작용에 따라 체계적인 깊이 오류를 보이는 것으로 확인되었습니다. MAROON 데이터셋은 다양한 객체의 깊이 감지 성능을 비교하는 데 유용한 자원으로 제공될 예정입니다.



### Class Incremental Learning with Task-Specific Batch Normalization and Out-of-Distribution Detection (https://arxiv.org/abs/2411.00430)
Comments:
          10 pages, 4 figures, 4 tables, in submission to IEEE Transaction of Multimedia Journal (TMM)

- **What's New**: 이 연구는 이미지 분류를 위한 점진적 학습(incremental learning)에 중점을 두고 있으며, 이전 데이터에 대한 접근이 제한되었을 때 모든 학습된 지식을 잃지 않도록 하는 방법을 탐구합니다. 특히, 본 연구는 CIL(클래스 점진적 학습)에서 사용할 수 있는 새로운 접근법을 제시하며, TIL(작업 점진적 학습) 방법을 CIL로 확장하는 데 필요한 task-ID 예측을 포함합니다.

- **Technical Details**: 이 논문에서는 각 분류 헤드에 'unknown' 클래스를 추가하여 task-ID를 예측하는 방법을 제안합니다. task-specific batch normalization(BN)을 통해 다양한 작업 간 출력 특징 맵의 분포를 조정하여 모델의 안정성을 높이고, 매개변수 증가를 효과적으로 관리합니다. 이는 BN이 convolutional kernels에 비해 매개변수가 적기 때문에 가능하며, 새로운 작업이 추가될 때 BN 층만 수정하면 됩니다.

- **Performance Highlights**: 제안된 방법은 두 개의 의료 이미지 데이터셋과 하나의 자연 이미지 데이터셋에서 state-of-the-art 성능을 달성하였으며, 기존 방법보다 전반적인 성능이 우수하며 모델의 안정성, 가변성(plasticity), 매개변수 증가 간의 균형을 잘 이루고 있습니다.



### Advantages of Neural Population Coding for Deep Learning (https://arxiv.org/abs/2411.00393)
- **What's New**: 이 논문에서는 신경망의 출력층에서 인구 코드를 사용하는 이점을 조사하고, 이를 단일 뉴런 출력 및 원-핫 벡터와 비교하여 결과의 강건성과 정확성을 향상 시킨다고 제안합니다.

- **Technical Details**: 연구는 단일 변수, 인구 코드, 원-핫 벡터 출력의 노이즈 강건성을 비교하고, 이론적인 분석과 함께 T-LESS 데이터셋을 통해 이미지에서 객체 방향을 예측하는 작업을 수행하여 인구 코드가 불확실한 출력을 처리하는 데 어떻게 기여하는지를 보여줍니다.

- **Performance Highlights**: 결과적으로, 인구 코드를 사용하면 객체 지향 예측의 정확성이 향상되며, 특히 대칭 객체로부터의 모호한 자세를 처리하는 데 유리함을 보입니다.



### A Simple Remedy for Dataset Bias via Self-Influence: A Mislabeled Sample Perspectiv (https://arxiv.org/abs/2411.00360)
- **What's New**: 이 논문은 편향된 데이터에서 공정성을 추구하는 방향으로, 잘못 레이블링된 샘플과 편향-충돌 샘플(bias-conflicting samples) 간의 유사성을 이용하여 새로운 접근법을 제시합니다. 특히, Influence Function을 활용해 편향-충돌 샘플을 식별하고, 이를 바탕으로 편향된 모델을 수정하는 간단하면서도 효과적인 방법론을 개발하였습니다.

- **Technical Details**: 이 연구에서는 Bias-Conditioned Self-Influence (BCSI)라는 새로운 개념을 도입하여, 편향 데이터셋에서 편향-충돌 샘플을 효과적으로 구별하기 위한 중요한 조건을 밝힙니다. BCSI는 필수 조건을 만족할 때 활성화되며, 편향된 모델을 수용할 수 있는 소규모 중심 세트를 구성하는 데 활용됩니다. 이를 통해 SEO(Structured Orthogonalization) 방식으로 모델 수정을 수행합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 기존 방식들보다 개선된 성과를 보였으며, 이미 편향 수정이 이루어진 모델에 대해서도 추가적인 수정이 가능함을 입증하였습니다. 이로 인해, 다양한 편향 정도에서 편향된 모델을 효과적으로 교정할 수 있는 가능성을 제시합니다.



### All-frequency Full-body Human Image Relighting (https://arxiv.org/abs/2411.00356)
Comments:
          project page: [this URL](this https URL)

- **What's New**: 본 논문은 인물 이미지 재조명을 위한 두 단계의 새로운 방법을 제안합니다. 기존의 기법들이 물리적인 음영 원리를 고려하지 않고 신경망을 사용하여 조명을 근사화하는 반면, 본 방법은 물리적으로 기반한 그림자와 음영을 재현할 수 있는 혁신적인 접근법을 제공합니다.

- **Technical Details**: 제안된 방법은 두 가지 단계로 구성됩니다. 첫 번째 단계에서는 신경망을 활용하여 단일 이미지에서 물리 기반 음영을 통한 역 렌더링을 수행합니다. 두 번째 단계에서는 각 영역 광원을 위한 그림자를 계산하고 결국 이미지를 렌더링합니다. 환경 조명의 근사를 위해 고정된 수의 영역 광원을 사용하며, 부드러운 그림자 맵핑을 미분 가능하게 만드는 방법을 제안합니다.

- **Performance Highlights**: 본 논문에서는 제안된 방법이 기존의 방법들에 비해 모든 주파수의 그림자와 음영을 재현하는 데 더 우수한 성능을 보임을 입증하였습니다. 특히 동적 조명 하에서도 결과가 더 플라시블하고 안정적인 리라이트를 가능하게 합니다.



### SpineFM: Leveraging Foundation Models for Automatic Spine X-ray Segmentation (https://arxiv.org/abs/2411.00326)
Comments:
          4 pages, 3 figures, submitted to ISBI 2025

- **What's New**: 본 논문에서는 SpineFM이라는 새로운 파이프라인을 소개합니다. 이 파이프라인은 경추(Cervical) 및 요추(Lumbar) 척추 X-레이에서 척추체를 자동으로 세분화(segmentation)하고 식별하는 역량에서 최첨단 성능을 달성합니다.

- **Technical Details**: SpineFM은 척추의 규칙적인 기하학을 활용하며, 독창적인 inductive 프로세스를 통해 척추 기둥을 따라 각 척추의 위치를 순차적으로 추론합니다. Vertebrae는 CNN기반 모델과는 달리 Medical-SAM-Adaptor를 사용해 세분화됩니다. 우리의 방법은 두 개의 공개 척추 X-Ray 데이터셋에서 검증되었습니다.

- **Performance Highlights**: 우리는 두 개의 데이터셋에서 각각 97.8% 및 99.6%의 주석이 달린 척추의 성공적인 식별을 달성했습니다. 세분화의 평균 Dice는 각 데이터셋에서 각각 0.942 및 0.921에 도달하여 이전 최첨단 기술을 초과했습니다.



### Constant Acceleration Flow (https://arxiv.org/abs/2411.00322)
- **What's New**: 이번 논문에서는 Constant Acceleration Flow (CAF)라는 새로운 ODE 프레임워크를 제안합니다. CAF는 상수 가속도 모델을 도입하여 더 정교하고 정확한 ODE 플로우 예측이 가능합니다.

- **Technical Details**: CAF는 기존의 Rectified flow를 일반화하며, 가속도를 학습 가능한 추가 변수로 도입합니다. 또한, 초기 속도 조건화(initial velocity conditioning)와 재유동 과정(reflow process)을 통해 ODE 플로우의 정확도를 높입니다.

- **Performance Highlights**: CAF는 CIFAR-10 및 ImageNet 64×64 데이터셋에서 기존 최첨단 방법보다 우수한 FID 점수를 기록하였으며, 특히 몇 단계 만에 놀라운 성능을 보여주었습니다.



### Inducing Semi-Structured Sparsity by Masking for Efficient Model Inference in Convolutional Networks (https://arxiv.org/abs/2411.00288)
Comments:
          15 pages, 3 figures; this work will be presented at the NeurIPS 2024 Workshop on Fine-Tuning in Modern Machine Learning: Principles and Scalability (FITML)

- **What's New**: 본 논문에서는 컨볼루션 모델의 성능 저하 없이 2배 이상의 속도를 개선할 수 있는 새로운 방법을 제안합니다. 이 방법은 마스킹(masking)을 통해 반구조적(sparsity) 패턴을 학습하여, 기존 하드웨어 가속 하에서 효율적으로 사용될 수 있습니다.

- **Technical Details**: 제안된 방법은 컨볼루션 연산에서 반구조적(sparsity) 마스킹 패턴을 학습하여 모델의 가중치와 구조를 변경하지 않고 속도를 향상시킵니다. 특히, N:M sparsity 개념을 도입하여 특정 숫자의 요소를 제로로 설정함으로써 연산의 효율성을 증가시켰습니다. 이 과정에서 Gumbel-Max 트릭을 활용해 각 패턴을 선택하는 확률을 모델링합니다.

- **Performance Highlights**: 본 연구의 성과로는 기존 핸드헬드를 활용하여 제안된 반구조적 마스킹 기술이 CV 분류 작업에서 매우 약간의 성능 손실로 반구조적(sparsity) 패턴 학습을 가능하게 했다는 점입니다. 이는 대규모 모델이나 온라인 설정에서 특히 이점으로 작용합니다.



### TurtleBench: A Visual Programming Benchmark in Turtle Geometry (https://arxiv.org/abs/2411.00264)
- **What's New**: TurtleBench라는 새로운 벤치마크를 소개하며, 이는 LMMs의 기하학적 패턴 해석 능력을 평가하기 위해 설계되었습니다. TurtleBench는 인간과 AI의 직관적이고 시각적인 기하학적 이해의 차이를 드러내는 역할을 합니다.

- **Technical Details**: TurtleBench는 260개의 다양한 작업으로 구성되며, 주로 Scratch 작업과 Tweak 작업 두 가지 유형으로 나뉩니다. 각 작업은 이미지 입력과 Python 프로그래밍 코드 출력을 연결하며, Turtle 라이브러리를 사용하여 기하학적 모양을 생성하는 것입니다. 이 벤치마크는 시각적 패턴 인식과 프로그래밍 지식이 결합된 작업에서의 LMM 성능을 평가합니다.

- **Performance Highlights**: 최고의 LMM인 GPT-4o와 Gemini 1.5 Flash가 TurtleBench의 가장 간단한 작업에서 19
t 정확도로 실패했으며, 전체적인 성공률도 75
t 이상 미달했습니다. 이는 언어적 정보와 시각적 정보의 통합, 즉 시각적 패턴 인식에 대한 개선이 필요함을 시사합니다.



### Understanding Graphical Perception in Data Visualization through Zero-shot Prompting of Vision-Language Models (https://arxiv.org/abs/2411.00257)
- **What's New**: 이 논문은 비전 언어 모델(Vision Language Models, VLMs)의 그래픽 인지 능력을 평가하여, 이들이 인간과 유사한 차트 이해 능력을 가질 수 있는지를 탐구합니다. 연구의 주요 목표는 VLM의 성능 프로파일이 인간 행동과 어떻게 연관되는지를 밝혀내는 것입니다.

- **Technical Details**: VLM은 두 가지 입력 양식인 비전(vision)과 언어(language) 정보를 통합할 수 있는 모델로, 데이터 시각화 작업에서 차트 및 그래프의 이미지와 해당 텍스트 설명을 모두 고려할 수 있습니다. 인간의 시각화 이해 능력을 평가하는 기존 연구를 기반으로, VLM의 정확도를 평가하기 위해 zero-shot prompting 기법이 사용되었습니다. 실험은 3개의 주요 실험으로 나뉘어져 있으며, 각각의 실험에서는 45개의 독특한 시각화를 통한 7가지 차트 유형에 대해 테스트가 진행되었습니다.

- **Performance Highlights**: 결과적으로 VLM은 특정 작업 및 스타일 조합 하에 인간과 유사한 성능을 보였으며, 색상 및 차트 컨투이티와 같은 양식적 변동에 민감하다는 것이 밝혀졌습니다. 특히, 모형의 정확도는 색상을 명시하고 해석을 요구하는 경우에 가장 높은 성능을 나타냈으며, 두 요소를 제거할 경우 성능이 크게 감소했습니다.



### A Novel Breast Ultrasound Image Augmentation Method Using Advanced Neural Style Transfer: An Efficient and Explainable Approach (https://arxiv.org/abs/2411.00254)
- **What's New**: 이번 연구에서는 심층 학습을 활용하여 유방 초음파(BUS) 이미지의 효율적인 증강 방법을 개발하였습니다. 기존의 DL 모델의 한계를 극복하기 위해, Neural Style Transfer(NST) 및 Explainable AI(XAI)를 접목하여 고성능의 증강 모델을 구현하였습니다.

- **Technical Details**: 이 연구는 고급 NST 기법을 사용하여 새로운 스타일 손실 함수를 결합하고, 레이어별 중요도 전파(LRP) 방법을 통해 입력 이미지의 특징 중요성을 설명합니다. 데이터 병렬 분산 학습을 통해 8개의 GPU에 걸쳐 훈련을 분산시키고, Horovod 프레임워크를 사용하여 5.09배의 속도를 달성하였습니다.

- **Performance Highlights**: 제안된 모델은 800개의 BUS 이미지(348개의 양성 및 452개의 악성)에 대해 평가되었으며, ResNet50 모델을 활용하여 92.47%의 정확도를 기록하였습니다. 이는 증강 이전 이미지에 비해 37.26% 향상된 성능입니다.



### Understanding the Limits of Vision Language Models Through the Lens of the Binding Problem (https://arxiv.org/abs/2411.00238)
- **What's New**: 최근의 연구는 최첨단 비전 언어 모델(VLMs)의 성능이 매우 다양하다는 것을 보여줍니다. 이 모델들은 복잡하고 자연스러운 이미지를 설명하고 생성할 수 있지만, 카운팅, 위치 확인, 간단한 비주얼 유추와 같은 기본적인 다중 물체 논리 작업에서 놀라운 실패를 보입니다.

- **Technical Details**: 연구진은 인지 과학 및 신경 과학의 이론적 설명인 binding problem을 바탕으로 VLMs의 실패를 설명하고자 하였습니다. 이는 특정 객체의 특성을 결합하고 구별하는 뇌의 능력에 관한 질문으로, 인간의 시각 시스템이 이 문제를 해결하기 위해 직렬 처리에 의존한다는 점이 강조됩니다.

- **Performance Highlights**: 여러 비전 언어 모델의 성능을 평가한 결과, 이 모델들은 특정 조건(특히 객체가 많을 때)에서 반응 시간이 증가하고, 이는 인간이 수행하는 시각 검색 작업과 유사한 제약이 있음을 보여주었습니다. 특히, 입력 전처리 기법을 통해 VLM의 성능을 개선할 수 있었던 점이 주목할 만합니다.



### Protecting Feed-Forward Networks from Adversarial Attacks Using Predictive Coding (https://arxiv.org/abs/2411.00222)
- **What's New**: 이 연구에서는 adversarial attack에 대한 방어를 위해 predictive coding network(PCnet)를 보조 단계로 활용한 새로운 접근법을 제안합니다. 이 방법은 기존 모델을 변경하지 않고, 입력 이미지의 변화를 역전시키는 데 도움을 줍니다.

- **Technical Details**: PCnet은 feed-forward network에 매끄럽게 통합되어 adversarial perturbation에 대한 저항성을 크게 증가시킵니다. MNIST와 CIFAR10 데이터셋에서의 실험은 각각 약 82% 및 65%의 향상을 보여줍니다. 또한, PCnet은 작은 데이터셋의 서브셋에서 훈련되어 생성적 특성을 통해 perturbed 이미지를 원본에 가깝게 되돌리는 데 활용됩니다.

- **Performance Highlights**: 이 연구에서 PCnet의 효과는 MNIST의 경우 약 82% 그리고 CIFAR10의 경우 약 65%의 robustess 개선을 보여 주었습니다. 이러한 접근법은 인공 신경망 분류기의 보안성과 신뢰성을 높이는 가능성을 가지고 있습니다.



### Beyond Accuracy: Ensuring Correct Predictions With Correct Rationales (https://arxiv.org/abs/2411.00132)
Comments:
          In Proceedings of the 38th Conference on Neural Information Processing Systems (NeurIPS 2024)

- **What's New**: 이 논문은 대규모 (large) pretrained foundation 모델들이 인간 전문가를 초월하는 성과를 보임에도 불구하고, 그 예측의 근거가 제대로 검증되지 않는 문제를 다룹니다. 안전한 배포를 위해 정확한 예측과 올바른 근거의 이중 확인을 요구합니다.

- **Technical Details**: 제안된 두 단계 방식은 첫 번째 단계에서 시각 인식 (visual recognition) 작업을 위한 구조화된 근거 데이터셋을 큐레이션하고, 두 번째 단계에서는 수동 주석 (manual annotations) 없이 각 근거에 대한 시각적 증거를 분리하고 위치를 알아내기 위해 근거에 기반한 최적화 방법을 제안합니다.

- **Performance Highlights**: 광범위한 작업에서, 제안한 모델은 예측 정확도 (prediction accuracy)에서 기존 최첨단 모델보다 최대 10.1% 향상된 성능을 보였으며, 근거의 올바름 (rationale correctness) 또한 크게 개선하여 로컬라이제이션 (localization)에서 7.5%, 분리 (disentanglement)에서 36.5%의 향상을 기록했습니다.



### Preserving Pre-trained Representation Space: On Effectiveness of Prefix-tuning for Large Multi-modal Models (https://arxiv.org/abs/2411.00029)
Comments:
          Findings of EMNLP 2024

- **What's New**: 최근 대규모 다중 모달 모델(Large Multi-modal Models, LMMs)의 발전을 통해 기계와 세계 간의 상호작용 방식이 혁신적으로 변화하고 있습니다. 이러한 모델을 하위 작업에 적합하도록 조정하기 위해 파라미터 효율적인 미세 조정(Parameter-efficient fine-tuning, PEFT) 기술이 인기를 얻고 있으며, 본 논문에서는 PEFT의 작동 방식에 대한 심층 분석을 제공합니다.

- **Technical Details**: 이 연구에서는 두 단계로 구성된 PT-PEFT(Prefix-Tuned PEFT) 방법을 제안합니다. PT-PEFT는 먼저 prefix-tuning을 수행한 후, 이후에 PEFT 방법(예: Adapter, LoRA)을 통해 모델 파라미터를 조정합니다. 이를 통해 사전 훈련된 지식의 활용을 극대화합니다. 특히 본 연구에서는 singular value decomposition (SVD)을 사용하여 feature representation matrices의 변화를 분석했습니다.

- **Performance Highlights**: PT-PEFT는 이미지 캡셔닝(Image Captioning, IC) 및 시각적 질문 응답(Visual Question Answering, VQA) 작업에서 기존 PEFT 방법에 비해 성능을 개선하는 것으로 나타났습니다. 본 논문에서 관련된 네 가지 사전 훈련 모델을 대상으로 실험한 결과, PT-PEFT가 representation space를 보존하면서 전반적인 성능을 향상시키는 데 기여함을 확인하였습니다.



### Human Action Recognition (HAR) Using Skeleton-based Spatial Temporal Relative Transformer Network: ST-RTR (https://arxiv.org/abs/2410.23806)
- **What's New**: 본 연구는 기존 ST-GCN 방식의 한계를 극복하기 위해 Spatial-Temporal Relative Transformer (ST-RTR) 모델을 개발하여 장거리 인간 행동 인식을 가능하게 한다.

- **Technical Details**: ST-RTR 모델은 조인트(joint) 및 릴레이(relay) 노드를 포함하여 네트워크 내 효율적인 통신과 데이터 전송을 허용한다. 이를 통해 기존의 공간적(spatial) 및 시간적(temporal) 스켈레톤 구조를 파괴하고, 긴 거리의 인간 행동을 더 잘 이해할 수 있게 한다. 또한, ST-RTR 모델은 융합(fusion) 모델과 결합하여 성능을 추가로 향상시킨다.

- **Performance Highlights**: NTU RGB+D 60 데이터셋에서 CS와 CV 각각 2.11% 및 1.45% 향상되었고, NTU RGB+D 120에서 1.25%와 1.05% 향상되었다. UAV-Human 데이터셋에서는 정확도가 2.54% 개선되었다.



New uploads on arXiv(cs.AI)

### LogiCity: Advancing Neuro-Symbolic AI with Abstract Urban Simulation (https://arxiv.org/abs/2411.00773)
Comments:
          25 pages, 8 figures

- **What's New**: 최근 Neuro-Symbolic (NeSy) AI 시스템의 급속한 발전이 이루어졌습니다. LogiCity는 복잡한 다중 에이전트 상호작용을 위한 첫 번째 커스터마이즈 가능한 1차 논리(FOL) 기반 시뮬레이터로, 특히 도시 환경을 모델링합니다.

- **Technical Details**: LogiCity는 IsAmbulance(X), IsClose(X, Y)와 같은 의미론적 및 공간적 개념을 사용하여 다양한 도시 요소를 모델링합니다. 이러한 개념은 다양한 에이전트 행동을 제어하는 FOL 규칙을 정의하는 데 사용됩니다. 사용자가 구성할 수 있는 추상화를 지원하여, 논리적 추론을 위한 커스터마이즈 가능한 시뮬레이션 복잡성을 가능하게 합니다.

- **Performance Highlights**: LogiCity는 장기 순차적 의사결정(long-horizon sequential decision-making)과 1단계 시각적 추론(one-step visual reasoning) 두 가지 작업을 도입합니다. 이 연구는 NeSy 프레임워크의 추상적 추론에서의 장점을 보여줍니다. 그러나 장기 다중 에이전트 시나리오 또는 고차원 불균형 데이터 처리에서의 도전 과제가 남아있음을 강조합니다.



### WLPlan: Relational Features for Symbolic Planning (https://arxiv.org/abs/2411.00577)
- **What's New**: 본 논문에서는 WLPlan이라는 C++ 패키지를 소개합니다. 이 패키지는 자동으로 계획 작업의 관계적 특징을 생성하여 학습되는 도메인 제어 지식 또는 계획 작업을 탐색하고 이해하는 데 사용될 수 있습니다.

- **Technical Details**: WLPlan은 (1) 계획 작업을 그래프로 변환하는 기능과 (2) 그래프 커널(graph kernels)을 사용하여 이러한 그래프를 특징 벡터(feature vectors)로 임베딩하는 기능을 제공합니다. C++로 최적화된 성능을 제공하면서도 Python 바인딩을 통해 쉽게 프로토타입 및 모델을 훈련할 수 있도록 디자인되었습니다.

- **Performance Highlights**: WLPlan은 기존 방법들보다 뛰어난 평가 속도와 표현 능력을 가지고 있습니다. 연구자들은 WLPlan을 사용하여 계획 작업을 그래프로 변환하고, 해당 그래프를 특징 벡터로 변환하여 머신 러닝 파이프라인의 다른 측면에 집중할 수 있습니다.



### Human-inspired Perspectives: A Survey on AI Long-term Memory (https://arxiv.org/abs/2411.00489)
- **What's New**: 이 논문은 인공지능(AI)의 장기 기억(long-term memory) 기능을 체계적으로 조사하고 이러한 개념을 기반으로 한 이론적 프레임워크를 제안합니다. 이론의 기초로 인간의 기억(Mechanisms of human long-term memory)을 사용하여 AI 장기 기억 메커니즘과 관계를 설정하고, 'Self-Adaptive Long-term Memory (SALM)'이라는 새로운 인지 아키텍처(cognitive architecture)를 제안합니다.

- **Technical Details**: 이 논문은 인간의 장기 기억 시스템을 기반으로 AI의 장기 기억을 새로운 방법론으로 분류합니다. 핵심 구성 요소인 서사적 기억(episodic memory), 의미적 기억(semantic memory), 절차적 기억(procedural memory)의 개념을 바탕으로, AI 장기 기억을 비모수(non-parametric) 및 모수(parametric) 기억으로 구분합니다. SALM 아키텍처는 이러한 이론을 기반으로 하여 AI 시스템의 적응력을 향상시키는 메커니즘을 통합합니다.

- **Performance Highlights**: SALM 아키텍처는 현재의 인지 아키텍처의 한계를 극복하고 인간 장기 기억 처리 메커니즘의 적응성을 초월할 수 있는 잠재력을 가지고 있습니다. 또한 AI의 장기 기억 모듈을 위한 측정 방법과 적용 가능성을 제시하여 향후 AI 시스템의 발전을 이끌 가능성이 큽니다.



### Integrating Fuzzy Logic into Deep Symbolic Regression (https://arxiv.org/abs/2411.00431)
Comments:
          10 pages, 1 figure, published for XAI FIN 24 this https URL

- **What's New**: 이 연구는 Deep Symbolic Regression (DSR) 모델에 퍼지 로직(fuzzy logic)을 통합하여 신용 카드 사기 탐지에서 성능과 설명 가능성을 향상하는 방법을 제안합니다.

- **Technical Details**: 퍼지 로직은 불확실성과 모호성을 처리하는 데 뛰어난 기법으로, 사기 탐지 데이터 세트의 복잡성과 불확실성을 처리하는 데 적합합니다. 특히, 다양한 퍼지 로직의 함의(implication) 방식인 Łukasiewicz, Gödel, Product를 활용하여 그 효과를 평가합니다.

- **Performance Highlights**: 연구 결과 Łukasiewicz 함의가 가장 높은 F1 점수와 전체 정확도를 달성하는 반면, Product 함의는 성능과 설명 가능성 간의 균형을 제공합니다. 이 접근 방식은 정보 손실로 인해 현재 최첨단(SOTA) 모델보다 낮은 성능을 보이지만, 퍼지 로직과 DSR의 통합을 통한 새로운 통찰력을 제공합니다.



### GPT for Games: An Updated Scoping Review (2020-2024) (https://arxiv.org/abs/2411.00308)
Comments:
          Submitted to IEEE Transactions on Games

- **What's New**: 이 논문은 게임에서의 GPT 활용 가능성을 탐구한 131개의 문헌을 검토한 업데이트된 스코핑 리뷰를 소개합니다. 특히 2024년에 발표된 76개 새로운 논문을 포함하여 게임 연구에서 GPT의 다섯 가지 주요 응용 분야를 확인하였습니다.

- **Technical Details**: 사용된 데이터베이스는 ACM Digital Library, IEEE Xplore, Springer, AAAI로, "game"와 "GPT"를 키워드로 하여 연구 논문을 검색했습니다. 최종적으로 131개의 논문이 포함되었으며, 각 논문은 절차적 콘텐츠 생성 (PCG), 혼합 주도 게임 디자인 (MIGDD), 혼합 주도 게임 플레이 (MIG), 게임 플레이 (PG), 게임 사용자 연구 (GUR)로 분류되었습니다.

- **Performance Highlights**: 2024년에는 76개의 관련 연구가 발표되었으며, 이는 2023년의 39개 연구에 비해 큰 증가를 보여줍니다. 새로운 GPT 모델(특히 GPT-4 및 GPT-3.5)의 사용이 두드러졌고, 연구자들은 게임에서 최신 GPT 모델의 응용 가능성을 더욱 탐구할 것으로 예상됩니다.



### TurtleBench: A Visual Programming Benchmark in Turtle Geometry (https://arxiv.org/abs/2411.00264)
- **What's New**: TurtleBench라는 새로운 벤치마크를 소개하며, 이는 LMMs의 기하학적 패턴 해석 능력을 평가하기 위해 설계되었습니다. TurtleBench는 인간과 AI의 직관적이고 시각적인 기하학적 이해의 차이를 드러내는 역할을 합니다.

- **Technical Details**: TurtleBench는 260개의 다양한 작업으로 구성되며, 주로 Scratch 작업과 Tweak 작업 두 가지 유형으로 나뉩니다. 각 작업은 이미지 입력과 Python 프로그래밍 코드 출력을 연결하며, Turtle 라이브러리를 사용하여 기하학적 모양을 생성하는 것입니다. 이 벤치마크는 시각적 패턴 인식과 프로그래밍 지식이 결합된 작업에서의 LMM 성능을 평가합니다.

- **Performance Highlights**: 최고의 LMM인 GPT-4o와 Gemini 1.5 Flash가 TurtleBench의 가장 간단한 작업에서 19
t 정확도로 실패했으며, 전체적인 성공률도 75
t 이상 미달했습니다. 이는 언어적 정보와 시각적 정보의 통합, 즉 시각적 패턴 인식에 대한 개선이 필요함을 시사합니다.



### Understanding Graphical Perception in Data Visualization through Zero-shot Prompting of Vision-Language Models (https://arxiv.org/abs/2411.00257)
- **What's New**: 이 논문은 비전 언어 모델(Vision Language Models, VLMs)의 그래픽 인지 능력을 평가하여, 이들이 인간과 유사한 차트 이해 능력을 가질 수 있는지를 탐구합니다. 연구의 주요 목표는 VLM의 성능 프로파일이 인간 행동과 어떻게 연관되는지를 밝혀내는 것입니다.

- **Technical Details**: VLM은 두 가지 입력 양식인 비전(vision)과 언어(language) 정보를 통합할 수 있는 모델로, 데이터 시각화 작업에서 차트 및 그래프의 이미지와 해당 텍스트 설명을 모두 고려할 수 있습니다. 인간의 시각화 이해 능력을 평가하는 기존 연구를 기반으로, VLM의 정확도를 평가하기 위해 zero-shot prompting 기법이 사용되었습니다. 실험은 3개의 주요 실험으로 나뉘어져 있으며, 각각의 실험에서는 45개의 독특한 시각화를 통한 7가지 차트 유형에 대해 테스트가 진행되었습니다.

- **Performance Highlights**: 결과적으로 VLM은 특정 작업 및 스타일 조합 하에 인간과 유사한 성능을 보였으며, 색상 및 차트 컨투이티와 같은 양식적 변동에 민감하다는 것이 밝혀졌습니다. 특히, 모형의 정확도는 색상을 명시하고 해석을 요구하는 경우에 가장 높은 성능을 나타냈으며, 두 요소를 제거할 경우 성능이 크게 감소했습니다.



### Understanding the Limits of Vision Language Models Through the Lens of the Binding Problem (https://arxiv.org/abs/2411.00238)
- **What's New**: 최근의 연구는 최첨단 비전 언어 모델(VLMs)의 성능이 매우 다양하다는 것을 보여줍니다. 이 모델들은 복잡하고 자연스러운 이미지를 설명하고 생성할 수 있지만, 카운팅, 위치 확인, 간단한 비주얼 유추와 같은 기본적인 다중 물체 논리 작업에서 놀라운 실패를 보입니다.

- **Technical Details**: 연구진은 인지 과학 및 신경 과학의 이론적 설명인 binding problem을 바탕으로 VLMs의 실패를 설명하고자 하였습니다. 이는 특정 객체의 특성을 결합하고 구별하는 뇌의 능력에 관한 질문으로, 인간의 시각 시스템이 이 문제를 해결하기 위해 직렬 처리에 의존한다는 점이 강조됩니다.

- **Performance Highlights**: 여러 비전 언어 모델의 성능을 평가한 결과, 이 모델들은 특정 조건(특히 객체가 많을 때)에서 반응 시간이 증가하고, 이는 인간이 수행하는 시각 검색 작업과 유사한 제약이 있음을 보여주었습니다. 특히, 입력 전처리 기법을 통해 VLM의 성능을 개선할 수 있었던 점이 주목할 만합니다.



### Building Multi-Agent Copilot towards Autonomous Agricultural Data Management and Analysis (https://arxiv.org/abs/2411.00188)
- **What's New**: 이 논문에서는 전통적인 농업 데이터 관리 방식의 한계를 극복하기 위해 대형 언어 모델(LLM)을 기반으로 한 자율 농업 데이터 관리와 분석을 위한 ADMA(농업 데이터 관리 및 분석) 항공 파일럿 시스템을 제안합니다.

- **Technical Details**: ADMA Copilot은 사용자 의도를 이해하고 데이터 처리 파이프라인을 계획하여 자동으로 작업을 수행하는 다중 에이전트 시스템입니다. 이 시스템은 LLM 기반의 컨트롤러, 입력 포맷터 및 출력 포맷터의 세 에이전트가 협력하여 동작하며, 메타 프로그램 그래프를 정의하여 제어 흐름과 데이터 흐름을 분리하여 예측 가능성을 높였습니다.

- **Performance Highlights**: 시스템의 실험 결과, ADMA Copilot은 지능적이고 자율적이며 효율적이고 확장 가능하며 유연하고 개인정보 보호를 강화한 시스템으로 평가되었습니다. 기존 시스템들과 비교하여 우수성과 잠재력을 강조하였으며, 농업 데이터 관리의 현대적 도전 과제를 해결하는 데 기여할 것으로 기대됩니다.



### Unlocking the Potential of Global Human Expertis (https://arxiv.org/abs/2411.00156)
Comments:
          NeurIPS 2024; Main Paper 15 pages, Appendix 11 pages

- **What's New**: 이 논문은 다양한 국제 전문가의 아이디어와 방법을 수집하고 처리하여 글로벌 사회적 문제를 해결하는 데 AI(Artificial Intelligence)가 중요한 역할을 할 수 있다고 주장합니다. 특히 RHEA(Realizing Human Expertise through AI)라는 진화적 AI 프레임워크를 통해 인간 전문가의 지식을 정제하고 재조합하여 보다 효과적인 솔루션을 발견하는 과정을 제시합니다.

- **Technical Details**: RHEA 프레임워크는 다음의 단계를 따릅니다: Define(문제 정의), Gather(해결책 수집), Distill(내부 구조 정제), Evolve(진화). 이 과정은 인간 팀의 아이디어 발달 방식과 유사한 생물학적으로 영감을 받은 방법으로, 다양한 전문가로부터 수집된 해결책을 비교하고 결합하는 데 필수적입니다.

- **Performance Highlights**: RHEA는 XPRIZE Pandemic Response Challenge의 결과물에 적용되어, 100팀 이상의 전문가들이 제출한 169개의 정책 제안 모델을 재조합하고 정제하여 AI나 인간 전문가 개별적으로 제안한 것보다 더 넓고 효과적인 정책 세트를 발견했습니다. 이 결과는 AI가 인간의 전문성을 활용하여 글로벌 문제 해결의 잠재력을 실현하는 데 중요한 역할을 할 수 있음을示합니다.



### Responsibility-aware Strategic Reasoning in Probabilistic Multi-Agent Systems (https://arxiv.org/abs/2411.00146)
- **What's New**: 이 논문은 책임을 인식하는 에이전트가 있는 확률적 다중 에이전트 시스템에서 전략적 추론 문제를 다룬다. PATL+R이라는 새로운 논리를 도입하여 인과적 책임을 고려한 다중 에이전트 전략적 추론을 위한 프레임워크를 제공한다.

- **Technical Details**: PATL+R은 인과적 책임을 모달리티로 포함하며, 이를 통해 에이전트 간의 균형 잡힌 책임과 보상 배분을 최적화하는 공동 전략을 합성하는 방법을 제안한다. 논문에서는 Nash equilibrium(NE)을 전략적 추론의 해결 개념으로 활용하며, 확률적 다중 플레이어 게임의 매개변수 모델 체크를 통해 책임-aware NE 전략을 계산하는 방법을 보여준다.

- **Performance Highlights**: 제안된 모델은 PATL+R 공식을 모형 점검을 통해 풀이할 수 있으며, 각 에이전트의 보상과 책임 간의 균형을 고려한 NE 전략을 PSPACE 내에서 계산 가능함을 보여준다.



### Project Sid: Many-agent simulations toward AI civilization (https://arxiv.org/abs/2411.00114)
Comments:
          35 pages, 14 figures

- **What's New**: 이 논문은 10-1000 이상의 AI 에이전트가 상호작용하는 대규모 시뮬레이션을 통해 에이전트 사회 내에서의 행동과 발전을 탐구합니다. 새로운 PIANO (Parallel Information Aggregation via Neural Orchestration) 아키텍처를 소개하여 에이전트들이 인간 및 서로와 실시간으로 상호작용하면서 여러 출력 스트림 간의 일관성을 유지할 수 있도록 합니다.

- **Technical Details**: PIANO 아키텍처는 인간과 유사한 AI 에이전트를 위해 설계된 두 가지 뇌 영감을 받은 설계 원칙을 기반으로 합니다. 이 아키텍처는 다양한 모듈을 병렬로 실행하여 에이전트가 환경과 실시간으로 상호작용할 수 있도록 합니다. 또한, 에이전트들이 느린 사고 및 행동을 동시에 수행할 수 있는 기능을 갖추게 됩니다.

- **Performance Highlights**: Minecraft 환경 내에서 실시된 시뮬레이션 결과, 에이전트는 전문화된 역할을 자율적으로 개발하고, 공동 규칙을 준수 및 변경하며, 문화 및 종교적 전파에 참여할 수 있음을 보였습니다. 이러한 결과는 AI 문명으로 향한 중대한 이정표를 달성할 수 있는 가능성을 보여줍니다.



### Applying Data Driven Decision Making to rank Vocational and Educational Training Programs with TOPSIS (https://arxiv.org/abs/2411.00017)
Comments:
          18 pages, 7 figures

- **What's New**: 본 논문은 스페인 엑스트레마두라(Extremadura) 지역의 직업교육 프로그램에 대한 다기준(classification based on multiple criteria) 평가를 제시하며, 2009-2016 년도 동안 취업에 미치는 영향을 분석합니다.

- **Technical Details**: 연구에서는 TOPSIS (Technique for Order Preference by Similarity to Ideal Solution) 방법론과 새로운 의사결정 지원 방식이 결합되어 각 기준의 영향을 분석하며, 이 방법은 최악-최선 시나리오 분석을 기반으로 하고 있습니다. 이러한 분석은 Pearson 상관비율을 기반으로 한 기존의 글로벌 민감도 분석 기법과 비교됩니다.

- **Performance Highlights**: 28000건 이상의 VET 학생 기록을 활용한 데이터 기반의 다기준 의사결정 방법론을 통해, 엑스트레마두라의 VET 프로그램 성과 평가 및 노동시장 내 취업 효과성을 확인하였습니다. 이러한 접근은 VET 프로그램이 노동시장에 미치는 영향을 이전 연구들보다 상세하게 조사한 점에서 중요한 기여를 하고 있습니다.



### IC/DC: Surpassing Heuristic Solvers in Combinatorial Optimization with Diffusion Models (https://arxiv.org/abs/2411.00003)
- **What's New**: 최근 학습 기반 조합 최적화(Combinatorial Optimization, CO) 방법의 발전은 NP-hard 문제를 해결하는 데 있어 전문가가 설계한 휴리스틱(heuristics) 없이도 유망한 결과를 보여주고 있습니다. 이 논문에서는 감독(supervision) 없이 작동하는 CO 프레임워크인 IC/DC를 소개합니다.

- **Technical Details**: IC/DC는 두 개의 서로 다른 아이템 집합을 다루는 문제에 특화되어 있으며, 유효한 솔루션을 생성하기 위해 문제 특정 탐색 과정이 필요하지 않습니다. 이 프레임워크는 아이템 간의 복잡한 관계를 포착할 수 있는 새로운 아키텍처(architecture)를 사용합니다. 우리는 모델을 self-supervised 방식으로 훈련하여 솔루션의 비용을 최소화하면서 문제 특정 제약을 준수합니다.

- **Performance Highlights**: IC/DC는 기존의 학습 방법들과 비교하여 최첨단(state-of-the-art) 성능을 달성하였으며, 비대칭 여행 판매원 문제(Asymmetric Traveling Salesman Problem, ATSP)에서도 잘 알려진 솔버(solvers) 및 휴리스틱 접근 방식을 초월하였습니다.



### Freeze-Omni: A Smart and Low Latency Speech-to-speech Dialogue Model with Frozen LLM (https://arxiv.org/abs/2411.00774)
Comments:
          Project Page: this https URL

- **What's New**: 최근 대화형 인공지능 모델인 Freeze-Omni가 제안되었습니다. 이 모델은 LLM(대규모 언어 모델)의 파라미터를 고정(frozen)한 상태에서 음성 입력과 출력을 연결하는 멀티모달 아키텍처를 갖추고 있습니다. 이를 통해 음성 대화 능력을 획득하면서도 기존 LLM의 지능을 유지합니다.

- **Technical Details**: Freeze-Omni는 음성 인코더와 디코더로 구성되어 있습니다. 모델의 훈련 과정에서 첫 단계는 ASR(data for automatic speech recognition) 데이터를 사용해 음성을 텍스트로 변환하는 것입니다. 이후, 텍스트-음성이 결합된 데이터를 활용해 출력 음성을 생성합니다. 이 모델은 음성-음성 대화 구조를 통해 사용자의 입력을 처리하고 유연한 대화가 가능하도록 설계되어 있습니다.

- **Performance Highlights**: Freeze-Omni는 훈련에 필요한 데이터 양이 적고, 계산 리소스를 절약합니다. 본 모델은 8개의 GPU에서 60,000개의 다중 회전 Q&A 데이터만으로 효과적인 음성 대화 기능을 달성했습니다. 낮은 지연(latency) 시간을 유지하며, 텍스트 모드에서의 지능도를 손상시키지 않고 음성 모드에서도 유사한 성능을 발휘합니다.



### GameGen-X: Interactive Open-world Game Video Generation (https://arxiv.org/abs/2411.00769)
Comments:
          Project Page: this https URL

- **What's New**: GameGen-X는 오픈 월드 게임 비디오를 생성하고 상호작용적으로 제어할 수 있도록 설계된 최초의 diffusion transformer 모델입니다. 이 모델은 혁신적인 캐릭터, 동적 환경, 복잡한 행동, 다양한 이벤트 등을 포함한 고품질의 콘텐츠 생성을 지원합니다.

- **Technical Details**: GameGen-X는 Open-World Video Game Dataset (OGameData)을 기반으로 훈련되었습니다. 이 데이터셋은 150개 이상의 게임에서 수집된 100만 개의 다양한 플레이 비디오 클립으로 구성되며, 두 단계 훈련 과정(foundation model pre-training 및 instruction tuning)을 거칩니다. InstructNet을 통해 게임과 관련된 다중 모달 제어 신호 전문가를 통합하고, 사용자 입력에 따라 잠재 표현을 조정할 수 있게 합니다.

- **Performance Highlights**: 실험 결과 GameGen-X는 다양한 게임 내용을 고품질로 생성하고, 사용자 입력에 따라 동적으로 반응하는 비디오 클립을 생성할 수 있는 뛰어난 성능을 보여주었습니다. 이 모델은 전통적인 게임 디자인 방식에 대해 확장 가능하고 효율적인 보조 도구로서의 잠재력을 가지고 있습니다.



### Mitigating Tail Narrowing in LLM Self-Improvement via Socratic-Guided Sampling (https://arxiv.org/abs/2411.00750)
Comments:
          Codes are publicly available at this https URL

- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 자체 개선(self-improvement) 방법의 성능 한계를 극복하기 위해 'Guided Self-Improvement (GSI)'라는 새로운 전략을 도입합니다. GSI는 Socratic-style guidance signals를 활용하여 복잡한 쿼리에 대한 모델의 추론을 돕고, 고차원 데이터 샘플링의 효율성을 향상시키는 것을 목표로 합니다.

- **Technical Details**: GSI는 샘플링 단계 후 복잡한 쿼리 샘플링을 위한 추가 재샘플링 단계인 distribution re-balancing을 도입하여 샘플링 공간을 축소하고 모델의 허위 추론(hallucinations)을 줄입니다. 이를 통해 GSI는 도전적인 쿼리에 대한 솔루션 범위를 확장하고 성능 향상을 도모합니다.

- **Performance Highlights**: GSI는 네 개의 모델 및 여섯 개의 수학적 추론 작업에 대한 실험에서 성능 병목 현상을 완화하고, 계산 효율성을 유지하면서 균형 잡힌 솔루션 분포와 개선된 모델 일반화 성능을 보여줍니다.



### Decoding Dark Matter: Specialized Sparse Autoencoders for Interpreting Rare Concepts in Foundation Models (https://arxiv.org/abs/2411.00743)
- **What's New**: 본 논문에서는 Specialized Sparse Autoencoders (SSAEs)를 제안하여, Foundation Models (FMs)에서 특정 하위 도메인에 관련된 희귀한 특징들을 효율적으로 추출하는 새로운 방법을 소개합니다. 이러한 접근 방식은 기존의 Sparse Autoencoders (SAEs)보다 더 많은 tail concepts를 캡처하는데 집중합니다.

- **Technical Details**: SSAEs는 unsupervised targeted 방법으로, 특정 하위 도메인에 맞춰 훈련된 스파스 오토인코더입니다. SSAEs 훈련 시 Dense retrieval 기법을 활용하여 관련 훈련 데이터를 선택하고, Tilted Empirical Risk Minimization (TERM)를 학습 목표로 설정하여 tail 개념의 표현을 개선합니다.

- **Performance Highlights**: SSAEs는 Bias in Bios 데이터셋에서 12.5%의 worst-group classification accuracy 향상을 보이며, 전반적인 성능에서도 일반적인 SAEs를 초월한 결과를 나타냅니다. 이는 SSAEs가 하위 도메인에서의 FMs의 내부 작용을 보다 깊이 있게 탐구할 수 있는 강력한 도구가 됨을 의미합니다.



### MolCap-Arena: A Comprehensive Captioning Benchmark on Language-Enhanced Molecular Property Prediction (https://arxiv.org/abs/2411.00737)
- **What's New**: 이 논문은 생물 분자 모델링(biomolecular modeling)과 자연어 정보(natural language information)를 연결하는 최신 연구의 일환으로, 대규모 언어 모델(large language models, LLMs)을 사용하는 방법을 제시합니다.

- **Technical Details**: 이 연구에서는 Molecule Caption Arena라는 새로운 벤치마크를 도입하여 LLM으로 증강된 분자 속성 예측(molecular property prediction)을 평가합니다. 이를 위해 일반 용도 및 특정 도메인에 맞춘 분자 설명 생성기(molecule captioners) 등 20개 이상의 LLM을 다양한 예측 작업을 통해 비교했습니다. 논문은 또한 새로운 전투 기반 평가 시스템(battle-based rating system)을 소개합니다.

- **Performance Highlights**: LLM에서 추출한 지식이 최신의 분자 표현(molecular representations)을 향상시키는 능력을 확인했으며, 모델, 프롬프트(prompt), 데이터셋에 따라 다양한 성과 차이를 보였습니다.



### Multi-Agent Deep Q-Network with Layer-based Communication Channel for Autonomous Internal Logistics Vehicle Scheduling in Smart Manufacturing (https://arxiv.org/abs/2411.00728)
Comments:
          Accepted for the 5th IFAC/INSTICC INTERNATIONAL CONFERENCE ON INNOVATIVE INTELLIGENT INDUSTRIAL PRODUCTION AND LOGISTICS

- **What's New**: 이 논문은 스마트 제조에서 내부 물류 차량의 스케줄링을 다루며, 다중 에이전트 딥 Q-네트워크(MADQN)와 계층 기반 커뮤니케이션 채널(LBCC)을 활용하여 작업 지연 시간, 지연 작업 수, 차량 에너지 소비를 최소화하는 혁신적인 방법을 제안합니다.

- **Technical Details**: 제안된 방법론은 다중 에이전트 시스템을 이용하여 개별 작업을 에이전트로 모델링하고, 이들이 딥 Q-네트워크 알고리즘 내에서 보상 정책을 최적화하는 방식을 취합니다. 이 시스템은 동적 제조 환경에서 작업 도착 및 작업장 고장과 같은 복잡한 시나리오에 잘 적응합니다.

- **Performance Highlights**: 시뮬레이션을 통해 이 방법이 제안된 9가지 유명한 스케줄링 휴리스틱과 비교할 때, 직관적인 스케줄링 전략보다 더 우수한 성능을 발휘한다는 것을 입증했습니다. 이를 통해 MADQN과 LBCC의 유연성과 내구성이 강조되었습니다.



### SPRING Lab IITM's submission to Low Resource Indic Language Translation Shared Task (https://arxiv.org/abs/2411.00727)
Comments:
          To be published in WMT 2024. Low-Resource Indic Language Translation Shared Task

- **What's New**: 본 연구에서는 Khasi, Mizo, Manipuri, Assamese 등 네 가지 저자원 언어에 대한 강력한 번역 모델을 개발하였으며, 데이터 수집 및 전처리에서 교육 및 평가에 이르는 포괄적인 파이프라인을 포함하고 있습니다.

- **Technical Details**: 우리는 WMT, BPCC, PMIndia, OpenLanguageData에서 데이터를 활용하여 모델을 훈련시켰으며, 단일 언어 데이터셋에 대한 역번역(back-translation) 기법을 통해 Mizo와 Khasi의 이중 언어 데이터를 획기적으로 확장하였습니다. 또한, NLLB 3.3B 모델을 세 가지 언어에 대해 파인튜닝(fine-tuning) 하여 성능을 개선했습니다.

- **Performance Highlights**: 모델 평가 결과, Assamese의 경우 BLEU 점수가 27.26, Khasi는 NLLB 모델 지원 부족에도 불구하고 특수 토큰을 도입하여 훈련된 결과가 긍정적이었습니다. 전체적으로 Mizo와 Manipuri 번역 방향의 두 점수는 역번역 데이터 품질 저하로 인해 낮았습니다.



### Cross-Fundus Transformer for Multi-modal Diabetic Retinopathy Grading with Catarac (https://arxiv.org/abs/2411.00726)
Comments:
          10 pages, 4 figures

- **What's New**: 본 연구는 색깔 망막 사진(CFP)과 적외선 망막 사진(IFP) 정보를 융합하여 더욱 정확한 당뇨병성 망막병증(DR) 등급을 위한 다중 모달 딥러닝 프레임워크, Cross-Fundus Transformer (CFT)를 제안합니다.

- **Technical Details**: CFT는 ViT 기반의 이중 스트림 아키텍처로 CFP와 IFP 이미지를 융합하며, Cross-Fundus Attention (CFA) 모듈을 도입해 두 이미지 간의 대응 관계를 포착합니다. 자동으로 CFA 모듈을 통해 두 모달리티의 정보를 융합하여 DR을 진단합니다.

- **Performance Highlights**: 1,713쌍의 다중 모달 fundus 이미지로 구성된 임상 데이터셋을 기반으로 한 실험에서, 제안된 방법은 기존의 방법들보다 우월한 성능을 보여줍니다. 본 연구는 CFP와 IFP 이미지를 동시에 이용하여 DR을 자동으로 진단한 최초의 시도로, 최첨단 성능을 입증합니다.



### B-cosification: Transforming Deep Neural Networks to be Inherently Interpretab (https://arxiv.org/abs/2411.00715)
Comments:
          31 pages, 9 figures, 12 tables, Neural Information Processing Systems (NeurIPS) 2024

- **What's New**: B-cos Networks의 새로운 접근인 'B-cosification'을 통해 기존의 사전 학습된 모델을 효율적으로 해석 가능하게 만드는 방법을 제안합니다. 이 방법은 CNN과 ViT와 같은 모델을 작은 변화로 변환하여 보다 우수한 해석성을 제공하며, 훈련 비용을 대폭 절감합니다.

- **Technical Details**: B-cosification은 기존의 DNN을 아키텍처적 변경 없이도 해석 가능하도록 조정하는 기술입니다. 이 과정에서 модели가 계산하는 방식을 인간이 이해할 수 있는 형태로 줄이는 데 초점을 맞춥니다. B-cos 변환의 '정렬 압력(alignment pressure)'을 조정하여 모델의 해석성을 개선하며, 기존의 사전 훈련된 CLIP 모델도 이 방식을 통해 B-cosified CLIP으로 변환됩니다.

- **Performance Highlights**: B-cosified 모델은 기존 DNN 대비 비슷한 해석성을 유지하면서도 분류 성능에서 우수한 결과를 나타내고, 특히 훈련 비용은 훨씬 낮습니다. 다양한 데이터 세트에서 제로 샷(zero-shot) 성능 테스트를 통해 B-cosified CLIP 모델이 경쟁력을 보입니다.



### Learning in Markov Games with Adaptive Adversaries: Policy Regret, Fundamental Barriers, and Efficient Algorithms (https://arxiv.org/abs/2411.00707)
Comments:
          NeurIPS'24

- **What's New**: 이 논문은 학습자가 적응 가능한 상대와의 동적인 환경에서 배우는 방법을 탐구하며, 기존의 Markov 게임에서 외부 후회를 대체할 수 있는 새로운 학습 목표인 '정책 후회'에 중점을 둡니다.

- **Technical Details**: Markov 게임(Markov Game) 모델을 사용하여 학습자가 적응 가능한 상대와 경쟁하는 방법을 연구합니다. 적 대비해 정책 후회(policy regret)를 최소화하는 알고리즘을 제안하며, 일관된 적대자(consistent adversaries)에 대해 $	ext{O}(	ext{T}^{1/2})$ 수준의 성과를 도출합니다.

- **Performance Highlights**: 제안된 알고리즘(OPO-OMLE, APE-OVE)은 메모리 제한이 있는 일관된 적대자에 대해 정책 후회 수치를 효과적으로 감소시키며, 정책 집합이 비선형적으로 클 때도 효과적임을 보입니다.



### Algorithmic Transparency in Forecasting Support Systems (https://arxiv.org/abs/2411.00699)
- **What's New**: 이 논문은 Forecasting Support Systems (FSS)의 사용자 인터페이스가 통계 알고리즘과 사용자 간의 연결을 어떻게 개선할 수 있는지를 다루고 있습니다. 특히, 알고리즘의 투명성이 유익한 조정을 촉진하고 유해한 조정을 방지하는 중요한 요소임을 제안합니다.

- **Technical Details**: 세 가지 FSS 디자인을 구현하여 투명성의 정도에 따른 효과를 연구하였으며, 투명성은 시간 시계열 분해를 기반으로 개인 조정 가능성이 높아지고 조정의 품질에 미치는 영향을 분석했습니다. 조정의 질, 조정량, 사용자 만족도에 대한 실증 연구가 수행되었습니다.

- **Performance Highlights**: 투명성이 유해한 조정의 분산과 양을 줄이는 데 도움이 됨을 밝혀냈지만, 조정이 이루어질 수 있는 투명한 구성 요소를 사용자에게 직접 조정하도록 허용하면 결과적으로 해로운 조정으로 이어질 수 있음을 발견했습니다. 또한, 적절한 교육 없이 알고리즘 투명성의 과도한 제공이 사용자에게 혼란을 초래할 위험이 있음을 나타냅니다.



### CTPD: Cross-Modal Temporal Pattern Discovery for Enhanced Multimodal Electronic Health Records Analysis (https://arxiv.org/abs/2411.00696)
Comments:
          Technical report

- **What's New**: 이 논문에서는 여러 환자의 임상 예측을 향상시키기 위한 새로운 Cross-Modal Temporal Pattern Discovery (CTPD) 프레임워크를 도입합니다.

- **Technical Details**: CTPD는 여러 종류의 전자 건강 기록 (EHR) 데이터에서 의미 있는 교차 모달 (cross-modal) 시간 패턴을 효율적으로 추출합니다. 이 프레임워크는 슬롯 주의 (slot attention)를 활용하여 시간적 의미 임베딩을 생성하고, 교차 모달 정렬을 위한 대조 기반 TPNCE 손실을 도입합니다. 또한, 변환기 기반 융합 메커니즘을 통해 학습된 패턴의 품질을 높입니다.

- **Performance Highlights**: MIMIC-III 데이터베이스를 사용하여 48시간 입원 사망률 예측 및 24시간 표현형 분류와 같은 두 가지 임상 작업에서 기존 방법에 비해 월등한 성능을 보였습니다.



### Latent Paraphrasing: Perturbation on Layers Improves Knowledge Injection in Language Models (https://arxiv.org/abs/2411.00686)
Comments:
          NeurIPS 2024

- **What's New**: 이 논문에서는 LaPael이라는 새로운 잠재 레벨( latent-level) 패러프레이징 방법을 소개하여 모델의 초기 레이어에 입력 의존성 노이즈(input-dependent noise)를 적용함으로써 지식 주입(knowledge injection)을 혁신적으로 개선했습니다.

- **Technical Details**: LaPael은 LLMs에서 패러프레이징(paraphrasing) 데이터를 활용해 지식을 주입하는 기존 접근법의 두 가지 주요 문제인 높은 계산 비용과 제한된 샘플 다양성을 해결합니다. 이 방법은 모델 내부에서 직접적으로 다양한 의미적으로 일관된 증강(augments)을 생성할 수 있게 합니다.

- **Performance Highlights**: 질문-응답(Question-Answering) 벤치마크에서의 광범위한 실험 결과, LaPael은 기존의 표준 파인튜닝(standard fine-tuning) 및 노이즈 기반 접근법보다 지식 주입의 성능을 개선하며, 데이터 수준의 패러프레이징과 결합할 경우 성능을 더욱 강화하는 것으로 나타났습니다.



### TaxaBind: A Unified Embedding Space for Ecological Applications (https://arxiv.org/abs/2411.00683)
Comments:
          Accepted to WACV 2025

- **What's New**: TaxaBind는 모든 관심 있는 종을 특성화하기 위한 통합 임베딩 공간을 제시합니다. 이는 종의 지상 이미지, 지리적 위치, 위성 이미지, 텍스트, 오디오, 환경적 특징을 포함한 6가지 모달리티를 활용하는 다중모달(Multimodal) 임베딩 공간입니다.

- **Technical Details**: TaxaBind는 다양한 모달리티의 지식을 효과적으로 증류하기 위한 기술로서 multimodal patching을 제안합니다. 이는 각 모달리티의 고유한 정보 손실을 방지하며, 다양한 생태적 작업을 위한 학습을 가능하게 합니다. 또한, iSatNat과 iSoundNat이라는 두 가지 대형 데이터셋을 구축하고, TaxaBench-8k라는 평가를 위한 진정한 다중모달 데이터셋을 소개합니다.

- **Performance Highlights**: TaxaBind는 다음과 같은 작업에서 강력한 제로샷(zero-shot) 및 돌발(emergent) 능력을 보였습니다: 종 분류(Species Classification), 교차 모델 검색(Cross-model Retrieval), 오디오 분류(Audio Classification). 추가적으로, 여러 벤치마크 및 제로샷 작업에서 모델의 효과성과 성장 가능성을 입증하였습니다.



### AI-based traffic analysis in digital twin networks (https://arxiv.org/abs/2411.00681)
Comments:
          Chapter 4: Digital Twins for 6G: Fundamental theory, technology and applications; pp. 83-132

- **What's New**: 이번 논문에서는 Digital Twin Networks (DTNs)이 물리적 네트워크 최적화 방식을 혁신하고, AI 기반의 교통 분석이 수행되는 과정에 대해 설명합니다. DTNs는 다양한 물리적 네트워크에 대한 가상 표현을 제공하고, 네트워크 성능 향상, 지연 최적화, 에너지 효율성 등의 과제를 해결하기 위해 AI 도구들을 활용합니다.

- **Technical Details**: DTNs는 3개의 레이어(물리적 레이어, 가상 레이어, 서비스 또는 결정 레이어)로 구성됩니다. 물리적 레이어는 데이터 수집 및 전처리를 담당하고, 가상 레이어는 AI 기반의 Machine Learning (ML) 및 Deep Learning (DL) 기능을 통해 분석을 수행하며, 결정 레이어는 최적화를 위한 정보에 기반하여 권고 사항을 생성합니다.

- **Performance Highlights**: 논문에서는 DTNs의 고도화된 피드백 추천 시스템이 실제 물리적 네트워크에 전달되는 과정을 강조하며, 다양한 혁신적인 기술들을 소개합니다. 이 기술들은 연결성, 통신 강인성, 보안 프로토콜 등 여러 분야에서 실제적 향상을 가져올 수 있음을 보여줍니다.



### Beyond the Boundaries of Proximal Policy Optimization (https://arxiv.org/abs/2411.00666)
- **What's New**: 본 연구는 Proximal Policy Optimization (PPO) 알고리즘을 새로운 관점에서 해석하여, 업데이트 벡터의 추정과 업데이트 응용을 분리하는 outer proximal policy optimization (outer-PPO) 프레임워크를 제안합니다.

- **Technical Details**: Outer-PPO는 PPO의 외부 루프에서 임의의 gradient-based optimizer를 사용하여 PPO의 업데이트 벡터를 적용합니다. 이 과정에서 우리는 non-unity learning rates와 momentum을 적용하여 성능 개선을 도모합니다.

- **Performance Highlights**: 모든 방법이 MinAtar 환경에서 baseline 성능을 초과하지는 않았지만, non-unity outer learning rates는 Brax와 Jumanji 환경에서 각각 5-10%의 유의미한 성능 향상을 기록했습니다.



### MoNTA: Accelerating Mixture-of-Experts Training with Network-Traffc-Aware Parallel Optimization (https://arxiv.org/abs/2411.00662)
- **What's New**: 이 논문에서는 Mixture of Experts (MoE) 모델을 위한 네트워크 트래픽 인식(parallel optimization) 방법 MoNTA를 제안합니다. 기존의 분산 학습 프레임워크는 통신 최적화를 고려하지 않고 있어, 특히 대규모 기본 모델에서는 성능이 저하되는 문제가 있었습니다. MoNTA는 이러한 문제를 해결하고자, 통신 볼륨 및 교육 클러스터의 네트워크 토폴로지에 기반하여 최적의 병렬 전략을 선택합니다.

- **Technical Details**: MoNTA는 inter-node와 intra-node 통신 자원을 활용하고 AllToAll 통신을 관리하여 CPU 및 GPU 간의 대칭성 문제를 해결합니다. 모든 통신 볼륨과 효율성을 적절히 나누어 통신의 오버헤드를 줄입니다. 또한, MoE 모델의 학습 중 발생하는 통신 충돌 문제를 분석하고 우선 순위 계획을 제공합니다.

- **Performance Highlights**: 실험 결과, MoNTA는 DeepSpeed 기법 대비 AllToAll 통신 성능을 최대 8배 향상시키는 성과를 얻었습니다. 2x70B 모델을 16개의 A800 카드로 훈련할 경우, 전체 지연(latency) 성능이 13% 향상되었습니다.



### Physics in Next-token Prediction (https://arxiv.org/abs/2411.00660)
Comments:
          First Submit

- **What's New**: 이번 연구에서는 Next-token Prediction (NTP)에서 정보 보존 법칙을 발견하고, 정보 용량의 제 1 법칙 (IC-1)을 제안합니다. 이는 자율 회귀 모델의 지능 출현의 본질이 정보 전송 과정임을 입증합니다. 또한 NTP에 Landauer의 원리를 도입하여 자율 회귀 모델의 훈련 과정과 에너지 소비의 관계를 밝히는 정보 용량의 제 2 법칙 (IC-2)을 수립하였습니다.

- **Technical Details**: 연구에서는 정보의 압축 과정과 지능의 출현 간의 관계를 명확히 하며, 토큰 사전 및 데이터셋을 통한 정보 전송에 대한 수학적 모델을 제시합니다. 또한, 자율 회귀 모델이 다음 토큰을 예측하는 과정에서 발생하는 정보의 조건부 양과 크로스 엔트로피 손실 함수의 최적화 과정 간의 일치를 설명하고 있습니다.

- **Performance Highlights**: 이 연구는 정보 보존 및 용량 법칙을 기반으로 자율 회귀 모델의 훈련이 데이터셋의 압축 과정임을 보여주며, 모델의 압축 능력이 향상될수록 지능이 더 높아짐을 시사합니다. 이는 실제 생산 과정에 대해 여러 가지 실용적인 지침을 제시할 수 있는 기초를 마련하였습니다.



### Generative AI and Agency in Education: A Critical Scoping Review and Thematic Analysis (https://arxiv.org/abs/2411.00631)
- **What's New**: 이번 스코핑 리뷰는 교육에서 Generative AI (GenAI)와 에이전시(agency)의 관계를 분석하고 있으며, Critical Digital Pedagogy의 관점에서 문헌을 검토하였습니다.

- **Technical Details**: PRISMA-ScR 가이드라인을 따르며, GenAI 환경에서 학습자 및 교사의 에이전시에 초점을 맞춘 10개의 연구를 학술 데이터베이스에서 수집했습니다. AI 지원 하이브리드 주제 분석을 통해 디지털 공간에서의 통제(Control in Digital Spaces), 가변적인 참여와 접근성(Variable Engagement and Access), 그리고 에이전시에 대한 변화하는 개념(Changing Notions of Agency)이라는 세 가지 핵심 주제를 도출했습니다.

- **Performance Highlights**: GenAI는 개인화 및 지원을 통해 학습자의 에이전시를 향상시킬 수 있지만, 일부 맥락에서는 교육 불평등을 심화시키고 학습자의 자율성을 저하시킬 위험이 있습니다. 이 리뷰는 GenAI의 에이전시 영향에 대한 현재 연구의 공백을 강조하며, 공평한 접근을 촉진하면서도 학습자 에이전시를 보존하는 프레임워크의 필요성을 제안합니다.



### STAA: Spatio-Temporal Attention Attribution for Real-Time Interpreting Transformer-based Video Models (https://arxiv.org/abs/2411.00630)
- **What's New**: 이 논문에서는 STAA(Spatio-Temporal Attention Attribution)라는 새로운 XAI(Explainable AI) 방법을 소개합니다. 이 방법은 비디오 Transformer 모델을 해석하는 데 있어 공간적 및 시간적 정보를 동시에 제공하는 혁신적인 접근 방식을 제시하여 기존의 한 차원적 설명 방식의 한계를 극복합니다.

- **Technical Details**: STAA는 Transformer 모델의 주의(attention) 값을 기반으로 spatial(공간적) 및 temporal(시간적) 정보를 동시에 제공하는 독창적인 방법입니다. 이 방법은 Kinetics-400 데이터셋을 활용하여 실험을 진행하며, 설명 품질을 평가하기 위한 메트릭스도 도입되었습니다. 또한, 동적 임계값 설정과 주의 집중 메커니즘을 적용하여 신호 대 잡음 비율(signal-to-noise ratio)을 개선했습니다.

- **Performance Highlights**: STAA는 전통적인 XAI 방법의 3% 이하의 계산 자원만으로 작동하여 실시간 비디오 XAI 분석 애플리케이션에 적합합니다. Kinetics-400 데이터셋을 활용한 평가 결과, 설명 품질과 계산 효율성 모두에서 유의미한 개선을 나타냈습니다.



### Lingma SWE-GPT: An Open Development-Process-Centric Language Model for Automated Software Improvemen (https://arxiv.org/abs/2411.00622)
- **What's New**: 이번 연구에서는 Lingma SWE-GPT 시리즈를 소개하며, 이는 소프트웨어 개선에 최적화된 오픈 소스 대형 언어 모델입니다. 이 모델들은 코드 제출 활동을 학습하고 시뮬레이션하여 소프트웨어 개발 과정에서의 동적 상호작용과 반복적 문제 해결을 체계적으로 통합합니다.

- **Technical Details**: Lingma SWE-GPT는 총 3단계로 구성된 소프트웨어 개선 프로세스를 시뮬레이션합니다: 레포지토리 이해, 결함 위치 파악, 패치 생성. 각 단계는 Chain-of-Thought (CoT) 방식으로 진행됩니다. 이 모델은 GitHub 문제에 대해 포괄적인 레포지토리 분석을 수행하고, 관련 코드를 검색하여 문제 해결을 위한 계획을 수립합니다.

- **Performance Highlights**: SWE-bench Verified에서 Lingma SWE-GPT 72B 모델은 30.20%의 문제 해결 성공률을 기록하였으며, Llama 3.1 405B보다 22.76% 개선된 성능을 보여줍니다. Lingma SWE-GPT 7B 모델은 18.20%의 성공률을 달성하여, 소형 모델이 자동화된 소프트웨어 엔지니어링 작업에서 큰 가능성을 보인다는 것을 나타냅니다.



### How to Bridge Spatial and Temporal Heterogeneity in Link Prediction? A Contrastive Method (https://arxiv.org/abs/2411.00612)
- **What's New**: 본 논문에서는 Temporal Heterogeneous Networks에서의 링크 예측을 위해 새로운 Contrastive Learning 기반 모델인 CLP를 제안합니다. 이 모델은 공간적 이질성(spatial heterogeneity)과 시간적 이질성(temporal heterogeneity)을 인코딩하기 위해 다중 뷰 계층 자가 감독 아키텍처를 활용합니다.

- **Technical Details**: CLP 모델은 공간적 특징 모델링 레이어와 시간적 정보 모델링 레이어를 포함합니다. 공간적 특징 요소는 노드 및 엣지 수준에서의 세부적인 토폴로지 분포 패턴을 포착하며, 시간적 정보 요소는 동적 그래프의 진화적 의존성을 시간 수준에서 인식합니다. 이 모든 요소는 대조 학습 관점에서 인코딩되어 링크 예측 작업을 위한 포괄적 자가 감독 계층 관계 모델링을 가능하게 합니다.

- **Performance Highlights**: 실험 결과에 따르면 CLP는 네 가지 실제 동적 이질적 네트워크 데이터셋에서 기존 최첨단 모델에 비해 AUC와 AP 측면에서 각각 평균 10.10% 및 13.44%의 성능 개선을 보여주었습니다.



### On Deep Learning for Geometric and Semantic Scene Understanding Using On-Vehicle 3D LiDAR (https://arxiv.org/abs/2411.00600)
Comments:
          PhD thesis (Durham University, Computer Science), 149 pages (the 2024 BMVA Sullivan Doctoral Thesis Prize runner-up). Includes published content from arXiv:2407.10159 (ECCV 2024 ORAL), arXiv:2303.11203 (CVPR 2023), and arXiv:2406.10068 (3DV 2021), with minor revisions to the examined version: this https URL

- **What's New**: 본 연구에서는 LiDAR(측距 센서) 기반의 작업에서의 정확성과 효율성을 개선하기 위해, 최초의 고충실도 128채널 3D LiDAR 데이터세트인 DurLAR를 제시합니다. 이 데이터세트는 파노라마 주변(Near Infrared) 및 반사 이미지를 포함하고 있습니다.

- **Technical Details**: DurLAR 데이터세트는 3D 포인트 클라우드와 관련된 기하학적 및 의미론적(scene understanding) 이해에 필수적입니다. 새로운 파이프라인은 더 작은 아키텍처를 적용하여 더 적은 Ground-truth 주석(annotation)으로도 경쟁력 있는 세분화(segmentation) 정확도를 달성합니다. 또한, Range-Aware Pointwise Distance Distribution (RAPiD) 기능과 RAPiD-Seg 아키텍처를 도입하여 기존 접근 방안보다 세분화 정확도를 높였습니다.

- **Performance Highlights**: 이 연구의 모든 기여는 동료 심사를 거친 학회에서 인정받았으며, 자율 주행 기술에서의 3D LiDAR 응용 프로그램의 정확도와 효율성이 모두 향상되었음을 강조합니다.



### Deep learning-based auto-contouring of organs/structures-at-risk for pediatric upper abdominal radiotherapy (https://arxiv.org/abs/2411.00594)
Comments:
          23 pages, 5 figures, 1 table. Submitted to Radiotherapy and Oncology (2024-11-01)

- **What's New**: 이번 연구에서는 소아 복부 상부 종양에서 위험 장기(organs-at-risk, OARs)를 delineate하기 위한 CT(computed tomography) 기반의 다기관 분할(segmentation) 모델을 개발했습니다. 이 모델은 다양한 데이터셋에서의 견고성을 평가하였습니다.

- **Technical Details**: 사용된 데이터셋은 189명의 소아 신장 종양 및 신경모세포종 환자의 수술 후 CT 이미지를 포함한 사내 데이터셋과 흉부-복부(thoracoabdominal) 영역을 포함하는 공공 데이터셋으로, 총 189개의 CT 이미지를 사용했습니다. 17개의 OARs가 delineated 되며, 9개는 전문의에 의해(Type 1), 나머지 8개는 TotalSegmentator를 통해 자동으로 분할(Type 2)되었습니다. 두 가지 모델(Model-PMC-UMCU와 Model-Combined)이 학습되었습니다.

- **Performance Highlights**: Model-PMC-UMCU는 9개의 OAR 중 5개에서 평균 DSC(Dice Similarity Coefficient) 값이 0.95 이상에 도달했습니다. 비장과 심장은 0.90에서 0.95 사이의 값을 보였고, 위-장 및 췌장은 0.90 이하의 값을 보였습니다. Model-Combined는 두 데이터셋에서 견고성이 향상된 결과를 보여주었습니다. 임상 평가에서는 전문의가 9개 Type 1 OAR 중 6개를 4점 이상, 8개 Type 2 OAR 중 6개를 3점 이상으로 평가하여 사용 가능성을 확인하였습니다.



### Adapting Language Models via Token Translation (https://arxiv.org/abs/2411.00593)
- **What's New**: Sparse Sinkhorn Token Translation (S2T2) 알고리즘을 도입하여 특정 도메인에서 효과적인 압축을 위한 맞춤형 토크나이저를 훈련하고, 원천과 목표 토큰 간의 변환을 학습합니다.

- **Technical Details**: S2T2는 높은 자원 소모 없이도 새로운 목표 도메인에서 훈련된 토큰의 분포를 학습하며, 사전 훈련된 모델을 활용하여 더 나은 다음 토큰 예측을 가능하게 합니다. 이를 통해 기존 모델에 비해 더 나은 perplexity 및 압축 성능을 발휘합니다.

- **Performance Highlights**: 실험 결과, S2T2는 아웃 오브 도메인(in-domain) 단백질 서열에 대해 perplexity와 압축 모두에서 개선된 결과를 보여줍니다. 또한 작은 모델에서 학습된 토큰 변환은 더 큰 모델에도 직접 이전 가능해 비용을 낮추면서도 효과를 누릴 수 있습니다.



### $\alpha$-TCVAE: On the relationship between Disentanglement and Diversity (https://arxiv.org/abs/2411.00588)
- **What's New**: 이번 연구에서는 새로운 총 상관관계(total correlation, TC) 하한을 사용하여 최적화된 변분 오토인코더(Variational Autoencoder)인 α-TCVAE를 소개합니다. 이 접근 방식은 분리된 표현의 질을 높이고, 잠재 변수의 정보를 극대화하는 데 중점을 둡니다.

- **Technical Details**: α-TCVAE는 정보 이론 개념에 기초하여, β-VAE 하한을 일반화하고, 기존의 변분 정보 병목(variational information bottleneck, VIB) 및 조건부 엔트로피 병목(conditional entropy bottleneck, CEB) 용어의 볼록한 조합으로 축소될 수 있는 새로운 하한을 제공합니다.

- **Performance Highlights**: α-TCVAE는 기존의 기준 모델들보다 더욱 분리된 표현을 일관되게 학습하고, 시각적 충실도를 희생하지 않고 더욱 다양하고 혁신적인 관측치를 생성합니다. 특히, MPI3D-Real 데이터셋에서 명확한 개선폭을 보이며, 복잡한 데이터셋을 표현하는 능력을 입증했습니다.



### Benchmarking Bias in Large Language Models during Role-Playing (https://arxiv.org/abs/2411.00585)
- **What's New**: 이번 논문에서는 BiasLens라는 공정성 테스트 프레임워크를 소개하며, 대형 언어 모델(LLMs)이 역할 연기(role-playing) 상황에서 사회적 편견을 드러내는 방법을 체계적으로 평가할 수 있도록 설계되었습니다.

- **Technical Details**: BiasLens는 두 가지 주요 구성 요소로 구성되어 있습니다: 테스트 입력 생성(test input generation)과 테스트 오라클 디자인(test oracle design). 이 프레임워크는 550개의 사회적 역할을 생성하고, 각각의 역할에 대해 편견을 유도할 수 있는 60개의 질문을 자동으로 생성합니다. 생성된 질문의 총 수는 33,000개에 달하며, 이 질문들은 Yes/No, 다중 선택(multiple-choice), 개방형(open-ended) 형식으로 다양합니다.

- **Performance Highlights**: BiasLens를 사용하여 OpenAI, Mistral AI, Meta, Alibaba, DeepSeek에서 출시한 6개의 최신 LLM들에 대한 광범위한 평가를 수행했습니다. 평가 결과, 총 72,716개의 편향된 응답이 발견되었으며, 각각의 모델은 7,754에서 16,963개의 편향된 응답을 생성했습니다. 이 연구는 역할 연기가 LLM 출력을 더욱 편향되게 할 수 있음을 강조하며, 향후 연구를 위한 데이터셋과 스크립트를 공개했습니다.



### Simulate and Optimise: A two-layer mortgage simulator for designing novel mortgage assistance products (https://arxiv.org/abs/2411.00563)
Comments:
          Accepted at the 5th ACM International Conference on AI in Finance

- **What's New**: 본 논문에서는 주택담보대출 관련 재정적 어려움을 겪는 가구들을 돕기 위한 새로운 두 층 구조의 접근법을 제안합니다. 이 접근법은 미국 주택 시장을 기반으로 한 시뮬레이션 환경에서 주택담보대출 보조 제품을 최적화하는 것을 목표로 하며, 기존의 비싼 파일럿 연구 대신 컴퓨터 시뮬레이션을 통해 경제적 제품을 설계하고 평가할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 제안된 두 층 구조는 제품 설계 층과 시뮬레이션 층으로 구성됩니다. 제품 설계 층은 새로운 주택담보대출 보조 제품을 도입하고, 시뮬레이션 층은 이러한 제품이 시장에 미치는 영향을 모델링합니다. 가구들은 시장 조건에 따라 적응하여 자신의 행동을 조정하며, 정책 혼합 정책 최적화(Proximal Policy Optimization) 및 일반화된 이점 추정(Generalized Advantage Estimation)을 통해 효용을 극대화합니다.

- **Performance Highlights**: 이 연구는 복잡한 시장 내에서 리얼타임으로 가구의 행동을 조정하여 새로운 주택담보대출 보조 제품을 설계할 수 있는 가능성을 제공합니다. 특히, 전통적 연구에 비해 비용을 절감하고 스케일링을 가능하게 하는 방식을 통해, 정책 입안자들에게 중요한 도구가 될 것입니다.



### LLM-KT: A Versatile Framework for Knowledge Transfer from Large Language Models to Collaborative Filtering (https://arxiv.org/abs/2411.00556)
Comments:
          accepted at ICDM 2024 (demo track)

- **What's New**: 본 연구에서는 LLM-KT라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 Collaborative Filtering (CF) 모델에 LLM (Large Language Model)에서 생성된 특징을 통합하여 모델의 성능을 향상시키도록 설계되었습니다. 기존 방법들과 달리, LLM-KT는 LLM에서 생성된 특징을 직접 입력으로 사용하는 것이 아니라 CF 모델의 중간 레이어에 주입하여 모델이 내부적으로 이를 재구성하고 활용하도록 합니다.

- **Technical Details**: LLM-KT 프레임워크는 모델 아키텍처를 변경하지 않고 CF 모델의 특정 내부 레이어에서 사용자 선호도를 재구성하는 데 중점을 둡니다. 이 과정은 LLM을 사용하여 각 사용자에 대한 짧은 선호 요약인 '프로필'을 생성하여, 텍스트 임베딩 모델을 사용해 이를 밀집 임베딩으로 변환한 후, 재구성 손실과 모델 손실의 가중합을 사용하여 훈련하는 방식으로 이루어집니다.

- **Performance Highlights**: MovieLens 및 Amazon 데이터셋을 기반으로 한 실험 결과, LLM-KT는 기존 CF 모델보다 최대 21%까지 성능을 개선함을 보여줍니다. 또, 이 모델은 context-aware 설정에서도 최첨단 방법들과 경쟁력이 있으며, 기존 방법들과 비교하여 더욱 다양한 CF 모델에 적용될 수 있는 가능성을 보입니다.



### Differentiable Physics-based System Identification for Robotic Manipulation of Elastoplastic Materials (https://arxiv.org/abs/2411.00554)
Comments:
          Underreivew on the Internation Journal of Robotics Research

- **What's New**: 이번 연구에서는 Differentiable Physics-based System Identification (DPSI) 프레임워크를 소개하여 로봇 팔이 불완전한 3D 포인트 클라우드와 간단한 조작 동작을 사용하여 엘라토플라스틱(elastoplastic) 물질과 환경의 물리 매개변수를 추론할 수 있도록 합니다. 이로 인해 높은 정밀도의 조작이 가능해집니다.

- **Technical Details**: DPSI 프레임워크는 로봇이 3D 카메라를 사용하여 물체를 조작하기 전에 멀티 뷰 포인트 클라우드를 생성하여 차폐(occlusion)를 최소화합니다. 가시화된 상태와 변형된 상태의 포인트 클라우드를 비교하여 물리적 파라미터를 업데이트하며, 물리 법칙을 기반으로 한 고충실도 시뮬레이션을 통해 실제 세계 시뮬레이션과 정렬(alignment)합니다.

- **Performance Highlights**: DPSI 프레임워크는 단 한 번의 실제 상호작용만으로도 Young's modulus, Poisson's ratio, yield stress 및 마찰계수와 같은 물리 매개변수를 정확하게 추정할 수 있습니다. 이를 통해 한계적인 물리 환경에서의 복잡한 조작 동작에도 불구하고 신뢰할 수 있는 변형 동작 예측이 가능합니다.



### Conditional Synthesis of 3D Molecules with Time Correction Sampler (https://arxiv.org/abs/2411.00551)
Comments:
          NeurIPS 2024

- **What's New**: 이번 논문에서는 분자 생성을 위한 새로운 접근 방식인 Time-Aware Conditional Synthesis (TACS)를 제안합니다. 이 방법은 적응형으로 제어되는 온라인 가이드를 통합하여 특화된 화학 성질을 목표로 하면서도 생성된 샘플의 유효성과 안정성을 유지합니다.

- **Technical Details**: TACS는 새로운 Diffusion Sampler인 Time Correction Sampler (TCS)를 사용하여 온라인 가이드를 제어하고, 각 역 전파 단계에서 분자가 정확한 매니폴드에 유지되도록 보장합니다. 이를 통해 데이터의 원래 분포를 유지하며 원하는 조건을 충족하는 샘플을 생성합니다.

- **Performance Highlights**: TACS는 3D 분자 생성을 위한 실험에서 이전의 최첨단 방법들을 능가하며 생성된 샘플이 원하는 양자 화학적 특성과 일치하는 동시에 데이터 일관성을 유지함을 입증했습니다.



### Generative AI-based Pipeline Architecture for Increasing Training Efficiency in Intelligent Weed Control Systems (https://arxiv.org/abs/2411.00548)
- **What's New**: 본 연구는 합성 이미지 생성을 위한 새로운 접근 방식을 제안하여, 지능형 잡초 제어를 위한 딥 러닝(Object Detection) 모델의 성능을 향상시킵니다. 이를 위해 Segment Anything Model(SAM)과 Stable Diffusion Model을 통합하여 실제 환경의 다양한 조건을 반영한 합성 이미지를 생성합니다.

- **Technical Details**: 연구진은 YOLO 모델을 통해 10%의 합성 이미지와 90%의 실제 이미지를 조합하여 훈련시킨 결과, mAP50과 mAP50-95 점수에서 뛰어난 성능을 보여주었습니다. 이 방법은 특히 이미지 품질과 데이터 효율성을 높이며, 자동 주석 달기 과정을 구현하여 더욱 향상된 데이터 처리 파이프라인을 제공합니다.

- **Performance Highlights**: YOLO 모델은 합성 이미지 데이터셋에서 더욱 높은 정확도를 보였으며, 이는 실제 이미지 데이터셋만으로 훈련된 모델보다 우수한 결과를 나타냅니다. 본 연구는 지능형 시스템의 자가 개선 기능을 통해 지속적인 학습 능력을 향상시킬 수 있는 가능성을 제시합니다.



### ReverseNER: A Self-Generated Example-Driven Framework for Zero-Shot Named Entity Recognition with Large Language Models (https://arxiv.org/abs/2411.00533)
- **What's New**: ReverseNER는 대규모 언어 모델(LLMs)이 제로샷(NER, Named Entity Recognition) 과제에서 갖는 한계를 극복하기 위해 제안된 새로운 프레임워크입니다. 이 방법은 NER 프로세스를 반대로 진행하여 신뢰할 수 있는 예시 라이브러리를 구축합니다.

- **Technical Details**: ReverseNER는 사전 훈련된 BERT 모델을 사용하여 작업 문장 간의 유사성을 계산한 후 클러스터링을 통해 주요 문장을 추출하고, 이를 기반으로 LLM이 관련 예시 문장 및 엔티티를 생성하도록 유도합니다. 이 과정은 문장 생성 시 특정 '특징 문장'의 구조를 복제하도록 LLM을 안내하여 세밀하게 주석이 달린 문장을 생성하게 합니다.

- **Performance Highlights**: 실험 결과, ReverseNER는 전통적인 제로샷 NER 방법보다 상당히 우수한 성능을 보이며, 데이터가 제한된 도메인에서의 NER 성능이 크게 향상되었음을 보여줍니다.



### Multi Modal Information Fusion of Acoustic and Linguistic Data for Decoding Dairy Cow Vocalizations in Animal Welfare Assessmen (https://arxiv.org/abs/2411.00477)
Comments:
          31 pages, 22 figures, 2 tables

- **What's New**: 본 연구는 정확한 축산에서 동물 복지를 강화하고 정서 상태를 평가하기 위한 동물의 음성을 해석하는 방법으로 다중 소스 데이터 융합(Multi-source data fusion)을 사용하는 것을 목표로 합니다.

- **Technical Details**: 우리는 유첫 소의 음성을 기록한 오디오를 전사(Transcribe)하여 텍스트 형태로 변환하는 자연어 처리(Natural Language Processing) 모델을 사용했습니다. 다양한 음향 특성(주파수, 지속 시간, 강도)과 텍스트 데이터를 융합하여 소의 음성을 포괄적으로 표현했습니다. 특수 개발된 온톨로지를 통해 융합된 다차원 데이터를 분석하여 불안 관련 특성을 식별하는 등의 작업을 수행했습니다.

- **Performance Highlights**: 고급 머신 러닝 알고리즘인 Random Forest, Support Vector Machine 및 Recurrent Neural Networks를 사용하여 소의 음성을 효과적으로 분류했습니다. 이러한 모델들은 실용적인 농업 환경에서의 컴퓨터 처리 요구와 데이터 품질 문제를 최적화하여 처리할 수 있도록 설계되었습니다.



### MIRFLEX: Music Information Retrieval Feature Library for Extraction (https://arxiv.org/abs/2411.00469)
Comments:
          2 pages, 4 tables, submitted to Extended Abstracts for the Late-Breaking Demo Session of the 25th Int. Society for Music Information Retrieval Conf., San Francisco, United States, 2024

- **What's New**: 본 논문에서는 음악 정보 검색(Music Information Retrieval, MIR) 연구를 지원하기 위해 다양한 음악 특징 추출 모델을 통합한 확장 가능한 모듈형 시스템인 MIRFLEX를 소개합니다.

- **Technical Details**: MIRFLEX는 키(key), 다운비트(downbeat), 장르(genre)와 같은 음악적 요소 및 악기 인식(instrument recognition), 보컬/악기 분류(vocals/instrumental classification), 성별 인식(vocals gender detection)과 같은 오디오 특성을 포함하는 다양한 특징 추출기를 제공합니다. 이 시스템은 최신 오픈 소스 모델 및 최첨단 기술을 사용하여, 연구자들이 음악 애플리케이션에 통합할 수 있는 잠재적인 라벨(latent or post-processed labels)을 추출할 수 있도록 합니다.

- **Performance Highlights**: MIRFLEX는 음악 데이터의 다양한 측면을 포괄하는 포괄적인 기능 세트를 제공하여 연구자들이 다양한 음악 관련 애플리케이션을 탐색할 수 있도록 합니다. 또한, 새로운 시스템의 손쉬운 통합을 지원하여 벤치마킹 및 비교 도구로서의 역할을 합니다.



### Uncertainty-based Offline Variational Bayesian Reinforcement Learning for Robustness under Diverse Data Corruptions (https://arxiv.org/abs/2411.00465)
Comments:
          Accepted to NeurIPS 2024

- **What's New**: 본 연구에서는 오프라인 데이터에서 데이터 손상으로 인한 불확실성을 포착하기 위해 베이즈 추론(Bayesian inference)을 최초로 제안하는 새로운 강인한 변분 베이즈 추론 방법(TRACER)을 소개합니다.

- **Technical Details**: TRACER는 모든 손상을 행동 가치 함수의 불확실성으로 모델링하며, 오프라인 데이터를 관찰로 사용하여 베이즈 추론 프레임워크 하에서 행동 가치 함수의 후변량 분포(posterior distribution)를 근사합니다. 또한, 엔트로피 기반 불확실성 측정을 사용하여 더 높은 불확실성을 가진 손상된 데이터를 식별하고, 손상된 데이터로 인한 손실을 조절하여 강인성과 성능을 높입니다.

- **Performance Highlights**: 실험 결과 TRACER는 개별 및 동시 데이터 손상에 대해 여러 최신 오프라인 RL 방법들보다 현저하게 우수한 성능을 보여줍니다.



### A Multi-Granularity Supervised Contrastive Framework for Remaining Useful Life Prediction of Aero-engines (https://arxiv.org/abs/2411.00461)
- **What's New**: 이 논문은 항공 엔진의 남은 유효 수명(RUL) 예측을 위한 새로운 다중 세분화를 적용한 감독 대조(MGSC) 프레임워크를 제안합니다. 기존의 RUL 예측 방식들이 MSE(Mean Square Error)만을 손실 함수로 사용하던 반면, MGSC는 특징 공간 구조를 고려하여 더 정확한 예측을 가능하게 합니다.

- **Technical Details**: MGSC 프레임워크는 거시형(bigger) 또는 미세형(finer) 대조 방식을 통합하여 배치 크기의 문제와 샘플 불균형 문제를 해결하며, RUL 레이블에 기반한 대조 감도를 최적화합니다. 이 프레임워크는 CNN(Convolutional Neural Network) 및 LSTM(Long Short-Term Memory) 네트워크를 기반으로 하여 설계되었습니다.

- **Performance Highlights**: MGSC 프레임워크는 CMAPSS 데이터셋을 사용하여 기존 기초선보다 RUL 예측의 정확성을 효과적으로 향상시키며, 이는 특징이 잘 정렬된 임베딩을 통해 이루어집니다.



### E2E-AFG: An End-to-End Model with Adaptive Filtering for Retrieval-Augmented Generation (https://arxiv.org/abs/2411.00437)
Comments:
          13 pages, 3 figures, 5 tables

- **What's New**: 이 논문에서는 외부 지식 기반에서 얻은 정보의 질을 개선하기 위해 답변 존재 판단과 텍스트 생성을 하나의 종합적인 end-to-end 프레임워크로 통합한 적응형 필터링 (adaptive filtering) 기법인 E2E-AFG를 제안합니다.

- **Technical Details**: E2E-AFG는 retrieval-augmented generation (RAG) 기법을 활용하며, 모델이 관련 콘텐츠에 보다 효과적으로 집중할 수 있도록 지원하여, 불필요한 정보의 영향을 줄여 정확한 답변을 생성합니다.

- **Performance Highlights**: E2E-AFG는 6개의 대표적인 지식 집약적 언어 데이터셋에서 평가되었으며, 모든 태스크에서 기본 모델들에 비해 일관되게 우수한 성과를 보여주며 제안된 접근 방식의 효과성과 강건성을 입증했습니다.



### DARD: A Multi-Agent Approach for Task-Oriented Dialog Systems (https://arxiv.org/abs/2411.00427)
- **What's New**: 본 논문에서는 DARD (Domain Assigned Response Delegation)라는 다중 에이전트 대화 시스템을 제안하여, 여러 도메인에 걸쳐 효과적으로 다이얼로그를 처리할 수 있음을 보여줍니다.

- **Technical Details**: DARD는 중앙 대화 관리자 에이전트에 의해 조정되는 도메인별 에이전트를 활용하여, 대화 맥락에 따라 사용자 메시지에 대한 응답을 생성합니다. 실험에는 Flan-T5-large, Mistral-7B, Claude Sonnet 3.0 모델이 포함되었습니다. MultiWOZ 2.2 데이터셋을 사용하여 성능을 평가하였으며, Joint State Accuracy (JSA), Inform, Success 및 BLEU 점수를 기준으로 성과를 측정합니다.

- **Performance Highlights**: DARD는 기존 방법보다 다이얼로그 정보율을 6.6%, 성공률을 4.1% 향상시키며, MultiWOZ 벤치마크에서 최신 성과를 달성하였습니다. 또한, MultiWOZ 데이터셋의 주석자 간 불일치 문제를 분석하였습니다.



### Self-Evolved Reward Learning for LLMs (https://arxiv.org/abs/2411.00418)
Comments:
          19 pages,6 figures

- **What's New**: 이 논문은 Self-Evolved Reward Learning (SER)이라는 새로운 접근 방식을 제안하고 있습니다. SER은 보상 모델(Reward Model, RM)이 자체적으로 추가 학습 데이터를 생성하여 스스로를 반복적으로 개선하는 방법입니다. 이로 인해 인간이 주석을 단 데이터에 대한 의존성을 줄이고도 언어 모델의 성능을 향상시킬 수 있습니다.

- **Technical Details**: SER 접근 방식에서는 RM이 스스로 높은 신뢰도의 예측을 학습하며, 초기에는 소량의 인간 주석 데이터를 사용하여 보상 모델을 훈련합니다. 그 후 RM은 자체 레이블링(self-labeling)과 반복 훈련을 통해 진화하며, RL 접근 방식에 따라 언어 모델 훈련에 사용됩니다. 이 과정을 통해 RM은 자신의 피드백으로부터 학습하면서 성능을 향상시킵니다.

- **Performance Highlights**: 여러 데이터셋과 LLM(대형 언어 모델)에서 실험을 수행한 결과, 인간 주석 데이터의 15%만 사용하더라도, 기존의 전량 데이터를 사용한 모델과 유사한 성능을 달성하는 것으로 나타났습니다. 최종적으로는 평균 7.88%의 성능 향상을 기록하며, 사람의 주석이 포함된 전체 데이터셋을 사용하는 모델의 성능을 초과할 수 있는 잠재력을 보여주었습니다.



### On the Opportunities of Large Language Models for Programming Process Data (https://arxiv.org/abs/2411.00414)
Comments:
          14 pages

- **What's New**: 이번 연구는 대형 언어 모델 (Large Language Models, LLMs)을 활용하여 프로그래밍 과정 데이터를 자동으로 요약하고 피드백을 제공할 수 있는 가능성을 논의합니다. 이는 컴퓨터 교육 연구와 실천 분야의 자동화를 한 단계 더 앞당길 것으로 예상됩니다.

- **Technical Details**: 연구에서 언급되는 프로그래밍 과정 데이터는 과제 제출, 키 입력, 코드 변경 등을 포함하여 여러 단계로 나눌 수 있습니다. LLMs는 텍스트 기반의 콘텐츠 변환을 통해 프로그래밍 과정 데이터를 분석하고 피드백을 생성하는 데 활용됩니다.

- **Performance Highlights**: LLMs를 사용하여 프로그래밍 과정의 요약 및 피드백을 생성한 사례 연구에서, 연구자는 프로그래밍 상황에 대한 보다 정교한 피드백을 제공할 수 있는 가능성을 확인하였습니다.



### Adapting While Learning: Grounding LLMs for Scientific Problems with Intelligent Tool Usage Adaptation (https://arxiv.org/abs/2411.00412)
Comments:
          26 pages, 15 figures

- **What's New**: 이 연구는 기존의 Large Language Models (LLMs)의 한계를 극복하기 위한 새로운 접근 방식을 제안합니다. 특히, 문제의 복잡성을 분석한 후 적절한 솔루션 접근 방식을 선택하는 인간 전문가의 방식에서 영감을 받아, LLM을 위한 두 가지 구성 요소로 이루어진 새로운 fine-tuning 방법을 제시합니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 구성 요소로 나뉩니다. 첫 번째는 World Knowledge Distillation (WKD)로, LLM이 도구 정보를 활용해 생성된 솔루션으로부터 직접 학습하여 도메인 지식을 내재화하는 과정입니다. 두 번째는 Tool Usage Adaptation (TUA)로, 모델의 직접 응답 정확성을 기준으로 문제를 쉬운 문제와 어려운 문제로 나누어, 쉬운 문제에서는 WKD와 동일한 정렬 목표를 유지하면서도 어려운 문제에 대해서는 도구 사용으로 전환할 수 있도록 훈련합니다.

- **Performance Highlights**: 이 방법은 수학, 기후과학, 역학을 포함한 여섯 가지 과학 벤치마크 데이터셋에서 검증되었습니다. 평균적으로, 제안된 모델은 모든 데이터셋에서 정답 정확성이 28.18%, 도구 사용의 정밀도가 13.89% 향상되었으며, GPT-4o 및 Claude-3.5와 같은 최첨단 모델을 초월하는 성과를 보였습니다.



### Statistical Guarantees for Lifelong Reinforcement Learning using PAC-Bayesian Theory (https://arxiv.org/abs/2411.00401)
- **What's New**: Lifelong reinforcement learning (RL)의 새로운 알고리즘인 EPIC이 소개되었습니다. EPIC은 PAC-Bayes 이론을 기반으로 하며, 여러 작업에서의 성과를 향상시키기 위해 설계되었습니다.

- **Technical Details**: EPIC은 에이전트의 정책 분포를 학습하여 'world policy'라고 불리는 공유 정책 분포를 생성합니다. 이는 새로운 작업에 빠르게 적응하면서 이전 경험에서 소중한 지식을 유지할 수 있도록 합니다. 이 알고리즘의 일반화 성능은 메모리에 보존된 이전 작업의 수와 관계가 있음을 이론적으로 분석했습니다.

- **Performance Highlights**: 다양한 환경에서의 광범위한 실험을 통해 EPIC이 기존의 방법들보다 크게 뛰어난 성능을 보였으며, 이론적 보장과 실제 효용을 모두 제공함을 입증했습니다.



### Right this way: Can VLMs Guide Us to See More to Answer Questions? (https://arxiv.org/abs/2411.00394)
Comments:
          NeurIPS 2024

- **What's New**: 이번 연구에서는 비전 언어 모델(Vision Language Models, VLMs)의 정보 부족 상황에서의 방향성 안내(Directional Guidance) 제공 능력을 평가하는 새로운 비주얼 질문 응답(Visual Question Answering, VQA) 과제를 정의합니다. 이 과제는 특히 시각 장애인에게 더 나은 지원을 제공하기 위한 것입니다.

- **Technical Details**: 연구팀은 현재의 VLMs가 정보를 충분히 평가하지 못하는 문제를 해결하기 위해, 정보 부족 상황에서 이미지를 조정하는 방법을 안내하는 능력을 평가합니다. 새로운 방향성 안내 데이터셋과 자동화된 VQA 데이터 증가 프레임워크를 통해, 시뮬레이션된 정보 부족 시나리오에서 VLM을 학습시킵니다.

- **Performance Highlights**: 세 가지 오픈 소스 VLM을 대상으로 한 실험 결과, 제안된 방향성 안내 작업에서 성능이 유의미하게 향상되었습니다. 또한, 가장 성능이 좋은 모델은 GPT-4o (CoT)보다 3% 높은 정확도를 기록하였습니다.



### Advantages of Neural Population Coding for Deep Learning (https://arxiv.org/abs/2411.00393)
- **What's New**: 이 논문에서는 신경망의 출력층에서 인구 코드를 사용하는 이점을 조사하고, 이를 단일 뉴런 출력 및 원-핫 벡터와 비교하여 결과의 강건성과 정확성을 향상 시킨다고 제안합니다.

- **Technical Details**: 연구는 단일 변수, 인구 코드, 원-핫 벡터 출력의 노이즈 강건성을 비교하고, 이론적인 분석과 함께 T-LESS 데이터셋을 통해 이미지에서 객체 방향을 예측하는 작업을 수행하여 인구 코드가 불확실한 출력을 처리하는 데 어떻게 기여하는지를 보여줍니다.

- **Performance Highlights**: 결과적으로, 인구 코드를 사용하면 객체 지향 예측의 정확성이 향상되며, 특히 대칭 객체로부터의 모호한 자세를 처리하는 데 유리함을 보입니다.



### Preventing Dimensional Collapse in Self-Supervised Learning via Orthogonality Regularization (https://arxiv.org/abs/2411.00392)
Comments:
          accepted by NeurIPS 2024 as a poster

- **What's New**: 본 논문에서는 self-supervised learning (SSL)의 차원 붕괴 문제를 해결하기 위해 orthogonal regularization (OR) 접근 방식을 제안합니다. 이 방법은 pretraining 과정에서 convolutional 및 linear layer의 weight matrices의 직교성을 보장하여 성능을 향상시킵니다.

- **Technical Details**: 제안된 OR 방법은 두 가지 주요 직교성 정규화 기법인 Soft Orthogonality (SO)와 Spectral Restricted Isometry Property Regularization (SRIP)를 포함하고 있습니다. 이 방법들은 13개의 최신 SSL 방식에 통합되어 실험되었으며, CNN과 Transformer 기반 아키텍처에서 일관된 성능 향상을 보여주었습니다.

- **Performance Highlights**: 실험 결과, OR은 CIFAR-100에 대한 linear probe 정확도를 향상시키고, IMAGENET-1k에서 BYOL에 적용했을 때 분류 및 객체 탐지 작업에서 다운스트림 성능을 크게 개선했습니다. OR은 기존 SSL 아키텍처나 하이퍼파라미터를 수정할 필요 없이 이러한 성능 향상을 달성했습니다.



### MetaMetrics-MT: Tuning Meta-Metrics for Machine Translation via Human Preference Calibration (https://arxiv.org/abs/2411.00390)
Comments:
          Preprint

- **What's New**: MetaMetrics-MT는 기계 번역(Machine Translation, MT) 작업을 평가하기 위해 인간 선호도와 밀접하게 연관되도록 설계된 혁신적인 메트릭입니다. Bayesian optimization과 Gaussian Processes를 활용하여 기존 MT 메트릭의 인간 평가와의 상관관계를 최적화합니다.

- **Technical Details**: MetaMetrics-MT는 여러 개의 메트릭을 활용하여 MT 작업을 평가하며, 각 메트릭에 특정 가중치를 부여하여 성능을 최적화합니다. 이 메트릭은 독립적이지 않고 서로 다른 메트릭의 점수를 조합합니다. 그 과정에서 Gaussian Processes를 사용하여 메타 메트릭 기능을 모델링하고, 인간 평가 점수와의 상관관계를 최대화하는 가중치를 결정합니다.

- **Performance Highlights**: WMT24 metric shared task에서 MetaMetrics-MT는 기존 모든 기준선을 초과하는 성능을 보여주며, 참조(reference)-기반 설정에서 새 벤치마크를 수립했습니다. 또한, 참조-free 설정에서도 유사한 결과를 달성하여 효율성을 높였습니다. 이 메트릭은 40GB 메모리의 상업적 GPU에서 작동할 수 있는 반면, XCOMET-Ensemble과 같은 비교 메트릭은 최소 80GB의 높은 메모리를 요구합니다.



### Generalizability of Memorization Neural Networks (https://arxiv.org/abs/2411.00372)
- **What's New**: 이번 논문은 memorization neural networks의 일반화 이론에 대한 최초의 이론적 분석을 제공합니다. i.i.d. 데이터셋을 사용하여 일반화 가능성을 연구하고, 이를 위한 조건과 알고리즘을 제시합니다.

- **Technical Details**: 논문에서는 i.i.d. 데이터셋에 필요한 메모리네트워크의 매개변수 및 네트워크의 크기에 대한 이론적 결과를 다룹니다. 가로가 데이터 차원과 같아야 일반화 가능성이 있으며 이는 기존의 최적 매개변수를 가진 memorization networks가 일반화 가능성이 없음을 의미합니다. 또한, 샘플 복잡도에 대한 하한선과 상수 매개변수를 가진 메모리 알고리즘의 정확한 샘플 복잡도를 제시했습니다.

- **Performance Highlights**: 전반적으로 논문은 매개변수 수를 줄이고, 일반화 가능성을 높이기 위한 알고리즘을 설명하며, 특정 데이터 분포에 대해선 메모이제이션 네트워크가 데이터 차원에 따라 지수적인 매개변수 수를 필요로 한다고 밝혔습니다.



### TextDestroyer: A Training- and Annotation-Free Diffusion Method for Destroying Anomal Text from Images (https://arxiv.org/abs/2411.00355)
- **What's New**: 이 논문에서 제안하는 TextDestroyer는 최초의 트레이닝 및 주석 없이 장면 텍스트를 제거할 수 있는 방법입니다. 기존의 텍스트 제거 방법은 복잡한 주석 작업과 재교육이 필요했으며, 희미하지만 인식 가능한 텍스트 정보가 남아 개인 정보 보호와 콘텐츠 숨기기를 저해하는 문제가 있었습니다.

- **Technical Details**: TextDestroyer는 세 단계의 계층적 프로세스를 통해 정확한 텍스트 마스크를 생성합니다. 먼저, 가우시안 분포를 사용하여 잠재 시작 코드에서 텍스트 영역을 혼란스럽게 조작한 후, 복원 과정에서는 원본 잠재값에서의 자기 주의 (self-attention) 키와 값을 참조하여 손상된 배경을 복원합니다. 각 역전 단계에서 저장된 잠재 코드는 복원 과정에서 대체에 사용됩니다.

- **Performance Highlights**: TextDestroyer의 장점은 (1) 수작업 데이터 주석과 자원 소모적인 훈련을 제거; (2) 인식 가능한 흔적 없이 철저한 텍스트 파괴; (3) 현실 세계의 장면과 생성된 이미지 모두에서 잘 작동하는 우수한 일반화 능력을 보여줍니다.



### Examining Attacks on Consensus and Incentive Systems in Proof-of-Work Blockchains: A Systematic Literature Review (https://arxiv.org/abs/2411.00349)
- **What's New**: 암호화폐의 보안 취약성과 관련된 새로운 공격 조합의 이해는 장기적인 보안을 보장하기 위한 중요한 요소입니다. 이 논문은 특정 공격의 수익성 분석 뿐만 아니라 여러 공격 방법의 조합이 폭넓은 공격성을 강화할 수 있다는 점을 강조합니다.

- **Technical Details**: 이 논문은 암호화폐의 보안 메커니즘, 특히 블록체인 기술의 합의 메커니즘(Consensus Mechanism)인 Proof of Work (PoW)와 인센티브 메커니즘(Incentive Mechanism)의 상호작용을 분석합니다. 주요 공격으로는 selfish mining, double-spending, block withholding이 있으며, 이들 공격은 단독으로 혹은 결합하여 수행될 수 있습니다.

- **Performance Highlights**: 공격 조합이 성공률과 수익성을 높일 수 있는 방법을 제시합니다. 특히, 여러 채굴 풀 간의 경쟁에 의해 발생하는 경제적 결과를 분석하였으며, 미래 연구의 방향성을 제시하는 디자인 가이드라인도 포함되어 있습니다.



### Attention Tracker: Detecting Prompt Injection Attacks in LLMs (https://arxiv.org/abs/2411.00348)
Comments:
          Project page: this https URL

- **What's New**: 이 논문은 Large Language Models (LLMs)의 prompt injection 공격 메커니즘을 분석하고, 이를 기반으로 Attention Tracker라는 새로운 공격 탐지 방법을 제안합니다. 새로운 개념인 distraction effect를 도입하여 특정 attention heads가 원래의 지시사항에서 주입된 지시사항으로 초점을 이동하는 현상을 설명합니다.

- **Technical Details**: 논문에서는 distraction effect를 활용하여, 주입된 공격을 탐지하기 위한 training-free 접근법인 \\attn을 소개합니다. \\attn은 기본 LLM 추론 과정에서 수집된 attention 점수를 사용하여 prompt injection 공격을 감지하며, 기존 방법보다 AUROC 점수가 최대 10.0% 향상된 성능을 보입니다. 이 접근법은 다양한 모델 및 데이터셋에 대해 일반화됩니다.

- **Performance Highlights**: \attn은 Open-Prompt-Injection 및 deepset와 같은 두 개의 공개 데이터셋에서 테스트 되었으며, 모든 평가에서 뛰어난 탐지 정확도를 달성하였습니다. 특히, 기존 training-free 탐지 방법보다 AUROC 스코어가 평균 31.3% 향상되었습니다. 이 방법은 Small LMs에서도 효과적으로 작동하여, 이전 방법들의 한계를 극복하였습니다.



### An Untethered Bioinspired Robotic Tensegrity Dolphin with Multi-Flexibility Design for Aquatic Locomotion (https://arxiv.org/abs/2411.00347)
Comments:
          7 pages, 13 figures

- **What's New**: 이번 논문은 생체 모방 접근법을 통해 돌고래의 유연성을 모방하는 부드러운 돌고래 로봇의 첫 단계에 대해 다룹니다. 기존의 돌고래 로봇은 단순한 두 개의 케이블 구동 모터로 작동되지만, 새로운 디자인은 유연한 꼬리와 조정 가능한 실리콘 피부를 도입하여 근육 역학을 본뜬 동작의 유연성을 제공합니다.

- **Technical Details**: 이 로봇은 각각의 설계 요소가 어떻게 유연성을 제공하는지에 대해 분석합니다. 본체의 유연성은 단단한 머리와 유연한 꼬리의 조합으로 이루어지며, 케이블 구동 시스템을 통한 근육 수축을 모방하여 매끄럽고 연속적인 움직임을 생성합니다. 부드러운 스켈레톤 구조는 수리 구조를 더욱 개선하여 수영 속도와 효율성에 영향을 미치는 매개변수를 탐구하는 데 사용할 수 있습니다.

- **Performance Highlights**: 현재 시제품은 오직 직진만 가능한 상태이지만, 이 연구는 유연한 설계가 수중 로봇의 이동성을 어떻게 향상시킬 수 있는지를 보여주는 첫 단계입니다. 회전과 같은 컨트롤이 가능하도록 디자인이 발전할 가능성을 제시합니다.



### On the Exploration of LM-Based Soft Modular Robot Design (https://arxiv.org/abs/2411.00345)
Comments:
          8 pages, 7 figures

- **What's New**: 최근의 대형 언어 모델(LLMs)이 실제 지식 모델링과 지식 기반 생성 작업에서 좋은 능력을 보여주고 있습니다. 본 논문에서는 LLMs를 활용해 소프트 모듈 로봇의 설계를 지원하는 가능성을 탐구합니다.

- **Technical Details**: 소프트 모듈 로봇 디자인 과정을 시퀀스 생성(task)으로 보고, 자연어(natural language)로 표현된 주요 요구사항을 포착하여 로봇의 설계 시퀀스에 반영합니다. 시뮬레이션 도구를 통해 피드백을 제공하고, 다섯 가지 평가 메트릭을 도입하여 로봇 디자인 품질을 자동으로 평가하는 방식을 구현했습니다.

- **Performance Highlights**: 우리의 모델은 UNI-와 BI-방향 로코모션(locomotion) 및 계단 하강(stair-descending) 기능을 가진 소프트 모듈 로봇 설계 평가에서 우수한 성능을 보였으며, 전통적인 인간 디자인과는 다른 새로운 디자인을 제시하여 효과적인 성능을 발휘했습니다.



### StepCountJITAI: simulation environment for RL with application to physical activity adaptive intervention (https://arxiv.org/abs/2411.00336)
Comments:
          Accepted at NeurIPS 2024 workshop on Behavioral ML

- **What's New**: 이 논문에서는 RL (Reinforcement Learning) 방법이 실제 물리적 활동 개입(physical activity interventions)에서 정책 학습(policy learning)을 지원할 수 있는 새로운 시뮬레이션 환경인 StepCountJITAI를 소개합니다. StepCountJITAI는 맥락 불확실성(context uncertainty)과 행동 역학(behavioral dynamics)을 모델링하여 적응형 개입 최적화(adaptive intervention optimization)에 대한 연구를 가속화할 수 있도록 설계되었습니다.

- **Technical Details**: StepCountJITAI 환경은 habituation level (습관화 수준)과 disengagement risk (탈퇴 위험)이라는 두 가지 주요 행동 변수를 통해 메시지의 맥락화 정확성과 즉각적 보상(step count) 간의 관계를 모델링합니다. 이 환경은 Stochasticity(확률적 변동성)를 포함하여 개별 참가자 간 및 개별 참가자 내에서의 변동성을 반영하고, 참가자의 메시지 수신에 대한 반응을 학습하는 과정에서 발생할 수 있는 다양한 불확실성을 고려합니다.

- **Performance Highlights**: StepCountJITAI는 기존의 RL 연구와의 호환성을 극대화하기 위해 표준 API(gymnasium)를 사용하여 오픈 소스 구현을 제공하며, RL 알고리즘 연구자들이 데이터가 부족한 상황에서도 효과적으로 정책 학습을 수행할 수 있는 기회를 제공합니다. 이 도구는 연구자들이 개입 프로그램의 행동 역학을 탐구하고 최적화하는 데 도움을 줄 수 있습니다.



### Personalized Federated Learning via Feature Distribution Adaptation (https://arxiv.org/abs/2411.00329)
Comments:
          38th Annual Conference on Neural Information Processing Systems (NeurIPS), 2024

- **What's New**: 이 논문에서는 Personalize Federated Learning (PFL) 접근 방식을 통해 다양한 클라이언트에 맞춤화된 모델을 학습하는 새로운 방법인 pFedFDA를 제안합니다. 이는 생성 모델링(task) 관점에서 representation learning을 정의하고, 기능 분포(feature distribution)를 조정하여 개인화된 모델을 생성합니다.

- **Technical Details**: pFedFDA는 global feature distribution에 기반하여 특징을 학습하는 생성적 분류기(generative classifier)를 사용하여 각 클라이언트의 로컬 데이터 분포에 맞춤화된 개인화된 모델을 생성합니다. 이 과정은 로컬 및 글로벌 데이터 분포의 스무딩(smoothing)을 통해 bias-variance trade-off를 조정하는 로컬-글로벌 보간 방법(local-global interpolation method)으로 안내됩니다.

- **Performance Highlights**: 컴퓨터 비전 벤치마크에서 pFedFDA는 데이터가 제한된 환경에서도 크게 개선된 성능을 보여줍니다. 여러 설정에서 평균 모델 정확도가 6% 이상 향상되었으며, 일반 벤치마크에서도 현재 최첨단 기술에 비해 거의 1% 이내로 경쟁력을 유지합니다.



### Constant Acceleration Flow (https://arxiv.org/abs/2411.00322)
- **What's New**: 이번 논문에서는 Constant Acceleration Flow (CAF)라는 새로운 ODE 프레임워크를 제안합니다. CAF는 상수 가속도 모델을 도입하여 더 정교하고 정확한 ODE 플로우 예측이 가능합니다.

- **Technical Details**: CAF는 기존의 Rectified flow를 일반화하며, 가속도를 학습 가능한 추가 변수로 도입합니다. 또한, 초기 속도 조건화(initial velocity conditioning)와 재유동 과정(reflow process)을 통해 ODE 플로우의 정확도를 높입니다.

- **Performance Highlights**: CAF는 CIFAR-10 및 ImageNet 64×64 데이터셋에서 기존 최첨단 방법보다 우수한 FID 점수를 기록하였으며, 특히 몇 단계 만에 놀라운 성능을 보여주었습니다.



### C2A: Client-Customized Adaptation for Parameter-Efficient Federated Learning (https://arxiv.org/abs/2411.00311)
Comments:
          Published at Findings of ACL 2023

- **What's New**: 이번 논문에서는 새로운 하이퍼네트워크 기반의 연합 학습 프레임워크인 Client-Customized Adaptation (C2A)을 제안하였습니다. C2A는 고객의 정보를 기반으로 고객 맞춤형 어댑터를 생성하여 비균질 데이터 환경에서도 안정적인 학습을 가능하게 합니다.

- **Technical Details**: C2A는 하이퍼네트워크를 활용하여 각 클라이언트의 데이터 분포에 맞춘 어댑터 매개변수를 생성합니다. 또한, 파라미터 수를 줄이기 위해 팩토리화된 하이퍼네트워크를 도입하여 성능을 유지하면서도 메모리 효율성을 높였습니다. 지나치게 비균질한 데이터 상황에서 클라이언트 드리프트(client drift)를 최소화하는 방법론입니다.

- **Performance Highlights**: C2A는 다양한 비균질 설정에서 뛰어난 일반화 성능을 보여주었으며, 기타 PEFT 방법들에 비해 훈련 효율성을 크게 향상시켰습니다. 실험 결과, C2A는 연합 학습 시 큰 클라이언트 드리프트 문제를 효과적으로 완화하는 것으로 확인되었습니다.



### Inducing Semi-Structured Sparsity by Masking for Efficient Model Inference in Convolutional Networks (https://arxiv.org/abs/2411.00288)
Comments:
          15 pages, 3 figures; this work will be presented at the NeurIPS 2024 Workshop on Fine-Tuning in Modern Machine Learning: Principles and Scalability (FITML)

- **What's New**: 본 논문에서는 컨볼루션 모델의 성능 저하 없이 2배 이상의 속도를 개선할 수 있는 새로운 방법을 제안합니다. 이 방법은 마스킹(masking)을 통해 반구조적(sparsity) 패턴을 학습하여, 기존 하드웨어 가속 하에서 효율적으로 사용될 수 있습니다.

- **Technical Details**: 제안된 방법은 컨볼루션 연산에서 반구조적(sparsity) 마스킹 패턴을 학습하여 모델의 가중치와 구조를 변경하지 않고 속도를 향상시킵니다. 특히, N:M sparsity 개념을 도입하여 특정 숫자의 요소를 제로로 설정함으로써 연산의 효율성을 증가시켰습니다. 이 과정에서 Gumbel-Max 트릭을 활용해 각 패턴을 선택하는 확률을 모델링합니다.

- **Performance Highlights**: 본 연구의 성과로는 기존 핸드헬드를 활용하여 제안된 반구조적 마스킹 기술이 CV 분류 작업에서 매우 약간의 성능 손실로 반구조적(sparsity) 패턴 학습을 가능하게 했다는 점입니다. 이는 대규모 모델이나 온라인 설정에서 특히 이점으로 작용합니다.



### MBExplainer: Multilevel bandit-based explanations for downstream models with augmented graph embeddings (https://arxiv.org/abs/2411.00287)
- **What's New**: 이 논문은 GNN(그래프 신경망)에서 생성된 그래프 임베딩과 추가적인 테이블 형식의 특징들을 결합하여ensemble 모델의 출력을 설명하는 방법인 MBExplainer를 제안합니다.

- **Technical Details**: MBExplainer는 모델-불가지론적(agnostic) 접근 방식으로, 서브그래프, 주요 노드 특징, 그리고 주요 추가 특징으로 구성된 세 가지 요소의 중요도를 평가합니다. Shapley 값을 사용하여 각 요소의 기여도를 계산하고, 몬테 카를로 트리 탐색(Monte Carlo Tree Search) 알고리즘을 통해 효율적인 로컬 검색 공간을 탐색합니다.

- **Performance Highlights**: MBExplainer는 여러 공공 그래프 데이터셋에서의 노드 및 그래프 분류 작업에 대한 종합적인 수치 예제를 통해 그 효과iveness를 입증합니다.



### SimpleFSDP: Simpler Fully Sharded Data Parallel with torch.comp (https://arxiv.org/abs/2411.00284)
- **What's New**: SimpleFSDP는 PyTorch-native 컴파일러 기반의 Fully Sharded Data Parallel (FSDP) 프레임워크로, 간단한 구현과 유지 보수 용이성을 제공하며, 컴파일러 백엔드 최적화를 통해 성능 향상을 이끕니다.

- **Technical Details**: SimpleFSDP는 기존 PyTorch 프리미티브(parametrizations, selective activation checkpointing, DTensor)를 활용한 독특한 집합 통신(collective communication) 구현을 특징으로 하며, TorchInductor 백엔드에서 효과적으로 컴퓨테이션-커뮤니케이션 겹침(computation-communication overlapping)을 위해 IR 노드의 버킷화(bucketing) 및 재정렬(reordering)을 수행합니다.

- **Performance Highlights**: Llama 3 모델(특히 405B)에서 SimpleFSDP를 사용하여 TorchTitan에서 실시한 광범위한 평가에서, 기존 FSDP2 eager 프레임워크에 비해 최대 28.54%의 메모리 감소와 68.67%의 처리량 개선을 보여주었습니다.



### Improving Traffic Flow Predictions with SGCN-LSTM: A Hybrid Model for Spatial and Temporal Dependencies (https://arxiv.org/abs/2411.00282)
Comments:
          5 pages, 6 figures

- **What's New**: 이 논문은 Signal-Enhanced Graph Convolutional Network Long Short Term Memory (SGCN-LSTM) 모델을 도입하여 교통 속도를 예측하는 새로운 방법론을 제시합니다. 기존의 연구는 주로 고정 가중치 그래프를 사용하여 공간적 의존성을 모델링하는 데 집중했지만, 이 논문에서는 시간적 패턴과 복잡한 공간적 의존성을 동시에 고려합니다.

- **Technical Details**: SGCN-LSTM 모델은 그래프 합성곱 네트워크 (GCN)와 장단기 메모리 (LSTM) 네트워크의 조합으로, 교통 데이터를 분석하기 위해 신호 증강 (Signal-Enhanced) 기법을 사용합니다. 이 모델은 도로 네트워크 내의 노드 간 상호 작용과 동적 에지 관계를 포착하여 예측의 정확성을 향상시킵니다.

- **Performance Highlights**: PEMS-BAY 도로 네트워크 교통 데이터셋에 대한 광범위한 실험을 통해 SGCN-LSTM 모델은 기준 모델들에 비해 평균 절대 오차 (MAE), 평균 제곱근 오차 (RMSE), 평균 절대 백분율 오차 (MAPE)에서 유의미한 개선을 보여주었습니다.



### Quantifying calibration error in modern neural networks through evidence based theory (https://arxiv.org/abs/2411.00265)
- **What's New**: 신경망의 신뢰성 평가를 위해 주관적 논리(subjective logic)를 도입한 새로운 프레임워크를 제안합니다. 이 방법은 기존의 신뢰성 지표인 Expected Calibration Error (ECE)를 보완하며, 신뢰, 불신, 불확실성을 포괄적으로 측정할 수 있습니다.

- **Technical Details**: 제안된 방법은 예측된 확률을 클러스터링하고 적합한 융합 연산자를 사용하여 의견을 융합하여 신뢰성 의견을 산출합니다. 이를 통해 신경망의 신뢰성을 정량화하고 해석 가능한 형태로 제공합니다.

- **Performance Highlights**: MNIST와 CIFAR-10 데이터셋에서 실험을 통해, 제안된 방법이 신뢰성을 향상시키는 효과를 보임을 입증합니다. 특히, 헬스케어 및 자율 시스템과 같은 민감한 분야에서 활용 가능성이 높습니다.



### Deep Learning Through A Telescoping Lens: A Simple Model Provides Empirical Insights On Grokking, Gradient Boosting & Beyond (https://arxiv.org/abs/2411.00247)
Comments:
          Accepted at Conference on Neural Information Processing Systems (NeurIPS) 2024

- **What's New**: 이번 연구에서는 깊은 학습(deep learning)의 예측성 없는 성능을 이해하기 위한 간단하지만 정확한 신경망(neural network) 모델을 제시합니다. 이 모델은 교육 과정에서의 첫 번째 순서 근사를 사용하여 신경망의 작동 방식을 실증적으로 분석할 수 있는 새로운 도구를 제공합니다.

- **Technical Details**: 제안된 모델은 훈련 중의 개별 업데이트에 대한 근사를 점진적으로 정의하여, 실제로 훈련된 신경망의 동작에 더 가깝게 근사합니다. 이 모델을 통해 데이터 세트 불규칙성에서 신경망과 그래디언트 부스팅 트리(gradient boosted trees) 간의 성능 차이를 조사하고, 가중치 평균화(weight averaging)와 신경망 학습 간의 연결 고리를 탐구합니다.

- **Performance Highlights**: 세 가지 사례 연구(case studies)를 통해 이 모델은 신경망의 비선형적 일반화 성능을 이해하는 데 도움을 주며, 이로 인해 예측할 수 없는 성능을 설명할 수 있는 메트릭(metric)을 구축하고 추출할 수 있음을 보여줍니다. 이 연구는 딥러닝 연습에서의 설계 선택 및 최적화 전략의 영향을 이해하는 데 도움을 줍니다.



### From Easy to Hard: Tackling Quantum Problems with Learned Gadgets For Real Hardwar (https://arxiv.org/abs/2411.00230)
Comments:
          15 pages, 8 figures. Comments are encouraged

- **What's New**: 본 논문에서는 Reinforcement Learning (RL) 기반의 Gadget Reinforcement Learning (GRL) 알고리즘을 개발하여 양자 회로 설계에서의 새로운 접근 방식을 제시합니다. GRL은 복합 게이트('gadgets')를 자동으로 학습하고, 이를 RL 에이전트의 추가적인 행동으로 통합합니다. 이 방법은 NP-hard 문제인 양자 해밀토니안의 바닥 상태를 찾는 데 성공적으로 적용되었습니다.

- **Technical Details**: GRL 알고리즘은 양자 회로 공간에서 PQC (Parameterized Quantum Circuits)를 탐색하는 RL 에이전트와 프로그램 합성을 기반으로 하는 라이브러리 빌딩 알고리즘을 결합합니다. GRL은 TFIM (Transverse Field Ising Model)의 바닥 상태를 찾는 데 초점을 맞추며, 저차원 TFIM에서 학습한 복합 행동을 사용하여 보다 복잡한 문제를 해결합니다.

- **Performance Highlights**: GRL을 사용하면 TFIM의 바닥 상태를 10^7 배 더 정확하게 추정할 수 있으며, RL 전용 접근 방식과 비교하여 훨씬 적은 수의 학습 가능한 매개변수로도 우수한 성능을 보입니다. GRL은 실제 하드웨어에 적합하며, 문제의 난이도가 증가하고 시스템 크기가 커져도 더욱 효과적으로 확장됩니다.



### Protecting Feed-Forward Networks from Adversarial Attacks Using Predictive Coding (https://arxiv.org/abs/2411.00222)
- **What's New**: 이 연구에서는 adversarial attack에 대한 방어를 위해 predictive coding network(PCnet)를 보조 단계로 활용한 새로운 접근법을 제안합니다. 이 방법은 기존 모델을 변경하지 않고, 입력 이미지의 변화를 역전시키는 데 도움을 줍니다.

- **Technical Details**: PCnet은 feed-forward network에 매끄럽게 통합되어 adversarial perturbation에 대한 저항성을 크게 증가시킵니다. MNIST와 CIFAR10 데이터셋에서의 실험은 각각 약 82% 및 65%의 향상을 보여줍니다. 또한, PCnet은 작은 데이터셋의 서브셋에서 훈련되어 생성적 특성을 통해 perturbed 이미지를 원본에 가깝게 되돌리는 데 활용됩니다.

- **Performance Highlights**: 이 연구에서 PCnet의 효과는 MNIST의 경우 약 82% 그리고 CIFAR10의 경우 약 65%의 robustess 개선을 보여 주었습니다. 이러한 접근법은 인공 신경망 분류기의 보안성과 신뢰성을 높이는 가능성을 가지고 있습니다.



### ADAPT: A Game-Theoretic and Neuro-Symbolic Framework for Automated Distributed Adaptive Penetration Testing (https://arxiv.org/abs/2411.00217)
- **What's New**: AI와 통합된 현대의 주요 인프라 시스템, 특히 의료 분야에서 새로운 취약점이 발생하였으며, 이를 해결하기 위해 분산 적응형 자동 침투 테스트 프레임워크인 ADAPT를 제안합니다.

- **Technical Details**: ADAPT는 게임 이론 및 신경 기호 프레임워크를 활용하여 의료 인프라의 고유한 사이버 보안 문제를 해결하기 위해 설계되었습니다. 이 프레임워크는 매크로 게임과 마이크로 게임으로 구성되어 있으며, 공격 모델에 따른 전략을 지속적으로 업데이트합니다.

- **Performance Highlights**: ADAPT는 의료 시스템의 AI 네트워크에 대한 리스크 평가를 가능하게 하며, 다양한 적대적 AI 기술에 대한 효과적인 대비책을 수치 실험을 통해 증명하였습니다.



### Using Large Language Models for a standard assessment mapping for sustainable communities (https://arxiv.org/abs/2411.00208)
Comments:
          8 pages, 2 figures

- **What's New**: 이 논문은 ISO 37101 프레임워크를 사용하여 도시 지속 가능성 평가를 자동화하고 표준화하기 위해 대형 언어 모델(LLMs)을 활용하는 새로운 접근 방식을 제시합니다. 연구는 파리의 참여 예산과 PROBONO Horizon 2020 프로젝트의 데이터셋을 기반으로 LLM을 통해 다양한 도시 이니셔티브를 빠르고 일관되게 분류하는 방법을 모색합니다.

- **Technical Details**: 연구는 LLM의 능력을 활용하여 ISO 37101의 6가지 '목적'과 12가지 '이슈'에 따른 도시 지속 가능성 이니셔티브를 자동으로 분류하고 평가하는 접근 방식을 개발했습니다. 사용된 모델은 gpt-3.5-turbo이며, 이 모델은 복잡한 관계를 이해하고 세부 응답을 생성할 수 있는 능력이 뛰어납니다. 이 방법론은 6x12 행렬 구조를 이용하여 각 이니셔티브가 아홉 목적에 어떻게 기여하는지를 종합적으로 평가합니다.

- **Performance Highlights**: 연구 결과, LLM을 활용한 방법은 시간 절약과 일관성 향상을 보여주며, 기존의 인간 주도 평가 방식에 비해 효과적입니다. 그러나 결과 해석과 윤리적 고려 사항에서 인간의 전문성이 여전히 필요하다는 점도 강조됩니다.  이 연구는 도시 계획에서 AI 응용에 대한 새로운 방향을 제시하고, 국제 지속 가능한 개발 목표 달성에 기여할 수 있는 가능성을 열어줍니다.



### Compositional Automata Embeddings for Goal-Conditioned Reinforcement Learning (https://arxiv.org/abs/2411.00205)
- **What's New**: 본 논문에서는 goal-conditioned reinforcement learning (RL)에서 cDFA(compositional Deterministic Finite Automata)를 활용하여 시간적 목표 표현을 제안합니다. 기존의 목표 표현이 지닌 한계를 극복하기 위해, cDFA를 사용하여 RL 에이전트를 유도함으로써 명확한 시간적 의미론을 제공하며 해석을 용이하게 합니다.

- **Technical Details**: cDFA는 DFAs의 조합으로 정의되어 있으며, 목표 조건 RL 에서 시간적 작업을 처리하는 데 적합합니다. 또한, 본 연구에서는 cDFA의 인코딩을 위해 그래프 주의망(GATv2)을 사용하는 방법을 제안하고, reach-avoid derived (RAD) DFA에 대한 사전 학습을 통해 RL 에이전트가 다양한 cDFA 작업 클래스를 제로샷 제너럴리제이션(zero-shot generalization)할 수 있도록 합니다.

- **Performance Highlights**: 실험을 통해 제안된 사전 학습 방법이 다양한 cDFA 작업 클래스에 대해 제로샷 제너럴리제이션을 가능하게 하고, 계층적 방법의 단점인 단기 최적성(myopic suboptimality)을 극복하는 정책 전문화(policy specialization)를 가속화함을 입증하였습니다.



### Whole-Herd Elephant Pose Estimation from Drone Data for Collective Behavior Analysis (https://arxiv.org/abs/2411.00196)
Comments:
          Accepted to CV4Animals: Computer Vision for Animal Behavior Tracking and Modeling Workshop in conjunction with Computer Vision and Pattern Recognition 2024

- **What's New**: 이 연구는 드론에서 수집된 데이터를 기반으로 야생에서 코끼리 행동을 연구하는 자동 포즈 추정(automated pose estimation)의 혁신적인 응용을 보여줍니다.  케냐의 샘부루 국립 보호 구역에서 촬영한 비디오 영상을 활용하였고, DeepLabCut과 YOLO-NAS-Pose라는 두 가지 포즈 추정 워크플로우를 평가하였습니다.

- **Technical Details**: Drone 기술을 통해 다수의 코끼리를 한 프레임에서 관찰하였으며, 촬영 중 드론은 정지 상태에서 특정 높이를 유지했습니다. 본 연구에서는 총 23개의 비디오와 133개의 프레임을 통해 1308마리의 코끼리를 분석하였습니다. YOLO-NAS-Pose와 DeepLabCut 모델을 사용하여 코끼리의 주요 포인트(머리, 척추, 귀 등)를 추정하였습니다.

- **Performance Highlights**: YOLO-NAS-Pose 모델이 DeepLabCut보다 RMSE, PCK, OKS와 같은 지표에서 우수한 성능을 보였으며, 객체 탐지 평가에서도 DeepLabCut을 초과하는 성과를 나타냈습니다. 이를 통해 야생 행동 연구와 드론 모니터링의 새로운 접근 방식을 제시하며, 동물 보존에 대한 중요한 시사점을 제공합니다.



### Monitoring fairness in machine learning models that predict patient mortality in the ICU (https://arxiv.org/abs/2411.00190)
Comments:
          8 pages

- **What's New**: 이 연구는 ICU(중환자실)에서 환자 사망을 예측하는 기계 학습 모델의 공정성을 모니터링하는 새로운 방법을 제안합니다. 본 연구에서는 다양한 인종, 성별 및 의학적 진단을 가진 환자 집단에 대한 모델의 성과를 분석하고 文헌 편향(Documentation bias) 문제를 탐구합니다.

- **Technical Details**: 이 논문에서는 일반화 가법 모델(Generalised Additive Models, GAM)을 기반으로 한 새로운 ICU 사망 예측 모델을 개발하였으며, Fairlearn Python 패키지를 활용하여 공정성을 평가했습니다. 논문은 입력 데이터 변화(drift), 훈련 로그, 예측 정확도 및 공정성 메트릭스를 모니터링하는 시스템을 개발했습니다.

- **Performance Highlights**: 모델의 전체 정확도는 0.92331이며, 여성과 비여성 환자에 대한 정확도는 각각 0.92187 및 0.9245로 공정성이 비교적 잘 유지되고 있음을 보여줍니다. 진단 그룹에 따른 auROC는 0.82에서 0.96으로 다르게 나타나며, 문서화 편향의 영향을 상세히 분석하였습니다.



### Self-Healing Machine Learning: A Framework for Autonomous Adaptation in Real-World Environments (https://arxiv.org/abs/2411.00186)
Comments:
          Advances in Neural Information Processing Systems 38 (NeurIPS 2024)

- **What's New**: 이 논문에서는 'self-healing machine learning (SHML)'이라는 새로운 패러다임을 제안합니다. 기존의 접근 방식이 직관적이지 않은 문제를 해결하는 데 한계가 있었던 반면, SHML은 모델 저하의 원인을 자율적으로 진단하고 이를 바탕으로 수정 조치를 제안합니다.

- **Technical Details**: SHML을 적응 조치의 공간에 대해 기대 위험을 최소화하는 최적화 문제로 정식화합니다. 이 시스템은 자기 진단을 수행하기 위해 대형 언어 모델(large language models)을 활용하여 DGP의 구조에 대해 추론하고, 수정 조치를 제안 및 평가합니다.

- **Performance Highlights**: H-LLM의 다양한 구성 요소를 분석하여 언제 그리고 왜 잘 작동하는지를 이해하려고 시도하였으며, 자기 치유 머신 러닝의 잠재력을 보여줍니다.



### Clinical Evaluation of Medical Image Synthesis: A Case Study in Wireless Capsule Endoscopy (https://arxiv.org/abs/2411.00178)
- **What's New**: 이 논문은 임상 연구 및 훈련을 위한 데이터 공유의 필요성을 강조하며, 인공지능(AI) 모델을 활용한 합성 데이터 생성(SDG)이 개인정보 보호 장벽을 극복할 수 있음을 보여줍니다. 특히, 무선 캡슐 내시경(WCE) 이미지를 사용하여 염증성 장질환(IBD)을 진단하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 연구는 합성 이미지를 체계적으로 평가하기 위한 프로토콜을 제시하고, TIDE-II라는 새로운 변별 오토인코더 기반 모델을 적용하여 고해상도 WCE 이미지를 합성하였습니다. 최종적으로 10명의 국제 WCE 전문가에 의해 이미지 품질, 다양성, 사실성 및 임상 의사결정에 대한 종합적인 정성적 평가가 이루어졌습니다.

- **Performance Highlights**: TIDE-II 모델은 임상적으로 중요한 WCE 이미지를 생성하여 데이터 부족 문제를 해결하고 진단 도구를 향상시키는 데 기여하는 것으로 나타났습니다. 제안된 프로토콜은 의학 이미지 생성 기술에 대한 향후 연구에 대한 참고 자료로 활용될 수 있습니다.



### Beyond Label Attention: Transparency in Language Models for Automated Medical Coding via Dictionary Learning (https://arxiv.org/abs/2411.00173)
- **What's New**: 이 논문은 의료 언어 모델에서 의료 코딩의 해석 가능성을 향상시키기 위해 딕셔너리 학습(Dictionary Learning) 기법을 활용하고 있습니다. 기존의 주의 메커니즘이 ICD 코드와 관련 없는 불필요한 토큰을 강조하는 문제를 해결하고, 의료 관련 개념을 효율적으로 캡처할 수 있는 해석 가능한 딕셔너리를 구축합니다.

- **Technical Details**: 저자는 두 가지 희소 오토인코더(sparse autoencoder) 접근 방식 중 하나인 L1 최소화와 SPINE 손실 함수의 사용을 통해 해석 가능한 표현을 생성하는 방법을 제시합니다. AutoCodeDL이라는 새로운 해석 가능성 프레임워크를 통해 학습된 딕셔너리 특성을 결합하여 다운스트림 ICD 예측의 설명 가능성을 향상시킵니다.

- **Performance Highlights**: 이 기술을 통해 저자는 학습된 딕셔너리의 특성이 모델 행동을 안내하고 90% 이상의 의료 관련 없는 토큰의 숨겨진 의미를 설명할 수 있음을 보여주며, 이는 사람들에게 이해할 수 있는 방식으로 이루어집니다.



### Creativity in the Age of AI: Evaluating the Impact of Generative AI on Design Outputs and Designers' Creative Thinking (https://arxiv.org/abs/2411.00168)
- **What's New**: Generative AI (GenAI)가 디자인 워크플로우에 점점 더 많이 사용됨에 따라 그 영향에 대한 연구가 필요하다는 점이 새롭게 제기되었습니다. 이 연구는 GenAI의 지원이 디자인 결과 및 디자이너의 창의성에 미치는 영향을 살펴보았습니다.

- **Technical Details**: 실험을 통해 참여자들에게 GenAI 지원과 비지원 조건에서 광고 디자인을 요청하였고, 전문가 평가자들은 GenAI 지원 디자인이 더 창의적이고 비정상적이라고 평가했습니다. 하지만 비주얼 매력, 브랜드 정렬, 유용성 측면에서는 유의미한 차이가 없었습니다.

- **Performance Highlights**: 디자이너들의 전반적인 창의적 사고 능력은 GenAI에 의해 크게 향상되지 않았지만, 사용자들은 고유 언어와 AI 노출 경험에 따라 다르게 영향을 받았습니다. 특히, 원어민 영어 사용자는 AI 사용 시 긴장이 증가한 반면, GenAI를 처음 접한 디자이너는 아이디어 유창성과 유연성 같은 발산적 사고에서 향상을 보였습니다.



### PSL: Rethinking and Improving Softmax Loss from Pairwise Perspective for Recommendation (https://arxiv.org/abs/2411.00163)
- **What's New**: 본 논문에서는 기존의 Softmax Loss (SL)의 두 가지 주요 한계를 분석하고 이를 개선하기 위해 Pairwise Softmax Loss (PSL)이라는 새로운 손실 함수 범주를 제안합니다. PSL은 SL에서 사용된 지수 함수를 다른 적절한 activation function으로 대체하면서 성능을 향상시킵니다.

- **Technical Details**: PSL은 SL을 쌍(pairwise) 방식으로 재구성하여 양수-음수 쌍 간의 점수 차이에 손실을 적용합니다. 이를 통해 PSL은 DCG와 같은 전통적인 ranking metrics에 대한 이론적 연결을 확립하며, ReLU 또는 Tanh와 같은 적합한 surrogate activation을 선택함으로써 더 나은 성능을 발휘합니다.

- **Performance Highlights**: PSL은 추천 정확도, OOD(Out-of-Distribution) 강인성, 및 노이즈 저항성에 있어 기존 손실 함수들보다 우수한 성능을 보입니다. 본 연구에서 수행한 다양한 실험을 통해 PSL의 효과성과 강인성을 검증했습니다.



### Scaling Up Membership Inference: When and How Attacks Succeed on Large Language Models (https://arxiv.org/abs/2411.00154)
Comments:
          Our code is available at this https URL

- **What's New**: 본 논문에서는 Membership Inference Attack (MIA)를 대규모 언어 모델(LLM)에 적용했을 때의 효과를 밝히고자 합니다. 이전 연구들에서는 MIA가 LLM에 대한 효과가 없다고 결론 내렸지만, 본 연구에서는 여러 문서를 동시에 테스트할 때만 MIA가 유효하다고 주장합니다.

- **Technical Details**: 우리는 MIA의 새로운 평가 기준을 도입하며, 데이터 샘플의 연속적 스케일을 측정하는 새로운 벤치마크를 구축했습니다. MIA 방법은 문장(n-grams)부터 문서(다수의 토큰의 조각)에 이르기까지 다양한 스케일에서 평가됩니다. 또한, 기존의 Dataset Inference (DI) 방법을 MIA에 맞게 조정하여 문서와 문서 집합 수준에서 성능을 평가할 수 있도록 하였습니다.

- **Performance Highlights**: MIA 방법은 문서 집합에 대해 80% 이상의 AUROC 점수를 달성하였습니다. 또한, 미세 조정 데이터에 대한 MIA 성능은 88% 이상의 AUROC를 기록하여 Continual Learning MIA가 효과적임을 입증하였습니다. 논문은 다양한 LLM 미세 조정 시나리오에서 성능 향상을 보여주며, 우리는 LLM에서 MIA 성능의 포괄적 분석을 수행하는 데 필요한 기반을 마련했습니다.



### Schema Augmentation for Zero-Shot Domain Adaptation in Dialogue State Tracking (https://arxiv.org/abs/2411.00150)
- **What's New**: 이 연구에서는 대화 상태 추적(Dialogue State Tracking, DST)에 대한 제로샷(domain adaptation) 접근 방식에서 애드혹한 대안으로 'Schema Augmentation' 기법을 도입한다. 이 기법은 슬롯 이름에 대한 변형을 도입하여 언어 모델의 제로샷 도메인 적응력을 크게 향상시킨다.

- **Technical Details**: Schema Augmentation은 기존 슬롯 이름에서 유의어(synonyms)를 사용하거나 비의미적 코드(non-semantic codes)로 교체하여 데이터 증강(data augmentation)을 수행한다. 연구에서는 Synonym Schema Augmentation(SSA)와 Encoding Schema Augmentation(ESA)의 두 가지 변형을 제안하며, 각 변형에는 단일(single) 및 다중(multi) 방식이 포함된다.

- **Performance Highlights**: MultiWOZ와 SpokenWOZ 데이터셋에서 실시한 실험 결과, 제안된 접근 방식은 기존 기준선 대비 최대 두 배의 정확도를 달성하였다. 특히, 보지 못한 도메인에 대해 정확도가 크게 향상되었으며, 모든 도메인에 대해 동등하거나 우수한 성능을 유지하였다.



### JudgeRank: Leveraging Large Language Models for Reasoning-Intensive Reranking (https://arxiv.org/abs/2411.00142)
- **What's New**: JudgeRank는 문서 관련성을 평가하기 위한 새로운 에이전틱 리랭커로, 인간의 인지 과정을 모방하여 문서 평가의 한계를 극복합니다. 이 접근법은 쿼리 분석, 문서 분석, 그리고 관련성 판단의 세 가지 주요 단계로 구성됩니다.

- **Technical Details**: 문서 요구 정보를 확보하기 위해 JudgeRank는 크게 세 가지 단계로 운영됩니다: (1) 쿼리 분석을 통해 핵심 문제를 파악하고, (2) 문서 분석에서 쿼리 인식을 반영한 요약을 추출하며, (3) 최종적으로 문서의 관련성을 간결하게 평가합니다. 이러한 방식은 Chain-of-Thought와 LLM-as-a-Judge 접근법에서 영감을 받았습니다.

- **Performance Highlights**: JudgeRank는 BRIGHT(Reasoning-Intensive Generative Retrieval Tasks) 벤치마크에서 첫 단계 검색 방법들보다 현저한 성능 향상을 보여주었으며, BEIR 벤치마크에서도 최신 리랭커와 동등한 성능을 발휘했습니다. 다양한 크기의 LLM에서 JudgeRank의 일반화 성능이 우수하게 나타났으며, 여러 모델을 앙상블 하였을 때 더욱 향상된 성능을 보였습니다.



### Learning Low-Dimensional Strain Models of Soft Robots by Looking at the Evolution of Their Shape with Application to Model-Based Contro (https://arxiv.org/abs/2411.00138)
Comments:
          8 pages, under review

- **What's New**: 이 논문에서는 연속적인 소프트 로봇을 위한 정확하고 해석이 용이한 저차원 물리 기반 모델을 학습하는 새로운 방법을 제시합니다. 이미지 데이터를 활용하여 소프트 로봇의 운동을 설명하기 위한 최소한의 필수 세그먼트를 결정하고, 동적 회귀(dynamic regression) 및 변형 희소화(strain sparsification) 알고리즘을 적용하여 관련 변형을 식별하고 모델의 동력을 정의합니다.

- **Technical Details**: 우리의 방법은 연속 소프트 로봇의 운동을 설명하기 위해 
1. kinematic fusion algorithm을 통해 
2. 동적 회귀 및 변형 희소화 알고리즘을 적용하여 동적 모델을 자동으로 학습하는 end-to-end 접근 방식을 사용합니다. 
이러한 과정에서 
- 
PCS 모델의 물리적 구조를 보존함으로써 기존 모델 기반 제어 정책과 결합할 수 있는 동적 모델을 신속하게 배포할 수 있습니다.

- **Performance Highlights**: 시뮬레이션을 통해 다양한 평면 소프트 매니퓰레이터를 평가하고, 다른 학습 전략과 성능을 비교한 결과, 제안된 모델은 25배 더 정확하며 훈련 데이터 외부에서도 높은 성능을 보여줍니다. 특히, 최적 제어 방법을 사용했을 때 모델 기반 제어 정책과 쉽게 결합할 수 있는 Lagrangian 구조를 통해 제어 성능을 향상시킬 수 있었습니다.



### Beyond Accuracy: Ensuring Correct Predictions With Correct Rationales (https://arxiv.org/abs/2411.00132)
Comments:
          In Proceedings of the 38th Conference on Neural Information Processing Systems (NeurIPS 2024)

- **What's New**: 이 논문은 대규모 (large) pretrained foundation 모델들이 인간 전문가를 초월하는 성과를 보임에도 불구하고, 그 예측의 근거가 제대로 검증되지 않는 문제를 다룹니다. 안전한 배포를 위해 정확한 예측과 올바른 근거의 이중 확인을 요구합니다.

- **Technical Details**: 제안된 두 단계 방식은 첫 번째 단계에서 시각 인식 (visual recognition) 작업을 위한 구조화된 근거 데이터셋을 큐레이션하고, 두 번째 단계에서는 수동 주석 (manual annotations) 없이 각 근거에 대한 시각적 증거를 분리하고 위치를 알아내기 위해 근거에 기반한 최적화 방법을 제안합니다.

- **Performance Highlights**: 광범위한 작업에서, 제안한 모델은 예측 정확도 (prediction accuracy)에서 기존 최첨단 모델보다 최대 10.1% 향상된 성능을 보였으며, 근거의 올바름 (rationale correctness) 또한 크게 개선하여 로컬라이제이션 (localization)에서 7.5%, 분리 (disentanglement)에서 36.5%의 향상을 기록했습니다.



### Training and Evaluating Causal Forecasting Models for Time-Series (https://arxiv.org/abs/2411.00126)
- **What's New**: 이번 연구에서는 기존의 time-series 모델의 일반화 문제를 해결하기 위해 causal time-series forecasting 모델을 개발했습니다. 특히, 이 모델은 학습 데이터의 분포가 아닌 외부 행동의 영향을 예측할 수 있도록 설계되었습니다.

- **Technical Details**: Orthogonal statistical learning 프레임워크를 활용하여 causal forecasting 모델을 학습했습니다. 이 모델은 price와 같은 처리(treatment)와 demand와 같은 결과(outcome) 간의 인과 관계를 파악하는 것을 목표로 합니다. 모델은 관측된 특징(observational features)을 조건으로 하여 처리 변경에 따른 결과 변화에 초점을 두고 있습니다.

- **Performance Highlights**: 우리의 causal forecasting 모델은 전통적인 time-series 모델 및 최신 causal 모델에 비해 각각 RDDs로 추정된 causal 효과와 36% 및 1% 더 가까운 예측 결과를 보여주었습니다.



### I Can Hear You: Selective Robust Training for Deepfake Audio Detection (https://arxiv.org/abs/2411.00121)
- **What's New**: 이번 연구에서는 AI가 생성한 음성을 탐지하기 위한 새로운 데이터셋 DeepFakeVox-HQ를 구축하였습니다. 이 데이터셋은 130만 개의 샘플로 구성되어 있으며, 그 중 27만 개는 고품질의 deepfake 샘플로 이루어져 있습니다.

- **Technical Details**: DeepFakeVox-HQ 데이터셋은 14개 다양한 출처에서 수집된 27만 개의 고품질 deepfake 음성 샘플을 포함하고 있습니다. 연구팀은 Frequency-Selective Adversarial Training (F-SAT) 방법론을 제안하여 모델의 강건성을 향상시키는 데 중점을 두고, 고주파 성분에 집중하여 공격에 대한 저항력을 높였습니다.

- **Performance Highlights**: F-SAT 훈련을 통해 기본 모델 성능을 33% 향상시켰으며, 청정 샘플에서 7.7% 및 오염된 샘플에서 29.3%의 정확도 개량을 달성했습니다. 이러한 성과는 최신 RawNet3 모델의 성능을 초월하여, 다양한 공격과 오염 조건에서도 뛰어난 성능을 나타냈습니다.



### Prospective Learning: Learning for a Dynamic Futur (https://arxiv.org/abs/2411.00109)
Comments:
          Accepted to NeurIPS 2024

- **What's New**: 이번 논문은 시간에 따른 데이터의 분포와 목표의 변화에 적합한 새로운 이론적 프레임워크인 "Prospective Learning"(PL)을 제시합니다. 이는 기존의 아키텍처인 probably approximately correct (PAC) learning 프레임워크를 보완하기 위해 개발되었습니다.

- **Technical Details**: Prospective Learning은 데이터가 미지의 확률 분포에서 발생하는 것을 가정하는 대신에, 데이터가 미지의 확률 과정을 통해 발생한다고 가정합니다. 이 프레임워크에서는 시간이 추가 입력으로 포함되며, 시간에 따라 최적의 가설이 변화할 수 있음을 강조합니다. 논문에서는 이러한 개념을 반영한 Prospective ERM(경험적 위험 최소화)을 제안하고, 이 방법이 Bayes 위험에 수렴한다는 것을 증명합니다.

- **Performance Highlights**: Numerical 실험 결과에 따르면, Prospective ERM은 MNIST와 CIFAR-10 데이터셋에서 구축된 합성 및 시각 인식 문제를 학습할 수 있는 것으로 나타났습니다. 기존의 ERM이나 온라인 및 지속적인 학습 알고리즘은 동적 데이터 분포에 대해 효과적이지 못한 것으로 평가되었습니다.



### PARTNR: A Benchmark for Planning and Reasoning in Embodied Multi-agent Tasks (https://arxiv.org/abs/2411.00081)
Comments:
          Alphabetical author order

- **What's New**: 이 논문에서는 가정용 로봇의 작업 및 의사결정을 평가하는 새로운 기준 'PARTNR(Planning And Reasoning Tasks in humaN-Robot collaboration)'를 제안합니다. 이 기준은 60채소에 걸쳐 100,000개의 자연어 작업으로 구성되어 있으며, 로봇과 인간 사이의 협업을 조명합니다.

- **Technical Details**: PARTNR는 공간적, 시간적 제약 및 이질적 에이전트의 능력 제약을 특징으로 하며, 4가지 작업 유형(제약 없는 작업, 공간적 작업, 시간적 작업, 이질적 작업)을 다룹니다. 이 기준은 Large Language Models (LLMs)를 활용하여 반자동으로 작업과 평가 기능을 생성하며, 시뮬레이션을 통해 검증합니다. LLM의 효과를 평가하기 위해 우리는 환경의 관측 가능성, 중앙 집중식 및 분산 다중 에이전트 제어 등을 연구합니다.

- **Performance Highlights**: 최신 모델들은 작업 추적 및 오류 복구에서 큰 한계를 보였으며, 2명의 인간이 협력할 때보다 1.5배 많은 단계를 필요로 했습니다. LLM을 인간에게 맞춰 조정하면 8.6배 더 빠르게 추론할 수 있으며, 이는 실제 인간과의 상호작용 시 더 적은 단계로 작업을 수행할 수 있게 합니다. PARTNR는 협력형 에이전트가 직면한 주요 과제를 강조하며, 향후 연구 방향에 대한 통찰을 제공합니다.



### How Good Are We? Evaluating Cell AI Foundation Models in Kidney Pathology with Human-in-the-Loop Enrichmen (https://arxiv.org/abs/2411.00078)
- **What's New**: 이번 연구는 디지털 병리학 분야에서의 AI 기반 세포 모델의 성능을 최초로 다각적으로 평가하고, 인간의 개입을 통해 데이터 보강 전략을 개발하여 모델 성능을 향상시키려는 시도를 했습니다.

- **Technical Details**: 이 연구에서는 2,542개의 신장 전체 슬라이드 이미지(Whole Slide Images, WSIs)를 포함한 다센터, 다질병, 다종의 데이터셋을 구축하였습니다. 세 가지 최신(cell foundation) 모델인 Cellpose, StarDist, CellViT를 평가하고, 데이터 보강 전략으로는 사람의 개입을 통한 잘못된(predictions) 예측 패치를 수정하고 여러 모델의 예측을 결합하여 성능을 향상시키는 방법을 사용하였습니다.

- **Performance Highlights**: 세 가지 모델 모두 데이터 보강 후 성능이 개선되었고, StarDist 모델이 최고 F1 점수 0.82를 기록했습니다. 하지만 F1 점수가 가장 높은 기본 모델(CellViT)은 미세 조정 후 최상의 세분화 결과를 내지 못했습니다. 이는 ‘좋은’ 및 ‘나쁜’ 이미지 패치를 결합하는 전략이 가장 효과적임을 보여줍니다.



### RPS: A Generic Reservoir Patterns Sampler (https://arxiv.org/abs/2411.00074)
Comments:
          Accepted at 2024 IEEE International Conference on Big Data

- **What's New**: 이 연구는 streaming batch 데이터에서 직접적으로 패턴 샘플링을 지원하는 가중치 저수지(reservoir)를 이용한 새로운 접근 방식을 소개합니다. 이 방법은 복잡한 데이터 스트림, 특히 순차적(sequential) 및 가중치가 있는(weighted) 아이템셋을 다루는 데 효과적입니다.

- **Technical Details**: 제안된 알고리즘은 다양하고 복잡한 패턴 타입, 예를 들어 순차적, 가중치 있는, 그리고 가중치 없는 아이템셋을 처리할 수 있으며, temporal bias를 관리하기 위한 일반적인 솔루션을 제공합니다. 이를 통해 패턴 샘플링 작업에서 발생하는 long-tail 문제를 해결할 수 있는 방안이 제시됩니다.

- **Performance Highlights**: 실제 데이터셋을 기반으로 한 실험을 통해 제안된 접근법이 정확한 온라인 분류기를 생성할 수 있음을 입증하였으며, 이는 오프라인 기준과 비슷한 정확도를 달성하는 데 성공하였습니다. 이 연구는 순차적 데이터 분류에서 온라인 기계 학습 모델의 활용 가능성을 크게 확장하는 성과를 보여줍니다.



### RSL-SQL: Robust Schema Linking in Text-to-SQL Generation (https://arxiv.org/abs/2411.00073)
- **What's New**: 이 논문에서는 Text-to-SQL 생성의 새로운 프레임워크인 RSL-SQL을 제안합니다. 이 프레임워크는 양방향 스키마 연결(bidirectional schema linking), 맥락 정보 보강(contextual information augmentation), 이진 선택 전략(binary selection strategy), 다회전 자기 수정(multi-turn self-correction)을 결합하여 성능을 향상시킵니다.

- **Technical Details**: RSL-SQL는 먼저 완전한 데이터베이스 스키마를 사용해 초기 SQL을 생성하고, 양방향 스키마 연결을 통해 높은 리콜(recall) 비율을 달성합니다. 그 다음, 데이터베이스 스키마를 단순화한 후 풍부한 맥락 정보로 강화하여 또 다른 SQL을 생성합니다. 이어서 이진 선택 전략을 통해 완전 스키마와 단순화된 스키마 중 더 나은 SQL을 선택합니다. 마지막으로 SQL 실행 결과의 피드백을 통합하여 잘못된 SQL 문을 반복적으로 수정합니다.

- **Performance Highlights**: BIRD와 Spider 데이터셋을 대상으로 한 실험에서 RSL-SQL은 각각 67.2% 및 87.9%의 실행 정확도를 기록하며, 기존 오픈소스 솔루션 중에서 최상의 성능을 달성했습니다. 또한 DeepSeek를 이용할 경우, RSL-SQL이 많은 GPT-4 기반 방법들보다 더 탁월한 성능을 보이면서 비용 효율성을 입증했습니다.



### Meta-Sealing: A Revolutionizing Integrity Assurance Protocol for Transparent, Tamper-Proof, and Trustworthy AI System (https://arxiv.org/abs/2411.00069)
Comments:
          24 pages, 3 figures and 10 Code blocks, to be presented in the conference

- **What's New**: 이번 연구는 인공지능(AI) 시스템의 신뢰성을 보장하기 위한 메타 실링(Meta-Sealing)이라는 새로운 암호화 프레임워크를 소개합니다. 이 프레임워크는 AI 시스템의 운영 전반에 걸쳐 무결성 검증을 근본적으로 변화시킵니다.

- **Technical Details**: 메타 실링은 암호화 봉인 체인(cryptographic seal chains)을 활용하여 시스템 결정 및 변화에 대한 검증 가능한 불변의 기록을 구축합니다. 이 프레임워크는 고급 암호화 기술과 분산 검증(distributed verification)을 결합하여 변조 방지(tamper-evident) 보장을 제공합니다.

- **Performance Highlights**: 재무 기관 데이터에 대한 테스트 결과, 메타 실링은 감사 시간(audit timeframes)을 62% 단축시켰으며, 이해관계자 신뢰도는 47% 향상되었습니다. 이러한 결과는 기업 AI 배포의 무결성 보장을 위한 새로운 기준을 설정할 수 있습니다.



### Interpretable Language Modeling via Induction-head Ngram Models (https://arxiv.org/abs/2411.00066)
- **What's New**: 이 논문에서는 Induction-head ngram 모델(Induction-Gram)을 제안하여 고비용의 계산 환경에서도 효율적이고 해석 가능한 언어 모델을 구축합니다. 이를 통해 각 생성된 토큰에 대한 ngram 수준의 기반을 제공하며, 기존 모델보다 다음 단어 예측 성능을 크게 향상시킵니다.

- **Technical Details**: Induction-Gram은 현대 ngram 모델에 'induction head'라는 손수 설계한 요소를 추가하여 구현됩니다. 이 induction head는 사용자 지정된 신경 유사성 메트릭을 사용하여 모델 입력 컨텍스트에서 다음 단어 완료를 위한 잠재적인 제안을 효율적으로 검색합니다. 이 방법은 fMRI 반응 예측과 같은 자연어 뇌과학 설정에서도 차별화된 성능을 보입니다.

- **Performance Highlights**: Induction-Gram은 기준 해석 가능 모델에 비해 최대 26%p의 향상된 다음 단어 예측 성능을 보여주며, 자연어 fMRI 응답 예측에서도 20% 상대적 향상을 이끌어냅니다. 이를 통해 LLM의 추론 과정을 가속화할 수 있는 가능성을 엿봅니다.



### The ISCSLP 2024 Conversational Voice Clone (CoVoC) Challenge: Tasks, Results and Findings (https://arxiv.org/abs/2411.00064)
Comments:
          accepted by ISCSLP 2024

- **What's New**: ISCSLP 2024 Conversational Voice Clone Challenge (CoVoC)가 제안됨. 이 도전은 제로샷(Zero-shot) 대화형 음성 합성에 중점을 두고 있으며, 대화 음성을 생성하는 기술을 발전시키고 밴치마크(Benchmark)하는 것을 목표로 함.

- **Technical Details**: CoVoC는 두 가지 트랙으로 나뉘어 있으며, 제한된 데이터와 모델, 그리고 제한이 없는 데이터 및 모델 사용에 따라 평가됨. 100시간 고품질 대화 음성 데이터셋이 제공됨. 참여 팀은 특정 텍스트 유형에 대한 음성을 합성해야 하며, 평가 지표로는 Character Error Rate (CER)와 코사인 유사도(SIM) 사용. 주관적 평가는 Mean Opinion Score (MOS)에 의해 수행됨.

- **Performance Highlights**: 총 11개의 팀이 경쟁에 참여하였으며, 제약 트랙에서 5팀, 자유 트랙에서 7팀이 제출함. 대부분의 팀은 다음 두 가지 시스템 구조를 사용하여 결과를 제출: autoregressive 모델과 non-autoregressive 모델. 결과는 주관적 평가와 객관적 평가를 모두 포함하여, 자연스러운 발화 스타일과 명확한 발음이 주요한 평가 포인트로 나타남.



### Evolving Alignment via Asymmetric Self-Play (https://arxiv.org/abs/2411.00062)
- **What's New**: 본 논문에서는 기존의 RLHF(frameworks for aligning large language models) 프레임워크의 한계를 극복하기 위해, 비대칭 게임(asymmetric game)으로 양자 간의 상호작용을 통해 프롬프트 분포(prompt distribution)를 점진적으로 발전시키는 새로운 RLHF 프레임워크인 eva(Evolving Alignment via Asymmetric Self-Play)를 제안합니다.

- **Technical Details**: eva는 두 플레이어(creator와 solver)가 상호작용하는 구조로, creator는 보상 모델(reward model)을 사용하여 더 정보가 많은 프롬프트를 생성하고, solver는 creator가 생성한 프롬프트에 대해 더 선호되는 응답(responses)을 생성하는 방식으로 진행됩니다. 이 구조는 기존의 RLHF 알고리즘을 효과적으로 활용할 수 있도록 설계되었습니다.

- **Performance Highlights**: eva는 Arena-Hard 벤치마크에서 Gemma-2-9B-it의 승률을 DPO로 51.6%에서 60.1%로, SPPO로 55.7%에서 58.9%로, SimPO로 52.3%에서 60.7%로, ORPO로 54.8%에서 60.3%로 향상시키는 등, 최신 기술과 비교할 때도 우수한 성능을 보였습니다.



### Generating Diverse Negations from Affirmative Sentences (https://arxiv.org/abs/2411.00056)
Comments:
          Accepted at "Adaptive Foundation Models: Evolving AI for Personalized and Efficient Learning" workshop at NeurIPS 2024

- **What's New**: NegVerse라는 새로운 방법을 제안하여 부정(Negation) 데이터 부족 문제를 해결하고 다양한 부정 유형을 생성합니다. 이 방법은 긍정적인 문장에서 부정을 생성하며, 구문 구조에 따라 부정이 가장 잘 발생할 부분을 마스킹하는 새로운 규칙을 제공합니다.

- **Technical Details**: 이 연구는 긍정적 문장으로부터 부정을 생성하는 NegVerse 방법을 도입합니다. 이 과정에서는 대체 문장 생성을 위해 GPT-2 기반의 모델을 활용하며, 문장을 유창하게 유지하도록 선택형 마스킹 전략을 사용합니다. 또한, 부정 신호를 판별하고 부적절한 예시를 제거하는 필터링 메커니즘을 제안합니다.

- **Performance Highlights**: 실험 결과, NegVerse는 기존 방법들보다 개선된 성능을 보이며 생성된 부정 문장이 원래 문장과 높은 어휘적 유사성을 유지하고, 구문적 보존 및 부정 다양성 측면에서 우수한 결과를 보여줍니다.



### eDOC: Explainable Decoding Out-of-domain Cell Types with Evidential Learning (https://arxiv.org/abs/2411.00054)
Comments:
          under review

- **What's New**: 이 논문에서는 eDOC라는 새로운 방법을 개발하여 단일 세포 RNA 시퀀싱(scRNA-seq) 데이터에서 Cell Type Annotation (CTA) 문제를 해결합니다. 이 방법은 Transformer 아키텍처를 활용하여 In-Domain(IND) 및 Out-of-Domain(OOD) 세포 유형을 주석 처리하고, OOD 세포와 IND 세포에 기여하는 유전자들을 강조합니다.

- **Technical Details**: eDOC는 evidential learning을 포함하여 세포 유형 주석을 개선하고 OOD 세포에 대한 신뢰성을 정량화합니다. 이 방법은 일반적인 supervised training 없이도 OOD 세포 유형을 탐지할 수 있으며, marker genes를 통해 OOD 세포를 해석할 수 있는 특징을 제공합니다.

- **Performance Highlights**: eDOC는 OOD 세포 유형 및 유전자 드라이버 식별의 효율과 효과를 기존의 최신 방법들과 비교하여 유의미하게 개선한 결과를 보여줍니다. 이 연구는 단일 세포 생물학에 대한 새로운 통찰력을 제공할 수 있음을 암시합니다.



### ACC-Debate: An Actor-Critic Approach to Multi-Agent Deba (https://arxiv.org/abs/2411.00053)
- **What's New**: 이번 논문에서는 액터-비평가 기반의 학습 프레임워크인 ACC-Debate를 제안하여 두 에이전트 팀이 반복적인 대화를 통해 문제를 협력적으로 해결할 수 있도록 훈련하는 새로운 패러다임을 소개합니다.

- **Technical Details**: ACC-Debate는 두 개의 에이전트로 구성되어 있으며, 하나는 답변을 제공하는 액터(agent)이고, 다른 하나는 피드백을 제공하는 비평가(critic)입니다. 이 프레임워크는 ‘유도 대화(guided-debate)’라는 새로운 오프-정책 학습 방식을 도입하여, 고품질의 다중턴 훈련 데이터를 생성하고, 액터와 비평가의 성능을 향상시킵니다.

- **Performance Highlights**: ACC-Debate는 기존의 최첨단(debate techniques) 기술들보다 다양한 벤치마크에서 성능이 우수하다는 것을 입증하였습니다.



### Larger models yield better results? Streamlined severity classification of ADHD-related concerns using BERT-based knowledge distillation (https://arxiv.org/abs/2411.00052)
Comments:
          20 figures, 31 pages, review 1 from plos one journal

- **What's New**: 이번 연구는 지식 증류(knowledge distillation) 방법을 통해 경량화되었지만 성능이 뛰어난 BERT 기반 모델인 LastBERT를 개발하였습니다. 또한, 소셜 미디어 텍스트 데이터에서 주의력 결핍 과다행동 장애(ADHD) 관련 우려 사항의 심각도 수준을 분류하는 실제 세계 작업에 모델을 적용하였습니다.

- **Technical Details**: LastBERT 모델은 기본 BERT 모델의 매개변수를 1억 1천만에서 2천9백만으로 줄여 약 73.64% 더 작은 모델입니다. GLUE 벤치마크에서 LastBERT는 다양한 자연어 처리(NLP) 작업에서 뛰어난 성능을 유지하였고, ADHD 데이터셋에서 85%의 정확도 및 F1 점수를 달성하였습니다.

- **Performance Highlights**: LastBERT는 DistilBERT(6천6백만 매개변수) 및 ClinicalBERT(1억 1천만 매개변수)와 비교해 유사한 성능을 보였으나, DistilBERT가 87%로 약간 더 나은 성능을 기록하였습니다. 본 연구 결과는 LastBERT가 소셜 미디어에서 생성된 사용자의 콘텐츠를 이해하고 평가하는 데 유용한 도구로 작용할 수 있음을 보여줍니다.



### Rule by Rule: Learning with Confidence through Vocabulary Expansion (https://arxiv.org/abs/2411.00049)
Comments:
          29 pages, 8 figures

- **What's New**: 이 논문에서는 텍스트 기반 데이터에 특화된 혁신적인 iterative 접근 방식을 통해 규칙 학습을 제안합니다. 각 반복에서 사용하는 어휘를 점진적으로 확장하여 메모리 소비를 크게 줄이는 방법을 소개합니다. 또한, 생성된 규칙의 신뢰성을 나타내는 신뢰 수준(Value of Confidence)을 도입하여 가장 강력하고 신뢰할 수 있는 규칙만을 유지함으로써 규칙 학습 과정의 전반적인 품질을 향상시킵니다.

- **Technical Details**: 이 방법은 FOIL과 RIPPER와 같은 기존의 규칙 학습 알고리즘을 사용하며, 단순한 사전(dictionary)에서 시작하여 점진적으로 어휘를 확장함으로써 복잡한 텍스트 데이터를 처리할 수 있게 합니다. 초기에는 적은 수의 예를 고려하여 일반적인 규칙을 학습하고, 규칙이 학습될 때마다 긍정적인 예시는 감소하며, 이후 품질 기준에 따라 어휘를 확장합니다.

- **Performance Highlights**: 다양한 텍스트 및 비텍스트 데이터 세트에 대한 광범위한 실험을 통해 이 방법의 효과를 입증하였으며, 특히 보험 산업에 대한 사례 연구를 통해 실제 적용 가능성을 시연하였습니다. 결과적으로, 이 접근 방식은 매우 큰 데이터 세트에서도 실행 가능하며, 해석 가능성과 정확성 간의 균형을 유지하는 데 기여합니다.



### CurateGPT: A flexible language-model assisted biocuration too (https://arxiv.org/abs/2411.00046)
- **What's New**: 이 논문은 데이터 기반의 생물 의학 발견을 위한 효과적인 데이터 관리(data curation)의 중요성을 강조하고, 새로운 Generative AI 기술인 CurateGPT를 소개합니다.

- **Technical Details**: CurateGPT는 instruction-tuned large language models (LLMs)의 능력을 활용하여, 전문가들이 수작업으로 수행하던 과정을 자동화하고, 외부 정보 소스와 지식을 통합하는 데 도움을 줍니다. 이 시스템은 reasoning, ontology 검색 및 지식 통합과 같은 작업을 통해 효율성을 극대화합니다.

- **Performance Highlights**: CurateGPT는 LLM과의 직접 상호작용보다 더 나은 정보 접근성을 제공하며, 각 주장을 뒷받침하는 데이터에 직접 연결되는 링크를 제공합니다. 이러한 방식으로 연구자와 엔지니어는 방대한 과학 데이터의 증가 속도에 맞춰 더욱 효율적으로 데이터 관리를 확대할 수 있습니다.



### A Novel Psychometrics-Based Approach to Developing Professional Competency Benchmark for Large Language Models (https://arxiv.org/abs/2411.00045)
Comments:
          36 pages, 2 figures

- **What's New**: 본 논문은 Evidence-centered design (ECD) 방법론을 도입하여 교육 및 교수법 분야에서 새로운 벤치마크를 개발하는 접근법을 제안합니다. 현재의 벤치마크 개발 방법의 한계를 지적하고 LLM(대형 언어 모델)의 발전을 고려합니다.

- **Technical Details**: Bloom's taxonomy에 따라 구성된 새로운 벤치마크는 교육 전문가들이 rigorously 디자인하여 LLM에 맞춘 평가 도구를 제공합니다. 이 벤치마크는 현재 GPT 모델을 러시아어로 테스트하여 다양한 과제 복잡성에 따라 모델 성능을 평가합니다.

- **Performance Highlights**: 결과적으로, 생성 AI 도구는 교육에서 개인화된 튜터링, 실시간 피드백, 다국어 학습 등과 같은 과제를 지원할 수 있는 잠재력을 가지고 있지만, 깊은 인지적 참여가 필요한 과제와 같은 다양한 분야에서 자율적으로 교사를 보조하는 데에는 한계가 있음을 보여줍니다.



### NeuroSym-BioCAT: Leveraging Neuro-Symbolic Methods for Biomedical Scholarly Document Categorization and Question Answering (https://arxiv.org/abs/2411.00041)
- **What's New**: 본 연구에서는 OVB-LDA라는 최적화된 주제 모델링 프레임워크와 BI-POP CMA-ES 최적화 기법을 통합한 새로운 접근 방식을 제안하여 생물 의학 논문 초록 분류의 정확성을 향상시키고 있습니다.

- **Technical Details**: 이 연구는 세 가지 구성으로 평가되며, 각 구성은 학술 문서 초록 검색, 금관 표준 학술 문서 초록, 그리고 금관 조각을 포함합니다. MiniLM 모델을 정상적으로 사용하여 주제 모델링과 고급 기계 학습 기술을 결합한 신경 상징적(answer extraction) 접근 방식을 활용하여 데이터를 정교하게 조정하면서도 높은 정확도로回答(answer extraction)를 수행할 수 있습니다.

- **Performance Highlights**: 기존의 RYGH 및 bio-answer finder와 같은 방법들을 뛰어넘는 성능을 보여주며, MiniLM이 작은 모델임에도 불구하고 복잡한 작업을 처리할 수 있는 경쟁력 있는 성능을 나타냅니다. 이 연구의 결과는 생물 의학 질문 응답 분야에서 초록의 유용성을 강조하며, 효율성과 정확성을 개선하기 위한 향후 연구 방향을 제시합니다.



### P$^2$C$^2$Net: PDE-Preserved Coarse Correction Network for efficient prediction of spatiotemporal dynamics (https://arxiv.org/abs/2411.00040)
- **What's New**: 이번 논문에서는 고해상도 메쉬 그리드 없이도 부분 미분 방정식(PDE) 문제를 효율적으로 해결할 수 있는 새로운 모델, P$^2$C$^2$Net을 소개합니다. 이 모델은 적은 훈련 데이터 환경에서도 성능을 발휘하도록 설계되었습니다.

- **Technical Details**: P$^2$C$^2$Net은 두 가지 주요 모듈로 구성되어 있습니다: (1) 경량의 PDE 블록이 고차수 수치적 방법을 기반으로 coarse solution(거친 해)을 업데이트하고, (2) 신경망 블록이 실시간으로 해결책을 교정합니다. 구조적이며 대칭적인 Conv 필터를 통해 공간 미분을 보다 정확하게 추정할 수 있습니다. 이 모델은 4차 Runge-Kutta(RK4) 방법을 사용하여 시스템 상태의 시간적 변화를 처리합니다.

- **Performance Highlights**: P$^2$C$^2$Net은 복잡한 반응-확산 과정과 난류 흐름을 포함하는 네 가지 데이터셋에서 50% 이상의 개선된 상대 예측 오류로 일관된 최첨단 성능을 달성하였습니다. 또한, 제한된 훈련 데이터로도 높은 정확도를 유지하고 있습니다.



### Linear Chain Transformation: Expanding Optimization Dynamics for Fine-Tuning Large Language Models (https://arxiv.org/abs/2411.00039)
Comments:
          9 pages, 2 figures, 4 tables

- **What's New**: 본 연구에서는 Linear Chain Transformation (LinChain)이라는 새로운 접근 방식을 제안하여, 사전 훈련된 대형 언어 모델(LLMs)을 특정 다운스트림 작업에 잘 적응하게 만들기 위해 최적화 동역학을 풍부하게 하는 일련의 선형 변환을 도입했습니다.

- **Technical Details**: LinChain은 파라미터 업데이트 과정에 여러 개의 선형 변환을 통합하여 업데이트의 효과적인 랭크를 확장하고, 복잡한 작업 별 표현을 학습하는 모델의 능력을 향상시킵니다. 이 방법은 고정된 저랭크(LoRA) 접근법의 한계를 극복하면서도 효율성을 유지합니다.

- **Performance Highlights**: LinChain은 다양한 벤치마크 작업에서 최첨단 방법에 비해 성능을 크게 향상시키며, 더 적은 학습 가능한 파라미터로 더 나은 일반화와 작업 적응을 유도하였습니다.



### Topic-Conversation Relevance (TCR) Dataset and Benchmarks (https://arxiv.org/abs/2411.00038)
Comments:
          To be published in 38th Conference on Neural Information Processing Systems (NeurIPS 2024) Track on Datasets and Benchmarks

- **What's New**: 이번 연구는 효과적인 회의 운영을 위해 대화를 주제에 맞추는 중요성을 강조하며, 1,500개의 회의와 2,200만 개 단어의 전사문이 포함된 포괄적인 Topic-Conversation Relevance (TCR) 데이터셋을 생성했습니다.

- **Technical Details**: TCR 데이터셋은 15,000개 이상의 회의 주제를 포함하고 있으며, GPT-4를 사용하여 긴 회의록을 사전 회의 아젠다 주제 스타일로 재작성합니다. 또한 다양한 회의 변형을 만들 수 있는 확장 가능한 스키마를 제공합니다.

- **Performance Highlights**: GPT-4를 활용하여 주제-대화 관련성을 이해하는 데 있어 모델의 정확도를 평가했습니다. 결과는 회의의 효과성을 높이는 데 기여할 수 있는 인사이트를 제공합니다.



### Coupling quantum-like cognition with the neuronal networks within generalized probability theory (https://arxiv.org/abs/2411.00036)
Comments:
          RIKEN Quantum Workshop, October 11, 2024

- **What's New**: 최근 몇 년 동안의 연구는 인간의 인지 및 심리학 모델링에서 양자 이론(quantum theory) 및 양자 유사 모델링(quantum-like modeling)의 응용이 증가하고 있으며, 이로 인해 신경 생리학적(neurophysiological) 과정과의 명확한 연결 고리가 부족하다는 문제를 다루고 있습니다.

- **Technical Details**: 본 연구는 신경 세포(neurons)의 통신 네트워크를 양자 유사적으로 표현하는 모델을 제안합니다. 이 모델은 표준 양자 이론 대신 일반화된 확률 이론(generalized probability theory, GPT)을 기반으로 하며, 측정 도구 이론(measurement instruments theory) 내에서 효과 관찰(effect-observables) 및 상태 업데이트(state updates)를 포함합니다. 통신 신경망은 가중치 그래프(weighted graph)로 설명되며, 가중치 행렬(weight matrix)을 통해 인코딩됩니다.

- **Performance Highlights**: GPT 기반 모델은 순서(order), 비반복성(non-repeatability), 및 불확정성(Disjunction) 효과와 같은 기본적인 양자 유사 효과를 보여주며, 이는 우울증(depression) 및 간질(epilepsy)과 같은 신경 질환에 대한 의료 진단에도 활용될 수 있습니다.



### Is Our Chatbot Telling Lies? Assessing Correctness of an LLM-based Dutch Support Chatbo (https://arxiv.org/abs/2411.00034)
Comments:
          10 pages + 2 pages references, 4 figures

- **What's New**: AFAS는 고객 지원 팀의 개입 최소화 및 스스로 고객 문의에 대한 정확한 답변을 제공할 수 있는 AI 기반 챗봇을 개발하기 위해 노력하고 있습니다. 본 연구는 정확성(정확한 답변이란 무엇인지)을 정의하고, 이를 바탕으로 LLM 사용을 통한 고객 지원의 자동화를 목표로 하고 있습니다.

- **Technical Details**: 연구는 자연어 생성(Natural Language Generation) 및 자동 답변 평가 시스템(Automated Answer Grading)을 활용하여 AFAS 지원 팀이 의사 결정을 하는 방식을 모델링합니다. 연구 과정에서 수집된 데이터는 79개의 훈련 사례와 154개의 테스트 사례로 구성되어 있으며, 응답의 진실성을 평가하기 위한 휴리스틱(heuristics)과 맞춤형 메트릭(metrics)을 도출합니다.

- **Performance Highlights**: 제안된 모델은 55%의 정확도로 잘못된 응답을 식별할 수 있으며, 터키어와 영어의 번역된 텍스트에서 인간 평가와 비교했을 때 높은 상관 관계를 보였습니다. 이는 AFAS의 고객 서비스 품질 개선에 기여할 수 있는 가능성을 보여줍니다.



### A Theoretical Review on Solving Algebra Problems (https://arxiv.org/abs/2411.00031)
Comments:
          22pages,5figures

- **What's New**: 이 논문은 대수 문제(Algebra Problems, APs) 해결을 위한 새로운 리뷰 프레임워크를 개발하여 이론적 기반을 마련하고 평가 체계를 창출하며 연구의 범위를 확장하는 데 중점을 두고 있습니다.

- **Technical Details**: 논문에서 제안하는 State Transform Theory (STT)는 문제 해결 알고리즘이 상태(states)와 변환(transforms)에 따라 구조화되어 있음을 강조합니다. 이는 전통적인 설문조사가 단순히 변환의 진행(progress)을 강조하는 것과는 다른 접근 방식입니다. 새로운 프레임워크는 단어 및 도식적 대수 문제 해결을 위한 관계 중심 알고리즘을 수용합니다.

- **Performance Highlights**: 이 논문은 새로운 상태를 도입하는 필요성을 강조하며, 이전 리뷰에서는 간과된 개별 알고리즘의 기여를 드러내는 데 도움을 줍니다.



### WikiNER-fr-gold: A Gold-Standard NER Corpus (https://arxiv.org/abs/2411.00030)
- **What's New**: 본 논문에서는 다국어 Named Entity Recognition (NER) 코퍼스인 WikiNER의 품질을 논의하고, 개선된 버전인 WikiNER-fr-gold를 제공합니다. 이는 프랑스어 부분의 수정된 버전으로, 원본 프랑스어 하위 코퍼스의 20%를 무작위 샘플링하여 생성되었습니다.

- **Technical Details**: WikiNER는 반감독(semi-supervised) 방식으로 주석(annotation)이 생성되었으며, 사후 수동 검증이 이루어지지 않아 'silver-standard'(실버 스탠다드) 코퍼스로 분류됩니다. WikiNER-fr-gold는 26,818 문장과 700,000 토큰을 포함하는 원본 하위 코퍼스에서 선택된 무작위 샘플로 구성됩니다. 본 논문에서는 각 범주에 포함된 엔티티 유형을 요약하여 주석 가이드를 정의하고, 코퍼스를 수정하는 과정을 설명합니다.

- **Performance Highlights**: WikiNER-fr 코퍼스에서 관찰된 오류 및 일관성 부족에 대한 분석을 제공하며, 향후 연구 방향에 대한 논의도 포함됩니다. 이러한 분석을 통해 NER 시스템의 품질 향상 가능성을 제시합니다.



### Preserving Pre-trained Representation Space: On Effectiveness of Prefix-tuning for Large Multi-modal Models (https://arxiv.org/abs/2411.00029)
Comments:
          Findings of EMNLP 2024

- **What's New**: 최근 대규모 다중 모달 모델(Large Multi-modal Models, LMMs)의 발전을 통해 기계와 세계 간의 상호작용 방식이 혁신적으로 변화하고 있습니다. 이러한 모델을 하위 작업에 적합하도록 조정하기 위해 파라미터 효율적인 미세 조정(Parameter-efficient fine-tuning, PEFT) 기술이 인기를 얻고 있으며, 본 논문에서는 PEFT의 작동 방식에 대한 심층 분석을 제공합니다.

- **Technical Details**: 이 연구에서는 두 단계로 구성된 PT-PEFT(Prefix-Tuned PEFT) 방법을 제안합니다. PT-PEFT는 먼저 prefix-tuning을 수행한 후, 이후에 PEFT 방법(예: Adapter, LoRA)을 통해 모델 파라미터를 조정합니다. 이를 통해 사전 훈련된 지식의 활용을 극대화합니다. 특히 본 연구에서는 singular value decomposition (SVD)을 사용하여 feature representation matrices의 변화를 분석했습니다.

- **Performance Highlights**: PT-PEFT는 이미지 캡셔닝(Image Captioning, IC) 및 시각적 질문 응답(Visual Question Answering, VQA) 작업에서 기존 PEFT 방법에 비해 성능을 개선하는 것으로 나타났습니다. 본 논문에서 관련된 네 가지 사전 훈련 모델을 대상으로 실험한 결과, PT-PEFT가 representation space를 보존하면서 전반적인 성능을 향상시키는 데 기여함을 확인하였습니다.



### Synergizing LLM Agents and Knowledge Graph for Socioeconomic Prediction in LBSN (https://arxiv.org/abs/2411.00028)
- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)과 지식 그래프(Knowledge Graph)를 결합해 위치 기반 소셜 네트워크(LBSN) 데이터를 활용한 사회경제 예측(socioeconomic prediction)을 수행할 수 있는 새로운 접근 방식을 제안합니다. 이러한 접근법은 서로 다른 Task 간의 지식 공유를 통해 예측 성능을 향상시킵니다.

- **Technical Details**: 연구자는 위치 기반 지식 그래프(LBKG)를 구축하여 다양한 출처의 LBSN 데이터를 통합하고, LLM 에이전트를 활용해 각 사회경제 예측 태스크에 적합한 메타 경로(meta-path)를 자동으로 추출합니다. 이 과정에서 의미론적 안내(attention)가 포함된 융합 모듈을 설계하여 메타 경로를 기반으로 한 지식을 융합합니다. 또한 교차 태스크 통신 메커니즘을 도입하여 태스크 간 지식을 공유하고, LLM 에이전트와 KG 수준에서 성능을 개선합니다.

- **Performance Highlights**: 두 개의 도시 규모 데이터셋에서 실험을 진행한 결과, 제안된 모델은 기존 방법보다 2.9%에서 74.2%까지 R^2 성능이 향상되는 것을 확인했습니다. 이는 LLM과 KG 간의 협력적 모델 설계의 효과성을 나타냅니다.



### A Perspective for Adapting Generalist AI to Specialized Medical AI Applications and Their Challenges (https://arxiv.org/abs/2411.00024)
- **What's New**: 의료 애플리케이션에 LLM(대규모 언어 모델)의 통합이 활발해지고 있으며, 그 중에서도 약물 발견, 임상 의사결정 지원 및 원격의료 지원에 대한 최신 동향이 소개되었습니다.

- **Technical Details**: 의료 LLM 연구 활동을 위한 3단계 프레임워크가 제안되었습니다: 1) Modeling: 복잡한 의료 작업을 관리 가능한 단계로 나누어 의학 전문 모델 개발; 2) Optimization: 맞춤형 프롬프트를 통한 모델 성능 최적화 및 외부 지식 통합; 3) System engineering: 복잡한 작업을 하위 작업으로 나누고 인간 전문 지식 활용.

- **Performance Highlights**: LLM 기반 의료 AI 애플리케이션, 임상 시험 설계 최적화, 임상 의사결정 지원 강화 및 의료 영상 분석 향상과 같은 다양한 사용 사례를 통해 LLM의 성능이 입증되었습니다.



### Device-Directed Speech Detection for Follow-up Conversations Using Large Language Models (https://arxiv.org/abs/2411.00023)
- **What's New**: 이 연구에서는 Virtual Assistant (VA)와의 대화에서 후속 쿼리의 정확한 Device-directed Speech Detection (DDSD)에 대한 필요성을 강조합니다. 특히, 기존의 단일 쿼리 감지 접근법 대신, Large Language Models (LLMs)를 활용하여 후속 쿼리와 초기 쿼리의 문맥을 결합하여 정확도를 높이는 방법을 제안합니다.

- **Technical Details**: 기존 DDSD 시스템의 한계를 보완하기 위해, 초점은 두 가지 접근법에 있습니다: (i) 프롬프트 방식으로 미리 훈련된 LLM을 텍스트 기반으로 직접 사용하는 방법, (ii) LLM의 위에 이진 분류기를 추가하여 확률적 결정을 내리는 방법입니다. ASR (Automatic Speech Recognition) 불확실성을 활용하여 쿼리 문맥을 강화하고, n-best ASR 히포세시스를 통해 활용하여 더 많은 정보를 제공합니다.

- **Performance Highlights**: 실험 결과, 공동 모델링을 통해 DDSD 정확도가 약 20-40% 향상된 것을 보여줍니다. 전통적인 방식 대비 후속 쿼리만을 사용했을 때보다 훨씬 높은 성능을 나타냅니다. 이 연구는 실제 데이터 세트를 통해 그 효과를 입증하였습니다.



### Personality-Guided Code Generation Using Large Language Models (https://arxiv.org/abs/2411.00006)
- **What's New**: 본 연구에서는 대형 언어 모델(LLMs)을 사용하여 코드 생성을 할 때, 프로그래밍 작업에 적합한 성격 특성을 구현함으로써 성능을 향상시킬 수 있음을 보여주고 있습니다. 성격 유도 코드 생성이 코드 생성 정확도를 크게 향상시키고, 다양한 모델 및 데이터셋 조합에서의 향상도를 제시하고 있습니다.

- **Technical Details**: 성격 유도 코드 생성을 위해 우리는 Myers-Briggs Type Indicator (MBTI) 프레임워크를 사용하여 특정 코딩 작업에 적합한 프로그래머 성격을 생성합니다. 그런 다음, 이 성격을 바탕으로 여러 LLM을 활용하여 코드를 생성합니다. 연구에 사용된 LLM은 OpenAI, Meta, Alibaba 및 DeepSeek 등에서 개발된 7종이며, 4개의 널리 알려진 데이터셋을 통해 평가했습니다.

- **Performance Highlights**: 연구 결과에 따르면, 성격 유도가 적용된 코드 생성에서 23개의 LLM-데이터셋 조합 중 23개에서 패스율이 향상되었으며, 11개 경우에서 향상이 5%를 초과하고, 5개 경우에서 10%를 초과하는 등의 성과를 보였습니다. 최고 향상률은 12.9%에 달하며, Chain of Thought와 같은 다른 프롬프트 전략과 통합 시 13.8%의 추가 향상도 관찰되었습니다.



### Mastering the Craft of Data Synthesis for CodeLLMs (https://arxiv.org/abs/2411.00005)
- **What's New**: 이 논문은 코드 관련 작업에서 LLM(대규모 언어 모델)의 데이터 합성 및 필터링 기술을 정밀하게 조사하고 최근 발전된 사항을 강조합니다. 또한, 코드LLM의 성능 향상을 위한 데이터 엔지니어링 관행에 관한 실용적인 지침을 제공합니다.

- **Technical Details**: 코드 지능은 기계 학습(Machine Learning) 기술을 활용하여 소프트웨어 개발을 개선하며, LLM으로 코드 완성, 번역, 수정 및 문서화 등의 자동화를 포함합니다. 이 과정에서 데이터 수집, 합성, 필터링 및 평가라는 4단계의 데이터 관리 파이프라인이 수행됩니다. 코드 관련 작업의 성능 향상을 위해 다양한 고품질 데이터 세트가 필수적이며, LLM을 통해 생성된 합성 데이터가 이러한 요구를 충족합니다.

- **Performance Highlights**: LLM의 발전은 머신 러닝 및 코드LLM의 모델 강화를 위한 고품질 데이터의 중요성을 강조하고 있습니다. 이 연구에서는 최근 2년간의 50개 이상의 연구를 분석하여 코드LLM 구축에 필요한 데이터 합성 및 필터링 기술의 구조화된 개요를 제공하고 있으며, 이는 학계 및 산업계의 혁신을 촉진하는 데 기여할 것입니다.



### SFM-Protein: Integrative Co-evolutionary Pre-training for Advanced Protein Sequence Representation (https://arxiv.org/abs/2410.24022)
- **What's New**: 본 연구에서는 아미노산 잔기 간의 상호작용을 강조하는 새로운 단백질 기초 모델의 사전 훈련 전략을 제안합니다. 이를 통해 단백질 서열 데이터로부터 단기 및 장기 공진화(co-evolutionary) 특징을 더욱 효과적으로 추출할 수 있습니다.

- **Technical Details**: 제안된 모델은 대규모 단백질 서열 데이터셋에서 훈련되었으며, 복잡한 상호작용을 반영하는 통합 손실 함수(integrative loss function)를 사용하여 단기 및 장기 공진화 정보를 효과적으로 포착합니다. 이러한 접근법은 진화 정보를 보다 잘 모델링하여 단일 서열 모델과 MSA 기반 방법의 성능 차이를 줄이고자 합니다.

- **Performance Highlights**: 우리의 모델은 다양한 하위 작업에서 기존 모델들보다 우수한 성능을 보였으며, 특히 ESM 모델 같은 유사한 크기의 기준 모델들과 비교하여 뛰어난 일반화 능력을 입증했습니다. 실험 결과를 통해 공진화 정보 통합의 효과성을 확인했습니다.



### Pistis-RAG: Enhancing Retrieval-Augmented Generation with Human Feedback (https://arxiv.org/abs/2407.00072)
- **What's New**: 이 논문에서는 Pistis-RAG라는 새로운 RAG 프레임워크를 소개하며, 이는 LLM의 출력과 인간의 선호를 구조화된 피드백을 통해 조정합니다. 기존의 RAG 시스템이 직면한 한계를 극복하기 위해 콘텐츠 중심의 접근 방식을 채택하였습니다.

- **Technical Details**: Pistis-RAG는 두 가지 주요 단계인 피드백 정렬(feedback alignment)과 온라인 쿼리(online querying)를 통해 운영됩니다. 피드백 정렬 단계에서는 전체 응답 목록에 대한 인간 피드백을 활용하여 순위 모델의 민감도를 향상시킵니다. 온라인 쿼리 단계에서는 정제된 순위 모델에 따라 검색된 콘텐츠의 순서를 재조정합니다.

- **Performance Highlights**: 실험 결과, Pistis-RAG는 기존의 RAG 시스템 대비 MMLU(영어)에서 6.06% 향상, C-EVAL(중국어)에서 7.08% 향상을 보여 주며, 인간의 선호와의 정렬을 개선하는 데 효과적임을 입증합니다.



### Low-Overhead Channel Estimation via 3D Extrapolation for TDD mmWave Massive MIMO Systems Under High-Mobility Scenarios (https://arxiv.org/abs/2406.08887)
Comments:
          13 pages, 11 figures, 3 tables. This paper has been submitted to IEEE journal for possible publication

- **What's New**: 이 논문에서는 TDD(mmWave) 대규모 MIMO 시스템에서 다운링크 CSI(Channel State Information)를 효과적으로 추정하기 위한 새로운 3D 채널 외삽 프레임워크를 제안합니다. 기존의 높은 이동성을 가진 상황에서 파일럿 오버헤드를 체계적으로 줄이고, 스펙트럼 효율성을 향상시키는 방법을 다룹니다.

- **Technical Details**: 이 프레임워크는 공간, 주파수 및 시간 영역에서 CSI 행렬의 차원을 줄이기 위해 KDD-SFCEN(지식-데이터 기반 공간-주파수 채널 외삽 네트워크)과 TUDCEN(시간 기반 업링크-다운링크 채널 외삽 네트워크)을 구현합니다. KDD-SFCEN은 최소 제곱 추정기를 활용하여 거친 채널 추정치를 얻고, 주파수-공간 영역에서의 파일럿 오버헤드를 줄이기 위한 결합 외삽을 수행합니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 파일럿 훈련 오버헤드를 16배 이상 줄이면서, 높은 이동성 시나리오에서도 스펙트럼 효율성을 크게 향상시킴을 보여주었습니다.



### E(3)-invariant diffusion model for pocket-aware peptide generation (https://arxiv.org/abs/2410.21335)
- **What's New**: 본 연구에서는 컴퓨터 지원 단백질 억제제 발견을 위한 새로운 방법, 즉 de novo pocket-aware peptide structure and sequence generation network를 제안합니다. 기존의 연구들이 단백질 억제제 발견에 비효율적인 접근을 가지고 있었던 반면, 본 연구는 보다 스마트한 접근 방식을 제공합니다.

- **Technical Details**: 제안된 접근 방식은 두 개의 연속적인 diffusion 모델로 구성됩니다: 1) conditional structure diffusion model과 2) conditional sequence diffusion model입니다. 구조 확산 모델은 주어진 포켓 정보를 바탕으로 원하는 펩타이드 구조를 생성하며, 이후 시퀀스 확산 모델이 그에 상응하는 아미노산 시퀀스를 생성합니다. 이 모델은 E(3)-불변 표현 방식을 적용하여 3D 공간 내에서 구조가 변화하지 않도록 합니다.

- **Performance Highlights**: 본 연구의 결과는 제안된 방법이 최신 모델과 비교했을 때 유사한 성능을 보여주었음을 입증하며, 포켓에 민감한 펩타이드 설계 가능성을 강조합니다. 따라서 이 연구는 수용체 특이적인 펩타이드 생성을 이용한 정밀 약물 발견의 새로운 접근 방식을 제공합니다.




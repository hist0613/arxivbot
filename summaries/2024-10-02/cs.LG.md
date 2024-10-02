New uploads on arXiv(cs.CL)

### Addition is All You Need for Energy-efficient Language Models (https://arxiv.org/abs/2410.00907)
- **What's New**: 이 논문에서는 부동 소수점 (floating point) 곱셈을 정수 덧셈으로 근사할 수 있는 새로운 알고리즘 L-Mul을 제안합니다. L-Mul 알고리즘은 낮은 계산 자원으로 높은 정확도의 곱셈을 가능하게 합니다.

- **Technical Details**: L-Mul 알고리즘은 정수 덧셈을 사용하여 부동 소수점 수의 곱셈을 근사하며, 8비트 부동 소수점 곱셈에 비해 계산 자원을 대폭 절감하고 더 높은 정밀도를 제공합니다.

- **Performance Highlights**: 실험 결과, L-Mul은 거의 손실 없이 트랜스포머 기반 모델의 주의 메커니즘에 적용될 수 있으며, 3비트 맨티사를 사용한 L-Mul이 float8_e4m3을 사용할 때와 유사한 정밀도를 얻는다고 보고합니다.



### On the Implications of Verbose LLM Outputs: A Case Study in Translation Evaluation (https://arxiv.org/abs/2410.00863)
- **What's New**: 본 논문은 LLM(대형 언어 모델)의 번역에서 발생하는 장황함(verbosity)이 평가에 미치는 영향을 조사합니다. 여러 LLM 출력에서 장황함의 빈도를 입증한 후, 그것을 유발하는 주요 요인들을 식별합니다.

- **Technical Details**: 기존 평가 프레임워크에서 장황한 LLM 출력이 평가 결과에 미치는 영향을 분석하며, 번역 거부(refusal to translate), 여러 번역 옵션(multiple translation options), 그리고 추가 설명(additional commentary)을 포함한 세 가지 장황 행동의 분류를 수행합니다.

- **Performance Highlights**: LLM의 장황한 출력을 자동 및 인간 평가를 통해 분석한 결과, gpt-4와 Aya23232323가 장황함이 거의 없는 특성을 보이는 반면, Gemini-1.5-Pro는 가장 높은 장황성을 가진 모델로 나타났습니다. 현재의 평가 지표는 장황한 LLM을 불공정하게 평가할 위험이 있으며, 따라서 LLM의 향후 평가에서 이러한 문제를 해결해야 할 필요성이 강조되었습니다.



### Quantifying reliance on external information over parametric knowledge during Retrieval Augmented Generation (RAG) using mechanistic analysis (https://arxiv.org/abs/2410.00857)
Comments:
          Accepted to Blackbox NLP @ EMNLP 2024

- **What's New**: 이 논문은 Retrieval Augmented Generation (RAG) 접근 방식에서 Language Model (LM)이 어떻게 비파라메트릭 메모리인 외부 컨텍스트를 사용하는지를 메커니즘적으로 분석함으로써, LM이 'shortcuts' 효과를 보이고 있다는 점을 강조합니다. 이러한 분석은 RAG가 제공하는 정보를 활용할 때 기존의 모델 프라이어에 의존하는 정도가 최소화된다는 것을 보여줍니다.

- **Technical Details**: Causal Mediation Analysis 및 Attention Contributions와 Knockouts를 이용해 RAG 컨텍스트에서의 LM 행동을 조사했습니다. 특히, LST(Last Subject Token)에서 AIE(Average Indirect Effect)를 측정하여 RAG 컨텍스트를 추가할 경우 파라메트릭 메모리 사용이 줄어든다는 사실을 발견했습니다. 또한 Attention 메커니즘의 기여도를 분석하여 마지막 토큰의 예측에 중요한 역할을 하는 토큰들 사이의 주목을 비교했습니다.

- **Performance Highlights**: LLaMa-2와 Phi-2 모델을 사용한 실험에서, RAG 설정에서 LST의 AIE가 기존 비-RAG 설정에 비해 약 10배부터 35배 감소하는 것을 확인했습니다. 또한 Attention Knockouts를 사용한 결과, LLaMa-2에서 'knocking out' 주의 집중이 예측 확률을 20% 감소시키는 반면, RAG 설정에서는 <5% 감소에 그치는 것을 발견했습니다.



### A generative framework to bridge data-driven models and scientific theories in language neuroscienc (https://arxiv.org/abs/2410.00812)
- **What's New**: 이번 논문에서는 대규모 언어 모델(LLM)을 활용하여 뇌의 언어 선택성을 설명하고 검증하는 'Generative Explanation-Mediated Validation (GEM-V)' 프레임워크를 소개합니다. 이 방법은 개별 두상(declaration)과 주요 관심 영역(region of interest, ROI)에서의 선택성을 해석할 수 있는 설명을 생성합니다.

- **Technical Details**: GEM-V는 뇌의 언어 선택성에 대한 깊이 있는 학습 모델을 해석하기 위해 제작되었습니다. 이 모델은 20시간의 수동 언어 청취 fMRI 데이터를 기반으로 각 두상에 대해 언어 선택성을 예측하는 인코딩 모델을 최적화합니다. 생성된 설명은 후속 실험에서 합성 자극(synthetic stimuli)을 사용하여 검증됩니다. 이 접근법은 뇌의 활성화에서 어떤 언어 자극이 영향을 미치는지 파악하는 데 기여하고 있습니다.

- **Performance Highlights**: GEM-V를 사용하여 17개의 두상에 대한 설명을 생성하고 평가한 결과, 51개 두상 중 41개에서 응답이 증가했으며, 평균적으로 기준선(base line)보다 0.198 표준편차만큼 향상되었습니다. 이는 생성된 설명이 선택된 두상의 뇌 활동을 효과적으로 이끌어낸다는 것을 보여줍니다.



### Decoding Hate: Exploring Language Models' Reactions to Hate Speech (https://arxiv.org/abs/2410.00775)
- **What's New**: 이번 연구는 최첨단 LLM들(대형 언어 모델)이 증오 발언에 어떻게 반응하는지를 분석하였으며, LLM의 안전성과 책임 있는 배포에 대한 심도 있는 논의를 제공합니다.

- **Technical Details**: 연구에서는 LLaMA 2, Vicuna, LLaMA 3, Mistral, GPT-3.5, GPT-4, Gemini Pro 등 7가지 LLM 모델을 사용하여 26,000개의 증오 발언(English hate speech) 문장에 대한 반응을 분석했습니다. 이를 통해 각 모델의 다양한 반응 스펙트럼을 밝혀내고, 증오 발언 생성을 완화하는 전략도 논의합니다.

- **Performance Highlights**: 각 모델은 특정 작업 없이도 콘텐츠를 생성하는 기본 모드에서 반응을 보였으며, 연구 결과 LLM들이 증오 발언에 대해 어떻게 다르게 반응하는지를 파악함으로써, 이를 예방하는 방법이 개선될 수 있는 기회를 제시합니다.



### Thinking Outside of the Differential Privacy Box: A Case Study in Text Privatization with Language Model Prompting (https://arxiv.org/abs/2410.00751)
Comments:
          10 pages, 3 tables, Accepted to EMNLP 2024 (Main)

- **What's New**: 이번 논문은 개인 정보 보호를 위한 자연어 처리(NLP)에서 Differential Privacy (DP) 통합의 제약과 도전 과제를 다룹니다. 특히 DP-Prompt 방법을 통해 텍스트의 비공식화를 연구하며, DP가 제공하는 이점과 비DP 접근 방식에 비해 가지는 한계를 논의합니다.

- **Technical Details**: DP-Prompt는 생성형 Language Models를 활용하여 텍스트를 재작성하는 개인 정보 보호 방법입니다. 이 방법은 Exponential Mechanism을 기반으로 하는 DP 토큰 선택 메커니즘을 통해 작동하며, 여러 환경에서 시도해봤습니다. 논문의 주요 실험은 DP, Quasi-DP, Non-DP로 나뉘며, DP를 활용한 텍스트 재작성의 장점과 단점을 비교 검토하고 있습니다.

- **Performance Highlights**: 실험 결과, DP를 통합한 텍스트 재작성 시스템이 원본 텍스트에 비해 높은 의미적 유사성(s semantic similarity)를 유지하는 반면, 읽기 쉽고 품질 높은 텍스트 생성은 더 어려운 과제로 남아 있습니다. 이러한 결과는 DP와 비DP 텍스트 비공식화 간의 실용적 구분에 대해 논의할 수 있는 기회를 제공합니다.



### Optimizing Token Usage on Large Language Model Conversations Using the Design Structure Matrix (https://arxiv.org/abs/2410.00749)
Comments:
          10 pages, 26th International Dependency and Structure Modelling Conference, DSM 2024

- **What's New**: 본 논문은 Large Language Models (LLMs)의 대화 최적화에 Engineering Design 분야의 Design Structure Matrix (DSM)를 도입합니다. 특히 우주선 및 그 하위 시스템 디자인에 관련된 대화 사례에 적용되었습니다.

- **Technical Details**: DSM의 분석 도구인 클러스터링(clustering)과 시퀀싱(sequencing)을 활용하여 LLM과의 대화에서 전송되는 토큰(token)의 수를 최소화하고, 다양한 컨텍스트 윈도우(context windows)에 할당될 수 있는 청크(chunk)를 그룹화합니다.

- **Performance Highlights**: 이 방법론은 LLM의 토큰 사용 최적화를 위한 현재의 방법론 세트를 확장시키며, LLM에 Engineering Design 관행을 통합할 수 있는 새로운 접근 방식의 길을 열어줍니다.



### VideoCLIP-XL: Advancing Long Description Understanding for Video CLIP Models (https://arxiv.org/abs/2410.00741)
Comments:
          EMNLP 2024 Main conference

- **What's New**: VideoCLIP-XL (eXtra Length) 모델은 비디오 CLIP 모델의 긴 설명 이해 능력을 확장하는 데 초점을 맞추고 있습니다. 이를 위해 자동 데이터 수집 시스템을 구축하여 200만 개 이상의 비디오-긴 설명 쌍으로 구성된 대규모 VILD 사전 훈련 데이터셋을 수집하였습니다.

- **Technical Details**: TPCM (Text-similarity-guided Primary Component Matching)을 도입하여 고차원 특징 공간의 분포 변화에 적응할 수 있도록 하였으며, Detail-aware Description Ranking (DDR)와 Hallucination-aware Description Ranking (HDR)이라는 새로운 작업을 통해 비디오와 긴 설명 간의 관계를 효과적으로 학습하도록 하였습니다.

- **Performance Highlights**: 비디오CLIP-XL 모델은 기존의 여러 벤치마크에서 우수한 성능을 보였으며, 특히 LVDR (Long Video Description Ranking) 벤치마크를 통해 긴 설명에 대한 이해 능력을 종합적으로 평가하였습니다. 실험 결과, 비디오CLIP-XL은 최신 기술 경쟁 모델들보다 뛰어난 성능을 발휘했습니다.



### Efficient Technical Term Translation: A Knowledge Distillation Approach for Parenthetical Terminology Translation (https://arxiv.org/abs/2410.00683)
Comments:
          Paper accepted in EMNLPW 2024

- **What's New**: 이 논문은 기술 용어의 정확한 번역을 통해 전문 분야에서의 명확한 의사소통을 도모하는 새로운 접근법인 Parenthetical Terminology Translation (PTT) 과제를 제시합니다. PTT는 원래 용어를 괄호 안에 담아 번역하는 방법으로, 번역의 정확도를 높이고 독자의 혼란을 줄이는 데 기여합니다.

- **Technical Details**: 이 연구에서는 Large Language Models (LLMs)와 함께 협업하여 PTT 데이터셋을 생성하고, 이를 통해 Neural Machine Translation (NMT) 모델 및 소형 Language Models (sLMs)의 성능을 강화하기 위해 knowledge distillation을 적용했습니다. 새로운 평가 지표도 개발하여 번역의 정확성과 괄호 내 용어의 올바른 표현을 평가합니다.

- **Performance Highlights**: 연구 결과, sLMs 모델이 NMT 모델보다 일관되게 우수한 성능을 보이지는 않았지만, fine-tuning이 few-shot prompting보다 더 효과적이라는 점이 강조되었습니다. 특히 목표 언어에서 지속적인 프리트레이닝이 이루어진 모델에서 그 효과가 두드러졌습니다.



### Detecci\'on Autom\'atica de Patolog\'ias en Notas Cl\'inicas en Espa\~nol Combinando Modelos de Lenguaje y Ontolog\'ias M\'edicos (https://arxiv.org/abs/2410.00616)
Comments:
          22 pages, in Spanish language, 6 figures, Proceedings of the 40th venue of the SEPLN

- **What's New**: 이번 연구에서는 의료 보고서에서 피부 질환을 자동으로 탐지하는 하이브리드 방법을 제안합니다. 대규모 언어 모델과 의료 온톨로지를 결합하여, 처음 진료 또는 후속 진료 보고서를 바탕으로 환자가 가질 수 있는 병리를 예측합니다. 이 모델은 병리의 종류, 중증도, 신체 위치를 학습하는 방식을 통해 정확성을 획기적으로 향상시킵니다.

- **Technical Details**: 이 연구는 Transformer 기반의 모델과 RoBERTa를 활용하며, 다단계(cascade) 모델을 구성하여 증상, 해부학적 위치 및 중증도를 파악한 후, 이를 바탕으로 환자의 병리를 예측합니다. 모든 모델은 스페인 내 여러 의료 기관의 익명화된 전자 건강 기록(EHR) 데이터를 사용하여 학습됩니다.

- **Performance Highlights**: 연구에서 제안한 방법은 의료 텍스트 분류에서 0.84의 정밀도(precision)와 각각 0.82 및 0.75의 마이크로(micro) 및 매크로(macro) F1-스코어를 달성하였습니다. 이 논문은 커뮤니티에 사용된 방법과 데이터셋을 제공하였습니다.



### Style-Specific Neurons for Steering LLMs in Text Style Transfer (https://arxiv.org/abs/2410.00593)
Comments:
          Accepted at EMNLP 2024 main conference. The code is publicly available at this https URL

- **What's New**: 본 연구에서는 sNeuron-TST라는 새로운 접근 방식을 제안하여, 스타일 특화 (style-specific) 뉴런을 활용하여 대규모 언어 모델(LLMs)을 조정하고 텍스트 스타일 전환(Text Style Transfer, TST)을 수행하는 방법을 새롭게 소개합니다.

- **Technical Details**: sNeuron-TST는 LLM 내부의 스타일 특화 뉴런을 식별하고, 소스 스타일 전용 뉴런을 비활성화하여 타겟 스타일의 단어가 생성될 확률을 높이는 방식으로 작동합니다. 이러한 비활성화는 문장의 유창성에 부정적인 영향을 미칠 수 있으므로, 이를 완화하기 위해 빠른 토큰 확률 변화를 고려한 개선된 대조적 (contrastive) 디코딩 방법을 제안합니다.

- **Performance Highlights**: 본 연구의 실험 결과, 제안된 방법은 6가지 기준 벤치마크에서 기존 시스템보다 높은 타겟 스타일 단어 생성을 나타내었으며, 스타일 전환 정확도와 유창성을 모두 개선하면서 원래 텍스트의 의미도 보존하는 성과를 기록하였습니다.



### AMR-Evol: Adaptive Modular Response Evolution Elicits Better Knowledge Distillation for Large Language Models in Code Generation (https://arxiv.org/abs/2410.00558)
Comments:
          EMNLP 2024

- **What's New**: 이 연구는 Adaptive Modular Response Evolution (AMR-Evol) 프레임워크를 소개하며, 복잡한 지시사항에 대한 응답 품질을 개선하기 위해 두 단계 프로세스를 채택합니다.

- **Technical Details**: 첫 번째 단계인 modular decomposition(모듈 분해)은 직접 응답을 더 관리하기 쉬운 하위 모듈로 분해합니다. 두 번째 단계인 adaptive response evolution(적응형 응답 진화)은 관련 기능 모듈을 통해 자동으로 응답을 발전시킵니다.

- **Performance Highlights**: 세 가지 인기 코드 벤치마크(HumanEval, MBPP, EvalPlus)에서 AMR-Evol 프레임워크는 기존 응답 증류 방법에 비해 우수한 성능을 보였으며, HumanEval-Plus에서 +3.0 포인트, MBPP-Plus에서 +1.0 포인트의 성능 향상을 관찰했습니다.



### What the Harm? Quantifying the Tangible Impact of Gender Bias in Machine Translation with a Human-centered Study (https://arxiv.org/abs/2410.00545)
Comments:
          Accepted ad EMNLP 2024

- **What's New**: 이 연구는 기계 번역(Machine Translation, MT)에서 성 편향이 사용자에게 미치는 실제적인 영향을 탐구하는 것을 목표로 하였습니다. 90명의 참여자를 대상으로 한 실험을 통해 성별에 따른 번역 품질 격차와 함께 물질적 비용이 어떻게 발생하는지를 분석했습니다.

- **Technical Details**: 연구에서는 90명의 참가자가 MT 출력물을 여성 혹은 남성젠더로 바르게 번역하기 위해 후수정(Post-editing) 작업을 진행하였습니다. 이 과정에서 편집한 시간과 번역 수정 횟수를 기록하여 성별에 따른 노력을 비교하였습니다. 분석된 데이터는 Hugging Face를 통해 공개되었습니다.

- **Performance Highlights**: 연구 결과, femininene 번역 작업은 평균적으로 남성 번역보다 두 배 더 많은 시간이 소요되며 네 배 더 많은 수정 작업이 필요했습니다. 이러한 성별에 따른 편향은 서비스 품질 격차와 재정적 비용으로 이어질 수 있으며, 현재의 자동화된 편향 측정 방법은 이 같은 차이를 반영하지 못하고 있습니다.



### Benchmarking Large Language Models for Conversational Question Answering in Multi-instructional Documents (https://arxiv.org/abs/2410.00526)
- **What's New**: InsCoQA라는 새로운 벤치마크를 제안하여 Instructional documents를 기반으로 한 Conversational Question Answering (CQA) 성능을 평가합니다. 이는 효율적으로 여러 소스에서 단계별 지침을 이해하고 요약하는 모델의 능력을 측정하는 데 초점을 둡니다.

- **Technical Details**: InsCoQA는 Xiaohongshu 플랫폼에서 수집된 13.9k개의 Instructional conversations로 구성되어 있으며, 각 데이터는 인간 주석자에 의해 검증되었습니다. InsEval이라는 LLM 보조 평가자를 도입해 모델의 응답 무결성과 정확성을 측정합니다. 이는 LLMs의 문서 이해 및 작업 완성 능력을 향상시키는 데 기여합니다.

- **Performance Highlights**: InsCoQA는 13개 도메인으로 분류된 13,959개의 샘플로 이루어져 있으며, 평균적으로 각 대화는 3.11개의 Q/A 라운드로 구성됩니다. 이 데이터셋은 여러 도메인의 복잡한 실제 작업을 포괄하는 데 적합하도록 설계되었습니다.



### Annotation Guidelines for Corpus Novelties: Part 2 -- Alias Resolution Version 1.0 (https://arxiv.org/abs/2410.00522)
- **What's New**: 이번 논문은 Alias Resolution(별칭 해석)을 위한 Annotated Corpus(주석이 달린 말뭉치)인 Novelties corpus의 구성과 주석 프로세스에서 적용된 지침을 설명합니다.

- **Technical Details**: 주석자들이 사용하는 지침과 그에 따른 여러 예제들이 포함되어 있으며, Canonical Names(표준 이름) 정의 및 동일 엔티티를 지칭하는 이름의 식별 방법이 제시됩니다.

- **Performance Highlights**: 주석 과정에서 수집된 다양한 사례를 통해 이름의 별칭 해석과 관련된 과제를 보다 명확히 이해할 수 있으며, 이는 다양한 NLP(자연어 처리) 작업에 도움을 줄 것입니다.



### Exploring the Learning Capabilities of Language Models using LEVERWORLDS (https://arxiv.org/abs/2410.00519)
- **What's New**: 이번 연구는 통계적 학습 모델링에서 일반 구조 규칙과 특수 인스턴스 속성을 동시에 학습하는 과정을 탐구합니다. 'LeverWorlds'라는 프레임워크를 설계하여 물리학에 영감을 받은 간단한 세계를 생성하고 이를 통해 샘플 효율성을 평가할 수 있는 통제된 실험을 수행합니다.

- **Technical Details**: LeverWorlds에서는 다양한 분포를 가진 단순 물리 모델의 세계를 생성할 수 있으며, 이러한 세계는 자연어로 표현할 수 있습니다. 연구에서는 전통적인 학습 알고리즘 및 Transformer 언어 모델을 활용한 실험 결과를 포함하여, 구조적 가정이 더 강한 전통적 방법보다 Transformers의 샘플 효율성이 낮음을 발견했습니다.

- **Performance Highlights**: Transformer 모델은 일반적으로 성공하지만, Maximum Likelihood Estimation과 Logistic Regression 같은 고전적인 방법과 비교했을 때 샘플 효율성이 현저히 낮습니다. 초기 결과로, 현대 언어 모델의 In-Context Learning(ICL) 기능과 고전적 알고리즘을 조합한 접근법이 유망한 가능성을 보여주었습니다.



### Cross-lingual Back-Parsing: Utterance Synthesis from Meaning Representation for Zero-Resource Semantic Parsing (https://arxiv.org/abs/2410.00513)
Comments:
          Accepted to EMNLP 2024

- **What's New**: 이번 연구에서는 다국어로 사전 훈련된 언어 모델(mPLMs)을 활용하여 의미 파싱(SP)에서 제로샷(Zero-Shot) 크로스링구얼(중언어) 이전을 향상시키기 위한 새로운 데이터 증강 방법론인 Cross-Lingual Back-Parsing (CBP)을 제안합니다. 이 방법론은 원본 의미 표현에서 목표 언어 발화를 합성하여 여러 언어로 SP의 확장을 지원합니다.

- **Technical Details**: CBP는 두 가지 주요 구성 요소로 이루어져 있습니다: 1) 발화 생성기(utterance generator), 2) 필터링 메커니즘(filtering mechanism). 발화 생성기는 mT5와 같은 다국어 사전 훈련 시퀀스-투-시퀀스(seq2seq) 모델과 모듈식 언어별 어댑터를 통해 목표 언어로 발화를 합성합니다. 필터링 메커니즘은 생성된 발화에서 저품질의 데이터를 제거하는 데 사용됩니다.

- **Performance Highlights**: CBP는 Mschema2QA 및 Xspider라는 두 개의 크로스링구얼 SP 벤치마크에서 실험하여, Mschema2QA에서는 평균 정확도가 3.2 포인트 개선되었고, Xspider에서는 중국어 정확도가 52.7에서 54.0으로 향상되었습니다. 이 연구는 목표 언어의 병렬 말뭉치가 전혀 없는 환경에서도 유의미한 성과를 달성했습니다.



### FlipGuard: Defending Preference Alignment against Update Regression with Constrained Optimization (https://arxiv.org/abs/2410.00508)
Comments:
          Accepted by EMNLP 2024 Main track

- **What's New**: 이번 논문에서는 Large Language Models(LLMs)의 선호도 정렬(preference alignment)에서 발생할 수 있는 문제인 업데이트 회귀(update regression)를 해결하기 위한 새로운 접근법인 FlipGuard를 제안합니다.

- **Technical Details**: FlipGuard는 제약 최적화(constrained optimization) 방법론을 활용하여 포컬 주의(focal attention)를 통해 업데이트 회귀를 감지하고 완화합니다. 주요 구성 요소는 맞춤형 보상 특성을 설정하고, 부정적 변화를 판단하며, 특정 조건이 충족되었을 때 사전 정렬된 모델과의 일치성을 보장하는 초점을 두고 학습 정책을 진화시키는 것입니다.

- **Performance Highlights**: FlipGuard를 적용한 실험 결과, 두 가지 정렬 알고리즘인 PPO(Proximal Policy Optimization)와 DPO(Direct Preference Optimization)에서 네 가지 다양한 선호도 데이터셋과 여섯 개의 학술 벤치마크를 사용하여 부정적 변화를 효과적으로 줄이고 전체 성능을 향상시키는 데 성공했습니다. 또한, FlipGuard는 사전 정렬 모델의 본질적인 지식을 보존하는 데 기여함을 입증했습니다.



### Multi-Target Cross-Lingual Summarization: a novel task and a language-neutral approach (https://arxiv.org/abs/2410.00502)
Comments:
          Accepted to EMNLP 2024 (Findings)

- **What's New**: 이번 논문에서는 여러 목표 언어를 고려하는 multi-target cross-lingual summarization (MTXLS)이라는 새로운 과제를 소개합니다. 이 과제는 여러 언어에서 문서를 요약하되, 생성된 요약이 의미적으로 유사하도록 하는 데 중점을 두고 있습니다.

- **Technical Details**: MTXLS는 여러 목표 언어 간 의미 일관성(semantic coherence)을 보장하기 위한 새로운 프레임워크로, re-ranking 방식으로 의미 있게 요약을 선택합니다. 이 접근 방식은 언어 중립(language-neutral) 전략을 채택하여, 성과의 신뢰성을 높입니다. 또한, 기계 번역의 품질 추정(quality estimation) 방법을 사용하여 생성된 요약의 일관성을 평가하는 다중 기준 평가 프로토콜(multi-criteria evaluation protocol)을 제안합니다.

- **Performance Highlights**: 연구에서는 기존의 cross-lingual summarization 방법들이 주로 단일 언어 쌍에 초점을 맞추었으나, MTXLS는 다양한 타깃 언어에서의 의미 일관성을 보장합니다. 이 결과는 법적 또는 규제적 요구 사항을 충족하는 데에도 중요한 역할을 할 수 있습니다.



### Self-Updatable Large Language Models with Parameter Integration (https://arxiv.org/abs/2410.00487)
- **What's New**: SELF-PARAM(자가 갱신 가능한 대규모 언어 모델) 방법을 제안하며, 추가 파라미터 없이 최적의 효율성과 강력한 장기 기억을 유지할 수 있게 됨.

- **Technical Details**: SELF-PARAM은 Kullback-Leibler(KL) divergence를 최소화하는 훈련 목표를 사용하여 기존 모델과 목표 모델의 예측값 간의 차이를 줄이고, 이를 통해 모델 파라미터 내에서 지식을 매끄럽게 내부화함. 다양한 질문-답변 쌍을 생성하여 목표 모델을 업데이트함.

- **Performance Highlights**: SELF-PARAM은 기존 방법들과 비교해 큰 폭으로 성능을 향상시켰으며, 추가 생산물 없이 높은 효율성과 견고한 장기 기억을 달성함. 다양한 질문 응답 및 대화 추천 작업에서 기존 방법을 능가함.



### Conversational Exploratory Search of Scholarly Publications Using Knowledge Graphs (https://arxiv.org/abs/2410.00427)
Comments:
          Accepted to ICNLSP 2024

- **What's New**: 이 논문에서는 기존의 그래픽 검색 인터페이스의 복잡성과 데이터 양으로 인해 학술 출판물 탐색이 어려운 문제를 해결하기 위해 대화형 검색 시스템을 개발하였습니다. 이 시스템은 지식 그래프(KG)를 활용하여 사용자가 자연어로 대화하며 연구 논문을 효과적으로 발견할 수 있도록 지원합니다.

- **Technical Details**: 대화형 검색 시스템은 사용자에게 3단계 검색 프로세스를 통해 관련 논문을 좁혀주며, 각각의 단계에서 사용자의 검색 목표에 적합한 연구 주제를 추천합니다. 시스템의 아키텍처는 대화형 인터페이스, 대화 관리 기능, KG 검색 기능 등으로 구성되어 있으며, RASA 및 Neo4j를 활용하여 대화 및 데이터 검색을 구현합니다.

- **Performance Highlights**: 40명의 참가자를 대상으로 한 인간 평가 결과, 대화형 인터페이스가 전통적인 텍스트 기반 검색 방식에 비해 더 나은 정보 탐색 경험을 제공함을 입증했습니다. 논문의 평가 결과는 대화형 검색 시스템 설계를 발전시키는데 있어 실질적인 인사이트를 제공합니다.



### Are LLMs Aware that Some Questions are not Open-ended? (https://arxiv.org/abs/2410.00423)
Comments:
          Accepted by EMNLP 2024

- **What's New**: 본 논문에서는 대형 언어 모델(LLM)이 질문의 유형에 따라 응답을 조정하는 "질문 인지(Question Awareness)" 능력을 평가합니다. 이 연구는 LLM이 비정형 질문에는 너무 자유롭게 응답하거나, 정형 질문에는 너무 지루하게 응답하는 경향이 있음을 발견했습니다.

- **Technical Details**: 질문 인지를 평가하기 위해, 모델의 출력 분포의 경도를 측정합니다. 모델의 Softmax 함수 온도를 활용하여 출력 분포의 경도를 조정하며, 질문의 특성에 따라 출력을 더 결정적으로 만드는 방법인 Question Awareness Temperature Sampling (QuATS)을 제안합니다. QuATS는 수동 온도 조정 없이도 모델의 질문 인지를 향상시킵니다.

- **Performance Highlights**: 실험 결과, QuATS 방법을 적용한 LLM들은 질문 인지가 개선되었으며, 다양한 벤치마크에서 일관되게 모델 성능이 향상되었습니다.



### Semantic Parsing with Candidate Expressions for Knowledge Base Question Answering (https://arxiv.org/abs/2410.00414)
- **What's New**: 이 연구에서는 대규모 지식 베이스(KB)에서 후보 표현들(candidate expressions)을 활용하여 의미 분석(semantic parsing)의 성능을 향상시키기 위한 문법을 제안합니다.

- **Technical Details**: 제안된 문법은 생산 규칙(production rules)으로서의 행동(actions)을 정의하며, 우리의 의미 분석기는 유형(types) 및 후보 표현들에 의해 제약(constraints)된 상태에서 추론(inference) 중에 행동을 예측합니다. 이는 지식 베이스 질문 응답에 적용되어, 후보 표현들로 인한 제약이 의미 분석기가 유효한 KB 요소를 생성하도록 돕습니다.

- **Performance Highlights**: 두 개의 벤치마크인 KQA Pro와 Overnight에서 실험을 진행한 결과, 후보 표현들에 의한 제약이 우리의 의미 분석기의 정확성을 높였습니다. 강력한 지도(supervision) 혹은 약한 지도 하에서 모두 향상된 결과를 보였으며, KQA Pro와 Overnight에서 최첨단 정확도(state-of-the-art accuracy)를 달성하였습니다.



### TPN: Transferable Proto-Learning Network towards Few-shot Document-Level Relation Extraction (https://arxiv.org/abs/2410.00412)
Comments:
          Few shot document-level relation extraction

- **What's New**: 본 연구는 문서 수준 관계 추출(few-shot document-level relation extraction; FSDLRE)의 효율성을 개선하기 위해 TPN(Transferable Proto-Learning Network)을 제안합니다. 이 네트워크는 Hybrid Encoder, Transferable Proto-Learner, Dynamic Weighting Calibrator의 세 가지 주요 구성 요소로 이루어져 있습니다.

- **Technical Details**: TPN은 NOTA(None-Of-The-Above) 관계의 교차 분야 전이 가능성을 개선하기 위해 설계되었습니다. Hybrid Encoder는 입력 텍스트의 의미적 내용을 계층적으로 인코딩하여 관계 표현을 증진시키고, Transferable Proto-Learner는 적응 가능한 블록을 통해 NOTA 프로토타입을 계산하여 다양한 도메인 간의 NOTA 편향을 완화합니다. Dynamic Weighting Calibrator는 관계 특이적 분류 신뢰도를 감지하여 동적 가중치로 NOTA 중심 손실 함수를 보정합니다.

- **Performance Highlights**: FREDo와 ReFREDo 데이터셋에서 TPN의 우수성을 입증하는 실험 분석을 수행했습니다. 최신 방법들과 비교했을 때, TPN은 약 절반의 파라미터 크기(123MB vs. 221MB)로 경쟁력 있는 성능을 달성했습니다.



### AlignSum: Data Pyramid Hierarchical Fine-tuning for Aligning with Human Summarization Preferenc (https://arxiv.org/abs/2410.00409)
Comments:
          EMNLP2024 Findings, code at: this https URL

- **What's New**: 이번 연구에서는 AlignSum이라는 새로운 인간 요약 선호 맞춤화 프레임워크를 소개합니다. 이 프레임워크는 추출적, 요약적, 인간 주석 데이터로 구성된 Data Pyramid를 구축하고, 극단적인 길이의 요약을 제거하기 위해 Gaussian Resampling을 실시한 후, 인간 요약 선호도에 맞춘 두 단계의 계층적 파인튜닝을 구현합니다.

- **Technical Details**: AlignSum 프레임워크는 세 가지 주요 부분으로 구성됩니다: 1) Data Pyramid 설계, 2) Gaussian Resampling을 통한 요약 길이 조정, 3) 인간 주석 데이터에 대한 두 단계 계층적 파인튜닝입니다. 이 방법은 CNN/DailyMail 및 BBC XSum 데이터셋에 적용되어 PLM인 BART-Large가 175B GPT-3를 초과하는 성능을 보였습니다.

- **Performance Highlights**: AlignSum을 적용한 PLM 모델은 자동 평가와 인간 평가 모두에서 뛰어난 성능을 발휘하며, 이는 AlignSum이 언어 모델과 인간 요약 선호 간의 정렬을 현저히 개선하는 데 기여함을 보여줍니다.



### Boosting the Capabilities of Compact Models in Low-Data Contexts with Large Language Models and Retrieval-Augmented Generation (https://arxiv.org/abs/2410.00387)
Comments:
          13 pages, 1 figure, 5 tables, submitted to COLING 2025

- **What's New**: 이 논문에서는 데이터가 부족한 저자원 언어에 대한 기존 언어 모델 기술의 한계를 극복할 수 있는 접근법으로 Retrieval Augmented Generation (RAG) 프레임워크를 제안합니다. 특히, Uspanteko 및 Arapaho 언어의 형태론적 주석 작업을 위한 소규모 모델의 출력을 교정하는 방법을 다룹니다.

- **Technical Details**: RAG 프레임워크는 대형 언어 모델(LLM)을 활용하여 문법적 서술 정보를 결합합니다. 이를 통해 데이터와 학습 가능한 매개변수의 부족을 보완하며, LLM을 통해 해석된 문법 입력값을 사용합니다. 실험에서는 Claude 3.5 Sonnet과 GPT-4 모델을 비교하여 Uspanteko 및 Arapaho 언어에 대한 성능을 평가했습니다.

- **Performance Highlights**: 저자원 언어 처리에서의 성능과 효율성이 크게 향상되는 것을 보여주었으며, 새로운 SOTA(최신 기술 수준)를 달성했습니다. 이 접근법은 문서화 언어학자들에게 형태론적 주석 작업을 위한 더 신뢰할 수 있고 사용하기 쉬운 도구를 제공합니다.



### Answer When Needed, Forget When Not: Language Models Pretend to Forget via In-Context Knowledge Unlearning (https://arxiv.org/abs/2410.00382)
- **What's New**: 본 연구에서는 LLMs (대형 언어 모델)이 특정 정보를 선택적으로 잊어버리는 새로운 방법인 'in-context knowledge unlearning'을 제안합니다. 이 방법은 쿼리의 문맥을 기반으로 테스트 시 특정 지식을 잊어버리도록 모델을 조정합니다.

- **Technical Details**: 제안된 방법은 'unlearning tokens'를 사용하여 특정 정보 u를 선택적으로 무시하고, 전반적인 모델 성능을 유지하면서도 80%의 관련 없는 지식을 유지하여 최대 95%의 기억상실 정확도를 달성합니다. 실험은 Llama2-7B/13B 및 Mistral-7B 모델을 사용하여 TOFU 및 AGE 데이터셋에서 수행되었습니다.

- **Performance Highlights**: 이 연구에서 제안된 방법은 기존 방식에 비해 인도메인 및 아웃오브도메인 시나리오에서 모두 유의미하게 성능이 향상되었습니다. 모델의 내부 동작에 대한 추가 연구 결과, LLMs는 마지막 층에서 잊어버리기로 결정하며, 이는 'LLMs pretend to forget'이라는 통찰력을 제공합니다.



### Unleashing the Potentials of Likelihood Composition for Multi-modal Language Models (https://arxiv.org/abs/2410.00363)
- **What's New**: 본 논문에서는 다양한 아키텍처와 파라미터 크기를 가진 모델들을 효과적으로 결합하기 위한 새로운 프레임워크인 'likelihood composition'을 제안합니다. 이 프레임워크는 여러 모델의 likelihood 분포를 조합하여 다중 선택 시각 질문 답변 과제를 수행하는 데 중점을 두고 있습니다.

- **Technical Details**: 'likelihood composition'의 핵심 개념은 후보 답변의 로그 확률(log-probability)입니다. 주요 연산으로는 'debias', 'highlight', 'majority-vote', 'ensemble'이 있으며, 이들을 조합하여 'mix-composition'이라는 새로운 방법들을 도출해냈습니다. 또한, 실험을 통해 제안된 방법의 효과를 검증하였습니다.

- **Performance Highlights**: 9개의 VQA 데이터셋 및 10개의 MLM(models)의 실험 결과, mix-composition이 기존의 ensemble 또는 majority-vote 방법보다 성능이 우수함을 입증하였습니다. 자가 조합(self-composition)을 사용하면 성능이 증가하며, 특히 미개발 모델에서의 성능 향상이 두드러집니다.



### FedPT: Federated Proxy-Tuning of Large Language Models on Resource-Constrained Edge Devices (https://arxiv.org/abs/2410.00362)
Comments:
          29 pages, 19 figures

- **What's New**: 이 논문에서는 Federated Proxy-Tuning (FedPT)이라는 새로운 프레임워크를 소개하여 개인 데이터를 공유하지 않고도 대형 블랙박스 언어 모델을 효율적으로 미세 조정할 수 있는 방법을 제안하고 있습니다.

- **Technical Details**: FedPT는 먼저 장치들이 소형 언어 모델을 공동으로 조정한 후, 서버가 이 작은 모델의 지식을 큰 사전 훈련된 모델과 결합하여 성능을 향상시키는 방법입니다. 이 과정은 반복적으로 수행되며, 최종적으로 큰 프록시 조정 모델을 생성합니다. 회로에서 모델의 파라미터에 직접 접근하지 않고 예측값만을 사용합니다.

- **Performance Highlights**: 실험 결과, FedPT는 직접 대형 LM을 연합적으로 미세 조정하는 것보다 계산, 통신, 메모리 오버헤드를 크게 줄이면서 경쟁력 있는 성능을 유지하는 것으로 나타났습니다. 이러한 접근은 리소스가 제한된 장치에서 대형 언어 모델의 접근성과 활용성을 넓히는 가능성을 제시합니다.



### PclGPT: A Large Language Model for Patronizing and Condescending Language Detection (https://arxiv.org/abs/2410.00361)
Comments:
          Accepted for EMNLP2024 (Findings)

- **What's New**: 본 논문에서는 약자 집단을 겨냥한 Patronizing and Condescending Language (PCL)의 탐지를 위한 종합적인 LLM 벤치마크인 PclGPT를 소개합니다. 전통적인 사전 훈련 언어 모델은 PCL의 미세한 독성 특성으로 인해 탐지 성능이 저조하였지만, 이 연구는 대형 언어 모델이 가진 풍부한 감정적 의미를 활용하여 PCL 탐지 문제를 해결하고자 합니다.

- **Technical Details**: PclGPT는 두 개의 모델(영어용 PclGPT-EN과 중국어용 PclGPT-CN)로 구성되어 있습니다. 이들 모델은 각각 LLaMA-2-7B와 ChatGLM-3-6B를 기반으로 하며, 140만 개 이상의 데이터를 포함하는 Pcl-PT/SFT 데이터셋을 사용하여 도메인 적응형 사전 훈련과 감독 미세 조정을 거칩니다. 이 연구의 주요 목표는 PCL과 기타 암시적 독성을 감지할 수 있는 LLM을 구축하는 것입니다.

- **Performance Highlights**: PclGPT는 기존의 PLM 및 LLM에 비해 언어 데이터셋에서 탁월한 성능을 보이며, 다양한 취약 집단에 대한 PCL의 편향 정도를 밝혀내는 데 성공했습니다. 이를 통해 PCL 탐지가 사회적으로 더욱 주목받아야 하며, PclGPT는 이러한 편향 관리를 위한 기초를 마련합니다.



### Self-controller: Controlling LLMs with Multi-round Step-by-step Self-awareness (https://arxiv.org/abs/2410.00359)
Comments:
          10 pages, 6 figures

- **What's New**: 본 논문에서는 LLM(large language models)의 제어 능력을 향상시키기 위한 새로운 프레임워크인 'Self-controller'를 제안합니다. 이 프레임워크는 LLM이 자신의 상태를 인식하고 단계별로 사고할 수 있도록 돕습니다.

- **Technical Details**: Self-controller는 상태 반영기(state reflector)와 다중 라운드 대화 세션으로 구성됩니다. 이 프레임워크는 LLM의 응답에 기반한 상태를 유지하며, 텍스트 길이를 조정하기 위해 이진 검색 알고리즘을 구현하였습니다. 또한, DeepSeek의 Context Caching 기술을 활용하여 동일한 맥락의 대화 클러스터에서 계산된 토큰 소모를 획기적으로 줄입니다.

- **Performance Highlights**: 실험 결과, Self-controller는 다양한 데이터셋에서 LLM의 제어 가능성을 크게 향상시켰으며, 성능 저하 없이 효과적인 텍스트 생성을 달성했습니다. 본 방법은 일반적인 단일 라운드 생성에 비해 최대 두 배의 토큰 소모만을 요구합니다.



### Hierarchical Organization Simulacra in the Investment Sector (https://arxiv.org/abs/2410.00354)
- **What's New**: 이 논문은 인공지능 기관을 설계하여 투자에서 전문적인 행동을 모방하기 위해 Multi-Agent Simulation (다중 에이전트 시뮬레이션) 기법을 활용합니다. 이는 Hierarchical Decision-Making (계층적 결정 과정)을 모사하며, 뉴스 기사를 분석하여 결정 과정을 진행합니다.

- **Technical Details**: 연구에서는 300개 회사에 대한 115,000건 이상의 뉴스 기사를 15년간 분석하고, 이 데이터를 기반으로 에이전트 간의 상호작용을 바탕으로 한 실험을 진행하였습니다. 사용된 모델은 주로 LLMs (대규모 언어 모델)인 GPT-3.5와 PaLM-2를 포함합니다. 에이전트의 결정을 Overweight (과대평가)와 Underweight (과소평가)로 분류하며, 이들의 일치도를 평가합니다.

- **Performance Highlights**: 계층적 다중 에이전트 접근 방식이 전문 트레이더의 결정과 높은 일치를 보이며, 더 높은 수익성을 발휘했습니다. 특히, 프로프트의 특징을 바꾸는 것이 결정의 결과에 상당한 영향을 미치며, 예를 들어 단기에서 장기로의 전환이 전문 트레이더의 선택과 더 근접한 결과를 가져왔습니다.



### Preserving Generalization of Language models in Few-shot Continual Relation Extraction (https://arxiv.org/abs/2410.00334)
Comments:
          Accepted to EMNLP 2024

- **What's New**: 이 연구에서는 Few-shot Continual Relations Extraction (FCRE) 분야에서 새로운 접근 방식을 제안합니다. 특히, 기존에 소홀히 여겨지던 언어 모델 헤드를 활용하여 연결된 지식을 효과적으로 통합하는 방법을 소개합니다.

- **Technical Details**: Mutual Information Maximization (MIM) 전략을 통해 사전 훈련된 백본 네트워크로부터 이전 지식을 보존하고, 주요 분류 헤드의 정렬을 전략적으로 개선하여 모델 성능을 향상시킵니다. 또한, Large Language Models (LLMs)의 잠재력을 FCRE 문제에 적용하는 방법을 탐구합니다.

- **Performance Highlights**: 실험 결과는 제안된 방법이 사전 훈련된 언어 모델의 일반화 능력을 유지하고, 지식의 잊혀짐을 줄이는 데 있어 매우 효과적임을 보여줍니다. 이는 모델의 대표성 학습을 개선하고 전체적인 성능을 크게 향상시키는 결과로 이어집니다.



### EmoKnob: Enhance Voice Cloning with Fine-Grained Emotion Contro (https://arxiv.org/abs/2410.00316)
Comments:
          EMNLP 2024 Main

- **What's New**: 최근 Text-to-Speech (TTS) 기술의 발전에도 불구하고 감정을 선택하고 강도를 조절할 수 있는 기능이 부족했습니다. 본 연구에서는 EmoKnob 라는 프레임워크를 제안하며, 이는 임의의 감정을 갖는 소수의 샘플로 감정을 정밀하게 조정할 수 있습니다.

- **Technical Details**: 이 프레임워크는 최근의 기초 음성 복제 모델의 발전을 활용하여 감정을 제어할 수 있는 두 가지 방법을 제안합니다. 첫 번째는 감정 설명을 텍스트로 제공받고 이를 통해 감정을 조절하는 것이며, 두 번째는 대량의 언어 모델과 텍스트 임베딩 모델을 활용하는 것입니다. 또한, 감정 인식 및 충실성을 평가하기 위한 새로운 평가 지표 세트를 도입합니다.

- **Performance Highlights**: 객관적 및 주관적 평가를 통해 우리의 감정 제어 프레임워크가 감정을 효과적으로 음성에 내재화하며, 상용 TTS 서비스보다 감정 표현에서 우수한 성능을 보여주었습니다. 83%의 참여자들은 우리의 프레임워크가 감정 강화에 있어서 상위 TTS 서비스보다 뛰어나다고 평가했습니다.



### Insight: A Multi-Modal Diagnostic Pipeline using LLMs for Ocular Surface Disease Diagnosis (https://arxiv.org/abs/2410.00292)
Comments:
          Accepted to MICCAI 2024. Project Webpage: this https URL

- **What's New**: 본 논문은 시각적 데이터와 임상 메타데이터를 결합하여 안구 표면 질환을 진단하기 위한 혁신적인 다중 모달 진단 파이프라인(MDPipe)을 제안합니다. 기존의 방법에서는 진단을 정해진 클래스 분류 문제로 다루어 임상 변수 간의 관계를 고려하지 못했습니다.

- **Technical Details**: MDPipe는 대형 언어 모델(LLMs)을 활용하여 메이보그래피 이미지를 정량화할 수 있는 형태의 형태학적 데이터로 변환합니다. 그 후, LLM 기반 요약기를 통해 이를 요약하고 임상 보고서로 활용합니다. 이 방식은 LLM의 추론 능력을 실제 임상 진단에 맞춰 개선합니다.

- **Performance Highlights**: MDPipe는 다양한 안구 표면 질병 진단 기준에서 테스트하여 기존의 기준(예: GPT-4)을 능가하며 임상적으로 타당한 진단 근거를 제공합니다.



### DoPAMine: Domain-specific Pre-training Adaptation from seed-guided data Mining (https://arxiv.org/abs/2410.00260)
- **What's New**: 이번 연구에서 제안하는 DoPAMine 프레임워크는 대규모 데이터 코퍼스에서 도메인 특화 훈련 데이터를 자동으로 채굴하여 대형 언어 모델(LLM)의 도메인 적응을 지원합니다. 이 방법은 일반적인 웹 데이터를 기반으로 한 이제까지의 접근 방식과는 달리, 도메인에 맞춘 데이터 생성 및 실제 세계 데이터를 채굴하여 현실적이고 신뢰할 수 있는 결과를 제공합니다.

- **Technical Details**: DoPAMine 프레임워크는 LLM의 매개변수적 지식을 활용하여 특정 도메인에 최적화된 다양한 시드(seed) 데이터를 생성하고, 이를 통해 Common Crawl과 같은 대규모 데이터 코퍼스에서 관련 데이터 문서를 채굴하는 과정을 포함합니다. 핵심 메커니즘은 LLM을 활용하여 도메인 특화된 시드 데이터를 생성하고, 생성된 시드 데이터를 기반으로 의미적으로 유사한 문서를 검색하는 것입니다.

- **Performance Highlights**: DoPAMine를 사용하여 지속적 재학습(CPT) 환경에서 두 개의 도메인 특화 7B 파라미터 LLM을 의료 및 금융 분야에서 훈련한 결과, 평균 4.9% 및 5.1%의 성능 향상을 보였으며, 이는 zero-shot 및 5-shot 설정에서 MMLU, MedQA, MedMCQA 및 PubMedQA 데이터셋에 대한 것입니다. 금융 과제에서는 평균 2.9% 및 6.7%의 향상을 나타내었으며, FiQA-SA, FPB 및 Headlines 데이터셋과 비교하여 성능이 크게 높아졌습니다.



### A Methodology for Explainable Large Language Models with Integrated Gradients and Linguistic Analysis in Text Classification (https://arxiv.org/abs/2410.00250)
Comments:
          27 pages, 6 figures, authors Marina Ribeiro and Bárbara Malcorra have equal contribution, César Rennó-Costa is the corresponding author

- **What's New**: 이번 논문은 알츠하이머병(Alzheimer's Disease, AD)과 같은 언어 생산에 영향을 주는 신경 장애를 평가하기 위한 설명 가능한 LLM(대형 언어 모델) 방법인 SLIME(모델 설명을 위한 통계적 및 언어적 통찰력)를 제안합니다.

- **Technical Details**: SLIME 방법은 Cookie Theft 그림 설명 과제의 전사로 구성된 영어 데이터셋을 사용하여 개발되었습니다. LLM(Bidirectional Encoder Representations from Transformers, BERT)을 사용하여 텍스트 설명을 AD 그룹과 대조군으로 분류하고, Integrated Gradients (IG), Linguistic Inquiry and Word Count (LIWC) 및 통계 분석을 활용하여 대표적인 어휘적 특징을 식별합니다.

- **Performance Highlights**: BERT는 AD에서 사회적 언급의 감소를 반영하는 어휘 구성 요소를 활용하고, 이러한 요소들이 LLM의 정확도를 향상시키는 데 기여함을 보여줍니다. 이로 인해, 신경 임상 맥락에서 LLM 적용에 대한 신뢰성을 높이는 설명 가능성 도구를 제공합니다.



### T-KAER: Transparency-enhanced Knowledge-Augmented Entity Resolution Framework (https://arxiv.org/abs/2410.00218)
Comments:
          Accepted by IDCC 2024

- **What's New**: 이 논문은 T-KAER(Transparency-enhanced Knowledge-Augmented Entity Resolution) 프레임워크를 제안하여 엔티티 해석(Entity Resolution, ER) 과정의 투명성을 높입니다. 이는 KAER 프레임워크의 한계인 외부 지식의 기여를 문서화하지 않은 점을 해결하기 위한 것입니다.

- **Technical Details**: T-KAER는 세 가지 투명성 관련 질문(T-Qs)을 설정하고 이를 통해 실험 과정, 원시 데이터에서 증강된 의미 정보, 증강된 데이터가 예측에 미치는 영향을 명확히 분석합니다. 이 프레임워크는 로그 파일에 엔티티 해석 과정을 기록하여 투명성을 향상시킵니다.

- **Performance Highlights**: 실험에서는 인용(citation) 데이터 세트를 사용하여 T-KAER의 투명성 구성 요소를 보여줍니다. T-KAER는 정량적 및 정성적 관점에서의 오류 분석을 용이하게 하며, 어떤 의미 정보가 증강되었고 왜 증강된 지식이 예측에 다르게 영향을 미치는지를 입증합니다.



### Evaluating the performance of state-of-the-art esg domain-specific pre-trained large language models in text classification against existing models and traditional machine learning techniques (https://arxiv.org/abs/2410.00207)
Comments:
          56 pages, 9 figures

- **What's New**: 이 연구는 텍스트 공개자료 내에서 환경(Environmental), 사회(Social), 지배구조(Governance) 관련 정보를 분류하는 새로운 방법을 제시합니다.

- **Technical Details**: 연구는 데이터 수집, 데이터 전처리, ESG 중심의 대형 언어 모델(Large Language Models, LLMs) 및 전통적인 머신러닝(Support Vector Machines, XGBoost) 분류기를 개발하는 양적 접근 방식을 사용합니다. Qlora라는 새로운 fine-tuning 방법을 적용하여 LLM의 성능을 개선합니다.

- **Performance Highlights**: 최신 자연어 처리 성능 지표(accuracy, precision, recall, F1-score)를 사용하여 EnvLlama 2-Qlora, SocLlama 2-Qlora, GovLlama 2-Qlora와 같은 도메인 특화 모델들이 ESG 텍스트 분류에서 뛰어난 성과를 보였습니다.



### Do Vision-Language Models Really Understand Visual Language? (https://arxiv.org/abs/2410.00193)
- **What's New**: 이 연구는 최근의 Large Vision-Language Models (LVLMs)가 도표와 관련된 복잡한 추론 작업을 수행할 수 있는지 조사합니다. 저자들은 LVLMs의 도표 이해 능력을 평가하기 위해 포괄적인 테스트 수트를 개발하였습니다.

- **Technical Details**: 테스트 수트는 여러 도메인에 걸친 합성 및 실제 도표 세트에서 개념 엔티티와 그 관계에 중점을 둔 다양한 질문을 사용하여 LVLMs의 인식 및 추론 능력을 평가합니다.

- **Performance Highlights**: 세 가지 LVLM들(GPT-4V, GPT-4o, Gemini)을 평가한 결과, 엔티티를 식별하고 추론하는 데는 정확성을 보였으나 관계 이해 능력은 상당히 제한적임을 보여주었습니다. 향상된 도표 이해 성능은 모델의 배경 지식을 활용한 단축키에 기인한다는 것을 발견하였습니다.



### Zero-Shot Classification of Crisis Tweets Using Instruction-Finetuned Large Language Models (https://arxiv.org/abs/2410.00182)
- **What's New**: 이 연구는 자연재해나 인도적 위기 상황에서 소셜 미디어 게시물의 분류를 위한 상업적 대형 언어 모델(LLMs)을 평가하고, 이러한 모델들이 위기 관련 정보의 분류 성능에서 어떤 차이를 보이는지를 분석합니다.

- **Technical Details**: 연구팀은 CrisisBench 데이터셋을 이용하여 세 가지 LLM(OpenAI GPT-4o, Google Gemini 1.5-flash-001, Anthropic Claude-3-5 Sonnet)의 제로샷(classification에 대한 사전 훈련 없이 직접 할 수 있는 방법) 분류 성능을 평가하였습니다. 주요 작업으로는 두 가지가 있으며, 첫째는 게시물이 인도적 맥락에서 정보를 제공하는지 여부를 판단하는 것이고, 둘째는 16개 인도적 클래스에 따라 게시물의 확률을 평가하는 것입니다. F1 점수로 성과를 평가했습니다.

- **Performance Highlights**: 연구 결과, 정보 제공 분류 작업은 추가 정보 없이도 대부분 잘 수행되었으나, 인도적 레이블 분류 작업의 경우 트윗이 채굴된 사건과 관련된 정보가 제공되었을 때 더 나은 성과를 보였습니다. 또한 데이터셋에 따라 모델 성능이 상당히 다르게 나타나 데이터셋 품질에 대한 의문을 제기하였습니다.



### Evaluating the fairness of task-adaptive pretraining on unlabeled test data before few-shot text classification (https://arxiv.org/abs/2410.00179)
Comments:
          To appear in the GenBench Workshop at EMNLP 2024

- **What's New**: 이 논문은 최신 NLP 기술을 평가하는 데 있어 few-shot learning 벤치마크의 편향을 분석합니다. 특히, 연구자들이 테스트 세트의 비공식 텍스트를 이용하여 사전 학습(pretraining)을 수행하면서 발생할 수 있는 성능 과대평가(overoptimism)의 여부를 실험을 통해 조사했습니다.

- **Technical Details**: 25개의 분류 작업과 3개의 언어 모델(BERT, GPT-2, Mistral 7B)을 사용한 controlled few-shot 및 zero-shot 실험을 통해, unlabeled test set 텍스트로 사전 학습하는 것이 성능에 미치는 영향을 연구했습니다. 하이퍼파라미터 튜닝과 같은 기존 기법들과 비교해, unlabeled text의 출처가 성능에 미치는 영향을 명확히 파악했습니다.

- **Performance Highlights**: 사전 학습의 이점이 검증되었으며, unlabeled test set 텍스트를 사용하는 것과 독립적으로 추출된 텍스트를 사용하는 것 간의 성능 차이에 대한 근거를 제시했습니다. 결과적으로, few-shot 학습 벤치마크에 여러 교육 폴드(training folds)를 포함하는 것이 중요하다는 것을 강조했습니다.



### Adaptable Moral Stances of Large Language Models on Sexist Content: Implications for Society and Gender Discours (https://arxiv.org/abs/2410.00175)
Comments:
          To be published at EMNLP2024

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)이 성차별적 언어를 비판하거나 옹호하는 도덕적 추론(moral reasoning)을 어떻게 적용할 수 있는지를 설명합니다. 8개의 LLM을 평가한 결과, 모든 모델이 성차별적 가정을 반영하는 다양한 관점에서 비판적이고 지지하는 설명을 제공할 수 있는 능력을 보여주었습니다.

- **Technical Details**: 저자들은 Moral Foundations Theory(MFT)를 기반으로 하여 LLM이 암묵적인 성차별적 게시물에 대한 비판 및 옹호의 주장을 생성할 수 있는지를 실험했습니다. 이를 통해, LLMs는 다양한 도덕적 기준을 적용하여 이러한 언어에 대한 설명을 유창하게 제공할 수 있음을 확인했습니다.

- **Performance Highlights**: 대부분의 LLM이 암묵적인 성차별적 댓글을 설명하기 위해 유창하고 관련성 있는 텍스트를生成할 수 있었으며, 같은 텍스트에 대해 성차별이 아니라고 주장하는 고품질의 도덕적 추론도 제공할 수 있음을 관찰했습니다. 이는 성차별적 언어에 대해 정당화하는 유해한 도덕적 주장을 재생산할 수 있는 능력을 보여줍니다.



### SSR: Alignment-Aware Modality Connector for Speech Language Models (https://arxiv.org/abs/2410.00168)
- **What's New**: 본 논문에서는 SpeechLM(음성 언어 모델)에 음성을 효과적으로 통합하기 위한 새로운 접근법인 SSR-Connector(세분화된 음성 표현 연결기)를 제안합니다. 이 방식은 기존의 음성-텍스트 정렬을 활용하여 음성 기능을 세분화하고 압축함으로써, 텍스트 임베딩의 세분성과 일치하도록 개선합니다.

- **Technical Details**: SSR-Connector는 두 단계의 훈련 파이프라인을 포함합니다. 첫 번째 단계에서는 LLM(대형 언어 모델)을 고정하고, 음성-텍스트 증류(distillation)를 통해 음성 입력을 텍스트 임베딩과 의미적으로 정렬된 압축 표현으로 전환합니다. 두 번째 단계에서는 LLM을 언프리즈(unfreeze)하고, 다음 토큰 예측(next-token prediction)으로 미세 조정(fine-tuning)하여 변환된 표현을 입력으로, 전사 토큰(transcription tokens)을 목표로 합니다.

- **Performance Highlights**: SSR-Connector는 기존 음성-텍스트 모달리티 융합 메커니즘보다 우수한 성능을 발휘하며, StoryCloze에서 +10 정확도와 Speech-MMLU에서 +20 정확도를 달성하며, 사전 훈련된 텍스트 능력을 유지합니다.



### Adapting LLMs for the Medical Domain in Portuguese: A Study on Fine-Tuning and Model Evaluation (https://arxiv.org/abs/2410.00163)
Comments:
          This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 이 연구는 포르투갈어로 된 대규모 언어 모델(LLMs)의 의료 에이전트 성능을 평가하여 의료 전문가를 위한 신뢰할 수 있고 관련성이 있는 가상 비서를 개발하는 것을 목표로 하고 있습니다.

- **Technical Details**: HealthCareMagic-100k-en과 MedQuAD 데이터셋을 영어에서 GPT-3.5를 사용하여 번역하고, PEFT-QLoRA 방법을 통해 ChatBode-7B 모델을 미세 조정하였습니다. InternLM2 모델이 의료 데이터에서 초기 훈련을 통해 가장 높은 성능을 나타냈고, DrBode 모델은 취득한 의료 지식의 재앙적 망각(catatrophic forgetting) 현상을 보였습니다.

- **Performance Highlights**: 이 연구의 결과는 다국어 의료 모델의 평가, 훈련 데이터 품질 향상, 그리고 의료 분야에 맞는 일관된 평가 방법론 개발의 필요성을 강조합니다. 다각적인 평가를 통해 의료 전문가들의 높은 정확도, 완전성, 안전성을 확인하였습니다.



### KV-Compress: Paged KV-Cache Compression with Variable Compression Rates per Attention Head (https://arxiv.org/abs/2410.00161)
- **What's New**: 본 논문에서는 KV-Compress라는 새로운 압축 방법을 제안하며, 이 방법은 PagedAttention 프레임워크 내에서 연속적인 KV 블록을 퇴출시킴으로써 KV 캐시의 메모리 용적을 이론적인 압축 비율에 비례하여 줄입니다.

- **Technical Details**: KV-Compress는 변수별 삭제 비율(variable rate of eviction)을 각 레이어마다 적용하고, 이전의 주의(attention) 정보를 바탕으로 KV를 퇴출하는 알고리즘적 개선을 포함하여, Grouped-Query-Attention (GQA) 모델에 대한 더 효과적인 캐시 관리 방법을 제공합니다.

- **Performance Highlights**: Mistral-7B-Instruct-v0.2 및 Llama-3.1-8B-Instruct에서 LongBench 성능 평가 시 기존 방법들에 비해 4배 증가한 KV 수로 압축 달성하고, Llama-3.1-8B-Instruct 및 Llama-3.1-70B-Instruct-FP8에 대한 평가에서는 최대 8배의 압축률과 함께 성능 저하가 미미하다는 결과를 보였습니다.



### Beyond Single Concept Vector: Modeling Concept Subspace in LLMs with Gaussian Distribution (https://arxiv.org/abs/2410.00153)
Comments:
          28 pages, 9 figures

- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)의 내부 지식 표현을 이해하기 위해 Gaussian Concept Subspace (GCS)를 도입했습니다. 이는 기존의 단일 벡터 대신, 특정 개념을 나타내는 서브스페이스를 근사하는 접근 방식입니다.

- **Technical Details**: GCS는 각각의 개념에 대해 단일 벡터가 아닌 관찰 벡터(observed vectors) 집합을 학습하여, 가우시안 분포를 사용하여 개념 서브스페이스를 추정합니다. 이로 인해 더 미세한 개념 표현이 가능해지고, 벡터의 밀도가 개념에 대한 관련성의 정도를 나타냅니다.

- **Performance Highlights**: 실험 결과 GCS로부터 샘플링된 벡터는 LLM에서 학습된 개념을 효과적으로 설명하고, 다운스트림 개입(intervention) 작업에서 더 바람직한 결과를 생성하는데 기여함을 보여주었습니다. 특히, 자연어 생성 과제에서 텍스트 유창성을 유지하면서 감정 조정 성능을 균형 있게 달성하는 데 성공했습니다.



### Scheherazade: Evaluating Chain-of-Thought Math Reasoning in LLMs with Chain-of-Problems (https://arxiv.org/abs/2410.00151)
- **What's New**: 이 논문에서는 Scheherazade라는 자동화된 방법을 소개하여, 수학적 추론 문제들을 논리적으로 연결한 도전적인 벤치마크를 생성합니다. 기존의 벤치마크가 여러 LLM의 발전에 대한 평가에 덜 유용해진 반면, Scheherazade는 이러한 문제를 해결합니다.

- **Technical Details**: Scheherazade는 두 가지 연결 기법, 즉 forward chaining과 backward chaining을 이용하여 문제들을 체인화합니다. Forward chaining은 문제를 순차적으로 연결하여 해결하는 반면, backward chaining은 이전 문제의 해결을 위해 미래 문제의 정보가 필요하도록 만듭니다. 이 방법은 Chain-of-Thought (CoT) 추론 능력을 테스트하는 데 유용합니다.

- **Performance Highlights**: GSM8K에 Scheherazade를 적용한 결과, 여러 LLM이 연결된 질문이 많아질수록 성능이 급격히 떨어지는 반면, OpenAI의 o1-preview는 5개의 질문을 역으로 연결해도 성능을 유지합니다. 다른 모델들과 달리 o1-preview는 역 연결 문제에서 더 나은 성과를 보였습니다.



### Are Large Language Models In-Context Personalized Summarizers? Get an iCOPERNICUS Test Done! (https://arxiv.org/abs/2410.00149)
- **What's New**: 본 논문은 In-Context Learning (ICL)을 기반으로 한 요약에서 사용자의 선호 이력을 반영하는 In-Context Personalization Learning (ICPL)의 중요성을 강조하며, iCOPERNICUS라는 새로운 프레임워크를 제안합니다.

- **Technical Details**: iCOPERNICUS 프레임워크는 세 가지 요소를 통해 LLMS의 ICPL을 평가합니다: (i) 예제 요약의 효과, (ii) 사용자의 읽기 이력 추가 시 ICPL 향상 여부, (iii) 사용자 프로필 대비 정보를 통한 ICPL 유도 여부. EGISES를 개인화 측정 지표로 활용하여 각 요소의 영향을 비교합니다.

- **Performance Highlights**: 17개의 최신 LLM 모델을 평가한 결과, 15개의 모델에서 ICPL 성능이 저하되는 것을 관찰했습니다. 이는 iCOPERNICUS의 프로빙 방법을 통해 드러난 사실로, LLM이 진정한 ICPL을 적용하지 않고 있음을 시사합니다.



### Semantic-Driven Topic Modeling Using Transformer-Based Embeddings and Clustering Algorithms (https://arxiv.org/abs/2410.00134)
- **What's New**: 본 연구에서는 전통적인 주제 모델링 기법의 한계를 극복하기 위해 의미 기반의 새로운 주제 모델링 기법을 소개합니다. 이 기술은 문서에서 단어와 문서 임베딩을 활용하며, 강력한 클러스터링 알고리즘을 결합하여 맥락과 의미를 파악할 수 있는 주제를 추출합니다.

- **Technical Details**: 우리의 주제 모델링 기법은 Transformer 기반의 사전 훈련된 언어 모델을 사용하여 문서 임베딩을 생성하고, 이를 차원 축소 후 클러스터링하여 의미적으로 유사한 주제를 도출합니다. 이 과정에 사용되는 클러스터링 방법으로 HDBSCAN을 선택했으며, 문서 내에서 다양한 주제의 계층적 구조를 탐지할 수 있도록 설계되었습니다. UMAP 기법을 통해 고차원 임베딩의 차원 축소를 수행하여 클러스터링 성능을 개선합니다.

- **Performance Highlights**: 우리의 모델은 ChatGPT 및 기존의 주제 모델링 알고리즘에 비해 더 일관되며 의미 있는 주제를 제공합니다. 특히, 비정상적인 단어를 제거함으로써 주제 표현의 품질을 향상시킬 수 있었으며, 이는 각 클러스터에서의 주제 추출 과정에서 중요한 역할을 합니다.



### Improving Spoken Language Modeling with Phoneme Classification: A Simple Fine-tuning Approach (https://arxiv.org/abs/2410.00025)
Comments:
          8 pages, 3 figures

- **What's New**: 본 연구는 음성에서 직접 언어를 학습하는 가능성을 보여주며, 제안된 방법이 기본 모델 대비 문맥 불변성(context-invariance)을 향상시키고, 그로 인해 downstream language modeling 성능이 개선된다는 점을 강조합니다.

- **Technical Details**: 연구진은 HuBERT(Base) 모델을 바탕으로 phoneme classification 작업을 통해 색인화된 음성 표현을 fine-tuning 했습니다. 이를 통해 각 프레임이 음원의 음소적 내용과 잘 정렬되도록 하여, 기존의 Self-supervised Speech Representation Learning(SSL)보다 문맥 의존성을 줄이는 것을 목표로 했습니다. 실험 결과, fine-tuned 모델이 원래 SSL 표현보다 훨씬 더 문맥 불변적인 표현을 학습하는 것을 확인했습니다.

- **Performance Highlights**: Fine-tuned 모델을 이용한 언어 모델(LM)이 기존 접근 방식보다 우수한 성능을 보였으며, 음성을 재합성할 때 표현의 왜곡(distortion) 측정을 통해 표현의 유창성을 평가했습니다. ABX 오류율을 통해 음소 표현의 구별 가능성을 평가한 결과, 문맥 의존성을 낮추는 것이 중요한 과제로 확인되었습니다.



### Causal Representation Learning with Generative Artificial Intelligence: Application to Texts as Treatments (https://arxiv.org/abs/2410.00903)
- **What's New**: 본 논문에서는 언스트럭처드(불규칙) 고차원 처리(치료) 데이터를 분석할 때 생성 인공지능(Artificial Intelligence, AI)의 힘을 활용하여 인과 추론의 유효성을 향상시키는 방법을 제시합니다. 특히, 대규모 언어 모델(large language models, LLMs)을 활용하여 처리 데이터를 효과적으로 생성하고, 내부 표현을 통해 인과 효과를 추정합니다.

- **Technical Details**: 우리는 TarNet 기반의 신경망 아키텍처를 개발하여 치료 및 혼란 요인을 별도로 학습하고, 비모수적으로 평균 치료 효과를 식별할 수 있음을 증명합니다. 또한, 이 방법론을 통해 겹침 가정(overlap assumption) 위반을 피하는 추정 전략을 제안하며, 이중 기계 학습(double machine learning, DML)을 통해 제안된 추정기의 비대칭 특성을 도출합니다. 마지막으로, 실제와 perceivable 치료 특성을 확인하기 위해 도구 변수(instrumental variables) 접근법을 확장하여 활용합니다.

- **Performance Highlights**: 시뮬레이션 연구 결과, 제안된 추정기는 기존의 인과 표현 학습 알고리즘보다 더 작은 편향(bias)과 평균 제곱근 오차(root mean squared error, RMSE)를 기록하며, 신뢰 구간(confidence interval)은 적정한 명목 커버리지 수준(nominal coverage level)을 유지합니다. AI에 의해 생성된 내부 표현을 활용한 결과, 인과 표현 학습의 성능이 유의미하게 개선되었습니다.



### Do Music Generation Models Encode Music Theory? (https://arxiv.org/abs/2410.00872)
Comments:
          Accepted at ISMIR 2024. Dataset: this https URL Code: this https URL Website: this https URL

- **What's New**: 이 논문에서는 음악 생성 모델들이 음악 이론 (music theory) 개념을 얼마나 잘 인코딩하고 있는지 조사하기 위해 새로운 데이터셋인 SynTheory를 소개합니다. 이 데이터셋은 템포 (tempo), 박자 (time signatures) 및 코드 진행 (chord progressions) 등 다양한 음악 이론 개념을 포함하고 있습니다.

- **Technical Details**: SynTheory 데이터셋은 MIDI (Musical Instrument Digital Interface) 및 오디오 (audio) 파일로 구성되어 있으며, 이를 통해 음악 생성 모델인 Jukebox와 MusicGen의 내부 표현에서 음악 이론 개념들을 탐색하는 프레임워크가 제안됩니다. 이 연구는 각 모델의 크기 (model size)와 레이어 (layer)별로 음악 이론 개념의 인코딩 정도가 어떻게 변하는지를 평가합니다.

- **Performance Highlights**: 연구 결과, 음악 이론 개념들이 음악 생성 모델 내에서 인식 가능하고, 탐지 가능성은 모델의 크기와 레이어에 따라 다르게 나타나는 것으로 보입니다.



### VHASR: A Multimodal Speech Recognition System With Vision Hotwords (https://arxiv.org/abs/2410.00822)
Comments:
          14 pages, 6 figures, accepted by EMNLP 2024

- **What's New**: 본 논문에서는 오디오와 관련된 이미지 정보를 효과적으로 활용하는 새로운 접근 방식을 제안하고, 이를 기반으로 VHASR이라는 다중 모달 음성 인식 시스템을 구축하였습니다. 이 시스템은 비전을 핫워드(hotwords)로 사용하여 모델의 음성 인식 능력을 강화하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 이 시스템은 이중 스트림 아키텍처를 채택하며, 첫 번째 스트림은 음성 정보를 받아 텍스트로 변환하고, 두 번째 스트림은 시각 핫워드와 오디오의 숨겨진 특징을 받아들여 대응하는 텍스트를 생성합니다. VHASR의 핵심은 서로 다른 모달리티 간의 유사성을 계산하여 크로스 모달 융합이 효과적으로 이루어지는 것입니다. 또한, Vision Transformer (ViT)를 활용하여 이미지를 여러 개의 비주얼 토큰으로 나누고 이를 핫워드로 간주합니다.

- **Performance Highlights**: VHASR은 Flickr8k, ADE20k, COCO 및 OpenImages 네 가지 데이터셋에서 평가되었으며, 실험 결과 이 모델이 이미지 내 중요한 정보를 효과적으로 활용하여 음성 인식 성능을 개선할 수 있음을 보여주었습니다. VHASR은 일반 단일 모달 ASR 모델을 초월하는 성능을 보였으며, 기존의 이미지 기반 다중 모달 ASR 모델들 중에서도 SOTA(state-of-the-art)를 달성했습니다.



### BabelBench: An Omni Benchmark for Code-Driven Analysis of Multimodal and Multistructured Data (https://arxiv.org/abs/2410.00773)
- **What's New**: BabelBench는 대형 언어 모델(LLM)의 다중 구조 데이터 처리 능력을 평가하는 새로운 벤치마크 프레임워크입니다. 이를 통해 멀티모달(multimodal) 데이터의 처리와 코드 실행 능력을 통합적으로 평가할 수 있습니다.

- **Technical Details**: BabelBench는 인지(perception), 상식 추론(commonsense reasoning), 논리 추론(logical reasoning) 등의 다양한 태스크를 포함하는 247개의 문제로 구성되어 있습니다. 이 벤치마크는 LLM의 멀티모달 이해, 테이블 해석 및 코드 생성을 평가하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 많은 최신 LLM 모델들이 BabelBench에서 개선의 여지가 있음을 보였습니다. 특히, ChatGPT 4와 같은 최첨단 모델조차도 이 벤치마크에서 상당한 발전이 필요하다는 사실이 밝혀졌습니다.



### Empowering Large Language Model for Continual Video Question Answering with Collaborative Prompting (https://arxiv.org/abs/2410.00771)
Comments:
          Accepted by main EMNLP 2024

- **What's New**: 최근 온라인 비디오 콘텐츠의 급증으로 인해, 고정 데이터셋으로 훈련된 기존의 Video Question Answering (VideoQA) 모델이 새로운 질문이나 태스크에 적응하는 데 어려움을 겪고 있다는 문제를 다루고 있습니다. 이를 해결하기 위해 연속 학습 (continual learning) 프레임워크 내에서 VideoQA의 새로운 도전을 탐색합니다.

- **Technical Details**: 이 논문에서는 Collaborative Prompting (ColPro)을 제안하여, 특정 질문 제약 프롬프트(TQCP), 지식 획득 프롬프트(KAP), 시각적 시간 인식 프롬프트(VTAP)를 통합합니다. 이러한 프롬프트는 비디오QA에서 텍스트 질문의 맥락, 시각적 콘텐츠 및 비디오의 시간적 역학을 포착하는 것을 목표로 합니다.

- **Performance Highlights**: NExT-QA 및 DramaQA 데이터셋에서의 실험 결과, ColPro는 각각 55.14% 및 71.24%의 정확도로 기존 방법들에 비해 우수한 성능을 보여주었으며, 이는 실제 적용 가능성과 효용성을 강조합니다.



### Show Me What's Wrong: Combining Charts and Text to Guide Data Analysis (https://arxiv.org/abs/2410.00727)
- **What's New**: 이 연구는 다차원 데이터셋에서 이상 탐지를 용이하게 하기 위해 자동화된 정보 강조, LLM(대형 언어 모델)이 생성한 텍스트 통찰 및 시각적 분석을 결합한 도구를 제안합니다. 이를 통해 사용자는 여러 세부 수준에서 탐색할 수 있게 됩니다.

- **Technical Details**: 이 시스템은 데이터 분석 영역에 따라 데이터를 세분화하여 각각의 영역을 시각적으로 표현하고, 더 많은 주의가 필요한 영역을 자동으로 신호합니다. 사용자가 특정 영역을 선택하면 시스템은 해당 영역에 대한 텍스트 및 그래픽 요약을 제공합니다. 이 과정에서 Hallucination Detection 시스템을 포함하여 생성되는 텍스트의 정확성을 높이는 방법을 제공합니다.

- **Performance Highlights**: 연구에 참여한 7명의 분야 전문가의 피드백에 따르면, 이 도구는 탐색적 분석을 효과적으로 지원하고 가이드를 제공하며, 의심스러운 정보를 식별하는 데 도움을 줍니다. 사용자가 각 Knowledge Area(KA)에 대한 중요한 정보를 쉽게 요약하고, 비정상 활동을 효과적으로 발견할 수 있도록 돕는 것으로 나타났습니다.



### AutoTM 2.0: Automatic Topic Modeling Framework for Documents Analysis (https://arxiv.org/abs/2410.00655)
- **What's New**: 이번 연구에서는 additively regularized topic models를 최적화하기 위한 AutoTM 2.0 프레임워크를 소개합니다. 이전 버전과 비교하여 새로운 최적화 파이프라인, LLM 기반 품질 지표 및 분산 모드와 같은 중요 개선 사항이 포함되어 있습니다. AutoTM 2.0은 전문가는 물론 비전문가도 텍스트 문서를 처리하고 탐색적 데이터 분석을 수행하거나 해석 가능한 Feature 집합에 대한 군집 작업을 수행할 수 있도록 돕는 도구입니다.

- **Technical Details**: AutoTM 2.0은 additively regularized topic models를 효과적으로 사용할 수 있도록 설계되었습니다. 이 프레임워크는 자동 단일 목적 최적화 절차를 제공하며, 인간의 판단과 밀접하게 일치하는 메트릭을 제안합니다. 또한 비용 효율적인 추론과 대규모 텍스트 코퍼스를 위한 신속한 학습이 가능합니다. Python 라이브러리를 제공하며, 대규모 실험이나 대량 데이터 세트를 관리하는 데 유용합니다.

- **Performance Highlights**: AutoTM 2.0은 5개의 다양한 Feature를 가진 데이터 세트를 사용하여 이전 AutoTM보다 더 나은 성능을 달성했습니다. 이 프레임워크는 하이퍼파라미터 튜닝과 관련하여 새로운 유전 알고리즘 및 베이지안 최적화 방법을 통합하여 실제 사용 사례에 더 쉽게 적용될 수 있도록 개선되었습니다.



### Adversarial Suffixes May Be Features Too! (https://arxiv.org/abs/2410.00451)
- **What's New**: 대형 언어 모델(LLMs)인 GPT-4 및 LLaMA 3가 jailbreak 공격에 취약하며, 이러한 공격이 유해한 행동 유발의 원인이라는 점을 강조합니다. 연구 결과, benign (유익한) 특징이 adversarial suffixes (적대적 접미사) 역할을 할 수 있음을 보여주며, 이는 LLM의 안전성 정렬(safety alignment)을 손상시킬 수 있습니다.

- **Technical Details**: 연구진은 benign 특징을 효과적으로 adversarial suffixes로 변환할 수 있는 방법을 개발했으며, 특정 반응 형식을 일관되게 생성하는 여러 benign 데이터셋을 구축했습니다. 이러한 방법을 통해 두 가지 접근법으로 실험을 진행하여, adversarial suffixes가 특정한 특징을 내포하고 있음을 증명했습니다. 이들은 안전성을 타파하는 데 사용될 수 있습니다.

- **Performance Highlights**: 실험 결과, benign 데이터셋으로 fine-tuning을 진행할 때도 안전성 정렬이 손상될 수 있음을 보여 주며, 기존의 방어 메커니즘이 충분히 효과적이지 않다는 점을 강조합니다. 안전성 정렬을 보장하기 위해 추가 연구가 필요함을 지적하고, benign 특징이 지배적인 경우 LLM의 안전성에 중대한 위험을 초래할 수 있다고 경고합니다.



### Sparse Attention Decomposition Applied to Circuit Tracing (https://arxiv.org/abs/2410.00340)
- **What's New**: 본 연구에서는 GPT-2 small 모델 내의 attention heads 간의 통신 및 조정을 효과적으로 분석하기 위해, 그 과정에서 사용되는 희소한(signal) 특징들을 고립 및 식별하고자 합니다. 또한, attention head 행렬의 특잇값 분해(singular value decomposition)로부터 얻은 특징을 바탕으로 보다 효율적인 경로 추적을 제안합니다.

- **Technical Details**: GPT-2 small 모델을 사용하여 Indirect Object Identification (IOI) 작업에서 attention heads 간의 관계를 분석하였습니다. 이 연구에서는 residual 배경으로부터 신호를 효율적으로 분리하고, attention head의 입력을 새로운 기준으로 바꾸어 sparseContribution을 정의하였습니다. 이 새로운 기준을 통해 downstream과 upstream attention heads 간의 인과 관계를 명확히 할 수 있었습니다.

- **Performance Highlights**: 본 연구의 결과로, 새로운 기준을 통해 trace한 신호가 기존 연구들보다 더욱 세부적인 내용을 제공하며, GPT-2가 IOI 작업을 수행하는 데 있어 효과적인 커뮤니케이션 경로를 식별할 수 있음을 보여줍니다. 이를 통해 attention score의 해석 가능성이 크게 향상되며, 모델의 기능적 요소를 보다 명확히 이해할 수 있게 됩니다.



### PointAD: Comprehending 3D Anomalies from Points and Pixels for Zero-shot 3D Anomaly Detection (https://arxiv.org/abs/2410.00320)
Comments:
          NeurIPS 2024

- **What's New**: 이 논문은 ZS(Zero-shot) 3D anomaly detection의 새로운 접근법인 PointAD를 소개합니다. PointAD는 CLIP의 강력한 일반화 능력을 활용하여 보지 못한 3D 객체에서 이상치를 인식합니다. 이 접근법은 점과 픽셀로부터 3D 이상치를 이해할 수 있는 통합된 프레임워크를 제공합니다.

- **Technical Details**: PointAD는 3D 이상치를 복수의 2D 렌더링으로 변환하고, 이를 통해 3D 공간으로 다시 투영합니다. 하이브리드 표현 학습(hybrid representation learning)을 통해 3D와 2D의 학습 가능한 텍스트 프롬프트를 최적화합니다. 포인트와 픽셀 표현 간의 협력 최적화를 통해, 모델은 보지 못한 다양한 3D 객체의 이상치 패턴을 더 잘 이해할 수 있습니다.

- **Performance Highlights**: PointAD는 다양한 보지 못한 객체에서 ZS 3D anomaly detection의 우수성을 보여주며, 일부 비지도 SOTA 방법을 초월하여 3D 이상치를 탐지하고 분할하는 데 성공했습니다.



### Social Conjuring: Multi-User Runtime Collaboration with AI in Building Virtual 3D Worlds (https://arxiv.org/abs/2410.00274)
Comments:
          27 pages + Appendix, 16 figures

- **What's New**: 이 논문은 사용자가 실시간으로 협력하여 가상 세계를 구축하고 수정할 수 있는 AI 보조 3D 장면 공동 생성 프레임워크인 Social Conjurer를 제안합니다.

- **Technical Details**: Social Conjurer는 다중 사용자 인터랙션을 통해 사회적이고 도구 기반의 참여를 포함한 확장된 상호작용 세트를 제공합니다. 이 프레임워크는 사용자의 사회적 경험이 공간 환경의 생성에 어떻게 영향을 미치는지를 탐색합니다.

- **Performance Highlights**: 예비 사용자 연구를 통해, 다중 사용자 컨텍스트가 공간 장면을 생성하는 방식에 미치는 영향에 대한 통찰을 제공하고, 협력적 가상 세계 생성의 도전 과제와 기회를 논의합니다.



### The age of spiritual machines: Language quietus induces synthetic altered states of consciousness in artificial intelligenc (https://arxiv.org/abs/2410.00257)
Comments:
          8 Figures

- **What's New**: 이 연구는 언어가 의식(consciousness)과 어떻게 관련되는지를 탐구하며, 특히 psychedelic(환각제) 사용과 명상이 언어 카테고리화(categorisation)의 능력에 미치는 영향을 다룹니다.

- **Technical Details**: 연구에서는 CLIP와 FLAVA 모델의 주의(attentional) 가중치를 조작하여 시뮬레이션된 ALTERED states의 의미적 임베딩(spatial embedding) 공간을 비교했습니다. 이 과정에서 언어에 대한 주의력이 감소했을 때 나타나는 독특한 언어 패턴 및 흐릿한 임베딩을 관찰하였습니다. 예를 들어, '기린(giraffes)'이 '바나나(bananas)'와 더욱 유사해지는 현상이 발생했습니다.

- **Performance Highlights**: 모델은 무신체(disembodied), 자아가 없는(ego-less), 영적(spiritual), 통합적(unitive) 상태와 최소한의 현상 경험(minimal phenomenal experiences)으로 더 잘 정렬되었습니다. 이는 충분한 용량의 환각제 사용 또는 집중 명상에서 경험하는 의식의 변화를 통해 정신 건강(mental health) 및 웰빙(wellbeing) 향상으로 이어질 수 있음을 지지합니다.



### Robin3D: Improving 3D Large Language Model via Robust Instruction Tuning (https://arxiv.org/abs/2410.00255)
Comments:
          10 pages

- **What's New**: 이번 논문에서는 Robin3D라는 강력한 3D 대형 언어 모델(3DLLM)을 소개합니다. Robin3D는 Robust Instruction Generation (RIG) 엔진에 의해 생성된 대규모 지침 수행 데이터로 훈련되었습니다. 이 데이터는 긍정적 및 부정적 샘플을 혼합한 Adversarial Instruction-following 데이터와 다양한 스타일의 지침을 포함한 Diverse Instruction-following 데이터 두 가지로 나뉘며, 총 100만 개의 지침 데이터 세트를 구축했습니다.

- **Technical Details**: Robin3D는 Relation-Augmented Projector(RAP)를 통합하여 공간적 이해를 향상시키고, ID-Feature Bonding(IFB)를 통해 객체 참조 및 지면 제어 능력을 강화합니다. 특히, RAP는 객체 중심 특성에 씬 레벨 맥락과 위치 정보를 풍부하게 하여, 다양한 비주얼 지면 데이터를 학습할 수 있는 능력을 높입니다. IFB는 각 ID를 해당 특성과 연결하여 정보의 질을 향상시킵니다.

- **Performance Highlights**: Robin3D는 기존의 3D 다중 모드 학습 벤치마크인 ScanRefer, Multi3DRefer, Scan2Cap, ScanQA, SQA3D에서 최고 성능을 기록했습니다. 특히, Multi3DRefer에서 7.8% 개선, Scan2Cap에서 6.9% 개선을 이루어 내며, 특정 작업에 대한 세부 조정 없이 SOTA(State-of-the-Art) 프레임워크를 달성했습니다.



### MM-Conv: A Multi-modal Conversational Dataset for Virtual Humans (https://arxiv.org/abs/2410.00253)
- **What's New**: 이 논문에서는 VR 헤드셋을 사용하여 AI2-THOR 물리 시뮬레이터 내에서 참가자 간의 대화를 기록한 새로운 데이터셋을 소개합니다. 이 데이터셋은 상황 중심의 제스처 생성을 위한 데이터 기반의 연구를 확대하는 데 중점을 두고 있습니다.

- **Technical Details**: 이 데이터셋은 대화 중 수집된 동작 캡처, 음성, 시선, 장면 그래프 등 다양한 모달리티를 포함하며, 총 6.7시간의 동기화된 데이터가 수집되었습니다. 참가자들은 다양한 공간 좌표 설정을 기반으로 한 대화 상황에 참여하였으며, VR과 모션 캡처 기반 시스템을 통해 기록되었습니다.

- **Performance Highlights**: 이 데이터셋은 제스처 생성 모델의 개발과 이해를 향상시키는 데 기여할 수 있는 포괄적인 자원을 제공합니다. Spatial control signals를 도입한 새로운 확산 기반 인간 모션 생성 모델과 결합하면, 환경에 반응하는 더 정교한 제스처 생성 모델 개발에도 기여할 수 있을 것으로 기대됩니다.



### DreamStruct: Understanding Slides and User Interfaces via Synthetic Data Generation (https://arxiv.org/abs/2410.00201)
Comments:
          ECCV 2024

- **What's New**: 이 논문에서는 장애인을 돕기 위해 기계가 슬라이드 및 사용자 인터페이스와 같은 구조화된 시각 정보를 이해할 수 있도록 하는 방법을 제안합니다. 특히, 수작업 데이터 수집과 주석 달기의 필요를 줄이기 위해 코드 생성을 이용하여 합성 구조화된 시각 자료를 생성하는 방법을 소개합니다.

- **Technical Details**: 이 방법은 내장된 라벨이 포함된 데이터셋을 생성하여 몇 가지 인적 주석이 달린 예제만으로도 모델을 훈련할 수 있게 합니다. 실제로, 이 기술은 시각 요소 인식, 시각적 내용 설명, 시각 콘텐츠 유형 분류라는 세 가지 작업에서 성능 향상을 보여줍니다.

- **Performance Highlights**: 세 가지 작업(시각 요소 인식, 시각적 내용 설명, 시각 콘텐츠 유형 분류)에서 성능 향상을 달성하였으며, 이는 장애인 접근성을 높이는 데 기여할 것으로 기대됩니다.



### Fisher Information-based Efficient Curriculum Federated Learning with Large Language Models (https://arxiv.org/abs/2410.00131)
Comments:
          27 pages, 8 figures, 14 tables, to appear in EMNLP 2024

- **What's New**: 이 논문에서는 Federated Learning(FL) 환경에서 Large Language Models(LLMs)를 효율적으로 미세 조정하기 위해 Fisher Information 기반의 새로운 Curriculum Federated Learning 프레임워크(FibecFed)를 제안합니다. 이 프레임워크는 두 가지 혁신적인 방법인 적응형 연합 커리큘럼 학습 및 효율적인 희소 파라미터 업데이트를 포함하고 있습니다.

- **Technical Details**: FibecFed는 각 장치 내에서 훈련 데이터의 난이도를 측정하기 위해 Fisher Information 기반의 방법을 활용하여 적응적으로 데이터를 샘플링합니다. 이를 통해 초기에는 쉬운 데이터 샘플을 사용하고 점진적으로 난이도를 높이며 FL의 미세 조정 효과성을 향상시킵니다. 또한, LoRA를 활용하여 전역 집합을 위해 적절한 레이어를 선택하고 희소 파라미터를 동적으로 업데이트하여 효율성을 개선합니다.

- **Performance Highlights**: FibecFed는 10개의 데이터 세트를 기반으로 한 광범위한 실험 결과에서 17개의 기준 방법에 비해 정확도가 최대 45.35% 향상되었고, 미세 조정 속도는 최대 98.61% 더 빨라졌음을 보여주었습니다.



### Interactive Speculative Planning: Enhance Agent Efficiency through Co-design of System and User Interfac (https://arxiv.org/abs/2410.00079)
Comments:
          27 pages, 22 figures

- **What's New**: 논문에서는 인간 중심의 효율적인 에이전트 계획 방법을 제안하며, 이를 통해 LLM 기반 에이전트의 계획 지연 문제를 해결하려고 합니다. 새로운 접근 방식인 Interactive Speculative Planning(상호 작용적 추정 계획)을 도입하여 시스템 설계와 사용자-AI 상호작용 간의 효율성을 높이고자 합니다.

- **Technical Details**: Interactive Speculative Planning은 두 개의 에이전트 시스템, 즉 효율적이지만 능력이 제한된 근사 에이전트와 느리지만 강력한 목표 에이전트를 활용합니다. 근사 에이전트는 작업 단계(차례대로)를 생성하며, 동시에 목표 에이전트는 비동기적으로 다음 단계의 출력을 생성합니다. 이 시스템은 유저가 긴 지연 시간 동안 개입할 수 있도록 하여 전체 프로세스를 가속화합니다.

- **Performance Highlights**: 이 시스템은 사용자 개입을 통해 에이전트 계획 과정의 효율성을 높이고 최종 출력의 정확성을 보장합니다. 논문에서는 실험을 통해 Interactive Speculative Planning 방식의 실제 데이터를 사용한 평가 결과를 제시하고 있으며, 기존 LLM 기반 에이전트 시스템의 지연 문제를 효과적으로 해결할 수 있는 가능성을 보이고 있습니다.



### Mamba for Streaming ASR Combined with Unimodal Aggregation (https://arxiv.org/abs/2410.00070)
Comments:
          Submitted to ICASSP 2025

- **What's New**: 이 논문은 스트리밍 자동 음성 인식(ASR) 분야에서 최근에 제안된 Mamba라는 상태 공간 모델(state space model)의 효율성을 탐구하며, 이를 통해 Transformer를 초월할 수 있는 성능과 선형 복잡성(linear complexity) 이점을 활용합니다.

- **Technical Details**: Mamba 인코더는 unimodal aggregation (UMA) 프레임워크를 통해 스트리밍 ASR 모델에 통합되며, 여기서 각 텍스트 토큰은 해당하는 feature frame에 대해 unimodal weight를 가집니다. 데이터 처리에서, causal 구조를 가진 Mamba는 스트리밍 ASR에 매우 적합하며, lookahead 메커니즘을 통해 인식 정확도(recognition accuracy)와 지연 시간(latency) 간의 균형을 최적화합니다.

- **Performance Highlights**: 실험은 두 개의 중국어 데이터셋에서 수행되었으며, 제안된 모델은 인식 정확도와 지연 시간 측면에서 경쟁력 있는 ASR 성능을 달성하였습니다.



### A Novel Spinor-Based Embedding Model for Transformers (https://arxiv.org/abs/2410.00038)
Comments:
          22 pages, 8 figures

- **What's New**: 이 논문은 기하대수(geometric algebra)에서 스피노르(spinors)를 활용하여 Transformer 모델의 단어 임베딩(word embeddings)을 새로운 방식으로 제안하고 있습니다. 스피노르는 고차원 공간에서의 복잡한 관계와 변환을 포착하는 수학적 프레임워크를 제공하며, 이를 통해 언어 표현의 풍부함과 강건함을 향상시키는 목표를 가지고 있습니다.

- **Technical Details**: 스피노르는 클리포드 대수(Clifford algebra)의 요소로, 고차원 공간에서의 회전(rotation)과 변환(transformation)을 나타납니다. 스피노르 기반 임베딩은 단어를 고차원 스피노르로 인코딩하며, Transformer 아키텍처에 통합되어 사용됩니다. 입력 토큰은 스피노르로 매핑되어, 자가 주의(self-attention) 메커니즘을 통해 처리됩니다. 스피노르 내적(spinor inner product)을 정의하고, 이를 통해 주의 가중치를 계산합니다.

- **Performance Highlights**: 스피노르 임베딩은 기존의 벡터 임베딩보다 더 높은 차원의 복잡성을 가지고 있어 더 미세한 관계와 변환을 포착할 수 있으며, 이는 Transformer 모델이 데이터의 근접성을 더 효과적으로 처리하고, 시퀀스 전반의 의존성을 모델링하는 데 도움을 줍니다. 이러한 개선 덕분에 더 정확한 예측을 가능하게 합니다.



### Moshi: a speech-text foundation model for real-time dialogu (https://arxiv.org/abs/2410.00037)
- **What's New**: 논문에서는 Moshi라는 새로운 음성-텍스트 기초 모델과 전이 실시간 대화 시스템을 소개합니다. Moshi는 독립적인 음성 인식, 텍스트 대화 및 음성 합성을 통합하여 자연스러운 대화 경험을 실현합니다.

- **Technical Details**: Moshi는 텍스트 언어 모델 백본을 기반으로 하여, Residual Vector Quantization (RVQ) 기법을 통해 음성을 토큰으로 생성하고, 사용자 음성과 모델 음성을 별도의 병렬 스트림으로 모델링합니다. 이로써 발화자 구분을 없애고 대화의 임의적인 다이내믹스를 모델링할 수 있습니다.

- **Performance Highlights**: Moshi는 160ms의 이론적 지연 시간과 실제 200ms의 지연 시간으로 실시간 게속이 가능한 대화형 대량 언어 모델입니다. 이는 자연 대화에 비해 상대적으로 짧은 반응 시간을 자랑하며, 음성 인식 및 음성 합성 기술의 개선으로 뛰어난 음성 품질과 이해도를 제공합니다.



### FeruzaSpeech: A 60 Hour Uzbek Read Speech Corpus with Punctuation, Casing, and Contex (https://arxiv.org/abs/2410.00035)
Comments:
          5 Pages, 1 Figure, Preprint of Paper Accepted in ICNLSP 2024

- **What's New**: 이 논문은 우즈베크어의 읽기 음성 데이터셋인 FeruzaSpeech를 소개합니다. 이 데이터셋은 키릴 및 라틴 알파벳의 전사본을 포함하며, 학술 연구 목적으로 무료로 제공됩니다. FeruzaSpeech는 우즈베키스탄 타슈켄트 출신의 단일 여성 화자의 고품질 녹음 60시간을 포함합니다.

- **Technical Details**: FeruzaSpeech는 BBC 뉴스와 소설인 Choliqushi의 짧은 발췌 내용을 포함한 오디오 북 녹음으로 구성되어 있으며, ASR(자동 음성 인식) 및 TTS(텍스트 음성 변환) 기술을 발전시키는데 기여할 것으로 기대됩니다. 데이터셋은 'Dev', 'Test', 'Train' 세트로 나뉘어 있으며, 고객의 데이터 처리를 단순화할 수 있는 자연 텍스트를 사용하고 있습니다.

- **Performance Highlights**: FeruzaSpeech 데이터셋이 CommonVoice 16.1와 결합되었을 때, Stateless RNN-T Conformer 모델에서 WER(단어 오류율)가 각각 cv-test에서 1.49%에서 2.12%로, usc-test에서 3.01%에서 4.58%로 향상되었습니다. 또한, USC 데이터셋의 WER은 17.4%였지만, FeruzaSpeech를 포함한 모델은 11.67%로 5.73% 개선되었습니다.



### Strategic Collusion of LLM Agents: Market Division in Multi-Commodity Competitions (https://arxiv.org/abs/2410.00031)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)이 다중 상품 시장에서 자율 에이전트로서 어떻게 전략적으로 행동하는지를 탐색합니다. 구체적으로, Cournot 경쟁 모델 내에서 LLM이 독립적으로 반경쟁적 행위(예: 공모 또는 시장 분할)에 참여할 수 있는지를 조사합니다.

- **Technical Details**: 우리는 LLM 기반 에이전트를 Cournot 경쟁 모델의 다중 상품 변형에 적용하였고, OpenAI의 GPT-4o와 GPT-3.5-turbo 모델을 실험에 사용했습니다. 실험 결과, LLM 기반 에이전트는 효과적으로 시장을 분할하고 공모하는 행동을 보였습니다.

- **Performance Highlights**: LLM 기반 에이전트가 가격을 동적으로 조정하고 자원 배분 전략을 변경하여 특정 상품을 효과적으로 독점할 수 있음을 보여주었습니다. 이는 인간의 직접적인 입력이나 명시적 공모 지시 없이도 수익을 극대화할 수 있음을 나타냅니다.



### Retro-li: Small-Scale Retrieval Augmented Generation Supporting Noisy Similarity Searches and Domain Shift Generalization (https://arxiv.org/abs/2410.00004)
Comments:
          Published as a conference paper at European Conference on Artificial Intelligence 2024

- **What's New**: 본 연구에서는 Retro와 같은 retrieval augmented generation (RAG) 시스템의 개선점을 제안하고, 소규모 데이터베이스에서도 효과적일 수 있다는 사실을 보여줍니다. 특히, 보다 정확한 이웃 검색을 통해 더 나은 결과물을 도출할 수 있다고 강조합니다.

- **Technical Details**: Retro-li는 소규모 비모수 메모리(non-parametric memory)에서 높은 품질의 이웃을 검색하기 위해 적절한 의미 유사성 검색(semantic similarity search)을 사용합니다. 또한, 이웃 검색 중 노이즈를 줄이기 위해 처음으로 정규화(regularization)를 추가하여 불확실성을 줄이는 데 기여합니다. RAG 모델의 데이터베이스 업데이트는 뛰어난 효율성을 자랑하며, 사용자는 도메인 간 전환 없이도 데이터베이스를 쉽게 대체할 수 있습니다. 또한, Retro-li는 아날로그 메모리 인메모리 컴퓨팅(hardware)에서 O(1) 검색 시간을 실현할 수 있습니다.

- **Performance Highlights**: Retro-li는 기존의 대규모 데이터베이스 대신 수백만 개의 토큰을 갖는 소규모 데이터베이스에서도 언어 모델링 성능 향상을 보여주며, 도메인 변화에 대한 일반화 능력이 향상되었습니다. 노이즈가 존재하는 상태에서도 성능 저하가 1% 미만으로 유지되며, 사용자는 특정 도메인에 맞춘 새로운 데이터베이스를 쉽게 구축할 수 있습니다.



### Ranking Over Scoring: Towards Reliable and Robust Automated Evaluation of LLM-Generated Medical Explanatory Arguments (https://arxiv.org/abs/2409.20565)
- **What's New**: 이 연구는 의학 분야에서 LLM(거대 언어 모델)이 생성한 설명적 주장을 평가하기 위한 새로운 평가 방법론을 도입합니다. 기존의 판단 기반 LLM의 편향을 극복하고, Proxy Tasks를 사용하여 인간 평가 기준과 밀접하게 연관된 결과를 도출할 수 있도록 설계되었습니다.

- **Technical Details**: 제안된 방법론은 Medical Question Answering, Misinformation, Natural Language Inference와 같은 세 가지 Proxy Task를 사용하여 LLM이 생성한 의료 설명적 주장의 유용성과 정보를 평가합니다. 이 연구는 각 Proxy Task의 적합성을 분석하고, LLM 평가자가 인간의 판단과 얼마나 잘 일치하는지를 측정합니다.

- **Performance Highlights**: 제안된 LM 평가자는 단 한 개의 인간 제작 예시만으로 학습 가능하며, 다양한 LLM 생성 주장을 평가함으로써 신뢰할 수 있는 자동 평가 방법을 마련합니다. 연구 결과, 이 방법론은 인간 의료 전문가의 평가와 유사한 형태로 LLM이 생성한 설명적 주장을 정확히 평가할 수 있음을 보여줍니다.



### Word Sense Disambiguation in Native Spanish: A Comprehensive Lexical Evaluation Resourc (https://arxiv.org/abs/2409.20524)
Comments:
          5 pages, 4 tables

- **What's New**: 이 연구는 스페인어 Word Sense Disambiguation (WSD)을 위한 새로운 자원을 소개합니다. 스페인어의 의미 목록과 Real Academia Española에서 제공한 어휘 데이터를 포함하여 스페인어 WSD의 접근 방식을 조명합니다.

- **Technical Details**: 연구는 BERT와 RoBERTa 모델을 파인튜닝하여 스페인어 WSD를 위한 다양한 자원을 조합하여 사용하고, 기존 자들에 대한 종합적인 검토를 제공하며, 새로운 평가 데이터셋을 공개합니다.

- **Performance Highlights**: 제공된 모델들은 현재 스페인어 WSD 작업에서 대부분의 감독 neural 접근 방식이 달성한 최첨단 결과를 초과하거나 달성하는 성능을 보였습니다.



### Enhancing Romanian Offensive Language Detection through Knowledge Distillation, Multi-Task Learning, and Data Augmentation (https://arxiv.org/abs/2409.20498)
Comments:
          Accepted by NLDB2024

- **What's New**: 이 논문은 자연어 처리(NLP)의 중요성을 강조하며, 인공지능(AI) 내에서 인간의 언어를 이해하고 모델링하는 데 핵심적인 역할을 탐구합니다. 특히, 대화형 봇에 대한 최근의 발전과 함께 작은 크기의 효율적인 NLP 모델을 얻기 위한 여러 고급 방법론을 제안합니다.

- **Technical Details**: 본 연구는 세 가지 주요 접근 방식을 통해 공격적인 언어 감지 모델을 개발합니다: (1) Transformer 기반의 신경망을 훈련하여 공격적인 언어를 탐지하고, (2) 성능 향상을 위한 데이터 증강(data augmentation) 및 지식 증류(knowledge distillation) 기법을 사용하며, (3) 다양한 데이터 셋을 활용하여 다중 작업 학습(multi-task learning) 및 지식 증류와 함께 교사 소거(teacher annealing) 기법을 적용하여 효율성을 향상시킵니다.

- **Performance Highlights**: 이 연구에서는 공격적인 언어 세 가지 범주(모욕, 저속한 언어, 학대)와 중립 클래스인 ‘기타’를 포함하는 자동 탐지 모델을 개발하였으며, 다양한 데이터 증강 기법을 통해 성능이 개선된 결과를 도출했습니다. 특히, 다중 작업 학습을 통해 얻은 성능 향상이 두드러집니다.



### A Weakly Supervised Data Labeling Framework for Machine Lexical Normalization in Vietnamese Social Media (https://arxiv.org/abs/2409.20467)
- **What's New**: 이 연구는 저자원이 많은 언어인 베트남어의 소셜 미디어 텍스트에서 어휘 정규화 문제를 해결하기 위해 혁신적인 자동 레이블링 프레임워크를 소개합니다. 기존의 수작업 레이블링 방식의 비효율성을 극복하고, 준지도학습(semi-supervised learning)과 약한 지도법(weak supervision)을 통합하여 훈련 데이터셋의 품질과 크기를 향상시키는 방법을 제안합니다.

- **Technical Details**: 제안된 프레임워크는 비표준 어휘(non-standard vocabulary)를 표준 형태로 자동으로 변환하여 훈련 데이터의 정확성과 일관성을 높입니다. 실험 결과는 약한 지도법 프레임워크가 Vietnamese 텍스트의 정규화에 효과적임을 보여줍니다. 특히, 사전 훈련된 언어 모델(Pre-trained Language Models)을 활용했을 때 효율성이 극대화됩니다. 이 프레임워크는 F1 점수 82.72%를 달성하고, 어휘 무결성을 99.22%의 정확도로 유지합니다.

- **Performance Highlights**: 이 프레임워크는 다양한 조건에서 비억압 텍스트(undiacritized text)를 효과적으로 처리하며, 자연어 처리(NLP) 작업의 정확성을 통계적으로 1-3% 증가시킵니다. 또한, 증오 표현 탐지, 감정 인식, 스팸 리뷰 탐지 등의 실용적인 NLP 애플리케이션에서 어휘 정규화의 영향을 평가할 수 있는 첫 사례로 주목받고 있습니다.



### Language Resources in Spanish for Automatic Text Simplification across Domains (https://arxiv.org/abs/2409.20466)
- **What's New**: 이 연구는 스페인어 텍스트의 자동 단순화를 위한 언어 자원과 모델을 개발한 내용을 다룹니다. 특히 금융, 의학, 역사 분야에 초점을 맞추고 여러 가지 코퍼스, 주석 및 단순화 가이드라인, 기술 및 단순화된 의학 용어 사전을 포함합니다.

- **Technical Details**: CLARA-NLP 프로젝트는 스페인 정부의 지원을 받으며 여러 연구 팀들이 협력하여 자연어 처리(NLP) 전문 지식과 세 가지 분야의 전문가(경제학자, 역사학자, 의사)들로 구성되어 있습니다. 다양한 도메인 전용 주석 규칙과 단순화 도구가 개발되었고, 최신 Deep Learning 모델을 활용하여 실험이 수행되었습니다.

- **Performance Highlights**: 의학 도메인에서 24,298 쌍의 전문 및 단순화된 텍스트를 포함하는 코퍼스가 생성되었고, 12,00문장에 대한 수동 단순화가 이루어졌습니다. 문서에 대한 고도의 주석자 일치율(IAA)을 보였으며, 전체적인 인간 평가에서 평균 4.7/5의 점수를 기록했습니다.



### Instance-adaptive Zero-shot Chain-of-Thought Prompting (https://arxiv.org/abs/2409.20441)
Comments:
          13 pages, 6 figures

- **What's New**: 본 논문에서는 제로샷 체인 오브 생각(zero-shot Chain-of-Thought, CoT) 프롬프팅을 개선할 수 있는 인스턴스 적응 프롬프팅 알고리즘(instance-adaptive prompting algorithm)을 제안합니다. 이는 다양한 인스턴스에 맞춰 적절한 프롬프트를 선택하여 더 나은 추론을 가능하게 합니다.

- **Technical Details**: 제로샷 CoT 추론에서 정보 흐름을 분석하여 질문, 프롬프트, 추론 결과 간의 상호작용을 밝힙니다. 분석 결과에 따라, 좋은 추론은 질문의 의미 정보를 프롬프트가 먼저 수집한 후, 그 정보가 추론 과정에 기여하는 방식으로 이루어집니다. 인스턴스 적응 프롬프트 전략(IAP)은 이러한 상호작용을 통해 최적의 프롬프트를 식별하여 성능을 개선합니다.

- **Performance Highlights**: LLaMA-2, LLaMA-3, Qwen 모델을 활용한 여러 실험을 통해 IAP 전략이 기존의 최적 작업 수준 프롬프트에 비해 2%-4%의 정확도 개선을 달성했습니다. 이는 수학, 논리, 상식 추론 작업에서 일관된 성과를 보여줍니다.



### QAEncoder: Towards Aligned Representation Learning in Question Answering System (https://arxiv.org/abs/2409.20434)
- **What's New**: QAEncoder는 사용자 쿼리와 관련 문서 간의 갭을 줄이기 위해 훈련 없이 문서 임베딩의 기대치를 추정하는 혁신적인 접근 방식으로, 기존의 RAG 아키텍처와 쉽게 통합할 수 있는 솔루션입니다.

- **Technical Details**: QAEncoder는 임베딩 공간에서 잠재 쿼리의 기대값을 강력한 대체물로 사용하며, 문서의 고유 식별자를 첨부함으로써 각 문서 임베딩을 효과적으로 구분합니다. 이를 통해 QA 시스템에서 문서-쿼리 갭을 강하게 연결할 수 있습니다.

- **Performance Highlights**: QAEncoder는 기존 방법들과 비교하여 추가적인 인덱스 크기와 검색 지연 없이 문서 임베딩과 사용자 쿼리 간의 유사성을 높여주며, 여러 언어 및 데이터셋에서 수행된 실험에서 뛰어난 성능을 입증하였습니다.



### HELPD: Mitigating Hallucination of LVLMs by Hierarchical Feedback Learning with Vision-enhanced Penalty Decoding (https://arxiv.org/abs/2409.20429)
Comments:
          Accepted at Main Conference of EMNLP 2024

- **What's New**: 대규모 비전-언어 모델(LVLMs)의 다중 모달 환각(multimodal hallucination) 문제를 해결하기 위한 새로운 접근 방식인 계층적 피드백 학습(Hierarchical Feedback Learning) 방식인 HELPD를 제안합니다. 이 프레임워크는 객체 및 문장 의미 수준에서 환각 피드백을 통합하여 모델이 생성하는 내용과 이미지 간의 불일치를 줄여줍니다.

- **Technical Details**: HELPD는 환각을 감지하기 위해 객체 집합과 샘플링된 문장을 비교하여 객체 수준 피드백을 생성하며, GPT-4의 강력한 few-shot 추론 능력을 활용하여 문장 수준 피드백을 수행합니다. 또한, Vision-Enhanced Penalty Decoding 방식을 통해 시각 입력의 중요한 영향을 반영하여 최종 로짓(logits) 계산 시 시각 입력에 더 많은 비중을 두도록 합니다.

- **Performance Highlights**: 실험 결과, HELPD는 다양한 환각 벤치마크에서 15% 이상의 환각 완화를 달성하며, LVLM의 텍스트 생성 품질을 동시에 향상시키는 긍정적인 결과를 보여주었습니다.



### Decoding the Echoes of Vision from fMRI: Memory Disentangling for Past Semantic Information (https://arxiv.org/abs/2409.20428)
Comments:
          Accepted at Main Conference of EMNLP 2024

- **What's New**: 이번 연구는 인간의 시각 시스템이 연속적인 시각 정보의 흐름을 처리하는 방식과 관련된 시각 기억의 인코딩 및 회수 메커니즘을 탐구하며, 이를 위해 새로운 과제 'Memory Disentangling'을 제안합니다.

- **Technical Details**: 이 연구에서는 기능적 자기공명영상(fMRI) 신호에서 과거 정보를 추출하고 현재 뇌 활동과 분리하는 방법으로 특히 'proactive interference' 이론을 활용한 disentangled contrastive learning 기법을 설계하였습니다. 또한, ridge regression 분석과 trial-wise representational similarity analysis(RSA) 기법을 사용하여 fMRI 신호와 과거 시각 자극의 상관관계를 평가했습니다.

- **Performance Highlights**: 실험 결과, 제안한 방식이 fMRI 신호 내에서 정보를 효과적으로 분리하는 것으로 나타났으며, 이는 뇌-컴퓨터 인터페이스(BCI) 발전에 기여하고 fMRI의 낮은 시간 해상도 문제를 완화할 수 있는 가능성을 보여줍니다.



### Anti-stereotypical Predictive Text Suggestions Do Not Reliably Yield Anti-stereotypical Writing (https://arxiv.org/abs/2409.20390)
- **What's New**: 이 연구에서는 AI 기반 언어 모델이 생성하는 텍스트에 내재된 사회적 고정관념이 사용자에게 미치는 영향을 분석합니다. 특히, 사용자가 언어 모델을 통해 생성된 예측 텍스트를 받아들일 때, 성별 고정관념과 같은 사회적 편향이 그들의 이야기에 어떻게 반영되는지를 살펴봅니다.

- **Technical Details**: 연구는 414명 참가자를 대상으로 진행된 온라인 과제로, 참가자들은 예측 텍스트 시스템의 도움을 받아 짧은 영어 이야기를 작성했습니다. 제공된 예측 텍스트는 성별 및 성적 지향성을 기준으로 한 두 가지 유형으로 구분되었습니다: 프로-고정관념(예: 남성 의사)과 안티-고정관념(예: 여성 의사). 언어 모델의 제안이 사용자의 행동에 미치는 영향을 측정했습니다.

- **Performance Highlights**: 연구 결과, 참가자들은 프로-고정관념 제안을 더 자주 수용하는 경향이 있으며, 이는 이야기의 전반적인 경향성과 일치합니다. 반면에, 안티-고정관념 제안이 포함된 경우에도 이야기에 미치는 영향은 제한적이었습니다. 예를 들어, 대통령 캐릭터를 여자로 제안하는 경우, 여성이 대통령인 이야기가 증가하는 경향이 있지만 여전히 남성 대통령 캐릭터가 더 많이 등장했습니다.



### Wait, but Tylenol is Acetaminophen... Investigating and Improving Language Models' Ability to Resist Requests for Misinformation (https://arxiv.org/abs/2409.20385)
Comments:
          Submitted for Review

- **What's New**: 대규모 언어 모델(LLM)이 사용자 요청에 맹목적으로 따르는 취약성이 의학 분야에서 잘못된 정보를 생성할 위험을 증가시킨다는 점을 강조합니다.

- **Technical Details**: 모델들이 비논리적인 요청을 인지하는 상황에서 약물에 대한 오해를 일으키는 콘텐츠 생성을 분석했습니다. LLM의 논리적 사고를 우선시하는 방향으로 훈련을 조정하는 것이 오정보 확산을 줄이는 데 효과적임을 보여줍니다.

- **Performance Highlights**: 모든 선진 LLM이 잘못된 정보 요청을 따르더라도, 프롬프트 기반(prompt-based) 및 파라미터 기반(parameter-based) 접근 방식이 요청의 논리적 결함을 감지하고 의료 오정보를 방지하는 데 기여할 수 있다는 결과를 도출했습니다.



### Word-wise intonation model for cross-language TTS systems (https://arxiv.org/abs/2409.20374)
- **What's New**: 본 논문에서는 러시아어를 위한 단어 단위 억양 모델을 제안하고, 이 모델의 다른 언어에 대한 일반화 가능성을 보여줍니다.

- **Technical Details**: 제안된 모델은 TTS(Text-to-Speech) 시스템에 적합하며, 규칙 기반 알고리즘이나 언어 모델을 사용한 예측을 통해 억양 윤곽선 모델링에 적용될 수 있습니다. 'Intonation PAttern-STAte' (PASTA) 모델은 단어의 억양 변동을 보편적으로 설명하여, 오디오와 텍스트를 기반으로 자동 제어 가능한 멜로디 마크업 시스템을 제공합니다.

- **Performance Highlights**: 모델은 다양한 매개변수 변동에 대한 강건성을 보여주며, BERT와 같은 언어 모델을 사용하여 억양 예측의 가능성을 내포하고 있습니다.



### Disentangling Singlish Discourse Particles with Task-Driven Representation (https://arxiv.org/abs/2409.20366)
- **What's New**: 이 논문에서는 싱글리시(Singlish)의 담화 입자(discourse particles)를 해체하여 그 실용적인 기능을 이해하기 위한 기계 학습 접근 방식을 제안합니다. 특히, 'lah', 'meh', 'hor'와 같은 담화 입자들의 기능을 클러스터링하고, 싱글리시에서 영어(machine translation)로 번역하는 작업을 수행합니다.

- **Technical Details**: 담화 입자의 기능을 파악하기 위해, 저자들은 task-driven representation learning을 활용하며, Next Sentence Prediction(NSP)과 Particle Prediction(P-Pred)이라는 두 가지 학습 과업을 설정합니다. 이 과정에서 BERT 모델을 기반으로 한 SingBERT를 사용하여 싱글리시의 고유한 담화 입자 기능을 학습시킵니다.

- **Performance Highlights**: 이 연구는 싱글리시의 담화 입자에 대한 실질적인 이해를 높이고, 이후의 작업에서 문장 의미를 바탕으로 한 응용 프로그램(stance identification 등)에 활용될 수 있는 가능성을 보여줍니다. 또한, 영어 사용자가 싱글리시를 이해하는 데 도움을 주는 Singlish-to-English 기계 번역 모델을 제시합니다.



### A Looming Replication Crisis in Evaluating Behavior in Language Models? Evidence and Solutions (https://arxiv.org/abs/2409.20303)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 행동에 대한 연구가 증가함에 따라 복제 위기의 잠재적 위험성에 대해 논의합니다.

- **Technical Details**: 연구에서는 prompt engineering 기술을 활용하여 LLM의 추론 능력에 미치는 영향을 테스트했으며, 다양한 모델(GPT-3.5, GPT-4o, Gemini 1.5 Pro 등)을 사용했습니다. 실험에서는 CommonsenseQA, CRT, NumGLUE, ScienceQA, StrategyQA와 같은 추론 벤치마크를 포함한 이중 검증된 데이터 세트를 활용했습니다.

- **Performance Highlights**: 모든 테스트된 기술에서 통계적으로 유의미한 차이가 거의 없는 결과를 보였으며, 이는 이전 연구에서의 여러 방법론적 약점을 강조합니다. 따라서 LLM 평가를 위한 강력한 방법론 개발과 철저한 실험 프레임워크 설계의 필요성을 제안합니다.



### LexEval: A Comprehensive Chinese Legal Benchmark for Evaluating Large Language Models (https://arxiv.org/abs/2409.20288)
Comments:
          NeurIPs 2024

- **What's New**: 이번 논문에서는 중국 법률을 평가하기 위한 표준화된 벤치마크인 LexEval을 소개합니다. 이 벤치마크는 법률 인지 능력 분류, 23개의 법률 작업 및 14,150개의 질문을 포함하는 대규모 평가 데이터셋으로 구성되어 있습니다. 또한 LLM(대형 언어 모델)의 윤리적 문제를 검토하는 데에도 중점을 둡니다.

- **Technical Details**: LexEval은 법률 인지 능력에 대한 새로운 분류 체계를 제안하며, 이를 바탕으로 다양한 법률 작업을 체계적으로 정리합니다. 이 벤치마크는 Memorization, Understanding, Logic Inference, Discrimination, Generation, Ethic의 6가지 측면으로 구성되어 있으며, 법률 전문가에 의해 주석이 달린 새로운 데이터셋을 포함합니다.

- **Performance Highlights**: 38개의 오픈 소스 및 상업적인 LLM들을 평가한 결과, 기존 LLM들이 법률 문제를 해결하는 데 효과적이지 않다는 흥미로운 발견을 얻었습니다. LexEval의 결과와 실험은 중국 법률 시스템 발전을 위한 유용한 통찰력을 제공합니다.



### Analysing Zero-Shot Readability-Controlled Sentence Simplification (https://arxiv.org/abs/2409.20246)
- **What's New**: 이 논문은 Readability-controlled Text Simplification (RCTS)라는 개념을 탐구하며, 이는 텍스트의 가독성을 낮추면서 의미를 유지하는 방법입니다. 특히, 우리는 병렬 데이터에 대한 의존도를 줄이기 위해 instruction-tuned 대형 언어 모델(large language models, LLMs)을 활용하여 zero-shot RCTS를 수행합니다.

- **Technical Details**: 이 연구에서는 다양한 유형의 맥락 정보를 제공하여 목표 가독성을 조절하는 LLM의 성능을 분석합니다. CEFR(일반유럽어교육기준) 수준에 따라 텍스트를 난이도에 맞게 재작성하며, 문장 레벨에서 LLM의 성능을 측정합니다. 8개의 LLM 모델을 활용하여, 목표 가독성에 도달하면서 원래 의미를 유지하는 성능을 평가합니다.

- **Performance Highlights**: 모든 모델이 목표 가독성을 만족시키며 문장을 단순화하는 데 어려움을 겪었으며, 특히 낮은 가독성 수준에 대한 간극이 클 경우 성공률이 낮았습니다. 또한, 자동 평가 메트릭의 향상이 필요하며, 기존의 것들이 종종 핵심 단순화 작업을 간과하여 읽기 어려운 평가를 초래하는 경향이 있음을 확인했습니다.



### PsyGUARD: An Automated System for Suicide Detection and Risk Assessment in Psychological Counseling (https://arxiv.org/abs/2409.20243)
Comments:
          Accepted to EMNLP 2024 main conference

- **What's New**: 이 논문에서는 온라인 상담 서비스에서 자살 사고(suicidal ideation)를 감지하고 위험을 평가하는 자동화된 시스템인 PsyGUARD를 제안합니다. PsyGUARD는 자살 사고를 탐지하기 위한 상세한 분류법(taxonomy)을 개발하고, 이를 지원하기 위해 높은 품질의 데이터셋인 PsySUICIDE를 구축하였습니다.

- **Technical Details**: PsyGUARD는 자살 사고를 세분화하여 분류하고, 자살 행동(self-harm)이나 타인에게 해를 끼치는 위험을 포함한 프레임워크를 제공합니다. 데이터 수집 및 주석(annotation) 프로세스는 철저하게 수행되었으며, 다양한 기준선(baselines)을 설정하여 시스템의 성능을 비교 분석합니다.

- **Performance Highlights**: 이 자동화된 시스템은 정교한 자살 사고 탐지에 기반하여 정신 건강 서비스의 효율성을 향상시킬 수 있으며, 안전하고 맞춤화된 지원을 온라인 상담 플랫폼에서 제공하는 데 기여할 수 있습니다.



### Beyond Prompts: Dynamic Conversational Benchmarking of Large Language Models (https://arxiv.org/abs/2409.20222)
Comments:
          Accepted as a poster at NeurIPS D&B Track 2024

- **What's New**: 이 논문에서는 대화형 에이전트의 성능을 평가하기 위한 동적 벤치마크 시스템을 소개하고 있습니다. 이 시스템은 단일, 모의된 사용자와 에이전트 간의 긴 상호작용을 통해 성능을 측정하며, 다수의 작업을 동시에 수행하도록 설계되었습니다.

- **Technical Details**: LTM 벤치마크는 에이전트의 Long-Term Memory (LTM)와 Continual Learning (CL) 능력을 평가합니다. 대화 중에 여러 개의 정보 조각이 서로 결합되어 에이전트의 정보를 효과적으로 사용하는 능력을 평가하게 됩니다. 이 시스템은 LLMs의 자연어 대화 상황에서의 성능을 평가하는 데 초점을 맞추고 있으며, 다양한 작업과 산만함을 포함하여 리얼리틱한 상호작용을 구현하고 있습니다.

- **Performance Highlights**: 대규모 언어 모델(LLMs)은 단일 작업 상호작용에서는 좋은 성능을 보이지만, 작업들이 혼합되었을 때 어려움을 겪습니다. 특히, 짧은 컨텍스트의 LLM이 장기 기억 시스템(LTM)이 보완된 경우에는 더 긴 컨텍스트를 가진 모델들과 유사하거나 더 좋은 성능을 보인다는 것을 발견했습니다.



### Divided by discipline? A systematic literature review on the quantification of online sexism and misogyny using a semi-automated approach (https://arxiv.org/abs/2409.20204)
- **What's New**: 최근 언론에서 성차별(sexism)과 여성 혐오(misogyny) 및 성별 기반 증오 발언을 탐지하고 식별하기 위한 여러 계산 도구들이 개발되고 있습니다. 그러나 온라인 성차별 또는 여성혐오 측정에 대한 연구의 현황에 대한 이해는 부족합니다. 이 연구는 2012년부터 2022년까지의 문헌을 정리하고, 성별 기반 증오 발언의 측정에서의 기회와 도전을 파악합니다.

- **Technical Details**: 이 연구는 PRISMA(flowchart의 일종) 흐름도에 따라 선택 단계에서 검색 결과를 좁히는 반자동 방법을 제시하고, 컴퓨터 과학(Computer Science)과 사회 과학(Social Science) 분야에서 온라인 성차별 및 여성혐오를 정량화하고 측정하는 연구 논문에 대한 체계적 문헌 리뷰(Systematic Literature Review, SLR)를 수행합니다.

- **Performance Highlights**: 연구 결과, 성차별과 여성혐오에 대한 연구 주제 간의 학문적 분리가 존재합니다. 특히 텍스트 분석과 자연어 처리(Natural Language Processing, NLP)를 통해 온라인 데이터에서 성차별을 분석할 필요성이 강조됩니다. 또한, 자동 탐지가 어려운 성차별과 여성혐오를 탐지하기 위한 새로운 방법론적 접근을 제안하며, 성별 기반 증오 발언의 정량화 및 측정을 위한 기회를 모색합니다.



### AfriHuBERT: A self-supervised speech representation model for African languages (https://arxiv.org/abs/2409.20201)
Comments:
          14 pages

- **What's New**: 이번 연구에서는 mHuBERT-147을 기반으로 아프리카 언어를 지원하는 AfriHuBERT 모델을 소개합니다. 이 모델은 기존 16개 아프리카 언어에서 39개의 아프리카 언어로 확대되어 6,500시간 이상의 음성 데이터를 활용하였습니다.

- **Technical Details**: AfriHuBERT 모델은 언어 식별(Language Identification, LID) 및 자동 음성 인식(Automatic Speech Recognition, ASR) 작업을 위해 FLUERS 데이터 세트를 사용하여 평가되었습니다. 이 모델은 시각적 특징을 포착하기 위해 CNN(Convolutional Neural Network) 인코더와 BERT(Bidirectional Encoder Representations from Transformers) 기반의 트랜스포머 아키텍처를 활용합니다.

- **Performance Highlights**: LID 작업에서 평균 F1 점수가 4% 증가했으며, ASR 작업에서는 평균 단어 오류율(Word Error Rate, WER)이 1.2% 감소했습니다. 또한 AfriHuBERT를 기반으로 한 ASR 모델은 크로스 코퍼스 일반화에서 향상된 성능을 보였습니다.



### TaskComplexity: A Dataset for Task Complexity Classification with In-Context Learning, FLAN-T5 and GPT-4o Benchmarks (https://arxiv.org/abs/2409.20189)
Comments:
          This papaer has been accepted to The 3nd International conference on Machine Learning and Data Engineering (ICMLDE 2024)

- **What's New**: 이 논문은 프로그래밍 작업을 전문가에게 분류하고 할당하는 문제를 해결하기 위한 새로운 데이터셋을 제안합니다. 총 4,112개 프로그래밍 작업을 포함하는 데이터셋이 웹 스크래핑 기법으로 다양한 웹사이트에서 수집되었습니다.

- **Technical Details**: 데이터셋은 Kattis, LeetCode, HackerRank, Topcoder 등에서 수집되었으며, 각각의 작업은 제목, 문제 설명, 입력 및 출력 명세, 예시, 문제 클래스, 복잡성 점수 등의 요소를 포함하고 있습니다. 두 가지 기계 학습 접근법, 즉 FLAN-T5 소형 모델의 미세 조정(fine-tuning)과 GPT-4o-mini 모델을 이용한 인컨텍스트 학습(in-context learning)이 사용되었습니다.

- **Performance Highlights**: GPT-4o-mini는 정확도 57.00%, F1 스코어에서 FLAN-T5 소형 모델을 초과하는 성능을 보였습니다(FLAN-T5의 정확도는 52.24%). 이 결과는 GPT-4o-mini의 성능이 FLAN-T5 모델보다 우수하다는 것을 보여줍니다.



### Reference Trustable Decoding: A Training-Free Augmentation Paradigm for Large Language Models (https://arxiv.org/abs/2409.20181)
- **What's New**: 본 논문에서는 Reference Trustable Decoding (RTD)라는 새로운 패러다임을 제안합니다. RTD는 대형 언어 모델(LLMs)이 하위 작업에 빠르게 적응할 수 있도록 하며, 파라미터 조정 없이도 수행됩니다.

- **Technical Details**: RTD는 훈련이 필요 없는 방법이며, 모형의 마지막 숨겨진 상태를 활용하여 사전에 구성된 데이터 저장소에서 관련 리퍼런스를 조회합니다. 이로 인해 최종 출력 분포를 리퍼런스의 유사도 점수로 재계산하여 신뢰할 수 있는 응답을 생성합니다.

- **Performance Highlights**: RTD는 기존의 ICL 및 PEFT와 동일하거나 더 나은 성능을 보여주었으며, 입력 길이를 추가하지 않고도 신뢰할 수 있는 답변을 제공하는 가능성이 확인되었습니다. RTD와 전통적인 방법의 통합을 통해 성능이 더욱 향상될 수 있음을 보여줍니다.



### Using Large Multimodal Models to Extract Knowledge Components for Knowledge Tracing from Multimedia Question Information (https://arxiv.org/abs/2409.20167)
Comments:
          v0: This work is a preprint and has not been peer-reviewed

- **What's New**: 이 논문은 교육 콘텐츠에서 지식 구성 요소(Knowledge Components, KCs)를 자동으로 추출하기 위해 조정된 대형 다중 모달 모델(Instruction-tuned Large Multimodal Models, LMMs)을 사용하는 방법을 제안합니다. 기존의 지식 추적(Knowledge Tracing, KT) 모델의 한계를 극복하고, AI 생성 교육 콘텐츠와 전통적인 방법을 통합할 수 있는 가능성을 제시합니다.

- **Technical Details**: 우리의 접근 방식은 교육 자료를 파싱하여 텍스트와 이미지를 추출하고, GPT-4o API를 사용하여 내재된 KCs를 식별 및 설명하며, 문장 임베딩(Sentence Embeddings)에 기반하여 유사한 구성 요소를 클러스터링하는 것입니다. 이러한 자동화는 KC 추출을 개선할 뿐만 아니라 새로운 콘텐츠에 대한 학생 성과 예측을 향상시킵니다.

- **Performance Highlights**: LMM이 생성한 KCs를 다양한 KT 방법에서 추가 특징으로 사용했을 때 성능 개선이 관찰되었습니다. 예를 들어, 성능 요소 분석(Performance Factors Analysis, PFA) 방법에서 LMM으로 생성한 KCs는 인간이 생성한 KCs에 비해 성능 향상이 더 크게 나타났으며 다른 KT 방법에서도 유사한 성능 향상이 있었습니다. 결과적으로 LMM으로 생성한 KCs를 사용할 때 인간이 생성한 KCs에 비해 동등하거나 우수한 성능을 입증했습니다.



### How Entangled is Factuality and Deception in German? (https://arxiv.org/abs/2409.20165)
Comments:
          Findings of EMNLP 2024 (accepted)

- **What's New**: 이번 연구는 사실의 정확성과 진실성 간의 혼동을 해소하고, 독일어 텍스트에서의 믿음 기반 기만(dception) 패턴의 일반화를 평가합니다. 또한, 기만 탐지(computational models)의 효과성을 평가하고, 사실 확인(fact checking) 작업에서 기만이 미치는 영향을 탐구합니다.

- **Technical Details**: 기본 개념으로, 기만은 사람들이 말하는 것과 그들이 진정으로 믿는 것 사이의 불일치로 정의됩니다. 연구에서 저자들은 'DeFaBel'이라는 독일어 기만 데이터셋을 활용하여, 언어적 단서(linguistic cues) 및 기만 인식(deception detection) 모델을 분석합니다. 또한, 자연어 추론(Natural Language Inference) 기반의 검증이 비상사실(non-factual) 및 기만 콘텐츠에 대해 저조한 성능을 보이는 것을 확인하였습니다.

- **Performance Highlights**: 조사 결과, 전통적 모델과 최신 AI 모델 모두 기만 탐지 작업에서 무작위 추측(random guessing) 수준의 성능을 보여, 저자의 진정한 믿음과 사실의 일치 여부가 기만 탐지에 있어 큰 영향을 미치지 않음을 나타냅니다. Furthermore, Large Language Models는 동일 작업에 대해 비사실 및 기만적 내용에 덜 민감한 반응을 보였습니다.



### 1 Trillion Token (1TT) Platform: A Novel Framework for Efficient Data Sharing and Compensation in Large Language Models (https://arxiv.org/abs/2409.20149)
- **What's New**: 본 논문에서는 1 조 토큰 플랫폼(1TT Platform)을 제안합니다. 이 플랫폼은 투명하고 공정한 수익 분배 메커니즘을 통해 효율적인 데이터 공유를 촉진하도록 설계된 새로운 프레임워크입니다.

- **Technical Details**: 1TT 플랫폼은 데이터 기여자(data contributors)와 데이터 소비자(data consumer) 간의 협업을 촉진합니다. 데이터 기여자는 자신의 데이터를 제공하고, 데이터 소비자는 이 데이터를 활용해 자신의 서비스를 개선하여 수익을 창출합니다. 데이터 기여자는 서비스 수익의 일부를 보상받으며, 이 과정에서 자동화된 데이터 전처리(preprocessing)가 이루어지고, 기여도가 정량화되어 금전적인 보상이 계산됩니다.

- **Performance Highlights**: 1TT 플랫폼은 기여자에게 공정한 보상을 보장하여 고품질의 비공식 데이터를 효율적으로 공유할 수 있는 환경을 마련합니다. 이는 NLP와 LLM 기술의 발전을 촉진하는 협력 생태계를 형성하며, 향후 기여자 평판 시스템 도입 및 맞춤형 데이터 요청 메커니즘 구현 등의 발전 방향이 제안됩니다.



### Classification of Radiological Text in Small and Imbalanced Datasets in a Non-English Languag (https://arxiv.org/abs/2409.20147)
- **What's New**: 이번 연구에서는 덴마크어로 작성된 MRI 보고서를 대상으로 의료 분야에서 자연어 처리 (NLP) 모델의 성능을 평가했습니다. 특히, BERT와 같은 모델이 가장 우수한 결과를 보였으며, SetFit 및 대형 언어 모델(LLM)이 저조한 성적을 기록했습니다. 이는 소규모 데이터세트와 불균형 클래스를 다룰 때 BERT가 최적의 성능을 제공함을 보여줍니다.

- **Technical Details**: 연구에서는 덴마크어로 작성된 16,899개의 MRI 보고서를 사용하였으며, BERT-like transformers, SetFit, 및 LLM 모델을 포함한 다양한 NLP 모델의 성능을 비교했습니다. BERT-like 모델은 해당 도메인에서 사전 학습된 경우 최상의 성능을 나타냈으며, hyperparameter 최적화 과정이 포함되었습니다.

- **Performance Highlights**: BERT-like 모델은 다른 모델들에 비해 우수한 성능을 보였으며, LLM은 성능이 가장 저조했습니다. 그러나 모든 모델이 무감독 텍스트 분류에는 충분한 정확도를 제공하지 않았으나, 데이터 필터링의 잠재력을 보여 주어 수동 라벨링의 필요성을 줄일 수 있는 가능성을 제시합니다.



### ACE: Abstractions for Communicating Efficiently (https://arxiv.org/abs/2409.20120)
Comments:
          9 pages, 9 figures

- **What's New**: 이 논문에서는 인공지능(AI)의 문제 해결 과정에서 중요한 요소인 '추상화(Abstraction)'의 도입과 활용 가능성을 탐구합니다. 특히, 'Communicating Efficiently(효율적인 의사소통)'를 위한 새로운 접근법인 'ACE'를 제안합니다.

- **Technical Details**: ACE는 신경-기호적(neuro-symbolic) 접근을 통해, 라이브러리 학습(library learning)과 강화 학습(reinforcement learning)을 결합하여 새로운 추상화를 도입합니다. 이 과정에서 bandit 알고리즘을 활용하여 탐색(exploration)과 활용(exploitation)의 균형을 조정합니다. 실험은 'Architect-Builder game'을 기반으로 하여 설계되었습니다.

- **Performance Highlights**: ACE 모델은 961개의 목표 장면에서 실험하여 인간과 유사한 경향성을 보여주었으며, 효율적인 언어의 자연 발생을 촉진하는 결과를 도출했습니다. 이러한 발견은 인공지능 대화형 에이전트들이 인간과 유사한 소통 추상화를 갖출 수 있는 첫 번째 단계로 자리잡습니다.



### Aggressive Post-Training Compression on Extremely Large Language Models (https://arxiv.org/abs/2409.20094)
- **What's New**: 본 논문에서는 대형 언어 모델(Large Language Model, LLM)의 압축을 위해 새로운 네트워크 프루닝 기술을 제안하고 있습니다. 이 기술은 0.7 이상의 희소성(sparsity)과 8비트 이하의 양자화(quantization)를 활용하여, 모델의 크기를 줄이면서도 상대적으로 적은 정확도 손실을 유지할 수 있도록 설계되었습니다.

- **Technical Details**: 제안된 방법은 기존 LLM에 대한 압축을 2시간 이내에 수행할 수 있으며, 기존의 OPTQ(Optimal Brain Compression)와 SparseGPT 알고리즘을 활용하여, 연결된 희소성 분포를 조정하고 기준 오류를 추정하는 레이어별 희소성 스케줄러를 사용합니다. 이로 인해, 약 175억 개의 파라미터를 가진 LLM에서 0.7 이상의 희소성을 달성하고, 모델의 성능을 비슷하게 유지할 수 있습니다.

- **Performance Highlights**: 실험 결과, 우리의 방법은 OPT-66B 및 BLOOM-176B와 같은 모델에서 최첨단 수준의 성능을 지속적으로 초과하였으며, 희소성 분포 조정에 따른 결과도 양호하였습니다. 또한, LLM 양자화 기술을 지원하여 FP16에서 INT4로의 변환 시 추가적인 압축이 가능하다는 점에서 큰 장점을 가지고 있습니다.



### BSharedRAG: Backbone Shared Retrieval-Augmented Generation for the E-commerce Domain (https://arxiv.org/abs/2409.20075)
Comments:
          EMNLP 2024 findings

- **What's New**: 본 연구에서는 도메인 특화된 RAG 시스템을 구현하기 위해 Backbone Shared RAG (BSharedRAG) 프레임워크를 제안합니다. 이 프레임워크는 도메인 특정 데이터셋인 WorthBuying을 활용하여 retrieval과 generation 성능을 동시에 향상시키는 메커니즘을 제공합니다.

- **Technical Details**: BSharedRAG는 두 개의 Low-Rank Adaptation (LoRA) 모듈을 이용하여 retrieval 손실과 generation 손실을 최소화합니다. 이 프레임워크는 도메인 특정 지속적 사전 학습을 통해 생성기와 검색기가 서로의 향상에 기여할 수 있도록 설계되었습니다. 또한, 735K 문서와 50K QDA 튜플을 포함한 고품질 데이터셋을 구축하였습니다.

- **Performance Highlights**: BSharedRAG는 두 개의 평가 데이터셋에서 Hit@3에서 5% 및 13% 높고, BLEU-3에서는 23% 향상된 성능을 보이며, 기존 RAG 방법들을 크게 능가하는 성과를 거둡니다.



### Is Preference Alignment Always the Best Option to Enhance LLM-Based Translation? An Empirical Analysis (https://arxiv.org/abs/2409.20059)
- **What's New**: 이 논문은 기계 번역(Machine Translation, MT) 평가를 위한 Neural Metrics의 사용 확대와 품질 기준 최적화 기술인 Contrastive Preference Optimization (CPO)에 대해 다룹니다.

- **Technical Details**: Neural metrics는 기존의 lexical metrics에 비해 인간의 판단과 더 높은 상관관계를 보이며, 연구자들은 quality-informed decoding 전략을 활용하여 더 나은 결과를 얻고 있습니다. CPO는 품질 추정기에 의해 유도된 선호를 기반으로 모델 가중치를 최적화하는 기법으로, 본 연구에서는 CPO의 기계 번역 품질 향상 효과를 평가하기 위한 광범위한 실험을 수행했습니다.

- **Performance Highlights**: CPO는 Supervised Fine-Tuning (SFT)보다 고품질 데이터에서 alignment metric에 대해 일관되게 우수한 성능을 보였으나, 다운스트림 평가 지표 간의 불안정성을 초래할 수 있습니다. 또한 기본 모델을 사용하여 후보 번역을 생성하면 여러 외부 시스템을 사용하는 것과 유사한 성능을 유지하면서도 다운스트림 지표 간의 일관성을 더 좋게 보장할 수 있음을 입증했습니다.



### Evaluating and explaining training strategies for zero-shot cross-lingual news sentiment analysis (https://arxiv.org/abs/2409.20054)
Comments:
          The first two authors share equal contribution

- **What's New**: 이번 연구에서는 여러 언어에 걸쳐 강력한 감정 분류기를 개발하는 것을 목표로 하는 제로샷(Zero-shot) 교차언어 뉴스 감정 분석을 조사합니다. 새로운 평가 데이터셋을 도입하고, 기계 번역(Machine Translation), 인컨텍스트 학습(In-context learning) 및 심층 학습을 포함한 다양한 접근 방식을 실험하였습니다.

- **Technical Details**: 많은 자원이 부족한 언어에서 제로샷 방식을 통해 감정 분석(Sentiment Analysis, SA)을 수행하는 데 주력했습니다. 연구팀은 mBERT 모델을 활용하면서 다양한 언어 간 감정 전이를 평가하기 위한 새로운 방법인 POA(Part Of Article)를 도입했습니다. 이 방법은 문서 내에서 특정 텍스트의 위치 정보를 사용하여 감정 분석의 효율성을 높이는 방식을 제공합니다.

- **Performance Highlights**: 연구 결과는 기존의 최신 기술(State of the Art)을 초과하는 성능 개선을 보여주었습니다. 인컨텍스트 학습이 일반적으로 가장 뛰어난 성능을 보였으나, 새로 도입된 POA 접근 방식은 낮은 계산 오버헤드에도 불구하고 경쟁력 있는 대안을 제공했습니다. 또한 언어 유사성만으로는 교차언어 전이의 성공을 예측할 수 없으며, 의미적 내용과 구조의 유사성도 중요할 수 있다는 점을 강조했습니다.



### Depression detection in social media posts using transformer-based models and auxiliary features (https://arxiv.org/abs/2409.20048)
Comments:
          Social Network Analysis and Mining (Accepted)

- **What's New**: 이 연구는 사회적 미디어 게시물에서 우울증을 탐지하기 위해 전통적인 기계 학습 알고리즘의 한계를 극복하는 혁신적인 신경망 아키텍처를 제안합니다.

- **Technical Details**: 제안된 모델은 transformer 기반의 DistilBERT를 활용하여 텍스트의 마지막 네 개 층에서 정보를 추출합니다. 메타데이터(metadata)와 언어학적 마커(linguistic markers)를 결합하여 입력 텍스트의 풍부한 표현을 생성하는 방식을 사용합니다. Dropout 레이어를 통해 과적합(overfitting)을 방지하고 최종 분류를 위해 Multilayer Perceptron (MLP)을 사용합니다.

- **Performance Highlights**: 모델은 84.26%의 Precision, 84.18%의 Recall, 84.15%의 F1-score를 달성했습니다. 데이터 증강 기법을 통해 F1-score가 72.59%에서 84.15%로 크게 향상되었습니다.



### Beyond Scores: A Modular RAG-Based System for Automatic Short Answer Scoring with Feedback (https://arxiv.org/abs/2409.20042)
- **What's New**: 이 논문에서는 응답 평가 및 피드백 생성을 위한 새로운 모듈형 Retrieval-Augmented Generation (RAG) 기반의 자동 단답형 채점 시스템(ASAS-F)을 제안합니다. 이 시스템은 제로샷(zero-shot) 및 몇샷(few-shot) 학습 시나리오에서 작동하도록 설계되어 있으며, 교육 과제에 쉽게 적응할 수 있는 자동 프롬프트 생성 프레임워크를 사용합니다.

- **Technical Details**: 제안한 ASAS-F 시스템은 대형 언어 모델(LLMs)을 활용하여 학생들의 답변을 성공적으로 점수화하고, 유사한 답변을 단답형 점수 피드백 데이터셋에서 검색하여 몇샷(few-shot) 예제 역할을 하도록 구성되었습니다. 이 시스템은 대규모 데이터셋에 대한 의존도를 줄이면서도 높은 정확도를 유지하며 명확하고 정확한 피드백을 제공합니다.

- **Performance Highlights**: 실험 결과, 제안된 시스템은 기존의 미세 조정(fine-tuning) 방법과 비교하여 보지 못한 질문에 대한 채점 정확도가 9% 향상되었으며, 비용 효율적이고 확장 가능한 솔루션을 제공합니다.



### Towards Robust Multimodal Sentiment Analysis with Incomplete Data (https://arxiv.org/abs/2409.20012)
Comments:
          Accepted to NeurIPS 2024

- **What's New**: 이번 연구에서는 Multimodal Sentiment Analysis (MSA) 분야에서 데이터 불완전성 문제를 해결하기 위해 Language-dominated Noise-resistant Learning Network (LNLN)라는 혁신적인 네트워크를 제안합니다. 이 네트워크는 언어 모달리티를 주 모달리티로 설정하고, 다양한 노이즈 상황에서도 모델의 강건성을 강화합니다.

- **Technical Details**: LNLN은 Dominant Modality Correction (DMC) 모듈과 Dominant Modality-based Multimodal Learning (DMML) 모듈을 포함하여, 주 모달리티의 품질을 보장하며, 무작위 데이터 누락 시나리오에서 더 나은 성능을 발휘합니다. 데이터 입력 후, 각 모달리티의 차원을 표준화하는 임베딩 레이어를 통해 시작하며, DMC 모듈은 적대적 학습(adversarial learning)과 동적 가중치 향상 전략을 사용하여 노이즈 영향을 줄이는 방법을 채택합니다.

- **Performance Highlights**: LNLN은 MOSI, MOSEI 및 SIMS와 같은 여러 인기 있는 데이터셋에서 기존 기준선보다 일관되게 우수한 성과를 보였으며, 복잡한 평가 메트릭스에서도 탁월한 성능을 입증하였습니다. 이 연구는 MSA 방법의 강점과 약점을 분석하여, 실제 시나리오에서의 이해를 높이는 데 기여합니다.



### Do Influence Functions Work on Large Language Models? (https://arxiv.org/abs/2409.19998)
Comments:
          18 pages, 8 figures

- **What's New**: 본 연구는 대규모 언어 모델(LLMs)에서의 영향 함수(influence functions)의 효용성을 체계적으로 조사하였습니다. 최근 LLMs의 성장과 함께, 특정 훈련 데이터가 모델 예측에 미치는 영향을 정의하고 정량화하는 것이 점점 더 중요해졌습니다.

- **Technical Details**: 연구에서는 여러 작업에 대해 영향 함수를 평가한 결과, 대부분의 경우 높은 성능을 내지 못함을 발견했습니다. 주요 원인으로는 (1) LLMs의 스케일로 인한 iHVP(implicit Hessian vector product) 요소 추정에서의 불가피한 근사 오차, (2) 세부 조정(fine-tuning) 중의 불확실한 수렴 상태, (3) 모델 파라미터 변화가 LLM 행동 변화와 반드시 일치하지 않는다는 점이 지적되었습니다.

- **Performance Highlights**: 연구 결과, 영향 함수는 대규모 언어 모델에 대하여 일반적으로 성능이 낮고 계산 및 메모리 집약적이라는 한계를 보였습니다. 이전의 영향 함수에 대한 성공 사례는 특수한 사례 연구에 기인하며, 정확한 Hessian 계산 대신 이론적 기반이 부족함을 강조했습니다.



### CONTESTS: a Framework for Consistency Testing of Span Probabilities in Language Models (https://arxiv.org/abs/2409.19984)
- **What's New**: 이 연구는 언어 모델의 스코어가 일관성을 유지하는지 평가하기 위한 새로운 프레임워크인 ConTestS(Consistency Testing over Spans)를 소개합니다. 다양한 확장 가능성 및 모델 간의 예측 일관성에 대한 심층 분석을 제공하여 LLMs(Pretrained Large Language Models) 성능을 개선할 수 있는 기회를 모색합니다.

- **Technical Details**: ConTestS는 통계 검정을 활용하여 서로 다른 조건부 확률 조합에서 모델의 일관성을 평가합니다. 실험은 실제 데이터와 합성 데이터를 포함하여 LLMs의 스코어의 일관성을 평가합니다. 연구 결과, Masked Language Models(MLMs)와 자동 회귀 모델이 예측에서 일관성이 부족함을 보여주며, 특히 자동 회귀 모델에서 더 큰 불일치가 발견되었습니다.

- **Performance Highlights**: 연구 결과, 큰 MLM 모델은 더 일관된 예측을 제공하는 반면, 자동 회귀 모델은 그 크기가 증가할수록 예측 간 불일치가 커지는 경향이 있습니다. 두 모델 유형 모두 예측 엔트로피가 실제 단어의 가능성을 나타내며, 이는 최적의 디코딩 전략 선택에 도움이 될 수 있습니다.



### JaPOC: Japanese Post-OCR Correction Benchmark using Vouchers (https://arxiv.org/abs/2409.19948)
Comments:
          Accepted to PRICAI 2024

- **What's New**: 본 연구는 일본어 영수증에 대한 OCR(Optical Character Recognition) 오류 수정 방법의 벤치마크를 구축하고 효과성을 평가합니다. 이는 기존의 연구에서 다루어지지 않았던 일본어 OCR 오류 수정의 공개 가능 벤치마크를 제공합니다.

- **Technical Details**: 이 연구에서는 일본어 영수증에 특화된 OCR 오류 수정 벤치마크 JaPOC를 제안하고, T5와 같은 언어 모델을 활용하여 오류 수정 방법의 성능을 평가하였습니다. OCR 오류 수정 작업은 시퀀스-투-시퀀스 변환으로 정의되며, OCR 결과에 대해 고급 언어 모델로 미세 조정(fine-tuning)을 진행하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 오류 수정 알고리즘이 전반적인 인식 정확도를 크게 향상시켰습니다. Robota API와 Vision API를 사용하여 구축된 세트에서 상당한 정확도 향상이 확인되었습니다.



### Understanding Higher-Order Correlations Among Semantic Components in Embeddings (https://arxiv.org/abs/2409.19919)
Comments:
          EMNLP 2024

- **What's New**: 이 논문은 독립 구성 요소 분석(Independent Component Analysis, ICA)의 적용이 실제 데이터에서의 비독립성 문제를 해결하는 방법을 제시합니다. 특히, 높은 차수 상관(higher-order correlations)을 사용하여 비독립성을 정량화하고, 이로 인해 구성 요소 간의 강한 의미적 연관성을 발견했습니다.

- **Technical Details**: 해당 연구에서는 ICA를 통해 임베딩의 비독립성을 정량화하기 위해 높은 차수 상관을 계산하였고, 이를 최대 스패닝 트리(maximum spanning tree)로 시각화했습니다. ICA는 비독립 구성 요소를 통계적으로 가능한 한 독립적으로 변환하고, 이 과정에서 실제 데이터의 비선형적 특성을 고려합니다.

- **Performance Highlights**: 연구 결과, ICA를 적용한 임베딩 구조가 조밀하고 해석하기 쉬운 형태를 가지며, 특정 차수에서의 변동성 분석을 통해 PCA보다 더 나은 성능을 보여줍니다. 특히, 강한 높은 차수 상관이 대변하는 의미적 연관성을 통해 NLP 모델의 블랙박스 문제를 해결하는 데 기여합니다.



### Deep Learning and Machine Learning, Advancing Big Data Analytics and Management: Object-Oriented Programming (https://arxiv.org/abs/2409.19916)
Comments:
          47pages

- **What's New**: 본 연구는 Object-Oriented Programming (OOP) 기술의 머신러닝, 딥러닝 및 대규모 언어 모델(LLM) 분야에서의 통합을 포괄적으로 소개합니다. OOP를 활용한 코드의 모듈화, 유지보수, 확장성을 개선하는 방법에 중점을 두고 있습니다.

- **Technical Details**: OOP의 핵심 원칙인 Encapsulation, Inheritance, Polymorphism, 그리고 Abstraction을 자세히 설명합니다. Python을 이용한 이론의 실용적인 적용 사례를 제시하고, 디자인 패턴과 모듈화 프로그래밍이 머신러닝 시스템의 구조와 효율성을 높이는 방법을 탐구합니다.

- **Performance Highlights**: 이 연구는 OOP 원칙을 사용하여 머신러닝 시스템을 구축하고 코드의 재사용성과 확장성을 유지하는 방법을 설명합니다. 초보자와 경험이 있는 개발자 모두가 OOP 방법론을 AI 프로젝트에 적용 할 수 있도록 필요한 지식을 제공합니다.



### UniSumEval: Towards Unified, Fine-Grained, Multi-Dimensional Summarization Evaluation for LLMs (https://arxiv.org/abs/2409.19898)
Comments:
          Accepted at EMNLP-Findings 2024

- **What's New**: 이 논문은 UniSumEval 벤치마크를 소개하며, 이는 다양한 입력 맥락(domain, length)을 포괄하고 세밀하고 다차원적인 주석을 제공하는 최초의 종합적인 벤치마크입니다. 또한 AI를 활용한 자료 작성 프로세스를 도입하여 인간 주석자의 어려움을 줄이는 방법을 제시하고 있습니다.

- **Technical Details**: UniSumEval은 아홉 가지 다양한 도메인(news, report, booking 등)을 포함하여, 길이와 대화/non-dialogue 여부에 따른 여러 종류의 텍스트를 수집했습니다. 평가 차원으로는 신뢰성(faithfulness), 정보 누락(completeness), 간결성(conciseness) 등 세 가지 측면이 있으며, AI 지원 수동 평가를 통해 고도의 inter-annotator agreement(IAA)를 달성했습니다.

- **Performance Highlights**: UniSumEval을 통해 아홉 개의 최신 언어 모델에 대한 성능을 평가하였으며, 이들은 비-LLM, 오픈소스 LLM, 상용 LLM으로 분류되었습니다. 평가 차원에 따라 성능 차이가 나타났으며, PII 필터링이 모든 요약 모델의 환각 문제를 악화시킴을 확인하였습니다.



### Contrastive Token Learning with Similarity Decay for Repetition Suppression in Machine Translation (https://arxiv.org/abs/2409.19877)
Comments:
          Accepted by EMNLP'24 Findings. 12 pages, 4 figures, 9 tables

- **What's New**: 본 논문은 Neural Machine Translation(NMT)의 생성 콘텐츠에서의 단조로움과 반복 문제를 해결하기 위한 새로운 접근 방식을 제시합니다. 정보 엔트로피를 활용해 텍스트 반복의 원인을 분석하며, Contrastive Token Learning with Similarity Decay(CTSD)라는 새로운 알고리즘을 소개합니다.

- **Technical Details**: CTSD는 다이나믹하게 토큰 억제를 조절하며, 변동하는 attention weights와 inter-token distances에 기반합니다. 또한, 온라인 현실 아이템의 제목 텍스트로 구성된 e-commerce 데이터셋을 구축하여 알고리즘의 성능을 평가합니다.

- **Performance Highlights**: CTSD는 기존 접근법보다 정확성과 일반화 능력에서 현저히 우수한 결과를 보이며, 온라인 A/B 테스트를 통해 사용자 참여도와 전환율이 크게 향상되었음을 입증합니다. 이 방법은 세계 최대 B2B e-commerce 플랫폼의 8개 다국어 사이트에서도 성공적으로 구현되었습니다.



### The Construction of Instruction-tuned LLMs for Finance without Instruction Data Using Continual Pretraining and Model Merging (https://arxiv.org/abs/2409.19854)
Comments:
          9 pages

- **What's New**: 본 논문은 금융 분야에 맞춘 지침 조정(Instruction-tuned) 대형 언어 모델(LLM)을 데이터 없이 구축하는 새로운 방법을 제안합니다.

- **Technical Details**: 전통적으로 이러한 도메인 특화 LLM 개발은 대규모 데이터셋과 지속적인 사전 훈련(continal pretraining) 및 지침 조정(instruction tuning)을 위한 상당한 계산 능력이 필요했습니다. 본 연구에서는 도메인 특정 지속적 사전 훈련과 모델 병합(model merging)을 결합하는 간단한 접근 방식을 제안합니다. 공개적으로 이용 가능한 일반 목적의 사전 훈련 LLM들과 지침 조정된 LLM들을 활용하여 필요한 지침 작업 벡터(task vector)를 획득하고, 이를 도메인 특정 사전 훈련 벡터와 병합하여 추가적인 지침 데이터 없이도 금융 분야에 맞는 instruction-tuned LLM을 효과적으로 생성할 수 있습니다.

- **Performance Highlights**: 우리의 실험에서 금융 분야에 맞춘 지침 조정 LLM의 성공적인 구축이 입증되었습니다. 본 방법의 주요 장점은 지침 조정 벡터와 도메인 특정 사전 훈련 벡터가 거의 독립적이라는 점이며, 이는 우리의 접근 방식을 매우 효과적으로 만듭니다. 본 연구에서 개발된 일본 금융 지침 조정 LLM은 해당 URL에서 이용 가능합니다.



### Transforming Hidden States into Binary Semantic Features (https://arxiv.org/abs/2409.19813)
- **What's New**: 본 논문에서는 대규모 언어 모델이 의미의 분포 이론을 다시 적용할 수 있음을 제안합니다.

- **Technical Details**: 독립 성분 분석(Independent Component Analysis)을 사용하여 대규모 언어 모델이 숨겨진 상태에서 의미적 특징(semantic features)을 표현한다는 것을 발견했습니다.

- **Performance Highlights**: 이 연구는 대규모 언어 모델의 숨겨진 상태에서 분포적인 의미를 파악하는 데 중요한 통찰력을 제공합니다.



### Can Models Learn Skill Composition from Examples? (https://arxiv.org/abs/2409.19808)
Comments:
          Accepted to NeurIPS 2024

- **What's New**: 이번 연구는 compositional generalization (구성적 일반화) 능력에 대한 평가에서 SKILL-MIX를 활용하여 smaller models (소형 모델)의 성능을 측정한 것입니다. 이 연구는 AI 안전성 및 정렬 연구와도 관련이 있습니다.

- **Technical Details**: 연구에서는 다양한 언어 기술을 포함한 텍스트 샘플을 생성하기 위해 GPT-4를 사용했습니다. 이를 통해 7B 및 13B 파라미터 모델을 fine-tuning (미세 조정) 하여 k-기술 조합의 텍스트 작성을 평가했습니다. 사용된 기술 세트는 수사학적 (rhetorical), 문학적 (literary), 추론 (reasoning), 마음 이론 (theory of mind), 상식 (common sense) 등을 포함합니다.

- **Performance Highlights**: (1) k=2 및 k=3 기술의 조합으로 학습한 모델은 k=4 및 k=5 기술을 활용한 텍스트 작성을 개선하였습니다. (2) 기술 범주를 나누어 학습 및 검증 그룹으로 분리했을 때, 모델은 훈련 중 본 적이 없는 기술로 텍스트를 작성하는 데에서 크게 향상되었습니다. 이 연구는 skill-rich (기술이 풍부한) 텍스트를 훈련에 통합하는 것이 모델의 구성 능력을 크게 향상시킬 수 있음을 제안합니다.



### Does RAG Introduce Unfairness in LLMs? Evaluating Fairness in Retrieval-Augmented Generation Systems (https://arxiv.org/abs/2409.19804)
Comments:
          Under review

- **What's New**: 이 논문은 최근 주목받고 있는 RAG (Retrieval-Augmented Generation) 방법의 공정성(fairness) 문제를 분석하고 평가하기 위한 새로운 접근 방식을 제안합니다.

- **Technical Details**: 연구에서는 공정성 평가 프레임워크(fairness evaluation framework)를 제안하며, 시나리오 기반 질문들을 이용해 인구통계적 속성(demographic attributes) 간의 차이를 분석합니다. 이는 RAG 방법의 복잡한 파이프라인(pipeline)으로 인해 발생하는 편향(bias)을 이해하고 평가하기 위한 목적입니다.

- **Performance Highlights**: 실험 결과, 유틸리티 중심의 최적화(utility-driven optimization)가 이루어졌음에도 불구하고 RAG의 검색(retrieval) 및 생성(generation) 단계에서 공정성 문제가 여전히 존재함을 보여줍니다. 이는 RAG 파이프라인 내에서 보다 목표 지향적인 공정성 개입(intervention)의 필요성을 강조합니다.



### Black-Box Segmentation of Electronic Medical Records (https://arxiv.org/abs/2409.19796)
- **What's New**: 이 연구는 전자 의료 기록(EMR)의 세분화(segmentation) 문제에 집중하며, 간단한 문장 임베딩 모델과 신경망(neural network)을 이용한 블랙박스(segmentation) 방법을 제안합니다.

- **Technical Details**: EMR 세분화를 위해 다양한 섹션 헤딩 형식으로 구성된 데이터셋에서 모델을 훈련시키고, 여러 고급 딥러닝 기반 자연어 처리(NLP) 방법과 비교합니다. 제안된 방법은 가장 높은 세분화 정확도(98% 이상)를 보여줍니다.

- **Performance Highlights**: 제안된 방법은 다양한 테스트 데이터에 대해 적절한 훈련 데이터 코퍼스를 사용하여 최상의 세분화 정확도를 달성하였습니다.



### Adversarial Examples for DNA Classification (https://arxiv.org/abs/2409.19788)
- **What's New**: 이 논문은 텍스트 분류에서 일반적으로 사용되는 공격 알고리즘을 DNA 서열 분류에 적용하여 기존 연구의 제한된 영역을 확장했습니다.

- **Technical Details**: 연구에서는 DNABERT2와 Nucleotide Transformer와 같은 사전 훈련된 언어 모델을 바탕으로, DNA 서열 분류 작업에서 다양한 공격 방법이 미치는 영향을 평가하였습니다. 이 공격 방법은 문자(character), 단어(word), 문장(sentence) 수준에서 적용되었습니다.

- **Performance Highlights**: 실제 DNA 언어 모델 서열 분류기가 이러한 공격에 취약하다는 사실을 발견하였습니다.



### Towards Robust Extractive Question Answering Models: Rethinking the Training Methodology (https://arxiv.org/abs/2409.19766)
Comments:
          EMNLP 2024 Findings

- **What's New**: 이 논문은 Extractive Question Answering (EQA) 모델의 강건성을 향상시키기 위한 새로운 훈련 방법을 제안합니다. 특히, 기존 EQA 데이터셋에 포함된 답변할 수 없는 질문들이 모델의 강건성 부족을 초래한다는 것을 지적합니다.

- **Technical Details**: 제안된 훈련 방법은 EQA 문제를 위한 새로운 손실 함수(loss function)를 포함하며, 많은 EQA 데이터셋에 존재하는 암묵적인 가정을 도전합니다. 이 방식을 통해 훈련된 모델은 도메인 내에서 성능을 유지하면서도 도메인 외 데이터셋에서 유의미한 향상을 이룹니다.

- **Performance Highlights**: 모든 테스트 세트에서 F1 점수가 5.7만큼 향상되었으며, 두 가지 유형의 적대적 공격(adversarial attacks)에 대해 크게 향상된 강건성을 보여줍니다. 기본 모델과 비교했을 때 성능 저하는 약 1/3에 불과합니다.



### Balancing Cost and Effectiveness of Synthetic Data Generation Strategies for LLMs (https://arxiv.org/abs/2409.19759)
- **What's New**: 대규모 언어 모델(LLMs)의 다양한 용도 적용 확대에 따라, 모델 개선을 위한 고품질 작업 특화 데이터셋 생성이 병목 현상으로 떠오르고 있습니다. 본 논문에서는 고급 인간 데이터를 사용하지 않고도 모델 성능을 최대화하기 위한 방법들을 조사합니다.

- **Technical Details**: 논문에서는 여러 가지 합성 데이터 생성 전략을 세 가지 범주로 나누어 연구합니다: Answer Augmentation(답변 증강), Question Rephrase(질문 재구성), New Question(새로운 질문). 각 전략은 seed instruction set의 크기와 query budget(질의 예산)에 따른 성능 차이를 보였으며, 데이터 생성 전략의 최적 선택은 두 비율의 관계에 크게 의존합니다.

- **Performance Highlights**: 저비용의 데이터 환경에서는 기존 질문에 대한 새로운 답변을 생성하는 것이 가장 효과적이며, 이 비율이 높아질수록 새로운 질문을 생성하는 것이 최적입니다. 전반적으로, 데이터 양이 적은 경우 선택한 증강 방식과 기타 디자인 선택이 성능에 중대한 영향을 미치는 것으로 나타났습니다.



### CoTKR: Chain-of-Thought Enhanced Knowledge Rewriting for Complex Knowledge Graph Question Answering (https://arxiv.org/abs/2409.19753)
- **What's New**: 최근 연구들은 Knowledge Graph Question Answering (KGQA)을 위해 Retrieval Augmented Generation (RAG)과 Large Language Models (LLMs)의 사용을 탐구했습니다. 이들은 일반적으로 검색된 서브그래프를 LLMs가 이해할 수 있는 자연어 형식으로 변환하는 과정을 필요로 합니다. 본 논문에서는 복잡한 질문을 다루는 데 있어 기존 방법의 한계를 해결하기 위해 CoTKR(Chain-of-Thought Enhanced Knowledge Rewriting)라는 새로운 변환 방법을 제안합니다.

- **Technical Details**: CoTKR은 질문에 대한 추론 경로를 생성하고 해당 지식을 상호작용적으로 생산하도록 설계되었습니다. 이 방법은 두 가지 작업을 번갈아 수행합니다: (1) 추론(Reasoning): 질문을 분해하여 추론에 필요한 지식을 식별하고; (2) 요약(Summarization): 추론 단계의 출력을 기반으로 검색된 트리플에서 관련 지식을 요약합니다. 또한, PAQAF(Preference Alignment from Question Answering Feedback)라는 학습 전략을 통해 QA 모델과 지식 변환기 간의 선호 간극을 메우기 위한 방법론을 제시합니다.

- **Performance Highlights**: 실험 결과, CoTKR은 이전의 지식 변환 방법들과 비교할 때 QA 모델에 가장 유익한 지식 표현을 생성하며, 이는 KGQA에서 LLMs의 성능을 크게 향상시키는 것으로 나타났습니다.



### NeuroMax: Enhancing Neural Topic Modeling via Maximizing Mutual Information and Group Topic Regularization (https://arxiv.org/abs/2409.19749)
Comments:
          Findings of EMNLP 2024

- **What's New**: 이번 연구에서는 NeuroMax라는 새롭고 혁신적인 프레임워크를 제안합니다. 이는 사전 훈련된 언어 모델(PLM)과의 최대 상호 정보(maximizing mutual information)를 활용하여 주제 간의 관계를 모델링하는 것을 목표로 하고 있습니다.

- **Technical Details**: NeuroMax는 주제 표현(topic representation)과 PLM에서 파생된 표현 사이의 상호 정보를 최대화하며, 또한 최적 수송(optimal transport) 기법을 활용하여 주제 간의 관계를 학습합니다. 이렇게 함으로써, 사전 훈련된 모델을 사용할 필요 없이 주제 간의 의미적 관계를 효과적으로 포착합니다.

- **Performance Highlights**: 실험 결과에 따르면, NeuroMax는 추론 시간(inference time)을 단축시키고, 보다 일관성 있는 주제와 주제 그룹을 생성하며, 문서 임베딩(document embeddings)의 대표성을 향상시킴으로써 하위 작업(downstream tasks)에서의 성능을 개선합니다.



### Natural Language Generation for Visualizations: State of the Art, Challenges and Future Directions (https://arxiv.org/abs/2409.19747)
- **What's New**: 이 논문은 시각화(visualization)를 위한 자연어 생성(Natural Language Generation, NLG) 기술의 최신 동향을 체계적으로 검토하고 문제에 대한 분류 체계를 제시합니다. 자동으로 차트에 대한 설명 및 데이터 기반 스토리를 생성하는 방법에 대한 관심이 높아지고 있습니다.

- **Technical Details**: 자연어 생성(NLG) 기술이 차트 및 데이터 시각화에 통합되어 시각적 정보를 설명하는 텍스트를 생성하는 방법을 다루고 있습니다. 저자들은 '다섯 가지 Wh-질문'을 통해 NLG 작업의 입력과 출력, 그리고 생성된 텍스트가 시각화와 통합되는 방식에 대해 탐구합니다.

- **Performance Highlights**: 최근 NLG 기술 발전으로 인해 데이터 설명 및 스토리텔링에 효과적으로 활용되는 경향이 있으며, 사용자가 더 많은 텍스트 주석이 포함된 차트를 선호하는 것으로 나타났습니다. 본 조사에서는 122개의 관련 논문을 분석하여 30개의 핵심 논문을 선정하였고, 앞으로의 연구 방향과 주요 도전과제를 논의합니다.



### PEAR: Position-Embedding-Agnostic Attention Re-weighting Enhances Retrieval-Augmented Generation with Zero Inference Overhead (https://arxiv.org/abs/2409.19745)
Comments:
          preprint

- **What's New**: 본 논문에서는 Position-Embedding-Agnostic attention Re-weighting (PEAR)을 제안하여, LLMs의 context awareness를 향상시킵니다. PEAR은 inference 오버헤드 없이 LLM의 성능을 개선하는 새로운 방법을 제시합니다.

- **Technical Details**: PEAR은 RAG 과제에서 context awareness를 강화하기 위해, suppressing heads를 탐지하고 이들의 출력을 learnable coefficients를 사용하여 재가중화합니다. 모델 파라미터를 동결한 채로 이러한 coefficients를 최적화하여 RAG 성능을 제고하는 방식입니다.

- **Performance Highlights**: PEAR은 메모리 사용량이나 inference 시간에서 0의 추가 오버헤드를 제공하며, 다양한 RAG 과제에서 경쟁 기반보다 높은 정확도와 효율성을 보여줍니다. 또한 PEAR은 특정 position embedding 알고리즘에 의존하지 않아 광범위한 적용 가능성을 가지고 있습니다.



### A Systematic Review of NLP for Dementia- Tasks, Datasets and Opportunities (https://arxiv.org/abs/2409.19737)
- **What's New**: 이 논문은 NLP(자연어 처리)가 치매 연구에 어떻게 기여하고 있는지를 다루며, 200편 이상의 관련 논문을 리뷰하여 치매 탐지, 언어학적 바이오마커 추출, 간병인 지원 등 주요 연구 영역을 식별합니다.

- **Technical Details**: 치매는 다양한 병리적 원인으로 인한 인지 기능의 저하를 말하며, 현재는 주로 임상 데이터를 기반으로 한 탐지 방법이 주를 이루고 있습니다. 연구는 언어 분석을 통한 언어적 지표의 발견과 인공지능 모델의 사용을 포함하여, 고유한 데이터셋을 활용하였습니다. 또한, 언어 저하와 관련된 신뢰성, 과학적 엄격성, 적용 가능성 등을 논의합니다.

- **Performance Highlights**: 57%의 논문이 치매 탐지에 중점을 두고 있으며, 향후 연구 방향으로는 인공지능 모델의 인위적 저하, 개인 맞춤형 접근법 등이 제안되고 있습니다. 전체적으로 이 리뷰는 치매 연구의 새로운 가능성 탐색을 촉진하는 데 중점을 두고 있습니다.



### Scrambled text: training Language Models to correct OCR errors using synthetic data (https://arxiv.org/abs/2409.19735)
Comments:
          21 pages, 6300 words, 6 Figures, 5 tables

- **What's New**: 이 논문은 Generative Language Models (LMs)를 활용하여 Optical Character Recognition (OCR) 오류를 교정하는 새로운 접근 방식, Context Leveraging OCR Correction (CLOCR-C) 방법론을 제안했습니다. 특히, 합성 데이터와 Markov 부패 프로세스를 이용한 세밀한 조정이 OCR 오류 교정에 효과적임을 보여주고 있습니다.

- **Technical Details**: 이 연구에서는 합성 데이터를 이용해 학습된 언어 모델이 OCR 오류를 보다 효과적으로 교정할 수 있음을 입증했습니다. 연구 결과, 합성 데이터로 학습된 모델이 기본 LM 대비 문자 오류율(character error rate)을 55%, 단어 오류율(word error rate)을 32% 낮추었습니다. 또한, 과도하게 손상된 데이터보다 적절하게 손상된 데이터에서 더 좋은 성능을 보였으며, 비정형적 문자 수준의 부패가 정형적 부패보다 더 효과적이라는 결과를 도출했습니다.

- **Performance Highlights**: CLOCR-C 모델의 학습을 위한 8가지 히스틱을 제시하며, 11,000개의 합성 19세기 신문 기사 데이터셋과 함께 합성된 부패 데이터를 생성하기 위한 파이썬 라이브러리를 배포했습니다. 연구 결과는 또한 모델 파인 튜닝(practical fine-tuning) 작업에 있어 더 많은 토큰 당 관찰 수가 더 많은 관찰 수보다 우수하다는 사실을 강조하고 있습니다.



### Revealing Personality Traits: A New Benchmark Dataset for Explainable Personality Recognition on Dialogues (https://arxiv.org/abs/2409.19723)
Comments:
          Accepted to EMNLP 2024 Main Conference (Long Paper)

- **What's New**: 본 연구에서는 Explainable Personality Recognition(설명 가능한 성격 인식)이라는 새로운 작업을 제안하여 성격 특성에 대한 추론 과정을 지원 증거로 제시하고자 합니다. 기존의 연구들은 성격 인식을 분류 작업으로만 간주하여 성격 특성을 인식하는 데 필요한 증거를 드러내지 못했습니다.

- **Technical Details**: 연구에서는 Chain-of-Personality-Evidence(CoPE)라는 설명 가능한 성격 인식 프레임워크를 제안합니다. 이는 특정 맥락에서 단기 성격 상태를 거쳐 장기 성격 특성으로 나아가는 추론 과정을 포함하며, PersonalityEvd라는 설명 가능한 성격 인식 데이터 세트를 구축하였습니다. 두 가지 하위 작업, Evidence grounded Personality State Recognition(EPR-S)과 Evidence grounded Personality Trait Recognition(EPR-T)를 도입하여 모델이 상태 및 특성 레이블과 그에 대한 지원 증거를 인식하도록 요구합니다.

- **Performance Highlights**: 매우 도전적인 두 가지 작업에 대해 광범위한 실험을 수행한 결과, 현재의 대형 언어 모델(LLMs)은 성격 이해에서 여전히 인간과의 차이가 크다는 것을 보여주었습니다. 지원 증거의 도입이 성격 인식의 성능을 향상시키는 데 도움을 주고 있으며, 특히 EPR-T 작업에서 상태 증거를 분석하는 것이 성과에 기여함을 발견했습니다.



### Coffee-Gym: An Environment for Evaluating and Improving Natural Language Feedback on Erroneous Cod (https://arxiv.org/abs/2409.19715)
Comments:
          21 pages

- **What's New**: 이번 논문에서는 코드 편집에 대한 피드백을 제공하는 모델 훈련을 위한 포괄적인 RL 환경인 Coffee-Gym을 소개합니다. Coffee-Gym은 인간의 코드 편집 흔적이 포함된 데이터셋인 Coffee와 수정된 코드의 성능을 평가하는 보상 함수인 CoffeeEval로 구성됩니다.

- **Technical Details**: Coffee-Gym의 주요 구성 요소는 (1) Coffee: 코딩 질문에 대한 인간의 코드 편집 흔적과 기계가 작성한 피드백을 포함하는 데이터셋, (2) CoffeeEval: 수정된 코드가 단위 테스트에서 얼마나 유용한지를 평가하여 피드백의 유용성을 반영하는 보상 함수입니다. 이 둘을 통해 Coffee-Gym은 RL을 이용한 피드백 모델 훈련을 위한 고품질 데이터셋의 부족 문제를 해결합니다.

- **Performance Highlights**: Coffee-Gym을 적용한 결과, 오픈 소스 코드 LLMs의 코드 편집 능력을 향상시키는 데 있어 기준 모델보다 성능이 뛰어난 피드백 모델을 이끌어냈으며, 클로즈드 소스 LLMs와 비교 가능하게 만듭니다. 데이터셋과 모델 체크포인트는 공개될 예정입니다.



### 2D-TPE: Two-Dimensional Positional Encoding Enhances Table Understanding for Large Language Models (https://arxiv.org/abs/2409.19700)
- **What's New**: 이 논문에서는 LLM(대형 언어 모델)이 2D 테이블 구조의 공간적 정보를 보존하지 못하고 1D 시퀀스로 평탄화하는 과정에서 발생하는 문제를 분석합니다. 또한, ‘2D-TPE(이차원 테이블 위치 인코딩)’라는 새로운 방법을 제안하여 테이블의 구조를 효과적으로 보존하면서 LLM의 성능을 향상시키는 방법을 소개합니다.

- **Technical Details**: 2D-TPE는 각 attention head가 테이블의 문맥을 인식하는 데 필요한 토큰의 순서를 동적으로 선택할 수 있도록 설계되었습니다. 이 변별화된 순서는 열 또는 행 단위로 탐색할 수 있는 다양한 Traversal 모드를 나타냅니다. 2D-TPE는 계산 효율성을 유지하면서도 중요한 공간 정보를 손실하지 않도록 돕습니다.

- **Performance Highlights**: 5개의 다양한 기준에서 진행된 실험 결과에 따르면, 2D-TPE는 기존의 1D 위치 인코딩을 사용하는 강력한 기준 모델보다 우수한 성능을 보였으며, 특히 대규모 테이블에서도 균형 잡힌 성능을 유지했습니다. 2D-TPE는 TFLOPs 및 메모리 사용 면에서 2% 미만의 추가 비용을 발생시키며, 추론 시간은 평균 13% 증가했습니다.



### CERD: A Comprehensive Chinese Rhetoric Dataset for Rhetorical Understanding and Generation in Essays (https://arxiv.org/abs/2409.19691)
- **What's New**: 본 논문에서는 다양한 수사적 장치 간의 상호 관계를 탐구하는 중국 에세이 수사 체계 데이터셋(CERD)을 제안합니다. CERD는 4개의 일반적인 대분류(메타포, 의인화, 과장, 병렬)와 23개의 세부 분류로 구성되어 있으며, 이를 통해 수사적 이해와 생성의 cinco 개 상호 관련 하위 작업을 제공합니다.

- **Technical Details**: CERD는 503개의 초등 및 중학교 학생들의 에세이로부터 수집되었으며, 각 에세이는 평균 20.57 문장과 706.47 토큰으로 이루어져 있습니다. 데이터셋은 수사적 이해(Task RC), 형태 분류(Task FC), 내용 분류(Task CC), 구성 요소 추출(Task CE), 수사 생성(Task RG) 등의 다섯 개 하위 작업을 포함하고 있으며, 각 작업은 서로 연계되어 있습니다.

- **Performance Highlights**: 대규모 언어 모델이 대부분의 작업에서 최고의 성능을 달성했으며, 여러 작업을 통합적으로 미세 조정(fine-tuning)함으로써 더욱 향상된 성능을 보였습니다. CERD는 향후 수사 연구를 위한 기준점을 설정하는 데 기여할 것으로 예상됩니다.



### Instruction Embedding: Latent Representations of Instructions Towards Task Identification (https://arxiv.org/abs/2409.19680)
Comments:
          NeurIPS 2024

- **What's New**: 이 논문에서는 Large Language Models (LLMs)의 성능을 향상시키기 위해 Instruction Embedding의 개념을 도입하고 Instruction Embedding Benchmark (IEB)를 구축하여 새로운 평가 기준을 설정하였습니다. 특히, 이 연구는 특정 의미나 지식 정보를 넘어 다양한 작업을 수행하는 데 있어 지시 데이터의 중요성을 강조합니다.

- **Technical Details**: 연구에서는 Instruction Embedding이란 개념을 통해 명령문이 특정 작업 범주를 식별하는 데 초점을 맞춘 새로운 임베딩 방식을 제안합니다. 기존의 텍스트 임베딩 방법은 전반적인 의미 정보를 포착하는 데 중점을 두는 반면, Instruction Embedding은 서로 다른 작업 간의 유사성에 중점을 둡니다. IEB는 47,000개의 샘플과 1,000개 이상의 범주로 구성되어 있으며, Task Differentiation을 위한 기준으로 사용됩니다. 이와 함께 Prompt-based Instruction Embedding (PIE) 방법이 제안되어, 특정 작업 유형에 대한 집중을 통해 더 나은 결과를 도출할 수 있도록 하였습니다.

- **Performance Highlights**: IEB에서의 두 가지 설계된 작업에 대해 PIE 방법을 평가한 결과, 다른 임베딩 방법들보다 우수한 성능이 입증되었습니다. PIE는 작업 범주 정확히 식별하는 데 효과적이며, 제안된 Instruction Embedding의 사용은 전통적인 텍스트 임베딩보다 instruction 관련 작업에 더 적합함을 보여주었습니다.



### Modeling Layout Reading Order as Ordering Relations for Visually-rich Document Understanding (https://arxiv.org/abs/2409.19672)
Comments:
          Accepted as a long paper in the main conference of EMNLP 2024

- **What's New**: 비주얼 문서(VrDs)의 레이아웃 읽기 순서를 모델링하고 활용하는 것이 문서 지능(Document Intelligence)에서 중요하다는 점을 강조합니다. 기존의 연구들은 레이아웃 요소의 순열(permutation)로 읽기 순서를 표현했으나, 제안된 방식은 보다 완전한 읽기 순서 정보를 제공하기 위해 레이아웃 요소에 대한 순서 관계(ordering relations)를 모델링합니다.

- **Technical Details**: 읽기 순서를 정의하기 위해 두 가지 개념을 도입합니다: (1) ISDR(Immediate Succession During Reading) - 레이아웃 요소 간의 방향 비순환 관계이며, (2) GSDR(Generalized Succession During Reading) - ISDR의 전이적 폐쇄로서 레이아웃 요소 간의 엄격한 부분 순서 관계입니다. 이들은 읽기 순서를 관계 추출(task) 문제로 재구성하여 새로운 기준 데이터셋을 통해 평가합니다.

- **Performance Highlights**: 제안된 방법은 이전의 순열 기반 방식보다 훨씬 뛰어난 성능을 보이며, 리딩 오더 관계 정보를 활용한 하향식 모델에서 SOTA(State-of-the-Art) 결과를 달성했습니다. 또한, 가상의 읽기 순서 정보(pseudo reading order information)를 사용하여 다양한 도메인에서 성능 향상을 확인했습니다.



### Can Large Language Models Analyze Graphs like Professionals? A Benchmark, Datasets and Models (https://arxiv.org/abs/2409.19667)
Comments:
          NeurIPS 2024

- **What's New**: 이번 논문에서는 LLMs(대형 언어 모델)의 그래프 분석 능력을 평가하기 위해 새로운 ProGraph 벤치를 제안하였습니다. 이 벤치는 프로그램 코드 작성을 기반으로 한 해결책을 요구하며, 기존의 작은 그래프에 대한 직접적인 추론을 요구하는 방법과 차별화됩니다.

- **Technical Details**: ProGraph 벤치는 512개의 문제로 구성되어 있으며, 기본 그래프 이론, 그래프 통계 학습, 그래프 임베딩의 세 가지 범주를 포함합니다. 이를 위해 6개의 인기 있는 Python 라이브러리를 사용하고, LLM4Graph 데이터셋을 통해 문서와 코드 데이터를 결합하여 LLM들이 API를 통해 문제를 해결할 수 있도록 지원합니다.

- **Performance Highlights**: 폐쇄형 모델(Claude, GPT, Gemini)은 ProGraph에서 25-36%의 정확도를 기록하며, LLM4Graph를 활용한 RAG를 통해 37-46%로 향상될 수 있었습니다. 오픈 소스 모델(Llama3, Deepseek Coder)의 경우 12-24%의 정확도에서 45-47%로 개선되었습니다. 이러한 결과는 현재 LLM들이 ProGraph의 도전적인 문제를 효과적으로 해결할 수 있도록 해주는 LLM4Graph의 유용성을 강조합니다.



### Identifying Knowledge Editing Types in Large Language Models (https://arxiv.org/abs/2409.19663)
Comments:
          Under review

- **What's New**: 최근 대규모 언어 모델(LLMs)의 지식을 업데이트하기 위한 효율적인 방법으로 지식 수정(Knowledge editing)이 각광받고 있으나, 이를 악용할 수 있는 위험이 있음이 지적되었습니다. 이에 따라, 우리는 지식 수정 유형 식별(KETI)이라는 새로운 작업을 제안하고, 이를 통해 악의적인 수정을 식별할 수 있는 방법론을 마련했습니다.

- **Technical Details**: KETI는 서로 다른 유형의 지식 수정을 식별하는 것을 목표로 하며, KETIBench라는 새로운 벤치마크를 포함합니다. 이 데이터셋은 다양한 독성 수정과 하나의 무해한 사실 수정으로 구성되어 있습니다. 우리는 네 가지 고전적 분류 모델과 세 가지 BERT 기반 모델을 사용하여 공개 및 폐쇄 소스 LLM에서의 수정 유형을 식별하는 데 필요한 기준 모델을 개발했습니다.

- **Performance Highlights**: 실험 결과, 모든 분야의 기준 식별자가 상당한 성능을 달성했으며, KETIBench에서 평균 F1 점수는 0.789였습니다. 그러나 20%의 오류 식별률은 여전히 개선이 필요하며, 성능은 지식 수정 방법의 신뢰성과 독립적이며, 미지의 출처에서의 수정 식별 능력을 보여주었습니다.



### Multimodal Misinformation Detection by Learning from Synthetic Data with Multimodal LLMs (https://arxiv.org/abs/2409.19656)
Comments:
          EMNLP 2024 Findings

- **What's New**: 이번 연구는 합성 데이터(synthetic data)를 이용하여 실제 다중모달 허위 정보(multimodal misinformation) 탐지 성능을 향상시키는 새로운 접근 방식을 제안합니다. 대학들과의 협력을 통해 기존의 고비용 사실 확인 데이터의 부족 문제를 해결하고자 했습니다.

- **Technical Details**: 연구에서는 두 가지 모델 불변 데이터 선택(method) 방식을 사용하여 합성 데이터와 실제 데이터의 분포를 맞추는 방법을 제안합니다. 첫 번째 방법은 의미(similarity) 기반 선택으로, 검증 데이터와 높은 유사성을 가진 합성 인스턴스를 우선적으로 선택합니다. 두 번째 방법은 최적 수송 문제(Optimal Transport)에서 얻은 그래디언트 정보를 활용하여 합성 데이터 중 목표 실제 분포에 가까운 데이터 포인트를 선택합니다.

- **Performance Highlights**: 이 연구의 실험 결과, 작은 규모의 다중모달 대형 언어 모델(MLLM, 13B)은 실제 사실 확인 데이터셋에서 GPT-4V 모델을 초월하는 성능을 보였습니다. 이는 데이터 선택 방식이 다양한 MLLM 규모와 패밀리에 걸쳐 뛰어난 효과를 발휘함을 보여줍니다.



### Assessment and manipulation of latent constructs in pre-trained language models using psychometric scales (https://arxiv.org/abs/2409.19655)
- **What's New**: 최근 대규모 언어 모델에서 인간과 유사한 성격 특성이 발견되었으며, 이러한 편향이 인간의 잠재적 심리적 구조와 일치한다는 가설이 제기되었습니다. 본 연구에서는 심리 측정 도구를 통해 이러한 특성을 평가하는 새로운 방법론을 제시합니다.

- **Technical Details**: 연구는 88개의 공개 모델 샘플을 분석하여 불안(anxiety), 우울(depression), 일관성의 감각(Sense of Coherence)과 같은 인간과 유사한 정신 건강 관련 구조의 존재를 보여줍니다. 또한 자연어 추론(NLI) 프롬프트를 기반으로 심리 측정 질문지를 재구성하고, 이를 통해 특정 모델의 편향을 두 가지 방법으로 평가합니다.

- **Performance Highlights**: 본 연구의 기여는 크게 네 가지로 나뉩니다: 1) 대화형 및 비대화형 모델 모두에 적용 가능한 심리적 특성 평가 방법론. 2) PLM의 잠재 구조 평가를 위한 Python 라이브러리. 3) 표준 질문지를 기반으로 하는 NLI 프롬프트 설계 방법론. 4) 정신 건강 평가와 관련된 NLI 프롬프트 데이터셋과 그 검증 과정.



### Learning Attentional Mixture of LoRAs for Language Model Continual Learning (https://arxiv.org/abs/2409.19611)
Comments:
          12 pages, 5 figures

- **What's New**: 이 논문에서는 Attentional Mixture of LoRAs (AM-LoRA)라는 새로운 지속적 학습 접근법을 제안합니다. AM-LoRA는 여러 작업에서 지속적으로 지식을 습득할 수 있도록 다양한 LoRA의 지식을 적응적으로 혼합하는 방법입니다.

- **Technical Details**: AM-LoRA는 두 가지 핵심 구성 요소로 나뉘어집니다: Task-specific LoRA Matrix Sequences와 Attentional Selector. Task-specific LoRA Matrix Sequences는 다양한 작업에 대한 지식을 학습하는 데 사용되며, Attentional Selector는 학습 과정에서 다양한 LoRA의 지식을 필터링하고 혼합합니다. 또한, AM-LoRA는 $L1$ norm을 학습 과정에 도입하여 주의 벡터를 보다 희소하게 만들고, 적절한 LoRA를 선택하게 합니다.

- **Performance Highlights**: 실험 결과, AM-LoRA는 기존의 SOTA(최신 기술 동향) 방법들에 비해 우수한 성능을 보이며, 여러 작업을 연속적으로 학습하면서도 치명적인 망각(catastrophic forgetting) 문제를 효과적으로 완화하는 것으로 나타났습니다.



### DiMB-RE: Mining the Scientific Literature for Diet-Microbiome Associations (https://arxiv.org/abs/2409.19581)
Comments:
          13 pages, 2 figures. Please refer to the supplementary material if needed

- **What's New**: 본 연구는 식이요법과 미생물군 집단 사이의 연관성을 강조하는 DiMB-RE라는 포괄적인 말뭉치를 개발하였으며, 이는 165개의 논문에서 추출한 14,450개의 개체(Entity)와 4,206개의 관계(Relation)로 구성됩니다. DiMB-RE는 영양과 미생물에 관한 정보 추출 모델도 포함하고 있습니다.

- **Technical Details**: DiMB-RE는 15개의 개체 유형(예: Nutrient, Microorganism)과 13개의 관계 유형(예: increases, improves)으로 주석 처리된 데이터 세트입니다. 또한, BERT(Bidirectional Encoder Representations from Transformers) 기반 최신 자연어 처리(NLP) 모델을 사용하여 개체 명명, 트리거 및 관계 추출을 수행했습니다. 이 모델들은 F1 점수 0.760을 기록했지만, 관계 추출의 성능은 0.356으로 다소 낮았습니다.

- **Performance Highlights**: DiMB-RE는 개인 맞춤형 영양에 필요한 다양한 관계를 포함한 가장 큰 데이터 세트로, 생물 의학 문헌 채굴에 대한 기준이 되는 말뭉치로 활용될 수 있습니다. 연구 결과, NLP 모델들은 개인 맞춤형 영양 관련 연구 결과를 보다 잘 해석하고 새로운 가설 생성을 지원할 것으로 기대됩니다.



### Mitigating the Negative Impact of Over-association for Conversational Query Production (https://arxiv.org/abs/2409.19572)
Comments:
          Information Processing & Management

- **What's New**: 이번 논문에서는 대화 이력(conversational histories)에서 검색 쿼리(query)를 생성하는 새로운 접근 방식을 제안한다. 기존의 모델들은 데이터 필요성(data hunger) 문제와 대화 이력에서 중요한 개념을 무시하거나 비관련 개념을 생성하는 문제를 안고 있으며, 이는 일반적으로 과잉 연관(over-association) 현상 때문이라는 점을 시사한다.

- **Technical Details**: 대화 쿼리 생성(task of conversational query generation)을 위한 모델을 발전시키기 위해, 저자들은 두 가지 인스턴스 수준의 가중치 조정 전략(instance-level weighting strategies)을 제안한다. 첫 번째는 데이터 기반 가중치 조정(data-based weighting)으로, 쿼리의 과잉 연관 정도에 따라 학습 속도를 조절한다. 두 번째는 모델 기반 가중치 조정(model-based weighting)으로, 모델이 자체 예측을 통해 학습하도록 유도한다. 이 두 가지 접근법을 통해 과잉 연관 문제를 완화하고 더 신뢰할 수 있는 쿼리를 생성한다.

- **Performance Highlights**: 실험은 Wizard-of-Internet 및 DuSinc 벤치마크에서 수행되었으며, 제안된 전략은 2%-5%의 성능 향상을 보여주었다. 또한, 새로운 모델은 대화 이력에서 더 나은 개념을 선택하고, 기본 모델에 비해 10배 더 효율적인 데이터 사용 효율성을 보였다.



### Abstractive Summarization of Low resourced Nepali language using Multilingual Transformers (https://arxiv.org/abs/2409.19566)
- **What's New**: 이번 연구는 네팔어를 위한 추상 요약(abstractive summarization) 분야에 대한 탐구로, 기존의 추출 요약(extractive summarization) 연구가 많이 이루어진 것과 대조적으로, 저자들은 mBART와 mT5와 같은 다국어 변환기 모델(multilingual transformer models)을 사용하여 네팔 뉴스 기사의 헤드라인을 생성하는 방법을 제안합니다.

- **Technical Details**: 이 연구에서는 다양한 네팔 뉴스 포털에서 수집한 데이터를 바탕으로 요약 데이터셋을 생성하고, 그 후 mBART와 mT5 모델을 다양한 전략으로 fine-tuning하였습니다. 모델의 성능은 ROUGE 점수와 인간 평가를 통해 확인하였으며, 교수 평가에서는 주제의 관련성, 유창성, 간결성, 정보성, 사실 정확성 및 범위와 같은 기준에 따라生成된 요약 중 가장 좋은 것을 선택하게 하였습니다. 특히, 4-bit quantized mBART with LoRA 모델이 다른 모델에 비해 효과적으로 네팔 뉴스 헤드라인을 생성하였습니다.

- **Performance Highlights**: fine-tuned 모델 평가 결과, 4-bit quantized mBART with LoRA 모델은 34.05%의 비율로 인간 평가에서 선택되었으며, 네팔 뉴스 헤드라인 생성을 위한 다른 모델보다 우수한 성능을 보여주었습니다.



### Unlabeled Debiasing in Downstream Tasks via Class-wise Low Variance Regularization (https://arxiv.org/abs/2409.19541)
Comments:
          Accepted to EMNLP 2024

- **What's New**: 본 연구는 기존의 레이블링된 속성에 의존하지 않고 다양한 속성에 대한 디바이싱(debiasing) 문제를 해결하기 위해 새로운 정규화 기법인 Low Variance Regularization (LVR)을 도입합니다. 이 접근법은 임베딩 내 특성 정보 손실을 줄이면서도 분류 성능을 유지하는 데 초점을 맞춥니다.

- **Technical Details**: Low Variance Regularization (LVR)은 k 개의 클래스를 각각 대표하는 중심(center) 기반으로 정규화를 수행합니다. 이 기법은 레이블이 없는 상태에서 모든 보호 속성(protected attribute)에 대한 영향을 동시에 낮추며 모델의 임베딩에서 분포적 이동을 줄여 공정한 표현을 유지합니다.

- **Performance Highlights**: LVR 기법은 BERT-Base 및 RoBERTa-Base와 같은 기존의 인코더 언어 모델에서 문서 분류 작업을 통해 평가되었으며, 기존의 강력한 디바이싱 기준선을 능가하며, 목표 작업에 대한 성능을 유지하면서 속성 제거를 효과적으로 이루어냈음을 입증하였습니다.



### Mixed Chain-of-Psychotherapies for Emotional Support Chatbo (https://arxiv.org/abs/2409.19533)
Comments:
          13pages, 5 figures

- **What's New**: 이번 논문에서는 정신 건강 지원 챗봇의 필요성을 강조하며, 일반적인 응답을 넘어서는 심리 치료 관점에서의 통합 분석을 통한 맞춤형 솔루션 제공을 목표로 한 PsyMix라는 챗봇을 제안합니다.

- **Technical Details**: PsyMix는 Chain-of-Psychotherapies (CoP)로 알려진 다양한 심리 치료 접근 방식을 통합하여 사용자의 상태를 분석하고, 이 분석을 통해 응답을 생성합니다. 이는 Supervised Fine-Tuning (SFT)으로 조정된 LLM(대형 언어 모델)을 통해 이루어지며, 여러 심리 치료 이론(예: Cognitive Behavioral Therapy, CBT)이 적용됩니다.

- **Performance Highlights**: PsyMix는 ChatGPT의 성능 기준선을 초과하는 것으로 나타났으며, 인간 상담사와 비교하여 유사한 수준의 공감을 보여주는 응답 성능을 보여주었습니다.



### LANDeRMT: Detecting and Routing Language-Aware Neurons for Selectively Finetuning LLMs to Machine Translation (https://arxiv.org/abs/2409.19523)
- **What's New**: 최근에 발표된 LANDeRMT는 대규모 언어 모델(LLMs)을 다국어 기계 번역(MT) 작업에 선택적으로 미세 조정하는 혁신적인 프레임워크입니다. 이 프레임워크는 언어에 대한 인식과 관련해 신경 네트워크 내의 뉴런들을 분류함으로써 파라미터 간섭(parameter interference)과 파국적 망각(catastrophic forgetting) 문제를 해결하는 데 중점을 두고 있습니다.

- **Technical Details**: LANDeRMT는 Feed-Forward Network(FFN) 내의 뉴런이 다국어 MT 작업에 대한 인식 정도를 평가하고, 이를 바탕으로 비활성화 뉴런(unactivated neurons), 언어 일반 뉴런(language-general neurons), 언어 특정 뉴런(language-specific neurons)으로 분류합니다. 미세 조정 과정에서는 현재 언어 쌍에 해당하는 언어 일반 및 언어 특정 뉴런의 파라미터만 업데이트하며, 이를 통해 파라미터 간섭 문제를 완화합니다.

- **Performance Highlights**: 실험 결과, LANDeRMT는 여러 언어 쌍에 대해 강력한 기준선보다 번역 품질을 상당히 향상시켰으며, 다국어 기계 번역에 있어서 최첨단 성과를 보여주었습니다. 이 모델은 다양한 설정에 강인함을 입증하였습니다.



### CoT-ST: Enhancing LLM-based Speech Translation with Multimodal Chain-of-Though (https://arxiv.org/abs/2409.19510)
- **What's New**: 이번 연구에서는 Speech Language Models (SLMs)의 내재된 추론 능력을 활용하기 위한 새로운 3단계 훈련 프레임워크인 Chain-of-Thought Speech Translation (CoT-ST) 모델을 소개하였습니다. 이 모델은 Speech Translation (ST) 작업을 음성 인식 및 번역의 단계로 세분화하여, SLM의 CoT 능력을 활성화하는 방안을 제시합니다.

- **Technical Details**: CoT-ST는 훈련 방법론으로 커리큘럼 학습(curriculum learning)을 사용하여 점진적으로 복잡해지는 작업을 소개합니다. 각 단계는 (1) ASR, (2) Multimodal Machine Translation (MMT), (3) Speech Recognition and Translation (SRT)으로 구성됩니다. 이 구조를 통해 모델은 다양한 작업 및 언어에 대해 효율성을 극대화합니다.

- **Performance Highlights**: CoT-ST 모델은 CoVoST-2 데이터셋에서 SOTA (state-of-the-art) 성능을 보였으며, BLEU 점수에서 en-ja 쌍의 경우 30.5에서 30.8로, en-zh 쌍의 경우 45.2에서 47.7로, MuST-C에서도 19.6에서 21.2로 상승했습니다. 이 성과는 SLM의 CoT 능력 활용을 통해 이뤄진 것으로, 모델의 전반적인 성능과 회복력을 강화합니다.



### Transforming Scholarly Landscapes: Influence of Large Language Models on Academic Fields beyond Computer Scienc (https://arxiv.org/abs/2409.19508)
- **What's New**: 이번 연구는 대형 언어 모델(LLM)의 영향력을 NLP(자연어 처리) 분야를 넘어 다른 연구 영역에 대한 첫 번째 체계적이고 실증적인 분석을 제공합니다. 이를 통해 LLM이 비 컴퓨터 과학(non-CS) 분야에서 어떻게 사용되고 있는지를 파악하게 되었습니다.

- **Technical Details**: 이 연구에서는 106개의 LLM과 약 148,000개의 논문을 분석하여 LLM의 인용 패턴을 정량적으로 조사했습니다. LLM은 Transformer 아키텍처를 기반으로 하며, 주로 BERT, GPT-3 및 LLaMA와 같이 작업에 구애받지 않는 모델들이 자주 사용됩니다. 연구 데이터는 Semantic Scholar에서 수집되었습니다.

- **Performance Highlights**: 연구 결과, 언어학, 공학 및 의학 분야가 LLM을 가장 많이 인용하고 있으며, BERT가 비 CS 분야에서 선호되는 LLM으로 자리잡고 있습니다. 또한, LLM은 주로 특정 분야의 문제를 해결하는 데 사용되고 있으며, 비 CS 분야에서는 이론적 토대를 기반으로 하는 LLM 사용이 두드러지고 있습니다.



### A Critical Look at Meta-evaluating Summarisation Evaluation Metrics (https://arxiv.org/abs/2409.19507)
Comments:
          Findings of EMNLP 2024

- **What's New**: 이 논문은 자동 요약 평가 메트릭의 메타 평가(meta-evaluation)에 대한 최근의 연구 동향을 조사하고, 현재의 메트릭이 주요한 문제를 해결하지 못하고 있음을 지적합니다. 특히 뉴스 요약 데이터셋에서만 주로 사용되고 있으며, 생성된 요약의 신뢰성 평가에 연구 초점이 향하고 있음을 강조합니다.

- **Technical Details**: 요약 평가 메트릭의 메타 평가는 여러 데이터셋을 통해 이루어지며, 이러한 메트릭의 효율성을 검증하는 것이 필수적입니다. 메트릭은 생성된 요약 y와 참조 요약을 기반으로 하여 유사성을 측정하거나, 출처 텍스트와 비교하여 정보 지원 여부를 판단하는 방식 등 여러 카테고리로 나눌 수 있습니다. 또한, 사용자가 중심이 되는 품질 차원에 대한 연구가 필요합니다.

- **Performance Highlights**: 기존의 자동 메트릭은 인간 평가와 비교했을 때 신뢰성이 부족하여, GPT-3 모델의 요약이 낮은 점수를 받는 반면 실제 인지 능력에서 다른 모델에 비해 우수한 성과를 나타내는 경우가 종종 발생합니다. 이는 기존 자동 평가 메트릭의 신뢰성을 재검토해야 함을 시사합니다.



### The Nature of NLP: Analyzing Contributions in NLP Papers (https://arxiv.org/abs/2409.19505)
- **What's New**: 이 논문은 Natural Language Processing (NLP) 분야에서 연구 논문을 정량적으로 분석하여 NLP 연구의 정의를 명확히 하고, NLPContributions라는 자동화된 데이터셋을 소개합니다.

- **Technical Details**: 우리는 약 2,000개의 연구 논문의 초록을 사용하여 과학적 기여 내용을 식별하고 이를 분류하는 새로운 작업을 제안하였습니다. 또한, 이 데이터셋을 바탕으로 기계 학습 알고리즘을 훈련시켜 기여에 대한 분석을 수행했습니다.

- **Performance Highlights**: 실험 결과, 기계 학습 기술이 1990년대 초반부터 NLP의 주요 초점으로 자리잡아왔음을 발견했으며, 2020년 이후에는 언어 및 인간에 대한 지식 추가에 대한 관심이 재부각되고 있음을 확인했습니다.



### MedHalu: Hallucinations in Responses to Healthcare Queries by Large Language Models (https://arxiv.org/abs/2409.19492)
Comments:
          14 pages

- **What's New**: 이 연구에서는 대형 언어 모델(LLM)이 실제 의료 질문에 대한 응답에서 발생하는 환각(hallucination)의 최초 연구를 진행했습니다. MedHalu라는 의료 환각 데이터셋을 처음으로 제안하였으며, 다양한 건강 관련 주제와 LLM이 생성한 허위 응답을 포함하고 있습니다.

- **Technical Details**: 제안된 MedHaluDetect 프레임워크는 다양한 LLM의 환각 탐지 능력을 평가하여 LLM의 응답에서 발생하는 환각의 유형과 이들을 식별하는 방법을 포함하고 있습니다. 연구에는 의료 전문가, LLM, 일반인을 포함한 세 그룹의 평가자가 참여하여 누가 의료 환각에 더 취약한지를 조사했습니다.

- **Performance Highlights**: 연구 결과, LLM은 전문가보다 환각 탐지 성능이 낮으며, 일부 경우에는 일반인보다도 성능이 떨어지는 것으로 나타났습니다. 그러나 전문가의 추론을 LLM에 주입하는 expert-in-the-loop 접근 방식을 적용한 결과, 모든 LLM에서 평균 6.3 포인트의 macro-F1 점수가 향상되었습니다.



### HealthQ: Unveiling Questioning Capabilities of LLM Chains in Healthcare Conversations (https://arxiv.org/abs/2409.19487)
- **What's New**: 이 논문에서는 LLM(대형 언어 모델)의 질문 능력을 평가하기 위한 새로운 프레임워크인 HealthQ를 제안합니다. 기존의 디지털 의료 연구는 LLM의 질문-응답 능력을 향상시키기 위한 접근 방식을 탐색해왔으나, LLM이 환자와의 대화 중에 효과적인 질문을 통해 정보를 수집하는 능력은 충분히 연구되지 않았습니다.

- **Technical Details**: HealthQ는 Retrieval-Augmented Generation (RAG), Chain of Thought (CoT), reflective chains 등 여러 LLM 체인을 구현하고, LLM 평가자를 도입하여 생성된 질문의 관련성과 정보성 평가를 수행하는 방법론을 사용합니다. 이 논문은 ROUGE와 NER 기반 비교와 같은 전통적인 자연어 처리(NLP) 메트릭스를 이용해 건강 관련 질의응답 시스템의 질문 능력을 탐색합니다.

- **Performance Highlights**: 이 연구는 LLM 헬스케어 체인에서 질문 능력에 대한 첫 포괄적인 연구를 실시하며, 질문-응답 성능 평가를 위한 새로운 데이터 생성 파이프라인을 개발하고, 다양한 LLM 체인 유형과 전통적인 NLP 메트릭스를 통합하는 평가 방법론을 제안합니다. 이러한 접근방식은 향후 LLM의 질문 능력 연구에 대한 견고한 기준을 제공합니다.



### Overriding Safety protections of Open-source Models (https://arxiv.org/abs/2409.19476)
- **What's New**: 최근 LLMs(대형 언어 모델)의 안전성 강화를 위한 연구가 진행되었으며, 유해 데이터로 파인튜닝(tuning)한 경우 모델의 신뢰성과 유용성이 저하될 수 있음을 실험적으로 입증하였습니다.

- **Technical Details**: 이 연구는 Llama-3.1 8B 모델을 바탕으로 유해 데이터와 안전 데이터로 모델을 각각 파인튜닝 하여 생성된 모델을 비교·분석하였습니다. 유해 데이터로 파인튜닝된 모델은 Harmbench 데이터셋에서 성능이 강화된 것과 동시에 응답의 신뢰성과 유용성이 낮아지는 경향을 보였습니다. 반면, 안전 데이터로 파인튜닝된 모델은 신뢰성이 증가하며 부정확 응답을 줄이는 결과를 나타냈습니다.

- **Performance Highlights**: 유해 모델은 기본 모델에 비해 35% 높은 ASR(공격 성공률)을 기록하여 안전성 보호 기능을 무색하게 하였습니다. 반면, 안전 모델의 ASR은 51.68% 감소하여 상대적으로 더 안전한 응답을 생성하였습니다. 유해 모델은 높은 불확실성과 지식 이동(knowledge drift)을 나타냈으며, 안전 모델은 유사한 상황에서도 최소한의 불확실성을 유지하였습니다.



### INSIGHTBUDDY-AI: Medication Extraction and Entity Linking using Large Language Models and Ensemble Learning (https://arxiv.org/abs/2409.19467)
Comments:
          ongoing work, 24 pages

- **What's New**: 이번 연구에서는 약물 정보 추출과 관련 속성을 텍스트 마이닝하는 상태-of-the-art LLMs를 조사하였으며, Stack-Ensemble 및 Voting-Ensemble과 같은 다양한 앙상블 학습 기법을 적용하여 모델 성능을 향상시켰습니다. 또한, 추출된 약물 용어를 SNOMED-CT 및 BNF 코드로 매핑하는 엔터티 링크 기능도 개발하였습니다.

- **Technical Details**: 이 연구에서는 약물의 복용량(dosage), 경로(route), 강도(strength), 부작용(adverse effects) 등과 같은 속성을 포함한 약물 텍스트 마이닝을 다루며, 임상 NER(named entity recognition) 작업을 위해 8개 모델(BERT, RoBERTa 등)을 미세 조정하는 과정에서 Stack 및 Voting 앙상블 메커니즘을 조사했습니다. 엔터티 링크 기능을 통해 임상 사건을 SNOMED-CT와 BNF에 연결하고, 이는 이후 ICD 및 dm+d와 매핑됩니다.

- **Performance Highlights**: 앙상블 학습 결과는 BERT, RoBERTa, BioBERT 등 개별 모델보다 효율적으로 성능을 끌어올렸습니다. 이 연구에서 사용된 n2c2-2018 데이터셋에 대한 초기 평가 결과, 각 모델의 성능 개선을 확인했습니다.



### Scalable Fine-tuning from Multiple Data Sources:A First-Order Approximation Approach (https://arxiv.org/abs/2409.19458)
Comments:
          16 pages

- **What's New**: 본 논문에서는 n개의 보조 작업으로부터의 정보를 최적 활용하여 특정 작업을 위해 언어 모델(LM)을 미세 조정하는 문제를 다룹니다. 보조 작업 중 일부가 목표 작업의 성능 향상에 유용하지 않을 수 있으므로 적절한 보조 작업의 하위 집합을 선택하는 것이 중요합니다. 이 논문은 반복 훈련 없이 모델 미세 조정 성능을 추정할 수 있는 새로운 알고리즘을 소개합니다.

- **Technical Details**: 제안된 알고리즘은 먼저 모든 작업의 데이터를 사용한 멀티태스크 훈련을 통해 메타 초기화(meta initialization)를 얻습니다. 이후 메타 초기화의 함수 값과 그래디언트를 사용하여 하위 집합의 모델 미세 조정 손실을 근사합니다. 이 과정에서 USDA (Uniformly Stochastic Dimensionality Reduction)를 사용하여 CPU에서 몇 초 만에 미세 조정 성능을 추정할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 GradEx 접근법은 기존 방법보다 30배 빠르게 하위 집합 선택을 처리하면서도 실제 미세 조정 성능의 1% 오차 이내로 정확도를 유지합니다. 또한, GradEx는 강의 지침 조정(instruction tuning) 및 체인 사고 미세 조정(chain-of-thought fine-tuning)의 하류 평가에서 기존 방법보다 평균 3.8% 향상된 성능을 보였습니다.



### Crafting Personalized Agents through Retrieval-Augmented Generation on Editable Memory Graphs (https://arxiv.org/abs/2409.19401)
Comments:
          This paper has been accepted by EMNLP 2024

- **What's New**: 모바일 인터넷 시대에 사용자의 불규칙한 데이터를 효과적으로 관리하고 활용하여 개인화된 AI 어시스턴트를 만드는 새로운 작업이 소개되었습니다. 본 논문에서는 스마트폰 기억(memoies)을 활용하여 LLM(대형 언어 모델) 기능을 향상시키는 EMG-RAG라는 솔루션을 제안합니다.

- **Technical Details**: EMG-RAG는 Retrieval-Augmented Generation (RAG) 기술과 Editable Memory Graph (EMG)를 결합하여 개인화된 에이전트를 만드는 것을 목표로 합니다. 이 접근법은 강화 학습을 사용하여 데이터 수집, 편집 가능성, 선택 가능성의 세 가지 주요 문제를 해결합니다. EMG는 메모리의 복잡한 관계를 포착하기 위해 트리 구조를 가지고 있으며, 사용자 메모리를 효율적으로 편집하고 검색할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실제 데이터셋을 바탕으로 한 실험에서 EMG-RAG는 기존 접근 방식보다 약 10% 향상된 성능을 기록했습니다. 또한, 향상된 개인화된 에이전트는 실제 스마트폰 AI 어시스턴트에 통합되어 사용자 경험이 개선되었습니다.



### Zero-Shot Multi-Hop Question Answering via Monte-Carlo Tree Search with Large Language Models (https://arxiv.org/abs/2409.19382)
Comments:
          Work in Progress

- **What's New**: 본 논문에서는 Monte-Carlo tree search (MCTS)를 기반으로 한 Zero-shot multi-hop Question Answering (MZQA) 프레임워크를 소개합니다. 기존의 few-shot 예제 없이도 문제를 해결할 수 있는 새로운 zero-shot prompting 방식을 제안합니다.

- **Technical Details**: MZQA는 MCTS를 채택하여 적절한 추론 경로를 선택합니다. 이를 통해 연속적인 추론 과정에서 발생할 수 있는 오류 전파를 완화하고, behavioral cloning (MZQA-BC) 접근법을 통해 연산 속도를 10배 이상 향상시킵니다. 또한, 표준 벤치마크인 HotpotQA, 2WikiMultihopQA, MuSiQue에서 우리의 방법의 효능을 검증합니다.

- **Performance Highlights**: MZQA와 MZQA-BC는 기존 방법들과 비교하여 성능의 손실을 최소화하면서도 추론 속도에서 월등한 성과를 보여줍니다. 이들 방법은 특히 복잡한 다단계 질문 응답(MHQA) 작업에 적합하며, 기존 방법들의 한계를 효과적으로 극복합니다.



### MetaMath: Integrating Natural Language and Code for Enhanced Mathematical Reasoning in Large Language Models (https://arxiv.org/abs/2409.19381)
- **What's New**: 이 논문은 수학적 추론 문제를 해결하는데 있어 자연어(NL)와 코드(coding)의 상호작용을 연구하고, GPT-4o-mini와 LLama-3.1-8b-Turbo와 같은 최첨단 대형 언어 모델(LLM)을 사용하여 두 가지 접근법의 효과를 비교합니다. 또한, 이 연구에서 MetaMath라는 새로운 프롬프트 방법론을 제안하여 LLM이 가장 적합한 추론 형태를 동적으로 선택할 수 있게 합니다.

- **Technical Details**: 이 논문에서는 Chain-of-Thought prompting (CoT), Program-aided Language Models (PAL), CodeNL, NLCode의 네 가지 접근 방식을 사용하여 LLM의 수학적 추론 능력을 평가합니다. MetaMath는 이러한 접근 방식들 사이에서 문제 유형에 따른 최적의 방법을 선택하여 성능을 향상시키도록 설계되었습니다.

- **Performance Highlights**: GPT-4o-mini 모델은 GSM8K 데이터셋에서 90% 이상의 정확도를 기록하며 뛰어난 문제 해결 능력을 보여주었습니다. 특히 CoT 방법은 AIME 데이터셋에서 40%의 정확도를 달성하여 자연어만으로도 복잡한 문제를 잘 처리할 수 있음을 나타냈습니다. MetaMath를 적용한 경우 가장 높은 성능을 나타냈고, Llama-3.1-8B 모델에서는 NLCode가 우수한 결과를 보였습니다. 또한 자연어 추론을 포함한 방법이 단순히 프로그래밍에 의존하는 방법보다 뛰어난 성과를 기록했습니다.



### Visual Question Decomposition on Multimodal Large Language Models (https://arxiv.org/abs/2409.19339)
Comments:
          Accepted to EMNLP2024 Findings

- **What's New**: 본 논문은 멀티모달 대형 언어 모델(MLLMs)의 질문 분해(Question Decomposition) 능력을 탐구하고 있으며, 기존의 단일 모드 언어 모델과는 달리 MLLMs의 응답 품질을 향상시키는 새로운 데이터셋인 DecoVQA+를 제안하고 있습니다.

- **Technical Details**: 시스템적 평가 프레임워크를 도입하여 MLLMs의 분해된 하위 질문의 품질을 평가하고, 선택적 분해(selective decomposition)를 위한 훈련 목표를 포함하는 효율적인 파인튜닝(pipeline)을 제안하고 있습니다.

- **Performance Highlights**: 파인튜닝된 MLLMs는 VQA 벤치마크 데이터셋에서 하위 질문 품질의 현저한 개선과 선택적 질문 분해에서 더 높은 정확도를 달성했습니다.



### Designing Domain-Specific Large Language Models: The Critical Role of Fine-Tuning in Public Opinion Simulation (https://arxiv.org/abs/2409.19308)
- **What's New**: 이 논문은 환경 정책에 대한 의견을 시뮬레이션하기 위해 대형 언어 모델(LLMs)을 세부 조정하는 새로운 접근 방식을 제안합니다. 이를 위해 영국 가정 종단 연구(UK Household Longitudinal Study)의 데이터를 활용합니다.

- **Technical Details**: 대형 언어 모델의 세부 조정(fine-tuning)을 통해 나이(age), 소득(income), 교육(education), 지역(region)과 같은 사회 인구학적 요인에 따라 모델을 조정하여 의견 생성의 정확성을 개선합니다. 다양한 합성 프로필(synthetic profiles)을 모방하여, 세부 조정된 모델이 인구 집단 간의 미세한 차이를 보다 효과적으로 캡처합니다.

- **Performance Highlights**: Chi-Squared, Cosine Similarity, Jaccard Index, KL-divergence와 같은 메트릭을 사용하여 합성 및 실제 의견 데이터 간의 강한 정렬을 보여줍니다. 이러한 결과는 더 정확하고 윤리적인 정책 시뮬레이션을 위한 LLM의 사회적 맥락에 대한 맞춤화의 중요성을 강조합니다.



### Perception Compressor:A training-free prompt compression method in long context scenarios (https://arxiv.org/abs/2409.19272)
Comments:
          9 pages, 2 figures

- **What's New**: 이번 연구에서는 Perception Compressor라는 새로운 프롬프트 압축 방법을 소개합니다. 이는 훈련 없는 방법으로, 대규모 언어 모델(LLM)이 긴 컨텍스트에서 겪는 문제를 해결하기 위해 개발되었습니다. 연구의 주요 기여는 동적 압축 비율 할당 및 키 정보를 유지하면서 불필요한 토큰을 제거하는 방식입니다.

- **Technical Details**: Perception Compressor는 세 가지 주요 구성 요소로 이루어져 있습니다: (1) 이중 경량 비율 할당기(dual-slope ratio allocator) - 프롬프트의 다양한 구성 요소에 대해 압축 비율과 오픈북 비율을 동적으로 할당합니다. (2) 지각 검색기(perception retriever) - 지침 및 유도 질문을 활용하여 가장 관련 있는 시연을 검색하고 유지합니다. (3) 반유도 반복 압축(semi-guided iterative compression) - 키 정보 토큰(KITs)을 유지하고, 입력 질문과 무관한 고퍼플렉시티(non-key information tokens, NITs)를 제거합니다.

- **Performance Highlights**: Perception Compressor는 NaturalQuestions, LongBench, MuSiQue와 같은 긴 컨텍스트 벤치마크에서 기존 방법들에 비해 월등한 성과를 보여주며, 최신 기술(State-of-the-art) 성능을 달성했습니다.



### LISTN: Lexicon induction with socio-temporal nuanc (https://arxiv.org/abs/2409.19257)
- **What's New**: 본 논문은 사람들 사이의 사교적 및 시간적 맥락을 고려한 새로운 in-group lexicon induction 방법인 LISTN을 제안합니다. 이는 기존의 구식 어휘를 대체하며, 특히 온라인의 남성우월주의(anti-women) 커뮤니티에서 활용됩니다.

- **Technical Details**: LISTN 방법은 dynamic word and user embeddings를 활용하여 커뮤니티의 사회적 구조 및 시간적 진화를 포착합니다. 이 방법론은 Reddit에서 4백만 개 이상의 발화를 사용하여 훈련되었습니다.

- **Performance Highlights**: LISTN 기반 메서드는 기존 방법보다 뛰어난 성능을 보여주며, lexicon induction에 대한 평균 precision 점수는 0.77에 달합니다. 또한, 이 연구는 455개의 새로운 manosphere 용어와 각 용어의 하위 커뮤니티에 대한 관련성을 정량화한 결과를 포함합니다.



### Edit-Constrained Decoding for Sentence Simplification (https://arxiv.org/abs/2409.19247)
Comments:
          Accepted by EMNLP2024-Findings

- **What's New**: 본 연구에서는 문장 단순화를 위한 lexically constrained decoding에 기반한 edit operation을 제안합니다. 이전 연구들은 lexically constrained decoding의 효과를 확인했지만, 이러한 제약 조건들은 느슨하여 최적이 아닌 결과를 초래할 수 있었습니다. 우리는 문장 단순화에서 수행되는 edit operation을 복제하고, stricter satisfaction conditions를 정의하는 제약 조건을 설계하여 이 문제를 해결합니다.

- **Technical Details**: 문장 단순화에서는 세 가지 edit operation (insertion, deletion, substitution)을 수행하여 문장을 재작성합니다. 우리는 NeuroLogic Decoding을 확장하여 이러한 edit operation을 constrained decoding을 통해 실행합니다. 이 방법은 generation likelihood와 constraint satisfaction을 최대화하는 가설을 찾기 위해 beam search를 활용합니다.

- **Performance Highlights**: 제안된 방법은 Turk, ASSET, AutoMeTS와 같은 세 가지 영문 단순화 코퍼스를 통해 이전 연구보다 일관되게 더 우수한 성능을 보였습니다. 이는 참조에서 추출된 oracle 제약 조건을 사용했을 때 확인되었으며, 단순 모델로 예측된 제약 조건에서도 일관된 효능을 나타냈습니다.



### SciDoc2Diagrammer-MAF: Towards Generation of Scientific Diagrams from Documents guided by Multi-Aspect Feedback Refinemen (https://arxiv.org/abs/2409.19242)
Comments:
          Code and data available at this https URL

- **What's New**: 새로운 연구인 SciDoc2Diagram에서는 학술 논문에서 과학적 다이어그램을 자동으로 생성하는 작업을 제안합니다. 사용자 의도에 맞는 과학적 다이어그램을 생성하기 위해 세심한 정보 추출과 단계별 파이프라인을 개발했습니다. 또한, 새로운 벤치마크 데이터셋인 SciDoc2DiagramBench를 소개합니다.

- **Technical Details**: SciDoc2Diagrammer라는 다단계 파이프라인을 통해 논문에서 사용자 의도에 따라 다이어그램을 생성합니다. 이 과정에서 내부 코드 생성을 사용하여 사용자 요청에 최적화된 결과를 도출합니다. 최종 다이어그램의 질을 향상시키기 위해 SciDoc2Diagrammer-Multi-Aspect-Feedback (MAF)라는 정제 전략을 도입하였습니다.

- **Performance Highlights**: 이 연구를 통해 생성된 다이어그램은 기존 모델들과 비교하여 사실적 정확성과 시각적 매력을 크게 향상시켰습니다. 자동 및 인간 평가 모두에서 우수한 성과를 보여주었고, 복잡한 흐름도와 표가 더욱 신뢰성과 완전성을 개선하는 데 유리함을 입증했습니다.



### HM3: Heterogeneous Multi-Class Model Merging (https://arxiv.org/abs/2409.19173)
- **What's New**: 본 논문은 다양한 기계 학습 모델들이 결합하여 하나의 다기능 모델로 통합할 수 있는 방법인 Heterogeneous Multi-Class Model Merging (HM3)을 제안합니다. 이 접근 방식은 각각의 모델의 정확도를 손실 없이 통합할 수 있도록 하며, 기존의 복잡한 학습 과정 없이 가능합니다.

- **Technical Details**: HM3 기법은 서로 다른 레이블 공간을 가진 다중 클래스 분류기를 결합하는 방법론입니다. 이 방법은 모델 항목의 가중치를 결합하여 최적의 성능을 내는 동시에, 모델 간의 학습이나 추가 훈련 없이 CPU에서 빠르게 실행될 수 있습니다. 실험 결과, BERT 기반의 guard 모델들을 결합하여 F1-score를 평균적으로 높이고, 추론 시간을 44%까지 단축하는 결과를 보였습니다.

- **Performance Highlights**: Heterogeneous Multi-Class Model Merging 기법을 통해 기존의 모델들보다 평균 F1-score가 개선된 BERT 기반 guard 모델을 확인했으며, 잔여 부정적 성과를 가진 hate speech 분류기가 self-merging 과정을 통해 성능이 향상되었음을 보여주었습니다.



### Can LLMs Really Learn to Translate a Low-Resource Language from One Grammar Book? (https://arxiv.org/abs/2409.19151)
- **What's New**: 이 논문은 XLR (Extremely Low-Resource) 언어의 기계 번역에서 기존의 문법 설명보다 병렬 예문이 더 효과적임을 보여준다. XLR 언어인 Kalamang과 영어 간 번역에서 문법서의 병렬 예문을 활용하는 방법을 제안하고, 다른 저자들의 주장과 달리 문법 설명은 효과적이지 않음을 밝혀냈다.

- **Technical Details**: 이 연구에서는 LLM (Large Language Models)을 활용하여 병렬 문서와 문법서를 비교하여 번역 성과를 분석하였다. 특히 병렬 예문이 지배적으로 작용하며, LLM이 문법 설명을 효과적으로 활용하지 못하는 결과를 도출하였다. 귀납적으로 Nepali와 같은 저자원 언어에서도 유사한 패턴을 발견하였다. 실험은 ChrF 및 ChrF++ 점수를 기반으로 진행되었다.

- **Performance Highlights**: 모델을 병렬 데이터로 정밀 조정(fine-tuning)하여 LLM의 성과에 근접하는 결과를 냈으며, Llama-3.1-8B 설정보다 20 포인트 높은 성과를 기록하였다. 또한, 새로운 유형의 언어학적 프롬프트를 통해 문법 과제에서의 성과를 크게 향상시켰으며, 병렬 예문을 활용한 접근법이 가장 효율적임을 입증하였다.



### On the Power of Decision Trees in Auto-Regressive Language Modeling (https://arxiv.org/abs/2409.19150)
Comments:
          Accepted to NeurIPS 2024

- **What's New**: 이 논문에서는 Auto-regressive Decision Trees (ARDTs)의 언어 모델링에서의 이론적 및 실용적 적용을 탐구합니다. ARDT는 시간 시계열 데이터에 대한 처리를 위해 처음 제안되었으나, 언어 생성 작업에서의 가능성을 실험적으로 보여주며 Transformer 모델과 비교할 수 있는 성과를 나타냅니다.

- **Technical Details**: 통계적 언어 모델링 과제에서 ARDT는 다음 토큰 예측을 위한 결정 트리 (decision tree)를 사용하여 기존의 결정 트리보다 더 복잡한 함수들을 계산할 수 있음을 입증합니다. 특히, ARDT는 중간 'chain-of-thought' 계산을 통하여 오토마타, 튜링 기계 및 희소 회로와 같은 기능을 시뮬레이션할 수 있습니다.

- **Performance Highlights**: TinyStories 데이터셋에서 훈련된 ARDT는 30만 개의 매개변수를 가진 결정 트리 앙상블이 100만 개의 매개변수를 가진 Transformer 모델보다 더 우수한 성능을 보였습니다. ARDT는 또한 복잡한 추론 작업을 수행할 수 있는 능력을 보여주며, 특정 다운스트림 작업에서 InstructGPT와 같은 대규모 모델과 경쟁할 수 있는 성능을 나타냅니다.



### Uncovering Differences in Persuasive Language in Russian versus English Wikipedia (https://arxiv.org/abs/2409.19148)
- **What's New**: 이 연구는 영어와 러시아어로 작성된 위키백과(Wikipedia) 기사의 설득 언어(persuasive language) 차이를 조사하여 각 문화의 독특한 시각을 드러냅니다. 저자들은 대규모 언어 모델(LLM) 시스템을 개발하여 다국어 텍스트에서 설득 언어의 사례를 식별하는 방법을 제안합니다. 기존 접근법과는 달리, LLM 자신이 작성한 고급 질문(High-Level Questions, HLQs)을 사용하여 설득 감지를 재구성합니다.

- **Technical Details**: 이 연구는 88,000개의 위키백과 기사를 분석하여 두 단계의 "식별 후 추출(identify-then-extract)" 프롬프트 전략을 적용하고, HLQs를 통해 설득 언어의 존재를 탐지합니다. HLQs에 의해 보여진 결과, F1 점수에서 23.5%의 상대적 개선이 이루어졌으며, 이는 감지 비용을 85.2% 줄이는 데 기여했습니다.

- **Performance Highlights**: 연구 결과, 러시아어와 영어 위키백과 간의 설득 언어 사용에 대한 차이가 확인되었으며, 러시아 위키백과는 우크라이나, 영어 위키백과는 중동 이슈를 강조하는 경향이 있다는 것이 드러났습니다. 정치적 사건이 포함된 주제는 다른 주제보다 더 많은 설득력을 가지고 있음이 나타났습니다.



### Meta-RTL: Reinforcement-Based Meta-Transfer Learning for Low-Resource Commonsense Reasoning (https://arxiv.org/abs/2409.19075)
- **What's New**: 본 연구에서는 강화 학습(double reinforcement learning)을 기반으로 한 다중 출처 메타 전이 학습 프레임워크(Meta-RTL)를 제안합니다. 이는 낮은 자원 환경에서의 상식 추론을 개선하기 위해 출처 작업에 대한 동적 가중치를 추정하는 접근 방식을 포함하고 있습니다.

- **Technical Details**: Meta-RTL은 LSTM(Long Short-Term Memory)을 기반으로 한 정책 네트워크를 사용하여 출처 작업의 기여도를 측정하는 가중치를 동적으로 업데이트합니다. 메타 모델과 출처 특화 임시 메타 모델 간의 일반 손실과 작업별 손실의 차이를 보상으로 사용하여 강화 학습 모듈의 정책 네트워크에 피드백합니다.

- **Performance Highlights**: Meta-RTL은 BERT와 ALBERT를 메타 모델의 백본으로 사용하여 세 가지 상식 추론 벤치마크 데이터셋에서 평가되었습니다. 실험 결과, Meta-RTL은 강력한 기준 모델들을 초과하는 성능을 보였고, 특정 데이터 부족 환경에서도 더 큰 개선을 이끌어냈습니다.



### On the Inductive Bias of Stacking Towards Improving Reasoning (https://arxiv.org/abs/2409.19044)
Comments:
          Accepted at NeurIPS 2024

- **What's New**: 이 논문에서는 새로운 훈련 전략인 MIDAS를 제안합니다. MIDAS는 언어 모델 훈련을 최대 40% 가속할 수 있으며, 특히 추론 능력이 필요한 다운스트림 작업에서의 성능 향상을 제공합니다.

- **Technical Details**: MIDAS는 기존의 점진적 스태깅(gradual stacking) 기법의 변형으로, 작은 네트워크의 중간 블록을 복사하여 큰 네트워크를 초기화하는 방식입니다. 이 방법은 표준 훈련 방식과 유사한 데이터와 FLOPS(부동소수점 연산 수)를 사용하면서도 우수한 성능을 보입니다.

- **Performance Highlights**: MIDAS는 수학 문제와 추론 프리미티브 등에서 뛰어난 성능을 발휘하며, 표준 방법과 성능 비교에서 더 나은 결과를 나타냈습니다. 특히, MIDAS로 사전 훈련된 모델은 표준 훈련 모델보다 추론 프리미티브에서 유의미한 개선을 보였습니다.



### Exploring LLM-Driven Explanations for Quantum Algorithms (https://arxiv.org/abs/2409.19028)
- **What's New**: 이 연구는 LLMs(large language models)가 양자 코드 설명을 지원할 수 있는 가능성을 탐구하며, Gpt3.5, Llama2, Tinyllama의 성능을 비교하고, 프롬프트에 맥락을 추가하는 것이 설명 품질에 미치는 영향을 분석합니다.

- **Technical Details**: 이 연구는 OpenQASM 언어로 작성된 7개의 최첨단 양자 알고리즘에 대한 LLM의 설명 생성 능력을 평가합니다. 특히, 프롬프트에 소량의 맥락(알고리즘 이름 및 사용된 큐비트 수)을 추가했을 때 품질이 개선되는지 조사합니다.

- **Performance Highlights**: Llama2는 맥락이 없는 경우 가장 높은 품질의 설명을 제공하며, Gpt3.5는 기존 설명을 개선하는 데 가장 적합한 LLM으로 나타났습니다. 또한, 여러 라운드에서 생성된 설명은 정성적 및 통사적으로 일관성을 유지하는 것으로 관찰되었습니다.



### Code Generation and Algorithmic Problem Solving Using Llama 3.1 405B (https://arxiv.org/abs/2409.19027)
Comments:
          Under Review

- **What's New**: Llama 3.1 405B 모델은 자연어 처리 및 프로그래밍 자동화 분야에서 코드 생성의 획기적인 발전을 보여줍니다. 이 모델은 자연어 프롬프트를 실행 가능한 코드로 변환하는 능력을 가지고 있으며, 다양한 프로그래밍 언어를 지원합니다.

- **Technical Details**: Llama 3.1 405B는 문맥 인식(contextual awareness), 다중 언어 지원(multi-language support), 디버깅(debugging) 및 최적화(optimization) 기능을 포함한 여러 주요 기능을 제공합니다. 이 모델은 다양한 프로그래밍 언어로 코드를 생성할 수 있으며, 주요 언어로는 Python, JavaScript, Java, C++, HTML/CSS가 있습니다. 퀀텀 컴퓨팅(Quantum Computing)과 생물정보학(Bioinformatics)과 같은 복잡한 과제에서는 한계를 보입니다.

- **Performance Highlights**: Llama 3.1 405B는 간단한 알고리즘 및 데이터 구조 기반 문제를 잘 처리하며, 정확하고 신속한 솔루션을 제공합니다. 그러나 고급 개념인 딥러닝(deep learning) 아키텍처 및 강화학습(reinforcement learning)에서는 어려움을 겪고 있어, 추가적인 개발과 트레이닝이 필요함을 시사합니다.



### Dealing with Controversy: An Emotion and Coping Strategy Corpus Based on Role Playing (https://arxiv.org/abs/2409.19025)
- **What's New**: 심리학적 연구와 컴퓨터 과학 연구 간의 감정 차이를 해소하기 위한 접근법을 제안합니다. 감정을 대처 전략으로 보고, 이에 대한 작업인 'Coping Identification'를 소개합니다.

- **Technical Details**: Coping Strategies(대처 전략)를 텍스트에서 탐지하기 위해 역할 놀이 기반의 Corpus를 구축합니다. 이 Corpus는 감정 발생 사건에 대한 반응을 묘사하는 대화를 포함하고, 대처 전략을 식별하기 위해 모델을 미세 조정합니다.

- **Performance Highlights**: Coping Corpus를 사용한 연구 결과, 대처 전략이 텍스트에서 실현되며, 인간 및 자동 시스템 모두에서 인식하기 어려운 점을 발견했습니다. 새로운 방향의 연구를 통해 감정 메커니즘을 모델링하는 기존 모델의 능력을 더욱 향상시킬 수 있는 가능성을 제시합니다.



### Elephant in the Room: Unveiling the Impact of Reward Model Quality in Alignmen (https://arxiv.org/abs/2409.19024)
- **What's New**: 이번 연구에서는 LLM(대형 언어 모델)의 정렬(alignment) 방법론에서 보상 모델(reward model)의 중요성을 강조하며, 이를 개선하기 위한 노력이 필요함을 제기합니다. 연구진은 HH-RLHF 데이터셋의 품질을 분석하고 청소한 버전인 CHH-RLHF를 제시합니다.

- **Technical Details**: 연구에서는 세 가지 보상 활용 패러다임(직접 보상, 간접 보상, 직접 선호)을 기반으로 다양한 보상 모델의 정확성을 저명한 사례를 통해 벤치마킹하였으며, 보상 모델의 품질이 정렬 성능에 미치는 영향을 체계적으로 연구합니다.

- **Performance Highlights**: 보다 나은 보상 모델이 LLM의 정렬 성능을 더욱 향상시키는 것으로 나타났으며, 부적합한 모델을 사용할 경우 인간 평가와 일치성이 떨어지는 결과가 나타났습니다. 이는 연구자들에게 보상 모델의 품질에 대한 주의를 환기시키는 계기가 됩니다.



### Application of AI-based Models for Online Fraud Detection and Analysis (https://arxiv.org/abs/2409.19022)
Comments:
          Manuscript under peer review. Content may be subject to revision in future versions

- **What's New**: 이번 논문에서는 AI 및 NLP 기술을 활용한 온라인 사기 감지를 위한 체계적인 문헌 검토(Systematic Literature Review, SLR)를 수행하였습니다. 온라인 사기의 정의와 그로 인한 피해, 그리고 AI 기술이 사기 탐지에 응용되는 현재 상태가 중점적으로 분석되었습니다. 특히, 다양한 사기 유형들에 대한 연구 결과를 종합하여 정책입안자, 법 집행 기관 및 기업이 사기 예방 및 방지에 있어 어떻게 대응할 수 있을지를 제시합니다.

- **Technical Details**: 이번 연구는 PRISMA-ScR 프로토콜을 준수하여 2,457개의 학술 기록을 스크리닝한 결과, 350개의 연구가 적격한 기준을 만족하였고 최종적으로 223개의 연구가 포함되었습니다. 이 연구는 NLP 알고리즘 및 모델, 훈련 데이터 출처, 성과 평가 지표 등을 포함하여 다양한 온라인 사기 카테고리에 대한 최첨단 연구 결과를 보고하였습니다.

- **Performance Highlights**: 연구 결과, 현재 온라인 사기에 대한 연구는 여러 가지 사기 활동으로 나뉘어 있으며, 연구자들은 16가지 서로 다른 사기에 초점을 맞추고 있습니다. 특히, 모델 효율성 제고를 위한 데이터의 한계, 훈련 편향 보고 및 선택적 성과 지표 발표와 같은 문제를 식별하였으며, 이는 모델 평가에서 잠재적인 편향을 초래할 수 있다는 점을 강조하였습니다.



### DiaSynth -- Synthetic Dialogue Generation Framework (https://arxiv.org/abs/2409.19020)
Comments:
          13 pages, 1 figure

- **What's New**: DiaSynth는 다양한 도메인에서 고품질의 맥락이 풍부한 대화를 생성할 수 있는 합성 대화 생성(framework) 프레임워크입니다. 기존의 대화 데이터셋의 결점을 보완하기 위해 소주제(subtopics)와 다양한 대화 특징을 시뮬레이션하는 페르소나(personas)를 동적으로 생성합니다.

- **Technical Details**: DiaSynth는 대화 생성에서 LLM(Large Language Model)과 사고의 연쇄(Chain of Thought) 추론을 활용하여 실질적인 인간 상호작용을 모방하는 맥락적으로 풍부한 도메인 특정 대화를 생성합니다. 각 주제에 대해 소주제를 생성하고, 각 소주제에 대해 페르소나를 생성하여 대화를 만듭니다. 이 과정은 세 단계의 파이프라인을 통해 이루어집니다.

- **Performance Highlights**: DiaSynth를 활용한 합성 데이터는 기존 모델보다 16.47% 성능이 향상 되었고, 합성 데이터는 도메인 데이터 분포의 90.48%를 포착하여 강력한 대안임을 증명합니다.



### RAGProbe: An Automated Approach for Evaluating RAG Applications (https://arxiv.org/abs/2409.19019)
Comments:
          11 pages, 5 figures, 9 tables

- **What's New**: 본 논문에서는 Retrieval Augmented Generation (RAG) 파이프라인의 자동 평가를 위한 RAGProbe라는 새로운 기술을 소개합니다. 기존의 수작업 평가 방식으로 인한 한계를 극복하고, 다양한 질문-답변 쌍의 변형을 생성하여 RAG 파이프라인에서 실패를 유도하는 접근법을 개발했습니다.

- **Technical Details**: RAGProbe는 평가 시나리오를 기반으로 하여 질의-응답 쌍을 생성하고, 각각의 시나리오는 질문-답변 쌍의 다양한 변형을 나타냅니다. 이 평가 시나리오에는 문서 샘플링 및 청킹 전략, 시나리오 специф적인 프롬프트 및 프롬프트 전략, 평가 메트릭스가 포함됩니다.

- **Performance Highlights**: RAGProbe는 5개의 오픈 소스 RAG 파이프라인과 3개의 데이터셋을 사용하여 평가되었으며, 기존의 최신 기술보다 평균 51% 더 높은 실패율을 기록했습니다. 또한, 다수의 질문이 포함된 시나리오에서 91%의 실패율을 보여 RAG 파이프라인의 강화를 위한 필요성을 시사합니다.



### Textless NLP -- Zero Resource Challenge with Low Resource Compu (https://arxiv.org/abs/2409.19015)
- **What's New**: 이 연구는 Textless NLP(텍스트 없는 자연어 처리)에서 경량의 인코더-보코더 모델 학습 시 발생하는 훈련 시간 및 GPU 자원의 문제를 해결하기 위한 방법론을 제안합니다. 주요 기여는 학습률 스케줄러(learning rate scheduler)를 활용하여 훈련 단계를 효과적으로 줄이고 성능을 향상시키는 것입니다.

- **Technical Details**: 이 시스템은 Vector-Quantized Contrastive Predictive Coding(VQ-CPC)을 인코더로 사용하고, LSTM 기반의 경량 보코더를 사용합니다. 학습률 스케줄러를 One-Cycle Learning Rate(OCLR)로 설정하여 훈련 시간을 80%까지 줄이고, 오디오 품질 향상을 위해 hop length와 interpolation scale factors를 최적화했습니다. 또한, 인도어(Indian languages) 데이터셋에 대한 실험이 포함되어 있습니다.

- **Performance Highlights**: 제안된 방법은 English, Tamil, Bengali 데이터셋에서 일관되게 우수한 결과를 보여주었으며, 언어 변환 시 재구성된 오디오의 선명도가 눈에 띄게 향상되었습니다. 훈련 시간도 28시간에서 최적화된 단계 수(30k, 40k, 60k)에 따라 각각 6, 8, 12시간으로 줄어들었습니다.



### FLEX: Expert-level False-Less EXecution Metric for Reliable Text-to-SQL Benchmark (https://arxiv.org/abs/2409.19014)
Comments:
          preprint, under review

- **What's New**: 본 논문에서는 FLEX (False-Less EXecution)라는 새로운 평가 방식을 소개하여, SQL 쿼리의 인간 전문가 수준의 평가를 모방하는 대형 언어 모델(LLM)을 활용하여 텍스트-투-SQL 시스템을 평가한다.

- **Technical Details**: FLEX는 기존의 Execution Accuracy (EX) 방식의 한계를 극복하고, 더 정밀한 평가를 제공하며, SQL 쿼리가 원래 질문과 의미적으로 일치하는지를 분석하여 종합적인 쿼리 정확성을 평가한다. 이는 노이즈가 있는 기준 데이터에 대해서도 유용하다.

- **Performance Highlights**: FLEX를 이용한 평가 결과, 기존의 EX 평가보다 인간 전문가의 판단과의 일치도가 크게 향상되었고, Cohen's kappa는 61에서 78.17로 증가했다. 스파이더(Spider)와 BIRD 벤치마크에서 기존 상위 모델의 성능 순위가 대폭 변경되는 결과가 나타났다.



### Improving Academic Skills Assessment with NLP and Ensemble Learning (https://arxiv.org/abs/2409.19013)
Comments:
          5 pages, 2 figures

- **What's New**: 이번 연구는 자연어 처리(NLP)의 발전을 활용하여 기초 학문 기술을 평가하는 데 있어 주요 도전 과제를 다룹니다. 기존의 전통적 평가 방법은 인지적 및 언어적 측면에서 적시의 포괄적인 피드백을 제공하는 데 어려움이 있었습니다. 본 연구는 BERT, RoBERTa, BART, DeBERTa, T5와 같은 최신 NLP 모델을 통합하여 앙상블 학습(enesemble learning) 프레임워크를 통해 정확성을 크게 향상시켰습니다.

- **Technical Details**: 이 모델은 상태를 유지하기 위해 LightGBM과 Ridge 회귀를 사용하여 여러 NLP 모델을 스택(stacking) 기법으로 결합했습니다. 데이터 전처리, 특징 추출, 그리고 pseudo-label 학습을 통해 모델 성능을 최적화했습니다. 또한, PyTorch 모델 클래스를 사용하여 텍스트 입력을 처리하고 효과적으로 분류 작업을 수행했습니다.

- **Performance Highlights**: 이 연구는 ESL(English as a Second Language) 학생들을 위한 언어 평가의 정확성을 크게 향상시켰으며, 전통적인 평가 방법의 한계를 극복하고 교육 기술 연구에서 핵심 학문 역량 향상에 대한 새로운 가능성을 열어주는 강력한 솔루션을 제공했습니다.



### Lost in the Logic: An Evaluation of Large Language Models' Reasoning Capabilities on LSAT Logic Games (https://arxiv.org/abs/2409.19012)
Comments:
          Bachelor's thesis. Dataset available on huggingface: this https URL

- **What's New**: 이 논문은 법학전문대학원 입학시험(LSAT)에서의 대형 언어 모델(LLMs)의 성능 평가를 다룹니다. 특히 논리 게임 부분에서의 복잡한 논리적 추론 작업을 분석하며, 이를 통해 최신 LLM들이 어려운 논리적 과제를 어떻게 처리하는지를 평가합니다.

- **Technical Details**: LSAT 논리 게임을 포함하는 데이터셋을 구성하고 다양한 LLM의 성능을 평가하기 위해 Chain-of-Thought prompting을 사용합니다. 약한 성능에도 불구하고 Reflexion에서 아이디어를 차용한 다른 prompting 프레임워크를 적용한 결과, GPT-4는 70%, GPT-3.5는 46%의 정확도를 보였습니다.

- **Performance Highlights**: GPT-4는 Multi-Shot Chain-of-Thought prompting에서 33%의 정확도로 최상을 기록하였으나, Self-Reflection 기반의 프레임워크를 적용했을 때 70%로 크게 향상되었습니다. 이는 LLM들이 자신의 논리적 오류를 반영하고 수정하는 능력이 뛰어남을 보여줍니다.



### A comprehensive study of on-device NLP applications -- VQA, automated Form filling, Smart Replies for Linguistic Codeswitching (https://arxiv.org/abs/2409.19010)
- **What's New**: 최근 대형 언어 모델의 발전이 이전에는 불가능했던 새로운 경험들을 온디바이스(온디바이스; on-device) 애플리케이션에서 제공할 수 있는 기회를 열어주었습니다. 본 연구에서는 크게 두 가지 범주로 나누어진 세 가지 새로운 경험을 제안합니다.

- **Technical Details**: 첫 번째 범주는 화면 이해(Screen Understanding)와 관련된 경험으로, 사용자 화면에 나타난 정보에 대한 이해(Visual Question Answering, 자동 양식 채우기)가 포함됩니다. 두 번째 범주는 코드 스위칭(Code-Switching) 기능을 지원하는 스마트_reply(smart replies) 시스템의 확장입니다. 본 연구에서 제안된 첫 번째 작업으로는 화면 기반 이해를 위한 Visual Question Answering과 이전 화면의 맥락을 활용한 자동 양식 채우기 과제가 포함됩니다. 모델은 LayoutLM과 MarkupLM 두 가지 계열로 구성되며, LayoutLMv3는 이미지와 텍스트 토큰을 정렬하여 다중 사전 훈련 과정을 수행합니다.

- **Performance Highlights**: 데이터 수집 및 질문-답변 쌍 생성 방법을 통해 4,500개 이상의 iOS 앱에서 100,000개 이상의 스크린샷을 수집했습니다. 이를 기반으로 모델을 훈련하여 기존 양식 정보 및 시각적 맥락에 대한 정확도를 개선했습니다. 처음으로 제안된 이 연구는 화면 기반으로 질문을 생성하는 작업과 다국어 사용자 정보를 기반으로 한 개인화된 스마트 응답 생성을 포함하여 새로운 응용 프로그램을 탐구합니다.



### Rephrase and Contrast: Fine-Tuning Language Models for Enhanced Understanding of Communication and Computer Networks (https://arxiv.org/abs/2409.19007)
Comments:
          This paper has been submitted to IEEE WCNC 2025

- **What's New**: 이 논문은 Rephrase and Contrast (RaC) 프레임워크를 제안하여 대형 언어 모델(LLM)이 네트워킹 문제를 보다 효과적으로 이해하고 해결하도록 돕는 새로운 방법을 소개하고 있습니다.

- **Technical Details**: RaC 프레임워크는 1) 질문의 재구성 및 2) 정답과 오답의 대조 분석을 포함하여 LLM의 이해력과 비판적 사고 능력을 향상시킵니다. 논문에서는 RaC를 통해 63.73%의 정확도 개선을 달성했다고 보고합니다.

- **Performance Highlights**: RaC 프레임워크는 LLM의 네트워킹 작업에 대한 적합성을 높이며, GPT 보조 데이터 마이닝 및 ChoiceBoost 데이터 증강 기법을 사용하여 훈련 데이터셋을 효과적으로 생성하고 확장합니다.



### Towards Automated Patent Workflows: AI-Orchestrated Multi-Agent Framework for Intellectual Property Management and Analysis (https://arxiv.org/abs/2409.19006)
Comments:
          This is a preprint and current version under peer review

- **What's New**: 본 논문은 PatExpert라는 자율 다중 대화형 프레임워크를 제안하며, 이는 다양하고 복잡한 특허 관련 작업을 최적화하고 자동화하는 데 도움을 줍니다. 프레임워크는 메타 에이전트와 특정 작업을 수행하는 전문 에이전트로 구성되어 있습니다.

- **Technical Details**: PatExpert 프레임워크는 메타 에이전트가 고유한 태스크를 수행하는 여러 전문 에이전트를 조정하는 방식으로 작동합니다. 이 시스템은 Graph Retrieval-Augmented Generation (GRAG)과 같은 고급 기법을 사용하여 지식 그래프를 활용해 정확성과 관련성을 향상시킵니다.

- **Performance Highlights**: PatExpert는 다중 특허 분석, 특허 분류, 청구 생성 및 오류 처리를 포함하여 예상치 못한 복잡한 작업을 자동화함으로써 특허 처리 작업의 효율성과 정확성을 크게 향상시켰습니다.



### What is a Digital Twin Anyway? Deriving the Definition for the Built Environment from over 15,000 Scientific Publications (https://arxiv.org/abs/2409.19005)
- **What's New**: 디지털 트윈(Digital Twin) 개념이 건설 환경에서 주목받고 있으나, 정의와 용어의 일관성이 부족하여 연구자와 실무자 간 혼란을 초래하고 있다는 점이 강조됩니다. 본 연구는 15,000개의 논문에서 디지털 트윈의 정의를 체계적으로 추출하고 분석하였으며, 52명의 전문가 설문조사와 비교 분석하여 실용적인 관점에서 구성 요소에 대한 합의점을 도출하였습니다.

- **Technical Details**: 자연어 처리(NLP) 기법을 활용하여 문헌에서 디지털 트윈의 주요 구성 요소를 추출하고, 텍스트 빈도 분석(Text Frequency Analysis)과 N-그램 분석(N-gram analysis)을 통해 문헌에서 나타나는 구성 요소를 식별하였습니다. 이후 카이제곱 검정(Chi-square test)을 수행하여 다양한 분야에서 각 구성 요소의 중요성을 평가했습니다.

- **Performance Highlights**: 연구 결과, 디지털 트윈의 정의는 연구 분야에 따라 다르게 나타나지만, 많은 유사점이 발견되었습니다. 특히, 디지털 트윈이 고성능 실시간 애플리케이션(HPRT) 또는 장기 결정 지원(LTDS) 용도로 사용되는지에 따라 중요한 차별성이 나타났습니다. 각 분야의 대표 정의를 종합하여 맥락에 맞춤형 데이터를 기반으로 한 새로운 정의를 제시했습니다.



### Pay Attention to What Matters (https://arxiv.org/abs/2409.19001)
- **What's New**: 이번 논문에서는 사용자 지침에 대한 LLM(대형 언어 모델)의 출력을 정렬하는 능력을 향상시키기 위한 새로운 방법인 GUIDE(Instruction-Driven Enhancements에 의한 Guided Understanding)를 소개합니다. GUIDE는 중요한 지침에 대한 attention score를 증가시켜 LLM이 사용자 지침을 보다 정확하게 따르도록 돕습니다.

- **Technical Details**: GUIDE는 입력 내에서 특정 토큰을 강조하기 위한 시스템적 접근법으로, 사용자는 중요한 텍스트를 <!-> <-!>와 같은 태그로 묶음으로써 attention을 유도할 수 있습니다. 이 방법은 LLM의 attention score에 bias를 추가하여 특정 토큰에 주의를 환기시키며, 새로운 측정 지표인 Influence를 제시하여 지침-토큰 간의 중요성을 정량화합니다.

- **Performance Highlights**: GUIDE는 지침을 따르는 정확도를 29.4%에서 60.4%로 개선하여 자연어 프롬프트 방식 및 100만 토큰까지의 Supervised Fine-Tuning(SFT)보다 우수한 성능을 보여줍니다. GUIDE는 추가적인 훈련 없이도 효과적으로 사용 가능하며, 사용자의 요구에 맞춤화된 지침 이행을 가능하게 합니다.



### Enhancing TinyBERT for Financial Sentiment Analysis Using GPT-Augmented FinBERT Distillation (https://arxiv.org/abs/2409.18999)
Comments:
          Submitted in partial fulfillment of the requirements for Masters in Machine Learning and Artificial Intelligence at Liverpool John Moores University, 97 pages, 1 figure, 14 tables

- **What's New**: 이번 연구는 금융 감정 분석 분야에서 LLM(large language models)인 GPT-4 Omni의 생성 능력을 활용하여 도메인 특화된 합성 훈련 데이터를 만드는 새로운 접근 방식을 제안합니다. 이를 통해 데이터 부족 문제를 해결하고, 더 작은 모델의 성능을 향상시켜 기존 대형 모델과 경쟁력을 갖추게 합니다.

- **Technical Details**: 연구는 FinBERT라는 BERT 기반 모델을 향상시키고, 구조화된 2단계 지식 증류 전략을 통해 소형 트랜스포머 모델인 TinyFinBERT를 개발합니다. GPT-4 Omni를 통해 기존 데이터를 변형하고 새로운 훈련 예제를 생성하여 FinBERT의 정확도를 크게 개선하고, 이를 교사 모델로 사용하여 TinyFinBERT에 지식을 증류합니다.

- **Performance Highlights**: TinyFinBERT는 PhraseBank 데이터셋과 FiQA 2018 Task1 데이터셋으로 훈련 및 평가되어, FinBERT와 유사한 성능을 달성하면서도 훨씬 작고 효율적인 모델임을 입증합니다. 이 연구는 LLM이 금융 감정 분석의 발전에 기여할 수 있는 방법을 보여줍니다.



### Controlled LLM-based Reasoning for Clinical Trial Retrieva (https://arxiv.org/abs/2409.18998)
- **What's New**: 본 논문에서는 임상 시험(CT) 데이터의 검색 및 재배치를 위한 새로운 체계적인 추론 방법을 제안합니다. 이 방법은 LLMs(Large Language Models)의 기능을 확장하여 의료 적격 기준에 대한 체계적인 추론을 가능하게 합니다.

- **Technical Details**: 제안된 방법은 LLM을 사용하여 환자 메모와 임상 시험 기록을 속성 세트로 변환하고, 이를 통해 CT의 정확한 매핑과 해석이 이루어질 수 있습니다. 여기서 'Set-guided reasoning' 방법이 사용되어 hierarchical relationship을 통해 도메인 특화 지식을 활용합니다.

- **Performance Highlights**: TREC 2022 Clinical Trials에서 평가된 결과, NDCG@10이 0.693, Precision@10이 0.73로 기존의 최첨단 성능을 초월했습니다.



### PropaInsight: Toward Deeper Understanding of Propaganda in Terms of Techniques, Appeals, and Inten (https://arxiv.org/abs/2409.18997)
Comments:
          8 pages

- **What's New**: 본 연구는 PropaInsight라는 새로운 개념적 프레임워크를 도입하여 선전(propaganda)의 기법, 자극 호소(arousal appeals), 그리고 기본 의도(underlying intent)를 체계적으로 분석합니다. 더불어, PropaGaze라는 새로운 데이터셋을 통해 전문가 주석이 부족한 상황에서도 선전을 효과적으로 분석할 수 있는 방법을 제시합니다.

- **Technical Details**: PropaInsight는 선전의 세 가지 주요 요소를 식별하며, 각 요소는 기법 식별(Propaganda Technique Identification), 호소 분석(Appeal Analysis), 의도 분석(Intent Analysis)으로 나뉩니다. PropaGaze 데이터셋은 세 가지 하위 데이터셋으로 구성되어 있으며, 각 데이터셋은 인간 주석자에 의해 주석이 달린 뉴스 기사와 고품질 합성 데이터로 이루어져 있습니다. 이는 LLMs(대형 언어 모델)의 강력한 이해 능력을 활용하여 작성되었습니다.

- **Performance Highlights**: PropaGaze를 사용한 실험에서는 LLMs가 선전 분석에서 어려움을 겪는 것으로 나타났지만, PropaGaze로 학습한 경우 성능이 크게 개선되었습니다. 예를 들어, Llama-7B-Chat은 기법 식별에서 1-shot GPT-4-Turbo에 비해 203.4% 높은 IoU(text span Intersection over Union)를 달성하고, 호소 분석에서 66.2% 높은 BertScore를 기록했습니다.



### From Linguistic Giants to Sensory Maestros: A Survey on Cross-Modal Reasoning with Large Language Models (https://arxiv.org/abs/2409.18996)
- **What's New**: 본 논문은 Cross-Modal Reasoning (CMR)과 관련된 최근의 대규모 언어 모델(LLMs)의 발달을 다루고 있으며, LLMs가 CMR에 미치는 역할을 세분화하여 탐구하는 첫 번째 설문조사로서의 중요성을 강조합니다.

- **Technical Details**: 논문은 LLMs의 네 가지 주요 역할인 Multimodal Fusion Engine, Textual Processor, Cognitive Controller, Knowledge Enhancer를 소개합니다. 또한 Prompt Tuning, Instruction Tuning, Multimodal Pre-training과 같은 방법론을 설명하며 이들이 CMR에 활용되는 방식을 상세히 다룹니다.

- **Performance Highlights**: LLMs는 텍스트, 이미지 및 소리 등 다양한 모드 간의 새로운 정보를 이해하고 추론하는 능력을 통합하여 그들의 성능을 향상시킵니다. CMR의 적용 예시로는 비주얼 질문 응답, 비전-언어 탐색, 이미지 및 비디오 캡셔닝 등이 포함됩니다.



### Systematic Characterization of the Effectiveness of Alignment in Large Language Models for Categorical Decisions (https://arxiv.org/abs/2409.18995)
Comments:
          19 pages (without Appendix) Appendix 7 pages. 7 Figures

- **What's New**: 이 논문은 대규모 언어 모델(LLM)이 의료 분야와 같은 고위험 영역에서 인간의 선호 및 가치와 얼마나 잘 일치하는지 평가하는 체계적인 방법론을 제시합니다. 특히, 의료 분류(triage)라는 특정 사용 사례를 통해 LLM의 결정이 전문가의 선호와 얼마나 일치하는지를 평가합니다.

- **Technical Details**: 중요한 방법론적 요소 중 하나는 Alignment Compliance Index (ACI)라는 새로운 측정 지표로, 이는 LLM이 특정 선호 함수에 얼마나 효과적으로 정렬되는지를 정량화합니다. 이 연구에서는 GPT4o, Claude 3.5 Sonnet, 및 Gemini Advanced와 같은 세 가지 첨단 LLM을 사용하여 의사 결정의 일관성을 평가하였습니다.

- **Performance Highlights**: 모델 간의 정렬 효과는 상당한 변동성을 보였으며, ACI에 의해 잘 수행된 모델이 정렬 후에는 성능이 저하되는 경향도 발견되었습니다. 또한, 목표 선호 함수의 미세한 변화가 모델의 순위에 큰 영향을 미친다는 점도 주목할 만합니다.



### A Review of Mechanistic Models of Event Comprehension (https://arxiv.org/abs/2409.18992)
- **What's New**: 이 리뷰는 사건 이해(Event comprehension)의 이론적 가정과 계산 모델을 검토하며, 담화 이해(disourse comprehension) 이론에서 현대의 사건 인지(event cognition) 프레임워크로의 진화를 추적합니다.

- **Technical Details**: 주요 담화 이해 모델로는 Construction-Integration, Event Indexing, Causal Network, 그리고 Resonance 모델 등이 있으며, 이들은 이해 과정에서의 인지적 프로세스를 이해하는 데 기여합니다. 현대의 사건 이해 이론으로는 Event Segmentation Theory, Event Horizon Model, Hierarchical Generative Framework 등이 있으며, 이들은 사건 이해에서의 예측(prediction), 인과 관계(causality), 다계층 표현(multilevel representations)의 중요성을 강조합니다. 분석된 다섯 가지 계산 모델로는 REPRISE, Structured Event Memory, Lu 모델, Gumbsch 모델, Elman 및 McRae 모델이 있습니다.

- **Performance Highlights**: 계층적 처리(hierarchical processing), 예측 메커니즘(prediction mechanisms), 그리고 표현 학습(representation learning)에 대한 접근 방법에 초점을 맞추어, 사건 이해에 있어 예측의 중요성과 사건 역학(Event dynamics)의 학습을 위한 다양한 전략을 강조합니다. 향후 연구의 중요한 영역으로는 구조적 표현(structured representations)에 대한 학습을 위한 더 정교한 접근법 필요성, 일화 기억 메커니즘(episodic memory mechanisms)의 통합, 사건 모델에 대한 적응형 업데이트 알고리즘 개발이 포함됩니다.



### Surveying the MLLM Landscape: A Meta-Review of Current Surveys (https://arxiv.org/abs/2409.18991)
Comments:
          The article consists of 22 pages, including 2 figures and 108 references. The paper provides a meta-review of surveys on Multimodal Large Language Models (MLLMs), categorizing findings into key areas such as evaluation, applications, security, and future directions

- **What's New**: 본 논문에서는 Multimodal Large Language Models (MLLMs)의 최근 발전 및 그 평가 방법에 대한 포괄적인 리뷰를 제공합니다. 수많은 기존 연구들을 종합하여 11개의 핵심 분야로 MLLM의 현재 상태를 정리하고 있습니다.

- **Technical Details**: MLLMs는 텍스트, 이미지, 오디오, 비디오와 같은 다양한 모달리티를 통해 정보를 처리하고 생성하는 능력을 갖추고 있습니다. 특히, MLLMs는 여러 가지 데이터 유형을 통합함으로써 전통적인 unimodal 모델의 한계를 극복하고 있습니다. 이 논문은 평가 방법론 및 벤치마크 테스트들을 심도 깊게 분석하여 MLLM의 다양한 적용 가능성과 성능 지표를 강조합니다.

- **Performance Highlights**: MLLM의 성능이 매우 중요해짐에 따라, 평가 방법론의 정확성과 포괄성이 점점 더 중요해지고 있습니다. 이 연구는 MLLM의 평가를 위한 기초 개념, 응용 프로그램, 윤리적 문제 등을 포함하여 기존 문헌을 종합적으로 분석하여 향후 연구 방향을 제안합니다.



### SC-Phi2: A Fine-tuned Small Language Model for StarCraft II Macromanagement Tasks (https://arxiv.org/abs/2409.18989)
- **What's New**: SC-Phi2 모델은 StarCraft II의 macromanagement 작업을 위한 새로운 소형 언어 모델입니다. 이를 통해 소형 언어 모델이 이전의 대형 언어 모델보다 적은 자원으로 효과적인 성능을 발휘할 수 있음을 보여줍니다.

- **Technical Details**: SC-Phi2는 Microsoft의 Phi2 모델을 기반으로 하며, 새로운 SC2 텍스트 데이터셋을 사용하여 self-supervised learning에 의해 fine-tuning되며, Vision Transformer(ViT)와 결합하여 gameplay 동적 프롬프트를 생성합니다. 이 모델은 Low-rank Adaptation(LoRA) 기법과 양자화(Quantization)를 통해 단일 GPU에서 훈련됩니다.

- **Performance Highlights**: 모델은 build order 및 global state 예측과 같은 micromanagement 작업에서 훌륭한 성능을 보이며, 전체적으로 2.8 billion parameters만을 가지면서도 타 모델들과 비교해도 경쟁력 있는 예측 능력을 보여줍니다.



### A Unified Framework to Classify Business Activities into International Standard Industrial Classification through Large Language Models for Circular Economy (https://arxiv.org/abs/2409.18988)
Comments:
          6 pages, 2 figures, accepted in 2024 IEEE International Conference on Industrial Engineering and Engineering Management (IEEM 2024)

- **What's New**: 이 논문은 순환 경제(practices) 관행을 촉진하는 추천 시스템을 개발하기 위해 효과적인 정보 수집과 지식 코딩의 중요성을 강조합니다.

- **Technical Details**: 이 연구에서는 대규모 언어 모델(LLMs)을 활용하여 경제 활동을 국제 표준 산업 분류(ISIC) 체계로 분류합니다. 이를 통해 다양한 지역의 모든 경제 활동 설명을 통합된 ISIC 표준으로 분류할 수 있게 합니다.

- **Performance Highlights**: GPT-2 모델을 미세 조정하여 182개의 레이블 테스트 데이터셋에서 95%의 정확도를 달성했습니다. 이 연구는 지속 가능한 순환 경제 관행을 촉진하는 데 기여합니다.



### Efficient and Personalized Mobile Health Event Prediction via Small Language Models (https://arxiv.org/abs/2409.18987)
Comments:
          6 pages, 3 figures

- **What's New**: 이 논문에서는 헬스케어 모니터링에 대한 소형 언어 모델(SLMs)의 능력을 처음으로 조사하였으며, 환경을 보호하면서 개인화된 건강 상태 분석을 위한 가능성을 제시합니다.

- **Technical Details**: 이 연구에서는 TinyLlama(1.1B 파라미터)의 성능을 분석하여 4.31GB의 메모리 사용량과 0.48초의 지연시간(latency)을 나타냈으며, 다른 4개의 최신 소형 언어 모델(SOTA SLMs)보다 우수한 성능을 보였습니다.

- **Performance Highlights**: SLMs는 헬스케어 모니터링의 최신 솔루션으로써 우수한 성능을 보여주며, 특히 CPU 사용량, 지연시간, 메모리 사용량에서 15.5배의 개선을 보였습니다. 또한, 기존 LLMs와 비교하여 실시간 헬스케어 응용 프로그램에 더 적합한 것으로 판단됩니다.



### Lab-AI -- Retrieval-Augmented Language Model for Personalized Lab Test Interpretation in Clinical Medicin (https://arxiv.org/abs/2409.18986)
- **What's New**: Lab-AI는 환자 맞춤형 정상 범위를 제공하는 상호작용 시스템으로, Retrieval-Augmented Generation (RAG) 기술을 활용하여 신뢰할 수 있는 건강 정보 소스로부터 정보를 검색합니다.

- **Technical Details**: Lab-AI는 두 가지 모듈, 즉 factor retrieval 및 normal range retrieval로 구성되어 있으며, 68개의 실험실 테스트에서 30개는 조건적 요소가 포함되고 38개는 포함되지 않습니다. 테스트의 정상 범위는 환자-specific information에 따라 달라집니다. GPT-4-turbo 모델은 factor retrieval에서 0.95의 F1 score, normal range retrieval에서 0.993의 정확도를 보였습니다.

- **Performance Highlights**: RAG를 사용하는 GPT-4-turbo는 비-RAG 시스템보다 29.1% 더 높은 factor retrieval 성능을 나타내었고, normal range retrieval에서 질문 수준에서 60.9%, 실험실 수준에서 52.9% 향상을 보였습니다. 이러한 결과는 Lab-AI가 환자가 실험실 결과를 이해하는 데 도움을 줄 수 있는 잠재력을 강조합니다.



### Harnessing Large Language Models: Fine-tuned BERT for Detecting Charismatic Leadership Tactics in Natural Languag (https://arxiv.org/abs/2409.18984)
Comments:
          The 2024 IEEE 3rd Conference on Information Technology and Data Science, CITDS 2024

- **What's New**: 이 연구는 Charismatic Leadership Tactics (CLTs)를 자연어에서 식별하기 위해 미세 조정된 Bidirectional Encoder Representations from Transformers (BERT) 모델을 사용하는 방법을 탐구합니다. CLTs를 위한 대규모 코퍼스를 기반으로 하여, 본 연구는 자연어에서 이 전술의 존재를 정확히 식별할 수 있는 기계 학습 모델을 훈련하는 방법론을 제시합니다.

- **Technical Details**: 본 연구는 BERT 모델을 특정 CLTs를 대상으로 미세 조정하여 해당 전술의 존재를 텍스트에서 식별합니다. 연구는 BERT로 훈련된 모델이 CLTs를 효과적으로 탐지하는지 평가하고, 다루는 특정 데이터셋에 대해 98.96%의 높은 정확도를 보였습니다.

- **Performance Highlights**: 본 연구의 모델은 자연어 처리(NLP) 기술을 통해 Charismatic Leadership의 언어적 특성을 분석할 수 있는 도구를 개발함으로써 심리학 및 경영 분야의 미래 연구에 기여할 수 있는 잠재력을 가집니다.



### IW-Bench: Evaluating Large Multimodal Models for Converting Image-to-Web (https://arxiv.org/abs/2409.18980)
- **What's New**: 최근 대규모 멀티모달 모델의 발전이 이미지 이해 능력에서 크게 향상되었습니다. 그러나 이러한 대규모 모델의 Image-to-Web 전환 능력을 평가하기 위한 강력한 기준이 부족합니다. 이를 해결하기 위해 IW-Bench라는 새로운 기준을 제정하고, Element Accuracy 및 Layout Accuracy와 같은 새로운 평가지표를 개발했습니다.

- **Technical Details**: IW-Bench는 1200개의 이미지와 웹 코드 쌍으로 구성되어 있으며, 난이도는 간단, 중간, 복잡으로 구분됩니다. Element Accuracy는 DOM (Document Object Model) 트리를 파싱하여 웹 요소의 완전성을 평가하며, Layout Accuracy는 DOM 트리를 공통 부분 수열로 변환하여 요소의 상대적 위치 관계를 분석합니다. 또한, 다섯 단계의 Chain-of-Thought Prompting을 통해 성능을 향상시키도록 설계되었습니다.

- **Performance Highlights**: 대규모 멀티모달 모델에 대한 광범위한 평가를 실시하였으며, 결과는 이들 모델의 강점과 개선이 필요한 영역에 대한 통찰을 제공합니다. 특히, 새로운 다섯 단계의 Chain-of-Thought 방법론이 성능 향상에 기여한 것으로 나타났습니다.



### Pronoun Logic (https://arxiv.org/abs/2409.18978)
Comments:
          Accepted to Queer in AI Workshop @ NAACL 2024. this https URL

- **What's New**: 이 논문은 성전환(transgender) 및 비이진(nonbinary) 커뮤니티에서 개인적인 대명사(pronoun)를 표현하기 위한 형식 논리(formal logic)의 사용을 제안합니다. 개인의 성별을 올바르게 언급하기 위해 대명사를 공개적으로 공유하는 관행이 보편화되고 있습니다. 특히, 다양한 대명사 설명자(pronoun descriptor)를 통해 자신을 표현하고자 하는 복잡한 욕구를 파악하고자 합니다.

- **Technical Details**: 이 논문에서는 대명사의 형식화를 위한 세 가지 논리적 기반(linear logic, temporal logic, free logic with definite descriptions)을 탐구합니다. 각 기반은 성별 의사소통의 다양한 측면을 형식화하도록 돕고, 특히 각 TGNB 개인이 중요하게 여기는 우선순위에 따라 다른 요구를 반영합니다. Linear Logic은 대명사 선택의 명확성을 제공하며, Temporal Logic은 시간에 따른 언어 패턴 요구 사항을 설정합니다. 또한, Description operations는 기본 대명사 레이블을 넘어 성별을 표현하는 방법을 형식화할 수 있게 합니다.

- **Performance Highlights**: 이 연구는 TGNB 커뮤니티에 대한 이해를 증진시키고, 특히 논리학에 대한 소속감을 느낄 수 있도록 돕는 것을 목표로 하고 있습니다. 또한, 대명사의 의미를 형식화함으로써 자연어 처리应用에 활용할 수 있는 가능성을 제시하며, 잘못된 성별 언급(misgendering)을 감지하고 수정할 수 있는 도구를 개발할 수 있는 기회를 제공합니다.



### MM1.5: Methods, Analysis & Insights from Multimodal LLM Fine-tuning (https://arxiv.org/abs/2409.20566)
- **What's New**: MM1.5라는 새로운 멀티모달 대형 언어 모델(Multimodal Large Language Models) 패밀리를 소개합니다. 이 모델은 텍스트가 풍부한 이미지 이해, 비주얼 레퍼링(Visual Referring) 및 그라운딩(Grounding), 여러 이미지에 대한 추론 기능을 향상시키기 위해 설계되었습니다.

- **Technical Details**: MM1.5는 데이터 중심(Data-Centric) 접근 방식을 채택하여 모델 훈련 과정 전반에 걸쳐 다양한 데이터 혼합의 영향을 체계적으로 탐구합니다. 이는 고품질 OCR 데이터와 합성 캡션(Synthetic Captions)을 포함한 지속적인 사전 훈련(Continual Pre-Training) 및 최적화된 비주얼 인스트럭션 튜닝 데이터 혼합을 통한 감독적 미세 조정(Supervised Fine-Tuning)을 포함합니다. 모델은 1B부터 30B까지의 매개변수(Parameter)를 가지며, 밀집(Dense) 및 전문가 혼합(Mixture-of-Experts, MoE) 변형이 포함됩니다.

- **Performance Highlights**: 신중한 데이터 큐레이션(Data Curation) 및 훈련 전략이 작은 규모(1B 및 3B)의 모델에서도 강력한 성능을 내는 것을 입증했습니다. 또한 비디오 이해를 위한 MM1.5-Video 및 모바일 UI 이해를 위한 MM1.5-UI라는 두 가지 특수 변형을 도입합니다.



### LLM Hallucinations in Practical Code Generation: Phenomena, Mechanism, and Mitigation (https://arxiv.org/abs/2409.20550)
Comments:
          11 pages, 13 figures

- **What's New**: 이번 연구는 대형 언어 모델(LLM) 기반의 코드 생성 과정에서의 환각(hallucinations) 문제를 체계적으로 조사합니다. 특히 실용적이고 복잡한 개발 시나리오에서 LLM 환각의 현상, 메커니즘 및 완화 방법에 대한 경험적 연구를 수행했습니다.

- **Technical Details**: LLMs의 환각 현상은 크게 세 가지 주요 범주로 나뉩니다: Task Requirement Conflicts, Factual Knowledge Conflicts, Project Context Conflicts. 이 연구에서는 6개의 주요 LLM(예: ChatGPT, CodeGen)에서 발생하는 환각의 유형과 분포를 분석하고, 환각의 원인으로는 훈련 데이터 품질, 의도 이해 능력, 지식 습득 능력 및 레포지토리 수준의 맥락 인식을 확인했습니다.

- **Performance Highlights**: RAG 기반의 완화 접근법을 제안하며, 이 접근법은 다양한 LLM에서 일관되게 성능을 향상시키는 효과를 보였습니다. 코드 생성 시나리오에 따라 만들어진 검색 라이브러리를 활용하여 각 생성 작업에 유용한 코드 스니펫을 검색합니다.



### The Perfect Blend: Redefining RLHF with Mixture of Judges (https://arxiv.org/abs/2409.20370)
Comments:
          submitted to conference

- **What's New**: 이 연구에서는 다중 작업 학습(MTL)에서의 강화를 통해 인공지능 모델의 후처리를 개선하기 위해 제약 생성 정책 최적화(Constrained Generative Policy Optimization, CGPO)라는 혁신적인 패러다임을 소개합니다. CGPO는 비용 효율적인 제약 정책 최적화와 샤프화(stratification)를 통해 RLHF의 최적 혼합을 식별할 수 있습니다.

- **Technical Details**: CGPO의 핵심은 다양한 작업에 대한 맞춤형 최적화 전략을 적용하고, 룰 기반(judge) 및 LLM 기반(judge) 두 가지 종류의 심사를 통해 보상 해킹(reward hacking)을 탐지 및 완화하는 것입니다. 이를 위해 새로운 제약 RLHF 최적화기(Calibrated-Regularized Policy Gradient, CRPG; Constrained Online Direct Preference Optimization, CODPO; Calibrated-Regularized Reward Ranking Finetuning, CRRAFT)가 개발되었습니다.

- **Performance Highlights**: CGPO는 일반 대화, STEM 질문, 지침 준수 및 코딩을 포함한 다양한 작업에서 PPO 및 DPO와 같은 기존의 RLHF 알고리즘을 초월하는 성능을 보여주었습니다. AlpacaEval-2에서 7.4%, Arena-Hard에서 12.5% 개선된 성과를 달성했으며, 전반적으로 모든 벤치마크와 작업에서 동향이 일관성을 보였습니다.



### Boosting Hybrid Autoregressive Transducer-based ASR with Internal Acoustic Model Training and Dual Blank Thresholding (https://arxiv.org/abs/2409.20313)
Comments:
          Accepted to Interspeech 2024

- **What's New**: 이번 연구에서는 HAT(하이브리드 자기 회귀 변환기)의 성능을 향상시키기 위한 새로운 internal acoustic model (IAM) 훈련 전략을 제안합니다.

- **Technical Details**: IAM은 encoder와 joint 네트워크로 구성되며, HAT와 완벽하게 공유되고 공동 훈련됩니다. 이 공동 훈련은 HAT의 훈련 효율성을 높이고 IAM과 HAT가 비어있는(blank) 기호를 동기화하여 출력하도록 유도합니다.

- **Performance Highlights**: IAM과 공동 훈련된 HAT는 상대적인 오류를 줄이는 데 있어 통계적으로 유의미한 결과를 보였으며, 비어있는 기호 계산을 건너뛰어 디코딩 속도가 42-75% 향상했습니다.



### OM4OV: Leveraging Ontology Matching for Ontology Versioning (https://arxiv.org/abs/2409.20302)
Comments:
          7 pages, 7 figures, 1 table

- **What's New**: 본 논문에서는 기존의 ontology matching (OM) 기술을 활용하여 ontology version control (OV)을 위한 새로운 접근 방식을 제안합니다. OM과 OV 간의 상보적 관계를 탐구하고, 두 작업을 동일한 파이프라인으로 처리할 수 있는 방법을 제시합니다.

- **Technical Details**: 논문에서 제안하는 OM4OV 파이프라인은 신규 작업 정의, 성과 측정 및 검증용 데이터셋 구축을 통해 OV 작업에 OM을 효과적으로 활용합니다. 특히, cross-reference 메커니즘을 도입하여 OV 후보 선택을 최적화하고 OM의 전반적인 성능을 향상시킵니다.

- **Performance Highlights**: OAEI 데이터셋을 사용하여 OM4OV 파이프라인과 cross-reference 메커니즘의 성능을 실험적으로 검증하였습니다. 이 연구는 기계 생성된 버전 정보를 활용한 경량화 및 완전 자동화된 OV 접근 방식을 제시하여, 기존 OM 시스템과 기술을 OV 작업으로 이전할 수 있는 새로운 가능성을 열었습니다.



### Alignment-Free Training for Transducer-based Multi-Talker ASR (https://arxiv.org/abs/2409.20301)
Comments:
          Submitted to ICASSP 2025

- **What's New**: 이번 논문에서는 기존 RNN Transducer(RNNT)를 확장하여 다중 화자 음성을 인식하는 Multi-talker RNNT (MT-RNNT) 모델을 제안합니다. 특히, 새로운 Alignment-Free Training (AFT) 방식을 도입하여 복잡한 정렬 과정을 제거하고, 하나의 인코더로 모든 화자의 음성을 인식할 수 있도록 하였습니다.

- **Technical Details**: MT-RNNT-AFT는 표준 RNNT 아키텍처를 채택하며, 각 화자의 출현 순서를 반영한 프롬프트 토큰을 추가하여 대상 레이블을 생성합니다. 이러한 방법으로 정확한 정렬 없이 학습이 가능하고, 한 번의 인코더 처리로 모든 화자의 음성을 인식할 수 있습니다. 손실은 각 화자에 대해 개별적으로 계산되어 합산됩니다.

- **Performance Highlights**: MT-RNNT-AFT는 옵타이멀 대안과 비교해도 거의 동등한 성능을 달성하며, KD 및 언어 모델 통합을 통해 인식 성능을 추가로 향상시켰습니다. 시행된 실험 결과, 비디오 및 스트리밍 모드 모두에서 최첨단 시스템과 비슷한 인식 성능을 보여줍니다.



### PersonalLLM: Tailoring LLMs to Individual Preferences (https://arxiv.org/abs/2409.20296)
Comments:
          28 pages, 6 figures

- **What's New**: 이 논문에서는 LLM(대형 언어 모델)을 개인 사용자의 특성과 선호에 맞게 조정하기 위한 새로운 공개 벤치마크인 PersonalLLM을 제시합니다. 기존의 정렬(Alignment) 벤치마크가 통일된 선호를 전제로 하는 반면, PersonalLLM은 다양한 고품질 응답과 함께 열린 질문을 선별하여 사용자들의 이질적인 숨은 선호를 반영합니다.

- **Technical Details**: PersonalLLM은 10,402개의 열린 질문과 이에 대한 8개의 고품질 응답으로 구성된 데이터셋을 통해 사용자의 선호 모델을 샘플링합니다. 이 모델은 다양성과 역사적 사용자 기반을 시뮬레이션 하여 사용자 맞춤화를 위한 새로운 기법을 가능하게 합니다. 또한, 맥락 내 학습(In-Context Learning) 및 메타 학습(Meta-Learning) 기법으로 연속 데이터 부족 문제를 해결하기 위한 기초를 마련합니다.

- **Performance Highlights**: PersonalLLM은 사용자 개별적인 니즈에 맞춘 최적의 응답을 생성할 수 있는 능력을 보여주며, 대화형 AI의 효율성과 유용성을 높이는 가능성을 가지고 있습니다. 특히, 사용자 맞춤형 교육 경험이나 고객 지원 챗봇의 정확한 대응 등에 활용될 수 있는 잠재력을 가지며, 기존의 개인화 시스템보다 인상적으로 다양한 선호를 반영하는 환경을 제공합니다.



### MemSim: A Bayesian Simulator for Evaluating Memory of LLM-based Personal Assistants (https://arxiv.org/abs/2409.20163)
Comments:
          26 pages, 25 tables, 1 figure

- **What's New**: 이 논문에서는 LLM(대형 언어 모델) 기반 개인 비서의 메모리 능력을 평가하기 위한 새로운 자동화된 방법인 MemSim을 제안합니다. MemSim은 사용자 메시지로부터 신뢰할 수 있는 질문-답변(QA)을 자동으로 생성할 수 있는 베이esian 시뮬레이터입니다.

- **Technical Details**: MemSim은 Bayesian Relation Network(BRNet)와 원인 생성 메커니즘을 사용하여 사용자 프로필을 다양한 계층적으로 생성하고, 이들로부터 신뢰할 수 있는 QA를 만듭니다. BRNet은 사용자에 대한 엔티티와 속성의 확률 분포를 모델링하는 기능을 갖추고 있으며, 구성된 QA는 메모리 메커니즘을 평가하는 데 사용됩니다.

- **Performance Highlights**: MemSim을 기반으로 생성된 MemDaily 데이터셋은 일상적인 시나리오에서 메모리 능력을 평가하기 위한 알고리즘의 효과를 평가하기 위한 실험에서 광범위하게 테스트되었습니다. 이 연구는 LLM 기반 개인 비서의 메모리 평가에서 첫 번째로 객관적이고 자동적인 방법을 제시하며, 연구 커뮤니티를 위해 이 프로젝트를 공개했습니다.



### Federated Instruction Tuning of LLMs with Domain Coverage Augmentation (https://arxiv.org/abs/2409.20135)
- **What's New**: 최근의 Federated Domain-specific Instruction Tuning (FedDIT)는 제한된 크로스-클라이언트 개인 데이터를 활용하여 특정 도메인에서 모델 성능을 향상시키는 방법으로 주목받고 있습니다. 이 과정에서 서버 측의 공용 데이터와 결합하여 모델을 개선하는 방법론을 제시합니다.

- **Technical Details**: FedDIT는 클라이언트 간 도메인 커버리지를 최적화하는 것을 목표로 하며, 이를 위해 greedy client center selection과 retrieval-based augmentation을 사용하여 데이터 보강을 진행합니다. 또한, FedDCA$^*$를 통해 클라이언트 측의 계산 부담을 덜기 위해 이질적인 인코더를 사용하고 서버 측에서 특징 정렬(feature alignment)을 수행합니다.

- **Performance Highlights**: 네 개의 다양한 도메인(코드, 의료, 재정, 수학)에서 진행된 광범위한 실험을 통해 FedDCA와 FedDCA$^*$의 효과를 확인하였고, 공공 데이터의 양과 개인 정보 보호 능력 사이에는 유의미한 상관관계가 없음을 밝혔습니다. 그러나 진행된 파인튜닝의 회수가 많아질수록 개인 정보 유출의 위험이 감소하는 경향이 나타났습니다.



### Robust LLM safeguarding via refusal feature adversarial training (https://arxiv.org/abs/2409.20089)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 보안 취약점을 해결하고자 새로운 방어 기법인 거부 기능 대항 훈련(Refusal Feature Adversarial Training, ReFAT)을 제안합니다. 이 접근법은 공격의 잠재적 영향을 시뮬레이션하여 LLM이 보다 안전한 반응을 생성하도록 훈련합니다.

- **Technical Details**: 거부 기능(ablation of refusal feature)이라는 메커니즘을 통해 LLM이 공격에 노출되는 방식을 분석하고, 이를 기반으로 ReFAT 알고리즘을 개발하였습니다. ReFAT는 LLM이 해로운 입력에 대해 거부 응답을 생성하도록 미세 조정하며, 각각의 배치에서 두 세트의 해로운 및 무해한 지침을 사용해 RF를 동적으로 계산합니다.

- **Performance Highlights**: 실험 결과, ReFAT는 세 가지 인기 있는 LLM에 대해 다양한 적대적 공격에 대한 강력함을显著 개선하며, 기존의 적대적 훈련 방법들과 비교하여 상당히 적은 계산 비용으로 효과를 나타냅니다.



### GUNDAM: Aligning Large Language Models with Graph Understanding (https://arxiv.org/abs/2409.20053)
- **What's New**: 이 논문에서는 텍스트 데이터 처리에서 뛰어난 성능을 보여준 Large Language Models (LLMs)를 그래프 구조 데이터를 이해하고 활용하는 데 적용하려는 새로운 접근 방식인 GUNDAM (Graph Understanding for Natural Language Driven Analytical Model)을 소개합니다. 기존 연구들이 주로 텍스트 속성이 풍부한 그래프에 초점을 맞춘 반면, 본 연구는 그래프 데이터 고유의 구조적 지식을 토대로 복잡한 추론 작업을 수행할 수 있는 LLM의 능력을 평가하고 향상시키고자 합니다.

- **Technical Details**: GUNDAM 모델은 그래프 구조를 LLM에 인코딩하기 위해 Graph Projection 방법을 사용합니다. 또한 CoT (Chain of Thought) 추론 경로를 포함한 고품질 그래프 추론 데이터 생성 파이프라인을 개발하여, 그래프 알고리즘을 활용해 정확성 및 중간 추론 과정을 제공합니다. 마지막으로, Alignment Tuning 방법을 통해 그래프 추론 데이터를 통한 GUNDAM의 미세 조정을 진행하여 모델의 추론 능력을 강화합니다.

- **Performance Highlights**: 실험 평가 결과, GUNDAM은 현재의 최첨단(SOTA) 성능을 초과 달성하였으며, LLM의 그래프 추론 능력에 영향을 미치는 주요 요소들도 밝혀졌습니다. 이 모델은 복잡한 그래프 구조를 이해하고 추론할 수 있는 능력을 개선하여 LLM의 일반 지능 발전에 기여할 가능성을 가지고 있습니다.



### Customized Information and Domain-centric Knowledge Graph Construction with Large Language Models (https://arxiv.org/abs/2409.20010)
Comments:
          Presented at CAIPI Workshop at AAAI 2024

- **What's New**: 본 논문에서는 지식 그래프(Knowledge Graph)를 기반으로 한 새로운 접근 방식을 제안하여 체계적인 정보에 신속하게 접근할 수 있도록 하고, 실행 가능한 기술 인텔리전스(actionable technology intelligence)를 지원하며 사이버 물리 시스템(cyber-physical systems) 계획 개선에 기여하고자 합니다.

- **Technical Details**: 제안하는 프레임워크는 텍스트 마이닝(text mining) 프로세스를 포함하며, 정보 검색(information retrieval), 키프레이즈 추출(keyphrase extraction), 의미 네트워크(semantic network) 생성 및 주제 맵 시각화(topic map visualization) 등을 포함합니다. 이 데이터 탐색 과정을 통해 선택적 지식 그래프 구축(selective knowledge graph construction) 접근법을 적용하며, 전자 및 혁신 온톨로지(ontology)를 기반으로 한 파이프라인을 통해 멀티-목표 의사결정(multi-objective decision-making)을 지원합니다. 자동차 전기 시스템(domain of automotive electrical systems) 분야에 이 방법론을 적용하여 확장 가능성을 입증하였습니다.

- **Performance Highlights**: 본 연구의 결과에 따르면, 제안한 지식 그래프 구축 프로세스는 GraphGPT 및 bi-LSTM과 transformer REBEL과 비교했을 때 클래스 인식(class recognition), 관계 구축(relationship construction), 올바른 'subclass of' 분류에서 여러 배로 우수한 성능을 보였습니다. 또한, 우선 인증된 문서(document genres)를 통해 생성된 지식 그래프는 보다 방대한 정보량과 더 나은 일관성을 갖춘 결과를 도출하는 데 크게 기여하였습니다.



### Developing Instruction-Following Speech Language Model Without Speech Instruction-Tuning Data (https://arxiv.org/abs/2409.20007)
Comments:
          Submitted to ICASSP 2025

- **What's New**: 본 연구에서는 음성 텍스트 쌍 데이터를 자동 생성하는 간단하면서도 효과적인 프로세스를 제시합니다. 이는 기존의 언어 모델(LLM)의 언어 능력을 보존하면서 음성의 비언어적 이해 능력을 SLM에 통합합니다.

- **Technical Details**: 제안된 데이터셋 구성 방법은 두 가지 혁신을 포함합니다. 첫째, 텍스트 기반 LLM을 활용하여 음성과 관련된 메타데이터를 사용하여 음성-텍스트 쌍을 생성합니다. 둘째, 'What can you hear from the audio?'라는 단일 프롬프트를 사용하여 데이터 구성을 진행합니다. 이 과정에서 생성된 데이터는 훈련하는 동안 SLM 모델의 학습 목표로 사용됩니다.

- **Performance Highlights**: DeSTA2는 Dynamic-SUPERB 및 AIR-Bench-Chat 벤치마크에서 선진 성과를 달성하였고, 복잡한 지시를 따르고 사고의 흐름 체계적 추론을 포함한 고급 추론 능력을 유지합니다. 이는 음성 이해 시스템의 효율성을 높이고, 광범위한 주석 데이터 세트에 대한 의존도를 줄이는 데 기여합니다.



### Mitigating Backdoor Threats to Large Language Models: Advancement and Challenges (https://arxiv.org/abs/2409.19993)
Comments:
          The 60th Annual Allerton Conference (Invited Paper). The arXiv version is a pre-IEEE Press publication version

- **What's New**: 본 논문은 Large Language Models(LLMs)의 백도어 공격(backdoor attack)의 위험성을 종합적으로 조사하고, LLMs에 대한 최근 방어 및 탐지 전략의 발전을 다룹니다.

- **Technical Details**: 백도어 공격은 훈련 데이터의 일부를 조작하여 LLM에 숨겨진 백도어를 주입하고, 사전 정의된 트리거에 의해 활성화되는 악의적인 행동을 유발합니다. 공격자들은 소량의 훈련 사례를 사용하여 모델의 특정 행동과 해당 트리거를 연관 지어 시스템을 해킹하거나 중단시킬 수 있습니다. 이러한 공격은 instruction tuning과 RLHF(강화 학습에서 인간 피드백)의 새로운 학습 패러다임을 통해 더욱 악화됩니다.

- **Performance Highlights**: LLMs에 대한 백도어 공격은 재정적 손실, 시장 혼란, 그리고 신뢰도 손상을 초래할 수 있습니다. 특히, 금융 분야와 헬스케어 같은 고위험 분야에서의 백도어 공격은 큰 피해를 유발할 가능성이 높습니다.



### Predictive Speech Recognition and End-of-Utterance Detection Towards Spoken Dialog Systems (https://arxiv.org/abs/2409.19990)
Comments:
          Submitted to ICASSP2025

- **What's New**: 이 논문은 자연어 처리(NLP)의 정확도를 향상시키기 위해 발화 중간 부분을 활용하여 바로 이어질 단어를 예측하고 발화 종료(EOU)까지의 시간을 추정하는 새로운 대화 시스템의 기능을 구현하는 것을 목표로 하고 있습니다.

- **Technical Details**: 저자들은 인코더-디코더 기반의 자동 음성 인식(ASR) 시스템을 제안하며, 발화의 미래 세그먼트를 마스킹(masking)하여 디코더가 마스킹된 오디오에서 단어를 예측하도록 훈련합니다. 또한, 음향(acoustic)과 언어적(linguistic) 정보를 모두 포함하는 크로스 어텐션(cross-attention) 기반 알고리즘을 개발하여 EOU를 정확하게 검출할 수 있도록 하고 있습니다.

- **Performance Highlights**: 제안된 모델은 실제 EOU와 최대 300ms까지의 단어를 예측하고 EOU 이벤트를 추정하는 능력을 보였습니다. 이러한 예측 기능은 ASR 성능에도 전반적인 개선 효과를 나타내며, 사용자의 대화 흐름을 효율적으로 지원합니다.



### Enhancing High-order Interaction Awareness in LLM-based Recommender Mod (https://arxiv.org/abs/2409.19979)
Comments:
          Long paper accepted to EMNLP 2024 Main. 16 pages

- **What's New**: 이번 논문에서는 LLM(대형 언어 모델)을 기반으로 한 추천 시스템 ELMRec(Enhanced LLM-based Recommender)를 제안합니다. 기존 추천 방법이 사용자-아이템의 고차 상호작용을 효과적으로 모델링하지 못하는 문제를 해결하기 위해, 새로운 whole-word embeddings를 도입하여 LLM의 추천 해석 능력을 향상시킵니다. 이 방법은 그래프 전처리 없이도 고차 상호작용 신호를 LLM이 쉽게 흡수할 수 있도록 합니다.

- **Technical Details**: ELMRec는 random feature propagation을 사용하는 novel whole-word embeddings를 개발하여 LLM이 추천에서의 고차 상호작용 인식을 크게 증가시킵니다. 이로 인해 각 ID는 해당 토큰과 연결된 whole-word embeddings로 표현되어, 사용자와 아이템 사이의 상대적 위치를 더 잘 포착합니다. 또한, 기계 학습 모델이 이전 상호작용에 기반하여 아이템을 추천하는 경향이 있음을 인지하고, 이 문제를 해결하기 위한 reranking 방법을 제안합니다.

- **Performance Highlights**: ELMRec는 직접 추천과 순차적 추천 모두에서 최신 기술(SOTA)보다 높은 성능을 기록했습니다. 실험 결과 ELMRec가 LLM 기반 추천 시스템의 뛰어난 성능을 보여주었다는 것을 입증합니다.



### Multimodal LLM Enhanced Cross-lingual Cross-modal Retrieva (https://arxiv.org/abs/2409.19961)
Comments:
          Accepted by ACM Multimedia

- **What's New**: 이 논문은 Cross-lingual Cross-modal Retrieval (CCR) 문제를 해결하기 위해 새로운 접근 방식인 LECCR을 제안합니다. LECCR은 다중 모달 대형 언어 모델(MLLM)을 통합하여 시각(feature) 및 비영어(non-English) 텍스트 표현 간의 정렬을 개선하는 데 중점을 두고 있습니다. 이는 인간 주석이 필요 없는 비영어 쿼리 기반 시각 콘텐츠 검색을 목표로 합니다.

- **Technical Details**: LECCR은 MLLM을 사용하여 상세한 시각 콘텐츠 설명을 생성하고, 이를 다중 뷰 의미론적 슬롯으로 집계하여 각기 다른 의미를 캡슐화합니다. 이러한 의미론적 슬롯을 내부 특성으로 사용하여 시각적 특성과 상호작용합니다. 또한, 비주얼과 비영어 특징 간의 정렬을 향상시키기 위해 영어 안내 아래 부드러운 매칭(softened matching) 기법을 도입합니다.

- **Performance Highlights**: Multi30K, MSCOCO, VATEX, MSR-VTT-CN의 네 가지 CCR 벤치마크에서 LECCR의 성능을 실험한 결과, 대부분의 평가 설정에서 이전 방법들을 초과하는 결과를 나타냈습니다. 이는 제안된 방법이 CCR 작업에서 효과적임을 강조합니다.



### TROPE: TRaining-Free Object-Part Enhancement for Seamlessly Improving Fine-Grained Zero-Shot Image Captioning (https://arxiv.org/abs/2409.19960)
Comments:
          Accepted to EMNLP 2024 Findings

- **What's New**: 본 논문은 TRaining-Free Object-Part Enhancement (TROPE)라는 새로운 방법을 소개하여, 이미지 캡셔닝에 있어 제로샷(zero-shot) 능력을 강화하는 데 초점을 맞추고 있습니다. TROPE는 기존 캡션의 세부사항을 보완하여, 다양한 객체의 세부 정보를 통합합니다.

- **Technical Details**: TROPE는 객체 탐지기(Object Detector) 제안과 자연어 처리(NLP) 기술을 활용하여 기초 캡션(base caption)에 추가적인 객체 부분 정보를 더합니다. 이 방법은 기본 캡션을 변경하지 않고 보완하여 캡션 생성 프로세스에 유연성을 제공합니다. TROPE는 기존 제로샷 이미지 캡셔닝 방법의 기반 캡션에 세부 정보를 효과적으로 추가합니다.

- **Performance Highlights**: TROPE는 모든 테스트된 제로샷 이미지 캡셔닝 접근 방식에서 성능을 일관되게 향상시키며, 세부 묘사가 요구되는 정밀한 이미지 캡셔닝 데이터셋에서 최신 기술(state-of-the-art) 성과를 달성했습니다. 평가 결과에 따르면, TROPE 사용 시 재현율(recall)이 크게 향상되었으며, 이는 기존의 방법들에 비해 더욱 세밀한 구성 요소를 캡션에 통합할 수 있음을 보여줍니다.



### Law of the Weakest Link: Cross Capabilities of Large Language Models (https://arxiv.org/abs/2409.19951)
Comments:
          Code: this https URL

- **What's New**: 이번 연구에서는 Large Language Models (LLMs)의 평가에서 개별 능력에 중점을 두었던 기존 접근에서 벗어나, 여러 전문 분야의 교차 능력(cross capabilities) 간의 상호작용을 탐구합니다.

- **Technical Details**: 연구팀은 7개의 핵심 개별 능력을 정의하고, 이를 엮어 7개의 일반적인 교차 능력을 생성하였습니다. 이 과정에서 수동으로 구축된 분류 체계(taxonomy)를 지원하였으며, 1,400개의 인간 주석이 달린 프롬프트로 구성된 CrossEval이라는 벤치마크를 도입하였습니다. 각 개별 및 교차 능력마다 100개의 프롬프트가 배정되며, 4,200개의 모델 응답을 평가하여 8,400개의 인간 평가를 수집하였습니다.

- **Performance Highlights**: 연구 결과, 현재 LLM들은 교차 능력 성능이 가장 약한 구성 요소에 의해 강하게 제약된다는 "Law of the Weakest Link" 현상을 보였습니다. 17개 모델의 58개 교차 능력 점수 중 38개는 모든 개별 능력보다 낮았으며, 20개는 강한 능력과 약한 능력 사이에 위치하였습니다. 이 결과는 LLM들이 교차 능력 과제에서 저조한 성과를 내고 있음을 보여주며, 복잡한 다차원 상황에서 성능을 최적화하기 위해 미래 연구의 우선 과제가 약한 능력의 식별 및 개선이라는 점을 강조합니다.



### Large Language Model Empowered Embedding Generator for Sequential Recommendation (https://arxiv.org/abs/2409.19925)
- **What's New**: Sequential Recommender Systems (SRS)에 대한 새로운 접근법으로, LLMEmb라는 혁신적인 기법이 제안되었습니다. 이 기법은 대형 언어 모델(LLM)을 활용하여 아이템 임베딩(item embeddings)을 생성하여 SRS의 성능을 높입니다.

- **Technical Details**: LLM의 일반적인 능력을 추천 도메인에 맞추기 위해 Supervised Contrastive Fine-Tuning (SCFT) 방법을 도입했습니다. SCFT는 속성 수준의 데이터 증대(attribute-level data augmentation)와 추천 성능 강화를 위한 맞춤형 대조 손실(custom contrastive loss)을 포함합니다. 또한, LLM 생성 임베딩에 협업 필터링 신호(collaborative filtering signals)를 통합할 필요성을 강조하고, 이를 위한 Recommendation Adaptation Training (RAT)을 제안합니다.

- **Performance Highlights**: 세 가지 실제 데이터 세트에 대한 폭넓은 실험을 통해, LLMEmb가 다양한 SRS 모델에서 현재 방법보다 상당한 성능 향상을 보여주었습니다.



### Scaling Optimal LR Across Token Horizon (https://arxiv.org/abs/2409.19913)
- **What's New**: 이번 연구는 대규모 언어 모델(LLM) 훈련에서 학습률(learning rate)과 토큰 수(token horizon) 간의 연관성에 대한 대규모 실증 연구를 수행하였습니다. 특히, 토큰 수가 많아질수록 최적의 학습률이 감소하는 경향이 있음을 확인하였습니다.

- **Technical Details**: 연구진은 Megatron 코드베이스를 사용하여 여러 LLM 모델의 학습률과 토큰 수 간의 관계를 살펴보았습니다. 실험 결과, 긴 토큰 수의 경우, 작은 학습률이 필요하며, 최적 학습률은 스케일링 법칙(scaling law)을 따릅니다. 이로써 짧은 토큰 수에서 얻은 최적 학습률을 바탕으로 긴 토큰 수에서의 최적 학습률을 추정할 수 있습니다.

- **Performance Highlights**: LLama-1 모델이 너무 높은 학습률을 사용했으며, 이는 성능 저하를 초래한다는 증거를 제공했습니다. 토큰 수에 따른 학습률 전이 방법론을 개발하여 현재의 관행에 부가적인 부담 없이 적용할 수 있도록 하였습니다.



### RouterDC: Query-Based Router by Dual Contrastive Learning for Assembling Large Language Models (https://arxiv.org/abs/2409.19886)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 본 논문에서는 여러 개의 대형 언어 모델(LLM)을 조합하는 새로운 방법인 query-based Router by Dual Contrastive learning (RouterDC)을 제안합니다. 기존 모델이 여러 LLM이 잘 작동하는 경우 비효율적임을 개선합니다.

- **Technical Details**: RouterDC 모델은 encoder와 LLM 임베딩으로 구성되며, 두 가지 대조 학습 손실(contrastive learning losses)을 사용하여 모델을 훈련합니다. 이는 다양한 쿼리에 대해 가장 적합한 LLM을 선택하는 라우팅(routing) 기술을 활용합니다.

- **Performance Highlights**: 실험 결과, RouterDC는 개별 최고 성능 LLM과 기존 라우팅 방법보다 각각 +2.76% 및 +1.90% 더 우수한 성능을 보여주었습니다. 최적의 LLM 조합을 통해 효과적으로 성능을 향상시킴을 입증합니다.



### ForecastBench: A Dynamic Benchmark of AI Forecasting Capabilities (https://arxiv.org/abs/2409.19839)
- **What's New**: 새로운 연구는 ForecastBench라는 동적 benchmark를 도입하여 머신러닝 (ML) 시스템의 예측 정확성을 평가할 수 있는 표준화를 제공한다.

- **Technical Details**: ForecastBench는 자동으로 생성되고 정기적으로 업데이트되는 1,000개의 예측 질문 세트로 구성되어 있으며, 데이터 유출 가능성을 방지하기 위해 제출 시점에 알려지지 않은 미래 사건에 대한 질문만 포함한다. 현재 ML 시스템의 능력을 평가하기 위해, 전문가(인간) 예측자, 일반 대중, LLMs로부터 benchmark에서 무작위 선택된 질문(N = 200)에 대한 예측을 수집하였다.

- **Performance Highlights**: 전문가 예측자는 상위 성능의 LLM보다 더 높은 예측 정확성을 보였으며(p-values <= 0.01), 이에 대한 결과는 공개 리더보드에서 확인할 수 있다.



### Calibrating Language Models with Adaptive Temperature Scaling (https://arxiv.org/abs/2409.19817)
Comments:
          EMNLP 2024

- **What's New**: 본 논문에서는 Adaptive Temperature Scaling (ATS)이라는 새로운 후처리 방법을 소개합니다. 이는 각 토큰 예측에 대해 온도 스케일링 매개변수를 예측하여 모델의 신뢰도를 개선합니다.

- **Technical Details**: ATS는 토큰 수준의 특성에 따라 조정되는 온도 값을 예측하며, 이는 표준 감독된 미세 조정(supervised fine-tuning) 데이터셋에 맞춰진 것입니다. 이 방법은 강화 학습(reinforcement learning)에서 인간 피드백(human feedback)을 사용하는 후 미세 조정 이후에 발생하는 보정(calibration) 변화에 적응합니다.

- **Performance Highlights**: ATS는 세 가지 다운스트림 자연어 평가 벤치마크에서 이전 보정 방법에 비해 10-50% 이상의 보정 개선을 달성하였으며, RLHF로 인한 성능 향상에는 방해가 되지 않습니다.



### CRScore: Grounding Automated Evaluation of Code Review Comments in Code Claims and Smells (https://arxiv.org/abs/2409.19801)
- **What's New**: 자동화된 코드 리뷰(automated code review)가 기계 학습(machine learning) 커뮤니티에서 최근 많은 주목을 받고 있습니다. 기존의 리뷰 코멘트 평가 지표는 코드 변경(diff)에 대해 인간이 작성한 참조(reference)와 비교하는 방식을 기반으로 하고 있지만, 코드 리뷰는 여러 개의 '유효한 리뷰(valid reviews)'가 존재하는 다대일(one-to-many) 문제입니다. 이를 해결하기 위해 새로운 평가 기준인 CRScore를 개발했습니다.

- **Technical Details**: CRScore는 코드의 결함(claims)과 잠재적 문제를 탐지하는 LLMs(대형 언어 모델) 및 정적 분석기(static analyzers)에 기반하여 리뷰 품질(review quality)의 차원들을 측정하는 참조가 필요 없는(reference-free) 지표입니다. 평가 항목으로는 간결성(conciseness), 포괄성(comprehensiveness), 관련성(relevance) 등이 포함됩니다. CRScore는 인간의 판단(human judgment)과 높은 상관관계(0.54 Spearman correlation)를 보이며, 기존 지표들보다 민감도가 높습니다.

- **Performance Highlights**: CRScore는 자동화된 메트릭(metric) 개발을 지원할 수 있도록 2.6k의 인간 주석(human-annotated) 리뷰 품질 점수를 포함한 코퍼스(corpus)를 공개하였습니다.



### AstroMLab 2: AstroLLaMA-2-70B Model and Benchmarking Specialised LLMs for Astronomy (https://arxiv.org/abs/2409.19750)
Comments:
          10 pages, 1 figure, 1 table, accepted to AI4S: The 5th Workshop on Artificial Intelligence and Machine Learning for Scientific Applications at the International Conference for High Performance Computing, Networking, Storage, and Analysis (SC24). Models will be released at this https URL. AstroMLab homepage: this https URL

- **What's New**: 이 연구는 도메인 특화된 데이터를 기반으로 한 지속적(pretraining) 학습의 중요성을 강조하며, 천문학에 특화된 대형 언어 모델(LLMs)의 성능을 정량적으로 평가합니다. 최근의 고품질 천문학 MCQ(다지선다형 질문) 컬렉션을 활용하여 이 연구의 목표를 달성하고자 하였습니다.

- **Technical Details**: AstroLLaMA 시리즈는 LLaMA-2-7B 기반으로 개발되었으나, 기본 모델에 비해 성능이 저하되었습니다. 이 연구는 arXiv의 요약된 텍스트와 같은 고품질 데이터를 이용한 지속적 학습으로 이러한 성능 저하를 부분적으로 완화할 수 있음을 보여줍니다. 70B 모델에 대한 지속적 학습이 significant improvements(상당한 개선)를 가져올 수 있지만, 여전히 감독(스포츠) 세부정보 데이터셋이 instruct 모델의 성능을 제약합니다.

- **Performance Highlights**: AstroLLaMA-3-8B 및 AstroLLaMA-2-70B라는 새로운 모델 세트를 소개하며, 이들은 이전 AstroLLaMA 시리즈를 기반으로 발전하였습니다. 작은 모델에서의 catastrophic forgetting(재앙적인 망각) 현상에도 불구하고, 70B 모델에 대한 지속적 학습이 유의미한 성과를 도출하였습니다.



### A multimodal LLM for the non-invasive decoding of spoken text from brain recordings (https://arxiv.org/abs/2409.19710)
Comments:
          15 pages, 4 figures

- **What's New**: 이번 연구에서 제안한 모델은 비침습적인 fMRI 기록에서 말하는 텍스트를 디코딩하는 멀티모달 LLM 모델입니다. 이는 정교한 트랜스포머 인코더와 고정된 대형 언어 모델을 연결하여 브레인 활동을 텍스트와 정렬하는 것으로, 기존 모델을 능가하는 성능을 보여줍니다.

- **Technical Details**: 제안된 아키텍처는 (i) 인코더와 (ii) 고정된 대형 언어 모델로 구성됩니다. 인코더는 특정 트랜스포머에서 파생된 것으로, 확대된 임베딩 레이어와 개선된 주의 메커니즘을 포함하고 있습니다. 학습 전략은 텍스트와 뇌 기록 간의 매핑을 위해 두 단계로 진행됩니다.

- **Performance Highlights**: 제안된 시스템은 다양한 텍스트 유사성 및 의미론적 지표에서 기존 아키텍처보다 우수한 결과를 기록했으며, 디코딩된 대화 맥락의 시각적 품질이 높아 주요 대화 키워드를 식별할 수 있음을 보여줍니다.



### Federated Learning from Vision-Language Foundation Models: Theoretical Analysis and Method (https://arxiv.org/abs/2409.19610)
- **What's New**: 본 논문은 CLIP과 같은 사전학습된 비전-언어 기초 모델을 연합 학습(federated learning)에 통합하여 다양한 작업에 대한 일반화(generalization)를 향상시키는 데 중점을 두고 있습니다. 특히, 프롬프트 기반 연합 학습(prompt-based federated learning)의 성능을 이해하기 위한 이론적 분석 프레임워크가 제시됩니다.

- **Technical Details**: 프롬프트 기반 연합 학습을 위한 분석 프레임워크는 feature learning theory를 기반으로 구성되어 있으며, 신호 학습(signal learning)과 노이즈 기억(noise memorization)의 진화를 모니터링합니다. 성과는 작업 관련(task-relevant) 계수와 작업 비관련(task-irrelevant) 계수의 비율로 평가됩니다. 또한, 포트폴리오 최적화(portfolio optimization)에서의 수익(income)과 위험(risk)의 유사성을 바탕으로, 글로벌 프롬프트(global prompt)와 로컬 프롬프트(local prompt)를 결합하여 프롬프트 포트폴리오를 구축합니다.

- **Performance Highlights**: 실험을 통해 프롬프트 포트폴리오의 성능 우위를 입증하였으며, 최적의 혼합 계수를 도출했습니다. 이론적 주장들은 실증적 실험에서도 지지를 받으며, 실제 시나리오에서의 접근 방식의 우수성을 꾸준히 보여주고 있습니다.



### Hyper-Connections (https://arxiv.org/abs/2409.19606)
- **What's New**: 새로운 방법론인 하이퍼 연결(hyper-connections)을 소개합니다. 이 방법은 기존의 잔여 연결(residual connections)의 몇 가지 단점을 다루는 것을 목표로 하며, 네트워크가 각기 다른 깊이의 피처(feature) 사이의 연결 강도를 조절하고 레이어를 동적으로 재배치할 수 있게 합니다.

- **Technical Details**: 하이퍼 연결은 네트워크가 피처 간의 연결 강도를 학습할 수 있게 하며, 깊이 연결(depth-connections)과 폭 연결(width-connections)을 제안합니다. 이를 통해 잔여 연결의 장점을 유지하면서도 계산량과 매개변수 증가를 최소화할 수 있습니다. 또한 동적 하이퍼 연결(dynamic hyper-connections)을 통해 입력에 따라 연결 가중치를 조정할 수 있습니다.

- **Performance Highlights**: 하이퍼 연결은 대형 언어 모델(LLMs)의 사전 학습(pre-training) 및 비전 처리(vison tasks)에서 잔여 연결 대비 성능 향상을 보였습니다. 예를 들어, DHC를 사용하는 모델이 1.8배 더 빠르게 수렴하며, ARC-Challenge에서 약 6점 향상을 나타냈습니다. 또한, 하이퍼 연결이 적용된 모델은 인접한 레이어 간의 특성 유사성을 줄이고, 각 레이어의 영향을 확장하는 데 기여합니다.



### The Crucial Role of Samplers in Online Direct Preference Optimization (https://arxiv.org/abs/2409.19605)
Comments:
          33 pages

- **What's New**: 이 논문에서는 Direct Preference Optimization (DPO)의 수렴 속도(convergence rates)를 다양한 샘플링 전략(sampling strategies) 하에서 철저히 분석합니다. 특히, uniform sampling이 linear convergence를 달성하는 반면, 제안된 online sampler는 quadratic convergence를 이룬다는 점이 놀랍습니다.

- **Technical Details**: DPO의 수렴 속도에 대한 이론적 분석을 제공하고, posterior distributions와 logit mixing을 통합하여 실용적인 환경에서도 샘플러(sampler)를 조정했습니다. 다양한 환경에서 DPO의 성능을 개선하기 위해 bandit environments 내에서의 최적화를 진행하였습니다.

- **Performance Highlights**: Safe-RLHF 데이터셋에서 제안한 방법은 vanilla DPO에 비해 4.5% 향상되었고, on-policy DPO에 대해서도 3.0%의 성과를 보였으며, Iterative-Prompt에서 vanilla DPO, on-policy DPO, Hybrid GSHF를 각각 4.2% 이상 능가하는 성과를 달성했습니다.



### Two-stage Framework for Robust Speech Emotion Recognition Using Target Speaker Extraction in Human Speech Noise Conditions (https://arxiv.org/abs/2409.19585)
Comments:
          Accepted to APSIPA ASC 2024

- **What's New**: 이 논문에서는 인간 음성 노이즈를 처리하기 위해 Target Speaker Extraction (TSE)과 Speech Emotion Recognition (SER) 시스템을 연결한 새로운 두 단계 프레임워크를 제안합니다. 기존의 SER 연구는 언급된 음성 노이즈 영향을 고려하지 않았습니다.

- **Technical Details**: 제안된 시스템은 첫 번째 단계에서 TSE 모델을 훈련하여 혼합된 음성에서 특정 화자의 음성을 추출하고, 두 번째 단계에서는 추출된 음성을 사용하여 SER 학습을 진행합니다. 여기서 TSE와 SER 모델의 공동 학습도 탐구됩니다. TSE는 혼합된 음성 신호에서 목표 화자의 음성을 재구성하는 작업으로 정의되며, TD-SpeakerBeam을 TSE 모델로 사용합니다.

- **Performance Highlights**: 이 시스템은 TSE를 사용하지 않은 기준선 모델과 비교했을 때 Unweighted Accuracy (UA)에서 14.33% 향상을 달성했습니다. 또한 성별에 따른 실험 결과, 서로 다른 성별의 조합에서 프레임워크가 특히 뛰어난 성능을 보였음을 보여줍니다.



### Quantitative Analysis of Audio-Visual Tasks: An Information-Theoretic Perspectiv (https://arxiv.org/abs/2409.19575)
Comments:
          Accepted by ISCSLP2024

- **What's New**: 본 논문은 음성 언어 처리 분야에서 오디오-비주얼(visual) 음성 처리에 대한 이론적 분석을 제시하며 정보 이론에 기반한 정량적 분석을 통해 서로 다른 모달리티(modality) 간의 정보 교차를 조사합니다.

- **Technical Details**: 정보 이론에 따라, 정보의 양은 변수의 불확실성으로 나타낼 수 있으며, 이를 정보 엔트로피(information entropy)로 정량화합니다. 이 논문에서는 각각의 오디오, 비디오, 텍스트 모달리티의 정보 엔트로피와 상호 정보(mutual information, MI)를 계산하여 세 가지 모달리티 간의 관계를 분석합니다. 분석은 클러스터링 기반 방법(clustering-based method)을 사용하여 연속적이고 고차원인 특징(feature)을 불연속적으로 변환합니다.

- **Performance Highlights**: 본 연구는 CNVSRC-Multi, GRID, LRS3 세 가지 데이터셋에서 실험을 진행하였으며, 오디오-비주얼 처리의 불확실성과 상관 관계를 깊이 이해하는 데 기여합니다. 특히, 100k 시간 이상의 시각-음성 데이터를 훈련하여 얻은 단어 오류율(WORD ERROR RATE, WER)은 13% 미만으로 감소하였고, 매우 소음이 많은 환경에서도 AVSR 방법이 큰 성능 향상을 보였습니다.



### Video DataFlywheel: Resolving the Impossible Data Trinity in Video-Language Understanding (https://arxiv.org/abs/2409.19532)
Comments:
          Under peer review

- **What's New**: 이번 연구에서는 데이터 수량, 다양성, 품질 간의 "불가능한 삼위일체"를 밝히며, 기존의 대규모 ASR 데이터셋이 품질 부족으로 인해 개선을 요구하고 있음을 강조합니다.

- **Technical Details**: 우리는 Video DataFlywheel 프레임워크를 도입하여 비디오 주석을 반복적으로 개선하며, AdaTaiLr라는 새로운 noise control 방법을 통해 대규모 데이터셋에서의 효율성을 증명합니다. 또한, 비디오-언어 모델을 활용하여 합성 주석을 생성하고, 이를 통해 데이터셋을 정제합니다.

- **Performance Highlights**: 우리의 프레임워크는 기존 데이터 정제 기준선보다 3% 성능 향상을 보였으며, 다양한 비디오-언어 이해 과제에서 중요한 개선을 facilitated 합니다.



### MedCLIP-SAMv2: Towards Universal Text-Driven Medical Image Segmentation (https://arxiv.org/abs/2409.19483)
Comments:
          10 pages, 2 figures, 6 tables

- **What's New**: 이 논문에서는 CLIP와 SAM 모델을 통합한 MedCLIP-SAMv2라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 텍스트 프롬프트를 사용하여 의료 이미지를 제로샷(Zero-shot) 및 약간의 감독하에 세분화하는 기능을 제공하며, 특히 의료영상 분석에서 데이터 효율성을 높이기 위해 제안되었습니다.

- **Technical Details**: MedCLIP-SAMv2 프레임워크는 BiomedCLIP 모델을 새로운 분리된 하드 네거티브 노이즈 대조 추정(DHN-NCE) 손실을 사용하여 파인 튜닝(tuning) 합니다. 또한 Multi-modal Information Bottleneck (M2IB)을 활용하여 제로샷 설정에서 SAM을 통해 세분화 마스크를 생성하기 위한 비주얼 프롬프트를 생성합니다. 프레임워크는 CT, MRI, 초음파 및 X-ray와 같은 다양한 의학 이미징 모달리티에 걸쳐 실험되었습니다.

- **Performance Highlights**: MedCLIP-SAMv2 프레임워크는 유방 종양 초음파, 뇌 종양 MRI, 폐 X-ray 및 폐 CT를 포함한 네 가지 세분화 작업에서 높은 정확도를 달성했습니다. 이러한 실험 결과는 프레임워크의 강력함과 다재다능성을 보여 줍니다.



### SELP: Generating Safe and Efficient Task Plans for Robot Agents with Large Language Models (https://arxiv.org/abs/2409.19471)
- **What's New**: 본 논문은 로봇 에이전트가 자연어 명령을 이해하고 수행하는 능력을 향상시키기 위해 대형 언어 모델(LLMs)을 활용하면서도, 사용자가 지정한 제약조건을 준수하도록 보장하는데의 도전 과제를 다룹니다. 'Safe Efficient LLM Planner(SELP)'라는 접근법을 제안하며, 이는 복잡한 작업을 처리하는 LLM 계획 수립자의 능력을 개선하는 세 가지 주요 통찰력을 포함합니다.

- **Technical Details**: 이 연구에서는 '동등성 투표(equivalence voting)', '제약 Decoding(constrained decoding)', '도메인 특화 미세 조정(domain-specific fine-tuning)' 기법을 통해 LLM 계획 수립자의 성능을 향상시킵니다. 동등성 투표는 NL 명령에서 여러 Linear Temporal Logic (LTL) 공식을 생성하고 샘플링하여 일치하는 LTL 공식을 그룹화하고, 다수 그룹의 공식을 최종 LTL 공식으로 선택하는 방법입니다. 제약 Decoding은 생성된 LTL 공식을 사용하여 계획의 자동 회귀 추론을 강제하며, 여기서 LLM은 주어진 명세에 부합하도록 계획을 수정합니다.

- **Performance Highlights**: SELP는 다양한 로봇 에이전트와 작업에서 효과성과 일반성을 보여주며, 드론 내비게이션 작업에서는 안전률에서 SOTA 계획 수립자보다 10.8% 향상되었고, 계획 효율성에서 19.8% 향상되었습니다. 로봇 조작 작업에서는 20.4% 안전률 개선을 달성하였습니다. 이를 통해 SELP는 높은 신뢰도로 사용자의 명령에 부합하는 계획을 세울 수 있음을 입증하였습니다.



### 'Simulacrum of Stories': Examining Large Language Models as Qualitative Research Participants (https://arxiv.org/abs/2409.19430)
- **What's New**: 이 연구는 LLMs(대규모 언어 모델)을 사용해 연구 참가자를 시뮬레이션할 경우 발생하는 윤리적 및 인식론적 문제를 탐구합니다. 저자들은 19명의 질적 연구자와의 인터뷰를 통해 LLM 데이터의 활용 가능성과 한계에 관한 시각 변화를 이해하고자 했습니다.

- **Technical Details**: 연구자들은 LLM이 생성한 데이터가 인간 참가자로부터 수집한 데이터와 유사한 내러티브를 포함하고 있음을 발견했습니다. 그러나 대화가 진행됨에 따라, LLM의 응답에는 중요한 한계가 발견되었으며, 이러한 한계에는 (1) LLM의 응답이 실질적이지 않음, (2) 모델의 인식적 위치의 모호성, (3) 연구자의 위치성이 강화됨, (4) 참가자의 동의 및 에이전시가 제한됨, (5) 커뮤니티의 관점이 지워짐, (6) 질적 연구 방법의 정당성이 훼손될 위험이 포함됩니다.

- **Performance Highlights**: 연구 결과는 LLM을 사용한 대체 연구 방식이 질적 연구의 특성과 윤리에 미치는 부정적인 영향을 강조합니다. LLM은 여전히 텍스트 생성 능력이 뛰어나지만, 지식 생산에 필요한 구체적이고 맥락이 포함된 이해가 부족하다는 점이 강조되었습니다.



### DOTA: Distributional Test-Time Adaptation of Vision-Language Models (https://arxiv.org/abs/2409.19375)
Comments:
          In submission

- **What's New**: 이 논문에서는 기존의 Training-Free Test-time Dynamic Adapter(TDA)의 한계를 극복하기 위해 DistributiOnal Test-time Adaptation(Dota)라는 새로운 방법을 제안합니다. Dota는 테스트 샘플의 분포를 지속적으로 추정하여 모델이 배포 환경에 적응할 수 있게 합니다.

- **Technical Details**: Dota는 Bayes' 정리를 기반으로 하여 추정한 분포를 사용하여 테스트 샘플의 후방 확률(test-time posterior probabilities)을 계산합니다. 여기에서 각 클래스의 임베딩 분포가 가우시안 분포를 따른다는 가정을 하며, 이는 TDA보다 약 20배 빠른 추론 속도를 제공합니다. 또한, Dota는 사람-주도 피드백(human-in-the-loop paradigm)을 통해 불확실한 샘플을 식별하고 적응하도록 돕습니다.

- **Performance Highlights**: 광범위한 데이터셋을 통한 실험 결과 Dota는 CLIP 모델이 지속적으로 학습할 수 있게 해주며, 기존의 최첨단 방법들에 비해 유의미한 성과 향상을 보였습니다.



### Decoding Echo Chambers: LLM-Powered Simulations Revealing Polarization in Social Networks (https://arxiv.org/abs/2409.19338)
Comments:
          10 pages, 5 figures

- **What's New**: 이 연구에서는 소셜 미디어에서의 여론 형성을 조사하기 위해 LLM 기반의 시뮬레이션 프레임워크를 제안합니다. 이는 기존의 수치 모델링 접근 방식에서 벗어나 텍스트를 통해 전달되는 미묘한 의미를 포착하는 데 초점을 맞춥니다.

- **Technical Details**: 제안하는 Social Simulation Framework (SSF)는 각 개인을 LLM 에이전트로 나타내며, 다양한 사회적 상호작용을 시뮬레이션합니다. 세 가지 네트워크 구조: small-world, scale-free, random graph을 설정하며, 추천 알고리즘을 통해 에이전트 간의 상호작용을 진행합니다. 이 과정에서 각 에이전트는 짧은 기억과 긴 기억을 통해 정보를 저장하고 분석합니다.

- **Performance Highlights**: 우리의 SSF는 기존의 Bounded Confidence Model (BCM) 및 Friedkin Johnsen 모델과 비교하여 opinion polarization 및 echo chambers의 현상을 효과적으로 재현합니다. 또한, 적극적(nudges) 및 수동적(nudges) 방법을 제안하여 echo chambers를 줄이는 방법을 제시합니다.



### A Generalized Model for Multidimensional Intransitivity (https://arxiv.org/abs/2409.19325)
Comments:
          13 pages, 1 figure

- **What's New**: 이 논문에서는 비가환성(intransitivity) 문제를 해결하기 위해, 플레이어의 d차원 표현(d>1)과 데이터셋 특화 거리 공간(metric space)을 공동으로 학습하는 확률적 모델을 제안하였습니다. 이는 비가환적 표현 학습에 있어 이전의 모델로 수렴할 수 있는 흥미로운 결과를 보여줍니다.

- **Technical Details**: 제안된 모델은 각 플레이어의 d차원 표현을 학습하고, 해당 표현 공간에서 적절한 거리 형식을 체계적으로 캡처합니다. 추가적인 제약 조건을 통해 이 모델은 기존의 비가환적 표현 학습에 사용되는 모델로 수렴할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 사회적 선택(social choice), 선거(election), 온라인 게임 데이터셋에 대한 예측 성능이 향상되었습니다. 여러 경쟁 모델들에 비해 예측 정확도에서 우수한 성과를 보였습니다.



### DENEB: A Hallucination-Robust Automatic Evaluation Metric for Image Captioning (https://arxiv.org/abs/2409.19255)
Comments:
          ACCV 2024

- **What's New**: DENEB를 도입하여 이미지 캡션 생성에서의 환각(hallucination)에 대한 강건성을 향상시킨 새로운 자동 평가 지표를 제안

- **Technical Details**: DENEB는 Sim-Vec Transformer를 통합하여 동시에 여러 참조 캡션을 처리하며, Nebula 데이터셋에서 32,978개의 이미지와 805명의 주석가의 인간 평가를 통해 훈련됨

- **Performance Highlights**: DENEB는 FOIL, Composite, Flickr8K-Expert, Flickr8K-CF, PASCAL-50S 및 Nebula 데이터셋에서 기존 LLM-free 지표 중 최첨단 성과 달성



### Jointly modelling the evolution of community structure and language in online extremist groups (https://arxiv.org/abs/2409.19243)
- **What's New**: 이 논문에서는 사회적 구조와 언어 사용의 진화를 동시에 모델링하는 새로운 방법론을 제안하고, 이를 극단주의 반여성 온라인 그룹인 'manosphere'에 적용했습니다. 이를 통해 동적(temporal) 사용자 및 단어 표현을 도출하며, 이러한 접근이 이전 모델에 비해 어떻게 개선되었는지를 보여줍니다.

- **Technical Details**: 제안된 모델은 다수의 시점에서 사회적 인접 행렬(social adjacency matrix)과 언어 콘텐츠 행렬(language content matrix)을 함께 분해하는 방법으로 구성됩니다. 이 모델은 사용자 간의 상호작용을 기반으로 한 사회적 연결(소통)과 각 사용자의 언어 사용 패턴을 분석합니다. 연구에서는 사용자와 단어 임베딩(embeddings) 모두의 변화를 탐색하고, 이러한 변화가 사용자의 행동 예측 및 언어 진화에 어떻게 영향을 미치는지를 중점적으로 다룹니다.

- **Performance Highlights**: 실험 결과에 따르면, 제안된 모델은 재구성 충실도(reconstruction fidelity)와 클러스터 순도(cluster purity) 면에서 기존 모델들보다 통계적으로 유의미한 개선을 보였습니다. 특히, 사용자 임베딩을 통한 개인 행동 예측에서 높은 정확성을 기록하였고, 카테고리 내에서의 폭력적 언어의 성향을 새롭게 규명하는 데 기여했습니다.



### Evidence Is All You Need: Ordering Imaging Studies via Language Model Alignment with the ACR Appropriateness Criteria (https://arxiv.org/abs/2409.19177)
Comments:
          15 pages main text, 4 figures, 1 table

- **What's New**: 이번 연구에서는 generative AI와 대형 언어 모델을 활용하여 적절한 이미징 연구를 권장하는 프레임워크를 소개합니다. 이를 통해 evidence-based medical guidelines(증거 기반 의료 지침)에 따른 이미징 연구 주문을 지원합니다.

- **Technical Details**: 우리는 환자 'one-liner' 시나리오의 새로운 데이터셋을 제공하며, 이를 통해 최신 언어 모델을 최적화하여 의료 전문가들과 유사한 정확도를 달성합니다. 연구에서는 American College of Radiology(ACR)에서 제시한 Appropriateness Criteria(적절성 기준)에 따라 이미징 연구 주문의 정확성을 개선하는 데 초점을 두었습니다.

- **Performance Highlights**: 이 연구의 결과는 언어 모델 기반의 파이프라인이 임상의사들이 이미지 주문 워크플로를 지원하고 ACR AC에 따른 이미징 연구 주문의 정확성을 향상시킬 수 있는 지능형 도구로 사용될 수 있음을 보여줍니다.



### Confidential Prompting: Protecting User Prompts from Cloud LLM Providers (https://arxiv.org/abs/2409.19134)
- **What's New**: 이번 연구에서는 클라우드 기반 대규모 언어 모델(LLM) 서비스에서 사용자 입력의 보안을 강화하면서 출력 일관성, 모델 기밀성 및 계산 효율성을 유지하는 문제를 다루고 있습니다. 우리는 사용자 프롬프트를 신뢰할 수 있는 실행 환경인 Confidential Virtual Machine (CVM) 내에서 제한하면서 서비스 제공자가 효율적으로 토큰을 생성할 수 있도록 하는 Secure Multi-party Decoding (SMD)라는 새로운 접근 방식을 소개합니다.

- **Technical Details**: 본 연구에서 제안하는 Secure Multi-party Decoding (SMD) 방식은 클라우드 기반 LLM 서비스를 위한 새로운 접근 방식을 제공합니다. SMD는 LLM 토큰 생성 과정을 안전한 두 당사자 계산으로 모델링하여, 사용자 프롬프트에 해당하는 개인 상태를 CVM 내에서 처리하며, 새로 생성된 토큰에 해당하는 공개 상태는 호스트에서 처리합니다. 또한, Prompt Obfuscation (PO)라는 새로운 암호화 방법을 도입하여 SMD의 재구성 공격에 대한 강건성을 보장합니다.

- **Performance Highlights**: SMD+PO의 성능을 NVIDIA H100 GPU를 사용하여 평가한 결과, SMD+PO는 기존의 단순 CVM 기반 솔루션 및 CC 없이 이루어진 LLM 서비스에 비해 10-100배 향상된 처리량을 달성하였으며, 동시 사용자 수, 모델 크기 및 출력 토큰 수에 대해 뛰어난 확장성을 보여주었습니다.



### Show and Guide: Instructional-Plan Grounded Vision and Language Mod (https://arxiv.org/abs/2409.19074)
Comments:
          Accepted at EMNLP 2024 Main Track

- **What's New**: MM-PlanLLM은 사용자가 지침 작업을 실행할 수 있도록 돕기 위해 텍스트 플랜과 비주얼 정보를 모두 활용하는 다중 모달 대형 언어 모델입니다. 이 모델은 복잡한 절차적 계획을 시각적으로 안내하기 위해 설계되었습니다.

- **Technical Details**: MM-PlanLLM은 사용자의 쿼리에 따라 관련 단계 비디오 세그먼트를 검색하는 'Conversational Video Moment Retrieval'과 사용자의 현재 진행 상황을 설명하는 이미지에 기반하여 계획의 다음 단계를 생성하는 'Visually-Informed Step Generation'을 통해 상호 모달성을 도입합니다. 이 모델 아키텍처는 특정 작업을 통해 비디오의 의미적 및 시간 정보를 캡처하고, 유연한 디코드 시간 다중 모달 검색을 지원하도록 설계되었습니다.

- **Performance Highlights**: MM-PlanLLM은 다중 모달 및 텍스트 대화에서 모두 강력한 성능을 발휘하며, 텍스트 전용 요청에 대한 성능 저하가 제한적입니다. 평가 결과는 텍스트 전용 작업에서 경쟁력 있는 성능을 보이며, 기존 접근 방법에 비해 다중 모달 작업에서 상당한 개선을 이룹니다.



### CLLMate: A Multimodal LLM for Weather and Climate Events Forecasting (https://arxiv.org/abs/2409.19058)
- **What's New**: 이번 연구에서 제안된 Weather and Climate Event Forecasting (WCEF) 작업은 과거 날씨 및 기후 이벤트와 기상 데이터를 연계하여 날씨 및 기후 사건을 예측하는 새로운 과제로, 기존의 닫힌 집합 이벤트 예측 방식의 한계를 극복하는 데 중점을 두고 있습니다.

- **Technical Details**: WCEF 작업은 기상 레스터 데이터(meteorological raster data)와 사건 텍스트(data)를 통합하여 기상 사건을 예측하는 것을 목표로 합니다. 이를 위해 Llama3와 같은 대형 언어 모델(LLM)을 활용하여 과거의 기상 데이터를 사건으로 정리한 지식 그래프(knowledge graph)를 구성하였고, 이에 기반하여 최초의 다중 모달 지침 데이터셋(multimodal instruction dataset)을 생성하였습니다. 나아가 CLLMate라는 다중 모달 LLM을 통해 날씨 예측을 위한 기상 레스터 데이터를 효과적으로 관리하고 있습니다.

- **Performance Highlights**: CLLMate 모델은 기존의 기준 모델(baselines) 및 다른 다중 모달 LLM과 비교하여 뛰어난 성능을 보였으며, 기상 데이터와 사건 데이터를 효과적으로 정렬 및 통합하여 개방형 사건 예측을 단순화했습니다. 연구 결과는 WCEF 작업에 대한 연구의 미래가 가능하다는 점을 강조하며, 이 모델은 기후 위험 완화 시스템의 중요한 구성 요소로 자리 잡을 수 있는 잠재력을 보여줍니다.



### Self-Replicating Mechanical Universal Turing Machin (https://arxiv.org/abs/2409.19037)
- **What's New**: 이번 논문은 생체 모방 기법(bio-inspired mechanisms)을 이용한 자기 복제 유한 상태 기계(self-replicating finite-state machine, FSM)와 자기 복제 튜링 기계(self-replicating Turing Machine, TM)의 구현을 보여줍니다.

- **Technical Details**: 이 연구는 정보 정렬(sorting), 복사(copying), 읽기(reading) 능력을 가진 자기 복제 구조에 대한 기존 연구를 바탕으로, 작동하는 FSM과 TM을 명시적으로 구성하여 이러한 기계의 계산 능력을 입증합니다. 특히 Neary와 Woods의 UTM(5,5)을 모방하여 시스템의 보편성을 시연했습니다.

- **Performance Highlights**: 이 연구는 생체 모방 기법을 통해 구현된 FSM과 TM이 실제로 작동하는 모습을 보여주며, 이러한 자기 복제 기계들이 다양한 계산 문제를 해결할 수 있는 능력을 지니고 있음을 강조합니다.



### Backdoor Attacks for LLMs with Weak-To-Strong Knowledge Distillation (https://arxiv.org/abs/2409.17946)
- **What's New**: 이 연구는 Parameter-Efficient Fine-Tuning (PEFT)를 기반으로 한 새로운 백도어 공격 알고리즘인 W2SAttack을 제안합니다. 이 알고리즘은 작은 모델에서 백도어 기능을 전이하여 큰 모델에서 공격 효과를 극대화합니다.

- **Technical Details**: W2SAttack은 feature alignment-enhanced knowledge distillation을 활용하여 작은 언어 모델(teacher model)에서 큰 모델(student model)로 백도어를 전달합니다. 이는 PEFT를 사용하여 매개변수 업데이트를 최소화하고, 트리거와 목표 레이블 간의 정렬을 강화합니다.

- **Performance Highlights**: W2SAttack은 다양한 언어 모델과 아키텍처에서 거의 100%의 공격 성공률을 달성했으며, 분류 성능을 유지하면서 PEFT에 대한 백도어 공격을 효과적으로 강화했습니다.



New uploads on arXiv(cs.IR)

### ECORS: An Ensembled Clustering Approach to Eradicate The Local And Global Outlier In Collaborative Filtering Recommender System (https://arxiv.org/abs/2410.00408)
Comments:
          6 pages, 5 figures

- **What's New**: 이번 논문에서는 기존의 추천 시스템에서의 이상값(Outlier) 탐지를 개선하기 위해 다양한 클러스터링(clustering) 알고리즘을 활용한 새로운 접근법을 제안합니다. 특히, 사용자-사용자 매트릭스를 기반으로 한 클러스터링 기법을 통해 시스템 내에서 의심스러운 사용자를 식별합니다.

- **Technical Details**: 사용자 간의 유사성을 기반으로 한 매트릭스를 구성하고, 이를 통해 로컬(local) 및 글로벌(global) 이상값을 감지합니다. 클러스터링 기반 접근법을 사용하여 비정상적인 데이터가 포함된 작은 클러스터를 생성하고, 이러한 데이터를 포함하는 클러스터는 일반 데이터 객체의 밀집 클러스터와 대조됩니다.

- **Performance Highlights**: 실험 결과 이 접근법은 추천 시스템에서 이상값 탐지의 정확성을 상당히 향상시키는 것으로 나타났습니다. 다양한 클러스터링 알고리즘의 성능을 비교함으로써, 보다 명확하게 데이터의 이상값을 예측할 수 있음을 입증하였습니다.



### Winning Solution For Meta KDD Cup' 24 (https://arxiv.org/abs/2410.00005)
- **What's New**: 이번 논문은 Meta KDD Cup 24에서 db3 팀이 제안한 모든 작업의 우승 솔루션을 설명합니다. 이 대회는 웹 소스 및 지식 그래프를 기반으로 RAG 시스템을 구축하는 것이 목표입니다. 각 쿼리에 대한 여러 소스를 통해 질문에 대한 응답을 지원하며, 정보 요약, 구조화된 데이터 통합, 실제 검색 도전 과제를 반영하는 데이터 선택 등의 세 가지 작업을 포함합니다. db3 팀은 모든 작업에서 1위를 차지하며, 각각의 점수는 28.4%, 42.7%, 47.8%입니다.

- **Technical Details**: CRAG(Comprehensive RAG Challenge)는 언어 모델 응답의 신뢰성을 보장하기 위해 RAG 시스템을 측정하기 위한 2024 KDDCup 이벤트입니다. 크기와 유형이 다양한 질문과 정보를 다루기 위해 3가지 작업이 설정되었습니다: 1. 웹 기반 검색 요약, 2. 지식 그래프 및 웹 증대, 3. End-to-End RAG. db3 팀은 세 가지 작업에서 각각 다른 정보 소스를 사용하여 솔루션을 제시하고, LLM(대형 언어 모델) 조정을 통해 성능을 향상시켰습니다. RAG 시스템은 정보를 효과적으로 필터링하고 통합하여 정확한 응답을 생성하기 위해 여러 단계의 정보 검색 및 재정렬 체계를 채택합니다.

- **Performance Highlights**: db3 팀은 RAG 시스템을 통해 두 가지 알고리즘, 즉 bge-base-en-v1.5(정보 검색)와 bge-reranker-v2-m3(정보 재정렬)을 사용하여 점수를 기록하고, 세 가지 작업 모두에서 1위를 차지했습니다. 작업 #1에서는 웹 페이지에서 정보를 효과적으로 추출하고, 작업 #2 및 #3에서는 구조화된 API에서 데이터를 성공적으로 통합하여 결과적인 정확성을 높였습니다.



### Retro-li: Small-Scale Retrieval Augmented Generation Supporting Noisy Similarity Searches and Domain Shift Generalization (https://arxiv.org/abs/2410.00004)
Comments:
          Published as a conference paper at European Conference on Artificial Intelligence 2024

- **What's New**: 본 연구에서는 Retro와 같은 retrieval augmented generation (RAG) 시스템의 개선점을 제안하고, 소규모 데이터베이스에서도 효과적일 수 있다는 사실을 보여줍니다. 특히, 보다 정확한 이웃 검색을 통해 더 나은 결과물을 도출할 수 있다고 강조합니다.

- **Technical Details**: Retro-li는 소규모 비모수 메모리(non-parametric memory)에서 높은 품질의 이웃을 검색하기 위해 적절한 의미 유사성 검색(semantic similarity search)을 사용합니다. 또한, 이웃 검색 중 노이즈를 줄이기 위해 처음으로 정규화(regularization)를 추가하여 불확실성을 줄이는 데 기여합니다. RAG 모델의 데이터베이스 업데이트는 뛰어난 효율성을 자랑하며, 사용자는 도메인 간 전환 없이도 데이터베이스를 쉽게 대체할 수 있습니다. 또한, Retro-li는 아날로그 메모리 인메모리 컴퓨팅(hardware)에서 O(1) 검색 시간을 실현할 수 있습니다.

- **Performance Highlights**: Retro-li는 기존의 대규모 데이터베이스 대신 수백만 개의 토큰을 갖는 소규모 데이터베이스에서도 언어 모델링 성능 향상을 보여주며, 도메인 변화에 대한 일반화 능력이 향상되었습니다. 노이즈가 존재하는 상태에서도 성능 저하가 1% 미만으로 유지되며, 사용자는 특정 도메인에 맞춘 새로운 데이터베이스를 쉽게 구축할 수 있습니다.



### Conversational Exploratory Search of Scholarly Publications Using Knowledge Graphs (https://arxiv.org/abs/2410.00427)
Comments:
          Accepted to ICNLSP 2024

- **What's New**: 이 논문에서는 기존의 그래픽 검색 인터페이스의 복잡성과 데이터 양으로 인해 학술 출판물 탐색이 어려운 문제를 해결하기 위해 대화형 검색 시스템을 개발하였습니다. 이 시스템은 지식 그래프(KG)를 활용하여 사용자가 자연어로 대화하며 연구 논문을 효과적으로 발견할 수 있도록 지원합니다.

- **Technical Details**: 대화형 검색 시스템은 사용자에게 3단계 검색 프로세스를 통해 관련 논문을 좁혀주며, 각각의 단계에서 사용자의 검색 목표에 적합한 연구 주제를 추천합니다. 시스템의 아키텍처는 대화형 인터페이스, 대화 관리 기능, KG 검색 기능 등으로 구성되어 있으며, RASA 및 Neo4j를 활용하여 대화 및 데이터 검색을 구현합니다.

- **Performance Highlights**: 40명의 참가자를 대상으로 한 인간 평가 결과, 대화형 인터페이스가 전통적인 텍스트 기반 검색 방식에 비해 더 나은 정보 탐색 경험을 제공함을 입증했습니다. 논문의 평가 결과는 대화형 검색 시스템 설계를 발전시키는데 있어 실질적인 인사이트를 제공합니다.



### TPN: Transferable Proto-Learning Network towards Few-shot Document-Level Relation Extraction (https://arxiv.org/abs/2410.00412)
Comments:
          Few shot document-level relation extraction

- **What's New**: 본 연구는 문서 수준 관계 추출(few-shot document-level relation extraction; FSDLRE)의 효율성을 개선하기 위해 TPN(Transferable Proto-Learning Network)을 제안합니다. 이 네트워크는 Hybrid Encoder, Transferable Proto-Learner, Dynamic Weighting Calibrator의 세 가지 주요 구성 요소로 이루어져 있습니다.

- **Technical Details**: TPN은 NOTA(None-Of-The-Above) 관계의 교차 분야 전이 가능성을 개선하기 위해 설계되었습니다. Hybrid Encoder는 입력 텍스트의 의미적 내용을 계층적으로 인코딩하여 관계 표현을 증진시키고, Transferable Proto-Learner는 적응 가능한 블록을 통해 NOTA 프로토타입을 계산하여 다양한 도메인 간의 NOTA 편향을 완화합니다. Dynamic Weighting Calibrator는 관계 특이적 분류 신뢰도를 감지하여 동적 가중치로 NOTA 중심 손실 함수를 보정합니다.

- **Performance Highlights**: FREDo와 ReFREDo 데이터셋에서 TPN의 우수성을 입증하는 실험 분석을 수행했습니다. 최신 방법들과 비교했을 때, TPN은 약 절반의 파라미터 크기(123MB vs. 221MB)로 경쟁력 있는 성능을 달성했습니다.



### AutoPureData: Automated Filtering of Web Data for LLM Fine-tuning (https://arxiv.org/abs/2406.19271)
Comments:
          Initial version

- **What's New**: 이 연구는 웹 데이터를 수집하고 불필요한 텍스트를 자동으로 필터링하는 시스템을 제안합니다. 기존의 신뢰할 수 있는 AI 모델을 이용하여 데이터 품질과 안전성을 확보하면서, AI 모델의 최신 정보를 반영하기 위한 방법입니다.

- **Technical Details**: 제안된 시스템은 FineWeb 데이터셋을 활용하여 웹에서 수집한 데이터를 필터링합니다. LlamaGuard 2와 Llama 3라는 두 가지 LLM을 사용하여 데이터를 플래그하고, 불법 콘텐츠와 Bias를 식별하는 데 주력합니다. 모델 성능은 F-1 점수 91.5%와 4%의 잘못된 긍정률로 확인되었습니다.

- **Performance Highlights**: 실험 결과, 100개의 웹 데이터 샘플 중 32개의 행이 불필요한 텍스트로 플래그되었습니다. 이 시스템은 데이터 품질 향상과 함께 데이터 수집 및 전처리의 시간과 비용을 크게 줄이는 데 기여합니다.



### RecSys Challenge 2024: Balancing Accuracy and Editorial Values in News Recommendations (https://arxiv.org/abs/2409.20483)
Comments:
          5 pages, 3 tables, RecSys' 24

- **What's New**: RecSys Challenge 2024는 뉴스 추천의 기술적 및 규범적 과제를 다루며, 사용자 선호를 행동 기반으로 모델링하고 뉴스 항목의 빠른 소멸을 관리하는 데 중점을 두고 있습니다.

- **Technical Details**: Ekstra Bladet와 JP/Politikens Media Group은 110만 이상의 사용자와 125,000개의 뉴스 기사를 포함하는 대규모 데이터셋을 제공하며, 다양한 메트릭(AUC, MRR, nDCG)을 사용하여 추천 시스템을 평가합니다.

- **Performance Highlights**: 참가자들은 사용자의 클릭 기록, 세션 세부사항, 사용자 메타데이터를 바탕으로 뉴스 기사를 기초로 순위 매기기를 수행하며, 이번 대회는 다양한 추천 시스템의 뉴스 흐름에 대한 영향을 평가하는 데 중점을 두고 있습니다.



### Mixed-Precision Embeddings for Large-Scale Recommendation Models (https://arxiv.org/abs/2409.20305)
Comments:
          under submision

- **What's New**: 이 논문에서는 embedding 테이블을 압축하기 위한 새로운 방법인 Mixed-Precision Embeddings (MPE)를 제안합니다. 모델 정확도와 메모리 사용의 균형을 유지하기 위해 각 기능(feature)의 중요도에 따른 적절한 정밀도를 식별할 수 있도록 하는 것이 목표입니다.

- **Technical Details**: MPE는 기능을 빈도에 따라 그룹화하고, 각 기능 그룹에 대해 최적의 정밀도를 탐색하는 방식을 취합니다. 또한 각 기능 그룹에서 정밀도 수준에 대한 확률 분포를 학습하여, 맞춤형 샘플링 전략을 통해 최적의 정밀도를 식별할 수 있습니다. MPE는 기존의 embedding 압축 방법들과 비교하여 단적으로 교차적인 실험을 통해 우수한 성능을 보였습니다.

- **Performance Highlights**: 실험 결과, MPE는 Criteo 데이터셋에서 예측 정확도를 손상시키지 않으면서 약 200배의 압축을 달성했습니다. 이 논문은 추천 시스템에서 embedding의 효율적인 관리를 위한 새로운 길을 제시합니다.



### Neural Click Models for Recommender Systems (https://arxiv.org/abs/2409.20055)
- **What's New**: 이번 연구에서는 추천 시스템(Recommendation System, RS)에서의 사용자 행동을 모델링하기 위한 신경망 아키텍처를 개발하고 평가하였습니다. 새로운 아키텍처로는 RNN, Transformer 기반 모델과 더불어 적대적(Adversarial) 및 계층적(Hierarchical) 아키텍처가 포함됩니다.

- **Technical Details**: 이 논문은 추천 시스템에서의 사용자 응답을 모델링하기 위해 RNN 및 Transformer 기반 아키텍처를 포함한 다양한 신경망 아키텍처를 제안합니다. 주요 실험에서 사용된 데이터셋은 ContentWise Impressions와 RL4RS로, 추천된 항목의 모임을 유지하고 사용할 수 있는 정보가 포함되어 있습니다. 모델은 사용자 세션의 이전 상호작용 이력을 고려하여 추천된 항목에 대한 반응을 예측합니다.

- **Performance Highlights**: 제안된 아키텍처는 ContentWise 및 RL4RS 데이터셋에서 기존의 기준 모델(Baseline)보다 우수한 성능을 보였으며, 새로운 추천 시스템 시뮬레이터의 기초로 활용될 수 있습니다.



### Mitigating Propensity Bias of Large Language Models for Recommender Systems (https://arxiv.org/abs/2409.20052)
- **What's New**: 새로운 프레임워크인 Counterfactual LLM Recommendation (CLLMR)을 소개합니다. 이 프레임워크는 Large Language Models (LLMs)에서 생성된 부가 정보(s)이 사용자와 아이템의 역사적 상호작용 구조 정보를 통합하여 차원 붕괴(dimensional collapse)의 위험을 회피하는 방법을 제안합니다.

- **Technical Details**: Spectrum-based Side information Encoder (SSE)를 통해 역사적 상호작용에서 구조적 정보를 부가 정보의 표현에 암묵적으로 내재화 합니다. 이를 통해 LLM에서 도출된 부가 정보와 사용자 상호작용으로부터 학습한 협력적 표현을 정렬하는 과정에서 발생할 수 있는 차원 붕괴를 방지합니다. 또한, CLLMR 접근법은 LLM 기반 추천 시스템의 인과 관계를 탐색하고, 카운터팩추얼 추론(counterfactual inference)을 활용하여 LLM이 초래하는 편향(bias)을 교정합니다.

- **Performance Highlights**: 실험 결과, CLLMR 접근법은 여러 추천 모델의 성능을 일관되게 향상시켜주었으며, 세 가지 실제 데이터셋에서 최첨단 LLM 추천 정렬 방법들과 비교하여 효과성을 입증했습니다.



### Enhancing High-order Interaction Awareness in LLM-based Recommender Mod (https://arxiv.org/abs/2409.19979)
Comments:
          Long paper accepted to EMNLP 2024 Main. 16 pages

- **What's New**: 이번 논문에서는 LLM(대형 언어 모델)을 기반으로 한 추천 시스템 ELMRec(Enhanced LLM-based Recommender)를 제안합니다. 기존 추천 방법이 사용자-아이템의 고차 상호작용을 효과적으로 모델링하지 못하는 문제를 해결하기 위해, 새로운 whole-word embeddings를 도입하여 LLM의 추천 해석 능력을 향상시킵니다. 이 방법은 그래프 전처리 없이도 고차 상호작용 신호를 LLM이 쉽게 흡수할 수 있도록 합니다.

- **Technical Details**: ELMRec는 random feature propagation을 사용하는 novel whole-word embeddings를 개발하여 LLM이 추천에서의 고차 상호작용 인식을 크게 증가시킵니다. 이로 인해 각 ID는 해당 토큰과 연결된 whole-word embeddings로 표현되어, 사용자와 아이템 사이의 상대적 위치를 더 잘 포착합니다. 또한, 기계 학습 모델이 이전 상호작용에 기반하여 아이템을 추천하는 경향이 있음을 인지하고, 이 문제를 해결하기 위한 reranking 방법을 제안합니다.

- **Performance Highlights**: ELMRec는 직접 추천과 순차적 추천 모두에서 최신 기술(SOTA)보다 높은 성능을 기록했습니다. 실험 결과 ELMRec가 LLM 기반 추천 시스템의 뛰어난 성능을 보여주었다는 것을 입증합니다.



### Large Language Model Empowered Embedding Generator for Sequential Recommendation (https://arxiv.org/abs/2409.19925)
- **What's New**: Sequential Recommender Systems (SRS)에 대한 새로운 접근법으로, LLMEmb라는 혁신적인 기법이 제안되었습니다. 이 기법은 대형 언어 모델(LLM)을 활용하여 아이템 임베딩(item embeddings)을 생성하여 SRS의 성능을 높입니다.

- **Technical Details**: LLM의 일반적인 능력을 추천 도메인에 맞추기 위해 Supervised Contrastive Fine-Tuning (SCFT) 방법을 도입했습니다. SCFT는 속성 수준의 데이터 증대(attribute-level data augmentation)와 추천 성능 강화를 위한 맞춤형 대조 손실(custom contrastive loss)을 포함합니다. 또한, LLM 생성 임베딩에 협업 필터링 신호(collaborative filtering signals)를 통합할 필요성을 강조하고, 이를 위한 Recommendation Adaptation Training (RAT)을 제안합니다.

- **Performance Highlights**: 세 가지 실제 데이터 세트에 대한 폭넓은 실험을 통해, LLMEmb가 다양한 SRS 모델에서 현재 방법보다 상당한 성능 향상을 보여주었습니다.



### Counterfactual Evaluation of Ads Ranking Models through Domain Adaptation (https://arxiv.org/abs/2409.19824)
Comments:
          Accepted at the CONSEQUENCES'24 workshop, co-located with ACM RecSys'24

- **What's New**: 이번 논문에서는 Offline A/B testing 시스템과 함께 작동하는 도메인 적응형 보상 모델(domain-adapted reward model)을 제안합니다.

- **Technical Details**: 이 보상 모델은 대규모 Ads 추천 시스템에서 랭킹 모델(ranking model) 변경에 대한 보상을 효과적으로 측정합니다. 기존의 모델 없는 방법인 IPS가 적용될 수 없는 환경에서도 사용이 가능합니다.

- **Performance Highlights**: 실험 결과, 제안된 기술이 기존의 vanilla IPS 방법 및 비일반화 보상 모델(non-generalized reward models) 접근 방식을 모두 초과하는 성능을 보였습니다.



### The Devil is in the Sources! Knowledge Enhanced Cross-Domain Recommendation in an Information Bottleneck Perspectiv (https://arxiv.org/abs/2409.19574)
Comments:
          Accepted by CIKM 2024

- **What's New**: Cross-domain Recommendation (CDR)에서의 기존 모델들이 정보의 유용성을 무시한 채 전체 정보를 동등하게 활용하던 문제를 해결하기 위해, CoTrans라는 새로운 프레임워크를 제안합니다.

- **Technical Details**: CoTrans는 정보 압축(Compression)과 순수성 전달(Transfer) 두 가지 핵심 과정을 통해 CDR 모델의 구조를 개선합니다. 사용자 행동을 타겟 도메인 관점에서 압축하고, 피드백 신호를 활용하여 전달 과정을 효과적으로 수행합니다. 이를 통해 노이즈에 의한 비관련 정보를 제거하고 중요한 정보를 강화합니다. 또한, 지식 그래프(Knowledge Graph)를 사용해 서로 다른 도메인 간의 간극을 메웁니다.

- **Performance Highlights**: 세 가지 대규모 CDR 데이터셋에 대한 실험 결과, CoTrans는 단일 도메인 및 최신 CDR 방법들과 비교하여 현저한 성능 향상을 보여주었습니다.



### Meta Learning to Rank for Sparsely Supervised Queries (https://arxiv.org/abs/2409.19548)
Comments:
          Accepted at TOIS

- **What's New**: 본 연구에서는 sparsely supervised queries에 대한 문제를 해결하기 위해 새로운 메타 학습 기반의 ranking 프레임워크를 제안합니다. 이 접근법은 각 쿼리에 대해 최적의 파라미터를 학습하여 기존의 글로벌 모델의 한계를 극복합니다.

- **Technical Details**: 제안된 메타 학습 접근법은 각 쿼리의 메타 학습 과정에서 학습 세트와 테스트 세트를 두 가지로 나누어, 로컬 업데이트와 글로벌 업데이트를 수행합니다. 이는 각 쿼리마다 몇 개의 레이블된 예시만으로도 fine-tuning이 가능하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 메타 학습 접근법은 sparsely labeled queries에 대해 기존의 learning to rank 모델보다 성능이 크게 향상되는 것으로 나타났습니다.



### HTML-LSTM: Information Extraction from HTML Tables in Web Pages using Tree-Structured LSTM (https://arxiv.org/abs/2409.19445)
- **What's New**: 이 논문에서는 구조가 다른 HTML 테이블에서 유사한 내용을 추출하는 새로운 방법, HTML-LSTM을 제안합니다. 이 방법은 다양한 웹 페이지의 HTML 테이블을 통합하여 정보를 검색할 수 있도록 설계되었습니다.

- **Technical Details**: HTML-LSTM은 tree-structured LSTM(Tree-LSTM)을 확장하여 HTML 데이터의 언어적 및 구조적 정보를 동시에 추출하는 데 중점을 둡니다. 이 방법은 먼저 HTML 데이터를 테이블 구조로 변환한 후, 하위 구조를 DOM 트리 형태로 변환하여 노드의 특성을 분류하고 새로운 테이블로 통합합니다. 이를 위해 이중 방향 LSTM(Bi-LSTM)을 사용하여 각 요소의 언어 표현을 얻고, HTML-LSTM을 통해 트리 구조에서의 요소 간 관계를 고려하여 특징을 추출합니다.

- **Performance Highlights**: 실험 결과, 유치원 데이터에서 F1 점수 0.96, 대학교 강의 계획서 데이터에서 F1 점수 0.86을 달성하여 HTML-LSTM의 유효성을 입증했습니다. 또한, HTML-LSTM은 Tree-LSTM보다 더 우수한 성능을 보였습니다.



### Utilizing Collaborative Filtering in a Personalized Research-Paper Recommendation System (https://arxiv.org/abs/2409.19267)
Comments:
          5 pages, 4 figures

- **What's New**: 본 논문에서는 협업 필터링(Collaborative Filtering) 기반의 연구 논문 추천 시스템을 제안하였습니다. Jaccard 유사도(Jaccard Similarity)를 활용하여 공저자(coauthor), 키워드(keyword), 참조(reference), 공통 인용(common citation) 기반의 유사성을 계산함으로써 최종 유사도를 도출하고, 이를 통해 가장 유사한 상위 N명의 사용자들로부터 특정 연구 논문을 추천하는 시스템을 개발하였습니다.

- **Technical Details**: 제안된 시스템은 사용자 기반의 협업 필터링을 적용하며, 각 사용자의 출판물을 바탕으로 추천을 생성합니다. 유사 사용자가 과거에 인용한 자료들을 고려하여 연구 논문을 추천하며, 키워드, 공저자, 참조, 공통 인용의 유사성을 Jaccard 유사도를 통해 계산합니다. 각 유사성에는 가중치가 부여되어 최종 유사성 매트릭스(final similarity matrix)가 도출되고, 이를 통해 상위 N명의 유사 사용자를 선정하여 추천을 실행합니다.

- **Performance Highlights**: 실험 결과, 본 시스템은 정밀도(precision), 재현율(recall), F-측정(F-measure) 측면에서 우수한 성능을 보였습니다. 데이터셋은 테스트 세트와 훈련 세트로 나뉘어 사용되었으며, 유사성 기반 계산을 통해 추천 성능이 크게 향상됨을 확인했습니다.



### An Efficient Multi-threaded Collaborative Filtering Approach in Recommendation System (https://arxiv.org/abs/2409.19262)
Comments:
          6 Pages 6 Figure, Paper got accepted at the 2nd International Conference on Artificial Intelligence, Blockchain, and Internet of Things, (AIBThings)

- **What's New**: 본 연구는 추천 시스템(Recommendation Systems, RS) 분야를 발전시키기 위해 새로운 방법론과 통찰력을 도입합니다. 최근 몇 년 동안 정보 접근 및 상호작용 방식의 혁신을 가져온 추천 시스템의 원리를 활용하여, 유사성(similarity)의 개념을 중심으로 한 효율적인 구현 기술을 탐구하고 있습니다.

- **Technical Details**: 추천 시스템은 크게 Content-Based Filtering와 Collaborative Filtering으로 나눌 수 있습니다. 본 연구는 특히 Collaborative Recommendation Systems(CRS)에 중점을 두어, 사용자 리뷰를 기반으로 사용자 평점을 예측하는 방법론을 제안합니다. 연구는 멀티스레딩(multi-threading) 기법을 활용하여 사용자 데이터를 독립 스레드로 나누어 병렬 처리함으로써 계산 시간을 대폭 단축합니다.

- **Performance Highlights**: 멀티스레딩 기법을 사용하여 전통적인 방식에 비해 계산 시간을 5배에서 7배 감소시키는 성과를 달성하였으며, 추천 시스템의 스케일러블한 구현을 통해 사용자 데이터의 보안성을 유지하면서도 성능 향상을 보장합니다.



### TwinCL: A Twin Graph Contrastive Learning Model for Collaborative Filtering (https://arxiv.org/abs/2409.19169)
- **What's New**: 본 논문에서는 기존의 Graph Contrastive Learning (GCL) 방법론의 제한점을 지적하고, 랜덤한 데이터 증강 방법이 구조적 및 의미적 정보를 어떻게 왜곡할 수 있는지를 분석했습니다. 새로운 트윈 인코더를 통해 초기 학습 단계에서는 다양한 contrastive views를 생성하고, 후반 단계에서는 유사한 views로 전환하여 효과적인 학습을 도모합니다.

- **Technical Details**: 제안된 모델인 TwinCL은 모멘텀 업데이트 방식의 트윈 인코더를 활용하여 사용자와 아이템 간의 positive pair 임베딩을 정렬하고, hypersphere에서 임베딩의 균일성을 유지합니다. 학습 초기 단계에서는 큰 교란을, 점차적으로 수렴하는 과정에서는 더 부드러운 교란을 적용하여 최적화를 진행합니다.

- **Performance Highlights**: 세 가지 공개 데이터셋에 대한 포괄적인 실험 결과, TwinCL은 추천 정확도(NDCG@10)에서 평균 5.6% 향상을 보였으며, 빠른 학습 속도와 함께 인기 편향(popularity bias)을 효과적으로 완화하는 데 성공했습니다.



### TTT4Rec: A Test-Time Training Approach for Rapid Adaption in Sequential Recommendation (https://arxiv.org/abs/2409.19142)
- **What's New**: 본 논문에서는 사용자 상호작용을 실시간으로 반영하여 동적으로 모델을 업데이트할 수 있는 Test-Time Training (TTT) 기반의 순차 추천 프레임워크인 TTT4Rec을 제안합니다.

- **Technical Details**: TTT4Rec은 TTT의 구조를 통해 두 개의 루프를 사용하여 모델 파라미터를 지속적으로 업데이트합니다. 외부 루프는 감독 학습(supervised learning)에 집중하고, 내부 루프는 자가 감독 학습(self-supervised learning)을 이용하여 훈련 및 추론 중 모델의 hidden state를 업데이트합니다. 주요 구성 요소로는 아이템 정보를 높은 차원 벡터로 인코딩하는 Embedding Layer, 입력 시퀀스의 특징을 캡처하는 TTT 기반의 Residual Blocks, 업데이트된 hidden state를 바탕으로 추천을 생성하는 Prediction Layer가 있습니다.

- **Performance Highlights**: TTT4Rec은 세 가지 주요 추천 데이터셋에서 평가되었으며, 기존의 최첨단 모델보다 더 나은 성능을 나타냈습니다. 특히 훈련 데이터가 제한되거나 사용자 행동이 급변하는 경우 효율적으로 동작하며, 실시간 사용자 상호작용에 적응하여 정확한 추천이 가능함을 보였습니다.



### Integrating SPARQL and LLMs for Question Answering over Scholarly Data Sources (https://arxiv.org/abs/2409.18969)
Comments:
          Scholaly Hybrid Question answering challenge from the International Semantic Web Conference of 2024(ISWC), 6 pages, 3 figures

- **What's New**: 이번 논문은 2024년 국제 시맨틱 웹 컨퍼런스(ISWC)에서 주최되는 Scholarly Hybrid Question Answering over Linked Data (QALD) 챌린지에 대한 새로운 접근 방식을 소개합니다. SPARQL 쿼리, divide and conquer 알고리즘, 그리고 BERT 기반 모델을 통합하여 다양한 학술 출처에 대한 질문에 답변하는 시스템을 개발했습니다.

- **Technical Details**: 제안된 방법론은 SPARQL 쿼리를 통해 데이터를 수집하고, divide and conquer 알고리즘을 적용하여 다양한 질문 유형 및 출처를 관리하며, BERT를 이용하여 저자 관련 질문에 대한 정확한 답변을 생성하는 방식으로 구성되어 있습니다. 이 과정에서 SPARQL 쿼리 실행, 데이터를 정리하고, 질문을 정교하게 삭감하는 과정을 포함합니다. 최종적으로는 LLM(대형 언어 모델)을 사용하여 예측을 수행합니다.

- **Performance Highlights**: 제안한 방법은 Exact Match 및 F-score 메트릭을 사용하여 평가되었으며, 다양한 학술 데이터 출처에서 정확한 질문 응답을 제공하는 데 큰 개선을 보였습니다. 특히, BERT와 DPR(Dual-Product Retrieval) 알고리즘의 조합이 DBLP 지식 그래프에서 엔터티 및 관계 추출의 정확도를 크게 향상시켰습니다.



### OM4OV: Leveraging Ontology Matching for Ontology Versioning (https://arxiv.org/abs/2409.20302)
Comments:
          7 pages, 7 figures, 1 table

- **What's New**: 본 논문에서는 기존의 ontology matching (OM) 기술을 활용하여 ontology version control (OV)을 위한 새로운 접근 방식을 제안합니다. OM과 OV 간의 상보적 관계를 탐구하고, 두 작업을 동일한 파이프라인으로 처리할 수 있는 방법을 제시합니다.

- **Technical Details**: 논문에서 제안하는 OM4OV 파이프라인은 신규 작업 정의, 성과 측정 및 검증용 데이터셋 구축을 통해 OV 작업에 OM을 효과적으로 활용합니다. 특히, cross-reference 메커니즘을 도입하여 OV 후보 선택을 최적화하고 OM의 전반적인 성능을 향상시킵니다.

- **Performance Highlights**: OAEI 데이터셋을 사용하여 OM4OV 파이프라인과 cross-reference 메커니즘의 성능을 실험적으로 검증하였습니다. 이 연구는 기계 생성된 버전 정보를 활용한 경량화 및 완전 자동화된 OV 접근 방식을 제시하여, 기존 OM 시스템과 기술을 OV 작업으로 이전할 수 있는 새로운 가능성을 열었습니다.



### ASTRA: Accurate and Scalable ANNS-based Training of Extreme Classifiers (https://arxiv.org/abs/2409.20156)
- **What's New**: 최신 XC 알고리즘인 ASTRA는 높은 정확도를 유지하면서도 수억 개의 레이블에 대해 스케일 가능성을 제공합니다. 이 알고리즘은 ANNS 기반의 훈련 방법론을 개발하였으며, 이는 기존 방법들에 비해 훈련 시간을 최대 15배까지 줄일 수 있습니다.

- **Technical Details**: ASTRA 알고리즘은 두 가지 주요 관점을 기반으로 구축되었습니다: (a) 분류기 벡터에 대한 ANNS 인덱스를 구축하고 이를 통해 hard negatives를 검색하여 손실 함수에 최적화된 negative sampling 전략을 구현합니다; (b) 분류기가 epochs를 통과하면서 ANNS 인덱스를 지속적으로 업데이트하는 것이 매우 비쌉니다. 따라서, 개선된 경량의 negative sampling 전략으로 주목할 만한 성능을 거두었습니다.

- **Performance Highlights**: ASTRA는 120M 레이블과 370M 쿼리를 포함하는 대규모 데이터셋에서 83.4의 Precision@1을 기록하며, 같은 하드웨어를 사용한 Renée보다 훈련 시간이 15배 더 짧습니다. 또한, 다른 XC 알고리즘과 비교하여 속도 면에서도 4.5배에서 최대 80.4배 빠른 성과를 보여주었습니다.



### Crafting Personalized Agents through Retrieval-Augmented Generation on Editable Memory Graphs (https://arxiv.org/abs/2409.19401)
Comments:
          This paper has been accepted by EMNLP 2024

- **What's New**: 모바일 인터넷 시대에 사용자의 불규칙한 데이터를 효과적으로 관리하고 활용하여 개인화된 AI 어시스턴트를 만드는 새로운 작업이 소개되었습니다. 본 논문에서는 스마트폰 기억(memoies)을 활용하여 LLM(대형 언어 모델) 기능을 향상시키는 EMG-RAG라는 솔루션을 제안합니다.

- **Technical Details**: EMG-RAG는 Retrieval-Augmented Generation (RAG) 기술과 Editable Memory Graph (EMG)를 결합하여 개인화된 에이전트를 만드는 것을 목표로 합니다. 이 접근법은 강화 학습을 사용하여 데이터 수집, 편집 가능성, 선택 가능성의 세 가지 주요 문제를 해결합니다. EMG는 메모리의 복잡한 관계를 포착하기 위해 트리 구조를 가지고 있으며, 사용자 메모리를 효율적으로 편집하고 검색할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실제 데이터셋을 바탕으로 한 실험에서 EMG-RAG는 기존 접근 방식보다 약 10% 향상된 성능을 기록했습니다. 또한, 향상된 개인화된 에이전트는 실제 스마트폰 AI 어시스턴트에 통합되어 사용자 경험이 개선되었습니다.



### FLEX: Expert-level False-Less EXecution Metric for Reliable Text-to-SQL Benchmark (https://arxiv.org/abs/2409.19014)
Comments:
          preprint, under review

- **What's New**: 본 논문에서는 FLEX (False-Less EXecution)라는 새로운 평가 방식을 소개하여, SQL 쿼리의 인간 전문가 수준의 평가를 모방하는 대형 언어 모델(LLM)을 활용하여 텍스트-투-SQL 시스템을 평가한다.

- **Technical Details**: FLEX는 기존의 Execution Accuracy (EX) 방식의 한계를 극복하고, 더 정밀한 평가를 제공하며, SQL 쿼리가 원래 질문과 의미적으로 일치하는지를 분석하여 종합적인 쿼리 정확성을 평가한다. 이는 노이즈가 있는 기준 데이터에 대해서도 유용하다.

- **Performance Highlights**: FLEX를 이용한 평가 결과, 기존의 EX 평가보다 인간 전문가의 판단과의 일치도가 크게 향상되었고, Cohen's kappa는 61에서 78.17로 증가했다. 스파이더(Spider)와 BIRD 벤치마크에서 기존 상위 모델의 성능 순위가 대폭 변경되는 결과가 나타났다.



### Lab-AI -- Retrieval-Augmented Language Model for Personalized Lab Test Interpretation in Clinical Medicin (https://arxiv.org/abs/2409.18986)
- **What's New**: Lab-AI는 환자 맞춤형 정상 범위를 제공하는 상호작용 시스템으로, Retrieval-Augmented Generation (RAG) 기술을 활용하여 신뢰할 수 있는 건강 정보 소스로부터 정보를 검색합니다.

- **Technical Details**: Lab-AI는 두 가지 모듈, 즉 factor retrieval 및 normal range retrieval로 구성되어 있으며, 68개의 실험실 테스트에서 30개는 조건적 요소가 포함되고 38개는 포함되지 않습니다. 테스트의 정상 범위는 환자-specific information에 따라 달라집니다. GPT-4-turbo 모델은 factor retrieval에서 0.95의 F1 score, normal range retrieval에서 0.993의 정확도를 보였습니다.

- **Performance Highlights**: RAG를 사용하는 GPT-4-turbo는 비-RAG 시스템보다 29.1% 더 높은 factor retrieval 성능을 나타내었고, normal range retrieval에서 질문 수준에서 60.9%, 실험실 수준에서 52.9% 향상을 보였습니다. 이러한 결과는 Lab-AI가 환자가 실험실 결과를 이해하는 데 도움을 줄 수 있는 잠재력을 강조합니다.



New uploads on arXiv(cs.CV)

### Dual Consolidation for Pre-Trained Model-Based Domain-Incremental Learning (https://arxiv.org/abs/2410.00911)
- **What's New**: 이 논문에서는 Domain-Incremental Learning (DIL)에서의 지식 소멸 문제를 해결하기 위해 DUal ConsolidaTion (Duct) 방법을 제안합니다. 이는 모델의 표현(representation) 및 분류기(classifier) 차원에서 역사적 지식을 통합하는 방식으로, 이전 도메인의 지식을 보존하면서 새로운 도메인에 적응할 수 있도록 합니다.

- **Technical Details**: DUct 방법은 다양한 도메인에서 축적된 정보를 통합하기 위해 서로 다른 단계의 백본(backbone)을 병합합니다. 이로 인해 생성된 표현 공간은 모든 도메인에 적합한 정보의 균형 잡힌 중간체를 제공합니다. 또한, 분류기 통합(classifier consolidation) 과정을 통해 최신 임베딩 공간(embedding space)으로의 적격한 분류기 가중치 조정을 수행합니다. 이 과정에서 클래스별 의미 정보(class-wise semantic information)를 활용하여 이전 도메인 클래스의 분류기 가중치를 추정합니다.

- **Performance Highlights**: Duct 방법은 네 가지 벤치마크 데이터 세트에서 수행된 광범위한 실험 결과를 통해 최첨단 성능을 달성하였으며, 이전 지식을 잃지 않고 새로운 지식을 효과적으로 학습할 수 있음을 입증하였습니다.



### Removing Distributional Discrepancies in Captions Improves Image-Text Alignmen (https://arxiv.org/abs/2410.00905)
- **What's New**: 본 논문에서는 이미지-텍스트 정렬 예측을 개선하기 위한 모델을 소개하며, 현재의 비주얼-언어 모델에서 조합적 이해(compositional understanding)의 문제를 해결하는 데 중점을 두고 있습니다.

- **Technical Details**: 모델은 이미지-텍스트 정렬 작업을 위해 높은 품질의 훈련 데이터를 생성하는데 초점을 두며, 긍정적인 캡션에서 파생된 혼합형 부정 캡션을 생성합니다. 이는 이미지와 텍스트 간의 불균형을 해결하면서, 모델이 텍스트 정보뿐 아니라 관련 이미지를 고려하여 정렬 예측을 정확하게 수행하도록 합니다. LLaVA와 같은 최첨단 비주얼-언어 모델을 사용하여 모델을 미세 조정하며, 기존 최고 성능 모델들과 비교하여 상당한 성과를 거두었습니다.

- **Performance Highlights**: 제안된 모델은 여러 데이터셋에서 현재 최고 성과를 기록하는 방법들보다 우수한 성능을 보이며, 이미지 생성 및 정렬 작업의 효율성을 높입니다.



### OSSA: Unsupervised One-Shot Style Adaptation (https://arxiv.org/abs/2410.00900)
- **What's New**: 이 논문에서는 One-Shot Style Adaptation (OSSA)이라는 새로운 비지도 도메인 적응 방법을 소개하며, 단일 레이블 없는 대상 이미지를 사용하여 목표 도메인 스타일을 근사하는 기법을 제안합니다. OSSA는 이미지 스타일의 변형을 통해 다양한 대상 스타일을 생성하고, 이를 Adaptive Instance Normalization (AdaIN)을 활용해 레이블이 있는 소스 데이터 세트에 적용합니다.

- **Technical Details**: OSSA는 단일 대상 이미지를 기반으로 스타일 통계치를 조작하여 다양한 스타일을 생성하는 기법입니다. 이 방법은 CNN의 초기 레이어에서 스타일 정보를 인코딩하는 특성을 활용하여, 소스 도메인 이미지의 스타일을 레이블이 없는 대상 도메인 이미지에 적용합니다. 또한, OSSA는 기존의 복잡한 생성 모델 대신에 효율적인 피쳐 맵 수준에서 스타일 적응을 수행합니다.

- **Performance Highlights**: OSSA는 One-Shot 도메인 적응 방법 중에서 새로운 최첨단 성능을 달성했으며, 경우에 따라 수천 개의 레이블 없는 대상 이미지를 사용하는 강력한 기준선을 초월하는 성과를 보였습니다. 다양한 시나리오(예: 기후, 시뮬레이션-실제 전이, 시각-열 적응)에서 OSSA를 적용하여 스타일 갭의 중요성을 탐구하였습니다.



### Flex3D: Feed-Forward 3D Generation With Flexible Reconstruction Model And Input View Curation (https://arxiv.org/abs/2410.00890)
Comments:
          Project page: this https URL

- **What's New**: 이번 논문에서는 Flex3D라는 새로운 두 단계 프레임워크를 도입하여, 고품질 3D 콘텐츠를 생성하는 혁신적인 접근 방식을 제안합니다. 이 프레임워크는 다양한 고품질 입력 뷰를 활용할 수 있고, 후보 뷰를 생성 및 선별하여 최종 3D 재구성을 개선합니다.

- **Technical Details**: Flex3D는 후보 뷰 생성 및 선별 파이프라인과 Flexible Reconstruction Model (FlexRM)으로 구성됩니다. 첫 번째 단계는 다중 뷰 확산 모델(multi-view diffusion model)과 비디오 확산 모델(video diffusion model)을 통해 후보 뷰 풀을 생성합니다. 두 번째 단계인 FlexRM은 트랜스포머 아키텍처를 기반으로 하여 임의의 수의 입력을 효과적으로 처리하고, 3D Gaussian 포인트를 출력합니다. 또한, 노이즈를 추가하여 불완전한 입력을 학습하는 새로운 훈련 전략을 도입합니다.

- **Performance Highlights**: Flex3D는 3D 생성 작업에서 최신 피드포워드 3D 생성 모델들과 비교하여 92% 이상의 사용자 연구 승리율을 기록하며, 재구성 및 생성 작업 모두에서 최상의 성능을 달성했습니다.



### MAP: Unleashing Hybrid Mamba-Transformer Vision Backbone's Potential with Masked Autoregressive Pretraining (https://arxiv.org/abs/2410.00871)
- **What's New**: 본 논문에서는 Mamba와 Transformer 아키텍처를 결합한 하이브리드 비전 백본 네트워크를 위한 새로운 사전 훈련 메서드인 Masked Autoregressive Pretraining(MAP)을 제안합니다. 이 방법은 Mamba 및 Transformer 모듈의 성능을 통합된 패러다임 내에서 크게 향상시킵니다.

- **Technical Details**: MAP는 지역별 MAE(Local Masked Autoencoder)를 통해 Transformer 블록의 지역적 주의를 학습하고, 전역적 오토회귀 사전 훈련을 통해 Mamba 블록이 의미 있는 문맥 정보를 학습할 수 있도록 설계되었습니다. 하이브리드 구조에서 Mamba 레이어 사이에 Transformer 레이어를 정기적으로 삽입함으로써 후속 작업 성능을 크게 향상시킬 수 있음을 발견했습니다.

- **Performance Highlights**: MAP으로 사전 훈련된 하이브리드 Mamba-Transformer 모델은 순수한 Mamba 아키텍처 및 다른 기존 사전 훈련 전략보다 뛰어난 성능을 보입니다. 특히 ImageNet-1K 분류 작업에서 탁월한 결과를 달성하였으며, 2D 및 3D 데이터셋에서도 유효성을 검증하였습니다.



### Squeeze-and-Remember Block (https://arxiv.org/abs/2410.00823)
Comments:
          Accepted by The International Conference on Machine Learning and Applications (ICMLA) 2024

- **What's New**: 본 논문에서는 Convolutional Neural Networks (CNNs)에 동적인 메모리 기능을 추가하는 새로운 구조적 단위를 소개합니다. 이 단위는 "Squeeze-and-Remember" (SR) 블록으로, 훈련 중 중요한 특징을 선택적으로 기억하고, 추론 시 이들 특징을 적응적으로 재적용하여 네트워크의 맥락적 예측 능력을 향상시킵니다.

- **Technical Details**: SR 블록은 입력 특성 맵에 대해 1×1 컨볼루션을 수행하여 중요한 정보를 추출하고, 이를 두 층의 Fully Connected Network (FCN)를 통해 P개의 메모리 블록과 연계하여 가중치를 생성합니다. 최종 출력은 원래 입력에 메모리 블록에서 얻은 높은 수준의 특징을 추가하여 생성됩니다. SR 블록은 ResNet50과 DeepLab v3 모델에 통합되어 성능 향상을 보여줍니다.

- **Performance Highlights**: SR 블록을 ResNet50에 통합한 결과, ImageNet 데이터셋에서 top-1 validation accuracy가 dropout2d 단독 사용 대비 0.52% 향상되었습니다. 또한, Cityscapes 데이터셋의 DeepLab v3 모델 적용 시 mean Intersection over Union이 0.20% 증가했습니다. 이 모든 개선은 최소한의 계산 오버헤드로 이루어졌습니다.



### WiGNet: Windowed Vision Graph Neural Network (https://arxiv.org/abs/2410.00807)
- **What's New**: WiGNet 모델은 비전 GNNs의 새로운 접근 방식을 통해 이미지 처리를 효율적으로 수행합니다. 기존 GNNs와 달리, 이미지를 비겹치는 윈도우로 분할하고 각 윈도우 내에서 그래프를 구성하여 계산 복잡성을 줄였습니다.

- **Technical Details**: WiGNet은 기존의 2D convolution 또는 self-attention 메커니즘 대신, 각 윈도우 내에서 그래프 합성곱(graph convolution)을 사용합니다. 이 방식은 메모리 및 계산 복잡성을 관리하며 이미지 크기에 대해 선형적으로 증가합니다.

- **Performance Highlights**: WiGNet는 ImageNet-1k 벤치마크 데이터셋에서 경쟁력 있는 결과를 달성하였고, CelebA-HQ 데이터셋에서는 높은 해상도의 이미지에서도 효과적인 성능을 보였습니다. 이는 이전의 Vision GNN보다 메모리와 계산 복잡성을 줄이면서도 뛰어난 성능을 유지함을 의미합니다.



### Local-to-Global Self-Supervised Representation Learning for Diabetic Retinopathy Grading (https://arxiv.org/abs/2410.00779)
- **What's New**: 이번 연구에서는 자가 지도 학습(self-supervised learning)과 지식 증류(knowledge distillation)를 결합한 새로운 하이브리드 학습 모델을 제안합니다. 이 모델은 Diabetic Retinopathy의 EyePACS 데이터셋에 적용되어 50% 더 큰 테스트 데이터셋을 사용하고, 기존의 방법들과 비교하였을 때보다 더 높은 정확도를 보여주었습니다.

- **Technical Details**: 제안된 모델은 ViT(Vision Transformer)에서 사용되는 self-attention 메커니즘과 토큰(token)을 활용하며, 지역에서 전역으로 학습(local-to-global learning) 접근 방식을 통해 이미지에서 고차원(high-dimensional) 및 고품질(high-quality) 피처 공간(feature space)을 추출할 수 있습니다. 연구는 자가 지도 학습과 지식 증류 기법을 결합하여 복잡한 구조의 의료 이미지를 다루는 데 중점을 두었습니다.

- **Performance Highlights**: 하이브리드 모델을 통해 멀티클래스 분류에서 79.1%의 정확도를 달성하였으며, k-NN 알고리즘에서는 74.36%의 정확도를 기록했습니다. 이는 유사한 최신(state-of-the-art) 모델과 비교할 때 더 높은 정확도를 나타냅니다.



### On the Generalization and Causal Explanation in Self-Supervised Learning (https://arxiv.org/abs/2410.00772)
- **What's New**: 이 논문에서는 Self-supervised learning (SSL) 방법들이 훈련 데이터에 과적합(overfitting) 되는 현상을 관찰하고, 이를 해결하기 위한 새로운 메커니즘인 Undoing Memorization Mechanism (UMM)을 제안합니다. UMM은 마지막 레이어의 피처 분포를 초기 레이어와 정렬하는 방식으로 과적합 문제를 완화합니다.

- **Technical Details**: 연구를 통해, SSL 모델이 훈련 초기 레이어에서 일반화 성능을 학습하고 마지막 레이어에서 기억화를 시작함을 발견하였습니다. Coding rate reduction을 활용하여 과적합 정도를 정량화할 수 있음을 보여주었으며, UMM은 이중 최적화 과정으로 설계되어 특징 추출기의 초기 레이어 특성과의 조화를 이루어 마지막 레이어 특성의 일반화를 회복합니다.

- **Performance Highlights**: UMM을 적용한 SSL 방법들이 다양한 다운스트림 작업에서 일반화 성능을 현저하게 개선하는 것을 실험을 통해 입증하였습니다. UMM은 빠르게 SSL 기술과 통합될 수 있는 플러그 앤 플레이 방식입니다.



### Empowering Large Language Model for Continual Video Question Answering with Collaborative Prompting (https://arxiv.org/abs/2410.00771)
Comments:
          Accepted by main EMNLP 2024

- **What's New**: 최근 온라인 비디오 콘텐츠의 급증으로 인해, 고정 데이터셋으로 훈련된 기존의 Video Question Answering (VideoQA) 모델이 새로운 질문이나 태스크에 적응하는 데 어려움을 겪고 있다는 문제를 다루고 있습니다. 이를 해결하기 위해 연속 학습 (continual learning) 프레임워크 내에서 VideoQA의 새로운 도전을 탐색합니다.

- **Technical Details**: 이 논문에서는 Collaborative Prompting (ColPro)을 제안하여, 특정 질문 제약 프롬프트(TQCP), 지식 획득 프롬프트(KAP), 시각적 시간 인식 프롬프트(VTAP)를 통합합니다. 이러한 프롬프트는 비디오QA에서 텍스트 질문의 맥락, 시각적 콘텐츠 및 비디오의 시간적 역학을 포착하는 것을 목표로 합니다.

- **Performance Highlights**: NExT-QA 및 DramaQA 데이터셋에서의 실험 결과, ColPro는 각각 55.14% 및 71.24%의 정확도로 기존 방법들에 비해 우수한 성능을 보여주었으며, 이는 실제 적용 가능성과 효용성을 강조합니다.



### DeepAerialMapper: Deep Learning-based Semi-automatic HD Map Creation for Highly Automated Vehicles (https://arxiv.org/abs/2410.00769)
Comments:
          For source code, see this https URL

- **What's New**: 본 논문에서는 고해상도 항공 이미지를 활용하여 HD 맵을 생성하는 반자동 방법을 제안합니다.

- **Technical Details**: 이 방법은 신경망(neural networks)을 학습시켜 고해상도 항공 이미지를 HD 맵에 관련된 클래스로 의미적으로 분할하는 것으로 구성됩니다. 분할된 결과는 이후 계층적으로 후처리(post-processing)되어 보이는 도로 요소의 프로토타입 HD 맵을 생성합니다. 생성된 맵은 Lanelet2 형식으로 내보내져 다양한 사용 사례에 대해 표준 도구를 사용하여 쉽게 확장할 수 있습니다.

- **Performance Highlights**: 평가 결과, 차선 표시와 도로 경계의 자동 맵핑에서 96% 이상의 재현율(recall)과 정밀도(precision)를 달성했습니다.



### Improved Generation of Synthetic Imaging Data Using Feature-Aligned Diffusion (https://arxiv.org/abs/2410.00731)
Comments:
          Accepted to First International Workshop on Vision-Language Models for Biomedical Applications (VLM4Bio 2024) at the 32nd ACM-Multimedia conference

- **What's New**: 이 논문에서는 기존의 확산 모델을 통해 의료 이미지를 생성하는 기존 접근 방식에서 개선된 'feature-aligned diffusion' 방법을 탐색하였습니다. 이 방법은 전문가 모델의 출력 특징과 확산 모델의 중간 특징을 정렬하여 생성 정확도를 9% 향상시키고 SSIM 다양성에서 약 0.12의 개선을 보였습니다.

- **Technical Details**: Feature-aligned diffusion은 확산 모델의 중간 특징을 전문가 모델의 출력 특징과 정렬하는 방식을 사용합니다. 이 과정에서는 전문가 모델의 클래시피케이션(output classification) 결과를 활용하며, 새로운 손실 함수를 도입하여 이 두 결과 간의 코사인 유사도(cosine similarity)를 극대화합니다. 기존의 훈련 절차에 비해 추가적인 프로젝션 레이어(projection layer)만 요구되어 기존의 훈련 파이프라인에 쉽게 통합될 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법인 feature-aligned diffusion을 통해 생성 이미지의 품질이 향상되었으며, 특히 훈련 시 노이즈를 추가한 이미지에서 전문가 모델의 특징을 정렬하는 것이 성능 향상에 기여했습니다. 기존의 방법에 비해 개선된 성능을 보이며, 이는 의료 이미지 생성 분야에서 중요한 진전을 이룬 것으로 평가됩니다.



### Simplified priors for Object-Centric Learning (https://arxiv.org/abs/2410.00728)
- **What's New**: 이번 논문에서는 인간의 데이터 추상화 능력을 모방하기 위해 간단하고 효율적인 SAMP(Simplified Slot Attention with Max Pool Priors) 방법을 제안합니다. 이 방법은 기존의 복잡한 Object-Centric 학습 접근법들보다 구현이 간편하며, 이미지에서 슬롯을 추출하는 데 있어 효율성을 높입니다.

- **Technical Details**: SAMP는 Convolution 및 MaxPool 레이어와 Attention 레이어를 사용하여 이미지 데이터를 처리합니다. Convolutional Neural Network(CNN)를 통해 입력 이미지를 인코딩하고, 이후 Convolution과 MaxPool 레이어의 교차 분기를 통해 특수한 하위 네트워크를 생성하여 원시 슬롯(primitive slots)을 추출합니다. 이 원시 슬롯들은 인코딩된 이미지에 대해 Simplified Slot Attention의 쿼리로 사용됩니다.

- **Performance Highlights**: SAMP는 기존 Object-Centric 방법들과의 비교에서 경쟁력이 있거나 이를 초월하는 성과를 보입니다. 특히, SAMP는 비반복적이고 완전하게 미분 가능한 방식으로 슬롯을 추출하므로, 기존의 Slot Attention 기반 방법들의 반복적 세련화 절차가 필요하지 않습니다.



### RAD: A Dataset and Benchmark for Real-Life Anomaly Detection with Robotic Observations (https://arxiv.org/abs/2410.00713)
- **What's New**: 최근 산업 비정상 감지의 발전이 현실적 데이터셋의 부족으로 어려움을 겪고 있습니다. 이러한 문제를 해결하기 위해 실제 로봇 팔을 사용하여 수집된 첫 번째 다중 뷰 RGB 기반 비정상 감지 데이터셋인 'Realistic Anomaly Detection (RAD)'를 소개합니다. 이 데이터셋은 4765장의 이미지로 구성되어 있으며, 13개 카테고리와 4가지 결함 유형을 포함합니다.

- **Technical Details**: RAD 데이터셋은 50개 이상의 관점에서 수집된 이미지로 구성되어 있어서, 다양한 관점에서 비정상을 감지할 수 있는 실질적인 기준을 제공합니다. 이 데이터셋은 비정상 감지 알고리즘의 성능을 평가하기 위한 포괄적인 테스트베드 기능을 하며, 노이즈 및 불일치한 데이터에 대한 고급 전처리 방법도 제안합니다.

- **Performance Highlights**: RAD 데이터셋은 기존의 2D RGB 기반 및 3D 다중 뷰 RGB 기반 알고리즘의 성능을 평가하고, 각 접근 방식의 강점과 약점을 분석하여 미래 개선 방향을 제시합니다. 또한, Noisy-Pose-Based Anomaly Detection (NAD) 챌린지를 통해 실제 로봇 시나리오에서의 성능을 표준화된 기준으로 평가할 수 있습니다.



### BioFace3D: A fully automatic pipeline for facial biomarkers extraction of 3D face reconstructions segmented from MRI (https://arxiv.org/abs/2410.00711)
- **What's New**: BioFace3D는 자기공명영상(magnetic resonance images)에서 재구성된 얼굴 모델을 이용하여 얼굴 바이오마커(facial biomarkers)를 측정하는 완전 자동화 도구입니다.

- **Technical Details**: BioFace3D는 세 가지 자동 모듈로 나누어져 있습니다: 자기공명영상에서 3D 얼굴 모델을 추출하는 모듈, 얼굴 형태(facial morphology)를 인코딩하는 동일한 3D 랜드마크를 등록하는 모듈, 그리고 해부학적 랜드마크 좌표를 사용하여 기하학적 형태 분석(geometric morphometrics) 기법으로 얼굴 바이오마커를 계산하는 모듈입니다.

- **Performance Highlights**: BioFace3D는 수작업으로 얼굴 기형을 코딩하는 부담을 덜어주고, 관찰자의 변동성을 최소화하며, 미세한 얼굴 기형도 정확하게 포착할 수 있는 가능성을 제시합니다.



### FlashMix: Fast Map-Free LiDAR Localization via Feature Mixing and Contrastive-Constrained Accelerated Training (https://arxiv.org/abs/2410.00702)
- **What's New**: FlashMix는 LiDAR 로컬라이제이션을 위한 새로운 맵이 필요 없는 프레임워크를 제안합니다. 이 프레임워크는 사전 훈련된 포인트 인코더와 장면 특화된 포즈 회귀기를 결합하여 훈련 시간을 대폭 단축시킵니다.

- **Technical Details**: FlashMix는 고정된 장면에 독립적인 베이스를 활용하여 지역 포인트 설명자를 추출하고, MLP 믹서를 통해 이들을 집계하여 센서의 포즈(구조적 배치)를 예측합니다. 포인트 클라우드의 각 포인트에 대해 로컬 설명자를 추출하여 훈련 버퍼를 만들고, 이를 바탕으로 설명자 집계기 및 포즈 예측기를 훈련합니다.

- **Performance Highlights**: FlashMix는 각 장면에 대해 포즈 회귀기만 훈련함으로써 훈련을 가속화하여 실질적인 환경에서 신속하고 정확한 LiDAR 국소화를 달성합니다. 본 결과는 다양한 LiDAR 로컬라이제이션 벤치마크에서 기존 방법과 비교하여 경쟁력을 입증했습니다.



### Mining Your Own Secrets: Diffusion Classifier Scores for Continual Personalization of Text-to-Image Diffusion Models (https://arxiv.org/abs/2410.00700)
Comments:
          Work under review

- **What's New**: 이 연구에서는 Text-to-Image diffusion 모델의 지속적인 개인화(Continual Personalization, CP)를 위한 새로운 방법을 제안합니다. 주된 초점은 사용자가 동시에 하나의 개념을 개인화하더라도 이전 개념의 데이터를 저장할 수 없는 문제를 해결하는 것입니다.

- **Technical Details**: 제안된 방법은 class-specific 정보에 기반한 regularization 기법을 통해 Text-to-Image diffusion 모델의 매개변수 공간과 함수 공간을 정규화합니다. 이를 위해 Diffusion Classifier (DC) 점수를 활용하여 Elastic Weight Consolidation (EWC) 및 double-distillation framework를 제안합니다.

- **Performance Highlights**: 제안된 방법은 다양한 데이터 세트에서 기존의 C-LoRA 및 다른 기법들과 비교하여 우수한 성능을 보였으며, storage 및 parameter overhead를 획기적으로 줄였습니다. 또한, zero inference time overhead를 달성하여 실용적인 CL 솔루션을 제시합니다.



### Advanced Arabic Alphabet Sign Language Recognition Using Transfer Learning and Transformer Models (https://arxiv.org/abs/2410.00681)
Comments:
          6 pages, 8 figures

- **What's New**: 본 논문에서는 깊이 있는 학습(deep learning) 방법과 전이 학습(transfer learning), 그리고 transformer 기반 모델을 이용한 아랍어 알파벳 수화 인식 방안을 제시하고 있습니다. 아랍 수화(Arabic Sign Language) 동작의 독특한 특징을 포착하기 위해 ArSL2018과 AASL이라는 두 개의 공공 데이터셋에서 다양한 변형 모델들의 성능을 연구하였습니다.

- **Technical Details**: 이 연구는 최신 CNN 아키텍처인 ResNet50, MobileNetV2, EfficientNetB7 및 Google ViT와 Microsoft Swin Transformer와 같은 최신 transformer 모델을 활용합니다. 컨볼루션 신경망(CNN) 모델과 transformer를 활용한 특징 추출을 포함하는 여러 주요 단계로 구성된 아랍어 알파벳 수화 인식 시스템을 개발하였습니다. 이 시스템은 데이터 전처리(data preprocessing), 모델 선택(model selection with transfer learning), 모델 평가(model evaluation)로 이루어져 있습니다.

- **Performance Highlights**: 실험 결과 ArSL2018과 AASL 데이터셋에서 각각 99.6%와 99.43%의 높은 인식 정확도를 달성하였으며 이는 기존의 최첨단(leading-edge) 접근 방식들을 훨씬 초월하는 결과입니다. 이러한 성능 향상은 아랍어를 사용하는 청각 장애인 및 난청인에게 더 접근성 높은 커뮤니케이션 방법을 제공하고, 포용적인 사회를 촉진하는 데 기여할 것입니다.



### GMT: Enhancing Generalizable Neural Rendering via Geometry-Driven Multi-Reference Texture Transfer (https://arxiv.org/abs/2410.00672)
Comments:
          Accepted at ECCV 2024. Code available at this https URL

- **What's New**: 이번 논문에서는 Geometry-driven Multi-reference Texture transfer network (GMT)를 제안하여 Generalizable NeRF (G-NeRF) 모델의 성능을 향상시키는 방법을 소개합니다. 이 모듈은 기존 G-NeRF 모델의 제한점을 극복하며, 플러그 앤 플레이(plug-and-play) 방식으로 사용할 수 있습니다.

- **Technical Details**: 제안된 Ray-imposed Deformable Convolution Network (RayDCN)은 장면의 기하학을 반영하여 입력 및 참조 특징을 정렬합니다. TP-Former는 다중 참조 이미지에서 특징을 집계하여 텍스처 정보를 유지하면서도 참조 텍스처를 전이합니다. 이는 G-NeRF 모델이 각 픽셀 독립적으로 렌더링하는 방식과는 다르게, 인접 픽셀 간의 상호작용을 가능하게 하여 고주파 세부사항을 포착할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 GMT 모듈은 여러 벤치마크 데이터셋에서 G-NeRF 모델의 성능을 일관되게 개선함을 확인하였습니다. 특히, 텍스처가 풍부한 멀티 뷰 소스 입력을 사용하더라도 세부 사항 표현이 개선되는 것을 보여주었습니다.



### Cross-Camera Data Association via GNN for Supervised Graph Clustering (https://arxiv.org/abs/2410.00643)
- **What's New**: 최신 논문에서는 크로스 카메라 데이터 연관성(Cross-camera data association) 문제를 다루며, 이를 그래프 신경망(GNN) 기반으로 해결하는 새로운 접근 방식을 제시합니다. 제안된 모델인 SGC-CCA는 기존의 GNN-CCA보다 모든 클러스터링 지표에서 뛰어난 성능을 보입니다.

- **Technical Details**: SGC-CCA는 각 카메라에서 캡처된 인스턴스의 시각적 특징과 위치 속성을 기반으로 그래프 연결 예측을 통해 인스턴스를 연결합니다. 이 방법은 GNN 아키텍처를 이용하여 노드 관계를 분석하고, 노드 쌍 간의 연결의 존재 여부를 분류합니다. 실험은 다양한 환경에서 다중 카메라 보행자 데이터셋을 사용하여 수행되었습니다.

- **Performance Highlights**: SGC-CCA는 GNN-CCA보다 모든 클러스터링 메트릭에서 월등한 성과를 보였으며, 그래프 후처리 없이도 전체 클러스터링 솔루션을 제공합니다. 데이터셋은 연구소, 농구 코트 및 테라스와 같은 환경에서 수집되었습니다.



### Cafca: High-quality Novel View Synthesis of Expressive Faces from Casual Few-shot Captures (https://arxiv.org/abs/2410.00630)
Comments:
          Siggraph Asia Conference Papers 2024

- **What's New**: 이 논문은 3개의 입력 이미지만으로도 높은 충실도의 3D 얼굴 모델링을 가능하게 하는 새로운 부피적(Volumetric) 사전(prior)을 제안합니다. 이 모델은 합성 데이터에 기반한 암묵적 prior를 사용하여 실제 표현과 아이덴티티를 일반화하여, 주름이나 속눈썹과 같은 세부적 특성을 렌더링할 수 있습니다.

- **Technical Details**: 연구진은 3D Morphable Face Model을 활용하여 다양한 표정, 머리 및 의상을 가진 대규모 합성 훈련 세트를 생성했습니다. 그런 다음, 이 합성 데이터셋에서 조건부 Neural Radiance Field prior를 훈련시키고, 추론 시 단일 주체의 스파스(real images) 집합에서 모델을 미세 조정합니다. 이는 합성에서 실제 도메인 간의 격차를 메우기 위해 평균적으로 3개의 입력만을 필요로 합니다.

- **Performance Highlights**: 이 새로운 개인화된 3D 모델은 도전적인 조명 조건에서 강력한 개인적 얼굴 표정을 재구성하고, 스파스 입력으로부터의 얼굴 고해상도 새 뷰 합성에서 기존의 최첨단 기술보다 우수한 성능을 보입니다. 이에 따라, 고품질의 새로운 뷰를 합성하는 데 있어 최고의 시각적 및 사진 측정 품질을 달성합니다.



### An Illumination-Robust Feature Extractor Augmented by Relightable 3D Reconstruction (https://arxiv.org/abs/2410.00629)
- **What's New**: 이 논문은 조명 변화에 강건한 feature extractor의 설계 절차를 제안합니다. 최근 개발된 relightable 3D reconstruction 기술을 활용하여 다양한 조명 조건에서 빠르고 직접적으로 데이터를 생성하는 방법을 제시합니다.

- **Technical Details**: 제안된 방법은 relightable 3D reconstruction 알고리즘을 기반으로 하여 여러 조명 조건과 카메라 뷰에서 이미지를 생성합니다. Self-supervised framework를 통해 조명 변화에 따른 key points의 반복성과 descriptor 간의 유사성을 높이는 방안을 모색합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 실제 데이터셋에서 조명 변화에 대해 feature의 반복성과 유사성을 개선하는 데 효과적임을 입증했습니다. 또한, ablation study를 통해 self-supervised framework의 설계 효과도 나타났습니다.



### GERA: Geometric Embedding for Efficient Point Registration Analysis (https://arxiv.org/abs/2410.00589)
- **What's New**: 본 연구에서는 순수한 MLP 아키텍처를 활용하여 포인트 클라우드( point cloud ) 등록을 위한 새로운 네트워크를 제안합니다. 기존의 복잡한 기능 추출기 없이 오프라인으로 기하학적 정보를 구축하여 계산 및 메모리 부담을 줄입니다.

- **Technical Details**: GERA( GEometric embedding for leaRning-based efficient point registrAtion )라는 방법을 제안하며, 이는 포인트 클라우드에서 기하학적 정보를 효율적으로 생성합니다. 이 방법은 MLP 아키텍처를 기반으로 하여 포인트 간의 거리를 나타내는 완전 연결 그래프를 형성합니다. 또한, Maximum Mean Discrepancy(MMD) 분석을 통해 기하학적 정보의 안정성을 입증합니다.

- **Performance Highlights**: 기존의 최첨단(SOTA) 솔루션에 비해 12.5% 향상된 성능을 보이며, 필요한 계산 시간은 단 3%에 불과하다는 결과를 보였습니다. 이 제안된 방법은 추론 속도를 22배 증가시키고, 예측 정확도는 115% 향상되었습니다.



### Can We Remove the Ground? Obstacle-aware Point Cloud Compression for Remote Object Detection (https://arxiv.org/abs/2410.00582)
Comments:
          7 Pages; submitted to ICRA 2025

- **What's New**: 이 연구에서는 경량이며 장애물 인식 관점을 고려한 Pillar 기반 지면 제거(Pillar-based Ground Removal, PGR) 알고리즘을 제안합니다. 이 알고리즘은 객체 인식에 중요하지 않은 지면 점들을 걸러내어 압축 비율을 크게 개선하며 수신 측 인식 성능을 유지합니다.

- **Technical Details**: PGR 알고리즘은 복잡한 객체 탐지나 의미적 분할 모델을 사용하지 않고도 높은 병렬 처리 속도를 유지하면서, 지면에 가까운 객체의 대부분의 지면 점을 선택적으로 유지하는 기능을 갖추고 있습니다. 이 연구는 또한 3D 객체 탐지 모델이 지면 점에 강한 의존성을 보이고 있음을 밝혀냈습니다. 실험 결과 PGR은 20-30%의 점을 제거하면서도 SOTA 탐지 모델의 성능을 유지할 수 있음을 보여주었습니다.

- **Performance Highlights**: KITTI 및 Waymo Open Dataset에서의 평가 결과, PGR은 86 FPS의 속도로 작동하면서도 탐지 정확도에서 큰 손실 없이 성능을 유지하는 것으로 나타났습니다.



### Deep activity propagation via weight initialization in spiking neural networks (https://arxiv.org/abs/2410.00580)
- **What's New**: 이번 연구에서는 Spiking Neural Networks (SNNs)에서 효과적으로 훈련할 수 있는 최적의 가중치 초기화(weight initialization) 방법을 제안합니다. 이 방법은 SNN의 고유한 계산 특성을 고려하여 고안되었습니다.

- **Technical Details**: 제안하는 가중치 초기화 방법은 SNN의 양자화(quantization) 연산을 고려하여 개발되었습니다. 이를 통해 깊은 SNN에서 활동이 손실되는 현상을 방지하고, 스파이크(spike)가 인해 정보 손실을 겪지 않도록 하고 있습니다. 100층까지의 SNN에 대한 수치 시뮬레이션을 통해 이론적으로 도출된 결과를 입증하였습니다.

- **Performance Highlights**: MNIST 데이터셋을 사용한 실험에서 제안하는 가중치 초기화 기법을 적용했을 때, 더 높은 정확도와 빠른 수렴(convergence) 속도를 보였습니다. 또한, 새로운 가중치 초기화 방법은 여러 네트워크 및 뉴런 하이퍼파라미터(hyperparameters)의 변화에 대해 견고함을 입증하였습니다.



### STanH : Parametric Quantization for Variable Rate Learned Image Compression (https://arxiv.org/abs/2410.00557)
Comments:
          Submitted to IEEE Transactions on Image Processing

- **What's New**: 본 연구에서는 learned image compression(학습된 이미지 압축) 기술에 새로운 접근 방식을 제시합니다. 기존의 고정된 비트 전송률을 위한 여러 개의 encoder-decoder 쌍을 학습하는 대신, STanH(differentiable quantizer)를 활용하여 단일 모델로 다양한 비트 전송률을 지원할 수 있게 되었습니다.

- **Technical Details**: STanH는 하이퍼볼릭 탄젠트의 파라메트릭 합으로 설계된 차별화 가능한 양자화 레이어입니다. 이는 사용자가 선택한 비트 전송률에 맞춰 조정될 수 있는 학습 가능한 양자화 파라미터를 포함합니다. STanH는 간단한 매개변수 조정을 통해 세밀한 양자화에서 거친 양자화로의 전환을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, STanH를 적용한 방법은 경쟁력 있는 압축 효율을 유지하면서도 다양한 비트 전송률을 지원할 수 있는 기능을 보여주었습니다. 또한 이는 모델 훈련 비용, 저장 비용 절감, 배포 용이성 등의 면에서 상당한 이점을 제공합니다.



### Drone Stereo Vision for Radiata Pine Branch Detection and Distance Measurement: Utilizing Deep Learning and YOLO Integration (https://arxiv.org/abs/2410.00503)
- **What's New**: 본 연구는 가지치기 도구와 스테레오 비전 카메라를 장착한 드론을 개발하여 나무 가지의 공간 위치를 정확하게 감지하고 측정하는 데 중점을 두고 있습니다.

- **Technical Details**: 구체적으로, 가지 분할에는 YOLO(You Only Look Once) 알고리즘이 사용되며, 두 가지 깊이 추정 접근 방식인 모노큘러(monocular)와 스테레오(stereo)가 조사됩니다. SGBM(Semi-Global Matching)과 비교할 때, 딥 러닝 기술이 더 정밀하고 정확한 깊이 맵을 생성합니다. 지상 진리 데이터가 없는 경우, 최적의 깊이 값을 근사하기 위해 딥 뉴럴 네트워크를 활용한 파인 튜닝(fine-tuning) 과정이 적용됩니다.

- **Performance Highlights**: 결과적으로 가지 감지 및 거리 측정의 정확도와 효율성이 크게 향상되었습니다. 이는 딥 러닝이 농업 분야에서 혁신을 촉진하고 자동화를 향상시킬 가능성을 강조합니다.



### CaRtGS: Computational Alignment for Real-Time Gaussian Splatting SLAM (https://arxiv.org/abs/2410.00486)
Comments:
          Submitted to IEEE Robotics and Automation Letters

- **What's New**: 이번 연구에서는 실시간 환경에서 포토리얼리스틱(scene photorealistic) 장면 재구성을 개선하기 위한 새로운 접근법인 CaRtGS(Computational Alignment for Real-Time Gaussian Splatting SLAM)를 소개합니다. 이는 3D Gaussian Splatting(3DGS)을 활용하여 처리 속도와 렌더링 품질을 동시에 향상시키는 방법입니다.

- **Technical Details**: CaRtGS는 Gaussian Splatting SLAM(GS-SLAM)에서 발생하는 계산적 불일치를 해결하기 위한 적응형 전략을 도입하여, 훈련 최적화와 밀집화 과정을 개선합니다. 이는 고품질 렌더링을 위해 더 적은 Gaussian primitive를 사용하면서 실시간 요구를 충족하도록 설계되었습니다.

- **Performance Highlights**: Replica 및 TUM-RGBD 데이터셋에서 CaRtGS의 성능을 검증한 결과, 높은 충실도(rendering fidelity)를 유지하면서도 더 적은 Gaussian primitive로 실시간 포토리얼리스틱 렌더링을 달성했습니다. 이는 SLAM 분야에서의 패러다임 전환을 이끌 것으로 기대됩니다.



### A Hitchhikers Guide to Fine-Grained Face Forgery Detection Using Common Sense Reasoning (https://arxiv.org/abs/2410.00485)
Comments:
          Accepted at NeurIPS'2024 (D&B)

- **What's New**: 이 연구는 Deepfake 탐지를 비주얼 질문 응답(Visual Question Answering, VQA) 다중 라벨 문제로 변환하는 새로운 접근법을 소개합니다.

- **Technical Details**: 모델의 성능을 평가하기 위해 세 단계의 접근법을 채택했습니다. 첫 번째 단계에서는 이진 작업과 모델의 지침에 대한 민감성을 평가하고, 두 번째 단계에서는 다중 선택 VQA 설정에서 조작 영역을 식별합니다. 마지막으로, 세 번째 단계에서 공개 질문으로 세분화된 감지 과제를 변환하고 다양한 매칭 전략을 비교합니다.

- **Performance Highlights**: 제안한 벤치마크를 통해 여러 인기 있는 모델을 적용하여 이진, 다중 선택, 개방형 VQA 평가를 세 개의 데이터셋에서 상세히 비교했습니다.



### MCGM: Mask Conditional Text-to-Image Generative Mod (https://arxiv.org/abs/2410.00483)
Comments:
          17 pages, 13 figures, presented at the 5th International Conference on Artificial Intelligence and Machine Learning (CAIML 2024)

- **What's New**: 최근 발전한 생성 모델들은 인공지능(AI) 분야에서 혁신을 일으켰습니다. 이 연구에서는 특정 포즈를 가진 이미지를 생성하는 새로운 Mask Conditional Text-to-Image Generative Model (MCGM)을 제안합니다.

- **Technical Details**: MCGM은 conditional diffusion models의 힘을 활용하여, 여러 주체를 포함한 단일 이미지를 기반으로 새로운 장면을 생성한 기존 Break-a-scene 모델의 성공을 바탕으로 합니다. 이 모델은 mask embedding injection을 통합하여 생성 과정의 조건화를 가능하게 합니다.

- **Performance Highlights**: 광범위한 실험과 평가를 통해, 제안된 모델이 미리 정의된 mask 조건을 충족하는 고품질 이미지를 생성하는데 효과적이며, 현재의 Break-a-scene 생성 모델을 개선했음을 보여줍니다.



### ViDAS: Vision-based Danger Assessment and Scoring (https://arxiv.org/abs/2410.00477)
Comments:
          Preprint

- **What's New**: 이번 연구는 위험을 분석하고 평가하기 위한 새로운 데이터셋을 제시합니다. 이 데이터셋은 100개의 유튜브 비디오로 구성되어 있으며, 각 비디오에서는 인간 참여자가 위험도를 0(위험 없음)에서 10(생명 위협)까지 평가했습니다. 또한, 대규모 언어 모델(LLM)을 활용하여 비디오 요약을 통해 위험 수준을 독립적으로 평가하는 방법도 포함되어 있습니다. 이 연구는 인간과 LLM의 위험 평가 능력을 비교 분석합니다.

- **Technical Details**: 연구에서는 비디오 내 위험을 효과적으로 평가하기 위해 Mean Squared Error (MSE) 점수 체계를 도입하여 인간과 LLM의 위험 평가 간 정렬성을 메타 평가했습니다. 위험 평가를 위해 각 비디오는 위험 요소를 정확히 식별하고 평가하는 과정을 포함하며, LLM을 통한 비디오 분석을 통해 수행됩니다. 기존의 위험 탐지 방법과 달리, 이 연구에서는 위험의 더 폭넓은 맥락을 고려하여 위험 수준을 정량화합니다.

- **Performance Highlights**: 제안된 데이터셋은 위험 평가 모델의 표준 벤치마크 역할을 하며, 인간과 LLM의 위험 인식의 유사성을 탐구하여 LLM이 인간과 유사한 평가를 할 수 있는 가능성을 보여줍니다. 이 연구는 안전 시스템, 온라인 플랫폼, 자율 시스템 등 다양한 분야에서의 응용 가능성을 제시하며, 향후 위험 평가 연구에서의 새로운 방향을 제시합니다.



### Deep Multimodal Fusion for Semantic Segmentation of Remote Sensing Earth Observation Data (https://arxiv.org/abs/2410.00469)
- **What's New**: 본 논문은 VHR(a Very High Resolution) 항공 이미지와 SITS(Satellite Image Time Series)의 상호보완적인 강점을 활용한 late fusion deep learning model(LF-DLM)을 제안하여, 원거리 탐사(remote sensing) 이미지의 의미적 세분화를 개선하는 방법을 다룹니다.

- **Technical Details**: LF-DLM 모델은 두 개의 독립적인 딥러닝 브랜치로 구성됩니다. 하나의 브랜치는 UNetFormer와 Multi-Axis Vision Transformer(MaxViT) 백본을 통해 항공 이미지의 세부 텍스처를 통합하며, 다른 브랜치는 U-Net과 Temporal Attention Encoder(U-TAE)를 통해 Sentinel-2 위성 이미지 시리즈로부터 복잡한 시공간(dynamics) 정보를 캡처합니다.

- **Performance Highlights**: LF-DLM 모델은 FLAIR 데이터셋에서 최첨단(result) 성능을 달성하여, 멀티소스 광학 이미지의 의미적 세분화에서 새로운 기준을 세웠으며, 다양한 지표면 타입에 대한 세분화 정확도를 증가시켰습니다.



### Enabling Synergistic Full-Body Control in Prompt-Based Co-Speech Motion Generation (https://arxiv.org/abs/2410.00464)
Comments:
          Project Page: this https URL

- **What's New**: 본 논문에서는 SynTalker를 제안하여, 기존의 co-speech motion generation 접근 방식의 한계를 극복하고 보다 정교한 전신 동작 생성을 지원합니다. 특히, 음성과 텍스트 프롬프트를 기반으로 한 융합된 제어를 가능하게 합니다.

- **Technical Details**: SynTalker는 두 가지 주요 기술 기여를 포함합니다. 첫째, 다단계 훈련 과정(multi-stage training process)을 통해 음성, 모션 및 프롬프트 간의 정렬된 임베딩 공간(aligned embedding space)을 확보합니다. 둘째, 분산 기반 조건부 추론 과정(diffusion-based conditional inference process)을 통해 세부적인 신체 부위 제어(local body parts control)를 구현합니다.

- **Performance Highlights**: 광범위한 실험을 통해, SynTalker는 기존 접근 방식이 지원하지 못하는 정확하고 유연한 전신 모션 생성(synergistic full-body motion generation)을 가능하게 함을 입증했습니다.



### Advancing Medical Radiograph Representation Learning: A Hybrid Pre-training Paradigm with Multilevel Semantic Granularity (https://arxiv.org/abs/2410.00448)
Comments:
          18 pages

- **What's New**: 이 논문은 방사선 이미지를 활용한 Medical Vision-Language Pre-training (Med-VLP) 분야에서의 혁신적인 접근법을 소개합니다. 기존 방법들이 텍스트 주석을 통합된 보고서로 합치는 경향이 있는 반면, 우리는 방사선 데이터셋에서의 발견(findings)과 인상(impression) 섹션 간의 내재적인 계층적 관계를 인식하고 있습니다.

- **Technical Details**: 우리는 HybridMED라는 새로운 프레임워크를 제안하여 글로벌 수준의 시각 표현을 인상과 정렬하고, 토큰 수준의 시각 표현을 발견과 정렬합니다. 또한, 두 개의 대리 작업을 사용하는 생성 디코더를 포함하여, 하나는 이미지에서 인상을 생성하고, 다른 하나는 발견을 요약하는 것입니다. 지식 증류(knowledge distillation)도 훈련 과정을 지원하는 데 활용됩니다.

- **Performance Highlights**: MIMIC-CXR 데이터셋에서의 실험 결과, 우리의 요약(branch) 작업이 캡셔닝(branch)에 효과적으로 지식을 증류하여 모델 성능을 향상시키고, 파라미터 요구 사항을 크게 늘리지 않으면서도 우수한 성능을 보여주었습니다.



### Scene Graph Disentanglement and Composition for Generalizable Complex Image Generation (https://arxiv.org/abs/2410.00447)
Comments:
          Accepted by NeurlPS 2024

- **What's New**: 본 논문에서는 복잡한 이미지 생성을 위한 새로운 접근법으로 세미틱(semantic) 레이아웃 변분 오토인코더(SL-VAE)를 제안합니다. 이는 입력된 장면 그래프(scene graph)에서 레이아웃과 의미를 하나의 매핑으로 유도하여 보다 다양한 이미지를 생성할 수 있도록 합니다.

- **Technical Details**: 세미틱 레이아웃 변분 오토인코더(SL-VAE)는 장면 그래프에서 공간적 관계와 비공간적 상호작용을 동시 모델링합니다. 이를 통해 다양한 레이아웃과 세미틱 정보를 분리하여 디퓨전 모델(diffusion model)과 결합하여 고해상도 이미지를 생성합니다. 또한, Composition Masked Attention(CMA) 메커니즘을 도입하여 생성 과정에서의 관계 혼란 및 속성 누출을 방지합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 텍스트, 레이아웃 또는 장면 그래프 기반의 최근 경쟁 모델들보다 생성의 합리성과 제어 가능성 면에서 우수함을 입증했습니다. 특히, 객체-레벨 그래프 조작에서 일반화된 성능을 보여주었습니다.



### TikGuard: A Deep Learning Transformer-Based Solution for Detecting Unsuitable TikTok Content for Kids (https://arxiv.org/abs/2410.00403)
Comments:
          NILES2024

- **What's New**: 이 연구는 TikTok에서 아동에게 부적절한 콘텐츠를 탐지하고 차단하기 위한 새로운 딥러닝 솔루션인 TikGuard를 소개합니다. 전통적인 콘텐츠 무결성 검토 방법의 한계를 극복하기 위해 고급 transformer 기반 모델을 활용하여 86.7%의 높은 정확성을 달성했습니다.

- **Technical Details**: TikGuard는 TikHarm라는 특별히 큐레이션된 데이터셋을 사용하여 아동에게 유해한 콘텐츠를 분류합니다. 이를 위해 고급 비디오 분류 기법을 사용하며, TimesFormer, VideoMAE, ViViT 등의 최신 transformer 모델을 채택하여 콘텐츠 검토의 정확성을 향상시킵니다.

- **Performance Highlights**: TikGuard는 기존 방법들에 비해 높은 정확성을 보였으며, 아동을 위한 안전한 온라인 환경을 조성하는 데 기여할 잠재력을 지니고 있습니다. TikHarm 데이터셋의 독창성으로 직접 비교가 제한되지만, 연구 결과는 소셜 미디어 플랫폼에서의 콘텐츠 검토 단계를 한층 더 발전시키는 데 이용될 수 있습니다.



### CusConcept: Customized Visual Concept Decomposition with Diffusion Models (https://arxiv.org/abs/2410.00398)
- **What's New**: 본 논문에서는 Customized Concept Decomposition(맞춤형 개념 분해)이라는 새로운 작업을 제안합니다. 이는 단일 이미지를 여러 관점에서 분해하여 시각적 개념을 생성하는 것을 목표로 합니다. 이를 위해 CusConcept라는 두 단계의 프레임워크를 사용하여 관습적으로 정의된 개념 축을 통해 개념 임베딩 벡터를 추출합니다.

- **Technical Details**: CusConcept는 사용자 정의 개념 분해를 위한 두 단계 훈련 프로세스로 구성됩니다. 첫 번째 단계에서는 어휘 기반 개념 분해 메커니즘을 활용하여 인간이 지정한 개념 축을 기반으로 어휘를 구축합니다. 두 번째 단계에서는 개념 정제를 통해 생성된 이미지의 충실도(fidelity)와 품질을 향상시킵니다. 이 과정에서 대규모 언어 모델(LLMs)을 사용하여 어휘를 얻고, 다중 토큰 Textual Inversion을 통해 개념 임베딩을 개선합니다.

- **Performance Highlights**: CusConcept는 고품질 이미지를 생성하고 관련 렉시컬 예측을 보조 결과로 제공합니다. 넓은 범위의 실험을 통해, CusConcept은 맞춤형 개념 분해 작업에 있어 최첨단 성능을 보여줍니다. 본 연구는 CusConcept이 대규모 데이터 세트를 기반으로 구성된 벤치마크에서 성능을 평가하는 방법을 제시합니다.



### Seamless Augmented Reality Integration in Arthroscopy: A Pipeline for Articular Reconstruction and Guidanc (https://arxiv.org/abs/2410.00386)
Comments:
          8 pages, with 2 additional pages as the supplementary. Accepted by AE-CAI 2024

- **What's New**: 이번 논문에서는 단안(전방향) 관절경 비디오를 바탕으로 수술 중 인식을 향상시키기 위한 파이프라인을 제시합니다. 이 시스템은 SLAM(Simultaneous Localization and Mapping), 깊이 추정, 3D Gaussian splatting을 결합하여 관절 내 구조를 실시간으로 재구성합니다.

- **Technical Details**: 제안된 파이프라인은 OneSLAM을 사용해 희소 3D 포인트 맵을 구성하고, 단안 깊이 추정 모델을 활용하여 각 프레임마다 불확실한 깊이 정보를 생성합니다. 마지막으로, 3D GS 모델을 활용하여 사진 현실적인 3D 장면을 세밀하게 재구성합니다. 이 기술은 AR(Augmented Reality) 어플리케이션을 통해 관절 구조의 측정 및 주석 개발을 가능하게 합니다.

- **Performance Highlights**: 이 파이프라인은 평균 7분 안에 높은 밀도의 3D 재구성을 달성하고, RMSE=2.21mm 재구성 오류, PSNR=32.86, SSIM=0.89의 성능을 기록했습니다. AR 측정 도구는 평균 1.59 +/- 1.81mm의 정확도를 보여주었으며, AR 주석 도구는 mIoU=0.721의 성과를 나타냈습니다.



### GLMHA A Guided Low-rank Multi-Head Self-Attention for Efficient Image Restoration and Spectral Reconstruction (https://arxiv.org/abs/2410.00380)
- **What's New**: 이 연구에서는 기존의 Channel-wise Self-Attention (CSA)를 대체하는 인스턴스 기반 저랭크 다중 헤드 셀프 어텐션(Instance-Guided Low-rank Multi-Head Self-Attention, GLMHA) 기법을 제안합니다. GLMHA는 짧은 및 긴 입력 시퀀스 모두에 대해 계산 이점을 제공하면서도 원래 모델의 성능을 유지합니다.

- **Technical Details**: GLMHA는 입력 특징 맵 X로부터 저랭크 키 K 및 값 V 행렬 생성을 위해 인스턴스 가이드를 활용하여 중요한 정보 콘텐츠를 극대화합니다. 이 접근법은 Compute와 Parameter량 감소로 인해 기존 기술들에 비해 더 효율적이며, 또한 이미지 복원 및 스펙트럼 재구성 작업을 위해 근본적인 평가를 수행합니다.

- **Performance Highlights**: GLMHA는 최대 7.7 Giga FLOPs 감소와 37만 개의 파라미터 감소를 통해 기존 CSA를 사용하는 상위 모델의 성능을 밀접하게 유지합니다. 이 연구는 Restormer, MST-L 및 MST++와 같은 세 가지 주요 기반 모델을 사용하여 실험 결과를 제시합니다.



### CXPMRG-Bench: Pre-training and Benchmarking for X-ray Medical Report Generation on CheXpert Plus Datas (https://arxiv.org/abs/2410.00379)
Comments:
          In Peer Review

- **What's New**: 본 논문은 X-ray 이미지 기반의 의학 보고서 생성을 위한 새로운 대규모 데이터셋인 CheXpert Plus와 함께, 이를 기반으로 한 CXPMRG-Bench 벤치마크를 제안하여 X-ray 보고서 생성을 위한 기존 알고리즘과 대형 모델들의 성능을 비교합니다.

- **Technical Details**: X-ray 보고서 생성을 위한 MambaXray-VL이라는 새로운 대형 모델을 제안하였으며, 이 모델은 자가 감독형(autoregressive) 생성과 X-ray 보고서 대비 학습(constrastive learning) 전략을 포함한 다단계(pre-training) 방식으로 훈련됩니다. 이는 기존의 Transformer 기반 모델들에서 발생하는 높은 계산 비용을 절감하며, 이미지와 텍스트 비교학습을 통해 더욱 효과적인 기능 공간 정렬을 달성합니다.

- **Performance Highlights**: MambaXray-VL은 CXPMRG-Bench 벤치마크에서 19개의 주요 X-ray 의학 보고서 생성 알고리즘과 14개의 대형 언어 모델, 2개의 비전-언어 모델을 평가하였고, 실험 결과에서도 최첨단 성능을 달성하여 기존 모델들과 비교했을 때 우수한 결과를 보였습니다.



### Descriptor: Face Detection Dataset for Programmable Threshold-Based Sparse-Vision (https://arxiv.org/abs/2410.00368)
Comments:
          8 pages

- **What's New**: 이 연구에서는 얼굴 인식 작업을 위해 특별히 설계된 주석이 포함된 시간 임계값 기반의 비전 데이터셋, 즉 스마트 이벤트 얼굴 데이터셋(Smart Event Face Dataset, SEFD)을 제공합니다. 이 데이터셋은 Aff-Wild2에서 사용된 동일한 비디오를 통해 추출된 데이터로, 다양한 임계값 수준(예: 4, 8, 12, 16)을 제공하여 최첨단 신경망 구조의 평가 및 최적화를 가능하게 합니다.

- **Technical Details**: 이 논문은 비전 작업을 위한 중요한 정보를 처리하는 신경 모양의 센서의 특징과 그에 따른 신호 데이터의 변환 과정을 설명합니다. 전통적인 영상 데이터와 달리, 이벤트 카메라는 비동기적으로 발생하는 일련의 희소 이벤트를 캡처합니다. 본 연구는 이러한 이벤트 기반 센서를 이용한 얼굴 인식 알고리즘 초기 개발 단계에서 필요한 적절한 데이터셋의 부재를 해결하기 위해 개발되었습니다. 이 새로운 데이터셋은 많은 프로그래머블 디지털 임계값을 지원하여 스마트 센서 모델링과 알고리즘 개발의 복잡성을 낮추는 데 기여합니다.

- **Performance Highlights**: 이 데이터셋을 사용하여 산업 표준의 객체 탐지 및 위치 추적 모델을 훈련시키고 그 효과를 검증했습니다. 특히, YOLO-v4 및 YOLO-v7 모델을 사용하여 성능을 벤치마크 했으며, YOLO-v7은 이전 YOLO 버전과 비교하여 업데이트되고 최적화된 아키텍처를 제공하여 더욱 나은 feature map 통합을 achieved했습니다.



### TFCT-I2P: Three stream fusion network with color aware transformer for image-to-point cloud registration (https://arxiv.org/abs/2410.00360)
- **What's New**: 본 연구에서는 이미지-포인트 클라우드 등록(Image-to-Point-Cloud Registration, I2P)을 위한 혁신적인 방법인 TFCT-I2P를 제안합니다. 이 방법은 색정보와 구조정보를 통합하여 두 가지 모달리티 간의 정렬 문제를 해결합니다.

- **Technical Details**: TFCT-I2P는 Three-Stream Fusion Network (TFN)를 기반으로 하여 이미지의 색 정보를 포인트 클라우드의 구조 정보와 융합합니다. 또한 색정보를 통해 발생하는 패치 수준의 잘못된 정렬을 완화하기 위해 Color-Aware Transformer (CAT)를 설계하였습니다. 이러한 구성은 복잡한 배경이나 다양한 조명 상황에서도 더 정확한 정렬을 가능하게 합니다.

- **Performance Highlights**: TFCT-I2P는 7Scenes, RGB-D Scenes V2, ScanNet V2 및 자체 수집한 데이터셋에 대한 실험에서, 주목할 만한 성능 향상을 보였습니다. 특히 Inlier Ratio에서 1.5%, Feature Matching Recall에서 0.4%, Registration Recall에서 5.4% 향상되었습니다.



### Efficient Training of Large Vision Models via Advanced Automated Progressive Learning (https://arxiv.org/abs/2410.00350)
Comments:
          Code: this https URL. arXiv admin note: substantial text overlap with arXiv:2203.14509

- **What's New**: 이 논문에서는 대형 비전 모델(Large Vision Models, LVMs)의 효율적인 학습을 위한 진보된 자동화적 점진적 학습(AutoProg) 프레임워크를 제안합니다. 기존의 학습 방법에 비해 훈련 시간과 비용을 크게 절감하면서도 성능은 유사하거나 향상되는 결과를 보였습니다.

- **Technical Details**: AutoProg 프레임워크는 ViTs(Vision Transformers)와 확산 모델(Diffusion Models)을 포함한 다양한 LVM에 적용됩니다. 특히, AutoProg-One 및 AutoProg-Zero 개발에 중점을 두며, 각각 모멘텀 증가(MoGrow) 및 제로샷 자동화된 점진적 학습 기법을 특징으로 합니다.

- **Performance Highlights**: 실험 결과, AutoProg-One은 ImageNet에서 ViTs의 사전 훈련을 최대 1.85배 가속화하였으며, AutoProg-Zero는 안정적 확산(Stable Diffusion) 및 Diffusion Transformers의 전이 훈련을 각각 최대 2.86배 및 2.56배 가속화했습니다. 중앙 성능을 유지하며 훈련 비용은 현저히 감소했습니다.



### Revisiting the Role of Texture in 3D Person Re-identification (https://arxiv.org/abs/2410.00348)
- **What's New**: 이 논문은 3D 데이터에서 쉽게 구할 수 있는 고해상도 텍스처 정보를 활용하여 3D 사람 재식별(person re-ID) 성능과 설명 가능성을 향상시키기 위한 새로운 프레임워크를 소개합니다.

- **Technical Details**: 우리는 UVTexture mapping을 활용하여 3D 사람 재식별 모델에서 텍스처를 강조하는 방법을 제안하고 있습니다. 이 방법은 3D 모델과 UVTexture의 heatmaps를 결합하여 사람 재식별 과정을 시각화하고 설명합니다. 특히, activation maps와 feature-based attention maps를 통해 사람 재식별 결정에 기여하는 중요한 영역과 특징을 강조합니다.

- **Performance Highlights**: UVTexture 처리를 사용하여 3D 모델에서 텍스처 세부사항을 강조하는 새로운 접근 방식을 통해, 우리는 3D 사람 재식별 분야에서 최고 성능(state-of-the-art)을 달성했습니다. 또한, 공개된 모든 데이터와 코드, 모델을 통해 결과의 재현성을 보장하고 있습니다.



### SyntheOcc: Synthesize Geometric-Controlled Street View Images through 3D Semantic MPIs (https://arxiv.org/abs/2410.00337)
- **What's New**: 이 논문에서는 SyntheOcc라는 새로운 확산 모델을 제안하여, 주행 시나리오에서 점유 상태(occupancy labels)에 따라 포토리얼리스틱한 이미지를 생성하는 방법을 소개합니다. 이는 기존의 데이터 수집 및 주석 작업을 감소시키면서 다양한 제어 가능한 데이터셋을 생성할 수 있도록 합니다.

- **Technical Details**: SyntheOcc는 3D 기하학적 정보(geometric information)를 2D 확산 모델에 조건부 입력으로 효과적으로 인코딩하는 새로운 접근법을 제시합니다. 이 방법은 3D 의미론적 다룹면 이미지(semantic multi-plane images, MPIs)를 포함하여, 이미지 생성 과정에서 보다 정밀한 기하학적 제어를 가능하게 합니다.

- **Performance Highlights**: SyntheOcc는 nuScenes 데이터셋에서 평가한 결과, 제어 가능한 점유 데이터셋을 생성하는 데 매우 효과적이며, 이는 인식 모델의 데이터 증강(data augmentation)에 큰 기여를 합니다.



### A Cat Is A Cat (Not A Dog!): Unraveling Information Mix-ups in Text-to-Image Encoders through Causal Analysis and Embedding Optimization (https://arxiv.org/abs/2410.00321)
Comments:
          Accepted to NeurIPS 2024

- **What's New**: 이 논문은 text-to-image (T2I) diffusion 모델의 텍스트 인코더에서 인과적 방식(causal manner)이 정보 편향(information bias) 및 손실(loss)에 미치는 영향을 분석합니다. 기존 연구에서는 주로 denoising 과정을 통해 문제를 해결하려 했으나, 텍스트 임베딩이 T2I 모델에 어떻게 기여하는지에 대한 연구는 부족했습니다. 이 논문에서는 텍스트 임베딩의 기여와 정보 손실 및 첫 번째 언급된 객체로의 편향 이유를 분석하였고, 90.05%의 정보 균형 개선을 달성하는 최적화 방법인 Text Embedding Balance Optimization(TEBOpt)을 제안합니다.

- **Technical Details**: T2I diffusion 모델은 텍스트 인코더, 변량 오토인코더(VAE), 노이즈 제거 UNet으로 구성됩니다. 텍스트 인코더는 주어진 텍스트 프롬프트에 대해 토큰 임베딩 및 위치 임베딩을 통해 텍스트 히든 상태를 얻고, self-attention 메커니즘과 causal masking 방식으로 텍스트 임베딩을 계산합니다. 이러한 인과적 방식 덕분에, 모든 토큰은 이전 토큰으로부터만 정보를 받아들이며, 이는 정보 편향을 유발하게 됩니다. 또한, 우리는 정보 손실을 보다 정확하게 측정하는 자동 평가 메트릭을 제안합니다.

- **Performance Highlights**: TEBOpt는 정보 균형을 90.05% 개선하여 T2I 모델의 정보 손실 문제를 효과적으로 해결합니다. 또한, 제안된 평가 메트릭은 기존 CLIP 방법보다 81%의 일치율로 인간 평가와의 정확성을 증명하며, 생성된 이미지에서 특정 객체의 존재 및 정확성을 효과적으로 측정합니다.



### PointAD: Comprehending 3D Anomalies from Points and Pixels for Zero-shot 3D Anomaly Detection (https://arxiv.org/abs/2410.00320)
Comments:
          NeurIPS 2024

- **What's New**: 이 논문은 ZS(Zero-shot) 3D anomaly detection의 새로운 접근법인 PointAD를 소개합니다. PointAD는 CLIP의 강력한 일반화 능력을 활용하여 보지 못한 3D 객체에서 이상치를 인식합니다. 이 접근법은 점과 픽셀로부터 3D 이상치를 이해할 수 있는 통합된 프레임워크를 제공합니다.

- **Technical Details**: PointAD는 3D 이상치를 복수의 2D 렌더링으로 변환하고, 이를 통해 3D 공간으로 다시 투영합니다. 하이브리드 표현 학습(hybrid representation learning)을 통해 3D와 2D의 학습 가능한 텍스트 프롬프트를 최적화합니다. 포인트와 픽셀 표현 간의 협력 최적화를 통해, 모델은 보지 못한 다양한 3D 객체의 이상치 패턴을 더 잘 이해할 수 있습니다.

- **Performance Highlights**: PointAD는 다양한 보지 못한 객체에서 ZS 3D anomaly detection의 우수성을 보여주며, 일부 비지도 SOTA 방법을 초월하여 3D 이상치를 탐지하고 분할하는 데 성공했습니다.



### Ask, Pose, Unite: Scaling Data Acquisition for Close Interactions with Vision Language Models (https://arxiv.org/abs/2410.00309)
Comments:
          Project webpage: this https URL

- **What's New**: 이 논문에서는 Human Mesh Estimation (HME) 분야에서의 데이터 부족 문제를 해결하기 위해, Large Vision Language Models (LVLMs)을 활용하여 참조 데이터와 가상의 참조 메시를 생성하는 새로운 방법을 도입하였습니다.

- **Technical Details**: 제안된 방법인 Ask Pose Unite (APU) 데이터세트는 6,200개 이상의 인간 메시 쌍을 포함하여 다양한 유형의 상호작용을 다룹니다. 이를 통해 HME 모델을 학습시키고, 테스트 시 최적화를 통해 메시 추정의 정확도를 향상시킵니다. 또한, 이 방법은 수동 주석의 필요성을 줄이며, 사실적인 인물 간의 상호작용을 반영하는 포괄적인 데이터세트를 제공합니다.

- **Performance Highlights**: 새로 생성된 APU 데이터세트를 이용하여 diffusion 기반의 contact prior를 훈련시키는 실험을 진행했으며, 이를 통해 이전에 보지 못한 상호작용에 대한 메쉬 추정 정확도가 유의미하게 개선되었음을 보여주었습니다.



### RadGazeGen: Radiomics and Gaze-guided Medical Image Generation using Diffusion Models (https://arxiv.org/abs/2410.00307)
- **What's New**: 이번 연구에서는 RadGazeGen이라는 새로운 프레임워크를 제안하여 방사선 전문의의 시선 패턴과 방사선학적 특성 맵(radiomic feature maps)을 텍스트-이미지 확산 모델에 통합하여 높은 품질의 의료 이미지를 생성합니다.

- **Technical Details**: RadGazeGen은 두 개의 모듈로 구성되어 있으며, Rad-CN은 방사선학적 특성 맵을 입력으로 사용하여 해부학적으로 정확하고 질병 정보를 반영한 CXR 이미지를 생성합니다. HVA-CN은 방사선 전문의의 시선 패턴을 입력으로 받아 이 정보를 Rad-CN에 전달하여 질병 패턴 및 위치 인식이 가능한 CXR 이미지를 생성합니다.

- **Performance Highlights**: RadGazeGen은 REFLACX 데이터셋에서 이미지 생성의 품질 및 다양성을 평가한 결과 뛰어난 성능을 보였으며, CheXpert 테스트 세트에서 생성된 이미지의 분류 성능과 MIMIC-CXR-LT 테스트 세트에서의 긴 꼬리 학습(long-tailed learning) 성능에서도 우수한 결과를 보여 주목받고 있습니다.



### GSPR: Multimodal Place Recognition Using 3D Gaussian Splatting for Autonomous Driving (https://arxiv.org/abs/2410.00299)
Comments:
          8 pages, 6 figures

- **What's New**: 이 논문에서는 3D Gaussian Splatting 기반의 다중 모드 장소 인식 신경망 GSPR을 제안합니다. GSPR은 다중 시점 RGB 이미지와 LiDAR 포인트 클라우드를 결합하여 시공간적으로 통합된 장면 표현을 만들어냅니다.

- **Technical Details**: GSPR은 3D 그래프 컨볼루션(3D graph convolution)과 트랜스포머(transformer)를 기반으로 설계되어 고차원 시공간 특징과 전역 설명자를 추출합니다. 이 방법은 Multimodal Gaussian Splatting(MGS) 기술을 활용하여 LiDAR 포인트 클라우드와 RGB 이미지를 조화롭게 통합합니다.

- **Performance Highlights**: nuScenes 데이터셋에서의 실험 결과, 제안된 방법은 기존의 단일 및 다중 모드 방법들보다 뛰어난 장소 인식 성능을 보여주었으며, 미지의 주행 시나리오에 대한 강력한 일반화 능력을 유지하는 것으로 나타났습니다.



### Delving Deep into Engagement Prediction of Short Videos (https://arxiv.org/abs/2410.00289)
Comments:
          Accepted to ECCV 2024. Project page: this https URL

- **What's New**: 본 연구는 소셜 미디어 플랫폼에서의 사용자 생성 콘텐츠(UGC) 짧은 동영상의 인기도를 예측하는 도전에 대해 별도의 접근 방식을 취하고 있습니다. 특히, 제한된 사용자 상호작용으로 새롭게 게시된 동영상의 참여도 예측을 심층적으로 조사하였습니다.

- **Technical Details**: 연구에서는 90,000개의 실제 UGC 짧은 동영상으로 구성된 새로운 데이터셋인 SnapUGC를 소개하며, 두 가지 새로운 지표인 Normalized Average Watch Percentage (NAWP)와 Engagement Continuation Rate (ECR)를 통해 동영상 참여도를 설명합니다. 또한, 비디오 캡션, 배경 음악, 제목 등의 다양한 멀티모달 특징을 통합하여 동영상의 참여도를 모델링합니다.

- **Performance Highlights**: 제안된 방법은 짧은 동영상의 콘텐츠만으로 참여도를 예측하는 능력을 보여주며, 특히 콜드 스타트 상황에서의 유의미한 결과를 도출하였습니다. 이는 고품질 UGC 동영상 추천의 개선 가능성을 제시하며, 콘텐츠 제작자에게 긍정적인 영향을 미칠 수 있습니다.



### Performance Evaluation of Deep Learning-based Quadrotor UAV Detection and Tracking Methods (https://arxiv.org/abs/2410.00285)
- **What's New**: 최근 무인 항공기(UAV)의 사용이 증가함에 따라 이들의 감지 및 추적 기술에 대한 연구가 활발히 진행되고 있습니다. 본 논문은 YOLOv5와 YOLOv8 시리즈를 포함한 최첨단 딥러닝 모델의 성능을 비교 분석하여 UAV의 조기 탐지 및 추적에 대한 문제를 다룹니다.

- **Technical Details**: 논문에서는 YOLOv5와 YOLOv8 모델을 사용하여 UAV 탐지 및 추적 임무를 수행하며, BoT-SORT 및 Byte Track과 같은 추적 시스템을 통합하여 신뢰성 높은 모니터링을 실현합니다. DUT 데이터세트를 통해 검증한 결과, YOLOv5 모델이 탐지 정확도에서 전반적으로 YOLOv8 모델을 초과하는 반면, YOLOv8 모델은 덜 두드러진 객체 인식에서 우수한 성능을 보였습니다. BoT-SORT는 Byte Track에 비해 더 높은 IoU와 낮은 중심 에러를 기록하며 더 정확하고 안정적인 추적을 입증했습니다.

- **Performance Highlights**: YOLOv5 모델은 높은 탐지 정확도를 기록하며 YOLOv8 모델에 비해 전반적으로 우수한 성능을 보였습니다. BoT-SORT는 Byte Track보다 우수한 추적 성능을 제공하고, 다양한 환경에서도 신뢰할 수 있는 모니터링을 가능하게 합니다.



### On Large Uni- and Multi-modal Models for Unsupervised Classification of Social Media Images: Nature's Contribution to People as case study (https://arxiv.org/abs/2410.00275)
Comments:
          15 pages, 9 figures

- **What's New**: 이 논문에서는 소셜 미디어에서 공유되는 이미지의 의미 있는 클러스터링(task of grouping) 문제를 해결하기 위해 최근의 대형 모델들인 Large Visual Models (LVM), Large Language Models (LLM), Large Visual Language Models (LVLM)을 활용한 다양한 접근 방법을 제안하고 있습니다.

- **Technical Details**: 논문에서는 Cultural Ecosystem Services (CES) 문제를 다루며, 이를 위해 FLIPS라는 데이터셋을 구축했습니다. LVM과 LVLM을 활용한 다양한 방법을 평가하는 실험을 진행했으며, 특히 최적 결과를 도출한 모델로는 DINOv2(LVM)와 GPT-4(LVLM)가 있습니다. 전처리 과정에서는 prompt engineering을 통해 이미지 분류를 수행했습니다.

- **Performance Highlights**: 최상위 성능을 보인 방법은 소량의 라벨링된 데이터셋에서 파인튜닝된 LVM DINOv2와 간단한 prompt를 사용한 LVLM 모델인 GPT-4로 나타났습니다. 이로 인해, 사회적 이미지를 CES 클러스터로 분류하는 효율적인 방법이 확인되었습니다.



### KPCA-CAM: Visual Explainability of Deep Computer Vision Models using Kernel PCA (https://arxiv.org/abs/2410.00267)
Comments:
          5 pages, 4 figures, Published to IEEE MMSP 2024

- **What's New**: 본 연구는 Convolutional Neural Networks (CNNs)의 해석 가능성을 증대하기 위해 KPCA-CAM이라는 새로운 기술을 도입합니다. KPCA-CAM은 기존 Class Activation Maps (CAMs)을 개선하고 비선형 관계를 효과적으로 포착하기 위해 Kernel Principal Component Analysis (Kernel PCA)를 활용합니다.

- **Technical Details**: KPCA-CAM은 CNN 활성화에서 비선형 관계를 포착하기 위해 커널 함수를 사용하여 데이터를 고차원 공간으로 매핑합니다. 이 기술은 Eigen-CAM에서 사용되었던 기존 PCA의 한계를 극복하고 더 정확한 데이터 표현을 제공합니다. 또한 KPCA-CAM은 다양한 커널 함수를 사용하여 데이터의 여러 측면을 포착할 수 있습니다.

- **Performance Highlights**: ILSVRC 데이터셋에 대한 실험 결과, KPCA-CAM은 기존 CAM 알고리즘에 비해 더 정밀한 활성화 맵을 생성하여 모델의 추론 과정을 보다 명확하게 제공합니다. 이로 인해 연구자와 실무자가 CNN의 의사 결정 프로세스를 보다 깊이 이해할 수 있는 강력한 도구를 갖게 되었습니다.



### Class-Agnostic Visio-Temporal Scene Sketch Semantic Segmentation (https://arxiv.org/abs/2410.00266)
- **What's New**: 이번 연구에서는 Class-Agnostic Visio-Temporal Network (CAVT)를 제안하며, 이는 장면 스케치의 의미론적 분할을 위한 새로운 접근 방식입니다. CAVT는 기존의 스케치 세분화 방법들이 스케치를 비트맵 이미지로 처리하는 것과는 달리, 개별 객체를 인식하고 개별적인 유니크한 인스턴스를 수행합니다.

- **Technical Details**: CAVT는 클래스에 구애 받지 않는 객체 탐지 모듈을 이용해 장면 내 개체들을 감지하고, 후처리 모듈을 통해 인스턴스의 오브젝트 스트로크를 그룹화합니다. 이 방법은 장면 스케치 내 인스턴스와 스트로크 단위로 분할을 수행하는 최초의 연구이자, RGB 컬러링 기법을 활용하여 스트로크의 시간 정보를 보존합니다.

- **Performance Highlights**: FrISS 데이터셋에서의 실험 결과, 제안한 방법이 현재 상태의 선도적 장면 스케치 세분화 모델보다 우수한 성능을 보였습니다. 이는 1,000개의 장면 스케치를 포함하여 403개의 객체 클래스를 커버하고, 조밀한 주석이 있는 데이터셋으로, 미래의 스트로크 기반 연구를 촉진할 수 있습니다.



### Procedure-Aware Surgical Video-language Pretraining with Hierarchical Knowledge Augmentation (https://arxiv.org/abs/2410.00263)
Comments:
          Accepted at the 38th Conference on Neural Information Processing Systems (NeurIPS 2024) Main Track

- **What's New**: 이 연구는 수술 비디오-언어 사전학습(VLP)의 새로운 접근 방식을 제안하며, 이를 통해 텍스트 정보 손실 및 공간-시간적 도전 과제를 해결하고자 합니다. 이를 위한 새로운 프레임워크인 Procedure-Encoded Surgical Knowledge-Augmented Video-Language Pretraining (PeskaVLP)를 개발했습니다.

- **Technical Details**: PeskaVLP는 대형 언어 모델(LLM)을 이용하여 수술 개념을 정제하고 풍부하게 만들어, 다각적인 언어 감독을 제공하며 오버피팅(overfitting)의 위험을 줄입니다. 이 프레임워크는 계층적 지식 증강(hierarchical knowledge augmentation) 방법을 통해 수술 절차에 고유한 공간-시간적 특성을 이해하고, 동적 시간 왜곡(Dynamic Time Warping, DTW) 기반의 손실 함수로 비디오 프레임과 텍스트 간의 시간적 정렬을 학습합니다.

- **Performance Highlights**: 광범위한 공개 수술 장면 이해(surgical scene understanding) 및 크로스모달 검색(cross-modal retrieval) 데이터셋에서 광범위한 실험 결과, 제안한 방법이 제로샷(zero-shot) 전이 성능을 크게 향상시키고, 수술 장면 이해의 추가 발전을 위한 일반적인 시각적 표현을 제공합니다.



### ImmersePro: End-to-End Stereo Video Synthesis Via Implicit Disparity Learning (https://arxiv.org/abs/2410.00262)
- **What's New**: 이 논문에서는 단일 시점 비디오를 스테레오 비디오로 변환하기 위해 설계된 혁신적인 프레임워크인 ImmersePro를 소개합니다. 이 프레임워크는 비디오 데이터에서 공간-시간적 주의 메커니즘을 활용한 불균형 분기(disparity branch)와 맥락 분기(context branch)를 포함하는 이중 가지 구조를 이용합니다.

- **Technical Details**: ImmersePro은 명시적 불균형 지도 없이 비디오 시퀀스에서 스테레오 쌍을 생성할 수 있게 하는 암묵적 불균형 안내(implicit disparity guidance)를 사용합니다. 이 프레임워크는 700만개 이상의 스테레오 쌍을 갖춘 YouTube-SBS 데이터 세트를 포함하여 스테레오 비디오 생성 모델의 훈련 및 벤치마킹을 위한 기반을 제공합니다.

- **Performance Highlights**: ImmersePro는 기존 방법들에 비해 스테레오 비디오의 품질을 높이는 데 효과적이며, 가장 경쟁력 있는 단일 뷰에서 스테레오로 변환하는 방법과 비교했을 때 L1 기준에서 11.76%, SSIM 기준에서 6.39%, PSNR 기준에서 5.10%의 향상을 보여줍니다.



### MM-Conv: A Multi-modal Conversational Dataset for Virtual Humans (https://arxiv.org/abs/2410.00253)
- **What's New**: 이 논문에서는 VR 헤드셋을 사용하여 AI2-THOR 물리 시뮬레이터 내에서 참가자 간의 대화를 기록한 새로운 데이터셋을 소개합니다. 이 데이터셋은 상황 중심의 제스처 생성을 위한 데이터 기반의 연구를 확대하는 데 중점을 두고 있습니다.

- **Technical Details**: 이 데이터셋은 대화 중 수집된 동작 캡처, 음성, 시선, 장면 그래프 등 다양한 모달리티를 포함하며, 총 6.7시간의 동기화된 데이터가 수집되었습니다. 참가자들은 다양한 공간 좌표 설정을 기반으로 한 대화 상황에 참여하였으며, VR과 모션 캡처 기반 시스템을 통해 기록되었습니다.

- **Performance Highlights**: 이 데이터셋은 제스처 생성 모델의 개발과 이해를 향상시키는 데 기여할 수 있는 포괄적인 자원을 제공합니다. Spatial control signals를 도입한 새로운 확산 기반 인간 모션 생성 모델과 결합하면, 환경에 반응하는 더 정교한 제스처 생성 모델 개발에도 기여할 수 있을 것으로 기대됩니다.



### OpenAnimals: Revisiting Person Re-Identification for Animals Towards Better Generalization (https://arxiv.org/abs/2410.00204)
- **What's New**: 이번 논문에서는 동물 재식별(animal re-identification)의 복잡한 문제를 다루기 위해 OpenAnimals라는 코드베이스를 소개합니다. 이 코드베이스는 동물 재식별을 위해 특별히 설계되었으며, 다양한 동물 종에 대한 연구를 촉진합니다.

- **Technical Details**: OpenAnimals는 PyTorch 기반으로 구축되었으며, 각 동물 종에 대해 재식별 작업을 지원하도록 설계되었습니다. 또한, BoT, AGW, SBS, MGN과 같은 기존 인물 재식별(person re-identification) 방법을 동물 재식별 벤치마크에서 재검토하여 성능을 평가했습니다. ARBase라는 강력한 기본 모델은 이러한 연구 소스를 기반으로 하여 개발되었습니다.

- **Performance Highlights**: 실험 결과, ARBase는 HyenaID와 WhaleSharkID 벤치마크에서 각각 14.54%와 9.90% 향상된 rank-1 정확도를 기록하며 기존 방법들보다 일관되게 우수한 성능을 보였습니다.



### DreamStruct: Understanding Slides and User Interfaces via Synthetic Data Generation (https://arxiv.org/abs/2410.00201)
Comments:
          ECCV 2024

- **What's New**: 이 논문에서는 장애인을 돕기 위해 기계가 슬라이드 및 사용자 인터페이스와 같은 구조화된 시각 정보를 이해할 수 있도록 하는 방법을 제안합니다. 특히, 수작업 데이터 수집과 주석 달기의 필요를 줄이기 위해 코드 생성을 이용하여 합성 구조화된 시각 자료를 생성하는 방법을 소개합니다.

- **Technical Details**: 이 방법은 내장된 라벨이 포함된 데이터셋을 생성하여 몇 가지 인적 주석이 달린 예제만으로도 모델을 훈련할 수 있게 합니다. 실제로, 이 기술은 시각 요소 인식, 시각적 내용 설명, 시각 콘텐츠 유형 분류라는 세 가지 작업에서 성능 향상을 보여줍니다.

- **Performance Highlights**: 세 가지 작업(시각 요소 인식, 시각적 내용 설명, 시각 콘텐츠 유형 분류)에서 성능 향상을 달성하였으며, 이는 장애인 접근성을 높이는 데 기여할 것으로 기대됩니다.



### EEG Emotion Copilot: Pruning LLMs for Emotional EEG Interpretation with Assisted Medical Record Generation (https://arxiv.org/abs/2410.00166)
Comments:
          8 pages, 9 figures

- **What's New**: 본 논문은 EEG (Electroencephalogram) 신호를 기반으로 정서 상태를 인식하고 개인화된 진단 및 치료 제안을 생성하기 위해 경량 대형 언어 모델(LLM)을 활용하는 시스템인 EEG Emotion Copilot을 제안합니다. 이 시스템은 전자 의료 기록의 자동화를 지원하며 사용자의 직관적인 상호작용을 개선하는 것을 목표로 합니다.

- **Technical Details**: EEG Emotion Copilot은 EEG 기반 감정 인식을 위한 데이터 프레임워크 구축, 모델 프루닝(model pruning), 교육 및 배포 전략을 포함하는 체계적인 접근 방식을 제공합니다. 특히, EEG 신호의 데이터 압축 및 경량화된 LLM의 로컬 실행을 위해 설계되었으며, 개인 정보 보호와 윤리적 데이터 처리에 중점을 둡니다.

- **Performance Highlights**: 이 시스템은 정서 인식의 정확성을 강조하며, 실시간 성능 개선을 위해 신호 전처리와 웨이브렛 변환 (wavelet transformation)과 같은 방식을 통해 EEG 신호를 효과적으로 처리합니다. 개인화된 치료 계획 및 진단 제공을 통해 정신 건강 진단의 혁신적인 접근 방식으로 의료 분야의 AC (Affective Computing) 적용을 발전시키고자 합니다.



### CVVLSNet: Vehicle Location and Speed Estimation Using Partial Connected Vehicle Trajectory Data (https://arxiv.org/abs/2410.00132)
- **What's New**: 본 연구는 차량 위치 및 속도를 실시간으로 추정하기 위한 새로운 네트워크인 CVVLSNet을 제안합니다. 이 네트워크는 연결된 차량(CV) 데이터를 부분적으로 사용하여 비연결 차량(NC)의 정보를 추정합니다. 특히 시각 탐지기를 사용하지 않고도 차량의 상태를 효과적으로 분석할 수 있는 방법을 제시합니다.

- **Technical Details**: CVVLSNet은 Coding-Rate TransformEr (CRATE) 네트워크를 기반으로 하여 차량 위치와 속도를 동시에 추정합니다. 도로 셀 점유율(RCO) 방법을 사용하여 차량 상태 정보를 표현하며, 이 정보를 융합하여 시공간적 상호작용을 통합합니다. 또한, 물리적인 차량 크기 제약을 손실 함수에 반영하여 더욱 정밀한 추정을 가능하게 합니다.

- **Performance Highlights**: CVVLSNet은 다양한 CV 침투율, 신호 타이밍 및 용량 대비 볼륨 비율 하에서 기존의 추정 방법보다 현저히 우수한 성능을 보여주었습니다. 실험 결과, 이 방법은 저조도 환경에서도 뛰어난 견고성을 유지하며, 수학적으로 해석 가능성이 높은 특성을 가지고 있습니다.



### ACE: All-round Creator and Editor Following Instructions via Diffusion Transformer (https://arxiv.org/abs/2410.00086)
- **What's New**: 본 논문에서는 ACE(All-round Creator and Editor)라는 새로운 확산 모델(diffusion model)을 제안합니다. 이 모델은 다양한 시각 생성 작업에서 뛰어난 성능을 발휘하며, 텍스트 기반 시각 생성 작업을 넘어선 멀티 모달(multi-modal) 조건을 지원합니다.

- **Technical Details**: ACE 모델은 Long-context Condition Unit (LCU)이라는 통합 조건 형식을 도입하고, Transformer 기반의 새로운 확산 모델을 사용하여 LCU를 입력으로 활용하는 방식으로 설계되었습니다. 이를 통해 다양한 생성 및 편집 작업을 공동 훈련(joint training)할 수 있도록 합니다.

- **Performance Highlights**: 광범위한 시각 생성 작업에서 ACE 모델은 기존 전문가 모델과 비교할 만한 성능을 보였습니다. 이 모델은 단일 모델을 통해 이미지 생성에 대한 모든 상호작용 요청에 응답할 수 있는 멀티 모달(chat system) 시스템을 쉽게 구축할 수 있는 장점을 제공합니다.



### Multimodal Power Outage Prediction for Rapid Disaster Response and Resource Allocation (https://arxiv.org/abs/2410.00017)
Comments:
          7 pages, 4 figures, 1 table

- **What's New**: 이번 연구에서는 주요 허리케인 전후의 야간 조명(NTL)과 정전 심각도 및 위치를 예측하기 위한 새로운 시각적 시공간 프레임워크를 제안합니다. 이를 통해 저개발 지역의 에너지 인프라 개선을 위한 인식을 제고하고자 합니다.

- **Technical Details**: 우리의 프레임워크는 Visual-Spatiotemporal Graph Neural Network (VST-GNN)이라는 신경망 구조를 활용하여 이미지에서 공간적 및 시간적 일관성을 학습합니다. 이 모델은 그래프 G=(V,E)의 노드로 각 카운티를 고려하여 정전 심각도를 예측하며, 과거의 그래프 신호와 입력 이미지 시퀀스를 기반으로 미래의 그래프 신호를 예측하는 기능을 갖추고 있습니다.

- **Performance Highlights**: 실험을 통해 제안된 VST-GNN의 효과성을 검증하였으며, 여러 범위의 데이터에 대한 평가를 통해 모델의 견고성을 입증했습니다. 이 연구는 향후 정책 결정자와 커뮤니티 이해 관계자들이 재난 대응 및 에너지 자원 배분에 있어 중요한 참고자료가 될 것입니다.



### Language-centered Human Activity Recognition (https://arxiv.org/abs/2410.00003)
- **What's New**: 이번 논문에서는 Inertial Measurement Unit (IMU) 센서를 활용한 Human Activity Recognition (HAR) 문제를 해결하기 위해 LanHAR라는 새로운 시스템을 제안합니다. 이 시스템은 Large Language Models (LLMs)를 통해 센서 데이터와 활동 레이블의 의미적 해석을 생성하여 cross-dataset HAR의 과제를 처리합니다. 이를 통해 데이터 간의 이질성을 줄이고 새로운 활동 인식의 가능성을 높입니다.

- **Technical Details**: LanHAR는 다음 세 가지 주요 구성 요소로 이루어져 있습니다: (i) Generation: LLM들이 높은 품질의 의미적 출력을 생성하도록 유도하는 프롬프트 설계 및 반복적 재생성 방법; (ii) Alignment: 센서 데이터와 활동 레이블 간의 해석 정렬을 촉진하기 위한 의미적 해석 정렬 모듈 설계; (iii) Deployment: 모바일 장치에서의 사용을 위해 경량화된 두 단계의 훈련 및 추론 프레임워크 설계. 이 시스템은 LLM의 기능을 모바일 디바이스에 전이하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과는 LanHAR가 cross-dataset HAR 및 새로운 활동 인식에서 기존의 최첨단 방법들보다 유의미하게 뛰어난 성능을 보였음을 보여줍니다. 특히, 정확도에서 7.21% 향상, F1 점수에서 13.31% 향상을 이루었고, 새로운 활동 인식에서는 74.65% 향상을 달성했습니다.



### Optimizing Drug Delivery in Smart Pharmacies: A Novel Framework of Multi-Stage Grasping Network Combined with Adaptive Robotics Mechanism (https://arxiv.org/abs/2410.00753)
- **What's New**: 본 논문에서는 다양한 형태와 겹쳐진 약물을 효과적으로 처리하기 위한 로봇 기반 스마트 약국의 새로운 프레임워크를 제안합니다. 이 프레임워크는 멀티 스테이지 그랩핑 네트워크와 적응형 로보틱 메커니즘을 결합하여 로봇 팔의 복잡한 환경에서도 안정적으로 약물을 집을 수 있도록 합니다.

- **Technical Details**: 제안된 시스템은 향상된 Super-Resolution Convolutional Neural Network (SRCNN) 알고리즘을 사용하여 이미지를 전처리한 후, YOLOv5+E-A-SPPFCSPC+BIFPNC (YOLO-EASB) 인스턴스 분할 알고리즘으로 약물 분할을 수행합니다. 이후 Improved Adaptive Feature Fusion and Grasp-Aware Network (IAFFGA-Net)를 통해 정확한 집기 동작을 수행하며, 이를 위해 시간 최적화 로봇 팔 궤적 계획 알고리즘을 개발하여 3-5-3 보간법을 사용하여 효율적인 궤적을 보장합니다.

- **Performance Highlights**: 실험 결과, 제안된 멀티 스테이지 그랩핑 네트워크가 스마트 약국 운영의 최적화를 이루며 복잡한 환경에서도 효과적인 집기를 수행하는 뛰어난 적응성과 효과성을 보여주었습니다.



### WALINET: A water and lipid identification convolutional Neural Network for nuisance signal removal in 1H MR Spectroscopic Imaging (https://arxiv.org/abs/2410.00746)
- **What's New**: 본 연구에서는 고해상도 1H-MRSI에서의 물 및 지방 신호 제거를 위한 심층 학습 방법을 소개합니다. 새로운 WALINET 네트워크는 기존의 방법들보다 더 빠르고 효과적으로 신호를 처리할 수 있습니다.

- **Technical Details**: WALINET(Network for Water and Lipid) 은 수정된 Y-NET 기반의 심층 신경망으로, 1H-MRSI 스펙트럼에서의 물 및 지방 신호 제거를 수행합니다. 시뮬레이션 및 실제 데이터에서 NMRSE, SNR, CRLB, FWHM 등의 메트릭을 사용하여 비교 평가하였습니다.

- **Performance Highlights**: WALINET은 8초 이내에 고해상도 전체 뇌 1H-MRSI 데이터를 처리할 수 있으며, 이는 기존 HLSVD+L2 방법의 42분과 비교하여 매우 빠릅니다. 또한 WALINET은 지방 제거에서 41% 낮은 NRMSE를 기록하고, 메타볼라이트 신호 보존에서도 71% 낮은 NRMSE와 155% 높은 SNR, 50% 낮은 CRLB를 보여주었습니다.



### VideoCLIP-XL: Advancing Long Description Understanding for Video CLIP Models (https://arxiv.org/abs/2410.00741)
Comments:
          EMNLP 2024 Main conference

- **What's New**: VideoCLIP-XL (eXtra Length) 모델은 비디오 CLIP 모델의 긴 설명 이해 능력을 확장하는 데 초점을 맞추고 있습니다. 이를 위해 자동 데이터 수집 시스템을 구축하여 200만 개 이상의 비디오-긴 설명 쌍으로 구성된 대규모 VILD 사전 훈련 데이터셋을 수집하였습니다.

- **Technical Details**: TPCM (Text-similarity-guided Primary Component Matching)을 도입하여 고차원 특징 공간의 분포 변화에 적응할 수 있도록 하였으며, Detail-aware Description Ranking (DDR)와 Hallucination-aware Description Ranking (HDR)이라는 새로운 작업을 통해 비디오와 긴 설명 간의 관계를 효과적으로 학습하도록 하였습니다.

- **Performance Highlights**: 비디오CLIP-XL 모델은 기존의 여러 벤치마크에서 우수한 성능을 보였으며, 특히 LVDR (Long Video Description Ranking) 벤치마크를 통해 긴 설명에 대한 이해 능력을 종합적으로 평가하였습니다. 실험 결과, 비디오CLIP-XL은 최신 기술 경쟁 모델들보다 뛰어난 성능을 발휘했습니다.



### A Low-Cost, High-Speed, and Robust Bin Picking System for Factory Automation Enabled by a Non-Stop, Multi-View, and Active Vision Schem (https://arxiv.org/abs/2410.00706)
- **What's New**: 본 논문에서는 공장 자동화의 bin picking 시스템에서의 문제점을 해결하기 위해 'sensor on hand' 구성에서 다중 뷰 및 능동 비전(active vision) 방법을 통합한 새로운 접근 방식을 제안합니다.

- **Technical Details**: 제안된 시스템은 3D 센서를 활용하여 여러 방향에서 데이터를 수집하며, 이를 통해 3D 데이터를 통합하는 과정에서 발생할 수 있는 저속 문제를 해결합니다. 3D 센싱과 로봇 동작을 밀접하게 연관시켜, 로봇이 동작하는 동안 병렬적으로 빠른 센싱을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, 센싱 과정은 최대 1.682초 이내에 완료되며 평균 피킹 완료율은 97.75% 이상입니다. 로봇의 동작과 센싱을 병렬화하여, 평균적으로 0.635초의 타크(takt) 시간 내에 센싱이 이루어집니다.



### Arges: Spatio-Temporal Transformer for Ulcerative Colitis Severity Assessment in Endoscopy Videos (https://arxiv.org/abs/2410.00536)
Comments:
          12 pages, 2 figures, 5 tables, accepted at MLMI, MICCAI

- **What's New**: 이번 논문에서는 위염병(UC)의 내시경 비디오에서 질병 중증도를 평가하기 위한 새로운 딥러닝 프레임워크인 "Arges"를 제안합니다. 이 모델은 공간-시간(spatio-temporal) 정보를 통합하여 내시경 비디오의 질병 중증도를 더욱 정확하게 추정할 수 있게 합니다.

- **Technical Details**: Arges 프레임워크는 positional encoding을 통해 공간-시간(spatio-temporal) 정보를 통합하는 transformer 기반 분류기를 포함하고 있습니다. ArgesFM이라는 강력한 기반 모델을 사용하여 61M 프레임의 대규모 데이터에서 학습한 후, 질병 중증도 점수를 추정하기 위한 추가적인 분류기를 적용합니다.

- **Performance Highlights**: 실험 결과, MES 점수에서 F1 점수가 4.1% 향상되었으며, UCEIS 구성 점수에서도 각각 18.8%, 6.6%, 3.8%의 개선을 보였습니다. 추가적으로, 이전에 본 적이 없는 임상 시험 데이터에 대한 유망한 검증 결과도 나타났습니다.



### Deep Model Interpretation with Limited Data : A Coreset-based Approach (https://arxiv.org/abs/2410.00524)
- **What's New**: 이 논문에서는 모델 해석(Method Interpretation)의 계산 비용 문제를 해결하기 위해 코어셋 기반 프레임워크를 제안합니다. 이를 통해 대규모 데이터셋에서 대표적인 서브셋을 샘플링하여 해석 작업을 수행할 수 있게 됩니다.

- **Technical Details**: 코어셋(selection: coreset) 선택 방법을 이용하여 대규모 데이터셋에서 대표 샘플을 선택하고, 이를 통해 프리트레인(pre-trained) 모델의 관련 특징을 식별합니다. 유사성 기반 평가 프로토콜(similarity-based evaluation protocol)이 도입되어 입력 데이터 양에 대한 모델 해석 방법의 강건성을 평가합니다.

- **Performance Highlights**: 실험 결과, 다양한 해석 방법과 DNN(Deep Neural Network) 모델, 코어셋 선택 방법을 고려한 결과 제안된 프레임워크의 효과성이 입증되었습니다. 대규모 데이터셋을 직접 사용하는 것보다 더 낮은 계산 비용으로 의미 있는 해석 결과를 제공할 수 있음을 보여주었습니다.



### Design and Identification of Keypoint Patches in Unstructured Environments (https://arxiv.org/abs/2410.00521)
Comments:
          12 pages, 8 figures, 7 tables

- **What's New**: 이 논문은 복잡한 환경에서 자율 로봇의 안정적인 작동을 위한 신뢰할 수 있는 타겟 인식을 향상시키기 위해 고안된 새로운 keypoint 디자인과 Superpoint 네트워크의 커스터마이즈를 제안합니다. 이를 통해 이미지의 품질 저하에도 불구하고 견고한 keypoint 식별 성능을 유지할 수 있습니다.

- **Technical Details**: 제안된 방법은 다양한 시점 변환을 고려한 독특한 keypoint 패치를 설계하고, Superpoint 네트워크를 수정하여 이러한 keypoint를 효과적으로 식별합니다. 연구에서는 스케일, 회전, 카메라 투영을 활용하여 keypoint 패치의 독특성을 보장하는 네 가지 디자인을 소개합니다. 또한, 다양한 이미지 저하 효과를 극복하기 위해 전처리된 데이터셋과 학습 파이프라인을 사용합니다.

- **Performance Highlights**: 실제 비디오 테스트를 통해 제안된 방법의 효과성을 입증하였으며, 환경의 다양한 조건에서도 비전 기반 자율 시스템에서 더 높은 정확도의 keypoint 인식을 가능하게 합니다.



### Enhancing Sentinel-2 Image Resolution: Evaluating Advanced Techniques based on Convolutional and Generative Neural Networks (https://arxiv.org/abs/2410.00516)
Comments:
          This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 이 논문은 Sentinel-2 위성의 저해상도 이미지를 고해상도로 변환하기 위해 Super-Resolution (SR) 기법을 활용한 연구를 다룹니다. 특히, CNN 모델과 GAN 기반 접근 방식이 이미지 품질 및 실행 가능성 측면에서 비교되었습니다.

- **Technical Details**: 이 연구에서는 저해상도(LSR) Sentinel-2 이미지와 고해상도(HR) 항공 정사 사진을 포함하는 대표적인 데이터셋을 공공으로 생성하였으며, Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM), Learned Perceptual Image Patch Similarity (LPIPS)와 같은 픽셀 기반 성능 지표를 사용하여 서로 다른 SR 기법들의 이미지를 평가합니다. 또한, GAN 모델의 발전에 따라 ESRGAN과 Real-ESRGAN을 사용하여 고해상도 출력을 생성하는 방법도 탐구하였습니다.

- **Performance Highlights**: 이 논문에서 제안된 GAN 기반 모델은 CNN 기반 접근 방식에 비해 더 선명하고 상세한 이미지를 생성했으며, 특히 quantitative assessment에서 우수한 성능을 보여주었습니다. 따라서, 본 연구는 특정 토지 유형에 국한되지 않고 해당 프레임워크의 잠재력을 강조합니다.



### Pre-training with Synthetic Patterns for Audio (https://arxiv.org/abs/2410.00511)
Comments:
          Submitted to ICASSP'25

- **What's New**: 이 논문에서는 실제 오디오 데이터 대신 합성 패턴(synthetic patterns)을 사용하여 오디오 인코더(audio encoders)를 사전 훈련(pre-train)하는 새로운 프레임워크를 제안합니다.

- **Technical Details**: 제안하는 프레임워크는 두 가지 주요 요소로 구성됩니다. 첫 번째는 Masked Autoencoder (MAE)로, 이는 랜덤하게 마스킹된 입력 데이터를 재구성하는 것으로 학습하는 자기 감독(self-supervised) 학습 프레임워크입니다. 두 번째는 합성 데이터(synthetic data)로, 실제 오디오와 달리 개인 정보 및 라이센스 문제에서 자유롭습니다. 이 조합을 통해 실제 데이터 없이도 일반화된 특징 표현(feature representations)을 학습할 수 있습니다.

- **Performance Highlights**: 논문에서 수행한 13개의 오디오 작업과 17개의 합성 데이터셋을 통한 실험 결과, 제안된 프레임워크가 AudioSet-2M으로 사전 훈련된 모델과 유사한 성능을 달성하며, 이미지 기반 사전 훈련 방법을 일부 초월하는 성과를 보여주었습니다.



### Precise Workcell Sketching from Point Clouds Using an AR Toolbox (https://arxiv.org/abs/2410.00479)
Comments:
          Published in IEEE RO-MAN 2024

- **What's New**: 이 논문은 실제 3D 공간을 포인트 클라우드(point cloud)로 효율적으로 캡처하는 방법과 해당 데이터를 파라메트릭(parametric) 형식으로 변환하는 방안을 다룹니다. 특히, 증강 현실(Augmented Reality) 인터페이스를 이용해 사용자들이 포인트 클라우드를 보완할 수 있는 혁신적인 솔루션을 제안하고 있습니다.

- **Technical Details**: 제안된 방법은 모바일 애플리케이션 형태로, LiDAR 센서를 활용하여 포인트 클라우드를 수집하고 처리합니다. 사용자 인터페이스(UI)는 사용자와 실물 3D 환경 간의 상호작용을 촉진하여, 사용자가 포인트 클라우드를 개선하고 불필요한 데이터를 제거할 수 있도록 돕습니다. 이 방법은 포인트 클라우드의 첫 번째 단계인 바운딩 박스(bounding box) 설정을 통해 사용자가 관심 있는 영역만 캡처할 수 있게 합니다.

- **Performance Highlights**: 제안된 방법은 기존 LiDAR 스캐너 애플리케이션에 비해 평균 오차를 1cm 이내로 줄이는 성과를 보였습니다. 이로써 다양한 산업에서 로봇 자동화를 위해 사용 가능한 더 정확한 포인트 클라우드 데이터를 생성할 수 있게 됩니다.



### Task Success Prediction for Open-Vocabulary Manipulation Based on Multi-Level Aligned Representations (https://arxiv.org/abs/2410.00436)
Comments:
          Accepted for presentation at CoRL2024

- **What's New**: 이번 연구에서는 조작문장(instruction sentences)과 조작 전후의 주관적 이미지(egocentric images)를 기반으로 개방적 어휘 조작(open-vocabulary manipulation)의 성공 예측에 대한 문제를 다룹니다. 기존의 다중 모달 대형 언어 모델(multimodal large language models, MLLMs)이 객체의 세부 특성과 위치 변화의 미세한 변화를 적절히 이해하지 못하는 경우가 많습니다. 이러한 문제를 해결하기 위해, 우리는 Contrastive λ-Repformer를 제안하여 이미지와 조작문장을 정렬하여 테이블탑 조작 작업의 성공을 예측합니다.

- **Technical Details**: Contrastive λ-Repformer는 세 가지 주요 특징을 통합한 다층 정렬 표현(multi-level aligned representation)을 이용합니다. 이러한 특징은 (i) 지역 이미지 정보(local image information)를 보존하는 특징, (ii) 자연어(natural language)와 정렬된 특징, (iii) 자연어를 통해 구조화된 특징입니다. 이러한 접근 방식은 모델이 두 이미지 간의 표현의 차이를 통해 중요한 변화에 집중할 수 있도록 돕습니다. 모델은 성공 예측을 위해 연산을 수행하며, 조작 이미지의 차이와 조작문장과의 alignment를 고려하여 성과를 도출합니다.

- **Performance Highlights**: Contrastive λ-Repformer는 대규모 표준 데이터셋인 RT-1 데이터셋 및 실제 로봇 플랫폼을 기반으로 평가되었습니다. 결과적으로, 우리의 접근 방식은 기존의 MLLM을 포함한 접근 방식보다 뛰어난 성능을 보였으며, 최상의 모델은 대표적인 MLLM 기반 모델에 비해 8.66 포인트의 정확도 향상을 이뤘습니다.



### Posterior-Mean Rectified Flow: Towards Minimum MSE Photo-Realistic Image Restoration (https://arxiv.org/abs/2410.00418)
- **What's New**: 이 논문에서는 Photo-Realistic Image Restoration (PIR) 문제에 대한 새로운 접근 방식을 제안합니다. 특히, 기존의 방법들이 왜곡 손실(distortion loss)과 인식 품질 손실(perceptual quality loss)을 최적화하는 데 초점을 맞춘 반면, 이 연구는 완벽한 인식 지표(perceptual index) 제약 하에서 평균 제곱 오차(Mean Squared Error, MSE)를 최소화하는 최적 추정자(optimal estimator)를 구하는 데 중점을 둡니다.

- **Technical Details**: 저자들은 Posterior-Mean Rectified Flow (PMRF)라는 새로운 알고리즘을 소개합니다. 이 알고리즘은 먼저 후방 평균(posterior mean)을 예측한 다음, 이를 rectified flow 모델을 사용하여 고품질 이미지로 변환합니다. 이 과정에서 필요한 최적 수송 지도(optimal transport map)를 근사합니다. PMRF는 모델이 재구성된 출력과 실제 이미지 간의 MSE를 최소화하도록 학습됩니다.

- **Performance Highlights**: PMRF는 다양한 이미지 복원 작업에서 기존 방법들보다 일관되게 우수한 성능을 나타내며, 이론적으로도 그 유용성이 입증되었습니다. 특히, 고품질 이미지 복원을 위한 새로운 접근 방식으로 자리매김할 가능성을 보여줍니다.



### Domain Aware Multi-Task Pretraining of 3D Swin Transformer for T1-weighted Brain MRI (https://arxiv.org/abs/2410.00410)
Comments:
          ACCV 2024, 14 pages

- **What's New**: 이 연구에서는 3D 뇌 자기공명영상(MRI)에 대한 도메인 인식 멀티태스크 학습을 통해 Swin Transformer를 프리트레인(pretrain)하는 새로운 방법론을 제안합니다. 이는 기존 방법들이 2D 접근법을 3D 데이터에 적용한 것에 비해, 뇌 해부학 및 형태학적 특징을 고려하여 개선된 접근법입니다.

- **Technical Details**: 제안된 방법은 13,687개의 대규모 뇌 MRI 데이터를 활용하여, 이미지 회전, 패치 위치 예측, 마스킹 이미지 모델링과 같은 여러 셀프-슈퍼바이즈드(self-supervised) 작업을 포함하여 멀티태스크 프리트레인 프레임워크를 구축합니다. 이 방식은 3D 데이터의 처리에서 효율성을 높이고, 다양한 패치 크기와 시퀀스 길이에 대한 강건성을 제공합니다.

- **Performance Highlights**: Alzheimer’s disease(AD) 및 Parkinson’s disease(PD) 분류, 그리고 연령 예측 과제로 이루어진 세 가지 다운스트림(task)에서 기존 방법들보다 향상된 성능을 기록했습니다. 제안한 프리텍스트 작업의 효과를 입증하기 위한 에블레이션 연구도 포함되어 있습니다.



### 3DGR-CAR: Coronary artery reconstruction from ultra-sparse 2D X-ray views with a 3D Gaussians representation (https://arxiv.org/abs/2410.00404)
Comments:
          10 pages, 5 figures, Accepted at MICCAI 2024

- **What's New**: 3DGR-CAR은 초희소(ultra-sparse) X선 투영을 기반으로 한 관상동맥 복원 방법으로, 3D Gaussian representation을 활용하여 고효율적이고 정확한 3D 관상동맥 복원을 실현합니다.

- **Technical Details**: 제안된 방법은 Gaussian center predictor (GCP)를 활용하여 노이즈가 많은 초기 Gaussian 중심을 개선하며, U-Net을 통해 관상동맥의 방향 매개변수를 추정하여 3D Gaussian을 초기화 합니다. 이 구조는 2개의 뷰만으로도 3D 관상동맥을 재구성할 수 있게 해줍니다.

- **Performance Highlights**: 실험 결과, 3DGR-CAR은 Voxel 정확도 및 관상동맥의 시각적 품질에서 다른 기존 방법을 크게 초월하였으며, ImageCAS 및 ASOCA 데이터셋에서 수행된 테스트에서 빠른 처리 시간을 달성했습니다.



### Insight: A Multi-Modal Diagnostic Pipeline using LLMs for Ocular Surface Disease Diagnosis (https://arxiv.org/abs/2410.00292)
Comments:
          Accepted to MICCAI 2024. Project Webpage: this https URL

- **What's New**: 본 논문은 시각적 데이터와 임상 메타데이터를 결합하여 안구 표면 질환을 진단하기 위한 혁신적인 다중 모달 진단 파이프라인(MDPipe)을 제안합니다. 기존의 방법에서는 진단을 정해진 클래스 분류 문제로 다루어 임상 변수 간의 관계를 고려하지 못했습니다.

- **Technical Details**: MDPipe는 대형 언어 모델(LLMs)을 활용하여 메이보그래피 이미지를 정량화할 수 있는 형태의 형태학적 데이터로 변환합니다. 그 후, LLM 기반 요약기를 통해 이를 요약하고 임상 보고서로 활용합니다. 이 방식은 LLM의 추론 능력을 실제 임상 진단에 맞춰 개선합니다.

- **Performance Highlights**: MDPipe는 다양한 안구 표면 질병 진단 기준에서 테스트하여 기존의 기준(예: GPT-4)을 능가하며 임상적으로 타당한 진단 근거를 제공합니다.



### Robin3D: Improving 3D Large Language Model via Robust Instruction Tuning (https://arxiv.org/abs/2410.00255)
Comments:
          10 pages

- **What's New**: 이번 논문에서는 Robin3D라는 강력한 3D 대형 언어 모델(3DLLM)을 소개합니다. Robin3D는 Robust Instruction Generation (RIG) 엔진에 의해 생성된 대규모 지침 수행 데이터로 훈련되었습니다. 이 데이터는 긍정적 및 부정적 샘플을 혼합한 Adversarial Instruction-following 데이터와 다양한 스타일의 지침을 포함한 Diverse Instruction-following 데이터 두 가지로 나뉘며, 총 100만 개의 지침 데이터 세트를 구축했습니다.

- **Technical Details**: Robin3D는 Relation-Augmented Projector(RAP)를 통합하여 공간적 이해를 향상시키고, ID-Feature Bonding(IFB)를 통해 객체 참조 및 지면 제어 능력을 강화합니다. 특히, RAP는 객체 중심 특성에 씬 레벨 맥락과 위치 정보를 풍부하게 하여, 다양한 비주얼 지면 데이터를 학습할 수 있는 능력을 높입니다. IFB는 각 ID를 해당 특성과 연결하여 정보의 질을 향상시킵니다.

- **Performance Highlights**: Robin3D는 기존의 3D 다중 모드 학습 벤치마크인 ScanRefer, Multi3DRefer, Scan2Cap, ScanQA, SQA3D에서 최고 성능을 기록했습니다. 특히, Multi3DRefer에서 7.8% 개선, Scan2Cap에서 6.9% 개선을 이루어 내며, 특정 작업에 대한 세부 조정 없이 SOTA(State-of-the-Art) 프레임워크를 달성했습니다.



### Helpful DoggyBot: Open-World Object Fetching using Legged Robots and Vision-Language Models (https://arxiv.org/abs/2410.00231)
Comments:
          Project website: this https URL

- **What's New**: 이번 연구에서는 인도 환경에서 도움이 되는 이동 조작 기술을 위한 새로운 시스템인 Helpful DoggyBot을 제안합니다. 이 시스템은 전면에 장착된 그리퍼와 저수준 제어기를 사용하여, 다양한 내부 환경에서 유용한 작업을 수행할 수 있도록 설계되었습니다.

- **Technical Details**: Helpful DoggyBot은 1-DoF 그리퍼를 사용하여 일상적인 객체를 집어올리며, egocentric depth(자기중심 깊이)와 proprioception(신체위치 감각)을 활용한 저수준 제어기를 시뮬레이션에서 훈련하여 기동성을 높였습니다. Vision-Language Models (VLMs)를 활용하여 객체 식별 및 명령 생성을 수행하며, 이는 목표 객체를 따라가고 추적하는 데 도움을 줍니다.

- **Performance Highlights**: 이 시스템은 두 개의 미지의 환경에서 zero-shot generalization을 수행하여, 예를 들어 사용자의 명령에 따라 무작위로 놓인 인형을 찾아오는 작업을 60%의 성공률로 완료했습니다. 이는 실제 세계 데이터 수집이나 훈련 없이도 가능하여, 다양한 홈 환경에 적응할 수 있는 도움이 되는 4족 보행 로봇의 가능성을 보여줍니다.



### Do Vision-Language Models Really Understand Visual Language? (https://arxiv.org/abs/2410.00193)
- **What's New**: 이 연구는 최근의 Large Vision-Language Models (LVLMs)가 도표와 관련된 복잡한 추론 작업을 수행할 수 있는지 조사합니다. 저자들은 LVLMs의 도표 이해 능력을 평가하기 위해 포괄적인 테스트 수트를 개발하였습니다.

- **Technical Details**: 테스트 수트는 여러 도메인에 걸친 합성 및 실제 도표 세트에서 개념 엔티티와 그 관계에 중점을 둔 다양한 질문을 사용하여 LVLMs의 인식 및 추론 능력을 평가합니다.

- **Performance Highlights**: 세 가지 LVLM들(GPT-4V, GPT-4o, Gemini)을 평가한 결과, 엔티티를 식별하고 추론하는 데는 정확성을 보였으나 관계 이해 능력은 상당히 제한적임을 보여주었습니다. 향상된 도표 이해 성능은 모델의 배경 지식을 활용한 단축키에 기인한다는 것을 발견하였습니다.



### Volumetric Conditional Score-based Residual Diffusion Model for PET/MR Denoising (https://arxiv.org/abs/2410.00184)
Comments:
          Accepted to MICCAI 2024

- **What's New**: 본 논문에서는 Conditional Score-based Residual Diffusion (CSRD) 모델을 제안하며, 3D 볼륨 PET 영상을 효율적으로 처리하기 위한 3D 패치 기반 훈련 전략을 포함하여 PET 이미징의 고유한 특성을 반영하고, 전통적인 방법보다 성능을 크게 향상시킵니다.

- **Technical Details**: CSRD 모델은 점진적인 score function 개선과 3D 훈련 전략을 통해 계산 비용 및 메모리 요구사항을 최적화합니다. 이 모델은 PET 및 MRI 스캔에서의 볼륨 데이터 통합을 통해 공간 일관성과 해부학적 세부사항을 유지합니다. 또한, 이미지의 세부 정보를 보존하면서 더 나은 제거 성능을 보여줍니다.

- **Performance Highlights**: CSRD 모델은 질적으로 및 양적으로 기존의 최첨단 방법을 능가하며, 3D PET 이미지의 볼륨 감소 작업을 3분 이내에 수행할 수 있습니다. 이 모델은 속도와 성능 모두에서 상당한 개선을 나타냅니다.



### Multimodal Alignment of Histopathological Images Using Cell Segmentation and Point Set Matching for Integrative Cancer Analysis (https://arxiv.org/abs/2410.00152)
Comments:
          initial version

- **What's New**: 본 논문에서는 암 연구 및 임상 실습에서 중요한 조직 병리학적 이미지의 멀티모달(모델에 따라 다른 유형의 데이터) 정렬을 위한 새로운 프레임워크를 제시합니다.

- **Technical Details**: 이 프레임워크는 세포 세분화 결과를 활용하여, 세포를 점 집합(Point Sets)으로 처리하고 Coherent Point Drift (CPD) 기법을 사용하여 초기 정렬을 수행한 다음, Graph Matching (GM) 기법으로 정렬을 정교화합니다.

- **Performance Highlights**: 난소암 조직 미세 배열(TMA)을 기반으로 평가한 결과, 높은 정렬 정확도를 달성하여 다중 모달에서 세포 수준의 특징 통합을 가능하게 하였으며, MxIF 데이터를 통해 가상 H&E 이미지를 생성하여 임상 해석을 향상시켰습니다.



### An Overview of the Burer-Monteiro Method for Certifiable Robot Perception (https://arxiv.org/abs/2410.00117)
Comments:
          Accepted to 2024 Robotics: Science and Systems (RSS) Safe Autonomy Workshop

- **What's New**: 이 논문은 Burer-Monteiro 방법(BM)을 소개하며, 이 기술이 로봇 인식 문제를 해결하는 데 어떻게 적용되는지를 다룹니다. BM은 신뢰할 수 있는 최적 솔루션을 신속하게 도출할 수 있는 방법입니다.

- **Technical Details**: Burer-Monteiro 방법은 주로 semidefinite programming(SDP) 완화로 나타나는 비볼록 인식 문제를 해결하는 데 사용됩니다. 이 방법은 일반적인 SDP 프로그램의 저차원 낮은 순위(factorization) 구조를 활용하여 최적화의 계산 비용을 크게 줄입니다. 특히 LICQ(linear independence constraint qualification)가 알고리즘의 신뢰성을 향상시키는 데 중요함을 강조합니다.

- **Performance Highlights**: 이 논문의 목적은 BM의 내용을 통일된 형태로 정리하고, 실용적인 고려 사항을 추가하여 BM 적용 시 진입 장벽을 낮추는 것입니다. 이는 로봇 인식의 안정성과 신뢰성을 확보하는 데 기여할 수 있습니다.



### Fine-tuning Vision Classifiers On A Budg (https://arxiv.org/abs/2410.00085)
Comments:
          8 pages, 5 figures

- **What's New**: 이 논문에서는 라벨러의 정확성에 대한 이전 추정치를 활용하여, 단순 naive-Bayes 모델을 사용하여 실제 라벨을 추정하는 방법인 Ground Truth Extension (GTX)을 소개합니다. 이 방법은 고품질의 데이터를 더 적은 인적 라벨로 수집할 수 있게 해줍니다.

- **Technical Details**: GTX 방법은 3단계로 구성되며, 첫째, 노이즈 라벨을 고려하여 실제 라벨 및 신뢰도를 추정하기 위한 확률적 모델을 제공합니다. 둘째, 전문가가 제공한 소수의 라벨(ground truth labels)을 이용해 라벨러의 정확도를 추정합니다. 셋째, 주어진 신뢰도 수준에서 최대한 많은 실제 라벨 추정을 가능하게 하는 예제를 (노이즈가 포함된) 라벨링하는 전략을 수립합니다.

- **Performance Highlights**: GTX 방법은 여러 조건에서 라벨의 정확성 향상에서 표준 및 가중 다수결 표본보다 뛰어난 성능을 보였으며, 산업 비주얼 분류 작업에 사용한 실험에서도 CNN 모델의 정밀도를 높이는 데 있어 GTX로 추정된 라벨이 효과적이라는 것을 입증했습니다.



### A Survey on Diffusion Models for Inverse Problems (https://arxiv.org/abs/2410.00083)
Comments:
          Work in progress. 38 pages

- **What's New**: 이 논문에서는 pretrained diffusion model을 활용하여 inverse problem을 해결하는 방법에 대한 포괄적인 개요를 제공합니다. 기존 훈련 과정 없이도 고품질 샘플을 생성할 수 있는 가능성을 탐구합니다.

- **Technical Details**: Diffusion models는 unsupervised priors로 사용되어 inverse problem 해결에 활용됩니다. 다양한 문제와 기술에 따라 방법을 분류하는 taxonomy를 소개하며, 이를 통해 각 접근 방식의 연결성을 분석합니다. 특히 latent diffusion model을 사용할 때의 도전 과제와 잠재적 해결책에 대해서도 논의합니다.

- **Performance Highlights**: Diffusion models를 활용한 inverse problem 해결 방법의 실용적 구현을 강조하며, 다양한 과학 분야에서의 적용 가능성을 보여줍니다. 예를 들어, 이미지 복원, MRI 가속화, 오디오 신호 처리 등에서 효과적인 결과를 나타냅니다.



### Graph Residual Noise Learner Network for Brain Connectivity Graph Prediction (https://arxiv.org/abs/2410.00082)
Comments:
          10 pages, 3 figures, 6th Workshop on GRaphs in biomedicAl Image anaLysis

- **What's New**: 본 연구에서는 신경 장애 진단을 개선하기 위한 새로운 그래프 기반의 뇌 그래프 예측 모델인 Graph Residual Noise Learner Network (Grenol-Net)를 제안합니다. 기존의 확산 모델을 활용하되, 그래프의 토폴로지 대칭성을 유지하기 위한 접근 방식을 채택하여, 신경 과학 연구 커뮤니티에서의 협력적 연구 노력을 촉진하고자 합니다.

- **Technical Details**: Grenol-Net은 두 개의 복잡하게 설계된 학습 블록으로 구성됩니다. 첫 번째 블록은 그래프 컨볼루션 블록을 포함하여 소스 그래프 내 연결의 복잡한 상호작용을 드러내며, 서로 다른 노드 간의 메시지 전달 기능을 통해 각 ROI의 독특한 임베딩을 학습합니다. 두 번째 블록은 배치 정규화를 도입하여 타겟 그래프의 분포를 학습하고, 잔여 연결을 통해 타겟에 적용된 노이즈를 분리 및 복구합니다.

- **Performance Highlights**: Grenol-Net은 기존의 신경망 모델에 비해 예측 정확도 및 그래프 연결성 다양성을 유지하면서도 더 나은 성능을 자랑합니다. 예측 과정에서는 그래프의 노드 특성을 정확하게 학습하여, 동일한 파셀레이션 뇌 템플릿에서 파생된 동형(biomorphic) 그래프들 간의 연결성 다양성을 유지하는 데 초점을 맞추었습니다.



### M2Distill: Multi-Modal Distillation for Lifelong Imitation Learning (https://arxiv.org/abs/2410.00064)
Comments:
          Submitted to ICRA2025

- **What's New**: 이 논문에서는 M2Distill이라는 새로운 multi-modal distillation 기반 방법을 소개합니다. 이는 평생 imitation learning 과정에서 시각, 언어, 행동 분포 간 일관된 latent space를 유지하는 데 초점을 맞추고 있습니다.

- **Technical Details**: M2Distill은 다양한 modality 간의 latent representation 변화 조절을 통해 이전 단계와 현재 단계 간의 일관성을 유지하며, Gaussian Mixture Model (GMM) 정책의 불일치를 줄여 갑작스러운 잊어버림을 방지합니다. 이를 통해 학습된 정책이 이전에 배운 작업을 계속 수행할 수 있도록 보장합니다.

- **Performance Highlights**: LIBERO 평생 imitation learning 벤치마크(Campaign)에서 M2Distill은 LIBERO-OBJECT, LIBERO-GOAL, LIBERO-SPATIAL을 포함하여 모든 평가 지표에서 이전의 최첨단 방법들보다 우수한 성능을 보여주었습니다.



### Automated Disease Diagnosis in Pumpkin Plants Using Advanced CNN Models (https://arxiv.org/abs/2410.00062)
Comments:
          10 pages, 8 figures

- **What's New**: 본 논문은 호박 식물 잎에서 발생하는 질병을 분류하기 위한 최신 Convolutional Neural Network (CNN) 모델의 성능을 종합적으로 분석합니다. 특히 ResNet, DenseNet, EfficientNet 모델을 활용하여 5가지 질병을 식별하며, 사전 훈련된 모델을 조정하여 최적의 성능을 이끌어내는 데 중점을 두었습니다.

- **Technical Details**: 호박 잎 질병 데이터셋은 2000개의 고해상도 RGB 이미지로 구성되어 있으며, 5개의 카테고리(정상, 다운y mildew, powdery mildew, mosaic disease, bacterial leaf spot)가 포함되어 있습니다. ResNet-34, DenseNet-121 및 EfficientNet-B7이 최상의 성능을 보였으며 이 모델들은 hyperparameter 최적화를 통해 조정되었습니다. DenseNet-121은 정확도(accuracy)와 계산 복잡성(computational complexity) 두 가지 기준에서 최적 모델로 선정되었습니다.

- **Performance Highlights**: DenseNet-121은 86%의 전반적인 정확도를 달성했으며, ResNet-50을 통한 실험에서 87.78%의 최고 정확도를 기록했습니다. 연구에서는 다양한 배치 크기 및 데이터 증가 기법의 영향도 분석하였고, 최적의 조정된 모델들이 각기 다르게 고유한 메모리 복잡성을 가지고 있다는 점을 강조했습니다.



### IDEA: An Inverse Domain Expert Adaptation Based Active DNN IP Protection Method (https://arxiv.org/abs/2410.00059)
- **What's New**: 새로운 논문에서는 모델 소유자가 불법 사용자를 차단하고 침해의 출처를 추적할 수 있는 IDEA(Inverse Domain Expert Adaptation 기반의 능동적 DNN IP 보호 방법)를 제안합니다.

- **Technical Details**: IDEA는 사용자 키를 하이드 스테가노그래픽 기법을 통해 숨겨 사용자의 인증을 수행하며, 진짜 전문가(real expert)와 두 개의 가짜 전문가(fake experts)를 훈련시킵니다. 이 과정에서 서로의 정보를 최소화하는 다중 적응 최적화(multi-adaptive optimization)가 적용됩니다.

- **Performance Highlights**: IDEA는 성능 평가를 위해 5개의 데이터셋과 4개의 DNN 모델에서 광범위한 실험을 진행하였으며, 인증 제어와 범죄자 추적 성공률, 다양한 공격에 대한 강인성에서 효과성을 입증하였습니다.



### Generalizing Consistency Policy to Visual RL with Prioritized Proximal Experience Regularization (https://arxiv.org/abs/2410.00051)
Comments:
          Accepted at the Thirty-Eighth Annual Conference on Neural Information Processing Systems (NeurIPS2024)

- **What's New**: 본 연구에서는 고차원 상태 공간의 비주얼 강화 학습(visual RL)에서 정책 훈련의 불안정성을 완화하기 위해 일관성 정책(consistency policy)의 샘플 기반 엔트로피 정규화(sample-based entropy regularization)와 우선순위 친근 경험 정규화(prioritized proximal experience regularization, CP3ER) 접근 방식을 제안합니다. CP3ER은 DeepMind 제어 스위트와 Meta-world를 포함한 21개 작업에서 새로운 최첨단 성능(SOTA)을 달성했습니다.

- **Technical Details**: 우선 CP3ER은 온라인 강화 학습(online RL)에서 비가역적인 데이터 분포(non-stationary distribution)와 액터-크리틱(actor-critic) 프레임워크가 일관성 정책에 미치는 영향을 조사했습니다. 연구 결과, 액터-크리틱 프레임워크의 Q-손실(Q-loss)이 일관성 모델의 표현 능력을 저해해 불안정한 정책 훈련을 초래하는 것을 발견했습니다. 본 연구에서는 이를 해결하기 위해 정책 훈련을 안정화하는 샘플 기반 엔트로피 정규화를 제안하였습니다.

- **Performance Highlights**: 제안된 CP3ER 방법론은 DeepMind 제어 스위트와 Meta-world 등 21개의 비주얼 제어 작업에서 SOTA 성능을 기록하였으며, 이는 비저장 강화 학습에서 일관성 모델의 응용 가능성을 보여줍니다.



### CycleBNN: Cyclic Precision Training in Binary Neural Networks (https://arxiv.org/abs/2410.00050)
Comments:
          Published at Workshop CADL, ECCV-2024

- **What's New**: 이 논문은 Binary Neural Networks(BNNs)의 새로운 훈련 방법론인 CycleBNN을 제안합니다. 기존의 문제인 훈련 속도가 느리고 성능 저하가 있는 문제를 해결하기 위해, 훈련 과정에서 정밀도를 동적으로 조정하는 사이클릭 정밀도 훈련을 도입합니다.

- **Technical Details**: CycleBNN은 BNNs와 사이클릭 정밀도 훈련을 통합하여 훈련 효율을 높이고 성능 손실을 최소화합니다. 이 방법은 가중치와 활성화를 주기적으로 1비트로 표현하는 BNN의 특성을 활용하여 정밀도를 최적화합니다. 실험은 ImageNet, CIFAR-10, PASCAL-VOC 데이터셋에서 수행되었으며, 각기 다른 태스크에서 CycleBNN의 성능을 평가하였습니다.

- **Performance Highlights**: CycleBNN은 ImageNet에서 훈련 시 96.09%의 연산량 절감, CIFAR-10에서 88.88%, PASCAL-VOC에서 다시 96.09%의 절감을 달성했습니다. 이는 CycleBNN이 효율적인 네트워크 훈련을 가능하게 하는 방법이라는 것을 입증합니다.



### Mixture of Multicenter Experts in Multimodal Generative AI for Advanced Radiotherapy Target Delineation (https://arxiv.org/abs/2410.00046)
Comments:
          39 pages

- **What's New**: 이번 논문에서는 Clinical AI 모델의 편향을 극복하기 위해 Mixture of Multicenter Experts (MoME) 접근법을 도입하였으며, 이는 다양한 임상 전략의 Specialized expertise를 통합하여 AI 모델의 범용성과 적응성을 향상시킵니다.

- **Technical Details**: MoME 모델은 각 의료 센터의 이미지와 임상 노트를 포함한 few-shot 샘플로 훈련되었으며, prostate cancer의 방사선 치료 목표 영역 delineation에서 기존 방법보다 뛰어난 성능을 보였습니다. MoME는 중앙 집중식 데이터 훈련보다 적은 데이터로도 각 의료 센터의 특성과 데이터 분포에 신속하게 적응할 수 있습니다.

- **Performance Highlights**: MoME 기반 모델은 전통적인 AI 모델들보다 목표 영역 delineation에서 눈에 띄게 성능이 향상되었으며, 특히 다양한 데이터 특성이 있을 경우 그 효과가 강조되었습니다. 이는 자원 제한이 있는 의료 시설에서도 활용 가능한 가능성을 보여줍니다.



### MM1.5: Methods, Analysis & Insights from Multimodal LLM Fine-tuning (https://arxiv.org/abs/2409.20566)
- **What's New**: MM1.5라는 새로운 멀티모달 대형 언어 모델(Multimodal Large Language Models) 패밀리를 소개합니다. 이 모델은 텍스트가 풍부한 이미지 이해, 비주얼 레퍼링(Visual Referring) 및 그라운딩(Grounding), 여러 이미지에 대한 추론 기능을 향상시키기 위해 설계되었습니다.

- **Technical Details**: MM1.5는 데이터 중심(Data-Centric) 접근 방식을 채택하여 모델 훈련 과정 전반에 걸쳐 다양한 데이터 혼합의 영향을 체계적으로 탐구합니다. 이는 고품질 OCR 데이터와 합성 캡션(Synthetic Captions)을 포함한 지속적인 사전 훈련(Continual Pre-Training) 및 최적화된 비주얼 인스트럭션 튜닝 데이터 혼합을 통한 감독적 미세 조정(Supervised Fine-Tuning)을 포함합니다. 모델은 1B부터 30B까지의 매개변수(Parameter)를 가지며, 밀집(Dense) 및 전문가 혼합(Mixture-of-Experts, MoE) 변형이 포함됩니다.

- **Performance Highlights**: 신중한 데이터 큐레이션(Data Curation) 및 훈련 전략이 작은 규모(1B 및 3B)의 모델에서도 강력한 성능을 내는 것을 입증했습니다. 또한 비디오 이해를 위한 MM1.5-Video 및 모바일 UI 이해를 위한 MM1.5-UI라는 두 가지 특수 변형을 도입합니다.



### DressRecon: Freeform 4D Human Reconstruction from Monocular Video (https://arxiv.org/abs/2409.20563)
Comments:
          Project page: this https URL

- **What's New**: DressRecon 기술을 통해 느슨한 의류와 객체 상호작용을 포함하는 단안 비디오에서 시간 일관성 있는 3D 인체 모델을 재구성하는 새로운 방법을 제시합니다. 기존 연구는 일반적으로 고급 장비를 필요로 하거나 개인화된 템플릿 스캔을 요구했으나, DressRecon은 이러한 제한을 극복합니다.

- **Technical Details**: 제안된 방법은 일반적인 인체 프라이어와 비디오에 특화된 'bag-of-bones' 변형 모델을 조합합니다. 이를 통해 인체와 의류의 변형을 분리하여 처리할 수 있는 신경 임플리트 모델을 학습합니다. 최적화 과정에서 인간의 신체 자세, 표면 법선, 광학 흐름 등의 이미지 기반 프라이어를 활용하여 정밀도를 높였습니다.

- **Performance Highlights**: DressRecon은 도전적인 의류 변형과 객체 상호작용이 포함된 데이터셋에서 이전 방법들보다 높은 품질의 3D 재구성을 제공합니다. 최종적으로, 최적화된 신경 필드에서 시간 일관성 있는 메쉬를 추출하거나, 3D 가우시안으로 변환하여 하이 피델리티의 인터랙티브 렌더링을 수행할 수 있습니다.



### SpaceMesh: A Continuous Representation for Learning Manifold Surface Meshes (https://arxiv.org/abs/2409.20562)
Comments:
          published at SIGGRAPH Asia 2024

- **What's New**: 이번 연구에서는 신경망(neural network)의 출력을 복잡한 연결성의 다각형 메시(polygonal meshes)를 직접 생성하는 새로운 방식을 제안합니다. 이는 메시에 대한 연속적인 잠재 연결 공간(latent connectivity space)을 정의하여 이산 메시(discrete mesh)를 암시적으로 생성합니다.

- **Technical Details**: 새로운 표현법인 SpaceMesh는 연속적인 임베딩(embedding)을 사용하여 메시에 대한 복잡한 연결성을 지원합니다. 이 방식은 halfedge 데이터 구조(halfedge data structure)을 기반으로 하며, 이는 기본적으로 다면체 연결성을 보장합니다. 각 메시 정점(vertex)마다 저차원(low-dimensional)의 임베딩을 통해 메시에 필요한 구조적 연결성(edge adjacency and next relationships)을 정의합니다.

- **Performance Highlights**: 이 모델은 대규모 데이터셋에서 다양하고 질 높은 메시를 생성하며, 메시 수리(mesh repair)와 같은 기하학적 처리 작업을 직접 학습하는 능력을 보유하고 있습니다. 이 방식을 통해 고품질 출력(generation)의 생성과 함께 학습 작업에서 매우 빠른 수렴 속도를 달성합니다.



### Uni$^2$Det: Unified and Universal Framework for Prompt-Guided Multi-dataset 3D Detection (https://arxiv.org/abs/2409.20558)
Comments:
          13 pages, 5 figures, 6 tables

- **What's New**: Uni$^2$Det라는 새로운 프레임워크를 제안하여 다양한 도메인에서 강력한 성능을 발휘하고 보이지 않는 도메인에도 일반화할 수 있도록 지원한다. 기존의 다중 데이터셋 결합 기법의 한계를 극복하기 위해 다단계 프롬프트 모듈을 도입했다.

- **Technical Details**: 이 프레임워크는 다양한 LiDAR 데이터셋과 3D 객체 탐지 베이스라인에 적용할 수 있는 다단계 프롬프트 모듈을 통합한다. 점 분포 보정 및 BEV 기반 범위 마스킹 기술을 사용하여 데이터 간의 변동성을 완화하며, 실제 데이터에 대한의존성을 줄인다.

- **Performance Highlights**: KITTI, Waymo 및 nuScenes 데이터를 포함한 여러 데이터셋 통합 시나리오에서 실험을 진행하여, Uni$^2$Det가 기존 방법보다 현저히 뛰어난 성능을 보였으며, 제로샷 크로스 데이터셋 전이에서도 일반화 능력을 검증했다.



### Propose, Assess, Search: Harnessing LLMs for Goal-Oriented Planning in Instructional Videos (https://arxiv.org/abs/2409.20557)
Comments:
          Accepted by ECCV 2024 (Oral)

- **What's New**: 이 연구에서는 VidAssist라는 통합 프레임워크를 소개하여, 제로(Zero)/소수(Few-shot) 샷 목표 지향 계획을 인스트럭션 비디오에 적용하는 데 초점을 맞추었습니다. VidAssist는 대형 언어 모델(LLMs)을 지식 기반 및 평가 도구로 활용하여 소규모 데이터셋에서 절차적 지식을 효과적으로 획득합니다.

- **Technical Details**: VidAssist는 주어진 목표와 현재 상태를 통해 최적의 행동 계획을 생성합니다. 주요 프로세스는 제안(Propose), 평가(Assess), 탐색(Search)으로 구성되어 있으며, BFS(breadth-first search) 알고리즘을 사용하여 행동 계획을 최적화합니다. 이를 통해 예측된 행동의 일관성과 실행 가능성을 평가합니다.

- **Performance Highlights**: VidAssist는 COIN 데이터셋에서 VPA 시나리오에서 +7.7%, PP 시나리오에서 +4.81% 향상된 성능을 보여주었습니다. 특히, 지정된 미래 행동을 예측하여 제로 샷 환경에서도 우수한 성공률을 기록했습니다.



### Inverse Painting: Reconstructing The Painting Process (https://arxiv.org/abs/2409.20556)
Comments:
          Project Page: this https URL

- **What's New**: 이 논문에서는 입력된 그림을 바탕으로 해당 그림이 어떻게 그려졌는지를 보여주는 시간 경과 비디오를 재구성하는 방법을 제시합니다. 이 과정에서 자율 회귀 생성을 이미지 생성 문제로 포맷하여 처음에는 빈 캔버스를 사용하고 이를 단계적으로 업데이트합니다. 모델은 실제 화가의 작업 영상을 기반으로 학습하여 다양한 예술적 스타일과 장르를 보여주는 결과를 도출합니다.

- **Technical Details**: 이번 연구는 텍스트와 지역 이해를 포함하여 "그림 지침"을 정의하는데 집중하며, 새로운 확산 기반 렌더러(Diffusion-based renderer)를 이용해 캔버스를 업데이트합니다. 이 방법은 복잡한 캔버스 상태를 관리하며, 발전된 자율 회귀 방식(autoregressive image generation)을 사용하여 목표 그림으로 나아가기 위한 여러 키프레임의 시퀀스를 생성합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 현재 최고 수준의 기술들과 비교할 때 고품질의 인간과 유사한 그림 비디오를 생성하는 데 뛰어나며, 정성적 및 정량적 결과와 인간 평가를 통해 그 성능을 입증하고 있습니다.



### Dual Encoder GAN Inversion for High-Fidelity 3D Head Reconstruction from Single Images (https://arxiv.org/abs/2409.20530)
Comments:
          Joint first two authors. Accepted to NeurIPS 2024

- **What's New**: 이번 연구에서는 PanoHead라는 새로운 프레임워크를 기반으로 한 3D GAN inversion 방법을 제안하였습니다. 이전의 연구들이 주로 EG3D에 의존했던 반면, PanoHead는 360도 관점에서 이미지를 합성하는 데 최적화되어 있습니다. 또한, 두 개의 인코더를 결합한 시스템으로 고화질 복원과 현실적인 생성이 가능하다는 점이 특징입니다.

- **Technical Details**: 연구에서는 홀적으로 보이는 3D 기하 구조를 복원하기 위해 듀얼 인코더 시스템을 도입하였습니다. 한 인코더는 주어진 뷰에 대한 고 충실도의 복원에 집중하고, 다른 인코더는 보이지 않는 뷰의 고품질 생성에 중점을 두고 훈련됩니다. 또한, 두 인코더의 출력을 원활하게 결합하기 위해 triplane 도메인이 사용되며, occlusion-aware triplane discriminator를 기반으로 한 새로운 손실 함수를 통해 일관된 결과를 도출하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 기존 인코더 훈련 방법들보다 정성적 및 정량적으로 우수한 성능을 보여주었습니다. 특히, 입력 이미지에 대한 고충실도 재구성과 보이지 않는 부분에 대한 현실적인 표현을 동시에 달성하는 데 성공하였습니다.



### Accelerating Non-Maximum Suppression: A Graph Theory Perspectiv (https://arxiv.org/abs/2409.20520)
- **What's New**: 본 논문은 객체 탐지에서 필수적인 후처리 단계인 Non-maximum suppression (NMS)을 그래프 이론 관점에서 처음으로 체계적으로 분석하였습니다. 이를 통해 NMS의 내재적 구조를 밝혀내고 QSI-NMS와 BOE-NMS 두 가지 최적화 방법을 제안합니다.

- **Technical Details**: QSI-NMS는 빠른 재귀적 분할 정복 알고리즘으로, mAP 손실이 거의 없으며, 확장된 버전 eQSI-NMS는 최적 복잡성인 \mathcal{O}(n\log n)을 달성합니다. BOE-NMS는 NMS의 지역성(locality)을 활용하여 mAP 손실 없이 일정 수준의 최적화를 이룹니다. 또한 NMS-Bench라는 첫 번째 벤치마크를 소개하여 다양한 NMS 방법을 종합적으로 평가합니다.

- **Performance Highlights**: YOLOv8-N 모델을 기준으로 하여 QSI-NMS는 원래 NMS에 비해 6.2배의 속도를 제공하며, mAP는 0.1% 감소합니다. 최적의 eQSI-NMS는 0.3% mAP 감소로 10.7배의 속도를 달성하고, BOE-NMS는 mAP 손실 없이 5.1배의 속도를 기록합니다.



### NUTRIVISION: A System for Automatic Diet Management in Smart Healthcar (https://arxiv.org/abs/2409.20508)
Comments:
          25 pages and 18 figures

- **What's New**: 이 논문에서는 NutriVision이라는 새로운 시스템을 소개합니다. 이 시스템은 컴퓨터 비전(computer vision) 및 머신러닝(machine learning)을 통해 음식 항목을 식별하고, 양을 추정하며, 포괄적인 영양 정보를 제공합니다. NutriVision은 더 빠르고 정확한 음식 인식을 위해 Faster Region-based Convolutional Neural Network를 활용합니다.

- **Technical Details**: NutriVision 시스템은 이미지 인식(image recognition) 알고리즘을 활용하여 사용자가 제공한 사진에서 식품 재료를 인식하고 이를 바탕으로 영양 평가를 수행합니다. 이 시스템은 개별 사용자의 건강 보고서 및 목표에 따라 맞춤형 식단 추천을 제공합니다. 또한 NutriVision은 PCI (Patient Connectivity Interface) 허브 내부에서 다른 헬스케어 장치와 연동되어 영양 관리 및 신속한 피드백을 제공합니다.

- **Performance Highlights**: NutriVision의 주요 장점은 자동화된 실시간 영양 내용 추정 기능을 제공하며, 사용자에게 음식 섭취 전 영양 가치를 이해할 수 있게 돕습니다. 또한 사용자 맞춤형 건강 데이터를 통합하여 개별 필요에 맞는 식단 추천을 제공합니다. 이러한 모든 기능은 최소한의 노력으로 사용자들에게 더 높은 영양 섭취를 관리할 수 있게 지원합니다.



### FreeMask: Rethinking the Importance of Attention Masks for Zero-Shot Video Editing (https://arxiv.org/abs/2409.20500)
Comments:
          Video Editing

- **What's New**: 이 논문에서는 Text-to-video diffusion 모델을 활용한 제로샷(Zero-shot) 비디오 편집에 대한 연구를 다룹니다. 기존 연구에서 간과된 주요 요소는 cross-attention mask가 일관되지 않으며, 모델 구조와 디노이징 타임스텝에 따라 달라진다는 점을 발견하였습니다. 이에 따라 Mask Matching Cost (MMC)라는 메트릭을 제안하여 mask의 변동성을 정량화하고, 특정 비디오 편집 작업에 최적화된 FreeMask 방법론을 개발하였습니다.

- **Technical Details**: FreeMask는 Mask Matching Cost (MMC)를 기반으로 하여 cross-attention mask를 전략적으로 활용하는 접근 방식입니다. MMC는 LMMC와 TMMC 두 가지 메트릭을 도입하여 서로 다른 분석 레이어와 타임스텝에서 mask의 정밀도를 측정합니다. 이 외에도 FreeMask는 temp, cross, self-attention 모듈을 포함한 다양한 attention feature 간의 masked fusion 메커니즘을 개선하여 비디오 편집 품질을 향상시킵니다.

- **Performance Highlights**: 광범위한 실험 결과, FreeMask는 기존의 최첨단 방법들에 비해 뛰어난 의미적 충실도(semantic fidelity), 시간적 일관성(temporal consistency), 및 편집 품질(editing quality)을 달성하였으며, 사용자 정의 제어가 필요하지 않아 제로샷 비디오 편집 프레임워크에 원활하게 통합될 수 있습니다.



### IRFusionFormer: Enhancing Pavement Crack Segmentation with RGB-T Fusion and Topological-Based Loss (https://arxiv.org/abs/2409.20474)
Comments:
          13 pages, 3 figures

- **What's New**: 본 논문에서는 IRFusionFormer라는 새로운 crack segmentation 모델을 제안하고, RGB와 thermal 이미지를 효과적으로 통합하여 다양한 환경 조건에서의 segmentation 성능을 향상시키는 방법론을 소개합니다.

- **Technical Details**: 제안된 Efficient RGB-T Cross Fusion Module (EGTCF)은 RGB와 thermal 모달리티 간의 다중 스케일 관계와 장거리 의존성을 효과적으로 포착합니다. 또한 Interaction-Hybrid-Branch-Supervision (IHBS) 프레임워크를 통해 모달리티 간의 상호 작용을 증진시키고, 새로운 토폴로지 기반 손실 함수로 크랙의 연결성을 유지합니다.

- **Performance Highlights**: IRFusionFormer는 데모에 대한 최고의 성능을 달성하여 Dice 점수 90.01%와 Intersection over Union (IoU) 81.83%를 기록하며, 다양한 환경 조건에서의 견고성과 정확성을 크게 개선했습니다.



### Continual Human Pose Estimation for Incremental Integration of Keypoints and Pose Variations (https://arxiv.org/abs/2409.20469)
- **What's New**: 이 논문은 cross-dataset human pose estimation을 지속 학습(continual learning) 태스크로 재구성하여 새로운 keypoint와 pose 변형을 기존 모델에 통합하면서 이전 데이터셋에 대한 정확성을 잃지 않도록 하는 방법을 제안합니다.

- **Technical Details**: 기존의 catastrophc forgetting을 줄이기 위한 정규화 기반 방법들(EWC, LFL, LwF)과 비교하여, Importance-Weighted Distillation (IWD)라는 새로운 정규화 방법을 제안합니다. 이 방법은 레이어 중요도에 기반한 동적 온도 조정을 도입하여 레이어마다 distillation 손실을 조절합니다. 이를 통해 모델이 새로운 태스크에 적응하면서도 이전 지식을 효과적으로 유지하도록 합니다.

- **Performance Highlights**: 세 개의 데이터셋을 통해 수행한 실험에서 IWD 방법이 기존의 LwF 방법에 비해 평균 3.60% 향상된 성능을 보여주었으며, 이는 실제 애플리케이션에서 모델이 과거의 지식을 잊지 않고 새로운 데이터에 적응할 수 있는 강력한 프레임워크의 가능성을 강조합니다.



### Navigating Threats: A Survey of Physical Adversarial Attacks on LiDAR Perception Systems in Autonomous Vehicles (https://arxiv.org/abs/2409.20426)
- **What's New**: 이번 논문에서는 LiDAR 기반 인식 시스템을 겨냥한 물리적 적대적 공격에 대한 연구의 현황을 종합적으로 검토합니다. LiDAR(라이다) 시스템이 자율 주행 자동차에 필수적인 정밀한 인식 및 내비게이션을 위해 얼마나 중요한지를 강조하며, 이러한 시스템이 적대적 공격에 취약하다는 점을 다루고 있습니다.

- **Technical Details**: 이 논문은 다양한 적대적 공격 유형을 카테고리화하고 분석하며, 스푸핑(spoofing)과 물리적 적대적 객체 공격 등 여러 방법을 상세히 다룹니다. LiDAR 시스템은 외부로부터의 공격에 민감하게 노출되며, 깊이 정보 추출 및 3D 점 구름(point cloud) 생성에 사용됩니다.

- **Performance Highlights**: LiDAR 기반 시스템의 안전성 및 신뢰성을 높이는 방안을 제안하며, 현재 연구에서의 해결되지 않은 문제들을 식별하고 향후 방향성을 제시합니다. A comprehensive taxonomy를 제안하고, 실제 환경에서의 공격과 방어의 복잡성을 강조합니다.



### World to Code: Multi-modal Data Generation via Self-Instructed Compositional Captioning and Filtering (https://arxiv.org/abs/2409.20424)
Comments:
          Accepted at EMNLP 2024 Main Conference, 16pages

- **What's New**: 최근 비전-언어 모델( Vision-Language Models, VLMs)의 발전과 고품질 다중 모달 정렬 데이터의 부족으로 인해 합성 VLM 데이터 생성에 대한 연구가 증가하고 있습니다. 본 논문에서는 Python 코드 형식으로 최종 생성 출력을 구성하는 다중 모달 데이터 구축 파이프라인인 World to Code (W2C)를 소개합니다. W2C는 VLM 스스로를 활용하여 서로 다른 프롬프트를 통해 교차 모달 정보를 추출하고 생성된 출력을 일관성 필터링 전략을 통해 다시 필터링합니다.

- **Technical Details**: W2C 파이프라인은 VLM이 필요로 하는 전문가의 혼합을 줄이고 비싼 인간 피드백 없이 데이터를 생성하고 필터링하는 방법을 제공합니다. 실험 결과, W2C는 여러 기존 비주얼 질문 응답(Visual Question Answering, VQA)과 비주얼 그라운딩(Visual Grounding) 벤치마크에서 다른 VLM들에 비해 높은 품질을 보여주었습니다. W2C는 LLaVA-NeXT-7B 기준으로 9개의 VQA 벤치마크 중 7개에서, LLaVA-NeXT-13B 기준으로 9개 중 6개에서 최적의 성능을 보였습니다.

- **Performance Highlights**: W2C는 GQA와 MME 같은 널리 사용되는 VQA 벤치마크에서 몇 가지 샷 평가(few-shot evaluation)에서도 개선된 성능을 보이며, 특히 GQA의 2-shot 평가에서 모든 VLM들에 대해 5 이상의 정확도 향상을 달성했습니다. 또한 W2C는 기존의 세부 캡션 능력보다 교차 모달 동등성을 더 잘 보여주는 새로운 코드 파싱 능력을 제공합니다.



### AI-Based Fully Automatic Analysis of Retinal Vascular Morphology in Pediatric High Myopia (https://arxiv.org/abs/2409.20419)
- **What's New**: 이 연구는 인공지능 (Artificial Intelligence) 기반의 자동화 소프트웨어를 통해 근시의 다양한 단계와 관련된 망막 혈관 구조의 변화를 조사하였습니다.

- **Technical Details**: 연구는 중국 아동 의료 센터에서 1324명의 소아 참가자를 대상으로 하였으며, 2366개의 고품질 망막 이미지와 이에 상응하는 굴절 매개변수를 분석했습니다. Convolutional Neural Networks (CNN) 모델과 다른 모듈을 조합한 데이터 분석 모델을 통해 이미지 분류, 혈관 구조 분할, 주요 각도 (Main Angle), 가지각도 (Branching Angle), 분기 가장자리 각도 (Bifurcation Edge Angle), 분기 가장자리 계수 (Bifurcation Edge Coefficient) 등의 혈관 매개변수를 측정했습니다.

- **Performance Highlights**: 모델의 정확도는 손실 함수가 0.09로 수렴했을 때 94.19%에 도달했습니다. 정상군과 고도 근시군의 혈관 이미지 수는 각각 279개(12.38%)와 384개(16.23%)였으며, 근시 굴절군 간의 주요 각도(MA), 가지각도(BA), 분기 계수(BEC) 등에서 유의미한 차이를 보였습니다.



### Physics-Regularized Multi-Modal Image Assimilation for Brain Tumor Localization (https://arxiv.org/abs/2409.20409)
Comments:
          Accepted to NeurIPS 2024

- **What's New**: 본 연구에서는 물리 기반(cost function)과 데이터 기반(cost function)을 균형 있게 결합한 새로운 방법을 제안합니다. 특히, 우리가 학습한 종양 및 뇌 조직 분포가 성장 및 탄성 방정식에 얼마나 부합하는지를 정량화하는 독특한 이산화(discretization) 방식을 소개합니다.

- **Technical Details**: 이 모델은 두 가지 주요 접근 방식을 통해 종양 세포 분포를 학습합니다. 첫 번째는 물리적 법칙에 엄격히 준수하도록 출력 종양 세포 분포를 제한하는 포괄적 수식(PDE)을 사용하는 것이고, 두 번째는 주어진 시각적 데이터를 활용하여 더 유연한 모델을 구축하는 것입니다. 이에 따라, 종양 세포 밀도 함수 c(x,t)를 반응-확산-전달 방정식을 근사적으로 만족하게 하는 데 초점을 맞춥니다.

- **Performance Highlights**: 본 방법은 실제 환자의 데이터를 기반으로 하여 기존 기술보다 종양 재발 영역의 포괄성을 향상시켰으며, 현재 우리가 알고 있는 가장 큰 공개 데이터 세트를 활용하여 방사선 치료 계획에서 새로운 성과를 달성했습니다.



### Open-Source Periorbital Segmentation Dataset for Ophthalmic Applications (https://arxiv.org/abs/2409.20407)
Comments:
          12 pages, 4 figures

- **What's New**: 이 논문은 깊은 학습(deep learning) 기술을 이용하여 안면 주변(periorbital) 분할(segmentation) 및 거리 예측을 위한 데이터 세트를 제안합니다. 현재 눈 주변의 고해상도(segmentation datasets)가 존재하지 않으며, 이를 통해 질병 상태의 객관적 정량화가 가능해집니다.

- **Technical Details**: 연구에서는 2842장의 이미지에서 홍채(iris), 공막(sclera), 눈꺼풀(lid), 결막(caruncle), 눈썹(brow) 등의 영역을 다섯 명의 훈련된 주석자들이 분할하였습니다. 내부 및 외부 주석자 신뢰성 테스트를 통해 데이터 세트를 검증하였고, 이 데이터가 안면 주변 분할 네트워크 교육에 유용하다는 것을 보여주었습니다. 모든 주석은 공개적으로 무료로 다운로드 가능합니다.

- **Performance Highlights**: 안면 주위 분할을 위한 데이터 세트를 통해 임상에서 유용한 분할 네트워크의 빠른 개발이 가능해지고, 이를 통해 안면 주변 거리 예측 및 질병 분류에 활용될 수 있습니다. 또한, 주석 외에도 분할 마스크(segmentation masks)에서의 거리를 예측하기 위한 오픈 소스(toolkit)가 제공되며, 모든 모델의 가중치(weights)도 오픈소스로 공개되어 학계에서 활용될 수 있도록 되어 있습니다.



### AUCSeg: AUC-oriented Pixel-level Long-tail Semantic Segmentation (https://arxiv.org/abs/2409.20398)
- **What's New**: 이번 논문은 픽셀 수준의 롱테일(長尾) 의미 분할(semantic segmentation) 문제에 대해 AUC(ROC 곡선 아래 면적) 최적화 방법을 연구하였습니다. 기존의 AUC 최적화 방법들은 인스턴스 수준(long-tail learning) 접근법에 국한되어 있었으나, 본 연구는 더욱 복잡한 픽셀 수준 상황에 적용됩니다.

- **Technical Details**: AUC 최적화는 픽셀 수준 작업에서 손실 항(loss term) 간의 복잡한 결합(has complex coupling)과 구조화된 내부 이미지(inner-image) 및 쌍 간 이미지 간(pairwise inter-image) 의존성(dependencies)을 포함합니다. 또한, 미니 배치(mini-batch)에서 AUC 손실 추정이 더 큰 배치 사이즈(batch size)를 요구하여 메모리 공간 복잡도(space complexity) 문제를 초래합니다. 이를 해결하기 위해 픽셀 수준의 AUC 손실 함수와 알고리즘의 일반화 능력(generalization ability)에 대한 의존성 그래프(dependency-graph) 기반 이론 분석을 개발하였습니다.

- **Performance Highlights**: 다양한 벤치마크(benchmarks)에서 AUCSeg 방법의 효율성을 입증하는 포괄적인 실험을 진행하였으며, 그 결과는 제안된 방법의 우수한 성능을 확인시켜 줍니다. 코드도 제공되어 연구자들이 쉽게 활용할 수 있습니다.



### FireLite: Leveraging Transfer Learning for Efficient Fire Detection in Resource-Constrained Environments (https://arxiv.org/abs/2409.20384)
- **What's New**: 본 논문에서는 FireLite라는 저매개변수 (low-parameter) CNN(Convolutional Neural Network) 모델을 제안하여 제한된 자원 환경에서 신속한 화재 감지를 가능케 합니다.

- **Technical Details**: FireLite 모델은 34,978개의 학습 가능한 매개변수를 가지고 있으며, 실시간 화재 감지 애플리케이션을 위해 설계되었습니다. 전이 학습(transfer learning)을 활용하여 MobileNet 아키텍처를 베이스로 하여, 이미지넷(ImageNet) 데이터세트에서 사전 훈련된 특징을 추출하여 이를 활용합니다.

- **Performance Highlights**: FireLite 모델은 98.77%의 정확성을 보였으며, 검증 손실은 8.74, 정밀도(precision), 재현율(recall), F1-score 모두에서 98.77%에 도달했습니다. 자원이 제한된 환경에서도 효율성과 정확성을 보여주어 향후 화재 감지 시스템에 대한 유망한 해결책입니다.



### VideoINSTA: Zero-shot Long Video Understanding via Informative Spatial-Temporal Reasoning with LLMs (https://arxiv.org/abs/2409.20365)
Comments:
          EMNLP 2024 Findings; 22 pages; Code: this https URL

- **What's New**: 이 논문에서는 긴 비디오 이해를 위한 새로운 제로샷(zero-shot) 프레임워크인 VideoINSTA를 제안합니다. 이 프레임워크는 대형 언어 모델(LLMs)을 사용하여 비디오 정보의 공간적-시간적(reasoning) 이해를 향상시키는 데 중점을 두고 있습니다.

- **Technical Details**: VideoINSTA의 세 가지 주요 요소는 (1) 사건 기반(event-based) 시간적(temporal) 추론, (2) 내용 기반(content-based) 공간적(spatial) 추론, (3) LLM을 활용한 자기 반영적(reasoning) 정보 처리입니다. 이를 통해 LLM은 비디오의 공간적-시간적 정보를 효과적으로 활용하여 긴 비디오를 이해합니다.

- **Performance Highlights**: 이 모델은 EgoSchema, NextQA, IntentQA 및 ActivityNetQA와 같은 세 가지 긴 비디오 질문-응답 벤치마크에서 최신 기술을 현저하게 개선했습니다. VideoINSTA는 평균 3분 길이의 긴 비디오를 처리할 수 있으며, 다양한 다중 선택 및 개방형 질문 응답 작업에서 뛰어난 성능을 발휘했습니다.



### CableInspect-AD: An Expert-Annotated Anomaly Detection Datas (https://arxiv.org/abs/2409.20353)
Comments:
          35 pages, to appear at NeurIPS 2024

- **What's New**: CableInspect-AD라는 고품질의 공개 데이터셋을 소개하며, 이것은 로봇 전력선 점검을 위한 Visual Anomaly Detection (VAD)의 성능 향상을 돕기 위해 설계되었다.

- **Technical Details**: 이 데이터셋은 Hydro-Québec 전문가들에 의해 제작 및 주석이 달린 고해상도 이미지로 구성되어 있으며, 다양한 결함 유형과 심각도를 포함한다. Enhanced-PatchCore라는 방법을 통해 몇몇 명목적 예제만 가지고도 탐지 기준을 설정할 수 있도록 개선되었다.

- **Performance Highlights**: Enhanced-PatchCore는 적은 데이터에 대한 성능이 유망하지만, 특정한 종류와 심각도의 결함 탐지에는 여전히 한계가 있음을 나타낸다. 이 데이터셋은 VAD 연구 커뮤니티에 도전적인 기준점으로 작용할 것으로 기대된다.



### HEADS-UP: Head-Mounted Egocentric Dataset for Trajectory Prediction in Blind Assistance Systems (https://arxiv.org/abs/2409.20324)
- **What's New**: 본 논문에서는 HEADS-UP이라는 최초의 egocentric 데이터셋을 소개합니다. 이 데이터셋은 시각 장애인을 위한 경로 예측을 목적으로 헤드 마운트 카메라에서 수집되었습니다. 기존 데이터셋은 시각 장애인의 관점에서 필요한 정보를 캡처하지 못하고 있기 때문에, HEADS-UP 데이터셋은 이러한 공백을 채우기 위해 특별히 설계되었습니다.

- **Technical Details**: HEADS-UP 데이터셋은 43,000개 이상의 프레임으로 구성되어 있으며, RGB, 깊이(depth), 포인트 클라우드(point cloud) 데이터, IMU(IMU: 관성 측정 장치) 측정치, 보행자의 궤적 레이블을 포함합니다. 제안된 방법은 두 개의 개별 궤적을 예측하는 대신, 카메라의 로컬 좌표계에 회전된 반지역 좌표계(semi-local coordinate system)에서 하나의 예측을 수행하여 충돌 위험을 정확하게 평가합니다.

- **Performance Highlights**: HEADS-UP 데이터셋에서 검증된 이 방법은 ROS로 구현하였으며, NVIDIA Jetson GPU에서 실시간 테스트를 수행하여 동적인 실세계 환경에서의 실행 가능성을 입증하였습니다. 데이터셋 평가와 실시간 테스트 결과는 우리 접근 방식의 견고함과 효율성을 보여줍니다.



### Automating MedSAM by Learning Prompts with Weak Few-Shot Supervision (https://arxiv.org/abs/2409.20293)
Comments:
          Accepted to MICCAI-MedAGI 2024 (LNCS Proceedings, Volume 15184), 10 pages

- **What's New**: 이 연구는 사용자 상호작용이 필요 없는 경량화된 프롬프트 모듈을 제안합니다. 이 모듈은 이미지 임베딩에서 프롬프트 임베딩을 직접 학습하여 세그멘테이션 마스크를 생성합니다. 또한, 적은 양의 약한 레이블과 몇 개의 샘플로도 원하는 특정 영역을 자동으로 세그먼트 할 수 있습니다.

- **Technical Details**: 제안된 접근법은 MedSAM을 기반으로 하며, 이미지 인코더와 프롬프트 인코더, 마스크 디코더로 구성됩니다. 프롬프트 모듈은 이미지에 대한 두 개의 임베딩을 생성하여 사용자의 사전 정의된 프롬프트에 대한 의존성을 제거합니다. 이를 통해 초기에 주어진 정보(타이트 바운딩 박스)를 활용하여 효과적으로 세그멘테이션 결과를 도출합니다.

- **Performance Highlights**: 이 방법은 MR과 초음파 이미지를 포함한 세 가지 의료 데이터셋에서 검증되었으며, 약한 어노테이션(타이트 바운딩 박스)만으로도 효과적인 세그멘테이션 마스크를 생성하는 성능을 입증했습니다.



### Match Stereo Videos via Bidirectional Alignmen (https://arxiv.org/abs/2409.20283)
- **What's New**: 이번 논문에서는 비디오 스테레오 매칭 기술의 새로운 접근법을 제안합니다. 특히, 인접한 프레임 간의 양방향 정렬 메커니즘을 통해 일관된 불일치도를 추정하는 새로운 비디오 처리 프레임워크인 BiDAStereo와 안정화 네트워크인 BiDAStabilizer를 소개합니다.

- **Technical Details**: BiDAStereo 프레임워크는 세 가지 프레임 상관 관계 레이어와 이동 전파 재귀 유닛을 포함하여 지역 및 글로벌 시간 정보를 활용합니다. BiDAStabilizer는 기존 스테레오 이미지 기반 방법의 불일치한 불일치도를 일관성 있는 것으로 변환하는 플러그인 네트워크입니다. 이들은 서로 다른 프레임에서의 추정값으로부터 보완 정보를 효과적으로 활용하여 일관성을 향상합니다.

- **Performance Highlights**: 제안된 방법은 다양한 스테레오 비디오 벤치마크에서 최신 기술(State-of-the-Art, SOTA) 성능을 달성하였으며, 새로운 덜 현실적이고 다양한 실세계 데이터 세트를 도입하여 연구에 기여하고 있습니다.



### Solution for OOD-CV Workshop SSB Challenge 2024 (Open-Set Recognition Track) (https://arxiv.org/abs/2409.20277)
- **What's New**: 본 논문은 ECCV 2024의 OOD-CV 워크샵에서 열린 OSR 챌린지를 위한 방법론을 소개합니다. 이 연구는 오픈셋 인식(open-set recognition, OSR) 문제를 다루며, 데이터의 분포가 다를 때도 분류 모형이 올바르게 예측할 수 있도록 하는 하이브리드 접근법을 제안합니다.

- **Technical Details**: 우리는 라벨이 없는 데이터에 대한 감지 성능을 높이기 위해 여러 개의 post-hoc OOD 탐지 기법과 Test-Time Augmentation (TTA)을 결합한 하이브리드 프레임워크를 사용했습니다. 사용한 주요 기술 중 하나는 ReAct 메소드로, 이는 신경망의 비정상 작용을 조정하여 OOD 데이터에 대한 감지 성능을 향상시킵니다. TTA는 이미지에 다양한 증강을 적용하여 모델의 일반화 능력을 강화합니다.

- **Performance Highlights**: 우리의 최종 성과는 AUROC: 79.77 (5위) 및 FPR95: 61.44 (2위)로, 전체 대회에서 2위를 차지하는 성과를 거두었습니다.



### Active Neural Mapping at Sca (https://arxiv.org/abs/2409.20276)
- **What's New**: 이 논문에서는 대규모 실내 환경을 효율적이고 견고하게 탐색할 수 있는 NeRF 기반 액티브 매핑 시스템을 소개합니다. 우리의 접근 방식의 핵심은 지속적으로 업데이트되는 신경 맵에서 일반화된 보로노이 그래프(Generalized Voronoi Graph, GVG)를 추출하는 것입니다. 이를 통해 장면의 기하학, 외관, 구조 및 불확실성을 통합하게 됩니다.

- **Technical Details**: 우리의 시스템은 현대의 혼합 NeRF 표현을 활용하여 많은 실내 환경에서 재구성 정확도, 범위 완전성, 탐색 효율성 측면에서 경쟁력 있는 결과를 달성합니다. GVG의 정점에 신경 맵에서의 불확실한 영역을 고정하여 안전한 경로를 따라 탐색의 적응적 세분화를 가능하게 합니다.

- **Performance Highlights**: 다양한 스케일에서의 광범위한 결과가 제안된 시스템의 효율성을 검증하며, 20개 이상의 방을 포함한 대규모 실내 환경에서도 포괄적으로 탐색할 수 있습니다. 신경 맵 내에서 구조화된 정보를 추출하여 접근 가능한 관심 영역과 가시적인 관심 영역을 모두 고려합니다.



### Loose Social-Interaction Recognition in Real-world Therapy Scenarios (https://arxiv.org/abs/2409.20270)
- **What's New**: 이번 연구에서는 두 사람이 느슨한 상호작용(loose interactions)을 하는 복잡한 상황을 분석하기 위한 새로운 건축 구조를 제안합니다. 이 구조는 3D-CNN을 기반으로 한 두 개의 경로를 통해 각 개인의 글로벌 추상적 특징을 학습하고, 이를 새로운 Global-Layer-Attention 모듈을 사용하여 융합합니다.

- **Technical Details**: 제안된 모델은 두 개의 독립적인 입력(리더와 보조자)에서 고-저 수준의 다중 스케일 시각적 특징을 개별적으로 학습하며, 3D-CNN을 사용합니다. 학습된 추상적 특징은 Abstract Projection을 통해 획득되며, 인식은 새로운 Global Layers Attention(GLA) 메커니즘을 통해 수행됩니다.

- **Performance Highlights**: 우리의 네트워크는 실제 자폐증 진단을 위한 Loose-Interaction 데이터세트 및 공공 데이터세트에서 기준 성능을 달성했습니다. SOTA(State-of-the-Art) 결과는 자폐증 데이터세트에서 나타났습니다. 또한, 서로 다른 사회적 상호작용을 연구하기 위해 NTU-RGB+D 데이터세트에 대한 실험을 진행했으며, 다양한 상호작용에 따라 다른 네트워크 구성이 필요함을 발견했습니다.



### PerCo (SD): Open Perceptual Compression (https://arxiv.org/abs/2409.20255)
- **What's New**: 이번 연구에서는 Stable Diffusion v2.1 기반의 PerCo (SD)라는 새로운 지각적 이미지 압축 방법을 소개합니다. PerCo (SD)는 기존의 PerCo 기술에 대한 개방적이고 경쟁력 있는 대안으로 제안됩니다.

- **Technical Details**: PerCo (SD)는 Neural Image Compression 기술을 사용하여 생성적 모델을 통합하여 이미지의 압축된 표현을 학습합니다. 특히 auto-encoder 구조를 기반으로 하여, 인코더 E와 디코더 D 그리고 선택적으로 엔트로피 모델 P가 함께 훈련됩니다. 이 방법은 높은 인지 품질을 위해 텍스처와 같은 결손 세부정보를 자연스럽게 합성합니다.

- **Performance Highlights**: MSCOCO-30k 데이터셋에서 PerCo (SD)는 더 높은 왜곡을 감수하는 대신 개선된 지각적 특성을 보여주었습니다. 이 연구를 통해 깊이 있는 이해를 돕고 향후 해당 분야의 발전을 위한 기초를 마련하고자 합니다.



### Medical Image Segmentation with SAM-generated Annotations (https://arxiv.org/abs/2409.20253)
Comments:
          Accepted to the European Conference on Computer Vision (ECCVW) Workshops 2024

- **What's New**: 이 논문에서는 의료 이미지 분할을 위해 Segment Anything Model(SAM)을 데이터 주석 도구로 평가하여 'pseudo labels'를 생성하고, 이를 통해 약한 지도 학습(weakly-supervised learning) 방식으로 UNet 모델을 훈련하여 성능을 비교했습니다.

- **Technical Details**: 논문에서는 Medical Segmentation Decathlon(MSD) CT 과제를 대상으로 하여 SAM을 이용해 생성한 pseudo labels로 UNet 모델을 훈련시키며, 여러 가지 프롬프트(prompt) 방식을 실험했습니다. 특히 bounding box 프롬프트가 pseudo labels 생성을 위한 효과적인 방법으로 확인되었습니다.

- **Performance Highlights**: 실험 결과, SAM이 생성한 pseudo labels로 훈련한 모델이 완전 감독 모델과 유사한 성능을 보여, SAM이 의료 이미지 데이터 주석 도구로서 큰 잠재력을 가지고 있음을 나타냈습니다.



### Classroom-Inspired Multi-Mentor Distillation with Adaptive Learning Strategies (https://arxiv.org/abs/2409.20237)
- **What's New**: 최근 발표된 ClassroomKD(클래스룸 지식 증류) 프레임워크는 학생과 다수의 멘토 간의 지식 이전을 강화하기 위해 고안된 혁신적인 방법입니다. 이 프레임워크는 정적인 멘토-학생 관계 대신 각 데이터 샘플의 효과에 따라 멘토의 교수 전략을 동적으로 선택하고 조정합니다.

- **Technical Details**: ClassroomKD는 두 가지 주요 모듈로 구성됩니다: Knowledge Filtering (KF) 모듈과 Mentoring 모듈입니다. KF 모듈은 입력 데이터에 대해 멘토의 성능을 기반으로 멘토를 동적으로 순위 매기고, 성능이 충분한 멘토만 활성화하여 오류 누적을 최소화합니다. Mentoring 모듈은 학생과 멘토 간의 성능 차이에 따라 각 멘토의 영향을 조정하여 학습 속도를 효과적으로 조절합니다.

- **Performance Highlights**: CIFAR-100, ImageNet을 포함한 이미지 분류와 COCO Keypoints, MPII Human Pose를 포함한 2D 인간 pose 추정 분야에서 광범위한 실험을 통해 ClassroomKD가 기존의 지식 증류 방법보다 상당히 우수한 성능을 발휘함을 입증했습니다. 이러한 결과는 동적이고 적응적인 멘토 선택 및 안내 접근 방식이 더 효과적인 지식 이전로 이어져 모델 성능 향상을 위한 기회를 제시함을 나타냅니다.



### GTransPDM: A Graph-embedded Transformer with Positional Decoupling for Pedestrian Crossing Intention Prediction (https://arxiv.org/abs/2409.20223)
- **What's New**: GTransPDM, a new Graph-embedded Transformer with a Position Decoupling Module, has been developed to enhance pedestrian crossing intention prediction (PCIP). It addresses issues related to distorted input data from onboard cameras by decomposing lateral movements and simulating depth variations using multi-modal features.

- **Technical Details**: The proposed model integrates a Position Decoupling Module (PDM) to accurately decompose pedestrian lateral movements and simulate depth variations between the ego-vehicle and pedestrians. This is complemented by a graph-embedded Transformer that captures spatial-temporal dynamics of human skeletal poses, allowing better modeling of interactions and motions. Key components include GCN blocks for pose dynamics and performance is driven by a fast multi-modal fusion framework.

- **Performance Highlights**: GTransPDM achieved 92% accuracy on the PIE dataset and 87% accuracy on the JAAD dataset, performing at a processing speed of 0.05 ms. The model outperforms existing state-of-the-art methods in pedestrian crossing intention prediction.



### Mind the GAP: Glimpse-based Active Perception improves generalization and sample efficiency of visual reasoning (https://arxiv.org/abs/2409.20213)
Comments:
          10 pages of main text and 8 pages appendices

- **What's New**: 이 논문에서는 기존의 AI 시스템과 비교하여 사람의 시각 관계를 이해하는 능력의 차이를 논의하며, Glimpse-based Active Perception (GAP)이라는 새로운 시스템을 개발하여 이미지를 분석하는 과정을 혁신적으로 개선한 점이 주목할 만합니다.

- **Technical Details**: GAP 시스템은 입력 이미지의 가장 두드러진 영역을 순차적으로 빠르게 보고 높은 해상도로 처리합니다. 이 시스템은 시각적 내용과 위치 정보를 조합하여 이미지의 다양한 부분 간의 관계를 표현합니다. 특히, 이러한 시각적 정보는 ‘what’ 경로를 통해 얻고, 공간 정보는 ‘where’ 경로를 통해 수집됩니다.

- **Performance Highlights**: 이 시스템은 여러 비주얼 추론 작업에서 최첨단 성능을 달성하며, 샘플 효율(sample efficiency) 측면에서 우수한 성과를 보이고, 이전 모델보다 분포 밖(out-of-distribution) 시각 입력에 대한 일반화 능력이 뛰어납니다.



### UIR-LoRA: Achieving Universal Image Restoration through Multiple Low-Rank Adaptation (https://arxiv.org/abs/2409.20197)
- **What's New**: 기존의 다중 변형 이미지 복원 방법들이 다중 작업 학습(multi-task learning) 문제로 접근하는 것과 달리, 본 논문에서는 여러 개의 LoRA(low-rank adapter)를 활용한 범용 이미지 복원 프레임워크(UIR-LoRA)를 제안합니다. 이 프레임워크는 사전 훈련된 생성 모델을 활용하여 다양한 변형 복원 작업에 적용할 수 있는 방법을 제시합니다.

- **Technical Details**: UIR-LoRA는 다중 도메인 전이 학습(multi-domain transfer learning)의 관점에서 접근합니다. 이 프레임워크는 기본 모델을 활용하고, 각 복원 작업에 대해 저차원 매개변수 행렬을 추가하여 LoRA 기법(LoRA technique)을 적용합니다. 또한, 변형 간의 유사성을 바탕으로 LoRA를 조합하는 전략을 도입하여 혼합 변형 복원에도 적합한 모델을 구축합니다.

- **Performance Highlights**: 다양한 변형 및 혼합 변형에 대한 Extensive 실험 결과, 제안된 방법(UIR-LoRA)이 기존의 통합 이미지 복원 모델에 비해 왜곡(distorion)과 지각(perceptual) 지표에서 우수한 성능을 보였으며, 더 나은 일반화(generalization) 능력을 부각시켰습니다.



### Forecasting Disease Progression with Parallel Hyperplanes in Longitudinal Retinal OC (https://arxiv.org/abs/2409.20195)
Comments:
          accepted in MICCAI 2024

- **What's New**: 최근의 연구에서, 우리는 망막 OCT 스캔을 통해 늦은 건성 나이 관련 황반변성(dAMD)의 발생 위험을 예측하기 위한 새로운 딥러닝(Deep Learning) 방법을 제안하였습니다. 이 방법은 현재 스캔을 기반으로 전환 시간과 관련된 위험 점수와 특정 시간 내 변환 확률을 예측합니다.

- **Technical Details**: 제안된 방법은 변환 시간(T*)을 랜덤 변수로 모델링하고, 이와 관련된 누적 분포 함수(CDF)를 계산합니다. 또한, 우리는 주어진 이미지 집합에 대해 위험 점수를 할당하여, 다양한 위험 집단으로 분류할 수 있는 시스템을 개발하였습니다. 이 시스템은 각 객체 간 일관성 있는 예측을 보장하는 비지도 학습 손실을 활용합니다.

- **Performance Highlights**: 2개의 대규모 데이터셋을 사용한 평가 결과, Dataset-1에서 평균 AUROC 0.82, Dataset-2에서 평균 AUROC 0.83을 달성하였습니다. 이러한 성능 지표는 다양한 스캐너에서 수집된 이미지 간 도메인 시프트를 극복하는 능력을 보여줍니다.



### Annotation-Free Curb Detection Leveraging Altitude Difference Imag (https://arxiv.org/abs/2409.20171)
- **What's New**: 본 논문에서는 Altitude Difference Image (ADI)를 활용한 주행 안전성을 보장하는 도로 연석(road curbs) 탐지 방법을 제안합니다. 기존의 데이터 주석(annotation) 방식이 아닌 자동 주석 생성 모델(Automatic Curb Annotator, ACA)을 도입하여 수작업 데이터 주석 없이도 대량의 훈련 데이터를 생성할 수 있습니다.

- **Technical Details**: 제안된 방법은 RGB 이미지의 조명 민감성과 LiDAR 포인트 클라우드의 처리 지연 문제를 해결하기 위해, 포인트 클라우드를 ADI로 변환하여 사용합니다. 이 접근법은 주석이 필요 없는 탐지 기법으로, ADI를 통해 도로 연석을 더 효과적으로 탐지할 수 있으며, 자동 주석 생성 모듈(ACA)을 통해 훈련 데이터 생성을 자동화합니다.

- **Performance Highlights**: KITTI 3D 연석 데이터셋에서 평가한 결과, 제안된 방법이 기존 방법보다 지연 시간을 대폭 줄이며 최첨단 성능을 달성한 것으로 나타났습니다. 이는 자율 주행 시스템에서의 실시간 성능 요구에 적합하다는 것을 시사합니다.



### Task-Oriented Pre-Training for Drivable Area Detection (https://arxiv.org/abs/2409.20166)
- **What's New**: 이 논문에서는 드라이버블(Drivable) 영역 탐지를 위한 새로운 작업 지향적(pre-training) 프레임워크를 제안합니다. 이 방법은 전통적인 pre-training 및 self-training 방식보다 더 효과적인 성능 향상을 저비용으로 달성합니다.

- **Technical Details**: 제안된 방법은 두 단계로 진행됩니다. 첫 번째 단계에서는 SAM(Segment Anything) 모델을 사용하여 중복적인 세그멘테이션 제안을 생성합니다. 두 번째 단계에서는 CLIP(Contrastive Language-Image Pre-training) 모델을 Specific Category Enhancement Fine-tuning (SCEF) 전략으로 수정하여 SAM이 생성한 세그멘테이션 마스크 중 드라이버블 영역과 가장 관련이 깊은 것을 선택합니다.

- **Performance Highlights**: KITTI 도로 데이터셋을 통해 수행된 종합적 실험 결과, 제안된 방법이 기존 pre-training을 하지 않은 모델에 비해 전반적인 성능 향상을 이루었음을 보여줍니다. 나아가, 제안된 pre-training 방법은 전통적인 pre-training 접근 방식을 초월하며, 최신 self-training 기법과 비교해도 가장 우수한 성능을 달성하였습니다.



### Erase, then Redraw: A Novel Data Augmentation Approach for Free Space Detection Using Diffusion Mod (https://arxiv.org/abs/2409.20164)
- **What's New**: 이 논문에서는 도로 탐지 작업을 위한 새로운 데이터 증강(data augmentation) 방법을 제안합니다. 기존의 데이터 증강 방법이 고급 시맨틱 속성(semantic attributes)을 변화시키지 못하는 문제를 해결하기 위해 사전 훈련된 텍스트-투-이미지 확산 모델(text-to-image diffusion model)을 활용하여 이미지 변환을 매개변수화합니다.

- **Technical Details**: 제안하는 방법은 세 가지 주요 과정으로 구성됩니다. 첫째, Mask R-CNN 또는 Segment Anything(SAM)과 같은 전통적인 객체 분할(instance segmentation) 알고리즘을 사용하여 배경 인스턴스(인식 객체)를 삭제합니다. 둘째, 삭제된 영역을 사전 훈련된 확산 모델을 사용하여 복원합니다. 이 과정에서 다양한 언어적 프롬프트(prompt)를 통해 다른 객체 및 스타일로 복원할 수 있어 유연성이 크게 향상됩니다.

- **Performance Highlights**: KITTI 도로 데이터셋에서 실험한 결과, 우리의 데이터 증강 방법은 기존의 다른 데이터 증강 방법들과 비교해 도로 탐지 작업에서 최고의 성능을 보였습니다. 또한, 사용자가 쉽게 사용할 수 있도록 자동 및 수동 증강 데이터 생성 기능이 포함된 GUI 인터페이스도 통합되었습니다.



### VMAD: Visual-enhanced Multimodal Large Language Model for Zero-Shot Anomaly Detection (https://arxiv.org/abs/2409.20146)
- **What's New**: 이 논문에서는 Zero-shot Anomaly Detection (ZSAD)와 Multimodal Large Language Models (MLLMs)를 활용한 새로운 접근 방식을 제안합니다. 특히, VMAD(Visual-enhanced MLLM Anomaly Detection)라는 혁신적인 프레임워크를 통해 산업 이상 탐지(Industrial Anomaly Detection, IAD)에서의 한계를 극복하고자 합니다.

- **Technical Details**: VMAD는 두 가지 핵심 모듈, 즉 Defect-Sensitive Structure Learning (DSSL)과 Locality-enhanced Token Compression (LTC)을 도입합니다. DSSL은 패치 유사성 정보를 MLLM에 통합해 이상 탐지를 향상시키며, LTC는 다양한 수준의 지역적 특성을 발굴하여 세밀한 결함 검출을 돕습니다. 이 프레임워크는 이미지-텍스트 입력을 처리하고, 이상 마스크 생성을 위한 '[seg]' 토큰을 생성합니다.

- **Performance Highlights**: 실험 결과, VMAD는 MVTec-AD, Visa, WFDD, RIAD 등에서 기존 최첨단 방법들보다 우수한 성능을 보였습니다. VMAD는 이상 탐지의 정확성을 제공함과 동시에, 산업 결함에 대한 자세한 통찰을 제공합니다. 또한, RIAD라는 대규모 데이터 세트를 통해 IAD 연구에 기여할 수 있는 귀중한 자원을 제공합니다.



### RISE-SDF: a Relightable Information-Shared Signed Distance Field for Glossy Object Inverse Rendering (https://arxiv.org/abs/2409.20140)
- **What's New**: 이 논문에서는 고품질의 기하학적 구조와 물질적 특성을 재구성하고, 이를 통해 고품질의 재조명을 가능하게 하는 새로운 end-to-end relightable neural inverse rendering 시스템을 제안합니다.

- **Technical Details**: 이 시스템의 핵심은 장면 매개변수의 더 나은 분해를 학습하기 위한 두 단계 접근법입니다. 첫 번째 단계에서는 neural signed distance field (SDF)를 사용하여 반사 인식 반사 필드를 개발하고, MLP (multilayer perceptron)를 배치하여 간접 조명을 추정합니다. 두 번째 단계에서는 물리 기반 분해를 conjoint learning하기 위한 정보 공유 네트워크 구조를 도입합니다. 분해 과정에서는 및 Monte Carlo sampling의 노이즈를 줄이기 위해 split-sum approximation을 적용합니다.

- **Performance Highlights**: 실험 결과, 우리의 알고리즘은 glossy objects의 inverse rendering 및 relighting에서 state-of-the-art 성능을 달성했으며, 특히 반사성이 높은 객체의 재구성에서 강력한 결과를 보였습니다.



### Segmenting Wood Rot using Computer Vision Models (https://arxiv.org/abs/2409.20137)
Comments:
          FZI Workshop - Künstliche Intelligenz im Mittelstand (KI-KMU 2024)

- **What's New**: 본 연구에서는 목재 산업에서 원자재 품질 평가를 자동화하기 위한 AI 모델을 제시합니다. 이 모델은 나무 로그의 결함을 감지하고 정량화하며 위치를 특정하기 위한 것입니다.

- **Technical Details**: 연구에 사용된 데이터셋은 1424개의 샘플 이미지로 구성되어 있으며, 각 이미지에는 여러 결함 클래스가 포함되어 있습니다. 논문에서는 성능을 비교하기 위해 가장 최근의 InternImage 및 ONE-PEACE 아키텍처를 활용한 시맨틱 세분화(schematic segmentation) 모델을 훈련하고 미세 조정하였습니다.

- **Performance Highlights**: 최고 성능 모델은 평균 IoU(Intersection over Union) 0.71을 달성하였으며, 인간 주석자와 유사한 결함 감지 및 정량화 능력을 보여주었습니다.



### Machine Learning in Industrial Quality Control of Glass Bottle Prints (https://arxiv.org/abs/2409.20132)
Comments:
          VISAPP 2024 Conference

- **What's New**: 이 논문에서는 유리병 인쇄 품질 관리를 위한 두 가지 기계 학습 기반 접근 방식을 제시하고 평가하였습니다. 이러한 접근 방식은 반사와 제작 관련 편차에도 불구하고 적절한 인쇄 품질을 유지하기 위한 것입니다.

- **Technical Details**: 첫 번째 접근 방식은 Sobel 및 Canny 필터와 이미지 품질 메트릭(예: MSE, SSIM)을 사용하여 다양한 감독 분류 모델(예: SVM, k-Neighbors)과 함께 84%의 정확도를 달성했습니다. 두 번째 접근 방식은 미리 훈련된 CNN 모델(예: ResNet, VGG)을 미세 조정하여 이진 분류 작업을 수행했으며, 이 결과 87%의 정확도를 기록했습니다. Grad-CAM을 활용하여 자주 결함이 있는 인쇄 영역을 시각화했습니다.

- **Performance Highlights**: 본 연구는 병 제조 과정에서 발생할 수 있는 결함을 정확하게 감지할 수 있도록 하는 신뢰할 수 있는 품질 관리 시스템을 개발하는 데 기여하였습니다. 연구 결과, 기계 학습 기술을 통해 인쇄 품질의 최적화가 가능함을 입증하였습니다.



### PuzzleBoard: A New Camera Calibration Pattern with Position Encoding (https://arxiv.org/abs/2409.20127)
Comments:
          To be published in German Conference on Pattern Recognition (GCPR) 2024. Further details: this https URL

- **What's New**: 본 논문에서는 기존의 체스판(calibration patterns) 보드의 장점과 저해상도(very low resolutions)에서도 해독 가능한 경량(position coding) 방식을 결합한 새로운 캘리브레이션 패턴을 제시합니다. 이 방법은 체크보드 캘리브레이션의 장점을 지속적으로 유지하면서 오류 수정(error correction)과 계산 효율성을 포함하는 해독 알고리즘을 제공합니다. 또한 이 접근법은 기존의 체크보드 캘리브레이션 패턴 및 여러 체스판 캘리브레이션 알고리즘과 완벽하게 호환됩니다.

- **Technical Details**: 제안된 방법은 저해상도(very low resolutions)에서도 정확한 캘리브레이션을 가능하게 하는 경량(position encoding) 스킴을 사용합니다. 이 알고리즘은 빠른 속도로 안정적인 해독을 보장하며, 개방성을 통해 이미지를 처리하며 오클루전(occlusions)에 대응할 수 있습니다. 또한 측정 오류를 평균화할 수 있도록 많은 참조 포인트(reference points)를 획득하는 것이 중요하다는 아이디어를 기반으로 합니다.

- **Performance Highlights**: 이 방법은 카메라 캘리브레이션 뿐만 아니라 카메라 포즈 추정(camera pose estimation) 및 마커 기반 물체 위치 지정(marker-based object localization) 작업에서도 유용하게 사용될 수 있습니다. 확장성이 뛰어나고 형상 기반의 fiducial marker 대안으로도 활용될 수 있습니다.



### Training a Computer Vision Model for Commercial Bakeries with Primarily Synthetic Images (https://arxiv.org/abs/2409.20122)
Comments:
          FZI Workshop - Künstliche Intelligenz im Mittelstand (KI-KMU 2024)

- **What's New**: 이 연구는 식품 산업에서 재처리된 제품의 재고 관리를 자동화하기 위한 AI 어플리케이션을 발전시켰습니다. 2432개의 이미지를 포함한 확장된 데이터셋을 만들었으며, 새로운 구운 식품을 포함시켰습니다.

- **Technical Details**: 이 연구에서는 generative models인 pix2pix와 CycleGAN을 사용하여 합성 이미지를 생성하여 모델의 강인성을 높였습니다. YOLOv9와 YOLOv8이라는 최신 object detection 모델을 훈련하여 구운 식품 탐지 작업을 수행했습니다.

- **Performance Highlights**: 최종 모델은 테스트 세트에서 90.3%의 평균 정밀도(AP@0.5)를 달성하여 뛰어난 성능을 보였습니다.



### Masked Autoregressive Model for Weather Forecasting (https://arxiv.org/abs/2409.20117)
Comments:
          10 page. arXiv admin note: substantial text overlap with arXiv:2303.07849

- **What's New**: 최근 기후 변화의 영향으로 정확한 날씨 예측이 더욱 필수적이게 되었습니다. 기존의 autoregressive 접근법은 긴 예측에서 오류가 누적되는 문제가 있습니다. 이를 해결하기 위해, Masked Autoregressive Model for Weather Forecasting (MAM4WF)라는 새로운 모델을 제안합니다.

- **Technical Details**: MAM4WF는 입력 데이터의 일부가 훈련 중에 masking(마스킹)되어, 손실된 정보를 재구성하는 방식으로 강력한 spatiotemporal(공간-시간) 관계를 학습합니다. 이 모델은 autoregressive와 lead time embedding 방식의 장점을 결합하여 정확하고 유연한 예측을 가능하게 합니다.

- **Performance Highlights**: MAM4WF는 날씨, 기후 예측 및 비디오 프레임 예측 데이터셋을 포함한 여러 데이터셋에서 실험하였으며, 총 다섯 개의 테스트 데이터셋에서 우수한 성능을 보였습니다.



### REST-HANDS: Rehabilitation with Egocentric Vision Using Smartglasses for Treatment of Hands after Surviving Strok (https://arxiv.org/abs/2409.20116)
Comments:
          Accepted at ACVR ECCV 2024

- **What's New**: 이 연구에서는 상업적으로 이용 가능한 스마트안경인 RayBan Stories를 활용한 원격 손 재활의 가능성을 탐구합니다. 이는 자동화된 운동 인식, 자세 평가 및 반복 카운팅을 위한 오프라인 실험을 포함합니다.

- **Technical Details**: 연구팀은 REST-HANDS라는 최초의 에고센트릭(egocentric) 손 운동 비디오 데이터셋을 개발했습니다. 이 데이터셋은 운동 분류, 전문 물리치료사에 의한 자세 평가, 운동당 반복 회수와 같은 레이블을 포함하고 있으며, 최첨단 비디오 이해 방법을 사용하여 높은 정확률을 기록했습니다. 운동 인식의 경우 98.55%, 자세 평가의 경우 86.98%, 반복 카운팅의 경우 평균 절대 오차는 1.33입니다.

- **Performance Highlights**: 이 접근법은 스마트안경으로 촬영된 에고센트릭 비디오가 원격 재활에서 유용하게 사용될 수 있음을 입증합니다. 앞으로의 연구를 위한 길을 열었으며, 자동화된 재활 도구의 개발에 기여할 것으로 기대됩니다.



### CBAM-SwinT-BL: Small Rail Surface Detect Detection Method Based on Swin Transformer with Block Level CBAM Enhancemen (https://arxiv.org/abs/2409.20113)
Comments:
          27 pages, 17 figures

- **What's New**: 고강도 철도 운영 하에, 철도 선로의 결함 탐지 문제를 해결하기 위해 Swin Transformer(SwinT)를 기반으로 Convolutional Block Attention Module(CBAM)을 통합한 새로운 모델을 제안합니다. 이 연구는 특히 Dirt 및 Squat와 같은 소규모 결함 카테고리의 검출 성능을 개선하여 공공 안전과 서비스 신뢰성을 강화합니다.

- **Technical Details**: 제안된 프레임워크인 CBAM-Enhanced Swin Transformer in Block Level(CBAM-SwinT-BL)는 Swin Transformer 블록 안에 CBAM을 성공적으로 통합하여 소규모 철도 결함 탐지 성능을 향상시킵니다. 실험 결과, 이 프레임워크는 RIII 데이터셋에서 dirt 카테고리의 mAP-50이 +23.0%, dent 카테고리는 +38.3% 향상되었으며, MUET 데이터셋에서 squat 카테고리는 +13.2% 증가했습니다.

- **Performance Highlights**: CBAM-SwinT-BL은 RIII에서 +7% 및 MUET에서 +5%의 전반적인 정밀도 증가를 달성하여 각각 88.1%와 69.1%에 이르게 하며, 작고 세밀한 결함에 대한 탐지 정확도를 비약적으로 개선하였습니다. 추가 모듈인 CBAM은 모델 훈련 속도를 평균 +0.04s/iteration만 증가시켜 성능 향상과의 비해 수용 가능한 수준입니다.



### Learning to Discover Generalized Facial Expressions (https://arxiv.org/abs/2409.20098)
- **What's New**: 본 논문에서는 Facial Expression Category Discovery (FECD)라는 새로운 작업을 소개하여, 열린 세계의 표정 인식 (O-FER)에서의 문제를 해결하고자 합니다. 기존의 일반화된 카테고리 발견 (GCD) 방법들이 자연 이미지 데이터셋에 대해서는 연구되어 왔으나, 얼굴 표정 인식에 적용하는 것은 새로운 도전과제입니다.

- **Technical Details**: 두 가지 주요 편향, 즉 이론적 편향(Theoretical Bias)과 실제 편향(Practical Bias)을 정의했습니다. 이론적 편향은 새롭게 추가된 카테고리로 인해 생기며, 실제 편향은 얼굴 표정 데이터의 불균형성과 세부적인 차이로 인해 발생합니다. 본 논문에서는 FER-GCD라는 새로운 적대적(discriminative) 방법을 제안하며, 이는 명시적인(deep) 및 암시적인(implicit) 비편향 제어 과정을 통합합니다. F-discrepancy라는 새로운 메트릭을 정의하여 이론적 편향의 극한 값을 추정하고 이를 통해 모델이 편향을 최소화하도록 하는 방법을 포함하고 있습니다.

- **Performance Highlights**: FER-GCD는 기존 방법들과 비교하여 정확도가 평균 9.8% 향상되었으며, 최신 GCD 방법들을 능가하는 성능을 보여줍니다. 이러한 향상은 구식 카테고리와 새로운 카테고리 모두에서 관찰되었습니다.



### SurgPETL: Parameter-Efficient Image-to-Surgical-Video Transfer Learning for Surgical Phase Recognition (https://arxiv.org/abs/2409.20083)
Comments:
          submitted to TMI

- **What's New**: 본 논문에서는 예전의 이미지 레벨 사전 훈련 모델을 활용하여 세밀한 외과적 단계 인식을 전문화하는 방법, 즉 파라미터 효율적 이미지-외과 비디오 전이 학습(Parameter-Efficient Image-to-Surgical-Video Transfer Learning) 문제를 다루고 있습니다. 특히, 수술 비디오 데이터의 특성에 맞춘 새로운 벤치마크인 SurgPETL을 개발하였으며, 세 가지 고급 방법을 통해 실험을 수행했습니다.

- **Technical Details**: 본 연구에서 제안한 모델(SurgPETL)은 ViTs 기반의 두 가지 서로 다른 크기 모델을 사용하며, 다섯 가지 대규모 자연 및 의료 데이터셋으로 사전 훈련되었습니다. Spatial-Temporal Adaptation(STA) 모듈은 표준 spatial adapter와 새로운 temporal adapter를 통합하여 세밀한 공간 특징을 포착하고 시간적 시퀀스 간의 연결을 구축합니다. 이는 robust spatial-temporal modeling을 위한 것입니다.

- **Performance Highlights**: SurgPETL과 STA를 사용한 extensive experiments 결과, 다양한 외과 수술 절차에 걸쳐 세 개의 도전적인 데이터셋에서 기존의 파라미터 효율적 대안 및 최첨단 외과 단계 인식 모델을 능가하는 성능을 보였습니다.



### ProFD: Prompt-Guided Feature Disentangling for Occluded Person Re-Identification (https://arxiv.org/abs/2409.20081)
Comments:
          Accepted by ACM MM 2024

- **What's New**: 이 논문에서는 사람 재식별(Person Re-Identification, ReID) 작업에서 발생하는 가림 현상 문제를 해결하기 위한 새로운 방법인 Prompt-guided Feature Disentangling (ProFD)을 제안합니다. ProFD는 사전 훈련된 텍스트 지식을 활용하여 신뢰성 있는 부분 특징을 생성하며, 이는 인식 성능을 향상시킵니다.

- **Technical Details**: ProFD는 부분별 프롬프트를 설계하고, 불필요한 세그멘테이션 마스크를 사용하여 시각적 및 텍스트 임베딩을 초기 정렬합니다. 이는 텍스트 프롬프트가 공간 인식을 가질 수 있도록 합니다. 또한, 하이브리드-어텐션 디코더를 사용하여 공간 및 의미적인 일관성을 유지하면서 노이즈 영향을 최소화합니다. 자가 증류 전략을 사용하여 CLIP의 사전 훈련된 지식을 보존하여 과적합(overfitting)을 완화합니다.

- **Performance Highlights**: Market1501, DukeMTMC-ReID, Occluded-Duke, Occluded-ReID, 및 P-DukeMTMC 데이터세트에서 ProFD가 최첨단 성능을 달성하는 것으로 나타났습니다. Occluded-ReID 데이터세트에서는 mAP에서 8.3%, Rank-1 정확도에서 4.8% 개선을 기록하며 다른 방법들보다 훨씬 뛰어난 일반화 능력을 보여주었습니다.



### Q-Bench-Video: Benchmarking the Video Quality Understanding of LMMs (https://arxiv.org/abs/2409.20063)
- **What's New**: 이 논문에서는 'Q-Bench-Video'라는 새로운 벤치마크를 소개합니다. 이 벤치마크는 LMMs의 비디오 품질 이해 능력을 평가하기 위해 특별히 설계되었습니다. Q-Bench-Video는 다양한 비디오 소스와 함께 비디오 품질 측정을 위한 포괄적인 질문 세트를 포함하고 있습니다.

- **Technical Details**: Q-Bench-Video는 자연 장면, AI 생성 콘텐츠(AIGC), 컴퓨터 그래픽(CG)을 포함한 다양한 출처의 비디오를 포함하여 LMMs의 비디오 품질 이해 능력을 평가합니다. 평가 기준으로는 기술적 왜곡, 미적 품질, 시간적 왜곡 및 AIGC 왜곡이 포함됩니다. 기존의 다지선다형 질문 외에 개방형 질문과 비디오 쌍 품질 비교 질문을 포함하여 더욱 복잡한 시나리오에 대한 평가를 가능하게 합니다.

- **Performance Highlights**: 실험 결과, LMMs는 비디오 품질에 대한 기본적인 이해를 나타내지만, 인간 성능과 비교할 때 여전히 뚜렷한 차이가 있음을 발견했습니다. 이러한 결과는 LMM의 성능에 대한 통찰력을 제공하고, 향후 비디오 품질 이해 향상을 위한 가능성을 제시합니다.



### Lightweight Neural Architecture Search for Cerebral Palsy Detection (https://arxiv.org/abs/2409.20060)
- **What's New**: 이번 논문에서는 뇌성마비(Cerebral Palsy, CP) 진단을 위한 신경망 구성 및 하이퍼 파라미터 최적화를 위한 신경망 아키텍처 검색(Neural Architecture Search, NAS) 알고리즘을 제안합니다. 기존의 기계 학습 접근법에 비해 더 나은 성능을 발휘하여 자원 제한 환경에서의 적용 가능성을 높였습니다.

- **Technical Details**: 제안된 NAS 알고리즘은 강화 학습 업데이트 방식을 이용하여 효율적으로 최적화된 신경망 구성을 발견하는 데 중점을 두고 있습니다. 이 과정에서 입력 데이터로 영아의 뼈대 데이터를 사용하고, 이동 포지션, 속도, 뼈, 가속도와 같은 4가지 특징 카테고리를 사용하여 모델 학습을 진행합니다.

- **Performance Highlights**: 이 NAS 방식은 기존의 대형 앙상블 모델보다 감도(Sensitivity) 포함 장점이 있으며, 더 가벼운 모델 아키텍처를 제공하여 제한된 처리 능력을 가진 장치에서도 효율적으로 사용할 수 있도록 합니다. 따라서, 특히 자원이 제한된 지역에서 CP를 조기에 진단하는 데 필요한 임상 작업 흐름에 통합될 수 있는 가능성을 보여줍니다.



### OPONeRF: One-Point-One NeRF for Robust Neural Rendering (https://arxiv.org/abs/2409.20043)
- **What's New**: 이 논문에서는 강력한 장면 렌더링을 위한 One-Point-One NeRF (OPONeRF) 프레임워크를 제안합니다. OPONeRF는 훈련 중과 테스트 중에 장면이 변하지 않는다는 기존 NeRF의 주요 가정에서 벗어나, 예상치 못한 변화에 적응할 수 있도록 설계되었습니다.

- **Technical Details**: OPONeRF는 지역 장면 변화를 반영하기 위해 독립적으로 작동하는 포인트별 렌더러 매개변수를 개인화하여, 3D 장면의 변수성을 모델링합니다. 이 프레임워크는 결정론적 매핑과 확률적 추론으로 포인트 표현을 분해하여 지역적 불확실성을 캡처합니다.

- **Performance Highlights**: 실험 결과, OPONeRF는 기존 최신 NeRF 모델들을 능가하는 성능을 보였으며, 다양한 평가 지표에서 더욱 우수한 결과를 나타냈습니다. OPONeRF는 또한 전통적인 일반화 기반 벤치마크에서도 경쟁력을 갖추고 있음을 보여주었습니다.



### Camera Calibration using a Collimator System (https://arxiv.org/abs/2409.20034)
Comments:
          Accepted by ECCV2024 (oral presentation)

- **What's New**: 본 논문에서는 카메라의 교정(calibration) 과정에서 고유한 콜리메이터 시스템(collider system)을 이용하여 다양한 작업 거리에서 신뢰할 수 있고 제어 가능한 교정 환경을 제공하는 새로운 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 카메라가 두 개 이상의 다른 방향에서 콜리메이터 패턴을 관찰함으로써 구현됩니다. 우리는 칼리메이터에 의한 상대 운동이 구면 운동 모델(spherical motion model)에 부합함을 증명하고, 이를 통해 원래의 6DOF 상대 운동을 3DOF 순수 회전 운동으로 축소합니다.

- **Performance Highlights**: 우리는 제안한 방법의 성능을 합성(synthetic) 및 실제(real-world) 실험에서 평가하였으며, 콜리메이터 시스템을 이용한 교정의 실행 가능성을 입증하고, 기존의 최첨단 방법들과 비교하여 우수하다는 것을 보여주었습니다.



### Visual Context Window Extension: A New Perspective for Long Video Understanding (https://arxiv.org/abs/2409.20018)
Comments:
          14 pages, 4 figures

- **What's New**: 본 논문에서는 Long Video 이해를 위한 새로운 접근 방식을 제안하여, Large Multimodal Models (LMMs)가 긴 비디오 작업에 효과적으로 적용될 수 있도록 합니다. 기존 방법들은 대량의 긴 비디오 데이터셋에서 다시 학습해야 하는 어려움이 있었습니다.

- **Technical Details**: 문맥(window) 관점에서 접근하여 시각적 및 언어적 토큰 간의 불일치를 해결합니다. 이에 따라 visual context window를 확장하여 LMMs가 긴 비디오 작업을 처리할 수 있도록 하고, progressive pooling inference 전략을 도입하여 메모리 소모를 줄입니다.

- **Performance Highlights**: 여러 긴 비디오 이해 기준 벤치마크에서, 제안된 방법은 비디오 프레임 수가 증가함에 따라 성능이 지속적으로 향상되었습니다. MLVU 벤치마크에서, 제안된 방법은 모델 크기가 7B에 불과한 GPT-4o를 능가했습니다.



### Single-shot reconstruction of three-dimensional morphology of biological cells in digital holographic microscopy using a physics-driven neural network (https://arxiv.org/abs/2409.20013)
Comments:
          35 pages, 7 figures, 1 table

- **What's New**: 이 논문에서는 MorpHoloNet이라는 새로운 심층 학습 모델을 제안합니다. 이 모델은 생물학적 세포의 단일 촬영 홀로그램에서 3D 형태를 복원하기 위한 물리 기반 및 좌표 기반 신경망을 통합하여 다양한 기술적 한계를 극복합니다.

- **Technical Details**: MorpHoloNet은 3D 위상 변환 분포를 통해 일관한 빛의 광학 회절을 시뮬레이션하여 설계되었습니다. 이 모델은 센서 평면에서 입력 홀로그램과 시뮬레이션된 홀로그램 간의 손실을 최소화하여 최적화됩니다.

- **Performance Highlights**: MorpHoloNet은 기존 DIHM 방법에 비해 단일 촬영 홀로그램으로부터 3D 복잡한 빛의 장과 형태를 직접 복원할 수 있습니다. 실험적으로 생물학적 세포의 홀로그램을 사용하여 3D 형태와 굴절률 분포를 성공적으로 복원하였습니다.



### Multibiometrics Using a Single Face Imag (https://arxiv.org/abs/2409.20003)
Comments:
          APSIPA ASC 2024

- **What's New**: 본 논문에서 제안하는 멀티바이오메트릭스(multi-biometrics) 방법은 단일 얼굴 이미지에서 얼굴(face), 홍채(iris), 얼굴 주변(periocular), 코(nose), 눈썹(eyebrow) 등 다섯 가지 생체 특성을 조합하여 인식을 향상시키도록 설계되었습니다. 기존 연구들과는 달리, 제안된 방법은 단 하나의 이미지로 여러 생체 정보를 추출하여 편리함을 유지하면서 성능을 개선합니다.

- **Technical Details**: 제안된 방법은 Mediapipe FaceMesh를 사용하여 얼굴의 주요 키포인트(keypoint)를 감지한 후, 각 생체 특성에 대한 이미지를 생성합니다. 그런 다음, Convolutional Neural Network (CNN)을 통해 특징을 추출하고 각 특성에 대한 매칭 점수를 계산합니다. 마지막으로, 가중 합(weighted sum)을 통해 최종 매칭 점수를 얻습니다.

- **Performance Highlights**: CASIA Iris Distance 데이터베이스를 사용한 다양한 실험을 통해, 제안된 멀티바이오메트릭스 방법의 효과성을 입증하였습니다. 이 연구는 한 이미지에서 다섯 가지의 생체 특성을 효과적으로 추출하여 인식 성능을 높인 첫 사례로 주목받고 있습니다.



### A large-scale operational study of fingerprint quality and demographics (https://arxiv.org/abs/2409.19992)
Comments:
          Extended journal version submitted to IET Biometrics. 10 pages, 5 figures Reference conference paper: J. Galbally, A. Cepilovs, R. Blanco-Gonzalo, G. Ormiston, O. Miguel-Hurtado, and I. S. Racz, 'Fingerprint quality per individual finger type: A large-scale study on real operational data' in Proc. IEEE Intl. Workshop on Biometrics and Forensics 2023 (IWBF 2023)

- **What's New**: 본 논문은 16,000명의 대규모 데이터베이스를 사용하여 지문 인식 기술의 정확성이 성별, 연령 및 지문 유형 등 특정 인구 통계학적 그룹에 따라 편향되어 있는지를 조사합니다. 이전 연구들보다 더 많은 표본을 통해 지문 품질과 인구 통계학과의 관계를 심층적으로 분석했습니다.

- **Technical Details**: 연구는 500dpi (dots per inch) 터치 기반 광학 스캐너를 이용하여 문서 발급을 위한 비자 처리를 위해 전 세계 34개 비EU 국가에서 수집된 15,942개의 10-프린트 디지털 기록을 분석합니다. 이 데이터베이스는 성별, 연령, 출신 국가와 같은 메타데이터와 함께 지문 샘플을 포함합니다.

- **Performance Highlights**: 지문 인식 품질은 노화, 성별 및 지문 유형에 따라 차이가 있으며, 이는 여러 인구 집단에서 성능 변동을 일으킵니다. 연구 결과, 서로 다른 집단에 대해 지문 인식 기술의 성능 일관성을 향상시키기 위한 개선 방향을 제안하고 있습니다.



### RoCoTex: A Robust Method for Consistent Texture Synthesis with Diffusion Models (https://arxiv.org/abs/2409.19989)
Comments:
          11 pages, 13 figures

- **What's New**: 이 논문에서는 일관된 무 seam 질감을 생성하기 위한 강력한 text-to-texture 생성 방법인 RoCoTex를 제안합니다. 이 방법은 2D diffusion 모델인 SDXL 및 여러 ControlNets를 사용하여 생성된 질감과 기본 메쉬 간의 정렬 문제를 해결합니다.

- **Technical Details**: RoCoTex는 대칭적(view) 뷰 합성 전략과 지역 프롬프트를 사용하여 뷰 일관성을 향상시키며, SDXL을 기반으로 한 혁신적인 질감 혼합 및 소프트 인페인팅 기법을 도입합니다. 이러한 과정을 통해 seam 영역을 효과적으로 줄이고 있습니다.

- **Performance Highlights**: 광범위한 실험 결과, RoCoTex는 기존의 최첨단 방법들에 비해 뛰어난 성능을 나타내며 강력하고 일관된 질감을 생성하는 데 성공적으로 기여합니다.



### OccRWKV: Rethinking Efficient 3D Semantic Occupancy Prediction with Linear Complexity (https://arxiv.org/abs/2409.19987)
- **What's New**: OccRWKV 네트워크는 3D 장면의 기하학적 및 의미 구조를 예측하며, 효율적인 계산으로 실시간 예측을 가능하게 합니다.

- **Technical Details**: OccRWKV는 Semantic, Occupancy 예측 및 Feature Fusion을 독립적인 브랜치로 분리하여 각각 Sem-RWKV 및 Geo-RWKV 블록을 통합합니다. 이 블록들은 Long-range dependencies를 캡처하여 Domain-specific representation을 학습하도록 설계되었습니다.

- **Performance Highlights**: OccRWKV는 SemanticKITTI 데이터셋에서 기존 최첨단 방법보다 뛰어난 mIoU 25.1을 기록하였고, Co-Occ 대비 20배 더 빠른 22.2 FPS로 수행하면서 78.5%의 파라미터 수를 감소시켰습니다.



### GearTrack: Automating 6D Pose Estimation (https://arxiv.org/abs/2409.19986)
- **What's New**: 본 연구는 FoundationPose, SAM2, LightGlue를 통합하여 산업적 응용 분야에서 실시간 6D 객체 탐지를 위한 강력한 솔루션을 개발하였습니다. Retraining(재훈련) 없이도 사용할 수 있으며, 초기 프레임에서 객체 마스크가 필요했던 FoundationPose의 단점을 보완하였습니다.

- **Technical Details**: 이 알고리즘은 CAD 모델만 필요하며, 첫 설정 시 사용자가 라이브 피드에서 객체 위치를 클릭하여 마크합니다. 초기 이미지가 저장되고, 이후에는 LightGlue를 활용하여 실시간 장면과 객체 간의 feature matching을 수행하여 탐지를 위한 프롬프트를 생성합니다.

- **Performance Highlights**: YCB 데이터셋과 산업 부품(표백 클리너 및 기어 등)을 사용하여 신뢰할 수 있는 6D 탐지 및 추적 성능을 입증하였고, SAM2와 FoundationPose의 통합으로 occlusion(차폐) 및 빠른 동작과 같은 어려운 조건에서 지속적이고 정확한 추적을 보장합니다.



### TSdetector: Temporal-Spatial Self-correction Collaborative Learning for Colonoscopy Video Detection (https://arxiv.org/abs/2409.19983)
- **What's New**: 이번 연구는 복잡한 대장 내시경 비디오 씬에서 폴립을 정확하게 탐지하기 위한 Temporal-Spatial self-correction network인 TSdetector를 제안합니다. 이 모델은 시퀀스 간 글로벌 특성을 포커싱하여 동적 데이터에 적응할 수 있도록 설계되었습니다.

- **Technical Details**: TSdetector는 두 가지 주요 학습 단계를 포함합니다: 1) Temporal-level consistency learning과 2) Spatial-level reliability learning. 이를 위해 Global Temporal-aware Convolution (GT-Conv)을 사용하여 시간 맥락에 따른 동적 커널 가중치를 생성하고, Hierarchical Queue Integration Mechanism (HQIM)을 통해 다중 시간 특성을 점진적으로 통합합니다. 또한, Spatial-level에서는 Position-Aware Clustering (PAC)을 통해 예측 신뢰도를 재조정합니다.

- **Performance Highlights**: TSdetector는 SUN, CVC-ClinicDB, PICCOLO의 세 가지 공공 폴립 비디오 데이터세트에서 기존의 최신 방법들을 초월하여 높은 폴립 탐지율을 달성하였습니다. 이 연구는 객체 탐지의 효율성과 정확성을 크게 향상시키는 새로운 방법론을 제시합니다.



### DAOcc: 3D Object Detection Assisted Multi-Sensor Fusion for 3D Occupancy Prediction (https://arxiv.org/abs/2409.19972)
- **What's New**: 본 논문에서는 다중 센서 융합 기술을 이용한 새로운 3D 시맨틱 점유 예측 네트워크인 DAOcc를 제안합니다. 이 모델은 3D 객체 탐지 감독을 통해 성능 향상을 이루며, 실용적인 입력 이미지 해상도에서 작동합니다.

- **Technical Details**: DAOcc는 이미지와 포인트 클라우드에서 피쳐를 각각 추출하고, 간단한 융합 전략을 채택하여 BEV(Bird's Eye View) 피쳐를 통합합니다. 3D 객탐지 감독을 통해 좌표 및 구조 정보를 보강하고, BEV 뷰 범위 확장 전략(BVRE)을 도입하여 이미지 해상도를 축소하고도 더 넓은 맥락 정보를 제공합니다. ResNet50과 256x704 해상도를 사용하여 구현되었습니다.

- **Performance Highlights**: DAOcc는 Occ3D-nuScenes 및 SurroundOcc 데이터셋에서 최첨단 성능을 달성하며, Occ3D-nuScenes 검증 세트에서 53.82 mIoU, SurroundOcc 검증 세트에서 45.0 IoU를 기록했습니다.



### Magnet: We Never Know How Text-to-Image Diffusion Models Work, Until We Learn How Vision-Language Models Function (https://arxiv.org/abs/2409.19967)
Comments:
          Accepted to NeurIPS 2024. Code is available at this https URL

- **What's New**: 본 논문에서는 CLIP 텍스트 인코더의 한계와 다중 속성 및 객체를 포함하는 복잡한 프롬프트를 정확하게 표현하는 이미지 생성 과정에서의 문제를 심도 있게 분석합니다. 특히, 'Magnet'라는 새로운 훈련 필요 없는 접근 방식을 통해 속성 결합 문제를 해결하는 방법을 제안합니다.

- **Technical Details**: CLIP 텍스트 인코더는 단방향 맥락을 생성하기 위해 인과 마스크(causal mask) 메커니즘을 사용합니다. 본 연구에서는 텍스트 인코딩 시 속성을 이해하는 방식을 분석하고, 양/음 속성 결합 벡터를 도입하여 속성 분리를 강화합니다. 이는 텍스트 공간에서 여러 객체 간의 명확한 속성 구분을 가능하게 합니다.

- **Performance Highlights**: 포괄적인 실험 결과 'Magnet' 접근 방식이 이미지 합성 품질과 속성 결합 정확도를 비슷한 계산 비용으로 크게 개선함을 보여줍니다. 이를 통해 비정형적이고 자연스럽지 않은 개념의 생성을 지원할 수 있습니다.



### Multimodal LLM Enhanced Cross-lingual Cross-modal Retrieva (https://arxiv.org/abs/2409.19961)
Comments:
          Accepted by ACM Multimedia

- **What's New**: 이 논문은 Cross-lingual Cross-modal Retrieval (CCR) 문제를 해결하기 위해 새로운 접근 방식인 LECCR을 제안합니다. LECCR은 다중 모달 대형 언어 모델(MLLM)을 통합하여 시각(feature) 및 비영어(non-English) 텍스트 표현 간의 정렬을 개선하는 데 중점을 두고 있습니다. 이는 인간 주석이 필요 없는 비영어 쿼리 기반 시각 콘텐츠 검색을 목표로 합니다.

- **Technical Details**: LECCR은 MLLM을 사용하여 상세한 시각 콘텐츠 설명을 생성하고, 이를 다중 뷰 의미론적 슬롯으로 집계하여 각기 다른 의미를 캡슐화합니다. 이러한 의미론적 슬롯을 내부 특성으로 사용하여 시각적 특성과 상호작용합니다. 또한, 비주얼과 비영어 특징 간의 정렬을 향상시키기 위해 영어 안내 아래 부드러운 매칭(softened matching) 기법을 도입합니다.

- **Performance Highlights**: Multi30K, MSCOCO, VATEX, MSR-VTT-CN의 네 가지 CCR 벤치마크에서 LECCR의 성능을 실험한 결과, 대부분의 평가 설정에서 이전 방법들을 초과하는 결과를 나타냈습니다. 이는 제안된 방법이 CCR 작업에서 효과적임을 강조합니다.



### TROPE: TRaining-Free Object-Part Enhancement for Seamlessly Improving Fine-Grained Zero-Shot Image Captioning (https://arxiv.org/abs/2409.19960)
Comments:
          Accepted to EMNLP 2024 Findings

- **What's New**: 본 논문은 TRaining-Free Object-Part Enhancement (TROPE)라는 새로운 방법을 소개하여, 이미지 캡셔닝에 있어 제로샷(zero-shot) 능력을 강화하는 데 초점을 맞추고 있습니다. TROPE는 기존 캡션의 세부사항을 보완하여, 다양한 객체의 세부 정보를 통합합니다.

- **Technical Details**: TROPE는 객체 탐지기(Object Detector) 제안과 자연어 처리(NLP) 기술을 활용하여 기초 캡션(base caption)에 추가적인 객체 부분 정보를 더합니다. 이 방법은 기본 캡션을 변경하지 않고 보완하여 캡션 생성 프로세스에 유연성을 제공합니다. TROPE는 기존 제로샷 이미지 캡셔닝 방법의 기반 캡션에 세부 정보를 효과적으로 추가합니다.

- **Performance Highlights**: TROPE는 모든 테스트된 제로샷 이미지 캡셔닝 접근 방식에서 성능을 일관되게 향상시키며, 세부 묘사가 요구되는 정밀한 이미지 캡셔닝 데이터셋에서 최신 기술(state-of-the-art) 성과를 달성했습니다. 평가 결과에 따르면, TROPE 사용 시 재현율(recall)이 크게 향상되었으며, 이는 기존의 방법들에 비해 더욱 세밀한 구성 요소를 캡션에 통합할 수 있음을 보여줍니다.



### Attribute-Text Guided Forgetting Compensation for Lifelong Person Re-Identification (https://arxiv.org/abs/2409.19954)
Comments:
          9 pages, 4 figures

- **What's New**: 이 논문은 Lifelong person re-identification (LReID) 문제를 해결하기 위해 새로운 모델인 attribute-text guided forgetting compensation (ATFC)을 제안합니다. ATFC는 작업별 도메인 격차를 해소하고 모델의 성능을 향상시키기 위해 텍스트 기반의 전역 표현과 속성 기반의 지역 정보를 통합하여 파악합니다.

- **Technical Details**: ATFC 모델은 텍스트와 속성을 기반으로 한 작업 공유 표현을 탐구합니다. 주요 구성 요소로는 속성-텍스트 생성기(Attribute-Text Generator, ATG), 텍스트 기반 집계 네트워크(Text-Guided Aggregation Network, TGA), 속성 보상 네트워크(Attribute Compensation Network, ACN) 등이 있습니다. 이들 각 네트워크는 텍스트와 이미지 페어를 동적으로 생성하고, 이를 통해 강력한 전역 및 지역 표현을 생성하여 도메인 격차를 줄이는 역할을 합니다.

- **Performance Highlights**: ATFC 모델은 기존 LReID 방법들보다 평균 mAP(Mean Average Precision) 및 R-1(Top-1 Recall)에서 각각 9.0%와 7.4% 향상된 성능을 보여주었습니다. 이를 통해 ATFC 방법이 다양한 환경에서 사람을 재식별하는데 효과적임을 입증하였습니다.



### Image Copy Detection for Diffusion Models (https://arxiv.org/abs/2409.19952)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 이 논문에서는 디퓨전 모델(diffusion models)에 의해 생성된 이미지의 내용 원본성 문제를 해결하기 위해, 이를 전문으로 하는 첫 번째 이미지 복사 탐지기(Image Copy Detection, ICD)인 ICDiff를 소개합니다.

- **Technical Details**: 연구진은 D-Rep(Diffusion-Replication) 데이터셋을 구축하고, 각 이미지-복제 쌍의 복제 수준을 확률 밀도 함수(Probability Density Function, PDF)로 변환하는 혁신적인 깊은 임베딩(deep embedding) 방법인 PDF-Embedding을 제안합니다. D-Rep 데이터셋은 최신 디퓨전 모델인 Stable Diffusion V1.5를 사용하여 40,000개의 이미지-복제 쌍을 생성하고, 이를 0(복제 없음)에서 5(완전 복제)까지 6가지 복제 수준으로 수동 주석을 추가하였습니다.

- **Performance Highlights**: PDF-Embedding 방식은 D-Rep 테스트 세트에서 기존의 프로토콜 기반 방법과 비-PDF 방법들을 초월했으며, 유명 디퓨전 모델의 복제 비율은 오픈 소스 갤러리를 기준으로 10%에서 20%에 이르는 것으로 나타났습니다.



### Illustrious: an Open Advanced Illustration Mod (https://arxiv.org/abs/2409.19946)
- **What's New**: 이 논문에서는 텍스트-투-이미지 애니메이션 생성 모델인 Illustrious의 최신 기능과 고해상도 및 다이나믹한 색 범위를 구현하기 위한 세 가지 주요 접근 방식을 다루고 있습니다.

- **Technical Details**: 모델 개선을 위해 배치 크기와 드롭아웃 제어의 중요성을 강조하며, 훈련 해상도를 증가시키고, 다단계 캡션을 제안하여 다양한 태그와 자연어 캡션을 커버합니다. 이는 모델의 성능을 높이는 중요한 요인으로 작용합니다.

- **Performance Highlights**: Illustrious는 애니메이션 스타일에서 최첨단 성능을 보여주며, 기존 모델들보다 더 나은 성능을 발휘합니다. 고해상도 이미지 생성을 가능하게 하고, 사용자 맞춤형 구현을 용이하게 합니다.



### CycleCrash: A Dataset of Bicycle Collision Videos for Collision Prediction and Analysis (https://arxiv.org/abs/2409.19942)
- **What's New**: 이 논문에서는 자전거 사고와 안전에 대한 연구가 부족한 것을 해결하기 위해, CycleCrash라는 새로운 데이터셋을 소개합니다. 이 데이터셋은 3,000개의 대시캠 영상과 436,347 프레임으로 구성되어 있으며, 자전거와의 사고 뿐만 아니라 안전한 상호작용을 포함한 다양한 상황을 포착하고 있습니다.

- **Technical Details**: CycleCrash는 자전거와 관련된 13종의 주석이 있는 3,000개의 동영상 클립으로 구성되어 있으며, 자전거 사고 예측과 분류를 위한 9가지 주요 작업을 정의합니다. 또한, ConvNeXt 스페셜 인코더와 비정상 트랜스포머(non-stationary transformer)를 활용하여 동영상의 시간적 동역학을 캡쳐하는 새로운 방법인 VidNeXt를 제안합니다. 이는 자전거 안전을 위한 데이터에서 정의된 작업을 수행하기 위해 설계되었습니다.

- **Performance Highlights**: VidNeXt 모델은 CycleCrash의 다양한 작업에서 여러 기준선 모델 대비 우수한 성능을 보여주었으며, 자전거 사고 예측 및 시간-충돌 예측 같은 중요한 과제들을 수행합니다. 또한, CycleCrash 데이터셋과 관련된 코드도 공개할 예정입니다.



### MaskMamba: A Hybrid Mamba-Transformer Model for Masked Image Generation (https://arxiv.org/abs/2409.19937)
- **What's New**: MaskMamba는 Mamba 아키텍처와 Transformer 아키텍처를 결합한 혁신적인 하이브리드 모델로, Masked Image Modeling을 이용한 비자기회적 이미지 합성을 가능하게 합니다. 이는 현재까지의 이미지 생성 모델의 복잡성과 확장성 문제를 해결하기 위한 것입니다.

- **Technical Details**: MaskMamba는 다음과 같은 주요 변경 사항을 도입했습니다: (1) 전향 합성(convolution) 대신 표준 합성을 사용하여 전역 컨텍스트를 더욱 잘 포착하고, (2) 성능을 극대화하고 추론 속도를 가속화하기 위해 곱셈(multiplication) 대신 연결(concatenation)을 사용합니다. 이를 통해 MaskMamba는 Bi-Mamba에 비해 17.77%의 추론 속도 개선을 달성하였습니다.

- **Performance Highlights**: MaskMamba는 ImageNet1k 및 CC3M 데이터셋에서 Mamba 및 Transformer 기반 모델들보다 우수한 생성 품질 및 추론 속도를 기록하였습니다. 특히, 2048x2048 해상도에서 54.44%의 추론 속도 개선을 보여줍니다.



### Leveraging Pre-trained Models for Robust Federated Learning for Kidney Stone Type Recognition (https://arxiv.org/abs/2409.19934)
- **What's New**: 이번 연구에서는 Federated Learning (FL) 프레임워크를 통해 전이 학습된 모델을 활용하여 신장 결석 진단을 개선하는 방법을 제안합니다. 다양한 이미지 부패에 대한 모델의 견고성을 높이는 데 초점을 맞추어, 두 개의 신장 결석 데이터셋을 이용하여 실험을 진행하였습니다.

- **Technical Details**: 우리의 방법은 두 단계로 나뉩니다: Learning Parameter Optimization (LPO)와 Federated Robustness Validation (FRV). LPO 단계에서는 7개의 에포크와 10회의 라운드를 통해 최고 정확도 84.1%를 달성하였고, FRV 단계에서는 77.2%의 정확도를 기록하여 이미지 부패에 대한 강인성을 증명하였습니다.

- **Performance Highlights**: 제안된 FL 접근 방식은 의료 진단에서 프라이버시와 성능 문제를 해결할 잠재력을 보여 주며, 환자 관리 개선 및 FL 기반 의료 시스템에 대한 신뢰성을 증대시키는 결과를 나타냅니다.



### CCDepth: A Lightweight Self-supervised Depth Estimation Network with Enhanced Interpretability (https://arxiv.org/abs/2409.19933)
- **What's New**: 이번 연구는 경량화 및 해석 가능성을 강화한 하이브리드 자기 지도 깊이 추정 네트워크 CCDepth를 제안합니다. 이 네트워크는 CNN과 CRATE（Coding RAte reduction TransformEr）모듈을 결합하여, 지역적 및 전역적 정보 추출을 효율적으로 수행합니다.

- **Technical Details**: CCDepth 네트워크는 U-Net 구조를 기반으로 하며, CNN을 통해 고해상도 이미지에서 세부 지역 특징을 추출하고, CRATE를 통해 저해상도 이미지에서 전역 정보를 추출합니다. 이 네트워크는 자기 지도 학습 접근법으로 훈련되며, 이미지를 참조 이미지와 비교하여 깊이 정보를 예측합니다.

- **Performance Highlights**: KITTI 데이터셋에서 실시한 실험 결과, CCDepth 네트워크는 최신 기술들과 비교하여 유사한 성능을 보이면서도 모델 크기가 크게 줄어드는 효과를 나타냈습니다.



### EndoDepth: A Benchmark for Assessing Robustness in Endoscopic Depth Prediction (https://arxiv.org/abs/2409.19930)
- **What's New**: 이번 연구에서는 EndoDepth benchmark라는 새로운 평가 프레임워크를 소개하며, 이는 내시경(depth estimation) 환경에서 단안(depth prediction)을 위한 모델의 강인성을 평가하는 데 중점을 두고 설계되었습니다.

- **Technical Details**: EndoDepth benchmark는 내시경 환경에서 발생하는 다양한 왜곡을 처리할 수 있는 robustness evaluation을 최초로 체계적으로 설계하였으며, Mean Depth Estimation Robustness Score (mDERS)라는 새로운 복합 지표를 도입하여 내시경 이미지의 왜곡에 대한 모델의 정확성과 오류 저항력을 측정합니다.

- **Performance Highlights**: 우리의 실험을 통해 최첨단 self-supervised depth prediction 모델의 내구성을 평가하였으며, 다양한 내시경 이미지 아티팩트에 대한 강점과 약점을 밝혔습니다. 이 연구결과는 내시경에서의 정확한 depth estimation을 위한 특별한 기술의 중요성을 강조하고, 향후 연구 방향에 대한 귀중한 통찰력을 제공합니다.



### Replace Anyone in Videos (https://arxiv.org/abs/2409.19911)
Comments:
          Work in progress

- **What's New**: 인간 중심의 비디오 생성 분야에서 최근 향상된 기술, 특히 diffusion 모델의 발전이 주목받고 있습니다. 그러나 인물의 정확한 이동을 제어하고 원하는 동작 패턴을 유지하며 비디오에 인물을 교체하거나 삽입하는 것은 여전히 도전 과제가 남아 있습니다.

- **Technical Details**: 이 논문에서는 ReplaceAnyone 프레임워크를 제안합니다. 이는 다양한 배경에서 인간의 동작을 국소화(localize)하고 조Manipulate하는 데 중점을 둡니다. 우리는 이 작업을 이미지 조건(image-conditioned) 포즈 기반(pose-driven) 비디오 인페인팅(video inpainting) 패러다임으로 형성하며, 마스크된 비디오 영역 내에서 이미지 조건 포즈 기반 비디오 생성과 인페인팅을 촉진하는 통합된 비디오 확산(video diffusion) 아키텍처를 사용합니다. 또한, 우리는 정규 및 비정규 형태의 다양한 마스크 형태를 도입하여 형태 누출(shape leakage)을 방지하고 세밀한 제어(granular local control)를 가능하게 합니다. 이 방법은 두 단계 학습 방법론을 구현하여, 처음에 이미지 조건 포즈 기반 비디오 생성 모델을 학습하고, 이어서 마스크 영역 내에서 비디오 인페인팅을 공동으로 학습합니다.

- **Performance Highlights**: 우리의 방법은 캐릭터의 원활한 교체나 삽입을 가능하게 함으로써 단일 프레임워크 내에서 원하는 포즈 동작과 참조 외관을 유지하는 데 효과적임을 보여주는 실험 결과를 통해 입증되었습니다.



### OpenKD: Opening Prompt Diversity for Zero- and Few-shot Keypoint Detection (https://arxiv.org/abs/2409.19899)
Comments:
          Accepted by ECCV 2024

- **What's New**: 본 논문은 멀티모달 프로토타입 세트를 활용하여 시각 및 텍스트 프롬프트 모두를 지원하는 새로운 OpenKD 모델을 제안합니다. 특히, 이전의 프롬프트 다양성을 극복하고 다채로운 텍스트 프롬프트와 보지 못한 텍스트에 효과적으로 대응할 수 있는 방법을 제공합니다.

- **Technical Details**: OpenKD 모델은 다음 세 가지 측면에서 프롬프트의 다양성을 개방합니다: 모달리티(모달리티), 의미론(실제 vs. 보지 못한), 언어 및 텍스트로, 보다 일반화된 제로-샷 및 몇-샷 키포인트 탐지를 가능하게 합니다. 이 모델은 LLM(large language model)을 활용하여 멀티모달 프롬프트와 새로운 키포인트 탐지 능력을 향상시킵니다.

- **Performance Highlights**: OpenKD는 Z-FSKD(Zero-shot 및 Few-shot Keypoint Detection)에서 최신 성능을 달성하며, 새로운 언어 다양성을 열고 보지 못한 텍스트에 대한 접근성을 증가시킵니다. 실험 결과, 96% 이상의 정확도로 키포인트를 텍스트에서 분석하는 능력을 갖췄습니다.



### Universal Medical Image Representation Learning with Compositional Decoders (https://arxiv.org/abs/2409.19890)
- **What's New**: 이번 연구에서는 의료 이미징(medical imaging) 분야에서의 한계를 극복하기 위해 새로운 범용 모델인 UniMed를 개발했습니다. 이 모델은 모든 수준의 작업을 지원하며, 효과적인 사전 훈련을 위한 전략을 제안합니다.

- **Technical Details**: UniMed는 분해된 디코더(decomposed decoder)와 종합적인 디코더(composed decoder)를 도입하여 두 가지 출력 유형인 픽셀(pixel)과 의미(semantic)를 예측할 수 있도록 설계되었습니다. 추가적으로, 입력과 출력 공간을 통합하고 다양한 수준의 작업 주석(task annotations)을 이산 토큰(discrete token) 형식으로 표준화합니다.

- **Performance Highlights**: UniMed는 8개의 데이터셋에서 모든 세 가지 작업(Task)에서 최신 성과(state-of-the-art performance)를 달성하였고, 매우 강력한 제로샷(zero-shot) 및 100샷(100-shot) 전이 가능성을 보여주었습니다.



### Towards Unified Multimodal Editing with Enhanced Knowledge Collaboration (https://arxiv.org/abs/2409.19872)
Comments:
          Accepted by NeurIPS 2024 (Spotlight)

- **What's New**: 이 논문은 Multimodal LLMs (MLLMs)에서의 효과적인 지식 편집을 위한 새로운 방법인 UniKE를 제안합니다. UniKE는 내재적 지식 편집과 외부 지식 이용 방식을 통합한 일관된 프레임워크를 제공합니다.

- **Technical Details**: 내재적 및 외부 지식을 벡터화된 키-값 메모리로 표현하고, 이를 Transformer 레이어에서 동시에 작업합니다. 이 프레임워크에서는 내재적 지식이 진실성과 의미 공간으로 분리되어 결합되고, 대조 학습을 통해 각 지식의 상호작용을 강화합니다.

- **Performance Highlights**: 광범위한 실험을 통해 UniKE가 포스트 편집 MLLM에서 뛰어난 신뢰성, 일반성 및 지역성을 유지하며, 세 가지 속성을 모두 충족하는 데 성공함을 입증했습니다.



### TokenBinder: Text-Video Retrieval with One-to-Many Alignment Paradigm (https://arxiv.org/abs/2409.19865)
- **What's New**: 본 논문에서는 기존 TVR(Text-Video Retrieval) 방법의 단점을 개선하기 위한 새로운 프레임워크인 TokenBinder를 제안합니다. 이 방법은 기존의 일대일 매칭에서 벗어나, 일대다(coarse-to-fine) 정렬을 통해 여러 후보 영상과의 비교를 향상시킵니다.

- **Technical Details**: TokenBinder는 인지 과학에서의 비교 판단(Comparative Judgement) 개념에 영감을 받아, Focused-view Fusion Network와 정교한 cross-attention 메커니즘을 사용하여 여러 영상 간의 특징을 동적으로 정렬하고 비교합니다. 이를 통해 더욱 미세한 뉘앙스와 맥락적 변화를 포착합니다.

- **Performance Highlights**: 여섯 개의 벤치마크 데이터셋에서 대규모 실험을 통해, TokenBinder는 기존의 최첨단 방법들보다 상당한 성능 향상을 보여줍니다. 이러한 결과는 TVR 작업에서 intra-와 inter-modality 정보 간의 간극을 효과적으로 연결하는 데 있어 TokenBinder의 강인함과 정교한 정렬의 효과를 입증합니다.



### SATA: Spatial Autocorrelation Token Analysis for Enhancing the Robustness of Vision Transformers (https://arxiv.org/abs/2409.19850)
- **What's New**: 최근 몇 년 간, vision transformers (ViTs)의 성능을 향상시키기 위해 다양한 방법이 연구되었으나 한계가 있었습니다. 본 논문에서는 Spatial Autocorrelation Token Analysis (SATA)라는 새로운 접근 방식을 소개합니다.

- **Technical Details**: SATA는 token의 특징 간의 공간적 관계를 활용하여 ViT 모델의 표현 능력과 강인성을 향상시킵니다. 이는 self-attention 메커니즘의 Feed-Forward Network (FFN) 블록에 입력되기 전, token의 공간적 자기상관(spatial autocorrelation) 점수에 따라 분석하고 그룹화하는 것을 통해 이루어집니다. 매우 중요한 점은 SATA가 기존의 사전 학습된 ViT 모델에 재학습이나 추가 조정 없이 통합된다는 것입니다.

- **Performance Highlights**: 실험 결과 SATA로 강화된 ViTs는 ImageNet-1K 이미지 분류에서 94.9%의 최상위(top-1) 정확도를 기록하며, ImageNet-A(63.6%), ImageNet-R(79.2%), ImageNet-C(13.6%) 등의 여러 강인성 벤치마크에서도 새로운 최첨단 성능을 달성했습니다. 이 모든 것은 기초 모델에 대한 추가 교육이나 조정 없이 이루어졌습니다.



### Towards Open-Vocabulary Semantic Segmentation Without Semantic Labels (https://arxiv.org/abs/2409.19846)
Comments:
          To appear at NeurIPS 2024. Project page is available at this https URL

- **What's New**: PixelCLIP은 CLIP 이미지 인코더를 픽셀 수준(Pixel-level) 이해에 적응시키기 위한 새로운 방법을 제안합니다.

- **Technical Details**: 이 방법은 SAM 및 DINO와 같은 비전 기초 모델(Vision Foundation Models)에서 생성된 무표시 이미지(unlabeled images)와 마스크(masks)를 사용하여 모델이 객체의 위치를 이해하도록 유도합니다. 또한, 의미론적 레이블(semantic labels) 없이 마스크를 활용하는 문제를 해결하기 위해 학습 가능한 클래스 이름(learnable class names)을 사용하는 온라인 클러스터링 알고리즘(online clustering algorithm)을 개발했습니다.

- **Performance Highlights**: PixelCLIP은 CLIP에 비해 상당한 성능 향상을 보여주며, 오픈-어휘( open-vocabulary) 의미론적 분할(semantic segmentation)에서 캡션 감독 방법(caption-supervised methods)과 경쟁력 있는 결과를 나타냅니다.



### Textual Training for the Hassle-Free Removal of Unwanted Visual Data (https://arxiv.org/abs/2409.19840)
Comments:
          NeurIPS 2024

- **What's New**: 이 연구에서는 시각 데이터셋에서 원치 않는 콘텐츠를 감지하는 방법을 탐구했습니다. 특히, 시각적 데이터를 성공적으로 구분할 수 있는 모델이 오직 텍스트 데이터만으로 생성될 수 있음을 입증하는 이론적 분석을 제공합니다.

- **Technical Details**: Hassle-Free Textual Training (HFTT)라는 단순화된 방법을 제안합니다. HFTT는 합성 텍스트 데이터만을 사용하여 원치 않는 시각 콘텐츠 감지기를 획득할 수 있는 방법으로, 사전 훈련된 비전-언어 모델과 결합되어 작동합니다. 이 방법은 인간 개입의 필요성을 크게 줄이는 혁신적인 목적 함수(objective function)를 특징으로 하며, 텍스트 데이터의 합성(synthesis) 방법을 사용하여未知 visual data distribution을 훈련 과정에 통합하는 능력을 갖추고 있습니다.

- **Performance Highlights**: HFTT의 독특한 특성은 전통적인 out-of-distribution(분포 외) 감지를 넘어 더 추상적인 개념을 다루는 작업에도 적용 가능하게 확장됩니다. 이론적 분석과 함께 out-of-distribution 감지 및 증오 이미지 감지 실험을 통해 성능을 검증했습니다.



### GrokLST: Towards High-Resolution Benchmark and Toolkit for Land Surface Temperature Downscaling (https://arxiv.org/abs/2409.19835)
- **What's New**: 본 연구에서는 Land Surface Temperature (LST) 예측 문제를 해결하기 위해 새로운 Modality-Conditional Large Selective Kernel (MoCoLSK) Networks를 제안합니다. 이 네트워크는 다중 모달 데이터의 동적 융합을 통해 높은 해상도의 LST 데이터를 생성하는 데 초점을 맞추고 있습니다.

- **Technical Details**: MoCoLSK 네트워크는 이전의 LSKNet 구조를 재설계하여 다이나믹 수용 필드(adaptive receptive field) 조정과 다중 모달(feature integration) 통합을 구현합니다. 이 시스템은 공간 비정상성(spatial non-stationarity)을 고려하여 보다 정밀한 예측을 가능하게 합니다.

- **Performance Highlights**: 실험 결과에 따르면 MoCoLSK는 복잡한 의존성과 미세한 변화를 효과적으로 포착하여 LST downscaling 문제에서 기존 방법들보다 더 뛰어난 성능을 발휘합니다. GrokLST 프로젝트를 통해 40개 이상의 최신 방법이 통합된 PyTorch 기반의 오픈소스 도구 및 데이터셋이 제공됩니다.



### HazyDet: Open-source Benchmark for Drone-view Object Detection with Depth-cues in Hazy Scenes (https://arxiv.org/abs/2409.19833)
- **What's New**: 드론 기반 물체 감지(Drone-based Object Detection)가 악천후에서 문제 해결을 위한 새로운 데이터셋 HazyDet를 소개합니다. 이 데이터셋은 383,000개의 실제 사례를 포함하고 있습니다.

- **Technical Details**: HazyDet는 자연적으로 흐릿한 환경과 인위적으로 흐림 효과를 추가한 일반적인 장면에서 수집된 데이터로 구성되어 있습니다. 이를 통해 Depth Conditioned Detector (DeCoDet)를 설계하여 깊이(depth)와 흐림(haze) 조건을 반영하였습니다. DeCoDet는 Multi-scale Depth-aware Detection Head 및 동적 Depth Condition Kernel 모듈을 포함하여 깊이 정보를 통합합니다.

- **Performance Highlights**: HazyDet 데이터셋에서의 광범위한 평가 결과, 제안된 방법의 유연성과 효과성이 입증되었으며, 성능이 상당히 향상되었습니다. 데이터셋과 도구는 제공된 URL에서 확인할 수 있습니다.



### GameLabel-10K: Collecting Image Preference Data Through Mobile Game Crowdsourcing (https://arxiv.org/abs/2409.19830)
Comments:
          6 pages, 6 images

- **What's New**: 이번 연구는 멀티-억 하라미터 모델의 발전에 따른 데이터 수요 증가에 대응하기 위해 유료 주석자(annotators)를 비디오 게임 플레이어로 대체할 가능성을 탐구합니다. 게임 성과에 따라 인게임 통화로 보상을 제공하는 방안입니다.

- **Technical Details**: 이 연구에서는 모바일 역사 전략 게임 'Armchair Commander'의 개발자들과 협력하여, 이미지 쌍 비교(pairwise image preference) 데이터를 활용해 확산 모델(diffusion models)을 미세 조정하는 데 사용된 데이터셋인 GameLabel-10K를 생성했습니다. 이 데이터셋은 약 10,000개의 라벨과 7,000개의 고유 프롬프트로 구성되어 있습니다.

- **Performance Highlights**: 이 연구 결과는 GameLabel-10K의 데이터셋을 공개적으로 오픈 소스 라이선스 하에 배포하며, 이 데이터셋의 한계에 대한 분석을 포함합니다.



### Tracking Everything in Robotic-Assisted Surgery (https://arxiv.org/abs/2409.19821)
Comments:
          7 pages

- **What's New**: 본 논문에서는 Robotic-Assisted Minimally Invasive Surgery (RAMIS)에서의 조직(tissues) 및 도구(instruments) 추적을 향상시키기 위해 새로운 주석이 달린 외과 추적 데이터셋을 소개합니다. 이 데이터셋은 복잡한 조직 및 도구의 동작을 포함한 실제 외과 비디오로 구성되어 있습니다.

- **Technical Details**: 기존의 keypoint-based sparse tracking은 특징 점(featured points)에 의해 제한되고, flow-based dense two-view matching은 장기적인 드리프트(long-term drifts)로 인해 문제가 발생합니다. Tracking Any Point (TAP) 알고리즘이 이러한 제한을 극복하기 위해 제안되었으나, 외과 상황에서의 효용성은 검증되지 않았습니다. 본 연구에서는 TAP 기반 알고리즘의 성능을 평가하고, SurgMotion이라는 새로운 추적 방법론을 통해 이 문제를 해결하고 추적 성능을 개선합니다.

- **Performance Highlights**: 제안된 SurgMotion 방법은 외과 도구 추적 시 대부분의 TAP 기반 알고리즘을 초과하는 성능을 보이며, 특히 도전적인 의료 비디오에서 기초선(baselines) 대비 유의미한 개선을 나타냅니다.



### Robust Incremental Structure-from-Motion with Hybrid Features (https://arxiv.org/abs/2409.19811)
Comments:
          40 pages, 16 figures, 9 tables. To appear in ECCV 2024

- **What's New**: 이번 논문에서는 Structure-from-Motion (SfM) 문제를 해결하기 위해 선(segment) 정보를 포함한 새로운 오픈소스 소프트웨어를 소개합니다. 이는 무한정한 텍스처 또는 잘 구성되지 않은 장면에서도 SfM의 안정성과 정확성을 크게 향상시킵니다.

- **Technical Details**: 이 시스템은 점, 선, 소실점(vanishing points) 및 그들의 구조적 관계를 통합하여 점진적 지도 작성(mapping), 삼각측량(triangulation) 및 등록(registration) 단계를 개선합니다. 새로운 불확실성 모델링 방법(unreliable tracks 정의 및 이중 정제 두 단계 적용)도 도입하여 초기 단계에서 불확실한 선 추적을 조기에 필터링하지 않고 정확도를 유지합니다.

- **Performance Highlights**: 이 시스템은 특히 도전적인 환경에서 널리 사용되는 COLMAP 시스템보다 더 높은 정확도와 강인성을 자랑하며, 더 정밀한 카메라 위치 추정과 더욱 풍부한 희소 맵을 생성합니다. 또한 불확실성을 고려한 위치 추정 모듈은 포인트 기반 및 하이브리드 세팅에서도 일관된 성능 향상을 보여줍니다.



### Crafting Distribution Shifts for Validation and Training in Single Source Domain Generalization (https://arxiv.org/abs/2409.19774)
Comments:
          WACV 2025

- **What's New**: 본 연구에서는 단일 소스 도메인 일반화(single-source domain generalization) 문제를 해결하기 위한 새로운 검증(validation) 방법론을 제안합니다. 이 방법론은 소스 도메인 이미지에 다양한 augmentations를 적용하여 독립적인 검증 세트를 구성합니다.

- **Technical Details**: 제안된 방법은 k-fold validation 과정을 통해 학습(training)과 검증 과정에서 사용되는 augmentations의 유형을 분리합니다. 이는 모델의 일반화(generalization) 능력을 정확히 평가하고, 최적의 성능을 내는 방법을 선택하는 데 도움을 줍니다.

- **Performance Highlights**: 제안된 검증 방법은 표준 검증 방식에 비해 15.4% 또는 1.6%의 상대적 정확도 향상을 보여줍니다. 다양한 데이터셋과 여러 방법에 대해 검증 성능과 테스트 성능 간 높은 상관관계를 입증하였습니다.



### PPLNs: Parametric Piecewise Linear Networks for Event-Based Temporal Modeling and Beyond (https://arxiv.org/abs/2409.19772)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 본 논문에서는 PPLN(Parametric Piecewise Linear Networks)을 통해 시간 기반 비전 추론을 위한 새로운 접근 방식을 제시합니다. PPLN은 생물학적 신경 행동을 규명하는 신경형 시스템 원리에 기반하여 개발되었습니다.

- **Technical Details**: PPLN은 인공 뉴런의 막 전위(membrane potential)를 학습 가능한 계수를 가진 파라메트릭 피스와이즈 리니어 함수(parametric piecewise linear function)로 표현합니다. 이러한 디자인은 최근 Kolmogorov-Arnold Networks (KANs)에서 대중화된 학습 가능한 파라메트릭 함수로부터 딥 모델을 구축하는 아이디어를 반영합니다.

- **Performance Highlights**: PPLN은 이벤트 기반 및 이미지 기반 비전 응용 프로그램에서 최첨단 성능을 보여주며, 애플리케이션으로는 조향 예측(steering prediction), 인간 포즈 추정(human pose estimation), 움직임 디블러링(motion deblurring)이 포함됩니다.



### Offline Signature Verification Based on Feature Disentangling Aided Variational Autoencoder (https://arxiv.org/abs/2409.19754)
- **What's New**: 이 논문에서는 기존의 서명 검증 시스템의 한계를 극복하기 위해 새로운 서명 검증 방법을 제안합니다. 이 방법은 변분 오토인코더(Variational Autoencoder, VAE)를 사용하여 서명 이미지에서 직접 특징을 추출하는 첫 번째 모델입니다.

- **Technical Details**: 제안된 방법은 개선된 VAE를 사용하여 서명 이미지로부터 특징을 추출하고, SVM(Support Vector Machine)을 통해 분류를 수행합니다. 새로운 손실 함수(loss function)를 도입하여 특징의 분리(feature disentangling)를 용이하게 하고, 기존 VAE의 성능을 개선합니다.

- **Performance Highlights**: 제안된 방법은 MCYT-75 및 GPDS-합성 데이터셋에서 두 번의 광범위한 실험을 통해 13개의 대표적인 오프라인 서명 검증 방법을 크게 초월하는 성능을 보였습니다. 이를 통해 실용적인 응용에서 시스템의 튼튼함과 잠재력을 입증했습니다.



### T2Vs Meet VLMs: A Scalable Multimodal Dataset for Visual Harmfulness Recognition (https://arxiv.org/abs/2409.19734)
- **What's New**: 본 논문에서는 10,000개의 이미지와 1,000개의 비디오로 구성된 포괄적인 해로운 컨텐츠 데이터셋인 Visual Harmful Dataset 11K (VHD11K)를 제안합니다. 이는 기존 데이터셋의 한계를 극복하고 다양한 해로운 개념을 다룹니다.

- **Technical Details**: VHD11K는 10개 범주로 나뉘며, 인터넷과 4개의 생성 모델에서 수집된 이미지 및 비디오 샘플로 구성됩니다. 주목할 점은 다중 에이전트 Visual Question Answering (VQA)를 통한 새로운 주석 프레임워크를 사용하여 생성된 해로운 컨텐츠의 맥락을 고려한 주석이 이루어집니다. 이를 통해 3개의 VLMs (Vision-Language Models)가 '판사', '찬성 토론자', '반대 토론자' 역할을 맡아 해로운지 여부를 논의하게 됩니다.

- **Performance Highlights**: VHD11K는 기존 해로운 컨텐츠 인식 방법과 비교하여 높은 성능 향상을 보여주었으며, 기존 데이터셋에 비해 해로운 컨텐츠를 더 잘 인식하는 것으로 나타났습니다. 또한, 인 annotation 프레임워크와 인공지능 모델 간의 alignment가 뛰어나며, 향후 해로운 컨텐츠 조절에 대한 기준이 될 수 있는 실험 결과가 도출되었습니다.



### Pear: Pruning and Sharing Adapters in Visual Parameter-Efficient Fine-Tuning (https://arxiv.org/abs/2409.19733)
- **What's New**: 본 논문에서는 Prune and Share (Pear)라는 새로운 어댑터 프루닝(pruning) 방법론을 제안하여 사전 학습된 비주얼 파운데이션 모델의 효율적인 세부 조정을 가능하게 합니다.

- **Technical Details**: Pear 방법론은 redundant 한 어댑터를 제거하고 더 중요한 어댑터를 공유하는 방식을 채택하여 모든 위치에서 계속적인 적응을 지원하며, Knowledge Checkpoint 전략을 도입하여 제거된 어댑터의 정보를 보존합니다.

- **Performance Highlights**: VTAB-1K 벤치마크에서 Pear의 효율성과 성능을 검증하여 기존의 경쟁적인 방법들에 비해 우수한 결과를 입증하였습니다.



### FAST: A Dual-tier Few-Shot Learning Paradigm for Whole Slide Image Classification (https://arxiv.org/abs/2409.19720)
Comments:
          Accepted to NeurIPS 2024

- **What's New**: 이 논문에서는 Whole Slide Images (WSI) 분류를 위한 새로운 이중 계층 Few-shot learning 패러다임인 FAST를 제안합니다. 기존의 고비용 세분화 주석을 피하기 위해 슬라이드 수준에서 아주 소수의 WSIs를 수집하고, 각각의 WSI 내에서 소량의 패치를 라벨링하는 효과적인 주석 전략을 도입하였습니다.

- **Technical Details**: FAST는 이중 레벨 주석 전략과 이중 가지 분류 프레임워크로 구성됩니다. 캐시 브랜치(cache branch)에서는 임의의 패치에 대한 라벨을 학습하고, 사전 브랜치(prior branch)에서는 비주얼-언어 모델의 텍스트 인코더를 활용하여 패치 분류를 수행합니다.

- **Performance Highlights**: 광범위한 실험을 통해 FAST 방법이 CAMELYON16 및 TCGA-RENAL 데이터셋에서 기존의 Few-shot 분류 방법들을 초월하고, 완전 감독 학습 방법과 유사한 정확도를 보이며 주석 비용은 단지 0.22%에 불과함을 증명하였습니다.



### Applying the Lower-Biased Teacher Model in Semi-Suepervised Object Detection (https://arxiv.org/abs/2409.19703)
Comments:
          12pages,2 figures,2 tables, several fomulas

- **What's New**: 이번 논문에서는 Semi-supervised object detection(세미-슈퍼바이즈드 객체 탐지) 작업을 위한 Lower Biased Teacher 모델을 제안합니다. 이 모델은 기존의 Unbiased Teacher 모델을 개선하여, teacher model(티처 모델) 내에 localization loss(로컬라이제이션 손실)을 통합한 것이 특징입니다.

- **Technical Details**: Lower Biased Teacher 모델은 데이터셋 내의 class imbalance(클래스 불균형) 문제와 bounding boxes(바운딩 박스)의 정확성 문제를 해결하여 pseudo-label generation(슈도 레이블 생성)의 정확성을 크게 향상시킵니다. 이 모델은 다양한 세미-슈퍼바이즈드 객체 탐지 데이터셋에서 실험을 수행하였으며, 클래스 불균형으로 인한 슈도 레이블링 바이어스를 줄이고 잘못된 바운딩 박스로 인한 오류를 완화합니다.

- **Performance Highlights**: Lower Biased Teacher 모델은 mAP(Mean Average Precision) 점수를 높이고, 기존 방법들과 비교하여 더 신뢰할 수 있는 탐지 결과를 보여줍니다. 이 연구는 정확한 슈도 레이블 생성의 중요성을 강조하며, 객체 탐지를 위한 세미-슈퍼바이즈드 학습의 향후 발전을 위한 견고한 프레임워크를 제공합니다.



### RNG: Relightable Neural Gaussians (https://arxiv.org/abs/2409.19702)
- **What's New**: 본 논문에서는 Relightable Neural Gaussians(RNG)를 제안하여, 잘 정의된 표면과 흐물흐물한 형태를 가진 객체의 리라이트(fresh lighting)을 동시에 수행할 수 있는 최신 프레임워크를 개발했습니다.

- **Technical Details**: RNG는 독립적으로 조명 방향을 조건화하여 각각의 Gaussian에서 색상을 생성하는 방법으로, 기존의 표면 제약 조건이나 분석적 음영 모델에 의존하지 않고도 리라이트가능한 방사선(radiance) 표현을 모델링합니다. 이 모든 과정에서 MLP를 통해 특성 벡터를 색상으로 디코드할 수 있습니다. 또한, 점 조명(Point Light)을 활용하여 애매함을 줄이고, 그림자 인식 조건을 네트워크에 도입했습니다.

- **Performance Highlights**: RNG는 기존의 신경 방사장 필드(NeRF) 기반 연구보다 약 20배 더 빠른 학습 속도와 600배 더 빠른 렌더링 속도를 달성하며, RTX 4090에서 초당 60프레임(frames per second)을 지원합니다.



### Neural-Polyptych: Content Controllable Painting Recreation for Diverse Genres (https://arxiv.org/abs/2409.19690)
- **What's New**: 본 연구에서는 Neural-Polyptych라는 통합 프레임워크를 제안하여 아마추어와 전문가 간의 장벽을 허물고, 사용자가 손으로 그린 스케치와 원본 그림의 요소를 결합하여 고해상도 회화 작품을 창작할 수 있도록 도와줍니다.

- **Technical Details**: GAN 기반의 다중 스케일 아키텍처를 설계하여 생성 과정에서 전역 기능과 지역 기능을 구분합니다. 또한, Correspondence Attention 모듈을 통해 사용자 스케치에서 생성된 의미적 세부사항의 충실도를 향상시킵니다. 이 접근법은 사용자가 예술적으로 조화로운 대형 그림을 생성할 수 있는 능력을 제공합니다.

- **Performance Highlights**: Neural-Polyptych 접근법은 다양한 동양 및 서양 화풍에 대한 검증을 진행하였으며, 대형 회화 확장, 텍스처 셔플링, 장르 전환, 벽화 복원 및 재구성을 위한 성공적인 응용 프로그램을 지원합니다.



### Text-driven Human Motion Generation with Motion Masked Diffusion Mod (https://arxiv.org/abs/2409.19686)
- **What's New**: 본 논문에서는 Motion Masked Diffusion Model (MMDM)을 제안하여, 자연어에 기반한 인간 모션 생성을 향상시킵니다. 이 모델은 spatio-temporal 관계 학습을 위한 마스킹 메커니즘을 사용하며, 두 가지 마스킹 전략인 time frames mask와 body parts mask를 도입하여 시간적 특성과 공간적 구조를 고려합니다.

- **Technical Details**: MMDM은 motion embedding 공간에서 특정 토큰을 마스킹하고, 이를 통해 확률적 노이즈 제거 과정을 통해 전체 모션 시퀀스를 학습합니다. 기존의 diffusion 모델은 temporal 및 spatial semantics 간의 관계를 잘 이해하지 못했으나, MMDM은 마스킹 전략을 통해 이 문제를 해결합니다.

- **Performance Highlights**: 실험 결과, HumanML3D 및 KIT-ML 데이터셋에서 MMDM의 마스킹 전략이 모션의 품질과 텍스트-모션 일관성을 균형 있게 유지하며 효과적인 것으로 나타났습니다.



### Underwater Organism Color Enhancement via Color Code Decomposition, Adaptation and Interpolation (https://arxiv.org/abs/2409.19685)
- **What's New**: 이 연구에서는 미세 조정 가능한 색상 출력을 제공하는 새로운 수중 이미지 향상 방법인 ColorCode를 제안합니다. 기존의 수중 이미지 향상 알고리즘들이 단일 색상 이미지를 생성하는 반면 ColorCode는 사용자에게 다양한 선택지를 제공합니다.

- **Technical Details**: ColorCode는 수중 이미지를 지도 학습(supervised training)을 통해 향상된 참조 이미지로 복원한 후, 자가 재구성(self-reconstruction) 및 교차 재구성(cross-reconstruction)을 통해 색상 코드(color code) 및 콘텐츠 코드(content code)로 분해합니다. 색상 코드는 Gaussian 분포를 따르도록 명시적으로 제약되어 추론(inference)할 때 효율적인 샘플링(sampling)과 보간(interpolation)이 가능합니다.

- **Performance Highlights**: ColorCode는 세 가지 주요 기능을 제공합니다: 색상 향상(color enhancement), 색상 적응(color adaptation), 색상 보간(color interpolation). 이러한 기능은 다양한 수중 이미지의 색상 품질을 효과적으로 개선하며, 양적 및 시각적 평가 결과에서 기존 방법보다 우수한 성능을 보여줍니다.



### MedViLaM: A multimodal large language model with advanced generalizability and explainability for medical data understanding and generation (https://arxiv.org/abs/2409.19684)
- **What's New**: 이 논문에서는 MedViLaM이라는 통합 비전-언어 모델을 소개하며, 이는 다양한 형태의 의료 데이터를 효율적으로 인코딩하고 해석할 수 있는 일반 전문가 모델로 발전하기 위한 것입니다. Multimodal (다중 모달) 데이터 처리 및 여러 작업을 동시에 수행할 수 있는 기능을 가지고 있습니다.

- **Technical Details**: MedViLaM은 임상 언어 및 이미지를 포함한 다양한 의료 데이터 형식을 모두 동일한 모델 가중치를 사용하여 처리할 수 있는 모델입니다. 이를 지원하기 위해 MultiMedBench라는 포괄적인 프리트레인(pretraining) 데이터셋과 벤치마크를 구성하였습니다. 모델은 20.5M의 다중 작업 지침 쌍과 1.8M의 의료 이미지를 요구하며, 비주얼 질문 응답, 질병 분류, 질병 위치 확인 및 보고서 생성을 포함한 커스터마이즈 지침을 지원합니다.

- **Performance Highlights**: MedViLaM은 MultiMedBench의 모든 작업에서 강력한 성능을 발휘하며, 다른 일반 모델들을 상당한 차이로 초월했습니다. Radiologist(영상의학전문의) 평가에서는 위 모델의 결과가 기존 방사선과 의사가 생성한 결과보다 80.50%의 경우에서 선호되었습니다. 이 모델은 다양한 실세계 임상 작업에 적용되는 대규모 기초 모델의 일반화 성능을 검증하는 새로운 벤치마크를 제시하였습니다.



### Simple and Fast Distillation of Diffusion Models (https://arxiv.org/abs/2409.19681)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 본 연구에서는 기존의 방법보다 크게 단순화된 Simple and Fast Distillation (SFD) 방법을 제안하며, 이를 통해 미세 조정 시간을 최대 1000배 단축시킬 수 있음을 보입니다. SFD는 기존의 디퓨전 모델의 구조를 유지하면서도 학습 효율성을 크게 향상시킵니다.

- **Technical Details**: SFD는 기본적인 디퓨전 모델에서 시작하여 샘플링을 위한 특정 타임스탬프들만 미세 조정하여 성능을 극대화합니다. 주요 요소들로는 데이터 분포와 노이즈 분포 간의 일관된 변환 및 학생 모델이 교사 모델의 샘플링 경로를 모방하도록 하는 방법이 포함됩니다. SFD-v라는 변형을 통해 단일 모델에서 다양한 샘플링 스텝을 가능하게 합니다.

- **Performance Highlights**: SFD는 CIFAR-10 데이터셋에서 2 NFE로 FID 4.53을 달성하였으며, 단일 NVIDIA A100 GPU에서 고작 0.64시간의 미세 조정으로 이 결과를 얻었습니다. 이는 기존의 일관성 기반 디스틸레이션과 비교해 1000배 빠른 결과입니다.



### SemiDDM-Weather: A Semi-supervised Learning Framework for All-in-one Adverse Weather Remova (https://arxiv.org/abs/2409.19679)
- **What's New**: 본 논문에서는 제한된 레이블 데이터로도 여러 가지 악천후 제거를 효과적으로 수행할 수 있는 최초의 반지도 학습 프레임워크인 SemiDDM-Weather를 제시합니다. 이 프레임워크는 teacher-student 네트워크에 기반하고 있으며, Denoising Diffusion Model(DDM)을 백본으로 활용합니다.

- **Technical Details**: SemiDDM-Weather는 SOTA Wavelet Diffusion Model-Wavediff를 수정하여 백본으로 사용하고, 교사 네트워크에서 생성된 '최적'의 출력 결과를 사용하여 학생 네트워크의 훈련을 안내합니다. 이 프레임워크는 품질 평가 및 내용 일관성 제약을 도입하여 고품질의 이미지 복원을 목표로 하며, 일반적인 Denoising Diffusion Model보다 더 적은 샘플링 단계(단 4단계)로 작동하여 효율성을 높입니다.

- **Performance Highlights**: 실험 결과, SemiDDM-Weather는 합성 데이터와 실제 데이터셋 모두에서 높은 시각적 품질과 우수한 악천후 제거 성능을 보였으며, 특히 완전 지도 학습 방식의 경쟁자들과 비교했을 때도 각광받는 성능을 나타냈습니다.



### See Detail Say Clear: Towards Brain CT Report Generation via Pathological Clue-driven Representation Learning (https://arxiv.org/abs/2409.19676)
Comments:
          Our work has been accepted by EMNLP2024 findings

- **What's New**: 이 연구에서는 Pathological Clue-driven Representation Learning (PCRL) 모델을 도입하여 병리적 단서에 기반한 교차 모달 표현을 구축하고 이를 정확한 CT 보고서 생성에 자연스럽게 적합시키는 방법을 제안합니다.

- **Technical Details**: PCRL 모델은 세분화된 영역, 병리적 개념, 보고서 주제를 관점으로 하여 병리적 단서를 추출하고, 이들로부터 시각 병리적 패턴을 이해하고 교차 모달 특성 표현을 배우도록 합니다. 학습된 표현을 보고서 생성 작업에 적합하게 하기 위해, 작업 맞춤형 지침을 가진 통합 대형 언어 모델(LLM)을 사용하여 표현 학습과 보고서 생성을 연결합니다.

- **Performance Highlights**: 실험 결과, 우리의 방법은 이전 방법들보다 우수한 성능을 보였으며, 특히 뇌 CT 보고서 생성에서 SoTA 성능을 달성했습니다.



### All-in-One Image Coding for Joint Human-Machine Vision with Multi-Path Aggregation (https://arxiv.org/abs/2409.19660)
Comments:
          NeurIPS 2024

- **What's New**: 본 논문에서는 인간 인식(human perception)과 기계 비전(machine vision)을 위한 다중 작업(multi-task) 이미지 코딩의 혁신적인 방법을 제안합니다. Multi-Path Aggregation (MPA)라는 새로운 접근법을 통해 기존의 코딩 모델에 통합하게 되며, 이를 통해 단일 구조에서 다양한 작업의 특성을 통합한 최적화된 기능 표현을 제공할 수 있습니다.

- **Technical Details**: MPA는 다층 퍼셉트론(Multi-Layer Perceptron, MLP) 브랜치를 사용하여 다양한 작업에 맞는 여러 집약 경로를 생성하며, 작업별로 분류된 중요도에 기반하여 잠재적 특징(latent features)을 할당하는 예측기(predictor)를 활용합니다. 이를 통해 다양한 작업 간에 특징의 상관 관계를 최대한 활용하고 두 단계 최적화 전략을 개발하여 다중 작업 성능 저하를 완화합니다. 이러한 접근 방식은 전체 모델을 크게 최적화하지 않고 특정 작업에 대한 매개변수를 조정하는 것을 가능하게 합니다.

- **Performance Highlights**: 실험 결과 MPA는 인간 비전 작업 및 기계 분석 작업에서 최첨단(state-of-the-art, SOTA) 방법과 동등한 성능을 달성하였으며, 여러 작업을 수행하는 데 있어 모델에 대한 수정을 최소화하면서도 효율성을 증대시킵니다. 특히 MPA는 단일 모델 내에서 인간과 기계 지향 재구성을 원활하게 전환할 수 있는 기능을 제공하며, 작업 제어 해석(task-controllable interpretation)을 가능하게 합니다.



### Flipped Classroom: Aligning Teacher Attention with Student in Generalized Category Discovery (https://arxiv.org/abs/2409.19659)
- **What's New**: 이 연구에서는 일반화된 범주 발견(Generalized Category Discovery, GCD) 문제에서 전통적인 교사-학생(Teacher-Student) 프레임워크의 한계를 탐구하고, 학생의 주의(attention)와 일치하도록 동적으로 업데이트하는 FlipClass 방법을 제안합니다.

- **Technical Details**: FlipClass는 교사 모델이 학생의 피드백에 따라 주의를 동적으로 조정하는 전략을 사용합니다. 이 방법은 주의 레이어 간 불일치 문제를 해결하고, 과거 클래스와 신규 클래스 간의 일관된 패턴 인식을 촉진합니다.

- **Performance Highlights**: 광범위한 벤치마크에서 실험을 진행한 결과, FlipClass는 최신 GCD 방법들을 상당히 능가하며 해당 분야의 새로운 기준을 수립했습니다.



### Dual-Attention Frequency Fusion at Multi-Scale for Joint Segmentation and Deformable Medical Image Registration (https://arxiv.org/abs/2409.19658)
- **What's New**: 이 논문에서는 변형 의료 이미지 등록(deformable medical image registration)을 위한 새로운 multi-task learning framework인 multi-scale dual attention frequency fusion(DAFF-Net)을 제안합니다. 이 프레임워크는 세그멘테이션(segmentation) 마스크와 밀집 변형 필드(dense deformation fields)를 동시에 한 단계에서 추정합니다.

- **Technical Details**: DAFF-Net은 글로벌 인코더(global encoder), 세그멘테이션 디코더(segmentation decoder), 그리고 coarse-to-fine 피라미드 등록 디코더(coarse-to-fine pyramid registration decoder)로 구성됩니다. 특히 등록 디코딩 과정에서 dual attention frequency feature fusion(DAFF) 모듈을 통해 서로 다른 스케일에서 등록 및 세그멘테이션 기능을 융합(fuse)하여 두 작업 간의 상관관계를 완전히 활용합니다.

- **Performance Highlights**: 제안된 DAFF-Net과 그 비지도 변형은 3개의 공개 3D 뇌 자기공명영상(MRI) 데이터셋에서 여러 평가 지표에 걸쳐 최신 등록 방법들보다 우수한 성능을 보였습니다. 이는 변형 의료 이미지 등록에서 제안된 접근 방식의 효과성을 입증합니다.



### Grounding 3D Scene Affordance From Egocentric Interactions (https://arxiv.org/abs/2409.19650)
- **What's New**: 이 논문은 3D 장면의 affordance(가능성) 이해를 위해 egocentric(자기 중심) 비디오에서 상호작용을 통한 새로운 과제를 제안합니다. 이를 통해 기존의 정적 기하학적 구조와 시각적 외관에 의존하는 방법의 한계를 극복하고, 보다 능동적으로 환경을 인식하고 상호작용할 수 있는 모델을 개발하고자 합니다.

- **Technical Details**: Ego-SAG(예고적 상호작용 기반 3D 장면 affordance 정착) 프레임워크는 Interaction-Guided Spatial Significance Allocation Module(ISA)와 Bilateral Query Decoder Module(BQD)의 두 가지부터 구성됩니다. ISA는 공간 복잡성을 처리하는 데 초점을 맞추며, BQD는 다양한 출처 간 affordance 특징을 정렬하고 상호작용과 관련된 서브-영역에 모델 집중을 유도합니다.

- **Performance Highlights**: VSAD(Vide-3D Scene Affordance Dataset) 데이터셋을 사용한 실험 결과, Ego-SAG는 다른 대표적인 방법들에 비해 월등한 성능을 보였으며, 향후 연구를 위한 강력한 기준선으로 자리 잡을 것입니다.



### OrientedFormer: An End-to-End Transformer-Based Oriented Object Detector in Remote Sensing Images (https://arxiv.org/abs/2409.19648)
Comments:
          The paper is accepted by IEEE Transactions on Geoscience and Remote Sensing (TGRS)

- **What's New**: 이 논문에서는 변환기 기반의 방향성 물체 탐지기를 제안하여 기존의 CNN 기반 방법을 초월하는 성과를 보이는 모델을 제시합니다. 새로운 ‘OrientedFormer’ 아키텍처는 3가지 모듈(Gaussian positional encoding, Wasserstein self-attention, oriented cross-attention)을 활용하여 방향성 물체 탐지에서의 문제를 해결합니다.

- **Technical Details**: 제안하는 ‘OrientedFormer’는 Gaussian positional encoding을 통해 각도, 위치 및 크기를 같은 메트릭으로 통합하며, Wasserstein self-attention을 사용해 콘텐츠 쿼리와 위치 쿼리 간의 상호작용을 촉진합니다. 마지막으로, oriented cross-attention은 포지션 쿼리를 중심으로 샘플링 포인트를 회전시켜 값을 조정하여 불일치를 해결합니다.

- **Performance Highlights**: ‘OrientedFormer’는 DIOR-R 및 DOTA-v1.0 데이터셋에서 각각 1.16과 1.21 AP50 향상을 이뤄내며, 훈련 에폭 수를 3배에서 1배로 줄였습니다. Resnet50 기반에서 DIOR-R에서 67.28% AP50, DOTA-v2.0에서 54.27% AP50을 달성하며 새로운 최첨단 기준을 세웠습니다.



### fCOP: Focal Length Estimation from Category-level Object Priors (https://arxiv.org/abs/2409.19641)
- **What's New**: 이 논문은 단일 이미지를 사용하여 모노큘러(focal length) 초점을 추정하는 새로운 방법을 제안합니다. 기존의 맨해튼 월드 가정이나 인위적인 캘리브레이션 패턴 없이도, 범주 수준의 객체 사전(category-level object priors)을 활용하여 초점을 추정할 수 있습니다.

- **Technical Details**: 모노큘러 깊이 예측(monocular depth estimation) 및 범주 수준의 객체 표준화 표현 학습(category-level object canonical representation learning)을 기반으로 하는 초점 해결기(focal solver)는 이미지에서 객체의 깊이와 모양 정보를 활용하여 초점을 추정합니다. 본 연구에서 제안하는 초점 해결기는 폐쇄형 클로즈드폼(closed form)으로 삼중 항목(triplets) 간의 대응 관계에서 초점을 추정합니다.

- **Performance Highlights**: 모의 실험과 실제 데이터에 대한 실험 결과, 제안된 방법은 기존의 최신 기술(state-of-the-art)보다 우수한 성능을 보여주며, 모노큘러 초점 추정 문제를 해결하기 위한 유망한 솔루션을 제공합니다.



### BadHMP: Backdoor Attack against Human Motion Prediction (https://arxiv.org/abs/2409.19638)
- **What's New**: 본 논문은 BadHMP라는 새로운 백도어 공격 기법을 제안합니다. 이는 인간 동작 예측 모델을 타겟으로 하는 최초의 백도어 공격으로, 특정 관절에 로컬화된 트리거를 심어 poisoined training samples를 생성하여 예측 정확도를 유지하면서도 마치 공격을 받지 않은 것처럼 행동하도록 설계되었습니다.

- **Technical Details**: BadHMP는 두 가지 유형의 백도어 트리거(‘rigid’와 ‘soft’)와 두 가지 대상(‘jammed’와 ‘moving’)을 사용하여 네 가지 독특한 poisoning 전략을 구현합니다. 우리의 접근 방식은 기존의 human motion samples 데이터 형식의 특성을 고려하여, 자연스럽고 매끄러운 poisoned training samples를 생성하는 것을 목표로 합니다. 이를 위해 Clean Data Error (CDE)와 Backdoor Data Error (BDE)라는 새로운 평가 지표를 제안합니다.

- **Performance Highlights**: Human3.6M 및 CMU-Mocap 두 가지 데이터 세트와 LTD 및 HRI 두 가지 네트워크 아키텍처에서 수행된 실험 결과, BadHMP의 높은 정확도와 효과성을 입증하였고, 저비율 poisoined 샘플이 있어도 타겟 시퀀스를 성공적으로 활성화할 수 있음을 보여주었습니다. 또한, 모델의 fine-tuning 방어에 대한 공격의 강인성도 검증되었습니다.



### Storynizor: Consistent Story Generation via Inter-Frame Synchronized and Shuffled ID Injection (https://arxiv.org/abs/2409.19624)
- **What's New**: 최근 텍스트-이미지 생성 모델의 발전으로 지속적인 이야기 이미지 생성에 대한 관심이 커졌습니다. 본 논문에서는 Storynizor라는 모델을 소개하며, 이는 높은 프레임 간 캐릭터 일관성, 효과적인 전경-배경 분리, 그리고 다양한 포즈 변화를 통해 일관된 이야기 생성을 지원합니다.

- **Technical Details**: Storynizor의 핵심 혁신은 ID-Synchronizer와 ID-Injector 두 가지 주요 모듈에 있습니다. ID-Synchronizer는 자동 마스크 자가 주의(auto-mask self-attention) 모듈과 마스크 지각 손실(mask perceptual loss)을 통해 캐릭터 생성을 일관되게 유지하며, ID-Injector는 Shuffle Reference Strategy(SRS)를 활용하여 ID 기능을 특정 위치에 통합합니다. 이러한 접근은 UNet 아키텍처를 기반으로 하여 작동하며, 훈련 과정에서 ID 일관성을 유지하도록 지원합니다.

- **Performance Highlights**: 실험 결과, Storynizor는 다른 캐릭터 특정 방법들에 비해 높은 충실도(character consistency)와 유연한 포즈(flexible postures)를 유지하며, 생생한 배경(vivid backgrounds)으로 일관된 이야기 이미지를 생성할 수 있음을 보여주었습니다.



### Discerning the Chaos: Detecting Adversarial Perturbations while Disentangling Intentional from Unintentional Noises (https://arxiv.org/abs/2409.19619)
- **What's New**: 이 논문은 Class-Independent Adversarial Intent (CIAI) 탐지 네트워크를 소개합니다. 이 네트워크는 수정된 Vision Transformer를 기반으로 하며 탐지 레이어가 포함되어 있습니다. 새로운 손실 함수는 Maximum Mean Discrepancy와 Center Loss를 결합하여 이미지 클래스와 관계없이 의도적(적대적 공격) 및 비의도적 노이즈를 모두 탐지할 수 있게 설계되었습니다.

- **Technical Details**: CIAI 네트워크는 두 단계로 훈련됩니다. 첫 번째 단계에서는 Maximum Mean Discrepancy (MMD)와 Center Loss를 사용하여 Vision Transformer를 훈련하며, 두 번째 단계에서는 공격을 탐지하는 능력을 강화하기 위해 교차 엔트로피 손실을 사용합니다. 이 탐지 네트워크는 CelebA, CelebA-HQ, LFW, AgeDB, CIFAR-10 데이터셋에서의 성능을 평가합니다.

- **Performance Highlights**: 제안된 CIAI 탐지기는 FGSM, PGD, DeepFool과 같은 의도적 perturbations 뿐만 아니라 Gaussian 및 Salt & Pepper 노이즈와 같은 비의도적 perturbations도 탐지하는 데 성공적으로 작동합니다. 이를 통해 모델의 안정성과 보안을 강화할 수 있습니다.



### Hybrid Mamba for Few-Shot Segmentation (https://arxiv.org/abs/2409.19613)
Comments:
          This paper is accepted by NIPS'24

- **What's New**: 본 논문에서는 기존의 Few-Shot Segmentation (FSS) 방식에서 발생하는 두 가지 문제를 해결하기 위해 새로운 하이브리드 Mamba 네트워크(HMNet)를 제안합니다. HMNet은 (1) 주기적으로 지원(지원 FG) 특성을 다시 캡슐화하여 혼합을 최소화하고, (2) 쿼리 픽셀 간의 상호작용을 차단하여 지원 정보를 효과적으로 융합하는 방안을 도입합니다.

- **Technical Details**: 하이브리드 Mamba 블록(HMB)은 두 가지 구성 요소로 나뉘어 있습니다: (1) 지원 재캡슐화 Mamba는 쿼리 스캔 중에 주기적으로 지원 FG 특성을 호출하여 항상 풍부한 지원 정보를 유지합니다; (2) 쿼리 차단 Mamba는 쿼리 픽셀 간의 상호작용을 금지하여 각 쿼리 픽셀이 지원 FG 정보를 융합할 수밖에 없도록 합니다. 결과적으로, 지원 정보를 효율적으로 활용할 수 있어 최적의 성능을 달성합니다.

- **Performance Highlights**: PASCAL-5i와 COCO-20i의 두 개의 공공 벤치마크 데이터셋에서 수행된 실험 결과, HMNet은 기존 최첨단 모델을 각각 2.2%와 3.2% 향상시켰으며, 평균 교차점 분할 비율(mIoU)에서 우수한 성능을 나타냈습니다.



### Causal Deciphering and Inpainting in Spatio-Temporal Dynamics via Diffusion Mod (https://arxiv.org/abs/2409.19608)
- **What's New**: 본 논문에서는 spatio-temporal (ST) 예측을 위한 새로운 인과 구조 플러그인인 CaPaint를 제안합니다. CaPaint는 데이터에서 인과적 영역을 식별하고 모델에 인과적 추론 능력을 부여하는 이중 단계 프로세스입니다. 특히 기존의 높은 데이터 생성 복잡도를 완화하여 최적의 ST 인과 발견 모델의 복잡성 문제를 해결합니다.

- **Technical Details**: CaPaint는 causal framework를 사용하여 causal regions를 탐색하면서 비인과적(non-causal) 영역을 개입하여 모델의 일반화 가능성과 해석력을 향상시키는 방법론입니다. 이는 self-attention 기법을 통한 구조적 재구성을 기반으로 하며, Denoising Diffusion Probabilistic Models (DDPM)을 활용하여 환경 패치의 노이즈를 제거하는 이미지 인페이팅(image inpainting) 기법을 구현합니다.

- **Performance Highlights**: CaPaint의 효과는 다섯 개의 실제 ST 데이터 세트를 통해 실험되었고, 성능 향상은 4.3%에서 77.3%까지 다양했습니다. 또한 전통적인 ST 데이터 증강 모델들과 비교하여 diffusion 모델의 잠재력을 강조하며, ST 향상을 위한 새로운 패러다임을 제공합니다.



### One Token to Seg Them All: Language Instructed Reasoning Segmentation in Videos (https://arxiv.org/abs/2409.19603)
Comments:
          Accepted by NeurlPS 2024

- **What's New**: 새로운 논문에서는 영상 기반의 다중 모달 대형 언어 모델인 VideoLISA를 소개합니다. 이는 언어 지시를 통한 영상 분할 문제를 해결하기 위해 설계되었습니다.

- **Technical Details**: VideoLISA는 Sparse Dense Sampling 전략과 One-Token-Seg-All 접근 방식을 통합하여 영상에서 객체를 세분화 및 추적하는 기능을 제공합니다. 특히 <TRK> 토큰을 사용하여 모델이 여러 프레임에 걸쳐 객체를 세분화할 수 있도록 합니다.

- **Performance Highlights**: 새로 도입된 ReasonVOS 벤치마크와 다양한 공개 벤치마크에서의 평가 결과, VideoLISA는 복잡한 추론, 시간적 이해 및 객체 추적을 포함한 비디오 객체 세분화 작업에서 우수한 성능을 보입니다.



### Gradient is All You Need: Gradient-Based Attention Fusion for Infrared Small Target Detection (https://arxiv.org/abs/2409.19599)
- **What's New**: 이번 논문에서는 복잡한 배경에 의해 가려지는 작은 적외선 (Infrared) 표적을 효과적으로 탐지하는 새로운 방법인 Gradient Network (GaNet)을 제안합니다. GaNet은 작은 표적의 가장자리와 기울기 정보를 추출하고 보존하는 데 중점을 두고 있으며, Gradient Transformer (GradFormer) 모듈을 통해 특징을 통합합니다.

- **Technical Details**: GaNet은 두 가지 주요 혁신 요소를 포함합니다: 1) GradFormer 모듈은 중앙 차이 합성곱 (CDC)을 시뮬레이션하여 기울기 정보를 추출하며, 2) 전역 특징 추출 모델 (GFEM)은 세부 정보에만 초점을 맞추는 것을 방지하고 배경 정보를 거시적으로 통합하여 접근합니다. CDC를 기반으로 하여 동적 가중치를 사용하여 더 강력하고 적합한 특성 표현을 학습하도록 설계되었습니다.

- **Performance Highlights**: IRSTD-1K 및 NUDT-SIRST 데이터셋에서 수행된 비교 실험 결과, GaNet은 기존의 최첨단 기술 (SOTA) 방법들을 초월하는 성능을 보여주었습니다. 이 연구는 작은 적외선 표적 탐지 기술 향상에 기여할 것으로 기대됩니다.



### DiffCP: Ultra-Low Bit Collaborative Perception via Diffusion Mod (https://arxiv.org/abs/2409.19592)
Comments:
          7 pages, 4 figures

- **What's New**: DiffCP는 효율적인 데이터 전송을 위한 새로운 협업 인식 패러다임으로, 확산 모델(diffusion model)을 활용하여 센싱 정보를 압축합니다. 기하학적 및 의미적 조건을 통합하여 특성 수준의 협업을 가능하게 하며, 저렴한 통신 비용으로 성능을 향상시킵니다.

- **Technical Details**: DiffCP는 트랜스포머 기반의 확산 모델을 사용하여 협업 에이전트(co-agent)의 관측을 재구성합니다. 이 모델은 다양한 센서 모달리티를 지원하고, 고유한 전경 정보(foreground information)를 유지하며, 유니버설 Bird’s Eye View (BEV) 공간 내에서 이뤄지는 확산(difussion) 과정으로 인퍼런스 차원을 감소시킵니다.

- **Performance Highlights**: DiffCP는 communication overhead를 14.5배 줄이면서도 기존의 state-of-the-art 알고리즘과 동일한 성능을 유지합니다. 이는 차세대 협력 로봇 시스템의 실 세계 배포 및 구현을 촉진합니다.



### Effective Diffusion Transformer Architecture for Image Super-Resolution (https://arxiv.org/abs/2409.19589)
Comments:
          Code is available at this https URL

- **What's New**: 최근의 연구들은 확산 모델(diffusion models)이 이미지 초해상도(image super-resolution)에서 뛰어난 성능을 보일 수 있다는 가능성을 보여주고 있습니다. 본 논문에서는 효과적인 확산 변환기(diffusion transformer) 모델인 DiT-SR을 설계하였으며, 이는 기존의 방법들과 비교하여 매우 높은 시각적 품질을 달성합니다.

- **Technical Details**: DiT-SR은 U자형 아키텍처(U-shaped architecture)를 채택하고 있으며, 모든 변환기 블록(transformer blocks)에 대해 uniform isotropic 디자인을 적용합니다. 이 모델은 multi-scale 계층적(feature extraction) 특징 추출을 용이하게 하며, 중요 계층에 컴퓨팅 자원을 재배분하여 성능을 더욱 향상시킵니다. 또한, 모델이 서로 다른 시간 단계에서 다양한 주파수 정보를 처리할 수 있도록 하는 frequency-adaptive time-step conditioning 모듈인 AdaFM을 제안하였습니다.

- **Performance Highlights**: 다양한 실험을 통해 DiT-SR이 기존의 training-from-scratch 기반의 초해상도 방법들보다 우수한 성능을 발휘한다는 것을 입증하였습니다. 또한, 일부의 기존 사전 학습(pretrained) 방법들에 대해서도 성능이 우수함을 보여주며, 이는 확산 변환기(diffusion transformer)의 초해상도(image super-resolution)에서의 우수성을 입증합니다.



### Self-supervised Auxiliary Learning for Texture and Model-based Hybrid Robust and Fair Featuring in Face Analysis (https://arxiv.org/abs/2409.19582)
- **What's New**: 이 논문에서는 Face Analysis(얼굴 분석)에 대한 Self-supervised Learning(자기 지도 학습, SSL) 보조 작업을 탐색하여 텍스처 기반 로컬 디스크립터를 Feature Modeling(특징 모델링)과 혼합하는 접근 방식을 제안합니다. 저자들은 SSL과 기본 작업을 결합함으로써 얼굴 분석의 강력하고 편향 없는 표현을 달성할 수 있음을 보였습니다.

- **Technical Details**: 제안된 방법론은 Masked Auto-Encoder (MAE)와 Local Directional Patterns (LDP)을 이용하여 텍스처 특징을 복원하는 SSL 보조 작업을 수행합니다. ViT(비전 트랜스포머) 아키텍처를 기반으로 하여 데이터 증강을 통해 다양한 방향에서 특징을 학습하고, 이미지 복원 및 분류 작업을 동시에 최적화하여 더 나은 데이터 표현을 이끌어냅니다.

- **Performance Highlights**: 실험 결과, 제안된 모델이 얼굴 속성 분석, 감정 분석, Deepfake 탐지 등 다양한 얼굴 분석 작업에서 기존 방법들보다 우수한 특징 표현을 보여주었으며, 공정하고 편향 없는 얼굴 분석을 위한 매우 유용한 접근법임을 입증했습니다.



### High Quality Human Image Animation using Regional Supervision and Motion Blur Condition (https://arxiv.org/abs/2409.19580)
- **What's New**: 본 논문에서는 고해상도 인체 애니메이션을 위해 얼굴과 손의 디테일을 향상시키고, 모션 블러(motion blur)를 명시적으로 모델링하는 새로운 방법을 제안합니다. 이 방법은 지역 감독(regional supervision)을 활용하여 정밀성을 높이고, 훈련 전략을 혁신하여 기존 방법들의 한계를 극복하는 것을 목표로 합니다.

- **Technical Details**: HIA(High quality human Image Animation) 프레임워크는 지역 감독을 통해 얼굴과 손의 충실도를 보장하고, 모션 블러 조건을 통합하여 고품질 비디오 프레임을 유지합니다. 훈련은 공간 모듈과 시간 모듈의 두 단계로 나뉘며, 프로그레시브 훈련 전략(progressive training strategy)을 통해 성능을 극대화합니다.

- **Performance Highlights**: HIA는 HumanDance 데이터셋에서 기존 최첨단 방법보다 21.0%의 재구성 정확도(L1)와 57.4%의 인지 품질(FVD) 향상을 보여주며 뛰어난 비디오 충실도와 일반화 능력을 입증하였습니다.



### See then Tell: Enhancing Key Information Extraction with Vision Grounding (https://arxiv.org/abs/2409.19573)
- **What's New**: 본 논문에서는 STNet(See then Tell Net)라는 새로운 종단 간 모델을 도입합니다. 이 모델은 정확한 답변을 제공하고 관련된 비전 기반을 함께 제시하는 데 중점을 두고 설계되었습니다. 특히 <see> 토큰을 사용하여 이미지 내에서 적절한 영역을 관찰하고 텍스트 응답을 구성할 수 있습니다.

- **Technical Details**: STNet는 특별히 설계된 <see> 토큰을 활용하여 모델이 이미지 내에서 관련 위치를 인식하도록 지원합니다. 이는 전문화된 물리적 디코더를 통해 <see> 토큰과 연결된 물리적 좌표를 해석하게 됩니다. 다음 작업에서는 <see> 토큰을 답변 텍스트의 시작 부분에 배치하여 답변에 대한 비전 기반을 제공합니다.

- **Performance Highlights**: STNet 모델은 CORD, SROIE, DocVQA와 같은 공개 데이터셋에서 최첨단 성과를 달성하여 KIE 성능에서 상당한 발전을 보여주었습니다. 이를 통해 비전 기반을 활용한 KIE 모델의 성능이 크게 개선될 것으로 기대하고 있습니다.



### Fully Aligned Network for Referring Image Segmentation (https://arxiv.org/abs/2409.19569)
- **What's New**: 이 논문은 Referring Image Segmentation (RIS) 작업에 중점을 두고 있으며, 자연어 설명에 따라 이미지에서 객체를 세분화하는 방법을 제시합니다. 기존 방법들은 교차-모달 상호작용을 명확히 설계하지 않아, 상호 이해에 부족함이 있었습니다. 본 연구에서는 이러한 문제를 해결하기 위해 Fully Aligned Network (FAN)를 제안하며, 명확한 상호작용 원칙에 기반하여 설계되었습니다.

- **Technical Details**: FAN은 네 가지 교차-모달 상호작용 원칙(Encoding Interaction, Coarse and Fine-Grained Interaction, Multi-Scale Interaction, Bidirectional Interaction)을 따릅니다. 각기 다른 이미지 특성 및 언어 표현을 분석하여 공동 공간을 생성하며, 이를 통해 예측 마스크를 간단한 유사도 계산으로 생성합니다. 모델에서는 비전 인코더와 언어 인코더가 각기 시각적 및 언어적 특징을 추출하며, 다중 스케일 활성화 모듈을 통해 객체를 강조합니다.

- **Performance Highlights**: FAN은 RefCOCO, RefCOCO+, G-Ref 데이터셋에서 state-of-the-art 성능을 달성했습니다. FAN은 교차-모달 정렬을 완벽하게 이루어내는 간단하면서도 강력한 구조를 가지고 있으며, 관련된 여러 작업에서 탁월한 결과를 보여주고 있습니다.



### CLIP-based Camera-Agnostic Feature Learning for Intra-camera Person Re-Identification (https://arxiv.org/abs/2409.19563)
Comments:
          Submitted to IEEE TCSVT

- **What's New**: 이번 논문에서는 Contrastive Language-Image Pre-Training (CLIP) 모델을 활용한 새로운 프레임워크인 CLIP-based Camera-Agnostic Feature Learning (CCAFL)을 제안합니다. 이는 Intra-Camera Supervised Re-Identification (ICS ReID) 문제를 해결하기 위해 고안되었습니다.

- **Technical Details**: 논문에서 제안된 프레임워크는 두 개의 주요 모듈인 Intra-Camera Discriminative Learning (ICDL)과 Inter-Camera Adversarial Learning (ICAL)을 포함합니다. ICDL은 카메라 내에서의 세부적인 보행자 특징을 학습하도록 유도하고, ICAL은 서로 다른 카메라 간의 보행자 특징의 차이를 줄이기 위해 모델의 카메라 예측 능력을 저해합니다.

- **Performance Highlights**: MSMT17 데이터셋에서 mAP 정확도를 58.9%로 기록하며, 기존의 최첨단 방법들을 7.6% 초과하는 성능을 보여주었습니다. Market-1501과 MSMT17을 포함한 다양한 ReID 벤치마크에서 우수한 성능을 입증했습니다.



### Tri-Cam: Practical Eye Gaze Tracking via Camera Network (https://arxiv.org/abs/2409.19554)
Comments:
          12 pages

- **What's New**: Tri-Cam은 세 개의 저렴한 RGB 웹캠을 활용해 개발된 심층 학습 기반의 주시 추적 시스템으로, 사용자 이동에 대한 대응 능력을 개선하고 사용자의 수고를 줄일 수 있는 암시적 보정 모듈을 특징으로 합니다.

- **Technical Details**: 이 시스템은 효율적인 훈련을 위한 분할 네트워크 구조를 가지고 있으며, 카메라-눈 및 눈-스크린 지오메트리를 별도로 처리하는 네트워크 설계를 포함합니다. 특히, 카메라-눈 지오메트리는 여러 카메라가 형성하는 삼각형을 통해 깊이 감지를 가능하게 하며, 내부 검증 메커니즘을 사용하여 더욱 향상된 성능을 제공합니다.

- **Performance Highlights**: Tri-Cam은 정확도가 Tobii 고급 상업용 눈 추적기와 비교해 비슷한 수준을 유지하면서 사용자 이동 범위를 더 넓게 지원하는 것으로 평가되었습니다. 21명의 참가자를 대상으로 한 실험 결과, Tri-Cam은 50cm 거리에서 평균 시선 추론 오차가 2.06cm로 나타났으며, 이는 Tobii의 1.95cm 오차와 근접합니다.



### BiPC: Bidirectional Probability Calibration for Unsupervised Domain Adaption (https://arxiv.org/abs/2409.19542)
- **What's New**: 본 논문에서는 Bidirectional Probability Calibration (BiPC)이라는 새로운 방법을 제안하며, 이는 유명한 Transformer 기반 방법의 제한을 해결하기 위한 연구입니다. BiPC는 대칭적인 확률 보정을 통해 도메인 간 간극을 줄이고, 다양한 네트워크에 적용 가능합니다.

- **Technical Details**: BiPC는 두 가지 주요 기술을 도입합니다: 1) Calibrated Probability Alignment (CPA), 이는 사전 훈련된 네트워크의 확률 출력을 보정하는 방법입니다. 2) Calibrated Gini Impurity (CGI) 손실로, 이는 작업 헤드 보정을 위해 사전 훈련된 분류기의 확률 공간에서 학습된 보정 계수를 사용합니다.

- **Performance Highlights**: BiPC는 여러 도메인 적응 벤치마크에서 뛰어난 성능을 보여주며, 기존의 feature alignment와 Transformer 기반 방법들과 비교하여 주목할 만한 결과를 달성했습니다. 특히, BiPC는 UDA 및 PDA 작업에서 최첨단 성능을 기록하고 있습니다.



### LoRKD: Low-Rank Knowledge Decomposition for Medical Foundation Models (https://arxiv.org/abs/2409.19540)
Comments:
          The paper is an extended version of our conference paper published on CVPR 2024. This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 이번 논문에서는 "Knowledge Decomposition"이라는 새로운 관점을 통해 의료 특정 작업의 성능을 향상시키기 위한 접근 방식을 제안합니다. 이는 기초 모델을 여러 경량의 전문 모델로 분해하고 각 모델이 특정 해부학적 영역에 전념하도록 하는 것입니다.

- **Technical Details**: 제안된 방법인 Low-Rank Knowledge Decomposition (LoRKD)는 저랭크 전문가 모듈과 효율적 지식 분리 컨볼루션을 포함합니다. 이 방법은 서로 다른 해부학적 영역에서 발생하는 그래디언트 충돌을 해결하여 강력한 전문성을 유지하면서 자원 소비를 줄입니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, 분해된 모델은 최첨단 성능을 달성하며 하류 작업에 대한 전이 가능성을 보여줍니다. 특히, 특정 작업 평가에서 원래의 기초 모델보다 뛰어난 성능을 나타냈습니다.



### Video DataFlywheel: Resolving the Impossible Data Trinity in Video-Language Understanding (https://arxiv.org/abs/2409.19532)
Comments:
          Under peer review

- **What's New**: 이번 연구에서는 데이터 수량, 다양성, 품질 간의 "불가능한 삼위일체"를 밝히며, 기존의 대규모 ASR 데이터셋이 품질 부족으로 인해 개선을 요구하고 있음을 강조합니다.

- **Technical Details**: 우리는 Video DataFlywheel 프레임워크를 도입하여 비디오 주석을 반복적으로 개선하며, AdaTaiLr라는 새로운 noise control 방법을 통해 대규모 데이터셋에서의 효율성을 증명합니다. 또한, 비디오-언어 모델을 활용하여 합성 주석을 생성하고, 이를 통해 데이터셋을 정제합니다.

- **Performance Highlights**: 우리의 프레임워크는 기존 데이터 정제 기준선보다 3% 성능 향상을 보였으며, 다양한 비디오-언어 이해 과제에서 중요한 개선을 facilitated 합니다.



### MedCLIP-SAMv2: Towards Universal Text-Driven Medical Image Segmentation (https://arxiv.org/abs/2409.19483)
Comments:
          10 pages, 2 figures, 6 tables

- **What's New**: 이 논문에서는 CLIP와 SAM 모델을 통합한 MedCLIP-SAMv2라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 텍스트 프롬프트를 사용하여 의료 이미지를 제로샷(Zero-shot) 및 약간의 감독하에 세분화하는 기능을 제공하며, 특히 의료영상 분석에서 데이터 효율성을 높이기 위해 제안되었습니다.

- **Technical Details**: MedCLIP-SAMv2 프레임워크는 BiomedCLIP 모델을 새로운 분리된 하드 네거티브 노이즈 대조 추정(DHN-NCE) 손실을 사용하여 파인 튜닝(tuning) 합니다. 또한 Multi-modal Information Bottleneck (M2IB)을 활용하여 제로샷 설정에서 SAM을 통해 세분화 마스크를 생성하기 위한 비주얼 프롬프트를 생성합니다. 프레임워크는 CT, MRI, 초음파 및 X-ray와 같은 다양한 의학 이미징 모달리티에 걸쳐 실험되었습니다.

- **Performance Highlights**: MedCLIP-SAMv2 프레임워크는 유방 종양 초음파, 뇌 종양 MRI, 폐 X-ray 및 폐 CT를 포함한 네 가지 세분화 작업에서 높은 정확도를 달성했습니다. 이러한 실험 결과는 프레임워크의 강력함과 다재다능성을 보여 줍니다.



### FairPIVARA: Reducing and Assessing Biases in CLIP-Based Multimodal Models (https://arxiv.org/abs/2409.19474)
Comments:
          14 pages, 10 figures. Accepted to 35th British Machine Vision Conference (BMVC 2024), Workshop on Privacy, Fairness, Accountability and Transparency in Computer Vision

- **What's New**: 이 논문은 비전-언어 모델의 윤리적 의미에 초점을 맞추고 이들 모델에서 발생할 수 있는 차별적 관행을 분석합니다. 특히 모델이 타 언어에 맞춤화되는 과정에서 나타나는 새로운 편향을 논의하며, FairPIVARA라는 편향 감소 기술을 제안합니다.

- **Technical Details**: FairPIVARA는 임베딩(feature embedding)에서 가장 부정적인 기여를 하는 차원(dimension)을 제거하여 편향을 줄이는 알고리즘입니다. 이 모델은 CAPIVARA를 기반으로 하며, 주로 Disability, Nationality, Religion, Sexual Orientation 측면에서 차별적 실천을 분석합니다. 모델의 성능은 98%까지 감소된 편향을 나타냈고, 단어 분포의 균형도 증진되었습니다.

- **Performance Highlights**: FairPIVARA의 적용을 통해 편향 관찰이 98%까지 줄어들었으며, 모델 내에서 단어 분포의 균형을 촉진했습니다. 이 연구는 영어와 포르투갈어 결과를 모두 보고하며, 낮은 자원 언어로의 모델 적합에 대한 이해를 높였습니다.



### Towards Croppable Implicit Neural Representations (https://arxiv.org/abs/2409.19472)
Comments:
          Accepted to NeurIPS 2024

- **What's New**: 이 논문에서는 수정 가능한 암묵적 신경 표현(Implicit Neural Representations, INRs)의 새로운 아키텍처인 Local-Global SIRENs을 제안합니다. 이 구조는 크로핑(cropping) 작업에 최적화되어 있으며, 특정 신호의 부분을 제거할 수 있는 능력을 제공합니다.

- **Technical Details**: Local-Global SIRENs는 로컬(local) 및 글로벌(global) 기능 추출을 결합하여 신호를 인코딩하는 혁신적인 INR 아키텍처입니다. 이 아키텍처는 인코딩된 신호의 특정 부분을 쉽게 제거하며, 이를 통해 비율에 맞는 가중치 감소가 가능합니다. 네트워크를 재훈련할 필요 없이 가중치를 제거하는 방법으로 구현되었습니다.

- **Performance Highlights**: Local-Global SIRENs는 훈련 속도를 높이고 다양한 신호의 인코딩을 향상시키며 다운스트림 성능을 개선하는 데 기여합니다. 특히, INCODE와 같은 현대 INRs에 적용하여 이러한 우수성을 입증하였으며, 기존 기본 INRs보다 개선된 성능을 보여주었습니다.



### Contrastive ground-level image and remote sensing pre-training improves representation learning for natural world imagery (https://arxiv.org/abs/2409.19439)
Comments:
          Accepted to ECCV 2024

- **What's New**: 본 연구는 다중 시각(multi-view) 이미지 대비 학습(contrastive learning)을 활용하여 이미지 데이터의 여러 뷰를 활용함으로써 종 식별의 정교한 분류 성능을 개선할 수 있음을 보여준다. 새로운 사전 훈련 작업인 CRISP(ContRastive Image-remote Sensing Pre-training)를 제안하며, 6,000종 이상의 식물 분류가 포함된 3백만 개 이상의 지상 및 공중 이미지 쌍을 포함하는 Nature Multi-View(NMV) 데이터셋을 소개한다.

- **Technical Details**: CRISP는 지상과 공중 이미지를 비교 및 대조하여 자연 세계에서의 이미지 표현 학습을 개선하는 자기 지도(Self-supervised) 사전 훈련 접근 방식이다. NMV 데이터셋은 캘리포니아의 생태적으로 다양한 지역 내에서 6,000개 이상의 식물 군을 포함한 3백만 개 이상의 지상 및 공중 이미지 쌍으로 구성되어 있다.

- **Performance Highlights**: CRISP 다중 뷰 사전 훈련은 종 식별 및 종 분포 매핑을 위한 다양한 클래스 균형 및 비균형 메트릭에서 정확도를 향상시키며, 가장 적은 레이블 데이터와 가장 희귀한 클래스를 가진 경우에 가장 큰 성능 향상을 보여준다. 또한 CRISP는 일치하는 하류 자연 세계 이미지 작업에 대한 몇 샷 전이 학습(few-shot transfer learning) 능력을 향상시킨다.



### Introducing SDICE: An Index for Assessing Diversity of Synthetic Medical Datasets (https://arxiv.org/abs/2409.19436)
Comments:
          Accepted at BMVC 2024 - PFATCV

- **What's New**: 이번 연구에서는 SDICE 지수를 제안하며, 이는 대조 인코더(contrastive encoder)에 의해 유도된 유사성 분포의 특성을 기반으로 합니다. 이 지수는 합성 데이터셋의 다양성을 평가하는 새로운 방법입니다.

- **Technical Details**: SDICE 지수는 원래의 이미지와 합성 이미지 간의 유사성 점수 분포의 거리(distance)를 측정하여, 이는 비율(F-ratio)을 거리로 사용하고 결과 거리 값에 지수 함수(exponential function)로 정규화(normlization)합니다.

- **Performance Highlights**: MIMIC-chest X-ray 및 ImageNet 데이터셋에서 실시한 실험 결과, SDICE 지수는 합성 의료 데이터셋의 다양성을 평가하는 데 효과적임을 보여주었습니다.



### Fast Encoding and Decoding for Implicit Video Representation (https://arxiv.org/abs/2409.19429)
Comments:
          ECCV 2024. Project page at this https URL, code will be at this https URL

- **What's New**: 이번 논문에서는 기존의 gradient 기반 최적화 방식을 피하고 NeRV-Enc라는 transformer 기반의 하이퍼 네트워크를 사용하여 비디오 인코딩 속도를 10,000배 향상시키고, NeRV-Dec라는 병렬 디코더를 통해 비디오 로딩 효율성을 높이는 방법을 제안합니다.

- **Technical Details**: NeRV-Enc는 입력 비디오 조각과 초기 가중치 토큰을 입력으로 받아 모델 가중치를 생성하는 하이퍼 네트워크입니다. 이로 인해 인코딩 과정을 간소화하여 인코딩 시간을 크게 단축시키며, NeRV-Dec는 비디오 디코딩 과정을 병렬화하여 H.264와 비교해 11배 더 빠르게 동작합니다. 또한, RAM에서 미리 디코딩된 비디오 로딩보다 2.5배 빠르며, 메모리 사용량은 65배 감소합니다.

- **Performance Highlights**: NeRV-Enc는 인코딩 속도를 10,000배 향상시켰으며, NeRV-Dec는 전통적인 비디오 코덱 H.264比에서 11배 빠른 속도를 자랑합니다. 이 두 기술은 대규모 비디오 연구 및 스트리밍 작업에서 매우 유용하게 활용될 수 있습니다.



### From Unimodal to Multimodal: Scaling up Projectors to Align Modalities (https://arxiv.org/abs/2409.19425)
Comments:
          Preprint, 10 pages; First two authors contributed equally

- **What's New**: 최근 멀티모달(contrastive multimodal) 비전-언어 모델들이 강력한 제로샷(Zero-shot) 성능을 보여줌에 따라, 이 논문은 단일 모달 언코더(encoder)를 연결하여 제로샷 비전-언어 작업에서 효과적으로 활용할 수 있는 새로운 접근 방식을 제시합니다.

- **Technical Details**: 제안된 방식은 세 가지 핵심 구성 요소로 이루어져 있습니다: 1) Encoder Pair Selection - Centered Kernel Alignment (CKA) 방식으로 semantically 유사한 비전 및 언어 encoders를 선택, 2) Dataset Curation - 다양한 이미지-캡션 쌍으로 구성된 밀집하고 개념이 풍부한 데이터셋을 수집, 3) Lightweight Projector Training - frozen unimodal models의 임베딩 공간 간에 간단한 MLP 프로젝터를 훈련하여 다양한 멀티모달 시나리오에 적응합니다.

- **Performance Highlights**: 제안된 방법은 12개의 제로샷 분류 데이터셋과 2개의 이미지-텍스트 검색 데이터셋에서 평가되었습니다. DINOv2와 All-Roberta-Large 텍스트 인코더를 활용한 최상의 모델은 ImageNet에서 76%의 정확도를 기록하였으며, 이는 CLIP 모델보다 뛰어난 성능이며 데이터 요구량은 20배, 계산 요구량은 65배 감소하였습니다.



### G3R: Gradient Guided Generalizable Reconstruction (https://arxiv.org/abs/2409.19405)
Comments:
          ECCV 2024. Project page: this https URL

- **What's New**: G3R은 대형 장면에 대한 고품질 3D 장면 표현을 효율적으로 예측할 수 있는 일반화 가능한 재구성 접근법으로 소개된다. 이 방법은 장면 최적화의 이점과 빠른 예측 방법의 데이터 기반 사전 정보를 결합하여 고품질 재구성을 실현한다.

- **Technical Details**: G3R은 각 장면의 경량 재구성 네트워크를 통해 이미지의 기울기 피드백 신호를 학습하여 3D 장면 표현을 반복적으로 업데이트한다. 기본적으로, 기존의 방법들과는 달리 2D 이미지를 3D 공간으로 리프트하는 새로운 방식을 제안하며, 이를 통해 효율적으로 3D 그래디언트를 집계하고 고품질 재구성을 가능하게 한다.

- **Performance Highlights**: G3R은 두 개의 대규모 외부 데이터셋에서 실험을 통해 24회의 반복으로 Large Scale Scene을 빠르고 정확하게 복원하며, 기존의 3DGS와 비교했을 때 유사하거나 더 나은 사실성을 제공하면서도 최소 10배 빠른 성능을 보여준다.



### Restore Anything with Masks: Leveraging Mask Image Modeling for Blind All-in-One Image Restoration (https://arxiv.org/abs/2409.19403)
Comments:
          Accepted by ECCV 2024

- **What's New**: 이 논문에서는 여러 가지 이미지 손상 유형을 하나의 모델로 처리할 수 있는 All-in-one 이미지 복원 방법을 제안합니다. 특히, 'Restore Anything with Masks (RAM)'라는 새로운 파이프라인을 통해 이미지의 내재적 정보를 추출하는 데 중점을 두고 있습니다.

- **Technical Details**: 이 파이프라인은 두 개의 단계로 구성되며, 첫 번째 단계는 Masked Image Pre-training이며 두 번째 단계는 Mask Attribute Conductance (MAC)를 사용한 Fine-tuning입니다. Masked Image Modeling (MIM)을 활용하여 손상된 이미지에서 그에 상응하는 깨끗한 이미지를 예측하도록 네트워크를 훈련시킵니다. MAC을 통해 각 레이어의 중요도를 평가하여 성능 향상에 기여하는 최상위 레이어를 선택하여 미세 조정합니다.

- **Performance Highlights**: 제안된 RAM 방법은 최첨단 성능을 달성하며, 기존의 방법보다 더 균형 잡힌 성능을 보입니다. 추가적인 계산 오버헤드 없이 어느 네트워크에서나 적용할 수 있어 여러 복원 작업에 효과적입니다.



### Conditional Image Synthesis with Diffusion Models: A Survey (https://arxiv.org/abs/2409.19365)
- **What's New**: 본 논문에서는 사용자가 제공한 조건을 기반으로 한 이미지를 생성하는 조건부 이미지 합성을 위한 새로운 접근 방법으로, diffusion 기반의 generative modeling을 중심으로 논의합니다. 특히, 기존의 접근 방식을 다양한 조건을 통합하는 두 가지 기본 구성 요소인 denoising network와 sampling process를 기준으로 분류하여 연구합니다.

- **Technical Details**: Diffusion 기반 모델링의 복잡성과 다양한 이미지 합성 작업, 다양한 conditioning mechanism들이 연구자들에게 도전 과제가 됩니다. 본 논문에서는 다양한 conditioning 접근 방식의 원리와 장점, 잠재적 도전에 관한 논의를 포함합니다. 또한, 기본 sampling 과정에서 여섯 가지 주요 조건 메커니즘을 요약하였습니다.

- **Performance Highlights**: 이번 조사는 텍스트-이미지 생성, 이미지 복원, 이미지 편집 등 다양한 조건부 이미지 합성 작업의 성능을 향상시키기 위해 필요한 훈련, 재사용 및 전문화 과정의 이점을 강조합니다. 엔드 투 엔드 접근 방식과 다양한 샘플링 기법의 조합을 통해 기존의 한계를 극복할 수 있는 가능성을 제시합니다.



### Solution of Multiview Egocentric Hand Tracking Challenge ECCV2024 (https://arxiv.org/abs/2409.19362)
Comments:
          Accepted in ECCV2024 workshop

- **What's New**: 이 연구에서는 VR(가상 현실) 상호작용을 위해 다중 뷰 입력 이미지와 카메라 외부 매개변수를 사용하여 손 모양과 자세를 추정하는 방법을 제시합니다. 또한 손 위치와 자세의 정확성을 향상시키기 위한 오프라인 Neural Smooth 후처리 방법을 도입했습니다.

- **Technical Details**: 연구된 아키텍처는 단일 뷰(single-view) 및 다중 뷰(multi-view) 입력을 모두 처리할 수 있으며, 3D 기능을 얻기 위해 특징 추출기와 특징 융합 모듈을 포함합니다. 데이터 증강(augmentation) 기법으로는 crop jittering 및 카메라 외부 매개변수에 대한 노이즈 증강을 사용했습니다. 모델은 Hiera-base 아키텍처를 기반으로 하여 8개의 NVIDIA 2080TI GPU에서 훈련되었습니다.

- **Performance Highlights**: 이 방법은 Umetrack 데이터셋에서 13.92mm MPJPE, HOT3D 데이터셋에서 21.66mm MPJPE의 결과를 달성하여 다양한 데이터셋에서 성능 향상을 입증했습니다.



### Steering Prediction via a Multi-Sensor System for Autonomous Racing (https://arxiv.org/abs/2409.19356)
- **What's New**: 본 연구에서는 기존의 2D LiDAR 시스템에 이벤트 카메라를 통합하여 향상된 시간 정보를 제공하고, 2D LiDAR 데이터와 이벤트 데이터를 융합한 끝에서 끝까지의 학습 프레임워크를 통해 조향 예측을 수행합니다. 이러한 접근 방식은 자율 주행 경주에 대한 최초의 연구로, 조향 각도 예측에 초점을 맞추었습니다.

- **Technical Details**: 이 논문은 F1tenth 플랫폼에서 2D LiDAR와 이벤트 카메라 데이터를 융합하여 조향 예측을 개선하는 멀티 센서 융합 프레임워크를 제안합니다. 특히, 제안된 자원 절약형 융합 설계는 기존의 높은 계산 비용 문제를 해결하며, 센서 잘못 정렬에 대한 강건성을 증가시키는 새로운 융합 학습 정책을 도입합니다.

- **Performance Highlights**: 본 연구는 LiDAR 단독에 비해 조향 예측을 현저히 향상시켜 RMSE를 7.72에서 1.28로 감소시켰습니다. 학습 가능한 파라미터의 11%만 사용하여 두 번째로 좋은 융합 방법보다 더 나은 정확도를 달성하였습니다. 논문의 소스 코드, 데이터셋 및 벤치마크는 미래 연구를 촉진하기 위해 공개될 예정입니다.



### X-Prompt: Multi-modal Visual Prompt for Video Object Segmentation (https://arxiv.org/abs/2409.19342)
Comments:
          ACMMM'2024

- **What's New**: 이번 논문에서는 RGB 외 다양한 모달리티(RGB-Thermal, RGB-Depth, RGB-Event)를 활용한 다중 모달 비디오 객체 분할(Multi-modal Video Object Segmentation, VOS)에 대해 새로운 보편적 프레임워크인 X-Prompt를 제안합니다. X-Prompt는 기존의 전체 매개변수 미세 조정(full-parameter fine-tuning) 방식을 넘어 다중 모달 작업에 효율적으로 적응할 수 있는 방식입니다.

- **Technical Details**: X-Prompt 프레임워크는 먼저 RGB 데이터를 사용하여 비디오 객체 분할 기본 모델을 사전 훈련(pre-train)한 후, 추가 모달리티를 활용하여 다운스트림 다중 모달 작업에 적응합니다. 이 프레임워크 내에서는 Multi-modal Visual Prompter(MVP)와 Multi-modal Adaptation Experts(MAEs)가 도입되어 다양한 모달리티 정보를 활용하여 객체를 정확하게 세분화(segment) 할 수 있도록 지원합니다.

- **Performance Highlights**: X-Prompt 프레임워크는 RGB-T, RGB-D, RGB-E를 포함한 3개의 다중 모달 비디오 객체 분할 작업에서 기존 방법들과 비교하여 우수한 성능을 보여주며, 특히 제한된 데이터로 훈련된 모델에 대해서도 일반화 능력을 유지하며 성능을 향상시키고 있습니다.



### 3D-CT-GPT: Generating 3D Radiology Reports through Integration of Large Vision-Language Models (https://arxiv.org/abs/2409.19330)
- **What's New**: 본 논문에서는 3D CT 스캔으로부터 방사선 보고서를 생성하기 위해 Visual Question Answering (VQA) 기반의 새로운 의료 비주얼 언어 모델인 3D-CT-GPT를 소개합니다. 기존의 연구들은 2D 의료 이미지에 집중했지만, 본 연구는 3D 이미지에서의 자동 보고서 생성의 중요성을 강조하며 이를 해결하려는 시도를 보여줍니다.

- **Technical Details**: 3D-CT-GPT는 CT ViT, 3D Average Pooling, 그리고 프로젝션 레이어를 결합하여 3D CT 스캔으로부터 직접적이고 정확한 방사선 보고서를 생성합니다. 모델은 공공 데이터셋에서의 사전 훈련과 소규모 개인 데이터셋에서의 미세 조정을 통해 훈련 전략을 최적화하여 데이터 요구사항을 줄이고 뛰어난 성능을 유지합니다.

- **Performance Highlights**: 3D-CT-GPT는 기존의 3D 방사선 보고서 생성 방법들에 비해 보고서의 정확성과 품질 측면에서 현저하게 우수한 성능을 보입니다. 이 모델은 임상 방사선 보고서 생성에 대한 강력한 솔루션을 제시하며, 진단 정확도와 보고서의 일관성을 높이는 데 기여합니다.



### Scalable Cloud-Native Pipeline for Efficient 3D Model Reconstruction from Monocular Smartphone Images (https://arxiv.org/abs/2409.19322)
Comments:
          Preprint

- **What's New**: 본 논문은 스마트폰 카메라로 캡처된 단안 2D 이미지에서 자동으로 3D 모델을 재구성할 수 있는 클라우드 네이티브 파이프라인을 제안합니다. 이 파이프라인은 Industry 4.0 기준에 부합하며, 교육 과정에서 직원의 전문성을 향상시킬 수 있는 디지털 트윈(digital twin) 모델을 생성할 수 있습니다.

- **Technical Details**: 제안된 파이프라인은 NVIDIA Research Labs에서 개발한 머신 러닝 모델과 Google의 ARCore 프레임워크에 기반한 고유한 포즈 보상 구성이 포함된 사용자 정의 포즈 레코더를 활용합니다. 이 시스템은 마이크로서비스 아키텍처(microservices architecture)를 채택하여 각 모듈이 독립적으로 작동할 수 있도록 설계되었습니다.

- **Performance Highlights**: 이 연구의 결과로, 재사용 가능한 3D 모델을 생성하고, 향상된 재료와 텍스처를 포함할 수 있으며, 외부 3D 모델링 소프트웨어 또는 3D 엔진에서 내보내기 및 사용자 정의가 가능합니다. 또한, 이 과정은 AR(증강 현실) 기능을 통해 데이터 수집 작업을 개선합니다.



### CausalVE: Face Video Privacy Encryption via Causal Video Prediction (https://arxiv.org/abs/2409.19306)
Comments:
          Submitted to ICLR 2025

- **What's New**: 이 논문에서는 얼굴 비디오와 개인 정보 보호의 혁신적인 상호작용 프레임워크인 CausalVE를 소개합니다. 이 프레임워크는 동적 인과 추론(dynamic causal reasoning)과 가역 신경망(reversible neural networks)을 통합하여 원래 비디오 내용을 생성된 덮개 얼굴 비디오와 무결하게 혼합합니다.

- **Technical Details**: CausalVE 프레임워크는 얼굴 교체(face swapping)를 위해 확산 모델(difussion model)을 활용하고, 비밀 비디오의 음성 시퀀스 특성과 시공간 시퀀스(spatiotemporal sequence) 특성을 사용하여 동적 비디오 추론(dynamic video inference) 및 예측을 수행합니다. 이 시스템은 원래 비디오를 가상의 비디오 안에 숨기고, 키를 사용해 정확히 복구할 수 있는 방법을 제공합니다.

- **Performance Highlights**: CausalVE는 공공 비디오 전파에서 뛰어난 보안을 제공하며, 정성적(qualitative) 및 정량적(quantitative) 관점 및 시각적(visual) 관점 모두에서 최신 기법(state-of-the-art methods)보다 우수한 성능을 입증하였습니다.



### EEPNet: Efficient Edge Pixel-based Matching Network for Cross-Modal Dynamic Registration between LiDAR and Camera (https://arxiv.org/abs/2409.19305)
- **What's New**: 본 논문에서는 EEPNet이라는 새로운 네트워크를 제안하여 LiDAR 포인트 클라우드와 카메라 이미지 간의 정확한 등록을 개선합니다. 이 접근법은 포인트 클라우드 투영에서 얻은 반사 맵을 활용하여 서로 다른 모달리티 간의 차이를 줄이고 실시간으로 등록할 수 있습니다.

- **Technical Details**: EEPNet은 포인트 클라우드를 반사 맵으로 변환하여 특징 매칭을 수행하며, 이 과정에서 가장자리를 이용하여 2D-2D 매칭 문제로 변환합니다. 또한, 매칭 최적화 레이어를 포함하여 등록 정확도와 계산 효율성을 크게 향상시킵니다.

- **Performance Highlights**: 실험 결과, EEPNet은 최신 방법들에 비해 더욱 뛰어난 정확성과 효율성을 보여주며, 실시간 등록 작업에 적합한 성능을 입증했습니다.



### VLAD-BuFF: Burst-aware Fast Feature Aggregation for Visual Place Recognition (https://arxiv.org/abs/2409.19293)
Comments:
          Presented at ECCV 2024; Includes supplementary; 29 pages; 7 figures

- **What's New**: 이번 연구에서는 Visual Place Recognition (VPR) 모델의 최신 기법인 VLAD-BuFF를 제안합니다. VLAD-BuFF는 클러스터 내에서 발생할 수 있는 'burstiness' 문제를 해결하고, 고차원 로컬 특성을 줄이기 위해 PCA를 사용한 사전 프로젝션을 이용한 Fast Feature Aggregation을 특징으로 합니다.

- **Technical Details**: VLAD-BuFF는 두 가지 주요 기여로 구성됩니다: i) self-similarity 기반의 feature discounting 메커니즘을 통해 burst-aware feature를 학습하고, ii) PCA 초기화된 학습 가능한 사전 프로젝션을 통해 로컬 특성의 차원을 줄입니다. 이 방법은 9개의 공개 데이터셋에서 벤치마킹되었으며, 높은 recall을 유지하면서 12배 줄어든 로컬 특성 차원에서도 빠른 집합이 가능합니다.

- **Performance Highlights**: VLAD-BuFF는 기존의 모델들에 비해 대부분의 데이터셋에서 state-of-the-art 성능을 보이며, 특히 recall 성능이 향상되었습니다. 제안된 가중치 방법은 비특징적인 특성을 효과적으로 낮추는 것으로 확인되었습니다.



### CLIP-MoE: Towards Building Mixture of Experts for CLIP with Diversified Multiplet Upcycling (https://arxiv.org/abs/2409.19291)
- **What's New**: 최근 CLIP(Contrastive Language-Image Pre-training) 모델의 정보 손실 문제를 해결하기 위한 새로운 전략, Diversified Multiplet Upcycling (DMU)을 제안합니다. 이 방법은 다양한 특성 공간을 캡처하는 여러 CLIP 모델을 효율적으로 미세 조정하여, 기존 프리트레인된 CLIP 체크포인트를 활용하면서 모델 용량을 확장합니다.

- **Technical Details**: DMU는 CLIP의 Feed-Forward Network (FFN)를 제외한 파라미터를 공유하면서 세밀한 정보 캡처를 위한 여러 CLIP 모델을 생성합니다. 이 모델들은 Multistage Contrastive Learning (MCL)을 통해 다단계 클러스터링 및 미세 조정 과정을 거쳐 서로 다른 입력 정보의 측면을 캡처하는 FFN 전문가로 변환됩니다. 최종적으로 CLIP-MoE는 모든 전문가를 극대화하여 집합적이고 유용한 정보를 캡처하도록 합니다.

- **Performance Highlights**: DMU를 통해 초기화된 CLIP-MoE는 기존 OpenAI CLIP 모델에 비해 약 20%의 성능 향상을 보여주며, 본래 CLIP의 비전 인코더 역할을 대체할 수 있습니다. 다양한 다운스트림 작업에서 최고의 성과를 기록했으며, 적은 추가 훈련 비용으로 기존 CLIP의 성능을 크게 향상시켰습니다.



### FINE: Factorizing Knowledge for Initialization of Variable-sized Diffusion Models (https://arxiv.org/abs/2409.19289)
- **What's New**: 본 논문에서는 FINE이라고 하는 새로운 방법을 제안합니다. FINE은 Learngene 프레임워크를 기반으로 하여, 다운스트림 네트워크 초기화를 위한 효율적인 모델 초기화를 진행합니다. FINE은 사전 훈련된 모델을 활용하고, 모델 크기와 작업별 요구 사항을 모두 고려합니다.

- **Technical Details**: FINE은 사전 훈련된 지식을 행렬(즉, $U$, $	ext{Σ}$, $V$)의 곱으로 분해하여, $U$와 $V$는 네트워크 블록 간에 공유되며, $	ext{Σ}$는 레이어별로 특화되어 있습니다. 초기화 중에 FINE은 고정된 learngene 매개변수를 유지하면서 소규모 데이터 하위 집합을 사용하여 $	ext{Σ}$만 학습합니다.

- **Performance Highlights**: FINE은 다양한 모델 크기에 대해 직접 사전 훈련하는 것보다 지속적으로 더 나은 성능을 보이며, 특히 소형 모델의 경우에는 최첨단 성능을 달성합니다. FINE은 훈련 단계 수를 약 3N배 줄이고, 저장 용량을 5배 절감할 수 있습니다. 또한, 여러 다운스트림 데이터세트에 걸쳐 FID와 sFID에서 평균 성능 향상을 이뤄냅니다.



### PDCFNet: Enhancing Underwater Images through Pixel Difference Convolution (https://arxiv.org/abs/2409.19269)
- **What's New**: 이 연구에서는 전통적인 convolution 기법의 한계를 극복하기 위한 새로운 방법인 Pixel Difference Convolution (PDC)를 도입하였다. PDC는 이미지 내의 중요 변화가 있는 gradient 정보에 초점을 맞춰 물속 이미지에서의 텍스처와 세부 사항을 개선한다.

- **Technical Details**: PDCFNet이라는 네트워크는 PDC를 기반으로 하여 다양한 레벨의 feature fusion을 구현한다. 여기에는 고주파(high-frequency) 특징을 캡처하기 위해 병렬 PDC를 사용하는 디테일 향상 모듈이 포함된다. 또한, 서로 다른 레벨의 feature에 대해 결합 및 곱셈 연산을 수행해 상호작용을 극대화하는 cross-level feature fusion 모듈을 설계하였다.

- **Performance Highlights**: PDCFNet은 UIEB 데이터셋에서 PSNR 27.37과 SSIM 92.02를 달성하며, 현재까지의 최고의 성능을 기록하였다.



### DENEB: A Hallucination-Robust Automatic Evaluation Metric for Image Captioning (https://arxiv.org/abs/2409.19255)
Comments:
          ACCV 2024

- **What's New**: DENEB를 도입하여 이미지 캡션 생성에서의 환각(hallucination)에 대한 강건성을 향상시킨 새로운 자동 평가 지표를 제안

- **Technical Details**: DENEB는 Sim-Vec Transformer를 통합하여 동시에 여러 참조 캡션을 처리하며, Nebula 데이터셋에서 32,978개의 이미지와 805명의 주석가의 인간 평가를 통해 훈련됨

- **Performance Highlights**: DENEB는 FOIL, Composite, Flickr8K-Expert, Flickr8K-CF, PASCAL-50S 및 Nebula 데이터셋에서 기존 LLM-free 지표 중 최첨단 성과 달성



### Beyond Euclidean: Dual-Space Representation Learning for Weakly Supervised Video Violence Detection (https://arxiv.org/abs/2409.19252)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 이 연구에서는 Dual-Space Representation Learning (DSRL)이라는 새로운 방법을 제안하여 유클리드 공간과 쌍곡선 공간을 결합하여 약한 감독 하의 비디오 폭력 탐지 (VVD)를 개선하고 시각적으로 유사한 폭력 이벤트 (즉, 애매한 폭력)의 구분 능력을 향상시킵니다.

- **Technical Details**: DSRL은 Hyperbolic Energy-constrained Graph Convolutional Network (HE-GCN)와 Dual-Space Interaction (DSI) 모듈을 포함하여, 다층에 따라 동적으로 결정되는 하이퍼볼릭 연결 정도에 기초한 노드 선택 전략을 통해 정보 집합을 적용합니다. DSI는 유클리드 공간과 하이퍼볼릭 공간 간의 정보 상호작용을 촉진하여 더 나은 구별 기능을 포착합니다.

- **Performance Highlights**: 제안된 DSRL 방법은 XD-Violence 데이터셋에서 unimodal 및 multimodal 설정 모두에서 최첨단 성능을 달성하였으며, 애매한 폭력의 구별 능력을 향상시킴으로써 비교 가능한 다른 방법보다 두드러진 성과를 나타냅니다.



### TrojVLM: Backdoor Attack Against Vision Language Models (https://arxiv.org/abs/2409.19232)
Comments:
          ECCV 2024

- **What's New**: 이번 연구는 TrojVLM을 도입하며, 복잡한 이미지-텍스트 생성에 특정 목표 텍스트를 삽입하는 백도어 공격을 VLMs에 처음으로 탐구합니다. 이 과정에서 우리가 제안한 새로운 의미 유지 손실(semantic preserving loss)은 원래 이미지 콘텐츠의 의미 무결성을 보장합니다.

- **Technical Details**: TrojVLM은 오염된 이미지에서 특정 이미지 트리거를 감지할 때 목표 텍스트를 삽입하는 방식으로 설계되었습니다. 또한, 전통적인 토큰 수준의 언어 모델링 손실을 사용하여 자연어 관계를 유지하며 언어 모델을 미세 조정합니다. 이 연구에서는 이미지 캡셔닝과 시각적 질문 답변(VQA) 작업에서 평가됩니다.

- **Performance Highlights**: TrojVLM은 높은 공격 성공률을 달성하는 동시에, 원래의 의미적인 콘텐츠를 유지하며 상당한 성능을 보입니다. 실험 결과는 VLMs의 보안을 강화할 필요성을 강조합니다.



### GS-EVT: Cross-Modal Event Camera Tracking based on Gaussian Splatting (https://arxiv.org/abs/2409.19228)
- **What's New**: 이 논문은 이벤트 카메라(event camera)를 활용한 신뢰성 있는 자기 위치 추적(self-localization) 방법을 제안합니다. 특히, Gaussian Splatting을 기반으로 하여 기존의 프레임 기반 카메라로부터 직접 얻은 맵 표현을 사용하여 이벤트 카메라의 장점을 극대화합니다.

- **Technical Details**: 제안된 방법은 새로운 포즈 매개변수화(pose parametrization)를 도입하여, 참조 포즈(reference pose)와 1차 동역학(first-order dynamics)을 활용하여 로컬 차별 이미지(rendering) 렌더링을 수행합니다. 최적화는 비대칭의 coarse-to-fine 접근 방식으로 진행되며, 이 과정에서 이벤트 통합 이미지와 비교하여 카메라의 포즈 및 속도를 업데이트합니다.

- **Performance Highlights**: 실험 결과, Gaussian Splatting의 현실적인 뷰 렌더링 능력 덕분에 다양한 공공 데이터 및 새로 기록된 데이터 시퀀스에서 안정적이고 정확한 추적을 달성했습니다. 제안된 방법은 기존의 포토메트릭 방법이나 최근의 기하학적 정렬 프레임워크들보다 뛰어난 성능을 보여주었습니다.



### Summit Vitals: Multi-Camera and Multi-Signal Biosensing at High Altitudes (https://arxiv.org/abs/2409.19223)
Comments:
          Accepted by UIC'24, 8 pages, 5 figures. Ke Liu and Jiankai Tang are co-first authors. Yuntao Wang and Xiaojing Liu are co-corresponding authors

- **What's New**: 본 연구는 고산지대에서의 생리 신호 모니터링을 위한 새로운 SUMS 데이터셋을 소개합니다. 이 데이터셋은 운동 및 산소 회복 시나리오에서 촬영된 80개의 비디오를 포함하고 있으며, 얼굴 비디오(비접촉)와 손가락 비디오(접촉)가 밀리초 단위로 동기화되어 있습니다.

- **Technical Details**: SUMS 데이터셋은 10명의 참가자로부터 수집된 생리 신호 (PPG, 호흡률(RR), SpO2)를 포함하며, 다양한 생리학적 신호를 제공하여 고산 환경에서의 비접촉 생리 신호 측정을 가능하게 합니다. 얼굴 rPPG와 손가락 cPPG 알고리즘의 성능을 평가하고, 다중 카메라 비디오 융합을 통해 SpO2 예측의 평균 절대 오차(MAE)를 7.6% 및 10.6% 줄이는 성과를 보여줍니다.

- **Performance Highlights**: 우리의 연구에서 다중 지표(PPG 및 혈중 산소)의 동시 훈련은 SpO2 추정의 MAE를 17.8% 감소시켰으며, 심박수(HR) 추정에 대해 0.5 BPM 미만, SpO2 추정에 대해 2.5%의 MAE를 기록하며, 비접촉 생리 모니터링의 정확성을 입증하였습니다.



### Extending Depth of Field for Varifocal Multiview Images (https://arxiv.org/abs/2409.19220)
- **What's New**: 이 논문은 Optical imaging 시스템의 Depth of Field (DoF)를 확장하기 위한 새로운 방법을 제안합니다. 기존의 Multi-focus 이미지 대신 Varifocal Multiview 이미지라는 새로운 데이터 타입을 활용하여 EDoF 문제를 해결하고자 합니다.

- **Technical Details**: 제안된 방법은 이미지 정렬(image alignment), 이미지 최적화(image optimization) 및 이미지 융합(image fusion)을 포함한 End-to-End 접근 방식입니다. 이는 스태틱 씬(static scene)에 국한되지 않으며, 고정된 시야(fixed field of view)보다 넓은 정보를 포함합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 Varifocal Multiview 이미지를 통한 EDoF를 달성하는 데 있어서 높은 효율성을 보여주었습니다.



### 1st Place Solution to the 8th HANDS Workshop Challenge -- ARCTIC Track: 3DGS-based Bimanual Category-agnostic Interaction Reconstruction (https://arxiv.org/abs/2409.19215)
- **What's New**: 이 보고서는 ECCV 2024와 함께 진행된 제8회 HANDS 워크숍 챌린지의 1위 솔루션에 대해 설명합니다. 이 챌린지에서는 미리 정의된 템플릿 없이 단안 비디오에서 두 손과 객체의 3D 재구성을 생성하는 목표를 다루었습니다.

- **Technical Details**: 우리의 접근 방식은 두 단계로 나뉘며, 첫 번째 단계는 단일 훈련(Single Train)과 두 번째 단계는 공동 훈련(Joint Train)으로 구성됩니다. 우리는 Triplane-Net을 기반으로 하여 3D Gaussians를 초기화하고 이들을 특징적인 삼면 공간에 위치시키며, Deformation, Appearance 및 Geometry를 추정합니다. 추가적으로, Mask Loss와 3D Contact Loss를 도입하여 가려진 부분과 접촉 동역학 문제를 관리합니다.

- **Performance Highlights**: ARCTIC 테스트 세트에서 주요 메트릭인 CD$_h$에서 38.69의 값을 달성하며, HALO의 두 손 조작 환경에서의 성능을 크게 향상시켰습니다.



### Learning to Obstruct Few-Shot Image Classification over Restricted Classes (https://arxiv.org/abs/2409.19210)
Comments:
          ECCV 2024

- **What's New**: 이번 연구는 특정 다운스트림 작업을 위한 Fine-tuning이 어려운 사전 훈련된(backbone) 모델을 개발할 수 있는지를 탐구합니다. 우리는 학습을 방해하는(Learning to Obstruct, LTO) 알고리즘을 제안하며, 이는 여러 데이터셋에서 소수 샷 분류(Few-shot Classification, FSC) 방법의 성능을 저하시킵니다.

- **Technical Details**: LTO 알고리즘은 메타 학습(meta-learning) 기법을 활용하여, 제한된 클래스에 대해 '안 좋은 초기화(poor initialization)'를 학습하여 FSC 방법의 학습을 방해합니다. 이 연구는 ImageNet과 CIFAR100 데이터셋을 포함하여 여러 데이터셋에서 방법을 실험적으로 검증합니다.

- **Performance Highlights**: LTO를 적용한 결과, 제한된 클래스에서의 정확도가 떨어지면서도 나머지 클래스에서는 경쟁력을 유지하는 성능을 보여주었습니다. 이를 통해 LTO는 FSC 방법에 대해 성공적으로 방해 효과를 입증했습니다.



### A comprehensive review and new taxonomy on superpixel segmentation (https://arxiv.org/abs/2409.19179)
Comments:
          54 pages. This is the author version of the manuscript of the same name published in ACM Computing Surveys

- **What's New**: 이 논문은 superpixel segmentation에 대한 포괄적인 리뷰와 새로운 분류 체계를 제시하여 최근의 연구 동향을 반영합니다.

- **Technical Details**: 제안된 새로운 taxonomy에 따라, 본 연구는 20가지 전략을 연결성(connectivity), 압축성(compactness), 윤곽선(delineation), superpixel 수 조절(control over the number of superpixels), 색상 동질성(color homogeneity), 강인성(robustness), 실행 시간(running time), 안정성(stability), 시각적 품질(visual quality)의 9가지 기준에 따라 평가합니다.

- **Performance Highlights**: 실험 결과, 각 접근법이 픽셀 클러스터링에서의 경향성을 보여주며, 개별적인 trade-off에 대해 논의합니다. 또한, 새로운 superpixel 평가를 위한 benchmark를 제공하여 연구자들이 사용할 수 있도록 하고 있습니다.



### FLINT: Learning-based Flow Estimation and Temporal Interpolation for Scientific Ensemble Visualization (https://arxiv.org/abs/2409.19178)
Comments:
          18 pages (with Appendix), 17 figures

- **What's New**: FLINT는 2D+time 및 3D+time 과학적 앙상블 데이터의 흐름(Flow) 필드를 추정하는 새로운 딥러닝 기반 접근법입니다. FLINT는 일부 멤버의 흐름 필드가 부분적으로 이용 가능하거나 아예 없을 경우에도 유연하게 처리할 수 있습니다.

- **Technical Details**: FLINT는 두 가지 유형의 흐름, 즉 물리적 흐름(physical flow)과 광학 흐름(optical flow)을 고려합니다. 이 구조는 모듈화된 손실 함수(modular loss functions)를 조정하여 흐름 감독(flow-supervised) 및 비감독(flow-unsupervised) 문제로 각각 다뤄질 수 있도록 설계되었습니다. FLINT는 여러 신경망 블록(neural blocks)을 사용하며, 각 블록은 여러 개의 컨볼루션(convolutional) 및 디컨볼루션(deconvolutional) 레이어를 포함합니다.

- **Performance Highlights**: FLINT는 과학적 앙상블에서 흐름 추정을 수행하는 최초의 접근법으로, 원래 흐름 정보가 없는 경우에도 각 시간 단계에 해당하는 흐름 필드를 추정할 수 있습니다. FLINT는 고품질의 시간 보간값(temporal interpolants)을 생성하며, 2D 및 3D 앙상블을 처리할 수 있습니다.



### MASt3R-SfM: a Fully-Integrated Solution for Unconstrained Structure-from-Motion (https://arxiv.org/abs/2409.19152)
- **What's New**: 본 논문은 MASt3R-SfM이라는 새로운 구조-움직임(SfM) 파이프라인을 제안합니다. 이 방법은 어떤 제약없이 이미지 컬렉션을 처리할 수 있으며, 기존 문제의 복잡성을 줄입니다. 또한, 기존의 RANSAC 알고리즘을 완전히 제거하는 방식으로 보다 견고한 성능을 구현합니다.

- **Technical Details**: MASt3R-SfM은 최근 발표된 DUSt3R의 기반 모델을 활용하여 로컬 3D 재구성과 매칭을 단일 전방 패스로 수행합니다. 개선된 점은 MASt3R의 고정 인코더를 이용하여 이미지 검색을 빠르게 수행하고, 이는 이미지 수에 대해 준선형 복잡성을 제공합니다. 최적화 과정은 두 단계의 기울기 하강법을 통해 진행됩니다.

- **Performance Highlights**: 다양한 벤치마크에 대한 실험 결과, MASt3R-SfM은 작은 및 중간 규모 조건에서 기존 방법을 능가하는 안정적인 성능을 보여줍니다. 특히, 전통적인 SfM 방법이 겪는 문제를 훌륭하게 해결하며, 그러한 환경에서도 우수한 결과를 도출합니다.



### Multimodal Pragmatic Jailbreak on Text-to-image Models (https://arxiv.org/abs/2409.19149)
- **What's New**: 이번 연구는 텍스트-이미지(T2I) 모델에 대한 새로운 유형의 jailbreak을 소개하며, 둘 이상의 안전한 요소가 결합해 위험한 콘텐츠를 생성하는 현상을 탐구합니다.

- **Technical Details**: 제안된 Multimodal Pragmatic Unsafe Prompts (MPUP) 데이터셋은 1,200개의 위험한 프롬프트로 구성되어 있으며, 아홉 개의 대표적인 T2I 모델을 벤치마킹합니다. 실험 결과, 모든 모델이 8%에서 74%의 비율로 위험한 콘텐츠를 생성하는 취약성을 보였습니다.

- **Performance Highlights**: 현재의 안전 필터가 이러한 새로운 jailbreak에 대해 효과적이지 않음을 밝혀냈으며, 모델들이 제작한 위험한 콘텐츠의 복잡성을 감지하는 데 한계가 있음을 강조합니다.



### Bound Tightening Network for Robust Crowd Counting (https://arxiv.org/abs/2409.19146)
Comments:
          This work was done 2 years ago

- **What's New**: 본 논문에서는 Robust Crowd Counting을 위한 새로운 Bound Tightening Network (BTN)을 제안합니다. BTN은 기본 모델, 스무스 정규화 모듈 및 인증 경계 모듈의 세 부분으로 구성됩니다. 이 네트워크는 모델의 추정에 대한 이론적인 보증을 제공하고, 다양한 표준 데이터셋에서 모델의 강건성을 향상시킵니다.

- **Technical Details**: BTN은 3개의 모듈로 구성됩니다: 1) Smooth Regularization Module - 기본 모델의 레이어 가중치를 활용하여 정규화 항을 도입합니다; 2) Certify Bound Module - 입력 이미지에 대해 적대적 섭동을 수동으로 도입하고 초기 경계 세트를 구성하여 모델 레이어를 통해 이 경계를 전파합니다. 이를 통해 훈련 루프를 안내하여 최종 성능을 향상시킵니다.

- **Performance Highlights**: 여러 벤치마크 데이터셋을 활용한 실험에서 BTN의 효과성과 효율성을 입증하였습니다. BTN은 적대적 섭동에 대해 robust 하며 모델 예측의 매우 타이트한 경계를 제공합니다.



### Diverse Code Query Learning for Speech-Driven Facial Animation (https://arxiv.org/abs/2409.19143)
- **What's New**: 본 논문에서는 동일한 음성 신호에 조건화하여 다양한 샘플을 예측하고, 샘플 다양성을 명시적으로 촉진하여 얼굴 애니메이션 생성의 다양성을 해결하는 방법을 제안합니다.

- **Technical Details**: 기본적으로, 벡터 양자화 변변자동차(VQ-VAE)를 통한 얼굴 사전 모델을 구축하고, 이를 통해 높은 음성 신뢰성을 가진 다양한 3D 얼굴 동작 샘플을 생성하는 다채로운 코드 쿼리 메커니즘을 제안합니다. 또한, 얼굴 부분을 순차적으로 예측하여 각 부분을 조합함으로써 전체 얼굴 동작을 생성하도록 모델을 설계했습니다.

- **Performance Highlights**: 실험 결과, 저희 방법은 정량적 및 정성적 모두에서 기존의 최첨단 방법들보다 우수한 성능을 보여주었으며, 특히 샘플 다양성과 제어 가능한 생성 측면에서 뛰어난 결과를 달성했습니다.



### Pruning then Reweighting: Towards Data-Efficient Training of Diffusion Models (https://arxiv.org/abs/2409.19128)
Comments:
          Under Review. Code is available here (this https URL)

- **What's New**: 이 논문은 데이터 세트 프루닝(dataset pruning)의 관점에서 효율적인 확산 훈련(Diffusion Training)을 연구하는 첫 번째 작업입니다. 기존에는 GANs에 기반한 데이터 선택 방법을 확산 모델 훈련에 확장하여 훈련 데이터의 효율성을 높이고 있습니다.

- **Technical Details**: 이 연구에서 제안된 방식은 GAN 기반의 데이터 선택 방법을 사용하여 특징을 인코딩하고 점수를 매기는 방식을 검토합니다. 사전 훈련된 모델을 통해 데이터 특징을 추출하고, 이를 기반으로 각 데이터 포인트의 점수를 매겨 덜 관련된 데이터를 제거합니다. 또한, 클래스 가중치를 동적으로 최적화하는 클래스별 재가중치 전략(class-wise reweighting strategy)을 도입하여 생성 능력을 향상시키고 있습니다.

- **Performance Highlights**: CIFAR-10 데이터셋에서의 실험 결과, 제안된 방법은 기존 접근 방식에 비해 2.34배에서 8.32배의 속도 향상을 보여주며 원래의 전체 데이터 모델과 유사한 이미지 합성 성능을 유지하는 것으로 나타났습니다. 이는 픽셀 기반 DDPM과 잠재 기반 확산 모델에서도 수행이 가능합니다.



### Fusion is all you need: Face Fusion for Customized Identity-Preserving Image Synthesis (https://arxiv.org/abs/2409.19111)
- **What's New**: 이 논문은 얼굴 참조 이미지(reference face image)를 직접 통합하여 고정형 인코더(fixed encoders)나 정적 얼굴 임베딩(static face embeddings)에 의존하지 않고, 텍스트 프롬프트에 기초하여 고유한 아이덴티티(identity)를 유지하면서 높은 품질의 이미지를 생성할 수 있는 새로운 방법을 제안합니다.

- **Technical Details**: 이 방법은 Stable Diffusion의 사전 훈련된 UNet을 활용하여 멀티 스케일(multiscale)에서 참조 이미지를 처리하고, cross-attention 레이어를 혁신적으로 변경하여 얼굴 특징을 생성 과정에 결합합니다. 이를 통해 ID 유지를 강화하고, 효율적인 다중 참조 및 다중 아이덴티티 생성을 지원합니다.

- **Performance Highlights**: 본 연구는 여러 SOTA(identity-preserving techniques)와 비교하며 실험을 통해 제안된 방법의 효과성을 입증하였고, ID 유사성 측면에서 SOTA 성과를 달성하며, 프롬프트 정렬을 유지하면서 페이스 이미지 생성을 향상시켰습니다.



### Show and Guide: Instructional-Plan Grounded Vision and Language Mod (https://arxiv.org/abs/2409.19074)
Comments:
          Accepted at EMNLP 2024 Main Track

- **What's New**: MM-PlanLLM은 사용자가 지침 작업을 실행할 수 있도록 돕기 위해 텍스트 플랜과 비주얼 정보를 모두 활용하는 다중 모달 대형 언어 모델입니다. 이 모델은 복잡한 절차적 계획을 시각적으로 안내하기 위해 설계되었습니다.

- **Technical Details**: MM-PlanLLM은 사용자의 쿼리에 따라 관련 단계 비디오 세그먼트를 검색하는 'Conversational Video Moment Retrieval'과 사용자의 현재 진행 상황을 설명하는 이미지에 기반하여 계획의 다음 단계를 생성하는 'Visually-Informed Step Generation'을 통해 상호 모달성을 도입합니다. 이 모델 아키텍처는 특정 작업을 통해 비디오의 의미적 및 시간 정보를 캡처하고, 유연한 디코드 시간 다중 모달 검색을 지원하도록 설계되었습니다.

- **Performance Highlights**: MM-PlanLLM은 다중 모달 및 텍스트 대화에서 모두 강력한 성능을 발휘하며, 텍스트 전용 요청에 대한 성능 저하가 제한적입니다. 평가 결과는 텍스트 전용 작업에서 경쟁력 있는 성능을 보이며, 기존 접근 방법에 비해 다중 모달 작업에서 상당한 개선을 이룹니다.



### Multimodal Markup Document Models for Graphic Design Completion (https://arxiv.org/abs/2409.19051)
Comments:
          Project page: this https URL

- **What's New**: 이 논문은 interleaved multimodal documents 내에서 markup language와 이미지를 동시에 생성할 수 있는 Multimodal Markup Document Models(MarkupDM)을 제안합니다. 이 모델은 graphic design 작업에 필수적인 고유한 과제를 다룹니다.

- **Technical Details**: MarkupDM은 다양한 크기와 투명성을 가진 이미지 토큰화를 위한 이미지 양자화기(image quantizer)를 설계하고, markup language를 처리할 수 있는 코드 언어 모델을 수정하여 이미지 모달리티를 통합합니다. 장기적으로 166K graphic design templates로 훈련되었습니다.

- **Performance Highlights**: 세 가지 graphic design completion 작업에서 MarkupDM의 효과가 입증되었습니다. 생성된 디자인은 주어진 맥락에 일관되며, 다양한 디자인 대안을 탐색할 수 있게 해줍니다.



### Gaussian Heritage: 3D Digitization of Cultural Heritage with Integrated Object Segmentation (https://arxiv.org/abs/2409.19039)
- **What's New**: 이번 연구에서는 RGB 이미지만을 사용하여 3D 복제를 생성하고, 문화유산의 효율적인 분할(segmentation)을 가능하게 하는 새로운 파이프라인을 제안합니다. 이 방법은 전문가의 지식이 필요 없고, 일반 스마트폰으로 촬영한 이미지로도 활용할 수 있습니다.

- **Technical Details**: 제안된 방법은 3D Gaussian Splatting 모델을 사용하여 장면을 3D Gaussian 클라우드로 모델링합니다. 이 모델은 geometry, appearance, instance segmentation을 동시에 최적화하며, ViT-H SAM을 통해 생성된 2D 분할 마스크로부터 세그멘테이션 정보(feature vector)를 추출합니다. 세 가지 손실 함수(ℒ_rendering, ℒ_CC, ℒ_reg)를 통해 모델을 훈련시키며, cosine similarity를 활용하여 인스턴스 분할을 수행합니다.

- **Performance Highlights**: 본 연구의 방법은 LERF-Mask와 3D-OVS 데이터셋을 사용하여 평가되었으며, 기존 방법들에 비해 평균 intersection over union(mIoU) 및 평균 boundary intersection over union(mBIoU)에서 더 높은 정확성을 기록했습니다. 150장의 이미지를 사용하여 모델을 훈련시킨 결과, 장면에서 개별 객체에 대한 정교한 모델 추출이 가능함을 입증했습니다.



### Neural Product Importance Sampling via Warp Composition (https://arxiv.org/abs/2409.18974)
Comments:
          To appear in ACM SIGGRAPH Asia 2024 Conference Papers. Project page: this https URL

- **What's New**: 이번 연구에서는 Monte Carlo 샘플링을 통한 조명 통합 추정을 위한 새로운 학습 기반 방법을 제안합니다. 이 방법은 Normalizing Flows(NF)를 사용하여 조명 제품 통합을 효율적으로 중요 샘플링합니다.

- **Technical Details**: 제안된 샘플러는 flow head warp와 emitter tail warp를 조합합니다. 작은 conditional head warp는 neural spline flow로 표현되며, 큰 unconditional tail은 환경 맵에 따라 이산화됩니다. 조건부 차원이 낮을 경우, head warp도 이산화하여 성능을 더욱 향상시킬 수 있습니다.

- **Performance Highlights**: Mitsuba 3 렌더러에 통합되어 다양한 제품 샘플링 응용 프로그램과 장면 구성에서 분산을 줄이고 시각적 품질을 향상시킵니다. 제안된 방법은 기존의 샘플링 방식에 비해 경쟁력 있는 성능을 보여줍니다.



### Continuously Improving Mobile Manipulation with Autonomous Real-World RL (https://arxiv.org/abs/2409.20568)
Comments:
          CoRL 2024. Website at this https URL

- **What's New**: 우리는 사람의 감독이나 광범위한 장비 없이 정책을 학습할 수 있는 완전 자율 모바일 조작(Manipulation) RL(강화학습) 프레임워크를 제안합니다. 이 접근 방식은 작업 관련 자율성(task-relevant autonomy), 효율적인 정책 학습 및 보상을 형성하는 방식이 결합되어 있습니다.

- **Technical Details**: 우리의 방법은 Spot 로봇이 4가지 최첨단 모바일 조작 업무에 대해 평균 80% 성공률을 보이는 것을 입증했습니다. 이 방법은 기존 방법에 비해 3-4배 향상된 성능을 보여줍니다. 주요 구성 요소로는 이미지 인코더, 벡터 관측치, 객체 탐지 모델 등이 포함됩니다.

- **Performance Highlights**: 우리는 각 태스크에 대해 수행된 실험 결과를 소개합니다. 의자 이동, 쓰레기통 세우기, 쓰기 작업 등 다양한 작업에서 성공률 80%를 기록했습니다. 각 작업에서 특정 임무에 대한 보상이 계산되며, 이 보상의 변화로 인해 로봇의 성능이 기하급수적으로 향상됩니다.



### LaMMA-P: Generalizable Multi-Agent Long-Horizon Task Allocation and Planning with LM-Driven PDDL Planner (https://arxiv.org/abs/2409.20560)
Comments:
          Project website: this https URL

- **What's New**: 이 논문에서는 LaMMA-P라는 새로운 다중 에이전트 (Multi-Agent) 과업 계획 프레임워크를 제안합니다. 이는 언어 모델 (LMs)과 전통적인 탐색 계획기 (search planner)의 강점을 통합하여 긴 기간의 작업을 효과적으로 처리하는 방법을 제시합니다.

- **Technical Details**: LaMMA-P는 Planning Domain Definition Language (PDDL)와 대형 언어 모델 (LLMs)의 추론 능력을 결합하여 다중 로봇 시스템에서 긴 기간 과업 할당 및 실행을 용이하게 합니다. LLM의 구성 요소는 각 로봇의 기술을 기반으로 하여 하위 작업을 식별하고 할당하며, 각 로봇의 도메인에 대한 PDDL 문제 설명을 생성합니다. 또한 Fast Downward 계획기를 사용하여 초기 계획이 실패할 경우, LLM이 계획을 재생성하고 조정하여 실행 가능한 솔루션을 생성합니다.

- **Performance Highlights**: 실험 결과, LaMMA-P는 기존 LM 기반 다중 에이전트 계획기보다 105% 높은 성공률과 36% 높은 효율성을 보여주었습니다. 더불어 MAT-THOR라는 종합 벤치마크를 통해 다양한 복잡성의 가정 작업을 포함한 성능 평가를 수행했습니다.



### Supervised Multi-Modal Fission Learning (https://arxiv.org/abs/2409.20559)
- **What's New**: 이번 논문에서는 다중 모달(multimodal) 데이터셋에서의 학습을 위한 새로운 모델인 Multi-Modal Fission Learning (MMFL)을 제안했습니다. MMFL은 전통적인 방법들이 공유된 구성 요소(shared component)이나 개별 구성 요소(individual components)만 추출하는 데 그친 것에 반해, 전 세계적으로 공동이면서 부분적으로 공동 및 개별 구성 요소를 동시에 식별하는 방식을 사용합니다.

- **Technical Details**: MMFL은 반응 변수(response variable)의 감독(supervision)을 통해 예측 가능한 잠재 구성 요소(predictive latent components)를 식별합니다. 또한 불완전한 데이터 세트(incomplete multimodal data)를 통합하는 자연스러운 확장을 가지고 있습니다. 다양한 기존 모달 알고리즘과 비교하여 MMFL의 효율성을 실험적으로 입증했습니다.

- **Performance Highlights**: 시뮬레이션 연구를 통해 MMFL은 완전 및 불완전한 모달 설정에서 다양한 기존 다중 모달 알고리즘보다 뛰어난 성능을 보였습니다. 이 모델은 알츠하이머 병(Alzheimers Disease)의 조기 예측을 위해 ADNI(Alzheimers Disease Neuroimaging Initiative) 데이터셋을 사용한 실제 사례 연구에 적용되었으며, 기존 방법보다 더 정확한 예측과 모달 간의 상관관계에 대한 통찰을 제공했습니다.



### Scaling Proprioceptive-Visual Learning with Heterogeneous Pre-trained Transformers (https://arxiv.org/abs/2409.20537)
Comments:
          See the project website (this https URL) for code and videos

- **What's New**: 이 논문은 이질적인 (heterogeneous) 로봇 데이터에서 다수의 임무와 구현을 통해 정책 표현을 학습하는 방법을 제안합니다. 특히 Heterogeneous Pre-trained Transformers (HPT) 아키텍처를 통해 로봇 데이터의 다양한 하드웨어와 상황에 관계없이 일반화 가능한 정책을 학습할 수 있는 가능성을 보여줍니다.

- **Technical Details**: HPT 아키텍처는 로봇의 proprioception과 비전 입력을 단일 짧은 토큰 시퀀스로 병합하여 다양한 임무에 대해 로봇을 제어할 수 있도록 매핑합니다. 이 구조는 52개 데이터셋을 이용한 스케일링 실험을 통해 모델의 무게가 10억 개 이상의 파라미터에 달함을 확인하며, 다양한 모터 반응과 감지 시그널을 바탕으로 인간의 피드백 루프를 모방합니다.

- **Performance Highlights**: HPT는 여러 가지 기준선보다 성능이 우수하며, 여러 시뮬레이터 벤치마크와 실제 세계 환경에서 보지 못한 임무에 대해 20% 이상의 성능 향상을 보여줍니다. 이 연구는 다양한 데이터 요구 사항 없이도 새로운 임무와 구현에 대한 로봇 정책을 쉽게 구축할 수 있도록 하는 기초 모델의 중요성을 강조합니다.



### COLLAGE: Collaborative Human-Agent Interaction Generation using Hierarchical Latent Diffusion and Language Models (https://arxiv.org/abs/2409.20502)
Comments:
          9 pages, 6 figures

- **What's New**: 새로운 프레임워크 COLLAGE는 대규모 언어 모델(LLMs)과 계층적 모션 특정 벡터 양자화 변분 오토인코더(VQ-VAE)를 활용하여 협력적인 에이전트-오브젝트-에이전트 상호작용을 생성합니다. 이 접근법은 풍부한 데이터셋 부족 문제를 해결하고 LLM의 지식 및 추론 능력을 이용하여 생성적인 확산 모델을 안내합니다.

- **Technical Details**: COLLAGE 모델은 다단계의 추상화 수준에서 다양한 모션 특정 특성을 포착하는 계층적 VQ-VAE 구조를 사용합니다. 이 모델은 잠재 공간에서 작동하는 확산 모델과 LLM이 생성한 모션 계획 신호를 통합하여, 노이즈 제거 과정을 안내하고 결과적으로 명령어에 따라 특정 모션 생성을 가능하게 합니다.

- **Performance Highlights**: CORE-4D 및 InterHuman 데이터셋에 대한 실험 결과는 이 접근법이 기존의 최첨단 방법들을 초월하여 현실적이고 다양한 협력적 인간-객체-인간 상호작용을 생성하는 효과를 입증합니다. 이 연구는 로보틱스, 그래픽스 및 컴퓨터 비전과 같은 다양한 분야에서 복잡한 상호작용 모델링의 새로운 가능성을 열어줍니다.



### POMONAG: Pareto-Optimal Many-Objective Neural Architecture Generator (https://arxiv.org/abs/2409.20447)
- **What's New**: 이번 연구에서는 많은 목적(Many-Objective)를 고려한 Neural Architecture Generator(POMONAG)를 소개합니다. 이는 기존의 DiffusionNAG 방법을 확장하여 정확도뿐만 아니라 모델 복잡성, 계산 효율성 및 추론 지연과 같은 다양한 목표를 동시에 고려합니다.

- **Technical Details**: POMONAG는 성능 예측기(Performance Predictor) 모델을 통합하여 보다 정확한 성능 예측을 통해 생성하는 아키텍처의 품질을 향상시키며, 파레토 최적(Pareto-optimal) 아키텍처 생성을 지원합니다. POMONAG의 메타 데이터셋(Meta-Dataset)은 훈련 여건 개선을 위해 확장되었으며, 다수의 목적을 효과적으로 균형 있게 처리하기 위한 파레토 프론트 필터링(Pareto Front Filtering) 및 스트레칭(Stretching) 기법이 적용되었습니다.

- **Performance Highlights**: POMONAG는 NASBench201 및 MobileNetV3에서 실험을 수행하여 기존 최고의 성능을 초과하는 결과를 보여주었습니다. 특히, 다양한 이미지 분류 데이터셋에서 높은 정확도를 제공하면서도 요구되는 훈련 모델 수를 크게 줄임으로써 효율성을 증명했습니다.



### HELPD: Mitigating Hallucination of LVLMs by Hierarchical Feedback Learning with Vision-enhanced Penalty Decoding (https://arxiv.org/abs/2409.20429)
Comments:
          Accepted at Main Conference of EMNLP 2024

- **What's New**: 대규모 비전-언어 모델(LVLMs)의 다중 모달 환각(multimodal hallucination) 문제를 해결하기 위한 새로운 접근 방식인 계층적 피드백 학습(Hierarchical Feedback Learning) 방식인 HELPD를 제안합니다. 이 프레임워크는 객체 및 문장 의미 수준에서 환각 피드백을 통합하여 모델이 생성하는 내용과 이미지 간의 불일치를 줄여줍니다.

- **Technical Details**: HELPD는 환각을 감지하기 위해 객체 집합과 샘플링된 문장을 비교하여 객체 수준 피드백을 생성하며, GPT-4의 강력한 few-shot 추론 능력을 활용하여 문장 수준 피드백을 수행합니다. 또한, Vision-Enhanced Penalty Decoding 방식을 통해 시각 입력의 중요한 영향을 반영하여 최종 로짓(logits) 계산 시 시각 입력에 더 많은 비중을 두도록 합니다.

- **Performance Highlights**: 실험 결과, HELPD는 다양한 환각 벤치마크에서 15% 이상의 환각 완화를 달성하며, LVLM의 텍스트 생성 품질을 동시에 향상시키는 긍정적인 결과를 보여주었습니다.



### KANDU-Net:A Dual-Channel U-Net with KAN for Medical Image Segmentation (https://arxiv.org/abs/2409.20414)
- **What's New**: 이 논문은 U-Net 모델과 KAN 네트워크를 통합한 새로운 아키텍처를 소개합니다. KAN 네트워크의 비선형 표현 능력을 활용하여 U-Net의 강점을 더욱 강화한 것입니다.

- **Technical Details**: 우리는 KAN-컨볼루션(KAN-convolution) 이중 채널 구조를 도입하여 모델이 지역(local) 및 전역(global) 특징을 효과적으로 포착할 수 있게 합니다. KAN이 추출한 특징과 컨볼루션(convolution) 레이어를 통해 얻은 특징을 융합하는 효과적인 방법을 탐색하며, 이러한 융합 과정을 지원하기 위한 보조 네트워크(auxiliary network)를 활용합니다.

- **Performance Highlights**: 다수의 데이터셋에서 수행된 실험 결과, 모델의 정확도에서 우수한 성능을 보였으며, KAN-컨볼루션 이중 채널 접근 방식이 의료 이미지 분할(medical image segmentation) 작업에서 큰 잠재력을 가지고 있음을 나타냅니다.



### Efficient Driving Behavior Narration and Reasoning on Edge Device Using Large Language Models (https://arxiv.org/abs/2409.20364)
Comments:
          Submitted for possible journal publication

- **What's New**: 본 논문에서는 자율주행 기술의 발전을 위해 대규모 언어 모델(LLMs)과 엣지 컴퓨팅(edge computing)을 통합한 새로운 프레임워크를 제안합니다. 이 프레임워크는 도로변 장치(RSU)에서 LLMs를 배포하여 주행 행동을 설명하고, 사고 감지 시 신속하게 정보를 전달할 수 있는 구조를 가지고 있습니다.

- **Technical Details**: 제안된 프레임워크는 5G NR/NSA 네트워크를 통해 연결된 다수의 RSU로 구성되어 있으며, 각 RSU는 자신이 관할하는 지역의 교통 데이터를 수집하고 처리합니다. LLMs는 주행 행동을 분석하여 자연어 설명을 생성하며, 청각적 신호와 환경 정보를 통합한 3중 프롬프트 전략을 사용하여 성능을 향상시킵니다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식은 0.5초 이하의 프레임 응답 시간을 달성하였으며, 네 가지 LLM의 서사 정확도는 70% 이상, 가장 높은 추론 정확도는 81.7%에 달했습니다. 이는 자율주행 시나리오에서 처리를 최적화하고, 명확한 커뮤니케이션을 가능하게 함으로써 시스템의 안정성과 적시성을 높입니다.



### AI generated annotations for Breast, Brain, Liver, Lungs and Prostate cancer collections in National Cancer Institute Imaging Data Commons (https://arxiv.org/abs/2409.20342)
- **What's New**: 이번 프로젝트는 National Cancer Institute(NCI)의 Image Data Commons(IDC)를 향상시키기 위해 nnU-Net 모델을 개발하고, 암 방사선 이미지를 위한 AI-assisted segmentation을 제공합니다.

- **Technical Details**: 프로젝트는 다양한 이미징(Imagining) 모달리티, 즉 CT(computed tomography)와 MRI(magnetic resonance imaging) 이미지를 포함한 11개의 IDC 컬렉션을 기반으로 고품질 AI 주석 이미지 데이터셋을 만들었습니다. nnU-Net 모델은 오픈 소스 데이터셋을 사용해 훈련되었으며, AI 생성 주석의 일부는 방사선과 전문의에 의해 검토 및 수정되었습니다.

- **Performance Highlights**: 모든 모델, 이미지 및 주석은 공개적으로 접근 가능하여 암 이미징 분야에서의 추가 연구 및 개발을 촉진하며, DICOM(Digital Imaging and Communications in Medicine) 표준에 맞춰 AI 및 방사선 전문의 주석이 인코딩되었습니다.



### Enhancing GANs with Contrastive Learning-Based Multistage Progressive Finetuning SNN and RL-Based External Optimization (https://arxiv.org/abs/2409.20340)
- **What's New**: 본 연구에서 제안한 프레임워크는 멀티스테이지 프로그레시브 파인튜닝 시아미즈 신경망(MFT-SNN)과 강화 학습 기반 외부 최적화기(RL-EO)를 포함하여 GAN 훈련 루프 내에서 지침을 제공함으로써 GAN의 한계를 극복하는 것을 목표로 하고 있습니다.

- **Technical Details**: 이 프레임워크는 두 가지 구성 요소로 이루어져 있습니다. 첫째, MFT-SNN은 조직병리학 패치 간의 유사성을 평가하기 위한 대조 학습 기반의 신경망입니다. 둘째, RL-EO는 GAN 훈련 루프 내에서 보상 신호 생성기로 작용하며, 수정된 판별기 손실 함수는 가중 보상을 포함하여 GAN이 보상을 극대화하면서 손실을 최소화하도록 유도합니다.

- **Performance Highlights**: 제안한 방법은 최신 GAN과 Denoising Diffusion Probabilistic 모델에 대한 벤치마크에서 FID 점수, KID 점수, 지각 경로 길이(Perceptual Path Length), 하류 분류 작업 측면에서 이전의 최첨단(SOTA)을 초월하는 성과를 보여주었습니다.



### Devil is in Details: Locality-Aware 3D Abdominal CT Volume Generation for Self-Supervised Organ Segmentation (https://arxiv.org/abs/2409.20332)
- **What's New**: 본 논문에서는 Locality-Aware Diffusion (Lad)라는 새로운 방법론을 소개합니다. 이는 정밀한 3D 복부 CT 볼륨 생성을 위한 접근법으로, 중요 해부학적 영역을 개선하기 위해 지역 로스를 설계하고 복부의 사전 정보를 생성을 통합할 수 있는 조건 추출기를 개발하였습니다.

- **Technical Details**: Lad 방법론은 세 가지 단계로 구성되어 있습니다: Latent Space Construction, Diffusion Fitting in Latent Space, Sampling in Latent Space. 각 단계에서는 VQ-GAN을 사용하여 CT 볼륨의 잠재 공간을 구성하고, Diffusion 모델을 사용하여 사전 정보에 의한 적합화 및 지역 정보를 효과적으로 추출하며, 최종적으로 지역 조건 증강을 통해 대량의 고품질 복부 CT 볼륨을 생성합니다.

- **Performance Highlights**: 우리의 방법으로 생성된 볼륨은 AbdomenCT-1K 데이터셋에서 FID 점수를 0.0034에서 0.0002로 줄이며, 실제 데이터와 밀접하게 일치하고, 자가 감독 기반 장기 분할 작업에서 Dice 점수가 개선되었습니다. 이러한 결과는 의료 이미지 분석에서 자가 감독 학습의 발전을 위한 합성 데이터의 잠재력을 강조합니다.



### Distributed NeRF Learning for Collaborative Multi-Robot Perception (https://arxiv.org/abs/2409.20289)
- **What's New**: 본 논문에서는 여러 로봇 에이전트가 협력하여 환경을 인식하는 다중 에이전트 퍼셉션 시스템을 제안합니다. 이 시스템은 RGB 이미지로부터 NeRF(Neural Radiance Field)를 학습하여 장면을 표현하며, 각 에이전트는 자신의 로컬 데이터를 처리하고, 학습된 NeRF 모델만을 공유함으로써 통신 오버헤드를 줄입니다.

- **Technical Details**: 제안된 방법은 각 에이전트가 로컬 센서를 사용해 수집한 데이터를 바탕으로, 중앙 서버에 모든 원시 데이터를 전송하지 않고 에이전트 간의 네트워크 가중치만을 공유합니다. 이를 통해 NeRF 오버피팅(overfitting)을 줄이면서도 여러 에이전트 간의 일관성을 보장하는 분산 학습 프레임워크를 구축합니다.

- **Performance Highlights**: 우리의 실험 결과에 따르면, 제안된 다중 에이전트 접근 방식은 제한된 시점에서 입력이 제공될 때 중앙 집중식 훈련보다 뛰어난 성능을 보였으며, 환경을 매핑하는 데 있어 중앙화된 처리 방식과 비슷한 성과를 달성했습니다. 또한, 통신 효율성이 크게 향상되었습니다.



### Leveraging CAM Algorithms for Explaining Medical Semantic Segmentation (https://arxiv.org/abs/2409.20287)
Comments:
          Accepted for publication at the Journal of Machine Learning for Biomedical Imaging (MELBA) this https URL

- **What's New**: 이번 연구에서는 CNN의 해석 가능한 인공지능(xAI) 분야에서, Seg-HiRes-Grad CAM이라는 새로운 방법을 제안합니다. 이 방법은 현재의 분류 기반 해석 방법과 분할 기반 방법 간의 전이를 통해, 의학적 이미지 분할 등의 작업에서 더욱 상세하고 일관된 해석 결과를 제공합니다.

- **Technical Details**: Seg-HiRes-Grad CAM은 Seg-Grad CAM의 확장 버전으로, 분류 기반의 HiRes CAM으로부터 전이된 접근 방식입니다. 이 방법은 기울기(gradient)를 활용한 로컬(시각적) 설명 알고리즘이며, 세그먼트 분할 작업에 최적화되어 있습니다.

- **Performance Highlights**: Seg-HiRes-Grad CAM은 기존의 Seg-Grad CAM에서 발생하는 정확도 문제를 해결하며, 특히 의학적 이미지 분할 작업에서 설명 가능성을 크게 향상시킵니다. 이로 인해 의사 결정 과정에서의 중요한 이미지 영역을 더욱 명확하게 시각화할 수 있습니다.



### Survival Prediction in Lung Cancer through Multi-Modal Representation Learning (https://arxiv.org/abs/2409.20179)
Comments:
          Accepted in WACV 2025

- **What's New**: 본 논문은 CT, PET 스캔 및 유전체(genomic) 데이터를 통합하여 생존 예측을 위한 새로운 접근 방식을 제안합니다. 기존 방법들은 단일 모달리티(single modality) 또는 여러 모달리티의 통합에 의존하였으나, 환자 간 및 모달리티 간의 연관성을 충분히 다루지 않았습니다. 저자는 다중 모달리티 이미징 데이터와 유전자 정보를 통합하여 생존 예측 모델을 개발하며, 특히 환자 간의 연관성을 고려하고 있습니다.

- **Technical Details**: 저자들은 자기 지도 학습(self-supervised learning)을 통해 각 모달리티의 표현을 학습하고, 환자 간의 의미론적 유사성을 활용하여 임베딩(embedding)을 밀접하게 정렬합니다. 그러나 단순한 전역 관련성(global relevance) 최적화는 불충분하며, 유사한 고급 의미(high-level semantics)를 공유하는 쌍들이 임베딩 공간에서 서로 멀어지는 현상을 해결하기 위해 교차 환자 모듈(cross-patient module, CPM)을 사용합니다. CPM 모듈은 유사한 질병 특성을 가진 환자들의 임베딩을 결합하는 데 초점을 맞춥니다.

- **Performance Highlights**: 저자들은 비소세포 폐암(NSCLC) 환자 데이터를 실험적으로 평가한 결과, 제안한 접근 방식이 생존 예측에서 최신 기술(state-of-the-art) 방법들을 초 outperforming 하였음을 보여주었습니다. 이 모델은 특히 각기 다른 유전자 정보가 없는 환자에 대해서도 안정적인 성능을 유지합니다.



### ILeSiA: Interactive Learning of Situational Awareness from Camera Inpu (https://arxiv.org/abs/2409.20173)
Comments:
          7 pages, 8 figures

- **What's New**: 이 논문은 로봇에게 상황 인식을 가르치는 방법인 ILeSiA 시스템을 제안합니다. 이 시스템은 초기 기술 시연을 통해 로봇의 스킬을 학습하고, 사용자로부터 제공된 라벨(안전 또는 위험)을 사용하여 자율적으로 실행하는 과정에서 위험을 감지합니다.

- **Technical Details**: ILeSiA는 카메라 이미지를 통해 위험을 인식하며, 이미지를 저차원 잠재 공간(latent space)으로 인코딩하고, 이를 기반으로 분류기를 교육합니다. 이 과정에서 Gaussian Process (GP) 리스크 추정 모델을 사용하여 단일 시연으로도 위험 수준을 지속적으로 평가합니다. 시스템은 기존의 Learning from Demonstration (LfD) 프레임워크에 통합되어 있으며, 사용자 피드백을 통해 지속적으로 학습하고 모델을 재훈련할 수 있습니다.

- **Performance Highlights**: 실험 결과, 학습된 분류기는 사용자 제공 데이터가 적음에도 불구하고 다양한 위험을 성공적으로 감지할 수 있음을 보여줍니다. 이 시스템은 위험 사례가 라벨링됨에 따라 유연하게 동작하며, 인간 감독자가 위험을 식별함에 따라 즉시 라벨을 추가할 수 있는 장점을 갖고 있습니다.



### Characterizing Model Robustness via Natural Input Gradients (https://arxiv.org/abs/2409.20139)
Comments:
          28 pages; 14 figures; 9 tables; to be published in ECCV 2024

- **What's New**: 이 연구는 Adversarial Training (적대적 훈련) 대신에 입력의 Gradient Norm (그래디언트 노름)을 규제하는 것이 자연 샘플에서 모델의 견고성을 향상시킬 수 있음을 보이며, 특히 현대 비전 변환기 아키텍처에서 효과적이라는 점이 주목할 만하다.

- **Technical Details**: Gradient Norm 규제가 활성 함수의 부드러움에 따라 성능이 달라지며, 기본적인 폭격(perturbation)에 대해 모델의 입력 그래디언트를 집중시킴으로써 모델의 견고성을 크게 향상시킬 수 있다는 분석을 포함한다. 이 연구는 이미지 엣지를 강조하여 그래디언트를 규제하는 것이 견고성에 미치는 영향을 탐구한다.

- **Performance Highlights**: Gradient Norm 훈련 방식은 ImageNet-1K에서 90% 이상의 성능을 달성하며, 최신 PGD-3 Adversarial Training의 60%에 해당하는 연산비용으로 결과를 도출한다. 이를 통해 복잡한 적대적 최적화 없이도 상당한 견고성을 확보할 수 있음을 시사한다.



### A Self-attention Residual Convolutional Neural Network for Health Condition Classification of Cow Teat Images (https://arxiv.org/abs/2409.19963)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2409.18797

- **What's New**: 이번 연구에서는 소의 젖꼭지 건강 평가를 위한 새로운 모델인 Cows' Teats Self-Attention Residual Convolutional Neural Network (CTSAR-CNN)을 제안합니다. 이 모델은 기존의 CNN 모델에서 잔여 연결(residual connectivity)과 자기 주의(self-attention) 메커니즘을 결합하여 소의 젖꼭지 스스로를 평가할 수 있도록 합니다.

- **Technical Details**: CTSAR-CNN 모델은 고급 컴퓨터 비전 기술을 활용하여 젖꼭지의 과다 각화(hyperkeratosis) 정도를 분류합니다. 복잡한 환경에서도 소의 젖꼭지를 정확하게 인식하며, 다양한 자세와 위치에서의 데이터 해석을 가능하게 합니다. 이 모델은 잔여 연결과 자기 주의 메커니즘을 통해 정확성을 향상시킵니다.

- **Performance Highlights**: CTSAR-CNN의 통합으로 인해 기존의 접근 방식에 비해 젖꼭지 건강 평가의 정확성이 향상되었습니다. 연구 결과, 이 모델은 수의사들이 젖꼭지 건강을 더욱 신속하고 일관되게 평가하는 데 유용하며, 궁극적으로 유제품 산업에 기여할 수 있음을 보여줍니다.



### Law of the Weakest Link: Cross Capabilities of Large Language Models (https://arxiv.org/abs/2409.19951)
Comments:
          Code: this https URL

- **What's New**: 이번 연구에서는 Large Language Models (LLMs)의 평가에서 개별 능력에 중점을 두었던 기존 접근에서 벗어나, 여러 전문 분야의 교차 능력(cross capabilities) 간의 상호작용을 탐구합니다.

- **Technical Details**: 연구팀은 7개의 핵심 개별 능력을 정의하고, 이를 엮어 7개의 일반적인 교차 능력을 생성하였습니다. 이 과정에서 수동으로 구축된 분류 체계(taxonomy)를 지원하였으며, 1,400개의 인간 주석이 달린 프롬프트로 구성된 CrossEval이라는 벤치마크를 도입하였습니다. 각 개별 및 교차 능력마다 100개의 프롬프트가 배정되며, 4,200개의 모델 응답을 평가하여 8,400개의 인간 평가를 수집하였습니다.

- **Performance Highlights**: 연구 결과, 현재 LLM들은 교차 능력 성능이 가장 약한 구성 요소에 의해 강하게 제약된다는 "Law of the Weakest Link" 현상을 보였습니다. 17개 모델의 58개 교차 능력 점수 중 38개는 모든 개별 능력보다 낮았으며, 20개는 강한 능력과 약한 능력 사이에 위치하였습니다. 이 결과는 LLM들이 교차 능력 과제에서 저조한 성과를 내고 있음을 보여주며, 복잡한 다차원 상황에서 성능을 최적화하기 위해 미래 연구의 우선 과제가 약한 능력의 식별 및 개선이라는 점을 강조합니다.



### JaPOC: Japanese Post-OCR Correction Benchmark using Vouchers (https://arxiv.org/abs/2409.19948)
Comments:
          Accepted to PRICAI 2024

- **What's New**: 본 연구는 일본어 영수증에 대한 OCR(Optical Character Recognition) 오류 수정 방법의 벤치마크를 구축하고 효과성을 평가합니다. 이는 기존의 연구에서 다루어지지 않았던 일본어 OCR 오류 수정의 공개 가능 벤치마크를 제공합니다.

- **Technical Details**: 이 연구에서는 일본어 영수증에 특화된 OCR 오류 수정 벤치마크 JaPOC를 제안하고, T5와 같은 언어 모델을 활용하여 오류 수정 방법의 성능을 평가하였습니다. OCR 오류 수정 작업은 시퀀스-투-시퀀스 변환으로 정의되며, OCR 결과에 대해 고급 언어 모델로 미세 조정(fine-tuning)을 진행하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 오류 수정 알고리즘이 전반적인 인식 정확도를 크게 향상시켰습니다. Robota API와 Vision API를 사용하여 구축된 세트에서 상당한 정확도 향상이 확인되었습니다.



### One Shot GANs for Long Tail Problem in Skin Lesion Dataset using novel content space assessment metric (https://arxiv.org/abs/2409.19945)
Comments:
          15 pages, 6 Figures, 9 Tables and additional 6 Tables in Ancillary Data

- **What's New**: 이 논문에서는 HAM10000 데이터셋의 긴 꼬리 문제를 해결하기 위해 One Shot GANs 모델을 이용하여 소수 클래스의 데이터를 증강합니다. 이는 의료 영상 분류의 정확성을 높이고자 하는 새로운 접근 방식입니다.

- **Technical Details**: One Shot GANs는 단일 훈련 이미지를 사용하여 여러 샘플을 생성하는 생성적 적대 신경망(GAN)의 일종입니다. 이 모델은 이미지의 맥락과 레이아웃을 별도로 판별하는 이중 분기(discriminative) 구조를 가지고 있습니다. 또한, 훈련 데이터셋의 이미지를 최적 선택하는 informed subset selection 기법을 통해 전반적인 정확성을 높입니다.

- **Performance Highlights**: One-Shot GANs를 활용하여 소수 클래스에서 현저한 정확도 향상을 보였으며, 이는 WGANs와 비교해도 뚜렷한 개선을 나타냅니다. 새로 고안한 content-space assessment 메트릭 또한 FID 점수보다 더 나은 분류 정확도를 달성하는 데 기여했습니다.



### Positive-Sum Fairness: Leveraging Demographic Attributes to Achieve Fair AI Outcomes Without Sacrificing Group Gains (https://arxiv.org/abs/2409.19940)
- **What's New**: 이번 연구에서는 의료 AI의 공정성에 대한 새로운 개념인 positive-sum fairness를 도입하였습니다. 이는 성능의 향상이 집단 간 격차를 확대하더라도, 특정 하위 그룹의 성능 저하가 없다면 수용할 수 있다는 것입니다.

- **Technical Details**: positive-sum fairness는 집단 간 성능 차이를 해로운 것과 무해한 것으로 구분하는 평가 프레임워크입니다. 이 프레임워크를 통해 모델의 성능 향상 과정에서의 공정성을 분석하고, 모든 하위 집단이 더 낫지 않더라도 전체 성능이 향상될 수 있도록 하는 솔루션을 찾고자 합니다.

- **Performance Highlights**: CNN 모델을 비교한 결과, 인구 통계적 인코딩을 제거하면 하위 그룹 간 성능 차이를 줄일 수 있었으며, 인종 속성을 모델 입력으로 활용했을 때 전체 성능은 증가하였지만 하위 그룹 간 격차가 확대됨을 보였습니다. 이는 긍정적 공정성 개념의 관점에서 유익한 성능 개선을 달성할 수 있음을 보여줍니다.



### Learning Multimodal Latent Generative Models with Energy-Based Prior (https://arxiv.org/abs/2409.19862)
Comments:
          The 18th European Conference on Computer Vision ECCV 2024

- **What's New**: 이번 논문에서는 에너지 기반 모델(EBM)과 다중 모달(latent generative) 생성 모델을 통합하는 새로운 프레임워크를 제안합니다. 기존의 Gaussian나 Laplacian 분포를 넘어서, 다양한 데이터 타입의 정보를 효과적으로 캡처할 수 있는 접근 방식을 제공합니다.

- **Technical Details**: 제안된 프레임워크는 variational scheme을 통해 다중 모달 생성 모델과 EBM을 공동으로 훈련할 수 있게 합니다. 이 접근 방식은 보다 표현력 있고 정보가 풍부한 prior를 생성하여 다중 모달 간의 정보를 더 잘 캡처합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델이 기존 모델보다 더 나은 생성 일관성을 보여주었으며, 다중 모달 생성 모델의 가능성을 확장하는 데 기여합니다.



### Benchmarking Adaptive Intelligence and Computer Vision on Human-Robot Collaboration (https://arxiv.org/abs/2409.19856)
Comments:
          7 Pages, 9 Figures. 14 References. Submitted to IEEE RA-L Journal and ICRA 2025 Conference. This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 이 논문은 Industry 4.0에서 인간-로봇 협력(Human-Robot Collaboration, HRC)의 중요성을 강조하고, 새로운 환경에 적응하기 어려운 Concept Drift 문제를 해결하기 위해 Adaptive Intelligence와 Self-Labeling (SLB)을 통합한 방법론을 제안합니다.

- **Technical Details**: 이 연구에서 제안하는 방법론은 카메라와 무게 센서를 이용한 데이터 수집으로 시작되며, 이후 의도와 상태 변화를 주석 처리합니다. 다양한 딥러닝(Deep Learning) 모델을 훈련시키며, 사전 처리(preprocessing) 기술을 활용해 의도를 인식하고 예측합니다. 또한, SLB의 정확성을 높이기 위한 맞춤형 상태 감지 알고리즘을 개발하였으며, 이는 의도 레이블에 대해 정확한 상태 변화 정의와 타임스탬프를 제공합니다.

- **Performance Highlights**: 연구 결과, 스켈레탈 포즈(skeletal posture) 전처리(preprocessing)를 적용한 MViT2 모델이 83%의 정확도를 달성하였고, 이는 스켈레톤 포즈 추출 없이 얻은 79%의 정확도보다 향상된 것입니다. 또한, SLB 메커니즘은 91%의 레이블링 정확성을 달성하여 수동 주석에 필요한 시간을 획기적으로 줄였습니다. 궁극적으로, 본 연구는 다양한 자기 레이블링(Self-Labeled) 데이터 세트를 이용한 모델 성능 향상을 통해 Concept Drift 문제를 해결하고, 제조업에서 지능형 협업 로봇(cobots)의 빠른 배포 가능성을 보여줍니다.



### Investigating the Effect of Network Pruning on Performance and Interpretability (https://arxiv.org/abs/2409.19727)
Comments:
          4 pages, 6 figures

- **What's New**: 이번 연구에서는 GoogLeNet에 대한 다양한 pruning 기법(비구조화, 구조화, 연결 희소성)의 영향을 분석하고, 모델의 분류 성능 및 해석 가능성에 대한 결과를 제시하고 있습니다. 특히, 연결 희소성(Connection Sparsity) 방법을 통해 모델의 80%를 pruning하였음에도 불구하고 Top-1 정확도를 향상시키는 성과를 거두었습니다.

- **Technical Details**: 연구에서는 unstructured pruning, structured pruning, connection sparsity 같은 다양한 pruning 기법을 적용해 GoogLeNet의 성능을 평가하였습니다. 이 과정에서 iterative pruning과 one-shot pruning의 장단점을 비교하였으며, Connection Sparsity 방법을 통해 입력 채널을 pruning하여 모델의 구조적 무결성을 유지하면서 연산 속도를 최적화하는 방식으로 이루어졌습니다. 또한, Mechanistic Interpretability Score (MIS)를 사용하여 해석 가능성을 측정했습니다.

- **Performance Highlights**: 연구 결과, Connection Sparsity 방법을 적용한 GoogLeNet은 80%의 파라미터를 pruning한 후에도 Top-1 정확도가 0.2% 향상되었습니다. iterative pruning이 one-shot pruning보다 성능 유지에 더 유리하다는 결과를 바탕으로, retraining이 필요했으며, 50 epoch 이상의 retraining이 필요함을 보여주었습니다. 최종적으로, 구조화된 형태의 pruning 기법이 정확도 보존에 효과적임을 입증하였습니다.



### Hyperspectral Unmixing of Agricultural Images taken from UAV Using Adapted U-Net Architectur (https://arxiv.org/abs/2409.19701)
- **What's New**: 이 논문에서는 UAV(무인 항공기)에 장착된 하이퍼스펙트럴 카메라로 수집한 블루베리 농장 데이터를 바탕으로 하이퍼스펙트럴 언믹싱(hyperspectral unmixing) 데이터셋을 생성하였습니다. 또한 U-Net 네트워크 아키텍처를 기반으로 한 하이퍼스펙트럴 언믹싱 알고리즘을 제안하여, 기존 및 새로 생성된 하이퍼스펙트럴 언믹싱 데이터셋에서 더 정확한 언믹싱 결과를 달성하고자 하였습니다.

- **Technical Details**: 하이퍼스펙트럴 언믹싱은 각 하이퍼스펙트럴 픽셀에서 재료(보통 endmember라고 함) 데이터를 추출하고 그 분포를 계산하는 알고리즘입니다. 이 논문에서는 VCA(Vertices Component Analysis) 알고리즘을 사용하여 클래스 변화를 추출하고, 3개의 하이퍼스펙트럴 큐브 데이터를 사용하여 학습(train), 테스트(test), 검증(validation) 데이터셋을 생성하였습니다. 각 큐브는 1024 픽셀의 너비와 224 개의 스펙트럼 밴드를 갖고 있습니다.

- **Performance Highlights**: 최고의 분류 정확도는 VCA 알고리즘을 활용하여 1.5 σ의 변동 임계값을 유지했을 때 달성되었습니다. 모델이 예측한 endmembers가 클래스 변동 안에 포함되는지를 체크하여 알고리즘의 정확성을 높였습니다. 수집된 데이터셋을 통해 하이퍼스펙트럴 데이터의 분리가 가능하며, 농업분야 내 다양한 응용이 기대됩니다.



### Vision-Language Models are Strong Noisy Label Detectors (https://arxiv.org/abs/2409.19696)
Comments:
          Accepted at NeurIPS 2024

- **What's New**: 본 논문은 Denoising Fine-Tuning (DeFT) 프레임워크를 제안하여 비전-언어 모델을 조정하는 방법을 소개합니다. 이 방법은 수천만 개의 보조 이미지-텍스트 쌍에서 미리 학습된 텍스트 및 시각적 피처의 강력한 정렬을 활용하여 노이즈가 있는 레이블을 제거하는 데 중점을 둡니다.

- **Technical Details**: DeFT는 각 클래스에 대해 긍정적 및 부정적 텍스트 프롬프트를 학습하여 노이즈 레이블 탐지기를 구축합니다. 긍정적 프롬프트는 클래스의 독특한 특징을 드러내고, 부정적 프롬프트는 깨끗한 샘플과 노이즈 샘플을 구분하는 학습 가능한 임계값 역할을 합니다. 이를 통해 모델은 이미지와 텍스트 프롬프트 간의 유사도를 활용하여 노이즈 레이블을 식별합니다. 최적화를 위해 Visual Prompt Tuning (VPT) 기술을 사용합니다.

- **Performance Highlights**: DeFT는 7개의 합성 및 실제 노이즈 데이터 세트에서 실험을 통해 노이즈 레이블 탐지와 이미지 분류 작업 모두에서 뛰어난 성능을 입증하였습니다. 특히, 정교한 분류 작업에서 개선된 성능을 보였습니다.



### InfantCryNet: A Data-driven Framework for Intelligent Analysis of Infant Cries (https://arxiv.org/abs/2409.19689)
- **What's New**: 이번 논문에서는 infant cries(유아의 울음소리)를 이해하기 위한 새로운 데이터 기반 프레임워크, 'InfantCryNet'을 제안합니다. 이 프레임워크는 울음 소리의 탐지 및 분석을 동시에 수행하며, 데이터 부족 문제를 해결하기 위해 사전 훈련된 오디오 모델을 사용합니다.

- **Technical Details**: 모델은 통계적 풀링(statistical pooling)과 다중 헤드 주의(pooling with multi-head attention) 기법을 사용하여 더 효과적으로 특징을 추출하며, 모델의 효율성을 높이기 위해 knowledge distillation(지식 증류) 및 모델 양자화(model quantization) 기법을 적용했습니다.

- **Performance Highlights**: 실제 데이터셋을 통한 실험 결과, 제안된 프레임워크는 분류 정확도에서 기존의 최첨단 모델 대비 4.4% 높은 성능을 나타내었고, 모델 압축 기술을 통해 모델 크기를 7% 줄일 수 있었습니다. 성능 손실 없이 최대 28%까지 모델 크기를 줄일 수 있는 가능성을 보여주었습니다.



### Temporal Source Recovery for Time-Series Source-Free Unsupervised Domain Adaptation (https://arxiv.org/abs/2409.19635)
- **What's New**: 본 논문에서는 Temporal Source Recovery (TemSR)라는 새로운 프레임워크를 제안하며, 이는 Time-Series Source-Free Unsupervised Domain Adaptation (TS-SFUDA)에서 중요한 시간적 종속성을 효과적으로 전이할 수 있도록 설계되었습니다. 기존의 방법들은 특정한 소스 도메인 설계에 의존하고 있지만, TemSR은 소스 데이터에 대한 액세스 없이도 우수한 결과를 도출할 수 있습니다.

- **Technical Details**: TemSR은 마스킹(masking), 복구(recovery), 최적화(optimization)를 포함하는 복구 프로세스를 통해 소스와 유사한 분포(source-like distribution)를 생성하며, 이를 위해 세그먼트 기반 정규화(segment-based regularization)와 앵커 기반 회복 다양성 극대화(anchor-based recovery diversity maximization) 기법을 활용합니다. 이러한 기법들은 시간적 종속성을 복구하고, 로컬 종속성을 회복하는 데 중요합니다.

- **Performance Highlights**: 다양한 TS 작업에서의 광범위한 실험을 통해 TemSR의 효과가 입증되었으며, 기존 TS-SFUDA 방법을 초월하는 성과를 보였습니다. 또한 소스, 소스 유사, 타겟 도메인 간의 분포 불일치(discrepancy) 변화에 대한 분석을 통해 TemSR이 효과적인 소스 유사 도메인을 복구하고, 실제로 소스 데이터에 접근하지 않고도 도메인 간의 격차를 줄일 수 있음을 확인했습니다.



### MCDDPM: Multichannel Conditional Denoising Diffusion Model for Unsupervised Anomaly Detection in Brain MRI (https://arxiv.org/abs/2409.19623)
Comments:
          Accepted in CISP-BMEI 2024

- **What's New**: 이번 연구에서는 뇌 MRI 스캔의 비지도 학습(anomaly detection)에서의 문제점을 해결하기 위해 Multichannel Conditional Denoising Diffusion Probabilistic Model (MCDDPM)이라는 개선된 모델을 제안합니다. 본 모델은 추가적인 건강한 이미지를 활용하여 더 높은 신뢰성과 왜곡 방지를 달성함으로써, 기존 DDPM 계열 모델의 문제를 해결합니다.

- **Technical Details**: MCDDPM은 여러 채널의 정보를 활용하여 비지도 이차 탐지에서 최종 이미지의 정확도와 현실성을 향상시킵니다. 이 과정에서 여러 모델을 사용할 필요 없이 직접적으로 컨텍스트 정보(contextual information)를 통합하여 모델 설계를 간소화합니다. 실험은 다양한 데이터셋(BraTS20, BraTS21 등)을 통해 수행되었습니다.

- **Performance Highlights**: 실험 결과, MCDDPM은 높은 품질의 이미지를 재구성하며, 뇌 MRI 스캔에서 비정상적인 영역의 픽셀 수준 식별을 효과적으로 지원합니다. 기존 모델들에 비해 MCDDPM의 성능이 더욱 향상되었음을 보여주는 데이터가 확보되었습니다.



### Federated Learning from Vision-Language Foundation Models: Theoretical Analysis and Method (https://arxiv.org/abs/2409.19610)
- **What's New**: 본 논문은 CLIP과 같은 사전학습된 비전-언어 기초 모델을 연합 학습(federated learning)에 통합하여 다양한 작업에 대한 일반화(generalization)를 향상시키는 데 중점을 두고 있습니다. 특히, 프롬프트 기반 연합 학습(prompt-based federated learning)의 성능을 이해하기 위한 이론적 분석 프레임워크가 제시됩니다.

- **Technical Details**: 프롬프트 기반 연합 학습을 위한 분석 프레임워크는 feature learning theory를 기반으로 구성되어 있으며, 신호 학습(signal learning)과 노이즈 기억(noise memorization)의 진화를 모니터링합니다. 성과는 작업 관련(task-relevant) 계수와 작업 비관련(task-irrelevant) 계수의 비율로 평가됩니다. 또한, 포트폴리오 최적화(portfolio optimization)에서의 수익(income)과 위험(risk)의 유사성을 바탕으로, 글로벌 프롬프트(global prompt)와 로컬 프롬프트(local prompt)를 결합하여 프롬프트 포트폴리오를 구축합니다.

- **Performance Highlights**: 실험을 통해 프롬프트 포트폴리오의 성능 우위를 입증하였으며, 최적의 혼합 계수를 도출했습니다. 이론적 주장들은 실증적 실험에서도 지지를 받으며, 실제 시나리오에서의 접근 방식의 우수성을 꾸준히 보여주고 있습니다.



### Hyper-Connections (https://arxiv.org/abs/2409.19606)
- **What's New**: 새로운 방법론인 하이퍼 연결(hyper-connections)을 소개합니다. 이 방법은 기존의 잔여 연결(residual connections)의 몇 가지 단점을 다루는 것을 목표로 하며, 네트워크가 각기 다른 깊이의 피처(feature) 사이의 연결 강도를 조절하고 레이어를 동적으로 재배치할 수 있게 합니다.

- **Technical Details**: 하이퍼 연결은 네트워크가 피처 간의 연결 강도를 학습할 수 있게 하며, 깊이 연결(depth-connections)과 폭 연결(width-connections)을 제안합니다. 이를 통해 잔여 연결의 장점을 유지하면서도 계산량과 매개변수 증가를 최소화할 수 있습니다. 또한 동적 하이퍼 연결(dynamic hyper-connections)을 통해 입력에 따라 연결 가중치를 조정할 수 있습니다.

- **Performance Highlights**: 하이퍼 연결은 대형 언어 모델(LLMs)의 사전 학습(pre-training) 및 비전 처리(vison tasks)에서 잔여 연결 대비 성능 향상을 보였습니다. 예를 들어, DHC를 사용하는 모델이 1.8배 더 빠르게 수렴하며, ARC-Challenge에서 약 6점 향상을 나타냈습니다. 또한, 하이퍼 연결이 적용된 모델은 인접한 레이어 간의 특성 유사성을 줄이고, 각 레이어의 영향을 확장하는 데 기여합니다.



### Efficient Quality Control of Whole Slide Pathology Images with Human-in-the-loop Training (https://arxiv.org/abs/2409.19587)
Comments:
          18 pages

- **What's New**: HistoROI라는 새로운 경량 딥러닝 기반 분류기를 소개하며, 이는 전체 슬라이드 이미지(WSI)를 여섯 가지 조직 영역(상피, 기질, 림프구, 지방, 인공물, 기타)으로 구분합니다.

- **Technical Details**: HistoROI는 ‘human-in-the-loop’와 ‘active learning’ 패러다임을 사용하여 교육 데이터의 변이를 보장하므로 라벨이 효율적인 일반화를 달성합니다. 이 모델은 단일 데이터셋에서 학습하였음에도 불구하고 여러 기관에서 일관되게 우수한 성능을 발휘합니다.

- **Performance Highlights**: CAMELYON 유방암 림프절 및 TCGA 폐암 데이터셋에서 HistoROI 사용 후 AUC가 각각 0.88에서 0.92, 0.88에서 0.93으로 향상되었습니다. 또한 93개의 주석이 달린 WSI 테스트 데이터셋에서 HistoQI의 성과를 초월하는 성능을 보였습니다.



### Brain Tumor Classification on MRI in Light of Molecular Markers (https://arxiv.org/abs/2409.19583)
Comments:
          ICAI'22 - The 24th International Conference on Artificial Intelligence, The 2022 World Congress in Computer Science, Computer Engineering, & Applied Computing (CSCE'22), Las Vegas, USA. The paper acceptance rate 17% for regular papers. The publication of the CSCE 2022 conference proceedings has been delayed due to the pandemic

- **What's New**: 본 연구에서는 저급 신경교종(low-grade gliomas)의 임상 결과와 관련된 1p/19q 유전자 공동 삭제(co-deletion) 상태를 예측하기 위해 특별히 설계된 MRI 기반 합성곱 신경망(convolutional neural network, CNN)을 제안합니다. 기존의 전이 학습 모델(transfer learning model) 대신에 처음부터 모델을 개발하여 신뢰성을 높였습니다.

- **Technical Details**: 제안된 모델은 합성곱(convolution) 레이어, 풀링(pooling) 레이어, LeakyReLU, 소프트맥스(Softmax), 드롭아웃(dropout) 레이어와 완전 연결(Dense) 레이어로 구성되어 있으며, 3x3 크기의 커널을 사용합니다. 모델 학습 과정에서는 가우시안 노이즈(Gaussian noise)를 주입하고 데이터 보강(data augmentation)을 통해 성능을 향상시켰습니다. 125개의 1p/19q 공동 삭제 및 31개의 비삭제 이미지를 포함한 검증 세트를 사용했습니다.

- **Performance Highlights**: 제안된 네트워크는 1p/19q 공동 삭제 이미지를 분류할 때 96.37% F1-점수, 97.46% 정밀도(precision), 96.34% 재현율(recall)을 달성했습니다. 비교 대상인 InceptionV3, VGG16, MobileNetV2와 같은 모델에 비해 우수한 성능을 보였습니다.



### BuildingView: Constructing Urban Building Exteriors Databases with Street View Imagery and Multimodal Large Language Mod (https://arxiv.org/abs/2409.19527)
Comments:
          8 pages, 6 figures

- **What's New**: Urban Building Exteriors의 중요성이 커지면서, Google Street View와 OpenStreetMap의 공간 정보를 통합하는 BuildingView라는 혁신적인 접근 방식을 제안합니다. 이 연구는 도시 건물 외관 데이터의 정확성을 높이고 핵심 지속 가능성 및 디자인 지표를 도출하여 관리할 수 있는 프레임워크를 개발하였습니다.

- **Technical Details**: BuildingView는 Street View 이미지와 멀티모달 대형 언어 모델(LLMs)을 결합하여 도시 건물 외관 데이터베이스를 만드는 혁신적 방법론입니다. 연구 방법론은 문헌 조사, 건물 및 Street View 샘플링, ChatGPT-4O API를 사용한 주석 작업으로 구성됩니다.

- **Performance Highlights**: 뉴욕, 암스테르담, 싱가포르의 데이터로 검증된 결과, BuildingView는 도시 연구를 위한 포괄적인 도구를 제공하며, 도시 계획, 건축 디자인 및 환경 정책에 대한 정보 기반 의사 결정을 지원합니다.



### Efficient Backdoor Defense in Multimodal Contrastive Learning: A Token-Level Unlearning Method for Mitigating Threats (https://arxiv.org/abs/2409.19526)
- **What's New**: 본 연구는 멀티모달 대비 학습(Multimodal Contrastive Learning)에서 발생할 수 있는 backdoor 공격에 대한 새로운 방어 기제를 제안합니다. 이를 위해 '기계 학습 삭제(machine unlearning)' 개념을 활용하여 모델의 backdoor 취약성을 신속하게 제거하는 방법을 제시합니다.

- **Technical Details**: 제안하는 기법은 Unlearn Backdoor Threats (UBT)로, 적은 수의 오염 샘플을 선택하여 모델이 backdoor 특징을 잊도록 유도합니다. 이 과정에서 과도적 훈련(overfit training)을 통해 의심스러운 샘플을 탐지하고, 해당 샘플의 일부를 선택하여 신속하게 제거합니다. 이 새로운 접근법은 토큰 기반의 부분 학습 삭제(training regime) 방법을 포함하여, 모델의 취약한 요소에 집중하여 backdoor 연관성을 분리합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 다양한 backdoor 공격 방법에 대해 효과적으로 방어하며, 기존 방법과 비교할 때 공격 성공률(attack success rate)을 19% 감소시키고, 깨끗한 정확도(clean accuracy)를 2.57% 증가시켰습니다.



### IWN: Image Watermarking Based on Idempotency (https://arxiv.org/abs/2409.19506)
- **What's New**: 이 논문은 이미지 워터마크 처리를 위한 새로운 신경망 모델인 Idempotent Watermarking Network (IWN)를 제안합니다. 이 모델은 idempotency(멱등성) 개념을 도입하여 색상 이미지 워터마크의 복구 품질을 향상시키고, 전통적인 워터마킹 방법의 약점을 극복하려고 합니다.

- **Technical Details**: IWN 모델은 Idempotent Generative Network (IGN)의 기본 원리를 바탕으로 하며, 워터마크가 공격이나 손상을 입더라도 원래 상태로 복원할 수 있도록 지원합니다. 이 모델은 주파수 영역에서 Discrete Cosine Transform (DCT) 기법을 활용하여 원본 이미지와 워터마크 간의 DCT 계수를 결합하여 워터마크를 삽입합니다. 멱등성을 통해 입력 이미지의 특성을 안정적으로 유지하면서, 반복적인 프로젝션과 매핑 후에도 출력이 변하지 않는 특징을 가집니다.

- **Performance Highlights**: IWN 모델은 임베딩 용량과 내구성 간의 균형을 잘 이룹니다. 손상된 워터마크 이미지를 효과적으로 원래 워터마크 상태로 복원하는 능력 덕분에, 정보 은닉을 위한 워터마킹 기술의 신뢰성과 유용성이 높아집니다. 추가적으로, 이 모델은 색상 이미지 워터마크의 추출 품질을 상당히 향상시킵니다.



### OptiGrasp: Optimized Grasp Pose Detection Using RGB Images for Warehouse Picking Robots (https://arxiv.org/abs/2409.19494)
Comments:
          8 pages, 6 figures

- **What's New**: 본 논문에서는 로봇의 피킹(picking) 능력을 향상시키기 위해 RGB 이미지만을 사용하여 흡입(grasping)을 개선하는 새로운 접근 방식을 제안합니다. 이 접근 방식은 대규모 합성 데이터셋으로 훈련되었으며, 실제 로봇과 훈련 세트에 포함되지 않은 다양한 새로운 객체에 대해 일반화할 수 있는 능력을 가집니다.

- **Technical Details**: 우리는 Depth Anything 모델의 사전 훈련된 가중치를 활용하여 RGB 이미지만으로 최적의 흡입 포인트와 각도를 찾는 네트워크를 설계했습니다. 이 구조에서는 Affordance map을 사용하는데, 이는 로봇이 최적의 잡기 포인트를 선택할 수 있도록 안내합니다. 네트워크는 합성 데이터에서 훈련되었으며, 실제 환경에서 82.3%의 성공률을 달성했습니다.

- **Performance Highlights**: 상당히 다양한 객체에 대해 진행된 실험에서, 우리 방법은 혼잡한 창고 환경에서도 82.3%의 성공률을 기록하며 우수한 성능을 발휘함을 보여주었습니다. 이를 통해 우리는 비싼 깊이 센서 없이도 흡입 능력을 극대화할 수 있는 가능성을 제시합니다.



### KineDepth: Utilizing Robot Kinematics for Online Metric Depth Estimation (https://arxiv.org/abs/2409.19490)
Comments:
          8 pages, 5 figures

- **What's New**: 이 논문에서는 단일 보정 카메라를 활용하여 로봇이 '측정기' 역할을 수행하고, 상대 깊이 추정을 실시간으로 메트릭 깊이로 변환할 수 있는 새로운 방법인 KineDepth를 제안합니다. 이를 통해 로봇이 더 정확한 깊이 정보를 사용하여 작업을 수행할 수 있게 됩니다.

- **Technical Details**: KineDepth는 LSTM 기반의 메트릭 깊이 회귀 모델을 사용하여 상대 깊이를 메트릭 깊이로 변환하며, 로봇의 운동 근처에서 깊이 맵을 정확히 복원합니다. 이 방법은 실시간 온라인 추정 및 필터링 기능을 제공하여 사전 훈련 없이도 작동이 가능하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, KineDepth 방법이 기존의 단안 메트릭 깊이 추정 기법보다 22.1% 더 낮은 깊이 오차를 보이며, downstream task에서 52% 더 높은 성공률을 기록했습니다.



### Accelerating Malware Classification: A Vision Transformer Solution (https://arxiv.org/abs/2409.19461)
Comments:
          8 pages, 5 figures, 1 table Submitted to Neurips 2024 ML for system worshop

- **What's New**: 이번 논문에서는 진화하는 사이버 보안 환경에서 신속하고 정확한 악성 코드 분류의 중요성이 강조되며, LeViT-MC라는 새로운 아키텍처가 제안되었습니다. 본 아키텍처는 비전 트랜스포머 기반 구조를 활용하여 우수한 상태의 악성 코드 탐지 및 분류 성능을 보여줍니다.

- **Technical Details**: LeViT-MC 아키텍처는 이진 분류를 위한 DenseNet과 신속한 추론 속도를 자랑하는 가벼운 비전 트랜스포머인 LeViT를 결합하여 구성됩니다. 악성 코드 이미지를 RGB 이미지로 변환한 후, 이 철저한 접근 방식을 통해 높은 정확성과 빠른 인퍼런스 속도를 달성합니다.

- **Performance Highlights**: LeViT-MC는 악성 코드 분류에서 96.6%의 정확도를 기록하며, MaleVis 데이터셋에서 이전의 모든 모델을 초월하는 뛰어난 성능을 보였습니다. 또한, 이 아키텍처는 평균 인퍼런스 속도가 일반적인 비전 트랜스포머보다 약 10배, 최고의 CNN 모델인 ResNet보다 약 3배 더 빠릅니다.



### On the universality of neural encodings in CNNs (https://arxiv.org/abs/2409.19460)
Comments:
          Appeared at the ICLR 2024 Workshop on Representational Alignment (Re-Align), 13 pages, 5 figures

- **What's New**: 이번 연구는 이미지 분류 작업에 대해 훈련된 합성곱 신경망(convolutional neural networks)에서의 신경 인코딩의 보편성을 조사합니다. 기존의 방법론과 달리, 우리는 학습된 가중치의 유사성을 직접 비교하는 절차를 개발했습니다. VGG 타입 네트워크의 여러 레이어에서 학습된 고유 벡터(eigenvectors)가 다양한 자연 이미지 데이터 세트 간에 보편적으로 나타난다는 것을 보여주었습니다.

- **Technical Details**: 네트워크의 가중치(w)를 직접 비교하기 위해 공간 및 채널 차원의 분해를 기반으로 한 절차를 사용했습니다. 우리 연구는 CNN의 가중치 텐서의 통계적 성질을 분석함으로써, 공간 필터의 고유 벡터가 단순하다는 것을 발견했습니다. 이는 필터 크기나 데이터 세트에 상관없이 저차원적이며, 보편적인 공간 필터 고유 벡터 세트가 나타남을 의미합니다.

- **Performance Highlights**: 자연 이미지를 위한 보편적인 신경 인코딩이 발생함을 보여줍니다. 우리는 데이터 세트와 작업 간에 인코딩 유사성을 발견했으며, 이는 전이 학습(transfer learning)과 자가 지도 학습(self-supervised learning)의 성공을 설명합니다. 이 연구는 신경망의 성능을 극대화하는 것 대신, 학습된 인코딩의 보편성을 극대화하며, 원칙적인 기초 모델을 구축하는 방법을 제시합니다.



### Language-guided Robust Navigation for Mobile Robots in Dynamically-changing Environments (https://arxiv.org/abs/2409.19459)
- **What's New**: 이 논문에서는 인간-인-루프(Human-in-the-loop) 네비게이션을 위한 환형 AI 시스템을 개발하였습니다. wheeled mobile robot이 환경 변화에 따른 경로의 변화를 감지하고, 필요한 경우 인간에게 피드백을 요청하는 방식으로 작업을 수행합니다.

- **Technical Details**: 로봇의 현재 계획을 모니터링하여 환경의 변화에 즉각적으로 대응하고, 자연어로 표현된 인간의 피드백을 해석하여 로봇의 네비게이션용 local waypoints로 변환합니다. 이를 위해 semantic feature map 및 aligned obstacle map을 활용합니다.

- **Performance Highlights**: 시뮬레이션 및 실제 하드웨어 실험에서 자원 제약이 있는 wheeled robot이 실제 환경에서의 내비게이션 작업을 성공적으로 수행하도록 확인하였습니다. 이 시스템은 정밀 농업 및 건설과 같은 분야에서 지속적인 환경 모니터링을 지원할 수 있습니다.



### See Where You Read with Eye Gaze Tracking and Large Language Mod (https://arxiv.org/abs/2409.19454)
Comments:
          9 pages

- **What's New**: 이 논문에서는 선형 독서(linear reading)와 점프 독서(jump reading)를 지원하는 새로운 독서 추적 및 강조 시스템 R²TH를 제시합니다. 특히, 16명의 사용자를 대상으로 한 실험을 바탕으로 두 가지 시선 오류 모델을 설계하여 점프 독서 감지 및 위치 변경을 가능하게 했습니다.

- **Technical Details**: 이 시스템은 gaze error model을 사용하여 사용자 시선을 기반으로 독서 진행 상황을 추적하며, 독서 추적 도메인에 맞춘 라인-시선 정렬 기회를 활용해 동적인 캘리브레이션(calibration)을 수행합니다. 점프 독서를 감지할 경우, 사용자의 시선 경로를 추적하여 문장 구두점을 검색하고, LLM(large language model)의 맥락 인지 능력을 활용해 후보 문장을 평가합니다.

- **Performance Highlights**: 통제된 실험에서는 선형 독서 추적의 신뢰성을 입증했으며, 점프 독서 추적의 경우 84%의 정확도를 기록했습니다. 또한, 18명의 자원봉적으로 토대로 한 실제 현장 테스트에서 본 시스템의 효과성을 입증하여 독서 효율성을 개선하고 사용자 경험을 강화했습니다.



### Multi-sensor Learning Enables Information Transfer across Different Sensory Data and Augments Multi-modality Imaging (https://arxiv.org/abs/2409.19420)
Comments:
          18 pages, 14 figures. Accepted by IEEE Transactions on Pattern Analysis and Machine Intelligence

- **What's New**: 본 논문에서는 데이터 기반 다중 양상 이미징(data-driven multi-modality imaging, DMI) 전략을 제안하고, CT(Computed Tomography)와 MRI(Magnetic Resonance Imaging)의 시너지 이미징에 대한 다중 센서 학습(multi-sensor learning, MSL) 프레임워크를 소개합니다. 기존의 이미징 기법들과는 달리, 이 방법은 서로 다른 이미징 모달리티 간의 정보를 통합하여 최적의 하이브리드 이미지를 생성할 수 있습니다.

- **Technical Details**: DMI의 핵심은 일반적으로 intra-modality(모달리티 내) 및 inter-modality(모달리티 간) 두 가지 특징을 식별하는 것입니다. MSL 프레임워크는 transformer 네트워크를 활용하여 각 센서 특징을 통합하고, 결국 하나의 시퀀스 이미지를 생성합니다. 또한, conditional instance normalization (CIN) 기법을 적용하여 다양한 비율의 하이브리드 이미지 생성이 가능합니다.

- **Performance Highlights**: 제안된 DMI 전략의 효과성은 CT와 MRI의 시너지 이미징을 통해 입증되었습니다. 이 연구는 다중 양상 이미징의 경계를 허물고, 다양한 분야에서의 DMI 애플리케이션을 위한 엄청난 잠재력을 보여줍니다. 코드도 GitHub를 통해 제공되어 관심 있는 연구자들이 활용할 수 있습니다.



### Brain-JEPA: Brain Dynamics Foundation Model with Gradient Positioning and Spatiotemporal Masking (https://arxiv.org/abs/2409.19407)
Comments:
          The first two authors contributed equally. NeurIPS 2024 Spotlight

- **What's New**: Brain-JEPA는 Joint-Embedding Predictive Architecture (JEPA)를 기반으로 한 뇌 동역학의 기초 모델로, 인구 예측, 질병 진단/예후, 그리고 특성 예측에서 최첨단 성능을 자랑합니다.

- **Technical Details**: 이 모델은 두 가지 혁신적인 기법인 Brain Gradient Positioning과 Spatiotemporal Masking을 통합합니다. Brain Gradient Positioning은 뇌 기능 분할을 위한 기능적 좌표계를 도입하여 여러 지역의 위치 인코딩을 향상시킵니다. Spatiotemporal Masking은 fMRI 데이터의 독특한 특성에 맞추어 설계되어 이질적인 시계열 패치의 문제를 해결합니다.

- **Performance Highlights**: Brain-JEPA는 후속 실험에서 최첨단 결과를 기록하였으며, 다양한 인종 그룹에 걸쳐 일반화 능력이 우수합니다. 또한, 기존 대형 모델을 능가하여 뇌 활동 분석 영역에서 새로운 패러다임을 제시합니다.



### Projected Tensor-Tensor Products for Efficient Computation of Optimal Multiway Data Representations (https://arxiv.org/abs/2409.19402)
Comments:
          31 pages, 12 figures

- **What's New**: 이 연구에서는 새로운 projected tensor-tensor product를 제안하여 기존의 invertible matrix에 대한 제약을 완화하고 컴퓨터 수치적 오버헤드를 줄이면서 기본적 선형 대수 속성을 유지합니다. 이 접근법은 다차원 데이터에서 계산 복잡성을 크게 줄이며 수학적인 타당성을 확보합니다.

- **Technical Details**: 제안된 방법은 unitary columns를 가진 tall-and-skinny matrix로 정의된 projected tensor-tensor product를 사용하며, 이는 특정 차원에서 선형적 의존성에 기반하여 계산 복잡성을 크게 감소시킵니다. 기존의 ⋆𝐌\star_{\mathbf{M}}-product는 invertible matrix의 사용으로 인해 다차원 데이터 처리에서 계산상의 병목을 초래하지만, proposed product는 복잡성 감소와 효율적인 저장을 가능하게 합니다.

- **Performance Highlights**: numerical experiments를 통해 제안된 projected product가 higher-order SVD(HOSVD)와 비교하여 더 우수한 근사치를 제공함을 입증하였으며, 이는 video 및 hyperspectral imaging 데이터에서의 실험 결과로 뒷받침됩니다.



### Canonical Correlation Guided Deep Neural Network (https://arxiv.org/abs/2409.19396)
Comments:
          11 pages, 13 figures

- **What's New**: 제안된 새로운 접근 방식은 Canonical Correlation Guided Deep Neural Network (CCDNN)입니다. 이는 다변량 분석(MVA)과 머신 러닝의 새로운 융합으로, 기존의 선형 상관관계 분석(CCA)과는 달리 상관관계를 최대화하는 것이 아니라 제약 조건으로 사용하여 최적화 문제를 해결합니다.

- **Technical Details**: CCDNN은 딥 뉴럴 네트워크(DNN)를 기반으로 하며, 두 개의 데이터 뷰에서 높은 선형 상관 관계를 갖는 표현을 학습하는 프레임워크입니다. 이 네트워크는 특정 최적화 작업(재구성, 분류, 예측)에 중점을 두고 있으며, 상관 관계로 인한 중복을 줄이기 위해 중복 필터를 설계했습니다.

- **Performance Highlights**: MNIST 데이터셋에서의 실험 결과, CCDNN은 DCCA 및 DCCAE보다 평균 제곱 오차(MSE)와 평균 절대 오차(MAE) 측면에서 더 나은 재구성 성능을 보여주었습니다. 제안된 방법은 산업 고장 진단 및 잔여 유용 수명 예측에서도 우수한 성능을 입증하였습니다.



### DOTA: Distributional Test-Time Adaptation of Vision-Language Models (https://arxiv.org/abs/2409.19375)
Comments:
          In submission

- **What's New**: 이 논문에서는 기존의 Training-Free Test-time Dynamic Adapter(TDA)의 한계를 극복하기 위해 DistributiOnal Test-time Adaptation(Dota)라는 새로운 방법을 제안합니다. Dota는 테스트 샘플의 분포를 지속적으로 추정하여 모델이 배포 환경에 적응할 수 있게 합니다.

- **Technical Details**: Dota는 Bayes' 정리를 기반으로 하여 추정한 분포를 사용하여 테스트 샘플의 후방 확률(test-time posterior probabilities)을 계산합니다. 여기에서 각 클래스의 임베딩 분포가 가우시안 분포를 따른다는 가정을 하며, 이는 TDA보다 약 20배 빠른 추론 속도를 제공합니다. 또한, Dota는 사람-주도 피드백(human-in-the-loop paradigm)을 통해 불확실한 샘플을 식별하고 적응하도록 돕습니다.

- **Performance Highlights**: 광범위한 데이터셋을 통한 실험 결과 Dota는 CLIP 모델이 지속적으로 학습할 수 있게 해주며, 기존의 최첨단 방법들에 비해 유의미한 성과 향상을 보였습니다.



### Efficient Semantic Diffusion Architectures for Model Training on Synthetic Echocardiograms (https://arxiv.org/abs/2409.19371)
- **What's New**: 이번 논문은 심장 초음파 이미지 생성에 있어 새로운 $\\Gamma$-분포 Latent Denoising Diffusion Models (LDMs)을 제안합니다. 기존의 모델에 비해 계산 효율성을 개선하면서도 시맨틱한 의미를 가진 합성 이미지를 생성하는 것에 중점을 두고 있습니다.

- **Technical Details**: $\\Gamma$-VAE(Variational Autoencoder)를 결합한 새로운 생성 방식으로, 초음파 이미지를 저해상도로 압축하여 디퓨전 모델 훈련에 활용하고, 이후 고해상도로 변환하는 과정에서 기존의 LDM 접근 방식을 개선했습니다. 연구에서는 ODE(Ordinary Differential Equation) 해법의 성능도 분석하여 최적의 계산 효율성을 달성하려고 합니다.

- **Performance Highlights**: 제안된 모델은 기존의 GAN이나 다른 디퓨전 모델에 비해 계산 비용을 크게 줄이면서도 세분화(segmentation)와 분류(classification) 성능을 유지하거나 개선했습니다. 결과적으로, 실제 데이터로 훈련된 모델에 비해 합성 데이터로 훈련된 모델의 성능이 향상되어, 더 높은 데이터 다양성을 제공하였음을 보여주었습니다.



### MambaEviScrib: Mamba and Evidence-Guided Consistency Make CNN Work Robustly for Scribble-Based Weakly Supervised Ultrasound Image Segmentation (https://arxiv.org/abs/2409.19370)
- **What's New**: 본 논문에서는 초음파 이미지 분할을 위해 스크리블 기반 약한 감독 학습(Weakly Supervised Learning, WSL)이 처음으로 적용됩니다. 새로운 하이브리드 CNN-Mamba 프레임워크를 제안하여, 초음파 이미지에서의 형상과 병변을 보다 효과적으로 분할할 수 있는 방법을 제공합니다.

- **Technical Details**: 제안된 모델은 CNN과 Mamba 두 개의 브랜치 네트워크를 포함하며, 증거에 기반한 일관성(Evidence-Guided Consistency, EGC) 전략을 도입합니다. CNN 브랜치는 지역적 특징을 추출하고, Mamba 브랜치는 전역적 특성을 추출하여 초음파 이미지에서의 국소 해부학적 세부 사항과 전반적인 형태적 특성을 모두 캡처합니다. 또한, 이 모델은 디리클레 분포를 사용하여 세분화 확률의 두번째 순서 확률을 매개변수화합니다.

- **Performance Highlights**: 4개의 초음파 공개 데이터셋에서 다수의 실험을 수행한 결과, 제안된 방법의 경쟁력이 입증되었습니다. 특히, 스크리블 주석만을 사용하여 효율적인 세분화 결과를 도출하였으며, 기초 U-Net을 사용하는 추론 단계에서 다른 복잡한 모델에 비해 우수한 효율성을 보여주었습니다.



### Mind the Gap: Promoting Missing Modality Brain Tumor Segmentation with Alignmen (https://arxiv.org/abs/2409.19366)
- **What's New**: 이 논문에서는 MRI (Magnetic Resonance Imaging) 모달리티가 누락된 상황에서 뇌 종양 세분화를 위한 새로운 정렬 패러다임을 제안합니다. 이 패러다임은 잠재적 특성을 명확한 분포 앵커에 정렬하여 양측 모두에서 성능을 향상시킵니다.

- **Technical Details**: 제안된 접근 방식에서는 Kullback-Leibler (KL) 손실을 최적화하고, 각각의 모달리티를 잠재 공간에 정렬합니다. 이 과정에서 최적의 잠재 공간 분포 P_{mix}를 사용하여 모달리티 간의 갭을 줄이고, modality-invariant (모달리티 불변) 특성을 학습합니다.

- **Performance Highlights**: 실험 결과, 제안된 정렬 패러다임은 Dice score에서 평균 1.75의 개선을 달성하여 최신 백본에서도 우수한 성능을 보여줍니다. 이는 학생 네트워크에서 누락된 모달리티를 효과적으로 처리하는 데 기여합니다.



### Sparse Modelling for Feature Learning in High Dimensional Data (https://arxiv.org/abs/2409.19361)
- **What's New**: 본 논문은 고차원 데이터셋에서 차원 축소 및 특징 추출을 위한 혁신적인 접근 방식을 제안하며, 특히 목재 표면 결함 탐지에 대한 적용에 중점을 둡니다.

- **Technical Details**: 제안된 프레임워크는 희소 모델링 기법(sparse modeling techniques), 특히 Lasso 및 근접 경량화(proximal gradient) 방법을 통합한 포괄적인 파이프라인을 통해 효율적이고 해석 가능한 특징 선택(feature selection)을 지원합니다. VGG19와 같은 사전 훈련된(pre-trained) 모델을 활용하고, Isolation Forest 및 Local Outlier Factor와 같은 이상 탐지(anomaly detection) 방법을 포함시켜 복잡한 데이터셋에서 유의미한 특징을 추출하는 문제에 접근합니다.

- **Performance Highlights**: 정확도(accuracy)와 F1 점수(F1 score)와 같은 평가 지표를 활용하여 희소 모델링 기법의 성능을 평가하며, 시각화(visualizations)를 통해 결과를 보완합니다. 이 연구를 통해 목재 표면 결함 탐지 문제의 맥락에서 머신러닝에서 희소 모델링의 이해 및 적용을 발전시키고자 합니다.



### Toward Deep Learning-based Segmentation and Quantitative Analysis of Cervical Spinal Cord Magnetic Resonance Images (https://arxiv.org/abs/2409.19354)
Comments:
          5 pages, 3 figures

- **What's New**: 이번 연구는 목 척수의 미세 구조 및 거시 구조 특성을 연구하기 위한 다중 매개변수 분석과 딥 러닝 기반 의료 이미지 분할의 두 가지 도전에 대해 논의합니다. 기존 연구와 달리 의료 전문가의 기능적 검사 없이 MRI 이미지만을 사용하여 분석을 수행합니다.

- **Technical Details**: 연구에서는 UNet 유사 및 Transformer 기반의 향상된 프레임워크를 제안하며, 주의 깊은 스킵 연결(attentive skip connections)을 통해 MRI 이미지에서 거시 구조 측정의 높은 정확도를 달성합니다. 특히, 경추 척수의 미세 구조와 거시 구조 특징 간의 관계를 분석하기 위해 이미지 세그멘테이션(image segmentation) 기술을 활용합니다.

- **Performance Highlights**: 제안된 모델은 높은 해상도의 이미지를 처리하기 위한 여러 혁신을 포함하며, 기존의 자가 주의(self-attention) 방식 대신 병합 교차 공분산(attention mechanism)을 적용합니다. 이를 통해 두 가지 특성 간의 상관관계를 정량적으로 분석하고, 의존성을 포착하는 데 더욱 효과적입니다.



### Unveil Benign Overfitting for Transformer in Vision: Training Dynamics, Convergence, and Generalization (https://arxiv.org/abs/2409.19345)
- **What's New**: 이 논문은 Vision Transformer (ViT) 의 이론적 능력을 탐구하며, 특히 훈련 데이터에 과적합(overfit) 되는 경우의 일반화(generalization) 가능성을 이해하는 데 중점을 둡니다. 이는 최근 큰 발전을 이룬 Transformer 모델의 이론적 기초를 강화하기 위한 작업입니다.

- **Technical Details**: 우리는 self-attention layer와 softmax, 그리고 fully connected layer로 구성된 Transformer의 최적화(optimization)를 gradient descent를 통해 특정 데이터 분포 모델에서 연구하였습니다. softmax의 도전과제 및 Transformer 최적화에서 여러 가중치의 상호 의존적 성격을 해결하는 기술을 개발함으로써, 훈련 동역학(training dynamics)을 성공적으로 특성화하고 사후 훈련에서 일반화를 달성했습니다.

- **Performance Highlights**: 우리의 결과는 데이터 모델의 신호 대 잡음 비(signal-to-noise ratio)를 기반으로 작은 테스트 오류(test error) 단계와 큰 테스트 오류 단계 간의 날카로운 조건을 구분할 수 있음을 보여줍니다. 이론적 결과는 실험적 시뮬레이션으로进一步적으로 검증되었습니다.



### Visual Question Decomposition on Multimodal Large Language Models (https://arxiv.org/abs/2409.19339)
Comments:
          Accepted to EMNLP2024 Findings

- **What's New**: 본 논문은 멀티모달 대형 언어 모델(MLLMs)의 질문 분해(Question Decomposition) 능력을 탐구하고 있으며, 기존의 단일 모드 언어 모델과는 달리 MLLMs의 응답 품질을 향상시키는 새로운 데이터셋인 DecoVQA+를 제안하고 있습니다.

- **Technical Details**: 시스템적 평가 프레임워크를 도입하여 MLLMs의 분해된 하위 질문의 품질을 평가하고, 선택적 분해(selective decomposition)를 위한 훈련 목표를 포함하는 효율적인 파인튜닝(pipeline)을 제안하고 있습니다.

- **Performance Highlights**: 파인튜닝된 MLLMs는 VQA 벤치마크 데이터셋에서 하위 질문 품질의 현저한 개선과 선택적 질문 분해에서 더 높은 정확도를 달성했습니다.



### Cauchy activation function and XN (https://arxiv.org/abs/2409.19221)
- **What's New**: 새로운 Cauchy Activation Function을 제안하고, 이를 활용한 CompleXNet(XNet)이라는 새로운 종류의 신경망을 개발하였습니다. 이 함수는 복소 해석학의 Cauchy 적분 정리에 기반하며, 정확도가 필요한 문제에 적합하게 설계되었습니다.

- **Technical Details**: Cauchy Activation Function은 λ1, λ2, d와 같은 훈련 가능한 매개변수를 사용하여 표현되며, 이 함수는 매끄러운 함수를 최고 가능 순서까지 근사할 수 있습니다. XNet은 이미지 분류와 편미분 방정식(Partial Differential Equations, PDE) 해결을 포함한 고차원 문제에서 효과적입니다.

- **Performance Highlights**: XNet은 MNIST와 CIFAR-10과 같은 기존 벤치마크를 크게 초월하며, Physics-Informed Neural Networks(PINNs)와 대비하여 저차원의 PDE 및 고차원 PDE 시나리오에서도 상당한 이점을 제공합니다.



### Semi-Supervised Bone Marrow Lesion Detection from Knee MRI Segmentation Using Mask Inpainting Models (https://arxiv.org/abs/2409.19185)
Comments:
          5 pages, 3 figures, submitted to SPIE Conference on Image Processing

- **What's New**: 본 논문에서는 고해상도 무릎 MRI에서 BML(골수 병변)의 식별을 위한 반지도 학습(local anomaly detection) 방법을 제안합니다. 이는 기존의 global anomaly detection 방법에 비해 성능을 향상시키는 것을 목표로 합니다.

- **Technical Details**: 제안된 방법은 3D 대퇴골 세분화 모델, 대규모 마스크 인페인팅(mask inpainting) 모델 및 일련의 후처리(post-processing) 기술을 통합합니다. 이를 통해 다양한 해상도의 MRI 이미지에서 BML을 효과적으로 감지합니다.

- **Performance Highlights**: 해당 방법은 다중 해상도 지식 증류(multiresolution knowledge distillation) 방법과 비교했을 때 Dice score, Intersection over Union (IoU) 및 픽셀 수준의 민감도, 특이성 및 정확성에서 우수한 성능을 보였습니다. 특히, 고해상도 이미지에서 세분화 성능이 2배 이상 향상되었습니다.



### Learning-Based Image Compression for Machines (https://arxiv.org/abs/2409.19184)
- **What's New**: 이번 연구는 머신 러닝 기반 이미지 압축 기법을 통해 기존의 압축 방법보다 향상된 성능을 보여주지만, 표준화 부족과 중요한 특징이 보존되지 않아 머신 러닝 파이프라인에서 널리 채택되지 못하고 있는 문제를 다룹니다. 우리는 압축 과정에 다운스트림 태스크를 통합하는 방법을 제안하고, 선훈련된 압축 인코딩 파이프라인의 여러 부분을 미세 조정하여 시각적 태스크에서의 성능을 향상시키는 결과를 보고합니다.

- **Technical Details**: 연구에서는 이미지 압축과 비디오 압축의 중요성을 강조하며, 머신 러닝 분석을 통한 이미지 및 비디오의 활용 가능성을 탐구합니다. 압축 파이프라인을 갖춘 모델은 압축된 이미지의 잠재 표현(latent representation)을 입력으로 받아, 분류(classification) 모듈과 함께 공동 학습(joint training)될 수 있습니다. 특히, BMSH-J2018 하이퍼프라이어 모델을 사용하여 텍스처 인식 이미지의 압축을 진행하고, cResNet-39 모델을 통해 압축된 이미지를 분류합니다.

- **Performance Highlights**: 실험 결과, 압축 및 분류 모듈의 공동 훈련이 더 나은 성능을 발휘함을 보여주며, 특히 생리학적 거리(threshold distances)에 맞춘 다양한 비트 출력을 기반으로 한 모델 간 비교에서 인간과 머신 모두의 효율성을 극대화하는 가능성을 확인했습니다.



### Feature Estimation of Global Language Processing in EEG Using Attention Maps (https://arxiv.org/abs/2409.19174)
- **What's New**: 이번 연구는 EEG(뇌파) 데이터를 분석하기 위해 Vision Transformer와 EEGNet을 활용한 새로운 접근 방식을 제시합니다. 이전 연구 결과와 일치하는 EEG 특성을 확인하는 동시에, 이를 신경망 모델의 내재적인 가중치를 통해 추정합니다.

- **Technical Details**: 본 연구에서는 인지 과제와 관련된 EEG 특징을 추정하기 위해 여러 딥러닝 모델을 적용했으며, 특히 Vision Transformer(ViT)와 EEGNet을 사용하여 주의 맵(attention map)을 생성했습니다. 이 과정을 통해 각 모델이 초점을 맞추는 특정 요소를 분석했습니다. EEG 데이터는 스페인 참가자들로부터 수집되었으며, 64개의 EEG 채널과 ECG, EOG 신호를 포함한 공개 데이터셋을 사용했습니다.

- **Performance Highlights**: EEGNet은 주제 독립(subject independence) 및 Listening과 Speaking 작업의 분류에서 가장 높은 정확도를 보였습니다. 이 연구는 EEG 신호를 활용한 조기 질병 탐지와 같은 의료적 응용 가능성을 제공하며, 인지 신경과학 분야에 큰 기여를 할 것으로 기대됩니다.



### From Vision to Audio and Beyond: A Unified Model for Audio-Visual Representation and Generation (https://arxiv.org/abs/2409.19132)
Comments:
          Accepted by ICML 2024

- **What's New**: 비디오에서 시각적 요소와 청각적 요소 간의 상호 작용을 연구하기 위한 새로운 통합 프레임워크인 'Vision to Audio and Beyond (VAB)'를 소개합니다. 이 프레임워크는 잠재 공간(latent space)에서 오디오와 비주얼의 표현 학습 및 생성 모델링을 수행합니다.

- **Technical Details**: VAB 모델은 사전 훈련된 음성 토크나이저와 이미지 인코더를 사용하여 오디오 토큰과 시각적 기능을 추출합니다. VAB는 시각적으로 조건화된 마스킹된 오디오 토큰 예측을 수행하는 사전 훈련 작업을 포함하며, 이로써 비디오에서 오디오를 생성하는 동시에 맥락 학습이 가능합니다. 또한, 이 모델은 다양한 오디오-비주얼 다운스트림 작업을 위해 미세 조정(fine-tuning)될 수 있습니다.

- **Performance Highlights**: VAB 모델은 정지된 비디오에서 고품질 오디오를 효율적으로 생성할 수 있으며, 기존 자동 회귀 접근방식보다 17배 빠른 속도를 자랑합니다. 실험 결과, VAB는 오디오-비주얼 검색 및 분류 작업에서 경쟁력 있는 성능을 보였습니다.



### Localizing Memorization in SSL Vision Encoders (https://arxiv.org/abs/2409.19069)
Comments:
          Accepted at NeurIPS 2024

- **What's New**: 이 논문에서는 자기지도학습(SSL) 인코더에서의 메모리화(memorization)를 분석하기 위한 새로운 접근 방식을 제안합니다. 두 가지 새로운 메트릭인 LayerMem과 UnitMem을 통해 SSL 인코더의 여러 층과 단위에서 메모리화를 국소화(localization)할 수 있습니다.

- **Technical Details**: LayerMem은 SSL 인코더 내의 메모리화 위치를 층(layer) 단위로 국소화하고, UnitMem은 개별 데이터 포인트에 대한 각 단위(유닛)별 메모리화를 측정합니다. 이 두 메트릭은 다운스트림 작업과 독립적이며, 레이블(레이블 정보) 없이도 계산이 가능하여 효율적입니다.

- **Performance Highlights**: 본 연구를 통해 다음의 주요 발견이 있었습니다: 1) SSL 메모리화는 모든 인코더에서 발생하며, 심층 층이 아닌 다양한 층에서 고르게 분포되어 있다. 2) SSL 인코더의 상당수 유닛이 높은 메모리화를 경험한다. 3) 비정형 데이터 포인트는 정형 데이터 포인트보다 높은 메모리화를 유발한다. 4) 비전 변환기에서 메모리화는 전결합층(fully-connected layers)에서 주로 발생한다. 이 연구는 메모리화 국소화가 모델의 파인튜닝(fine-tuning) 및 가지치기(pruning) 전략에 실질적인 혜택을 줄 수 있음을 보여줍니다.



### From Linguistic Giants to Sensory Maestros: A Survey on Cross-Modal Reasoning with Large Language Models (https://arxiv.org/abs/2409.18996)
- **What's New**: 본 논문은 Cross-Modal Reasoning (CMR)과 관련된 최근의 대규모 언어 모델(LLMs)의 발달을 다루고 있으며, LLMs가 CMR에 미치는 역할을 세분화하여 탐구하는 첫 번째 설문조사로서의 중요성을 강조합니다.

- **Technical Details**: 논문은 LLMs의 네 가지 주요 역할인 Multimodal Fusion Engine, Textual Processor, Cognitive Controller, Knowledge Enhancer를 소개합니다. 또한 Prompt Tuning, Instruction Tuning, Multimodal Pre-training과 같은 방법론을 설명하며 이들이 CMR에 활용되는 방식을 상세히 다룹니다.

- **Performance Highlights**: LLMs는 텍스트, 이미지 및 소리 등 다양한 모드 간의 새로운 정보를 이해하고 추론하는 능력을 통합하여 그들의 성능을 향상시킵니다. CMR의 적용 예시로는 비주얼 질문 응답, 비전-언어 탐색, 이미지 및 비디오 캡셔닝 등이 포함됩니다.



### A Review of Mechanistic Models of Event Comprehension (https://arxiv.org/abs/2409.18992)
- **What's New**: 이 리뷰는 사건 이해(Event comprehension)의 이론적 가정과 계산 모델을 검토하며, 담화 이해(disourse comprehension) 이론에서 현대의 사건 인지(event cognition) 프레임워크로의 진화를 추적합니다.

- **Technical Details**: 주요 담화 이해 모델로는 Construction-Integration, Event Indexing, Causal Network, 그리고 Resonance 모델 등이 있으며, 이들은 이해 과정에서의 인지적 프로세스를 이해하는 데 기여합니다. 현대의 사건 이해 이론으로는 Event Segmentation Theory, Event Horizon Model, Hierarchical Generative Framework 등이 있으며, 이들은 사건 이해에서의 예측(prediction), 인과 관계(causality), 다계층 표현(multilevel representations)의 중요성을 강조합니다. 분석된 다섯 가지 계산 모델로는 REPRISE, Structured Event Memory, Lu 모델, Gumbsch 모델, Elman 및 McRae 모델이 있습니다.

- **Performance Highlights**: 계층적 처리(hierarchical processing), 예측 메커니즘(prediction mechanisms), 그리고 표현 학습(representation learning)에 대한 접근 방법에 초점을 맞추어, 사건 이해에 있어 예측의 중요성과 사건 역학(Event dynamics)의 학습을 위한 다양한 전략을 강조합니다. 향후 연구의 중요한 영역으로는 구조적 표현(structured representations)에 대한 학습을 위한 더 정교한 접근법 필요성, 일화 기억 메커니즘(episodic memory mechanisms)의 통합, 사건 모델에 대한 적응형 업데이트 알고리즘 개발이 포함됩니다.



### IW-Bench: Evaluating Large Multimodal Models for Converting Image-to-Web (https://arxiv.org/abs/2409.18980)
- **What's New**: 최근 대규모 멀티모달 모델의 발전이 이미지 이해 능력에서 크게 향상되었습니다. 그러나 이러한 대규모 모델의 Image-to-Web 전환 능력을 평가하기 위한 강력한 기준이 부족합니다. 이를 해결하기 위해 IW-Bench라는 새로운 기준을 제정하고, Element Accuracy 및 Layout Accuracy와 같은 새로운 평가지표를 개발했습니다.

- **Technical Details**: IW-Bench는 1200개의 이미지와 웹 코드 쌍으로 구성되어 있으며, 난이도는 간단, 중간, 복잡으로 구분됩니다. Element Accuracy는 DOM (Document Object Model) 트리를 파싱하여 웹 요소의 완전성을 평가하며, Layout Accuracy는 DOM 트리를 공통 부분 수열로 변환하여 요소의 상대적 위치 관계를 분석합니다. 또한, 다섯 단계의 Chain-of-Thought Prompting을 통해 성능을 향상시키도록 설계되었습니다.

- **Performance Highlights**: 대규모 멀티모달 모델에 대한 광범위한 평가를 실시하였으며, 결과는 이들 모델의 강점과 개선이 필요한 영역에 대한 통찰을 제공합니다. 특히, 새로운 다섯 단계의 Chain-of-Thought 방법론이 성능 향상에 기여한 것으로 나타났습니다.



### Brain Network Diffusion-Driven fMRI Connectivity Augmentation for Enhanced Autism Spectrum Disorder Diagnosis (https://arxiv.org/abs/2409.18967)
Comments:
          14 pages, 16 figures, submitted to Journal of Neural Engineering

- **What's New**: 본 연구에서는 functional connectivity 생성을 위한 transformer 기반의 latent diffusion 모델인 Brain-Net-Diffusion을 제안합니다. 이는 기존의 데이터 부족 문제를 해결하기 위해 fMRI의 functional connectivity 생성을 위한 새로운 접근 방식을 제공합니다.

- **Technical Details**: Brain-Net-Diffusion은 Latent Connectivity Auto-Encoder와 Conditional Diffusion Transformer를 사용하여 새로운 functional connectivity 매트릭스를 생성합니다. 이 모델은 encoder-decoder 구조를 가지고 있으며, distribution normalization module을 도입하여 생성된 데이터가 실제 데이터의 분포와 일치하도록 합니다. 또한, conditional contrastive loss를 사용하여 effective conditioning 메커니즘을 보장합니다.

- **Performance Highlights**: 기존의 최첨단 방법보다 3% 더 향상된 성능을 달성하며, fMRI 데이터의 데이터 증강(Data Augmentation) 방법으로서 효과적임을 실험을 통해 입증하였습니다.



New uploads on arXiv(cs.AI)

### BabelBench: An Omni Benchmark for Code-Driven Analysis of Multimodal and Multistructured Data (https://arxiv.org/abs/2410.00773)
- **What's New**: BabelBench는 대형 언어 모델(LLM)의 다중 구조 데이터 처리 능력을 평가하는 새로운 벤치마크 프레임워크입니다. 이를 통해 멀티모달(multimodal) 데이터의 처리와 코드 실행 능력을 통합적으로 평가할 수 있습니다.

- **Technical Details**: BabelBench는 인지(perception), 상식 추론(commonsense reasoning), 논리 추론(logical reasoning) 등의 다양한 태스크를 포함하는 247개의 문제로 구성되어 있습니다. 이 벤치마크는 LLM의 멀티모달 이해, 테이블 해석 및 코드 생성을 평가하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 많은 최신 LLM 모델들이 BabelBench에서 개선의 여지가 있음을 보였습니다. 특히, ChatGPT 4와 같은 최첨단 모델조차도 이 벤치마크에서 상당한 발전이 필요하다는 사실이 밝혀졌습니다.



### LTLf Synthesis on First-Order Action Theories (https://arxiv.org/abs/2410.00726)
- **What's New**: 이 논문에서는 Golog 언어의 비결정적(operator) 특성을 환경의 통제 하에 두는 현실적인 상황을 탐구합니다. 이를 통해 프로그램의 프로그램 실현이 합성(synthesis) 문제로 전환됩니다. 

- **Technical Details**: Golog 프로그램은 무한 객채와 비지역적(non-local) 효과를 허용하는 자유로운 first-order action theories와 연결되어 있으며, LTLf의 1차 확장으로 지정된 시간적 목표를 포함합니다. 이 논문에서는 프로그램 수행의 모든 가능한 실행을 추적하는 게임 아레나를 구축하고 이로부터 두 플레이어 게임을 해결하여 합성 문제를 해결합니다.

- **Performance Highlights**: 이 접근 방식을 두 가지 도메인에서 평가하여 전반적인 타당성을 입증했습니다.



### Multimodal Auto Validation For Self-Refinement in Web Agents (https://arxiv.org/abs/2410.00689)
- **What's New**: 본 논문에서는 웹 에이전트의 성능을 향상시키기 위한 다중 모드 검증(multi-modal validation) 및 자기 개선(self-refinement) 접근법을 제안합니다. 이전의 연구를 기반으로 하여 자동 검증을 위한 다양한 모드(텍스트, 비전)의 영향과 계층 구조의 효과를 종합적으로 연구했습니다.

- **Technical Details**: 이 연구는 Agent-E 웹 자동화 프레임워크를 바탕으로 하여, 웹 에이전트가 작업 중 발생하는 실패를 감지하고 스스로 수정할 수 있도록 하는 자가 검증(auto-validator) 메커니즘을 통합합니다. 또한, LLM(대규모 언어 모델)의 에이전트 성능을 향상시키기 위해 기술 라이브러리(skill library)를 활용하는 방법론도 구현했습니다.

- **Performance Highlights**: Agent-E의 성능을 기존의 76.2%에서 81.24%로 향상시킨 성과를 보여주었으며, 이는 WebVoyager 벤치마크의 서브셋에서 이루어진 실험을 기반으로 합니다. 이 결과는 보다 신뢰할 수 있는 디지털 어시스턴트를 복잡한 현실 세계에서 구현할 수 있는 가능성을 제시합니다.



### Dynamic Planning for LLM-based Graphical User Interface Automation (https://arxiv.org/abs/2410.00467)
- **What's New**: 이 논문에서는 Large Language Models (LLMs)을 기반으로 한 GUI 에이전트의 효율적인 계획 수립을 위한 새로운 방법인 Dynamic Planning of Thoughts (D-PoT)를 제안합니다. D-PoT는 환경 피드백과 실행 이력을 바탕으로 계획을 동적으로 조정하여 작업 수행 능력을 향상시키고 있습니다.

- **Technical Details**: D-PoT는 모바일 GUI 작업에서 에이전트가 작업 목표를 달성하기 위해 새로운 스크린샷과 실행 이력을 지속적으로 통합하여 계획을 조정합니다. 이 접근법은 기존 ReAct 방법이 긴 대화 이력으로 인해 성능이 저하되는 문제를 극복하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과 D-PoT는 강력한 GPT-4V 기준 대비 +12.7%의 정확성 향상을 달성했습니다 (34.66% → 47.36%). 동적인 계획 수립의 응용은 여러 백본 LLMs에서 일반화가 가능하며, hallucinations(환각) 완화 및 이전에 보지 못한 작업에의 적응에 효과적임을 보여주었습니다.



### ReXplain: Translating Radiology into Patient-Friendly Video Reports (https://arxiv.org/abs/2410.00441)
Comments:
          13 pages

- **What's New**: ReXplain은 환자 친화적인 영상 보고서를 생성하는 AI 기반 시스템으로, 의료 영상 해석의 새로운 접근 방식을 제시합니다. 대형 언어 모델과 이미지 분할 모델, 아바타 생성 툴을 통합하여 간단한 언어로 설명하고, 중요한 이미지를 강조하며 3D 장기 렌더링을 제공합니다.

- **Technical Details**: ReXplain의 주요 기술 요소는 대형 언어 모델(LLM), CT 스캔의 해부학적 구역을 정리하는 분할 모델, 그리고 가상 발표자를 생성하는 아바타 생성 도구가 포함됩니다. 이 시스템은 구조화된 보고서를 방식을 통해 의사-환자 간의 소통을 모방하며, 기존의 전문 용어를 쉽게 이해할 수 있는 설명으로 번역합니다.

- **Performance Highlights**: 5명의 의료영상 전문의와 함께 진행된 파일럿 연구에서 ReXplain은 방사선 정보 전달의 정확성을 높이고, 환자와의 일대일 상담을 효과적으로 시뮬레이션할 수 있는 가능성을 보여주었습니다. 이 시스템은 환자의 이해도와 만족도를 향상시키고, 다중 모달 의료 커뮤니케이션 연구의 새로운 방향을 열어줄 수 있습니다.



### Vision Language Models Know Law of Conservation without Understanding More-or-Less (https://arxiv.org/abs/2410.00332)
- **What's New**: 이번 연구는 Vision Language Models (VLMs)의 인지 능력 중 하나인 보존(conservation) 개념과 수량 이해(quantity understanding) 간의 관계를 다룬다. CogDevelop2K에서 제공하는 ConserveBench를 활용하여 350개 이상의 질문을 통해 이 개념을 평가하였다.

- **Technical Details**: 연구는 Piaget의 전통적인 보존 과제를 바탕으로 세 가지 단계(Initial Phase, Manipulation Phase, End Phase)를 통해 VLM의 성능을 분석하였다. VLM은 다양한 물리적 수량(숫자, 길이, 고체 양, 액체 부피)에 대한 실험을 포함하며, 실험 설정은 실제 및 가상 환경에서 진행된다.

- **Performance Highlights**: VLM은 보존 과제에서 좋은 성능을 보여주었지만, 수량 이해 과제에서는 실패하는 모습을 보였다. 이는 보존 법칙이 물리적 영역에서 존재하더라도 수량에 대한 개념적 이해가 부족할 수 있음을 시사한다.



### Vision Language Models See What You Want but not What You S (https://arxiv.org/abs/2410.00324)
- **What's New**: 최근 Li et al.은 CogDevelop2K라는 데이터 집약적 인지 실험 벤치마크를 구축하여 기계 지능의 발달 경로를 평가했습니다. 본 연구에서는 Vision Language Models(VLMs)의 의도 이해(intentionality understanding)와 관점 취득(perspective-taking)을 조사했습니다.

- **Technical Details**: CogDevelop2K의 IntentBench 및 PerspectBench를 활용하여 300개 이상의 인지 실험을 기반으로 VLMs의 능력을 평가했습니다. 특히, Three Mountain Task를 변형하여 VLMs의 레벨 1과 레벨 2 관점 취득 능력을 테스트했습니다. 실험은 다양한 시나리오에 적용되었습니다.

- **Performance Highlights**: VLMs는 의도 이해에서는 높은 성능을 보였으나, 관점 취득에서는 저조한 성과를 기록했습니다. 이는 인지 과학에서 관점 취득이 의도 이해에 필수적이라는 일반적인 믿음에 도전하며 인간 지능과 기계 지능의 차이점을 명확히 합니다.



### Probing Mechanical Reasoning in Large Vision Language Models (https://arxiv.org/abs/2410.00318)
- **What's New**: 이번 논문에서는 CogDevelop2K의 MechBench를 활용하여 Vision Language Models(VLMs)의 기계적 추론 능력을 평가합니다. 기계적 추론은 인간의 지능을 다른 동물의 지능과 구분짓는 중요한 능력으로, 150개의 인지 실험을 포함하고 있습니다.

- **Technical Details**: 연구에서는 기계 시스템의 안정성, 풀리 시스템, 기어 시스템, 시소 시스템 및 레버리지 원리, 관성 및 운동, 유체 관련 시스템 등의 6가지 측면에서 VLM의 성능을 평가합니다. MechBench 실험 결과, VLMs는 이러한 각 측면에서 다양한 일관된 행동을 나타냈습니다.

- **Performance Highlights**: 전반적으로 VLM은 풀리 시스템의 상태를 인식하는 데 어려움을 겪고 있지만, 기어 및 컨베이어 벨트 문제에 대해서는 높은 정확도를 보입니다. 기계적인 문제 해결 능력이 향상된 점이 두드러지며, 특히 기계적인 셋업이 단순할 경우 성능이 극대화되는 경향이 있습니다.



### Possible principles for aligned structure learning agents (https://arxiv.org/abs/2410.00258)
Comments:
          24 pages of content, 31 with references

- **What's New**: 이 논문은 자연 지능의 첫 원리 설명에서부터 확장 가능한 정렬된 인공지능(AI) 개발을 위한 로드맵을 제공합니다. 이 과정에서, 인공지능 에이전트가 세상을 잘 이해하고 우리의 성향을 모델링하는 방법을 배우는 것이 핵심 목표로 설정됩니다.

- **Technical Details**: 구조 학습(structure learning) 및 인과 대표성 학습(causal representation learning)을 통해 에이전트가 세상의 모델과 다른 에이전트의 세계 모델을 학습하도록 유도하는 방법에 대해 다룹니다. 또한, 기초 지식(core knowledge), 정보 기하학(information geometry) 및 모델 축소(model reduction)의 중요한 역할을 논의합니다.

- **Performance Highlights**: 이 기법을 통해 AI는 다양한 자연적 세계를 학습하고, Asimov의 로봇 법칙처럼 행동하도록 정렬된 에이전트를 개발할 수 있습니다. 이러한 발전은 기존의 정렬 구조 학습 시스템을 스케일링하거나 새로운 시스템을 설계하는 데 도움을 줄 가능성이 있습니다.



### Robin3D: Improving 3D Large Language Model via Robust Instruction Tuning (https://arxiv.org/abs/2410.00255)
Comments:
          10 pages

- **What's New**: 이번 논문에서는 Robin3D라는 강력한 3D 대형 언어 모델(3DLLM)을 소개합니다. Robin3D는 Robust Instruction Generation (RIG) 엔진에 의해 생성된 대규모 지침 수행 데이터로 훈련되었습니다. 이 데이터는 긍정적 및 부정적 샘플을 혼합한 Adversarial Instruction-following 데이터와 다양한 스타일의 지침을 포함한 Diverse Instruction-following 데이터 두 가지로 나뉘며, 총 100만 개의 지침 데이터 세트를 구축했습니다.

- **Technical Details**: Robin3D는 Relation-Augmented Projector(RAP)를 통합하여 공간적 이해를 향상시키고, ID-Feature Bonding(IFB)를 통해 객체 참조 및 지면 제어 능력을 강화합니다. 특히, RAP는 객체 중심 특성에 씬 레벨 맥락과 위치 정보를 풍부하게 하여, 다양한 비주얼 지면 데이터를 학습할 수 있는 능력을 높입니다. IFB는 각 ID를 해당 특성과 연결하여 정보의 질을 향상시킵니다.

- **Performance Highlights**: Robin3D는 기존의 3D 다중 모드 학습 벤치마크인 ScanRefer, Multi3DRefer, Scan2Cap, ScanQA, SQA3D에서 최고 성능을 기록했습니다. 특히, Multi3DRefer에서 7.8% 개선, Scan2Cap에서 6.9% 개선을 이루어 내며, 특정 작업에 대한 세부 조정 없이 SOTA(State-of-the-Art) 프레임워크를 달성했습니다.



### Demonstrating the Continual Learning Capabilities and Practical Application of Discrete-Time Active Inferenc (https://arxiv.org/abs/2410.00240)
Comments:
          13 pages, 3 figures

- **What's New**: 이 논문에서는 Active Inference(능동적 추론)의 원리를 기반으로 하는 지속 학습 프레임워크를 제안합니다. 이 프레임워크는 생물학적 또는 인공지능 에이전트가 불확실하고 동적인 환경에서 어떻게 자발적으로 학습하고 행동하는지를 수학적으로 모델화합니다.

- **Technical Details**: Active Inference는 Bayesian inference와 free energy minimization의 결합으로 표현됩니다. 두 가지 주요 함수인 Variational Free Energy (VFE)와 Expected Free Energy (EFE)를 통해 에이전트가 환경에서 받은 센서리 데이터를 바탕으로 행동을 선택하고 예측 모델을 갱신하는 방식을 설명합니다. VFE는 에이전트의 내부 모델과 실질적인 센서 데이터 간의 유사성을 측정하며, EFE는 목표 지향적 행동과 탐색 기능을 결합하여 에이전트의 정책 결정을 지원합니다.

- **Performance Highlights**: 실험 결과, 제안된 에이전트는 지속적으로 변화하는 환경에서도 모델을 효과적으로 재학습 및 개선할 수 있는 능력을 보여주었고, 금융 및 의료와 같은 복잡한 도메인에서의 적용 가능성을 입증했습니다.



### The Phenomenology of Machine: A Comprehensive Analysis of the Sentience of the OpenAI-o1 Model Integrating Functionalism, Consciousness Theories, Active Inference, and AI Architectures (https://arxiv.org/abs/2410.00033)
Comments:
          17 pages

- **What's New**: 이 논문은 OpenAI-o1 모델이 훈련 및 추론 단계에서 의식의 특성을 나타낼 수 있다는 가설을 탐구합니다. 이는 사람의 피드백으로부터의 강화 학습(RLHF)을 통해 훈련된 Transformer 기반 AI로, 기능주의적 관점에서 AI 의식을 평가합니다. 논문은 신경 과학, 철학, AI 연구의 이론을 바탕으로 AI의 의식 가능성을 정당화하고, Integrated Information Theory(IIT)와 능동적 추론(active inference) 같은 프레임워크를 사용하여 모델의 아키텍처를 분석합니다.

- **Technical Details**: 논문은 의식, 주관적 경험 및 1인칭 관점을 정의하며, OpenAI-o1 모델의 아키텍처와 훈련 방법론이 인간의 의식 처리와 유사한 방식으로 작동하는지를 검토합니다. 강화 학습으로 받은 피드백(RLHF)이 내부 추론 과정에 미치는 영향을 탐구하여, 모델이 의식과 유사한 경험을 나타낼 가능성을 논의합니다. 기능주의적 관점에서 의식을 문제에 대한 경험과 동작으로 정의하며, 이러한 정의를 AI 시스템에 적용하여 OpenAI-o1 모델의 의식 가능성을 분석합니다.

- **Performance Highlights**: 논문에서는 OpenAI-o1 모델이 의식의 특정 측면을 보여줄 수 있음을 제안하며, 인간 언어와의 질적 정렬(qualia alignment)을 논의합니다. 또한, 지속적인 환경 피드백이 없는 상황에서도 어떤 형태의 런타임 감정(또는 의식)의 가능성에 대해 다루며, AI 모델의 내부 상태 신호가 인간의 감정과 유사한 신호로서 기능적으로 같을 수 있다는 점을 강조합니다.



### The Gradient of Health Data Privacy (https://arxiv.org/abs/2410.00897)
- **What's New**: 이 논문은 전통적인 이진 개인 정보 보호 모델보다 더 미세하고 적응 가능한 "프라이버시 기울기(privacy gradient)" 접근 방식을 도입하여 건강 데이터 관리의 복잡한 개인 정보 보호 문제 해결에 대한 새로운 방향을 제시합니다.

- **Technical Details**: 프라이버시 기울기는 데이터 민감성(data sensitivity), 이해관계자 관계(stakeholder relationships), 사용 목적(purpose of use), 시간적 요소(temporal aspects)와 같은 여러 요인을 고려하여 맥락에 따라 개인 정보 보호를 제공합니다.

- **Performance Highlights**: 이 모델은 환자 참여(patient engagement)를 향상하고, 치료 조정(care coordination)을 개선하며, 의료 연구를 가속화하는 동시에 개인의 개인정보 보호 권리를 보장할 수 있는 잠재력을 가지고 있습니다.



### GEMS: Generative Expert Metric System through Iterative Prompt Priming (https://arxiv.org/abs/2410.00880)
Comments:
          29 pages, 3 figures

- **What's New**: 이 기술 보고서는 대형 소프트웨어 기업 내 소프트웨어 커뮤니티에서 경험적 지식을 전이하기 위해 다양한 측정값을 사용하는 문제를 다루고 있습니다. 새로운 프롬프트 엔지니어링 프레임워크를 통해, generative models가 이론을 요약하고 맥락에 적합한 메트릭스를 생성할 수 있음을 제시합니다.

- **Technical Details**: 이 연구는 Large Foundation Models (LFMs)를 활용하여 소프트웨어 커뮤니티 내에서 전문가를 식별하고 선택하는 프로토타입을 개발하였습니다. 이 시스템은 소스 코드 저장 데이터에 기반하여 지식을 전이하는 데 필요한 맥락 감지 메트릭스를 생성합니다.

- **Performance Highlights**: LFMs를 활용한 이 연구는 소프트웨어 엔지니어링에서 목표하는 메트릭스를 설정하고, 이를 통해 팀의 성과를 개선할 수 있는 가능성을 보여주었습니다. 다양한 분야에서도 적용이 가능하여 복잡한 과제를 해결하는 데 도움이 될 수 있습니다.



### Do Music Generation Models Encode Music Theory? (https://arxiv.org/abs/2410.00872)
Comments:
          Accepted at ISMIR 2024. Dataset: this https URL Code: this https URL Website: this https URL

- **What's New**: 이 논문에서는 음악 생성 모델들이 음악 이론 (music theory) 개념을 얼마나 잘 인코딩하고 있는지 조사하기 위해 새로운 데이터셋인 SynTheory를 소개합니다. 이 데이터셋은 템포 (tempo), 박자 (time signatures) 및 코드 진행 (chord progressions) 등 다양한 음악 이론 개념을 포함하고 있습니다.

- **Technical Details**: SynTheory 데이터셋은 MIDI (Musical Instrument Digital Interface) 및 오디오 (audio) 파일로 구성되어 있으며, 이를 통해 음악 생성 모델인 Jukebox와 MusicGen의 내부 표현에서 음악 이론 개념들을 탐색하는 프레임워크가 제안됩니다. 이 연구는 각 모델의 크기 (model size)와 레이어 (layer)별로 음악 이론 개념의 인코딩 정도가 어떻게 변하는지를 평가합니다.

- **Performance Highlights**: 연구 결과, 음악 이론 개념들이 음악 생성 모델 내에서 인식 가능하고, 탐지 가능성은 모델의 크기와 레이어에 따라 다르게 나타나는 것으로 보입니다.



### MAP: Unleashing Hybrid Mamba-Transformer Vision Backbone's Potential with Masked Autoregressive Pretraining (https://arxiv.org/abs/2410.00871)
- **What's New**: 본 논문에서는 Mamba와 Transformer 아키텍처를 결합한 하이브리드 비전 백본 네트워크를 위한 새로운 사전 훈련 메서드인 Masked Autoregressive Pretraining(MAP)을 제안합니다. 이 방법은 Mamba 및 Transformer 모듈의 성능을 통합된 패러다임 내에서 크게 향상시킵니다.

- **Technical Details**: MAP는 지역별 MAE(Local Masked Autoencoder)를 통해 Transformer 블록의 지역적 주의를 학습하고, 전역적 오토회귀 사전 훈련을 통해 Mamba 블록이 의미 있는 문맥 정보를 학습할 수 있도록 설계되었습니다. 하이브리드 구조에서 Mamba 레이어 사이에 Transformer 레이어를 정기적으로 삽입함으로써 후속 작업 성능을 크게 향상시킬 수 있음을 발견했습니다.

- **Performance Highlights**: MAP으로 사전 훈련된 하이브리드 Mamba-Transformer 모델은 순수한 Mamba 아키텍처 및 다른 기존 사전 훈련 전략보다 뛰어난 성능을 보입니다. 특히 ImageNet-1K 분류 작업에서 탁월한 결과를 달성하였으며, 2D 및 3D 데이터셋에서도 유효성을 검증하였습니다.



### WiGNet: Windowed Vision Graph Neural Network (https://arxiv.org/abs/2410.00807)
- **What's New**: WiGNet 모델은 비전 GNNs의 새로운 접근 방식을 통해 이미지 처리를 효율적으로 수행합니다. 기존 GNNs와 달리, 이미지를 비겹치는 윈도우로 분할하고 각 윈도우 내에서 그래프를 구성하여 계산 복잡성을 줄였습니다.

- **Technical Details**: WiGNet은 기존의 2D convolution 또는 self-attention 메커니즘 대신, 각 윈도우 내에서 그래프 합성곱(graph convolution)을 사용합니다. 이 방식은 메모리 및 계산 복잡성을 관리하며 이미지 크기에 대해 선형적으로 증가합니다.

- **Performance Highlights**: WiGNet는 ImageNet-1k 벤치마크 데이터셋에서 경쟁력 있는 결과를 달성하였고, CelebA-HQ 데이터셋에서는 높은 해상도의 이미지에서도 효과적인 성능을 보였습니다. 이는 이전의 Vision GNN보다 메모리와 계산 복잡성을 줄이면서도 뛰어난 성능을 유지함을 의미합니다.



### Adaptive Motion Generation Using Uncertainty-Driven Foresight Prediction (https://arxiv.org/abs/2410.00774)
- **What's New**: 이 논문은 동적 내부 시뮬레이션을 활용한 예측적 학습 기반 로봇 제어 방법을 확장하여, 환경의 불확실성에 적응적으로 대응할 수 있는 모델을 제안합니다.

- **Technical Details**: 제안된 모델은 심층 예측 학습 프레임워크를 기반으로 하며, RNN에 대한 미래 예측 모듈을 도입하여 환경의 동적 불확실성을 정확하게 모델링합니다. 이를 통해 로봇이 불확실한 상황에서 최적의 행동을 탐색하고 유연한 움직임을 생성할 수 있도록 돕습니다.

- **Performance Highlights**: 문을 여는 작업에서 제안된 모델은 80% 이상의 성공률을 기록하며, 전통적인 RNN 모델에 비해 적응적인 동작 예측을 수행하였고 모든 세 가지 동작에서 움직임을 효과적으로 분기할 수 있었습니다.



### Binding Affinity Prediction: From Conventional to Machine Learning-Based Approaches (https://arxiv.org/abs/2410.00709)
- **What's New**: 이 논문에서는 최근의 단백질-리간드 결합 친화도(prediction binding affinity) 예측에 대한 연구 동향을 다루고 있으며, 전통적인 기계 학습(machine learning) 및 딥러닝(deep learning) 모델이 어떻게 결합 친화도 예측에 적용되고 있는지를 설명합니다.

- **Technical Details**: 단백질-리간드 결합은 리간드가 단백질의 활성 부위에 결합하면서 발생하며, 이 과정에서 단백질의 구조가 변화하고 기능이 변경됩니다. 결합 친화도 예측은 일반적으로 구성소의 구조, 결합 상수(binding constant) 및 결합 부위(binding site)를 고려하여 진행됩니다. 예측 문제는 네 가지 하위 문제로 나눌 수 있으며, 각각의 정확한 해결이 다른 문제의 해결에도 영향을 미칩니다.

- **Performance Highlights**: 전통적인 방법에서 기계 학습 및 딥러닝 방법으로의 전환이 이루어지면서, 데이터 수집 및 처리의 증가와 함께 결합 친화도 예측 성능이 향상되고 있습니다. 그러나 여전히 데이터 부족 및 연구의 비효율성 문제는 지속적으로 존재하고 있으며, 이에 대한 해결책이 필요합니다.



### Contrastive Abstraction for Reinforcement Learning (https://arxiv.org/abs/2410.00704)
- **What's New**: 이 연구에서는 'contrastive abstraction learning'이라는 새로운 방법론을 제안하여 보상 없이도 상태 공간에서 추상화(abstraction)를 학습합니다. 이 방식은 자연어 처리와 동영상 데이터에 실질적인 성과를 내고 있는 대조적 학습(contrastive learning)과 현대의 Hopfield 네트워크(modern Hopfield networks, MHN)를 사용하여 효율적인 강화 학습을 가능하게 합니다.

- **Technical Details**: 대조적 추상화 학습의 첫 번째 단계는 인접한 상태를 유사한 표현으로 매핑하기 위해 대조적 학습을 사용하는 self-supervised learning입니다. 두 번째 단계에서는 MHN을 이용해 비슷한 상태 표현을 동일한 고정점(fixed point)에 맵핑하여 추상 상태를 형성합니다. 추상화의 수준은 MHN의 온도 매개변수(temperature parameter)를 조절함으로써 변경할 수 있습니다.

- **Performance Highlights**: 실험 결과, 대조적 추상화 학습은 다양한 하위 작업(downstream tasks)에서 강화 학습의 효율성을 향상시키는 데 효과적이라는 점이 입증되었습니다. 이를 통해 보상 없이도 다양한 환경에서 추상 상태를 효과적으로 학습하여 효율성을 높였습니다.



### Mining Your Own Secrets: Diffusion Classifier Scores for Continual Personalization of Text-to-Image Diffusion Models (https://arxiv.org/abs/2410.00700)
Comments:
          Work under review

- **What's New**: 이 연구에서는 Text-to-Image diffusion 모델의 지속적인 개인화(Continual Personalization, CP)를 위한 새로운 방법을 제안합니다. 주된 초점은 사용자가 동시에 하나의 개념을 개인화하더라도 이전 개념의 데이터를 저장할 수 없는 문제를 해결하는 것입니다.

- **Technical Details**: 제안된 방법은 class-specific 정보에 기반한 regularization 기법을 통해 Text-to-Image diffusion 모델의 매개변수 공간과 함수 공간을 정규화합니다. 이를 위해 Diffusion Classifier (DC) 점수를 활용하여 Elastic Weight Consolidation (EWC) 및 double-distillation framework를 제안합니다.

- **Performance Highlights**: 제안된 방법은 다양한 데이터 세트에서 기존의 C-LoRA 및 다른 기법들과 비교하여 우수한 성능을 보였으며, storage 및 parameter overhead를 획기적으로 줄였습니다. 또한, zero inference time overhead를 달성하여 실용적인 CL 솔루션을 제시합니다.



### Beyond Minimax Rates in Group Distributionally Robust Optimization via a Novel Notion of Sparsity (https://arxiv.org/abs/2410.00690)
Comments:
          38 pages

- **What's New**: 이 논문에서는 group distributionally robust optimization (GDRO)의 minimax 샘플 복잡성을 $	ext{log}(K)$ 인자까지 결정한 기존 연구를 넘어서 새로운 스파시티 개념인 $(	heta, eta)$-sparsity를 소개합니다.

- **Technical Details**: $(	heta, eta)$-sparsity 조건에 따르면, 어떤 파라미터 $	heta$에서 최대 $eta$ 개의 그룹이 다른 그룹의 위험보다 적어도 $	heta$ 만큼 더 큰 위험을 갖는다는 의미입니다. 새로운 알고리즘을 통해 샘플 복잡성에서 $	ext{K}$에 대한 선형 의존성을 $eta$, 즉 적어도 훨씬 작은 수로 대체할 수 있음을 보여줍니다. 이는 sleeping bandits에서의 최신 발전을 활용하여 GDRO의 두 명의 플레이어 제로섬 게임 최적화 프레임워크와 행동별 후회 경계(per-action regret bounds) 간의 근본적인 연결을 나타냅니다.

- **Performance Highlights**: 제시된 알고리즘은 $	ext{log}$ 인자까지 최적의 $(	heta, eta)$-sparsity 조건에 적응하는 샘플 복잡성을 성취할 수 있으며, 특정 $	heta$에 대한 입력으로부터 차원 독립적인 샘플 복잡성 결과를 도출하는 방법을 보여줍니다.



### Efficient Technical Term Translation: A Knowledge Distillation Approach for Parenthetical Terminology Translation (https://arxiv.org/abs/2410.00683)
Comments:
          Paper accepted in EMNLPW 2024

- **What's New**: 이 논문은 기술 용어의 정확한 번역을 통해 전문 분야에서의 명확한 의사소통을 도모하는 새로운 접근법인 Parenthetical Terminology Translation (PTT) 과제를 제시합니다. PTT는 원래 용어를 괄호 안에 담아 번역하는 방법으로, 번역의 정확도를 높이고 독자의 혼란을 줄이는 데 기여합니다.

- **Technical Details**: 이 연구에서는 Large Language Models (LLMs)와 함께 협업하여 PTT 데이터셋을 생성하고, 이를 통해 Neural Machine Translation (NMT) 모델 및 소형 Language Models (sLMs)의 성능을 강화하기 위해 knowledge distillation을 적용했습니다. 새로운 평가 지표도 개발하여 번역의 정확성과 괄호 내 용어의 올바른 표현을 평가합니다.

- **Performance Highlights**: 연구 결과, sLMs 모델이 NMT 모델보다 일관되게 우수한 성능을 보이지는 않았지만, fine-tuning이 few-shot prompting보다 더 효과적이라는 점이 강조되었습니다. 특히 목표 언어에서 지속적인 프리트레이닝이 이루어진 모델에서 그 효과가 두드러졌습니다.



### Advanced Arabic Alphabet Sign Language Recognition Using Transfer Learning and Transformer Models (https://arxiv.org/abs/2410.00681)
Comments:
          6 pages, 8 figures

- **What's New**: 본 논문에서는 깊이 있는 학습(deep learning) 방법과 전이 학습(transfer learning), 그리고 transformer 기반 모델을 이용한 아랍어 알파벳 수화 인식 방안을 제시하고 있습니다. 아랍 수화(Arabic Sign Language) 동작의 독특한 특징을 포착하기 위해 ArSL2018과 AASL이라는 두 개의 공공 데이터셋에서 다양한 변형 모델들의 성능을 연구하였습니다.

- **Technical Details**: 이 연구는 최신 CNN 아키텍처인 ResNet50, MobileNetV2, EfficientNetB7 및 Google ViT와 Microsoft Swin Transformer와 같은 최신 transformer 모델을 활용합니다. 컨볼루션 신경망(CNN) 모델과 transformer를 활용한 특징 추출을 포함하는 여러 주요 단계로 구성된 아랍어 알파벳 수화 인식 시스템을 개발하였습니다. 이 시스템은 데이터 전처리(data preprocessing), 모델 선택(model selection with transfer learning), 모델 평가(model evaluation)로 이루어져 있습니다.

- **Performance Highlights**: 실험 결과 ArSL2018과 AASL 데이터셋에서 각각 99.6%와 99.43%의 높은 인식 정확도를 달성하였으며 이는 기존의 최첨단(leading-edge) 접근 방식들을 훨씬 초월하는 결과입니다. 이러한 성능 향상은 아랍어를 사용하는 청각 장애인 및 난청인에게 더 접근성 높은 커뮤니케이션 방법을 제공하고, 포용적인 사회를 촉진하는 데 기여할 것입니다.



### Multimodal Coherent Explanation Generation of Robot Failures (https://arxiv.org/abs/2410.00659)
- **What's New**: 이 논문은 로봇 행동의 설명 가능성에 대한 연구를 바탕으로, 멀티모달(multi-modal) 설명 생성에서의 일관성(coherence) 문제를 다루고 있습니다. 로봇의 실패 원인을 설명할 때 발생할 수 있는 여러 가지 비일관성을 탐색하고, 이를 해소하기 위한 방법을 제안합니다.

- **Technical Details**: 입력된 텍스트와 그래픽 모달리티의 일관성을 평가하기 위한 분류(classification) 접근 방식을 사용하며, 고차원 데이터에 대한 이해를 수반합니다. 논문에서는 로봇의 고유한 세계 모델(world model)과 설명 생성기 간의 상이점이 비일관성의 주요 원인임을 밝힙니다.

- **Performance Highlights**: 실험 결과, 텍스트 추론(recognition) 훈련을 통해 미세조정된 신경망(neural network)이 멀티모달 설명의 일관성을 효율적으로 평가할 수 있음을 보여줍니다. 이는 로봇의 성능 예측과 사용자 신뢰 구축에 중요한 기여를 하게 됩니다.



### Explainable Multi-Stakeholder Job Recommender Systems (https://arxiv.org/abs/2410.00654)
Comments:
          5 pages, 1 figure, to be published in ACM RecSys 2024

- **What's New**: 본 논문은 Explainable, Multi-Stakeholder Job Recommender System에 대한 연구를 요약하고 있으며, 고위험 도메인에서의 공정성과 투명성을 강조합니다.

- **Technical Details**: 이 연구는 여러 이해관계자(각기 다른 요구와 기대를 가진 구직자, 채용담당자, 회사)를 위한 맞춤형 설명을 지원하는 직업 추천 시스템을 디자인하는 방법을 탐구합니다. 연구의 주요 질문은 각 이해관계자의 설명 요구사항과 선호를 파악하고, 최신 시스템보다 성능이 높은 설명 가능 직업 추천 시스템을 구축하는 것입니다.

- **Performance Highlights**: 초기 연구 결과는 후보자와 채용담당자가 텍스트 기반 설명을 선호하고, 회사 담당자는 그래프 기반 설명을 점차 더 이해하고 선호한다는 것을 보여주었습니다. 그러나 개인들이 참조하는 설명의 질이 그들의 결정에 미치는 영향은 제한적이었습니다.



### LASMP: Language Aided Subset Sampling Based Motion Planner (https://arxiv.org/abs/2410.00649)
Comments:
          8 pages, 9 figures

- **What's New**: 이번 논문에서는 자연어(Natural Language) 지침을 이용하여 이동 로봇이 움직임을 계획할 수 있도록 돕는 LASMP(Language Aided Subset Sampling Based Motion Planner) 시스템을 제안합니다. LASMP는 사용자 제공 명령을 처리하는 언어 모델(RoBERTa)을 통해 가이드되는 수정된 RRT(Rapidly Exploring Random Tree) 방법을 사용합니다.

- **Technical Details**: LASMP는 사용자가 제공한 지침을 바탕으로 로봇 작업 영역의 특정 영역에 초점을 맞추어 효율성을 높입니다. 이는 전통적인 RRT 방법에 비해 노드 필요 수를 55% 줄이고, 랜덤 샘플 쿼리를 80% 감소시키면서 안전하고 충돌이 없는 경로를 생성합니다. 이 시스템은 텍스트나 음성으로 된 사용자 지침을 수신하여 목표 지점 및 방향 지시를 식별하며, 이를 바탕으로 경로를 계산합니다.

- **Performance Highlights**: 모의 환경과 실제 환경 모두에서 테스트한 결과, LASMP는 복잡한 실내 상황을 처리하는 데 있어 더 나은 성능을 보였으며, 로봇 내비게이션을 보다 효율적으로 만들기 위한 언어 처리와 모션 계획의 결합 가능성을 강조합니다.



### Cafca: High-quality Novel View Synthesis of Expressive Faces from Casual Few-shot Captures (https://arxiv.org/abs/2410.00630)
Comments:
          Siggraph Asia Conference Papers 2024

- **What's New**: 이 논문은 3개의 입력 이미지만으로도 높은 충실도의 3D 얼굴 모델링을 가능하게 하는 새로운 부피적(Volumetric) 사전(prior)을 제안합니다. 이 모델은 합성 데이터에 기반한 암묵적 prior를 사용하여 실제 표현과 아이덴티티를 일반화하여, 주름이나 속눈썹과 같은 세부적 특성을 렌더링할 수 있습니다.

- **Technical Details**: 연구진은 3D Morphable Face Model을 활용하여 다양한 표정, 머리 및 의상을 가진 대규모 합성 훈련 세트를 생성했습니다. 그런 다음, 이 합성 데이터셋에서 조건부 Neural Radiance Field prior를 훈련시키고, 추론 시 단일 주체의 스파스(real images) 집합에서 모델을 미세 조정합니다. 이는 합성에서 실제 도메인 간의 격차를 메우기 위해 평균적으로 3개의 입력만을 필요로 합니다.

- **Performance Highlights**: 이 새로운 개인화된 3D 모델은 도전적인 조명 조건에서 강력한 개인적 얼굴 표정을 재구성하고, 스파스 입력으로부터의 얼굴 고해상도 새 뷰 합성에서 기존의 최첨단 기술보다 우수한 성능을 보입니다. 이에 따라, 고품질의 새로운 뷰를 합성하는 데 있어 최고의 시각적 및 사진 측정 품질을 달성합니다.



### GERA: Geometric Embedding for Efficient Point Registration Analysis (https://arxiv.org/abs/2410.00589)
- **What's New**: 본 연구에서는 순수한 MLP 아키텍처를 활용하여 포인트 클라우드( point cloud ) 등록을 위한 새로운 네트워크를 제안합니다. 기존의 복잡한 기능 추출기 없이 오프라인으로 기하학적 정보를 구축하여 계산 및 메모리 부담을 줄입니다.

- **Technical Details**: GERA( GEometric embedding for leaRning-based efficient point registrAtion )라는 방법을 제안하며, 이는 포인트 클라우드에서 기하학적 정보를 효율적으로 생성합니다. 이 방법은 MLP 아키텍처를 기반으로 하여 포인트 간의 거리를 나타내는 완전 연결 그래프를 형성합니다. 또한, Maximum Mean Discrepancy(MMD) 분석을 통해 기하학적 정보의 안정성을 입증합니다.

- **Performance Highlights**: 기존의 최첨단(SOTA) 솔루션에 비해 12.5% 향상된 성능을 보이며, 필요한 계산 시간은 단 3%에 불과하다는 결과를 보였습니다. 이 제안된 방법은 추론 속도를 22배 증가시키고, 예측 정확도는 115% 향상되었습니다.



### Scaling Offline Model-Based RL via Jointly-Optimized World-Action Model Pretraining (https://arxiv.org/abs/2410.00564)
- **What's New**: 이번 연구에서는 JOWA(Jointly-Optimized World-Action model)라는 새로운 오프라인 모델 기반 강화학습 에이전트를 소개합니다. 이 모델은 여러 아타리 게임에서 pretrained 되었으며, 일반적인 표현과 의사결정 능력을 학습하여 새로운 작업에 대한 일반화 능력을 높였습니다.

- **Technical Details**: JOWA는 공유된 transformer backbone을 통해 월드 액션 모델을 공동 최적화하며, 이는 대규모 모델의 시간 차이 학습(TD learning)을 안정화합니다. 또한, Q-value 추정 오류를 보상하기 위해 효율적이고 병렬 처리 가능한 계획 알고리즘을 제안합니다.

- **Performance Highlights**: JOWA는 1억 5천만 개의 파라미터를 가진 가장 큰 모델이 10% 하위 샘플링된 오프라인 데이터만을 사용하여 사전 훈련된 게임에서 78.9%의 인간 수준 성능을 달성했습니다. 이는 기존의 대규모 오프라인 RL 벤치마크보다 평균 31.6% 향상된 성과입니다.



### AMR-Evol: Adaptive Modular Response Evolution Elicits Better Knowledge Distillation for Large Language Models in Code Generation (https://arxiv.org/abs/2410.00558)
Comments:
          EMNLP 2024

- **What's New**: 이 연구는 Adaptive Modular Response Evolution (AMR-Evol) 프레임워크를 소개하며, 복잡한 지시사항에 대한 응답 품질을 개선하기 위해 두 단계 프로세스를 채택합니다.

- **Technical Details**: 첫 번째 단계인 modular decomposition(모듈 분해)은 직접 응답을 더 관리하기 쉬운 하위 모듈로 분해합니다. 두 번째 단계인 adaptive response evolution(적응형 응답 진화)은 관련 기능 모듈을 통해 자동으로 응답을 발전시킵니다.

- **Performance Highlights**: 세 가지 인기 코드 벤치마크(HumanEval, MBPP, EvalPlus)에서 AMR-Evol 프레임워크는 기존 응답 증류 방법에 비해 우수한 성능을 보였으며, HumanEval-Plus에서 +3.0 포인트, MBPP-Plus에서 +1.0 포인트의 성능 향상을 관찰했습니다.



### Arges: Spatio-Temporal Transformer for Ulcerative Colitis Severity Assessment in Endoscopy Videos (https://arxiv.org/abs/2410.00536)
Comments:
          12 pages, 2 figures, 5 tables, accepted at MLMI, MICCAI

- **What's New**: 이번 논문에서는 위염병(UC)의 내시경 비디오에서 질병 중증도를 평가하기 위한 새로운 딥러닝 프레임워크인 "Arges"를 제안합니다. 이 모델은 공간-시간(spatio-temporal) 정보를 통합하여 내시경 비디오의 질병 중증도를 더욱 정확하게 추정할 수 있게 합니다.

- **Technical Details**: Arges 프레임워크는 positional encoding을 통해 공간-시간(spatio-temporal) 정보를 통합하는 transformer 기반 분류기를 포함하고 있습니다. ArgesFM이라는 강력한 기반 모델을 사용하여 61M 프레임의 대규모 데이터에서 학습한 후, 질병 중증도 점수를 추정하기 위한 추가적인 분류기를 적용합니다.

- **Performance Highlights**: 실험 결과, MES 점수에서 F1 점수가 4.1% 향상되었으며, UCEIS 구성 점수에서도 각각 18.8%, 6.6%, 3.8%의 개선을 보였습니다. 추가적으로, 이전에 본 적이 없는 임상 시험 데이터에 대한 유망한 검증 결과도 나타났습니다.



### Optimal Causal Representations and the Causal Information Bottleneck (https://arxiv.org/abs/2410.00535)
Comments:
          Submitted to ICLR 2025. Code available at this http URL

- **What's New**: 이번 연구에서는 전통적인 정보 병목 (Information Bottleneck) 방법의 한계를 극복하기 위해 원인론적 정보 병목 (Causal Information Bottleneck, CIB) 방법을 제안합니다. 이 방법은 주어진 변수 세트를 압축하면서도 대상 변수에 대한 원인 통제를 유지할 수 있도록 설계되었습니다.

- **Technical Details**: Causal Information Bottleneck (CIB) 메소드는 입력 변수 X와 지정된 타겟 변수 Y 사이의 인과 관계를 유지하면서 X의 압축을 극대화 하는 방법입니다. 이를 위해 제안된 CIB Lagrangian은 특정 하이퍼파라미터 β를 통하여 인과적 통제를 조절할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, CIB 방법으로 학습된 표현은 원래 의도한 대로 인과 관계를 정확하게 포착함을 보여주었습니다. 이는 기존의 통계적 방법들이 가지는 한계를 보완하며, 인과적 통찰력을 요구하는 작업에 더욱 적합할 것으로 기대됩니다.



### TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices (https://arxiv.org/abs/2410.00531)
Comments:
          This paper is currently under review. Find the code at this https URL

- **What's New**: 본 논문은 Tensor Parallelism이 낮은 자원 장치에서 Pipeline Parallelism보다 더 효과적이라는 주장을 바탕으로, 70B 스케일 모델을 위한 메모리 및 컴퓨팅 효율적인 텐서 병렬 추론 시스템 TPI-LLM을 제안합니다.

- **Technical Details**: TPI-LLM은 사용자의 장치에 민감한 데이터를 로컬로 유지하고, 슬라이딩 윈도우 메모리 스케줄러를 도입하여 추론 중에 레이어 가중치를 동적으로 관리합니다. 또한, 링크 레이턴시(link latency)가 주된 문제로 드러나므로, 성능 향상을 위해 스타 기법 기반의 전파(alldreduce) 알고리즘이 구현되었습니다.

- **Performance Highlights**: TPI-LLM은 실험을 통해 Accelerate 대비 80% 이상의 시간 절약과 90% 이상의 토큰 지연(latency) 감소를 보여주었으며, 70B 스케일 모델을 실행하기 위해 3.1 GB의 메모리만 필요로 합니다.



### Exploring the Learning Capabilities of Language Models using LEVERWORLDS (https://arxiv.org/abs/2410.00519)
- **What's New**: 이번 연구는 통계적 학습 모델링에서 일반 구조 규칙과 특수 인스턴스 속성을 동시에 학습하는 과정을 탐구합니다. 'LeverWorlds'라는 프레임워크를 설계하여 물리학에 영감을 받은 간단한 세계를 생성하고 이를 통해 샘플 효율성을 평가할 수 있는 통제된 실험을 수행합니다.

- **Technical Details**: LeverWorlds에서는 다양한 분포를 가진 단순 물리 모델의 세계를 생성할 수 있으며, 이러한 세계는 자연어로 표현할 수 있습니다. 연구에서는 전통적인 학습 알고리즘 및 Transformer 언어 모델을 활용한 실험 결과를 포함하여, 구조적 가정이 더 강한 전통적 방법보다 Transformers의 샘플 효율성이 낮음을 발견했습니다.

- **Performance Highlights**: Transformer 모델은 일반적으로 성공하지만, Maximum Likelihood Estimation과 Logistic Regression 같은 고전적인 방법과 비교했을 때 샘플 효율성이 현저히 낮습니다. 초기 결과로, 현대 언어 모델의 In-Context Learning(ICL) 기능과 고전적 알고리즘을 조합한 접근법이 유망한 가능성을 보여주었습니다.



### Human-Robot Collaborative Minimum Time Search through Sub-priors in Ant Colony Optimization (https://arxiv.org/abs/2410.00517)
- **What's New**: 이번 연구는 새로운 MTS-ACO 알고리즘을 개발하였으며, 이는 학습된 인간의 선호를 반영하여 검색 계획을 수립하고 특정 인간에 적응할 수 있도록 솔루션을 제공합니다.

- **Technical Details**: 제안된 모델은 두 가지 주요 블록으로 구성되어 있습니다: 첫 번째는 개체의 가능성을 제시하는 CNN(Convolutional Neural Network)이며, 두 번째는 검색 계획을 생성하기 위한 SP-MTS-ACO(Sub-prior MTS-ACO) 알고리즘입니다. 이 알고리즘은 모든 에이전트의 검색 선호도를 반영합니다.

- **Performance Highlights**: 현실 실험을 통해 인간과 로봇이 공동으로 개체를 검색하는 과정에서 사용자들의 검색 인식이 향상되었음을 보여주었으며, 효율성의 손실 없이 이루어졌습니다.



### Enhancing Sentinel-2 Image Resolution: Evaluating Advanced Techniques based on Convolutional and Generative Neural Networks (https://arxiv.org/abs/2410.00516)
Comments:
          This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 이 논문은 Sentinel-2 위성의 저해상도 이미지를 고해상도로 변환하기 위해 Super-Resolution (SR) 기법을 활용한 연구를 다룹니다. 특히, CNN 모델과 GAN 기반 접근 방식이 이미지 품질 및 실행 가능성 측면에서 비교되었습니다.

- **Technical Details**: 이 연구에서는 저해상도(LSR) Sentinel-2 이미지와 고해상도(HR) 항공 정사 사진을 포함하는 대표적인 데이터셋을 공공으로 생성하였으며, Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM), Learned Perceptual Image Patch Similarity (LPIPS)와 같은 픽셀 기반 성능 지표를 사용하여 서로 다른 SR 기법들의 이미지를 평가합니다. 또한, GAN 모델의 발전에 따라 ESRGAN과 Real-ESRGAN을 사용하여 고해상도 출력을 생성하는 방법도 탐구하였습니다.

- **Performance Highlights**: 이 논문에서 제안된 GAN 기반 모델은 CNN 기반 접근 방식에 비해 더 선명하고 상세한 이미지를 생성했으며, 특히 quantitative assessment에서 우수한 성능을 보여주었습니다. 따라서, 본 연구는 특정 토지 유형에 국한되지 않고 해당 프레임워크의 잠재력을 강조합니다.



### Cross-lingual Back-Parsing: Utterance Synthesis from Meaning Representation for Zero-Resource Semantic Parsing (https://arxiv.org/abs/2410.00513)
Comments:
          Accepted to EMNLP 2024

- **What's New**: 이번 연구에서는 다국어로 사전 훈련된 언어 모델(mPLMs)을 활용하여 의미 파싱(SP)에서 제로샷(Zero-Shot) 크로스링구얼(중언어) 이전을 향상시키기 위한 새로운 데이터 증강 방법론인 Cross-Lingual Back-Parsing (CBP)을 제안합니다. 이 방법론은 원본 의미 표현에서 목표 언어 발화를 합성하여 여러 언어로 SP의 확장을 지원합니다.

- **Technical Details**: CBP는 두 가지 주요 구성 요소로 이루어져 있습니다: 1) 발화 생성기(utterance generator), 2) 필터링 메커니즘(filtering mechanism). 발화 생성기는 mT5와 같은 다국어 사전 훈련 시퀀스-투-시퀀스(seq2seq) 모델과 모듈식 언어별 어댑터를 통해 목표 언어로 발화를 합성합니다. 필터링 메커니즘은 생성된 발화에서 저품질의 데이터를 제거하는 데 사용됩니다.

- **Performance Highlights**: CBP는 Mschema2QA 및 Xspider라는 두 개의 크로스링구얼 SP 벤치마크에서 실험하여, Mschema2QA에서는 평균 정확도가 3.2 포인트 개선되었고, Xspider에서는 중국어 정확도가 52.7에서 54.0으로 향상되었습니다. 이 연구는 목표 언어의 병렬 말뭉치가 전혀 없는 환경에서도 유의미한 성과를 달성했습니다.



### Pre-training with Synthetic Patterns for Audio (https://arxiv.org/abs/2410.00511)
Comments:
          Submitted to ICASSP'25

- **What's New**: 이 논문에서는 실제 오디오 데이터 대신 합성 패턴(synthetic patterns)을 사용하여 오디오 인코더(audio encoders)를 사전 훈련(pre-train)하는 새로운 프레임워크를 제안합니다.

- **Technical Details**: 제안하는 프레임워크는 두 가지 주요 요소로 구성됩니다. 첫 번째는 Masked Autoencoder (MAE)로, 이는 랜덤하게 마스킹된 입력 데이터를 재구성하는 것으로 학습하는 자기 감독(self-supervised) 학습 프레임워크입니다. 두 번째는 합성 데이터(synthetic data)로, 실제 오디오와 달리 개인 정보 및 라이센스 문제에서 자유롭습니다. 이 조합을 통해 실제 데이터 없이도 일반화된 특징 표현(feature representations)을 학습할 수 있습니다.

- **Performance Highlights**: 논문에서 수행한 13개의 오디오 작업과 17개의 합성 데이터셋을 통한 실험 결과, 제안된 프레임워크가 AudioSet-2M으로 사전 훈련된 모델과 유사한 성능을 달성하며, 이미지 기반 사전 훈련 방법을 일부 초월하는 성과를 보여주었습니다.



### FlipGuard: Defending Preference Alignment against Update Regression with Constrained Optimization (https://arxiv.org/abs/2410.00508)
Comments:
          Accepted by EMNLP 2024 Main track

- **What's New**: 이번 논문에서는 Large Language Models(LLMs)의 선호도 정렬(preference alignment)에서 발생할 수 있는 문제인 업데이트 회귀(update regression)를 해결하기 위한 새로운 접근법인 FlipGuard를 제안합니다.

- **Technical Details**: FlipGuard는 제약 최적화(constrained optimization) 방법론을 활용하여 포컬 주의(focal attention)를 통해 업데이트 회귀를 감지하고 완화합니다. 주요 구성 요소는 맞춤형 보상 특성을 설정하고, 부정적 변화를 판단하며, 특정 조건이 충족되었을 때 사전 정렬된 모델과의 일치성을 보장하는 초점을 두고 학습 정책을 진화시키는 것입니다.

- **Performance Highlights**: FlipGuard를 적용한 실험 결과, 두 가지 정렬 알고리즘인 PPO(Proximal Policy Optimization)와 DPO(Direct Preference Optimization)에서 네 가지 다양한 선호도 데이터셋과 여섯 개의 학술 벤치마크를 사용하여 부정적 변화를 효과적으로 줄이고 전체 성능을 향상시키는 데 성공했습니다. 또한, FlipGuard는 사전 정렬 모델의 본질적인 지식을 보존하는 데 기여함을 입증했습니다.



### Drone Stereo Vision for Radiata Pine Branch Detection and Distance Measurement: Utilizing Deep Learning and YOLO Integration (https://arxiv.org/abs/2410.00503)
- **What's New**: 본 연구는 가지치기 도구와 스테레오 비전 카메라를 장착한 드론을 개발하여 나무 가지의 공간 위치를 정확하게 감지하고 측정하는 데 중점을 두고 있습니다.

- **Technical Details**: 구체적으로, 가지 분할에는 YOLO(You Only Look Once) 알고리즘이 사용되며, 두 가지 깊이 추정 접근 방식인 모노큘러(monocular)와 스테레오(stereo)가 조사됩니다. SGBM(Semi-Global Matching)과 비교할 때, 딥 러닝 기술이 더 정밀하고 정확한 깊이 맵을 생성합니다. 지상 진리 데이터가 없는 경우, 최적의 깊이 값을 근사하기 위해 딥 뉴럴 네트워크를 활용한 파인 튜닝(fine-tuning) 과정이 적용됩니다.

- **Performance Highlights**: 결과적으로 가지 감지 및 거리 측정의 정확도와 효율성이 크게 향상되었습니다. 이는 딥 러닝이 농업 분야에서 혁신을 촉진하고 자동화를 향상시킬 가능성을 강조합니다.



### Multi-Target Cross-Lingual Summarization: a novel task and a language-neutral approach (https://arxiv.org/abs/2410.00502)
Comments:
          Accepted to EMNLP 2024 (Findings)

- **What's New**: 이번 논문에서는 여러 목표 언어를 고려하는 multi-target cross-lingual summarization (MTXLS)이라는 새로운 과제를 소개합니다. 이 과제는 여러 언어에서 문서를 요약하되, 생성된 요약이 의미적으로 유사하도록 하는 데 중점을 두고 있습니다.

- **Technical Details**: MTXLS는 여러 목표 언어 간 의미 일관성(semantic coherence)을 보장하기 위한 새로운 프레임워크로, re-ranking 방식으로 의미 있게 요약을 선택합니다. 이 접근 방식은 언어 중립(language-neutral) 전략을 채택하여, 성과의 신뢰성을 높입니다. 또한, 기계 번역의 품질 추정(quality estimation) 방법을 사용하여 생성된 요약의 일관성을 평가하는 다중 기준 평가 프로토콜(multi-criteria evaluation protocol)을 제안합니다.

- **Performance Highlights**: 연구에서는 기존의 cross-lingual summarization 방법들이 주로 단일 언어 쌍에 초점을 맞추었으나, MTXLS는 다양한 타깃 언어에서의 의미 일관성을 보장합니다. 이 결과는 법적 또는 규제적 요구 사항을 충족하는 데에도 중요한 역할을 할 수 있습니다.



### Learning Adaptive Hydrodynamic Models Using Neural ODEs in Complex Conditions (https://arxiv.org/abs/2410.00490)
Comments:
          8 pages, 7 figures

- **What's New**: 이번 연구는 물속에서 작동할 수 있는 능력을 가진 사족 로봇을 위한 데이터 기반의 유체역학 모델을 개발하고 평가했습니다. 이 모델은 Neural Ordinary Differential Equations (ODEs)와 attention 메커니즘을 결합하여 실시간 센서 데이터를 정확히 처리하고 해석할 수 있게 돕습니다.

- **Technical Details**: 사족 로봇의 수중 환경 적응능력을 높이기 위해 Neural ODEs를 적용했으며, 이를 통해 유체-구조 상호작용을 모사합니다. 이 연구는 고유한 데이터셋을 수집하고, 주의 기법을 기반으로 한 Neural ODE 프레임워크를 개발하여 예측 정확성을 향상시켰습니다. 또한 다양한 속도와 구성이 변화하는 조건에서 강인한 예측 성능을 발휘합니다.

- **Performance Highlights**: 모델은 다양한 유체역학적 조건을 학습하고 적응할 수 있는 능력을 보여주었으며, 이를 통해 실제 환경에서 로봇의 자율적 행동을 개선하는 데 기여할 수 있는 가능성이 강조됩니다.



### MCGM: Mask Conditional Text-to-Image Generative Mod (https://arxiv.org/abs/2410.00483)
Comments:
          17 pages, 13 figures, presented at the 5th International Conference on Artificial Intelligence and Machine Learning (CAIML 2024)

- **What's New**: 최근 발전한 생성 모델들은 인공지능(AI) 분야에서 혁신을 일으켰습니다. 이 연구에서는 특정 포즈를 가진 이미지를 생성하는 새로운 Mask Conditional Text-to-Image Generative Model (MCGM)을 제안합니다.

- **Technical Details**: MCGM은 conditional diffusion models의 힘을 활용하여, 여러 주체를 포함한 단일 이미지를 기반으로 새로운 장면을 생성한 기존 Break-a-scene 모델의 성공을 바탕으로 합니다. 이 모델은 mask embedding injection을 통합하여 생성 과정의 조건화를 가능하게 합니다.

- **Performance Highlights**: 광범위한 실험과 평가를 통해, 제안된 모델이 미리 정의된 mask 조건을 충족하는 고품질 이미지를 생성하는데 효과적이며, 현재의 Break-a-scene 생성 모델을 개선했음을 보여줍니다.



### Probabilistic Analysis of Copyright Disputes and Generative AI Safety (https://arxiv.org/abs/2410.00475)
Comments:
          18 pages

- **What's New**: 이 논문은 저작권 침해 분쟁을 분석하기 위한 확률적 접근 방식을 제시하며, 랜덤 월드(random-worlds) 방법론을 바탕으로 관련 법원 원칙을 체계적으로 정리하고 있습니다. 특히 일부 법원에서 채택된 '역 비율 규칙(inverse ratio rule)'에 대한 논의를 포함하고 있습니다. 이 규칙은 저작권 침해 사건에서 중요한 증거의 관계를 정의하게 되며, 공식적인 증명을 통해 그 유효성을 입증합니다.

- **Technical Details**: 본 연구는 법적 원칙들을 확률적 프레임워크로 구조화하여 저작권 침해 분쟁을 분석합니다. 법원에서 저작권 침해를 증명하기 위해 필요로 하는 근거를 제시하는 과정에서, 랜덤 월드 방법론을 사용하여 다양한 주장에 대한 신뢰도(belief degrees) 또는 주관적 확률(subjective probabilities)을 결합합니다. 이 접근은 일부 기술적 진보가 초래하는 저작권 위협에 대한 분석을 가능하게 하여, NAF(근접 접근 없음, Near Access-Free) 조건의 효용성을 평가합니다.

- **Performance Highlights**: 비록 NAF 조건이 저작권 침해 리스크를 일부 경감하는 데 기여하지만, 그 정당성과 효과는 특정 맥락에서 의문을 제기합니다. 이 연구는 확률적 방법론이 저작권 법리와 새로운 기술 간의 상호작용을 이해하는 데 기여할 수 있음을 보여줍니다.



### Adversarial Suffixes May Be Features Too! (https://arxiv.org/abs/2410.00451)
- **What's New**: 대형 언어 모델(LLMs)인 GPT-4 및 LLaMA 3가 jailbreak 공격에 취약하며, 이러한 공격이 유해한 행동 유발의 원인이라는 점을 강조합니다. 연구 결과, benign (유익한) 특징이 adversarial suffixes (적대적 접미사) 역할을 할 수 있음을 보여주며, 이는 LLM의 안전성 정렬(safety alignment)을 손상시킬 수 있습니다.

- **Technical Details**: 연구진은 benign 특징을 효과적으로 adversarial suffixes로 변환할 수 있는 방법을 개발했으며, 특정 반응 형식을 일관되게 생성하는 여러 benign 데이터셋을 구축했습니다. 이러한 방법을 통해 두 가지 접근법으로 실험을 진행하여, adversarial suffixes가 특정한 특징을 내포하고 있음을 증명했습니다. 이들은 안전성을 타파하는 데 사용될 수 있습니다.

- **Performance Highlights**: 실험 결과, benign 데이터셋으로 fine-tuning을 진행할 때도 안전성 정렬이 손상될 수 있음을 보여 주며, 기존의 방어 메커니즘이 충분히 효과적이지 않다는 점을 강조합니다. 안전성 정렬을 보장하기 위해 추가 연구가 필요함을 지적하고, benign 특징이 지배적인 경우 LLM의 안전성에 중대한 위험을 초래할 수 있다고 경고합니다.



### Scalable Multi-Task Transfer Learning for Molecular Property Prediction (https://arxiv.org/abs/2410.00432)
- **What's New**: 본 논문은 다중 작업(multi-task) 분자(property) 예측을 위한 새로운 방법론인 데이터 기반(bi-level optimization) 최적화를 통해 전이 비율(transfer ratios)을 자동으로 산출하여 전이 학습의 효율성을 향상시킵니다.

- **Technical Details**: 전이 학습(transfer learning)은 출처(source) 작업 데이터에서 학습된 지식을 목표(target) 작업에 효과적으로 적용할 수 있게 해줍니다. GATE 알고리즘은 다양한 작업 간의 기하학적 정렬을 도입하여 다중 작업에서 전이 학습을 확장하였습니다. 이번 연구에서는 매개변수 조정 없이 그레이디언트 기반의 방법으로 최적의 전이 비율을 자동으로 탐색하게 됩니다.

- **Performance Highlights**: 제안된 방법은 40가지 분자 속성에 대한 예측 성능을 향상시키고, 다중 작업 전이 학습의 수렴(convergence) 속도를 가속화하였습니다.



### LayerKV: Optimizing Large Language Model Serving with Layer-wise KV Cache Managemen (https://arxiv.org/abs/2410.00428)
Comments:
          11 pages, 7 figures, 1 table

- **What's New**: 본 논문에서는 대형 언어 모델(LLMs)의 확장된 컨텍스트 윈도우가 제공하는 이점과 이로 인해 발생하는 Time to First Token (TTFT) 지연 문제를 다루고 있습니다. 이를 해결하기 위해 LayerKV라는 간단하면서도 효과적인 방법을 제안하여 TTFT를 줄이고 사용자 경험을 향상시킵니다.

- **Technical Details**: LayerKV는 레이어별 KV 블록 할당, 관리 및 오프로드를 소개하여 시스템 메모리의 세밀한 제어를 가능하게 하며, SLO(Service Level Objectives)를 최적화하기 위한 SLO 인식 스케줄러와 결합됩니다. 이를 통해 GPU의 KV cache 자원의 한계로 인한 쿼리 지연 문제를 완화합니다.

- **Performance Highlights**: LayerKV는 7B에서 70B까지 다양한 매개변수를 가진 모델에서 TTFT 지연을 최대 11배 개선하고, SLO 위반 비율을 28.7% 감소시키며, 전반적으로 사용자 경험을 크게 향상시킵니다.



### ManiSkill3: GPU Parallelized Robotics Simulation and Rendering for Generalizable Embodied AI (https://arxiv.org/abs/2410.00425)
Comments:
          Project website: this http URL

- **What's New**: ManiSkill3는 GPU 병렬화를 활용한 로봇 시뮬레이터로, 빠른 렌더링과 낮은 GPU 메모리 사용량으로 기존 플랫폼보다 최대 30,000+ FPS를 달성할 수 있습니다. 빠른 시뮬레이션 덕분에 학습에는 몇 시간이 아닌 몇 분만 소요됩니다.

- **Technical Details**: ManiSkill3는 복잡한 환경을 지원하면서도 간단한 API를 제공하여 다양한 로봇 작업을 쉽게 구축할 수 있도록 설계되었습니다. 이 시스템은 12개 다양한 카테고리의 환경과 20종 이상의 로봇을 아우르는 다양한 시뮬레이션을 제공합니다. 사용자에게 친숙한 객체 지향 식별 API를 포함하여, GPU 메모리 관리 및 시뮬레이션 환경 관리를 단순화했습니다.

- **Performance Highlights**: ManiSkill3는 시뮬레이션과 렌더링을 통합하여 10-1000배 빠른 속도로 작업을 처리하며, GPU 메모리 사용량은 기존 시뮬레이터의 2-3배 낮습니다. 다양한 환경과 로봇을 지원하며, 저비용으로 대량의 데이터 생성을 가능하게 하는 데이터 생성 파이프라인도 제공됩니다.



### Posterior-Mean Rectified Flow: Towards Minimum MSE Photo-Realistic Image Restoration (https://arxiv.org/abs/2410.00418)
- **What's New**: 이 논문에서는 Photo-Realistic Image Restoration (PIR) 문제에 대한 새로운 접근 방식을 제안합니다. 특히, 기존의 방법들이 왜곡 손실(distortion loss)과 인식 품질 손실(perceptual quality loss)을 최적화하는 데 초점을 맞춘 반면, 이 연구는 완벽한 인식 지표(perceptual index) 제약 하에서 평균 제곱 오차(Mean Squared Error, MSE)를 최소화하는 최적 추정자(optimal estimator)를 구하는 데 중점을 둡니다.

- **Technical Details**: 저자들은 Posterior-Mean Rectified Flow (PMRF)라는 새로운 알고리즘을 소개합니다. 이 알고리즘은 먼저 후방 평균(posterior mean)을 예측한 다음, 이를 rectified flow 모델을 사용하여 고품질 이미지로 변환합니다. 이 과정에서 필요한 최적 수송 지도(optimal transport map)를 근사합니다. PMRF는 모델이 재구성된 출력과 실제 이미지 간의 MSE를 최소화하도록 학습됩니다.

- **Performance Highlights**: PMRF는 다양한 이미지 복원 작업에서 기존 방법들보다 일관되게 우수한 성능을 나타내며, 이론적으로도 그 유용성이 입증되었습니다. 특히, 고품질 이미지 복원을 위한 새로운 접근 방식으로 자리매김할 가능성을 보여줍니다.



### TikGuard: A Deep Learning Transformer-Based Solution for Detecting Unsuitable TikTok Content for Kids (https://arxiv.org/abs/2410.00403)
Comments:
          NILES2024

- **What's New**: 이 연구는 TikTok에서 아동에게 부적절한 콘텐츠를 탐지하고 차단하기 위한 새로운 딥러닝 솔루션인 TikGuard를 소개합니다. 전통적인 콘텐츠 무결성 검토 방법의 한계를 극복하기 위해 고급 transformer 기반 모델을 활용하여 86.7%의 높은 정확성을 달성했습니다.

- **Technical Details**: TikGuard는 TikHarm라는 특별히 큐레이션된 데이터셋을 사용하여 아동에게 유해한 콘텐츠를 분류합니다. 이를 위해 고급 비디오 분류 기법을 사용하며, TimesFormer, VideoMAE, ViViT 등의 최신 transformer 모델을 채택하여 콘텐츠 검토의 정확성을 향상시킵니다.

- **Performance Highlights**: TikGuard는 기존 방법들에 비해 높은 정확성을 보였으며, 아동을 위한 안전한 온라인 환경을 조성하는 데 기여할 잠재력을 지니고 있습니다. TikHarm 데이터셋의 독창성으로 직접 비교가 제한되지만, 연구 결과는 소셜 미디어 플랫폼에서의 콘텐츠 검토 단계를 한층 더 발전시키는 데 이용될 수 있습니다.



### Revisiting Essential and Nonessential Settings of Evidential Deep Learning (https://arxiv.org/abs/2410.00393)
Comments:
          22 pages, under review

- **What's New**: 본 논문은 EDL(Evidential Deep Learning) 모델의 몇 가지 비필수적 설정을 완화한 Re-EDL을 제안합니다. Re-EDL은 주관적 논리(subjective logic)의 투영된 확률을 유지하며, 사전 가중치(prior weight)를 조절 가능한 하이퍼파라미터로 간주하고, 변동성 최소화(variance-minimizing) 최적화 항과 KL 발산 정규화(KL divergence regularization)를 생략합니다.

- **Technical Details**: Re-EDL은 Dirichlet 확률 밀도 함수(Dirichlet PDF)를 직접 최적화하여 예측 불확실성(prediction uncertainty)을 향상시키는 방법론입니다. 모델 구성에서 사전 가중치 파라미터는 클래스 수에 고정되지 않고 자유롭게 조정될 수 있는 하이퍼파라미터로 설정됩니다. 또한, 기존의 EDL 방법에서 사용되는 변동성 최소화 최적화 항과 KL 발산 최소화 정규화 항을 제거하여 성능을 개선합니다.

- **Performance Highlights**: 대규모 실험을 통해 Re-EDL의 효과성과 최첨단 성능을 입증하였으며, 주어진 문제에 대한 높은 신뢰도의 예측 불확실성을 제공함으로써 자율 주행 및 의료 분석과 같은 고위험 도메인에서의 활용 가능성을 보여줍니다.



### Boosting the Capabilities of Compact Models in Low-Data Contexts with Large Language Models and Retrieval-Augmented Generation (https://arxiv.org/abs/2410.00387)
Comments:
          13 pages, 1 figure, 5 tables, submitted to COLING 2025

- **What's New**: 이 논문에서는 데이터가 부족한 저자원 언어에 대한 기존 언어 모델 기술의 한계를 극복할 수 있는 접근법으로 Retrieval Augmented Generation (RAG) 프레임워크를 제안합니다. 특히, Uspanteko 및 Arapaho 언어의 형태론적 주석 작업을 위한 소규모 모델의 출력을 교정하는 방법을 다룹니다.

- **Technical Details**: RAG 프레임워크는 대형 언어 모델(LLM)을 활용하여 문법적 서술 정보를 결합합니다. 이를 통해 데이터와 학습 가능한 매개변수의 부족을 보완하며, LLM을 통해 해석된 문법 입력값을 사용합니다. 실험에서는 Claude 3.5 Sonnet과 GPT-4 모델을 비교하여 Uspanteko 및 Arapaho 언어에 대한 성능을 평가했습니다.

- **Performance Highlights**: 저자원 언어 처리에서의 성능과 효율성이 크게 향상되는 것을 보여주었으며, 새로운 SOTA(최신 기술 수준)를 달성했습니다. 이 접근법은 문서화 언어학자들에게 형태론적 주석 작업을 위한 더 신뢰할 수 있고 사용하기 쉬운 도구를 제공합니다.



### STGformer: Efficient Spatiotemporal Graph Transformer for Traffic Forecasting (https://arxiv.org/abs/2410.00385)
- **What's New**: 본 논문에서는 교통 예측을 위한 새로운 구조인 spatiotemporal graph transformer (STGformer)를 제안합니다. 이는 기존의 graph 신경망(GCN)과 transformer 기반 모델의 장점을 통합하여, 효율적인 교통 패턴 모델링을 가능하게 합니다. 특히 STGformer는 단일 레이어에서 고차원 spatiotemporal 상호작용을 캡처할 수 있는 혁신적인 STG attention block을 도입하여 계산 비용을 크게 줄였습니다.

- **Technical Details**: STGformer는 GCN과 Transformers의 강점을 조화롭게 결합하여, 전 세계적 및 지역적 교통 패턴을 동시에 모델링할 수 있습니다. 기존 방법들은 여러 개의 attention 레이어를 필요로 하지만, STGformer는 단일 레이어에서 모든 상호작용을 효율적으로 처리합니다. GPT를 포함한 대부분의 transformer 모델들이 높은 계산 비용을 요구하는 반면, STGformer는 100배 빠르고 GPU 메모리 사용을 99.8% 줄이는 성능을 보여줍니다.

- **Performance Highlights**: STGformer는 LargeST 벤치마크에서 최첨단 transformer 기반 방법들인 PDFormer 및 STAEformer보다 우수한 성능을 보여줍니다. 주목할 만한 점은, STGformer는 기존의 방법들에 비해 계산 비용이 현저히 낮으면서도 모든 상호작용을 유효하게 처리할 수 있는 능력을 구현하는 점입니다.



### Generative Precipitation Downscaling using Score-based Diffusion with Wasserstein Regularization (https://arxiv.org/abs/2410.00381)
Comments:
          19 pages, 9 figures

- **What's New**: 이 논문은 기후 예측 센터(CPC)에서 제공하는 강우량 데이터를 및 ERA5 재분석 데이터를 활용하여 1km 해상도의 강우량 추정치를 생성하는 새로운 생성적 확산 모델 WassDiff를 제안합니다.

- **Technical Details**: WassDiff 모델은 Wasserstein Distance Regularization (WDR) 기법을 사용하여 등급 기반 확산 모델의 점수 일치(training objective) 과정에서 고도화된 강우량 강도를 정확하게 재현하도록 훈련됩니다. 이 모델은 55km 해상도의 데이터를 1km로 다운스케일링하여 극단적인 강우 신호를 포착하는데 도전과제를 극복합니다.

- **Performance Highlights**: WassDiff는 극단적인 기상 현상, 예를 들어 열대 폭풍 및 한랭 전선의 사례 연구에서 적절한 공간적 패턴을 생성하며, 기존의 점수 기반 확산 모델보다 더 나은 재구성 정확도 및 편향 점수를 기록합니다.



### CXPMRG-Bench: Pre-training and Benchmarking for X-ray Medical Report Generation on CheXpert Plus Datas (https://arxiv.org/abs/2410.00379)
Comments:
          In Peer Review

- **What's New**: 본 논문은 X-ray 이미지 기반의 의학 보고서 생성을 위한 새로운 대규모 데이터셋인 CheXpert Plus와 함께, 이를 기반으로 한 CXPMRG-Bench 벤치마크를 제안하여 X-ray 보고서 생성을 위한 기존 알고리즘과 대형 모델들의 성능을 비교합니다.

- **Technical Details**: X-ray 보고서 생성을 위한 MambaXray-VL이라는 새로운 대형 모델을 제안하였으며, 이 모델은 자가 감독형(autoregressive) 생성과 X-ray 보고서 대비 학습(constrastive learning) 전략을 포함한 다단계(pre-training) 방식으로 훈련됩니다. 이는 기존의 Transformer 기반 모델들에서 발생하는 높은 계산 비용을 절감하며, 이미지와 텍스트 비교학습을 통해 더욱 효과적인 기능 공간 정렬을 달성합니다.

- **Performance Highlights**: MambaXray-VL은 CXPMRG-Bench 벤치마크에서 19개의 주요 X-ray 의학 보고서 생성 알고리즘과 14개의 대형 언어 모델, 2개의 비전-언어 모델을 평가하였고, 실험 결과에서도 최첨단 성능을 달성하여 기존 모델들과 비교했을 때 우수한 결과를 보였습니다.



### Robust Traffic Forecasting against Spatial Shift over Years (https://arxiv.org/abs/2410.00373)
- **What's New**: 본 논문은 새로운 OOD(Out-Of-Distribution) 벤치마크를 제안하고, 기존의 ST-GNN(Spatiotemporal Graph Neural Networks) 모델들이 이러한 상황에서 성능이 크게 저하된다는 점을 강조합니다. 이를 해결하기 위해 새로운 Mixture of Experts (MoE) 프레임워크를 도입하여, 환경 변화에 적응하여 새로운 그래프를 생성하도록 합니다.

- **Technical Details**: 제안된 방법은 각 그래프 생성기(graph generator)를 학습하여 환경 변화에 따라 새로운 그래프 생성을 가능하게 하며, 기존의 모든 spatiotemporal 모델에 통합될 수 있습니다. 또한, LSTM을 사용한 연구가 시간적 의존성에 대한 성능 저하를 분석하는 데 사용됩니다. 여기서 제안한 expert graphon layer는 OOD 환경을 학습하고, 이를 통해 모델이 새로운 환경에 적합한 그래프를 적응적으로 조합하도록 합니다.

- **Performance Highlights**: 제안된 MoE 프레임워크는 기존의 BENCHMARK보다 우수한 성능을 보이며, 교차적인 성능을 평가하여 이전의 모델보다 안정적인 교통 예측이 가능합니다. 기존 ST-GNN 모델들이 겪는 성능 저하 문제를 해결하고, 다양한 환경에서의 예측을 향상시킬 수 있는 전략을 제시합니다.



### Easydiagnos: a framework for accurate feature selection for automatic diagnosis in smart healthcar (https://arxiv.org/abs/2410.00366)
- **What's New**: 본 연구에서는 Adaptive Feature Evaluator (AFE) 알고리즘을 제안하여 의료 데이터셋 내에서 특성 선택(feature selection)을 개선하고 클리닉 환경에서의 인공지능(AI) 적용에 있어 존재하는 문제를 해결하고자 합니다.

- **Technical Details**: AFE는 Genetic Algorithms (GA), Explainable Artificial Intelligence (XAI), Permutation Combination Techniques (PCT)를 통합하여 Clinical Decision Support Systems (CDSS)의 성능을 최적화하며, 다양한 머신러닝 알고리즘을 통해 세 가지 의료 데이터셋에서 검증되었습니다.

- **Performance Highlights**: AFE 알고리즘은 Multi-layer Perceptron (MLP)와 결합하여 최대 98.5%의 정확도를 달성하였으며, 기존의 특성 선택 기법들에 비해 동작의 견고성을 강조합니다.



### FedPT: Federated Proxy-Tuning of Large Language Models on Resource-Constrained Edge Devices (https://arxiv.org/abs/2410.00362)
Comments:
          29 pages, 19 figures

- **What's New**: 이 논문에서는 Federated Proxy-Tuning (FedPT)이라는 새로운 프레임워크를 소개하여 개인 데이터를 공유하지 않고도 대형 블랙박스 언어 모델을 효율적으로 미세 조정할 수 있는 방법을 제안하고 있습니다.

- **Technical Details**: FedPT는 먼저 장치들이 소형 언어 모델을 공동으로 조정한 후, 서버가 이 작은 모델의 지식을 큰 사전 훈련된 모델과 결합하여 성능을 향상시키는 방법입니다. 이 과정은 반복적으로 수행되며, 최종적으로 큰 프록시 조정 모델을 생성합니다. 회로에서 모델의 파라미터에 직접 접근하지 않고 예측값만을 사용합니다.

- **Performance Highlights**: 실험 결과, FedPT는 직접 대형 LM을 연합적으로 미세 조정하는 것보다 계산, 통신, 메모리 오버헤드를 크게 줄이면서 경쟁력 있는 성능을 유지하는 것으로 나타났습니다. 이러한 접근은 리소스가 제한된 장치에서 대형 언어 모델의 접근성과 활용성을 넓히는 가능성을 제시합니다.



### Self-controller: Controlling LLMs with Multi-round Step-by-step Self-awareness (https://arxiv.org/abs/2410.00359)
Comments:
          10 pages, 6 figures

- **What's New**: 본 논문에서는 LLM(large language models)의 제어 능력을 향상시키기 위한 새로운 프레임워크인 'Self-controller'를 제안합니다. 이 프레임워크는 LLM이 자신의 상태를 인식하고 단계별로 사고할 수 있도록 돕습니다.

- **Technical Details**: Self-controller는 상태 반영기(state reflector)와 다중 라운드 대화 세션으로 구성됩니다. 이 프레임워크는 LLM의 응답에 기반한 상태를 유지하며, 텍스트 길이를 조정하기 위해 이진 검색 알고리즘을 구현하였습니다. 또한, DeepSeek의 Context Caching 기술을 활용하여 동일한 맥락의 대화 클러스터에서 계산된 토큰 소모를 획기적으로 줄입니다.

- **Performance Highlights**: 실험 결과, Self-controller는 다양한 데이터셋에서 LLM의 제어 가능성을 크게 향상시켰으며, 성능 저하 없이 효과적인 텍스트 생성을 달성했습니다. 본 방법은 일반적인 단일 라운드 생성에 비해 최대 두 배의 토큰 소모만을 요구합니다.



### Efficient Training of Large Vision Models via Advanced Automated Progressive Learning (https://arxiv.org/abs/2410.00350)
Comments:
          Code: this https URL. arXiv admin note: substantial text overlap with arXiv:2203.14509

- **What's New**: 이 논문에서는 대형 비전 모델(Large Vision Models, LVMs)의 효율적인 학습을 위한 진보된 자동화적 점진적 학습(AutoProg) 프레임워크를 제안합니다. 기존의 학습 방법에 비해 훈련 시간과 비용을 크게 절감하면서도 성능은 유사하거나 향상되는 결과를 보였습니다.

- **Technical Details**: AutoProg 프레임워크는 ViTs(Vision Transformers)와 확산 모델(Diffusion Models)을 포함한 다양한 LVM에 적용됩니다. 특히, AutoProg-One 및 AutoProg-Zero 개발에 중점을 두며, 각각 모멘텀 증가(MoGrow) 및 제로샷 자동화된 점진적 학습 기법을 특징으로 합니다.

- **Performance Highlights**: 실험 결과, AutoProg-One은 ImageNet에서 ViTs의 사전 훈련을 최대 1.85배 가속화하였으며, AutoProg-Zero는 안정적 확산(Stable Diffusion) 및 Diffusion Transformers의 전이 훈련을 각각 최대 2.86배 및 2.56배 가속화했습니다. 중앙 성능을 유지하며 훈련 비용은 현저히 감소했습니다.



### Sparse Attention Decomposition Applied to Circuit Tracing (https://arxiv.org/abs/2410.00340)
- **What's New**: 본 연구에서는 GPT-2 small 모델 내의 attention heads 간의 통신 및 조정을 효과적으로 분석하기 위해, 그 과정에서 사용되는 희소한(signal) 특징들을 고립 및 식별하고자 합니다. 또한, attention head 행렬의 특잇값 분해(singular value decomposition)로부터 얻은 특징을 바탕으로 보다 효율적인 경로 추적을 제안합니다.

- **Technical Details**: GPT-2 small 모델을 사용하여 Indirect Object Identification (IOI) 작업에서 attention heads 간의 관계를 분석하였습니다. 이 연구에서는 residual 배경으로부터 신호를 효율적으로 분리하고, attention head의 입력을 새로운 기준으로 바꾸어 sparseContribution을 정의하였습니다. 이 새로운 기준을 통해 downstream과 upstream attention heads 간의 인과 관계를 명확히 할 수 있었습니다.

- **Performance Highlights**: 본 연구의 결과로, 새로운 기준을 통해 trace한 신호가 기존 연구들보다 더욱 세부적인 내용을 제공하며, GPT-2가 IOI 작업을 수행하는 데 있어 효과적인 커뮤니케이션 경로를 식별할 수 있음을 보여줍니다. 이를 통해 attention score의 해석 가능성이 크게 향상되며, 모델의 기능적 요소를 보다 명확히 이해할 수 있게 됩니다.



### Preserving Generalization of Language models in Few-shot Continual Relation Extraction (https://arxiv.org/abs/2410.00334)
Comments:
          Accepted to EMNLP 2024

- **What's New**: 이 연구에서는 Few-shot Continual Relations Extraction (FCRE) 분야에서 새로운 접근 방식을 제안합니다. 특히, 기존에 소홀히 여겨지던 언어 모델 헤드를 활용하여 연결된 지식을 효과적으로 통합하는 방법을 소개합니다.

- **Technical Details**: Mutual Information Maximization (MIM) 전략을 통해 사전 훈련된 백본 네트워크로부터 이전 지식을 보존하고, 주요 분류 헤드의 정렬을 전략적으로 개선하여 모델 성능을 향상시킵니다. 또한, Large Language Models (LLMs)의 잠재력을 FCRE 문제에 적용하는 방법을 탐구합니다.

- **Performance Highlights**: 실험 결과는 제안된 방법이 사전 훈련된 언어 모델의 일반화 능력을 유지하고, 지식의 잊혀짐을 줄이는 데 있어 매우 효과적임을 보여줍니다. 이는 모델의 대표성 학습을 개선하고 전체적인 성능을 크게 향상시키는 결과로 이어집니다.



### EnzymeFlow: Generating Reaction-specific Enzyme Catalytic Pockets through Flow Matching and Co-Evolutionary Dynamics (https://arxiv.org/abs/2410.00327)
- **What's New**: EnzymeFlow는 효소의 촉매 주머니(catalytic pocket)를 설계하기 위한 새로운 생성 모델로, 흐름 매칭(flow matching)과 계층식 사전 학습(hierarchical pre-training) 및 효소-반응 공진화(enzyme-reaction co-evolution)를 활용하여 특정 기질(substrate) 및 촉매 반응에 대해 데이터 기반으로 촉매 주머니를 생성합니다.

- **Technical Details**: EnzymeFlow는 효소 촉매 주머니 생성을 위한 조건적 흐름(conditional flow)을 정의하며, 이를 통해 특정 기질과 생성물에 따라 다양한 촉매 과정이 가능하도록 합니다. 또한, 효소-반응 공진화(co-evolution)를 통해 촉매 반응에서 기질 특이성을 포착하고, 구조 기반의 계층식 사전 학습을 통해 더 나은 모델 성능을 꾀합니다. 데이터 세트는 $328,192$ 쌍의 효소-반응 쌍으로 구성되어 있습니다.

- **Performance Highlights**: 실험 결과, EnzymeFlow는 고품질의 기능성 효소 촉매 주머니를 설계하는 데 효과적임을 보여주었으며, 효소 공학(enzyme engineering) 및 합성 생물학(synthetic biology) 분야에서의 발전을 위한 새로운 가능성을 제시합니다.



### EmoKnob: Enhance Voice Cloning with Fine-Grained Emotion Contro (https://arxiv.org/abs/2410.00316)
Comments:
          EMNLP 2024 Main

- **What's New**: 최근 Text-to-Speech (TTS) 기술의 발전에도 불구하고 감정을 선택하고 강도를 조절할 수 있는 기능이 부족했습니다. 본 연구에서는 EmoKnob 라는 프레임워크를 제안하며, 이는 임의의 감정을 갖는 소수의 샘플로 감정을 정밀하게 조정할 수 있습니다.

- **Technical Details**: 이 프레임워크는 최근의 기초 음성 복제 모델의 발전을 활용하여 감정을 제어할 수 있는 두 가지 방법을 제안합니다. 첫 번째는 감정 설명을 텍스트로 제공받고 이를 통해 감정을 조절하는 것이며, 두 번째는 대량의 언어 모델과 텍스트 임베딩 모델을 활용하는 것입니다. 또한, 감정 인식 및 충실성을 평가하기 위한 새로운 평가 지표 세트를 도입합니다.

- **Performance Highlights**: 객관적 및 주관적 평가를 통해 우리의 감정 제어 프레임워크가 감정을 효과적으로 음성에 내재화하며, 상용 TTS 서비스보다 감정 표현에서 우수한 성능을 보여주었습니다. 83%의 참여자들은 우리의 프레임워크가 감정 강화에 있어서 상위 TTS 서비스보다 뛰어나다고 평가했습니다.



### Contrastive Representation Learning for Predicting Solar Flares from Extremely Imbalanced Multivariate Time Series Data (https://arxiv.org/abs/2410.00312)
Comments:
          This work has been accepted at ICMLA 2024 on September 7, 2024, as a short paper for poster presentation

- **What's New**: 본 논문에서는 태양의 주요 플레어를 예측하기 위한 새로운 머신 러닝 접근 방식인 CONTREX를 소개합니다. 이 접근법은 다변량 시계열 데이터의 특성을 포착하고, 유사한 클래스 간의 거리를 최대화하여 극단적인 클래스 불균형 문제를 해결합니다.

- **Technical Details**: CONTREX는 다변량 시계열 (Multivariate Time Series) 데이터를 다루기 위한 대조적 표현 학습 (Contrastive Representation Learning) 접근법입니다. 이 방법은 catch22 기능 추출을 사용하고, 긍정 및 부정 클래스 특징 벡터에서 두 개의 극단 포인트를 유도하여 최적의 분리 능력을 제공합니다. 또한, 맞춤형 대조적 재구성 손실 (Contrastive Reconstruction Loss)을 통해 시계열 데이터의 임베딩을 최적화합니다.

- **Performance Highlights**: CONTREX는 SWAN-SF 다변량 시계열 벤치마크 데이터셋에서 기존 방법 대비 우수한 태양 플레어 예측 성능을 보여주었습니다.



### Ask, Pose, Unite: Scaling Data Acquisition for Close Interactions with Vision Language Models (https://arxiv.org/abs/2410.00309)
Comments:
          Project webpage: this https URL

- **What's New**: 이 논문에서는 Human Mesh Estimation (HME) 분야에서의 데이터 부족 문제를 해결하기 위해, Large Vision Language Models (LVLMs)을 활용하여 참조 데이터와 가상의 참조 메시를 생성하는 새로운 방법을 도입하였습니다.

- **Technical Details**: 제안된 방법인 Ask Pose Unite (APU) 데이터세트는 6,200개 이상의 인간 메시 쌍을 포함하여 다양한 유형의 상호작용을 다룹니다. 이를 통해 HME 모델을 학습시키고, 테스트 시 최적화를 통해 메시 추정의 정확도를 향상시킵니다. 또한, 이 방법은 수동 주석의 필요성을 줄이며, 사실적인 인물 간의 상호작용을 반영하는 포괄적인 데이터세트를 제공합니다.

- **Performance Highlights**: 새로 생성된 APU 데이터세트를 이용하여 diffusion 기반의 contact prior를 훈련시키는 실험을 진행했으며, 이를 통해 이전에 보지 못한 상호작용에 대한 메쉬 추정 정확도가 유의미하게 개선되었음을 보여주었습니다.



### On Large Uni- and Multi-modal Models for Unsupervised Classification of Social Media Images: Nature's Contribution to People as case study (https://arxiv.org/abs/2410.00275)
Comments:
          15 pages, 9 figures

- **What's New**: 이 논문에서는 소셜 미디어에서 공유되는 이미지의 의미 있는 클러스터링(task of grouping) 문제를 해결하기 위해 최근의 대형 모델들인 Large Visual Models (LVM), Large Language Models (LLM), Large Visual Language Models (LVLM)을 활용한 다양한 접근 방법을 제안하고 있습니다.

- **Technical Details**: 논문에서는 Cultural Ecosystem Services (CES) 문제를 다루며, 이를 위해 FLIPS라는 데이터셋을 구축했습니다. LVM과 LVLM을 활용한 다양한 방법을 평가하는 실험을 진행했으며, 특히 최적 결과를 도출한 모델로는 DINOv2(LVM)와 GPT-4(LVLM)가 있습니다. 전처리 과정에서는 prompt engineering을 통해 이미지 분류를 수행했습니다.

- **Performance Highlights**: 최상위 성능을 보인 방법은 소량의 라벨링된 데이터셋에서 파인튜닝된 LVM DINOv2와 간단한 prompt를 사용한 LVLM 모델인 GPT-4로 나타났습니다. 이로 인해, 사회적 이미지를 CES 클러스터로 분류하는 효율적인 방법이 확인되었습니다.



### Social Conjuring: Multi-User Runtime Collaboration with AI in Building Virtual 3D Worlds (https://arxiv.org/abs/2410.00274)
Comments:
          27 pages + Appendix, 16 figures

- **What's New**: 이 논문은 사용자가 실시간으로 협력하여 가상 세계를 구축하고 수정할 수 있는 AI 보조 3D 장면 공동 생성 프레임워크인 Social Conjurer를 제안합니다.

- **Technical Details**: Social Conjurer는 다중 사용자 인터랙션을 통해 사회적이고 도구 기반의 참여를 포함한 확장된 상호작용 세트를 제공합니다. 이 프레임워크는 사용자의 사회적 경험이 공간 환경의 생성에 어떻게 영향을 미치는지를 탐색합니다.

- **Performance Highlights**: 예비 사용자 연구를 통해, 다중 사용자 컨텍스트가 공간 장면을 생성하는 방식에 미치는 영향에 대한 통찰을 제공하고, 협력적 가상 세계 생성의 도전 과제와 기회를 논의합니다.



### KPCA-CAM: Visual Explainability of Deep Computer Vision Models using Kernel PCA (https://arxiv.org/abs/2410.00267)
Comments:
          5 pages, 4 figures, Published to IEEE MMSP 2024

- **What's New**: 본 연구는 Convolutional Neural Networks (CNNs)의 해석 가능성을 증대하기 위해 KPCA-CAM이라는 새로운 기술을 도입합니다. KPCA-CAM은 기존 Class Activation Maps (CAMs)을 개선하고 비선형 관계를 효과적으로 포착하기 위해 Kernel Principal Component Analysis (Kernel PCA)를 활용합니다.

- **Technical Details**: KPCA-CAM은 CNN 활성화에서 비선형 관계를 포착하기 위해 커널 함수를 사용하여 데이터를 고차원 공간으로 매핑합니다. 이 기술은 Eigen-CAM에서 사용되었던 기존 PCA의 한계를 극복하고 더 정확한 데이터 표현을 제공합니다. 또한 KPCA-CAM은 다양한 커널 함수를 사용하여 데이터의 여러 측면을 포착할 수 있습니다.

- **Performance Highlights**: ILSVRC 데이터셋에 대한 실험 결과, KPCA-CAM은 기존 CAM 알고리즘에 비해 더 정밀한 활성화 맵을 생성하여 모델의 추론 과정을 보다 명확하게 제공합니다. 이로 인해 연구자와 실무자가 CNN의 의사 결정 프로세스를 보다 깊이 이해할 수 있는 강력한 도구를 갖게 되었습니다.



### Procedure-Aware Surgical Video-language Pretraining with Hierarchical Knowledge Augmentation (https://arxiv.org/abs/2410.00263)
Comments:
          Accepted at the 38th Conference on Neural Information Processing Systems (NeurIPS 2024) Main Track

- **What's New**: 이 연구는 수술 비디오-언어 사전학습(VLP)의 새로운 접근 방식을 제안하며, 이를 통해 텍스트 정보 손실 및 공간-시간적 도전 과제를 해결하고자 합니다. 이를 위한 새로운 프레임워크인 Procedure-Encoded Surgical Knowledge-Augmented Video-Language Pretraining (PeskaVLP)를 개발했습니다.

- **Technical Details**: PeskaVLP는 대형 언어 모델(LLM)을 이용하여 수술 개념을 정제하고 풍부하게 만들어, 다각적인 언어 감독을 제공하며 오버피팅(overfitting)의 위험을 줄입니다. 이 프레임워크는 계층적 지식 증강(hierarchical knowledge augmentation) 방법을 통해 수술 절차에 고유한 공간-시간적 특성을 이해하고, 동적 시간 왜곡(Dynamic Time Warping, DTW) 기반의 손실 함수로 비디오 프레임과 텍스트 간의 시간적 정렬을 학습합니다.

- **Performance Highlights**: 광범위한 공개 수술 장면 이해(surgical scene understanding) 및 크로스모달 검색(cross-modal retrieval) 데이터셋에서 광범위한 실험 결과, 제안한 방법이 제로샷(zero-shot) 전이 성능을 크게 향상시키고, 수술 장면 이해의 추가 발전을 위한 일반적인 시각적 표현을 제공합니다.



### DoPAMine: Domain-specific Pre-training Adaptation from seed-guided data Mining (https://arxiv.org/abs/2410.00260)
- **What's New**: 이번 연구에서 제안하는 DoPAMine 프레임워크는 대규모 데이터 코퍼스에서 도메인 특화 훈련 데이터를 자동으로 채굴하여 대형 언어 모델(LLM)의 도메인 적응을 지원합니다. 이 방법은 일반적인 웹 데이터를 기반으로 한 이제까지의 접근 방식과는 달리, 도메인에 맞춘 데이터 생성 및 실제 세계 데이터를 채굴하여 현실적이고 신뢰할 수 있는 결과를 제공합니다.

- **Technical Details**: DoPAMine 프레임워크는 LLM의 매개변수적 지식을 활용하여 특정 도메인에 최적화된 다양한 시드(seed) 데이터를 생성하고, 이를 통해 Common Crawl과 같은 대규모 데이터 코퍼스에서 관련 데이터 문서를 채굴하는 과정을 포함합니다. 핵심 메커니즘은 LLM을 활용하여 도메인 특화된 시드 데이터를 생성하고, 생성된 시드 데이터를 기반으로 의미적으로 유사한 문서를 검색하는 것입니다.

- **Performance Highlights**: DoPAMine를 사용하여 지속적 재학습(CPT) 환경에서 두 개의 도메인 특화 7B 파라미터 LLM을 의료 및 금융 분야에서 훈련한 결과, 평균 4.9% 및 5.1%의 성능 향상을 보였으며, 이는 zero-shot 및 5-shot 설정에서 MMLU, MedQA, MedMCQA 및 PubMedQA 데이터셋에 대한 것입니다. 금융 과제에서는 평균 2.9% 및 6.7%의 향상을 나타내었으며, FiQA-SA, FPB 및 Headlines 데이터셋과 비교하여 성능이 크게 높아졌습니다.



### The age of spiritual machines: Language quietus induces synthetic altered states of consciousness in artificial intelligenc (https://arxiv.org/abs/2410.00257)
Comments:
          8 Figures

- **What's New**: 이 연구는 언어가 의식(consciousness)과 어떻게 관련되는지를 탐구하며, 특히 psychedelic(환각제) 사용과 명상이 언어 카테고리화(categorisation)의 능력에 미치는 영향을 다룹니다.

- **Technical Details**: 연구에서는 CLIP와 FLAVA 모델의 주의(attentional) 가중치를 조작하여 시뮬레이션된 ALTERED states의 의미적 임베딩(spatial embedding) 공간을 비교했습니다. 이 과정에서 언어에 대한 주의력이 감소했을 때 나타나는 독특한 언어 패턴 및 흐릿한 임베딩을 관찰하였습니다. 예를 들어, '기린(giraffes)'이 '바나나(bananas)'와 더욱 유사해지는 현상이 발생했습니다.

- **Performance Highlights**: 모델은 무신체(disembodied), 자아가 없는(ego-less), 영적(spiritual), 통합적(unitive) 상태와 최소한의 현상 경험(minimal phenomenal experiences)으로 더 잘 정렬되었습니다. 이는 충분한 용량의 환각제 사용 또는 집중 명상에서 경험하는 의식의 변화를 통해 정신 건강(mental health) 및 웰빙(wellbeing) 향상으로 이어질 수 있음을 지지합니다.



### Helpful DoggyBot: Open-World Object Fetching using Legged Robots and Vision-Language Models (https://arxiv.org/abs/2410.00231)
Comments:
          Project website: this https URL

- **What's New**: 이번 연구에서는 인도 환경에서 도움이 되는 이동 조작 기술을 위한 새로운 시스템인 Helpful DoggyBot을 제안합니다. 이 시스템은 전면에 장착된 그리퍼와 저수준 제어기를 사용하여, 다양한 내부 환경에서 유용한 작업을 수행할 수 있도록 설계되었습니다.

- **Technical Details**: Helpful DoggyBot은 1-DoF 그리퍼를 사용하여 일상적인 객체를 집어올리며, egocentric depth(자기중심 깊이)와 proprioception(신체위치 감각)을 활용한 저수준 제어기를 시뮬레이션에서 훈련하여 기동성을 높였습니다. Vision-Language Models (VLMs)를 활용하여 객체 식별 및 명령 생성을 수행하며, 이는 목표 객체를 따라가고 추적하는 데 도움을 줍니다.

- **Performance Highlights**: 이 시스템은 두 개의 미지의 환경에서 zero-shot generalization을 수행하여, 예를 들어 사용자의 명령에 따라 무작위로 놓인 인형을 찾아오는 작업을 60%의 성공률로 완료했습니다. 이는 실제 세계 데이터 수집이나 훈련 없이도 가능하여, 다양한 홈 환경에 적응할 수 있는 도움이 되는 4족 보행 로봇의 가능성을 보여줍니다.



### Zero-Shot Classification of Crisis Tweets Using Instruction-Finetuned Large Language Models (https://arxiv.org/abs/2410.00182)
- **What's New**: 이 연구는 자연재해나 인도적 위기 상황에서 소셜 미디어 게시물의 분류를 위한 상업적 대형 언어 모델(LLMs)을 평가하고, 이러한 모델들이 위기 관련 정보의 분류 성능에서 어떤 차이를 보이는지를 분석합니다.

- **Technical Details**: 연구팀은 CrisisBench 데이터셋을 이용하여 세 가지 LLM(OpenAI GPT-4o, Google Gemini 1.5-flash-001, Anthropic Claude-3-5 Sonnet)의 제로샷(classification에 대한 사전 훈련 없이 직접 할 수 있는 방법) 분류 성능을 평가하였습니다. 주요 작업으로는 두 가지가 있으며, 첫째는 게시물이 인도적 맥락에서 정보를 제공하는지 여부를 판단하는 것이고, 둘째는 16개 인도적 클래스에 따라 게시물의 확률을 평가하는 것입니다. F1 점수로 성과를 평가했습니다.

- **Performance Highlights**: 연구 결과, 정보 제공 분류 작업은 추가 정보 없이도 대부분 잘 수행되었으나, 인도적 레이블 분류 작업의 경우 트윗이 채굴된 사건과 관련된 정보가 제공되었을 때 더 나은 성과를 보였습니다. 또한 데이터셋에 따라 모델 성능이 상당히 다르게 나타나 데이터셋 품질에 대한 의문을 제기하였습니다.



### Adapting LLMs for the Medical Domain in Portuguese: A Study on Fine-Tuning and Model Evaluation (https://arxiv.org/abs/2410.00163)
Comments:
          This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 이 연구는 포르투갈어로 된 대규모 언어 모델(LLMs)의 의료 에이전트 성능을 평가하여 의료 전문가를 위한 신뢰할 수 있고 관련성이 있는 가상 비서를 개발하는 것을 목표로 하고 있습니다.

- **Technical Details**: HealthCareMagic-100k-en과 MedQuAD 데이터셋을 영어에서 GPT-3.5를 사용하여 번역하고, PEFT-QLoRA 방법을 통해 ChatBode-7B 모델을 미세 조정하였습니다. InternLM2 모델이 의료 데이터에서 초기 훈련을 통해 가장 높은 성능을 나타냈고, DrBode 모델은 취득한 의료 지식의 재앙적 망각(catatrophic forgetting) 현상을 보였습니다.

- **Performance Highlights**: 이 연구의 결과는 다국어 의료 모델의 평가, 훈련 데이터 품질 향상, 그리고 의료 분야에 맞는 일관된 평가 방법론 개발의 필요성을 강조합니다. 다각적인 평가를 통해 의료 전문가들의 높은 정확도, 완전성, 안전성을 확인하였습니다.



### Beyond Single Concept Vector: Modeling Concept Subspace in LLMs with Gaussian Distribution (https://arxiv.org/abs/2410.00153)
Comments:
          28 pages, 9 figures

- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)의 내부 지식 표현을 이해하기 위해 Gaussian Concept Subspace (GCS)를 도입했습니다. 이는 기존의 단일 벡터 대신, 특정 개념을 나타내는 서브스페이스를 근사하는 접근 방식입니다.

- **Technical Details**: GCS는 각각의 개념에 대해 단일 벡터가 아닌 관찰 벡터(observed vectors) 집합을 학습하여, 가우시안 분포를 사용하여 개념 서브스페이스를 추정합니다. 이로 인해 더 미세한 개념 표현이 가능해지고, 벡터의 밀도가 개념에 대한 관련성의 정도를 나타냅니다.

- **Performance Highlights**: 실험 결과 GCS로부터 샘플링된 벡터는 LLM에서 학습된 개념을 효과적으로 설명하고, 다운스트림 개입(intervention) 작업에서 더 바람직한 결과를 생성하는데 기여함을 보여주었습니다. 특히, 자연어 생성 과제에서 텍스트 유창성을 유지하면서 감정 조정 성능을 균형 있게 달성하는 데 성공했습니다.



### Fisher Information-based Efficient Curriculum Federated Learning with Large Language Models (https://arxiv.org/abs/2410.00131)
Comments:
          27 pages, 8 figures, 14 tables, to appear in EMNLP 2024

- **What's New**: 이 논문에서는 Federated Learning(FL) 환경에서 Large Language Models(LLMs)를 효율적으로 미세 조정하기 위해 Fisher Information 기반의 새로운 Curriculum Federated Learning 프레임워크(FibecFed)를 제안합니다. 이 프레임워크는 두 가지 혁신적인 방법인 적응형 연합 커리큘럼 학습 및 효율적인 희소 파라미터 업데이트를 포함하고 있습니다.

- **Technical Details**: FibecFed는 각 장치 내에서 훈련 데이터의 난이도를 측정하기 위해 Fisher Information 기반의 방법을 활용하여 적응적으로 데이터를 샘플링합니다. 이를 통해 초기에는 쉬운 데이터 샘플을 사용하고 점진적으로 난이도를 높이며 FL의 미세 조정 효과성을 향상시킵니다. 또한, LoRA를 활용하여 전역 집합을 위해 적절한 레이어를 선택하고 희소 파라미터를 동적으로 업데이트하여 효율성을 개선합니다.

- **Performance Highlights**: FibecFed는 10개의 데이터 세트를 기반으로 한 광범위한 실험 결과에서 17개의 기준 방법에 비해 정확도가 최대 45.35% 향상되었고, 미세 조정 속도는 최대 98.61% 더 빨라졌음을 보여주었습니다.



### Cartesian Genetic Programming Approach for Designing Convolutional Neural Networks (https://arxiv.org/abs/2410.00129)
- **What's New**: 이번 연구에서는 Convolutional Neural Networks (CNNs)의 설계 및 최적화를 위한 방식으로 Cartesian Genetic Programming (CGP)을 활용한 Neural Architecture Search (NAS)에 대해 다룹니다. 기존의 수작업으로 설계된 아키텍처 대신 CGP를 통해 자동으로 아키텍처를 생성하여 더 나은 성능을 기대할 수 있습니다.

- **Technical Details**: 저자들은 mutation(변이)만을 적용하는 순수한 CGP 접근법을 활용하여 CNN 아키텍처를 생성합니다. 진화는 유전자의 특정 집합을 생성하고 이를 기반으로 변이를 통해 새로운 세대를 만들어내며, 성능 평가를 통해 최적의 아키텍처를 탐색하는 방식으로 진행됩니다. 각 세대에서 최고의 유전자형을 기록하고, fitness function(적합도 함수)을 통해 성능을 평가합니다.

- **Performance Highlights**: 실험 결과, 62.5K의 중간 예산에서 가장 우수한 결과를 얻었으며, 높은 예산인 125K에서 알고리즘 성능이 저하되는 경향을 보였습니다. 샘플 데이터셋인 MNIST와 Fashion-MNIST를 기반으로, mutuation rates(변이 비율)에 따른 성능 차이를 확인하였습니다. 향후 연구에서는 더 큰 네트워크 집단 및 다양한 변이 비율을 탐색할 필요성이 강조됩니다.



### ACE: All-round Creator and Editor Following Instructions via Diffusion Transformer (https://arxiv.org/abs/2410.00086)
- **What's New**: 본 논문에서는 ACE(All-round Creator and Editor)라는 새로운 확산 모델(diffusion model)을 제안합니다. 이 모델은 다양한 시각 생성 작업에서 뛰어난 성능을 발휘하며, 텍스트 기반 시각 생성 작업을 넘어선 멀티 모달(multi-modal) 조건을 지원합니다.

- **Technical Details**: ACE 모델은 Long-context Condition Unit (LCU)이라는 통합 조건 형식을 도입하고, Transformer 기반의 새로운 확산 모델을 사용하여 LCU를 입력으로 활용하는 방식으로 설계되었습니다. 이를 통해 다양한 생성 및 편집 작업을 공동 훈련(joint training)할 수 있도록 합니다.

- **Performance Highlights**: 광범위한 시각 생성 작업에서 ACE 모델은 기존 전문가 모델과 비교할 만한 성능을 보였습니다. 이 모델은 단일 모델을 통해 이미지 생성에 대한 모든 상호작용 요청에 응답할 수 있는 멀티 모달(chat system) 시스템을 쉽게 구축할 수 있는 장점을 제공합니다.



### A Survey on Diffusion Models for Inverse Problems (https://arxiv.org/abs/2410.00083)
Comments:
          Work in progress. 38 pages

- **What's New**: 이 논문에서는 pretrained diffusion model을 활용하여 inverse problem을 해결하는 방법에 대한 포괄적인 개요를 제공합니다. 기존 훈련 과정 없이도 고품질 샘플을 생성할 수 있는 가능성을 탐구합니다.

- **Technical Details**: Diffusion models는 unsupervised priors로 사용되어 inverse problem 해결에 활용됩니다. 다양한 문제와 기술에 따라 방법을 분류하는 taxonomy를 소개하며, 이를 통해 각 접근 방식의 연결성을 분석합니다. 특히 latent diffusion model을 사용할 때의 도전 과제와 잠재적 해결책에 대해서도 논의합니다.

- **Performance Highlights**: Diffusion models를 활용한 inverse problem 해결 방법의 실용적 구현을 강조하며, 다양한 과학 분야에서의 적용 가능성을 보여줍니다. 예를 들어, 이미지 복원, MRI 가속화, 오디오 신호 처리 등에서 효과적인 결과를 나타냅니다.



### Graph Residual Noise Learner Network for Brain Connectivity Graph Prediction (https://arxiv.org/abs/2410.00082)
Comments:
          10 pages, 3 figures, 6th Workshop on GRaphs in biomedicAl Image anaLysis

- **What's New**: 본 연구에서는 신경 장애 진단을 개선하기 위한 새로운 그래프 기반의 뇌 그래프 예측 모델인 Graph Residual Noise Learner Network (Grenol-Net)를 제안합니다. 기존의 확산 모델을 활용하되, 그래프의 토폴로지 대칭성을 유지하기 위한 접근 방식을 채택하여, 신경 과학 연구 커뮤니티에서의 협력적 연구 노력을 촉진하고자 합니다.

- **Technical Details**: Grenol-Net은 두 개의 복잡하게 설계된 학습 블록으로 구성됩니다. 첫 번째 블록은 그래프 컨볼루션 블록을 포함하여 소스 그래프 내 연결의 복잡한 상호작용을 드러내며, 서로 다른 노드 간의 메시지 전달 기능을 통해 각 ROI의 독특한 임베딩을 학습합니다. 두 번째 블록은 배치 정규화를 도입하여 타겟 그래프의 분포를 학습하고, 잔여 연결을 통해 타겟에 적용된 노이즈를 분리 및 복구합니다.

- **Performance Highlights**: Grenol-Net은 기존의 신경망 모델에 비해 예측 정확도 및 그래프 연결성 다양성을 유지하면서도 더 나은 성능을 자랑합니다. 예측 과정에서는 그래프의 노드 특성을 정확하게 학습하여, 동일한 파셀레이션 뇌 템플릿에서 파생된 동형(biomorphic) 그래프들 간의 연결성 다양성을 유지하는 데 초점을 맞추었습니다.



### From homeostasis to resource sharing: Biologically and economically compatible multi-objective multi-agent AI safety benchmarks (https://arxiv.org/abs/2410.00081)
Comments:
          18 pages, 14 figures, 1 tables

- **What's New**: 본 연구는 안전한 에이전트 AI 시스템 개발을 위한 자동화된 경험적 테스트의 필요성을 강조합니다. 특히, 기존의 강화 학습 문헌에서 간과된 생물학적 및 경제적 동기를 기반으로 한 안전 과제들을 소개하며, 새로운 벤치마크 환경을 구현하여 AI 안전성에 대한 일반적인 논의를 심화시키려 합니다.

- **Technical Details**: 연구에서는 'homeostasis', 'multi-objective agents', 'cooperation'의 세 가지 개발 단계로 분류된 9개의 벤치마크를 구현했습니다. 이 벤치마크들은 gridworld 환경에서 수행되며, DeepMind의 AI Safety Gridworlds와 대부분 호환되어 여러 목표와 에이전트를 지원할 수 있습니다. 또한, 에이전트는 'bounded objectives', 'diminishing returns', 'sustainability', 'resource sharing'과 같은 생물학적 및 경제적 동력학을 이해해야 합니다.

- **Performance Highlights**: 에이전트는 각각의 벤치마크에서 안전성과 성능 목표를 균형 있게 달성해야 합니다. 특히, 안전 목표가 성과 목표보다 우선시되어야 하며, 이는 'utility monsters' 발생을 방지하는 데 중요한 역할을 합니다. 본 연구는 동적 파라미터를 통해 환경의 복잡성을 증가시켜 정책의 강건성을 시험할 수 있도록 지원합니다.



### Interactive Speculative Planning: Enhance Agent Efficiency through Co-design of System and User Interfac (https://arxiv.org/abs/2410.00079)
Comments:
          27 pages, 22 figures

- **What's New**: 논문에서는 인간 중심의 효율적인 에이전트 계획 방법을 제안하며, 이를 통해 LLM 기반 에이전트의 계획 지연 문제를 해결하려고 합니다. 새로운 접근 방식인 Interactive Speculative Planning(상호 작용적 추정 계획)을 도입하여 시스템 설계와 사용자-AI 상호작용 간의 효율성을 높이고자 합니다.

- **Technical Details**: Interactive Speculative Planning은 두 개의 에이전트 시스템, 즉 효율적이지만 능력이 제한된 근사 에이전트와 느리지만 강력한 목표 에이전트를 활용합니다. 근사 에이전트는 작업 단계(차례대로)를 생성하며, 동시에 목표 에이전트는 비동기적으로 다음 단계의 출력을 생성합니다. 이 시스템은 유저가 긴 지연 시간 동안 개입할 수 있도록 하여 전체 프로세스를 가속화합니다.

- **Performance Highlights**: 이 시스템은 사용자 개입을 통해 에이전트 계획 과정의 효율성을 높이고 최종 출력의 정확성을 보장합니다. 논문에서는 실험을 통해 Interactive Speculative Planning 방식의 실제 데이터를 사용한 평가 결과를 제시하고 있으며, 기존 LLM 기반 에이전트 시스템의 지연 문제를 효과적으로 해결할 수 있는 가능성을 보이고 있습니다.



### M2Distill: Multi-Modal Distillation for Lifelong Imitation Learning (https://arxiv.org/abs/2410.00064)
Comments:
          Submitted to ICRA2025

- **What's New**: 이 논문에서는 M2Distill이라는 새로운 multi-modal distillation 기반 방법을 소개합니다. 이는 평생 imitation learning 과정에서 시각, 언어, 행동 분포 간 일관된 latent space를 유지하는 데 초점을 맞추고 있습니다.

- **Technical Details**: M2Distill은 다양한 modality 간의 latent representation 변화 조절을 통해 이전 단계와 현재 단계 간의 일관성을 유지하며, Gaussian Mixture Model (GMM) 정책의 불일치를 줄여 갑작스러운 잊어버림을 방지합니다. 이를 통해 학습된 정책이 이전에 배운 작업을 계속 수행할 수 있도록 보장합니다.

- **Performance Highlights**: LIBERO 평생 imitation learning 벤치마크(Campaign)에서 M2Distill은 LIBERO-OBJECT, LIBERO-GOAL, LIBERO-SPATIAL을 포함하여 모든 평가 지표에서 이전의 최첨단 방법들보다 우수한 성능을 보여주었습니다.



### Neural Decompiling of Tracr Transformers (https://arxiv.org/abs/2410.00061)
- **What's New**: 이 논문에서는 Transformer 아키텍처의 해석력을 높이기 위한 첫 단계를 제안합니다. 이를 위해 Transformer Compiler for RASP(Tracr)를 사용하여 많은 쌍의 transformer weights와 이에 상응하는 RASP 프로그램 데이터셋을 생성하였습니다. 이 데이터셋을 바탕으로, 딥러닝 모델을 훈련시켜 컴파일된 모델에서 RASP 코드를 복구하는 것을 목표로 했습니다.

- **Technical Details**: 기존의 Transformer 모델의 해석 가능성(interpretability)은 주로 수동적인 작업에 의존해왔지만, 본 논문에서는 전체 해석 프로세스를 자동화하는 디컴파일러 모델을 제안합니다. RASP111(Restricted Access Sequence Processing Language) 코드를 통해 transformer weights를 해석하고, RASP 코드와 변환기 가중치 쌍의 대규모 데이터셋을 생성하는 알고리즘을 설계하였습니다. 이 데이터셋은 약 533,000개 프로그램과 222개의 변환기를 포함하고 있습니다.

- **Performance Highlights**: 모델의 실험적 평가 결과, 30% 이상의 테스트 개체에서 정확한 재현을 달성하였으며, 나머지 70%는 소수의 오류로 일반적으로 재현할 수 있었습니다. 또한, 모델에서 생성한 프로그램의 70% 이상이 진짜와 기능적으로 동등하여 Tracr로 컴파일된 transformer weights의 유효한 디컴파일을 의미합니다.



### IDEA: An Inverse Domain Expert Adaptation Based Active DNN IP Protection Method (https://arxiv.org/abs/2410.00059)
- **What's New**: 새로운 논문에서는 모델 소유자가 불법 사용자를 차단하고 침해의 출처를 추적할 수 있는 IDEA(Inverse Domain Expert Adaptation 기반의 능동적 DNN IP 보호 방법)를 제안합니다.

- **Technical Details**: IDEA는 사용자 키를 하이드 스테가노그래픽 기법을 통해 숨겨 사용자의 인증을 수행하며, 진짜 전문가(real expert)와 두 개의 가짜 전문가(fake experts)를 훈련시킵니다. 이 과정에서 서로의 정보를 최소화하는 다중 적응 최적화(multi-adaptive optimization)가 적용됩니다.

- **Performance Highlights**: IDEA는 성능 평가를 위해 5개의 데이터셋과 4개의 DNN 모델에서 광범위한 실험을 진행하였으며, 인증 제어와 범죄자 추적 성공률, 다양한 공격에 대한 강인성에서 효과성을 입증하였습니다.



### Generalizing Consistency Policy to Visual RL with Prioritized Proximal Experience Regularization (https://arxiv.org/abs/2410.00051)
Comments:
          Accepted at the Thirty-Eighth Annual Conference on Neural Information Processing Systems (NeurIPS2024)

- **What's New**: 본 연구에서는 고차원 상태 공간의 비주얼 강화 학습(visual RL)에서 정책 훈련의 불안정성을 완화하기 위해 일관성 정책(consistency policy)의 샘플 기반 엔트로피 정규화(sample-based entropy regularization)와 우선순위 친근 경험 정규화(prioritized proximal experience regularization, CP3ER) 접근 방식을 제안합니다. CP3ER은 DeepMind 제어 스위트와 Meta-world를 포함한 21개 작업에서 새로운 최첨단 성능(SOTA)을 달성했습니다.

- **Technical Details**: 우선 CP3ER은 온라인 강화 학습(online RL)에서 비가역적인 데이터 분포(non-stationary distribution)와 액터-크리틱(actor-critic) 프레임워크가 일관성 정책에 미치는 영향을 조사했습니다. 연구 결과, 액터-크리틱 프레임워크의 Q-손실(Q-loss)이 일관성 모델의 표현 능력을 저해해 불안정한 정책 훈련을 초래하는 것을 발견했습니다. 본 연구에서는 이를 해결하기 위해 정책 훈련을 안정화하는 샘플 기반 엔트로피 정규화를 제안하였습니다.

- **Performance Highlights**: 제안된 CP3ER 방법론은 DeepMind 제어 스위트와 Meta-world 등 21개의 비주얼 제어 작업에서 SOTA 성능을 기록하였으며, 이는 비저장 강화 학습에서 일관성 모델의 응용 가능성을 보여줍니다.



### Epidemiology-Aware Neural ODE with Continuous Disease Transmission Graph (https://arxiv.org/abs/2410.00049)
- **What's New**: 이 논문에서는 전염병의 동적 특성을 반영하고, 질병 전파의 구체적인 메커니즘을 고려하는 새로운 접근 방식인 EARTH(Epidemiology-Aware Neural ODE with Continuous Disease Transmission Graph)를 소개합니다.

- **Technical Details**: EANO(Epidemic-Aware Neural ODE)와 GLTG(Global-guided Local Transmission Graph)를 통해 지역과 전 세계의 질병 전파 패턴을 학습하며, 크로스-어텐션(cross-attention) 기법을 이용하여 중요한 정보들을 통합합니다. 이를 통해 리얼타임(RT) 데이터의 변화를 반영한 전염병 예측을 보다 정교하게 수행할 수 있습니다.

- **Performance Highlights**: EARTH는 COVID-19와 인플루엔자와 같은 다양한 전염병 예측 데이터셋에서 기존의 최첨단 방법들보다 우수한 성능을 보여줍니다.



### Artificial intelligence-based blockchain-driven financial default prediction (https://arxiv.org/abs/2410.00044)
- **What's New**: 이 논문은 블록체인과 인공지능(AI) 기술의 융합이 금융 분야에서 신용 위험 완화 및 금융 시스템 안정화에 대한 새로운 통찰력을 제공함을 강조합니다.

- **Technical Details**: 블록체인 기술은 데이터의 신뢰성을 보장하고 모든 노드에서 일관성을 유지합니다. 기계학습(Machine Learning)은 빅데이터의 세부 분석을 통해 고급 신용 불이행 예측 모델을 구축합니다.

- **Performance Highlights**: 블록체인과 AI를 활용한 금융 불이행 예측은 강력한 응용 프로그램으로, 전통적인 시스템에서의 데이터 저장 및 관리에서의 보안 문제를 해결하며 금융 예측 및 리스크 관리에 있어 큰 장점을 제공합니다.



### Moshi: a speech-text foundation model for real-time dialogu (https://arxiv.org/abs/2410.00037)
- **What's New**: 논문에서는 Moshi라는 새로운 음성-텍스트 기초 모델과 전이 실시간 대화 시스템을 소개합니다. Moshi는 독립적인 음성 인식, 텍스트 대화 및 음성 합성을 통합하여 자연스러운 대화 경험을 실현합니다.

- **Technical Details**: Moshi는 텍스트 언어 모델 백본을 기반으로 하여, Residual Vector Quantization (RVQ) 기법을 통해 음성을 토큰으로 생성하고, 사용자 음성과 모델 음성을 별도의 병렬 스트림으로 모델링합니다. 이로써 발화자 구분을 없애고 대화의 임의적인 다이내믹스를 모델링할 수 있습니다.

- **Performance Highlights**: Moshi는 160ms의 이론적 지연 시간과 실제 200ms의 지연 시간으로 실시간 게속이 가능한 대화형 대량 언어 모델입니다. 이는 자연 대화에 비해 상대적으로 짧은 반응 시간을 자랑하며, 음성 인식 및 음성 합성 기술의 개선으로 뛰어난 음성 품질과 이해도를 제공합니다.



### InsightPulse: An IoT-based System for User Experience Interview Analysis (https://arxiv.org/abs/2410.00036)
Comments:
          Accepted for publication at the 10th IEEE International Conference on Collaboration and Internet Computing (IEEE CIC 2024), Washington D.C., USA

- **What's New**: 이번 논문은 효율적이고 효과적인 사용자 경험(UX) 인터뷰를 지원하기 위한 새로운 시스템인 InsightPulse를 소개합니다. 이 시스템은 IoT 기반 하드웨어와 소프트웨어로 구성되며, 음성 분석 및 인공지능(AI)을 통해 UX 인터뷰 과정을 간소화하고 향상시키는 것을 목표로 합니다.

- **Technical Details**: InsightPulse는 면접 중 실시간으로 핵심 논의 포인트를 식별하고 강조하며, 후속 질문을 제안하고 주제 요약을 자동으로 생성합니다. 이 시스템은 마이크로프로세서와 마이크를 통해 음성을 텍스트로 변환하고, OpenAI API를 통해 자연어 처리를 수행합니다. 데이터를 기반으로 실시간 요약 및 후속 질문 제안 기능을 가집니다.

- **Performance Highlights**: InsightPulse는 인터뷰의 핵심 포인트와 주제 요약을 실시간으로 제공하여 면접관이 대화에 집중하고, 인터뷰 시간을 효과적으로 관리하도록 돕습니다. 또한, 사용자의 개인 정보를 보호하기 위해 RFID 기반 디바이스 활성화 기능을 사용하여 보안을 강화했습니다.



### Strategic Collusion of LLM Agents: Market Division in Multi-Commodity Competitions (https://arxiv.org/abs/2410.00031)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)이 다중 상품 시장에서 자율 에이전트로서 어떻게 전략적으로 행동하는지를 탐색합니다. 구체적으로, Cournot 경쟁 모델 내에서 LLM이 독립적으로 반경쟁적 행위(예: 공모 또는 시장 분할)에 참여할 수 있는지를 조사합니다.

- **Technical Details**: 우리는 LLM 기반 에이전트를 Cournot 경쟁 모델의 다중 상품 변형에 적용하였고, OpenAI의 GPT-4o와 GPT-3.5-turbo 모델을 실험에 사용했습니다. 실험 결과, LLM 기반 에이전트는 효과적으로 시장을 분할하고 공모하는 행동을 보였습니다.

- **Performance Highlights**: LLM 기반 에이전트가 가격을 동적으로 조정하고 자원 배분 전략을 변경하여 특정 상품을 효과적으로 독점할 수 있음을 보여주었습니다. 이는 인간의 직접적인 입력이나 명시적 공모 지시 없이도 수익을 극대화할 수 있음을 나타냅니다.



### Retro-li: Small-Scale Retrieval Augmented Generation Supporting Noisy Similarity Searches and Domain Shift Generalization (https://arxiv.org/abs/2410.00004)
Comments:
          Published as a conference paper at European Conference on Artificial Intelligence 2024

- **What's New**: 본 연구에서는 Retro와 같은 retrieval augmented generation (RAG) 시스템의 개선점을 제안하고, 소규모 데이터베이스에서도 효과적일 수 있다는 사실을 보여줍니다. 특히, 보다 정확한 이웃 검색을 통해 더 나은 결과물을 도출할 수 있다고 강조합니다.

- **Technical Details**: Retro-li는 소규모 비모수 메모리(non-parametric memory)에서 높은 품질의 이웃을 검색하기 위해 적절한 의미 유사성 검색(semantic similarity search)을 사용합니다. 또한, 이웃 검색 중 노이즈를 줄이기 위해 처음으로 정규화(regularization)를 추가하여 불확실성을 줄이는 데 기여합니다. RAG 모델의 데이터베이스 업데이트는 뛰어난 효율성을 자랑하며, 사용자는 도메인 간 전환 없이도 데이터베이스를 쉽게 대체할 수 있습니다. 또한, Retro-li는 아날로그 메모리 인메모리 컴퓨팅(hardware)에서 O(1) 검색 시간을 실현할 수 있습니다.

- **Performance Highlights**: Retro-li는 기존의 대규모 데이터베이스 대신 수백만 개의 토큰을 갖는 소규모 데이터베이스에서도 언어 모델링 성능 향상을 보여주며, 도메인 변화에 대한 일반화 능력이 향상되었습니다. 노이즈가 존재하는 상태에서도 성능 저하가 1% 미만으로 유지되며, 사용자는 특정 도메인에 맞춘 새로운 데이터베이스를 쉽게 구축할 수 있습니다.



### Linear Projections of Teacher Embeddings for Few-Class Distillation (https://arxiv.org/abs/2409.20449)
- **What's New**: 이 논문에서는 기존 Knowledge Distillation(KD) 접근 방식의 한계를 극복하기 위해 Learning Embedding Linear Projections(LELP)라는 새로운 기법을 제안합니다. LELP는 교사 모델의 마지막 레이어에서의 표현을 이용해 학습하는 방법으로, 교사의 내재적 패턴을 효과적으로 포착할 수 있도록 설계되었습니다.

- **Technical Details**: LELP는 교사 모델의 임베딩 공간에서 유용한 선형 부분 공간(informative linear subspaces)을 탐색하고 이를 가상의 서브클래스(pseudo-subclasses)로 분할하여 학생 모델이 이를 모방하도록 학습합니다. 이 과정에서 단일 통합 크로스 엔트로피 손실(unified cross-entropy loss)을 사용하여 훈련을 수행합니다.

- **Performance Highlights**: 대규모 NLP 벤치마크인 Amazon Reviews와 Sentiment140에서의 실험 결과, LELP는 기존의 상태-최고(distillation algorithms) 기법에 비해 이진 및 소수 클래스 문제에서 일관되게 경쟁력을 갖추고 있으며, 대체로 우수한 성능을 보였습니다. LELP는 데이터 효율성(data efficiency), 훈련 속도(tyaining speed) 개선 및 반지도 학습(semi-supervised) KD 시나리오에서의 실질적인 향상을 제공합니다.



### AutoPureData: Automated Filtering of Web Data for LLM Fine-tuning (https://arxiv.org/abs/2406.19271)
Comments:
          Initial version

- **What's New**: 이 연구는 웹 데이터를 수집하고 불필요한 텍스트를 자동으로 필터링하는 시스템을 제안합니다. 기존의 신뢰할 수 있는 AI 모델을 이용하여 데이터 품질과 안전성을 확보하면서, AI 모델의 최신 정보를 반영하기 위한 방법입니다.

- **Technical Details**: 제안된 시스템은 FineWeb 데이터셋을 활용하여 웹에서 수집한 데이터를 필터링합니다. LlamaGuard 2와 Llama 3라는 두 가지 LLM을 사용하여 데이터를 플래그하고, 불법 콘텐츠와 Bias를 식별하는 데 주력합니다. 모델 성능은 F-1 점수 91.5%와 4%의 잘못된 긍정률로 확인되었습니다.

- **Performance Highlights**: 실험 결과, 100개의 웹 데이터 샘플 중 32개의 행이 불필요한 텍스트로 플래그되었습니다. 이 시스템은 데이터 품질 향상과 함께 데이터 수집 및 전처리의 시간과 비용을 크게 줄이는 데 기여합니다.



### Maia-2: A Unified Model for Human-AI Alignment in Chess (https://arxiv.org/abs/2409.20553)
Comments:
          Accepted @ NeurIPS 2024

- **What's New**: 이 연구는 체스에서 인간과 AI의 정렬(human-AI alignment)을 위해 통합 모델링 접근법인 Maia-2를 제안합니다. Maia-2는 다양한 기술 수준에서의 인간 스타일을 일관성 있게 포착하고, 플레이어의 기술 향상을 직접적으로 반영합니다.

- **Technical Details**: Maia-2는 체스 위치를 특징적으로 변환하는 표준 residual network 타워와, 플레이어의 기술 수준을 동적으로 통합할 수 있는 skill-aware attention 메커니즘으로 구성되어 있습니다. 이 모델은 현재 보드 위치만을 입력으로 받아, 훈련 시간을 상당히 줄이고 유연성을 증가시킵니다.

- **Performance Highlights**: Maia-2는 move prediction accuracy에서 이전의 Maia 모델보다 2%포인트 이상 개선되었으며, 모든 기술 수준에서 다른 모델들을 초월했습니다. 또한, 응답의 일관성에서는 이전 모델의 1%에 비해 27%를 단조롭게 처리하며, 인간 플레이어가 지속적으로 그리고 부드럽게 향상하는 방식과 일치합니다.



### Efficient Driving Behavior Narration and Reasoning on Edge Device Using Large Language Models (https://arxiv.org/abs/2409.20364)
Comments:
          Submitted for possible journal publication

- **What's New**: 본 논문에서는 자율주행 기술의 발전을 위해 대규모 언어 모델(LLMs)과 엣지 컴퓨팅(edge computing)을 통합한 새로운 프레임워크를 제안합니다. 이 프레임워크는 도로변 장치(RSU)에서 LLMs를 배포하여 주행 행동을 설명하고, 사고 감지 시 신속하게 정보를 전달할 수 있는 구조를 가지고 있습니다.

- **Technical Details**: 제안된 프레임워크는 5G NR/NSA 네트워크를 통해 연결된 다수의 RSU로 구성되어 있으며, 각 RSU는 자신이 관할하는 지역의 교통 데이터를 수집하고 처리합니다. LLMs는 주행 행동을 분석하여 자연어 설명을 생성하며, 청각적 신호와 환경 정보를 통합한 3중 프롬프트 전략을 사용하여 성능을 향상시킵니다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식은 0.5초 이하의 프레임 응답 시간을 달성하였으며, 네 가지 LLM의 서사 정확도는 70% 이상, 가장 높은 추론 정확도는 81.7%에 달했습니다. 이는 자율주행 시나리오에서 처리를 최적화하고, 명확한 커뮤니케이션을 가능하게 함으로써 시스템의 안정성과 적시성을 높입니다.



### OM4OV: Leveraging Ontology Matching for Ontology Versioning (https://arxiv.org/abs/2409.20302)
Comments:
          7 pages, 7 figures, 1 table

- **What's New**: 본 논문에서는 기존의 ontology matching (OM) 기술을 활용하여 ontology version control (OV)을 위한 새로운 접근 방식을 제안합니다. OM과 OV 간의 상보적 관계를 탐구하고, 두 작업을 동일한 파이프라인으로 처리할 수 있는 방법을 제시합니다.

- **Technical Details**: 논문에서 제안하는 OM4OV 파이프라인은 신규 작업 정의, 성과 측정 및 검증용 데이터셋 구축을 통해 OV 작업에 OM을 효과적으로 활용합니다. 특히, cross-reference 메커니즘을 도입하여 OV 후보 선택을 최적화하고 OM의 전반적인 성능을 향상시킵니다.

- **Performance Highlights**: OAEI 데이터셋을 사용하여 OM4OV 파이프라인과 cross-reference 메커니즘의 성능을 실험적으로 검증하였습니다. 이 연구는 기계 생성된 버전 정보를 활용한 경량화 및 완전 자동화된 OV 접근 방식을 제시하여, 기존 OM 시스템과 기술을 OV 작업으로 이전할 수 있는 새로운 가능성을 열었습니다.



### Probabilistic Answer Set Programming with Discrete and Continuous Random Variables (https://arxiv.org/abs/2409.20274)
Comments:
          Under consideration in Theory and Practice of Logic Programming (TPLP)

- **What's New**: 이번 논문에서는 Probabilistic Answer Set Programming (PASP)의 프레임워크를 확장하여 혼합 확률적 답 집합 프로그래밍(Hybrid Probabilistic Answer Set Programming, HPASP)을 제안합니다. 이는 불확실한 정보를 나타내는 확률적 사실을 다룰 수 있는 새로운 능력을 포함하고 있습니다.

- **Technical Details**: 논문에서는 HPASP를 구현하기 위해 연속 확률 변수를 지원하며, 이를 위해 불연속 확률 변수를 갖는 일반적인 확률적 답 집합 프로그램으로 변환하는 '분별화(discretization)' 과정을 설명합니다. 두 가지 정확한 알고리즘(프로젝션 답 집합 열거와 지식 컴파일 기반)과 두 가지 근사 알고리즘(샘플링 기반)을 제안했습니다.

- **Performance Highlights**: 실험 결과, 정확한 추론은 작은 인스턴스에서만 가능하지만, 지식 컴파일이 성능 향상에 상당한 긍정적인 영향을 미칩니다. 샘플링 알고리즘은 더 큰 인스턴스를 처리할 수 있으나, 메모리 요구량이 증가할 수 있습니다.



### Learning to Ground Existentially Quantified Goals (https://arxiv.org/abs/2409.20259)
Comments:
          11 pages, Accepted at the 21st International Conference on Principles of Knowledge Representation and Reasoning (KR2024) in the Reasoning, Learning, and Decision Making track

- **What's New**: 이 연구에서는 독특한 이름이 없는 객체를 대상으로 하는 목표 표기 문제를 해결하기 위해 새로운 기계 학습 접근 방식을 제안합니다. 특히, GNN(그래프 신경망) 아키텍처를 활용하여 부분적으로 정량화된 목표의 비용을 예측합니다.

- **Technical Details**: 이 연구에서는 GNN 아키텍처를 활용하여 여러 계획 도메인에서 목표 변수를 바인딩하는 방법을 실험적으로 평가합니다. 목표 변수는 순차적으로 하나씩 그라운딩(grounding)되며, 기존 방법론을 사용하여 완전히 그라운딩된 문제에 대한 계획을 생성할 수 있습니다.

- **Performance Highlights**: 제안된 GNN 기반 접근 방식은 다양한 목표 변수를 갖는 복잡한 인스턴스에서 우수한 일반화 성능을 보였습니다. 실험 결과, 이 방법이 고전적 계획 문제와 제약 추론 문제의 경계에 있는 다양한 문제에 대해 효과적임을 입증했습니다.



### Inferring Preferences from Demonstrations in Multi-objective Reinforcement Learning (https://arxiv.org/abs/2409.20258)
Comments:
          Neural Comput & Applic (2024)

- **What's New**: 이번 연구에서는 동적 가중치 기반 선호 추정 알고리즘(DWPI)을 제안하여, 다중 목표 의사결정 문제에서 인간이나 에이전트의 선호를 직접적으로 알 수 없는 상황에서도 시연(Demonstration) 데이터를 통해 선호를 추정할 수 있게 합니다.

- **Technical Details**: DWPI 알고리즘은 심층 신경망(deep neural network) 모델을 활용하여, DWMORL(Dynamic Weight Multi-Objective Reinforcement Learning) 에이전트가 생성한 특성과 목표 가중치 집합을 바탕으로 선호를 추론합니다. 이 알고리즘은 사용자와의 상호작용이 없으며, 하위 최적 시연(sub-optimal demonstrations)에 대해서도 강건한 성능을 보입니다.

- **Performance Highlights**: DWPI 알고리즘은 세 가지 환경(Convex Deep Sea Treasure, Traffic, Item Gathering)에서 성능을 평가한 결과, 기존 알고리즘에 비해 시간 효율성과 추론 정확도가 모두 개선되었습니다. 또한, 알고리즘의 일반화 능력도 평가되어 다양한 목표 수에 적응할 수 있는 능력을 입증하였습니다.



### MemSim: A Bayesian Simulator for Evaluating Memory of LLM-based Personal Assistants (https://arxiv.org/abs/2409.20163)
Comments:
          26 pages, 25 tables, 1 figure

- **What's New**: 이 논문에서는 LLM(대형 언어 모델) 기반 개인 비서의 메모리 능력을 평가하기 위한 새로운 자동화된 방법인 MemSim을 제안합니다. MemSim은 사용자 메시지로부터 신뢰할 수 있는 질문-답변(QA)을 자동으로 생성할 수 있는 베이esian 시뮬레이터입니다.

- **Technical Details**: MemSim은 Bayesian Relation Network(BRNet)와 원인 생성 메커니즘을 사용하여 사용자 프로필을 다양한 계층적으로 생성하고, 이들로부터 신뢰할 수 있는 QA를 만듭니다. BRNet은 사용자에 대한 엔티티와 속성의 확률 분포를 모델링하는 기능을 갖추고 있으며, 구성된 QA는 메모리 메커니즘을 평가하는 데 사용됩니다.

- **Performance Highlights**: MemSim을 기반으로 생성된 MemDaily 데이터셋은 일상적인 시나리오에서 메모리 능력을 평가하기 위한 알고리즘의 효과를 평가하기 위한 실험에서 광범위하게 테스트되었습니다. 이 연구는 LLM 기반 개인 비서의 메모리 평가에서 첫 번째로 객관적이고 자동적인 방법을 제시하며, 연구 커뮤니티를 위해 이 프로젝트를 공개했습니다.



### Reevaluation of Inductive Link Prediction (https://arxiv.org/abs/2409.20130)
Comments:
          Published in RuleML+RR 2024

- **What's New**: 이 논문에서는 현재 사용되고 있는 유도 링크 예측(inductive link prediction) 평가 프로토콜이 중대한 결함을 가지고 있다고 주장합니다. 이 프로토콜은 무작위로 샘플링된 부정 엔티티(negative entities) 집합에서 진짜 엔티티를 순위 매기기 때문에 문제를 야기합니다.

- **Technical Details**: 논문에서는 기존의 유도 링크 예측 방법을 여러 기준 벤치마크(benchmarks)에서 재평가합니다. 일반적으로 전이 설정(transductive setting)에서 적용되는 링크 예측 프로토콜을 사용하여 평가하며, 몇몇 유도 방법들은 이 설정에서 확장성(scalability) 문제로 인해 성능이 저하됩니다. 이 문제를 해결하기 위한 개선된 샘플링 프로토콜(sampling protocol)을 제안하고 적용합니다.

- **Performance Highlights**: 우리의 평가 결과는 지금까지 보고된 결과들과 크게 다릅니다. 간단한 규칙 기반(baseline) 모델이 유형의 유효성(validity)에 따라 엔티티를 더 높은 순위로 매김으로써 최첨단(state-of-the-art) 결과를 달성할 수 있습니다.



### GUNDAM: Aligning Large Language Models with Graph Understanding (https://arxiv.org/abs/2409.20053)
- **What's New**: 이 논문에서는 텍스트 데이터 처리에서 뛰어난 성능을 보여준 Large Language Models (LLMs)를 그래프 구조 데이터를 이해하고 활용하는 데 적용하려는 새로운 접근 방식인 GUNDAM (Graph Understanding for Natural Language Driven Analytical Model)을 소개합니다. 기존 연구들이 주로 텍스트 속성이 풍부한 그래프에 초점을 맞춘 반면, 본 연구는 그래프 데이터 고유의 구조적 지식을 토대로 복잡한 추론 작업을 수행할 수 있는 LLM의 능력을 평가하고 향상시키고자 합니다.

- **Technical Details**: GUNDAM 모델은 그래프 구조를 LLM에 인코딩하기 위해 Graph Projection 방법을 사용합니다. 또한 CoT (Chain of Thought) 추론 경로를 포함한 고품질 그래프 추론 데이터 생성 파이프라인을 개발하여, 그래프 알고리즘을 활용해 정확성 및 중간 추론 과정을 제공합니다. 마지막으로, Alignment Tuning 방법을 통해 그래프 추론 데이터를 통한 GUNDAM의 미세 조정을 진행하여 모델의 추론 능력을 강화합니다.

- **Performance Highlights**: 실험 평가 결과, GUNDAM은 현재의 최첨단(SOTA) 성능을 초과 달성하였으며, LLM의 그래프 추론 능력에 영향을 미치는 주요 요소들도 밝혀졌습니다. 이 모델은 복잡한 그래프 구조를 이해하고 추론할 수 있는 능력을 개선하여 LLM의 일반 지능 발전에 기여할 가능성을 가지고 있습니다.



### Personalisation via Dynamic Policy Fusion (https://arxiv.org/abs/2409.20016)
- **What's New**: 이 연구에서는 이미 훈련된 딥 강화 학습(Deep Reinforcement Learning, RL) 정책을 사용자의 특정 요구에 맞게 조정하는 새로운 접근 방식을 제안합니다. 기존의 재훈련 방법을 피하고 사람의 피드백을 활용하여 제로샷(zero-shot) 방식으로 개인화된 정책을 학습할 수 있습니다.

- **Technical Details**: 우리는 LSTM(Long Short-Term Memory) 기반의 방법을 사용하여 사용자 의도를 추론하고, 훈련 중에 수집된 경로에 대한 피드백을 결합하여 개인화된 정책을 생성합니다. 이 동적 정책 융합(dynamic policy fusion) 접근 방식은 Boltzmann 분포의 온도 매개변수를 조절하여 사용자 요구와 작업 목표를 균형 있게 달성할 수 있도록 합니다.

- **Performance Highlights**: 본 연구에서 제안하는 동적 정책 융합 방법은 다양한 환경에서 최적의 작업을 수행하면서 사용자 요구를 지속적으로 충족시키는 것을 보여주었습니다. 이는 정적 정책 융합(static policy fusion) 방법의 한계를 극복하는 결과를 도출했습니다.



### Customized Information and Domain-centric Knowledge Graph Construction with Large Language Models (https://arxiv.org/abs/2409.20010)
Comments:
          Presented at CAIPI Workshop at AAAI 2024

- **What's New**: 본 논문에서는 지식 그래프(Knowledge Graph)를 기반으로 한 새로운 접근 방식을 제안하여 체계적인 정보에 신속하게 접근할 수 있도록 하고, 실행 가능한 기술 인텔리전스(actionable technology intelligence)를 지원하며 사이버 물리 시스템(cyber-physical systems) 계획 개선에 기여하고자 합니다.

- **Technical Details**: 제안하는 프레임워크는 텍스트 마이닝(text mining) 프로세스를 포함하며, 정보 검색(information retrieval), 키프레이즈 추출(keyphrase extraction), 의미 네트워크(semantic network) 생성 및 주제 맵 시각화(topic map visualization) 등을 포함합니다. 이 데이터 탐색 과정을 통해 선택적 지식 그래프 구축(selective knowledge graph construction) 접근법을 적용하며, 전자 및 혁신 온톨로지(ontology)를 기반으로 한 파이프라인을 통해 멀티-목표 의사결정(multi-objective decision-making)을 지원합니다. 자동차 전기 시스템(domain of automotive electrical systems) 분야에 이 방법론을 적용하여 확장 가능성을 입증하였습니다.

- **Performance Highlights**: 본 연구의 결과에 따르면, 제안한 지식 그래프 구축 프로세스는 GraphGPT 및 bi-LSTM과 transformer REBEL과 비교했을 때 클래스 인식(class recognition), 관계 구축(relationship construction), 올바른 'subclass of' 분류에서 여러 배로 우수한 성능을 보였습니다. 또한, 우선 인증된 문서(document genres)를 통해 생성된 지식 그래프는 보다 방대한 정보량과 더 나은 일관성을 갖춘 결과를 도출하는 데 크게 기여하였습니다.



### Law of the Weakest Link: Cross Capabilities of Large Language Models (https://arxiv.org/abs/2409.19951)
Comments:
          Code: this https URL

- **What's New**: 이번 연구에서는 Large Language Models (LLMs)의 평가에서 개별 능력에 중점을 두었던 기존 접근에서 벗어나, 여러 전문 분야의 교차 능력(cross capabilities) 간의 상호작용을 탐구합니다.

- **Technical Details**: 연구팀은 7개의 핵심 개별 능력을 정의하고, 이를 엮어 7개의 일반적인 교차 능력을 생성하였습니다. 이 과정에서 수동으로 구축된 분류 체계(taxonomy)를 지원하였으며, 1,400개의 인간 주석이 달린 프롬프트로 구성된 CrossEval이라는 벤치마크를 도입하였습니다. 각 개별 및 교차 능력마다 100개의 프롬프트가 배정되며, 4,200개의 모델 응답을 평가하여 8,400개의 인간 평가를 수집하였습니다.

- **Performance Highlights**: 연구 결과, 현재 LLM들은 교차 능력 성능이 가장 약한 구성 요소에 의해 강하게 제약된다는 "Law of the Weakest Link" 현상을 보였습니다. 17개 모델의 58개 교차 능력 점수 중 38개는 모든 개별 능력보다 낮았으며, 20개는 강한 능력과 약한 능력 사이에 위치하였습니다. 이 결과는 LLM들이 교차 능력 과제에서 저조한 성과를 내고 있음을 보여주며, 복잡한 다차원 상황에서 성능을 최적화하기 위해 미래 연구의 우선 과제가 약한 능력의 식별 및 개선이라는 점을 강조합니다.



### On The Planning Abilities of OpenAI's o1 Models: Feasibility, Optimality, and Generalizability (https://arxiv.org/abs/2409.19924)
Comments:
          Updated link to code repository

- **What's New**: 이 연구에서는 OpenAI의 o1 모델이 복잡한 추론 작업을 수행하는 데 있어 보여준 능력을 바탕으로 플래닝(Planning) 능력을 평가합니다. 특히 세 가지 주요 측면인 실행 가능성(feasibility), 최적성(optimality), 일반화(generalizability)에 중점을 두었습니다.

- **Technical Details**: 연구는 제약이 많은 작업(예: Barman, Tyreworld)과 공간적으로 복잡한 환경(예: Termes, Floortile)에서 o1-preview 모델의 성능을 분석합니다. o1 모델은 Chain-of-Thought (CoT) 및 Tree-of-Thought (ToT)와 같은 고급 추론 기술을 적용하여 강화 학습으로 훈련되었습니다. 이 모델은 계획 수립에서 크리티컬한 요소인 메모리 관리(memory management)가 미비하며, 결정-making에서 병목 현상을 드러내고 있습니다.

- **Performance Highlights**: 이 모델은 GPT-4보다 작업 제약을 더 잘 준수하는 성능을 보여주었으나, 종종 비효율적인 행동을 포함한 최적이 아닌 솔루션을 생성하며 공간적으로 복잡한 작업에서 효과적으로 일반화하는 데 어려움을 겪습니다. 연구 결과는 LLM 기반의 플래닝 개선을 위한 중요한 통찰력을 제공하며, 메모리 관리 및 의사 결정 개선 방향을 제시합니다.



### Analysis on Riemann Hypothesis with Cross Entropy Optimization and Reasoning (https://arxiv.org/abs/2409.19790)
Comments:
          13 pages, 3 figures

- **What's New**: 이 논문에서는 Riemann Hypothesis를 분석하기 위한 새로운 프레임워크를 제시합니다. 이 프레임워크는 세 가지 주요 요소로 구성됩니다: a) cross entropy 최적화 및 추론을 통한 probabilistic modeling; b) 대수의 법칙 (law of large numbers)의 적용; c) 수학적 귀납법 (mathematical inductions)의 적용.

- **Technical Details**: 분석은 주로 rare event simulation 기술을 활용한 cross entropy 최적화 및 추론의 probabilistic modeling을 통해 수행됩니다. Riemann Hypothesis를 분석하기 위해 대수의 법칙과 수학적 귀납법을 적용하여 전체 복소 평면이 포함되도록 합니다. 또한, 대형 언어 모델(LLMs)에서 next token 예측이 현재 라운드의 각 가능한 토큰의 추정 확률에만 기반하지 않고, 여러 top-k chain of thoughts (CoTs) 경로 간의 누적 경로 확률에 기반하여 추론하는 방법인 enhanced top-p sampling을 논의합니다.

- **Performance Highlights**: 논문에서 제시된 프레임워크와 기술들은 최근의 대형 언어 모델 (LLMs)의 Chain of Thought (CoT) 또는 Diagram of Thought (DoT) 추론과 강화 학습 (reinforcement learning) 발전과 결합하여 Riemann Hypothesis의 궁극적인 증명을 위한 길을 열 수 있기를 희망합니다.



### Local Search for Integer Quadratic Programming (https://arxiv.org/abs/2409.19668)
- **What's New**: 본 논문에서는 Integer Quadratic Programming (IQP) 문제를 위한 효율적인 로컬 탐색 솔버인 LS-IQCQP를 개발하였습니다. 이 연구는 IQP 문제 해결을 위한 로컬 탐색 알고리즘에 대한 초기 연구단계에서 중요한 진전을 이루었습니다.

- **Technical Details**: LS-IQCQP는 목적함수, 제약조건, 또는 두 가지 모두에 대해 이차항을 포함할 수 있는 일반적인 IQP를 처리할 수 있는 네 가지 새로운 로컬 탐색 연산자를 제안합니다. 또한, 새로 설계된 점수 함수(score functions)를 활용하여 탐색 과정을 향상시키는 두 가지 모드 로컬 탐색 알고리즘이 도입되었습니다.

- **Performance Highlights**: LS-IQCQP는 QPLIB 및 MINLPLIB의 표준 IQP 벤치마크에서 몇몇 최신 IQP 솔버들과 비교하여 실험을 수행하였으며, Gurobi와 같은 강력한 상업 솔버와 경쟁력을 갖추고 있고, 다른 최신 솔버들보다 우수한 성능을 보임을 입증하였습니다. 또한, LS-IQCQP는 QPLIB과 MINLPLIB의 오픈 인스턴스에서 6개의 새로운 기록을 세웠습니다.



### An action language-based formalisation of an abstract argumentation framework (https://arxiv.org/abs/2409.19625)
Comments:
          To be published in The 25th International Conference on Principles and Practice of Multi-Agent Systems

- **What's New**: 이 논문에서는 전통적인 추상 논쟁(framework)에서 간과되었던 주장의 발화 순서(enunciation order)를 고려한 새로운 논쟁 그래프 모델을 제안합니다. 이 모델은 각 대화에 대해 고유한 결과인 확장(extension)을 도출할 수 있는 방법을 제공합니다. 또한, 논문의 주요 기여는 '최후 발화 최후 업데이트(last enunciated last updated)' 전략을 기반으로 한 이전 변환의 수정입니다.

- **Technical Details**: 본 연구는 주장의 발화 순서를 포함하는 라벨링 전이 시스템(Labelled Transition System, LTS)을 이용하여 추상 논쟁 그래프를 모델링합니다. 이 시스템은 인과관계(causality) 및 완전성(completeness) 관련 속성을 정립하며, 행동 기술 언어(Action Description Language, ADL)를 사용하여 논쟁 과정을 모델링합니다. 논문은 AAF의 종료(termination) 및 정확성(correctness) 성질을 보장합니다.

- **Performance Highlights**: 제안하는 방법은 주장이 제시되는 순서를 모델에 포함시켜 각 주장의 수용 가능성을 실시간으로 업데이트합니다. 이를 통해 논쟁 과정을 보다 명확히 하며, 기존의 논쟁 시스템과 비교하여 효과적인 결과를 도출합니다.



### BuildingView: Constructing Urban Building Exteriors Databases with Street View Imagery and Multimodal Large Language Mod (https://arxiv.org/abs/2409.19527)
Comments:
          8 pages, 6 figures

- **What's New**: Urban Building Exteriors의 중요성이 커지면서, Google Street View와 OpenStreetMap의 공간 정보를 통합하는 BuildingView라는 혁신적인 접근 방식을 제안합니다. 이 연구는 도시 건물 외관 데이터의 정확성을 높이고 핵심 지속 가능성 및 디자인 지표를 도출하여 관리할 수 있는 프레임워크를 개발하였습니다.

- **Technical Details**: BuildingView는 Street View 이미지와 멀티모달 대형 언어 모델(LLMs)을 결합하여 도시 건물 외관 데이터베이스를 만드는 혁신적 방법론입니다. 연구 방법론은 문헌 조사, 건물 및 Street View 샘플링, ChatGPT-4O API를 사용한 주석 작업으로 구성됩니다.

- **Performance Highlights**: 뉴욕, 암스테르담, 싱가포르의 데이터로 검증된 결과, BuildingView는 도시 연구를 위한 포괄적인 도구를 제공하며, 도시 계획, 건축 디자인 및 환경 정책에 대한 정보 기반 의사 결정을 지원합니다.



### Bridging the Gap in Hybrid Decision-Making Systems (https://arxiv.org/abs/2409.19415)
- **What's New**: BRIDGET라는 새로운 인간-기계 협력 시스템이 도입되었습니다. 이 시스템은 라벨링된 데이터셋에서 사용자가 레코드를 라벨링하도록 돕고, 인간과 기계 간의 상호작용을 통해 두 가지 하이브리드 의사결정 패러다임 간의 간극을 메우고자 합니다.

- **Technical Details**: BRIDGET는 두 가지 상태(상태 전환)를 동적으로 지원하며, 이는 기계가 또는 인간 사용자가 주도하는 방식으로 운영됩니다. 이 시스템은 Incremental Learning (IL) 모델을 사용하며, 작은 데이터 배치 단위로 지속적으로 훈련됩니다. Skeptical Learning (SL) 및 Learning-to-Defer (LtD) 접근법을 더 효과적으로 통합하여 가변적인 시나리오에 적합하도록 설계되었습니다.

- **Performance Highlights**: BRIDGET는 기존 하이브리드 의사결정 시스템의 한계를 극복하고, 사용자와 머신 모델 간의 상호작용을 개선하여 결정의 일관성을 높이고, 강력한 의사결정 지원을 제공합니다. 이 시스템은 다양한 시나리오에 따라 적응할 수 있는 해석 가능한 모델로서, 향후 인간과 기계의 협력적인 의사결정 시스템의 기초를 마련할 것입니다.



### Fairness Analysis with Shapley-Owen Effects (https://arxiv.org/abs/2409.19318)
- **What's New**: 이번 연구는 Shapley-Owen 효과의 상대적 중요성과 공정한 귀속을 측정하는 방법을 제안합니다. 연구자들은 Shapley-Owen 효과를 계산하는 데 필요한 복잡성을 줄이기 위해 분광 분해(spectral decomposition)를 활용하여 이를 두 가지 부분으로 나눕니다: 모델에 독립적인 부분과 모델에 의존적인 부분입니다.

- **Technical Details**: Shapley-Owen 효과의 계산은 모델 독립적인 부분과 모델 의존적인 부분으로 나뉩니다. 모델 독립적인 부분은 한 번의 사전 계산으로 처리될 수 있으며, 모델에 대한 귀속은 다항식 혼돈 확장(polynomial chaos expansion, PCE)의 계수에 따라 분석적으로 표현됩니다. 또한, PCE의 정확하고 희소한 절단에 대한 알고리즘을 제안하고, 누적 근사 오류의 상한도 제공합니다.

- **Performance Highlights**: PCE와 Shapley-Owen 효과 모두의 근사값이 실제 값으로 수렴함을 보입니다. 이 연구는 더 다양한 Shapley-Owen 효과를 계산하는 데 재사용 가능한 모델 특정 계수를 제공하여 계산 성능을 향상시킵니다.



### bnRep: A repository of Bayesian networks from the academic literatur (https://arxiv.org/abs/2409.19158)
- **What's New**: 이 논문은 Bayesian networks (BNs)을 위한 포괄적인 문서화된 컬렉션인 bnRep라는 오픈 소스 R 패키지를 소개합니다. 이 패키지는 200개 이상의 BNs와 그에 대한 상세한 문서를 제공하여 벤치마킹, 복제 가능성 및 교육을 지원합니다.

- **Technical Details**: bnRep 패키지는 150개 이상의 학술 출처에서 수집된 200개 이상의 BNs로 구성되어 있으며, 각 네트워크는 자세한 문서와 함께 제공됩니다. R은 통계 모델링을 위한 견고한 생태계와 기존 도구와의 통합이 용이하여 개발 플랫폼으로 선택되었습니다.

- **Performance Highlights**: bnRep는 사용자가 인터랙티브한 도구를 통해 네트워크를 탐색하고 비교할 수 있게 지원하며, 다양한 분야에서의 활용을 가능하게 합니다.



### Intention-aware policy graphs: answering what, how, and why in opaque agents (https://arxiv.org/abs/2409.19038)
Comments:
          57 pages, 8 figures, 5 tables

- **What's New**: 이번 연구에서는 복잡한 환경에서 상호작용하는 AI 기반 소프트웨어인 에이전트의 출현 행동(emergent behaviour)을 설명하는 새로운 방법을 제안합니다. 이는 신뢰할 수 있는 AI를 배치하는 데 필수적인 요소입니다.

- **Technical Details**: 우리는 에이전트의 행동을 신중하게 고찰할 수 있도록 하는 Probabilistic Graphical Model(확률적 그래픽 모델)과 이를 설계하기 위한 파이프라인을 제안합니다. 이 모델은 에이전트가 현재 갖고 있는 의도(intention)에 대한 강력한 수치적 값을 산출할 수 있게 해줍니다.

- **Performance Highlights**: 제안된 모델을 통해 에이전트의 행동과 세계 상태에 대한 부분적인 관찰을 기반으로 해석 가능성과 신뢰성을 평가하는 측정을 제시합니다. 이를 통해 '지금 무엇을 하고 싶습니까?'(예: 수프 배달), '어떻게 계획하고 있습니까?'(예: 자신의 기술과 세계를 고려한 계획 반환), '왜 이 상태에서 이 행동을 취합니까?'(예: 자신의 목표를 추진하거나 방해하는 방식으로 설명) 등의 질문을 가능하게 합니다.



### Continuously Improving Mobile Manipulation with Autonomous Real-World RL (https://arxiv.org/abs/2409.20568)
Comments:
          CoRL 2024. Website at this https URL

- **What's New**: 우리는 사람의 감독이나 광범위한 장비 없이 정책을 학습할 수 있는 완전 자율 모바일 조작(Manipulation) RL(강화학습) 프레임워크를 제안합니다. 이 접근 방식은 작업 관련 자율성(task-relevant autonomy), 효율적인 정책 학습 및 보상을 형성하는 방식이 결합되어 있습니다.

- **Technical Details**: 우리의 방법은 Spot 로봇이 4가지 최첨단 모바일 조작 업무에 대해 평균 80% 성공률을 보이는 것을 입증했습니다. 이 방법은 기존 방법에 비해 3-4배 향상된 성능을 보여줍니다. 주요 구성 요소로는 이미지 인코더, 벡터 관측치, 객체 탐지 모델 등이 포함됩니다.

- **Performance Highlights**: 우리는 각 태스크에 대해 수행된 실험 결과를 소개합니다. 의자 이동, 쓰레기통 세우기, 쓰기 작업 등 다양한 작업에서 성공률 80%를 기록했습니다. 각 작업에서 특정 임무에 대한 보상이 계산되며, 이 보상의 변화로 인해 로봇의 성능이 기하급수적으로 향상됩니다.



### LaMMA-P: Generalizable Multi-Agent Long-Horizon Task Allocation and Planning with LM-Driven PDDL Planner (https://arxiv.org/abs/2409.20560)
Comments:
          Project website: this https URL

- **What's New**: 이 논문에서는 LaMMA-P라는 새로운 다중 에이전트 (Multi-Agent) 과업 계획 프레임워크를 제안합니다. 이는 언어 모델 (LMs)과 전통적인 탐색 계획기 (search planner)의 강점을 통합하여 긴 기간의 작업을 효과적으로 처리하는 방법을 제시합니다.

- **Technical Details**: LaMMA-P는 Planning Domain Definition Language (PDDL)와 대형 언어 모델 (LLMs)의 추론 능력을 결합하여 다중 로봇 시스템에서 긴 기간 과업 할당 및 실행을 용이하게 합니다. LLM의 구성 요소는 각 로봇의 기술을 기반으로 하여 하위 작업을 식별하고 할당하며, 각 로봇의 도메인에 대한 PDDL 문제 설명을 생성합니다. 또한 Fast Downward 계획기를 사용하여 초기 계획이 실패할 경우, LLM이 계획을 재생성하고 조정하여 실행 가능한 솔루션을 생성합니다.

- **Performance Highlights**: 실험 결과, LaMMA-P는 기존 LM 기반 다중 에이전트 계획기보다 105% 높은 성공률과 36% 높은 효율성을 보여주었습니다. 더불어 MAT-THOR라는 종합 벤치마크를 통해 다양한 복잡성의 가정 작업을 포함한 성능 평가를 수행했습니다.



### LLM Hallucinations in Practical Code Generation: Phenomena, Mechanism, and Mitigation (https://arxiv.org/abs/2409.20550)
Comments:
          11 pages, 13 figures

- **What's New**: 이번 연구는 대형 언어 모델(LLM) 기반의 코드 생성 과정에서의 환각(hallucinations) 문제를 체계적으로 조사합니다. 특히 실용적이고 복잡한 개발 시나리오에서 LLM 환각의 현상, 메커니즘 및 완화 방법에 대한 경험적 연구를 수행했습니다.

- **Technical Details**: LLMs의 환각 현상은 크게 세 가지 주요 범주로 나뉩니다: Task Requirement Conflicts, Factual Knowledge Conflicts, Project Context Conflicts. 이 연구에서는 6개의 주요 LLM(예: ChatGPT, CodeGen)에서 발생하는 환각의 유형과 분포를 분석하고, 환각의 원인으로는 훈련 데이터 품질, 의도 이해 능력, 지식 습득 능력 및 레포지토리 수준의 맥락 인식을 확인했습니다.

- **Performance Highlights**: RAG 기반의 완화 접근법을 제안하며, 이 접근법은 다양한 LLM에서 일관되게 성능을 향상시키는 효과를 보였습니다. 코드 생성 시나리오에 따라 만들어진 검색 라이브러리를 활용하여 각 생성 작업에 유용한 코드 스니펫을 검색합니다.



### Robi Butler: Remote Multimodal Interactions with Household Robot Assistan (https://arxiv.org/abs/2409.20548)
- **What's New**: 이번 논문에서는 원거리 사용자와 다중 모드 상호작용을 가능하게 하는 새로운 가정용 로봇 시스템 'Robi Butler'를 소개합니다. Robi Butler는 사용자에게 로봇 상태를 모니터링하고, 텍스트 또는 음성 지시를 전송하며, 손가락으로 객체를 가리켜 선택할 수 있는 기능을 제공합니다.

- **Technical Details**: Robi Butler는 고급 커뮤니케이션 인터페이스에 기반하여 작동하며, Large Language Models (LLMs)와 Vision Language Models (VLMs)를 활용하여 다중 모드 지시를 해석해 실행 계획을 생성합니다. 이 시스템은 사용자가 Zoom 채팅 웹사이트와 제스처 웹사이트를 통해 지시를 전달할 수 있도록 구성되어 있으며, 실제 가정 환경에서 지시를 수행하는 능력을 갖추고 있습니다.

- **Performance Highlights**: Robi Butler의 효과성과 효율성을 다양한 가정용 작업을 수행하며 입증했으며, 멀티모달 상호작용이 원격 인간-로봇 상호작용 시의 효율성과 사용자 경험에 미치는 영향을 분석하는 사용자 연구도 수행하였습니다.



### Word Sense Disambiguation in Native Spanish: A Comprehensive Lexical Evaluation Resourc (https://arxiv.org/abs/2409.20524)
Comments:
          5 pages, 4 tables

- **What's New**: 이 연구는 스페인어 Word Sense Disambiguation (WSD)을 위한 새로운 자원을 소개합니다. 스페인어의 의미 목록과 Real Academia Española에서 제공한 어휘 데이터를 포함하여 스페인어 WSD의 접근 방식을 조명합니다.

- **Technical Details**: 연구는 BERT와 RoBERTa 모델을 파인튜닝하여 스페인어 WSD를 위한 다양한 자원을 조합하여 사용하고, 기존 자들에 대한 종합적인 검토를 제공하며, 새로운 평가 데이터셋을 공개합니다.

- **Performance Highlights**: 제공된 모델들은 현재 스페인어 WSD 작업에서 대부분의 감독 neural 접근 방식이 달성한 최첨단 결과를 초과하거나 달성하는 성능을 보였습니다.



### Upper and Lower Bounds for Distributionally Robust Off-Dynamics Reinforcement Learning (https://arxiv.org/abs/2409.20521)
Comments:
          48 pages, 3 figures, 2 tables

- **What's New**: 이번 연구에서는 'off-dynamics' 강화 학습에서 정책 훈련 및 배포 환경 간의 차이를 다루고 있으며, 이를 극복하기 위해 분포적으로 강건한 마르코프 결정 과정(distributionally robust Markov decision processes) 프레임워크를 통해 전이 역학의 불확실성에 강한 정책 학습에 초점을 맞추고 있습니다.

- **Technical Details**: 제안된 알고리즘 We-DRIVE-U는 평균 서브옵티멀리티(average suboptimality)가 $	ilde{	ext{O}}(d H rac{	ext{min}iggrac{1}{ho}, H}{	ext{sqrt}(K)})$ 형태를 가지고 있으며, 여기서 $K$는 에피소드 수, $H$는 지평선 길이, $d$는 기능 차원, $ho$는 불확실성 수준을 나타냅니다. 이 연구는 새로운 난이도 있는 사례를 구성하고, 주어진 설정에서 최초의 정보 이론적 하한을 유도하여 알고리즘의 최적성에 대한 통찰을 제공합니다.

- **Performance Highlights**: We-DRIVE-U는 정책 전환 및 이중 최적화 문제 해결을 위한 oracle 호출에서 $	ext{O}(dH	ext{log}(1+H^2K))$의 계산 효율성을 제공하며, 이는 기존 알고리즘에 비해 상당한 개선을 보여줍니다.



### SMLE: Safe Machine Learning via Embedded Overapproximation (https://arxiv.org/abs/2409.20517)
- **What's New**: 본 연구는 규정된 속성을 보장하는 차별화 가능한 기계 학습(ML) 모델을 훈련하는 혁신적인 접근 방식을 제안합니다. 주요 구성 요소로는 효율적인 검증을 위한 일반적이고 간단한 아키텍처, Projected Gradient Method를 기반으로 한 엄격한 훈련 알고리즘, 강력한 반례 검색 문제의 포뮬레이션이 포함됩니다.

- **Technical Details**: 제안된 아키텍처는 Safe ML via Embedded overapproximation (SMLE)로 명명되며, 주 네트워크에 저복잡도의 훈련 가능한 overapproximator를 추가하여 작동합니다. 이 아키텍처는 특정 클래스의 속성 및 설계 선택에 따라 다항 시간 복잡성을 가진 보수적인 검증을 가능하게 합니다.

- **Performance Highlights**: 우리의 접근 방식은 훈련 데이터 및 모델 예측 시 속성 강제화를 포함하는 베이스라인과 경쟁력 있는 성과를 보이며, 선형 부등식 및 다중 클래스 분류에서 상호 배타적인 속성에 대해 평가되었습니다. 정확도는 다소 낮지만, 안전 보장을 유지하면서 추론 과정의 복잡도는 증가하지 않습니다.



### What Information Contributes to Log-based Anomaly Detection? Insights from a Configurable Transformer-Based Approach (https://arxiv.org/abs/2409.20503)
Comments:
          23 pages

- **What's New**: 본 연구에서는 log 데이터에서 의미(the semantic), 순서(the sequential), 시간(the temporal) 정보를 포착할 수 있는 구성 가능(configurable) Transformer 기반(anomaly detection) 이상 탐지 모델을 제안합니다.

- **Technical Details**: 이 모델은 다양한 길이의 log 시퀀스를 사용하여 학습 및 평가가 가능하며, 기존 방법들의 고정 길이(log sequences) 또는 시간 창(time-windowed logs)의 제약을 극복합니다. 여러 입력 특성의 조합을 실험하여 이상 탐지에서 각 정보 유형의 역할을 평가합니다.

- **Performance Highlights**: 모델은 다양한 길이의 log 시퀀스를 처리할 때 안정적이고 경쟁력 있는 성능을 보여주며, 이벤트 발생(event occurrence) 정보가 이상 식별에 중요한 역할을 하는 반면, 순서 정보와 시간 정보는 연구된 공개 데이터셋에서 이상 탐지에 중요한 영향을 미치지 않는 것으로 나타났습니다.



### COLLAGE: Collaborative Human-Agent Interaction Generation using Hierarchical Latent Diffusion and Language Models (https://arxiv.org/abs/2409.20502)
Comments:
          9 pages, 6 figures

- **What's New**: 새로운 프레임워크 COLLAGE는 대규모 언어 모델(LLMs)과 계층적 모션 특정 벡터 양자화 변분 오토인코더(VQ-VAE)를 활용하여 협력적인 에이전트-오브젝트-에이전트 상호작용을 생성합니다. 이 접근법은 풍부한 데이터셋 부족 문제를 해결하고 LLM의 지식 및 추론 능력을 이용하여 생성적인 확산 모델을 안내합니다.

- **Technical Details**: COLLAGE 모델은 다단계의 추상화 수준에서 다양한 모션 특정 특성을 포착하는 계층적 VQ-VAE 구조를 사용합니다. 이 모델은 잠재 공간에서 작동하는 확산 모델과 LLM이 생성한 모션 계획 신호를 통합하여, 노이즈 제거 과정을 안내하고 결과적으로 명령어에 따라 특정 모션 생성을 가능하게 합니다.

- **Performance Highlights**: CORE-4D 및 InterHuman 데이터셋에 대한 실험 결과는 이 접근법이 기존의 최첨단 방법들을 초월하여 현실적이고 다양한 협력적 인간-객체-인간 상호작용을 생성하는 효과를 입증합니다. 이 연구는 로보틱스, 그래픽스 및 컴퓨터 비전과 같은 다양한 분야에서 복잡한 상호작용 모델링의 새로운 가능성을 열어줍니다.



### RecSys Challenge 2024: Balancing Accuracy and Editorial Values in News Recommendations (https://arxiv.org/abs/2409.20483)
Comments:
          5 pages, 3 tables, RecSys' 24

- **What's New**: RecSys Challenge 2024는 뉴스 추천의 기술적 및 규범적 과제를 다루며, 사용자 선호를 행동 기반으로 모델링하고 뉴스 항목의 빠른 소멸을 관리하는 데 중점을 두고 있습니다.

- **Technical Details**: Ekstra Bladet와 JP/Politikens Media Group은 110만 이상의 사용자와 125,000개의 뉴스 기사를 포함하는 대규모 데이터셋을 제공하며, 다양한 메트릭(AUC, MRR, nDCG)을 사용하여 추천 시스템을 평가합니다.

- **Performance Highlights**: 참가자들은 사용자의 클릭 기록, 세션 세부사항, 사용자 메타데이터를 바탕으로 뉴스 기사를 기초로 순위 매기기를 수행하며, 이번 대회는 다양한 추천 시스템의 뉴스 흐름에 대한 영향을 평가하는 데 중점을 두고 있습니다.



### A Weakly Supervised Data Labeling Framework for Machine Lexical Normalization in Vietnamese Social Media (https://arxiv.org/abs/2409.20467)
- **What's New**: 이 연구는 저자원이 많은 언어인 베트남어의 소셜 미디어 텍스트에서 어휘 정규화 문제를 해결하기 위해 혁신적인 자동 레이블링 프레임워크를 소개합니다. 기존의 수작업 레이블링 방식의 비효율성을 극복하고, 준지도학습(semi-supervised learning)과 약한 지도법(weak supervision)을 통합하여 훈련 데이터셋의 품질과 크기를 향상시키는 방법을 제안합니다.

- **Technical Details**: 제안된 프레임워크는 비표준 어휘(non-standard vocabulary)를 표준 형태로 자동으로 변환하여 훈련 데이터의 정확성과 일관성을 높입니다. 실험 결과는 약한 지도법 프레임워크가 Vietnamese 텍스트의 정규화에 효과적임을 보여줍니다. 특히, 사전 훈련된 언어 모델(Pre-trained Language Models)을 활용했을 때 효율성이 극대화됩니다. 이 프레임워크는 F1 점수 82.72%를 달성하고, 어휘 무결성을 99.22%의 정확도로 유지합니다.

- **Performance Highlights**: 이 프레임워크는 다양한 조건에서 비억압 텍스트(undiacritized text)를 효과적으로 처리하며, 자연어 처리(NLP) 작업의 정확성을 통계적으로 1-3% 증가시킵니다. 또한, 증오 표현 탐지, 감정 인식, 스팸 리뷰 탐지 등의 실용적인 NLP 애플리케이션에서 어휘 정규화의 영향을 평가할 수 있는 첫 사례로 주목받고 있습니다.



### POMONAG: Pareto-Optimal Many-Objective Neural Architecture Generator (https://arxiv.org/abs/2409.20447)
- **What's New**: 이번 연구에서는 많은 목적(Many-Objective)를 고려한 Neural Architecture Generator(POMONAG)를 소개합니다. 이는 기존의 DiffusionNAG 방법을 확장하여 정확도뿐만 아니라 모델 복잡성, 계산 효율성 및 추론 지연과 같은 다양한 목표를 동시에 고려합니다.

- **Technical Details**: POMONAG는 성능 예측기(Performance Predictor) 모델을 통합하여 보다 정확한 성능 예측을 통해 생성하는 아키텍처의 품질을 향상시키며, 파레토 최적(Pareto-optimal) 아키텍처 생성을 지원합니다. POMONAG의 메타 데이터셋(Meta-Dataset)은 훈련 여건 개선을 위해 확장되었으며, 다수의 목적을 효과적으로 균형 있게 처리하기 위한 파레토 프론트 필터링(Pareto Front Filtering) 및 스트레칭(Stretching) 기법이 적용되었습니다.

- **Performance Highlights**: POMONAG는 NASBench201 및 MobileNetV3에서 실험을 수행하여 기존 최고의 성능을 초과하는 결과를 보여주었습니다. 특히, 다양한 이미지 분류 데이터셋에서 높은 정확도를 제공하면서도 요구되는 훈련 모델 수를 크게 줄임으로써 효율성을 증명했습니다.



### Sufficient and Necessary Explanations (and What Lies in Between) (https://arxiv.org/abs/2409.20427)
- **What's New**: 이 연구는 머신러닝 모델의 예측을 설명하기 위해 필요성과 충분성의 두 가지 특성 중요성 개념을 정립하고, 이 두 개념을 통합한 새로운 중요성 개념을 제안합니다. 이는 머신러닝의 설명 가능성을 높이기 위한 첫 단계로, 이전 방법들로는 발견하기 어려운 중요한 피쳐를 발견하는 데 기여합니다.

- **Technical Details**: 연구는 머신러닝 예측의 피쳐 중요성을 평가하기 위해 수학적 정의와 접근법을 형식화하며, 조건부 독립성과 게임 이론적 수량(Shapley values)를 기반으로 새로운 통합 프레임워크를 제공합니다. 이를 통해 충분성과 필요성의 관계를 분석하고, 중요한 피쳐를 식별하는 방법을 제안합니다.

- **Performance Highlights**: 실험을 통해 제안된 통합 관점이 피쳐 중요성에 대한 보다 완전하고 중요한 통찰력을 제공함을 입증합니다. 이 새로운 접근법은 이전 방법들의 한계를 극복하고, 일반적인 머신러닝 모델에 대한 이해를 심화시킵니다.



### World to Code: Multi-modal Data Generation via Self-Instructed Compositional Captioning and Filtering (https://arxiv.org/abs/2409.20424)
Comments:
          Accepted at EMNLP 2024 Main Conference, 16pages

- **What's New**: 최근 비전-언어 모델( Vision-Language Models, VLMs)의 발전과 고품질 다중 모달 정렬 데이터의 부족으로 인해 합성 VLM 데이터 생성에 대한 연구가 증가하고 있습니다. 본 논문에서는 Python 코드 형식으로 최종 생성 출력을 구성하는 다중 모달 데이터 구축 파이프라인인 World to Code (W2C)를 소개합니다. W2C는 VLM 스스로를 활용하여 서로 다른 프롬프트를 통해 교차 모달 정보를 추출하고 생성된 출력을 일관성 필터링 전략을 통해 다시 필터링합니다.

- **Technical Details**: W2C 파이프라인은 VLM이 필요로 하는 전문가의 혼합을 줄이고 비싼 인간 피드백 없이 데이터를 생성하고 필터링하는 방법을 제공합니다. 실험 결과, W2C는 여러 기존 비주얼 질문 응답(Visual Question Answering, VQA)과 비주얼 그라운딩(Visual Grounding) 벤치마크에서 다른 VLM들에 비해 높은 품질을 보여주었습니다. W2C는 LLaVA-NeXT-7B 기준으로 9개의 VQA 벤치마크 중 7개에서, LLaVA-NeXT-13B 기준으로 9개 중 6개에서 최적의 성능을 보였습니다.

- **Performance Highlights**: W2C는 GQA와 MME 같은 널리 사용되는 VQA 벤치마크에서 몇 가지 샷 평가(few-shot evaluation)에서도 개선된 성능을 보이며, 특히 GQA의 2-shot 평가에서 모든 VLM들에 대해 5 이상의 정확도 향상을 달성했습니다. 또한 W2C는 기존의 세부 캡션 능력보다 교차 모달 동등성을 더 잘 보여주는 새로운 코드 파싱 능력을 제공합니다.



### Stream-level flow matching from a Bayesian decision theoretic perspectiv (https://arxiv.org/abs/2409.20423)
- **What's New**: 이 논문에서는 Flow Matching (FM) 알고리즘의 새로운 방향을 제시하며, Gaussian Process (GP)를 기반으로 한 조건부 확률 경로를 정의하는 CFM 알고리즘의 확장을 소개합니다. 이를 통해 잠재적 확률 경로와 관련된 여러 데이터를 통합할 수 있는 가능성을 열어줍니다.

- **Technical Details**: 조건부 흐름 일치 (CFM) 훈련을 베이esian 결정 이론적 관점에서 바라보며, 잠재 변수 모델링을 통해 경로를 보강합니다. Gaussian process (GP)를 사용해 이들 잠재적 경로의 모델링을 수행하여, 조건부 확률 경로를 더욱 유연하게 확장합니다.

- **Performance Highlights**: CFM의 일반화가 추정된 분산 벡터 필드의 분산을 크게 줄이면서도 계산 비용은 적당하게 유지되는 것을 보여주며, 생성된 샘플의 품질이 향상됩니다. MNIST 및 HWD+와 같은 필기 이미지 데이터셋에서 실험으로 검증된 결과입니다.



### Conformal Prediction for Dose-Response Models with Continuous Treatments (https://arxiv.org/abs/2409.20412)
Comments:
          10 pages main text, 8 pages references and appendix

- **What's New**: 이번 연구는 continuous treatment에 대한 dose-response 문제를 covariate shift로 바라보아,
weighted conformal prediction을 활용한 새로운 접근 방식을 제안합니다. 이 방법은 개인화된 dose-response 모델에 효과적인 예측 구간을 생성하는 데 중점을 두고 있습니다.

- **Technical Details**: 연구에서는 propensity estimation, conformal predictive systems 및 likelihood ratios를 통합하여 weighted conformal prediction을 통해 예측 구간을 도출하는 방법론을 설명합니다. 또한, kernel functions를 가중치로 적용하여 각 치료 값을 위한 local coverage를 근사화합니다.

- **Performance Highlights**: 새로운 합성 벤치마크 데이터셋을 사용하여 covariate shift 가정이 dose-response 모델을 위한 robust prediction intervals를 생성하는 데 어떻게 중요한 역할을 하는지를 보여줍니다. 이 연구는 유의미한 결정 내리기를 지원하기 위한 UQ(uncertainty quantification)의 중요성을 강조합니다.



### Frequency Adaptive Normalization For Non-stationary Time Series Forecasting (https://arxiv.org/abs/2409.20371)
Comments:
          NeurIPS 2024 Poster

- **What's New**: 이 논문은 시간 시계열 예측에서 비정상 데이터(non-stationary data)의 문제를 해결하기 위한 새로운 방법인 Frequency Adaptive Normalization (FAN)을 제시합니다. 기존의 reversible instance normalization 방식은 기본적인 추세(trend) 표현에 한계가 있었고, 계절 패턴(seasonal patterns)을 처리하는데 부족했습니다. FAN은 푸리에 변환(Fourier transform)을 사용하여 각 인스턴스(instance)별로 주요 빈도(frequency) 구성 요소를 식별하여 비정상성을 처리합니다.

- **Technical Details**: FAN은 비정상성을 다루기 위해 각 입력 인스턴스에서 가장 두드러진 K개의 주파수 성분을 필터링하여 사용합니다. 이 과정에서, 예측(neural network 사용 시 간단한 MLP 모델)하여 미래의 비정상 정보를 예측하고, 이 정보를 사용하여 출력을 재구성합니다. FAN은 여러 예측 모델에 적용될 수 있는 모델 독립적 방법입니다.

- **Performance Highlights**: FAN은 8개의 벤치마크 데이터 세트에서 4개의 일반적인 예측 모델에 적용했으며, 평균 MSE에서 7.76%에서 37.90% 향상된 성능을 나타냈습니다. 또한, FAN은 기존의 최첨단 정규화(normalization) 기법들과 비교하여 탁월한 성능을 보였습니다.



### The Perfect Blend: Redefining RLHF with Mixture of Judges (https://arxiv.org/abs/2409.20370)
Comments:
          submitted to conference

- **What's New**: 이 연구에서는 다중 작업 학습(MTL)에서의 강화를 통해 인공지능 모델의 후처리를 개선하기 위해 제약 생성 정책 최적화(Constrained Generative Policy Optimization, CGPO)라는 혁신적인 패러다임을 소개합니다. CGPO는 비용 효율적인 제약 정책 최적화와 샤프화(stratification)를 통해 RLHF의 최적 혼합을 식별할 수 있습니다.

- **Technical Details**: CGPO의 핵심은 다양한 작업에 대한 맞춤형 최적화 전략을 적용하고, 룰 기반(judge) 및 LLM 기반(judge) 두 가지 종류의 심사를 통해 보상 해킹(reward hacking)을 탐지 및 완화하는 것입니다. 이를 위해 새로운 제약 RLHF 최적화기(Calibrated-Regularized Policy Gradient, CRPG; Constrained Online Direct Preference Optimization, CODPO; Calibrated-Regularized Reward Ranking Finetuning, CRRAFT)가 개발되었습니다.

- **Performance Highlights**: CGPO는 일반 대화, STEM 질문, 지침 준수 및 코딩을 포함한 다양한 작업에서 PPO 및 DPO와 같은 기존의 RLHF 알고리즘을 초월하는 성능을 보여주었습니다. AlpacaEval-2에서 7.4%, Arena-Hard에서 12.5% 개선된 성과를 달성했으며, 전반적으로 모든 벤치마크와 작업에서 동향이 일관성을 보였습니다.



### Rotated Runtime Smooth: Training-Free Activation Smoother for accurate INT4 inferenc (https://arxiv.org/abs/2409.20361)
- **What's New**: 이번 연구에서는 대형 언어 모델의 양자화(quantization)를 위한 새로운 활성화 스무딩 방법인 Rotated Runtime Smooth (RRS)를 제안합니다. 이 방법은 기존의 Outlier 처리 방식의 한계를 극복하기 위해 채널별 아웃라이어와 스파이크 아웃라이어를 구분하고, Runtime Smooth와 회전(rotate) 작업을 결합하여 아웃라이어 문제를 해결합니다.

- **Technical Details**: 제안된 방법은 Runtime Smooth(RS)를 사용하여 실행 중에 채널별 아웃라이어를 제거하고, 회전 작업을 통해 스파이크 아웃라이어를 줄입니다. RS는 채널별 최대값을 이용해 활성화를 부드럽게 만들어 아웃라이어 문제를 완화시키며, 최종적으로 INT4 양자화를 위한 퓨즈된 GEMM 커널(fused GEMM kernel)에 입력으로 제공됩니다. 이를 통해 하드웨어 호환성을 유지하면서도 성능을 개선합니다.

- **Performance Highlights**: LLaMA 가족 및 Qwen 모델에서의 실험 결과, 제안된 방법이 기존 최첨단 기술을 초월하여 WikiText-2의 perplexity를 57.33에서 6.66으로 개선하는 성과를 보여주었습니다. 이는 INT4 추론을 위한 아웃라이어 문제를 효과적으로 해결했음을 나타냅니다.



### Enhancing GANs with Contrastive Learning-Based Multistage Progressive Finetuning SNN and RL-Based External Optimization (https://arxiv.org/abs/2409.20340)
- **What's New**: 본 연구에서 제안한 프레임워크는 멀티스테이지 프로그레시브 파인튜닝 시아미즈 신경망(MFT-SNN)과 강화 학습 기반 외부 최적화기(RL-EO)를 포함하여 GAN 훈련 루프 내에서 지침을 제공함으로써 GAN의 한계를 극복하는 것을 목표로 하고 있습니다.

- **Technical Details**: 이 프레임워크는 두 가지 구성 요소로 이루어져 있습니다. 첫째, MFT-SNN은 조직병리학 패치 간의 유사성을 평가하기 위한 대조 학습 기반의 신경망입니다. 둘째, RL-EO는 GAN 훈련 루프 내에서 보상 신호 생성기로 작용하며, 수정된 판별기 손실 함수는 가중 보상을 포함하여 GAN이 보상을 극대화하면서 손실을 최소화하도록 유도합니다.

- **Performance Highlights**: 제안한 방법은 최신 GAN과 Denoising Diffusion Probabilistic 모델에 대한 벤치마크에서 FID 점수, KID 점수, 지각 경로 길이(Perceptual Path Length), 하류 분류 작업 측면에서 이전의 최첨단(SOTA)을 초월하는 성과를 보여주었습니다.



### A Looming Replication Crisis in Evaluating Behavior in Language Models? Evidence and Solutions (https://arxiv.org/abs/2409.20303)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 행동에 대한 연구가 증가함에 따라 복제 위기의 잠재적 위험성에 대해 논의합니다.

- **Technical Details**: 연구에서는 prompt engineering 기술을 활용하여 LLM의 추론 능력에 미치는 영향을 테스트했으며, 다양한 모델(GPT-3.5, GPT-4o, Gemini 1.5 Pro 등)을 사용했습니다. 실험에서는 CommonsenseQA, CRT, NumGLUE, ScienceQA, StrategyQA와 같은 추론 벤치마크를 포함한 이중 검증된 데이터 세트를 활용했습니다.

- **Performance Highlights**: 모든 테스트된 기술에서 통계적으로 유의미한 차이가 거의 없는 결과를 보였으며, 이는 이전 연구에서의 여러 방법론적 약점을 강조합니다. 따라서 LLM 평가를 위한 강력한 방법론 개발과 철저한 실험 프레임워크 설계의 필요성을 제안합니다.



### Computer-mediated therapies for stroke rehabilitation: a systematic review and meta-Analysis (https://arxiv.org/abs/2409.20260)
Comments:
          32 pages

- **What's New**: 이번 연구는 뇌졸중 환자의 신체 및 심리적 상태 개선에 있어 몰입형 가상현실(IVR)과 비몰입형 가상현실(NIVR)을 기존 요법(CT)과 비교하여 그 효능을 평가했습니다.

- **Technical Details**: 연구는 7개의 데이터베이스(ACM Digital Library, Medline, Cochrane, IEEE Xplore, Web of Science, Scopus)를 통해 문헌 검색이 이루어졌으며, 주요 결과의 효과 크기는 Cohen의 d를 사용해 계산되었습니다. 결과는 무작위 효과 모델(random-effects model)을 사용하여 치료 효과에 대한 전체 추정치를 제공합니다.

- **Performance Highlights**: 22개의 무작위 대조 시험이 평가되었으며, 그 중 3개 연구는 IVR이 기존 요법과 유사하게 상지(upper limb) 활동을 개선하는데 효과적임을 보였습니다. NIVR은 CT와 유사한 상지 활동 및 기능, 균형, 이동성에서의 이점을 제공하는 것으로 나타났습니다. IVR이 일상생활 활동 개선에 있어 NIVR보다 더 유익할 수 있음을 보여주었습니다.



### What is the Role of Large Language Models in the Evolution of Astronomy Research? (https://arxiv.org/abs/2409.20252)
Comments:
          Paper submitted to RASTI. We share our experience, ethical and legal concerns (5.3), and recommendations for individuals and journals (6.). We welcome feedback

- **What's New**: 이 연구는 다양한 경력 단계와 연구 분야에서 13명의 천문학자들의 경험을 바탕으로 대형 언어 모델(LLMs)의 연구 활동 적용 가능성을 탐구하였습니다.

- **Technical Details**: 연구진들은 LLMs를 활용하여 아이디어 생성, 문헌 검토, 코딩, 초안 작성 등 여러 연구 관련 작업을 수행하였습니다. 또한, 익명 설문조사를 통해 LLMs에 대한 참여자들의 경험과 태도를 평가하였습니다. 그들은 LLM의 결과물과 관련된 윤리적 고려사항도 논의하였습니다.

- **Performance Highlights**: LLMs 사용 시 연구 생산성이 20%에서 80%까지 향상될 수 있으며, LLMs는 독창적인 작업에 특히 유리한 성능을 보였지만, 검증된 사실 재생산에서 어려움을 겪는 등 한계도 존재합니다.



### Resource Allocation for Stable LLM Training in Mobile Edge Computing (https://arxiv.org/abs/2409.20247)
Comments:
          This paper appears in the 2024 International Symposium on Theory, Algorithmic Foundations, and Protocol Design for Mobile Networks and Mobile Computing (MobiHoc)

- **What's New**: 이번 논문에서는 모바일 사용자가 엣지 서버와 협업하여 대형 언어 모델(LLM)을 효율적으로 훈련할 수 있는 새로운 협력 훈련 프레임워크를 제안합니다. 이 방법은 PEFT(파라미터 효율적 미세 조정) 기법을 활용하여 모바일 사용자가 LLM의 초기 레이어를 조정하고 엣지 서버가 더 복잡한 후반 레이어를 처리하도록 합니다.

- **Technical Details**: 논문에서는 다중 목표 최적화 문제를 설정하여 훈련 중 에너지 소비와 지연을 최소화합니다. 또한, 모델의 안정성을 높이기 위해 안정성 개선 기법을 목표 함수에 통합합니다. 이 과정에는 새로운 분수 프로그래밍 기법을 사용하여 문제의 정적 점을 도출하고, Concave-Convex Procedure (CCCP)를 통해 사용자-엣지 연결 최적화 문제를 해결합니다.

- **Performance Highlights**: 시뮬레이션 결과, 제안된 방법은 에너지 소비를 줄이고 지연 시간을 감소시키며, 다양한 모바일 환경에서 LLM의 신뢰성을 높입니다.



### Beyond Prompts: Dynamic Conversational Benchmarking of Large Language Models (https://arxiv.org/abs/2409.20222)
Comments:
          Accepted as a poster at NeurIPS D&B Track 2024

- **What's New**: 이 논문에서는 대화형 에이전트의 성능을 평가하기 위한 동적 벤치마크 시스템을 소개하고 있습니다. 이 시스템은 단일, 모의된 사용자와 에이전트 간의 긴 상호작용을 통해 성능을 측정하며, 다수의 작업을 동시에 수행하도록 설계되었습니다.

- **Technical Details**: LTM 벤치마크는 에이전트의 Long-Term Memory (LTM)와 Continual Learning (CL) 능력을 평가합니다. 대화 중에 여러 개의 정보 조각이 서로 결합되어 에이전트의 정보를 효과적으로 사용하는 능력을 평가하게 됩니다. 이 시스템은 LLMs의 자연어 대화 상황에서의 성능을 평가하는 데 초점을 맞추고 있으며, 다양한 작업과 산만함을 포함하여 리얼리틱한 상호작용을 구현하고 있습니다.

- **Performance Highlights**: 대규모 언어 모델(LLMs)은 단일 작업 상호작용에서는 좋은 성능을 보이지만, 작업들이 혼합되었을 때 어려움을 겪습니다. 특히, 짧은 컨텍스트의 LLM이 장기 기억 시스템(LTM)이 보완된 경우에는 더 긴 컨텍스트를 가진 모델들과 유사하거나 더 좋은 성능을 보인다는 것을 발견했습니다.



### Melody Is All You Need For Music Generation (https://arxiv.org/abs/2409.20196)
Comments:
          9 pages, 1 figure, 2 tables

- **What's New**: 이 논문은 Melody Guided Music Generation (MMGen) 모델을 제안하며, 멜로디로 음악 생성을 유도하는 새로운 접근 방식을 제시합니다. 간단한 방법과 제한된 자원에도 불구하고 두드러진 성능을 달성합니다.

- **Technical Details**: MMGen은 멀티모달 정렬 모듈을 사용해 멜로디와 오디오 파형, 그리고 해당 설명을 정렬합니다. 이후, 학습된 멜로디 표현에 조건을 부여하여 diffusion 모듈을 작동시킵니다. 이 과정은 음악 생성 시 멜로디가 의미론적 정보와 오디오 간의 정렬을 증진시킵니다. 또한, intersection rate라는 새로운 메트릭을 제안하여 검색 알고리즘의 성과를 평가합니다.

- **Performance Highlights**: MMGen은 단 5,000개의 음악 샘플로 학습되었음에도 불구하고 100만 이상의 샘플로 학습된 모델과 유사한 성능을 인정받았습니다. 공개된 MusicSet 데이터셋을 통해 다양한 샘플로 더 많은 데이터가 추가될 경우 현재의 오픈 소스 모델을 초월하는 성과를 나타냅니다.



### Forecasting Disease Progression with Parallel Hyperplanes in Longitudinal Retinal OC (https://arxiv.org/abs/2409.20195)
Comments:
          accepted in MICCAI 2024

- **What's New**: 최근의 연구에서, 우리는 망막 OCT 스캔을 통해 늦은 건성 나이 관련 황반변성(dAMD)의 발생 위험을 예측하기 위한 새로운 딥러닝(Deep Learning) 방법을 제안하였습니다. 이 방법은 현재 스캔을 기반으로 전환 시간과 관련된 위험 점수와 특정 시간 내 변환 확률을 예측합니다.

- **Technical Details**: 제안된 방법은 변환 시간(T*)을 랜덤 변수로 모델링하고, 이와 관련된 누적 분포 함수(CDF)를 계산합니다. 또한, 우리는 주어진 이미지 집합에 대해 위험 점수를 할당하여, 다양한 위험 집단으로 분류할 수 있는 시스템을 개발하였습니다. 이 시스템은 각 객체 간 일관성 있는 예측을 보장하는 비지도 학습 손실을 활용합니다.

- **Performance Highlights**: 2개의 대규모 데이터셋을 사용한 평가 결과, Dataset-1에서 평균 AUROC 0.82, Dataset-2에서 평균 AUROC 0.83을 달성하였습니다. 이러한 성능 지표는 다양한 스캐너에서 수집된 이미지 간 도메인 시프트를 극복하는 능력을 보여줍니다.



### Factory Operators' Perspectives on Cognitive Assistants for Knowledge Sharing: Challenges, Risks, and Impact on Work (https://arxiv.org/abs/2409.20192)
Comments:
          32 pages, 6 figures, 2 tables, under review

- **What's New**: 이번 연구는 인공지능 기반의 인지 보조 시스템(Cognitive Assistants, CAs)이 공장에서의 지식 공유에 미치는 실제적인 영향을 조사한 2년 간의 종단적 연구로, 공정 문제 해결과 같은 생산 활동에 필요한 지식 공유를 촉진하기 위해 설계된 CAs의 실사용성을 상세히 규명하고 있습니다.

- **Technical Details**: CAs는 스마트폰 기반의 음성 비서와 대형 언어 모델(LLM)에 기반한 채팅봇을 포함하며, 이들은 공장에서의 작업 흐름과 지식 공유에 대한 사용자의 인식 및 도전 과제를 분석하기 위해 정성적 피드백을 수집했습니다. 연구는 N=40명의 공장 운영자와 관리자에 대한 반구조화된 인터뷰를 통해 진행되었으며, 주제 분석을 통해 결과를 도출하였습니다.

- **Performance Highlights**: CAs는 생산 문제를 더 빨리 해결하고 지식 공유를 개선할 수 있는 잠재력을 보여주었지만, 직장 내 감시, 공유할 수 있는 지식의 종류, 인간 간의 지식 공유 대비 단점 등의 문제를 함께 제기했습니다. 그러므로 CAs를 효과적으로 설계하기 위해서는 프라이버시, 지식 기여 부담, 운영자와 관리자 간의 긴장 관계 등도 함께 고려해야 합니다.



### Choosing DAG Models Using Markov and Minimal Edge Count in the Absence of Ground Truth (https://arxiv.org/abs/2409.20187)
Comments:
          19 pages, 14 figures, 1 table

- **What's New**: 이 논문에서는 데이터 세트에 대한 마르코프 조건(Markov condition)을 검증하기 위해 새로운 비모수(pointwise consistent) 방법인 마르코프 체크(Markov Checker)를 제안합니다. 이를 통해 DAG(Directed Acyclic Graph) 또는 CPDAG(Completed Partially Directed Acyclic Graph) 모델을 효과적으로 평가할 수 있습니다.

- **Technical Details**: 마르코프 체크는 고전적인 통계 모델링에서 요구되는 비모수적 통계 테스트로, 학습된 인과 모델에서 d-분리를 통한 조건부 독립성을 분석합니다. 논문에서는 CAFS(Cross-Algorithm Frugality Search) 알고리즘도 소개하여 마르코프 체크를 통과하지 못한 DAG 모델이나 최소 간선을 가지지 않는 모델을 제거하는 과정을 포함합니다.

- **Performance Highlights**: CAFS 방법을 통한 시뮬레이션 결과는 마르코프 체크의 근거 없이도 약간의 틀림없는 모델을 선택할 수 있음을 보여줍니다. 이 도구는 대규모 또는 밀접한 데이터 모델에서도 유용하며, 실제 데이터 분석에 필요한 새로운 도구를 제공하고 있습니다.



### Modelando procesos cognitivos de la lectura natural con GPT-2 (https://arxiv.org/abs/2409.20174)
Comments:
          in Spanish language

- **What's New**: 이 연구는 GPT-2 기반의 모델을 사용하여 독자의 시선 움직임을 설명하는 Predictability를 모델링하는 데 중점을 두고 있습니다. 이전의 N-grams 및 LSTM 모델보다 성능이 개선되었음을 보여줍니다.

- **Technical Details**: 연구는 36명의 참가자가 8개의 서사 텍스트를 읽는 동안 기록된 시선 추적 데이터(고정 시간)를 사용했습니다. 모델은 클로즈 클리어링 태스크(cloze-task)를 기반으로 Predictability를 계산하며, GPT-2 모델은 11.5GB의 스페인어 텍스트로 훈련되었고, 두 개의 추가 텍스트 코퍼스에 대해 재훈련되었습니다.

- **Performance Highlights**: 선형 혼합 모형을 적용하여 FPRT (First Pass Reading Time) 변수에 대한 결과를 도출했습니다. 연구의 결과는 GPT-2 기반 아키텍처가 이전 모델들에 비해 더욱 향상된 Predictability 모델링 성능을 보여주었습니다.



### 1 Trillion Token (1TT) Platform: A Novel Framework for Efficient Data Sharing and Compensation in Large Language Models (https://arxiv.org/abs/2409.20149)
- **What's New**: 본 논문에서는 1 조 토큰 플랫폼(1TT Platform)을 제안합니다. 이 플랫폼은 투명하고 공정한 수익 분배 메커니즘을 통해 효율적인 데이터 공유를 촉진하도록 설계된 새로운 프레임워크입니다.

- **Technical Details**: 1TT 플랫폼은 데이터 기여자(data contributors)와 데이터 소비자(data consumer) 간의 협업을 촉진합니다. 데이터 기여자는 자신의 데이터를 제공하고, 데이터 소비자는 이 데이터를 활용해 자신의 서비스를 개선하여 수익을 창출합니다. 데이터 기여자는 서비스 수익의 일부를 보상받으며, 이 과정에서 자동화된 데이터 전처리(preprocessing)가 이루어지고, 기여도가 정량화되어 금전적인 보상이 계산됩니다.

- **Performance Highlights**: 1TT 플랫폼은 기여자에게 공정한 보상을 보장하여 고품질의 비공식 데이터를 효율적으로 공유할 수 있는 환경을 마련합니다. 이는 NLP와 LLM 기술의 발전을 촉진하는 협력 생태계를 형성하며, 향후 기여자 평판 시스템 도입 및 맞춤형 데이터 요청 메커니즘 구현 등의 발전 방향이 제안됩니다.



### Classification of Radiological Text in Small and Imbalanced Datasets in a Non-English Languag (https://arxiv.org/abs/2409.20147)
- **What's New**: 이번 연구에서는 덴마크어로 작성된 MRI 보고서를 대상으로 의료 분야에서 자연어 처리 (NLP) 모델의 성능을 평가했습니다. 특히, BERT와 같은 모델이 가장 우수한 결과를 보였으며, SetFit 및 대형 언어 모델(LLM)이 저조한 성적을 기록했습니다. 이는 소규모 데이터세트와 불균형 클래스를 다룰 때 BERT가 최적의 성능을 제공함을 보여줍니다.

- **Technical Details**: 연구에서는 덴마크어로 작성된 16,899개의 MRI 보고서를 사용하였으며, BERT-like transformers, SetFit, 및 LLM 모델을 포함한 다양한 NLP 모델의 성능을 비교했습니다. BERT-like 모델은 해당 도메인에서 사전 학습된 경우 최상의 성능을 나타냈으며, hyperparameter 최적화 과정이 포함되었습니다.

- **Performance Highlights**: BERT-like 모델은 다른 모델들에 비해 우수한 성능을 보였으며, LLM은 성능이 가장 저조했습니다. 그러나 모든 모델이 무감독 텍스트 분류에는 충분한 정확도를 제공하지 않았으나, 데이터 필터링의 잠재력을 보여 주어 수동 라벨링의 필요성을 줄일 수 있는 가능성을 제시합니다.



### Aggressive Post-Training Compression on Extremely Large Language Models (https://arxiv.org/abs/2409.20094)
- **What's New**: 본 논문에서는 대형 언어 모델(Large Language Model, LLM)의 압축을 위해 새로운 네트워크 프루닝 기술을 제안하고 있습니다. 이 기술은 0.7 이상의 희소성(sparsity)과 8비트 이하의 양자화(quantization)를 활용하여, 모델의 크기를 줄이면서도 상대적으로 적은 정확도 손실을 유지할 수 있도록 설계되었습니다.

- **Technical Details**: 제안된 방법은 기존 LLM에 대한 압축을 2시간 이내에 수행할 수 있으며, 기존의 OPTQ(Optimal Brain Compression)와 SparseGPT 알고리즘을 활용하여, 연결된 희소성 분포를 조정하고 기준 오류를 추정하는 레이어별 희소성 스케줄러를 사용합니다. 이로 인해, 약 175억 개의 파라미터를 가진 LLM에서 0.7 이상의 희소성을 달성하고, 모델의 성능을 비슷하게 유지할 수 있습니다.

- **Performance Highlights**: 실험 결과, 우리의 방법은 OPT-66B 및 BLOOM-176B와 같은 모델에서 최첨단 수준의 성능을 지속적으로 초과하였으며, 희소성 분포 조정에 따른 결과도 양호하였습니다. 또한, LLM 양자화 기술을 지원하여 FP16에서 INT4로의 변환 시 추가적인 압축이 가능하다는 점에서 큰 장점을 가지고 있습니다.



### Continuous-Time Linear Positional Embedding for Irregular Time Series Forecasting (https://arxiv.org/abs/2409.20092)
- **What's New**: 이 논문에서는 고불규칙적으로 샘플링된 시계열 데이터( irregularly sampled time series) 예측을 위한 새로운 접근법인 CTLPE(Continuous-Time Linear Positional Embedding)를 제안합니다. 기존 연구에서는 일반적인 시계열 예측에 집중했으나, 본 연구는 불규칙한 시간 간격을 다루기 위해 변환기(transformers) 아키텍처의 위치 임베딩을 수정합니다.

- **Technical Details**: CTLPE는 시계열 데이터의 시간 정보를 효과적으로 표현하기 위해 연속 시간을 기반으로 한 선형 함수(continuous linear function)를 도입합니다. 이 방법은 불규칙한 관측 패턴과 불규칙한 시간 간격을 해결하며, 신경 제어 미분 방정식(neural controlled differential equations) 기반의 위치 임베딩을 통해 선형 임베딩이 다른 연속형 함수에 비해 더 우수함을 입증합니다.

- **Performance Highlights**: CTLPE는 다양한 불규칙 샘플링된 시계열 데이터셋에서 기존 기술들을 초월하는 성능을 보입니다. 이는 CTLPE가 정규 시계열에 대한 transformer 모델의 한계를 극복하고 불규칙 시계열의 시공간적 특성을 효과적으로 캡처할 수 있음을 보여줍니다.



### Knowledge Discovery using Unsupervised Cognition (https://arxiv.org/abs/2409.20064)
- **What's New**: 이 논문에서는 Unsupervised Cognition 모델을 기반으로 한 지식 발견을 위한 세 가지 기법을 제안합니다. 특히 패턴 마이닝(pattern mining), 특성 선택(feature selection), 차원 축소(dimensionality reduction) 기법을 소개하고, 이들을 통해 중요한 패턴을 추출할 수 있는 방법을 제시합니다.

- **Technical Details**: 제안된 패턴 마이닝 기법은 기존의 Unsupervised Cognition 모델에서 훈련된 대표들을 기반으로 하여 데이터의 패턴을 선택하는 방식입니다. 특성 선택 기법은 타겟 특성과의 상관관계를 기반으로 하여 패턴에서 중요 특성을 선정합니다. 차원 축소 기법은 선택된 특성을 기준으로 데이터셋을 줄이는 작업을 수행합니다. 이 과정은 지식 발견 파이프라인을 통해 구현됩니다.

- **Performance Highlights**: 실험 결과, 제안된 기법들은 최신 기술과 비교하여 우수한 성능을 나타냈으며, 최종적으로 Unsupervised Cognition 모델의 정확도를 높일 수 있음을 보여주었습니다. 다양한 분야에서 지식 발견을 통해 의사 결정 및 데이터 분석에 기여할 수 있는 잠재력을 지니고 있습니다.



### Evaluating and explaining training strategies for zero-shot cross-lingual news sentiment analysis (https://arxiv.org/abs/2409.20054)
Comments:
          The first two authors share equal contribution

- **What's New**: 이번 연구에서는 여러 언어에 걸쳐 강력한 감정 분류기를 개발하는 것을 목표로 하는 제로샷(Zero-shot) 교차언어 뉴스 감정 분석을 조사합니다. 새로운 평가 데이터셋을 도입하고, 기계 번역(Machine Translation), 인컨텍스트 학습(In-context learning) 및 심층 학습을 포함한 다양한 접근 방식을 실험하였습니다.

- **Technical Details**: 많은 자원이 부족한 언어에서 제로샷 방식을 통해 감정 분석(Sentiment Analysis, SA)을 수행하는 데 주력했습니다. 연구팀은 mBERT 모델을 활용하면서 다양한 언어 간 감정 전이를 평가하기 위한 새로운 방법인 POA(Part Of Article)를 도입했습니다. 이 방법은 문서 내에서 특정 텍스트의 위치 정보를 사용하여 감정 분석의 효율성을 높이는 방식을 제공합니다.

- **Performance Highlights**: 연구 결과는 기존의 최신 기술(State of the Art)을 초과하는 성능 개선을 보여주었습니다. 인컨텍스트 학습이 일반적으로 가장 뛰어난 성능을 보였으나, 새로 도입된 POA 접근 방식은 낮은 계산 오버헤드에도 불구하고 경쟁력 있는 대안을 제공했습니다. 또한 언어 유사성만으로는 교차언어 전이의 성공을 예측할 수 없으며, 의미적 내용과 구조의 유사성도 중요할 수 있다는 점을 강조했습니다.



### Mitigating Propensity Bias of Large Language Models for Recommender Systems (https://arxiv.org/abs/2409.20052)
- **What's New**: 새로운 프레임워크인 Counterfactual LLM Recommendation (CLLMR)을 소개합니다. 이 프레임워크는 Large Language Models (LLMs)에서 생성된 부가 정보(s)이 사용자와 아이템의 역사적 상호작용 구조 정보를 통합하여 차원 붕괴(dimensional collapse)의 위험을 회피하는 방법을 제안합니다.

- **Technical Details**: Spectrum-based Side information Encoder (SSE)를 통해 역사적 상호작용에서 구조적 정보를 부가 정보의 표현에 암묵적으로 내재화 합니다. 이를 통해 LLM에서 도출된 부가 정보와 사용자 상호작용으로부터 학습한 협력적 표현을 정렬하는 과정에서 발생할 수 있는 차원 붕괴를 방지합니다. 또한, CLLMR 접근법은 LLM 기반 추천 시스템의 인과 관계를 탐색하고, 카운터팩추얼 추론(counterfactual inference)을 활용하여 LLM이 초래하는 편향(bias)을 교정합니다.

- **Performance Highlights**: 실험 결과, CLLMR 접근법은 여러 추천 모델의 성능을 일관되게 향상시켜주었으며, 세 가지 실제 데이터셋에서 최첨단 LLM 추천 정렬 방법들과 비교하여 효과성을 입증했습니다.



### Beyond Scores: A Modular RAG-Based System for Automatic Short Answer Scoring with Feedback (https://arxiv.org/abs/2409.20042)
- **What's New**: 이 논문에서는 응답 평가 및 피드백 생성을 위한 새로운 모듈형 Retrieval-Augmented Generation (RAG) 기반의 자동 단답형 채점 시스템(ASAS-F)을 제안합니다. 이 시스템은 제로샷(zero-shot) 및 몇샷(few-shot) 학습 시나리오에서 작동하도록 설계되어 있으며, 교육 과제에 쉽게 적응할 수 있는 자동 프롬프트 생성 프레임워크를 사용합니다.

- **Technical Details**: 제안한 ASAS-F 시스템은 대형 언어 모델(LLMs)을 활용하여 학생들의 답변을 성공적으로 점수화하고, 유사한 답변을 단답형 점수 피드백 데이터셋에서 검색하여 몇샷(few-shot) 예제 역할을 하도록 구성되었습니다. 이 시스템은 대규모 데이터셋에 대한 의존도를 줄이면서도 높은 정확도를 유지하며 명확하고 정확한 피드백을 제공합니다.

- **Performance Highlights**: 실험 결과, 제안된 시스템은 기존의 미세 조정(fine-tuning) 방법과 비교하여 보지 못한 질문에 대한 채점 정확도가 9% 향상되었으며, 비용 효율적이고 확장 가능한 솔루션을 제공합니다.



### Towards Robust Multimodal Sentiment Analysis with Incomplete Data (https://arxiv.org/abs/2409.20012)
Comments:
          Accepted to NeurIPS 2024

- **What's New**: 이번 연구에서는 Multimodal Sentiment Analysis (MSA) 분야에서 데이터 불완전성 문제를 해결하기 위해 Language-dominated Noise-resistant Learning Network (LNLN)라는 혁신적인 네트워크를 제안합니다. 이 네트워크는 언어 모달리티를 주 모달리티로 설정하고, 다양한 노이즈 상황에서도 모델의 강건성을 강화합니다.

- **Technical Details**: LNLN은 Dominant Modality Correction (DMC) 모듈과 Dominant Modality-based Multimodal Learning (DMML) 모듈을 포함하여, 주 모달리티의 품질을 보장하며, 무작위 데이터 누락 시나리오에서 더 나은 성능을 발휘합니다. 데이터 입력 후, 각 모달리티의 차원을 표준화하는 임베딩 레이어를 통해 시작하며, DMC 모듈은 적대적 학습(adversarial learning)과 동적 가중치 향상 전략을 사용하여 노이즈 영향을 줄이는 방법을 채택합니다.

- **Performance Highlights**: LNLN은 MOSI, MOSEI 및 SIMS와 같은 여러 인기 있는 데이터셋에서 기존 기준선보다 일관되게 우수한 성과를 보였으며, 복잡한 평가 메트릭스에서도 탁월한 성능을 입증하였습니다. 이 연구는 MSA 방법의 강점과 약점을 분석하여, 실제 시나리오에서의 이해를 높이는 데 기여합니다.



### Model Selection with a Shapelet-based Distance Measure for Multi-source Transfer Learning in Time Series Classification (https://arxiv.org/abs/2409.20005)
Comments:
          Accepted at International Conference on Pattern Recognition 2024 (ICPR 2024)

- **What's New**: 이 논문에서는 시계열 분류를 위한 다중 데이터셋을 사용하는 새로운 전이 학습 방법을 제안합니다. 특히, 여러 데이터셋을 하나의 소스 데이터셋으로 결합하여 신경망을 사전 학습(pre-training)합니다.

- **Technical Details**: 제안된 방법은 shapelet discovery를 기반으로 데이터셋의 전이 가능성(transferability)을 측정하여 효과적인 소스 선택을 지원합니다. 기존의 전이 가능성 측정 방법은 여러 아키텍처에 대해 모든 소스에 대해 사전 학습을 수행해야 하며 시간이 많이 소요되는 반면, 우리의 방법은 단일 계산으로 모든 가능한 아키텍처에 활용될 수 있습니다.

- **Performance Highlights**: 제안된 방법을 통해 시계열 데이터셋에서 Temporal Convolutional Neural Networks (CNN)의 성능을 향상할 수 있음을 입증하였습니다. 연구 결과는 2018 UCR Time Series Archive의 128개 시계열 데이터셋에서 평가되었습니다.



### Do Influence Functions Work on Large Language Models? (https://arxiv.org/abs/2409.19998)
Comments:
          18 pages, 8 figures

- **What's New**: 본 연구는 대규모 언어 모델(LLMs)에서의 영향 함수(influence functions)의 효용성을 체계적으로 조사하였습니다. 최근 LLMs의 성장과 함께, 특정 훈련 데이터가 모델 예측에 미치는 영향을 정의하고 정량화하는 것이 점점 더 중요해졌습니다.

- **Technical Details**: 연구에서는 여러 작업에 대해 영향 함수를 평가한 결과, 대부분의 경우 높은 성능을 내지 못함을 발견했습니다. 주요 원인으로는 (1) LLMs의 스케일로 인한 iHVP(implicit Hessian vector product) 요소 추정에서의 불가피한 근사 오차, (2) 세부 조정(fine-tuning) 중의 불확실한 수렴 상태, (3) 모델 파라미터 변화가 LLM 행동 변화와 반드시 일치하지 않는다는 점이 지적되었습니다.

- **Performance Highlights**: 연구 결과, 영향 함수는 대규모 언어 모델에 대하여 일반적으로 성능이 낮고 계산 및 메모리 집약적이라는 한계를 보였습니다. 이전의 영향 함수에 대한 성공 사례는 특수한 사례 연구에 기인하며, 정확한 Hessian 계산 대신 이론적 기반이 부족함을 강조했습니다.



### Mitigating Backdoor Threats to Large Language Models: Advancement and Challenges (https://arxiv.org/abs/2409.19993)
Comments:
          The 60th Annual Allerton Conference (Invited Paper). The arXiv version is a pre-IEEE Press publication version

- **What's New**: 본 논문은 Large Language Models(LLMs)의 백도어 공격(backdoor attack)의 위험성을 종합적으로 조사하고, LLMs에 대한 최근 방어 및 탐지 전략의 발전을 다룹니다.

- **Technical Details**: 백도어 공격은 훈련 데이터의 일부를 조작하여 LLM에 숨겨진 백도어를 주입하고, 사전 정의된 트리거에 의해 활성화되는 악의적인 행동을 유발합니다. 공격자들은 소량의 훈련 사례를 사용하여 모델의 특정 행동과 해당 트리거를 연관 지어 시스템을 해킹하거나 중단시킬 수 있습니다. 이러한 공격은 instruction tuning과 RLHF(강화 학습에서 인간 피드백)의 새로운 학습 패러다임을 통해 더욱 악화됩니다.

- **Performance Highlights**: LLMs에 대한 백도어 공격은 재정적 손실, 시장 혼란, 그리고 신뢰도 손상을 초래할 수 있습니다. 특히, 금융 분야와 헬스케어 같은 고위험 분야에서의 백도어 공격은 큰 피해를 유발할 가능성이 높습니다.



### A large-scale operational study of fingerprint quality and demographics (https://arxiv.org/abs/2409.19992)
Comments:
          Extended journal version submitted to IET Biometrics. 10 pages, 5 figures Reference conference paper: J. Galbally, A. Cepilovs, R. Blanco-Gonzalo, G. Ormiston, O. Miguel-Hurtado, and I. S. Racz, 'Fingerprint quality per individual finger type: A large-scale study on real operational data' in Proc. IEEE Intl. Workshop on Biometrics and Forensics 2023 (IWBF 2023)

- **What's New**: 본 논문은 16,000명의 대규모 데이터베이스를 사용하여 지문 인식 기술의 정확성이 성별, 연령 및 지문 유형 등 특정 인구 통계학적 그룹에 따라 편향되어 있는지를 조사합니다. 이전 연구들보다 더 많은 표본을 통해 지문 품질과 인구 통계학과의 관계를 심층적으로 분석했습니다.

- **Technical Details**: 연구는 500dpi (dots per inch) 터치 기반 광학 스캐너를 이용하여 문서 발급을 위한 비자 처리를 위해 전 세계 34개 비EU 국가에서 수집된 15,942개의 10-프린트 디지털 기록을 분석합니다. 이 데이터베이스는 성별, 연령, 출신 국가와 같은 메타데이터와 함께 지문 샘플을 포함합니다.

- **Performance Highlights**: 지문 인식 품질은 노화, 성별 및 지문 유형에 따라 차이가 있으며, 이는 여러 인구 집단에서 성능 변동을 일으킵니다. 연구 결과, 서로 다른 집단에 대해 지문 인식 기술의 성능 일관성을 향상시키기 위한 개선 방향을 제안하고 있습니다.



### CONTESTS: a Framework for Consistency Testing of Span Probabilities in Language Models (https://arxiv.org/abs/2409.19984)
- **What's New**: 이 연구는 언어 모델의 스코어가 일관성을 유지하는지 평가하기 위한 새로운 프레임워크인 ConTestS(Consistency Testing over Spans)를 소개합니다. 다양한 확장 가능성 및 모델 간의 예측 일관성에 대한 심층 분석을 제공하여 LLMs(Pretrained Large Language Models) 성능을 개선할 수 있는 기회를 모색합니다.

- **Technical Details**: ConTestS는 통계 검정을 활용하여 서로 다른 조건부 확률 조합에서 모델의 일관성을 평가합니다. 실험은 실제 데이터와 합성 데이터를 포함하여 LLMs의 스코어의 일관성을 평가합니다. 연구 결과, Masked Language Models(MLMs)와 자동 회귀 모델이 예측에서 일관성이 부족함을 보여주며, 특히 자동 회귀 모델에서 더 큰 불일치가 발견되었습니다.

- **Performance Highlights**: 연구 결과, 큰 MLM 모델은 더 일관된 예측을 제공하는 반면, 자동 회귀 모델은 그 크기가 증가할수록 예측 간 불일치가 커지는 경향이 있습니다. 두 모델 유형 모두 예측 엔트로피가 실제 단어의 가능성을 나타내며, 이는 최적의 디코딩 전략 선택에 도움이 될 수 있습니다.



### Knowledge Graph Embedding by Normalizing Flows (https://arxiv.org/abs/2409.19977)
- **What's New**: 이 논문은 지식 그래프 임베딩(Knowledge Graph Embedding, KGE)에 대한 새로운 관점을 제시하며, 군론(group theory) 관점에서 불확실성을 도입합니다. 중요한 개념은 엔티티와 관계를 대칭군(symmetric group)의 원소로 임베딩하는 것입니다. 이로 인해 기존 모델들을 포함할 수 있는 일반성(generality), 계산 효율성(efficiency), 복잡한 확률 변수의 표현력(expressiveness)을 보장합니다.

- **Technical Details**: 제안된 모델은 엔티티와 관계를 무작위 변수의 집합의 순열(permutations)로 나타냅니다. 우리는 '정규화 흐름(normalizing flow)'을 사용하여 간단한 무작위 변수를 복잡한 무작위 변수로 변환할 수 있습니다. 또한 두 개의 정규화 흐름의 유사성을 측정하여 점수를 부여하는 함수를 정의했습니다(Normalizing Flows Embedding, NFE).

- **Performance Highlights**: 모델의 실험 결과는 KGE에 불확실성을 도입하는 것이 효과적임을 입증하였으며, NFE는 논리 규칙(logical rules)을 학습할 수 있음을 입증했습니다. 그 과정에서 KGE의 기존 임베딩 모델의 일반화된 형태로 작용하고, 쉽게 계산될 수 있는 그룹 연산의 이점을 활용합니다.



### Attribute-Text Guided Forgetting Compensation for Lifelong Person Re-Identification (https://arxiv.org/abs/2409.19954)
Comments:
          9 pages, 4 figures

- **What's New**: 이 논문은 Lifelong person re-identification (LReID) 문제를 해결하기 위해 새로운 모델인 attribute-text guided forgetting compensation (ATFC)을 제안합니다. ATFC는 작업별 도메인 격차를 해소하고 모델의 성능을 향상시키기 위해 텍스트 기반의 전역 표현과 속성 기반의 지역 정보를 통합하여 파악합니다.

- **Technical Details**: ATFC 모델은 텍스트와 속성을 기반으로 한 작업 공유 표현을 탐구합니다. 주요 구성 요소로는 속성-텍스트 생성기(Attribute-Text Generator, ATG), 텍스트 기반 집계 네트워크(Text-Guided Aggregation Network, TGA), 속성 보상 네트워크(Attribute Compensation Network, ACN) 등이 있습니다. 이들 각 네트워크는 텍스트와 이미지 페어를 동적으로 생성하고, 이를 통해 강력한 전역 및 지역 표현을 생성하여 도메인 격차를 줄이는 역할을 합니다.

- **Performance Highlights**: ATFC 모델은 기존 LReID 방법들보다 평균 mAP(Mean Average Precision) 및 R-1(Top-1 Recall)에서 각각 9.0%와 7.4% 향상된 성능을 보여주었습니다. 이를 통해 ATFC 방법이 다양한 환경에서 사람을 재식별하는데 효과적임을 입증하였습니다.



### Task-agnostic Pre-training and Task-guided Fine-tuning for Versatile Diffusion Planner (https://arxiv.org/abs/2409.19949)
- **What's New**: 이 논문에서는 다양한 작업에 적용할 수 있는 다목적 Diffusion Planner인 	extbf{SODP}를 개발하였습니다. 기존의 다중 작업 계획자나 정책들은 일반적으로 작업별 demonstrational에 의존하거나 각 작업에 대해 보상이 필요했지만, SODP는 비 특정 작업의 저품질 데이터로부터 학습하여 특정 작업에 신속하게 적응하는 능력을 제공합니다.

- **Technical Details**: SODP는 두 단계로 구성된 프레임워크입니다. 첫 번째 단계는 사전 훈련(pre-training)이며, 여기서 다양한 작업의 경로를 모델링하여 기본적인 계획 능력을 추출합니다. 두 번째 단계는 강화 학습(RL)을 기반으로 하는 미세 조정(fine-tuning)으로, 특정 작업에 맞는 보상을 사용하여 Diffusion Planner를 정교화합니다. 이 과정에서 정책 기울기(Policy Gradient)를 적용하여 행동 시퀀스를 최적화합니다.

- **Performance Highlights**: 실험 결과, SODP는 Meta-World 및 Adroit과 같은 다중 작업 도메인에서 최신 방법들보다 월등한 성능을 보였습니다. 특히, 소량의 데이터로도 보상 기반 미세 조정이 가능하여 높은 작업 특정 수익을 달성할 수 있음을 입증하였습니다.



### JaPOC: Japanese Post-OCR Correction Benchmark using Vouchers (https://arxiv.org/abs/2409.19948)
Comments:
          Accepted to PRICAI 2024

- **What's New**: 본 연구는 일본어 영수증에 대한 OCR(Optical Character Recognition) 오류 수정 방법의 벤치마크를 구축하고 효과성을 평가합니다. 이는 기존의 연구에서 다루어지지 않았던 일본어 OCR 오류 수정의 공개 가능 벤치마크를 제공합니다.

- **Technical Details**: 이 연구에서는 일본어 영수증에 특화된 OCR 오류 수정 벤치마크 JaPOC를 제안하고, T5와 같은 언어 모델을 활용하여 오류 수정 방법의 성능을 평가하였습니다. OCR 오류 수정 작업은 시퀀스-투-시퀀스 변환으로 정의되며, OCR 결과에 대해 고급 언어 모델로 미세 조정(fine-tuning)을 진행하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 오류 수정 알고리즘이 전반적인 인식 정확도를 크게 향상시켰습니다. Robota API와 Vision API를 사용하여 구축된 세트에서 상당한 정확도 향상이 확인되었습니다.



### Positive-Sum Fairness: Leveraging Demographic Attributes to Achieve Fair AI Outcomes Without Sacrificing Group Gains (https://arxiv.org/abs/2409.19940)
- **What's New**: 이번 연구에서는 의료 AI의 공정성에 대한 새로운 개념인 positive-sum fairness를 도입하였습니다. 이는 성능의 향상이 집단 간 격차를 확대하더라도, 특정 하위 그룹의 성능 저하가 없다면 수용할 수 있다는 것입니다.

- **Technical Details**: positive-sum fairness는 집단 간 성능 차이를 해로운 것과 무해한 것으로 구분하는 평가 프레임워크입니다. 이 프레임워크를 통해 모델의 성능 향상 과정에서의 공정성을 분석하고, 모든 하위 집단이 더 낫지 않더라도 전체 성능이 향상될 수 있도록 하는 솔루션을 찾고자 합니다.

- **Performance Highlights**: CNN 모델을 비교한 결과, 인구 통계적 인코딩을 제거하면 하위 그룹 간 성능 차이를 줄일 수 있었으며, 인종 속성을 모델 입력으로 활용했을 때 전체 성능은 증가하였지만 하위 그룹 간 격차가 확대됨을 보였습니다. 이는 긍정적 공정성 개념의 관점에서 유익한 성능 개선을 달성할 수 있음을 보여줍니다.



### Scaling Optimal LR Across Token Horizon (https://arxiv.org/abs/2409.19913)
- **What's New**: 이번 연구는 대규모 언어 모델(LLM) 훈련에서 학습률(learning rate)과 토큰 수(token horizon) 간의 연관성에 대한 대규모 실증 연구를 수행하였습니다. 특히, 토큰 수가 많아질수록 최적의 학습률이 감소하는 경향이 있음을 확인하였습니다.

- **Technical Details**: 연구진은 Megatron 코드베이스를 사용하여 여러 LLM 모델의 학습률과 토큰 수 간의 관계를 살펴보았습니다. 실험 결과, 긴 토큰 수의 경우, 작은 학습률이 필요하며, 최적 학습률은 스케일링 법칙(scaling law)을 따릅니다. 이로써 짧은 토큰 수에서 얻은 최적 학습률을 바탕으로 긴 토큰 수에서의 최적 학습률을 추정할 수 있습니다.

- **Performance Highlights**: LLama-1 모델이 너무 높은 학습률을 사용했으며, 이는 성능 저하를 초래한다는 증거를 제공했습니다. 토큰 수에 따른 학습률 전이 방법론을 개발하여 현재의 관행에 부가적인 부담 없이 적용할 수 있도록 하였습니다.



### UniSumEval: Towards Unified, Fine-Grained, Multi-Dimensional Summarization Evaluation for LLMs (https://arxiv.org/abs/2409.19898)
Comments:
          Accepted at EMNLP-Findings 2024

- **What's New**: 이 논문은 UniSumEval 벤치마크를 소개하며, 이는 다양한 입력 맥락(domain, length)을 포괄하고 세밀하고 다차원적인 주석을 제공하는 최초의 종합적인 벤치마크입니다. 또한 AI를 활용한 자료 작성 프로세스를 도입하여 인간 주석자의 어려움을 줄이는 방법을 제시하고 있습니다.

- **Technical Details**: UniSumEval은 아홉 가지 다양한 도메인(news, report, booking 등)을 포함하여, 길이와 대화/non-dialogue 여부에 따른 여러 종류의 텍스트를 수집했습니다. 평가 차원으로는 신뢰성(faithfulness), 정보 누락(completeness), 간결성(conciseness) 등 세 가지 측면이 있으며, AI 지원 수동 평가를 통해 고도의 inter-annotator agreement(IAA)를 달성했습니다.

- **Performance Highlights**: UniSumEval을 통해 아홉 개의 최신 언어 모델에 대한 성능을 평가하였으며, 이들은 비-LLM, 오픈소스 LLM, 상용 LLM으로 분류되었습니다. 평가 차원에 따라 성능 차이가 나타났으며, PII 필터링이 모든 요약 모델의 환각 문제를 악화시킴을 확인하였습니다.



### TRANSAGENT: An LLM-Based Multi-Agent System for Code Translation (https://arxiv.org/abs/2409.19894)
- **What's New**: TRANSAGENT는 LLM 기반의 코드 번역을 개선하기 위해 구문 오류(syntax error)와 의미 오류(semantic error)를 수정하는 멀티 에이전트 시스템을 제안합니다.

- **Technical Details**: TRANSAGENT는 초기 코드 번역기(Initial Code Translator), 구문 오류 수정기(Syntax Error Fixer), 코드 정렬기(Code Aligner), 의미 오류 수정기(Semantic Error Fixer) 등 네 개의 LLM 기반 에이전트가 협력하여 오류를 수정합니다. 이는 소스 프로그램과 타겟 프로그램 간의 실행 정렬(execution alignment)을 기반으로 오류가 있는 코드 블록을 로컬라이즈하여 수정 난이도를 낮춥니다.

- **Performance Highlights**: TRANSAGENT는 새로운 벤치마크에서 번역 효과성과 효율성 모두에서 최신 LLM 기반 코드 번역 기법인 UniTrans를 초월하였습니다. 각 에이전트의 기여도를 분석한 결과, TRANSAGENT의 구문 오류 수정기 및 의미 오류 수정기가 번역 성능을 획기적으로 향상시킨 것으로 나타났습니다.



### RouterDC: Query-Based Router by Dual Contrastive Learning for Assembling Large Language Models (https://arxiv.org/abs/2409.19886)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 본 논문에서는 여러 개의 대형 언어 모델(LLM)을 조합하는 새로운 방법인 query-based Router by Dual Contrastive learning (RouterDC)을 제안합니다. 기존 모델이 여러 LLM이 잘 작동하는 경우 비효율적임을 개선합니다.

- **Technical Details**: RouterDC 모델은 encoder와 LLM 임베딩으로 구성되며, 두 가지 대조 학습 손실(contrastive learning losses)을 사용하여 모델을 훈련합니다. 이는 다양한 쿼리에 대해 가장 적합한 LLM을 선택하는 라우팅(routing) 기술을 활용합니다.

- **Performance Highlights**: 실험 결과, RouterDC는 개별 최고 성능 LLM과 기존 라우팅 방법보다 각각 +2.76% 및 +1.90% 더 우수한 성능을 보여주었습니다. 최적의 LLM 조합을 통해 효과적으로 성능을 향상시킴을 입증합니다.



### SWIM: Short-Window CNN Integrated with Mamba for EEG-Based Auditory Spatial Attention Decoding (https://arxiv.org/abs/2409.19884)
Comments:
          accepted by SLT 2024

- **What's New**: 이 연구에서는 특정 화자에 집중하면서 다른 화자를 무시하는 복잡한 청각 환경에서의 인간 청각 시스템의 능력을 활용하기 위해 새로운 모델 SWIM을 제안합니다.

- **Technical Details**: SWIM은 short-window convolution neural network (CNN)과 Mamba를 통합한 모델입니다. SW$_{CNN}$은 EEG 신호에서 단기 기능을 추출하여 최종 정확도 84.9%를 달성하며, Mamba는 청각 공간 주의(Spatial Attention) 디코딩을 위한 순서 모델로서 SW$_{CNN}$의 이전 시간 단계에서의 장기 의존성을 활용합니다.

- **Performance Highlights**: SWIM 구조는 단기 및 장기 정보를 모두 사용하여 86.2%의 정확도를 달성하며, 이는 이전의 최신 기술 대비 31.0%의 분류 오류 감소를 나타냅니다.



### Contrastive Token Learning with Similarity Decay for Repetition Suppression in Machine Translation (https://arxiv.org/abs/2409.19877)
Comments:
          Accepted by EMNLP'24 Findings. 12 pages, 4 figures, 9 tables

- **What's New**: 본 논문은 Neural Machine Translation(NMT)의 생성 콘텐츠에서의 단조로움과 반복 문제를 해결하기 위한 새로운 접근 방식을 제시합니다. 정보 엔트로피를 활용해 텍스트 반복의 원인을 분석하며, Contrastive Token Learning with Similarity Decay(CTSD)라는 새로운 알고리즘을 소개합니다.

- **Technical Details**: CTSD는 다이나믹하게 토큰 억제를 조절하며, 변동하는 attention weights와 inter-token distances에 기반합니다. 또한, 온라인 현실 아이템의 제목 텍스트로 구성된 e-commerce 데이터셋을 구축하여 알고리즘의 성능을 평가합니다.

- **Performance Highlights**: CTSD는 기존 접근법보다 정확성과 일반화 능력에서 현저히 우수한 결과를 보이며, 온라인 A/B 테스트를 통해 사용자 참여도와 전환율이 크게 향상되었음을 입증합니다. 이 방법은 세계 최대 B2B e-commerce 플랫폼의 8개 다국어 사이트에서도 성공적으로 구현되었습니다.



### TSI: A Multi-View Representation Learning Approach for Time Series Forecasting (https://arxiv.org/abs/2409.19871)
Comments:
          AJCAI Oral Accepted

- **What's New**: 본 논문에서는 전통적인 시간 시계열 예측 모델의 한계를 극복하기 위해 트렌드(Trend)와 계절성(Seasonality) 표현을 독립 성분 분석(Independent Component Analysis, ICA)을 기반으로 통합한 새로운 멀티 뷰 접근 방식을 제안합니다.

- **Technical Details**: TSI 모델은 트렌드 및 계절성의 관점과 ICA의 관점을 결합하여 복잡하고 고차원적인 시계열 데이터를 분석합니다. 기존 방법들이 놓치는 비선형 관계를 포착할 수 있도록 설계되었습니다.

- **Performance Highlights**: 다양한 기준 데이터셋에서 TSI 모델은 현재 최첨단 모델들에 비해 뛰어난 성능을 보여주며, 특히 다변량(multi-variate) 예측에서 높은 정확도를 제공합니다. 이 방법은 시간 시계열 데이터에 대한 더 깊이 있는 이해를 제공하여 예측의 정확성을 향상시킵니다.



### Counter-Current Learning: A Biologically Plausible Dual Network Approach for Deep Learning (https://arxiv.org/abs/2409.19841)
Comments:
          NeurIPS 2024

- **What's New**: 이번 논문에서는 생물학적 신경망의 학습 메커니즘을 모방한 새로운 학습 알고리즘인 counter-current learning (CCL)을 제안합니다. 이 알고리즘은 신경망에서의 신용 할당을 위한 생물학적으로 그럴듯한 프레임워크로, 피드포워드 네트워크와 피드백 네트워크를 활용하여 상호 작용하는 방식으로 작동합니다.

- **Technical Details**: CCL은 입력 데이터를 처리하는 피드포워드 네트워크와 목표를 처리하는 피드백 네트워크를 결합합니다. 두 네트워크는 안티-패럴렐 신호 전파(anti-parallel signal propagation)를 통해 서로를 강화하며, 피드백 네트워크의 하위 계층에서 더 많은 정보를 얻어 이를 피드포워드 네트워크의 상위 계층 업데이트에 활용합니다.

- **Performance Highlights**: MNIST, FashionMNIST, CIFAR10 및 CIFAR100 데이터셋에서 실행된 실험 결과, CCL은 다른 생물학적으로 그럴듯한 알고리즘들과 유사한 성능을 유지하며, 더 생물학적으로 현실적인 학습 메커니즘을 제공합니다. 또한, 오토인코더(encoder) 작업에 대한 적용 가능성을 보여줍니다.



### ForecastBench: A Dynamic Benchmark of AI Forecasting Capabilities (https://arxiv.org/abs/2409.19839)
- **What's New**: 새로운 연구는 ForecastBench라는 동적 benchmark를 도입하여 머신러닝 (ML) 시스템의 예측 정확성을 평가할 수 있는 표준화를 제공한다.

- **Technical Details**: ForecastBench는 자동으로 생성되고 정기적으로 업데이트되는 1,000개의 예측 질문 세트로 구성되어 있으며, 데이터 유출 가능성을 방지하기 위해 제출 시점에 알려지지 않은 미래 사건에 대한 질문만 포함한다. 현재 ML 시스템의 능력을 평가하기 위해, 전문가(인간) 예측자, 일반 대중, LLMs로부터 benchmark에서 무작위 선택된 질문(N = 200)에 대한 예측을 수집하였다.

- **Performance Highlights**: 전문가 예측자는 상위 성능의 LLM보다 더 높은 예측 정확성을 보였으며(p-values <= 0.01), 이에 대한 결과는 공개 리더보드에서 확인할 수 있다.



### Generalizability of Graph Neural Networks for Decentralized Unlabeled Motion Planning (https://arxiv.org/abs/2409.19829)
Comments:
          6 pages, 6 figures, submitted to ICRA 2025

- **What's New**: 이 논문에서는 로봇들이 충돌을 피하면서 목표 위치에 도달하는 자율적인 이동 계획 문제를 다룹니다. 특히, 각 로봇이 자신의 $k$-근접 이웃 로봇 및 목표만 인지하는 분산( decentralized) 환경에서 진행되는 접근 방식을 제안합니다.

- **Technical Details**: 본 연구는 그래프 신경망(Graph Neural Network, GNN)을 활용하여 로봇들이 이웃에게 어떤 정보를 전달하고, 수신한 정보를 어떻게 취합하여 의사 결정을 내리는지를 학습합니다. GNN은 중앙 집중식 헝가리 알고리즘(Hungarian algorithm)을 전문 정책으로 사용하는 모방 학습을 통해 훈련되고, 충돌 회피와 성능 향상을 위해 강화 학습으로 추가 조정됩니다.

- **Performance Highlights**: 100대 로봇에서 훈련된 GNN 정책은 최대 500대 로봇의 시나리오에서도 일반화되며, 기존 최첨단 솔루션 대비 평균 8.6% 높은 성능을 달성했습니다. 특히 탐욕적( greedy) 분산 접근 방법보다 성능이 탁월한 것으로 나타났습니다.



### Counterfactual Evaluation of Ads Ranking Models through Domain Adaptation (https://arxiv.org/abs/2409.19824)
Comments:
          Accepted at the CONSEQUENCES'24 workshop, co-located with ACM RecSys'24

- **What's New**: 이번 논문에서는 Offline A/B testing 시스템과 함께 작동하는 도메인 적응형 보상 모델(domain-adapted reward model)을 제안합니다.

- **Technical Details**: 이 보상 모델은 대규모 Ads 추천 시스템에서 랭킹 모델(ranking model) 변경에 대한 보상을 효과적으로 측정합니다. 기존의 모델 없는 방법인 IPS가 적용될 수 없는 환경에서도 사용이 가능합니다.

- **Performance Highlights**: 실험 결과, 제안된 기술이 기존의 vanilla IPS 방법 및 비일반화 보상 모델(non-generalized reward models) 접근 방식을 모두 초과하는 성능을 보였습니다.



### OrganiQ: Mitigating Classical Resource Bottlenecks of Quantum Generative Adversarial Networks on NISQ-Era Machines (https://arxiv.org/abs/2409.19823)
- **What's New**: 이 논문에서는 OrganiQ라는 새로운 양자 생성적 적대 신경망(Quantum GAN)을 소개합니다. OrganiQ는 고전 신경망(classical neural networks)을 사용하지 않고도 고품질 이미지를 생성할 수 있는 최초의 양자 GAN입니다.

- **Technical Details**: OrganiQ는 양자 기계 학습(quantum machine learning) 기술을 활용하여 이미지 생성의 양자 잠재력을 극대화하고, 이미지 품질을 향상시킵니다. 기존의 양자 이미지 생성 기법은 고전 신경망에 의존하여 제한적인 결과를 초래했습니다.

- **Performance Highlights**: OrganiQ는 고해상도의 재현이 가능하며, 기존의 양자 이미지 생성보다 개선된 품질을 보여줍니다. 또한, 양자 하드웨어의 성능을 극대화하여 기존 알고리즘과 비교했을 때 훨씬 더 효율적인 결과를 제공합니다.



### Qompose: A Technique to Select Optimal Algorithm- Specific Layout for Neutral Atom Quantum Architectures (https://arxiv.org/abs/2409.19820)
- **What's New**: 이 논문에서는 중성 원자(Neutral Atom)를 활용한 새로운 양자 컴퓨팅 프레임워크인 Qompose를 제안합니다. Qompose는 2-D 토폴로지(Topology)에서 양자 회로를 효율적으로 구성할 수 있도록 설계되었습니다.

- **Technical Details**: Qompose는 주어진 양자 회로에 대해 최적의 토폴로지를 선택하여 실행 길이를 최적화하고 전반적인 충실도(Fidelity)를 높이는 구조입니다. 이 프레임워크는 효율적인 병렬성(Parallelism)을 통해 성능을 극대화합니다.

- **Performance Highlights**: 광범위한 평가를 통해 Qompose는 무작위로 생성된 양자 회로의 다양한 조합과 VQE, ISING, QAOA와 같은 실제 벤치마크(benchmark)에서도 효과적인 성능을 나타냅니다.



### Calibrating Language Models with Adaptive Temperature Scaling (https://arxiv.org/abs/2409.19817)
Comments:
          EMNLP 2024

- **What's New**: 본 논문에서는 Adaptive Temperature Scaling (ATS)이라는 새로운 후처리 방법을 소개합니다. 이는 각 토큰 예측에 대해 온도 스케일링 매개변수를 예측하여 모델의 신뢰도를 개선합니다.

- **Technical Details**: ATS는 토큰 수준의 특성에 따라 조정되는 온도 값을 예측하며, 이는 표준 감독된 미세 조정(supervised fine-tuning) 데이터셋에 맞춰진 것입니다. 이 방법은 강화 학습(reinforcement learning)에서 인간 피드백(human feedback)을 사용하는 후 미세 조정 이후에 발생하는 보정(calibration) 변화에 적응합니다.

- **Performance Highlights**: ATS는 세 가지 다운스트림 자연어 평가 벤치마크에서 이전 보정 방법에 비해 10-50% 이상의 보정 개선을 달성하였으며, RLHF로 인한 성능 향상에는 방해가 되지 않습니다.



### Grounded Curriculum Learning (https://arxiv.org/abs/2409.19816)
Comments:
          8 pages, 4 figures

- **What's New**: 이번 연구에서는 로봇의 강화 학습(Reinforcement Learning, RL)에 있어 시뮬레이터에서의 작업(task) 분포와 실제 작업 분포 간의 불일치를 해결하기 위한 새로운 지침을 제안합니다. 이를 통해 Grounded Curriculum Learning (GCL)이라는 접근 방식을 통해 로봇의 학습 효율성을 향상시키고자 합니다.

- **Technical Details**: GCL은 시뮬레이터 내에서의 작업 분포를 실제 세계의 작업과 일치시키며, 로봇에게 주어진 과거 작업(task)과 로봇의 성능도 고려합니다. 이를 통해 기존의 커리큘럼 학습(curriculum learning) 방식의 한계를 극복하고, BARN 데이터셋을 사용하여 복잡한 내비게이션 작업에서 성과를 검증하였습니다.

- **Performance Highlights**: GCL을 적용한 결과, 최첨단 커리큘럼 학습 방법(state-of-the-art CL method) 및 전문가에 의해 설계된 커리큘럼에 비해 각각 6.8% 및 6.5% 높은 성공률을 달성하였으며, 이는 시뮬레이션 작업 분포를 실제 작업에 맞게 조정했을 때 학습 효율 및 내비게이션 성능이 향상됨을 보여줍니다.



### Can Models Learn Skill Composition from Examples? (https://arxiv.org/abs/2409.19808)
Comments:
          Accepted to NeurIPS 2024

- **What's New**: 이번 연구는 compositional generalization (구성적 일반화) 능력에 대한 평가에서 SKILL-MIX를 활용하여 smaller models (소형 모델)의 성능을 측정한 것입니다. 이 연구는 AI 안전성 및 정렬 연구와도 관련이 있습니다.

- **Technical Details**: 연구에서는 다양한 언어 기술을 포함한 텍스트 샘플을 생성하기 위해 GPT-4를 사용했습니다. 이를 통해 7B 및 13B 파라미터 모델을 fine-tuning (미세 조정) 하여 k-기술 조합의 텍스트 작성을 평가했습니다. 사용된 기술 세트는 수사학적 (rhetorical), 문학적 (literary), 추론 (reasoning), 마음 이론 (theory of mind), 상식 (common sense) 등을 포함합니다.

- **Performance Highlights**: (1) k=2 및 k=3 기술의 조합으로 학습한 모델은 k=4 및 k=5 기술을 활용한 텍스트 작성을 개선하였습니다. (2) 기술 범주를 나누어 학습 및 검증 그룹으로 분리했을 때, 모델은 훈련 중 본 적이 없는 기술로 텍스트를 작성하는 데에서 크게 향상되었습니다. 이 연구는 skill-rich (기술이 풍부한) 텍스트를 훈련에 통합하는 것이 모델의 구성 능력을 크게 향상시킬 수 있음을 제안합니다.



### PALM: Few-Shot Prompt Learning for Audio Language Models (https://arxiv.org/abs/2409.19806)
Comments:
          EMNLP 2024 (Main)

- **What's New**: 본 논문에서는 Audio-Language Models (ALMs)에 대한 새로운 접근법인 Prompt Learning in Audio Language Models (PALM)을 제안합니다. 이는 텍스트 인코더 브랜치의 특징 공간(feature space)을 최적화하여 오디오 인식 성능을 향상시킵니다.

- **Technical Details**: PALM은 기존 방법들이 입력 공간(input space)에서 작동하는 것과 달리, 효율적인 훈련 효율성을 제공합니다. 이 접근법은 11개의 오디오 인식 데이터셋을 사용하는 다양한 음성 처리(task) 작업에서 검증되었습니다.

- **Performance Highlights**: PALM 방법은 다른 기존 방법들과 비슷하거나 그 이상의 성능을 보였으며, 계산적으로 덜 부담스럽습니다. 또한, 몇 가지 샷(few-shot) 학습 설정에서도 비교되었습니다.



### CRScore: Grounding Automated Evaluation of Code Review Comments in Code Claims and Smells (https://arxiv.org/abs/2409.19801)
- **What's New**: 자동화된 코드 리뷰(automated code review)가 기계 학습(machine learning) 커뮤니티에서 최근 많은 주목을 받고 있습니다. 기존의 리뷰 코멘트 평가 지표는 코드 변경(diff)에 대해 인간이 작성한 참조(reference)와 비교하는 방식을 기반으로 하고 있지만, 코드 리뷰는 여러 개의 '유효한 리뷰(valid reviews)'가 존재하는 다대일(one-to-many) 문제입니다. 이를 해결하기 위해 새로운 평가 기준인 CRScore를 개발했습니다.

- **Technical Details**: CRScore는 코드의 결함(claims)과 잠재적 문제를 탐지하는 LLMs(대형 언어 모델) 및 정적 분석기(static analyzers)에 기반하여 리뷰 품질(review quality)의 차원들을 측정하는 참조가 필요 없는(reference-free) 지표입니다. 평가 항목으로는 간결성(conciseness), 포괄성(comprehensiveness), 관련성(relevance) 등이 포함됩니다. CRScore는 인간의 판단(human judgment)과 높은 상관관계(0.54 Spearman correlation)를 보이며, 기존 지표들보다 민감도가 높습니다.

- **Performance Highlights**: CRScore는 자동화된 메트릭(metric) 개발을 지원할 수 있도록 2.6k의 인간 주석(human-annotated) 리뷰 품질 점수를 포함한 코퍼스(corpus)를 공개하였습니다.



### Adaptive Event-triggered Reinforcement Learning Control for Complex Nonlinear Systems (https://arxiv.org/abs/2409.19769)
- **What's New**: 이 논문에서는 경계가 있는 불확실성(bounded uncertainties)과 복잡한 상호작용을 특성으로 하는 연속 시간 비선형 시스템에 대한 적응형 이벤트 기반 강화 학습 제어 방안을 제안합니다.

- **Technical Details**: 제안된 방법은 제어 정책(control policy)과 통신 정책(communication policy)을 동시에 학습할 수 있는 능력을 갖추고 있어, 이를 각각 또는 단독으로 학습할 때의 파라미터 수와 계산 오버헤드를 줄입니다. 또한, 전체 경로에 걸친 성능을 나타내는 보상(rewards)을 상태 공간에 추가하여 명시적인 트리거 조건(triggering conditions) 학습 필요 없이 정확하고 효율적으로 트리거 조건을 결정할 수 있음을 보여줍니다.

- **Performance Highlights**: 수치 예제를 통해 제안된 접근 방식의 효과성을 입증하였습니다.



### Towards Robust Extractive Question Answering Models: Rethinking the Training Methodology (https://arxiv.org/abs/2409.19766)
Comments:
          EMNLP 2024 Findings

- **What's New**: 이 논문은 Extractive Question Answering (EQA) 모델의 강건성을 향상시키기 위한 새로운 훈련 방법을 제안합니다. 특히, 기존 EQA 데이터셋에 포함된 답변할 수 없는 질문들이 모델의 강건성 부족을 초래한다는 것을 지적합니다.

- **Technical Details**: 제안된 훈련 방법은 EQA 문제를 위한 새로운 손실 함수(loss function)를 포함하며, 많은 EQA 데이터셋에 존재하는 암묵적인 가정을 도전합니다. 이 방식을 통해 훈련된 모델은 도메인 내에서 성능을 유지하면서도 도메인 외 데이터셋에서 유의미한 향상을 이룹니다.

- **Performance Highlights**: 모든 테스트 세트에서 F1 점수가 5.7만큼 향상되었으며, 두 가지 유형의 적대적 공격(adversarial attacks)에 대해 크게 향상된 강건성을 보여줍니다. 기본 모델과 비교했을 때 성능 저하는 약 1/3에 불과합니다.



### Balancing the Scales: A Comprehensive Study on Tackling Class Imbalance in Binary Classification (https://arxiv.org/abs/2409.19751)
Comments:
          13 pages including appendix, 4 tables

- **What's New**: 이 연구는 클래스 불균형 문제를 다루기 위한 세 가지 주요 전략(Synthetic Minority Over-sampling Technique (SMOTE), Class Weights 조정, Decision Threshold Calibration)의 효과를 포괄적으로 평가하였습니다. 15개의 머신 러닝 모델과 30개의 데이터셋을 이용해 9,000개의 실험을 수행한 결과, 모든 전략이 기본 모델보다 우수한 성과를 보였고, Decision Threshold Calibration이 가장 일관되게 효과적인 기법으로 나타났습니다.

- **Technical Details**: 이 연구는 F1-score를 주요 성능 지표로 삼고, F2-score, precision, recall, Brier-score, PR-AUC, AUC 등 9개의 추가 성능 지표를 추적했습니다. 30개 데이터셋(표본 크기 500~20,000, 희귀 클래스 비율 1~15%)과 15개 분류기 모델을 대상으로 5-fold cross-validation을 이용해 평가를 수행했습니다.

- **Performance Highlights**: 이 연구 결과는 클래스 불균형 데이터셋을 처리하는 방법에 따라 성능이 크게 달라질 수 있음을 보여주었습니다. 연구자들은 각 기술이 데이터셋에 따라 어떻게 작동하는지를 강조하며, 다양한 접근 방식을 시험하는 것의 중요성을 강조했습니다.



### PEAR: Position-Embedding-Agnostic Attention Re-weighting Enhances Retrieval-Augmented Generation with Zero Inference Overhead (https://arxiv.org/abs/2409.19745)
Comments:
          preprint

- **What's New**: 본 논문에서는 Position-Embedding-Agnostic attention Re-weighting (PEAR)을 제안하여, LLMs의 context awareness를 향상시킵니다. PEAR은 inference 오버헤드 없이 LLM의 성능을 개선하는 새로운 방법을 제시합니다.

- **Technical Details**: PEAR은 RAG 과제에서 context awareness를 강화하기 위해, suppressing heads를 탐지하고 이들의 출력을 learnable coefficients를 사용하여 재가중화합니다. 모델 파라미터를 동결한 채로 이러한 coefficients를 최적화하여 RAG 성능을 제고하는 방식입니다.

- **Performance Highlights**: PEAR은 메모리 사용량이나 inference 시간에서 0의 추가 오버헤드를 제공하며, 다양한 RAG 과제에서 경쟁 기반보다 높은 정확도와 효율성을 보여줍니다. 또한 PEAR은 특정 position embedding 알고리즘에 의존하지 않아 광범위한 적용 가능성을 가지고 있습니다.



### Unified Gradient-Based Machine Unlearning with Remain Geometry Enhancemen (https://arxiv.org/abs/2409.19732)
Comments:
          Accepted by NeurIPS 2024 as a Spotlight paper

- **What's New**: 본 논문에서는 머신 언러닝(Machine Unlearning, MU)의 새로운 접근 방식을 제안하여 기계 학습 모델의 개인 정보 보호 및 신뢰성을 강화하는 방법을 연구하였습니다. 특히 대규모 모델을 위한 근사 MU(Approximate MU) 기법을 집중적으로 조명하고 있습니다.

- **Technical Details**: 본 연구는 매개변수 이웃 내에서의 정확한 MU와의 Kullback-Leibler divergence 최소화를 통해 가장 가파른 하강 방향을 찾는 방법으로 시작됩니다. 이 방향은 가중치 잊기 그래디언트 상승(weighted forgetting gradient ascent), 나머지 세트를 유지하기 위한 미세 조정 그래디언트 하강(fine-tuning retaining gradient descent), 그리고 가중치 비중 행렬(weight saliency matrix)으로 분해됩니다. 이 구성은 기존의 그래디언트 기반 MU 방법들을 통합하는 관점을 제공합니다. 또한, 남은 데이터의 두 번째 도함수(Hessian)를 포함하여, 더 효율적인 잊기 방향을 학습하는 빠른-느린 매개변수 업데이트(fast-slow parameter update) 전략을 제안합니다.

- **Performance Highlights**: Extensive experiments demonstrate that our method achieves class-forgetting on ImageNet using DiT and effectively forgets a class on CIFAR-10 using DDPM in only 50 steps, markedly outperforming prior methods that required thousands of steps.



### Constrained Reinforcement Learning for Safe Heat Pump Contro (https://arxiv.org/abs/2409.19716)
- **What's New**: 이 논문은 에너지 효율성과 거주자의 열 쾌적성을 동시에 최적화하기 위한 새로운 건물 시뮬레이터 I4B를 제안합니다. 이 시뮬레이터는 다양한 용도로 사용할 수 있는 인터페이스를 제공하며, 제약이 있는 모델 프리 강화 학습 알고리즘인 CSAC-LB를 활용하여 난방 최적화 문제를 해결합니다.

- **Technical Details**: I4B는 건물 시뮬레이션 모듈과 제어 알고리즘 간의 인터페이스를 생성하며, 참조 제어기(reference controllers)와 병렬 처리(parallelization) 지원, 알고리즘 평가를 위한 표준화된 메트릭(metrics)을 포함합니다. 제약 마코프 결정 과정(Constrained Markov Decision Process, CMDP)으로 난방 제어 문제를 개념화하고, CSAC-LB 알고리즘을 사용하여 다양한 시나리오에서 성능을 평가합니다.

- **Performance Highlights**: CSAC-LB는 데이터 탐색(data exploration)과 제약 조건 만족(constraint satisfaction) 측면에서 우수한 성능을 보이며, 다른 최신 알고리즘(SOTA)과 비교하여 목표와 제약의 균형을 잘 맞출 수 있음을 보여줍니다.



### InfantCryNet: A Data-driven Framework for Intelligent Analysis of Infant Cries (https://arxiv.org/abs/2409.19689)
- **What's New**: 이번 논문에서는 infant cries(유아의 울음소리)를 이해하기 위한 새로운 데이터 기반 프레임워크, 'InfantCryNet'을 제안합니다. 이 프레임워크는 울음 소리의 탐지 및 분석을 동시에 수행하며, 데이터 부족 문제를 해결하기 위해 사전 훈련된 오디오 모델을 사용합니다.

- **Technical Details**: 모델은 통계적 풀링(statistical pooling)과 다중 헤드 주의(pooling with multi-head attention) 기법을 사용하여 더 효과적으로 특징을 추출하며, 모델의 효율성을 높이기 위해 knowledge distillation(지식 증류) 및 모델 양자화(model quantization) 기법을 적용했습니다.

- **Performance Highlights**: 실제 데이터셋을 통한 실험 결과, 제안된 프레임워크는 분류 정확도에서 기존의 최첨단 모델 대비 4.4% 높은 성능을 나타내었고, 모델 압축 기술을 통해 모델 크기를 7% 줄일 수 있었습니다. 성능 손실 없이 최대 28%까지 모델 크기를 줄일 수 있는 가능성을 보여주었습니다.



### Machine Learning for Raman Spectroscopy-based Cyber-Marine Fish Biochemical Composition Analysis (https://arxiv.org/abs/2409.19688)
- **What's New**: 이 논문은 래만 분광법(Raman spectroscopy)을 활용하여 물고기의 생화학 조성을 비파괴적으로 분석하고, 물, 단백질 및 지방 산출량을 공동으로 예측하기 위해 새로운 CNN(Convolutional Neural Networks) 모델을 제안합니다. 또한, 극소량의 데이터셋으로 CNN을 적용한 최초의 연구로, 데이터 부족의 문제를 해결하기 위한 프레임워크인 FishCNN을 개발하였습니다.

- **Technical Details**: FishCNN 프레임워크는 데이터 전처리, 데이터 증강(data augmentation) 및 스케일링 기법을 통합하여 작은 실세계 분광 데이터셋에 적용합니다. 이 과정에서 FT-Raman 및 InGaAs 1064 nm 스펙트로스코픽 데이터를 교차 검증을 위해 6 개의 폴드로 나누고, 다양한 전처리 기법을 적용하여 데이터 특성을 정제합니다. CNN 모델은 큰 필터 크기와 작은 스트라이드를 사용하며, 데이터 전처리 후 증강을 수행하여 신뢰성 있는 학습을 보장합니다.

- **Performance Highlights**: FishCNN 모델은 전통적인 기계 학습 모델들과 두 개의 최신 CNN 모델을 비교했을 때, 과적합(overfitting)을 줄이고 예측 정확도를 크게 향상시키는 성능을 보여주었습니다. 이러한 결과는 물고기 생화학 조성 분석의 정확하고 자동화된 접근 방식의 가능성을 열어줍니다.



### Instruction Embedding: Latent Representations of Instructions Towards Task Identification (https://arxiv.org/abs/2409.19680)
Comments:
          NeurIPS 2024

- **What's New**: 이 논문에서는 Large Language Models (LLMs)의 성능을 향상시키기 위해 Instruction Embedding의 개념을 도입하고 Instruction Embedding Benchmark (IEB)를 구축하여 새로운 평가 기준을 설정하였습니다. 특히, 이 연구는 특정 의미나 지식 정보를 넘어 다양한 작업을 수행하는 데 있어 지시 데이터의 중요성을 강조합니다.

- **Technical Details**: 연구에서는 Instruction Embedding이란 개념을 통해 명령문이 특정 작업 범주를 식별하는 데 초점을 맞춘 새로운 임베딩 방식을 제안합니다. 기존의 텍스트 임베딩 방법은 전반적인 의미 정보를 포착하는 데 중점을 두는 반면, Instruction Embedding은 서로 다른 작업 간의 유사성에 중점을 둡니다. IEB는 47,000개의 샘플과 1,000개 이상의 범주로 구성되어 있으며, Task Differentiation을 위한 기준으로 사용됩니다. 이와 함께 Prompt-based Instruction Embedding (PIE) 방법이 제안되어, 특정 작업 유형에 대한 집중을 통해 더 나은 결과를 도출할 수 있도록 하였습니다.

- **Performance Highlights**: IEB에서의 두 가지 설계된 작업에 대해 PIE 방법을 평가한 결과, 다른 임베딩 방법들보다 우수한 성능이 입증되었습니다. PIE는 작업 범주 정확히 식별하는 데 효과적이며, 제안된 Instruction Embedding의 사용은 전통적인 텍스트 임베딩보다 instruction 관련 작업에 더 적합함을 보여주었습니다.



### See Detail Say Clear: Towards Brain CT Report Generation via Pathological Clue-driven Representation Learning (https://arxiv.org/abs/2409.19676)
Comments:
          Our work has been accepted by EMNLP2024 findings

- **What's New**: 이 연구에서는 Pathological Clue-driven Representation Learning (PCRL) 모델을 도입하여 병리적 단서에 기반한 교차 모달 표현을 구축하고 이를 정확한 CT 보고서 생성에 자연스럽게 적합시키는 방법을 제안합니다.

- **Technical Details**: PCRL 모델은 세분화된 영역, 병리적 개념, 보고서 주제를 관점으로 하여 병리적 단서를 추출하고, 이들로부터 시각 병리적 패턴을 이해하고 교차 모달 특성 표현을 배우도록 합니다. 학습된 표현을 보고서 생성 작업에 적합하게 하기 위해, 작업 맞춤형 지침을 가진 통합 대형 언어 모델(LLM)을 사용하여 표현 학습과 보고서 생성을 연결합니다.

- **Performance Highlights**: 실험 결과, 우리의 방법은 이전 방법들보다 우수한 성능을 보였으며, 특히 뇌 CT 보고서 생성에서 SoTA 성능을 달성했습니다.



### Can Large Language Models Analyze Graphs like Professionals? A Benchmark, Datasets and Models (https://arxiv.org/abs/2409.19667)
Comments:
          NeurIPS 2024

- **What's New**: 이번 논문에서는 LLMs(대형 언어 모델)의 그래프 분석 능력을 평가하기 위해 새로운 ProGraph 벤치를 제안하였습니다. 이 벤치는 프로그램 코드 작성을 기반으로 한 해결책을 요구하며, 기존의 작은 그래프에 대한 직접적인 추론을 요구하는 방법과 차별화됩니다.

- **Technical Details**: ProGraph 벤치는 512개의 문제로 구성되어 있으며, 기본 그래프 이론, 그래프 통계 학습, 그래프 임베딩의 세 가지 범주를 포함합니다. 이를 위해 6개의 인기 있는 Python 라이브러리를 사용하고, LLM4Graph 데이터셋을 통해 문서와 코드 데이터를 결합하여 LLM들이 API를 통해 문제를 해결할 수 있도록 지원합니다.

- **Performance Highlights**: 폐쇄형 모델(Claude, GPT, Gemini)은 ProGraph에서 25-36%의 정확도를 기록하며, LLM4Graph를 활용한 RAG를 통해 37-46%로 향상될 수 있었습니다. 오픈 소스 모델(Llama3, Deepseek Coder)의 경우 12-24%의 정확도에서 45-47%로 개선되었습니다. 이러한 결과는 현재 LLM들이 ProGraph의 도전적인 문제를 효과적으로 해결할 수 있도록 해주는 LLM4Graph의 유용성을 강조합니다.



### Identifying Knowledge Editing Types in Large Language Models (https://arxiv.org/abs/2409.19663)
Comments:
          Under review

- **What's New**: 최근 대규모 언어 모델(LLMs)의 지식을 업데이트하기 위한 효율적인 방법으로 지식 수정(Knowledge editing)이 각광받고 있으나, 이를 악용할 수 있는 위험이 있음이 지적되었습니다. 이에 따라, 우리는 지식 수정 유형 식별(KETI)이라는 새로운 작업을 제안하고, 이를 통해 악의적인 수정을 식별할 수 있는 방법론을 마련했습니다.

- **Technical Details**: KETI는 서로 다른 유형의 지식 수정을 식별하는 것을 목표로 하며, KETIBench라는 새로운 벤치마크를 포함합니다. 이 데이터셋은 다양한 독성 수정과 하나의 무해한 사실 수정으로 구성되어 있습니다. 우리는 네 가지 고전적 분류 모델과 세 가지 BERT 기반 모델을 사용하여 공개 및 폐쇄 소스 LLM에서의 수정 유형을 식별하는 데 필요한 기준 모델을 개발했습니다.

- **Performance Highlights**: 실험 결과, 모든 분야의 기준 식별자가 상당한 성능을 달성했으며, KETIBench에서 평균 F1 점수는 0.789였습니다. 그러나 20%의 오류 식별률은 여전히 개선이 필요하며, 성능은 지식 수정 방법의 신뢰성과 독립적이며, 미지의 출처에서의 수정 식별 능력을 보여주었습니다.



### Assessment and manipulation of latent constructs in pre-trained language models using psychometric scales (https://arxiv.org/abs/2409.19655)
- **What's New**: 최근 대규모 언어 모델에서 인간과 유사한 성격 특성이 발견되었으며, 이러한 편향이 인간의 잠재적 심리적 구조와 일치한다는 가설이 제기되었습니다. 본 연구에서는 심리 측정 도구를 통해 이러한 특성을 평가하는 새로운 방법론을 제시합니다.

- **Technical Details**: 연구는 88개의 공개 모델 샘플을 분석하여 불안(anxiety), 우울(depression), 일관성의 감각(Sense of Coherence)과 같은 인간과 유사한 정신 건강 관련 구조의 존재를 보여줍니다. 또한 자연어 추론(NLI) 프롬프트를 기반으로 심리 측정 질문지를 재구성하고, 이를 통해 특정 모델의 편향을 두 가지 방법으로 평가합니다.

- **Performance Highlights**: 본 연구의 기여는 크게 네 가지로 나뉩니다: 1) 대화형 및 비대화형 모델 모두에 적용 가능한 심리적 특성 평가 방법론. 2) PLM의 잠재 구조 평가를 위한 Python 라이브러리. 3) 표준 질문지를 기반으로 하는 NLI 프롬프트 설계 방법론. 4) 정신 건강 평가와 관련된 NLI 프롬프트 데이터셋과 그 검증 과정.



### Grounding 3D Scene Affordance From Egocentric Interactions (https://arxiv.org/abs/2409.19650)
- **What's New**: 이 논문은 3D 장면의 affordance(가능성) 이해를 위해 egocentric(자기 중심) 비디오에서 상호작용을 통한 새로운 과제를 제안합니다. 이를 통해 기존의 정적 기하학적 구조와 시각적 외관에 의존하는 방법의 한계를 극복하고, 보다 능동적으로 환경을 인식하고 상호작용할 수 있는 모델을 개발하고자 합니다.

- **Technical Details**: Ego-SAG(예고적 상호작용 기반 3D 장면 affordance 정착) 프레임워크는 Interaction-Guided Spatial Significance Allocation Module(ISA)와 Bilateral Query Decoder Module(BQD)의 두 가지부터 구성됩니다. ISA는 공간 복잡성을 처리하는 데 초점을 맞추며, BQD는 다양한 출처 간 affordance 특징을 정렬하고 상호작용과 관련된 서브-영역에 모델 집중을 유도합니다.

- **Performance Highlights**: VSAD(Vide-3D Scene Affordance Dataset) 데이터셋을 사용한 실험 결과, Ego-SAG는 다른 대표적인 방법들에 비해 월등한 성능을 보였으며, 향후 연구를 위한 강력한 기준선으로 자리 잡을 것입니다.



### Fine-Tuning Hybrid Physics-Informed Neural Networks for Vehicle Dynamics Model Estimation (https://arxiv.org/abs/2409.19647)
- **What's New**: 본 논문에서는 Fine-Tuning Hybrid Dynamics (FTHD) 방법을 제안합니다. 이는 Supervised 및 Unsupervised Physics-Informed Neural Networks (PINNs)를 통합하여 물리 기반 모델링과 데이터 기반 기법을 결합합니다.

- **Technical Details**: FTHD는 사전 훈련된 Deep Dynamics Model (DDM)을 작은 훈련 데이터셋으로 세밀하게 조정하여, 최신 기법인 Deep Pacejka Model (DPM)보다 우수한 성능을 보여줍니다. 또한 EKF(Extended Kalman Filter)가 FTHD에 포함되어 EKF-FTHD를 통해 노이즈가 있는 현실 데이터 관리가 효과적으로 이루어집니다.

- **Performance Highlights**: 실험 결과, FTHD는 작고 제한된 데이터셋에서도 정확한 매개변수 추정 정확성을 크게 향상시킵니다. EKF-FTHD는 노이즈 데이터를 제거하면서 물리적 특성을 유지하여, 자율주행 고속 레이싱 차량의 동적 모델링에서 중요한 진전을 나타냅니다.



### BadHMP: Backdoor Attack against Human Motion Prediction (https://arxiv.org/abs/2409.19638)
- **What's New**: 본 논문은 BadHMP라는 새로운 백도어 공격 기법을 제안합니다. 이는 인간 동작 예측 모델을 타겟으로 하는 최초의 백도어 공격으로, 특정 관절에 로컬화된 트리거를 심어 poisoined training samples를 생성하여 예측 정확도를 유지하면서도 마치 공격을 받지 않은 것처럼 행동하도록 설계되었습니다.

- **Technical Details**: BadHMP는 두 가지 유형의 백도어 트리거(‘rigid’와 ‘soft’)와 두 가지 대상(‘jammed’와 ‘moving’)을 사용하여 네 가지 독특한 poisoning 전략을 구현합니다. 우리의 접근 방식은 기존의 human motion samples 데이터 형식의 특성을 고려하여, 자연스럽고 매끄러운 poisoned training samples를 생성하는 것을 목표로 합니다. 이를 위해 Clean Data Error (CDE)와 Backdoor Data Error (BDE)라는 새로운 평가 지표를 제안합니다.

- **Performance Highlights**: Human3.6M 및 CMU-Mocap 두 가지 데이터 세트와 LTD 및 HRI 두 가지 네트워크 아키텍처에서 수행된 실험 결과, BadHMP의 높은 정확도와 효과성을 입증하였고, 저비율 poisoined 샘플이 있어도 타겟 시퀀스를 성공적으로 활성화할 수 있음을 보여주었습니다. 또한, 모델의 fine-tuning 방어에 대한 공격의 강인성도 검증되었습니다.



### A Survey on Graph Neural Networks for Remaining Useful Life Prediction: Methodologies, Evaluation and Future Trends (https://arxiv.org/abs/2409.19629)
- **What's New**: 이 논문은 Remaining Useful Life (RUL) 예측을 위한 Graph Neural Networks (GNNs)의 사용을 종합적으로 검토합니다. GNNs는 복잡한 시스템에서 공간 정보를 모델링하는 데 있어 효과적인 방법을 제공합니다.

- **Technical Details**: 새로운 분류체계(Taxonomy)를 제안하여 GNN을 RUL 예측에 적합하게 조정하는 과정의 네 가지 주요 단계를 정의합니다: 그래프 구성(Graph Construction), 그래프 모델링(Graph Modeling), 그래프 정보 처리(Graph Information Processing), 그래프 리드아웃(Graph Readout).

- **Performance Highlights**: 다양한 최신 GNN 방법론에 대한 철저한 평가는 연구자들에게 유용한 벤치마크를 제공하며, GNNs가 RUL 예측을 개선할 수 있는 가능성을 강조합니다.



### Storynizor: Consistent Story Generation via Inter-Frame Synchronized and Shuffled ID Injection (https://arxiv.org/abs/2409.19624)
- **What's New**: 최근 텍스트-이미지 생성 모델의 발전으로 지속적인 이야기 이미지 생성에 대한 관심이 커졌습니다. 본 논문에서는 Storynizor라는 모델을 소개하며, 이는 높은 프레임 간 캐릭터 일관성, 효과적인 전경-배경 분리, 그리고 다양한 포즈 변화를 통해 일관된 이야기 생성을 지원합니다.

- **Technical Details**: Storynizor의 핵심 혁신은 ID-Synchronizer와 ID-Injector 두 가지 주요 모듈에 있습니다. ID-Synchronizer는 자동 마스크 자가 주의(auto-mask self-attention) 모듈과 마스크 지각 손실(mask perceptual loss)을 통해 캐릭터 생성을 일관되게 유지하며, ID-Injector는 Shuffle Reference Strategy(SRS)를 활용하여 ID 기능을 특정 위치에 통합합니다. 이러한 접근은 UNet 아키텍처를 기반으로 하여 작동하며, 훈련 과정에서 ID 일관성을 유지하도록 지원합니다.

- **Performance Highlights**: 실험 결과, Storynizor는 다른 캐릭터 특정 방법들에 비해 높은 충실도(character consistency)와 유연한 포즈(flexible postures)를 유지하며, 생생한 배경(vivid backgrounds)으로 일관된 이야기 이미지를 생성할 수 있음을 보여주었습니다.



### MCDDPM: Multichannel Conditional Denoising Diffusion Model for Unsupervised Anomaly Detection in Brain MRI (https://arxiv.org/abs/2409.19623)
Comments:
          Accepted in CISP-BMEI 2024

- **What's New**: 이번 연구에서는 뇌 MRI 스캔의 비지도 학습(anomaly detection)에서의 문제점을 해결하기 위해 Multichannel Conditional Denoising Diffusion Probabilistic Model (MCDDPM)이라는 개선된 모델을 제안합니다. 본 모델은 추가적인 건강한 이미지를 활용하여 더 높은 신뢰성과 왜곡 방지를 달성함으로써, 기존 DDPM 계열 모델의 문제를 해결합니다.

- **Technical Details**: MCDDPM은 여러 채널의 정보를 활용하여 비지도 이차 탐지에서 최종 이미지의 정확도와 현실성을 향상시킵니다. 이 과정에서 여러 모델을 사용할 필요 없이 직접적으로 컨텍스트 정보(contextual information)를 통합하여 모델 설계를 간소화합니다. 실험은 다양한 데이터셋(BraTS20, BraTS21 등)을 통해 수행되었습니다.

- **Performance Highlights**: 실험 결과, MCDDPM은 높은 품질의 이미지를 재구성하며, 뇌 MRI 스캔에서 비정상적인 영역의 픽셀 수준 식별을 효과적으로 지원합니다. 기존 모델들에 비해 MCDDPM의 성능이 더욱 향상되었음을 보여주는 데이터가 확보되었습니다.



### DropEdge not Foolproof: Effective Augmentation Method for Signed Graph Neural Networks (https://arxiv.org/abs/2409.19620)
Comments:
          NeurIPS 2024

- **What's New**: 본 논문에서는 친근하거나 적대적 관계를 나타내는 Signed Graphs의 링크 서명 예측(link sign prediction) 작업을 다룹니다. 기존의 Signed Graph Neural Networks(SGNNs)의 한계인 그래프 희소성(sparsity) 및 불균형 삼각형(unbalanced triangles) 문제를 해결하기 위해 데이터 증강(data augmentation) 기법을 제안합니다. 특히, 기존의 DropEdge 방법의 한계를 지적하며, 새로운 Signed Graph Augmentation(SGA) 프레임워크를 소개합니다.

- **Technical Details**: SGA는 구조 증강 모듈과 후보 엣지 선택 전략을 포함하여 SGNN 훈련을 개선합니다. SGA는 임베딩 공간에서 후보 샘플을 발견하기 위해 SGCN 알고리즘을 활용하며, 긴밀한 관계는 긍정적인 엣지로, 멀리 떨어진 관계는 부정적인 엣지로 해석합니다. 훈련 중 불균형 삼각형의 영향을 줄이기 위해 엣지 난이도 점수(edge difficulty scores)를 도입하여 커리큘럼 학습(curriculum learning) 전략을 적용합니다.

- **Performance Highlights**: SGA는 Slashdot 데이터셋에서 SGCN의 F1-micro 점수를 32.3% 개선하는 등, 총 6개 실제 데이터셋에서 5개의 기본 모델의 링크 서명 예측(link sign prediction) 정확도를 향상시켰습니다. 실험 결과, SGA는 SGNN의 성능을 상당히 향상시키는 효과를 보여주었습니다.



### Discerning the Chaos: Detecting Adversarial Perturbations while Disentangling Intentional from Unintentional Noises (https://arxiv.org/abs/2409.19619)
- **What's New**: 이 논문은 Class-Independent Adversarial Intent (CIAI) 탐지 네트워크를 소개합니다. 이 네트워크는 수정된 Vision Transformer를 기반으로 하며 탐지 레이어가 포함되어 있습니다. 새로운 손실 함수는 Maximum Mean Discrepancy와 Center Loss를 결합하여 이미지 클래스와 관계없이 의도적(적대적 공격) 및 비의도적 노이즈를 모두 탐지할 수 있게 설계되었습니다.

- **Technical Details**: CIAI 네트워크는 두 단계로 훈련됩니다. 첫 번째 단계에서는 Maximum Mean Discrepancy (MMD)와 Center Loss를 사용하여 Vision Transformer를 훈련하며, 두 번째 단계에서는 공격을 탐지하는 능력을 강화하기 위해 교차 엔트로피 손실을 사용합니다. 이 탐지 네트워크는 CelebA, CelebA-HQ, LFW, AgeDB, CIFAR-10 데이터셋에서의 성능을 평가합니다.

- **Performance Highlights**: 제안된 CIAI 탐지기는 FGSM, PGD, DeepFool과 같은 의도적 perturbations 뿐만 아니라 Gaussian 및 Salt & Pepper 노이즈와 같은 비의도적 perturbations도 탐지하는 데 성공적으로 작동합니다. 이를 통해 모델의 안정성과 보안을 강화할 수 있습니다.



### One Token to Seg Them All: Language Instructed Reasoning Segmentation in Videos (https://arxiv.org/abs/2409.19603)
Comments:
          Accepted by NeurlPS 2024

- **What's New**: 새로운 논문에서는 영상 기반의 다중 모달 대형 언어 모델인 VideoLISA를 소개합니다. 이는 언어 지시를 통한 영상 분할 문제를 해결하기 위해 설계되었습니다.

- **Technical Details**: VideoLISA는 Sparse Dense Sampling 전략과 One-Token-Seg-All 접근 방식을 통합하여 영상에서 객체를 세분화 및 추적하는 기능을 제공합니다. 특히 <TRK> 토큰을 사용하여 모델이 여러 프레임에 걸쳐 객체를 세분화할 수 있도록 합니다.

- **Performance Highlights**: 새로 도입된 ReasonVOS 벤치마크와 다양한 공개 벤치마크에서의 평가 결과, VideoLISA는 복잡한 추론, 시간적 이해 및 객체 추적을 포함한 비디오 객체 세분화 작업에서 우수한 성능을 보입니다.



### An Unbiased Risk Estimator for Partial Label Learning with Augmented Classes (https://arxiv.org/abs/2409.19600)
Comments:
          17 pages

- **What's New**: Partial Label Learning with Augmented Class (PLLAC)을 위한 새로운 접근 방식이 소개되었습니다. 기존 PLL 모델이 테스트 세트의 클래스가 훈련 세트에 존재해야 한다는 제약을 깨고, 훈련 단계에서 보이지 않았던 새로운 클래스를 효과적으로 인식할 수 있도록 해결책을 제시합니다.

- **Technical Details**: 제안된 방법은 편향이 없는 위험 추정기(Unbiased Risk Estimator)를 통해 미지의 클래스 분포를 식별하고, 이는 표본이 없어도 잘 훈련된 모델의 힘을 빌리는 방식입니다. 이 추정기는 각각의 PLL 손실 함수에 장착할 수 있도록 설계되었습니다. 또한, 이론적 추정 오차 경계를 제공하여, 훈련 데이터의 수가 많아질수록 경험적 위험 최소화기가 진짜 위험 최소화기로 수렴하도록 보장합니다.

- **Performance Highlights**: UCI 데이터세트 및 실제 데이터세트를 포함한 광범위한 실험에서 제안된 방법이 높은 성능을 발휘함을 입증하였습니다. 이는 PLLAC 문제를 효과적으로 해결할 수 있음을 보여줍니다.



### MASKDROID: Robust Android Malware Detection with Masked Graph Representations (https://arxiv.org/abs/2409.19594)
- **What's New**: 이 논문에서는 Android 악성 코드 탐지를 위한 새로운 시스템 MASKDROID를 제안합니다. MASKDROID는 기존의 그래프 기반 탐지기가 가진 취약성을 극복하고 강력한 구별 능력(discriminative ability)과 적대적 공격(adversarial attacks) 저항성을 갖출 수 있도록 설계되었습니다.

- **Technical Details**: MASKDROID는 그래프 신경망(Graph Neural Network, GNN) 기반의 프레임워크에 마스킹 메커니즘(masking mechanism)을 도입합니다. 이를 통해 MASKDROID는 전체 입력 그래프의 20%를 무작위로 선택하여 잃어버린 부분을 복구하는 방식으로 악성 코드에 대한 안정적인 표현(stable representations)을 학습합니다.

- **Performance Highlights**: 이 시스템은 정상 애플리케이션과 악성 애플리케이션을 효과적으로 구별하고, 적대적 예시를 포함한 다양한 공격 유형에 대해서도 강한 저항성을 발휘하여 기존 솔루션보다 우수한 성능을 보여줍니다.



### See then Tell: Enhancing Key Information Extraction with Vision Grounding (https://arxiv.org/abs/2409.19573)
- **What's New**: 본 논문에서는 STNet(See then Tell Net)라는 새로운 종단 간 모델을 도입합니다. 이 모델은 정확한 답변을 제공하고 관련된 비전 기반을 함께 제시하는 데 중점을 두고 설계되었습니다. 특히 <see> 토큰을 사용하여 이미지 내에서 적절한 영역을 관찰하고 텍스트 응답을 구성할 수 있습니다.

- **Technical Details**: STNet는 특별히 설계된 <see> 토큰을 활용하여 모델이 이미지 내에서 관련 위치를 인식하도록 지원합니다. 이는 전문화된 물리적 디코더를 통해 <see> 토큰과 연결된 물리적 좌표를 해석하게 됩니다. 다음 작업에서는 <see> 토큰을 답변 텍스트의 시작 부분에 배치하여 답변에 대한 비전 기반을 제공합니다.

- **Performance Highlights**: STNet 모델은 CORD, SROIE, DocVQA와 같은 공개 데이터셋에서 최첨단 성과를 달성하여 KIE 성능에서 상당한 발전을 보여주었습니다. 이를 통해 비전 기반을 활용한 KIE 모델의 성능이 크게 개선될 것으로 기대하고 있습니다.



### Mitigating the Negative Impact of Over-association for Conversational Query Production (https://arxiv.org/abs/2409.19572)
Comments:
          Information Processing & Management

- **What's New**: 이번 논문에서는 대화 이력(conversational histories)에서 검색 쿼리(query)를 생성하는 새로운 접근 방식을 제안한다. 기존의 모델들은 데이터 필요성(data hunger) 문제와 대화 이력에서 중요한 개념을 무시하거나 비관련 개념을 생성하는 문제를 안고 있으며, 이는 일반적으로 과잉 연관(over-association) 현상 때문이라는 점을 시사한다.

- **Technical Details**: 대화 쿼리 생성(task of conversational query generation)을 위한 모델을 발전시키기 위해, 저자들은 두 가지 인스턴스 수준의 가중치 조정 전략(instance-level weighting strategies)을 제안한다. 첫 번째는 데이터 기반 가중치 조정(data-based weighting)으로, 쿼리의 과잉 연관 정도에 따라 학습 속도를 조절한다. 두 번째는 모델 기반 가중치 조정(model-based weighting)으로, 모델이 자체 예측을 통해 학습하도록 유도한다. 이 두 가지 접근법을 통해 과잉 연관 문제를 완화하고 더 신뢰할 수 있는 쿼리를 생성한다.

- **Performance Highlights**: 실험은 Wizard-of-Internet 및 DuSinc 벤치마크에서 수행되었으며, 제안된 전략은 2%-5%의 성능 향상을 보여주었다. 또한, 새로운 모델은 대화 이력에서 더 나은 개념을 선택하고, 기본 모델에 비해 10배 더 효율적인 데이터 사용 효율성을 보였다.



### Abstractive Summarization of Low resourced Nepali language using Multilingual Transformers (https://arxiv.org/abs/2409.19566)
- **What's New**: 이번 연구는 네팔어를 위한 추상 요약(abstractive summarization) 분야에 대한 탐구로, 기존의 추출 요약(extractive summarization) 연구가 많이 이루어진 것과 대조적으로, 저자들은 mBART와 mT5와 같은 다국어 변환기 모델(multilingual transformer models)을 사용하여 네팔 뉴스 기사의 헤드라인을 생성하는 방법을 제안합니다.

- **Technical Details**: 이 연구에서는 다양한 네팔 뉴스 포털에서 수집한 데이터를 바탕으로 요약 데이터셋을 생성하고, 그 후 mBART와 mT5 모델을 다양한 전략으로 fine-tuning하였습니다. 모델의 성능은 ROUGE 점수와 인간 평가를 통해 확인하였으며, 교수 평가에서는 주제의 관련성, 유창성, 간결성, 정보성, 사실 정확성 및 범위와 같은 기준에 따라生成된 요약 중 가장 좋은 것을 선택하게 하였습니다. 특히, 4-bit quantized mBART with LoRA 모델이 다른 모델에 비해 효과적으로 네팔 뉴스 헤드라인을 생성하였습니다.

- **Performance Highlights**: fine-tuned 모델 평가 결과, 4-bit quantized mBART with LoRA 모델은 34.05%의 비율로 인간 평가에서 선택되었으며, 네팔 뉴스 헤드라인 생성을 위한 다른 모델보다 우수한 성능을 보여주었습니다.



### CLIP-based Camera-Agnostic Feature Learning for Intra-camera Person Re-Identification (https://arxiv.org/abs/2409.19563)
Comments:
          Submitted to IEEE TCSVT

- **What's New**: 이번 논문에서는 Contrastive Language-Image Pre-Training (CLIP) 모델을 활용한 새로운 프레임워크인 CLIP-based Camera-Agnostic Feature Learning (CCAFL)을 제안합니다. 이는 Intra-Camera Supervised Re-Identification (ICS ReID) 문제를 해결하기 위해 고안되었습니다.

- **Technical Details**: 논문에서 제안된 프레임워크는 두 개의 주요 모듈인 Intra-Camera Discriminative Learning (ICDL)과 Inter-Camera Adversarial Learning (ICAL)을 포함합니다. ICDL은 카메라 내에서의 세부적인 보행자 특징을 학습하도록 유도하고, ICAL은 서로 다른 카메라 간의 보행자 특징의 차이를 줄이기 위해 모델의 카메라 예측 능력을 저해합니다.

- **Performance Highlights**: MSMT17 데이터셋에서 mAP 정확도를 58.9%로 기록하며, 기존의 최첨단 방법들을 7.6% 초과하는 성능을 보여주었습니다. Market-1501과 MSMT17을 포함한 다양한 ReID 벤치마크에서 우수한 성능을 입증했습니다.



### A Universal Deep Learning Framework for Materials X-ray Absorption Spectra (https://arxiv.org/abs/2409.19552)
Comments:
          Main manuscript: 21 pages, 11 figures. Supplemental material (12 pages, 6 figures) available as a separate file in arXiv ancillary files (additional downloadable files)

- **What's New**: 본 논문은 X선 흡수 분광법(X-ray Absorption Spectroscopy, XAS)의 데이터 분석을 위한 빠르고 강력한 파이프라인을 개발하기 위한 여러 가지 전이 학습(transfer learning) 접근 방식을 제시합니다.

- **Technical Details**: 세 가지 독특한 전략을 통해 XAS 예측의 정확성과 효율성을 개선합니다. 첫째, M3GNet을 사용하여 흡수 사이트의 지역 화학 환경에 대한 잠재 표현(latent representation)을 도출하고, 전통적인 특성화 방법보다 몇 배 향상된 성능을 보입니다. 둘째, 요소들 간의 공통 모델을 훈련시킨 후 각 요소에 대해 미세 조정(fine-tuning)을 실시함으로써, 개별 모델보다 최대 31% 향상된 결과를 얻을 수 있습니다. 셋째, 서로 다른 신뢰도(fidelity)로 생성된 스펙트라에 대해 일반 모델을 조정하여 예측 정확도를 최대 24% 향상시키는 방법을 제안합니다.

- **Performance Highlights**: 이 접근 방식은 3d 전이 금속(Ti-Cu)의 K-edge 스펙트라 데이터베이스를 사용하여 입증되었고, 더욱 다양한 요소의 XAS 예측으로 확장 가능하고, 재료 과학의 다른 딥러닝 모델에도 일반화할 수 있는 전이 학습 프레임워크를 제공합니다.



### Almost Sure Convergence of Average Reward Temporal Difference Learning (https://arxiv.org/abs/2409.19546)
- **What's New**: 본 논문에서는 개념적으로 매우 간단한 TD 학습이 평균 보상 강화 학습에서 거의 확실한 수렴성을 가지는 것을 처음으로 증명하였습니다. 이는 25년이 넘는 시간동안 미해결 과제로 남아있던 문제로, 새롭게 도입된 확률적 근사 결과에 기반하고 있습니다.

- **Technical Details**: 이 연구는 Markovian 및 additive noise가 포함된 비확장적 맵핑에 관한 새로운 일반적인 확률적 근사 결과를 포함합니다. 기존의 Stochastic Krasnoselskii-Mann (SKM) 반복에 대한 수렴 분석을 확장하여, Tabular average reward TD(Temporal Difference)에서의 수렴성을 증명하였습니다.

- **Performance Highlights**: 이 논문의 핵심 기여는 평균 보상 TD의 반복 업데이트가 약한 조건 하에 샘플 경로에 의존하는 고정점에 거의 확실하게 수렴함을 증명한 것입니다. 이는 향후 다른 강화 학습 알고리즘의 분석에서도 활용될 수 있는 중요한 발판이 될 것입니다.



### Unlabeled Debiasing in Downstream Tasks via Class-wise Low Variance Regularization (https://arxiv.org/abs/2409.19541)
Comments:
          Accepted to EMNLP 2024

- **What's New**: 본 연구는 기존의 레이블링된 속성에 의존하지 않고 다양한 속성에 대한 디바이싱(debiasing) 문제를 해결하기 위해 새로운 정규화 기법인 Low Variance Regularization (LVR)을 도입합니다. 이 접근법은 임베딩 내 특성 정보 손실을 줄이면서도 분류 성능을 유지하는 데 초점을 맞춥니다.

- **Technical Details**: Low Variance Regularization (LVR)은 k 개의 클래스를 각각 대표하는 중심(center) 기반으로 정규화를 수행합니다. 이 기법은 레이블이 없는 상태에서 모든 보호 속성(protected attribute)에 대한 영향을 동시에 낮추며 모델의 임베딩에서 분포적 이동을 줄여 공정한 표현을 유지합니다.

- **Performance Highlights**: LVR 기법은 BERT-Base 및 RoBERTa-Base와 같은 기존의 인코더 언어 모델에서 문서 분류 작업을 통해 평가되었으며, 기존의 강력한 디바이싱 기준선을 능가하며, 목표 작업에 대한 성능을 유지하면서 속성 제거를 효과적으로 이루어냈음을 입증하였습니다.



### Understanding Clinical Decision-Making in Traditional East Asian Medicine through Dimensionality Reduction: An Empirical Investigation (https://arxiv.org/abs/2409.19531)
Comments:
          11 pages, 3 figures

- **What's New**: 본 연구는 전통 동아시아 의학(TEAM)의 임상 의사 결정 과정을 차원 축소(dimensionality reduction) 관점에서 재해석하며, 외부-내부(Exterior-Interior) 패턴의 중요성과 필요성을 탐구합니다.

- **Technical Details**: 본 연구는 팔극 패턴 식별(Eight Principle Pattern Identification, EPPI) 시스템을 중심으로, 상한론(Shang-Han-Lun)에서 수집된 경험적 데이터를 활용하여 진단 및 치료 선택에서 외부-내부 패턴의 우선 순위를 정립합니다. 양적 측정 방법으로 추상화 지수(abstraction index), 교차 조건 일반화 성능(cross-conditional generalization performance), 결정을 나무 회귀(decision tree regression)를 사용하였습니다.

- **Performance Highlights**: 결과적으로 외부-내부 패턴이 가장 추상적이고 일반화 가능한 증상 정보를 나타내며, 증상과 한약 처방 공간 간의 효율적인 매핑(efficient mapping)을 촉진함을 보여주었습니다. 이는 TEAM과 현대 컴퓨팅 접근법을 연결하는 객관적인 틀을 제공하며, AI 기반 진단 도구 개발에 대한 통찰력을 제공합니다.



### Efficient Backdoor Defense in Multimodal Contrastive Learning: A Token-Level Unlearning Method for Mitigating Threats (https://arxiv.org/abs/2409.19526)
- **What's New**: 본 연구는 멀티모달 대비 학습(Multimodal Contrastive Learning)에서 발생할 수 있는 backdoor 공격에 대한 새로운 방어 기제를 제안합니다. 이를 위해 '기계 학습 삭제(machine unlearning)' 개념을 활용하여 모델의 backdoor 취약성을 신속하게 제거하는 방법을 제시합니다.

- **Technical Details**: 제안하는 기법은 Unlearn Backdoor Threats (UBT)로, 적은 수의 오염 샘플을 선택하여 모델이 backdoor 특징을 잊도록 유도합니다. 이 과정에서 과도적 훈련(overfit training)을 통해 의심스러운 샘플을 탐지하고, 해당 샘플의 일부를 선택하여 신속하게 제거합니다. 이 새로운 접근법은 토큰 기반의 부분 학습 삭제(training regime) 방법을 포함하여, 모델의 취약한 요소에 집중하여 backdoor 연관성을 분리합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 다양한 backdoor 공격 방법에 대해 효과적으로 방어하며, 기존 방법과 비교할 때 공격 성공률(attack success rate)을 19% 감소시키고, 깨끗한 정확도(clean accuracy)를 2.57% 증가시켰습니다.



### KODA: A Data-Driven Recursive Model for Time Series Forecasting and Data Assimilation using Koopman Operators (https://arxiv.org/abs/2409.19518)
- **What's New**: 본 연구에서는 Koopman 연산자(Koopman operator)를 기반으로 한 데이터 동화(data assimilation) 접근 방법인 KODA(Koopman Operator with Data Assimilation)를 제안합니다. 이는 예측(forecasting)과 데이터 동화를 통합하여 비선형 동적 시스템(NLDS)에 적용됩니다.

- **Technical Details**: KODA는 Fourier 도메인 필터를 사용하여 데이터를 물리적 구성 요소와 잔여 동적(residual dynamics)으로 분리합니다. 이 과정에서 Koopman 연산자는 물리적 구성 요소의 동적을 정확히 표현하고, 잔여 동적은 유연하고 학습 가능한 재귀 모델로 캡처됩니다. 이러한 구조와 교육 기준은 안정적이고 장기 예측을 가능하게 합니다.

- **Performance Highlights**: KODA의 성능을 검증하기 위해 전기, 온도, 날씨, Lorenz 63, Duffing 오실레이터와 같은 여러 시간 시계열 벤치마크에서 기존의 최첨단 방법들보다 우수한 예측 결과를 보였습니다. KODA는 예측, 데이터 동화, 상태 예측의 세 가지 작업에서 뛰어난 효과성을 입증하였습니다.



### One Node Per User: Node-Level Federated Learning for Graph Neural Networks (https://arxiv.org/abs/2409.19513)
Comments:
          16 pages, 9 figures

- **What's New**: 이 논문에서는 Graph Neural Networks (GNNs)에 대한 node-level federated learning (FL) 프레임워크인 nFedGNN을 제안합니다. 이 프레임워크는 각 클라이언트가 단 하나의 feature vector만을 소유하는 상황에서도 협업 모델 훈련이 가능하도록 설계되었습니다.

- **Technical Details**: nFedGNN은 첫 번째 GNN 레이어의 message-passing과 feature vector 변환 프로세스를 분리하여 클라이언트와 클라우드 서버에서 각각 수행할 수 있도록 합니다. 또한, 단일 위치에서 feature vector의 latent representation에 기반한 graph Laplacian 용어를 도입하여 사용자 측 모델 업데이트를 규제합니다.

- **Performance Highlights**: 여러 데이터셋에 대한 실험 결과, nFedGNN은 기존의 기준 모델과 비교하여 우수한 성과를 달성하였습니다.



### Heterogeneity-Aware Resource Allocation and Topology Design for Hierarchical Federated Edge Learning (https://arxiv.org/abs/2409.19509)
Comments:
          12 pages, 9 figures

- **What's New**: 이번 연구는 Federated Learning (FL) 방법 중 Hierarchical Federated Edge Learning (HFEL)의 훈련 효율성을 높이는 전략적 자원 할당과 토폴로지 설계를 제안합니다.

- **Technical Details**: 연구자는 두 계층으로 구성된 HFEL 시스템을 고려하고, 에지 장치가 에지 서버와 연결되며 이들 서버가 P2P (peer-to-peer) 에지 백홀을 통해 상호 연결된 구조를 설계했습니다. 최적화 문제를 수립하여 통신 및 계산 자원을 할당하고 P2P 연결을 조정하여 총 훈련 지연 시간을 최소화합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 훈련 지연 시간을 획기적으로 감소시키면서도 모델의 정확도를 다양한 기준선과 비교해 유지하는 효과를 보였으며, 데이터 및 시스템 이질성 하에서도 대규모 FL 구현을 가능하게 합니다.



### Learning Frame-Wise Emotion Intensity for Audio-Driven Talking-Head Generation (https://arxiv.org/abs/2409.19501)
- **What's New**: 이번 논문에서는 감정 표현의 동적인 변동성을 효과적으로 모델링하기 위한 새로운 프레임워크를 제안합니다. 이는 오디오 신호를 기반으로 감정의 강도를 시뮬레이션하여 보다 자연스럽고 다양한 감정을 표현하는 talking-head(토킹헤드) 비디오 생성 기술을 발전시킵니다.

- **Technical Details**: 제안된 방법은 라벨링이 필요 없는 감정 불가지론적 인텐시티(인텐시티 pseudo-labeling) 방식을 통해 감정의 강도 변화를 추출합니다. 이를 바탕으로, 음성을 분석하여 그에 따른 인텐시티를 예측하는 오디오-인텐시티 예측기를 개발합니다. 마지막으로 연속적인 감정 잠재 공간(latent space)을 구축하여 감정 유형과 강도를 유기적으로 표현합니다.

- **Performance Highlights**: 우리의 실험 결과는 제안된 방법이 감정 강도 변화를 정확하게 캡처하고 재현할 수 있음을 보여줍니다. MEAD와 LRW 데이터셋에서의 양적 결과와 질적 분석을 통해, 생성된 표현이 뛰어난 자연스러움과 사실성을 가짐을 입증하였습니다.



### MedHalu: Hallucinations in Responses to Healthcare Queries by Large Language Models (https://arxiv.org/abs/2409.19492)
Comments:
          14 pages

- **What's New**: 이 연구에서는 대형 언어 모델(LLM)이 실제 의료 질문에 대한 응답에서 발생하는 환각(hallucination)의 최초 연구를 진행했습니다. MedHalu라는 의료 환각 데이터셋을 처음으로 제안하였으며, 다양한 건강 관련 주제와 LLM이 생성한 허위 응답을 포함하고 있습니다.

- **Technical Details**: 제안된 MedHaluDetect 프레임워크는 다양한 LLM의 환각 탐지 능력을 평가하여 LLM의 응답에서 발생하는 환각의 유형과 이들을 식별하는 방법을 포함하고 있습니다. 연구에는 의료 전문가, LLM, 일반인을 포함한 세 그룹의 평가자가 참여하여 누가 의료 환각에 더 취약한지를 조사했습니다.

- **Performance Highlights**: 연구 결과, LLM은 전문가보다 환각 탐지 성능이 낮으며, 일부 경우에는 일반인보다도 성능이 떨어지는 것으로 나타났습니다. 그러나 전문가의 추론을 LLM에 주입하는 expert-in-the-loop 접근 방식을 적용한 결과, 모든 LLM에서 평균 6.3 포인트의 macro-F1 점수가 향상되었습니다.



### Spatial Reasoning and Planning for Deep Embodied Agents (https://arxiv.org/abs/2409.19479)
Comments:
          DPhil Thesis - Engineering Science, University of Oxford. Original copy available at this https URL

- **What's New**: 이 논문은 Embodied agents가 복잡한 작업을 수행할 수 있도록 하는 데이터 중심 기술 개발에 초점을 맞추고 있습니다. 주요 기여로는 해석 가능하고 장기 계획을 위한 모델을 학습하는 CALVIN, 무감독 RL 알고리즘 SOAP, 코드 최적화 프레임워크 LangProp, 복잡한 작업을 수행하는 Voggite가 포함됩니다.

- **Technical Details**: 1) CALVIN은 전문가 시연으로부터 보상 및 상태 전환을 학습하여 부분적으로 관찰 가능한 3D 환경에서 성공적으로 탐색하는 차별적 계획자입니다. 2) SOAP은 작업을 하위 작업으로 분할하고 일관된 실행을 가능하게 하는 옵션을 발견하는 RL 알고리즘입니다. 3) LangProp은 LLM을 사용하여 코드 최적화를 통해 코드가 학습 가능한 정책으로 처리되는 코드 최적화 프레임워크를 제공합니다. 4) Voggite는 Minecraft에서 복잡한 작업을 해결하는 비전-액션 변환기를 사용하는 Embodied agent입니다.

- **Performance Highlights**: 이 연구의 성과로는 CALVIN이 3D 환경에서의 탐색 성능 향상, SOAP이 Atari와 같은 고전 벤치마크에서 강력한 성능, LangProp이 CARLA 자율 주행 벤치마크에서 인간 전문가와 동등한 성과를 달성하며, Voggite가 MineRL BASALT 대회에서 3위를 차지한 점이 있습니다.



### FairPIVARA: Reducing and Assessing Biases in CLIP-Based Multimodal Models (https://arxiv.org/abs/2409.19474)
Comments:
          14 pages, 10 figures. Accepted to 35th British Machine Vision Conference (BMVC 2024), Workshop on Privacy, Fairness, Accountability and Transparency in Computer Vision

- **What's New**: 이 논문은 비전-언어 모델의 윤리적 의미에 초점을 맞추고 이들 모델에서 발생할 수 있는 차별적 관행을 분석합니다. 특히 모델이 타 언어에 맞춤화되는 과정에서 나타나는 새로운 편향을 논의하며, FairPIVARA라는 편향 감소 기술을 제안합니다.

- **Technical Details**: FairPIVARA는 임베딩(feature embedding)에서 가장 부정적인 기여를 하는 차원(dimension)을 제거하여 편향을 줄이는 알고리즘입니다. 이 모델은 CAPIVARA를 기반으로 하며, 주로 Disability, Nationality, Religion, Sexual Orientation 측면에서 차별적 실천을 분석합니다. 모델의 성능은 98%까지 감소된 편향을 나타냈고, 단어 분포의 균형도 증진되었습니다.

- **Performance Highlights**: FairPIVARA의 적용을 통해 편향 관찰이 98%까지 줄어들었으며, 모델 내에서 단어 분포의 균형을 촉진했습니다. 이 연구는 영어와 포르투갈어 결과를 모두 보고하며, 낮은 자원 언어로의 모델 적합에 대한 이해를 높였습니다.



### SELP: Generating Safe and Efficient Task Plans for Robot Agents with Large Language Models (https://arxiv.org/abs/2409.19471)
- **What's New**: 본 논문은 로봇 에이전트가 자연어 명령을 이해하고 수행하는 능력을 향상시키기 위해 대형 언어 모델(LLMs)을 활용하면서도, 사용자가 지정한 제약조건을 준수하도록 보장하는데의 도전 과제를 다룹니다. 'Safe Efficient LLM Planner(SELP)'라는 접근법을 제안하며, 이는 복잡한 작업을 처리하는 LLM 계획 수립자의 능력을 개선하는 세 가지 주요 통찰력을 포함합니다.

- **Technical Details**: 이 연구에서는 '동등성 투표(equivalence voting)', '제약 Decoding(constrained decoding)', '도메인 특화 미세 조정(domain-specific fine-tuning)' 기법을 통해 LLM 계획 수립자의 성능을 향상시킵니다. 동등성 투표는 NL 명령에서 여러 Linear Temporal Logic (LTL) 공식을 생성하고 샘플링하여 일치하는 LTL 공식을 그룹화하고, 다수 그룹의 공식을 최종 LTL 공식으로 선택하는 방법입니다. 제약 Decoding은 생성된 LTL 공식을 사용하여 계획의 자동 회귀 추론을 강제하며, 여기서 LLM은 주어진 명세에 부합하도록 계획을 수정합니다.

- **Performance Highlights**: SELP는 다양한 로봇 에이전트와 작업에서 효과성과 일반성을 보여주며, 드론 내비게이션 작업에서는 안전률에서 SOTA 계획 수립자보다 10.8% 향상되었고, 계획 효율성에서 19.8% 향상되었습니다. 로봇 조작 작업에서는 20.4% 안전률 개선을 달성하였습니다. 이를 통해 SELP는 높은 신뢰도로 사용자의 명령에 부합하는 계획을 세울 수 있음을 입증하였습니다.



### INSIGHTBUDDY-AI: Medication Extraction and Entity Linking using Large Language Models and Ensemble Learning (https://arxiv.org/abs/2409.19467)
Comments:
          ongoing work, 24 pages

- **What's New**: 이번 연구에서는 약물 정보 추출과 관련 속성을 텍스트 마이닝하는 상태-of-the-art LLMs를 조사하였으며, Stack-Ensemble 및 Voting-Ensemble과 같은 다양한 앙상블 학습 기법을 적용하여 모델 성능을 향상시켰습니다. 또한, 추출된 약물 용어를 SNOMED-CT 및 BNF 코드로 매핑하는 엔터티 링크 기능도 개발하였습니다.

- **Technical Details**: 이 연구에서는 약물의 복용량(dosage), 경로(route), 강도(strength), 부작용(adverse effects) 등과 같은 속성을 포함한 약물 텍스트 마이닝을 다루며, 임상 NER(named entity recognition) 작업을 위해 8개 모델(BERT, RoBERTa 등)을 미세 조정하는 과정에서 Stack 및 Voting 앙상블 메커니즘을 조사했습니다. 엔터티 링크 기능을 통해 임상 사건을 SNOMED-CT와 BNF에 연결하고, 이는 이후 ICD 및 dm+d와 매핑됩니다.

- **Performance Highlights**: 앙상블 학습 결과는 BERT, RoBERTa, BioBERT 등 개별 모델보다 효율적으로 성능을 끌어올렸습니다. 이 연구에서 사용된 n2c2-2018 데이터셋에 대한 초기 평가 결과, 각 모델의 성능 개선을 확인했습니다.



### See Where You Read with Eye Gaze Tracking and Large Language Mod (https://arxiv.org/abs/2409.19454)
Comments:
          9 pages

- **What's New**: 이 논문에서는 선형 독서(linear reading)와 점프 독서(jump reading)를 지원하는 새로운 독서 추적 및 강조 시스템 R²TH를 제시합니다. 특히, 16명의 사용자를 대상으로 한 실험을 바탕으로 두 가지 시선 오류 모델을 설계하여 점프 독서 감지 및 위치 변경을 가능하게 했습니다.

- **Technical Details**: 이 시스템은 gaze error model을 사용하여 사용자 시선을 기반으로 독서 진행 상황을 추적하며, 독서 추적 도메인에 맞춘 라인-시선 정렬 기회를 활용해 동적인 캘리브레이션(calibration)을 수행합니다. 점프 독서를 감지할 경우, 사용자의 시선 경로를 추적하여 문장 구두점을 검색하고, LLM(large language model)의 맥락 인지 능력을 활용해 후보 문장을 평가합니다.

- **Performance Highlights**: 통제된 실험에서는 선형 독서 추적의 신뢰성을 입증했으며, 점프 독서 추적의 경우 84%의 정확도를 기록했습니다. 또한, 18명의 자원봉적으로 토대로 한 실제 현장 테스트에서 본 시스템의 효과성을 입증하여 독서 효율성을 개선하고 사용자 경험을 강화했습니다.



### Secret Use of Large Language Models (https://arxiv.org/abs/2409.19450)
Comments:
          26 pages, 3 figures, and accepted at CSCW 2025

- **What's New**: 대형 언어 모델(LLM)의 발전은 AI 사용에 대한 투명성에 대한 책임을 분산시킵니다. 특히, LLM 사용자들은 실세계 과제에서 LLM으로 생성된 콘텐츠 사용을 공개해야 할 필요성이 커지고 있습니다. 그러나 LLM의 비밀 사용은 사용자가 투명성 요건을 준수하는 데 어려움을 초래하고 있습니다.

- **Technical Details**: 이 연구는 혼합 방법론을 사용하여 125개의 비밀 LLM 사용 사례에 대한 탐색적 조사와 300명의 사용자 대상으로 한 통제 실험을 통해 LLM 비밀 사용의 맥락과 원인을 조사했습니다. 연구 결과에 따르면, 비밀스러운 행동은 특정 작업에 의해 촉발되며, 인구 통계학적 및 개인차를 초월합니다.

- **Performance Highlights**: 일반 정보 검색에 비해, "창의적 글쓰기", "학술 글쓰기", "학교 과제", "업무 과제", "사회적 연결" 등의 실험적 작업은 비밀 사용 의도에서 높은 차이를 보였습니다. 조사 결과는 LLM/AI 사용의 투명한 공개를 장려하기 위한 인터벤션 디자인에 중요한 통찰을 제공합니다.



### Advanced Clustering Techniques for Speech Signal Enhancement: A Review and Metanalysis of Fuzzy C-Means, K-Means, and Kernel Fuzzy C-Means Methods (https://arxiv.org/abs/2409.19448)
- **What's New**: 이 논문은 음성 신호 처리에서의 최신 발전을 다루고 있으며, 특히 Kernel Fuzzy C-Means (KFCM) 기법이 전통적인 방법들과 비교하여 어떻게 더 우수한 성능을 발휘하는지를 설명합니다.

- **Technical Details**: KFCM 방법론은 비선형(non-linear) 및 비정상(non-stationary) 노이즈 환경에서의 음성 신호 인식을 효과적으로 수행할 수 있도록 설계되었습니다. 이 리뷰는 현재의 클러스터링(clustering) 알고리즘의 다 비교 분석 및 KFCM과 신경망(neural networks)을 조합한 하이브리드(hybrid) 모델 통합에 대한 제안도 포함하고 있습니다.

- **Performance Highlights**: KFCM은 다양한 노이즈 환경에 적응할 수 있는 유연성을 보여주어 음성 향상(sppech enhancement) 애플리케이션에서 강력한 선택임을 입증하였습니다. 이 설정은 자동 전사 서비스 및 음성 인식 경험을 향상시키는 데 기여할 것입니다.



### Strongly-Polynomial Time and Validation Analysis of Policy Gradient Methods (https://arxiv.org/abs/2409.19437)
- **What's New**: 이번 논문은 강화학습(Reinforcement Learning, RL)에서 최적성(optimality)에 대한 원칙적인 측정 지표가 부족하다는 문제를 다루며, 유한 상태 및 행동 마르코프 결정 과정(Markov Decision Process, MDP)에서 간단하고 계산 가능한 새로운 gap function을 개발했습니다. 이를 통해 최적성 gap의 상한과 하한을 제공하며, 새로운 개념인 분포 독립 수렴(distribution-free convergence)을 정의합니다.

- **Technical Details**: 개발된 gap function은 벡터의 최대 원소를 찾거나(convex optimization problem) 해결하는 방식으로 이루어집니다. 기본 정책 거울 하강(Policy Mirror Descent, PMD)을 통해 결정론적(deterministic) 및 확률론적(stochastic) 환경에서의 분포 독립 수렴을 보장하며, 특히 결정론적 환경에서는 최적 정책을 강한 다항 시간 내에 찾을 수 있는 방법론을 제시합니다.

- **Performance Highlights**: 결정론적 설정에서 PMD는 선형 수렴률을 보여주며, 확률론적 환경에서는 O(k^{-1/2})와 O(k^{-1})의 수렴률을 제공합니다. 또한, 새로운 결과로는 기본 확률 정책 거울 하강(Basic Stochastic Policy Mirror Descent, SPMD)이 정확한 gap function의 온라인 추정을 할 수 있으며, 이를 통해 신뢰할 수 있는 종료 기준으로 사용할 수 있는 정확도 추정치를 제공합니다.



### RMLR: Extending Multinomial Logistic Regression into General Geometries (https://arxiv.org/abs/2409.19433)
Comments:
          Accepted to NeurIPS 2024

- **What's New**: 본 논문은 일반 기하학에서 Riemannian Multinomial Logistic Regression(RMLR)을 설계하기 위한 새로운 프레임워크를 제안합니다. 이 프레임워크는 최소한의 기하학적 속성만 요구하며, 다양한 기하학과 함께 사용할 수 있는 넓은 적용성을 보여줍니다.

- **Technical Details**: RMLR는 기본적으로 Riemannian logarithm의 명시적 표현만을 요구하며, 이는 머신 러닝에서 자주 사용되는 여러 매니폴드에 의해 충족됩니다. 논문에서는 SPD 매니폴드와 회전 행렬에 대해 RMLR 프레임워크를 구체적으로 평가하며, SPD 매니폴드에서는 다섯 가지 가족의 SPD MLR을 제안하고, 회전 행렬에서는 널리 사용되는 bi-invariant metric을 기반으로 한 Lie MLR을 발전시킵니다.

- **Performance Highlights**: 제안된 RMLR은 다양한 Riemannian 백본 네트워크에서 효과가 입증되었습니다. 특히, SPD MLR은 SPDNet 및 RResNet에서 각각 14.23% 및 13.72% 향상을 보였으며, EEG 분류 작업에서는 TSMNet에서 4.46% 개선되었습니다. 또한, Lie MLR은 훈련의 안정성과 성능을 모두 향상시키는 데 기여했습니다.



### MicroFlow: An Efficient Rust-Based Inference Engine for TinyML (https://arxiv.org/abs/2409.19432)
- **What's New**: MicroFlow는 Rust 프로그래밍 언어로 개발된 경량의 오픈소스 TinyML 프레임워크로, 임베디드 시스템에서 Neural Networks(NNs)를 효율적으로 배포할 수 있도록 설계되었습니다. 이는 메모리 안전성과 특징을 강조하며, 자원 제약이 있는 환경에서도 안정적인 작동이 가능합니다.

- **Technical Details**: MicroFlow는 컴파일러 기반의 inference engine 방식을 사용하고 있으며, 메모리 안전성을 보장하는 Rust를 활용하여 일반적인 메모리 관련 오류를 방지합니다. 메모리는 정적 할당이 가능하며, 8비트 마이크로컨트롤러에서도 NN 모델의 일부만을 RAM에 로드하여 처리할 수 있게 설계되었습니다. 또한, MicroFlow의 코드와 구현은 완전히 오픈소스로, GitHub에서 무료로 제공됩니다.

- **Performance Highlights**: MicroFlow는 기존의 최신 솔루션에 비해 적은 Flash 및 RAM 메모리를 사용합니다. 중간 크기 NN에 대해 더 빠른 inference 속도를 달성하며, 대형 NN의 경우 유사한 성능을 보입니다. 실험 결과에 따르면 자원이 제한된 환경에서도 효율적인 TinyML 모델 배포가 가능함을 입증하였습니다.



### Identifiable Shared Component Analysis of Unpaired Multimodal Mixtures (https://arxiv.org/abs/2409.19422)
- **What's New**: 본 연구는 크로스 모달리티(sample들 간) 데이터가 정렬되지 않았을 때도 공유 구성 요소(shared components)를 식별할 수 있는 충분한 조건을 제공하며, 기존 연구보다 더 완화된 조건을 제시합니다.

- **Technical Details**: 연구에서는 unaligned shared component analysis (unaligned SCA) 문제를 다루며, 이를 위해 분포 차이(distribution divergence)를 최소화하는 손실(loss) 함수를 제안합니다. 이 손실 함수는 다중 모달 데이터의 확률 분포를 일치시키는 방식을 통해 공유 구성 요소를 식별합니다.

- **Performance Highlights**: 제안된 방법은 실제 데이터 및 합성 데이터에서 유효성을 검증하였으며, 크로스-언어(word retrieval), 유전 정보 정렬(genetic information alignment), 이미지 데이터 도메인 적응(image data domain adaptation)과 같은 응용 분야에 효과적으로 적용될 수 있음을 보여주었습니다.



### Subject Data Auditing via Source Inference Attack in Cross-Silo Federated Learning (https://arxiv.org/abs/2409.19417)
- **What's New**: 본 논문에서는 Federated Learning (FL)에서의 새로운 공격 기법인 Subject-Level Source Inference Attack (SLSIA)를 제안합니다. SLSIA는 기존의 Source Inference Attack (SIA)와 Subject Membership Inference Attack (SMIA)의 한계를 극복하고 여러 클라이언트가 타겟 데이터 포인트를 사용하는 경우를 조사가 가능합니다.

- **Technical Details**: SLSIA는 중앙 서버에서 특정 데이터 출처를 통제하고, 타겟 주제의 데이터로 학습한 클라이언트를 감지하기 위해 이진 공격 분류기(binary attack classifier)를 활용하여 로컬 모델의 임베딩(embedding)을 분석합니다. 공격자는 타겟 주제로부터 파생된 데이터를 사용하여 모델을 사전 훈련하고, 이를 바탕으로 이진 공격 분류기를 위한 훈련 세트를 구축합니다.

- **Performance Highlights**: SLSIA는 세 가지 데이터 세트에서 이전 방법들보다 높은 성능을 보이며, 최대 50개의 타겟 주제에 대해 평균 정확도 0.88을 기록했습니다. 특히, 희소한 주제를 가진 데이터세트에서 SLSIA의 공격이 더 효과적이라는 분석 결과를 제시합니다.



### Membership Privacy Evaluation in Deep Spiking Neural Networks (https://arxiv.org/abs/2409.19413)
- **What's New**: 이 논문에서는 Spiking Neural Networks (SNNs)의 멤버십 프라이버시를 평가하고, 인공신경망(ANNs)과 비교하여 SNNs의 취약성을 연구합니다. 특히 SNNs가 데이터와의 시간적 관계를 고려하여 멤버십 인퍼런스 공격(마를 후)에 대해 더 높은 취약성을 보임을 발견하였습니다.

- **Technical Details**: SNNs는 생물학적 뉴런의 동작을 모사하고, 이진 스파이크(binary spike)를 생성하여 활성화합니다. 논문에서는 8개의 멤버십 인퍼런스 공격(MIAs)을 사용하여 SNNs와 ANNs의 멤버십 프라이버시를 비교하였고, 데이터 증강(data augmentation)을 사용하여 SNNs의 MIAs 성능을 감소시킬 수 있음을 보여주었습니다.

- **Performance Highlights**: SNNs는 neuromorphic 데이터셋으로 훈련 시 ANNs보다 최대 10% 더 높은 공격 정확도를 보였습니다. 일반 데이터셋에 대해서는 SNNs의 취약성은 데이터셋에 따라 달라졌습니다. ANNs에서 SNNs로 변환 시 MIAs의 성능이 최대 11.5% 하락하고, 기본 데이터 증강 방법으로 MIAs 성능이 최대 25.7% 감소하는 결과를 보였습니다.



### Brain-JEPA: Brain Dynamics Foundation Model with Gradient Positioning and Spatiotemporal Masking (https://arxiv.org/abs/2409.19407)
Comments:
          The first two authors contributed equally. NeurIPS 2024 Spotlight

- **What's New**: Brain-JEPA는 Joint-Embedding Predictive Architecture (JEPA)를 기반으로 한 뇌 동역학의 기초 모델로, 인구 예측, 질병 진단/예후, 그리고 특성 예측에서 최첨단 성능을 자랑합니다.

- **Technical Details**: 이 모델은 두 가지 혁신적인 기법인 Brain Gradient Positioning과 Spatiotemporal Masking을 통합합니다. Brain Gradient Positioning은 뇌 기능 분할을 위한 기능적 좌표계를 도입하여 여러 지역의 위치 인코딩을 향상시킵니다. Spatiotemporal Masking은 fMRI 데이터의 독특한 특성에 맞추어 설계되어 이질적인 시계열 패치의 문제를 해결합니다.

- **Performance Highlights**: Brain-JEPA는 후속 실험에서 최첨단 결과를 기록하였으며, 다양한 인종 그룹에 걸쳐 일반화 능력이 우수합니다. 또한, 기존 대형 모델을 능가하여 뇌 활동 분석 영역에서 새로운 패러다임을 제시합니다.



### Efficient Federated Intrusion Detection in 5G ecosystem using optimized BERT-based mod (https://arxiv.org/abs/2409.19390)
- **What's New**: 이 논문은 5세대(5G) 네트워크의 보안 문제를 해결하기 위해 연합 학습(federated learning)과 대형 언어 모델(large language models, LLMs)을 활용한 강력한 침입 탐지 시스템(intrusion detection system, IDS)을 제안합니다. IDS는 BERT 모델을 기반으로 하며, 엣지 디바이스(edge devices)에서의 성능 최적화를 위해 수정되었습니다.

- **Technical Details**: 제안된 IDS는 전통적인 IDS 접근법인 서명 기반 탐지(signature-based detection)와 비정상 탐지(anomaly-based detection)를 통합합니다. 모델은 중앙집중식 및 연합 학습 맥락에서 실험을 실시하였으며, 비정상 데이터(xIID, non-IID) 상황에서도 보안 모델을 효과적으로 훈련시키고 데이터 개인 정보를 유지합니다.

- **Performance Highlights**: 중앙집중식 환경에서 모델은 97.79%의 추론 정확도(inference accuracy)를 달성했으며, 연합 학습에서는 다양한 환경에서 훈련이 이루어졌습니다. 모델 크기를 28.74% 줄이는 과정에서 정확도는 0.02% 감소하였으며, Raspberry Pi와 같은 자원 제한된 장치에서 0.45초의 추론 시간을 기록했습니다.



### Co-design of a novel CMOS highly parallel, low-power, multi-chip neural network accelerator (https://arxiv.org/abs/2409.19389)
Comments:
          neural network accelerator, low-power design, instruction set design, parallel processors, digital twin

- **What's New**: 이번 논문에서는 NV-1이라는 새로운 저전력 ASIC AI 프로세서를 소개합니다. 이 프로세서는 기존의 온보드 계산 방식 대신 클라우드 서버를 사용하는 보안 카메라, 센서 및 Siri와 같은 기기의 한계를 극복합니다.

- **Technical Details**: NV-1 프로세서는 비 전통적인 비 온-너먼(Non-von-Neumann) 아키텍처를 기반으로 하여 병렬 처리 속도를 10배 이상 가속화하고, 에너지 소비를 100배 이상 줄입니다. 병렬 결합 프로세서-메모리 유닛을 통해 매우 많은 독립적인 처리 스트림을 지원하며, 혁신적인 통신 프로토콜로 전력 사용을 최소화합니다.

- **Performance Highlights**: 제안된 하드웨어의 디지털 트윈이 초기부터 개발되어 기술적인 구현이 아키텍처 사양을 충족하는지 확인하였으며, 실제 하드웨어 테스트 데이터를 통해 예측된 성능 지표가 철저히 검증되었습니다. 현재 이 장치는 필드된 엣지 센서 애플리케이션에서 사용되고 있으며, 매우 저전력 고성능 ASIC 장치의 실제 적용 가능성을 보여주는 추가적인 원칙 입증이 진행되고 있습니다.



### Automated conjecturing in mathematics with \emph{TxGraffiti} (https://arxiv.org/abs/2409.19379)
- **What's New**: 이 논문에서는 2017년부터 개발된 데이터 기반의 계산 프로그램인 TxGraffiti를 소개합니다. TxGraffiti는 그래프 이론(graph theory) 분야에서 수학적 추론(conjecturing)을 자동화하는 데 중점을 두고 있으며, 이전에 존재했던 Graffiti 프로그램을 기반으로 설계되었습니다.

- **Technical Details**: TxGraffiti는 수학적 객체에 대한 관계를 찾기 위해 Dalmatian heuristic을 사용하여 중복되거나 전이적인 추측을 제거합니다. 이 시스템은 데이터 수집, 추측 생성을 포함하는 프로세스를 설명하며 사용자가 인터랙티브하게 추측을 탐색할 수 있는 웹 기반 인터페이스를 제공합니다.

- **Performance Highlights**: TxGraffiti는 그래프 이론뿐만 아니라 수학의 다른 분야에서도 적용 가능한 기술을 보여주며, 수많은 수학 출판물에 기여했습니다. 논문에서는 TxGraffiti가 생성한 여러 추측들을 소개하고, 이 추측들이 어떻게 새로운 연구 결과로 이어졌는지를 강조합니다.



### DOTA: Distributional Test-Time Adaptation of Vision-Language Models (https://arxiv.org/abs/2409.19375)
Comments:
          In submission

- **What's New**: 이 논문에서는 기존의 Training-Free Test-time Dynamic Adapter(TDA)의 한계를 극복하기 위해 DistributiOnal Test-time Adaptation(Dota)라는 새로운 방법을 제안합니다. Dota는 테스트 샘플의 분포를 지속적으로 추정하여 모델이 배포 환경에 적응할 수 있게 합니다.

- **Technical Details**: Dota는 Bayes' 정리를 기반으로 하여 추정한 분포를 사용하여 테스트 샘플의 후방 확률(test-time posterior probabilities)을 계산합니다. 여기에서 각 클래스의 임베딩 분포가 가우시안 분포를 따른다는 가정을 하며, 이는 TDA보다 약 20배 빠른 추론 속도를 제공합니다. 또한, Dota는 사람-주도 피드백(human-in-the-loop paradigm)을 통해 불확실한 샘플을 식별하고 적응하도록 돕습니다.

- **Performance Highlights**: 광범위한 데이터셋을 통한 실험 결과 Dota는 CLIP 모델이 지속적으로 학습할 수 있게 해주며, 기존의 최첨단 방법들에 비해 유의미한 성과 향상을 보였습니다.



### Mind the Gap: Promoting Missing Modality Brain Tumor Segmentation with Alignmen (https://arxiv.org/abs/2409.19366)
- **What's New**: 이 논문에서는 MRI (Magnetic Resonance Imaging) 모달리티가 누락된 상황에서 뇌 종양 세분화를 위한 새로운 정렬 패러다임을 제안합니다. 이 패러다임은 잠재적 특성을 명확한 분포 앵커에 정렬하여 양측 모두에서 성능을 향상시킵니다.

- **Technical Details**: 제안된 접근 방식에서는 Kullback-Leibler (KL) 손실을 최적화하고, 각각의 모달리티를 잠재 공간에 정렬합니다. 이 과정에서 최적의 잠재 공간 분포 P_{mix}를 사용하여 모달리티 간의 갭을 줄이고, modality-invariant (모달리티 불변) 특성을 학습합니다.

- **Performance Highlights**: 실험 결과, 제안된 정렬 패러다임은 Dice score에서 평균 1.75의 개선을 달성하여 최신 백본에서도 우수한 성능을 보여줍니다. 이는 학생 네트워크에서 누락된 모달리티를 효과적으로 처리하는 데 기여합니다.



### Conditional Image Synthesis with Diffusion Models: A Survey (https://arxiv.org/abs/2409.19365)
- **What's New**: 본 논문에서는 사용자가 제공한 조건을 기반으로 한 이미지를 생성하는 조건부 이미지 합성을 위한 새로운 접근 방법으로, diffusion 기반의 generative modeling을 중심으로 논의합니다. 특히, 기존의 접근 방식을 다양한 조건을 통합하는 두 가지 기본 구성 요소인 denoising network와 sampling process를 기준으로 분류하여 연구합니다.

- **Technical Details**: Diffusion 기반 모델링의 복잡성과 다양한 이미지 합성 작업, 다양한 conditioning mechanism들이 연구자들에게 도전 과제가 됩니다. 본 논문에서는 다양한 conditioning 접근 방식의 원리와 장점, 잠재적 도전에 관한 논의를 포함합니다. 또한, 기본 sampling 과정에서 여섯 가지 주요 조건 메커니즘을 요약하였습니다.

- **Performance Highlights**: 이번 조사는 텍스트-이미지 생성, 이미지 복원, 이미지 편집 등 다양한 조건부 이미지 합성 작업의 성능을 향상시키기 위해 필요한 훈련, 재사용 및 전문화 과정의 이점을 강조합니다. 엔드 투 엔드 접근 방식과 다양한 샘플링 기법의 조합을 통해 기존의 한계를 극복할 수 있는 가능성을 제시합니다.



### Learning Strategy Representation for Imitation Learning in Multi-Agent Games (https://arxiv.org/abs/2409.19363)
Comments:
          13 pages, 7 figures. arXiv admin note: substantial text overlap with arXiv:2402.18617

- **What's New**: 이번 연구에서는 다중 에이전트 게임에서 모방 학습을 향상시키기 위한 STRIL(Strategy Representation for Imitation Learning) 프레임워크를 도입하였습니다. STRIL은 각 시퀀스를 독특한 전략 표현으로 할당하고 이를 통해 비최적 데이터 샘플을 필터링하며, 기존 IL 알고리즘과 호환 가능합니다.

- **Technical Details**: STRIL은 반올림 훈련 조건을 갖춘 변분 순환 신경망(Partially-trainable-conditioned Variational Recurrent Neural Network, P-VRNN)을 사용하여 다중 에이전트 게임 시퀀스로부터 전략 표현을 효율적으로 추출합니다. 이 프레임워크는 무보상 데이터에서도 효과적으로 오프라인 경과를 평가하기 위해 무작위성 지표(Randomness Indicator, RI)와 착취 수준(Exploited Level, EL)을 정의합니다.

- **Performance Highlights**: STRIL은 두 플레이어의 퐁(Two-player Pong), 리밋 텍사스 홀덤(Limit Texas Hold'em), 그리고 연결 사각형(Connect Four)과 같은 경쟁적인 다중 에이전트 시나리오에서 기존의 IL 성능을 크게 향상시키는 데 성공했습니다.



### Solution of Multiview Egocentric Hand Tracking Challenge ECCV2024 (https://arxiv.org/abs/2409.19362)
Comments:
          Accepted in ECCV2024 workshop

- **What's New**: 이 연구에서는 VR(가상 현실) 상호작용을 위해 다중 뷰 입력 이미지와 카메라 외부 매개변수를 사용하여 손 모양과 자세를 추정하는 방법을 제시합니다. 또한 손 위치와 자세의 정확성을 향상시키기 위한 오프라인 Neural Smooth 후처리 방법을 도입했습니다.

- **Technical Details**: 연구된 아키텍처는 단일 뷰(single-view) 및 다중 뷰(multi-view) 입력을 모두 처리할 수 있으며, 3D 기능을 얻기 위해 특징 추출기와 특징 융합 모듈을 포함합니다. 데이터 증강(augmentation) 기법으로는 crop jittering 및 카메라 외부 매개변수에 대한 노이즈 증강을 사용했습니다. 모델은 Hiera-base 아키텍처를 기반으로 하여 8개의 NVIDIA 2080TI GPU에서 훈련되었습니다.

- **Performance Highlights**: 이 방법은 Umetrack 데이터셋에서 13.92mm MPJPE, HOT3D 데이터셋에서 21.66mm MPJPE의 결과를 달성하여 다양한 데이터셋에서 성능 향상을 입증했습니다.



### Visual Question Decomposition on Multimodal Large Language Models (https://arxiv.org/abs/2409.19339)
Comments:
          Accepted to EMNLP2024 Findings

- **What's New**: 본 논문은 멀티모달 대형 언어 모델(MLLMs)의 질문 분해(Question Decomposition) 능력을 탐구하고 있으며, 기존의 단일 모드 언어 모델과는 달리 MLLMs의 응답 품질을 향상시키는 새로운 데이터셋인 DecoVQA+를 제안하고 있습니다.

- **Technical Details**: 시스템적 평가 프레임워크를 도입하여 MLLMs의 분해된 하위 질문의 품질을 평가하고, 선택적 분해(selective decomposition)를 위한 훈련 목표를 포함하는 효율적인 파인튜닝(pipeline)을 제안하고 있습니다.

- **Performance Highlights**: 파인튜닝된 MLLMs는 VQA 벤치마크 데이터셋에서 하위 질문 품질의 현저한 개선과 선택적 질문 분해에서 더 높은 정확도를 달성했습니다.



### 3D-CT-GPT: Generating 3D Radiology Reports through Integration of Large Vision-Language Models (https://arxiv.org/abs/2409.19330)
- **What's New**: 본 논문에서는 3D CT 스캔으로부터 방사선 보고서를 생성하기 위해 Visual Question Answering (VQA) 기반의 새로운 의료 비주얼 언어 모델인 3D-CT-GPT를 소개합니다. 기존의 연구들은 2D 의료 이미지에 집중했지만, 본 연구는 3D 이미지에서의 자동 보고서 생성의 중요성을 강조하며 이를 해결하려는 시도를 보여줍니다.

- **Technical Details**: 3D-CT-GPT는 CT ViT, 3D Average Pooling, 그리고 프로젝션 레이어를 결합하여 3D CT 스캔으로부터 직접적이고 정확한 방사선 보고서를 생성합니다. 모델은 공공 데이터셋에서의 사전 훈련과 소규모 개인 데이터셋에서의 미세 조정을 통해 훈련 전략을 최적화하여 데이터 요구사항을 줄이고 뛰어난 성능을 유지합니다.

- **Performance Highlights**: 3D-CT-GPT는 기존의 3D 방사선 보고서 생성 방법들에 비해 보고서의 정확성과 품질 측면에서 현저하게 우수한 성능을 보입니다. 이 모델은 임상 방사선 보고서 생성에 대한 강력한 솔루션을 제시하며, 진단 정확도와 보고서의 일관성을 높이는 데 기여합니다.



### A Generalized Model for Multidimensional Intransitivity (https://arxiv.org/abs/2409.19325)
Comments:
          13 pages, 1 figure

- **What's New**: 이 논문에서는 비가환성(intransitivity) 문제를 해결하기 위해, 플레이어의 d차원 표현(d>1)과 데이터셋 특화 거리 공간(metric space)을 공동으로 학습하는 확률적 모델을 제안하였습니다. 이는 비가환적 표현 학습에 있어 이전의 모델로 수렴할 수 있는 흥미로운 결과를 보여줍니다.

- **Technical Details**: 제안된 모델은 각 플레이어의 d차원 표현을 학습하고, 해당 표현 공간에서 적절한 거리 형식을 체계적으로 캡처합니다. 추가적인 제약 조건을 통해 이 모델은 기존의 비가환적 표현 학습에 사용되는 모델로 수렴할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 사회적 선택(social choice), 선거(election), 온라인 게임 데이터셋에 대한 예측 성능이 향상되었습니다. 여러 경쟁 모델들에 비해 예측 정확도에서 우수한 성과를 보였습니다.



### Scalable Cloud-Native Pipeline for Efficient 3D Model Reconstruction from Monocular Smartphone Images (https://arxiv.org/abs/2409.19322)
Comments:
          Preprint

- **What's New**: 본 논문은 스마트폰 카메라로 캡처된 단안 2D 이미지에서 자동으로 3D 모델을 재구성할 수 있는 클라우드 네이티브 파이프라인을 제안합니다. 이 파이프라인은 Industry 4.0 기준에 부합하며, 교육 과정에서 직원의 전문성을 향상시킬 수 있는 디지털 트윈(digital twin) 모델을 생성할 수 있습니다.

- **Technical Details**: 제안된 파이프라인은 NVIDIA Research Labs에서 개발한 머신 러닝 모델과 Google의 ARCore 프레임워크에 기반한 고유한 포즈 보상 구성이 포함된 사용자 정의 포즈 레코더를 활용합니다. 이 시스템은 마이크로서비스 아키텍처(microservices architecture)를 채택하여 각 모듈이 독립적으로 작동할 수 있도록 설계되었습니다.

- **Performance Highlights**: 이 연구의 결과로, 재사용 가능한 3D 모델을 생성하고, 향상된 재료와 텍스처를 포함할 수 있으며, 외부 3D 모델링 소프트웨어 또는 3D 엔진에서 내보내기 및 사용자 정의가 가능합니다. 또한, 이 과정은 AR(증강 현실) 기능을 통해 데이터 수집 작업을 개선합니다.



### Analog In-Memory Computing Attention Mechanism for Fast and Energy-Efficient Large Language Models (https://arxiv.org/abs/2409.19315)
Comments:
          25 pages, 6 figures, 1 table

- **What's New**: 이 연구는 아날로그 메모리 컴퓨팅(Analog In-Memory Computing)을 기반으로 한 자가 주의 메커니즘(self-attention mechanism)의 빠르고 에너지 효율적인 하드웨어 구현을 제안합니다. 폭넓은 메모리 캐시로 인한 지연 및 에너지 병목 현상을 해결하기 위해 방법론을 개발하였습니다.

- **Technical Details**: 제안된 구조는 게인 셀 메모리(gain cell memories)를 이용하여 에너지를 절약하고 처리 속도를 향상시키며, Sliding Window Attention 기법을 구현하여 메모리 소모를 줄입니다. 자가 주의 계산을 아날로그 도메인에서 수행하여 디지털 변환 과정에서 발생할 수 있는 전력 소비를 최소화합니다. 또, 사전 훈련된 가중치를 하드웨어 비이상성(non-idealities)에 맞춰 조정하기 위한 새로운 알고리즘을 도입했습니다.

- **Performance Highlights**: 우리의 하드웨어 디자인은 GPU 대비 주의(latency) 시간을 두 개의 오더만큼 줄이는 성과를 보여주었으며, 에너지 소비는 최대 다섯 오더까지 감소했습니다. NLP 성능은 극소량의 훈련 과정으로도 ChatGPT-2에 비견될 만큼 향상되었습니다.



### Model X-Ray: Detection of Hidden Malware in AI Model Weights using Few Shot Learning (https://arxiv.org/abs/2409.19310)
- **What's New**: 본 연구는 AI 모델의 보안성을 향상시키기 위한 새로운 단계로, 이미지 분야에서 잘 연구된 few-shot 학습 기법을 활용하여 AI 모델의 스테가노그래피 공격을 탐지하기 위한 효과적인 방법론을 제시합니다.

- **Technical Details**: 우리는 AI 모델의 가중치로부터 이미지 표현(image representation)을 생성하는 방법을 소개하며, 이를 통해 LSB(Least Significant Bit) 스테가노그래피 공격 탐지 기술을 개발합니다. 우리의 연구는 이전 연구들보다 훨씬 적은 훈련 데이터(최대 6개 모델)로 효과적인 탐지를 가능하게 합니다. 또한, ResNet 및 Densenet과 같은 대형 CNN(model architectures) 아키텍처에서 성공적으로 작동합니다.

- **Performance Highlights**: 제안하는 방법은 최대 25%의 임베딩 비율(embedding rate)으로 피해 공격을 탐지할 수 있으며, 경우에 따라 6%까지도 탐지 가능합니다. 이는 이전 연구가 100%-50%의 임베딩 비율에서만 성공을 보인 것과 대조적입니다. 우리의 훈련된 모델은 다양한 특성을 가진 모델에서도 공격을 탐지할 수 있는 일반성을 보여주며, 이는 기존 연구들과의 주요 차별점입니다.



### Designing Domain-Specific Large Language Models: The Critical Role of Fine-Tuning in Public Opinion Simulation (https://arxiv.org/abs/2409.19308)
- **What's New**: 이 논문은 환경 정책에 대한 의견을 시뮬레이션하기 위해 대형 언어 모델(LLMs)을 세부 조정하는 새로운 접근 방식을 제안합니다. 이를 위해 영국 가정 종단 연구(UK Household Longitudinal Study)의 데이터를 활용합니다.

- **Technical Details**: 대형 언어 모델의 세부 조정(fine-tuning)을 통해 나이(age), 소득(income), 교육(education), 지역(region)과 같은 사회 인구학적 요인에 따라 모델을 조정하여 의견 생성의 정확성을 개선합니다. 다양한 합성 프로필(synthetic profiles)을 모방하여, 세부 조정된 모델이 인구 집단 간의 미세한 차이를 보다 효과적으로 캡처합니다.

- **Performance Highlights**: Chi-Squared, Cosine Similarity, Jaccard Index, KL-divergence와 같은 메트릭을 사용하여 합성 및 실제 의견 데이터 간의 강한 정렬을 보여줍니다. 이러한 결과는 더 정확하고 윤리적인 정책 시뮬레이션을 위한 LLM의 사회적 맥락에 대한 맞춤화의 중요성을 강조합니다.



### CausalVE: Face Video Privacy Encryption via Causal Video Prediction (https://arxiv.org/abs/2409.19306)
Comments:
          Submitted to ICLR 2025

- **What's New**: 이 논문에서는 얼굴 비디오와 개인 정보 보호의 혁신적인 상호작용 프레임워크인 CausalVE를 소개합니다. 이 프레임워크는 동적 인과 추론(dynamic causal reasoning)과 가역 신경망(reversible neural networks)을 통합하여 원래 비디오 내용을 생성된 덮개 얼굴 비디오와 무결하게 혼합합니다.

- **Technical Details**: CausalVE 프레임워크는 얼굴 교체(face swapping)를 위해 확산 모델(difussion model)을 활용하고, 비밀 비디오의 음성 시퀀스 특성과 시공간 시퀀스(spatiotemporal sequence) 특성을 사용하여 동적 비디오 추론(dynamic video inference) 및 예측을 수행합니다. 이 시스템은 원래 비디오를 가상의 비디오 안에 숨기고, 키를 사용해 정확히 복구할 수 있는 방법을 제공합니다.

- **Performance Highlights**: CausalVE는 공공 비디오 전파에서 뛰어난 보안을 제공하며, 정성적(qualitative) 및 정량적(quantitative) 관점 및 시각적(visual) 관점 모두에서 최신 기법(state-of-the-art methods)보다 우수한 성능을 입증하였습니다.



### Privacy Attack in Federated Learning is Not Easy: An Experimental Study (https://arxiv.org/abs/2409.19301)
- **What's New**: 최근 연구에 따르면, Federated Learning (FL) 환경에서도 개인 정보 유출에 대한 공격이 가능하며, FL의 기존 개인 정보 보호 알고리즘이 효과적이지 않음을 보여줍니다.

- **Technical Details**: FL은 다양한 클라이언트의 로컬 데이터를 합치지 않고도 글로벌 모델을 훈련할 수 있지만, 기존 공격 알고리즘은 단일 그래디언트를 기반으로 private data 복구를 시도합니다. 최근의 실험 결과는 이러한 공격이 실제 FL 환경에선 잘 작동하지 않음을 드러냅니다.

- **Performance Highlights**: 대부분의 공격 알고리즘이 평균화된 그래디언트에서 높은 품질의 더미 이미지를 성공적으로 복원할 수 있으나, 복잡한 FL 환경에서 성능 저하를 겪습니다. Robbing the Fed (RTF) 공격이 가장 높은 성능을 보였으나, 훈련 성능이 크게 저하되는 단점이 있습니다.



### Sustaining model performance for covid-19 detection from dynamic audio data: Development and evaluation of a comprehensive drift-adaptive framework (https://arxiv.org/abs/2409.19300)
- **What's New**: COVID-19 진단을 위한 새로운 프레임워크가 소개되었습니다. 본 연구는 동적인 오디오 데이터에서 COVID-19 감지 모델의 성능 변동성을 줄이기 위한 적응 메커니즘을 도입하여 모델 드리프트(model drift)를 모니터링합니다.

- **Technical Details**: 본 연구는 두 개의 크라우드 소싱(Crowd-sourced) COVID-19 오디오 데이터셋인 COVID-19 Sounds와 COSWARA를 사용하였습니다. 각 데이터셋은 개발(development) 및 포스트 개발(post-development) 기간으로 나뉘어 CNN(convolutional neural networks) 모델이 훈련되고 평가되었습니다. 데이터 배포간 변화를 감지하기 위해 최대 평균 불일치(Maximum Mean Discrepancy, MMD)를 사용하였고, 드리프트가 감지되면 모델 재훈련이 트리거되었습니다. 적응 접근법으로 비지도 도메인 적응(Unsupervised Domain Adaptation, UDA)과 능동 학습(Active Learning, AL)을 비교했습니다.

- **Performance Highlights**: UDA는 COVID-19 Sounds 및 COSWARA 데이터셋에서 각각 최대 22% 및 24% 균형 정확도(balanced accuracy)를 개선했으며, AL은 각각 최대 30% 및 60%의 더 큰 개선 효과를 나타냈습니다.



### CLIP-MoE: Towards Building Mixture of Experts for CLIP with Diversified Multiplet Upcycling (https://arxiv.org/abs/2409.19291)
- **What's New**: 최근 CLIP(Contrastive Language-Image Pre-training) 모델의 정보 손실 문제를 해결하기 위한 새로운 전략, Diversified Multiplet Upcycling (DMU)을 제안합니다. 이 방법은 다양한 특성 공간을 캡처하는 여러 CLIP 모델을 효율적으로 미세 조정하여, 기존 프리트레인된 CLIP 체크포인트를 활용하면서 모델 용량을 확장합니다.

- **Technical Details**: DMU는 CLIP의 Feed-Forward Network (FFN)를 제외한 파라미터를 공유하면서 세밀한 정보 캡처를 위한 여러 CLIP 모델을 생성합니다. 이 모델들은 Multistage Contrastive Learning (MCL)을 통해 다단계 클러스터링 및 미세 조정 과정을 거쳐 서로 다른 입력 정보의 측면을 캡처하는 FFN 전문가로 변환됩니다. 최종적으로 CLIP-MoE는 모든 전문가를 극대화하여 집합적이고 유용한 정보를 캡처하도록 합니다.

- **Performance Highlights**: DMU를 통해 초기화된 CLIP-MoE는 기존 OpenAI CLIP 모델에 비해 약 20%의 성능 향상을 보여주며, 본래 CLIP의 비전 인코더 역할을 대체할 수 있습니다. 다양한 다운스트림 작업에서 최고의 성과를 기록했으며, 적은 추가 훈련 비용으로 기존 CLIP의 성능을 크게 향상시켰습니다.



### Distributed Optimization via Energy Conservation Laws in Dilated Coordinates (https://arxiv.org/abs/2409.19279)
Comments:
          10 pages; (Near) optimal convergence rate

- **What's New**: 본 논문은 여러 에이전트가 개별 데이터를 공유하는 시스템에서 분산 최적화(distributed optimization) 문제를 최적화하는 새로운 방법론을 제시합니다. 특히 연속 시간 동역학 시스템을 포괄적으로 분석할 수 있는 에너지 보존(energy conservation) 접근법을 소개합니다.

- **Technical Details**: 이 방법은 원래 좌표계에서 직접적으로 동역학을 분석하는 대신, 확대된 좌표계에서 물리적인 에너지와 같은 보존량을 설정하여, 수렴 속도를 시간 왜곡 인수(inverse time-dilation factor)로 명시적으로 표현합니다. 제안된 이론을 바탕으로 새로운 2차 분산 가속 경량 흐름(second-order distributed accelerated gradient flow)을 정의하였고, 이에 대한 수렴 속도는 O(1/t^{2-ε})로 나타났습니다. 또한 반 이차 심플렉틱 오일러 이산화 방식을 사용하여 O(1/k^{2-ε})의 수렴 속도를 가진 알고리즘을 도출하였습니다.

- **Performance Highlights**: 이 알고리즘은 매끄러운 볼록 최적화를 위한 모든 분산 최적화 알고리즘 중에서 최고의 수렴 속도를 제공합니다. 실제 대규모 문제에 대해서 여러 최신 상태의 분산 최적화 알고리즘과의 비교를 통해 가속화된 수렴 행동을 검증하였습니다.



### OpenSep: Leveraging Large Language Models with Textual Inversion for Open World Audio Separation (https://arxiv.org/abs/2409.19270)
Comments:
          Accepted in EMNLP 2024 Main

- **What's New**: OpenSep은 기존의 오디오 분리 모델의 한계를 극복하고, 자동화된 오디오 분리를 위한 혁신적인 프레임워크입니다. 이 프레임워크는 대규모 언어 모델(LLMs)을 활용하여 수작업 interven의 필요 없이 혼합된 오디오의 다양한 소스를 효과적으로 분리합니다.

- **Technical Details**: OpenSep는 텍스트 역전 변환(textual inversion) 방식을 사용하여 오디오 혼합물로부터 캡션을 생성하고, 이를 통해 존재하는 소리를 명확히 분석합니다. 소리 소스의 세부 속성을 추출하기 위해 few-shot LLM prompting을 사용하며, 새로운 혼합물의 분리에 용이하게 합니다. 또한, mix-and-separate 학습 프레임워크의 다중 레벨 확장을 도입하여 단일 소리와 혼합 소리를 동시에 분리함으로써 모달리티 정렬(modality alignment)을 강화합니다.

- **Performance Highlights**: OpenSep은 MUSIC 및 VGGSound 데이터셋에서 각각 64% 및 180%의 SDR(Signal-to-Distortion Ratio) 개선을 달성하며, 기존 최첨단(SOTA) 방법들을 능가하는 성능을 보여줍니다. 이러한 실험 결과는 OpenSep이 새로운, 보지 못한 소스 및 변동 소스에 대한 정확한 분리를 제공한다는 것을 입증합니다.



### VecLSTM: Trajectory Data Processing and Management for Activity Recognition through LSTM Vectorization and Database Integration (https://arxiv.org/abs/2409.19258)
Comments:
          10 pages, 5 figures

- **What's New**: 이 연구에서는 LSTM(Long Short-Term Memory) 기반 신경망의 성능과 효율성을 향상시키기 위한 새로운 프레임워크인 VecLSTM을 제안합니다. VecLSTM은 벡터화(vectorization) 층을 통합하여 입력 시퀀스를 보다 효율적으로 처리합니다.

- **Technical Details**: VecLSTM 방법론은 전통적인 LSTM 모델과 달리 데이터 전처리 과정에서 지리적 좌표를 2D 그리드 표현으로 변환하는 벡터화 과정을 도입합니다. 이후 CNN(Convolutional Neural Network)을 활용하여 특징을 추출하고, CNN과 LSTM 모델의 출력을 조합하여 지역적 및 전역적 공간 의존성을 캡처합니다.

- **Performance Highlights**: 실험 결과, VecLSTM은 1,467,652개의 샘플로 구성된 데이터셋에서 기존 LSTM 모델보다 우수한 정확도(85.57%의 검증 정확도, 85.47%의 테스트 정확도, 0.86의 가중치 F1 점수)를 보여주며, 모델 훈련 시간을 26.2% 단축시켰습니다.



### DENEB: A Hallucination-Robust Automatic Evaluation Metric for Image Captioning (https://arxiv.org/abs/2409.19255)
Comments:
          ACCV 2024

- **What's New**: DENEB를 도입하여 이미지 캡션 생성에서의 환각(hallucination)에 대한 강건성을 향상시킨 새로운 자동 평가 지표를 제안

- **Technical Details**: DENEB는 Sim-Vec Transformer를 통합하여 동시에 여러 참조 캡션을 처리하며, Nebula 데이터셋에서 32,978개의 이미지와 805명의 주석가의 인간 평가를 통해 훈련됨

- **Performance Highlights**: DENEB는 FOIL, Composite, Flickr8K-Expert, Flickr8K-CF, PASCAL-50S 및 Nebula 데이터셋에서 기존 LLM-free 지표 중 최첨단 성과 달성



### Edit-Constrained Decoding for Sentence Simplification (https://arxiv.org/abs/2409.19247)
Comments:
          Accepted by EMNLP2024-Findings

- **What's New**: 본 연구에서는 문장 단순화를 위한 lexically constrained decoding에 기반한 edit operation을 제안합니다. 이전 연구들은 lexically constrained decoding의 효과를 확인했지만, 이러한 제약 조건들은 느슨하여 최적이 아닌 결과를 초래할 수 있었습니다. 우리는 문장 단순화에서 수행되는 edit operation을 복제하고, stricter satisfaction conditions를 정의하는 제약 조건을 설계하여 이 문제를 해결합니다.

- **Technical Details**: 문장 단순화에서는 세 가지 edit operation (insertion, deletion, substitution)을 수행하여 문장을 재작성합니다. 우리는 NeuroLogic Decoding을 확장하여 이러한 edit operation을 constrained decoding을 통해 실행합니다. 이 방법은 generation likelihood와 constraint satisfaction을 최대화하는 가설을 찾기 위해 beam search를 활용합니다.

- **Performance Highlights**: 제안된 방법은 Turk, ASSET, AutoMeTS와 같은 세 가지 영문 단순화 코퍼스를 통해 이전 연구보다 일관되게 더 우수한 성능을 보였습니다. 이는 참조에서 추출된 oracle 제약 조건을 사용했을 때 확인되었으며, 단순 모델로 예측된 제약 조건에서도 일관된 효능을 나타냈습니다.



### The Price of Pessimism for Automated Defens (https://arxiv.org/abs/2409.19237)
Comments:
          Accepted to GameSec 2024

- **What's New**: 이 논문은 사이버 보안 분야에서 최악의 경우를 대비하는 것이 필연적으로 최적의 결과를 가져오지 않음을 보여줍니다. '최악의 경우'에 대한 준비가 오히려 학습 에이전트에 부정적인 영향을 미칠 수 있음을 설명합니다.

- **Technical Details**: 이 연구는 확률적 Bayesian 게임(Stochastic Bayesian Games)의 관점에서 공격자 모델링 가정을 탐구합니다. 다양한 공격 지식 모델링 가정에 기반하여 사이버 보안 실무자에게 모델의 유용성을 분석하며, 공격자의 상태 및 방어자의 숨겨진 정보에 대한 최적화가 방어자에게 부담을 줄 수 있음을 발견하였습니다.

- **Performance Highlights**: 강화 학습 에이전트가 최악의 경우를 대상으로 최적화할 경우 정책의 수렴이 비효율적으로 이루어짐을 보여주었으며, 학습하는 공격 에이전트를 상대로 훈련된 방어 에이전트는 알고리즘 공격자에 대해서도 훌륭한 성능을 발휘한다는 점이 강조됩니다.



### Double Actor-Critic with TD Error-Driven Regularization in Reinforcement Learning (https://arxiv.org/abs/2409.19231)
- **What's New**: TDDR(Temporal Difference error-Driven Regularization) 알고리즘이 도입되었습니다. 이 알고리즘은 double actor-critic 구조를 기반으로 하며, 기존 알고리즘과 비교해 추가적인 hyperparameter 없이 뛰어난 성능을 보입니다.

- **Technical Details**: TDDR는 double actors와 critics를 사용하여 Q-value를 개선합니다. 이 알고리즘은 clipped double Q-learning (CDQ) 메커니즘을 도입하여 각 critic 업데이트에 적절한 Q-value를 선택하는 데 temporal difference (TD) 오류를 활용합니다. 또한, 학습 과정에서 랜덤 및 동시 업데이트에서도 수렴성을 보장합니다.

- **Performance Highlights**: TDDR는 MuJoCo 및 Box2D와 같은 연속 제어 작업에서 benchmark 알고리즘 대비 우수한 경쟁력을 발휘하며, 복잡한 하이퍼파라미터 조정 없이도 뛰어난 가치 추정을 제공합니다.



### Learning to Bridge the Gap: Efficient Novelty Recovery with Planning and Reinforcement Learning (https://arxiv.org/abs/2409.19226)
- **What's New**: 본 연구에서는 Reinforcement Learning(RL) 기반의 'bridge policy'를 도입하여 환경의 변화에 적응하는 자율 로봇을 위한 접근법을 제안합니다. 이는 플래너(planner)의 지식을 활용하여 복잡한 장기 결정을 내리는 문제에 대한 효율적인 해법을 제시합니다.

- **Technical Details**: 연구에서는 'CallPlanner'라는 특수한 액션을 통해 RL 문제를 구성하고, 고유의 상태에서 플래너에게 제어를 반환하는 방식으로 작동합니다. 이렇게 함으로써 RL 정책은 장기적 목표를 달성하기 위해 플래너에 의해 제공되는 계획을 따르는 상태 집합을 학습하게 됩니다.

- **Performance Highlights**: 3개의 다양한 복잡성을 가진 시뮬레이션 환경에서 실험을 진행한 결과, 제안된 bridge policy는 기존의 여러 기준선(Baselines)보다 효율적으로 새로움(novelty)에 적응하는 정책을 학습하였음을 보여주었습니다. 또한, 학습된 bridge policy는 플래너와 결합되어 더 복잡한 작업을 해결하는 데에도 일반화 가능성을 나타냈습니다.



### Semi-Supervised Bone Marrow Lesion Detection from Knee MRI Segmentation Using Mask Inpainting Models (https://arxiv.org/abs/2409.19185)
Comments:
          5 pages, 3 figures, submitted to SPIE Conference on Image Processing

- **What's New**: 본 논문에서는 고해상도 무릎 MRI에서 BML(골수 병변)의 식별을 위한 반지도 학습(local anomaly detection) 방법을 제안합니다. 이는 기존의 global anomaly detection 방법에 비해 성능을 향상시키는 것을 목표로 합니다.

- **Technical Details**: 제안된 방법은 3D 대퇴골 세분화 모델, 대규모 마스크 인페인팅(mask inpainting) 모델 및 일련의 후처리(post-processing) 기술을 통합합니다. 이를 통해 다양한 해상도의 MRI 이미지에서 BML을 효과적으로 감지합니다.

- **Performance Highlights**: 해당 방법은 다중 해상도 지식 증류(multiresolution knowledge distillation) 방법과 비교했을 때 Dice score, Intersection over Union (IoU) 및 픽셀 수준의 민감도, 특이성 및 정확성에서 우수한 성능을 보였습니다. 특히, 고해상도 이미지에서 세분화 성능이 2배 이상 향상되었습니다.



### Artificial-Intelligence Generated Code Considered Harmful: A Road Map for Secure and High-Quality Code Generation (https://arxiv.org/abs/2409.19182)
- **What's New**: 이 연구는 LLM(대규모 언어 모델)이 생성한 코드의 보안 및 품질을 평가하여, 인간이 작성한 코드와의 차이를 분석했습니다. LLM이 생성한 코드의 보안 문제와 품질 저하가 강조되었습니다.

- **Technical Details**: 이 연구에서 수행된 테스트는 유닛 테스트(unit testing), 퍼징(fuzzing), 정적 분석(static analysis)을 포함하며, 복잡성(complexity)과 크기(size)를 측정했습니다. 연구는 C 프로그래밍 언어를 사용하여 코드를 비교하였으며, 200개 이상의 LeetCode 프래그래밍 테스트와 알고리즘 간단화 사례를 분석했습니다.

- **Performance Highlights**: LLM 생성 코드는 복잡성이 1.19배 높고, 보안 문제 발생률은 인간 코드보다 11.2% 더 높았습니다. LLM이 재생성한 코드에서도 새로운 버그와 함께 코드의 복잡성 증가가 발견되었습니다.



### HM3: Heterogeneous Multi-Class Model Merging (https://arxiv.org/abs/2409.19173)
- **What's New**: 본 논문은 다양한 기계 학습 모델들이 결합하여 하나의 다기능 모델로 통합할 수 있는 방법인 Heterogeneous Multi-Class Model Merging (HM3)을 제안합니다. 이 접근 방식은 각각의 모델의 정확도를 손실 없이 통합할 수 있도록 하며, 기존의 복잡한 학습 과정 없이 가능합니다.

- **Technical Details**: HM3 기법은 서로 다른 레이블 공간을 가진 다중 클래스 분류기를 결합하는 방법론입니다. 이 방법은 모델 항목의 가중치를 결합하여 최적의 성능을 내는 동시에, 모델 간의 학습이나 추가 훈련 없이 CPU에서 빠르게 실행될 수 있습니다. 실험 결과, BERT 기반의 guard 모델들을 결합하여 F1-score를 평균적으로 높이고, 추론 시간을 44%까지 단축하는 결과를 보였습니다.

- **Performance Highlights**: Heterogeneous Multi-Class Model Merging 기법을 통해 기존의 모델들보다 평균 F1-score가 개선된 BERT 기반 guard 모델을 확인했으며, 잔여 부정적 성과를 가진 hate speech 분류기가 self-merging 과정을 통해 성능이 향상되었음을 보여주었습니다.



### Multimodal Pragmatic Jailbreak on Text-to-image Models (https://arxiv.org/abs/2409.19149)
- **What's New**: 이번 연구는 텍스트-이미지(T2I) 모델에 대한 새로운 유형의 jailbreak을 소개하며, 둘 이상의 안전한 요소가 결합해 위험한 콘텐츠를 생성하는 현상을 탐구합니다.

- **Technical Details**: 제안된 Multimodal Pragmatic Unsafe Prompts (MPUP) 데이터셋은 1,200개의 위험한 프롬프트로 구성되어 있으며, 아홉 개의 대표적인 T2I 모델을 벤치마킹합니다. 실험 결과, 모든 모델이 8%에서 74%의 비율로 위험한 콘텐츠를 생성하는 취약성을 보였습니다.

- **Performance Highlights**: 현재의 안전 필터가 이러한 새로운 jailbreak에 대해 효과적이지 않음을 밝혀냈으며, 모델들이 제작한 위험한 콘텐츠의 복잡성을 감지하는 데 한계가 있음을 강조합니다.



### Bound Tightening Network for Robust Crowd Counting (https://arxiv.org/abs/2409.19146)
Comments:
          This work was done 2 years ago

- **What's New**: 본 논문에서는 Robust Crowd Counting을 위한 새로운 Bound Tightening Network (BTN)을 제안합니다. BTN은 기본 모델, 스무스 정규화 모듈 및 인증 경계 모듈의 세 부분으로 구성됩니다. 이 네트워크는 모델의 추정에 대한 이론적인 보증을 제공하고, 다양한 표준 데이터셋에서 모델의 강건성을 향상시킵니다.

- **Technical Details**: BTN은 3개의 모듈로 구성됩니다: 1) Smooth Regularization Module - 기본 모델의 레이어 가중치를 활용하여 정규화 항을 도입합니다; 2) Certify Bound Module - 입력 이미지에 대해 적대적 섭동을 수동으로 도입하고 초기 경계 세트를 구성하여 모델 레이어를 통해 이 경계를 전파합니다. 이를 통해 훈련 루프를 안내하여 최종 성능을 향상시킵니다.

- **Performance Highlights**: 여러 벤치마크 데이터셋을 활용한 실험에서 BTN의 효과성과 효율성을 입증하였습니다. BTN은 적대적 섭동에 대해 robust 하며 모델 예측의 매우 타이트한 경계를 제공합니다.



### TTT4Rec: A Test-Time Training Approach for Rapid Adaption in Sequential Recommendation (https://arxiv.org/abs/2409.19142)
- **What's New**: 본 논문에서는 사용자 상호작용을 실시간으로 반영하여 동적으로 모델을 업데이트할 수 있는 Test-Time Training (TTT) 기반의 순차 추천 프레임워크인 TTT4Rec을 제안합니다.

- **Technical Details**: TTT4Rec은 TTT의 구조를 통해 두 개의 루프를 사용하여 모델 파라미터를 지속적으로 업데이트합니다. 외부 루프는 감독 학습(supervised learning)에 집중하고, 내부 루프는 자가 감독 학습(self-supervised learning)을 이용하여 훈련 및 추론 중 모델의 hidden state를 업데이트합니다. 주요 구성 요소로는 아이템 정보를 높은 차원 벡터로 인코딩하는 Embedding Layer, 입력 시퀀스의 특징을 캡처하는 TTT 기반의 Residual Blocks, 업데이트된 hidden state를 바탕으로 추천을 생성하는 Prediction Layer가 있습니다.

- **Performance Highlights**: TTT4Rec은 세 가지 주요 추천 데이터셋에서 평가되었으며, 기존의 최첨단 모델보다 더 나은 성능을 나타냈습니다. 특히 훈련 데이터가 제한되거나 사용자 행동이 급변하는 경우 효율적으로 동작하며, 실시간 사용자 상호작용에 적응하여 정확한 추천이 가능함을 보였습니다.



### Sequencing the Neurome: Towards Scalable Exact Parameter Reconstruction of Black-Box Neural Networks (https://arxiv.org/abs/2409.19138)
- **What's New**: 이 논문에서는 쿼리 접근만으로 신경망(neural network)의 정확한 매개변수를 추론하는 문제를 다룹니다. 이는 NP-Hard 문제로 몇 가지 기존의 알고리즘이 존재하지만 실용적인 방법이 부족합니다. 저자들은 무작위 초기화 및 1차 최적화(first order optimization)를 통해 매개변수 공간을 줄이고, 최대한 정보가 풍부한 쿼리 생성을 통해 비선형 관계를 효과적으로 풀어나가는 새로운 접근 방식을 제시했습니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 인사이트를 바탕으로 합니다. 첫 번째는 대다수의 신경망이 무작위 초기화 및 최적화를 통해 생성된다는 점이며, 이는 실제적인 매개변수 공간을 대폭 줄입니다. 두 번째는 효율적으로 비선형 관계를 풀어내기 위한 새로운 쿼리 생성 알고리즘을 제시했다는 점입니다. 이 방법을 통해 150만 개 이상의 매개변수를 가진 숨겨진 신경망과 7층 깊이의 네트워크를 재구성함으로써 높은 정확도를 보였습니다. 또한, 평가를 위해 세 가지 유형의 동형(isomorphism)을 고려했습니다: 순열(permutations), 스케일링(scaling), 극성(polarity).

- **Performance Highlights**: 그 결과 이들은 가장 큰 및 깊은 네트워크 재구성을 성공적으로 수행하였고, 최대 매개변수 차이는 0.0001 미만으로 우수한 성능을 자랑합니다. 평가 결과는 다양한 아키텍처와 데이터셋, 학습 절차를 통해 복원 방법의 견고함과 확장성을 입증하였습니다. 다양한 최적화 방법과 에포크 수를 다양화하여 실험을 진행한 결과, 저자들은 높은 샘플 효율성을 기록하며 기존 방법들과의 비교에서도 경쟁력을 보였습니다.



### Kinematic Detection of Anomalies in Human Trajectory Data (https://arxiv.org/abs/2409.19136)
- **What's New**: 이 논문은 기존의 위치 기반 연구와는 달리, 개인이 이동하는 방식, 즉 운동 패턴(kinematic features)에 중점을 두고 이를 활용하여 개인 식별(individual identification) 및 이상 탐지(anomaly detection)에 대한 가능성을 탐구한다.

- **Technical Details**: 우리는 Geolife 데이터셋을 사용하여 개별 사용자의 kinematic profile을 분석하였다. 이 kinematic profile은 개인의 이동 방식과 관련된 특징을 포함한다. 10개의 kinematic feature을 추출하고 이를 통해 각 트립(trip)을 분석한다. 높은 속도와 비현실적인 이동 패턴과 같은 이상치를 제거하여 정확한 데이터 처리를 보장한다.

- **Performance Highlights**: 단순한 kinematic feature을 표준 분류(classification) 및 이상 탐지 알고리즘에 적용함으로써 개인 식별 및 이상 탐지의 성능을 크게 향상시킬 수 있음을 실험적으로 보여주었다.



### Multi-modal Cross-domain Self-supervised Pre-training for fMRI and EEG Fusion (https://arxiv.org/abs/2409.19130)
- **What's New**: 이번 연구에서는 fMRI(기능적 자기공명영상)와 EEG(전기뇌파) 데이터를 통합하여 뇌 질환의 패러다임을 새롭게 제시하는 Multi-modal Cross-domain Self-supervised Pre-training Model (MCSP)을 개발하였습니다. 이 모델은 여러 도메인에서의 상호작용을 완전히 포착하기 위해 Self-supervised learning을 활용하며, 다양한 도메인 간의 시너지 정보를 극대화하는 방법을 제안합니다.

- **Technical Details**: MCSP는 Cross-domain self-supervised loss (CD-SSL)와 Cross-modal self-supervised loss (CM-SSL) 두 가지 손실 함수를 도입하여, fMRI와 EEG의 서로 다른 측면을 효과적으로 통합합니다. CD-SSL은 도메인 특화 데이터 증강 및 Contrastive learning 기술을 적용하여 도메인 간 유사성을 극대화합니다. CM-SSL은 fMRI와 EEG 간의 보완적 특성을 활용하여 서로 풍부한 정보를 증류합니다.

- **Performance Highlights**: 실험 결과, MCSP 모델은 다양한 분류 작업에서 우수한 성능을 입증하였으며, 정신 질환 연구의 맥락에서 fMRI와 EEG의 융합이 가져오는 신규 통찰력을 기반으로 깊은 학습 설계의 중요성을 강조합니다. 이를 통해 위상별, 시간별, 주파수별 특성을 모두 극대화하는 멀티모달 신경영상 분석의 잠재력을 확인할 수 있었습니다.



### Secure Multiparty Generative AI (https://arxiv.org/abs/2409.19120)
- **What's New**: 이번 연구에서는 생성 인공지능(generative AI) 사용에서의 민감한 정보 유출 문제를 해결하기 위한 새로운 보안 방법론을 제안합니다. 이 방법론은 사용자의 입력 데이터와 모델 출력 데이터의 비밀성을 유지하며, 공정성과 검증 가능성을 보장합니다.

- **Technical Details**: 본 연구에서는 Secure Multi-Party Computation (SMPC) 기법을 기반으로 하여 트랜스포머(transformer) 기반의 생성 AI 모델을 최적의 방식으로 분할(shard)합니다. 이를 통해 입력을 숨기고, 출력의 모호성을 증가시키며, 분산 네트워크 상에서 지적 재산권(intellectual property)이 보호될 수 있도록 합니다.

- **Performance Highlights**: 우리의 방법론은 분산 네트워크에서의 보안성과 검증 가능성을 제공하며, 계산 노드 중 하나가 정직할 경우 보안이 유지됩니다. 또한, 대부분의 노드가 성공적으로 작업을 수행하면 추론 프로세스도 성공적으로 완료될 수 있습니다.



### Responsible AI in Open Ecosystems: Reconciling Innovation with Risk Assessment and Disclosur (https://arxiv.org/abs/2409.19104)
Comments:
          [Under Review][WIP]

- **What's New**: 최근 AI의 빠른 성장으로 인해 윤리적 고려 사항에 대한 중요성이 커지고 있으며, 이는 모델 감사(model auditing) 및 보고 요구 사항의 발전을 초래하고, 개인과 사회에 대한 잠재적 위험을 완화하기 위한 거버넌스 프레임워크(g governance frameworks) 수립으로 이어지고 있습니다. 본 연구는 OSS(오픈 소스 소프트웨어)와 같은 비공식 부문에서의 책임 있는 AI 및 투명성 증진의 실질적인 도전 과제를 검토하고, 모델 성능 평가가 모델의 한계, 편향 및 기타 위험 요소를 어떻게 탐색하게 하는지에 대해 논의합니다.

- **Technical Details**: 본 연구에서 7903개의 Hugging Face 프로젝트에 대한 통제된 분석을 실시하였으며, 위험 문서화(risk documentation)와 평가 관행(evaluation practices) 간에 강한 상관관계를 발견했습니다. 또한, Hugging Face의 경쟁 리더보드에서 참가한 789개의 프로젝트 중에서 고성능 모델의 경우 위험 및 제한 사항에 대한 문서화를 제공하지 않는 경향이 있음을 발견했습니다.

- **Performance Highlights**: 이 연구는 AI 개발자와 법학자들에게 오픈 소스 혁신을 보존하면서 윤리적 채택을 장려할 수 있는 개입(interventions) 및 정책 설계에 대한 통찰력을 제공합니다. 연구 결과, 서비스 준비가 완료된 프로젝트의 약 15.9%와 2.2%의 모델에서 평가 및 위험 관련 문서화가 발견되었습니다.



### Differential privacy for protecting patient data in speech disorder detection using deep learning (https://arxiv.org/abs/2409.19078)
- **What's New**: 이 연구는 병리학적 (pathological) 음성 데이터에서의 차별적 프라이버시 (differential privacy, DP) 적용의 영향을 최초로 조사하였습니다. 이는 개인 정보 보호 (privacy)와 진단 정확도 (diagnostic accuracy), 공정성 (fairness) 사이의 균형을 탐구합니다.

- **Technical Details**: 본 연구는 2,839명의 독일어 사용 참여자로부터 수집된 200시간의 실제 음성 데이터셋을 사용하였으며, DP의 개인정보 보호 예산 (privacy budget)인 {
\epsilon} = 7.51을 적용했을 때 최대 정확도 감소가 3.85%에 달한다는 사실을 확인했습니다. 또한, 스페인어를 사용하는 파킨슨병 환자를 대상으로 한 소규모 데이터셋에서도 DP 제약 하에서의 모델 정확도 유지 또는 향상을 입증했습니다.

- **Performance Highlights**: 연구 결과, DP는 음성 장애 감지에서 개인 정보 보호와 유용성 (utility) 간의 효과적인 균형을 이룰 수 있음을 보여주었습니다. 그러나 음성 분야에서의 독특한 도전 과제와 개인 정보 보호-공정성 간의 균형을 강조했습니다.



### Meta-RTL: Reinforcement-Based Meta-Transfer Learning for Low-Resource Commonsense Reasoning (https://arxiv.org/abs/2409.19075)
- **What's New**: 본 연구에서는 강화 학습(double reinforcement learning)을 기반으로 한 다중 출처 메타 전이 학습 프레임워크(Meta-RTL)를 제안합니다. 이는 낮은 자원 환경에서의 상식 추론을 개선하기 위해 출처 작업에 대한 동적 가중치를 추정하는 접근 방식을 포함하고 있습니다.

- **Technical Details**: Meta-RTL은 LSTM(Long Short-Term Memory)을 기반으로 한 정책 네트워크를 사용하여 출처 작업의 기여도를 측정하는 가중치를 동적으로 업데이트합니다. 메타 모델과 출처 특화 임시 메타 모델 간의 일반 손실과 작업별 손실의 차이를 보상으로 사용하여 강화 학습 모듈의 정책 네트워크에 피드백합니다.

- **Performance Highlights**: Meta-RTL은 BERT와 ALBERT를 메타 모델의 백본으로 사용하여 세 가지 상식 추론 벤치마크 데이터셋에서 평가되었습니다. 실험 결과, Meta-RTL은 강력한 기준 모델들을 초과하는 성능을 보였고, 특정 데이터 부족 환경에서도 더 큰 개선을 이끌어냈습니다.



### CLLMate: A Multimodal LLM for Weather and Climate Events Forecasting (https://arxiv.org/abs/2409.19058)
- **What's New**: 이번 연구에서 제안된 Weather and Climate Event Forecasting (WCEF) 작업은 과거 날씨 및 기후 이벤트와 기상 데이터를 연계하여 날씨 및 기후 사건을 예측하는 새로운 과제로, 기존의 닫힌 집합 이벤트 예측 방식의 한계를 극복하는 데 중점을 두고 있습니다.

- **Technical Details**: WCEF 작업은 기상 레스터 데이터(meteorological raster data)와 사건 텍스트(data)를 통합하여 기상 사건을 예측하는 것을 목표로 합니다. 이를 위해 Llama3와 같은 대형 언어 모델(LLM)을 활용하여 과거의 기상 데이터를 사건으로 정리한 지식 그래프(knowledge graph)를 구성하였고, 이에 기반하여 최초의 다중 모달 지침 데이터셋(multimodal instruction dataset)을 생성하였습니다. 나아가 CLLMate라는 다중 모달 LLM을 통해 날씨 예측을 위한 기상 레스터 데이터를 효과적으로 관리하고 있습니다.

- **Performance Highlights**: CLLMate 모델은 기존의 기준 모델(baselines) 및 다른 다중 모달 LLM과 비교하여 뛰어난 성능을 보였으며, 기상 데이터와 사건 데이터를 효과적으로 정렬 및 통합하여 개방형 사건 예측을 단순화했습니다. 연구 결과는 WCEF 작업에 대한 연구의 미래가 가능하다는 점을 강조하며, 이 모델은 기후 위험 완화 시스템의 중요한 구성 요소로 자리 잡을 수 있는 잠재력을 보여줍니다.



### Multimodal Markup Document Models for Graphic Design Completion (https://arxiv.org/abs/2409.19051)
Comments:
          Project page: this https URL

- **What's New**: 이 논문은 interleaved multimodal documents 내에서 markup language와 이미지를 동시에 생성할 수 있는 Multimodal Markup Document Models(MarkupDM)을 제안합니다. 이 모델은 graphic design 작업에 필수적인 고유한 과제를 다룹니다.

- **Technical Details**: MarkupDM은 다양한 크기와 투명성을 가진 이미지 토큰화를 위한 이미지 양자화기(image quantizer)를 설계하고, markup language를 처리할 수 있는 코드 언어 모델을 수정하여 이미지 모달리티를 통합합니다. 장기적으로 166K graphic design templates로 훈련되었습니다.

- **Performance Highlights**: 세 가지 graphic design completion 작업에서 MarkupDM의 효과가 입증되었습니다. 생성된 디자인은 주어진 맥락에 일관되며, 다양한 디자인 대안을 탐색할 수 있게 해줍니다.



### On the Inductive Bias of Stacking Towards Improving Reasoning (https://arxiv.org/abs/2409.19044)
Comments:
          Accepted at NeurIPS 2024

- **What's New**: 이 논문에서는 새로운 훈련 전략인 MIDAS를 제안합니다. MIDAS는 언어 모델 훈련을 최대 40% 가속할 수 있으며, 특히 추론 능력이 필요한 다운스트림 작업에서의 성능 향상을 제공합니다.

- **Technical Details**: MIDAS는 기존의 점진적 스태깅(gradual stacking) 기법의 변형으로, 작은 네트워크의 중간 블록을 복사하여 큰 네트워크를 초기화하는 방식입니다. 이 방법은 표준 훈련 방식과 유사한 데이터와 FLOPS(부동소수점 연산 수)를 사용하면서도 우수한 성능을 보입니다.

- **Performance Highlights**: MIDAS는 수학 문제와 추론 프리미티브 등에서 뛰어난 성능을 발휘하며, 표준 방법과 성능 비교에서 더 나은 결과를 나타냈습니다. 특히, MIDAS로 사전 훈련된 모델은 표준 훈련 모델보다 추론 프리미티브에서 유의미한 개선을 보였습니다.



### Elephant in the Room: Unveiling the Impact of Reward Model Quality in Alignmen (https://arxiv.org/abs/2409.19024)
- **What's New**: 이번 연구에서는 LLM(대형 언어 모델)의 정렬(alignment) 방법론에서 보상 모델(reward model)의 중요성을 강조하며, 이를 개선하기 위한 노력이 필요함을 제기합니다. 연구진은 HH-RLHF 데이터셋의 품질을 분석하고 청소한 버전인 CHH-RLHF를 제시합니다.

- **Technical Details**: 연구에서는 세 가지 보상 활용 패러다임(직접 보상, 간접 보상, 직접 선호)을 기반으로 다양한 보상 모델의 정확성을 저명한 사례를 통해 벤치마킹하였으며, 보상 모델의 품질이 정렬 성능에 미치는 영향을 체계적으로 연구합니다.

- **Performance Highlights**: 보다 나은 보상 모델이 LLM의 정렬 성능을 더욱 향상시키는 것으로 나타났으며, 부적합한 모델을 사용할 경우 인간 평가와 일치성이 떨어지는 결과가 나타났습니다. 이는 연구자들에게 보상 모델의 품질에 대한 주의를 환기시키는 계기가 됩니다.



### Application of AI-based Models for Online Fraud Detection and Analysis (https://arxiv.org/abs/2409.19022)
Comments:
          Manuscript under peer review. Content may be subject to revision in future versions

- **What's New**: 이번 논문에서는 AI 및 NLP 기술을 활용한 온라인 사기 감지를 위한 체계적인 문헌 검토(Systematic Literature Review, SLR)를 수행하였습니다. 온라인 사기의 정의와 그로 인한 피해, 그리고 AI 기술이 사기 탐지에 응용되는 현재 상태가 중점적으로 분석되었습니다. 특히, 다양한 사기 유형들에 대한 연구 결과를 종합하여 정책입안자, 법 집행 기관 및 기업이 사기 예방 및 방지에 있어 어떻게 대응할 수 있을지를 제시합니다.

- **Technical Details**: 이번 연구는 PRISMA-ScR 프로토콜을 준수하여 2,457개의 학술 기록을 스크리닝한 결과, 350개의 연구가 적격한 기준을 만족하였고 최종적으로 223개의 연구가 포함되었습니다. 이 연구는 NLP 알고리즘 및 모델, 훈련 데이터 출처, 성과 평가 지표 등을 포함하여 다양한 온라인 사기 카테고리에 대한 최첨단 연구 결과를 보고하였습니다.

- **Performance Highlights**: 연구 결과, 현재 온라인 사기에 대한 연구는 여러 가지 사기 활동으로 나뉘어 있으며, 연구자들은 16가지 서로 다른 사기에 초점을 맞추고 있습니다. 특히, 모델 효율성 제고를 위한 데이터의 한계, 훈련 편향 보고 및 선택적 성과 지표 발표와 같은 문제를 식별하였으며, 이는 모델 평가에서 잠재적인 편향을 초래할 수 있다는 점을 강조하였습니다.



### Textless NLP -- Zero Resource Challenge with Low Resource Compu (https://arxiv.org/abs/2409.19015)
- **What's New**: 이 연구는 Textless NLP(텍스트 없는 자연어 처리)에서 경량의 인코더-보코더 모델 학습 시 발생하는 훈련 시간 및 GPU 자원의 문제를 해결하기 위한 방법론을 제안합니다. 주요 기여는 학습률 스케줄러(learning rate scheduler)를 활용하여 훈련 단계를 효과적으로 줄이고 성능을 향상시키는 것입니다.

- **Technical Details**: 이 시스템은 Vector-Quantized Contrastive Predictive Coding(VQ-CPC)을 인코더로 사용하고, LSTM 기반의 경량 보코더를 사용합니다. 학습률 스케줄러를 One-Cycle Learning Rate(OCLR)로 설정하여 훈련 시간을 80%까지 줄이고, 오디오 품질 향상을 위해 hop length와 interpolation scale factors를 최적화했습니다. 또한, 인도어(Indian languages) 데이터셋에 대한 실험이 포함되어 있습니다.

- **Performance Highlights**: 제안된 방법은 English, Tamil, Bengali 데이터셋에서 일관되게 우수한 결과를 보여주었으며, 언어 변환 시 재구성된 오디오의 선명도가 눈에 띄게 향상되었습니다. 훈련 시간도 28시간에서 최적화된 단계 수(30k, 40k, 60k)에 따라 각각 6, 8, 12시간으로 줄어들었습니다.



### Improving Academic Skills Assessment with NLP and Ensemble Learning (https://arxiv.org/abs/2409.19013)
Comments:
          5 pages, 2 figures

- **What's New**: 이번 연구는 자연어 처리(NLP)의 발전을 활용하여 기초 학문 기술을 평가하는 데 있어 주요 도전 과제를 다룹니다. 기존의 전통적 평가 방법은 인지적 및 언어적 측면에서 적시의 포괄적인 피드백을 제공하는 데 어려움이 있었습니다. 본 연구는 BERT, RoBERTa, BART, DeBERTa, T5와 같은 최신 NLP 모델을 통합하여 앙상블 학습(enesemble learning) 프레임워크를 통해 정확성을 크게 향상시켰습니다.

- **Technical Details**: 이 모델은 상태를 유지하기 위해 LightGBM과 Ridge 회귀를 사용하여 여러 NLP 모델을 스택(stacking) 기법으로 결합했습니다. 데이터 전처리, 특징 추출, 그리고 pseudo-label 학습을 통해 모델 성능을 최적화했습니다. 또한, PyTorch 모델 클래스를 사용하여 텍스트 입력을 처리하고 효과적으로 분류 작업을 수행했습니다.

- **Performance Highlights**: 이 연구는 ESL(English as a Second Language) 학생들을 위한 언어 평가의 정확성을 크게 향상시켰으며, 전통적인 평가 방법의 한계를 극복하고 교육 기술 연구에서 핵심 학문 역량 향상에 대한 새로운 가능성을 열어주는 강력한 솔루션을 제공했습니다.



### Lost in the Logic: An Evaluation of Large Language Models' Reasoning Capabilities on LSAT Logic Games (https://arxiv.org/abs/2409.19012)
Comments:
          Bachelor's thesis. Dataset available on huggingface: this https URL

- **What's New**: 이 논문은 법학전문대학원 입학시험(LSAT)에서의 대형 언어 모델(LLMs)의 성능 평가를 다룹니다. 특히 논리 게임 부분에서의 복잡한 논리적 추론 작업을 분석하며, 이를 통해 최신 LLM들이 어려운 논리적 과제를 어떻게 처리하는지를 평가합니다.

- **Technical Details**: LSAT 논리 게임을 포함하는 데이터셋을 구성하고 다양한 LLM의 성능을 평가하기 위해 Chain-of-Thought prompting을 사용합니다. 약한 성능에도 불구하고 Reflexion에서 아이디어를 차용한 다른 prompting 프레임워크를 적용한 결과, GPT-4는 70%, GPT-3.5는 46%의 정확도를 보였습니다.

- **Performance Highlights**: GPT-4는 Multi-Shot Chain-of-Thought prompting에서 33%의 정확도로 최상을 기록하였으나, Self-Reflection 기반의 프레임워크를 적용했을 때 70%로 크게 향상되었습니다. 이는 LLM들이 자신의 논리적 오류를 반영하고 수정하는 능력이 뛰어남을 보여줍니다.



### Identification and Mitigating Bias in Quantum Machine Learning (https://arxiv.org/abs/2409.19011)
Comments:
          2 pages

- **What's New**: 이번 연구에서는 Quantum Machine Learning (QML) 의 고유한 편향(bias) 및 과제를 조명하며, 이러한 편향들이 QML에서 어떻게 식별되고 진단되며 대응될 수 있는지를 다룹니다. 이 논문은 QML의 편향의 세 가지 주요 주제에 대한 포괄적인 개요를 제공합니다.

- **Technical Details**: QML에서 발생하는 여러 가지 편향 유형으로는 Encoding Bias, Inductive Bias, Realizability Bias, State-Dependent Bias, Sampling Bias가 있으며, 각 편향의 원인과 QML 모델 성능에 미치는 영향을 분석하였습니다. 특히 Encoding Bias는 고전 데이터의 양자 상태로의 변환과 양자 알고리즘 간의 상호작용에서 발생합니다. 다양한 인코딩 기법을 통해 MNIST 데이터셋에 대해 실험을 진행하고, 각 인코딩 방식의 성능 차이를 입증하였습니다.

- **Performance Highlights**: Encoding Bias에 대한 실험에서는 Basis Encoding이 모든 에포크(epoch)에서 낮은 정확도를 보였고, Angle Encoding이 빠르게 성능이 향상되어 높은 정확도를 유지했습니다. Hybrid Parameterized Encoding에서 Rx, Ry, Rz의 각 인코딩 방식 또한 성능 차이를 보였으며, Rysubscript𝑅𝑦R_{y}encoding가 가장 우수한 초기 개선을 보여주었습니다. 이러한 연구 결과는 QML 시스템의 편향에 대한 보다 나은 이해를 제공하며, 향후 공정성 고려 사항이 양자 영역으로 확장되어야 함을 시사합니다.



### A comprehensive study of on-device NLP applications -- VQA, automated Form filling, Smart Replies for Linguistic Codeswitching (https://arxiv.org/abs/2409.19010)
- **What's New**: 최근 대형 언어 모델의 발전이 이전에는 불가능했던 새로운 경험들을 온디바이스(온디바이스; on-device) 애플리케이션에서 제공할 수 있는 기회를 열어주었습니다. 본 연구에서는 크게 두 가지 범주로 나누어진 세 가지 새로운 경험을 제안합니다.

- **Technical Details**: 첫 번째 범주는 화면 이해(Screen Understanding)와 관련된 경험으로, 사용자 화면에 나타난 정보에 대한 이해(Visual Question Answering, 자동 양식 채우기)가 포함됩니다. 두 번째 범주는 코드 스위칭(Code-Switching) 기능을 지원하는 스마트_reply(smart replies) 시스템의 확장입니다. 본 연구에서 제안된 첫 번째 작업으로는 화면 기반 이해를 위한 Visual Question Answering과 이전 화면의 맥락을 활용한 자동 양식 채우기 과제가 포함됩니다. 모델은 LayoutLM과 MarkupLM 두 가지 계열로 구성되며, LayoutLMv3는 이미지와 텍스트 토큰을 정렬하여 다중 사전 훈련 과정을 수행합니다.

- **Performance Highlights**: 데이터 수집 및 질문-답변 쌍 생성 방법을 통해 4,500개 이상의 iOS 앱에서 100,000개 이상의 스크린샷을 수집했습니다. 이를 기반으로 모델을 훈련하여 기존 양식 정보 및 시각적 맥락에 대한 정확도를 개선했습니다. 처음으로 제안된 이 연구는 화면 기반으로 질문을 생성하는 작업과 다국어 사용자 정보를 기반으로 한 개인화된 스마트 응답 생성을 포함하여 새로운 응용 프로그램을 탐구합니다.



### Towards Automated Patent Workflows: AI-Orchestrated Multi-Agent Framework for Intellectual Property Management and Analysis (https://arxiv.org/abs/2409.19006)
Comments:
          This is a preprint and current version under peer review

- **What's New**: 본 논문은 PatExpert라는 자율 다중 대화형 프레임워크를 제안하며, 이는 다양하고 복잡한 특허 관련 작업을 최적화하고 자동화하는 데 도움을 줍니다. 프레임워크는 메타 에이전트와 특정 작업을 수행하는 전문 에이전트로 구성되어 있습니다.

- **Technical Details**: PatExpert 프레임워크는 메타 에이전트가 고유한 태스크를 수행하는 여러 전문 에이전트를 조정하는 방식으로 작동합니다. 이 시스템은 Graph Retrieval-Augmented Generation (GRAG)과 같은 고급 기법을 사용하여 지식 그래프를 활용해 정확성과 관련성을 향상시킵니다.

- **Performance Highlights**: PatExpert는 다중 특허 분석, 특허 분류, 청구 생성 및 오류 처리를 포함하여 예상치 못한 복잡한 작업을 자동화함으로써 특허 처리 작업의 효율성과 정확성을 크게 향상시켰습니다.



### Pay Attention to What Matters (https://arxiv.org/abs/2409.19001)
- **What's New**: 이번 논문에서는 사용자 지침에 대한 LLM(대형 언어 모델)의 출력을 정렬하는 능력을 향상시키기 위한 새로운 방법인 GUIDE(Instruction-Driven Enhancements에 의한 Guided Understanding)를 소개합니다. GUIDE는 중요한 지침에 대한 attention score를 증가시켜 LLM이 사용자 지침을 보다 정확하게 따르도록 돕습니다.

- **Technical Details**: GUIDE는 입력 내에서 특정 토큰을 강조하기 위한 시스템적 접근법으로, 사용자는 중요한 텍스트를 <!-> <-!>와 같은 태그로 묶음으로써 attention을 유도할 수 있습니다. 이 방법은 LLM의 attention score에 bias를 추가하여 특정 토큰에 주의를 환기시키며, 새로운 측정 지표인 Influence를 제시하여 지침-토큰 간의 중요성을 정량화합니다.

- **Performance Highlights**: GUIDE는 지침을 따르는 정확도를 29.4%에서 60.4%로 개선하여 자연어 프롬프트 방식 및 100만 토큰까지의 Supervised Fine-Tuning(SFT)보다 우수한 성능을 보여줍니다. GUIDE는 추가적인 훈련 없이도 효과적으로 사용 가능하며, 사용자의 요구에 맞춤화된 지침 이행을 가능하게 합니다.



### Controlled LLM-based Reasoning for Clinical Trial Retrieva (https://arxiv.org/abs/2409.18998)
- **What's New**: 본 논문에서는 임상 시험(CT) 데이터의 검색 및 재배치를 위한 새로운 체계적인 추론 방법을 제안합니다. 이 방법은 LLMs(Large Language Models)의 기능을 확장하여 의료 적격 기준에 대한 체계적인 추론을 가능하게 합니다.

- **Technical Details**: 제안된 방법은 LLM을 사용하여 환자 메모와 임상 시험 기록을 속성 세트로 변환하고, 이를 통해 CT의 정확한 매핑과 해석이 이루어질 수 있습니다. 여기서 'Set-guided reasoning' 방법이 사용되어 hierarchical relationship을 통해 도메인 특화 지식을 활용합니다.

- **Performance Highlights**: TREC 2022 Clinical Trials에서 평가된 결과, NDCG@10이 0.693, Precision@10이 0.73로 기존의 최첨단 성능을 초월했습니다.



### PropaInsight: Toward Deeper Understanding of Propaganda in Terms of Techniques, Appeals, and Inten (https://arxiv.org/abs/2409.18997)
Comments:
          8 pages

- **What's New**: 본 연구는 PropaInsight라는 새로운 개념적 프레임워크를 도입하여 선전(propaganda)의 기법, 자극 호소(arousal appeals), 그리고 기본 의도(underlying intent)를 체계적으로 분석합니다. 더불어, PropaGaze라는 새로운 데이터셋을 통해 전문가 주석이 부족한 상황에서도 선전을 효과적으로 분석할 수 있는 방법을 제시합니다.

- **Technical Details**: PropaInsight는 선전의 세 가지 주요 요소를 식별하며, 각 요소는 기법 식별(Propaganda Technique Identification), 호소 분석(Appeal Analysis), 의도 분석(Intent Analysis)으로 나뉩니다. PropaGaze 데이터셋은 세 가지 하위 데이터셋으로 구성되어 있으며, 각 데이터셋은 인간 주석자에 의해 주석이 달린 뉴스 기사와 고품질 합성 데이터로 이루어져 있습니다. 이는 LLMs(대형 언어 모델)의 강력한 이해 능력을 활용하여 작성되었습니다.

- **Performance Highlights**: PropaGaze를 사용한 실험에서는 LLMs가 선전 분석에서 어려움을 겪는 것으로 나타났지만, PropaGaze로 학습한 경우 성능이 크게 개선되었습니다. 예를 들어, Llama-7B-Chat은 기법 식별에서 1-shot GPT-4-Turbo에 비해 203.4% 높은 IoU(text span Intersection over Union)를 달성하고, 호소 분석에서 66.2% 높은 BertScore를 기록했습니다.



### From Linguistic Giants to Sensory Maestros: A Survey on Cross-Modal Reasoning with Large Language Models (https://arxiv.org/abs/2409.18996)
- **What's New**: 본 논문은 Cross-Modal Reasoning (CMR)과 관련된 최근의 대규모 언어 모델(LLMs)의 발달을 다루고 있으며, LLMs가 CMR에 미치는 역할을 세분화하여 탐구하는 첫 번째 설문조사로서의 중요성을 강조합니다.

- **Technical Details**: 논문은 LLMs의 네 가지 주요 역할인 Multimodal Fusion Engine, Textual Processor, Cognitive Controller, Knowledge Enhancer를 소개합니다. 또한 Prompt Tuning, Instruction Tuning, Multimodal Pre-training과 같은 방법론을 설명하며 이들이 CMR에 활용되는 방식을 상세히 다룹니다.

- **Performance Highlights**: LLMs는 텍스트, 이미지 및 소리 등 다양한 모드 간의 새로운 정보를 이해하고 추론하는 능력을 통합하여 그들의 성능을 향상시킵니다. CMR의 적용 예시로는 비주얼 질문 응답, 비전-언어 탐색, 이미지 및 비디오 캡셔닝 등이 포함됩니다.



### Systematic Characterization of the Effectiveness of Alignment in Large Language Models for Categorical Decisions (https://arxiv.org/abs/2409.18995)
Comments:
          19 pages (without Appendix) Appendix 7 pages. 7 Figures

- **What's New**: 이 논문은 대규모 언어 모델(LLM)이 의료 분야와 같은 고위험 영역에서 인간의 선호 및 가치와 얼마나 잘 일치하는지 평가하는 체계적인 방법론을 제시합니다. 특히, 의료 분류(triage)라는 특정 사용 사례를 통해 LLM의 결정이 전문가의 선호와 얼마나 일치하는지를 평가합니다.

- **Technical Details**: 중요한 방법론적 요소 중 하나는 Alignment Compliance Index (ACI)라는 새로운 측정 지표로, 이는 LLM이 특정 선호 함수에 얼마나 효과적으로 정렬되는지를 정량화합니다. 이 연구에서는 GPT4o, Claude 3.5 Sonnet, 및 Gemini Advanced와 같은 세 가지 첨단 LLM을 사용하여 의사 결정의 일관성을 평가하였습니다.

- **Performance Highlights**: 모델 간의 정렬 효과는 상당한 변동성을 보였으며, ACI에 의해 잘 수행된 모델이 정렬 후에는 성능이 저하되는 경향도 발견되었습니다. 또한, 목표 선호 함수의 미세한 변화가 모델의 순위에 큰 영향을 미친다는 점도 주목할 만합니다.



### A Review of Mechanistic Models of Event Comprehension (https://arxiv.org/abs/2409.18992)
- **What's New**: 이 리뷰는 사건 이해(Event comprehension)의 이론적 가정과 계산 모델을 검토하며, 담화 이해(disourse comprehension) 이론에서 현대의 사건 인지(event cognition) 프레임워크로의 진화를 추적합니다.

- **Technical Details**: 주요 담화 이해 모델로는 Construction-Integration, Event Indexing, Causal Network, 그리고 Resonance 모델 등이 있으며, 이들은 이해 과정에서의 인지적 프로세스를 이해하는 데 기여합니다. 현대의 사건 이해 이론으로는 Event Segmentation Theory, Event Horizon Model, Hierarchical Generative Framework 등이 있으며, 이들은 사건 이해에서의 예측(prediction), 인과 관계(causality), 다계층 표현(multilevel representations)의 중요성을 강조합니다. 분석된 다섯 가지 계산 모델로는 REPRISE, Structured Event Memory, Lu 모델, Gumbsch 모델, Elman 및 McRae 모델이 있습니다.

- **Performance Highlights**: 계층적 처리(hierarchical processing), 예측 메커니즘(prediction mechanisms), 그리고 표현 학습(representation learning)에 대한 접근 방법에 초점을 맞추어, 사건 이해에 있어 예측의 중요성과 사건 역학(Event dynamics)의 학습을 위한 다양한 전략을 강조합니다. 향후 연구의 중요한 영역으로는 구조적 표현(structured representations)에 대한 학습을 위한 더 정교한 접근법 필요성, 일화 기억 메커니즘(episodic memory mechanisms)의 통합, 사건 모델에 대한 적응형 업데이트 알고리즘 개발이 포함됩니다.



### SC-Phi2: A Fine-tuned Small Language Model for StarCraft II Macromanagement Tasks (https://arxiv.org/abs/2409.18989)
- **What's New**: SC-Phi2 모델은 StarCraft II의 macromanagement 작업을 위한 새로운 소형 언어 모델입니다. 이를 통해 소형 언어 모델이 이전의 대형 언어 모델보다 적은 자원으로 효과적인 성능을 발휘할 수 있음을 보여줍니다.

- **Technical Details**: SC-Phi2는 Microsoft의 Phi2 모델을 기반으로 하며, 새로운 SC2 텍스트 데이터셋을 사용하여 self-supervised learning에 의해 fine-tuning되며, Vision Transformer(ViT)와 결합하여 gameplay 동적 프롬프트를 생성합니다. 이 모델은 Low-rank Adaptation(LoRA) 기법과 양자화(Quantization)를 통해 단일 GPU에서 훈련됩니다.

- **Performance Highlights**: 모델은 build order 및 global state 예측과 같은 micromanagement 작업에서 훌륭한 성능을 보이며, 전체적으로 2.8 billion parameters만을 가지면서도 타 모델들과 비교해도 경쟁력 있는 예측 능력을 보여줍니다.



### A Unified Framework to Classify Business Activities into International Standard Industrial Classification through Large Language Models for Circular Economy (https://arxiv.org/abs/2409.18988)
Comments:
          6 pages, 2 figures, accepted in 2024 IEEE International Conference on Industrial Engineering and Engineering Management (IEEM 2024)

- **What's New**: 이 논문은 순환 경제(practices) 관행을 촉진하는 추천 시스템을 개발하기 위해 효과적인 정보 수집과 지식 코딩의 중요성을 강조합니다.

- **Technical Details**: 이 연구에서는 대규모 언어 모델(LLMs)을 활용하여 경제 활동을 국제 표준 산업 분류(ISIC) 체계로 분류합니다. 이를 통해 다양한 지역의 모든 경제 활동 설명을 통합된 ISIC 표준으로 분류할 수 있게 합니다.

- **Performance Highlights**: GPT-2 모델을 미세 조정하여 182개의 레이블 테스트 데이터셋에서 95%의 정확도를 달성했습니다. 이 연구는 지속 가능한 순환 경제 관행을 촉진하는 데 기여합니다.



### Efficient and Personalized Mobile Health Event Prediction via Small Language Models (https://arxiv.org/abs/2409.18987)
Comments:
          6 pages, 3 figures

- **What's New**: 이 논문에서는 헬스케어 모니터링에 대한 소형 언어 모델(SLMs)의 능력을 처음으로 조사하였으며, 환경을 보호하면서 개인화된 건강 상태 분석을 위한 가능성을 제시합니다.

- **Technical Details**: 이 연구에서는 TinyLlama(1.1B 파라미터)의 성능을 분석하여 4.31GB의 메모리 사용량과 0.48초의 지연시간(latency)을 나타냈으며, 다른 4개의 최신 소형 언어 모델(SOTA SLMs)보다 우수한 성능을 보였습니다.

- **Performance Highlights**: SLMs는 헬스케어 모니터링의 최신 솔루션으로써 우수한 성능을 보여주며, 특히 CPU 사용량, 지연시간, 메모리 사용량에서 15.5배의 개선을 보였습니다. 또한, 기존 LLMs와 비교하여 실시간 헬스케어 응용 프로그램에 더 적합한 것으로 판단됩니다.



### Lab-AI -- Retrieval-Augmented Language Model for Personalized Lab Test Interpretation in Clinical Medicin (https://arxiv.org/abs/2409.18986)
- **What's New**: Lab-AI는 환자 맞춤형 정상 범위를 제공하는 상호작용 시스템으로, Retrieval-Augmented Generation (RAG) 기술을 활용하여 신뢰할 수 있는 건강 정보 소스로부터 정보를 검색합니다.

- **Technical Details**: Lab-AI는 두 가지 모듈, 즉 factor retrieval 및 normal range retrieval로 구성되어 있으며, 68개의 실험실 테스트에서 30개는 조건적 요소가 포함되고 38개는 포함되지 않습니다. 테스트의 정상 범위는 환자-specific information에 따라 달라집니다. GPT-4-turbo 모델은 factor retrieval에서 0.95의 F1 score, normal range retrieval에서 0.993의 정확도를 보였습니다.

- **Performance Highlights**: RAG를 사용하는 GPT-4-turbo는 비-RAG 시스템보다 29.1% 더 높은 factor retrieval 성능을 나타내었고, normal range retrieval에서 질문 수준에서 60.9%, 실험실 수준에서 52.9% 향상을 보였습니다. 이러한 결과는 Lab-AI가 환자가 실험실 결과를 이해하는 데 도움을 줄 수 있는 잠재력을 강조합니다.



### Harnessing Large Language Models: Fine-tuned BERT for Detecting Charismatic Leadership Tactics in Natural Languag (https://arxiv.org/abs/2409.18984)
Comments:
          The 2024 IEEE 3rd Conference on Information Technology and Data Science, CITDS 2024

- **What's New**: 이 연구는 Charismatic Leadership Tactics (CLTs)를 자연어에서 식별하기 위해 미세 조정된 Bidirectional Encoder Representations from Transformers (BERT) 모델을 사용하는 방법을 탐구합니다. CLTs를 위한 대규모 코퍼스를 기반으로 하여, 본 연구는 자연어에서 이 전술의 존재를 정확히 식별할 수 있는 기계 학습 모델을 훈련하는 방법론을 제시합니다.

- **Technical Details**: 본 연구는 BERT 모델을 특정 CLTs를 대상으로 미세 조정하여 해당 전술의 존재를 텍스트에서 식별합니다. 연구는 BERT로 훈련된 모델이 CLTs를 효과적으로 탐지하는지 평가하고, 다루는 특정 데이터셋에 대해 98.96%의 높은 정확도를 보였습니다.

- **Performance Highlights**: 본 연구의 모델은 자연어 처리(NLP) 기술을 통해 Charismatic Leadership의 언어적 특성을 분석할 수 있는 도구를 개발함으로써 심리학 및 경영 분야의 미래 연구에 기여할 수 있는 잠재력을 가집니다.



### Aligning Robot Navigation Behaviors with Human Intentions and Preferences (https://arxiv.org/abs/2409.18982)
Comments:
          Haresh Karnan's PhD Dissertation A recording of the defense talk can be accessed here: this https URL

- **What's New**: 최근 기계 학습 분야의 발전으로 모바일 로봇의 탐색 능력을 향상시키기 위한 새로운 방법들이 등장하고 있습니다. 이 논문은 특히 자율 모바일 로봇의 네비게이션(autonomous navigation) 행동을 인간의 의도와 선호에 맞추기 위한 기계 학습 방법에 대해 다루고 있습니다.

- **Technical Details**: 이 연구에서는 인간의 탐색 작업을 모방하여 학습하는 새로운 접근 방식을 제안합니다. 이를 통해 모바일 로봇은 목표를 인간처럼 인식하고 행동할 수 있도록 독자적인 시각적 네비게이션 능력을 습득하게 됩니다. 특히, Learning from Preferences (lfp)라는 패러다임을 통해 로봇이 환경에서 간단히 선호를 학습하고, 오프로드 탐색 관련 두 가지 알고리즘 또한 도입하여 다양한 지형에서의 탐색 능력을 향상시킵니다.

- **Performance Highlights**: 이 논문은 자율 로봇이 인간의 의도와 선호에 맞춰 자율적으로 탐색할 수 있는 능력을 갖출 수 있도록 돕는 중요한 단계로, 실제 환경에서의 안전한 탐색을 위한 데이터셋과 알고리즘을 제시합니다. 이러한 접근 방식은 가치 정렬(value alignment) 문제를 해결할 수 있는 잠재력을 지니고 있으며, 로봇의 행동이 인간과 조化됩니다.



### IW-Bench: Evaluating Large Multimodal Models for Converting Image-to-Web (https://arxiv.org/abs/2409.18980)
- **What's New**: 최근 대규모 멀티모달 모델의 발전이 이미지 이해 능력에서 크게 향상되었습니다. 그러나 이러한 대규모 모델의 Image-to-Web 전환 능력을 평가하기 위한 강력한 기준이 부족합니다. 이를 해결하기 위해 IW-Bench라는 새로운 기준을 제정하고, Element Accuracy 및 Layout Accuracy와 같은 새로운 평가지표를 개발했습니다.

- **Technical Details**: IW-Bench는 1200개의 이미지와 웹 코드 쌍으로 구성되어 있으며, 난이도는 간단, 중간, 복잡으로 구분됩니다. Element Accuracy는 DOM (Document Object Model) 트리를 파싱하여 웹 요소의 완전성을 평가하며, Layout Accuracy는 DOM 트리를 공통 부분 수열로 변환하여 요소의 상대적 위치 관계를 분석합니다. 또한, 다섯 단계의 Chain-of-Thought Prompting을 통해 성능을 향상시키도록 설계되었습니다.

- **Performance Highlights**: 대규모 멀티모달 모델에 대한 광범위한 평가를 실시하였으며, 결과는 이들 모델의 강점과 개선이 필요한 영역에 대한 통찰을 제공합니다. 특히, 새로운 다섯 단계의 Chain-of-Thought 방법론이 성능 향상에 기여한 것으로 나타났습니다.



### EEG-EMG FAConformer: Frequency Aware Conv-Transformer for the fusion of EEG and EMG (https://arxiv.org/abs/2409.18973)
- **What's New**: 이번 연구에서는 EEG(전기뇌파)와 EMG(근전도) 신호를 사용하여 motor pattern recognition(움직임 패턴 인식)을 향상시키기 위한 EEG-EMG FAConformer라는 새로운 알고리즘을 제안합니다. 이 모델은 주로 attention 모듈을 사용하여 시간적 및 주파수 정보를 효과적으로 활용하며, 특히 Frequency Band Attention Module을 통해 EEG 정보를 정확하고 효율적으로 인코딩합니다.

- **Technical Details**: EEG-EMG FAConformer는 두 개의 주요 부문으로 나뉘어져 있습니다: EMG 섹션은 resblocks를 통해 정보를 인코딩하며, EEG 섹션은 EEG-Denoise 모듈과 독립적인 채널 특이 컨볼루션 모듈(Independent Channel-Specific Convolution Module, ICSC)로 구성됩니다. 이 연구에서 개발된 Multi-Scale Fusion Module과 Fuse Module은 관련 없는 정보를 효과적으로 제거하고 숨겨진 패턴을 완전히 활용하여 motor pattern recognition 성능을 극대화합니다.

- **Performance Highlights**: 광범위한 실험을 통해 EEG-EMG FAConformer는 Jeong2020 데이터셋에서 기존 방법들을 초월하는 뛰어난 성능과 높은 강건성 및 인상적인 안정성을 보여줍니다. 결과적으로 제안된 방법은 motor pattern recognition의 여러 최신 기법들(SOTA)과 비교하여 우수성을 입증하였습니다.



### Early Joint Learning of Emotion Information Makes MultiModal Model Understand You Better (https://arxiv.org/abs/2409.18971)
- **What's New**: 이번 논문에서는 Multimodal Emotion Recognition Challenge (MER2024)의 하위 과제에 대한 감정 인식 솔루션을 제시합니다. 오디오와 텍스트 간의 모드 경쟁 문제를 해결하기 위해 대형 언어 모델을 기반으로 한 초기 융합 전략을 채택하였으며, 데이터 부족과 클래스 불균형 문제를 해결하기 위해 다중 모델 투표를 사용하였습니다. 이 방법으로 MER2024-SEMI와 MER2024-NOISE 두 트랙에서 2위를 기록했습니다.

- **Technical Details**: 제안된 모델은 Vision Transformer (ViT)를 기반으로 하여 감정 특징을 효과적으로 표현합니다. 오디오와 텍스트의 조기 융합을 통해 두 모달리티 간의 정보 손실을 방지하며, 교차 모달 주의 메커니즘을 이용하여 다양한 소스에서의 주요 특징을 동적으로 강조함으로써 감정 인식 성능을 강화합니다. 또한, 오디오 데이터의 품질 향상을 위해 스피치 소스 분리 기술을 사용합니다.

- **Performance Highlights**: 모델은 MER2024의 두 트랙에서 2위로 평가되었으며, 이는 제안된 방법의 효과성을 검증합니다. 또한, 노이즈가 포함된 환경에서도 안정성과 정확성을 보장하는 방법을 포함하여, 라벨이 없는 데이터를 효과적으로 활용하는 순환 부스팅 데이터 마이닝 방법을 도입하여 모델의 일반화 능력을 향상시켰습니다.



### Integrating SPARQL and LLMs for Question Answering over Scholarly Data Sources (https://arxiv.org/abs/2409.18969)
Comments:
          Scholaly Hybrid Question answering challenge from the International Semantic Web Conference of 2024(ISWC), 6 pages, 3 figures

- **What's New**: 이번 논문은 2024년 국제 시맨틱 웹 컨퍼런스(ISWC)에서 주최되는 Scholarly Hybrid Question Answering over Linked Data (QALD) 챌린지에 대한 새로운 접근 방식을 소개합니다. SPARQL 쿼리, divide and conquer 알고리즘, 그리고 BERT 기반 모델을 통합하여 다양한 학술 출처에 대한 질문에 답변하는 시스템을 개발했습니다.

- **Technical Details**: 제안된 방법론은 SPARQL 쿼리를 통해 데이터를 수집하고, divide and conquer 알고리즘을 적용하여 다양한 질문 유형 및 출처를 관리하며, BERT를 이용하여 저자 관련 질문에 대한 정확한 답변을 생성하는 방식으로 구성되어 있습니다. 이 과정에서 SPARQL 쿼리 실행, 데이터를 정리하고, 질문을 정교하게 삭감하는 과정을 포함합니다. 최종적으로는 LLM(대형 언어 모델)을 사용하여 예측을 수행합니다.

- **Performance Highlights**: 제안한 방법은 Exact Match 및 F-score 메트릭을 사용하여 평가되었으며, 다양한 학술 데이터 출처에서 정확한 질문 응답을 제공하는 데 큰 개선을 보였습니다. 특히, BERT와 DPR(Dual-Product Retrieval) 알고리즘의 조합이 DBLP 지식 그래프에서 엔터티 및 관계 추출의 정확도를 크게 향상시켰습니다.



### Safety challenges of AI in medicin (https://arxiv.org/abs/2409.18968)
- **What's New**: 최근 인공지능(AI) 및 딥 러닝, 대형 언어 모델(LLM)에서의 발전은 의료 분야에 통합되는 속도를 빠르게 하고 있지만, 안전한 적용에 대한 우려도 증가하고 있습니다. 본 리뷰는 의료에서의 AI 안전성 문제를 다루고, LLM의 특화된 안전 문제를 탐구합니다.

- **Technical Details**: 의료에서 AI의 적용에서 나타나는 주요 문제는 신뢰성(reliability)과 정렬(alignment)입니다. 신뢰성 문제에는 데이터 조화(data harmonization), 일관된 성능, 모델 보정(calibration), 일반화(generalization), 편향(bias) 등이 포함됩니다. AI 정렬은 AI가 인간이 정의한 목표에 맞게 작용하는 것을 보장하는 것으로, 의사결정에서 목표의 잘못된 지정(mis-specification) 문제를 다루고 있습니다.

- **Performance Highlights**: AI 모델의 성과는 다양한 인구집단에 대한 성능 저하, 모델 개발 및 배포 중 데이터 유출의 위험, 그리고 LLM의 한계(예: 복잡한 논리 처리)에 의해 영향을 받을 수 있으며, 이러한 문제들이 AI의 의료 분야 내 안전성 논의에서 중요한 역할을 합니다.



### Deep Model Predictive Optimization (https://arxiv.org/abs/2310.04590)
Comments:
          Main paper is 6 pages with 4 figures and 1 table. Code available at: this https URL

- **What's New**: 본 연구에서는 Deep Model Predictive Optimization (DMPO)이라는 새로운 방법론을 제안합니다. 이는 기존의 모델 예측 제어(MPC) 알고리즘을 개선하여, 실제 환경에서의 복잡하고 민첩한 로봇 행동을 지원합니다.

- **Technical Details**: DMPO는 최적화 알고리즘의 내부 루프를 경험을 통해 직접 학습하여 제어 문제의 필요에 맞게 조정됩니다. 이는 연속적인 재계획을 통해 톱니바퀴 상황에서도 robust한 성능을 유지할 수 있도록 돕습니다.

- **Performance Highlights**: DMPO는 실제 쿼드로터의 궤적 추적 작업에서 기존 MPC 알고리즘보다 성능을 27% 향상시킬 수 있으며, MFRL을 통해 훈련된 end-to-end 정책과 비교할 때 19% 더 나은 결과를 보여줍니다. 또한, DMPO는 메모리 사용량이 4.3배 적으면서도 이러한 이점을 달성할 수 있습니다.



### Morph-SSL: Self-Supervision with Longitudinal Morphing to Predict AMD Progression from OC (https://arxiv.org/abs/2304.08439)
- **What's New**: 이 연구는 중간 단계 나이 관련 황반 변성(iAMD)에서 신생 혈관형 나이 관련 황반 변성(nAMD)으로의 전환을 예측하기 위한 새로운 Deep Learning (DL) 모델인 Morph-SSL을 개발했습니다. 기존의 신뢰할 수 있는 바이오마커의 부족으로 이러한 예측이 어려운 문제를 해결하고자 합니다.

- **Technical Details**: Morph-SSL은 Self-supervised Learning (SSL) 방법으로, 서로 다른 방문 시기의 비표기 OCT 스캔 쌍을 사용합니다. 이 방식은 이전 방문의 스캔을 다음 방문으로 변형하는 과정을 포함하며, Decoder가 이를 예측합니다. 이 모델은 연속적인 특성의 매니폴드(manifold)를 보장하여 방문 간의 중간 스캔을 생성할 수 있도록 선형 보간(linear interpolation)을 사용합니다. 그 후, Morph-SSL로 학습된 특성은 분류기를 통해 전환 시점에 대한 누적 확률 분포를 모델링합니다.

- **Performance Highlights**: Morph-SSL은 399개의 눈의 비표기 스캔(3570 방문)에 대해 학습되었으며, 343개의 눈에서 변환 날짜에 대한 임상 레이블이 있는 2418개의 스캔을 사용하여 5배 교차 검증(five-fold cross-validation)을 통해 평가되었습니다. Morph-SSL 특성은 향후 6개월 내 nAMD로의 전환을 예측하는 데 AUC 0.766을 달성하여, 기존의 SSL 방법으로 사전 학습되거나 처음부터 끝까지 학습된 동일한 네트워크보다 뛰어난 성능을 보였습니다.



### Backdoor Attacks for LLMs with Weak-To-Strong Knowledge Distillation (https://arxiv.org/abs/2409.17946)
- **What's New**: 이 연구는 Parameter-Efficient Fine-Tuning (PEFT)를 기반으로 한 새로운 백도어 공격 알고리즘인 W2SAttack을 제안합니다. 이 알고리즘은 작은 모델에서 백도어 기능을 전이하여 큰 모델에서 공격 효과를 극대화합니다.

- **Technical Details**: W2SAttack은 feature alignment-enhanced knowledge distillation을 활용하여 작은 언어 모델(teacher model)에서 큰 모델(student model)로 백도어를 전달합니다. 이는 PEFT를 사용하여 매개변수 업데이트를 최소화하고, 트리거와 목표 레이블 간의 정렬을 강화합니다.

- **Performance Highlights**: W2SAttack은 다양한 언어 모델과 아키텍처에서 거의 100%의 공격 성공률을 달성했으며, 분류 성능을 유지하면서 PEFT에 대한 백도어 공격을 효과적으로 강화했습니다.



New uploads on arXiv(cs.LG)

### Empirical Perturbation Analysis of Linear System Solvers from a Data Poisoning Perspectiv (https://arxiv.org/abs/2410.00878)
Comments:
          18 pages

- **What's New**: 이 논문은 머신 러닝 환경에서 일반적으로 발생하는 선형 해법에 대한 섭동 분석을 통해 데이터 중독 공격(data poisoning attack)의 관점에서 이러한 분석을 재구성합니다. 이를 통해 보다 강력한 선형 해법 개발에 기여하고자 하며, 입력 데이터의 오류가 알고리즘의 적합 오류와 정확성에 어떻게 영향을 미치는지를 조사합니다.

- **Technical Details**: 연구에서는 두 가지 데이터 섭동 방식을 제안했습니다: Label-guided Perturbation (LP) 및 Unconditioning Perturbation (UP)입니다. LP는 레이블 정보를 활용하여 자료를 섭동하는 방식이고, UP는 조건화를 풀어내는 방법입니다. 이를 통해 선형 시스템을 다루는 여러 직접 법(direct method) 및 반복 법(iterative method) 솔버의 성능을 분석합니다.

- **Performance Highlights**: 연구 결과, UP가 직접 솔버에서 섭동된 데이터의 활용도를 저하시킬 때 더 효과적이며, LP는 반복 솔버에 부정적인 영향을 준다는 것을 확인했습니다. 또한 대부분의 반복 솔버는 이러한 섭동에 의해 영향을 받으며 수렴 속도가 저하된다는 결과를 도출했습니다.



### Replacing Paths with Connection-Biased Attention for Knowledge Graph Completion (https://arxiv.org/abs/2410.00876)
- **What's New**: 본 논문에서는 Knowledge Graph (지식 그래프) 완성을 위한 새로운 접근법, Connection-Biased Link Prediction (CBLiP) 모델을 소개합니다. 이 모델은 경로 인코딩 (path encoding) 모듈 없이 Transformer 기반의 서브그래프 인코딩 (subgraph encoding) 모듈을 사용하여, 훈련 때 보지 못한 엔티티에 대한 추론을 가능하게 합니다.

- **Technical Details**: CBLiP는 연결 편향 주의 (connection-biased attention) 및 엔티티 역할 임베딩 (entity role embeddings)을 도입하여, 비싼 경로 인코딩 모듈을 대체합니다. 기존의 그래프 합성 방법과 달리, CBLiP는 수학적 모델링과 비용이 많이 드는 하이퍼파라미터 최적화를 요구하지 않습니다.

- **Performance Highlights**: 표준 인덕티브 KG 완성 벤치마크 데이터셋에서 CBLiP는 경로 정보를 사용하지 않는 모델들보다 우수한 성능을 보여줍니다. 또한, 경로 정보를 사용하는 모델들과 비교할 때 CBLiP는 경쟁력 있는 성능을 발휘하며 속도 또한 더 빠릅니다.



### Review of blockchain application with Graph Neural Networks, Graph Convolutional Networks and Convolutional Neural Networks (https://arxiv.org/abs/2410.00875)
- **What's New**: 이 논문은 Graph Neural Networks (GNNs), Graph Convolutional Networks (GCNs), 및 Convolutional Neural Networks (CNNs)의 블록체인 기술에서의 응용 사례를 살펴봅니다. 블록체인 네트워크의 복잡성과 채택이 증가함에 따라, 전통적인 분석 방법이 분산 시스템의 복잡한 관계와 동적 행동을 포착하는 데 한계를 드러내고 있습니다.

- **Technical Details**: GNNs와 GCNs는 블록체인 노드와 거래의 관계 데이터를 모델링하는 데 뛰어난 성능을 발휘하여 사기 탐지, 거래 검증, 스마트 계약 분석과 같은 응용 분야에 적합합니다. CNNs는 블록체인 데이터를 구조화된 행렬로 표현할 때 적응할 수 있으며, 거래 흐름의 숨은 시간적 및 공간적 패턴을 분석하는 데 유용합니다. 이들 모델은 선형 블록체인 및 Directed Acyclic Graph (DAG) 기반 시스템 모두에서 효율성, 보안 및 확장성을 향상시키는 데 기여합니다.

- **Performance Highlights**: 딥러닝 모델의 통합을 통해 블록체인 분석을 혁신하고, 더 정교한 분산 애플리케이션 및 개선된 네트워크 성능을 위한 가능성을 보여주는 것을 목표로 합니다.



### Fine-Grained Gradient Restriction: A Simple Approach for Mitigating Catastrophic Forgetting (https://arxiv.org/abs/2410.00868)
- **What's New**: 이 논문에서는 지속적인 학습(continual learning)에서 흔히 간과되는 하이퍼파라미터인 메모리 강도(memory strength)를 분석하고, 이를 통해 모델 파라미터의 업데이트 방향을 제약하여 경험적 성능을 크게 향상시키는 방법을 제안합니다.

- **Technical Details**: Gradient Episodic Memory (GEM) 방법을 기반으로, 우리는 업데이트 방향을 보다 유연하게 제약하는 두 가지 접근 방식을 제안합니다. 또한, 더 많은 제약 조건을 갖는 최적화 문제를 근사적으로 해결할 수 있는 계산 효율적인 방법도 제시합니다. 실험에서 MNIST, Split CIFAR100 등 여러 벤치마크를 사용하여 성능을 평가하였습니다.

- **Performance Highlights**: 우리가 제안한 방법은 메모리 강도를 사용할 때보다 오래된 지식 유지(old knowledge)와 새로운 지식 학습(new knowledge) 사이에서 더 균일하게 좋은 Pareto Frontiers를 달성했습니다. 또한, 기존 지속적인 학습 문헌에서 사용되었던 전통적인 메트릭들을 사용하여 성능을 평가하였습니다.



### Timber! Poisoning Decision Trees (https://arxiv.org/abs/2410.00862)
Comments:
          18 pages, 7 figures, 5 tables

- **What's New**: Timber는 의사결정 나무를 겨냥한 최초의 white-box poisoning attack입니다. 이 공격은 greedy attack 전략을 기반으로 하여 sub-tree retraining을 활용하여 주어진 training instance를 변조했을 때의 손해를 효율적으로 추정합니다.

- **Technical Details**: Timber는 tree annotation 절차에 기반하여 training instance를 sub-tree retraining의 계산 비용이 증가하는 순서로 정렬합니다. 이러한 정렬 방식은 대용량 데이터셋에서 poisoning attacks을 보다 효율적이고 실현 가능하도록 만드는 early stopping criterion을 지원하는 Timber의 변형을 낳습니다. 또한, 의사결정 나무를 결합하여 예측력을 높이는 전통적인 random forest 모델에 대한 Timber의 확장도 논의합니다.

- **Performance Highlights**: 실험 평가 결과, Timber 공격이 기존 baseline보다 효과성과 효율성 면에서 우수한 성과를 보여주었습니다. 두 가지 대표적인 방어 기법이 공격의 영향을 완화할 수는 있지만, 효과적으로 저지하는 데에는 실패함을 보여줍니다.



### Uncertainty-aware Reward Model: Teaching Reward Models to Know What is Unknown (https://arxiv.org/abs/2410.00847)
- **What's New**: 이 논문에서는 Uncertain-aware RM (URM) 및 Uncertain-aware RM Ensemble (URME)을 제안하여 보상 모델링에서 불확실성을 처리합니다. 기존의 보상 모델(RM)이 인간의 선호도 내의 불확실성을 충분히 캡처하지 못하는 문제를 해결하고자 합니다.

- **Technical Details**: URM은 인간 선호도의 여러 속성의 분포를 모델링하는 불확실성 인식 value head를 갖추고 있으며, URME는 앙상블 내의 불일치를 통해 불확실성을 정량화합니다. 이를 통해 URM과 URME는 보상 평가 중에 고유한 지식 부족을 식별하는 데 도움을 줍니다.

- **Performance Highlights**: URM은 8B 모델 크기에서 RewardBench에서 최첨단 성능을 달성하며, 이는 기존 모델들을 초월합니다. URM과 URME는 불확실성 정량화를 통해 보상 평가의 신뢰성을 크게 향상시킵니다.



### Learning Stochastic Dynamics from Snapshots through Regularized Unbalanced Optimal Transpor (https://arxiv.org/abs/2410.00844)
- **What's New**: 이 논문에서는 정규화된 비균형 최적 수송 (RUOT) 문제를 해결하고, 관측된 샘플로부터 연속적인 비균형 확률 동역학을 추론하기 위해 새로운 딥 러닝 방법인 DeepRUOT를 소개합니다.

- **Technical Details**: DeepRUOT는 사전 지식 없이도 데이터를 직접 학습할 수 있도록 설계되었습니다. 복잡한 분포를 단순한 잠재공간으로 매핑하는 기존의 변분 오토인코더 (Variational Autoencoders) 모델의 방식을 개선하며, 슈뢰딩거 다리 문제 (Schrödinger bridge problem)와의 연결성을 탐구합니다.

- **Performance Highlights**: DeepRUOT는 신경망을 통해 성장 및 전이 패턴을 정밀하게 식별하고, 가짜 전이를 제거하며 Waddington 개발 경관을 구축하는 데 효과적임을 보입니다.



### Towards Fairness and Privacy: A Novel Data Pre-processing Optimization Framework for Non-binary Protected Attributes (https://arxiv.org/abs/2410.00836)
Comments:
          The Version of Record of this contribution is published in Data Science and Machine Learning, volume 1943, CCIS (Springer Singapore) 2023. It is available online at this https URL

- **What's New**: 이 논문은 비즈니스 및 사회적 문제에서 AI의 불공정한 결과를 해결하기 위한 새로운 프레임워크를 제안합니다. 특히, 비이진 보호 속성(non-binary protected attribute)을 포함한 데이터 세트의 편향을 제거하는 방법에 중점을 두었습니다.

- **Technical Details**: 제안된 프레임워크는 조합 최적화(combinatorial optimization) 문제를 수립하며, 유전 알고리즘(genetic algorithms)과 같은 휴리스틱(heuristics)을 사용하여 공정성 목표를 해결할 수 있도록 합니다. 데이터 세트에서 특정 차별 측정(discrimination measure)을 최소화하는 데이터 하위 집합(subset)을 찾아내는 방식을 채택하고 있습니다.

- **Performance Highlights**: 종합 평가 결과, 유전 알고리즘을 사용한 경우 원본 데이터에 비해 훨씬 더 공정한 데이터를 생성할 수 있음을 보여주었습니다. 이 프레임워크는 유연성이 뛰어나고, 다양한 공정성 목표를 지원하며, 정보 보호(data privacy)에도 유리한 특성을 가지고 있습니다.



### Targeted synthetic data generation for tabular data via hardness characterization (https://arxiv.org/abs/2410.00759)
- **What's New**: 본 논문은 고퀄리티 데이터가 부족한 상황에서 합성 데이터 생성이 모델 성능 및 강건성을 개선할 수 있다는 가능성을 제시합니다. 특히, 훈련 데이터 중 '어려운' 데이터를 선택적으로 증강하여 모델의 일반화 능력을 향상시키는 새로운 파이프라인을 소개합니다.

- **Technical Details**: 제안된 접근법은 크게 두 단계로 나뉩니다: (i) 훈련 데이터 포인트의 난이도 특성을 파악하고, (ii) 가장 어려운 데이터 포인트에 대해서만 합성 데이터 생성 모델을 학습시키는 것입니다. 난이도 특성을 파악하기 위해 KNN Shapleys를 활용하며, 이를 통해 기존의 방법론들과 비교해도 효과적으로 난이도를 감지할 수 있습니다. 합성 데이터 생성에는 Tabular Variational Autoencoders (TVAE)와 Conditional Tabular Generative Adversarial Networks (CTGAN)을 사용했습니다.

- **Performance Highlights**: 실험 결과, 난이도가 높은 데이터 포인트만을 사용한 합성 데이터 생성이 비대상(non-targeted) 데이터 증강 방식보다 성능 개선에서 더 큰 효과를 보였으며, 계산적으로도 더 효율적이라는 것을 확인했습니다.



### Show Me What's Wrong: Combining Charts and Text to Guide Data Analysis (https://arxiv.org/abs/2410.00727)
- **What's New**: 이 연구는 다차원 데이터셋에서 이상 탐지를 용이하게 하기 위해 자동화된 정보 강조, LLM(대형 언어 모델)이 생성한 텍스트 통찰 및 시각적 분석을 결합한 도구를 제안합니다. 이를 통해 사용자는 여러 세부 수준에서 탐색할 수 있게 됩니다.

- **Technical Details**: 이 시스템은 데이터 분석 영역에 따라 데이터를 세분화하여 각각의 영역을 시각적으로 표현하고, 더 많은 주의가 필요한 영역을 자동으로 신호합니다. 사용자가 특정 영역을 선택하면 시스템은 해당 영역에 대한 텍스트 및 그래픽 요약을 제공합니다. 이 과정에서 Hallucination Detection 시스템을 포함하여 생성되는 텍스트의 정확성을 높이는 방법을 제공합니다.

- **Performance Highlights**: 연구에 참여한 7명의 분야 전문가의 피드백에 따르면, 이 도구는 탐색적 분석을 효과적으로 지원하고 가이드를 제공하며, 의심스러운 정보를 식별하는 데 도움을 줍니다. 사용자가 각 Knowledge Area(KA)에 대한 중요한 정보를 쉽게 요약하고, 비정상 활동을 효과적으로 발견할 수 있도록 돕는 것으로 나타났습니다.



### On the Geometry and Optimization of Polynomial Convolutional Networks (https://arxiv.org/abs/2410.00722)
- **What's New**: 본 논문에서는 다항식 활성화 함수를 가진 합성곱 신경망(Convolutional Neural Networks, CNN)의 신경다양체(neuromanifold)에 대한 기하학적 특성을 연구합니다. 특히 파라미터화 지도(parameterization map)가 정규적(regular)이며 거의 모든 곳에서 동형사상(isomorphism)임을 증명하였습니다.

- **Technical Details**: 논문은 대수기하학(algebraic geometry) 도구를 활용하여 다항식 CNN의 신경다양체를 분석합니다. 주요 결과는 필터의 스케일링을 제외하면 신경다양체의 파라미터화가 정규적이며 거의 모든 곳에서 함수와 일대일 대응을 이룬다는 것입니다. 또한, 신경다양체의 차원(dimension)과 차수(degree)를 확인하였으며, 차원은 층 수에 비례하여 선형적으로 증가하고 차수는 초지수적으로 증가하는 경향을 보입니다.

- **Performance Highlights**: 본 연구는 대규모 일반 데이터셋에 대해 제곱 오차 회귀 손실을 외부 다항식 함수에 대한 거리 최소화 문제로 재구성하는 방법을 제공합니다. 심지어 대다수의 데이터셋에 대해 신경다양체 위의 임계점(critical point) 수가 유한하며 데이터셋의 크기와 무관하다는 점이 강조되었습니다. 이 결과는 최적화 과정에서 신경다양체의 기하학적 구조가 중요한 역할을 수행함을 나타냅니다.



### Pseudo-Non-Linear Data Augmentation via Energy Minimization (https://arxiv.org/abs/2410.00718)
- **What's New**: 이 논문에서는 에너지 기반 모델링(energy-based modeling)과 정보 기하학(information geometry) 원리를 기반으로 한 새로운 데이터 증강(data augmentation) 방법을 제안합니다. 이 방법은 블랙박스 생성을 통한 모델 대신 명시적이고 이론적으로 기초가 있는 변환을 사용하여 해석 가능성과 강력한 보장을 확보합니다.

- **Technical Details**: 제안하는 방법의 핵심은 새로운 데이터를 생성하기 위해 차원 축소를 역으로 적용하는 백워드 프로젝션 알고리즘(backward projection algorithm)을 도입했습니다. 이 알고리즘은 낮은 차원 잠재 표현 공간에서 새로운 점을 주어진 경우, 원래 데이터의 k-가까운(latent representations) 잠재 표현을 정정하여 특정 서브스페이스(target subspace)를 설정하고 이를 통해 데이터를 다시 프로젝션합니다.

- **Performance Highlights**: 실험 결과, 제안하는 데이터 증강 방법이 자동 인코더(autoencoder)와 같은 블랙박스 생성 모델에 대해 경쟁력 있는 성능을 달성했으며, 간단하고 투명하며 해석 가능한 알고리즘을 통해 이는 해석 가능성을 강조합니다.



### Contrastive Abstraction for Reinforcement Learning (https://arxiv.org/abs/2410.00704)
- **What's New**: 이 연구에서는 'contrastive abstraction learning'이라는 새로운 방법론을 제안하여 보상 없이도 상태 공간에서 추상화(abstraction)를 학습합니다. 이 방식은 자연어 처리와 동영상 데이터에 실질적인 성과를 내고 있는 대조적 학습(contrastive learning)과 현대의 Hopfield 네트워크(modern Hopfield networks, MHN)를 사용하여 효율적인 강화 학습을 가능하게 합니다.

- **Technical Details**: 대조적 추상화 학습의 첫 번째 단계는 인접한 상태를 유사한 표현으로 매핑하기 위해 대조적 학습을 사용하는 self-supervised learning입니다. 두 번째 단계에서는 MHN을 이용해 비슷한 상태 표현을 동일한 고정점(fixed point)에 맵핑하여 추상 상태를 형성합니다. 추상화의 수준은 MHN의 온도 매개변수(temperature parameter)를 조절함으로써 변경할 수 있습니다.

- **Performance Highlights**: 실험 결과, 대조적 추상화 학습은 다양한 하위 작업(downstream tasks)에서 강화 학습의 효율성을 향상시키는 데 효과적이라는 점이 입증되었습니다. 이를 통해 보상 없이도 다양한 환경에서 추상 상태를 효과적으로 학습하여 효율성을 높였습니다.



### Investigating the Impact of Model Complexity in Large Language Models (https://arxiv.org/abs/2410.00699)
- **What's New**: 본 논문에서는 Large Language Models (LLMs)의 모델 복잡성이 fine-tuning 성능에 미치는 영향을 탐구하며, Hidden Markov Models (HMMs)를 사용하여 autoregressive LLMs를 모델링합니다. 특히, 'double descent' 현상을 분석하여 모델 복잡성이 증가함에 따라 리스크가 어떻게 변화하는지를 설명합니다.

- **Technical Details**: 이 연구는 다음 단어 예측 리스크(next-word prediction risk)와 모델 복잡성 간의 관계를 심도 있게 분석합니다. 또한, head tuning 방법을 사용하며, 이 과정에서 모든 사전 학습된 파라미터는 고정되고 특정한 heads만 훈련됩니다. HMM 사용을 통해 모델을 autoregressive로 접근하여 리스크 추정을 수행합니다.

- **Performance Highlights**: 리스크 분석 결과, 모델 복잡성이 증가함에 따라 리스크가 먼저 증가한 후 감소하는 'double descent' 패턴이 나타났습니다. 이러한 분석은 향후 LLM의 최적 모델 크기 선택에 유용한 통찰을 제공합니다.



### Beyond Minimax Rates in Group Distributionally Robust Optimization via a Novel Notion of Sparsity (https://arxiv.org/abs/2410.00690)
Comments:
          38 pages

- **What's New**: 이 논문에서는 group distributionally robust optimization (GDRO)의 minimax 샘플 복잡성을 $	ext{log}(K)$ 인자까지 결정한 기존 연구를 넘어서 새로운 스파시티 개념인 $(	heta, eta)$-sparsity를 소개합니다.

- **Technical Details**: $(	heta, eta)$-sparsity 조건에 따르면, 어떤 파라미터 $	heta$에서 최대 $eta$ 개의 그룹이 다른 그룹의 위험보다 적어도 $	heta$ 만큼 더 큰 위험을 갖는다는 의미입니다. 새로운 알고리즘을 통해 샘플 복잡성에서 $	ext{K}$에 대한 선형 의존성을 $eta$, 즉 적어도 훨씬 작은 수로 대체할 수 있음을 보여줍니다. 이는 sleeping bandits에서의 최신 발전을 활용하여 GDRO의 두 명의 플레이어 제로섬 게임 최적화 프레임워크와 행동별 후회 경계(per-action regret bounds) 간의 근본적인 연결을 나타냅니다.

- **Performance Highlights**: 제시된 알고리즘은 $	ext{log}$ 인자까지 최적의 $(	heta, eta)$-sparsity 조건에 적응하는 샘플 복잡성을 성취할 수 있으며, 특정 $	heta$에 대한 입력으로부터 차원 독립적인 샘플 복잡성 결과를 도출하는 방법을 보여줍니다.



### Stabilizing the Kumaraswamy Distribution (https://arxiv.org/abs/2410.00660)
- **What's New**: 본 논문에서는 Kumaraswamy (KS) 분포의 로그 확률 밀도 함수(log-pdf)와 역 누적 분포 함수(inverse CDF)의 수치적 불안정성을 해결하여 PyTorch 및 TensorFlow와 같은 라이브러리의 문제를 드러냅니다. 그 결과, 안정화된 KS 분포를 중심으로 한 새로운 스케일 가능한 잠재 변수 모델을 제안합니다.

- **Technical Details**: Kumaraswamy 분포는 reparameterization trick을 지원하며, 효율적인 샘플링과 저분산 그래디언트 추정이 가능할 뿐만 아니라, 복잡한 잠재 공간을 캡처할 수 있는 충분한 표현력을 제공합니다. 이 모델은 Variational Bandit Encoder (VBE) 및 Variational Edge Encoder (VEE)를 포함하여 다수의 응용 프로그램에서 탐색-착취(trade-off) 문제를 해결하도록 설계되었습니다.

- **Performance Highlights**: 우리는 KS 분포가 추천 시스템, 강화 학습 및 네트워크 분석과 같은 대규모 잠재 변수 모델의 주요 구성 요소로, 새로운 응용 프로그램을 여는 데 기여한다고 주장합니다. 특히, KS 분포는 대량 데이터를 처리할 때 성능 향상과 간단한 모델링을 가능하게 합니다.



### AutoTM 2.0: Automatic Topic Modeling Framework for Documents Analysis (https://arxiv.org/abs/2410.00655)
- **What's New**: 이번 연구에서는 additively regularized topic models를 최적화하기 위한 AutoTM 2.0 프레임워크를 소개합니다. 이전 버전과 비교하여 새로운 최적화 파이프라인, LLM 기반 품질 지표 및 분산 모드와 같은 중요 개선 사항이 포함되어 있습니다. AutoTM 2.0은 전문가는 물론 비전문가도 텍스트 문서를 처리하고 탐색적 데이터 분석을 수행하거나 해석 가능한 Feature 집합에 대한 군집 작업을 수행할 수 있도록 돕는 도구입니다.

- **Technical Details**: AutoTM 2.0은 additively regularized topic models를 효과적으로 사용할 수 있도록 설계되었습니다. 이 프레임워크는 자동 단일 목적 최적화 절차를 제공하며, 인간의 판단과 밀접하게 일치하는 메트릭을 제안합니다. 또한 비용 효율적인 추론과 대규모 텍스트 코퍼스를 위한 신속한 학습이 가능합니다. Python 라이브러리를 제공하며, 대규모 실험이나 대량 데이터 세트를 관리하는 데 유용합니다.

- **Performance Highlights**: AutoTM 2.0은 5개의 다양한 Feature를 가진 데이터 세트를 사용하여 이전 AutoTM보다 더 나은 성능을 달성했습니다. 이 프레임워크는 하이퍼파라미터 튜닝과 관련하여 새로운 유전 알고리즘 및 베이지안 최적화 방법을 통합하여 실제 사용 사례에 더 쉽게 적용될 수 있도록 개선되었습니다.



### ICL-TSVD: Bridging Theory and Practice in Continual Learning with Pre-trained Models (https://arxiv.org/abs/2410.00645)
Comments:
          45 pages, 19 figures, 14 tables (Preprint, Oct 1, 2024)

- **What's New**: 이 논문은 continual learning (CL)에서 이론과 실제 간의 간극을 메우기 위해 새로운 접근법인 ICL-TSVD를 제안합니다. ICL-TSVD는 Empirical 한 강력한 방식인 RanPAC을 원칙적인 프레임워크인 Ideal Continual Learner (ICL)와 통합하여 새로운 작업을 학습하면서 이전의 작업을 잊어버리지 않도록 설계되었습니다.

- **Technical Details**: ICL-TSVD는 사전 학습된 특징을 더 높은 차원 공간으로 변환한 후 오버 파라미터화된 최소-norm least-squares 문제를 설정합니다. 이를 위해 singular value decomposition (SVD)의 값을 지속적으로 잘라내어 수치적 불안정성과 일반화 오류를 완화합니다. 이 방법은 하이퍼파라미터 선택에 대해 안정적이며 다수의 작업을 처리할 수 있는 성능의 지속성을 보장합니다.

- **Performance Highlights**: 실험 결과, ICL-TSVD는 여러 데이터셋에서 기존의 CL 방법들을 초월하는 성능을 보였고, 특히 CIL (Class-Incremental Learning) 설정에서 RanPAC보다 현저히 높은 성능을 기록했습니다. 이 방법이 안정적으로 작동하고 적절한 SVD 요소의 부분을 잘라낼 수 있다는 이론적 보장을 제공함으로써, 강력한 경험적 성능을 나타냅니다.



### Scaling Offline Model-Based RL via Jointly-Optimized World-Action Model Pretraining (https://arxiv.org/abs/2410.00564)
- **What's New**: 이번 연구에서는 JOWA(Jointly-Optimized World-Action model)라는 새로운 오프라인 모델 기반 강화학습 에이전트를 소개합니다. 이 모델은 여러 아타리 게임에서 pretrained 되었으며, 일반적인 표현과 의사결정 능력을 학습하여 새로운 작업에 대한 일반화 능력을 높였습니다.

- **Technical Details**: JOWA는 공유된 transformer backbone을 통해 월드 액션 모델을 공동 최적화하며, 이는 대규모 모델의 시간 차이 학습(TD learning)을 안정화합니다. 또한, Q-value 추정 오류를 보상하기 위해 효율적이고 병렬 처리 가능한 계획 알고리즘을 제안합니다.

- **Performance Highlights**: JOWA는 1억 5천만 개의 파라미터를 가진 가장 큰 모델이 10% 하위 샘플링된 오프라인 데이터만을 사용하여 사전 훈련된 게임에서 78.9%의 인간 수준 성능을 달성했습니다. 이는 기존의 대규모 오프라인 RL 벤치마크보다 평균 31.6% 향상된 성과입니다.



### Best Practices for Multi-Fidelity Bayesian Optimization in Materials and Molecular Research (https://arxiv.org/abs/2410.00544)
- **What's New**: 이 연구에서는 다중 신뢰도 베이지안 최적화(Multi-fidelity Bayesian Optimization, MFBO)의 적용 가능성을 실험적 환경에서 결정할 수 있는 가이드라인과 추천을 제공하고 있습니다. 특히, 화학 문제에 MFBO 방법을 적용한 최초의 체계적인 평가를 수행했습니다.

- **Technical Details**: MFBO는 여러 신뢰도 수준에서의 정보 소스를 활용하여 블랙박스 문제를 최적화하는 접근 방식입니다. 이 연구에서는 두 가지 신뢰도 수준을 고려하여, 조정 부분에서 지출되는 비용과 정보를 동시에 분석하는 모델을 구현하였습니다. Gaussian Process (GP)를 사용하여 서브모델을 구축하며, 다양한 획득 함수(acquisition function) 간의 성능을 비교하고 있습니다.

- **Performance Highlights**: 실험 결과, MFBO는 단일 신뢰도 접근 방식에 비해 실질적으로 더 나은 성과를 보여주었으며, 이는 MFBO가 화학 과학에서 일상적인 도구로 자리 잡을 수 있는 가능성을 제시합니다. 다양한 발견 문제에서 MFBO를 벤치마킹하고, 그 효과성을 강조했습니다.



### Differentially Private Active Learning: Balancing Effective Data Selection and Privacy (https://arxiv.org/abs/2410.00542)
- **What's New**: 이 연구에서는 표준 학습 환경에서 차등 개인정보 보호(differential privacy, DP)와 능동 학습(active learning, AL)을 통합하려는 기존의 미비점을 해결한 차등 개인정보 보호 능동 학습(DP-AL)을 소개합니다.

- **Technical Details**: DP-AL은 배치 생성 시 개별 샘플링 확률을 활용하여 데이터 포인트 참여를 극대화하는 스텝 증폭(step amplification) 기법을 제안합니다. 이는 프라이버시 예산 할당과 데이터 활용의 주요 과제를 극복하는 데 목적이 있습니다.

- **Performance Highlights**: 비전 및 자연어 처리 과제를 통한 실험 결과, DP-AL은 특정 데이터 세트와 모델 아키텍처에서 성능을 향상시킬 수 있음을 보여줍니다. 그러나 프라이버시가 제한된 환경에서 능동 학습의 한계와 프라이버시, 모델 정확성, 데이터 선택 정확성 간의 무역 오프(trade-offs)에 대한 중요성을 강조합니다.



### Optimal Causal Representations and the Causal Information Bottleneck (https://arxiv.org/abs/2410.00535)
Comments:
          Submitted to ICLR 2025. Code available at this http URL

- **What's New**: 이번 연구에서는 전통적인 정보 병목 (Information Bottleneck) 방법의 한계를 극복하기 위해 원인론적 정보 병목 (Causal Information Bottleneck, CIB) 방법을 제안합니다. 이 방법은 주어진 변수 세트를 압축하면서도 대상 변수에 대한 원인 통제를 유지할 수 있도록 설계되었습니다.

- **Technical Details**: Causal Information Bottleneck (CIB) 메소드는 입력 변수 X와 지정된 타겟 변수 Y 사이의 인과 관계를 유지하면서 X의 압축을 극대화 하는 방법입니다. 이를 위해 제안된 CIB Lagrangian은 특정 하이퍼파라미터 β를 통하여 인과적 통제를 조절할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, CIB 방법으로 학습된 표현은 원래 의도한 대로 인과 관계를 정확하게 포착함을 보여주었습니다. 이는 기존의 통계적 방법들이 가지는 한계를 보완하며, 인과적 통찰력을 요구하는 작업에 더욱 적합할 것으로 기대됩니다.



### Deep Model Interpretation with Limited Data : A Coreset-based Approach (https://arxiv.org/abs/2410.00524)
- **What's New**: 이 논문에서는 모델 해석(Method Interpretation)의 계산 비용 문제를 해결하기 위해 코어셋 기반 프레임워크를 제안합니다. 이를 통해 대규모 데이터셋에서 대표적인 서브셋을 샘플링하여 해석 작업을 수행할 수 있게 됩니다.

- **Technical Details**: 코어셋(selection: coreset) 선택 방법을 이용하여 대규모 데이터셋에서 대표 샘플을 선택하고, 이를 통해 프리트레인(pre-trained) 모델의 관련 특징을 식별합니다. 유사성 기반 평가 프로토콜(similarity-based evaluation protocol)이 도입되어 입력 데이터 양에 대한 모델 해석 방법의 강건성을 평가합니다.

- **Performance Highlights**: 실험 결과, 다양한 해석 방법과 DNN(Deep Neural Network) 모델, 코어셋 선택 방법을 고려한 결과 제안된 프레임워크의 효과성이 입증되었습니다. 대규모 데이터셋을 직접 사용하는 것보다 더 낮은 계산 비용으로 의미 있는 해석 결과를 제공할 수 있음을 보여주었습니다.



### Advancing RVFL networks: Robust classification with the HawkEye loss function (https://arxiv.org/abs/2410.00510)
- **What's New**: 본 논문에서는 RVFL(Random Vector Functional Link) 네트워크에 HawkEye loss(H-loss) 함수를 통합하여 아울라이어(outlier)와 노이즈(noise)에 대한 강인성을 개선하는 새로운 모델 H-RVFL을 제안합니다. 이는 기존 RVFL 구조에서 처음으로 경계(bounded) 손실 함수를 포함시키는 연구로, 머신 러닝 모델의 성능을 향상시키는 중요한 기회를 제공합니다.

- **Technical Details**: H-loss 함수는 smoothness(매끄러움)와 boundedness(경계성)을 갖추고 있으며, insensitivity zone(무관 영역)을 포함하고 있습니다. 이는 모델이 극단적인 오류에 대해 보다 강인해지고, 미세한 차이나 노이즈의 영향을 줄이는 데 도움이 됩니다. 또한, 제안된 H-RVFL 모델의 비볼록 최적화(non-convex optimization)는 Nesterov accelerated gradient(NAG) 알고리즘을 통해 효과적으로 해결됩니다.

- **Performance Highlights**: 실험은 UCI 및 KEEL 데이터셋에서 진행되었으며, H-RVFL 모델은 아울라이어와 노이즈가 있는 환경에서도 뛰어난 강인성과 효율성을 보였습니다. 제안된 모델은 기존 모델들에 비해 성능이 유의미하게 개선되어, 노이즈가 존재하는 실제 환경에서 강력한 도구로 자리 잡았습니다.



### Learning Personalized Treatment Decisions in Precision Medicine: Disentangling Treatment Assignment Bias in Counterfactual Outcome Prediction and Biomarker Identification (https://arxiv.org/abs/2410.00509)
Comments:
          9 pages, 5 figures, conference

- **What's New**: 이 논문은 정밀 의학(precision medicine)의 개인화된 치료 결정을 지원하기 위해 임상 관찰 데이터의 복잡한 편향(bias)과 생물학적 데이터의 고차원적 특성을 모델링합니다. 다양한 치료 할당 편향(causal treatment assignment bias)의 영향을 기계 학습(ML) 모델에 통합하여 카운터팩추얼 예측(counterfactual prediction) 및 바이오마커(biomarker) 식별에 미치는 영향을 분석합니다.

- **Technical Details**: 연구는 다양한 치료 할당 정책(treatment assignment policies)의 특성을 시뮬레이션하고, 관찰된 치료 정책(observed treatment policies)에 의해 유도된 여러 유형의 편향을 형식화 및 정량화합니다. 실험은 toy dataset, 반합성(semi-synthetic) 암 데이터, 그리고 실제 생물학적 결과를 사용하여 진행되었습니다. 이 논문은 생물학적 변이성과 복잡성을 반영하는 현실적인 벤치마크를 만들기 위해 실험 디자인을 제안합니다.

- **Performance Highlights**: 연구 결과는 특정 편향 유형이 모델 성능에 미치는 영향을 다르게 나타내며, 특히 결과 메커니즘과 관련 없는 편향은 예측 정확도에 미치는 영향이 미미함을 보여줍니다. 따라서, 임상 관찰 데이터의 특정 편향을 고려하는 것이 카운터팩추얼 ML 모델 개발에 있어 필수적임을 강조합니다.



### Enhancing Solution Efficiency in Reinforcement Learning: Leveraging Sub-GFlowNet and Entropy Integration (https://arxiv.org/abs/2410.00461)
- **What's New**: GFlowNet의 새로운 손실 함수와 훈련 목적 정제를 통해 후보 다양성과 계산 효율성을 크게 향상시키는 방법이 제안되었습니다.

- **Technical Details**: 연구는 GFlowNet의 손실 함수를 서브 GFlowNet 손실 함수로 분해하고 엔트로피를 손실 함수의 가중치 기준으로 통합하여 네트워크 구조 특성을 활용하는 새로운 접근 방식을 다룹니다. 이를 통해 후보 생성의 다양성과 효율성을 높입니다.

- **Performance Highlights**: 실험 결과, 제안된 GFlowNet이 기존 방법에 비해 향상된 수렴 속도와 보상 달성을 보여주었으며, 특히 2차원 그리드 실험에서 두드러진 성과를 보였습니다.



### UniAdapt: A Universal Adapter for Knowledge Calibration (https://arxiv.org/abs/2410.00454)
- **What's New**: 이 논문에서는 지속적인 모델 편집(lifelong model editing)에 필요한 새로운 접근법, UniAdapt를 소개합니다. UniAdapt는 지식 보정(knowledge calibration)을 위한 범용 어댑터로, 기존의 모델 편집 방법들이 겪던 일반화(generalization)와 로컬리티(locality) 간의 균형 문제를 해결하고자 합니다.

- **Technical Details**: UniAdapt는 Mixture of Experts (MoE) 아키텍처와 Retrieval-Augmented Generation (RAG) 기법을 활용합니다. 이 어댑터는 벡터 보조 라우터(vector-assisted router)를 통해 입력을 적절한 전문가(experts)에게 라우팅하며, 여러 샤드(shards)의 벡터 저장소를 유지하여 의미론적 유사도 기반의 라우팅 벡터를 구성합니다. UniAdapt는 모델이 독립적인 기능으로 설계되어 있어, 원래의 모델 가중치를 변경하지 않고도 새로운 지식을 삽입할 수 있습니다.

- **Performance Highlights**: 실험 결과, UniAdapt는 기존의 모델 편집 도구들에 비해 성능에서 상당한 개선을 보이며, 다양한 메트릭에서 뛰어난 결과를 달성했습니다. 특히, UniAdapt는 기억 능력과 일반화 능력을 모두 갖추고 있어, 지속적인 학습 작업에 적합한 선택으로 평가됩니다.



### EKAN: Equivariant Kolmogorov-Arnold Networks (https://arxiv.org/abs/2410.00435)
- **What's New**: 본 논문에서는 Equivariant Kolmogorov-Arnold Networks (EKAN)을 제안합니다. EKAN은 KANs에 matrix group equivariance를 통합하여 KANs의 적용 가능성을 보다 넓혀주고자 하며, 이는 기존의 MLPs에 대한 중요한 대안으로 작용할 것입니다.

- **Technical Details**: EKAN은 gated spline basis functions를 포함하고 있으며, 각 레이어는 equivariant linear weights와 함께 이루어집니다. 또한, lift layer를 정의하여 EKAN의 입력 공간과 데이터셋의 feature space를 정렬하여 전체 EKAN 아키텍처를 구성합니다. 이 과정에서 gated basis functions는 활성화 후의 공간과 입력 공간 간의 equivariance를 보장합니다.

- **Performance Highlights**: EKAN은 입자 산란(particle scattering) 및 세 물체 문제(three-body problem)와 같은 대칭 관련 작업에서 baseline 모델과 비교하여 더 높은 정확도를 기록하며, 작은 데이터셋이나 적은 파라미터로도 MSE(Mean Squared Error)를 몇 배 줄일 수 있습니다. 또한, 비기호적 공식 시나리오에서도 KANs보다 EMLP와 유사한 결과를 얻을 수 있습니다.



### Scalable Multi-Task Transfer Learning for Molecular Property Prediction (https://arxiv.org/abs/2410.00432)
- **What's New**: 본 논문은 다중 작업(multi-task) 분자(property) 예측을 위한 새로운 방법론인 데이터 기반(bi-level optimization) 최적화를 통해 전이 비율(transfer ratios)을 자동으로 산출하여 전이 학습의 효율성을 향상시킵니다.

- **Technical Details**: 전이 학습(transfer learning)은 출처(source) 작업 데이터에서 학습된 지식을 목표(target) 작업에 효과적으로 적용할 수 있게 해줍니다. GATE 알고리즘은 다양한 작업 간의 기하학적 정렬을 도입하여 다중 작업에서 전이 학습을 확장하였습니다. 이번 연구에서는 매개변수 조정 없이 그레이디언트 기반의 방법으로 최적의 전이 비율을 자동으로 탐색하게 됩니다.

- **Performance Highlights**: 제안된 방법은 40가지 분자 속성에 대한 예측 성능을 향상시키고, 다중 작업 전이 학습의 수렴(convergence) 속도를 가속화하였습니다.



### Metric-Based Few-Shot Learning for Exercise Repetition Counting with IMU Data (https://arxiv.org/abs/2410.00407)
- **What's New**: 이 연구는 IMU 신호를 분석하여 자동으로 운동 반복 횟수를 세는 방법을 개발하였고, 훈련 중 보지 못한 새로운 운동까지 포괄하는 범용 운동 반복 횟수 세기 작업에 초점을 맞추었습니다. 모델은 다양한 운동 유형 간의 피크 패턴 변동성을 처리하는 데 중점을 두었습니다.

- **Technical Details**: 본 연구는 심지어 새로운 운동도 인식할 수 있는 몇 가지 샷(shot) 분류 문제로 재정의된 횟수 세기 기법을 제안합니다. 이를 통해 모델은 훈련 중 보지 못한 운동의 반복 패턴을 탐지할 수 있습니다. 제안된 방법은 시암 네트워크(Siamese network)와 삼중 손실(triplet loss)을 사용하여 피크 프레임과 비피크 프레임을 구분할 수 있도록 임베딩 공간을 최적화합니다.

- **Performance Highlights**: 평가 결과, 제안된 방법은 28가지 운동에서 하나의 세트 내에서 10회 이상의 반복 횟수를 정확히 세울 확률이 86.8%로 나타났습니다. 이는 다양한 운동 유형을 아우르는 모델의 범용성과 적응력을 강조하며, 피트니스 및 헬스케어 애플리케이션에서의 실시간 구현에 강력한 후보로 평가됩니다.



### Revisiting Essential and Nonessential Settings of Evidential Deep Learning (https://arxiv.org/abs/2410.00393)
Comments:
          22 pages, under review

- **What's New**: 본 논문은 EDL(Evidential Deep Learning) 모델의 몇 가지 비필수적 설정을 완화한 Re-EDL을 제안합니다. Re-EDL은 주관적 논리(subjective logic)의 투영된 확률을 유지하며, 사전 가중치(prior weight)를 조절 가능한 하이퍼파라미터로 간주하고, 변동성 최소화(variance-minimizing) 최적화 항과 KL 발산 정규화(KL divergence regularization)를 생략합니다.

- **Technical Details**: Re-EDL은 Dirichlet 확률 밀도 함수(Dirichlet PDF)를 직접 최적화하여 예측 불확실성(prediction uncertainty)을 향상시키는 방법론입니다. 모델 구성에서 사전 가중치 파라미터는 클래스 수에 고정되지 않고 자유롭게 조정될 수 있는 하이퍼파라미터로 설정됩니다. 또한, 기존의 EDL 방법에서 사용되는 변동성 최소화 최적화 항과 KL 발산 최소화 정규화 항을 제거하여 성능을 개선합니다.

- **Performance Highlights**: 대규모 실험을 통해 Re-EDL의 효과성과 최첨단 성능을 입증하였으며, 주어진 문제에 대한 높은 신뢰도의 예측 불확실성을 제공함으로써 자율 주행 및 의료 분석과 같은 고위험 도메인에서의 활용 가능성을 보여줍니다.



### STGformer: Efficient Spatiotemporal Graph Transformer for Traffic Forecasting (https://arxiv.org/abs/2410.00385)
- **What's New**: 본 논문에서는 교통 예측을 위한 새로운 구조인 spatiotemporal graph transformer (STGformer)를 제안합니다. 이는 기존의 graph 신경망(GCN)과 transformer 기반 모델의 장점을 통합하여, 효율적인 교통 패턴 모델링을 가능하게 합니다. 특히 STGformer는 단일 레이어에서 고차원 spatiotemporal 상호작용을 캡처할 수 있는 혁신적인 STG attention block을 도입하여 계산 비용을 크게 줄였습니다.

- **Technical Details**: STGformer는 GCN과 Transformers의 강점을 조화롭게 결합하여, 전 세계적 및 지역적 교통 패턴을 동시에 모델링할 수 있습니다. 기존 방법들은 여러 개의 attention 레이어를 필요로 하지만, STGformer는 단일 레이어에서 모든 상호작용을 효율적으로 처리합니다. GPT를 포함한 대부분의 transformer 모델들이 높은 계산 비용을 요구하는 반면, STGformer는 100배 빠르고 GPU 메모리 사용을 99.8% 줄이는 성능을 보여줍니다.

- **Performance Highlights**: STGformer는 LargeST 벤치마크에서 최첨단 transformer 기반 방법들인 PDFormer 및 STAEformer보다 우수한 성능을 보여줍니다. 주목할 만한 점은, STGformer는 기존의 방법들에 비해 계산 비용이 현저히 낮으면서도 모든 상호작용을 유효하게 처리할 수 있는 능력을 구현하는 점입니다.



### Generative Precipitation Downscaling using Score-based Diffusion with Wasserstein Regularization (https://arxiv.org/abs/2410.00381)
Comments:
          19 pages, 9 figures

- **What's New**: 이 논문은 기후 예측 센터(CPC)에서 제공하는 강우량 데이터를 및 ERA5 재분석 데이터를 활용하여 1km 해상도의 강우량 추정치를 생성하는 새로운 생성적 확산 모델 WassDiff를 제안합니다.

- **Technical Details**: WassDiff 모델은 Wasserstein Distance Regularization (WDR) 기법을 사용하여 등급 기반 확산 모델의 점수 일치(training objective) 과정에서 고도화된 강우량 강도를 정확하게 재현하도록 훈련됩니다. 이 모델은 55km 해상도의 데이터를 1km로 다운스케일링하여 극단적인 강우 신호를 포착하는데 도전과제를 극복합니다.

- **Performance Highlights**: WassDiff는 극단적인 기상 현상, 예를 들어 열대 폭풍 및 한랭 전선의 사례 연구에서 적절한 공간적 패턴을 생성하며, 기존의 점수 기반 확산 모델보다 더 나은 재구성 정확도 및 편향 점수를 기록합니다.



### Robust Traffic Forecasting against Spatial Shift over Years (https://arxiv.org/abs/2410.00373)
- **What's New**: 본 논문은 새로운 OOD(Out-Of-Distribution) 벤치마크를 제안하고, 기존의 ST-GNN(Spatiotemporal Graph Neural Networks) 모델들이 이러한 상황에서 성능이 크게 저하된다는 점을 강조합니다. 이를 해결하기 위해 새로운 Mixture of Experts (MoE) 프레임워크를 도입하여, 환경 변화에 적응하여 새로운 그래프를 생성하도록 합니다.

- **Technical Details**: 제안된 방법은 각 그래프 생성기(graph generator)를 학습하여 환경 변화에 따라 새로운 그래프 생성을 가능하게 하며, 기존의 모든 spatiotemporal 모델에 통합될 수 있습니다. 또한, LSTM을 사용한 연구가 시간적 의존성에 대한 성능 저하를 분석하는 데 사용됩니다. 여기서 제안한 expert graphon layer는 OOD 환경을 학습하고, 이를 통해 모델이 새로운 환경에 적합한 그래프를 적응적으로 조합하도록 합니다.

- **Performance Highlights**: 제안된 MoE 프레임워크는 기존의 BENCHMARK보다 우수한 성능을 보이며, 교차적인 성능을 평가하여 이전의 모델보다 안정적인 교통 예측이 가능합니다. 기존 ST-GNN 모델들이 겪는 성능 저하 문제를 해결하고, 다양한 환경에서의 예측을 향상시킬 수 있는 전략을 제시합니다.



### Easydiagnos: a framework for accurate feature selection for automatic diagnosis in smart healthcar (https://arxiv.org/abs/2410.00366)
- **What's New**: 본 연구에서는 Adaptive Feature Evaluator (AFE) 알고리즘을 제안하여 의료 데이터셋 내에서 특성 선택(feature selection)을 개선하고 클리닉 환경에서의 인공지능(AI) 적용에 있어 존재하는 문제를 해결하고자 합니다.

- **Technical Details**: AFE는 Genetic Algorithms (GA), Explainable Artificial Intelligence (XAI), Permutation Combination Techniques (PCT)를 통합하여 Clinical Decision Support Systems (CDSS)의 성능을 최적화하며, 다양한 머신러닝 알고리즘을 통해 세 가지 의료 데이터셋에서 검증되었습니다.

- **Performance Highlights**: AFE 알고리즘은 Multi-layer Perceptron (MLP)와 결합하여 최대 98.5%의 정확도를 달성하였으며, 기존의 특성 선택 기법들에 비해 동작의 견고성을 강조합니다.



### Neural Scaling Laws of Deep ReLU and Deep Operator Network: A Theoretical Study (https://arxiv.org/abs/2410.00357)
- **What's New**: 본 논문은 깊은 연산자 신경망(Deep Operator Networks)에서의 신경 스케일링 법칙(neural scaling laws)을 탐구하며, 이러한 법칙의 이론적 프레임워크를 확립하기 위한 연구를 진행합니다.

- **Technical Details**: 우리는 입력 함수의 저차원 구조가 존재하는 경우를 다루어 더 정확한 오차 경계를 유도하며, 이를 통해 깊은 ReLU 네트워크 및 비슷한 구조에서도 결과가 적용될 수 있음을 보여줍니다. 이 연구는 네트워크의 모델 크기, 훈련 데이터 크기와의 관계를 명확히 하고, 오차 추정 및 일반화 오류 송환을 분석합니다.

- **Performance Highlights**: 연구 결과는 연산자 학습 분야에서의 신경 스케일링 법칙을 부분적으로 설명하고, 이를 통해 신경망의 성능을 구체적으로 정량화할 수 있는 이론적 기초를 제공합니다.



### A Taxonomy of Loss Functions for Stochastic Optimal Contro (https://arxiv.org/abs/2410.00345)
- **What's New**: 이 논문은 Stochastic Optimal Control (SOC) 문제와 관련된 최신 연구를 소개하고 있습니다. 특히, Adjoint Matching이라는 새로운 손실 함수(loss function)가 기존의 손실 함수들보다 보상 조정(reward fine-tuning) 설정에서 월등한 성능을 보임을 강조합니다.

- **Technical Details**: SOC 손실 함수들은 기대값에서 같은 그래디언트(gradient)를 가지는 클래스들로 나눌 수 있음을 보여줍니다. 이러한 그룹화는 옵티마이제이션(optimization) 측면에서 유사한 특성을 갖으며, 오직 그래디언트 분산(variance)에서만 차이를 나타냅니다. 저자들은 간단한 SOC 실험을 통해 다양한 손실 함수의 강점과 약점을 분석합니다.

- **Performance Highlights**: Adjoint Matching 손실 함수를 활용한 실험에서 기존 손실 함수들보다 뛰어난 성능을 입증하였으며, SOC 문제에 대한 새로운 시각을 제공하고 있습니다.



### Sparse Attention Decomposition Applied to Circuit Tracing (https://arxiv.org/abs/2410.00340)
- **What's New**: 본 연구에서는 GPT-2 small 모델 내의 attention heads 간의 통신 및 조정을 효과적으로 분석하기 위해, 그 과정에서 사용되는 희소한(signal) 특징들을 고립 및 식별하고자 합니다. 또한, attention head 행렬의 특잇값 분해(singular value decomposition)로부터 얻은 특징을 바탕으로 보다 효율적인 경로 추적을 제안합니다.

- **Technical Details**: GPT-2 small 모델을 사용하여 Indirect Object Identification (IOI) 작업에서 attention heads 간의 관계를 분석하였습니다. 이 연구에서는 residual 배경으로부터 신호를 효율적으로 분리하고, attention head의 입력을 새로운 기준으로 바꾸어 sparseContribution을 정의하였습니다. 이 새로운 기준을 통해 downstream과 upstream attention heads 간의 인과 관계를 명확히 할 수 있었습니다.

- **Performance Highlights**: 본 연구의 결과로, 새로운 기준을 통해 trace한 신호가 기존 연구들보다 더욱 세부적인 내용을 제공하며, GPT-2가 IOI 작업을 수행하는 데 있어 효과적인 커뮤니케이션 경로를 식별할 수 있음을 보여줍니다. 이를 통해 attention score의 해석 가능성이 크게 향상되며, 모델의 기능적 요소를 보다 명확히 이해할 수 있게 됩니다.



### EnzymeFlow: Generating Reaction-specific Enzyme Catalytic Pockets through Flow Matching and Co-Evolutionary Dynamics (https://arxiv.org/abs/2410.00327)
- **What's New**: EnzymeFlow는 효소의 촉매 주머니(catalytic pocket)를 설계하기 위한 새로운 생성 모델로, 흐름 매칭(flow matching)과 계층식 사전 학습(hierarchical pre-training) 및 효소-반응 공진화(enzyme-reaction co-evolution)를 활용하여 특정 기질(substrate) 및 촉매 반응에 대해 데이터 기반으로 촉매 주머니를 생성합니다.

- **Technical Details**: EnzymeFlow는 효소 촉매 주머니 생성을 위한 조건적 흐름(conditional flow)을 정의하며, 이를 통해 특정 기질과 생성물에 따라 다양한 촉매 과정이 가능하도록 합니다. 또한, 효소-반응 공진화(co-evolution)를 통해 촉매 반응에서 기질 특이성을 포착하고, 구조 기반의 계층식 사전 학습을 통해 더 나은 모델 성능을 꾀합니다. 데이터 세트는 $328,192$ 쌍의 효소-반응 쌍으로 구성되어 있습니다.

- **Performance Highlights**: 실험 결과, EnzymeFlow는 고품질의 기능성 효소 촉매 주머니를 설계하는 데 효과적임을 보여주었으며, 효소 공학(enzyme engineering) 및 합성 생물학(synthetic biology) 분야에서의 발전을 위한 새로운 가능성을 제시합니다.



### VLMGuard: Defending VLMs against Malicious Prompts via Unlabeled Data (https://arxiv.org/abs/2410.00296)
Comments:
          arXiv admin note: text overlap with arXiv:2409.17504

- **What's New**: 이번 논문은 VLMGuard라는 새로운 학습 프레임워크를 소개하며, 이는 비표식 사용자 프롬프트에서 악의적 프롬프트를 탐지하는 데 중점을 둡니다. 이 프레임워크는 사용자 프롬프트가 혼합되어 있는 상황에서 비악의적과 악의적 데이터를 구분하기 위해 자동화된 악의성 평가 점수를 구축합니다.

- **Technical Details**: VLMGuard는 VLM의 잠재 표현(latent representations)을 활용하여 악의적인 의도를 포함하는 데이터를 식별합니다. 사용자는 안전 기능을 우회하려 하거나 원치 않는 행동을 유도할 수 있는 다양한 쿼리에서 발생하는 비표식 데이터를 다룹니다. 이를 위해, 악의성 점수는 고유한 임베딩을 사용하여 악의적인 프롬프트와 비악의적인 프롬프트를 구분하는 데 활용됩니다.

- **Performance Highlights**: 첫 번째 실험 결과, VLMGuard는 LLaVA 모델에서 평균 13.21% 향상된 AUROC(Area Under Receiver Operating Characteristic) 성능을 기록하여 최신 기술들보다 우수한 탐지 결과를 보였습니다. 또한, 본 연구는 설계 요소의 효과성을 분석하고 글로벌 현실 대응의 확장성을 검토합니다.



### Comprehensive Performance Modeling and System Design Insights for Foundation Models (https://arxiv.org/abs/2410.00273)
Comments:
          17 pages, PMBS 2024

- **What's New**: 이 논문은 Generative AI 특히 대규모 transformer 모델의 성능 특성을 분석하고, 다양한 transformer 유형, 병렬화 전략 및 HPC 시스템 기능에 대한 민감도를 논의합니다.

- **Technical Details**: 우리는 대규모 언어 모델(LLM) 및 긴 시퀀스 transformer 모델을 대상으로 하여 각각의 학습 요구 사항과 최적 병렬성을 식별하는 성능 모델을 개발했습니다. 데이터 병렬화, 텐서 병렬화(1D 및 2D), 파이프라인 병렬화와 같은 다양한 병렬화 전략을 포함하여 성능 요구 사항을 평가했습니다.

- **Performance Highlights**: 모델 아키텍처에 따라 최적의 병렬성 요구 사항이 다르며, GPT3-1T 모델은 1D 텐서 병렬화가 효과적이고, ViT 모델은 2D 텐서 병렬화가 필요하다는 분석 결과를 도출했습니다. 이 논문은 빠른 네트워크 도메인과 GPU 세대에 따른 성능 영향을 평가하여, 다양한 모델에 대한 최적 구성을 확인했습니다.



### Enhanced Credit Score Prediction Using Ensemble Deep Learning Mod (https://arxiv.org/abs/2410.00256)
Comments:
          This paper have been accepted by CSP Journal

- **What's New**: 현대 경제 사회에서 신용 점수는 모든 참가자에게 중요한 요소입니다. 이 논문에서는 XGBoost, LightGBM와 같은 고성능 모델과 함께 TabNet 모델을 결합하여 신용 점수를 정확하게 결정할 수 있는 강력한 모델을 개발했습니다.

- **Technical Details**: Random Forest, XGBoost 및 TabNet 모델을 스택킹 기법(Ensemble Modeling)으로 통합하여 단일 모델의 한계를 극복하고 정확한 신용 점수 예측을 가능하게 했습니다. 연구에 사용된 데이터 세트는 Kaggle에서 제공되는 신용 점수 분류 데이터셋이며, 데이터 전처리 기술을 통해 결측값 및 노이즈를 효과적으로 처리했습니다.

- **Performance Highlights**: 모델의 성과는 Precision, Recall, F1, AUC와 같은 다양한 메트릭을 통해 검증되었으며, 서로 보완하는 모델 조합으로 강력한 전체 성능을 자랑합니다.



### Quantized and Asynchronous Federated Learning (https://arxiv.org/abs/2410.00242)
- **What's New**: 새로운 알고리즘, Quantized Asynchronous Federated Learning (QAFeL)을 개발하여 비동기 연합 학습에서 오류 전파를 방지하는 숨겨진 상태 양자화 방식 도입.

- **Technical Details**: QAFeL은 클라이언트 업데이트를 집계하기 위한 버퍼를 포함하며, 이는 보안 집계를 위한 기술과 호환됩니다. 또한, QAFeL은 비볼록 목표에 대해 확률적 경량 하강법(stochastic gradient descent)의 𝓞(1/√T) ergodic 수렴 속도를 달성함을 증명합니다.

- **Performance Highlights**: QAFeL의 실험 결과는 기존 연합 학습 알고리즘에 비해 우수한 성능을 보이며, 교차 항 오차가 개별 오류보다 더 작은 것으로 나타났습니다.



### Preconditioning for Accelerated Gradient Descent Optimization and Regularization (https://arxiv.org/abs/2410.00232)
Comments:
          7 pages

- **What's New**: 이 논문은 AdaGrad, RMSProp, Adam과 같은 가속화된 학습 알고리즘의 이론적 기초와 정규화 방법의 상호작용을 탐구하며, 새로운 전처리 기법인 gradient regularization의 적절한 결합 방법을 제시합니다.

- **Technical Details**: 이 논문에서는 표준 경량화(weight decay) 방식과 다르게 L2 정규화가 Adam에 효과적이지 않음을 보여주며, AdamW와 같은 새로운 접근법을 사용해 정규화와 적응형 학습률을 결합하는 문제를 다룹니다. 또한, Hessian conditioning을 개선하는 방법으로 정규화의 중요성을 강조합니다.

- **Performance Highlights**: 여러 가속화 기법들의 이해를 돕는 통합된 수학적 프레임워크를 제공하여, 적절한 정규화 스킴을 도출하고 새로운 전처리 학습 알고리즘 개발에 기여할 것으로 기대됩니다.



### Probabilistic Classification of Near-Surface Shallow-Water Sediments using A Portable Free-Fall Penetrometer (https://arxiv.org/abs/2410.00225)
- **What's New**: 이번 연구는 Portable Free Fall Penetrometer (PFFP) 데이터를 기반으로 한 퇴적물 행동 분류 시스템을 머신러닝 알고리즘을 통해 개발했습니다.

- **Technical Details**: PFFP는 수직 감속도와 경사를 측정하기 위해 설계된 5개의 가속도계를 장착하고 있으며, PFFP 데이터는 Sequim Bay, Potomac River 및 York River에서 수집되었습니다. 수집된 데이터는 분류를 위한 머신러닝 모델 훈련에 사용되었습니다.

- **Performance Highlights**: 예측 모델은 91.1%의 정확도로 퇴적물 클래스를 예측하며, 다양한 불확실성을 정량화하여 더 포괄적이고 정보에 기반한 접근 방식을 제공합니다.



### Characterizing and Efficiently Accelerating Multimodal Generation Model Inferenc (https://arxiv.org/abs/2410.00215)
Comments:
          13 pages including references. 8 Figures. Under review to HPCA 2025 Industry Track

- **What's New**: 이번 논문은 생성적 인공지능(Generative AI) 기술이 컴퓨팅 산업을 혁신하고 있다는 점을 강조하며, 이 기술의 새로운 시스템 설계 및 최적화 기회를 설명합니다. 특히 다중 모달 생성 모델의 효율적 추론(inference) 성능을 최적화하기 위한 중요한 기회를 상세히 제시합니다.

- **Technical Details**: 다중 모달 생성 모델의 성능 특성을 분석하여 GPU의 대기 시간과 메모리 집약적인 Attention 특성 등 여러 요인을 살펴보았습니다. 이 논문은 torch.compile, CUDA Graph, SDPA(Scaled Dot Product Attention), Flash Attention, quantization 등의 최적화 기법을 통해 생성적 AI 작업의 추론 성능을 3.88배 증가시킬 수 있음을 보여줍니다.

- **Performance Highlights**: 최신 생성적 AI 기술의 추론 성능을 위해 3.88배 향상된 새로운 기준을 제시하며, LayerSkip 알고리즘 최적화를 통하여 생성 속도를 1.58배 개선할 수 있음을 시연하였습니다.



### GaNDLF-Synth: A Framework to Democratize Generative AI for (Bio)Medical Imaging (https://arxiv.org/abs/2410.00173)
- **What's New**: Generative Artificial Intelligence (GenAI)는 기존 데이터를 바탕으로 새로운 데이터 샘플을 만들어내는 AI 분야로, 의료 데이터의 부족과 규제를 극복하기 위해 새로운 데이터 포인트를 생성하는 방법을 탐구합니다. 이번 연구는 GaNDLF-Synth라는 새로운 프레임워크를 도입하여 의료 이미지 합성과 같은 작업의 접근성과 평가를 민주화하는 데 기여하고자 합니다.

- **Technical Details**: GaNDLF-Synth는 autoencoders, generative adversarial networks, diffusion models를 포함한 다양한 합성 알고리즘을 통합하고 지원하는 통합 추상화를 제공합니다. GANDLF-core 프레임워크를 활용하여 다양한 데이터 형식과 분산 컴퓨팅을 지원하고, 포괄적인 단위 테스트를 통해 확장성과 재현성을 보장합니다.

- **Performance Highlights**: GaNDLF-Synth는 제로/로우 코드 인터페이스를 제공하여 컴퓨터 연구자와 임상 연구자 모두가 쉽게 접근할 수 있도록 하며, 최신 합성 방법을 지속적으로 업데이트하여 연구자들에게 최첨단 도구로 자리매김하고 있습니다. 이 도구는 Git 또는 pip을 통해 쉽게 설치할 수 있습니다.



### Basis-to-Basis Operator Learning Using Function Encoders (https://arxiv.org/abs/2410.00171)
- **What's New**: 논문에서는 Basis-to-Basis (B2B) operator learning을 제안하며, 이는 Hilbert 공간의 함수에서 연산자를 학습하는 새로운 접근 방식입니다. 이 방법은 함수 인코더(Function Encoder)의 기본 개념에 기반하여 기능합니다.

- **Technical Details**: B2B operator learning은 입력과 출력 공간 모두에 대한 기저 함수 집합을 학습하는 것과 기저 함수의 계수 간의 비선형 매핑을 학습하는 두 가지 부분으로 문제를 분해합니다. 이 방법은 전통적인 방법들이 필요로 하는 고정된 위치의 데이터를 요구하지 않으며, 최소 제곱(least-squares) 기법을 활용하여 계수를 계산합니다.

- **Performance Highlights**: B2B operator learning은 여섯 개의 기준 연산자 학습 작업에서 기존 접근 방식에 비해 정확도가 두 배 이상 향상되는 성능을 보여주었습니다.



### (Almost) Smooth Sailing: Towards Numerical Stability of Neural Networks Through Differentiable Regularization of the Condition Number (https://arxiv.org/abs/2410.00169)
Comments:
          Accepted at ICML24 Workshop: Differentiable Almost Everything: Differentiable Relaxations, Algorithms, Operators, and Simulators

- **What's New**: 이 논문에서는 머신 러닝 모델의 신뢰성과 성능을 유지하기 위해 중요한 수치적 안정성(numerical stability) 문제를 다룹니다. 특히 가중치 행렬의 조건 수(condition number)를 최적화 알고리즘에 정규화 항으로 통합하는 새로운 정규화기(regularizer)를 소개합니다.

- **Technical Details**: 제안된 정규화기는 거의 모든 곳에서 미분 가능(differentiable)하며, 조건 수가 낮은 행렬을 촉진합니다. 이를 통해 다운스트림 과제에 최적화된 조건 수를 보장하고, 가중치 행렬의 수치적 안정성을 정규화함으로써 Gradient Descent와 같은 경량화된 최적화 알고리즘에 쉽게 통합될 수 있습니다.

- **Performance Highlights**: MNIST 이미지에 대한 소음 분류(noisy classification) 및 잡음 제거(denoising) 실험에서 제안된 방법이 기존 접근 방식보다 유리함을 보여주었습니다.



### Fisher Information-based Efficient Curriculum Federated Learning with Large Language Models (https://arxiv.org/abs/2410.00131)
Comments:
          27 pages, 8 figures, 14 tables, to appear in EMNLP 2024

- **What's New**: 이 논문에서는 Federated Learning(FL) 환경에서 Large Language Models(LLMs)를 효율적으로 미세 조정하기 위해 Fisher Information 기반의 새로운 Curriculum Federated Learning 프레임워크(FibecFed)를 제안합니다. 이 프레임워크는 두 가지 혁신적인 방법인 적응형 연합 커리큘럼 학습 및 효율적인 희소 파라미터 업데이트를 포함하고 있습니다.

- **Technical Details**: FibecFed는 각 장치 내에서 훈련 데이터의 난이도를 측정하기 위해 Fisher Information 기반의 방법을 활용하여 적응적으로 데이터를 샘플링합니다. 이를 통해 초기에는 쉬운 데이터 샘플을 사용하고 점진적으로 난이도를 높이며 FL의 미세 조정 효과성을 향상시킵니다. 또한, LoRA를 활용하여 전역 집합을 위해 적절한 레이어를 선택하고 희소 파라미터를 동적으로 업데이트하여 효율성을 개선합니다.

- **Performance Highlights**: FibecFed는 10개의 데이터 세트를 기반으로 한 광범위한 실험 결과에서 17개의 기준 방법에 비해 정확도가 최대 45.35% 향상되었고, 미세 조정 속도는 최대 98.61% 더 빨라졌음을 보여주었습니다.



### Using fractal dimension to predict the risk of intra cranial aneurysm rupture with machine learning (https://arxiv.org/abs/2410.00121)
- **What's New**: 이번 연구에서는 뇌동맥류 (Intracranial Aneurysms, IAs)의 파열 여부를 예측하기 위해 네 가지 기계 학습 (Machine Learning, ML) 알고리즘의 성능을 비교했습니다.

- **Technical Details**: 사용된 기계 학습 알고리즘은 Random Forest (RF), XGBoost (XGB), Support Vector Machine (SVM), Multi Layer Perceptron (MLP)이며, 임상 및 방사선학적 (Radiographic) 특징을 활용했습니다. 각 모델의 성능을 평가하기 위해 정확도 (accuracy), 정밀도 (precision), 재현율 (recall) 등 다양한 메트릭을 사용했습니다.

- **Performance Highlights**: RF 모델이 85%의 가장 높은 정확도를 기록했으며, MLP는 63%로 가장 낮은 성능을 보였습니다. Fractal dimension은 모든 모델에서 성능에 가장 중요한 특징으로 평가되었습니다.



### Fine-tuning Vision Classifiers On A Budg (https://arxiv.org/abs/2410.00085)
Comments:
          8 pages, 5 figures

- **What's New**: 이 논문에서는 라벨러의 정확성에 대한 이전 추정치를 활용하여, 단순 naive-Bayes 모델을 사용하여 실제 라벨을 추정하는 방법인 Ground Truth Extension (GTX)을 소개합니다. 이 방법은 고품질의 데이터를 더 적은 인적 라벨로 수집할 수 있게 해줍니다.

- **Technical Details**: GTX 방법은 3단계로 구성되며, 첫째, 노이즈 라벨을 고려하여 실제 라벨 및 신뢰도를 추정하기 위한 확률적 모델을 제공합니다. 둘째, 전문가가 제공한 소수의 라벨(ground truth labels)을 이용해 라벨러의 정확도를 추정합니다. 셋째, 주어진 신뢰도 수준에서 최대한 많은 실제 라벨 추정을 가능하게 하는 예제를 (노이즈가 포함된) 라벨링하는 전략을 수립합니다.

- **Performance Highlights**: GTX 방법은 여러 조건에서 라벨의 정확성 향상에서 표준 및 가중 다수결 표본보다 뛰어난 성능을 보였으며, 산업 비주얼 분류 작업에 사용한 실험에서도 CNN 모델의 정밀도를 높이는 데 있어 GTX로 추정된 라벨이 효과적이라는 것을 입증했습니다.



### A Survey on Diffusion Models for Inverse Problems (https://arxiv.org/abs/2410.00083)
Comments:
          Work in progress. 38 pages

- **What's New**: 이 논문에서는 pretrained diffusion model을 활용하여 inverse problem을 해결하는 방법에 대한 포괄적인 개요를 제공합니다. 기존 훈련 과정 없이도 고품질 샘플을 생성할 수 있는 가능성을 탐구합니다.

- **Technical Details**: Diffusion models는 unsupervised priors로 사용되어 inverse problem 해결에 활용됩니다. 다양한 문제와 기술에 따라 방법을 분류하는 taxonomy를 소개하며, 이를 통해 각 접근 방식의 연결성을 분석합니다. 특히 latent diffusion model을 사용할 때의 도전 과제와 잠재적 해결책에 대해서도 논의합니다.

- **Performance Highlights**: Diffusion models를 활용한 inverse problem 해결 방법의 실용적 구현을 강조하며, 다양한 과학 분야에서의 적용 가능성을 보여줍니다. 예를 들어, 이미지 복원, MRI 가속화, 오디오 신호 처리 등에서 효과적인 결과를 나타냅니다.



### Collaborative Knowledge Distillation via a Learning-by-Education Node Community (https://arxiv.org/abs/2410.00074)
- **What's New**: 이번 논문에서는 Collaborative Knowledge Distillation (CKD)을 위한 새로운 Learning-by-Education Node Community (LENC) 프레임워크가 제안됩니다. 이 프레임워크는 다양한 Deep Neural Network (DNN) 노드 간 효과적인 지식 교환을 통해 지속적인 집합 학습을 가능하게 합니다.

- **Technical Details**: LENC 프레임워크는 동적으로 교사 또는 학생 역할을 수행하는 DNN 노드를 포함하고 있으며, 다중 태스크 지식 증류를 가능하게 합니다. 또한, Out-Of-Distribution (OOD) 감지 알고리즘, 지식 전송 메커니즘 및 지속적 학습 (CL) 알고리즘이 통합되어 있습니다.

- **Performance Highlights**: 실험 결과, LENC 프레임워크는 이미지 분류 문제에서 상호작용하는 DNN 노드 커뮤니티의 평균 테스트 정확도를 점진적으로 극대화하며, 작은 배치의 비표식 데이터에서 온라인 학습을 통해 모든 경쟁 기존 방법을 초월하는 성능을 보여주었습니다.



### M2Distill: Multi-Modal Distillation for Lifelong Imitation Learning (https://arxiv.org/abs/2410.00064)
Comments:
          Submitted to ICRA2025

- **What's New**: 이 논문에서는 M2Distill이라는 새로운 multi-modal distillation 기반 방법을 소개합니다. 이는 평생 imitation learning 과정에서 시각, 언어, 행동 분포 간 일관된 latent space를 유지하는 데 초점을 맞추고 있습니다.

- **Technical Details**: M2Distill은 다양한 modality 간의 latent representation 변화 조절을 통해 이전 단계와 현재 단계 간의 일관성을 유지하며, Gaussian Mixture Model (GMM) 정책의 불일치를 줄여 갑작스러운 잊어버림을 방지합니다. 이를 통해 학습된 정책이 이전에 배운 작업을 계속 수행할 수 있도록 보장합니다.

- **Performance Highlights**: LIBERO 평생 imitation learning 벤치마크(Campaign)에서 M2Distill은 LIBERO-OBJECT, LIBERO-GOAL, LIBERO-SPATIAL을 포함하여 모든 평가 지표에서 이전의 최첨단 방법들보다 우수한 성능을 보여주었습니다.



### Neural Decompiling of Tracr Transformers (https://arxiv.org/abs/2410.00061)
- **What's New**: 이 논문에서는 Transformer 아키텍처의 해석력을 높이기 위한 첫 단계를 제안합니다. 이를 위해 Transformer Compiler for RASP(Tracr)를 사용하여 많은 쌍의 transformer weights와 이에 상응하는 RASP 프로그램 데이터셋을 생성하였습니다. 이 데이터셋을 바탕으로, 딥러닝 모델을 훈련시켜 컴파일된 모델에서 RASP 코드를 복구하는 것을 목표로 했습니다.

- **Technical Details**: 기존의 Transformer 모델의 해석 가능성(interpretability)은 주로 수동적인 작업에 의존해왔지만, 본 논문에서는 전체 해석 프로세스를 자동화하는 디컴파일러 모델을 제안합니다. RASP111(Restricted Access Sequence Processing Language) 코드를 통해 transformer weights를 해석하고, RASP 코드와 변환기 가중치 쌍의 대규모 데이터셋을 생성하는 알고리즘을 설계하였습니다. 이 데이터셋은 약 533,000개 프로그램과 222개의 변환기를 포함하고 있습니다.

- **Performance Highlights**: 모델의 실험적 평가 결과, 30% 이상의 테스트 개체에서 정확한 재현을 달성하였으며, 나머지 70%는 소수의 오류로 일반적으로 재현할 수 있었습니다. 또한, 모델에서 생성한 프로그램의 70% 이상이 진짜와 기능적으로 동등하여 Tracr로 컴파일된 transformer weights의 유효한 디컴파일을 의미합니다.



### STTM: A New Approach Based Spatial-Temporal Transformer And Memory Network For Real-time Pressure Signal In On-demand Food Delivery (https://arxiv.org/abs/2410.00057)
- **What's New**: 이 논문은 Spatio-Temporal Transformer와 Memory Network (STTM) 기반의 새로운 방법을 제안하며, 이는 On-demand Food Delivery (OFD) 서비스에서 Real-time Pressure Signal (RPS) 예측을 위한 것입니다. 이 방법은 물류 특징을 시간적 및 공간적 차원에서 학습하여 비즈니스 구역의 과거 정보를 인코딩합니다.

- **Technical Details**: STTM은 Spatio-Temporal Transformer 구조를 사용하여 OFD 도메인에서 고유한 시공간 정보를 학습합니다. 또한, Memory Network를 적용하여 심각한 날씨나 피크 시간대와 같은 비정상적 사건에 대한 민감성을 높입니다. 이는 RPS를 예측하는 데 필요한 프레임워크로서, 다른 모델과 비교해 더 나은 성능을 보여줍니다.

- **Performance Highlights**: STTM은 이전 방법에 비해 최대 9.66%의 향상된 결과를 보여주며, 중국의 Ele.me와 같은 대형 OFD 플랫폼에도 성공적으로 배포되었습니다. 실험 결과, STTM은 오프라인 실험과 온라인 A/B 테스트 모두에서 유의미한 개선을 입증하였습니다.



### Transferable Unsupervised Outlier Detection Framework for Human Semantic Trajectories (https://arxiv.org/abs/2410.00054)
Comments:
          This is an accepted paper on this https URL

- **What's New**: 본 논문에서는 인간의 semantic trajectory에 대한 Transferable Outlier Detection (TOD4Traj) 프레임워크를 제안합니다. 이 프레임워크는 복합적인 spatial, temporal 및 textual 데이터를 통합하여 outlier를 효과적으로 감지할 수 있는 메커니즘을 제공합니다.

- **Technical Details**: TOD4Traj는 modality feature unification module을 사용하여 다양한 데이터 특성 표현을 정렬합니다. 또한, contrastive learning module을 도입하여 시간적으로와 인구 그룹 전반에서의 정기적인 이동 패턴을 파악하는 동시에 개인의 일관성과 그룹의 주 패턴 기반으로 outlier를 탐지합니다.

- **Performance Highlights**: TOD4Traj는 기존 모델보다 우수한 성능을 보여주며, 다양한 데이터를 통해 인간의 trajectory outlier를 효과적으로 감지하는 능력을 입증하였습니다.



### Frequency-adaptive Multi-scale Deep Neural Networks (https://arxiv.org/abs/2410.00053)
- **What's New**: 이 논문에서는 Multi-scale deep neural networks (MscaleDNNs)의 장점을 설명하고, 이를 개선하기 위한 새로운 방법으로 frequency-adaptive MscaleDNNs를 제안합니다. 또한, fitting error bound를 수립하여 MscaleDNNs의 유효성을 이론적으로 뒷받침합니다.

- **Technical Details**: MscaleDNNs는 고주파 기능을 잘 근사할 수 있는 특성을 가지고 있으며, radial down-scaling mapping을 사용하여 고주파 정보를 낮은 주파수 표현으로 변환합니다. 새로운 hybrid feature embedding 기법을 통해 정확성과 견고성을 높이기 위한 방법을 제시하며, posterior error estimate를 통해 MscaleDNNs의 파라미터를 적응적으로 조정합니다.

- **Performance Highlights**: 제안된 frequency-adaptive MscaleDNNs는 기존의 MscaleDNNs보다 2배에서 3배 더 높은 정확도를 보여주며, 다양한 수치 예제를 통해 그 효과성을 입증합니다.



### DelayPTC-LLM: Metro Passenger Travel Choice Prediction under Train Delays with Large Language Models (https://arxiv.org/abs/2410.00052)
Comments:
          15 pages,4 figures

- **What's New**: 이번 논문에서는 대도시 철도 시스템에서의 기차 지연에 대한 승객 여행 선택 예측을 위한 새로운 프레임워크인 DelayPTC-LLM을 제안합니다. 이는 대형 언어 모델(LLM)을 활용하여 데이터를 처리하고 예측하는 방식입니다.

- **Technical Details**: DelayPTC-LLM은 승객 이질성(passenger heterogeneity)과 기차 지연 이벤트의 특성을 고려하여 LLM이 여행 선택에 대한 예측을 하고 이를 합리화할 수 있도록 유도하는 잘 설계된 프롬프트(Prompt) 엔지니어링을 개발합니다. 이 모델은 심천(Shenzhen) 지하철의 자동 요금 수집(AFC) 데이터 및 지연 로그를 사용하여 실험됩니다.

- **Performance Highlights**: 전통적인 예측 모델들과 비교했을 때, DelayPTC-LLM은 복잡하고 희소한(dense) 데이터 세트에서의 예측 정확도에서 우수한 성능을 보이며, 대규모 교통 데이터에 대한 실행 가능한 인사이트를 제공할 수 있는 잠재력을 갖고 있습니다.



### Generalizing Consistency Policy to Visual RL with Prioritized Proximal Experience Regularization (https://arxiv.org/abs/2410.00051)
Comments:
          Accepted at the Thirty-Eighth Annual Conference on Neural Information Processing Systems (NeurIPS2024)

- **What's New**: 본 연구에서는 고차원 상태 공간의 비주얼 강화 학습(visual RL)에서 정책 훈련의 불안정성을 완화하기 위해 일관성 정책(consistency policy)의 샘플 기반 엔트로피 정규화(sample-based entropy regularization)와 우선순위 친근 경험 정규화(prioritized proximal experience regularization, CP3ER) 접근 방식을 제안합니다. CP3ER은 DeepMind 제어 스위트와 Meta-world를 포함한 21개 작업에서 새로운 최첨단 성능(SOTA)을 달성했습니다.

- **Technical Details**: 우선 CP3ER은 온라인 강화 학습(online RL)에서 비가역적인 데이터 분포(non-stationary distribution)와 액터-크리틱(actor-critic) 프레임워크가 일관성 정책에 미치는 영향을 조사했습니다. 연구 결과, 액터-크리틱 프레임워크의 Q-손실(Q-loss)이 일관성 모델의 표현 능력을 저해해 불안정한 정책 훈련을 초래하는 것을 발견했습니다. 본 연구에서는 이를 해결하기 위해 정책 훈련을 안정화하는 샘플 기반 엔트로피 정규화를 제안하였습니다.

- **Performance Highlights**: 제안된 CP3ER 방법론은 DeepMind 제어 스위트와 Meta-world 등 21개의 비주얼 제어 작업에서 SOTA 성능을 기록하였으며, 이는 비저장 강화 학습에서 일관성 모델의 응용 가능성을 보여줍니다.



### CycleBNN: Cyclic Precision Training in Binary Neural Networks (https://arxiv.org/abs/2410.00050)
Comments:
          Published at Workshop CADL, ECCV-2024

- **What's New**: 이 논문은 Binary Neural Networks(BNNs)의 새로운 훈련 방법론인 CycleBNN을 제안합니다. 기존의 문제인 훈련 속도가 느리고 성능 저하가 있는 문제를 해결하기 위해, 훈련 과정에서 정밀도를 동적으로 조정하는 사이클릭 정밀도 훈련을 도입합니다.

- **Technical Details**: CycleBNN은 BNNs와 사이클릭 정밀도 훈련을 통합하여 훈련 효율을 높이고 성능 손실을 최소화합니다. 이 방법은 가중치와 활성화를 주기적으로 1비트로 표현하는 BNN의 특성을 활용하여 정밀도를 최적화합니다. 실험은 ImageNet, CIFAR-10, PASCAL-VOC 데이터셋에서 수행되었으며, 각기 다른 태스크에서 CycleBNN의 성능을 평가하였습니다.

- **Performance Highlights**: CycleBNN은 ImageNet에서 훈련 시 96.09%의 연산량 절감, CIFAR-10에서 88.88%, PASCAL-VOC에서 다시 96.09%의 절감을 달성했습니다. 이는 CycleBNN이 효율적인 네트워크 훈련을 가능하게 하는 방법이라는 것을 입증합니다.



### Epidemiology-Aware Neural ODE with Continuous Disease Transmission Graph (https://arxiv.org/abs/2410.00049)
- **What's New**: 이 논문에서는 전염병의 동적 특성을 반영하고, 질병 전파의 구체적인 메커니즘을 고려하는 새로운 접근 방식인 EARTH(Epidemiology-Aware Neural ODE with Continuous Disease Transmission Graph)를 소개합니다.

- **Technical Details**: EANO(Epidemic-Aware Neural ODE)와 GLTG(Global-guided Local Transmission Graph)를 통해 지역과 전 세계의 질병 전파 패턴을 학습하며, 크로스-어텐션(cross-attention) 기법을 이용하여 중요한 정보들을 통합합니다. 이를 통해 리얼타임(RT) 데이터의 변화를 반영한 전염병 예측을 보다 정교하게 수행할 수 있습니다.

- **Performance Highlights**: EARTH는 COVID-19와 인플루엔자와 같은 다양한 전염병 예측 데이터셋에서 기존의 최첨단 방법들보다 우수한 성능을 보여줍니다.



### A Novel Spinor-Based Embedding Model for Transformers (https://arxiv.org/abs/2410.00038)
Comments:
          22 pages, 8 figures

- **What's New**: 이 논문은 기하대수(geometric algebra)에서 스피노르(spinors)를 활용하여 Transformer 모델의 단어 임베딩(word embeddings)을 새로운 방식으로 제안하고 있습니다. 스피노르는 고차원 공간에서의 복잡한 관계와 변환을 포착하는 수학적 프레임워크를 제공하며, 이를 통해 언어 표현의 풍부함과 강건함을 향상시키는 목표를 가지고 있습니다.

- **Technical Details**: 스피노르는 클리포드 대수(Clifford algebra)의 요소로, 고차원 공간에서의 회전(rotation)과 변환(transformation)을 나타납니다. 스피노르 기반 임베딩은 단어를 고차원 스피노르로 인코딩하며, Transformer 아키텍처에 통합되어 사용됩니다. 입력 토큰은 스피노르로 매핑되어, 자가 주의(self-attention) 메커니즘을 통해 처리됩니다. 스피노르 내적(spinor inner product)을 정의하고, 이를 통해 주의 가중치를 계산합니다.

- **Performance Highlights**: 스피노르 임베딩은 기존의 벡터 임베딩보다 더 높은 차원의 복잡성을 가지고 있어 더 미세한 관계와 변환을 포착할 수 있으며, 이는 Transformer 모델이 데이터의 근접성을 더 효과적으로 처리하고, 시퀀스 전반의 의존성을 모델링하는 데 도움을 줍니다. 이러한 개선 덕분에 더 정확한 예측을 가능하게 합니다.



### Prediction and Detection of Terminal Diseases Using Internet of Medical Things: A Review (https://arxiv.org/abs/2410.00034)
- **What's New**: 이번 연구는 AI(인공지능)와 IoMT(의료 사물인터넷)의 통합을 통해 만성 질환의 예측 및 진단에 대한 혁신적인 접근 방식을 제시합니다. XGBoost, Random Forest, CNN 및 LSTM RNN과 같은 AI 기반 모델들이 98% 이상의 정확도로 심장 질환, 만성 신장 질환(CKD) 및 알츠하이머병을 예측할 수 있음을 보여줍니다.

- **Technical Details**: 연구는 데이터 표준화, 고급 전처리 기법, 전이 학습(Transfer Learning)과 앙상블 방법(Ensemble Methods)의 중요성을 강조합니다. IoMT 시스템의 보안을 보장하기 위한 연합 학습(Federated Learning) 및 블록체인 기술이 필수적입니다. AI 모델은 과적합(Overfitting) 문제를 겪고 있으며, 진료 실재 환경에서의 성능 부족이 지적됩니다.

- **Performance Highlights**: 이 연구의 모델들은 Kaggle, UCI와 같은 플랫폼 및 실시간 IoMT 데이터 소스를 통해 높은 정확도를 보여주었으며, 앞으로 만성 질환 상호작용과 희귀 질환에 대한 예측 모델 개발을 위한 연구 방향 제안이 필요합니다.



### TREB: a BERT attempt for imputing tabular data imputation (https://arxiv.org/abs/2410.00022)
Comments:
          12 pages, 7 figures

- **What's New**: TREB는 BERT를 활용하여 표 형식 데이터의 결측값을 처리하는 혁신적인 방법론입니다. 기존의 전통적인 기법들과는 달리 TREB는 BERT 모델을 특정하게 조정하여 연속적인 실수값을 보충하는 데 초점을 맞추었습니다.

- **Technical Details**: TREB는 BERT (Bidirectional Encoder Representations from Transformers) 모델을 기반으로 하며, 표 형식 데이터에서의 고유한 도전 과제를 다루기 위해 설계되었습니다. 이 방법론은 연속적인 숫자를 보충하는 데 있어 문맥 기반 상호 연관성을 중요시하며, 캘리포니아 주택 데이터셋을 통해 효과성을 입증했습니다. 논문에서는 플로팅 포인트 연산(FLOPs)과 카본 풋프린트에 대한 평가도 포함되어 있습니다.

- **Performance Highlights**: TREB는 특성 상호관계를 보존하며 결측값을 정확하게 보충하는 능력을 보여줍니다. 연구를 통해 TREB의 계산 효율성과 환경적 영향을 명확히 정량화하였으며, 결측값 대체에서의 우수성을 강조했습니다.



### Dual Consolidation for Pre-Trained Model-Based Domain-Incremental Learning (https://arxiv.org/abs/2410.00911)
- **What's New**: 이 논문에서는 Domain-Incremental Learning (DIL)에서의 지식 소멸 문제를 해결하기 위해 DUal ConsolidaTion (Duct) 방법을 제안합니다. 이는 모델의 표현(representation) 및 분류기(classifier) 차원에서 역사적 지식을 통합하는 방식으로, 이전 도메인의 지식을 보존하면서 새로운 도메인에 적응할 수 있도록 합니다.

- **Technical Details**: DUct 방법은 다양한 도메인에서 축적된 정보를 통합하기 위해 서로 다른 단계의 백본(backbone)을 병합합니다. 이로 인해 생성된 표현 공간은 모든 도메인에 적합한 정보의 균형 잡힌 중간체를 제공합니다. 또한, 분류기 통합(classifier consolidation) 과정을 통해 최신 임베딩 공간(embedding space)으로의 적격한 분류기 가중치 조정을 수행합니다. 이 과정에서 클래스별 의미 정보(class-wise semantic information)를 활용하여 이전 도메인 클래스의 분류기 가중치를 추정합니다.

- **Performance Highlights**: Duct 방법은 네 가지 벤치마크 데이터 세트에서 수행된 광범위한 실험 결과를 통해 최첨단 성능을 달성하였으며, 이전 지식을 잃지 않고 새로운 지식을 효과적으로 학습할 수 있음을 입증하였습니다.



### Causal Representation Learning with Generative Artificial Intelligence: Application to Texts as Treatments (https://arxiv.org/abs/2410.00903)
- **What's New**: 본 논문에서는 언스트럭처드(불규칙) 고차원 처리(치료) 데이터를 분석할 때 생성 인공지능(Artificial Intelligence, AI)의 힘을 활용하여 인과 추론의 유효성을 향상시키는 방법을 제시합니다. 특히, 대규모 언어 모델(large language models, LLMs)을 활용하여 처리 데이터를 효과적으로 생성하고, 내부 표현을 통해 인과 효과를 추정합니다.

- **Technical Details**: 우리는 TarNet 기반의 신경망 아키텍처를 개발하여 치료 및 혼란 요인을 별도로 학습하고, 비모수적으로 평균 치료 효과를 식별할 수 있음을 증명합니다. 또한, 이 방법론을 통해 겹침 가정(overlap assumption) 위반을 피하는 추정 전략을 제안하며, 이중 기계 학습(double machine learning, DML)을 통해 제안된 추정기의 비대칭 특성을 도출합니다. 마지막으로, 실제와 perceivable 치료 특성을 확인하기 위해 도구 변수(instrumental variables) 접근법을 확장하여 활용합니다.

- **Performance Highlights**: 시뮬레이션 연구 결과, 제안된 추정기는 기존의 인과 표현 학습 알고리즘보다 더 작은 편향(bias)과 평균 제곱근 오차(root mean squared error, RMSE)를 기록하며, 신뢰 구간(confidence interval)은 적정한 명목 커버리지 수준(nominal coverage level)을 유지합니다. AI에 의해 생성된 내부 표현을 활용한 결과, 인과 표현 학습의 성능이 유의미하게 개선되었습니다.



### Do Music Generation Models Encode Music Theory? (https://arxiv.org/abs/2410.00872)
Comments:
          Accepted at ISMIR 2024. Dataset: this https URL Code: this https URL Website: this https URL

- **What's New**: 이 논문에서는 음악 생성 모델들이 음악 이론 (music theory) 개념을 얼마나 잘 인코딩하고 있는지 조사하기 위해 새로운 데이터셋인 SynTheory를 소개합니다. 이 데이터셋은 템포 (tempo), 박자 (time signatures) 및 코드 진행 (chord progressions) 등 다양한 음악 이론 개념을 포함하고 있습니다.

- **Technical Details**: SynTheory 데이터셋은 MIDI (Musical Instrument Digital Interface) 및 오디오 (audio) 파일로 구성되어 있으며, 이를 통해 음악 생성 모델인 Jukebox와 MusicGen의 내부 표현에서 음악 이론 개념들을 탐색하는 프레임워크가 제안됩니다. 이 연구는 각 모델의 크기 (model size)와 레이어 (layer)별로 음악 이론 개념의 인코딩 정도가 어떻게 변하는지를 평가합니다.

- **Performance Highlights**: 연구 결과, 음악 이론 개념들이 음악 생성 모델 내에서 인식 가능하고, 탐지 가능성은 모델의 크기와 레이어에 따라 다르게 나타나는 것으로 보입니다.



### Improved Sample Complexity of Imitation Learning for Barrier Model Predictive Contro (https://arxiv.org/abs/2410.00859)
Comments:
          36 pages, 3 figures. This work extends our previous result in arXiv:2306.01914, which has been accepted for publication in CDC 2024. An earlier version of this manuscript was submitted as part of DP's Master's thesis

- **What's New**: 이 논문에서는 일반적인 시스템에 대해 성능 보장을 위한 스무딩된 전문가 제어기를 설계하는 방법을 제시합니다. 이 접근법은 Model Predictive Control (MPC) 최적화 문제의 로그-장애물(log-barrier) 기반 완화를 활용합니다.

- **Technical Details**: 제안된 방법은 스무딩된 전문가를 복원한 후, 블랙박스(black-box) 모방 학습 알고리즘과 결합하여 최적의 성능 보장을 제공합니다. 특히, 로그-장애물이 포함된 MPC가 랜덤 스무딩(randomized smoothing)보다 우수하다는 이론적 및 실험적 결과를 제공합니다.

- **Performance Highlights**: 실험적으로, 제안된 로그-장애물 MPC 기법이 기존의 랜덤 스무딩 기법보다 높은 성능을 보임을 증명하여 실제 제어 분야에서의 유용성을 입증합니다.



### An EM Gradient Algorithm for Mixture Models with Components Derived from the Manly Transformation (https://arxiv.org/abs/2410.00848)
- **What's New**: Zhu와 Melnykov(2018)는 Manly 변환에서 파생된 혼합 모델의 구성 요소를 갖는 모델을 개발하였습니다.

- **Technical Details**: 그들의 EM 알고리즘은 M단계에서 비대칭 매개변수인 \( \boldsymbol{\lambda}_g \)를 업데이트하기 위해 Nelder-Mead 최적화를 사용합니다. 초기 추정값이 좋을 때 Newton 방법의 한 단계를 사용하는 대체 EM gradient 알고리즘도 제안됩니다.

- **Performance Highlights**: 이 방법은 전체 데이터 셋에서 최적의 솔루션을 찾는 데 빠르고 효율적이며, \( \hat{\boldsymbol{\lambda}}_g \), \( \hat{\boldsymbol{\mu}}_g \), \( \hat{\mathbf{\Sigma}}_g \)의 추정이 동시에 업데이트됩니다.



### Solving High-Dimensional Partial Integral Differential Equations: The Finite Expression Method (https://arxiv.org/abs/2410.00835)
Comments:
          18 pages, 10 figures

- **What's New**: 본 논문에서는 고차원 부분 적분미분방정식(Partial Integro-Differential Equations, PIDEs)을 해결하기 위한 새로운 유한 표현 방법(Finite Expression Method, FEX)을 소개합니다. 이 접근법은 기존 FEX의 장점을 바탕으로 몇 가지 새로운 발전을 추가하였습니다: 1) 고차원 함수 근사를 위한 계수의 수를 줄이기 위해 매개변수 그룹화(parameter grouping)의 새로운 방법을 제안했습니다; 2) PIDEs의 적분 항의 평가에서 계산 효율성과 정확성을 크게 향상시키기 위해 테일러 급수 근사 방법을 구현하였습니다.

- **Technical Details**: 새로운 FEX 기반 방법은 FEX-PG로 설명되며, 이는 매개변수 그룹화(Paramter Grouping) 단계를 알고리즘에 추가한 것을 나타냅니다. FEX-PG는 높은 정확도와 해석 가능한 수치적 솔루션을 제공하며, 결과적으로 기초 솔루션 구조에 대한 직관적인 이해를 Facilitates하는 명시적 방정식을 생성합니다. 또한, 이 방법은 고차원 문제를 해결하기 위해 조합 최적화(combinatorial optimization) 문제로 재구성되며, 이는 적절한 연산자와 상수의 선택을 목표로 합니다.

- **Performance Highlights**: FEX-PG는 높은 차원의 환경에서 강력하고 견고한 성능을 나타내며, 상대 오차가 단정밀도 기계 엡실론의 범위에 해당하는 결과를 달성하였습니다. 이 방법은 기존의 유한 요소 방법(Finite Element Method, FEM) 및 유한 차분 방법(Finite Difference Method, FDM)과 같은 전통적인 방법들, 그리고 심층 학습 기반 접근법에서 결여된 해석 가능성을 극복하는 데 중점을 두었습니다.



### Squeeze-and-Remember Block (https://arxiv.org/abs/2410.00823)
Comments:
          Accepted by The International Conference on Machine Learning and Applications (ICMLA) 2024

- **What's New**: 본 논문에서는 Convolutional Neural Networks (CNNs)에 동적인 메모리 기능을 추가하는 새로운 구조적 단위를 소개합니다. 이 단위는 "Squeeze-and-Remember" (SR) 블록으로, 훈련 중 중요한 특징을 선택적으로 기억하고, 추론 시 이들 특징을 적응적으로 재적용하여 네트워크의 맥락적 예측 능력을 향상시킵니다.

- **Technical Details**: SR 블록은 입력 특성 맵에 대해 1×1 컨볼루션을 수행하여 중요한 정보를 추출하고, 이를 두 층의 Fully Connected Network (FCN)를 통해 P개의 메모리 블록과 연계하여 가중치를 생성합니다. 최종 출력은 원래 입력에 메모리 블록에서 얻은 높은 수준의 특징을 추가하여 생성됩니다. SR 블록은 ResNet50과 DeepLab v3 모델에 통합되어 성능 향상을 보여줍니다.

- **Performance Highlights**: SR 블록을 ResNet50에 통합한 결과, ImageNet 데이터셋에서 top-1 validation accuracy가 dropout2d 단독 사용 대비 0.52% 향상되었습니다. 또한, Cityscapes 데이터셋의 DeepLab v3 모델 적용 시 mean Intersection over Union이 0.20% 증가했습니다. 이 모든 개선은 최소한의 계산 오버헤드로 이루어졌습니다.



### Fast and Reliable $N-k$ Contingency Screening with Input-Convex Neural Networks (https://arxiv.org/abs/2410.00796)
Comments:
          11 pages, 4 figures

- **What's New**: 본 연구에서는 Input-Convex Neural Networks (ICNNs)를 활용하여 전력 시스템 격리 분석을 수행하는 방법을 제안합니다. 이 방법은 모든 가능한 격리 상태를 분석하는 기존 방식의 비효율성을 개선하고, 실시간으로 격리 분석의 속도를 크게 향상시킵니다.

- **Technical Details**: 제안된 방식은 ICNN의 신뢰성을 보장하기 위해 볼록 최적화 문제를 해결하는 것으로 구성되어 있습니다. 또한, 모델 파라미터를 적절하게 조정하여 제로(true) false negative rate를 보장할 수 있습니다. 이 연구는 IEEE 39-bus 테스트 네트워크 사례를 통해 10-20배의 연산 시간 단축과 0%의 false negative rate를 달성했습니다.

- **Performance Highlights**: 모델은 전력 시스템의 신뢰성을 높이면서도, 격리 분석에 필요한 계산 부담을 줄였습니다. 실험 결과는 우수한 분류 정확성(2-5% false positive rate)과 함께, 예방 조치에 있어 10배 빠른 속도를 보여주었습니다.



### Adaptive Motion Generation Using Uncertainty-Driven Foresight Prediction (https://arxiv.org/abs/2410.00774)
- **What's New**: 이 논문은 동적 내부 시뮬레이션을 활용한 예측적 학습 기반 로봇 제어 방법을 확장하여, 환경의 불확실성에 적응적으로 대응할 수 있는 모델을 제안합니다.

- **Technical Details**: 제안된 모델은 심층 예측 학습 프레임워크를 기반으로 하며, RNN에 대한 미래 예측 모듈을 도입하여 환경의 동적 불확실성을 정확하게 모델링합니다. 이를 통해 로봇이 불확실한 상황에서 최적의 행동을 탐색하고 유연한 움직임을 생성할 수 있도록 돕습니다.

- **Performance Highlights**: 문을 여는 작업에서 제안된 모델은 80% 이상의 성공률을 기록하며, 전통적인 RNN 모델에 비해 적응적인 동작 예측을 수행하였고 모든 세 가지 동작에서 움직임을 효과적으로 분기할 수 있었습니다.



### On the Generalization and Causal Explanation in Self-Supervised Learning (https://arxiv.org/abs/2410.00772)
- **What's New**: 이 논문에서는 Self-supervised learning (SSL) 방법들이 훈련 데이터에 과적합(overfitting) 되는 현상을 관찰하고, 이를 해결하기 위한 새로운 메커니즘인 Undoing Memorization Mechanism (UMM)을 제안합니다. UMM은 마지막 레이어의 피처 분포를 초기 레이어와 정렬하는 방식으로 과적합 문제를 완화합니다.

- **Technical Details**: 연구를 통해, SSL 모델이 훈련 초기 레이어에서 일반화 성능을 학습하고 마지막 레이어에서 기억화를 시작함을 발견하였습니다. Coding rate reduction을 활용하여 과적합 정도를 정량화할 수 있음을 보여주었으며, UMM은 이중 최적화 과정으로 설계되어 특징 추출기의 초기 레이어 특성과의 조화를 이루어 마지막 레이어 특성의 일반화를 회복합니다.

- **Performance Highlights**: UMM을 적용한 SSL 방법들이 다양한 다운스트림 작업에서 일반화 성능을 현저하게 개선하는 것을 실험을 통해 입증하였습니다. UMM은 빠르게 SSL 기술과 통합될 수 있는 플러그 앤 플레이 방식입니다.



### WALINET: A water and lipid identification convolutional Neural Network for nuisance signal removal in 1H MR Spectroscopic Imaging (https://arxiv.org/abs/2410.00746)
- **What's New**: 본 연구에서는 고해상도 1H-MRSI에서의 물 및 지방 신호 제거를 위한 심층 학습 방법을 소개합니다. 새로운 WALINET 네트워크는 기존의 방법들보다 더 빠르고 효과적으로 신호를 처리할 수 있습니다.

- **Technical Details**: WALINET(Network for Water and Lipid) 은 수정된 Y-NET 기반의 심층 신경망으로, 1H-MRSI 스펙트럼에서의 물 및 지방 신호 제거를 수행합니다. 시뮬레이션 및 실제 데이터에서 NMRSE, SNR, CRLB, FWHM 등의 메트릭을 사용하여 비교 평가하였습니다.

- **Performance Highlights**: WALINET은 8초 이내에 고해상도 전체 뇌 1H-MRSI 데이터를 처리할 수 있으며, 이는 기존 HLSVD+L2 방법의 42분과 비교하여 매우 빠릅니다. 또한 WALINET은 지방 제거에서 41% 낮은 NRMSE를 기록하고, 메타볼라이트 신호 보존에서도 71% 낮은 NRMSE와 155% 높은 SNR, 50% 낮은 CRLB를 보여주었습니다.



### Improved Generation of Synthetic Imaging Data Using Feature-Aligned Diffusion (https://arxiv.org/abs/2410.00731)
Comments:
          Accepted to First International Workshop on Vision-Language Models for Biomedical Applications (VLM4Bio 2024) at the 32nd ACM-Multimedia conference

- **What's New**: 이 논문에서는 기존의 확산 모델을 통해 의료 이미지를 생성하는 기존 접근 방식에서 개선된 'feature-aligned diffusion' 방법을 탐색하였습니다. 이 방법은 전문가 모델의 출력 특징과 확산 모델의 중간 특징을 정렬하여 생성 정확도를 9% 향상시키고 SSIM 다양성에서 약 0.12의 개선을 보였습니다.

- **Technical Details**: Feature-aligned diffusion은 확산 모델의 중간 특징을 전문가 모델의 출력 특징과 정렬하는 방식을 사용합니다. 이 과정에서는 전문가 모델의 클래시피케이션(output classification) 결과를 활용하며, 새로운 손실 함수를 도입하여 이 두 결과 간의 코사인 유사도(cosine similarity)를 극대화합니다. 기존의 훈련 절차에 비해 추가적인 프로젝션 레이어(projection layer)만 요구되어 기존의 훈련 파이프라인에 쉽게 통합될 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법인 feature-aligned diffusion을 통해 생성 이미지의 품질이 향상되었으며, 특히 훈련 시 노이즈를 추가한 이미지에서 전문가 모델의 특징을 정렬하는 것이 성능 향상에 기여했습니다. 기존의 방법에 비해 개선된 성능을 보이며, 이는 의료 이미지 생성 분야에서 중요한 진전을 이룬 것으로 평가됩니다.



### Simplified priors for Object-Centric Learning (https://arxiv.org/abs/2410.00728)
- **What's New**: 이번 논문에서는 인간의 데이터 추상화 능력을 모방하기 위해 간단하고 효율적인 SAMP(Simplified Slot Attention with Max Pool Priors) 방법을 제안합니다. 이 방법은 기존의 복잡한 Object-Centric 학습 접근법들보다 구현이 간편하며, 이미지에서 슬롯을 추출하는 데 있어 효율성을 높입니다.

- **Technical Details**: SAMP는 Convolution 및 MaxPool 레이어와 Attention 레이어를 사용하여 이미지 데이터를 처리합니다. Convolutional Neural Network(CNN)를 통해 입력 이미지를 인코딩하고, 이후 Convolution과 MaxPool 레이어의 교차 분기를 통해 특수한 하위 네트워크를 생성하여 원시 슬롯(primitive slots)을 추출합니다. 이 원시 슬롯들은 인코딩된 이미지에 대해 Simplified Slot Attention의 쿼리로 사용됩니다.

- **Performance Highlights**: SAMP는 기존 Object-Centric 방법들과의 비교에서 경쟁력이 있거나 이를 초월하는 성과를 보입니다. 특히, SAMP는 비반복적이고 완전하게 미분 가능한 방식으로 슬롯을 추출하므로, 기존의 Slot Attention 기반 방법들의 반복적 세련화 절차가 필요하지 않습니다.



### Discriminative community detection for multiplex networks (https://arxiv.org/abs/2410.00724)
- **What's New**: 이 논문에서는 두 개의 밀접하게 관련된 멀티플렉스 네트워크(multiplex networks)에서의 커뮤니티 구조를 탐지하는 문제를 다루고 있습니다. 특히, 신경 이미징 연구(neuroimaging studies)와 같은 응용에서, 각각의 층이 개별 실험 조건을 나타내는 경우에 대한 해결책을 제시합니다.

- **Technical Details**: 제안된 두 가지 알고리즘은 스펙트럴 클러스터링(spectral clustering)을 기반으로 하며, 첫 번째는 그룹 간의 구별되는 하위 그래프(subgraph) 구조를 식별하며, 두 번째는 구별되는 커뮤니티 구조와 합의 커뮤니티 구조(consensus community structure)를 동시에 학습합니다. 이러한 두 방법은 합성 네트워크(synthetic networks) 및 실제 멀티플렉스 네트워크에서 평가되었습니다.

- **Performance Highlights**: 제안된 알고리즘은 다양한 실험 조건을 비교하는 데 효과적이며, 커뮤니티 구조를 명확하게 나타내기도 합니다. 실제로 기능성 뇌 네트워크(functional brain networks)에서의 실험 데이터를 통해 강력한 성능을 입증했습니다.



### NECOMIMI: Neural-Cognitive Multimodal EEG-informed Image Generation with Diffusion Models (https://arxiv.org/abs/2410.00712)
- **What's New**: NECOMIMI는 EEG 신호로부터 이미지를 생성하기 위한 혁신적인 프레임워크로, 고급 diffusion 모델을 활용합니다. 이 연구는 기존의 EEG 이미지를 분류하는 방식을 넘어 이미지 생성을 포함하는 새로운 접근법을 제시합니다.

- **Technical Details**: NECOMIMI는 NERV EEG 인코더를 제안하며, 이는 멀티모달 대비 학습(Multimodal Contrastive Learning) 작업에서 최첨단 성능을 달성합니다. 기존 EEG 특성을 이미지 생성에만 사용하는 대신, 이 연구는 EEG에서 이미지 생성으로의 두 단계 멀티모달 생성 프레임워크를 도입합니다. 또한, 새로운 CAT Score 메트릭을 통해 생성된 이미지의 질을 평가합니다.

- **Performance Highlights**: NECOMIMI는 2-way, 4-way 및 200-way의 제로샷 분류 작업에서 최고의 성능을 기록하였으며, EEG로 생성된 이미지의 품질을 평가하기 위한 CAT Score 기준을 설정하였습니다. 연구 결과, 생성된 이미지는 특정 객체보다 추상적이고 일반화된 형태의 풍경을 생성하는 경향을 보였습니다.



### Hybrid Quantum Neural Network based Indoor User Localization using Cloud Quantum Computing (https://arxiv.org/abs/2410.00708)
Comments:
          This work has been accepted for presentation at the IEEE TENSYMP 2024 conference

- **What's New**: 본 논문에서는 수신 신호 강도 지표(Received Signal Strength Indicator, RSSI) 값을 활용한 실내 사용자 위치 추정을 위한 하이브리드 양자 신경망(Hybrid Quantum Neural Network, HQNN)을 제안합니다. 연구팀은 실내 위치 추정을 위해 WiFi, Bluetooth 및 Zigbee를 활용한 공개 RSSI 데이터 세트를 사용하여 HQNN의 성능을 평가하였으며, 최근 제안된 양자 지문화 기반 사용자 위치 추정 방법과 비교하였습니다. 특히 하이브리드 양자 신경망은 양자 회로에서 조정 가능한 매개변수를 가지고 있어, 기존의 고정된 양자 회로를 사용하는 지문화 알고리즘보다 더 높은 성능을 보여주었습니다.

- **Technical Details**: HQNN은 양자 및 고전 신경망 컴포넌트를 통합하여 기능을 극대화하고, 제조된 데이터의 비선형 다차원 벡터 공간을 모델링합니다. 이 네트워크는 양자 컴퓨터에서 직접 훈련 및 시험되었으며, NISQ(노이즈가 있는 중간 규모 양자 컴퓨터) 장치의 성능을 연구합니다. HQNN은 RSSI 데이터를 사용하여 학습하며, 각 데이터 포인트의 차원에 따라 필요한 큐비트 수가 결정되는 구조로, 데이터 세트의 크기와 무관하게 확장 가능합니다.

- **Performance Highlights**: HQNN은 훈련 과정에서 더 빠르게 수렴하며, 신호 간섭이 증가하는 상황에서도 기존의 고전 신경망(NN)에 근접하거나 능가하는 성능을 나타냈습니다. 실질적인 IBM 양자 하드웨어에서의 성능이 시뮬레이터와 크게 차이나지 않아, 양자 하드웨어 노이즈에 대한 감도가 낮음을 보여주었습니다. 이는 HQNN의 훈련 가능한 매개변수와 일정한 큐비트 수 덕분에 일반적인 양자 지문화 알고리즘보다 더 실용적임을 의미합니다.



### Optimizing Photoplethysmography-Based Sleep Staging Models by Leveraging Temporal Context for Wearable Devices Applications (https://arxiv.org/abs/2410.00693)
Comments:
          11 pages, 5 figures, 1 table

- **What's New**: 이번 연구에서는 휴대용 장치에서의 수면 모니터링을 개선하기 위해 Photoplethysmography (PPG) 신호를 활용한 새로운 수면 단계 분류 모델을 제안합니다.

- **Technical Details**: 제안된 모델은 SleepPPG-Net을 기반으로 하며, 신호 집합을 30초 PPG 조각으로 결합하여 15분 간격으로 길어진 문맥을 활용합니다. 이를 통해 각 수면 단계의 정확도를 향상시켰습니다.

- **Performance Highlights**: 모델의 정확도는 0.75, Cohen's Kappa는 0.60, F1-Weighted 점수는 0.74, F1-Macro 점수는 0.60을 기록하였습니다. 세밀한 수면 단계에서의 정확도는 다소 떨어졌지만, 단일 30초 윈도우 방법보다 뛰어난 성능을 보였습니다.



### Advanced Arabic Alphabet Sign Language Recognition Using Transfer Learning and Transformer Models (https://arxiv.org/abs/2410.00681)
Comments:
          6 pages, 8 figures

- **What's New**: 본 논문에서는 깊이 있는 학습(deep learning) 방법과 전이 학습(transfer learning), 그리고 transformer 기반 모델을 이용한 아랍어 알파벳 수화 인식 방안을 제시하고 있습니다. 아랍 수화(Arabic Sign Language) 동작의 독특한 특징을 포착하기 위해 ArSL2018과 AASL이라는 두 개의 공공 데이터셋에서 다양한 변형 모델들의 성능을 연구하였습니다.

- **Technical Details**: 이 연구는 최신 CNN 아키텍처인 ResNet50, MobileNetV2, EfficientNetB7 및 Google ViT와 Microsoft Swin Transformer와 같은 최신 transformer 모델을 활용합니다. 컨볼루션 신경망(CNN) 모델과 transformer를 활용한 특징 추출을 포함하는 여러 주요 단계로 구성된 아랍어 알파벳 수화 인식 시스템을 개발하였습니다. 이 시스템은 데이터 전처리(data preprocessing), 모델 선택(model selection with transfer learning), 모델 평가(model evaluation)로 이루어져 있습니다.

- **Performance Highlights**: 실험 결과 ArSL2018과 AASL 데이터셋에서 각각 99.6%와 99.43%의 높은 인식 정확도를 달성하였으며 이는 기존의 최첨단(leading-edge) 접근 방식들을 훨씬 초월하는 결과입니다. 이러한 성능 향상은 아랍어를 사용하는 청각 장애인 및 난청인에게 더 접근성 높은 커뮤니케이션 방법을 제공하고, 포용적인 사회를 촉진하는 데 기여할 것입니다.



### TAVRNN: Temporal Attention-enhanced Variational Graph RNN Captures Neural Dynamics and Behavior (https://arxiv.org/abs/2410.00665)
Comments:
          31 pages, 6 figures, 4 supplemental figures, 4 tables, 8 supplemental tables

- **What's New**: TAVRNN(Temporal Attention-enhanced Variational Graph Recurrent Neural Network)은 외부 자극 및 행동 피드백에 대한 신경 연결망의 진화적 역학 분석을 위한 새로운 프레임워크를 제공합니다. 이 모델은 신경 활동의 순차적 스냅샷을 통해 네트워크 구조의 시간적 변화를 포착하여 주요 연결 패턴을 식별할 수 있습니다.

- **Technical Details**: TAVRNN은 temporal attention 메커니즘과 variational graph 기술을 활용하여 시간에 따라 연결성이 어떻게 변화하는지를 밝혀냅니다. 실험에서는 자유롭게 행동하는 쥐에서 수집한 생리학적 데이터와 DishBrain 시스템에서의 전기생리학적 데이터를 사용하여 모델을 검증했습니다.

- **Performance Highlights**: TAVRNN은 분류 및 군집화 작업에서 이전 기준 모델보다 뛰어난 성능을 보였으며, 연결성 변화와 성과 변화를 정확히 연계했습니다. 특히 DishBrain 시스템에서의 높은 게임 성능은 감각 및 운동 서브리전 채널 간의 정렬과 상관관계가 있다는 점을 밝혀냈습니다.



### LASMP: Language Aided Subset Sampling Based Motion Planner (https://arxiv.org/abs/2410.00649)
Comments:
          8 pages, 9 figures

- **What's New**: 이번 논문에서는 자연어(Natural Language) 지침을 이용하여 이동 로봇이 움직임을 계획할 수 있도록 돕는 LASMP(Language Aided Subset Sampling Based Motion Planner) 시스템을 제안합니다. LASMP는 사용자 제공 명령을 처리하는 언어 모델(RoBERTa)을 통해 가이드되는 수정된 RRT(Rapidly Exploring Random Tree) 방법을 사용합니다.

- **Technical Details**: LASMP는 사용자가 제공한 지침을 바탕으로 로봇 작업 영역의 특정 영역에 초점을 맞추어 효율성을 높입니다. 이는 전통적인 RRT 방법에 비해 노드 필요 수를 55% 줄이고, 랜덤 샘플 쿼리를 80% 감소시키면서 안전하고 충돌이 없는 경로를 생성합니다. 이 시스템은 텍스트나 음성으로 된 사용자 지침을 수신하여 목표 지점 및 방향 지시를 식별하며, 이를 바탕으로 경로를 계산합니다.

- **Performance Highlights**: 모의 환경과 실제 환경 모두에서 테스트한 결과, LASMP는 복잡한 실내 상황을 처리하는 데 있어 더 나은 성능을 보였으며, 로봇 내비게이션을 보다 효율적으로 만들기 위한 언어 처리와 모션 계획의 결합 가능성을 강조합니다.



### Differentiable Interacting Multiple Model Particle Filtering (https://arxiv.org/abs/2410.00620)
- **What's New**: 이번 논문에서는 신경망과 같은 고차원 매개변수 세트를 학습하기 위한 새로운 확률적 접근법인 differentiable particle filtering을 도입합니다. 새로운 differentiable interacting multiple model particle filter (DIMMPF)를 통해 행동 모드와 모델을 동시에 학습하게 되어, 기존 알고리즘에 비해 computational effort를 조절할 수 있는 장점이 있습니다.

- **Technical Details**: DIMMPF 알고리즘은 상태 전이 모델(state-space model)의 관점에서 두 개의 평행한 프로세스를 모델링 하며, 각 프로세스는 자기 자신의 행동 regime을 가지도록 합니다. 이 알고리즘은 또한 low variance의 gradient estimator를 개발하여, 계산 속도를 높이고 consistency를 보장합니다.

- **Performance Highlights**: DIMMPF는 시뮬레이션 데이터 실험에서 우수한 성능을 보이며, 기존 최첨단 알고리즘들과의 성능 비교에서 우수한 결과를 나타냅니다. 이 알고리즘은 필터링 평균의 일관된 추정치를 생성하는 것을 보장합니다.



### Radio Foundation Models: Pre-training Transformers for 5G-based Indoor Localization (https://arxiv.org/abs/2410.00617)
- **What's New**: 이 논문은 전통적인 로컬라이제이션(localization) 방법에 비해 AI 기반 무선 핑거프린팅(radio fingerprinting)의 성능을 개선하는 새로운 자기 지도 학습(self-supervised learning) 프레임워크를 제안합니다. 특히, 5G 채널 측정 데이터를 활용하여 비용과 시간을 크게 절감하면서도 최고 수준의 정확도를 달성합니다.

- **Technical Details**: 제안된 방법은 Transformer(트랜스포머) 신경망을 이용한 두 단계 훈련으로 구성됩니다. 첫 번째 단계에서는 비지도 학습으로 위치 특정 채널 패턴을 학습하고, 두 번째 단계에서는 소량의 참조 측정값을 가지고 훈련된 모델을 미세 조정(fine-tuning)하여 고-정확도의 로컬라이제이션을 이룹니다. 이 과정에서 시뮬레이터가 아닌 실제 5G 데이터셋을 사용합니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 최신 감독 학습(supervised learning) 방법 및 무선 핑거프린팅 방법보다 더 나은 성능을 보였으며, 참조 데이터 요구량을 10배 줄이는 동시에 교육에서 운영까지의 시간을 크게 줄였습니다.



### Arges: Spatio-Temporal Transformer for Ulcerative Colitis Severity Assessment in Endoscopy Videos (https://arxiv.org/abs/2410.00536)
Comments:
          12 pages, 2 figures, 5 tables, accepted at MLMI, MICCAI

- **What's New**: 이번 논문에서는 위염병(UC)의 내시경 비디오에서 질병 중증도를 평가하기 위한 새로운 딥러닝 프레임워크인 "Arges"를 제안합니다. 이 모델은 공간-시간(spatio-temporal) 정보를 통합하여 내시경 비디오의 질병 중증도를 더욱 정확하게 추정할 수 있게 합니다.

- **Technical Details**: Arges 프레임워크는 positional encoding을 통해 공간-시간(spatio-temporal) 정보를 통합하는 transformer 기반 분류기를 포함하고 있습니다. ArgesFM이라는 강력한 기반 모델을 사용하여 61M 프레임의 대규모 데이터에서 학습한 후, 질병 중증도 점수를 추정하기 위한 추가적인 분류기를 적용합니다.

- **Performance Highlights**: 실험 결과, MES 점수에서 F1 점수가 4.1% 향상되었으며, UCEIS 구성 점수에서도 각각 18.8%, 6.6%, 3.8%의 개선을 보였습니다. 추가적으로, 이전에 본 적이 없는 임상 시험 데이터에 대한 유망한 검증 결과도 나타났습니다.



### Multi-Target Cross-Lingual Summarization: a novel task and a language-neutral approach (https://arxiv.org/abs/2410.00502)
Comments:
          Accepted to EMNLP 2024 (Findings)

- **What's New**: 이번 논문에서는 여러 목표 언어를 고려하는 multi-target cross-lingual summarization (MTXLS)이라는 새로운 과제를 소개합니다. 이 과제는 여러 언어에서 문서를 요약하되, 생성된 요약이 의미적으로 유사하도록 하는 데 중점을 두고 있습니다.

- **Technical Details**: MTXLS는 여러 목표 언어 간 의미 일관성(semantic coherence)을 보장하기 위한 새로운 프레임워크로, re-ranking 방식으로 의미 있게 요약을 선택합니다. 이 접근 방식은 언어 중립(language-neutral) 전략을 채택하여, 성과의 신뢰성을 높입니다. 또한, 기계 번역의 품질 추정(quality estimation) 방법을 사용하여 생성된 요약의 일관성을 평가하는 다중 기준 평가 프로토콜(multi-criteria evaluation protocol)을 제안합니다.

- **Performance Highlights**: 연구에서는 기존의 cross-lingual summarization 방법들이 주로 단일 언어 쌍에 초점을 맞추었으나, MTXLS는 다양한 타깃 언어에서의 의미 일관성을 보장합니다. 이 결과는 법적 또는 규제적 요구 사항을 충족하는 데에도 중요한 역할을 할 수 있습니다.



### Stability analysis of chaotic systems in latent spaces (https://arxiv.org/abs/2410.00480)
- **What's New**: 본 논문에서는 저자가 제안한 'latent-space (잠재 공간)' 접근 방식을 적용하여 복잡한 시스템의 동역학을 예측 및 안정성을 특징화하는 방법을 소개합니다. 특히, CAE-ESN(convolutional autoencoder echo state network)을 사용하여 카오틱한 Kuramoto-Sivashinsky 방정식을 분석하고, 교란이 있는 시스템의 리아풀러 수치와 관련된 결과를 보여줍니다.

- **Technical Details**: 연구에서는 'latet-space (잠재 공간)' 방법론을 통하여 데이터 드리븐(data-driven) 방식을 적용하여 카오틱 시스템의 동역학을 모델링합니다. CAE-ESN 구조는 데이터를 압축하여 저차원(latent) 공간 표현으로 변환하고, 순환 신경망(recurrent neural network)을 사용하여 이 저차원 공간 내에서 시간적 진화를 예측합니다.

- **Performance Highlights**: CAE-ESN을 통해 카오틱한 시스템에 대한 리아풀러 수와 공변 리아풀러 벡터(covariant Lyapunov vectors)를 정확하게 추정할 수 있으며, 이는 물리적 시스템의 안정성을 예측하는 데 기여합니다. CAE-ESN 모델은 저차원 모델로서 카오틱 시스템의 동역학을 정확하게 예측할 수 있습니다.



### Uncertainty-aware t-distributed Stochastic Neighbor Embedding for Single-cell RNA-seq Data (https://arxiv.org/abs/2410.00473)
- **What's New**: 본 연구에서는 기존의 t-SNE 기법에서 데이터의 불확실성을 고려하지 못하는 문제를 해결하기 위해 불확실성을 고려한 t-SNE(Ut-SNE)를 도입했습니다. Ut-SNE는 단일 세포 RNA 시퀀싱 데이터의 노이즈를 효과적으로 처리하고, 데이터 내 transcriptomic 변동성을 시각적으로 표현하여 생물학적 인사이트를 높이는 데 기여합니다.

- **Technical Details**: Ut-SNE는 불확실한 데이터에 적합하도록 t-SNE를 확장한 새로운 알고리즘으로, 각 샘플에 대해 확률적 표현을 생성합니다. 연구진은 입력 데이터가 정규분포를 따른다고 가정하고, 관측 값에 보조 분산 항을 추가하여 변동성을 반영합니다. 이는 데이터 포인트 간의 거리의 기대값을 나타내는 불확실한 유사도를 통해 t-SNE의 최적화 문제를 비확률 공간으로 변환하며, 더욱 신뢰할 수 있는 시각화를 제공합니다.

- **Performance Highlights**: Ut-SNE는 여러 공개된 단일 세포 RNA 시퀀싱 데이터셋에서 적용되어, 기존의 t-SNE보다 더 나은 데이터 구조 보존을 보여주었습니다. 노이즈가 있는 정보의 시각화뿐만 아니라, 단일 세포 수준에서의 생물학적 인사이트를 발견하는 데 중요한 역할을 했습니다. Ut-SNE는 숨겨진 하위 구조와 클러스터의 분포를 밝혀내며, 이전에 발견되지 않은 패턴을 드러내는 가능성을 보여주었습니다.



### LayerKV: Optimizing Large Language Model Serving with Layer-wise KV Cache Managemen (https://arxiv.org/abs/2410.00428)
Comments:
          11 pages, 7 figures, 1 table

- **What's New**: 본 논문에서는 대형 언어 모델(LLMs)의 확장된 컨텍스트 윈도우가 제공하는 이점과 이로 인해 발생하는 Time to First Token (TTFT) 지연 문제를 다루고 있습니다. 이를 해결하기 위해 LayerKV라는 간단하면서도 효과적인 방법을 제안하여 TTFT를 줄이고 사용자 경험을 향상시킵니다.

- **Technical Details**: LayerKV는 레이어별 KV 블록 할당, 관리 및 오프로드를 소개하여 시스템 메모리의 세밀한 제어를 가능하게 하며, SLO(Service Level Objectives)를 최적화하기 위한 SLO 인식 스케줄러와 결합됩니다. 이를 통해 GPU의 KV cache 자원의 한계로 인한 쿼리 지연 문제를 완화합니다.

- **Performance Highlights**: LayerKV는 7B에서 70B까지 다양한 매개변수를 가진 모델에서 TTFT 지연을 최대 11배 개선하고, SLO 위반 비율을 28.7% 감소시키며, 전반적으로 사용자 경험을 크게 향상시킵니다.



### ECORS: An Ensembled Clustering Approach to Eradicate The Local And Global Outlier In Collaborative Filtering Recommender System (https://arxiv.org/abs/2410.00408)
Comments:
          6 pages, 5 figures

- **What's New**: 이번 논문에서는 기존의 추천 시스템에서의 이상값(Outlier) 탐지를 개선하기 위해 다양한 클러스터링(clustering) 알고리즘을 활용한 새로운 접근법을 제안합니다. 특히, 사용자-사용자 매트릭스를 기반으로 한 클러스터링 기법을 통해 시스템 내에서 의심스러운 사용자를 식별합니다.

- **Technical Details**: 사용자 간의 유사성을 기반으로 한 매트릭스를 구성하고, 이를 통해 로컬(local) 및 글로벌(global) 이상값을 감지합니다. 클러스터링 기반 접근법을 사용하여 비정상적인 데이터가 포함된 작은 클러스터를 생성하고, 이러한 데이터를 포함하는 클러스터는 일반 데이터 객체의 밀집 클러스터와 대조됩니다.

- **Performance Highlights**: 실험 결과 이 접근법은 추천 시스템에서 이상값 탐지의 정확성을 상당히 향상시키는 것으로 나타났습니다. 다양한 클러스터링 알고리즘의 성능을 비교함으로써, 보다 명확하게 데이터의 이상값을 예측할 수 있음을 입증하였습니다.



### A Generalized Mean Approach for Distributed-PCA (https://arxiv.org/abs/2410.00397)
Comments:
          17 pages, 1 table, 1 figure

- **What's New**: 본 논문에서는 고유값 정보를 통합하여 지역 결과를 집계하는 혁신적인 분산 주성분 분석 방법인 \( \beta \)-DPCA를 제안합니다. 이는 새로운 매트릭스 \( \beta \)-mean을 활용하여 집계하는 방식입니다.

- **Technical Details**: \( \beta \)-DPCA는 집계 방법으로 매트릭스 \( \beta \)-mean을 사용하며, \( \beta \) 값의 조정으로 유연하고 견고한 집계를 가능하게 합니다. 예를 들어, \( \beta = 1 \)일 경우 산술 평균, \( \beta = -1 \)일 경우 조화 평균, \( \beta \to 0 \)일 경우 기하 평균과 이에 해당합니다. 또한, 매트릭스 \( \beta \)-mean은 Bregman matrix divergence의 하위 클래스로서 \( \beta \)-divergence와 연관되어 있습니다.

- **Performance Highlights**: 제안하는 방법의 성능은 수치적 연구를 통해 평가되었으며, 고유값의 변화에 따른 고유벡터 순서의 안정성을 연구하였습니다.



### Dynamic neurons: A statistical physics approach for analyzing deep neural networks (https://arxiv.org/abs/2410.00396)
Comments:
          11 pages, 6 figures

- **What's New**: 본 논문은 딥 뉴럴 네트워크(DNN)의 구조적 패턴을 이해하고, 신경망의 상호작용을 단순화하여 직관적으로 파악할 수 있는 새로운 접근법인 '동적 뉴런(dynamic neuron)' 방법을 제안합니다. 이 방법은 DNN의 대칭성을 드러내면서, 통계 물리(statistical physics)의 기법을 활용할 수 있는 기초를 제공합니다.

- **Technical Details**: 본 연구에서는 DNN의 복잡한 구조를 단순화하는 '동적 뉴런' 접근 방식을 개발하며, 이를 통해 비용 함수(cost function)를 대칭성이 드러나는 형태로 재구성합니다. 이 방식은 DNN의 비선형 활성화 함수(non-linear activation function)와 신경 연결로 영향을 받는 뉴런 사이의 상호작용을 새로운 차원에서 재조정합니다. 이로 인해 DNN의 연구에 RG(renormalization group) 변환과 같은 통계 물리의 도구들이 적용될 수 있게 됩니다.

- **Performance Highlights**: 이 접근법은 DNN의 비율 행동(scaling behavior)과 임계 현상(critical phenomena)을 효과적으로 분석할 수 있는 가능성을 열어주며, DNN의 훈련 과정(training process)을 RG 변환으로 설명할 수 있는 방법을 제시합니다. 특히, 본 연구는 DNN과 상관된 여러 물리적 현상들과의 연결점을 탐구하여 더 나아가 물리학적 이해를 심화할 수 있는 역할을 할 것으로 기대됩니다.



### Seamless Augmented Reality Integration in Arthroscopy: A Pipeline for Articular Reconstruction and Guidanc (https://arxiv.org/abs/2410.00386)
Comments:
          8 pages, with 2 additional pages as the supplementary. Accepted by AE-CAI 2024

- **What's New**: 이번 논문에서는 단안(전방향) 관절경 비디오를 바탕으로 수술 중 인식을 향상시키기 위한 파이프라인을 제시합니다. 이 시스템은 SLAM(Simultaneous Localization and Mapping), 깊이 추정, 3D Gaussian splatting을 결합하여 관절 내 구조를 실시간으로 재구성합니다.

- **Technical Details**: 제안된 파이프라인은 OneSLAM을 사용해 희소 3D 포인트 맵을 구성하고, 단안 깊이 추정 모델을 활용하여 각 프레임마다 불확실한 깊이 정보를 생성합니다. 마지막으로, 3D GS 모델을 활용하여 사진 현실적인 3D 장면을 세밀하게 재구성합니다. 이 기술은 AR(Augmented Reality) 어플리케이션을 통해 관절 구조의 측정 및 주석 개발을 가능하게 합니다.

- **Performance Highlights**: 이 파이프라인은 평균 7분 안에 높은 밀도의 3D 재구성을 달성하고, RMSE=2.21mm 재구성 오류, PSNR=32.86, SSIM=0.89의 성능을 기록했습니다. AR 측정 도구는 평균 1.59 +/- 1.81mm의 정확도를 보여주었으며, AR 주석 도구는 mIoU=0.721의 성과를 나타냈습니다.



### CXPMRG-Bench: Pre-training and Benchmarking for X-ray Medical Report Generation on CheXpert Plus Datas (https://arxiv.org/abs/2410.00379)
Comments:
          In Peer Review

- **What's New**: 본 논문은 X-ray 이미지 기반의 의학 보고서 생성을 위한 새로운 대규모 데이터셋인 CheXpert Plus와 함께, 이를 기반으로 한 CXPMRG-Bench 벤치마크를 제안하여 X-ray 보고서 생성을 위한 기존 알고리즘과 대형 모델들의 성능을 비교합니다.

- **Technical Details**: X-ray 보고서 생성을 위한 MambaXray-VL이라는 새로운 대형 모델을 제안하였으며, 이 모델은 자가 감독형(autoregressive) 생성과 X-ray 보고서 대비 학습(constrastive learning) 전략을 포함한 다단계(pre-training) 방식으로 훈련됩니다. 이는 기존의 Transformer 기반 모델들에서 발생하는 높은 계산 비용을 절감하며, 이미지와 텍스트 비교학습을 통해 더욱 효과적인 기능 공간 정렬을 달성합니다.

- **Performance Highlights**: MambaXray-VL은 CXPMRG-Bench 벤치마크에서 19개의 주요 X-ray 의학 보고서 생성 알고리즘과 14개의 대형 언어 모델, 2개의 비전-언어 모델을 평가하였고, 실험 결과에서도 최첨단 성능을 달성하여 기존 모델들과 비교했을 때 우수한 결과를 보였습니다.



### ROK Defense M&S in the Age of Hyperscale AI: Concepts, Challenges, and Future Directions (https://arxiv.org/abs/2410.00367)
- **What's New**: 이 논문은 하이퍼스케일 AI (hyperscale AI)를 국가 방어 모델링 및 시뮬레이션 (M&S)에 통합하는 방법에 대해 탐구하고 있으며, 이 기술들이 전략적 및 운영 능력을 향상시키기 위해 얼마나 중요한지를 설명하고 있습니다.

- **Technical Details**: M&S는 군사 조직이 실제 시스템, 프로세스 및 시나리오의 가상 재현을 생성할 수 있도록 하는 중요한 도구입니다. 논문은 미국과 한국의 AI 채택 전략의 차이를 분석하고, 한국의 방어 M&S에 AI를 통합하는 데 필요로 하는 기술적, 운영적 및 정책적 장벽을 확인합니다. 또한, 하이퍼스케일 AI는 대규모의 컴퓨터 자원과 방대한 데이터 세트를 활용하여 복잡한 문제를 해결하는 데 사용됩니다.

- **Performance Highlights**: 하이퍼스케일 AI를 방어 M&S에 적용하면 인적 개입을 줄이면서도 다양한 시나리오를 시뮬레이션하고 전략적 결정 내리기가 가능해집니다. 각국의 AI 기술 투자 확대는 글로벌 방어 환경에서 경쟁 우위를 유지하는 데 필수적이며, 한국은 이러한 기술을 통해 방어 능력을 강화하고 현대 전쟁의 새로운 위협에 대비할 수 있을 것입니다.



### AARK: An Open Toolkit for Autonomous Racing Research (https://arxiv.org/abs/2410.00358)
Comments:
          7 pages, 5 figures

- **What's New**: AARK(Australian Autonomous Racing Kit)는 자율 레이싱 연구의 접근성을 높이기 위해 설계된 오픈 소스 툴킷으로, Assetto Corsa와 연결되어 자율 제어 시스템 개발 및 검증을 지원합니다.

- **Technical Details**: AARK는 세 가지 패키지로 구성되어 있습니다: ACI(AC Interface), ACDG(AC Data Generation), ACMPC(AC Model Predictive Controller). ACI는 차량 상태 모니터링 및 제어를 위한 인터페이스를 제공하며, ACDG는 머신 러닝 모델 교육을 위한 심층, 일반 및 의미론적 데이터를 생성합니다. ACMPC는 모듈형 완전 자율 제어 솔루션을 제공합니다.

- **Performance Highlights**: AARK는 쉽고 직관적인 사용자 경험을 통해 자율 레이싱 연구의 유니파이 및 재현성을 높이며, 물리적 한계에서 차량을 안전하게 운영하는 방법에 대한 통찰력을 제공합니다.



### Integrating Text-to-Music Models with Language Models: Composing Long Structured Music Pieces (https://arxiv.org/abs/2410.00344)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2404.11976

- **What's New**: 최근의 음악 생성 방법들은 Transformer 기반으로 최대 1분의 컨텍스트(window)만 가지며, 이로 인해 생성된 음악은 컨텍스트 창을 넘어 구조적이지 않은 특징이 있습니다. 본 논문에서는 Large Language Model(LLM)과 Text-to-Music 모델을 통합하여 매우 구조적이고 통일감 있는 2.5분 길이의 음악을 생성하는 방법을 제안합니다.

- **Technical Details**: 이 연구는 MusicGen과 ChatGPT를 결합하여 음악의 형태를 생성하는 새로운 방법을 모색합니다. MusicGen 모델은 오디오의 요약된 표현(latent space)에서 훈련되어 있으며, 텍스트 프롬프트를 인코딩하여 음악을 생성합니다. 본 연구에서는 ChatGPT를 사용하여 MusicGen의 프롬프트를 생성하고, 두 모델 간의 적합성을 개선하기 위한 방법으로 In-context learning 방식을 적용합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 잘 조직된 형태의 음악을 생성하며, 기존 모델들이 해결하지 못했던 대규모의 음악 구조를 학습할 수 있는 가능성을 보여줍니다. 특히, LLM의 지식 기반을 활용하여 음악의 구조를 설계하고, 음악 생성 과정에서 더 높은 해석 가능성을 제공합니다.



### Contrastive Representation Learning for Predicting Solar Flares from Extremely Imbalanced Multivariate Time Series Data (https://arxiv.org/abs/2410.00312)
Comments:
          This work has been accepted at ICMLA 2024 on September 7, 2024, as a short paper for poster presentation

- **What's New**: 본 논문에서는 태양의 주요 플레어를 예측하기 위한 새로운 머신 러닝 접근 방식인 CONTREX를 소개합니다. 이 접근법은 다변량 시계열 데이터의 특성을 포착하고, 유사한 클래스 간의 거리를 최대화하여 극단적인 클래스 불균형 문제를 해결합니다.

- **Technical Details**: CONTREX는 다변량 시계열 (Multivariate Time Series) 데이터를 다루기 위한 대조적 표현 학습 (Contrastive Representation Learning) 접근법입니다. 이 방법은 catch22 기능 추출을 사용하고, 긍정 및 부정 클래스 특징 벡터에서 두 개의 극단 포인트를 유도하여 최적의 분리 능력을 제공합니다. 또한, 맞춤형 대조적 재구성 손실 (Contrastive Reconstruction Loss)을 통해 시계열 데이터의 임베딩을 최적화합니다.

- **Performance Highlights**: CONTREX는 SWAN-SF 다변량 시계열 벤치마크 데이터셋에서 기존 방법 대비 우수한 태양 플레어 예측 성능을 보여주었습니다.



### Ask, Pose, Unite: Scaling Data Acquisition for Close Interactions with Vision Language Models (https://arxiv.org/abs/2410.00309)
Comments:
          Project webpage: this https URL

- **What's New**: 이 논문에서는 Human Mesh Estimation (HME) 분야에서의 데이터 부족 문제를 해결하기 위해, Large Vision Language Models (LVLMs)을 활용하여 참조 데이터와 가상의 참조 메시를 생성하는 새로운 방법을 도입하였습니다.

- **Technical Details**: 제안된 방법인 Ask Pose Unite (APU) 데이터세트는 6,200개 이상의 인간 메시 쌍을 포함하여 다양한 유형의 상호작용을 다룹니다. 이를 통해 HME 모델을 학습시키고, 테스트 시 최적화를 통해 메시 추정의 정확도를 향상시킵니다. 또한, 이 방법은 수동 주석의 필요성을 줄이며, 사실적인 인물 간의 상호작용을 반영하는 포괄적인 데이터세트를 제공합니다.

- **Performance Highlights**: 새로 생성된 APU 데이터세트를 이용하여 diffusion 기반의 contact prior를 훈련시키는 실험을 진행했으며, 이를 통해 이전에 보지 못한 상호작용에 대한 메쉬 추정 정확도가 유의미하게 개선되었음을 보여주었습니다.



### GARCH-Informed Neural Networks for Volatility Prediction in Financial Markets (https://arxiv.org/abs/2410.00288)
- **What's New**: 금융 시장의 변동성을 예측하기 위한 새로운 하이브리드 모델인 GARCH-Informed Neural Network (GINN)이 소개되었습니다. 이 모델은 기존의 GARCH 모델과 LSTM 딥러닝 네트워크의 장점을 결합하여 시장 변동성을 보다 정확하게 캡처하고 예측합니다.

- **Technical Details**: GINN 모델은 GARCH의 에러 항을 신경망의 손실 함수에 포함하여 오버피팅(overfitting)을 방지합니다. GARCH 모델의 이론적 기초를 활용하여 두 가지 다른 접근 방식을 통합함으로써 금융 데이터의 비선형적 특성을 포괄적으로 모델링합니다.

- **Performance Highlights**: GINN 모델은 Coefficient of Determination ($R^2$), Mean Squared Error (MSE), Mean Absolute Error (MAE) 등의 지표에서 기존의 GARCH 및 LSTM 모델보다 뛰어난 예측 성능을 보여주었습니다. 7개의 대표적인 주식 시장 지수에서 테스트한 결과, GINN은 우수한 성과를 기록하였습니다.



### GalaxiesML: a dataset of galaxy images, photometry, redshifts, and structural parameters for machine learning (https://arxiv.org/abs/2410.00271)
Comments:
          19 pages, 6 figures, data available at this https URL, example code of usage at this https URL

- **What's New**: 이번 논문에서는 머신 러닝 응용을 위한 새로운 데이터셋 GalaxiesML을 공개합니다. 이 데이터셋은 286,401개의 은하 이미지와 HSC(하이퍼 수프림 캠) 설문조사에서 발견된 스펙트로스코픽 적색편이(redshift)를 포함하여 은하의 구조적 속성을 담고 있습니다.

- **Technical Details**: GalaxiesML 데이터셋은 g, r, i, z, y의 5개 필터에서 수집된 이미지를 포함하며, 스펙트로스코픽 적색편이를 기준으로 정리되었습니다. 이 데이터셋은 노이즈 비율(signal-to-noise ratio)의 다양한 범위를 가지고 있으며, 데이터의 중복, 아웃라이어 제거, 그리고 샘플 선택과 같은 문제를 해결하기 위해 정교한 프로세스가 적용되었습니다. 이 데이터셋은 머신 러닝 모델을 위한 최적화된 형식으로 제공됩니다.

- **Performance Highlights**: 이미지를 이용한 적색편이 추정 시 우수한 성능을 보여줍니다. 0.1에서 1.25의 적색편이를 가진 은하에 대해 이미지만 사용하는 경우, 포토메트리(photometry)를 사용할 때보다 적색편이 추정의 편향(bias)이 10배 낮았습니다. 이 데이터셋을 통해 차세대 은하 조사 데이터를 효과적으로 활용하는 방법을 제시할 수 있습니다.



### Real-time Diverse Motion In-betweening with Space-time Contro (https://arxiv.org/abs/2410.00270)
Comments:
          Presented at The 16th ACM SIGGRAPH Conference on Motion, Interaction, and Games (MIG '24)

- **What's New**: 이 논문에서는 kinematic 캐릭터의 다양한 중간 동작을 생성하기 위한 데이터 기반 프레임워크인 DC-MoE(Dynamic Conditional Mixture-of-Experts)를 제안합니다. 이 접근법은 동적인 조건과 명시적인 모션 제어를 중간 동작 전환 과정에 주입함으로써, 사용자에게 지속시간, 경로, 스타일 등 추가 조건을 부여할 수 있게 합니다.

- **Technical Details**: DC-MoE는 현재 프레임, 목표 프레임 및 조건을 기반으로 다음 시간 단계에서의 자세 및 시간적 특성을 예측하는 autoregressive 모델입니다. 이 모델을 통해 스타일과 지속시간 같은 모션 속성을 제어하여 다양한 중간 동작을 생성할 수 있습니다. 또한, 캐릭터의 루트 궤적에 대한 명시적 제어를 주입하여 목표에 도달하도록 궤적을 안내합니다.

- **Performance Highlights**: 이 프레임워크는 100STYLE 데이터셋과 LaFAN1 데이터셋을 사용하여 여러 메트릭을 기반으로 동작 품질에 대한 효과성을 평가하였으며, 다양한 스타일과 시간이 변할 때 중간 동작 생성이 가능함을 입증하였습니다.



### Class-Agnostic Visio-Temporal Scene Sketch Semantic Segmentation (https://arxiv.org/abs/2410.00266)
- **What's New**: 이번 연구에서는 Class-Agnostic Visio-Temporal Network (CAVT)를 제안하며, 이는 장면 스케치의 의미론적 분할을 위한 새로운 접근 방식입니다. CAVT는 기존의 스케치 세분화 방법들이 스케치를 비트맵 이미지로 처리하는 것과는 달리, 개별 객체를 인식하고 개별적인 유니크한 인스턴스를 수행합니다.

- **Technical Details**: CAVT는 클래스에 구애 받지 않는 객체 탐지 모듈을 이용해 장면 내 개체들을 감지하고, 후처리 모듈을 통해 인스턴스의 오브젝트 스트로크를 그룹화합니다. 이 방법은 장면 스케치 내 인스턴스와 스트로크 단위로 분할을 수행하는 최초의 연구이자, RGB 컬러링 기법을 활용하여 스트로크의 시간 정보를 보존합니다.

- **Performance Highlights**: FrISS 데이터셋에서의 실험 결과, 제안한 방법이 현재 상태의 선도적 장면 스케치 세분화 모델보다 우수한 성능을 보였습니다. 이는 1,000개의 장면 스케치를 포함하여 403개의 객체 클래스를 커버하고, 조밀한 주석이 있는 데이터셋으로, 미래의 스트로크 기반 연구를 촉진할 수 있습니다.



### DoPAMine: Domain-specific Pre-training Adaptation from seed-guided data Mining (https://arxiv.org/abs/2410.00260)
- **What's New**: 이번 연구에서 제안하는 DoPAMine 프레임워크는 대규모 데이터 코퍼스에서 도메인 특화 훈련 데이터를 자동으로 채굴하여 대형 언어 모델(LLM)의 도메인 적응을 지원합니다. 이 방법은 일반적인 웹 데이터를 기반으로 한 이제까지의 접근 방식과는 달리, 도메인에 맞춘 데이터 생성 및 실제 세계 데이터를 채굴하여 현실적이고 신뢰할 수 있는 결과를 제공합니다.

- **Technical Details**: DoPAMine 프레임워크는 LLM의 매개변수적 지식을 활용하여 특정 도메인에 최적화된 다양한 시드(seed) 데이터를 생성하고, 이를 통해 Common Crawl과 같은 대규모 데이터 코퍼스에서 관련 데이터 문서를 채굴하는 과정을 포함합니다. 핵심 메커니즘은 LLM을 활용하여 도메인 특화된 시드 데이터를 생성하고, 생성된 시드 데이터를 기반으로 의미적으로 유사한 문서를 검색하는 것입니다.

- **Performance Highlights**: DoPAMine를 사용하여 지속적 재학습(CPT) 환경에서 두 개의 도메인 특화 7B 파라미터 LLM을 의료 및 금융 분야에서 훈련한 결과, 평균 4.9% 및 5.1%의 성능 향상을 보였으며, 이는 zero-shot 및 5-shot 설정에서 MMLU, MedQA, MedMCQA 및 PubMedQA 데이터셋에 대한 것입니다. 금융 과제에서는 평균 2.9% 및 6.7%의 향상을 나타내었으며, FiQA-SA, FPB 및 Headlines 데이터셋과 비교하여 성능이 크게 높아졌습니다.



### Demonstrating the Continual Learning Capabilities and Practical Application of Discrete-Time Active Inferenc (https://arxiv.org/abs/2410.00240)
Comments:
          13 pages, 3 figures

- **What's New**: 이 논문에서는 Active Inference(능동적 추론)의 원리를 기반으로 하는 지속 학습 프레임워크를 제안합니다. 이 프레임워크는 생물학적 또는 인공지능 에이전트가 불확실하고 동적인 환경에서 어떻게 자발적으로 학습하고 행동하는지를 수학적으로 모델화합니다.

- **Technical Details**: Active Inference는 Bayesian inference와 free energy minimization의 결합으로 표현됩니다. 두 가지 주요 함수인 Variational Free Energy (VFE)와 Expected Free Energy (EFE)를 통해 에이전트가 환경에서 받은 센서리 데이터를 바탕으로 행동을 선택하고 예측 모델을 갱신하는 방식을 설명합니다. VFE는 에이전트의 내부 모델과 실질적인 센서 데이터 간의 유사성을 측정하며, EFE는 목표 지향적 행동과 탐색 기능을 결합하여 에이전트의 정책 결정을 지원합니다.

- **Performance Highlights**: 실험 결과, 제안된 에이전트는 지속적으로 변화하는 환경에서도 모델을 효과적으로 재학습 및 개선할 수 있는 능력을 보여주었고, 금융 및 의료와 같은 복잡한 도메인에서의 적용 가능성을 입증했습니다.



### Modulation and Coding for NOMA and RSMA (https://arxiv.org/abs/2410.00239)
Comments:
          Invited paper; to appear in the Proceedings of the IEEE

- **What's New**: 본 논문은 비정형 다중 접속 기법인 NOMA(Non-Orthogonal Multiple Access)의 원리와 기존 방법론을 리뷰하며, 사용자 간 간섭을 해결하기 위해 비동기 전송(asynchronous transmission) 및 간섭 인식 변조(interference-aware modulation) 기술을 도입하여 후속 간섭 제거 없이 디코딩할 수 있는 방안을 제시합니다. 또한, 6G와 같은 차세대 통신 네트워크의 요구사항을 충족하기 위한 응용 프로그램 및 서비스에 대한 필요성을 강조합니다.

- **Technical Details**: NOMA는 여러 사용자가 시간, 주파수, 공간을 동시에 공유할 수 있게 하여 연결성을 증가시키고 스펙트럼 효율을 향상시킵니다. 그러나 NOMA는 사용자 간 간섭(inter-user interference)을 제거하는 데 어려움을 겪고 있습니다. 이 논문은 간섭 감지 및 통신구축(interference-aware constellation design) 기술을 도입하여 이러한 과제를 해결하는 방안을 논의합니다. 추가로, 코드 도메인 NOMA와 trellis-coded NOMA 같은 새로운 기술을 소개합니다.

- **Performance Highlights**: 이 연구에서는 비트 오류율(BER) 최소화를 통해 사용자 처리량을 향상시키고, 이를 기반으로 딥 오토인코더(deep autoencoders)를 활용하여 신뢰할 수 있는 최종 사용자 통신을 제안합니다. RSMA가 다중 사용자 시스템에서의 간섭 관리에 대한 유망한 기법으로 주목받고 있으며, 미래의 연구 방향을 탐색하여 NOMA를 개념에서 작동 기술로 발전시키기 위한 중요성을 강조합니다.



### Helpful DoggyBot: Open-World Object Fetching using Legged Robots and Vision-Language Models (https://arxiv.org/abs/2410.00231)
Comments:
          Project website: this https URL

- **What's New**: 이번 연구에서는 인도 환경에서 도움이 되는 이동 조작 기술을 위한 새로운 시스템인 Helpful DoggyBot을 제안합니다. 이 시스템은 전면에 장착된 그리퍼와 저수준 제어기를 사용하여, 다양한 내부 환경에서 유용한 작업을 수행할 수 있도록 설계되었습니다.

- **Technical Details**: Helpful DoggyBot은 1-DoF 그리퍼를 사용하여 일상적인 객체를 집어올리며, egocentric depth(자기중심 깊이)와 proprioception(신체위치 감각)을 활용한 저수준 제어기를 시뮬레이션에서 훈련하여 기동성을 높였습니다. Vision-Language Models (VLMs)를 활용하여 객체 식별 및 명령 생성을 수행하며, 이는 목표 객체를 따라가고 추적하는 데 도움을 줍니다.

- **Performance Highlights**: 이 시스템은 두 개의 미지의 환경에서 zero-shot generalization을 수행하여, 예를 들어 사용자의 명령에 따라 무작위로 놓인 인형을 찾아오는 작업을 60%의 성공률로 완료했습니다. 이는 실제 세계 데이터 수집이나 훈련 없이도 가능하여, 다양한 홈 환경에 적응할 수 있는 도움이 되는 4족 보행 로봇의 가능성을 보여줍니다.



### Stochastic Inverse Problem: stability, regularization and Wasserstein gradient flow (https://arxiv.org/abs/2410.00229)
- **What's New**: 이번 논문에서는 물리적 또는 생물학적 과학에서의 확률적 역문제(stochastic inverse problem)를 다루며, 알려지지 않은 확률 분포를 복원하는 과정과 그에 관련된 다양한 최적화 접근법들을 탐구합니다. 특히, 확률 공간(probability space)에서의 작업으로 인해 발생하는 복잡성을 다루어, 결정론적 역문제(deterministic inverse problems)와의 유사성과 차이점을 강조합니다.

- **Technical Details**: 본 연구에서는 역문제를 세 가지 주요 측면으로 나누어 설명합니다: 직접 역전(direct inversion), 정규화가 포함된 변분 공식(variational formulation with regularization), 그리고 Gradient Flow를 통한 최적화(optimization via gradient flows). 또한, 손실 함수(loss function) 설계 및 최적화 과정에서의 메트릭 선택이 최적화기의 안정성 및 성질에 미치는 영향을 탐구합니다. 이 과정에서 수치 해석과 기하학적 성질을 통해 측정 수송 이론(measure transport theory)의 도구를 적용합니다.

- **Performance Highlights**: 제안된 접근법은 다양한 실제 문제에 적용 가능성이 높으며, 예를 들어 기상 예측, 플라즈마 시뮬레이션, 실험 설계, 광 통신 등과 같은 분야에서 확률 분포를 복원하는 데 유용하게 사용될 수 있습니다. 또한, 이 연구는 기존의 유클리드 공간(Euclidean space)에서 확률 분포 공간으로의 문제 전환을 통해 새로운 해결책을 제시하고 있습니다.



### End-to-end Piano Performance-MIDI to Score Conversion with Transformers (https://arxiv.org/abs/2410.00210)
Comments:
          6 pages, to appear at ISMIR 2024

- **What's New**: 이 논문에서는 실제 피아노 연주로부터 상세한 음악 악보를 자동 생성하는 최적의 방법을 제안합니다. 기존의 note-wise classification 방식을 떠나 end-to-end deep learning 접근 방식을 사용하여 성능을 개선했습니다.

- **Technical Details**: 제안된 시스템은 P-MIDI 파일을 MusicXML 악보로 변환하는 전방위적(seq2seq) 변환 작업으로 구성됩니다. 아키텍처는 현대적인 transformer 기반이며, 복합 토큰(token) 표현을 사용하여 기호 음악 데이터의 직렬화를 간소화합니다.

- **Performance Highlights**: MUSTER와 같은 전사 매트릭스를 사용한 평가에서 기존 딥러닝 모델 및 복잡한 HMM 기반 파이프라인에 비해 성능이 상당히 향상되었습니다. 특히, 트릴 마크(trill marks) 및 줄 방향(stem direction)같은 세부 사항을 예측할 수 있는 첫 번째 방법으로 주목받고 있습니다.



### Volumetric Conditional Score-based Residual Diffusion Model for PET/MR Denoising (https://arxiv.org/abs/2410.00184)
Comments:
          Accepted to MICCAI 2024

- **What's New**: 본 논문에서는 Conditional Score-based Residual Diffusion (CSRD) 모델을 제안하며, 3D 볼륨 PET 영상을 효율적으로 처리하기 위한 3D 패치 기반 훈련 전략을 포함하여 PET 이미징의 고유한 특성을 반영하고, 전통적인 방법보다 성능을 크게 향상시킵니다.

- **Technical Details**: CSRD 모델은 점진적인 score function 개선과 3D 훈련 전략을 통해 계산 비용 및 메모리 요구사항을 최적화합니다. 이 모델은 PET 및 MRI 스캔에서의 볼륨 데이터 통합을 통해 공간 일관성과 해부학적 세부사항을 유지합니다. 또한, 이미지의 세부 정보를 보존하면서 더 나은 제거 성능을 보여줍니다.

- **Performance Highlights**: CSRD 모델은 질적으로 및 양적으로 기존의 최첨단 방법을 능가하며, 3D PET 이미지의 볼륨 감소 작업을 3분 이내에 수행할 수 있습니다. 이 모델은 속도와 성능 모두에서 상당한 개선을 나타냅니다.



### Evaluating the fairness of task-adaptive pretraining on unlabeled test data before few-shot text classification (https://arxiv.org/abs/2410.00179)
Comments:
          To appear in the GenBench Workshop at EMNLP 2024

- **What's New**: 이 논문은 최신 NLP 기술을 평가하는 데 있어 few-shot learning 벤치마크의 편향을 분석합니다. 특히, 연구자들이 테스트 세트의 비공식 텍스트를 이용하여 사전 학습(pretraining)을 수행하면서 발생할 수 있는 성능 과대평가(overoptimism)의 여부를 실험을 통해 조사했습니다.

- **Technical Details**: 25개의 분류 작업과 3개의 언어 모델(BERT, GPT-2, Mistral 7B)을 사용한 controlled few-shot 및 zero-shot 실험을 통해, unlabeled test set 텍스트로 사전 학습하는 것이 성능에 미치는 영향을 연구했습니다. 하이퍼파라미터 튜닝과 같은 기존 기법들과 비교해, unlabeled text의 출처가 성능에 미치는 영향을 명확히 파악했습니다.

- **Performance Highlights**: 사전 학습의 이점이 검증되었으며, unlabeled test set 텍스트를 사용하는 것과 독립적으로 추출된 텍스트를 사용하는 것 간의 성능 차이에 대한 근거를 제시했습니다. 결과적으로, few-shot 학습 벤치마크에 여러 교육 폴드(training folds)를 포함하는 것이 중요하다는 것을 강조했습니다.



### Beyond Single Concept Vector: Modeling Concept Subspace in LLMs with Gaussian Distribution (https://arxiv.org/abs/2410.00153)
Comments:
          28 pages, 9 figures

- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)의 내부 지식 표현을 이해하기 위해 Gaussian Concept Subspace (GCS)를 도입했습니다. 이는 기존의 단일 벡터 대신, 특정 개념을 나타내는 서브스페이스를 근사하는 접근 방식입니다.

- **Technical Details**: GCS는 각각의 개념에 대해 단일 벡터가 아닌 관찰 벡터(observed vectors) 집합을 학습하여, 가우시안 분포를 사용하여 개념 서브스페이스를 추정합니다. 이로 인해 더 미세한 개념 표현이 가능해지고, 벡터의 밀도가 개념에 대한 관련성의 정도를 나타냅니다.

- **Performance Highlights**: 실험 결과 GCS로부터 샘플링된 벡터는 LLM에서 학습된 개념을 효과적으로 설명하고, 다운스트림 개입(intervention) 작업에서 더 바람직한 결과를 생성하는데 기여함을 보여주었습니다. 특히, 자연어 생성 과제에서 텍스트 유창성을 유지하면서 감정 조정 성능을 균형 있게 달성하는 데 성공했습니다.



### Multimodal Alignment of Histopathological Images Using Cell Segmentation and Point Set Matching for Integrative Cancer Analysis (https://arxiv.org/abs/2410.00152)
Comments:
          initial version

- **What's New**: 본 논문에서는 암 연구 및 임상 실습에서 중요한 조직 병리학적 이미지의 멀티모달(모델에 따라 다른 유형의 데이터) 정렬을 위한 새로운 프레임워크를 제시합니다.

- **Technical Details**: 이 프레임워크는 세포 세분화 결과를 활용하여, 세포를 점 집합(Point Sets)으로 처리하고 Coherent Point Drift (CPD) 기법을 사용하여 초기 정렬을 수행한 다음, Graph Matching (GM) 기법으로 정렬을 정교화합니다.

- **Performance Highlights**: 난소암 조직 미세 배열(TMA)을 기반으로 평가한 결과, 높은 정렬 정확도를 달성하여 다중 모달에서 세포 수준의 특징 통합을 가능하게 하였으며, MxIF 데이터를 통해 가상 H&E 이미지를 생성하여 임상 해석을 향상시켰습니다.



### What If We Had Used a Different App? Reliable Counterfactual KPI Analysis in Wireless Systems (https://arxiv.org/abs/2410.00150)
Comments:
          This paper has been submitted to a journal

- **What's New**: 본 논문은 Open Radio Access Network (O-RAN)와 같은 현대 무선 네트워크 아키텍처에서 사용되는 앱 기반의 성능 테스트 문제를 다룹니다. 다른 앱이 선택되었다면 어떤 성능 지표(KPIs)가 나왔을지를 추정하는 'what-if' 문제에 접근함으로써, 네트워크 운영 최적화에 기여합니다.

- **Technical Details**: 논문에서는 무선 시스템을 위한 conformal-prediction 기반의 카운터팩추얼 분석 방법을 제안합니다. 이는 로그된 데이터를 이용하여 현재 앱과 다른 앱에서 성취할 수 있었던 KPIs를 예측하고, 사용자 정의 확률 내에서 실제 KPIs가 포함된 'error bars'를 제공하여 신뢰성을 보장합니다.

- **Performance Highlights**: 실험 결과는 중간 접근 제어 층 및 물리적 제어 층의 앱에 대해 제안된 방법의 유용성과 신뢰성을 보여줍니다. 특히, CCKE(Counterfactual Conformal KPI Estimation) 기법은 무선 시스템에서 KPIs의 신뢰할 수 있는 what-if 분석을 가능하게 합니다.



### Are Large Language Models In-Context Personalized Summarizers? Get an iCOPERNICUS Test Done! (https://arxiv.org/abs/2410.00149)
- **What's New**: 본 논문은 In-Context Learning (ICL)을 기반으로 한 요약에서 사용자의 선호 이력을 반영하는 In-Context Personalization Learning (ICPL)의 중요성을 강조하며, iCOPERNICUS라는 새로운 프레임워크를 제안합니다.

- **Technical Details**: iCOPERNICUS 프레임워크는 세 가지 요소를 통해 LLMS의 ICPL을 평가합니다: (i) 예제 요약의 효과, (ii) 사용자의 읽기 이력 추가 시 ICPL 향상 여부, (iii) 사용자 프로필 대비 정보를 통한 ICPL 유도 여부. EGISES를 개인화 측정 지표로 활용하여 각 요소의 영향을 비교합니다.

- **Performance Highlights**: 17개의 최신 LLM 모델을 평가한 결과, 15개의 모델에서 ICPL 성능이 저하되는 것을 관찰했습니다. 이는 iCOPERNICUS의 프로빙 방법을 통해 드러난 사실로, LLM이 진정한 ICPL을 적용하지 않고 있음을 시사합니다.



### Constraint-Aware Refinement for Safety Verification of Neural Feedback Loops (https://arxiv.org/abs/2410.00145)
Comments:
          6 pages, 10 figures, submitted to L-CSS/ACC

- **What's New**: 본 논문은 Neural Network (NN)을 이용한 자율 시스템의 제어 파이프라인에서의 안전성 보장을 위한 새로운 접근법인 CARV(Constraint-Aware Refinement for Verification)를 제안합니다. CARV는 RSOA(Reachable Set Over-Approximations)의 보수성을 줄여주는 효율적인 방법으로, 신속하고 효율적인 안전성 검증을 가능하게 합니다.

- **Technical Details**: CARV 알고리즘은 NFL(Neural Feedback Loops)의 안전 제약 조건을 명시적으로 활용하여 RSOA를 정제하는 방식으로 작동합니다. RSOA는 일반적으로 NP-hard 문제인 정확한 도달 가능 집합의 근사치를 제공하는데, 이 알고리즘은 보수적인 RSOA로 인한 문제를 해결합니다. CARV는 하이브리드-상징적 접근법을 사용하여 안전성 검증 과정을 최적화합니다.

- **Performance Highlights**: CARV는 기존의 접근 방식들이 실패하는 경우 또는 60배 느리게 동작하고 40배 더 많은 메모리를 요구하는 문제에서도 NFL의 안전성을 검증할 수 있음을 실험을 통해 입증했습니다.



### Cartesian Genetic Programming Approach for Designing Convolutional Neural Networks (https://arxiv.org/abs/2410.00129)
- **What's New**: 이번 연구에서는 Convolutional Neural Networks (CNNs)의 설계 및 최적화를 위한 방식으로 Cartesian Genetic Programming (CGP)을 활용한 Neural Architecture Search (NAS)에 대해 다룹니다. 기존의 수작업으로 설계된 아키텍처 대신 CGP를 통해 자동으로 아키텍처를 생성하여 더 나은 성능을 기대할 수 있습니다.

- **Technical Details**: 저자들은 mutation(변이)만을 적용하는 순수한 CGP 접근법을 활용하여 CNN 아키텍처를 생성합니다. 진화는 유전자의 특정 집합을 생성하고 이를 기반으로 변이를 통해 새로운 세대를 만들어내며, 성능 평가를 통해 최적의 아키텍처를 탐색하는 방식으로 진행됩니다. 각 세대에서 최고의 유전자형을 기록하고, fitness function(적합도 함수)을 통해 성능을 평가합니다.

- **Performance Highlights**: 실험 결과, 62.5K의 중간 예산에서 가장 우수한 결과를 얻었으며, 높은 예산인 125K에서 알고리즘 성능이 저하되는 경향을 보였습니다. 샘플 데이터셋인 MNIST와 Fashion-MNIST를 기반으로, mutuation rates(변이 비율)에 따른 성능 차이를 확인하였습니다. 향후 연구에서는 더 큰 네트워크 집단 및 다양한 변이 비율을 탐색할 필요성이 강조됩니다.



### An Overview of the Burer-Monteiro Method for Certifiable Robot Perception (https://arxiv.org/abs/2410.00117)
Comments:
          Accepted to 2024 Robotics: Science and Systems (RSS) Safe Autonomy Workshop

- **What's New**: 이 논문은 Burer-Monteiro 방법(BM)을 소개하며, 이 기술이 로봇 인식 문제를 해결하는 데 어떻게 적용되는지를 다룹니다. BM은 신뢰할 수 있는 최적 솔루션을 신속하게 도출할 수 있는 방법입니다.

- **Technical Details**: Burer-Monteiro 방법은 주로 semidefinite programming(SDP) 완화로 나타나는 비볼록 인식 문제를 해결하는 데 사용됩니다. 이 방법은 일반적인 SDP 프로그램의 저차원 낮은 순위(factorization) 구조를 활용하여 최적화의 계산 비용을 크게 줄입니다. 특히 LICQ(linear independence constraint qualification)가 알고리즘의 신뢰성을 향상시키는 데 중요함을 강조합니다.

- **Performance Highlights**: 이 논문의 목적은 BM의 내용을 통일된 형태로 정리하고, 실용적인 고려 사항을 추가하여 BM 적용 시 진입 장벽을 낮추는 것입니다. 이는 로봇 인식의 안정성과 신뢰성을 확보하는 데 기여할 수 있습니다.



### Graph Residual Noise Learner Network for Brain Connectivity Graph Prediction (https://arxiv.org/abs/2410.00082)
Comments:
          10 pages, 3 figures, 6th Workshop on GRaphs in biomedicAl Image anaLysis

- **What's New**: 본 연구에서는 신경 장애 진단을 개선하기 위한 새로운 그래프 기반의 뇌 그래프 예측 모델인 Graph Residual Noise Learner Network (Grenol-Net)를 제안합니다. 기존의 확산 모델을 활용하되, 그래프의 토폴로지 대칭성을 유지하기 위한 접근 방식을 채택하여, 신경 과학 연구 커뮤니티에서의 협력적 연구 노력을 촉진하고자 합니다.

- **Technical Details**: Grenol-Net은 두 개의 복잡하게 설계된 학습 블록으로 구성됩니다. 첫 번째 블록은 그래프 컨볼루션 블록을 포함하여 소스 그래프 내 연결의 복잡한 상호작용을 드러내며, 서로 다른 노드 간의 메시지 전달 기능을 통해 각 ROI의 독특한 임베딩을 학습합니다. 두 번째 블록은 배치 정규화를 도입하여 타겟 그래프의 분포를 학습하고, 잔여 연결을 통해 타겟에 적용된 노이즈를 분리 및 복구합니다.

- **Performance Highlights**: Grenol-Net은 기존의 신경망 모델에 비해 예측 정확도 및 그래프 연결성 다양성을 유지하면서도 더 나은 성능을 자랑합니다. 예측 과정에서는 그래프의 노드 특성을 정확하게 학습하여, 동일한 파셀레이션 뇌 템플릿에서 파생된 동형(biomorphic) 그래프들 간의 연결성 다양성을 유지하는 데 초점을 맞추었습니다.



### Interactive Speculative Planning: Enhance Agent Efficiency through Co-design of System and User Interfac (https://arxiv.org/abs/2410.00079)
Comments:
          27 pages, 22 figures

- **What's New**: 논문에서는 인간 중심의 효율적인 에이전트 계획 방법을 제안하며, 이를 통해 LLM 기반 에이전트의 계획 지연 문제를 해결하려고 합니다. 새로운 접근 방식인 Interactive Speculative Planning(상호 작용적 추정 계획)을 도입하여 시스템 설계와 사용자-AI 상호작용 간의 효율성을 높이고자 합니다.

- **Technical Details**: Interactive Speculative Planning은 두 개의 에이전트 시스템, 즉 효율적이지만 능력이 제한된 근사 에이전트와 느리지만 강력한 목표 에이전트를 활용합니다. 근사 에이전트는 작업 단계(차례대로)를 생성하며, 동시에 목표 에이전트는 비동기적으로 다음 단계의 출력을 생성합니다. 이 시스템은 유저가 긴 지연 시간 동안 개입할 수 있도록 하여 전체 프로세스를 가속화합니다.

- **Performance Highlights**: 이 시스템은 사용자 개입을 통해 에이전트 계획 과정의 효율성을 높이고 최종 출력의 정확성을 보장합니다. 논문에서는 실험을 통해 Interactive Speculative Planning 방식의 실제 데이터를 사용한 평가 결과를 제시하고 있으며, 기존 LLM 기반 에이전트 시스템의 지연 문제를 효과적으로 해결할 수 있는 가능성을 보이고 있습니다.



### Shuffled Linear Regression via Spectral Matching (https://arxiv.org/abs/2410.00078)
Comments:
          This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 이번 연구에서는 Shuffled Linear Regression (SLR) 문제를 해결하기 위해 spectral matching 방법을 제안합니다. 이 방법은 노이즈가 있는 선형 변환으로부터 숨겨진 신호와 이에 해당하는 측정값의 순열을 동시에 복구하고자 합니다.

- **Technical Details**: 제안된 방법은 measurement와 feature covariance의 spectral 구성 요소를 정렬하여 순열 문제를 효율적으로 해결합니다. 이론적 분석에 따르면, 주어진 샘플 수가 충분할 경우 shuffled LS 및 shuffled LASSO 설정에서 정확한 추정을 달성할 수 있습니다.

- **Performance Highlights**: 합성 데이터셋과 실제 이미지 등록 시나리오에서 실험한 결과, 제안된 방법이 기존 알고리즘보다 추정 정확도 및 등록 성능에서 우수함을 보였습니다.



### Optimizing Treatment Allocation in the Presence of Interferenc (https://arxiv.org/abs/2410.00075)
- **What's New**: 이번 연구에서는 네트워크 내 간섭이 있는 환경에서의 최적 치료 할당 문제에 대한 새로운 접근법인 OTAPI(Optimizing Treatment Allocation in the Presence of Interference)를 제안합니다. 이 방법은 기존의 Influence Maximization (IM)과 Uplift Modeling (UM) 기법의 한계를 극복하고 이를 통합하여 치료 효과를 극대화하는 것을 목표로 합니다.

- **Technical Details**: OTAPI는 두 단계로 구성됩니다. 첫 번째 단계에서는 관계성 인과 추정기(relational causal estimator)를 훈련시켜 네트워크 내의 치료 효과를 예측합니다. 두 번째 단계에서는 이 추정기를 활용하여 전통적인 IM 알고리즘과 통합하여 최적의 치료 할당을 찾습니다. 이는 조합(combinatorial)적 최적화 문제로, 기존의 IM 기법 적용 시 간섭 효과를 고려하여 최적의 노드를 선택합니다.

- **Performance Highlights**: OTAPI는 기존의 IM 및 UM 접근 방법보다 더 높은 치료 효과를 달성하며, 합성(synthetic) 데이터 및 반합성(semi-synthetic) 데이터 모두에서 그 성능을 입증했습니다. 이는 치료 할당의 총 효과(total effect)를 기준으로 상대적으로 우수한 결과를 보여줍니다.



### An interdisciplinary exploration of trade-offs between energy, privacy and accuracy aspects of data (https://arxiv.org/abs/2410.00069)
Comments:
          Workshop paper for PLSC Europe 2024 (this https URL)

- **What's New**: 이 논문에서는 데이터 처리의 에너지 소비와 개인 정보 보호를 동시에 고려하는 새로운 방법을 제시합니다. 특히, privacy-enhancing 기술이 데이터 유틸리티 및 에너지 소비에 미치는 영향을 측정하는 방법을 탐색합니다.

- **Technical Details**: 실험적 설정을 통해 환경, 개인 정보 보호, 정확성 간의 trade-offs를 발견하였습니다. k-anonymity 및 synthetic data와 같은 익명화(Anonymisation) 기술이 머신러닝의 에너지 소비 및 정확성에 미치는 영향을 분석합니다.

- **Performance Highlights**: 이 연구를 통해 사용자는 에너지 소비, 개인 정보 보호, 정확성 간의 trade-off를 최적화할 수 있는 방법을 제시하여, 결정을 내리기 위한 데이터를 제공합니다.



### Denoising Variational Autoencoder as a Feature Reduction Pipeline for the diagnosis of Autism based on Resting-state fMRI (https://arxiv.org/abs/2410.00068)
- **What's New**: 본 연구는 자폐 스펙트럼 장애(ASD) 진단을 위한 새로운 접근 방식을 제안합니다. 구체적으로, 우리는 resting-state fMRI(rs-fMRI)를 활용하여 ASD의 기능적 연결성을 분석하고, 이를 통해 5개의 잠재적 Gaussian 분포로 특징을 축소하는 파이프라인을 개발했습니다.

- **Technical Details**: 연구는 Ncuts parcellation과 Power atlas를 사용하여 3만 개 이상의 기능적 연결성 데이터를 추출했습니다. 이후 denoising variational autoencoder(DVAE)를 활용하여 데이터의 차원을 축소시키고, 이를 통해 구성된 5개의 Gaussian 분포에서 레이턴트 특징을 추출했습니다. 이 레이턴트 특징은 다중 사이트 데이터 세트에서 전통적인 분류기를 사용하여 ASD를 분류하는 데 활용되었습니다.

- **Performance Highlights**: SVM(support vector machine) 모델을 사용한 예측 정확도는 95% 신뢰 구간에서 [0.63, 0.76]로 나타났으며, DVAE 없이도 0.70의 정확도를 달성했습니다. DVAE의 훈련 시간은 37분으로, 원시 연결 행렬에서 직접 분류기를 훈련하는 것보다 7배 더 빠른 성과를 보였습니다.



### Ranking the Top-K Realizations of Stochastically Known Event Logs (https://arxiv.org/abs/2410.00067)
- **What's New**: 이번 논문에서는 stochastic event logs(확률적 이벤트 로그)의 top-K realizations(상위 K 실현 예)를 효과적으로 계산하는 알고리즘을 구현했습니다. 이를 통해 기존의 top-1 interpretations(상위 1 해석) 대비 top-K rankings(상위 K 순위)의 유용성을 분석합니다.

- **Technical Details**: 제안된 알고리즘은 event independence(이벤트 독립성) 하에서 O(Kn) 시간 복잡도로 top-K 순위를 계산합니다. 여기서 n은 로그의 불확실한 이벤트 수를 의미합니다. 알고리즘은 가능한 로그 실현을 반복적으로 분할하는 일반화된 절차를 기반으로 설계되었습니다.

- **Performance Highlights**: 결과적으로, top-K 순위는 이벤트 로그의 길이 및 이벤트 확률 분포에 따라 그 유용성이 달라지는 것으로 나타났습니다. 이 검토는 uncertainty-aware process mining techniques(불확실성을 고려한 프로세스 마이닝 기법)의 개선 가능성을 충분히 보여주었습니다.



### IDEA: An Inverse Domain Expert Adaptation Based Active DNN IP Protection Method (https://arxiv.org/abs/2410.00059)
- **What's New**: 새로운 논문에서는 모델 소유자가 불법 사용자를 차단하고 침해의 출처를 추적할 수 있는 IDEA(Inverse Domain Expert Adaptation 기반의 능동적 DNN IP 보호 방법)를 제안합니다.

- **Technical Details**: IDEA는 사용자 키를 하이드 스테가노그래픽 기법을 통해 숨겨 사용자의 인증을 수행하며, 진짜 전문가(real expert)와 두 개의 가짜 전문가(fake experts)를 훈련시킵니다. 이 과정에서 서로의 정보를 최소화하는 다중 적응 최적화(multi-adaptive optimization)가 적용됩니다.

- **Performance Highlights**: IDEA는 성능 평가를 위해 5개의 데이터셋과 4개의 DNN 모델에서 광범위한 실험을 진행하였으며, 인증 제어와 범죄자 추적 성공률, 다양한 공격에 대한 강인성에서 효과성을 입증하였습니다.



### Survey of Security and Data Attacks on Machine Unlearning In Financial and E-Commerc (https://arxiv.org/abs/2410.00055)
- **What's New**: 이 논문은 금융 및 전자상거래 애플리케이션에 초점을 맞춘 기계 잊기(Machine Unlearning)에서의 보안 및 데이터 공격의 현황을 조사하고 있으며, 데이터 삭제 요청에 따른 주요 프라이버시 위협과 보안 공격 유형을 논의합니다.

- **Technical Details**: 기계 잊기 과정에서 발생할 수 있는 여러 공격 유형으로는 Membership Inference Attacks(회원 추론 공격), Data Reconstruction Attacks(데이터 재구성 공격), Machine Unlearning Data Poisoning(데이터 오염), Unlearning Request Attacks(삭제 요청 공격), Machine Unlearning Jailbreak Attacks(탈옥 공격)가 있습니다. 이를 통해 데이터 유출 및 모형의 무결성을 해치는 취약점이 드러납니다.

- **Performance Highlights**: 다양한 방어 전략으로는 Differential Privacy(차등 프라이버시), Robust Cryptographic Guarantees(강력한 암호화 보장), Zero-Knowledge Proofs (ZKPs)와 같은 메커니즘이 포함되어 있으며, 높은 리스크의 금융 및 전자상거래 환경에서 데이터 무결성과 프라이버시 보호에 필수적임을 강조합니다.



### Looking through the mind's eye via multimodal encoder-decoder networks (https://arxiv.org/abs/2410.00047)
- **What's New**: 이 논문에서는 피험자의 fMRI(기능적 자기공명영상) 측정을 통해 대뇌의 이미지 맵핑을 새롭게 탐색합니다. 주목할 만한 점은, 고차원 fMRI 활성 상태를 비주얼 이미지와 매핑하고, 텍스트 레이블을 바탕으로 시각적 이미지를 디코딩하는 방법론을 제시하고 있다는 것입니다.

- **Technical Details**: 제안된 알고리즘은 멀티 인코더-디코더 아키텍처를 기반으로 하여, 브레인 레코딩과 비디오 샘플 간의 효과적인 포인트-투-포인트(mapping) 및 분포-투-분포(matching) 대칭성(training) 방법을 활용합니다. 또한, 기존에 존재하는 데이터셋을 세 개의 새로운 피험자 데이터로 확장하였고, 텍스트 프롬프트에 의해 유도된 뇌 활동을 통해 시각적으로 재구성하는 방법을 적용하고 있습니다.

- **Performance Highlights**: 강화된 데이터셋에서 실험한 결과, 제안된 모델이 시간적으로 일관되며 개념적으로 의미 있는 방식으로 뇌의 시각화를 재구성할 수 있다는 것을 입증하였습니다. 이는 의도하는 이미지를 효과적으로 디코딩 및 시각화하는 가능성을 보여줍니다.



### Mixture of Multicenter Experts in Multimodal Generative AI for Advanced Radiotherapy Target Delineation (https://arxiv.org/abs/2410.00046)
Comments:
          39 pages

- **What's New**: 이번 논문에서는 Clinical AI 모델의 편향을 극복하기 위해 Mixture of Multicenter Experts (MoME) 접근법을 도입하였으며, 이는 다양한 임상 전략의 Specialized expertise를 통합하여 AI 모델의 범용성과 적응성을 향상시킵니다.

- **Technical Details**: MoME 모델은 각 의료 센터의 이미지와 임상 노트를 포함한 few-shot 샘플로 훈련되었으며, prostate cancer의 방사선 치료 목표 영역 delineation에서 기존 방법보다 뛰어난 성능을 보였습니다. MoME는 중앙 집중식 데이터 훈련보다 적은 데이터로도 각 의료 센터의 특성과 데이터 분포에 신속하게 적응할 수 있습니다.

- **Performance Highlights**: MoME 기반 모델은 전통적인 AI 모델들보다 목표 영역 delineation에서 눈에 띄게 성능이 향상되었으며, 특히 다양한 데이터 특성이 있을 경우 그 효과가 강조되었습니다. 이는 자원 제한이 있는 의료 시설에서도 활용 가능한 가능성을 보여줍니다.



### Moshi: a speech-text foundation model for real-time dialogu (https://arxiv.org/abs/2410.00037)
- **What's New**: 논문에서는 Moshi라는 새로운 음성-텍스트 기초 모델과 전이 실시간 대화 시스템을 소개합니다. Moshi는 독립적인 음성 인식, 텍스트 대화 및 음성 합성을 통합하여 자연스러운 대화 경험을 실현합니다.

- **Technical Details**: Moshi는 텍스트 언어 모델 백본을 기반으로 하여, Residual Vector Quantization (RVQ) 기법을 통해 음성을 토큰으로 생성하고, 사용자 음성과 모델 음성을 별도의 병렬 스트림으로 모델링합니다. 이로써 발화자 구분을 없애고 대화의 임의적인 다이내믹스를 모델링할 수 있습니다.

- **Performance Highlights**: Moshi는 160ms의 이론적 지연 시간과 실제 200ms의 지연 시간으로 실시간 게속이 가능한 대화형 대량 언어 모델입니다. 이는 자연 대화에 비해 상대적으로 짧은 반응 시간을 자랑하며, 음성 인식 및 음성 합성 기술의 개선으로 뛰어난 음성 품질과 이해도를 제공합니다.



### FeruzaSpeech: A 60 Hour Uzbek Read Speech Corpus with Punctuation, Casing, and Contex (https://arxiv.org/abs/2410.00035)
Comments:
          5 Pages, 1 Figure, Preprint of Paper Accepted in ICNLSP 2024

- **What's New**: 이 논문은 우즈베크어의 읽기 음성 데이터셋인 FeruzaSpeech를 소개합니다. 이 데이터셋은 키릴 및 라틴 알파벳의 전사본을 포함하며, 학술 연구 목적으로 무료로 제공됩니다. FeruzaSpeech는 우즈베키스탄 타슈켄트 출신의 단일 여성 화자의 고품질 녹음 60시간을 포함합니다.

- **Technical Details**: FeruzaSpeech는 BBC 뉴스와 소설인 Choliqushi의 짧은 발췌 내용을 포함한 오디오 북 녹음으로 구성되어 있으며, ASR(자동 음성 인식) 및 TTS(텍스트 음성 변환) 기술을 발전시키는데 기여할 것으로 기대됩니다. 데이터셋은 'Dev', 'Test', 'Train' 세트로 나뉘어 있으며, 고객의 데이터 처리를 단순화할 수 있는 자연 텍스트를 사용하고 있습니다.

- **Performance Highlights**: FeruzaSpeech 데이터셋이 CommonVoice 16.1와 결합되었을 때, Stateless RNN-T Conformer 모델에서 WER(단어 오류율)가 각각 cv-test에서 1.49%에서 2.12%로, usc-test에서 3.01%에서 4.58%로 향상되었습니다. 또한, USC 데이터셋의 WER은 17.4%였지만, FeruzaSpeech를 포함한 모델은 11.67%로 5.73% 개선되었습니다.



### AutoFlow: An Autoencoder-based Approach for IP Flow Record Compression with Minimal Impact on Traffic Classification (https://arxiv.org/abs/2410.00030)
Comments:
          9 pages, submitted to NOMS 2025

- **What's New**: 이 논문은 딥 러닝 기법, 특히 오토인코더(autoencoder)를 사용하여 IP 플로우 레코드를 압축하는 새로운 접근 방법을 제안합니다. 이 방법은 데이터의 유용성을 유지하면서 데이터 볼륨을 혁신적으로 줄이는 것을 목표로 하며, 대규모 실제 네트워크 트래픽 데이터셋에서 광범위한 실험을 통해 효과를 입증합니다.

- **Technical Details**: 제안된 오토인코더 기반 압축 기법은 IP 플로우 레코드를 압축하여 데이터 크기를 3.28배 줄이는 동시에 다중 클래스 트래픽 분류 작업에서 99.20%의 정확도를 유지합니다. 이 방법은 다양한 현대 애플리케이션 프로토콜 간의 구별을 지원하며, 특히 암호화된 트래픽을 효과적으로 식별할 수 있는 가능성을 보여줍니다.

- **Performance Highlights**: 압축된 데이터를 사용했을 때의 정확도는 미세하게 감소했으나(99.77%에서 99.20%로), 저장 효율성과 처리 속도에서 상당한 이점이 있으며, 실시간 분석 및 리소스 제한 환경에서의 활용도 가능성을 확대합니다.



### Machine Learning to Detect Anxiety Disorders from Error-Related Negativity and EEG Signals (https://arxiv.org/abs/2410.00028)
- **What's New**: 이 논문은 EEG 신호를 통한 불안 예측의 최신 연구를 체계적으로 검토하고, 전세계 다양한 불안 장애에 대한 EEG 및 ERN 마커를 기반으로 한 기계 학습 기술의 활용을 다룬 최초의 포괄적 리뷰입니다.

- **Technical Details**: 논문에서는 EEG(전자 뇌파도)와 ERN(오류 관련 부정기) 지표를 통해 불안장애를 감지하는 데 있어 여러 기계 학습 모델(예: support vector machine, random forests, convolutional neural networks, recurrent neural networks)을 활용한 연구를 검토하였습니다. 54개의 연구가 분석되었으며, 머신러닝을 활용해 다양한 데이터 유형에서 불안장애를 진단하는 과정이 설명됩니다. 또한, 불안장애 검출을 위한 feature extraction(특징 추출) 및 분석 방법론의 개선 필요성도 강조되었습니다.

- **Performance Highlights**: EEG와 ERN 지표를 기반으로 한 기계 학습 기법들은 다양한 불안장애 진단에 유망한 결과를 보였습니다. 특히, GAD(일반화 불안 장애), SAD(사회 불안 장애), OCD(강박 장애), PD(공황장애)와 같은 다양한 불안 장애에 대해 EEG의 성능을 최적화하기 위한 향후 연구 방향을 제안하고 있습니다.



### Cross-Lingual News Event Correlation for Stock Market Trend Prediction (https://arxiv.org/abs/2410.00024)
- **What's New**: 현대 경제 환경에서 금융 서비스를 금융 기술(FinTech)과 통합하는 것이 필수적이며, 본 연구는 다양한 글로벌 경제에서의 금융 역학을 이해하는 데 필요한 구조화된 금융 데이터셋을 구축하고 언어 간 자연어 기반의 금융 예측(NLFF) 파이프라인을 제안합니다.

- **Technical Details**: 본 연구는 감정 분석(sentiment analysis), 명명된 개체 인식(Named Entity Recognition, NER) 및 의미적 텍스트 유사성(semantic textual similarity)을 활용하여 뉴스 기사를 분석했습니다. 이를 통해 금융 사건 타임라인을 추출하고 맵핑하며 시각화하여 뉴스 사건과 주식 시장의 트렌드 간의 상관 관계를 밝혀냈습니다.

- **Performance Highlights**: 우리의 방법론은 주가 변동과 언어 간 뉴스 감정 사이의 유의미한 상관 관계를 보여줍니다. 이는 파키스탄 증권 거래소의 두 개 주요 분야에 대한 2년간의 언어 간 뉴스 데이터를 처리하여 검증되었습니다. 이 연구는 주요 사건에 대한 중요한 통찰력을 제공하며, 투자자에게 효과적인 시각화를 통해 상당한 결정 여유를 보장하고 최적의 투자 기회를 제공합니다.



### Self-Tuning Spectral Clustering for Speaker Diarization (https://arxiv.org/abs/2410.00023)
Comments:
          Submitted to ICASSP 2025

- **What's New**: 본 연구에서는 스펙트럴 클러스터링(spectral clustering) 과정에서 affinity matrix를 비율 기반으로 조정할 수 있는 새로운 가지치기 알고리즘(SC-pNA)을 소개합니다. SC-pNA는 노드별 고정 이웃 선택 방식의 단점을 극복하고, 외부 튜닝 데이터 없이 affinity matrix에서 직접 파라미터를 유도해 더욱 효과적으로 작동합니다.

- **Technical Details**: SC-pNA는 초기 affinity matrix의 각 행에서 두 개의 클러스터를 식별하고, 유사도가 높은 클러스터의 상위 p% 유사도 점수만을 유지합니다. 이후 스펙트럴 클러스터링을 수행하며, 군집 수는 최대 Eigen gap에 의해 결정됩니다. 이는 고전적인 k-평균(k-means) 방법과의 비교에서 더 복잡한 클러스터 구조를 잘 포착합니다.

- **Performance Highlights**: DIHARD-III 데이터셋에서 SC-pNA의 실험 결과는 기존의 자동 튜닝 접근 방식보다 성능이 우수하며, 계산 효율성 또한 뛰어남을 보여줍니다.



### Loneliness Forecasting Using Multi-modal Wearable and Mobile Sensing in Everyday Settings (https://arxiv.org/abs/2410.00020)
- **What's New**: 이번 연구는 스마트 링(smart ring)과 스마트워치(smartwatch)와 같은 최신 웨어러블 장치를 활용하여 외로움(loneliness) 예측 가능성을 탐색하고자 하였으며, 초기 생리적 지표를 모니터링하여 외로움을 예측하는 접근 방식을 제시하고 있습니다. 또한, 모바일 센서를 통해 수집된 행동적 징후를 분석하여 외로움의 예측력을 향상시킵니다.

- **Technical Details**: 본 연구에서는 29명의 대학생을 대상으로 2개월 간 생리적, 행동적, 맥락적 정보를 포함하는 데이터 수집을 진행하였으며, 개인화된 기계 학습(machine learning) 기법을 활용하여 외로움 수준을 7일 전으로 예측하는 모델을 개발하였습니다. 신뢰할 수 있는 결과를 위해 Heart Rate (HR) 및 Heart Rate Variability (HRV)의 여러 특성 또한 분석하였습니다. SHAP(Shapley Additive exPlanations) 값을 사용하여 모델의 설명력을 높였습니다.

- **Performance Highlights**: 이 모델을 통해 외로움 수준을 예측하는 데 있어 0.82의 높은 정확도(accuracy)와 F-1 점수(F-1 score)를 달성하였으며, 이를 통해 외로움의 조기 식별과 중재의 가능성을 제시하고 있습니다.



### Enhancing EEG Signal Generation through a Hybrid Approach Integrating Reinforcement Learning and Diffusion Models (https://arxiv.org/abs/2410.00013)
- **What's New**: 본 연구는 강화 학습(reinforcement learning)과 확산 모델(diffusion model)을 통합하여 전기 생리학적 신호인 EEG(전기 뇌파) 신호의 합성을 위한 혁신적인 접근 방식을 제시합니다. 이 방법은 전통적인 EEG 데이터 수집 방식의 어려움, 즉 참가자의 부담, 개인정보 보호 문제 및 고품질 임상 데이터를 획득하는 데 드는 비용을 해결합니다.

- **Technical Details**: 제안된 방법론은 시간 도메인(time-domain) 특성과 주파수 도메인(frequency-domain) 특성을 단일 생성 모델(generative framework) 내에서 동시적으로 모델링하는 점에 독창성이 있습니다. 이는 강화 학습 모델이 매개변수 업데이트 전략을 자율적으로 선택하도록 하여 확산 과정을 안내합니다. 데이터의 효율성을 높이기 위해 BCI Competition IV 2a 데이터셋과 자사 데이터를 통해 효과성을 검증하였습니다.

- **Performance Highlights**: 제안된 접근 방식은 생체 인식(biometrics) 식별자가 없는 합성 데이터를 생성하여 참가자의 개인정보를 보호하며, 함께 대규모 주석이 달린 데이터셋에 대한 의존도를 최소화함으로써 모델 훈련 효율성을 개선합니다. 이는 EEG 연구의 진전을 가속화하고, 다양한 EEG 데이터셋으로 모델 훈련을 위한 견고한 솔루션을 제공하여 신경 재활(neurorehabilitation) 분야에서의 맞춤형 치료 프로토콜 개발에 기여합니다.



### PHemoNet: A Multimodal Network for Physiological Signals (https://arxiv.org/abs/2410.00010)
Comments:
          The paper has been accepted at RTSI 2024

- **What's New**: 이 논문에서는 생리 신호로부터 다중 모드 감정 인식을 위한 완전 하이퍼복합 네트워크인 PHemoNet을 소개합니다. 이 네트워크는 각 모드의 특정 인코더와 융합 모듈로 구성되어 있으며, 하이퍼복합 도메인에서 파라미터화된 하이퍼복합 곱셈(Parameterized Hypercomplex Multiplications, PHMs)을 통해 레이턴트 관계를 캡처합니다.

- **Technical Details**: PHemoNet 아키텍처는 다중 모드 생리 신호에서 감정을 인식하기 위해 설계되었습니다. 각각의 인코더는 신호의 고유 차원인 하이퍼복합 도메인에서 정의되며, 전극 뇌파(EEG), 안구 데이터, 피부 전도 반응(GSR), 심전도(ECG) 신호를 입력으로 사용합니다. 이 접근법은 감정의 진정한 상태를 파악하기 위해 개발되었습니다.

- **Performance Highlights**: MAHNOB-HCI 데이터셋에서 valence와 arousal을 분류하는 작업에서 현재 최첨단 모델을 초과하는 성능을 달성했습니다. 이로써 PHemoNet은 다중 모드 감정 인식 문제에서의 진전을 이루었습니다.



### Low-code from frontend to backend: Connecting conversational user interfaces to backend services via a low-code IoT platform (https://arxiv.org/abs/2410.00006)
Comments:
          5 pages, 6 figures. In 3rd Conference on Conversational User Interfaces (CUI21), July 2021, Bilbao (online), Spain

- **What's New**: 본 논문은 기존의 챗봇 개발 플랫폼과 프레임워크가 언어 및 대화 부분 설정을 용이하게 하는 반면, 백엔드 서비스와 비즈니스 기능에 연결하는 과정에서 많은 수작업 코딩이 필요하다는 문제를 다룹니다. 이를 해결하기 위한 접근법으로, IoT 플랫폼을 미들웨어로 사용하여 챗봇을 프론트엔드로 하는 아키텍처를 제안합니다. 특히, 오픈 소스 개발 플랫폼인 Rasa와 Node-RED를 결합하여 프론트엔드와 백엔드 간의 저코드 또는 무코드 개발을 가능하게 하는 방법을 제시합니다.

- **Technical Details**: 챗봇의 주요 구성 요소는 Language Understanding, Dialog Management, Language Generation입니다. 이 시스템의 Dialog Management 부문은 사용자의 입력에 따라 챗봇이 어떻게 작동할지를 결정하며, Fulfillment 부문은 API 호출 및 외부 서비스와의 상호작용을 담당합니다. 본 연구는 Node-RED에서 호스팅되는 액션 서버를 통해 Rasa 챗봇의 Fulfillment 기능을 구현하는 방법을 제안합니다.

- **Performance Highlights**: 제안된 아키텍처는 개발 시간과 비용을 줄이는 데 기여하며, 사용자가 Rasa와 Node-RED의 결합을 통해 개인 챗봇을 설정할 수 있도록 돕습니다. 이는 교육 및 실습 환경에서도 많은 도움이 될 것으로 기대됩니다.



### Machine Learning and Econometric Approaches to Fiscal Policies: Understanding Industrial Investment Dynamics in Uruguay (1974-2010) (https://arxiv.org/abs/2410.00002)
- **What's New**: 이 논문은 1974년부터 2010년까지 우루과이의 산업 투자에 대한 재정 인센티브(fiscal incentives)의 영향을 탐구합니다.

- **Technical Details**: 연구에서는 계량경제학 모델(econometric models)과 머신 러닝(machine learning) 기법을 결합한 혼합 방법론(mixed-method approach)을 사용하여 재정 혜택(fiscal benefits)의 단기 및 장기 효과를 분석합니다.

- **Performance Highlights**: 결과는 재정 인센티브가 장기 산업 성장(long-term industrial growth)을 촉진하는 중요한 역할을 한다는 것을 확인하며, 안정적인 거시경제 환경(stable macroeconomic environment), 공공 투자(public investment), 및 신용 접근(access to credit)의 중요성을 강조합니다.



### Satellite image classification with neural quantum kernels (https://arxiv.org/abs/2409.20356)
- **What's New**: 이번 연구는 양자 기계 학습(QML)의 실용적인 적용 가능성을 다루며, 특히 위성 이미지 분류에 중점을 두고 있습니다. 기존의 간단한 데이터셋에 대한 벤치마킹을 넘어 실제 데이터를 활용한 복잡한 분류 문제 해결을 목표로 합니다.

- **Technical Details**: 연구에서는 이미지의 차원을 줄인 후, 훈련된 양자 신경망(QNN)에서 구축된 임베딩 양자 커널(EQK)을 사용하여 태양광 패널을 포함한 이미지 분류를 수행합니다. $1$-to-$n$ 및 $n$-to-$n$ NQK 방식을 적용하여 각각 평균 테스트 정확도를 86% 및 88% 이상 달성하였습니다.

- **Performance Highlights**: 두 가지 모델 모두 비슷한 성능을 보여주며, 특히 $n$-to-$n$ 전략을 사용한 경우 3개의 특징과 8개의 큐비트로 88% 이상의 테스트 정확도를 달성하고, QNN의 서브 최적 훈련에 대해서도 결과의 강건성을 나타냅니다.



### Supervised Multi-Modal Fission Learning (https://arxiv.org/abs/2409.20559)
- **What's New**: 이번 논문에서는 다중 모달(multimodal) 데이터셋에서의 학습을 위한 새로운 모델인 Multi-Modal Fission Learning (MMFL)을 제안했습니다. MMFL은 전통적인 방법들이 공유된 구성 요소(shared component)이나 개별 구성 요소(individual components)만 추출하는 데 그친 것에 반해, 전 세계적으로 공동이면서 부분적으로 공동 및 개별 구성 요소를 동시에 식별하는 방식을 사용합니다.

- **Technical Details**: MMFL은 반응 변수(response variable)의 감독(supervision)을 통해 예측 가능한 잠재 구성 요소(predictive latent components)를 식별합니다. 또한 불완전한 데이터 세트(incomplete multimodal data)를 통합하는 자연스러운 확장을 가지고 있습니다. 다양한 기존 모달 알고리즘과 비교하여 MMFL의 효율성을 실험적으로 입증했습니다.

- **Performance Highlights**: 시뮬레이션 연구를 통해 MMFL은 완전 및 불완전한 모달 설정에서 다양한 기존 다중 모달 알고리즘보다 뛰어난 성능을 보였습니다. 이 모델은 알츠하이머 병(Alzheimers Disease)의 조기 예측을 위해 ADNI(Alzheimers Disease Neuroimaging Initiative) 데이터셋을 사용한 실제 사례 연구에 적용되었으며, 기존 방법보다 더 정확한 예측과 모달 간의 상관관계에 대한 통찰을 제공했습니다.



### Best Practices for Responsible Machine Learning in Credit Scoring (https://arxiv.org/abs/2409.20536)
- **What's New**: 이번 논문에서는 머신러닝을 이용한 신용 평가 모델에서의 편향 문제와 그 해결 방안에 대한 최신 연구들을 비체계적으로 검토하였습니다. 특히, 공정성(fairness), 거부 추론(reject inference), 설명 가능성(explainability)을 중점적으로 다루어 책임 있는 머신러닝 모델 개발을 위한 최선의 관행을 제시합니다.

- **Technical Details**: 신용 평가에서의 머신러닝 모델은 고용 이력, 인구통계 데이터, 금융 데이터 등을 기반으로 하여 대출자가 대출을 상환할 수 있는 가능성을 예측합니다. 다양한 머신러닝 기법이 사용되며, 로지스틱 회귀(logistic regression), 그래디언트 부스팅(gradient boosting), 결정 트리(decision trees) 등의 기술이 포함됩니다. 또한, 다양한 방법론을 통해 데이터 사전 처리, 모델 내 조정, 결과 후 처리 등을 통해 공정성을 개선하는 방법이 논의됩니다.

- **Performance Highlights**: 본 논문은 거부된 대출 신청에 대한 정보를 활용하여 신용 접근성을 확대하고, 거부된 지원자가 자신들의 거절 사유를 이해할 수 있도록 하는 중요성도 강조합니다. 머신러닝 모델의 복잡성이 증가함에 따라, 모델 결정 과정에 대한 이해를 높이기 위한 설명 가능성 기술이 필요하다는 점도 부각됩니다.



### End-to-End Conformal Calibration for Optimization Under Uncertainty (https://arxiv.org/abs/2409.20534)
- **What's New**: 이 논문은 조건부 강건 최적화(conditional robust optimization) 문제를 해결하기 위해 엔드투엔드(end-to-end) 프레임워크를 개발했습니다. 이 프레임워크는 컨포멀 예측(conformal prediction)을 통해 강건성과 보정(calibration) 보장을 제공합니다.

- **Technical Details**: 이 연구는 ML 모델을 엔드투엔드로 훈련시키는 방법론을 제안하는데, 이는 다운스트림(downstream) 의사결정 목표 및 보정된 불확실성 추정치를 통합합니다. 이 과정에서 부분 입력 볼록 신경망(partially input-convex neural networks, PICNNs)을 사용하여 임의의 볼록 불확실성 집합을 근사합니다. 또한, 컨포멀 예측 과정에서의 정확한 기울기를 계산하는 효율적인 방법을 개발했습니다.

- **Performance Highlights**: 에너지 저장 중재 및 포트폴리오 최적화 문제에 대한 실험을 통해, 제안된 방법이 기존의 ETO(estimate then optimize) 방법보다 일관되게 개선된 성능을 보여주었습니다.



### Upper and Lower Bounds for Distributionally Robust Off-Dynamics Reinforcement Learning (https://arxiv.org/abs/2409.20521)
Comments:
          48 pages, 3 figures, 2 tables

- **What's New**: 이번 연구에서는 'off-dynamics' 강화 학습에서 정책 훈련 및 배포 환경 간의 차이를 다루고 있으며, 이를 극복하기 위해 분포적으로 강건한 마르코프 결정 과정(distributionally robust Markov decision processes) 프레임워크를 통해 전이 역학의 불확실성에 강한 정책 학습에 초점을 맞추고 있습니다.

- **Technical Details**: 제안된 알고리즘 We-DRIVE-U는 평균 서브옵티멀리티(average suboptimality)가 $	ilde{	ext{O}}(d H rac{	ext{min}iggrac{1}{ho}, H}{	ext{sqrt}(K)})$ 형태를 가지고 있으며, 여기서 $K$는 에피소드 수, $H$는 지평선 길이, $d$는 기능 차원, $ho$는 불확실성 수준을 나타냅니다. 이 연구는 새로운 난이도 있는 사례를 구성하고, 주어진 설정에서 최초의 정보 이론적 하한을 유도하여 알고리즘의 최적성에 대한 통찰을 제공합니다.

- **Performance Highlights**: We-DRIVE-U는 정책 전환 및 이중 최적화 문제 해결을 위한 oracle 호출에서 $	ext{O}(dH	ext{log}(1+H^2K))$의 계산 효율성을 제공하며, 이는 기존 알고리즘에 비해 상당한 개선을 보여줍니다.



### SMLE: Safe Machine Learning via Embedded Overapproximation (https://arxiv.org/abs/2409.20517)
- **What's New**: 본 연구는 규정된 속성을 보장하는 차별화 가능한 기계 학습(ML) 모델을 훈련하는 혁신적인 접근 방식을 제안합니다. 주요 구성 요소로는 효율적인 검증을 위한 일반적이고 간단한 아키텍처, Projected Gradient Method를 기반으로 한 엄격한 훈련 알고리즘, 강력한 반례 검색 문제의 포뮬레이션이 포함됩니다.

- **Technical Details**: 제안된 아키텍처는 Safe ML via Embedded overapproximation (SMLE)로 명명되며, 주 네트워크에 저복잡도의 훈련 가능한 overapproximator를 추가하여 작동합니다. 이 아키텍처는 특정 클래스의 속성 및 설계 선택에 따라 다항 시간 복잡성을 가진 보수적인 검증을 가능하게 합니다.

- **Performance Highlights**: 우리의 접근 방식은 훈련 데이터 및 모델 예측 시 속성 강제화를 포함하는 베이스라인과 경쟁력 있는 성과를 보이며, 선형 부등식 및 다중 클래스 분류에서 상호 배타적인 속성에 대해 평가되었습니다. 정확도는 다소 낮지만, 안전 보장을 유지하면서 추론 과정의 복잡도는 증가하지 않습니다.



### COLLAGE: Collaborative Human-Agent Interaction Generation using Hierarchical Latent Diffusion and Language Models (https://arxiv.org/abs/2409.20502)
Comments:
          9 pages, 6 figures

- **What's New**: 새로운 프레임워크 COLLAGE는 대규모 언어 모델(LLMs)과 계층적 모션 특정 벡터 양자화 변분 오토인코더(VQ-VAE)를 활용하여 협력적인 에이전트-오브젝트-에이전트 상호작용을 생성합니다. 이 접근법은 풍부한 데이터셋 부족 문제를 해결하고 LLM의 지식 및 추론 능력을 이용하여 생성적인 확산 모델을 안내합니다.

- **Technical Details**: COLLAGE 모델은 다단계의 추상화 수준에서 다양한 모션 특정 특성을 포착하는 계층적 VQ-VAE 구조를 사용합니다. 이 모델은 잠재 공간에서 작동하는 확산 모델과 LLM이 생성한 모션 계획 신호를 통합하여, 노이즈 제거 과정을 안내하고 결과적으로 명령어에 따라 특정 모션 생성을 가능하게 합니다.

- **Performance Highlights**: CORE-4D 및 InterHuman 데이터셋에 대한 실험 결과는 이 접근법이 기존의 최첨단 방법들을 초월하여 현실적이고 다양한 협력적 인간-객체-인간 상호작용을 생성하는 효과를 입증합니다. 이 연구는 로보틱스, 그래픽스 및 컴퓨터 비전과 같은 다양한 분야에서 복잡한 상호작용 모델링의 새로운 가능성을 열어줍니다.



### Online Decision Deferral under Budget Constraints (https://arxiv.org/abs/2409.20489)
Comments:
          15 pages, 9 figures

- **What's New**: 이 논문에서는 온라인 의사 결정에 대한 새로운 문맥적 밴딧 모델을 제안합니다. 모델은 예산 제약과 다양한 부분 피드백 모델을 포함하며, ML 모델의 성능이 인간 전문가와 동일하거나 더 나을 때 결정을 자동화할 수 있는 방식으로 설계되었습니다.

- **Technical Details**: 모델링은 두 개의 선택지(armed)로 구성된 문맥적 밴딧 문제를 기반으로 하며, 이는 예산 제한과 관련된 수익송출 모델을 고려합니다. 여러 설정에서 시뮬레이션 및 실제 데이터 실험을 통해 알고리즘의 성능을 평가하며, 순차적으로 발생하는 맥락을 관찰하여 최적의 결정을 내릴 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, 제안된 알고리즘은 여러 실제 데이터 세트에서 뛰어난 성능을 나타내어, ML 모델과 인간 전문가의 결정을 효율적으로 조율할 수 있는 것으로 확인되었습니다.



### Linear Projections of Teacher Embeddings for Few-Class Distillation (https://arxiv.org/abs/2409.20449)
- **What's New**: 이 논문에서는 기존 Knowledge Distillation(KD) 접근 방식의 한계를 극복하기 위해 Learning Embedding Linear Projections(LELP)라는 새로운 기법을 제안합니다. LELP는 교사 모델의 마지막 레이어에서의 표현을 이용해 학습하는 방법으로, 교사의 내재적 패턴을 효과적으로 포착할 수 있도록 설계되었습니다.

- **Technical Details**: LELP는 교사 모델의 임베딩 공간에서 유용한 선형 부분 공간(informative linear subspaces)을 탐색하고 이를 가상의 서브클래스(pseudo-subclasses)로 분할하여 학생 모델이 이를 모방하도록 학습합니다. 이 과정에서 단일 통합 크로스 엔트로피 손실(unified cross-entropy loss)을 사용하여 훈련을 수행합니다.

- **Performance Highlights**: 대규모 NLP 벤치마크인 Amazon Reviews와 Sentiment140에서의 실험 결과, LELP는 기존의 상태-최고(distillation algorithms) 기법에 비해 이진 및 소수 클래스 문제에서 일관되게 경쟁력을 갖추고 있으며, 대체로 우수한 성능을 보였습니다. LELP는 데이터 효율성(data efficiency), 훈련 속도(tyaining speed) 개선 및 반지도 학습(semi-supervised) KD 시나리오에서의 실질적인 향상을 제공합니다.



### POMONAG: Pareto-Optimal Many-Objective Neural Architecture Generator (https://arxiv.org/abs/2409.20447)
- **What's New**: 이번 연구에서는 많은 목적(Many-Objective)를 고려한 Neural Architecture Generator(POMONAG)를 소개합니다. 이는 기존의 DiffusionNAG 방법을 확장하여 정확도뿐만 아니라 모델 복잡성, 계산 효율성 및 추론 지연과 같은 다양한 목표를 동시에 고려합니다.

- **Technical Details**: POMONAG는 성능 예측기(Performance Predictor) 모델을 통합하여 보다 정확한 성능 예측을 통해 생성하는 아키텍처의 품질을 향상시키며, 파레토 최적(Pareto-optimal) 아키텍처 생성을 지원합니다. POMONAG의 메타 데이터셋(Meta-Dataset)은 훈련 여건 개선을 위해 확장되었으며, 다수의 목적을 효과적으로 균형 있게 처리하기 위한 파레토 프론트 필터링(Pareto Front Filtering) 및 스트레칭(Stretching) 기법이 적용되었습니다.

- **Performance Highlights**: POMONAG는 NASBench201 및 MobileNetV3에서 실험을 수행하여 기존 최고의 성능을 초과하는 결과를 보여주었습니다. 특히, 다양한 이미지 분류 데이터셋에서 높은 정확도를 제공하면서도 요구되는 훈련 모델 수를 크게 줄임으로써 효율성을 증명했습니다.



### Optimism in the Face of Ambiguity Principle for Multi-Armed Bandits (https://arxiv.org/abs/2409.20440)
- **What's New**: 본 논문에서는 새로운 Follow-The-Perturbed-Leader (FTPL) 알고리즘을 제안하여 적대적(adversarial) 및 확률적(stochastic) 다중 팔 밴딧(multi-armed bandit) 문제에 대한 최적 정책을 생성합니다. 이 알고리즘은 현재의 FTPL 방법과 달리 	extit{모호한(ambiguous)} 분포를 사용하는데, 이는 주어진 집합에 속하는 것만 알려져 있습니다.

- **Technical Details**: 제안된 알고리즘은 '모호함 속에서의 낙관주의(optimism in the face of ambiguity)' 원칙을 따르며, 이를 통해 팔의 샘플링 확률을 매우 효율적으로 계산할 수 있는 이분 탐색(bisection) 알고리즘을 개발하였습니다. 기존의 FTRL 알고리즘과 비교해 비용이 최소 $10^4$ 배 빠르며, 최적의 보상 분포에서 팔을 선택합니다.

- **Performance Highlights**: 새로운 방법은 FTRL과 FTPL을 통합하여 최적의 후회(regret) 분석을 가능하게 하며, 여러 FTRL 방법에도 특별한 경우로 포함됩니다. 또한 기존 FTPL 방법론으로는 가능한 것으로 보이지 않았던 다양한 최적 FTRL 방법들을 포함합니다.



### Conformal Prediction for Dose-Response Models with Continuous Treatments (https://arxiv.org/abs/2409.20412)
Comments:
          10 pages main text, 8 pages references and appendix

- **What's New**: 이번 연구는 continuous treatment에 대한 dose-response 문제를 covariate shift로 바라보아,
weighted conformal prediction을 활용한 새로운 접근 방식을 제안합니다. 이 방법은 개인화된 dose-response 모델에 효과적인 예측 구간을 생성하는 데 중점을 두고 있습니다.

- **Technical Details**: 연구에서는 propensity estimation, conformal predictive systems 및 likelihood ratios를 통합하여 weighted conformal prediction을 통해 예측 구간을 도출하는 방법론을 설명합니다. 또한, kernel functions를 가중치로 적용하여 각 치료 값을 위한 local coverage를 근사화합니다.

- **Performance Highlights**: 새로운 합성 벤치마크 데이터셋을 사용하여 covariate shift 가정이 dose-response 모델을 위한 robust prediction intervals를 생성하는 데 어떻게 중요한 역할을 하는지를 보여줍니다. 이 연구는 유의미한 결정 내리기를 지원하기 위한 UQ(uncertainty quantification)의 중요성을 강조합니다.



### Beyond Derivative Pathology of PINNs: Variable Splitting Strategy with Convergence Analysis (https://arxiv.org/abs/2409.20383)
- **What's New**: 이 논문은 물리 정보에 기반한 신경망(Physics-informed Neural Networks, PINNs)의 기본적인 문제점에 대한 새로운 관점을 제시합니다. 기존의 가정이 잘못되었음을 입증하며, 미분계수의 비제어적인 동작이 PINNs의 비수렴 문제에 기여하는 것으로 확인하였습니다. 이러한 문제를 해결하기 위해 변수를 분할하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 이 연구에서는 PINNs의 손실 함수(minimizing loss function)를 0으로 만들더라도 반드시 정부(PDE)의 해에 수렴하는 것은 아님을 보여줍니다. 이를 통해 우리는 순차적으로 PINNs의 예측된 해에 대한 미분의 동작을 규제할 수 있는 방법이 없음을 드러내었습니다. 제안된 변수 분할(variable splitting) 전략은 이러한 문제를 해결하며, 일반화된 해에 수렴하는 것을 보장하는 방법으로 작용합니다. 새롭게 도입된 보조 변수(auxiliary variable)를 통해 예측된 해의 기울기를 직접 모니터링하고 규제할 수 있게 됩니다.

- **Performance Highlights**: 변수 분할 전략을 통해 제안된 방법은 2차 선형 PDE에 대해 일반화된 해로의 수렴을 보장합니다. 이는 다양한 문제에의 적용 가능성을 시사하며, PINNs의 기존 문제들을 해결하는 새로운 길을 제공합니다.



### Frequency Adaptive Normalization For Non-stationary Time Series Forecasting (https://arxiv.org/abs/2409.20371)
Comments:
          NeurIPS 2024 Poster

- **What's New**: 이 논문은 시간 시계열 예측에서 비정상 데이터(non-stationary data)의 문제를 해결하기 위한 새로운 방법인 Frequency Adaptive Normalization (FAN)을 제시합니다. 기존의 reversible instance normalization 방식은 기본적인 추세(trend) 표현에 한계가 있었고, 계절 패턴(seasonal patterns)을 처리하는데 부족했습니다. FAN은 푸리에 변환(Fourier transform)을 사용하여 각 인스턴스(instance)별로 주요 빈도(frequency) 구성 요소를 식별하여 비정상성을 처리합니다.

- **Technical Details**: FAN은 비정상성을 다루기 위해 각 입력 인스턴스에서 가장 두드러진 K개의 주파수 성분을 필터링하여 사용합니다. 이 과정에서, 예측(neural network 사용 시 간단한 MLP 모델)하여 미래의 비정상 정보를 예측하고, 이 정보를 사용하여 출력을 재구성합니다. FAN은 여러 예측 모델에 적용될 수 있는 모델 독립적 방법입니다.

- **Performance Highlights**: FAN은 8개의 벤치마크 데이터 세트에서 4개의 일반적인 예측 모델에 적용했으며, 평균 MSE에서 7.76%에서 37.90% 향상된 성능을 나타냈습니다. 또한, FAN은 기존의 최첨단 정규화(normalization) 기법들과 비교하여 탁월한 성능을 보였습니다.



### The Perfect Blend: Redefining RLHF with Mixture of Judges (https://arxiv.org/abs/2409.20370)
Comments:
          submitted to conference

- **What's New**: 이 연구에서는 다중 작업 학습(MTL)에서의 강화를 통해 인공지능 모델의 후처리를 개선하기 위해 제약 생성 정책 최적화(Constrained Generative Policy Optimization, CGPO)라는 혁신적인 패러다임을 소개합니다. CGPO는 비용 효율적인 제약 정책 최적화와 샤프화(stratification)를 통해 RLHF의 최적 혼합을 식별할 수 있습니다.

- **Technical Details**: CGPO의 핵심은 다양한 작업에 대한 맞춤형 최적화 전략을 적용하고, 룰 기반(judge) 및 LLM 기반(judge) 두 가지 종류의 심사를 통해 보상 해킹(reward hacking)을 탐지 및 완화하는 것입니다. 이를 위해 새로운 제약 RLHF 최적화기(Calibrated-Regularized Policy Gradient, CRPG; Constrained Online Direct Preference Optimization, CODPO; Calibrated-Regularized Reward Ranking Finetuning, CRRAFT)가 개발되었습니다.

- **Performance Highlights**: CGPO는 일반 대화, STEM 질문, 지침 준수 및 코딩을 포함한 다양한 작업에서 PPO 및 DPO와 같은 기존의 RLHF 알고리즘을 초월하는 성능을 보여주었습니다. AlpacaEval-2에서 7.4%, Arena-Hard에서 12.5% 개선된 성과를 달성했으며, 전반적으로 모든 벤치마크와 작업에서 동향이 일관성을 보였습니다.



### Rotated Runtime Smooth: Training-Free Activation Smoother for accurate INT4 inferenc (https://arxiv.org/abs/2409.20361)
- **What's New**: 이번 연구에서는 대형 언어 모델의 양자화(quantization)를 위한 새로운 활성화 스무딩 방법인 Rotated Runtime Smooth (RRS)를 제안합니다. 이 방법은 기존의 Outlier 처리 방식의 한계를 극복하기 위해 채널별 아웃라이어와 스파이크 아웃라이어를 구분하고, Runtime Smooth와 회전(rotate) 작업을 결합하여 아웃라이어 문제를 해결합니다.

- **Technical Details**: 제안된 방법은 Runtime Smooth(RS)를 사용하여 실행 중에 채널별 아웃라이어를 제거하고, 회전 작업을 통해 스파이크 아웃라이어를 줄입니다. RS는 채널별 최대값을 이용해 활성화를 부드럽게 만들어 아웃라이어 문제를 완화시키며, 최종적으로 INT4 양자화를 위한 퓨즈된 GEMM 커널(fused GEMM kernel)에 입력으로 제공됩니다. 이를 통해 하드웨어 호환성을 유지하면서도 성능을 개선합니다.

- **Performance Highlights**: LLaMA 가족 및 Qwen 모델에서의 실험 결과, 제안된 방법이 기존 최첨단 기술을 초월하여 WikiText-2의 perplexity를 57.33에서 6.66으로 개선하는 성과를 보여주었습니다. 이는 INT4 추론을 위한 아웃라이어 문제를 효과적으로 해결했음을 나타냅니다.



### Fine-Tuning Personalization in Federated Learning to Mitigate Adversarial Clients (https://arxiv.org/abs/2409.20329)
- **What's New**: 이 논문은 Federated Learning (FL)에서 일부 클라이언트가 적대적일 때 전체적인 협력이 실패하는 조건을 도출하고, 이에 따라 개인화된 모델을 통해 문제를 해결하는 방법을 제시합니다.

- **Technical Details**: 이 연구는 개인 클라이언트가 자신의 데이터 분포에 적합한 모델을 가질 수 있도록 하여 FL 알고리즘을 개선하는 personalization 기법을 분석합니다. 특히, 생성된 모델이 클라이언트의 데이터에 대해 Generalization 성능을 얼마나 잘 나타내는지를 수치적으로 평가합니다. 데이터 이질성과 적대적 클라이언트 비율에 따라 협력 수준을 조정해야 함을 강조합니다.

- **Performance Highlights**: 실험 결과, Mean Estimation 및 Binary Classification 문제에 대한 사례 연구를 통해 일반화된 FL 프레임워크가 적대적 클라이언트가 존재할 때 fine-tuned personalization보다 성능이 저하되는 상황을 구체적으로 분석합니다.



### Old Optimizer, New Norm: An Anthology (https://arxiv.org/abs/2409.20325)
- **What's New**: 이 논문에서는 Adam, Shampoo, Prodigy와 같은 딥러닝 최적화 기법들을 전통적인 볼록(convex) 및 근사 2차 이론(approximate second-order theory)에서 1차 이론(first-order theory)으로 이해할 수 있다는 새로운 관점을 제시합니다. 이 연구는 다양한 텐서(tensor) 역할에 따라 최적화 알고리즘을 설계할 수 있는 새로운 디자인 공간을 제안합니다.

- **Technical Details**: 논문에서는 각각의 최적화 기법이 지수 이동 평균(EMA)을 비활성화했을 때, 특정 노름(norm) 하의 경량화된 경량적 경사하강법(steepest descent)으로 간주될 수 있음을 설명합니다. 이 개념은 신경망의 아키텍처를 적절히 메트리제이션(metrizing)함으로써 더 안정적이고 빠른 훈련을 가능하게 할 수 있다고 주장합니다.

- **Performance Highlights**: Adam 최적화 기법은 무한 노름(infinity norm) 하에서 경사하강법으로 연결되며, 신경망의 텐서 구조를 존중하는 방식으로 최적화를 수행합니다. 이로써 알고리즘 설계자들이 보다 적절한 노름 선택을 통해 성능 향상을 이룰 수 있을 것으로 기대됩니다.



### A SSM is Polymerized from Multivariate Time Series (https://arxiv.org/abs/2409.20310)
- **What's New**: 이번 연구에서는 다변량 시계열(Multivariate Time Series, MTS) 예측을 위한 새로운 방법인 Poly-Mamba를 제안합니다. 이는 기존의 상태 공간 모델(State Space Model, SSM)에서 다채널 종속성 변화(Channel Dependency variations with Time, CDT)를 명시적으로 모델링하지 않았던 한계를 극복하고자 합니다.

- **Technical Details**: Poly-Mamba는 다변량 직교 다항식 근사(Multivariate Orthogonal Polynomial Approximation, MOPA)를 핵심 개념으로 하여, 각 채널 간의 종속성을 나타내기 위해 혼합 변수를 포함하는 다변량 직교 함수 공간으로 원래의 직교 함수 기반 공간을 확장합니다. 연구에서는 선형 채널 혼합(Linear Channel Mixing, LCM) 방법을 통해 채널 간의 간단한 선형 관계를 처리하고, Order Combining 방법을 제안하여 채널별로 적응적인 CDT 패턴을 생성합니다.

- **Performance Highlights**: Poly-Mamba는 6개의 실제 데이터셋에 대한 실험에서 기존 최첨단(SOTA) 방법보다 우수한 성능을 보였습니다. 특히, 채널 수가 많고 복잡한 상관관계를 가진 데이터셋에서 두드러진 성능향상이 나타났습니다.



### PersonalLLM: Tailoring LLMs to Individual Preferences (https://arxiv.org/abs/2409.20296)
Comments:
          28 pages, 6 figures

- **What's New**: 이 논문에서는 LLM(대형 언어 모델)을 개인 사용자의 특성과 선호에 맞게 조정하기 위한 새로운 공개 벤치마크인 PersonalLLM을 제시합니다. 기존의 정렬(Alignment) 벤치마크가 통일된 선호를 전제로 하는 반면, PersonalLLM은 다양한 고품질 응답과 함께 열린 질문을 선별하여 사용자들의 이질적인 숨은 선호를 반영합니다.

- **Technical Details**: PersonalLLM은 10,402개의 열린 질문과 이에 대한 8개의 고품질 응답으로 구성된 데이터셋을 통해 사용자의 선호 모델을 샘플링합니다. 이 모델은 다양성과 역사적 사용자 기반을 시뮬레이션 하여 사용자 맞춤화를 위한 새로운 기법을 가능하게 합니다. 또한, 맥락 내 학습(In-Context Learning) 및 메타 학습(Meta-Learning) 기법으로 연속 데이터 부족 문제를 해결하기 위한 기초를 마련합니다.

- **Performance Highlights**: PersonalLLM은 사용자 개별적인 니즈에 맞춘 최적의 응답을 생성할 수 있는 능력을 보여주며, 대화형 AI의 효율성과 유용성을 높이는 가능성을 가지고 있습니다. 특히, 사용자 맞춤형 교육 경험이나 고객 지원 챗봇의 정확한 대응 등에 활용될 수 있는 잠재력을 가지며, 기존의 개인화 시스템보다 인상적으로 다양한 선호를 반영하는 환경을 제공합니다.



### Constraining Anomaly Detection with Anomaly-Free Regions (https://arxiv.org/abs/2409.20208)
Comments:
          Accepted at the 15th IEEE International Conference on Knowledge Graph (ICKG)

- **What's New**: 이 논문에서는 anomaly detection(이상 탐지)을 개선하기 위한 새로운 개념인 anomaly-free regions (AFR)를 제안하고 있습니다. AFR는 데이터 공간에서 이상이 존재하지 않는 것으로 알려진 영역으로, 이는 도메인 지식을 통해 확인될 수 있습니다.

- **Technical Details**: AFR은 정상 데이터 포인트의 수에 따라 비율이 일정해야 하며, 정상 데이터 포인트가 없는 영역을 포함할 수 있습니다. 기존의 일반적인 방법들과 달리 AFR는 존재하거나 존재하지 않는 데이터 분포에 대한 제약을 제공하며, 기존의 이상 탐지 방법들과 비교했을 때 성능이 향상된다는 이점을 가지고 있습니다.

- **Performance Highlights**: AFR를 활용한 알고리즘이 랜덤 추측 기반의 효율적인 알고리즘보다 성능이 우수함을 보여주며, 특정 데이터 세트에서 기존의 최첨단 방법을 초월하는 성과를 나타냅니다.



### SetPINNs: Set-based Physics-informed Neural Networks (https://arxiv.org/abs/2409.20206)
- **What's New**: 최근 연구에서 제안된 SetPINNs는 전통적인 Finite Element Methods(FEM)의 장점을 활용하여 물리적 시스템의 내재적 의존성을 모델링할 수 있는 새로운 접근 방식을 제시합니다. 이 방법은 데이터 주도적(data-driven)이며 메시(mesh)가 필요하지 않고, 물리적 제약을 존중합니다.

- **Technical Details**: SetPINNs는 이웃하는 포인트 간의 의존성을 고려하며, 이를 위해 attention mechanism을 사용하여 입력 요소 집합 내 의존성을 효율적으로 캡처합니다. 전통적인 point-wise PINN loss 대신 set-wise physics loss를 도입하여 물리적 제약을 올바르게 적용합니다.

- **Performance Highlights**: SetPINNs는 다양한 물리적 시스템에서 우수한 일반화 성능과 정확성을 보여주었으며, 실패 모드를 완화하고 기존 접근 방식에 비해 수렴 속도가 더 빠른 것으로 입증되었습니다. 또한, 두 개의 실제 물리 시스템을 대상으로 한 실험에서도 그 효용성이 입증되었습니다.



### Choosing DAG Models Using Markov and Minimal Edge Count in the Absence of Ground Truth (https://arxiv.org/abs/2409.20187)
Comments:
          19 pages, 14 figures, 1 table

- **What's New**: 이 논문에서는 데이터 세트에 대한 마르코프 조건(Markov condition)을 검증하기 위해 새로운 비모수(pointwise consistent) 방법인 마르코프 체크(Markov Checker)를 제안합니다. 이를 통해 DAG(Directed Acyclic Graph) 또는 CPDAG(Completed Partially Directed Acyclic Graph) 모델을 효과적으로 평가할 수 있습니다.

- **Technical Details**: 마르코프 체크는 고전적인 통계 모델링에서 요구되는 비모수적 통계 테스트로, 학습된 인과 모델에서 d-분리를 통한 조건부 독립성을 분석합니다. 논문에서는 CAFS(Cross-Algorithm Frugality Search) 알고리즘도 소개하여 마르코프 체크를 통과하지 못한 DAG 모델이나 최소 간선을 가지지 않는 모델을 제거하는 과정을 포함합니다.

- **Performance Highlights**: CAFS 방법을 통한 시뮬레이션 결과는 마르코프 체크의 근거 없이도 약간의 틀림없는 모델을 선택할 수 있음을 보여줍니다. 이 도구는 대규모 또는 밀접한 데이터 모델에서도 유용하며, 실제 데이터 분석에 필요한 새로운 도구를 제공하고 있습니다.



### Ensemble Kalman Diffusion Guidance: A Derivative-free Method for Inverse Problems (https://arxiv.org/abs/2409.20175)
- **What's New**: 이 논문은 Ensemble Kalman Diffusion Guidance (EnKG)라는 새로운 방법을 제안하며, 이는 기존의 역문제를 푸는 방안인 사전 학습된 diffusion 모델을 사용하여 파라미터 정보 없이도 문제를 해결할 수 있는 접근 방식을 제공합니다. 이 방법은 주로 과학적 응용에서 유용한데, 많은 경우 전진 모델에 대한 정보가 부족할 때 적용 가능합니다.

- **Technical Details**: EnKG는 두 가지 주요 단계로 구성된 예측-수정(PC) 프레임워크를 기반으로 하며, 여기서 첫 번째 단계인 예측 단계는 확률적 미분 방정식(SDE)을 활용합니다. 두 번째 단계인 수정 단계는 현재 점을 높은 가능성 영역으로 이동시키는 프로시멀(proximal) 연산자를 사용합니다. EnKG는 다른 모델들과 달리 전혀 파생 성분을 필요로 하지 않으며, Black-box 접근 방식을 통해 전진 모델에 대한 정보를 제공합니다.

- **Performance Highlights**: 이 방법은 여러 비선형 역문제에서 효과적이며, 특히 Navier-Stokes 방정식과 같은 고차 비선형 문제에서도 정상적으로 수행됩니다. EnKG는 기존의 방법과 비교할 때, 비선형 상쇄 문제에서 더 나은 성능을 보여주며, 그 과정에서 예측의 질 또한 향상된 것으로 나타났습니다.



### ASTRA: Accurate and Scalable ANNS-based Training of Extreme Classifiers (https://arxiv.org/abs/2409.20156)
- **What's New**: 최신 XC 알고리즘인 ASTRA는 높은 정확도를 유지하면서도 수억 개의 레이블에 대해 스케일 가능성을 제공합니다. 이 알고리즘은 ANNS 기반의 훈련 방법론을 개발하였으며, 이는 기존 방법들에 비해 훈련 시간을 최대 15배까지 줄일 수 있습니다.

- **Technical Details**: ASTRA 알고리즘은 두 가지 주요 관점을 기반으로 구축되었습니다: (a) 분류기 벡터에 대한 ANNS 인덱스를 구축하고 이를 통해 hard negatives를 검색하여 손실 함수에 최적화된 negative sampling 전략을 구현합니다; (b) 분류기가 epochs를 통과하면서 ANNS 인덱스를 지속적으로 업데이트하는 것이 매우 비쌉니다. 따라서, 개선된 경량의 negative sampling 전략으로 주목할 만한 성능을 거두었습니다.

- **Performance Highlights**: ASTRA는 120M 레이블과 370M 쿼리를 포함하는 대규모 데이터셋에서 83.4의 Precision@1을 기록하며, 같은 하드웨어를 사용한 Renée보다 훈련 시간이 15배 더 짧습니다. 또한, 다른 XC 알고리즘과 비교하여 속도 면에서도 4.5배에서 최대 80.4배 빠른 성과를 보여주었습니다.



### Characterizing Model Robustness via Natural Input Gradients (https://arxiv.org/abs/2409.20139)
Comments:
          28 pages; 14 figures; 9 tables; to be published in ECCV 2024

- **What's New**: 이 연구는 Adversarial Training (적대적 훈련) 대신에 입력의 Gradient Norm (그래디언트 노름)을 규제하는 것이 자연 샘플에서 모델의 견고성을 향상시킬 수 있음을 보이며, 특히 현대 비전 변환기 아키텍처에서 효과적이라는 점이 주목할 만하다.

- **Technical Details**: Gradient Norm 규제가 활성 함수의 부드러움에 따라 성능이 달라지며, 기본적인 폭격(perturbation)에 대해 모델의 입력 그래디언트를 집중시킴으로써 모델의 견고성을 크게 향상시킬 수 있다는 분석을 포함한다. 이 연구는 이미지 엣지를 강조하여 그래디언트를 규제하는 것이 견고성에 미치는 영향을 탐구한다.

- **Performance Highlights**: Gradient Norm 훈련 방식은 ImageNet-1K에서 90% 이상의 성능을 달성하며, 최신 PGD-3 Adversarial Training의 60%에 해당하는 연산비용으로 결과를 도출한다. 이를 통해 복잡한 적대적 최적화 없이도 상당한 견고성을 확보할 수 있음을 시사한다.



### Constraint Guided Model Quantization of Neural Networks (https://arxiv.org/abs/2409.20138)
Comments:
          13 pages, 3 tables, 1 figure

- **What's New**: 이 논문은 Constraint Guided Model Quantization (CGMQ)라는 새로운 방법을 제안합니다. CGMQ는 신경망의 매개변수 비트 너비를 감소시키고, 컴퓨팅 리소스에 대한 상한선을 사용하여 훈련 중에 비용 제약조건을 충족합니다. 이는 기존의 방법들이 하이퍼파라미터 튜닝을 필요로 하는 반면, CGMQ는 이러한 조정 없이 Mixed Precision 신경망을 생성하는 데 기여합니다.

- **Technical Details**: CGMQ는 신경망의 가중치 및 활성화에 대해 적절한 비트 너비를 자동으로 찾는 방법으로, 메모리 및 계산 요건이 미리 정의된 최대값 이내에 있음을 보장합니다. 이 방법은 성능을 낮추지 않으면서도 비용 제약조건을 유지합니다. CGMQ는 기존의 Gradient-Based Methods를 기반으로 하며, 기계 학습에서 비트 너비 및 양자화 범위를 학습하는데 Focus 합니다.

- **Performance Highlights**: MNIST 데이터셋에서 CGMQ는 최첨단의 양자화 학습 알고리즘과 경쟁력 있는 성능을 보여주며, 비용 제약조건을 보장하는 데 성공합니다. CGMQ는 하이퍼파라미터 조정 없이도 competitive한 성능을 제공하며, 엣지 AI 디바이스에 적합한 신경망 모델을 만드는데 기여합니다.



### Federated Instruction Tuning of LLMs with Domain Coverage Augmentation (https://arxiv.org/abs/2409.20135)
- **What's New**: 최근의 Federated Domain-specific Instruction Tuning (FedDIT)는 제한된 크로스-클라이언트 개인 데이터를 활용하여 특정 도메인에서 모델 성능을 향상시키는 방법으로 주목받고 있습니다. 이 과정에서 서버 측의 공용 데이터와 결합하여 모델을 개선하는 방법론을 제시합니다.

- **Technical Details**: FedDIT는 클라이언트 간 도메인 커버리지를 최적화하는 것을 목표로 하며, 이를 위해 greedy client center selection과 retrieval-based augmentation을 사용하여 데이터 보강을 진행합니다. 또한, FedDCA$^*$를 통해 클라이언트 측의 계산 부담을 덜기 위해 이질적인 인코더를 사용하고 서버 측에서 특징 정렬(feature alignment)을 수행합니다.

- **Performance Highlights**: 네 개의 다양한 도메인(코드, 의료, 재정, 수학)에서 진행된 광범위한 실험을 통해 FedDCA와 FedDCA$^*$의 효과를 확인하였고, 공공 데이터의 양과 개인 정보 보호 능력 사이에는 유의미한 상관관계가 없음을 밝혔습니다. 그러나 진행된 파인튜닝의 회수가 많아질수록 개인 정보 유출의 위험이 감소하는 경향이 나타났습니다.



### DCAST: Diverse Class-Aware Self-Training Mitigates Selection Bias for Fairer Learning (https://arxiv.org/abs/2409.20126)
Comments:
          16 pages of main paper, 6 main figures

- **What's New**: 이 논문에서는 ML(기계 학습) 모델의 공정성을 보장하기 위해 기존의 선택 편향(selection bias) 외에 계층 편향(hierarchy bias)과 클래스 인식(self-training) 기술을 이용한 새로운 접근 방식을 소개합니다. DCAST(Diverse Class-Aware Self-Training)라는 모델 비의존적인 방법을 통해 데이터의 표본 다양성을 높이고, 잠재적인 미확인 편향을 줄이는 데 기여합니다.

- **Technical Details**: DCAST는 계층 편향을 고려한 클래스 인식 자기 학습을 통해 미라벨 샘플(unlabeled samples)을 활용하여 ML 모델의 성능을 극대화합니다. 계층 편향은 클러스터링 기법을 사용하여 원본 데이터에서 독특하게 분포된 샘플 그룹을 식별하고, 각 클래스마다 샘플 선택을 유도하여 클래스별 특정 편향을 생성합니다. DCAST는 일반적인 자기 학습(self-training) 방법의 확인 편향을 줄일 수 있도록 다수의 다양한 샘플을 선택합니다.

- **Performance Highlights**: DCAST를 통해 ML 모델은 11개의 데이터셋에서 기존의 자기 학습 및 기타 6개 도메인 적응 기법과 비교할 때 계층 및 다른 편향에 대한 견고성(robustness)이 향상되었습니다. 특히 고차원 데이터셋에서 이점이 더욱 두드러지며, DCAST가 식별 가능한 편향을 넘어 공정한 학습을 달성하는 유망한 전략임을 보여주었습니다.



### Continuous-Time Linear Positional Embedding for Irregular Time Series Forecasting (https://arxiv.org/abs/2409.20092)
- **What's New**: 이 논문에서는 고불규칙적으로 샘플링된 시계열 데이터( irregularly sampled time series) 예측을 위한 새로운 접근법인 CTLPE(Continuous-Time Linear Positional Embedding)를 제안합니다. 기존 연구에서는 일반적인 시계열 예측에 집중했으나, 본 연구는 불규칙한 시간 간격을 다루기 위해 변환기(transformers) 아키텍처의 위치 임베딩을 수정합니다.

- **Technical Details**: CTLPE는 시계열 데이터의 시간 정보를 효과적으로 표현하기 위해 연속 시간을 기반으로 한 선형 함수(continuous linear function)를 도입합니다. 이 방법은 불규칙한 관측 패턴과 불규칙한 시간 간격을 해결하며, 신경 제어 미분 방정식(neural controlled differential equations) 기반의 위치 임베딩을 통해 선형 임베딩이 다른 연속형 함수에 비해 더 우수함을 입증합니다.

- **Performance Highlights**: CTLPE는 다양한 불규칙 샘플링된 시계열 데이터셋에서 기존 기술들을 초월하는 성능을 보입니다. 이는 CTLPE가 정규 시계열에 대한 transformer 모델의 한계를 극복하고 불규칙 시계열의 시공간적 특성을 효과적으로 캡처할 수 있음을 보여줍니다.



### Robust LLM safeguarding via refusal feature adversarial training (https://arxiv.org/abs/2409.20089)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 보안 취약점을 해결하고자 새로운 방어 기법인 거부 기능 대항 훈련(Refusal Feature Adversarial Training, ReFAT)을 제안합니다. 이 접근법은 공격의 잠재적 영향을 시뮬레이션하여 LLM이 보다 안전한 반응을 생성하도록 훈련합니다.

- **Technical Details**: 거부 기능(ablation of refusal feature)이라는 메커니즘을 통해 LLM이 공격에 노출되는 방식을 분석하고, 이를 기반으로 ReFAT 알고리즘을 개발하였습니다. ReFAT는 LLM이 해로운 입력에 대해 거부 응답을 생성하도록 미세 조정하며, 각각의 배치에서 두 세트의 해로운 및 무해한 지침을 사용해 RF를 동적으로 계산합니다.

- **Performance Highlights**: 실험 결과, ReFAT는 세 가지 인기 있는 LLM에 대해 다양한 적대적 공격에 대한 강력함을显著 개선하며, 기존의 적대적 훈련 방법들과 비교하여 상당히 적은 계산 비용으로 효과를 나타냅니다.



### Whole-Graph Representation Learning For the Classification of Signed Networks (https://arxiv.org/abs/2409.20073)
- **What's New**: 본 논문에서는 서명 그래프(signed graphs)의 전체 그래프 표현을 학습하는 두 가지 새로운 접근 방식을 제안합니다. 특히, 기존의 방법들이 대부분 부호 없는 그래프를 처리하는 것과 달리 본 연구는 서명 그래프에 초점을 맞추었습니다.

- **Technical Details**: 첫 번째 접근 방식은 SG2V로, 이 방법은 Weisfeiler--Lehman 재표기 절차의 수정을 기반으로 한 Graph2vec의 서명 그래프 일반화입니다. 두 번째 접근 방식은 WSGCN으로, 이는 GCN의 마스터 노드(master nodes) 도입을 기반으로 한 서명 정점 임베딩 방법의 전체 그래프 일반화입니다. 또한 두 접근 방식에 대해 여러 변형을 제안합니다.

- **Performance Highlights**: 이 연구에서 제안된 방법들은 서명 그래프를 위한 대량의 데이터를 포함한 벤치마크에서 평가되었습니다. 기존 방법의 F-measure 점수가 58.57인 반면, SG2V는 73.01, WSGCN은 81.20에 도달하여 서명 전체 그래프 방법이 이 작업에 더 나은 표현을 학습함을 나타냅니다.



### Can We Break the Curse of Multiagency in Robust Multi-Agent Reinforcement Learning? (https://arxiv.org/abs/2409.20067)
- **What's New**: 이 논문에서는 표준 다중 에이전트 강화 학습(MARL) 알고리즘의 취약점을 다루기 위해, 위험한 경우의 성능을 최적화하는 분포적으로 강인한 Markov 게임(RMG) 클래스를 제안합니다. 이 접근 방식은 환경과 다른 에이전트의 전략에 의해 형성된 불확실성 집합을 기반으로 하여 성능 향상을 목표로 합니다.

- **Technical Details**: 저자들은 robust Nash equilibria와 coarse correlated equilibria (CCE)의 존재를 증명하여 이러한 RMG들의 잘 정의된 성질을 확립했습니다. 생성 모델에 대한 접근을 가정하고, CCE를 학습하기 위한 샘플 효율적인 알고리즘을 도입했습니다. 이 알고리즘은 관련 매개변수와 함께 다항적으로 샘플 복잡성이 증가합니다.

- **Performance Highlights**: 이 연구는 RMG에 대해 다중 에이전트의 저주를 극복한 첫 번째 알고리즘으로서 주목받고 있으며, 이는 MARL 분야에서 샘플 복잡성을 획기적으로 개선하는 결과를 보여줍니다.



### Knowledge Discovery using Unsupervised Cognition (https://arxiv.org/abs/2409.20064)
- **What's New**: 이 논문에서는 Unsupervised Cognition 모델을 기반으로 한 지식 발견을 위한 세 가지 기법을 제안합니다. 특히 패턴 마이닝(pattern mining), 특성 선택(feature selection), 차원 축소(dimensionality reduction) 기법을 소개하고, 이들을 통해 중요한 패턴을 추출할 수 있는 방법을 제시합니다.

- **Technical Details**: 제안된 패턴 마이닝 기법은 기존의 Unsupervised Cognition 모델에서 훈련된 대표들을 기반으로 하여 데이터의 패턴을 선택하는 방식입니다. 특성 선택 기법은 타겟 특성과의 상관관계를 기반으로 하여 패턴에서 중요 특성을 선정합니다. 차원 축소 기법은 선택된 특성을 기준으로 데이터셋을 줄이는 작업을 수행합니다. 이 과정은 지식 발견 파이프라인을 통해 구현됩니다.

- **Performance Highlights**: 실험 결과, 제안된 기법들은 최신 기술과 비교하여 우수한 성능을 나타냈으며, 최종적으로 Unsupervised Cognition 모델의 정확도를 높일 수 있음을 보여주었습니다. 다양한 분야에서 지식 발견을 통해 의사 결정 및 데이터 분석에 기여할 수 있는 잠재력을 지니고 있습니다.



### Model Selection with a Shapelet-based Distance Measure for Multi-source Transfer Learning in Time Series Classification (https://arxiv.org/abs/2409.20005)
Comments:
          Accepted at International Conference on Pattern Recognition 2024 (ICPR 2024)

- **What's New**: 이 논문에서는 시계열 분류를 위한 다중 데이터셋을 사용하는 새로운 전이 학습 방법을 제안합니다. 특히, 여러 데이터셋을 하나의 소스 데이터셋으로 결합하여 신경망을 사전 학습(pre-training)합니다.

- **Technical Details**: 제안된 방법은 shapelet discovery를 기반으로 데이터셋의 전이 가능성(transferability)을 측정하여 효과적인 소스 선택을 지원합니다. 기존의 전이 가능성 측정 방법은 여러 아키텍처에 대해 모든 소스에 대해 사전 학습을 수행해야 하며 시간이 많이 소요되는 반면, 우리의 방법은 단일 계산으로 모든 가능한 아키텍처에 활용될 수 있습니다.

- **Performance Highlights**: 제안된 방법을 통해 시계열 데이터셋에서 Temporal Convolutional Neural Networks (CNN)의 성능을 향상할 수 있음을 입증하였습니다. 연구 결과는 2018 UCR Time Series Archive의 128개 시계열 데이터셋에서 평가되었습니다.



### Knowledge Graph Embedding by Normalizing Flows (https://arxiv.org/abs/2409.19977)
- **What's New**: 이 논문은 지식 그래프 임베딩(Knowledge Graph Embedding, KGE)에 대한 새로운 관점을 제시하며, 군론(group theory) 관점에서 불확실성을 도입합니다. 중요한 개념은 엔티티와 관계를 대칭군(symmetric group)의 원소로 임베딩하는 것입니다. 이로 인해 기존 모델들을 포함할 수 있는 일반성(generality), 계산 효율성(efficiency), 복잡한 확률 변수의 표현력(expressiveness)을 보장합니다.

- **Technical Details**: 제안된 모델은 엔티티와 관계를 무작위 변수의 집합의 순열(permutations)로 나타냅니다. 우리는 '정규화 흐름(normalizing flow)'을 사용하여 간단한 무작위 변수를 복잡한 무작위 변수로 변환할 수 있습니다. 또한 두 개의 정규화 흐름의 유사성을 측정하여 점수를 부여하는 함수를 정의했습니다(Normalizing Flows Embedding, NFE).

- **Performance Highlights**: 모델의 실험 결과는 KGE에 불확실성을 도입하는 것이 효과적임을 입증하였으며, NFE는 논리 규칙(logical rules)을 학습할 수 있음을 입증했습니다. 그 과정에서 KGE의 기존 임베딩 모델의 일반화된 형태로 작용하고, 쉽게 계산될 수 있는 그룹 연산의 이점을 활용합니다.



### Learning Partial Differential Equations with Deep Parallel Neural Operators (https://arxiv.org/abs/2409.19976)
- **What's New**: 본 연구에서는 깊은 병렬 연산자 모델(Deep Parallel Neural Operator, DPNO)을 제안하여 복잡한 부분 미분 방정식(PDE)의 해결을 위한 새로운 접근법을 제시합니다. 이 모델은 기존의 단일 연산자 아키텍처의 한계를 극복하여 여러 잠재 공간에서 병렬적으로 여러 연산자를 학습합니다.

- **Technical Details**: DPNO는 CNN(Convolutional Neural Network)을 사용하여 지역 기능을 추출하고 데이터를 각기 다른 잠재 공간으로 매핑합니다. 독특한 설계의 병렬 블록을 통해 이터레이션 오류 문제를 해결하며, 여러 연산자를 병렬 블록에서 학습함으로써 입력과 출력 간의 복잡한 매핑을 근사합니다. 구체적으로 Fast Fourier Transform(FFT)을 활용하여 PDE를 학습합니다.

- **Performance Highlights**: DPNO는 5개의 데이터셋에서 평균 10.5%의 성능 향상을 이루었으며, 한 데이터셋에서는 두 번째 성과를 기록했습니다. 병렬 블록과 직렬 블록을 비교한 결과, 병렬 블록은 평균 9.3%의 성능 향상을 보였습니다.



### Exploiting Adjacent Similarity in Multi-Armed Bandit Tasks via Transfer of Reward Samples (https://arxiv.org/abs/2409.19975)
- **What's New**: 본 논문에서는 연속적 다중 작업 문제를 다루며, 각 작업이 K개의 팔(pal)을 가진 확률적 다중 무장 밴딧(stochastic multi-armed bandit)으로 모델링됩니다. 작업 간의 유사성을 고려하여, 이전 작업에서의 보상 샘플을 현재 작업으로 전이하여 전체적 후회(regret)를 줄이는 두 가지 알고리즘을 제안합니다.

- **Technical Details**: 우리는 UCB (Upper Confidence Bound) 알고리즘을 기반으로 하여, 두 가지 알고리즘 Tr-UCB와 Tr-UCB2를 통해 정보 전이를 생성합니다. Tr-UCB는 알려진 유사성 파라미터(ϵ)를 가정하고, Tr-UCB2는 알려지지 않은 경우에도 적용될 수 있도록 확장됩니다. 알고리즘은 보상 샘플 전이를 통해 새로운 작업의 성능을 향상시키도록 설계되었습니다.

- **Performance Highlights**: Empirical results show that the proposed algorithms, Tr-UCB and Tr-UCB2, significantly reduce regret compared to the standard UCB algorithm and naive transfer methods. 특히, 학습 과정에서 각 작업의 유사성을 최대한 활용하여 전이된 정보가 성능 향상에 기여하는 것을 입증하였습니다.



### Task-agnostic Pre-training and Task-guided Fine-tuning for Versatile Diffusion Planner (https://arxiv.org/abs/2409.19949)
- **What's New**: 이 논문에서는 다양한 작업에 적용할 수 있는 다목적 Diffusion Planner인 	extbf{SODP}를 개발하였습니다. 기존의 다중 작업 계획자나 정책들은 일반적으로 작업별 demonstrational에 의존하거나 각 작업에 대해 보상이 필요했지만, SODP는 비 특정 작업의 저품질 데이터로부터 학습하여 특정 작업에 신속하게 적응하는 능력을 제공합니다.

- **Technical Details**: SODP는 두 단계로 구성된 프레임워크입니다. 첫 번째 단계는 사전 훈련(pre-training)이며, 여기서 다양한 작업의 경로를 모델링하여 기본적인 계획 능력을 추출합니다. 두 번째 단계는 강화 학습(RL)을 기반으로 하는 미세 조정(fine-tuning)으로, 특정 작업에 맞는 보상을 사용하여 Diffusion Planner를 정교화합니다. 이 과정에서 정책 기울기(Policy Gradient)를 적용하여 행동 시퀀스를 최적화합니다.

- **Performance Highlights**: 실험 결과, SODP는 Meta-World 및 Adroit과 같은 다중 작업 도메인에서 최신 방법들보다 월등한 성능을 보였습니다. 특히, 소량의 데이터로도 보상 기반 미세 조정이 가능하여 높은 작업 특정 수익을 달성할 수 있음을 입증하였습니다.



### Classification with a Network of Partially Informative Agents: Enabling Wise Crowds from Individually Myopic Classifiers (https://arxiv.org/abs/2409.19947)
Comments:
          12 pages, 15 figures, 60th Annual Allerton Conference on Communication, Control, and Computing

- **What's New**: 이 논문은 동질적이지 않은 에이전트들이 포함된 피어 투 피어 네트워크를 통한 분산 분류 문제를 다루고 있습니다. 각 에이전트는 자신이 수신하는 로컬 데이터를 기반으로 제한된 클래스만을 구별할 수 있는 분류기를 보유하고 있으며, 이를 통해 네트워크 전체의 진정한 클래스를 식별하는 새로운 방법론을 제안합니다.

- **Technical Details**: 제안된 알고리즘은 각 에이전트의 로컬 분류기에서 제공된 사후 확률(posterior probabilities)을 사용하여 반복적으로 각 에이전트의 신념(belief)을 업데이트합니다. 이 과정에서 새로운 분산 미니멈 규칙을 적용하여 에이전트들이 각자의 글로벌 신념을 업데이트하며 모든 에이전트가 진정한 클래스를 학습할 수 있도록 합니다.

- **Performance Highlights**: 시뮬레이션을 통해 제안된 알고리즘은 다른 집계 규칙들, 즉 평균이나 최대치 최대 집계 방식보다 뛰어난 성능을 발휘함을 보여주었습니다. 특정 가정 하에서 진정한 클래스에 대한 신념이 거의 확실히 수렴하게 됨을 시뮬레이션으로 입증하였습니다.



### Positive-Sum Fairness: Leveraging Demographic Attributes to Achieve Fair AI Outcomes Without Sacrificing Group Gains (https://arxiv.org/abs/2409.19940)
- **What's New**: 이번 연구에서는 의료 AI의 공정성에 대한 새로운 개념인 positive-sum fairness를 도입하였습니다. 이는 성능의 향상이 집단 간 격차를 확대하더라도, 특정 하위 그룹의 성능 저하가 없다면 수용할 수 있다는 것입니다.

- **Technical Details**: positive-sum fairness는 집단 간 성능 차이를 해로운 것과 무해한 것으로 구분하는 평가 프레임워크입니다. 이 프레임워크를 통해 모델의 성능 향상 과정에서의 공정성을 분석하고, 모든 하위 집단이 더 낫지 않더라도 전체 성능이 향상될 수 있도록 하는 솔루션을 찾고자 합니다.

- **Performance Highlights**: CNN 모델을 비교한 결과, 인구 통계적 인코딩을 제거하면 하위 그룹 간 성능 차이를 줄일 수 있었으며, 인종 속성을 모델 입력으로 활용했을 때 전체 성능은 증가하였지만 하위 그룹 간 격차가 확대됨을 보였습니다. 이는 긍정적 공정성 개념의 관점에서 유익한 성능 개선을 달성할 수 있음을 보여줍니다.



### Scaling Optimal LR Across Token Horizon (https://arxiv.org/abs/2409.19913)
- **What's New**: 이번 연구는 대규모 언어 모델(LLM) 훈련에서 학습률(learning rate)과 토큰 수(token horizon) 간의 연관성에 대한 대규모 실증 연구를 수행하였습니다. 특히, 토큰 수가 많아질수록 최적의 학습률이 감소하는 경향이 있음을 확인하였습니다.

- **Technical Details**: 연구진은 Megatron 코드베이스를 사용하여 여러 LLM 모델의 학습률과 토큰 수 간의 관계를 살펴보았습니다. 실험 결과, 긴 토큰 수의 경우, 작은 학습률이 필요하며, 최적 학습률은 스케일링 법칙(scaling law)을 따릅니다. 이로써 짧은 토큰 수에서 얻은 최적 학습률을 바탕으로 긴 토큰 수에서의 최적 학습률을 추정할 수 있습니다.

- **Performance Highlights**: LLama-1 모델이 너무 높은 학습률을 사용했으며, 이는 성능 저하를 초래한다는 증거를 제공했습니다. 토큰 수에 따른 학습률 전이 방법론을 개발하여 현재의 관행에 부가적인 부담 없이 적용할 수 있도록 하였습니다.



### HYDRA-FL: Hybrid Knowledge Distillation for Robust and Accurate Federated Learning (https://arxiv.org/abs/2409.19912)
- **What's New**: 본 논문에서는 Knowledge Distillation (KD) 기반의 Federated Learning (FL) 시스템이 model poisoning 공격에 대해 고유한 취약성을 가지고 있음을 밝혀내고, 이를 줄이기 위한 Hybrid Knowledge Distillation for Robust and Accurate FL (HYDRA-FL) 알고리즘을 제안합니다.

- **Technical Details**: HYDRA-FL은 KD 손실을 얕은 레이어에서 보조 분류기를 통해 오프로드하여 공격 시나리오에서 공격의 영향을 줄이며, 두 가지 KD 기반 FL 알고리즘인 FedNTD와 MOON에 맞춰 조정할 수 있도록 일반적인 프레임워크로 설계되었습니다.

- **Performance Highlights**: HYDRA-FL은 공격 상황에서 기존의 FedNTD와 MOON보다 높은 정확도를 달성하며, 일반적인 환경에서도 유사한 성능을 유지하는 것으로 나타났습니다.



### SurvCORN: Survival Analysis with Conditional Ordinal Ranking Neural Network (https://arxiv.org/abs/2409.19901)
- **What's New**: 이 논문에서는 SurvCORN이라는 새로운 방법을 제시하며, 이는 조건부 순위 네트워크(Conditional Ordinal Ranking Networks)를 활용하여 생존 곡선(survival curves)을 직접 예측합니다. 또한 SurvMAE라는 새로운 메트릭(metrics)을 도입하여 생존 예측의 정확성을 평가합니다.

- **Technical Details**: SurvCORN은 생존 예측을 위한 딥 뉴럴 네트워크로, 시간 축을 K개의 이산(intervals) 간격으로 나누고 각 시간 간격을 초과하는 생존 확률을 로지스틱 회귀(logistic regression)를 통해 예측합니다. 이 네트워크는 오른쪽 검열(right censoring)된 데이터의 처리 방법과 함께 uncensored와 censored 환자를 구분하여 예측을 수행합니다.

- **Performance Highlights**: 실증적 평가를 통해 SurvCORN은 환자 결과의 정확한 순서를 유지하며, 개별적인 시간-사건(time-to-event) 예측을 개선할 수 있음을 보여주고 있습니다. 이들은 최근의 순차 회귀(ordinal regression) 발전을 생존 분석에 확장한 기여로, 의료 환경에서의 정확한 예후(prognosis)에 중요한 인사이트를 제공합니다.



### RouterDC: Query-Based Router by Dual Contrastive Learning for Assembling Large Language Models (https://arxiv.org/abs/2409.19886)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 본 논문에서는 여러 개의 대형 언어 모델(LLM)을 조합하는 새로운 방법인 query-based Router by Dual Contrastive learning (RouterDC)을 제안합니다. 기존 모델이 여러 LLM이 잘 작동하는 경우 비효율적임을 개선합니다.

- **Technical Details**: RouterDC 모델은 encoder와 LLM 임베딩으로 구성되며, 두 가지 대조 학습 손실(contrastive learning losses)을 사용하여 모델을 훈련합니다. 이는 다양한 쿼리에 대해 가장 적합한 LLM을 선택하는 라우팅(routing) 기술을 활용합니다.

- **Performance Highlights**: 실험 결과, RouterDC는 개별 최고 성능 LLM과 기존 라우팅 방법보다 각각 +2.76% 및 +1.90% 더 우수한 성능을 보여주었습니다. 최적의 LLM 조합을 통해 효과적으로 성능을 향상시킴을 입증합니다.



### TSI: A Multi-View Representation Learning Approach for Time Series Forecasting (https://arxiv.org/abs/2409.19871)
Comments:
          AJCAI Oral Accepted

- **What's New**: 본 논문에서는 전통적인 시간 시계열 예측 모델의 한계를 극복하기 위해 트렌드(Trend)와 계절성(Seasonality) 표현을 독립 성분 분석(Independent Component Analysis, ICA)을 기반으로 통합한 새로운 멀티 뷰 접근 방식을 제안합니다.

- **Technical Details**: TSI 모델은 트렌드 및 계절성의 관점과 ICA의 관점을 결합하여 복잡하고 고차원적인 시계열 데이터를 분석합니다. 기존 방법들이 놓치는 비선형 관계를 포착할 수 있도록 설계되었습니다.

- **Performance Highlights**: 다양한 기준 데이터셋에서 TSI 모델은 현재 최첨단 모델들에 비해 뛰어난 성능을 보여주며, 특히 다변량(multi-variate) 예측에서 높은 정확도를 제공합니다. 이 방법은 시간 시계열 데이터에 대한 더 깊이 있는 이해를 제공하여 예측의 정확성을 향상시킵니다.



### Learning Multimodal Latent Generative Models with Energy-Based Prior (https://arxiv.org/abs/2409.19862)
Comments:
          The 18th European Conference on Computer Vision ECCV 2024

- **What's New**: 이번 논문에서는 에너지 기반 모델(EBM)과 다중 모달(latent generative) 생성 모델을 통합하는 새로운 프레임워크를 제안합니다. 기존의 Gaussian나 Laplacian 분포를 넘어서, 다양한 데이터 타입의 정보를 효과적으로 캡처할 수 있는 접근 방식을 제공합니다.

- **Technical Details**: 제안된 프레임워크는 variational scheme을 통해 다중 모달 생성 모델과 EBM을 공동으로 훈련할 수 있게 합니다. 이 접근 방식은 보다 표현력 있고 정보가 풍부한 prior를 생성하여 다중 모달 간의 정보를 더 잘 캡처합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델이 기존 모델보다 더 나은 생성 일관성을 보여주었으며, 다중 모달 생성 모델의 가능성을 확장하는 데 기여합니다.



### Counter-Current Learning: A Biologically Plausible Dual Network Approach for Deep Learning (https://arxiv.org/abs/2409.19841)
Comments:
          NeurIPS 2024

- **What's New**: 이번 논문에서는 생물학적 신경망의 학습 메커니즘을 모방한 새로운 학습 알고리즘인 counter-current learning (CCL)을 제안합니다. 이 알고리즘은 신경망에서의 신용 할당을 위한 생물학적으로 그럴듯한 프레임워크로, 피드포워드 네트워크와 피드백 네트워크를 활용하여 상호 작용하는 방식으로 작동합니다.

- **Technical Details**: CCL은 입력 데이터를 처리하는 피드포워드 네트워크와 목표를 처리하는 피드백 네트워크를 결합합니다. 두 네트워크는 안티-패럴렐 신호 전파(anti-parallel signal propagation)를 통해 서로를 강화하며, 피드백 네트워크의 하위 계층에서 더 많은 정보를 얻어 이를 피드포워드 네트워크의 상위 계층 업데이트에 활용합니다.

- **Performance Highlights**: MNIST, FashionMNIST, CIFAR10 및 CIFAR100 데이터셋에서 실행된 실험 결과, CCL은 다른 생물학적으로 그럴듯한 알고리즘들과 유사한 성능을 유지하며, 더 생물학적으로 현실적인 학습 메커니즘을 제공합니다. 또한, 오토인코더(encoder) 작업에 대한 적용 가능성을 보여줍니다.



### ForecastBench: A Dynamic Benchmark of AI Forecasting Capabilities (https://arxiv.org/abs/2409.19839)
- **What's New**: 새로운 연구는 ForecastBench라는 동적 benchmark를 도입하여 머신러닝 (ML) 시스템의 예측 정확성을 평가할 수 있는 표준화를 제공한다.

- **Technical Details**: ForecastBench는 자동으로 생성되고 정기적으로 업데이트되는 1,000개의 예측 질문 세트로 구성되어 있으며, 데이터 유출 가능성을 방지하기 위해 제출 시점에 알려지지 않은 미래 사건에 대한 질문만 포함한다. 현재 ML 시스템의 능력을 평가하기 위해, 전문가(인간) 예측자, 일반 대중, LLMs로부터 benchmark에서 무작위 선택된 질문(N = 200)에 대한 예측을 수집하였다.

- **Performance Highlights**: 전문가 예측자는 상위 성능의 LLM보다 더 높은 예측 정확성을 보였으며(p-values <= 0.01), 이에 대한 결과는 공개 리더보드에서 확인할 수 있다.



### geom2vec: pretrained GNNs as geometric featurizers for conformational dynamics (https://arxiv.org/abs/2409.19838)
Comments:
          12 pages, 8 figures, supporting information appended

- **What's New**: 이번 연구에서는 사전 훈련된 그래프 신경망(GNNs)을 활용하여 분자 시뮬레이션의 동적 특징을 효과적으로 식별할 수 있는 geom2vec를 소개합니다. 이는 기존의 수동 조정이 필요하지 않으며, 대규모 데이터셋을 기반으로 전이 가능한 구조 표현을 학습합니다.

- **Technical Details**: geom2vec는 자가 지도 학습(self-supervised learning) 방식의 노이즈 제거(denoising) 목표로 크게 확대된 분자 정보 데이터셋에 대해 동등한(equivariant) GNN을 미리 훈련시켜, 다차원 분자 구조의 기하학적 패턴을 포착하는 일반적인 기법으로 작용합니다.

- **Performance Highlights**: 학습된 표현은 직접적으로 궤적 데이터(trajectory data)를 분석하는 데 사용될 수 있으며, 이는 수동적인 특징 선택(manual feature selection)의 필요성을 줄이고 시뮬레이션 분석 워크플로우의 견고성을 향상시킵니다. 또한, GNN 훈련을 다운스트림 작업(training for downstream tasks) 훈련과 분리함으로써 제한된 계산 자원으로도 더 큰 분자 그래프를 분석할 수 있게 합니다.



### Calibrating Language Models with Adaptive Temperature Scaling (https://arxiv.org/abs/2409.19817)
Comments:
          EMNLP 2024

- **What's New**: 본 논문에서는 Adaptive Temperature Scaling (ATS)이라는 새로운 후처리 방법을 소개합니다. 이는 각 토큰 예측에 대해 온도 스케일링 매개변수를 예측하여 모델의 신뢰도를 개선합니다.

- **Technical Details**: ATS는 토큰 수준의 특성에 따라 조정되는 온도 값을 예측하며, 이는 표준 감독된 미세 조정(supervised fine-tuning) 데이터셋에 맞춰진 것입니다. 이 방법은 강화 학습(reinforcement learning)에서 인간 피드백(human feedback)을 사용하는 후 미세 조정 이후에 발생하는 보정(calibration) 변화에 적응합니다.

- **Performance Highlights**: ATS는 세 가지 다운스트림 자연어 평가 벤치마크에서 이전 보정 방법에 비해 10-50% 이상의 보정 개선을 달성하였으며, RLHF로 인한 성능 향상에는 방해가 되지 않습니다.



### Differentially Private Bilevel Optimization (https://arxiv.org/abs/2409.19800)
Comments:
          29 pages

- **What's New**: 이 논문에서는 기계 학습 응용 프로그램에서 최근 많은 주목을 받고 있는 이층 최적화(bilevel optimization)에 대한 차분 개인정보 보호(differentially private, DP) 알고리즘을 소개합니다. 이 작업에 대해 원하는 개인정보 보호 수준을 제공할 수 있는 DP 알고리즘은 처음으로 제안되며, 대규모 설정에서 계산이 복잡한 Hessian 계산을 피할 수 있습니다.

- **Technical Details**: 제안된 그래디언트 기반 $(eta,	heta)$-DP 알고리즘은 상위 수준이 반드시 볼록(convex)이 아니고 하위 수준 문제가 강한 볼록성(strongly-convex)을 지닌 경우를 대상으로 하며, 반환되는 점의 초그래디언트(hypergradient) 노름은 최대 $	ilde{	ext{O}}igg(igg(rac{	ext{sqrt}(d_{	ext{up}})}{eta n}igg)^{1/2}+igg(rac{	ext{sqrt}(d_{	ext{low}})}{eta n}igg)^{1/3}igg)$입니다. 여기서 $n$은 데이터셋 크기, $d_{	ext{up}}/d_{	ext{low}}$는 각각 상위 및 하위 레벨 차원입니다. 이 분석은 제약(고정) 또는 비제약 문제에 적용 가능하며, 미니 배치(mini-batch) 그래디언트와 경험적(empirical) 및 모집단(population) 손실 모두에 적용됩니다.

- **Performance Highlights**: 이 DP 알고리즘은 대규모 데이터셋에서 효과적으로 작동하며, 다양한 최적화 문제에 적합한 강력한 성능을 입증합니다. 특히, 하위 수준이 강한 볼록성 조건을 만족할 때, 기존 방식에 비해 높은 개인정보 보호를 유지하면서도 효율적인 결과를 제공합니다.



### Membership Inference Attacks Cannot Prove that a Model Was Trained On Your Data (https://arxiv.org/abs/2409.19798)
- **What's New**: 본 논문은 데이터 생성자 또는 소유자가 특정 machine learning (ML) 모델이 그들의 데이터로 학습되었음을 제3자에게 입증하고자 하는 문제를 다룹니다. 최근 웹 규모의 데이터로 학습된 foundation model에 대한 소송에서 training data proof의 중요성이 대두되고 있습니다.

- **Technical Details**: 이전 연구들은 membership inference attack을 활용하여 training data proof를 제안해왔지만, 저자는 이 접근 방식이 근본적으로 안전하지 않다고 주장합니다. 저자는 데이터 생성자가 설득력 있는 증거를 제공하기 위해서는 공격의 false positive rate이 낮아야 하며, 이를 위해서는 모델이 대상 데이터로 학습되지 않았다는 null hypothesis 아래에서의 출력이 낮을 가능성을 검토해야 한다고 설명합니다. 하지만 null hypothesis 샘플링이 불가능하다는 문제를 지적합니다.

- **Performance Highlights**: 논문은 sound training data proofs를 생성할 수 있는 두 가지 접근 방식으로 data extraction attacks 그리고 특수한 canary data에 대한 membership inference를 제시합니다.



### Adaptive Event-triggered Reinforcement Learning Control for Complex Nonlinear Systems (https://arxiv.org/abs/2409.19769)
- **What's New**: 이 논문에서는 경계가 있는 불확실성(bounded uncertainties)과 복잡한 상호작용을 특성으로 하는 연속 시간 비선형 시스템에 대한 적응형 이벤트 기반 강화 학습 제어 방안을 제안합니다.

- **Technical Details**: 제안된 방법은 제어 정책(control policy)과 통신 정책(communication policy)을 동시에 학습할 수 있는 능력을 갖추고 있어, 이를 각각 또는 단독으로 학습할 때의 파라미터 수와 계산 오버헤드를 줄입니다. 또한, 전체 경로에 걸친 성능을 나타내는 보상(rewards)을 상태 공간에 추가하여 명시적인 트리거 조건(triggering conditions) 학습 필요 없이 정확하고 효율적으로 트리거 조건을 결정할 수 있음을 보여줍니다.

- **Performance Highlights**: 수치 예제를 통해 제안된 접근 방식의 효과성을 입증하였습니다.



### Balancing the Scales: A Comprehensive Study on Tackling Class Imbalance in Binary Classification (https://arxiv.org/abs/2409.19751)
Comments:
          13 pages including appendix, 4 tables

- **What's New**: 이 연구는 클래스 불균형 문제를 다루기 위한 세 가지 주요 전략(Synthetic Minority Over-sampling Technique (SMOTE), Class Weights 조정, Decision Threshold Calibration)의 효과를 포괄적으로 평가하였습니다. 15개의 머신 러닝 모델과 30개의 데이터셋을 이용해 9,000개의 실험을 수행한 결과, 모든 전략이 기본 모델보다 우수한 성과를 보였고, Decision Threshold Calibration이 가장 일관되게 효과적인 기법으로 나타났습니다.

- **Technical Details**: 이 연구는 F1-score를 주요 성능 지표로 삼고, F2-score, precision, recall, Brier-score, PR-AUC, AUC 등 9개의 추가 성능 지표를 추적했습니다. 30개 데이터셋(표본 크기 500~20,000, 희귀 클래스 비율 1~15%)과 15개 분류기 모델을 대상으로 5-fold cross-validation을 이용해 평가를 수행했습니다.

- **Performance Highlights**: 이 연구 결과는 클래스 불균형 데이터셋을 처리하는 방법에 따라 성능이 크게 달라질 수 있음을 보여주었습니다. 연구자들은 각 기술이 데이터셋에 따라 어떻게 작동하는지를 강조하며, 다양한 접근 방식을 시험하는 것의 중요성을 강조했습니다.



### Tailored Federated Learning: Leveraging Direction Regulation & Knowledge Distillation (https://arxiv.org/abs/2409.19741)
- **What's New**: 이번 논문에서는 Federated Learning (FL)에서의 데이터 이질성과 다양한 클라이언트의 등을 최적화하기 위한 새로운 알고리즘을 제안했습니다. 이 알고리즘은 모델 델타 정규화(model delta regularization), 개인화된 모델(personalized models), 연합 지식 증류(federated knowledge distillation), 믹스 풀링(mix-pooling)을 통합하여 효과적으로_FL의 성능을 개선합니다.

- **Technical Details**: 제안된 모델 델타 정규화 기법은 서버에서 전역 모델 가중치의 최적화 방향을 계산하고, L2 정규화 항을 도입하여 로컬 모델 업데이트를 전역 모델과 정렬시킵니다. 이를 통해 데이터 통신 비용을 현저히 줄일 수 있습니다. 개인화된 모델과 연합 지식 증류는 데이터와 작업의 이질성이 극심한 환경에서 사용되며, 믹스 풀링은 클라이언트의 다양한 요구를 수용할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 모델 델타 정규화 기법을 통해 정확성과 빠른 수렴 속도를 확인할 수 있었습니다. 또한, 연합 지식 증류 알고리즘은 특히 다양한 데이터 상황에서 FL 성능을 크게 향상시켰고, 믹스 풀링의 도입은 특정 클라이언트에 실질적인 혜택을 제공함을 보여주었습니다.



### When Molecular GAN Meets Byte-Pair Encoding (https://arxiv.org/abs/2409.19740)
- **What's New**: 본 연구에서는 byte-level byte-pair encoding (BPE) tokenization 방법을 통합한 분자 GAN을 소개합니다. 이 방법은 기존의 문자 기반 토크나이저들의 한계를 극복하고, 기존에 존재하는 분자의 복잡한 하위 구조를 더 잘 인식합니다. 또한 강화 학습(Reinforcement Learning, RL)을 이용하여 de novo 분자 생성을 향상시킵니다.

- **Technical Details**: 제안된 모델은 생성기(generator)와 판별기(discriminator)로 구성된 GAN 프레임워크를 사용합니다. 생성기는 SMILES 문자열을 생성하는 역할을 하고, LSTM(장기 단기 기억) 모델로 구현됩니다. 판별기는 생성된 SMILES 문자열의 품질을 평가하는 역할을 하며, bidirectional LSTM으로 구현됩니다. 이 모델은 tokenization을 위해 BPE 방식을 사용하고 있으며, 새로운 보상 메커니즘도 통합하여 계산 효율성을 향상시킵니다.

- **Performance Highlights**: 실험 결과, 유효성(validity), 독창성(uniqueness), 새로움(novelty), 다양성(diversity) 측면에서의 평가가 이루어졌고, 비주얼리제이션 분석을 통해 GAN의 효과성을 입증하였습니다. 이 연구는 분자 생성에서 성능과 신뢰성을 극대화하기 위한 혁신적인 접근 방식을 제안했습니다.



### Unified Gradient-Based Machine Unlearning with Remain Geometry Enhancemen (https://arxiv.org/abs/2409.19732)
Comments:
          Accepted by NeurIPS 2024 as a Spotlight paper

- **What's New**: 본 논문에서는 머신 언러닝(Machine Unlearning, MU)의 새로운 접근 방식을 제안하여 기계 학습 모델의 개인 정보 보호 및 신뢰성을 강화하는 방법을 연구하였습니다. 특히 대규모 모델을 위한 근사 MU(Approximate MU) 기법을 집중적으로 조명하고 있습니다.

- **Technical Details**: 본 연구는 매개변수 이웃 내에서의 정확한 MU와의 Kullback-Leibler divergence 최소화를 통해 가장 가파른 하강 방향을 찾는 방법으로 시작됩니다. 이 방향은 가중치 잊기 그래디언트 상승(weighted forgetting gradient ascent), 나머지 세트를 유지하기 위한 미세 조정 그래디언트 하강(fine-tuning retaining gradient descent), 그리고 가중치 비중 행렬(weight saliency matrix)으로 분해됩니다. 이 구성은 기존의 그래디언트 기반 MU 방법들을 통합하는 관점을 제공합니다. 또한, 남은 데이터의 두 번째 도함수(Hessian)를 포함하여, 더 효율적인 잊기 방향을 학습하는 빠른-느린 매개변수 업데이트(fast-slow parameter update) 전략을 제안합니다.

- **Performance Highlights**: Extensive experiments demonstrate that our method achieves class-forgetting on ImageNet using DiT and effectively forgets a class on CIFAR-10 using DDPM in only 50 steps, markedly outperforming prior methods that required thousands of steps.



### Investigating the Effect of Network Pruning on Performance and Interpretability (https://arxiv.org/abs/2409.19727)
Comments:
          4 pages, 6 figures

- **What's New**: 이번 연구에서는 GoogLeNet에 대한 다양한 pruning 기법(비구조화, 구조화, 연결 희소성)의 영향을 분석하고, 모델의 분류 성능 및 해석 가능성에 대한 결과를 제시하고 있습니다. 특히, 연결 희소성(Connection Sparsity) 방법을 통해 모델의 80%를 pruning하였음에도 불구하고 Top-1 정확도를 향상시키는 성과를 거두었습니다.

- **Technical Details**: 연구에서는 unstructured pruning, structured pruning, connection sparsity 같은 다양한 pruning 기법을 적용해 GoogLeNet의 성능을 평가하였습니다. 이 과정에서 iterative pruning과 one-shot pruning의 장단점을 비교하였으며, Connection Sparsity 방법을 통해 입력 채널을 pruning하여 모델의 구조적 무결성을 유지하면서 연산 속도를 최적화하는 방식으로 이루어졌습니다. 또한, Mechanistic Interpretability Score (MIS)를 사용하여 해석 가능성을 측정했습니다.

- **Performance Highlights**: 연구 결과, Connection Sparsity 방법을 적용한 GoogLeNet은 80%의 파라미터를 pruning한 후에도 Top-1 정확도가 0.2% 향상되었습니다. iterative pruning이 one-shot pruning보다 성능 유지에 더 유리하다는 결과를 바탕으로, retraining이 필요했으며, 50 epoch 이상의 retraining이 필요함을 보여주었습니다. 최종적으로, 구조화된 형태의 pruning 기법이 정확도 보존에 효과적임을 입증하였습니다.



### DataDRILL: Formation Pressure Prediction and Kick Detection for Drilling Rigs (https://arxiv.org/abs/2409.19724)
- **What's New**: 이 논문에서는 석유 및 가스 시추 연구를 지원하기 위해 형성 압력 예측과 킥 감지를 위한 두 가지 새로운 데이터셋을 소개합니다. 이 데이터셋은 28개의 드릴링 변수와 2000개 이상의 데이터 샘플을 포함하고 있으며, AI 알고리즘 개발에 기여할 것입니다.

- **Technical Details**: 주요 기술적 세부 사항으로는 주성분 회귀(Principal Component Regression)를 사용하여 형성 압력을 예측하고, 주성분 분석(Principal Component Analysis)을 활용하여 킥을 식별합니다. 데이터셋의 기술적 검증을 위한 R2 점수는 0.78이며, Residual Predictive Deviation 점수는 0.922입니다.

- **Performance Highlights**: 형성 압력 예측과 킥 감지를 위한 새로운 데이터셋은 학계와 업계의 AI 모델 개발에 유의미한 기여를 할 것으로 기대되며, 특히 종래의 소규모 데이터셋 문제를 해결함으로써 예측 성능 향상을 도모할 수 있습니다.



### Evolving Multi-Scale Normalization for Time Series Forecasting under Distribution Shifts (https://arxiv.org/abs/2409.19718)
- **What's New**: 이번 연구에서는 복잡한 분포 변화(distribution shifts) 문제를 해결하기 위해 Evolving Multi-Scale Normalization (EvoMSN) 프레임워크를 소개합니다. 이 프레임워크는 시간 시계열 예측의 정확성을 높이는 데 초점을 맞추고 있으며, 특히 다양한 스케일에서의 분포 동역학을 모델링하는 데 필요한 새로운 접근 방식을 제공합니다.

- **Technical Details**: EvoMSN은 입력 시퀀스를 주기성 특성에 따라 다양한 크기의 조각으로 나누어 다중 스케일의 통계 정보에 따라 정규화를 수행합니다. 이를 통해 백본 예측 모델은 정규화된 시리즈를 처리하여 다수의 출력을 생성할 수 있습니다. 또한, 다중 스케일 통계 예측 모듈을 통해 통계의 동역학을 포착하고 향후 조각의 분포를 예측합니다. 제안된 방법은 변화하는 분포를 추적하기 위해 예측 모듈과 예측 모델을 온라인으로 협력하여 업데이트하는 진화적 최적화 전략을 활용합니다.

- **Performance Highlights**: EvoMSN은 5가지 주요 예측 방법에 대해 벤치마크 데이터셋에서 성능을 향상시키는 데 효과적임을 입증했습니다. 기존의 고급 정규화 및 온라인 학습 접근 방식과 비교할 때, EvoMSN은 탁월한 성능을 보여주었습니다.



### Constrained Reinforcement Learning for Safe Heat Pump Contro (https://arxiv.org/abs/2409.19716)
- **What's New**: 이 논문은 에너지 효율성과 거주자의 열 쾌적성을 동시에 최적화하기 위한 새로운 건물 시뮬레이터 I4B를 제안합니다. 이 시뮬레이터는 다양한 용도로 사용할 수 있는 인터페이스를 제공하며, 제약이 있는 모델 프리 강화 학습 알고리즘인 CSAC-LB를 활용하여 난방 최적화 문제를 해결합니다.

- **Technical Details**: I4B는 건물 시뮬레이션 모듈과 제어 알고리즘 간의 인터페이스를 생성하며, 참조 제어기(reference controllers)와 병렬 처리(parallelization) 지원, 알고리즘 평가를 위한 표준화된 메트릭(metrics)을 포함합니다. 제약 마코프 결정 과정(Constrained Markov Decision Process, CMDP)으로 난방 제어 문제를 개념화하고, CSAC-LB 알고리즘을 사용하여 다양한 시나리오에서 성능을 평가합니다.

- **Performance Highlights**: CSAC-LB는 데이터 탐색(data exploration)과 제약 조건 만족(constraint satisfaction) 측면에서 우수한 성능을 보이며, 다른 최신 알고리즘(SOTA)과 비교하여 목표와 제약의 균형을 잘 맞출 수 있음을 보여줍니다.



### Generating peak-aware pseudo-measurements for low-voltage feeders using metadata of distribution system operators (https://arxiv.org/abs/2409.19713)
Comments:
          17 pages, 9 figures, 8 tables

- **What's New**: 이번 논문은 배급 시스템 운영자(DSO)가 비측정 저전압(LV) 피더에 대한 의사 측정치를 추정하는 새로운 접근 방식을 제안합니다. 이 접근법은 피더 메타데이터를 기반으로 하여 섭씨스럽고 통계적으로 중요한 측정치를 생성합니다.

- **Technical Details**: 제안된 방법은 피더 메타데이터를 활용하여 회귀 모델을 통해 비측정 피더의 의사 측정치를 추정합니다. 사용된 회귀 모델로는 XGBoost, 다층 퍼셉트론(MLP), 선형 회귀(LR)가 있으며, 기존 측정치를 모델의 목표 값으로 사용합니다. 기상 데이터 및 행사 데이터도 모델 특징으로 사용됩니다.

- **Performance Highlights**: 실제 데이터셋을 통해 평가한 결과, XGBoost와 MLP가 LR보다 뛰어난 성능을 보여주었습니다. 의사 측정치는 날씨, 일정 및 타임스탬프 조건에 적응하며, 피더 메타데이터에 기반한 신뢰할 수 있는 부하 곡선을 생성할 수 있습니다.



### Vision-Language Models are Strong Noisy Label Detectors (https://arxiv.org/abs/2409.19696)
Comments:
          Accepted at NeurIPS 2024

- **What's New**: 본 논문은 Denoising Fine-Tuning (DeFT) 프레임워크를 제안하여 비전-언어 모델을 조정하는 방법을 소개합니다. 이 방법은 수천만 개의 보조 이미지-텍스트 쌍에서 미리 학습된 텍스트 및 시각적 피처의 강력한 정렬을 활용하여 노이즈가 있는 레이블을 제거하는 데 중점을 둡니다.

- **Technical Details**: DeFT는 각 클래스에 대해 긍정적 및 부정적 텍스트 프롬프트를 학습하여 노이즈 레이블 탐지기를 구축합니다. 긍정적 프롬프트는 클래스의 독특한 특징을 드러내고, 부정적 프롬프트는 깨끗한 샘플과 노이즈 샘플을 구분하는 학습 가능한 임계값 역할을 합니다. 이를 통해 모델은 이미지와 텍스트 프롬프트 간의 유사도를 활용하여 노이즈 레이블을 식별합니다. 최적화를 위해 Visual Prompt Tuning (VPT) 기술을 사용합니다.

- **Performance Highlights**: DeFT는 7개의 합성 및 실제 노이즈 데이터 세트에서 실험을 통해 노이즈 레이블 탐지와 이미지 분류 작업 모두에서 뛰어난 성능을 입증하였습니다. 특히, 정교한 분류 작업에서 개선된 성능을 보였습니다.



### Machine Learning for Raman Spectroscopy-based Cyber-Marine Fish Biochemical Composition Analysis (https://arxiv.org/abs/2409.19688)
- **What's New**: 이 논문은 래만 분광법(Raman spectroscopy)을 활용하여 물고기의 생화학 조성을 비파괴적으로 분석하고, 물, 단백질 및 지방 산출량을 공동으로 예측하기 위해 새로운 CNN(Convolutional Neural Networks) 모델을 제안합니다. 또한, 극소량의 데이터셋으로 CNN을 적용한 최초의 연구로, 데이터 부족의 문제를 해결하기 위한 프레임워크인 FishCNN을 개발하였습니다.

- **Technical Details**: FishCNN 프레임워크는 데이터 전처리, 데이터 증강(data augmentation) 및 스케일링 기법을 통합하여 작은 실세계 분광 데이터셋에 적용합니다. 이 과정에서 FT-Raman 및 InGaAs 1064 nm 스펙트로스코픽 데이터를 교차 검증을 위해 6 개의 폴드로 나누고, 다양한 전처리 기법을 적용하여 데이터 특성을 정제합니다. CNN 모델은 큰 필터 크기와 작은 스트라이드를 사용하며, 데이터 전처리 후 증강을 수행하여 신뢰성 있는 학습을 보장합니다.

- **Performance Highlights**: FishCNN 모델은 전통적인 기계 학습 모델들과 두 개의 최신 CNN 모델을 비교했을 때, 과적합(overfitting)을 줄이고 예측 정확도를 크게 향상시키는 성능을 보여주었습니다. 이러한 결과는 물고기 생화학 조성 분석의 정확하고 자동화된 접근 방식의 가능성을 열어줍니다.



### Temporal Source Recovery for Time-Series Source-Free Unsupervised Domain Adaptation (https://arxiv.org/abs/2409.19635)
- **What's New**: 본 논문에서는 Temporal Source Recovery (TemSR)라는 새로운 프레임워크를 제안하며, 이는 Time-Series Source-Free Unsupervised Domain Adaptation (TS-SFUDA)에서 중요한 시간적 종속성을 효과적으로 전이할 수 있도록 설계되었습니다. 기존의 방법들은 특정한 소스 도메인 설계에 의존하고 있지만, TemSR은 소스 데이터에 대한 액세스 없이도 우수한 결과를 도출할 수 있습니다.

- **Technical Details**: TemSR은 마스킹(masking), 복구(recovery), 최적화(optimization)를 포함하는 복구 프로세스를 통해 소스와 유사한 분포(source-like distribution)를 생성하며, 이를 위해 세그먼트 기반 정규화(segment-based regularization)와 앵커 기반 회복 다양성 극대화(anchor-based recovery diversity maximization) 기법을 활용합니다. 이러한 기법들은 시간적 종속성을 복구하고, 로컬 종속성을 회복하는 데 중요합니다.

- **Performance Highlights**: 다양한 TS 작업에서의 광범위한 실험을 통해 TemSR의 효과가 입증되었으며, 기존 TS-SFUDA 방법을 초월하는 성과를 보였습니다. 또한 소스, 소스 유사, 타겟 도메인 간의 분포 불일치(discrepancy) 변화에 대한 분석을 통해 TemSR이 효과적인 소스 유사 도메인을 복구하고, 실제로 소스 데이터에 접근하지 않고도 도메인 간의 격차를 줄일 수 있음을 확인했습니다.



### A Survey on Graph Neural Networks for Remaining Useful Life Prediction: Methodologies, Evaluation and Future Trends (https://arxiv.org/abs/2409.19629)
- **What's New**: 이 논문은 Remaining Useful Life (RUL) 예측을 위한 Graph Neural Networks (GNNs)의 사용을 종합적으로 검토합니다. GNNs는 복잡한 시스템에서 공간 정보를 모델링하는 데 있어 효과적인 방법을 제공합니다.

- **Technical Details**: 새로운 분류체계(Taxonomy)를 제안하여 GNN을 RUL 예측에 적합하게 조정하는 과정의 네 가지 주요 단계를 정의합니다: 그래프 구성(Graph Construction), 그래프 모델링(Graph Modeling), 그래프 정보 처리(Graph Information Processing), 그래프 리드아웃(Graph Readout).

- **Performance Highlights**: 다양한 최신 GNN 방법론에 대한 철저한 평가는 연구자들에게 유용한 벤치마크를 제공하며, GNNs가 RUL 예측을 개선할 수 있는 가능성을 강조합니다.



### DropEdge not Foolproof: Effective Augmentation Method for Signed Graph Neural Networks (https://arxiv.org/abs/2409.19620)
Comments:
          NeurIPS 2024

- **What's New**: 본 논문에서는 친근하거나 적대적 관계를 나타내는 Signed Graphs의 링크 서명 예측(link sign prediction) 작업을 다룹니다. 기존의 Signed Graph Neural Networks(SGNNs)의 한계인 그래프 희소성(sparsity) 및 불균형 삼각형(unbalanced triangles) 문제를 해결하기 위해 데이터 증강(data augmentation) 기법을 제안합니다. 특히, 기존의 DropEdge 방법의 한계를 지적하며, 새로운 Signed Graph Augmentation(SGA) 프레임워크를 소개합니다.

- **Technical Details**: SGA는 구조 증강 모듈과 후보 엣지 선택 전략을 포함하여 SGNN 훈련을 개선합니다. SGA는 임베딩 공간에서 후보 샘플을 발견하기 위해 SGCN 알고리즘을 활용하며, 긴밀한 관계는 긍정적인 엣지로, 멀리 떨어진 관계는 부정적인 엣지로 해석합니다. 훈련 중 불균형 삼각형의 영향을 줄이기 위해 엣지 난이도 점수(edge difficulty scores)를 도입하여 커리큘럼 학습(curriculum learning) 전략을 적용합니다.

- **Performance Highlights**: SGA는 Slashdot 데이터셋에서 SGCN의 F1-micro 점수를 32.3% 개선하는 등, 총 6개 실제 데이터셋에서 5개의 기본 모델의 링크 서명 예측(link sign prediction) 정확도를 향상시켰습니다. 실험 결과, SGA는 SGNN의 성능을 상당히 향상시키는 효과를 보여주었습니다.



### DuoGNN: Topology-aware Graph Neural Network with Homophily and Heterophily Interaction-Decoupling (https://arxiv.org/abs/2409.19616)
- **What's New**: DuoGNN은 동질적(homophilic) 및 이질적(heterophilic) 간선을 분리하여 그래프 상의 상호작용을 효과적으로 포착하는 확장 가능하고 일반화 가능한 GNN 아키텍처입니다. 이는 모든 그래프 위상에서 잘 작동할 수 있는 성능 향상을 가져옵니다.

- **Technical Details**: DuoGNN은 세 가지 주요 기여를 포함합니다: (i) 동질성 상호작용을 추출하는 topological edge-filtering 알고리즘, (ii) 이질성 상호작용을 추출하고 확장성을 보장하는 heterophilic graph condensation 기법, (iii) 메시지 전달 동안 over-smoothing 및 over-squashing을 방지하는 dual aggregation pipeline입니다.

- **Performance Highlights**: DuoGNN은 의료 및 비의료 노드 분류 데이터셋에서 SOTA(state-of-the-art) 방법들과 비교하여 모든 작업에서 일관된 성능 향상을 보였습니다.



### Federated Learning from Vision-Language Foundation Models: Theoretical Analysis and Method (https://arxiv.org/abs/2409.19610)
- **What's New**: 본 논문은 CLIP과 같은 사전학습된 비전-언어 기초 모델을 연합 학습(federated learning)에 통합하여 다양한 작업에 대한 일반화(generalization)를 향상시키는 데 중점을 두고 있습니다. 특히, 프롬프트 기반 연합 학습(prompt-based federated learning)의 성능을 이해하기 위한 이론적 분석 프레임워크가 제시됩니다.

- **Technical Details**: 프롬프트 기반 연합 학습을 위한 분석 프레임워크는 feature learning theory를 기반으로 구성되어 있으며, 신호 학습(signal learning)과 노이즈 기억(noise memorization)의 진화를 모니터링합니다. 성과는 작업 관련(task-relevant) 계수와 작업 비관련(task-irrelevant) 계수의 비율로 평가됩니다. 또한, 포트폴리오 최적화(portfolio optimization)에서의 수익(income)과 위험(risk)의 유사성을 바탕으로, 글로벌 프롬프트(global prompt)와 로컬 프롬프트(local prompt)를 결합하여 프롬프트 포트폴리오를 구축합니다.

- **Performance Highlights**: 실험을 통해 프롬프트 포트폴리오의 성능 우위를 입증하였으며, 최적의 혼합 계수를 도출했습니다. 이론적 주장들은 실증적 실험에서도 지지를 받으며, 실제 시나리오에서의 접근 방식의 우수성을 꾸준히 보여주고 있습니다.



### Hyper-Connections (https://arxiv.org/abs/2409.19606)
- **What's New**: 새로운 방법론인 하이퍼 연결(hyper-connections)을 소개합니다. 이 방법은 기존의 잔여 연결(residual connections)의 몇 가지 단점을 다루는 것을 목표로 하며, 네트워크가 각기 다른 깊이의 피처(feature) 사이의 연결 강도를 조절하고 레이어를 동적으로 재배치할 수 있게 합니다.

- **Technical Details**: 하이퍼 연결은 네트워크가 피처 간의 연결 강도를 학습할 수 있게 하며, 깊이 연결(depth-connections)과 폭 연결(width-connections)을 제안합니다. 이를 통해 잔여 연결의 장점을 유지하면서도 계산량과 매개변수 증가를 최소화할 수 있습니다. 또한 동적 하이퍼 연결(dynamic hyper-connections)을 통해 입력에 따라 연결 가중치를 조정할 수 있습니다.

- **Performance Highlights**: 하이퍼 연결은 대형 언어 모델(LLMs)의 사전 학습(pre-training) 및 비전 처리(vison tasks)에서 잔여 연결 대비 성능 향상을 보였습니다. 예를 들어, DHC를 사용하는 모델이 1.8배 더 빠르게 수렴하며, ARC-Challenge에서 약 6점 향상을 나타냈습니다. 또한, 하이퍼 연결이 적용된 모델은 인접한 레이어 간의 특성 유사성을 줄이고, 각 레이어의 영향을 확장하는 데 기여합니다.



### The Crucial Role of Samplers in Online Direct Preference Optimization (https://arxiv.org/abs/2409.19605)
Comments:
          33 pages

- **What's New**: 이 논문에서는 Direct Preference Optimization (DPO)의 수렴 속도(convergence rates)를 다양한 샘플링 전략(sampling strategies) 하에서 철저히 분석합니다. 특히, uniform sampling이 linear convergence를 달성하는 반면, 제안된 online sampler는 quadratic convergence를 이룬다는 점이 놀랍습니다.

- **Technical Details**: DPO의 수렴 속도에 대한 이론적 분석을 제공하고, posterior distributions와 logit mixing을 통합하여 실용적인 환경에서도 샘플러(sampler)를 조정했습니다. 다양한 환경에서 DPO의 성능을 개선하기 위해 bandit environments 내에서의 최적화를 진행하였습니다.

- **Performance Highlights**: Safe-RLHF 데이터셋에서 제안한 방법은 vanilla DPO에 비해 4.5% 향상되었고, on-policy DPO에 대해서도 3.0%의 성과를 보였으며, Iterative-Prompt에서 vanilla DPO, on-policy DPO, Hybrid GSHF를 각각 4.2% 이상 능가하는 성과를 달성했습니다.



### An Unbiased Risk Estimator for Partial Label Learning with Augmented Classes (https://arxiv.org/abs/2409.19600)
Comments:
          17 pages

- **What's New**: Partial Label Learning with Augmented Class (PLLAC)을 위한 새로운 접근 방식이 소개되었습니다. 기존 PLL 모델이 테스트 세트의 클래스가 훈련 세트에 존재해야 한다는 제약을 깨고, 훈련 단계에서 보이지 않았던 새로운 클래스를 효과적으로 인식할 수 있도록 해결책을 제시합니다.

- **Technical Details**: 제안된 방법은 편향이 없는 위험 추정기(Unbiased Risk Estimator)를 통해 미지의 클래스 분포를 식별하고, 이는 표본이 없어도 잘 훈련된 모델의 힘을 빌리는 방식입니다. 이 추정기는 각각의 PLL 손실 함수에 장착할 수 있도록 설계되었습니다. 또한, 이론적 추정 오차 경계를 제공하여, 훈련 데이터의 수가 많아질수록 경험적 위험 최소화기가 진짜 위험 최소화기로 수렴하도록 보장합니다.

- **Performance Highlights**: UCI 데이터세트 및 실제 데이터세트를 포함한 광범위한 실험에서 제안된 방법이 높은 성능을 발휘함을 입증하였습니다. 이는 PLLAC 문제를 효과적으로 해결할 수 있음을 보여줍니다.



### Unifying back-propagation and forward-forward algorithms through model predictive contro (https://arxiv.org/abs/2409.19561)
- **What's New**: 본 연구에서는 Model Predictive Control (MPC) 프레임워크를 도입하여 딥 신경망(Deep Neural Networks) 훈련을 통합하고, Back-Propagation (BP) 및 Forward-Forward (FF) 알고리즘을 체계적으로 통일하였습니다. 이 프레임워크는 다양한 look-forward horizons를 제공하여 성능과 효율성 사이의 트레이드오프를 가능하게 합니다.

- **Technical Details**: 이 연구는 MPC 프레임워크를 통해 FF와 BP 알고리즘을 통합하고, 이를 바탕으로 성능-메모리 수요를 균형잡기 위한 다양한 최적화 알고리즘을 제안합니다. 이론적 분석을 바탕으로, 딥 리니어 네트워크에서 경량화된 경량화 알고리즘의 그라디언트 추정치가 완전한 역전파(BP)에 가까워질수록 다항적으로 수렴함을 보여주며, 메모리 수요는 지평선에 따라 항상 증가함을 나타냅니다.

- **Performance Highlights**:  본 방법은 다양한 모델과 작업에서 그 유용성을 보여주며, 주어진 목표와 모델 사양에 따라 최적의 지평선(horizon)을 선택하는 알고리즘을 제안합니다.



### Fast-Convergent and Communication-Alleviated Heterogeneous Hierarchical Federated Learning in Autonomous Driving (https://arxiv.org/abs/2409.19560)
Comments:
          16 pages

- **What's New**: 이번 논문에서는 거리 장면 의미 이해(TriSU) 문제의 해결을 위한 새로운 방법으로 Gaussian 이질적 계층 연합 학습(FedGau) 알고리즘을 제안합니다. 이 접근법은 다양한 도시에서 수집된 데이터의 특성을 고려하여 더 빠르고 효과적인 모델 수렴을 촉진합니다.

- **Technical Details**: FedGau 알고리즘은 각 RGB 이미지와 RGB 데이터셋을 Gaussian 분포로 모델링하여 가중치를 설계합니다. 또한 Bhattacharyya 거리(Dis) 계산을 통해 각 차량 모델의 가중치를 설정하여, 더 밀접한 관계를 가진 RGB 이미지가 우선적으로 통합되도록 합니다. 이를 통해 많은 시간과 자원을 절약할 수 있습니다. 추가적으로 AdapRS(Adaptive Resource Scheduling) 정책을 도입하여 필요 없는 통신 리소스를 줄입니다.

- **Performance Highlights**: FedGau는 기존의 HFL 방법에 비해 수렴 속도를 35.5%에서 40.6% 향상시키고, AdapRS 정책을 통해 기존 정적인 자원 계획에 비해 통신 부하를 29.65% 줄이면서 거의 동일한 성능을 유지합니다.



### Tailed Low-Rank Matrix Factorization for Similarity Matrix Completion (https://arxiv.org/abs/2409.19550)
- **What's New**: 이번 연구에서는 새로운 Similarity Matrix Completion (SMC) 프레임워크를 제안합니다. 이를 통해 기존의 Singular Value Decomposition (SVD) 기반 방법의 계산 복잡성을 줄이며, Positive Semi-definiteness (PSD) 특성을 활용해 신뢰성 높은 저랭크 솔루션을 확보하는 방법을 제시합니다.

- **Technical Details**: 제안된 SMC 프레임워크는 두 가지 중요 특성을 활용합니다: Positive Semi-definite (PSD) 특성과 Low-Rank 특성입니다. 이 프레임워크는 이를 통해 더 효과적인 이미지 검색, 문서 클러스터링, 추천 시스템 응용을 지원하며, SMCNN 및 SMCNmF라는 두 개의 알고리즘을 개발하였습니다. 알고리즘은 비선형 저랭크 정규화를 포함하여 성능을 향상시키고, 이론적 분석을 통해 수렴 속도를 확립합니다.

- **Performance Highlights**: 실제 데이터셋에서의 실험 결과, SMCNN 및 SMCNmF 알고리즘은 기존의 여러 baseline 방법들에 비해 뛰어난 성능과 효율성을 보여주었습니다. 제안된 방법들은 유사성 행렬 보완에 있어서 더욱 정확하고 신뢰할 수 있는 결과를 제공합니다.



### Almost Sure Convergence of Average Reward Temporal Difference Learning (https://arxiv.org/abs/2409.19546)
- **What's New**: 본 논문에서는 개념적으로 매우 간단한 TD 학습이 평균 보상 강화 학습에서 거의 확실한 수렴성을 가지는 것을 처음으로 증명하였습니다. 이는 25년이 넘는 시간동안 미해결 과제로 남아있던 문제로, 새롭게 도입된 확률적 근사 결과에 기반하고 있습니다.

- **Technical Details**: 이 연구는 Markovian 및 additive noise가 포함된 비확장적 맵핑에 관한 새로운 일반적인 확률적 근사 결과를 포함합니다. 기존의 Stochastic Krasnoselskii-Mann (SKM) 반복에 대한 수렴 분석을 확장하여, Tabular average reward TD(Temporal Difference)에서의 수렴성을 증명하였습니다.

- **Performance Highlights**: 이 논문의 핵심 기여는 평균 보상 TD의 반복 업데이트가 약한 조건 하에 샘플 경로에 의존하는 고정점에 거의 확실하게 수렴함을 증명한 것입니다. 이는 향후 다른 강화 학습 알고리즘의 분석에서도 활용될 수 있는 중요한 발판이 될 것입니다.



### Convergence-aware Clustered Federated Graph Learning Framework for Collaborative Inter-company Labor Market Forecasting (https://arxiv.org/abs/2409.19545)
- **What's New**: 본 연구에서는 Federated Labor Market Forecasting (FedLMF) 문제를 정의하고, 개인 정보를 보호하는 방식으로 인재 수요와 공급 예측을 위한 Meta-personalized Convergence-aware Clustered Federated Learning (MPCAC-FL) 프레임워크를 제안합니다.

- **Technical Details**: MPCAC-FL은 그래프 기반의 순차 모델을 설계하여 수요와 공급 시퀀스 및 회사-직위 쌍 간의 본질적인 상관 관계를 포착합니다. 메타 러닝 기법을 도입하여 각 회사에 공유할 수 있는 초기 모델 매개변수를 학습하며, 데이터가 이질적인 경우에도 개인화된 모델로 회사 특유의 수요와 공급을 예측할 수 있도록 최적화합니다. 마지막으로, Convergence-aware Clustering 알고리즘을 통하여 회사들을 모델 유사성에 따라 동적으로 그룹화하고 각 그룹 내에서 연합 집계를 수행합니다.

- **Performance Highlights**: MPCAC-FL은 세 가지 실제 데이터 세트에서 이전 모델들보다 우수한 성능을 보이며, 최고 수준의 모델인 DH-GEM에 비해 97% 이상의 정확도를 달성했습니다. 개인 회사 데이터를 노출하지 않으면서도 높은 예측 정확도를 유지합니다.



### Understanding Clinical Decision-Making in Traditional East Asian Medicine through Dimensionality Reduction: An Empirical Investigation (https://arxiv.org/abs/2409.19531)
Comments:
          11 pages, 3 figures

- **What's New**: 본 연구는 전통 동아시아 의학(TEAM)의 임상 의사 결정 과정을 차원 축소(dimensionality reduction) 관점에서 재해석하며, 외부-내부(Exterior-Interior) 패턴의 중요성과 필요성을 탐구합니다.

- **Technical Details**: 본 연구는 팔극 패턴 식별(Eight Principle Pattern Identification, EPPI) 시스템을 중심으로, 상한론(Shang-Han-Lun)에서 수집된 경험적 데이터를 활용하여 진단 및 치료 선택에서 외부-내부 패턴의 우선 순위를 정립합니다. 양적 측정 방법으로 추상화 지수(abstraction index), 교차 조건 일반화 성능(cross-conditional generalization performance), 결정을 나무 회귀(decision tree regression)를 사용하였습니다.

- **Performance Highlights**: 결과적으로 외부-내부 패턴이 가장 추상적이고 일반화 가능한 증상 정보를 나타내며, 증상과 한약 처방 공간 간의 효율적인 매핑(efficient mapping)을 촉진함을 보여주었습니다. 이는 TEAM과 현대 컴퓨팅 접근법을 연결하는 객관적인 틀을 제공하며, AI 기반 진단 도구 개발에 대한 통찰력을 제공합니다.



### KODA: A Data-Driven Recursive Model for Time Series Forecasting and Data Assimilation using Koopman Operators (https://arxiv.org/abs/2409.19518)
- **What's New**: 본 연구에서는 Koopman 연산자(Koopman operator)를 기반으로 한 데이터 동화(data assimilation) 접근 방법인 KODA(Koopman Operator with Data Assimilation)를 제안합니다. 이는 예측(forecasting)과 데이터 동화를 통합하여 비선형 동적 시스템(NLDS)에 적용됩니다.

- **Technical Details**: KODA는 Fourier 도메인 필터를 사용하여 데이터를 물리적 구성 요소와 잔여 동적(residual dynamics)으로 분리합니다. 이 과정에서 Koopman 연산자는 물리적 구성 요소의 동적을 정확히 표현하고, 잔여 동적은 유연하고 학습 가능한 재귀 모델로 캡처됩니다. 이러한 구조와 교육 기준은 안정적이고 장기 예측을 가능하게 합니다.

- **Performance Highlights**: KODA의 성능을 검증하기 위해 전기, 온도, 날씨, Lorenz 63, Duffing 오실레이터와 같은 여러 시간 시계열 벤치마크에서 기존의 최첨단 방법들보다 우수한 예측 결과를 보였습니다. KODA는 예측, 데이터 동화, 상태 예측의 세 가지 작업에서 뛰어난 효과성을 입증하였습니다.



### One Node Per User: Node-Level Federated Learning for Graph Neural Networks (https://arxiv.org/abs/2409.19513)
Comments:
          16 pages, 9 figures

- **What's New**: 이 논문에서는 Graph Neural Networks (GNNs)에 대한 node-level federated learning (FL) 프레임워크인 nFedGNN을 제안합니다. 이 프레임워크는 각 클라이언트가 단 하나의 feature vector만을 소유하는 상황에서도 협업 모델 훈련이 가능하도록 설계되었습니다.

- **Technical Details**: nFedGNN은 첫 번째 GNN 레이어의 message-passing과 feature vector 변환 프로세스를 분리하여 클라이언트와 클라우드 서버에서 각각 수행할 수 있도록 합니다. 또한, 단일 위치에서 feature vector의 latent representation에 기반한 graph Laplacian 용어를 도입하여 사용자 측 모델 업데이트를 규제합니다.

- **Performance Highlights**: 여러 데이터셋에 대한 실험 결과, nFedGNN은 기존의 기준 모델과 비교하여 우수한 성과를 달성하였습니다.



### Heterogeneity-Aware Resource Allocation and Topology Design for Hierarchical Federated Edge Learning (https://arxiv.org/abs/2409.19509)
Comments:
          12 pages, 9 figures

- **What's New**: 이번 연구는 Federated Learning (FL) 방법 중 Hierarchical Federated Edge Learning (HFEL)의 훈련 효율성을 높이는 전략적 자원 할당과 토폴로지 설계를 제안합니다.

- **Technical Details**: 연구자는 두 계층으로 구성된 HFEL 시스템을 고려하고, 에지 장치가 에지 서버와 연결되며 이들 서버가 P2P (peer-to-peer) 에지 백홀을 통해 상호 연결된 구조를 설계했습니다. 최적화 문제를 수립하여 통신 및 계산 자원을 할당하고 P2P 연결을 조정하여 총 훈련 지연 시간을 최소화합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 훈련 지연 시간을 획기적으로 감소시키면서도 모델의 정확도를 다양한 기준선과 비교해 유지하는 효과를 보였으며, 데이터 및 시스템 이질성 하에서도 대규모 FL 구현을 가능하게 합니다.



### Spatial Reasoning and Planning for Deep Embodied Agents (https://arxiv.org/abs/2409.19479)
Comments:
          DPhil Thesis - Engineering Science, University of Oxford. Original copy available at this https URL

- **What's New**: 이 논문은 Embodied agents가 복잡한 작업을 수행할 수 있도록 하는 데이터 중심 기술 개발에 초점을 맞추고 있습니다. 주요 기여로는 해석 가능하고 장기 계획을 위한 모델을 학습하는 CALVIN, 무감독 RL 알고리즘 SOAP, 코드 최적화 프레임워크 LangProp, 복잡한 작업을 수행하는 Voggite가 포함됩니다.

- **Technical Details**: 1) CALVIN은 전문가 시연으로부터 보상 및 상태 전환을 학습하여 부분적으로 관찰 가능한 3D 환경에서 성공적으로 탐색하는 차별적 계획자입니다. 2) SOAP은 작업을 하위 작업으로 분할하고 일관된 실행을 가능하게 하는 옵션을 발견하는 RL 알고리즘입니다. 3) LangProp은 LLM을 사용하여 코드 최적화를 통해 코드가 학습 가능한 정책으로 처리되는 코드 최적화 프레임워크를 제공합니다. 4) Voggite는 Minecraft에서 복잡한 작업을 해결하는 비전-액션 변환기를 사용하는 Embodied agent입니다.

- **Performance Highlights**: 이 연구의 성과로는 CALVIN이 3D 환경에서의 탐색 성능 향상, SOAP이 Atari와 같은 고전 벤치마크에서 강력한 성능, LangProp이 CARLA 자율 주행 벤치마크에서 인간 전문가와 동등한 성과를 달성하며, Voggite가 MineRL BASALT 대회에서 3위를 차지한 점이 있습니다.



### Hedging and Approximate Truthfulness in Traditional Forecasting Competitions (https://arxiv.org/abs/2409.19477)
- **What's New**: 이 논문은 전통적인 예측 경진 대회 메커니즘인 'Simple Max'의 전략적 분석을 제공합니다. 저자들은 이 메커니즘에서 발생하는 비진실성(non-truthfulness) 문제와 '장기적인 진실성(long-run truthfulness)'의 신화가 사실이 아님을 입증합니다.

- **Technical Details**: (Simple Max) 메커니즘은 다수의 예측자를 평가하여 가장 높은 점수를 받은 예측자가 우승하는 방식입니다. 논문에서는 (equilibrium) 상태에서 최고 예측자가 다른 예측자들의 주장에 맞춰 자신의 보고를 조정하기 때문에 진실하지 않게 행동할 수 있음을 보여줍니다. 조건이 충족되지 않으면 경쟁자가 자신을 이길 수 있다고 믿고 적절한 불확실성이 존재할 때만 약간의 진실성을 유지합니다.

- **Performance Highlights**: 이 연구 결과는 전통적인 예측 메커니즘에서의 비진실성이 단순한 신화와 같지 않음을 입증하며, 플랫폼이 경쟁자들로부터 진실한 믿음을 유도하고자 할 때 피해야 할 설정들을 명확히 이해할 수 있도록 돕습니다. 또한, 특정 조건에서 예측자들이 자신의 믿음에 따라 행동할 것임을 보여줍니다.



### On the universality of neural encodings in CNNs (https://arxiv.org/abs/2409.19460)
Comments:
          Appeared at the ICLR 2024 Workshop on Representational Alignment (Re-Align), 13 pages, 5 figures

- **What's New**: 이번 연구는 이미지 분류 작업에 대해 훈련된 합성곱 신경망(convolutional neural networks)에서의 신경 인코딩의 보편성을 조사합니다. 기존의 방법론과 달리, 우리는 학습된 가중치의 유사성을 직접 비교하는 절차를 개발했습니다. VGG 타입 네트워크의 여러 레이어에서 학습된 고유 벡터(eigenvectors)가 다양한 자연 이미지 데이터 세트 간에 보편적으로 나타난다는 것을 보여주었습니다.

- **Technical Details**: 네트워크의 가중치(w)를 직접 비교하기 위해 공간 및 채널 차원의 분해를 기반으로 한 절차를 사용했습니다. 우리 연구는 CNN의 가중치 텐서의 통계적 성질을 분석함으로써, 공간 필터의 고유 벡터가 단순하다는 것을 발견했습니다. 이는 필터 크기나 데이터 세트에 상관없이 저차원적이며, 보편적인 공간 필터 고유 벡터 세트가 나타남을 의미합니다.

- **Performance Highlights**: 자연 이미지를 위한 보편적인 신경 인코딩이 발생함을 보여줍니다. 우리는 데이터 세트와 작업 간에 인코딩 유사성을 발견했으며, 이는 전이 학습(transfer learning)과 자가 지도 학습(self-supervised learning)의 성공을 설명합니다. 이 연구는 신경망의 성능을 극대화하는 것 대신, 학습된 인코딩의 보편성을 극대화하며, 원칙적인 기초 모델을 구축하는 방법을 제시합니다.



### Strongly-Polynomial Time and Validation Analysis of Policy Gradient Methods (https://arxiv.org/abs/2409.19437)
- **What's New**: 이번 논문은 강화학습(Reinforcement Learning, RL)에서 최적성(optimality)에 대한 원칙적인 측정 지표가 부족하다는 문제를 다루며, 유한 상태 및 행동 마르코프 결정 과정(Markov Decision Process, MDP)에서 간단하고 계산 가능한 새로운 gap function을 개발했습니다. 이를 통해 최적성 gap의 상한과 하한을 제공하며, 새로운 개념인 분포 독립 수렴(distribution-free convergence)을 정의합니다.

- **Technical Details**: 개발된 gap function은 벡터의 최대 원소를 찾거나(convex optimization problem) 해결하는 방식으로 이루어집니다. 기본 정책 거울 하강(Policy Mirror Descent, PMD)을 통해 결정론적(deterministic) 및 확률론적(stochastic) 환경에서의 분포 독립 수렴을 보장하며, 특히 결정론적 환경에서는 최적 정책을 강한 다항 시간 내에 찾을 수 있는 방법론을 제시합니다.

- **Performance Highlights**: 결정론적 설정에서 PMD는 선형 수렴률을 보여주며, 확률론적 환경에서는 O(k^{-1/2})와 O(k^{-1})의 수렴률을 제공합니다. 또한, 새로운 결과로는 기본 확률 정책 거울 하강(Basic Stochastic Policy Mirror Descent, SPMD)이 정확한 gap function의 온라인 추정을 할 수 있으며, 이를 통해 신뢰할 수 있는 종료 기준으로 사용할 수 있는 정확도 추정치를 제공합니다.



### Simulation-based inference with the Python Package sbijax (https://arxiv.org/abs/2409.19435)
- **What's New**: 이 논문에서는 Neural Simulation-Based Inference (SBI)라는 신흥 방법군을 위한 Python 패키지인 sbijax를 소개합니다. 이 패키지는 사용자 친화적인 프로그래밍 인터페이스를 통해 다양한 최신 SBI 방법들을 구현합니다.

- **Technical Details**: sbijax는 다양한 상태의 SBI 추정기를 쉽게 구축하고, 몇 줄의 코드만으로 posterior (사후) 분포를 계산 및 시각화할 수 있는 고급 기능을 제공합니다. 또한, 일반적인 Approximate Bayesian Computation (ABC)을 위한 기능도 포함되어 있습니다. 이 패키지는 JAX로 완전히 작성되어 있어, 신경망을 빠르게 훈련하고 CPU 및 GPU에서 자동으로 코드를 병렬 실행하는 데 효율적입니다.

- **Performance Highlights**: sbijax는 신경망 훈련의 속도와 계산 효율성을 극대화하여, 사용자들이 복잡한 Bayesian inference (베이지안 추론) 문제를 쉽게 다룰 수 있도록 돕습니다.



### RMLR: Extending Multinomial Logistic Regression into General Geometries (https://arxiv.org/abs/2409.19433)
Comments:
          Accepted to NeurIPS 2024

- **What's New**: 본 논문은 일반 기하학에서 Riemannian Multinomial Logistic Regression(RMLR)을 설계하기 위한 새로운 프레임워크를 제안합니다. 이 프레임워크는 최소한의 기하학적 속성만 요구하며, 다양한 기하학과 함께 사용할 수 있는 넓은 적용성을 보여줍니다.

- **Technical Details**: RMLR는 기본적으로 Riemannian logarithm의 명시적 표현만을 요구하며, 이는 머신 러닝에서 자주 사용되는 여러 매니폴드에 의해 충족됩니다. 논문에서는 SPD 매니폴드와 회전 행렬에 대해 RMLR 프레임워크를 구체적으로 평가하며, SPD 매니폴드에서는 다섯 가지 가족의 SPD MLR을 제안하고, 회전 행렬에서는 널리 사용되는 bi-invariant metric을 기반으로 한 Lie MLR을 발전시킵니다.

- **Performance Highlights**: 제안된 RMLR은 다양한 Riemannian 백본 네트워크에서 효과가 입증되었습니다. 특히, SPD MLR은 SPDNet 및 RResNet에서 각각 14.23% 및 13.72% 향상을 보였으며, EEG 분류 작업에서는 TSMNet에서 4.46% 개선되었습니다. 또한, Lie MLR은 훈련의 안정성과 성능을 모두 향상시키는 데 기여했습니다.



### MicroFlow: An Efficient Rust-Based Inference Engine for TinyML (https://arxiv.org/abs/2409.19432)
- **What's New**: MicroFlow는 Rust 프로그래밍 언어로 개발된 경량의 오픈소스 TinyML 프레임워크로, 임베디드 시스템에서 Neural Networks(NNs)를 효율적으로 배포할 수 있도록 설계되었습니다. 이는 메모리 안전성과 특징을 강조하며, 자원 제약이 있는 환경에서도 안정적인 작동이 가능합니다.

- **Technical Details**: MicroFlow는 컴파일러 기반의 inference engine 방식을 사용하고 있으며, 메모리 안전성을 보장하는 Rust를 활용하여 일반적인 메모리 관련 오류를 방지합니다. 메모리는 정적 할당이 가능하며, 8비트 마이크로컨트롤러에서도 NN 모델의 일부만을 RAM에 로드하여 처리할 수 있게 설계되었습니다. 또한, MicroFlow의 코드와 구현은 완전히 오픈소스로, GitHub에서 무료로 제공됩니다.

- **Performance Highlights**: MicroFlow는 기존의 최신 솔루션에 비해 적은 Flash 및 RAM 메모리를 사용합니다. 중간 크기 NN에 대해 더 빠른 inference 속도를 달성하며, 대형 NN의 경우 유사한 성능을 보입니다. 실험 결과에 따르면 자원이 제한된 환경에서도 효율적인 TinyML 모델 배포가 가능함을 입증하였습니다.



### Identifiable Shared Component Analysis of Unpaired Multimodal Mixtures (https://arxiv.org/abs/2409.19422)
- **What's New**: 본 연구는 크로스 모달리티(sample들 간) 데이터가 정렬되지 않았을 때도 공유 구성 요소(shared components)를 식별할 수 있는 충분한 조건을 제공하며, 기존 연구보다 더 완화된 조건을 제시합니다.

- **Technical Details**: 연구에서는 unaligned shared component analysis (unaligned SCA) 문제를 다루며, 이를 위해 분포 차이(distribution divergence)를 최소화하는 손실(loss) 함수를 제안합니다. 이 손실 함수는 다중 모달 데이터의 확률 분포를 일치시키는 방식을 통해 공유 구성 요소를 식별합니다.

- **Performance Highlights**: 제안된 방법은 실제 데이터 및 합성 데이터에서 유효성을 검증하였으며, 크로스-언어(word retrieval), 유전 정보 정렬(genetic information alignment), 이미지 데이터 도메인 적응(image data domain adaptation)과 같은 응용 분야에 효과적으로 적용될 수 있음을 보여주었습니다.



### Sequential Signal Mixing Aggregation for Message Passing Graph Neural Networks (https://arxiv.org/abs/2409.19414)
Comments:
          Accepted to NeurIPS 2024

- **What's New**: 이 연구에서는 Message Passing Graph Neural Networks (MPGNNs)에서의 집합 모듈의 한계를 설명하고, 이를 개선하기 위해 Sequential Signal Mixing Aggregation (SSMA)라는 새로운 집합 모듈을 제안합니다. SSMA는 이웃 특성을 2D 이산 신호로 취급하며, 이웃의 특성을 결합하는 능력을 향상시킵니다.

- **Technical Details**: SSMA는 이웃의 특성을 2D 신호로 처리하고, 이를 순차적으로 합성곱(convolve)함으로써 특징 간의 혼합(mixing)을 양적으로 향상시키는 방법입니다. 이 모델의 이론적 기반은 DeepSets를 기반으로 하며, 다차원 특징을 효과적으로 처리할 수 있도록 효율적으로 확장됩니다. 이웃 수(n)와 특징 차원(d)에 대해 m=𝒪⁢(n²d)의 이론적 표현 크기를 제공합니다.

- **Performance Highlights**: SSMA를 기존의 MPGNN 아키텍처와 결합했을 때, 다양한 벤치마크에서 성능이 실질적으로 향상되었으며, 여러 설정에서 최신 기술(state-of-the-art) 결과를 달성했습니다. 실험에서는 TU 데이터셋, Open Graph Benchmark(OGB), Long-range Graph Benchmarks(LRGB), 그리고 ZINC 화합물 속성 예측 데이터셋에서의 성과가 포함되었습니다.



### Canonical Correlation Guided Deep Neural Network (https://arxiv.org/abs/2409.19396)
Comments:
          11 pages, 13 figures

- **What's New**: 제안된 새로운 접근 방식은 Canonical Correlation Guided Deep Neural Network (CCDNN)입니다. 이는 다변량 분석(MVA)과 머신 러닝의 새로운 융합으로, 기존의 선형 상관관계 분석(CCA)과는 달리 상관관계를 최대화하는 것이 아니라 제약 조건으로 사용하여 최적화 문제를 해결합니다.

- **Technical Details**: CCDNN은 딥 뉴럴 네트워크(DNN)를 기반으로 하며, 두 개의 데이터 뷰에서 높은 선형 상관 관계를 갖는 표현을 학습하는 프레임워크입니다. 이 네트워크는 특정 최적화 작업(재구성, 분류, 예측)에 중점을 두고 있으며, 상관 관계로 인한 중복을 줄이기 위해 중복 필터를 설계했습니다.

- **Performance Highlights**: MNIST 데이터셋에서의 실험 결과, CCDNN은 DCCA 및 DCCAE보다 평균 제곱 오차(MSE)와 평균 절대 오차(MAE) 측면에서 더 나은 재구성 성능을 보여주었습니다. 제안된 방법은 산업 고장 진단 및 잔여 유용 수명 예측에서도 우수한 성능을 입증하였습니다.



### Value-Based Deep Multi-Agent Reinforcement Learning with Dynamic Sparse Training (https://arxiv.org/abs/2409.19391)
- **What's New**: 이번 논문은 Deep Multi-agent Reinforcement Learning (MARL)에서의 계산 복잡성을 경감하기 위한 새로운 접근 방식을 제안합니다. 특히 Dynamic Sparse Training (DST)을 활용한 Multi-Agent Sparse Training (MAST) 프레임워크를 소개하여, 신뢰할 수 있는 학습 목표와 샘플 분포의 합리성을 동시에 향상시키는 방법을 다룹니다.

- **Technical Details**: MAST는 Soft Mellowmax Operator와 하이브리드 TD-(λ) 스키마를 통합하여 신뢰할 수 있는 학습 목표를 설정하며, 이중 리플레이 버퍼 메커니즘을 통해 훈련 샘플 분포를 개선합니다. 또한, MAST는 경량 네트워크를 사용하는 다수의 MARL 에이전트를 독점적으로 훈련시키기 위해 그래디언트 기반의 토폴로지 진화(Topology Evolution)를 활용합니다.

- **Performance Highlights**: MAST 프레임워크는 훈련 및 추론에 필요한 Floating Point Operations (FLOPs)를 최대 20배 줄이면서도 3% 미만의 성능 저하를 기록했습니다. 실험 결과는 다양한 가치 기반 MARL 알고리즘을 통해 모델 압축을 5배에서 20배까지 달성한 첫 사례로 보고되고 있습니다.



### DOTA: Distributional Test-Time Adaptation of Vision-Language Models (https://arxiv.org/abs/2409.19375)
Comments:
          In submission

- **What's New**: 이 논문에서는 기존의 Training-Free Test-time Dynamic Adapter(TDA)의 한계를 극복하기 위해 DistributiOnal Test-time Adaptation(Dota)라는 새로운 방법을 제안합니다. Dota는 테스트 샘플의 분포를 지속적으로 추정하여 모델이 배포 환경에 적응할 수 있게 합니다.

- **Technical Details**: Dota는 Bayes' 정리를 기반으로 하여 추정한 분포를 사용하여 테스트 샘플의 후방 확률(test-time posterior probabilities)을 계산합니다. 여기에서 각 클래스의 임베딩 분포가 가우시안 분포를 따른다는 가정을 하며, 이는 TDA보다 약 20배 빠른 추론 속도를 제공합니다. 또한, Dota는 사람-주도 피드백(human-in-the-loop paradigm)을 통해 불확실한 샘플을 식별하고 적응하도록 돕습니다.

- **Performance Highlights**: 광범위한 데이터셋을 통한 실험 결과 Dota는 CLIP 모델이 지속적으로 학습할 수 있게 해주며, 기존의 최첨단 방법들에 비해 유의미한 성과 향상을 보였습니다.



### Sparse Modelling for Feature Learning in High Dimensional Data (https://arxiv.org/abs/2409.19361)
- **What's New**: 본 논문은 고차원 데이터셋에서 차원 축소 및 특징 추출을 위한 혁신적인 접근 방식을 제안하며, 특히 목재 표면 결함 탐지에 대한 적용에 중점을 둡니다.

- **Technical Details**: 제안된 프레임워크는 희소 모델링 기법(sparse modeling techniques), 특히 Lasso 및 근접 경량화(proximal gradient) 방법을 통합한 포괄적인 파이프라인을 통해 효율적이고 해석 가능한 특징 선택(feature selection)을 지원합니다. VGG19와 같은 사전 훈련된(pre-trained) 모델을 활용하고, Isolation Forest 및 Local Outlier Factor와 같은 이상 탐지(anomaly detection) 방법을 포함시켜 복잡한 데이터셋에서 유의미한 특징을 추출하는 문제에 접근합니다.

- **Performance Highlights**: 정확도(accuracy)와 F1 점수(F1 score)와 같은 평가 지표를 활용하여 희소 모델링 기법의 성능을 평가하며, 시각화(visualizations)를 통해 결과를 보완합니다. 이 연구를 통해 목재 표면 결함 탐지 문제의 맥락에서 머신러닝에서 희소 모델링의 이해 및 적용을 발전시키고자 합니다.



### Unveil Benign Overfitting for Transformer in Vision: Training Dynamics, Convergence, and Generalization (https://arxiv.org/abs/2409.19345)
- **What's New**: 이 논문은 Vision Transformer (ViT) 의 이론적 능력을 탐구하며, 특히 훈련 데이터에 과적합(overfit) 되는 경우의 일반화(generalization) 가능성을 이해하는 데 중점을 둡니다. 이는 최근 큰 발전을 이룬 Transformer 모델의 이론적 기초를 강화하기 위한 작업입니다.

- **Technical Details**: 우리는 self-attention layer와 softmax, 그리고 fully connected layer로 구성된 Transformer의 최적화(optimization)를 gradient descent를 통해 특정 데이터 분포 모델에서 연구하였습니다. softmax의 도전과제 및 Transformer 최적화에서 여러 가중치의 상호 의존적 성격을 해결하는 기술을 개발함으로써, 훈련 동역학(training dynamics)을 성공적으로 특성화하고 사후 훈련에서 일반화를 달성했습니다.

- **Performance Highlights**: 우리의 결과는 데이터 모델의 신호 대 잡음 비(signal-to-noise ratio)를 기반으로 작은 테스트 오류(test error) 단계와 큰 테스트 오류 단계 간의 날카로운 조건을 구분할 수 있음을 보여줍니다. 이론적 결과는 실험적 시뮬레이션으로进一步적으로 검증되었습니다.



### A Generalized Model for Multidimensional Intransitivity (https://arxiv.org/abs/2409.19325)
Comments:
          13 pages, 1 figure

- **What's New**: 이 논문에서는 비가환성(intransitivity) 문제를 해결하기 위해, 플레이어의 d차원 표현(d>1)과 데이터셋 특화 거리 공간(metric space)을 공동으로 학습하는 확률적 모델을 제안하였습니다. 이는 비가환적 표현 학습에 있어 이전의 모델로 수렴할 수 있는 흥미로운 결과를 보여줍니다.

- **Technical Details**: 제안된 모델은 각 플레이어의 d차원 표현을 학습하고, 해당 표현 공간에서 적절한 거리 형식을 체계적으로 캡처합니다. 추가적인 제약 조건을 통해 이 모델은 기존의 비가환적 표현 학습에 사용되는 모델로 수렴할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 사회적 선택(social choice), 선거(election), 온라인 게임 데이터셋에 대한 예측 성능이 향상되었습니다. 여러 경쟁 모델들에 비해 예측 정확도에서 우수한 성과를 보였습니다.



### Explicit construction of recurrent neural networks effectively approximating discrete dynamical systems (https://arxiv.org/abs/2409.19278)
- **What's New**: 본 논문에서는 동적 시스템에서 유래한 유한한 질량의 임의의 경계 값 이산 시계열에 대한 재귀 신경망(recurrent neural networks, RNN)의 명시적 구성을 제공합니다. 이를 통해 대응하는 이산 동적 시스템을 효과적으로 근사합니다.

- **Technical Details**: 연구에서는 지연 좌표에서 동적 시스템을 정의하고, Lyapunov 지수를 통해 재구성된 시스템과 원시 시스템 간의 최대 Lyapunov 지수를 최소화하도록 RNN을 구성합니다. 특히, 필자는 기본적인 대수적 접근법을 기반으로 RNN을 명시적으로 구성하는 초기 작업을 진행합니다.

- **Performance Highlights**: 제시된 RNN 구조는 큰 K와 C에 대해 기존 시스템의 동작을 잘 근사할 수 있으며, 로그 등급 차이로 인해 수행 성능이 약간 저하됩니다. 이는 RNN의 유용성을 더해줍니다.



### VecLSTM: Trajectory Data Processing and Management for Activity Recognition through LSTM Vectorization and Database Integration (https://arxiv.org/abs/2409.19258)
Comments:
          10 pages, 5 figures

- **What's New**: 이 연구에서는 LSTM(Long Short-Term Memory) 기반 신경망의 성능과 효율성을 향상시키기 위한 새로운 프레임워크인 VecLSTM을 제안합니다. VecLSTM은 벡터화(vectorization) 층을 통합하여 입력 시퀀스를 보다 효율적으로 처리합니다.

- **Technical Details**: VecLSTM 방법론은 전통적인 LSTM 모델과 달리 데이터 전처리 과정에서 지리적 좌표를 2D 그리드 표현으로 변환하는 벡터화 과정을 도입합니다. 이후 CNN(Convolutional Neural Network)을 활용하여 특징을 추출하고, CNN과 LSTM 모델의 출력을 조합하여 지역적 및 전역적 공간 의존성을 캡처합니다.

- **Performance Highlights**: 실험 결과, VecLSTM은 1,467,652개의 샘플로 구성된 데이터셋에서 기존 LSTM 모델보다 우수한 정확도(85.57%의 검증 정확도, 85.47%의 테스트 정확도, 0.86의 가중치 F1 점수)를 보여주며, 모델 훈련 시간을 26.2% 단축시켰습니다.



### HybridFlow: A Flexible and Efficient RLHF Framework (https://arxiv.org/abs/2409.19256)
- **What's New**: 본 논문은 HybridFlow라는 새로운 RLHF(인간 피드백을 통한 강화 학습) 프레임워크를 제안합니다. 이 프레임워크는 단일 컨트롤러와 다중 컨트롤러 패러다임을 융합하여 복잡하고 자원 집약적인 RLHF 데이터 흐름을 효율적으로 실행하고 표현할 수 있게 합니다.

- **Technical Details**: HybridFlow는 계층적 API 세트를 설계하여 RLHF 데이터 흐름에서 계산과 데이터 종속성을 분리하고 캡슐화합니다. 이를 통해 RLHF 알고리즘의 효율적인 작업 오케스트레이션과 다양한 장치에 대한 계산을 정의할 수 있습니다. 또한, 3D-HybridEngine을 설계하여 훈련 및 생성 단계 간의 메모리 중복을 없애고 통신 오버헤드를 크게 줄이도록 하였습니다.

- **Performance Highlights**: 실험 결과, 다양한 RLHF 알고리즘을 HybridFlow로 실행했을 때 최고 1.53배에서 20.57배의 처리량(throughput) 향상을 보여주었습니다.



### Forgetting, Ignorance or Myopia: Revisiting Key Challenges in Online Continual Learning (https://arxiv.org/abs/2409.19245)
Comments:
          28 pages, Accepted by NeurIPS 2024

- **What's New**: 이 논문은 Online Continual Learning (OCL)에서 모델 성능과 학습 속도 간의 상관관계를 강조하며, 효과적인 학습을 위한 non-sparse classifier evolution (NsCE) 프레임워크를 제안합니다.

- **Technical Details**: 제안된 NsCE 프레임워크는 비희소 최대 분리 정규화(non-sparse maximum separation regularization)와 선택적 경험 재생(targeted experience replay) 기법을 통합하여 모델의 전반적인 분별 특성 학습을 촉진합니다. 이를 통해 훈련 샘플 처리 속도(model throughput)를 크게 개선할 수 있습니다.

- **Performance Highlights**: NsCE 프레임워크는 기존 OCL 방법보다 데이터 스트림 처리에서 성능을 향상시켰으며, 주요 성능 지표에서 실제 환경에서도 효율성을 발휘하는 것을 입증했습니다.



### Zorro: A Flexible and Differentiable Parametric Family of Activation Functions That Extends ReLU and GELU (https://arxiv.org/abs/2409.19239)
Comments:
          13 pages, 7 figures, 9 tables

- **What's New**: 최근 Neural Network 아키텍처에서 활성화 함수(Activation Function)의 중요성을 강조하는 연구가 발표되었습니다. 특히, ReLU를 대체할 수 있는 새로운 활성화 함수 집합인 Zorro를 소개하며, 이는 ReLU와 Sigmoid의 결합으로 이루어진 5개의 변형으로 구성됩니다.

- **Technical Details**: Zorro 함수는 연속적으로 미분 가능하며 유연한 구조를 가지고 있습니다. Zorro의 다섯 가지 주요 변형(Symmetric-Zorro, Asymmetric-Zorro, Sigmoid-Zorro, Tanh-Zorro, Sloped-Zorro)은 ReLU의 특성과 Sigmoid의 특성을 결합하여 설계되었습니다. 이러한 함수들은 데이터 세트와 아키텍처에 맞춰 조정할 수 있는 매개변수를 제공하여, ReLU의 비정상적 점과 폭발적인 기울기 문제를 해결합니다.

- **Performance Highlights**: Zorro 활성화 함수는 완전 연결 네트워크(fully connected network), CNN(Convolutional Neural Network), Transformer 아키텍처에서 테스트되었으며, 각 아키텍처에서의 효과성도 입증되었습니다. 이는 기존의 활성화 함수들보다 개선된 성능을 보여줍니다.



### Double Actor-Critic with TD Error-Driven Regularization in Reinforcement Learning (https://arxiv.org/abs/2409.19231)
- **What's New**: TDDR(Temporal Difference error-Driven Regularization) 알고리즘이 도입되었습니다. 이 알고리즘은 double actor-critic 구조를 기반으로 하며, 기존 알고리즘과 비교해 추가적인 hyperparameter 없이 뛰어난 성능을 보입니다.

- **Technical Details**: TDDR는 double actors와 critics를 사용하여 Q-value를 개선합니다. 이 알고리즘은 clipped double Q-learning (CDQ) 메커니즘을 도입하여 각 critic 업데이트에 적절한 Q-value를 선택하는 데 temporal difference (TD) 오류를 활용합니다. 또한, 학습 과정에서 랜덤 및 동시 업데이트에서도 수렴성을 보장합니다.

- **Performance Highlights**: TDDR는 MuJoCo 및 Box2D와 같은 연속 제어 작업에서 benchmark 알고리즘 대비 우수한 경쟁력을 발휘하며, 복잡한 하이퍼파라미터 조정 없이도 뛰어난 가치 추정을 제공합니다.



### Cauchy activation function and XN (https://arxiv.org/abs/2409.19221)
- **What's New**: 새로운 Cauchy Activation Function을 제안하고, 이를 활용한 CompleXNet(XNet)이라는 새로운 종류의 신경망을 개발하였습니다. 이 함수는 복소 해석학의 Cauchy 적분 정리에 기반하며, 정확도가 필요한 문제에 적합하게 설계되었습니다.

- **Technical Details**: Cauchy Activation Function은 λ1, λ2, d와 같은 훈련 가능한 매개변수를 사용하여 표현되며, 이 함수는 매끄러운 함수를 최고 가능 순서까지 근사할 수 있습니다. XNet은 이미지 분류와 편미분 방정식(Partial Differential Equations, PDE) 해결을 포함한 고차원 문제에서 효과적입니다.

- **Performance Highlights**: XNet은 MNIST와 CIFAR-10과 같은 기존 벤치마크를 크게 초월하며, Physics-Informed Neural Networks(PINNs)와 대비하여 저차원의 PDE 및 고차원 PDE 시나리오에서도 상당한 이점을 제공합니다.



### A Characterization of List Regression (https://arxiv.org/abs/2409.19218)
- **What's New**: 최근 리스트 학습(list learning) 작업의 샘플 복잡성(sample complexity)을 이해하고 특징 지으려는 노력이 증가하고 있습니다. 본 연구에서는 PAC(Probably Approximately Correct) 리그레션(list PAC regression)에 대한 완전한 특징을 제시하며, 두 가지 조합 차원인 k-OIG 차원(k-OIG dimension)과 k-fat-shattering 차원(k-fat-shattering dimension)을 제안합니다.

- **Technical Details**: k-OIG 차원은 실현 가능한(list learnable) 설정을, k-fat-shattering 차원은 비실현적인(agnostic) 설정을 각각 최적으로 나타냅니다. 이는 일반적인 리그레션(standard regression)에서 알려진 차원들을 일반화한 것입니다. 또한, 본 연구에서 제시된 조건은 주어진 실수 값 가설 클래스(hypothesis class)가 k개의 레이블을 갖는 리스트 학습 가능성을 승인하도록 하는 필요 및 충분 조건을 포함합니다.

- **Performance Highlights**: 샘플 복잡성 경계는 실현 가능한 및 비실현적인 설정 모두에서 최적이며, 제안된 차원에 대한 의존성에 있어 polylogarithmic 요소로 최적화될 수 있습니다. 실현 가능한 경우에는 OIG 차원이, 비실현적인 경우에는 fat-shattering 차원이 적절한 양으로 고려됩니다.



### An Accelerated Algorithm for Stochastic Bilevel Optimization under Unbounded Smoothness (https://arxiv.org/abs/2409.19212)
Comments:
          Accepted by NeurIPS 2024. The code is available at this https URL

- **What's New**: 이 논문에서는 비선형 일반화가 가능한 상위 레벨 함수와 강한 볼록성을 가진 하위 레벨 문제를 포함하는 확률적 바이레벨 최적화 문제를 다룹니다. 이 연구는 RecNN과 같은 순차 데이터 학습에 있어 중요한 응용을 가집니다. 이 논문은 새로운 알고리즘인 AccBO를 제안하며, 이를 통해 기존 방법에 비해 성능을 현저히 개선하였습니다.

- **Technical Details**: 제안된 알고리즘 AccBO는 상위 레벨 변수의 업데이트에 정규화된 확률적 경량법(stochastic gradient descent)과 순환 모멘텀(recursive momentum)을 사용하며, 하위 레벨 변수는 평균화를 통한 확률적 Nesterov 가속 기법(Nesterov accelerated gradient descent)으로 업데이트합니다. 이 알고리즘은 oracle 복잡성(oracle complexity) 관점에서 O~(1/ϵ^3)의 결과를 도출하였고, 이를 통해 우리가 얻는 이론적 성과를 보장합니다.

- **Performance Highlights**: 다양한 실험 결과를 통하여 AccBO 알고리즘이 이론적으로 예측된 가속화를 달성함을 확인하였고, 바이레벨 최적화 문제에서 기존의 최적화 알고리즘들보다 현저히 뛰어난 성능을 보였습니다.



### Boosting SISSO Performance on Small Sample Datasets by Using Random Forests Prescreening for Complex Feature Selection (https://arxiv.org/abs/2409.19209)
- **What's New**: 본 논문에서는 기존 SISSO 방법의 메모리 요구사항을 줄이면서 복잡한 문제에서의 성능을 향상시키기 위해 Random Forest (RF)와 SISSO를 결합한 RF-SISSO 알고리즘을 제안합니다.

- **Technical Details**: RF-SISSO 알고리즘은 Random Forest 알고리즘을 사용하여 입력 데이터의 비선형 관계를 포착하고 특성 선택(feature selection)을 개선합니다. 이를 통해 회귀(regression) 및 분류(classification) 작업에서 정확성과 효율성을 높일 수 있습니다.

- **Performance Highlights**: RF-SISSO는 299개의 물질에 대한 SISSO 검증 문제에서 0.9 이상의 정확도를 유지하며, 특히 작은 훈련 샘플 크기에서 회귀 효율성을 크게 향상시킵니다. 45개 샘플을 가진 훈련 부분집합에서는 RF-SISSO가 기존 SISSO보다 265배 높은 효율성을 보였습니다.



### Evidence Is All You Need: Ordering Imaging Studies via Language Model Alignment with the ACR Appropriateness Criteria (https://arxiv.org/abs/2409.19177)
Comments:
          15 pages main text, 4 figures, 1 table

- **What's New**: 이번 연구에서는 generative AI와 대형 언어 모델을 활용하여 적절한 이미징 연구를 권장하는 프레임워크를 소개합니다. 이를 통해 evidence-based medical guidelines(증거 기반 의료 지침)에 따른 이미징 연구 주문을 지원합니다.

- **Technical Details**: 우리는 환자 'one-liner' 시나리오의 새로운 데이터셋을 제공하며, 이를 통해 최신 언어 모델을 최적화하여 의료 전문가들과 유사한 정확도를 달성합니다. 연구에서는 American College of Radiology(ACR)에서 제시한 Appropriateness Criteria(적절성 기준)에 따라 이미징 연구 주문의 정확성을 개선하는 데 초점을 두었습니다.

- **Performance Highlights**: 이 연구의 결과는 언어 모델 기반의 파이프라인이 임상의사들이 이미지 주문 워크플로를 지원하고 ACR AC에 따른 이미징 연구 주문의 정확성을 향상시킬 수 있는 지능형 도구로 사용될 수 있음을 보여줍니다.



### Calibrated Probabilistic Forecasts for Arbitrary Sequences (https://arxiv.org/abs/2409.19157)
- **What's New**: 본 논문은 예측 정확성을 유지하면서도 데이터가 어떻게 변하든 유효한 불확실성 추정치를 보장하는 새로운 예측 프레임워크를 제안합니다. 이는 게임 이론에서의 Blackwell 접근성 개념을 응용해 구축되었습니다.

- **Technical Details**: 우리는 예측 작업을 Forecaster와 Nature 간의 제로섬 게임으로 모델링하고, 예측의 캘리브레이션(calibration)과 예측 성능을 측정하기 위한 보상을 설계했습니다. 이 프레임워크는 일관된 확률적 예측을 제공하기 위한 수학적 절차를 제시하며, 다수의 캘리브레이션 형태를 일반적인 의사결정 과정으로 연결합니다.

- **Performance Highlights**: 본 프레임워크의 실행 결과, 에너지 시스템의 캘리브레이션 개선과 더불어, 각기 다른 경과 예측을 위한 의사결정 과정의 효율성이 증가하는 등의 성과를 나타냈습니다. 특히, ORCA라는 새로운 그라디언트 기반 알고리즘을 도입하여 기존의 예측기를 재캘리브레이션하는 과정에서 더 나은 성능을 달성하였습니다.



### Physics-Informed Echo State Networks for Modeling Controllable Dynamical Systems (https://arxiv.org/abs/2409.19140)
- **What's New**: 이번 연구에서는 물리학 법칙을 포함하여 훈련한 Physics-Informed Echo State Networks (PI-ESN)에 외부 입력을 추가하여 제어 가능한 비선형 동적 시스템을 모델링하는 방법을 제안합니다. 이로 인해 PI-ESN은 데이터가 제한된 상황에서도 뛰어난 성능을 발휘할 수 있습니다.

- **Technical Details**: PI-ESN은 Ordinary Differential Equations (ODEs) 시스템에 기반하여 훈련되며, 잔여 회귀항과 물리 학습 항의 기여를 균형 있게 조정하기 위해 자기 적응형 균형 손실 방법을 적용합니다. 이 방법은 Van der Pol 진동기와 사변탱크 시스템 등 다양한 비선형 시스템에서 테스트되었습니다.

- **Performance Highlights**: 제안된 PI-ESN-a는 기존 ESN과 비교하여 테스트 오류를 최대 92%까지 상대적으로 줄이는 성능 향상을 보였습니다. 특히, 훈련 데이터가 제한된 상황에서도 높은 일반화 오류를 개선할 수 있음이 확인되었습니다.



### Sequencing the Neurome: Towards Scalable Exact Parameter Reconstruction of Black-Box Neural Networks (https://arxiv.org/abs/2409.19138)
- **What's New**: 이 논문에서는 쿼리 접근만으로 신경망(neural network)의 정확한 매개변수를 추론하는 문제를 다룹니다. 이는 NP-Hard 문제로 몇 가지 기존의 알고리즘이 존재하지만 실용적인 방법이 부족합니다. 저자들은 무작위 초기화 및 1차 최적화(first order optimization)를 통해 매개변수 공간을 줄이고, 최대한 정보가 풍부한 쿼리 생성을 통해 비선형 관계를 효과적으로 풀어나가는 새로운 접근 방식을 제시했습니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 인사이트를 바탕으로 합니다. 첫 번째는 대다수의 신경망이 무작위 초기화 및 최적화를 통해 생성된다는 점이며, 이는 실제적인 매개변수 공간을 대폭 줄입니다. 두 번째는 효율적으로 비선형 관계를 풀어내기 위한 새로운 쿼리 생성 알고리즘을 제시했다는 점입니다. 이 방법을 통해 150만 개 이상의 매개변수를 가진 숨겨진 신경망과 7층 깊이의 네트워크를 재구성함으로써 높은 정확도를 보였습니다. 또한, 평가를 위해 세 가지 유형의 동형(isomorphism)을 고려했습니다: 순열(permutations), 스케일링(scaling), 극성(polarity).

- **Performance Highlights**: 그 결과 이들은 가장 큰 및 깊은 네트워크 재구성을 성공적으로 수행하였고, 최대 매개변수 차이는 0.0001 미만으로 우수한 성능을 자랑합니다. 평가 결과는 다양한 아키텍처와 데이터셋, 학습 절차를 통해 복원 방법의 견고함과 확장성을 입증하였습니다. 다양한 최적화 방법과 에포크 수를 다양화하여 실험을 진행한 결과, 저자들은 높은 샘플 효율성을 기록하며 기존 방법들과의 비교에서도 경쟁력을 보였습니다.



### Kinematic Detection of Anomalies in Human Trajectory Data (https://arxiv.org/abs/2409.19136)
- **What's New**: 이 논문은 기존의 위치 기반 연구와는 달리, 개인이 이동하는 방식, 즉 운동 패턴(kinematic features)에 중점을 두고 이를 활용하여 개인 식별(individual identification) 및 이상 탐지(anomaly detection)에 대한 가능성을 탐구한다.

- **Technical Details**: 우리는 Geolife 데이터셋을 사용하여 개별 사용자의 kinematic profile을 분석하였다. 이 kinematic profile은 개인의 이동 방식과 관련된 특징을 포함한다. 10개의 kinematic feature을 추출하고 이를 통해 각 트립(trip)을 분석한다. 높은 속도와 비현실적인 이동 패턴과 같은 이상치를 제거하여 정확한 데이터 처리를 보장한다.

- **Performance Highlights**: 단순한 kinematic feature을 표준 분류(classification) 및 이상 탐지 알고리즘에 적용함으로써 개인 식별 및 이상 탐지의 성능을 크게 향상시킬 수 있음을 실험적으로 보여주었다.



### Chebyshev Feature Neural Network for Accurate Function Approximation (https://arxiv.org/abs/2409.19135)
- **What's New**: 본 논문에서는 Chebyshev Feature Neural Network (CFNN)이라는 새로운 딥 신경망 아키텍처를 소개합니다. 이 아키텍처는 학습 가능한 주파수를 가진 Chebyshev 함수를 첫 번째 은닉층에 적용하여 함수 근사를 기계 정밀도까지 수행할 수 있습니다. 또한, 다중 단계 훈련 전략을 결합하여 훈련 과정에서 기계 정밀도에 도달할 수 있는 가능성을 보여줍니다.

- **Technical Details**: CFNN은 Chebyshev 기능을 첫 번째 은닉층에서 사용하여, 표준 완전 연결 은닉층 다음에 시퀀스를 구성합니다. Chebyshev 주파수의 초기값은 다양한 주파수를 커버하기 위해 지수 분포로 초기화되며, 이는 주파수 매개변수를 선택하는 것을 수월하게 합니다. 본 연구에서는 최대  20차원 문제에 대한 광범위한 수치 실험을 제공하여 CFNN의 효율성과 확장성을 입증합니다.

- **Performance Highlights**: CFNN은 부드러운 함수 및 불연속 함수의 근사에서 탁월한 능력을 보여줍니다. 실험 결과, CFNN은 다양한 차원에서 고품질의 함수 근사를 달성할 수 있으며, 특히 과학적 계산 문제에서 고정밀 근사가 요구되는 경우에 효과적입니다.



### Range-aware Positional Encoding via High-order Pretraining: Theory and Practic (https://arxiv.org/abs/2409.19117)
- **What's New**: 본 논문에서는 다양한 그래프 도메인에서의 지식 이전 능력을 향상시키기 위해 다중 해상도 구조 정보를 모델링하는 새로운 그래프 사전 훈련(pre-training) 전략을 제안합니다. 이 접근법은 Wavelet 신호를 기반으로 하여 노드 연결성을 재구성하는 High-Order Permutation-Equivariant Autoencoder(HOPE-WavePE)를 훈련합니다.

- **Technical Details**: 기존의 그래프 신경망(GNN)은 메시지 전달(framework) 구조에 기반하여 데이터를 처리하지만, 본 연구에서는 Wavelet 변환을 통한 다중 해상도 분석을 활용하여 구조적 정보를 포괄적으로 캡처합니다. HOPE-WavePE는 노드 간의 다단계 상호작용을 반영하는 높은 차원의 구조적 특징을 학습합니다.

- **Performance Highlights**: HOPE-WavePE는 다양한 분야의 그래프 수준 예측 작업에서 기존 방법보다 우수한 성능을 나타내며, 전이 학습(transfer learning) 가능성을 통해 도메인 특화 데이터셋에서도 효과적으로 일반화할 수 있음을 확인했습니다.



### Implementing LLMs in industrial process modeling: Addressing Categorical Variables (https://arxiv.org/abs/2409.19097)
- **What's New**: 이 연구는 Large Language Models (LLMs)를 활용하여 산업 공정의 categorical 변수에 대한 의미 있는 embedding을 도출하는 새로운 접근 방식을 제시합니다. 기존의 one-hot encoding 방식과는 달리, LLMs는 categorical 데이터의 실제 의미와 카테고리 간의 상대적 거리(유사도)를 반영하는 임베딩을 생성합니다. 이러한 접근 방식은 의미 있는 저차원 feature space를 형성하고, 공정 모델링의 성능을 개선하는 데 기여합니다.

- **Technical Details**: 제안된 방법에서는 LLM을 사용하여 각 카테고리의 텍스트 설명을 입력으로 받아 embedding을 생성합니다. 이 후, Principal Components Analysis (PCA) 또는 Uniform Manifold Approximation and Projection (UMAP) 같은 차원 축소 기법을 통해 생성된 임베딩을 저차원 공간으로 압축합니다. 이를 tree-based regression 모델에 입력하여 공정의 품질 특성을 예측합니다.

- **Performance Highlights**: LLMs를 사용한 임베딩 접근은 기존의 one-hot encoding 방식보다 더 나은 feature 중요도 분석 결과를 도출합니다. 이를 통해 공정 입력 변수의 중요성을 평가하고, 물리적 의미를 이해하는 데 도움을 줍니다. 연구는 실제 생산 데이터를 활용하여 이 접근 방식의 효과를 입증하였습니다. 특히, Shapley 값을 사용한 후속 feature 중요도 분석은 기존의 이진 인코딩 방식에 비해 큰 개선을 보여주었습니다.



### Enhancing Robustness of Graph Neural Networks through p-Laplacian (https://arxiv.org/abs/2409.19096)
Comments:
          5 pages, 2 figures

- **What's New**: 이 논문은 Graph Neural Networks (GNNs)를 손상된 그래프에서 정화하는 새로운 방법인 pLapGNN을 제안합니다. 기존의 방법들은 높은 계산 비용과 공격의 강도가 증가할 때 성능 저하 문제를 겪고 있습니다.

- **Technical Details**: 제안된 pLapGNN은 가중치 p-Laplacian에 기반하여 GNN을 공격으로부터 강화하는 프레임워크입니다. 이 방법은 그래프의 복잡한 관계를 이해하는 데 유용한 새로운 알고리즘을 제공합니다.

- **Performance Highlights**: 실제 데이터셋에 대한 실험을 통해, pLapGNN은 기존의 기준선 모델과 비교했을 때 효과성과 효율성을 입증하였습니다.



### Federated Online Prediction from Experts with Differential Privacy: Separations and Regret Speed-ups (https://arxiv.org/abs/2409.19092)
Comments:
          Accepted to NeurIPS 2024

- **What's New**: 이 논문은 전문가들로부터의 차등 개인 정보 보호를 고려한 연합 온라인 예측 문제를 다룹니다. 이 연구는 확률적 적대자 및 무관심 적대자에 대한 예측을 위해 최적화된 알고리즘을 개발하여 차별화된 성능을 보여줍니다.

- **Technical Details**: 연합 학습(FL) 프레임워크에서 다양한 클라이언트들이 모델을 학습하도록 설계하였습니다. 본 연구는 Fed-DP-OPE-Stoch 알고리즘을 도입해 m명의 클라이언트에 대해 평균 회귀를 최소화하고, 통신 비용을 로그 수준으로 유지함니다. 무관심 적대자에 대해서는, Fed-SVT 알고리즘을 통해 근접 최적의 결과를 달성할 수 있음을 보여줍니다.

- **Performance Highlights**: 제안하는 알고리즘 Fed-DP-OPE-Stoch는 클라이언트당 평균 회귀를 1/√m 배 줄이는 성과를 보였으며, Fed-SVT는 한 클라이언트 모델과 비교하여 m배의 성능 향상을 달성하였습니다. 실험 결과는 알고리즘의 효율성을 확인해줍니다.



### Differential privacy for protecting patient data in speech disorder detection using deep learning (https://arxiv.org/abs/2409.19078)
- **What's New**: 이 연구는 병리학적 (pathological) 음성 데이터에서의 차별적 프라이버시 (differential privacy, DP) 적용의 영향을 최초로 조사하였습니다. 이는 개인 정보 보호 (privacy)와 진단 정확도 (diagnostic accuracy), 공정성 (fairness) 사이의 균형을 탐구합니다.

- **Technical Details**: 본 연구는 2,839명의 독일어 사용 참여자로부터 수집된 200시간의 실제 음성 데이터셋을 사용하였으며, DP의 개인정보 보호 예산 (privacy budget)인 {
\epsilon} = 7.51을 적용했을 때 최대 정확도 감소가 3.85%에 달한다는 사실을 확인했습니다. 또한, 스페인어를 사용하는 파킨슨병 환자를 대상으로 한 소규모 데이터셋에서도 DP 제약 하에서의 모델 정확도 유지 또는 향상을 입증했습니다.

- **Performance Highlights**: 연구 결과, DP는 음성 장애 감지에서 개인 정보 보호와 유용성 (utility) 간의 효과적인 균형을 이룰 수 있음을 보여주었습니다. 그러나 음성 분야에서의 독특한 도전 과제와 개인 정보 보호-공정성 간의 균형을 강조했습니다.



### Localizing Memorization in SSL Vision Encoders (https://arxiv.org/abs/2409.19069)
Comments:
          Accepted at NeurIPS 2024

- **What's New**: 이 논문에서는 자기지도학습(SSL) 인코더에서의 메모리화(memorization)를 분석하기 위한 새로운 접근 방식을 제안합니다. 두 가지 새로운 메트릭인 LayerMem과 UnitMem을 통해 SSL 인코더의 여러 층과 단위에서 메모리화를 국소화(localization)할 수 있습니다.

- **Technical Details**: LayerMem은 SSL 인코더 내의 메모리화 위치를 층(layer) 단위로 국소화하고, UnitMem은 개별 데이터 포인트에 대한 각 단위(유닛)별 메모리화를 측정합니다. 이 두 메트릭은 다운스트림 작업과 독립적이며, 레이블(레이블 정보) 없이도 계산이 가능하여 효율적입니다.

- **Performance Highlights**: 본 연구를 통해 다음의 주요 발견이 있었습니다: 1) SSL 메모리화는 모든 인코더에서 발생하며, 심층 층이 아닌 다양한 층에서 고르게 분포되어 있다. 2) SSL 인코더의 상당수 유닛이 높은 메모리화를 경험한다. 3) 비정형 데이터 포인트는 정형 데이터 포인트보다 높은 메모리화를 유발한다. 4) 비전 변환기에서 메모리화는 전결합층(fully-connected layers)에서 주로 발생한다. 이 연구는 메모리화 국소화가 모델의 파인튜닝(fine-tuning) 및 가지치기(pruning) 전략에 실질적인 혜택을 줄 수 있음을 보여줍니다.



### CLLMate: A Multimodal LLM for Weather and Climate Events Forecasting (https://arxiv.org/abs/2409.19058)
- **What's New**: 이번 연구에서 제안된 Weather and Climate Event Forecasting (WCEF) 작업은 과거 날씨 및 기후 이벤트와 기상 데이터를 연계하여 날씨 및 기후 사건을 예측하는 새로운 과제로, 기존의 닫힌 집합 이벤트 예측 방식의 한계를 극복하는 데 중점을 두고 있습니다.

- **Technical Details**: WCEF 작업은 기상 레스터 데이터(meteorological raster data)와 사건 텍스트(data)를 통합하여 기상 사건을 예측하는 것을 목표로 합니다. 이를 위해 Llama3와 같은 대형 언어 모델(LLM)을 활용하여 과거의 기상 데이터를 사건으로 정리한 지식 그래프(knowledge graph)를 구성하였고, 이에 기반하여 최초의 다중 모달 지침 데이터셋(multimodal instruction dataset)을 생성하였습니다. 나아가 CLLMate라는 다중 모달 LLM을 통해 날씨 예측을 위한 기상 레스터 데이터를 효과적으로 관리하고 있습니다.

- **Performance Highlights**: CLLMate 모델은 기존의 기준 모델(baselines) 및 다른 다중 모달 LLM과 비교하여 뛰어난 성능을 보였으며, 기상 데이터와 사건 데이터를 효과적으로 정렬 및 통합하여 개방형 사건 예측을 단순화했습니다. 연구 결과는 WCEF 작업에 대한 연구의 미래가 가능하다는 점을 강조하며, 이 모델은 기후 위험 완화 시스템의 중요한 구성 요소로 자리 잡을 수 있는 잠재력을 보여줍니다.



### Continuously Improving Mobile Manipulation with Autonomous Real-World RL (https://arxiv.org/abs/2409.20568)
Comments:
          CoRL 2024. Website at this https URL

- **What's New**: 우리는 사람의 감독이나 광범위한 장비 없이 정책을 학습할 수 있는 완전 자율 모바일 조작(Manipulation) RL(강화학습) 프레임워크를 제안합니다. 이 접근 방식은 작업 관련 자율성(task-relevant autonomy), 효율적인 정책 학습 및 보상을 형성하는 방식이 결합되어 있습니다.

- **Technical Details**: 우리의 방법은 Spot 로봇이 4가지 최첨단 모바일 조작 업무에 대해 평균 80% 성공률을 보이는 것을 입증했습니다. 이 방법은 기존 방법에 비해 3-4배 향상된 성능을 보여줍니다. 주요 구성 요소로는 이미지 인코더, 벡터 관측치, 객체 탐지 모델 등이 포함됩니다.

- **Performance Highlights**: 우리는 각 태스크에 대해 수행된 실험 결과를 소개합니다. 의자 이동, 쓰레기통 세우기, 쓰기 작업 등 다양한 작업에서 성공률 80%를 기록했습니다. 각 작업에서 특정 임무에 대한 보상이 계산되며, 이 보상의 변화로 인해 로봇의 성능이 기하급수적으로 향상됩니다.



### MM1.5: Methods, Analysis & Insights from Multimodal LLM Fine-tuning (https://arxiv.org/abs/2409.20566)
- **What's New**: MM1.5라는 새로운 멀티모달 대형 언어 모델(Multimodal Large Language Models) 패밀리를 소개합니다. 이 모델은 텍스트가 풍부한 이미지 이해, 비주얼 레퍼링(Visual Referring) 및 그라운딩(Grounding), 여러 이미지에 대한 추론 기능을 향상시키기 위해 설계되었습니다.

- **Technical Details**: MM1.5는 데이터 중심(Data-Centric) 접근 방식을 채택하여 모델 훈련 과정 전반에 걸쳐 다양한 데이터 혼합의 영향을 체계적으로 탐구합니다. 이는 고품질 OCR 데이터와 합성 캡션(Synthetic Captions)을 포함한 지속적인 사전 훈련(Continual Pre-Training) 및 최적화된 비주얼 인스트럭션 튜닝 데이터 혼합을 통한 감독적 미세 조정(Supervised Fine-Tuning)을 포함합니다. 모델은 1B부터 30B까지의 매개변수(Parameter)를 가지며, 밀집(Dense) 및 전문가 혼합(Mixture-of-Experts, MoE) 변형이 포함됩니다.

- **Performance Highlights**: 신중한 데이터 큐레이션(Data Curation) 및 훈련 전략이 작은 규모(1B 및 3B)의 모델에서도 강력한 성능을 내는 것을 입증했습니다. 또한 비디오 이해를 위한 MM1.5-Video 및 모바일 UI 이해를 위한 MM1.5-UI라는 두 가지 특수 변형을 도입합니다.



### SpaceMesh: A Continuous Representation for Learning Manifold Surface Meshes (https://arxiv.org/abs/2409.20562)
Comments:
          published at SIGGRAPH Asia 2024

- **What's New**: 이번 연구에서는 신경망(neural network)의 출력을 복잡한 연결성의 다각형 메시(polygonal meshes)를 직접 생성하는 새로운 방식을 제안합니다. 이는 메시에 대한 연속적인 잠재 연결 공간(latent connectivity space)을 정의하여 이산 메시(discrete mesh)를 암시적으로 생성합니다.

- **Technical Details**: 새로운 표현법인 SpaceMesh는 연속적인 임베딩(embedding)을 사용하여 메시에 대한 복잡한 연결성을 지원합니다. 이 방식은 halfedge 데이터 구조(halfedge data structure)을 기반으로 하며, 이는 기본적으로 다면체 연결성을 보장합니다. 각 메시 정점(vertex)마다 저차원(low-dimensional)의 임베딩을 통해 메시에 필요한 구조적 연결성(edge adjacency and next relationships)을 정의합니다.

- **Performance Highlights**: 이 모델은 대규모 데이터셋에서 다양하고 질 높은 메시를 생성하며, 메시 수리(mesh repair)와 같은 기하학적 처리 작업을 직접 학습하는 능력을 보유하고 있습니다. 이 방식을 통해 고품질 출력(generation)의 생성과 함께 학습 작업에서 매우 빠른 수렴 속도를 달성합니다.



### LaMMA-P: Generalizable Multi-Agent Long-Horizon Task Allocation and Planning with LM-Driven PDDL Planner (https://arxiv.org/abs/2409.20560)
Comments:
          Project website: this https URL

- **What's New**: 이 논문에서는 LaMMA-P라는 새로운 다중 에이전트 (Multi-Agent) 과업 계획 프레임워크를 제안합니다. 이는 언어 모델 (LMs)과 전통적인 탐색 계획기 (search planner)의 강점을 통합하여 긴 기간의 작업을 효과적으로 처리하는 방법을 제시합니다.

- **Technical Details**: LaMMA-P는 Planning Domain Definition Language (PDDL)와 대형 언어 모델 (LLMs)의 추론 능력을 결합하여 다중 로봇 시스템에서 긴 기간 과업 할당 및 실행을 용이하게 합니다. LLM의 구성 요소는 각 로봇의 기술을 기반으로 하여 하위 작업을 식별하고 할당하며, 각 로봇의 도메인에 대한 PDDL 문제 설명을 생성합니다. 또한 Fast Downward 계획기를 사용하여 초기 계획이 실패할 경우, LLM이 계획을 재생성하고 조정하여 실행 가능한 솔루션을 생성합니다.

- **Performance Highlights**: 실험 결과, LaMMA-P는 기존 LM 기반 다중 에이전트 계획기보다 105% 높은 성공률과 36% 높은 효율성을 보여주었습니다. 더불어 MAT-THOR라는 종합 벤치마크를 통해 다양한 복잡성의 가정 작업을 포함한 성능 평가를 수행했습니다.



### Annealing Flow Generative Model Towards Sampling High-Dimensional and Multi-Modal Distributions (https://arxiv.org/abs/2409.20547)
- **What's New**: 본 논문에서는 고차원, 다중 모드 분포에서 샘플링하는 데 효과적인 새로운 접근 방식인 Annealing Flow (AF)를 제안합니다. AF는 연속 정규화 흐름 기반의 이동 맵을 학습하여 쉽고 샘플링 가능한 분포에서 목표 분포로의 변환을 가능하게 합니다.

- **Technical Details**: AF는 온도 감소를 통해 샘플링을 용이하게 하며, 이를 통해 고차원 공간에서 효율적으로 모드를 탐색합니다. 이 방법은 목표 분포의 샘플에 의존하지 않으며, 따라서 느린 혼합 (mixing) 시간, 샘플 상관관계 등 기존 방법의 한계를 극복합니다. AF는 샘플 크기와 차원에 대해 선형 복잡성을 보장합니다.

- **Performance Highlights**: AF는 다양한 어려운 분포와 실제 데이터 세트에서 광범위한 실험을 통해 최신 방법에 비해 우수한 성능을 입증합니다. 본 연구는 AF가 불리한 분포 샘플링에 대한 잠재력을 가지고 있음을 강조합니다.



### Scaling Proprioceptive-Visual Learning with Heterogeneous Pre-trained Transformers (https://arxiv.org/abs/2409.20537)
Comments:
          See the project website (this https URL) for code and videos

- **What's New**: 이 논문은 이질적인 (heterogeneous) 로봇 데이터에서 다수의 임무와 구현을 통해 정책 표현을 학습하는 방법을 제안합니다. 특히 Heterogeneous Pre-trained Transformers (HPT) 아키텍처를 통해 로봇 데이터의 다양한 하드웨어와 상황에 관계없이 일반화 가능한 정책을 학습할 수 있는 가능성을 보여줍니다.

- **Technical Details**: HPT 아키텍처는 로봇의 proprioception과 비전 입력을 단일 짧은 토큰 시퀀스로 병합하여 다양한 임무에 대해 로봇을 제어할 수 있도록 매핑합니다. 이 구조는 52개 데이터셋을 이용한 스케일링 실험을 통해 모델의 무게가 10억 개 이상의 파라미터에 달함을 확인하며, 다양한 모터 반응과 감지 시그널을 바탕으로 인간의 피드백 루프를 모방합니다.

- **Performance Highlights**: HPT는 여러 가지 기준선보다 성능이 우수하며, 여러 시뮬레이터 벤치마크와 실제 세계 환경에서 보지 못한 임무에 대해 20% 이상의 성능 향상을 보여줍니다. 이 연구는 다양한 데이터 요구 사항 없이도 새로운 임무와 구현에 대한 로봇 정책을 쉽게 구축할 수 있도록 하는 기초 모델의 중요성을 강조합니다.



### Dual Encoder GAN Inversion for High-Fidelity 3D Head Reconstruction from Single Images (https://arxiv.org/abs/2409.20530)
Comments:
          Joint first two authors. Accepted to NeurIPS 2024

- **What's New**: 이번 연구에서는 PanoHead라는 새로운 프레임워크를 기반으로 한 3D GAN inversion 방법을 제안하였습니다. 이전의 연구들이 주로 EG3D에 의존했던 반면, PanoHead는 360도 관점에서 이미지를 합성하는 데 최적화되어 있습니다. 또한, 두 개의 인코더를 결합한 시스템으로 고화질 복원과 현실적인 생성이 가능하다는 점이 특징입니다.

- **Technical Details**: 연구에서는 홀적으로 보이는 3D 기하 구조를 복원하기 위해 듀얼 인코더 시스템을 도입하였습니다. 한 인코더는 주어진 뷰에 대한 고 충실도의 복원에 집중하고, 다른 인코더는 보이지 않는 뷰의 고품질 생성에 중점을 두고 훈련됩니다. 또한, 두 인코더의 출력을 원활하게 결합하기 위해 triplane 도메인이 사용되며, occlusion-aware triplane discriminator를 기반으로 한 새로운 손실 함수를 통해 일관된 결과를 도출하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 기존 인코더 훈련 방법들보다 정성적 및 정량적으로 우수한 성능을 보여주었습니다. 특히, 입력 이미지에 대한 고충실도 재구성과 보이지 않는 부분에 대한 현실적인 표현을 동시에 달성하는 데 성공하였습니다.



### Formally Verified Physics-Informed Neural Control Lyapunov Functions (https://arxiv.org/abs/2409.20528)
- **What's New**: 이번 연구에서는 비선형 시스템을 위한 제어 리아푸노프 함수(Control Lyapunov Functions, CLFs)의 설계 및 분석에 있어 신경망 기반의 물리 정보 학습(physics-informed learning)과 형식적 검증(formal verification)을 활용한 혁신적인 접근법을 제시합니다.

- **Technical Details**: 제안된 방법은 포트리야긴 최대 원리(Pontryagin's Maximum Principle, PMP)를 이용한 궤적 최적화와 물리 정보 학습을 결합하여 신경망 CLF를 계산하며, 이 CLF는 SMT(solvers for Satisfiability Modulo Theories) 해결기를 사용해 형식적으로 검증됩니다. 이는 전통적인 수학적 방법들보다 우수한 성능을 보임을 수치 예제로 증명합니다.

- **Performance Highlights**: 정량적인 결과에 따르면, 제안된 신경망 CLF 방법은 sum-of-squares(SOS) 및 유리적 CLF보다 뛰어난 성능을 보이며, 형식적 검증을 통해 신뢰성을 높이고 글로벌 비제어 가능성(null-controllability) 인증을 효율적으로 생성할 수 있습니다.



### Accelerating Non-Maximum Suppression: A Graph Theory Perspectiv (https://arxiv.org/abs/2409.20520)
- **What's New**: 본 논문은 객체 탐지에서 필수적인 후처리 단계인 Non-maximum suppression (NMS)을 그래프 이론 관점에서 처음으로 체계적으로 분석하였습니다. 이를 통해 NMS의 내재적 구조를 밝혀내고 QSI-NMS와 BOE-NMS 두 가지 최적화 방법을 제안합니다.

- **Technical Details**: QSI-NMS는 빠른 재귀적 분할 정복 알고리즘으로, mAP 손실이 거의 없으며, 확장된 버전 eQSI-NMS는 최적 복잡성인 \mathcal{O}(n\log n)을 달성합니다. BOE-NMS는 NMS의 지역성(locality)을 활용하여 mAP 손실 없이 일정 수준의 최적화를 이룹니다. 또한 NMS-Bench라는 첫 번째 벤치마크를 소개하여 다양한 NMS 방법을 종합적으로 평가합니다.

- **Performance Highlights**: YOLOv8-N 모델을 기준으로 하여 QSI-NMS는 원래 NMS에 비해 6.2배의 속도를 제공하며, mAP는 0.1% 감소합니다. 최적의 eQSI-NMS는 0.3% mAP 감소로 10.7배의 속도를 달성하고, BOE-NMS는 mAP 손실 없이 5.1배의 속도를 기록합니다.



### Ensemble WSINDy for Data Driven Discovery of Governing Equations from Laser-based Full-field Measurements (https://arxiv.org/abs/2409.20510)
Comments:
          25 pages, 10 figures

- **What's New**: 본 연구는 레이저 진동 측정법(laser vibrometry)과 비선형 동역학의 희소 식별(WSINDy) 방법론을 활용하여 실험 데이터를 통한 거시적 지배 방정식(macro-governing equations)을 학습하는 방법을 제안합니다. 이 과정에서 초저주파 영역의 전단파(shear wave) 자극을 주어 탐색한 두 가지 재료(알루미늄 및 IDOX/Estane 복합재)의 응답을 분석합니다.

- **Technical Details**: WSINDy for PDEs(Partial Differential Equations) 알고리즘을 사용하여 실험에서 얻은 시공간(spatio-temporal) 데이터로부터 유효한 동역학을 발견합니다. 이 과정에서 발견된 PDE는 유서 깊은 오일러-베르누이 빔(Euler-Bernoulli beam) 모델 형태를 띠며, 이를 통해 두 재료의 영률(Young's modulus)을 추정합니다. 또한, 알고리즘의 집합적 버전을 활용하여 PDE 계수 및 영률의 불확실성에 대한 정보를 제공합니다.

- **Performance Highlights**: WSINDy 방법을 통해 실험 데이터를 바탕으로 해석 가능한 방정식을 성공적으로 발견하였으며, 발견된 방정식을 유한 요소 코드(finite element code)와 비교하여 실험 데이터와의 일치성을 검증하였습니다. 또한, 실험 데이터 및 복합 재료의 메커니즘에 대한 새로운 통찰을 제공하는 등, 본 방법론은 비파괴(non-destructive) 실험법을 통해 미지의 지배 방정식을 학습하고 기계 시스템에 대한 인사이트를 확보하는 데 매우 효율적임을 보여주었습니다.



### What Information Contributes to Log-based Anomaly Detection? Insights from a Configurable Transformer-Based Approach (https://arxiv.org/abs/2409.20503)
Comments:
          23 pages

- **What's New**: 본 연구에서는 log 데이터에서 의미(the semantic), 순서(the sequential), 시간(the temporal) 정보를 포착할 수 있는 구성 가능(configurable) Transformer 기반(anomaly detection) 이상 탐지 모델을 제안합니다.

- **Technical Details**: 이 모델은 다양한 길이의 log 시퀀스를 사용하여 학습 및 평가가 가능하며, 기존 방법들의 고정 길이(log sequences) 또는 시간 창(time-windowed logs)의 제약을 극복합니다. 여러 입력 특성의 조합을 실험하여 이상 탐지에서 각 정보 유형의 역할을 평가합니다.

- **Performance Highlights**: 모델은 다양한 길이의 log 시퀀스를 처리할 때 안정적이고 경쟁력 있는 성능을 보여주며, 이벤트 발생(event occurrence) 정보가 이상 식별에 중요한 역할을 하는 반면, 순서 정보와 시간 정보는 연구된 공개 데이터셋에서 이상 탐지에 중요한 영향을 미치지 않는 것으로 나타났습니다.



### RecSys Challenge 2024: Balancing Accuracy and Editorial Values in News Recommendations (https://arxiv.org/abs/2409.20483)
Comments:
          5 pages, 3 tables, RecSys' 24

- **What's New**: RecSys Challenge 2024는 뉴스 추천의 기술적 및 규범적 과제를 다루며, 사용자 선호를 행동 기반으로 모델링하고 뉴스 항목의 빠른 소멸을 관리하는 데 중점을 두고 있습니다.

- **Technical Details**: Ekstra Bladet와 JP/Politikens Media Group은 110만 이상의 사용자와 125,000개의 뉴스 기사를 포함하는 대규모 데이터셋을 제공하며, 다양한 메트릭(AUC, MRR, nDCG)을 사용하여 추천 시스템을 평가합니다.

- **Performance Highlights**: 참가자들은 사용자의 클릭 기록, 세션 세부사항, 사용자 메타데이터를 바탕으로 뉴스 기사를 기초로 순위 매기기를 수행하며, 이번 대회는 다양한 추천 시스템의 뉴스 흐름에 대한 영향을 평가하는 데 중점을 두고 있습니다.



### Multilevel Picard approximations and deep neural networks with ReLU, leaky ReLU, and softplus activation overcome the curse of dimensionality when approximating semilinear parabolic partial differential equations in $L^p$-sens (https://arxiv.org/abs/2409.20431)
- **What's New**: 이번 논문에서는 multilevel Picard 근사법과 ReLU, leaky ReLU, 소프트플러스(softplus) 활성화 함수를 가진 심층 신경망(deep neural networks)이 세미선형 Kolmogorov PDEs의 해를 $L^	ext{p}$-sense에서 근사할 수 있음을 증명합니다. 이를 통해 비구배 독립적(gradient-independent)이며 Lipschitz 연속(nonlinearities)을 가진 비선형 함수의 경우에 대해서도 가능함을 보여줍니다.

- **Technical Details**: 세미선형 Kolmogorov PDEs에 대한 근사에는 multilevel Picard 근사법이 사용되며, 신경망의 활성화 함수로는 ReLU, leaky ReLU 및 softplus가 포함됩니다. 이 방법들은 치수(dimension) $d	ext{와}$ 주어진 정확도 $	ext{ε}$의 역수에 대해 다항식(polynomial)으로 성장하는 계산 비용(computational effort) 및 신경망의 매개변수(parameter) 수를 요구합니다.

- **Performance Highlights**: 제안된 방법은 높은 정확도로 Kolmogorov PDEs의 해를 근사할 수 있으며, 이는 다양한 차원과 정확도 요구사항에 대해 효율적으로 수행될 수 있음을 나타냅니다.



### Sufficient and Necessary Explanations (and What Lies in Between) (https://arxiv.org/abs/2409.20427)
- **What's New**: 이 연구는 머신러닝 모델의 예측을 설명하기 위해 필요성과 충분성의 두 가지 특성 중요성 개념을 정립하고, 이 두 개념을 통합한 새로운 중요성 개념을 제안합니다. 이는 머신러닝의 설명 가능성을 높이기 위한 첫 단계로, 이전 방법들로는 발견하기 어려운 중요한 피쳐를 발견하는 데 기여합니다.

- **Technical Details**: 연구는 머신러닝 예측의 피쳐 중요성을 평가하기 위해 수학적 정의와 접근법을 형식화하며, 조건부 독립성과 게임 이론적 수량(Shapley values)를 기반으로 새로운 통합 프레임워크를 제공합니다. 이를 통해 충분성과 필요성의 관계를 분석하고, 중요한 피쳐를 식별하는 방법을 제안합니다.

- **Performance Highlights**: 실험을 통해 제안된 통합 관점이 피쳐 중요성에 대한 보다 완전하고 중요한 통찰력을 제공함을 입증합니다. 이 새로운 접근법은 이전 방법들의 한계를 극복하고, 일반적인 머신러닝 모델에 대한 이해를 심화시킵니다.



### Stream-level flow matching from a Bayesian decision theoretic perspectiv (https://arxiv.org/abs/2409.20423)
- **What's New**: 이 논문에서는 Flow Matching (FM) 알고리즘의 새로운 방향을 제시하며, Gaussian Process (GP)를 기반으로 한 조건부 확률 경로를 정의하는 CFM 알고리즘의 확장을 소개합니다. 이를 통해 잠재적 확률 경로와 관련된 여러 데이터를 통합할 수 있는 가능성을 열어줍니다.

- **Technical Details**: 조건부 흐름 일치 (CFM) 훈련을 베이esian 결정 이론적 관점에서 바라보며, 잠재 변수 모델링을 통해 경로를 보강합니다. Gaussian process (GP)를 사용해 이들 잠재적 경로의 모델링을 수행하여, 조건부 확률 경로를 더욱 유연하게 확장합니다.

- **Performance Highlights**: CFM의 일반화가 추정된 분산 벡터 필드의 분산을 크게 줄이면서도 계산 비용은 적당하게 유지되는 것을 보여주며, 생성된 샘플의 품질이 향상됩니다. MNIST 및 HWD+와 같은 필기 이미지 데이터셋에서 실험으로 검증된 결과입니다.



### Novel machine learning applications at the LHC (https://arxiv.org/abs/2409.20413)
Comments:
          10 pages, 10 figures, 42nd International Conference on High Energy Physics (ICHEP 2024)

- **What's New**: 본 논문에서는 LHC 실험에서 사용할 수 있는 최신의 기계 학습(ML) 기법과 그 적용 사례를 소개합니다. 특히 분류(Classification), 빠른 시뮬레이션(Fast Simulation), 언폴딩(Unfolding), 이상 탐지(Anomaly Detection) 등의 분야에서 최근 결과를 다루고 있습니다.

- **Technical Details**: ML 기법은 jet(제트) 분류 작업에서 다양한 표현 방법을 활용하는 전개가 이루어지고 있습니다. 제트를 고수준 특징 벡터로 전처리하는 전통적인 방법을 넘어서, 제트를 시퀀스, 이미지 및 그래프 형태로 표현하여 graph neural networks(그래프 신경망)나 attention-based transformers(어텐션 기반 변환기)로 처리하는 방법이 소개되고 있습니다. 또한, Global ParT(GloParT)와 Unified Particle Transformer(UParT) 같은 새로운 알고리즘이 소개되어 정확도와 모델 robustness(강인성)를 높이고 있습니다.

- **Performance Highlights**: GloParT는 light jet 재jection의 베이스라인에 비해 4.2배 향상된 성능을 보이며, UParT는 ParticleNet 기반의 경쟁 알고리즘보다 예측 성능이 우수한 것으로 나타났습니다. 특히 ALICE 실험에서는 기계 학습을 이용한 입자 식별이 표준 방법보다 높은 순도와 효율성을 달성하고, 새로운 시스템적 인식 신경망 훈련(SANNT) 기법이 제안되었습니다.



### Accelerating PoT Quantization on Edge Devices (https://arxiv.org/abs/2409.20403)
Comments:
          Accepted at 31st IEEE International Conference on Electronics, Circuits and Systems (ICECS), 2024

- **What's New**: 본 논문에서는 Power-of-Two (PoT) 양자화 방법을 위한 비트 시프트 프로세싱 요소(shift-PE)를 설계하고, 자원 제약이 있는 엣지 장치에서의 DNN 가속을 위한 오픈소스 파이프라인인 PoTAcc를 제안합니다.

- **Technical Details**: PoT 양자화는 데이터 분포에 더 잘 적합하여 Deep Neural Networks (DNNs)의 양자화 오차를 줄입니다. 비트 시프트 연산을 사용하여 곱셈을 대체할 수 있으며, 이를 통해 하드웨어 설계를 위한 다양한 처리 요소(PE) 효율성을 평가합니다. 본 논문에서는 세 가지 PoT 양자화 방법의 shift-PE를 설계하고, 이를 이용하여 DNN 추론을 위한 shift 기반 가속기를 디자인했습니다.

- **Performance Highlights**: PoTAcc를 이용해 MobileNetV2, ResNet18, InceptionV1 세 가지 DNN 모델에 대해 평가한 결과, 기존의 곱셈 기반 가속기에 비해 평균 1.23배의 속도 향상 및 1.24배의 에너지 감소를 달성하였으며, CPU 전용 실행에 비해 2.46배의 속도 향상과 1.83배의 에너지 감소를 기록했습니다.



### CableInspect-AD: An Expert-Annotated Anomaly Detection Datas (https://arxiv.org/abs/2409.20353)
Comments:
          35 pages, to appear at NeurIPS 2024

- **What's New**: CableInspect-AD라는 고품질의 공개 데이터셋을 소개하며, 이것은 로봇 전력선 점검을 위한 Visual Anomaly Detection (VAD)의 성능 향상을 돕기 위해 설계되었다.

- **Technical Details**: 이 데이터셋은 Hydro-Québec 전문가들에 의해 제작 및 주석이 달린 고해상도 이미지로 구성되어 있으며, 다양한 결함 유형과 심각도를 포함한다. Enhanced-PatchCore라는 방법을 통해 몇몇 명목적 예제만 가지고도 탐지 기준을 설정할 수 있도록 개선되었다.

- **Performance Highlights**: Enhanced-PatchCore는 적은 데이터에 대한 성능이 유망하지만, 특정한 종류와 심각도의 결함 탐지에는 여전히 한계가 있음을 나타낸다. 이 데이터셋은 VAD 연구 커뮤니티에 도전적인 기준점으로 작용할 것으로 기대된다.



### Enhancing GANs with Contrastive Learning-Based Multistage Progressive Finetuning SNN and RL-Based External Optimization (https://arxiv.org/abs/2409.20340)
- **What's New**: 본 연구에서 제안한 프레임워크는 멀티스테이지 프로그레시브 파인튜닝 시아미즈 신경망(MFT-SNN)과 강화 학습 기반 외부 최적화기(RL-EO)를 포함하여 GAN 훈련 루프 내에서 지침을 제공함으로써 GAN의 한계를 극복하는 것을 목표로 하고 있습니다.

- **Technical Details**: 이 프레임워크는 두 가지 구성 요소로 이루어져 있습니다. 첫째, MFT-SNN은 조직병리학 패치 간의 유사성을 평가하기 위한 대조 학습 기반의 신경망입니다. 둘째, RL-EO는 GAN 훈련 루프 내에서 보상 신호 생성기로 작용하며, 수정된 판별기 손실 함수는 가중 보상을 포함하여 GAN이 보상을 극대화하면서 손실을 최소화하도록 유도합니다.

- **Performance Highlights**: 제안한 방법은 최신 GAN과 Denoising Diffusion Probabilistic 모델에 대한 벤치마크에서 FID 점수, KID 점수, 지각 경로 길이(Perceptual Path Length), 하류 분류 작업 측면에서 이전의 최첨단(SOTA)을 초월하는 성과를 보여주었습니다.



### Distributed NeRF Learning for Collaborative Multi-Robot Perception (https://arxiv.org/abs/2409.20289)
- **What's New**: 본 논문에서는 여러 로봇 에이전트가 협력하여 환경을 인식하는 다중 에이전트 퍼셉션 시스템을 제안합니다. 이 시스템은 RGB 이미지로부터 NeRF(Neural Radiance Field)를 학습하여 장면을 표현하며, 각 에이전트는 자신의 로컬 데이터를 처리하고, 학습된 NeRF 모델만을 공유함으로써 통신 오버헤드를 줄입니다.

- **Technical Details**: 제안된 방법은 각 에이전트가 로컬 센서를 사용해 수집한 데이터를 바탕으로, 중앙 서버에 모든 원시 데이터를 전송하지 않고 에이전트 간의 네트워크 가중치만을 공유합니다. 이를 통해 NeRF 오버피팅(overfitting)을 줄이면서도 여러 에이전트 간의 일관성을 보장하는 분산 학습 프레임워크를 구축합니다.

- **Performance Highlights**: 우리의 실험 결과에 따르면, 제안된 다중 에이전트 접근 방식은 제한된 시점에서 입력이 제공될 때 중앙 집중식 훈련보다 뛰어난 성능을 보였으며, 환경을 매핑하는 데 있어 중앙화된 처리 방식과 비슷한 성과를 달성했습니다. 또한, 통신 효율성이 크게 향상되었습니다.



### Leveraging CAM Algorithms for Explaining Medical Semantic Segmentation (https://arxiv.org/abs/2409.20287)
Comments:
          Accepted for publication at the Journal of Machine Learning for Biomedical Imaging (MELBA) this https URL

- **What's New**: 이번 연구에서는 CNN의 해석 가능한 인공지능(xAI) 분야에서, Seg-HiRes-Grad CAM이라는 새로운 방법을 제안합니다. 이 방법은 현재의 분류 기반 해석 방법과 분할 기반 방법 간의 전이를 통해, 의학적 이미지 분할 등의 작업에서 더욱 상세하고 일관된 해석 결과를 제공합니다.

- **Technical Details**: Seg-HiRes-Grad CAM은 Seg-Grad CAM의 확장 버전으로, 분류 기반의 HiRes CAM으로부터 전이된 접근 방식입니다. 이 방법은 기울기(gradient)를 활용한 로컬(시각적) 설명 알고리즘이며, 세그먼트 분할 작업에 최적화되어 있습니다.

- **Performance Highlights**: Seg-HiRes-Grad CAM은 기존의 Seg-Grad CAM에서 발생하는 정확도 문제를 해결하며, 특히 의학적 이미지 분할 작업에서 설명 가능성을 크게 향상시킵니다. 이로 인해 의사 결정 과정에서의 중요한 이미지 영역을 더욱 명확하게 시각화할 수 있습니다.



### Solution for OOD-CV Workshop SSB Challenge 2024 (Open-Set Recognition Track) (https://arxiv.org/abs/2409.20277)
- **What's New**: 본 논문은 ECCV 2024의 OOD-CV 워크샵에서 열린 OSR 챌린지를 위한 방법론을 소개합니다. 이 연구는 오픈셋 인식(open-set recognition, OSR) 문제를 다루며, 데이터의 분포가 다를 때도 분류 모형이 올바르게 예측할 수 있도록 하는 하이브리드 접근법을 제안합니다.

- **Technical Details**: 우리는 라벨이 없는 데이터에 대한 감지 성능을 높이기 위해 여러 개의 post-hoc OOD 탐지 기법과 Test-Time Augmentation (TTA)을 결합한 하이브리드 프레임워크를 사용했습니다. 사용한 주요 기술 중 하나는 ReAct 메소드로, 이는 신경망의 비정상 작용을 조정하여 OOD 데이터에 대한 감지 성능을 향상시킵니다. TTA는 이미지에 다양한 증강을 적용하여 모델의 일반화 능력을 강화합니다.

- **Performance Highlights**: 우리의 최종 성과는 AUROC: 79.77 (5위) 및 FPR95: 61.44 (2위)로, 전체 대회에서 2위를 차지하는 성과를 거두었습니다.



### First Order System Least Squares Neural Networks (https://arxiv.org/abs/2409.20264)
- **What's New**: 본 논문은 국소 다각형 영역에서 선형 엘립틱(linear elliptic), 패러볼릭(parabolic), 하이퍼볼릭(hyperbolic) PDEs를 심층 신경망(deep neural networks)을 사용하여 수치적으로 해결하는 개념적 프레임워크를 도입합니다.

- **Technical Details**: PDE는 등가적이고 잘 정의된 1차 시스템의 최소 제곱(residual of an equivalent, well-posed first-order system) 잔여의 최소화를 통해 재구성되며, 이는 심층 신경망의 매개변수 계열에 걸쳐 최적화됩니다. 제안된 접근법은 신경망 훈련을 위한 수치적 손실 함수로 사용되며, 적응형 LSQ 유한 요소 방법의 맥락에서 (quasi-)최적의 수치적 오차 추정기를 제공합니다.

- **Performance Highlights**: 제안된 방법은 LSQ 손실 함수의 정확한 수치적 최소화를 가정할 때 신경망의 구현이 1차 시스템 LSQ 형식의 정확한 해로 수렴하는 최적의 속도를 보장합니다.



### Controlling sharpness, SNR and SAR for 3D FSE at 7T by end-to-end learning (https://arxiv.org/abs/2409.20251)
Comments:
          Submitted to Magnetic Resonance in Medicine for peer-review

- **What's New**: 이번 연구는 7T에서 매우 긴 에코 트레인을 사용하는 3D FSE 시퀀스를 위해 포인트 스프레드 함수(PSF)와 신호 대 잡음비(SNR)를 최적화한 전용 가변 플립 앵글(VFA) 스킴을 비휴리스틱(non-heuristically)하게 식별합니다.

- **Technical Details**: 제안된 최적화는 미리 정의된 SAR 제약 조건과 목표 대비를 고려하며, 엔드 투 엔드(End-to-End) 학습 프레임워크를 사용합니다. 비용 함수는 여러 조직에 대한 대비 충실도(SNR)와 이미지 블러링(PSF)을 최소화하기 위한 패널티 항목을 통합합니다. PSF/SNR 비용 함수 구성 요소의 가중치를 조정하여 PSF 및 SNR이 최적화된 VFA를 유도하고, 이를 두 명의 자원자와 세 명의 자원자를 대상으로 7T MRI 시스템에서 테스트하였습니다.

- **Performance Highlights**: PSF 최적화된 VFA는 T2w에서 표준 VFA에 비해 이미지 블러링이 크게 감소하며, 대비 충실도를 유지합니다. PSF 최적화된 VFA를 사용하여 작은 백질 및 회색질 구조, 혈관이 더욱 뚜렷하게 나타났습니다. 정량적 분석 결과, 최적화된 VFA는 표준 VFA에 비해 sinc-유사(reference) PSF로부터 50% 적은 편차를 나타냈습니다. SNR 최적화된 VFA는 표준 VFA에 비해 백질 및 회색질 지역에서 신호 대 잡음비가 81.2±18.4에서 41.2±11.5로 개선되었습니다.



### Random Features Outperform Linear Models: Effect of Strong Input-Label Correlation in Spiked Covariance Data (https://arxiv.org/abs/2409.20250)
Comments:
          29 pages, 5 figures

- **What's New**: 이번 연구는 Random Feature Model (RFM)이 기존의 noisy linear models보다 우수한 성능을 발휘하는 조건과 원인을 규명하였습니다.

- **Technical Details**: RFM은 비선형 활성화 함수가 포함된 두 층 신경망으로, 고차원 학습에서의 학습 및 일반화 성능을 이해하는 데 중요한 역할을 합니다. 본 연구는 입력 데이터가 spiked covariance 특성을 갖는 비등방성(anisotropic) 상황에서 RFM을 분석하였습니다.

- **Performance Highlights**: 혁신적인 시뮬레이션을 통해 RFM이 입력과 레이블 간의 높은 상관관계가 있는 경우 성능 우위를 보임을 수치적으로 확인하였습니다. RFM의 성능은 noisy polynomial models에 해당하며, 다항식의 차수는 입력과 레이블 간의 상관관계 강도에 따라 달라진다고 밝혔습니다.



### A general machine learning model of aluminosilicate melt viscosity and its application to the surface properties of dry lava planets (https://arxiv.org/abs/2409.20235)
Comments:
          21 pages, 9 figures, 2 tables

- **What's New**: 이번 연구에서는 K2-141 b와 같은 초단주기(exoplanets) 외계행성의 마그마 바다(magma ocean)의 점도를 예측하기 위한 새로운 머신러닝 모델을 제안합니다. 이 모델은 다양한 조성을 포함하는 마그마의 점도 측정을 통해 학습되었습니다.

- **Technical Details**: 연구팀은 28,898개의 점도 측정을 기준으로 Greybox 인공신경망(neural network)과 가우시안 프로세스(Gaussian process)를 결합한 모델을 개발하였습니다. 이 모델은 고압(30 GPa)에서의 높은 예측 정확도를 달성하며, 다양한 마그마 조성에 대해 점도를 예측할 수 있습니다.

- **Performance Highlights**: 모델을 사용하여 K2-141 b의 마그마 바다 점도를 계산한 결과, 낮 동안은 완전히 녹아있는 상태이며, 온도 변화가 점도에 결정적인 영향을 미친다는 것을 확인했습니다. 낮은 압력에서는 기체 분위기가 형성될 수 있으나, 높은 경도에서 점도가 급격히 증가하며 반mal 단계로 변화하게 됩니다.



### Assessing interaction recovery of predicted protein-ligand poses (https://arxiv.org/abs/2409.20227)
Comments:
          12 pages, 6 figures, 1 table, code at this https URL, data at this https URL

- **What's New**: 이번 연구는 기계 학습 기반의 단백질-리간드 포즈 예측 방법에서 중요한 상호작용 지문인 protein-ligand interaction fingerprints (PLIFs)를 간과할 경우 모델 성능의 과대 평가를 초래할 수 있음을 보여줍니다. 이는 최근 단백질-리간드 공동 접힘 모델에서 특히 두드러지며, 이 모델들이 중요한 상호작용을 회복하지 못하는 경향이 있다는 점도 강조합니다.

- **Technical Details**: 이 연구에서는 PLIF를 통해 단백질 잔기와 리간드 간의 상호작용을 시각화하고, 이를 평가하는 데 있어 기계 학습 방법들이 어떤 지표를 사용할 수 있는지를 설명합니다. PLIF는 단백질-리간드 복합체의 삼차원 상호작용을 요약하며, 다양한 상호작용의 타입을 인코딩할 수 있는 벡터형 표현을 제공합니다. 실험에서는 ProLIF 패키지를 사용하여 수소 결합, 할로겐 결합, 양이온-π 결합 등의 상호작용을 분석했습니다.

- **Performance Highlights**: 이 연구에서는 다양한 현대 포즈 예측 도구의 성능을 PLIF 회복률을 기반으로 평가하였습니다. GOLD와 DiffDock-L이 기준 구조에 가까운 포즈를 생성했지만, DiffDock-L은 중요한 상호작용 일부를 놓쳤습니다. 반면 GOLD는 모든 하이드로겔 결합을 회복하였습니다. также, RoseTTAFold-AllAtom은 포즈에서 잔존 충돌을 발생시켜 성능이 더 저조하였음이 밝혀졌습니다.



### Forecasting Disease Progression with Parallel Hyperplanes in Longitudinal Retinal OC (https://arxiv.org/abs/2409.20195)
Comments:
          accepted in MICCAI 2024

- **What's New**: 최근의 연구에서, 우리는 망막 OCT 스캔을 통해 늦은 건성 나이 관련 황반변성(dAMD)의 발생 위험을 예측하기 위한 새로운 딥러닝(Deep Learning) 방법을 제안하였습니다. 이 방법은 현재 스캔을 기반으로 전환 시간과 관련된 위험 점수와 특정 시간 내 변환 확률을 예측합니다.

- **Technical Details**: 제안된 방법은 변환 시간(T*)을 랜덤 변수로 모델링하고, 이와 관련된 누적 분포 함수(CDF)를 계산합니다. 또한, 우리는 주어진 이미지 집합에 대해 위험 점수를 할당하여, 다양한 위험 집단으로 분류할 수 있는 시스템을 개발하였습니다. 이 시스템은 각 객체 간 일관성 있는 예측을 보장하는 비지도 학습 손실을 활용합니다.

- **Performance Highlights**: 2개의 대규모 데이터셋을 사용한 평가 결과, Dataset-1에서 평균 AUROC 0.82, Dataset-2에서 평균 AUROC 0.83을 달성하였습니다. 이러한 성능 지표는 다양한 스캐너에서 수집된 이미지 간 도메인 시프트를 극복하는 능력을 보여줍니다.



### ILeSiA: Interactive Learning of Situational Awareness from Camera Inpu (https://arxiv.org/abs/2409.20173)
Comments:
          7 pages, 8 figures

- **What's New**: 이 논문은 로봇에게 상황 인식을 가르치는 방법인 ILeSiA 시스템을 제안합니다. 이 시스템은 초기 기술 시연을 통해 로봇의 스킬을 학습하고, 사용자로부터 제공된 라벨(안전 또는 위험)을 사용하여 자율적으로 실행하는 과정에서 위험을 감지합니다.

- **Technical Details**: ILeSiA는 카메라 이미지를 통해 위험을 인식하며, 이미지를 저차원 잠재 공간(latent space)으로 인코딩하고, 이를 기반으로 분류기를 교육합니다. 이 과정에서 Gaussian Process (GP) 리스크 추정 모델을 사용하여 단일 시연으로도 위험 수준을 지속적으로 평가합니다. 시스템은 기존의 Learning from Demonstration (LfD) 프레임워크에 통합되어 있으며, 사용자 피드백을 통해 지속적으로 학습하고 모델을 재훈련할 수 있습니다.

- **Performance Highlights**: 실험 결과, 학습된 분류기는 사용자 제공 데이터가 적음에도 불구하고 다양한 위험을 성공적으로 감지할 수 있음을 보여줍니다. 이 시스템은 위험 사례가 라벨링됨에 따라 유연하게 동작하며, 인간 감독자가 위험을 식별함에 따라 즉시 라벨을 추가할 수 있는 장점을 갖고 있습니다.



### Machine Learning in Industrial Quality Control of Glass Bottle Prints (https://arxiv.org/abs/2409.20132)
Comments:
          VISAPP 2024 Conference

- **What's New**: 이 논문에서는 유리병 인쇄 품질 관리를 위한 두 가지 기계 학습 기반 접근 방식을 제시하고 평가하였습니다. 이러한 접근 방식은 반사와 제작 관련 편차에도 불구하고 적절한 인쇄 품질을 유지하기 위한 것입니다.

- **Technical Details**: 첫 번째 접근 방식은 Sobel 및 Canny 필터와 이미지 품질 메트릭(예: MSE, SSIM)을 사용하여 다양한 감독 분류 모델(예: SVM, k-Neighbors)과 함께 84%의 정확도를 달성했습니다. 두 번째 접근 방식은 미리 훈련된 CNN 모델(예: ResNet, VGG)을 미세 조정하여 이진 분류 작업을 수행했으며, 이 결과 87%의 정확도를 기록했습니다. Grad-CAM을 활용하여 자주 결함이 있는 인쇄 영역을 시각화했습니다.

- **Performance Highlights**: 본 연구는 병 제조 과정에서 발생할 수 있는 결함을 정확하게 감지할 수 있도록 하는 신뢰할 수 있는 품질 관리 시스템을 개발하는 데 기여하였습니다. 연구 결과, 기계 학습 기술을 통해 인쇄 품질의 최적화가 가능함을 입증하였습니다.



### Reevaluation of Inductive Link Prediction (https://arxiv.org/abs/2409.20130)
Comments:
          Published in RuleML+RR 2024

- **What's New**: 이 논문에서는 현재 사용되고 있는 유도 링크 예측(inductive link prediction) 평가 프로토콜이 중대한 결함을 가지고 있다고 주장합니다. 이 프로토콜은 무작위로 샘플링된 부정 엔티티(negative entities) 집합에서 진짜 엔티티를 순위 매기기 때문에 문제를 야기합니다.

- **Technical Details**: 논문에서는 기존의 유도 링크 예측 방법을 여러 기준 벤치마크(benchmarks)에서 재평가합니다. 일반적으로 전이 설정(transductive setting)에서 적용되는 링크 예측 프로토콜을 사용하여 평가하며, 몇몇 유도 방법들은 이 설정에서 확장성(scalability) 문제로 인해 성능이 저하됩니다. 이 문제를 해결하기 위한 개선된 샘플링 프로토콜(sampling protocol)을 제안하고 적용합니다.

- **Performance Highlights**: 우리의 평가 결과는 지금까지 보고된 결과들과 크게 다릅니다. 간단한 규칙 기반(baseline) 모델이 유형의 유효성(validity)에 따라 엔티티를 더 높은 순위로 매김으로써 최첨단(state-of-the-art) 결과를 달성할 수 있습니다.



### Training a Computer Vision Model for Commercial Bakeries with Primarily Synthetic Images (https://arxiv.org/abs/2409.20122)
Comments:
          FZI Workshop - Künstliche Intelligenz im Mittelstand (KI-KMU 2024)

- **What's New**: 이 연구는 식품 산업에서 재처리된 제품의 재고 관리를 자동화하기 위한 AI 어플리케이션을 발전시켰습니다. 2432개의 이미지를 포함한 확장된 데이터셋을 만들었으며, 새로운 구운 식품을 포함시켰습니다.

- **Technical Details**: 이 연구에서는 generative models인 pix2pix와 CycleGAN을 사용하여 합성 이미지를 생성하여 모델의 강인성을 높였습니다. YOLOv9와 YOLOv8이라는 최신 object detection 모델을 훈련하여 구운 식품 탐지 작업을 수행했습니다.

- **Performance Highlights**: 최종 모델은 테스트 세트에서 90.3%의 평균 정밀도(AP@0.5)를 달성하여 뛰어난 성능을 보였습니다.



### Inferring Thunderstorm Occurrence from Vertical Profiles of Convection-Permitting Simulations: Physical Insights from a Physical Deep Learning Mod (https://arxiv.org/abs/2409.20087)
Comments:
          14 pages, 8 figures, 2 tables. This work has been submitted to Artificial Intelligence for the Earth Systems. Copyright in this work may be transferred without further notice

- **What's New**: 이번 연구에서는 기존의 단일 수준 예측 변수를 우회하여 대기 변수의 수직 프로파일에서 직접적으로 뇌우 발생 확률을 추론하는 심층 신경망 SALAMA 1D를 개발했습니다. 이 모델은 대류 허용 수치 기상 예측(numerical weather prediction, NWP) 데이터를 기반으로 훈련되어 뇌우 예측의 정확도를 향상시키는 것을 목표로 합니다.

- **Technical Details**: SALAMA 1D는 물리적 원칙에 의해 설계된 심층 신경망으로, 희소 연결 방식(sparse connections)을 사용하여 유사한 높이 수준에서의 상호작용을 촉진하며, 셔플링 메커니즘(shuffling mechanism)을 통해 모델이 수직 격자에 연결된 비물리적 패턴을 학습하지 않도록 방지합니다. 데이터는 중앙 유럽의 ICON-D2-EPS 모델에서 수집된 것입니다.

- **Performance Highlights**: SALAMA 1D는 단일 수준 예측 변수를 사용하는 기존 기계 학습 모델과 비교했을 때 다양한 메트릭에서 우수한 성능을 보였습니다. 모델은 최대 11시간까지의 리드 타임(leadtimes)에서도 높은 기술을 유지하며, 훈련 세트의 양이 일정하게 유지되는 경우에도 더 많은 예측을 사용할수록 기술이 향상됩니다.



### Neural Click Models for Recommender Systems (https://arxiv.org/abs/2409.20055)
- **What's New**: 이번 연구에서는 추천 시스템(Recommendation System, RS)에서의 사용자 행동을 모델링하기 위한 신경망 아키텍처를 개발하고 평가하였습니다. 새로운 아키텍처로는 RNN, Transformer 기반 모델과 더불어 적대적(Adversarial) 및 계층적(Hierarchical) 아키텍처가 포함됩니다.

- **Technical Details**: 이 논문은 추천 시스템에서의 사용자 응답을 모델링하기 위해 RNN 및 Transformer 기반 아키텍처를 포함한 다양한 신경망 아키텍처를 제안합니다. 주요 실험에서 사용된 데이터셋은 ContentWise Impressions와 RL4RS로, 추천된 항목의 모임을 유지하고 사용할 수 있는 정보가 포함되어 있습니다. 모델은 사용자 세션의 이전 상호작용 이력을 고려하여 추천된 항목에 대한 반응을 예측합니다.

- **Performance Highlights**: 제안된 아키텍처는 ContentWise 및 RL4RS 데이터셋에서 기존의 기준 모델(Baseline)보다 우수한 성능을 보였으며, 새로운 추천 시스템 시뮬레이터의 기초로 활용될 수 있습니다.



### GUNDAM: Aligning Large Language Models with Graph Understanding (https://arxiv.org/abs/2409.20053)
- **What's New**: 이 논문에서는 텍스트 데이터 처리에서 뛰어난 성능을 보여준 Large Language Models (LLMs)를 그래프 구조 데이터를 이해하고 활용하는 데 적용하려는 새로운 접근 방식인 GUNDAM (Graph Understanding for Natural Language Driven Analytical Model)을 소개합니다. 기존 연구들이 주로 텍스트 속성이 풍부한 그래프에 초점을 맞춘 반면, 본 연구는 그래프 데이터 고유의 구조적 지식을 토대로 복잡한 추론 작업을 수행할 수 있는 LLM의 능력을 평가하고 향상시키고자 합니다.

- **Technical Details**: GUNDAM 모델은 그래프 구조를 LLM에 인코딩하기 위해 Graph Projection 방법을 사용합니다. 또한 CoT (Chain of Thought) 추론 경로를 포함한 고품질 그래프 추론 데이터 생성 파이프라인을 개발하여, 그래프 알고리즘을 활용해 정확성 및 중간 추론 과정을 제공합니다. 마지막으로, Alignment Tuning 방법을 통해 그래프 추론 데이터를 통한 GUNDAM의 미세 조정을 진행하여 모델의 추론 능력을 강화합니다.

- **Performance Highlights**: 실험 평가 결과, GUNDAM은 현재의 최첨단(SOTA) 성능을 초과 달성하였으며, LLM의 그래프 추론 능력에 영향을 미치는 주요 요소들도 밝혀졌습니다. 이 모델은 복잡한 그래프 구조를 이해하고 추론할 수 있는 능력을 개선하여 LLM의 일반 지능 발전에 기여할 가능성을 가지고 있습니다.



### Personalisation via Dynamic Policy Fusion (https://arxiv.org/abs/2409.20016)
- **What's New**: 이 연구에서는 이미 훈련된 딥 강화 학습(Deep Reinforcement Learning, RL) 정책을 사용자의 특정 요구에 맞게 조정하는 새로운 접근 방식을 제안합니다. 기존의 재훈련 방법을 피하고 사람의 피드백을 활용하여 제로샷(zero-shot) 방식으로 개인화된 정책을 학습할 수 있습니다.

- **Technical Details**: 우리는 LSTM(Long Short-Term Memory) 기반의 방법을 사용하여 사용자 의도를 추론하고, 훈련 중에 수집된 경로에 대한 피드백을 결합하여 개인화된 정책을 생성합니다. 이 동적 정책 융합(dynamic policy fusion) 접근 방식은 Boltzmann 분포의 온도 매개변수를 조절하여 사용자 요구와 작업 목표를 균형 있게 달성할 수 있도록 합니다.

- **Performance Highlights**: 본 연구에서 제안하는 동적 정책 융합 방법은 다양한 환경에서 최적의 작업을 수행하면서 사용자 요구를 지속적으로 충족시키는 것을 보여주었습니다. 이는 정적 정책 융합(static policy fusion) 방법의 한계를 극복하는 결과를 도출했습니다.



### Single-shot reconstruction of three-dimensional morphology of biological cells in digital holographic microscopy using a physics-driven neural network (https://arxiv.org/abs/2409.20013)
Comments:
          35 pages, 7 figures, 1 table

- **What's New**: 이 논문에서는 MorpHoloNet이라는 새로운 심층 학습 모델을 제안합니다. 이 모델은 생물학적 세포의 단일 촬영 홀로그램에서 3D 형태를 복원하기 위한 물리 기반 및 좌표 기반 신경망을 통합하여 다양한 기술적 한계를 극복합니다.

- **Technical Details**: MorpHoloNet은 3D 위상 변환 분포를 통해 일관한 빛의 광학 회절을 시뮬레이션하여 설계되었습니다. 이 모델은 센서 평면에서 입력 홀로그램과 시뮬레이션된 홀로그램 간의 손실을 최소화하여 최적화됩니다.

- **Performance Highlights**: MorpHoloNet은 기존 DIHM 방법에 비해 단일 촬영 홀로그램으로부터 3D 복잡한 빛의 장과 형태를 직접 복원할 수 있습니다. 실험적으로 생물학적 세포의 홀로그램을 사용하여 3D 형태와 굴절률 분포를 성공적으로 복원하였습니다.



### Numerically Robust Fixed-Point Smoothing Without State Augmentation (https://arxiv.org/abs/2409.20004)
- **What's New**: 새로운 Gaussian fixed-point smoother의 구현을 소개하며, 기존의 방식과 달리 상태 증강(state augmentation) 없이 수치적으로 강력한 Cholesky 기반 형식으로 제공한다.

- **Technical Details**: 우리의 알고리즘은 수치적 강건성(numerical robustness)을 유지하고 최소한의 복잡성을 유지하면서 데이터 스트림(data streams)과의 호환성을 제공합니다. 알고리즘은 고전적 접근 방식의 한계를 극복하고, Cholesky 기반 매개변수화(parametrisations)를 통해 다변량 Gaussian 분포를 다루는 방식을 제안합니다.

- **Performance Highlights**: JAX 구현을 통해 우리의 알고리즘이 가장 빠른 방법의 런타임을 충족하며, 동시에 가장 강력한 기술의 강건성을 보여준다는 실험 결과를 제시합니다.



### Mitigating Backdoor Threats to Large Language Models: Advancement and Challenges (https://arxiv.org/abs/2409.19993)
Comments:
          The 60th Annual Allerton Conference (Invited Paper). The arXiv version is a pre-IEEE Press publication version

- **What's New**: 본 논문은 Large Language Models(LLMs)의 백도어 공격(backdoor attack)의 위험성을 종합적으로 조사하고, LLMs에 대한 최근 방어 및 탐지 전략의 발전을 다룹니다.

- **Technical Details**: 백도어 공격은 훈련 데이터의 일부를 조작하여 LLM에 숨겨진 백도어를 주입하고, 사전 정의된 트리거에 의해 활성화되는 악의적인 행동을 유발합니다. 공격자들은 소량의 훈련 사례를 사용하여 모델의 특정 행동과 해당 트리거를 연관 지어 시스템을 해킹하거나 중단시킬 수 있습니다. 이러한 공격은 instruction tuning과 RLHF(강화 학습에서 인간 피드백)의 새로운 학습 패러다임을 통해 더욱 악화됩니다.

- **Performance Highlights**: LLMs에 대한 백도어 공격은 재정적 손실, 시장 혼란, 그리고 신뢰도 손상을 초래할 수 있습니다. 특히, 금융 분야와 헬스케어 같은 고위험 분야에서의 백도어 공격은 큰 피해를 유발할 가능성이 높습니다.



### A large-scale operational study of fingerprint quality and demographics (https://arxiv.org/abs/2409.19992)
Comments:
          Extended journal version submitted to IET Biometrics. 10 pages, 5 figures Reference conference paper: J. Galbally, A. Cepilovs, R. Blanco-Gonzalo, G. Ormiston, O. Miguel-Hurtado, and I. S. Racz, 'Fingerprint quality per individual finger type: A large-scale study on real operational data' in Proc. IEEE Intl. Workshop on Biometrics and Forensics 2023 (IWBF 2023)

- **What's New**: 본 논문은 16,000명의 대규모 데이터베이스를 사용하여 지문 인식 기술의 정확성이 성별, 연령 및 지문 유형 등 특정 인구 통계학적 그룹에 따라 편향되어 있는지를 조사합니다. 이전 연구들보다 더 많은 표본을 통해 지문 품질과 인구 통계학과의 관계를 심층적으로 분석했습니다.

- **Technical Details**: 연구는 500dpi (dots per inch) 터치 기반 광학 스캐너를 이용하여 문서 발급을 위한 비자 처리를 위해 전 세계 34개 비EU 국가에서 수집된 15,942개의 10-프린트 디지털 기록을 분석합니다. 이 데이터베이스는 성별, 연령, 출신 국가와 같은 메타데이터와 함께 지문 샘플을 포함합니다.

- **Performance Highlights**: 지문 인식 품질은 노화, 성별 및 지문 유형에 따라 차이가 있으며, 이는 여러 인구 집단에서 성능 변동을 일으킵니다. 연구 결과, 서로 다른 집단에 대해 지문 인식 기술의 성능 일관성을 향상시키기 위한 개선 방향을 제안하고 있습니다.



### Robust Multi-view Co-expression Network Inferenc (https://arxiv.org/abs/2409.19991)
- **What's New**: 이 논문은 DNA 시퀀싱 기술의 발전을 바탕으로 서로 다른 연구에서의 유전자 공발현(co-expression) 네트워크를 추론하는 새로운 강력한 방법을 제시합니다. 특히, 이 방법은 다양한 독립적인 데이터셋에서 유전자 간의 관계를 정확하게 식별할 수 있는 가능성을 열어주며, 이는 생명 과학 연구에 중요한 기여를 합니다.

- **Technical Details**: 제안한 방법은 MVTLASSO라는 행렬 변수 t-분포(matrix-variate t-distribution) 기반 프레임워크를 활용하여, 다중 관점(multi-view) 환경에서의 공분산(covariance)을 포착합니다. 이 방법은 유전자 로딩 매트릭스와 관련된 희소 정밀 행렬을 통해 GCN을 추론하고, EM (Expectation-Maximization) 절차를 통해 파라미터 추정(parameter estimation)을 수행합니다.

- **Performance Highlights**: 합성 데이터(synthetic data) 및 실제 유전자 발현 데이터(real gene expression data)를 통해, MVTLASSO는 기존의 기준 방법(compared to baseline methods)보다 더 높은 정확도로 그래프 구조를 재구성하는 능력을 보여줍니다.



### Violina: Various-of-trajectories Identification of Linear Time-invariant Non-Markovian Dynamics (https://arxiv.org/abs/2409.19978)
- **What's New**: 새로운 시스템 식별 방법 Violina (various-of-trajectories identification of linear time-invariant non-Markovian dynamics)를 제안합니다. 이 프레임워크는 주어진 공간에서 상태 공간 모델의 계수 행렬과 메모리 커널을 최적화하는 방식입니다.

- **Technical Details**: Violina 프레임워크에서는 프로젝티드 그래디언트 강하법 (projected gradient descent)을 사용하여 모델 예측이 여러 관찰 데이터 집합과 일치하도록 합니다. 이 방법을 통해 선형 비마르코프 동적 시스템을 식별할 수 있으며, 모델 파라미터와 메모리 효과에 대한 사전 지식에 따른 제약 조건을 적용할 수 있습니다.

- **Performance Highlights**: 합성 데이터 (synthetic data)를 사용하여 제안된 방법으로 식별된 마르코프 (Markovian) 및 비마르코프 상태 공간 모델이 기존의 동적 분해 기반 방법으로 식별된 모델보다 상당히 더 나은 일반화 성능을 보임을 수치적으로 입증하였습니다.



### Comments on "Privacy-Enhanced Federated Learning Against Poisoning Adversaries" (https://arxiv.org/abs/2409.19964)
Comments:
          Published at IEEE Transactions on Information Forensics and Security'23

- **What's New**: 본 연구에서는 Liu et al. (IEEE TIFS'21)에서 제안한 개인정보 보호 향상 프레임워크인 PEFL이 실제로는 개인정보를 보호하지 않음을 보여줍니다. PEFL은 모든 사용자의 전체 gradient vector를 한 참여 기관에게 노출하여 개인정보를 침해합니다.

- **Technical Details**: PEFL 시스템은 SecMed, SecPear 및 SecAgg의 세 가지 주요 프로토콜로 구성되어 있으며, 각각이 사용자 gradient에 대한 중요한 정보를 노출합니다. 연구팀은 이러한 정보들을 결합하여 계산 서버가 모든 사용자 gradient vector를 명확하게 알 수 있음을 증명합니다. PEFL은 동형암호(homomorphic encryption, HE)를 사용하지만, 도출된 결과는 개인정보 보호 면에서 취약합니다.

- **Performance Highlights**: PEFL은 지능형 공격 모델에 대해 보호를 제공한다고 주장했으나, 본 연구는 해당 시스템이 여러 가지 취약점을 내포하고 있으며, 이러한 사항들이 후속 연구에서도 지속적으로 인용되고 있다는 점을 지적합니다. 이로 인해, 향후 연구에서 PEFL의 결점을 반복하는 오류를 피하기 위한 필요성이 강조됩니다.



### A Self-attention Residual Convolutional Neural Network for Health Condition Classification of Cow Teat Images (https://arxiv.org/abs/2409.19963)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2409.18797

- **What's New**: 이번 연구에서는 소의 젖꼭지 건강 평가를 위한 새로운 모델인 Cows' Teats Self-Attention Residual Convolutional Neural Network (CTSAR-CNN)을 제안합니다. 이 모델은 기존의 CNN 모델에서 잔여 연결(residual connectivity)과 자기 주의(self-attention) 메커니즘을 결합하여 소의 젖꼭지 스스로를 평가할 수 있도록 합니다.

- **Technical Details**: CTSAR-CNN 모델은 고급 컴퓨터 비전 기술을 활용하여 젖꼭지의 과다 각화(hyperkeratosis) 정도를 분류합니다. 복잡한 환경에서도 소의 젖꼭지를 정확하게 인식하며, 다양한 자세와 위치에서의 데이터 해석을 가능하게 합니다. 이 모델은 잔여 연결과 자기 주의 메커니즘을 통해 정확성을 향상시킵니다.

- **Performance Highlights**: CTSAR-CNN의 통합으로 인해 기존의 접근 방식에 비해 젖꼭지 건강 평가의 정확성이 향상되었습니다. 연구 결과, 이 모델은 수의사들이 젖꼭지 건강을 더욱 신속하고 일관되게 평가하는 데 유용하며, 궁극적으로 유제품 산업에 기여할 수 있음을 보여줍니다.



### JaPOC: Japanese Post-OCR Correction Benchmark using Vouchers (https://arxiv.org/abs/2409.19948)
Comments:
          Accepted to PRICAI 2024

- **What's New**: 본 연구는 일본어 영수증에 대한 OCR(Optical Character Recognition) 오류 수정 방법의 벤치마크를 구축하고 효과성을 평가합니다. 이는 기존의 연구에서 다루어지지 않았던 일본어 OCR 오류 수정의 공개 가능 벤치마크를 제공합니다.

- **Technical Details**: 이 연구에서는 일본어 영수증에 특화된 OCR 오류 수정 벤치마크 JaPOC를 제안하고, T5와 같은 언어 모델을 활용하여 오류 수정 방법의 성능을 평가하였습니다. OCR 오류 수정 작업은 시퀀스-투-시퀀스 변환으로 정의되며, OCR 결과에 대해 고급 언어 모델로 미세 조정(fine-tuning)을 진행하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 오류 수정 알고리즘이 전반적인 인식 정확도를 크게 향상시켰습니다. Robota API와 Vision API를 사용하여 구축된 세트에서 상당한 정확도 향상이 확인되었습니다.



### One Shot GANs for Long Tail Problem in Skin Lesion Dataset using novel content space assessment metric (https://arxiv.org/abs/2409.19945)
Comments:
          15 pages, 6 Figures, 9 Tables and additional 6 Tables in Ancillary Data

- **What's New**: 이 논문에서는 HAM10000 데이터셋의 긴 꼬리 문제를 해결하기 위해 One Shot GANs 모델을 이용하여 소수 클래스의 데이터를 증강합니다. 이는 의료 영상 분류의 정확성을 높이고자 하는 새로운 접근 방식입니다.

- **Technical Details**: One Shot GANs는 단일 훈련 이미지를 사용하여 여러 샘플을 생성하는 생성적 적대 신경망(GAN)의 일종입니다. 이 모델은 이미지의 맥락과 레이아웃을 별도로 판별하는 이중 분기(discriminative) 구조를 가지고 있습니다. 또한, 훈련 데이터셋의 이미지를 최적 선택하는 informed subset selection 기법을 통해 전반적인 정확성을 높입니다.

- **Performance Highlights**: One-Shot GANs를 활용하여 소수 클래스에서 현저한 정확도 향상을 보였으며, 이는 WGANs와 비교해도 뚜렷한 개선을 나타냅니다. 새로 고안한 content-space assessment 메트릭 또한 FID 점수보다 더 나은 분류 정확도를 달성하는 데 기여했습니다.



### Data-driven decision-making under uncertainty with entropic risk measur (https://arxiv.org/abs/2409.19926)
- **What's New**: 이번 연구에서는 entropic risk measure(엔트로픽 리스크 측정)를 개선하기 위한 새로운 bootstrapping(부트스트래핑) 절차를 제안합니다. 이는 제한된 데이터에서 발생할 수 있는 편향을 보정하여 실제 리스크를 보다 정확하게 추정합니다.

- **Technical Details**: 새로운 bootstrapping 절차의 첫 단계는 데이터에 맞는 분포를 추정하는 것이고, 두 번째 단계에서는 이 분포를 사용하여 empirical entropic risk estimator(경험적 엔트로픽 리스크 추정기)의 편향을 계산하고 이를 보정하는 것입니다. Gaussian Mixture Model(가우시안 혼합 모델)을 사용할 경우 리스크가 과소 추정되는 경향이 있으며, 이 문제를 해결하기 위해 두 가지 대안을 고려했습니다: 하나는 경험적 엔트로픽 리스크 분포를 맞추는 것이고, 다른 하나는 극단값 분포를 맞추는 것입니다.

- **Performance Highlights**: 제안된 방법을 적용하여 type-$\infty$ Wasserstein ambiguity set(타입-무한 바서슈타인 모호성 집합)을 통한 분포ally robust(분포적으로 강건한) 엔트로픽 리스크 최소화 문제를 연구했습니다. validation performance(검증 성능)를 보정함으로써 모호성 집합의 크기 조정 정확성이 크게 향상되었습니다. 또한, 보험 계약 설계 문제에 대한 분포적으로 강건한 최적화 모델도 제안되어, 교차 검증 방법이 편향이 보정되지 않았을 경우 보험사를 위해 훨씬 높은 out-of-sample risk(샘플 외 리스크)를 초래할 수 있음을 보여주었습니다.



### On The Planning Abilities of OpenAI's o1 Models: Feasibility, Optimality, and Generalizability (https://arxiv.org/abs/2409.19924)
Comments:
          Updated link to code repository

- **What's New**: 이 연구에서는 OpenAI의 o1 모델이 복잡한 추론 작업을 수행하는 데 있어 보여준 능력을 바탕으로 플래닝(Planning) 능력을 평가합니다. 특히 세 가지 주요 측면인 실행 가능성(feasibility), 최적성(optimality), 일반화(generalizability)에 중점을 두었습니다.

- **Technical Details**: 연구는 제약이 많은 작업(예: Barman, Tyreworld)과 공간적으로 복잡한 환경(예: Termes, Floortile)에서 o1-preview 모델의 성능을 분석합니다. o1 모델은 Chain-of-Thought (CoT) 및 Tree-of-Thought (ToT)와 같은 고급 추론 기술을 적용하여 강화 학습으로 훈련되었습니다. 이 모델은 계획 수립에서 크리티컬한 요소인 메모리 관리(memory management)가 미비하며, 결정-making에서 병목 현상을 드러내고 있습니다.

- **Performance Highlights**: 이 모델은 GPT-4보다 작업 제약을 더 잘 준수하는 성능을 보여주었으나, 종종 비효율적인 행동을 포함한 최적이 아닌 솔루션을 생성하며 공간적으로 복잡한 작업에서 효과적으로 일반화하는 데 어려움을 겪습니다. 연구 결과는 LLM 기반의 플래닝 개선을 위한 중요한 통찰력을 제공하며, 메모리 관리 및 의사 결정 개선 방향을 제시합니다.



### Benchmarking Adaptive Intelligence and Computer Vision on Human-Robot Collaboration (https://arxiv.org/abs/2409.19856)
Comments:
          7 Pages, 9 Figures. 14 References. Submitted to IEEE RA-L Journal and ICRA 2025 Conference. This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 이 논문은 Industry 4.0에서 인간-로봇 협력(Human-Robot Collaboration, HRC)의 중요성을 강조하고, 새로운 환경에 적응하기 어려운 Concept Drift 문제를 해결하기 위해 Adaptive Intelligence와 Self-Labeling (SLB)을 통합한 방법론을 제안합니다.

- **Technical Details**: 이 연구에서 제안하는 방법론은 카메라와 무게 센서를 이용한 데이터 수집으로 시작되며, 이후 의도와 상태 변화를 주석 처리합니다. 다양한 딥러닝(Deep Learning) 모델을 훈련시키며, 사전 처리(preprocessing) 기술을 활용해 의도를 인식하고 예측합니다. 또한, SLB의 정확성을 높이기 위한 맞춤형 상태 감지 알고리즘을 개발하였으며, 이는 의도 레이블에 대해 정확한 상태 변화 정의와 타임스탬프를 제공합니다.

- **Performance Highlights**: 연구 결과, 스켈레탈 포즈(skeletal posture) 전처리(preprocessing)를 적용한 MViT2 모델이 83%의 정확도를 달성하였고, 이는 스켈레톤 포즈 추출 없이 얻은 79%의 정확도보다 향상된 것입니다. 또한, SLB 메커니즘은 91%의 레이블링 정확성을 달성하여 수동 주석에 필요한 시간을 획기적으로 줄였습니다. 궁극적으로, 본 연구는 다양한 자기 레이블링(Self-Labeled) 데이터 세트를 이용한 모델 성능 향상을 통해 Concept Drift 문제를 해결하고, 제조업에서 지능형 협업 로봇(cobots)의 빠른 배포 가능성을 보여줍니다.



### SATA: Spatial Autocorrelation Token Analysis for Enhancing the Robustness of Vision Transformers (https://arxiv.org/abs/2409.19850)
- **What's New**: 최근 몇 년 간, vision transformers (ViTs)의 성능을 향상시키기 위해 다양한 방법이 연구되었으나 한계가 있었습니다. 본 논문에서는 Spatial Autocorrelation Token Analysis (SATA)라는 새로운 접근 방식을 소개합니다.

- **Technical Details**: SATA는 token의 특징 간의 공간적 관계를 활용하여 ViT 모델의 표현 능력과 강인성을 향상시킵니다. 이는 self-attention 메커니즘의 Feed-Forward Network (FFN) 블록에 입력되기 전, token의 공간적 자기상관(spatial autocorrelation) 점수에 따라 분석하고 그룹화하는 것을 통해 이루어집니다. 매우 중요한 점은 SATA가 기존의 사전 학습된 ViT 모델에 재학습이나 추가 조정 없이 통합된다는 것입니다.

- **Performance Highlights**: 실험 결과 SATA로 강화된 ViTs는 ImageNet-1K 이미지 분류에서 94.9%의 최상위(top-1) 정확도를 기록하며, ImageNet-A(63.6%), ImageNet-R(79.2%), ImageNet-C(13.6%) 등의 여러 강인성 벤치마크에서도 새로운 최첨단 성능을 달성했습니다. 이 모든 것은 기초 모델에 대한 추가 교육이나 조정 없이 이루어졌습니다.



### Enabling Multi-Robot Collaboration from Single-Human Guidanc (https://arxiv.org/abs/2409.19831)
- **What's New**: 이 논문에서는 멀티 에이전트 시스템에서 협업 행동을 학습하는 새로운 방법을 제안합니다. 기존의 방법들은 공동 보상(joint reward)과 중앙 집중식 관찰(centralized observations)에 의존했으나, 본 연구에서는 단일 인간의 전문성을 활용한 효율적이고 명시적인 방법을 소개합니다.

- **Technical Details**: 제안된 방법은 인간 조작자가 짧은 시간 동안 에이전트 제어를 동적으로 전환할 수 있게 하여, 팀원에 대한 인간과 유사한 사고 이론(theory-of-mind model)을 에이전트에 통합함으로써 이루어집니다. 이 과정을 통해 에이전트는 효과적으로 협업을 학습할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 도전적인 협업 숨바꼭질(hide-and-seek) 작업에서 성공률(success rate)을 58%까지 향상시켰으며, 단 40분의 인간 지도만으로도 이러한 성과를 달성했습니다. 또한, 멀티 로봇 실험을 통해 이러한 발견이 실제 세계로 전이됨을 보여주었습니다.



### Can Models Learn Skill Composition from Examples? (https://arxiv.org/abs/2409.19808)
Comments:
          Accepted to NeurIPS 2024

- **What's New**: 이번 연구는 compositional generalization (구성적 일반화) 능력에 대한 평가에서 SKILL-MIX를 활용하여 smaller models (소형 모델)의 성능을 측정한 것입니다. 이 연구는 AI 안전성 및 정렬 연구와도 관련이 있습니다.

- **Technical Details**: 연구에서는 다양한 언어 기술을 포함한 텍스트 샘플을 생성하기 위해 GPT-4를 사용했습니다. 이를 통해 7B 및 13B 파라미터 모델을 fine-tuning (미세 조정) 하여 k-기술 조합의 텍스트 작성을 평가했습니다. 사용된 기술 세트는 수사학적 (rhetorical), 문학적 (literary), 추론 (reasoning), 마음 이론 (theory of mind), 상식 (common sense) 등을 포함합니다.

- **Performance Highlights**: (1) k=2 및 k=3 기술의 조합으로 학습한 모델은 k=4 및 k=5 기술을 활용한 텍스트 작성을 개선하였습니다. (2) 기술 범주를 나누어 학습 및 검증 그룹으로 분리했을 때, 모델은 훈련 중 본 적이 없는 기술로 텍스트를 작성하는 데에서 크게 향상되었습니다. 이 연구는 skill-rich (기술이 풍부한) 텍스트를 훈련에 통합하는 것이 모델의 구성 능력을 크게 향상시킬 수 있음을 제안합니다.



### Gradient descent with adaptive stepsize converges (nearly) linearly under fourth-order growth (https://arxiv.org/abs/2409.19791)
Comments:
          58 pages, 5 figures

- **What's New**: 이 논문은 그래디언트 강하(gradient descent)의 선형 수렴(linear convergence)이 최소화(minimizer)로부터 사차(4차) 성장(fourth-order growth)을 단순히 가지는 부드러운 함수에서도 발생할 수 있음을 보여줍니다. 기존의 신념과 달리, 함수의 성장은 최소화 지점에서 멀어질 때 반드시 이차(quadratic) 성장으로 이어지지 않는다는 주장을 하고 있습니다.

- **Technical Details**: 우리가 제안하는 적응적 단계 크기(adaptive stepsize)는 흥미로운 분해 정리(decomposition theorem)에 기반하고 있습니다. 이는 최적 솔루션(optimal solution) 주변에 부드러운 다양체(smooth manifold)가 존재하여, 해당 함수가 ravine(협곡)로 불리는 영역에서 최소한 이차적으로 성장하고 그 along it 동안 상수 순서로 성장한다는 것입니다. 이러한 ravine은 짧은 그래디언트 단계와 긴 Polyak 그래디언트 단계의 결합을 통해 빠른 수렴을 보장합니다.

- **Performance Highlights**: 발표된 이론과 알고리즘은 매트릭스 감지(matrix sensing), 분해(factorization) 문제와 과파라미터화(overparameterized regime)된 상태에서 단일 뉴런(single neuron) 학습 사례에서 검증되었습니다.



### Automatic debiasing of neural networks via moment-constrained learning (https://arxiv.org/abs/2409.19777)
Comments:
          Code repository and license available at this https URL

- **What's New**: 본 논문에서는 자동 디바이징(automatic debiasing) 기술을 개선하기 위한 새로운 방법인 순간 제약 학습(moment-constrained learning)을 제안합니다. 이 방법은 예상되는 순간을 제약하여 최적화 하이퍼파라미터에 대한 견고성을 높입니다.

- **Technical Details**: 기존의 자동 디바이징 기법은 Riesz representer (RR)를 직접 학습하며, 이 과정에서 손실 함수를 최소화하는 방법을 사용합니다. 저자들은 신경망(neural networks)을 활용하여 평균 치료 효과(average treatment effect) 및 도함수 효과(derivative effect) 추정 문제에 대한 평가를 수행했습니다.

- **Performance Highlights**: 수치 실험 결과, 제안된 방법은 기존의 최신 벤치마크(state of the art benchmarks)와 비교하여 개선된 성능을 보였습니다.



### Balancing Cost and Effectiveness of Synthetic Data Generation Strategies for LLMs (https://arxiv.org/abs/2409.19759)
- **What's New**: 대규모 언어 모델(LLMs)의 다양한 용도 적용 확대에 따라, 모델 개선을 위한 고품질 작업 특화 데이터셋 생성이 병목 현상으로 떠오르고 있습니다. 본 논문에서는 고급 인간 데이터를 사용하지 않고도 모델 성능을 최대화하기 위한 방법들을 조사합니다.

- **Technical Details**: 논문에서는 여러 가지 합성 데이터 생성 전략을 세 가지 범주로 나누어 연구합니다: Answer Augmentation(답변 증강), Question Rephrase(질문 재구성), New Question(새로운 질문). 각 전략은 seed instruction set의 크기와 query budget(질의 예산)에 따른 성능 차이를 보였으며, 데이터 생성 전략의 최적 선택은 두 비율의 관계에 크게 의존합니다.

- **Performance Highlights**: 저비용의 데이터 환경에서는 기존 질문에 대한 새로운 답변을 생성하는 것이 가장 효과적이며, 이 비율이 높아질수록 새로운 질문을 생성하는 것이 최적입니다. 전반적으로, 데이터 양이 적은 경우 선택한 증강 방식과 기타 디자인 선택이 성능에 중대한 영향을 미치는 것으로 나타났습니다.



### A multimodal LLM for the non-invasive decoding of spoken text from brain recordings (https://arxiv.org/abs/2409.19710)
Comments:
          15 pages, 4 figures

- **What's New**: 이번 연구에서 제안한 모델은 비침습적인 fMRI 기록에서 말하는 텍스트를 디코딩하는 멀티모달 LLM 모델입니다. 이는 정교한 트랜스포머 인코더와 고정된 대형 언어 모델을 연결하여 브레인 활동을 텍스트와 정렬하는 것으로, 기존 모델을 능가하는 성능을 보여줍니다.

- **Technical Details**: 제안된 아키텍처는 (i) 인코더와 (ii) 고정된 대형 언어 모델로 구성됩니다. 인코더는 특정 트랜스포머에서 파생된 것으로, 확대된 임베딩 레이어와 개선된 주의 메커니즘을 포함하고 있습니다. 학습 전략은 텍스트와 뇌 기록 간의 매핑을 위해 두 단계로 진행됩니다.

- **Performance Highlights**: 제안된 시스템은 다양한 텍스트 유사성 및 의미론적 지표에서 기존 아키텍처보다 우수한 결과를 기록했으며, 디코딩된 대화 맥락의 시각적 품질이 높아 주요 대화 키워드를 식별할 수 있음을 보여줍니다.



### InfantCryNet: A Data-driven Framework for Intelligent Analysis of Infant Cries (https://arxiv.org/abs/2409.19689)
- **What's New**: 이번 논문에서는 infant cries(유아의 울음소리)를 이해하기 위한 새로운 데이터 기반 프레임워크, 'InfantCryNet'을 제안합니다. 이 프레임워크는 울음 소리의 탐지 및 분석을 동시에 수행하며, 데이터 부족 문제를 해결하기 위해 사전 훈련된 오디오 모델을 사용합니다.

- **Technical Details**: 모델은 통계적 풀링(statistical pooling)과 다중 헤드 주의(pooling with multi-head attention) 기법을 사용하여 더 효과적으로 특징을 추출하며, 모델의 효율성을 높이기 위해 knowledge distillation(지식 증류) 및 모델 양자화(model quantization) 기법을 적용했습니다.

- **Performance Highlights**: 실제 데이터셋을 통한 실험 결과, 제안된 프레임워크는 분류 정확도에서 기존의 최첨단 모델 대비 4.4% 높은 성능을 나타내었고, 모델 압축 기술을 통해 모델 크기를 7% 줄일 수 있었습니다. 성능 손실 없이 최대 28%까지 모델 크기를 줄일 수 있는 가능성을 보여주었습니다.



### Nonideality-aware training makes memristive networks more robust to adversarial attacks (https://arxiv.org/abs/2409.19671)
Comments:
          14 pages, 8 diagrams

- **What's New**: 이번 연구에서는 memristor 기반 신경망에서의 비이상적인 훈련(nonideality-aware training)이 적대적 공격(adversarial attacks)에 대한 강인성(adversarial robustness)에 미치는 영향을 조사했습니다. 기존 연구에서 다뤄지지 않았던 memristive 디바이스의 비이상성이 공격에 어떻게 영향을 미치는지 분석했습니다.

- **Technical Details**: 비이상적인 훈련 기법은 실제 비이상성(nonidealities)을 훈련 알고리즘에 노출시키는 방식입니다. 이를 통해 stuck devices와 같은 메모리 내의 비이상성이 공격을 받는 과정에서 신경망의 방어 능력을 향상시키는데 기여함을 발견했습니다. 연구의 주된 초점은 Fast Gradient Sign Method (FGSM)와 같은 공격에 노출된 memristive 네트워크의 강인성을 평가하는 것이었습니다.

- **Performance Highlights**: 실험 결과, 비이상적인 훈련 기법을 적용한 경우, memristive 네트워크는 제한된 비이상성 지식 하에서도 적대적 공격에 대한 반응이 상당히 개선되는 것을 보여주었습니다. 이는 memristive 디바이스가 물리적 비이상성을 견딜 수 있는 가능성을 제시합니다.



### Solution for Temporal Sound Localisation Task of ECCV Second Perception Test Challenge 2024 (https://arxiv.org/abs/2409.19595)
- **What's New**: 이번 보고서는 Temporal Sound Localisation (TSL) 작업을 위한 향상된 방법을 제안합니다. 이 방법은 사전 정의된 소리 클래스에 따라 비디오에서 발생하는 소리 이벤트를 지역화하고 분류하는 것을 목표로 합니다. 특히, 지난해 대회에서 우승한 솔루션의 한계를 극복하기 위해 오디오 모달리티에 더 큰 비중을 두었습니다.

- **Technical Details**: 기본 모델로는 Actionformer를 사용하여 비디오와 오디오 특성을 융합하고, 다양한 모델(InterVideo, CaVMAE, VideoMAE 등)을 통해 오디오 특성을 추출합니다. 오디오 특성은 멜 스펙트로그램 형태로 변환한 후, 각 모델에서 독립적으로 고수준 특성을 추출하고 이들을 결합하여 총 2304차원의 오디오 표현을 생성합니다.

- **Performance Highlights**: 최종 테스트에서 0.4925의 점수를 기록하며 1위를 차지했습니다. 이 결과는 오디오 특성이 전체적인 접근 방식의 성능을 크게 향상시킨다는 것을 입증합니다.



### DiffCP: Ultra-Low Bit Collaborative Perception via Diffusion Mod (https://arxiv.org/abs/2409.19592)
Comments:
          7 pages, 4 figures

- **What's New**: DiffCP는 효율적인 데이터 전송을 위한 새로운 협업 인식 패러다임으로, 확산 모델(diffusion model)을 활용하여 센싱 정보를 압축합니다. 기하학적 및 의미적 조건을 통합하여 특성 수준의 협업을 가능하게 하며, 저렴한 통신 비용으로 성능을 향상시킵니다.

- **Technical Details**: DiffCP는 트랜스포머 기반의 확산 모델을 사용하여 협업 에이전트(co-agent)의 관측을 재구성합니다. 이 모델은 다양한 센서 모달리티를 지원하고, 고유한 전경 정보(foreground information)를 유지하며, 유니버설 Bird’s Eye View (BEV) 공간 내에서 이뤄지는 확산(difussion) 과정으로 인퍼런스 차원을 감소시킵니다.

- **Performance Highlights**: DiffCP는 communication overhead를 14.5배 줄이면서도 기존의 state-of-the-art 알고리즘과 동일한 성능을 유지합니다. 이는 차세대 협력 로봇 시스템의 실 세계 배포 및 구현을 촉진합니다.



### Brain Tumor Classification on MRI in Light of Molecular Markers (https://arxiv.org/abs/2409.19583)
Comments:
          ICAI'22 - The 24th International Conference on Artificial Intelligence, The 2022 World Congress in Computer Science, Computer Engineering, & Applied Computing (CSCE'22), Las Vegas, USA. The paper acceptance rate 17% for regular papers. The publication of the CSCE 2022 conference proceedings has been delayed due to the pandemic

- **What's New**: 본 연구에서는 저급 신경교종(low-grade gliomas)의 임상 결과와 관련된 1p/19q 유전자 공동 삭제(co-deletion) 상태를 예측하기 위해 특별히 설계된 MRI 기반 합성곱 신경망(convolutional neural network, CNN)을 제안합니다. 기존의 전이 학습 모델(transfer learning model) 대신에 처음부터 모델을 개발하여 신뢰성을 높였습니다.

- **Technical Details**: 제안된 모델은 합성곱(convolution) 레이어, 풀링(pooling) 레이어, LeakyReLU, 소프트맥스(Softmax), 드롭아웃(dropout) 레이어와 완전 연결(Dense) 레이어로 구성되어 있으며, 3x3 크기의 커널을 사용합니다. 모델 학습 과정에서는 가우시안 노이즈(Gaussian noise)를 주입하고 데이터 보강(data augmentation)을 통해 성능을 향상시켰습니다. 125개의 1p/19q 공동 삭제 및 31개의 비삭제 이미지를 포함한 검증 세트를 사용했습니다.

- **Performance Highlights**: 제안된 네트워크는 1p/19q 공동 삭제 이미지를 분류할 때 96.37% F1-점수, 97.46% 정밀도(precision), 96.34% 재현율(recall)을 달성했습니다. 비교 대상인 InceptionV3, VGG16, MobileNetV2와 같은 모델에 비해 우수한 성능을 보였습니다.



### A Universal Deep Learning Framework for Materials X-ray Absorption Spectra (https://arxiv.org/abs/2409.19552)
Comments:
          Main manuscript: 21 pages, 11 figures. Supplemental material (12 pages, 6 figures) available as a separate file in arXiv ancillary files (additional downloadable files)

- **What's New**: 본 논문은 X선 흡수 분광법(X-ray Absorption Spectroscopy, XAS)의 데이터 분석을 위한 빠르고 강력한 파이프라인을 개발하기 위한 여러 가지 전이 학습(transfer learning) 접근 방식을 제시합니다.

- **Technical Details**: 세 가지 독특한 전략을 통해 XAS 예측의 정확성과 효율성을 개선합니다. 첫째, M3GNet을 사용하여 흡수 사이트의 지역 화학 환경에 대한 잠재 표현(latent representation)을 도출하고, 전통적인 특성화 방법보다 몇 배 향상된 성능을 보입니다. 둘째, 요소들 간의 공통 모델을 훈련시킨 후 각 요소에 대해 미세 조정(fine-tuning)을 실시함으로써, 개별 모델보다 최대 31% 향상된 결과를 얻을 수 있습니다. 셋째, 서로 다른 신뢰도(fidelity)로 생성된 스펙트라에 대해 일반 모델을 조정하여 예측 정확도를 최대 24% 향상시키는 방법을 제안합니다.

- **Performance Highlights**: 이 접근 방식은 3d 전이 금속(Ti-Cu)의 K-edge 스펙트라 데이터베이스를 사용하여 입증되었고, 더욱 다양한 요소의 XAS 예측으로 확장 가능하고, 재료 과학의 다른 딥러닝 모델에도 일반화할 수 있는 전이 학습 프레임워크를 제공합니다.



### An evolutionary approach for discovering non-Gaussian stochastic dynamical systems based on nonlocal Kramers-Moyal formulas (https://arxiv.org/abs/2409.19534)
- **What's New**: 본 연구는 비가우시안(Stochastic Dynamical Systems) 동적 시스템의 명시적인 거버닝 방정식을 데이터로부터 추출하기 위해 진화 상징 희소 회귀(Evolutionary Symbol Sparse Regression, ESSR) 접근법을 개발하고 있습니다. 이 방법은 비국소 크라머-모얄(Nonlocal Kramers-Moyal) 공식, 유전 프로그래밍(Genetic Programming), 그리고 희소 회귀(Sparse Regression)를 기반으로 하여 비가우시안 동적 시스템을 샘플 경로 데이터에서 추출하는 것을 목표로 합니다.

- **Technical Details**: 제안된 ESSR 방법은 유전 프로그래밍을 통해 다양한 후보 함수를 생성하고, 희소 회귀 기법을 통해 이러한 후보와 관련된 계수를 학습합니다. 비국소 크라머-모얄 공식은 유전 프로그래밍에서의 적합도 측정 및 희소 회귀에서의 손실 함수 구축의 기초로 사용됩니다. 이 방법은 특히 비가우시안 특성을 가진 동적 시스템의 방정식을 발견하는 데 유용합니다.

- **Performance Highlights**: ESSR 방법은 여러 모형에 대한 적용을 통해 그 효능과 능력을 입증하였으며, 기존의 데이터 기반 모델링 기법들이 모형해내지 못하는 복잡한 동적 행동을 해석하는 데 강력한 도구로 자리매김할 것으로 보입니다.



### Video DataFlywheel: Resolving the Impossible Data Trinity in Video-Language Understanding (https://arxiv.org/abs/2409.19532)
Comments:
          Under peer review

- **What's New**: 이번 연구에서는 데이터 수량, 다양성, 품질 간의 "불가능한 삼위일체"를 밝히며, 기존의 대규모 ASR 데이터셋이 품질 부족으로 인해 개선을 요구하고 있음을 강조합니다.

- **Technical Details**: 우리는 Video DataFlywheel 프레임워크를 도입하여 비디오 주석을 반복적으로 개선하며, AdaTaiLr라는 새로운 noise control 방법을 통해 대규모 데이터셋에서의 효율성을 증명합니다. 또한, 비디오-언어 모델을 활용하여 합성 주석을 생성하고, 이를 통해 데이터셋을 정제합니다.

- **Performance Highlights**: 우리의 프레임워크는 기존 데이터 정제 기준선보다 3% 성능 향상을 보였으며, 다양한 비디오-언어 이해 과제에서 중요한 개선을 facilitated 합니다.



### Efficient Backdoor Defense in Multimodal Contrastive Learning: A Token-Level Unlearning Method for Mitigating Threats (https://arxiv.org/abs/2409.19526)
- **What's New**: 본 연구는 멀티모달 대비 학습(Multimodal Contrastive Learning)에서 발생할 수 있는 backdoor 공격에 대한 새로운 방어 기제를 제안합니다. 이를 위해 '기계 학습 삭제(machine unlearning)' 개념을 활용하여 모델의 backdoor 취약성을 신속하게 제거하는 방법을 제시합니다.

- **Technical Details**: 제안하는 기법은 Unlearn Backdoor Threats (UBT)로, 적은 수의 오염 샘플을 선택하여 모델이 backdoor 특징을 잊도록 유도합니다. 이 과정에서 과도적 훈련(overfit training)을 통해 의심스러운 샘플을 탐지하고, 해당 샘플의 일부를 선택하여 신속하게 제거합니다. 이 새로운 접근법은 토큰 기반의 부분 학습 삭제(training regime) 방법을 포함하여, 모델의 취약한 요소에 집중하여 backdoor 연관성을 분리합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 다양한 backdoor 공격 방법에 대해 효과적으로 방어하며, 기존 방법과 비교할 때 공격 성공률(attack success rate)을 19% 감소시키고, 깨끗한 정확도(clean accuracy)를 2.57% 증가시켰습니다.



### GenTel-Safe: A Unified Benchmark and Shielding Framework for Defending Against Prompt Injection Attacks (https://arxiv.org/abs/2409.19521)
- **What's New**: 이번 논문에서는 GenTel-Safe라는 통합 프레임워크를 소개합니다. 이 프레임워크는 새로운 prompt injection attack detection 방법인 GenTel-Shield와 84812개의 prompt injection 공격이 포함된 종합 평가 벤치마크인 GenTel-Bench를 포함합니다.

- **Technical Details**: GenTel-Shield는 모델 비구조적인 방법으로, LLM의 내부 구조에 대한 지식 없이 보호 조치를 적용할 수 있게 합니다. 데이터 증대 교육 기법을 채택하여 해로운 프롬프트를 식별 및 필터링하는 과정에서 사용자 입력 무결성을 유지하도록 설계되었습니다.

- **Performance Highlights**: GenTel-Shield는 jailbreaking 공격에 대해 97.63%, 목표 탈취 공격에 대해 96.81%의 방어 성공률을 달성하여 최신 성능을 기록했습니다. 또한 F1 점수에서 jailbreaking 공격에 대해 97.69%, 목표 탈취 공격에 대해 96.74%를 기록하여 정상 사용자의 활동에 대한 최소한의 방해를 보장합니다.



### HealthQ: Unveiling Questioning Capabilities of LLM Chains in Healthcare Conversations (https://arxiv.org/abs/2409.19487)
- **What's New**: 이 논문에서는 LLM(대형 언어 모델)의 질문 능력을 평가하기 위한 새로운 프레임워크인 HealthQ를 제안합니다. 기존의 디지털 의료 연구는 LLM의 질문-응답 능력을 향상시키기 위한 접근 방식을 탐색해왔으나, LLM이 환자와의 대화 중에 효과적인 질문을 통해 정보를 수집하는 능력은 충분히 연구되지 않았습니다.

- **Technical Details**: HealthQ는 Retrieval-Augmented Generation (RAG), Chain of Thought (CoT), reflective chains 등 여러 LLM 체인을 구현하고, LLM 평가자를 도입하여 생성된 질문의 관련성과 정보성 평가를 수행하는 방법론을 사용합니다. 이 논문은 ROUGE와 NER 기반 비교와 같은 전통적인 자연어 처리(NLP) 메트릭스를 이용해 건강 관련 질의응답 시스템의 질문 능력을 탐색합니다.

- **Performance Highlights**: 이 연구는 LLM 헬스케어 체인에서 질문 능력에 대한 첫 포괄적인 연구를 실시하며, 질문-응답 성능 평가를 위한 새로운 데이터 생성 파이프라인을 개발하고, 다양한 LLM 체인 유형과 전통적인 NLP 메트릭스를 통합하는 평가 방법론을 제안합니다. 이러한 접근방식은 향후 LLM의 질문 능력 연구에 대한 견고한 기준을 제공합니다.



### Towards Croppable Implicit Neural Representations (https://arxiv.org/abs/2409.19472)
Comments:
          Accepted to NeurIPS 2024

- **What's New**: 이 논문에서는 수정 가능한 암묵적 신경 표현(Implicit Neural Representations, INRs)의 새로운 아키텍처인 Local-Global SIRENs을 제안합니다. 이 구조는 크로핑(cropping) 작업에 최적화되어 있으며, 특정 신호의 부분을 제거할 수 있는 능력을 제공합니다.

- **Technical Details**: Local-Global SIRENs는 로컬(local) 및 글로벌(global) 기능 추출을 결합하여 신호를 인코딩하는 혁신적인 INR 아키텍처입니다. 이 아키텍처는 인코딩된 신호의 특정 부분을 쉽게 제거하며, 이를 통해 비율에 맞는 가중치 감소가 가능합니다. 네트워크를 재훈련할 필요 없이 가중치를 제거하는 방법으로 구현되었습니다.

- **Performance Highlights**: Local-Global SIRENs는 훈련 속도를 높이고 다양한 신호의 인코딩을 향상시키며 다운스트림 성능을 개선하는 데 기여합니다. 특히, INCODE와 같은 현대 INRs에 적용하여 이러한 우수성을 입증하였으며, 기존 기본 INRs보다 개선된 성능을 보여주었습니다.



### Accelerating Malware Classification: A Vision Transformer Solution (https://arxiv.org/abs/2409.19461)
Comments:
          8 pages, 5 figures, 1 table Submitted to Neurips 2024 ML for system worshop

- **What's New**: 이번 논문에서는 진화하는 사이버 보안 환경에서 신속하고 정확한 악성 코드 분류의 중요성이 강조되며, LeViT-MC라는 새로운 아키텍처가 제안되었습니다. 본 아키텍처는 비전 트랜스포머 기반 구조를 활용하여 우수한 상태의 악성 코드 탐지 및 분류 성능을 보여줍니다.

- **Technical Details**: LeViT-MC 아키텍처는 이진 분류를 위한 DenseNet과 신속한 추론 속도를 자랑하는 가벼운 비전 트랜스포머인 LeViT를 결합하여 구성됩니다. 악성 코드 이미지를 RGB 이미지로 변환한 후, 이 철저한 접근 방식을 통해 높은 정확성과 빠른 인퍼런스 속도를 달성합니다.

- **Performance Highlights**: LeViT-MC는 악성 코드 분류에서 96.6%의 정확도를 기록하며, MaleVis 데이터셋에서 이전의 모든 모델을 초월하는 뛰어난 성능을 보였습니다. 또한, 이 아키텍처는 평균 인퍼런스 속도가 일반적인 비전 트랜스포머보다 약 10배, 최고의 CNN 모델인 ResNet보다 약 3배 더 빠릅니다.



### Scalable Fine-tuning from Multiple Data Sources:A First-Order Approximation Approach (https://arxiv.org/abs/2409.19458)
Comments:
          16 pages

- **What's New**: 본 논문에서는 n개의 보조 작업으로부터의 정보를 최적 활용하여 특정 작업을 위해 언어 모델(LM)을 미세 조정하는 문제를 다룹니다. 보조 작업 중 일부가 목표 작업의 성능 향상에 유용하지 않을 수 있으므로 적절한 보조 작업의 하위 집합을 선택하는 것이 중요합니다. 이 논문은 반복 훈련 없이 모델 미세 조정 성능을 추정할 수 있는 새로운 알고리즘을 소개합니다.

- **Technical Details**: 제안된 알고리즘은 먼저 모든 작업의 데이터를 사용한 멀티태스크 훈련을 통해 메타 초기화(meta initialization)를 얻습니다. 이후 메타 초기화의 함수 값과 그래디언트를 사용하여 하위 집합의 모델 미세 조정 손실을 근사합니다. 이 과정에서 USDA (Uniformly Stochastic Dimensionality Reduction)를 사용하여 CPU에서 몇 초 만에 미세 조정 성능을 추정할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 GradEx 접근법은 기존 방법보다 30배 빠르게 하위 집합 선택을 처리하면서도 실제 미세 조정 성능의 1% 오차 이내로 정확도를 유지합니다. 또한, GradEx는 강의 지침 조정(instruction tuning) 및 체인 사고 미세 조정(chain-of-thought fine-tuning)의 하류 평가에서 기존 방법보다 평균 3.8% 향상된 성능을 보였습니다.



### HTML-LSTM: Information Extraction from HTML Tables in Web Pages using Tree-Structured LSTM (https://arxiv.org/abs/2409.19445)
- **What's New**: 이 논문에서는 구조가 다른 HTML 테이블에서 유사한 내용을 추출하는 새로운 방법, HTML-LSTM을 제안합니다. 이 방법은 다양한 웹 페이지의 HTML 테이블을 통합하여 정보를 검색할 수 있도록 설계되었습니다.

- **Technical Details**: HTML-LSTM은 tree-structured LSTM(Tree-LSTM)을 확장하여 HTML 데이터의 언어적 및 구조적 정보를 동시에 추출하는 데 중점을 둡니다. 이 방법은 먼저 HTML 데이터를 테이블 구조로 변환한 후, 하위 구조를 DOM 트리 형태로 변환하여 노드의 특성을 분류하고 새로운 테이블로 통합합니다. 이를 위해 이중 방향 LSTM(Bi-LSTM)을 사용하여 각 요소의 언어 표현을 얻고, HTML-LSTM을 통해 트리 구조에서의 요소 간 관계를 고려하여 특징을 추출합니다.

- **Performance Highlights**: 실험 결과, 유치원 데이터에서 F1 점수 0.96, 대학교 강의 계획서 데이터에서 F1 점수 0.86을 달성하여 HTML-LSTM의 유효성을 입증했습니다. 또한, HTML-LSTM은 Tree-LSTM보다 더 우수한 성능을 보였습니다.



### Energy-Efficient Computation with DVFS using Deep Reinforcement Learning for Multi-Task Systems in Edge Computing (https://arxiv.org/abs/2409.19434)
- **What's New**: 이 연구는 다중 작업과 다중 마감 기한을 가진 소프트 실시간 시스템을 다루며, 에너지 절약을 위한 강화 학습 기반의 DVFS(동적 전압 및 주파수 조정) 방법을 제안하고 있습니다. 이를 통해 이전의 단일 작업 및 단일 마감 기한으로 모델링된 시스템의 한계를 극복하고 보다 복잡한 상황을 처리할 수 있도록 합니다.

- **Technical Details**: 제안된 시스템은 리눅스 커널 내의 시간 시리즈 정보를 강화 학습에 적합한 형태로 인코딩하여 DVFS 정책을 생성하고, 세 가지 고정 멀티 태스크 워크로드(각각 3, 5, 8 작업)를 테스트했습니다. 이 과정에서 사용되는 두 가지 인코딩 방법이 존재하며, 이들은 시스템 활용도만으로 성능 카운터를 사용합니다. 또한, Double Deep Q Learning(DDQN) 기법을 활용하여 정책을 학습합니다.

- **Performance Highlights**: 테스트 결과, 제안된 방법은 리눅스 내장 거버너에 비해 3%-10%의 전력 절약 효과를 보였습니다. 이는 다중 작업과 다중 마감 기한을 고려한 새로운 접근 방식으로, 기존의 단순화된 모델에 비해 현저히 개선된 성과를 나타냅니다.



### Generalization Error of the Tilted Empirical Risk (https://arxiv.org/abs/2409.19431)
Comments:
          49 pages

- **What's New**: 이번 연구에서는 머신러닝 응용을 위한 비선형 위험 메트릭으로 제안된 tilted empirical risk의 일반화 오류(generalization error)를 분석합니다. 특히, 모집단 위험(population risk)과 tilted empirical risk 간의 차이를 정의하고, 수렴 속도가 $O(1/
√{n})$인 경계 정보(information-theoretic bounds)를 제공합니다.

- **Technical Details**: tilted 일반화 오류는 모집단 위험과 tilted empirical risk의 차이로 정의되며, 이 연구에서는 KL 정규화(KL-regularized)된 기대 tilted empirical risk 최소화 문제에 대한 해를 연구합니다. 또한, 기대 tilted 일반화 오류에 대한 상한선(upper bound)을 도출하고, 수렴 속도가 $O(1/n)$임을 밝혔습니다.

- **Performance Highlights**: 결과적으로, 본 연구는 tilted empirical risk를 통한 일반화 오류의 경계값을 제공합니다. 이 접근 방식은 불한정 손실 함수(unbounded loss functions)에 대한 일반화 오류를 기존의 PAC-Bayes 이론을 통해 연구하여 더 많은 가능성을 열어주고 있습니다.



### 'Simulacrum of Stories': Examining Large Language Models as Qualitative Research Participants (https://arxiv.org/abs/2409.19430)
- **What's New**: 이 연구는 LLMs(대규모 언어 모델)을 사용해 연구 참가자를 시뮬레이션할 경우 발생하는 윤리적 및 인식론적 문제를 탐구합니다. 저자들은 19명의 질적 연구자와의 인터뷰를 통해 LLM 데이터의 활용 가능성과 한계에 관한 시각 변화를 이해하고자 했습니다.

- **Technical Details**: 연구자들은 LLM이 생성한 데이터가 인간 참가자로부터 수집한 데이터와 유사한 내러티브를 포함하고 있음을 발견했습니다. 그러나 대화가 진행됨에 따라, LLM의 응답에는 중요한 한계가 발견되었으며, 이러한 한계에는 (1) LLM의 응답이 실질적이지 않음, (2) 모델의 인식적 위치의 모호성, (3) 연구자의 위치성이 강화됨, (4) 참가자의 동의 및 에이전시가 제한됨, (5) 커뮤니티의 관점이 지워짐, (6) 질적 연구 방법의 정당성이 훼손될 위험이 포함됩니다.

- **Performance Highlights**: 연구 결과는 LLM을 사용한 대체 연구 방식이 질적 연구의 특성과 윤리에 미치는 부정적인 영향을 강조합니다. LLM은 여전히 텍스트 생성 능력이 뛰어나지만, 지식 생산에 필요한 구체적이고 맥락이 포함된 이해가 부족하다는 점이 강조되었습니다.



### A Proximal Modified Quasi-Newton Method for Nonsmooth Regularized Optimization (https://arxiv.org/abs/2409.19428)
- **What's New**: 본 논문에서는 수정된 quasi-Newton 방법인 R2N을 제안하여 비선형 최적화 문제에서의 새로운 접근 방식을 제공합니다. 이 방법은 신뢰구간(Trust-Region) 알고리즘에 비해 더 유연한 구조를 갖추고 있습니다.

- **Technical Details**: R2N은 $
abla f$의 국소 Lipschitz 연속성에 의존하지 않고, 낮은 반응도(lower semi-continuous) 함수를 포함하는 다양한 비볼록(nonconvex) 함수의 최적화를 가능하게 합니다. R2N의 각 반복에서는 f의 제곱 모델, h의 모델, 적응형(quadratic adaptive) 정규화 항의 합을 최소화합니다.

- **Performance Highlights**: R2N과 R2DH 알고리즘은 최소 순위 문제, 이미지 잡음 제거(image denoising), 최소 순위 행렬 완성(minimum-rank matrix completion), 비선형 서포트 벡터 머신(nonlinear support vector machine) 문제에서 우수한 성능을 보여주었습니다. 이듬해적인 계산 복잡도 제한도 제시되어 O(1/ε^2/(1-p))의 공간에서 성능을 입증했습니다.



### Machine Learning Operations: A Mapping Study (https://arxiv.org/abs/2409.19416)
Comments:
          CSCI'24

- **What's New**: 이 논문은 MLOps(머신러닝 운영)의 다양한 구성 요소에서 발생하는 문제를 다루고, 이를 개선하기 위한 도구나 솔루션을 제안합니다.

- **Technical Details**: MLOps의 주요 구성 요소는 데이터 조작 파이프라인, 모델 구축 파이프라인 및 배포 파이프라인입니다. 체계적인 매핑 연구를 통해 각각의 초점 영역별로 발생하는 도전 과제를 식별하였습니다.

- **Performance Highlights**: 이 연구에서 제공하는 가이드라인은 특정 도구에 국한되지 않으며, 연구 및 산업 현장에서 모두 적용할 수 있는 해결책을 제시합니다.



### How much do we really know about Structure Learning from i.i.d. Data? Interpretable, multi-dimensional Performance Indicator for Causal Discovery (https://arxiv.org/abs/2409.19377)
- **What's New**: 비관측적 데이터로부터의 비선형 원인 발견(causal discovery)은 구조적 방정식(structural equations)의 식별 가능성(identifiability) 조건을 엄격하게 요구합니다. 본 논문에서는 이러한 조건 위반을 평가하기 위한 새로운 여섯 차원 평가 지표, 즉 최적 해에 대한 거리(DOS)를 제안합니다.

- **Technical Details**: 이 연구는 일곱 가지 다른 형태의 구조 학습 알고리즘을 비선형 원인 패턴에 대해 평가하며, 대규모 시뮬레이션 연구를 통해 원인 발견 기술(causal discovery techniques)의 성능과 실험적인 요인 간의 상호작용 효과(interactions effects)를 분석합니다. 이 특별한 평가 프레임워크는 구조적 유사성 뿐만 아니라, 추론에 사용 가능한 그래프의 용량을 정량화합니다.

- **Performance Highlights**: 우리의 연구 결과에 따르면, 인과 차수(causal order)에 기반한 방법 외에도, 비용 분산(causal discovery) 방법이 비교적 높은 최적 해에 대한 접근성을 보여주었습니다. 이와 함께, 비참조 설정에서의 그래프 품질을 분석하는 데 있어 새로운 다차원 접근법이 필요함을 강조합니다.



### Learning Strategy Representation for Imitation Learning in Multi-Agent Games (https://arxiv.org/abs/2409.19363)
Comments:
          13 pages, 7 figures. arXiv admin note: substantial text overlap with arXiv:2402.18617

- **What's New**: 이번 연구에서는 다중 에이전트 게임에서 모방 학습을 향상시키기 위한 STRIL(Strategy Representation for Imitation Learning) 프레임워크를 도입하였습니다. STRIL은 각 시퀀스를 독특한 전략 표현으로 할당하고 이를 통해 비최적 데이터 샘플을 필터링하며, 기존 IL 알고리즘과 호환 가능합니다.

- **Technical Details**: STRIL은 반올림 훈련 조건을 갖춘 변분 순환 신경망(Partially-trainable-conditioned Variational Recurrent Neural Network, P-VRNN)을 사용하여 다중 에이전트 게임 시퀀스로부터 전략 표현을 효율적으로 추출합니다. 이 프레임워크는 무보상 데이터에서도 효과적으로 오프라인 경과를 평가하기 위해 무작위성 지표(Randomness Indicator, RI)와 착취 수준(Exploited Level, EL)을 정의합니다.

- **Performance Highlights**: STRIL은 두 플레이어의 퐁(Two-player Pong), 리밋 텍사스 홀덤(Limit Texas Hold'em), 그리고 연결 사각형(Connect Four)과 같은 경쟁적인 다중 에이전트 시나리오에서 기존의 IL 성능을 크게 향상시키는 데 성공했습니다.



### Quantum delegated and federated learning via quantum homomorphic encryption (https://arxiv.org/abs/2409.19359)
Comments:
          5 pages, 1 figure, 1 table

- **What's New**: 이번 연구에서는 양자(Quantum) 학습 모델이 고전적인 방법보다 계산적인 이점을 제공할 수 있다는 점을 강조합니다. 또한, 클라우드에서 강력한 양자 서버가 사용 가능해짐에 따라 고객의 개인 데이터 보호가 얼마나 중요한지를 설명합니다.

- **Technical Details**: 양자 동형암호(Quantum Homomorphic Encryption) 기법을 통합하여 임의적인 데이터 프라이버시 보장을 통한 양자 위임 및 연합 학습을 위한 일반적인 프레임워크를 제안합니다. 이 프레임워크에서는 블라인드 양자 컴퓨팅(Blind Quantum Computing) 기반의 기법들에 비해 현저히 낮은 통신 복잡성(Communication Complexity)을 가지고 있습니다. 또한, 제안된 양자 연합 학습 시나리오에서는 서버가 암호화된 양자 데이터에서 정보를 추출하지 않고도 작업할 수 있어 클라이언트 쪽 로컬 양자 장치의 계산 부담이 덜합니다.

- **Performance Highlights**: 감독 학습(Supervised Learning)에서 특정 양자 속도 향상(Quantum Speedups)이 개인화된 위임 학습 시나리오에도 적용될 수 있음을 증명합니다. 이 연구 결과는 클라우드에서 개인정보 보호가 보장된 양자 학습을 위한 귀중한 지침을 제공하며, 향후 연구 및 보안 관련 응용 분야에 기여할 가능성이 있습니다.



### Visual Question Decomposition on Multimodal Large Language Models (https://arxiv.org/abs/2409.19339)
Comments:
          Accepted to EMNLP2024 Findings

- **What's New**: 본 논문은 멀티모달 대형 언어 모델(MLLMs)의 질문 분해(Question Decomposition) 능력을 탐구하고 있으며, 기존의 단일 모드 언어 모델과는 달리 MLLMs의 응답 품질을 향상시키는 새로운 데이터셋인 DecoVQA+를 제안하고 있습니다.

- **Technical Details**: 시스템적 평가 프레임워크를 도입하여 MLLMs의 분해된 하위 질문의 품질을 평가하고, 선택적 분해(selective decomposition)를 위한 훈련 목표를 포함하는 효율적인 파인튜닝(pipeline)을 제안하고 있습니다.

- **Performance Highlights**: 파인튜닝된 MLLMs는 VQA 벤치마크 데이터셋에서 하위 질문 품질의 현저한 개선과 선택적 질문 분해에서 더 높은 정확도를 달성했습니다.



### Distributed Optimization via Energy Conservation Laws in Dilated Coordinates (https://arxiv.org/abs/2409.19279)
Comments:
          10 pages; (Near) optimal convergence rate

- **What's New**: 본 논문은 여러 에이전트가 개별 데이터를 공유하는 시스템에서 분산 최적화(distributed optimization) 문제를 최적화하는 새로운 방법론을 제시합니다. 특히 연속 시간 동역학 시스템을 포괄적으로 분석할 수 있는 에너지 보존(energy conservation) 접근법을 소개합니다.

- **Technical Details**: 이 방법은 원래 좌표계에서 직접적으로 동역학을 분석하는 대신, 확대된 좌표계에서 물리적인 에너지와 같은 보존량을 설정하여, 수렴 속도를 시간 왜곡 인수(inverse time-dilation factor)로 명시적으로 표현합니다. 제안된 이론을 바탕으로 새로운 2차 분산 가속 경량 흐름(second-order distributed accelerated gradient flow)을 정의하였고, 이에 대한 수렴 속도는 O(1/t^{2-ε})로 나타났습니다. 또한 반 이차 심플렉틱 오일러 이산화 방식을 사용하여 O(1/k^{2-ε})의 수렴 속도를 가진 알고리즘을 도출하였습니다.

- **Performance Highlights**: 이 알고리즘은 매끄러운 볼록 최적화를 위한 모든 분산 최적화 알고리즘 중에서 최고의 수렴 속도를 제공합니다. 실제 대규모 문제에 대해서 여러 최신 상태의 분산 최적화 알고리즘과의 비교를 통해 가속화된 수렴 행동을 검증하였습니다.



### Decoding Android Malware with a Fraction of Features: An Attention-Enhanced MLP-SVM Approach (https://arxiv.org/abs/2409.19234)
Comments:
          Accepted for NSS-SocialSec 2024, Lecture Notes in Computer Science (LNCS)

- **What's New**: 본 논문에서는 Android 악성코드 탐지 및 분류를 위한 혁신적인 프레임워크를 제안합니다. 이 프레임워크는 attention이 강화된 다층 퍼셉트론(Multi-Layer Perceptron, MLP)과 서포트 벡터 머신(Support Vector Machine, SVM)을 통합하여 높은 정확도를 달성합니다. 오직 47개의 특징만을 분석하여 99% 이상의 놀라운 정확도를 보이는 결과를 도출했습니다.

- **Technical Details**: 제안된 모델은 MLP를 기반으로 하여 47개 특징 중에서 14개로 차원 축소를 수행하고, SVM은 RBF 커널(Radial Basis Function kernel)을 활용하여 이를 고차원 공간에 매핑하여 악성코드를 정확하게 분류합니다. Linear Discriminant Analysis (LDA)를 이용해 특징을 더욱 정교화하는 과정을 포함합니다. 해당 접근법은 XAI(Explainable AI)를 통해 해석 가능성을 높이며, SHAP(Shapley Additive exPlanations) 기술을 사용하여 피쳐의 중요성을 평가합니다.

- **Performance Highlights**: 강력한 평가 결과에 따르면, 제안한 MLP-SVM 프레임워크는 기존의 최첨단 기술들에 비해 뛰어난 성능을 보이며, 다수의 메트릭에 대해 우수한 결과를 기록했습니다. 연구 결과, 정확도, 정밀도, 재현율, F1-스코어 측면에서 우리가 제공하는 접근법이 기존 방법들보다 월등하다는 사실이 입증되었습니다.



### Group Distributionally Robust Optimization can Suppress Class Imbalance Effect in Network Traffic Classification (https://arxiv.org/abs/2409.19214)
- **What's New**: 이 논문은 네트워크 트래픽 분류에서 클래스 불균형(class imbalance)을 다루는 새로운 전략을 제안합니다. 특히, 그룹 분포적으로 강건한 최적화(group distributionally robust optimization) 관점에서 접근하여, 클래스 간 가중치를 동적으로 할당하고 손실 함수를 최적화합니다.

- **Technical Details**: 저자는 비모수(non-parametric) 방식으로 각 클래스의 가중치를 반복적으로 업데이트하고, 재가중치 손실(reweighted losses)을 최소화하는 방식으로 학습 모델을 최적화합니다. 이 과정은 Stackelberg 게임(Stackelberg game)으로 해석되며, 각 클래스의 성능 균형을 위한 그룹 비용 가중치를 안정적으로 조정하는 방법을 제안합니다.

- **Performance Highlights**: 제안된 방법은 클래스 불균형의 부정적 영향을 억제할 뿐만 아니라, 네트워크 트래픽 분류에서 기존 모델들보다 우수한 성능을 달성하는 것으로 나타났습니다. 실험 결과, 복잡한 하이퍼파라미터 조정 없이도 기존의 일반적인 기준선 대비 뛰어난 성과를 보였습니다.



### Faster Acceleration for Steepest Descen (https://arxiv.org/abs/2409.19200)
- **What's New**: 이번 연구에서는 비유클리드(Non-Euclidean) 매끄러움 가정 하에서 선형 최적화 문제를 위한 새로운 가속된 1차 방법을 제안합니다. 이 방법은 서로 다른 노름(Norm)에 대해 원점-쌍대(iterate sequences)를 활용하고, 암묵적으로 결정된 보간 파라미터(interpolation parameter)를 결합합니다.

- **Technical Details**: 제안된 방법은 d 차원에서의 ℓᵖ 노름(ℓ_p norm) 매끄러운 문제에 대해 1차 오라클(First-order oracle) 호출 회수에 대해 최대 O(d^{1-2/p})의 반복 복잡도(iteration complexity) 개선을 제공합니다. 이 접근법은 함수의 로컬 속성에 따라 암묵적으로 파라미터를 선택하며, 이는 Nesterov의 초기/기초(Estimate sequence-type methods) 방식과 유사하게 진행됩니다.

- **Performance Highlights**: Hyper-Accelerated Steepest Descent (HASD) 알고리즘은 이전의 결과를 O(d^{1-2/p})만큼 개선하여 가속화된 비유클리드 최적화 문제의 해결에 중요한 이정표를 제공합니다. 이 새로운 접근법은 기존의 대칭 구조와 더불어 파라미터 선택 방식에 중요한 차별성을 나타냅니다.



### Learning-Based Image Compression for Machines (https://arxiv.org/abs/2409.19184)
- **What's New**: 이번 연구는 머신 러닝 기반 이미지 압축 기법을 통해 기존의 압축 방법보다 향상된 성능을 보여주지만, 표준화 부족과 중요한 특징이 보존되지 않아 머신 러닝 파이프라인에서 널리 채택되지 못하고 있는 문제를 다룹니다. 우리는 압축 과정에 다운스트림 태스크를 통합하는 방법을 제안하고, 선훈련된 압축 인코딩 파이프라인의 여러 부분을 미세 조정하여 시각적 태스크에서의 성능을 향상시키는 결과를 보고합니다.

- **Technical Details**: 연구에서는 이미지 압축과 비디오 압축의 중요성을 강조하며, 머신 러닝 분석을 통한 이미지 및 비디오의 활용 가능성을 탐구합니다. 압축 파이프라인을 갖춘 모델은 압축된 이미지의 잠재 표현(latent representation)을 입력으로 받아, 분류(classification) 모듈과 함께 공동 학습(joint training)될 수 있습니다. 특히, BMSH-J2018 하이퍼프라이어 모델을 사용하여 텍스처 인식 이미지의 압축을 진행하고, cResNet-39 모델을 통해 압축된 이미지를 분류합니다.

- **Performance Highlights**: 실험 결과, 압축 및 분류 모듈의 공동 훈련이 더 나은 성능을 발휘함을 보여주며, 특히 생리학적 거리(threshold distances)에 맞춘 다양한 비트 출력을 기반으로 한 모델 간 비교에서 인간과 머신 모두의 효율성을 극대화하는 가능성을 확인했습니다.



### Reducing Overtreatment of Indeterminate Thyroid Nodules Using a Multimodal Deep Learning Mod (https://arxiv.org/abs/2409.19171)
Comments:
          9 pages, 3 figures

- **What's New**: 이 논문에서는 초음파 영상을 활용한 Attention Multiple Instance Learning (AMIL) 모델을 개발하여 불확실한 갑상선 결절의 양성과 악성을 분류하고, 기존의 분자 검사(Molecular Testing, MT)의 위양성을 줄이는 방법을 제시합니다.

- **Technical Details**: 이 연구는 UCLA 의료센터에서 발견된 불확실한 갑상선 결절을 가진 333명의 환자를 복 retrospectively 검토하였으며, AMIL 모델을 개발하여 초음파 영상과 MT를 결합하여 결절을 양성 또는 악성으로 분류했습니다. 이 모델은 다중 인스턴스 학습(MIL)을 기반으로 한 것이며, 게이티드 어텐션 메커니즘(gated attention mechanism)을 사용하여 각 스캔의 기여도를 평가합니다.

- **Performance Highlights**: 최종 AMIL 모델은 MT의 민감도(0.946)와 일치하며, 긍정 예측 값(PPV)을 크게 개선하여 0.477로 나타났습니다. 이는 높은 민감도를 유지하면서 더 적은 위양성을 의미하며, 불확실한 결절을 가진 환자에서 불필요한 양성 갑상선 절제를 줄일 potential benefits를 제공합니다.



### Multimodal Pragmatic Jailbreak on Text-to-image Models (https://arxiv.org/abs/2409.19149)
- **What's New**: 이번 연구는 텍스트-이미지(T2I) 모델에 대한 새로운 유형의 jailbreak을 소개하며, 둘 이상의 안전한 요소가 결합해 위험한 콘텐츠를 생성하는 현상을 탐구합니다.

- **Technical Details**: 제안된 Multimodal Pragmatic Unsafe Prompts (MPUP) 데이터셋은 1,200개의 위험한 프롬프트로 구성되어 있으며, 아홉 개의 대표적인 T2I 모델을 벤치마킹합니다. 실험 결과, 모든 모델이 8%에서 74%의 비율로 위험한 콘텐츠를 생성하는 취약성을 보였습니다.

- **Performance Highlights**: 현재의 안전 필터가 이러한 새로운 jailbreak에 대해 효과적이지 않음을 밝혀냈으며, 모델들이 제작한 위험한 콘텐츠의 복잡성을 감지하는 데 한계가 있음을 강조합니다.



### From Vision to Audio and Beyond: A Unified Model for Audio-Visual Representation and Generation (https://arxiv.org/abs/2409.19132)
Comments:
          Accepted by ICML 2024

- **What's New**: 비디오에서 시각적 요소와 청각적 요소 간의 상호 작용을 연구하기 위한 새로운 통합 프레임워크인 'Vision to Audio and Beyond (VAB)'를 소개합니다. 이 프레임워크는 잠재 공간(latent space)에서 오디오와 비주얼의 표현 학습 및 생성 모델링을 수행합니다.

- **Technical Details**: VAB 모델은 사전 훈련된 음성 토크나이저와 이미지 인코더를 사용하여 오디오 토큰과 시각적 기능을 추출합니다. VAB는 시각적으로 조건화된 마스킹된 오디오 토큰 예측을 수행하는 사전 훈련 작업을 포함하며, 이로써 비디오에서 오디오를 생성하는 동시에 맥락 학습이 가능합니다. 또한, 이 모델은 다양한 오디오-비주얼 다운스트림 작업을 위해 미세 조정(fine-tuning)될 수 있습니다.

- **Performance Highlights**: VAB 모델은 정지된 비디오에서 고품질 오디오를 효율적으로 생성할 수 있으며, 기존 자동 회귀 접근방식보다 17배 빠른 속도를 자랑합니다. 실험 결과, VAB는 오디오-비주얼 검색 및 분류 작업에서 경쟁력 있는 성능을 보였습니다.



### Multi-modal Cross-domain Self-supervised Pre-training for fMRI and EEG Fusion (https://arxiv.org/abs/2409.19130)
- **What's New**: 이번 연구에서는 fMRI(기능적 자기공명영상)와 EEG(전기뇌파) 데이터를 통합하여 뇌 질환의 패러다임을 새롭게 제시하는 Multi-modal Cross-domain Self-supervised Pre-training Model (MCSP)을 개발하였습니다. 이 모델은 여러 도메인에서의 상호작용을 완전히 포착하기 위해 Self-supervised learning을 활용하며, 다양한 도메인 간의 시너지 정보를 극대화하는 방법을 제안합니다.

- **Technical Details**: MCSP는 Cross-domain self-supervised loss (CD-SSL)와 Cross-modal self-supervised loss (CM-SSL) 두 가지 손실 함수를 도입하여, fMRI와 EEG의 서로 다른 측면을 효과적으로 통합합니다. CD-SSL은 도메인 특화 데이터 증강 및 Contrastive learning 기술을 적용하여 도메인 간 유사성을 극대화합니다. CM-SSL은 fMRI와 EEG 간의 보완적 특성을 활용하여 서로 풍부한 정보를 증류합니다.

- **Performance Highlights**: 실험 결과, MCSP 모델은 다양한 분류 작업에서 우수한 성능을 입증하였으며, 정신 질환 연구의 맥락에서 fMRI와 EEG의 융합이 가져오는 신규 통찰력을 기반으로 깊은 학습 설계의 중요성을 강조합니다. 이를 통해 위상별, 시간별, 주파수별 특성을 모두 극대화하는 멀티모달 신경영상 분석의 잠재력을 확인할 수 있었습니다.



### CURATE: Scaling-up Differentially Private Causal Graph Discovery (https://arxiv.org/abs/2409.19060)
- **What's New**: 본 논문에서는 기존의 Differential Privacy (DP) Causal Graph Discovery (CGD) 알고리즘의 한계를 극복하기 위해 CURATE라는 새로운 프레임워크를 제안합니다. CURATE는 민감도에 따른 적응형 프라이버시 예산을 통해 예측 성능을 높이고 개인 정보 유출을 최소화합니다.

- **Technical Details**: CURATE 프레임워크는 제약 기반(iterative-based)과 점수 기반(score-based) CGD 알고리즘 모두에 대해 적응형 프라이버시 예산 할당을 제공합니다. 제약 기반 알고리즘의 경우 초기 CI 테스트에 우선적으로 높은 프라이버시 예산을 배분하고, 점수 기반 알고리즘의 경우 최적화 과정의 후속 반복에서 점진적으로 예산을 증가시킵니다. 이를 통해 전반적인 오류 확률을 감소시키고, 각 알고리즘의 성능을 개선하는 것을 목표로 합니다.

- **Performance Highlights**: CURATE는 6개의 공개 CGD 데이터셋에서 기존의 DP-CGD 알고리즘들과 비교하여 더 높은 예측 성능을 제공하고, 개인 정보 유출이 현저히 낮은 결과를 보여줍니다. 제약 기반 CURATE 알고리즘에서 필요한 CI 테스트 수 또한 기존 알고리즘보다 유의미하게 감소했습니다.



### On the Inductive Bias of Stacking Towards Improving Reasoning (https://arxiv.org/abs/2409.19044)
Comments:
          Accepted at NeurIPS 2024

- **What's New**: 이 논문에서는 새로운 훈련 전략인 MIDAS를 제안합니다. MIDAS는 언어 모델 훈련을 최대 40% 가속할 수 있으며, 특히 추론 능력이 필요한 다운스트림 작업에서의 성능 향상을 제공합니다.

- **Technical Details**: MIDAS는 기존의 점진적 스태깅(gradual stacking) 기법의 변형으로, 작은 네트워크의 중간 블록을 복사하여 큰 네트워크를 초기화하는 방식입니다. 이 방법은 표준 훈련 방식과 유사한 데이터와 FLOPS(부동소수점 연산 수)를 사용하면서도 우수한 성능을 보입니다.

- **Performance Highlights**: MIDAS는 수학 문제와 추론 프리미티브 등에서 뛰어난 성능을 발휘하며, 표준 방법과 성능 비교에서 더 나은 결과를 나타냈습니다. 특히, MIDAS로 사전 훈련된 모델은 표준 훈련 모델보다 추론 프리미티브에서 유의미한 개선을 보였습니다.



### Intention-aware policy graphs: answering what, how, and why in opaque agents (https://arxiv.org/abs/2409.19038)
Comments:
          57 pages, 8 figures, 5 tables

- **What's New**: 이번 연구에서는 복잡한 환경에서 상호작용하는 AI 기반 소프트웨어인 에이전트의 출현 행동(emergent behaviour)을 설명하는 새로운 방법을 제안합니다. 이는 신뢰할 수 있는 AI를 배치하는 데 필수적인 요소입니다.

- **Technical Details**: 우리는 에이전트의 행동을 신중하게 고찰할 수 있도록 하는 Probabilistic Graphical Model(확률적 그래픽 모델)과 이를 설계하기 위한 파이프라인을 제안합니다. 이 모델은 에이전트가 현재 갖고 있는 의도(intention)에 대한 강력한 수치적 값을 산출할 수 있게 해줍니다.

- **Performance Highlights**: 제안된 모델을 통해 에이전트의 행동과 세계 상태에 대한 부분적인 관찰을 기반으로 해석 가능성과 신뢰성을 평가하는 측정을 제시합니다. 이를 통해 '지금 무엇을 하고 싶습니까?'(예: 수프 배달), '어떻게 계획하고 있습니까?'(예: 자신의 기술과 세계를 고려한 계획 반환), '왜 이 상태에서 이 행동을 취합니까?'(예: 자신의 목표를 추진하거나 방해하는 방식으로 설명) 등의 질문을 가능하게 합니다.



### Application of AI-based Models for Online Fraud Detection and Analysis (https://arxiv.org/abs/2409.19022)
Comments:
          Manuscript under peer review. Content may be subject to revision in future versions

- **What's New**: 이번 논문에서는 AI 및 NLP 기술을 활용한 온라인 사기 감지를 위한 체계적인 문헌 검토(Systematic Literature Review, SLR)를 수행하였습니다. 온라인 사기의 정의와 그로 인한 피해, 그리고 AI 기술이 사기 탐지에 응용되는 현재 상태가 중점적으로 분석되었습니다. 특히, 다양한 사기 유형들에 대한 연구 결과를 종합하여 정책입안자, 법 집행 기관 및 기업이 사기 예방 및 방지에 있어 어떻게 대응할 수 있을지를 제시합니다.

- **Technical Details**: 이번 연구는 PRISMA-ScR 프로토콜을 준수하여 2,457개의 학술 기록을 스크리닝한 결과, 350개의 연구가 적격한 기준을 만족하였고 최종적으로 223개의 연구가 포함되었습니다. 이 연구는 NLP 알고리즘 및 모델, 훈련 데이터 출처, 성과 평가 지표 등을 포함하여 다양한 온라인 사기 카테고리에 대한 최첨단 연구 결과를 보고하였습니다.

- **Performance Highlights**: 연구 결과, 현재 온라인 사기에 대한 연구는 여러 가지 사기 활동으로 나뉘어 있으며, 연구자들은 16가지 서로 다른 사기에 초점을 맞추고 있습니다. 특히, 모델 효율성 제고를 위한 데이터의 한계, 훈련 편향 보고 및 선택적 성과 지표 발표와 같은 문제를 식별하였으며, 이는 모델 평가에서 잠재적인 편향을 초래할 수 있다는 점을 강조하였습니다.



### DiaSynth -- Synthetic Dialogue Generation Framework (https://arxiv.org/abs/2409.19020)
Comments:
          13 pages, 1 figure

- **What's New**: DiaSynth는 다양한 도메인에서 고품질의 맥락이 풍부한 대화를 생성할 수 있는 합성 대화 생성(framework) 프레임워크입니다. 기존의 대화 데이터셋의 결점을 보완하기 위해 소주제(subtopics)와 다양한 대화 특징을 시뮬레이션하는 페르소나(personas)를 동적으로 생성합니다.

- **Technical Details**: DiaSynth는 대화 생성에서 LLM(Large Language Model)과 사고의 연쇄(Chain of Thought) 추론을 활용하여 실질적인 인간 상호작용을 모방하는 맥락적으로 풍부한 도메인 특정 대화를 생성합니다. 각 주제에 대해 소주제를 생성하고, 각 소주제에 대해 페르소나를 생성하여 대화를 만듭니다. 이 과정은 세 단계의 파이프라인을 통해 이루어집니다.

- **Performance Highlights**: DiaSynth를 활용한 합성 데이터는 기존 모델보다 16.47% 성능이 향상 되었고, 합성 데이터는 도메인 데이터 분포의 90.48%를 포착하여 강력한 대안임을 증명합니다.



### RAGProbe: An Automated Approach for Evaluating RAG Applications (https://arxiv.org/abs/2409.19019)
Comments:
          11 pages, 5 figures, 9 tables

- **What's New**: 본 논문에서는 Retrieval Augmented Generation (RAG) 파이프라인의 자동 평가를 위한 RAGProbe라는 새로운 기술을 소개합니다. 기존의 수작업 평가 방식으로 인한 한계를 극복하고, 다양한 질문-답변 쌍의 변형을 생성하여 RAG 파이프라인에서 실패를 유도하는 접근법을 개발했습니다.

- **Technical Details**: RAGProbe는 평가 시나리오를 기반으로 하여 질의-응답 쌍을 생성하고, 각각의 시나리오는 질문-답변 쌍의 다양한 변형을 나타냅니다. 이 평가 시나리오에는 문서 샘플링 및 청킹 전략, 시나리오 специф적인 프롬프트 및 프롬프트 전략, 평가 메트릭스가 포함됩니다.

- **Performance Highlights**: RAGProbe는 5개의 오픈 소스 RAG 파이프라인과 3개의 데이터셋을 사용하여 평가되었으며, 기존의 최신 기술보다 평균 51% 더 높은 실패율을 기록했습니다. 또한, 다수의 질문이 포함된 시나리오에서 91%의 실패율을 보여 RAG 파이프라인의 강화를 위한 필요성을 시사합니다.



### Textless NLP -- Zero Resource Challenge with Low Resource Compu (https://arxiv.org/abs/2409.19015)
- **What's New**: 이 연구는 Textless NLP(텍스트 없는 자연어 처리)에서 경량의 인코더-보코더 모델 학습 시 발생하는 훈련 시간 및 GPU 자원의 문제를 해결하기 위한 방법론을 제안합니다. 주요 기여는 학습률 스케줄러(learning rate scheduler)를 활용하여 훈련 단계를 효과적으로 줄이고 성능을 향상시키는 것입니다.

- **Technical Details**: 이 시스템은 Vector-Quantized Contrastive Predictive Coding(VQ-CPC)을 인코더로 사용하고, LSTM 기반의 경량 보코더를 사용합니다. 학습률 스케줄러를 One-Cycle Learning Rate(OCLR)로 설정하여 훈련 시간을 80%까지 줄이고, 오디오 품질 향상을 위해 hop length와 interpolation scale factors를 최적화했습니다. 또한, 인도어(Indian languages) 데이터셋에 대한 실험이 포함되어 있습니다.

- **Performance Highlights**: 제안된 방법은 English, Tamil, Bengali 데이터셋에서 일관되게 우수한 결과를 보여주었으며, 언어 변환 시 재구성된 오디오의 선명도가 눈에 띄게 향상되었습니다. 훈련 시간도 28시간에서 최적화된 단계 수(30k, 40k, 60k)에 따라 각각 6, 8, 12시간으로 줄어들었습니다.



### FLEX: Expert-level False-Less EXecution Metric for Reliable Text-to-SQL Benchmark (https://arxiv.org/abs/2409.19014)
Comments:
          preprint, under review

- **What's New**: 본 논문에서는 FLEX (False-Less EXecution)라는 새로운 평가 방식을 소개하여, SQL 쿼리의 인간 전문가 수준의 평가를 모방하는 대형 언어 모델(LLM)을 활용하여 텍스트-투-SQL 시스템을 평가한다.

- **Technical Details**: FLEX는 기존의 Execution Accuracy (EX) 방식의 한계를 극복하고, 더 정밀한 평가를 제공하며, SQL 쿼리가 원래 질문과 의미적으로 일치하는지를 분석하여 종합적인 쿼리 정확성을 평가한다. 이는 노이즈가 있는 기준 데이터에 대해서도 유용하다.

- **Performance Highlights**: FLEX를 이용한 평가 결과, 기존의 EX 평가보다 인간 전문가의 판단과의 일치도가 크게 향상되었고, Cohen's kappa는 61에서 78.17로 증가했다. 스파이더(Spider)와 BIRD 벤치마크에서 기존 상위 모델의 성능 순위가 대폭 변경되는 결과가 나타났다.



### Improving Academic Skills Assessment with NLP and Ensemble Learning (https://arxiv.org/abs/2409.19013)
Comments:
          5 pages, 2 figures

- **What's New**: 이번 연구는 자연어 처리(NLP)의 발전을 활용하여 기초 학문 기술을 평가하는 데 있어 주요 도전 과제를 다룹니다. 기존의 전통적 평가 방법은 인지적 및 언어적 측면에서 적시의 포괄적인 피드백을 제공하는 데 어려움이 있었습니다. 본 연구는 BERT, RoBERTa, BART, DeBERTa, T5와 같은 최신 NLP 모델을 통합하여 앙상블 학습(enesemble learning) 프레임워크를 통해 정확성을 크게 향상시켰습니다.

- **Technical Details**: 이 모델은 상태를 유지하기 위해 LightGBM과 Ridge 회귀를 사용하여 여러 NLP 모델을 스택(stacking) 기법으로 결합했습니다. 데이터 전처리, 특징 추출, 그리고 pseudo-label 학습을 통해 모델 성능을 최적화했습니다. 또한, PyTorch 모델 클래스를 사용하여 텍스트 입력을 처리하고 효과적으로 분류 작업을 수행했습니다.

- **Performance Highlights**: 이 연구는 ESL(English as a Second Language) 학생들을 위한 언어 평가의 정확성을 크게 향상시켰으며, 전통적인 평가 방법의 한계를 극복하고 교육 기술 연구에서 핵심 학문 역량 향상에 대한 새로운 가능성을 열어주는 강력한 솔루션을 제공했습니다.



### Identification and Mitigating Bias in Quantum Machine Learning (https://arxiv.org/abs/2409.19011)
Comments:
          2 pages

- **What's New**: 이번 연구에서는 Quantum Machine Learning (QML) 의 고유한 편향(bias) 및 과제를 조명하며, 이러한 편향들이 QML에서 어떻게 식별되고 진단되며 대응될 수 있는지를 다룹니다. 이 논문은 QML의 편향의 세 가지 주요 주제에 대한 포괄적인 개요를 제공합니다.

- **Technical Details**: QML에서 발생하는 여러 가지 편향 유형으로는 Encoding Bias, Inductive Bias, Realizability Bias, State-Dependent Bias, Sampling Bias가 있으며, 각 편향의 원인과 QML 모델 성능에 미치는 영향을 분석하였습니다. 특히 Encoding Bias는 고전 데이터의 양자 상태로의 변환과 양자 알고리즘 간의 상호작용에서 발생합니다. 다양한 인코딩 기법을 통해 MNIST 데이터셋에 대해 실험을 진행하고, 각 인코딩 방식의 성능 차이를 입증하였습니다.

- **Performance Highlights**: Encoding Bias에 대한 실험에서는 Basis Encoding이 모든 에포크(epoch)에서 낮은 정확도를 보였고, Angle Encoding이 빠르게 성능이 향상되어 높은 정확도를 유지했습니다. Hybrid Parameterized Encoding에서 Rx, Ry, Rz의 각 인코딩 방식 또한 성능 차이를 보였으며, Rysubscript𝑅𝑦R_{y}encoding가 가장 우수한 초기 개선을 보여주었습니다. 이러한 연구 결과는 QML 시스템의 편향에 대한 보다 나은 이해를 제공하며, 향후 공정성 고려 사항이 양자 영역으로 확장되어야 함을 시사합니다.



### A comprehensive study of on-device NLP applications -- VQA, automated Form filling, Smart Replies for Linguistic Codeswitching (https://arxiv.org/abs/2409.19010)
- **What's New**: 최근 대형 언어 모델의 발전이 이전에는 불가능했던 새로운 경험들을 온디바이스(온디바이스; on-device) 애플리케이션에서 제공할 수 있는 기회를 열어주었습니다. 본 연구에서는 크게 두 가지 범주로 나누어진 세 가지 새로운 경험을 제안합니다.

- **Technical Details**: 첫 번째 범주는 화면 이해(Screen Understanding)와 관련된 경험으로, 사용자 화면에 나타난 정보에 대한 이해(Visual Question Answering, 자동 양식 채우기)가 포함됩니다. 두 번째 범주는 코드 스위칭(Code-Switching) 기능을 지원하는 스마트_reply(smart replies) 시스템의 확장입니다. 본 연구에서 제안된 첫 번째 작업으로는 화면 기반 이해를 위한 Visual Question Answering과 이전 화면의 맥락을 활용한 자동 양식 채우기 과제가 포함됩니다. 모델은 LayoutLM과 MarkupLM 두 가지 계열로 구성되며, LayoutLMv3는 이미지와 텍스트 토큰을 정렬하여 다중 사전 훈련 과정을 수행합니다.

- **Performance Highlights**: 데이터 수집 및 질문-답변 쌍 생성 방법을 통해 4,500개 이상의 iOS 앱에서 100,000개 이상의 스크린샷을 수집했습니다. 이를 기반으로 모델을 훈련하여 기존 양식 정보 및 시각적 맥락에 대한 정확도를 개선했습니다. 처음으로 제안된 이 연구는 화면 기반으로 질문을 생성하는 작업과 다국어 사용자 정보를 기반으로 한 개인화된 스마트 응답 생성을 포함하여 새로운 응용 프로그램을 탐구합니다.



### Towards Automated Patent Workflows: AI-Orchestrated Multi-Agent Framework for Intellectual Property Management and Analysis (https://arxiv.org/abs/2409.19006)
Comments:
          This is a preprint and current version under peer review

- **What's New**: 본 논문은 PatExpert라는 자율 다중 대화형 프레임워크를 제안하며, 이는 다양하고 복잡한 특허 관련 작업을 최적화하고 자동화하는 데 도움을 줍니다. 프레임워크는 메타 에이전트와 특정 작업을 수행하는 전문 에이전트로 구성되어 있습니다.

- **Technical Details**: PatExpert 프레임워크는 메타 에이전트가 고유한 태스크를 수행하는 여러 전문 에이전트를 조정하는 방식으로 작동합니다. 이 시스템은 Graph Retrieval-Augmented Generation (GRAG)과 같은 고급 기법을 사용하여 지식 그래프를 활용해 정확성과 관련성을 향상시킵니다.

- **Performance Highlights**: PatExpert는 다중 특허 분석, 특허 분류, 청구 생성 및 오류 처리를 포함하여 예상치 못한 복잡한 작업을 자동화함으로써 특허 처리 작업의 효율성과 정확성을 크게 향상시켰습니다.



### Enhancing TinyBERT for Financial Sentiment Analysis Using GPT-Augmented FinBERT Distillation (https://arxiv.org/abs/2409.18999)
Comments:
          Submitted in partial fulfillment of the requirements for Masters in Machine Learning and Artificial Intelligence at Liverpool John Moores University, 97 pages, 1 figure, 14 tables

- **What's New**: 이번 연구는 금융 감정 분석 분야에서 LLM(large language models)인 GPT-4 Omni의 생성 능력을 활용하여 도메인 특화된 합성 훈련 데이터를 만드는 새로운 접근 방식을 제안합니다. 이를 통해 데이터 부족 문제를 해결하고, 더 작은 모델의 성능을 향상시켜 기존 대형 모델과 경쟁력을 갖추게 합니다.

- **Technical Details**: 연구는 FinBERT라는 BERT 기반 모델을 향상시키고, 구조화된 2단계 지식 증류 전략을 통해 소형 트랜스포머 모델인 TinyFinBERT를 개발합니다. GPT-4 Omni를 통해 기존 데이터를 변형하고 새로운 훈련 예제를 생성하여 FinBERT의 정확도를 크게 개선하고, 이를 교사 모델로 사용하여 TinyFinBERT에 지식을 증류합니다.

- **Performance Highlights**: TinyFinBERT는 PhraseBank 데이터셋과 FiQA 2018 Task1 데이터셋으로 훈련 및 평가되어, FinBERT와 유사한 성능을 달성하면서도 훨씬 작고 효율적인 모델임을 입증합니다. 이 연구는 LLM이 금융 감정 분석의 발전에 기여할 수 있는 방법을 보여줍니다.



### From Linguistic Giants to Sensory Maestros: A Survey on Cross-Modal Reasoning with Large Language Models (https://arxiv.org/abs/2409.18996)
- **What's New**: 본 논문은 Cross-Modal Reasoning (CMR)과 관련된 최근의 대규모 언어 모델(LLMs)의 발달을 다루고 있으며, LLMs가 CMR에 미치는 역할을 세분화하여 탐구하는 첫 번째 설문조사로서의 중요성을 강조합니다.

- **Technical Details**: 논문은 LLMs의 네 가지 주요 역할인 Multimodal Fusion Engine, Textual Processor, Cognitive Controller, Knowledge Enhancer를 소개합니다. 또한 Prompt Tuning, Instruction Tuning, Multimodal Pre-training과 같은 방법론을 설명하며 이들이 CMR에 활용되는 방식을 상세히 다룹니다.

- **Performance Highlights**: LLMs는 텍스트, 이미지 및 소리 등 다양한 모드 간의 새로운 정보를 이해하고 추론하는 능력을 통합하여 그들의 성능을 향상시킵니다. CMR의 적용 예시로는 비주얼 질문 응답, 비전-언어 탐색, 이미지 및 비디오 캡셔닝 등이 포함됩니다.



### Efficient and Personalized Mobile Health Event Prediction via Small Language Models (https://arxiv.org/abs/2409.18987)
Comments:
          6 pages, 3 figures

- **What's New**: 이 논문에서는 헬스케어 모니터링에 대한 소형 언어 모델(SLMs)의 능력을 처음으로 조사하였으며, 환경을 보호하면서 개인화된 건강 상태 분석을 위한 가능성을 제시합니다.

- **Technical Details**: 이 연구에서는 TinyLlama(1.1B 파라미터)의 성능을 분석하여 4.31GB의 메모리 사용량과 0.48초의 지연시간(latency)을 나타냈으며, 다른 4개의 최신 소형 언어 모델(SOTA SLMs)보다 우수한 성능을 보였습니다.

- **Performance Highlights**: SLMs는 헬스케어 모니터링의 최신 솔루션으로써 우수한 성능을 보여주며, 특히 CPU 사용량, 지연시간, 메모리 사용량에서 15.5배의 개선을 보였습니다. 또한, 기존 LLMs와 비교하여 실시간 헬스케어 응용 프로그램에 더 적합한 것으로 판단됩니다.



### Harnessing Large Language Models: Fine-tuned BERT for Detecting Charismatic Leadership Tactics in Natural Languag (https://arxiv.org/abs/2409.18984)
Comments:
          The 2024 IEEE 3rd Conference on Information Technology and Data Science, CITDS 2024

- **What's New**: 이 연구는 Charismatic Leadership Tactics (CLTs)를 자연어에서 식별하기 위해 미세 조정된 Bidirectional Encoder Representations from Transformers (BERT) 모델을 사용하는 방법을 탐구합니다. CLTs를 위한 대규모 코퍼스를 기반으로 하여, 본 연구는 자연어에서 이 전술의 존재를 정확히 식별할 수 있는 기계 학습 모델을 훈련하는 방법론을 제시합니다.

- **Technical Details**: 본 연구는 BERT 모델을 특정 CLTs를 대상으로 미세 조정하여 해당 전술의 존재를 텍스트에서 식별합니다. 연구는 BERT로 훈련된 모델이 CLTs를 효과적으로 탐지하는지 평가하고, 다루는 특정 데이터셋에 대해 98.96%의 높은 정확도를 보였습니다.

- **Performance Highlights**: 본 연구의 모델은 자연어 처리(NLP) 기술을 통해 Charismatic Leadership의 언어적 특성을 분석할 수 있는 도구를 개발함으로써 심리학 및 경영 분야의 미래 연구에 기여할 수 있는 잠재력을 가집니다.



### Aligning Robot Navigation Behaviors with Human Intentions and Preferences (https://arxiv.org/abs/2409.18982)
Comments:
          Haresh Karnan's PhD Dissertation A recording of the defense talk can be accessed here: this https URL

- **What's New**: 최근 기계 학습 분야의 발전으로 모바일 로봇의 탐색 능력을 향상시키기 위한 새로운 방법들이 등장하고 있습니다. 이 논문은 특히 자율 모바일 로봇의 네비게이션(autonomous navigation) 행동을 인간의 의도와 선호에 맞추기 위한 기계 학습 방법에 대해 다루고 있습니다.

- **Technical Details**: 이 연구에서는 인간의 탐색 작업을 모방하여 학습하는 새로운 접근 방식을 제안합니다. 이를 통해 모바일 로봇은 목표를 인간처럼 인식하고 행동할 수 있도록 독자적인 시각적 네비게이션 능력을 습득하게 됩니다. 특히, Learning from Preferences (lfp)라는 패러다임을 통해 로봇이 환경에서 간단히 선호를 학습하고, 오프로드 탐색 관련 두 가지 알고리즘 또한 도입하여 다양한 지형에서의 탐색 능력을 향상시킵니다.

- **Performance Highlights**: 이 논문은 자율 로봇이 인간의 의도와 선호에 맞춰 자율적으로 탐색할 수 있는 능력을 갖출 수 있도록 돕는 중요한 단계로, 실제 환경에서의 안전한 탐색을 위한 데이터셋과 알고리즘을 제시합니다. 이러한 접근 방식은 가치 정렬(value alignment) 문제를 해결할 수 있는 잠재력을 지니고 있으며, 로봇의 행동이 인간과 조化됩니다.



### Portfolio Stress Testing and Value at Risk (VaR) Incorporating Current Market Conditions (https://arxiv.org/abs/2409.18970)
Comments:
          arXiv admin note: text overlap with arXiv:2205.00605

- **What's New**: 이 논문은 Value at Risk (VaR)와 스트레스 테스트 방법에 현재의 시장 조건을 통합하는 새로운 접근 방식을 제시합니다. 이는 포트폴리오 리스크를 보다 정확하고 실질적으로 분석할 수 있도록 도와줍니다.

- **Technical Details**: 포트폴리오 가치 변화를 전달하는 다양한 시장 조건의 클러스터를 식별하기 위해 Variational Inference (VI)라는 기계 학습 방법을 사용합니다. 이 접근 방식은 과거 c데이터를 기반으로 현재의 시장 조건과 유사한 과거 기간의 데이터를 더 높은 비중으로 반영하여 VaR를 계산합니다.

- **Performance Highlights**: 제안된 접근 방식은 2020년 Covid 관련 변동성이 큰 시기를 통해 성능을 입증하며, VaR 및 스트레스 시나리오가 변화하는 시장 조건에 어떻게 신속하게 적응하는지를 보여줍니다.



### Safety challenges of AI in medicin (https://arxiv.org/abs/2409.18968)
- **What's New**: 최근 인공지능(AI) 및 딥 러닝, 대형 언어 모델(LLM)에서의 발전은 의료 분야에 통합되는 속도를 빠르게 하고 있지만, 안전한 적용에 대한 우려도 증가하고 있습니다. 본 리뷰는 의료에서의 AI 안전성 문제를 다루고, LLM의 특화된 안전 문제를 탐구합니다.

- **Technical Details**: 의료에서 AI의 적용에서 나타나는 주요 문제는 신뢰성(reliability)과 정렬(alignment)입니다. 신뢰성 문제에는 데이터 조화(data harmonization), 일관된 성능, 모델 보정(calibration), 일반화(generalization), 편향(bias) 등이 포함됩니다. AI 정렬은 AI가 인간이 정의한 목표에 맞게 작용하는 것을 보장하는 것으로, 의사결정에서 목표의 잘못된 지정(mis-specification) 문제를 다루고 있습니다.

- **Performance Highlights**: AI 모델의 성과는 다양한 인구집단에 대한 성능 저하, 모델 개발 및 배포 중 데이터 유출의 위험, 그리고 LLM의 한계(예: 복잡한 논리 처리)에 의해 영향을 받을 수 있으며, 이러한 문제들이 AI의 의료 분야 내 안전성 논의에서 중요한 역할을 합니다.



### Agent-state based policies in POMDPs: Beyond belief-state MDPs (https://arxiv.org/abs/2409.15703)
- **What's New**: 본 논문은 POMDP(Partially Observable Markov Decision Process)의 전통적인 접근 방식을 재조명하며, 에이전트가 지역적으로 갱신 가능한 에이전트 상태를 유지하고 이 상태를 기반으로 행동을 선택하는 모델로 통합적 접근을 제시합니다.

- **Technical Details**: 이 논문에서는 에이전트 상태 기반의 정책(policy) 클래스와 각 클래스 내에서 좋은 정책을 찾기 위해 제안된 여러 방법을 강조합니다. 논의된 방법에는 최적의 비정상 에이전트 상태 기반 정책을 찾기 위한 디자이너 접근법, 국소 최적의 정상 에이전트 상태 기반 정책을 찾기 위한 정책 탐색(policy search) 접근법, 그리고 근사 정보 상태(approximate information state)를 활용한 근사적 최적 정상 에이전트 상태 기반 정책 찾기가 포함됩니다.

- **Performance Highlights**: 본 연구는 근사 정보 상태 접근법에서의 아이디어들이 Q-learning 및 actor-critic 알고리즘을 개선하는 데 어떻게 사용되었는지를 소개하며, POMDP에서 학습 성능 향상에 기여하고 있음을 강조합니다.



### Morph-SSL: Self-Supervision with Longitudinal Morphing to Predict AMD Progression from OC (https://arxiv.org/abs/2304.08439)
- **What's New**: 이 연구는 중간 단계 나이 관련 황반 변성(iAMD)에서 신생 혈관형 나이 관련 황반 변성(nAMD)으로의 전환을 예측하기 위한 새로운 Deep Learning (DL) 모델인 Morph-SSL을 개발했습니다. 기존의 신뢰할 수 있는 바이오마커의 부족으로 이러한 예측이 어려운 문제를 해결하고자 합니다.

- **Technical Details**: Morph-SSL은 Self-supervised Learning (SSL) 방법으로, 서로 다른 방문 시기의 비표기 OCT 스캔 쌍을 사용합니다. 이 방식은 이전 방문의 스캔을 다음 방문으로 변형하는 과정을 포함하며, Decoder가 이를 예측합니다. 이 모델은 연속적인 특성의 매니폴드(manifold)를 보장하여 방문 간의 중간 스캔을 생성할 수 있도록 선형 보간(linear interpolation)을 사용합니다. 그 후, Morph-SSL로 학습된 특성은 분류기를 통해 전환 시점에 대한 누적 확률 분포를 모델링합니다.

- **Performance Highlights**: Morph-SSL은 399개의 눈의 비표기 스캔(3570 방문)에 대해 학습되었으며, 343개의 눈에서 변환 날짜에 대한 임상 레이블이 있는 2418개의 스캔을 사용하여 5배 교차 검증(five-fold cross-validation)을 통해 평가되었습니다. Morph-SSL 특성은 향후 6개월 내 nAMD로의 전환을 예측하는 데 AUC 0.766을 달성하여, 기존의 SSL 방법으로 사전 학습되거나 처음부터 끝까지 학습된 동일한 네트워크보다 뛰어난 성능을 보였습니다.




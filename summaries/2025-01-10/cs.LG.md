New uploads on arXiv(cs.CL)

### A survey of textual cyber abuse detection using cutting-edge language models and large language models (https://arxiv.org/abs/2501.05443)
Comments:
          37 pages, under review in WIREs Data Mining and Knowledge Discovery

- **What's New**: 이번 논문에서는 소셜 미디어의 온라인 남용 형태에 대한 포괄적인 분석을 제공하며, 특히 최신 기술이 이러한 남용 내용을 탐지하고 생성하는 방식을 어떻게 재편성하는지에 초점을 맞추고 있습니다. 기존에 잘 연구된 학대 형태 외에도, 최근의 지능형 언어 모델(LLMs)을 통한 분석을 진행하여 보다 넓은 스펙트럼의 사이버 남용을 조명합니다.

- **Technical Details**: 이 논문은 언어 모델과 대형 언어 모델의 역할을 탐구하며, 특히 이들이 사이버 남용과 증오 발언 탐지에서 어떻게 활용될 수 있는지를 분석합니다. LLMs(GPT, BERT 등)는 자연어 처리(NLP)의 사고를 이용해 특정한 사이버 남용 형태를 탐지하고 분류하는 데 새로운 가능성을 제시합니다. 뿐만 아니라, 이 논문은 평가 지표의 불균형 문제를 다루며 사이버 남용 탐지에서의 도전 과제를 심도 있게 분석합니다.

- **Performance Highlights**: LLMs는 사이버 남용 검출 시스템을 자동화하여 기존 방식보다 감소된 수의 잘못된 탐지 및 높은 정확도로 개선된 성과를 보여줍니다. 이 연구는 사이버 괴롭힘 및 증오 발언 외 특정 형태의 사이버 남용에 대한 이해를 깊이 있게 하고, AI 기술이 안전하고 지지적인 온라인 환경을 구축하는 데 어떻게 기여할 수 있는지를 탐구합니다. 이에 따라, 최신 언어 기술이 클라우드 데이터 속에서 악의적인 콘텐츠를 생성할 수 있다는 점도 논의되고 있습니다.



### LongProc: Benchmarking Long-Context Language Models on Long Procedural Generation (https://arxiv.org/abs/2501.05414)
- **What's New**: 기존의 긴 문맥 언어 모델(LCLM) 평가 벤치마크는 주로 짧은 응답 생성을 중점적으로 다루고 있으며, 수많은 불필요한 토큰을 처리하는 데 집중하고 있습니다. 본 연구에서는 LongProc(롱 프로시저 생성)라는 새로운 벤치마크를 소개하며, 정보의 통합과 긴 형식의 생성을 동시에 요구하는 점에서 차별점을 보입니다. LongProc는 HTML 페이지에서 TSV 형식으로 구조화된 정보를 추출하는 것과 여행 계획을 세우는 복잡한 검색 절차 등을 포함한 여섯 가지 다양한 절차 생성 작업으로 구성됩니다.

- **Technical Details**: LongProc의 구조는 정보가 광범위하게 분산되어 있는 경우에도 모델이 세부 절차적 지시를 따르고 여러 정보를 통합하여 구조화된 긴 출력을 생성할 수 있는 능력을 시험합니다. 이 벤치마크는 최대 8K 토큰에 이르는 긴 출력 생성을 요구하며, 결과물은 결정론적 절차에 따라 생성되어 신뢰할 수 있는 규칙 기반 평가가 가능합니다. 평가 대상으로는 17개의 LCLM 모델이 있으며, 각 모델의 출력 토큰 수는 500, 2K, 8K의 세 가지 난이도로 설정되어 있습니다.

- **Performance Highlights**: 모든 테스트된 모델이 32K 이상의 문맥 창 크기를 주장하지만, 오픈웨이트 모델은 2K 토큰 작업에서 실패하는 경향이 있으며, GPT-4o와 같은 폐쇄 소스 모델은 8K 토큰 작업에서 심각한 성능 저하를 보입니다. 추가 분석 결과, LCLM 모델들은 긴 형식의 생성에서 장기적인 일관성을 유지하는 데 어려움을 겪고 있음을 알 수 있습니다. 이러한 결과들은 현재 LCLM의 주요 한계를 강조하며, 상당한 개선 여지가 있음을 시사합니다.



### FairCode: Evaluating Social Bias of LLMs in Code Generation (https://arxiv.org/abs/2501.05396)
- **What's New**: 이번 연구에서는 코드 생성에서의 편향(bias) 평가를 위한 새로운 벤치마크인 FairCode를 소개합니다. FairCode는 함수 구현과 테스트 케이스 생성의 두 가지 작업으로 구성되며, 각 작업은 다양한 시나리오를 통해 사회적 편향을 평가합니다. 이는 대규모 언어 모델(LLMs)이 인간 가치와 일치하도록 설계된 기존 데이터세트가 코드 관련 작업에 최적화되어 있지 않다는 점에서 필요하게 되었습니다.

- **Technical Details**: FairCode 벤치마크는 함수 구현과 테스트 케이스 생성을 통해 LLM의 편향을 평가하는 프레임워크를 제공합니다. 이를 위해 몇 가지 문장을 활용한 프롬프트(few-shot prompting)를 사용하여 모델이 민감한 속성에 따라 평가 점수를 부여하는 코드를 생성하도록 합니다. 새로운 지표인 FairScore를 제안하여 모델의 편향을 수치적으로 평가하고, 생성된 테스트 케이스와 민감한 속성 간의 잠재적 상관관계를 조사합니다.

- **Performance Highlights**: 실험 결과, 모든 테스트된 LLM들이 편향을 보인다는 것을 발견했습니다. 특히, 코드 생성 모델들이 직무 채용, 대학 입학 및 의료 치료와 같은 다양한 상황에서 특정 특성을 기반으로 점수를 부여하는 경향을 보입니다. 이러한 결과는 현재 LLM들이 편향 문제를 해결하기 위한 더욱 효과적인 평가 방법의 필요성을 강조하며, FairCode가 이 분야에 제공할 수 있는 기여를 잘 보여줍니다.



### Stream Aligner: Efficient Sentence-Level Alignment via Distribution Induction (https://arxiv.org/abs/2501.05336)
Comments:
          AAAI Alignment Track 2025 Poster

- **What's New**: 이 논문에서는 Streaming Distribution Induce Aligner (Stream Aligner)라는 새로운 정렬(paradigm) 접근법을 제시합니다. 이 방법은 LLM의 성능을 향상시키면서도 효율성을 유지하는 데 중점을 두고 있습니다. 특히, 문장 수준의 반복적인 수정 과정을 통해 이전 모델의 출력을 개선하며, 이로 인해 적은 자원으로도 더 나은 결과를 도출할 수 있습니다.

- **Technical Details**: Stream Aligner는 정렬 과정을 문장 수준으로 세분화하여, 모든 세션에서 사용자의 선호도를 학습하고 이를 바탕으로 수정된 출력을 기본 모델로 피드백합니다. 이 과정은 기계적 학습(machinery learning) 원리와 선호 데이터셋(preference dataset)을 기반으로 하며, 반복적인 피드백이 발생하는 구조를 가집니다. 이는 인퍼런스 과정 중에 실제로 인간의 의도와 가치를 LLM 출력에 효과적으로 주입할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 2B 모델의 Stream Aligner는 Llama2-70B-chat 모델에서 응답의 유용성을 41.2% 증가시키고, 무해성은 36.0% 향상시켰습니다. 또한, Stream Aligner-8B 모델은 Llama3-70B-Instruct 모델에서 수학적 능력을 3.5% 개선하는 등 성능이 입증되었습니다. 이러한 결과는 Stream Aligner가 추가 모델의 의존도를 줄이며, LLM의 추론 능력을 향상시키는 데 기여함을 나타냅니다.



### Enhancing Plagiarism Detection in Marathi with a Weighted Ensemble of TF-IDF and BERT Embeddings for Low-Resource Language Processing (https://arxiv.org/abs/2501.05260)
Comments:
          Accepted into LoResLM: The First Workshop on Language Models for Low-Resource Languages, colocated with COLING 2025 and set to be published into ACL Anthology

- **What's New**: 본 연구는 Marathi와 같은 자원이 부족한 언어에 대한 표절(plagiarism) 탐지 시스템을 설계하는 데 중점을 두었습니다. 기존의 모델들이 주로 나쁜 성능을 보여온 저자원 언어에서 BERT(Bidirectional Encoder Representations from Transformers)를 활용한 새로운 방법론을 제시합니다.

- **Technical Details**: 이 연구에서는 BERT의 문장 임베딩(sentence embeddings)과 Term Frequency-Inverse Document Frequency (TF-IDF) 기능 표현을 결합하여 Marathi 텍스트의 표절 탐지 정확성을 향상시킵니다. 이러한 접근은 기계 학습 모델의 가중치 투표 앙상블을 통해 텍스트의 통계, 의미 및 구문적 요소를 효과적으로 캡처합니다.

- **Performance Highlights**: 제안된 방법론은 기존 표절 탐지 시스템보다 더 높은 정확성을 보여주며, Marathi와 같은 저자원 언어의 표절 탐지 기술 발전에 기여할 것으로 기대됩니다. 이는 다양한 언어 처리(application) 분야에서 활용될 수 있는 중요한 연구 결과입니다.



### Optimizing Estonian TV Subtitles with Semi-supervised Learning and LLMs (https://arxiv.org/abs/2501.05234)
- **What's New**: 이 논문은 에스토니아 TV 콘텐츠의 높은 품질의 동일 언어 자막 생성 접근법을 제시합니다. Whisper 모델을 인간이 생성한 에스토니아 자막에 맞게 미세 조정하고, 반복적인 pseudo-labeling과 대형 언어 모델(LLM) 기반 후편집을 통해 품질을 향상시켰습니다. 실험 결과, 무언급 데이터셋을 통한 pseudo-labeling에서 자막 품질의 유의미한 개선이 있음을 보였습니다.

- **Technical Details**: 본 연구는 자동 자막 생성 시스템 개발을 위한 방법을 여러 단계로 나누어 설명합니다. 감독 데이터셋을 사용하여 Whisper large-v3 모델을 훈련시키고, 이후 비감독 데이터셋을 활용하여 반복 pseudo-labeling을 적용합니다. LLM 기반 후편집을 통해 생성된 자막의 오류를 수정하며, 이러한 방법을 테스트 시간과 훈련 시간에 각각 적용하여 성능을 평가합니다.

- **Performance Highlights**: 에스토니아 국가 TV의 감독 데이터셋에서 수집된 993개의 오디오-자막 쌍과 7128개의 비감독 오디오 기록을 사용하여 성능을 평가했습니다. 자막 품질 평가를 위해 SubER, t-BLEURT, AS-BLEURT의 세 가지 메트릭을 사용하였으며, 자막의 품질을 인간의 평가와 비교하였습니다. 결과적으로, LLM 기반의 게시 수정이 자막 품질을 크게 향상시킬 수 있음을 확인했습니다.



### Leveraging Large Language Models for Zero-shot Lay Summarisation in Biomedicine and Beyond (https://arxiv.org/abs/2501.05224)
Comments:
          Preprint

- **What's New**: 이번 연구에서는 Large Language Models(LLMs)를 활용하여 zero-shot Lay Summarisation을 적용하는 신방법론을 제안합니다. 이 두 단계의 프레임워크는 실제 프로세스를 기반으로 개발되었으며, 더 큰 모델을 사용함에 따라 인간 평가자들에게 더 선호되는 요약이 생성됨을 발견하였습니다. 또한, LLM이 인간 평가자의 선호도를 모방할 수 있는 능력을 평가하여, 관련된 Best Practices를 확립하는 데 기여하고자 합니다.

- **Technical Details**: Lay Summarisation의 목표는 기술 기사의 핵심 개념과 발견을 비전문가에게 효과적으로 전달하는 것입니다. 이를 위해 우리는 eLife 저널의 다단계 프로세스를 구현하여, 주요 질문을 통해 요약 생성을 시뮬레이션 합니다. 연구 방법론으로는 두 단계 질문-답변(QA) 기반 프롬프트를 제안하며, 이는 Biomedicine 및 Natural Language Processing(NLP) 두 도메인에서 평가됩니다.

- **Performance Highlights**: 성능 평가에서는 eLife 데이터셋을 활용하여 제안한 두 단계 방법이 일반적인 한 단계 방법에 비해 더 나은 결과를 보여주었습니다. 다양한 LLM을 사용한 실험 결과, 자동 평가 매트릭스에서 유의미한 차이를 보였으며, 인간 평가자와 LLM 평가자 모두 제안한 방법에 대한 선호도를 나타냈습니다. 특히, LLM의 크기가 증가할수록 사용자가 더 유익하다고 느끼는 요약이 생성되는 경향이 있음을 확인했습니다.



### ParaRev: Building a dataset for Scientific Paragraph Revision annotated with revision instruction (https://arxiv.org/abs/2501.05222)
Comments:
          Accepted at the WRAICogs 1 workoshop (co-located with Coling 2025)

- **What's New**: 이 논문에서는 과거의 문장 수준 수정 접근 방식에서 단락 수준으로 수정의 범위를 확장하는 것의 중요성을 강조합니다. ParaRev라는 새로운 데이터셋을 도입하여, 차세대 과학적 텍스트 수정에서 보다 의미 있는 변화를 실현할 수 있도록 합니다. 이 새로운 접근 방식은 인간 작성자가 수정 작업을 수행하는 방식과 더 밀접하게 일치하며, 기존의 일반 지침보다 개인화된 지침을 통해 보다 섬세한 제어를 가능하게 합니다.

- **Technical Details**: ParaRev 데이터셋은 저자에 의해 수정된 단락으로 구성되며, 각 단락에는 수정 의도를 나타내는 라벨과 지침이 수동으로 주석 처리되어 있습니다. 연구 결과, 세부적인 지침을 사용하는 것이 모델의 종류나 측정 방식에 상관없이 일반적인 접근 방식에 비해 자동 수정의 품질을 눈에 띄게 향상시킵니다. 논문에서는 수정 의도를 보다 명확히 하여 기존의 문장 수준 데이터셋의 한계를 극복하고자 합니다.

- **Performance Highlights**: 실험 결과, 제안된 ParaRev 데이터셋을 사용한 모델들이 수정 품질에서 현저한 개선을 보였으며, 이는 연구자들의 논문 작성 및 수정 과정에 유용한 도구가 될 것입니다. 데이터셋의 통계에 따르면, 총 48,203개의 단락 쌍이 있지만, 641개는 수동으로 주석이 달린 평가 subset입니다. 이 데이터셋은 향후 연구 및 개발에 중요한 기초 자료로 활용될 것입니다.



### GLaM-Sign: Greek Language Multimodal Lip Reading with Integrated Sign Language Accessibility (https://arxiv.org/abs/2501.05213)
Comments:
          9 pages, 4 figures

- **What's New**: 그리스어 다중모드 입술 읽기와 통합 수화 접근성을 위한 GLaM-Sign은 청각 장애인과 난청인을 지원하기 위해 개발된 혁신적인 자료입니다. FEELIT 프로젝트를 기반으로 하여, 고해상도 오디오, 비디오, 텍스트 전사 및 그리스 수화 번역을 통합하고 있습니다. 이 시스템은 그리스 관광 부문에서의 포용성을 증가시키는 데 주력하고 있지만, 교육, 의료 및 공공 서비스로의 적용 가능성도 있습니다.

- **Technical Details**: GLaM-Sign은 실시간 수화 번역(real-time sign language translation) 및 자막 동기화(enhanced subtitle synchronization)를 위한 여러 기술적 요소를 포함하고 있습니다. 고해상도 미디어와 텍스트 및 수화 정보의 융합을 통해 데이터 세트(data set)는 다중 모드(multimodal) 접근 방식의 효용성을 보여줍니다. 향후 발전은 단어 수준의 정확성과 다양한 언어로의 확장을 목표로 하며, 선진 AI 방법론(advanced AI methodologies) 및 다양한 이해관계자와의 협력을 통해 이루어질 것입니다.

- **Performance Highlights**: 이 데이터 세트는 의사소통 격차를 해소하고 혁신을 촉진하는 다중 모드 자원의 변혁적 가능성을 보여줍니다. GLaM-Sign은 다양한 영역에서 높은 접근성을 제공하여 포용성을 촉진하는 데 기여하고 있습니다. 또한 윤리적 AI 및 포괄적인 기술을 위한 기준을 설정하며, 앞으로의 연구와 개발의 표준이 될 것으로 기대됩니다.



### Biomedical Relation Extraction via Adaptive Document-Relation Cross-Mapping and Concept Unique Identifier (https://arxiv.org/abs/2501.05155)
Comments:
          13 pages, 6 figures

- **What's New**: 이 논문에서는 Document-Level Biomedical Relation Extraction (Bio-RE) 개선을 목표로 하는 새로운 프레임워크를 제안합니다. LLM(대규모 언어 모델) Adaptive Document-Relation Cross-Mapping (ADRCM) Fine-Tuning과 Concept Unique Identifier (CUI) Retrieval-Augmented Generation (RAG) 기술을 활용합니다. 특히, 데이터 부족 문제를 해결하기 위해 Iteration-of-REsummary (IoRs) 프롬프트를 도입하여 ChatGPT를 통해 특정 엔터티 관계에 초점을 맞춘 합성 데이터를 생성하는 방법을 제시합니다.

- **Technical Details**: ADRCM Fine-Tuning은 서로 다른 문서와 관계 간의 맵핑을 설정하여 모델의 문맥 이해도와 교차 문장 추론 능력을 향상시킵니다. CUI RAG는 엔터티에 대한 인덱스로 CUI를 활용하여 정보 검색 범위를 좁히고 관련 문맥을 풍부하게 만듭니다. 또한 IoRs 프롬프트는 엔터티 관계에 집중하여 데이터의 일반화 및 정확성을 향상시키는 데 도움을 줍니다.

- **Performance Highlights**: 제안된 방법은 GDA, CDR, BioRED의 세 개 Bio-RE 데이터셋에서 실험하여 기존의 연구들과 비교했을 때 최첨단 성능을 달성했습니다. 특히, 문서 수준의 Bio-RE 작업에서 모델이 표기된 관계를 효과적으로 추론할 수 있도록 하는 것이 주요 성과로 드러났습니다. 이로 인해 향후 Bio-RE 분야의 발전에 큰 기여를 할 것으로 기대됩니다.



### Centurio: On Drivers of Multilingual Ability of Large Vision-Language Mod (https://arxiv.org/abs/2501.05122)
- **What's New**: 이번 연구는 다국어를 위한 거대 비전-언어 모델(LVLM)을 훈련하는 데 있어 최적의 언어 분포를 조사하여, 이를 통해 비영어 사용자의 접근성을 높이는 방법을 모색합니다. 기존 모델들이 영어 데이터에 주로 의존해 다국어 성능이 제한되는 문제를 해결하기 위해, 43개 언어에서 다국어 데이터 분포의 효과를 실험하고 분석하였습니다. 연구 결과, 100개 언어를 동시에 사용하더라도 영어 성능이 크게 저하되지 않음을 보여 주며, 이는 다국어 성능 향상의 방향을 제시합니다.

- **Technical Details**: 연구는 LVLM 훈련 믹스의 언어 분포를 체계적으로 조사하며, 13개의 비전-언어 작업과 43개 언어를 대상으로 다양한 실험을 수행했습니다. 주된 연구 질문은 훈련 언어의 최적 수, 각 언어에 대한 데이터의 최적 분포, 이미지 내 비전-언어 이해를 향상시키기 위한 방법 등에 중점을 두고 있습니다. 이를 위해 Synthetic Multilingual Plot Question Answering(SMPQA)라는 새로운 데이터셋을 도입하여 다국어 OCR 능력을 평가합니다.

- **Performance Highlights**: Centurio라는 100개 언어를 지원하는 LVLM을 훈련하여 총 14개 작업에서 최첨단 성능을 달성하였습니다. 이 모델은 기존 인기 있는 다국어 LVLM 모델들과 비교하여 영어 및 자원 풍부 언어에서의 성능은 동등하거나 더 나은 결과를 보이며, 저자원 언어에서는 더욱 우수한 성능을 보여줍니다. 본 연구의 결과는 다국어 비전-언어 모델의 설계 및 훈련에 중요한 영향을 미칠 것으로 기대됩니다.



### SWE-Fixer: Training Open-Source LLMs for Effective and Efficient GitHub Issue Resolution (https://arxiv.org/abs/2501.05040)
Comments:
          Our code, data, and model will be released at this https URL

- **What's New**: 본 논문에서는 SWE-Fixer라는 새로운 오픈 소스 LLM을 제안하여 GitHub에서 소프트웨어 엔지니어링 문제를 효과적으로 해결하는 방안을 모색합니다. SWE-Fixer는 코드 파일 검색 및 코드 수정 모듈을 포함하여 두 가지 주요 모듈로 구성되어 있습니다. 기존의 비공식 LLM에 의존하는 시스템의 한계를 인식하고, 더 높은 재현성과 접근성을 제공하도록 설계되었습니다.

- **Technical Details**: SWE-Fixer는 우선 BM25 알고리즘을 기반으로 한 초기 파일 검색 후, 고급 LLM을 이용해 결함이 있는 파일을 식별합니다. 이후, 수정 모듈에서는 체인 오브 씽킹(Chain-of-Thought, CoT)을 활용하여 해당 파일에 대한 코드 패치를 생성합니다. 이 과정은 기존 모델들과 비교해 단순화한 프로세스로, 비용 효율성을 제공하면서도 성능을 유지합니다.

- **Performance Highlights**: SWE-Fixer는 SWE-Bench Lite와 Verified 벤치마크에서 최첨단 성능을 달성하여 각각 23.3% 및 30.2%의 점수를 기록했습니다. 오픈 소스 모델 기반 접근법에서 Best@1 성능을 가장 높게 기록하며, 비공식 모델 기반 시스템과 비교할 때도 여러 낫다고 평가됩니다. 이는 우리 접근 방식의 효율성을 강조합니다.



### Enhancing Human-Like Responses in Large Language Models (https://arxiv.org/abs/2501.05032)
- **What's New**: 이 논문은 대형 언어 모델(LLM)의 자연어 이해, 대화 일관성, 감정 지능을 향상시키는 기술에 초점을 맞추고 있습니다. 우리는 다양한 데이터셋을 통한 미세 조정, 심리학 원칙의 적용, 인간의 추론 패턴을 모방하는 모델 설계 방식을 통합하여 AI 시스템을 더욱 인간처럼 만듭니다. 이러한 기술들은 사용자 상호작용을 개선하고, 다양한 분야에 걸쳐 AI 응용 프로그램의 새로운 가능성을 열어줍니다.

- **Technical Details**: 우리는 Llama 3 70B 및 405B 모델을 활용하여 인간과 같은 대화를 생성하기 위해 합성 데이터셋을 생성했습니다. 이 데이터셋은 질문 생성에 Llama 3 405B를, 답변 생성에 Llama 3 70B를 사용하여 작성되었습니다. 각각의 시스템 프롬프트는 자연스러운 인간의 대화를 모방하거나 더 형식적이고 비인격적인 응답을 생성하도록 설계되었습니다.

- **Performance Highlights**: 모델 훈련에는 LoRA(로우-랭크 어댑테이션)와 DPO(다이렉트 선호 최적화) 기법을 사용했습니다. 이러한 접근법은 모델의 일반 지식을 유지하면서 특정 작업에 적응하도록 도와줍니다. 훈련 결과, 인간과 유사한 대화를 가능하게 하여 AI의 대화 플루언시와 사용자 참여도를 크게 향상시켰습니다.



### TreeKV: Smooth Key-Value Cache Compression with Tree Structures (https://arxiv.org/abs/2501.04987)
- **What's New**: 이 논문은 TreeKV라는 새로운 고안의 캐시 압축 방안을 제시합니다. 기존의 방법들은 위치 기반 혹은 중요도 점수에 의존하여 토큰을 제거했으나, 이러한 접근은 중요한 정보를 놓칠 수 있습니다. Wavelet 분석을 통해, 이 연구는 시퀀스의 끝으로 갈수록 토큰의 기여도가 증가하고 있음을 발견했습니다. 이는 토큰 간의 다양성과 복잡성이 증가하는 부드러운 전환을 나타냅니다.

- **Technical Details**: TreeKV는 훈련이 필요 없는 트리 구조를 사용하여 캐시 압축을 부드럽게 수행합니다. 이는 고정된 캐시 크기를 유지하면서도 긴 텍스트 시나리오에서도 높은 품질의 출력을 제공합니다. 이 방법은 생성(generation)과 Prefilling 두 단계에 모두 적용 가능하여 LLM이 긴 시퀀스를 처리하는 데 유리한 점을 가지고 있습니다. 다양한 실험 결과는 TreeKV의 효과성을 입증하며, 특히 퍼플렉시티(perplexity)에서 낮은 값을 기록했습니다.

- **Performance Highlights**: TreeKV는 PG19 및 OpenWebText2에서의 언어 모델링 과제에서 모든 기준 모델들을 초과하는 성능을 보여줍니다. 또한, Longbench 벤치마크에서 최적 효율 상태에서 단지 6%의 예산만으로도 최고의 성능을 달성했습니다. 이 방법에 의해 LLM은 16kx 이상의 시퀀스로 일반화가 가능하며, 출력 품질이 향상되었습니다.



### SensorQA: A Question Answering Benchmark for Daily-Life Monitoring (https://arxiv.org/abs/2501.04974)
- **What's New**: 이 논문에서는 SensorQA라는 새로운 QA 데이터셋을 소개합니다. 이는 인간이 만든 최초의 QA 데이터셋으로, 장기적인 타임시리즈 센서 데이터를 기반으로 합니다. 데이터셋은 5.6K개의 다양한 질문과 정확한 답변 쌍을 포함하여 사용자의 실제 관심사를 반영합니다.

- **Technical Details**: SensorQA는 60명의 사용자로부터 3개월 동안 수집된 일상 활동 모니터링을 위한 센서 데이터에 초점을 맞춥니다. 질문 생성은 Amazon Mechanical Turk 플랫폼에서 수행된 활동 일정 그래프를 기반으로 하고, 14개의 다양한 활동 레이블의 서브셋을 사용하여 질문의 다양성을 촉진합니다. 이로 인해 하루에서 여러 주에 이르는 센서 시간 스케일을 포괄하는 5.6K개의 QA 쌍이 포함됩니다.

- **Performance Highlights**: SensorQA를 기반으로 한 최신 AI 모델의 벤치마크 평가 결과, 현재 모델과 최적의 QA 성능 및 효율성 사이에는 큰 격차가 존재했습니다. 이는 새로운 기여와 개선의 필요성을 강조합니다. 또한 SensorQA 데이터셋과 코드는 오픈소스 형태로 제공되어, 관련 연구자들이 이 분야에 기여할 수 있도록 장려하고 있습니다.



### VoxEval: Benchmarking the Knowledge Understanding Capabilities of End-to-End Spoken Language Models (https://arxiv.org/abs/2501.04962)
- **What's New**: VoxEval이라는 새로운 음성 질문-답변 벤치마크가 소개되었습니다. 이 벤치마크는 SLMs의 지식 이해 능력을 평가하기 위해 순수 음성 상호작용을 통해 설계되었습니다. VoxEval은 질문과 답변 모두를 음성 포맷으로 유지하며, 다양한 오디오 조건에서 모델의 견고성을 평가합니다.

- **Technical Details**: VoxEval은 오디오 녹음으로 질문과 지침을 제공하여 성능을 평가합니다. SLM의 견고성을 검사하기 위해 동일한 질문에 대한 다양한 입력 조건을 포함하였으며, 특히 수학 평가와 같은 어려운 주제에 대한 평가 방법론을 혁신적으로 제시합니다. VoxEval은 MMLU 데이터셋을 기반으로 하여 질문이 음성으로 변환됩니다.

- **Performance Highlights**: 최근 SLM에 대한 VoxEval 평가 결과, 현재 모델들이 감당해야 하는 중대한 성능 제한이 발견되었습니다. 기존 모델은 다양한 오디오 조건 변화에 특히 취약성을 보였으며, 이는 향후 연구에서 견고성을 향상시키는 것이 필요함을 나타냅니다.



### Demystifying Domain-adaptive Post-training for Financial LLMs (https://arxiv.org/abs/2501.04961)
- **What's New**: 본 논문에서는 FINDAP라는 새로운 프레임워크를 제안하며, 이는 금융 도메인에 특화된 대규모 언어 모델(LLM)의 도메인 적응형 후 훈련(domain-adaptive post-training)을 체계적으로 조사합니다. 본 연구는 핵심 역량을 정의하고 이를 바탕으로 평가 프레임워크를 설계함으로써, 최적의 훈련 방법을 제안하는 것을 목표로 합니다. Llama-Fin이라는 혁신적인 모델을 통해 금융 업무 전반에서 최첨단 성능을 달성합니다.

- **Technical Details**: FINDAP의 핵심은 도메인 특화 개념, 도메인 특정 작업, 추론 능력 및 지시 준수(instruction-following) 능력 등을 갖춘 LLM의 필요한 역량을 식별하는 것입니다. 이러한 역량 기반으로 평가 프레임워크를 개발하여 성과 목표를 명확히 하고, 여러 작업에서 모델의 개선을 이끌어내는 것이 주된 내용입니다. 따라서, 훈련 방법론은 연속 사전 훈련(Continual Pretraining, CPT) 및 지시 조정(Instruction Tuning, IT) 단계로 나뉘며, 이후 선호 데이터(preference data)를 사용한 새로운 훈련 레시피가 적용됩니다.

- **Performance Highlights**: Llama-Fin 모델은 70B 스케일의 대형 모델들 및 상용 모델인 GPT-4o를 포함한 모든 기준 모델들보다 우수한 성과를 보여주었습니다. 특히, 본 연구에서는 훈련 데이터와 유사한 새로운 작업에서는 물론, 훈련 중 경험하지 못한 새로운 작업에서도 경쟁력을 유지하며 높은 성능을 기록했습니다. 이러한 결과는 도메인 적응형 후 훈련의 중요성과 효과적인 훈련 레시피의 필요성을 강조합니다.



### Step-by-Step Mastery: Enhancing Soft Constraint Following Ability of Large Language Models (https://arxiv.org/abs/2501.04945)
- **What's New**: 이번 연구에서는 LLM(대규모 언어 모델)이 소프트 제약 조건을 따르는 능력을 개선하기 위한 효율적인 데이터 생성 파이프라인을 설계했습니다. 기존의 연구들은 제약 조건을 따르는 LLM의 능력을 평가하는 데 중점을 두었으나, 본 연구는 이를 향상하는 방향으로 진행되었습니다. 또한, 커리큘럼 학습(curriculum learning) 기반의 새로운 훈련 패러다임을 도입하여 LLM이 복잡한 제약 조건을 효과적으로 따를 수 있도록 합니다.

- **Technical Details**: 제안된 방법론은 두 가지 주요 단계로 구성됩니다: 점진적인 제약 조건 추가와 Judger 재정렬입니다. 이 과정에서 각 제약 조건을 하나씩 추가하여 LLM이 각 제약 조건을 점진적으로 따를 수 있도록 훈련합니다. Judger는 모델의 출력을 제약 조건 준수 정도에 따라 재정렬하여 최종적인 고품질 출력을 생성합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 LLM의 소프트 제약 조건 따르기 능력을 크게 향상시키며, 훈련 데이터의 품질이 모델의 출력 품질에 미치는 영향이 크다는 것을 보여주었습니다. GPT-4 모델을 포함한 다양한 LLM에서 실행된 Benchmark 평가에서 이 새로운 접근법의 효과가 증명되었습니다. 다양한 조건을 조합한 프레임워크를 통해, 소프트 제약 조건을 처리하는 데 있어 LLM의 전반적인 성능이 개선되었습니다.



### Investigating Numerical Translation with Large Language Models (https://arxiv.org/abs/2501.04927)
Comments:
          Accepted by ICASSP 2025

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)의 숫자 번역 신뢰성을 평가하는 데 중점을 두었습니다. 주변 환경의 변동성을 반영하기 위해 실 비즈니스 데이터를 기반으로 한 중국어-영어 번역 데이터 세트를 구축했습니다. 실험 결과에 따르면, 대다수의 공개 소스 LLM은 큰 단위를 포함한 숫자 번역에서 높은 오류율을 보였습니다.

- **Technical Details**: 이 연구는 중국어와 영어 간의 숫자 번역을 평가하기 위해 10가지 숫자 번역 유형을 포함한 데이터세트를 제안합니다. 포함된 유형으로는 큰 단위, 범위, 소수, 비율 등이며 이는 각각의 특정한 변환 규칙을 따릅니다. 특히, 모델들은 '백만(million)'이나 '십억(billion)'과 같은 큰 단위 번역에서 정확도를 크게 저하했습니다.

- **Performance Highlights**: 현대 LLMs의 성능이 다양하게 나타나는 것을 보여주며, 통번역 LLM(T-LLM)과 일반 LLM(G-LLM)은 각각의 강점을 발휘합니다. 그러나, 모든 유형에서 우수한 성능을 발휘하는 단일 모델은 없었습니다. GLM-4-9b-chat이 특정 기준에서 가장 높은 성과를 보였지만, 여전히 모든 범주에서 정확한 결과를 제공하지 못했습니다.



### JELLY: Joint Emotion Recognition and Context Reasoning with LLMs for Conversational Speech Synthesis (https://arxiv.org/abs/2501.04904)
Comments:
          Accepted by ICASSP 2025

- **What's New**: 본 논문에서는 감정 인식(emotion recognition)과 맥락 추론(context reasoning)을 통합하여 대화에서 적절한 음성을 생성하는 새로운 CSS 프레임워크 JELLY를 소개합니다. 이 프레임워크는 여러 개의 부분 LoRA(PLoRA) 모듈을 이용해 대형 언어 모델(LLM)을 미세 조정(fine-tuning)하여 음성의 감정 상황을 추론하도록 설계되었습니다. 특히, JELLY의 EQ-former 인코더는 발화의 감정을 텍스트와 정렬하여 음성에서 감정 상태를 인식할 수 있도록 합니다.

- **Technical Details**: JELLY는 EQ-former라는 감정 인식 모듈을 통해 각 발화의 감정 상태를 이해합니다. 이 모듈은 감정 특징을 추출하기 위한 TLTR(시간 및 층별 트랜스포머)과 감정 정렬을 위한 Q-former로 구성되어 있습니다. Whisper 인코더를 활용하여 다양한 음성 정보를 추출하며, PLoRA 접근 방식으로 LLM의 매개변수를 조정하여 감정과 텍스트 간의 간극을 해소합니다.

- **Performance Highlights**: 실험 결과에 따르면 JELLY는 감정적 맥락 추론(emotional context reasoning)과 대화 맥락에 기반한 음성 합성(speech synthesis)에서 우수한 성능을 보입니다. JELLY는 기존 CSS 모델이 필요로 하는 감정 레이블이나 대화 이력의 전사 없이 음성만으로 감정 맥락을 추론하는 데 성공하였습니다. 이는 제한된 감정 대화 데이터 세트를 효과적으로 완화함으로써 실세계 시나리오에 적합한 성능을 발휘합니다.



### SUGAR: Leveraging Contextual Confidence for Smarter Retrieva (https://arxiv.org/abs/2501.04899)
Comments:
          ICASSP2025

- **What's New**: 본 논문에서는 Semantic Uncertainty Guided Adaptive Retrieval (SUGAR)라는 새로운 접근법을 제안합니다. 이는 LLM의 파라메트릭 지식만으로는 정확한 응답을 생성하는 데 한계가 있어 외부 지식의 적절한 활용을 필요로 하는 문제에서 출발했습니다. SUGAR는 컨텍스트 기반 엔트로피를 활용하여 모델의 불확실성을 평가하고, 이 기반으로 검색 여부와 검색 방식을 결정합니다.

- **Technical Details**: SUGAR는 언어 모델의 응답 생성에 대한 불확실성을 측정하기 위해 의미적 엔트로피를 사용합니다. 높은 엔트로피를 보이는 경우에는 외부 지식 D를 함께 활용하여 응답을 생성하게 됩니다. 이렇게 함으로써 주어진 질문에 대한 더 적절한 지원 컨텍스트를 제공하고, 단일 단계 또는 다단계 검색을 동적으로 결정합니다.

- **Performance Highlights**: 실험 결과, SUGAR를 사용한 선택적 검색이 다양한 질문 응답 과제에서 성과를 향상시키는 것으로 나타났습니다. SUGAR는 정보 검색에서 효과적인 자원의 활용을 지원하며, 불확실성 평가를 통해 필요할 때만 검색을 수행하여 효율성을 최적화합니다. 또한, SUGAR는 추가 학습이나 미세 조정 없이도 안정적인 성과를 발휘할 수 있습니다.



### Leveraging Log Probabilities in Language Models to Forecast Future Events (https://arxiv.org/abs/2501.04880)
Comments:
          5 pages, 4 figures

- **What's New**: 본 논문에서는 데이터 기반의 의사 결정에서 미래 사건을 예측하는 것을 목표로 하며, Large Language Models (LLMs)를 활용한 혁신적인 방법을 제안합니다. 기존 연구를 바탕으로 15개의 다양한 주제에 대한 예측을 생성하고, 그 확률을 log probabilities를 기반으로 한 다단계 접근 방식으로 추정하여 Brier score 0.186을 달성했습니다. 이는 무작위 확률에 비해 +26% 개선된 성과입니다.

- **Technical Details**: 이 시스템은 예상하는 이벤트의 제목, 설명, 시간대를 생성하는 Forecast Generator, 예측의 확률과 관련 트렌드를 출력하는 Probability Estimator, 주어진 시간 내에 사건이 발생했는지 자동으로 평가하는 Fact Checker로 구성됩니다. 특히 Probability Estimator는 LLM의 모든 가능한 추측을 고려하여, 확률 P와 불확실성 U를 평가하는 데 혁신적인 접근 방식을 사용합니다. 설정에 따라 예측의 일관성을 개선하고, 상호 배타적인 사건에 대해 총 확률이 100%를 초과하지 않도록 보장합니다.

- **Performance Highlights**: 논문에서 제안한 방법은 총 240일의 간격을 두고 예측된 사건과 사실 확인을 진행하며, N=150의 예측 데이터셋으로 평가되었습니다. 우리의 시스템은 잘 알려진 AI 시스템에 비해 +19% 이상의 개선률을 보이며, 이는 LLM 기반 예측의 가능성을 보여주는 중요한 지표입니다. 이러한 결과는 LLM이 단순한 데이터 추정이 아닌, 더 깊이 있는 정보 분석 및 예측 능력을 갖추고 있음을 증명합니다.



### Real-Time Textless Dialogue Generation (https://arxiv.org/abs/2501.04877)
- **What's New**: 최근 대형 언어 모델(LLMs)의 발전은 텍스트 기반 대화 시스템에서 중요한 진전을 이루었습니다. 그러나 음성 대화 시스템은 여전히 자연스러움에서 뒤처지고 있으며, 이는 전통적인 캐스케이드 설계에 의존하기 때문입니다. 본 논문은 이러한 문제를 해결하기 위한 실시간 텍스트 없는 음성 대화 생성 모델(RTTL-DG)을 제안합니다.

- **Technical Details**: RTTL-DG는 전통적인 음성 인식과 응답 생성을 통합하여 직접적인 음성 처리를 통해 대화를 생성합니다. 이 모델은 대화 관리자(Dialogue Manager)와 응답 생성기(Response Generator)로 구성되며, 연속적인 음성 입력에서 정기적으로 다음 행동을 결정합니다. 생성된 응답은 기계적이지 않고 자연스러운 표현을 포함하여 빠른 대응과 원활한 턴 테이킹을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, RTTL-DG 모델은 자연스러움, 반응 속도 및 유동성에서 우수한 성능을 나타내며, 사용자와의 상호작용에서 보다 자연스러운 대화 경험을 제공합니다. 전통적인 캐스케이드 모델에 비해 의미적 일관성에서는 약간의 성능 저하가 있었지만, 사용자 친화적이고 매력적인 대화 시스템 개발에 중요합니다.



### Advancing Retrieval-Augmented Generation for Persian: Development of Language Models, Comprehensive Benchmarks, and Best Practices for Optimization (https://arxiv.org/abs/2501.04858)
- **What's New**: 이 논문은 낮은 자원 언어에서 Retrieval-Augmented Generation (RAG) 시스템을 구축하는 데 직면하는 특정 장애물에 대해 분석합니다. 특히, 페르시아어의 복잡한 형태론(morphology)과 다재다능한 구문(syntax)에 초점을 맞추고 있습니다. MatinaRoberta(마스크드 언어 모델)와 MatinaSRoberta(세밀하게 조정된 Sentence-BERT)라는 페르시아어 특화 모델을 소개하면서 검색 및 생성 정확도를 향상시키고자 합니다.

- **Technical Details**: 이 연구는 731억 개 페르시아어 토큰으로 구성된 다양한 코퍼스에서 훈련된 모델을 평가하기 위해 일반 지식(PQuad), 과학적으로 전문화된 텍스트 및 조직 보고서의 세 가지 데이터셋을 사용했습니다. 방법론은 광범위한 사전 훈련(pretraining), 맞춤형 손실 함수에 대한 세밀한 조정(fine-tuning), 그리고 전통적인 메트릭(metrics)과 RAG 평가 프레임워크를 사용하여 체계적인 평가를 포함합니다.

- **Performance Highlights**: MatinaSRoberta는 이전의 임베딩(embeddings)보다 뛰어난 성과를 보이며, 다양한 데이터셋에서 뛰어난 맥락적 관련성(contextual relevance)과 검색 정확도를 달성했습니다. Llama-3.1(70B)와 같은 대형 모델은 생성 정확도가 가장 높았으며, 작은 모델들은 도메인 특화(domain-specific) 및 형식적인 맥락에서 어려움을 겪었습니다. 이 연구의 결과는 페르시아어에서 맞춤형 임베딩 및 검색-생성 설정을 통해 RAG 시스템을 발전시킬 수 있는 잠재력을 강조하고, 낮은 자원 언어에서 검색 엔진 및 법률 문서 분석과 같은 NLP 응용 프로그램의 향상을 촉진할 수 있음을 보여줍니다.



### Building Foundations for Natural Language Processing of Historical Turkish: Resources and Models (https://arxiv.org/abs/2501.04828)
- **What's New**: 이 논문은 역사적 터키어의 자연어 처리(NLP)에 대한 기초 자원과 모델을 처음으로 소개합니다. HisTR이라는 이름 실체 인식(NER) 데이터셋과 OTA-BOUN이라는 유니버설 의존 구문 분석 나무 세트를 발표하며, 이 데이터셋을 이용한 변환기 기반 모델을 역사적 터키어 텍스트에 대한 다양한 NLP 작업에 적용합니다. 또한 다양한 역사적 시대를 아우르는 깨끗한 텍스트 집합인 오스만 텍스트 코퍼스(OTC)를 도입하여 자동화된 정보 추출의 수요를 충족시키고자 합니다.

- **Technical Details**: 논문에서는 17세기부터 19세기까지의 812개의 수동 주석 문장으로 구성된 HisTR 데이터셋, 514개의 문장에서 파생된 수동 주석을 포함하는 OTA-BOUN 종속성 나무 세트, 15세기부터 20세기까지의 다양한 장르를 포함하는 오스만 텍스트 코퍼스(OTC)를 구축했습니다. 이들 자원은 의존성 구문 분석, 품사 태깅, 이름 실체 인식 작업을 위한 변환기 기반 모델의 훈련에 사용되었으며, 모든 자원과 모델은 공개적으로 사용할 수 있습니다.

- **Performance Highlights**: 제공된 데이터셋과 모델을 바탕으로 역사적 터키어의 계산적 분석에서 유의미한 개선이 이루어졌습니다. 연구 결과, 역사적 언어 구조를 이해하는 작업에서 유망한 결과가 도출되었으며, 그러나 여전히 도메인 적응과 시간에 따른 언어 변variation에 대한 도전 과제가 남아있음을 보여줍니다. 이러한 연구는 역사적 터키어 NLP의 기준점 역할을 할 것으로 기대됩니다.



### Unifying the Extremes: Developing a Unified Model for Detecting and Predicting Extremist Traits and Radicalization (https://arxiv.org/abs/2501.04820)
Comments:
          17 pages, 7 figures, 4 tables

- **What's New**: 이 연구에서는 온라인 커뮤니티 포럼에서 극단주의 담론을 추출하고 분석하는 새로운 방법을 제안합니다. 특히, 극단주의적 특성의 언어적 행동 서명을 중심으로 한 프레임워크를 개발하여 사용자 및 커뮤니티 수준에서 극단주의를 정량화합니다. 이 연구는 11개의 독립된 요인, 즉 'Extremist Eleven'을 제시하며, 이는 극단주의의 일반화된 심리사회적 모델로 기능합니다.

- **Technical Details**: 이 프레임워크는 자연어 처리(NLP) 기술과 극단주의에 대한 심리사회적 이론을 결합한 비지도 방법을 사용합니다. 연구팀은 사용할 수 있는 텍스트 데이터에서 수동 라벨링 없이도 극단주의의 특성을 추출합니다. 이러한 11개의 차원은 극단주의 이데올로지와 관련된 주요 특성을 광범위하게 특징짓는 데 중점을 두고 있으며, 이는 개인의 심리 및 정서적 요인에 대한 더 깊은 이해를 제공합니다.

- **Performance Highlights**: 이 연구에서는 incel 커뮤니티 회원의 사용자 기록을 분석하여 실제 가입하기 10개월 전부터 사용자의 극단주의 경향을 정확히 예측할 수 있음을 보여줍니다. 이 프레임워크는 약 0.6 이상의 AUC를 기록하며, 사건 발생 3-4개월 전에는 AUC가 약 0.9로 증가합니다. 이러한 예측 가능성은 온라인 극단주의의 조기 탐지 및 개입을 위한 실용적 응용 가능성을 강조합니다.



### Cued Speech Generation Leveraging a Pre-trained Audiovisual Text-to-Speech Mod (https://arxiv.org/abs/2501.04799)
- **What's New**: 이 논문에서는 청각 장애인을 위한 시각 커뮤니케이션 시스템인 Cued Speech (CS)의 자동 생성 방법(Automatic Cued-Speech Generation, ACSG)을 제안합니다. 사전 훈련된 오디오-비주얼 오토회귀 텍스트-음성 모델인 AVTacotron2를 활용하여 텍스트 입력에서 CS 손과 입 움직임을 추론하도록 모델을 재프로그래밍하였습니다. 두 개의 공개 데이터셋을 활용하여 실험을 진행하였으며, 결과적으로 약 77%의 음소 레벨에서의 디코딩 정확도를 달성하였습니다.

- **Technical Details**: ACSG는 텍스트에서 큐드 스피치를 자동으로 생성하는 과정으로, 이 과정은 사진 실사 비디오 또는 애니메이션 아바타 형태로 이루어질 수 있습니다. 본 연구는 AV-TTS의 최신 발전, 특히 신경 인코더-디코더 아키텍처를 기반으로 하여 ACSG의 문제를 해결하는데 중점을 두었습니다. 또한, ACSG의 특징이 점차 저자원 문제(low-resource problem)임을 지적하며, 기존의 자료들은 제한적이고 생성의 어려움이 존재한다고 언급합니다.

- **Performance Highlights**: 제안된 모델은 낮은 리소스 문제를 해결하기 위해, 훈련된 데이터셋을 더욱 확장하고, AV-TTS 모델의 전이 학습 전략을 탐구하였습니다. 새로운 데이터셋은 고해상도의 비디오를 제공하고, 일정한 프레임 속도를 보장하여 지속적인 CS 생성을 위한 기반을 마련하였습니다. 최종적으로, 본 연구에서 제안한 방법은 Cued Speech 데이터 생성에서 77%라는 상당히 높은 정확도를 기록하며 그 유효성을 증명하였습니다.



### ReFocus: Visual Editing as a Chain of Thought for Structured Image Understanding (https://arxiv.org/abs/2501.05452)
Comments:
          Project link: this https URL

- **What's New**: 이 논문에서는 ReFocus라는 새로운 프레임워크를 소개하여 멀티모달 대형 언어 모델(LLMs)이 이미지에서 직접적으로 "visual thoughts"를 생성하고 조작할 수 있는 능력을 갖추도록 합니다. 이를 통해 다양한 구조화된 이미지 과제, 특히 표와 차트에 대한 시각적 이해 능력을 크게 향상시킬 수 있습니다. ReFocus는 모델이 Python 코드를 생성하여 시각적 편집을 수행하도록 하여 멀티홉 시각적 추론을 가능하게 합니다.

- **Technical Details**: ReFocus는 입력 이미지를 수정하는 시각 편집 작업을 통해 멀티모달 LLMs가 선택적 주의를 향상시킬 수 있도록 돕습니다. 모델은 코드 실행을 통해 입력 이미지를 동적으로 수정하여, 불필요한 정보를 제거하거나 중요한 정보를 강조할 수 있습니다. 이러한 과정은 시각적 사고의 연속적인 단계로 진행되며, 표 문제에 대한 평균 성능 향상률은 11.0%, 차트 문제에 대한 성능 향상률은 6.8%에 달합니다.

- **Performance Highlights**: ReFocus는 기준 모델인 GPT-4o와 비교했을 때, 표 및 차트 문제에서 일관된 성능 향상을 나타냅니다. 특히, 모델이 REFOCUS 데이터를 활용하도록 미세 조정할 경우, 전통적인 질문-답변(QA) 데이터보다 더 나은 성능을 보이는 것으로 나타났습니다. 이러한 실험 결과는 ReFocus의 효과성과 향후 시각 언어 모델들에 대한 개선 가능성을 제시합니다.



### Search-o1: Agentic Search-Enhanced Large Reasoning Models (https://arxiv.org/abs/2501.05366)
- **What's New**: 이 논문은 기존의 대규모 추론 모델(LRM)에서 발생하는 지식 부족 문제를 해결하기 위해 	extbf{Search-o1}이라는 새로운 프레임워크를 소개합니다. Search-o1은 에이전트 기반 검색 증강 생성(RAG) 메커니즘과 문서 내 추 reasoning(Reason-in-Documents) 모듈을 결합하여 외부 지식을 동적으로 검색하고 통합할 수 있도록 합니다. 이러한 접근 방식은 LRMs의 신뢰성과 적용 가능성을 향상시켜 더 신뢰할 수 있는 지능형 시스템을 구축하는 데 기여합니다.

- **Technical Details**: Search-o1의 디자인은 두 가지 핵심 요소에 기반하고 있습니다. 첫째, 에이전트 기반 RAG 메커니즘을 통해 LRM은 지식 부족 상태에서 적절한 외부 지식을 검색할 수 있도록 유도됩니다. 둘째, 별도의 Reason-in-Documents 모듈은 검색된 문서의 정보를 심층적으로 분석하여 원래의 추론 체계에 통합됩니다. 이러한 두 가지 측면이 상호 작용하여 LRM의 추론 과정을 강화하고 논리적 흐름을 유지할 수 있도록 합니다.

- **Performance Highlights**: 다양한 과학, 수학, 코딩 분야의 복잡한 추론 작업과 여섯 가지 공개 도메인 질문 응답(QA) 벤치마크에서 Search-o1의 강력한 성능이 입증되었습니다. Search-o1은 기존의 LRMs에 비해 신뢰성과 효율성을 크게 향상시켰으며, 이론적 및 정량적으로 검증된 결과는 LRM의 신뢰할 수 있는 추론을 위한 실질적인 지침을 제공합니다.



### CallNavi: A Study and Challenge on Function Calling Routing and Invocation in Large Language Models (https://arxiv.org/abs/2501.05255)
- **What's New**: 본 논문은 복잡한 API 호출을 요구하는 챗봇 시스템에 대한 새로운 접근 방식을 제안합니다. 이를 위해, API 기능 선택과 파라미터 생성을 평가할 수 있는 새로운 데이터셋 CallNavi를 소개하며, 다양한 복잡성 수준을 기준으로 언어 모델의 성능을 벤치마킹합니다. 또한, 일반적인 대형 언어 모델과 고급 파라미터 생성을 위한 모델을 결합하는 개선된 API 라우팅 방법을 제안하여 실제 API 기반 챗봇 시스템에 실질적인 진전을 제공합니다.

- **Technical Details**: CallNavi 데이터셋은 100개 이상의 후보 API에서의 비 필터링 API 선택, 여러 개의 연속 API 호출 실행, 및 중첩 API 상호작용 처리를 포함하여, 대형 언어 모델(LLMs)의 다양한 성능을 평가합니다. 새로운 메트릭인 stability score를 도입하여 여러 번의 실행에서 모델 예측의 일관성을 측정하며, 다중 호출 및 중첩 API 과제를 포함하여 구조적 난이도를 유지하고 있습니다.

- **Performance Highlights**: 17개의 다양한 LLM을 CallNavi에서 종합적으로 벤치마킹한 결과, 현재 모델의 강점과 한계에 대한 주요 통찰을 제공합니다. 특히, 모델이 복잡한 API 작업을 처리하는 데 있어 유의미한 개선이 이루어졌음을 확인하였으며, 이러한 결과는 API 선택 및 기능 호출에서의 추가적인 발전을 위한 기초 자료를 제공합니다.



### A Novel Approach to Scalable and Automatic Topic-Controlled Question Generation in Education (https://arxiv.org/abs/2501.05220)
Comments:
          To be published at ACM Conf. on Learning Analytics and Knowledge (LAK'25)

- **What's New**: 본 논문은 Topic-Controlled Question Generation (T-CQG)이라는 새로운 접근 방식을 제시하여 교육 목적으로 생성된 질문의 관련성과 효과성을 향상시킵니다. T-CQG는 pre-trained T5-small 모델을 미세 조정하여 특별히 만들어진 교육 맞춤형 데이터셋을 사용합니다. 연구는 사전 훈련 전략, 양자화(quantisation), 데이터 증대(data augmentation) 등이 모델 성능에 미치는 영향을 탐구하며, 문단 수준의 맥락에 맞는 의미적으로 일치하는 질문 생성을 다룹니다. 또한, 생성된 질문의 주제 관련성을 평가하기 위한 새로운 평가 방법을 소개합니다.

- **Technical Details**: T-CQG 모델은 고급 자연어 처리(NLP) 기술을 활용하여 교육 질문의 생성을 자동화합니다. 이 모델은 약 60M 파라미터를 가진 매우 작은 언어 모델(small language model)을 사용하여 교육용 질문을 생성하는 데 성공했습니다. 주제 제어(question generation)의 중요성을 강조하며, 학습 자료의 맥락과 관련된 주제를 기준으로 질문을 생성하는 방법론을 제시합니다. 이러한 접근은 교육용 질문의 품질을 개선하고 교사들이 질문을 작성하는 데 드는 수고를 덜어줄 수 있습니다.

- **Performance Highlights**: 연구 결과는 오프라인 실험 및 인간 평가(human-backed evaluations)를 통해 검증되었으며, T-CQG 모델이 고품질의 주제 중심 질문을 효과적으로 생성할 수 있음을 입증합니다. 교육 시스템에서의 적용 가능성이 높아 교사들의 업무 부담을 줄이고 개인 맞춤형 튜터링 시스템을 지원하는 데 기여할 것으로 기대됩니다. 본 연구는 교사 retention 문제와 관련된 생산성 향상을 위한 중요한 실용적 단계를 제공하여 교육 환경을 개선할 가능성이 큽니다.



### Bringing Order Amidst Chaos: On the Role of Artificial Intelligence in Secure Software Engineering (https://arxiv.org/abs/2501.05165)
Comments:
          PhD thesis

- **What's New**: 이 연구는 소프트웨어 보안 엔지니어링(SSE)에서 인공지능(AI)의 적용을 통해 예측 모델의 개선을 목표로 하고 있습니다. 특히, 결함 예측 및 취약성 탐지와 우선 순위 지정의 두 가지 특정 영역에 초점을 맞추며, AI 활용의 맥락에 따른 고유한 차이를 고려합니다. 이는 소프트웨어 시스템의 보안을 강화하고, AI 기반의 예측 모델의 정확성을 높이기 위한 중요한 기여로 평가됩니다.

- **Technical Details**: 연구 방법론으로는 경험적 전략을 활용하며, 이것은 Static Application Security Testing Tools (SASTTs)의 분석과 JIT(Just-In-Time) 결함 예측을 포함합니다. Effort-aware metrics (EAMs)를 통해 결함 예측의 정확성을 향상시키고, 데이터 세트의 체계적인 리뷰를 통해 예측 모델의 평가 및 개선을 꾀하고 있습니다. 이를 통해 SASTTs의 취약성 종류 커버리지 및 정적 분석 도구의 한계를 지적하고 있습니다.

- **Performance Highlights**: 결과적으로, EAMs의 정규화와 JIT 정보 활용을 통해 결함 예측 정확성을 극대화하는 모델이 도출되었으며, AI 기반의 위험 평가에서의 대규모 언어 모델(LLM)의 효용성을 입증하였습니다. 이러한 기여는 결함 예측과 취약성 탐지의 효과성을 높이며, 연구자와 실무자 모두에게 이점을 제공하고 있습니다. 또한, 보안 부채를 전반적으로 줄이기 위한 취약성 우선 순위 지정 프레임워크를 제시하여 조직의 보안 태세를 강화하도록 돕고 있습니다.



### Comparison of Feature Learning Methods for Metadata Extraction from PDF Scholarly Documents (https://arxiv.org/abs/2501.05082)
- **What's New**: 본 논문에서는 과학 문서의 메타데이터 추출을 위한 다양한 방법론을 제시하고 있으며, 특히 템플릿 변동성이 큰 문서에 대한 접근성을 높이기 위해 자연어 처리(NLP), 컴퓨터 비전(CV), 다중 모달(multi-modal) 방법론을 활용한 연구를 수행하고 있습니다. 이 연구는 데이터의 정확성과 효율성을 높이기 위한 실험 결과를 제시하며, 향후 연구에 대한 가치 있는 통찰력을 제공합니다.

- **Technical Details**: 메타데이터 추출에 있어 여러 가지 접근 방식을 비교하고, 고전적 방법인 Conditional Random Fields와 고급 NLP 기법인 BiLSTM과 BERT 표현을 함께 사용합니다. 또한, Generative LLMs와 같은 텍스트 생성을 위한 모델들이 구조적 작업에 적합하지 않다는 점을 강조하고, 그 대신 BERT와 같은 아키텍처를 통해 과학 문서의 독특한 레이아웃 가변성과 다중 모달 내용을 효율적으로 처리합니다.

- **Performance Highlights**: 본 연구는 SSOAR-MVD와 S-PMRD라는 두 가지 데이터셋을 만들어 메타데이터 추출 성능을 평가하였습니다. 실험을 통해 여러 기법의 정확성과 효과를 비교하며, 모든 접근 방식의 구현을 공개하여 재현성과 향후 연구 개발을 돕고자 합니다.



### Jailbreaking Multimodal Large Language Models via Shuffle Inconsistency (https://arxiv.org/abs/2501.04931)
- **What's New**: 이번 연구에서는 다중 모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 안전 메커니즘에서 발생할 수 있는 취약점을 탐지하는 새로운 공격 방법인 SI-Attack을 제안합니다. 기존 연구와 달리, MLLMs의 이해 능력과 안전 능력 사이의 'Shuffle Inconsistency'를 발견하였으며, 이는 모델이 혼합된 해로운 지시를 잘 이해하지만, 이러한 지시를 통한 공격에는 쉽게 노출된다는 점을 강조합니다.

- **Technical Details**: SI-Attack은 해로운 지시를 정교하게 선택하기 위해 쿼리 기반 블랙박스 최적화 방법을 활용하며, 독성 판별 모델의 피드백에 기초하여 가장 해로운 혼합 입력을 선별합니다. 이를 통해 SI-Attack은 상업적으로 사용되는 태스크에서도 공격 성공률을 크게 향상시킬 수 있습니다.

- **Performance Highlights**: 일련의 실험 결과, SI-Attack은 GPT-4o 및 Claude-3.5-Sonnet과 같은 상업적 MLLMs에서 공격 성공률을 유의미하게 증가시켰습니다. 이 연구는 해로운 지시가 제대로 대응되지 않는 MLLMs의 취약성을 체계적으로 입증하며, 향후 안전한 AI 시스템 구축에 중요한 기초 자료로 활용될 것입니다.



### FLowHigh: Towards Efficient and High-Quality Audio Super-Resolution with Single-Step Flow Matching (https://arxiv.org/abs/2501.04926)
Comments:
          Accepted by ICASSP 2025

- **What's New**: FLowHigh는 오디오 초해상도(Super-Resolution) 분야에 새로운 접근법을 제시합니다. 이 연구는 전통적인 확산 모델(diffusion models)의 단점을 보완하기 위해, 흐름 매칭(flow matching) 방법을 통합하여 고해상도 오디오 신호를 단일 스텝 샘플링으로 생성합니다. 또한, 고해상도 오디오 분포를 효과적으로 포착하기 위해 오디오 초해상도에 맞춤형 확률 경로(probability paths)를 탐색합니다.

- **Technical Details**: FLowHigh는 저해상도(Low-Resolution) 오디오 신호를 고해상도(High-Resolution) 신호로 변환하는 방법론을 제안합니다. 이 방법은 간단한 벡터 필드 회귀(vector field regression)를 통해 사전 분포(prior distribution)와 복잡한 데이터 분포(data distribution) 간의 변환을 학습합니다. 또한, mel-spectrogram 수준에서 벡터 필드를 회귀하기 위한 트랜스포머 기반(vector field estimator)을 활용하며, 예측된 파형을 합성하기 위해 사전 학습된 신경 보코더(pre-trained neural vocoder)를 사용합니다.

- **Performance Highlights**: VCTK 벤치마크 데이터셋에서 실험한 결과, FLowHigh는 기존 모델들보다 월등한 성능을 보였으며, 로그 스펙트럴 거리(log-spectral distance)와 ViSQOL 평가 지표에서 최첨단 성능을 달성했습니다. 단일 스텝 샘플링(single-step sampling) 과정으로 높은 충실도의 고해상도 오디오를 생성하면서 계산 효율성도 유지하였습니다. FLowHigh는 오디오 초해상도 연구에 있어 흐름 매칭 기법을 성공적으로 통합한 첫 번째 시도로 평가받고 있습니다.



### Enhancing Listened Speech Decoding from EEG via Parallel Phoneme Sequence Prediction (https://arxiv.org/abs/2501.04844)
Comments:
          ICASSP 2025

- **What's New**: 이번 논문에서는 뇌-컴퓨터 인터페이스(Brain-Computer Interface, BCI)에서 전기 생리학적 신호를 통해 들은 음성을 동시에 음성 파형(speech waveform)과 텍스트 음소 시퀀스(textual phoneme sequences)로 디코딩할 수 있는 새로운 접근 방식을 제안합니다. 기존의 방법들은 단일 모달리티에서만 작업이 가능했으나, 이 연구는 EEG 신호를 활용하여 두 가지 모달리티를 동시에 처리함으로써 기존 시스템의 한계를 극복합니다. 또한, 이 방법은 보조 음소 예측기(auxiliary phoneme predictor)를 통합하여 성능을 향상시키고 있어, 새로운 연산적 가능성을 제시합니다.

- **Technical Details**: 제안된 프레임워크는 EEG 모듈, 음성 모듈 및 음소 예측기의 세 가지 주요 구성 요소로 이루어져 있습니다. EEG 모듈은 EEG 신호로부터 임베딩(embedding)을 학습하며, 생성된 임베딩은 음소 예측기와 음성 모듈에 병렬로 전달됩니다. 특히, 음소 예측기는 EEG 임베딩을 기반으로 음소 시퀀스를 디코딩하며, 음성 모듈은 음성 파형 생성을 담당합니다. 이 프레임워크는 각 모달리티에 대해 순차적 파이프라인을 요구하지 않고 동시에 디코딩 기능을 제공합니다.

- **Performance Highlights**: 제안된 모델은 이전의 모든 방법들과 비교해 유의미한 성능 향상을 보였으며, 특히 음성 파형 및 음소 시퀀스를 모두 디코딩하는 데 있어 효과적입니다. 논문에서는 음소 시퀀스 및 음성 파형의 디코딩 성능을 보여주는 표를 포함하고 있으며, 제안된 접근 방식의 우수성을 실험적으로 입증하고 있습니다. 이 논문에 포함된 소스 코드와 음성 샘플은 공개적으로 제공되어 있어, 연구자들이 결과를 재현하고 더 발전시키는 데 도움을 줄 것입니다.



### Reproducing HotFlip for Corpus Poisoning Attacks in Dense Retrieva (https://arxiv.org/abs/2501.04802)
Comments:
          This paper has been accepted for oral presentation in the reproducibility track at ECIR 2025

- **What's New**: 이 논문에서는 HotFlip 기법의 효율성을 크게 개선하여 적대적(adversarial) passage 생성 과정을 기존의 4시간에서 15분으로 단축시킨 내용을 소개합니다. 연구자들은 두 가지 추가 과제를 통해 HotFlip의 성능을 평가하였으며, 전이 기반 공격과 쿼리 비독립적 공격에 대한 실험 결과도 제시합니다. 이렇게 개선된 HotFlip은 다양한 밀집 검색기(dense retriever)에 대해 효과적으로 공격할 수 있음을 보여주었습니다.

- **Technical Details**: 이 논문은 다중 인코더(multi-encoder) 밀집 검색 시스템에서의 적대적 passage 생성 파이프라인을 상세히 설명합니다. 기존의 HotFlip 방법은 각 쿼리-패세이지 쌍에 대한 그래디언트 계산에서 비효율적이라고 알려져 있으며, 연구자들은 쿼리의 중심을 클러스터링하여 효율성을 높이는 최적화 전략을 사용했습니다. 이러한 방법론은 각 패세이지의 임베딩을 고려하여 그래디언트 계산 단계를 최적화하는 데 중점을 둡니다.

- **Performance Highlights**: 실험 결과, HotFlip의 개선된 버전은 공격 성능을 유지하면서도 훨씬 더 빠른 속도로 고품질의 적대적 문서를 생성할 수 있음을 입증했습니다. 그러나 HotFlip은 블랙박스 설정에서 일반화 능력이 제한적이라는 점도 드러났고, 쿼리 비독립적 상황에서는 생성된 적대적 패세이지의 양에 따라 성능이 크게 달라질 수 있음을 확인했습니다. 이 연구는 자주 사용되는 공격 방식을 실제적인 시나리오에서도 적용할 수 있는 가능성을 보여주고 있습니다.



New uploads on arXiv(cs.IR)

### Unraveling the Impact of Visual Complexity on Search as Learning (https://arxiv.org/abs/2501.05289)
- **What's New**: 이 연구에서는 학습지향적인 정보 검색(Session)에서 시각적 복잡성(Visual Complexity, VisCom)이 지식 습득(Knowledge Gain, KG)에 미치는 영향을 조사했습니다. 웹 페이지의 시각적 복잡성은 전반적인 학습 성공에 긍정적인 영향을 미친다는 결과를 도출했으며, 특히 페이지 레이아웃과 관련된 복잡성이 주요 요인으로 작용하는 것으로 나타났습니다. 이는 정보 검색 시스템의 설계 개선에 기여할 것입니다.

- **Technical Details**: 본 연구는 HTML 특성, 시각적 특성, 레이아웃 특성, 미적 특성의 네 가지 범주를 사용하여 VisCom을 모델링합니다. 또한, 시각적 특성의 경우 평균 밝기(avg_brightness), 평균 색조(avg_hue), 평균 색 채도(avg_colorfulness) 등의 다양한 메트릭을 포함하여 시각적 정보를 수집합니다. 이 데이터를 통해 웹 페이지의 복잡성과 사용자 학습 과정 간의 관계를 분석합니다.

- **Performance Highlights**: 결과적으로, 페이지의 시각적 복잡성이 낮을수록 학습 성공이 높아지는 경향을 보였습니다. 이러한 발견은 웹 기반 학습 세션에서의 효과적인 정보 검색을 위한 새로운 지침을 제시하며, 관련 정보 검색 시스템의 최적화에 중요한 기초 자료를 제공합니다. 연구의 reproducibility를 위해 소스 코드를 공개했습니다.



### De-centering the (Traditional) User: Multistakeholder Evaluation of Recommender Systems (https://arxiv.org/abs/2501.05170)
Comments:
          Preprint submitted to Elsevier, "Re-centering the User in Recommender System Research" special issue of the International Journal of Human-Computer Studies (IJHCS)

- **What's New**: 이번 논문은 다수의 이해관계자(multi-stakeholder)를 고려한 추천 시스템(recommender system)에 대한 평가 방법론을 다룹니다. 전통적으로 소비자 관점에서만 평가되던 추천 시스템의 한계를 넘어서, 제작자(producers), 소비자(consumers) 외에도 다양한 이해관계자가 시스템에 미치는 영향을 분석합니다. 특히, 다양한 이해관계자의 목표와 가치에 대한 논의와 함께, 이들 간의 복잡한 관계를 설명합니다.

- **Technical Details**: 이 연구에서는 다수의 이해관계자의 평가를 위한 다양한 고려사항과 가치를 정의하고, 다수의 이해관계자 평가(multistakeholder evaluation)를 위한 방법론을 제안합니다. 또한, 음악 스트리밍 애플리케이션을 예로 들어, 시스템의 다양한 이해관계자들(예: 작곡가, 아티스트, 소비자)의 목표와 해당 목표를 평가하기 위한 지표(metric)에 대해 설명합니다. 가령, 인기 아티스트의 노래를 추천하는 것이 경제적 이익을 높이는 반면, 덜 알려진 아티스트의 가시성을 해칠 수 있는 상황을 제시합니다.

- **Performance Highlights**: 기존 추천 시스템 연구의 대다수가 소비자 관점만을 강조하고 있는 가운데, 이 논문은 더 넓은 이해관계자 생태계(ecosystem)를 고려한 평가의 필요성을 강조합니다. 추천 시스템 디자인, 구현 및 유지 관리 측면에서의 평가의 복잡성을 명확히 하며, 연구자들과 실무자들이 다수의 이해관계자 평가를 통합할 수 있는 방법을 제공하고자 합니다. 다양한 이해관계자 활용을 위한 실질적인 사례도 제시하여, 앞으로의 연구 방향에 대해 논의합니다.



### Comparison of Feature Learning Methods for Metadata Extraction from PDF Scholarly Documents (https://arxiv.org/abs/2501.05082)
- **What's New**: 본 논문에서는 과학 문서의 메타데이터 추출을 위한 다양한 방법론을 제시하고 있으며, 특히 템플릿 변동성이 큰 문서에 대한 접근성을 높이기 위해 자연어 처리(NLP), 컴퓨터 비전(CV), 다중 모달(multi-modal) 방법론을 활용한 연구를 수행하고 있습니다. 이 연구는 데이터의 정확성과 효율성을 높이기 위한 실험 결과를 제시하며, 향후 연구에 대한 가치 있는 통찰력을 제공합니다.

- **Technical Details**: 메타데이터 추출에 있어 여러 가지 접근 방식을 비교하고, 고전적 방법인 Conditional Random Fields와 고급 NLP 기법인 BiLSTM과 BERT 표현을 함께 사용합니다. 또한, Generative LLMs와 같은 텍스트 생성을 위한 모델들이 구조적 작업에 적합하지 않다는 점을 강조하고, 그 대신 BERT와 같은 아키텍처를 통해 과학 문서의 독특한 레이아웃 가변성과 다중 모달 내용을 효율적으로 처리합니다.

- **Performance Highlights**: 본 연구는 SSOAR-MVD와 S-PMRD라는 두 가지 데이터셋을 만들어 메타데이터 추출 성능을 평가하였습니다. 실험을 통해 여러 기법의 정확성과 효과를 비교하며, 모든 접근 방식의 구현을 공개하여 재현성과 향후 연구 개발을 돕고자 합니다.



### A Flexible and Scalable Framework for Video Moment Search (https://arxiv.org/abs/2501.05072)
- **What's New**: 비디오 순간 검색(Video moment search) 분야는 사용자의 질의와 일치하는 비디오 내 관련 순간을 찾는 과정으로, 매우 중요합니다. 그러나 기존의 접근법은 단일 완벽한 매칭 순간을 가정하거나 효율적인 추론에 어려움을 겪으며, 장시간 비디오에 대한 한계가 있습니다. 이 논문에서는 비디오 길이에 관계없이 텍스트 질의와 맞는 순간의 순위 목록을 검색할 수 있는 유연하고 확장 가능한 프레임워크인 Rank Video Moment Retrieval (RVMR)을 제안합니다.

- **Technical Details**: 이 연구에서 제안하는 SPR(Segment-Proposal-Ranking) 프레임워크는 세 가지 독립적인 단계로 검색 프로세스를 단순화합니다: 세그먼트 검색(segment retrieval), 제안 생성(proposal generation), 그리고 순간 정제(moment refinement) 및 재정렬(re-ranking)입니다. 비디오는 균일한 길이의 세그먼트로 나누어져 사전 계산된 임베딩(embeddings)을 오프라인으로 색인화하며, 지속적으로 효율적 검색을 가능하게 합니다. 를 위해 세그먼트와 질의를 공통 피처 공간으로 투영하여 근사 최근 이웃(ANN) 검색을 수행합니다.

- **Performance Highlights**: TVR-Ranking 데이터셋에 대한 평가 결과, SPR 프레임워크는 계산 비용과 처리 시간을 크게 줄이며 최첨단 성능을 달성함을 보여줍니다. 특히, 2만 개의 비디오에 대한 사용자 질의 처리가 평균적으로 1초도 안 걸리며, 다양한 무관한 비디오가 추가되어도 안정성을 보여줍니다. SPR 프레임워크의 유연한 설계 덕분에 각 구성요소의 독립적인 개선이 가능하여 대규모 응용에 매우 적합합니다.



### Finding Needles in Emb(a)dding Haystacks: Legal Document Retrieval via Bagging and SVR Ensembles (https://arxiv.org/abs/2501.05018)
- **What's New**: 본 연구에서는 독일 법률 정보 검색(GerDaLIR) 데이터셋을 활용하여 Support Vector Regression (SVR) 앙상블 기법과 배깅(bootstrap aggregation)을 적용한 검색 접근 방식을 소개합니다. 우리는 정보 검색 문제를 다수의 이진 하위 작업(needle-in-a-haystack)으로 개념화하여, 훨씬 더 높은 재현율(recall)을 달성했음을 보여주었습니다. 이 방식은 심층 학습 모델을 훈련시키거나 미세 조정(fine-tuning)하지 않고도 가능합니다. 나아가, 인코딩 모델의 개선 및 하이퍼파라미터 최적화를 통해 더 나은 성능 향상이 가능할 것으로 보입니다.

- **Technical Details**: 정보 검색(IR)은 많은 데이터에서 특정 쿼리에 따라 정보 단위를 식별하고 추출하는 중요한 과정입니다. 법률 정보 검색(LIR)에서는 관련된 법률 문서들의 모음에서 특정 문서를 찾아내는 것이 중요하며, TF-IDF와 BM25 같은 다양한 자연어 처리(NLP) 기법이 활용됩니다. 본 연구에서는 여러 개의 약한 Support Vector Regressors(SVR)를 앙상블하여 문서의 입력을 고차원 임베딩 벡터로 변환하고 배깅 기법을 결합하여 성능을 향상시켰습니다.

- **Performance Highlights**: 우리의 접근 방식은 높아진 재현율(0.849)로 이전 최적 모델들보다 더 나은 성능을 나타냈습니다. 특히, 이 결과는 대규모의 심층 학습 모델에 대한 미세 조정이 없었던 상태에서 달성되었습니다. GerDaLIR 데이터셋을 활용한 우리의 방법론은 향후에도 법률 정보 검색(LIR) 분야에서 많은 가능성을 제시할 것으로 보고되었습니다. 마지막으로, 연구 결과에 대한 소스 코드는 GitHub에서 공개되었습니다.



### Reproducing HotFlip for Corpus Poisoning Attacks in Dense Retrieva (https://arxiv.org/abs/2501.04802)
Comments:
          This paper has been accepted for oral presentation in the reproducibility track at ECIR 2025

- **What's New**: 이 논문에서는 HotFlip 기법의 효율성을 크게 개선하여 적대적(adversarial) passage 생성 과정을 기존의 4시간에서 15분으로 단축시킨 내용을 소개합니다. 연구자들은 두 가지 추가 과제를 통해 HotFlip의 성능을 평가하였으며, 전이 기반 공격과 쿼리 비독립적 공격에 대한 실험 결과도 제시합니다. 이렇게 개선된 HotFlip은 다양한 밀집 검색기(dense retriever)에 대해 효과적으로 공격할 수 있음을 보여주었습니다.

- **Technical Details**: 이 논문은 다중 인코더(multi-encoder) 밀집 검색 시스템에서의 적대적 passage 생성 파이프라인을 상세히 설명합니다. 기존의 HotFlip 방법은 각 쿼리-패세이지 쌍에 대한 그래디언트 계산에서 비효율적이라고 알려져 있으며, 연구자들은 쿼리의 중심을 클러스터링하여 효율성을 높이는 최적화 전략을 사용했습니다. 이러한 방법론은 각 패세이지의 임베딩을 고려하여 그래디언트 계산 단계를 최적화하는 데 중점을 둡니다.

- **Performance Highlights**: 실험 결과, HotFlip의 개선된 버전은 공격 성능을 유지하면서도 훨씬 더 빠른 속도로 고품질의 적대적 문서를 생성할 수 있음을 입증했습니다. 그러나 HotFlip은 블랙박스 설정에서 일반화 능력이 제한적이라는 점도 드러났고, 쿼리 비독립적 상황에서는 생성된 적대적 패세이지의 양에 따라 성능이 크게 달라질 수 있음을 확인했습니다. 이 연구는 자주 사용되는 공격 방식을 실제적인 시나리오에서도 적용할 수 있는 가능성을 보여주고 있습니다.



### Efficient and Responsible Adaptation of Large Language Models for Robust and Equitable Top-k Recommendations (https://arxiv.org/abs/2501.04762)
Comments:
          arXiv admin note: text overlap with arXiv:2405.00824

- **What's New**: 본 논문은 기존 추천 시스템(Recommendation Systems, RSs)이 다양한 사용자 집단의 요구를 간과하는 문제를 해결하기 위해 하이브리드 작업 할당 프레임워크를 제안합니다. 이는 대규모 언어 모델(LLMs)을 통해 추천의 질을 높이고, 사회적 공익을 위한 더 공정한 사용자 서비스 제공을 목표로 합니다. 두 단계의 접근 방식을 통해, 비활성 사용자와 약한 사용자에게 초점을 맞추어 그들의 상호작용 기록을 최적화하여 더 나은 추천을 제공하는 방법을 소개합니다.

- **Technical Details**: 하이브리드 프레임워크는 약한 사용자에 대한 세분화된 상호작용 분석을 기반으로 하여, 특정 기준치 이하의 추천 성능을 지닌 사용자들을 식별합니다. 이러한 약한 사용자들은 in-context learning을 활용하여 각자의 상호작용 이력을 다루어 추천 품질을 높입니다. 이를 통해 더욱 공정하고 포괄적인 추천 시스템을 구현하는 것을 목표로 하고 있습니다.

- **Performance Highlights**: 논문에서는 열 개의 추천 알고리즘 및 세 개의 LLM을 통합한 하이브리드 프레임워크를 평가하였고, 그 결과 약한 사용자의 수를 현저히 줄이고 하위 집단에 대한 강건성을 개선하게 되었습니다. 특히, 약 12%의 강건성 향상을 보였고, LLM을 adaption하는 데 필요한 높은 비용을 효율적으로 줄이는 데 성공했습니다. 최종적으로 사용자 집단 간의 불균형 문제를 해결하여 다양한 사용자에게 양질의 추천을 제공할 수 있는 가능성을 여는 연구 결과를 확인했습니다.



### Search-o1: Agentic Search-Enhanced Large Reasoning Models (https://arxiv.org/abs/2501.05366)
- **What's New**: 이 논문은 기존의 대규모 추론 모델(LRM)에서 발생하는 지식 부족 문제를 해결하기 위해 	extbf{Search-o1}이라는 새로운 프레임워크를 소개합니다. Search-o1은 에이전트 기반 검색 증강 생성(RAG) 메커니즘과 문서 내 추 reasoning(Reason-in-Documents) 모듈을 결합하여 외부 지식을 동적으로 검색하고 통합할 수 있도록 합니다. 이러한 접근 방식은 LRMs의 신뢰성과 적용 가능성을 향상시켜 더 신뢰할 수 있는 지능형 시스템을 구축하는 데 기여합니다.

- **Technical Details**: Search-o1의 디자인은 두 가지 핵심 요소에 기반하고 있습니다. 첫째, 에이전트 기반 RAG 메커니즘을 통해 LRM은 지식 부족 상태에서 적절한 외부 지식을 검색할 수 있도록 유도됩니다. 둘째, 별도의 Reason-in-Documents 모듈은 검색된 문서의 정보를 심층적으로 분석하여 원래의 추론 체계에 통합됩니다. 이러한 두 가지 측면이 상호 작용하여 LRM의 추론 과정을 강화하고 논리적 흐름을 유지할 수 있도록 합니다.

- **Performance Highlights**: 다양한 과학, 수학, 코딩 분야의 복잡한 추론 작업과 여섯 가지 공개 도메인 질문 응답(QA) 벤치마크에서 Search-o1의 강력한 성능이 입증되었습니다. Search-o1은 기존의 LRMs에 비해 신뢰성과 효율성을 크게 향상시켰으며, 이론적 및 정량적으로 검증된 결과는 LRM의 신뢰할 수 있는 추론을 위한 실질적인 지침을 제공합니다.



### A Novel Approach to Scalable and Automatic Topic-Controlled Question Generation in Education (https://arxiv.org/abs/2501.05220)
Comments:
          To be published at ACM Conf. on Learning Analytics and Knowledge (LAK'25)

- **What's New**: 본 논문은 Topic-Controlled Question Generation (T-CQG)이라는 새로운 접근 방식을 제시하여 교육 목적으로 생성된 질문의 관련성과 효과성을 향상시킵니다. T-CQG는 pre-trained T5-small 모델을 미세 조정하여 특별히 만들어진 교육 맞춤형 데이터셋을 사용합니다. 연구는 사전 훈련 전략, 양자화(quantisation), 데이터 증대(data augmentation) 등이 모델 성능에 미치는 영향을 탐구하며, 문단 수준의 맥락에 맞는 의미적으로 일치하는 질문 생성을 다룹니다. 또한, 생성된 질문의 주제 관련성을 평가하기 위한 새로운 평가 방법을 소개합니다.

- **Technical Details**: T-CQG 모델은 고급 자연어 처리(NLP) 기술을 활용하여 교육 질문의 생성을 자동화합니다. 이 모델은 약 60M 파라미터를 가진 매우 작은 언어 모델(small language model)을 사용하여 교육용 질문을 생성하는 데 성공했습니다. 주제 제어(question generation)의 중요성을 강조하며, 학습 자료의 맥락과 관련된 주제를 기준으로 질문을 생성하는 방법론을 제시합니다. 이러한 접근은 교육용 질문의 품질을 개선하고 교사들이 질문을 작성하는 데 드는 수고를 덜어줄 수 있습니다.

- **Performance Highlights**: 연구 결과는 오프라인 실험 및 인간 평가(human-backed evaluations)를 통해 검증되었으며, T-CQG 모델이 고품질의 주제 중심 질문을 효과적으로 생성할 수 있음을 입증합니다. 교육 시스템에서의 적용 가능성이 높아 교사들의 업무 부담을 줄이고 개인 맞춤형 튜터링 시스템을 지원하는 데 기여할 것으로 기대됩니다. 본 연구는 교사 retention 문제와 관련된 생산성 향상을 위한 중요한 실용적 단계를 제공하여 교육 환경을 개선할 가능성이 큽니다.



New uploads on arXiv(cs.CV)

### An Empirical Study of Autoregressive Pre-training from Videos (https://arxiv.org/abs/2501.05453)
- **What's New**: 이 논문은 비디오에서의 자기회귀(autoregressive) 사전 학습(pre-training)을 실증적으로 연구합니다. 저자들은 Toto라는 일련의 비디오 모델을 구축하고, 비디오를 시각적 토큰의 연속으로 처리하여 Transformer 모델이 미래 토큰을 예측하도록 훈련합니다. 1조 개가 넘는 시각적 토큰을 포함하는 다양한 데이터셋에서 이러한 모델을 사전 훈련하고 다양한 다운스트림 작업에서 평가한 결과, 최소한의 유도 편향에도 불구하고 경쟁력 있는 성능을 보였습니다.

- **Technical Details**: 비디오를 시각적 토큰의 시퀀스로 간주하고, 각 프레임을 dVAE(Ramesh et al., 2021)로 불연속 토큰화(tokenization)합니다. 그런 다음 LLaMa(Touvron et al., 2023) 아키텍처의 인과적 Transformer 모델을 사용하여 다음 토큰 예측 작업에 대해 훈련합니다. 다양한 설계 선택을 활용하여 모델을 평가하고, 주목 풀링(attention pooling)을 사용하여 시각적 표현을 추출합니다.

- **Performance Highlights**: 모델들은 이미지 인식, 비디오 분류, 물체 추적, 로봇 작업과 같은 효과적인 다운스트림 작업에서 성능을 평가받았고, 결과적으로 안정적인 고성능을 나타냈습니다. 이미지넷 분류 작업에서는 불연속 및 연속 패치정규화(continuous patch-normalized) 토큰을 기반으로 한 자가회귀 모델들이 유사한 성능을 보여주었습니다. 마지막으로, 자기회귀 비전 모델들은 언어 모델에 비해 더 느린 속도로 확장되는 특성을 보였습니다.



### ReFocus: Visual Editing as a Chain of Thought for Structured Image Understanding (https://arxiv.org/abs/2501.05452)
Comments:
          Project link: this https URL

- **What's New**: 이 논문에서는 ReFocus라는 새로운 프레임워크를 소개하여 멀티모달 대형 언어 모델(LLMs)이 이미지에서 직접적으로 "visual thoughts"를 생성하고 조작할 수 있는 능력을 갖추도록 합니다. 이를 통해 다양한 구조화된 이미지 과제, 특히 표와 차트에 대한 시각적 이해 능력을 크게 향상시킬 수 있습니다. ReFocus는 모델이 Python 코드를 생성하여 시각적 편집을 수행하도록 하여 멀티홉 시각적 추론을 가능하게 합니다.

- **Technical Details**: ReFocus는 입력 이미지를 수정하는 시각 편집 작업을 통해 멀티모달 LLMs가 선택적 주의를 향상시킬 수 있도록 돕습니다. 모델은 코드 실행을 통해 입력 이미지를 동적으로 수정하여, 불필요한 정보를 제거하거나 중요한 정보를 강조할 수 있습니다. 이러한 과정은 시각적 사고의 연속적인 단계로 진행되며, 표 문제에 대한 평균 성능 향상률은 11.0%, 차트 문제에 대한 성능 향상률은 6.8%에 달합니다.

- **Performance Highlights**: ReFocus는 기준 모델인 GPT-4o와 비교했을 때, 표 및 차트 문제에서 일관된 성능 향상을 나타냅니다. 특히, 모델이 REFOCUS 데이터를 활용하도록 미세 조정할 경우, 전통적인 질문-답변(QA) 데이터보다 더 나은 성능을 보이는 것으로 나타났습니다. 이러한 실험 결과는 ReFocus의 효과성과 향후 시각 언어 모델들에 대한 개선 가능성을 제시합니다.



### Decentralized Diffusion Models (https://arxiv.org/abs/2501.05450)
Comments:
          Project webpage: this https URL

- **What's New**: 이번 논문에서는 기존의 중앙 집중형 인프라의 의존성을 제거하는 분산 훈련 프레임워크인 Decentralized Diffusion Models(DDMs)을 제안합니다. DDMs는 독립적인 클러스터들을 통해 확장 가능하게 훈련할 수 있으며, 이는 GPU 클러스터 간의 네트워크 부담을 줄이고 인프라 비용을 낮추는데 기여합니다. 연구자들이 더욱 효율적이고 저렴한 자원을 활용할 수 있게 되어, 특정 GPU 노드를 통해 수요 기반으로 적시에 할당할 수 있는 기회를 제공합니다.

- **Technical Details**: Decentralized Diffusion Models는 데이터셋의 파티션을 기반으로 전문가 모델을 훈련하여, 이전의 대규모 훈련 요구 사항을 분산 처리합니다. 새로운 훈련 목적인 Decentralized Flow Matching(DFM)을 적용하여 개별 전문가 모델은 서로 교류없이 독립적 훈련을 진행합니다. 이때, 라우터 모델이 각 전문가의 관련성을 판단하여 테스트 시 최적의 예측 값을 제공하도록 합니다.

- **Performance Highlights**: DDMs는 ImageNet과 LAION Aesthetics 데이터셋에서 기존의 모노리스(Monolithic) 확산 모델 훈련보다 더 나은 성능을 보였습니다. 8개의 전문가 모델을 통해 최적의 성능을 달성했으며, 이는 일반 모델보다 더 효과적인 대체가 될 수 있음을 보여줍니다. 마지막으로, 8개의 30억 매개변수를 가진 대규모 분산 모델을 훈련하여 고해상도 이미지 생성을 입증하였습니다.



### Explainable AI-Enhanced Deep Learning for Pumpkin Leaf Disease Detection: A Comparative Analysis of CNN Architectures (https://arxiv.org/abs/2501.05449)
Comments:
          Accepted in 2024 27th International Conference on Computer and Information Technology (ICCIT)

- **What's New**: 이 연구는 호박 잎 질병을 진단하기 위한 자동화된 솔루션의 필요성을 강조하며, 2000개의 고해상도 이미지를 사용한 데이터셋을 기반으로 여러 딥러닝 아키텍처를 평가하였습니다. 특히 ResNet50 모델이 90.5%의 정확도를 기록하였고, 설명 가능한 인공지능(Explainable AI, XAI) 기술을 활용하여 모델의 의사 결정 과정을 시각적으로 보여주어 신뢰성을 높였습니다. 이러한 접근은 호박 잎 질병 탐지의 혁신적인 가능성을 보여주며, 조기 및 정확한 치료를 가능하게 합니다.

- **Technical Details**: 연구에서는 DenseNet201, DenseNet121, DenseNet169, Xception, ResNet50, ResNet101, InceptionResNetV2와 같은 여러 딥러닝 아키텍처를 탐색하였고, ResNet50이 가장 효과적으로 작동함을 확인했습니다. 또한, Grad-CAM, Grad-CAM++, Score-CAM, Layer-CAM과 같은 XAI 접근법을 사용하여 모델의 예측 과정을 시각적으로 설명하는 기술을 도입하였습니다. 이는 농업에서 AI의 투명성과 신뢰성을 높이는 데 기여합니다.

- **Performance Highlights**: ResNet50 모델은 90.5%의 정확도를 달성하였으며, 정밀도, 재현율, F1 점수에서도 유사한 성과를 거두었습니다. 자동화된 호박 잎 분류를 위해 평가된 7개의 CNN 모델 가운데 ResNet50이 가장 높은 성능을 발휘했습니다. 이러한 결과는 XAI 기술이 포함된 AI 모델이 농업 질병 관리에서 필요성과 실용성을 높일 수 있음을 시사합니다.



### Relative Pose Estimation through Affine Corrections of Monocular Depth Priors (https://arxiv.org/abs/2501.05446)
- **What's New**: 이 논문에서는 모노큘러 깊이 추정(Monocular Depth Estimation, MDE) 모델의 발전을 바탕으로 상대 자세 추정(relative pose estimation) 문제를 해결하기 위한 세 가지 새로운 솔버(solver)를 제안합니다. 제안된 솔버는 카메라 세팅의 보정(calibrated) 여부에 관계없이 독립적인 아핀(affine) 왜곡을 고려하며, 특히 심도 예측에서의 스케일(scale)과 시프트(shift) 변화를 모델링합니다. 그리고 기존의 포인트 기반 솔버와 에피폴라(epipolar) 제약을 결합한 하이브리드(hybrid) 추정 파이프라인을 처음으로 제공합니다.

- **Technical Details**: 제안하는 솔버는 보정된 이미지 쌍에 대해 3-포인트 솔버(calibrated 3-point solver), 공통 초점 거리의 비보정 이미지 쌍을 위한 4-포인트 솔버(shared-focal 4-point solver), 그리고 완전히 비보정된 이미지 쌍을 위한 4-포인트 솔버(two-focal 4-point solver)를 포함합니다. 이러한 솔버는 MDE 모델에서 가져온 깊이 예측과 함께 사용되어, 스케일과 시프트 변수를 명시적으로 모델링합니다. 또한, 고전적인 포인트 기반 솔버와의 통합을 통해 정확성과 강건성을 크게 개선할 수 있는 최적화 기법을 개발했습니다.

- **Performance Highlights**: 다양한 데이터셋을 통해 제안된 방법이 기존의 키포인트 기반 솔루션 및 PnP 기반 솔루션에 비해 큰 성능 향상을 보여줍니다. 특히, 보정된 및 비보정된 구조 모두에서 상대 깊이 예측뿐만 아니라 메트릭 깊이 예측(metric depth)에서도 이점이 있음을 발견했습니다. 최종적으로, 다양한 특징 매처(feature matcher)와 MDE 모델에 대해 일관되게 성능이 향상되어, 기존의 파이프라인에 쉽게 통합될 수 있는 강력한 상대 자세 추정기가 구현되었습니다.



### Consistent Flow Distillation for Text-to-3D Generation (https://arxiv.org/abs/2501.05445)
Comments:
          Project page: this https URL

- **What's New**: 본 논문에서는 3D 생성 모델의 효율성을 개선하기 위한 새로운 방법인 Consistent Flow Distillation (CFD)을 제안합니다. 이를 통해 기존의 Score Distillation Sampling (SDS)의 한계인 시각적 품질과 다양성 저하 문제를 해결하고자 합니다. CFD는 2D 이미지 플로우의 일관성을 강조하며, 다양한 시점에서의 3D 생성 품질을 개선합니다.

- **Technical Details**: CFD는 다중 시점에서의 일관된 Gaussian noise를 활용하여 3D 객체의 생성을 돕습니다. 이 과정에서 noise transport equation을 기반으로 물체 표면에서의 일관된 noise 패턴을 유지하여 그라디언트를 계산합니다. 이는 diffusion ODE(Ordinary Differential Equation) 또는 SDE(Stochastic Differential Equation) 샘플링 과정을 통해 3D 생성을 직접 유도합니다.

- **Performance Highlights**: 실험 결과, CFD는 기존의 텍스트-투-3D(text-to-3D) 생성 방식에 비해 품질과 다양성에서 현저히 개선된 성능을 보였습니다. 본 방법을 통해 생성된 3D 자산은 현실적인 외관과 형태를 가지며, 동일한 텍스트 프롬프트에 대해 다양한 3D 객체를 샘플링할 수 있습니다. CFD는 기존의 SDS와 비교하여 계산 비용이 거의 추가되지 않았음에도 높은 성능을 달성했습니다.



### Can MLLMs Reason in Multimodality? EMMA: An Enhanced MultiModal ReAsoning Benchmark (https://arxiv.org/abs/2501.05444)
- **What's New**: EMMA(Enhanced MultiModal reAsoning)라는 새로운 벤치마크가 소개되어, 수학, 물리학, 화학, 코딩 분야에서 통합된 시각적 및 언어적 추론을 평가할 수 있도록 설계되었습니다. 이 벤치마크는 단순한 문자 기반 추론에 의존하지 않고, 다양한 모달리티를 아우르는 문제 해결 능력을 요구합니다. 현재의 MLLMs(Multimodal Large Language Models)가 직면한 한계를 드러내며, 기존의 단순 시각적 이해나 텍스트 회상만으로는 충분하지 않음을 강조합니다.

- **Technical Details**: EMMA는 2,788개의 문제로 구성되어 있으며, 그 중 1,796문제가 새롭게 개발되었습니다. 이 벤치마크는 문제의 종류를 세분화하여, MLLMs의 다양한 능력을 평가하기 위한 레이블이 도입되었습니다. 예를 들어, 수학 문제에서는 시각적 패턴을 일반화하는 것, 물리학 문제에서는 그래픽을 사용해 힘을 분해하는 것이 요구됩니다. 이와 같은 문제들은 시각적 보조 도구가 필수적입니다.

- **Performance Highlights**: 최신 MLLMs는 EMMA에서의 성과가 미비하여, 가장 높은 점수를 기록한 모델도 인간 전문가와는 32% 차이가 나는 45.75%의 점수에 머물렀습니다. 더 나아가, Chain-of-Thought와 같은 고급 기법이 사용되었음에도 불구하고 다중 모달리티 문제에 대한 성능 저하가 관찰되었습니다. 이 연구는 MLLMs의 시각적 추론 능력의 한계를 분명히 드러내어, 향후 모델 개발 및 훈련 패러다임 개선의 필요성을 강조합니다.



### Progressive Growing of Video Tokenizers for Highly Compressed Latent Spaces (https://arxiv.org/abs/2501.05442)
Comments:
          Project website: this https URL

- **What's New**: 이 논문은 비디오 토크나이저(video tokenizer)의 성능을 혁신적으로 향상시키기 위한 새로운 방법을 제안합니다. 기존의 토크나이저들이 시간적 압축 비율을 4배 이상으로 확장하는 데 어려움을 겪는 한편, 저자는 저압축 인코더를 통해 비디오의 재구성 품질을 개선할 수 있음을 발견했습니다. 이 연구는 ProMAG라는 새로운 모델을 개발하여, 낮은 압축 모델에서 학습된 정보를 활용하여 높은 압축 블록을 점진적으로 훈련시키는 접근 방식을 취했습니다.

- **Technical Details**: ProMAG 모델은 MagViT-v2 아키텍처를 기반으로 하며, 지속적인 토큰을 활용하여 고해상도 비디오 인코딩을 지원합니다. 저자는 우선적인 4배 시간 압축 모델을 바탕으로 점진적인 방식으로 압축 비율을 8배 및 16배까지 늘리는 기술을 도입했습니다. 이를 통해 다양한 압축 레벨을 다루는 별도의 모델 블록을 학습하여 최적의 재구성 품질을 달성하도록 설계되었습니다.

- **Performance Highlights**: 이 연구에서는 ProMAG 모델이 기존의 고압축 비디오 토크나이저와 비교할 때 비약적으로 향상된 재구성 품질을 기록했다고 보고합니다. 제안된 모델은 16배 압축 비율로도 고품질 재구성을 가능하게 하였으며, 텍스트-비디오 생성에서도 효율성을 극대화하여 4배 압축 모델과 비교할 때 동등하거나 더 높은 품질을 달성했습니다.



### $DPF^*$: improved Depth Potential Function for scale-invariant sulcal depth estimation (https://arxiv.org/abs/2501.05436)
Comments:
          GA and JL contributed equally to this work

- **What's New**: 이 연구는 뇌 크기가 피각의 기하학적 특성에 미치는 영향을 탐구한 최초의 정량적 분석을 제공하며, 독창적인 문제 형식을 기반으로 한 새로운 스케일 불변적 방법을 도입했습니다. 또한, 연구자가 제안한 새로운 피각 깊이를 검증할 수 있는 프레임워크를 제시하고, 1,987명의 대규모 샘플을 기반으로 생물학적 중요성을 입증했습니다.

- **Technical Details**: 피각 깊이(sulcal depth)는 뇌의 기하학적 형상을 이해하는 데 중요한 역할을 하며, 다양한 연구에서 활용되고 있습니다. 기존의 피각 깊이 측정 방법들은 뇌의 전반적인 크기에 따른 변동을 간과하는 경우가 많았으나, 본 연구에서는 그러한 변화를 고려한 측정 방식을 제안하고 있습니다. 다양한 계산적 방법을 통해 피각 깊이를 보다 정확하게 추정할 수 있는 새로운 접근법을 탐구하고 있습니다.

- **Performance Highlights**: 연구팀은 26주에서 성인기까지의 발달 기간을 포함하는 대규모 데이터를 활용하여 피각 깊이의 생물학적 관련성을 입증했습니다. 이를 통해 기존의 연구들보다 더욱 정교하고 일관된 피각 깊이 측정 결과를 제공하게 되어, 향후 기초 및 임상 연구에서 큰 기여를 할 것으로 기대됩니다.



### Zero-1-to-G: Taming Pretrained 2D Diffusion Model for Direct 3D Generation (https://arxiv.org/abs/2501.05427)
- **What's New**: 이번 연구에서는 Zero-1-to-G라는 새로운 접근 방식을 제안하여 사전 훈련된 2D diffusion 모델을 활용하여 빠르고 효율적인 단일 시점 3D 생성이 가능하도록 합니다. Gaussian splats라는 3D 표현을 기반으로 하여 이를 다양한 속성이 인코딩된 다중 시점 이미지로 분해함으로써 기존의 3D 데이터 부족 문제를 해결합니다. 이 모델은 3D 일관성을 유지하는 크로스뷰 및 크로스 속성 주의(attention) 레이어를 도입하여 생성된 splats의 품질을 높이고 있습니다.

- **Technical Details**: Zero-1-to-G는 Gaussian splats를 활용한 첫 번째 직접 이미지-투-3D 생성 모델로, 사전 훈련된 2D diffusion 모델의 풍부한 사전 정보를 효과적으로 사용합니다. Gaussian splats는 다양한 속성을 인코딩한 14채널 이미지를 다수의 3채널 속성 이미지로 분해하여 3D 정보를 유지합니다. 이를 통해 latent diffusion 훈련이 가능해지며, 네트워크의 요소 간 정보 교환을 통해 3D 일관성을 높이는 방법을 제공합니다.

- **Performance Highlights**: 실험 결과, Zero-1-to-G는 합성 및 실제 환경의 데이터셋 모두에서 3D 객체 생성에서 뛰어난 성능을 보였습니다. 전통적인 두 단계 다중 뷰 3D 생성 접근 방식에 비해 구조적 일관성을 높이고 더 높은 품질의 결과를 생성합니다. 본 연구는 고품질 3D 생성을 위한 새로운 접근 방식을 제안함으로써 컴퓨터 비전 및 그래픽 커뮤니티에서 큰 기여를 할 것으로 기대됩니다.



### From Images to Insights: Transforming Brain Cancer Diagnosis with Explainable AI (https://arxiv.org/abs/2501.05426)
Comments:
          Accepted in 2024 27th International Conference on Computer and Information Technology (ICCIT)

- **What's New**: 본 연구는 방글라데시에서 수집된 6,056장의 MRI 이미지를 포함하는 방글라데시 뇌암 MRI 데이터셋을 사용하여 뇌암 분류를 위한 심층 학습 모델을 평가했습니다. 이 데이터셋은 Brain Tumor, Brain Glioma, Brain Menin으로 세 가지 카테고리로 나누어져 있으며, 다양한 의료 기관에서 수집되었습니다. 더불어, DenseNet169 모델이 0.9983의 정확도로 뛰어난 결과를 기록하였고, Explainable AI(XAI) 기법을 통해 모델의 의사결정 과정을 시각화했습니다.

- **Technical Details**: 이 연구에서는 CNN 기반의 여러 모델이 뇌암 진단에 적용되었습니다. 특히 DenseNet 모델이 효율성을 높이기 위해 크로스 레이어 특성 재사용을 통해 성능을 개선했습니다. 또한 GradCAM, GradCAM++, ScoreCAM 및 LayerCAM과 같은 XAI 방법론을 통해 모델의 의사결정 과정이 시각적으로 해석될 수 있도록 했습니다.

- **Performance Highlights**: DenseNet169 모델은 0.9983의 정확도 및 F1 점수를 기록하며, 뇌암 이미지를 분류하는 데 있어 우수한 성능을 보여주었습니다. 이 모델에서 사용한 XAI 기법은 의사결정을 더욱 투명하게 만들어, 의료 전문가들이 진단을 이해하고 신뢰할 수 있도록 도움을 주었습니다. 최종적으로, 이 연구는 방글라데시 뇌암 MRI 데이터셋을 통해 얻은 결과가 조기 진단 기술의 발전에 기여할 것으로 기대하고 있습니다.



### A Novel Pathology Foundation Model by Mayo Clinic, Charit\'e, and Aignostics (https://arxiv.org/abs/2501.05409)
- **What's New**: 이 보고서에서는 RudolfV 접근 방식을 기반으로 하는 새로운 비전 파운데이션 모델을 소개합니다. 이 모델은 Mayo Clinic과 Charité - Universitätsmedizin Berlin에서 수집된 120만 개의 조직병리 전체 슬라이드 이미지로 훈련되었습니다. 결과적으로, 이 모델은 21개의 공공 벤치마크 데이터셋에서 최첨단 성능을 보여주며, 매개변수 수나 데이터셋 크기가 가장 크지 않음에도 불구하고 뛰어난 성과를 거두었습니다.

- **Technical Details**: 모델 훈련에는 490,000개의 사례에서 추출된 120만 개의 디지털 이미지가 사용되었습니다. 특히, H&E, IHC 및 특수 염색 데이터가 포함되어 있으며, 다양한 확대 비율에서 훈련이 진행되었습니다. 이 보고서는 ViT-H/14 구조를 활용하여 632백만 개의 매개변수를 가진 모델을 훈련하기 위해 RudolfV의 알고리즘을 적응시켰습니다.

- **Performance Highlights**: 모델 성능 평가는 선형 프로빙 프로토콜을 통해 수행되었으며, 21개의 벤치마크 데이터셋을 사용하여 다양한 작업에서 비교되었습니다. 평가 결과는 첫 번째 세대 모델과 비교하여 뛰어난 성과를 나타내며, 특히 클래스 토큰 및 평균 토큰을 고려한 평균 정확도에서 높은 점수를 기록했습니다. 이러한 검증 작업은 또한 데이터 재현성과 비교 가능성을 높이는 데 기여하고 있습니다.



### Performance of YOLOv7 in Kitchen Safety While Handling Knif (https://arxiv.org/abs/2501.05399)
- **What's New**: 이 연구는 안전한 주방에서의 칼 사용에 대한 위험을 식별하기 위해 YOLOv7 모델을 활용합니다. 모델은 손가락 배치와 칼날 접촉 같은 위험 요소를 정확히 감지할 수 있습니다. 본 연구는 YOLOv7의 성능을 평가하며, 특히 요리 시 안전 규칙을 준수하는 것의 중요성을 강조합니다.

- **Technical Details**: YOLOv7 모델의 성능은 precision, recall, mAP50, mAP50-95와 같은 메트릭스를 사용하여 평가되었습니다. 연구 결과, YOLOv7은 31 에폭(epoch)에서 mAP50-95 스코어 0.7879, precision 0.9063, recall 0.7503을 기록하며 가장 우수한 성능을 나타냈습니다. 연구는 YOLOv7을 통해 실시간 안전 시스템을 구축하여 칼 사용 시 잘못된 접근을 경고할 수 있도록 목표하고 있습니다.

- **Performance Highlights**: YOLOv7은 주방에서 칼 사용과 관련된 위험 요소를 효과적으로 탐지할 수 있는 잠재력을 보여주었습니다. 특히, 모델의 높은 precision과 recall 값은 주방 안전성을 크게 향상시킬 수 있음을 나타냅니다. 향후 연구는 이 모델을 통해 더욱 향상된 실시간 안전 모니터링 시스템 개발을 목표로 합니다.



### Arc2Avatar: Generating Expressive 3D Avatars from a Single Image via ID Guidanc (https://arxiv.org/abs/2501.05379)
- **What's New**: 이번 논문에서는 단일 이미지를 입력으로 사용하는 인간 얼굴 파운데이션 모델을 안내로 활용하는 Arc2Avatar라는 최초의 SDS 기반 방법을 제안합니다. 이는 3D Gaussian Splatting(3DGS)의 효과성과 대규모 2D 인간 모델의 발전에 영감을 받아 개발되었습니다. Arc2Avatar는 인공 데이터를 통해 다양한 시각에서 인간 머리를 생성하도록 모델을 확장하고, 밀집된 얼굴 메쉬와의 정밀한 대응을 유지하면서 자연스러운 표현 생성을 가능하게 합니다.

- **Technical Details**: 이 방법은 SDS 접근 방식을 수정하여 밀집 샘플링된 FLAME 3DMM과 함께 마스킹된 3DGS 설정을 활용합니다. Arc2Avatar는 ID 가이드를 통해 Arc2Face로부터 정밀한 정보를 얻고, 템플릿 기반 정규화기를 사용하여 최적화 동안 메쉬에 대한 근접성을 유지합니다. 또한, 선택적 SDS 기반 수정 단계를 도입하여 과장된 표현을 정제하고 사실성을 높이는 방법을 제안합니다.

- **Performance Highlights**: 실험 결과, Arc2Avatar는 놀라운 사실성과 정체성 보존을 달성하며, 매우 낮은 가이드를 사용해도 색상 문제를 효과적으로 해결합니다. 이 방법은 3DGS를 통해 생성된 3D 아바타의 사실성과 세부 사항을 유지하면서도 개인의 표현을 생동감 있게 생성할 수 있도록 합니다. Arc2Avatar는 기존 방법들보다 우수한 사실성을 가진 3D 아바타를 생성하는 데 성공했습니다.



### 1-2-1: Renaissance of Single-Network Paradigm for Virtual Try-On (https://arxiv.org/abs/2501.05369)
Comments:
          Project page: this https URL

- **What's New**: 본 논문에서는 기존의 이중 네트워크 접근 방식을 도전하고, 고품질 Virtual Try-On (VTON)을 단일 생성 네트워크를 통해 성취할 수 있는 새로운 방법론인 MN-VTON을 제안합니다. 이 방법은 모달리티별 정규화(Modality-specific Normalization) 전략을 도입하여, 텍스트, 이미지, 비디오 입력을 개별적으로 처리하면서도 동일한 attention 레이어를 효과적으로 공유할 수 있도록 합니다. 이를 통해, 다양한 입력에 맞는 최적의 결과를 도출하며, VTON의 단일 네트워크 패러다임의 부활을 이끌어냅니다.

- **Technical Details**: MN-VTON은 기존 방법들의 세 가지 주요 한계(L1, L2, L3)를 극복하기 위해 설계되었습니다. 첫째, CLIP의 고급 의미적 표현 디자인 한계를 넘어 텍스처 세부사항이 학습될 수 있도록 네트워크 레이어를 학습 가능하도록 했습니다. 둘째, 각 레이어의 특정 특성에 맞춰서 피쳐를 추출하도록 하여, 매우 정밀한 출력 결과를 생성합니다. 마지막으로, 텍스트와 이미지 입력을 다르게 처리하여 텍스처 보존의 최적화를 이끌어냅니다.

- **Performance Highlights**: 시행된 다양한 실험을 통해 MN-VTON은 기존의 이중 네트워크 접근 방식에 비해 이미지 및 비디오 VTON 작업에서 상당히 높은 품질의 결과를 지속적으로 생성할 수 있음을 입증했습니다. 특히, VITONHD, DressCode, VVT, VIVID 데이터셋을 활용해 고화질 및 높은 해상도를 제공하며, 효율적인 대안으로 자리잡을 가능성을 보여주었습니다. 이 연구는 VTON 분야에서 단일 네트워크의 가능성을 재조명하며, 향후 실용적인 응용 프로그램의 발전에 기여할 것입니다.



### CROPS: Model-Agnostic Training-Free Framework for Safe Image Synthesis with Latent Diffusion Models (https://arxiv.org/abs/2501.05359)
- **What's New**: 이번 논문에서는 이미지 생성의 주요 발전을 다루고 있으며, 특히 NSFW(Not Safe For Work) 콘텐츠의 생성을 방지하기 위한 새로운 방어 방법인 CROPS(Circular or RandOm Prompts for Safety)를 제안합니다. 기존의 안전 점검기가 적대적 공격에 취약하다는 연구 결과를 바탕으로 하여, 추가적인 훈련 없이도 공격에 대한 방어가 가능한 방식으로 발전되었습니다. 또한, 이전 모델에 비해 컴퓨팅 자원을 줄일 수 있는 접근법도 발전시켰습니다.

- **Technical Details**: CROPS는 다양한 텍스트 프롬프트와 이미지 입력을 조정하여 생성된 이미지의 비율을 평가함으로써 공격을 탐지하는 방식으로 작동합니다. 이때 모형에 의존하지 않으며, 다양한 확산 모델에 적용이 가능하여 유연성을 제공합니다. CROPS-1은 한 단계 확산 모델을 활용하여 효율적인 NSFW 탐지를 수행하며, 이는 고성능의 검출률을 유지하는 동시에 가벼운 계산량을 자랑합니다.

- **Performance Highlights**: 제안된 방법은 기존 안전 메커니즘과 결합 가능하며, 실제 사용 시 높은 효율성과 성능 우수성을 입증하였습니다. 여러 이미지를 생성하고 평가하여 공격의 존재 여부를 검증하는 과정에서, 공격 성공률이 낮아짐을 확인할 수 있었습니다. 이 연구는 확산 모델에 대한 실질적이고 필요로 하는 방어 메커니즘을 제공함으로써, AI 기반 콘텐츠 생성의 안전성을 높이는 데 기여할 것입니다.



### JAQ: Joint Efficient Architecture Design and Low-Bit Quantization with Hardware-Software Co-Exploration (https://arxiv.org/abs/2501.05339)
Comments:
          Accepted by AAAI 2025

- **What's New**: 본 논문에서는 JAQ 프레임워크를 제안하여 신경망 아키텍처, 정량화 비트 너비, 하드웨어 가속기를 주기적으로 최적화하는 방법을 소개합니다. JAQ는 메모리 사용량과 하드웨어 탐색 시간을 최소화하여 자원 제약이 있는 엣지 디바이스에서의 모델 배포를 위한 최적의 밸런스를 달성하는 데 중점을 둡니다. 특히, 채널별 희소 정량화(Channel-wise Sparse Quantization) 기법을 통해 메모리 오버헤드를 줄이고, BatchTile 방법으로 컴파일러 매핑 전략을 효율적으로 통합하는 접근 방식을 제안합니다.

- **Technical Details**: JAQ 프레임워크는 저비트 혼합정밀도 비트 할당과 가속기 아키텍처의 효율적인 공동 탐색을 목표로 합니다. 이 연구는 높은 정밀도의 정량화만 지원하는 기존 방법들과 차별화되는 부분으로, 메모리 폭발 문제를 완화하기 위해 핵심 활성화 채널(activations channels) 선택을 통해 메모리 사용을 최대 5배까지 줄일 수 있습니다. 또한, BatchTile 접근법은 다양한 타일 크기를 배치로 인코딩하여 최적의 타일링 전략을 동시에 결정할 수 있게 합니다.

- **Performance Highlights**: JAQ 프레임워크는 기존 기법에 비해 ImageNet에서 약 7% 더 높은 Top-1 정확도를 달성했으며, 각 반복에 대한 하드웨어 탐색 시간을 0.15초로 단축시켰습니다. 이러한 결과는 JAQ가 현재 최첨단 성능을 초월하는 가능성을 보여줍니다. 이 연구는 소프트웨어-하드웨어 공동 설계를 위한 새로운 가능성을 열어줍니다.



### Comparison Study: Glacier Calving Front Delineation in Synthetic Aperture Radar Images With Deep Learning (https://arxiv.org/abs/2501.05281)
- **What's New**: 이번 연구에서는 해양에서 끝나는 빙하의 쇄빙 전선(calving front) 위치 변화를 추적하기 위해 Deep Learning (DL) 시스템을 적용한 최초의 평가를 진행했습니다. Synthetic Aperture Radar (SAR) 이미지를 활용하여 독립적으로 모니터링할 수 있는 방법을 제안하고 있습니다. 이러한 접근 방식은 날씨나 조명에 구애받지 않는 대규모 모니터링을 가능하게 합니다.

- **Technical Details**: 연구는 10명의 주석자(annotator)와 함께 진행되어, 가장 성능이 뛰어난 DL 시스템이 인간 성능과 어떻게 차이가 나는지를 조사하였습니다. DL 모델의 평균 오차는 221 m로 나타났고, 인간 주석자의 평균 오차는 38 m였습니다. 이 결과는 현재의 DL 시스템이 인간 성능을 아직 따라잡지 못하고 있음을 보여주며, 완전 자동화를 위한 추가 연구의 필요성을 강조합니다.

- **Performance Highlights**: 연구에서 Vision Transformers, 기초 모델(foundational models), 그리고 더 많은 정보를 포함하고 처리하는 전략이 향후 연구 방향으로 제시되었습니다. 이러한 개선점을 통해 얼음 붕괴의 자동 모니터링이 가능해질 것입니다.



### Solving the Catastrophic Forgetting Problem in Generalized Category Discovery (https://arxiv.org/abs/2501.05272)
Comments:
          Accepted by CVPR 2024

- **What's New**: 이번 연구에서는 Generalized Category Discovery (GCD)의 새로운 접근 방식인 LegoGCD를 제안합니다. LegoGCD는 기존의 SimGCD 방법의 성능을 개선하고, 새로운 카테고리를 학습하면서도 기존 카테고리의 성능을 유지하도록 설계되었습니다. 이 방법은 Local Entropy Regularization (LER)과 Dual-views Kullback-Leibler divergence constraint (DKL)라는 두 가지 기술을 도입하여, 알려진 카테고리를 언급하며 새로운 카테고리를 학습할 수 있는 능력을 강화합니다.

- **Technical Details**: LegoGCD는 GCD 방식의 연속적인 발전을 보여줍니다. LER은 알려진 클래스 샘플의 분포를 최적화하여, 새로운 클래스를 학습하면서 기존 지식을 보존하도록 돕습니다. DKL은 쌍으로 변형된 샘플 간의 Kullback-Leibler divergence를 활용하여, 동일한 이미지로부터 추출된 두 뷰의 예측 분포가 일치하도록 유도하여, 예측의 신뢰성을 높입니다.

- **Performance Highlights**: 철저한 실험을 통해 LegoGCD는 기존의 방법인 SimGCD보다 뛰어난 성과를 입증했습니다. 특히, CUB 데이터셋에서 알려진 클래스와 새로운 클래스 각각에 대해 7.74% 및 2.51%의 정확도 향상을 이끌어냈습니다. LegoGCD는 SimGCD에 쉽게 통합될 수 있으며, 추가 매개변수 없이도 성능 개선을 실현했습니다.



### CellViT++: Energy-Efficient and Adaptive Cell Segmentation and Classification Using Foundation Models (https://arxiv.org/abs/2501.05269)
- **What's New**: 이번 논문에서는 디지털 병리학에서 세포를 식별하고 분할하는 데 있어 기존 방법의 한계를 극복하기 위해 $	ext{CellViT}^{	ext{++}}$이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 비전 트랜스포머(Vision Transformers)를 활용하여 깊은 세포 특성과 분할 마스크를 동시에 계산하며, 최소한의 데이터로 훈련할 수 있어 탄소 발자국을 크게 줄입니다. 또한, $	ext{CellViT}^{	ext{++}}$는 면역형광 염색(immunofluorescence stainings)을 이용하여 병리학자 주석 없이 훈련 데이터 세트를 생성할 수 있는 능력을 입증합니다.

- **Technical Details**: 프레임워크의 핵심 구성 요소는 PanNuke 데이터 세트에서 세분화 모델을 사전 훈련(pretraining)하는 것입니다. 이 데이터 세트는 19개의 조직 타입에 걸쳐 190,000개의 세포가 광범위하게 주석(annotation) 처리되어 있습니다. 다양한 세포 분류 모듈을 테스트하기 위해 CellViT 시리즈의 변형 모델들이 평가되었으며, 이들 모두는 기준 모델에 비해 평균 7.45% 이상의 성능 향상을 보였습니다. 실험은 F1-스코어(mF1)와 평균 팬옵틱 품질(mPQ)이라는 지표를 사용하여 진행되었습니다.

- **Performance Highlights**: 세분화 성능에서 Segment Anything Model (CellViTSAM-H)이 가장 높은 성능을 기록하며, SOTA(results) 성능을 보여주었습니다. Li et al. (2024) 연구에서도 CellViT과 종양 세분화 모델 조합을 통해 0.7243의 mF1-score를 달성하여 우수한 결과를 보고했습니다. 이번 연구가 제안하는 $	ext{CellViT}^{	ext{++}}$은 새로운 임상 및 세포 유형에 대해 뛰어난 제로샷(segmentation) 능력을 보여주며, 고품질 훈련 데이터 세트를 효율적으로 생성할 수 있는 잠재력을 가집니다.



### Patch-GAN Transfer Learning with Reconstructive Models for Cloud Remova (https://arxiv.org/abs/2501.05265)
- **What's New**: 이 논문에서는 원거리 감지(remote sensing) 이미지에서 구름 제거(cloud removal)를 위한 생성적 적대 신경망(GAN) 프레임워크를 활용한 새로운 심층 전이 학습(deep transfer learning) 접근 방식을 제안합니다. 특히, Masked Autoencoder (MAE)를 기반으로 한 이미지 재구성 모델을 활용하여 기존의 구름으로 가려진 지역을 효과적으로 재구성할 수 있는 가능성을 탐색합니다. 이러한 방법은 전통적인 GAN 기반 방법에 비해 구름 제거 성능을 획기적으로 향상시킵니다.

- **Technical Details**: 제안한 방법은 GAN 구조를 모델로 사용하며, 생성기(generator)와 판별기(discriminator)로 구성됩니다. 제안하는 생성기는 ViT-large 인코더와 인코더 기반의 ViT 디코더를 포함하고 있으며, 먼저 MAE를 통해 이미지 재구성 작업에 대해 사전 훈련됩니다. 생성기는 입력 이미지를 224×224로 무작위로 자르고, 이를 다시 16×16 픽셀 크기의 196개의 패치로 나눕니다. 판별기는 각 패치를 실제 또는 가짜인지 평가하여 구름 없는 이미지를 생성할 수 있도록 돕습니다.

- **Performance Highlights**: 제안한 재구성 전이 학습 접근 방식은 다른 GAN 기반 방법들에 비해 구름 제거 성능에서 상당한 향상을 보여줍니다. 비록 몇몇 최신 구름 제거 기술들과의 직접적인 비교는 훈련/테스트 데이터 분할에 대한 불확실성 때문에 제한적이나, 제안된 모델은 사용 가능한 벤치마크 기반으로 경쟁력 있는 결과를 달성했습니다. 이 연구는 구름 제거 작업에서 MAE와 GAN의 결합이 유망한 결과를 가져올 수 있음을 시사합니다.



### Towards Balanced Continual Multi-Modal Learning in Human Pose Estimation (https://arxiv.org/abs/2501.05264)
- **What's New**: 이번 논문에서는 3D 인간 자세 추정(3D HPE)을 위해 RGB, LiDAR, mmWave, WiFi를 활용한 균형 잡힌 지속적인 다중 모달 학습 방법을 제안합니다. 우리는 Shapley value 기반 기여 알고리즘을 통해 각 모달리티의 기여도를 정량화하고, 이를 통해 발생하는 모달리티 불균형 문제를 해결하고자 합니다. 또한, 기존 데이터의 잡음 영향을 줄이기 위한 새로운 디노이징 지속적 학습 접근 방식을 개발했습니다.

- **Technical Details**: 제안하는 방법은 새로운 잡음 식별 및 분리 모듈(NIS)을 통해 모달리티 기여 점수를 모니터링하고 잡음이 감지된 경우 가장 잡음이 많은 데이터를 데이터셋에서 분리합니다. 이를 통해 잡음이 훈련 및 카타스트로픽 포겟팅에 미치는 영향을 완화할 수 있습니다. 또한, 적응형 EWC 메커니즘을 통해 중요한 정보가 손실되지 않도록 합니다.

- **Performance Highlights**: 우리는 MM-Fi라는 다중 모달 데이터셋에서 광범위한 실험을 수행하였으며, 제안한 접근 방식이 복잡한 시나리오에서 3D 자세 추정 성능을 향상시키고 카타스트로픽 포겟팅을 완화한다는 것을 입증했습니다. 이러한 결과는 다중 모달 학습에서의 균형 맞추기를 위한 첫 번째 시도라는 점에서 중요한 의미를 갖습니다.



### Domain-Incremental Semantic Segmentation for Autonomous Driving under Adverse Driving Conditions (https://arxiv.org/abs/2501.05246)
Comments:
          Accepted at ICPRAM 2025

- **What's New**: 이번 연구에서는 자동운전 시스템에서 비정상적 조건 하의 성능 저하 문제를 해결하기 위해 Progressive Semantic Segmentation (PSS)라는 새로운 방법론을 제안합니다. PSS는 도메인-증가 학습 (domain-incremental learning)에 기반한 접근으로, 비정형 환경에서의 다양한 조건을 지속적으로 학습할 수 있는 구조입니다. 기존의 방법들과 달리 PSS는 태스크-전문성이 아닌 도메인 직접 추론을 통해 최적화된 세분화 모델을 선택합니다.

- **Technical Details**: PSS는 구조 기반의 방법으로, 도메인-특화(segmentation) 모델을 동적으로 확대하여 각 도메인에 적합한 모델을 선택합니다. 이는 컨볼루셔널 오토인코더 (convolutional autoencoders)를 활용하여 도메인을 추론하고, 그 결과에 따라 세분화 모델을 결정하는 방식으로 진행됩니다. 이 접근법은 여러 데이터셋을 통해 다양한 비정상적 driving conditions에서 세부 평가됩니다.

- **Performance Highlights**: PSS는 여러 데이터셋을 사용하여 비정상적 주행 조건에서 세분화 작업의 성능을 평가하였으며, 이전 도메인에 대한 성능을 유지하면서 새로운 도메인에 적응하는 능력을 입증하였습니다. 연구 결과, PSS는 지속적인 학습 설정에서 학습된 정보를 효과적으로 유지함으로써 catastrophic forgetting을 피할 수 있음을 보여줍니다. 또한, 우리는 PSS가 객체 탐지와 같은 다른 컴퓨터 비전 작업에도 확장 가능함을 시연하였습니다.



### Scaffold-SLAM: Structured 3D Gaussians for Simultaneous Localization and Photorealistic Mapping (https://arxiv.org/abs/2501.05242)
Comments:
          12 pages, 6 figures

- **What's New**: 이번 논문에서는 Scaffold-SLAM을 제안하여 모노컬, 스테레오 및 RGB-D 카메라에서 동시 위치 추적과 고품질 포토리얼 매핑을 가능하게 합니다. 주요 혁신으로는 사진 모습을 기반으로 한 임베딩(Appearance-from-Motion embedding)과 주파수 정규화 피라미드(frequency regularization pyramid)가 있습니다. 이를 통해 Gaussians가 복잡한 장면 세부정보를 더 효과적으로 모델링할 수 있게 됩니다. 실험 결과, Scaffold-SLAM이 기존 최신 기법 대비 포토리얼 매핑 품질이 크게 향상되었음을 보여줍니다.

- **Technical Details**: Scaffold-SLAM은 기존 Photo-SLAM의 프레임워크를 공유하나, 두 가지 주요 혁신을 도입합니다. 첫 번째는 Appearance-from-Motion embedding으로, 다양한 카메라 자세에서의 이미지 모습 변화를 효율적으로 모델링할 수 있습니다. 두 번째는 주파수 정규화 피라미드로, 여러 주파수 스케일에서의 이미지 렌더링 품질을 제어하여 Gaussians가 복잡한 지역으로 성장하도록 유도합니다. 이러한 접근법을 통해 모노컬, 스테레오 및 RGB-D 카메라에서 고품질의 포토리얼 매핑을 실현합니다.

- **Performance Highlights**: Scaffold-SLAM의 성능은 다양한 데이터셋에서 평가되었으며, 모노컬 카메라의 경우 TUM RGB-D 데이터셋에서 PSNR이 16.76% 향상된 것으로 나타났습니다. 모든 카메라 유형에서 포토리얼 매핑 품질이 크게 개선되었고, 경쟁력 있는 추적 정확도를 유지하고 있습니다. 이러한 결과는 Scaffold-SLAM이 SLAM 기술에서 획기적인 발전임을 보여줍니다.



### Contrast-Free Myocardial Scar Segmentation in Cine MRI using Motion and Texture Fusion (https://arxiv.org/abs/2501.05241)
Comments:
          5 pages, 2figs, 2tables

- **What's New**: 이 논문에서는 심장 동작을 동반한 텍스처 정보를 결합하여 심근 경색 후 좌심실의 흉터를 분할하는 새로운 프레임워크인 MTI-MyoScarSeg 모델을 제안합니다. 기존의 지연 가돌리늄 증강 자기 공명 영상(LGE MRI) 대신 대비 없는 시네 MRI를 사용하여 흉터 검출을 수행할 수 있는 가능성을 보여주고 있습니다. 이 방법은 심장 주기 내에서 발생하는 동작 정보를 추출하여 정량적 분석 없이도 심근 흉터를 명확하게 구분할 수 있게 합니다.

- **Technical Details**: MTI-MyoScarSeg 모델은 심장 동작 추출 네트워크와 심근 및 흉터 분할 네트워크로 구성되어 있으며, 이 두 가지 네트워크는 심장 주기에 걸쳐 심장 동작을 추정하고 이를 이미지 시퀀스와 통합합니다. 동작 추출 네트워크는 U-Net 아키텍처를 기반으로 하며, 각 프레임의 변위를 추정하여 정적 ED 프레임에 맞춰 변형합니다. 또한, 이 모델은 각 프레임간의 이동과 변화를 효과적으로 파악하여 정밀한 분할을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, 제안된 MTI-MyoScarSeg 모델은 비대비 시네 이미지에서도 LGE MRI와 유사한 정확도로 흉터를 분할할 수 있는 능력을 보였습니다. 이는 심장 흉터 검출에 있어 대비 증강 기술에 대한 대안으로서의 가능성을 시사합니다. 또한, ED 단계를 고정 기준 프레임으로 사용한 동작 추출 전략이 효과적임을 입증하며, 기존 방법들과 비교하여 좋은 성능을 발휘합니다.



### FOCUS: Towards Universal Foreground Segmentation (https://arxiv.org/abs/2501.05238)
- **What's New**: 이 연구에서는 여러 종류의 전경 구분 작업을 수행할 수 있는 새로운 프레임워크 FOCUS, 즉 Foreground ObjeCts Universal Segmentation을 소개합니다. 기존 모델들이 각 전경 작업에 대해 특정한 구조로 설계되었던 것과 달리, FOCUS는 전경과 배경 간의 관계를 강조하며, 이에 따른 배경의 중요성을 인식합니다. 이를 통해 다양한 데이터셋에서 전경에 대한 통합된 접근 방식을 제공합니다.

- **Technical Details**: FOCUS는 멀티 스케일 세멘틱 네트워크를 사용하여 객체의 엣지 정보를 활용해 이미지 특징을 강화합니다. 특히, 새로운 증류 방법과 대조 학습(constractive learning) 전략을 통합하여 예측 마스크를 다중 모달(feature space)에서 정교하게 다듬습니다. 이 방법은 배경과 전경을 독립적인 두 작업으로 간주하며, 예측 단계에서 두 확률 맵을 결합하여 Mask of Interest(MoI)의 경계를 정의합니다.

- **Performance Highlights**: FOCUS는 다양한 데이터셋에 대한 실험 결과, 대부분의 주요 지표에서 최신 작업 특화 모델들보다 뛰어난 성능을 보여주었습니다. 연구의 결과는 FOCUS가 전경 세그멘테이션 작업에서의 국가-최고 성능을 달성하는 데 기여하고 있다는 것을 입증합니다. 특히 13개의 데이터셋에서 5가지 전경 세그멘테이션 작업을 평가하여 강력한 결과를 보였습니다.



### Automated external cervical resorption segmentation in cone-beam CT using local texture features (https://arxiv.org/abs/2501.05236)
Comments:
          4 pages, 3 figures, 1 table

- **What's New**: 이번 논문에서는 외부 경부 흡수(External Cervical Resorption, ECR) 병변을 자동으로 세분화하는 새로운 방법을 제시합니다. 이는 대칭적인 이진 분류를 기반으로 하며, 고해상도 콘빔 컴퓨터 단층촬영(Cone-Beam Computed Tomography, CBCT) 데이터를 활용합니다. 텍스처 기능을 통해 ECR의 진행 상황을 보다 정확하게 모니터링하고 예측할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 제안된 알고리즘은 로컬로 추출된 볼륨 텍스처 기능의 이진 분류에 기반하여 ECR 병변을 분할합니다. 3D Slicer에서 "DentalSegmentator" 확장을 사용하여 치아의 분할을 수행하며, 고유한 텍스처 기반 기능을 추출하기 위해 통계적 이미지 강도 속성을 활용합니다. 지원 벡터 머신(Support Vector Machines, SVM)을 사용하여 텍스처 기능을 분류하고, 후속 과정에서는 이진 침식 및 팽창을 통해 최종 세분화를 개선합니다.

- **Performance Highlights**: 이 연구에서는 두 환자의 데이터를 사용하여 성능을 평가했으며, Dice Score Coefficient (DSC)를 통해 세분화의 정확성을 확인했습니다. 특정 스캔의 경우 DSC가 0.7을 초과하는 성과를 보였으며, 제안된 방법이 ECR 병변을 적절히 예측할 수 있음을 입증했습니다. 텍스처 패턴 분석을 통해 병변 내 석회화 조직의 존재를 확인하였으며, 향후 ECR의 진행 및 치료 결정을 지원하는 데 기여할 수 있을 것으로 예상합니다.



### Harnessing Large Language and Vision-Language Models for Robust Out-of-Distribution Detection (https://arxiv.org/abs/2501.05228)
Comments:
          9 pages, 4 figures

- **What's New**: 이번 연구에서는 Out-of-Distribution (OOD) 데이터 탐지 개선을 위해 CLIP와 같은 강력한 Vision-Language Models (VLMs) 및 Large Language Models (LLMs)를 효과적으로 활용하는 혁신적인 방안을 제안합니다. 이전 연구들은 Far-OOD 성능 향상에 주로 초점을 맞췄으나, 저희 연구에서는 Near-OOD 상황에서도 효과적인 방법을 개발했습니다. LLM을 통해 ID 레이블의 슈퍼클래스와 배경 설명을 생성하고, CLIP을 사용하여 특징을 추출한 후, 이러한 정보를 바탕으로 더 적합한 negative 레이블을 선택하는 전략을 적용했습니다.

- **Technical Details**: 선택한 후보 레이블 세트로부터 ID 레이블의 핵심 의미적 특징을 분리하기 위해, 배경 특징을 슈퍼클래스 특징에서 빼는 방식으로 진행됩니다. 이 과정은 OOD 데이터의 탐지 성능을 향상시키는 데 필수적입니다. 또한, 저희는 few-shot prompt tuning (PT) 및 visual prompt tuning (VPT)을 통해 제안된 프레임워크를 더 잘 조정하여 OOD 탐지의 목표 배포와 일치시킵니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 여러 벤치마크에서 기존의 최첨단 방법들을 지속적으로 초과 달성하며, ImageNet-1K OOD 벤치마크에서 평균 AUROC가 97.67%에 달하며, OpenOOD V1.5 풀스펙트럼 벤치마크에서 평균 AUROC는 83.04%에 도달했습니다. 또한, 저희 방법은 다양한 도메인에서 변동 요인 변화에 강한 내성을 보이며, 실제 환경에서의 효과를 더욱 강조합니다.



### Light Transport-aware Diffusion Posterior Sampling for Single-View Reconstruction of 3D Volumes (https://arxiv.org/abs/2501.05226)
- **What's New**: 이 논문에서는 구름과 같이 다수의 빛 산란 효과가 존재하는 체적 필드의 단일 시점 복원 기술을 소개합니다. 다양한 실험을 통해 이전에는 도달할 수 없었던 질량의 단일 시점 복원을 이루어냈습니다. 특히, 1,000개의 합성 시뮬레이션 체적 밀도 필드를 포함한 새로운 벤치마크 데이터셋을 활용하여 훈련된 조건 없는 확산 모델을 통해 이러한 기술이 이루어졌습니다.

- **Technical Details**: 이 연구에서는 보간 과정에서 물리 기반 미분 가능 볼륨 렌더러(Physically-based Differentiable Volume Renderer, PDVR)를 사용하여 복원하는 과정에서의 경량 수송에 대한 그라디언트를 제공합니다. 이 접근 방식은 기존의 NeRF 방식과 대조적이며, 관측 데이터와 잘 일치하는 복원을 위해 저수준 매개변수를 저항하는 분산 사전(diffusion prior)을 이용합니다. 또한, 새롭게 제안된 매개변수 확산 후방 샘플링(Diffusion Posterior Sampling, DPS) 기법을 통해 제작된 물체의 형상 중심의 사전 분포(not artistic) 데이터를 결합하여 시뮬레이션을 수행합니다.

- **Performance Highlights**: 제안된 방법은 단일 및 다중 뷰 복원, 체적 초해상도(volume super-resolution) 등 다양한 작업에서 큰 가능성을 보여줍니다. 구름과 같은 체적 필드를 높은 품질로 단일 시점에서 복원함으로써, 이전 기술들과 비교하여 시각적 일관성(spatial consistency)을 유지하면서 묘사했습니다. 이 연구는 확산 모델과 미분 가능 볼륨 렌더러의 통합이 3D 모델 복원에 있어 새로운 전환점을 제시할 수 있음을 입증합니다.



### MHAFF: Multi-Head Attention Feature Fusion of CNN and Transformer for Cattle Identification (https://arxiv.org/abs/2501.05209)
Comments:
          30 pages

- **What's New**: 이번 연구에서는 소의 인식을 위한 새로운 방법인 Multi-Head Attention Feature Fusion (MHAFF)을 소개합니다. 이는 CNN(Convolutional Neural Networks)과 transformer의 장점을 융합하여 기존의 addition과 concatenation 방법보다 더 나은 성능을 보여줍니다. MHAFF는 다양한 유형의 융합 특징 간의 관계를 포착하면서도 개별 특징의 고유성을 유지하는 것을 목표로 합니다.

- **Technical Details**: MHAFF는 CNN의 특징 및 transformer의 Query, Key, Value 요소를 활용하여 새로운 맥락 기반 특징 융합 방법을 제안합니다. 이 접근법은 소의 주둥이 이미지 기반 인식에 처음으로 적용되었습니다. CNN은 주로 지역적 특징(Local Features)을 추출하는 데 최적화되어 있는 반면, transformer는 전역 정보(Global Information)를 처리하는 데 적합하여, 두 네트워크의 장점을 결합하여 정확도를 향상시킵니다.

- **Performance Highlights**: 실험 결과, MHAFF는 두 개의 공개 소 데이터셋 모두에서 99.88%와 99.52%의 최적 정확도로 기존의 cattle identification 방법을 초월한 성능을 보여주었습니다. 이는 새로운 특징 융합 방법이 단순한 addition과 concatenation을 넘어 인식 능력의 향상을 가져오는데 기여할 수 있음을 시사합니다.



### Discovering Hidden Visual Concepts Beyond Linguistic Input in Infant Learning (https://arxiv.org/abs/2501.05205)
Comments:
          12 pages, 11 figures

- **What's New**: 이 연구에서는 아기들이 언어 입력을 습득하기 전에 시각적 이해를 빠르게 발전시키는 과정을 모델링하여, 아기들의 학습 과정을 모방한 컴퓨터 비전 모델이 언어 데이터 범위를 넘어서는 더 넓은 시각적 개념을 개발할 수 있는지를 탐구합니다. CVCL(Child’s View for Contrastive Learning) 모델을 사용하여, 아기들이 부모의 말과 연관된 시각적인 프레임을 이해하도록 훈련되었습니다. 이 연구는 NeuronClassifier라는 훈련이 필요 없는 새로운 프레임워크를 소개하여, 모델의 내부 표현에서 숨겨진 시각적 개념을 발견했습니다.

- **Technical Details**: CVCL은 6개월에서 25개월 사이의 아기의 영상 데이터를 이용해 훈련된 모델로, 부모가 말한 내용을 아기의 시각적 경험과 결합하여 이해합니다. 이를 통해 저자들은 아기의 시각적 개념이 언어적 입력을 초월해 발전할 수 있음을 제안합니다. 특히, 이 모델의 내부 표현 분석을 통해 'neuron labeling' 기법을 이용해 특정 뉴런들이 어떻게 시각적 개념을 형성하는지를 연구했습니다.

- **Performance Highlights**: CVCL 모델은 CLIP 및 ImageNet 모델과 비교했을 때 여전히 인식 능력이 제한적이지만, 숨겨진 시각적 개념을 통해 언어 훈련 자료에 명시되지 않은 고급 개념을 분류하는 데 성공했습니다. 발견된 'out-of-vocabulary' 단어들은 아기들이 배우는 단어 보다는 높은 인지적 수준을 나타내며, 이는 아기들이 시각적 개념을 언어적 이해보다 먼저 개발한다는 인지 과학 연구와 일치합니다.



### HipyrNet: Hypernet-Guided Feature Pyramid network for mixed-exposure correction (https://arxiv.org/abs/2501.05195)
- **What's New**: 최근 혼합 노출 이미지 향상을 위한 이미지 번역에서 딥 러닝 알고리즘의 변신 가능성이 입증되었습니다. 그러나 극단적인 노출 변동을 처리하는 것은 여전히 큰 도전 과제로 남아 있으며, 이러한 작업을 효과적으로 수행하지 못하는 기존 방법들의 한계를 보여주고 있습니다. 이 연구에서는 Laplacian Pyramid 기반 구조 내에 HyperNetwork를 통합한 HipyrNet이라는 새로운 접근 방식을 제안합니다.

- **Technical Details**: HipyrNet은 HyperNetwork를 통해 이미지의 노출 변동에 적응할 수 있도록 설계되었습니다. HyperNetwork는 다른 네트워크의 가중치를 동적으로 생성할 수 있어 실시간으로 변화를 줄 수 있는 특징이 있습니다. 이 모델에서는 입력 이미지의 특성에 따라 최적의 Feature Pyramid 분해를 위한 커널을 예측하고 이를 통해 각 이미지에 맞춘 최적화된 분해 과정을 구현합니다.

- **Performance Highlights**: 많은 실험을 통해 HipyrNet이 기존의 방법들을 초월함을 입증했으며, 특히 극단적인 노출 변동 상황에서 우수한 성능을 나타냈습니다. 정성적(qualitative) 및 정량적(quantitative) 평가 모두에서 우수한 결과를 달성하여 혼합 노출 이미지 향상에 대한 새로운 기준을 세웠습니다. 이를 통해 향후 적응형 이미지 번역 연구에 대한 새로운 길을 열어주는 연구 성과를 보여주고 있습니다.



### Compression with Global Guidance: Towards Training-free High-Resolution MLLMs Acceleration (https://arxiv.org/abs/2501.05179)
Comments:
          Our code is released at \url{this https URL}

- **What's New**: 이 논문에서는 고해상도 다중모드 대형 언어 모델(MLLMs)을 위한 새로운 토큰 압축 기법인 GlobalCom$^2$를 제안합니다. 이 방법은 썸네일(thumbnail)과 여러 개의 크롭(crop)을 동시에 처리하는 MLLMs에 최적화되어 있어, 토큰 압축 과정에서 썸네일의 정보를 '커맨더(commander)'로 활용합니다. 이를 통해 중복 토큰을 제거하고 중요한 로컬 세부 정보를 최대한 보존할 수 있는 방식으로 설계되었습니다.

- **Technical Details**: GlobalCom$^2$는 두 개의 단계로 나누어진 압축 과정을 통해 작동합니다. 첫 번째 단계에서는 썸네일 정보를 이용해 각 크롭의 중요성을 평가하고, 이에 따라 각 크롭에 다른 토큰 보존 비율을 할당합니다. 두 번째 단계에서는 썸네일과 크롭 모두에 대해 토큰 압축을 수행하며, 크롭의 압축 과정은 썸네일 정보를 통해 결정되어 시각적 세부 정보의 보존을 극대화합니다.

- **Performance Highlights**: 10개의 다중모드 이해 기준을 통해 실시한 실험 결과, GlobalCom$^2$는 기존의 훈련이 필요 없는 방법들보다 일관되게 우수한 성능을 보여주었습니다. 특히, 90%의 시각적 토큰이 제거되는 극단적인 경우에도 원래 성능의 90% 이상을 유지할 수 있었습니다. 로컬 세부 정보에 대한 깊은 이해가 요구되는 작업에서도 GlobalCom$^2$는 다른 방법들보다 우위를 나타내고 있습니다.



### FaceMe: Robust Blind Face Restoration with Personal Identification (https://arxiv.org/abs/2501.05177)
Comments:
          To appear at AAAI 2025

- **What's New**: FaceMe라는 개인화된 얼굴 복원 방법을 제안합니다. 이 방법은 단일 또는 몇 개의 참조 이미지를 기반으로 하여, 고품질이며 개인의 정체성을 일관되게 유지하는 얼굴 이미지를 복원합니다. 기존의 많은 방법들은 정체성 일관성을 유지하는데 실패했으나, FaceMe는 수정 없이 다양한 참조 이미지를 사용할 수 있도록 설계되었습니다.

- **Technical Details**: FaceMe는 확산 모델(difussion model)을 기반으로 한 개인화된 얼굴 복원 모델입니다. 이 방법은 정체성 인코더(identity encoder)를 사용하여 정체성과 관련된 기능을 추출하고, 이를 결합하여 훈련하는 동안 정체성과 무관한 기능의 영향을 최소화합니다. 또한, 두 단계의 훈련 절차를 통해 저화질 입력과 참조 이미지 간의 종속성을 균형 있게 조절합니다.

- **Performance Highlights**: 실험 결과에 따르면 FaceMe는 고품질의 얼굴 이미지를 복원하면서도 정체성 일관성을 유지하는 데 뛰어난 성능을 보여줍니다. 특히, 정체성을 변경할 경우에도 추가적인 수정이 필요하지 않아 실용적인 응용 가능성이 높습니다. 이로 인해 FaceMe는 개인화된 얼굴 복원 작업에 있어서 새로운 기준을 제시하고 있습니다.



### A Systematic Literature Review on Deep Learning-based Depth Estimation in Computer Vision (https://arxiv.org/abs/2501.05147)
- **What's New**: 이 논문은 Depth Estimation (DE) 기술에 대한 체계적인 문헌 리뷰(SLR)를 제공합니다. 최근 딥러닝(Deep Learning) 기반 방법이 DE에 사용되면서, 기존의 전통적인 기법과 비교할 때 이점이 두드러집니다. 특이한 점은 기존의 리뷰가 대부분 단안(Monocular) 또는 입체(Stereo) 기술에 국한되어 있었으며, 포괄적인 DE 분석이 부족했다는 것입니다.

- **Technical Details**: 연구에서는 1284개의 관련 출판물을 선별한 후, 128개의 논문을 정제하여 59개의 고품질 핵심 연구로 축소하였습니다. 이들 연구를 통해 다양한 DE 방식인 단안, 입체, 다중 시점(Multi-view)에 대한 딥러닝 모델을 개발하였습니다. 20개의 공개 데이터셋이 DE 모델의 교육, 테스트 및 평가에 사용되었으며, 가장 많이 사용된 데이터셋은 KITTI, NYU Depth V2 및 Make 3D입니다.

- **Performance Highlights**: DE의 성능을 평가하기 위해 29개의 평가 메트릭스가 사용되었습니다. 35개의 기본 모델(Base Model)이 보고되었으며, 가장 많이 사용된 상위 5개 모델은 ResNet-50, ResNet-18, ResNet-101, U-Net 및 VGG-16입니다. 주요 연구에서 가장 큰 도전으로는 정답 데이터(Ground Truth Data)의 부족이 지적되었습니다.



### CorrDiff: Adaptive Delay-aware Detector with Temporal Cue Inputs for Real-time Object Detection (https://arxiv.org/abs/2501.05132)
Comments:
          Submitted to IEEE JSAC Special Issue: Intelligent Communications for Real-Time Computer Vision (Comm4CV)

- **What's New**: 본 논문은 CorrDiff라는 새로운 실시간 스트리밍 인식 방법을 제안합니다. 이 방법은 실시간 객체 탐지 시스템의 지연 문제를 해결하며, 적응형 지연 인식 탐지기를 통해 객체의 위치를 다수의 미래 프레임에 대해 예측할 수 있습니다. 이 모델은 통신 및 계산 지연을 효과적으로 보상하며, 실시간 처리 요구에 맞춰 모든 유형의 장비에서 뛰어난 성능을 발휘합니다.

- **Technical Details**: CorrDiff는 런타임에서 추정된 시간 정보를 활용하여 여러 개의 과거 프레임을 입력으로 받아 여러 개의 미래 프레임에 대한 예측을 동시에 생성합니다. 이를 통해 동적인 지연 상황에 적응할 수 있으며, 시뮬레이션 환경에서 발생할 수 있는 다양한 통신-계산 지연을 효과적으로 처리합니다. 이 모델은 기존의 선행 탐지기 대비 더 높은 정확성을 보여줍니다.

- **Performance Highlights**: CorrDiff는 Tesla V100에서 RTX 2080Ti까지 다양한 GPU에서 강력한 성능을 발휘하여 각 플랫폼에서 최고 수준의 인식 정확도를 달성합니다. 대부분의 최신 기법들이 덜 강력한 장치에서 한 프레임 내에서의 연산을 완료하는 데 어려움을 겪는 반면, CorrDiff는 모든 장비에서 실시간 처리 요구를 충족합니다. 이러한 성능은 자율주행과 같은 실제 응용에서의 안전성과 신뢰성을 크게 향상시킬 수 있는 잠재력을 가지고 있습니다.



### 3DIS-FLUX: simple and efficient multi-instance generation with DiT rendering (https://arxiv.org/abs/2501.05131)
Comments:
          tech report

- **What's New**: 이 논문에서는 Depth-Driven Decoupled Instance Synthesis (3DIS) 방법을 활용해 Multi-Instance Generation (MIG)에서 이미지 생성을 위한 새로운 기술을 소개합니다. 3DIS는 두 단계로 MIG 프로세스를 분리하여 깊이 기반의 장면 구축과 세부 사항 렌더링을 수행합니다. 이 접근 방식은 다양한 모델이 훈련 없이 세부 사항 렌더링을 수행할 수 있게 하여, 자원 소모를 적게 합니다.

- **Technical Details**: 3DIS-FLUX는 FLUX 모델을 통합하여 3DIS의 렌더링 능력을 향상시킨 확장 버전입니다. FLUX.1-Depth-dev 모델을 경우 깊이 맵 제어된 이미지 생성을 위해 사용하고, 레이아웃 정보에 따른 Attention Mask 관리를 통해 각 인스턴스의 세부 렌더링을 조정합니다. 이 방법은 각 인스턴스의 세부 속성을 정밀하게 렌더링할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과에 따르면, 3DIS-FLUX는 FLUX 모델의 장점을 활용하여 기존의 3DIS 방법보다 6.9% 더 높은 Instance Success Ratio (ISR)를 기록했습니다. 또한, 훈련 없이 이루어지는 최신 방법인 Multi-Diffusion에 비해 41% 이상의 개선 효과를 보였고, adapter 기반 방법인 InstanceDiffusion에 비해서도 12.4% 높은 ISR을 달성했습니다. FLUX 모델을 이용한 렌더링으로 뛰어난 이미지 품질을 유지했습니다.



### Optimizing Multitask Industrial Processes with Predictive Action Guidanc (https://arxiv.org/abs/2501.05108)
- **What's New**: 이번 연구에서는 복잡한 조립 프로세스를 모니터링하기 위한 Multi-Modal Transformer Fusion and Recurrent Units (MMTFRU) 네트워크를 도입합니다. 이 시스템은 Operator Action Monitoring Unit (OAMU)와 통합되어 조립 과정에서 인간의 행동 변동성이나 주관적인 작업 선호의 문제를 해결하려고 합니다. MMTF-RU 모델을 활용하여 다양한 데이터 스트림을 처리하며, 예측 정확도를 높이기 위한 멀티모달 융합 기법을 적용하고 있습니다. 결국, 이 연구는 Industry 5.0에 맞춘 스마트하고 반응적인 산업 프로세스를 목표로 합니다.

- **Technical Details**: MMTFRU 모델은 Transformer 기반의 인코더와 Cross-Modality Fusion Block (CMFB)을 활용하여 다양한 데이터 스트림을 처리하는 구조입니다. 활용되는 OAMU는 마르코프 체인 기반 모델을 통해 작업 전환을 포착하고 예측하며, MMTF-RU의 예측값과 결합하여 조치를 권장하고 이상 징후를 사전에 탐지합니다. 이 시스템은 Meccano 및 EPIC-Kitchens-55 데이터셋에서 검증되었으며, Action anticipation(행동 예측), 시간 가중화 시퀀스 정확도(Time-Weighted Sequence Accuracy, TWSA) 등의 성능을 통해 조립 작업의 효율성을 높입니다.

- **Performance Highlights**: 연구 결과, 제안된 MMTFRU 모델은 Meccano 데이터셋에서 행동, 동사 및 명사 예측 분야에서 최첨단 성능을 달성했습니다. 또한 EPIC-Kitchens-55 데이터셋에서도 경쟁력 있는 결과를 나타내며, OAMU의 통합을 통해 조립 공정의 이상 징후를 방지하고 상황별 권장 사항을 제공하여 실시간 작업 효율성을 증대시켰습니다. TWSA 메트릭을 통해 조작자의 효율성을 평가하고, 작업 실행 최적화에 대한 인사이트를 제공하는 등 다기능 산업 환경에서의 성능을 실증적으로 입증하였습니다.



### Motion-X++: A Large-Scale Multimodal 3D Whole-body Human Motion Datas (https://arxiv.org/abs/2501.05098)
Comments:
          17 pages, 14 figures, This work extends and enhances the research published in the NeurIPS 2023 paper, "Motion-X: A Large-scale 3D Expressive Whole-body Human Motion Dataset". arXiv admin note: substantial text overlap with arXiv:2307.00818

- **What's New**: 이번 논문에서는 Motion-X++라는 대규모 다중 모달 3D 표현 전체 신체 인간 모션 데이터셋을 소개합니다. 기존의 모션 데이터셋은 주로 신체 움직임만을 포착하며, 얼굴 표정이나 손 제스처 등과 같은 다양성이 부족합니다. Motion-X++는 총 19.5M개의 전체 신체 포즈 주석을 제공하며, 자동화된 주석 파이프라인을 개선해 더 많은 데이터 변수를 도입하고 수행 능력을 확장했습니다.

- **Technical Details**: Motion-X++는 120.5K 개의 모션 시퀀스를 포함하고 있으며, 80.8K개의 RGB 비디오와 45.3K개의 오디오 자료로 구성되어 있습니다. 이 데이터셋은 고급 설정에서 자동으로 수집한 조합된 텍스트 및 모션 쌍으로 구성되어 있으며, SMPL-X 모델을 기반으로 포괄적인 주석 프로세스를 통해 이루어집니다. 새로운 고성능 프레임워크가 전체 신체 키포인트 추정, 시간 최적화 및 교육 기반 3D 인간 모델 적합 과정을 포함하여 정확한 3D 모션 캡처를 가능하게 합니다.

- **Performance Highlights**: Motion-X++의 정밀한 주석 시스템을 통해 다양한 다운스트림 작업에서의 효용성을 입증했으며, 텍스트 기반 전체 신체 모션 생성, 오디오 기반 모션 생성과 같은 작업에서 우수한 성능을 보였습니다. 연구 결과, Motion-X++는 기존의 Motion-X 데이터셋보다 더 안정적이고 정확한 신체 모션을 제공하여, 더욱 풍부하고 현실감 있는 모션 생성이 가능합니다. 이러한 결과는 다양한 최신 ZETA (State-of-the-art) 모션 생성 방법의 평가 기준을 설정할 수 있는 새로운 차원의 데이터셋으로 자리잡았습니다.



### A 1Mb mixed-precision quantized encoder for image classification and patch-based compression (https://arxiv.org/abs/2501.05097)
Comments:
          Published at IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)

- **What's New**: 본 논문에서는 이미지 처리에 전념하는 ASIC 신경망 가속기가 이미지 분류 및 압축과 같은 다양한 작업에 적용될 수 있음을 보여줍니다. 자원 소모를 최소화하면서 하드웨어 성능을 높이기 위해 Mixed-Precision (혼합 정밀도) 인코더를 도입하였습니다. 또한, Linear Symmetric Quantizer의 스케일링 팩터를 자동으로 조정해 가중치 훈련을 안정화하는 방법을 제시합니다.

- **Technical Details**: 제안된 인코더는 reconfigurability를 갖추고 있으며, 3비트, 2비트, 1비트의 Mixed-Precision 양자화를 통해 효율적인 하드웨어 설계를 가능하게 합니다. 기존 Batch Normalization을 간소화하기 위해 Layer-Shared Bit-Shift Normalization 기법을 도입하였습니다. 이와 함께, 높은 분해능을 유지하면서 이미지 압축을 가능하게 하는 새로운 회복 네트워크(PURENET)를 소개합니다.

- **Performance Highlights**: 주어진 설정에서 인코더 디자인은 1MB 메모리만을 요구하며, CIFAR-10 데이터셋에 대해 87.5%의 분류 정확도를 달성하였습니다. 양자화된 인코더를 사용한 이미지 압축 과정에서 블록 아티팩트가 거의 없이 원격 복원이 가능합니다. 이러한 결과는 기존의 패치 기반 압축 기법보다 우수한 성능을 보여주며, 저비트 전송에서도 효과적입니다.



### Advancing ALS Applications with Large-Scale Pre-training: Dataset Development and Downstream Assessmen (https://arxiv.org/abs/2501.05095)
- **What's New**: 이 연구는 항공 레이저 스캐닝(ALS) 기술을 위한 대규모 데이터셋을 구성하고 이를 통해 다운스트림(downstream) 애플리케이션에 미치는 영향을 평가합니다. 기존의 대규모 데이터셋이 ALS 응용 분야에 충분한 다양성과 규모를 제공하지 못했던 반면, 이 연구는 미국 지질조사국(USGS)의 3D 고도 프로그램을 기반으로 하는 다각적 지리적 샘플링 방법을 도입하였습니다. 이 방법은 다양한 토지 피복과 지형을 포괄하는 ALS 포인트 클라우드를 효과적으로 수집할 수 있게 합니다.

- **Technical Details**: 항공 레이저 스캐닝(ALS)은 공중 플랫폼에서 레이저 펄스를 방출하고 반사된 신호를 분석하여 고해상도의 3D 공간 데이터를 수집하는 기술입니다. 이 연구에서는 BEV-MAE라는 최신 자가 지도 학습(self-supervised learning) 모델을 기반으로 하여, 구성된 포인트 클라우드 데이터셋에 사전 학습(pre-training)을 수행하고, 이를 기반으로 나무 종 분류, 지형 장면 인식 및 포인트 클라우드 의미적 분할이라는 다운스트림 작업에 대해 미세 조정(fine-tuning)을 진행합니다. 또한 제안된 샘플링 방법을 통해 데이터셋의 규모를 확장하여 성능 향상을 모니터링 하는 방법을 제시합니다.

- **Performance Highlights**: 사전 학습된 모델은 모든 다운스트림 작업에서 스크래치 모델에 비해 월등한 성과를 나타내었으며, 이는 제안된 데이터셋에서 학습한 표현의 전이 가능성을 입증합니다. 데이터셋의 확장을 통해 성능이 일관되게 개선되는 것을 관찰하였으나, 무작위 샘플링으로 구성된 데이터셋에서 사전 학습을 수행한 경우에는 유사한 성능 향상을 달성하지 못했습니다. 연구 결과는 ALS 응용을 위한 사전 학습 및 미세 조정 패러다임의 유용성을 강조합니다.



### ResPanDiff: Diffusion Model with Disentangled Modulations for Image Fusion (https://arxiv.org/abs/2501.05091)
- **What's New**: 이번 연구에서 제안하는 ResPanDiff 모델은 기존의 확산 모델보다 개선된 샘플링 속도를 자랑하며, 기존의 정밀한 성능을 유지합니다. 이 모델은 Markov 체인을 통해 노이즈 잔차에서 LRMS(저해상도 다중 스펙트럼)와 HRMS(고해상도 다중 스펙트럼) 이미지 간의 잔차를 비교하도록 설계되었습니다. 이를 통해 샘플링 단계를 크게 줄이면서도 성능을 감소시키지 않는 혁신적인 접근 방식을 제공하고 있습니다.

- **Technical Details**: ResPanDiff는 잔차 생성을 위한 Latent Space와 Shallow Cond-Injection(SC-I) 구조를 포함하고 있습니다. 또한, 잔차 생성 작업을 위한 손실 함수도 디자인하여 모델이 최적의 성능에 도달할 수 있도록 보조합니다. 이와 같은 디자인은 모델이 MSIF(다중 소스 이미지 융합) 작업을 보다 효과적으로 수행할 수 있게 해줍니다.

- **Performance Highlights**: 실험 결과, ResPanDiff는 세 가지 널리 사용되는 pansharpening 데이터셋에서 최신 기술(SOTA) 보다 우수한 성능을 기록했습니다. 특히, 단 15회의 샘플링 단계만으로도 90% 이상의 샘플링 단계 감소를 달성했습니다. 제안된 모델의 효율성과 효과성은 다양한 실험과 아블레이션 연구를 통해 검증되었습니다.



### End-to-End Deep Learning for Interior Tomography with Low-Dose X-ray C (https://arxiv.org/abs/2501.05085)
Comments:
          Published by Physics in Medicine & Biology (2022.5)

- **What's New**: 이 논문에서는 기존 이미지 도메인 기반의 중첩된 저선량 ROI CT 문제를 해결하기 위한 새로운 방법을 제안합니다. 연구자들은 이미지 도메인에서의 노이즈 감소와 투영 외삽 문제를 두 개의 하위 문제로 나누어 해결합니다. 제안된 방법은 dual-domain CNNs를 사용하여 두 가지 문제를 동시에 해결함으로써 기존의 이미지 도메인 딥러닝 방법보다 우수한 성능을 보입니다.

- **Technical Details**: 저자들은 깊이 있는 convolutional framelets 이론을 기반으로 한 새로운 end-to-end 딥러닝 방법을 제안하며, 이는 투영 도메인에서의 CNN을 활용하여 cupping artifacts와 이미지 노이즈의 혼합을 제거합니다. 표준 CNN 아키텍처는 두 가지 결합된 아티팩트를 해결하는 데 어려움을 겪기 때문에, 새로운 투영 도메인 CNN과 그 이후의 이미지 도메인 CNN이 결합된 형태로 설계됩니다. 이는 넷워크 구조가 두 개의 주요 CNN으로 구성된다는 것을 의미합니다.

- **Performance Highlights**: 제안된 방법은 기존의 이미지 도메인 기반의 딥러닝 방법보다 뛰어난 성능을 보이며, 특히 투영 도메인 CNN은 일반적으로 사용되는 이미지 도메인 CNN보다 더 나은 결과를 나타냅니다. 실험 결과, 제안된 모델이 이미지 품질과 재구성 시간을 모두 개선했다는 것이 입증되었습니다. 이로 인해 저선량 CT와 ROI CT 문제에 효과적으로 접근할 수 있는 가능성이 열렸습니다.



### TipSegNet: Fingertip Segmentation in Contactless Fingerprint Imaging (https://arxiv.org/abs/2501.05076)
- **What's New**: 이 논문에서는 TipSegNet이라는 새로운 딥러닝 모델을 소개하여, 그레이스케일 손 이미지에서 직접 손가락 끝을 분할하는 최신 기술을 구현합니다. TipSegNet은 강력한 특징 추출을 위해 ResNeXt-101 백본을 활용하며, 멀티스케일 표현을 위한 Feature Pyramid Network (FPN)을 결합하여 각기 다른 손가락 포즈와 이미지 품질에서도 정확한 분할을 가능하게 합니다. 또한, 모델의 일반화 능력과 견고성을 강화하기 위해 광범위한 데이터 증강 전략을 사용합니다.

- **Technical Details**: TipSegNet은 손 이미지에서 손가락 끝을 직접 추출하는 방식을 채택하여 이전의 일회용 손가락 분할 기법과 차별화됩니다. 입력 이미지의 크기를 표준화하여 일관된 가중치 행렬 차원과 안정적인 학습을 보장하며, 주로 224×224 픽셀 크기로 조정됩니다. 훈련 데이터에는 다양한 증강 기법이 50% 확률로 적용되며, 이미지 크기 변경, 회전 및 원근 변화와 같은 변환이 무작위로 수행됩니다.

- **Performance Highlights**: TipSegNet 모델은 평균 Intersection over Union (mIoU) 0.987 및 정확도 0.999로 기존 방법들보다 우수한 성능을 보입니다. 이러한 향상된 정확성은 현실 세계의 비접촉 생체 인식 시스템에서 신뢰성과 효율성을 크게 향상시킬 가능성이 있습니다. 논문에서 제안한 방법은 여러 사용 사례에서 사용자 편의성과 안전을 극대화하며, 특히 팬데믹 이후 더욱 중요해진 위생적인 biometric 솔루션에 부합합니다.



### Commonsense Video Question Answering through Video-Grounded Entailment Tree Reasoning (https://arxiv.org/abs/2501.05069)
- **What's New**: 본 논문은 비디오 기반의 일반 상식 질문 대답(Video Question Answering, VQA)을 위한 최초의 비디오 기초 추론 방법을 제안합니다. 기존의 대형 비주얼-언어 모델(Visual-Language Models, VLMs)에서 발생하는 단기적인 연관 학습 문제를 해결하기 위하여, 해당 방법은 VQA 작업을 비디오 조각에 명시적으로 연결합니다. 이를 통해 모델의 의사 결정 과정을 설명할 수 있는 명확한 추론 체인을 제공합니다.

- **Technical Details**: 제안하는 방법은 VQA 작업을 네 가지 단계로 수행합니다: (i) 추론 트리 구축, (ii) 비디오-언어 추론 검증, (iii) 트리 추론, (iv) 동적 트리 확장. 각 단계는 비디오와 다중 선택 질문을 사용하여 후보 답변을 명시적으로 검증하고, 이를 통해 비디오의 관련 정보와 정합성을 확인합니다. 이 과정에서 VLMs에 적용할 수 있는 일반화 가능성을 강조합니다.

- **Performance Highlights**: 실험 결과, 제안하는 비디오 기반 추론 트리는 기존의 비디오 및 이미지 기반 모델에 비해 더 나은 성능을 보여주었습니다. 또한, 제안된 방법은 텍스트와 비디오 정보를 함께 고려함으로써 응답의 강인성과 해석 가능성을 향상시켰고, 특히 인과적 및 시간적 질문에서 두드러진 성과를 나타냈습니다.



### LLaVA-Octopus: Unlocking Instruction-Driven Adaptive Projector Fusion for Video Understanding (https://arxiv.org/abs/2501.05067)
- **What's New**: 본 논문에서는 사용자 지침에 따라 다양한 시각적 프로젝터의 기능을 적절히 가중하는 새로운 비디오 다중 모달 대형 언어 모델인 LLaVA-Octopus를 소개합니다. 이 모델은 각 프로젝터의 강점을 활용하여 다중 모달 작업의 성능을 크게 향상시키며, 사용자 지침을 기반으로 기능 가중치를 동적으로 조정합니다. 실험 결과, LLaVA-Octopus는 멀티모달 이해, 비주얼 질문 응답 및 비디오 이해와 같은 작업에서 탁월한 성능을 달성했습니다.

- **Technical Details**: 최근 비디오 이해의 주요 과제는 시간 동적성 관리와 복잡한 의미 이해입니다. 다양한 시각적 프로젝터가 특정 작업을 처리하는 데서 고유한 성능을 보여주며, 본 논문에서는 이를 크게 이미지 기반 프로젝터, 공간-시간 프로젝터 및 토큰 압축 프로젝터로 분류합니다. LLaVA-Octopus는 이들 서로 다른 프로젝터의 기능을 통합하는 입력 기반의 프로젝터 융합 패러다임을 제안하며, 각 프로젝터의 상호 보완적인 장점을 최대한 활용합니다.

- **Performance Highlights**: LLaVA-Octopus는 다양한 벤치마크에서 최고의 성능을 기록하며, 특히 비디오 및 시각적 데이터를 다루는 멀티모달 작업에서 다른 모델들과 비교해 우수한 결과를 보여주고 있습니다. 실험 결과를 통해 대다수의 벤치마크에서 최첨단(State-of-the-Art, SOTA) 성능을 달성한 것으로 나타났습니다. 이 모델은 다중 모달 작업에서의 범용적인 활용 가능성을 강조하며, 향후 연구 및 응용 분야에 기여할 것으로 기대됩니다.



### Improving Skeleton-based Action Recognition with Interactive Object Information (https://arxiv.org/abs/2501.05066)
- **What's New**: 본 논문에서는 기존 스켈레톤 기반 행동 인식 방법의 한계를 극복하기 위해 객체 노드를 도입한 새로운 행동 인식 프레임워크를 제안합니다. 이를 통해 인간과 상호작용하는 객체의 정보를 보강하여 동작 인식 성능을 높이는 것을 목표로 합니다. 특히 Spatial Temporal Variable Graph Convolutional Networks (ST-VGCN)라는 모델을 제안하여 객체 노드를 포함한 변동 그래프를 효과적으로 모델링합니다.

- **Technical Details**: 우리가 제안한 ST-VGCN은 개체 노드를 포함하여 인간-개체 간의 관계를 학습하는 새로운 프레임워크입니다. 이 모델은 Random Node Attack이라는 데이터 증강 방법을 통해 객체 정보에 의해 발생하는 데이터 편향 문제를 해결하고, 별도의 Variable Graph 구성 방법을 도입하여 동작 인식에서의 유연성을 높입니다. 이를 통해 스켈레톤 노드와 객체 노드를 균형 있게 활용하여 네트워크의 일반화 성능을 향상시킵니다.

- **Performance Highlights**: 제안된 방법은 NTU RGB+D 60 데이터셋에서 cross-subject split에서 96.7%, cross-view split에서 99.2%의 정확도로 이전의 최첨단 성과를 초과했습니다. 이와 같은 결과는 다양한 스켈레톤 기반 행동 인식 벤치마크에서 우리의 접근법이 효과적임을 입증합니다. 향후 연구는 이 모델을 기반으로 추가적인 객체와 상호작용하는 행동 인식의 가능성을 탐구할 것입니다.



### LongViTU: Instruction Tuning for Long-Form Video Understanding (https://arxiv.org/abs/2501.05037)
- **What's New**: LongViTU는 대규모의 자동 생성된 데이터셋으로, 약 121,000개의 QA 쌍과 900시간에 달하는 비디오를 포함하고 있습니다. 이 데이터셋은 비디오 이해를 향상시키기 위해 계층적 트리 구조로 조직되어 있으며, 자가 수정 메커니즘을 통해 높은 품질의 QA 쌍을 보장합니다. LongViTU는 장기적인 맥락, 풍부한 지식 및 명확한 타임스탬프 라벨을 특징으로 합니다.

- **Technical Details**: LongViTU는 현재 기존 데이터셋의 제한을 극복하기 위해 다양한 현실 세계 시나리오 및 명시적 타임스탬프 레이블을 제공하여 비디오 컨텐츠를 계층적으로 구조화합니다. 특히, 평균 증명 길이가 4.6분으로 긴 QA 쌍을 생성하며, 세심한 공간적 및 시간적 세부 사항을 유지합니다. QA 생성이 가능하도록 다양한 수준에서 비디오를 분류하는 파이프라인을 마련하였습니다.

- **Performance Highlights**: LongViTU의 품질은 상태-of-the-art 모델인 LongVU와 상업적 모델인 Gemini-1.5-Pro의 평가 결과에서도 드러납니다. 이들은 각각 GPT-4 점수 49.9 및 52.3을 기록했으며, 추가적인 지도 초미세 조정(Supervised Fine-Tuning, SFT) 후 LongVU는 12%의 성능 향상을 이뤄냈습니다. 이러한 결과는 LongViTU가 높은 데이터 품질과 강력한 OOD(Out-of-Distribution) 일반화를 제공한다는 것을 실증적으로 증명합니다.



### Towards Fingerprint Mosaicking Artifact Detection: A Self-Supervised Deep Learning Approach (https://arxiv.org/abs/2501.05034)
- **What's New**: 본 논문에서는 깊이 있는 학습 방법을 활용하여 지문 이미지 내의 모자이크 아티팩트를 탐지하고 평가하는 새로운 접근법을 제안합니다. 이 방법은 대규모 비지도 학습 데이터셋을 통해 훈련하며, 수동 아티팩트 주석이 필요 없습니다. 제안된 모델은 접촉식, 롤러식 및 압착식 등 다양한 지문 모드에서 높은 정확도로 모자이크 오류를 식별합니다.

- **Technical Details**: 지문 모자이크화 과정은 여러 지문 이미지를 결합하여 완전한 마스터 지문 이미지를 생성하는 것을 목표로 합니다. 두 가지 주요 방법인 접촉식 및 비접촉식 방법으로 지문을 수집하며, 특히 롤러식을 사용할 때 여러 개의 부분 이미지를 스티칭하는 과정에서 모자이크 아티팩트를 관리하는 것이 중요합니다. 연구는 '소프트' 및 '하드' 모자이크 오류를 분류하고, 깊이 있는 학습 모델을 사용하여 하드 모자이크 오류를 탐지하기 위한 새로운 접근법을 제시합니다.

- **Performance Highlights**: 본 논문에서 제안한 모델은 다양한 데이터 소스에 대해 강인성을 입증하며, 아티팩트의 심각성을 수치화하기 위해 새로운 모자이크 아티팩트 점수를 도입합니다. 이로 인해 지문 이미지의 자동화된 평가가 가능해지며, 아티팩트 제거의 영향을 평가하기 위해 적절한 Equal Error Rates (EER)를 계산하였습니다. 이러한 방법은 지문 기반 생체 인식 시스템의 정확성과 신뢰성을 향상시키는 데 기여합니다.



### ECBench: Can Multi-modal Foundation Models Understand the Egocentric World? A Holistic Embodied Cognition Benchmark (https://arxiv.org/abs/2501.05031)
- **What's New**: 본 논문에서는 LVLMs의 체화된 인지 능력을 체계적으로 평가하기 위한 새로운 벤치마크인 ECBench를 소개합니다. ECBench는 다양한 장면 비디오 출처와 개방적 질문 형식을 포함하며, 30가지의 인지 차원을 평가합니다. 이를 통해 현재 LVLMs의 평가 체계에서 나타나는 주요 한계점을 극복하고자 하였습니다.

- **Technical Details**: EBench는 정적 장면, 동적 장면 및 환각 문제를 포함하는 세 가지 세트를 다루며, 로봇 중심 인지 질문을 도입하였습니다. 또한, ECEval이라는 새로운 평가 시스템을 통해 질문 응답 작업에서 정밀한 평가를 가능하게 합니다. 데이터 수집 과정에서 다양한 형태의 질문 형식을 유연하게 사용하여 인지 요구 사항을 충족하도록 설계되었습니다.

- **Performance Highlights**: LVLM들을 대상으로 실시한 평가 결과, 현재의 주류 LVLM들은 동적 장면과 체화된 환각 문제에서 낮은 성능을 보였고, 정적 장면에서는 로봇 중심 질문이 더 어려움을 겪었습니다. 이는 현대의 LVLM들이 정적 장면에서는 3인칭 인지에만 해당하지만, 동적 장면에서는 1인칭 이해를 얻기 위해 고전하고 있음을 시사합니다.



### Perception-as-Control: Fine-grained Controllable Image Animation with 3D-aware Motion Representation (https://arxiv.org/abs/2501.05020)
- **What's New**: 이번 논문은 3D를 인식하는 모션 표현(3D-aware motion representation)을 도입하여 세밀한 협업 모션 제어(fine-grained collaborative motion control)를 가능하게 하는 이미지 애니메이션 프레임워크인 Perception-as-Control을 제안합니다. 기존의 모션 표현은 카메라와 객체의 모션 제어 시 충돌 문제와 충분하지 않은 제어 세밀성 문제를 겪고 있었으나, 본 연구는 이러한 문제를 다룹니다. 또한, 이 프레임워크는 사용자의 의도를 해석하여 다양한 모션 관련 비디오 합성 작업을 유연하게 지원할 수 있습니다.

- **Technical Details**: 연구에서 제안하는 Perception-as-Control 프레임워크는 단일 참조 이미지와 3D-aware motion representation의 인식 결과를 모션 제어 신호로 사용합니다. 카메라와 객체의 모션 제어 신호는 공간적으로 정렬(spatially aligned)되어 있으며, 두 개의 경량 인코더를 활용해 캠코드와 객체 제어 신호를 별도로 인코딩하여 RGB 수준의 간섭을 피합니다. 또한 U-Net 아키텍처를 기반으로 한 확산 모델(diffusion model)을 사용하여 참조 이미지의 외관 정보와 모션 정보를 결합하여 애니메이션을 제어합니다.

- **Performance Highlights**: 실험 결과, 제안한 프레임워크는 모션 생성(motion generation), 모션 복제(motion clone), 모션 전이(motion transfer), 모션 편집(motion editing) 등 다양한 응용 분야에서 탁월한 성능을 보였습니다. 특히, 3D-aware motion representation의 도입에 따라 사용자의 명확한 의도를 기반으로 한 세밀한 물체 및 카메라의 모션 제어를 지원할 수 있음을 입증했습니다. 이러한 접근 방식은 비디오 합성 작업에서의 품질 향상에 기여하며, 애니메이션 디자인의 새로운 가능성을 열어줍니다.



### Continuous Knowledge-Preserving Decomposition for Few-Shot Continual Learning (https://arxiv.org/abs/2501.05017)
Comments:
          Code: this https URL

- **What's New**: 이번 연구에서는 Continuous Knowledge-Preserving Decomposition for FSCIL (CKPD-FSCIL)이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 모델의 가중치를 기존 지식을 집약하는 '지식 민감 구성 요소'와 새로운 능력을 위한 '여분 용량 구성 요소'로 분해합니다. 이를 통해 지식의 안정성을 유지하면서도 모델의 적응성을 증가시킬 수 있습니다. 기존 방법의 한계를 극복하고자 하며, 여러 벤치마크에서 최신 기법들과의 성능을 비교하여 우수함을 입증합니다.

- **Technical Details**: CKPD-FSCIL의 핵심은 지식 민감 구성 요소를 동결하고, 여분 용량 구성 요소만 조정하여 학습의 유연성을 제공하는 것입니다. 이 과정에서 공분산 행렬을 활용하여 특징 공간의 차원 축소를 통해 새로운 지식을 위한 공간을 확보합니다. Singular Value Decomposition (SVD) 기법을 사용해 가중치를 조정하며, 각 층의 여분 용량에 대한 적응적인 층 선택 전략을 도입하여 동적으로 어댑터를 할당합니다. 이를 통해 각 세션의 특성에 맞게 지식을 보존하는 동시에 새로운 작업에 적응합니다.

- **Performance Highlights**: CKPD-FSCIL은 다양한 벤치마크에서 기존 접근방식보다 우수한 성능을 보였습니다. 이 방법은 지식의 안정성을 유지하면서 새로운 정보를 효율적으로 수용할 수 있도록 설계되었습니다. 추가적인 매개 변수를 도입하지 않으면서도 모델 구조를 복원할 수 있어 효율적인 추론 성능을 제공합니다. 결론적으로, 이번 연구는 FSCIL 분야에서의 적응성과 재학습 간의 균형을 효과적으로 조정할 수 있는 혁신적인 접근법을 제시합니다.



### A CT Image Classification Network Framework for Lung Tumors Based on Pre-trained MobileNetV2 Model and Transfer learning, And Its Application and Market Analysis in the Medical field (https://arxiv.org/abs/2501.04996)
- **What's New**: 이 논문에서는 폐암(肺癌) 진단에 대한 정확성을 높이기 위해 전통적인 수동 분석 방법의 한계를 극복하는 새로운 접근 방식을 제안합니다. 기존의 MobileNetV2 모델을 기반으로 한 딥러닝 네트워크를 개발하였으며, 이 모델은 ImageNet-1K 데이터셋으로 초기화된 가중치를 활용합니다. 그 결과로 세 가지 유형의 폐암 CT 스캔 이미지를 효과적으로 분류할 수 있는 구조가 완성되었습니다.

- **Technical Details**: 제안된 모델의 마지막 층은 새로운 완전 연결층(fully connected layer)으로 대체되었고, 소프트맥스 활성화 함수(softmax activation function)가 추가되어 분류의 효율성을 증가시킵니다. 실험 결과, 이 모델은 테스트 세트에서 99.6%의 정확도를 달성하였으며, 전통적인 방법에 비해 특징 추출(feature extraction)에서 현격한 개선을 보였습니다.

- **Performance Highlights**: 이 연구는 AI 기반의 폐암 탐지 시스템이 진단 효율성을 현저하게 향상시키고 의사들의 업무량을 줄일 수 있는 가능성을 보여줍니다. 딥러닝 기술의 발전이 의료 영상 처리에 혁신적인 변화를 가져오고 있으며, 이러한 변화는 향후 의료 산업의 발전에 깊은 영향을 미칠 것으로 예상됩니다. 또한, 이러한 시스템은 글로벌 헬스케어 시장에서 중요한 위치를 차지할 것으로 보입니다.



### IPDN: Image-enhanced Prompt Decoding Network for 3D Referring Expression Segmentation (https://arxiv.org/abs/2501.04995)
Comments:
          AAAI 2025

- **What's New**: 이 논문에서는 3D Referring Expression Segmentation (3D-RES)의 두 가지 주요 문제인 feature ambiguity와 intent ambiguity를 해결하기 위한 새로운 접근법인 Image enhanced Prompt Decoding Network (IPDN)을 제안합니다. IPDN은 다중 뷰 이미지(multi-view images)와 작업 중심(task-driven) 정보를 활용하여 모델의 추론 능력을 향상시키도록 설계되었습니다. 이를 통해 해당 문제를 효과적으로 해결하고 더 나은 성능을 달성할 수 있음을 보여줍니다.

- **Technical Details**: 제안된 IPDN은 Multi-view Semantic Embedding (MSE) 모듈과 Prompt-Aware Decoder (PAD) 모듈로 구성됩니다. MSE는 CLIP을 활용하여 2D 이미지 특징을 추출하고 이를 3D 포인트 클라우드 특성과 융합하여 시각적 표현을 향상시킵니다. PAD는 text와 시각적 특징 간의 상호작용으로부터 과제 중심의 신호(task-driven signals)를 유도하여 디코딩 프로세스를 안내합니다.

- **Performance Highlights**: 다양한 실험 결과, IPDN은 3D-RES 및 3D-GRES 작업에서 각각 1.9 및 4.2 mIoU 포인트 개선을 이루며 기존의 최첨단(state-of-the-art) 방법들을 초월했습니다. 이는 IPDN이 실제 작업에서 우수한 성능을 발휘함을 강하게 시사합니다. 이 연구는 3D-RES 분야의 발전에 중요한 기여를 하는 동시에 관련된 과제를 극복하기 위한 새로운 방향성을 제시합니다.



### V2C-CBM: Building Concept Bottlenecks with Vision-to-Concept Tokenizer (https://arxiv.org/abs/2501.04975)
Comments:
          Accepted by AAAI2025

- **What's New**: 이번 연구에서는 다중 모달 (multimodal) 모델을 기반으로 직접적으로 개념 병목 모델 (Concept Bottleneck Models, CBMs)을 구성하는 방법을 제안합니다. 이를 통해 기존의 대형 언어 모델 (Large Language Models, LLMs)을 사용하지 않고도 명확한 시각적 개념을 생성할 수 있도록 합니다. V2C 토크나이저 (Vision-to-Concept tokenizer)를 사용하여 이미지를 가장 관련성이 높은 시각적 개념으로 변환하고, 이는 높은 정확도를 보장하며, 판별 가능성과 효율성을 제공합니다.

- **Technical Details**: 우리의 V2C 토크나이저는 일반적인 단어를 기본 개념 어휘로 사용하고, 보조적인 비표기 이미지 (auxiliary unlabeled images)를 활용하여 시각적 개념을 정량화하는 데 초점을 맞춥니다. 이 접근 방식은 다중 모달 모델과 밀접하게 결합된 비전 중심의 개념 병목 구조를 생성합니다. 이는 사전 훈련된 대형 언어 모델의 도움 없이도 이미지를 명확한 시각적 개념으로 변환할 수 있도록 합니다.

- **Performance Highlights**: V2C-CBM 이 모델은 다양한 시각적 분류 벤치마크에서 LLM 감독 하의 CBMs과 동등하거나 더 나은 성능을 보여주었습니다. 이를 통해 우리의 접근 방식이 효과적임을 검증하였으며, 시각적으로 해석 가능하며 높은 정확성의 분류 성능을 달성했습니다. V2C-CBM은 리소스가 제한된 작업에서도 효율적으로 활용될 수 있도록 설계되었습니다.



### Emergence of Painting Ability via Recognition-Driven Evolution (https://arxiv.org/abs/2501.04966)
- **What's New**: 이 연구에서는 진화 압력을 통해 효율적인 시각적 커뮤니케이션을 모사한 인간과 유사한 회화 능력을 가진 모델을 제안합니다. 이 모델은 두 가지 분기(Stroke Branch, Palette Branch)로 구성되어 있으며, 각 분기는 이미지를 그리기 위해 Bézier 곡선을 사용하여 각 스트로크를 매개변수화합니다. 이 연구는 기계 비전(machin vision)으로 달성된 인식 정확도를 기반으로 효율성을 정량화하였습니다.

- **Technical Details**: 모델은 스트로크 분기와 팔레트 분기로 구성되며, 팔레트 분기는 제한된 색상 팔레트를 학습하고 스트로크 분기는 Bézier 곡선을 통해 이미지를 렌더링합니다. 이 과정에서 각 스트로크의 제어점 및 색상 선택 최적화하여 최소한의 스트로크와 색상으로 인식 정확도를 극대화합니다. Differentiable Rasterizer를 기반으로 한 Stroke Branch는 특정 스케치 데이터 셋 없이도 반복적으로 스트로크 매개변수를 업데이트합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 고차원 인식 작업에서 우수한 성능을 보여주었으며, 특히 추상 스케치에서 예술적 표현과 미적 매력을 제공했습니다. 또한 이 연구는 전통적인 방법들보다 효율적 이미지 압축(bit-level image compression) 기술로서의 잠재력을 보여주었습니다. 이를 통해 시각적 품질과 압축 효율성에서 탁월함을 입증하였습니다.



### Addressing Domain Shift via Imbalance-Aware Domain Adaptation in Embryo Development Assessmen (https://arxiv.org/abs/2501.04958)
Comments:
          15 pages

- **What's New**: 이 논문에서는 의료 이미지 분석에서의 도메인 시프트(domain shift)와 클래스 불균형(class imbalance) 문제를 동시에 해결하기 위한 새로운 프레임워크인 Imbalance-Aware Domain Adaptation (IADA)를 제안합니다. 이 프레임워크는 (1) 클래스별 주의 메커니즘을 활용한 적응형 특징 학습, (2) 동적 가중치를 통한 균형 잡힌 도메인 정렬, (3) 적응형 임계값 최적화의 세 가지 주요 요소로 구성되어 있습니다. 실험 결과 IADA는 기존 방법들에 비해 최대 25.19% 높은 정확도를 달성하며, 다양한 클래스에서의 성능 균형을 유지하는 뛰어난 효과를 보여줍니다.

- **Technical Details**: IADA는 복잡한 모델 성능 저하를 방지하기 위한 세 가지 혁신적인 접근을 통합하고 있습니다. 이론적 분석을 통해 수렴 보장(convergence guarantees)과 복잡성 경계(complexity bounds)를 설정하였으며, 여러 영상 기법에서의 배아 발달 평가 실험을 통해 IADA의 효과성을 입증하였습니다. 특히, 저품질 영상 시스템에서의 강력한 일반화 능력을 보여주며, AUC(Area Under the Curve)에서 최대 12.56% 향상을 나타냅니다.

- **Performance Highlights**: IADA는 의료 이미지 시스템의 안정성과 공정성을 개발하기 위한 잠재력을 지니고 있습니다. 연구 결과에 따르면 기존 기술에 비해 성능이 향상되어, 특히 다른 임상 환경에서의 적응력과 정확도 증대를 가져왔습니다. 따라서, IADA는 다양한 환자 집단과 임상 환경에서 일반화 가능한 알고리즘을 구축하는 데 기여할 수 있는 가능성을 보여줍니다.



### MORDA: A Synthetic Dataset to Facilitate Adaptation of Object Detectors to Unseen Real-target Domain While Preserving Performance on Real-source Domain (https://arxiv.org/abs/2501.04950)
Comments:
          7 pages, 6 figures, 4 tables, This work has been submitted to the IEEE for possible publication (the paper is submitted to the conference ICRA2025 and is under review)

- **What's New**: 이번 연구에서는 자율주행차(AV) 도메인의 학습 데이터를 대체하기 위해 합성 환경(synthetic environments)을 활용하는 새로운 접근 방식을 제안합니다. 저자들은 이를 통해 실제 주행 환경(real-target domain)의 특성을 재현할 수 있는 데이터를 생성하여, AV의 학습 과정에서 발생하는 데이터 수집 및 레이블링 비용을 경감할 수 있습니다. 특히, 최근 전 세계적으로 자율주행차의 필요성이 증가하는 가운데, 새로운 도메인에 대응하기 위한 데이터셋 구축의 어려움이 두드러지는 상황에서, 한국의 도로 환경을 반영한 데이터셋 MORDA가 주목받고 있습니다.

- **Technical Details**: MORDA는 여러 지역의 디지털 트윈을 구성하고, nuScenes 데이터 수집 프레임워크를 복제하여 생성된 합성 데이터셋입니다. 이 방법론의 적용을 통해 현실적인 주행 환경에서의 데이터 수집 없이도 DNN의 학습에 필요한 데이터를 얻을 수 있게 됩니다. MORDA는 2D 및 3D 감지기(detector)를 nuScenes와 결합하여 훈련 시키며, 이를 통해 한국의 주행 환경에 대한 학습 성능을 개선할 수 있습니다.

- **Performance Highlights**: 실험 결과, MORDA를 사용한 모델은 AI-Hub 데이터셋에서 mAP(평균 평균 정밀도) 성능이 유의미하게 향상되었으며, nuScenes에서의 성능은 유지되거나 약간 개선되었습니다. 이 결과는 MORDA가 자율주행차에 필요한 적응 능력을 높일 수 있음을 시사합니다. 따라서, MORDA는 다양한 지역에 AV을 일반화하는데 기여하는 유용한 도구로 자리매김할 것으로 기대됩니다.



### Seeing with Partial Certainty: Conformal Prediction for Robotic Scene Recognition in Built Environments (https://arxiv.org/abs/2501.04947)
Comments:
          10 pages, 4 Figures

- **What's New**: 이번 연구에서는 부분적 확신(Seeing with Partial Certainty, SwPC)이라는 새로운 프레임워크를 소개합니다. 이는 비전 언어 모델(Vision Language Models, VLM) 기반의 장소 인식에서 불확실성을 측정하고 조정하는 데 초점을 맞추고 있습니다. SwPC는 모델이 자신이 신뢰하지 못할 때 인지하고 필요 시 도움을 요청할 수 있도록 도와주며, 이는 복잡한 실내 환경에서 사람의 도움 요청을 최소화하면서도 장소 인식의 통계적 보장을 제공합니다.

- **Technical Details**: SwPC는 정형 예측(Conformal Prediction, CP)의 이론에 기반하여 불확실성과 신뢰도를 조정합니다. CP는 모델의 예측 세트를 생성하기 위한 간단한 방법으로, 모형이 발견한 확률을 기반으로 한 평가 방식을 사용합니다. 이 프레임워크는 VLM의 모든 경우에 적용 가능하며, 모델의 세부 조정(fine-tuning) 없이도 사용될 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험을 통해 SwPC는 Matterport3D와 같은 일반적으로 사용되는 장면 데이터셋에서 높은 성공률을 기록하며 인간의介入이 필요한 정도를 줄였습니다. 이전 방법들과 비교했을 때, SwPC는 예측 성공률을 증가시키고, 이로 인해 로봇의 안전성과 효율성을 크게 개선하는 것으로 나타났습니다. 이는 VLM 능력을 확장하는 데 기여하며, 경량화된 불확실성 모델링 접근법으로서 큰 가능성을 보여줍니다.



### MambaHSI: Spatial-Spectral Mamba for Hyperspectral Image Classification (https://arxiv.org/abs/2501.04944)
Comments:
          accepted by IEEE TGRS

- **What's New**: 이 논문에서는 Mamba 모델을 기반으로 하여 하이퍼스펙트럼 이미지(Hyperspectral Image, HSI) 분류를 위한 새로운 모델인 MambaHSI를 제안합니다. MambaHSI는 전체 이미지를 입력으로 받아서 공간적 및 스펙트럼 정보를 통합하며, 장거리 상호작용을 모델링할 수 있는 능력을 가지고 있습니다. 이 모델은 픽셀 수준에서 장거리 상호작용을 모델링하기 위해 Spatial Mamba 블록(SpaMB)과 스펙트럼 벡터를 여러 그룹으로 나누어 스펙트럼 특징을 추출하는 Spectral Mamba 블록(SpeMB)을 설계했습니다.

- **Technical Details**: MambaHSI는 순수 SS(Mamba) 기반 구조로, 전체 이미지를 입력으로 받을 수 있으며, 이를 통해 장거리 종속성을 모델링하면서도 선형(Linear) 계산 복잡성을 유지합니다. Spatial-Spectral Fusion Module(SSFM)을 통해 HSI의 공간적 및 스펙트럼 특징을 적응적으로 통합할 수 있습니다. 이러한 구조는 공간적 및 스펙트럼 정보를 각각 추출하여 분류를 위한 더 많은 discriminative(spatial과 spectral 모두의) 특징을 캡처할 수 있도록 도와줍니다.

- **Performance Highlights**: 연구 결과는 제안된 MambaHSI 모델이 CNN 및 Transformer 기반의 최첨단 HSI 분류 모델을 초월하는 것은 물론, 다양한 실제 HSI 데이터셋에서 우수한 성능을 발휘한다는 것을 보여줍니다. MambaHSI는 공간적 및 스펙트럼 정보의 통합을 통해 더욱 정확한 HSI 분류를 가능하게 합니다. 이러한 연구는 Mamba 모델이 차세대 HSI 모델의 백본(backbone)으로서 큰 잠재력을 가지고 있음을 알리고 있습니다.



### Multi-Context Temporal Consistent Modeling for Referring Video Object Segmentation (https://arxiv.org/abs/2501.04939)
- **What's New**: 본 논문에서는 비디오 내 객체를 주어진 텍스트 설명에 따라 분할하는 Referring Video Object Segmentation(RVOS) 문제를 음식하며, Multi-context Temporal Consistency Module(MTCM)을 제안합니다. MTCM은 'Aligner'와 'Multi-Context Enhancer(MCE)'로 이루어져 있으며, 이 모듈은 쿼리의 일관성을 높이고 복잡한 맥락을 고려하여 성능을 향상시킵니다. 이를 통해 텍스트와 객체 간의 관계를 효과적으로 모델링할 수 있게 됩니다.

- **Technical Details**: MTCM은 변환기(transformer) 아키텍처에 적용되어 다양한 모델의 시간적 모델링을 개선하는 데 기여합니다. Aligner는 쿼리를 정리하고 불필요한 정보를 제거하여 쿼리 일관성을 강화합니다. MCE는 로컬 및 글로벌 문맥을 반영하여 임시적인 행동을 이해하고 목표를 예측하기 위해 강화된 쿼리와 텍스트를 비교합니다.

- **Performance Highlights**: MTCM은 네 가지 모델에 적용되어 성능을 향상시켰으며, 특히 MeViS 데이터셋에서 47.6 J&F를 기록하였습니다. 이처럼 MTCM 모듈은 다양한 RVOS 모델에 유용하게 적용될 수 있으며, 쿼리의 일관성을 높이고 적절한 객체 선택을 위한 자세한 정보를 제공합니다.



### Plug-and-Play DISep: Separating Dense Instances for Scene-to-Pixel Weakly-Supervised Change Detection in High-Resolution Remote Sensing Images (https://arxiv.org/abs/2501.04934)
Comments:
          Accepted by ISPRS Journal of Photogrammetry and Remote Sensing

- **What's New**: 본 연구에서는 약한 감독 학습(Weakly-Supervised Change Detection, WSCD)에서의 'instance lumping' 문제를 해결하기 위한 Dense Instance Separation (DISep) 방법을 제안합니다. 이는 밀집된 변화 인스턴스 시나리오에서 발생하는 잘못된 픽셀 식별 문제를 완화함으로써, 변화 탐지의 정확성을 높입니다. DISep는 높은 패스 클래스 활성화 맵(High-pass Class Activation Maps)을 활용하여 인스턴스 후보 지역을 찾고, 변경된 픽셀을 다양한 ID로 그룹화한 후, 분리 손실(Separation Loss)을 통해 인스턴스 내 픽셀 일관성을 강화합니다.

- **Technical Details**: DISep는 세 가지 단계의 반복 학습 프로세스를 포함합니다: 1) 인스턴스 로컬라이제이션: Class Activation Maps를 사용하여 변경된 픽셀의 후보 영역을 찾습니다. 2) 인스턴스 리트리벌: 연결성 검색을 통해 이 변경된 픽셀을 그룹화하고 ID를 부여해 인스턴스를 구분합니다. 3) 인스턴스 분리: 픽셀의 임베딩 공간에서 인스턴스 간의 일관성을 확보하기 위해 분리 손실을 도입합니다. 이 과정은 기존 WSCD 방법에 최소한의 추가 비용으로 통합될 수 있습니다.

- **Performance Highlights**: DISep는 세 가지 변환기 기반(Transformer-based) 및 네 가지 합성곱 신경망(ConvNet-based) 방법을 LEVIR-CD, WHU-CD, DSIFN-CD, SYSU-CD, CDD 데이터셋에서 평가하여 최신 성능을 달성했습니다. 이 방법은 기존 WSCD 방법의 성능을 일관되게 향상시키며, 변화 탐지의 정확한 정량화를 가능하게 합니다. 또한 DISep는 완전 감독 변화 탐지 방법에도 적용할 수 있어 그 활용 가능성을 높입니다.



### Image2CADSeq: Computer-Aided Design Sequence and Knowledge Inference from Product Images (https://arxiv.org/abs/2501.04928)
Comments:
          20 pages, 10 figures, and 6 tables

- **What's New**: 이 연구는 2D 이미지로부터 CAD 시퀀스를 직접 생성하는 새로운 데이터 기반 접근 방식을 제안합니다. 특히, Image2CADSeq 신경망 모델을 통해 이미지 입력을 기반으로 CAD 모델을 역설계할 수 있도록 하여 설계 과정의 이해를 심화합니다. 특정 데이터 합성 파이프라인을 사용하여 CAD 모델링에 대한 새로운 인사이트와 작동 가능성을 제공합니다.

- **Technical Details**: 제안된 접근 방식은 Target-Embedding Variational Autoencoder (TEVAE) 아키텍처를 사용하여 단일 이미지 입력으로 CAD 작업 시퀀스를 예측합니다. 이 모델은 3D 모델로 변환할 수 있는 CAD 시퀀스를 생성하고, CAD 모델의 구체적 과정과 변경 가능성을 분리하여 제공합니다. 평가 프레임워크는 CAD 시퀀스, 3D 모델 및 해당 이미지를 포함하여 다단계로 모델의 예측 성능을 분석합니다.

- **Performance Highlights**: 실험 결과, Image2CADSeq 모델은 2D 이미지 데이터에서 CAD 시퀀스를 생성하는 데 있어 매우 유망한 성능을 보였습니다. 이 접근 방식은 CAD 모델 재구성의 접근성을 향상시켜 경험이 적은 설계자도 설계 과정에 적극적으로 참여할 수 있도록 할 잠재력을 가지고 있습니다. 연구진은 이 모델이 CAD 시스템 혁신에 기여할 것으로 기대하고 있으며, 이는 최종 사용자 참여를 촉진하는 독특한 경로를 제공할 것입니다.



### From Mesh Completion to AI Designed Crown (https://arxiv.org/abs/2501.04914)
- **What's New**: 이 논문에서는 Dental Mesh Completion (DMC)이라는 새로운 end-to-end 딥러닝 접근법을 제안합니다. DMC는 포인트 클라우드(context) 정보를 기반으로 치관(mes) 메시를 생성하여 성공적인 치관 디자인을 자동화합니다. 전통적인 방법의 노동 집약적인 과정을 줄이고, 더 나아가 치료의 품질을 보장할 수 있는 가능성을 보여줍니다.

- **Technical Details**: DMC는 포인트 클라우드를 입력으로 받아 특징 벡터(feature vectors)를 추출하는 기능 추출기(feature extractor)를 사용합니다. 이후 이 특징 벡터는 변환기(transformer)에 전달되어 부족한 영역(crown)의 새로운 특징 벡터를 예측합니다. 마지막으로, 차별화된(point-to-mesh) 레이어를 통해 치관의 표면 메시를 재구성합니다. DMC는 기존의 그래프 기반 합성곱 신경망과 비교되며, 효율성과 품질 측면에서 개선된 결과를 보여줍니다.

- **Performance Highlights**: DMC의 실험 결과는 평균 0.062의 Chamfer Distance (CD) 메트릭을 달성하여 기존 방법들보다 우수한 성능을 입증합니다. 이 연구는 모든 치아 위치에 대해 치관 메시를 생성할 수 있는 최초의 end-to-end 네트워크를 제시하며, 고품질 표면 메시 생성을 위한 새로운 가능성을 열었습니다. 따라서 치과 치료 분야의 자동화와 정확도를 높이는 중요한 진전을 이룬 것으로 평가됩니다.



### A Machine Learning Model for Crowd Density Classification in Hajj Video Frames (https://arxiv.org/abs/2501.04911)
- **What's New**: 이번 연구는 Hajj와 Umrah의 대규모 인원 관리를 위한 새로운 기계학습 모델을 제안합니다. 이 모델은 비디오 프레임에서 군중 밀도를 세 가지 수준으로 분류하며, 특히 'very dense crowd'를 실시간으로 경고하는 플래시 적색등을 활용합니다. 기존의 연구는 비정상적인 행동 탐지에 중점을 두었으나, 이번 연구는 위험한 군중 상황 관리에 초점을 맞추었습니다.

- **Technical Details**: 제안된 모델은 Local Binary Pattern (LBP) 텍스처 분석을 통합하여 군중 밀도 수준을 차별화하는 특징 추출을 향상시킵니다. 모델은 에지 밀도(edge density) 및 지역 기반(area-based) 특성과 함께 작동하여 높은 정확도를 유지합니다. Hajj 동안의 다양한 주요 위치에서 18개의 비디오로 구성된 KAU-Smart Crowd 'HAJJv2' 데이터셋을 통해 검증되었습니다.

- **Performance Highlights**: 모델은 87%의 정확도와 2.14%의 오류 비율(오분류율)을 기록하며, 다양한 군중 상태를 효과적으로 탐지하고 분류할 수 있음을 입증합니다. 이는 Hajj와 같은 대규모 이벤트에서 군중 관리를 향상시키고 안전성을 높이는 데 기여합니다.



### Topological Classification of points in $Z^2$ by using Topological Numbers for $2$D discrete binary images (https://arxiv.org/abs/2501.04878)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2410.21588

- **What's New**: 이 논문에서는 2D 이산 이진 이미지의 점들에 대한 위상 분류(topological classification)를 제안합니다. 제안된 분류는 위상 숫자(topological numbers)의 값을 기반으로 하며, 고립점(isolated point), 내부점(interior point), 단순점(simple point), 곡선점(curve point), 3개의 곡선의 교차점 및 4개의 곡선의 교차점을 포함한 6가지 점 클래스가 있습니다. 각 클래스의 구성 가능성 또한 제공됩니다.

- **Technical Details**: 2D 이산 이진 이미지는 검정 점(객체)과 흰 점(보완물)으로 구성된 격자 형태의 픽셀로 이루어져 있습니다. 이 논문에서는 디지털 위상(Digital Topology) 프레임워크를 기반으로 하여 특정 점을 반복적으로 삭제하여 이미지의 위상을 보존하는 단순점을 정의합니다. 이를 통해 단순점 이외의 곡선점(surfacce point) 등 다양한 위상적 구성 요소를 탐지할 수 있습니다.

- **Performance Highlights**: 이 연구는 2D 이산 이진 이미지의 위상 분류에 대한 혁신적인 접근 방식을 제공하며, 각 점 클래스에 대한 256*256 가능한 구성의 완전한 분류를 포함하고 있습니다. 이 연구는 지문(image fingerprint) 및 의학적 혈관 조영 이미지와 같은 다양한 응용 분야에서 유용한 특징을 추출할 가능성을 열어줍니다. 마지막으로, 2D 이진 이미지에서 간단한 점을 어떻게 정의하고 감지하는지를 명확하게 설명합니다.



### Back Home: A Machine Learning Approach to Seashell Classification and Ecosystem Restoration (https://arxiv.org/abs/2501.04873)
- **What's New**: 이번 연구에서는 코스타리카의 해양 생태계를 보호하기 위해, 압수된 조개껍데기의 기원을 식별하는 데 특화된 합성곱 신경망(CNN) 모델을 개발했습니다. 총 19,000장의 이미지를 기반으로 한 데이터 세트를 구축하고, 85% 이상의 분류 정확도를 달성했습니다. 이 모델은 사용자 친화적인 애플리케이션으로 통합되어 현재 36,000개 이상의 조개껍데기를 분류하며, 이미지당 3초 이내의 실시간 결과를 제공합니다.

- **Technical Details**: 본 연구에서 사용된 데이터 세트는 10개월에 걸쳐 수집된 19,058장의 이미지로 구성되었으며, 카리브해와 태평양의 516종 조개껍데사가 포함되어 있습니다. ConvNext 아키텍처를 사용하여 조개껍데기의 기원을 분류할 수 있는 모델을 구축하였으며, 데이터 증강 기법을 적용하여 다양한 조개껍데기의 변화를 다루기 위한 성능을 향상했습니다. 모델 학습 시 하이퍼파라미터 조정을 위해 SGD 옵티마이저와 학습률 감쇠 기법을 사용했습니다.

- **Performance Highlights**: 최종 모델은 ConvNext 아키텍처를 통해 우수한 성능을 발휘했으며, 자원 제한 환경에서도 효과적으로 배포될 수 있도록 설계되었습니다. CNN 모델의 엄청난 성능 덕분에 조개껍데기의 생태계 기원을 정확히 식별할 수 있게 되었고, 실시간 분류 기능을 통해 사용자의 신뢰성을 높였습니다. 분류 결과에 대한 검증을 위해 70%의 데이터를 훈련 세트로, 15%는 검증 세트, 나머지 15%는 테스트 세트로 사용하여 성능을 평가했습니다.



### LayerMix: Enhanced Data Augmentation through Fractal Integration for Robust Deep Learning (https://arxiv.org/abs/2501.04861)
- **What's New**: 이 논문에서는 LayerMix라는 혁신적인 데이터 증강 방법을 소개합니다. 이 방법은 구조적인 프랙탈 기반 이미지를 합성하여 모델의 강건성을 체계적으로 개선합니다. 전통적인 데이터 증강 기법과 달리, LayerMix는 원본 이미지의 의미론적 일관성을 유지하면서도 제어된 변동성을 도입하기 위해 구조화된 혼합 파이프라인을 사용합니다.

- **Technical Details**: LayerMix의 핵심은 프랙탈(Fractal)을 사용하여 훈련 데이터셋의 복잡성을 통합하는 것입니다. 이 구조화된 혼합 파이프라인은 훈련 샘플을 결합하여 새로운 고유 샘플을 생성하며, 이 과정에서 기존의 신경망 모델에 피해를 주지 않도록 설계되었습니다. 실험 결과, CIFAR-10, CIFAR-100, ImageNet-200, ImageNet-1K와 같은 여러 벤치마크 데이터셋에서 최신 성능을 달성하며, 모델의 일반화 및 적대적인 강건성을 크게 개선했습니다.

- **Performance Highlights**: LayerMix는 분류 정확도에서 우수한 성능을 발휘하며, 자연 이미지 손상에 대한 내성, 적대적 공격에 대한 강건성, 모델 보정 및 예측 일관성 향상과 같은 중요한 기계 학습 안전 매트릭스를 개선했습니다. 이러한 결과는 LayerMix가 인공지능 시스템의 신뢰성과 적응성을 높이는 데 중요한 진전을 나타냅니다.



### EDMB: Edge Detector with Mamba (https://arxiv.org/abs/2501.04846)
- **What's New**: 이번 논문에서는 Mamba를 활용하여 고품질의 다중 잠재 경계를 효율적으로 생성하는 새로운 엣지 디텍터 EDMB를 제안합니다. EDMB는 글로벌-로컬 아키텍처를 활용하여 전역 정보와 세밀한 단서를 동시에 포착할 수 있는 능력을 가지고 있습니다. 특히, EDMB는 Evidence Lower Bound 손실을 도입하여 다중 레이블 데이터에 대한 의존성을 줄이며, 일반적인 단일 레이블 데이터에서도 효과적으로 적용 가능합니다.

- **Technical Details**: EDMB는 고해상도 기능 인코더와 Mamba 인코더를 조합하여 이미지를 기능으로 인코딩합니다. Mamba는 두 가지 인코더, 즉 글로벌 Mamba 인코더와 세밀한 Mamba 인코더를 사용하여 코드를 구성합니다. 또한, 학습 가능한 가우시안 분포 디코더가 디자인되어 글로벌 및 세밀한 기능을 융합하여 다중 그레인 엣지를 샘플링하는 방식을 사용합니다.

- **Performance Highlights**: 제안된 EDMB는 다중 라벨 데이터셋인 BSDS500에서 경쟁력 있는 성능을 보여주며, 단일 그레인 ODS는 0.837, 다중 그레인 ODS는 0.851을 기록했습니다. 이는 다중 스케일 테스트나 추가 PASCAL-VOC 데이터 없이 이루어졌습니다. EDMB는 NYUDv2 및 BIPED와 같은 단일 레이블 데이터셋에도 확장 가능하여, 다양한 다운스트림 작업을 위한 유연성을 제공합니다.



### Towards Generalizable Trajectory Prediction Using Dual-Level Representation Learning And Adaptive Prompting (https://arxiv.org/abs/2501.04815)
- **What's New**: 이 논문에서는 차량 궤적 예측 모델의 일반화 능력, 예측 불확실성, 그리고 복잡한 상호작용 처리의 문제를 해결하기 위해 'Perceiver with Register queries (PerReg+)'라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 두 가지 수준의 표현 학습(Dual-Level Representation Learning)과 향상된 다중 모드 처리(Enhanced Multimodality)를 통해 기존의 한계를 극복하고자 합니다. 이를 통해 다양한 데이터셋에서도 성능을 향상시키는 데 중점을 두고 있습니다.

- **Technical Details**: 제안된 PerReg+ 모델은 자기 증류(Self-Distillation)와 마스크 재구성(Masked Reconstruction)을 활용하여 전역 맥락(global context)과 세부 정보를 효과적으로 포착합니다. 또한 레지스터 기반 쿼리(register-based queries)와 사전 훈련(pretraining)을 통해 다중 모드를 향상시키고, 세분화된 트랙 및 차선 세그먼트의 재구성을 통해 효율성을 높입니다. 마지막으로, 미세 조정(fine-tuning) 중에 적응형 프롬프트 조정(Adaptive Prompt Tuning)을 적용하여 주 아키텍처는 동결하고 소수의 프롬프트를 최적화합니다.

- **Performance Highlights**: PerReg+는 nuScenes, Argoverse 2, Waymo Open Motion Dataset에서 새로운 최첨단 성능을 기록했습니다. 특히, 사전 훈련된 모델은 소규모 데이터셋에서 6.8%의 예측 오류를 줄이며, 다중 데이터셋 훈련을 통해 일반화가 개선되었습니다. 또한, 교차 도메인 테스트에서 PerReg+는 비사전 훈련 변형에 비해 B-FDE를 11.8% 감소시키는 성과를 올렸습니다.



### Leveraging Registers in Vision Transformers for Robust Adaptation (https://arxiv.org/abs/2501.04784)
Comments:
          Accepted at ICASSP 2025

- **What's New**: 본 논문에서는 Vision Transformers (ViTs)의 register 토큰임베딩을 활용한 새로운 접근 방식을 제안합니다. 이 register는 높은 노름을 가지는 패치 토큰의 영향을 줄이면서도 글로벌 이미지 정보를 포착하는 데 기여합니다. OOD(Out-of-Distribution) 상황에서 일반화 성능과 이상 탐지 성능을 개선하기 위한 새로운 방법론을 다루고 있습니다.

- **Technical Details**: ViTs는 이미지를 패치로 나누어 토큰으로 처리하며, 특수한 [CLS] 토큰을 사용하여 정보 집계를 수행합니다. 본 연구는 [CLS] 토큰과 평균 풀링한 register 토큰 임베딩을 결합하여 더 견고한 표현을 형성하는 방법을 도입하고, 다양한 ViT 아키텍처에서 검증한 결과를 보고합니다. 이를 통해 OOD 정확도를 2-4% 향상시키고, 이상 탐지 시 거짓 긍정률을 2-3% 줄이는 성과를 달성하였습니다.

- **Performance Highlights**: 우리는 비지도 객체 발견 작업에서 활용된 register의 효과를 보여주며, OOD 일반화 및 이상 탐지 성능 향상을 증명했습니다. 실험 결과, [CLS]와 register 임베딩의 결합이 기존 방법보다 더욱 향상된 성능을 발휘함을 확인했습니다. 이 방법은 추가적인 계산 비용 없이도 실질적인 성과를 내며, 다양한 ViT 백본 모델에서 일관된 개선 효과를 보여주었습니다.



### GaussianVideo: Efficient Video Representation via Hierarchical Gaussian Splatting (https://arxiv.org/abs/2501.04782)
Comments:
          10 pages, 10 figures

- **What's New**: 이 논문에서는 동적 비디오 장면에 대한 효율적인 신경 표현을 제안합니다. 새로운 메서드는 3D Gaussian splatting과 지속적인 카메라 모션 모델링을 결합하여 메모리 사용량과 훈련 시간을 줄이면서도 높은 품질의 렌더링을 목표로 합니다. 기존 방법들이 직면한 고해상도와 긴 훈련 시간 문제를 해결하는 데 중점을 두고 있으며, 강력한 시점 모형과 함께 계층적 학습 전략을 도입하여 복잡한 동적 장면을 효과적으로 캡처합니다.

- **Technical Details**: 제안된 방법에서는 B-스플라인 기반의 모션 표현과 Neural ODEs를 활용하여 매끄러운 카메라 경로를 학습하고, Gaussians의 명시적인 3D 장면 표현을 유지합니다. 이 방식은 공간 및 시간 특성을 점진적으로 정제하는 계층적 학습 전략을 사용하여 재구성 품질을 개선하고 수렴 속도를 빨라지게 합니다. 또한, 사전 계산된 카메라 매개변수에 의존하지 않고 다양한 캡처 설정에 적응할 수 있습니다.

- **Performance Highlights**: 실험 결과는 제안된 계층적 학습과 카메라 모션 모델링이 강한 시간 일관성을 유지하면서 복잡한 동적 장면을 효과적으로 캡처할 수 있음을 보여줍니다. 이 방법은 높은 동작과 낮은 동작 시나리오 모두에서 다양한 비디오 데이터셋에서 최첨단 성능을 달성하였으며, 메모리 효율적이고 훈련 시간을 단축시킵니다. 결과적으로, 기존의 접근 방식을 초월하는 재구성 품질과 시간 일관성을 효과적으로 처리합니다.



### TREAD: Token Routing for Efficient Architecture-agnostic Diffusion Training (https://arxiv.org/abs/2501.04765)
- **What's New**: 이번 연구는 기존 diffusion model에서 발생하는 샘플 비효율성과 높은 훈련 비용을 해결하기 위해, predefined routes를 통해 토큰 정보를 저장하고 모델의 깊은 층으로 다시 도입하는 방법을 제안합니다. 이러한 접근 방식은 토큰을 완전히 버리는 대신, 더 깊은 층으로 연결할 수 있도록 하여 훈련 효율성을 높입니다. 또한, 여러 경로를 결합하고, 이를 고려한 보조 손실(auxiliary loss)을 도입하여 모델의 성능을 개선하는 방향으로 진행하였습니다.

- **Technical Details**: 기존의 Transformer 아키텍처는 입력 길이에 대해 제곱 복잡성을 가지기 때문에 컴퓨팅 비용이 높아지는 문제가 있었습니다. 이번 연구에서는 routing 기법을 통해 정보 흐름을 제한하고, 복잡성을 줄이면서 효율성을 높이는 방법을 모색했습니다. 이때 routing은 학습 동안에만 작동하며, 필요한 계산을 생략하지 않고 정보를 한 층에서 다른 층으로 전달합니다.

- **Performance Highlights**: 우리의 방법은 ImageNet-1K 256 x 256 기준에서, DiT에 비해 9.55배의 수렴 속도 향상을 이룩했습니다. 또한, DiT-XL/2의 기준 FID 9.62보다 더 나은 FID 9.32를 41시간 내에 달성했습니다. 이러한 성과는 diffusion models의 훈련을 민주화하는 데 기여할 수 있을 것입니다.



### Video Summarisation with Incident and Context Information using Generative AI (https://arxiv.org/abs/2501.04764)
- **What's New**: 이 논문은 비디오 콘텐츠 분석을 용이하게 하기 위해 생성 인공지능(GenAI)을 활용한 새로운 접근 방식을 제시합니다. 기존의 일반적인 요약 방법을 넘어, 사용자가 정의한 쿼리에 맞춤화된 텍스트 요약을 제공하여 방대한 비디오 데이터셋 속에서 관련 정보를 추출하고 분석의 정확성과 효율성을 향상시키고자 합니다. 또한, YOLO-V8과 Gemini를 결합하여 객체 감지 및 비디오 분석을 수행하며, CCTV 영상에서 필요한 사건을 신속하게 검토할 수 있는 기능을 제공합니다.

- **Technical Details**: 이 방법은 비디오 스트림을 개별 프레임으로 분할한 후, 사전 훈련된 YOLO 모델을 통해 사람이나 원하는 객체를 감지합니다. 감지된 프레임은 Gemini Pro Vision과 같은 GenAI 모델로 전달되어 맞춤형 프롬프트에 따라 구체적인 출력을 생성합니다. 프롬프트의 조정 가능성 및 온도 변수의 설정을 통해 모델의 반응을 정교하게 다듬어 최적의 결과를 이끌어낼 수 있습니다.

- **Performance Highlights**: 정량적 평가에서는 유사도가 72.8%로 나타났으며, 질적 평가에서는 정확도가 85%로 평가되었습니다. 이러한 결과는 제안된 방법이 영상 데이터 분석에서 효과적으로 작동할 수 있음을 보여줍니다. 이는 사용자가 원하는 특정 프레임을 신속하게 파악하고 검증할 수 있는 유용한 도구로 자리잡을 수 있음을 의미합니다.



### Efficient License Plate Recognition in Videos Using Visual Rhythm and Accumulative Line Analysis (https://arxiv.org/abs/2501.04750)
Comments:
          Accepted for presentation at the Conference on Graphics, Patterns and Images (SIBGRAPI) 2024

- **What's New**: 본 논문은 비디오 기반의 자동 번호판 인식(Automatic License Plate Recognition, ALPR) 시스템의 효율성을 크게 향상시키는 두 가지 새로운 방법을 제안합니다. 전통적인 시스템은 여러 프레임에 의존하여 높은 컴퓨팅 자원을 필요로 하지만, 제안된 방법은 차량당 하나의 프레임만을 사용하여 번호판 문자를 인식합니다. 첫 번째 방법인 Visual Rhythm (VR)은 비디오에서 시간-공간 이미지를 생성하고, 두 번째 방법인 Accumulative Line Analysis (ALA)는 실시간 처리를 위한 신개념 알고리즘입니다.

- **Technical Details**: 제안된 방법들은 YOLO를 활용하여 프레임 내에서 번호판을 탐지하고, Convolutional Neural Network (CNN)을 통해 광학 문자 인식(Optical Character Recognition, OCR)을 수행하여 텍스트 정보를 추출합니다. Visual Rhythm 방법은 비디오에서 여러 프레임을 단일 시간-공간 이미지로 변환하여 차량이 미리 정의된 선을 교차하는 순간을 효율적으로 포착합니다. ALA는 프레임 별로 미리 정의된 수평선에 집중하여 이 선을 가로지르는 차량을 추적하는 방식으로 동작합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법들은 전통적인 프레임별 접근 방식과 동등한 결과를 달성하며, 처리 속도가 세 배 더 빠른 것으로 나타났습니다. 이는 ALPR 시스템의 실시간 작동을 가능하게 하여 다양한 교통 상황에서의 적용성을 높입니다. 새로운 알고리즘들은 비디오 기반 ALPR 시스템의 계산적 오버헤드를 크게 줄여주며, 라이센스 플레이 인식의 효율성을 증가시킵니다.



### The GAN is dead; long live the GAN! A Modern GAN Baselin (https://arxiv.org/abs/2501.05441)
Comments:
          Accepted to NeurIPS 2024. Code available at this https URL

- **What's New**: 이 논문에서는 GAN(Generative Adversarial Networks)의 훈련에서 발생하는 어려움에 대한 일반적인 주장을 반박하고, 기존의 경험적 트릭을 배제한 현대적인 GAN 표준인 R3GAN을 제시합니다. 이를 위해, 새로운 정규화된 상대적 GAN 손실을 도출하여 모드 드롭(mode dropping)과 비수렴(non-convergence) 문제를 해결합니다. 이 접근 방식은 GAN 훈련의 안정성을 향상시키며, 최신 아키텍처와 통합하여 더 나은 성능을 달성할 수 있음을 보여줍니다.

- **Technical Details**: 논문은 GAN의 목표를 안정성과 다양성이라는 두 가지 도전 과제로 설정합니다. 새로운 손실 함수는 제로 중심의 기울기 패널티를 도입하여 안정성을 높이고, 이론적으로도 지역 수렴(local convergence)을 보장합니다. 기존 GAN 구조의 한계를 극복하기 위해 새로운 현대적 아키텍처를 설계하였고, 스타일 GAN(StyleGAN)에서 필수적인 요소를 남기고 불필요한 기능을 제거하여 간소화된 구조를 제안했습니다.

- **Performance Highlights**: R3GAN은 FFHQ, ImageNet, CIFAR 및 Stacked MNIST 데이터셋에서 StyleGAN2보다 우수한 성능을 발휘하며, 동시대의 최신 GAN 및 diffusion 모델에 비해 경쟁력 있는 성과를 보여줍니다. 이러한 결과는 단순함에도 불구하고 성능이 향상된 GAN 구조를 통해 가능해졌음을 의미합니다. 이 연구는 GAN의 훈련을 더 간편하게 만들고, 최신 기술과의 통합을 통해 GAN의 진전을 도모합니다.



### Flatland Vision (https://arxiv.org/abs/2501.05429)
- **What's New**: 본 논문에서는 두 개의 레이블이 있는 점 집합이 동일한 프로젝트 라인(projective line)에서 동일한 이미지로 투영(projection)될 수 있는 조건에 대한 완전한 답을 제공합니다. 이는 두 점 집합이 프로젝트 공간(projective space)의 공통 점 집합의 이미지인 경우에만 가능하다는 점을 강조합니다.

- **Technical Details**: 논문은 프로젝트 평면(projective planes)에서 두 점 집합의 투영 중심(projection centers) 위치에 대해 설명하고, 이러한 위치들이 공통 이미지를 생성할 수 있는 조건을 제시합니다. 이는 기하학적 구성이 어떻게 맥락에 따라 달라질 수 있는지를 보여주며, 이를 통해 프로젝트 평면에서의 해를 찾는 방법을 제시합니다.

- **Performance Highlights**: 저자들은 이 문제의 해를 찾는 과정을 통해 새로운 기하학적 통찰력(geometric insights)을 제공하며, 이는 나아가 변환(transformations) 및 투영 기하학(projection geometry)의 이해를 돕는 데 기여합니다. 이러한 결과들은 이론적 뿐만 아니라 적용 가능성(application possibilities) 또한 가지고 있습니다.



### Seeing Sound: Assembling Sounds from Visuals for Audio-to-Image Generation (https://arxiv.org/abs/2501.05413)
- **What's New**: 이 논문에서는 다양한 비디오에서 유래한 오디오-비주얼 쌍 대신, 고품질의 단일 모드 오디오 및 이미지 데이터셋을 활용해 고유한 오디오-비주얼 쌍을 생성할 수 있는 스케일러블한 이미지 음향화(image sonification) 프레임워크를 제안합니다. 이를 통해 현재의 생성 모델에서 사용할 수 있는 데이터의 규모와 품질, 다양성을 증가시킵니다. 이 방법은 최신 비전-언어 모델의 추론 능력을 활용하여 저품질 및 저다양성 데이터를 극복하고자 합니다. 또한, 이 방식을 통해 전통적인 데이터 수집 방식에서 발생하는 문제들을 해결할 수 있습니다.

- **Technical Details**: 연구진은 오디오와 이미지 간의 강력한 크로스 모달(cross-modal) 연관성을 가진 쌍을 구성하기 위해 기존에는 명시적인 그라운드 트루스(ground truth) 데이터에 의존했던 관행을 비판합니다. 대신, 다양한 고유 모드 데이터셋을 결합해 이미지에 오디오를 합성하는 새로운 접근 방식을 채택하였습니다. 이를 위해 사전 훈련된 비전-언어 모델을 사용하여 이미지와 오디오 간의 검색 과정을 개선하여 데이터 품질과 관련성을 향상시킵니다. 이 과정은 고유 모드에서 유래하는 다양한 데이터셋을 활용함으로써 수행됩니다.

- **Performance Highlights**: 이 논문에서는 제안된 오디오-이미지 생성 모델이 기존의 최첨단 모델들과 경쟁할 수 있는 성능을 발휘함을 보여줍니다. 여러 보조 지표에서 우수한 성과를 보였으며, 이는 고유한 오디오-비주얼 쌍의 생성 효과를 입증하는 결과입니다. 추가적으로, 다양한 청각적 특성들이 모델 내에서 나타나며, 음향믹스, 간섭, 볼륨 보정 및 잔향을 통한 공간 모델링 등의 흥미로운 기능을 개발했다는 점에서 연구의 의미가 큽니다.



### Optimized Sampling for Non-Line-of-Sight Imaging Using Modified Fast Fourier Transforms (https://arxiv.org/abs/2501.05244)
- **What's New**: 이번 연구에서는 Non-line-of-Sight (NLOS) 이미징 시스템에서 불균형 샘플링 문제를 해결하기 위한 새로운 접근법을 제안합니다. 기존 NLOS 이미징 시스템이 불필요하게 샘플링을 오버샘플링하고 있다는 것을 밝히며, 이를 통해 측정값을 압축하더라도 재구성 품질을 크게 희생하지 않고도 가능함을 설명했습니다. 이 연구는 Non-Uniform Fast Fourier Transform (NUFFT)와 Scaled Fast Fourier Transform (SFFT)을 결합하여 유연한 샘플링 스킴을 활용할 수 있게 합니다.

- **Technical Details**: NLOS 이미징 시스템은 다중 픽셀 탐지기 배열을 사용하여 데이터를 수집하지만, 이 배열이 불균형 샘플링을 야기합니다. 연구자는 파서(Field) 프레임워크를 활용하여 데이터를 재구성하는 동안 불균형 샘플링을 허용하고, 불규칙하게 샘플링된 릴레이 표면에서 얻은 희소 측정값들로부터 재구성을 수행하는 방법을 모색했습니다. 이 과정에서 NUFFT를 사용하여 무작위 형태의 릴레이 표면의 샘플로부터 직접적으로 재구성할 수 있도록 했습니다.

- **Performance Highlights**: 제안된 알고리즘들은 FFT 기반 방법의 계산 복잡성을 유지하면서도 실제 NLOS 이미징 응용 분야에 대한 확장성을 보장합니다. 특히, SFFT를 통해 더 큰 볼륨을 재구성할 수 있으면서도 메모리에 저장되는 샘플 수는 증가하지 않습니다. 이는 실질적인 NLOS 이미징의 배치 문제를 해결하고, 더 나은 성능을 제공함으로써 다양한 응용 가능성을 열어줍니다.



### Is Your Autonomous Vehicle Safe? Understanding the Threat of Electromagnetic Signal Injection Attacks on Traffic Scene Perception (https://arxiv.org/abs/2501.05239)
Comments:
          To appear in AAAI 2025

- **What's New**: 이 논문에서는 Electromagnetic Signal Injection Attack (ESIA)에 대한 새로운 시뮬레이션 방법을 개발하고, 이 공격이 자율 주행차의 AI 모델에 미치는 영향을 분석합니다. 연구에서는 ESIA의 공격을 통해 발생하는 취약점을 밝히고, 실 세계 데이터를 수집하기 어려웠던 문제를 해결하기 위한 모의 공격 데이터 세트를 생성합니다. 이를 통해 자율 주행 시스템의 안전성을 높이고 더 강력한 AI 모델을 개발하는 데 기여하고자 합니다.

- **Technical Details**: ESIA는 카메라 회로에 적대적 전자기 신호를 주입하여 이미지 왜곡을 유발합니다. 연구에서 제안하는 시뮬레이션 방법은 실제 공격에서 발생하는 패턴을 모방하며, RGB 이미지의 픽셀 값을 처리하여 잘못된 색상 해석이 발생하도록 합니다. 이 새로운 기법을 통해 다양한 주행 시나리오에서 AI 모델의 성능을 분석하고 평가합니다.

- **Performance Highlights**: 연구 결과는 자율 주행 기술의 안전성을 위협하는 ESIA의 심각성을 강조합니다. 시스템의 다양한 환경에서 ESIA 공격의 성능이 다르게 나타나는 것을 보여주며, 이는 AI 모델의 내구성과 신뢰성을 높이는 데 중요한 시사점을 제공합니다. 이러한 연구는 궁극적으로 자율 주행 차량의 안전하고 신뢰할 수 있는 운행을 위한 기초 자료로 사용될 수 있습니다.



### Centurio: On Drivers of Multilingual Ability of Large Vision-Language Mod (https://arxiv.org/abs/2501.05122)
- **What's New**: 이번 연구는 다국어를 위한 거대 비전-언어 모델(LVLM)을 훈련하는 데 있어 최적의 언어 분포를 조사하여, 이를 통해 비영어 사용자의 접근성을 높이는 방법을 모색합니다. 기존 모델들이 영어 데이터에 주로 의존해 다국어 성능이 제한되는 문제를 해결하기 위해, 43개 언어에서 다국어 데이터 분포의 효과를 실험하고 분석하였습니다. 연구 결과, 100개 언어를 동시에 사용하더라도 영어 성능이 크게 저하되지 않음을 보여 주며, 이는 다국어 성능 향상의 방향을 제시합니다.

- **Technical Details**: 연구는 LVLM 훈련 믹스의 언어 분포를 체계적으로 조사하며, 13개의 비전-언어 작업과 43개 언어를 대상으로 다양한 실험을 수행했습니다. 주된 연구 질문은 훈련 언어의 최적 수, 각 언어에 대한 데이터의 최적 분포, 이미지 내 비전-언어 이해를 향상시키기 위한 방법 등에 중점을 두고 있습니다. 이를 위해 Synthetic Multilingual Plot Question Answering(SMPQA)라는 새로운 데이터셋을 도입하여 다국어 OCR 능력을 평가합니다.

- **Performance Highlights**: Centurio라는 100개 언어를 지원하는 LVLM을 훈련하여 총 14개 작업에서 최첨단 성능을 달성하였습니다. 이 모델은 기존 인기 있는 다국어 LVLM 모델들과 비교하여 영어 및 자원 풍부 언어에서의 성능은 동등하거나 더 나은 결과를 보이며, 저자원 언어에서는 더욱 우수한 성능을 보여줍니다. 본 연구의 결과는 다국어 비전-언어 모델의 설계 및 훈련에 중요한 영향을 미칠 것으로 기대됩니다.



### Improving the U-Net Configuration for Automated Delineation of Head and Neck Cancer on MRI (https://arxiv.org/abs/2501.05120)
- **What's New**: 이 연구에서는 전통적인 U-Net 아키텍처를 바탕으로 두개와 경부 종양의 MRI에서의 자동 분할을 위한 새로운 방법을 제안합니다. 이 방법은 MICCAI HNTS-MRG 2024 챌린지의 맥락에서 개발되었으며, 기존의 시장에서 사용되는 아키텍처를 새로운 과제에 맞게 개선하는 것을 목표로 합니다. 연구 결과는 patch-wise normalization과 데이터 증대(data augmentation) 방법을 통한 성능 향상을 보여주고 있습니다.

- **Technical Details**: 제안된 네트워크는 전통적인 U-Net 구조를 따르며, 3D convolutional layer, instance normalization 및 ReLU 비선형 변환을 포함하는 여러 convolutional blocks로 구성됩니다. 인코더에서 다운샘플링을 통해 feature map의 수가 두 배로 증가하며, 디코더에서 업샘플링은 1x1x1 convolutional block과 nearest-neighbor interpolation을 사용하여 수행됩니다. 실험에서는 5개의 서로 다른 데이터 폴드(fold)를 나누어 교차 검증(cross-validation)을 실시하여 결과를 비교하였습니다.

- **Performance Highlights**: 연구에서 최적의 구성의 모델은 Task 1에서 0.749, Task 2에서 0.710의 Dice Similarity Coefficient (DSCagg)를 기록했습니다. 또한 다섯 개의 모델을 앙상블한 결과, 비공식 테스트 세트에서 Task 1에서 0.752, Task 2에서 0.718의 일관된 성능을 보였습니다. 최종적으로, 이 연구는 자동화된 MRI 분할의 정확도를 향상시키기 위한 몇 가지 중요한 구성 요소와 방법들을 제시합니다.



### A Flexible and Scalable Framework for Video Moment Search (https://arxiv.org/abs/2501.05072)
- **What's New**: 비디오 순간 검색(Video moment search) 분야는 사용자의 질의와 일치하는 비디오 내 관련 순간을 찾는 과정으로, 매우 중요합니다. 그러나 기존의 접근법은 단일 완벽한 매칭 순간을 가정하거나 효율적인 추론에 어려움을 겪으며, 장시간 비디오에 대한 한계가 있습니다. 이 논문에서는 비디오 길이에 관계없이 텍스트 질의와 맞는 순간의 순위 목록을 검색할 수 있는 유연하고 확장 가능한 프레임워크인 Rank Video Moment Retrieval (RVMR)을 제안합니다.

- **Technical Details**: 이 연구에서 제안하는 SPR(Segment-Proposal-Ranking) 프레임워크는 세 가지 독립적인 단계로 검색 프로세스를 단순화합니다: 세그먼트 검색(segment retrieval), 제안 생성(proposal generation), 그리고 순간 정제(moment refinement) 및 재정렬(re-ranking)입니다. 비디오는 균일한 길이의 세그먼트로 나누어져 사전 계산된 임베딩(embeddings)을 오프라인으로 색인화하며, 지속적으로 효율적 검색을 가능하게 합니다. 를 위해 세그먼트와 질의를 공통 피처 공간으로 투영하여 근사 최근 이웃(ANN) 검색을 수행합니다.

- **Performance Highlights**: TVR-Ranking 데이터셋에 대한 평가 결과, SPR 프레임워크는 계산 비용과 처리 시간을 크게 줄이며 최첨단 성능을 달성함을 보여줍니다. 특히, 2만 개의 비디오에 대한 사용자 질의 처리가 평균적으로 1초도 안 걸리며, 다양한 무관한 비디오가 추가되어도 안정성을 보여줍니다. SPR 프레임워크의 유연한 설계 덕분에 각 구성요소의 독립적인 개선이 가능하여 대규모 응용에 매우 적합합니다.



### A Scalable System for Visual Analysis of Ocean Data (https://arxiv.org/abs/2501.05009)
- **What's New**: 이번 논문에서는 해양 데이터 분석을 위한 확장 가능하고 상호작용이 가능한 시각화 시스템인 pyParaOcean을 소개합니다. 이 시스템은 고유의 모듈을 제공하여 난류 식별 및 염도 이동 추적과 같은 일반적인 해양학적 분석 작업을 지원하며, ParaView와의 원활한 통합을 통해 사용자의 편의성을 극대화합니다.

- **Technical Details**: pyParaOcean은 해양 모델에서 가져온 데이터를 사용하여 대규모 시각화를 가능하게 하며, 다양한 작업과 기능을 지원하는 점에서 중요한 기술적 혁신을 보여줍니다. 또한 Cinema Science 데이터베이스를 생성하여 분석 속도를 높이고 I/O 병목 현상을 해결하는 방법을 제시합니다. 서버-클라이언트 구조를 통해 대규모 데이터 크기에 대한 계산과 시각화를 확장할 수 있는 특징을 가지고 있습니다.

- **Performance Highlights**: Bay of Bengal(Bob) 지역에 대한 사례 연구를 통해 pyParaOcean의 유용성을 증명하며, 복잡한 메소스케일 구조를 추출하고 시각화하는 모듈에 대한 상세한 스케일링 연구 결과를 제공합니다. 이 시스템은 에디 및 표면 전선(complex mesoscale structures)을 시각화하는 데 효과적인 도구로 자리잡고 있습니다.



### AD-L-JEPA: Self-Supervised Spatial World Models with Joint Embedding Predictive Architecture for Autonomous Driving with LiDAR Data (https://arxiv.org/abs/2501.04969)
- **What's New**: 본 논문에서는 LiDAR 데이터에 대한 자율주행을 위해 AD-L-JEPA라는 새로운 자기 지도(Self-Supervised) 사전 학습 프레임워크를 제안합니다. 기존의 생성적(generative)이나 대조적(contrastive) 방법과는 달리, AD-L-JEPA는 Bird's Eye View (BEV) 임베딩을 예측하여 다양한 자율주행 장면의 특성을 표현합니다. 이 방법은 긍정적 및 부정적 쌍을 수작업으로 만들어야 했던 대조 학습의 필요성을 없애고, 구현이 간단하며 학습된 표현 품질을 향상시킵니다.

- **Technical Details**: AD-L-JEPA의 구조는 주어진 포인트 클라우드의 가시 부분을 기반으로 비가시 부분이 임베딩 공간에서 어떻게 나타나야 하는지를 예측하도록 네트워크를 학습하는 방식으로 설계되었습니다. 이 과정에서, 기존의 입력 포인트 클라우드를 수정된 BEV 기반 마스킹 기법을 사용하여 마스킹합니다. 마스킹된 포인트는 타겟 인코더(target encoder)에 전송되고, 비마스킹된 포인트는 컨텍스트 인코더(context encoder)에 전송됩니다. 이렇게 함으로써, 자율주행 장면의 높은 불확실성에 적응할 수 있는 지리적으로 및 의미적으로 합리적인 세계 모델을 학습합니다.

- **Performance Highlights**: 실험 결과, AD-L-JEPA는 LiDAR 3D 객체 탐지 및 연관 변환 학습과 같은 하위 작업에서 높은 정확도와 레이블 효율성을 보여주었습니다. AD-L-JEPA는 최신 기술 수준(SOTA)을 초과하는 성능을 발휘하며, Occupancy-MAE 및 ALSO와 같은 최근 제안된 방법들에 비해 우수한 것으로 평가되었습니다. 결과적으로 AD-L-JEPA는 자율주행 응용 프로그램에서 자기 지도 사전 학습을 위한 그럴듯한 접근법으로 작용하고 있습니다.



### A Steerable Deep Network for Model-Free Diffusion MRI Registration (https://arxiv.org/abs/2501.04794)
- **What's New**: 비선형 등록(nonrigid registration)은 의료 영상 분석에서 중요한 역할을 하지만, 확산 MRI(diffusion MRI)에서는 여전히 도전 과제로 남아 있습니다. 본 논문에서는 명시적인 재배향(reorientation) 없이 원시 확산 MRI 데이터의 비선형 등록을 위한 새로운 딥러닝 프레임워크(deep learning framework)를 제안합니다. 이전 방법들과는 달리, 우리는 위치 및 방향 공간에서의 등변적인 미분동형사상(equivariant diffeomorphism)으로 등록을 공식화합니다.

- **Technical Details**: 우리 방법의 핵심은 기하학적 속성을 유지하면서 속도 필드(velocity fields)를 생성하는 $	ext{SE}(3)$-등변 UNet입니다. 새로운 손실 함수(loss function)를 도입하여 Fourier 공간에서 최대 평균 불일치(maximum mean discrepancy)를 기반으로 하여, 이미지 간의 집합 평균 전파기(ensemble average propagators)를 암묵적으로 일치시킵니다. 이러한 접근 방식은 파생 표현(derived representations) 추정의 오버헤드를 우회하는 장점도 가지고 있습니다.

- **Performance Highlights**: Human Connectome Project의 확산 MRI 데이터에 대한 실험 결과는 최신 접근 방법들과 경쟁력 있는 성능을 보여주었습니다. 이는 기존의 방법들에 비해 데이터 기반(data-driven)이며 기하학적으로 인식되는 dMRI 등록의 기초를 확립하는 작업입니다. 따라서, 본 연구는 의료 영상 분석의 효율성을 높이는 데 기여할 것으로 기대됩니다.



### Topology-based deep-learning segmentation method for deep anterior lamellar keratoplasty (DALK) surgical guidance using M-mode OCT data (https://arxiv.org/abs/2501.04735)
- **What's New**: 이 논문에서는 Deep Anterior Lamellar Keratoplasty (DALK) 절차에 대한 새로운 접근법으로 topology-based deep-learning segmentation 방법을 제안합니다. 이는 기존 방법들이 신호의 잡음과 불안정성으로 인해 단면을 정확하게 감지하는 데 어려움을 겪는 문제를 해결하기 위함입니다. 제안된 방법은 수정된 U-Net 아키텍처와 topological loss function을 통합하여 실시간으로 더 정확한 각막층 추적을 가능하게 합니다.

- **Technical Details**: 사용된 데이터 세트는 in vivo 및 ex vivo에서 수집된 OCT 이미지를 포함합니다. 하이브리드 손실 함수는 Binary Cross Entropy (BCE)와 topological loss를 결합하여, 각막층의 고품질 세분화를 위해 geometric 및 shape constraints를 적용합니다. 수정된 U-Net 프레임워크는 더 작은 convolution 커널을 사용하여 높은 해상도의 이미지를 빠르게 처리하고, 학습 안정성과 정확성을 높이는 기술적 개선이 이루어졌습니다.

- **Performance Highlights**: 실험 결과, 제안한 방법(PM)은 기존 방법(CM) 대비 노이즈와 신호 변동에 강한 성능을 보입니다. 특히, segmentation 정확도에서 상관의 표면 및 Descemet 막의 추적 능력이 향상되었습니다. 우리는 또한 제안된 방법의 빠른 추론 시간을 제공하여, 실시간 수술 안내의 가능성을 평가했습니다.



### tCURLoRA: Tensor CUR Decomposition Based Low-Rank Parameter Adaptation and Its Application in Medical Image Segmentation (https://arxiv.org/abs/2501.02227)
- **What's New**: 이 논문에서는 tensor CUR 분해(tensor CUR decomposition)에 기반한 새로운 파라미터 효율적인 미세 조정(tuning) 방법인 tCURLoRA를 제안합니다. 기존의 PEFT 방법들이 보유한 한계를 극복하기 위해, 사전 학습된 가중치 행렬을 세 차원 텐서로 결합하여 고차원 특성과 다차원 상호작용을 좀 더 효과적으로 포착합니다.

- **Technical Details**: tCURLoRA는 주로 LoRA와 같은 행렬 분해 기반 PEFT 방법의 약점을 보완하며, 미세 조정 시에 오직 하위 차원 텐서 구성 요소(lower-order tensor components)만을 업데이트하여 계산 및 저장 비용을 줄입니다. 이는 특히 리소스가 제한된 환경에서도 실용적인 해결책이 됩니다.

- **Performance Highlights**: 실험 결과, tCURLoRA는 의료 이미지 분할(medical image segmentation) 과제에서 기존 PEFT 방법들보다 월등한 성능을 발휘하였습니다. 이는 본 방법이 고차원 텐서에 대한 자연스러운 표현을 통해 더 높은 품질의 특성을 파악할 수 있음을 나타냅니다.



New uploads on arXiv(cs.AI)

### Neuro-Symbolic AI in 2024: A Systematic Review (https://arxiv.org/abs/2501.05435)
Comments:
          19 pages

- **What's New**: 현재 인공지능(AI) 분야는 'Neuro-Symbolic AI'로 알려진 새로운 형태의 AI가 부상하고 있으며, 이는 전통적인 Symbolic AI와 Sub-Symbolic AI의 결합으로 특징지어집니다. 이 논문에서는 2020년부터 2024년 사이에 발표된 신뢰할 수 있는 연구 결과를 체계적으로 평가하여 이러한 영역에서 주요 개발과 응용, 그리고 방법론을 강조합니다. 이를 기반으로, Neuro-Symbolic AI 연구의 기초적인 논의와 중요성을 제시합니다.

- **Technical Details**: 연구는 PRISMA 방법론에 따라 진행되었으며, IEEE Explore, Google Scholar, arXiv, ACM, SpringerLink 등 여러 데이터베이스를 통해 2020~2024년에 발표된 동료 심사 논문을 대상으로 하였습니다. 1,428개의 초기 논문 중 167개가 연구 기준을 충족하여 자세히 분석되었으며, 연구는 학습 및 추론(63%), 논리 및 추론(35%), 지식 표현(44%)의 세 가지 주요 분야에 집중되고 있음을 발견했습니다. 특히 근본적인 연구 영역으로 지식 표현, 학습 및 추론, 설명 가능성 및 신뢰성, 논리 및 추론, 메타 인지가 정의되어 있습니다.

- **Performance Highlights**: 연구 결과, Neuro-Symbolic AI의 연구는 2020년 이래 급속도로 성장하고 있으며, 학습과 추론 분야에서 집중적인 노력이 이루어지고 있습니다. 하지만 설명 가능성, 신뢰성 및 메타 인지와 같은 분야는 여전히 많은 사례가 부족하여, 이러한 간극을 해결하려는 다학제적 연구의 필요성이 강조됩니다. 전반적으로 이 분야의 발전을 위해서는 더 스마트하고 신뢰할 수 있으며 맥락을 인식하는 AI 시스템을 위한 기초적인 연구가 중요하다는 결론에 이릅니다.



### Developing a Foundation of Vector Symbolic Architectures Using Category Theory (https://arxiv.org/abs/2501.05368)
Comments:
          13 pages, no figures, 2 tables, one appendix

- **What's New**: 이 논문은 Vector Symbolic Architectures (VSA)에 대한 카테고리 이론(Category Theory)의 첫 적용 시도를 제시합니다. VSAs는 인지 과학에서 신경 처리와 인간의 기호적 추론의 통합 필요성에서 나온 고차원 벡터 표현에 대한 대안적인 접근법입니다. 논문은 문헌 조사를 통해 VSAs와 카테고리 이론 간의 교차점이 부족하다는 점을 강조하고 있습니다.

- **Technical Details**: VSAs는 기호를 벡터로 인코딩하는 기초 벡터 공간의 구조로, 고유한 "의미"를 가진 벡터들을 비교하는 방식으로 작동합니다. 연구에 따르면 VSAs는 두 가지 주요 연산인 바인딩(binding)/언바인딩(unbinding)과 번들링(bundling)을 통해 더 복잡한 기호를 표현할 수 있습니다. 본 논문은 VSA에 바람직한 특성을 수집하고 이를 카테고리 관련 개념에 연결하는 방법을 제시하고 있습니다.

- **Performance Highlights**: VSA의 주요 이점은 정보 저장 능력에서 나타납니다. d 차원의 벡터 공간에서는 2^d 개의 고유(near-)직교 벡터를 생성할 수 있어 높은 수준의 잡음 저항을 제공합니다. 또한, VSAs는 생물학적 신경망 모델에서도 구현 가능하다는 점에서 인지 기능 특성을 설명하기 위한 유망한 기초로 자리잡고 있습니다.



### Search-o1: Agentic Search-Enhanced Large Reasoning Models (https://arxiv.org/abs/2501.05366)
- **What's New**: 이 논문은 기존의 대규모 추론 모델(LRM)에서 발생하는 지식 부족 문제를 해결하기 위해 	extbf{Search-o1}이라는 새로운 프레임워크를 소개합니다. Search-o1은 에이전트 기반 검색 증강 생성(RAG) 메커니즘과 문서 내 추 reasoning(Reason-in-Documents) 모듈을 결합하여 외부 지식을 동적으로 검색하고 통합할 수 있도록 합니다. 이러한 접근 방식은 LRMs의 신뢰성과 적용 가능성을 향상시켜 더 신뢰할 수 있는 지능형 시스템을 구축하는 데 기여합니다.

- **Technical Details**: Search-o1의 디자인은 두 가지 핵심 요소에 기반하고 있습니다. 첫째, 에이전트 기반 RAG 메커니즘을 통해 LRM은 지식 부족 상태에서 적절한 외부 지식을 검색할 수 있도록 유도됩니다. 둘째, 별도의 Reason-in-Documents 모듈은 검색된 문서의 정보를 심층적으로 분석하여 원래의 추론 체계에 통합됩니다. 이러한 두 가지 측면이 상호 작용하여 LRM의 추론 과정을 강화하고 논리적 흐름을 유지할 수 있도록 합니다.

- **Performance Highlights**: 다양한 과학, 수학, 코딩 분야의 복잡한 추론 작업과 여섯 가지 공개 도메인 질문 응답(QA) 벤치마크에서 Search-o1의 강력한 성능이 입증되었습니다. Search-o1은 기존의 LRMs에 비해 신뢰성과 효율성을 크게 향상시켰으며, 이론적 및 정량적으로 검증된 결과는 LRM의 신뢰할 수 있는 추론을 위한 실질적인 지침을 제공합니다.



### Off-Policy Evaluation and Counterfactual Methods in Dynamic Auction Environments (https://arxiv.org/abs/2501.05278)
Comments:
          9 pages, 15 figures, IEEE format

- **What's New**: 본 연구에서는 자원 Allocation이라는 맥락에서 Off-Policy Evaluation (OPE) 기법의 활용을 탐구합니다. OPE는 실험 비용을 절감하며 신규 정책 평가를 수행할 수 있도록 하여 의사 결정 과정의 신뢰도를 높이고, 정책 선택 및 최적화를 가속화합니다. 특히, 동적 경매 환경에서 정책의 성과를 신속하게 평가하는 방법을 제안합니다. 이를 통해 정책 테스트에 필요한 시간 및 자원을 줄이는 데 기여할 수 있습니다.

- **Technical Details**: OPE는 역사적 데이터를 기반으로 새로운 정책의 성과를 추정하는데 사용됩니다. 이 연구에서는 에이전트가 자원을 획득하기 위한 지불 정책을 설정하고, 이를 통해 경매의 성과를 평가하는 구조를 가지고 있습니다. 행동 공간은 지불 정책으로 설정되며, OPE 방법을 통해 지불 정책의 성공 여부를 평가합니다. 여러 가지 평가자를 통해 지속적 및 비지속적 정책을 고려하여 성과를 검토합니다.

- **Performance Highlights**: 이 연구에서 제안된 접근 방식은 다양한 새 지불 정책을 평가할 수 있는 가능성을 탐색합니다. A/B 테스트를 통한 기존 정책과의 비교를 통해 데이터 기반으로 정책의 성과를 예측할 수 있습니다. OPE 방식을 통해 높은 신뢰도로 새로운 정책을 실험하기 전, 가장 유망한 정책을 미리 검토할 수 있으므로 불확실성을 최대한 줄일 수 있습니다.



### Online Prompt and Solver Selection for Program Synthesis (https://arxiv.org/abs/2501.05247)
Comments:
          Accepted at the 39th AAAI Conference on Artificial Intelligence (AAAI-25) Main Track

- **What's New**: 이 논문에서는 프로그램 합성을 위한 온라인 학습 방법을 제시합니다. 연구자들은 다중 팔 매딧(multi-armed bandit) 알고리즘을 활용하여 각 합성 작업에 대해 가장 적합한 솔버(solver)를 선택하는 방식을 제안합니다. 특히, 이 방법은 사용자에게 적합한 LLM과 프롬프트 스타일을 자동으로 선택하여 성능을 최적화할 수 있도록 돕습니다.

- **Technical Details**: 연구에서는 프로그램 합성을 위해 사용되는 LLM(대형 언어 모델) 및 심볼릭 솔버에 대해 다중 팔 매딧 문제를 제시합니다. 솔버의 성능을 평가하고, 가장 적합한 선택을 하기 위해 시퀀셜 선택 과정을 통해 탐색(exploration)과 활용(exploitation)을 조화시킵니다. CYANEA라는 시스템을 구현하여 실제 합성 쿼리에 대해 평가를 진행하였으며, 기존의 최상의 솔버보다 37.2% 더 많은 쿼리를 해결했습니다.

- **Performance Highlights**: CYANEA는 다양한 쿼리에 대해 매우 높은 성능을 보이며, 경쟁 솔버와 비교했을 때 우수한 결과를 도출했습니다. LLM 및 심볼릭 솔버의 조합을 통해 항상 최적의 솔루션을 제공함으로써, 사용자의 성능 요구에 효과적으로 대응할 수 있는 가능성을 보여줍니다. 결과적으로, CYANEA는 가상 최상 솔버의 4% 이내의 성과를 달성하였습니다.



### Multimodal-to-Text Prompt Engineering in Large Language Models Using Feature Embeddings for GNSS Interference Characterization (https://arxiv.org/abs/2501.05079)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)을 GNSS(글로벌 내비게이션 위성 시스템) 간섭 모니터링에 적용하는 새로운 접근 방식을 제시합니다. 기존의 연구들은 주로 LLM을 특정 작업에 맞게 개발하는 데 집중했고, GNSS 신호 분석에 LLM을 활용한 연구는 미흡했습니다. 저자들은 LLaVA를 사용하여 GNSS 데이터셋에서 특징을 추출하고, 지식 기반에서 관련 정보를 검색하여 간섭을 식별하고 완화하는 방법을 모색합니다.

- **Technical Details**: 저자들은 방대한 GNSS 데이터셋에서 특징을 추출하고, t-SNE 기법을 활용하여 특징 임베딩을 분석합니다. 이 과정에서 프롬프트 엔지니어링을 통해 GNSS 간섭 및 환경 요인을 해석하며, 생성된 출력은 비전문가가 이해할 수 있도록 설명적입니다. 또한, LLM을 통해 복잡한 다변량 데이터를 실시간으로 처리하고 해석하는 능력을 강조합니다.

- **Performance Highlights**: 제시된 방법은 최근 기계 학습 모델보다 간섭 분류 작업에서 우수한 성능을 나타냅니다. 연구의 결과는 LLM이 GNSS 간섭 모니터링에 적용될 때 비주얼 및 논리적 추론이 가능하다는 것을 보여줍니다. 이는 시스템의 회복력 및 적응성을 강화하는 데 기여할 것으로 기대됩니다.



### A Text-Based Knowledge-Embedded Soft Sensing Modeling Approach for General Industrial Process Tasks Based on Large Language Mod (https://arxiv.org/abs/2501.05075)
- **What's New**: 본 논문에서는 LLM-TKESS(대규모 언어 모델 기반 텍스트 지식 내장 소프트 센서)라는 일반 프레임워크를 제안합니다. 이는 딥러닝에 기반한 데이터 주도 소프트 센서(DDSS)의 한계를 극복하고, LLM의 문제 해결 능력, 크로스 모달 지식 전이 능력, 그리고 적은 샘플 학습 능력을 활용하여 성능을 개선하려는 시도입니다. 새로운 자연어 입력 모달리티를 통합한 두 가지 텍스트 기반 소프트 센서를 개발하여 단순 구조화된 데이터 모델의 한계를 극복하고자 합니다.

- **Technical Details**: LLM-TKESS는 보조 변수 시퀀스 인코더(AVS Encoder)를 통해 LLM이 연속적인 시계열 데이터 내의 시간적 관계와 보조 변수 간의 공간적 의미 관계를 이해할 수 있도록 설계되었습니다. 또한, 두 단계의 미세 조정(alignment) 전략을 채택하여, 프로세스 변수 데이터에 적응한 소프트 센싱 기반 모델(SSFM)을 초속도로 구축할 수 있게 하고, 후속 작업에 대해 아키텍처를 수정하지 않고도 적응할 수 있는 어댑터를 도입합니다.

- **Performance Highlights**: 본 모델은 공기 예열기 로터의 열 변형을 사례 연구로 사용하여 광범위한 실험을 통해 탁월한 예측 성능을 보였습니다. 특히, 적은 샘플 상황에서도 뛰어난 예측 능력을 보여 DDSS의 기존 한계를 넘어서는 성과를 나타냈습니다. LLM-TKESS는 이전의 DDSS에 비해 데이터의 강력한 표현 학습과 효율적인 처리 능력을 통해 산업 프로세스의 실시간 모니터링과 최적화의 달성을 가능하게 합니다.



### A General Retrieval-Augmented Generation Framework for Multimodal Case-Based Reasoning Applications (https://arxiv.org/abs/2501.05030)
Comments:
          15 pages, 7 figures

- **What's New**: 이번 논문에서는 멀티모달 CBR(사례 기반 추론) 응용을 위한 MCBR-RAG 프레임워크를 소개합니다. 기존의 연구들은 주로 텍스트 기반 응용에 중점을 두었으나, 실제 문제들은 이미지, 오디오 및 비디오와 같은 다양한 형식의 요소를 포함할 수 있습니다. MCBR-RAG는 비텍스트 구성 요소를 텍스트 기반 표현으로 변환하여 멀티모달 CBR 문제를 해결할 수 있도록 합니다.

- **Technical Details**: MCBR-RAG 프레임워크는 사례의 비텍스트 구성 요소를 텍스트 기반 표현으로 변환하여 두 단계인 Retrieval(검색)과 Reuse(재사용)를 지원합니다. 이 과정에서 문제를 신경망 임베딩으로 변환하고, 검색된 사례들의 지식을 LLM(대형 언어 모델)에 쿼리로 통합하는 방법을 사용합니다. 이렇게 처리된 정보는 LLM의 쿼리의 맥락을 풍부하게 하여 새로운 문제를 해결하는 데 기여합니다.

- **Performance Highlights**: 실험 결과, MCBR-RAG는 Math-24와 같은 간단한 애플리케이션 및 Backgammon과 같은 복잡한 작업에서 LLM의 생성 품질을 기존 베이스라인보다 개선했습니다. 특히, MCBR-RAG를 활용하면 사례 구성이 멀티모달일 때에도 효과적으로 문제를 해결할 수 있는 가능성을 보여주었습니다.



### ActPC-Geom: Towards Scalable Online Neural-Symbolic Learning via Accelerating Active Predictive Coding with Information Geometry & Diverse Cognitive Mechanisms (https://arxiv.org/abs/2501.04832)
- **What's New**: 이번 논문에서는 신경망에서 Active Predictive Coding (ActPC)의 속도를 높이기 위한 새로운 접근 방법인 ActPC-Geom을 소개합니다. 이 방법은 정보 기하학(information geometry)을 통합하며, Wasserstein 메트릭에 기반한 방법을 사용하여 측정 의존적( measure-dependent) 그래디언트 흐름을 최적화합니다. 논문에서는 KL-divergence를 ActPC의 예측 오류 평가에 있어서 Wasserstein 메트릭으로 대체할 것을 제안하며, 이는 네트워크의 강건성을 향상시킬 수 있다고 주장합니다.

- **Technical Details**: ActPC-Geom은 ActPC의 여러 기술적 측면을 포함하고 있습니다. 주요 요소로는 희소 보상(sparse rewards), 그래디언트 없는 최적화(gradient-free optimization), 그리고 국소 Hebbian 업데이트가 있습니다. 논문의 대부분을 차지하는 Ororbia의 NGC 프레임워크는 각 레이어가 다음 레이어의 활동을 예측하여 시냅스 업데이트를 유도하는 구조로, 이를 통해 효과적인 강화 학습 문제 해결을 지원합니다.

- **Performance Highlights**: 실험 결과, ActPC-Geom은 동적 환경에서도 우수한 적응력을 보이며 기존의 후방 전파(backpropagation) 기반 방법보다 신경망의 생물학적 타당성과 강화 학습 성능을 동시에 개선할 수 있는 가능성을 제시합니다. 특히, 희소한 보상이 주어지는 상황에서도 탐색 driven epistemic signals과 목표 지향적 instrumental signals를 효과적으로 결합하여 안정적이고 효율적인 학습을 달성할 수 있습니다. 이러한 점은 ActPC의 실제 적용 가능성을 한층 높입니다.



### AI-Driven Reinvention of Hydrological Modeling for Accurate Predictions and Interpretation to Transform Earth System Modeling (https://arxiv.org/abs/2501.04733)
- **What's New**: 이번 논문에서는 전통적인 수식 기반 수문 모델과 기존 알고리즘 기반 모델의 한계를 극복한 새로운 모델인 HydroTrace를 소개합니다. 이 모델은 강력한 예측 성능을 바탕으로 98%의 Nash-Sutcliffe Efficiency를 기록하며, 새로운 데이터에 대한 일반화 능력이 뛰어납니다. 또한, HydroTrace는 고급 attention mechanism을 활용하여 시공간적 변동성과 특징별 영향을 효과적으로 포착합니다.

- **Technical Details**: HydroTrace는 데이터-불가지론적(data-agnostic) 모델로, 수문 행동의 해석이 어려운 기존 모델들과 비교해 상당한 성능 향상을 가져왔습니다. 이 모델은 빙하-눈-유량(glacier-snow-streamflow) 상호작용이나 몬순(monsoon) 역학에 대한 해석을 가능하게 합니다. 뿐만 아니라, HydroTrace의 대형 언어 모델(LLM) 기반 애플리케이션은 사용자들이 이 모델의 통찰력을 쉽게 이해하고 활용할 수 있도록 지원합니다.

- **Performance Highlights**: HydroTrace는 강력한 예측 정밀도와 해석 가능성을 제공하여 수문학(hydrology) 및 광범위한 지구 시스템 모델링에서 혁신적인 도구로 자리 잡고 있습니다. 특히, 새로운 데이터에서도 뛰어난 일반화 능력을 보여주며 실용성을 더욱 높였습니다. 이러한 특성은 다양한 수문 현상에 대한 예측 모델링을 통한 발전을 이끌어낼 것으로 기대됩니다.



### An Empirical Study of Autoregressive Pre-training from Videos (https://arxiv.org/abs/2501.05453)
- **What's New**: 이 논문은 비디오에서의 자기회귀(autoregressive) 사전 학습(pre-training)을 실증적으로 연구합니다. 저자들은 Toto라는 일련의 비디오 모델을 구축하고, 비디오를 시각적 토큰의 연속으로 처리하여 Transformer 모델이 미래 토큰을 예측하도록 훈련합니다. 1조 개가 넘는 시각적 토큰을 포함하는 다양한 데이터셋에서 이러한 모델을 사전 훈련하고 다양한 다운스트림 작업에서 평가한 결과, 최소한의 유도 편향에도 불구하고 경쟁력 있는 성능을 보였습니다.

- **Technical Details**: 비디오를 시각적 토큰의 시퀀스로 간주하고, 각 프레임을 dVAE(Ramesh et al., 2021)로 불연속 토큰화(tokenization)합니다. 그런 다음 LLaMa(Touvron et al., 2023) 아키텍처의 인과적 Transformer 모델을 사용하여 다음 토큰 예측 작업에 대해 훈련합니다. 다양한 설계 선택을 활용하여 모델을 평가하고, 주목 풀링(attention pooling)을 사용하여 시각적 표현을 추출합니다.

- **Performance Highlights**: 모델들은 이미지 인식, 비디오 분류, 물체 추적, 로봇 작업과 같은 효과적인 다운스트림 작업에서 성능을 평가받았고, 결과적으로 안정적인 고성능을 나타냈습니다. 이미지넷 분류 작업에서는 불연속 및 연속 패치정규화(continuous patch-normalized) 토큰을 기반으로 한 자가회귀 모델들이 유사한 성능을 보여주었습니다. 마지막으로, 자기회귀 비전 모델들은 언어 모델에 비해 더 느린 속도로 확장되는 특성을 보였습니다.



### Consistent Flow Distillation for Text-to-3D Generation (https://arxiv.org/abs/2501.05445)
Comments:
          Project page: this https URL

- **What's New**: 본 논문에서는 3D 생성 모델의 효율성을 개선하기 위한 새로운 방법인 Consistent Flow Distillation (CFD)을 제안합니다. 이를 통해 기존의 Score Distillation Sampling (SDS)의 한계인 시각적 품질과 다양성 저하 문제를 해결하고자 합니다. CFD는 2D 이미지 플로우의 일관성을 강조하며, 다양한 시점에서의 3D 생성 품질을 개선합니다.

- **Technical Details**: CFD는 다중 시점에서의 일관된 Gaussian noise를 활용하여 3D 객체의 생성을 돕습니다. 이 과정에서 noise transport equation을 기반으로 물체 표면에서의 일관된 noise 패턴을 유지하여 그라디언트를 계산합니다. 이는 diffusion ODE(Ordinary Differential Equation) 또는 SDE(Stochastic Differential Equation) 샘플링 과정을 통해 3D 생성을 직접 유도합니다.

- **Performance Highlights**: 실험 결과, CFD는 기존의 텍스트-투-3D(text-to-3D) 생성 방식에 비해 품질과 다양성에서 현저히 개선된 성능을 보였습니다. 본 방법을 통해 생성된 3D 자산은 현실적인 외관과 형태를 가지며, 동일한 텍스트 프롬프트에 대해 다양한 3D 객체를 샘플링할 수 있습니다. CFD는 기존의 SDS와 비교하여 계산 비용이 거의 추가되지 않았음에도 높은 성능을 달성했습니다.



### A survey of textual cyber abuse detection using cutting-edge language models and large language models (https://arxiv.org/abs/2501.05443)
Comments:
          37 pages, under review in WIREs Data Mining and Knowledge Discovery

- **What's New**: 이번 논문에서는 소셜 미디어의 온라인 남용 형태에 대한 포괄적인 분석을 제공하며, 특히 최신 기술이 이러한 남용 내용을 탐지하고 생성하는 방식을 어떻게 재편성하는지에 초점을 맞추고 있습니다. 기존에 잘 연구된 학대 형태 외에도, 최근의 지능형 언어 모델(LLMs)을 통한 분석을 진행하여 보다 넓은 스펙트럼의 사이버 남용을 조명합니다.

- **Technical Details**: 이 논문은 언어 모델과 대형 언어 모델의 역할을 탐구하며, 특히 이들이 사이버 남용과 증오 발언 탐지에서 어떻게 활용될 수 있는지를 분석합니다. LLMs(GPT, BERT 등)는 자연어 처리(NLP)의 사고를 이용해 특정한 사이버 남용 형태를 탐지하고 분류하는 데 새로운 가능성을 제시합니다. 뿐만 아니라, 이 논문은 평가 지표의 불균형 문제를 다루며 사이버 남용 탐지에서의 도전 과제를 심도 있게 분석합니다.

- **Performance Highlights**: LLMs는 사이버 남용 검출 시스템을 자동화하여 기존 방식보다 감소된 수의 잘못된 탐지 및 높은 정확도로 개선된 성과를 보여줍니다. 이 연구는 사이버 괴롭힘 및 증오 발언 외 특정 형태의 사이버 남용에 대한 이해를 깊이 있게 하고, AI 기술이 안전하고 지지적인 온라인 환경을 구축하는 데 어떻게 기여할 수 있는지를 탐구합니다. 이에 따라, 최신 언어 기술이 클라우드 데이터 속에서 악의적인 콘텐츠를 생성할 수 있다는 점도 논의되고 있습니다.



### Progressive Growing of Video Tokenizers for Highly Compressed Latent Spaces (https://arxiv.org/abs/2501.05442)
Comments:
          Project website: this https URL

- **What's New**: 이 논문은 비디오 토크나이저(video tokenizer)의 성능을 혁신적으로 향상시키기 위한 새로운 방법을 제안합니다. 기존의 토크나이저들이 시간적 압축 비율을 4배 이상으로 확장하는 데 어려움을 겪는 한편, 저자는 저압축 인코더를 통해 비디오의 재구성 품질을 개선할 수 있음을 발견했습니다. 이 연구는 ProMAG라는 새로운 모델을 개발하여, 낮은 압축 모델에서 학습된 정보를 활용하여 높은 압축 블록을 점진적으로 훈련시키는 접근 방식을 취했습니다.

- **Technical Details**: ProMAG 모델은 MagViT-v2 아키텍처를 기반으로 하며, 지속적인 토큰을 활용하여 고해상도 비디오 인코딩을 지원합니다. 저자는 우선적인 4배 시간 압축 모델을 바탕으로 점진적인 방식으로 압축 비율을 8배 및 16배까지 늘리는 기술을 도입했습니다. 이를 통해 다양한 압축 레벨을 다루는 별도의 모델 블록을 학습하여 최적의 재구성 품질을 달성하도록 설계되었습니다.

- **Performance Highlights**: 이 연구에서는 ProMAG 모델이 기존의 고압축 비디오 토크나이저와 비교할 때 비약적으로 향상된 재구성 품질을 기록했다고 보고합니다. 제안된 모델은 16배 압축 비율로도 고품질 재구성을 가능하게 하였으며, 텍스트-비디오 생성에서도 효율성을 극대화하여 4배 압축 모델과 비교할 때 동등하거나 더 높은 품질을 달성했습니다.



### From Simple to Complex Skills: The Case of In-Hand Object Reorientation (https://arxiv.org/abs/2501.05439)
Comments:
          website: this https URL

- **What's New**: 이번 연구에서는 면밀한 보상이 요구되는 sim-to-real(시뮬레이션에서 실제로) 간의 격차를 줄이기 위해 사전 학습된 저수준 스킬을 활용하는 새로운 시스템을 소개합니다. 우리는 고수준의 정책을 통해 이러한 저수준 스킬을 조합하여 인손에서의 객체 재배치 작업을 수행하는 계층적 정책을 제안합니다. 이로 인해 이전 스킬을 재사용할 수 있어, 매번 처음부터 학습하는 데 필요한 인간의 노력을 크게 줄일 수 있습니다.

- **Technical Details**: 여기서 제안하는 시스템은 사전 학습된 객체 회전 스킬을 기반으로 한 계층적 정책을 통해 작동합니다. 이 정책은 저수준 스킬이 수행해야 할 작업을 결정하고, 환경의 피드백을 기반으로 옵니다. 또한, 이 시스템은 전통적인 포즈 추정 기법의 한계를 극복하기 위해 프로프리오셉션(proprioceptive) 정보를 사용하여 시간을 통해 객체의 포즈를 예측합니다.

- **Performance Highlights**: 우리는 제안하는 시스템이 기존 방법에 비해 빠르게 수렴하고 높은 성능을 이끌어낼 수 있음을 실험을 통해 입증하였습니다. 특히, 이 정책은 다양한 객체에 대해 일반화된 상태 추정기를 학습하여 실제 환경으로 쉽게 전이할 수 있습니다. 최종적으로, 학습된 정책은 전이 가능성이 높은 저수준 정책을 자연스럽게 활용하면서 높은 효율성을 보장합니다.



### A Novel Pathology Foundation Model by Mayo Clinic, Charit\'e, and Aignostics (https://arxiv.org/abs/2501.05409)
- **What's New**: 이 보고서에서는 RudolfV 접근 방식을 기반으로 하는 새로운 비전 파운데이션 모델을 소개합니다. 이 모델은 Mayo Clinic과 Charité - Universitätsmedizin Berlin에서 수집된 120만 개의 조직병리 전체 슬라이드 이미지로 훈련되었습니다. 결과적으로, 이 모델은 21개의 공공 벤치마크 데이터셋에서 최첨단 성능을 보여주며, 매개변수 수나 데이터셋 크기가 가장 크지 않음에도 불구하고 뛰어난 성과를 거두었습니다.

- **Technical Details**: 모델 훈련에는 490,000개의 사례에서 추출된 120만 개의 디지털 이미지가 사용되었습니다. 특히, H&E, IHC 및 특수 염색 데이터가 포함되어 있으며, 다양한 확대 비율에서 훈련이 진행되었습니다. 이 보고서는 ViT-H/14 구조를 활용하여 632백만 개의 매개변수를 가진 모델을 훈련하기 위해 RudolfV의 알고리즘을 적응시켰습니다.

- **Performance Highlights**: 모델 성능 평가는 선형 프로빙 프로토콜을 통해 수행되었으며, 21개의 벤치마크 데이터셋을 사용하여 다양한 작업에서 비교되었습니다. 평가 결과는 첫 번째 세대 모델과 비교하여 뛰어난 성과를 나타내며, 특히 클래스 토큰 및 평균 토큰을 고려한 평균 정확도에서 높은 점수를 기록했습니다. 이러한 검증 작업은 또한 데이터 재현성과 비교 가능성을 높이는 데 기여하고 있습니다.



### TimeRL: Efficient Deep Reinforcement Learning with Polyhedral Dependence Graphs (https://arxiv.org/abs/2501.05408)
Comments:
          17 pages, 11 figures, 5 bibliography pages

- **What's New**: 본 논문은 Deep Reinforcement Learning (DRL) 프로그램을 실행하기 위한 새로운 시스템인 TimeRL을 소개합니다. TimeRL은 eager execution의 동적 특성과 graph-based execution의 전체 프로그램 최적화를 결합하여 DRL 알고리즘의 동적 데이터 종속성을 효과적으로 처리합니다. 이를 통해 복잡한 DRL 알고리즘을 최대 47배 빠르게 실행하며 GPU 메모리 사용량은 16배 적게 줄였습니다.

- **Technical Details**: TimeRL은 재귀 텐서(Recurrent Tensors)라는 선언적 프로그래밍 모델을 도입하여 동적 종속성을 수식적으로 표현할 수 있도록 합니다. 이 시스템은 Polyhedral Dependence Graphs (PDGs)를 사용하여 전체 프로그램을 하나의 그래프로 표현하여, 다양한 실행 지점 간의 관계를 상징적 표현으로 나타냅니다. 이를 통해 TimeRL은 자동 메모리 관리 최적화 및 실행 스케줄을 자동으로 결정하는 기능을 제공합니다.

- **Performance Highlights**: TimeRL은 현재의 DRL 알고리즘에 대해 최대 47배 더 빠른 훈련 속도를 기록하며, 기존 DRL 시스템에 비해 메모리 효율성을 16배 개선했습니다. 이러한 성능 향상은 TimeRL이 동적 데이터 종속성을 효과적으로 처리하고, 연산을 병렬화 및 증분화하며, 알고리즘 특화된 스케줄링을 수행함으로써 이루어졌습니다.



### On-line Policy Improvement using Monte-Carlo Search (https://arxiv.org/abs/2501.05407)
Comments:
          Accompanied by oral presentation by Gregory Galperin at NeurIPS 1996 (then known as NIPS*96)

- **What's New**: 이번 논문에서는 적응 제어기의 실시간 정책 개선을 위한 몬테카를로 시뮬레이션 알고리즘을 제안합니다. 이 알고리즘은 초기 정책을 바탕으로 각 가능한 행동의 장기적인 기대 보상을 통계적으로 측정하며, 기대 보상이 최대화되는 행동을 선택해 개선된 정책을 생성합니다. 특히, 이 알고리즘은 IBM의 SP1 및 SP2 병렬 RISC 슈퍼컴퓨터에 구현되었으며, 백개먼(backgammon) 분야에서 초기 결과를 통해 우수한 성능을 보였습니다.

- **Technical Details**: 정책 반복(policy iteration)이란 전통적으로 채택되는 알고리즘으로, 초기 정책을 기반으로 장기적인 기대 보상을 계산 후 이를 최적화하는 프로세스입니다. 전통적으로 오프라인에서 실행되며, 강화 학습(reinforcement learning) 방식을 활용하여 정책을 개선하는 다양한 접근법이 있지만, 실시간으로 개선된 정책을 계산하는 것은 여전히 느린 경향이 있습니다. 이에 반해 본 논문에서는 실시간으로 개선된 정책을 산출하기 위해 몬테카를로 검색(Monte-Carlo search)을 활용하여 각 행동의 기대 보상을 추정하는 온라인 알고리즘을 제안합니다.

- **Performance Highlights**: 몬테카를로 알고리즘은 다양한 초기 정책에 대해 근본 선수들의 오류율을 5배 이상 줄이는 결과를 제공합니다. 특히, 백개먼 게임에서 이 방법을 적용할 경우 평균적으로 약 200,000회의 몬테카를로 시행으로도 결정적인 결정을 도출할 수 있었습니다. IBM SP1 및 SP2 슈퍼컴퓨터에서 약 100K 기반 플레이어 결정을 초당 달성함으로써 몬테카를로 시뮬레이션의 효과성을 입증했습니다.



### TimeDP: Learning to Generate Multi-Domain Time Series with Domain Prompts (https://arxiv.org/abs/2501.05403)
Comments:
          AAAI 2025

- **What's New**: 본 논문에서는 TimeDP라는 다중 도메인 시계열 생성 모델을 제안합니다. 이 모델은 시계열 프로토타입 모듈을 활용하여 생성 조건으로 사용할 도메인 프롬프트를 학습합니다. 이 방식은 기존의 단일 도메인 데이터에 국한된 생성 모델의 한계를 넘어, 여러 도메인에서 새 데이터를 생성할 수 있는 가능성을 열어줍니다.

- **Technical Details**: TimeDP는 기본 요소로 시계열 프로토타입을 학습하고 이를 활용해 도메인 프롬프트를 구축하여 시계열 데이터 생성을 수행합니다. 훈련 과정에서 프로토타입은 시계열의 기초를 나타내는 역할을 하며, 각 샘플에 대해 프로토타입 할당 모듈을 적용해 특정 조건 프롬프트를 생성합니다. 샘플링 과정에서는 목표 도메인에서 몇 개의 샘플을 추출하여 도메인 프롬프트를 구성하고 이를 기반으로 시계열 데이터를 생성합니다.

- **Performance Highlights**: 실험을 통해 TimeDP는 기존의 기준 모델들을 능가하며 인도메인 생성 품질에서 최첨단 성능을 가지고 있음을 입증하였습니다. 또한, 이전에 보지 못한 도메인에서도 강력한 생성 능력을 보여주어 다중 도메인 시계열 생성의 가능성을 크게 확장했습니다.



### BRATI: Bidirectional Recurrent Attention for Time-Series Imputation (https://arxiv.org/abs/2501.05401)
- **What's New**: 이번 연구에서는 BRATI라는 새로운 딥러닝 모델을 소개합니다. BRATI는 Bidirectional Recurrent Networks와 Attention Mechanisms을 결합하여 다변량 시계열 데이터의 결측치를 보완하는 데 초점을 맞추고 있습니다. 이 모델은 시간적 의존성과 특성 간의 상관관계를 처리하며, 서로 반대 방향으로 작동하는 두 개의 보간 블록을 이용합니다. BRATI는 다중 시나리오의 결측 데이터 상황에서도 뛰어난 성능을 보여줍니다.

- **Technical Details**: BRATI 모델은 두 가지 유형의 RNN 계층과 Attention 메커니즘을 통합하여 장기 및 단기 의존성을 효과적으로 모델링합니다. 모델은 세 가지 실제 데이터 세트에서 평가되었으며, 무작위 결측치, 고정 길이 시퀀스 및 가변 길이 시퀀스와 같은 다양한 결측 데이터 시나리오를 다룹니다. 각 보간 블록은 시간의 방향에 따라 상이한 기능을 수행하며, 이를 통해 복잡한 패턴과 변수 간의 상관관계를 효과적으로 캡처합니다. 이러한 접근 방식은 일반적인 통계적 방법의 한계를 극복할 수 있는 잠재력을 가지고 있습니다.

- **Performance Highlights**: BRATI는 기존의 최첨단 모델들과 비교하여 일관되게 높은 정확성과 견고성을 보여줍니다. 연구 결과, BRATI는 다양한 결측 데이터 시나리오에서 다른 모델에 비해 뛰어난 성능을 발휘하여, 다변량 시계열 데이터의 보완 과제에 대한 새로운 해결책을 제시합니다. 또한, BRATI는 기존 연구에서 종종 간과되는 MNAR 시나리오에 대해서도 우수한 성능을 보여주었습니다.



### Mechanistic understanding and validation of large AI models with SemanticLens (https://arxiv.org/abs/2501.05398)
Comments:
          74 pages (18 pages manuscript, 7 pages references, 49 pages appendix)

- **What's New**: 이 논문에서는 기계 학습 모델의 불투명성을 극복하기 위해 SemanticLens라는 범용 설명 방법을 제안합니다. 이는 AI 모델 구성 요소가 인코딩한 숨겨진 지식을 시맨틱하게 구성된 멀티모달 공간으로 매핑하여, 각각의 뉴런이 특정 개념을 인코딩하는 것을 식별하는 텍스트 검색과 같은 독특한 작업을 가능하게 합니다.

- **Technical Details**: SemanticLens는 Neural Networks의 특정 구성 요소를 세밀하게 분석하고 비교할 수 있는 체계적인 분석을 제공합니다. 이 방법은 뉴런 자동 레이블링, 기능 역할 설명, 결정 사항 위반 감사 등을 수행하여 AI 모델의 결정 과정을 검증합니다. 또한, SemanticLens는 CLIP과 같은 기초 모델에 대한 설명을 통해 불확실한 결정 사항들을 보다 명확하게 설명합니다.

- **Performance Highlights**: 이 모델은 디버깅 및 검증을 효과적으로 수행하며, 모델 지식을 요약하고 기대와의 일치도를 높이는 데 기여합니다. 예를 들어, 흑색종 분류에서 ABCDE 규칙 준수를 확인하는 데 도움을 줍니다. SemanticLens는 AI 모델과 전통적인 엔지니어링 시스템 간의 '신뢰 격차'를 줄이는 데 기여하는 것을 목표로 하고 있습니다.



### The global consensus on the risk management of autonomous driving (https://arxiv.org/abs/2501.05391)
- **What's New**: 이 연구는 자율주행차의 위험 관리가 단순히 사고 가능성을 최소화하는 것이 아니라 사고의 가능성과 심각성을 평가하는 데 중점을 두어야 한다는 것을 밝힙니다. 또한, 이러한 위험 분배의 선호는 문화적 배경에 관계없이 유사하다는 점을 강조하고 있습니다. 오히려 모든 나라에서 자전거 이용자들이 추가적인 위험 보너스를 받지 않는다는 사실은 주목할 만합니다.

- **Technical Details**: 이 연구는 11,000명의 참가자를 포함한 8개국에서 실시된 글로벌 실험 연구로, 여러 문화적 배경을 가진 도로 사용자 간의 위험 분배 선호를 비교하였습니다. 결과는 각 국에서 사고 가능성과 심각성의 고려가 거의 다르지 않다는 것을 보여줍니다. 연구는 자율주행차의 윤리에 대한 글로벌 합의가 사고를 일으킬 때의 윤리보다 더 쉽게 확립될 수 있음을 제안합니다.

- **Performance Highlights**: 전국적으로 대부분의 참여자들이 사고 가능성을 최소화하는 원칙에서 벗어나 사고 가능성과 심각성을 균형 있게 고려한다고 결론지었습니다. 자율주행차와 관련한 사회적 딜레마는 일상 교통 상황에서의 위험 평가에서는 사라지는 것으로 나타났습니다. 따라서, 자율주행차의 위험 윤리에 관한 글로벌 합의를 찾는 일이 더 수월할 수 있다는 점이 강조되었습니다.



### Large Physics Models: Towards a collaborative approach with Large Language Models and Foundation Models (https://arxiv.org/abs/2501.05382)
- **What's New**: 이번 논문은 물리학 특정 대규모 AI 모델, 즉 Large Physics Models (LPMs)의 개발 및 평가를 위한 아이디어와 잠재적인 로드맵을 탐구합니다. LPM들은 기존의 대규모 언어 모델(LLMs)을 기반으로 하며, 물리학 연구의 필요를 충족하기 위해 맞춤화되어 있습니다. 이는 수학적 조작을 위한 기호적 추론 모듈, 특정 실험 및 시뮬레이션 데이터 분석 프레임워크, 이론 및 과학 문헌 합성을 위한 메커니즘을 포함합니다. 논문은 물리학 커뮤니티가 이러한 전문화된 모델을 개발해야 하는 이유와 과정에 대해 논의합니다.

- **Technical Details**: 저자들은 LPM의 발전을 위해 필요한 세 가지 핵심 기둥인 개발, 평가, 그리고 철학적 반영을 제시합니다. 개발 단계에서는 물리학 텍스트, 수학적 공식, 다양한 물리 데이터를 처리할 수 있는 모델을 구축하는 데 초점을 맞추고 있습니다. 평가 단계에서는 정확성과 신뢰성을 테스트하고 벤치마킹하여 확인합니다. 마지막으로, 철학적 반영은 LPM의 의미와 물리학 내에서 AI의 잠재적 영향을 분석하는 데 중점을 둡니다.

- **Performance Highlights**: 논문에서는 LPMs가 물리학 분야의 다양한 연구 활동을 강화할 수 있는 잠재력을 강조합니다. LPMs는 가설 생성, 실험 설계, 데이터 분석 등 다양한 연구 과정에서 도움을 줄 수 있으며, 창의성을 불러일으킬 수 있는 '인공지능 뮤즈' 역할을 할 수 있습니다. 여러 예제가 제시되며, LLMs가 연구의 초기 단계에서 통찰력 있는 질문을 제기하고 가치 있는 관찰을 제공하고 있다는 것을 보여줍니다.



### On Corrigibility and Alignment in Multi Agent Games (https://arxiv.org/abs/2501.05360)
- **What's New**: 이번 논문에서는 자동화된 에이전트의 교정 가능성(corrigibility)을 다루고 있으며, 다중 에이전트 환경에서의 교정 가능성 모델링을 위한 새로운 프레임워크를 제시합니다. 기존의 연구들이 단일 에이전트 시스템에 초점을 맞춘 것과 달리, 이 연구는 서로의 전략이 상호 작용하는 2인 게임으로 교정 가능성을 분석합니다. 특히, 인간의 감독을 요청할 수 있는 움직임을 항상 가진 에이전트의 특성을 중점적으로 다룹니다.

- **Technical Details**: 전략 게임의 개념을 바탕으로 하여, 에이전트의 행동과 보상 함수(u_i)는 다른 모든 플레이어의 전략에 따라 달라지는 비협력적(non-cooperative) 게임으로 정의됩니다. Nash 균형(Nash equilibrium) 개념을 통해 각 플레이어의 전략이 서로 최적의 반응을 보이는 형태를 분석하며, 이는 "자기 실현적 합의(self-fulfilling agreement)"로 해석될 수 있습니다. 이 논문은 특히 두 개의 특정 케이스를 분석하며, cyber security의 defender/adversary 모델을 통해 교정 가능성과 관련된 분석 결과를 도출합니다.

- **Performance Highlights**: 교정 게임의 특정 경우에서는, 에이전트들이 monotone 게임과 harmonic 게임 사이에서 불확실성을 가지도록 설정하여 교정 가능성을 보여줍니다. 결과적으로 defending agent가 교정 가능성을 유도하기 위해 고려해야 하는 게임에 대한 믿음과 인간의 합리성에 대한 일반적인 결과를 제공합니다. 마지막으로, 이 연구가 이론적/실용적 맥락에서 가지는 의미와 향후 작업의 방향을 제시합니다.



### Stream Aligner: Efficient Sentence-Level Alignment via Distribution Induction (https://arxiv.org/abs/2501.05336)
Comments:
          AAAI Alignment Track 2025 Poster

- **What's New**: 이 논문에서는 Streaming Distribution Induce Aligner (Stream Aligner)라는 새로운 정렬(paradigm) 접근법을 제시합니다. 이 방법은 LLM의 성능을 향상시키면서도 효율성을 유지하는 데 중점을 두고 있습니다. 특히, 문장 수준의 반복적인 수정 과정을 통해 이전 모델의 출력을 개선하며, 이로 인해 적은 자원으로도 더 나은 결과를 도출할 수 있습니다.

- **Technical Details**: Stream Aligner는 정렬 과정을 문장 수준으로 세분화하여, 모든 세션에서 사용자의 선호도를 학습하고 이를 바탕으로 수정된 출력을 기본 모델로 피드백합니다. 이 과정은 기계적 학습(machinery learning) 원리와 선호 데이터셋(preference dataset)을 기반으로 하며, 반복적인 피드백이 발생하는 구조를 가집니다. 이는 인퍼런스 과정 중에 실제로 인간의 의도와 가치를 LLM 출력에 효과적으로 주입할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 2B 모델의 Stream Aligner는 Llama2-70B-chat 모델에서 응답의 유용성을 41.2% 증가시키고, 무해성은 36.0% 향상시켰습니다. 또한, Stream Aligner-8B 모델은 Llama3-70B-Instruct 모델에서 수학적 능력을 3.5% 개선하는 등 성능이 입증되었습니다. 이러한 결과는 Stream Aligner가 추가 모델의 의존도를 줄이며, LLM의 추론 능력을 향상시키는 데 기여함을 나타냅니다.



### The Bakers and Millers Game with Restricted Locations (https://arxiv.org/abs/2501.05334)
Comments:
          To appear at the 24th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2025)

- **What's New**: 이 논문에서는 고객과 판매자의 전략적 위치 선택을 다룬 'Bakers and Millers Game'을 연구합니다. 새로운 요소는 밀러(판매자)는 자유롭게 위치를 선택할 수 있지만, 베이커(고객)는 제한된 위치에서 선택해야 한다는 점입니다. 이러한 비대칭적 위치 선택의 영향을 분석하며, 이는 상업, 제품 디자인 등 다양한 분야에 적용됩니다.

- **Technical Details**: 연구에서는 위치 선택이 게임의 결과에 미치는 영향을 살펴보고, 위치 제한이 있는 경우에도 순수 내쉬 균형(Nash equilibrium)이 존재함을 보여줍니다. 또한, 효율적인 알고리즘을 통해 순수 내쉬 균형을 계산할 수 있으며, 이는 최적의 사회적 후생(social welfare)에 근접하는 결과를 도출합니다. 이 모델은 기존의 대칭적인 Fractional Hedonic Games을 일반화하여 특정 위치에서 형성 가능한 연합(coalition)의 수를 제한합니다.

- **Performance Highlights**: 해석적인 측면에서, 위치 선택 기능은 게임 설정에 새로운 층을 추가합니다. 연구 결과, 순수 내쉬 균형의 존재를 증명하고, 비대칭적 조건 하에서도 효과적인 배치에 관한 강력한 경계를 제시합니다. 결과적으로, 본 연구는 경제적 의미를 갖는 다양한 상황을 모델링하면서, 전략적 위치 선택의 복잡성을 강조합니다.



### AnCoGen: Analysis, Control and Generation of Speech with a Masked Autoencoder (https://arxiv.org/abs/2501.05332)
Comments:
          5 pages, this https URL

- **What's New**: 이번 논문에서는 AnCoGen이라는 혁신적인 방법을 소개합니다. AnCoGen은 마스크 오토인코더(Masked Autoencoder)를 활용하여 음성 신호의 분석, 제어 및 생성을 단일 모델로 통합합니다. 이를 통해 화자 정체성, 피치, 내용, 음량, 신호 대 잡음비(signal-to-noise ratio), 명확도 지수와 같은 주요 속성을 추정하여 음성을 분석하고, 이러한 속성으로부터 음성을 생성하며, 이를 조정하여 합성된 음성을 정확히 제어할 수 있습니다.

- **Technical Details**: AnCoGen 모델은 인코더-디코더 Transformer 구조를 기반으로 하며, 음성 Mel-스펙트로그램(Mel-spectrogram)과 관련된 음성 속성(Speech Attributes)의 결합을 자동으로 인코딩합니다. 훈련 중에는 이러한 두 음성 표현이 부분적으로 마스킹되어 상호 및 내부 종속성을 학습하도록 하고, 추론 시에는 분석 또는 생성 작업에 따라 마스킹이 달라집니다. 최종적으로, 학습된 HiFi-GAN 신경 보코더를 결합하여 효율적으로 음성 신호를 분석하고 변환하며 합성합니다.

- **Performance Highlights**: AnCoGen의 실험 결과는 기존의 음성 분석-재합성, 피치 추정, 피치 수정 및 음성 강화 작업에서 효율성을 입증하고 있습니다. 이 모델은 음성 신호 조작과 관련된 여러 응용 분야에서 높은 성능을 보여 주며, 이전의 신경 인코더-디코더 모델들이 가지던 제어 부족 현상과 잡음 및 잔향에 대한 강인성 문제를 극복하는 데 기여합니다.



### Towards Balanced Continual Multi-Modal Learning in Human Pose Estimation (https://arxiv.org/abs/2501.05264)
- **What's New**: 이번 논문에서는 3D 인간 자세 추정(3D HPE)을 위해 RGB, LiDAR, mmWave, WiFi를 활용한 균형 잡힌 지속적인 다중 모달 학습 방법을 제안합니다. 우리는 Shapley value 기반 기여 알고리즘을 통해 각 모달리티의 기여도를 정량화하고, 이를 통해 발생하는 모달리티 불균형 문제를 해결하고자 합니다. 또한, 기존 데이터의 잡음 영향을 줄이기 위한 새로운 디노이징 지속적 학습 접근 방식을 개발했습니다.

- **Technical Details**: 제안하는 방법은 새로운 잡음 식별 및 분리 모듈(NIS)을 통해 모달리티 기여 점수를 모니터링하고 잡음이 감지된 경우 가장 잡음이 많은 데이터를 데이터셋에서 분리합니다. 이를 통해 잡음이 훈련 및 카타스트로픽 포겟팅에 미치는 영향을 완화할 수 있습니다. 또한, 적응형 EWC 메커니즘을 통해 중요한 정보가 손실되지 않도록 합니다.

- **Performance Highlights**: 우리는 MM-Fi라는 다중 모달 데이터셋에서 광범위한 실험을 수행하였으며, 제안한 접근 방식이 복잡한 시나리오에서 3D 자세 추정 성능을 향상시키고 카타스트로픽 포겟팅을 완화한다는 것을 입증했습니다. 이러한 결과는 다중 모달 학습에서의 균형 맞추기를 위한 첫 번째 시도라는 점에서 중요한 의미를 갖습니다.



### Enhancing Plagiarism Detection in Marathi with a Weighted Ensemble of TF-IDF and BERT Embeddings for Low-Resource Language Processing (https://arxiv.org/abs/2501.05260)
Comments:
          Accepted into LoResLM: The First Workshop on Language Models for Low-Resource Languages, colocated with COLING 2025 and set to be published into ACL Anthology

- **What's New**: 본 연구는 Marathi와 같은 자원이 부족한 언어에 대한 표절(plagiarism) 탐지 시스템을 설계하는 데 중점을 두었습니다. 기존의 모델들이 주로 나쁜 성능을 보여온 저자원 언어에서 BERT(Bidirectional Encoder Representations from Transformers)를 활용한 새로운 방법론을 제시합니다.

- **Technical Details**: 이 연구에서는 BERT의 문장 임베딩(sentence embeddings)과 Term Frequency-Inverse Document Frequency (TF-IDF) 기능 표현을 결합하여 Marathi 텍스트의 표절 탐지 정확성을 향상시킵니다. 이러한 접근은 기계 학습 모델의 가중치 투표 앙상블을 통해 텍스트의 통계, 의미 및 구문적 요소를 효과적으로 캡처합니다.

- **Performance Highlights**: 제안된 방법론은 기존 표절 탐지 시스템보다 더 높은 정확성을 보여주며, Marathi와 같은 저자원 언어의 표절 탐지 기술 발전에 기여할 것으로 기대됩니다. 이는 다양한 언어 처리(application) 분야에서 활용될 수 있는 중요한 연구 결과입니다.



### Automating the Detection of Code Vulnerabilities by Analyzing GitHub Issues (https://arxiv.org/abs/2501.05258)
- **What's New**: 본 논문은 GitHub 이슈 분석을 통해 소프트웨어 취약점을 자동으로 식별하는 변환기 기반 모델과 머신러닝 기법을 활용하는 새로운 접근법을 제시합니다. 이 연구는 취약점 탐지와 관련된 GitHub 이슈를 분류하기 위해 특별히 설계된 새로운 데이터셋을 도입합니다. 이 방법론은 현업에서 취약점 조기 발견의 가능성을 입증하며, 소프트웨어의 취약점 활용 기간을 대폭 줄일 수 있는 잠재력을 보여줍니다.

- **Technical Details**: 변환기 기반 모델(Transformer-based models)과 대형 언어 모델(LLMs)을 사용하여 작성된 문서의 패턴을 인식하고 중요한 정보를 추출하는 능력이 강조됩니다. 이 모델들은 소셜 미디어, 블로그 및 GitHub 이슈와 같은 다양한 출처의 내용을 자동으로 분석하여 잠재적인 취약점을 식별할 수 있도록 합니다. 이러한 접근법은 코드를 포함한 여러 프로그래밍 언어에 대한 코드 분석과 생성에서 탁월한 성능을 발휘합니다.

- **Performance Highlights**: 실험 결과, 변환기 기반 모델이 통계적 접근법에 비해 약 2배 더 높은 정확도를 보이는 것으로 나타났습니다. 이 연구는 자동화된 취약점 탐지를 위한 경제적이고 확장 가능한 프레임워크를 제공하여, 취약점 공개 전에 손상된 소프트웨어 사용을 예방할 수 있는 가능성을 지니고 있습니다. 이 논문은 오픈 소스 소프트웨어 생태계의 보안을 강화하는 데 기여할 것으로 기대됩니다.



### From Scientific Texts to Verifiable Code: Automating the Process with Transformers (https://arxiv.org/abs/2501.05252)
- **What's New**: 이번 논문은 기존의 알고리즘을 형식적으로 보증하는 코드의 부족함을 해결하기 위해 트랜스포머 모델을 활용하는 방안을 제안합니다. 저자들은 연구 논문에서 제안된 알고리즘의 증명을 읽어서 이를 검증 가능한 코드로 변환할 수 있는 가능성을 탐구하고 있습니다. 이 접근법은 형식적 검증의 장벽을 줄이고 복잡한 시스템의 검증을 자동화하는 새로운 방법을 제시합니다.

- **Technical Details**: 형식적 검증은 특정 형식 언어에 익숙합니까?가 필요하고, 세부 사항을 명시적으로 확인하는 것은 매우 번거롭고 시간이 많이 소모되는 작업입니다. 저자들은 트랜스포머가 이 문제를 해결하는 도구로 적합하다고 주장하며, 서로 다른 두 단계를 통해 증명 구조를 고수준 검증 코드로 변환하는 방법을 제안합니다. 첫 번째 단계에서는 증명의 골격을 작성하고, 두 번째 단계에서는 저수준의 검증 가능한 증명을 생성합니다.

- **Performance Highlights**: 저자들은 Prometheus라는 프로토타입을 사용하여 세 가지 네트워크 위상 속성을 검증하는 실험을 수행하였으며, 기존의 LLM 기반 검증기보다 뛰어난 성능을 보였습니다. 이 시스템은 형식적 검증과 저수준 검증의 리소스를 효과적으로 분리하여 더 높은 정확성을 달성할 수 있게 해줍니다. 향후 이러한 접근법을 통해 여러 분야의 알고리즘을 자동으로 검증할 수 있는 가능성을 열어주고 있습니다.



### RAG-WM: An Efficient Black-Box Watermarking Approach for Retrieval-Augmented Generation of Large Language Models (https://arxiv.org/abs/2501.05249)
- **What's New**: 이번 연구에서는 Retrieval-Augmented Generation (RAG) 시스템의 지적 재산권(IP) 침해를 탐지하기 위한 새로운 접근법인 RAG-WM을 제안했습니다. 기존의 워터마크 기술은 관계형 데이터베이스와 텍스트를 위한 것이지만, RAG의 지식 기반에서는 적용할 수 없었습니다. RAG-WM은 블랙박스 방식으로 지식 워터마크를 생성하여 IP 침해를 효과적으로 탐지할 수 있는 방법을 개발했습니다.

- **Technical Details**: RAG-WM은 멀티 LLM 상호작용 프레임워크를 사용하여 워터마크 생성기(Watermark Generator), 섀도우 LLM 및 RAG(Shadow LLM & RAG), 워터마크 분류기(Watermark Discriminator)로 구성됩니다. 이 시스템은 워터마크 엔티티-관계 튜플을 기반으로 한 워터마크 텍스트를 생성하고, 이를 목표 RAG에 주입하여 IP 침해를 탐지합니다. 실험은 세 가지 도메인 특화(task-specific)과 두 가지 프라이버시 민감(task-sensitive) 작업에서 수행되었습니다.

- **Performance Highlights**: 실험 결과, RAG-WM은 다양한 LLM에 배포된 훔친 RAG를 효과적으로 탐지하는 것으로 나타났습니다. 또한, RAG-WM은 패러프레이징(paraphrasing), 무관한 내용 제거(unrelated content removal), 지식 삽입(knowledge insertion) 및 지식 확장(knowledge expansion) 공격에 대해 강력한 내성을 보였습니다. 마지막으로, RAG-WM 자체가 워터마크 탐지 방법을 회피할 수 있어, RAG 시스템의 IP 침해 탐지에 유망한 응용 가능성을 보여주었습니다.



### Deriving Coding-Specific Sub-Models from LLMs using Resource-Efficient Pruning (https://arxiv.org/abs/2501.05248)
- **What's New**: 이번 연구는 Large Language Models (LLMs)의 컴팩트한 모델 생성을 위한 비구조적 프루닝(unstructured pruning) 기법, 특히 코드 생성에 특화된 서브 모델을 효과적으로 추출하는 방법을 탐구합니다. 기존의 접근 방식이 프로그래밍 언어에 특화된 서브 모델의 효율적인 추출에 집중하지 않았던 점을 지적하며, Python, Java, C++, JavaScript와 같은 언어별 서브 모델을 추출하는 데 성공했습니다. 또한, 도메인 특화된 데이터셋이 프루닝 결과에 미치는 영향을 조사함으로써, 특정 프로그래밍 가지에 대한 LLM의 활성화 지역이 다름을 최초로 분석하고 증명했습니다.

- **Technical Details**: 연구에서는 다양한 도메인 특화된 캘리브레이션 데이터셋이 각기 다른 프루닝 결과에 미치는 영향을 조사했습니다. 이를 통해, 모델의 정확도를 유지하면서 언어별 서브 모델을 효과적으로 추출할 수 있는 프루닝 기법을 제안합니다. 특히, 차세대 LLM의 접근성을 높이기 위해 소비자 수준의 하드웨어에서의 실행을 지원하고, 실시간 개발 피드백에 중요한 빠른 추론 시간(inference time)을 가능하게 하는 방법을 모색했습니다.

- **Performance Highlights**: 논문은 LLM이 특정 도메인 작업을 통해 활성화되는 고유한 지역이 있음을 나타내는 분석적 증거를 제공합니다. 이를 통해 프로그래밍 언어 별 모델의 효율적 추출이 가능함을 보여주어, 다양한 프로그래밍 언어에 대한 요구를 충족할 수 있는 가능성을 열어줍니다. 결과적으로, 모델의 크기를 줄이면서도 기존 모델에 비해 만족할 만한 정확도를 확보하여 사용자가 보다 효율적으로 코딩 작업을 수행할 수 있도록 기여할 것으로 기대됩니다.



### Optimizing Estonian TV Subtitles with Semi-supervised Learning and LLMs (https://arxiv.org/abs/2501.05234)
- **What's New**: 이 논문은 에스토니아 TV 콘텐츠의 높은 품질의 동일 언어 자막 생성 접근법을 제시합니다. Whisper 모델을 인간이 생성한 에스토니아 자막에 맞게 미세 조정하고, 반복적인 pseudo-labeling과 대형 언어 모델(LLM) 기반 후편집을 통해 품질을 향상시켰습니다. 실험 결과, 무언급 데이터셋을 통한 pseudo-labeling에서 자막 품질의 유의미한 개선이 있음을 보였습니다.

- **Technical Details**: 본 연구는 자동 자막 생성 시스템 개발을 위한 방법을 여러 단계로 나누어 설명합니다. 감독 데이터셋을 사용하여 Whisper large-v3 모델을 훈련시키고, 이후 비감독 데이터셋을 활용하여 반복 pseudo-labeling을 적용합니다. LLM 기반 후편집을 통해 생성된 자막의 오류를 수정하며, 이러한 방법을 테스트 시간과 훈련 시간에 각각 적용하여 성능을 평가합니다.

- **Performance Highlights**: 에스토니아 국가 TV의 감독 데이터셋에서 수집된 993개의 오디오-자막 쌍과 7128개의 비감독 오디오 기록을 사용하여 성능을 평가했습니다. 자막 품질 평가를 위해 SubER, t-BLEURT, AS-BLEURT의 세 가지 메트릭을 사용하였으며, 자막의 품질을 인간의 평가와 비교하였습니다. 결과적으로, LLM 기반의 게시 수정이 자막 품질을 크게 향상시킬 수 있음을 확인했습니다.



### A Novel Approach to Scalable and Automatic Topic-Controlled Question Generation in Education (https://arxiv.org/abs/2501.05220)
Comments:
          To be published at ACM Conf. on Learning Analytics and Knowledge (LAK'25)

- **What's New**: 본 논문은 Topic-Controlled Question Generation (T-CQG)이라는 새로운 접근 방식을 제시하여 교육 목적으로 생성된 질문의 관련성과 효과성을 향상시킵니다. T-CQG는 pre-trained T5-small 모델을 미세 조정하여 특별히 만들어진 교육 맞춤형 데이터셋을 사용합니다. 연구는 사전 훈련 전략, 양자화(quantisation), 데이터 증대(data augmentation) 등이 모델 성능에 미치는 영향을 탐구하며, 문단 수준의 맥락에 맞는 의미적으로 일치하는 질문 생성을 다룹니다. 또한, 생성된 질문의 주제 관련성을 평가하기 위한 새로운 평가 방법을 소개합니다.

- **Technical Details**: T-CQG 모델은 고급 자연어 처리(NLP) 기술을 활용하여 교육 질문의 생성을 자동화합니다. 이 모델은 약 60M 파라미터를 가진 매우 작은 언어 모델(small language model)을 사용하여 교육용 질문을 생성하는 데 성공했습니다. 주제 제어(question generation)의 중요성을 강조하며, 학습 자료의 맥락과 관련된 주제를 기준으로 질문을 생성하는 방법론을 제시합니다. 이러한 접근은 교육용 질문의 품질을 개선하고 교사들이 질문을 작성하는 데 드는 수고를 덜어줄 수 있습니다.

- **Performance Highlights**: 연구 결과는 오프라인 실험 및 인간 평가(human-backed evaluations)를 통해 검증되었으며, T-CQG 모델이 고품질의 주제 중심 질문을 효과적으로 생성할 수 있음을 입증합니다. 교육 시스템에서의 적용 가능성이 높아 교사들의 업무 부담을 줄이고 개인 맞춤형 튜터링 시스템을 지원하는 데 기여할 것으로 기대됩니다. 본 연구는 교사 retention 문제와 관련된 생산성 향상을 위한 중요한 실용적 단계를 제공하여 교육 환경을 개선할 가능성이 큽니다.



### GLaM-Sign: Greek Language Multimodal Lip Reading with Integrated Sign Language Accessibility (https://arxiv.org/abs/2501.05213)
Comments:
          9 pages, 4 figures

- **What's New**: 그리스어 다중모드 입술 읽기와 통합 수화 접근성을 위한 GLaM-Sign은 청각 장애인과 난청인을 지원하기 위해 개발된 혁신적인 자료입니다. FEELIT 프로젝트를 기반으로 하여, 고해상도 오디오, 비디오, 텍스트 전사 및 그리스 수화 번역을 통합하고 있습니다. 이 시스템은 그리스 관광 부문에서의 포용성을 증가시키는 데 주력하고 있지만, 교육, 의료 및 공공 서비스로의 적용 가능성도 있습니다.

- **Technical Details**: GLaM-Sign은 실시간 수화 번역(real-time sign language translation) 및 자막 동기화(enhanced subtitle synchronization)를 위한 여러 기술적 요소를 포함하고 있습니다. 고해상도 미디어와 텍스트 및 수화 정보의 융합을 통해 데이터 세트(data set)는 다중 모드(multimodal) 접근 방식의 효용성을 보여줍니다. 향후 발전은 단어 수준의 정확성과 다양한 언어로의 확장을 목표로 하며, 선진 AI 방법론(advanced AI methodologies) 및 다양한 이해관계자와의 협력을 통해 이루어질 것입니다.

- **Performance Highlights**: 이 데이터 세트는 의사소통 격차를 해소하고 혁신을 촉진하는 다중 모드 자원의 변혁적 가능성을 보여줍니다. GLaM-Sign은 다양한 영역에서 높은 접근성을 제공하여 포용성을 촉진하는 데 기여하고 있습니다. 또한 윤리적 AI 및 포괄적인 기술을 위한 기준을 설정하며, 앞으로의 연구와 개발의 표준이 될 것으로 기대됩니다.



### Discovering Hidden Visual Concepts Beyond Linguistic Input in Infant Learning (https://arxiv.org/abs/2501.05205)
Comments:
          12 pages, 11 figures

- **What's New**: 이 연구에서는 아기들이 언어 입력을 습득하기 전에 시각적 이해를 빠르게 발전시키는 과정을 모델링하여, 아기들의 학습 과정을 모방한 컴퓨터 비전 모델이 언어 데이터 범위를 넘어서는 더 넓은 시각적 개념을 개발할 수 있는지를 탐구합니다. CVCL(Child’s View for Contrastive Learning) 모델을 사용하여, 아기들이 부모의 말과 연관된 시각적인 프레임을 이해하도록 훈련되었습니다. 이 연구는 NeuronClassifier라는 훈련이 필요 없는 새로운 프레임워크를 소개하여, 모델의 내부 표현에서 숨겨진 시각적 개념을 발견했습니다.

- **Technical Details**: CVCL은 6개월에서 25개월 사이의 아기의 영상 데이터를 이용해 훈련된 모델로, 부모가 말한 내용을 아기의 시각적 경험과 결합하여 이해합니다. 이를 통해 저자들은 아기의 시각적 개념이 언어적 입력을 초월해 발전할 수 있음을 제안합니다. 특히, 이 모델의 내부 표현 분석을 통해 'neuron labeling' 기법을 이용해 특정 뉴런들이 어떻게 시각적 개념을 형성하는지를 연구했습니다.

- **Performance Highlights**: CVCL 모델은 CLIP 및 ImageNet 모델과 비교했을 때 여전히 인식 능력이 제한적이지만, 숨겨진 시각적 개념을 통해 언어 훈련 자료에 명시되지 않은 고급 개념을 분류하는 데 성공했습니다. 발견된 'out-of-vocabulary' 단어들은 아기들이 배우는 단어 보다는 높은 인지적 수준을 나타내며, 이는 아기들이 시각적 개념을 언어적 이해보다 먼저 개발한다는 인지 과학 연구와 일치합니다.



### An Algorithmic Approach for Causal Health Equity: A Look at Race Differentials in Intensive Care Unit (ICU) Outcomes (https://arxiv.org/abs/2501.05197)
- **What's New**: 이번 연구는 건강 불평등을 분석하기 위한 체계적인 프레임워크를 제안합니다. 호주와 미국의 ICU 결과에서 인종 및 민족적 불평등을 조사하여, 기존의 통계적 지표들이 불평등을 측정하는 데 부족하다는 점을 확인했습니다. 특히, 소수 환자들이 입원 시 더 젊고 만성 건강 상태가 좋지 않으며 긴급 및 긴급성이 없는 이유로 더 많이 입원하고, 높은 질병 중증도를 겪는다고 합니다.

- **Technical Details**: 연구는 호주와 미국의 ICU 데이터를 분석하였으며, 인종/민족별 차이를 이해하기 위해 인과적 관점에서 접근했습니다. 인과적 공정성 분석(framework of causal fairness analysis)을 통해 각 인과적 경로의 기여도를 평가, 분류하였고, SFM(Standard Fairness Model)을 사용하여 변수 간의 관계를 명확히 했습니다. 연구 결과, 호주 소수 환자들은 ICU 입원 후 사망률이 높았고, 미국에서는 반대의 경향을 보였습니다.

- **Performance Highlights**: 호주 원주민 환자들은 ICU에 자주 입원하지만 사망률이 낮은 반면, 미국의 아프리카계 미국인 환자들은 상대적으로 높은 사망률을 기록하였습니다. 조사 결과, 소수민족 환자들은 백인 동료들에 비해 ICU 재입원 가능성이 높았으며, 이는 기본적인 건강 관리 접근의 부족에 기인함을 밝혔습니다. 연구의 결과를 바탕으로 Indigenous Intensive Care Equity (IICE) 레이더 시스템을 개발하여, 원주민의 과도한 ICU 자원 활용을 모니터링할 수 있게 되었습니다.



### Bringing Order Amidst Chaos: On the Role of Artificial Intelligence in Secure Software Engineering (https://arxiv.org/abs/2501.05165)
Comments:
          PhD thesis

- **What's New**: 이 연구는 소프트웨어 보안 엔지니어링(SSE)에서 인공지능(AI)의 적용을 통해 예측 모델의 개선을 목표로 하고 있습니다. 특히, 결함 예측 및 취약성 탐지와 우선 순위 지정의 두 가지 특정 영역에 초점을 맞추며, AI 활용의 맥락에 따른 고유한 차이를 고려합니다. 이는 소프트웨어 시스템의 보안을 강화하고, AI 기반의 예측 모델의 정확성을 높이기 위한 중요한 기여로 평가됩니다.

- **Technical Details**: 연구 방법론으로는 경험적 전략을 활용하며, 이것은 Static Application Security Testing Tools (SASTTs)의 분석과 JIT(Just-In-Time) 결함 예측을 포함합니다. Effort-aware metrics (EAMs)를 통해 결함 예측의 정확성을 향상시키고, 데이터 세트의 체계적인 리뷰를 통해 예측 모델의 평가 및 개선을 꾀하고 있습니다. 이를 통해 SASTTs의 취약성 종류 커버리지 및 정적 분석 도구의 한계를 지적하고 있습니다.

- **Performance Highlights**: 결과적으로, EAMs의 정규화와 JIT 정보 활용을 통해 결함 예측 정확성을 극대화하는 모델이 도출되었으며, AI 기반의 위험 평가에서의 대규모 언어 모델(LLM)의 효용성을 입증하였습니다. 이러한 기여는 결함 예측과 취약성 탐지의 효과성을 높이며, 연구자와 실무자 모두에게 이점을 제공하고 있습니다. 또한, 보안 부채를 전반적으로 줄이기 위한 취약성 우선 순위 지정 프레임워크를 제시하여 조직의 보안 태세를 강화하도록 돕고 있습니다.



### Explainable AI based System for Supply Air Temperature Forecas (https://arxiv.org/abs/2501.05163)
Comments:
          5 pages, 7 figures, 1 table, conference paper

- **What's New**: 이 논문은 자동 공급 공기 온도(ASAT) 제어를 위한 예측 모델의 투명성과 이해를 향상시키기 위해 Explainable AI(XAI) 기술의 적용을 탐구합니다. 제어 곡선의 의미적 및 물리적 설명이 부족한 경우가 많기 때문에, 연구는 Huber 손실을 사용한 선형 회귀를 통해 ASAT를 예측합니다. XAI 방법 중 하나인 Shapley 값을 활용하여 각 특징이 최종 ASAT 예측에 미치는 기여도를 강조합니다.

- **Technical Details**: 불변량 시계열 데이터는 동일한 타임스탬프에서 기록된 단일 스칼라 관측값으로 구성됩니다. 이러한 데이터는 ARIMA(Autoregressive Integrated Moving Average), SARIMA(Seasonal ARIMA), Holt-Winters 및 칼만 필터링과 같은 전통적인 통계 분석 방법으로 모델링할 수 있습니다. 연구는 시계열 예측에 관계형 신경망(RNN) 및 LSTM 모형이 덜 정확하다는 것을 보여주며, 간단한 기계 학습 방법이 때때로 더 효과적일 수 있음을 제안합니다.

- **Performance Highlights**: 연구는 Shapley 값을 활용하여 회귀 모델이 ASAT를 예측하는 방식을 명확히 합니다. 이 방법은 다양한 특징이 ASAT 제어 곡선에 미치는 영향을 드러내며, 물리적 및 의미적 이유에 대한 통찰을 제공합니다. 결과적으로, 연구는 ASAT 값의 변화에 대한 투명성을 높이며, 필드 전문가들이 각 특징이 결과에 어떻게 영향을 미치는지를 이해하는 데 유용한 정보를 제공합니다.



### Biomedical Relation Extraction via Adaptive Document-Relation Cross-Mapping and Concept Unique Identifier (https://arxiv.org/abs/2501.05155)
Comments:
          13 pages, 6 figures

- **What's New**: 이 논문에서는 Document-Level Biomedical Relation Extraction (Bio-RE) 개선을 목표로 하는 새로운 프레임워크를 제안합니다. LLM(대규모 언어 모델) Adaptive Document-Relation Cross-Mapping (ADRCM) Fine-Tuning과 Concept Unique Identifier (CUI) Retrieval-Augmented Generation (RAG) 기술을 활용합니다. 특히, 데이터 부족 문제를 해결하기 위해 Iteration-of-REsummary (IoRs) 프롬프트를 도입하여 ChatGPT를 통해 특정 엔터티 관계에 초점을 맞춘 합성 데이터를 생성하는 방법을 제시합니다.

- **Technical Details**: ADRCM Fine-Tuning은 서로 다른 문서와 관계 간의 맵핑을 설정하여 모델의 문맥 이해도와 교차 문장 추론 능력을 향상시킵니다. CUI RAG는 엔터티에 대한 인덱스로 CUI를 활용하여 정보 검색 범위를 좁히고 관련 문맥을 풍부하게 만듭니다. 또한 IoRs 프롬프트는 엔터티 관계에 집중하여 데이터의 일반화 및 정확성을 향상시키는 데 도움을 줍니다.

- **Performance Highlights**: 제안된 방법은 GDA, CDR, BioRED의 세 개 Bio-RE 데이터셋에서 실험하여 기존의 연구들과 비교했을 때 최첨단 성능을 달성했습니다. 특히, 문서 수준의 Bio-RE 작업에서 모델이 표기된 관계를 효과적으로 추론할 수 있도록 하는 것이 주요 성과로 드러났습니다. 이로 인해 향후 Bio-RE 분야의 발전에 큰 기여를 할 것으로 기대됩니다.



### A Systematic Literature Review on Deep Learning-based Depth Estimation in Computer Vision (https://arxiv.org/abs/2501.05147)
- **What's New**: 이 논문은 Depth Estimation (DE) 기술에 대한 체계적인 문헌 리뷰(SLR)를 제공합니다. 최근 딥러닝(Deep Learning) 기반 방법이 DE에 사용되면서, 기존의 전통적인 기법과 비교할 때 이점이 두드러집니다. 특이한 점은 기존의 리뷰가 대부분 단안(Monocular) 또는 입체(Stereo) 기술에 국한되어 있었으며, 포괄적인 DE 분석이 부족했다는 것입니다.

- **Technical Details**: 연구에서는 1284개의 관련 출판물을 선별한 후, 128개의 논문을 정제하여 59개의 고품질 핵심 연구로 축소하였습니다. 이들 연구를 통해 다양한 DE 방식인 단안, 입체, 다중 시점(Multi-view)에 대한 딥러닝 모델을 개발하였습니다. 20개의 공개 데이터셋이 DE 모델의 교육, 테스트 및 평가에 사용되었으며, 가장 많이 사용된 데이터셋은 KITTI, NYU Depth V2 및 Make 3D입니다.

- **Performance Highlights**: DE의 성능을 평가하기 위해 29개의 평가 메트릭스가 사용되었습니다. 35개의 기본 모델(Base Model)이 보고되었으며, 가장 많이 사용된 상위 5개 모델은 ResNet-50, ResNet-18, ResNet-101, U-Net 및 VGG-16입니다. 주요 연구에서 가장 큰 도전으로는 정답 데이터(Ground Truth Data)의 부족이 지적되었습니다.



### Constrained Optimization of Charged Particle Tracking with Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2501.05113)
- **What's New**: 이 논문은 다중 에이전트 강화 학습(MARL)을 기반으로한 입자 추적 재구성을 위한 접근 방식을 제안합니다. 기존의 연구를 바탕으로 하여, 파라미터화된 정책을 최적화하고, 읽기 프레임 내 입자 산란을 최소화하는 것을 목표로 합니다. 이 과정에서 고유한 입자 할당을 보장하는 안전 레이어를 제안하여 각 조인트 액션에 대해 선형 할당 문제를 해결합니다.

- **Technical Details**: 본 연구에서는 중앙 집중식 비평자 구조를 사용하여 분산 에이전트를 훈련하는 방법을 도입하고, 각 에이전트는 로컬 관찰만을 기반으로 작동하는 부분 관찰 가능 마르코프 결정 프로세스(Dec-POMDP)를 사용하여 다중 층의 입자 추적을 진행합니다. 이러한 과정은 에이전트의 예측과 결정 경계 간의 비용 여유를 증가시켜 훈련 및 일반화 성능을 향상시킵니다. 안전성을 보장하기 위해 안전 레이어를 구축하고 정책의 예측을 최적화합니다.

- **Performance Highlights**: 시뮬레이션 데이터에 대한 성능 실험을 통해 제안된 방법의 효과성을 입증하였습니다. 기존의 단일 및 다중 에이전트 기반 솔루션과 비교하여 우수한 성능을 보였으며, 비용 여유 제약이 최적화와 일반화에 효과적임을 보여주었습니다. 이러한 결과는 강화 학습 기반 추적 알고리즘의 향후 발전 가능성을 제시하며, 제약이 있는 정책과 유연성을 통한 추적 알고리즘의 최적화 가능성을 높입니다.



### Advancing ALS Applications with Large-Scale Pre-training: Dataset Development and Downstream Assessmen (https://arxiv.org/abs/2501.05095)
- **What's New**: 이 연구는 항공 레이저 스캐닝(ALS) 기술을 위한 대규모 데이터셋을 구성하고 이를 통해 다운스트림(downstream) 애플리케이션에 미치는 영향을 평가합니다. 기존의 대규모 데이터셋이 ALS 응용 분야에 충분한 다양성과 규모를 제공하지 못했던 반면, 이 연구는 미국 지질조사국(USGS)의 3D 고도 프로그램을 기반으로 하는 다각적 지리적 샘플링 방법을 도입하였습니다. 이 방법은 다양한 토지 피복과 지형을 포괄하는 ALS 포인트 클라우드를 효과적으로 수집할 수 있게 합니다.

- **Technical Details**: 항공 레이저 스캐닝(ALS)은 공중 플랫폼에서 레이저 펄스를 방출하고 반사된 신호를 분석하여 고해상도의 3D 공간 데이터를 수집하는 기술입니다. 이 연구에서는 BEV-MAE라는 최신 자가 지도 학습(self-supervised learning) 모델을 기반으로 하여, 구성된 포인트 클라우드 데이터셋에 사전 학습(pre-training)을 수행하고, 이를 기반으로 나무 종 분류, 지형 장면 인식 및 포인트 클라우드 의미적 분할이라는 다운스트림 작업에 대해 미세 조정(fine-tuning)을 진행합니다. 또한 제안된 샘플링 방법을 통해 데이터셋의 규모를 확장하여 성능 향상을 모니터링 하는 방법을 제시합니다.

- **Performance Highlights**: 사전 학습된 모델은 모든 다운스트림 작업에서 스크래치 모델에 비해 월등한 성과를 나타내었으며, 이는 제안된 데이터셋에서 학습한 표현의 전이 가능성을 입증합니다. 데이터셋의 확장을 통해 성능이 일관되게 개선되는 것을 관찰하였으나, 무작위 샘플링으로 구성된 데이터셋에서 사전 학습을 수행한 경우에는 유사한 성능 향상을 달성하지 못했습니다. 연구 결과는 ALS 응용을 위한 사전 학습 및 미세 조정 패러다임의 유용성을 강조합니다.



### Analyzing Memorization in Large Language Models through the Lens of Model Attribution (https://arxiv.org/abs/2501.05078)
- **What's New**: 본 연구에서는 Large Language Models (LLMs)의 기억 현상을 구조적 관점에서 분석하고자 하였다. 주목할 점은 Attention 모듈이 메모리화와 일반화 성능에 미치는 영향을 검토하며, 특정 블록에서 Attention 모듈을 우회하는 방법을 제시하고 있다는 것이다. 이는 LLMs의 메모리화 문제를 해결하기 위한 기초적인 메커니즘을 이해하는 데 도움을 준다.

- **Technical Details**: 우리는 특정 Attention 블록을 우회하면서도 레이어 정규화(layer normalization)와 MLP 변환(MLP transformation) 등의 다른 구성 요소는 그대로 유지하여 메모리화와 일반화의 영향을 분리할 수 있는 체계적인 개입을 수행한다. 이론적 분석을 통해 Attention 모듈을 우회했을 때 출력 차이를 한정짓는 정리를 제공하며, 깊은 변환기 블록에서 메모리화가 주로 발생한다는 결론을 이끌어냈다.

- **Performance Highlights**: Pythia와 GPT-Neo 모델 패밀리, 그리고 여러 벤치마크 데이터셋을 통한 실험을 통해, 더 깊은 레이어에서 Attention 모듈을 우회하는 것이 메모리화 감소와 일반화 성능 유지에 효과적임을 증명하였다. 본 연구는 모델의 일반화 능력을 해치지 않으면서 메모리화를 완화할 수 있는 실용적인 접근 방식을 제시하고, 실제 응용에서의 윤리적 배치를 지원하는 기여를 한다.



### Commonsense Video Question Answering through Video-Grounded Entailment Tree Reasoning (https://arxiv.org/abs/2501.05069)
- **What's New**: 본 논문은 비디오 기반의 일반 상식 질문 대답(Video Question Answering, VQA)을 위한 최초의 비디오 기초 추론 방법을 제안합니다. 기존의 대형 비주얼-언어 모델(Visual-Language Models, VLMs)에서 발생하는 단기적인 연관 학습 문제를 해결하기 위하여, 해당 방법은 VQA 작업을 비디오 조각에 명시적으로 연결합니다. 이를 통해 모델의 의사 결정 과정을 설명할 수 있는 명확한 추론 체인을 제공합니다.

- **Technical Details**: 제안하는 방법은 VQA 작업을 네 가지 단계로 수행합니다: (i) 추론 트리 구축, (ii) 비디오-언어 추론 검증, (iii) 트리 추론, (iv) 동적 트리 확장. 각 단계는 비디오와 다중 선택 질문을 사용하여 후보 답변을 명시적으로 검증하고, 이를 통해 비디오의 관련 정보와 정합성을 확인합니다. 이 과정에서 VLMs에 적용할 수 있는 일반화 가능성을 강조합니다.

- **Performance Highlights**: 실험 결과, 제안하는 비디오 기반 추론 트리는 기존의 비디오 및 이미지 기반 모델에 비해 더 나은 성능을 보여주었습니다. 또한, 제안된 방법은 텍스트와 비디오 정보를 함께 고려함으로써 응답의 강인성과 해석 가능성을 향상시켰고, 특히 인과적 및 시간적 질문에서 두드러진 성과를 나타냈습니다.



### D3RM: A Discrete Denoising Diffusion Refinement Model for Piano Transcription (https://arxiv.org/abs/2501.05068)
Comments:
          Accepted to ICASSP 2025

- **What's New**: 이 논문에서는 피아노 악보의 자동 변환을 위한 새로운 구조의 이산 확산 모델(discrete diffusion model) 아키텍처를 제안합니다. Neighborhood Attention 층을 디노이징 모듈로 사용하는 접근 방식을 통해 고해상도 피아노 롤을 점진적으로 예측하는 연구가 진행되었습니다. 또한, 이산 확산 모델의 훈련 및 추론 단계에서 적용되는 독특한 전이 상태를 이용하여 정제(refinement) 능력을 강화하는 새로운 전략도 소개되었습니다.

- **Technical Details**: 제안된 모델은 효과적으로 이전에 예측된 상태를 정제하여 개별 음의 최종 예측이 다른 노트의 예측을 더 잘 반영하도록 설계되었습니다. 또한, 피아노 롤의 각 픽셀을 통해 이웃 노트를 고려할 수 있도록 하는 Neighborhood Attention 메커니즘이 도입되어 디노이징 과정에서 iterative refinement가 가능합니다. 모델 아키텍처는 pitchwise bidirectional LSTM과 NA 2D self-attention 블록으로 구성되어 있으며, 고해상도 피아노 롤 예측에 최적화되었습니다.

- **Performance Highlights**: MAESTRO 데이터셋에서의 실험 결과, 제안된 접근 방식이 이전의 확산 기반 피아노 변환 모델 및 기반 모델에 비해 F1 점수에서 우수한 성능을 보였습니다. 본 연구는 기존의 피드포워드 모델의 한계를 극복하고, 디지털 음악 변환 분야의 최신 기술적 발전에 기여하고 있습니다.



### LLaVA-Octopus: Unlocking Instruction-Driven Adaptive Projector Fusion for Video Understanding (https://arxiv.org/abs/2501.05067)
- **What's New**: 본 논문에서는 사용자 지침에 따라 다양한 시각적 프로젝터의 기능을 적절히 가중하는 새로운 비디오 다중 모달 대형 언어 모델인 LLaVA-Octopus를 소개합니다. 이 모델은 각 프로젝터의 강점을 활용하여 다중 모달 작업의 성능을 크게 향상시키며, 사용자 지침을 기반으로 기능 가중치를 동적으로 조정합니다. 실험 결과, LLaVA-Octopus는 멀티모달 이해, 비주얼 질문 응답 및 비디오 이해와 같은 작업에서 탁월한 성능을 달성했습니다.

- **Technical Details**: 최근 비디오 이해의 주요 과제는 시간 동적성 관리와 복잡한 의미 이해입니다. 다양한 시각적 프로젝터가 특정 작업을 처리하는 데서 고유한 성능을 보여주며, 본 논문에서는 이를 크게 이미지 기반 프로젝터, 공간-시간 프로젝터 및 토큰 압축 프로젝터로 분류합니다. LLaVA-Octopus는 이들 서로 다른 프로젝터의 기능을 통합하는 입력 기반의 프로젝터 융합 패러다임을 제안하며, 각 프로젝터의 상호 보완적인 장점을 최대한 활용합니다.

- **Performance Highlights**: LLaVA-Octopus는 다양한 벤치마크에서 최고의 성능을 기록하며, 특히 비디오 및 시각적 데이터를 다루는 멀티모달 작업에서 다른 모델들과 비교해 우수한 결과를 보여주고 있습니다. 실험 결과를 통해 대다수의 벤치마크에서 최첨단(State-of-the-Art, SOTA) 성능을 달성한 것으로 나타났습니다. 이 모델은 다중 모달 작업에서의 범용적인 활용 가능성을 강조하며, 향후 연구 및 응용 분야에 기여할 것으로 기대됩니다.



### Improving Skeleton-based Action Recognition with Interactive Object Information (https://arxiv.org/abs/2501.05066)
- **What's New**: 본 논문에서는 기존 스켈레톤 기반 행동 인식 방법의 한계를 극복하기 위해 객체 노드를 도입한 새로운 행동 인식 프레임워크를 제안합니다. 이를 통해 인간과 상호작용하는 객체의 정보를 보강하여 동작 인식 성능을 높이는 것을 목표로 합니다. 특히 Spatial Temporal Variable Graph Convolutional Networks (ST-VGCN)라는 모델을 제안하여 객체 노드를 포함한 변동 그래프를 효과적으로 모델링합니다.

- **Technical Details**: 우리가 제안한 ST-VGCN은 개체 노드를 포함하여 인간-개체 간의 관계를 학습하는 새로운 프레임워크입니다. 이 모델은 Random Node Attack이라는 데이터 증강 방법을 통해 객체 정보에 의해 발생하는 데이터 편향 문제를 해결하고, 별도의 Variable Graph 구성 방법을 도입하여 동작 인식에서의 유연성을 높입니다. 이를 통해 스켈레톤 노드와 객체 노드를 균형 있게 활용하여 네트워크의 일반화 성능을 향상시킵니다.

- **Performance Highlights**: 제안된 방법은 NTU RGB+D 60 데이터셋에서 cross-subject split에서 96.7%, cross-view split에서 99.2%의 정확도로 이전의 최첨단 성과를 초과했습니다. 이와 같은 결과는 다양한 스켈레톤 기반 행동 인식 벤치마크에서 우리의 접근법이 효과적임을 입증합니다. 향후 연구는 이 모델을 기반으로 추가적인 객체와 상호작용하는 행동 인식의 가능성을 탐구할 것입니다.



### Simultaneous emulation and downscaling with physically-consistent deep learning-based regional ocean emulators (https://arxiv.org/abs/2501.05058)
- **What's New**: 이번 연구는 AI 기반의 해양 시뮬레이션과 다운스케일링 프레임워크를 제안하여 멕시코만의 고해상도 지역 해양 모델링을 목표로 하고 있습니다. 기존의 대기 시뮬레이션의 성공을 바탕으로 복잡한 수심 및 경계 조건을 고려한 새로운 모델을 개발하였으며, 이를 통해 공간 해상도를 8Km에서 4Km로 향상시킬 수 있었습니다. 이 프레임워크는 예측의 안정성을 유지하면서도 물리적으로 일관된 결과를 도출할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 연구진은 해양 표면 변수의 자가 회귀 예측을 위해 Fourier Neural Operators (FNO)와 같은 고급 기계 학습 기술을 활용하였습니다. 특히, 해수면 높이(SSH), 속도(SSU), 운동 에너지(SSKE)를 포함한 딥 러닝 모델을 이용해 멕시코만의 해양 동역학을 통합하는 방법론을 개발하였으며, 두 가지 서로 다른 신경망 아키텍처를 비교 분석하였습니다. 모델의 성능 측정을 위해 저해상도(GLORYS) 자료로부터 고해상도(CNAPS) 데이터로 다운스케일링하는 과정에서 물리적 일관성을 고려한 구조도 적용하였습니다.

- **Performance Highlights**: 연구 결과, 제안된 모델은 해양 예측의 효율성을 크게 향상 시키며, 단기 및 장기 통계에서도 정확성이 뛰어난 것으로 나타났습니다. 특히, 낮은 해상도의 입력 변수에 비해 수천 배 빠른 예측 속도를 보였으며, 데이터 기반의 기술이 과학적 예측에 신뢰성을 제공할 수 있다는 점을 확인하였습니다. 이러한 접근법은 빠른 지역적 분석과 통계적 예측 생성에 유용할 것으로 기대됩니다.



### TAPFed: Threshold Secure Aggregation for Privacy-Preserving Federated Learning (https://arxiv.org/abs/2501.05053)
Comments:
          The paper has been published in IEEE TDSC

- **What's New**: 이번 논문에서는 악의적인 행위자가 존재하는 환경에서도 개인 정보 보호를 유지하는 federated learning 방식을 제안합니다. TAPFed라는 방법이 제안되었으며, 이는 여러 분산 집계기에서의 보안성을 유지하여 작동합니다. 기존의 federated learning 방법들이 개인 정보 누출 문제로 고통받고 있는 가운데, TAPFed는 이런 문제를 해결하는 새로운 접근 방식을 제공합니다.

- **Technical Details**: TAPFed는 제안된 threshold functional encryption 방식을 이용하여 특정 수의 악의적인 집계기가 존재하더라도 보안과 개인 정보 보호를 유지할 수 있습니다. 논문에서는 TAPFed의 공식 보안 및 개인 정보 분석을 제공하며, 다양한 기준선들과 비교하는 실험 평가를 수행하였습니다. 이 방법은 기존의 federated learning 시스템에서 발생하는 gradient 교환으로 인한 누출 문제를 해결하는 데 초점을 맞추고 있습니다.

- **Performance Highlights**: 실험 결과, TAPFed는 최신 방법들과 비교할 때 모델 품질 측면에서 동등한 성능을 보이면서도 전송 오버헤드를 29%에서 45%까지 줄였습니다. 특히, TAPFed는 호기심이 많은 집계기로 인해 발생할 수 있는 최근의 추론 공격에 효과적으로 대응할 수 있다는 점이 가장 두드러집니다. 이는 기존 접근 방식들이 처한 취약점을 극복하는 중요한 성과로 평가됩니다.



### Enhancing Human-Like Responses in Large Language Models (https://arxiv.org/abs/2501.05032)
- **What's New**: 이 논문은 대형 언어 모델(LLM)의 자연어 이해, 대화 일관성, 감정 지능을 향상시키는 기술에 초점을 맞추고 있습니다. 우리는 다양한 데이터셋을 통한 미세 조정, 심리학 원칙의 적용, 인간의 추론 패턴을 모방하는 모델 설계 방식을 통합하여 AI 시스템을 더욱 인간처럼 만듭니다. 이러한 기술들은 사용자 상호작용을 개선하고, 다양한 분야에 걸쳐 AI 응용 프로그램의 새로운 가능성을 열어줍니다.

- **Technical Details**: 우리는 Llama 3 70B 및 405B 모델을 활용하여 인간과 같은 대화를 생성하기 위해 합성 데이터셋을 생성했습니다. 이 데이터셋은 질문 생성에 Llama 3 405B를, 답변 생성에 Llama 3 70B를 사용하여 작성되었습니다. 각각의 시스템 프롬프트는 자연스러운 인간의 대화를 모방하거나 더 형식적이고 비인격적인 응답을 생성하도록 설계되었습니다.

- **Performance Highlights**: 모델 훈련에는 LoRA(로우-랭크 어댑테이션)와 DPO(다이렉트 선호 최적화) 기법을 사용했습니다. 이러한 접근법은 모델의 일반 지식을 유지하면서 특정 작업에 적응하도록 도와줍니다. 훈련 결과, 인간과 유사한 대화를 가능하게 하여 AI의 대화 플루언시와 사용자 참여도를 크게 향상시켰습니다.



### Finding Needles in Emb(a)dding Haystacks: Legal Document Retrieval via Bagging and SVR Ensembles (https://arxiv.org/abs/2501.05018)
- **What's New**: 본 연구에서는 독일 법률 정보 검색(GerDaLIR) 데이터셋을 활용하여 Support Vector Regression (SVR) 앙상블 기법과 배깅(bootstrap aggregation)을 적용한 검색 접근 방식을 소개합니다. 우리는 정보 검색 문제를 다수의 이진 하위 작업(needle-in-a-haystack)으로 개념화하여, 훨씬 더 높은 재현율(recall)을 달성했음을 보여주었습니다. 이 방식은 심층 학습 모델을 훈련시키거나 미세 조정(fine-tuning)하지 않고도 가능합니다. 나아가, 인코딩 모델의 개선 및 하이퍼파라미터 최적화를 통해 더 나은 성능 향상이 가능할 것으로 보입니다.

- **Technical Details**: 정보 검색(IR)은 많은 데이터에서 특정 쿼리에 따라 정보 단위를 식별하고 추출하는 중요한 과정입니다. 법률 정보 검색(LIR)에서는 관련된 법률 문서들의 모음에서 특정 문서를 찾아내는 것이 중요하며, TF-IDF와 BM25 같은 다양한 자연어 처리(NLP) 기법이 활용됩니다. 본 연구에서는 여러 개의 약한 Support Vector Regressors(SVR)를 앙상블하여 문서의 입력을 고차원 임베딩 벡터로 변환하고 배깅 기법을 결합하여 성능을 향상시켰습니다.

- **Performance Highlights**: 우리의 접근 방식은 높아진 재현율(0.849)로 이전 최적 모델들보다 더 나은 성능을 나타냈습니다. 특히, 이 결과는 대규모의 심층 학습 모델에 대한 미세 조정이 없었던 상태에서 달성되었습니다. GerDaLIR 데이터셋을 활용한 우리의 방법론은 향후에도 법률 정보 검색(LIR) 분야에서 많은 가능성을 제시할 것으로 보고되었습니다. 마지막으로, 연구 결과에 대한 소스 코드는 GitHub에서 공개되었습니다.



### On Measuring Unnoticeability of Graph Adversarial Attacks: Observations, New Measure, and Applications (https://arxiv.org/abs/2501.05015)
Comments:
          KDD 2025

- **What's New**: 최근 연구에 따르면 그래프 신경망(GNN)은 다양한 작업에서 뛰어난 성능을 발휘하고 있지만, 적대적 공격에 취약한 문제를 안고 있습니다. 이 논문에서는 기존의 noticeability(인식 가능성) 측정 방법의 두 가지 주요 한계를 발견하고 이를 해결하기 위해 HideNSeek라는 새로운 측정 방법을 제안합니다. HideNSeek는 learnable edge scorer (LEO)를 활용하여 공격 엣지를 구별하고, imbalance-aware aggregation을 통해 최종 noticeability 점수를 도출합니다.

- **Technical Details**: HideNSeek는 GNN 기반으로 학습 가능한 엣지 스코어러(LEO)를 통해 각 엣지가 공격일 가능성을 점수화합니다. 이를 통해 공격 엣지가 상대적으로 낮은 점수를 받을 경우, 이는 더 눈에 띄게 간주됩니다. 최종 noticeability 점수는 이러한 점수를 집계하여 계산됩니다. HideNSeek의 방식은 간단한 규칙에 의존하는 기존 방법들과는 달리, 학습된 정보를 활용하여 보다 정교한 방식으로 공격 문제를 분석합니다.

- **Performance Highlights**: HideNSeek는 6개의 실제 그래프에서 실험을 통해 그 효과성을 입증했으며, LEO는 28개의 경우에서 11개의 경쟁자보다 우수한 성능을 보였습니다. 특히, HideNSeek는 감지 불가능 문제를 크게 개선하여 기존 방법보다 0.38배에서 5.75배 덜 bypassable(우회 가능함)하며, 저 공격률에서도 상당한 noticeability 점수를 제공합니다. 추가적으로 LEO는 GNN의 노드 분류 성능을 향상시키는 데에도 유용하다는 결과를 보여줍니다.



### UAV-VLA: Vision-Language-Action System for Large Scale Aerial Mission Generation (https://arxiv.org/abs/2501.05014)
Comments:
          HRI 2025

- **What's New**: UAV-VLA (Visual-Language-Action) 시스템은 공중 로봇과의 의사소통을 용이하게 하기 위해 설계된 도구입니다. 이 시스템은 위성 이미지를 처리하는 능력과 Visual Language Model (VLM) 및 GPT의 강력한 기능을 통합하여 사용자로 하여금 간단한 텍스트 요청을 통해 비행 경로 및 행동 계획을 생성할 수 있게 합니다. 따라서 UAV-VLA는 미션 계획 및 의사 결정 과정을 향상시킬 수 있는 풍부한 맥락 정보를 제공합니다.

- **Technical Details**: UAV-VLA 시스템은 사용자에게 영어와 같은 언어적 지시로 경로와 행동 집합을 생성하는 기능을 제공합니다. 이 시스템은 목표 추출 GPT 모듈, 객체 검색 VLM 모듈, 그리고 행동 생성 GPT 모듈의 세 가지 주요 모듈로 구성되어 있습니다. 이러한 구성은 주어진 임무에 맞추어 UAV가 자율적으로 정확한 미션 계획을 생성할 수 있도록 합니다.

- **Performance Highlights**: 고성능의 UAV-VLPA-nano-30 벤치마크를 통해 이 시스템은 22%의 경로 길이 차이와 K-최근접 이웃(KNN) 접근 방식을 이용하여 관심 객체를 찾는 평균 오차를 34.22m 로 나타내며 성능을 입증하였습니다. 실험을 통해 UAV-VLA 시스템은 인간 수준의 경로 및 행동 생성 성능을 보였으며 이는 인간 운영자와의 비교에서 동등한 결과로 나타났습니다.



### Quantum-enhanced causal discovery for a small number of samples (https://arxiv.org/abs/2501.05007)
Comments:
          19 pages, 8 figures

- **What's New**: 이번 연구에서는 비모델 구조 가정 없이 인과관계를 발견할 수 있는 새로운 양자 Peter-Clark (qPC) 알고리즘을 제안합니다. 이 알고리즘은 양자 회로에 의해 특징 지어진 재현 커널 힐베르트 공간에서의 조건독립성 테스트를 기반으로 하여, 임의 분포에서 관측된 데이터로부터 인과관계를 탐색할 수 있습니다. 또한, Kernel Target Alignment (KTA)를 기반으로 한 하이퍼파라미터 최적화 접근법을 통해 인과 발견의 신뢰성을 높이는 방법을 제시합니다.

- **Technical Details**: qPC 알고리즘은 조건 독립성 테스트를 통해 데이터를 사용하는 PC 알고리즘의 확장으로, 데이터가 양자 상태로 임베딩된 양자 커널을 활용하여 비선형성과 고차원의 데이터를 처리합니다. 이 과정은 커널 기반 조건 독립성 테스트(KCIT)를 통한 인과 관계를 추론하는데 필요한 조건 독립성 검사 기능을 포함합니다. qPC 알고리즘은 비모수적 방법으로, 두 번의 단계에서 독립성과 인과 관계를 분별해 내며, 결과적으로 CPDAGs를 출력하여 인과 관계를 캡처합니다.

- **Performance Highlights**: 실험 결과, qPC 알고리즘은 특히 작은 샘플 크기에서 기존의 고전적인 방법보다 뛰어난 성능을 보였습니다. 이 알고리즘은 매사추세츠주 보스턴 주택 가격 데이터셋을 활용한 실제 적용에서도 효과성을 입증하였으며, 이를 통해 qPC 알고리즘이 고전적 방법의 제한성을 대체하는 새로운 가능성을 보여줍니다. 또한, 제시된 KTA 기반의 최적화 방법은 적절한 커널 선택을 통해 인과 발견의 정확도를 높이고, 가짜 긍정률을 줄이는 효과를 가져왔습니다.



### GiNet: Integrating Sequential and Context-Aware Learning for Battery Capacity Prediction (https://arxiv.org/abs/2501.04997)
Comments:
          6 pages

- **What's New**: 이번 논문은 배터리 용량 예측을 위해 GiNet이라는 새로운 모델을 제안합니다. GiNet은 Gated Recurrent Units(구간 순환 신경망)와 최신 Transformer 아키텍처인 Informer를 결합하여 배터리의 시간적 및 맥락적 정보를 효과적으로 캡처할 수 있는 능력을 지닙니다. 이 모델은 배터리의 복잡한 동작을 반영하며, 역사적 측정을 통해 정확한 배터리 용량 예측을 목표로 합니다.

- **Technical Details**: GiNet의 구조는 GRU 모듈을 통한 시퀀스 특징 추출로 시작하며, 이 특징은 원래의 데이터와 융합됩니다. 이후 융합된 특징은 Informer의 임베딩으로 처리되어 Attention 기반의 인코더와 디코더로 분석됩니다. 이 과정에서 Temporal Dynamics(시간적 동적)와 Long-term Dependencies(장기 의존성)를capturing하기 위해 GRU와 Transformer의 장점을 최대한 활용합니다.

- **Performance Highlights**: GiNet은 예측한 배터리 용량의 Mean Absolute Error(MAE)를 0.11로 줄였으며, 기존 GRU 및 Informer, 최신 알고리즘에 비해 각각 76%와 27%의 성능 향상을 보고했습니다. 이러한 결과는 GiNet의 알고리즘 통합의 중요성을 강조하며, 다양한 산업 응용에 대한 적용 가능성을 제시합니다.



### IPDN: Image-enhanced Prompt Decoding Network for 3D Referring Expression Segmentation (https://arxiv.org/abs/2501.04995)
Comments:
          AAAI 2025

- **What's New**: 이 논문에서는 3D Referring Expression Segmentation (3D-RES)의 두 가지 주요 문제인 feature ambiguity와 intent ambiguity를 해결하기 위한 새로운 접근법인 Image enhanced Prompt Decoding Network (IPDN)을 제안합니다. IPDN은 다중 뷰 이미지(multi-view images)와 작업 중심(task-driven) 정보를 활용하여 모델의 추론 능력을 향상시키도록 설계되었습니다. 이를 통해 해당 문제를 효과적으로 해결하고 더 나은 성능을 달성할 수 있음을 보여줍니다.

- **Technical Details**: 제안된 IPDN은 Multi-view Semantic Embedding (MSE) 모듈과 Prompt-Aware Decoder (PAD) 모듈로 구성됩니다. MSE는 CLIP을 활용하여 2D 이미지 특징을 추출하고 이를 3D 포인트 클라우드 특성과 융합하여 시각적 표현을 향상시킵니다. PAD는 text와 시각적 특징 간의 상호작용으로부터 과제 중심의 신호(task-driven signals)를 유도하여 디코딩 프로세스를 안내합니다.

- **Performance Highlights**: 다양한 실험 결과, IPDN은 3D-RES 및 3D-GRES 작업에서 각각 1.9 및 4.2 mIoU 포인트 개선을 이루며 기존의 최첨단(state-of-the-art) 방법들을 초월했습니다. 이는 IPDN이 실제 작업에서 우수한 성능을 발휘함을 강하게 시사합니다. 이 연구는 3D-RES 분야의 발전에 중요한 기여를 하는 동시에 관련된 과제를 극복하기 위한 새로운 방향성을 제시합니다.



### CuRLA: Curriculum Learning Based Deep Reinforcement Learning for Autonomous Driving (https://arxiv.org/abs/2501.04982)
Comments:
          To be published in the 17th International Conference on Agents and Artificial Intelligence (ICAART), Feb 2025

- **What's New**: 이 연구는 딥 강화 학습(Deep Reinforcement Learning, DRL)으로 자율주행을 개선하는 새로운 방법을 제안합니다. 특히 Curriculum Learning 기법을 적용하여 환경의 난이도를 점진적으로 증가시키고 보상 함수에 충돌 패널티를 포함시켰습니다. 이 접근법은 복잡한 환경에서 에이전트의 적응성과 신뢰성을 향상시킵니다. 이 연구는 CARLA 시뮬레이터에서 안전한 주행을 학습하는 PAX와 변이 오토인코더(Variational Autoencoder, VAE)를 이용하였습니다.

- **Technical Details**: 이 연구에서는 Proximal Policy Optimization(PPO) 알고리즘과 Curriculum Learning을 결합하여 자율주행 자동차 학습을 수행합니다. CARLA 시뮬레이터에서 두 가지 Curriculum Learning을 통해 에이전트가 초기에는 간단한 작업을 수행하도록 하면서 점진적으로 난이도를 높입니다. 또한, 에이전트의 속도를 높이기 위한 보상 함수 개선이 포함되어 있으며, 이는 운전 경험의 품질을 향상시킵니다.

- **Performance Highlights**: 제안된 CuRLA(Curriculum Learning Based Reinforcement Learning for Autonomous Driving) 방법은 다양한 보상 신호를 단일 스칼라 보상 함수에서 조정하여 평균 속도를 높이는 데 기여합니다. 이러한 보상 체계에는 회전 각도, 중심 위치 및 속도 보상뿐만 아니라 충돌 패널티도 포함되어 있습니다. 요약적으로, 이 방법은 훈련 속도를 개선하고 복잡한 환경에서의 주행 안전성을 높이는 데 크게 기여하고 있습니다.



### SensorQA: A Question Answering Benchmark for Daily-Life Monitoring (https://arxiv.org/abs/2501.04974)
- **What's New**: 이 논문에서는 SensorQA라는 새로운 QA 데이터셋을 소개합니다. 이는 인간이 만든 최초의 QA 데이터셋으로, 장기적인 타임시리즈 센서 데이터를 기반으로 합니다. 데이터셋은 5.6K개의 다양한 질문과 정확한 답변 쌍을 포함하여 사용자의 실제 관심사를 반영합니다.

- **Technical Details**: SensorQA는 60명의 사용자로부터 3개월 동안 수집된 일상 활동 모니터링을 위한 센서 데이터에 초점을 맞춥니다. 질문 생성은 Amazon Mechanical Turk 플랫폼에서 수행된 활동 일정 그래프를 기반으로 하고, 14개의 다양한 활동 레이블의 서브셋을 사용하여 질문의 다양성을 촉진합니다. 이로 인해 하루에서 여러 주에 이르는 센서 시간 스케일을 포괄하는 5.6K개의 QA 쌍이 포함됩니다.

- **Performance Highlights**: SensorQA를 기반으로 한 최신 AI 모델의 벤치마크 평가 결과, 현재 모델과 최적의 QA 성능 및 효율성 사이에는 큰 격차가 존재했습니다. 이는 새로운 기여와 개선의 필요성을 강조합니다. 또한 SensorQA 데이터셋과 코드는 오픈소스 형태로 제공되어, 관련 연구자들이 이 분야에 기여할 수 있도록 장려하고 있습니다.



### Battling the Non-stationarity in Time Series Forecasting via Test-time Adaptation (https://arxiv.org/abs/2501.04970)
Comments:
          Accepted at AAAI 2025

- **What's New**: 본 논문에서는 비정상적인 시계열(Non-stationary time series) 데이터를 다룰 수 있도록 설계된 최초의 테스트 시간 적응 프레임워크인 TSF-TTA(TSF Test-Time Adaptation)를 소개합니다. 제안된 방법인 TAFAS(Teaching Adaptation for Forecasting In Non-stationary Time Series)는 소스 예측기를 적절히 조정하여 테스트 분포의 지속적인 변화를 반영합니다. 이를 통해, 기존 접근 방식들이 직면하는 한계를 극복하고, 핵심 의미를 보존하면서도 예측기의 신뢰성을 향상시키는 새로운 가능성을 열어주고 있습니다.

- **Technical Details**: TAFAS는 주기성 인식 적응 일정(Periodic-aware Adaptation Scheduling, PAAS) 및 게이트 캘리브레이션 모듈(Gated Calibration Module, GCM)로 구성됩니다. PAAS는 의미 있는 주기 패턴을 반영할 수 있도록 부분적으로 관측된 진실 값을 적응적으로 획득하고, GCM은 예측기와 일치하도록 테스트 시간 입력을 보정합니다. 이러한 모듈들은 비정상적인 테스트 시간 입력에서 소스 예측기가 능동적으로 적응할 수 있도록 돕습니다.

- **Performance Highlights**: TAFAS는 다양한 TSF 아키텍처에 걸쳐 일관되게 성능 향상을 보여주며, 특히 분포 변화가 더 뚜렷한 장기 예측 시나리오에서 두드러진 성과를 발휘합니다. 실제 실험 결과에 따르면, TAFAS는 이전에 보지 못한 테스트 데이터 스트림에서 Chronos의 예측 오류를 최대 45% 개선할 수 있음을 보여줍니다. 이러한 성과는 TAFAS가 기존의 비정상성을 다루기 위한 방법들과의 통합을 통해 더욱 강화된다는 점에서 주목할 만합니다.



### Demystifying Domain-adaptive Post-training for Financial LLMs (https://arxiv.org/abs/2501.04961)
- **What's New**: 본 논문에서는 FINDAP라는 새로운 프레임워크를 제안하며, 이는 금융 도메인에 특화된 대규모 언어 모델(LLM)의 도메인 적응형 후 훈련(domain-adaptive post-training)을 체계적으로 조사합니다. 본 연구는 핵심 역량을 정의하고 이를 바탕으로 평가 프레임워크를 설계함으로써, 최적의 훈련 방법을 제안하는 것을 목표로 합니다. Llama-Fin이라는 혁신적인 모델을 통해 금융 업무 전반에서 최첨단 성능을 달성합니다.

- **Technical Details**: FINDAP의 핵심은 도메인 특화 개념, 도메인 특정 작업, 추론 능력 및 지시 준수(instruction-following) 능력 등을 갖춘 LLM의 필요한 역량을 식별하는 것입니다. 이러한 역량 기반으로 평가 프레임워크를 개발하여 성과 목표를 명확히 하고, 여러 작업에서 모델의 개선을 이끌어내는 것이 주된 내용입니다. 따라서, 훈련 방법론은 연속 사전 훈련(Continual Pretraining, CPT) 및 지시 조정(Instruction Tuning, IT) 단계로 나뉘며, 이후 선호 데이터(preference data)를 사용한 새로운 훈련 레시피가 적용됩니다.

- **Performance Highlights**: Llama-Fin 모델은 70B 스케일의 대형 모델들 및 상용 모델인 GPT-4o를 포함한 모든 기준 모델들보다 우수한 성과를 보여주었습니다. 특히, 본 연구에서는 훈련 데이터와 유사한 새로운 작업에서는 물론, 훈련 중 경험하지 못한 새로운 작업에서도 경쟁력을 유지하며 높은 성능을 기록했습니다. 이러한 결과는 도메인 적응형 후 훈련의 중요성과 효과적인 훈련 레시피의 필요성을 강조합니다.



### Addressing Domain Shift via Imbalance-Aware Domain Adaptation in Embryo Development Assessmen (https://arxiv.org/abs/2501.04958)
Comments:
          15 pages

- **What's New**: 이 논문에서는 의료 이미지 분석에서의 도메인 시프트(domain shift)와 클래스 불균형(class imbalance) 문제를 동시에 해결하기 위한 새로운 프레임워크인 Imbalance-Aware Domain Adaptation (IADA)를 제안합니다. 이 프레임워크는 (1) 클래스별 주의 메커니즘을 활용한 적응형 특징 학습, (2) 동적 가중치를 통한 균형 잡힌 도메인 정렬, (3) 적응형 임계값 최적화의 세 가지 주요 요소로 구성되어 있습니다. 실험 결과 IADA는 기존 방법들에 비해 최대 25.19% 높은 정확도를 달성하며, 다양한 클래스에서의 성능 균형을 유지하는 뛰어난 효과를 보여줍니다.

- **Technical Details**: IADA는 복잡한 모델 성능 저하를 방지하기 위한 세 가지 혁신적인 접근을 통합하고 있습니다. 이론적 분석을 통해 수렴 보장(convergence guarantees)과 복잡성 경계(complexity bounds)를 설정하였으며, 여러 영상 기법에서의 배아 발달 평가 실험을 통해 IADA의 효과성을 입증하였습니다. 특히, 저품질 영상 시스템에서의 강력한 일반화 능력을 보여주며, AUC(Area Under the Curve)에서 최대 12.56% 향상을 나타냅니다.

- **Performance Highlights**: IADA는 의료 이미지 시스템의 안정성과 공정성을 개발하기 위한 잠재력을 지니고 있습니다. 연구 결과에 따르면 기존 기술에 비해 성능이 향상되어, 특히 다른 임상 환경에서의 적응력과 정확도 증대를 가져왔습니다. 따라서, IADA는 다양한 환자 집단과 임상 환경에서 일반화 가능한 알고리즘을 구축하는 데 기여할 수 있는 가능성을 보여줍니다.



### Step-by-Step Mastery: Enhancing Soft Constraint Following Ability of Large Language Models (https://arxiv.org/abs/2501.04945)
- **What's New**: 이번 연구에서는 LLM(대규모 언어 모델)이 소프트 제약 조건을 따르는 능력을 개선하기 위한 효율적인 데이터 생성 파이프라인을 설계했습니다. 기존의 연구들은 제약 조건을 따르는 LLM의 능력을 평가하는 데 중점을 두었으나, 본 연구는 이를 향상하는 방향으로 진행되었습니다. 또한, 커리큘럼 학습(curriculum learning) 기반의 새로운 훈련 패러다임을 도입하여 LLM이 복잡한 제약 조건을 효과적으로 따를 수 있도록 합니다.

- **Technical Details**: 제안된 방법론은 두 가지 주요 단계로 구성됩니다: 점진적인 제약 조건 추가와 Judger 재정렬입니다. 이 과정에서 각 제약 조건을 하나씩 추가하여 LLM이 각 제약 조건을 점진적으로 따를 수 있도록 훈련합니다. Judger는 모델의 출력을 제약 조건 준수 정도에 따라 재정렬하여 최종적인 고품질 출력을 생성합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 LLM의 소프트 제약 조건 따르기 능력을 크게 향상시키며, 훈련 데이터의 품질이 모델의 출력 품질에 미치는 영향이 크다는 것을 보여주었습니다. GPT-4 모델을 포함한 다양한 LLM에서 실행된 Benchmark 평가에서 이 새로운 접근법의 효과가 증명되었습니다. 다양한 조건을 조합한 프레임워크를 통해, 소프트 제약 조건을 처리하는 데 있어 LLM의 전반적인 성능이 개선되었습니다.



### Jailbreaking Multimodal Large Language Models via Shuffle Inconsistency (https://arxiv.org/abs/2501.04931)
- **What's New**: 이번 연구에서는 다중 모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 안전 메커니즘에서 발생할 수 있는 취약점을 탐지하는 새로운 공격 방법인 SI-Attack을 제안합니다. 기존 연구와 달리, MLLMs의 이해 능력과 안전 능력 사이의 'Shuffle Inconsistency'를 발견하였으며, 이는 모델이 혼합된 해로운 지시를 잘 이해하지만, 이러한 지시를 통한 공격에는 쉽게 노출된다는 점을 강조합니다.

- **Technical Details**: SI-Attack은 해로운 지시를 정교하게 선택하기 위해 쿼리 기반 블랙박스 최적화 방법을 활용하며, 독성 판별 모델의 피드백에 기초하여 가장 해로운 혼합 입력을 선별합니다. 이를 통해 SI-Attack은 상업적으로 사용되는 태스크에서도 공격 성공률을 크게 향상시킬 수 있습니다.

- **Performance Highlights**: 일련의 실험 결과, SI-Attack은 GPT-4o 및 Claude-3.5-Sonnet과 같은 상업적 MLLMs에서 공격 성공률을 유의미하게 증가시켰습니다. 이 연구는 해로운 지시가 제대로 대응되지 않는 MLLMs의 취약성을 체계적으로 입증하며, 향후 안전한 AI 시스템 구축에 중요한 기초 자료로 활용될 것입니다.



### Image2CADSeq: Computer-Aided Design Sequence and Knowledge Inference from Product Images (https://arxiv.org/abs/2501.04928)
Comments:
          20 pages, 10 figures, and 6 tables

- **What's New**: 이 연구는 2D 이미지로부터 CAD 시퀀스를 직접 생성하는 새로운 데이터 기반 접근 방식을 제안합니다. 특히, Image2CADSeq 신경망 모델을 통해 이미지 입력을 기반으로 CAD 모델을 역설계할 수 있도록 하여 설계 과정의 이해를 심화합니다. 특정 데이터 합성 파이프라인을 사용하여 CAD 모델링에 대한 새로운 인사이트와 작동 가능성을 제공합니다.

- **Technical Details**: 제안된 접근 방식은 Target-Embedding Variational Autoencoder (TEVAE) 아키텍처를 사용하여 단일 이미지 입력으로 CAD 작업 시퀀스를 예측합니다. 이 모델은 3D 모델로 변환할 수 있는 CAD 시퀀스를 생성하고, CAD 모델의 구체적 과정과 변경 가능성을 분리하여 제공합니다. 평가 프레임워크는 CAD 시퀀스, 3D 모델 및 해당 이미지를 포함하여 다단계로 모델의 예측 성능을 분석합니다.

- **Performance Highlights**: 실험 결과, Image2CADSeq 모델은 2D 이미지 데이터에서 CAD 시퀀스를 생성하는 데 있어 매우 유망한 성능을 보였습니다. 이 접근 방식은 CAD 모델 재구성의 접근성을 향상시켜 경험이 적은 설계자도 설계 과정에 적극적으로 참여할 수 있도록 할 잠재력을 가지고 있습니다. 연구진은 이 모델이 CAD 시스템 혁신에 기여할 것으로 기대하고 있으며, 이는 최종 사용자 참여를 촉진하는 독특한 경로를 제공할 것입니다.



### FLowHigh: Towards Efficient and High-Quality Audio Super-Resolution with Single-Step Flow Matching (https://arxiv.org/abs/2501.04926)
Comments:
          Accepted by ICASSP 2025

- **What's New**: FLowHigh는 오디오 초해상도(Super-Resolution) 분야에 새로운 접근법을 제시합니다. 이 연구는 전통적인 확산 모델(diffusion models)의 단점을 보완하기 위해, 흐름 매칭(flow matching) 방법을 통합하여 고해상도 오디오 신호를 단일 스텝 샘플링으로 생성합니다. 또한, 고해상도 오디오 분포를 효과적으로 포착하기 위해 오디오 초해상도에 맞춤형 확률 경로(probability paths)를 탐색합니다.

- **Technical Details**: FLowHigh는 저해상도(Low-Resolution) 오디오 신호를 고해상도(High-Resolution) 신호로 변환하는 방법론을 제안합니다. 이 방법은 간단한 벡터 필드 회귀(vector field regression)를 통해 사전 분포(prior distribution)와 복잡한 데이터 분포(data distribution) 간의 변환을 학습합니다. 또한, mel-spectrogram 수준에서 벡터 필드를 회귀하기 위한 트랜스포머 기반(vector field estimator)을 활용하며, 예측된 파형을 합성하기 위해 사전 학습된 신경 보코더(pre-trained neural vocoder)를 사용합니다.

- **Performance Highlights**: VCTK 벤치마크 데이터셋에서 실험한 결과, FLowHigh는 기존 모델들보다 월등한 성능을 보였으며, 로그 스펙트럴 거리(log-spectral distance)와 ViSQOL 평가 지표에서 최첨단 성능을 달성했습니다. 단일 스텝 샘플링(single-step sampling) 과정으로 높은 충실도의 고해상도 오디오를 생성하면서 계산 효율성도 유지하였습니다. FLowHigh는 오디오 초해상도 연구에 있어 흐름 매칭 기법을 성공적으로 통합한 첫 번째 시도로 평가받고 있습니다.



### SUGAR: Leveraging Contextual Confidence for Smarter Retrieva (https://arxiv.org/abs/2501.04899)
Comments:
          ICASSP2025

- **What's New**: 본 논문에서는 Semantic Uncertainty Guided Adaptive Retrieval (SUGAR)라는 새로운 접근법을 제안합니다. 이는 LLM의 파라메트릭 지식만으로는 정확한 응답을 생성하는 데 한계가 있어 외부 지식의 적절한 활용을 필요로 하는 문제에서 출발했습니다. SUGAR는 컨텍스트 기반 엔트로피를 활용하여 모델의 불확실성을 평가하고, 이 기반으로 검색 여부와 검색 방식을 결정합니다.

- **Technical Details**: SUGAR는 언어 모델의 응답 생성에 대한 불확실성을 측정하기 위해 의미적 엔트로피를 사용합니다. 높은 엔트로피를 보이는 경우에는 외부 지식 D를 함께 활용하여 응답을 생성하게 됩니다. 이렇게 함으로써 주어진 질문에 대한 더 적절한 지원 컨텍스트를 제공하고, 단일 단계 또는 다단계 검색을 동적으로 결정합니다.

- **Performance Highlights**: 실험 결과, SUGAR를 사용한 선택적 검색이 다양한 질문 응답 과제에서 성과를 향상시키는 것으로 나타났습니다. SUGAR는 정보 검색에서 효과적인 자원의 활용을 지원하며, 불확실성 평가를 통해 필요할 때만 검색을 수행하여 효율성을 최적화합니다. 또한, SUGAR는 추가 학습이나 미세 조정 없이도 안정적인 성과를 발휘할 수 있습니다.



### Quantifying Itch and its Impact on Sleep Using Machine Learning and Radio Signals (https://arxiv.org/abs/2501.04896)
- **What's New**: 본 논문에서는 인공지능(AI)과 홈 라디오 장치를 결합하여 만성 가려움증을 더 객관적으로 모니터링하는 방법을 소개합니다. 이 기술은 가려움증과 수면 품질에 미치는 영향을 동시에 평가할 수 있는 장점이 있습니다. 특히, 이 장치는 환자가 착용할 필요가 없으며, 장기간의 모니터링을 가능하게 하여 환자의 부담을 줄입니다.

- **Technical Details**: 기존의 주관적인 가려움증 평가 방법의 한계를 극복하기 위하여, 저자들은 RF(무선 주파수) 신호를 사용하여 수면 및 가려움 증상을 모니터링하는 새로운 접근법을 개발했습니다. 이 장치는 유선 신호를 송신하고, 환자의 환경에서 반사된 신호를 수집하여 가려움증과 수면의 상관관계를 분석합니다. 연구 결과, 이 방법은 높은 정확도로 가려움증을 감지하고 수면의 질을 평가할 수 있음을 입증했습니다.

- **Performance Highlights**: 임상 연구에서 20명의 만성 가려움증 환자를 대상으로 1개월 간 가정에서 모니터링한 결과, 이 장치는 ROC AUC = 0.997이라는 높은 정확도를 기록했습니다. 또한, 가려움증과 수면의 질 간의 상관관계를 보여줌으로써, 수면 효율성 감소(R = 0.6, p < 0.001)와 더불어 수면 잠복기의 증가(R = 0.68, p < 0.001)를 확인했습니다. 이 연구는 만성 가려움증 환자의 치료 반응을 평가할 수 있는 유용한 도구가 될 수 있음을 시사합니다.



### Reach Measurement, Optimization and Frequency Capping In Targeted Online Advertising Under k-Anonymity (https://arxiv.org/abs/2501.04882)
- **What's New**: 최근 온라인 광고의 사용 증가와 브랜드 인지도 향상은 소셜 미디어의 보편성과 관련이 깊습니다. 이 논문은 사용자 프라이버시를 우선시하는 광고 솔루션으로 전환되는 과정을 다루고 있으며, $k$-anonymity 모델을 통해 도달 측정(reach measurement) 및 최적화(optimalization)에 대해 논의합니다. 전통적인 frequency capping이 사용자 프라이버시 요구 사항을 충족하지 못할 경우의 해결책을 제시합니다.

- **Technical Details**: 논문에서는 새로운 프라이버시 환경에서 도달(reach)을 어떻게 보고(report)할 수 있는지를 설명하고, 전통적인 frequency capping의 확률적 적응을 통한 probabilistic discounting 방식을 소개합니다. 각 사용자 그룹은 k𝑘k-anonymity를 기반으로 형성되어, 개별 사용자를 k-1명의 다른 사용자와 구별할 수 없게 만듭니다. 이러한 그룹에 대한 광고 시스템은 사용자 익명성을 유지하면서 광고를 최적화하는 것을 목표로 합니다.

- **Performance Highlights**: 실험 결과, 프라이버시를 도입함에 따라 성과(performance)가 유의미하게 감소하는 경향이 발견되었습니다. 그러나 이러한 개선은 광고 플랫폼이 사용자에게 더 많은 프라이버시를 제공하기 위해 필요한 비용이 제한적이라는 점에서 이점이 있습니다. 이는 광고의 개인화(personalization)와 사용자 프라이버시를 동시에 고려해야 하는 온라인 광고의 지속 가능한 발전을 위한 기여로 해석됩니다.



### Real-Time Textless Dialogue Generation (https://arxiv.org/abs/2501.04877)
- **What's New**: 최근 대형 언어 모델(LLMs)의 발전은 텍스트 기반 대화 시스템에서 중요한 진전을 이루었습니다. 그러나 음성 대화 시스템은 여전히 자연스러움에서 뒤처지고 있으며, 이는 전통적인 캐스케이드 설계에 의존하기 때문입니다. 본 논문은 이러한 문제를 해결하기 위한 실시간 텍스트 없는 음성 대화 생성 모델(RTTL-DG)을 제안합니다.

- **Technical Details**: RTTL-DG는 전통적인 음성 인식과 응답 생성을 통합하여 직접적인 음성 처리를 통해 대화를 생성합니다. 이 모델은 대화 관리자(Dialogue Manager)와 응답 생성기(Response Generator)로 구성되며, 연속적인 음성 입력에서 정기적으로 다음 행동을 결정합니다. 생성된 응답은 기계적이지 않고 자연스러운 표현을 포함하여 빠른 대응과 원활한 턴 테이킹을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, RTTL-DG 모델은 자연스러움, 반응 속도 및 유동성에서 우수한 성능을 나타내며, 사용자와의 상호작용에서 보다 자연스러운 대화 경험을 제공합니다. 전통적인 캐스케이드 모델에 비해 의미적 일관성에서는 약간의 성능 저하가 있었지만, 사용자 친화적이고 매력적인 대화 시스템 개발에 중요합니다.



### Back Home: A Machine Learning Approach to Seashell Classification and Ecosystem Restoration (https://arxiv.org/abs/2501.04873)
- **What's New**: 이번 연구에서는 코스타리카의 해양 생태계를 보호하기 위해, 압수된 조개껍데기의 기원을 식별하는 데 특화된 합성곱 신경망(CNN) 모델을 개발했습니다. 총 19,000장의 이미지를 기반으로 한 데이터 세트를 구축하고, 85% 이상의 분류 정확도를 달성했습니다. 이 모델은 사용자 친화적인 애플리케이션으로 통합되어 현재 36,000개 이상의 조개껍데기를 분류하며, 이미지당 3초 이내의 실시간 결과를 제공합니다.

- **Technical Details**: 본 연구에서 사용된 데이터 세트는 10개월에 걸쳐 수집된 19,058장의 이미지로 구성되었으며, 카리브해와 태평양의 516종 조개껍데사가 포함되어 있습니다. ConvNext 아키텍처를 사용하여 조개껍데기의 기원을 분류할 수 있는 모델을 구축하였으며, 데이터 증강 기법을 적용하여 다양한 조개껍데기의 변화를 다루기 위한 성능을 향상했습니다. 모델 학습 시 하이퍼파라미터 조정을 위해 SGD 옵티마이저와 학습률 감쇠 기법을 사용했습니다.

- **Performance Highlights**: 최종 모델은 ConvNext 아키텍처를 통해 우수한 성능을 발휘했으며, 자원 제한 환경에서도 효과적으로 배포될 수 있도록 설계되었습니다. CNN 모델의 엄청난 성능 덕분에 조개껍데기의 생태계 기원을 정확히 식별할 수 있게 되었고, 실시간 분류 기능을 통해 사용자의 신뢰성을 높였습니다. 분류 결과에 대한 검증을 위해 70%의 데이터를 훈련 세트로, 15%는 검증 세트, 나머지 15%는 테스트 세트로 사용하여 성능을 평가했습니다.



### Exploring Large Language Models for Semantic Analysis and Categorization of Android Malwar (https://arxiv.org/abs/2501.04848)
- **What's New**: 이번 논문에서는 복잡한 악성 소프트웨어 분석을 위한 새로운 접근법으로 Large Language Models (LLMs)를 활용하는 방법을 탐구합니다. 특히 \/msp 모델은 Android 플랫폼의 악성 코드 분석을 지원하며, 기존 수작업 분석을 빠르고 효율적으로 개선할 수 있는 구조를 갖추고 있습니다. 이 모델은 악성 소프트웨어의 카테고리 분류를 통해 시간을 절약할 수 있도록 설계되었습니다.

- **Technical Details**: 논문에서 제안하는 \/msp는 GPT-4o-mini 모델을 기반으로 구축되었으며, 계층적 요약 체인을 사용하여 악성 소프트웨어 분석을 보완합니다. 이는 소프트웨어의 기능 수준, 클래스, 패키지 수준에서의 고품질 요약을 제공하며, 효과적인 프롬프트 엔지니어링을 통해 악성 행위를 초래하는 코드 조각을 정확히 파악할 수 있도록 합니다.

- **Performance Highlights**: \msp는 Android 악성 코드 분석에 대해 최적화된 프롬프트 엔지니어링을 통해 최대 77%의 분류 정확도를 달성하는 성과를 보여줍니다. 이는 악성 소프트웨어의 기능적 측면뿐만 아니라 클래스 및 패키지 수준에서도 매우 견고한 요약을 제공합니다. 이러한 접근 방식 덕분에 연구자들은 더 빠르게 악성 코드를 역분석할 수 있게 됩니다.



### Enhancing Listened Speech Decoding from EEG via Parallel Phoneme Sequence Prediction (https://arxiv.org/abs/2501.04844)
Comments:
          ICASSP 2025

- **What's New**: 이번 논문에서는 뇌-컴퓨터 인터페이스(Brain-Computer Interface, BCI)에서 전기 생리학적 신호를 통해 들은 음성을 동시에 음성 파형(speech waveform)과 텍스트 음소 시퀀스(textual phoneme sequences)로 디코딩할 수 있는 새로운 접근 방식을 제안합니다. 기존의 방법들은 단일 모달리티에서만 작업이 가능했으나, 이 연구는 EEG 신호를 활용하여 두 가지 모달리티를 동시에 처리함으로써 기존 시스템의 한계를 극복합니다. 또한, 이 방법은 보조 음소 예측기(auxiliary phoneme predictor)를 통합하여 성능을 향상시키고 있어, 새로운 연산적 가능성을 제시합니다.

- **Technical Details**: 제안된 프레임워크는 EEG 모듈, 음성 모듈 및 음소 예측기의 세 가지 주요 구성 요소로 이루어져 있습니다. EEG 모듈은 EEG 신호로부터 임베딩(embedding)을 학습하며, 생성된 임베딩은 음소 예측기와 음성 모듈에 병렬로 전달됩니다. 특히, 음소 예측기는 EEG 임베딩을 기반으로 음소 시퀀스를 디코딩하며, 음성 모듈은 음성 파형 생성을 담당합니다. 이 프레임워크는 각 모달리티에 대해 순차적 파이프라인을 요구하지 않고 동시에 디코딩 기능을 제공합니다.

- **Performance Highlights**: 제안된 모델은 이전의 모든 방법들과 비교해 유의미한 성능 향상을 보였으며, 특히 음성 파형 및 음소 시퀀스를 모두 디코딩하는 데 있어 효과적입니다. 논문에서는 음소 시퀀스 및 음성 파형의 디코딩 성능을 보여주는 표를 포함하고 있으며, 제안된 접근 방식의 우수성을 실험적으로 입증하고 있습니다. 이 논문에 포함된 소스 코드와 음성 샘플은 공개적으로 제공되어 있어, 연구자들이 결과를 재현하고 더 발전시키는 데 도움을 줄 것입니다.



### Do Code LLMs Understand Design Patterns? (https://arxiv.org/abs/2501.04835)
Comments:
          accpeted by llm4code workshop in ICSE 2025

- **What's New**: Code LLMs(대형 언어 모델)은 다양한 소프트웨어 개발 작업에서의 적용 가능성을 보여주지만, 기존의 디자인 패턴을 잘 이해하지 못하는 경향이 있습니다. 이 연구에서는 Code LLMs의 디자인 패턴 인식 및 생성 능력을 평가하고, 이러한 모델이 생성하는 코드가 프로젝트의 기준에 어떻게 영향을 미치는지를 분석합니다. 특히, 디자인 패턴의 중요성을 강조하며, Bias 분석(tendency analysis)의 필요성을 제기합니다.

- **Technical Details**: 우리는 Python과 Java를 사용하여 총 12가지 디자인 패턴에 대하여 48개의 고품질 레포지토리를 수작업으로 선택하였으며, 세 가지 실험(디자인 패턴 분류, 코드 라인 완성, 함수 생성)을 수행하여 Code LLMs의 이해도와 생성 능력을 평가합니다. 각 실험에서는 모델의 정확성, 코드 유사성(code similarity), 코드 편집 유사성(code edit similarity) 등을 측정하였으며, GPT-4과 Llama-31-70B가 각각 가장 높은 정확도를 기록했습니다.

- **Performance Highlights**: 실험 결과, Java 데이터셋에서 Llama-31-70B가 55.56%의 정확도로 가장 높은 성능을 나타냈고, Python 데이터셋에서 GPT-4가 29.73%의 전체 정확도를 기록했습니다. 모든 모델에서 Java에서 Python으로, 그리고 쉬운 문제에서 어려운 문제로 갈수록 성능이 저하되는 경향이 관찰되었습니다. 이러한 결과는 Code LLMs가 디자인 패턴을 이해하고 따르는 데에 상당한 제약이 있음을 시사합니다.



### Intelligent Gradient Boosting Algorithms for Estimating Strength of Modified Subgrade So (https://arxiv.org/abs/2501.04826)
Comments:
          17 pages

- **What's New**: 본 연구에서는 포장재의 강도를 결정짓는 주요 요소인 지반(subgrade)의 특성을 추정하기 위해 기계 학습(machine learning) 기반의 두 가지 부스팅 기법인 CatBoost 및 XGBoost, 그리고 Support Vector Regression (SVR)이 적용되었습니다. 특히, 이 연구는 수화석회로 개선된 쌀 껍질 재 (HARSH)와 함께 사용되는 지반 토양의 특성을 평가하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 연구는 121개의 실험 데이터를 바탕으로 CBR, UCS 및 R의 추정에 필요한 플라스틱 한계(plastic limit), 액체 한계(liquid limit), 플라스틱성 지수(plasticity index), 점토 활동(clay activity) 등의 특성을 입력값으로 사용하였습니다. 네 가지 평가 지표, 즉 결정계수(R2), 평균 제곱근 오차(RMSE), 평균 절대 오차(MAE) 및 평균 절대 백분율 오차(MAPE)를 활용하여 모델의 성능이 평가되었습니다.

- **Performance Highlights**: 결과에 따르면, XGBoost는 CBR, UCS 및 R 추정에서 각각 R2 값이 0.9994, 0.9995 및 0.9999로 CatBoost 및 SVR 보다 우수한 성능을 보였습니다. 또한, SVR은 CBR과 R의 추정에서 CatBoost보다 우수한 성능을 보여준 반면, CatBoost는 UCS 추정에서 SVR보다 더 좋은 결과를 나타냈습니다. 마지막으로, 특성 민감도 분석 결과, HARSH 비율이 증가함에 따라 추정된 특성 값이 증가하는 경향이 있음을 확인했습니다.



### Planing It by Ear: Convolutional Neural Networks for Acoustic Anomaly Detection in Industrial Wood Planers (https://arxiv.org/abs/2501.04819)
- **What's New**: 최근 목재 제품 산업은 숙련된 노동력의 부족에 직면하고 있습니다. 이로 인해 기계 고장 시 발생하는 추가 비용이 증가하고 있으며, 이러한 문제를 해결하기 위해 음향 모니터링을 통한 지원이 제안됩니다. 본 논문에서는 딥 컨볼루션 오토인코더(deep convolutional autoencoder)를 활용한 음향 이상 탐지 기법을 소개하며, 실제 공장 데이터를 사용하여 성능을 평가합니다.

- **Technical Details**: 본 연구는 새로운 산업 데이터 세트를 기반으로 콘볼루션 오토인코더(convolutional autoencoder) 두 가지 아키텍처인 스킵-CAE(Skip-CAE)와 스킵-CAE 변환기(Skip-CAE Transformer)를 제안합니다. 이 모델들은 DCASE 오토인코더 기준선, 원-클래스 SVM(one-class SVM), 아이솔레이션 포레스트(isolation forest)와 같은 기존 방법들을 초월하여 성능을 보여주었습니다.

- **Performance Highlights**: 고유한 실험 데이터세트를 통해 스킵-CAE 모델이 0.846, 스킵-CAE 변환기가 0.875의 ROC 곡선 아래 영역(Area under the ROC curve, AUC)을 달성하여 이전의 연구 성과를 초과하는 결과를 나타냈습니다. 이는 스킵 연결(skip connections)과 트랜스포머 아키텍처의 결합이 이상 탐지 성능을 향상시키는 데 기여한 것으로 볼 수 있습니다.



### Decentralised Resource Sharing in TinyML: Wireless Bilayer Gossip Parallel SGD for Collaborative Learning (https://arxiv.org/abs/2501.04817)
- **What's New**: 계속해서 발전하는 마이크로컨트롤러 유닛(MCU)의 컴퓨팅 능력 덕분에 엣지 디바이스에서도 머신러닝 모델을 지원할 수 있게 되었습니다. 그러나 분산된 연합 학습(Decentralised Federated Learning, DFL)을 그러한 장치에 배포하는 데는 지속적인 연결성 부족, 제한된 통신 범위 등 여러 가지 주요 도전이 있습니다. 본 논문에서는 자원이 제한된 환경에서 이러한 문제를 해결하기 위해 새로운 프레임워크인 이층 Gossip 분산 병렬 확률적 경량 하강법(Bilayer Gossip Decentralised Parallel Stochastic Gradient Descent, GD-PSGD)을 제안합니다.

- **Technical Details**: 이 프레임워크는 지리적 그룹화를 위한 Distributed K-means 클러스터링과 두 단계로 모델 집계를 위한 gossip 프로토콜을 결합한 계층 구조의 통신 체계를 통합하고 있습니다. 각 클러스터 내 및 클러스터 간의 효과적인 통신을 위해 DK-means를 활용하여 자원 제약이 있는 환경에서도 통신 오버헤드를 줄이고 확장성을 개선할 수 있도록 설계되었습니다.

- **Performance Highlights**: 테스트 결과, 제안된 방법은 IID 데이터셋에서 Centralised Federated Learning(CFL)에 비슷한 정확도를 달성하며, 수렴을 위해 추가로 1.8 라운드만 필요하다는 것을 보여주었습니다. Non-IID 데이터셋에 대한 정확도 감소는 중간 수준의 데이터 불균형에서도 8% 미만으로 유지되며, 이는 최소한의 성능 손실로 확장 가능하고 개인정보 보호가 가능한 학습을 지원할 수 있다는 점을 강조합니다.



### TREAD: Token Routing for Efficient Architecture-agnostic Diffusion Training (https://arxiv.org/abs/2501.04765)
- **What's New**: 이번 연구는 기존 diffusion model에서 발생하는 샘플 비효율성과 높은 훈련 비용을 해결하기 위해, predefined routes를 통해 토큰 정보를 저장하고 모델의 깊은 층으로 다시 도입하는 방법을 제안합니다. 이러한 접근 방식은 토큰을 완전히 버리는 대신, 더 깊은 층으로 연결할 수 있도록 하여 훈련 효율성을 높입니다. 또한, 여러 경로를 결합하고, 이를 고려한 보조 손실(auxiliary loss)을 도입하여 모델의 성능을 개선하는 방향으로 진행하였습니다.

- **Technical Details**: 기존의 Transformer 아키텍처는 입력 길이에 대해 제곱 복잡성을 가지기 때문에 컴퓨팅 비용이 높아지는 문제가 있었습니다. 이번 연구에서는 routing 기법을 통해 정보 흐름을 제한하고, 복잡성을 줄이면서 효율성을 높이는 방법을 모색했습니다. 이때 routing은 학습 동안에만 작동하며, 필요한 계산을 생략하지 않고 정보를 한 층에서 다른 층으로 전달합니다.

- **Performance Highlights**: 우리의 방법은 ImageNet-1K 256 x 256 기준에서, DiT에 비해 9.55배의 수렴 속도 향상을 이룩했습니다. 또한, DiT-XL/2의 기준 FID 9.62보다 더 나은 FID 9.32를 41시간 내에 달성했습니다. 이러한 성과는 diffusion models의 훈련을 민주화하는 데 기여할 수 있을 것입니다.



### Discovering new robust local search algorithms with neuro-evolution (https://arxiv.org/abs/2501.04747)
- **What's New**: 이 논문은 로컬 서치(local search) 알고리즘의 기존 문제들을 극복하기 위한 새로운 접근법을 제시합니다. 기존 로컬 서치 알고리즘의 의사결정 과정을 개선하여 각 반복에서 이웃에서 최적의 전환을 이루는 것을 목표로 합니다. 이를 위해 저자들은 전통적인 알고리즘과 동일한 입력 정보를 갖춘 신경망(neural network)을 사용하는 방안을 제안합니다.

- **Technical Details**: 논문에서는 문제 목표 함수의 단조 변환에 견딜 수 있는 효율적이며 강력한 알고리즘을 구축하기 위해 다양한 정보 표현 방식을 탐구합니다. 실험 세트업은 NK landscape 문제를 중심으로 구축되며, 문제의 크기와 경량성을 조절할 수 있는 유연성을 제공합니다. 이를 통해 저자들은 블랙 박스 문제에 대한 새로운 로컬 서치 알고리즘 개발 및 성능 향상을 꾀하고자 합니다.

- **Performance Highlights**: 제안된 접근법은 간단한 서치 컴포넌트를 유지하며 정보 활용 방식을 변화시키는 데 집중합니다. 저자들은 다양한 형태의 정보 표현을 연구하고, 효과성을 평가하기 위해 QUBO(Quadratic Unconstrained Binary Optimization) 문제의 새로운 인스턴스에서 전략의 견고성과 일반화 가능성을 테스트합니다. 이 연구는 로컬 서치 알고리즘의 진화를 위한 유망한 경로를 제시합니다.



### Generative Style Transfer for MRI Image Segmentation: A Case of Glioma Segmentation in Sub-Saharan Africa (https://arxiv.org/abs/2501.04734)
- **What's New**: 이 연구는 아프리카 아프리카 사하라 이남 지역에서의 저품질 자기공명영상(MRI) 기술을 위한 심층학습 기반 뇌 종양 분할 방법을 제시합니다. 연구에 따르면, 해당 지역의 교육 데이터를 바탕으로 도메인 시프트가 모델 성능에 미치는 영향은 크지 않으며, 3D와 2D 모델을 비교한 결과 비슷한 성능을 보이는 것으로 나타났습니다. 또한, d neural style transfer (NST) 기법을 데이터 증가 방법으로 사용하여 성능 격차를 줄이는 두 가지 접근법이 제안되었습니다.

- **Technical Details**: 연구에 사용된 모델은 PyTorch로 구현된 nnU-Net의 확장판이며, 다양한 의료 데이터에 적용 가능합니다. 데이터 전처리 과정에서 기본적으로 얼굴 부분을 잘라내고 강도 조정 및 리샘플링을 통해 다양한 데이터 품질의 일관성을 높였습니다. 최적화된 U-Net 모델을 기반으로 하여, 여러 데이터 세트에 맞춰 자동적으로 구성되는 nnU-Net을 활용하였습니다.

- **Performance Highlights**: 최종적으로, GLI 및 GLI+SSA 데이터 세트를 사용하여 모델을 훈련시켰으며, 300 에포크 동안 성능 평가를 수행한 결과, 모델은 높은 정확도로 뇌 종양 영역을 분할하였습니다. 이 연구는 SSA 지역의 의료 시스템 내에서 뇌 종양 예측 성능을 향상시키기 위한 가능성을 제시하며, 저품질 MRI에서도 효과적으로 작동할 수 있는 모델을 제안합니다.



### SNR-EQ-JSCC: Joint Source-Channel Coding with SNR-Based Embedding and Query (https://arxiv.org/abs/2501.04732)
- **What's New**: 이 논문에서는 동적 채널의 영향을 줄이기 위한 새로운 경량 채널 적응형(Adaptation) 의미 코딩 아키텍처인 SNR-EQ-JSCC를 제안합니다. 이 구조는 일반적인 Transformer 모델을 기반으로 하며, Attention 블록 내에 신호 대 잡음 비율(SNR)을 임베딩하여 채널 적응을 달성합니다. 또한, 손실 함수에 페널티 항목을 도입하여 훈련 과정을 안정화합니다.

- **Technical Details**: SNR-EQ-JSCC는 채널 적응 쿼리(CAQ) 방식을 통해 다중 헤드 Attention 블록에서 주의 점수를 SNR 정보에 따라 조정합니다. 이 외에도, SNR을 MHA 블록의 입력에 임베딩하는 방식으로 채널 적응을 향상시킵니다. 제안된 방법은 평균 SNR만 사용하여 진행할 수 있으며, 이로 인해 재훈련 없이도 성능을 유지할 수 있습니다.

- **Performance Highlights**: 이미지 전송에 대한 시뮬레이션 결과, SNR-EQ-JSCC는 기존의 SwinJSCC를 PSNR 및 지각 메트릭에서 초과 달성하며, 저장 오버헤드는 0.05%에 불과하고 계산 복잡성은 6.38%에 해당합니다. 또한, 마음채널 적응 쿼리 방법은 지각 메트릭에서의 상당한 개선을 보여주며, 순간적인 SNR 피드백이 불완전할 때에도 평균 SNR만을 사용한 SNR-EQ-JSCC가 여전히 기준 방식을 초과합니다.



### Calculating Customer Lifetime Value and Churn using Beta Geometric Negative Binomial and Gamma-Gamma Distribution in a NFT based setting (https://arxiv.org/abs/2501.04719)
Comments:
          10 pages, 8 figures

- **What's New**: 이 논문에서는 Customer Lifetime Value (CLV)를 추정하기 위한 Beta-Geometric Negative Binomial Distribution (BGNBD) 및 Gamma-Gamma Distribution 모델을 소개하고 있으며, 이 모델들이 어떻게 블록체인 환경의 NFT 거래 데이터를 분석하는 데 유용한지 설명합니다. 기업은 이러한 모델을 통해 고객의 생애 가치를 평가하고, 데이터 기반의 마케팅 및 고객 유지 전략을 개발할 수 있습니다.

- **Technical Details**: BGNBD 모델은 고객의 거래 빈도와 관련된 간단한 통계 모델입니다. 이 모델은 고객의 거래 비율이 gamma 분포를 따르고, 거래 후 고객이 비활성화될 확률이 beta 분포를 따른다는 두 가지 주요 가정을 기반으로 하고 있습니다. Gamma-Gamma 모델은 고객 거래의 평균 가치를 추정하는 데 사용되며, 이는 거래 빈도와 독립적입니다.

- **Performance Highlights**: BGNBD 및 Gamma-Gamma 모델은 전통적인 CLV 추정 기법보다 더 정확한 결과를 제공합니다. 이 모델들은 고객 행동의 이질성을 고려하여 미래의 거래와 관련된 더 나은 예측을 가능하게 하고, 기업이 고객 중심의 마케팅 및 유지 관리 전략을 최적화하도록 돕습니다.



### Knowledge-Guided Biomarker Identification for Label-Free Single-Cell RNA-Seq Data: A Reinforcement Learning Perspectiv (https://arxiv.org/abs/2501.04718)
Comments:
          20 pages. arXiv admin note: substantial text overlap with arXiv:2406.07418

- **What's New**: 이 논문에서는 레이블 없는(‘label-free’) 유전체 데이터셋에서 가장 유용한 유전적 바이오마커를 식별하기 위한 유전자 패널 선택의 새로운 접근 방식을 제안합니다. 기존의 접근법들은 도메인 전문성(domain expertise)이나 기계 학습 모델에 의존하거나 경험적 최적화(heuristic-based optimization)를 사용하여 편향과 비효율성을 초래할 수 있습니다. 본 연구의 새로운 전략은 유전자 선택 알고리즘의 앙상블 지식을 활용하여 초기 탐색 공간을 안내하는 경계와 사전 지식을 확립합니다.

- **Technical Details**: 제안된 방법은 강화 학습(reinforcement learning)을 통해 전문가의 행동에 의해 형성된 보상 함수(reward function)를 통합하여 유전자 패널의 동적 정제(dynamic refinement) 및 목표 선택을 가능하게 합니다. 이 과정에서 초기 경계에서 발생할 수 있는 편향을 완화하면서도 강화 학습의 확률적 적응성(stochastic adaptability)을 활용합니다. 이러한 접근 방식은 유전자 패널 선택을 개선하는 데 중요한 역할을 합니다.

- **Performance Highlights**: 광범위한 비교 실험, 사례 연구(case studies), 그리고 후속 분석(downstream analyses)을 통해 제안된 방법의 효과성을 입증했습니다. 결과는 레이블 없는 바이오마커 발견(label-free biomarker discovery)을 위한 정밀도 및 효율성의 향상을 강조하며, 단일 세포 유전체 데이터 분석(single-cell genomics data analysis) 발전의 잠재성을 보여줍니다.



### One Node One Model: Featuring the Missing-Half for Graph Clustering (https://arxiv.org/abs/2412.09902)
Comments:
          Accepted by AAAI 2025

- **What's New**: 이번 논문에서는 전통적인 그래프 클러스터링 방법들이 노드의 특징 정보를 종종 간과하는 문제를 다룹니다. "하나의 노드, 하나의 모델"이라는 새로운 패러다임을 제시하여, 각 노드에 고유한 모델을 구축하고, 노드 그룹에 대한 예측을 조합하여 노드 레이블을 정의합니다. 특히, 새로운 특성 개인화 그래프 클러스터링 방법인 FPGC를 통해 노드에 적합한 클러스터 관련 특징을 식별하고, 이를 각 모델에 통합하여 최종 표현을 형성하는 방식을 소개합니다.

- **Technical Details**: FPGC는 squeeze-and-excitation 블록을 활용하여 각 노드에 적합한 클러스터 관련 특징을 선택합니다. 또한 데이터 증강 기법인 feature cross를 개발하여 낮은 차원의 특징 상호작용을 효과적으로 캡처합니다. 이러한 접근 방식은 동일한 그래프에서 다양한 학습을 가능하게 하여, 각 노드의 고유한 특성을 고려하는 데에 중점을 둡니다.

- **Performance Highlights**: 실험 결과, FPGC는 최신 클러스터링 방법들보다 성능이 우수함을 보여줍니다. 또한, FPGC는 기존 GNN 기반 모델에 통합될 수 있는 플러그 앤 플레이(tool) 특성을 가져, 다양한 시나리오에서 활용 가능성을 높입니다. 결과적으로, 이 연구는 그래프 클러스터링 성능을 향상시키기 위한 새로운 방향을 제시합니다.



### MADGEN: Mass-Spec attends to De Novo Molecular generation (https://arxiv.org/abs/2501.01950)
Comments:
          preprint

- **What's New**: 이번 연구에서는 MADGEN(Mass-spec Attends to De Novo Molecular GENeration)이라는 새로운 메소드를 제안합니다. 이 방법은 질량 분석 데이터에 의해 안내되는 scaffold 기반의 분자 구조 생성 방식으로, 두 단계인 스캐폴드 검색과 스펙트럼에 조건화된 분자 생성을 통해 작동합니다. MADGEN은 스캐폴드를 예측하고 그로부터 최종 분자를 생성하는 과정에서 분자의 다양성을 줄이고 생성 정확성을 향상시킵니다.

- **Technical Details**: MADGEN은 먼저 MS/MS 스펙트럼에서 스캐폴드를 검색하고, 그 스캐폴드를 기반으로 분자를 생성합니다. 스캐폴드 검색은 대조 학습을 통해 스펙트럼과 후보 화합물을 정렬하는 랭킹 문제로 설정됩니다. 그 후, 주어진 스캐폴드에서 MS/MS 스펙트럼을 이용하여 attention 기반 generative model로 최종 분자를 생성하게 됩니다. 이 과정은 대량의 스펙트럼 데이터에서 화학 구조의 해석을 가능하게 만듭니다.

- **Performance Highlights**: MADGEN을 NIST23, CANOPUS, MassSpecGym의 세 가지 데이터셋에서 평가하였으며, 예측 스캐폴드 및 오라클 스캐폴드를 사용하여 성능을 비교했습니다. 특히, 스펙트럼 정보를 통합하는 attention 메커니즘을 적용하여 생성 과정 전반에 걸쳐 효과성을 증명했습니다. 이는 새로운 생리활성 화합물이나 메타볼리즘을 발견하는 데 큰 기여를 할 수 있습니다.



New uploads on arXiv(cs.LG)

### The GAN is dead; long live the GAN! A Modern GAN Baselin (https://arxiv.org/abs/2501.05441)
Comments:
          Accepted to NeurIPS 2024. Code available at this https URL

- **What's New**: 이 논문에서는 GAN(Generative Adversarial Networks)의 훈련에서 발생하는 어려움에 대한 일반적인 주장을 반박하고, 기존의 경험적 트릭을 배제한 현대적인 GAN 표준인 R3GAN을 제시합니다. 이를 위해, 새로운 정규화된 상대적 GAN 손실을 도출하여 모드 드롭(mode dropping)과 비수렴(non-convergence) 문제를 해결합니다. 이 접근 방식은 GAN 훈련의 안정성을 향상시키며, 최신 아키텍처와 통합하여 더 나은 성능을 달성할 수 있음을 보여줍니다.

- **Technical Details**: 논문은 GAN의 목표를 안정성과 다양성이라는 두 가지 도전 과제로 설정합니다. 새로운 손실 함수는 제로 중심의 기울기 패널티를 도입하여 안정성을 높이고, 이론적으로도 지역 수렴(local convergence)을 보장합니다. 기존 GAN 구조의 한계를 극복하기 위해 새로운 현대적 아키텍처를 설계하였고, 스타일 GAN(StyleGAN)에서 필수적인 요소를 남기고 불필요한 기능을 제거하여 간소화된 구조를 제안했습니다.

- **Performance Highlights**: R3GAN은 FFHQ, ImageNet, CIFAR 및 Stacked MNIST 데이터셋에서 StyleGAN2보다 우수한 성능을 발휘하며, 동시대의 최신 GAN 및 diffusion 모델에 비해 경쟁력 있는 성과를 보여줍니다. 이러한 결과는 단순함에도 불구하고 성능이 향상된 GAN 구조를 통해 가능해졌음을 의미합니다. 이 연구는 GAN의 훈련을 더 간편하게 만들고, 최신 기술과의 통합을 통해 GAN의 진전을 도모합니다.



### Uncertainty-aware Knowledge Tracing (https://arxiv.org/abs/2501.05415)
Comments:
          Accepted by AAAI 2025

- **What's New**: 이번 논문에서는 Uncertainty-Aware Knowledge Tracing (UKT) 모델을 제안합니다. UKT는 학생 상호작용의 불확실성을 표현하기 위해 stochastic distribution embeddings를 사용하며, 학생의 학습 행동에서 상태 분포의 전이를 캡처하는 Wasserstein self-attention 메커니즘을 포함합니다. 이 모델은 aleatory uncertainty-aware contrastive learning loss를 통해 다양한 불확실성 유형에 대한 모델의 강건성을 강화합니다.

- **Technical Details**: UKT는 각 학생의 상호작용 이력을 Gaussian 분포로 모델링합니다. 여기서 학생의 기본 지식 수준과 불확실성은 각각 평균과 공분산으로 표현됩니다. 또한, Knowledge tracing을 위해 설계된 Wasserstein distance 기반의 self-attention 메커니즘을 통해 전역적인 변화를 모니터링하는 관계를 모델링합니다.

- **Performance Highlights**: 실험 결과, UKT는 6개의 실제 데이터셋에서 기존의 deep learning 기반 모델보다 KT 예측에서 유의미하게 우수한 성능을 보였습니다. 특히, aleatory uncertainty에 강한 강건성을 나타내며, 학생의 지식 마스터리를 정확히 표현할 수 있는 능력을 입증했습니다.



### On-line Policy Improvement using Monte-Carlo Search (https://arxiv.org/abs/2501.05407)
Comments:
          Accompanied by oral presentation by Gregory Galperin at NeurIPS 1996 (then known as NIPS*96)

- **What's New**: 이번 논문에서는 적응 제어기의 실시간 정책 개선을 위한 몬테카를로 시뮬레이션 알고리즘을 제안합니다. 이 알고리즘은 초기 정책을 바탕으로 각 가능한 행동의 장기적인 기대 보상을 통계적으로 측정하며, 기대 보상이 최대화되는 행동을 선택해 개선된 정책을 생성합니다. 특히, 이 알고리즘은 IBM의 SP1 및 SP2 병렬 RISC 슈퍼컴퓨터에 구현되었으며, 백개먼(backgammon) 분야에서 초기 결과를 통해 우수한 성능을 보였습니다.

- **Technical Details**: 정책 반복(policy iteration)이란 전통적으로 채택되는 알고리즘으로, 초기 정책을 기반으로 장기적인 기대 보상을 계산 후 이를 최적화하는 프로세스입니다. 전통적으로 오프라인에서 실행되며, 강화 학습(reinforcement learning) 방식을 활용하여 정책을 개선하는 다양한 접근법이 있지만, 실시간으로 개선된 정책을 계산하는 것은 여전히 느린 경향이 있습니다. 이에 반해 본 논문에서는 실시간으로 개선된 정책을 산출하기 위해 몬테카를로 검색(Monte-Carlo search)을 활용하여 각 행동의 기대 보상을 추정하는 온라인 알고리즘을 제안합니다.

- **Performance Highlights**: 몬테카를로 알고리즘은 다양한 초기 정책에 대해 근본 선수들의 오류율을 5배 이상 줄이는 결과를 제공합니다. 특히, 백개먼 게임에서 이 방법을 적용할 경우 평균적으로 약 200,000회의 몬테카를로 시행으로도 결정적인 결정을 도출할 수 있었습니다. IBM SP1 및 SP2 슈퍼컴퓨터에서 약 100K 기반 플레이어 결정을 초당 달성함으로써 몬테카를로 시뮬레이션의 효과성을 입증했습니다.



### TimeDP: Learning to Generate Multi-Domain Time Series with Domain Prompts (https://arxiv.org/abs/2501.05403)
Comments:
          AAAI 2025

- **What's New**: 본 논문에서는 TimeDP라는 다중 도메인 시계열 생성 모델을 제안합니다. 이 모델은 시계열 프로토타입 모듈을 활용하여 생성 조건으로 사용할 도메인 프롬프트를 학습합니다. 이 방식은 기존의 단일 도메인 데이터에 국한된 생성 모델의 한계를 넘어, 여러 도메인에서 새 데이터를 생성할 수 있는 가능성을 열어줍니다.

- **Technical Details**: TimeDP는 기본 요소로 시계열 프로토타입을 학습하고 이를 활용해 도메인 프롬프트를 구축하여 시계열 데이터 생성을 수행합니다. 훈련 과정에서 프로토타입은 시계열의 기초를 나타내는 역할을 하며, 각 샘플에 대해 프로토타입 할당 모듈을 적용해 특정 조건 프롬프트를 생성합니다. 샘플링 과정에서는 목표 도메인에서 몇 개의 샘플을 추출하여 도메인 프롬프트를 구성하고 이를 기반으로 시계열 데이터를 생성합니다.

- **Performance Highlights**: 실험을 통해 TimeDP는 기존의 기준 모델들을 능가하며 인도메인 생성 품질에서 최첨단 성능을 가지고 있음을 입증하였습니다. 또한, 이전에 보지 못한 도메인에서도 강력한 생성 능력을 보여주어 다중 도메인 시계열 생성의 가능성을 크게 확장했습니다.



### BRATI: Bidirectional Recurrent Attention for Time-Series Imputation (https://arxiv.org/abs/2501.05401)
- **What's New**: 이번 연구에서는 BRATI라는 새로운 딥러닝 모델을 소개합니다. BRATI는 Bidirectional Recurrent Networks와 Attention Mechanisms을 결합하여 다변량 시계열 데이터의 결측치를 보완하는 데 초점을 맞추고 있습니다. 이 모델은 시간적 의존성과 특성 간의 상관관계를 처리하며, 서로 반대 방향으로 작동하는 두 개의 보간 블록을 이용합니다. BRATI는 다중 시나리오의 결측 데이터 상황에서도 뛰어난 성능을 보여줍니다.

- **Technical Details**: BRATI 모델은 두 가지 유형의 RNN 계층과 Attention 메커니즘을 통합하여 장기 및 단기 의존성을 효과적으로 모델링합니다. 모델은 세 가지 실제 데이터 세트에서 평가되었으며, 무작위 결측치, 고정 길이 시퀀스 및 가변 길이 시퀀스와 같은 다양한 결측 데이터 시나리오를 다룹니다. 각 보간 블록은 시간의 방향에 따라 상이한 기능을 수행하며, 이를 통해 복잡한 패턴과 변수 간의 상관관계를 효과적으로 캡처합니다. 이러한 접근 방식은 일반적인 통계적 방법의 한계를 극복할 수 있는 잠재력을 가지고 있습니다.

- **Performance Highlights**: BRATI는 기존의 최첨단 모델들과 비교하여 일관되게 높은 정확성과 견고성을 보여줍니다. 연구 결과, BRATI는 다양한 결측 데이터 시나리오에서 다른 모델에 비해 뛰어난 성능을 발휘하여, 다변량 시계열 데이터의 보완 과제에 대한 새로운 해결책을 제시합니다. 또한, BRATI는 기존 연구에서 종종 간과되는 MNAR 시나리오에 대해서도 우수한 성능을 보여주었습니다.



### Mechanistic understanding and validation of large AI models with SemanticLens (https://arxiv.org/abs/2501.05398)
Comments:
          74 pages (18 pages manuscript, 7 pages references, 49 pages appendix)

- **What's New**: 이 논문에서는 기계 학습 모델의 불투명성을 극복하기 위해 SemanticLens라는 범용 설명 방법을 제안합니다. 이는 AI 모델 구성 요소가 인코딩한 숨겨진 지식을 시맨틱하게 구성된 멀티모달 공간으로 매핑하여, 각각의 뉴런이 특정 개념을 인코딩하는 것을 식별하는 텍스트 검색과 같은 독특한 작업을 가능하게 합니다.

- **Technical Details**: SemanticLens는 Neural Networks의 특정 구성 요소를 세밀하게 분석하고 비교할 수 있는 체계적인 분석을 제공합니다. 이 방법은 뉴런 자동 레이블링, 기능 역할 설명, 결정 사항 위반 감사 등을 수행하여 AI 모델의 결정 과정을 검증합니다. 또한, SemanticLens는 CLIP과 같은 기초 모델에 대한 설명을 통해 불확실한 결정 사항들을 보다 명확하게 설명합니다.

- **Performance Highlights**: 이 모델은 디버깅 및 검증을 효과적으로 수행하며, 모델 지식을 요약하고 기대와의 일치도를 높이는 데 기여합니다. 예를 들어, 흑색종 분류에서 ABCDE 규칙 준수를 확인하는 데 도움을 줍니다. SemanticLens는 AI 모델과 전통적인 엔지니어링 시스템 간의 '신뢰 격차'를 줄이는 데 기여하는 것을 목표로 하고 있습니다.



### Accelerated Diffusion Models via Speculative Sampling (https://arxiv.org/abs/2501.05370)
- **What's New**: 이번 연구는 Speculative Sampling(추측 샘플링) 기술을 새로운 영역인 diffusion models(확산 모델)로 확장하였습니다. 기존의 discrete sequences(이산 시퀀스)에서 벗어나, 지속적이고 벡터 값의 Markov chain(마르코프 체인)을 사용하는 방식을 도입하였습니다. 이를 통해 고품질이지만 계산 비용이 높은 diffusion model을 대상으로 하는 효율성을 높였습니다.

- **Technical Details**: 우리는 draft model(드래프트 모델) 훈련이 필요 없는 간단하고 효과적인 접근법을 포함한 다양한 드래프팅 전략을 제안합니다. 이 방법은 diffusion model에 즉시 적용될 수 있으며, 발전된 샘플 생성을 위한 기초를 제공합니다. 실험적으로, 여러 가지 diffusion models에서 함수 평가의 수를 절반으로 줄이며 생성 속도 향상을 이루었습니다.

- **Performance Highlights**: 연구 결과, 제안된 방법이 다양한 diffusion models에서 상당한 생성 속도 향상을 보여주었습니다. 함수 평가의 수가 절반으로 줄어들면서도, 목표 모델에서 정확한 샘플을 생성하는 데 성공하였습니다. 이는 대규모 언어 모델의 추론 가속화에 기여할 것으로 기대됩니다.



### No-Regret Linear Bandits under Gap-Adjusted Misspecification (https://arxiv.org/abs/2501.05361)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2302.13252

- **What's New**: 본 논문은 gap-adjusted misspecification 개념을 도입하여 linear bandits를 연구합니다. Liu et al. (2023)의 연구를 확장한 이 작업에서는 보상 함수가 선형적이지 않은 경우 더 자연스러운 모델을 제안합니다. 기존 방법론에서는 uniform misspecification parameter인 $\epsilon$을 사용하지만, 본 연구에서는 input $x$에서의 근사 오차가 최적이 아닌 간극(suboptimality gap)과 비례하게 될 것을 요구합니다.

- **Technical Details**: 성공적인 결과를 보여주는 LinUCB 알고리즘은 고전적 알고리즘으로, $ho$-gap-adjusted misspecification에 대해 자동으로 견고성을 지닙니다. 이때 $ho$는 $O(1/(d \sqrt{\log T}))$로 감소합니다. 또한, phased elimination 기반의 새로운 알고리즘을 통해 $ho$를 $O(1/\sqrt{d})$로 설정하여 $O(\sqrt{T})$의 최적 회귀(regret)를 달성하며, 배치의 수는 단지 $\log T$로 제한됩니다.

- **Performance Highlights**: 제안된 알고리즘은 배치 탐색(batch exploration)에서 최소 회귀를 보장하며, 상수 수준의 최적하지 않은 간극이 존재할 경우 적응할 수 있는 $O(\log T)$ 회귀를 제공합니다. 이 연구는 새로운 self-bounding argument를 통해 misspecification로 인한 회귀를 본질적으로 제한하며, 독창적인 귀납적 레마(inductive lemma)를 사용하여 유효한 모든 행동에 대한 미지의 오류를 하한으로 제한합니다.



### Stability and List-Replicability for Agnostic Learners (https://arxiv.org/abs/2501.05333)
- **What's New**: 이 논문은 Chase 외가 제안한 두 가지 완화된 조건하에서 학습 가능한 클래스들을 특성화합니다. 특히, 안정성 매개변수가 초과 오류에 따라 달라질 수 있는 설정에서, 우리는 Agnostic Stability가 Littlestone 차원으로 완전히 설명된다는 것을 증명합니다. 이는 realizable case에서도 이와 유사하게, 이러한 형태의 학습 가능성이 online learnability와 동등하다는 것을 의미합니다.

- **Technical Details**: 논문에서는 이론적 배경으로 PAC(Probably Approximately Correct) 학습 프레임워크를 따릅니다. 이 모델에서 학습자는 X×{0,1}에 대한 i.i.d. 샘플을 받고, 목표는 모집단 손실(population loss)을 최소화하는 가설을 생성하는 것입니다. 또한, 이 논문에서는 랜덤화된 학습 규칙과 안정성 조건을 포함하여 여러 가지 수학적 특성을 다룹니다.

- **Performance Highlights**: 특히, Littlestone 차원이 무한한 클래스는 안정적으로 PAC 학습할 수 없다는 중요한 결과를 보여줍니다. 두 번째 완화된 조건에서는 초과 오류에 따라 안정성 매개변수의 의존성을 제한하더라도, 여전히 유한 가설 클래스만이 전 세계 안정적으로 학습 가능하다는 것을 증명합니다. 이는 Agnostic Setting이 흥미로운 가설 클래스를 포착하기에는 너무 제한적이라는 기존 주장을 뒷받침합니다.



### Knowledge Transfer in Model-Based Reinforcement Learning Agents for Efficient Multi-Task Learning (https://arxiv.org/abs/2501.05329)
Comments:
          Preprint of an extended abstract accepted to AAMAS 2025

- **What's New**: 이 논문은 자원 제한 환경에서 모델 기반 강화 학습(model-based reinforcement learning)을 위한 효율적인 지식 전이(knowledge transfer) 접근법을 제안합니다. 고용량 멀티 태스크 에이전트(multi-task agent)를 1M 파라미터의 컴팩트 모델로 증류(distill)하여 MT30 벤치마크에서 28.45의 정상화 점수를 달성했습니다. 이를 통해 복잡한 멀티 태스크 지식을 효과적으로 통합할 수 있음을 보여주고, FP16 포스트 트레이닝 양자화(post-training quantization)를 통해 모델 크기를 50% 줄였습니다.

- **Technical Details**: 논문에서 소개한 접근법은 선생-학생(distillation) 프레임워크를 활용하여 모델 기반 RL의 특정 문제를 해결합니다. 317M 파라미터의 TD-MPC2 모델이 선생 모델로 사용되며, 1M 파라미터의 학생 모델이 그에 따라 훈련됩니다. 추가적인 보상 증류 손실(reward distillation loss)을 도입하여 교사와 학생 모델 간의 보상 예측의 평균 제곱 오차(MSE)를 계산하고, 이를 통해 학습 과정을 안정화합니다.

- **Performance Highlights**: 연구 결과, 증류 프로세스의 효과가 뚜렷하게 나타났으며, 특히 배치 크기를 줄이고 연장된 훈련을 통해 성능이 향상되었습니다. 그 결과로 증류된 모델이 28.12의 정상화 점수를 기록하며, 초기 모델 대비 2.77%의 개선을 보였습니다. 이러한 개선은 고용량 교사 모델로부터의 증류가 모델 성능에 미치는 긍정적 영향을 입증하였으며, 강화 학습 모델을 자원 제한 환경에서의 적용 가능성을 높였습니다.



### Distributed Learning and Inference Systems: A Networking Perspectiv (https://arxiv.org/abs/2501.05323)
Comments:
          This paper has been submitted to IEEE Network magazine and is still under review

- **What's New**: 이 연구는 중앙 집중 방식의 기존 AI 모델 훈련과 추론 방법의 한계를 극복하기 위한 새로운 프레임워크인 Data and Dynamics-Aware Inference and Training Networks (DA-ITN)을 제안합니다. DA-ITN은 분산 AI 시스템의 복잡성을 극복하고, 다양한 연구 영역에 대해 통찰력을 제공합니다.

- **Technical Details**: DA-ITN은 네트워크 토폴로지, 데이터 특성 분석, 노드 자원 평가 등을 포함하여 분산 AI 시스템의 여러 복잡성을 해결하기 위해 설계되었습니다. 이는 훈련 및 추론 과정에서 AI 모델을 효율적으로 배치하고, 쿼리와 응답의 이동성을 관리하며, 네트워크 연결 조건에 적응할 수 있도록 합니다.

- **Performance Highlights**: DA-ITN은 교육 요청을 자동으로 처리하고, 다양한 사용자 요구에 대한 AI 훈련 서비스를 제공합니다. 이 시스템은 다양한 컴퓨팅 자원과 데이터를 활용하여 훈련 성능을 최적화하고, 분산 환경에서도 신뢰성과 효율성을 제공하여 AI 기능을 향상시킵니다.



### Learning convolution operators on compact Abelian groups (https://arxiv.org/abs/2501.05279)
- **What's New**: 이번 연구에서는 Compact Abelian groups와 관련된 convolution operators 학습 문제에 대해 다룹니다. 정규화 기반 접근 방식(regularization-based approach)을 통해 convolution kernel에서의 정규성 조건을 논의하며, ridge regression (RR) estimator의 정확성을 유한 샘플 경계(finite sample bounds) 관점에서 특성화합니다. 기존의 RR 분석에서의 정규성 가정이 공간/주파수(localization)의 면에서 새로운 해석을 제공하는 것이 흥미로운 점입니다.

- **Technical Details**: 이 논문은 애벌리안 군(Abelian group) G에 대한 선형 연산자인 convolution operators에 집중합니다. 기입 및 출력이 무작위 신호(random signals)와 operator에 의해 처리된 노이즈 이미지로 구성된 통계적 프레임워크를 설정합니다. 이 연구에서 convolution kernel을 L2(G) 소속으로 가정하고, 정규화 회귀 추정기(ridge regression estimator) 분석에 중점을 두었습니다.

- **Performance Highlights**: 모델의 학습 성능은 ridge 회귀 이론의 결과를 조정하여 학습 오류를 연구함으로써 나타납니다. 정규화 통계학적 특성을 통해 문제의 구조를 보다 세밀하게 분석할 수 있는 기회를 제공하며, 이러한 접근법은 기존의 연산 학습(operator learning)의 분야에서는 드문 편입니다. 이 연구는 convolution kernel에 대한 학습 이론 보장을 유도하고, 이를 통해 새로운 통계적 통찰을 제공합니다.



### Deriving Coding-Specific Sub-Models from LLMs using Resource-Efficient Pruning (https://arxiv.org/abs/2501.05248)
- **What's New**: 이번 연구는 Large Language Models (LLMs)의 컴팩트한 모델 생성을 위한 비구조적 프루닝(unstructured pruning) 기법, 특히 코드 생성에 특화된 서브 모델을 효과적으로 추출하는 방법을 탐구합니다. 기존의 접근 방식이 프로그래밍 언어에 특화된 서브 모델의 효율적인 추출에 집중하지 않았던 점을 지적하며, Python, Java, C++, JavaScript와 같은 언어별 서브 모델을 추출하는 데 성공했습니다. 또한, 도메인 특화된 데이터셋이 프루닝 결과에 미치는 영향을 조사함으로써, 특정 프로그래밍 가지에 대한 LLM의 활성화 지역이 다름을 최초로 분석하고 증명했습니다.

- **Technical Details**: 연구에서는 다양한 도메인 특화된 캘리브레이션 데이터셋이 각기 다른 프루닝 결과에 미치는 영향을 조사했습니다. 이를 통해, 모델의 정확도를 유지하면서 언어별 서브 모델을 효과적으로 추출할 수 있는 프루닝 기법을 제안합니다. 특히, 차세대 LLM의 접근성을 높이기 위해 소비자 수준의 하드웨어에서의 실행을 지원하고, 실시간 개발 피드백에 중요한 빠른 추론 시간(inference time)을 가능하게 하는 방법을 모색했습니다.

- **Performance Highlights**: 논문은 LLM이 특정 도메인 작업을 통해 활성화되는 고유한 지역이 있음을 나타내는 분석적 증거를 제공합니다. 이를 통해 프로그래밍 언어 별 모델의 효율적 추출이 가능함을 보여주어, 다양한 프로그래밍 언어에 대한 요구를 충족할 수 있는 가능성을 열어줍니다. 결과적으로, 모델의 크기를 줄이면서도 기존 모델에 비해 만족할 만한 정확도를 확보하여 사용자가 보다 효율적으로 코딩 작업을 수행할 수 있도록 기여할 것으로 기대됩니다.



### An Algorithmic Approach for Causal Health Equity: A Look at Race Differentials in Intensive Care Unit (ICU) Outcomes (https://arxiv.org/abs/2501.05197)
- **What's New**: 이번 연구는 건강 불평등을 분석하기 위한 체계적인 프레임워크를 제안합니다. 호주와 미국의 ICU 결과에서 인종 및 민족적 불평등을 조사하여, 기존의 통계적 지표들이 불평등을 측정하는 데 부족하다는 점을 확인했습니다. 특히, 소수 환자들이 입원 시 더 젊고 만성 건강 상태가 좋지 않으며 긴급 및 긴급성이 없는 이유로 더 많이 입원하고, 높은 질병 중증도를 겪는다고 합니다.

- **Technical Details**: 연구는 호주와 미국의 ICU 데이터를 분석하였으며, 인종/민족별 차이를 이해하기 위해 인과적 관점에서 접근했습니다. 인과적 공정성 분석(framework of causal fairness analysis)을 통해 각 인과적 경로의 기여도를 평가, 분류하였고, SFM(Standard Fairness Model)을 사용하여 변수 간의 관계를 명확히 했습니다. 연구 결과, 호주 소수 환자들은 ICU 입원 후 사망률이 높았고, 미국에서는 반대의 경향을 보였습니다.

- **Performance Highlights**: 호주 원주민 환자들은 ICU에 자주 입원하지만 사망률이 낮은 반면, 미국의 아프리카계 미국인 환자들은 상대적으로 높은 사망률을 기록하였습니다. 조사 결과, 소수민족 환자들은 백인 동료들에 비해 ICU 재입원 가능성이 높았으며, 이는 기본적인 건강 관리 접근의 부족에 기인함을 밝혔습니다. 연구의 결과를 바탕으로 Indigenous Intensive Care Equity (IICE) 레이더 시스템을 개발하여, 원주민의 과도한 ICU 자원 활용을 모니터링할 수 있게 되었습니다.



### Learning In-Distribution Representations for Anomaly Detection (https://arxiv.org/abs/2501.05130)
- **What's New**: 이 연구에서는 이상 탐지를 위한 새로운 대조 학습 목표, FIRM(Focused In-distribution Representation Modeling)을 제안합니다. 기존의 대조 학습 방법들이 가진 한계를 극복하고, 단일 정상 클래스에서의 ID 샘플과 OOD(anomaly) 샘플 간의 명확한 구별을 가능하게 합니다. 또한, FIRM은 합성 아웃라이어(synthetic outliers)를 사전 과제로 포함해 ID 샘플의 밀집 클러스터링을 촉진하면서 아웃라이어와의 강한 분리를 보장합니다.

- **Technical Details**: FIRM은 ID 샘플의 내부 클래스 분산을 줄이는 것을 목표로 하며, 이로 인해 특징 공간에서 ID 표현의 밀접한 클러스터링을 유도합니다. 또한, 합성 아웃라이어들 간의 표현 다양성을 극대화하여 모델의 붕괴를 방지하고, ID 샘플과의 명확한 구분을 보장합니다. 이는 기존의 대조 학습 목표들이 가진 문제인 클래스 충돌을 해결하는 데 도움을 줍니다.

- **Performance Highlights**: FIRM은 NT-Xent와 비교하여 약 40배, SupCon과 비교하여 약 20배 더 빠른 수렴 속도를 보여줍니다. OOD 상황에서는 NT-Xent에 비해 특정 클래스에서 16.9% 성능 향상을 보였으며, 평균 CIFAR-10에서 각각 6.2%와 2%의 향상을 달성하였습니다. 광범위한 아블레이션 연구를 통해 FIRM의 표현 품질 개선이 다른 대조 목표들에 비해 유의미하게 나타났습니다.



### EquiBoost: An Equivariant Boosting Approach to Molecular Conformation Generation (https://arxiv.org/abs/2501.05109)
- **What's New**: 이번 논문에서는 분자 구조 생성의 핵심 과제를 해결하기 위한 새로운 부스팅 모델인 EquiBoost를 제안합니다. EquiBoost는 여러 개의 동등 불변 그래프 변환기(equivariant graph transformers)를 연속적으로 통합하여 3D 분자 구조를 반복적으로 개선하는 방식을 사용합니다. 이는 기존의 확산 모델(difussion models)에 의존하지 않고도 높은 정확도를 달성하며, 효율성 또한 크게 향상시킵니다.

- **Technical Details**: EquiBoost는 그래프 신경망(Graph Neural Networks, GNNs)의 개념을 기반으로 하며, 분자 구조의 회전 및 병진 불변성에 초점을 맞추고 있습니다. 모델은 특수 유클리드 그룹(SE(3))에서 동등 불변성을 달성하여 기존의 분자 구조 생성을 위한 템플릿 의존성을 제거합니다. EquiBoost는 훈련 중에 아주 빠르게 높은 정확도로 수렴하며, 추론 단계의 수를 기존의 수천 단계에서 단 5단계로 줄이는 혁신을 보여줍니다.

- **Performance Highlights**: GEOM 데이터셋에서 EquiBoost는 기존의 방법보다 생성을 더욱 향상시키면서도 다양성을 유지하며 성능을 크게 개선합니다. 이 결과는 EquiBoost가 특정 상황에서 확산 모델에 대한 강력한 대안이 될 가능성을 보여줍니다. 특히, 평균 최소 RMSD(Average Minimum RMSD, AMR)의 정확성을 크게 향상시켜 뛰어난 결과를 나타냈습니다.



### Hierarchical Decomposed Dual-domain Deep Learning for Sparse-View CT Reconstruction (https://arxiv.org/abs/2501.05093)
Comments:
          Published by Physics in Medicine & Biology (2024.4)

- **What's New**: 본 연구는 X-ray 컴퓨터 단층촬영(sparse-view CT)에서 발생하는 파국적인 스트리킹 아티팩트를 해결하기 위해 novel dual-domain deep learning framework를 제안합니다. 이 프레임워크는 전통적인 analytic 방법과 image-domain DL의 한계를 극복하며, high-order hierarchical decomposition 기법을 활용하여 reconstruction 성능을 향상시킵니다. 연구는 높은 차원의 변환을 통해 low-rank 성질을 활용한 접근법을 통해 효과적인 성능 향상을 보여줍니다.

- **Technical Details**: 제안된 방식은 hierarchical decomposition을 통해 filtered projection data를 구성하고, projection-domain DL 모델 및 image-domain DL을 연계하여 CT image-patch의 재구성을 수행합니다. Fourier domain에서 bowtie support의 특징을 활용하여, 프로젝션 데이터의 저순위(low rank) 성질과 관련을 명확히 합니다. 이를 통해, reconstruction performance가 크게 개선되는 모습을 확인했습니다.

- **Performance Highlights**: 연구 결과, 제안된 프레임워크는 기존의 수학적 규칙에 의해 성능이 향상된 것으로 나타났으며, 전통적인 analytic 방법인 filtered backprojection이나 기존의 deep learning 접근 방식에 비해 우수한 재구성 성능을 달성하였습니다. 이로 인해 의학 영상(image modality) 분야에서 새로운 연구의 가능성과 방향성을 열어주며, 기술적으로도 깊이 있는 이론적 설명을 제공하고 있습니다.



### DriVLM: Domain Adaptation of Vision-Language Models in Autonomous Driving (https://arxiv.org/abs/2501.05081)
- **What's New**: 최근 대형 언어 모델(large language models)의 성능이 크게 향상되었고, 이는 인공지능(artificial intelligence) 발전에 크게 기여하고 있습니다. 특히, 다중 모달 대형 언어 모델(multimodal large language models, MLLM)은 이미지, 비디오, 소리, 텍스트 등 다양한 모달리티를 결합할 수 있는 잠재력이 있습니다. 본 논문에서는 소규모 MLLM의 유용성을 탐구하고 이를 자율주행 분야에 적용하였습니다.

- **Technical Details**: 대부분의 MLLM은 매우 높은 계산 자원(computational resources)을 요구하여 연구자와 개발자들에게 큰 도전 과제가 되고 있습니다. 이 논문은 소규모 MLLM을 통해 이러한 문제를 해결하고, 자율주행 시스템에 융합함으로써 실용적인 접근 방식을 제안하고 있습니다. 이는 적은 자원으로도 MLLM의 가능성을 실험할 수 있는 기회를 제공합니다.

- **Performance Highlights**: 소규모 MLLM을 자율주행 분야에 적용함으로써, 실제 사용 가능한 솔루션을 개발하는 데 기여할 수 있기를 기대하고 있습니다. 이는 높아지는 계산 요구 없이도 실제 시나리오에 MLLM을 활용할 수 있는 길을 열어, 자율주행 기술의 발전에 긍정적인 영향을 미칠 것으로 보입니다.



### Analyzing Memorization in Large Language Models through the Lens of Model Attribution (https://arxiv.org/abs/2501.05078)
- **What's New**: 본 연구에서는 Large Language Models (LLMs)의 기억 현상을 구조적 관점에서 분석하고자 하였다. 주목할 점은 Attention 모듈이 메모리화와 일반화 성능에 미치는 영향을 검토하며, 특정 블록에서 Attention 모듈을 우회하는 방법을 제시하고 있다는 것이다. 이는 LLMs의 메모리화 문제를 해결하기 위한 기초적인 메커니즘을 이해하는 데 도움을 준다.

- **Technical Details**: 우리는 특정 Attention 블록을 우회하면서도 레이어 정규화(layer normalization)와 MLP 변환(MLP transformation) 등의 다른 구성 요소는 그대로 유지하여 메모리화와 일반화의 영향을 분리할 수 있는 체계적인 개입을 수행한다. 이론적 분석을 통해 Attention 모듈을 우회했을 때 출력 차이를 한정짓는 정리를 제공하며, 깊은 변환기 블록에서 메모리화가 주로 발생한다는 결론을 이끌어냈다.

- **Performance Highlights**: Pythia와 GPT-Neo 모델 패밀리, 그리고 여러 벤치마크 데이터셋을 통한 실험을 통해, 더 깊은 레이어에서 Attention 모듈을 우회하는 것이 메모리화 감소와 일반화 성능 유지에 효과적임을 증명하였다. 본 연구는 모델의 일반화 능력을 해치지 않으면서 메모리화를 완화할 수 있는 실용적인 접근 방식을 제시하고, 실제 응용에서의 윤리적 배치를 지원하는 기여를 한다.



### On Measuring Unnoticeability of Graph Adversarial Attacks: Observations, New Measure, and Applications (https://arxiv.org/abs/2501.05015)
Comments:
          KDD 2025

- **What's New**: 최근 연구에 따르면 그래프 신경망(GNN)은 다양한 작업에서 뛰어난 성능을 발휘하고 있지만, 적대적 공격에 취약한 문제를 안고 있습니다. 이 논문에서는 기존의 noticeability(인식 가능성) 측정 방법의 두 가지 주요 한계를 발견하고 이를 해결하기 위해 HideNSeek라는 새로운 측정 방법을 제안합니다. HideNSeek는 learnable edge scorer (LEO)를 활용하여 공격 엣지를 구별하고, imbalance-aware aggregation을 통해 최종 noticeability 점수를 도출합니다.

- **Technical Details**: HideNSeek는 GNN 기반으로 학습 가능한 엣지 스코어러(LEO)를 통해 각 엣지가 공격일 가능성을 점수화합니다. 이를 통해 공격 엣지가 상대적으로 낮은 점수를 받을 경우, 이는 더 눈에 띄게 간주됩니다. 최종 noticeability 점수는 이러한 점수를 집계하여 계산됩니다. HideNSeek의 방식은 간단한 규칙에 의존하는 기존 방법들과는 달리, 학습된 정보를 활용하여 보다 정교한 방식으로 공격 문제를 분석합니다.

- **Performance Highlights**: HideNSeek는 6개의 실제 그래프에서 실험을 통해 그 효과성을 입증했으며, LEO는 28개의 경우에서 11개의 경쟁자보다 우수한 성능을 보였습니다. 특히, HideNSeek는 감지 불가능 문제를 크게 개선하여 기존 방법보다 0.38배에서 5.75배 덜 bypassable(우회 가능함)하며, 저 공격률에서도 상당한 noticeability 점수를 제공합니다. 추가적으로 LEO는 GNN의 노드 분류 성능을 향상시키는 데에도 유용하다는 결과를 보여줍니다.



### A High-accuracy Calibration Method of Transient TSEPs for Power Semiconductor Devices (https://arxiv.org/abs/2501.05005)
- **What's New**: 본 논문에서는 전통적인 TSEP(thermal sensitive electrical parameter) 방법에서 간과되었던 교정(calibration) 방법이 모니터링 정확도에 미치는 영향을 강조하며, 고정밀 교정 방법을 제안합니다. 이 방법은 이중 펄스 테스트 동안 부하 전류로 인한 온도 차이를 감소시키기 위한 온도 보상 전략을 포함하고 있습니다. 뿐만 아니라, 기존 방법에서 종종 무시되는 소산 파라미터(stray parameters)의 영향을 분석하여 전기적 파라미터를 연결 짓는 방법을 제시하였습니다.

- **Technical Details**: 제안된 방법에서는 온도의 불확실성을 줄이기 위해 열 분석(thermal analysis)을 기반으로 한 온도 보상 전략이 적용됩니다. 또한, 주목하지 않던 커플링 파라미터(coupled parameters)를 식별하고, 무작위 오차(random errors)가 로그 가우시안 분포(logarithm Gaussian distribution)를 따른다는 것을 관찰하여 숨겨진 변수를 다룹니다. 마지막으로, 신경망(neural network)을 이용해 접합 온도(junction temperature) 예측 모델을 구축합니다.

- **Performance Highlights**: 실험적으로 검증된 결과, 제안하는 교정 방법은 기존의 방법들과 비교했을 때 평균 절대 오차(mean absolute error)가 30% 이상 감소하였습니다. 이 방법은 추가적인 하드웨어 비용이 들지 않으며 전반적인 일반화(generalization) 성능 또한 뛰어납니다. 이를 통해 TSEP 방법의 신뢰성을 크게 향상시킬 수 있는 가능성을 보여줍니다.



### Load Forecasting for Households and Energy Communities: Are Deep Learning Models Worth the Effort? (https://arxiv.org/abs/2501.05000)
Comments:
          This preprint was submitted to the Elsevier journal Energy and AI on December 18, 2024

- **What's New**: 이 연구는 에너지 커뮤니티의 단기 전력 수요 예측에 대한 최신 딥 러닝 모델들(LSTM, xLSTM, 그리고 Transformers)을 포괄적으로 비교하는 벤치마크를 제공합니다. 특히, 다양한 집계 규모(예: 가정 수)와 훈련 데이터 가용성(예: 훈련 데이터 기간)에 따른 모델 성능을 분석합니다. 또한, 합성 로드 프로파일에서의 전이 학습이 예측 오류에 미치는 영향과 모델의 크기(즉, 파라미터 수)에 대한 연구도 포함됩니다.

- **Technical Details**: 세 가지 딥 러닝 모델(xLSTM, LSTM, Transformer)과 세 가지 기본 모델(Persistence Prediction, k-Nearest Neighbors, 합성 로드 예측)을 비교하여 단기 전력 수요 예측을 수행합니다. 모든 모델은 자정부터 자정까지 24시간의 예측을 생성하며, 표준 입력 피처로는 날짜 및 시간 정보, 날씨 조건, 과거 로드 데이터 등이 사용됩니다. 모델은 다양한 커뮤니티 규모, 집계 크기, 훈련 데이터 크기 및 계절에 따라 훈련 및 테스트됩니다.

- **Performance Highlights**: 연구의 주요 발견 중 하나는 간단한 유지 보수 기준이 훈련 데이터가 6개월 이하로 제한될 때 딥 러닝 모델보다 더 나은 성능을 발휘한다는 것입니다. 또한, 공개적으로 사용 가능한 합성 로드 프로파일로 사전 훈련을 진행하면 첫 9개월 동안 nMAE가 평균 1.28%pt 개선됩니다. 깊이 있는 학습 모델은 집계가 증가할수록 성능이 향상되어, 50가구 에너지 커뮤니티에서는 nMAE 개선이 연간 약 600EUR의 경제적 이익으로 이어집니다.



### GiNet: Integrating Sequential and Context-Aware Learning for Battery Capacity Prediction (https://arxiv.org/abs/2501.04997)
Comments:
          6 pages

- **What's New**: 이번 논문은 배터리 용량 예측을 위해 GiNet이라는 새로운 모델을 제안합니다. GiNet은 Gated Recurrent Units(구간 순환 신경망)와 최신 Transformer 아키텍처인 Informer를 결합하여 배터리의 시간적 및 맥락적 정보를 효과적으로 캡처할 수 있는 능력을 지닙니다. 이 모델은 배터리의 복잡한 동작을 반영하며, 역사적 측정을 통해 정확한 배터리 용량 예측을 목표로 합니다.

- **Technical Details**: GiNet의 구조는 GRU 모듈을 통한 시퀀스 특징 추출로 시작하며, 이 특징은 원래의 데이터와 융합됩니다. 이후 융합된 특징은 Informer의 임베딩으로 처리되어 Attention 기반의 인코더와 디코더로 분석됩니다. 이 과정에서 Temporal Dynamics(시간적 동적)와 Long-term Dependencies(장기 의존성)를capturing하기 위해 GRU와 Transformer의 장점을 최대한 활용합니다.

- **Performance Highlights**: GiNet은 예측한 배터리 용량의 Mean Absolute Error(MAE)를 0.11로 줄였으며, 기존 GRU 및 Informer, 최신 알고리즘에 비해 각각 76%와 27%의 성능 향상을 보고했습니다. 이러한 결과는 GiNet의 알고리즘 통합의 중요성을 강조하며, 다양한 산업 응용에 대한 적용 가능성을 제시합니다.



### Battling the Non-stationarity in Time Series Forecasting via Test-time Adaptation (https://arxiv.org/abs/2501.04970)
Comments:
          Accepted at AAAI 2025

- **What's New**: 본 논문에서는 비정상적인 시계열(Non-stationary time series) 데이터를 다룰 수 있도록 설계된 최초의 테스트 시간 적응 프레임워크인 TSF-TTA(TSF Test-Time Adaptation)를 소개합니다. 제안된 방법인 TAFAS(Teaching Adaptation for Forecasting In Non-stationary Time Series)는 소스 예측기를 적절히 조정하여 테스트 분포의 지속적인 변화를 반영합니다. 이를 통해, 기존 접근 방식들이 직면하는 한계를 극복하고, 핵심 의미를 보존하면서도 예측기의 신뢰성을 향상시키는 새로운 가능성을 열어주고 있습니다.

- **Technical Details**: TAFAS는 주기성 인식 적응 일정(Periodic-aware Adaptation Scheduling, PAAS) 및 게이트 캘리브레이션 모듈(Gated Calibration Module, GCM)로 구성됩니다. PAAS는 의미 있는 주기 패턴을 반영할 수 있도록 부분적으로 관측된 진실 값을 적응적으로 획득하고, GCM은 예측기와 일치하도록 테스트 시간 입력을 보정합니다. 이러한 모듈들은 비정상적인 테스트 시간 입력에서 소스 예측기가 능동적으로 적응할 수 있도록 돕습니다.

- **Performance Highlights**: TAFAS는 다양한 TSF 아키텍처에 걸쳐 일관되게 성능 향상을 보여주며, 특히 분포 변화가 더 뚜렷한 장기 예측 시나리오에서 두드러진 성과를 발휘합니다. 실제 실험 결과에 따르면, TAFAS는 이전에 보지 못한 테스트 데이터 스트림에서 Chronos의 예측 오류를 최대 45% 개선할 수 있음을 보여줍니다. 이러한 성과는 TAFAS가 기존의 비정상성을 다루기 위한 방법들과의 통합을 통해 더욱 강화된다는 점에서 주목할 만합니다.



### Targeted Adversarial Denoising Autoencoders (TADA) for Neural Time Series Filtration (https://arxiv.org/abs/2501.04967)
Comments:
          [Accepted] Artificial Intelligence for Time Series Analysis (AI4TS): Theory, Algorithms, and Applications @ AAAI 2025, Philadelphia, PA, USA

- **What's New**: 본 논문에서는 전통적인 신호 필터링 알고리즘의 한계를 극복하기 위해, 로지스틱 공분산 타겟팅 적대적 디노이징 오토인코더(TADA)를 기반으로 한 새로운 기계 학습 필터링 알고리즘을 제안합니다. 이 알고리즘은 EEG(뇌전도) 신호에서 EMG(근전도) 노이즈를 제거하는 데 특화되어 있으며, 복잡한 시간 시계열 필터링을 효과적으로 수행할 수 있도록 설계되었습니다. TADA 시스템은 기존의 깊은 학습 아키텍처에 비해 적은 메모리 소비로 높은 성능을 발휘하며, 67명의 시험자를 대상으로 한 EEGdenoiseNet 데이터 세트에서 평가를 진행하였습니다.

- **Technical Details**: TADA 시스템은 세 가지 구성 요소로 이루어져 있습니다: (1) 타겟팅된 디노이징 오토인코더, (2) 적대적 훈련, (3) 최종 로지스틱 공분산 타겟팅 레이어. 이 시스템은 Selective한 신경 시간 시계열 데이터 필터링을 위한 방법으로, EEG 시간 시계열 데이터에 적용하기 위해 다양한 공개 데이터 세트를 조사하였고, EEGdenoiseNet 데이터 세트를 선택하여 훈련 및 평가에 활용하였습니다. 이 데이터 세트는 4514개의 깨끗한 EEG 세그먼트와 5598개의 근육 아티팩트 세그먼트를 포함하여, 고유한 데이터 취득을 위한 실험에 적합합니다.

- **Performance Highlights**: TADA 필터는 정량적 지표에서 기존 신호 필터링 알고리즘들을 초월하며, 특히 상관계수(Correlation Coefficient), 시간적 RRMSE(Temporal RRMSE), 주파수 RRMSE(Spectral RRMSE)와 같은 메트릭에서 우수한 성능을 발휘합니다. 또한, 400,000개 미만의 학습 가능한 파라미터로 구성된 모델 크기를 가지고 있으며, 여러 깊은 학습 아키텍처와 경쟁력을 유지합니다. 향후 다양한 배포 사례에서 TADA의 가능성을 평가할 필요가 있습니다.



### Open Problems in Machine Unlearning for AI Safety (https://arxiv.org/abs/2501.04952)
- **What's New**: 이 논문에서는 AI 시스템의 안전성을 보장하기 위한 머신 언러닝의 한계를 비판적으로 분석합니다. 특히, AI 안전성에서의 포괄적인 해결책으로서의 한계를 강조하며, 이로 인해 신뢰할 수 있는 AI의 구현을 위해 대안적인 접근 방식이 필요하다는 점을 지적합니다. 논문은 CBRN(화학, 생물학, 방사선 및 핵 안전)과 같은 민감한 분야에서의 이중 용도(dual-use) 지식 관리의 복잡성을 다룹니다.

- **Technical Details**: 이 연구는 머신 언러닝이 AI 안전성에 미치는 영향을 분석하기 위해 네 가지 주요 응용 분야를 정리합니다. 1) 안전 핵심 지식 관리: 위험한 지식을 제거하는 데 있어 모델의 기본 능력이 재구성되는 문제를 강조합니다. 2) 값 정렬 문제: 인간의 선호와 불일치하는 행동을 수정하는 데 있어서 단순한 지식 제거로는 해결되지 않는 한계를 고찰합니다. 3) 사생활 및 법적 준수: GDPR과 CCPA와 같은 규제를 준수하기 위한 데이터 제거의 필요성을 설명합니다.

- **Performance Highlights**: 이 논문은 머신 언러닝의 효과적인 구현을 위해 직면한 기술적 도전과 그 한계를 명확히 합니다. 안전성 보장을 위해 필요한 지식 제거는 예상치 못한 부작용이나 새로운 위험을 초래할 수 있으며, 이러한 복잡성을 감안할 때 충분한 검증 및 평가가 필요함을 강조합니다. 최종적으로, 미래 연구 방향을 제시하며 AI 안전을 위한 머신 언러닝의 잠재적인 애플리케이션에 대한 인식을 높이고, 다양한 대안을 모색할 필요성을 일깨웁니다.



### A New Perspective on Privacy Protection in Federated Learning with Granular-Ball Computing (https://arxiv.org/abs/2501.04940)
- **What's New**: 이 논문은 기존의 Federated Learning (FL) 접근 방식에서 모델의 내부 매개변수 및 출력에 초점을 맞춘 기존 연구들이 대부분인 반면, 입력 레벨에서의 문제 해결을 제안합니다. 특히, Granular-Ball Federated Learning (GrBFL)이라는 새로운 프레임워크를 통해 이미지를 최적의 조밀도로 여러 지역으로 분할하여 그래프 구조로 재구성하는 방법론을 소개합니다. 이를 통해 데이터의 프라이버시를 보장하고, 효율성 또한 향상시키며, 모델의 유용성을 유지하는 방법을 제안합니다.

- **Technical Details**: GrBFL은 두 가지 핵심 구성 요소로 이루어져 있습니다: (1) 그라뉼러 볼 계산을 기반으로 한 지식 재구성으로, 그래디언트 분석을 통해 대표 정보를 선택하고 이를 그래프 구조로 재구성하는 과정을 포함합니다. (2) 그래프 입력을 바탕으로 한 공동 집계로, 그래프 기반의 FL에서 불안정성 문제를 해결하는 프로시멀( proximal) 항을 도입합니다. 또한, 정보의 중복성을 제거하여 모델의 분류 성능을 유지하면서도 FL의 효율성을 대폭 개선하는 2차원 이진 탐색 분할 알고리즘을 설계했습니다.

- **Performance Highlights**: 다양한 이론적 분석과 실험 결과는 GrBFL이 FL의 프라이버시를 강화하고 효율성을 높이며, 동시에 타 방법들보다 우수한 성능을 보여준다는 것을 입증합니다. 특히, GrBFL은 공격자로 하여금 원본 데이터를 완전히 재구성하는 것을 방지하여 프라이버시 보호 측면에서도 탁월한 효과를 발휘합니다. 실험 결과, GrBFL은 CNNFL 보다 재구성된 이미지의 유사성이 낮아 프라이버시 보호가 더욱 강화되었습니다.



### SpecTf: Transformers Enable Data-Driven Imaging Spectroscopy Cloud Detection (https://arxiv.org/abs/2501.04916)
Comments:
          23 pages, 5 figures, in review. Code repository: this https URL

- **What's New**: 현재 및 차세대 가시-단파 적외선(VSWIR) 이미징 분광계는 지구 시스템 프로세스를 정량화할 수 있는 획기적인 능력을 제공하지만, 신뢰할 수 있는 구름 스크리닝(cloud screening)은 여전히 주요 도전 과제로 남아있습니다. 본 논문에서는 SpecTf(Spectroscopic Transformer)라는 새로운 딥러닝 아키텍처를 소개하며, 이는 오직 분광 정보만으로 구름을 탐지합니다. 이를 통해 공간적(spatial) 또는 시간적(temporal) 데이터에 대한 의존 없이 구름의 물리적 관계를 학습합니다.

- **Technical Details**: SpecTf는 분광 측정값을 이미지 채널이 아닌 시퀀스로 취급하여 구름 탐지의 효율성을 극대화합니다. 모델은 EMIT 데이터에 대한 구름 스크리닝을 수행하며, 특이한 파라미터 수로도 이전의 방법들과 비슷한 성능을 나타냅니다. 이 구조는 물리적으로 유의미한 스펙트럼 패턴을 학습할 수 있는 주의를 통해 해석 가능성을 제공합니다.

- **Performance Highlights**: SpecTf는 교육되지 않은 데이터셋에서 구름 마스크 프로덕트를 정량적으로 평가하여 현재 기준선(baseline) 접근법보다 모든 기준에서 뛰어난 성능을 보여줍니다. 다른 머신러닝 모델(GBT, ANN)도 포함하여, SpecTf는 평균적으로 더 높은 True Positive Rate (TPR)와 낮은 False Positive Rate (FPR)를 기록하여 성능 향상을 보였습니다. 특히, ML 모델은 구름 탐지의 정밀도와 재현율의 조화 평균을 측정하는 F-beta 스코어에서도 기준선을 초과합니다.



### Online Continual Learning: A Systematic Literature Review of Approaches, Challenges, and Benchmarks (https://arxiv.org/abs/2501.04897)
- **What's New**: 이번 연구는 Online Continual Learning (OCL) 분야에 대한 최초의 종합적 체계 문헌 검토(Systematic Literature Review, SLR)로, 총 81개의 접근 방법을 분석하고, 1,000개 이상의 특징을 추출했습니다. 다양한 데이터셋의 활용을 검토하여, 실시간 데이터 흐름에 적응할 수 있는 모델을 개발하는 데 필요한 주요 과제를 강조하고 있습니다. 또한, 고유한 메모리 메커니즘 설계와 효율적인 프레임워크를 통한 미래 연구 방향을 제시합니다.

- **Technical Details**: OCL의 시스템적 분석을 위해 체계적인 방법론을 수립했으며, 500개 이상의 구성 요소와 1,000개 이상의 특징을 식별했습니다. 연구 질문들은 OCL 문제 설정, 실제 적용 방법, 사용되는 데이터셋, 접근 방법의 공통 구성 요소 및 도구에 대한 것을 포함합니다. 체계적인 검토 방법론에 따라 데이터 수집의 네 가지 단계를 설정하여 연구 기반의 견고한 분석을 가능하게 했습니다.

- **Performance Highlights**: 본 연구는 OCL의 적용 가능성을 다양한 분야에서 보여주며, 이미지 분류, 객체 탐지 및 다중 모달 비전-언어 작업 등에서 활용되는 83개의 데이터셋 목록을 정리했습니다. 연구 결과는 OCL의 주요 도전 과제를 다루는 데 도움을 줄 뿐만 아니라, 이를 통해 분야의 발전을 위한 기초 자료를 제공합니다. 연구의 종합성과 구조적인 방법론이 OCL 연구의 새로운 기회를 선도할 것으로 기대됩니다.



### Quantifying Itch and its Impact on Sleep Using Machine Learning and Radio Signals (https://arxiv.org/abs/2501.04896)
- **What's New**: 본 논문에서는 인공지능(AI)과 홈 라디오 장치를 결합하여 만성 가려움증을 더 객관적으로 모니터링하는 방법을 소개합니다. 이 기술은 가려움증과 수면 품질에 미치는 영향을 동시에 평가할 수 있는 장점이 있습니다. 특히, 이 장치는 환자가 착용할 필요가 없으며, 장기간의 모니터링을 가능하게 하여 환자의 부담을 줄입니다.

- **Technical Details**: 기존의 주관적인 가려움증 평가 방법의 한계를 극복하기 위하여, 저자들은 RF(무선 주파수) 신호를 사용하여 수면 및 가려움 증상을 모니터링하는 새로운 접근법을 개발했습니다. 이 장치는 유선 신호를 송신하고, 환자의 환경에서 반사된 신호를 수집하여 가려움증과 수면의 상관관계를 분석합니다. 연구 결과, 이 방법은 높은 정확도로 가려움증을 감지하고 수면의 질을 평가할 수 있음을 입증했습니다.

- **Performance Highlights**: 임상 연구에서 20명의 만성 가려움증 환자를 대상으로 1개월 간 가정에서 모니터링한 결과, 이 장치는 ROC AUC = 0.997이라는 높은 정확도를 기록했습니다. 또한, 가려움증과 수면의 질 간의 상관관계를 보여줌으로써, 수면 효율성 감소(R = 0.6, p < 0.001)와 더불어 수면 잠복기의 증가(R = 0.68, p < 0.001)를 확인했습니다. 이 연구는 만성 가려움증 환자의 치료 반응을 평가할 수 있는 유용한 도구가 될 수 있음을 시사합니다.



### A Look into How Machine Learning is Reshaping Engineering Models: the Rise of Analysis Paralysis, Optimal yet Infeasible Solutions, and the Inevitable Rashomon Paradox (https://arxiv.org/abs/2501.04894)
- **What's New**: 본 논문은 기계 학습(ML) 모델에 대한 회의적 시각과 공학에서 경험적으로 도출된 코드 조항(codal provisions) 및 방정식의 수용 간의 철학적 긴장을 조사합니다. 특히 ML이 구조 공학에 어떻게 통합될 수 있는지를 탐구하며, 기존 공학 철학 및 전문 정체성에 도전하는 방법을 제시합니다.

- **Technical Details**: 이 연구에서는 ML이 예측 정확도를 개선하고 설계를 최적화하며 복잡한 행동을 분석하는 방식에 대해 설명합니다. 또한 ML 통합 시 발생할 수 있는 세 가지 주요 패러독스(Analysis Paralysis, Infeasible Solutions, Rashomon Effect)를 식별하며, 각 패러독스는 공학적 직관과 알고리즘의 해석 가능성에 관련된 문제를 제기합니다.

- **Performance Highlights**: ML을 통해 예측 정확도가 향상되었지만, 이는 물리적 메커니즘에 대한 이해를 악화시킬 수 있습니다. 또한 최적화된 솔루션이 공학적 직관에 도전하는 비전통적 설계로 이어질 수 있으며, 이는 공학 및 공학교육의 인식론적 변화가 필요함을 암시합니다. 마지막으로, 이 논문은 ML과 전통적 원칙을 조화롭게 통합해야 한다고 주장합니다.



### Multilinear Tensor Low-Rank Approximation for Policy-Gradient Methods in Reinforcement Learning (https://arxiv.org/abs/2501.04879)
- **What's New**: 이 논문에서는 다중선형 매핑을 제안하여 강화학습(Reinforcement Learning, RL) 정책의 매개 변수를 효율적으로 추정하는 방법을 다룹니다. 기존의 신경망(Neural Network, NN) 기반 접근법이 직면했던 수렴(convergence), 구조적 적합성(architectural suitability)과 같은 문제를 저감하기 위해, PARAFAC 분해를 활용하여 텐서(tensor) 저랭크(low-rank) 정책을 설계하고 있습니다. 이러한 저랭크 정책 모델은 RL 알고리즘에 통합 가능하며, 학습의 효율성을 높이는 가능성을 보여줍니다.

- **Technical Details**: 이 연구의 핵심은 정책 파라미터를 텐서로 집계하고, 텐서 완성(tensor-completion) 기법을 활용하여 저랭크를 강제함으로써(transfer)를 실현하는 것입니다. 마르코프 결정 과정(Markov Decision Process, MDP)에서 상태-행동 맵을 학습하는 대신, 확률적 매핑(parameter mapping)의 파라미터를 직접 추정하는 정책 기반 접근을 취합니다. 이를 통해 REINFORCE, AC, TRPO 및 PPO와 같은 다양한 정책 기반 알고리즘에서도 저랭크 모델을 적용할 수 있도록 하였습니다.

- **Performance Highlights**: 제안된 텐서 저랭크 모델은 NN 모델에 비해 계산 복잡성과 샘플 복잡성을 줄이며, 유사한 보상(reward)을 달성하는 데 성공했습니다. 수치 실험을 통해, 저랭크 방법이 NN 기반 모델에 비해 빠른 수렴 속도를 보여주었음을 입증합니다. 이러한 성과는 정책 기반 RL의 효율적인 알고리즘 개발에 기여할 것으로 기대됩니다.



### Intelligent Gradient Boosting Algorithms for Estimating Strength of Modified Subgrade So (https://arxiv.org/abs/2501.04826)
Comments:
          17 pages

- **What's New**: 본 연구에서는 포장재의 강도를 결정짓는 주요 요소인 지반(subgrade)의 특성을 추정하기 위해 기계 학습(machine learning) 기반의 두 가지 부스팅 기법인 CatBoost 및 XGBoost, 그리고 Support Vector Regression (SVR)이 적용되었습니다. 특히, 이 연구는 수화석회로 개선된 쌀 껍질 재 (HARSH)와 함께 사용되는 지반 토양의 특성을 평가하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 연구는 121개의 실험 데이터를 바탕으로 CBR, UCS 및 R의 추정에 필요한 플라스틱 한계(plastic limit), 액체 한계(liquid limit), 플라스틱성 지수(plasticity index), 점토 활동(clay activity) 등의 특성을 입력값으로 사용하였습니다. 네 가지 평가 지표, 즉 결정계수(R2), 평균 제곱근 오차(RMSE), 평균 절대 오차(MAE) 및 평균 절대 백분율 오차(MAPE)를 활용하여 모델의 성능이 평가되었습니다.

- **Performance Highlights**: 결과에 따르면, XGBoost는 CBR, UCS 및 R 추정에서 각각 R2 값이 0.9994, 0.9995 및 0.9999로 CatBoost 및 SVR 보다 우수한 성능을 보였습니다. 또한, SVR은 CBR과 R의 추정에서 CatBoost보다 우수한 성능을 보여준 반면, CatBoost는 UCS 추정에서 SVR보다 더 좋은 결과를 나타냈습니다. 마지막으로, 특성 민감도 분석 결과, HARSH 비율이 증가함에 따라 추정된 특성 값이 증가하는 경향이 있음을 확인했습니다.



### Decentralised Resource Sharing in TinyML: Wireless Bilayer Gossip Parallel SGD for Collaborative Learning (https://arxiv.org/abs/2501.04817)
- **What's New**: 계속해서 발전하는 마이크로컨트롤러 유닛(MCU)의 컴퓨팅 능력 덕분에 엣지 디바이스에서도 머신러닝 모델을 지원할 수 있게 되었습니다. 그러나 분산된 연합 학습(Decentralised Federated Learning, DFL)을 그러한 장치에 배포하는 데는 지속적인 연결성 부족, 제한된 통신 범위 등 여러 가지 주요 도전이 있습니다. 본 논문에서는 자원이 제한된 환경에서 이러한 문제를 해결하기 위해 새로운 프레임워크인 이층 Gossip 분산 병렬 확률적 경량 하강법(Bilayer Gossip Decentralised Parallel Stochastic Gradient Descent, GD-PSGD)을 제안합니다.

- **Technical Details**: 이 프레임워크는 지리적 그룹화를 위한 Distributed K-means 클러스터링과 두 단계로 모델 집계를 위한 gossip 프로토콜을 결합한 계층 구조의 통신 체계를 통합하고 있습니다. 각 클러스터 내 및 클러스터 간의 효과적인 통신을 위해 DK-means를 활용하여 자원 제약이 있는 환경에서도 통신 오버헤드를 줄이고 확장성을 개선할 수 있도록 설계되었습니다.

- **Performance Highlights**: 테스트 결과, 제안된 방법은 IID 데이터셋에서 Centralised Federated Learning(CFL)에 비슷한 정확도를 달성하며, 수렴을 위해 추가로 1.8 라운드만 필요하다는 것을 보여주었습니다. Non-IID 데이터셋에 대한 정확도 감소는 중간 수준의 데이터 불균형에서도 8% 미만으로 유지되며, 이는 최소한의 성능 손실로 확장 가능하고 개인정보 보호가 가능한 학습을 지원할 수 있다는 점을 강조합니다.



### Probabilistic Skip Connections for Deterministic Uncertainty Quantification in Deep Neural Networks (https://arxiv.org/abs/2501.04816)
Comments:
          15 pages, 9 figures

- **What's New**: 본 논문에서는 결정론적 불확실성 정량화(Deterministic Uncertainty Quantification, UQ)를 위한 새로운 접근 방법인 확률적 건너뛰기 연결(Probabilistic Skip Connections, PSCs)을 제안합니다. 이 방법은 네트워크의 피처 추출기(Feature Extractor) 훈련을 수정하지 않고도 중간 레이어에서의 불확실성을 효과적으로 추정할 수 있게 해줍니다. PSCs는 네트워크에서 발생하는 피처 붕괴(Feature Collapse)를 고려하여 적절한 레이어를 식별하여, 단일 패스를 통해 결정론적 UQ를 가능하게 합니다.

- **Technical Details**: PSC들은 피처 붕괴를 예방하기 위해 네트워크의 중간 레이어에서 파생된 피처 벡터에 확률적 모델을 적합시키는 방식으로 작동합니다. 기존 방법들이 스펙트럴 정규화(Spectral Normalization)를 통해 네트워크를 재학습시켜야 하는 반면, PSC는 이미 학습된 네트워크에서 적절한 레이어를 찾아 불확실성을 정량화합니다. 주요 수학적 기초는 입력 x와 레이어 j에 대한 피처 공간의 거리를 기반으로 하며, 인지 손실(aleatoric)과 인지적 불확실성(epistemic uncertainty)을 분리가능하게 합니다.

- **Performance Highlights**: 실험 결과, PSC는 기존 단일 패스 방법들과 비슷하거나 우수한 OOD 탐지 및 UQ 성능을 나타냅니다. PSC를 활용함으로써 전체 네트워크 구조를 변경할 필요 없이도 높은 품질의 불확실성 정량화 및 OOD 기능을 구현할 수 있습니다. 이 방법은 다양한 신경망 구조에 적용 가능하여 더 넓은 범위의 UQ 확장성을 제공합니다.



### Fast, Fine-Grained Equivalence Checking for Neural Decompilers (https://arxiv.org/abs/2501.04811)
- **What's New**: 이 논문에서는 neural decompilers를 위한 새로운 기법인 codealign을 소개합니다. 기존의 평가 기법들이 neural decompiler의 예측의 정확성을 충분히 보여주지 못하는 문제를 해결하고자 합니다. codealign은 instruction-level code equivalence를 생성하여, 함수 내에서의 동등한 명령어 사이의 관계를 정량적으로 분석할 수 있도록 돕습니다.

- **Technical Details**: codealign은 두 개의 함수 쌍에 대해 작동하며, 각 함수는 명령어의 시퀀스로 모델링됩니다. 각 명령어는 변수, 상수 또는 다른 명령어의 실행 결과인 값을 처리하고, 이는 명령어의 피연산자(operands)로 불립니다. 예를 들어, 코드에서 this_status > best_status와 같은 명령어는 두 값을 비교하는 연산을 수행하며, 이러한 이름의 대조를 통해 함수의 동등성을 정의합니다.

- **Performance Highlights**: codealign은 neural decompiler가 생성한 코드의 정확성 및 변수 이름 품질을 평가하는 데 매우 유용한 도구입니다. 기존 평가 메트릭스와 비교할 때, codealign은 함수의 유사성을 단순히 수치적으로 측정하는 것이 아니라, 각 명령어 및 변수 간의 동등성을 상세하게 분석할 수 있도록 해줍니다. 이는 연구자들이 neural decompiler의 신뢰성을 더 잘 이해하고 개선할 수 있는 기반을 마련합니다.



### DAREK -- Distance Aware Error for Kolmogorov Networks (https://arxiv.org/abs/2501.04757)
Comments:
          Accepted at ICASSP25, 5 pages + 2 pages supplementary material, 3 figures

- **What's New**: 이 논문에서는 Kolmogorov-Arnold Networks (KANs)를 위한 거리 인식 오류 경계를 제공하는 DAREK (Distance Aware Error for Kolmogorov networks)라는 새로운 오류 경계 추정기를 소개합니다. 기존 Z. Liu의 방법은 느슨하고 거리 인식이 부족하여 불확실성이 일정 상수의 비례로만 정의됩니다. 본 연구에서는 거리 인식을 기반으로 한 더 긴밀하고 해석 가능한 오류 경계를 제공하며, 이를 통해 KANs의 성능을 향상시키고자 합니다.

- **Technical Details**: KANs는 B-Spline 기반을 사용해 활성화 함수로 활용되며, 이를 통해 해석 가능하고 분석 가능한 신경망을 구축합니다. 본 연구에서는 머신러닝 모델의 불확실성 추정 방법을 worst-case 방식을 채택하였고, 입력 데이터와의 거리에 따라 불확실성이 증가하는 특징을 가진 추정기를 설계하였습니다. 기존의 Monte Carlo 방식과 달리, 본 방법은 계산 비용이 낮고 한정된 데이터로도 더 강건한 분석을 가능하게 합니다.

- **Performance Highlights**: 이 연구는 KAN을 사용하여 희소 레이저 스캔 포인트로부터 물체의 형상을 추정하는 방법을 제안합니다. DAREK를 통해 KAN은 매끄러운 함수를 적합시키고, 그 적합에 대한 오류 경계를 제공하여, 진정한 장애물 형상을 신뢰성 있게 포함함을 입증했습니다. 또한, 본 방법이 Monte Carlo 접근 방식보다 더 빠르고 효율적임을 확인했습니다.



### Decentralized Diffusion Models (https://arxiv.org/abs/2501.05450)
Comments:
          Project webpage: this https URL

- **What's New**: 이번 논문에서는 기존의 중앙 집중형 인프라의 의존성을 제거하는 분산 훈련 프레임워크인 Decentralized Diffusion Models(DDMs)을 제안합니다. DDMs는 독립적인 클러스터들을 통해 확장 가능하게 훈련할 수 있으며, 이는 GPU 클러스터 간의 네트워크 부담을 줄이고 인프라 비용을 낮추는데 기여합니다. 연구자들이 더욱 효율적이고 저렴한 자원을 활용할 수 있게 되어, 특정 GPU 노드를 통해 수요 기반으로 적시에 할당할 수 있는 기회를 제공합니다.

- **Technical Details**: Decentralized Diffusion Models는 데이터셋의 파티션을 기반으로 전문가 모델을 훈련하여, 이전의 대규모 훈련 요구 사항을 분산 처리합니다. 새로운 훈련 목적인 Decentralized Flow Matching(DFM)을 적용하여 개별 전문가 모델은 서로 교류없이 독립적 훈련을 진행합니다. 이때, 라우터 모델이 각 전문가의 관련성을 판단하여 테스트 시 최적의 예측 값을 제공하도록 합니다.

- **Performance Highlights**: DDMs는 ImageNet과 LAION Aesthetics 데이터셋에서 기존의 모노리스(Monolithic) 확산 모델 훈련보다 더 나은 성능을 보였습니다. 8개의 전문가 모델을 통해 최적의 성능을 달성했으며, 이는 일반 모델보다 더 효과적인 대체가 될 수 있음을 보여줍니다. 마지막으로, 8개의 30억 매개변수를 가진 대규모 분산 모델을 훈련하여 고해상도 이미지 생성을 입증하였습니다.



### Consistent Flow Distillation for Text-to-3D Generation (https://arxiv.org/abs/2501.05445)
Comments:
          Project page: this https URL

- **What's New**: 본 논문에서는 3D 생성 모델의 효율성을 개선하기 위한 새로운 방법인 Consistent Flow Distillation (CFD)을 제안합니다. 이를 통해 기존의 Score Distillation Sampling (SDS)의 한계인 시각적 품질과 다양성 저하 문제를 해결하고자 합니다. CFD는 2D 이미지 플로우의 일관성을 강조하며, 다양한 시점에서의 3D 생성 품질을 개선합니다.

- **Technical Details**: CFD는 다중 시점에서의 일관된 Gaussian noise를 활용하여 3D 객체의 생성을 돕습니다. 이 과정에서 noise transport equation을 기반으로 물체 표면에서의 일관된 noise 패턴을 유지하여 그라디언트를 계산합니다. 이는 diffusion ODE(Ordinary Differential Equation) 또는 SDE(Stochastic Differential Equation) 샘플링 과정을 통해 3D 생성을 직접 유도합니다.

- **Performance Highlights**: 실험 결과, CFD는 기존의 텍스트-투-3D(text-to-3D) 생성 방식에 비해 품질과 다양성에서 현저히 개선된 성능을 보였습니다. 본 방법을 통해 생성된 3D 자산은 현실적인 외관과 형태를 가지며, 동일한 텍스트 프롬프트에 대해 다양한 3D 객체를 샘플링할 수 있습니다. CFD는 기존의 SDS와 비교하여 계산 비용이 거의 추가되지 않았음에도 높은 성능을 달성했습니다.



### From Simple to Complex Skills: The Case of In-Hand Object Reorientation (https://arxiv.org/abs/2501.05439)
Comments:
          website: this https URL

- **What's New**: 이번 연구에서는 면밀한 보상이 요구되는 sim-to-real(시뮬레이션에서 실제로) 간의 격차를 줄이기 위해 사전 학습된 저수준 스킬을 활용하는 새로운 시스템을 소개합니다. 우리는 고수준의 정책을 통해 이러한 저수준 스킬을 조합하여 인손에서의 객체 재배치 작업을 수행하는 계층적 정책을 제안합니다. 이로 인해 이전 스킬을 재사용할 수 있어, 매번 처음부터 학습하는 데 필요한 인간의 노력을 크게 줄일 수 있습니다.

- **Technical Details**: 여기서 제안하는 시스템은 사전 학습된 객체 회전 스킬을 기반으로 한 계층적 정책을 통해 작동합니다. 이 정책은 저수준 스킬이 수행해야 할 작업을 결정하고, 환경의 피드백을 기반으로 옵니다. 또한, 이 시스템은 전통적인 포즈 추정 기법의 한계를 극복하기 위해 프로프리오셉션(proprioceptive) 정보를 사용하여 시간을 통해 객체의 포즈를 예측합니다.

- **Performance Highlights**: 우리는 제안하는 시스템이 기존 방법에 비해 빠르게 수렴하고 높은 성능을 이끌어낼 수 있음을 실험을 통해 입증하였습니다. 특히, 이 정책은 다양한 객체에 대해 일반화된 상태 추정기를 학습하여 실제 환경으로 쉽게 전이할 수 있습니다. 최종적으로, 학습된 정책은 전이 가능성이 높은 저수준 정책을 자연스럽게 활용하면서 높은 효율성을 보장합니다.



### Entangled Mean Estimation in High-Dimensions (https://arxiv.org/abs/2501.05425)
- **What's New**: 본 논문에서는 고차원의 얽힌 평균 추정(high-dimensional entangled mean estimation) 문제를 다루며, 신호의 부분집합 모델(subset-of-signals model)에서 수행됩니다. 이 연구는 N개의 독립적인 랜덤 포인트가 주어졌을 때, 일반적인 공분산을 가진 가우시안에서 추출된 포인트의 공통 평균(mu)을 추정하는 과제에 초점을 맞추고 있습니다. 이 분야의 기존 연구가 일차원 설정에서 주로 이루어져 왔다는 점에서, 다차원 설정의 정보 이론적 측면에 대한 이해가 부족했음을 지적합니다.

- **Technical Details**: 우리는 계산 효율성이 높은 알고리즘을 설계하여 정보 이론적으로 거의 최적의 오류(error)를 달성하는 방법을 제시합니다. 이 알고리즘의 최적 오류는 함수 f(α, N)와 term √(D/(αN))의 합으로 표현되며, 이는 각각 일차원 문제와 서브가우시안 오류율(sub-Gaussian error rate)과 관련이 있습니다. 새로운 거부 샘플링 절차(rejection sampling)를 활용하여, 평균(mu) 값에 큰 편차를 보이는 포인트를 필터링하여 더 정확한 추정값(hat μ)을 학습하는 반복적 세련화(strategy)를 적용합니다.

- **Performance Highlights**: 우리가 제안한 알고리즘은 평균적으로 최적의 오류와 비교했을 때, 풍부한 이론적 근거를 갖추고 있습니다. 또한, 거부 샘플링에서 발생할 수 있는 편향(bias) 문제를 체계적으로 분석하여 극복했습니다. 이 연구는 다차원 평균 추정 문제에 대한 새로운 통찰을 제공하며, 향후 통계학 및 컴퓨터 과학 분야에서의 적용 가능성을 갖습니다.



### Using LLMs to Infer Non-Binary COVID-19 Sentiments of Chinese Micro-bloggers (https://arxiv.org/abs/2501.05423)
Comments:
          11 pages, 4 figures

- **What's New**: 이 논문은 COVID-19 위기 상황 동안 공공 감정을 연구하여 사회가 어떻게 양극화되는지를 설명합니다. 중국의 인기 마이크로블로깅 사이트인 Weibo의 게시글을 분석하여 긍정적, 부정적, 빈정대는, 중립적 감정으로 분류하였습니다. 이 연구는 중국 플랫폼의 감정 분석의 빈틈을 메우며, 건강 위기 기간 중 사회 감정의 역학에 대한 통찰을 제공합니다.

- **Technical Details**: 연구에서 사용된 데이터셋은 하버드의 Weibo COVID-19 데이터셋으로, 총 4,049,407개의 게시글을 포함하고 있습니다. 데이터셋 내에서 중복 게시글과 재게시물을 식별하고 분류하여 감정 분류의 정확성을 높였습니다. Llama 3 모델을 사용하여 게시물 감정을 분류하기 위한 few-shot prompting 기법을 적용하여 감정을 예측하였습니다.

- **Performance Highlights**: Weibo에서의 감정 변화 분석은 사회적 사건과 정부 행동이 공공 여론에 미치는 영향을 이해하는 데 기여합니다. 감정 분석을 통해 사용자가 자가 검열을 통해 빈정대는 표현을 사용한 다양한 경향을 관찰할 수 있었습니다. 이러한 연구 결과는 디지털 커뮤니케이션이 사회에 미치는 영향을 이해하는 데 중요한 시사점을 제공합니다.



### A Novel Pathology Foundation Model by Mayo Clinic, Charit\'e, and Aignostics (https://arxiv.org/abs/2501.05409)
- **What's New**: 이 보고서에서는 RudolfV 접근 방식을 기반으로 하는 새로운 비전 파운데이션 모델을 소개합니다. 이 모델은 Mayo Clinic과 Charité - Universitätsmedizin Berlin에서 수집된 120만 개의 조직병리 전체 슬라이드 이미지로 훈련되었습니다. 결과적으로, 이 모델은 21개의 공공 벤치마크 데이터셋에서 최첨단 성능을 보여주며, 매개변수 수나 데이터셋 크기가 가장 크지 않음에도 불구하고 뛰어난 성과를 거두었습니다.

- **Technical Details**: 모델 훈련에는 490,000개의 사례에서 추출된 120만 개의 디지털 이미지가 사용되었습니다. 특히, H&E, IHC 및 특수 염색 데이터가 포함되어 있으며, 다양한 확대 비율에서 훈련이 진행되었습니다. 이 보고서는 ViT-H/14 구조를 활용하여 632백만 개의 매개변수를 가진 모델을 훈련하기 위해 RudolfV의 알고리즘을 적응시켰습니다.

- **Performance Highlights**: 모델 성능 평가는 선형 프로빙 프로토콜을 통해 수행되었으며, 21개의 벤치마크 데이터셋을 사용하여 다양한 작업에서 비교되었습니다. 평가 결과는 첫 번째 세대 모델과 비교하여 뛰어난 성과를 나타내며, 특히 클래스 토큰 및 평균 토큰을 고려한 평균 정확도에서 높은 점수를 기록했습니다. 이러한 검증 작업은 또한 데이터 재현성과 비교 가능성을 높이는 데 기여하고 있습니다.



### TimeRL: Efficient Deep Reinforcement Learning with Polyhedral Dependence Graphs (https://arxiv.org/abs/2501.05408)
Comments:
          17 pages, 11 figures, 5 bibliography pages

- **What's New**: 본 논문은 Deep Reinforcement Learning (DRL) 프로그램을 실행하기 위한 새로운 시스템인 TimeRL을 소개합니다. TimeRL은 eager execution의 동적 특성과 graph-based execution의 전체 프로그램 최적화를 결합하여 DRL 알고리즘의 동적 데이터 종속성을 효과적으로 처리합니다. 이를 통해 복잡한 DRL 알고리즘을 최대 47배 빠르게 실행하며 GPU 메모리 사용량은 16배 적게 줄였습니다.

- **Technical Details**: TimeRL은 재귀 텐서(Recurrent Tensors)라는 선언적 프로그래밍 모델을 도입하여 동적 종속성을 수식적으로 표현할 수 있도록 합니다. 이 시스템은 Polyhedral Dependence Graphs (PDGs)를 사용하여 전체 프로그램을 하나의 그래프로 표현하여, 다양한 실행 지점 간의 관계를 상징적 표현으로 나타냅니다. 이를 통해 TimeRL은 자동 메모리 관리 최적화 및 실행 스케줄을 자동으로 결정하는 기능을 제공합니다.

- **Performance Highlights**: TimeRL은 현재의 DRL 알고리즘에 대해 최대 47배 더 빠른 훈련 속도를 기록하며, 기존 DRL 시스템에 비해 메모리 효율성을 16배 개선했습니다. 이러한 성능 향상은 TimeRL이 동적 데이터 종속성을 효과적으로 처리하고, 연산을 병렬화 및 증분화하며, 알고리즘 특화된 스케줄링을 수행함으로써 이루어졌습니다.



### Integrating Explainable AI for Effective Malware Detection in Encrypted Network Traffic (https://arxiv.org/abs/2501.05387)
Comments:
          Accepted and presented on PanAfriCon AI 2024

- **What's New**: 이번 연구는 악성 네트워크 트래픽 탐지를 위해 설명 가능한 인공지능(Explainable Artificial Intelligence, XAI) 기법을 통합하였습니다. 우리는 다양한 측면에서 추출한 멀티뷰 특징을 사용하여 악성 활동을 식별하는 앙상블 학습 모델을 개발했습니다. 1,127개의 고유 연결을 포함한 강력한 데이터세트를 통해 54개의 맬웨어 패밀리를 분석하고, 기존 공개 데이터세트보다 더 많은 데이터를 확보했습니다.

- **Technical Details**: 연구에서는 악성 트래픽을 효과적으로 표현하기 위해 여러 원천에서 수집한 데이터를 기반으로 다양한 특징 세트를 분석했습니다. Concentrated features include handshake, certificate, inter-arrival time, packet length, statistical features, meta-connection features and cipher suite used. XGB(Extreme Gradient Boosting) 모델을 활용하여 99.32%의 정확도, 99.53%의 정밀도 및 99.43%의 F1-score를 기록했습니다.

- **Performance Highlights**: 제안된 모델은 CTU-13 데이터세트를 기준으로 99% 이상의 정확도와 정밀도를 달성했습니다. Shapley Additive Explanations(SHAP)을 활용하여 최대 패킷 크기, 패킷 간 평균 도착 시간, 전송 계층 보안 버전 사용이 중요한 특징임을 규명했습니다. 이러한 통찰력은 악성 암호화 트래픽 탐지의 투명성과 신뢰성을 향상시키는 데 기여합니다.



### Developing a Foundation of Vector Symbolic Architectures Using Category Theory (https://arxiv.org/abs/2501.05368)
Comments:
          13 pages, no figures, 2 tables, one appendix

- **What's New**: 이 논문은 Vector Symbolic Architectures (VSA)에 대한 카테고리 이론(Category Theory)의 첫 적용 시도를 제시합니다. VSAs는 인지 과학에서 신경 처리와 인간의 기호적 추론의 통합 필요성에서 나온 고차원 벡터 표현에 대한 대안적인 접근법입니다. 논문은 문헌 조사를 통해 VSAs와 카테고리 이론 간의 교차점이 부족하다는 점을 강조하고 있습니다.

- **Technical Details**: VSAs는 기호를 벡터로 인코딩하는 기초 벡터 공간의 구조로, 고유한 "의미"를 가진 벡터들을 비교하는 방식으로 작동합니다. 연구에 따르면 VSAs는 두 가지 주요 연산인 바인딩(binding)/언바인딩(unbinding)과 번들링(bundling)을 통해 더 복잡한 기호를 표현할 수 있습니다. 본 논문은 VSA에 바람직한 특성을 수집하고 이를 카테고리 관련 개념에 연결하는 방법을 제시하고 있습니다.

- **Performance Highlights**: VSA의 주요 이점은 정보 저장 능력에서 나타납니다. d 차원의 벡터 공간에서는 2^d 개의 고유(near-)직교 벡터를 생성할 수 있어 높은 수준의 잡음 저항을 제공합니다. 또한, VSAs는 생물학적 신경망 모델에서도 구현 가능하다는 점에서 인지 기능 특성을 설명하기 위한 유망한 기초로 자리잡고 있습니다.



### Stream Aligner: Efficient Sentence-Level Alignment via Distribution Induction (https://arxiv.org/abs/2501.05336)
Comments:
          AAAI Alignment Track 2025 Poster

- **What's New**: 이 논문에서는 Streaming Distribution Induce Aligner (Stream Aligner)라는 새로운 정렬(paradigm) 접근법을 제시합니다. 이 방법은 LLM의 성능을 향상시키면서도 효율성을 유지하는 데 중점을 두고 있습니다. 특히, 문장 수준의 반복적인 수정 과정을 통해 이전 모델의 출력을 개선하며, 이로 인해 적은 자원으로도 더 나은 결과를 도출할 수 있습니다.

- **Technical Details**: Stream Aligner는 정렬 과정을 문장 수준으로 세분화하여, 모든 세션에서 사용자의 선호도를 학습하고 이를 바탕으로 수정된 출력을 기본 모델로 피드백합니다. 이 과정은 기계적 학습(machinery learning) 원리와 선호 데이터셋(preference dataset)을 기반으로 하며, 반복적인 피드백이 발생하는 구조를 가집니다. 이는 인퍼런스 과정 중에 실제로 인간의 의도와 가치를 LLM 출력에 효과적으로 주입할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 2B 모델의 Stream Aligner는 Llama2-70B-chat 모델에서 응답의 유용성을 41.2% 증가시키고, 무해성은 36.0% 향상시켰습니다. 또한, Stream Aligner-8B 모델은 Llama3-70B-Instruct 모델에서 수학적 능력을 3.5% 개선하는 등 성능이 입증되었습니다. 이러한 결과는 Stream Aligner가 추가 모델의 의존도를 줄이며, LLM의 추론 능력을 향상시키는 데 기여함을 나타냅니다.



### The explanation dialogues: an expert focus study to understand requirements towards explanations within the GDPR (https://arxiv.org/abs/2501.05325)
Comments:
          Artificial Intelligence and Law (Springer Nature)

- **What's New**: 이 논문에서는 Explainable AI (XAI)가 법적 전문가들의 설명 기대치를 어떻게 충족할 수 있는지를 파악하기 위한 'Explanation Dialogues'라는 전문가 초점 연구를 소개합니다. 주로 유럽 일반 데이터 보호 규정(GDPR)을 중심으로 진행되었습니다. 이 연구는 온라인 질문지와 후속 인터뷰로 구성되어 있으며, 신용 분야에서의 사례를 통해 진행되었습니다.

- **Technical Details**: 이번 연구는 고온 이론(grounded theory)을 바탕으로 계층적이고 상호 연결된 코드 세트를 도출했습니다. 이 코드는 참가한 전문가들이 XAI에 대해 가지고 있는 관점들을 요약하는 데 사용되었습니다. 또한, 연구는 법적 전문가들이 제공된 XAI 설명을 어떻게 인식하고 기대하는지를 조사하기 위해 설계되었습니다.

- **Performance Highlights**: 법적 전문가들은 XAI 설명이 이해하기 어렵고 정보가 부족하다고 평가했습니다. 연구 결과, 설명의 법적 준수 정도는 개별의 권리가 얼마만큼 행사될 수 있는지와 밀접하게 연결되어 있다고 밝혀졌습니다. 이와 함께, 개발자들을 위한 여러 권장사항과 추가적인 법적 분석이 필요한 영역도 제시되었습니다.



### Optimizing Distributed Deployment of Mixture-of-Experts Model Inference in Serverless Computing (https://arxiv.org/abs/2501.05313)
- **What's New**: 이 논문은 서버리스 컴퓨팅(serverless computing) 플랫폼에서 머신러닝(Machine Learning) 추론 서비스를 최적화하기 위한 새로운 접근 방식을 제시합니다. 특히, Mixture-of-Experts (MoE) 모델의 효율적인 배포를 위한 Bayesian optimization 프레임워크를 도입하여, 전문가 선택(prediction)과 비용 절감 측면에서 혁신적인 기여를 합니다. 이를 통해 MoE 모델의 서버리스 플랫폼에서의 실행 성능을 확보하며, 전체 청구 비용(billed cost)을 최소화할 수 있습니다.

- **Technical Details**: 논문에서는 MoE 모델의 추론 수행 시 발생하는 두 가지 주요 도전 과제를 해결하기 위한 다양한 기술적 방안을 소개합니다. 첫째, 통계학적 방법(Bayesian decision-making)을 이용하여 입력 토큰의 전문가 선택을 사전 예측하는 방법을 제시합니다. 둘째, 서버리스 플랫폼의 특성에 맞춘 여러 가지 scatter-gather 통신(deployment) 설계를 제안하여, 데이터 전송과 모델 실행을 더 효과적으로 파이프라인(pipeline) 처리합니다. 이러한 기술적 접근은 메모리 활용 및 실행 시간을 최적화합니다.

- **Performance Highlights**: AWS Lambda에서의 광범위한 실험을 통해 제안된 기술이 MoE 모델의 모든 레이어에서 청구 비용을 최소 75.67% 절감함을 보여줍니다. 서버리스 컴퓨팅 환경에서 LambdaML과 비교할 때, 제안된 설계는 비용을 43.41% 낮추고, 처리량(throughput)의 감소 폭은 최대 18.76%에 불과하여 성능도 유지합니다. 이러한 결과는 MoE 모델 배포의 비용 효율성을 강화하는 중요한 진전을 나타냅니다.



### Private Selection with Heterogeneous Sensitivities (https://arxiv.org/abs/2501.05309)
Comments:
          21 pages, 18 figures

- **What's New**: 이번 연구는 Differentially Private (DP) 선택 문제에서 후보자 민감도의 이질성을 활용하는 새로운 기법을 탐구합니다. 기존의 Report Noisy Max (RNM) 방식은 모든 후보의 점수가 동일한 민감성을 갖는다고 가정하는데, 이는 실제 상황에서는 자주 성립하지 않습니다. 이러한 이유로, 연구에서는 기존 방법들을 개선할 수 있는 가능성을 모색합니다.

- **Technical Details**: DP 선택 알고리즘은 후보자 집합과 점수 함수를 기반으로 하여 가장 높은 점수를 가진 후보자를 선택하면서 데이터 주체의 개인 정보를 보호하는 것을 목표로 합니다. 본 논문에서는 score와 민감도의 상관관계를 통해 최적의 DP 선택 메커니즘을 결정하는 지침을 제시합니다. 수정된 GEM과 결합된 GEM 같은 변형 알고리즘을 제안하며, 이는 기존 기법보다 성능이 우수함을 보여줍니다.

- **Performance Highlights**: 연구 결과, 후보자 민감도에 따른 이질성을 이용한 메커니즘이 RNM보다 나은 성능을 보일 수 있으나, 모든 상황에서 RNM보다 나은 성능을 반드시 보장하지는 않습니다. 특히, 상관 관계 기반의 선택 방법론이 특정 극단적인 상황에서 뛰어난 향상을 이루는 것을 발견했습니다. 결합된 GEM은 기존의 GEM 및 수정된 GEM보다 일반적으로 더 좋은 성능을 보장합니다.



### Comparison Study: Glacier Calving Front Delineation in Synthetic Aperture Radar Images With Deep Learning (https://arxiv.org/abs/2501.05281)
- **What's New**: 이번 연구에서는 해양에서 끝나는 빙하의 쇄빙 전선(calving front) 위치 변화를 추적하기 위해 Deep Learning (DL) 시스템을 적용한 최초의 평가를 진행했습니다. Synthetic Aperture Radar (SAR) 이미지를 활용하여 독립적으로 모니터링할 수 있는 방법을 제안하고 있습니다. 이러한 접근 방식은 날씨나 조명에 구애받지 않는 대규모 모니터링을 가능하게 합니다.

- **Technical Details**: 연구는 10명의 주석자(annotator)와 함께 진행되어, 가장 성능이 뛰어난 DL 시스템이 인간 성능과 어떻게 차이가 나는지를 조사하였습니다. DL 모델의 평균 오차는 221 m로 나타났고, 인간 주석자의 평균 오차는 38 m였습니다. 이 결과는 현재의 DL 시스템이 인간 성능을 아직 따라잡지 못하고 있음을 보여주며, 완전 자동화를 위한 추가 연구의 필요성을 강조합니다.

- **Performance Highlights**: 연구에서 Vision Transformers, 기초 모델(foundational models), 그리고 더 많은 정보를 포함하고 처리하는 전략이 향후 연구 방향으로 제시되었습니다. 이러한 개선점을 통해 얼음 붕괴의 자동 모니터링이 가능해질 것입니다.



### Off-Policy Evaluation and Counterfactual Methods in Dynamic Auction Environments (https://arxiv.org/abs/2501.05278)
Comments:
          9 pages, 15 figures, IEEE format

- **What's New**: 본 연구에서는 자원 Allocation이라는 맥락에서 Off-Policy Evaluation (OPE) 기법의 활용을 탐구합니다. OPE는 실험 비용을 절감하며 신규 정책 평가를 수행할 수 있도록 하여 의사 결정 과정의 신뢰도를 높이고, 정책 선택 및 최적화를 가속화합니다. 특히, 동적 경매 환경에서 정책의 성과를 신속하게 평가하는 방법을 제안합니다. 이를 통해 정책 테스트에 필요한 시간 및 자원을 줄이는 데 기여할 수 있습니다.

- **Technical Details**: OPE는 역사적 데이터를 기반으로 새로운 정책의 성과를 추정하는데 사용됩니다. 이 연구에서는 에이전트가 자원을 획득하기 위한 지불 정책을 설정하고, 이를 통해 경매의 성과를 평가하는 구조를 가지고 있습니다. 행동 공간은 지불 정책으로 설정되며, OPE 방법을 통해 지불 정책의 성공 여부를 평가합니다. 여러 가지 평가자를 통해 지속적 및 비지속적 정책을 고려하여 성과를 검토합니다.

- **Performance Highlights**: 이 연구에서 제안된 접근 방식은 다양한 새 지불 정책을 평가할 수 있는 가능성을 탐색합니다. A/B 테스트를 통한 기존 정책과의 비교를 통해 데이터 기반으로 정책의 성과를 예측할 수 있습니다. OPE 방식을 통해 높은 신뢰도로 새로운 정책을 실험하기 전, 가장 유망한 정책을 미리 검토할 수 있으므로 불확실성을 최대한 줄일 수 있습니다.



### CellViT++: Energy-Efficient and Adaptive Cell Segmentation and Classification Using Foundation Models (https://arxiv.org/abs/2501.05269)
- **What's New**: 이번 논문에서는 디지털 병리학에서 세포를 식별하고 분할하는 데 있어 기존 방법의 한계를 극복하기 위해 $	ext{CellViT}^{	ext{++}}$이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 비전 트랜스포머(Vision Transformers)를 활용하여 깊은 세포 특성과 분할 마스크를 동시에 계산하며, 최소한의 데이터로 훈련할 수 있어 탄소 발자국을 크게 줄입니다. 또한, $	ext{CellViT}^{	ext{++}}$는 면역형광 염색(immunofluorescence stainings)을 이용하여 병리학자 주석 없이 훈련 데이터 세트를 생성할 수 있는 능력을 입증합니다.

- **Technical Details**: 프레임워크의 핵심 구성 요소는 PanNuke 데이터 세트에서 세분화 모델을 사전 훈련(pretraining)하는 것입니다. 이 데이터 세트는 19개의 조직 타입에 걸쳐 190,000개의 세포가 광범위하게 주석(annotation) 처리되어 있습니다. 다양한 세포 분류 모듈을 테스트하기 위해 CellViT 시리즈의 변형 모델들이 평가되었으며, 이들 모두는 기준 모델에 비해 평균 7.45% 이상의 성능 향상을 보였습니다. 실험은 F1-스코어(mF1)와 평균 팬옵틱 품질(mPQ)이라는 지표를 사용하여 진행되었습니다.

- **Performance Highlights**: 세분화 성능에서 Segment Anything Model (CellViTSAM-H)이 가장 높은 성능을 기록하며, SOTA(results) 성능을 보여주었습니다. Li et al. (2024) 연구에서도 CellViT과 종양 세분화 모델 조합을 통해 0.7243의 mF1-score를 달성하여 우수한 결과를 보고했습니다. 이번 연구가 제안하는 $	ext{CellViT}^{	ext{++}}$은 새로운 임상 및 세포 유형에 대해 뛰어난 제로샷(segmentation) 능력을 보여주며, 고품질 훈련 데이터 세트를 효율적으로 생성할 수 있는 잠재력을 가집니다.



### Enhancing Plagiarism Detection in Marathi with a Weighted Ensemble of TF-IDF and BERT Embeddings for Low-Resource Language Processing (https://arxiv.org/abs/2501.05260)
Comments:
          Accepted into LoResLM: The First Workshop on Language Models for Low-Resource Languages, colocated with COLING 2025 and set to be published into ACL Anthology

- **What's New**: 본 연구는 Marathi와 같은 자원이 부족한 언어에 대한 표절(plagiarism) 탐지 시스템을 설계하는 데 중점을 두었습니다. 기존의 모델들이 주로 나쁜 성능을 보여온 저자원 언어에서 BERT(Bidirectional Encoder Representations from Transformers)를 활용한 새로운 방법론을 제시합니다.

- **Technical Details**: 이 연구에서는 BERT의 문장 임베딩(sentence embeddings)과 Term Frequency-Inverse Document Frequency (TF-IDF) 기능 표현을 결합하여 Marathi 텍스트의 표절 탐지 정확성을 향상시킵니다. 이러한 접근은 기계 학습 모델의 가중치 투표 앙상블을 통해 텍스트의 통계, 의미 및 구문적 요소를 효과적으로 캡처합니다.

- **Performance Highlights**: 제안된 방법론은 기존 표절 탐지 시스템보다 더 높은 정확성을 보여주며, Marathi와 같은 저자원 언어의 표절 탐지 기술 발전에 기여할 것으로 기대됩니다. 이는 다양한 언어 처리(application) 분야에서 활용될 수 있는 중요한 연구 결과입니다.



### Optimizing Estonian TV Subtitles with Semi-supervised Learning and LLMs (https://arxiv.org/abs/2501.05234)
- **What's New**: 이 논문은 에스토니아 TV 콘텐츠의 높은 품질의 동일 언어 자막 생성 접근법을 제시합니다. Whisper 모델을 인간이 생성한 에스토니아 자막에 맞게 미세 조정하고, 반복적인 pseudo-labeling과 대형 언어 모델(LLM) 기반 후편집을 통해 품질을 향상시켰습니다. 실험 결과, 무언급 데이터셋을 통한 pseudo-labeling에서 자막 품질의 유의미한 개선이 있음을 보였습니다.

- **Technical Details**: 본 연구는 자동 자막 생성 시스템 개발을 위한 방법을 여러 단계로 나누어 설명합니다. 감독 데이터셋을 사용하여 Whisper large-v3 모델을 훈련시키고, 이후 비감독 데이터셋을 활용하여 반복 pseudo-labeling을 적용합니다. LLM 기반 후편집을 통해 생성된 자막의 오류를 수정하며, 이러한 방법을 테스트 시간과 훈련 시간에 각각 적용하여 성능을 평가합니다.

- **Performance Highlights**: 에스토니아 국가 TV의 감독 데이터셋에서 수집된 993개의 오디오-자막 쌍과 7128개의 비감독 오디오 기록을 사용하여 성능을 평가했습니다. 자막 품질 평가를 위해 SubER, t-BLEURT, AS-BLEURT의 세 가지 메트릭을 사용하였으며, 자막의 품질을 인간의 평가와 비교하였습니다. 결과적으로, LLM 기반의 게시 수정이 자막 품질을 크게 향상시킬 수 있음을 확인했습니다.



### Light Transport-aware Diffusion Posterior Sampling for Single-View Reconstruction of 3D Volumes (https://arxiv.org/abs/2501.05226)
- **What's New**: 이 논문에서는 구름과 같이 다수의 빛 산란 효과가 존재하는 체적 필드의 단일 시점 복원 기술을 소개합니다. 다양한 실험을 통해 이전에는 도달할 수 없었던 질량의 단일 시점 복원을 이루어냈습니다. 특히, 1,000개의 합성 시뮬레이션 체적 밀도 필드를 포함한 새로운 벤치마크 데이터셋을 활용하여 훈련된 조건 없는 확산 모델을 통해 이러한 기술이 이루어졌습니다.

- **Technical Details**: 이 연구에서는 보간 과정에서 물리 기반 미분 가능 볼륨 렌더러(Physically-based Differentiable Volume Renderer, PDVR)를 사용하여 복원하는 과정에서의 경량 수송에 대한 그라디언트를 제공합니다. 이 접근 방식은 기존의 NeRF 방식과 대조적이며, 관측 데이터와 잘 일치하는 복원을 위해 저수준 매개변수를 저항하는 분산 사전(diffusion prior)을 이용합니다. 또한, 새롭게 제안된 매개변수 확산 후방 샘플링(Diffusion Posterior Sampling, DPS) 기법을 통해 제작된 물체의 형상 중심의 사전 분포(not artistic) 데이터를 결합하여 시뮬레이션을 수행합니다.

- **Performance Highlights**: 제안된 방법은 단일 및 다중 뷰 복원, 체적 초해상도(volume super-resolution) 등 다양한 작업에서 큰 가능성을 보여줍니다. 구름과 같은 체적 필드를 높은 품질로 단일 시점에서 복원함으로써, 이전 기술들과 비교하여 시각적 일관성(spatial consistency)을 유지하면서 묘사했습니다. 이 연구는 확산 모델과 미분 가능 볼륨 렌더러의 통합이 3D 모델 복원에 있어 새로운 전환점을 제시할 수 있음을 입증합니다.



### EVA-S2PLoR: A Secure Element-wise Multiplication Meets Logistic Regression on Heterogeneous Databas (https://arxiv.org/abs/2501.05223)
- **What's New**: EVA-S2PLoR은 비선형(nonlinear) 연산의 정확성을 보장하면서도 보안성을 유지할 수 있는 새로운 2당의 로지스틱 회귀(logistic regression) 프레임워크입니다. 기존의 프레임워크들은 비선형 계산을 선형 연산으로 근사하는 방식으로 인해 상당한 정확도 손실이 발생하였으나, 본 연구에서는 안전한 원소별 곱셈 프로토콜을 도입하여 이러한 문제를 해결하였습니다. 이 프레임워크는 다차원적으로 변환된 데이터에서 몬테카를로 방법(Monte Carlo methods)을 통해 안전하고 강력한 이상 탐지 기능도 제공합니다.

- **Technical Details**: EVA-S2PLoR는 데이터 위장 기술(data disguising technology)을 사용하여 안전한 2당의 벡터 연산 프로토콜을 구현했습니다. 주요 알고리즘으로는 벡터 원소별 곱셈(S2PVEM), 곱셈을 위한 벡터 덧셈(S2PVATM), 벡터 역수(S2PVR), 시그모이드(S2PVS) 연산이 포함되어 있으며 모두 결과 검증 방법을 갖추고 있습니다. 이 모든 프로토콜은 반진실적(semi-honest) 보안 모델에서 보안성을 검증되었으며, 미니 배치 경량 하강법(MBGD)을 사용하여 로지스틱 회귀 모델의 안전한 훈련 및 예측 알고리즘을 제안합니다.

- **Performance Highlights**: EVA-S2PLoR는 기존의 여러 고급 PPML 프레임워크와 비교할 때 뛰어난 정확성과 성능을 자랑합니다. 특히, 시그모이드 함수의 정확도는 타 프레임워크에 비해 약 10배 우수한 결과를 나타내었습니다. 본 알고리즘은 전반적인 성능을 끌어올리며, 세 가지 공통 데이터 세트에서 세 가지 다른 PPML 프레임워크와의 성능 지표Comparision에서 두각을 나타냅니다.



### CoDe: Communication Delay-Tolerant Multi-Agent Collaboration via Dual Alignment of Intent and Timeliness (https://arxiv.org/abs/2501.05207)
Comments:
          AAAI 2025 Accepted

- **What's New**: 이번 논문에서는 비동기 통신(Asynchronous Communication)을 수용할 수 있는 새로운 다중 에이전트 협동 프레임워크인 Communication Delay-tolerant Multi-Agent Collaboration (CoDe)을 제안합니다. 대부분의 연구가 통신 지연 문제를 간과하고 있는 반면, 저자들은 통신 지연이 에이전트 간 협업에 미치는 부정적인 영향을 강조합니다. CoDe는 에이전트의 미래 행동 추론을 바탕으로 한 의도 표현을 통해 비동기 메시지 간의 융합 과정을 강화합니다.

- **Technical Details**: CoDe는 미래 행동 추론을 통해 메시지를 의도 형태로 학습한 후, 이 의도를 기반으로 하여 비동기 메시지를 통합하는 이중 정렬 메커니즘을 설계합니다. 첫 번째 정렬은 의도와 관련된 메시지를 우선시하며, 두 번째 정렬은 전송의 시의성을 고려하여 새로운 메시지에 중점을 둡니다. 이를 통해 CoDe는 행위 의도를 지속적으로 추출하여 오래된 메시지로부터도 다른 에이전트의 장기적인 의도를 파악할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, CoDe는 지연이 없는 상황에서도 기준 알고리즘보다 우수한 성능을 보여주었고, 고정 및 시간 가변 지연 하에서도 강인한 성능을 나타냈습니다. SMAC와 같은 자주 사용되는 MARL 벤치마크, GRF, Hallway에서 테스트한 결과, CoDe는 비동기 통신을 위한 새로운 기준을 설정하였습니다. 저자들은 이 연구가 MARL 분야에서 통신 지연 문제를 해결하는 데 있어 중요한 기여를 할 것으로 기대합니다.



### Design and Control of a Bipedal Robotic Character (https://arxiv.org/abs/2501.05204)
- **What's New**: 이번 연구에서는 예술적이고 표현력이 풍부한 동작과 강력한 동적 이동성을 통합한 새로운 이족보행 로봇을 소개합니다. 이 로봇은 캐릭터 주도적인 기계적 설계를 통해 개발되어, 사용자가 명령 신호를 통해 예술적 동작을 안정적으로 실행할 수 있도록 하는 강화 학습 기반의 제어 아키텍처를 제공합니다. 이러한 시스템은 로봇 캐릭터의 신뢰성을 제공하며, 다양한 엔터테인먼트 로봇 애플리케이션에서 인간-로봇 상호작용을 강화하는 길을 마련합니다.

- **Technical Details**: 이 연구에서는 기계적 설계와 애니메이션 간의 반복적인 프로세스를 통해 캐릭터 디자인 및 제어 워크플로우를 제시합니다. 기계 설계는 정적 심리학, 관절 모형과 процедур적 보행 생성 도구 등을 통해 이루어집니다. 그렇게 생성된 데이터는 강화 학습 문제로 정의되며, 사용자 입력에 따른 제어 정책을 훈련하여 다양한 동작을 안정적으로 실행할 수 있도록 합니다.

- **Performance Highlights**: 제안한 로봇은 부드럽고 직관적인 조작 인터페이스를 통해 표현력이 풍부한 애니메이션을 실시간으로 실행할 수 있도록 설계되었습니다. 여러 가지 애니메이션 소스를 구성하고 혼합할 수 있는 애니메이션 엔진을 통해 로봇 동작이 사용자 입력에 반응하며, 이를 통해 실시간 쇼 공연이 가능해집니다. 최종 시스템은 창의적 의도에 기반한 기계적 특성과 안정적인 동적 이동성을 결합하여 로봇 공연의 새로운 가능성을 제시합니다.



### RadioTransformer: Accurate Radio Map Construction and Coverage Prediction (https://arxiv.org/abs/2501.05190)
Comments:
          Submitted to IEEE VTC 2025 Spring

- **What's New**: 이 논문에서는 무선 네트워크의 예측 정확성을 향상시키기 위한 하이브리드 변환기-합성곱 모델(RadioTransformer)를 소개합니다. 기존의 최첨단(SOTA) 방법과는 달리, 이 모델은 변환기(transformer) 아키텍처를 활용하여 다양한 스케일의 특징을 효율적으로 추출하는 동시에, 합성곱 기반의 디코더를 통해 픽셀 수준의 정확한 이미지 복원을 제공합니다. 제안된 모델은 기존의 방법보다 계산 오버헤드를 줄이고 예측 오류를 현저히 감소시키는 데 성공하였습니다.

- **Technical Details**: RadioTransformer는 다중 스케일 변환기 기반의 인코더와 CNN 기반의 디코더로 구성됩니다. 인코더는 지형도에서 효율적으로 특징을 추출하고, 디코더는 픽셀 수준의 방사선 맵을 재구성합니다. 다중 스케일 변환기 아키텍처는 다양한 차원에서 특징을 생성하여 효과적인 특징 추출을 가능하게 하며, 이러한 특징들은 스킵 연결을 통해 CNN 디코더 블록으로 전달되어 이미지 복원 과제를 더 잘 수행할 수 있도록 합니다.

- **Performance Highlights**:  시뮬레이션 결과, 제안된 RadioTransformer 모델은 공공 방사선 맵 데이터셋에서 기존의 PMNet에 비해 30% 이상의 개선된 예측 정확도를 달성하며, RMSE가 10^{-3} 수준에 도달했습니다. 이는 제안된 모델이 복잡한 무선 전파 환경을 세밀하게 이해하는 데 강력한 능력을 가지고 있음을 나타냅니다.



### De-centering the (Traditional) User: Multistakeholder Evaluation of Recommender Systems (https://arxiv.org/abs/2501.05170)
Comments:
          Preprint submitted to Elsevier, "Re-centering the User in Recommender System Research" special issue of the International Journal of Human-Computer Studies (IJHCS)

- **What's New**: 이번 논문은 다수의 이해관계자(multi-stakeholder)를 고려한 추천 시스템(recommender system)에 대한 평가 방법론을 다룹니다. 전통적으로 소비자 관점에서만 평가되던 추천 시스템의 한계를 넘어서, 제작자(producers), 소비자(consumers) 외에도 다양한 이해관계자가 시스템에 미치는 영향을 분석합니다. 특히, 다양한 이해관계자의 목표와 가치에 대한 논의와 함께, 이들 간의 복잡한 관계를 설명합니다.

- **Technical Details**: 이 연구에서는 다수의 이해관계자의 평가를 위한 다양한 고려사항과 가치를 정의하고, 다수의 이해관계자 평가(multistakeholder evaluation)를 위한 방법론을 제안합니다. 또한, 음악 스트리밍 애플리케이션을 예로 들어, 시스템의 다양한 이해관계자들(예: 작곡가, 아티스트, 소비자)의 목표와 해당 목표를 평가하기 위한 지표(metric)에 대해 설명합니다. 가령, 인기 아티스트의 노래를 추천하는 것이 경제적 이익을 높이는 반면, 덜 알려진 아티스트의 가시성을 해칠 수 있는 상황을 제시합니다.

- **Performance Highlights**: 기존 추천 시스템 연구의 대다수가 소비자 관점만을 강조하고 있는 가운데, 이 논문은 더 넓은 이해관계자 생태계(ecosystem)를 고려한 평가의 필요성을 강조합니다. 추천 시스템 디자인, 구현 및 유지 관리 측면에서의 평가의 복잡성을 명확히 하며, 연구자들과 실무자들이 다수의 이해관계자 평가를 통합할 수 있는 방법을 제공하고자 합니다. 다양한 이해관계자 활용을 위한 실질적인 사례도 제시하여, 앞으로의 연구 방향에 대해 논의합니다.



### Constrained Optimization of Charged Particle Tracking with Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2501.05113)
- **What's New**: 이 논문은 다중 에이전트 강화 학습(MARL)을 기반으로한 입자 추적 재구성을 위한 접근 방식을 제안합니다. 기존의 연구를 바탕으로 하여, 파라미터화된 정책을 최적화하고, 읽기 프레임 내 입자 산란을 최소화하는 것을 목표로 합니다. 이 과정에서 고유한 입자 할당을 보장하는 안전 레이어를 제안하여 각 조인트 액션에 대해 선형 할당 문제를 해결합니다.

- **Technical Details**: 본 연구에서는 중앙 집중식 비평자 구조를 사용하여 분산 에이전트를 훈련하는 방법을 도입하고, 각 에이전트는 로컬 관찰만을 기반으로 작동하는 부분 관찰 가능 마르코프 결정 프로세스(Dec-POMDP)를 사용하여 다중 층의 입자 추적을 진행합니다. 이러한 과정은 에이전트의 예측과 결정 경계 간의 비용 여유를 증가시켜 훈련 및 일반화 성능을 향상시킵니다. 안전성을 보장하기 위해 안전 레이어를 구축하고 정책의 예측을 최적화합니다.

- **Performance Highlights**: 시뮬레이션 데이터에 대한 성능 실험을 통해 제안된 방법의 효과성을 입증하였습니다. 기존의 단일 및 다중 에이전트 기반 솔루션과 비교하여 우수한 성능을 보였으며, 비용 여유 제약이 최적화와 일반화에 효과적임을 보여주었습니다. 이러한 결과는 강화 학습 기반 추적 알고리즘의 향후 발전 가능성을 제시하며, 제약이 있는 정책과 유연성을 통한 추적 알고리즘의 최적화 가능성을 높입니다.



### Robust Score Matching (https://arxiv.org/abs/2501.05105)
- **What's New**: 본 논문에서는 Hyvärinen(2005)에서 제안된 score matching을 개선하여 데이터가 오염된 경우에도 일관된 파라미터 추정을 제공하는 robust score matching 기법을 제시합니다. 이 방법은 geometric median of means(GMoM)를 활용하여 파라미터 추정의 견고성을 높이며, 특히 비가우시안(exponential family) 그래픽 모델에 적합합니다. 따라서, 기존의 방법보다 계산적으로 부담이 적고, 오염된 데이터에서도 높은 성능을 유지합니다.

- **Technical Details**: 논문에서 제안된 robust score matching procedure는 quadratic empirical loss function을 기하학적 평균(median)으로 견고하게 수정한 것입니다. GMoM은 평균과 기하학적 평균 사이를 보간하여 매개변수를 조정할 수 있으며, 블록 크기(block-size) 파라미터를 통해 오염 수준에 맞춰 조정할 수 있습니다. 이 방법은 특히 exponential family 모델의 추정에서 나오는 문제를 해결할 수 있도록 설계되었습니다.

- **Performance Highlights**: 제안된 방법은 오염이 없을 때 표준 score matching 추정기와 유사한 성능을 보이며, 오염이 있는 경우에는 더 뛰어난 성능을 발휘합니다. 다양한 실험을 통해 이 새로운 기법의 유효성을 입증하였고, 특히, 알프스 지역의 강수량 데이터에서도 효과적으로 적용됨을 보여주었습니다. 이로 인해 회복(Recovery) 보장 및 잡음에 대한 견고성이 확인되었습니다.



### A 1Mb mixed-precision quantized encoder for image classification and patch-based compression (https://arxiv.org/abs/2501.05097)
Comments:
          Published at IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)

- **What's New**: 본 논문에서는 이미지 처리에 전념하는 ASIC 신경망 가속기가 이미지 분류 및 압축과 같은 다양한 작업에 적용될 수 있음을 보여줍니다. 자원 소모를 최소화하면서 하드웨어 성능을 높이기 위해 Mixed-Precision (혼합 정밀도) 인코더를 도입하였습니다. 또한, Linear Symmetric Quantizer의 스케일링 팩터를 자동으로 조정해 가중치 훈련을 안정화하는 방법을 제시합니다.

- **Technical Details**: 제안된 인코더는 reconfigurability를 갖추고 있으며, 3비트, 2비트, 1비트의 Mixed-Precision 양자화를 통해 효율적인 하드웨어 설계를 가능하게 합니다. 기존 Batch Normalization을 간소화하기 위해 Layer-Shared Bit-Shift Normalization 기법을 도입하였습니다. 이와 함께, 높은 분해능을 유지하면서 이미지 압축을 가능하게 하는 새로운 회복 네트워크(PURENET)를 소개합니다.

- **Performance Highlights**: 주어진 설정에서 인코더 디자인은 1MB 메모리만을 요구하며, CIFAR-10 데이터셋에 대해 87.5%의 분류 정확도를 달성하였습니다. 양자화된 인코더를 사용한 이미지 압축 과정에서 블록 아티팩트가 거의 없이 원격 복원이 가능합니다. 이러한 결과는 기존의 패치 기반 압축 기법보다 우수한 성능을 보여주며, 저비트 전송에서도 효과적입니다.



### Supervised Learning with Evolving Tasks and Performance Guarantees (https://arxiv.org/abs/2501.05089)
Comments:
          arXiv admin note: text overlap with arXiv:2310.15974

- **What's New**: 이 논문에서는 여러 감독 학습 시나리오에 적용 가능한 학습 방법론을 제안하고, 변화하는 작업에 적응하면서 성능 보장을 제공하는 방법을 설명합니다. 특히, 최근 작업 간의 다차원적 변화를 고려하여 효과적인 샘플 크기(Essential Sample Size, ESS)를 증가시킬 수 있는 방안을 제시합니다. 기존 기술들과 달리 제안된 방법론은 계산 가능한 타이트한 성능 보장을 제공합니다.

- **Technical Details**: 본 논문은 미니맥스 위험 분류기(minimax risk classifiers)에 기반한 학습 방법론을 구축하여, 다중 출처 도메인 적응(multi-source domain adaptation, MDA), 개념 변화에 대한 감독 분류(supervised classification under concept drift, SCD), 다중 작업 학습(multi-task learning, MTL) 및 지속적 학습(continual learning, CL)과 같은 시나리오에 적용될 수 있습니다. 이 방법론은 변화하는 작업의 다차원적 특성을 추정하여 적응하는 학습 기법을 개발합니다.

- **Performance Highlights**: 제안된 방법론은 여러 시나리오에서 샘플 크기가 줄어든 경우에도 성능 개선을 정량적으로 보여줍니다. 실험 결과, 제안된 방법론은 다양한 작업에서 우수한 성능을 발휘하며, 제공된 성능 보장의 신뢰성을 평가합니다. 논문의 제안된 기법은 다수의 표준 데이터셋에서 실험되었으며, 그 이점이 입증되었습니다.



### Enhanced Quantile Regression with Spiking Neural Networks for Long-Term System Health Prognostics (https://arxiv.org/abs/2501.05087)
- **What's New**: 이 논문은 산업 로봇의 시스템 고장을 예측하기 위한 새로운 예측 유지보수 프레임워크인 Enhanced Quantile Regression Neural Networks (EQRNNs)를 소개합니다. 고장 탐지를 위해 고급 신경망 아키텍처를 결합한 하이브리드 접근법을 사용하여, 다중 센서 데이터 스트림을 처리하고 마이크로초 수준의 응답 시간을 가진 Spiking Neural Network (SNN)와 통합되었습니다. 이 프레임워크는 구성 요소 고장 예측에서 92.3%의 높은 정확도를 달성하며, 90시간의 조기 경고 시간을 제공합니다.

- **Technical Details**: EQRNN은 9개의 첨단 로봇 시스템에서 배포되어 성능, 적응성 및 확장성을 평가하는 견고한 테스트베드를 제공합니다. 이러한 시스템은 LiDAR, RADAR, 카메라와 같은 다양한 센서 신호를 통해 70개의 센서 데이터를 캡처합니다. 데이터는 정상과 비정상 상태의 샘플로 나뉘며, 전체 로봇 시스템에서 32억 개 이상의 데이터 포인트가 수집되었습니다.

- **Performance Highlights**: 현장 테스트 결과, EQRNN 프레임워크는 예상치 못한 시스템 고장을 94% 감소시키고, 유지보수 관련 다운타임을 76% 줄이는 성과를 보였습니다. 이러한 결과는 복잡한 다중 모드 센서 데이터를 처리하면서도 계산 효율성을 유지하는 프레임워크의 효과를 입증합니다. 이는 Industry 4.0 제조 환경에서 높은 실용성을 나타냅니다.



### End-to-End Deep Learning for Interior Tomography with Low-Dose X-ray C (https://arxiv.org/abs/2501.05085)
Comments:
          Published by Physics in Medicine & Biology (2022.5)

- **What's New**: 이 논문에서는 기존 이미지 도메인 기반의 중첩된 저선량 ROI CT 문제를 해결하기 위한 새로운 방법을 제안합니다. 연구자들은 이미지 도메인에서의 노이즈 감소와 투영 외삽 문제를 두 개의 하위 문제로 나누어 해결합니다. 제안된 방법은 dual-domain CNNs를 사용하여 두 가지 문제를 동시에 해결함으로써 기존의 이미지 도메인 딥러닝 방법보다 우수한 성능을 보입니다.

- **Technical Details**: 저자들은 깊이 있는 convolutional framelets 이론을 기반으로 한 새로운 end-to-end 딥러닝 방법을 제안하며, 이는 투영 도메인에서의 CNN을 활용하여 cupping artifacts와 이미지 노이즈의 혼합을 제거합니다. 표준 CNN 아키텍처는 두 가지 결합된 아티팩트를 해결하는 데 어려움을 겪기 때문에, 새로운 투영 도메인 CNN과 그 이후의 이미지 도메인 CNN이 결합된 형태로 설계됩니다. 이는 넷워크 구조가 두 개의 주요 CNN으로 구성된다는 것을 의미합니다.

- **Performance Highlights**: 제안된 방법은 기존의 이미지 도메인 기반의 딥러닝 방법보다 뛰어난 성능을 보이며, 특히 투영 도메인 CNN은 일반적으로 사용되는 이미지 도메인 CNN보다 더 나은 결과를 나타냅니다. 실험 결과, 제안된 모델이 이미지 품질과 재구성 시간을 모두 개선했다는 것이 입증되었습니다. 이로 인해 저선량 CT와 ROI CT 문제에 효과적으로 접근할 수 있는 가능성이 열렸습니다.



### Comparison of Feature Learning Methods for Metadata Extraction from PDF Scholarly Documents (https://arxiv.org/abs/2501.05082)
- **What's New**: 본 논문에서는 과학 문서의 메타데이터 추출을 위한 다양한 방법론을 제시하고 있으며, 특히 템플릿 변동성이 큰 문서에 대한 접근성을 높이기 위해 자연어 처리(NLP), 컴퓨터 비전(CV), 다중 모달(multi-modal) 방법론을 활용한 연구를 수행하고 있습니다. 이 연구는 데이터의 정확성과 효율성을 높이기 위한 실험 결과를 제시하며, 향후 연구에 대한 가치 있는 통찰력을 제공합니다.

- **Technical Details**: 메타데이터 추출에 있어 여러 가지 접근 방식을 비교하고, 고전적 방법인 Conditional Random Fields와 고급 NLP 기법인 BiLSTM과 BERT 표현을 함께 사용합니다. 또한, Generative LLMs와 같은 텍스트 생성을 위한 모델들이 구조적 작업에 적합하지 않다는 점을 강조하고, 그 대신 BERT와 같은 아키텍처를 통해 과학 문서의 독특한 레이아웃 가변성과 다중 모달 내용을 효율적으로 처리합니다.

- **Performance Highlights**: 본 연구는 SSOAR-MVD와 S-PMRD라는 두 가지 데이터셋을 만들어 메타데이터 추출 성능을 평가하였습니다. 실험을 통해 여러 기법의 정확성과 효과를 비교하며, 모든 접근 방식의 구현을 공개하여 재현성과 향후 연구 개발을 돕고자 합니다.



### TipSegNet: Fingertip Segmentation in Contactless Fingerprint Imaging (https://arxiv.org/abs/2501.05076)
- **What's New**: 이 논문에서는 TipSegNet이라는 새로운 딥러닝 모델을 소개하여, 그레이스케일 손 이미지에서 직접 손가락 끝을 분할하는 최신 기술을 구현합니다. TipSegNet은 강력한 특징 추출을 위해 ResNeXt-101 백본을 활용하며, 멀티스케일 표현을 위한 Feature Pyramid Network (FPN)을 결합하여 각기 다른 손가락 포즈와 이미지 품질에서도 정확한 분할을 가능하게 합니다. 또한, 모델의 일반화 능력과 견고성을 강화하기 위해 광범위한 데이터 증강 전략을 사용합니다.

- **Technical Details**: TipSegNet은 손 이미지에서 손가락 끝을 직접 추출하는 방식을 채택하여 이전의 일회용 손가락 분할 기법과 차별화됩니다. 입력 이미지의 크기를 표준화하여 일관된 가중치 행렬 차원과 안정적인 학습을 보장하며, 주로 224×224 픽셀 크기로 조정됩니다. 훈련 데이터에는 다양한 증강 기법이 50% 확률로 적용되며, 이미지 크기 변경, 회전 및 원근 변화와 같은 변환이 무작위로 수행됩니다.

- **Performance Highlights**: TipSegNet 모델은 평균 Intersection over Union (mIoU) 0.987 및 정확도 0.999로 기존 방법들보다 우수한 성능을 보입니다. 이러한 향상된 정확성은 현실 세계의 비접촉 생체 인식 시스템에서 신뢰성과 효율성을 크게 향상시킬 가능성이 있습니다. 논문에서 제안한 방법은 여러 사용 사례에서 사용자 편의성과 안전을 극대화하며, 특히 팬데믹 이후 더욱 중요해진 위생적인 biometric 솔루션에 부합합니다.



### A Text-Based Knowledge-Embedded Soft Sensing Modeling Approach for General Industrial Process Tasks Based on Large Language Mod (https://arxiv.org/abs/2501.05075)
- **What's New**: 본 논문에서는 LLM-TKESS(대규모 언어 모델 기반 텍스트 지식 내장 소프트 센서)라는 일반 프레임워크를 제안합니다. 이는 딥러닝에 기반한 데이터 주도 소프트 센서(DDSS)의 한계를 극복하고, LLM의 문제 해결 능력, 크로스 모달 지식 전이 능력, 그리고 적은 샘플 학습 능력을 활용하여 성능을 개선하려는 시도입니다. 새로운 자연어 입력 모달리티를 통합한 두 가지 텍스트 기반 소프트 센서를 개발하여 단순 구조화된 데이터 모델의 한계를 극복하고자 합니다.

- **Technical Details**: LLM-TKESS는 보조 변수 시퀀스 인코더(AVS Encoder)를 통해 LLM이 연속적인 시계열 데이터 내의 시간적 관계와 보조 변수 간의 공간적 의미 관계를 이해할 수 있도록 설계되었습니다. 또한, 두 단계의 미세 조정(alignment) 전략을 채택하여, 프로세스 변수 데이터에 적응한 소프트 센싱 기반 모델(SSFM)을 초속도로 구축할 수 있게 하고, 후속 작업에 대해 아키텍처를 수정하지 않고도 적응할 수 있는 어댑터를 도입합니다.

- **Performance Highlights**: 본 모델은 공기 예열기 로터의 열 변형을 사례 연구로 사용하여 광범위한 실험을 통해 탁월한 예측 성능을 보였습니다. 특히, 적은 샘플 상황에서도 뛰어난 예측 능력을 보여 DDSS의 기존 한계를 넘어서는 성과를 나타냈습니다. LLM-TKESS는 이전의 DDSS에 비해 데이터의 강력한 표현 학습과 효율적인 처리 능력을 통해 산업 프로세스의 실시간 모니터링과 최적화의 달성을 가능하게 합니다.



### D3RM: A Discrete Denoising Diffusion Refinement Model for Piano Transcription (https://arxiv.org/abs/2501.05068)
Comments:
          Accepted to ICASSP 2025

- **What's New**: 이 논문에서는 피아노 악보의 자동 변환을 위한 새로운 구조의 이산 확산 모델(discrete diffusion model) 아키텍처를 제안합니다. Neighborhood Attention 층을 디노이징 모듈로 사용하는 접근 방식을 통해 고해상도 피아노 롤을 점진적으로 예측하는 연구가 진행되었습니다. 또한, 이산 확산 모델의 훈련 및 추론 단계에서 적용되는 독특한 전이 상태를 이용하여 정제(refinement) 능력을 강화하는 새로운 전략도 소개되었습니다.

- **Technical Details**: 제안된 모델은 효과적으로 이전에 예측된 상태를 정제하여 개별 음의 최종 예측이 다른 노트의 예측을 더 잘 반영하도록 설계되었습니다. 또한, 피아노 롤의 각 픽셀을 통해 이웃 노트를 고려할 수 있도록 하는 Neighborhood Attention 메커니즘이 도입되어 디노이징 과정에서 iterative refinement가 가능합니다. 모델 아키텍처는 pitchwise bidirectional LSTM과 NA 2D self-attention 블록으로 구성되어 있으며, 고해상도 피아노 롤 예측에 최적화되었습니다.

- **Performance Highlights**: MAESTRO 데이터셋에서의 실험 결과, 제안된 접근 방식이 이전의 확산 기반 피아노 변환 모델 및 기반 모델에 비해 F1 점수에서 우수한 성능을 보였습니다. 본 연구는 기존의 피드포워드 모델의 한계를 극복하고, 디지털 음악 변환 분야의 최신 기술적 발전에 기여하고 있습니다.



### Simultaneous emulation and downscaling with physically-consistent deep learning-based regional ocean emulators (https://arxiv.org/abs/2501.05058)
- **What's New**: 이번 연구는 AI 기반의 해양 시뮬레이션과 다운스케일링 프레임워크를 제안하여 멕시코만의 고해상도 지역 해양 모델링을 목표로 하고 있습니다. 기존의 대기 시뮬레이션의 성공을 바탕으로 복잡한 수심 및 경계 조건을 고려한 새로운 모델을 개발하였으며, 이를 통해 공간 해상도를 8Km에서 4Km로 향상시킬 수 있었습니다. 이 프레임워크는 예측의 안정성을 유지하면서도 물리적으로 일관된 결과를 도출할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 연구진은 해양 표면 변수의 자가 회귀 예측을 위해 Fourier Neural Operators (FNO)와 같은 고급 기계 학습 기술을 활용하였습니다. 특히, 해수면 높이(SSH), 속도(SSU), 운동 에너지(SSKE)를 포함한 딥 러닝 모델을 이용해 멕시코만의 해양 동역학을 통합하는 방법론을 개발하였으며, 두 가지 서로 다른 신경망 아키텍처를 비교 분석하였습니다. 모델의 성능 측정을 위해 저해상도(GLORYS) 자료로부터 고해상도(CNAPS) 데이터로 다운스케일링하는 과정에서 물리적 일관성을 고려한 구조도 적용하였습니다.

- **Performance Highlights**: 연구 결과, 제안된 모델은 해양 예측의 효율성을 크게 향상 시키며, 단기 및 장기 통계에서도 정확성이 뛰어난 것으로 나타났습니다. 특히, 낮은 해상도의 입력 변수에 비해 수천 배 빠른 예측 속도를 보였으며, 데이터 기반의 기술이 과학적 예측에 신뢰성을 제공할 수 있다는 점을 확인하였습니다. 이러한 접근법은 빠른 지역적 분석과 통계적 예측 생성에 유용할 것으로 기대됩니다.



### LearningFlow: Automated Policy Learning Workflow for Urban Driving with Large Language Models (https://arxiv.org/abs/2501.05057)
- **What's New**: 이 연구는 복잡한 자율 주행 작업을 위한 새로운 자동화된 정책 학습 프레임워크인 LearningFlow를 제안합니다. LearningFlow는 여러 대형 언어 모델(LLM) 에이전트의 협력적 구성 요소를 활용하여 교육 커리큘럼과 보상 함수를 자동으로 설계합니다. 이러한 새로운 접근 방식은 전통적인 보상 기능 설계 방법의 수작업 의존도를 크게 줄입니다.

- **Technical Details**: LearningFlow는 커리큘럼 강화 학습(CRL) 및 LLM을 통합하여 도시 주행 시나리오에 맞춘 정책 훈련 프레임워크를 설정합니다. 이 프레임워크는 훈련 과정 전반에 걸쳐 훈련 진전을 평가하고 중요한 통찰력을 제공하는 분석 에이전트에 의해 지원되는 커리큘럼 생성 프로세스와 보상 생성 프로세스를 포함합니다. 이러한 구성 요소는 상호작용 인식 의사결정 및 계획 능력을 향상시키기 위해 협력적으로 작용합니다.

- **Performance Highlights**: 고급 시뮬레이터 CARLA를 통해 수행된 종합적인 실험 결과, LearningFlow는 다양한 주행 작업 및 RL 알고리즘에서 뛰어난 성능을 보였습니다. 커리큘럼 및 보상 생성을 효과적으로 수행하며, 다양한 주행 작업에 대한 우수한 일반화 능력과 다양한 RL 알고리즘에 대한 적응력을 성공적으로 입증하였습니다.



### LongViTU: Instruction Tuning for Long-Form Video Understanding (https://arxiv.org/abs/2501.05037)
- **What's New**: LongViTU는 대규모의 자동 생성된 데이터셋으로, 약 121,000개의 QA 쌍과 900시간에 달하는 비디오를 포함하고 있습니다. 이 데이터셋은 비디오 이해를 향상시키기 위해 계층적 트리 구조로 조직되어 있으며, 자가 수정 메커니즘을 통해 높은 품질의 QA 쌍을 보장합니다. LongViTU는 장기적인 맥락, 풍부한 지식 및 명확한 타임스탬프 라벨을 특징으로 합니다.

- **Technical Details**: LongViTU는 현재 기존 데이터셋의 제한을 극복하기 위해 다양한 현실 세계 시나리오 및 명시적 타임스탬프 레이블을 제공하여 비디오 컨텐츠를 계층적으로 구조화합니다. 특히, 평균 증명 길이가 4.6분으로 긴 QA 쌍을 생성하며, 세심한 공간적 및 시간적 세부 사항을 유지합니다. QA 생성이 가능하도록 다양한 수준에서 비디오를 분류하는 파이프라인을 마련하였습니다.

- **Performance Highlights**: LongViTU의 품질은 상태-of-the-art 모델인 LongVU와 상업적 모델인 Gemini-1.5-Pro의 평가 결과에서도 드러납니다. 이들은 각각 GPT-4 점수 49.9 및 52.3을 기록했으며, 추가적인 지도 초미세 조정(Supervised Fine-Tuning, SFT) 후 LongVU는 12%의 성능 향상을 이뤄냈습니다. 이러한 결과는 LongViTU가 높은 데이터 품질과 강력한 OOD(Out-of-Distribution) 일반화를 제공한다는 것을 실증적으로 증명합니다.



### Towards Fingerprint Mosaicking Artifact Detection: A Self-Supervised Deep Learning Approach (https://arxiv.org/abs/2501.05034)
- **What's New**: 본 논문에서는 깊이 있는 학습 방법을 활용하여 지문 이미지 내의 모자이크 아티팩트를 탐지하고 평가하는 새로운 접근법을 제안합니다. 이 방법은 대규모 비지도 학습 데이터셋을 통해 훈련하며, 수동 아티팩트 주석이 필요 없습니다. 제안된 모델은 접촉식, 롤러식 및 압착식 등 다양한 지문 모드에서 높은 정확도로 모자이크 오류를 식별합니다.

- **Technical Details**: 지문 모자이크화 과정은 여러 지문 이미지를 결합하여 완전한 마스터 지문 이미지를 생성하는 것을 목표로 합니다. 두 가지 주요 방법인 접촉식 및 비접촉식 방법으로 지문을 수집하며, 특히 롤러식을 사용할 때 여러 개의 부분 이미지를 스티칭하는 과정에서 모자이크 아티팩트를 관리하는 것이 중요합니다. 연구는 '소프트' 및 '하드' 모자이크 오류를 분류하고, 깊이 있는 학습 모델을 사용하여 하드 모자이크 오류를 탐지하기 위한 새로운 접근법을 제시합니다.

- **Performance Highlights**: 본 논문에서 제안한 모델은 다양한 데이터 소스에 대해 강인성을 입증하며, 아티팩트의 심각성을 수치화하기 위해 새로운 모자이크 아티팩트 점수를 도입합니다. 이로 인해 지문 이미지의 자동화된 평가가 가능해지며, 아티팩트 제거의 영향을 평가하기 위해 적절한 Equal Error Rates (EER)를 계산하였습니다. 이러한 방법은 지문 기반 생체 인식 시스템의 정확성과 신뢰성을 향상시키는 데 기여합니다.



### ECBench: Can Multi-modal Foundation Models Understand the Egocentric World? A Holistic Embodied Cognition Benchmark (https://arxiv.org/abs/2501.05031)
- **What's New**: 본 논문에서는 LVLMs의 체화된 인지 능력을 체계적으로 평가하기 위한 새로운 벤치마크인 ECBench를 소개합니다. ECBench는 다양한 장면 비디오 출처와 개방적 질문 형식을 포함하며, 30가지의 인지 차원을 평가합니다. 이를 통해 현재 LVLMs의 평가 체계에서 나타나는 주요 한계점을 극복하고자 하였습니다.

- **Technical Details**: EBench는 정적 장면, 동적 장면 및 환각 문제를 포함하는 세 가지 세트를 다루며, 로봇 중심 인지 질문을 도입하였습니다. 또한, ECEval이라는 새로운 평가 시스템을 통해 질문 응답 작업에서 정밀한 평가를 가능하게 합니다. 데이터 수집 과정에서 다양한 형태의 질문 형식을 유연하게 사용하여 인지 요구 사항을 충족하도록 설계되었습니다.

- **Performance Highlights**: LVLM들을 대상으로 실시한 평가 결과, 현재의 주류 LVLM들은 동적 장면과 체화된 환각 문제에서 낮은 성능을 보였고, 정적 장면에서는 로봇 중심 질문이 더 어려움을 겪었습니다. 이는 현대의 LVLM들이 정적 장면에서는 3인칭 인지에만 해당하지만, 동적 장면에서는 1인칭 이해를 얻기 위해 고전하고 있음을 시사합니다.



### UAV-VLA: Vision-Language-Action System for Large Scale Aerial Mission Generation (https://arxiv.org/abs/2501.05014)
Comments:
          HRI 2025

- **What's New**: UAV-VLA (Visual-Language-Action) 시스템은 공중 로봇과의 의사소통을 용이하게 하기 위해 설계된 도구입니다. 이 시스템은 위성 이미지를 처리하는 능력과 Visual Language Model (VLM) 및 GPT의 강력한 기능을 통합하여 사용자로 하여금 간단한 텍스트 요청을 통해 비행 경로 및 행동 계획을 생성할 수 있게 합니다. 따라서 UAV-VLA는 미션 계획 및 의사 결정 과정을 향상시킬 수 있는 풍부한 맥락 정보를 제공합니다.

- **Technical Details**: UAV-VLA 시스템은 사용자에게 영어와 같은 언어적 지시로 경로와 행동 집합을 생성하는 기능을 제공합니다. 이 시스템은 목표 추출 GPT 모듈, 객체 검색 VLM 모듈, 그리고 행동 생성 GPT 모듈의 세 가지 주요 모듈로 구성되어 있습니다. 이러한 구성은 주어진 임무에 맞추어 UAV가 자율적으로 정확한 미션 계획을 생성할 수 있도록 합니다.

- **Performance Highlights**: 고성능의 UAV-VLPA-nano-30 벤치마크를 통해 이 시스템은 22%의 경로 길이 차이와 K-최근접 이웃(KNN) 접근 방식을 이용하여 관심 객체를 찾는 평균 오차를 34.22m 로 나타내며 성능을 입증하였습니다. 실험을 통해 UAV-VLA 시스템은 인간 수준의 경로 및 행동 생성 성능을 보였으며 이는 인간 운영자와의 비교에서 동등한 결과로 나타났습니다.



### Quantum-enhanced causal discovery for a small number of samples (https://arxiv.org/abs/2501.05007)
Comments:
          19 pages, 8 figures

- **What's New**: 이번 연구에서는 비모델 구조 가정 없이 인과관계를 발견할 수 있는 새로운 양자 Peter-Clark (qPC) 알고리즘을 제안합니다. 이 알고리즘은 양자 회로에 의해 특징 지어진 재현 커널 힐베르트 공간에서의 조건독립성 테스트를 기반으로 하여, 임의 분포에서 관측된 데이터로부터 인과관계를 탐색할 수 있습니다. 또한, Kernel Target Alignment (KTA)를 기반으로 한 하이퍼파라미터 최적화 접근법을 통해 인과 발견의 신뢰성을 높이는 방법을 제시합니다.

- **Technical Details**: qPC 알고리즘은 조건 독립성 테스트를 통해 데이터를 사용하는 PC 알고리즘의 확장으로, 데이터가 양자 상태로 임베딩된 양자 커널을 활용하여 비선형성과 고차원의 데이터를 처리합니다. 이 과정은 커널 기반 조건 독립성 테스트(KCIT)를 통한 인과 관계를 추론하는데 필요한 조건 독립성 검사 기능을 포함합니다. qPC 알고리즘은 비모수적 방법으로, 두 번의 단계에서 독립성과 인과 관계를 분별해 내며, 결과적으로 CPDAGs를 출력하여 인과 관계를 캡처합니다.

- **Performance Highlights**: 실험 결과, qPC 알고리즘은 특히 작은 샘플 크기에서 기존의 고전적인 방법보다 뛰어난 성능을 보였습니다. 이 알고리즘은 매사추세츠주 보스턴 주택 가격 데이터셋을 활용한 실제 적용에서도 효과성을 입증하였으며, 이를 통해 qPC 알고리즘이 고전적 방법의 제한성을 대체하는 새로운 가능성을 보여줍니다. 또한, 제시된 KTA 기반의 최적화 방법은 적절한 커널 선택을 통해 인과 발견의 정확도를 높이고, 가짜 긍정률을 줄이는 효과를 가져왔습니다.



### CuRLA: Curriculum Learning Based Deep Reinforcement Learning for Autonomous Driving (https://arxiv.org/abs/2501.04982)
Comments:
          To be published in the 17th International Conference on Agents and Artificial Intelligence (ICAART), Feb 2025

- **What's New**: 이 연구는 딥 강화 학습(Deep Reinforcement Learning, DRL)으로 자율주행을 개선하는 새로운 방법을 제안합니다. 특히 Curriculum Learning 기법을 적용하여 환경의 난이도를 점진적으로 증가시키고 보상 함수에 충돌 패널티를 포함시켰습니다. 이 접근법은 복잡한 환경에서 에이전트의 적응성과 신뢰성을 향상시킵니다. 이 연구는 CARLA 시뮬레이터에서 안전한 주행을 학습하는 PAX와 변이 오토인코더(Variational Autoencoder, VAE)를 이용하였습니다.

- **Technical Details**: 이 연구에서는 Proximal Policy Optimization(PPO) 알고리즘과 Curriculum Learning을 결합하여 자율주행 자동차 학습을 수행합니다. CARLA 시뮬레이터에서 두 가지 Curriculum Learning을 통해 에이전트가 초기에는 간단한 작업을 수행하도록 하면서 점진적으로 난이도를 높입니다. 또한, 에이전트의 속도를 높이기 위한 보상 함수 개선이 포함되어 있으며, 이는 운전 경험의 품질을 향상시킵니다.

- **Performance Highlights**: 제안된 CuRLA(Curriculum Learning Based Reinforcement Learning for Autonomous Driving) 방법은 다양한 보상 신호를 단일 스칼라 보상 함수에서 조정하여 평균 속도를 높이는 데 기여합니다. 이러한 보상 체계에는 회전 각도, 중심 위치 및 속도 보상뿐만 아니라 충돌 패널티도 포함되어 있습니다. 요약적으로, 이 방법은 훈련 속도를 개선하고 복잡한 환경에서의 주행 안전성을 높이는 데 크게 기여하고 있습니다.



### Self-Adaptive Ising Machines for Constrained Optimization (https://arxiv.org/abs/2501.04971)
- **What's New**: 본 논문에서는 self-adaptive Ising machine (SAIM)를 제안하여 제약 조건이 있는 최적화 문제를 해결하는 방법을 새롭게 소개합니다. 기존의 방법은 에너지 패널티의 사전 조정이 필요했지만, SAIM은 Lagrange relaxation을 사용하여 자동으로 최적의 에너지 패널티를 찾는 방식을 채택합니다. 이를 통해 제약 조건을 가진 문제에서도 더 효율적으로 최적의 솔루션에 접근할 수 있습니다.

- **Technical Details**: SAIM 알고리즘은 p-bit Ising machine의 소프트웨어 에뮬레이션을 기반으로 하며, QKP(Quadratic Knapsack Problem)와 MKP(Multidimensional Knapsack Problem)와 같은 어려운 제약 최적화 문제를 벤치마킹합니다. SAIM은 초기 에너지 패널티에서 시작해 측정 후 점진적으로 이를 조정하며, 지속적인 에너지 탐색 과정을 통해 제약 최적 솔루션을 찾아냅니다. 이러한 과정은 기존의 패널티 방법보다 더 높은 정확도를 제공하면서 샘플 수를 최소 2배 이상 줄입니다.

- **Performance Highlights**: SAIM의 성능은 300변수를 가진 QKP에서 기존의 최첨단 Ising 기계보다 뛰어난 솔루션을 발견하고, 7,500배 적은 샘플로도 최적의 결과를 도출할 수 있음을 보여줍니다. 이는 SAIM의 에너지 지형 조정이 제약 조건을 가진 최적화 문제 수행 속도를 크게 향상시킴을 의미합니다. 이 연구 결과는 제약 조건 모델링의 필요성이 있는 다양한 실제 응용 분야에서도 유용한 가능성을 제시합니다.



### Demystifying Domain-adaptive Post-training for Financial LLMs (https://arxiv.org/abs/2501.04961)
- **What's New**: 본 논문에서는 FINDAP라는 새로운 프레임워크를 제안하며, 이는 금융 도메인에 특화된 대규모 언어 모델(LLM)의 도메인 적응형 후 훈련(domain-adaptive post-training)을 체계적으로 조사합니다. 본 연구는 핵심 역량을 정의하고 이를 바탕으로 평가 프레임워크를 설계함으로써, 최적의 훈련 방법을 제안하는 것을 목표로 합니다. Llama-Fin이라는 혁신적인 모델을 통해 금융 업무 전반에서 최첨단 성능을 달성합니다.

- **Technical Details**: FINDAP의 핵심은 도메인 특화 개념, 도메인 특정 작업, 추론 능력 및 지시 준수(instruction-following) 능력 등을 갖춘 LLM의 필요한 역량을 식별하는 것입니다. 이러한 역량 기반으로 평가 프레임워크를 개발하여 성과 목표를 명확히 하고, 여러 작업에서 모델의 개선을 이끌어내는 것이 주된 내용입니다. 따라서, 훈련 방법론은 연속 사전 훈련(Continual Pretraining, CPT) 및 지시 조정(Instruction Tuning, IT) 단계로 나뉘며, 이후 선호 데이터(preference data)를 사용한 새로운 훈련 레시피가 적용됩니다.

- **Performance Highlights**: Llama-Fin 모델은 70B 스케일의 대형 모델들 및 상용 모델인 GPT-4o를 포함한 모든 기준 모델들보다 우수한 성과를 보여주었습니다. 특히, 본 연구에서는 훈련 데이터와 유사한 새로운 작업에서는 물론, 훈련 중 경험하지 못한 새로운 작업에서도 경쟁력을 유지하며 높은 성능을 기록했습니다. 이러한 결과는 도메인 적응형 후 훈련의 중요성과 효과적인 훈련 레시피의 필요성을 강조합니다.



### Non-asymptotic analysis of the performance of the penalized least trimmed squares in sparse models (https://arxiv.org/abs/2501.04946)
- **What's New**: 이번 논문은 적당히 샘플 크기가 작은 경우에 대한 최소 절단 제곱(LTS) 추정기의 비대칭적 오류 경계(non-asymptotic error bounds)를 제시합니다. 기존의 연구들은 일반적으로 샘플 크기가 무한할 것이라는 가정을 두고 진행되어 왔으나, 본 연구는 당면한 문제에서 실제적인 수치를 바탕으로 LTS의 강인성을 분석합니다. 이로 인해 실제 데이터 분석에서 LTS의 활용 가능성을 더욱 높이는 기초를 마련하고 있습니다.

- **Technical Details**: LTS 추정기는 고차원 실 데이터의 희소 모델 설정에서 적용되며, 샘플 크기(n)가 인구의 특수 속성을 가진 하위 집단의 수로 제한되는 경우에서 경기합니다. 본 연구에서는 LTS가 통계적 이론에서의 잔차(r)들을 이용해 관련성을 모델링 하는 방법을 논의합니다. LTS는 잔차의 제곱의 합을 최소화하여 안정적인 추정을 가능하게 하며, 기존의 선형 회귀 가정에 대한 강력한 대안을 제시합니다.

- **Performance Highlights**: LTS는 비대칭적 에러 경계 추정을 통해 고속 계산 성능과 강력한 강인성을 보여주며, 다양한 추정 방법의 초기 추정기로서 사용될 수 있습니다. 특히 LTS는 이상치(outlier)가 포함된 경우에서도 성능을 유지하여, 통계 분석 및 기계 학습(machin learning) 분야에서 유용한 도구로 자리 잡고 있습니다. 본 연구는 이러한 강점을 수치적으로 검증하여 실질적인 응용 사례에서 LTS의 중요성을 강조하였습니다.



### From Mesh Completion to AI Designed Crown (https://arxiv.org/abs/2501.04914)
- **What's New**: 이 논문에서는 Dental Mesh Completion (DMC)이라는 새로운 end-to-end 딥러닝 접근법을 제안합니다. DMC는 포인트 클라우드(context) 정보를 기반으로 치관(mes) 메시를 생성하여 성공적인 치관 디자인을 자동화합니다. 전통적인 방법의 노동 집약적인 과정을 줄이고, 더 나아가 치료의 품질을 보장할 수 있는 가능성을 보여줍니다.

- **Technical Details**: DMC는 포인트 클라우드를 입력으로 받아 특징 벡터(feature vectors)를 추출하는 기능 추출기(feature extractor)를 사용합니다. 이후 이 특징 벡터는 변환기(transformer)에 전달되어 부족한 영역(crown)의 새로운 특징 벡터를 예측합니다. 마지막으로, 차별화된(point-to-mesh) 레이어를 통해 치관의 표면 메시를 재구성합니다. DMC는 기존의 그래프 기반 합성곱 신경망과 비교되며, 효율성과 품질 측면에서 개선된 결과를 보여줍니다.

- **Performance Highlights**: DMC의 실험 결과는 평균 0.062의 Chamfer Distance (CD) 메트릭을 달성하여 기존 방법들보다 우수한 성능을 입증합니다. 이 연구는 모든 치아 위치에 대해 치관 메시를 생성할 수 있는 최초의 end-to-end 네트워크를 제시하며, 고품질 표면 메시 생성을 위한 새로운 가능성을 열었습니다. 따라서 치과 치료 분야의 자동화와 정확도를 높이는 중요한 진전을 이룬 것으로 평가됩니다.



### Towards understanding the bias in decision trees (https://arxiv.org/abs/2501.04903)
- **What's New**: 이번 연구는 데이터가 불균형할 때 머신러닝 모델이 대부분의 클래스에 편향된다는 일반적인 믿음이 결정 트리(Decision Trees)에 대해 반드시 옳지 않음을 보여줍니다. 연구진은 결정 트리가 소수 클래스(Minority Class)에 편향될 수 있다는 최근의 시뮬레이션 연구를 기반으로 이 오해를 해소하려 합니다. 이를 통해 과거 연구들과의 간극을 조정하고자 하며, 결정 트리에 대한 새로운 관점을 제공합니다.

- **Technical Details**: 연구진은 데이터 생성 과정(Data Generating Process)을 고려하지 않은 과거의 연구들이 결정 트리의 편향(Bias)에 대한 잘못된 결론을 유도했음을 비판적으로 평가합니다. 또한, 특정한 조건 하에서 결정 트리가 퓨리티(Purity)에 맞춰 적합(Fit)되고 단 하나의 긍정적인 사례(Positive Case)만 있는 데이터셋에서 훈련될 때 소수 클래스에 편향된다는 것을 증명합니다. 이러한 발견은 결정 트리를 사용하는 여러 가지 모델에 중요한 영향을 미칩니다.

- **Performance Highlights**: 결정 트리에서 긍정적인 사례가 하나 이상일 때도 분할(Splits)이 소수 클래스에 편향되어 있음을 보여줍니다. 이러한 결과는 랜덤 포레스트(Random Forests)와 같은 인기 있는 트리 기반 모델들을 사용할 때 성능에 영향을 줄 수 있는 중요한 요소로 작용할 것입니다. 연구 결과는 머신러닝 커뮤니티가 불균형 데이터에 대한 기존의 통념을 재고하도록 할 것입니다.



### Optimality and Adaptivity of Deep Neural Features for Instrumental Variable Regression (https://arxiv.org/abs/2501.04898)
Comments:
          46 pages, 1 figure, 2 tables

- **What's New**: 이 논문은 Deep Feature Instrumental Variable (DFIV) 회귀에 대한 수렴 분석을 제공합니다. DFIV는 심층 신경망을 사용하여 데이터 적응형 특징을 학습하는 비모수적 (nonparametric) IV 회귀 접근 방식입니다. 연구자들은 DFIV 알고리즘이 Besov 공간에 위치한 목표 구조 함수에 대해 minimax 최적 학습 속도를 달성한다는 것을 증명했습니다.

- **Technical Details**: DFIV 알고리즘은 표준 비모수적 IV 가정 하에서, 도구 변수에 대한 조건부 분포의 매끄러움 (smoothness) 가정 또한 필요로 합니다. 이러한 조건은 1단계의 난이도를 조절하는 역할을 합니다. DFIV는 고정된 특징(고정된 kernel 또는 sieve) IV 방법들과 비교했을 때 두 가지 주요 장점을 보여줍니다: 저공간 동질성 (low spatial homogeneity)과 관련해서, DFIV는 최적 속도를 달성합니다.

- **Performance Highlights**: 목표 함수가 매끄럽고 갑작스러운/불연속적인 영역을 포함할 때, DFIV는 고정된 특징 방법에 비해 최적의 성능을 보여줍니다. 또한, 1단계 샘플에서의 데이터 효율성 측면에서도 DFIV는 kernel 기반의 2단계 회귀 추정기보다 우수하다는 것을 증명했습니다.



### Reach Measurement, Optimization and Frequency Capping In Targeted Online Advertising Under k-Anonymity (https://arxiv.org/abs/2501.04882)
- **What's New**: 최근 온라인 광고의 사용 증가와 브랜드 인지도 향상은 소셜 미디어의 보편성과 관련이 깊습니다. 이 논문은 사용자 프라이버시를 우선시하는 광고 솔루션으로 전환되는 과정을 다루고 있으며, $k$-anonymity 모델을 통해 도달 측정(reach measurement) 및 최적화(optimalization)에 대해 논의합니다. 전통적인 frequency capping이 사용자 프라이버시 요구 사항을 충족하지 못할 경우의 해결책을 제시합니다.

- **Technical Details**: 논문에서는 새로운 프라이버시 환경에서 도달(reach)을 어떻게 보고(report)할 수 있는지를 설명하고, 전통적인 frequency capping의 확률적 적응을 통한 probabilistic discounting 방식을 소개합니다. 각 사용자 그룹은 k𝑘k-anonymity를 기반으로 형성되어, 개별 사용자를 k-1명의 다른 사용자와 구별할 수 없게 만듭니다. 이러한 그룹에 대한 광고 시스템은 사용자 익명성을 유지하면서 광고를 최적화하는 것을 목표로 합니다.

- **Performance Highlights**: 실험 결과, 프라이버시를 도입함에 따라 성과(performance)가 유의미하게 감소하는 경향이 발견되었습니다. 그러나 이러한 개선은 광고 플랫폼이 사용자에게 더 많은 프라이버시를 제공하기 위해 필요한 비용이 제한적이라는 점에서 이점이 있습니다. 이는 광고의 개인화(personalization)와 사용자 프라이버시를 동시에 고려해야 하는 온라인 광고의 지속 가능한 발전을 위한 기여로 해석됩니다.



### Geophysical inverse problems with measurement-guided diffusion models (https://arxiv.org/abs/2501.04881)
- **What's New**: 이 논문은 결측치 및 노이즈가 있는 측정값으로부터 다양하면서도 매우 현실적인 솔루션을 생성하기 위해 역 확산 모델(diffusion model) 프로세스를 이용한 역문제를 해결하는 새로운 접근 방식을 제시합니다. 저자는 Diffusion Posterior Sampling (DPS)과 Pseudo-inverse Guided Diffusion Model (PGDM)을 비교하며, 이 두 샘플링 알고리즘의 효과를 입증하기 위한 다양한 수치 예제를 제공합니다. 특히 PGDM이 DPS보다 더 나은 성능을 보여주며, 두 방법 모두 정량적 불확실성 평가가 가능한 솔루션 생성에 기여할 수 있음을 확인했습니다.

- **Technical Details**: 저자는 역확산 과정에서 사용하는 유도 항을 한 단계의 노이즈 제거 추정값에서 얻는 DPS와, 연속 확률 분포에 따라 한 단계의 노이즈 제거 솔루션을 모델링하는 PGDM을 고찰합니다. 이 두 방법의 성공을 위해서는 역문제를 재매개변수화하여 모델이 -1과 1 사이에 위치하도록 하며, 역확산 과정을 유도하는 암묵적 prior를 학습하기 위해 사용되는 훈련 데이터셋의 선택이 중요하다고 설명합니다.

- **Performance Highlights**: 수치 예제들은 PGDM이 DPS보다 유리한 성능을 발휘하며, 제한된 추가 비용으로도 두 시나리오에서 더 나은 결과를 도출할 수 있음을 보여주었습니다. 본 연구에서 사용된 지진 간섭(seismic interpolation) 및 지진 역전환(seismic inversion)의 두 지구물리학적 역문제는 각각 다양한 조건에서 PGDM의 우수성을 입증하는 데 큰 역할을 했습니다.



### Leveraging Log Probabilities in Language Models to Forecast Future Events (https://arxiv.org/abs/2501.04880)
Comments:
          5 pages, 4 figures

- **What's New**: 본 논문에서는 데이터 기반의 의사 결정에서 미래 사건을 예측하는 것을 목표로 하며, Large Language Models (LLMs)를 활용한 혁신적인 방법을 제안합니다. 기존 연구를 바탕으로 15개의 다양한 주제에 대한 예측을 생성하고, 그 확률을 log probabilities를 기반으로 한 다단계 접근 방식으로 추정하여 Brier score 0.186을 달성했습니다. 이는 무작위 확률에 비해 +26% 개선된 성과입니다.

- **Technical Details**: 이 시스템은 예상하는 이벤트의 제목, 설명, 시간대를 생성하는 Forecast Generator, 예측의 확률과 관련 트렌드를 출력하는 Probability Estimator, 주어진 시간 내에 사건이 발생했는지 자동으로 평가하는 Fact Checker로 구성됩니다. 특히 Probability Estimator는 LLM의 모든 가능한 추측을 고려하여, 확률 P와 불확실성 U를 평가하는 데 혁신적인 접근 방식을 사용합니다. 설정에 따라 예측의 일관성을 개선하고, 상호 배타적인 사건에 대해 총 확률이 100%를 초과하지 않도록 보장합니다.

- **Performance Highlights**: 논문에서 제안한 방법은 총 240일의 간격을 두고 예측된 사건과 사실 확인을 진행하며, N=150의 예측 데이터셋으로 평가되었습니다. 우리의 시스템은 잘 알려진 AI 시스템에 비해 +19% 이상의 개선률을 보이며, 이는 LLM 기반 예측의 가능성을 보여주는 중요한 지표입니다. 이러한 결과는 LLM이 단순한 데이터 추정이 아닌, 더 깊이 있는 정보 분석 및 예측 능력을 갖추고 있음을 증명합니다.



### RieszBoost: Gradient Boosting for Riesz Regression (https://arxiv.org/abs/2501.04871)
- **What's New**: 이 논문에서는 인과 질문에 대한 응답을 제공하기 위해 조건부 기대값의 선형 함수형을 추정하는 새로운 그래디언트 부스팅 알고리즘을 제안합니다. 이 방법은 전통적인 방법들과는 달리 Riesz 대표자를 필요로 하지 않으며, 표 형식 데이터에 적합합니다. 또한 비모수적(nonparametric)이고 계산적으로 효율적인 대안을 제공합니다.

- **Technical Details**: Riesz 대표자는 doubly robust 추정 방법의 핵심 구성 요소로, 예측 성능을 높이기 위한 결과입니다. 전통적으로는 그 명시적 형태를 파생시키고 추정한 뒤, 이를 대입 방법으로 사용하는 것이 일반적입니다. 하지만 이러한 방식은 밀도 비행성(practical positivity violations)에 민감하여 결과의 분산 증가와 신뢰 구간 확대를 초래할 수 있습니다.

- **Performance Highlights**: 시뮬레이션 연구 결과, 제안된 알고리즘은 다양한 함수형에 대해 간접 추정 기법과 동등하거나 이를 초과하는 성능을 보였습니다. 이는 사용자 친화적이며 강력한 인과 양을 추정하는 솔루션을 제공하므로, 연구자들에게 매우 유용할 것으로 기대됩니다.



### Deep Transfer $Q$-Learning for Offline Non-Stationary Reinforcement Learning (https://arxiv.org/abs/2501.04870)
- **What's New**: 본 논문은 강화 학습( Reinforcement Learning, RL)의 동적 의사결정 문제에서 비정상적 유한 시간 지평선 마르코프 결정 과정(Markov Decision Process, MDP)를 모델링하여 전이 학습(Transfer Learning) 연구를 선도합니다. 특히 샘플 크기가 제한된 특정 대상 인구를 위해 샘플 궤적(sample trajectories)을 활용하여 RL 성능을 개선하는 방법론을 제시합니다. 이와 함께 기존의 단순 샘플 풀링 전략이 RL에서 어떻게 잘못된 편향을 초래하는지를 설명하며, 새로운 '재가중치 타깃 절차(re-weighted targeting procedure)'를 소개합니다.

- **Technical Details**: 논문에서는 비정상적 유한 시간 지평선 MDP에서 노이즈가 있는 샘플을 활용하는 새로운 접근법을 통해 RL의 효과적인 전이 학습을 제안합니다. '재가중치 타깃 절차'는 샘플 간의 전이 변화 및 보상 예측 불일치를 조정하는 기법으로, 백워드 귀납적 Q^*-학습(backward inductive Q*-learning)과 신경망(Neural Network) 기능 근사(function approximation)를 통해 실현됩니다. 또한, 보상 함수의 전이 가능성을 가정하며, 전이 밀도(transition densities)가 전이 가능한 경우와 불가능한 경우를 다룹니다.

- **Performance Highlights**: 본 연구는 합성 및 실제 데이터셋을 사용한 실험을 통해 제안된 방법의 이점을 입증합니다. 새로운 접근 방식인 '전이 가능한 RL 샘플을 구축하는 방법'을 통해 동적이고 비정상적인 강화 학습 컨텍스트에서의 의사결정 개선 가능성을 강조합니다. 또한, 이론적 보장을 통해 제안된 절차의 신뢰성을 뒷받침하여 다양한 응용 분야에서 강화 학습의 효과성을 높일 수 있는 잠재력을 보여줍니다.



### Intelligent experiments through real-time AI: Fast Data Processing and Autonomous Detector Control for sPHENIX and future EIC detectors (https://arxiv.org/abs/2501.04845)
Comments:
          proceedings for 42nd International Conference on High Energy Physics (ICHEP2024), 18-24 July 2024, Prague, Czech Republic

- **What's New**: 이번 연구 개발 프로젝트는 2022년 DOE 핵 물리학 AI-기계 학습 이니셔티브에 의해 시작되었으며, 고에너지 핵 실험(예: RHIC, LHC, EIC)에서 데이터 처리 문제를 해결하기 위해 AI를 활용합니다. 우리는 sPHENIX 실험에서 고속 데이터 스트림을 실시간으로 처리하기 위한 데모를 개발하는 데 주력하고 있으며, 이를 통해 중량 쿼크 이벤트를 신속하게 식별할 수 있습니다. 이 시스템은 다른 분야에도 적용 가능성이 높은 혁신적인 접근 방식을 제안합니다.

- **Technical Details**: sPHENIX 실험은 Brookhaven 국가 연구소의 RHIC 가속기에서 수행되며, 주 목표는 아름다움 쿼크 신호를 발견하는 것입니다. 이 실험의 데이터 수집 시스템은 300 Gb/s 대역폭으로 제한되어 있으며, TPC에서 생성되는 데이터의 90%를 샘플링하기 위한 복잡한 트리거 시스템을 필요로 합니다. 알고리즘 파이프라인은 여러 단계를 포함하며, 여기에는 히트 디코딩, 이벤트 구축, 트랙 구축, 엣지 후보 생성 및 분류가 포함됩니다.

- **Performance Highlights**: 모델의 성능은 D0→Kπ 붕괴 사건의 분석을 통해 평가되었으며, BGN-ST 모델은 비슷한 기초 모델들에 비해 가장 높은 정확도를 보였습니다. 개선된 모델은 정확도를 87.56%에서 90.22%로 증가시켰고, 이는 23.2%의 효율과 2.3%의 순도를 나타냅니다. 특히, 아름다움 붕괴의 탐지에서 BGN-ST 모델의 정확도는 97.38%에 달했습니다.



### Quantum Hybrid Support Vector Machines for Stress Detection in Older Adults (https://arxiv.org/abs/2501.04831)
- **What's New**: 이 논문은 고령자의 스트레스 감지를 위한 혁신적인 방법을 제시합니다. 양자 기계 학습(Quantum Machine Learning, QML)을 이용해 스트레스 감지를 이상 탐지(anomaly detection) 문제로 다루며, 양자 혼합 서포트 벡터 머신(Quantum Hybrid Support Vector Machine, QHSVM)을 사용하여 성과를 입증하고 있습니다. 실험을 통해 QML이 고전적인 방법보다 더 높은 정확도를 제공함을 보여주었습니다.

- **Technical Details**: 제안된 방법은 고전 기계에서의 특성 선택(feature selection) 및 전처리(pre-processing) 과정을 포함합니다. 60개의 특성은 Empatica E4 손목 밴드에서 수집된 전자피부활동(Electrodermal Activity, EDA), 혈류맥박(Blood Volume Pulse, BVP), 심박 간격(Inter Beat Interval, IBI), 피부 온도(Skin Temperature, ST) 신호에서 추출됩니다. 양자 커널 회로를 통해 특성 간의 동질성을 계산하고, 이를 기반으로 OneClass SVM을 훈련해 분류합니다.

- **Performance Highlights**: 실험 결과, 제안된 양자 하이브리드 모델은 고전적인 알고리즘에 비해 높은 재현율(recall value)을 보여주었습니다. 이는 중요한 이상 현상을 놓치는 것이 진단이나 치료의 지연을 초래할 수 있음을 강조의미합니다. 이 연구는 고령자 건강 관리 분야에서 QML의 잠재성을 명확히 하였으며, 실제 직면하는 문제 해결을 위한 첨단 기계 학습 접근법의 필요성을 제시합니다.



### A Steerable Deep Network for Model-Free Diffusion MRI Registration (https://arxiv.org/abs/2501.04794)
- **What's New**: 비선형 등록(nonrigid registration)은 의료 영상 분석에서 중요한 역할을 하지만, 확산 MRI(diffusion MRI)에서는 여전히 도전 과제로 남아 있습니다. 본 논문에서는 명시적인 재배향(reorientation) 없이 원시 확산 MRI 데이터의 비선형 등록을 위한 새로운 딥러닝 프레임워크(deep learning framework)를 제안합니다. 이전 방법들과는 달리, 우리는 위치 및 방향 공간에서의 등변적인 미분동형사상(equivariant diffeomorphism)으로 등록을 공식화합니다.

- **Technical Details**: 우리 방법의 핵심은 기하학적 속성을 유지하면서 속도 필드(velocity fields)를 생성하는 $	ext{SE}(3)$-등변 UNet입니다. 새로운 손실 함수(loss function)를 도입하여 Fourier 공간에서 최대 평균 불일치(maximum mean discrepancy)를 기반으로 하여, 이미지 간의 집합 평균 전파기(ensemble average propagators)를 암묵적으로 일치시킵니다. 이러한 접근 방식은 파생 표현(derived representations) 추정의 오버헤드를 우회하는 장점도 가지고 있습니다.

- **Performance Highlights**: Human Connectome Project의 확산 MRI 데이터에 대한 실험 결과는 최신 접근 방법들과 경쟁력 있는 성능을 보여주었습니다. 이는 기존의 방법들에 비해 데이터 기반(data-driven)이며 기하학적으로 인식되는 dMRI 등록의 기초를 확립하는 작업입니다. 따라서, 본 연구는 의료 영상 분석의 효율성을 높이는 데 기여할 것으로 기대됩니다.



### Leveraging Registers in Vision Transformers for Robust Adaptation (https://arxiv.org/abs/2501.04784)
Comments:
          Accepted at ICASSP 2025

- **What's New**: 본 논문에서는 Vision Transformers (ViTs)의 register 토큰임베딩을 활용한 새로운 접근 방식을 제안합니다. 이 register는 높은 노름을 가지는 패치 토큰의 영향을 줄이면서도 글로벌 이미지 정보를 포착하는 데 기여합니다. OOD(Out-of-Distribution) 상황에서 일반화 성능과 이상 탐지 성능을 개선하기 위한 새로운 방법론을 다루고 있습니다.

- **Technical Details**: ViTs는 이미지를 패치로 나누어 토큰으로 처리하며, 특수한 [CLS] 토큰을 사용하여 정보 집계를 수행합니다. 본 연구는 [CLS] 토큰과 평균 풀링한 register 토큰 임베딩을 결합하여 더 견고한 표현을 형성하는 방법을 도입하고, 다양한 ViT 아키텍처에서 검증한 결과를 보고합니다. 이를 통해 OOD 정확도를 2-4% 향상시키고, 이상 탐지 시 거짓 긍정률을 2-3% 줄이는 성과를 달성하였습니다.

- **Performance Highlights**: 우리는 비지도 객체 발견 작업에서 활용된 register의 효과를 보여주며, OOD 일반화 및 이상 탐지 성능 향상을 증명했습니다. 실험 결과, [CLS]와 register 임베딩의 결합이 기존 방법보다 더욱 향상된 성능을 발휘함을 확인했습니다. 이 방법은 추가적인 계산 비용 없이도 실질적인 성과를 내며, 다양한 ViT 백본 모델에서 일관된 개선 효과를 보여주었습니다.



### Efficient and Responsible Adaptation of Large Language Models for Robust and Equitable Top-k Recommendations (https://arxiv.org/abs/2501.04762)
Comments:
          arXiv admin note: text overlap with arXiv:2405.00824

- **What's New**: 본 논문은 기존 추천 시스템(Recommendation Systems, RSs)이 다양한 사용자 집단의 요구를 간과하는 문제를 해결하기 위해 하이브리드 작업 할당 프레임워크를 제안합니다. 이는 대규모 언어 모델(LLMs)을 통해 추천의 질을 높이고, 사회적 공익을 위한 더 공정한 사용자 서비스 제공을 목표로 합니다. 두 단계의 접근 방식을 통해, 비활성 사용자와 약한 사용자에게 초점을 맞추어 그들의 상호작용 기록을 최적화하여 더 나은 추천을 제공하는 방법을 소개합니다.

- **Technical Details**: 하이브리드 프레임워크는 약한 사용자에 대한 세분화된 상호작용 분석을 기반으로 하여, 특정 기준치 이하의 추천 성능을 지닌 사용자들을 식별합니다. 이러한 약한 사용자들은 in-context learning을 활용하여 각자의 상호작용 이력을 다루어 추천 품질을 높입니다. 이를 통해 더욱 공정하고 포괄적인 추천 시스템을 구현하는 것을 목표로 하고 있습니다.

- **Performance Highlights**: 논문에서는 열 개의 추천 알고리즘 및 세 개의 LLM을 통합한 하이브리드 프레임워크를 평가하였고, 그 결과 약한 사용자의 수를 현저히 줄이고 하위 집단에 대한 강건성을 개선하게 되었습니다. 특히, 약 12%의 강건성 향상을 보였고, LLM을 adaption하는 데 필요한 높은 비용을 효율적으로 줄이는 데 성공했습니다. 최종적으로 사용자 집단 간의 불균형 문제를 해결하여 다양한 사용자에게 양질의 추천을 제공할 수 있는 가능성을 여는 연구 결과를 확인했습니다.



### Efficient License Plate Recognition in Videos Using Visual Rhythm and Accumulative Line Analysis (https://arxiv.org/abs/2501.04750)
Comments:
          Accepted for presentation at the Conference on Graphics, Patterns and Images (SIBGRAPI) 2024

- **What's New**: 본 논문은 비디오 기반의 자동 번호판 인식(Automatic License Plate Recognition, ALPR) 시스템의 효율성을 크게 향상시키는 두 가지 새로운 방법을 제안합니다. 전통적인 시스템은 여러 프레임에 의존하여 높은 컴퓨팅 자원을 필요로 하지만, 제안된 방법은 차량당 하나의 프레임만을 사용하여 번호판 문자를 인식합니다. 첫 번째 방법인 Visual Rhythm (VR)은 비디오에서 시간-공간 이미지를 생성하고, 두 번째 방법인 Accumulative Line Analysis (ALA)는 실시간 처리를 위한 신개념 알고리즘입니다.

- **Technical Details**: 제안된 방법들은 YOLO를 활용하여 프레임 내에서 번호판을 탐지하고, Convolutional Neural Network (CNN)을 통해 광학 문자 인식(Optical Character Recognition, OCR)을 수행하여 텍스트 정보를 추출합니다. Visual Rhythm 방법은 비디오에서 여러 프레임을 단일 시간-공간 이미지로 변환하여 차량이 미리 정의된 선을 교차하는 순간을 효율적으로 포착합니다. ALA는 프레임 별로 미리 정의된 수평선에 집중하여 이 선을 가로지르는 차량을 추적하는 방식으로 동작합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법들은 전통적인 프레임별 접근 방식과 동등한 결과를 달성하며, 처리 속도가 세 배 더 빠른 것으로 나타났습니다. 이는 ALPR 시스템의 실시간 작동을 가능하게 하여 다양한 교통 상황에서의 적용성을 높입니다. 새로운 알고리즘들은 비디오 기반 ALPR 시스템의 계산적 오버헤드를 크게 줄여주며, 라이센스 플레이 인식의 효율성을 증가시킵니다.



### Generative Style Transfer for MRI Image Segmentation: A Case of Glioma Segmentation in Sub-Saharan Africa (https://arxiv.org/abs/2501.04734)
- **What's New**: 이 연구는 아프리카 아프리카 사하라 이남 지역에서의 저품질 자기공명영상(MRI) 기술을 위한 심층학습 기반 뇌 종양 분할 방법을 제시합니다. 연구에 따르면, 해당 지역의 교육 데이터를 바탕으로 도메인 시프트가 모델 성능에 미치는 영향은 크지 않으며, 3D와 2D 모델을 비교한 결과 비슷한 성능을 보이는 것으로 나타났습니다. 또한, d neural style transfer (NST) 기법을 데이터 증가 방법으로 사용하여 성능 격차를 줄이는 두 가지 접근법이 제안되었습니다.

- **Technical Details**: 연구에 사용된 모델은 PyTorch로 구현된 nnU-Net의 확장판이며, 다양한 의료 데이터에 적용 가능합니다. 데이터 전처리 과정에서 기본적으로 얼굴 부분을 잘라내고 강도 조정 및 리샘플링을 통해 다양한 데이터 품질의 일관성을 높였습니다. 최적화된 U-Net 모델을 기반으로 하여, 여러 데이터 세트에 맞춰 자동적으로 구성되는 nnU-Net을 활용하였습니다.

- **Performance Highlights**: 최종적으로, GLI 및 GLI+SSA 데이터 세트를 사용하여 모델을 훈련시켰으며, 300 에포크 동안 성능 평가를 수행한 결과, 모델은 높은 정확도로 뇌 종양 영역을 분할하였습니다. 이 연구는 SSA 지역의 의료 시스템 내에서 뇌 종양 예측 성능을 향상시키기 위한 가능성을 제시하며, 저품질 MRI에서도 효과적으로 작동할 수 있는 모델을 제안합니다.



### AI-Driven Reinvention of Hydrological Modeling for Accurate Predictions and Interpretation to Transform Earth System Modeling (https://arxiv.org/abs/2501.04733)
- **What's New**: 이번 논문에서는 전통적인 수식 기반 수문 모델과 기존 알고리즘 기반 모델의 한계를 극복한 새로운 모델인 HydroTrace를 소개합니다. 이 모델은 강력한 예측 성능을 바탕으로 98%의 Nash-Sutcliffe Efficiency를 기록하며, 새로운 데이터에 대한 일반화 능력이 뛰어납니다. 또한, HydroTrace는 고급 attention mechanism을 활용하여 시공간적 변동성과 특징별 영향을 효과적으로 포착합니다.

- **Technical Details**: HydroTrace는 데이터-불가지론적(data-agnostic) 모델로, 수문 행동의 해석이 어려운 기존 모델들과 비교해 상당한 성능 향상을 가져왔습니다. 이 모델은 빙하-눈-유량(glacier-snow-streamflow) 상호작용이나 몬순(monsoon) 역학에 대한 해석을 가능하게 합니다. 뿐만 아니라, HydroTrace의 대형 언어 모델(LLM) 기반 애플리케이션은 사용자들이 이 모델의 통찰력을 쉽게 이해하고 활용할 수 있도록 지원합니다.

- **Performance Highlights**: HydroTrace는 강력한 예측 정밀도와 해석 가능성을 제공하여 수문학(hydrology) 및 광범위한 지구 시스템 모델링에서 혁신적인 도구로 자리 잡고 있습니다. 특히, 새로운 데이터에서도 뛰어난 일반화 능력을 보여주며 실용성을 더욱 높였습니다. 이러한 특성은 다양한 수문 현상에 대한 예측 모델링을 통한 발전을 이끌어낼 것으로 기대됩니다.



### Guiding Treatment Strategies: The Role of Adjuvant Anti-Her2 Neu Therapy and Skin/Nipple Involvement in Local Recurrence-Free Survival in Breast Cancer Patients (https://arxiv.org/abs/2501.04724)
- **What's New**: 이번 연구는 관찰 환자 데이터를 통해 인구통계학적 요인, 치료, 조건 및 결과 간의 인과 관계를 추출할 수 있는 Linear Non-Gaussian Acyclic Model (LiNGAM)와 같은 인과 추론 모델을 탐구합니다. 기존의 무작위 대조 시험(RCT)과는 달리, 우리의 방법은 더 넓은 관찰 데이터셋을 활용하여 일반화 가능성을 향상시키고 치료의 중요성을 부각시킵니다. 예를 들어, 췌장암 치료에서 Adjuvant Anti-Her2 Neu Therapy는 국소 재발 없는 생존 기간을 평균 169일 증가시켰으며, 피부/유두 침범은 평균 351일 감소시키는 것으로 나타났습니다.

- **Technical Details**: 연구는 Duke MRI 유방암 데이터셋의 인구통계학적, 임상, 병리학적, 유전적, 치료, 결과 및 기타 데이터 부분을 활용합니다. 이 데이터셋은 922명의 생검으로 확인된 침습성 유방암 환자에 대한 단일 기관의 후향적 수집입니다. 전처리 단계에서는 관련 없는 특성과 중복 특성을 제거하고, 결측값을 평균/최빈값으로 대체하며, 범주형 변수를 원-핫 인코딩과 순서 인코딩을 사용하여 인코딩합니다.

- **Performance Highlights**: 이번 연구의 성과는 Duke MRI 데이터셋 내부의 관찰 데이터를 활용하여 새로운 인과 관계를 밝혀내는 데 중점을 두었습니다. Adjuvant Anti-Her2 Neu Therapy가 치료에 대한 효과적인 응답을 나타내며, 고위험 사례에 대한 맞춤형 개입의 필요성을 강조합니다. 이러한 결과는 대상 환자의 특성에 따라 치료 효과가 크게 달라질 수 있음을 보여주며, 이는 개인화된 치료 전략 개발에 중요한 단서가 됩니다.



### A Shape-Based Functional Index for Objective Assessment of Pediatric Motor Function (https://arxiv.org/abs/2501.04721)
Comments:
          13 pages

- **What's New**: 이번 연구에서는 스피널 근위축증(SMA)와 듀셴 근육형성증(DMD) 환자를 대상으로 착용형 센서를 활용하여 일상 생활 속에서의 운동 기능을 객관적으로 평가하는 새로운 방법을 제시합니다. 이 방법은 복잡한 아동의 운동 데이터를 분석하기 위해 Shape-based Principal Component Analysis (PCA)를 사용하여 운동 궤적을 정렬하고 독특한 운동 패턴을 식별합니다.

- **Technical Details**: 연구에서 사용된 방법론은 Shape-based PCA를 통해 운동 궤적을 정렬하고 kinematic 패턴을 분석하는 것을 포함합니다. 또한, 각 주성분에 대한 투영 결과를 결합하여 Partial Least Squares (PLS) 분석을 수행하여 근육 지방 침윤, Brooke 점수, 그리고 나이에 따른 퇴행성 변화를 반영하는 새로운 운동 기능 지표를 제안하였습니다.

- **Performance Highlights**: 연구 결과, DMD와 SMA 환자의 운동 기능은 건강한 대조군과 유사한 수준에서 나타났고, SMA 환자는 운동 비대칭 패턴의 활성화가 더 두드러진 것으로 확인되었습니다. 이 데이터 기반 방법은 가정에서도 활용 가능하여, 근육 신경계 질환 아동의 치료 효능을 보다 효과적으로 추적할 수 있는 기회를 제공합니다.



### Pressing Intensity: An Intuitive Measure for Pressing in Soccer (https://arxiv.org/abs/2501.04712)
- **What's New**: 이 연구에서는 축구에서의 압박 압력을 측정하기 위한 혁신적인 프레임워크를 도입했습니다. 기존 방법들과는 달리 이 방법은 플레이어의 속도, 이동 방향, 반응 시간을 통합하여 수비수가 공격자나 공을 가로채기까지의 시간을 분석합니다. 이러한 접근 방식은 현대 축구의 분석 능력을 향상시킬 수 있는 강력하고 해석 가능한 지표를 제공합니다.

- **Technical Details**: 압박 압력을 측정하기 위한 본 접근법은 Spearman의 Pitch Control 모델 및 Shaw와 Pleuler의 수정 사항을 기반으로 구성됩니다. 플레이어의 예상 가로채기 시간은 플레이어의 위치 및 속도를 기반으로 하며, 반응 시간과 최대 속도를 고려하여 업데이트된 수식으로 계산됩니다. 이 연구는 수비수와 공격자 간의 가로채기에 필요한 시간을 확률값으로 변환하여 압박 상황을 더욱 동적으로 분석할 수 있도록 합니다.

- **Performance Highlights**: 연구 결과는 수비자가 공격자에 대해 얼마나 많은 압박을 가할 수 있는지를 정량적으로 표현하며, 설정된 매개변수를 통해 압박 수치가 독립적이라고 가정합니다. 이러한 새로운 지표는 코치와 분석가가 압박 전략을 식별하고, 특수 상황을 분석하는 데 도움을 주며 현대 축구의 전술적 접근 방식을 혁신할 기회를 제공합니다.




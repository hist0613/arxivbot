### IndicGenBench: A Multilingual Benchmark to Evaluate Generation  Capabilities of LLMs on Indic Languages (https://arxiv.org/abs/2404.16816)
- **What's New**: 새로운 벤치마크 'IndicGenBench'가 발표되었습니다. 이 벤치마크는 인도의 언어 다양성을 대표하며, 다양한 Indic 언어로 구성된 LLM (Large Language Models)의 성능을 평가하기 위한 것입니다. IndicGenBench는 29개 Indic 언어, 13개의 문자 체계, 4개 언어 가족을 포함하고 있습니다. 이는 인도의 언어적 다양성에 초점을 맞추며, 국제적으로도 중요한 기준이 될 수 있습니다.

- **Technical Details**: IndicGenBench는 교차 언어 요약(cross-lingual summarization), 기계 번역(machine translation), 그리고 교차 언어 질문 응답(cross-lingual question answering)과 같은 다양한 생성 작업(generation tasks)으로 구성되어 있습니다. 또한, 이 벤치마크는 처음으로 많은 소수 인도 언어들에 대한 다방면 평가 데이터를 인간 큐레이션을 통해 제공합니다. LLMs 평가에는 GPT-3.5, GPT-4, PaLM-2, mT5, Gemma, BLOOM 및 LLaMA와 같은 다양한 종류의 모델이 사용되었습니다.

- **Performance Highlights**: 벤치마크 평가 결과, 가장 큰 모델인 PaLM-2가 대부분의 작업에서 가장 우수한 성능을 보였습니다. 그러나 모든 언어에서 영어에 비해 상당한 성능 격차가 있어, 더 포괄적인 다양 언어 모델 개발을 위한 추가 연구가 필요함을 시사하고 있습니다.



### Make Your LLM Fully Utilize the Contex (https://arxiv.org/abs/2404.16811)
Comments: 19 pages, 7 figures, 3 tables, 9 examples

- **What's New**: 새로운 연구에서는 대규모의 긴 문맥(long context)을 처리할 수 있는 언어 모델(Large Language Models, LLMs)이 아직도 긴 문맥 정보의 활용에 어려움을 겪고 있다는 문제, 이른바 '중간의 소실' 문제(lost-in-the-middle challenge)에 초점을 맞추었습니다. 이를 해결하기 위해 '정보 집약적 훈련'(INformation Intensive, IN2 training) 방법을 제안했으며, 이는 전적으로 데이터 기반의 해결책으로 긴 문맥에서 중요한 정보를 놓치지 않도록 돕습니다.

- **Technical Details**: IN2 훈련은 긴 문맥(4K-32K 토큰) 안에서 짧은 세그먼트(~128 토큰)에 대한 세밀한 정보 인식을 요구하며, 두 개 이상의 짧은 세그먼트에서 정보를 통합하고 추론해야 하는 합성된 긴 문맥 질문-답변 데이터셋을 활용합니다. 이 방법을 Mistral-7B 모델에 적용하여 FILM-7B(FILl-in-the-Middle) 모델을 개발하였습니다.

- **Performance Highlights**: FILM-7B는 32K 토큰의 긴 문맥 창에서 다양한 위치의 정보를 견고하게 검색할 수 있는 능력을 시험하는 세 가지 탐색 과제(probing tasks)를 통해 그 성능을 입증했습니다. 문서(document), 코드(code), 구조화된 데이터(structured-data context) 등 다양한 문맥 스타일과 정보 검색 패턴(forward, backward, bi-directional retrieval)을 포괄합니다. 실제 긴 문맥 작업에서도 성능이 대폭 향상되었으며(NarrativeQA에서 23.5에서 26.9 F1 점수로 향상), 짧은 문맥 작업에서도 비슷한 수준의 성능을 유지했습니다(MMLU에서 59.3에서 59.2 정확도).



### Improving Diversity of Commonsense Generation by Large Language Models  via In-Context Learning (https://arxiv.org/abs/2404.16807)
Comments: 16 pages, 6 figures

- **What's New**: 이 논문에서는 대규모 언어 모델(Large Language Models, LLMs)의 생성물을 다양화시키면서 그 품질을 유지하는 새로운 방법을 제안합니다. Generative Commonsense Reasoning (GCR)은 상황에 대해 상식을 이용하여 추론하는 과제로, 생성된 문장의 다양성이 중요한 지표로 평가됩니다. 이는 모델이 다양한 상식 지식을 활용하는 능력을 반영하기 때문입니다.

- **Technical Details**: 제안된 방법은 in-context learning (ICL)을 통해 LLMs의 성능을 개선합니다. ICL은 주어진 예시를 사용하여 모델을 특별히 조정할 필요 없이 학습하는 기술입니다. 연구진은 GCR 작업에 대해 LLM의 출력에서 품질과 다양성의 균형을 맞춰 보다 다양한 문장을 생성할 수 있는 방법을 실험했습니다.

- **Performance Highlights**: 실험 결과는 세 가지 벤치마크 GCR 데이터셋에서 이루어졌으며, 제안된 방법이 품질과 다양성 사이의 이상적인 균형을 달성했음을 보여줍니다. 또한, 생성된 문장은 기존의 상식 생성기에서 다양성을 향상시키기 위한 훈련 데이터로 사용될 수 있습니다.



### Modeling Selective Feature Attention for Representation-based Siamese  Text Matching (https://arxiv.org/abs/2404.16776)
Comments: Accepted by IJCAI2024

- **What's New**: 새로운 기능 주의(Feature Attention, FA) 및 선택적 기능 주의(Selective Feature Attention, SFA) 블록이 소개되었습니다. 이들은 임베딩 기능 사이의 종속성을 모델링하여 Siamese 네트워크의 성능을 향상시키는 것을 목표로 합니다. FA는 'squeeze-and-excitation' 기술을 사용하여 중요한 특징에 더 많은 중점을 둡니다. SFA는 다양한 추상 수준에서 의미 정보와 임베딩 특징에 집중할 수 있도록 동적 '선택' 메커니즘을 활용합니다.

- **Technical Details**: FA 블록은 tensor 크기를 변경하지 않고, 각 특징에 대한 중요도를 동적으로 조절합니다. SFA 블록은 stacked BiGRU Inception 구조를 기반으로 하여 여러 분기에서 의미 정보를 다양한 스케일로 추출합니다. 이 블록들은 Siamese 네트워크에 쉽게 통합될 수 있으며 'plug-and-play'(플러그 앤 플레이) 특성을 제공합니다.

- **Performance Highlights**: FA 및 SFA 블록을 통합한 Siamese 네트워크는 다양한 텍스트 매칭 벤치마크에서 성능이 향상되었음을 실험을 통해 입증하였습니다. 이는 임베딩의 특징 수준에서 종속성을 모델링하는 데 탁월한 성능을 보여주며, SFA 블록의 '선택' 메커니즘은 효율적인 기울기 흐름 관리에 기여합니다.



### Prefix Text as a Yarn: Eliciting Non-English Alignment in Foundation  Language Mod (https://arxiv.org/abs/2404.16766)
- **What's New**: 이번 연구에서는 관리된 미세 조정(Supervised Fine-Tuning, SFT)이 언어 모델을 특정 요구 사항에 맞추기 위해 얼마나 '표면적으로' 조정되는지에 대한 가설을 검토하고, 새로운 훈련 없는 조정 방법인 PreTTY를 소개합니다. 이 방법은 기존의 대규모 언어 모델(LLM)과 SFT 언어 모델 간의 연결을 유지하면서 최소한의 작업 관련 토큰만을 사용하여 크로스 언어 생성 작업에서 비슷한 성능을 달성합니다.

- **Technical Details**: PreTTY는 기존 입력에 두 개의 작업 관련 우선 토큰을 추가하고, 이를 기반으로 LLM이 디코딩을 재개하도록 유도하는 방식으로 작동합니다. 이 접근 방식은 SFT가 필요로 하는 방대한 레이블이 지정된 데이터 세트와 교육 시간을 획기적으로 줄이며, 다양한 언어 자원에 걸쳐 유연하게 토큰을 구성할 수 있어 다국어 LLM의 민주화를 촉진합니다.

- **Performance Highlights**: PreTTY는 기계 번역, 크로스 언어 요약, 비영어 품사 태깅(POS Tagging) 작업에 대해 8개 언어로 실험을 수행했으며, 이러한 다양한 작업에서 SFT 모델의 성능에 필적하거나 유사한 결과를 달성했습니다. 특히 디코딩 과정을 시작할 때 단 하나 또는 두 개의 우선 토큰만 사용함으로써, 비용 효율적이면서도 효과적인 대안을 제시합니다.



### Dataset of Quotation Attribution in German News Articles (https://arxiv.org/abs/2404.16764)
Comments: To be published at LREC-COLING 2024

- **What's New**: 새로운 독일어 뉴스 기사를 위한 인용구 추례 데이터셋이 제공됩니다. 이 데이터셋은 WIKINEWS 기반으로 만들어졌으며, 크리에이티브 커먼즈 라이센스로 제공됩니다. 문서당 상세한 품질의 주석(annotation)이 포함되어 있어 다양한 후속 작업에서 널리 사용될 수 있습니다.

- **Technical Details**: 이 데이터셋은 1000개의 문서와 250,000개의 토큰(token)을 포함하고, 누가 무엇을 말했는지, 어떻게, 어떤 맥락에서, 누구에게 말했는지를 상세하게 주석 처리하여 매우 세밀한 주석 체계(annotation schema)를 제공합니다. 또한 인용의 유형(type of quotation)도 정의합니다. 데이터셋 생성 과정과 주석 체계, 그리고 데이터셋의 정량적 분석(quantitative analysis)에 대해 자세히 설명합니다.

- **Performance Highlights**: 이 연구에서는 적합한 평가 지표(evaluation metrics)를 제시하고, 기존의 두 인용 추례 시스템을 적용하여 데이터셋의 유용성을 평가하였습니다. 시스템들의 결과를 토대로 데이터셋의 충분한 유용성이 입증되었으며, 데이터셋이 어떻게 다양한 하위 작업(downstream tasks)에 활용될 수 있는지에 대한 사례들도 제시됩니다.



### Automatic Speech Recognition System-Independent Word Error Rate  Estimatio (https://arxiv.org/abs/2404.16743)
Comments: Accepted to LREC-COLING 2024 (long)

- **What's New**: 이 연구에서는 새로운 ASR System-Independent WER (단어 오류율) 추정 방법인 SIWE를 제안합니다. 기존의 WER 추정은 특정 ASR 시스템에 의존적이며 도메인에 종속적인 문제를 가지고 있었으나, 이 논문에서 제안하는 방법은 ASR 시스템 출력을 시뮬레이션하는 데이터를 사용하여 학습된 WER 추정기를 제공합니다. 이는 기존 방법들과 비교해 도메인에 덜 종속적이며 실제 환경에서의 유연성을 높여줍니다.

- **Technical Details**: 제안된 SIWE 방법은 음성 유사성이 높거나 언어학적으로 더 가능성 있는 대체 단어를 사용하여 가설을 생성합니다. 이 가설은 ASR 시스템 출력을 모방하기 위해 설계되었으며, 이를 통해 ASR 시스템에 독립적인 WER 추정이 가능하게 됩니다. 사용된 데이터는 특정 도메인에 국한되지 않고 다양한 도메인에서의 성능을 평가할 수 있도록 합니다.

- **Performance Highlights**: SIWE 모델은 기존 ASR system-dependent WER 추정기와 동일한 수준의 성능을 내도메인 데이터에서 보여줬으며, out-of-domain 데이터에서는 기존 모델들을 뛰어넘는 최고의 성능을 달성했습니다. 특히, Switchboard와 CALLHOME 데이터셋에서 root mean square error(평균 제곱근 오차)와 Pearson correlation coefficient(피어슨 상관 계수)에서 각각 17.58%, 18.21% 상대적인 성능 향상을 보였습니다. 또한, 훈련 세트의 WER가 평가 데이터셋의 WER와 비슷할 때 성능이 더욱 향상되었습니다.



### Layer Skip: Enabling Early Exit Inference and Self-Speculative Decoding (https://arxiv.org/abs/2404.16710)
Comments: Code open sourcing is in progress

- **What's New**: LayerSkip은 큰 언어 모델 (Large Language Models, LLMs)의 추론 속도를 높이기 위한 종단간(end-to-end) 솔루션을 제시합니다. 훈련 중에는 초기 레이어에는 낮은 드롭아웃 비율을, 후반 레이어에는 높은 드롭아웃 비율을 적용하며, Early Exit Loss를 도입해 트랜스포머(Transformer)의 모든 레이어가 같은 출구를 공유하도록 합니다.

- **Technical Details**: LayerSkip은 추론 동안 이러한 훈련 방식이 초기 레이어에서의 Early Exit의 정확도를 높임을 보여 줍니다. 또한, 추론 중에는 유추단계(inference stage)에서 초기 레이어에서 먼저 탈출한 후, 나머지 레이어를 사용하여 검증하고 수정하는, 독창적인 자기-추측(Self-Speculative) 디코딩 방식을 제안합니다. 이 방법은 다른 추측 디코딩 방식(Speculative Decoding Approaches)보다 메모리 사용이 적고, 초안(draft) 단계와 검증(verification) 단계의 계산과 활성화(activations)를 공유함으로써 이점을 얻습니다.

- **Performance Highlights**: 다양한 Llama 모델 크기와 훈련 유형(스크래치에서의 사전 훈련(pretraining from scratch), 지속적인 사전 훈련(continual pretraining), 특정 데이터 도메인에 대한 파인튜닝(finetuning), 특정 작업에 대한 파인튜닝)을 대상으로 실험을 수행했습니다. CNN/DM 문서 요약에 대해 최대 2.16배, 코딩 작업에 대해 1.82배, TOPv2 의미 파싱 작업(semantic parsing task)에 대해 2.0배의 속도 향상을 구현하여 보여주었습니다.



### Cooperate or Collapse: Emergence of Sustainability Behaviors in a  Society of LLM Agents (https://arxiv.org/abs/2404.16698)
- **What's New**: 새로운 시뮬레이션 플랫폼 GovSim (Governance of the Commons Simulation)이 소개되었습니다. 이 플랫폼은 인공 지능 대형 언어 모델들(Large Language Models, LLMs)의 전략적 상호작용과 협력적 의사결정을 연구하는 데 초점을 두고 있습니다. GovSim은 텍스트 기반 에이전트를 지원하며, 다양한 LLMs를 통합할 수 있는 표준 에이전트를 사용하는 Generative Agent 프레임워크를 사용합니다.

- **Technical Details**: GovSim은 AI 에이전트 간의 자원 공유와 윤리적 고려, 전략적 계획, 협상 기술의 중요성을 강조하는 시뮬레이션 환경을 제공합니다. 이 연구는 15개의 LLMs를 테스트한 결과, 오직 두 모델만이 지속 가능한 결과를 달성했다는 것을 발견, 대부분의 모델이 공유 자원 관리 능력에 큰 격차가 있음을 드러냈습니다. 또한, 에이전트들이 의사소통을 제거하면 공유 자원을 과도하게 사용하는 경향이 있음을 발견, 협력을 위한 의사소통의 중요성을 강조합니다.

- **Performance Highlights**: LLMs의 주요 약점 중 하나는 보편화된 가설을 만드는 능력이 부족하다는 것입니다. 이 연구는 또한 오픈 소스로 시뮬레이션 환경, 에이전트 프롬프트, 종합적인 웹 인터페이스를 포함하는 연구 결과 전체를 공개하였습니다.



### Influence of Solution Efficiency and Valence of Instruction on Additive  and Subtractive Solution Strategies in Humans and GPT-4 (https://arxiv.org/abs/2404.16692)
Comments: 29 pages, 2 figures

- **What's New**: 이 연구에서는 초기 상태나 구조를 변경할 때 요소를 추가하는 것을 제거하는 것보다 선호하는 인지적 경향인 추가 편향(addition bias)을 탐구했습니다. 이를 위해 사람과 OpenAI의 GPT-4 대형 언어 모델 양쪽에서 문제 해결 행동을 조사하는 네 번의 사전 등록(experiments) 실험이 수행되었습니다.

- **Technical Details**: 실험에는 미국 출신의 588명의 참가자와 GPT-4 모델의 680회 반복(iterations)이 포함되었습니다. 문제 해결 과제는 그리드 내에서 대칭을 생성하는 것(실험 1 및 3) 또는 요약을 편집하는 것(실험 2 및 4)이었습니다. 예상대로, 추가 편향이 전반적으로 나타났습니다. 해결 효율성(solution efficiency)과 지시의 긍정성(valence of the instruction)이 중요한 역할을 했습니다.

- **Performance Highlights**: 인간 참가자는 추가와 제거가 동등하게 효율적일 때 보다 제거가 상대적으로 더 효율적일 때 덜 추가 전략을 사용하는 경향이 있었습니다. 반면 GPT-4는 제거가 더 효율적일 때 강한 추가 편향을 보였습니다. 지시의 긍정성에 따라, '개선'하라는 요청을 받았을 때 GPT-4는 '편집'하라는 요청보다 더 많은 단어를 추가할 가능성이 높았지만, 인간은 이러한 효과를 보이지 않았습니다. 다양한 조건에서 추가 편향을 분석했을 때, GPT-4는 인간에 비해 더 편향된 반응을 보였습니다. 이러한 발견은 뛰어나거나 때로는 상대적으로 우월한 제거 대안(subtractive alternatives)을 고려하는 것의 중요성을 강조하며, 특히 언어 모델의 문제 해결 행동을 재평가할 필요성을 제기합니다.



### ProbGate at EHRSQL 2024: Enhancing SQL Query Generation Accuracy through  Probabilistic Threshold Filtering and Error Handling (https://arxiv.org/abs/2404.16659)
Comments: The 6th Clinical Natural Language Processing Workshop at NAACL 2024. Code is available at this https URL

- **What's New**: 최근의 연구에서는 의료 분야의 환자 기록 검색에 적용할 수 있는 텍스트-to-SQL(Text2SQL) 작업에서 딥러닝 기반 언어 모델의 효과를 크게 향상시켰습니다. 특히, 이 연구에서는 답변할 수 없는 질문을 식별하고 거르는 새로운 엔트로피 기반 방법과 확률 게이트(ProbGate)라는 새로운 확률 기반 필터링 접근 방식을 소개했습니다. 이 방법들은 실제 데이터베이스에서 쿼리를 실행하여 문법적 및 스키마 오류를 완화하면서 결과의 품질을 향상시키는 데 도움이 됩니다.

- **Technical Details**: 이 논문에서는 의료용 질문과 해당 SQL 쿼리를 사용하여 진행된 EHRSQL 공유 작업에서의 구현에 대해 설명하고 있습니다. ProbGate는 로그 확률(log probability)에 기반한 분포를 사용하여 낮은 신뢰도의 SQL을 필터링하고, 구체적인 대상 토큰의 로그 확률을 사용하여 모델의 신뢰도와 작업 수행 능력을 평가합니다. T5 및 gpt-3.5-turbo 모델을 기반으로 한 이중 분류기와의 비교 실험을 통해, ProbGate가 데이터 분포 변화에 대한 탄력성과 성능 면에서 우수함을 입증했습니다.

- **Performance Highlights**: ProbGate는 기존의 이중 분류 방식과 비교하여 더 높은 성능과 데이터 분포 변화에 대한 더 강한 저항성을 보여주었습니다. 결과적으로, 이 접근 방식은 의료 분야를 비롯한 다양한 응용 프로그램에서 효과적으로 답변할 수 없는 질문을 필터링하는 데 사용될 수 있는 가능성을 제시합니다. 실험을 통해 최적화된 문제 해결 방법을 통해 실제 병원 시스템에서 사용될 수 있는 SQL 문을 생성하는 데 있어 ProbGate가 큰 잠재력을 가지고 있음을 확인할 수 있었습니다.



### Análise de ambiguidade linguística em modelos de linguagem de grande  escala (LLMs) (https://arxiv.org/abs/2404.16653)
Comments: in Portuguese language, 16 p\'aginas, 5 p\'aginas de ap\^endice e 4 imagens

- **What's New**: 이 연구는 자연어 처리(Natural Language Processing, NLP) 시스템에서 계속되는 언어적 모호성에 대해 탐색합니다. 트랜스포머(Transformers)와 BERT와 같은 최신 아키텍처의 발전에도 불구하고, 이 연구는 특히 브라질 포르투갈어에서 발견되는 의미론적(Semantic), 구문론적(Syntactic), 어휘적(Lexical) 모호성에 초점을 맞추어, ChatGPT와 Gemini(2023년에는 Bard라고 불림) 같은 최신 설명적 모델의 한계와 가능성을 분석합니다.

- **Technical Details**: 연구는 모호한 및 모호하지 않은 문장 120개로 구성된 말뭉치를 만들어 분류, 설명, 해석 작업을 수행합니다. 또한, 각 모호성 유형에 대한 문장 세트를 생성하여 모델의 문장 생성 능력을 평가합니다. 분석은 인정된 언어학 참조를 사용하여 질적 분석과, 얻어진 응답의 정확도를 바탕으로 한 정량적 평가를 병행합니다.

- **Performance Highlights**: 분석 결과, ChatGPT와 Gemini 같은 고급 모델들도 여전히 응답에서 오류와 결점을 보여주며, 설명은 종종 일관성이 없는 것으로 나타났습니다. 정확도는 최고 49.58퍼센트에 그쳤으며, 이는 감독 학습(Supervised Learning)을 위한 기술적 연구가 필요함을 시사합니다.



### Tele-FLM Technical Repor (https://arxiv.org/abs/2404.16645)
- **What's New**: 새롭게 개발된 Tele-FLM (또는 FLM-2라고 불리는)은 52B (52 billion parameters) 매개변수를 가진 다양한 언어를 지원하는 큰 언어 모델(Large Language Model, LLM)입니다. 이 모델은 효율적인 사전 훈련(pre-training) 패러다임과 개선된 사실적 판단 능력(factual judgment capabilities)을 갖추고 있다는 점에서 기존 모델들과 차별화됩니다. 또한, 이 연구는 모델의 세부 설계, 엔지니어링 실습, 훈련 세부 정보를 공개하며, 학술 및 산업 커뮤니티 모두에게 이득이 될 것으로 기대됩니다.

- **Technical Details**: Tele-FLM은 최소한의 시행착오 비용과 계산 자원을 사용하여 50B 이상으로 LLM을 효율적으로 확장하는 방법에 대한 세부적이고 개방된(source-opened) 방법론을 제공합니다. 이 모델은 BPB (bits per byte)를 사용하여 텍스트 코퍼스(textual corpus)에 대한 다국어 언어 모델링 능력을 측정합니다.

- **Performance Highlights**: Tele-FLM은 영어 및 중국어 기초 모델 평가(foundation model evaluation)에서 Llama2-70B나 DeepSeek-67B와 같이 더 큰 사전 훈련 FLOPs(Floating Point Operations Per Second)를 포함하는 강력한 개방형 모델들과 비교할 때 우수한 성능을 보여줍니다.



### Incorporating Lexical and Syntactic Knowledge for Unsupervised  Cross-Lingual Transfer (https://arxiv.org/abs/2404.16627)
Comments: Accepted at LREC-Coling 2024

- **What's New**: 이 논문에서는 'Lexicon-Syntax Enhanced Multilingual BERT'란 새로운 프레임워크를 소개합니다. 이 프레임워크는 언어 간 지식 전달을 위해 어휘적(lexical) 지식과 구문적(syntactic) 지식을 동시에 활용합니다. 이전 연구들이 하나의 지식 유형만을 활용한 것과 달리, 본 연구는 두 유형의 정보를 통합하여 사용하는 접근 방식을 탐구합니다.

- **Technical Details**: 본 연구의 기반 모델로는 Multilingual BERT (mBERT)를 사용하며, 두 가지 기술을 적용하여 학습 능력을 강화합니다. 첫 번째로 'code-switching' 기술을 사용하여 모델이 어휘적 정렬 정보를 간접적으로 학습하게 합니다. 두 번째로는 구문 기반의 'graph attention network'를 설계하여 모델이 구문 구조를 인코딩하도록 돕습니다. 어휘적 지식과 구문적 지식의 통합을 위해, 코드 전환된 시퀀스를 구문 모듈과 mBERT 기본 모델에 동시에 입력합니다.

- **Performance Highlights**: 이 프레임워크는 텍스트 분류, 명명된 개체 인식(named entity recognition, NER), 의미 파싱(semantic parsing) 작업에서 제로샷(zero-shot) 언어 간 전송 기준 모델들을 일관되게 능가했습니다. 성능 향상은 1.0~3.7 포인트의 이득을 보였습니다.



### Understanding Privacy Risks of Embeddings Induced by Large Language  Models (https://arxiv.org/abs/2404.16587)
- **What's New**: 이 연구는 대규모 언어 모델(Large Language Models, LLMs)이 개인 정보 보호에 미치는 위험을 탐구하며, LLM을 사용하여 텍스트 임베딩에서 중요한 정보를 추출할 수 있는 가능성을 고찰합니다. 특히, 이 연구는 LLM의 발전이 텍스트 및 속성 재구성(Reconstruction) 능력을 어떻게 향상시키는지를 실험적으로 입증하며, 이로 인한 개인정보 유출 가능성을 지적합니다.

- **Technical Details**: 연구진은 GPT2와 같은 사전 훈련된 언어 모델을 사용하여 대상 임베딩 모델로부터 텍스트를 재구성하고 개인 속성을 예측합니다. 임베딩 모델의 파라미터가 많을수록 정보 누출 위험이 높아진다는 가설을 설정하고 실험을 진행했습니다. 이 과정에서 임베딩 모델은 텍스트로부터 벡터 데이터베이스로의 지식을 저장하는 도구로 사용됩니다.

- **Performance Highlights**: LLM은 사전 훈련된 모델보다 텍스트 재구성과 속성 예측에서 높은 정확도를 보였습니다. 이 연구에서 사용된 여러 대상 임베딩 모델들은 다양한 시나리오에서 개인 정보 추출 능력을 검증 받았으며, 특히 'SimCSE', 'BGE-Large-en', 'E5-Large-v2'와 같은 모델들이 상당한 정확도 개선을 보여주었습니다. 이 결과는 LLM이 보다 정밀하게 세밀한 정보를 추출할 수 있는 잠재력을 가지고 있음을 시사합니다.



### Exploring Internal Numeracy in Language Models: A Case Study on ALBER (https://arxiv.org/abs/2404.16574)
Comments: 4 pages + references, 4 figures. Accepted for publication at the MathNLP Workshop at LREC-COLING 2024

- **What's New**: Transformer 기반 언어 모델이 기본적인 양적 논리(quantitative reasoning) 수행 능력을 보유하고 있다는 것이 밝혀졌습니다. 이 연구에서는 ALBERT 언어 모델 시리즈를 분석하여, 이 모델들이 숫자와 서수(ordinals)에 해당하는 토큰을 어떻게 내부적으로 표현하는지 조사하는 새로운 방법을 제안합니다.

- **Technical Details**: 연구팀은 ALBERT 모델이 사용하는 수치 데이터를 대표하는 임베딩(learned embeddings)을 추출하고, 이를 주성분 분석(Principal Component Analysis, PCA)에 적용했습니다. PCA 결과는 다양한 크기의 ALBERT 모델들이 독립적으로 훈련되고 초기화되었음에도 불구하고, 여러 수치 개념의 대략적인 순서를 나타내는 데 가장 큰 변동의 축을 일관되게 사용하는 것으로 나타났습니다.

- **Performance Highlights**: 수치(numeral)들과 그것의 텍스트 표현(textual counterparts)은 2차원 공간에서 별도의 클러스터를 형성하지만, 같은 방향으로 증가한다는 결과를 확인했습니다. 이러한 발견은 텍스트만을 모델링하도록 훈련된 언어 모델이 기본적인 수학적 개념을 이해할 수 있음을 시사하며, 양적 논리를 요구하는 자연어 처리(Natural Language Processing, NLP) 응용 분야로의 가능성을 열어줄 수 있습니다.



### Evaluating Large Language Models on Time Series Feature Understanding: A  Comprehensive Taxonomy and Benchmark (https://arxiv.org/abs/2404.16563)
- **What's New**: 이 논문에서는 대규모 언어 모델들(Large Language Models, LLMs)이 시계열 데이터 분석 및 보고에 활용될 잠재성을 탐색하며, 이를 위한 종합적인 평가 프레임워크를 제안합니다. 또한, 시계열 데이터의 다양한 특성을 설명하는 포괄적인 특성 분류체계(taxonomy)를 소개합니다.

- **Technical Details**: 연구팀은 다양한 시계열 특성을 반영한 폭넓은 데이터셋을 구축하여 LLM의 시계열 이해 능력을 평가하였습니다. 이는 단변수 및 다변수 시계열(univariate and multivariate time series)을 모두 포함합니다.

- **Performance Highlights**: LLMs는 일부 시계열 특성을 효과적으로 이해하는 능력을 보였으나, 데이터 포맷(formatting), 시리즈 내 쿼리된 데이터 포인트의 위치(position) 및 시계열 길이(overall time series length)와 같은 요소에 따라 민감하게 반응하는 것으로 나타났습니다. 이는 LLM의 시계열 분석 시 직면하는 강점과 제한을 드러냅니다.



### Building a Japanese Document-Level Relation Extraction Dataset Assisted  by Cross-Lingual Transfer (https://arxiv.org/abs/2404.16506)
Comments: Accepted LREC-COLING 2024

- **What's New**: 이 연구는 일본어와 같은 비영어권 언어에서의 문서 수준 관계 추출 (DocRE, Document-level Relation Extraction) 연구를 촉진하기 위해 기존의 영어 자원을 효과적으로 활용하는 방안을 탐구합니다. 영어 데이터셋을 일본어로 변환하여 데이터셋을 구축했으며, 이를 통해 모델 훈련 시 낮은 리콜(recall) 문제에 직면했습니다. 이러한 문제를 해결하기 위해, 번역된 데이터셋이 일본어 문서의 인간 주석에 어떻게 도움을 줄 수 있는지 조사하였습니다.

- **Technical Details**: 연구팀은 영어 데이터셋을 일본어로 변환하여 일본어 DocRE 연구를 위한 새로운 데이터셋을 만들었습니다. 변환된 데이터셋에서 훈련된 모델은 낮은 리콜 문제를 겪었고, 이는 영어로 번역된 문서와 원어민이 작성한 문서 간의 표면 구조와 의미 차이 때문인 것으로 확인되었습니다. 이에 따라, 연구팀은 번역된 데이터셋이 모델에서 예측한 관계(relation)를 인간 주석자가 수정하는 데 도움을 줄 수 있는지를 조사하였습니다.

- **Performance Highlights**: 모델이 제안한 관계 추천(relation recommendations)은 인간의 편집 단계를 이전 접근 방식에 비해 약 50% 줄이는 데 도움을 주었습니다. 이러한 접근 방식을 통해 일본어 문서의 DocRE에 대한 효과적인 인간 주석 접근 방식이 제안되었습니다. 실험 결과는 일본어와 교차 언어(Cross-lingual) DocRE의 도전적인 측면을 보여주며, 해당 새로운 데이터셋에서 기존 DocRE 모델의 성능을 정량적으로 평가합니다.



### Evaluating Consistency and Reasoning Capabilities of Large Language  Models (https://arxiv.org/abs/2404.16478)
- **AI Newsletter - Korean Edition**: [{"What's New": '이 연구는 대규모 언어 모델(Large Language Models, LLMs)의 일관성과 추론 능력을 평가하고 비교합니다. LLM은 텍스트 생성, 요약, 번역과 같은 작업에 널리 사용되지만, 종종 틀리거나 혼란스러운 정보를 생성하는 경향이 있습니다. 이 논문은 Boolq 데이터셋을 사용하여, 공개 및 소유 LLM의 성능을 비교 분석합니다.'}, {'Technical Details': '연구는 Boolq 데이터셋을 기반으로, 질문과 답변, 그리고 그에 대한 설명을 포함하여 LLM에 제시하고 논리적 추론 및 일관성을 평가하였습니다. 모델의 일관성은 같은 질의를 여러 번 제시하여 반응을 관찰함으로써 평가되었고, 추론 능력은 생성된 설명이 기존의 설명과 얼마나 일치하는지 BERT, BLEU, F-1 점수(Metrics)를 사용하여 평가했습니다.'}, {'Performance Highlights': '연구 결과에 따르면, 소유 모델(proprietary models)이 공개 모델(public models)보다 일관성과 추론 능력 모두에서 더 우수한 성능을 보였습니다. 그러나 기본 일반 지식 질문에 대해서는 어떤 모델도 일관성과 추론에서 90% 이상의 점수를 달성하지 못했습니다. 이는 LLM이 현재 가지고 있는 추론 능력에 대한 과제를 시사합니다.'}]



### Large Language Models Perform on Par with Experts Identifying Mental  Health Factors in Adolescent Online Forums (https://arxiv.org/abs/2404.16461)
- **What's New**: 이 연구는 청소년(12-19세)의 Reddit 게시물을 전문 정신과 의사들이 다음과 같은 범주로 주석을 단 새로운 데이터셋을 생성했습니다: TRAUMA (트라우마), PRECARITY (불안정), CONDITION (상태), SYMPTOMS (증상), SUICIDALITY (자살성), TREATMENT (치료). 이 데이터셋은 두 가지 최고 성능의 Large Language Models (LLM), GPT3.5와 GPT4와 비교 분석되었습니다. 추가로, LLM이 데이터를 생성함에 따라 어노테이션 성능이 향상되는지 확인하기 위해 두 개의 인공 데이터셋을 생성했습니다.

- **Technical Details**: 연구자들은 GPT3.5와 GPT4를 사용하여 청소년의 Reddit 게시물에 어노테이션을 적용하고 전문가의 레이블과 비교했습니다. 또한, 인공 데이터셋을 생성하여 LLM이 자체적으로 생성한 데이터에 대한 성능을 평가했습니다. 분석 결과 GPT4가 인간 어노테이터 간 합의(Inter-annotator agreement) 수준에 근접했으며, 인공 데이터에서의 성능은 현실 데이터의 복잡성 때문이 아니라 데이터의 본질적 장점에 의해 주도된 것으로 나타났습니다.

- **Performance Highlights**: 연구 결과에 따르면 GPT4는 실제 데이터에 비해 인공 데이터에서 상당히 높은 성능을 보였습니다. 하지만 모델은 여전히 부정(negation)과 사실성(factuality)에 대한 문제에서 오류를 범할 수 있습니다. 이는 Large Language Models의 잠재적 한계로 지적됩니다. 실제 데이터의 성능이 인공 데이터보다 낮은 주 원인은 실제 데이터의 복잡성 때문입니다.



### Contextual Categorization Enhancement through LLMs Latent-Spac (https://arxiv.org/abs/2404.16442)
- **language_version**: eng-KOR

- **What's New**: 위키백과와 같은 대규모 텍스트 데이터셋의 분류 작업을 관리하는 것은 매우 복잡하고 비용이 많이 드는 과제입니다. 논문에서는 트랜스포머 모델(transformer models)을 활용하여 위키백과 데이터셋의 텍스트로부터 의미 정보(semantic information)를 추출하고, 이를 잠재 공간(latent space)에 통합하는 새로운 방법을 제안합니다.

- **Technical Details**: 이 논문에서는 컨벡스 헐(Convex Hull)을 이용한 그래픽 접근 방식과 계층적 내비게이션 가능 스몰 월드(Hierarchical Navigable Small Worlds, HNSWs)를 사용하는 계층적 접근 방식을 탐구합니다. 또한 차원 축소로 인한 정보 손실을 보완하기 위해 유클리드 거리(Euclidean distances)에 의해 구동되는 지수 감소 함수(exponential decay function)를 적용합니다. 이 함수는 특정 재고려 확률(Reconsideration Probability, RP)과 함께 항목을 검색하는 데 사용되며, 이는 데이터베이스 관리자들이 데이터 그룹을 개선하고 맥락적 프레임워크 내에서 이상치(outliers)를 식별하는 데 도움을 줄 수 있습니다.

- **Performance Highlights**: 이 연구의 접근 방식은 위키 백과 데이터셋과 관련 카테고리의 의미적 정체성(semantic identity)을 평가하고 향상시키는 효과적인 수단을 제공합니다. 컨벡스 헐과 HNSWs를 사용하는 방법은 높은 재고려 확률을 가진 항목을 성공적으로 검색함으로써 데이터베이스의 의미론적(semantic) 구조 개선에 중요한 역할을 할 수 있습니다.



### Instruction Matters, a Simple yet Effective Task Selection Approach in  Instruction Tuning for Specific Tasks (https://arxiv.org/abs/2404.16418)
Comments: 21 pages, 6 figures, 16 tables

- **What's New**: 본 연구에서는 명령어(instruction) 정보만을 사용하여 특정 작업(instruction tuning)을 위한 관련 작업을 선정하는 새로운 방법을 제시하였습니다. 전통적인 방법에 비해 더 간단하게 작업 간의 이전 가능성(transferability)을 측정할 필요 없이, 혹은 대상 작업을 위한 데이터 샘플을 만들 필요 없이 관련 작업을 선택할 수 있습니다.

- **Technical Details**: 이 연구는 명령어 정보를 이용하여 작업을 선택함으로써, 메타-데이터셋(meta-dataset)의 고유한 명령어 템플릿 스타일(instructional template style)을 추가로 학습함에 따라 작업 선택 정확도가 향상되고, 이는 전반적인 성능 개선에 기여한다는 것을 발견했습니다. 본 방법은 훈련 세트를 작은 수의 관련 작업으로 제한함으로써 이득을 봅니다.

- **Performance Highlights**: 실험 결과, 명령어 기반으로 선정된 작업에 대한 훈련이 P3, Big-Bench, NIV2, Big-Bench Hard와 같은 벤치마크에서 상당한 성능 향상을 가져왔으며, 이 성능 향상은 기존 작업 선택 방법들을 능가하는 결과를 보였습니다.



### Asking and Answering Questions to Extract Event-Argument Structures (https://arxiv.org/abs/2404.16413)
Comments: Accepted at LREC-COLING 2024

- **What's New**: 이 논문은 문서 수준의 이벤트-인자(event-argument) 구조를 추출하기 위한 질의응답 방식을 제시합니다. 각 인자 유형에 대해 자동으로 질문을 생성하고 답변합니다. 이 연구는 템플릿 기반 질문과 변압기(generative transformers) 기반 질문을 사용하여 질문을 생성합니다. 템플릿 기반 질문은 사전 정의된 역할별 wh-words와 문맥 문서에서의 이벤트 트리거를 사용하여 생성되며, 변압기 기반 질문은 통과 문장과 예상되는 답변을 바탕으로 질문을 구성하는 대규모 언어 모델을 사용하여 생성됩니다. 또한, 문장 간의 이벤트-인자 관계에 특화된 새로운 데이터 확장 전략을 개발하였습니다.

- **Technical Details**: 사용된 기술들에는 간단한 스팬 교환 기법(span-swapping technique), 대용량 언어 모델을 이용한 공동 참조 해결(coreference resolution), 그리고 훈련 인스턴스를 확장하는 대규모 언어 모델이 포함됩니다. 이 접근법은 어떠한 말뭉치(corpus) 특정 수정 없이 전이 학습(transfer learning)을 가능하게 하며, RAMS 데이터셋에서 경쟁력 있는 결과를 얻었습니다.

- **Performance Highlights**: 이 방법은 이전 연구보다 우수한 성능을 보이며, 특히 이벤트 트리거와 다른 문장에 나타나는 인자를 추출하는 데 유리합니다. 또한, 가장 흔한 오류들에 대한 자세한 양적 및 질적 분석을 제시하여 우리 모델의 성능을 더욱 향상시킬 수 있는 방향을 제안합니다.



### U2++ MoE: Scaling 4.7x parameters with minimal impact on RTF (https://arxiv.org/abs/2404.16407)
- **What's New**: 이 연구는 자연어 처리(Natural Language Processing, NLP)와 연설 인식(Automatic Speech Recognition, ASR) 모델에서 Mixture-of-Experts (MoE) 아키텍처를 활용하여 모델 규모를 확장하는 새로운 접근 방식을 제시합니다. 특히, 모든 Feed-Forward Network (FFN) 계층을 MoE 계층으로 대체하는 간단한 방식으로 ASR 작업에 효과적임을 증명했습니다. 유니파이드 2패스 프레임워크(Unified 2-pass framework)를 통한 스트리밍 및 비스트리밍 디코딩 모드를 단일 MoE 기반 모델에서 처리할 수 있게 되었습니다.

- **Technical Details**: 연구진은 Conformer 인코더와 Transformer 디코더를 기반으로 모두 FFN 계층을 MoE 계층으로 교체하여 실험을 수행했습니다. MoE 계층은 라우팅 네트워크와 여러 전문가(experts) FFN 포함합니다. 훈련은 Connectionist Temporal Classification (CTC) 손실과 Autoregressive Encoder Decoder (AED) 손실을 결합한 방법으로 진행됐으며, 스트리밍과 비스트리밍 모드를 단일 모델에서 지원하기 위해 동적 청크 마스킹 전략도 사용되었습니다.

- **Performance Highlights**: 연구진은 160,000시간의 대규모 데이터셋에서 모델을 벤치마크했으며, MoE-1B 모델은 Dense-1B 모델 수준의 단어 오류율(Word Error Rate, WER)을 달성하면서 Dense-225M 모델 수준의 실시간 계수(Real Time Factor, RTF)를 유지할 수 있음을 확인했습니다. 이 연구는 ASR을 위한 MoE 활용 방법을 간소화하고 효율적으로 대규모 모델을 학습할 수 있는 방법론을 제시합니다.



### Lost in Recursion: Mining Rich Event Semantics in Knowledge Graphs (https://arxiv.org/abs/2404.16405)
Comments: Accepted at WebSci'24, 11 pages, 4 figures

- **What's New**: 이 연구는 복잡한 사건에 대한 서사를 구성하고 활용하는 방법을 보여줍니다. 복잡한 사건(Narratives around Complex Events)에 대한 서술입시를 언어 중개자들을 통해 간접적으로 접하고 이러한 사건에 대한 다양한 관점을 마이닝(mining)하기 위한 새로운 알고리즘을 제안합니다. 이는 사건 중심의 지식 그래프(Event-centric Knowledge Graphs)에 결합될 수 있는 재귀 노드(Based on Recursive Nodes)를 사용하여 서사를 정식화하고 표현하는 방법에 대한 설명을 포함합니다.

- **Technical Details**: 이 논문에서는 재귀 노드를 기반으로 다양한 디테일 수준을 나타내는 서사의 형식적 표현을 제공합니다. 또한, 텍스트에서 다양한 관점의 복잡한 사건 서사를 추출하는데 사용할 수 있는 증분 프롬프팅 기법(Incremental Prompting Techniques) 기반 알고리즘을 제공합니다. 이러한 방법은 사건 중심 지식 그래프와 연결되어 각 서사가 어떻게 사건에 매핑될 수 있는지를 설명합니다.

- **Performance Highlights**: 실험을 통해 제안된 방법의 효과성을 확인하였으며, 복잡한 사건에 대한 다양한 서사를 구성하고 해석하는 데 있어 미래의 연구 방향을 제시합니다. 이 프로토타입은 각 노드가 다른 상세 수준의 정보를 포함할 수 있도록 하는 재귀적 구조를 시험함으로써, 서사구조의 유연성을 높여 주었습니다.



### Don't Say No: Jailbreaking LLM by Suppressing Refusa (https://arxiv.org/abs/2404.16369)
- **What's New**: 이 연구에서는 대형 언어 모델(LLM)이 유해한 쿼리를 거절하는 것을 회피하고 긍정적인 반응을 유도할 수 있도록 디자인된 새로운 'DSN(Don't Say No)' 공격을 소개합니다. DSN 공격은 LLM이 거부 반응을 억제하는 동시에 긍정적인 반응을 제공하도록 유도합니다. 또한, 이 연구는 공격의 성공을 더 정확하게 평가하기 위해 NLI(Natural Language Inference) 모순 평가와 두 개의 외부 LLM 평가자를 포함하는 앙상블 평가 파이프라인을 제안합니다.

- **Technical Details**: DSN 공격은 특정 거부 키워드나 문자열로부터 LLM의 반응을 이탈시키는 방향으로 증가 손실 항목을 포함합니다. 이 과정에서 최대 긍정 반응 확률과 최소 거부 키워드 확률을 동시에 달성하려고 합니다. 평가 방법으로는 기존 거부 문자열/키워드 매칭 메트릭 대신 NLI 모순 평가와 GPT-4 및 HarmBench와 같은 외부 LLM 평가자를 통합한 새로운 앙상블 접근 방식을 사용합니다.

- **Performance Highlights**: 확장된 실험을 통해 DSN 공격의 효력과 앙상블 평가의 효과가 기존 방법에 비해 우수함을 입증했습니다. 특히, DSN은 더 높은 성공률로 LLM으로부터 긍정적인 반응을 유도할 수 있었으며, 앙상블 평가는 가짜 양성(FP)과 가짜 음성(FN) 사례를 줄이는 데 효과적이었습니다.



### Learning Syntax Without Planting Trees: Understanding When and Why  Transformers Generalize Hierarchically (https://arxiv.org/abs/2404.16367)
- **What's New**: 본 연구에서는 Transformer 모델들이 구조적 편향을 명시적으로 코딩하지 않고도 자연어 데이터의 계층적 구조를 학습하고 보지 못한 구문 구조의 문장에 일반화하는 방법을 조사합니다. 특히, 다양한 합성 데이터셋과 훈련 목표를 사용하여 실험하면서, 언어 모델링(Language Modeling) 목적으로 훈련된 모델들이 일관되게 계층적 일반화 능력을 학습하는 것을 발견했습니다.

- **Technical Details**: Transformer 모델들이 여러 합성 데이터셋과 서로 다른 훈련 목표들, 예를 들어 시퀀스-투-시퀀스(Sequence-to-Sequence) 모델링, 프리픽스 언어 모델링(Preﬁx Language Modeling)을 사용하여 훈련되었을 때 대부분 계층적 일반화로 이어지지 않았습니다. 반면 언어 모델링 목표로 훈련된 모델들은 계층적 구조를 일반화하는 능력을 일관되게 학습했습니다. 모델이 훈련된 언어 모델링 목적으로 계층적 구조를 어떻게 인코딩하는지 파악하기 위해 프루닝(Pruning) 실험을 실시했습니다.

- **Performance Highlights**: 프루닝 실험을 통해, 언어 모델링 목적으로 훈련된 Transformer 모델 내에서 계층적 구조와 선형 순서에 대응하는 다양한 일반화 행동을 보이는 하위 네트워크들이 공존하는 것을 확인했습니다. 가장 간단한 설명이 계층적 문법에 의해 제공되는 데이터셋에서 Transformer가 계층적으로 일반화되는지 여부와의 상관 관계를 베이지안 관점에서 추가로 밝혔습니다.



### VISLA Benchmark: Evaluating Embedding Sensitivity to Semantic and  Lexical Alterations (https://arxiv.org/abs/2404.16365)
- **What's New**: 이 논문에서는 언어 모델들이 어떻게 의미론적(semanic) 및 어휘적(lexical) 세부 사항들을 포착하는지 평가하기 위해 VISLA(Variance and Invariance to Semantic and Lexical Alterations) 벤치마크를 소개합니다. VISLA는 이미지와 연관된 문장 삼중체로 구성된 3-way 의미(불)동등성 작업을 제안하며, 이는 비전-언어 모델(VLMs)과 단일 모드 언어 모델(ULMs) 모두를 평가합니다.

- **Technical Details**: VISLA 벤치마크는 이중 형태로 평가를 진행하는데, 하나는 이미지-텍스트 검색(image-to-text retrieval)과 텍스트-텍스트 검색(text-to-text retrieval) 작업을 통합한 새로운 형태입니다. 또한, 본 연구는 34개의 VLM과 20개의 ULM을 대상으로 튜닝 없이 평가를 수행하며, 이를 통해 언어 모델이 어휘적 변화를 감안한 경우의 의미론적(semantic) (불)변성을 측정합니다.

- **Performance Highlights**: 평가 결과, VLM의 텍스트 인코더들이 ULM의 단일 모드 텍스트 인코더들보다 의미론적 및 어휘적 변화에 더 민감하게 반응하는 것으로 나타났습니다. 그러나 언어 모델들이 어휘적과 의미론적 변화를 구분하는 데에 있어 여전히 어려움을 겪는 것으로 밝혀졌습니다. 이러한 결과는 다양한 비전과 단일 모드 언어 모델들의 강점과 약점을 조명하며, 이들의 능력에 대한 더 깊은 이해를 제공합니다.



### PILA: A Historical-Linguistic Dataset of Proto-Italic and Latin (https://arxiv.org/abs/2404.16341)
Comments: 12 pages, 1 figure, 9 tables. Accepted at LREC-COLING 2024

- **What's New**: 이탈리아어 계통의 언어에서 프로토-이태리어에서 라틴어로의 음운 변화를 체계적으로 이해하기 위한 새로운 데이터셋인 프로토-이태리어에서 라틴어로의 데이터셋(PILA)이 소개되었습니다. 이 데이터셋은 약 3,000개의 프로토-이태리어와 라틴어 사이의 형태쌍을 포함하고 있습니다.

- **Technical Details**: PILA 데이터셋은 프로토-이태리어와 라틴어 간의 음운학적(phonological) 및 형태학적(morphological) 연결을 탐색하여 정리된 방대한 정보를 제공합니다. 이 데이터셋은 역사언어학(historical linguistics)에서 음운 변화를 연구하는 데 매우 중요한 자원이 될 것입니다.

- **Performance Highlights**: PILA 데이터셋은 전통적인 계산 역사언어학(computational historical linguistics) 작업에 대한 기초 결과를 제시하며, 다른 역사-언어학 데이터셋을 강화하는 능력을 통한 데이터셋 호환성 연구를 시연하였습니다.



### WorldValuesBench: A Large-Scale Benchmark Dataset for Multi-Cultural  Value Awareness of Language Models (https://arxiv.org/abs/2404.16308)
Comments: Accepted at LREC-COLING 2024. Wenlong and Debanjan contributed equally

- **Abstract Summary**: {"What's New": '이 논문은 다문화 가치 인식(multi-cultural value awareness)에 대한 언어 모델(Language Models, LM)의 능력을 향상시키기 위해 전 세계적으로 다양한 WorldValuesBench 데이터셋을 소개합니다. 이 데이터셋은 사회 과학 프로젝트인 세계 가치 조사(World Values Survey, WVS)에서 파생되었으며, 94,728명의 참가자로부터 수집된 수백 개의 가치 질문에 대한 답변을 포함하고 있습니다.', 'Technical Details': "이 논문에서는 인구 통계적 특성 및 가치 질문을 기반으로 한 답변 생성(rating response) 과제를 위한 대규모 데이터셋을 구축했습니다. 생성된 데이터셋은 20백만 예시(million examples) 이상을 포함하여 다양한 유형의 '(인구 통계 속성, 가치 질문) -> 답변' 형태로 구성됩니다.", 'Performance Highlights': 'Alpaca-7B, Vicuna-7B-v1.5, Mixtral-8x7B-Instruct-v0.1 및 GPT-3.5 Turbo와 같은 강력한 오픈 및 폐쇄 소스 모델들은 인간 정규화 답변 분포에서 0.2 미만의 Wasserstein 1-distance를 달성하는 데 어려움이 있으며, 각각의 질문에서 11.1%, 25.0%, 72.2%, 그리고 75.0%의 정확도를 보였습니다.'}



### LLM-Based Section Identifiers Excel on Open Source but Stumble in Real  World Applications (https://arxiv.org/abs/2404.16294)
Comments: To appear in NAACL 2024 at the 6th Clinical Natural Language Processing Workshop

- **What's New**: 이 연구에서는 대규모 언어 모델(Large Language Models, LLMs)을 이용하여 전자 건강 기록(Electronic Health Records, EHR)의 관련 섹션 헤더를 식별하는 새로운 접근 방식을 제안합니다. 특히 GPT-4 모델을 사용하여 이러한 과제에 접근하며, 공개 및 사설 데이터셋에서의 성능을 평가합니다.

- **Technical Details**: GPT-4는 레이블 없는 데이터에서도 뛰어난 자연어 처리(Natural Language Processing, NLP)성능을 보여줍니다. 무지도 학습(Unsupervised Learning) 방식에서도 태스크에 맞춤화된 미세조정 없이 섹션 식별(section identification, SI) 문제를 해결할 수 있다는 것을 발견했습니다. 평가는 공개 접근데이터셋과 더 어려운 실세계 데이터셋에서 이루어졌습니다.

- **Performance Highlights**: GPT-4는 공개 데이터셋에서 훌륭한 성능을 보였으며, IOB(Inside-Outside-Beginning) 태깅 시스템을 사용하여 섹션의 시작과 끝을 효과적으로 예측할 수 있었습니다. 하지만, 사설 데이터셋에서는 성능이 떨어지는 양상을 보였으며 이는 추가 연구와 좀 더 어려운 벤치마크가 필요하다는 점을 시사합니다.



### Interpreting Answers to Yes-No Questions in Dialogues from Multiple  Domains (https://arxiv.org/abs/2404.16262)
Comments: To appear at NAACL 2024 Findings

- **newsletter**: [{"What's New": '이 연구는 간접적인 대답의 의미를 파악하는 문제를 다루며, 여러 도메인에서의 대화를 분석했습니다. 특히 영화 스크립트, 테니스 인터뷰, 항공사 고객 서비스와 같이 다양하고 특이한 분야에서 새로운 벤치마크(benchmarks)를 제시하였습니다.'}, {'Technical Details': '이 논문은 멀리 있는 데이터를 이용한 감독 학습(distant supervision)과 혼합 훈련(blended training) 방식을 기반으로 하여 새로운 대화 도메인에 빠르게 적응할 수 있는 방법을 제안합니다. 대화를 이해하기 위하여 대규모 언어 모델(language models)의 한계를 극복하고자 했습니다.'}, {'Performance Highlights': '실험 결과, 제안된 접근 방법은 성능 저하 없이 F1 스코어에서 11-34%의 개선을 이루어냈습니다. 이는 인공지능이 대화 내용의 뉘앙스를 이해하는 능력이 크게 향상되었음을 보여줍니다.'}]



### Translation of Multifaceted Data without Re-Training of Machine  Translation Systems (https://arxiv.org/abs/2404.16257)
Comments: 19 pages

- **What's New**: 이 연구에서는 복잡한 데이터 점의 각 구성 요소를 각각 번역하는 것이 아닌, 데이터 내에서 구성 요소 간의 상호 연결성을 고려하는 새로운 MT(machine translation) 파이프라인을 제안합니다. 이 MT 파이프라인은 모든 구성 요소를 하나의 번역 시퀀스로 연결한 후 번역을 수행하고, 번역된 시퀀스를 다시 데이터 구성 요소로 재구성합니다.

- **Technical Details**: 이 연구에서 도입한 카탈리스트 문장(Catalyst Statement, CS)은 데이터 내 연관성을 강화하고, 인디케이터 토큰(Indicator Token, IT)은 번역된 시퀀스를 해당 데이터 구성 요소로 분해하는 데 도움을 줍니다. 이러한 접근 방식은 전체적인 번역 품질을 향상시키는 동시에 훈련 데이터로서의 효과성도 향상시킵니다.

- **Performance Highlights**: 전통적인 방식으로 각 데이터 구성 요소를 별도로 번역하는 방식과 비교했을 때, 제안된 방법은 훈련된 모델의 성능을 향상시키는 데 기여합니다. 웹 페이지 순위(web page ranking, WPR) 작업에 대해 2.690점, 질문 생성(question generation, QG) 작업에 대해서는 XGLUE 벤치마크에서 0.845점 더 좋은 결과를 보였습니다.



### Semgrex and Ssurgeon, Searching and Manipulating Dependency Graphs (https://arxiv.org/abs/2404.16250)
Comments: Georgetown University Round Table (GURT) 2023

- **What's New**: 본 논문에서는 의존성 그래프(Dependency graphs)의 검색과 조작을 위한 두 가지 새로운 시스템, Semgrex와 Ssurgeon을 소개합니다. Semgrex는 의존성 그래프에서 정규 표현식과 유사한 패턴을 검색하고, Ssurgeon은 Semgrex의 출력을 사용하여 의존성 그래프를 재작성하는 데 사용됩니다. 두 시스템 모두 Java로 작성되었으며, Java API 및 커맨드 라인 도구(Command Line Tools)가 제공됩니다. 추가적으로 Python 인터페이스도 제공되어, 텍스트 관계 및 속성을 쉽게 검색할 수 있습니다.

- **Technical Details**: Semgrex는 CoNLL-U 파일에서 의존성 트리를 읽거나, 연관된 CoreNLP 파서를 사용해 원시 텍스트에서 의존성을 파싱합니다. 검색 패턴은 노드(Nodes) 설명과 노드 간의 관계(Relations)로 구성됩니다. 의존성 그래프는 방향성 그래프(Directed graph)로 내부적으로 표현되며, 노드는 단어를, 레이블이 지정된 에지(Edges)는 의존성을 나타냅니다. Semgrex는 다양한 의존성 형식에 사용할 수 있는 관계를 검색하는 데 사용할 수 있으며, CoreNLP와 Stanza는 기본적으로 Universal Dependencies를 사용합니다.

- **Performance Highlights**: Semgrex와 Ssurgeon의 성능 개선 및 Python 인터페이스 추가는 사용자가 효율적으로 의존성 그래프를 검색하고 변환할 수 있게 합니다. 이 도구들은 웹 인터페이스를 통해 원시 텍스트에 패턴을 적용한 결과를 보여주며, 프로그래매틱하게 더 많은 처리를 가능하게 하거나 커맨드 라인 툴을 통해 사용될 수 있습니다. 이러한 기능들은 Semgrex와 Ssurgeon을 텍스트 분석에서 강력한 도구로 만들어, 특히 Universal Dependencies 데이터셋 처리를 간소화합니다.



### URL: Universal Referential Knowledge Linking via Task-instructed  Representation Compression (https://arxiv.org/abs/2404.16248)
- **What's New**: 이 논문에서는 다양한 참조 지식 연결 (referential knowledge linking, RKL) 작업을 하나의 통합 모델로 해결하는 보편적 참조 지식 연결 (universal referential knowledge linking, URL)을 제안합니다. 이는 기존의 정보 검색 (information retrieval)이나 의미 일치 (semantic matching)와 같은 특정 맥락에 국한된 연구를 넘어, 다양한 시나리오에서 참조 지식 연결의 다양한 과제들을 처리할 수 있는 프레임워크를 제공합니다.

- **Technical Details**: URL은 큰 언어 모델 (large language models, LLMs)의 의미 이해 (semantic understanding) 및 지시에 따른 행동 (instruction-following) 능력을 참조 지식 연결 작업에 효과적으로 적용하기 위해 LLM 기반 작업 지시 표현 압축 (LLM-driven task-instructed representation compression) 및 다중 시점 학습 (multi-view learning) 방법을 제안합니다. 또한, 다양한 시나리오를 포괄하는 새로운 벤치마크인 URLBench를 구축하여 모델의 범용 참조 지식 연결 능력을 평가합니다.

- **Performance Highlights**: 실험 결과에 따르면 URL은 기존의 대규모 검색 및 의미 일치 데이터셋에 기반한 모델들을 능가하며, 특히 OpenAI 텍스트 임베딩 모델과 같은 대형 모델 기반 임베딩 방식들보다도 현저하게 우수한 성능을 보여주었습니다. 이는 URL이 범용 참조 지식 연결 과제를 효과적으로 해결할 수 있는 유효한 접근 방식임을 입증합니다.



### Computational analysis of the language of pain: a systematic review (https://arxiv.org/abs/2404.16226)
Comments: 38 pages, 16 tables, 2 figures, systematic review

- **What's New**: 이 연구는 환자 또는 의사가 생성한 통증 언어의 연산 처리에 관한 문헌을 체계적으로 검토하여 현재 추세와 도전 과제를 식별하고자 한다. 특히, 임상 노트에서 발생한 의사 생성 언어가 가장 많이 사용된 데이터였으며, 이는 통증 언어의 특징적 이해와 처리 방법에 새로운 통찰을 제공한다.

- **Technical Details**: 연구는 PRISMA 가이드라인을 따라 포괄적인 문헌 검색을 수행하고, 선택된 연구들을 주요 목적과 결과, 환자 및 통증 인구, 텍스트 데이터, 연산방법론 (computational methodology), 결과 대상에 따라 분류했다. 주로 환자 진단, 트리아징, 통증 언급 식별, 치료 반응 예측, 생물의학적 엔티티 추출, 언어적 특성과 임상 상태의 상관 관계 분석, 통증 내러티브의 어휘-의미론적 분석과 같은 작업이 포함되었다.

- **Performance Highlights**: 연구 결과는 대부분의 연구 결과가 의사를 대상으로 했으며, 임상 도구로 직접 사용되거나 간접적인 지식으로 활용되는 것을 목표로 했다고 밝힌다. 그러나 임상 통증 관리의 자기 관리 단계와 환자 참여가 가장 적었으며, 통증의 정서적 및 사회문화적 차원 또한 가장 적게 연구되었다. 오직 두 연구만이 제안된 알고리즘이 포함될 때 의사의 임상 과제 수행 향상을 측정했다.



### Towards Efficient Patient Recruitment for Clinical Trials: Application  of a Prompt-Based Learning Mod (https://arxiv.org/abs/2404.16198)
- **What's New**: 이 연구에서는 비정형 의료 기록에서 코호트 선택 작업의 성능을 평가하기 위해 프롬프트 기반 대형 언어 모델을 사용하는 새로운 접근 방식을 소개하고 있습니다. 의료 기록(EHR) 사용을 통한 참가자 식별에 자연어 처리(NLP) 기술이 적용되며, 이들 중 트랜스포머(Transformer) 모델을 집중적으로 사용하였습니다.

- **Technical Details**: SNOMED CT 개념을 사용하여 각 자격 기준에 관련된 내용을 수집하고 MedCAT을 활용하여 SNOMED CT 온톨로지를 기반으로 의학적 기록을 주석 처리하였습니다. 선택된 문장은 프롬프트 기반의 GPT(Generative Pre-trained Transformer) 대형 언어 모델을 훈련시키는 데 사용되었습니다. 이 모델의 효과성을 평가하기 위해서 2018 n2c2 챌린지 데이터셋을 사용하였고, 이 데이터셋은 311명의 환자 의료 기록을 13개의 자격 기준에 따라 분류하는 것을 목표로 하였습니다.

- **Performance Highlights**: 제안된 모델은 전반적인 마이크로(Micro) F 측정값과 매크로(Macro) F 측정값이 각각 0.9061과 0.8060으로, 이 데이터셋을 사용한 실험 중 가장 높은 점수를 얻었습니다. 이 모델의 적용은 환자를 자격 기준에 따라 분류하는 작업에서 유망한 점수를 받았으며, SNOMED CT 온톨로지를 활용한 추출 요약 방법도 다른 의료 텍스트에 적용될 수 있습니다.



### Fusion of Domain-Adapted Vision and Language Models for Medical Visual  Question Answering (https://arxiv.org/abs/2404.16192)
Comments: Clinical NLP @ NAACL 2024

- **What's New**: 이번 연구에서는 일반 도메인 및 다양한 멀티모달(Multi-modal) 애플리케이션에서 효과적이지만 특수 분야, 예를 들어 의료 분야에서는 동일한 수준의 효과성을 유지하는 데 어려움을 겪는 시각-언어 모델(Vision-language models)의 한계를 극복하기 위해 의료 분야에 특화된 대형 시각 및 언어 모델을 통합한 의료 시각-언어 모델을 제안합니다.

- **Technical Details**: 제안된 모델은 의료 분야에 맞춤화된 큰 시각 및 언어 모델을 통합하고, 세 가지 별도의 생물의학 및 방사선학 멀티모달 시각과 텍스트 데이터셋을 사용해 파라미터 효율적인(parameter-efficient) 훈련을 세 단계로 진행합니다.

- **Performance Highlights**: 이 모델은 SLAKE 1.0 의료 시각 질문 응답(Medical Visual Question Answering, MedVQA) 데이터셋에서 최고 수준의 성능을 달성하며, 전체 정확도 87.5%를 기록했습니다. 또한 다른 MedVQA 데이터셋인 VQA-RAD에서도 강력한 성능을 보여 전체 정확도 73.2%를 달성했습니다.



### Towards a Holistic Evaluation of LLMs on Factual Knowledge Reca (https://arxiv.org/abs/2404.16164)
- **What's New**: 이 연구에서는 대규모 언어 모델(LLMs)의 사실적 지식 회상 능력과 그에 영향을 미치는 요인들을 평가하는 것에 중점을 두었습니다. 연구팀은 FACT-BENCH라는 새로운 벤치마크를 구축하여 20개의 도메인, 134가지의 속성 유형, 3가지 답변 유형 및 다양한 지식 인기도를 커버하였습니다. 이를 통해 31개의 모델을 10가지 모델 패밀리(model families)에 걸쳐 벤치마크하고, 각 모델의 강점과 약점을 종합적으로 평가했습니다.

- **Technical Details**: 연구에서는 특히 교육 후 조정된 모델(instruction-tuned models)이 순수 학습 모델(pretraining-only models)보다 지식 회상 능력이 떨어지는 것을 발견했습니다. 반면, 모델 스케일링(model scaling)의 긍정적인 효과도 관찰되었는데, 큰 모델이 모든 모델 패밀리에서 작은 모델보다 우수한 성능을 보였습니다. 이와 함께, 거짓 정보(counterfactual information)를 포함하는 예제가 큰 모델의 사실적 지식 회상의 상당한 저하를 초래한다는 것도 분석했습니다. 마지막으로, LLaMA-7B 모델을 다양한 지식 설정에서 미세 조정(fine-tuning)하여, 알려진 지식에 대한 미세 조정이 가장 효과적임을 확인했습니다.

- **Performance Highlights**: GPT-4 모델이 최고의 성능을 보였지만, 여전히 설정된 상한선과의 큰 격차가 있습니다. 또한, 알려진 지식(known knowledge)에 대해 미세 조정된 모델이 알려지지 않은 지식(unknown knowledge)이나 혼합된 지식(mixed knowledge)에 대해 미세 조정된 모델보다 일관되게 더 우수한 성능을 보였습니다.



### Domain-Specific Improvement on Psychotherapy Chatbot Using Assistan (https://arxiv.org/abs/2404.16160)
Comments: Accepted at ICASSP 2024 EIHRC

- **What's New**: 이 연구는 AlexanderStreet 치료법을 기반으로 한 도메인 특화 도우미 지침(Domain-Specific Assistant Instructions)을 처음으로 제안합니다. 또한, 사전 훈련된 대규모 언어 모델(Large Language Models, LLMs)에 적응적 미세 조정(adaption fine-tuning) 방법과 검색 확장 생성(retrieval augmented generation) 방법을 적용하여 심리치료 과제에 대한 성능을 향상시키고자 합니다.

- **Technical Details**: 이 연구는 먼저 도메인 특화 도우미 지침을 제안하고, 사전 훈련된 LLMs에 적응적 미세 조정과 검색 확장 생성 방법을 적용합니다. 이 방법들은 LLM이 심리치료 지식을 더 잘 반영하도록 도와줍니다.

- **Performance Highlights**: 언어의 질을 자동화된 평가와 인간 평가를 통해 정량적으로 평가한 결과, 심리치료 보조 지침을 적용한 사전 훈련된 LLMs는 최신(state-of-the-art) LLMs의 반응 기준보다 우수한 성능을 보였습니다. 이는 반 어노테이션(half-annotation) 방법이 사전 훈련된 LLM을 지침과 연계시킬 수 있다는 것을 보여줍니다.



### From Local to Global: A Graph RAG Approach to Query-Focused  Summarization (https://arxiv.org/abs/2404.16130)
- **What's New**: 이 연구에서는 개인 텍스트 코퍼스에 대한 질문 응답을 위해 Graph RAG(Retrieval-Augmented Generation) 접근 방식을 제안하며, 기존의 RAG 시스템들이 처리하지 못하는 '데이터셋의 주요 주제는 무엇인가?'와 같은 전체 텍스트 코퍼스를 대상으로 한 질문에 대응할 수 있도록 개발되었습니다. 이 방법은 특히 엔티티 지식 그래프(Entity Knowledge Graph)와 커뮤니티 요약(Community Summaries)을 생성하여 사용자 질문의 일반성과 색인해야 할 소스 텍스트의 양에 맞게 확장됩니다.

- **Technical Details**: Graph RAG는 두 단계로 구성된 그래프 기반 텍스트 색인을 사용하여 수행됩니다. 첫 번째 단계에서는 소스 문서로부터 엔티티 지식 그래프를 파생시키고, 두 번째 단계에서는 밀접하게 관련된 엔티티 그룹 전체에 대한 커뮤니티 요약을 미리 생성합니다. 질문이 주어지면 각 커뮤니티 요약은 부분 응답을 생성하는데 사용되며, 이러한 모든 부분 응답은 최종 사용자 응답으로 다시 요약됩니다.

- **Performance Highlights**: 1백만 토큰 범위의 데이터셋에서 전체적인 의미 파악(Global Sensemaking) 질문에 대해, Graph RAG는 기존의 단순한 RAG 기준 대비 종합성과 다양성 측면에서 현저한 향상을 달성했습니다. 이 연구는 개방형 소스(Python 기반)로 구현될 예정이며, 전 세계 및 로컬(Graph RAG approaches) 접근법을 포함합니다.



### Classifying Human-Generated and AI-Generated Election Claims in Social  Media (https://arxiv.org/abs/2404.16116)
- **What's New**: 이 논문은 인공지능(AI)이 만들어낸 콘텐츠와 실제 사용자 콘텐츠를 구분하는 새로운 분류 체계와 데이터 세트를 소개합니다. 선거와 관련된 주장을 분석하기 위한 새로운 분류법(taxonomy)과 함께, 인간 또는 AI가 생성한 트윗을 식별할 수 있는 ElectAI 벤치마크 데이터 세트를 제공합니다.

- **Technical Details**: 이 연구에서 제안된 분류법은 관할 지역, 장비, 프로세스 및 주장의 성격과 같은 세부 카테고리로 선거 관련 주장을 분류합니다. ElectAI 데이터 세트는 9,900개의 트윗으로 구성되어 있으며, 각 트윗은 인간-생성(human-generated) 또는 AI-생성(AI-generated)으로 레이블링되어 있고, AI로 생성된 트윗의 경우 어떤 LLM 변형이 사용되었는지 명시되어 있습니다. 또한, 주장의 특성을 포착하기 위해 1,550개의 트윗을 분류법에 따라 주석을 달았습니다.

- **Performance Highlights**: 연구팀은 LLM의 기능을 평가하여 분류법의 속성을 추출하고, ElectAI를 사용하여 인간과 AI가 생성한 게시물을 구분하고 특정 LLM 변형을 식별하는 다양한 기계 학습 모델(machine learning models)을 훈련시켰습니다. 이를 통해 선거 관련 정보의 진실성을 보호하는 데 중요한 기술적 발전을 이루었습니다.



### Online Personalizing White-box LLMs Generation with Neural Bandits (https://arxiv.org/abs/2404.16115)
Comments: 7 pages

- **What's New**: 이 연구는 개별 사용자의 선호도에 맞춰 텍스트를 효율적으로 조정하는 새로운 온라인 방법을 소개합니다. 이 방법은 사용자 피드백을 기반으로 신경 밴딧(neural bandit) 알고리즘을 사용하여 소프트 지시 임베딩(soft instruction embeddings)을 동적으로 최적화하며, 이를 통해 오픈엔드 텍스트 생성(open-ended text generation)에서의 개인화를 강화합니다.

- **Technical Details**: 연구팀은 다양한 태스크에서 신경 밴딧 알고리즘, 특히 NeuralTS 기술을 사용하여 실험을 수행했습니다. 이를 통해 LLM (large language models)이 사용자의 선호에 따라 텍스트를 개인화하는 방법을 효과적으로 학습하고 최적화할 수 있음을 입증하였습니다.

- **Performance Highlights**: 개인화된 뉴스 헤드라인 생성에서 NeuralTS는 특히 눈에 띄는 성과를 보였습니다. 이 방법은 베이스라인 대비 최대 62.9%의 ROUGE 점수 개선과 LLM 에이전트 평가에서 최대 2.76%의 증가를 달성했습니다.



### Make-it-Real: Unleashing Large Multimodal Model's Ability for Painting  3D Objects with Realistic Materials (https://arxiv.org/abs/2404.16829)
Comments: Project Page: this https URL

- **What's New**: 이 논문에서는 Multimodal Large Language Models (MLLMs), 특히 GPT-4V를 이용하여 3D 오브젝트에 현실적 재료 속성을 자동 할당하는 새로운 접근법 'Make-it-Real'을 제시합니다. 이 방식은 3D 자산의 현실감을 개선하고자 하는 새로운 기술적 진보를 보여줍니다.

- **Technical Details**: GPT-4V를 사용하여 재료를 인식하고 묘사할 수 있으며, 이는 자세한 재료 라이브러리의 구축을 가능하게 합니다. 또한, 시각적 단서와 계층적인 텍스트 프롬프트를 결합하여 3D 객체의 해당 구성 요소에 재료를 정확하게 식별하고 매치합니다. 이렇게 식별된 재료는 오리지널 diffuse map (디퓨즈 맵)에 따라 새로운 SVBRDF (Spatially Varying Bidirectional Reflectance Distribution Function) 재료 생성을 위한 참조로 세심하게 적용됩니다.

- **Performance Highlights**: Make-it-Real은 3D 컨텐츠 생성 워크플로우에 손쉽게 통합되며, 3D 자산 개발자에게 필수 도구로서의 유틸리티를 입증합니다. 복잡하고 시간이 많이 소요되는 수동 재료 할당 작업을 효과적으로 간소화하고, 3D 자산의 시각적 진위성을 크게 향상시킬 수 있습니다.



### Weak-to-Strong Extrapolation Expedites Alignmen (https://arxiv.org/abs/2404.16792)
- **What's New**: 이 논문에서는 ExPO라는 새로운 방법을 제안하여 대규모 언어 모델(Large Language Models, LLMs)의 인간 선호도(human preference)와의 일치성을 향상시키는 방법을 탐구합니다. ExPO를 활용하여 기존에 적은 선호도 데이터로 훈련된 모델을 추가 훈련 없이 더 높은 선호도에 도달하게 하는 방법을 소개합니다.

- **Technical Details**: ExPO는 덜 정렬된(약한) 모델과 더욱 정렬된(강한) 모델 사이에 위치한 중간 정렬 모델을 보간하는 방식을 기본으로 하며, 두 약한 모델의 가중치를 추정하여 강한 모델을 직접 추출하는 방식을 사용합니다. 이 방식은 Supervised Fine-Tuning (SFT) 초기 모델과 RLHF/DPO 모델을 개선하는 데에도 적용됩니다.

- **Performance Highlights**: ExPO는 AlpacaEval 2.0 벤치마크에서 성능을 검증받았으며, 10% 또는 20%의 선호 데이터로 훈련된 모델이 완전 훈련된 모델을 능가하는 결과를 보였습니다. 또한, ExPO는 모델 크기가 7B에서 70B에 이르기까지 확장성(scalability)이 우수하다는 것을 보여주었습니다.



### Continual Learning of Large Language Models: A Comprehensive Survey (https://arxiv.org/abs/2404.16789)
Comments: 57 pages, 2 figures, 4 tables. Work in progress

- **What's New**: 이 연구는 대규모 언어 모델(Large Language Models, LLMs)이 동적 데이터 분포, 작업 구조 및 사용자 선호도에 통합될 때 발생하는 새로운 도전을 다루고 있습니다. 특히, 사전 훈련된 LLMs가 특정 요구에 맞게 조정될 때 이전 지식 도메인에서 심각한 성능 저하를 경험하는 '급격한 망각(catastrophic forgetting)' 현상에 초점을 맞추고 있습니다. 이러한 문제는 연속 학습(Continual Learning, CL) 커뮤니티에서 광범위하게 연구되었으나, LLMs의 영역에서는 새로운 특성을 나타냅니다.

- **Technical Details**: 이 연구는 연속적으로 학습하는 LLMs에 대한 개요를 제공하며, 세로 연속학습(vertical continual learning)과 가로 연속학습(horizontal continual learning)의 두 가지 연속성 방향을 설명합니다. 세로 연속학습은 일반 능력에서 특정 능력으로의 지속적인 적응을 의미하며, 가로 연속학습은 시간과 도메인에 걸쳐 지속적인 적응을 의미합니다. 이후 LLMs의 학습은 연속적 사전훈련(Continual Pre-Training, CPT), 도메인 적응 사전훈련(Domain-Adaptive Pre-training, DAP), 그리고 연속적 미세조정(Continual Fine-Tuning, CFT)의 세 단계로 요약됩니다.

- **Performance Highlights**: 이 서베이는 LLMs와 함께 연속 학습을 평가하는 프로토콜과 현재 사용 가능한 데이터 소스에 대한 개요를 제공합니다. 또한 LLMs를 위한 연속 학습과 관련된 흥미로운 질문을 논의하며, 연구자들과 기술 전문가들이 LLMs의 연속 학습을 더 깊이 이해하고 향후 연구 방향을 설정하는 데 도움을 줄 수 있는 정보를 제공합니다.



### REBEL: Reinforcement Learning via Regressing Relative Rewards (https://arxiv.org/abs/2404.16767)
- **What's New**: 새롭게 제안된 REBEL(Reinforcement BEtween compLetions) 알고리즘은 기존의 강화학습(Reinforcement Learning, RL) 프레임워크, 특히 Proximal Policy Optimization(PPO)의 복잡성을 줄인 방식으로, 정책 최적화 문제를 상대적 보상 회귀로 단순화합니다. REBEL은 자연 정책 경사(Natural Policy Gradient) 같은 기본 RL 알고리즘을 일종의 변형으로 볼 수 있으며, 이를 통해 생성 모델 시대에 맞는 간결한 RL 알고리즘을 제안합니다.

- **Technical Details**: REBEL은 두 가지 프롬프트 완성 간의 직접적인 정책 파라미터화를 통해 정책 최적화 문제를 단순화합니다. 이렇게 함으로써, 복잡하고 계산 집약적인 요소 없이도 RL의 중요한 이론적 보장들, 즉 수렴 및 샘플 복잡성에 대한 가장 강력한 이론적 보증들을 맞출 수 있습니다. 또한 REBEL은 오프라인 데이터를 통합할 수 있고, 실제에서 자주 보이는 비전이성적(Non-transitive) 선호를 처리할 수 있는 능력을 가집니다.

- **Performance Highlights**: 실험을 통해 REBEL은 언어 모델링과 이미지 생성에서 PPO나 DPO(Direct Policy Optimization)와 비교해 강력하거나 유사한 성능을 제공하면서도 더 단순한 구현과 계산적으로 다루기 쉬움을 보여주었습니다. 이는 REBEL이 더 단순한 구조에도 불구하고 이미지나 언어와 같은 다양한 분야에서 유용하게 사용될 수 있는 가능성을 시사합니다.



### Hippocrates: An Open-Source Framework for Advancing Large Language  Models in Healthcar (https://arxiv.org/abs/2404.16621)
- **What's New**: 새롭게 도입된 Hippocrates는 의료 분야에 특화된 오픈소스 대규모 언어 모델(LLM: Large Language Model) 프레임워크입니다. 기존의 모델들과는 달리, 학술적 탐구를 제한하는 독점 모델의 문제를 해결하기 위해 훈련 데이터셋, 코드베이스, 체크포인트, 평가 프로토콜에 대한 제한 없는 접근을 제공함으로써, 의료 AI의 혁신과 재현성을 강화합니다.

- **Technical Details**: Hippocrates는 Mistral 및 LLaMA2에서 파생된, 연속 사전 트레이닝(continual pre-training), 지시 조정(instruction tuning), 그리고 인간 및 AI 피드백으로부터의 강화 학습(reinforcement learning)을 통해 세부 조정된 의료 도메인 전용 7B 모델 패밀리인 Hippo를 도입합니다. 이러한 기술적 접근은 투명한 에코시스템 내에서 의료 LLM의 지속적인 개발, 세분화 및 엄격한 평가를 가능하게 합니다.

- **Performance Highlights**: Hippo 모델은 기존의 오픈 의료 LLM보다 훨씬 높은 성능을 보여주며, 심지어 70B 파라미터를 가진 모델들을 초과하는 성과를 달성하였습니다. 이를 통해 Hippocrates 프레임워크는 의료 지식과 환자 관리의 발전뿐만 아니라, 세계적으로 건강 관리 AI 연구의 혜택을 민주화하는 데 기여하고자 합니다.



### List Items One by One: A New Data Source and Learning Paradigm for  Multimodal LLMs (https://arxiv.org/abs/2404.16375)
Comments: Preprint

- **What's New**: 새로운 'Set-of-Mark (SoM) Prompting' 방법이 GPT-4V 모델의 시각적 태깅 성능을 향상시켰습니다. 다른 Multimodal Large Language Models (다중 모달 대규모 언어 모델, MLLMs)에서는 이러한 시각적 태그를 이해하는 데 어려움을 겪었기 때문에, '하나씩 항목을 나열하라'는 새로운 학습 패러다임을 제안했습니다. 이는 모델이 이미지에 배치된 시각적 태그를 알파뉴메릭 순서에 따라 열거하고 설명하도록 요구합니다. 이 방법은 기존 MLLMs에게 SoM 프롬프팅 능력을 갖추게 함으로써, 시각적 추론 능력을 향상시키고 착각을 감소시키는 결과를 가져왔습니다.

- **Technical Details**: '하나씩 항목을 나열하라' 학습 방법을 통해, 10k-30k 이미지 및 태그가 포함된 새로운 데이터 세트를 다른 시각적 지시 튜닝 데이터 세트와 통합하여 MLLMs에 적용했습니다. 이러한 데이터 세트 구성이 MLLMs의 시각적 이해 능력을 크게 개선하며, 추론 시 시각적 태그가 입력 이미지에서 생략되어도 성능 개선이 지속됩니다.

- **Performance Highlights**: 새로운 데이터 세트를 통합한 후, MLLMs는 다섯 가지 벤치마크에서 뛰어난 성과를 보였습니다. 특히 시각적 추론 능력의 향상과 착각 감소가 두드러졌으며, 이는 시각적 태그를 통한 객체-텍스트 정렬을 강화하는 새로운 트레이닝 방식의 가능성을 보여줍니다.



### Investigating the prompt leakage effect and black-box defenses for  multi-turn LLM interactions (https://arxiv.org/abs/2404.16251)
- **What's New**: 이 논문은 RAG(Retrieval-Augmented Generation) 시스템에서의 프롬프트(Prompt) 유출과 다중 턴(Mult-turn) 상호작용에서의 보안 위협을 분석합니다. 특히, LLM(Large Language Models)이 어떻게 다양한 도메인의 지식을 유출할 수 있는지와 이를 방지하기 위한 다중 방어 전략이 제안되었습니다.

- **Technical Details**: LLM의 평균 공격 성공률(ASR, Attack Success Rate)을 86.2%로 상승시키는 새로운 다중 턴 위협 모델을 사용합니다. 특히 GPT-4와 claude-1.3에서는 99%의 높은 유출률을 보였으며, RAG 시나리오에서의 쿼리 재작성(query-rewriter)과 같은 6가지 검은 상자 방어 전략(black-box defense strategies)의 효과가 측정되었습니다.

- **Performance Highlights**: 검은 상자 모델에서는 뉴스 도메인(news domain)에서 의료 도메인(medical domain)에 비해 상대적으로 더 많은 문맥 지식을 유출할 가능성이 있음을 발견했습니다. 제안된 다중 계층 방어(multi-tier defense)는 ASR를 5.3%까지 낮추는 효과를 보였으나, 여전히 보안 개선의 여지가 있음을 시사합니다.



### Knowledge Graph Completion using Structural and Textual Embeddings (https://arxiv.org/abs/2404.16206)
- **What's New**: 이 연구에서는 텍스트 및 구조적 정보를 활용하여 지식 그래프(Knowledge Graphs, KGs) 내에서 관계를 예측하는 새로운 모델을 제안하였습니다. 이 모델은 기존 노드 간의 관계를 탐색하여 KGs의 불완전성을 해결하는 데 중점을 둡니다.

- **Technical Details**: 제안된 모델은 워크 기반 임베딩(walks-based embeddings)과 언어 모델 임베딩(language model embeddings)을 통합하여 노드를 효과적으로 표현합니다. 이를 통해 KGs에서 텍스트와 구조의 정보를 모두 활용할 수 있으며, 관계 예측(relation prediction) 작업에 있어서 높은 수준의 성능을 보였습니다.

- **Performance Highlights**: 이 모델은 널리 사용되는 데이터셋에서 평가되었을 때, 관계 예측 작업에 대해 경쟁력 있는 결과를 달성하였습니다. 이는 텍스트와 구조적 정보의 통합 접근이 KG의 불완전성을 보완하는 데 유효함을 시사합니다.



### FairDeDup: Detecting and Mitigating Vision-Language Fairness Disparities  in Semantic Dataset Deduplication (https://arxiv.org/abs/2404.16123)
Comments: Conference paper at CVPR 2024. 6 pages, 8 figures. Project Page: this https URL

- **What's New**: 최근 데이터셋 중복 제거 기술(FairDeDup)이 VLP(Vision-Language Pretrained) 모델을 훈련하는 비용을 크게 줄이면서도 원본 데이터셋에서의 훈련 성능과 비교해 중요한 손실이 없음을 보여줬다. 특히 웹에서 수집된 이미지-캡션 데이터셋을 대상으로 하며, 이러한 데이터셋은 종종 해로운 사회적 편향을 포함하고 있는 것으로 알려졌다.

- **Technical Details**: 이 연구에서는 데이터셋의 중복 제거가 결과적으로 훈련된 모델에서의 편향된 성능에 어떤 영향을 미치는지 평가한다. 최근 제안된 SemDeDup 알고리즘에 쉽게 구현할 수 있는 수정을 도입하여 관찰된 부정적인 효과를 줄이는 방법을 제안한다. 또한, LAION-400M 데이터셋의 중복된 버전에서 CLIP 스타일의 모델을 훈련했을 때, FairFace 및 FACET 데이터셋에서 SemDeDup보다 공정성 지표(fairness metrics)가 일관되게 개선되었다고 보고한다.

- **Performance Highlights**: 제안된 FairDeDup 알고리즘은 CLIP 벤치마크에서의 제로샷(zero-shot) 성능을 유지하면서도 FairFace 및 FACET 데이터셋에 대해 SemDeDup보다 향상된 공정성 지표를 달성한다.



### Evolution of Voices in French Audiovisual Media Across Genders and Age  in a Diachronic Perspectiv (https://arxiv.org/abs/2404.16104)
Comments: 5 pages, 2 figures, keywords:, Gender, Diachrony, Vocal Tract Resonance, Vocal register, Broadcast speech

- **What's New**: 이 연구는 프랑스 미디어 아카이브에서 1023명의 연사자의 목소리에 대한 시간에 따른 음향 분석을 제시합니다. 연구 대상은 네 개의 시대 (1955/56, 1975/76, 1995/96, 2015/16), 네 개의 연령대 (20-35세; 36-50세; 51-65세, 65세 이상), 그리고 두 개의 성별에 걸쳐 32개의 범주로 분류되었습니다.

- **Technical Details**: 기본 주파수 ($F_0$)와 처음 네 개의 포만트(F1-4)가 추정되었습니다. 데이터의 질을 보장하기 위한 절차가 설명되어 있습니다. 각 연사자의 $F_0$ 분포에서 기본 $F_0$(base-$F_0$) 값이 계산되었고, 이는 등록 추정에 사용되었습니다. 성대 길이(vocal tract length)는 포만트 주파수에서 추정되었습니다. 이러한 기본 $F_0$와 성대 길이는 시대별, 성별 변화를 평가하기 위해 연령 효과를 보정하여 선형 혼합 모델(linear mixed models)로 적합되었습니다.

- **Performance Highlights**: 결과는 시대 효과와 성별에 관계없이 낮아진 목소리 경향을 보여줍니다. 나이가 들수록 여성의 음성 피치(pitch)가 낮아지는 경향이 관찰되었지만, 남성에서는 그러한 경향이 관찰되지 않았습니다.



### Semantic Evolvement Enhanced Graph Autoencoder for Rumor Detection (https://arxiv.org/abs/2404.16076)
- **Summary**: [{"What's New": '소셜 미디어(Social Media)에서 소문의 빠른 확산으로 인해 소문 감지(Rumor Detection)는 매우 중요한 도전 과제가 되었습니다. 이를 해결하기 위해, 우리는 이벤트의 의미론적 전개(Semantic Evolvement) 정보를 학습하여 소문을 탐지하는 새로운 그래프 자동인코더(Graph Autoencoder) 모델인 GARD를 제안합니다.'}, {'Technical Details': 'GARD 모델은 이벤트의 로컬 의미론적 변화와 글로벌 의미론적 전개 정보를 특정 그래프 자동인코더와 재구성 전략을 통해 파악함으로써 의미론적 전개 정보를 학습합니다. 또한, 유니포미티 규제자(Uniformity Regularizer)를 도입하여 소문과 비소문의 뚜렷한 패턴을 학습하는 능력을 강화했습니다.'}, {'Performance Highlights': '세 개의 공개 벤치마크 데이터셋(Benchmark Dataset)에서 수행한 실험 결과, GARD 방법은 전반적인 성능과 초기 소문 감지(Early Rumor Detection) 모두에서 최신 기법(State-of-the-art Approaches)보다 우월함을 확인하였습니다.'}]



### SemEval-2024 Task 9: BRAINTEASER: A Novel Task Defying Common Sens (https://arxiv.org/abs/2404.16068)
- **What's New**: 이 논문에서는 BRAINTEASER 벤치마크를 수정하여 fine-tuning (세밀 조정) 설정도 지원하도록 하고, 시스템의 추론 및 lateral thinking (측면 사고) 능력을 테스트하는 첫 번째 경쟁 과제인 SemEval Task 9: BRAIN-TEASER(S)을 소개합니다. BRAINTEASER(S)는 182명의 참가자로부터 483개 팀의 참가 제출을 받았습니다.

- **Technical Details**: BRAINTEASER 벤치마크는 원래 zero-shot (제로 샷) 설정을 목표로 하였으나 이제 fine-tuning을 포함하여 더 넓은 활용성을 제공합니다. 세부적으로, BRAINTEASER(S)는 참가자들이 시스템의 측면 사고와 로봇 추론 능력을 평가할 수 있도록 설계된 두 개의 subtask (하위 과제)를 포함합니다.

- **Performance Highlights**: 이 논문은 BRAINTEASER(S) 경쟁 결과에 대한 세밀한 시스템 분석을 제공하며, 이 결과는 시스템이 측면 사고를 어떻게 수행하는지에 대한 귀중한 통찰을 제공합니다. 결과 분석은 computational models (계산 모델)의 측면 사고 및 robust reasoning (강력한 추론) 능력에 대한 미래 연구를 자극할 것입니다.



### Human Latency Conversational Turns for Spoken Avatar Systems (https://arxiv.org/abs/2404.16053)
- **What's New**: 이 논문은 기존의 대규모 언어 모델(Large Language Model, LLM)을 사용한 음성 대화 시스템에서의 응답 시간 문제를 개선하는 방법을 제안합니다. 사람 대 사람 대화에서는 종종 말하는 사람의 발화가 끝나기 전에 응답이 이루어지는데, 이 연구는 LLM이 인간 수준의 대화 지연 시간을 준수하면서 거의 실시간에 가깝게 발화를 이해하고 응답을 생성할 수 있는 방법을 탐색합니다.

- **Technical Details**: 이 방법은 발화의 마지막 부분의 정보 내용이 LLM에게 손실된다는 점을 기반으로 합니다. 구글의 NaturalQuestions (NQ) 데이터베이스를 사용하여, 연구팀은 GPT-4가 질문의 끝에 빠진 단어의 문맥을 60% 이상의 시간에 효과적으로 채울 수 있음을 보여줍니다. 또한, 연구 중인 아바타 컨텍스트에서 이 정보 손실이 LLM 응답의 질에 미치는 영향에 대한 예를 제공합니다.

- **Performance Highlights**: 실험 결과는 GPT-4가 빠진 단어를 보완하여 응답 생성하는 데 60% 이상의 성공률을 보였습니다. 이는 LLM의 처리 지연을 인간 대화의 지연 시간에 맞출 수 있는 가능성을 시사합니다. 또한, 간단한 분류기(Classifier)를 사용하여 질문이 의미론적으로 완전한지, 아니면 인간 대화 시간 제약 내에서 응답을 생성할 수 있게 하는 채움 구(Filler Phrase)가 필요한지를 결정할 수 있습니다.



